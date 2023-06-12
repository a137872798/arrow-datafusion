// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! This module provides a builder for creating LogicalPlans

use crate::expr_rewriter::{
    coerce_plan_expr_for_schema, normalize_col,
    normalize_col_with_schemas_and_ambiguity_check, normalize_cols,
    rewrite_sort_cols_by_aggs,
};
use crate::type_coercion::binary::comparison_coercion;
use crate::utils::{columnize_expr, compare_sort_expr, exprlist_to_fields, from_plan};
use crate::{and, binary_expr, DmlStatement, Operator, WriteOp};
use crate::{
    logical_plan::{
        Aggregate, Analyze, CrossJoin, Distinct, EmptyRelation, Explain, Filter, Join,
        JoinConstraint, JoinType, Limit, LogicalPlan, Partitioning, PlanType, Prepare,
        Projection, Repartition, Sort, SubqueryAlias, TableScan, ToStringifiedPlan,
        Union, Unnest, Values, Window,
    },
    utils::{
        can_hash, expand_qualified_wildcard, expand_wildcard,
        find_valid_equijoin_key_pair, group_window_expr_by_sort_keys,
    },
    Expr, ExprSchemable, TableSource,
};
use arrow::datatypes::{DataType, Schema, SchemaRef};
use datafusion_common::{
    Column, DFField, DFSchema, DFSchemaRef, DataFusionError, OwnedTableReference, Result,
    ScalarValue, TableReference, ToDFSchema,
};
use std::any::Any;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::convert::TryFrom;
use std::sync::Arc;

/// Default table name for unnamed table
pub const UNNAMED_TABLE: &str = "?table?";

/// Builder for logical plans
///
/// ```
/// # use datafusion_expr::{lit, col, LogicalPlanBuilder, logical_plan::table_scan};
/// # use datafusion_common::Result;
/// # use arrow::datatypes::{Schema, DataType, Field};
/// #
/// # fn main() -> Result<()> {
/// #
/// # fn employee_schema() -> Schema {
/// #    Schema::new(vec![
/// #           Field::new("id", DataType::Int32, false),
/// #           Field::new("first_name", DataType::Utf8, false),
/// #           Field::new("last_name", DataType::Utf8, false),
/// #           Field::new("state", DataType::Utf8, false),
/// #           Field::new("salary", DataType::Int32, false),
/// #       ])
/// #   }
/// #
/// // Create a plan similar to
/// // SELECT last_name
/// // FROM employees
/// // WHERE salary < 1000
/// let plan = table_scan(
///              Some("employee"),
///              &employee_schema(),
///              None,
///            )?
///            // Keep only rows where salary < 1000
///            .filter(col("salary").lt_eq(lit(1000)))?
///            // only show "last_name" in the final results
///            .project(vec![col("last_name")])?
///            .build()?;
///
/// # Ok(())
/// # }
/// ```
/// 通过该对象构建逻辑计划
#[derive(Debug, Clone)]
pub struct LogicalPlanBuilder {
    plan: LogicalPlan,
}

/// 逻辑计划生成器
impl LogicalPlanBuilder {
    /// Create a builder from an existing plan
    pub fn from(plan: LogicalPlan) -> Self {
        Self { plan }
    }

    /// Return the output schema of the plan build so far
    /// 返回plan关联的schema
    pub fn schema(&self) -> &DFSchemaRef {
        self.plan.schema()
    }

    /// Create an empty relation.
    ///
    /// `produce_one_row` set to true means this empty node needs to produce a placeholder row.
    /// produce_one_row 是否生成一个占位行
    /// empty 生成一个空的执行计划
    pub fn empty(produce_one_row: bool) -> Self {
        Self::from(LogicalPlan::EmptyRelation(EmptyRelation {
            produce_one_row,
            schema: DFSchemaRef::new(DFSchema::empty()),
        }))
    }

    /// Create a values list based relation, and the schema is inferred from data, consuming
    /// `value`. See the [Postgres VALUES](https://www.postgresql.org/docs/current/queries-values.html)
    /// documentation for more details.
    ///
    /// By default, it assigns the names column1, column2, etc. to the columns of a VALUES table.
    /// The column names are not specified by the SQL standard and different database systems do it differently,
    /// so it's usually better to override the default names with a table alias list.
    ///
    /// If the values include params/binders such as $1, $2, $3, etc, then the `param_data_types` should be provided.
    /// 该逻辑计划直接对应了一组value
    pub fn values(mut values: Vec<Vec<Expr>>) -> Result<Self> {
        if values.is_empty() {
            return Err(DataFusionError::Plan("Values list cannot be empty".into()));
        }

        // 代表每行有多少列
        let n_cols = values[0].len();

        if n_cols == 0 {
            return Err(DataFusionError::Plan(
                "Values list cannot be zero length".into(),
            ));
        }
        let empty_schema = DFSchema::empty();

        // 对应每一列数据的类型
        let mut field_types: Vec<Option<DataType>> = Vec::with_capacity(n_cols);
        for _ in 0..n_cols {
            field_types.push(None);
        }
        // hold all the null holes so that we can correct their data types later
        // <row号，col号> 记录列值为null的变量坐标
        let mut nulls: Vec<(usize, usize)> = Vec::new();
        // 行号和行数据
        for (i, row) in values.iter().enumerate() {
            // 每行的列数都要一致
            if row.len() != n_cols {
                return Err(DataFusionError::Plan(format!(
                    "Inconsistent data length across values list: got {} values in row {} but expected {}",
                    row.len(),
                    i,
                    n_cols
                )));
            }

            field_types = row
                .iter()
                .enumerate()
                // col号和expr
                .map(|(j, expr)| {
                    // 传入的是个null值
                    if let Expr::Literal(ScalarValue::Null) = expr {
                        nulls.push((i, j));
                        // 之前还没被赋值的情况 Null自然对应None 而如果后面出现了值 并修改了类型后，会更新field_type
                        Ok(field_types[j].clone())
                    } else {
                        // 获取value的类型
                        let data_type = expr.get_type(&empty_schema)?;
                        // 代表同一列的数据前后不一致
                        if let Some(prev_data_type) = &field_types[j] {
                            if prev_data_type != &data_type {
                                let err = format!("Inconsistent data type across values list at row {i} column {j}");
                                return Err(DataFusionError::Plan(err));
                            }
                        }
                        Ok(Some(data_type))
                    }
                })
                .collect::<Result<Vec<Option<DataType>>>>()?;
        }

        // 此时已经得到所有field的类型了
        // 这里生成了一组虚拟列名  column[]
        let fields = field_types
            .iter()
            .enumerate()
            .map(|(j, data_type)| {
                // naming is following convention https://www.postgresql.org/docs/current/queries-values.html
                let name = &format!("column{}", j + 1);

                // 这时使用的列名是类似占位符的东西 column1,column2,...
                DFField::new_unqualified(
                    name,
                    data_type.clone().unwrap_or(DataType::Utf8),
                    true,
                )
            })
            .collect::<Vec<_>>();

        // 把之前的null值变成类型相关的None
        for (i, j) in nulls {
            values[i][j] = Expr::Literal(ScalarValue::try_from(fields[j].data_type())?);
        }

        // 将基于虚拟列名生成的schema与 values合成 Values
        let schema =
            DFSchemaRef::new(DFSchema::new_with_metadata(fields, HashMap::new())?);
        Ok(Self::from(LogicalPlan::Values(Values { schema, values })))
    }

    /// Convert a table provider into a builder with a TableScan
    ///
    /// Note that if you pass a string as `table_name`, it is treated
    /// as a SQL identifier, as described on [`TableReference`] and
    /// thus is normalized
    ///
    /// # Example:
    /// ```
    /// # use datafusion_expr::{lit, col, LogicalPlanBuilder,
    /// #  logical_plan::builder::LogicalTableSource, logical_plan::table_scan
    /// # };
    /// # use std::sync::Arc;
    /// # use arrow::datatypes::{Schema, DataType, Field};
    /// # use datafusion_common::TableReference;
    /// #
    /// # let employee_schema = Arc::new(Schema::new(vec![
    /// #           Field::new("id", DataType::Int32, false),
    /// # ])) as _;
    /// # let table_source = Arc::new(LogicalTableSource::new(employee_schema));
    /// // Scan table_source with the name "mytable" (after normalization)
    /// # let table = table_source.clone();
    /// let scan = LogicalPlanBuilder::scan("MyTable", table, None);
    ///
    /// // Scan table_source with the name "MyTable" by enclosing in quotes
    /// # let table = table_source.clone();
    /// let scan = LogicalPlanBuilder::scan(r#""MyTable""#, table, None);
    ///
    /// // Scan table_source with the name "MyTable" by forming the table reference
    /// # let table = table_source.clone();
    /// let table_reference = TableReference::bare("MyTable");
    /// let scan = LogicalPlanBuilder::scan(table_reference, table, None);
    /// ```
    /// 代表生成一个对目标表的查询计划
    pub fn scan(
        table_name: impl Into<OwnedTableReference>,  // 查询的目标表
        table_source: Arc<dyn TableSource>,
        projection: Option<Vec<usize>>,  // 描述查询结果需要保留哪些列
    ) -> Result<Self> {
        Self::scan_with_filters(table_name, table_source, projection, vec![])
    }

    /// Create a [DmlStatement] for inserting the contents of this builder into the named table
    /// 产生一个DML操作
    pub fn insert_into(
        input: LogicalPlan,  // plan包含了多个expr 执行plan就代表完成insert操作 只要指定目标表就行了
        table_name: impl Into<OwnedTableReference>,
        table_schema: &Schema,
    ) -> Result<Self> {
        let table_schema = table_schema.clone().to_dfschema_ref()?;

        // 简单的组装而已
        Ok(Self::from(LogicalPlan::Dml(DmlStatement {
            table_name: table_name.into(),
            table_schema,
            op: WriteOp::Insert,
            input: Arc::new(input),
        })))
    }

    /// Convert a table provider into a builder with a TableScan
    pub fn scan_with_filters(
        table_name: impl Into<OwnedTableReference>,
        table_source: Arc<dyn TableSource>,
        projection: Option<Vec<usize>>,  // 对应 Select xxx的部分
        filters: Vec<Expr>, // 对应 Where 的部分
    ) -> Result<Self> {
        let table_name = table_name.into();

        if table_name.table().is_empty() {
            return Err(DataFusionError::Plan(
                "table_name cannot be empty".to_string(),
            ));
        }

        // 获取表的元数据信息
        let schema = table_source.schema();

        // 针对需要保留的列 转换成schema
        let projected_schema = projection
            .as_ref()
            .map(|p| {
                // 加工schema 只需要保留这些列
                DFSchema::new_with_metadata(
                    p.iter()
                        .map(|i| {
                            DFField::from_qualified(
                                table_name.clone(),
                                schema.field(*i).clone(),
                            )
                        })
                        .collect(),
                    schema.metadata().clone(),
                )
            })
            .unwrap_or_else(|| {
                DFSchema::try_from_qualified_schema(table_name.clone(), &schema)
            })?;

        let table_scan = LogicalPlan::TableScan(TableScan {
            table_name,
            source: table_source,
            projected_schema: Arc::new(projected_schema),  // 本次需要的列的元数据
            projection,  // 需要保留的列的下标
            filters,  // 过滤数据集的过滤器
            fetch: None,  // limit
        });
        Ok(Self::from(table_scan))
    }

    /// Wrap a plan in a window  在原plan上追加一层窗口函数
    pub fn window_plan(
        input: LogicalPlan,
        window_exprs: Vec<Expr>,  // 作用上去的窗口表达式
    ) -> Result<LogicalPlan> {
        let mut plan = input;
        // 不同的窗口函数 分区键和排序键都可能会不同 把这些函数按照相同的分区键/排序键进行分组
        let mut groups = group_window_expr_by_sort_keys(&window_exprs)?;
        // To align with the behavior of PostgreSQL, we want the sort_keys sorted as same rule as PostgreSQL that first
        // we compare the sort key themselves and if one window's sort keys are a prefix of another
        // put the window with more sort keys first. so more deeply sorted plans gets nested further down as children.
        // The sort_by() implementation here is a stable sort.
        // Note that by this rule if there's an empty over, it'll be at the top level
        // 已经分完组后 还需要对分组结果进行排序  这样作用到结果上的窗口函数就有了先后顺序
        groups.sort_by(|(key_a, _), (key_b, _)| {
            for ((first, _), (second, _)) in key_a.iter().zip(key_b.iter()) {
                // 简单理解为比较col的下标
                let key_ordering = compare_sort_expr(first, second, plan.schema());
                match key_ordering {
                    Ordering::Less => {
                        return Ordering::Less;
                    }
                    Ordering::Greater => {
                        return Ordering::Greater;
                    }
                    Ordering::Equal => {}
                }
            }
            key_b.len().cmp(&key_a.len())
        });

        // 将使用相同分区键的窗口函数 包装成一个window逻辑计划 并包装在input上
        for (_, exprs) in groups {
            let window_exprs = exprs.into_iter().cloned().collect::<Vec<_>>();
            // Partition and sorting is done at physical level, see the EnforceDistribution
            // and EnforceSorting rules.
            // 按照不同的分区键分组的窗口函数 分别包装到内部plan上
            plan = LogicalPlanBuilder::from(plan)
                .window(window_exprs)?
                .build()?;
        }
        Ok(plan)
    }
    /// Apply a projection without alias.
    /// 在原有计划上追加一层投影
    pub fn project(
        self,
        expr: impl IntoIterator<Item = impl Into<Expr>>,
    ) -> Result<Self> {
        Ok(Self::from(project(self.plan, expr)?))
    }

    /// Select the given column indices
    /// 与上面不同的是 传入的是列的下标
    pub fn select(self, indices: impl IntoIterator<Item = usize>) -> Result<Self> {
        let fields = self.plan.schema().fields();
        let exprs: Vec<_> = indices
            .into_iter()
            .map(|x| Expr::Column(fields[x].qualified_column()))
            .collect();
        self.project(exprs)
    }

    /// Apply a filter
    /// 生成过滤表达式
    pub fn filter(self, expr: impl Into<Expr>) -> Result<Self> {
        // 就是补全过滤表达式中列的表信息  因为表信息来自于plan
        let expr = normalize_col(expr.into(), &self.plan)?;
        Ok(Self::from(LogicalPlan::Filter(Filter::try_new(
            expr,
            Arc::new(self.plan),
        )?)))
    }

    /// Make a builder for a prepare logical plan from the builder's plan
    /// 提前处理sql 之后只需要参数就可以执行sql 相比每次都要传入不同的sql 减少了重复优化的开销
    pub fn prepare(self, name: String, data_types: Vec<DataType>) -> Result<Self> {
        Ok(Self::from(LogicalPlan::Prepare(Prepare {
            name,
            data_types,
            input: Arc::new(self.plan),
        })))
    }

    /// Limit the number of rows returned
    ///
    /// `skip` - Number of rows to skip before fetch any row.
    ///
    /// `fetch` - Maximum number of rows to fetch, after skipping `skip` rows,
    ///          if specified.
    /// 追加一层limit
    pub fn limit(self, skip: usize, fetch: Option<usize>) -> Result<Self> {
        Ok(Self::from(LogicalPlan::Limit(Limit {
            skip,
            fetch,
            input: Arc::new(self.plan),
        })))
    }

    /// Apply an alias
    pub fn alias(self, alias: impl Into<OwnedTableReference>) -> Result<Self> {
        Ok(Self::from(subquery_alias(self.plan, alias)?))
    }

    /// Add missing sort columns to all downstream projection
    ///
    /// Thus, if you have a LogialPlan that selects A and B and have
    /// not requested a sort by C, this code will add C recursively to
    /// all input projections.
    ///
    /// Adding a new column is not correct if there is a `Distinct`
    /// node, which produces only distinct values of its
    /// inputs. Adding a new column to its input will result in
    /// potententially different results than with the original column.
    ///
    /// For example, if the input is like:
    ///
    /// Distinct(A, B)
    ///
    /// If the input looks like
    ///
    /// a | b | c
    /// --+---+---
    /// 1 | 2 | 3
    /// 1 | 2 | 4
    ///
    /// Distinct (A, B) --> (1,2)
    ///
    /// But Distinct (A, B, C) --> (1, 2, 3), (1, 2, 4)
    ///  (which will appear as a (1, 2), (1, 2) if a and b are projected
    ///
    /// See <https://github.com/apache/arrow-datafusion/issues/5065> for more details
    /// 当排序需要的某些列 没有出现在plan的schema中时  需要补上这些列
    fn add_missing_columns(
        curr_plan: LogicalPlan,
        missing_cols: &[Column],
        is_distinct: bool,  // 代表外层有一个Distinct计划
    ) -> Result<LogicalPlan> {
        match curr_plan {

            // 输入的计划上已经包含了一层投影
            LogicalPlan::Projection(Projection {
                input,
                mut expr,
                schema: _,
                // 这些缺失的列能够在投影前被找到  那么就要将本次的投影后置
            }) if missing_cols.iter().all(|c| input.schema().has_column(c)) => {

                // 从input.schema上找到field关联的table
                let mut missing_exprs = missing_cols
                    .iter()
                    .map(|c| normalize_col(Expr::Column(c.clone()), &input))
                    .collect::<Result<Vec<_>>>()?;

                // Do not let duplicate columns to be added, some of the
                // missing_cols may be already present but without the new
                // projected alias.
                missing_exprs.retain(|e| !expr.contains(e));

                // 首次调用是false  在往内递归的过程中可能会遇到Distinct类型的计划
                if is_distinct {
                    Self::ambiguous_distinct_check(&missing_exprs, missing_cols, &expr)?;
                }
                // 在内层投影上额外保留这些字段  这样外层的sort就可以找到这些列了
                expr.extend(missing_exprs);
                // 更新投影信息
                Ok(project((*input).clone(), expr)?)
            }

            // 其他情况
            _ => {
                // 这里的前提相当于是认定sort需要的列 一定在前面出现过了 要往前回溯找到对应的投影表达式 并改写投影计划  确保排序列可以保留到后面
                let is_distinct =
                    is_distinct || matches!(curr_plan, LogicalPlan::Distinct(_));
                let new_inputs = curr_plan
                    .inputs()
                    .into_iter()
                    .map(|input_plan| {
                        Self::add_missing_columns(
                            (*input_plan).clone(),
                            missing_cols,
                            is_distinct,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;

                // 替换内部的plan
                let expr = curr_plan.expressions();
                from_plan(&curr_plan, &expr, &new_inputs)
            }
        }
    }

    // TODO distinct检查
    fn ambiguous_distinct_check(
        missing_exprs: &[Expr],  // 缺失的列对应的expr
        missing_cols: &[Column],  // 缺失的列
        projection_exprs: &[Expr],  // 本次投影后保留的expr
    ) -> Result<()> {
        if missing_exprs.is_empty() {
            return Ok(());
        }

        // if the missing columns are all only aliases for things in
        // the existing select list, it is ok
        //
        // This handles the special case for
        // SELECT col as <alias> ORDER BY <alias>
        //
        // As described in https://github.com/apache/arrow-datafusion/issues/5293
        let all_aliases = missing_exprs.iter().all(|e| {
            projection_exprs.iter().any(|proj_expr| {
                if let Expr::Alias(expr, _) = proj_expr {
                    e == expr.as_ref()
                } else {
                    false
                }
            })
        });
        if all_aliases {
            return Ok(());
        }

        let missing_col_names = missing_cols
            .iter()
            .map(|col| col.flat_name())
            .collect::<String>();

        Err(DataFusionError::Plan(format!(
            "For SELECT DISTINCT, ORDER BY expressions {missing_col_names} must appear in select list",
        )))
    }

    /// Apply a sort
    /// 将一组(因为结果可以按照多个列 不同的排序方式进行排序)排序表达式作用在查询结果上
    pub fn sort(
        self,
        exprs: impl IntoIterator<Item = impl Into<Expr>> + Clone,
    ) -> Result<Self> {
        // TODO 看过去就是做了cast或者修改col的所属表信息  不影响排序逻辑
        let exprs = rewrite_sort_cols_by_aggs(exprs, &self.plan)?;

        let schema = self.plan.schema();

        // Collect sort columns that are missing in the input plan's schema
        // 排序用到的col可能不包含在schema中
        let mut missing_cols: Vec<Column> = vec![];
        exprs
            .clone()
            .into_iter()
            .try_for_each::<_, Result<()>>(|expr| {
                // 简单理解expr是col
                let columns = expr.to_columns()?;

                // 这里是排序用到的所有列
                columns.into_iter().for_each(|c| {
                    // 排序涉及到的这些列 并没有在声明的schema中 意味着无法获取这些col的信息
                    if schema.field_from_column(&c).is_err() {
                        missing_cols.push(c);
                    }
                });

                Ok(())
            })?;

        // 所有排序键能够在schema中找到  不需要其他处理
        if missing_cols.is_empty() {
            return Ok(Self::from(LogicalPlan::Sort(Sort {
                expr: normalize_cols(exprs, &self.plan)?,
                input: Arc::new(self.plan),
                fetch: None,
            })));
        }

        // remove pushed down sort columns
        // 为了是排序有效  可能会在plan.schema上添加很多一开始没用的col
        // 然后为了保证查询结果与预期的一致 要额外用一层投影处理
        let new_expr = schema
            .fields()
            .iter()
            .map(|f| Expr::Column(f.qualified_column()))
            .collect();

        let is_distinct = false;
        // 现在需要将缺失的列补充到内部plan上
        let plan = Self::add_missing_columns(self.plan, &missing_cols, is_distinct)?;
        let sort_plan = LogicalPlan::Sort(Sort {
            expr: normalize_cols(exprs, &plan)?,
            input: Arc::new(plan),
            fetch: None,
        });

        // 隐藏掉添加的col
        Ok(Self::from(LogicalPlan::Projection(Projection::try_new(
            new_expr,
            Arc::new(sort_plan),
        )?)))
    }

    /// Apply a union, preserving duplicate rows
    pub fn union(self, plan: LogicalPlan) -> Result<Self> {
        Ok(Self::from(union(self.plan, plan)?))
    }

    /// Apply a union, removing duplicate rows
    /// union的同时去掉重复的列
    pub fn union_distinct(self, plan: LogicalPlan) -> Result<Self> {
        // unwrap top-level Distincts, to avoid duplication
        // 即使在union前已经去重了   但是union后实际上可能又会出现一样的数据  所以修改成union后distinct
        let left_plan: LogicalPlan = match self.plan {
            LogicalPlan::Distinct(Distinct { input }) => (*input).clone(),
            _ => self.plan,
        };
        let right_plan: LogicalPlan = match plan {
            LogicalPlan::Distinct(Distinct { input }) => (*input).clone(),
            _ => plan,
        };

        // union后去重
        Ok(Self::from(LogicalPlan::Distinct(Distinct {
            input: Arc::new(union(left_plan, right_plan)?),
        })))
    }

    /// Apply deduplication: Only distinct (different) values are returned)
    pub fn distinct(self) -> Result<Self> {
        Ok(Self::from(LogicalPlan::Distinct(Distinct {
            input: Arc::new(self.plan),
        })))
    }

    /// Apply a join with on constraint.
    ///
    /// Filter expression expected to contain non-equality predicates that can not be pushed
    /// down to any of join inputs.
    /// In case of outer join, filter applied to only matched rows.
    pub fn join(
        self,
        right: LogicalPlan,
        join_type: JoinType,
        join_keys: (Vec<impl Into<Column>>, Vec<impl Into<Column>>),
        filter: Option<Expr>,
    ) -> Result<Self> {
        self.join_detailed(right, join_type, join_keys, filter, false)
    }

    // 对column表达式进行标准化处理  返回权限定名
    pub(crate) fn normalize(
        plan: &LogicalPlan,
        column: impl Into<Column> + Clone,
    ) -> Result<Column> {
        let schema = plan.schema();
        let fallback_schemas = plan.fallback_normalize_schemas();
        let using_columns = plan.using_columns()?;
        column.into().normalize_with_schemas_and_ambiguity_check(
            &[&[schema], &fallback_schemas],
            &using_columns,
        )
    }

    /// Apply a join with on constraint and specified null equality
    /// If null_equals_null is true then null == null, else null != null
    pub fn join_detailed(
        self,
        right: LogicalPlan,   // 需要与该计划合并
        join_type: JoinType,  // 表示join的方式
        join_keys: (Vec<impl Into<Column>>, Vec<impl Into<Column>>),  // 连表用的字段
        filter: Option<Expr>,  // 对join结果进行过滤  (where的部分)
        null_equals_null: bool,  // null是否等于null
    ) -> Result<Self> {
        // 连表的列数量肯定是匹配的
        if join_keys.0.len() != join_keys.1.len() {
            return Err(DataFusionError::Plan(
                "left_keys and right_keys were not the same length".to_string(),
            ));
        }

        // 确保where使用的列出现在2个plan的schema中
        let filter = if let Some(expr) = filter {
            let filter = normalize_col_with_schemas_and_ambiguity_check(
                expr,
                &[&[self.schema(), right.schema()]],
                &[],
            )?;
            Some(filter)
        } else {
            None
        };

        let (left_keys, right_keys): (Vec<Result<Column>>, Vec<Result<Column>>) =
            join_keys
                .0
                .into_iter()
                .zip(join_keys.1.into_iter())
                // 遍历连表用到的列值对
                .map(|(l, r)| {
                    let l = l.into();
                    let r = r.into();

                    match (&l.relation, &r.relation) {
                        // 2个列都标记了所属的表
                        (Some(lr), Some(rr)) => {

                            // 表达式左侧对应的是join的左表
                            let l_is_left =
                                self.plan.schema().field_with_qualified_name(lr, &l.name);
                            // 左侧对应join右表
                            let l_is_right =
                                right.schema().field_with_qualified_name(lr, &l.name);
                            // 右侧对应join左表
                            let r_is_left =
                                self.plan.schema().field_with_qualified_name(rr, &r.name);
                            // 右侧对应join右表
                            let r_is_right =
                                right.schema().field_with_qualified_name(rr, &r.name);

                            match (l_is_left, l_is_right, r_is_left, r_is_right) {
                                // 左右，右左是正常情况
                                (_, Ok(_), Ok(_), _) => (Ok(r), Ok(l)),
                                (Ok(_), _, _, Ok(_)) => (Ok(l), Ok(r)),
                                _ => (
                                    // 这里是开启检查  不知道啥意义
                                    Self::normalize(&self.plan, l),
                                    Self::normalize(&right, r),
                                ),
                            }
                        }

                        // 左边的列声明了表  右边没有
                        (Some(lr), None) => {
                            let l_is_left =
                                self.plan.schema().field_with_qualified_name(lr, &l.name);
                            let l_is_right =
                                right.schema().field_with_qualified_name(lr, &l.name);

                            match (l_is_left, l_is_right) {
                                (Ok(_), _) => (Ok(l), Self::normalize(&right, r)),
                                (_, Ok(_)) => (Self::normalize(&self.plan, r), Ok(l)),
                                _ => (
                                    Self::normalize(&self.plan, l),
                                    Self::normalize(&right, r),
                                ),
                            }
                        }
                        (None, Some(rr)) => {
                            let r_is_left =
                                self.plan.schema().field_with_qualified_name(rr, &r.name);
                            let r_is_right =
                                right.schema().field_with_qualified_name(rr, &r.name);

                            match (r_is_left, r_is_right) {
                                (Ok(_), _) => (Ok(r), Self::normalize(&right, l)),
                                (_, Ok(_)) => (Self::normalize(&self.plan, l), Ok(r)),
                                _ => (
                                    Self::normalize(&self.plan, l),
                                    Self::normalize(&right, r),
                                ),
                            }
                        }
                        (None, None) => {
                            let mut swap = false;
                            let left_key = Self::normalize(&self.plan, l.clone())
                                .or_else(|_| {
                                    swap = true;
                                    Self::normalize(&right, l)
                                });
                            if swap {
                                (Self::normalize(&self.plan, r), left_key)
                            } else {
                                (left_key, Self::normalize(&right, r))
                            }
                        }
                    }
                })
                .unzip();

        // 在上面经过了顺序的调整  left_keys对应的字段是从左plan中获取的 right_keys对应右plan
        let left_keys = left_keys.into_iter().collect::<Result<Vec<Column>>>()?;
        let right_keys = right_keys.into_iter().collect::<Result<Vec<Column>>>()?;

        let on = left_keys
            .into_iter()
            .zip(right_keys.into_iter())
            .map(|(l, r)| (Expr::Column(l), Expr::Column(r)))
            .collect();
        let join_schema =
            build_join_schema(self.plan.schema(), right.schema(), &join_type)?;

        Ok(Self::from(LogicalPlan::Join(Join {
            left: Arc::new(self.plan),
            right: Arc::new(right),
            on,
            filter,
            join_type,
            join_constraint: JoinConstraint::On,
            schema: DFSchemaRef::new(join_schema),
            null_equals_null,
        })))
    }

    /// Apply a join with using constraint, which duplicates all join columns in output schema.
    /// TODO
    pub fn join_using(
        self,
        right: LogicalPlan,
        join_type: JoinType,
        using_keys: Vec<impl Into<Column> + Clone>,
    ) -> Result<Self> {
        let left_keys: Vec<Column> = using_keys
            .clone()
            .into_iter()
            .map(|c| Self::normalize(&self.plan, c))
            .collect::<Result<_>>()?;
        let right_keys: Vec<Column> = using_keys
            .into_iter()
            .map(|c| Self::normalize(&right, c))
            .collect::<Result<_>>()?;

        let on: Vec<(_, _)> = left_keys.into_iter().zip(right_keys.into_iter()).collect();
        let join_schema =
            build_join_schema(self.plan.schema(), right.schema(), &join_type)?;
        let mut join_on: Vec<(Expr, Expr)> = vec![];
        let mut filters: Option<Expr> = None;
        for (l, r) in &on {
            if self.plan.schema().has_column(l)
                && right.schema().has_column(r)
                && can_hash(self.plan.schema().field_from_column(l)?.data_type())
            {
                join_on.push((Expr::Column(l.clone()), Expr::Column(r.clone())));
            } else if self.plan.schema().has_column(l)
                && right.schema().has_column(r)
                && can_hash(self.plan.schema().field_from_column(r)?.data_type())
            {
                join_on.push((Expr::Column(r.clone()), Expr::Column(l.clone())));
            } else {
                let expr = binary_expr(
                    Expr::Column(l.clone()),
                    Operator::Eq,
                    Expr::Column(r.clone()),
                );
                match filters {
                    None => filters = Some(expr),
                    Some(filter_expr) => filters = Some(and(expr, filter_expr)),
                }
            }
        }

        if join_on.is_empty() {
            let join = Self::from(self.plan).cross_join(right)?;
            join.filter(filters.ok_or_else(|| {
                DataFusionError::Internal("filters should not be None here".to_string())
            })?)
        } else {
            Ok(Self::from(LogicalPlan::Join(Join {
                left: Arc::new(self.plan),
                right: Arc::new(right),
                on: join_on,
                filter: filters,
                join_type,
                join_constraint: JoinConstraint::Using,
                schema: DFSchemaRef::new(join_schema),
                null_equals_null: false,
            })))
        }
    }

    /// Apply a cross join
    /// TODO 笛卡尔积应该就是没有连表条件吧
    pub fn cross_join(self, right: LogicalPlan) -> Result<Self> {
        let schema = self.plan.schema().join(right.schema())?;
        Ok(Self::from(LogicalPlan::CrossJoin(CrossJoin {
            left: Arc::new(self.plan),
            right: Arc::new(right),
            schema: DFSchemaRef::new(schema),
        })))
    }

    /// Repartition
    /// 给plan包装一层分区
    pub fn repartition(self, partitioning_scheme: Partitioning) -> Result<Self> {
        Ok(Self::from(LogicalPlan::Repartition(Repartition {
            input: Arc::new(self.plan),
            partitioning_scheme,
        })))
    }

    /// Apply a window functions to extend the schema
    /// 追加窗口函数
    pub fn window(
        self,
        window_expr: impl IntoIterator<Item = impl Into<Expr>>,  // 这组窗口函数具有一样的分区键
    ) -> Result<Self> {
        // 简单理解就是将col变成全限定名
        let window_expr = normalize_cols(window_expr, &self.plan)?;
        let all_expr = window_expr.iter();

        // 确保这组表达式没有同名的
        validate_unique_names("Windows", all_expr.clone())?;
        // 获取window函数能够作用的field范围
        let mut window_fields: Vec<DFField> = self.plan.schema().fields().clone();

        // 除了plan的field外 还会追加窗口表达式用到的字段
        // exprlist_to_fields 当发现plan内部已经存在聚合函数同名field时 就会避免重复添加
        window_fields.extend_from_slice(&exprlist_to_fields(all_expr, &self.plan)?);
        let metadata = self.plan.schema().metadata().clone();

        Ok(Self::from(LogicalPlan::Window(Window {
            input: Arc::new(self.plan),
            window_expr,
            // 相当于把表示窗口函数结果的字段追加到schema中   (聚合/window结果会对应一个虚拟field 以displayname作为field名)
            schema: Arc::new(DFSchema::new_with_metadata(window_fields, metadata)?),
        })))
    }

    /// Apply an aggregate: grouping on the `group_expr` expressions
    /// and calculating `aggr_expr` aggregates for each distinct
    /// value of the `group_expr`;
    /// 简单的组合
    pub fn aggregate(
        self,
        group_expr: impl IntoIterator<Item = impl Into<Expr>>,
        aggr_expr: impl IntoIterator<Item = impl Into<Expr>>,
    ) -> Result<Self> {
        let group_expr = normalize_cols(group_expr, &self.plan)?;
        let aggr_expr = normalize_cols(aggr_expr, &self.plan)?;
        Ok(Self::from(LogicalPlan::Aggregate(Aggregate::try_new(
            Arc::new(self.plan),
            group_expr,
            aggr_expr,
        )?)))
    }

    /// Create an expression to represent the explanation of the plan
    ///
    /// if `analyze` is true, runs the actual plan and produces
    /// information about metrics during run.
    ///
    /// if `verbose` is true, prints out additional details.
    pub fn explain(self, verbose: bool, analyze: bool) -> Result<Self> {
        // explain的schema是固定的  (plan_type,plan)
        let schema = LogicalPlan::explain_schema();
        let schema = schema.to_dfschema_ref()?;

        if analyze {
            Ok(Self::from(LogicalPlan::Analyze(Analyze {
                verbose,
                input: Arc::new(self.plan),
                schema,
            })))
        } else {
            let stringified_plans =
                vec![self.plan.to_stringified(PlanType::InitialLogicalPlan)];

            Ok(Self::from(LogicalPlan::Explain(Explain {
                verbose,
                plan: Arc::new(self.plan),
                stringified_plans,
                schema,
                logical_optimization_succeeded: false,
            })))
        }
    }

    /// Process intersect set operator  TODO
    pub fn intersect(
        left_plan: LogicalPlan,
        right_plan: LogicalPlan,
        is_all: bool,
    ) -> Result<LogicalPlan> {
        LogicalPlanBuilder::intersect_or_except(
            left_plan,
            right_plan,
            JoinType::LeftSemi,
            is_all,
        )
    }

    /// Process except set operator  TODO
    pub fn except(
        left_plan: LogicalPlan,
        right_plan: LogicalPlan,
        is_all: bool,
    ) -> Result<LogicalPlan> {
        LogicalPlanBuilder::intersect_or_except(
            left_plan,
            right_plan,
            JoinType::LeftAnti,
            is_all,
        )
    }

    /// Process intersect or except
    /// TODO 先忽略 LeftSemi/LeftAnti
    fn intersect_or_except(
        left_plan: LogicalPlan,
        right_plan: LogicalPlan,
        join_type: JoinType,
        is_all: bool,
    ) -> Result<LogicalPlan> {
        let left_len = left_plan.schema().fields().len();
        let right_len = right_plan.schema().fields().len();

        if left_len != right_len {
            return Err(DataFusionError::Plan(format!(
                "INTERSECT/EXCEPT query must have the same number of columns. Left is {left_len} and right is {right_len}."
            )));
        }

        let join_keys = left_plan
            .schema()
            .fields()
            .iter()
            .zip(right_plan.schema().fields().iter())
            .map(|(left_field, right_field)| {
                (
                    (Column::from_name(left_field.name())),
                    (Column::from_name(right_field.name())),
                )
            })
            .unzip();
        if is_all {
            LogicalPlanBuilder::from(left_plan)
                .join_detailed(right_plan, join_type, join_keys, None, true)?
                .build()
        } else {
            LogicalPlanBuilder::from(left_plan)
                .distinct()?
                .join_detailed(right_plan, join_type, join_keys, None, true)?
                .build()
        }
    }

    /// Build the plan
    pub fn build(self) -> Result<LogicalPlan> {
        Ok(self.plan)
    }

    /// Apply a join with the expression on constraint.
    ///
    /// equi_exprs are "equijoin" predicates expressions on the existing and right inputs, respectively.
    ///
    /// filter: any other filter expression to apply during the join. equi_exprs predicates are likely
    /// to be evaluated more quickly than the filter expressions
    /// 与另一个plan进行join
    pub fn join_with_expr_keys(
        self,
        right: LogicalPlan,
        join_type: JoinType,
        // 用于join的字段
        equi_exprs: (Vec<impl Into<Expr>>, Vec<impl Into<Expr>>),
        filter: Option<Expr>,
    ) -> Result<Self> {
        if equi_exprs.0.len() != equi_exprs.1.len() {
            return Err(DataFusionError::Plan(
                "left_keys and right_keys were not the same length".to_string(),
            ));
        }

        let join_key_pairs = equi_exprs
            .0
            .into_iter()
            .zip(equi_exprs.1.into_iter())
            .map(|(l, r)| {
                let left_key = l.into();
                let right_key = r.into();

                // 确保连接用的field 出现在左plan或者右plan的字段中
                let left_using_columns = left_key.to_columns()?;
                let normalized_left_key = normalize_col_with_schemas_and_ambiguity_check(
                    left_key,
                    &[&[self.plan.schema(), right.schema()]],
                    &[left_using_columns],
                )?;

                // 同样要检测右表
                let right_using_columns = right_key.to_columns()?;
                let normalized_right_key = normalize_col_with_schemas_and_ambiguity_check(
                    right_key,
                    &[&[self.plan.schema(), right.schema()]],
                    &[right_using_columns],
                )?;

                // find valid equijoin   校验
                find_valid_equijoin_key_pair(
                        &normalized_left_key,
                        &normalized_right_key,
                        self.plan.schema().clone(),
                        right.schema().clone(),
                    )?.ok_or_else(||
                        DataFusionError::Plan(format!(
                            "can't create join plan, join key should belong to one input, error key: ({normalized_left_key},{normalized_right_key})"
                        )))
            })
            .collect::<Result<Vec<_>>>()?;

        //
        let join_schema =
            build_join_schema(self.plan.schema(), right.schema(), &join_type)?;

        Ok(Self::from(LogicalPlan::Join(Join {
            left: Arc::new(self.plan),
            right: Arc::new(right),
            on: join_key_pairs,
            filter,
            join_type,
            join_constraint: JoinConstraint::On,
            schema: DFSchemaRef::new(join_schema),
            null_equals_null: false,
        })))
    }

    /// Unnest the given column.   针对该列 解除嵌套
    pub fn unnest_column(self, column: impl Into<Column>) -> Result<Self> {
        Ok(Self::from(unnest(self.plan, column.into())?))
    }
}

/// Creates a schema for a join operation.
/// The fields from the left side are first
/// join请求 将2个schema进行合并
pub fn build_join_schema(
    left: &DFSchema,
    right: &DFSchema,
    join_type: &JoinType,
) -> Result<DFSchema> {
    let right_fields = right.fields();
    let left_fields = left.fields();

    let fields: Vec<DFField> = match join_type {
        JoinType::Inner | JoinType::Full | JoinType::Right => {
            // left then right    右连接为什么左边不需要设置成nullable呢
            left_fields
                .iter()
                .chain(right_fields.iter())
                .cloned()
                .collect()
        }
        JoinType::Left => {
            // left then right, right set to nullable in case of not matched scenario
            // 因为左连接 左边的数据总会展示 右边没有符合条件的记录 col会自动变成null
            let right_fields_nullable: Vec<DFField> = right_fields
                .iter()
                .map(|f| f.clone().with_nullable(true))
                .collect();

            left_fields
                .iter()
                .chain(&right_fields_nullable)
                .cloned()
                .collect()
        }
        // TODO
        JoinType::LeftSemi | JoinType::LeftAnti => {
            // Only use the left side for the schema
            left_fields.clone()
        }
        JoinType::RightSemi | JoinType::RightAnti => {
            // Only use the right side for the schema
            right_fields.clone()
        }
    };

    let mut metadata = left.metadata().clone();
    metadata.extend(right.metadata().clone());
    DFSchema::new_with_metadata(fields, metadata)
}

/// Errors if one or more expressions have equal names.
/// 检测是否出现了重名对象
pub(crate) fn validate_unique_names<'a>(
    node_name: &str,
    expressions: impl IntoIterator<Item = &'a Expr>, // 检查表达式
) -> Result<()> {
    let mut unique_names = HashMap::new();
    expressions.into_iter().enumerate().try_for_each(|(position, expr)| {
        let name = expr.display_name()?;
        match unique_names.get(&name) {
            None => {
                unique_names.insert(name, (position, expr));
                Ok(())
            },
            Some((existing_position, existing_expr)) => {
                Err(DataFusionError::Plan(
                    format!("{node_name} require unique expression names \
                             but the expression \"{existing_expr:?}\" at position {existing_position} and \"{expr:?}\" \
                             at position {position} have the same name. Consider aliasing (\"AS\") one of them.",
                            )
                ))
            }
        }
    })
}

pub fn project_with_column_index(
    expr: Vec<Expr>,
    input: Arc<LogicalPlan>,
    schema: DFSchemaRef,
) -> Result<LogicalPlan> {
    let alias_expr = expr
        .into_iter()
        .enumerate()
        .map(|(i, e)| match e {
            ignore_alias @ Expr::Alias { .. } => ignore_alias,
            ignore_col @ Expr::Column { .. } => ignore_col,
            // 给其他类型套上一层别名
            x => x.alias(schema.field(i).name()),
        })
        .collect::<Vec<_>>();
    Ok(LogicalPlan::Projection(Projection::try_new_with_schema(
        alias_expr, input, schema,
    )?))
}

/// Union two logical plans.
/// 将2个对象进行组合
pub fn union(left_plan: LogicalPlan, right_plan: LogicalPlan) -> Result<LogicalPlan> {
    let left_col_num = left_plan.schema().fields().len();

    // check union plan length same.
    let right_col_num = right_plan.schema().fields().len();
    // 合并的2个表 列数要一致
    if right_col_num != left_col_num {
        return Err(DataFusionError::Plan(format!(
            "Union queries must have the same number of columns, (left is {left_col_num}, right is {right_col_num})")
        ));
    }

    // create union schema
    let union_schema = (0..left_col_num)
        .map(|i| {
            let left_field = left_plan.schema().field(i);
            let right_field = right_plan.schema().field(i);

            // 左右2个表达式的 is_nullable/data_type 合并后得到field
            let nullable = left_field.is_nullable() || right_field.is_nullable();
            let data_type =
                comparison_coercion(left_field.data_type(), right_field.data_type())
                    .ok_or_else(|| {
                        DataFusionError::Plan(format!(
                    "UNION Column {} (type: {}) is not compatible with column {} (type: {})",
                    right_field.name(),
                    right_field.data_type(),
                    left_field.name(),
                    left_field.data_type()
                ))
                    })?;

            Ok(DFField::new(
                left_field.qualifier().cloned(),
                left_field.name(),
                data_type,
                nullable,
            ))
        })
        .collect::<Result<Vec<_>>>()?
        .to_dfschema()?;

    let inputs = vec![left_plan, right_plan]
        .into_iter()
        .flat_map(|p| match p {
            LogicalPlan::Union(Union { inputs, .. }) => inputs,
            x => vec![Arc::new(x)],
        })
        .map(|p| {
            // 简单理解就是在col与schema类型不匹配的情况下 套上一层Cast
            let plan = coerce_plan_expr_for_schema(&p, &union_schema)?;
            match plan {
                LogicalPlan::Projection(Projection { expr, input, .. }) => {
                    Ok(Arc::new(project_with_column_index(
                        expr,
                        input,
                        Arc::new(union_schema.clone()),
                    )?))
                }
                x => Ok(Arc::new(x)),
            }
        })
        .collect::<Result<Vec<_>>>()?;

    if inputs.is_empty() {
        return Err(DataFusionError::Plan("Empty UNION".to_string()));
    }

    Ok(LogicalPlan::Union(Union {
        inputs,
        schema: Arc::new(union_schema),
    }))
}

/// Create Projection
/// # Errors
/// This function errors under any of the following conditions:
/// * Two or more expressions have the same name
/// * An invalid expression is used (e.g. a `sort` expression)
/// 在原逻辑计划基础上 追加投影表达式
pub fn project(
    plan: LogicalPlan,
    expr: impl IntoIterator<Item = impl Into<Expr>>,  // 每个expr对应一个要展示的列
) -> Result<LogicalPlan> {
    let input_schema = plan.schema();
    let mut projected_expr = vec![];
    for e in expr {
        let e = e.into();
        match e {
            Expr::Wildcard => {
                projected_expr.extend(expand_wildcard(input_schema, &plan)?)
            }
            Expr::QualifiedWildcard { ref qualifier } => {
                projected_expr.extend(expand_qualified_wildcard(qualifier, input_schema)?)
            }
            // TODO 忽略前2个
            // 简单来说就是对col进行标准化后存入projected_expr
            _ => projected_expr
                .push(columnize_expr(normalize_col(e, &plan)?, input_schema)),
        }
    }
    validate_unique_names("Projections", projected_expr.iter())?;
    let input_schema = DFSchema::new_with_metadata(
        // 根据plan信息将expr 转换成field
        exprlist_to_fields(&projected_expr, &plan)?,
        plan.schema().metadata().clone(),
    )?;

    Ok(LogicalPlan::Projection(Projection::try_new_with_schema(
        projected_expr,
        Arc::new(plan.clone()),
        DFSchemaRef::new(input_schema),
    )?))
}

/// Create a SubqueryAlias to wrap a LogicalPlan.
/// 给子查询套上一层别名
pub fn subquery_alias(
    plan: LogicalPlan,
    alias: impl Into<OwnedTableReference>,
) -> Result<LogicalPlan> {
    Ok(LogicalPlan::SubqueryAlias(SubqueryAlias::try_new(
        plan, alias,
    )?))
}

/// Create a LogicalPlanBuilder representing a scan of a table with the provided name and schema.
/// This is mostly used for testing and documentation.
pub fn table_scan<'a>(
    name: Option<impl Into<TableReference<'a>>>,
    table_schema: &Schema,  // 这个方法产生的scan 实际上没有数据可查询    因为只能从tableSource中得到表结构 无法得到查询数据的逻辑计划
    projection: Option<Vec<usize>>,
) -> Result<LogicalPlanBuilder> {
    let table_source = table_source(table_schema);
    let name = name
        .map(|n| n.into())
        .unwrap_or_else(|| OwnedTableReference::bare(UNNAMED_TABLE))
        .to_owned_reference();
    LogicalPlanBuilder::scan(name, table_source, projection)
}

fn table_source(table_schema: &Schema) -> Arc<dyn TableSource> {
    let table_schema = Arc::new(table_schema.clone());
    Arc::new(LogicalTableSource { table_schema })
}

/// Wrap projection for a plan, if the join keys contains normal expression.
/// 对join结果加上一层投影
pub fn wrap_projection_for_join_if_necessary(
    join_keys: &[Expr],
    input: LogicalPlan,
) -> Result<(LogicalPlan, Vec<Column>, bool)> {
    let input_schema = input.schema();
    let alias_join_keys: Vec<Expr> = join_keys
        .iter()
        .map(|key| {
            // The display_name() of cast expression will ignore the cast info, and show the inner expression name.
            // If we do not add alais, it will throw same field name error in the schema when adding projection.
            // For example:
            //    input scan : [a, b, c],
            //    join keys: [cast(a as int)]
            //
            //  then a and cast(a as int) will use the same field name - `a` in projection schema.
            //  https://github.com/apache/arrow-datafusion/issues/4478
            if matches!(key, Expr::Cast(_)) || matches!(key, Expr::TryCast(_)) {
                let alias = format!("{key:?}");
                key.clone().alias(alias)
            } else {
                key.clone()
            }
        })
        .collect::<Vec<_>>();

    // 有不是column的 join_key 就代表需要追加投影
    let need_project = join_keys.iter().any(|key| !matches!(key, Expr::Column(_)));
    let plan = if need_project {
        let mut projection = expand_wildcard(input_schema, &input)?;
        let join_key_items = alias_join_keys
            .iter()
            .flat_map(|expr| expr.try_into_col().is_err().then_some(expr))
            .cloned()
            .collect::<HashSet<Expr>>();
        projection.extend(join_key_items);

        LogicalPlanBuilder::from(input)
            .project(projection)?
            .build()?
    } else {
        input
    };

    let join_on = alias_join_keys
        .into_iter()
        .map(|key| {
            key.try_into_col()
                .or_else(|_| Ok(Column::from_name(key.display_name()?)))
        })
        .collect::<Result<Vec<_>>>()?;

    Ok((plan, join_on, need_project))
}

/// Basic TableSource implementation intended for use in tests and documentation. It is expected
/// that users will provide their own TableSource implementations or use DataFusion's
/// DefaultTableSource.
pub struct LogicalTableSource {
    table_schema: SchemaRef,
}

impl LogicalTableSource {
    /// Create a new LogicalTableSource
    pub fn new(table_schema: SchemaRef) -> Self {
        Self { table_schema }
    }
}

impl TableSource for LogicalTableSource {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.table_schema.clone()
    }
}

/// Create an unnest plan.
pub fn unnest(input: LogicalPlan, column: Column) -> Result<LogicalPlan> {
    // 从schema中找到col对应的field
    let unnest_field = input.schema().field_from_column(&column)?;

    // Extract the type of the nested field in the list.
    // 主要是针对List类型 将元素类型抽取出来
    let unnested_field = match unnest_field.data_type() {
        DataType::List(field)
        | DataType::FixedSizeList(field, _)
        | DataType::LargeList(field) => DFField::new(
            unnest_field.qualifier().cloned(),
            unnest_field.name(),
            field.data_type().clone(),
            unnest_field.is_nullable(),
        ),
        _ => {
            // If the unnest field is not a list type return the input plan.
            return Ok(input);
        }
    };

    // Update the schema with the unnest column type changed to contain the nested type.
    let input_schema = input.schema();
    let fields = input_schema
        .fields()
        .iter()
        .map(|f| {
            // 将list类型替换成了内部类型
            if f == unnest_field {
                unnested_field.clone()
            } else {
                f.clone()
            }
        })
        .collect::<Vec<_>>();

    let schema = Arc::new(DFSchema::new_with_metadata(
        fields,
        input_schema.metadata().clone(),
    )?);

    // 用Unnest表示内部的list类型已经被解除嵌套了
    Ok(LogicalPlan::Unnest(Unnest {
        input: Arc::new(input),
        // 表示哪个col被解除嵌套
        column: unnested_field.qualified_column(),
        schema,
    }))
}

#[cfg(test)]
mod tests {
    use crate::{expr, expr_fn::exists};
    use arrow::datatypes::{DataType, Field};
    use datafusion_common::{OwnedTableReference, SchemaError, TableReference};

    use crate::logical_plan::StringifiedPlan;

    use super::*;
    use crate::{col, in_subquery, lit, scalar_subquery, sum};

    #[test]
    fn plan_builder_simple() -> Result<()> {
        let plan =
            table_scan(Some("employee_csv"), &employee_schema(), Some(vec![0, 3]))?
                .filter(col("state").eq(lit("CO")))?
                .project(vec![col("id")])?
                .build()?;

        let expected = "Projection: employee_csv.id\
        \n  Filter: employee_csv.state = Utf8(\"CO\")\
        \n    TableScan: employee_csv projection=[id, state]";

        assert_eq!(expected, format!("{plan:?}"));

        Ok(())
    }

    #[test]
    fn plan_builder_schema() {
        let schema = employee_schema();
        let projection = None;
        let plan =
            LogicalPlanBuilder::scan("employee_csv", table_source(&schema), projection)
                .unwrap();
        let expected = DFSchema::try_from_qualified_schema(
            TableReference::bare("employee_csv"),
            &schema,
        )
        .unwrap();
        assert_eq!(&expected, plan.schema().as_ref());

        // Note scan of "EMPLOYEE_CSV" is treated as a SQL identifer
        // (and thus normalized to "employee"csv") as well
        let projection = None;
        let plan =
            LogicalPlanBuilder::scan("EMPLOYEE_CSV", table_source(&schema), projection)
                .unwrap();
        assert_eq!(&expected, plan.schema().as_ref());
    }

    #[test]
    fn plan_builder_empty_name() {
        let schema = employee_schema();
        let projection = None;
        let err =
            LogicalPlanBuilder::scan("", table_source(&schema), projection).unwrap_err();
        assert_eq!(
            err.to_string(),
            "Error during planning: table_name cannot be empty"
        );
    }

    #[test]
    fn plan_builder_aggregate() -> Result<()> {
        let plan =
            table_scan(Some("employee_csv"), &employee_schema(), Some(vec![3, 4]))?
                .aggregate(
                    vec![col("state")],
                    vec![sum(col("salary")).alias("total_salary")],
                )?
                .project(vec![col("state"), col("total_salary")])?
                .limit(2, Some(10))?
                .build()?;

        let expected = "Limit: skip=2, fetch=10\
                \n  Projection: employee_csv.state, total_salary\
                \n    Aggregate: groupBy=[[employee_csv.state]], aggr=[[SUM(employee_csv.salary) AS total_salary]]\
                \n      TableScan: employee_csv projection=[state, salary]";

        assert_eq!(expected, format!("{plan:?}"));

        Ok(())
    }

    #[test]
    fn plan_builder_sort() -> Result<()> {
        let plan =
            table_scan(Some("employee_csv"), &employee_schema(), Some(vec![3, 4]))?
                .sort(vec![
                    Expr::Sort(expr::Sort::new(Box::new(col("state")), true, true)),
                    Expr::Sort(expr::Sort::new(Box::new(col("salary")), false, false)),
                ])?
                .build()?;

        let expected = "Sort: employee_csv.state ASC NULLS FIRST, employee_csv.salary DESC NULLS LAST\
        \n  TableScan: employee_csv projection=[state, salary]";

        assert_eq!(expected, format!("{plan:?}"));

        Ok(())
    }

    #[test]
    fn plan_using_join_wildcard_projection() -> Result<()> {
        let t2 = table_scan(Some("t2"), &employee_schema(), None)?.build()?;

        let plan = table_scan(Some("t1"), &employee_schema(), None)?
            .join_using(t2, JoinType::Inner, vec!["id"])?
            .project(vec![Expr::Wildcard])?
            .build()?;

        // id column should only show up once in projection
        let expected = "Projection: t1.id, t1.first_name, t1.last_name, t1.state, t1.salary, t2.first_name, t2.last_name, t2.state, t2.salary\
        \n  Inner Join: Using t1.id = t2.id\
        \n    TableScan: t1\
        \n    TableScan: t2";

        assert_eq!(expected, format!("{plan:?}"));

        Ok(())
    }

    #[test]
    fn plan_builder_union_combined_single_union() -> Result<()> {
        let plan =
            table_scan(Some("employee_csv"), &employee_schema(), Some(vec![3, 4]))?;

        let plan = plan
            .clone()
            .union(plan.clone().build()?)?
            .union(plan.clone().build()?)?
            .union(plan.build()?)?
            .build()?;

        // output has only one union
        let expected = "Union\
        \n  TableScan: employee_csv projection=[state, salary]\
        \n  TableScan: employee_csv projection=[state, salary]\
        \n  TableScan: employee_csv projection=[state, salary]\
        \n  TableScan: employee_csv projection=[state, salary]";

        assert_eq!(expected, format!("{plan:?}"));

        Ok(())
    }

    #[test]
    fn plan_builder_union_distinct_combined_single_union() -> Result<()> {
        let plan =
            table_scan(Some("employee_csv"), &employee_schema(), Some(vec![3, 4]))?;

        let plan = plan
            .clone()
            .union_distinct(plan.clone().build()?)?
            .union_distinct(plan.clone().build()?)?
            .union_distinct(plan.build()?)?
            .build()?;

        // output has only one union
        let expected = "\
        Distinct:\
        \n  Union\
        \n    TableScan: employee_csv projection=[state, salary]\
        \n    TableScan: employee_csv projection=[state, salary]\
        \n    TableScan: employee_csv projection=[state, salary]\
        \n    TableScan: employee_csv projection=[state, salary]";

        assert_eq!(expected, format!("{plan:?}"));

        Ok(())
    }

    #[test]
    fn plan_builder_union_different_num_columns_error() -> Result<()> {
        let plan1 =
            table_scan(TableReference::none(), &employee_schema(), Some(vec![3]))?;
        let plan2 =
            table_scan(TableReference::none(), &employee_schema(), Some(vec![3, 4]))?;

        let expected = "Error during planning: Union queries must have the same number of columns, (left is 1, right is 2)";
        let err_msg1 = plan1.clone().union(plan2.clone().build()?).unwrap_err();
        let err_msg2 = plan1.union_distinct(plan2.build()?).unwrap_err();

        assert_eq!(err_msg1.to_string(), expected);
        assert_eq!(err_msg2.to_string(), expected);

        Ok(())
    }

    #[test]
    fn plan_builder_simple_distinct() -> Result<()> {
        let plan =
            table_scan(Some("employee_csv"), &employee_schema(), Some(vec![0, 3]))?
                .filter(col("state").eq(lit("CO")))?
                .project(vec![col("id")])?
                .distinct()?
                .build()?;

        let expected = "\
        Distinct:\
        \n  Projection: employee_csv.id\
        \n    Filter: employee_csv.state = Utf8(\"CO\")\
        \n      TableScan: employee_csv projection=[id, state]";

        assert_eq!(expected, format!("{plan:?}"));

        Ok(())
    }

    #[test]
    fn exists_subquery() -> Result<()> {
        let foo = test_table_scan_with_name("foo")?;
        let bar = test_table_scan_with_name("bar")?;

        let subquery = LogicalPlanBuilder::from(foo)
            .project(vec![col("a")])?
            .filter(col("a").eq(col("bar.a")))?
            .build()?;

        let outer_query = LogicalPlanBuilder::from(bar)
            .project(vec![col("a")])?
            .filter(exists(Arc::new(subquery)))?
            .build()?;

        let expected = "Filter: EXISTS (<subquery>)\
        \n  Subquery:\
        \n    Filter: foo.a = bar.a\
        \n      Projection: foo.a\
        \n        TableScan: foo\
        \n  Projection: bar.a\
        \n    TableScan: bar";
        assert_eq!(expected, format!("{outer_query:?}"));

        Ok(())
    }

    #[test]
    fn filter_in_subquery() -> Result<()> {
        let foo = test_table_scan_with_name("foo")?;
        let bar = test_table_scan_with_name("bar")?;

        let subquery = LogicalPlanBuilder::from(foo)
            .project(vec![col("a")])?
            .filter(col("a").eq(col("bar.a")))?
            .build()?;

        // SELECT a FROM bar WHERE a IN (SELECT a FROM foo WHERE a = bar.a)
        let outer_query = LogicalPlanBuilder::from(bar)
            .project(vec![col("a")])?
            .filter(in_subquery(col("a"), Arc::new(subquery)))?
            .build()?;

        let expected = "Filter: bar.a IN (<subquery>)\
        \n  Subquery:\
        \n    Filter: foo.a = bar.a\
        \n      Projection: foo.a\
        \n        TableScan: foo\
        \n  Projection: bar.a\
        \n    TableScan: bar";
        assert_eq!(expected, format!("{outer_query:?}"));

        Ok(())
    }

    #[test]
    fn select_scalar_subquery() -> Result<()> {
        let foo = test_table_scan_with_name("foo")?;
        let bar = test_table_scan_with_name("bar")?;

        let subquery = LogicalPlanBuilder::from(foo)
            .project(vec![col("b")])?
            .filter(col("a").eq(col("bar.a")))?
            .build()?;

        // SELECT (SELECT a FROM foo WHERE a = bar.a) FROM bar
        let outer_query = LogicalPlanBuilder::from(bar)
            .project(vec![scalar_subquery(Arc::new(subquery))])?
            .build()?;

        let expected = "Projection: (<subquery>)\
        \n  Subquery:\
        \n    Filter: foo.a = bar.a\
        \n      Projection: foo.b\
        \n        TableScan: foo\
        \n  TableScan: bar";
        assert_eq!(expected, format!("{outer_query:?}"));

        Ok(())
    }

    #[test]
    fn projection_non_unique_names() -> Result<()> {
        let plan = table_scan(
            Some("employee_csv"),
            &employee_schema(),
            // project id and first_name by column index
            Some(vec![0, 1]),
        )?
        // two columns with the same name => error
        .project(vec![col("id"), col("first_name").alias("id")]);

        match plan {
            Err(DataFusionError::SchemaError(SchemaError::AmbiguousReference {
                field:
                    Column {
                        relation: Some(OwnedTableReference::Bare { table }),
                        name,
                    },
            })) => {
                assert_eq!("employee_csv", table);
                assert_eq!("id", &name);
                Ok(())
            }
            _ => Err(DataFusionError::Plan(
                "Plan should have returned an DataFusionError::SchemaError".to_string(),
            )),
        }
    }

    #[test]
    fn aggregate_non_unique_names() -> Result<()> {
        let plan = table_scan(
            Some("employee_csv"),
            &employee_schema(),
            // project state and salary by column index
            Some(vec![3, 4]),
        )?
        // two columns with the same name => error
        .aggregate(vec![col("state")], vec![sum(col("salary")).alias("state")]);

        match plan {
            Err(DataFusionError::SchemaError(SchemaError::AmbiguousReference {
                field:
                    Column {
                        relation: Some(OwnedTableReference::Bare { table }),
                        name,
                    },
            })) => {
                assert_eq!("employee_csv", table);
                assert_eq!("state", &name);
                Ok(())
            }
            _ => Err(DataFusionError::Plan(
                "Plan should have returned an DataFusionError::SchemaError".to_string(),
            )),
        }
    }

    fn employee_schema() -> Schema {
        Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("first_name", DataType::Utf8, false),
            Field::new("last_name", DataType::Utf8, false),
            Field::new("state", DataType::Utf8, false),
            Field::new("salary", DataType::Int32, false),
        ])
    }

    #[test]
    fn stringified_plan() {
        let stringified_plan =
            StringifiedPlan::new(PlanType::InitialLogicalPlan, "...the plan...");
        assert!(stringified_plan.should_display(true));
        assert!(!stringified_plan.should_display(false)); // not in non verbose mode

        let stringified_plan =
            StringifiedPlan::new(PlanType::FinalLogicalPlan, "...the plan...");
        assert!(stringified_plan.should_display(true));
        assert!(stringified_plan.should_display(false)); // display in non verbose mode too

        let stringified_plan =
            StringifiedPlan::new(PlanType::InitialPhysicalPlan, "...the plan...");
        assert!(stringified_plan.should_display(true));
        assert!(!stringified_plan.should_display(false)); // not in non verbose mode

        let stringified_plan =
            StringifiedPlan::new(PlanType::FinalPhysicalPlan, "...the plan...");
        assert!(stringified_plan.should_display(true));
        assert!(stringified_plan.should_display(false)); // display in non verbose mode

        let stringified_plan = StringifiedPlan::new(
            PlanType::OptimizedLogicalPlan {
                optimizer_name: "random opt pass".into(),
            },
            "...the plan...",
        );
        assert!(stringified_plan.should_display(true));
        assert!(!stringified_plan.should_display(false));
    }

    fn test_table_scan_with_name(name: &str) -> Result<LogicalPlan> {
        let schema = Schema::new(vec![
            Field::new("a", DataType::UInt32, false),
            Field::new("b", DataType::UInt32, false),
            Field::new("c", DataType::UInt32, false),
        ]);
        table_scan(Some(name), &schema, None)?.build()
    }

    #[test]
    fn plan_builder_intersect_different_num_columns_error() -> Result<()> {
        let plan1 =
            table_scan(TableReference::none(), &employee_schema(), Some(vec![3]))?;
        let plan2 =
            table_scan(TableReference::none(), &employee_schema(), Some(vec![3, 4]))?;

        let expected = "Error during planning: INTERSECT/EXCEPT query must have the same number of columns. \
         Left is 1 and right is 2.";
        let err_msg1 =
            LogicalPlanBuilder::intersect(plan1.build()?, plan2.build()?, true)
                .unwrap_err();

        assert_eq!(err_msg1.to_string(), expected);

        Ok(())
    }

    #[test]
    fn plan_builder_unnest() -> Result<()> {
        // Unnesting a simple column should return the child plan.
        let plan = nested_table_scan("test_table")?
            .unnest_column("scalar")?
            .build()?;

        let expected = "TableScan: test_table";
        assert_eq!(expected, format!("{plan:?}"));

        // Unnesting the strings list.
        let plan = nested_table_scan("test_table")?
            .unnest_column("strings")?
            .build()?;

        let expected = "\
        Unnest: test_table.strings\
        \n  TableScan: test_table";
        assert_eq!(expected, format!("{plan:?}"));

        // Check unnested field is a scalar
        let field = plan
            .schema()
            .field_with_name(Some(&TableReference::bare("test_table")), "strings")
            .unwrap();
        assert_eq!(&DataType::Utf8, field.data_type());

        // Unnesting multiple fields.
        let plan = nested_table_scan("test_table")?
            .unnest_column("strings")?
            .unnest_column("structs")?
            .build()?;

        let expected = "\
        Unnest: test_table.structs\
        \n  Unnest: test_table.strings\
        \n    TableScan: test_table";
        assert_eq!(expected, format!("{plan:?}"));

        // Check unnested struct list field should be a struct.
        let field = plan
            .schema()
            .field_with_name(Some(&TableReference::bare("test_table")), "structs")
            .unwrap();
        assert!(matches!(field.data_type(), DataType::Struct(_)));

        // Unnesting missing column should fail.
        let plan = nested_table_scan("test_table")?.unnest_column("missing");
        assert!(plan.is_err());

        Ok(())
    }

    fn nested_table_scan(table_name: &str) -> Result<LogicalPlanBuilder> {
        // Create a schema with a scalar field, a list of strings, and a list of structs.
        let struct_field = Field::new_struct(
            "item",
            vec![
                Field::new("a", DataType::UInt32, false),
                Field::new("b", DataType::UInt32, false),
            ],
            false,
        );
        let string_field = Field::new("item", DataType::Utf8, false);
        let schema = Schema::new(vec![
            Field::new("scalar", DataType::UInt32, false),
            Field::new_list("strings", string_field, false),
            Field::new_list("structs", struct_field, false),
        ]);

        table_scan(Some(table_name), &schema, None)
    }

    #[test]
    fn test_union_after_join() -> Result<()> {
        let values = vec![vec![lit(1)]];

        let left = LogicalPlanBuilder::values(values.clone())?
            .alias("left")?
            .build()?;
        let right = LogicalPlanBuilder::values(values)?
            .alias("right")?
            .build()?;

        let join = LogicalPlanBuilder::from(left).cross_join(right)?.build()?;

        let _ = LogicalPlanBuilder::from(join.clone())
            .union(join)?
            .build()?;

        Ok(())
    }
}
