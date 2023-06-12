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

use crate::planner::{ContextProvider, PlannerContext, SqlToRel};
use crate::utils::{
    check_columns_satisfy_exprs, extract_aliases, rebase_expr, resolve_aliases_to_exprs,
    resolve_columns, resolve_positions_to_exprs,
};
use datafusion_common::{DataFusionError, Result};
use datafusion_expr::expr_rewriter::{
    normalize_col, normalize_col_with_schemas_and_ambiguity_check,
};
use datafusion_expr::logical_plan::builder::project;
use datafusion_expr::utils::{
    expand_qualified_wildcard, expand_wildcard, expr_as_column_expr, expr_to_columns,
    find_aggregate_exprs, find_window_exprs,
};
use datafusion_expr::Expr::Alias;
use datafusion_expr::{
    Expr, Filter, GroupingSet, LogicalPlan, LogicalPlanBuilder, Partitioning,
};
use sqlparser::ast::{Expr as SQLExpr, WildcardAdditionalOptions};
use sqlparser::ast::{Select, SelectItem, TableWithJoins};
use std::collections::HashSet;
use std::sync::Arc;

impl<'a, S: ContextProvider> SqlToRel<'a, S> {

    /// Generate a logic plan from an SQL select
    pub(super) fn select_to_plan(
        &self,
        select: Select,  // 处理select语句
        planner_context: &mut PlannerContext,
    ) -> Result<LogicalPlan> {
        // check for unsupported syntax first  这里是一些还不支持的属性
        if !select.cluster_by.is_empty() {
            return Err(DataFusionError::NotImplemented("CLUSTER BY".to_string()));
        }
        if !select.lateral_views.is_empty() {
            return Err(DataFusionError::NotImplemented("LATERAL VIEWS".to_string()));
        }
        if select.qualify.is_some() {
            return Err(DataFusionError::NotImplemented("QUALIFY".to_string()));
        }
        if select.top.is_some() {
            return Err(DataFusionError::NotImplemented("TOP".to_string()));
        }
        if !select.sort_by.is_empty() {
            return Err(DataFusionError::NotImplemented("SORT BY".to_string()));
        }
        if select.into.is_some() {
            return Err(DataFusionError::NotImplemented("INTO".to_string()));
        }

        // process `from` clause   处理from的部分
        let plan = self.plan_from_tables(select.from, planner_context)?;
        // 当没有指定from的时候 会产生一个空对象
        let empty_from = matches!(plan, LogicalPlan::EmptyRelation(_));

        // process `where` clause  处理where的部分 产生一个filter类型的plan
        let plan = self.plan_selection(select.selection, plan, planner_context)?;

        // process the SELECT expressions, with wildcards expanded.
        // 现在处理select部分  得到所有要展示的Column
        let select_exprs = self.prepare_select_exprs(
            &plan,
            select.projection,  // 代表需要展示的结果列
            empty_from,
            planner_context,
        )?;

        // having and group by clause may reference aliases defined in select projection
        // 生成投影plan
        let projected_plan = self.project(plan.clone(), select_exprs.clone())?;
        let mut combined_schema = (**projected_plan.schema()).clone();
        // 一般来说2个schema是一样的
        combined_schema.merge(plan.schema());

        // this alias map is resolved and looked up in both having exprs and group by exprs
        // 提取出别名与被包裹的Column表达式
        let alias_map = extract_aliases(&select_exprs);

        // Optionally the HAVING expression.
        let having_expr_opt = select
            .having
            .map::<Result<Expr>, _>(|having_expr| {
                let having_expr = self.sql_expr_to_logical_expr(
                    having_expr,
                    &combined_schema,
                    planner_context,
                )?;
                // This step "dereferences" any aliases in the HAVING clause.
                //
                // This is how we support queries with HAVING expressions that
                // refer to aliased columns.
                //
                // For example:
                //
                //   SELECT c1, MAX(c2) AS m FROM t GROUP BY c1 HAVING m > 10;
                //
                // are rewritten as, respectively:
                //
                //   SELECT c1, MAX(c2) AS m FROM t GROUP BY c1 HAVING MAX(c2) > 10;
                //   提取出having用到的别名
                let having_expr = resolve_aliases_to_exprs(&having_expr, &alias_map)?;
                // resolve_aliases_to_exprs 是的having中使用到的列名找到了相对应的表信息
                normalize_col(having_expr, &projected_plan)
            })
            .transpose()?;

        // The outer expressions we will search through for
        // aggregates. Aggregates may be sourced from the SELECT...
        let mut aggr_expr_haystack = select_exprs.clone();
        // ... or from the HAVING.     聚合函数必然作用在select或者having字段上
        if let Some(having_expr) = &having_expr_opt {
            aggr_expr_haystack.push(having_expr.clone());
        }

        // All of the aggregate expressions (deduplicated).
        // 找到聚合函数表达式
        let aggr_exprs = find_aggregate_exprs(&aggr_expr_haystack);

        // All of the group by expressions  这里是group by 相关的
        let group_by_exprs = select
            .group_by
            .into_iter()
            .map(|e| {
                // 产生group by表达式   group by 其实也就是一个column
                let group_by_expr =
                    self.sql_expr_to_logical_expr(e, &combined_schema, planner_context)?;
                // aliases from the projection can conflict with same-named expressions in the input
                let mut alias_map = alias_map.clone();
                for f in plan.schema().fields() {
                    alias_map.remove(f.name());
                }

                // 为group by中不知道表信息的列绑定表信息
                let group_by_expr = resolve_aliases_to_exprs(&group_by_expr, &alias_map)?;

                // 支持使用标量值来表示group by  标量值会在select中找到一个对应的列
                let group_by_expr =
                    resolve_positions_to_exprs(&group_by_expr, &select_exprs)
                        .unwrap_or(group_by_expr);
                let group_by_expr = normalize_col(group_by_expr, &projected_plan)?;
                // 确保col在schema中
                self.validate_schema_satisfies_exprs(
                    plan.schema(),
                    &[group_by_expr.clone()],
                )?;
                Ok(group_by_expr)
            })
            .collect::<Result<Vec<Expr>>>()?;

        // process group by, aggregation or having   包含聚合表达式或者group by
        let (plan, mut select_exprs_post_aggr, having_expr_post_aggr) = if !group_by_exprs
            .is_empty()
            || !aggr_exprs.is_empty()
        {
            // 将表达式组合后 成为一个Aggregate类型的plan
            self.aggregate(
                plan,
                &select_exprs,
                having_expr_opt.as_ref(),
                group_by_exprs,
                aggr_exprs,
            )?
        } else {
            // having必须与group by 一同出现
            match having_expr_opt {
                Some(having_expr) => return Err(DataFusionError::Plan(
                    format!("HAVING clause references: {having_expr} must appear in the GROUP BY clause or be used in an aggregate function"))),
                None => (plan, select_exprs, having_expr_opt)
            }
        };

        // having要作用在Aggregate之上
        let plan = if let Some(having_expr_post_aggr) = having_expr_post_aggr {
            LogicalPlanBuilder::from(plan)
                .filter(having_expr_post_aggr)?
                .build()?
        } else {
            // 没有having函数  不需要追加filter了
            plan
        };

        // process window function  检查select中是否有窗口函数
        let window_func_exprs = find_window_exprs(&select_exprs_post_aggr);

        let plan = if window_func_exprs.is_empty() {
            plan
        } else {
            // 在select的结果上套一层窗口函数 (在聚合函数之后)
            let plan = LogicalPlanBuilder::window_plan(plan, window_func_exprs.clone())?;

            // re-write the projection
            select_exprs_post_aggr = select_exprs_post_aggr
                .iter()
                .map(|expr| rebase_expr(expr, &window_func_exprs, &plan))
                .collect::<Result<Vec<Expr>>>()?;

            plan
        };

        // final projection  修正投影的列
        let plan = project(plan, select_exprs_post_aggr)?;

        // process distinct clause
        // 如果有distinct表达式 对结果集进行去重
        let plan = if select.distinct {
            LogicalPlanBuilder::from(plan).distinct()?.build()
        } else {
            Ok(plan)
        }?;

        // DISTRIBUTE BY
        // 需要对结果进行分区
        if !select.distribute_by.is_empty() {
            let x = select
                .distribute_by
                .iter()
                .map(|e| {
                    self.sql_expr_to_logical_expr(
                        e.clone(),
                        &combined_schema,
                        planner_context,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            LogicalPlanBuilder::from(plan)
                .repartition(Partitioning::DistributeBy(x))?
                .build()
        } else {
            Ok(plan)
        }
    }

    // 处理where部分
    fn plan_selection(
        &self,
        selection: Option<SQLExpr>,
        plan: LogicalPlan,  // 对应from转换后的plan
        planner_context: &mut PlannerContext,
    ) -> Result<LogicalPlan> {
        match selection {
            Some(predicate_expr) => {
                // 获取所有可能出现的schema   where的条件字段肯定要出现在其中的
                let fallback_schemas = plan.fallback_normalize_schemas();
                // 如果是子查询的情况  该属性对应外部语句的schema
                let outer_query_schema = planner_context.outer_query_schema().cloned();
                let outer_query_schema_vec = outer_query_schema
                    .as_ref()
                    .map(|schema| vec![schema])
                    .unwrap_or_else(Vec::new);

                // where转换后的结果 成为了针对查询结果的过滤器
                let filter_expr =
                    self.sql_to_expr(predicate_expr, plan.schema(), planner_context)?;
                // 收集查询过程中用到的所有列
                let mut using_columns = HashSet::new();
                expr_to_columns(&filter_expr, &mut using_columns)?;
                // 需要确保where条件用到的col都出现在schema中
                let filter_expr = normalize_col_with_schemas_and_ambiguity_check(
                    filter_expr,
                    &[&[plan.schema()], &fallback_schemas, &outer_query_schema_vec],
                    &[using_columns],
                )?;

                Ok(LogicalPlan::Filter(Filter::try_new(
                    filter_expr,
                    Arc::new(plan),
                )?))
            }
            // 代表没有where过滤条件
            None => Ok(plan),
        }
    }

    // 处理select的from
    pub(crate) fn plan_from_tables(
        &self,
        mut from: Vec<TableWithJoins>,  // from后面会跟着目标表名，同时如何涉及连表操作 会有join
        planner_context: &mut PlannerContext,
    ) -> Result<LogicalPlan> {
        match from.len() {
            // 没有指定查询目标 返回空数据
            0 => Ok(LogicalPlanBuilder::empty(true).build()?),
            1 => {
                let from = from.remove(0);
                // 生成表的查询计划。如果有join，会嵌套在plan上
                self.plan_table_with_joins(from, planner_context)
            }
            _ => {
                let mut plans = from
                    .into_iter()
                    .map(|t| self.plan_table_with_joins(t, planner_context));

                let mut left = LogicalPlanBuilder::from(plans.next().unwrap()?);

                // TODO 产生笛卡尔积  可以先忽略这种情况
                for right in plans {
                    left = left.cross_join(right?)?;
                }
                Ok(left.build()?)
            }
        }
    }

    /// Returns the `Expr`'s corresponding to a SQL query's SELECT expressions.
    ///
    /// Wildcards are expanded into the concrete list of columns.
    /// 处理select部分
    fn prepare_select_exprs(
        &self,
        plan: &LogicalPlan,  // 对应的plan是在from的基础上套了一层where(filter)
        projection: Vec<SelectItem>,  // 对应需要展示的结果列
        empty_from: bool,    // 代表没有指定from 也就是无法查询到数据
        planner_context: &mut PlannerContext,
    ) -> Result<Vec<Expr>> {
        projection
            .into_iter()
            .map(|expr| self.sql_select_to_rex(expr, plan, empty_from, planner_context))
            // 将每个映射的列转换成Column  通配符会被转换成相关表的所有Column
            .flat_map(|result| match result {
                Ok(vec) => vec.into_iter().map(Ok).collect(),
                Err(err) => vec![Err(err)],
            })
            .collect::<Result<Vec<Expr>>>()
    }

    /// Generate a relational expression from a select SQL expression
    /// 针对select的每个列字段进行触发
    fn sql_select_to_rex(
        &self,
        sql: SelectItem,  // 对应查询的某个列
        plan: &LogicalPlan,  // from+where组合后的plan
        empty_from: bool,  // from是否没有指定目标
        planner_context: &mut PlannerContext,
    ) -> Result<Vec<Expr>> {
        match sql {
            // expr实际上对应的是col (虽然外层还可能嵌套乱七八糟的东西)
            SelectItem::UnnamedExpr(expr) => {
                let expr = self.sql_to_expr(expr, plan.schema(), planner_context)?;
                // 确保expr的列 出现在plan的schema中
                let col = normalize_col_with_schemas_and_ambiguity_check(
                    expr,
                    &[&[plan.schema()]],
                    &plan.using_columns()?,
                )?;
                Ok(vec![col])
            }
            SelectItem::ExprWithAlias { expr, alias } => {
                // 对应结果列
                let select_expr =
                    self.sql_to_expr(expr, plan.schema(), planner_context)?;
                let col = normalize_col_with_schemas_and_ambiguity_check(
                    select_expr,
                    &[&[plan.schema()]],
                    &plan.using_columns()?,
                )?;
                // 套上一层别名
                let expr = Alias(Box::new(col), self.normalizer.normalize(alias));
                Ok(vec![expr])
            }
            // 代表查询的是* 所有字段
            SelectItem::Wildcard(options) => {
                Self::check_wildcard_options(options)?;

                if empty_from {
                    return Err(DataFusionError::Plan(
                        "SELECT * with no tables specified is not valid".to_string(),
                    ));
                }
                // do not expand from outer schema
                expand_wildcard(plan.schema().as_ref(), plan)
            }
            // 代表 table.*
            SelectItem::QualifiedWildcard(ref object_name, options) => {
                Self::check_wildcard_options(options)?;

                let qualifier = format!("{object_name}");
                // do not expand from outer schema   将某个表的通配符展开
                expand_qualified_wildcard(&qualifier, plan.schema().as_ref())
            }
        }
    }

    fn check_wildcard_options(options: WildcardAdditionalOptions) -> Result<()> {
        let WildcardAdditionalOptions {
            opt_exclude,
            opt_except,
            opt_rename,
            opt_replace,
        } = options;

        if opt_exclude.is_some()
            || opt_except.is_some()
            || opt_rename.is_some()
            || opt_replace.is_some()
        {
            Err(DataFusionError::NotImplemented(
                "wildcard * with EXCLUDE, EXCEPT, RENAME or REPLACE not supported "
                    .to_string(),
            ))
        } else {
            Ok(())
        }
    }

    /// Wrap a plan in a projection
    /// 产生一个投影类型的plan
    fn project(&self, input: LogicalPlan, expr: Vec<Expr>) -> Result<LogicalPlan> {
        // 确保列可以在schema中找到
        self.validate_schema_satisfies_exprs(input.schema(), &expr)?;
        LogicalPlanBuilder::from(input).project(expr)?.build()
    }

    /// Create an aggregate plan.
    ///
    /// An aggregate plan consists of grouping expressions, aggregate expressions, and an
    /// optional HAVING expression (which is a filter on the output of the aggregate).
    ///
    /// # Arguments
    ///
    /// * `input`           - The input plan that will be aggregated. The grouping, aggregate, and
    ///                       "having" expressions must all be resolvable from this plan.
    /// * `select_exprs`    - The projection expressions from the SELECT clause.
    /// * `having_expr_opt` - Optional HAVING clause.
    /// * `group_by_exprs`  - Grouping expressions from the GROUP BY clause. These can be column
    ///                       references or more complex expressions.
    /// * `aggr_exprs`      - Aggregate expressions, such as `SUM(a)` or `COUNT(1)`.
    ///
    /// # Return
    ///
    /// The return value is a triplet of the following items:
    ///
    /// * `plan`                   - A [LogicalPlan::Aggregate] plan for the newly created aggregate.
    /// * `select_exprs_post_aggr` - The projection expressions rewritten to reference columns from
    ///                              the aggregate
    /// * `having_expr_post_aggr`  - The "having" expression rewritten to reference a column from
    ///                              the aggregate
    /// 当存在group by 或者聚合函数时  调用该方法进行聚合
    fn aggregate(
        &self,
        input: LogicalPlan,  // 已经整合了 from where
        select_exprs: &[Expr],  // 对应要查询的所有列
        having_expr_opt: Option<&Expr>,  // having表达式
        group_by_exprs: Vec<Expr>,  // group by表达式
        aggr_exprs: Vec<Expr>,  // 聚合表达式
    ) -> Result<(LogicalPlan, Vec<Expr>, Option<Expr>)> {
        // create the aggregate plan  这里就是简单的赋值
        let plan = LogicalPlanBuilder::from(input.clone())
            .aggregate(group_by_exprs.clone(), aggr_exprs.clone())?
            .build()?;

        // in this next section of code we are re-writing the projection to refer to columns
        // output by the aggregate plan. For example, if the projection contains the expression
        // `SUM(a)` then we replace that with a reference to a column `SUM(a)` produced by
        // the aggregate plan.

        // combine the original grouping and aggregate expressions into one list (note that
        // we do not add the "having" expression since that is not part of the projection)
        let mut aggr_projection_exprs = vec![];

        for expr in &group_by_exprs {
            match expr {
                // TODO 忽略前3种情况
                Expr::GroupingSet(GroupingSet::Rollup(exprs)) => {
                    aggr_projection_exprs.extend_from_slice(exprs)
                }
                Expr::GroupingSet(GroupingSet::Cube(exprs)) => {
                    aggr_projection_exprs.extend_from_slice(exprs)
                }
                Expr::GroupingSet(GroupingSet::GroupingSets(lists_of_exprs)) => {
                    for exprs in lists_of_exprs {
                        aggr_projection_exprs.extend_from_slice(exprs)
                    }
                }
                // 存入分组表达式
                _ => aggr_projection_exprs.push(expr.clone()),
            }
        }
        aggr_projection_exprs.extend_from_slice(&aggr_exprs);

        // now attempt to resolve columns and replace with fully-qualified columns
        // 解析column
        let aggr_projection_exprs = aggr_projection_exprs
            .iter()
            .map(|expr| resolve_columns(expr, &input))
            .collect::<Result<Vec<Expr>>>()?;

        // next we replace any expressions that are not a column with a column referencing
        // an output column from the aggregate schema
        let column_exprs_post_aggr = aggr_projection_exprs
            .iter()
            .map(|expr| expr_as_column_expr(expr, &input))
            .collect::<Result<Vec<Expr>>>()?;

        // next we re-write the projection  修改select的列 还是补充col的全限定名
        let select_exprs_post_aggr = select_exprs
            .iter()
            .map(|expr| rebase_expr(expr, &aggr_projection_exprs, &input))
            .collect::<Result<Vec<Expr>>>()?;

        // finally, we have some validation that the re-written projection can be resolved
        // from the aggregate output columns
        // 要求查询的列 必须在column_exprs_post_aggr中
        check_columns_satisfy_exprs(
            &column_exprs_post_aggr,
            &select_exprs_post_aggr,
            "Projection references non-aggregate values",
        )?;

        // Rewrite the HAVING expression to use the columns produced by the
        // aggregation.
        let having_expr_post_aggr = if let Some(having_expr) = having_expr_opt {
            let having_expr_post_aggr =
                rebase_expr(having_expr, &aggr_projection_exprs, &input)?;

            // 也是做检查
            check_columns_satisfy_exprs(
                &column_exprs_post_aggr,
                &[having_expr_post_aggr.clone()],
                "HAVING clause references non-aggregate values",
            )?;

            Some(having_expr_post_aggr)
        } else {
            None
        };

        Ok((plan, select_exprs_post_aggr, having_expr_post_aggr))
    }
}
