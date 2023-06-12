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

//! Aggregates functionalities

use crate::execution::context::TaskContext;
use crate::physical_plan::aggregates::no_grouping::AggregateStream;
use crate::physical_plan::metrics::{
    BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet,
};
use crate::physical_plan::{
    DisplayFormatType, Distribution, ExecutionPlan, Partitioning,
    SendableRecordBatchStream, Statistics,
};
use arrow::array::ArrayRef;
use arrow::datatypes::{Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion_common::{DataFusionError, Result};
use datafusion_expr::Accumulator;
use datafusion_physical_expr::expressions::{Avg, CastExpr, Column, Sum};
use datafusion_physical_expr::{
    expressions, AggregateExpr, PhysicalExpr, PhysicalSortExpr,
};
use std::any::Any;
use std::collections::HashMap;

use arrow::compute::DEFAULT_CAST_OPTIONS;
use std::sync::Arc;

mod no_grouping;
mod row_hash;

use crate::physical_plan::aggregates::row_hash::GroupedHashAggregateStream;
use crate::physical_plan::EquivalenceProperties;
pub use datafusion_expr::AggregateFunction;
use datafusion_physical_expr::aggregate::row_accumulator::RowAccumulator;
use datafusion_physical_expr::equivalence::project_equivalence_properties;
pub use datafusion_physical_expr::expressions::create_aggregate_expr;
use datafusion_physical_expr::normalize_out_expr_with_alias_schema;

/// Hash aggregate modes   数据聚合方式
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum AggregateMode {
    /// Partial aggregate that can be applied in parallel across input partitions
    Partial,
    /// Final aggregate that produces a single partition of output    代表聚合结果是单分区的
    Final,
    /// Final aggregate that works on pre-partitioned data.
    ///
    /// This requires the invariant that all rows with a particular
    /// grouping key are in the same partitions, such as is the case
    /// with Hash repartitioning on the group keys. If a group key is
    /// duplicated, duplicate groups would be produced
    FinalPartitioned,
    /// Applies the entire logical aggregation operation in a single operator,
    /// as opposed to Partial / Final modes which apply the logical aggregation using
    /// two operators.
    Single,
}

/// Represents `GROUP BY` clause in the plan (including the more general GROUPING SET)
/// In the case of a simple `GROUP BY a, b` clause, this will contain the expression [a, b]
/// and a single group [false, false].
/// In the case of `GROUP BY GROUPING SET/CUBE/ROLLUP` the planner will expand the expression
/// into multiple groups, using null expressions to align each group.
/// For example, with a group by clause `GROUP BY GROUPING SET ((a,b),(a),(b))` the planner should
/// create a `PhysicalGroupBy` like
/// PhysicalGroupBy {
///     expr: [(col(a), a), (col(b), b)],
///     null_expr: [(NULL, a), (NULL, b)],
///     groups: [
///         [false, false], // (a,b)
///         [false, true],  // (a) <=> (a, NULL)
///         [true, false]   // (b) <=> (NULL, b)
///     ]
/// }
#[derive(Clone, Debug, Default)]
pub struct PhysicalGroupBy {
    /// Distinct (Physical Expr, Alias) in the grouping set
    /// 分组相关的列 string代表别名
    expr: Vec<(Arc<dyn PhysicalExpr>, String)>,
    /// Corresponding NULL expressions for expr
    null_expr: Vec<(Arc<dyn PhysicalExpr>, String)>,
    /// Null mask for each group in this grouping set. Each group is
    /// composed of either one of the group expressions in expr or a null
    /// expression in null_expr. If `groups[i][j]` is true, then the the
    /// j-th expression in the i-th group is NULL, otherwise it is `expr[j]`.
    /// 这个bool 代表is_null
    groups: Vec<Vec<bool>>,
}

impl PhysicalGroupBy {
    /// Create a new `PhysicalGroupBy`
    pub fn new(
        expr: Vec<(Arc<dyn PhysicalExpr>, String)>,
        null_expr: Vec<(Arc<dyn PhysicalExpr>, String)>,
        groups: Vec<Vec<bool>>,
    ) -> Self {
        Self {
            expr,
            null_expr,
            groups,
        }
    }

    /// Create a GROUPING SET with only a single group. This is the "standard"
    /// case when building a plan from an expression such as `GROUP BY a,b,c`
    /// expr 对应分组使用的col  string对应expr名字
    pub fn new_single(expr: Vec<(Arc<dyn PhysicalExpr>, String)>) -> Self {
        let num_exprs = expr.len();
        Self {
            expr,
            null_expr: vec![],
            // 最外层内一个元素  内部创建跟expr等量的false
            groups: vec![vec![false; num_exprs]],
        }
    }

    /// Returns true if this GROUP BY contains NULL expressions
    /// groups中有任何true 就代表包含null值
    pub fn contains_null(&self) -> bool {
        self.groups.iter().flatten().any(|is_null| *is_null)
    }

    /// Returns the group expressions   返回分组表达式 (一般也就是column 代表基于该列进行分组)
    pub fn expr(&self) -> &[(Arc<dyn PhysicalExpr>, String)] {
        &self.expr
    }

    /// Returns the null expressions
    pub fn null_expr(&self) -> &[(Arc<dyn PhysicalExpr>, String)] {
        &self.null_expr
    }

    /// Returns the group null masks    返回标记为null的掩码
    pub fn groups(&self) -> &[Vec<bool>] {
        &self.groups
    }

    /// Returns true if this `PhysicalGroupBy` has no group expressions
    pub fn is_empty(&self) -> bool {
        self.expr.is_empty()
    }
}

impl PartialEq for PhysicalGroupBy {
    fn eq(&self, other: &PhysicalGroupBy) -> bool {
        self.expr.len() == other.expr.len()
            && self
                .expr
                .iter()
                .zip(other.expr.iter())
                .all(|((expr1, name1), (expr2, name2))| expr1.eq(expr2) && name1 == name2)
            && self.null_expr.len() == other.null_expr.len()
            && self
                .null_expr
                .iter()
                .zip(other.null_expr.iter())
                .all(|((expr1, name1), (expr2, name2))| expr1.eq(expr2) && name1 == name2)
            && self.groups == other.groups
    }
}

// 有不同的数据流类型
enum StreamType {
    AggregateStream(AggregateStream),
    GroupedHashAggregateStream(GroupedHashAggregateStream),
}

impl From<StreamType> for SendableRecordBatchStream {
    fn from(stream: StreamType) -> Self {
        match stream {
            StreamType::AggregateStream(stream) => Box::pin(stream),
            StreamType::GroupedHashAggregateStream(stream) => Box::pin(stream),
        }
    }
}

/// Hash aggregate execution plan   聚合执行计划
#[derive(Debug)]
pub struct AggregateExec {
    /// Aggregation mode (full, partial)    描述了聚合的模式
    pub(crate) mode: AggregateMode,
    /// Group by expressions          分组表达式 或者说如何分组
    pub(crate) group_by: PhysicalGroupBy,
    /// Aggregate expressions          聚合表达式
    pub(crate) aggr_expr: Vec<Arc<dyn AggregateExpr>>,
    /// FILTER (WHERE clause) expression for each aggregate expression   对数据集进行过滤  过滤器和聚合表达式还是一一对应的
    pub(crate) filter_expr: Vec<Option<Arc<dyn PhysicalExpr>>>,
    /// Input plan, could be a partial aggregate or the input to the aggregate    聚合也是一种包装 需要通过input拿到数据
    pub(crate) input: Arc<dyn ExecutionPlan>,
    /// Schema after the aggregate is applied
    /// 该schema中 只包含分组键 以及聚合会用到的field
    schema: SchemaRef,
    /// Input schema before any aggregation is applied. For partial aggregate this will be the
    /// same as input.schema() but for the final aggregate it will be the same as the input
    /// to the partial aggregate       聚合前数据的schema
    pub(crate) input_schema: SchemaRef,
    /// The alias map used to normalize out expressions like Partitioning and PhysicalSortExpr
    /// The key is the column from the input schema and the values are the columns from the output schema
    /// 原分组列  在新的schema下的名字和下标
    alias_map: HashMap<Column, Vec<Column>>,
    /// Execution Metrics    存储执行过程中的指标信息
    metrics: ExecutionPlanMetricsSet,
}

impl AggregateExec {

    /// Create a new hash aggregate execution plan
    pub fn try_new(
        mode: AggregateMode,
        group_by: PhysicalGroupBy,  // 内部包含分组相关的物理表达式
        aggr_expr: Vec<Arc<dyn AggregateExpr>>,   // 物理聚合表达式 内部有聚合逻辑
        filter_expr: Vec<Option<Arc<dyn PhysicalExpr>>>,  // 起到过滤作用
        input: Arc<dyn ExecutionPlan>,
        input_schema: SchemaRef,
    ) -> Result<Self> {

        // 根据分组用到的field 以及聚合函数的结果field 产生新的schema
        let schema = create_schema(
            &input.schema(),
            &group_by.expr,
            &aggr_expr,
            group_by.contains_null(),
            mode,
        )?;

        let schema = Arc::new(schema);

        // 存储别名信息  key是原始列  value在新的schema下有不同的名字和下标
        let mut alias_map: HashMap<Column, Vec<Column>> = HashMap::new();

        // 遍历参与分组的各个表达式
        for (expression, name) in group_by.expr.iter() {
            // 可以理解为分组相关的表达式都是Column
            if let Some(column) = expression.as_any().downcast_ref::<Column>() {

                // 找到col 在新的schema下的下标
                let new_col_idx = schema.index_of(name)?;
                // When the column name is the same, but index does not equal, treat it as Alias
                // 代表给col赋予了别名 (name即新名 column.name为旧名)   或者下标发生了变化
                if (column.name() != name) || (column.index() != new_col_idx) {
                    let entry = alias_map.entry(column.clone()).or_insert_with(Vec::new);
                    entry.push(Column::new(name, new_col_idx));
                }
            };
        }

        Ok(AggregateExec {
            mode,
            group_by,
            aggr_expr,
            filter_expr,
            input,
            schema,
            input_schema,
            alias_map,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }

    /// Aggregation mode (full, partial)
    pub fn mode(&self) -> &AggregateMode {
        &self.mode
    }

    /// Grouping expressions   返回分组相关的表达式
    pub fn group_expr(&self) -> &PhysicalGroupBy {
        &self.group_by
    }

    /// Grouping expressions as they occur in the output schema
    pub fn output_group_expr(&self) -> Vec<Arc<dyn PhysicalExpr>> {
        // Update column indices. Since the group by columns come first in the output schema, their
        // indices are simply 0..self.group_expr(len).
        self.group_by
            .expr()
            .iter()
            .enumerate()
            // 重新返回分组用的列  但是idx变小了
            .map(|(index, (_col, name))| {
                Arc::new(expressions::Column::new(name, index)) as Arc<dyn PhysicalExpr>
            })
            .collect()
    }

    /// Aggregate expressions
    pub fn aggr_expr(&self) -> &[Arc<dyn AggregateExpr>] {
        &self.aggr_expr
    }

    /// FILTER (WHERE clause) expression for each aggregate expression
    pub fn filter_expr(&self) -> &[Option<Arc<dyn PhysicalExpr>>] {
        &self.filter_expr
    }

    /// Input plan
    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }

    /// Get the input schema before any aggregates are applied
    pub fn input_schema(&self) -> SchemaRef {
        self.input_schema.clone()
    }

    // 调用execute时 会转发到该方法
    fn execute_typed(
        &self,
        partition: usize,   // 处理某个分区的数据
        context: Arc<TaskContext>,
    ) -> Result<StreamType> {
        // 单次拉取的记录上限
        let batch_size = context.session_config().batch_size();
        // 执行内部计划 获取结果集
        let input = self.input.execute(partition, Arc::clone(&context))?;
        // TODO
        let baseline_metrics = BaselineMetrics::new(&self.metrics, partition);

        // 根据是否有分组字段  产生2个不同的包装流
        if self.group_by.expr.is_empty() {
            // 不需要分组 直接对结果聚合
            Ok(StreamType::AggregateStream(AggregateStream::new(
                self.mode,
                self.schema.clone(),
                self.aggr_expr.clone(),
                self.filter_expr.clone(),
                input,
                baseline_metrics,
                context,
                partition,
            )?))
        } else {
            // 先对数据进行分组 之后再聚合
            Ok(StreamType::GroupedHashAggregateStream(
                GroupedHashAggregateStream::new(
                    self.mode,
                    self.schema.clone(),
                    self.group_by.clone(),  // 多传入一个分组表达式
                    self.aggr_expr.clone(),
                    self.filter_expr.clone(),
                    input,
                    baseline_metrics,
                    batch_size,
                    context,
                    partition,
                )?,
            ))
        }
    }
}

impl ExecutionPlan for AggregateExec {
    /// Return a reference to Any that can be used for down-casting
    fn as_any(&self) -> &dyn Any {
        self
    }

    /// 返回的schema是对应聚合后的
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    /// Get the output partitioning of this plan
    fn output_partitioning(&self) -> Partitioning {
        match &self.mode {
            // 为什么要区分这么多的聚合类型呢   就是为了在前面的阶段分区处理来提高性能
            // 但是实际上并没有改写分区方式呀
            AggregateMode::Partial | AggregateMode::Single => {
                // Partial and Single Aggregation will not change the output partitioning but need to respect the Alias
                let input_partition = self.input.output_partitioning();
                match input_partition {
                    // 采用hash方式
                    Partitioning::Hash(exprs, part) => {
                        // 这里要改写表达式(实际上就是改写column)
                        let normalized_exprs = exprs
                            .into_iter()
                            .map(|expr| {
                                normalize_out_expr_with_alias_schema(
                                    expr,
                                    &self.alias_map,  // 原来的列需要在这里做一层转换
                                    &self.schema,
                                )
                            })
                            .collect::<Vec<_>>();
                        Partitioning::Hash(normalized_exprs, part)
                    }
                    _ => input_partition,
                }
            }
            // Final Aggregation's output partitioning is the same as its real input  使用原来的分区方式
            _ => self.input.output_partitioning(),
        }
    }

    /// Specifies whether this plan generates an infinite stream of records.
    /// If the plan does not support pipelining, but it its input(s) are
    /// infinite, returns an error to indicate this.    
    fn unbounded_output(&self, children: &[bool]) -> Result<bool> {
        if children[0] {
            Err(DataFusionError::Plan(
                "Aggregate Error: `GROUP BY` clause (including the more general GROUPING SET) is not supported for unbounded inputs.".to_string(),
            ))
        } else {
            Ok(false)
        }
    }

    fn output_ordering(&self) -> Option<&[PhysicalSortExpr]> {
        None
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        match &self.mode {
            AggregateMode::Partial | AggregateMode::Single => {
                vec![Distribution::UnspecifiedDistribution]
            }
            AggregateMode::FinalPartitioned => {
                vec![Distribution::HashPartitioned(self.output_group_expr())]
            }
            AggregateMode::Final => vec![Distribution::SinglePartition],
        }
    }

    fn equivalence_properties(&self) -> EquivalenceProperties {
        // 产生一个基于新schema的空对象
        let mut new_properties = EquivalenceProperties::new(self.schema());

        // 主要就是做别名处理
        project_equivalence_properties(
            self.input.equivalence_properties(),
            &self.alias_map,
            &mut new_properties,
        );
        new_properties
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![self.input.clone()]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,  // 用本对象包裹给定的child
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(AggregateExec::try_new(
            self.mode,
            self.group_by.clone(),
            self.aggr_expr.clone(),
            self.filter_expr.clone(),
            children[0].clone(),
            self.input_schema.clone(),
        )?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        self.execute_typed(partition, context)
            .map(|stream| stream.into())
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn fmt_as(
        &self,
        t: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default => {
                write!(f, "AggregateExec: mode={:?}", self.mode)?;
                let g: Vec<String> = if self.group_by.groups.len() == 1 {
                    self.group_by
                        .expr
                        .iter()
                        .map(|(e, alias)| {
                            let e = e.to_string();
                            if &e != alias {
                                format!("{e} as {alias}")
                            } else {
                                e
                            }
                        })
                        .collect()
                } else {
                    self.group_by
                        .groups
                        .iter()
                        .map(|group| {
                            let terms = group
                                .iter()
                                .enumerate()
                                .map(|(idx, is_null)| {
                                    if *is_null {
                                        let (e, alias) = &self.group_by.null_expr[idx];
                                        let e = e.to_string();
                                        if &e != alias {
                                            format!("{e} as {alias}")
                                        } else {
                                            e
                                        }
                                    } else {
                                        let (e, alias) = &self.group_by.expr[idx];
                                        let e = e.to_string();
                                        if &e != alias {
                                            format!("{e} as {alias}")
                                        } else {
                                            e
                                        }
                                    }
                                })
                                .collect::<Vec<String>>()
                                .join(", ");
                            format!("({terms})")
                        })
                        .collect()
                };

                write!(f, ", gby=[{}]", g.join(", "))?;

                let a: Vec<String> = self
                    .aggr_expr
                    .iter()
                    .map(|agg| agg.name().to_string())
                    .collect();
                write!(f, ", aggr=[{}]", a.join(", "))?;
            }
        }
        Ok(())
    }

    fn statistics(&self) -> Statistics {
        // TODO stats: group expressions:
        // - once expressions will be able to compute their own stats, use it here
        // - case where we group by on a column for which with have the `distinct` stat
        // TODO stats: aggr expression:
        // - aggregations somtimes also preserve invariants such as min, max...
        match self.mode {
            AggregateMode::Final | AggregateMode::FinalPartitioned
                if self.group_by.expr.is_empty() =>
            {
                Statistics {
                    num_rows: Some(1),
                    is_exact: true,
                    ..Default::default()
                }
            }
            _ => Statistics {
                // the output row count is surely not larger than its input row count
                num_rows: self.input.statistics().num_rows,
                is_exact: false,
                ..Default::default()
            },
        }
    }
}

// 根据不同的情况  产生一个新的schema
fn create_schema(
    input_schema: &Schema,  // 原schema
    group_expr: &[(Arc<dyn PhysicalExpr>, String)],  // 分组相关的表达式
    aggr_expr: &[Arc<dyn AggregateExpr>],   // 聚合数据使用的表达式
    contains_null_expr: bool,   // 是否包含null值
    mode: AggregateMode,  // 聚合采用的模式
) -> Result<Schema> {

    let mut fields = Vec::with_capacity(group_expr.len() + aggr_expr.len());

    // 每个表达式 对应一个分组列
    for (expr, name) in group_expr {
        fields.push(Field::new(
            name,
            expr.data_type(input_schema)?,   // 该列必然来自于之前的schema 所以可以拿到类型
            // In cases where we have multiple grouping sets, we will use NULL expressions in
            // order to align the grouping sets. So the field must be nullable even if the underlying
            // schema field is not.
            contains_null_expr || expr.nullable(input_schema)?,
        ))
    }

    // 根据不同模式 扩充不同的field
    match mode {
        AggregateMode::Partial => {
            // in partial mode, the fields of the accumulator's state
            // 这种情况 存储的是中间状态的field
            for expr in aggr_expr {
                fields.extend(expr.state_fields()?.iter().cloned())
            }
        }
        AggregateMode::Final
        | AggregateMode::FinalPartitioned
        | AggregateMode::Single => {
            // in final mode, the field with the final result of the accumulator
            // 存储聚合函数产生的field
            for expr in aggr_expr {
                fields.push(expr.field()?)
            }
        }
    }

    Ok(Schema::new(fields))
}

fn group_schema(schema: &Schema, group_count: usize) -> SchemaRef {
    let group_fields = schema.fields()[0..group_count].to_vec();
    Arc::new(Schema::new(group_fields))
}

/// returns physical expressions to evaluate against a batch
/// The expressions are different depending on `mode`:
/// * Partial: AggregateExpr::expressions
/// * Final: columns of `AggregateExpr::state_fields()`
/// 不同的mode 以不同的方式使用这些聚合表达式
fn aggregate_expressions(
    aggr_expr: &[Arc<dyn AggregateExpr>],   // 一组聚合表达式
    mode: &AggregateMode,
    col_idx_base: usize,   // 应该是表示从第几列开始聚合
) -> Result<Vec<Vec<Arc<dyn PhysicalExpr>>>> {
    match mode {
        AggregateMode::Partial | AggregateMode::Single => Ok(aggr_expr
            .iter()
            .map(|agg| {
                // 遍历每个聚合表达式   并根据不同的类型走不同的逻辑

                // pre_cast_type 代表在执行表达式前 提前转换的目标数据类型
                let pre_cast_type = if let Some(Sum {
                    data_type,
                    pre_cast_to_sum_type,
                    ..
                }) = agg.as_any().downcast_ref::<Sum>()
                {
                    // 代表先转换成sum的目标类型
                    if *pre_cast_to_sum_type {
                        Some(data_type.clone())
                    } else {
                        None
                    }
                // 该聚合表达式是计算平均数
                } else if let Some(Avg {
                    sum_data_type,
                    pre_cast_to_sum_type,
                    ..
                }) = agg.as_any().downcast_ref::<Avg>()
                {
                    // 也是代表要提前转换
                    if *pre_cast_to_sum_type {
                        Some(sum_data_type.clone())
                    } else {
                        None
                    }
                } else {
                    None
                };

                // 给agg的子表达式 套上一层cast 使得数据在被聚合前先转换成跟结果一致的类型
                agg.expressions()
                    .into_iter()
                    .map(|expr| {
                        pre_cast_type.clone().map_or(expr.clone(), |cast_type| {
                            Arc::new(CastExpr::new(expr, cast_type, DEFAULT_CAST_OPTIONS))
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .collect()),
        // in this mode, we build the merge expressions of the aggregation
        AggregateMode::Final | AggregateMode::FinalPartitioned => {
            let mut col_idx_base = col_idx_base;
            // 这里将所有聚合表达式下的子字段展开
            Ok(aggr_expr
                .iter()
                .map(|agg| {
                    // 将聚合表达式的 状态字段展开 变成column
                    let exprs = merge_expressions(col_idx_base, agg)?;
                    col_idx_base += exprs.len();
                    Ok(exprs)
                })
                .collect::<Result<Vec<_>>>()?)
        }
    }
}

/// uses `state_fields` to build a vec of physical column expressions required to merge the
/// AggregateExpr' accumulator's state.
///
/// `index_base` is the starting physical column index for the next expanded state field.
/// 简单来看 就是铺开了聚合表达式内部的column
fn merge_expressions(
    index_base: usize,
    expr: &Arc<dyn AggregateExpr>,
) -> Result<Vec<Arc<dyn PhysicalExpr>>> {
    Ok(expr
        .state_fields()?    // state_field 是在聚合的中间过程需要记录的状态列
        .iter()
        .enumerate()
        .map(|(idx, f)| {
            Arc::new(Column::new(f.name(), index_base + idx)) as Arc<dyn PhysicalExpr>
        })
        .collect::<Vec<_>>())
}

pub(crate) type AccumulatorItem = Box<dyn Accumulator>;
pub(crate) type RowAccumulatorItem = Box<dyn RowAccumulator>;

// 产生表达式对应的累加器
fn create_accumulators(
    aggr_expr: &[Arc<dyn AggregateExpr>],
) -> Result<Vec<AccumulatorItem>> {
    aggr_expr
        .iter()
        .map(|expr| expr.create_accumulator())
        .collect::<Result<Vec<_>>>()
}

// 只有支持行格式的聚合表达式 会调用该方法
fn create_row_accumulators(
    aggr_expr: &[Arc<dyn AggregateExpr>],
) -> Result<Vec<RowAccumulatorItem>> {
    let mut state_index = 0;
    aggr_expr
        .iter()
        .map(|expr| {
            let result = expr.create_row_accumulator(state_index);
            state_index += expr.state_fields().unwrap().len();
            result
        })
        .collect::<Result<Vec<_>>>()
}

/// returns a vector of ArrayRefs, where each entry corresponds to either the
/// final value (mode = Final, FinalPartitioned and Single) or states (mode = Partial)
/// 提取出累加器内的数据
fn finalize_aggregation(
    accumulators: &[AccumulatorItem],  // 每种表达式对应一个聚合器
    mode: &AggregateMode,   // 不同的聚合模式 累加器的使用方法不同
) -> Result<Vec<ArrayRef>> {
    match mode {
        AggregateMode::Partial => {
            // build the vector of states
            let a = accumulators
                .iter()
                // 返回累加器得到的各种状态
                .map(|accumulator| accumulator.state())
                .map(|value| {
                    value.map(|e| {
                        // 将一个值转换成ArrayRef
                        e.iter().map(|v| v.to_array()).collect::<Vec<ArrayRef>>()
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            // 将每个累加器的各种状态平铺开
            Ok(a.iter().flatten().cloned().collect::<Vec<_>>())
        }
        AggregateMode::Final
        | AggregateMode::FinalPartitioned
        | AggregateMode::Single => {
            // merge the state to the final value
            accumulators
                .iter()  // evaluate只会产生一个值  就是最有意义的值  state可能是多个值 参照sum
                .map(|accumulator| accumulator.evaluate().map(|v| v.to_array()))
                .collect::<Result<Vec<ArrayRef>>>()
        }
    }
}

/// Evaluates expressions against a record batch.
fn evaluate(
    expr: &[Arc<dyn PhysicalExpr>],
    batch: &RecordBatch,
) -> Result<Vec<ArrayRef>> {
    expr.iter()
        .map(|expr| expr.evaluate(batch))
        .map(|r| r.map(|v| v.into_array(batch.num_rows())))
        .collect::<Result<Vec<_>>>()
}

/// Evaluates expressions against a record batch.
fn evaluate_many(
    expr: &[Vec<Arc<dyn PhysicalExpr>>],
    batch: &RecordBatch,
) -> Result<Vec<Vec<ArrayRef>>> {
    expr.iter()
        .map(|expr| evaluate(expr, batch))
        .collect::<Result<Vec<_>>>()
}

fn evaluate_optional(
    expr: &[Option<Arc<dyn PhysicalExpr>>],
    batch: &RecordBatch,
) -> Result<Vec<Option<ArrayRef>>> {
    expr.iter()
        .map(|expr| {
            expr.as_ref()
                .map(|expr| expr.evaluate(batch))
                .transpose()
                .map(|r| r.map(|v| v.into_array(batch.num_rows())))
        })
        .collect::<Result<Vec<_>>>()
}

// 将数据集 按照分组列进行分组
fn evaluate_group_by(
    group_by: &PhysicalGroupBy,
    batch: &RecordBatch,
) -> Result<Vec<Vec<ArrayRef>>> {

    // 取出分组相关列的值
    let exprs: Vec<ArrayRef> = group_by
        .expr
        .iter()
        .map(|(expr, _)| {
            let value = expr.evaluate(batch)?;
            Ok(value.into_array(batch.num_rows()))
        })
        .collect::<Result<Vec<_>>>()?;

    // 默认为空 反正也是取出某几列的值
    let null_exprs: Vec<ArrayRef> = group_by
        .null_expr
        .iter()
        .map(|(expr, _)| {
            let value = expr.evaluate(batch)?;
            Ok(value.into_array(batch.num_rows()))
        })
        .collect::<Result<Vec<_>>>()?;

    // 根据isnull提示  选择不同的列值
    Ok(group_by
        .groups
        .iter()
        .map(|group| {
            group
                .iter()
                .enumerate()
                .map(|(idx, is_null)| {
                    if *is_null {
                        null_exprs[idx].clone()
                    } else {
                        exprs[idx].clone()
                    }
                })
                .collect()
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use crate::execution::context::{SessionConfig, TaskContext};
    use crate::execution::runtime_env::{RuntimeConfig, RuntimeEnv};
    use crate::from_slice::FromSlice;
    use crate::physical_plan::aggregates::{
        AggregateExec, AggregateMode, PhysicalGroupBy,
    };
    use crate::physical_plan::expressions::{col, Avg};
    use crate::test::assert_is_pending;
    use crate::test::exec::{assert_strong_count_converges_to_zero, BlockingExec};
    use crate::{assert_batches_sorted_eq, physical_plan::common};
    use arrow::array::{Float64Array, UInt32Array};
    use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
    use arrow::record_batch::RecordBatch;
    use datafusion_common::{DataFusionError, Result, ScalarValue};
    use datafusion_physical_expr::expressions::{lit, ApproxDistinct, Count, Median};
    use datafusion_physical_expr::{AggregateExpr, PhysicalExpr, PhysicalSortExpr};
    use futures::{FutureExt, Stream};
    use std::any::Any;
    use std::sync::Arc;
    use std::task::{Context, Poll};

    use super::StreamType;
    use crate::physical_plan::coalesce_partitions::CoalescePartitionsExec;
    use crate::physical_plan::{
        ExecutionPlan, Partitioning, RecordBatchStream, SendableRecordBatchStream,
        Statistics,
    };
    use crate::prelude::SessionContext;

    /// some mock data to aggregates
    fn some_data() -> (Arc<Schema>, Vec<RecordBatch>) {
        // define a schema.
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::UInt32, false),
            Field::new("b", DataType::Float64, false),
        ]));

        // define data.
        (
            schema.clone(),
            vec![
                RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        Arc::new(UInt32Array::from_slice([2, 3, 4, 4])),
                        Arc::new(Float64Array::from_slice([1.0, 2.0, 3.0, 4.0])),
                    ],
                )
                .unwrap(),
                RecordBatch::try_new(
                    schema,
                    vec![
                        Arc::new(UInt32Array::from_slice([2, 3, 3, 4])),
                        Arc::new(Float64Array::from_slice([1.0, 2.0, 3.0, 4.0])),
                    ],
                )
                .unwrap(),
            ],
        )
    }

    async fn check_grouping_sets(input: Arc<dyn ExecutionPlan>) -> Result<()> {
        let input_schema = input.schema();

        let grouping_set = PhysicalGroupBy {
            expr: vec![
                (col("a", &input_schema)?, "a".to_string()),
                (col("b", &input_schema)?, "b".to_string()),
            ],
            null_expr: vec![
                (lit(ScalarValue::UInt32(None)), "a".to_string()),
                (lit(ScalarValue::Float64(None)), "b".to_string()),
            ],
            groups: vec![
                vec![false, true],  // (a, NULL)
                vec![true, false],  // (NULL, b)
                vec![false, false], // (a,b)
            ],
        };

        let aggregates: Vec<Arc<dyn AggregateExpr>> = vec![Arc::new(Count::new(
            lit(1i8),
            "COUNT(1)".to_string(),
            DataType::Int64,
        ))];

        let session_ctx = SessionContext::new();
        let task_ctx = session_ctx.task_ctx();

        let partial_aggregate = Arc::new(AggregateExec::try_new(
            AggregateMode::Partial,
            grouping_set.clone(),
            aggregates.clone(),
            vec![None],
            input,
            input_schema.clone(),
        )?);

        let result =
            common::collect(partial_aggregate.execute(0, task_ctx.clone())?).await?;

        let expected = vec![
            "+---+-----+-----------------+",
            "| a | b   | COUNT(1)[count] |",
            "+---+-----+-----------------+",
            "|   | 1.0 | 2               |",
            "|   | 2.0 | 2               |",
            "|   | 3.0 | 2               |",
            "|   | 4.0 | 2               |",
            "| 2 |     | 2               |",
            "| 2 | 1.0 | 2               |",
            "| 3 |     | 3               |",
            "| 3 | 2.0 | 2               |",
            "| 3 | 3.0 | 1               |",
            "| 4 |     | 3               |",
            "| 4 | 3.0 | 1               |",
            "| 4 | 4.0 | 2               |",
            "+---+-----+-----------------+",
        ];
        assert_batches_sorted_eq!(expected, &result);

        let groups = partial_aggregate.group_expr().expr().to_vec();

        let merge = Arc::new(CoalescePartitionsExec::new(partial_aggregate));

        let final_group: Vec<(Arc<dyn PhysicalExpr>, String)> = groups
            .iter()
            .map(|(_expr, name)| Ok((col(name, &input_schema)?, name.clone())))
            .collect::<Result<_>>()?;

        let final_grouping_set = PhysicalGroupBy::new_single(final_group);

        let merged_aggregate = Arc::new(AggregateExec::try_new(
            AggregateMode::Final,
            final_grouping_set,
            aggregates,
            vec![None],
            merge,
            input_schema,
        )?);

        let result =
            common::collect(merged_aggregate.execute(0, task_ctx.clone())?).await?;
        assert_eq!(result.len(), 1);

        let batch = &result[0];
        assert_eq!(batch.num_columns(), 3);
        assert_eq!(batch.num_rows(), 12);

        let expected = vec![
            "+---+-----+----------+",
            "| a | b   | COUNT(1) |",
            "+---+-----+----------+",
            "|   | 1.0 | 2        |",
            "|   | 2.0 | 2        |",
            "|   | 3.0 | 2        |",
            "|   | 4.0 | 2        |",
            "| 2 |     | 2        |",
            "| 2 | 1.0 | 2        |",
            "| 3 |     | 3        |",
            "| 3 | 2.0 | 2        |",
            "| 3 | 3.0 | 1        |",
            "| 4 |     | 3        |",
            "| 4 | 3.0 | 1        |",
            "| 4 | 4.0 | 2        |",
            "+---+-----+----------+",
        ];

        assert_batches_sorted_eq!(&expected, &result);

        let metrics = merged_aggregate.metrics().unwrap();
        let output_rows = metrics.output_rows().unwrap();
        assert_eq!(12, output_rows);

        Ok(())
    }

    /// build the aggregates on the data from some_data() and check the results
    async fn check_aggregates(input: Arc<dyn ExecutionPlan>) -> Result<()> {
        let input_schema = input.schema();

        let grouping_set = PhysicalGroupBy {
            expr: vec![(col("a", &input_schema)?, "a".to_string())],
            null_expr: vec![],
            groups: vec![vec![false]],
        };

        let aggregates: Vec<Arc<dyn AggregateExpr>> = vec![Arc::new(Avg::new(
            col("b", &input_schema)?,
            "AVG(b)".to_string(),
            DataType::Float64,
        ))];

        let session_ctx = SessionContext::new();
        let task_ctx = session_ctx.task_ctx();

        let partial_aggregate = Arc::new(AggregateExec::try_new(
            AggregateMode::Partial,
            grouping_set.clone(),
            aggregates.clone(),
            vec![None],
            input,
            input_schema.clone(),
        )?);

        let result =
            common::collect(partial_aggregate.execute(0, task_ctx.clone())?).await?;

        let expected = vec![
            "+---+---------------+-------------+",
            "| a | AVG(b)[count] | AVG(b)[sum] |",
            "+---+---------------+-------------+",
            "| 2 | 2             | 2.0         |",
            "| 3 | 3             | 7.0         |",
            "| 4 | 3             | 11.0        |",
            "+---+---------------+-------------+",
        ];
        assert_batches_sorted_eq!(expected, &result);

        let merge = Arc::new(CoalescePartitionsExec::new(partial_aggregate));

        let final_group: Vec<(Arc<dyn PhysicalExpr>, String)> = grouping_set
            .expr
            .iter()
            .map(|(_expr, name)| Ok((col(name, &input_schema)?, name.clone())))
            .collect::<Result<_>>()?;

        let final_grouping_set = PhysicalGroupBy::new_single(final_group);

        let merged_aggregate = Arc::new(AggregateExec::try_new(
            AggregateMode::Final,
            final_grouping_set,
            aggregates,
            vec![None],
            merge,
            input_schema,
        )?);

        let result =
            common::collect(merged_aggregate.execute(0, task_ctx.clone())?).await?;
        assert_eq!(result.len(), 1);

        let batch = &result[0];
        assert_eq!(batch.num_columns(), 2);
        assert_eq!(batch.num_rows(), 3);

        let expected = vec![
            "+---+--------------------+",
            "| a | AVG(b)             |",
            "+---+--------------------+",
            "| 2 | 1.0                |",
            "| 3 | 2.3333333333333335 |", // 3, (2 + 3 + 2) / 3
            "| 4 | 3.6666666666666665 |", // 4, (3 + 4 + 4) / 3
            "+---+--------------------+",
        ];

        assert_batches_sorted_eq!(&expected, &result);

        let metrics = merged_aggregate.metrics().unwrap();
        let output_rows = metrics.output_rows().unwrap();
        assert_eq!(3, output_rows);

        Ok(())
    }

    /// Define a test source that can yield back to runtime before returning its first item ///

    #[derive(Debug)]
    struct TestYieldingExec {
        /// True if this exec should yield back to runtime the first time it is polled
        pub yield_first: bool,
    }

    impl ExecutionPlan for TestYieldingExec {
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn schema(&self) -> SchemaRef {
            some_data().0
        }

        fn output_partitioning(&self) -> Partitioning {
            Partitioning::UnknownPartitioning(1)
        }

        fn output_ordering(&self) -> Option<&[PhysicalSortExpr]> {
            None
        }

        fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
            vec![]
        }

        fn with_new_children(
            self: Arc<Self>,
            _: Vec<Arc<dyn ExecutionPlan>>,
        ) -> Result<Arc<dyn ExecutionPlan>> {
            Err(DataFusionError::Internal(format!(
                "Children cannot be replaced in {self:?}"
            )))
        }

        fn execute(
            &self,
            _partition: usize,
            _context: Arc<TaskContext>,
        ) -> Result<SendableRecordBatchStream> {
            let stream = if self.yield_first {
                TestYieldingStream::New
            } else {
                TestYieldingStream::Yielded
            };

            Ok(Box::pin(stream))
        }

        fn statistics(&self) -> Statistics {
            let (_, batches) = some_data();
            common::compute_record_batch_statistics(&[batches], &self.schema(), None)
        }
    }

    /// A stream using the demo data. If inited as new, it will first yield to runtime before returning records
    enum TestYieldingStream {
        New,
        Yielded,
        ReturnedBatch1,
        ReturnedBatch2,
    }

    impl Stream for TestYieldingStream {
        type Item = Result<RecordBatch>;

        fn poll_next(
            mut self: std::pin::Pin<&mut Self>,
            cx: &mut Context<'_>,
        ) -> Poll<Option<Self::Item>> {
            match &*self {
                TestYieldingStream::New => {
                    *(self.as_mut()) = TestYieldingStream::Yielded;
                    cx.waker().wake_by_ref();
                    Poll::Pending
                }
                TestYieldingStream::Yielded => {
                    *(self.as_mut()) = TestYieldingStream::ReturnedBatch1;
                    Poll::Ready(Some(Ok(some_data().1[0].clone())))
                }
                TestYieldingStream::ReturnedBatch1 => {
                    *(self.as_mut()) = TestYieldingStream::ReturnedBatch2;
                    Poll::Ready(Some(Ok(some_data().1[1].clone())))
                }
                TestYieldingStream::ReturnedBatch2 => Poll::Ready(None),
            }
        }
    }

    impl RecordBatchStream for TestYieldingStream {
        fn schema(&self) -> SchemaRef {
            some_data().0
        }
    }

    //// Tests ////

    #[tokio::test]
    async fn aggregate_source_not_yielding() -> Result<()> {
        let input: Arc<dyn ExecutionPlan> =
            Arc::new(TestYieldingExec { yield_first: false });

        check_aggregates(input).await
    }

    #[tokio::test]
    async fn aggregate_grouping_sets_source_not_yielding() -> Result<()> {
        let input: Arc<dyn ExecutionPlan> =
            Arc::new(TestYieldingExec { yield_first: false });

        check_grouping_sets(input).await
    }

    #[tokio::test]
    async fn aggregate_source_with_yielding() -> Result<()> {
        let input: Arc<dyn ExecutionPlan> =
            Arc::new(TestYieldingExec { yield_first: true });

        check_aggregates(input).await
    }

    #[tokio::test]
    async fn aggregate_grouping_sets_with_yielding() -> Result<()> {
        let input: Arc<dyn ExecutionPlan> =
            Arc::new(TestYieldingExec { yield_first: true });

        check_grouping_sets(input).await
    }

    #[tokio::test]
    async fn test_oom() -> Result<()> {
        let input: Arc<dyn ExecutionPlan> =
            Arc::new(TestYieldingExec { yield_first: true });
        let input_schema = input.schema();

        let session_ctx = SessionContext::with_config_rt(
            SessionConfig::default(),
            Arc::new(
                RuntimeEnv::new(RuntimeConfig::default().with_memory_limit(1, 1.0))
                    .unwrap(),
            ),
        );
        let task_ctx = session_ctx.task_ctx();

        let groups_none = PhysicalGroupBy::default();
        let groups_some = PhysicalGroupBy {
            expr: vec![(col("a", &input_schema)?, "a".to_string())],
            null_expr: vec![],
            groups: vec![vec![false]],
        };

        // something that allocates within the aggregator
        let aggregates_v0: Vec<Arc<dyn AggregateExpr>> = vec![Arc::new(Median::new(
            col("a", &input_schema)?,
            "MEDIAN(a)".to_string(),
            DataType::UInt32,
        ))];

        // use slow-path in `hash.rs`
        let aggregates_v1: Vec<Arc<dyn AggregateExpr>> =
            vec![Arc::new(ApproxDistinct::new(
                col("a", &input_schema)?,
                "APPROX_DISTINCT(a)".to_string(),
                DataType::UInt32,
            ))];

        // use fast-path in `row_hash.rs`.
        let aggregates_v2: Vec<Arc<dyn AggregateExpr>> = vec![Arc::new(Avg::new(
            col("b", &input_schema)?,
            "AVG(b)".to_string(),
            DataType::Float64,
        ))];

        for (version, groups, aggregates) in [
            (0, groups_none, aggregates_v0),
            (1, groups_some.clone(), aggregates_v1),
            (2, groups_some, aggregates_v2),
        ] {
            let partial_aggregate = Arc::new(AggregateExec::try_new(
                AggregateMode::Partial,
                groups,
                aggregates,
                vec![None; 3],
                input.clone(),
                input_schema.clone(),
            )?);

            let stream = partial_aggregate.execute_typed(0, task_ctx.clone())?;

            // ensure that we really got the version we wanted
            match version {
                0 => {
                    assert!(matches!(stream, StreamType::AggregateStream(_)));
                }
                1 => {
                    assert!(matches!(stream, StreamType::GroupedHashAggregateStream(_)));
                }
                2 => {
                    assert!(matches!(stream, StreamType::GroupedHashAggregateStream(_)));
                }
                _ => panic!("Unknown version: {version}"),
            }

            let stream: SendableRecordBatchStream = stream.into();
            let err = common::collect(stream).await.unwrap_err();

            // error root cause traversal is a bit complicated, see #4172.
            let err = err.find_root();
            assert!(
                matches!(err, DataFusionError::ResourcesExhausted(_)),
                "Wrong error type: {err}",
            );
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_drop_cancel_without_groups() -> Result<()> {
        let session_ctx = SessionContext::new();
        let task_ctx = session_ctx.task_ctx();
        let schema =
            Arc::new(Schema::new(vec![Field::new("a", DataType::Float32, true)]));

        let groups = PhysicalGroupBy::default();

        let aggregates: Vec<Arc<dyn AggregateExpr>> = vec![Arc::new(Avg::new(
            col("a", &schema)?,
            "AVG(a)".to_string(),
            DataType::Float64,
        ))];

        let blocking_exec = Arc::new(BlockingExec::new(Arc::clone(&schema), 1));
        let refs = blocking_exec.refs();
        let aggregate_exec = Arc::new(AggregateExec::try_new(
            AggregateMode::Partial,
            groups.clone(),
            aggregates.clone(),
            vec![None],
            blocking_exec,
            schema,
        )?);

        let fut = crate::physical_plan::collect(aggregate_exec, task_ctx);
        let mut fut = fut.boxed();

        assert_is_pending(&mut fut);
        drop(fut);
        assert_strong_count_converges_to_zero(refs).await;

        Ok(())
    }

    #[tokio::test]
    async fn test_drop_cancel_with_groups() -> Result<()> {
        let session_ctx = SessionContext::new();
        let task_ctx = session_ctx.task_ctx();
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Float32, true),
            Field::new("b", DataType::Float32, true),
        ]));

        let groups =
            PhysicalGroupBy::new_single(vec![(col("a", &schema)?, "a".to_string())]);

        let aggregates: Vec<Arc<dyn AggregateExpr>> = vec![Arc::new(Avg::new(
            col("b", &schema)?,
            "AVG(b)".to_string(),
            DataType::Float64,
        ))];

        let blocking_exec = Arc::new(BlockingExec::new(Arc::clone(&schema), 1));
        let refs = blocking_exec.refs();
        let aggregate_exec = Arc::new(AggregateExec::try_new(
            AggregateMode::Partial,
            groups,
            aggregates.clone(),
            vec![None],
            blocking_exec,
            schema,
        )?);

        let fut = crate::physical_plan::collect(aggregate_exec, task_ctx);
        let mut fut = fut.boxed();

        assert_is_pending(&mut fut);
        drop(fut);
        assert_strong_count_converges_to_zero(refs).await;

        Ok(())
    }
}
