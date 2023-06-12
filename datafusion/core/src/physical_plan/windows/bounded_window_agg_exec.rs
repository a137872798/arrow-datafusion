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

//! Stream and channel implementations for window function expressions.
//! The executor given here uses bounded memory (does not maintain all
//! the input data seen so far), which makes it appropriate when processing
//! infinite inputs.

use crate::error::Result;
use crate::execution::context::TaskContext;
use crate::physical_plan::expressions::PhysicalSortExpr;
use crate::physical_plan::metrics::{
    BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet,
};
use crate::physical_plan::{
    ColumnStatistics, DisplayFormatType, Distribution, ExecutionPlan, Partitioning,
    RecordBatchStream, SendableRecordBatchStream, Statistics, WindowExpr,
};
use arrow::{
    array::{Array, ArrayRef},
    compute::{concat, concat_batches, SortColumn},
    datatypes::{Schema, SchemaBuilder, SchemaRef},
    record_batch::RecordBatch,
};
use datafusion_common::{DataFusionError, ScalarValue};
use futures::stream::Stream;
use futures::{ready, StreamExt};
use std::any::Any;
use std::cmp::min;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use crate::physical_plan::windows::{
    calc_requirements, get_ordered_partition_by_indices,
};
use datafusion_common::utils::{evaluate_partition_ranges, get_at_indices};
use datafusion_physical_expr::window::{
    PartitionBatchState, PartitionBatches, PartitionKey, PartitionWindowAggStates,
    WindowAggState, WindowState,
};
use datafusion_physical_expr::{
    EquivalenceProperties, PhysicalExpr, PhysicalSortRequirement,
};
use indexmap::IndexMap;
use log::debug;

/// Window execution plan
/// 代表该窗口函数使用的内存可控
#[derive(Debug)]
pub struct BoundedWindowAggExec {
    /// Input plan
    input: Arc<dyn ExecutionPlan>,
    /// Window function expression    使用同一排序键的一组窗口表达式
    window_expr: Vec<Arc<dyn WindowExpr>>,
    /// Schema after the window is run
    schema: SchemaRef,
    /// Schema before the window
    input_schema: SchemaRef,
    /// Partition Keys   使用的分区键
    pub partition_keys: Vec<Arc<dyn PhysicalExpr>>,
    /// Execution metrics
    metrics: ExecutionPlanMetricsSet,
    /// Partition by indices that defines preset for existing ordering
    // see `get_ordered_partition_by_indices` for more details.
    ordered_partition_by_indices: Vec<usize>,
}

impl BoundedWindowAggExec {
    /// Create a new execution plan for window aggregates
    pub fn try_new(
        window_expr: Vec<Arc<dyn WindowExpr>>,
        input: Arc<dyn ExecutionPlan>,
        input_schema: SchemaRef,
        partition_keys: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Result<Self> {
        let schema = create_schema(&input_schema, &window_expr)?;
        let schema = Arc::new(schema);

        // 内部plan的排序键 映射到外部window分区键的下标
        let ordered_partition_by_indices =
            get_ordered_partition_by_indices(window_expr[0].partition_by(), &input);
        Ok(Self {
            input,
            window_expr,
            schema,
            input_schema,
            partition_keys,
            metrics: ExecutionPlanMetricsSet::new(),
            ordered_partition_by_indices,
        })
    }

    /// Window expressions
    pub fn window_expr(&self) -> &[Arc<dyn WindowExpr>] {
        &self.window_expr
    }

    /// Input plan
    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }

    /// Get the input schema before any window functions are applied
    pub fn input_schema(&self) -> SchemaRef {
        self.input_schema.clone()
    }

    /// Return the output sort order of partition keys: For example
    /// OVER(PARTITION BY a, ORDER BY b) -> would give sorting of the column a
    // We are sure that partition by columns are always at the beginning of sort_keys
    // Hence returned `PhysicalSortExpr` corresponding to `PARTITION BY` columns can be used safely
    // to calculate partition separation points
    // 获得window的排序键
    pub fn partition_by_sort_keys(&self) -> Result<Vec<PhysicalSortExpr>> {
        // Partition by sort keys indices are stored in self.ordered_partition_by_indices.
        let sort_keys = self.input.output_ordering().unwrap_or(&[]);
        get_at_indices(sort_keys, &self.ordered_partition_by_indices)
    }
}

impl ExecutionPlan for BoundedWindowAggExec {
    /// Return a reference to Any that can be used for downcasting
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![self.input.clone()]
    }

    /// Get the output partitioning of this plan
    fn output_partitioning(&self) -> Partitioning {
        // As we can have repartitioning using the partition keys, this can
        // be either one or more than one, depending on the presence of
        // repartitioning.
        self.input.output_partitioning()
    }

    fn unbounded_output(&self, children: &[bool]) -> Result<bool> {
        Ok(children[0])
    }

    fn output_ordering(&self) -> Option<&[PhysicalSortExpr]> {
        self.input().output_ordering()
    }

    fn required_input_ordering(&self) -> Vec<Option<Vec<PhysicalSortRequirement>>> {
        let partition_bys = self.window_expr()[0].partition_by();
        let order_keys = self.window_expr()[0].order_by();
        if self.ordered_partition_by_indices.len() < partition_bys.len() {
            vec![calc_requirements(partition_bys, order_keys)]
        } else {
            let partition_bys = self
                .ordered_partition_by_indices
                .iter()
                .map(|idx| &partition_bys[*idx]);
            vec![calc_requirements(partition_bys, order_keys)]
        }
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        if self.partition_keys.is_empty() {
            debug!("No partition defined for BoundedWindowAggExec!!!");
            vec![Distribution::SinglePartition]
        } else {
            vec![Distribution::HashPartitioned(self.partition_keys.clone())]
        }
    }

    fn equivalence_properties(&self) -> EquivalenceProperties {
        self.input().equivalence_properties()
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(BoundedWindowAggExec::try_new(
            self.window_expr.clone(),
            children[0].clone(),
            self.input_schema.clone(),
            self.partition_keys.clone(),
        )?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let input = self.input.execute(partition, context)?;
        let stream = Box::pin(SortedPartitionByBoundedWindowStream::new(
            self.schema.clone(),
            self.window_expr.clone(),
            input,
            BaselineMetrics::new(&self.metrics, partition),
            self.partition_by_sort_keys()?,
            self.ordered_partition_by_indices.clone(),
        )?);
        Ok(stream)
    }

    fn fmt_as(
        &self,
        t: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default => {
                write!(f, "BoundedWindowAggExec: ")?;
                let g: Vec<String> = self
                    .window_expr
                    .iter()
                    .map(|e| {
                        format!(
                            "{}: {:?}, frame: {:?}",
                            e.name().to_owned(),
                            e.field(),
                            e.get_window_frame()
                        )
                    })
                    .collect();
                write!(f, "wdw=[{}]", g.join(", "))?;
            }
        }
        Ok(())
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn statistics(&self) -> Statistics {
        let input_stat = self.input.statistics();
        let win_cols = self.window_expr.len();
        let input_cols = self.input_schema.fields().len();
        // TODO stats: some windowing function will maintain invariants such as min, max...
        let mut column_statistics = Vec::with_capacity(win_cols + input_cols);
        if let Some(input_col_stats) = input_stat.column_statistics {
            column_statistics.extend(input_col_stats);
        } else {
            column_statistics.extend(vec![ColumnStatistics::default(); input_cols]);
        }
        column_statistics.extend(vec![ColumnStatistics::default(); win_cols]);
        Statistics {
            is_exact: input_stat.is_exact,
            num_rows: input_stat.num_rows,
            column_statistics: Some(column_statistics),
            total_byte_size: None,
        }
    }
}

// 在原有schema基础上 追加window产生的field
fn create_schema(
    input_schema: &Schema,
    window_expr: &[Arc<dyn WindowExpr>],
) -> Result<Schema> {
    let capacity = input_schema.fields().len() + window_expr.len();
    let mut builder = SchemaBuilder::with_capacity(capacity);
    builder.extend(input_schema.fields.iter().cloned());
    // append results to the schema
    for expr in window_expr {
        builder.push(expr.field()?);
    }
    Ok(builder.finish())
}

/// This trait defines the interface for updating the state and calculating
/// results for window functions. Depending on the partitioning scheme, one
/// may have different implementations for the functions within.
pub trait PartitionByHandler {
    /// Constructs output columns from window_expression results.  通过窗口表达式计算结果
    fn calculate_out_columns(&self) -> Result<Option<Vec<ArrayRef>>>;
    /// Prunes the window state to remove any unnecessary information
    /// given how many rows we emitted so far.   数据是有限的 就体现在要清理之前数据产生的影响
    fn prune_state(&mut self, n_out: usize) -> Result<()>;
    /// Updates record batches for each partition when new batches are
    /// received.  处理新的数据集
    fn update_partition_batch(&mut self, record_batch: RecordBatch) -> Result<()>;
}

/// stream for window aggregation plan
/// assuming partition by column is sorted (or without PARTITION BY expression)
pub struct SortedPartitionByBoundedWindowStream {
    schema: SchemaRef,
    input: SendableRecordBatchStream,
    /// The record batch executor receives as input (i.e. the columns needed
    /// while calculating aggregation results).
    input_buffer: RecordBatch,
    /// We separate `input_buffer_record_batch` based on partitions (as
    /// determined by PARTITION BY columns) and store them per partition
    /// in `partition_batches`. We use this variable when calculating results
    /// for each window expression. This enables us to use the same batch for
    /// different window expressions without copying.
    // Note that we could keep record batches for each window expression in
    // `PartitionWindowAggStates`. However, this would use more memory (as
    // many times as the number of window expressions).
    partition_buffers: PartitionBatches,
    /// An executor can run multiple window expressions if the PARTITION BY
    /// and ORDER BY sections are same. We keep state of the each window
    /// expression inside `window_agg_states`.
    window_agg_states: Vec<PartitionWindowAggStates>,
    finished: bool,
    window_expr: Vec<Arc<dyn WindowExpr>>,
    partition_by_sort_keys: Vec<PhysicalSortExpr>,
    baseline_metrics: BaselineMetrics,
    ordered_partition_by_indices: Vec<usize>,
}

impl PartitionByHandler for SortedPartitionByBoundedWindowStream {
    /// This method constructs output columns using the result of each window expression
    /// 将state转换成输出结果
    fn calculate_out_columns(&self) -> Result<Option<Vec<ArrayRef>>> {

        // 统计产生了多少条记录
        let n_out = self.calculate_n_out_row();
        if n_out == 0 {
            Ok(None)
        } else {
            self.input_buffer
                .columns()
                .iter()
                .map(|elem| Ok(elem.slice(0, n_out)))
                .chain(
                    // 将每个窗口函数表达式的结果追加到 原来的结果集上
                    self.window_agg_states
                        .iter()
                        .map(|elem| get_aggregate_result_out_column(elem, n_out)),
                )
                .collect::<Result<Vec<_>>>()
                .map(Some)
        }
    }

    /// Prunes sections of the state that are no longer needed when calculating
    /// results (as determined by window frame boundaries and number of results generated).
    // For instance, if first `n` (not necessarily same with `n_out`) elements are no longer needed to
    // calculate window expression result (outside the window frame boundary) we retract first `n` elements
    // from `self.partition_batches` in corresponding partition.
    // For instance, if `n_out` number of rows are calculated, we can remove
    // first `n_out` rows from `self.input_buffer_record_batch`.
    // 每当将多少条记录合并出的结果返回后 要消除前面对应数量记录的影响  体现了滑动也确保内存开销是空控的
    fn prune_state(&mut self, n_out: usize) -> Result<()> {
        // Prune `self.window_agg_states`:   减小out_col
        self.prune_out_columns(n_out)?;
        // Prune `self.partition_batches`:   去掉不需要的数据集
        self.prune_partition_batches()?;
        // Prune `self.input_buffer_record_batch`:   去掉输入集
        self.prune_input_batch(n_out)?;
        Ok(())
    }

    // 拉取到一组数据后触发该方法
    fn update_partition_batch(&mut self, record_batch: RecordBatch) -> Result<()> {
        // 对应内部plan的排序键
        let partition_columns = self.partition_columns(&record_batch)?;
        let num_rows = record_batch.num_rows();
        if num_rows > 0 {

            // 产生多个分区
            let partition_points =
                evaluate_partition_ranges(num_rows, &partition_columns)?;

            // 按分区处理数据
            for partition_range in partition_points {
                // 拿到该分区第一行的分区键数据   既然能划分到同一分区 该分区下分区键的值应该是一样的
                let partition_row = partition_columns
                    .iter()
                    .map(|arr| {
                        ScalarValue::try_from_array(&arr.values, partition_range.start)
                    })
                    .collect::<Result<PartitionKey>>()?;

                // 得到该分区的数据集
                let partition_batch = record_batch.slice(
                    partition_range.start,
                    partition_range.end - partition_range.start,
                );

                // 获取该分区的状态
                if let Some(partition_batch_state) =
                    self.partition_buffers.get_mut(&partition_row)
                {
                    // 将数据集合并到该分区下
                    partition_batch_state.record_batch = concat_batches(
                        &self.input.schema(),
                        [&partition_batch_state.record_batch, &partition_batch],
                    )?;
                } else {
                    // 新建state
                    let partition_batch_state = PartitionBatchState {
                        record_batch: partition_batch,
                        is_end: false,
                    };
                    self.partition_buffers
                        .insert(partition_row, partition_batch_state);
                };
            }
        }

        // 此时数据集已经按照分区划分好了 len就是总分区数
        let n_partitions = self.partition_buffers.len();
        for (idx, (_, partition_batch_state)) in
            self.partition_buffers.iter_mut().enumerate()
        {
            partition_batch_state.is_end |= idx < n_partitions - 1;
        }

        // input_buffer中存储所有的数据集
        self.input_buffer = if self.input_buffer.num_rows() == 0 {
            record_batch
        } else {
            concat_batches(&self.input.schema(), [&self.input_buffer, &record_batch])?
        };

        Ok(())
    }
}

impl Stream for SortedPartitionByBoundedWindowStream {
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        let poll = self.poll_next_inner(cx);
        self.baseline_metrics.record_poll(poll)
    }
}

impl SortedPartitionByBoundedWindowStream {
    /// Create a new BoundedWindowAggStream
    /// 初始化数据流
    pub fn new(
        schema: SchemaRef,
        window_expr: Vec<Arc<dyn WindowExpr>>,
        input: SendableRecordBatchStream,
        baseline_metrics: BaselineMetrics,
        partition_by_sort_keys: Vec<PhysicalSortExpr>,
        ordered_partition_by_indices: Vec<usize>,
    ) -> Result<Self> {
        let state = window_expr.iter().map(|_| IndexMap::new()).collect();
        let empty_batch = RecordBatch::new_empty(schema.clone());

        // In BoundedWindowAggExec all partition by columns should be ordered.
        if window_expr[0].partition_by().len() != ordered_partition_by_indices.len() {
            return Err(DataFusionError::Internal(
                "All partition by columns should have an ordering".to_string(),
            ));
        }

        Ok(Self {
            schema,
            input,
            input_buffer: empty_batch,
            partition_buffers: IndexMap::new(),
            window_agg_states: state,
            finished: false,
            window_expr,
            baseline_metrics,
            partition_by_sort_keys,
            ordered_partition_by_indices,
        })
    }

    // 处理state中囤积的数据 产生结果集
    fn compute_aggregates(&mut self) -> Result<RecordBatch> {
        // calculate window cols
        for (cur_window_expr, state) in
            // window_agg_states 按分区键存储对应分区在window处理下的结果
            self.window_expr.iter().zip(&mut self.window_agg_states)
        {
            // 处理分区数据 还需要考量之前的状态
            cur_window_expr.evaluate_stateful(&self.partition_buffers, state)?;
        }

        let schema = self.schema.clone();
        // 基于当前 state的数据产生输出结果
        let columns_to_show = self.calculate_out_columns()?;
        if let Some(columns_to_show) = columns_to_show {
            let n_generated = columns_to_show[0].len();
            // 每产生多少行记录 就要抹掉最前面对应数量的记录
            self.prune_state(n_generated)?;
            Ok(RecordBatch::try_new(schema, columns_to_show)?)
        } else {
            Ok(RecordBatch::new_empty(schema))
        }
    }

    // 拉取数据并使用window函数处理
    #[inline]
    fn poll_next_inner(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<RecordBatch>>> {
        if self.finished {
            return Poll::Ready(None);
        }

        let result = match ready!(self.input.poll_next_unpin(cx)) {
            Some(Ok(batch)) => {
                // 将所有数据按照分区存储到不同的state中
                self.update_partition_batch(batch)?;
                self.compute_aggregates()
            }
            Some(Err(e)) => Err(e),
            None => {
                // 内部plan的数据已经读取完了
                self.finished = true;
                for (_, partition_batch_state) in self.partition_buffers.iter_mut() {
                    partition_batch_state.is_end = true;
                }
                // 处理剩余的数据
                self.compute_aggregates()
            }
        };
        Poll::Ready(Some(result))
    }

    /// Calculates how many rows [SortedPartitionByBoundedWindowStream]
    /// can produce as output.
    /// 判断会产生多少行记录
    fn calculate_n_out_row(&self) -> usize {
        // Different window aggregators may produce results with different rates.
        // We produce the overall batch result with the same speed as slowest one.
        self.window_agg_states
            .iter()
            .map(|window_agg_state| {   // 对应每个window函数 下所有分区的状态
                // Store how many elements are generated for the current
                // window expression:
                let mut cur_window_expr_out_result_len = 0;
                // We iterate over `window_agg_state`, which is an IndexMap.
                // Iterations follow the insertion order, hence we preserve
                // sorting when partition columns are sorted.
                for (_, WindowState { state, .. }) in window_agg_state.iter() {
                    cur_window_expr_out_result_len += state.out_col.len();
                    // If we do not generate all results for the current
                    // partition, we do not generate results for next
                    // partition --  otherwise we will lose input ordering.
                    if state.n_row_result_missing > 0 {
                        break;
                    }
                }
                // 将所有分区每个state产出的总条数求和
                cur_window_expr_out_result_len
            })
            .min()
            .unwrap_or(0)
    }

    /// Prunes the sections of the record batch (for each partition)
    /// that we no longer need to calculate the window function result.
    /// 去掉不需要的分区数据 因为他们已经产生了结果
    fn prune_partition_batches(&mut self) -> Result<()> {
        // Remove partitions which we know already ended (is_end flag is true).
        // Since the retain method preserves insertion order, we still have
        // ordering in between partitions after removal.
        // 将end的数据移除
        self.partition_buffers
            .retain(|_, partition_batch_state| !partition_batch_state.is_end);

        // The data in `self.partition_batches` is used by all window expressions.
        // Therefore, when removing from `self.partition_batches`, we need to remove
        // from the earliest range boundary among all window expressions. Variable
        // `n_prune_each_partition` fill the earliest range boundary information for
        // each partition. This way, we can delete the no-longer-needed sections from
        // `self.partition_batches`.
        // For instance, if window frame one uses [10, 20] and window frame two uses
        // [5, 15]; we only prune the first 5 elements from the corresponding record
        // batch in `self.partition_batches`.

        // Calculate how many elements to prune for each partition batch
        let mut n_prune_each_partition: HashMap<PartitionKey, usize> = HashMap::new();
        for window_agg_state in self.window_agg_states.iter_mut() {
            // 移除end数据
            window_agg_state.retain(|_, WindowState { state, .. }| !state.is_end);
            for (partition_row, WindowState { state: value, .. }) in window_agg_state {
                let n_prune =
                    min(value.window_frame_range.start, value.last_calculated_index);
                if let Some(current) = n_prune_each_partition.get_mut(partition_row) {
                    if n_prune < *current {
                        *current = n_prune;
                    }
                } else {
                    n_prune_each_partition.insert(partition_row.clone(), n_prune);
                }
            }
        }

        let err = || DataFusionError::Execution("Expects to have partition".to_string());
        // Retract no longer needed parts during window calculations from partition batch:
        for (partition_row, n_prune) in n_prune_each_partition.iter() {
            let partition_batch_state = self
                .partition_buffers
                .get_mut(partition_row)
                .ok_or_else(err)?;
            let batch = &partition_batch_state.record_batch;
            partition_batch_state.record_batch =
                batch.slice(*n_prune, batch.num_rows() - n_prune);

            // Update state indices since we have pruned some rows from the beginning:
            for window_agg_state in self.window_agg_states.iter_mut() {
                window_agg_state[partition_row].state.prune_state(*n_prune);
            }
        }
        Ok(())
    }

    /// Prunes the section of the input batch whose aggregate results
    /// are calculated and emitted.
    fn prune_input_batch(&mut self, n_out: usize) -> Result<()> {
        let n_to_keep = self.input_buffer.num_rows() - n_out;
        let batch_to_keep = self
            .input_buffer
            .columns()
            .iter()
            .map(|elem| elem.slice(n_out, n_to_keep))
            .collect::<Vec<_>>();
        self.input_buffer =
            RecordBatch::try_new(self.input_buffer.schema(), batch_to_keep)?;
        Ok(())
    }

    /// Prunes emitted parts from WindowAggState `out_col` field.
    /// 减小out_col
    fn prune_out_columns(&mut self, n_out: usize) -> Result<()> {
        // We store generated columns for each window expression in the `out_col`
        // field of `WindowAggState`. Given how many rows are emitted, we remove
        // these sections from state.
        for partition_window_agg_states in self.window_agg_states.iter_mut() {
            let mut running_length = 0;
            // Remove `n_out` entries from the `out_col` field of `WindowAggState`.
            // Preserve per partition ordering by iterating in the order of insertion.
            // Do not generate a result for a new partition without emitting all results
            // for the current partition.
            for (
                _,
                WindowState {
                    state: WindowAggState { out_col, .. },
                    ..
                },
            ) in partition_window_agg_states
            {
                if running_length < n_out {
                    let n_to_del = min(out_col.len(), n_out - running_length);
                    let n_to_keep = out_col.len() - n_to_del;
                    // 只保留后面的部分
                    *out_col = out_col.slice(n_to_del, n_to_keep);
                    running_length += n_to_del;
                }
            }
        }
        Ok(())
    }

    /// Get Partition Columns
    /// 这样可以反向还原回 内部plan的排序键 是通过get_ordered_partition_by_indices保证了顺序
    pub fn partition_columns(&self, batch: &RecordBatch) -> Result<Vec<SortColumn>> {
        self.ordered_partition_by_indices
            .iter()
            .map(|idx| self.partition_by_sort_keys[*idx].evaluate_to_sort_column(batch))
            .collect()
    }
}

impl RecordBatchStream for SortedPartitionByBoundedWindowStream {
    /// Get the schema
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

/// Calculates the section we can show results for expression
/// 表示从state中展示  len行记录
fn get_aggregate_result_out_column(
    partition_window_agg_states: &PartitionWindowAggStates,
    len_to_show: usize,
) -> Result<ArrayRef> {
    let mut result = None;
    let mut running_length = 0;
    // We assume that iteration order is according to insertion order
    for (
        _,
        WindowState {
            state: WindowAggState { out_col, .. },
            ..
        },
    ) in partition_window_agg_states
    {
        if running_length < len_to_show {
            // 要展示的数量
            let n_to_use = min(len_to_show - running_length, out_col.len());
            let slice_to_use = out_col.slice(0, n_to_use);
            // 将多分区结果聚合
            result = Some(match result {
                Some(arr) => concat(&[&arr, &slice_to_use])?,
                None => slice_to_use,
            });
            running_length += n_to_use;
        } else {
            break;
        }
    }
    if running_length != len_to_show {
        return Err(DataFusionError::Execution(format!(
            "Generated row number should be {len_to_show}, it is {running_length}"
        )));
    }
    result
        .ok_or_else(|| DataFusionError::Execution("Should contain something".to_string()))
}
