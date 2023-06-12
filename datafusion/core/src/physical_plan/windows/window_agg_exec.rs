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

use crate::error::Result;
use crate::execution::context::TaskContext;
use crate::physical_plan::common::transpose;
use crate::physical_plan::expressions::PhysicalSortExpr;
use crate::physical_plan::metrics::{
    BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet,
};
use crate::physical_plan::windows::{
    calc_requirements, get_ordered_partition_by_indices,
};
use crate::physical_plan::{
    ColumnStatistics, DisplayFormatType, Distribution, EquivalenceProperties,
    ExecutionPlan, Partitioning, PhysicalExpr, RecordBatchStream,
    SendableRecordBatchStream, Statistics, WindowExpr,
};
use arrow::compute::{concat, concat_batches};
use arrow::datatypes::SchemaBuilder;
use arrow::error::ArrowError;
use arrow::{
    array::ArrayRef,
    datatypes::{Schema, SchemaRef},
    record_batch::RecordBatch,
};
use datafusion_common::utils::{evaluate_partition_ranges, get_at_indices};
use datafusion_common::DataFusionError;
use datafusion_physical_expr::PhysicalSortRequirement;
use futures::stream::Stream;
use futures::{ready, StreamExt};
use std::any::Any;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

/// Window execution plan
/// 该窗口函数使用的内存不可控
#[derive(Debug)]
pub struct WindowAggExec {
    /// Input plan
    pub(crate) input: Arc<dyn ExecutionPlan>,
    /// Window function expression
    window_expr: Vec<Arc<dyn WindowExpr>>,
    /// Schema after the window is run
    schema: SchemaRef,
    /// Schema before the window
    input_schema: SchemaRef,
    /// Partition Keys
    pub partition_keys: Vec<Arc<dyn PhysicalExpr>>,
    /// Execution metrics
    metrics: ExecutionPlanMetricsSet,
    /// Partition by indices that defines preset for existing ordering
    // see `get_ordered_partition_by_indices` for more details.
    ordered_partition_by_indices: Vec<usize>,
}

impl WindowAggExec {

    /// Create a new execution plan for window aggregates
    pub fn try_new(
        window_expr: Vec<Arc<dyn WindowExpr>>,   // 使用同一排序键的窗口表达式
        input: Arc<dyn ExecutionPlan>,
        input_schema: SchemaRef,
        partition_keys: Vec<Arc<dyn PhysicalExpr>>,  // 使用的分区键
    ) -> Result<Self> {
        // 输入的schema 加上窗口函数产生的field 合成新的schema
        let schema = create_schema(&input_schema, &window_expr)?;
        let schema = Arc::new(schema);

        // 内部plan排序键 到外部window分区键的映射
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
    // 可以得到外部window的分区键
    pub fn partition_by_sort_keys(&self) -> Result<Vec<PhysicalSortExpr>> {
        // Partition by sort keys indices are stored in self.ordered_partition_by_indices.
        // 代表内部表达式 的排序键
        let sort_keys = self.input.output_ordering().unwrap_or(&[]);
        // ordered_partition_by_indices是内部排序键 到外部window分区键的下标  那么作用到sort_keys上 自然就变成了 window的分区键
        get_at_indices(sort_keys, &self.ordered_partition_by_indices)
    }
}

impl ExecutionPlan for WindowAggExec {
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
    /// 分区基于内部的plan
    fn output_partitioning(&self) -> Partitioning {
        // because we can have repartitioning using the partition keys
        // this would be either 1 or more than 1 depending on the presense of
        // repartitioning
        self.input.output_partitioning()
    }

    /// Specifies whether this plan generates an infinite stream of records.
    /// If the plan does not support pipelining, but it its input(s) are
    /// infinite, returns an error to indicate this.    
    fn unbounded_output(&self, children: &[bool]) -> Result<bool> {
        if children[0] {
            Err(DataFusionError::Plan(
                "Window Error: Windowing is not currently support for unbounded inputs."
                    .to_string(),
            ))
        } else {
            Ok(false)
        }
    }

    // 产生的数据结果顺序 由内部plan决定
    fn output_ordering(&self) -> Option<&[PhysicalSortExpr]> {
        self.input().output_ordering()
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    // 获取期望的排序顺序  完全不知道这个方法是干嘛的
    fn required_input_ordering(&self) -> Vec<Option<Vec<PhysicalSortRequirement>>> {
        // 获取窗口函数的分区键和顺序键
        let partition_bys = self.window_expr()[0].partition_by();
        let order_keys = self.window_expr()[0].order_by();
        // 代表input的排序键 能够转换成的分区键的数量
        if self.ordered_partition_by_indices.len() < partition_bys.len() {
            vec![calc_requirements(partition_bys, order_keys)]
        } else {
            // 按照顺序对分区键重新排序
            let partition_bys = self
                .ordered_partition_by_indices
                .iter()
                .map(|idx| &partition_bys[*idx]);
            vec![calc_requirements(partition_bys, order_keys)]
        }
    }

    // 是否需要分区
    fn required_input_distribution(&self) -> Vec<Distribution> {
        if self.partition_keys.is_empty() {
            vec![Distribution::SinglePartition]
        } else {
            // 代表基于分区键 进行分区
            vec![Distribution::HashPartitioned(self.partition_keys.clone())]
        }
    }

    fn equivalence_properties(&self) -> EquivalenceProperties {
        self.input().equivalence_properties()
    }

    // 替换内部的数据源 (plan)
    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(WindowAggExec::try_new(
            self.window_expr.clone(),
            children[0].clone(),
            self.input_schema.clone(),
            self.partition_keys.clone(),
        )?))
    }

    // 执行计划
    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        // 内部plan产生基础数据
        let input = self.input.execute(partition, context)?;

        // 对stream做一层包装
        let stream = Box::pin(WindowAggStream::new(
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
                write!(f, "WindowAggExec: ")?;
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

    // 返回统计i结果
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

        // 针对窗口函数产生的field  初始化统计对象
        column_statistics.extend(vec![ColumnStatistics::default(); win_cols]);
        Statistics {
            is_exact: input_stat.is_exact,
            num_rows: input_stat.num_rows,
            column_statistics: Some(column_statistics),
            total_byte_size: None,
        }
    }
}

// 将window产生的field 追加到schema上
fn create_schema(
    input_schema: &Schema,
    window_expr: &[Arc<dyn WindowExpr>],
) -> Result<Schema> {
    let capacity = input_schema.fields().len() + window_expr.len();
    let mut builder = SchemaBuilder::with_capacity(capacity);
    builder.extend(input_schema.fields().iter().cloned());
    // append results to the schema
    for expr in window_expr {
        builder.push(expr.field()?);
    }
    Ok(builder.finish())
}

/// Compute the window aggregate columns
/// 每个分区 被多个窗口函数作用后的结果
fn compute_window_aggregates(
    window_expr: &[Arc<dyn WindowExpr>],
    batch: &RecordBatch,
) -> Result<Vec<ArrayRef>> {
    window_expr
        .iter()
        .map(|window_expr| window_expr.evaluate(batch))
        .collect()
}

/// stream for window aggregation plan   将内部的数据流用window包装
pub struct WindowAggStream {
    schema: SchemaRef,
    input: SendableRecordBatchStream,

    // 暂存此时拉取到的全部数据
    batches: Vec<RecordBatch>,
    finished: bool,
    window_expr: Vec<Arc<dyn WindowExpr>>,
    partition_by_sort_keys: Vec<PhysicalSortExpr>,
    baseline_metrics: BaselineMetrics,
    ordered_partition_by_indices: Vec<usize>,
}

impl WindowAggStream {
    /// Create a new WindowAggStream
    pub fn new(
        schema: SchemaRef,
        window_expr: Vec<Arc<dyn WindowExpr>>,
        input: SendableRecordBatchStream,
        baseline_metrics: BaselineMetrics,
        partition_by_sort_keys: Vec<PhysicalSortExpr>,  // window分区键
        ordered_partition_by_indices: Vec<usize>,  // 内部plan的排序键 通过这些下标提取出来的列就是window分区键
    ) -> Result<Self> {
        // In WindowAggExec all partition by columns should be ordered.
        if window_expr[0].partition_by().len() != ordered_partition_by_indices.len() {
            return Err(DataFusionError::Internal(
                "All partition by columns should have an ordering".to_string(),
            ));
        }
        Ok(Self {
            schema,
            input,
            batches: vec![],
            finished: false,
            window_expr,
            baseline_metrics,
            partition_by_sort_keys,
            ordered_partition_by_indices,
        })
    }

    // 处理每批拉取到的数据
    fn compute_aggregates(&self) -> Result<RecordBatch> {
        // record compute time on drop  记录开始处理时间
        let _timer = self.baseline_metrics.elapsed_compute().timer();

        // 将此前暂存的batch数据 合并成一个recordBatch
        let batch = concat_batches(&self.input.schema(), &self.batches)?;
        // 没有数据需要处理返回空结果
        if batch.num_rows() == 0 {
            return Ok(RecordBatch::new_empty(self.schema.clone()));
        }

        // 将window的分区键 又转换成内部plan的排序键
        let partition_by_sort_keys = self
            .ordered_partition_by_indices
            .iter()
            .map(|idx| self.partition_by_sort_keys[*idx].evaluate_to_sort_column(&batch))
            .collect::<Result<Vec<_>>>()?;

        // 将记录按行分成多个分区
        let partition_points =
            evaluate_partition_ranges(batch.num_rows(), &partition_by_sort_keys)?;

        let mut partition_results = vec![];
        // Calculate window cols  计算每个分区的结果
        for partition_point in partition_points {
            // 得到每个分区的长度
            let length = partition_point.end - partition_point.start;
            partition_results.push(compute_window_aggregates(
                &self.window_expr,
                &batch.slice(partition_point.start, length),
            )?)
        }
        let columns = transpose(partition_results)
            .iter()
            .map(|elems| concat(&elems.iter().map(|x| x.as_ref()).collect::<Vec<_>>()))
            .collect::<Vec<_>>()
            .into_iter()
            .collect::<Result<Vec<ArrayRef>, ArrowError>>()?;

        // combine with the original cols
        // note the setup of window aggregates is that they newly calculated window
        // expression results are always appended to the columns
        let mut batch_columns = batch.columns().to_vec();
        // calculate window cols   窗口函数的每个列会合并到产生的结果上 作为新的结果
        batch_columns.extend_from_slice(&columns);
        Ok(RecordBatch::try_new(self.schema.clone(), batch_columns)?)
    }
}

impl Stream for WindowAggStream {
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        // 调用该方法 触发数据拉取行为  并基于窗口函数产生结果列
        let poll = self.poll_next_inner(cx);
        self.baseline_metrics.record_poll(poll)
    }
}

impl WindowAggStream {

    // 拉取数据
    #[inline]
    fn poll_next_inner(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<RecordBatch>>> {
        if self.finished {
            return Poll::Ready(None);
        }

        loop {
            // 从下游对象获取数据
            let result = match ready!(self.input.poll_next_unpin(cx)) {
                Some(Ok(batch)) => {
                    self.batches.push(batch);
                    continue;
                }
                Some(Err(e)) => Err(e),
                None => self.compute_aggregates(),
            };

            self.finished = true;

            return Poll::Ready(Some(result));
        }
    }
}

impl RecordBatchStream for WindowAggStream {
    /// Get the schema
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}
