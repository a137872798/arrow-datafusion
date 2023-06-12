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

//! Sort that deals with an arbitrary size of the input.
//! It will do in-memory sorting if it has enough memory budget
//! but spills to disk if needed.

use crate::error::{DataFusionError, Result};
use crate::execution::context::TaskContext;
use crate::execution::memory_pool::{
    human_readable_size, MemoryConsumer, MemoryReservation,
};
use crate::execution::runtime_env::RuntimeEnv;
use crate::physical_plan::common::{batch_byte_size, IPCWriter, SizedRecordBatchStream};
use crate::physical_plan::expressions::PhysicalSortExpr;
use crate::physical_plan::metrics::{
    BaselineMetrics, CompositeMetricsSet, MemTrackingMetrics, MetricsSet,
};
use crate::physical_plan::sorts::merge::streaming_merge;
use crate::physical_plan::stream::{RecordBatchReceiverStream, RecordBatchStreamAdapter};
use crate::physical_plan::{
    DisplayFormatType, Distribution, EmptyRecordBatchStream, ExecutionPlan, Partitioning,
    RecordBatchStream, SendableRecordBatchStream, Statistics,
};
use crate::prelude::SessionConfig;
use arrow::array::{make_array, Array, ArrayRef, MutableArrayData};
pub use arrow::compute::SortOptions;
use arrow::compute::{concat, lexsort_to_indices, take, SortColumn, TakeOptions};
use arrow::datatypes::SchemaRef;
use arrow::error::ArrowError;
use arrow::ipc::reader::FileReader;
use arrow::record_batch::RecordBatch;
use datafusion_physical_expr::EquivalenceProperties;
use futures::{Stream, StreamExt, TryStreamExt};
use log::{debug, error};
use std::any::Any;
use std::cmp::{min, Ordering};
use std::fmt;
use std::fmt::{Debug, Formatter};
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::task::{Context, Poll};
use tempfile::NamedTempFile;
use tokio::sync::mpsc::{Receiver, Sender};
use tokio::task;

/// Sort arbitrary size of data to get a total order (may spill several times during sorting based on free memory available).
///
/// The basic architecture of the algorithm:
/// 1. get a non-empty new batch from input
/// 2. check with the memory manager if we could buffer the batch in memory
/// 2.1 if memory sufficient, then buffer batch in memory, go to 1.
/// 2.2 if the memory threshold is reached, sort all buffered batches and spill to file.
///     buffer the batch in memory, go to 1.
/// 3. when input is exhausted, merge all in memory batches and spills to get a total order.
/// 该对象用于对数据集排序
struct ExternalSorter {
    schema: SchemaRef,
    /// 每批收到的数据 我们都会先进行排序 并暂存在内存中  在最后需要导出数据时 在做一次排序
    in_mem_batches: Vec<BatchWithSortArray>,
    /// 内存不足时 需要一些临时文件辅助
    spills: Vec<NamedTempFile>,
    /// Sort expressions
    expr: Vec<PhysicalSortExpr>,
    session_config: Arc<SessionConfig>,
    runtime: Arc<RuntimeEnv>,
    metrics_set: CompositeMetricsSet,
    metrics: BaselineMetrics,
    fetch: Option<usize>,
    reservation: MemoryReservation,
    partition_id: usize,
}

impl ExternalSorter {
    pub fn new(
        partition_id: usize,  // 针对某个分区创建的排序对象   但是不是有将所有分区数据合并后再排序的选择吗
        schema: SchemaRef,
        expr: Vec<PhysicalSortExpr>,  // 排序列
        metrics_set: CompositeMetricsSet,
        session_config: Arc<SessionConfig>,
        runtime: Arc<RuntimeEnv>,
        fetch: Option<usize>,
    ) -> Self {
        let metrics = metrics_set.new_intermediate_baseline(partition_id);

        let reservation = MemoryConsumer::new(format!("ExternalSorter[{partition_id}]"))
            .with_can_spill(true)
            .register(&runtime.memory_pool);

        Self {
            schema,
            in_mem_batches: vec![],
            spills: vec![],
            expr,
            session_config,
            runtime,
            metrics_set,
            metrics,
            fetch,
            reservation,
            partition_id,
        }
    }

    // 添加数据集
    async fn insert_batch(
        &mut self,
        input: RecordBatch,
        tracking_metrics: &MemTrackingMetrics,
    ) -> Result<()> {

        if input.num_rows() > 0 {
            let size = batch_byte_size(&input);
            if self.reservation.try_grow(size).is_err() {
                self.spill().await?;
                self.reservation.try_grow(size)?
            }

            self.metrics.mem_used().add(size);
            // NB timer records time taken on drop, so there are no
            // calls to `timer.done()` below.
            let _timer = tracking_metrics.elapsed_compute().timer();

            // 对单个数据集进行排序  得到该数据集的排序结果 已经排序后的排序列
            let partial = sort_batch(input, self.schema.clone(), &self.expr, self.fetch)?;

            // The resulting batch might be smaller (or larger, see #3747) than the input
            // batch due to either a propagated limit or the re-construction of arrays. So
            // for being reliable, we need to reflect the memory usage of the partial batch.
            // 评估这个临时的数据集需要占用多少内存
            let new_size = batch_byte_size(&partial.sorted_batch);
            match new_size.cmp(&size) {
                Ordering::Greater => {
                    // We don't have to call try_grow here, since we have already used the
                    // memory (so spilling right here wouldn't help at all for the current
                    // operation). But we still have to record it so that other requesters
                    // would know about this unexpected increase in memory consumption.
                    let new_size_delta = new_size - size;
                    self.reservation.grow(new_size_delta);
                    self.metrics.mem_used().add(new_size_delta);
                }
                Ordering::Less => {
                    let size_delta = size - new_size;
                    self.reservation.shrink(size_delta);
                    self.metrics.mem_used().sub(size_delta);
                }
                Ordering::Equal => {}
            }

            // 将结果暂存起来
            self.in_mem_batches.push(partial);
        }
        Ok(())
    }

    fn spilled_before(&self) -> bool {
        !self.spills.is_empty()
    }

    /// MergeSort in mem batches as well as spills into total order with `SortPreservingMergeStream`.
    /// 对维护在内存中的数据集做一次排序处理 并产生一个stream
    fn sort(&mut self) -> Result<SendableRecordBatchStream> {
        let batch_size = self.session_config.batch_size();

        // 代表由于内存不足 使用了额外的文件
        if self.spilled_before() {
            let intermediate_metrics = self
                .metrics_set
                .new_intermediate_tracking(self.partition_id, &self.runtime.memory_pool);
            let mut merge_metrics = self
                .metrics_set
                .new_final_tracking(self.partition_id, &self.runtime.memory_pool);

            let mut streams = vec![];

            // 先处理本次内存中的数据
            if !self.in_mem_batches.is_empty() {
                let in_mem_stream = in_mem_partial_sort(
                    &mut self.in_mem_batches,
                    self.schema.clone(),
                    &self.expr,
                    batch_size,
                    intermediate_metrics,
                    self.fetch,
                )?;
                // TODO: More accurate, dynamic memory accounting (#5885)
                merge_metrics.init_mem_used(self.reservation.free());
                streams.push(in_mem_stream);
            }

            // 加载文件中的数据
            for spill in self.spills.drain(..) {
                let stream = read_spill_as_stream(spill, self.schema.clone())?;
                streams.push(stream);
            }

            // 合并多个stream的数据
            streaming_merge(
                streams,
                self.schema.clone(),
                &self.expr,
                merge_metrics,
                self.session_config.batch_size(),
            )

            // 处理之前每批排序好的数据
        } else if !self.in_mem_batches.is_empty() {
            let tracking_metrics = self
                .metrics_set
                .new_final_tracking(self.partition_id, &self.runtime.memory_pool);

            // 处理 in_mem_batches 数据
            let result = in_mem_partial_sort(
                &mut self.in_mem_batches,
                self.schema.clone(),
                &self.expr,
                batch_size,
                tracking_metrics,
                self.fetch,
            );
            // Report to the memory manager we are no longer using memory  此时已经完成排序了 不再需要管理内存
            self.reservation.free();
            result
        } else {
            Ok(Box::pin(EmptyRecordBatchStream::new(self.schema.clone())))
        }
    }

    fn used(&self) -> usize {
        self.metrics.mem_used().value()
    }

    fn spilled_bytes(&self) -> usize {
        self.metrics.spilled_bytes().value()
    }

    fn spill_count(&self) -> usize {
        self.metrics.spill_count().value()
    }

    // 代表申请的内存不够用了
    async fn spill(&mut self) -> Result<usize> {
        // we could always get a chance to free some memory as long as we are holding some
        if self.in_mem_batches.is_empty() {
            return Ok(0);
        }

        debug!("Spilling sort data of ExternalSorter to disk whilst inserting");

        let tracking_metrics = self
            .metrics_set
            .new_intermediate_tracking(self.partition_id, &self.runtime.memory_pool);

        // 创建临时文件
        let spillfile = self.runtime.disk_manager.create_tmp_file("Sorting")?;

        // 先处理内存中的数据 生成数据流  产出的数据就是排序好的数据
        let stream = in_mem_partial_sort(
            &mut self.in_mem_batches,
            self.schema.clone(),
            &self.expr,
            self.session_config.batch_size(),
            tracking_metrics,
            self.fetch,
        );

        // 将数据先写入到文件中
        spill_partial_sorted_stream(&mut stream?, spillfile.path(), self.schema.clone())
            .await?;
        self.reservation.free();
        let used = self.metrics.mem_used().set(0);
        self.metrics.record_spill(used);
        self.spills.push(spillfile);
        Ok(used)
    }
}

impl Debug for ExternalSorter {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.debug_struct("ExternalSorter")
            .field("memory_used", &self.used())
            .field("spilled_bytes", &self.spilled_bytes())
            .field("spill_count", &self.spill_count())
            .finish()
    }
}

/// consume the non-empty `sorted_batches` and do in_mem_sort
/// 处理 in_mem的数据
fn in_mem_partial_sort(
    buffered_batches: &mut Vec<BatchWithSortArray>,  // 之前每批排序好的数据
    schema: SchemaRef,
    expressions: &[PhysicalSortExpr],   //  排序列
    batch_size: usize,
    tracking_metrics: MemTrackingMetrics,
    fetch: Option<usize>,
) -> Result<SendableRecordBatchStream> {
    assert_ne!(buffered_batches.len(), 0);
    if buffered_batches.len() == 1 {
        // 只有一批数据直接包装成stream就好
        let result = buffered_batches.pop();
        Ok(Box::pin(SizedRecordBatchStream::new(
            schema,
            vec![Arc::new(result.unwrap().sorted_batch)],
            tracking_metrics,
        )))
    } else {

        // 提取出排序完的 排序列以及数据集
        let (sorted_arrays, batches): (Vec<Vec<ArrayRef>>, Vec<RecordBatch>) =
            buffered_batches
                .drain(..)
                .map(|b| {
                    let BatchWithSortArray {
                        sort_arrays,
                        sorted_batch: batch,
                    } = b;
                    (sort_arrays, batch)
                })
                .unzip();

        // 得到的迭代器 每次迭代可以得到 已经排序完的长度为batch_size的数据
        let sorted_iter = {
            // NB timer records time taken on drop, so there are no
            // calls to `timer.done()` below.
            let _timer = tracking_metrics.elapsed_compute().timer();
            get_sorted_iter(&sorted_arrays, expressions, batch_size, fetch)?
        };
        Ok(Box::pin(SortedSizedRecordBatchStream::new(
            schema,
            batches,
            sorted_iter,
            tracking_metrics,
        )))
    }
}

// 一个下标对象
#[derive(Debug, Copy, Clone)]
struct CompositeIndex {
    batch_idx: u32,  // 描述该下标对应第几批数据集
    row_idx: u32,  // 对应该数据集下的第几行
}

/// Get sorted iterator by sort concatenated `SortColumn`s
fn get_sorted_iter(
    sort_arrays: &[Vec<ArrayRef>],  //  对应每批排序好的排序键的值
    expr: &[PhysicalSortExpr],   // 描述排序规则
    batch_size: usize,
    fetch: Option<usize>,
) -> Result<SortedIterator> {

    // index对应已经排序完的某批数据集的某行
    let row_indices = sort_arrays
        .iter()
        .enumerate()
        .flat_map(|(i, arrays)| {
            // 为每批每行数据 生成Index对象
            (0..arrays[0].len()).map(move |r| CompositeIndex {
                // since we original use UInt32Array to index the combined mono batch,
                // component record batches won't overflow as well,
                // use u32 here for space efficiency.
                batch_idx: i as u32,
                row_idx: r as u32,
            })
        })
        .collect::<Vec<CompositeIndex>>();

    // 将多个数据集下相同排序列的值 纵向排列
    let sort_columns = expr
        .iter()
        .enumerate()
        // i 对应的是每个排序列
        .map(|(i, expr)| {
            // 拿到每批数据集该排序列的值
            let columns_i = sort_arrays
                .iter()
                .map(|cs| cs[i].as_ref())
                .collect::<Vec<&dyn Array>>();

            // 纵向累加
            Ok(SortColumn {
                values: concat(columns_i.as_slice())?,
                options: Some(expr.options),
            })
        })
        .collect::<Result<Vec<_>>>()?;

    // 对纵向拼接后的排序列进行重新排序 返回的排序后的行号
    let indices = lexsort_to_indices(&sort_columns, fetch)?;

    // Calculate composite index based on sorted indices
    // i 就是行号呀   也就是 batch_idx * row_idx
    let row_indices = indices
        .values()
        .iter()
        .map(|i| row_indices[*i as usize])
        .collect();

    Ok(SortedIterator::new(row_indices, batch_size))
}

struct SortedIterator {
    /// Current logical position in the iterator  已经读取过多少数据
    pos: usize,
    /// Sorted composite index of where to find the rows in buffered batches  存储的是排序好的行号 通过index的batch_idx/row_idx 可以从数据集中回查原纪录
    composite: Vec<CompositeIndex>,
    /// Maximum batch size to produce  单次返回的行数
    batch_size: usize,
}

impl SortedIterator {
    fn new(composite: Vec<CompositeIndex>, batch_size: usize) -> Self {
        Self {
            pos: 0,
            composite,
            batch_size,
        }
    }

    fn memory_size(&self) -> usize {
        std::mem::size_of_val(self) + std::mem::size_of_val(&self.composite[..])
    }
}

impl Iterator for SortedIterator {
    type Item = Vec<CompositeSlice>;

    /// Emit a max of `batch_size` positions each time
    fn next(&mut self) -> Option<Self::Item> {
        // 这里代表总长度
        let length = self.composite.len();
        // 所有数据都已经被使用过
        if self.pos >= length {
            return None;
        }

        let current_size = min(self.batch_size, length - self.pos);

        // Combine adjacent indexes from the same batch to make a slice,
        // for more efficient `extend` later.
        // pos能够找到下标  而下标中的batch_idx/row_idx 可以从之前排序好的数据集定位到行数据
        let mut last_batch_idx = self.composite[self.pos].batch_idx;
        let mut indices_in_batch = Vec::with_capacity(current_size);

        // 存储排序结果
        let mut slices = vec![];
        // 遍历index
        for ci in &self.composite[self.pos..self.pos + current_size] {
            if ci.batch_idx != last_batch_idx {
                // 将连续的顺序数据 变成slice装入vec中
                group_indices(last_batch_idx, &mut indices_in_batch, &mut slices);
                last_batch_idx = ci.batch_idx;
            }
            // 当切换batch时 才一次性加载数据
            indices_in_batch.push(ci.row_idx);
        }

        assert!(
            !indices_in_batch.is_empty(),
            "There should have at least one record in a sort output slice."
        );
        group_indices(last_batch_idx, &mut indices_in_batch, &mut slices);

        self.pos += current_size;
        Some(slices)
    }
}

/// Group continuous indices into a slice for better `extend` performance
fn group_indices(
    batch_idx: u32,   // 上次使用的batch_idx
    positions: &mut Vec<u32>,   // 存储该batch下需要读取的行号
    output: &mut Vec<CompositeSlice>,  //
) {
    // 对positions排序
    positions.sort_unstable();

    // 记录上次出现的位置
    let mut last_pos = 0;
    let mut run_length = 0;
    for pos in positions.iter() {
        if run_length == 0 {
            last_pos = *pos;
            run_length = 1;
            // 代表数据是连续的
        } else if *pos == last_pos + 1 {
            run_length += 1;
            last_pos = *pos;
        } else {
            // 当发现不连续的数据时   才触发push    连续出现的数据作为一个slice
            output.push(CompositeSlice {
                batch_idx,  // 数据所在的batch
                // 下面2个字段 框出一个行范围
                start_row_idx: last_pos + 1 - run_length,
                len: run_length as usize,
            });

            // 更新起点
            last_pos = *pos;
            run_length = 1;
        }
    }
    assert!(
        run_length > 0,
        "There should have at least one record in a sort output slice."
    );
    output.push(CompositeSlice {
        batch_idx,
        start_row_idx: last_pos + 1 - run_length,
        len: run_length as usize,
    });
    positions.clear()
}

/// Stream of sorted record batches  通过该对象读取排序完的数据
struct SortedSizedRecordBatchStream {
    schema: SchemaRef,
    batches: Vec<RecordBatch>,  // 对应之前排序好的数据集
    sorted_iter: SortedIterator,  // 迭代会返回排序好的数据所在的batch/row 需要回到batches检索数据
    num_cols: usize,  // 共有多少列
    metrics: MemTrackingMetrics,
}

impl SortedSizedRecordBatchStream {
    /// new
    pub fn new(
        schema: SchemaRef,
        batches: Vec<RecordBatch>,
        sorted_iter: SortedIterator,
        mut metrics: MemTrackingMetrics,
    ) -> Self {
        let size = batches.iter().map(batch_byte_size).sum::<usize>()
            + sorted_iter.memory_size();
        metrics.init_mem_used(size);
        let num_cols = batches[0].num_columns();
        SortedSizedRecordBatchStream {
            schema,
            batches,
            sorted_iter,
            num_cols,
            metrics,
        }
    }
}

impl Stream for SortedSizedRecordBatchStream {
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        _: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        // 遍历存储排序顺序的迭代器
        match self.sorted_iter.next() {
            None => Poll::Ready(None),
            Some(slices) => {

                // 代表连续的多少行
                let num_rows = slices.iter().map(|s| s.len).sum();
                // 将每个col对应的Array组合起来得到结果集
                let output = (0..self.num_cols)
                    .map(|i| {
                        let arrays = self
                            .batches
                            .iter()
                            .map(|b| b.column(i).to_data())
                            .collect::<Vec<_>>();

                        // 得到该列在每批数据集下的数据
                        let arrays = arrays.iter().collect();
                        let mut mutable = MutableArrayData::new(arrays, false, num_rows);
                        for x in slices.iter() {
                            // 代表将该位置的数据填充到mutable中
                            mutable.extend(
                                x.batch_idx as usize,
                                x.start_row_idx as usize,
                                x.start_row_idx as usize + x.len,
                            );
                        }
                        // 最后只会生成一个Array
                        make_array(mutable.freeze())
                    })
                    .collect::<Vec<_>>();
                let batch =
                    RecordBatch::try_new(self.schema.clone(), output).map_err(Into::into);
                let poll = Poll::Ready(Some(batch));
                self.metrics.record_poll(poll)
            }
        }
    }
}

struct CompositeSlice {
    batch_idx: u32,
    start_row_idx: u32,
    len: usize,
}

impl RecordBatchStream for SortedSizedRecordBatchStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

// 排序时 内存中可能会存有大量数据
async fn spill_partial_sorted_stream(
    in_mem_stream: &mut SendableRecordBatchStream,
    path: &Path,
    schema: SchemaRef,
) -> Result<()> {
    let (sender, receiver) = tokio::sync::mpsc::channel(2);
    let path: PathBuf = path.into();

    // 开启一个后台线程 接收者将数据写入到文件
    let handle = task::spawn_blocking(move || write_sorted(receiver, path, schema));
    // 将数据写入
    while let Some(item) = in_mem_stream.next().await {
        sender.send(item).await.ok();
    }
    drop(sender);
    match handle.await {
        Ok(r) => r,
        Err(e) => Err(DataFusionError::Execution(format!(
            "Error occurred while spilling {e}"
        ))),
    }
}

fn read_spill_as_stream(
    path: NamedTempFile,
    schema: SchemaRef,
) -> Result<SendableRecordBatchStream> {
    let (sender, receiver): (Sender<Result<RecordBatch>>, Receiver<Result<RecordBatch>>) =
        tokio::sync::mpsc::channel(2);
    let join_handle = task::spawn_blocking(move || {
        if let Err(e) = read_spill(sender, path.path()) {
            error!("Failure while reading spill file: {:?}. Error: {}", path, e);
        }
    });
    Ok(RecordBatchReceiverStream::create(
        &schema,
        receiver,
        join_handle,
    ))
}

// 将数据写入文件
fn write_sorted(
    mut receiver: Receiver<Result<RecordBatch>>,
    path: PathBuf,
    schema: SchemaRef,
) -> Result<()> {
    let mut writer = IPCWriter::new(path.as_ref(), schema.as_ref())?;
    while let Some(batch) = receiver.blocking_recv() {
        writer.write(&batch?)?;
    }
    writer.finish()?;
    debug!(
        "Spilled {} batches of total {} rows to disk, memory released {}",
        writer.num_batches,
        writer.num_rows,
        human_readable_size(writer.num_bytes as usize),
    );
    Ok(())
}

fn read_spill(sender: Sender<Result<RecordBatch>>, path: &Path) -> Result<()> {
    let file = BufReader::new(File::open(path)?);
    let reader = FileReader::try_new(file, None)?;
    for batch in reader {
        sender
            .blocking_send(batch.map_err(Into::into))
            .map_err(|e| DataFusionError::Execution(format!("{e}")))?;
    }
    Ok(())
}

/// Sort execution plan.
///
/// This operator supports sorting datasets that are larger than the
/// memory allotted by the memory manager, by spilling to disk.
#[derive(Debug)]
pub struct SortExec {
    /// Input schema
    pub(crate) input: Arc<dyn ExecutionPlan>,
    /// Sort expressions
    /// 排序键
    expr: Vec<PhysicalSortExpr>,
    /// Containing all metrics set created during sort
    metrics_set: CompositeMetricsSet,
    /// Preserve partitions of input plan. If false, the input partitions
    /// will be sorted and merged into a single output partition.
    /// 是否要保持分区的状态
    preserve_partitioning: bool,
    /// Fetch highest/lowest n results
    fetch: Option<usize>,
}

impl SortExec {
    /// Create a new sort execution plan
    #[deprecated(since = "22.0.0", note = "use `new` and `with_fetch`")]
    pub fn try_new(
        expr: Vec<PhysicalSortExpr>,
        input: Arc<dyn ExecutionPlan>,
        fetch: Option<usize>,
    ) -> Result<Self> {
        Ok(Self::new(expr, input).with_fetch(fetch))
    }

    /// Create a new sort execution plan that produces a single,
    /// sorted output partition.
    /// 通过一组排序键实现排序功能
    pub fn new(expr: Vec<PhysicalSortExpr>, input: Arc<dyn ExecutionPlan>) -> Self {
        Self {
            expr,
            input,
            metrics_set: CompositeMetricsSet::new(),
            preserve_partitioning: false,
            fetch: None,
        }
    }

    /// Create a new sort execution plan with the option to preserve
    /// the partitioning of the input plan
    #[deprecated(
        since = "22.0.0",
        note = "use `new`, `with_fetch` and `with_preserve_partioning` instead"
    )]
    pub fn new_with_partitioning(
        expr: Vec<PhysicalSortExpr>,
        input: Arc<dyn ExecutionPlan>,
        preserve_partitioning: bool,
        fetch: Option<usize>,
    ) -> Self {
        Self::new(expr, input)
            .with_fetch(fetch)
            .with_preserve_partitioning(preserve_partitioning)
    }

    /// Whether this `SortExec` preserves partitioning of the children
    pub fn preserve_partitioning(&self) -> bool {
        self.preserve_partitioning
    }

    /// Specify the partitioning behavior of this sort exec
    ///
    /// If `preserve_partitioning` is true, sorts each partition
    /// individually, producing one sorted strema for each input partition.
    ///
    /// If `preserve_partitioning` is false, sorts and merges all
    /// input partitions producing a single, sorted partition.
    /// true 代表每个分区分开排序     false 代表将分区数据合并后再排序
    pub fn with_preserve_partitioning(mut self, preserve_partitioning: bool) -> Self {
        self.preserve_partitioning = preserve_partitioning;
        self
    }

    /// Whether this `SortExec` preserves partitioning of the children
    /// 指定拉取的数量
    pub fn with_fetch(mut self, fetch: Option<usize>) -> Self {
        self.fetch = fetch;
        self
    }

    /// Input schema
    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }

    /// Sort expressions
    pub fn expr(&self) -> &[PhysicalSortExpr] {
        &self.expr
    }

    /// If `Some(fetch)`, limits output to only the first "fetch" items
    pub fn fetch(&self) -> Option<usize> {
        self.fetch
    }
}

impl ExecutionPlan for SortExec {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.input.schema()
    }

    /// Get the output partitioning of this plan
    fn output_partitioning(&self) -> Partitioning {
        // 保持分区状态
        if self.preserve_partitioning {
            self.input.output_partitioning()
        } else {
            // 最终只有一个分区
            Partitioning::UnknownPartitioning(1)
        }
    }

    /// Specifies whether this plan generates an infinite stream of records.
    /// If the plan does not support pipelining, but it its input(s) are
    /// infinite, returns an error to indicate this.
    fn unbounded_output(&self, children: &[bool]) -> Result<bool> {
        if children[0] {
            Err(DataFusionError::Plan(
                "Sort Error: Can not sort unbounded inputs.".to_string(),
            ))
        } else {
            Ok(false)
        }
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        if self.preserve_partitioning {
            vec![Distribution::UnspecifiedDistribution]
        } else {
            // global sort
            // TODO support RangePartition and OrderedDistribution
            vec![Distribution::SinglePartition]
        }
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![self.input.clone()]
    }

    fn benefits_from_input_partitioning(&self) -> bool {
        false
    }

    fn output_ordering(&self) -> Option<&[PhysicalSortExpr]> {
        Some(&self.expr)
    }

    fn equivalence_properties(&self) -> EquivalenceProperties {
        self.input.equivalence_properties()
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let new_sort = SortExec::new(self.expr.clone(), children[0].clone())
            .with_fetch(self.fetch)
            .with_preserve_partitioning(self.preserve_partitioning);

        Ok(Arc::new(new_sort))
    }

    // 将排序逻辑包装在input产生的结果集上
    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        debug!("Start SortExec::execute for partition {} of context session_id {} and task_id {:?}", partition, context.session_id(), context.task_id());

        debug!(
            "Start invoking SortExec's input.execute for partition: {}",
            partition
        );

        let input = self.input.execute(partition, context.clone())?;

        debug!("End SortExec's input.execute for partition: {}", partition);

        // 从包装后的流读取到的数据 自动完成了排序
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            futures::stream::once(do_sort(
                input,
                partition,
                self.expr.clone(),
                self.metrics_set.clone(),
                context,
                self.fetch(),
            ))
            .try_flatten(),
        )))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics_set.aggregate_all())
    }

    fn fmt_as(
        &self,
        t: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default => {
                let expr: Vec<String> = self.expr.iter().map(|e| e.to_string()).collect();
                match self.fetch {
                    Some(fetch) => {
                        write!(f, "SortExec: fetch={fetch}, expr=[{}]", expr.join(","))
                    }
                    None => write!(f, "SortExec: expr=[{}]", expr.join(",")),
                }
            }
        }
    }

    fn statistics(&self) -> Statistics {
        self.input.statistics()
    }
}

struct BatchWithSortArray {
    // 排序列在排序后的状态
    sort_arrays: Vec<ArrayRef>,
    // 排序后的数据集
    sorted_batch: RecordBatch,
}

// 对数据集进行排序
fn sort_batch(
    batch: RecordBatch,
    schema: SchemaRef,
    expr: &[PhysicalSortExpr],
    fetch: Option<usize>,
) -> Result<BatchWithSortArray> {

    // 取出排序列数据 搭配排序option
    let sort_columns = expr
        .iter()
        .map(|e| e.evaluate_to_sort_column(&batch))
        .collect::<Result<Vec<SortColumn>>>()?;

    // TODO 排序逻辑由arrow-ord包实现  不细看了 返回的是排序完后的行号
    let indices = lexsort_to_indices(&sort_columns, fetch)?;

    // reorder all rows based on sorted indices
    let sorted_batch = RecordBatch::try_new(
        schema,
        batch
            .columns()
            .iter()
            // 按照行号读取每列的数据 并拼接成数据集
            .map(|column| {
                take(
                    column.as_ref(),
                    &indices,
                    // disable bound check overhead since indices are already generated from
                    // the same record batch
                    Some(TakeOptions {
                        check_bounds: false,
                    }),
                )
            })
            .collect::<Result<Vec<ArrayRef>, ArrowError>>()?,
    )?;

    // 排序列在排序后的数据
    let sort_arrays = sort_columns
        .into_iter()
        .map(|sc| {
            Ok(take(
                sc.values.as_ref(),
                &indices,
                Some(TakeOptions {
                    check_bounds: false,
                }),
            )?)
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(BatchWithSortArray {
        sort_arrays,
        sorted_batch,
    })
}

// 对数据集进行排序
async fn do_sort(
    mut input: SendableRecordBatchStream,  // 数据集
    partition_id: usize,
    expr: Vec<PhysicalSortExpr>,  // 分区键
    metrics_set: CompositeMetricsSet,
    context: Arc<TaskContext>,
    fetch: Option<usize>,  // 代表要获取多少条记录
) -> Result<SendableRecordBatchStream> {
    debug!(
        "Start do_sort for partition {} of context session_id {} and task_id {:?}",
        partition_id,
        context.session_id(),
        context.task_id()
    );
    let schema = input.schema();
    let tracking_metrics =
        metrics_set.new_intermediate_tracking(partition_id, context.memory_pool());

    // 排序逻辑由该对象提供
    let mut sorter = ExternalSorter::new(
        partition_id,
        schema.clone(),
        expr,
        metrics_set,
        Arc::new(context.session_config().clone()),
        context.runtime_env(),
        fetch,
    );

    while let Some(batch) = input.next().await {
        let batch = batch?;
        // 将数据填充到排序对象中   可以看到这里是将input的所有数据都加载出来了  毕竟为了排序的效果 肯定是要在内存中装载全部数据的
        sorter.insert_batch(batch, &tracking_metrics).await?;
    }

    // 生成排序结果
    let result = sorter.sort();
    debug!(
        "End do_sort for partition {} of context session_id {} and task_id {:?}",
        partition_id,
        context.session_id(),
        context.task_id()
    );
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::context::SessionConfig;
    use crate::execution::runtime_env::RuntimeConfig;
    use crate::physical_plan::coalesce_partitions::CoalescePartitionsExec;
    use crate::physical_plan::collect;
    use crate::physical_plan::expressions::col;
    use crate::physical_plan::memory::MemoryExec;
    use crate::prelude::SessionContext;
    use crate::test;
    use crate::test::assert_is_pending;
    use crate::test::exec::{assert_strong_count_converges_to_zero, BlockingExec};
    use arrow::array::*;
    use arrow::compute::SortOptions;
    use arrow::datatypes::*;
    use datafusion_common::cast::{as_primitive_array, as_string_array};
    use futures::FutureExt;
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_in_mem_sort() -> Result<()> {
        let session_ctx = SessionContext::new();
        let task_ctx = session_ctx.task_ctx();
        let partitions = 4;
        let csv = test::scan_partitioned_csv(partitions)?;
        let schema = csv.schema();

        let sort_exec = Arc::new(SortExec::new(
            vec![
                // c1 string column
                PhysicalSortExpr {
                    expr: col("c1", &schema)?,
                    options: SortOptions::default(),
                },
                // c2 uin32 column
                PhysicalSortExpr {
                    expr: col("c2", &schema)?,
                    options: SortOptions::default(),
                },
                // c7 uin8 column
                PhysicalSortExpr {
                    expr: col("c7", &schema)?,
                    options: SortOptions::default(),
                },
            ],
            Arc::new(CoalescePartitionsExec::new(csv)),
        ));

        let result = collect(sort_exec, task_ctx).await?;

        assert_eq!(result.len(), 1);

        let columns = result[0].columns();

        let c1 = as_string_array(&columns[0])?;
        assert_eq!(c1.value(0), "a");
        assert_eq!(c1.value(c1.len() - 1), "e");

        let c2 = as_primitive_array::<UInt32Type>(&columns[1])?;
        assert_eq!(c2.value(0), 1);
        assert_eq!(c2.value(c2.len() - 1), 5,);

        let c7 = as_primitive_array::<UInt8Type>(&columns[6])?;
        assert_eq!(c7.value(0), 15);
        assert_eq!(c7.value(c7.len() - 1), 254,);

        assert_eq!(
            session_ctx.runtime_env().memory_pool.reserved(),
            0,
            "The sort should have returned all memory used back to the memory manager"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_sort_spill() -> Result<()> {
        // trigger spill there will be 4 batches with 5.5KB for each
        let config = RuntimeConfig::new().with_memory_limit(12288, 1.0);
        let runtime = Arc::new(RuntimeEnv::new(config)?);
        let session_ctx = SessionContext::with_config_rt(SessionConfig::new(), runtime);

        let partitions = 4;
        let csv = test::scan_partitioned_csv(partitions)?;
        let schema = csv.schema();

        let sort_exec = Arc::new(SortExec::new(
            vec![
                // c1 string column
                PhysicalSortExpr {
                    expr: col("c1", &schema)?,
                    options: SortOptions::default(),
                },
                // c2 uin32 column
                PhysicalSortExpr {
                    expr: col("c2", &schema)?,
                    options: SortOptions::default(),
                },
                // c7 uin8 column
                PhysicalSortExpr {
                    expr: col("c7", &schema)?,
                    options: SortOptions::default(),
                },
            ],
            Arc::new(CoalescePartitionsExec::new(csv)),
        ));

        let task_ctx = session_ctx.task_ctx();
        let result = collect(sort_exec.clone(), task_ctx).await?;

        assert_eq!(result.len(), 1);

        // Now, validate metrics
        let metrics = sort_exec.metrics().unwrap();

        assert_eq!(metrics.output_rows().unwrap(), 100);
        assert!(metrics.elapsed_compute().unwrap() > 0);
        assert!(metrics.spill_count().unwrap() > 0);
        assert!(metrics.spilled_bytes().unwrap() > 0);

        let columns = result[0].columns();

        let c1 = as_string_array(&columns[0])?;
        assert_eq!(c1.value(0), "a");
        assert_eq!(c1.value(c1.len() - 1), "e");

        let c2 = as_primitive_array::<UInt32Type>(&columns[1])?;
        assert_eq!(c2.value(0), 1);
        assert_eq!(c2.value(c2.len() - 1), 5,);

        let c7 = as_primitive_array::<UInt8Type>(&columns[6])?;
        assert_eq!(c7.value(0), 15);
        assert_eq!(c7.value(c7.len() - 1), 254,);

        assert_eq!(
            session_ctx.runtime_env().memory_pool.reserved(),
            0,
            "The sort should have returned all memory used back to the memory manager"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_sort_fetch_memory_calculation() -> Result<()> {
        // This test mirrors down the size from the example above.
        let avg_batch_size = 6000;
        let partitions = 4;

        // A tuple of (fetch, expect_spillage)
        let test_options = vec![
            // Since we don't have a limit (and the memory is less than the total size of
            // all the batches we are processing, we expect it to spill.
            (None, true),
            // When we have a limit however, the buffered size of batches should fit in memory
            // since it is much lower than the total size of the input batch.
            (Some(1), false),
        ];

        for (fetch, expect_spillage) in test_options {
            let config = RuntimeConfig::new()
                .with_memory_limit(avg_batch_size * (partitions - 1), 1.0);
            let runtime = Arc::new(RuntimeEnv::new(config)?);
            let session_ctx =
                SessionContext::with_config_rt(SessionConfig::new(), runtime);

            let csv = test::scan_partitioned_csv(partitions)?;
            let schema = csv.schema();

            let sort_exec = Arc::new(
                SortExec::new(
                    vec![
                        // c1 string column
                        PhysicalSortExpr {
                            expr: col("c1", &schema)?,
                            options: SortOptions::default(),
                        },
                        // c2 uin32 column
                        PhysicalSortExpr {
                            expr: col("c2", &schema)?,
                            options: SortOptions::default(),
                        },
                        // c7 uin8 column
                        PhysicalSortExpr {
                            expr: col("c7", &schema)?,
                            options: SortOptions::default(),
                        },
                    ],
                    Arc::new(CoalescePartitionsExec::new(csv)),
                )
                .with_fetch(fetch),
            );

            let task_ctx = session_ctx.task_ctx();
            let result = collect(sort_exec.clone(), task_ctx).await?;
            assert_eq!(result.len(), 1);

            let metrics = sort_exec.metrics().unwrap();
            let did_it_spill = metrics.spill_count().unwrap() > 0;
            assert_eq!(did_it_spill, expect_spillage, "with fetch: {fetch:?}");
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_sort_metadata() -> Result<()> {
        let session_ctx = SessionContext::new();
        let task_ctx = session_ctx.task_ctx();
        let field_metadata: HashMap<String, String> =
            vec![("foo".to_string(), "bar".to_string())]
                .into_iter()
                .collect();
        let schema_metadata: HashMap<String, String> =
            vec![("baz".to_string(), "barf".to_string())]
                .into_iter()
                .collect();

        let mut field = Field::new("field_name", DataType::UInt64, true);
        field.set_metadata(field_metadata.clone());
        let schema = Schema::new_with_metadata(vec![field], schema_metadata.clone());
        let schema = Arc::new(schema);

        let data: ArrayRef =
            Arc::new(vec![3, 2, 1].into_iter().map(Some).collect::<UInt64Array>());

        let batch = RecordBatch::try_new(schema.clone(), vec![data]).unwrap();
        let input =
            Arc::new(MemoryExec::try_new(&[vec![batch]], schema.clone(), None).unwrap());

        let sort_exec = Arc::new(SortExec::new(
            vec![PhysicalSortExpr {
                expr: col("field_name", &schema)?,
                options: SortOptions::default(),
            }],
            input,
        ));

        let result: Vec<RecordBatch> = collect(sort_exec, task_ctx).await?;

        let expected_data: ArrayRef =
            Arc::new(vec![1, 2, 3].into_iter().map(Some).collect::<UInt64Array>());
        let expected_batch =
            RecordBatch::try_new(schema.clone(), vec![expected_data]).unwrap();

        // Data is correct
        assert_eq!(&vec![expected_batch], &result);

        // explicitlty ensure the metadata is present
        assert_eq!(result[0].schema().fields()[0].metadata(), &field_metadata);
        assert_eq!(result[0].schema().metadata(), &schema_metadata);

        Ok(())
    }

    #[tokio::test]
    async fn test_lex_sort_by_float() -> Result<()> {
        let session_ctx = SessionContext::new();
        let task_ctx = session_ctx.task_ctx();
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Float32, true),
            Field::new("b", DataType::Float64, true),
        ]));

        // define data.
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Float32Array::from(vec![
                    Some(f32::NAN),
                    None,
                    None,
                    Some(f32::NAN),
                    Some(1.0_f32),
                    Some(1.0_f32),
                    Some(2.0_f32),
                    Some(3.0_f32),
                ])),
                Arc::new(Float64Array::from(vec![
                    Some(200.0_f64),
                    Some(20.0_f64),
                    Some(10.0_f64),
                    Some(100.0_f64),
                    Some(f64::NAN),
                    None,
                    None,
                    Some(f64::NAN),
                ])),
            ],
        )?;

        let sort_exec = Arc::new(SortExec::new(
            vec![
                PhysicalSortExpr {
                    expr: col("a", &schema)?,
                    options: SortOptions {
                        descending: true,
                        nulls_first: true,
                    },
                },
                PhysicalSortExpr {
                    expr: col("b", &schema)?,
                    options: SortOptions {
                        descending: false,
                        nulls_first: false,
                    },
                },
            ],
            Arc::new(MemoryExec::try_new(&[vec![batch]], schema, None)?),
        ));

        assert_eq!(DataType::Float32, *sort_exec.schema().field(0).data_type());
        assert_eq!(DataType::Float64, *sort_exec.schema().field(1).data_type());

        let result: Vec<RecordBatch> = collect(sort_exec.clone(), task_ctx).await?;
        let metrics = sort_exec.metrics().unwrap();
        assert!(metrics.elapsed_compute().unwrap() > 0);
        assert_eq!(metrics.output_rows().unwrap(), 8);
        assert_eq!(result.len(), 1);

        let columns = result[0].columns();

        assert_eq!(DataType::Float32, *columns[0].data_type());
        assert_eq!(DataType::Float64, *columns[1].data_type());

        let a = as_primitive_array::<Float32Type>(&columns[0])?;
        let b = as_primitive_array::<Float64Type>(&columns[1])?;

        // convert result to strings to allow comparing to expected result containing NaN
        let result: Vec<(Option<String>, Option<String>)> = (0..result[0].num_rows())
            .map(|i| {
                let aval = if a.is_valid(i) {
                    Some(a.value(i).to_string())
                } else {
                    None
                };
                let bval = if b.is_valid(i) {
                    Some(b.value(i).to_string())
                } else {
                    None
                };
                (aval, bval)
            })
            .collect();

        let expected: Vec<(Option<String>, Option<String>)> = vec![
            (None, Some("10".to_owned())),
            (None, Some("20".to_owned())),
            (Some("NaN".to_owned()), Some("100".to_owned())),
            (Some("NaN".to_owned()), Some("200".to_owned())),
            (Some("3".to_owned()), Some("NaN".to_owned())),
            (Some("2".to_owned()), None),
            (Some("1".to_owned()), Some("NaN".to_owned())),
            (Some("1".to_owned()), None),
        ];

        assert_eq!(expected, result);

        Ok(())
    }

    #[tokio::test]
    async fn test_drop_cancel() -> Result<()> {
        let session_ctx = SessionContext::new();
        let task_ctx = session_ctx.task_ctx();
        let schema =
            Arc::new(Schema::new(vec![Field::new("a", DataType::Float32, true)]));

        let blocking_exec = Arc::new(BlockingExec::new(Arc::clone(&schema), 1));
        let refs = blocking_exec.refs();
        let sort_exec = Arc::new(SortExec::new(
            vec![PhysicalSortExpr {
                expr: col("a", &schema)?,
                options: SortOptions::default(),
            }],
            blocking_exec,
        ));

        let fut = collect(sort_exec, task_ctx);
        let mut fut = fut.boxed();

        assert_is_pending(&mut fut);
        drop(fut);
        assert_strong_count_converges_to_zero(refs).await;

        assert_eq!(
            session_ctx.runtime_env().memory_pool.reserved(),
            0,
            "The sort should have returned all memory used back to the memory manager"
        );

        Ok(())
    }
}
