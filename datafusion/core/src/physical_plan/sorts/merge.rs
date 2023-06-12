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

use crate::common::Result;
use crate::physical_plan::metrics::MemTrackingMetrics;
use crate::physical_plan::sorts::builder::BatchBuilder;
use crate::physical_plan::sorts::cursor::Cursor;
use crate::physical_plan::sorts::stream::{
    FieldCursorStream, PartitionedStream, RowCursorStream,
};
use crate::physical_plan::{
    PhysicalSortExpr, RecordBatchStream, SendableRecordBatchStream,
};
use arrow::datatypes::{DataType, SchemaRef};
use arrow::record_batch::RecordBatch;
use arrow_array::*;
use futures::Stream;
use std::pin::Pin;
use std::task::{ready, Context, Poll};

macro_rules! primitive_merge_helper {
    ($t:ty, $($v:ident),+) => {
        merge_helper!(PrimitiveArray<$t>, $($v),+)
    };
}

macro_rules! merge_helper {
    ($t:ty, $sort:ident, $streams:ident, $schema:ident, $tracking_metrics:ident, $batch_size:ident) => {{
        let streams = FieldCursorStream::<$t>::new($sort, $streams);
        return Ok(Box::pin(SortPreservingMergeStream::new(
            Box::new(streams),
            $schema,
            $tracking_metrics,
            $batch_size,
        )));
    }};
}

/// Perform a streaming merge of [`SendableRecordBatchStream`]
/// 将多个数据流聚合
pub(crate) fn streaming_merge(
    streams: Vec<SendableRecordBatchStream>,
    schema: SchemaRef,
    expressions: &[PhysicalSortExpr],  // 需要按这个顺序输出流
    tracking_metrics: MemTrackingMetrics,
    batch_size: usize,
) -> Result<SendableRecordBatchStream> {
    // Special case single column comparisons with optimized cursor implementations
    // 只有一个排序列
    if expressions.len() == 1 {
        let sort = expressions[0].clone();
        let data_type = sort.expr.data_type(schema.as_ref())?;
        downcast_primitive! {
            data_type => (primitive_merge_helper, sort, streams, schema, tracking_metrics, batch_size),
            DataType::Utf8 => merge_helper!(StringArray, sort, streams, schema, tracking_metrics, batch_size)
            DataType::LargeUtf8 => merge_helper!(LargeStringArray, sort, streams, schema, tracking_metrics, batch_size)
            DataType::Binary => merge_helper!(BinaryArray, sort, streams, schema, tracking_metrics, batch_size)
            DataType::LargeBinary => merge_helper!(LargeBinaryArray, sort, streams, schema, tracking_metrics, batch_size)
            _ => {}
        }
    }

    // 使用多个排序列
    let streams = RowCursorStream::try_new(schema.as_ref(), expressions, streams)?;

    // 无论有多少排序列  最终都是生成SortPreservingMergeStream对象
    Ok(Box::pin(SortPreservingMergeStream::new(
        Box::new(streams),
        schema,
        tracking_metrics,
        batch_size,
    )))
}

/// A fallible [`PartitionedStream`] of [`Cursor`] and [`RecordBatch`]
type CursorStream<C> = Box<dyn PartitionedStream<Output = Result<(C, RecordBatch)>>>;

/// 该对象包含了多个流  并按照顺序将他们输出
#[derive(Debug)]
struct SortPreservingMergeStream<C> {
    in_progress: BatchBuilder,

    /// The sorted input streams to merge together
    streams: CursorStream<C>,

    /// used to record execution metrics
    tracking_metrics: MemTrackingMetrics,

    /// If the stream has encountered an error   代表遇到了错误
    aborted: bool,

    /// A loser tree that always produces the minimum cursor
    ///
    /// Node 0 stores the top winner, Nodes 1..num_streams store
    /// the loser nodes
    ///
    /// This implements a "Tournament Tree" (aka Loser Tree) to keep
    /// track of the current smallest element at the top. When the top
    /// record is taken, the tree structure is not modified, and only
    /// the path from bottom to top is visited, keeping the number of
    /// comparisons close to the theoretical limit of `log(S)`.
    ///
    /// reference: <https://en.wikipedia.org/wiki/K-way_merge_algorithm#Tournament_Tree>
    loser_tree: Vec<usize>,

    /// If the most recently yielded overall winner has been replaced
    /// within the loser tree. A value of `false` indicates that the
    /// overall winner has been yielded but the loser tree has not
    /// been updated
    loser_tree_adjusted: bool,

    /// target batch size
    batch_size: usize,

    /// Vector that holds cursors for each non-exhausted input partition
    /// 维护每个stream的数据光标  因为每个stream内的数据已经提前排序好了
    cursors: Vec<Option<C>>,
}

impl<C: Cursor> SortPreservingMergeStream<C> {
    fn new(
        streams: CursorStream<C>,   // 对应一个分区流 每个分区对应内部一个stream  (本对象内部有多个stream)
        schema: SchemaRef,
        tracking_metrics: MemTrackingMetrics,
        batch_size: usize,
    ) -> Self {
        let stream_count = streams.partitions();

        Self {
            // 存储数据的容器
            in_progress: BatchBuilder::new(schema, stream_count, batch_size),
            streams,
            tracking_metrics,
            aborted: false,
            cursors: (0..stream_count).map(|_| None).collect(),
            loser_tree: vec![],
            loser_tree_adjusted: false,
            batch_size,
        }
    }

    /// If the stream at the given index is not exhausted, and the last cursor for the
    /// stream is finished, poll the stream for the next RecordBatch and create a new
    /// cursor for the stream from the returned result
    /// 加载某个stream的数据
    fn maybe_poll_stream(
        &mut self,
        cx: &mut Context<'_>,
        idx: usize,
    ) -> Poll<Result<()>> {
        // 代表该stream此时有还未消化的数据
        if self.cursors[idx].is_some() {
            // Cursor is not finished - don't need a new RecordBatch yet
            return Poll::Ready(Ok(()));
        }

        match futures::ready!(self.streams.poll_next(cx, idx)) {
            None => Poll::Ready(Ok(())),
            Some(Err(e)) => Poll::Ready(Err(e)),

            // 代表stream的数据被加载到内存中    cursor对应排序列的值    batch对应数据集
            Some(Ok((cursor, batch))) => {
                self.cursors[idx] = Some(cursor);
                self.in_progress.push_batch(idx, batch);
                Poll::Ready(Ok(()))
            }
        }
    }

    // 该方法作为读取流的入口  触发内部数据排序
    fn poll_next_inner(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<RecordBatch>>> {
        if self.aborted {
            return Poll::Ready(None);
        }
        // try to initialize the loser tree
        // 一开始 tree内部无数据
        if self.loser_tree.is_empty() {
            // Ensure all non-exhausted streams have a cursor from which
            // rows can be pulled
            // 先加载每个stream的数据
            for i in 0..self.streams.partitions() {
                if let Err(e) = ready!(self.maybe_poll_stream(cx, i)) {
                    self.aborted = true;
                    return Poll::Ready(Some(Err(e)));
                }
            }
            // 此时数据已经被加载到内存中了  可以初始化loser_tree了
            // 调用完后 会将当前cursor的所有值按照顺序在tree内排好
            self.init_loser_tree();
        }

        // NB timer records time taken on drop, so there are no
        // calls to `timer.done()` below.
        let elapsed_compute = self.tracking_metrics.elapsed_compute().clone();
        let _timer = elapsed_compute.timer();

        loop {
            // Adjust the loser tree if necessary, returning control if needed
            // 代表tree中的光标已经被更新  需要更新树的head节点
            if !self.loser_tree_adjusted {
                // winner对应的时 stream_idx
                let winner = self.loser_tree[0];

                // 确保数据已经加载
                if let Err(e) = ready!(self.maybe_poll_stream(cx, winner)) {
                    self.aborted = true;
                    return Poll::Ready(Some(Err(e)));
                }
                // 二叉堆那套
                self.update_loser_tree();
            }

            // 代表需要读取该stream的下一条记录
            let stream_idx = self.loser_tree[0];
            // 推进光标
            if self.advance(stream_idx) {
                self.loser_tree_adjusted = false;
                // 存储下一行数据所在的stream
                self.in_progress.push_row(stream_idx);
                if self.in_progress.len() < self.batch_size {
                    continue;
                }
            }

            // 当in_progress内囤积了一个batch的数据时 产生batch数据并返回
            return Poll::Ready(self.in_progress.build_record_batch().transpose());
        }
    }

    fn advance(&mut self, stream_idx: usize) -> bool {
        let slot = &mut self.cursors[stream_idx];
        match slot.as_mut() {
            Some(c) => {
                c.advance();
                // 当该数据消化完后要滞空 之后会触发下一批的拉取
                if c.is_finished() {
                    *slot = None;
                }
                true
            }
            None => false,
        }
    }

    /// Returns `true` if the cursor at index `a` is greater than at index `b`
    #[inline]
    fn is_gt(&self, a: usize, b: usize) -> bool {
        match (&self.cursors[a], &self.cursors[b]) {
            (None, _) => true,
            (_, None) => false,
            (Some(ac), Some(bc)) => ac.cmp(bc).then_with(|| a.cmp(&b)).is_gt(),
        }
    }

    /// Attempts to initialize the loser tree with one value from each
    /// non exhausted input, if possible
    /// 初始化树结构   可以看作一个堆
    fn init_loser_tree(&mut self) {
        // Init loser tree
        // 长度与stream数量一致
        self.loser_tree = vec![usize::MAX; self.cursors.len()];
        for i in 0..self.cursors.len() {
            let mut winner = i;
            // 转换为要比较的树节点   按照堆的特性 这样刚好可以排满整棵树
            let mut cmp_node = (self.cursors.len() + i) / 2;

            // 一开始node都是Max先忽略
            while cmp_node != 0 && self.loser_tree[cmp_node] != usize::MAX {
                // 当节点已经有一个非Max的值时
                let challenger = self.loser_tree[cmp_node];
                // 如果本次的值更小 会向tree的头部推进  就像二叉堆一样
                if self.is_gt(winner, challenger) {
                    self.loser_tree[cmp_node] = winner;
                    winner = challenger;
                }

                cmp_node /= 2;
            }
            self.loser_tree[cmp_node] = winner;
        }
        self.loser_tree_adjusted = true;
    }

    /// Attempts to update the loser tree, following winner replacement, if possible
    fn update_loser_tree(&mut self) {
        let mut winner = self.loser_tree[0];
        // Replace overall winner by walking tree of losers
        let mut cmp_node = (self.cursors.len() + winner) / 2;
        while cmp_node != 0 {
            let challenger = self.loser_tree[cmp_node];
            if self.is_gt(winner, challenger) {
                self.loser_tree[cmp_node] = winner;
                winner = challenger;
            }
            cmp_node /= 2;
        }
        self.loser_tree[0] = winner;
        self.loser_tree_adjusted = true;
    }
}

// 外部通过将SortPreservingMergeStream看作普通stream 来读取数据
impl<C: Cursor + Unpin> Stream for SortPreservingMergeStream<C> {
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        let poll = self.poll_next_inner(cx);
        self.tracking_metrics.record_poll(poll)
    }
}

impl<C: Cursor + Unpin> RecordBatchStream for SortPreservingMergeStream<C> {
    fn schema(&self) -> SchemaRef {
        self.in_progress.schema().clone()
    }
}
