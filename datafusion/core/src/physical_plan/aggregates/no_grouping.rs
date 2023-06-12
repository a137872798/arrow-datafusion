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

//! Aggregate without grouping columns

use crate::execution::context::TaskContext;
use crate::physical_plan::aggregates::{
    aggregate_expressions, create_accumulators, finalize_aggregation, AccumulatorItem,
    AggregateMode,
};
use crate::physical_plan::metrics::{BaselineMetrics, RecordOutput};
use crate::physical_plan::{RecordBatchStream, SendableRecordBatchStream};
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use datafusion_common::Result;
use datafusion_physical_expr::{AggregateExpr, PhysicalExpr};
use futures::stream::BoxStream;
use std::borrow::Cow;
use std::sync::Arc;
use std::task::{Context, Poll};

use crate::execution::memory_pool::{MemoryConsumer, MemoryReservation};
use crate::physical_plan::filter::batch_filter;
use futures::stream::{Stream, StreamExt};

/// stream struct for aggregation without grouping columns
/// 只有聚合表达式 和过滤器   没有分组参数的数据流
pub(crate) struct AggregateStream {
    // 通过流可以拿到数据集
    stream: BoxStream<'static, Result<RecordBatch>>,
    schema: SchemaRef,
}

/// Actual implementation of [`AggregateStream`].
///
/// This is wrapped into yet another struct because we need to interact with the async memory management subsystem
/// during poll. To have as little code "weirdness" as possible, we chose to just use [`BoxStream`] together with
/// [`futures::stream::unfold`]. The latter requires a state object, which is [`GroupedHashAggregateStreamV2Inner`].
struct AggregateStreamInner {
    schema: SchemaRef,
    mode: AggregateMode,
    input: SendableRecordBatchStream,   // 待处理的数据集
    baseline_metrics: BaselineMetrics,
    aggregate_expressions: Vec<Vec<Arc<dyn PhysicalExpr>>>,   // 根据聚合模式加工过的表达式  最外层的vec中每个vec对应一个聚合函数
    filter_expressions: Vec<Option<Arc<dyn PhysicalExpr>>>,   // 过滤数据集
    accumulators: Vec<AccumulatorItem>,   // 每个聚合函数对应一个累加器
    reservation: MemoryReservation,
    finished: bool,
}

impl AggregateStream {

    #[allow(clippy::too_many_arguments)]
    /// Create a new AggregateStream   不分组场景下对数据聚合   基于一个普通的数据流初始化聚合数据流
    pub fn new(
        mode: AggregateMode,   // 聚合模式
        schema: SchemaRef,
        aggr_expr: Vec<Arc<dyn AggregateExpr>>,  // 聚合表达式  包含了聚合逻辑
        filter_expr: Vec<Option<Arc<dyn PhysicalExpr>>>,   // 用于过滤数据集
        input: SendableRecordBatchStream,   // 未被聚合的数据流
        baseline_metrics: BaselineMetrics,  // 内部包含多个指标
        context: Arc<TaskContext>,
        partition: usize,   // 数据集对应的分区
    ) -> Result<Self> {
        // 不同的聚合模式 使用聚合表达式的方式也不同
        // Partial | Single 在使用前可能会先将数据转换成结果类型
        // Final | FinalPartitioned 会将每个聚合表达式展开
        let aggregate_expressions = aggregate_expressions(&aggr_expr, &mode, 0)?;

        let filter_expressions = match mode {
            AggregateMode::Partial | AggregateMode::Single => filter_expr,
            // final是第二阶段  实际上filter已经在第一阶段生效过了  所以不需要重复使用
            AggregateMode::Final | AggregateMode::FinalPartitioned => {
                vec![None; aggr_expr.len()]
            }
        };

        // 为每个表达式创建相关的累加器
        let accumulators = create_accumulators(&aggr_expr)?;

        // 注册内存消耗对象   返回的对象会记录内存的消耗情况
        let reservation = MemoryConsumer::new(format!("AggregateStream[{partition}]"))
            .register(context.memory_pool());

        let inner = AggregateStreamInner {
            schema: Arc::clone(&schema),
            mode,
            input,   // 待处理的数据集
            baseline_metrics,  // 记录计算中的各种指标
            aggregate_expressions,  // 传入加工过的聚合表达式
            filter_expressions,  // 过滤数据集的表达式
            accumulators,  // 各个聚合表达式对应的累加器
            reservation,
            finished: false,
        };

        // 产生数据流
        let stream = futures::stream::unfold(inner, |mut this| async move {
            if this.finished {
                return None;
            }

            let elapsed_compute = this.baseline_metrics.elapsed_compute();

            loop {
                // 每次读取一组数据
                let result = match this.input.next().await {
                    Some(Ok(batch)) => {
                        let timer = elapsed_compute.timer();
                        // 这组数据将会作用在各个累加器上
                        let result = aggregate_batch(
                            &this.mode,
                            batch,
                            &mut this.accumulators,
                            &this.aggregate_expressions,
                            &this.filter_expressions,
                        );

                        // 此时数据已经累加到accumulator中了
                        timer.done();

                        // allocate memory
                        // This happens AFTER we actually used the memory, but simplifies the whole accounting and we are OK with
                        // overshooting a bit. Also this means we either store the whole record batch or not.
                        match result
                            .and_then(|allocated| this.reservation.try_grow(allocated))
                        {
                            Ok(_) => continue,
                            Err(e) => Err(e),
                        }
                    }
                    Some(Err(e)) => Err(e),
                    None => {
                        // 此时已经拉取完所有数据了
                        this.finished = true;
                        let timer = this.baseline_metrics.elapsed_compute().timer();

                        // 之前的数据已经经过累加器的处理了  现在就是要把数据导出来
                        // 当agg mode 为Partial 时  返回累加器的状态字段   其余mode 返回累加结果
                        let result = finalize_aggregation(&this.accumulators, &this.mode)
                            .and_then(|columns| {
                                RecordBatch::try_new(this.schema.clone(), columns)
                                    .map_err(Into::into)
                            })
                            .record_output(&this.baseline_metrics);

                        timer.done();

                        result
                    }
                };

                this.finished = true;
                // 返回聚合结果
                return Some((result, this));
            }
        });

        // seems like some consumers call this stream even after it returned `None`, so let's fuse the stream.
        let stream = stream.fuse();
        let stream = Box::pin(stream);

        Ok(Self { schema, stream })
    }
}

impl Stream for AggregateStream {
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        let this = &mut *self;
        // 代理到内部的流 拉取数据
        this.stream.poll_next_unpin(cx)
    }
}

impl RecordBatchStream for AggregateStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

/// Perform group-by aggregation for the given [`RecordBatch`].
///
/// If successful, this returns the additional number of bytes that were allocated during this process.
///
/// 每拉取到一组数据  会通过每个聚合表达式进行聚合
fn aggregate_batch(
    mode: &AggregateMode,
    batch: RecordBatch,  // 待聚合数据
    accumulators: &mut [AccumulatorItem],  // 每个聚合表达式 对应一个累加器
    expressions: &[Vec<Arc<dyn PhysicalExpr>>],  // 根据mode加工后的聚合表达式
    filters: &[Option<Arc<dyn PhysicalExpr>>],  // 每个聚合表达式对应一个过滤器
) -> Result<usize> {
    let mut allocated = 0usize;

    // 1.1 iterate accumulators and respective expressions together
    // 1.2 filter the batch if necessary
    // 1.3 evaluate expressions
    // 1.4 update / merge accumulators with the expressions' values

    // 1.1
    accumulators
        .iter_mut()  // 遍历每个累加器
        .zip(expressions)  // 将每个累加器关联的聚合函数的子表达式接上
        .zip(filters)     // 再拼接上对应的过滤器
        .try_for_each(|((accum, expr), filter)| {
            // 1.2 将过滤器作用在结果集上
            let batch = match filter {
                Some(filter) => Cow::Owned(batch_filter(&batch, filter)?),
                None => Cow::Borrowed(&batch),
            };
            // 1.3 提取出聚合需要的相关列值
            let values = &expr
                .iter()
                .map(|e| e.evaluate(&batch))  // 相当于聚合函数从过滤后的结果集中提取关键列
                .map(|r| r.map(|v| v.into_array(batch.num_rows())))
                .collect::<Result<Vec<_>>>()?;

            // 1.4
            let size_pre = accum.size();  // 此前累加器中已经累计的内存量
            let res = match mode {
                // 采用不同聚合方式
                AggregateMode::Partial | AggregateMode::Single => {
                    accum.update_batch(values)
                }
                // 这种聚合模式下  获取的是状态列 而不是数据列 每个累加器的状态不一样 但是都能够正确利用
                AggregateMode::Final | AggregateMode::FinalPartitioned => {
                    accum.merge_batch(values)
                }
            };
            let size_post = accum.size();
            allocated += size_post.saturating_sub(size_pre);
            res
        })?;

    Ok(allocated)
}
