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

//! Hash aggregation through row format

use std::cmp::min;
use std::ops::Range;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::vec;

use ahash::RandomState;
use arrow::row::{OwnedRow, RowConverter, SortField};
use datafusion_physical_expr::hash_utils::create_hashes;
use futures::ready;
use futures::stream::{Stream, StreamExt};

use crate::execution::context::TaskContext;
use crate::execution::memory_pool::proxy::{RawTableAllocExt, VecAllocExt};
use crate::physical_plan::aggregates::{
    evaluate_group_by, evaluate_many, evaluate_optional, group_schema, AccumulatorItem,
    AggregateMode, PhysicalGroupBy, RowAccumulatorItem,
};
use crate::physical_plan::metrics::{BaselineMetrics, RecordOutput};
use crate::physical_plan::{aggregates, AggregateExpr, PhysicalExpr};
use crate::physical_plan::{RecordBatchStream, SendableRecordBatchStream};

use crate::execution::memory_pool::{MemoryConsumer, MemoryReservation};
use arrow::array::{new_null_array, Array, ArrayRef, PrimitiveArray, UInt32Builder};
use arrow::compute::{cast, filter};
use arrow::datatypes::{DataType, Schema, UInt32Type};
use arrow::{compute, datatypes::SchemaRef, record_batch::RecordBatch};
use datafusion_common::cast::as_boolean_array;
use datafusion_common::utils::get_arrayref_at_indices;
use datafusion_common::{Result, ScalarValue};
use datafusion_expr::Accumulator;
use datafusion_row::accessor::RowAccessor;
use datafusion_row::layout::RowLayout;
use datafusion_row::reader::{read_row, RowReader};
use datafusion_row::MutableRecordBatch;
use hashbrown::raw::RawTable;

/// Grouping aggregate with row-format aggregation states inside.
///
/// For each aggregation entry, we use:
/// - [Arrow-row] represents grouping keys for fast hash computation and comparison directly on raw bytes.
/// - [WordAligned] row to store aggregation state, designed to be CPU-friendly when updates over every field are often.
///
/// The architecture is the following:
///
/// 1. For each input RecordBatch, update aggregation states corresponding to all appeared grouping keys.
/// 2. At the end of the aggregation (e.g. end of batches in a partition), the accumulator converts its state to a RecordBatch of a single row
/// 3. The RecordBatches of all accumulators are merged (`concatenate` in `rust/arrow`) together to a single RecordBatch.
/// 4. The state's RecordBatch is `merge`d to a new state
/// 5. The state is mapped to the final value
///
/// [Arrow-row]: OwnedRow
/// [WordAligned]: datafusion_row::layout
/// 在分组的基础上聚合数据
pub(crate) struct GroupedHashAggregateStream {
    schema: SchemaRef,
    input: SendableRecordBatchStream,
    mode: AggregateMode,

    normal_aggr_expr: Vec<Arc<dyn AggregateExpr>>,
    /// Aggregate expressions not supporting row accumulation
    normal_aggregate_expressions: Vec<Vec<Arc<dyn PhysicalExpr>>>,
    /// Filter expression for each normal aggregate expression
    normal_filter_expressions: Vec<Option<Arc<dyn PhysicalExpr>>>,

    /// Aggregate expressions supporting row accumulation
    row_aggregate_expressions: Vec<Vec<Arc<dyn PhysicalExpr>>>,
    /// Filter expression for each row aggregate expression
    row_filter_expressions: Vec<Option<Arc<dyn PhysicalExpr>>>,
    row_accumulators: Vec<RowAccumulatorItem>,
    row_converter: RowConverter,
    row_aggr_schema: SchemaRef,
    row_aggr_layout: Arc<RowLayout>,

    group_by: PhysicalGroupBy,

    aggr_state: AggregationState,
    exec_state: ExecutionState,
    baseline_metrics: BaselineMetrics,
    random_state: RandomState,
    /// size to be used for resulting RecordBatches
    batch_size: usize,
    /// if the result is chunked into batches,
    /// last offset is preserved for continuation.
    row_group_skip_position: usize,
    /// keeps range for each accumulator in the field
    /// first element in the array corresponds to normal accumulators
    /// second element in the array corresponds to row accumulators
    indices: [Vec<Range<usize>>; 2],
}

#[derive(Debug)]
/// tracks what phase the aggregation is in   此时的聚合阶段
enum ExecutionState {
    // 还处于读取输入数据的阶段
    ReadingInput,
    // 正在产出结果
    ProducingOutput,
    // 已经完成数据聚合
    Done,
}

// 将所有聚合表达式涉及到的field打散后合成schema
fn aggr_state_schema(aggr_expr: &[Arc<dyn AggregateExpr>]) -> Result<SchemaRef> {
    let fields = aggr_expr
        .iter()
        .flat_map(|expr| expr.state_fields().unwrap().into_iter())
        .collect::<Vec<_>>();
    Ok(Arc::new(Schema::new(fields)))
}

impl GroupedHashAggregateStream {
    /// Create a new GroupedHashAggregateStream
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mode: AggregateMode,   // 聚合模式会影响到累加器的行为
        schema: SchemaRef,
        group_by: PhysicalGroupBy,   // 内部有分组列信息
        aggr_expr: Vec<Arc<dyn AggregateExpr>>,
        filter_expr: Vec<Option<Arc<dyn PhysicalExpr>>>,
        input: SendableRecordBatchStream,
        baseline_metrics: BaselineMetrics,
        batch_size: usize,
        context: Arc<TaskContext>,
        partition: usize,
    ) -> Result<Self> {
        let timer = baseline_metrics.elapsed_compute().timer();

        let mut start_idx = group_by.expr.len();
        // 这里初始化了很多容器
        let mut row_aggr_expr = vec![];
        let mut row_agg_indices = vec![];
        let mut row_aggregate_expressions = vec![];
        let mut row_filter_expressions = vec![];
        let mut normal_aggr_expr = vec![];
        let mut normal_agg_indices = vec![];
        let mut normal_aggregate_expressions = vec![];
        let mut normal_filter_expressions = vec![];
        // The expressions to evaluate the batch, one vec of expressions per aggregation.
        // Assuming create_schema() always puts group columns in front of aggregation columns, we set
        // col_idx_base to the group expression count.
        // 聚合相关的列 排在group by的列之后   根据mode的类型决定是给col包装一层cast 还是转而需要读取累加器的状态字段
        let all_aggregate_expressions =
            aggregates::aggregate_expressions(&aggr_expr, &mode, start_idx)?;
        let filter_expressions = match mode {
            AggregateMode::Partial | AggregateMode::Single => filter_expr,
            // final是第二阶段  实际上filter已经在第一阶段生效过了  所以不需要重复使用
            AggregateMode::Final | AggregateMode::FinalPartitioned => {
                vec![None; aggr_expr.len()]
            }
        };

        // expr对应某个聚合表达式  others对应该聚合表达式下的所有子表达式  filter对应该聚合表达式相关的过滤器
        for ((expr, others), filter) in aggr_expr
            .iter()
            .zip(all_aggregate_expressions.into_iter())
            .zip(filter_expressions.into_iter())
        {
            let n_fields = match mode {
                // In partial aggregation, we keep additional fields in order to successfully
                // merge aggregation results downstream.
                // 第一阶段 产生的是 state
                AggregateMode::Partial => expr.state_fields()?.len(),
                // 第二阶段 聚合自然只会产生一个字段
                _ => 1,
            };
            // Stores range of each expression:
            let aggr_range = Range {
                start: start_idx,
                end: start_idx + n_fields,
            };

            // 除了加入的vec不同外  要加入的数据是一样的
            // TODO 先不考虑行格式
            if expr.row_accumulator_supported() {
                row_aggregate_expressions.push(others);
                row_filter_expressions.push(filter.clone());
                row_agg_indices.push(aggr_range);
                row_aggr_expr.push(expr.clone());
            } else {
                normal_aggregate_expressions.push(others);
                normal_filter_expressions.push(filter.clone());
                normal_agg_indices.push(aggr_range);
                normal_aggr_expr.push(expr.clone());
            }
            // 注意 这个start_idx会不断的累加
            start_idx += n_fields;
        }

        // 先不考虑行格式   那么返回空vec
        let row_accumulators = aggregates::create_row_accumulators(&row_aggr_expr)?;
        // 如果是空  schema也是空
        let row_aggr_schema = aggr_state_schema(&row_aggr_expr)?;

        // 为分组field 生成一个schema
        let group_schema = group_schema(&schema, group_by.expr.len());
        // 看来行转换器是需要的
        let row_converter = RowConverter::new(
            group_schema  // 只需要将分组列转成行
                .fields()
                .iter()
                .map(|f| SortField::new(f.data_type().clone()))
                .collect(),
        )?;

        let row_aggr_layout = Arc::new(RowLayout::new(&row_aggr_schema));

        let name = format!("GroupedHashAggregateStream[{partition}]");

        let aggr_state = AggregationState {
            reservation: MemoryConsumer::new(name).register(context.memory_pool()),
            // 简单看作hashmap
            map: RawTable::with_capacity(0),
            // 存储每个组的状态
            group_states: Vec::with_capacity(0),
        };

        timer.done();

        // 初始状态为 等待读取数据
        let exec_state = ExecutionState::ReadingInput;

        Ok(GroupedHashAggregateStream {
            schema: Arc::clone(&schema),
            input,
            mode,
            normal_aggr_expr,
            normal_aggregate_expressions,
            normal_filter_expressions,
            row_aggregate_expressions,
            row_filter_expressions,
            row_accumulators,
            row_converter,
            row_aggr_schema,
            row_aggr_layout,
            group_by,
            aggr_state,
            exec_state,
            baseline_metrics,
            random_state: Default::default(),
            batch_size,
            row_group_skip_position: 0,
            // 每个聚合函数相关字段的range (起始偏移量到终止偏移量)
            indices: [normal_agg_indices, row_agg_indices],
        })
    }
}

impl Stream for GroupedHashAggregateStream {
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        let elapsed_compute = self.baseline_metrics.elapsed_compute().clone();

        loop {
            // exec_state 对应此时聚合执行的阶段
            match self.exec_state {
                ExecutionState::ReadingInput => {
                    // 读取内部的数据
                    match ready!(self.input.poll_next_unpin(cx)) {
                        // new batch to aggregate
                        Some(Ok(batch)) => {
                            let timer = elapsed_compute.timer();
                            // 对数据分组聚合
                            let result = self.group_aggregate_batch(batch);
                            timer.done();

                            // allocate memory
                            // This happens AFTER we actually used the memory, but simplifies the whole accounting and we are OK with
                            // overshooting a bit. Also this means we either store the whole record batch or not.
                            // 记录内存消耗量
                            let result = result.and_then(|allocated| {
                                self.aggr_state.reservation.try_grow(allocated)
                            });

                            if let Err(e) = result {
                                return Poll::Ready(Some(Err(e)));
                            }
                        }
                        // inner had error, return to caller
                        Some(Err(e)) => return Poll::Ready(Some(Err(e))),
                        // inner is done, producing output
                        None => {
                            // 当下层的数据都处理完后 进行第二阶段
                            self.exec_state = ExecutionState::ProducingOutput;
                        }
                    }
                }

                ExecutionState::ProducingOutput => {
                    let timer = elapsed_compute.timer();
                    // 产出结果
                    let result = self.create_batch_from_map();

                    timer.done();
                    self.row_group_skip_position += self.batch_size;

                    match result {
                        // made output
                        Ok(Some(result)) => {
                            let batch = result.record_output(&self.baseline_metrics);
                            return Poll::Ready(Some(Ok(batch)));
                        }
                        // end of output
                        Ok(None) => {
                            self.exec_state = ExecutionState::Done;
                        }
                        // error making output
                        Err(error) => return Poll::Ready(Some(Err(error))),
                    }
                }
                ExecutionState::Done => return Poll::Ready(None),
            }
        }
    }
}

impl RecordBatchStream for GroupedHashAggregateStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

impl GroupedHashAggregateStream {
    // Update the row_aggr_state according to groub_by values (result of group_by_expressions)
    // 基于分组列组成的行  更新分组状态
    fn update_group_state(
        &mut self,
        group_values: &[ArrayRef],  // 分组关联的列值
        allocated: &mut usize,
    ) -> Result<Vec<usize>> {
        // 将分组相关的列值 转换成了行记录
        let group_rows = self.row_converter.convert_columns(group_values)?;
        // 获取行数
        let n_rows = group_rows.num_rows();
        // 1.1 construct the key from the group values
        // 1.2 construct the mapping key if it does not exist
        // 1.3 add the row' index to `indices`

        // track which entries in `aggr_state` have rows in this batch to aggregate
        // 维护出现的group下标
        let mut groups_with_rows = vec![];

        // 1.1 Calculate the group keys for the group values
        // 每一行的hash需要存储下来
        let mut batch_hashes = vec![0; n_rows];
        // 计算每行的hash值 并存储在batch_hashes中
        create_hashes(group_values, &self.random_state, &mut batch_hashes)?;

        let AggregationState {
            map, group_states, ..
        } = &mut self.aggr_state;

        // 遍历每行 以及hash值
        for (row, hash) in batch_hashes.into_iter().enumerate() {
            // 后面的函数是用于判断 是否为想要的entry
            let entry = map.get_mut(hash, |(_hash, group_idx)| {
                // verify that a group that we are inserting with hash is
                // actually the same key value as the group in
                // existing_idx  (aka group_values @ row)
                // hash相同的情况下 判断行值是否是一样的    相同行值的行 group_idx应该是一样的 所以可以共用一个state
                let group_state = &group_states[*group_idx];
                // 判断2行是否真的一致
                group_rows.row(row) == group_state.group_by_values.row()
            });

            match entry {
                // Existing entry for this group value  代表行记录已经出现过
                Some((_hash, group_idx)) => {
                    let group_state = &mut group_states[*group_idx];

                    // 1.3
                    if group_state.indices.is_empty() {
                        groups_with_rows.push(*group_idx);
                    };

                    // 记录属于同一组的所有行号
                    group_state.indices.push_accounted(row as u32, allocated); // remember this row
                }
                //  1.2 Need to create new entry
                None => {

                    // 根据聚合表达式产生累加器
                    let accumulator_set =
                        aggregates::create_accumulators(&self.normal_aggr_expr)?;
                    // Add new entry to group_states and save newly created index
                    // 代表此时出现了一个新组 创建对应的state
                    let group_state = GroupState {
                        // 每行对应一个state 同时也只有唯一的行值
                        group_by_values: group_rows.row(row).owned(),
                        // 用0填充容器
                        aggregation_buffer: vec![
                            0;
                            self.row_aggr_layout.fixed_part_width()
                        ],
                        accumulator_set,  // 每个聚合表达式对应的累加器
                        // 记录行号
                        indices: vec![row as u32], // 1.3
                    };

                    // 此时该组对应的下标
                    let group_idx = group_states.len();

                    // NOTE: do NOT include the `GroupState` struct size in here because this is captured by
                    // `group_states` (see allocation down below)
                    // 增加的内存开销
                    *allocated += (std::mem::size_of::<u8>()
                        * group_state.group_by_values.as_ref().len())
                        + (std::mem::size_of::<u8>()
                            * group_state.aggregation_buffer.capacity())
                        + (std::mem::size_of::<u32>() * group_state.indices.capacity());

                    // Allocation done by normal accumulators
                    *allocated += (std::mem::size_of::<Box<dyn Accumulator>>()
                        * group_state.accumulator_set.capacity())
                        + group_state
                            .accumulator_set
                            .iter()
                            .map(|accu| accu.size())
                            .sum::<usize>();

                    // for hasher function, use precomputed hash value  记录该group占用的内存
                    map.insert_accounted(
                        (hash, group_idx),
                        |(hash, _group_index)| *hash,
                        allocated,
                    );

                    // 主要是扩容
                    group_states.push_accounted(group_state, allocated);

                    groups_with_rows.push(group_idx);
                }
            };
        }
        Ok(groups_with_rows)
    }

    // Update the accumulator results, according to row_aggr_state.
    #[allow(clippy::too_many_arguments)]
    fn update_accumulators(
        &mut self,
        groups_with_rows: &[usize],  // 本次处理数据集 产生的group_idx聚合
        offsets: &[usize],    // 每个group会关联多个行号 offsets记录每个组下行号的总长度
        row_values: &[Vec<ArrayRef>],  // 不支持row处理的情况下 row_values/row_filter_values 应该是没用的
        normal_values: &[Vec<ArrayRef>],  // 外层多个聚合表达式   内存每个聚合表达式下每个组相关的列数据
        row_filter_values: &[Option<ArrayRef>],
        normal_filter_values: &[Option<ArrayRef>],  // 外层对应多个聚合表达式 内层代表过滤的BooleanArray
        allocated: &mut usize,   // 记录消耗的内存
    ) -> Result<()> {
        // 2.1 for each key in this batch
        // 2.2 for each aggregation
        // 2.3 `slice` from each of its arrays the keys' values
        // 2.4 update / merge the accumulator with the values
        // 2.5 clear indices
        groups_with_rows
            .iter()
            .zip(offsets.windows(2)) // offsets相当于记录了该group下有多少行 与group是对齐的
            .try_for_each(|(group_idx, offsets)| {
                // 找到以该方式分组的state信息
                let group_state = &mut self.aggr_state.group_states[*group_idx];
                // 2.2
                // Process row accumulators TODO 忽略行模式
                self.row_accumulators
                    .iter_mut()
                    .zip(row_values.iter())
                    .zip(row_filter_values.iter())
                    .try_for_each(|((accumulator, aggr_array), filter_opt)| {
                        let values = slice_and_maybe_filter(
                            aggr_array,
                            filter_opt.as_ref(),
                            offsets,
                        )?;
                        let mut state_accessor =
                            RowAccessor::new_from_layout(self.row_aggr_layout.clone());
                        state_accessor
                            .point_to(0, group_state.aggregation_buffer.as_mut_slice());
                        match self.mode {
                            AggregateMode::Partial | AggregateMode::Single => {
                                accumulator.update_batch(&values, &mut state_accessor)
                            }
                            AggregateMode::FinalPartitioned | AggregateMode::Final => {
                                // note: the aggregation here is over states, not values, thus the merge
                                accumulator.merge_batch(&values, &mut state_accessor)
                            }
                        }
                    })?;
                // normal accumulators
                group_state
                    // 每个聚合表达式对应一个列累加器
                    .accumulator_set
                    .iter_mut()
                    .zip(normal_values.iter()) // normal_values的遍历也是对应每个聚合表达式 与accumulator_set对应
                    .zip(normal_filter_values.iter())
                    .try_for_each(|((accumulator, aggr_array), filter_opt)| {

                        // 得到的结果已经是filter处理后的结果了
                        let values = slice_and_maybe_filter(
                            aggr_array,
                            filter_opt.as_ref(),
                            offsets,
                        )?;
                        let size_pre = accumulator.size();
                        let res = match self.mode {
                            // 该模式是直接作用在累加器上
                            AggregateMode::Partial | AggregateMode::Single => {
                                accumulator.update_batch(&values)
                            }
                            // 该模式是将状态值作用在累加器上
                            AggregateMode::FinalPartitioned | AggregateMode::Final => {
                                // note: the aggregation here is over states, not values, thus the merge
                                accumulator.merge_batch(&values)
                            }
                        };
                        let size_post = accumulator.size();
                        *allocated += size_post.saturating_sub(size_pre);
                        res
                    })
                    // 2.5  本批数据已经处理完了  可以清理行号了  等下批数据会重新处理 重新记录行号  现在不清理的化会影响到下次
                    .and({
                        group_state.indices.clear();
                        Ok(())
                    })
            })?;
        Ok(())
    }

    /// Perform group-by aggregation for the given [`RecordBatch`].
    ///
    /// If successful, this returns the additional number of bytes that were allocated during this process.
    /// 对数据集进行分组聚合
    fn group_aggregate_batch(&mut self, batch: RecordBatch) -> Result<usize> {
        // Evaluate the grouping expressions:
        // 取出group by需要的列值
        let group_by_values = evaluate_group_by(&self.group_by, &batch)?;
        // Keep track of memory allocated:
        let mut allocated = 0usize;

        // Evaluate the aggregation expressions.
        // We could evaluate them after the `take`, but since we need to evaluate all
        // of them anyways, it is more performant to do it while they are together.
        // TODO
        let row_aggr_input_values =
            evaluate_many(&self.row_aggregate_expressions, &batch)?;

        // 只考虑列式存储  取出聚合需要的列数据  然后因为可能是多个聚合表达式  所以外部还有一层vec
        let normal_aggr_input_values =
            evaluate_many(&self.normal_aggregate_expressions, &batch)?;
        // TODO
        let row_filter_values = evaluate_optional(&self.row_filter_expressions, &batch)?;

        // 将过滤表达式作用在数据集上 得到一个BooleanArray 代表要保留哪些行
        let normal_filter_values =
            evaluate_optional(&self.normal_filter_expressions, &batch)?;

        // 在转换前 此时的行数
        let row_converter_size_pre = self.row_converter.size();

        // group_by_values 是一个二维容器 但是目前外层只看到一个值 猜测是有多种分组方式
        for group_values in &group_by_values {
            // 通过分组数据 更新分组状态  返回本次分组数据对应的group_idx 跟行号不一样 相同的行记录的group_idx是一样的
            let groups_with_rows =
                self.update_group_state(group_values, &mut allocated)?;

            // Collect all indices + offsets based on keys in this vec
            let mut batch_indices: UInt32Builder = UInt32Builder::with_capacity(0);
            let mut offsets = vec![0];
            let mut offset_so_far = 0;

            // 遍历出现的group_idx
            for &group_idx in groups_with_rows.iter() {
                // 将相同组下的所有行号存储在batch_indices中
                let indices = &self.aggr_state.group_states[group_idx].indices;
                batch_indices.append_slice(indices);
                offset_so_far += indices.len();
                // 将偏移量信息记录到 offsets中
                offsets.push(offset_so_far);
            }

            // 此时该array中就已经拥有了
            let batch_indices = batch_indices.finish();

            // row_aggr_input_values 先当作空的
            let row_values = get_at_indices(&row_aggr_input_values, &batch_indices)?;
            // 将行号在聚合列上对应的数据取出来  外层vec代表多个聚合表达式
            let normal_values =
                get_at_indices(&normal_aggr_input_values, &batch_indices)?;
            // TODO
            let row_filter_values =
                get_optional_filters(&row_filter_values, &batch_indices);
            // 取出这些行对应的filter结果  注意列数据的排序方式 应该跟batch_indices保持一致  也就是相同group的数据是放在一起的  即使处理前数据分散
            // 同时同组下的数量跟 offsets对应位置的值是一样的(值代表长度)
            let normal_filter_values =
                get_optional_filters(&normal_filter_values, &batch_indices);

            // 此时数据集 已经作用在聚合表达式上并产生结果了
            self.update_accumulators(
                &groups_with_rows,
                &offsets,
                &row_values,
                &normal_values,
                &row_filter_values,
                &normal_filter_values,
                &mut allocated,
            )?;
        }

        allocated += self
            .row_converter
            .size()
            .saturating_sub(row_converter_size_pre);
        Ok(allocated)
    }
}

/// The state that is built for each output group.
#[derive(Debug)]
pub struct GroupState {
    /// The actual group by values, stored sequentially
    /// 该组的行值
    group_by_values: OwnedRow,

    // Accumulator state, stored sequentially
    pub aggregation_buffer: Vec<u8>,

    // Accumulator state, one for each aggregate that doesn't support row accumulation
    pub accumulator_set: Vec<AccumulatorItem>,

    /// scratch space used to collect indices for input rows in a
    /// bach that have values to aggregate. Reset on each batch
    /// 记录属于同一组的所有行号
    pub indices: Vec<u32>,
}

/// The state of all the groups
pub struct AggregationState {
    pub reservation: MemoryReservation,

    /// Logically maps group values to an index in `group_states`
    ///
    /// Uses the raw API of hashbrown to avoid actually storing the
    /// keys in the table
    ///
    /// keys: u64 hashes of the GroupValue
    /// values: (hash, index into `group_states`)
    pub map: RawTable<(u64, usize)>,

    /// State for each group
    pub group_states: Vec<GroupState>,
}

impl std::fmt::Debug for AggregationState {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        // hashes are not store inline, so could only get values
        let map_string = "RawTable";
        f.debug_struct("AggregationState")
            .field("map", &map_string)
            .field("group_states", &self.group_states)
            .finish()
    }
}

impl GroupedHashAggregateStream {

    /// Create a RecordBatch with all group keys and accumulator' states or values.
    /// 借助第一阶段生成的数据 产生结果
    fn create_batch_from_map(&mut self) -> Result<Option<RecordBatch>> {

        // 代表上次产生的结果相当于消耗了多少行  如果首次触发 之前积累的行数是0
        let skip_items = self.row_group_skip_position;
        // 因为是按组生成结果的  skip_items > self.aggr_state.group_states.len() 生成的记录总数已经超过了总组数 也就是所有组的数据都已经被消耗 没有需要处理的数据了
        if skip_items > self.aggr_state.group_states.len() {
            return Ok(None);
        }
        // 没有相关的组数据 返回空结果集
        if self.aggr_state.group_states.is_empty() {
            let schema = self.schema.clone();
            return Ok(Some(RecordBatch::new_empty(schema)));
        }

        // 代表本轮处理到第几组
        let end_idx = min(
            skip_items + self.batch_size,
            self.aggr_state.group_states.len(),
        );
        // 拿到这些组对应的state
        let group_state_chunk = &self.aggr_state.group_states[skip_items..end_idx];

        // 这些group下没有数据 返回空
        if group_state_chunk.is_empty() {
            let schema = self.schema.clone();
            return Ok(Some(RecordBatch::new_empty(schema)));
        }

        // Buffers for each distinct group (i.e. row accumulator memories)
        // TODO 忽略行模式  也就是空容器
        let mut state_buffers = group_state_chunk
            .iter()
            .map(|gs| gs.aggregation_buffer.clone())
            .collect::<Vec<_>>();

        // 输出列 至于分组列 和 聚合(或累加器状态)列
        let output_fields = self.schema.fields();
        // Store row accumulator results (either final output or intermediate state):
        // TODO 基于行累加器产生结果
        let row_columns = match self.mode {
            AggregateMode::Partial => {
                read_as_batch(&state_buffers, &self.row_aggr_schema)
            }
            AggregateMode::Final
            | AggregateMode::FinalPartitioned
            | AggregateMode::Single => {
                let mut results = vec![];
                for (idx, acc) in self.row_accumulators.iter().enumerate() {
                    let mut state_accessor = RowAccessor::new(&self.row_aggr_schema);
                    let current = state_buffers
                        .iter_mut()
                        .map(|buffer| {
                            state_accessor.point_to(0, buffer);
                            acc.evaluate(&state_accessor)
                        })
                        .collect::<Result<Vec<_>>>()?;
                    // Get corresponding field for row accumulator
                    let field = &output_fields[self.indices[1][idx].start];
                    let result = if current.is_empty() {
                        Ok(arrow::array::new_empty_array(field.data_type()))
                    } else {
                        let item = ScalarValue::iter_to_array(current)?;
                        // cast output if needed (e.g. for types like Dictionary where
                        // the intermediate GroupByScalar type was not the same as the
                        // output
                        cast(&item, field.data_type())
                    }?;
                    results.push(result);
                }
                results
            }
        };

        // Store normal accumulator results (either final output or intermediate state):
        // 基于normal累加器产生结果
        let mut columns = vec![];

        // 对应每个聚合列在schema的下标 如果是累加器状态 可能会有多个列
        for (idx, &Range { start, end }) in self.indices[0].iter().enumerate() {
            // 遍历这些聚合用的field
            for (field_idx, field) in output_fields[start..end].iter().enumerate() {
                let current = match self.mode {
                    // 将累加器对应状态列取出来
                    AggregateMode::Partial => ScalarValue::iter_to_array(
                        group_state_chunk.iter().map(|group_state| {
                            group_state.accumulator_set[idx]
                                .state()
                                .map(|v| v[field_idx].clone())
                                .expect("Unexpected accumulator state in hash aggregate")
                        }),
                    ),
                    // 取累加器的结果
                    AggregateMode::Final
                    | AggregateMode::FinalPartitioned
                    | AggregateMode::Single => ScalarValue::iter_to_array(
                        group_state_chunk.iter().map(|group_state| {
                            group_state.accumulator_set[idx]
                                .evaluate()
                                .expect("Unexpected accumulator state in hash aggregate")
                        }),
                    ),
                }?;
                // Cast output if needed (e.g. for types like Dictionary where
                // the intermediate GroupByScalar type was not the same as the
                // output
                // 转换成目标类型
                let result = cast(&current, field.data_type())?;
                columns.push(result);
            }
        }

        // Stores the group by fields  拿到聚合列的数据后 还需要获取分组列的数据
        let group_buffers = group_state_chunk
            .iter()
            .map(|gs| gs.group_by_values.row())
            .collect::<Vec<_>>();

        // 行转列
        let mut output: Vec<ArrayRef> = self.row_converter.convert_rows(group_buffers)?;

        // The size of the place occupied by row and normal accumulators
        let extra: usize = self
            .indices
            .iter()
            .flatten()
            .map(|Range { start, end }| end - start)
            .sum();
        let empty_arr = new_null_array(&DataType::Null, 1);
        output.extend(std::iter::repeat(empty_arr).take(extra));

        // Write results of both accumulator types to the corresponding location in
        // the output schema:
        let results = [columns.into_iter(), row_columns.into_iter()];

        // 将聚合列赋值给空列
        for (outer, mut current) in results.into_iter().enumerate() {
            for &Range { start, end } in self.indices[outer].iter() {
                for item in output.iter_mut().take(end).skip(start) {
                    *item = current.next().expect("Columns cannot be empty");
                }
            }
        }
        Ok(Some(RecordBatch::try_new(self.schema.clone(), output)?))
    }
}

fn read_as_batch(rows: &[Vec<u8>], schema: &Schema) -> Vec<ArrayRef> {
    let row_num = rows.len();
    let mut output = MutableRecordBatch::new(row_num, Arc::new(schema.clone()));
    let mut row = RowReader::new(schema);

    for data in rows {
        row.point_to(0, data);
        read_row(&row, &mut output, schema);
    }

    output.output_as_columns()
}

fn get_at_indices(
    input_values: &[Vec<ArrayRef>],
    batch_indices: &PrimitiveArray<UInt32Type>,
) -> Result<Vec<Vec<ArrayRef>>> {
    input_values
        .iter()
        // 只取行号为index的记录  内层的array 对应某个聚合表达式相关的每个列值
        .map(|array| get_arrayref_at_indices(array, batch_indices))
        .collect()
}

fn get_optional_filters(
    original_values: &[Option<Arc<dyn Array>>],
    batch_indices: &PrimitiveArray<UInt32Type>,
) -> Vec<Option<Arc<dyn Array>>> {
    original_values
        .iter()
        .map(|array| {
            array.as_ref().map(|array| {
                compute::take(
                    array.as_ref(),
                    batch_indices,
                    None, // None: no index check
                )
                .unwrap()
            })
        })
        .collect()
}


fn slice_and_maybe_filter(
    aggr_array: &[ArrayRef],  // 每个聚合表达式需要的列数据
    filter_opt: Option<&Arc<dyn Array>>,   // 每个聚合表达式关联的过滤器
    offsets: &[usize],  // 当前组行开始的位置 和结束位置
) -> Result<Vec<ArrayRef>> {

    // 取到该group下 聚合所需的列值
    let sliced_arrays: Vec<ArrayRef> = aggr_array
        .iter()
        .map(|array| array.slice(offsets[0], offsets[1] - offsets[0]))
        .collect();

    // 获取对应的 BooleanArray
    let filtered_arrays = match filter_opt.as_ref() {
        Some(f) => {
            let sliced = f.slice(offsets[0], offsets[1] - offsets[0]);
            let filter_array = as_boolean_array(&sliced)?;

            sliced_arrays
                .iter()
                .map(|array| filter(array, filter_array).unwrap())
                .collect::<Vec<ArrayRef>>()
        }
        None => sliced_arrays,
    };
    Ok(filtered_arrays)
}
