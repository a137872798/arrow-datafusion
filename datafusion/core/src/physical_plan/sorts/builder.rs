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
use arrow::compute::interleave;
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;

#[derive(Debug, Copy, Clone, Default)]
struct BatchCursor {
    /// The index into BatchBuilder::batches
    batch_idx: usize,
    /// The row index within the given batch
    row_idx: usize,
}

/// Provides an API to incrementally build a [`RecordBatch`] from partitioned [`RecordBatch`]
#[derive(Debug)]
pub struct BatchBuilder {
    /// The schema of the RecordBatches yielded by this stream
    schema: SchemaRef,

    /// Maintain a list of [`RecordBatch`] and their corresponding stream
    /// 维护每个stream 此时在内存中的数据
    batches: Vec<(usize, RecordBatch)>,

    /// The current [`BatchCursor`] for each stream
    cursors: Vec<BatchCursor>,

    /// The accumulated stream indexes from which to pull rows
    /// Consists of a tuple of `(batch_idx, row_idx)`
    /// 存储本次应当返回的数据集  每条记录通过batch+row定位
    indices: Vec<(usize, usize)>,
}

impl BatchBuilder {
    /// Create a new [`BatchBuilder`] with the provided `stream_count` and `batch_size`
    /// stream_count 对应分区数
    pub fn new(schema: SchemaRef, stream_count: usize, batch_size: usize) -> Self {
        Self {
            schema,
            batches: Vec::with_capacity(stream_count * 2),
            cursors: vec![BatchCursor::default(); stream_count],
            indices: Vec::with_capacity(batch_size),
        }
    }

    /// Append a new batch in `stream_idx`
    /// 暂存每个stream此时加载到内存中的数据
    pub fn push_batch(&mut self, stream_idx: usize, batch: RecordBatch) {
        let batch_idx = self.batches.len();
        // 数据存入vec
        self.batches.push((stream_idx, batch));
        self.cursors[stream_idx] = BatchCursor {
            batch_idx,  // 代表该流对应的数据在 batches的哪一批
            row_idx: 0,
        }
    }

    /// Append the next row from `stream_idx`
    /// 代表下一行要读取的数据所在的stream
    pub fn push_row(&mut self, stream_idx: usize) {
        let cursor = &mut self.cursors[stream_idx];
        let row_idx = cursor.row_idx;
        cursor.row_idx += 1;

        // 通过索引定位数据集中的数据
        self.indices.push((cursor.batch_idx, row_idx));
    }

    /// Returns the number of in-progress rows in this [`BatchBuilder`]
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Returns `true` if this [`BatchBuilder`] contains no in-progress rows
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Returns the schema of this [`BatchBuilder`]
    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    /// Drains the in_progress row indexes, and builds a new RecordBatch from them
    ///
    /// Will then drop any batches for which all rows have been yielded to the output
    ///
    /// Returns `None` if no pending rows
    /// 消化此时维护的索引值 从数据集中抽取列  组成新的RecordBatch
    pub fn build_record_batch(&mut self) -> Result<Option<RecordBatch>> {
        if self.is_empty() {
            return Ok(None);
        }

        // 将所有列的数据合在一起就是结果集
        let columns = (0..self.schema.fields.len())
            .map(|column_idx| {
                // 将该列在不同数据集的部分 和在一起  再通过indices来精准定位
                let arrays: Vec<_> = self
                    .batches
                    .iter()
                    .map(|(_, batch)| batch.column(column_idx).as_ref())
                    .collect();
                Ok(interleave(&arrays, &self.indices)?)
            })
            .collect::<Result<Vec<_>>>()?;

        self.indices.clear();

        // New cursors are only created once the previous cursor for the stream
        // is finished. This means all remaining rows from all but the last batch
        // for each stream have been yielded to the newly created record batch
        //
        // We can therefore drop all but the last batch for each stream
        let mut batch_idx = 0;
        let mut retained = 0;

        // 不再被需要的batch可以释放
        self.batches.retain(|(stream_idx, _)| {
            // 最新的光标使用到了该batch 就不能被丢弃
            let stream_cursor = &mut self.cursors[*stream_idx];
            let retain = stream_cursor.batch_idx == batch_idx;
            batch_idx += 1;

            if retain {
                // 要更新这些光标的下标   因为之前的indices都已经被消耗了  不用考虑他们的影响
                stream_cursor.batch_idx = retained;
                retained += 1;
            }
            retain
        });

        Ok(Some(RecordBatch::try_new(self.schema.clone(), columns)?))
    }
}
