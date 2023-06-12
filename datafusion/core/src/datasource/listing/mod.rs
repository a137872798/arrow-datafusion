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

//! A table that uses the `ObjectStore` listing capability
//! to get the list of files to process.

mod helpers;
mod table;
mod url;

use crate::error::Result;
use chrono::TimeZone;
use datafusion_common::ScalarValue;
use futures::Stream;
use object_store::{path::Path, ObjectMeta};
use std::pin::Pin;
use std::sync::Arc;

pub use self::url::ListingTableUrl;
pub use table::{ListingOptions, ListingTable, ListingTableConfig};

/// Stream of files get listed from object store
pub type PartitionedFileStream =
    Pin<Box<dyn Stream<Item = Result<PartitionedFile>> + Send + Sync + 'static>>;

/// Only scan a subset of Row Groups from the Parquet file whose data "midpoint"
/// lies within the [start, end) byte offsets. This option can be used to scan non-overlapping
/// sections of a Parquet file in parallel.
#[derive(Debug, Clone)]
pub struct FileRange {
    /// Range start
    pub start: i64,
    /// Range end
    pub end: i64,
}

#[derive(Debug, Clone)]
/// A single file or part of a file that should be read, along with its schema, statistics
/// A single file that should be read, along with its schema, statistics
/// and partition column values that need to be appended to each row.
/// 可分区文件
pub struct PartitionedFile {
    /// Path for the file (e.g. URL, filesystem path, etc)
    /// 对应某个文件的元数据
    pub object_meta: ObjectMeta,
    /// Values of partition columns to be appended to each row.
    ///
    /// These MUST have the same count, order, and type than the [`table_partition_cols`].
    ///
    /// You may use [`wrap_partition_value_in_dict`] to wrap them if you have used [`wrap_partition_type_in_dict`] to wrap the column type.
    ///
    ///
    /// [`wrap_partition_type_in_dict`]: crate::physical_plan::file_format::wrap_partition_type_in_dict
    /// [`wrap_partition_value_in_dict`]: crate::physical_plan::file_format::wrap_partition_value_in_dict
    /// [`table_partition_cols`]: table::ListingOptions::table_partition_cols
    /// 分区文件 如何将它们划分分区  依靠的就是这组分区键的值  值就记录在该字段中
    pub partition_values: Vec<ScalarValue>,
    /// An optional file range for a more fine-grained parallel execution
    /// 文件被拆分为多个部分  partition_values 代表影响分区的值 比如上下限？
    pub range: Option<FileRange>,
    /// An optional field for user defined per object metadata
    pub extensions: Option<Arc<dyn std::any::Any + Send + Sync>>,
}

impl PartitionedFile {
    /// Create a simple file without metadata or partition
    /// 创建空文件 里头啥都没
    pub fn new(path: String, size: u64) -> Self {
        Self {
            object_meta: ObjectMeta {
                location: Path::from(path),
                last_modified: chrono::Utc.timestamp_nanos(0),
                size: size as usize,
            },
            partition_values: vec![],
            range: None,
            extensions: None,
        }
    }

    /// Create a file range without metadata or partition
    /// 比上面多一个range
    pub fn new_with_range(path: String, size: u64, start: i64, end: i64) -> Self {
        Self {
            object_meta: ObjectMeta {
                location: Path::from(path),
                last_modified: chrono::Utc.timestamp_nanos(0),
                size: size as usize,
            },
            partition_values: vec![],
            range: Some(FileRange { start, end }),
            extensions: None,
        }
    }
}

impl From<ObjectMeta> for PartitionedFile {
    fn from(object_meta: ObjectMeta) -> Self {
        PartitionedFile {
            object_meta,
            partition_values: vec![],
            range: None,
            extensions: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::datasource::listing::ListingTableUrl;
    use datafusion_execution::object_store::{
        DefaultObjectStoreRegistry, ObjectStoreRegistry,
    };
    use object_store::local::LocalFileSystem;
    use std::sync::Arc;
    use url::Url;

    #[test]
    fn test_object_store_listing_url() {
        let listing = ListingTableUrl::parse("file:///").unwrap();
        let store = listing.object_store();
        assert_eq!(store.as_str(), "file:///");

        let listing = ListingTableUrl::parse("s3://bucket/").unwrap();
        let store = listing.object_store();
        assert_eq!(store.as_str(), "s3://bucket/");
    }

    #[test]
    fn test_get_store_hdfs() {
        let sut = DefaultObjectStoreRegistry::default();
        let url = Url::parse("hdfs://localhost:8020").unwrap();
        sut.register_store(&url, Arc::new(LocalFileSystem::new()));
        let url = ListingTableUrl::parse("hdfs://localhost:8020/key").unwrap();
        sut.get_store(url.as_ref()).unwrap();
    }

    #[test]
    fn test_get_store_s3() {
        let sut = DefaultObjectStoreRegistry::default();
        let url = Url::parse("s3://bucket/key").unwrap();
        sut.register_store(&url, Arc::new(LocalFileSystem::new()));
        let url = ListingTableUrl::parse("s3://bucket/key").unwrap();
        sut.get_store(url.as_ref()).unwrap();
    }

    #[test]
    fn test_get_store_file() {
        let sut = DefaultObjectStoreRegistry::default();
        let url = ListingTableUrl::parse("file:///bucket/key").unwrap();
        sut.get_store(url.as_ref()).unwrap();
    }

    #[test]
    fn test_get_store_local() {
        let sut = DefaultObjectStoreRegistry::default();
        let url = ListingTableUrl::parse("../").unwrap();
        sut.get_store(url.as_ref()).unwrap();
    }
}
