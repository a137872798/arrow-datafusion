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

use crate::{Expr, LogicalPlan};
use arrow::datatypes::SchemaRef;
use datafusion_common::Result;
use std::any::Any;

///! Table source

/// Indicates whether and how a filter expression can be handled by a
/// TableProvider for table scans.
/// 描述条件语句如何作用在表上
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TableProviderFilterPushDown {
    /// The expression cannot be used by the provider.    表不支持表达式查询
    Unsupported,
    /// The expression can be used to help minimise the data retrieved,
    /// but the provider cannot guarantee that all returned tuples
    /// satisfy the filter. The Filter plan node containing this expression
    /// will be preserved.
    /// 可以帮助命中有效数据 但是无法保证所有返回数据一定满足表达式
    Inexact,
    /// The provider guarantees that all returned data satisfies this
    /// filter expression. The Filter plan node containing this expression
    /// will be removed.
    /// 确保返回数据都满足表达式
    Exact,
}

/// Indicates the type of this table for metadata/catalog purposes.
/// 描述表类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableType {
    /// An ordinary physical table.  物理表
    Base,
    /// A non-materialised table that itself uses a query internally to provide data.  视图
    View,
    /// A transient table.  内存表
    Temporary,
}

/// The TableSource trait is used during logical query planning and optimizations and
/// provides access to schema information and filter push-down capabilities. This trait
/// provides a subset of the functionality of the TableProvider trait in the core
/// datafusion crate. The TableProvider trait provides additional capabilities needed for
/// physical query execution (such as the ability to perform a scan). The reason for
/// having two separate traits is to avoid having the logical plan code be dependent
/// on the DataFusion execution engine. Other projects may want to use DataFusion's
/// logical plans and have their own execution engine.
/// 将它与tableProvides分开  是为了在logical plans 和execution engine上分开
pub trait TableSource: Sync + Send {
    fn as_any(&self) -> &dyn Any;

    /// Get a reference to the schema for this table
    /// 获取表的元数据信息
    fn schema(&self) -> SchemaRef;

    /// Get the type of this table for metadata/catalog purposes.
    /// 获取表类型 默认是物理表
    fn table_type(&self) -> TableType {
        TableType::Base
    }

    /// Tests whether the table provider can make use of a filter expression
    /// to optimise data retrieval.
    /// 判断表是否支持使用该表达式 默认不支持
    #[deprecated(since = "20.0.0", note = "use supports_filters_pushdown instead")]
    fn supports_filter_pushdown(
        &self,
        _filter: &Expr,  // 这不就是sql吗...
    ) -> Result<TableProviderFilterPushDown> {
        Ok(TableProviderFilterPushDown::Unsupported)
    }

    /// Tests whether the table provider can make use of any or all filter expressions
    /// to optimise data retrieval.
    #[allow(deprecated)]
    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],   // 使用一组表达式检索数据
    ) -> Result<Vec<TableProviderFilterPushDown>> {
        filters
            .iter()
            .map(|f| self.supports_filter_pushdown(f))
            .collect()
    }

    /// Get the Logical plan of this table provider, if available.
    /// 逻辑计划是归属于表的
    fn get_logical_plan(&self) -> Option<&LogicalPlan> {
        None
    }
}
