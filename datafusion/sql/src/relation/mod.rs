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

use crate::planner::{ContextProvider, PlannerContext, SqlToRel};
use datafusion_common::{DataFusionError, Result};
use datafusion_expr::{LogicalPlan, LogicalPlanBuilder};
use sqlparser::ast::TableFactor;

mod join;

impl<'a, S: ContextProvider> SqlToRel<'a, S> {

    // 抛开复杂的情况 简单理解就是这里会将针对某个表的子查询包装成一个执行计划
    fn create_relation(
        &self,
        relation: TableFactor,
        planner_context: &mut PlannerContext,
    ) -> Result<LogicalPlan> {
        let (plan, alias) = match relation {
            // 代表引用了另一个表
            TableFactor::Table { name, alias, .. } => {
                // normalize name and alias
                // 对表名进行规范化解析
                let table_ref = self.object_name_to_table_reference(name)?;
                let table_name = table_ref.to_string();
                let cte = planner_context.get_cte(&table_name);
                (
                    match (
                        cte,
                        self.schema_provider.get_table_provider(table_ref.clone()),
                    ) {
                        // 这个应该是理解为复用cte的计划
                        (Some(cte_plan), _) => Ok(cte_plan.clone()),
                        // 对另一个表的引用 变成了一个查询计划
                        (_, Ok(provider)) => {
                            LogicalPlanBuilder::scan(table_ref, provider, None)?.build()
                        }
                        (None, Err(e)) => Err(e),
                    }?,
                    alias,
                )
            }

            // TODO
            TableFactor::Derived {
                subquery, alias, ..
            } => {
                let logical_plan = self.query_to_plan(*subquery, planner_context)?;
                (logical_plan, alias)
            }
            TableFactor::NestedJoin {
                table_with_joins,
                alias,
            } => (
                // 内部表也允许是join形成的临时表
                self.plan_table_with_joins(*table_with_joins, planner_context)?,
                alias,
            ),
            // @todo Support TableFactory::TableFunction?
            _ => {
                return Err(DataFusionError::NotImplemented(format!(
                    "Unsupported ast node {relation:?} in create_relation"
                )));
            }
        };

        // 这是表级别的别名
        if let Some(alias) = alias {
            self.apply_table_alias(plan, alias)
        } else {
            Ok(plan)
        }
    }
}
