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

use crate::var_provider::is_system_variables;
use crate::{
    execution_props::ExecutionProps,
    expressions::{
        self, binary, like, Column, DateTimeIntervalExpr, GetIndexedFieldExpr, Literal,
    },
    functions, udf,
    var_provider::VarType,
    PhysicalExpr,
};
use arrow::datatypes::{DataType, Schema};
use datafusion_common::{DFSchema, DataFusionError, Result, ScalarValue};
use datafusion_expr::expr::Cast;
use datafusion_expr::{
    binary_expr, Between, BinaryExpr, Expr, GetIndexedField, Like, Operator, TryCast,
};
use std::sync::Arc;

/// Create a physical expression from a logical expression ([Expr]).
///
/// # Arguments
///
/// * `e` - The logical expression
/// * `input_dfschema` - The DataFusion schema for the input, used to resolve `Column` references
///                      to qualified or unqualified fields by name.
/// * `input_schema` - The Arrow schema for the input, used for determining expression data types
///                    when performing type coercion.
/// 产生一个物理expr
pub fn create_physical_expr(
    e: &Expr,
    input_dfschema: &DFSchema,
    input_schema: &Schema,
    execution_props: &ExecutionProps,  // 一些辅助信息
) -> Result<Arc<dyn PhysicalExpr>> {
    if input_schema.fields.len() != input_dfschema.fields().len() {
        return Err(DataFusionError::Internal(format!(
            "create_physical_expr expected same number of fields, got \
                     Arrow schema with {}  and DataFusion schema with {}",
            input_schema.fields.len(),
            input_dfschema.fields().len()
        )));
    }
    match e {
        // 屏蔽了别名的影响吗
        Expr::Alias(expr, ..) => Ok(create_physical_expr(
            expr,
            input_dfschema,
            input_schema,
            execution_props,
        )?),
        Expr::Column(c) => {
            let idx = input_dfschema.index_of_column(c)?;
            Ok(Arc::new(Column::new(&c.name, idx)))
        }
        // 将标量包装成物理计划
        Expr::Literal(value) => Ok(Arc::new(Literal::new(value.clone()))),
        // 对应一个标量类型 以及名字
        Expr::ScalarVariable(_, variable_names) => {
            // @@ 开头的就是系统标量
            if is_system_variables(variable_names) {
                // 从提供的变量中找到 系统变量
                match execution_props.get_var_provider(VarType::System) {
                    Some(provider) => {
                        // 获取系统标量 并包装成常量返回
                        let scalar_value = provider.get_value(variable_names.clone())?;
                        Ok(Arc::new(Literal::new(scalar_value)))
                    }
                    _ => Err(DataFusionError::Plan(
                        "No system variable provider found".to_string(),
                    )),
                }
            } else {
                // 同上 只是varType不同
                match execution_props.get_var_provider(VarType::UserDefined) {
                    Some(provider) => {
                        let scalar_value = provider.get_value(variable_names.clone())?;
                        Ok(Arc::new(Literal::new(scalar_value)))
                    }
                    _ => Err(DataFusionError::Plan(
                        "No user defined variable provider found".to_string(),
                    )),
                }
            }
        }
        // IsTrue 用于判断内部的表达式是否为true
        Expr::IsTrue(expr) => {
            // 将本表达式 与 True进行合并  变成一个二元表达式
            let binary_op = binary_expr(
                expr.as_ref().clone(),
                Operator::IsNotDistinctFrom,
                Expr::Literal(ScalarValue::Boolean(Some(true))),
            );
            create_physical_expr(
                &binary_op,
                input_dfschema,
                input_schema,
                execution_props,
            )
        }
        Expr::IsNotTrue(expr) => {
            let binary_op = binary_expr(
                expr.as_ref().clone(),
                Operator::IsDistinctFrom,
                Expr::Literal(ScalarValue::Boolean(Some(true))),
            );
            create_physical_expr(
                &binary_op,
                input_dfschema,
                input_schema,
                execution_props,
            )
        }
        Expr::IsFalse(expr) => {
            let binary_op = binary_expr(
                expr.as_ref().clone(),
                Operator::IsNotDistinctFrom,
                Expr::Literal(ScalarValue::Boolean(Some(false))),
            );
            create_physical_expr(
                &binary_op,
                input_dfschema,
                input_schema,
                execution_props,
            )
        }
        Expr::IsNotFalse(expr) => {
            let binary_op = binary_expr(
                expr.as_ref().clone(),
                Operator::IsDistinctFrom,
                Expr::Literal(ScalarValue::Boolean(Some(false))),
            );
            create_physical_expr(
                &binary_op,
                input_dfschema,
                input_schema,
                execution_props,
            )
        }
        Expr::IsUnknown(expr) => {
            let binary_op = binary_expr(
                expr.as_ref().clone(),
                Operator::IsNotDistinctFrom,
                Expr::Literal(ScalarValue::Boolean(None)),
            );
            create_physical_expr(
                &binary_op,
                input_dfschema,
                input_schema,
                execution_props,
            )
        }
        Expr::IsNotUnknown(expr) => {
            let binary_op = binary_expr(
                expr.as_ref().clone(),
                Operator::IsDistinctFrom,
                Expr::Literal(ScalarValue::Boolean(None)),
            );
            create_physical_expr(
                &binary_op,
                input_dfschema,
                input_schema,
                execution_props,
            )
        }

        // 以上都是一个套路 转变成BinaryExpr

        Expr::BinaryExpr(BinaryExpr { left, op, right }) => {
            // Create physical expressions for left and right operands
            // 分别处理左右
            let lhs = create_physical_expr(
                left,
                input_dfschema,
                input_schema,
                execution_props,
            )?;
            let rhs = create_physical_expr(
                right,
                input_dfschema,
                input_schema,
                execution_props,
            )?;
            // Match the data types and operator to determine the appropriate expression, if
            // they are supported temporal types and operations, create DateTimeIntervalExpr,
            // else create BinaryExpr.
            match (
                lhs.data_type(input_schema)?,
                op,
                rhs.data_type(input_schema)?,
            ) {
                (
                    DataType::Date32 | DataType::Date64 | DataType::Timestamp(_, _),
                    Operator::Plus | Operator::Minus,
                    DataType::Interval(_),
                ) => Ok(Arc::new(DateTimeIntervalExpr::try_new(
                    lhs,
                    *op,
                    rhs,
                    input_schema,
                )?)),
                (
                    DataType::Interval(_),
                    Operator::Plus | Operator::Minus,
                    DataType::Date32 | DataType::Date64 | DataType::Timestamp(_, _),
                ) => Ok(Arc::new(DateTimeIntervalExpr::try_new(
                    rhs,
                    *op,
                    lhs,
                    input_schema,
                )?)),
                (
                    DataType::Timestamp(_, _),
                    Operator::Minus,
                    DataType::Timestamp(_, _),
                ) => Ok(Arc::new(DateTimeIntervalExpr::try_new(
                    lhs,
                    *op,
                    rhs,
                    input_schema,
                )?)),
                (
                    DataType::Interval(_),
                    Operator::Plus | Operator::Minus,
                    DataType::Interval(_),
                ) => Ok(Arc::new(DateTimeIntervalExpr::try_new(
                    lhs,
                    *op,
                    rhs,
                    input_schema,
                )?)),
                // TODO 忽略时间表达式相关的
                _ => {
                    // Note that the logical planner is responsible
                    // for type coercion on the arguments (e.g. if one
                    // argument was originally Int32 and one was
                    // Int64 they will both be coerced to Int64).
                    //
                    // There should be no coercion during physical
                    // planning.
                    // 包装成一个物理的二元表达式   TODO 细节不看了   只掌握主流程
                    binary(lhs, *op, rhs, input_schema)
                }
            }
        }

        // 对应like表达式
        Expr::Like(Like {
            negated,
            expr,
            pattern,
            escape_char,
        }) => {
            if escape_char.is_some() {
                return Err(DataFusionError::Execution(
                    "LIKE does not support escape_char".to_string(),
                ));
            }
            let physical_expr = create_physical_expr(
                expr,
                input_dfschema,
                input_schema,
                execution_props,
            )?;
            let physical_pattern = create_physical_expr(
                pattern,
                input_dfschema,
                input_schema,
                execution_props,
            )?;

            // 将内部转换成物理表达式后  外部套上一层like  具体的正则匹配逻辑在 arrow-string包下 先不看
            like(
                *negated,
                false,
                physical_expr,
                physical_pattern,
                input_schema,
            )
        }
        Expr::ILike(Like {
            negated,
            expr,
            pattern,
            escape_char,
        }) => {
            if escape_char.is_some() {
                return Err(DataFusionError::Execution(
                    "ILIKE does not support escape_char".to_string(),
                ));
            }
            let physical_expr = create_physical_expr(
                expr,
                input_dfschema,
                input_schema,
                execution_props,
            )?;
            let physical_pattern = create_physical_expr(
                pattern,
                input_dfschema,
                input_schema,
                execution_props,
            )?;
            like(
                *negated,
                true,
                physical_expr,
                physical_pattern,
                input_schema,
            )
        }

        // case when
        Expr::Case(case) => {
            let expr: Option<Arc<dyn PhysicalExpr>> = if let Some(e) = &case.expr {
                Some(create_physical_expr(
                    e.as_ref(),
                    input_dfschema,
                    input_schema,
                    execution_props,
                )?)
            } else {
                None
            };
            let when_expr = case
                .when_then_expr
                .iter()
                .map(|(w, _)| {
                    create_physical_expr(
                        w.as_ref(),
                        input_dfschema,
                        input_schema,
                        execution_props,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            let then_expr = case
                .when_then_expr
                .iter()
                .map(|(_, t)| {
                    create_physical_expr(
                        t.as_ref(),
                        input_dfschema,
                        input_schema,
                        execution_props,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            let when_then_expr: Vec<(Arc<dyn PhysicalExpr>, Arc<dyn PhysicalExpr>)> =
                when_expr
                    .iter()
                    .zip(then_expr.iter())
                    .map(|(w, t)| (w.clone(), t.clone()))
                    .collect();
            let else_expr: Option<Arc<dyn PhysicalExpr>> =
                if let Some(e) = &case.else_expr {
                    Some(create_physical_expr(
                        e.as_ref(),
                        input_dfschema,
                        input_schema,
                        execution_props,
                    )?)
                } else {
                    None
                };

            // 就是分别转换内部表达式  并重新组合
            Ok(expressions::case(expr, when_then_expr, else_expr)?)
        }

        // cast的逻辑由 arrow-cast实现
        Expr::Cast(Cast { expr, data_type }) => expressions::cast(
            create_physical_expr(expr, input_dfschema, input_schema, execution_props)?,
            input_schema,
            data_type.clone(),
        ),
        Expr::TryCast(TryCast { expr, data_type }) => expressions::try_cast(
            create_physical_expr(expr, input_dfschema, input_schema, execution_props)?,
            input_schema,
            data_type.clone(),
        ),
        Expr::Not(expr) => expressions::not(create_physical_expr(
            expr,
            input_dfschema,
            input_schema,
            execution_props,
        )?),
        Expr::Negative(expr) => expressions::negative(
            create_physical_expr(expr, input_dfschema, input_schema, execution_props)?,
            input_schema,
        ),
        // 将列值变成null位图
        Expr::IsNull(expr) => expressions::is_null(create_physical_expr(
            expr,
            input_dfschema,
            input_schema,
            execution_props,
        )?),
        Expr::IsNotNull(expr) => expressions::is_not_null(create_physical_expr(
            expr,
            input_dfschema,
            input_schema,
            execution_props,
        )?),

        // 一般内部的expr是一个复合类型 (list,struct)  通过key检索到复合结构中的某个值
        Expr::GetIndexedField(GetIndexedField { key, expr }) => {
            Ok(Arc::new(GetIndexedFieldExpr::new(
                create_physical_expr(
                    expr,
                    input_dfschema,
                    input_schema,
                    execution_props,
                )?,
                key.clone(),
            )))
        }

        // 该计划对应一个函数  将表达式作为参数  触发函数
        Expr::ScalarFunction { fun, args } => {
            let physical_args = args
                .iter()
                .map(|e| {
                    create_physical_expr(e, input_dfschema, input_schema, execution_props)
                })
                .collect::<Result<Vec<_>>>()?;
            functions::create_physical_expr(
                fun,
                &physical_args,
                input_schema,
                execution_props,
            )
        }

        // 跟上面差不多
        Expr::ScalarUDF { fun, args } => {
            let mut physical_args = vec![];
            for e in args {
                physical_args.push(create_physical_expr(
                    e,
                    input_dfschema,
                    input_schema,
                    execution_props,
                )?);
            }
            // udfs with zero params expect null array as input
            if args.is_empty() {
                physical_args.push(Arc::new(Literal::new(ScalarValue::Null)));
            }
            udf::create_physical_expr(fun.clone().as_ref(), &physical_args, input_schema)
        }
        Expr::Between(Between {
            expr,
            negated,
            low,
            high,
        }) => {
            let value_expr = create_physical_expr(
                expr,
                input_dfschema,
                input_schema,
                execution_props,
            )?;
            let low_expr =
                create_physical_expr(low, input_dfschema, input_schema, execution_props)?;
            let high_expr = create_physical_expr(
                high,
                input_dfschema,
                input_schema,
                execution_props,
            )?;

            // rewrite the between into the two binary operators
            // 多个二元表达式的叠加
            let binary_expr = binary(
                binary(value_expr.clone(), Operator::GtEq, low_expr, input_schema)?,
                Operator::And,
                binary(value_expr.clone(), Operator::LtEq, high_expr, input_schema)?,
                input_schema,
            );

            if *negated {
                expressions::not(binary_expr?)
            } else {
                binary_expr
            }
        }
        Expr::InList {
            expr,
            list,
            negated,
        } => match expr.as_ref() {
            Expr::Literal(ScalarValue::Utf8(None)) => {
                Ok(expressions::lit(ScalarValue::Boolean(None)))
            }
            _ => {
                let value_expr = create_physical_expr(
                    expr,
                    input_dfschema,
                    input_schema,
                    execution_props,
                )?;

                let list_exprs = list
                    .iter()
                    .map(|expr| {
                        create_physical_expr(
                            expr,
                            input_dfschema,
                            input_schema,
                            execution_props,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;
                expressions::in_list(value_expr, list_exprs, negated, input_schema)
            }
        },
        other => Err(DataFusionError::NotImplemented(format!(
            "Physical plan does not support logical expression {other:?}"
        ))),
    }
}
