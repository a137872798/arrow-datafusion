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

//! Udaf module contains functions and structs supporting user-defined aggregate functions.

use crate::Expr;
use crate::{
    AccumulatorFunctionImplementation, ReturnTypeFunction, Signature, StateTypeFunction,
};
use std::fmt::{self, Debug, Formatter};
use std::sync::Arc;

/// Logical representation of a user-defined aggregate function (UDAF)
/// A UDAF is different from a UDF in that it is stateful across batches.
/// 用户定义的聚合函数
#[derive(Clone)]
pub struct AggregateUDF {
    /// name
    pub name: String,
    /// signature  描述参数的类型数量
    pub signature: Signature,
    /// Return type   返回结果类型
    pub return_type: ReturnTypeFunction,
    /// actual implementation   根据数据类型产生对应的累加器
    pub accumulator: AccumulatorFunctionImplementation,
    /// the accumulator's state's description as a function of the return type  聚合函数内部的累加器需要state的支撑 该fun可以得出state的类型
    pub state_type: StateTypeFunction,
}

impl Debug for AggregateUDF {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.debug_struct("AggregateUDF")
            .field("name", &self.name)
            .field("signature", &self.signature)
            .field("fun", &"<FUNC>")
            .finish()
    }
}

impl PartialEq for AggregateUDF {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.signature == other.signature
    }
}

impl Eq for AggregateUDF {}

impl std::hash::Hash for AggregateUDF {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        self.signature.hash(state);
    }
}

impl AggregateUDF {
    /// Create a new AggregateUDF
    pub fn new(
        name: &str,
        signature: &Signature,
        return_type: &ReturnTypeFunction,
        accumulator: &AccumulatorFunctionImplementation,
        state_type: &StateTypeFunction,
    ) -> Self {
        Self {
            name: name.to_owned(),
            signature: signature.clone(),
            return_type: return_type.clone(),
            accumulator: accumulator.clone(),
            state_type: state_type.clone(),
        }
    }

    /// creates a logical expression with a call of the UDAF
    /// This utility allows using the UDAF without requiring access to the registry.
    /// call 就是将参数与本聚合函数 组合成一个聚合表达式
    pub fn call(&self, args: Vec<Expr>) -> Expr {
        Expr::AggregateUDF {
            fun: Arc::new(self.clone()),
            args,
            filter: None,
        }
    }
}
