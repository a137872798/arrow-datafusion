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

//! Signature module contains foundational types that are used to represent signatures, types,
//! and return types of functions in DataFusion.

use arrow::datatypes::DataType;

///A function's volatility, which defines the functions eligibility for certain optimizations
/// 函数的波动性/易变性  定义了函数是否适合被进行某种优化
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
pub enum Volatility {
    /// Immutable - An immutable function will always return the same output when given the same
    /// input. An example of this is [super::BuiltinScalarFunction::Cos].
    /// 不可变的，当函数被选定时 总是返回相同的值
    Immutable,
    /// Stable - A stable function may return different values given the same input across different
    /// queries but must return the same value for a given input within a query. An example of
    /// this is [super::BuiltinScalarFunction::Now].
    /// 一个稳定的函数 在给定相同输入的时候可能会产生不同的值
    Stable,
    /// Volatile - A volatile function may change the return value from evaluation to evaluation.
    /// Multiple invocations of a volatile function may return different results when used in the
    /// same query. An example of this is [super::BuiltinScalarFunction::Random].
    /// 使用相同函数的多次调用 可能会产生不同结果
    Volatile,
}

/// A function's type signature, which defines the function's supported argument types.
/// 函数签名 用于定义参数类型
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeSignature {
    /// arbitrary number of arguments of an common type out of a list of valid types  代表只要是给定的类型 允许任意数量 但是同一次调用的所有参数类型是一致的
    // A function such as `concat` is `Variadic(vec![DataType::Utf8, DataType::LargeUtf8])`
    Variadic(Vec<DataType>),
    /// arbitrary number of arguments of an arbitrary but equal type  任意数量参数 但是类型一致
    // A function such as `array` is `VariadicEqual`
    // The first argument decides the type used for coercion
    VariadicEqual,
    /// arbitrary number of arguments with arbitrary types    任意数量任意类型的参数
    VariadicAny,
    /// fixed number of arguments of an arbitrary but equal type out of a list of valid types 可能是p2中的任意类型 但是出现次数都是p1
    // A function of one argument of f64 is `Uniform(1, vec![DataType::Float64])`
    // A function of one argument of f64 or f32 is `Uniform(1, vec![DataType::Float32, DataType::Float64])`
    Uniform(usize, Vec<DataType>),
    /// exact number of arguments of an exact type   每个类型仅出现一次
    Exact(Vec<DataType>),
    /// fixed number of arguments of arbitrary types   任意类型固定数量参数
    Any(usize),
    /// One of a list of signatures 代表可能是内部的其中一种
    OneOf(Vec<TypeSignature>),
}

///The Signature of a function defines its supported input types as well as its volatility.
/// 代表函数签名
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Signature {
    /// type_signature - The types that the function accepts. See [TypeSignature] for more information.
    pub type_signature: TypeSignature,
    /// volatility - The volatility of the function. See [Volatility] for more information.
    pub volatility: Volatility,
}

impl Signature {
    /// new - Creates a new Signature from any type signature and the volatility.
    pub fn new(type_signature: TypeSignature, volatility: Volatility) -> Self {
        Signature {
            type_signature,
            volatility,
        }
    }
    /// variadic - Creates a variadic signature that represents an arbitrary number of arguments all from a type in common_types.
    /// 创建一个任意数量的参数 有给定的类型范围
    pub fn variadic(common_types: Vec<DataType>, volatility: Volatility) -> Self {
        Self {
            type_signature: TypeSignature::Variadic(common_types),
            volatility,
        }
    }
    /// variadic_equal - Creates a variadic signature that represents an arbitrary number of arguments of the same type.
    pub fn variadic_equal(volatility: Volatility) -> Self {
        Self {
            type_signature: TypeSignature::VariadicEqual,
            volatility,
        }
    }
    /// variadic_any - Creates a variadic signature that represents an arbitrary number of arguments of any type.
    pub fn variadic_any(volatility: Volatility) -> Self {
        Self {
            type_signature: TypeSignature::VariadicAny,
            volatility,
        }
    }
    /// uniform - Creates a function with a fixed number of arguments of the same type, which must be from valid_types.
    pub fn uniform(
        arg_count: usize,
        valid_types: Vec<DataType>,
        volatility: Volatility,
    ) -> Self {
        Self {
            type_signature: TypeSignature::Uniform(arg_count, valid_types),
            volatility,
        }
    }
    /// exact - Creates a signature which must match the types in exact_types in order.
    pub fn exact(exact_types: Vec<DataType>, volatility: Volatility) -> Self {
        Signature {
            type_signature: TypeSignature::Exact(exact_types),
            volatility,
        }
    }
    /// any - Creates a signature which can a be made of any type but of a specified number
    pub fn any(arg_count: usize, volatility: Volatility) -> Self {
        Signature {
            type_signature: TypeSignature::Any(arg_count),
            volatility,
        }
    }
    /// one_of Creates a signature which can match any of the [TypeSignature]s which are passed in.
    pub fn one_of(type_signatures: Vec<TypeSignature>, volatility: Volatility) -> Self {
        Signature {
            type_signature: TypeSignature::OneOf(type_signatures),
            volatility,
        }
    }
}
