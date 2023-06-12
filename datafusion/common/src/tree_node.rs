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

//! This module provides common traits for visiting or rewriting tree nodes easily.

use std::sync::Arc;

use crate::Result;

/// Trait for tree node. It can be [`ExecutionPlan`], [`PhysicalExpr`], [`LogicalPlan`], [`Expr`], etc.
/// 应该是指sql语句在编译后  变成了一个树形结构 然后每个节点对应expr/plan等
pub trait TreeNode: Sized {
    /// Use preorder to iterate the node on the tree so that we can stop fast for some cases.
    ///
    /// [`op`] can be used to collect some info from the tree node
    ///      or do some checking for the tree node.
    /// 将op作用到该节点上
    fn apply<F>(&self, op: &mut F) -> Result<VisitRecursion>
    where
        F: FnMut(&Self) -> Result<VisitRecursion>,
    {
        match op(self)? {
            VisitRecursion::Continue => {}
            // If the recursion should skip, do not apply to its children. And let the recursion continue
            // 跳过子节点 并通知下个节点继续执行
            VisitRecursion::Skip => return Ok(VisitRecursion::Continue),
            // If the recursion should stop, do not apply to its children
            // 跳过本节点 并通知下个节点终止执行
            VisitRecursion::Stop => return Ok(VisitRecursion::Stop),
        };

        // 如果有子节点 作用到子节点上
        self.apply_children(&mut |node| node.apply(op))
    }

    /// Visit the tree node using the given [TreeNodeVisitor]
    /// It performs a depth first walk of an node and its children.
    ///
    /// For an node tree such as
    /// ```text
    /// ParentNode
    ///    left: ChildNode1
    ///    right: ChildNode2
    /// ```
    ///
    /// The nodes are visited using the following order
    /// ```text
    /// pre_visit(ParentNode)
    /// pre_visit(ChildNode1)
    /// post_visit(ChildNode1)
    /// pre_visit(ChildNode2)
    /// post_visit(ChildNode2)
    /// post_visit(ParentNode)
    /// ```
    ///
    /// If an Err result is returned, recursion is stopped immediately
    ///
    /// If [`VisitRecursion::Stop`] is returned on a call to pre_visit, no
    /// children of that node will be visited, nor is post_visit
    /// called on that node. Details see [`TreeNodeVisitor`]
    ///
    /// If using the default [`post_visit`] with nothing to do, the [`apply`] should be preferred
    /// 通过观察者模式处理
    fn visit<V: TreeNodeVisitor<N = Self>>(
        &self,
        visitor: &mut V,
    ) -> Result<VisitRecursion> {
        // 前置处理
        match visitor.pre_visit(self)? {
            VisitRecursion::Continue => {}
            // If the recursion should skip, do not apply to its children. And let the recursion continue
            VisitRecursion::Skip => return Ok(VisitRecursion::Continue),
            // If the recursion should stop, do not apply to its children
            VisitRecursion::Stop => return Ok(VisitRecursion::Stop),
        };

        // 作用到子节点上  注意对内部子节点也是采用前置/后置处理
        match self.apply_children(&mut |node| node.visit(visitor))? {
            VisitRecursion::Continue => {}
            // If the recursion should skip, do not apply to its children. And let the recursion continue
            // 子节点处理后返回skip 就代表跳过后置处理
            VisitRecursion::Skip => return Ok(VisitRecursion::Continue),
            // If the recursion should stop, do not apply to its children
            VisitRecursion::Stop => return Ok(VisitRecursion::Stop),
        }

        // 后置处理
        visitor.post_visit(self)
    }

    /// Convenience utils for writing optimizers rule: recursively apply the given `op` to the node tree.
    /// When `op` does not apply to a given node, it is left unchanged.
    /// The default tree traversal direction is transform_up(Postorder Traversal).
    /// 通过函数对节点进行转换   返回结果表示 可能转换成功 可能转换失败
    fn transform<F>(self, op: &F) -> Result<Self>
    where
        F: Fn(Self) -> Result<Transformed<Self>>,
    {
        self.transform_up(op)
    }

    /// transform_down/up 区别在于先作用到自身 再作用到子节点 还是反过来

    /// Convenience utils for writing optimizers rule: recursively apply the given 'op' to the node and all of its
    /// children(Preorder Traversal).
    /// When the `op` does not apply to a given node, it is left unchanged.
    fn transform_down<F>(self, op: &F) -> Result<Self>
    where
        F: Fn(Self) -> Result<Transformed<Self>>,
    {
        // 先作用到自身 再作用到子节点
        let after_op = op(self)?.into();
        // 注意针对子节点的顺序是一致的 也是先作用到自身 再往下传递
        after_op.map_children(|node| node.transform_down(op))
    }

    /// Convenience utils for writing optimizers rule: recursively apply the given 'op' first to all of its
    /// children and then itself(Postorder Traversal).
    /// When the `op` does not apply to a given node, it is left unchanged.
    fn transform_up<F>(self, op: &F) -> Result<Self>
    where
        F: Fn(Self) -> Result<Transformed<Self>>,
    {
        // 先作用到子节点 再作用到自身  同理针对每个子节点 也是会更进一步往上个子节点递归
        let after_op_children = self.map_children(|node| node.transform_up(op))?;

        // 最后作用到自身
        let new_node = op(after_op_children)?.into();
        Ok(new_node)
    }

    /// Transform the tree node using the given [TreeNodeRewriter]
    /// It performs a depth first walk of an node and its children.
    ///
    /// For an node tree such as
    /// ```text
    /// ParentNode
    ///    left: ChildNode1
    ///    right: ChildNode2
    /// ```
    ///
    /// The nodes are visited using the following order
    /// ```text
    /// pre_visit(ParentNode)
    /// pre_visit(ChildNode1)
    /// mutate(ChildNode1)
    /// pre_visit(ChildNode2)
    /// mutate(ChildNode2)
    /// mutate(ParentNode)
    /// ```
    ///
    /// If an Err result is returned, recursion is stopped immediately
    ///
    /// If [`false`] is returned on a call to pre_visit, no
    /// children of that node will be visited, nor is mutate
    /// called on that node
    ///
    /// If using the default [`pre_visit`] with [`true`] returned, the [`transform`] should be preferred
    /// 改写节点   是从上往下的
    fn rewrite<R: TreeNodeRewriter<N = Self>>(self, rewriter: &mut R) -> Result<Self> {
        let need_mutate = match rewriter.pre_visit(&self)? {
            RewriteRecursion::Mutate => return rewriter.mutate(self),  // 执行修改并立即返回
            RewriteRecursion::Stop => return Ok(self),
            // 剩下2种情况 子节点都会被改写 区别是之后是否会修改本节点
            RewriteRecursion::Continue => true,
            RewriteRecursion::Skip => false,
        };

        // 改写所有子节点
        let after_op_children = self.map_children(|node| node.rewrite(rewriter))?;

        // now rewrite this node itself
        if need_mutate {
            rewriter.mutate(after_op_children)
        } else {
            Ok(after_op_children)
        }
    }

    /// Apply the closure `F` to the node's children
    /// 作用到每个子节点上
    fn apply_children<F>(&self, op: &mut F) -> Result<VisitRecursion>
    where
        F: FnMut(&Self) -> Result<VisitRecursion>;

    /// Apply transform `F` to the node's children, the transform `F` might have a direction(Preorder or Postorder)
    /// 将transform 作用到子节点上
    fn map_children<F>(self, transform: F) -> Result<Self>
    where
        F: FnMut(Self) -> Result<Self>;
}

/// Implements the [visitor
/// pattern](https://en.wikipedia.org/wiki/Visitor_pattern) for recursively walking [`TreeNode`]s.
///
/// [`TreeNodeVisitor`] allows keeping the algorithms
/// separate from the code to traverse the structure of the `TreeNode`
/// tree and makes it easier to add new types of tree node and
/// algorithms.
///
/// When passed to[`TreeNode::visit`], [`TreeNode::pre_visit`]
/// and [`TreeNode::post_visit`] are invoked recursively
/// on an node tree.
///
/// If an [`Err`] result is returned, recursion is stopped
/// immediately.
///
/// If [`VisitRecursion::Stop`] is returned on a call to pre_visit, no
/// children of that tree node are visited, nor is post_visit
/// called on that tree node
///
/// If [`VisitRecursion::Stop`] is returned on a call to post_visit, no
/// siblings of that tree node are visited, nor is post_visit
/// called on its parent tree node
///
/// If [`VisitRecursion::Skip`] is returned on a call to pre_visit, no
/// children of that tree node are visited.
/// 观察者  支持前置/后置处理
pub trait TreeNodeVisitor: Sized {
    /// The node type which is visitable.
    type N: TreeNode;

    /// Invoked before any children of `node` are visited.
    fn pre_visit(&mut self, node: &Self::N) -> Result<VisitRecursion>;

    /// Invoked after all children of `node` are visited. Default
    /// implementation does nothing.
    fn post_visit(&mut self, _node: &Self::N) -> Result<VisitRecursion> {
        Ok(VisitRecursion::Continue)
    }
}

/// Trait for potentially recursively transform an [`TreeNode`] node
/// tree. When passed to `TreeNode::rewrite`, `TreeNodeRewriter::mutate` is
/// invoked recursively on all nodes of a tree.
/// 节点改写器
pub trait TreeNodeRewriter: Sized {
    /// The node type which is rewritable.
    type N: TreeNode;

    /// Invoked before (Preorder) any children of `node` are rewritten /
    /// visited. Default implementation returns `Ok(Recursion::Continue)`
    fn pre_visit(&mut self, _node: &Self::N) -> Result<RewriteRecursion> {
        Ok(RewriteRecursion::Continue)
    }

    /// Invoked after (Postorder) all children of `node` have been mutated and
    /// returns a potentially modified node.
    fn mutate(&mut self, node: Self::N) -> Result<Self::N>;
}

/// Controls how the [TreeNode] recursion should proceed for [`rewrite`].
#[derive(Debug)]
pub enum RewriteRecursion {
    /// Continue rewrite this node tree.
    Continue,
    /// Call 'op' immediately and return.
    Mutate,
    /// Do not rewrite the children of this node.
    Stop,
    /// Keep recursive but skip apply op on this node
    Skip,
}

/// Controls how the [TreeNode] recursion should proceed for [`visit`].
#[derive(Debug)]
pub enum VisitRecursion {
    /// Continue the visit to this node tree.
    Continue,
    /// Keep recursive but skip applying op on the children
    Skip,
    /// Stop the visit to this node tree.
    Stop,
}

pub enum Transformed<T> {
    /// The item was transformed / rewritten somehow  已经被转换/重写
    Yes(T),
    /// The item was not transformed
    No(T),
}

impl<T> Transformed<T> {
    pub fn into(self) -> T {
        match self {
            Transformed::Yes(t) => t,
            Transformed::No(t) => t,
        }
    }

    pub fn into_pair(self) -> (T, bool) {
        match self {
            Transformed::Yes(t) => (t, true),
            Transformed::No(t) => (t, false),
        }
    }
}

/// Helper trait for implementing [`TreeNode`] that have children stored as Arc's
///
/// If some trait object, such as `dyn T`, implements this trait,
/// its related `Arc<dyn T>` will automatically implement [`TreeNode`]
/// 动态树节点
pub trait DynTreeNode {
    /// Returns all children of the specified TreeNode   返回所有子节点
    fn arc_children(&self) -> Vec<Arc<Self>>;

    /// construct a new self with the specified children  替换内部的子节点
    fn with_new_arc_children(
        &self,
        arc_self: Arc<Self>,
        new_children: Vec<Arc<Self>>,
    ) -> Result<Arc<Self>>;
}

/// Blanket implementation for Arc for any tye that implements
/// [`DynTreeNode`] (such as [`Arc<dyn PhysicalExpr>`])
/// 动态节点 也是可以将函数作用到子节点的  相比普通节点 他们可以替换子节点
impl<T: DynTreeNode + ?Sized> TreeNode for Arc<T> {
    fn apply_children<F>(&self, op: &mut F) -> Result<VisitRecursion>
    where
        F: FnMut(&Self) -> Result<VisitRecursion>,
    {
        for child in self.arc_children() {
            match op(&child)? {
                VisitRecursion::Continue => {}
                VisitRecursion::Skip => return Ok(VisitRecursion::Continue),
                VisitRecursion::Stop => return Ok(VisitRecursion::Stop),
            }
        }

        Ok(VisitRecursion::Continue)
    }

    fn map_children<F>(self, transform: F) -> Result<Self>
    where
        F: FnMut(Self) -> Result<Self>,
    {
        let children = self.arc_children();
        if !children.is_empty() {
            let new_children: Result<Vec<_>> =
                children.into_iter().map(transform).collect();
            let arc_self = Arc::clone(&self);
            // 使用转换后的新children产生一个自己的副本
            self.with_new_arc_children(arc_self, new_children?)
        } else {
            Ok(self)
        }
    }
}
