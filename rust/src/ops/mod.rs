//! **STUB — Mouse owns the contents of this directory.**
//!
//! Per-family ONNX op handlers. One op = one claim predicate + one translate handler + one row in
//! [`crate::registry::REGISTRY`]. Adding an op must never require an edit to `ep.rs`,
//! `factory.rs`, `sys.rs`, or the engine.
//!
//! # The two rules that apply to everything under this directory
//!
//! Both are enforced by `rust/tests/layering.rs`, which fails the build (not the review) on a
//! violation. See `DESIGN.md` §4.2.
//!
//! 1. **No ORT ABI.** Nothing here may name `crate::sys`, an `Ort*` type, or an `OrtApi` function
//!    pointer. Handlers see [`crate::engine::NodeDesc`], [`crate::engine::TensorRef`],
//!    [`crate::engine::OutRef`] and [`crate::registry::NodeView`] — never `OrtNode`, `OrtValue`,
//!    `OrtKernelContext` or `OrtStatus`.
//!
//! 2. **No raw Vulkan.** Nothing here may name `ash`, `vk::`, or a Vulkan handle, and nothing here
//!    may contain the token `unsafe`. Handlers express intent through
//!    [`crate::engine::DispatchContext`] and [`crate::engine::KernelRequest`]; the engine owns
//!    descriptor sets, barriers, pipelines and submission.
//!
//! Why this is worth rejecting a working PR over: the MLX reference inherited a backend that
//! already handled memory, scheduling and dtypes. We do not — every op here is a shader, a
//! descriptor layout, a barrier and a workgroup calculation. If those details bleed into sixty op
//! modules, the first driver quirk becomes a sixty-file change instead of a one-file change.
//!
//! # Shape of a handler (M0, Mouse)
//!
//! ```ignore
//! use crate::engine::{DispatchContext, EpResult, KernelRequest, NodeDesc};
//! use crate::registry::{DeclineReason, NodeView};
//!
//! pub fn claim_add(view: &NodeView<'_>) -> Result<(), DeclineReason> {
//!     if view.num_inputs() != 2 {
//!         return Err("Add: expected exactly 2 inputs".into());
//!     }
//!     Ok(())
//! }
//!
//! pub fn translate_add(node: &NodeDesc, ctx: &mut dyn DispatchContext) -> EpResult<()> {
//!     let a = ctx.resolve(&node.inputs[0])?;
//!     let b = ctx.resolve(&node.inputs[1])?;
//!     let out = ctx.bind_output(&node.outputs[0], /* desc */ todo!())?;
//!     ctx.dispatch(KernelRequest {
//!         shader: "elementwise_binary",
//!         spec_constants: vec![],
//!         push_constants: vec![],
//!         bindings: vec![a, b, out],
//!         workgroups: [1, 1, 1],
//!     })
//! }
//! ```
//!
//! Then one row in `registry::REGISTRY`: `("Add", crate::ops::elementwise::claim_add)`.

// M0 op families, per DESIGN.md §8.2. Mouse creates these files:
//
// pub mod elementwise;   // Add, Sub, Mul, Div, ... (M0 starts here, with Add)
// pub mod math;
// pub mod reduction;
// pub mod shape;
// pub mod matmul;
// pub mod norm;
