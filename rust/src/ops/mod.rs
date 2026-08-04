//! Per-family ONNX op handlers, and the machinery that makes adding one a table row.
//!
//! **Mouse owns this directory.**
//!
//! # The two rules that apply to everything under here
//!
//! Both are enforced by `rust/tests/layering.rs`, which fails the build (not the review) on a
//! violation. See `DESIGN.md` §4.2.
//!
//! 1. **No ORT ABI.** Nothing here may name the raw graph types or an ABI function pointer.
//!    Handlers see [`crate::engine::NodeDesc`], [`crate::engine::TensorRef`],
//!    [`crate::engine::OutRef`] and [`crate::registry::NodeView`].
//!
//! 2. **No raw Vulkan.** Nothing here may name a Vulkan handle or contain the token for an
//!    unchecked block. Handlers express intent through [`crate::engine::DispatchContext`] and
//!    [`crate::engine::KernelRequest`]; the engine owns descriptor sets, barriers, pipelines and
//!    submission.
//!
//! Why this is worth rejecting a working change over: the MLX reference inherited a backend that
//! already handled memory, scheduling and dtypes. We do not — every op here is a shader, a
//! descriptor layout, a barrier and a workgroup calculation. If those details bleed into sixty op
//! modules, the first driver quirk becomes a sixty-file change instead of a one-file change.
//!
//! # How an op is added
//!
//! Not by writing a module. By writing **one row**:
//!
//! ```ignore
//! //  op    domain  opset window     caps     kernel                    claim             translate             status
//! "Add",    Ai,     7 ..= OPSET_ANY, NUMERIC, kernel!(EwBinary, "add"), claim::ew_binary, templates::ew_binary, Staged(NO_SHADER);
//! ```
//!
//! That row simultaneously:
//!
//! * registers the op under its domain-qualified name and opset window;
//! * tells the shared claim predicate which dtypes to accept — and therefore which to decline,
//!   with a machine-readable reason;
//! * names the GLSL template and generates the `ew_binary_add_{f32,f16,i64,i32}` variant stems;
//! * puts those variants into the build manifest [`common::variants::manifest`] that the shader
//!   build consumes;
//! * and points at the shared handler that does the broadcasting, binding and dispatch.
//!
//! Nothing in the boundary layer or the engine changes. That property is the entire schedule
//! argument in `OP_COVERAGE.md` §5.6.
//!
//! # Ordering
//!
//! The machinery lands before op #1 on purpose. Hand-writing the first twenty ops and refactoring
//! afterwards costs the leverage *and* the schedule, because the refactor never happens once
//! twenty conformance tests depend on twenty bespoke shapes.

/// Shared helpers: dtype sets, claim predicates, shape planning, shader variants, templates.
pub mod common;

/// Tier-1 elementwise ops — the table that exercises the machinery.
pub mod elementwise;

/// Fused attention: `GroupQueryAttention`, `RotaryEmbedding`, `MultiHeadAttention`.
pub mod attention;

/// The RMSNorm family the LLM path runs on.
pub mod norm;

/// Indexing — `Gather` and the ops that read a tensor through another tensor's values.
pub mod indexing;

/// Convolution — `Conv`, the spatial-window op the CNN model class rests on.
pub mod conv;

/// Weight-only quantization: `MatMulNBits` and the block-quantized dequant path.
pub mod quant;

/// Mixture of experts: `QMoE`, then `MoE`.
pub mod moe;

/// Linear attention and SSM state — the Qwen3.5 hybrid path.
pub mod ssm;

/// The minimum-viable-subgraph rule and the partitioning metrics.
pub mod partition;

/// The machine-readable record of claim decisions that tests and measurement read.
pub mod claim_log;
