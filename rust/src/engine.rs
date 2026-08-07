//! **STUB — the module seam Switch fills in.**
//!
//! This file defines the *vocabulary* that the three layers around the Vulkan engine share, and
//! nothing else. There is deliberately no Vulkan code here: `ash`, `VkDevice`, command buffers,
//! pipelines, descriptors and the allocator arrive with Switch's engine implementation
//! (`ENGINE.md` §1 calls this module the boundary, and the wrapper types it will add are *not*
//! `pub` outside it).
//!
//! # Who reads what
//!
//! | Consumer | Uses | Never sees |
//! |---|---|---|
//! | `ep.rs` (Tank, L0) | [`Plan`], [`NodeDesc`], [`EpError`] | Vulkan handles |
//! | `ops/*.rs` (Mouse, L2) | [`DispatchContext`], [`NodeDesc`], [`TensorRef`], [`OutRef`], [`BufferView`], [`KernelRequest`] | anything from `sys::ort`, anything from `ash` |
//! | `vk/*` (Switch, L3) | implements [`DispatchContext`] | anything from `sys::ort` |
//!
//! Both directions of that table are enforced mechanically by `tests/layering.rs`, not by review.
//!
//! # What Switch owns from here
//!
//! 1. A `vk` module tree (instance, device, caps, memory, pipeline, descriptor, command, shaders)
//!    per `DESIGN.md` §3, with everything raw kept private to it.
//! 2. A real capability probe replacing [`probe_devices`] — `vkEnumeratePhysicalDevices` plus the
//!    capability gate from `decisions.md` ("Vulkan API baseline: capability-set, not version
//!    floor"): Vulkan ≥ 1.1 core, a compute queue, `synchronization2`, `subgroup_size_control`,
//!    subgroup BASIC+ARITHMETIC in COMPUTE, `maxComputeWorkGroupInvocations ≥ 256`,
//!    `maxComputeSharedMemorySize ≥ 16 KiB`. [`DeviceInfo`] is the shape the answer must take;
//!    `factory.rs` already consumes it and needs no change.
//! 3. The concrete [`DispatchContext`] implementor that records a command buffer.
//!
//! # What Mouse owns from here
//!
//! Op handlers in `src/ops/` that read a [`NodeDesc`] and issue [`KernelRequest`]s through a
//! [`DispatchContext`]. The trait below is the *entire* vocabulary an op handler has.

// The Vulkan crates are dependencies of record from M0 (decisions.md: "Rust Vulkan crate: ash +
// gpu-allocator") so the choice is pinned in Cargo.lock before any engine code exists. Naming
// them here keeps `unused_crate_dependencies` quiet without pretending to use them.
use ash as _;
use gpu_allocator as _;

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fmt;

// -------------------------------------------------------------------------------------------
// Errors
// -------------------------------------------------------------------------------------------

/// Every failure that can occur below the ORT ABI boundary.
///
/// `ep.rs` converts these into an `OrtStatus` at the boundary; no layer below `ep.rs` ever
/// constructs an `OrtStatus`, which is layering rule 1 stated as a type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EpError {
    /// The graph asked for something we do not implement. Not a bug — the claim predicate should
    /// have declined it, so reaching this in `Compile` is an invariant violation worth logging.
    Unsupported(String),
    /// A shape, dtype or attribute combination that is invalid per the ONNX spec.
    InvalidGraph(String),
    /// The Vulkan engine failed (device lost, OOM, pipeline creation, submission).
    Engine(String),
    /// An internal invariant was violated. Always a bug in this crate.
    Internal(String),
}

impl fmt::Display for EpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EpError::Unsupported(m) => write!(f, "unsupported: {m}"),
            EpError::InvalidGraph(m) => write!(f, "invalid graph: {m}"),
            EpError::Engine(m) => write!(f, "vulkan engine error: {m}"),
            EpError::Internal(m) => write!(f, "internal error: {m}"),
        }
    }
}

impl std::error::Error for EpError {}

/// Result alias used everywhere below the ABI boundary.
pub type EpResult<T> = Result<T, EpError>;

// -------------------------------------------------------------------------------------------
// Tensor vocabulary
// -------------------------------------------------------------------------------------------

/// The element types the EP can represent. A superset of what any given device supports — the
/// capability gate decides which are actually usable (`DESIGN.md` §7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DType {
    F32,
    F16,
    I64,
    I32,
    U8,
    Bool,
}

impl DType {
    /// Size of one element in bytes.
    pub const fn byte_size(self) -> usize {
        match self {
            DType::F32 | DType::I32 => 4,
            DType::F16 => 2,
            DType::I64 => 8,
            DType::U8 | DType::Bool => 1,
        }
    }
}

/// A fully resolved tensor shape and dtype. v0 is static-shape only (`DESIGN.md` §1.2).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorDesc {
    pub dtype: DType,
    pub shape: Vec<i64>,
}

impl TensorDesc {
    pub fn new(dtype: DType, shape: Vec<i64>) -> Self {
        Self { dtype, shape }
    }

    /// Total element count. Returns `None` on overflow or on a negative (symbolic) dimension.
    pub fn element_count(&self) -> Option<usize> {
        let mut n: usize = 1;
        for &d in &self.shape {
            let d = usize::try_from(d).ok()?;
            n = n.checked_mul(d)?;
        }
        Some(n)
    }

    /// Total size in bytes, or `None` on overflow / symbolic dims.
    pub fn byte_size(&self) -> Option<usize> {
        self.element_count()?.checked_mul(self.dtype.byte_size())
    }
}

/// A reference to one of a node's *inputs*, resolved by the plan builder.
///
/// Op handlers never see a raw `OrtValue`; they see this and hand it to
/// [`DispatchContext::resolve`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorRef {
    /// Name of the value in the fused subgraph.
    pub name: String,
    /// Static description where known at compile time.
    pub desc: Option<TensorDesc>,
    /// True when this input is a graph initializer (a constant), i.e. prepackable.
    pub is_initializer: bool,
}

/// A reference to one of a node's *outputs*.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutRef {
    pub name: String,
    pub desc: Option<TensorDesc>,
}

/// An opaque handle to a buffer the engine owns.
///
/// It is deliberately *not* a `VkBuffer` and deliberately carries no offset: op code cannot
/// resolve it, bind it, free it, or reason about its memory. Only the engine can, via its private
/// side table. This is the type-level half of layering rule 2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferView(u64);

impl BufferView {
    /// Construct a handle. Engine-internal: the token is meaningless outside the engine that
    /// minted it.
    pub fn from_raw(token: u64) -> Self {
        Self(token)
    }

    /// The opaque token. Only the minting engine may interpret this.
    pub fn as_raw(self) -> u64 {
        self.0
    }
}

/// The geometry of a **prefix alias** — see [`DispatchContext::bind_prefix_output`].
///
/// The relation is stated by the handler, which knows the tensor's semantics, and *checked* by
/// the engine, which knows the byte sizes. Nothing about it is inferred from a shape at the
/// engine end: the axis a KV cache grows along is not a fact `vk/` can read off a rank.
///
/// The declared copy is `outer_blocks` regions of `src_block_bytes`, region `k` running from
/// `k * src_block_bytes` in the source to `k * dst_block_bytes` in the destination. For
/// `[B, Nkv, P, D]` fp16 into `[B, Nkv, P+S, D]` fp16 that is `B*Nkv` blocks of `P*D*2` bytes
/// at a destination stride of `(P+S)*D*2`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefixLayout {
    /// Number of contiguous source regions, i.e. the product of the dimensions outside the axis
    /// the tensor grows along.
    pub outer_blocks: u64,
    /// Bytes per source block. `outer_blocks * src_block_bytes` must equal the input's size.
    pub src_block_bytes: u64,
    /// Destination stride in bytes. Must be `>= src_block_bytes`.
    pub dst_block_bytes: u64,
}

impl PrefixLayout {
    /// Does this geometry account for exactly `in_bytes` of source and land entirely inside
    /// `out_bytes` of destination?
    ///
    /// Every arithmetic step is checked: the geometry arrives from a translate handler that
    /// computed it from ONNX shapes, and a shape that overflows `u64` must refuse rather than
    /// wrap into a plausible-looking offset.
    pub fn fits(&self, in_bytes: u64, out_bytes: u64) -> bool {
        let last_end = self
            .outer_blocks
            .checked_sub(1)
            .and_then(|k| k.checked_mul(self.dst_block_bytes))
            .and_then(|off| off.checked_add(self.src_block_bytes));
        self.outer_blocks > 0
            && self.src_block_bytes > 0
            && self.src_block_bytes <= self.dst_block_bytes
            && self.outer_blocks.checked_mul(self.src_block_bytes) == Some(in_bytes)
            && last_end.is_some_and(|e| e <= out_bytes)
    }
}

// -------------------------------------------------------------------------------------------
// Device memory for ORT-owned tensors
// -------------------------------------------------------------------------------------------

/// The seam that lets ORT's tensors live in device memory instead of host staging.
///
/// # Why this trait exists rather than a direct call
///
/// [`crate::allocator`] must not name an `ash` type and `vk/` must not name an `sys::ort` type —
/// layering rule 2, enforced by `tests/layering.rs`. The allocator needs a `VkBuffer` per span and
/// cannot say so. This trait is the only vocabulary they share: sizes, byte slices, and opaque
/// [`BufferView`] tokens.
///
/// # What an implementor owes
///
/// * [`alloc`](DeviceMemoryProvider::alloc) returns a device-local buffer of at least `size`
///   bytes, or `None` — and `None` must be *survivable*: the allocator falls back to host staging,
///   which is slower but correct. Refusing is always better than returning a buffer that is not
///   device-local, because the whole point of the counter this feeds is to distinguish the two.
/// * [`upload`](DeviceMemoryProvider::upload) and [`download`](DeviceMemoryProvider::download) are
///   **synchronous**: they must not return until the bytes are visible to the other side. ORT's
///   `CopyTensors` has no completion handle to wait on, so an asynchronous copy here would be a
///   silent race, which is the failure class this project has spent the most time on.
/// * [`free`](DeviceMemoryProvider::free) may be called at any time after `alloc`, including
///   during teardown.
pub trait DeviceMemoryProvider: Send + Sync {
    /// Allocate `size` bytes of device-local memory. `None` means "use host staging instead".
    fn alloc(&self, size: usize) -> Option<BufferView>;

    /// Release a buffer previously returned by [`alloc`](DeviceMemoryProvider::alloc).
    fn free(&self, view: BufferView);

    /// Copy host bytes into the buffer at `offset`. Must be complete on return.
    fn upload(&self, view: BufferView, offset: usize, src: &[u8]) -> Result<(), String>;

    /// Copy bytes out of the buffer at `offset` into `dst`. Must be complete on return.
    fn download(&self, view: BufferView, offset: usize, dst: &mut [u8]) -> Result<(), String>;

    /// Whether this device shares one physical heap with the host.
    ///
    /// Reported, never used to skip a copy. UMA and discrete are different performance problems
    /// and `bench/compare.py` refuses to compare them; a number that silently changes meaning
    /// between them is the same defect in a different costume.
    fn is_unified_memory(&self) -> bool;
}

static PROVIDERS: std::sync::LazyLock<
    std::sync::Mutex<
        std::collections::HashMap<usize, std::sync::Arc<dyn DeviceMemoryProvider + 'static>>,
    >,
> = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

/// Register the device-memory provider for `device_index`. Called by the engine once it has a
/// logical device; until then the allocator stages on the host, which is correct but slower.
pub fn register_device_memory_provider(
    device_index: usize,
    provider: std::sync::Arc<dyn DeviceMemoryProvider + 'static>,
) {
    if let Ok(mut m) = PROVIDERS.lock() {
        m.insert(device_index, provider);
    }
}

/// The provider for `device_index`, if the engine has stood one up yet.
pub fn device_memory_provider(
    device_index: usize,
) -> Option<std::sync::Arc<dyn DeviceMemoryProvider + 'static>> {
    PROVIDERS.lock().ok()?.get(&device_index).cloned()
}

// -------------------------------------------------------------------------------------------
// Node / plan vocabulary
// -------------------------------------------------------------------------------------------

/// An ONNX attribute value, copied generically out of the graph at `Compile` time.
#[derive(Debug, Clone, PartialEq)]
pub enum AttrValue {
    Int(i64),
    Float(f32),
    String(String),
    Ints(Vec<i64>),
    Floats(Vec<f32>),
    Strings(Vec<String>),
}

/// One node of a fused subgraph, fully detached from the ORT ABI.
///
/// This is the *only* representation `ops/` ever sees of a node. Everything in it was read out of
/// `OrtNode` once, at `Compile` time, and owned; nothing here borrows from ORT.
#[derive(Debug, Clone, Default)]
pub struct NodeDesc {
    pub op_type: String,
    pub domain: String,
    pub since_version: i32,
    pub name: String,
    pub attributes: BTreeMap<String, AttrValue>,
    pub inputs: Vec<TensorRef>,
    pub outputs: Vec<OutRef>,
}

impl NodeDesc {
    /// Domain-qualified op name, e.g. `Add` or `com.microsoft::Attention`. Used as the registry
    /// key and in diagnostics; `Attention` exists in two domains and merging them would report one
    /// count for two different ops.
    pub fn qualified_name(&self) -> Cow<'_, str> {
        if self.domain.is_empty() || self.domain == "ai.onnx" {
            Cow::Borrowed(&self.op_type)
        } else {
            Cow::Owned(format!("{}::{}", self.domain, self.op_type))
        }
    }
}

/// Compiled fused subgraph: topologically ordered nodes, I/O binding table, and any
/// prepack requests emitted by op handlers at Compile time.
///
/// Owned by the `OrtNodeComputeInfo` that `ep.rs` hands back to ORT, and dropped when ORT
/// releases it. Prepacked weight buffers and any cached command-buffer recordings hang off
/// this struct for the lifetime of the compiled subgraph.
#[derive(Default)]
pub struct Plan {
    pub nodes: Vec<NodeDesc>,
    pub inputs: Vec<TensorRef>,
    pub outputs: Vec<OutRef>,
    /// Prepack requests collected from op handlers during the Compile pass.
    ///
    /// The engine processes these after all nodes are visited: for each request it calls
    /// `pack_fn`, uploads the result to device memory, and moves the handles into `prepacked`.
    /// Callers of the engine's compile logic should drain this vec and populate `prepacked`
    /// before the Plan is handed to ORT.
    pub prepack_requests: Vec<PrepackRequest>,
    /// Results of processed prepack requests, keyed by [`PackKey`].
    ///
    /// Populated by the engine during Compile; empty until the engine processes
    /// `prepack_requests`. At Compute time, [`DispatchContext::resolve_prepacked`] looks up
    /// packed buffers here.
    pub prepacked: std::collections::HashMap<PackKey, PrepackResult>,
}

// -------------------------------------------------------------------------------------------
// Prepack seam — weight prepacking for block-quantized kernels (OP_COVERAGE.md §8.2.1)
// -------------------------------------------------------------------------------------------

/// Tile/variant configuration that determines packed weight memory layout.
///
/// Part of the [`PackKey`] cache key (`OP_COVERAGE.md` §8.2.1 P2). Distinct from
/// specialization constants because it affects *memory layout*, not just shader execution:
/// two dispatches with different tile configs would consume data in incompatible formats.
///
/// Chosen by the engine once per device during `Compile`, then fixed for the lifetime of the
/// `Plan`. This is what makes "prepack must run after device selection" true (P1).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TileConfig {
    /// Tile width for the B matrix (N-dimension, output columns).
    pub tile_n: u32,
    /// Tile height for the A matrix (M-dimension, output rows).
    pub tile_m: u32,
    /// Quantization block size; must match the `block_size` attribute of the op.
    pub block_size: u32,
}

/// Cache key for a prepacked weight result (P2).
///
/// The engine stores exactly one [`PrepackResult`] per `PackKey` in [`Plan::prepacked`].
/// An initializer shared by two nodes (e.g., a weight matrix used in both prefill and decode
/// variants) must produce the same key so it is uploaded once, not twice.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PackKey {
    /// ONNX graph value name of the initializer. Stable across `Run` calls for the same model.
    pub initializer: String,
    /// Tile/variant config that determines the packed layout.
    pub config: TileConfig,
    /// Shader variant stem (e.g. `"matmul_nbits_q4_b32_gemv"`).
    pub variant: &'static str,
}

/// Input to a pack function (P6 — pure, no Vulkan handles).
///
/// All fields are views into the original initializer bytes from ORT. The pack function reads
/// these and writes [`PackOutput`]; it never allocates device memory and never calls Vulkan.
/// This is what keeps the nibble-unpack/interleave logic inside `src/ops/**` layering rules.
pub struct PackInput<'a> {
    /// The quantized weight bytes, in ONNX layout (input 1 of `MatMulNBits`).
    pub weight: &'a [u8],
    /// Scale bytes, in ONNX layout (input 2 of `MatMulNBits`).
    pub scales: &'a [u8],
    /// Zero-point bytes, if present in the graph (input 3 of `MatMulNBits`).
    pub zero_points: Option<&'a [u8]>,
    /// The tile config that determines the packed layout.
    pub config: &'a TileConfig,
}

/// Output of a pack function (P3 — scales and zero-points as separate allocations).
///
/// Separate `Vec<u8>` for each logical tensor (P3). Interleaving would save one descriptor
/// binding but destroy the dense `uvec4` streaming that is the entire GEMV bandwidth argument.
pub struct PackOutput {
    /// The repacked weight bytes in GPU-friendly layout.
    pub packed_weight: Vec<u8>,
    /// The repacked scale bytes.
    pub packed_scales: Vec<u8>,
    /// The repacked zero-point bytes; `None` for symmetric quantization (no zero-points input).
    pub packed_zero_points: Option<Vec<u8>>,
}

/// A prepack request emitted by a kernel handler at Compile time (P1).
///
/// The engine collects these from [`CompileContext::request_prepack`] during `Compile`, then
/// — after all nodes have been visited — calls `pack_fn(input)`, uploads the result to device
/// memory, and stores the buffer handles in [`Plan::prepacked`]. The original ONNX-layout
/// initializer bytes are then droppable (P4).
///
/// `pack_fn` must live in `ops::quant::prepack` (P6): it is a pure `PackInput → PackOutput`
/// function with no Vulkan handles, keeping the nibble-unpack/interleave logic next to the
/// kernel that consumes it.
pub struct PrepackRequest {
    /// Cache key. If a matching entry already exists in `Plan::prepacked`, this request is a
    /// no-op (P2) and `pack_fn` is never called for this initializer again.
    pub key: PackKey,
    /// The quantized weight tensor (input 1 of `MatMulNBits`, etc.).
    pub weight: TensorRef,
    /// The scales tensor (input 2).
    pub scales: TensorRef,
    /// The zero-points tensor (input 3), if present. `None` for symmetric quantization.
    pub zero_points: Option<TensorRef>,
    /// The pure byte-layout transform (P6). Called exactly once per unique [`PackKey`].
    ///
    /// The function pointer type `fn(PackInput<'_>) -> PackOutput` uses an implicit HRTB:
    /// `for<'a> fn(PackInput<'a>) -> PackOutput`. Any concrete function with this signature
    /// (e.g., `ops::quant::prepack_matmul_nbits`) satisfies this type.
    pub pack_fn: fn(PackInput<'_>) -> PackOutput,
}

/// Device-buffer handles produced by processing one [`PrepackRequest`].
///
/// Stored in [`Plan::prepacked`] keyed by [`PackKey`]. At Compute time,
/// [`DispatchContext::resolve_prepacked`] looks up packed buffers here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrepackResult {
    /// The packed weight buffer (binding 0 of the quantized kernel).
    pub weight: BufferView,
    /// The scales buffer (binding 1 — separate from weight, per P3).
    pub scales: BufferView,
    /// The zero-points buffer (binding 2); `None` for symmetric quantization.
    pub zero_points: Option<BufferView>,
}

/// Context available to op handlers during the Compile phase.
///
/// Called once per claimed node, during the Compile pass, before any Compute call. Handlers
/// that need weight prepacking emit [`PrepackRequest`]s here; all other ops leave Compile
/// as a no-op via the `None` path in the registry's `compile_hook_for`.
///
/// Separate from [`DispatchContext`] because Compile and Compute are different phases:
/// Compile has access to initializer bytes and device tile-config; Compute has command-buffer
/// recording state. The two phases must not be conflated (see `ENGINE.md` §3.5.1).
pub trait CompileContext {
    /// Emit a prepack request.
    ///
    /// If a result for `req.key` already exists in `Plan::prepacked`, this is a no-op (P2).
    /// The engine processes all requests after visiting all nodes, so the emission order does
    /// not matter and emitting the same key twice is safe.
    fn request_prepack(&mut self, req: PrepackRequest) -> EpResult<()>;
}

// -------------------------------------------------------------------------------------------
// Indirect dispatch seam — QMoE / device-computed workgroup counts (OP_COVERAGE.md §9.5 #4)
// -------------------------------------------------------------------------------------------

/// A request to run a compute shader with device-computed workgroup counts.
///
/// Used by `QMoE`: a prior dispatch writes `[workgroups_x, workgroups_y, workgroups_z]`
/// (as `[u32; 3]`) into `dispatch_buffer` at `dispatch_offset`, then `dispatch_indirect`
/// issues `vkCmdDispatchIndirect` from that buffer. The engine inserts a
/// `ShaderWrite → ShaderRead` barrier automatically on `dispatch_buffer` between the
/// write dispatch and this call.
///
/// Defined as a separate struct (rather than a `workgroups` enum inside [`KernelRequest`])
/// to avoid changing `KernelRequest` and all existing call sites. Unification can happen later
/// once the design stabilises.
#[derive(Debug, Clone)]
pub struct IndirectKernelRequest {
    /// Shader stem as embedded by `build.rs`.
    pub shader: &'static str,
    /// Specialization constants, in binding order.
    pub spec_constants: Vec<u32>,
    /// Push-constant payload.
    pub push_constants: Vec<u8>,
    /// Storage-buffer bindings, in set-0 binding order.
    pub bindings: Vec<BufferView>,
    /// Buffer holding the workgroup counts. Written by a prior dispatch.
    pub dispatch_buffer: BufferView,
    /// Byte offset within `dispatch_buffer` where the `[u32; 3]` counts start.
    /// Must be a multiple of 4.
    pub dispatch_offset: u64,
}

// -------------------------------------------------------------------------------------------
// Dispatch seam
// -------------------------------------------------------------------------------------------

/// A request to run one compute shader. The engine decides descriptor sets, barriers, pipeline
/// selection and submission; the handler only says *what*.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelRequest {
    /// Shader stem as embedded by `build.rs` (`shaders/glsl/<stem>.comp`).
    pub shader: &'static str,
    /// Specialization constants, in binding order.
    pub spec_constants: Vec<u32>,
    /// Push-constant payload, already laid out to match the shader's block.
    pub push_constants: Vec<u8>,
    /// Storage-buffer bindings, in set-0 binding order.
    pub bindings: Vec<BufferView>,
    /// Workgroup counts for `vkCmdDispatch`.
    pub workgroups: [u32; 3],
}

/// The entire vocabulary an op handler has.
///
/// An implementor of this trait is the engine. A handler cannot hold a command buffer, allocate
/// memory, create a pipeline, or emit a barrier — it expresses intent and the engine does the
/// rest. `DESIGN.md` §4.2 rule 2; `ENGINE.md` §1.
///
/// **STUB:** Switch implements this over a recording `VkCommandBuffer`. The signatures are the
/// contract; the bodies are not here.
pub trait DispatchContext {
    /// Resolve a graph input / intermediate to a buffer the engine owns.
    fn resolve(&mut self, r: &TensorRef) -> EpResult<BufferView>;

    /// Declare and bind an output tensor, returning the buffer to write into.
    fn bind_output(&mut self, o: &OutRef, desc: TensorDesc) -> EpResult<BufferView>;

    /// Allocate a scratch buffer that lives until the end of this subgraph execution.
    fn alloc_temp(&mut self, desc: TensorDesc) -> EpResult<BufferView>;

    /// Record one dispatch.
    fn dispatch(&mut self, k: KernelRequest) -> EpResult<()>;

    /// Read a constant `int64` initializer on the host, for ops whose *shape* depends on a
    /// constant input (`Reshape`'s shape input, `Slice`'s starts/ends). Returns `None` when the
    /// input is not a constant.
    fn read_const_i64(&self, r: &TensorRef) -> Option<Vec<i64>>;

    // ── Seam 1: prepacked weight resolution (OP_COVERAGE.md §8.2.1 P5) ──────────────────

    /// Resolve a prepacked weight buffer at Compute time (P5).
    ///
    /// Called by handlers that use block-quantized weights, in place of `resolve` for the
    /// weight, scales, and zero-points inputs. The engine automatically returns the handles
    /// uploaded during `Compile` from `Plan::prepacked`.
    ///
    /// Returns `Err(EpError::Internal)` if no prepack result exists for `key` — this indicates
    /// a programming error: the `CompileContext::request_prepack` call should have run during
    /// `Compile` before this is called during `Compute`.
    ///
    /// **Default:** returns an internal error. Concrete engine implementations must override.
    fn resolve_prepacked(&self, key: &PackKey) -> EpResult<PrepackResult> {
        let _ = key;
        Err(EpError::Internal(
            "resolve_prepacked not implemented in this DispatchContext (stub)".into(),
        ))
    }

    // ── Seam 2: KV-cache in-place aliasing (OP_COVERAGE.md §9.5 #3) ─────────────────────

    /// Declare that output `out` aliases input `input` — they use the same device allocation.
    ///
    /// Used by `GroupQueryAttention` to update the KV cache in-place: `present_key` writes
    /// into the `past_key` allocation, eliminating a full-cache copy per decode step.
    ///
    /// The engine validates that the op's claim predicate guarantees no read-after-write hazard
    /// through the alias. The returned `BufferView` is the same handle as `resolve(input)` but
    /// typed as an output binding.
    ///
    /// **Coordination note for Tank (M2):** the handle-based allocator uses generation-stamped
    /// quarantine on free. An aliased output must NOT trigger a free-and-reallocate cycle — the
    /// existing handle must stay live for the entire Compute pass of this Plan. The engine's
    /// alias table must mark the handle as both input and output so the allocator's quarantine
    /// is not triggered between the input read and the output write. Please confirm this is
    /// compatible with the quarantine protocol or propose an alternative.
    ///
    /// **Default:** calls `self.resolve(input)` — this returns the input buffer as the output
    /// handle, which is correct semantics for implementations that do not track aliasing
    /// separately (e.g., the `Recorder` test stub).
    fn bind_aliased_output(
        &mut self,
        input: &TensorRef,
        out: &OutRef,
        desc: TensorDesc,
    ) -> EpResult<BufferView> {
        let _ = out;
        let _ = desc;
        self.resolve(input)
    }

    /// Does the caller use the **shared-buffer (arena)** KV-cache convention?
    ///
    /// `false` — the default and the shipping answer — means the growing convention: `present`
    /// is a strictly larger, separate allocation and the past tokens must be materialised into
    /// it. `true` means `past` is a fixed arena whose extent is a *capacity*, not a length; the
    /// true past length is carried by `seqlens_k`, and `present` aliases `past`.
    ///
    /// This is on the context rather than read from the environment inside `ops/` because
    /// `ops/` is not allowed to reach configuration or the ABI (`tests/layering.rs`), and
    /// because a stub context in a unit test must be able to state the convention it is
    /// testing without setting a process-wide variable.
    ///
    /// **Default:** `false`. An op handler that does not consult it keeps shipping behaviour.
    fn kv_arena(&self) -> bool {
        false
    }

    // ── Seam 3b: the growing-cache PREFIX alias ────────────────────────────────────────

    /// Does this engine implement [`DispatchContext::bind_prefix_output`]?
    ///
    /// The **growing** KV convention (`present` is `[B, Nkv, P+S, D]`, `past` is
    /// `[B, Nkv, P, D]`) is not an alias in the arena sense — the extents differ, so one buffer
    /// cannot *be* the other. But `present`'s first `P` tokens are, per outer block, exactly
    /// `past`'s bytes: the kernel's own first act is to copy them across. Under the prefix
    /// alias the engine stages `past` **into `present`'s prefix** instead, binds `present` for
    /// the `past_*` slots too, and never allocates a `past` buffer at all.
    ///
    /// MEASURED 2026-08-04 (`bench/results/ctx4096_BEFORE.json`), which is why this exists: on
    /// Phi-3.5 the shipping lane's device-local peak is the resident weight cache
    /// (2,290,839,552 B) plus **two** copies of the KV extent — `past` inputs and `present`
    /// outputs, 393,216 B per past token each. The second copy is thrown away at the end of
    /// every `Compute`, and it is what puts the peak over this device's budget somewhere
    /// between `past_len` 2048 and 3072.
    ///
    /// **Default:** `false`. A handler that does not consult it keeps shipping behaviour, and a
    /// stub context in a unit test can state the convention it is testing without a
    /// process-wide variable — the same reason [`DispatchContext::kv_arena`] is here.
    fn kv_growing_alias(&self) -> bool {
        false
    }

    /// Declare that `out` is a **growing superset** of the already-resolved input `input`:
    /// per outer block, `input`'s bytes are the head of `out`'s block.
    ///
    /// Returns the buffer to bind for `out` **and for `input`** — the caller must use the
    /// returned view for both, and must tell its kernel to read the input at `out`'s stride
    /// (for `gqa_f16` that is `past_stride == present_len`, which is exactly the condition its
    /// `copy_leader` predicate already reads, so no shader changes).
    ///
    /// `input` is a [`BufferView`], not a [`TensorRef`], on purpose: resolving a name twice
    /// advances the positional-mode token counter, so a seam that re-resolved would mint a
    /// different token for a single-node island than the one the handler already holds.
    ///
    /// **Default:** ignores the relation and returns a plain [`DispatchContext::bind_output`],
    /// which is the shipping path and is correct for any allocation.
    fn bind_prefix_output(
        &mut self,
        input: BufferView,
        out: &OutRef,
        desc: TensorDesc,
        layout: PrefixLayout,
    ) -> EpResult<BufferView> {
        let _ = input;
        let _ = layout;
        self.bind_output(out, desc)
    }

    // ── Seam 4: indirect dispatch for QMoE (OP_COVERAGE.md §9.5 #2) ─────────────────────

    /// Record a dispatch with device-computed workgroup counts.
    ///
    /// Used by `QMoE` (masked-dense first pass is not this; this is the fast MoE path where a
    /// prior dispatch writes the workgroup counts into `k.dispatch_buffer`). The engine
    /// inserts a `ShaderWrite → ShaderRead` barrier on `k.dispatch_buffer` automatically.
    ///
    /// **Default:** returns an internal error. Concrete engine implementations must override
    /// when the QMoE fast path is enabled.
    fn dispatch_indirect(&mut self, k: IndirectKernelRequest) -> EpResult<()> {
        let _ = k;
        Err(EpError::Internal(
            "dispatch_indirect not implemented in this DispatchContext (stub)".into(),
        ))
    }
}

// -------------------------------------------------------------------------------------------
// The compiled-session seam (re-exports)
// -------------------------------------------------------------------------------------------

// `ep.rs` needs to name the compiled artefact, the recorder that produces it and the session that
// executes it. Those types live in `vk::session` because that is where they are implemented, but
// the module dependency table (`DESIGN.md` §4.3, enforced by `tests/layering.rs`) says the ABI
// boundary layer talks to `engine` and never to `vk`. Re-exporting them here is what makes that
// true rather than aspirational: the names are engine vocabulary, the implementation stays
// Switch's, and `ep.rs` gains no way to reach a raw Vulkan handle.
//
// Nothing re-exported here may expose an `ash` type in its public signature. That is the property
// the layering rule actually protects, and a re-export is only legal while it holds.
pub(crate) use crate::vk::session::{CompileRecorder, CompiledKernel, VulkanSession};

// -------------------------------------------------------------------------------------------
// Device capability probe (STUB)
// -------------------------------------------------------------------------------------------

/// One Vulkan physical device that passed the capability gate, described without any Vulkan type.
///
/// `factory.rs` correlates these with ORT's `OrtHardwareDevice` list by `(vendor_id, device_id)`
/// and advertises one `OrtEpDevice` per entry (`DESIGN.md` §2.3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    /// Index into `vkEnumeratePhysicalDevices`, and the value of the `ep.device_index` option.
    pub index: usize,
    /// `VkPhysicalDeviceProperties::deviceName`.
    pub name: String,
    /// `VkPhysicalDeviceProperties::vendorID`. Reported to ORT as the EP device's vendor ID —
    /// there is no single hardware vendor for this EP (`DESIGN.md` §3.1).
    pub vendor_id: u32,
    /// `VkPhysicalDeviceProperties::deviceID`.
    pub device_id: u32,
    /// `VkPhysicalDeviceProperties::apiVersion`, formatted `major.minor.patch`.
    pub api_version: String,
    /// `VkPhysicalDeviceProperties::driverVersion`, vendor-formatted.
    pub driver_version: String,
    /// Discrete / integrated / virtual / CPU. Drives both scoring and the ORT device-type match.
    pub kind: DeviceKind,
    /// The stable, driver-reported identity of this physical device (issue #18).
    ///
    /// Held as one value rather than three loose fields because every consumer that matters —
    /// the selector, the proof frame, the ledger, the modelrunner's evidence — needs the whole
    /// of it or none of it, and because [`DeviceIdentity::key`] is the single place the
    /// canonical key is derived.
    pub identity: DeviceIdentity,
}

impl DeviceInfo {
    /// This device's canonical frame key: UUID when the driver reported one, the fail-closed
    /// fallback otherwise. See [`DeviceKey`].
    pub fn key(&self) -> DeviceKey {
        self.identity.key(&self.name, self.index)
    }
}

/// The stable, driver-reported identity of one physical device (issue #18).
///
/// **Never fabricated.** Every field is `Option` and `None` means "this driver/platform did not
/// report it", which is a different fact from "it did not match" — a selector asking for an
/// identity kind no enumerated device reports gets
/// [`DeviceSelectionError::UnsupportedIdentity`][crate::vk::instance::DeviceSelectionError::UnsupportedIdentity],
/// not `NotFound`.
#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DeviceIdentity {
    /// `VkPhysicalDeviceIDProperties::deviceUUID`, 32 lowercase hex characters, no separators.
    ///
    /// **The stable identity this EP treats as ground truth.** Unlike `index` (an ordinal that
    /// depends on enumeration order, which the loader is free to change across driver updates,
    /// reboots or `VK_ICD_FILENAMES` edits) and unlike `(vendor_id, device_id)` (which is the
    /// same for every card of the same model), the UUID names *this physical device instance*.
    ///
    /// `VkPhysicalDeviceIDProperties` is Vulkan 1.1 core and the §7.2 gate already requires at
    /// least 1.1, so in practice every device this EP advertises has one. It is still an
    /// `Option`, because a driver that leaves the struct untouched returns all zeros, and an
    /// all-zero UUID is **not** an identity: it is the absence of one, it compares equal across
    /// every device that has it, and treating it as a value is exactly the name-collision defect
    /// one layer down. [`crate::vk::instance::query_device_identity`] maps all-zero to `None`,
    /// and every reader is then obliged to say `(unavailable)` rather than print 32 zeros.
    pub uuid: Option<String>,
    /// `VkPhysicalDeviceIDProperties::deviceLUID`, 16 lowercase hex characters, present only
    /// when `deviceLUIDValid == VK_TRUE`.
    ///
    /// LUIDs are a Windows/D3D-interop concept; most Linux and essentially all Android/MoltenVK
    /// drivers report `deviceLUIDValid = VK_FALSE`. `None` here means "this driver did not
    /// provide one," never "absent because unset".
    pub luid: Option<String>,
    /// PCI location as `domain:bus:device.function` (all lowercase hex), from
    /// `VK_EXT_pci_bus_info` when the device advertises that extension.
    ///
    /// Capability-gated: MoltenVK and many mobile/virtualized ICDs do not expose PCI location at
    /// all (there may be no PCI bus to report), so this is `None` there rather than a fabricated
    /// value. Present on essentially every desktop Linux/Windows discrete or integrated GPU.
    pub pci: Option<String>,
}

impl DeviceIdentity {
    /// A UUID-only identity, for the callers that have nothing else (tests, and the ledger).
    pub fn from_uuid(uuid: impl Into<String>) -> Self {
        Self {
            uuid: Some(uuid.into()),
            ..Self::default()
        }
    }

    /// The canonical key this device is counted, framed and attributed under.
    ///
    /// `name` and `physical_index` are used **only** by the fail-closed fallback — see
    /// [`DeviceKey::Unidentified`]. When a UUID exists it alone decides, so the key of a device
    /// is the same in every process on every machine that ever opens it.
    pub fn key(&self, name: &str, physical_index: usize) -> DeviceKey {
        match &self.uuid {
            Some(u) if !u.is_empty() => DeviceKey::Uuid(u.clone()),
            _ => DeviceKey::Unidentified {
                name: name.to_string(),
                physical_index,
            },
        }
    }
}

/// **The canonical identity a proof frame, a counter and a ledger entry are keyed by.**
///
/// Issue #18's whole point in one type. Before it, `allocator::tally` keyed frames on
/// `deviceName`, so two physical GPUs of the same model collapsed into one frame: both declared
/// `SHARED`, `frames_declared()` returned 1, `MIXED` never fired, and a process standing on two
/// devices reported a single frame over a population drawn from both. A name is not an identity.
///
/// # The fail-closed fallback contract
///
/// [`DeviceKey::Uuid`] is the normal case and the only one that can ever prove two observations
/// were made on the same physical device. [`DeviceKey::Unidentified`] exists because a driver may
/// leave `VkPhysicalDeviceIDProperties` unpopulated, and the alternatives are both worse:
/// fabricating a UUID (a false identity) or falling back to the name (the defect being fixed).
///
/// Its contract is **fail-closed in both directions**, and both halves are load-bearing:
///
/// * **Never collapses.** The key carries the physical enumeration index, so two same-named
///   devices without UUIDs are two keys and a process on both reports `MIXED`. A false `MIXED`
///   (same device enumerated at two indices in one process — which `acquire_ep_device`'s
///   per-physical-index owner map already prevents) costs a disclosed frame; a false `SHARED`
///   costs an unattributed measurement.
/// * **Never proves.** It is process-local by construction, so it may not be compared across
///   processes. [`crate::registry::device_state`] therefore returns `DeviceUnattributed` — never
///   `Proven` — for any run or entry whose identity is `Unidentified`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DeviceKey {
    /// The driver reported `VkPhysicalDeviceIDProperties::deviceUUID`: 32 lowercase hex
    /// characters naming this physical device instance, stable across processes and reboots.
    Uuid(String),
    /// No UUID was reported. Process-local, never comparable across processes — see the
    /// fail-closed contract on [`DeviceKey`].
    Unidentified {
        /// The device name, for the human reading the artifact. Not an identity.
        name: String,
        /// The `vkEnumeratePhysicalDevices` index, which is what keeps two same-named
        /// unidentified devices from collapsing into one frame within this process.
        physical_index: usize,
    },
}

impl DeviceKey {
    /// The UUID, or `None` when this device reported none.
    pub fn uuid(&self) -> Option<&str> {
        match self {
            DeviceKey::Uuid(u) => Some(u.as_str()),
            DeviceKey::Unidentified { .. } => None,
        }
    }

    /// Whether this key names a device by an identity that survives leaving this process.
    pub fn is_identified(&self) -> bool {
        matches!(self, DeviceKey::Uuid(_))
    }

    /// The single string form written into artifacts and log lines.
    ///
    /// `uuid:<32 hex>` or `unidentified:<name>#<physical index>`. The prefix is not decoration:
    /// it is what stops a reader — or a future comparison — from mistaking the fallback for an
    /// identity.
    pub fn canonical(&self) -> String {
        match self {
            DeviceKey::Uuid(u) => format!("{DEVICE_KEY_UUID_PREFIX}{u}"),
            DeviceKey::Unidentified {
                name,
                physical_index,
            } => format!("{DEVICE_KEY_UNIDENTIFIED_PREFIX}{name}#{physical_index}"),
        }
    }

    /// The inverse of [`DeviceKey::canonical`] — read a key back off the wire.
    ///
    /// `None` for anything that is not one of the two canonical shapes, including a `uuid:` whose
    /// body is not 32 hex characters. A key that does not round-trip is not a key: admitting a
    /// malformed one would let a hand-written or truncated artifact re-enter the comparison as if
    /// the EP had emitted it.
    ///
    /// The `unidentified:` shape is `<name>#<index>` with the index at the **end**, which is what
    /// makes this parse possible at all: a `deviceName` may contain a colon, so the
    /// `<index>:<name>` spelling three documents used to claim could not be split back apart
    /// without already knowing which half was numeric.
    pub fn from_canonical(s: &str) -> Option<DeviceKey> {
        if let Some(hex) = s.strip_prefix(DEVICE_KEY_UUID_PREFIX) {
            return key_is_portable_identity(s).then(|| DeviceKey::Uuid(hex.to_string()));
        }
        let rest = s.strip_prefix(DEVICE_KEY_UNIDENTIFIED_PREFIX)?;
        let (name, index) = rest.rsplit_once('#')?;
        if name.is_empty() {
            return None;
        }
        Some(DeviceKey::Unidentified {
            name: name.to_string(),
            physical_index: index.parse().ok()?,
        })
    }

    /// Test-only: a deterministic synthetic UUID key derived from `seed`.
    ///
    /// Distinct seeds give distinct 32-hex-character keys, which is exactly what the
    /// identical-name multi-GPU tests need: two devices whose `name` is byte-identical but whose
    /// identity is not. Never used outside `#[cfg(test)]`/`#[doc(hidden)]` call sites — a
    /// fabricated UUID in production would be the very thing [`DeviceKey`] exists to prevent.
    #[doc(hidden)]
    pub fn synthetic_for_test(seed: &str) -> DeviceKey {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in seed.as_bytes() {
            h ^= u64::from(*b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        let lo = h.rotate_left(17) ^ 0x9e37_79b9_7f4a_7c15;
        DeviceKey::Uuid(format!("{h:016x}{lo:016x}"))
    }
}

impl std::fmt::Display for DeviceKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.canonical())
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// THE DEVICE-LIST WIRE FORMAT (issue #18, blocker B1)
//
// One format, written down once, with one reader per language. What made this a blocker rather
// than a typo is that the format had never been *stated*: `counters.rs` emitted one shape,
// `gen_proof_ledger.py` parsed another, and neither side was wrong on its own terms — so nothing
// failed. `LedgerEntry.device_uuid` was simply `""` on every entry ever written, and the two
// stable-identity rows of `registry::device_state` (`Proven` / `ProvenElsewhere`) were
// unreachable. A silent hole in a proof frame is worse than a loud one, and the only structural
// defence is a single documented format with tests that read the *emitted* string.
// ──────────────────────────────────────────────────────────────────────────────

/// The separator between elements of every device list this EP writes.
///
/// A device name may contain a comma, a slash, a colon, an `=` and parentheses (`llvmpipe (LLVM
/// 20.1.2, 256 bits)`, `Intel(R) Iris(R) Xe Graphics`, `… rev=3`). `"; "` is the one separator no
/// observed `VkPhysicalDeviceProperties::deviceName` has ever contained, which is why it — and
/// not `,` — is the list separator.
pub const DEVICE_LIST_SEPARATOR: &str = "; ";

/// Format a device list into **the canonical wire form**: bare values, `"; "`-separated, in
/// physical enumeration order.
///
/// This is the form of the `running_device_names` and `running_device_uuids` counters, and it is
/// what every artifact in `bench/results/` carries — `"NVIDIA RTX A1000"`,
/// `"uuid:aadf33d4d118155fcc60c22b5c352463"`, `"Intel(R) Iris(R) Xe Graphics; NVIDIA GeForce RTX
/// 4060 Laptop GPU"`. There is **no positional prefix**: position in the list *is* the index, so
/// writing it again would be a second source of truth for the same fact.
///
/// The `alloc_device_frame_session_devices` counter is a *different* counter with a *different*
/// format (`"1=The Session's Device"`): it reports a sparse map keyed by the factory device index
/// ORT asked for, where position is genuinely not the index. [`parse_device_list`] tolerates that
/// shape so a reader pointed at the wrong counter degrades to a right answer instead of an empty
/// one, but nothing this EP writes under `running_device_*` uses it.
pub fn format_device_list<I>(items: I) -> String
where
    I: IntoIterator,
    I::Item: AsRef<str>,
{
    items
        .into_iter()
        .map(|s| s.as_ref().to_string())
        .collect::<Vec<_>>()
        .join(DEVICE_LIST_SEPARATOR)
}

/// Read a device list off the wire — **the only reader**, on the Rust side, of the format
/// [`format_device_list`] writes.
///
/// Accepts the canonical bare form and tolerates the indexed `N=value` form of
/// `alloc_device_frame_session_devices`. The tolerance is deliberately asymmetric with the
/// writer: a reader that *requires* the prefix returns an empty list against every real counters
/// file this project has ever produced (that was the defect), whereas a reader that merely
/// tolerates it cannot be wrong about a value that never carries one.
///
/// Only a **purely numeric** prefix is positional, so `llvmpipe (LLVM 20.1.2, 256 bits) rev=3`
/// survives intact rather than being renamed to ` 3`. Empty elements are dropped: `"0="` is the
/// absence of an identity, not an identity that happens to be empty, and admitting it would let
/// one emptiness compare equal to another.
///
/// The sentinels `none` and `unknown` — what `allocator::tally` returns for "no session has
/// opened a device" and "the mutex was poisoned" — are **not** handled here, because they are
/// statements about the whole list rather than elements of it. Callers reject them first; see
/// [`crate::registry::running_device_uuids`].
pub fn parse_device_list(raw: &str) -> Vec<String> {
    raw.split(DEVICE_LIST_SEPARATOR)
        .filter_map(strip_positional_index)
        .collect()
}

/// Drop a leading `N=` positional prefix, if present, and return the remainder when non-empty.
///
/// `"0=uuid:aabb"` → `"uuid:aabb"`; `"uuid:aabb"` → `"uuid:aabb"`; `"0="` → `None`;
/// `"a name with = in it"` → itself.
fn strip_positional_index(item: &str) -> Option<String> {
    let t = item.trim();
    let body = match t.split_once('=') {
        Some((head, rest)) if !head.is_empty() && head.bytes().all(|b| b.is_ascii_digit()) => rest,
        _ => t,
    };
    let body = body.trim();
    (!body.is_empty()).then(|| body.to_string())
}

/// Whether `key` is a canonical identity that may be compared **across processes**.
///
/// `uuid:<hex>` is; `unidentified:<name>#<index>` is not, and neither is anything else. This is
/// the one predicate that decides whether a comparison is *possible*, and it is separate from
/// whether the comparison *succeeds* — [`crate::registry::device_state`] needs both, and reading
/// a failed comparison off an impossible one is how a process-local fallback would launder itself
/// into a claim about hardware.
pub fn key_is_portable_identity(key: &str) -> bool {
    key.strip_prefix(DEVICE_KEY_UUID_PREFIX)
        .is_some_and(|hex| hex.len() == 32 && hex.bytes().all(|b| b.is_ascii_hexdigit()))
}

/// The prefix on a [`DeviceKey::Uuid`] canonical string.
pub const DEVICE_KEY_UUID_PREFIX: &str = "uuid:";

/// The prefix on a [`DeviceKey::Unidentified`] canonical string.
///
/// The canonical shape is `unidentified:<deviceName>#<physical enumerate index>` — **name first,
/// index last, separated by `#`**. It is written down here because it was documented in two
/// mutually exclusive ways (`unidentified:<index>:<name>` in `DESIGN.md`, `PLATFORMS.md` and the
/// modelrunner fixtures; `unidentified:<name>#<index>` in the code that actually emits it), and a
/// fallback whose spelling is ambiguous is a fallback a reader cannot recognise.
///
/// `#` rather than a second `:` is load-bearing: a device name may contain a colon, so
/// `unidentified:<index>:<name>` cannot be split back apart without knowing the index is numeric,
/// while `#` is followed by digits to end of string and always can be.
pub const DEVICE_KEY_UNIDENTIFIED_PREFIX: &str = "unidentified:";

/// `VkPhysicalDeviceType`, minus the Vulkan type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DeviceKind {
    /// Software rasterizer or CPU implementation (lavapipe, SwiftShader) — the CI lane.
    Cpu,
    Virtual,
    Integrated,
    Discrete,
}

impl DeviceKind {
    /// Selection score: discrete > integrated > virtual > CPU (`ENGINE.md` §2.2).
    pub const fn score(self) -> u32 {
        match self {
            DeviceKind::Discrete => 4,
            DeviceKind::Integrated => 3,
            DeviceKind::Virtual => 2,
            DeviceKind::Cpu => 1,
        }
    }
}

/// Enumerate Vulkan physical devices that satisfy the capability gate, best first.
///
/// The contract `factory.rs` relies on, and which the real implementation must keep:
///
/// * **Never fails the host.** A machine with no Vulkan loader, no ICD, or a broken driver returns
///   `Ok(vec![])` and logs a warning. It must not return `Err`, must not panic, and must not
///   abort — session creation has to keep working and fall back to CPU (`DESIGN.md` §2.3, M0 exit
///   criterion 4).
/// * **Sorted best-first** by [`DeviceKind::score`], so index 0 is the default device.
/// * **Cheap enough to call from `GetSupportedDevices`,** which is on the session-creation path.
///   It is *not* called from `CreateEpFactories`: a plugin must be cheap to load even on a machine
///   that will never use it (`DESIGN.md` §5.1).
/// * **Returns zero devices when built without shaders** (`DESIGN.md` §7.8 condition 3). A
///   shader-less artifact must not present itself as a working EP — ORT must fall back to CPU
///   rather than dispatching work to kernels that do not exist.
///
/// Returning an empty list today is exactly what a driverless machine will do tomorrow, so the
/// factory path this stub feeds is the same code that ships.
pub fn probe_devices() -> Vec<DeviceInfo> {
    use crate::vk::instance::Instance;

    // Guard: DESIGN.md §7.8 condition 3 — a shader-less artifact must advertise zero devices.
    if !shaders::has_any() {
        log::warn!(
            "probe_devices: built without shaders (ALLOW_MISSING_GLSLC build) — \
             advertising zero devices so ORT falls back to CPU. \
             This artifact cannot dispatch compute and must not be shipped."
        );
        return Vec::new();
    }

    let Some(inst) = Instance::create(false) else {
        // Instance::create already logged the reason.
        return Vec::new();
    };

    inst.enumerate_capable_devices()
        .into_iter()
        .map(|d| d.info)
        .collect()
}

/// The devices the factory should advertise to ORT (§6.5).
///
/// Identical to [`probe_devices`] unless a selector is set, in which case exactly one device is
/// returned — the selector is a **pin**.
///
/// **Precedence: [`ONNXRUNTIME_EP_VULKAN_DEVICE_SELECTOR`][crate::vk::instance::ENV_DEVICE_SELECTOR_STRICT]
/// (stable identity, strict) outranks [`ONNXRUNTIME_EP_VULKAN_DEVICE`][crate::vk::instance::ENV_DEVICE_SELECTOR]
/// (index/name-substring, lenient).** If the strict selector is set and cannot be resolved to
/// exactly one device, **zero devices are advertised** — never a fallback to a different GPU —
/// and the reason is logged at ERROR (issue #18). If it is not set, behaviour is exactly what it
/// was before issue #18: the legacy selector pins when set, and an unset selector advertises
/// every capable device and lets ORT's policy choose.
///
/// # One authoritative selection path, and what `ep.device_selector` may and may not do
///
/// This function runs inside `GetSupportedDevices`, **before any session exists**, so no session
/// option can be visible here. That is a fact about ORT's plugin ABI, not a choice, and it has
/// one dangerous consequence which issue #18's first revision shipped: if the session option
/// could *redirect* the session to a device this function did not advertise, ORT would bind X,
/// the session would open Y, and the `OrtEpDevice` metadata a caller reads for evidence would
/// carry X's identity over kernels that ran on Y.
///
/// So the selection path is single and authoritative here, and
/// [`ep.device_selector`][crate::ep::EpOptions::device_selector] is **subtractive only**: it may
/// refuse a binding it disagrees with (`acquire_ep_device` returns `None`, the session falls back
/// to the CPU EP), and it may never move the session to another device. Pinning — choosing which
/// GPU runs — is the environment variable's job precisely because it is readable from here.
///
/// **Why the pin has to bite here and not later.** ORT chooses which advertised `OrtEpDevice` to
/// bind, and it keys the allocator it later asks us for by that choice. If we advertise a device
/// the selector did not name, ORT may bind it while the compute session opens the pinned one —
/// and then the run needs two `VkDevice`s to proceed, which is exactly what §6.5 forbids and what
/// `alloc_device_frame = SPLIT-DEVICE` was reporting on the Intel lane. Advertising only the
/// pinned device removes the divergence at its source: with one device on offer, "the device ORT
/// bound" and "the device the selector names" cannot be two different devices.
///
/// Unpinned behaviour is unchanged — every capable device is advertised and ORT's policy chooses;
/// `CreateEp` then follows whatever it bound.
pub fn devices_to_advertise() -> Vec<DeviceInfo> {
    let all = probe_devices();
    if all.is_empty() {
        return all;
    }

    // Strict, stable-identity selector: highest precedence, and it never falls back (issue #18).
    match crate::vk::instance::select_device_strict(&all, None) {
        Ok(Some(sel)) => {
            let picked = all[sel].clone();
            log::info!(
                "VulkanExecutionProvider: {} resolved to '{}' ({}, physical enumerate index \
                 {}). Advertising ONLY that device, so ORT cannot bind a device other than the \
                 one the compute session will open (§6.5).",
                crate::vk::instance::ENV_DEVICE_SELECTOR_STRICT,
                picked.name,
                picked.key(),
                picked.index,
            );
            return vec![picked];
        }
        Ok(None) => { /* not set — fall through to the legacy selector below, unchanged */ }
        Err(e) => {
            log::error!(
                "VulkanExecutionProvider: {} {e}. Refusing to advertise ANY Vulkan device rather \
                 than silently binding a different GPU (issue #18 device-selection contract). \
                 Sessions will fall back to the CPU EP until the selector is fixed or unset.",
                crate::vk::instance::ENV_DEVICE_SELECTOR_STRICT,
            );
            return Vec::new();
        }
    }

    if !crate::vk::instance::selector_is_pinned() {
        return all;
    }
    let names: Vec<&str> = all.iter().map(|d| d.name.as_str()).collect();
    let Some(sel) = crate::vk::instance::select_by_selector(&names) else {
        return all;
    };
    let picked = all[sel].clone();
    log::info!(
        "VulkanExecutionProvider: {} is pinned to selector index {sel} → '{}' (physical \
         enumerate index {}). Advertising ONLY that device, so ORT cannot bind a device other \
         than the one the compute session will open (§6.5).",
        crate::vk::instance::ENV_DEVICE_SELECTOR,
        picked.name,
        picked.index,
    );
    vec![picked]
}

/// Run the Vulkan loader probe and return a formatted diagnostic report string.
///
/// This is the backend for `epctl --probe-loader`. It bypasses the shader guard in
/// [`probe_devices`] so that CI can verify Vulkan ICD availability independently of whether
/// this artifact was compiled with shaders.
///
/// Never panics. Returns a human-readable multi-line string.
pub fn loader_probe_report() -> String {
    crate::vk::instance::probe_loader_report()
}

/// Embedded SPIR-V modules, generated by `build.rs` from `shaders/glsl/*.comp`.
///
/// Empty until Switch adds the first shader. Kept here so the engine has one obvious place to look
/// and `build.rs` has one obvious consumer.
pub mod shaders {
    include!(concat!(env!("OUT_DIR"), "/shader_modules.rs"));
    include!(concat!(env!("OUT_DIR"), "/shader_toolchain.rs"));

    /// Look up an embedded SPIR-V module by shader stem.
    pub fn find(stem: &str) -> Option<&'static [u8]> {
        SHADER_MODULES
            .iter()
            .find(|(name, _)| *name == stem)
            .map(|(_, bytes)| *bytes)
    }

    /// Look up an embedded module's **source-closure digest** by shader stem (§8.9.19 part 2).
    ///
    /// `None` means this build embeds no such module — a different fact from "its source hashed
    /// to nothing", and the caller must keep them apart or a deleted kernel reads as an unchanged
    /// one.
    pub fn source_digest(stem: &str) -> Option<&'static str> {
        SHADER_SOURCE_DIGESTS
            .iter()
            .find(|(name, _)| *name == stem)
            .map(|(_, digest)| *digest)
    }

    /// `glslc --version` from the build that produced this artifact — a FRAME component.
    #[inline]
    pub fn toolchain() -> &'static str {
        SHADER_TOOLCHAIN
    }

    /// True when at least one SPIR-V module was compiled into this artifact.
    ///
    /// False for escape-hatch builds (`ONNXRUNTIME_EP_VULKAN_ALLOW_MISSING_GLSLC=1`).
    /// Used by [`super::probe_devices`] and `ep::get_capability_impl` to enforce
    /// `DESIGN.md` §7.8 condition 3: a shader-less artifact must advertise zero devices
    /// and claim nothing.
    #[inline]
    pub fn has_any() -> bool {
        !SHADER_MODULES.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Phi-3.5 decode step at past_len 2048: `past` is [1,32,2048,96] f16, `present` is
    /// [1,32,2049,96] f16, so 32 blocks of 2048*96*2 bytes landing at a 2049*96*2 stride.
    fn phi35_layout(past_len: u64) -> (PrefixLayout, u64, u64) {
        let (blocks, head_dim, elem) = (32u64, 96u64, 2u64);
        let src = past_len * head_dim * elem;
        let dst = (past_len + 1) * head_dim * elem;
        (
            PrefixLayout {
                outer_blocks: blocks,
                src_block_bytes: src,
                dst_block_bytes: dst,
            },
            blocks * src,
            blocks * dst,
        )
    }

    #[test]
    fn a_prefix_layout_that_tiles_the_input_and_fits_the_output_is_accepted() {
        let (layout, in_sz, out_sz) = phi35_layout(2048);
        assert!(layout.fits(in_sz, out_sz));
    }

    #[test]
    fn a_prefix_layout_is_accepted_when_the_last_block_ends_exactly_at_the_output_end() {
        // The tightest legal case: equal strides, so the last region ends on the final byte.
        let layout = PrefixLayout {
            outer_blocks: 4,
            src_block_bytes: 16,
            dst_block_bytes: 16,
        };
        assert!(layout.fits(64, 64));
        assert!(!layout.fits(64, 63));
    }

    #[test]
    fn a_prefix_layout_that_does_not_account_for_every_source_byte_is_refused() {
        let (mut layout, in_sz, out_sz) = phi35_layout(2048);
        layout.outer_blocks = 31;
        assert!(
            !layout.fits(in_sz, out_sz),
            "31 blocks leave a whole head's past unwritten; staging it would be a wrong answer"
        );
    }

    #[test]
    fn a_prefix_layout_wider_than_its_destination_stride_is_refused() {
        // src > dst makes the regions overlap in the destination, which the spec forbids and
        // which no amount of ordering makes well-defined.
        let layout = PrefixLayout {
            outer_blocks: 4,
            src_block_bytes: 20,
            dst_block_bytes: 16,
        };
        assert!(!layout.fits(80, 4096));
    }

    #[test]
    fn a_prefix_layout_whose_last_block_runs_past_the_output_is_refused() {
        let (layout, in_sz, _) = phi35_layout(2048);
        // Note the arithmetic: the *natural* output size (blocks * dst_block_bytes) has
        // `dst - src` bytes of slack past the last region, because the final block is a source
        // block at a destination stride. The binding case is therefore one byte below the last
        // region's end, not one byte below the buffer's nominal size — shaving the buffer by one
        // byte is still legal, and a test that asserted otherwise would be asserting the wrong
        // bound.
        let last_end = (layout.outer_blocks - 1) * layout.dst_block_bytes + layout.src_block_bytes;
        assert!(layout.fits(in_sz, last_end));
        assert!(!layout.fits(in_sz, last_end - 1));
    }

    #[test]
    fn a_degenerate_prefix_layout_is_refused_rather_than_treated_as_a_no_op() {
        for layout in [
            PrefixLayout {
                outer_blocks: 0,
                src_block_bytes: 16,
                dst_block_bytes: 16,
            },
            PrefixLayout {
                outer_blocks: 4,
                src_block_bytes: 0,
                dst_block_bytes: 16,
            },
        ] {
            assert!(!layout.fits(0, 4096));
            assert!(!layout.fits(64, 4096));
        }
    }

    #[test]
    fn a_prefix_layout_that_overflows_u64_refuses_instead_of_wrapping() {
        let layout = PrefixLayout {
            outer_blocks: u64::MAX,
            src_block_bytes: 4,
            dst_block_bytes: u64::MAX,
        };
        assert!(!layout.fits(0, u64::MAX));
        assert!(!layout.fits(u64::MAX, u64::MAX));
    }

    #[test]
    fn dtype_sizes() {
        assert_eq!(DType::F32.byte_size(), 4);
        assert_eq!(DType::F16.byte_size(), 2);
        assert_eq!(DType::I64.byte_size(), 8);
        assert_eq!(DType::Bool.byte_size(), 1);
    }

    #[test]
    fn tensor_desc_sizes() {
        let d = TensorDesc::new(DType::F32, vec![2, 3, 4]);
        assert_eq!(d.element_count(), Some(24));
        assert_eq!(d.byte_size(), Some(96));
    }

    #[test]
    fn symbolic_dims_do_not_produce_a_size() {
        let d = TensorDesc::new(DType::F32, vec![-1, 3]);
        assert_eq!(d.element_count(), None);
        assert_eq!(d.byte_size(), None);
    }

    #[test]
    fn qualified_name_distinguishes_domains() {
        let mut n = NodeDesc {
            op_type: "Attention".into(),
            ..Default::default()
        };
        assert_eq!(n.qualified_name(), "Attention");
        n.domain = "com.microsoft".into();
        assert_eq!(n.qualified_name(), "com.microsoft::Attention");
        n.domain = "ai.onnx".into();
        assert_eq!(n.qualified_name(), "Attention");
    }

    #[test]
    fn device_scoring_prefers_discrete() {
        let mut kinds = [
            DeviceKind::Cpu,
            DeviceKind::Discrete,
            DeviceKind::Virtual,
            DeviceKind::Integrated,
        ];
        kinds.sort_by_key(|k| std::cmp::Reverse(k.score()));
        assert_eq!(kinds[0], DeviceKind::Discrete);
        assert_eq!(kinds[3], DeviceKind::Cpu);
    }

    #[test]
    fn probe_never_panics_even_without_a_vulkan_loader() {
        // On machines without a Vulkan ICD (including most CI containers), probe_devices must
        // return an empty Vec, not panic and not fail the process. On Trinity's lavapipe lanes
        // it will return at least one device; that is also correct behaviour.
        let devices = probe_devices();
        // The result is either empty (no ICD, or shader-less build) or a sorted list — all valid.
        // Verify sort invariant: each device's score >= the next device's score.
        for w in devices.windows(2) {
            assert!(
                w[0].kind.score() >= w[1].kind.score(),
                "probe_devices must return devices sorted best-first"
            );
        }
    }

    // ── DESIGN.md §7.8 condition 3: shader-less artifact guards ─────────────────────────
    //
    // These three tests work in BOTH build modes:
    //   • shader-complete build (glslc present, normal CI): has_any() == true
    //   • escape-hatch build (ALLOW_MISSING_GLSLC=1): has_any() == false
    //
    // Each test asserts the invariant that holds in the current build, so the suite stays
    // green in both modes rather than requiring a conditional skip or a separate CI matrix.

    #[test]
    fn shader_module_table_matches_build_mode() {
        // has_any() and SHADER_MODULES.is_empty() must always agree — they describe the same fact.
        let table_non_empty = !shaders::SHADER_MODULES.is_empty();
        assert_eq!(
            shaders::has_any(),
            table_non_empty,
            "has_any() must agree with !SHADER_MODULES.is_empty()"
        );

        if shaders::has_any() {
            // Shader-complete build: M0's first shader (ew_binary_add_f32) must be present.
            assert!(
                shaders::find("ew_binary_add_f32").is_some(),
                "ew_binary_add_f32 must be compiled into a shader-complete build"
            );
            // Old placeholder stem must never appear.
            assert!(
                shaders::find("elementwise_binary").is_none(),
                "no shader is registered under the legacy stem 'elementwise_binary'"
            );
        } else {
            // Escape-hatch build: no shader is available.
            assert!(shaders::find("ew_binary_add_f32").is_none());
            assert!(shaders::find("elementwise_binary").is_none());
        }
    }

    #[test]
    fn has_any_reflects_build_state() {
        // has_any() is a thin wrapper over SHADER_MODULES.is_empty(). This test documents
        // and pins both directions:
        //   • shader-complete build → has_any() is true, SHADER_MODULES is non-empty.
        //   • escape-hatch build   → has_any() is false, SHADER_MODULES is empty.
        if shaders::has_any() {
            assert!(
                !shaders::SHADER_MODULES.is_empty(),
                "has_any() true implies SHADER_MODULES non-empty"
            );
        } else {
            assert!(
                shaders::SHADER_MODULES.is_empty(),
                "has_any() false implies SHADER_MODULES is empty"
            );
        }
    }

    #[test]
    fn probe_devices_returns_empty_when_no_shaders() {
        // DESIGN.md §7.8 condition 3: a shader-less artifact must advertise zero devices.
        // When shaders are compiled in, probe_devices() may return real devices (depending on
        // whether an ICD is present); the zero-device guarantee only applies to the no-shader
        // case. This test only runs its assertion in the escape-hatch build.
        if !shaders::has_any() {
            let devices = probe_devices();
            assert!(
                devices.is_empty(),
                "shader-less build must advertise zero devices (got {} devices)",
                devices.len()
            );
        }
        // In a shader-complete build the test is a no-op (passes trivially): probe_devices()
        // returns whatever the ICD reports, which is correct behaviour.
    }

    // ── Seam 1: prepack vocabulary ────────────────────────────────────────────────────────

    fn dummy_pack_fn(input: PackInput<'_>) -> PackOutput {
        PackOutput {
            packed_weight: input.weight.to_vec(),
            packed_scales: input.scales.to_vec(),
            packed_zero_points: input.zero_points.map(|zp| zp.to_vec()),
        }
    }

    #[test]
    fn pack_key_hash_equality() {
        let config = TileConfig {
            tile_n: 8,
            tile_m: 4,
            block_size: 32,
        };
        let k1 = PackKey {
            initializer: "weight0".into(),
            config: config.clone(),
            variant: "matmul_nbits_q4_b32_gemv",
        };
        let k2 = PackKey {
            initializer: "weight0".into(),
            config: config.clone(),
            variant: "matmul_nbits_q4_b32_gemv",
        };
        let k3 = PackKey {
            initializer: "weight1".into(),
            config,
            variant: "matmul_nbits_q4_b32_gemv",
        };
        assert_eq!(k1, k2, "same initializer+config+variant must be equal");
        assert_ne!(k1, k3, "different initializer must differ");

        use std::collections::HashMap;
        let mut m = HashMap::new();
        m.insert(k1, 42u32);
        assert_eq!(m[&k2], 42, "equal keys must hit same map entry");
    }

    #[test]
    fn pack_key_different_tile_config_differs() {
        let k1 = PackKey {
            initializer: "w".into(),
            config: TileConfig {
                tile_n: 8,
                tile_m: 4,
                block_size: 32,
            },
            variant: "v",
        };
        let k2 = PackKey {
            initializer: "w".into(),
            config: TileConfig {
                tile_n: 16,
                tile_m: 4,
                block_size: 32,
            },
            variant: "v",
        };
        assert_ne!(k1, k2, "different tile_n must differ");
    }

    #[test]
    fn prepack_result_construction() {
        let bv = BufferView::from_raw(42);
        let r = PrepackResult {
            weight: bv,
            scales: bv,
            zero_points: None,
        };
        assert_eq!(r.weight, bv);
        assert!(r.zero_points.is_none());
    }

    #[test]
    fn prepack_result_with_zero_points() {
        let bv = BufferView::from_raw(1);
        let bv2 = BufferView::from_raw(2);
        let r = PrepackResult {
            weight: bv,
            scales: bv,
            zero_points: Some(bv2),
        };
        assert_eq!(r.zero_points.unwrap().as_raw(), 2);
    }

    #[test]
    fn plan_default_empty() {
        let p = Plan::default();
        assert!(p.nodes.is_empty());
        assert!(p.prepack_requests.is_empty());
        assert!(p.prepacked.is_empty());
    }

    #[test]
    fn prepack_request_stores_pack_fn() {
        let config = TileConfig {
            tile_n: 8,
            tile_m: 4,
            block_size: 32,
        };
        let req = PrepackRequest {
            key: PackKey {
                initializer: "w".into(),
                config: config.clone(),
                variant: "v",
            },
            weight: TensorRef {
                name: "weight".into(),
                desc: None,
                is_initializer: true,
            },
            scales: TensorRef {
                name: "scales".into(),
                desc: None,
                is_initializer: true,
            },
            zero_points: None,
            pack_fn: dummy_pack_fn,
        };
        let weight_data: &[u8] = &[1, 2, 3, 4];
        let scale_data: &[u8] = &[5, 6];
        let out = (req.pack_fn)(PackInput {
            weight: weight_data,
            scales: scale_data,
            zero_points: None,
            config: &config,
        });
        assert_eq!(out.packed_weight, &[1, 2, 3, 4]);
        assert_eq!(out.packed_scales, &[5, 6]);
        assert!(out.packed_zero_points.is_none());
    }

    // ── Seam 4: indirect dispatch vocab ──────────────────────────────────────────────────

    #[test]
    fn indirect_kernel_request_construction() {
        let bv = BufferView::from_raw(5);
        let req = IndirectKernelRequest {
            shader: "qmoe_dispatch",
            spec_constants: vec![64],
            push_constants: vec![],
            bindings: vec![bv],
            dispatch_buffer: bv,
            dispatch_offset: 0,
        };
        assert_eq!(req.shader, "qmoe_dispatch");
        assert_eq!(req.dispatch_buffer.as_raw(), 5);
        assert_eq!(req.dispatch_offset, 0);
    }

    // ── Default DispatchContext method smoke tests ────────────────────────────────────────

    struct StubCtx;
    impl DispatchContext for StubCtx {
        fn resolve(&mut self, _r: &TensorRef) -> EpResult<BufferView> {
            Ok(BufferView::from_raw(0))
        }
        fn bind_output(&mut self, _o: &OutRef, _d: TensorDesc) -> EpResult<BufferView> {
            Ok(BufferView::from_raw(0))
        }
        fn alloc_temp(&mut self, _d: TensorDesc) -> EpResult<BufferView> {
            Ok(BufferView::from_raw(0))
        }
        fn dispatch(&mut self, _k: KernelRequest) -> EpResult<()> {
            Ok(())
        }
        fn read_const_i64(&self, _r: &TensorRef) -> Option<Vec<i64>> {
            None
        }
    }

    #[test]
    fn default_resolve_prepacked_returns_err() {
        let ctx = StubCtx;
        let key = PackKey {
            initializer: "w".into(),
            config: TileConfig {
                tile_n: 8,
                tile_m: 4,
                block_size: 32,
            },
            variant: "v",
        };
        let result = ctx.resolve_prepacked(&key);
        assert!(result.is_err(), "default should return Err");
        match result {
            Err(EpError::Internal(_)) => {}
            other => panic!("expected Internal, got {:?}", other),
        }
    }

    #[test]
    fn default_bind_aliased_output_resolves_input() {
        let mut ctx = StubCtx;
        let input = TensorRef {
            name: "past_key".into(),
            desc: None,
            is_initializer: false,
        };
        let out = OutRef {
            name: "present_key".into(),
            desc: None,
        };
        let result =
            ctx.bind_aliased_output(&input, &out, TensorDesc::new(DType::F16, vec![1, 1, 1, 96]));
        assert!(
            result.is_ok(),
            "default aliased should return resolve result"
        );
    }

    #[test]
    fn default_dispatch_indirect_returns_err() {
        let mut ctx = StubCtx;
        let bv = BufferView::from_raw(0);
        let req = IndirectKernelRequest {
            shader: "s",
            spec_constants: vec![],
            push_constants: vec![],
            bindings: vec![],
            dispatch_buffer: bv,
            dispatch_offset: 0,
        };
        let result = ctx.dispatch_indirect(req);
        assert!(
            result.is_err(),
            "default dispatch_indirect should return Err"
        );
        match result {
            Err(EpError::Internal(_)) => {}
            other => panic!("expected Internal, got {:?}", other),
        }
    }

    // ══════════════════════════════════════════════════════════════════════════════════════
    // Issue #18 — DeviceIdentity / DeviceKey: the fail-closed contract.
    //
    // Every test below is a property of the *key*, not of the selector. The selector was already
    // right at PR #54's rejected head; what was wrong is that everything downstream of it kept
    // comparing device NAMES, and a name is shared by every card of a model. These tests are what
    // make "keyed by identity" checkable rather than asserted.
    // ══════════════════════════════════════════════════════════════════════════════════════

    fn info_named(name: &str, physical_index: usize, uuid: Option<&str>) -> DeviceInfo {
        DeviceInfo {
            index: physical_index,
            name: name.to_string(),
            vendor_id: 0x10de,
            device_id: 0x27a0,
            api_version: "1.3.290".to_string(),
            driver_version: "560.94".to_string(),
            kind: DeviceKind::Discrete,
            identity: DeviceIdentity {
                uuid: uuid.map(str::to_string),
                luid: None,
                pci: None,
            },
        }
    }

    #[test]
    fn two_identically_named_devices_with_different_uuids_are_different_keys() {
        // THE DEFECT, stated as a test. Before issue #18 the frame ledger keyed on this name, so
        // these two physical cards collapsed into one entry and a SHARED declaration on each
        // reported one frame instead of two.
        let a = info_named(
            "NVIDIA RTX A1000 Laptop GPU",
            0,
            Some("a".repeat(32).as_str()),
        );
        let b = info_named(
            "NVIDIA RTX A1000 Laptop GPU",
            1,
            Some("b".repeat(32).as_str()),
        );
        assert_eq!(a.name, b.name, "the premise: the NAMES are byte-identical");
        assert_ne!(a.key(), b.key(), "the identities must not be");
        assert_ne!(a.key().canonical(), b.key().canonical());
    }

    #[test]
    fn the_same_uuid_is_the_same_key_regardless_of_enumeration_position() {
        // The other half: a stable identity must survive the enumeration order changing, which is
        // exactly what a driver update or a VK_ICD_FILENAMES edit does. Keying on the index would
        // make a proof stop applying to the device that produced it.
        let u = "0123456789abcdef0123456789abcdef";
        let first = info_named("Some GPU", 0, Some(u));
        let moved = info_named("Some GPU", 3, Some(u));
        assert_eq!(first.key(), moved.key());
    }

    #[test]
    fn a_device_with_no_uuid_falls_back_without_collapsing() {
        // FAIL-CLOSED PROPERTY 1: the fallback must never merge two devices. If it did, this
        // whole change would reintroduce the collision on exactly the platforms (MoltenVK, some
        // Android ICDs) least able to notice.
        let a = info_named("Mali-G78", 0, None);
        let b = info_named("Mali-G78", 1, None);
        assert_ne!(a.key(), b.key());
        assert_eq!(
            a.key(),
            DeviceKey::Unidentified {
                name: "Mali-G78".to_string(),
                physical_index: 0
            }
        );
    }

    #[test]
    fn the_fallback_key_never_claims_to_be_an_identity() {
        // FAIL-CLOSED PROPERTY 2: it must never *prove* either. `is_identified()` is what
        // `registry::device_state` reads to refuse `Proven`, and the `unidentified:` prefix is
        // what stops a human or a future comparison from reading the fallback as a UUID.
        let unnamed = info_named("Mali-G78", 0, None);
        assert!(!unnamed.key().is_identified());
        assert_eq!(unnamed.key().uuid(), None);
        assert!(unnamed.key().canonical().starts_with("unidentified:"));

        let identified = info_named("Mali-G78", 0, Some(&"c".repeat(32)));
        assert!(identified.key().is_identified());
        assert_eq!(identified.key().uuid(), Some("c".repeat(32).as_str()));
        assert!(identified.key().canonical().starts_with("uuid:"));
    }

    #[test]
    fn canonical_round_trips_through_display() {
        let k = DeviceKey::Uuid("d".repeat(32));
        assert_eq!(format!("{k}"), k.canonical());
        let u = DeviceKey::Unidentified {
            name: "A Device".to_string(),
            physical_index: 2,
        };
        assert_eq!(format!("{u}"), "unidentified:A Device#2");
    }

    #[test]
    fn synthetic_test_keys_are_distinct_per_seed_and_shaped_like_uuids() {
        // The identical-name tests elsewhere in this crate depend on this: same name, different
        // seed, different key. A helper that collided would make those tests pass vacuously.
        let a = DeviceKey::synthetic_for_test("Device A");
        let b = DeviceKey::synthetic_for_test("Device B");
        assert_ne!(a, b);
        assert_eq!(DeviceKey::synthetic_for_test("Device A"), a);
        for k in [&a, &b] {
            let u = k.uuid().expect("synthetic keys are uuid keys");
            assert_eq!(u.len(), 32, "{u} must be 32 hex characters");
            assert!(
                u.bytes()
                    .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
            );
        }
    }

    #[test]
    fn device_identity_from_uuid_is_the_only_way_a_uuid_gets_in() {
        let id = DeviceIdentity::from_uuid("e".repeat(32));
        assert_eq!(id.uuid.as_deref(), Some("e".repeat(32).as_str()));
        assert_eq!(id.luid, None);
        assert_eq!(id.pci, None);
        // And the absent case is genuinely absent — not an empty string that would compare equal
        // to another absent one and rebuild the collision.
        let none = DeviceIdentity::default();
        assert_eq!(none.uuid, None);
        assert_eq!(
            none.key("X", 4),
            DeviceKey::Unidentified {
                name: "X".to_string(),
                physical_index: 4
            }
        );
        // An empty-string UUID is treated as absence, not as an identity every empty-string
        // device would share.
        let empty = DeviceIdentity::from_uuid("");
        assert_eq!(
            empty.key("X", 4),
            DeviceKey::Unidentified {
                name: "X".to_string(),
                physical_index: 4
            }
        );
    }
}
