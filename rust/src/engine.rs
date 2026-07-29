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
    fn bind_aliased_output(&mut self, input: &TensorRef, out: &OutRef) -> EpResult<BufferView> {
        let _ = out;
        self.resolve(input)
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
}

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

    /// Look up an embedded SPIR-V module by shader stem.
    pub fn find(stem: &str) -> Option<&'static [u8]> {
        SHADER_MODULES
            .iter()
            .find(|(name, _)| *name == stem)
            .map(|(_, bytes)| *bytes)
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

    #[test]
    fn no_shaders_are_embedded_yet() {
        assert!(shaders::SHADER_MODULES.is_empty());
        assert!(shaders::find("elementwise_binary").is_none());
    }

    // ── DESIGN.md §7.8 condition 3: shader-less artifact guards ─────────────────────────

    #[test]
    fn has_any_returns_false_in_this_build() {
        // This build was compiled without shaders (no glslc on the build machine, or
        // ALLOW_MISSING_GLSLC=1). Verify the guard function reflects the build state correctly.
        // When the first real shader is added and compiled, this test will fail — and should then
        // be updated to assert `has_any() == true`. The failure is the signal that condition 3's
        // probe path changes from "always false" to "real check against build state".
        assert!(
            !shaders::has_any(),
            "has_any() must be false in a shader-less build (SHADER_MODULES is empty)"
        );
    }

    #[test]
    fn probe_devices_returns_empty_when_no_shaders() {
        // DESIGN.md §7.8 condition 3: a shader-less build must advertise zero devices.
        // In this build, SHADER_MODULES is empty, so probe_devices() must return empty
        // regardless of whether a Vulkan ICD is present.
        //
        // This test is currently green because shaders::has_any() == false. When the first
        // shader is compiled in, this test will fail and should be removed (at that point
        // probe_devices() correctly returns real devices on Trinity's lavapipe lane).
        if !shaders::has_any() {
            let devices = probe_devices();
            assert!(
                devices.is_empty(),
                "shader-less build must advertise zero devices (got {} devices)",
                devices.len()
            );
        }
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
        let result = ctx.bind_aliased_output(&input, &out);
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
}
