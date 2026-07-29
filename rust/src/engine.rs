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

/// A compiled fused subgraph: topologically ordered nodes plus the I/O binding table.
///
/// Owned by the `OrtNodeComputeInfo` that `ep.rs` hands back to ORT, and dropped when ORT releases
/// it. Prepacked weights and cached command-buffer recordings hang off this once Switch lands
/// them.
#[derive(Debug, Clone, Default)]
pub struct Plan {
    pub nodes: Vec<NodeDesc>,
    pub inputs: Vec<TensorRef>,
    pub outputs: Vec<OutRef>,
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

/// **STUB — Switch replaces the body.**
///
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
///
/// Returning an empty list today is exactly what a driverless machine will do tomorrow, so the
/// factory path this stub feeds is the same code that ships.
pub fn probe_devices() -> Vec<DeviceInfo> {
    log::warn!(
        "Vulkan device probe is a stub (M0): no physical devices enumerated, so the EP will \
         advertise nothing and every node stays on the CPU EP. Switch replaces \
         engine::probe_devices with a real vkEnumeratePhysicalDevices + capability gate."
    );
    Vec::new()
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
    fn probe_is_a_stub_that_never_fails() {
        assert!(probe_devices().is_empty());
    }

    #[test]
    fn no_shaders_are_embedded_yet() {
        assert!(shaders::SHADER_MODULES.is_empty());
        assert!(shaders::find("elementwise_binary").is_none());
    }
}
