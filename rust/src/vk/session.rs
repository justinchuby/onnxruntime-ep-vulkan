//! Session-level Vulkan resources — device, allocator, command pool, pipeline cache.
//!
//! [`VulkanSession`] is created once per ORT session (in [`crate::ep::VulkanEp::new`]) and owns
//! all Vulkan state reused across `Compile` and `Compute` calls. It is stored in `VulkanEp`
//! behind `Option<Box<VulkanSession>>`; [`crate::ep::SubgraphComputeInfo`] holds a raw pointer
//! into the `Box` which is valid because ORT guarantees the EP outlives all compiled compute
//! infos (`ReleaseNodeComputeInfos` is called before the EP is destroyed).
//!
//! The dispatch path here is the same sequence as `dispatch_integration.rs` but reads from and
//! writes to ORT-allocated CPU tensor buffers rather than test-generated data.

use std::collections::HashMap;
use std::sync::Arc;

use ash::vk;

use super::{
    alloc::{Allocator, GpuBuffer, MemClass, record_download, record_upload},
    barrier::{Access, BufferDep},
    cmd::{CommandPool, create_and_submit, wait_fence_then_destroy},
    device::{Device, register_ep_device},
    instance::{CapableDevice, Instance},
    pipeline::{DispatchDescriptorPool, PipelineCache, PipelineKey},
    timestamp::GpuQueryPool,
};
use crate::{
    engine::{
        BufferView, DType, DispatchContext, EpResult, KernelRequest, NodeDesc, OutRef, TensorDesc,
        TensorRef,
    },
    ep::EpOptions,
    sys::ort,
    trace::{self, GpuInterval, GpuTimestampCalibration, GpuTimestampReport, Phase, Transfer},
};

// ──────────────────────────────────────────────────────────────────────────────
// CompiledKernel
// ──────────────────────────────────────────────────────────────────────────────

/// Recipe for re-deriving push constants and workgroups at each Compute call.
///
/// Stored on a `CompiledKernel` when that kernel's input shapes were symbolic at Compile time.
/// `dispatch_ort` reads the actual tensor shapes from ORT, patches the stored `NodeDesc`,
/// re-runs the translate handler through a [`ShapeOnlyRecorder`], and uses the resulting
/// push constants and workgroups for the actual `vkCmdDispatch`.
///
/// The `node_desc` carries the symbolic shapes in `TensorRef::desc::shape` (dims are -1 for
/// symbolic axes). At Compute time those -1s are replaced with the concrete values reported
/// by `KernelContext_GetInput` + `GetTensorTypeAndShape`.
#[derive(Clone)]
pub(crate) struct DynKernelRecipe {
    /// Node description with symbolic dims as -1 in `TensorRef::desc::shape`.
    pub(crate) node_desc: crate::engine::NodeDesc,
    /// The registry spec this kernel was claimed against (static lifetime).
    pub(crate) spec: &'static crate::registry::OpSpec,
}

/// One pre-compiled dispatch template from the Compile phase.
///
/// The `bindings` vec encodes indices into the subgraph's input/output tables rather than live
/// [`BufferView`] handles — those change per Compute call. Encoding:
/// - `token < n_plan_inputs` → GPU input buffer at index `token`
/// - `token >= n_plan_inputs` → GPU output buffer at index `token - n_plan_inputs`
///
/// For static-shape kernels `dyn_recipe` is `None` and `push_constants`/`workgroups` are baked.
/// For dynamic-shape kernels `dyn_recipe` is `Some` and push constants/workgroups are recomputed
/// at each Compute call from the real ORT tensor shapes.
#[derive(Clone)]
pub(crate) struct CompiledKernel {
    /// Shader stem, e.g. `"ew_binary_add_f32"`.
    pub(crate) shader: &'static str,
    /// Specialization constants baked at Compile time.
    pub(crate) spec_constants: Vec<u32>,
    /// Push-constant bytes baked from static shapes. Empty when `dyn_recipe` is `Some`.
    pub(crate) push_constants: Vec<u8>,
    /// Workgroup counts computed at Compile time. `[0, 0, 0]` when `dyn_recipe` is `Some`.
    pub(crate) workgroups: [u32; 3],
    /// Buffer index tokens. See struct doc for encoding.
    pub(crate) bindings: Vec<u64>,
    /// Byte sizes of temporary GPU buffers allocated via `alloc_temp`.  These are scratch
    /// buffers used by translate handlers (e.g. `skip_norm`'s slot-3 residual write when the
    /// caller does not request slot 3).  They sit *above* the output-buffer range in the token
    /// encoding: `token - n_plan_inputs >= n_ort_outputs` → temp buffer at that index.
    ///
    /// For static kernels this is filled during `CompileRecorder::alloc_temp`; for dynamic
    /// kernels it is derived from `ShapeOnlyRecorder::temp_descs` at Compute time.
    pub(crate) temp_byte_sizes: Vec<u64>,
    /// Number of subgraph-level inputs (= `plan.inputs.len()`).
    pub(crate) n_plan_inputs: usize,
    /// Recipe for recomputing push constants and workgroups at Compute time.
    /// `None` for statically-shaped kernels. `Some` for dynamically-shaped kernels.
    #[allow(clippy::box_collection)]
    // `DynKernelRecipe` is large; boxing avoids inflating `Vec<CompiledKernel>`
    pub(crate) dyn_recipe: Option<Box<DynKernelRecipe>>,
}

/// A `CompiledKernel` cannot derive `Debug` automatically because `DynKernelRecipe` contains
/// a `&'static OpSpec` whose internal function-pointer fields do not implement `Debug`.
impl std::fmt::Debug for CompiledKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledKernel")
            .field("shader", &self.shader)
            .field("spec_constants", &self.spec_constants)
            .field("push_constants_len", &self.push_constants.len())
            .field("workgroups", &self.workgroups)
            .field("bindings", &self.bindings)
            .field("n_plan_inputs", &self.n_plan_inputs)
            .field(
                "dyn_recipe",
                &self
                    .dyn_recipe
                    .as_ref()
                    .map(|r| r.node_desc.op_type.as_str()),
            )
            .finish()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// CompileRecorder — recording DispatchContext for compile_impl
// ──────────────────────────────────────────────────────────────────────────────

/// Sentinel token for an absent optional input: outside every real token range, so no
/// binding, patch, or allocation step resolves it to a tensor.
pub(crate) const NO_TOKEN: u64 = u64::MAX;

/// A [`DispatchContext`] that records kernel templates without issuing Vulkan commands.
///
/// Used by `compile_impl` to run the registry's translate handlers and extract
/// [`CompiledKernel`]s.
///
/// # Token encoding
///
/// For single-node islands the recorder uses *positional* counting (token = input/output
/// index in the node's own slot list). For multi-node islands it switches to *name-based*
/// assignment: external inputs and outputs receive tokens determined by their position in
/// `plan.inputs`/`plan.outputs`, and intermediate outputs (values produced by one node and
/// consumed by another within the same island) receive tokens beyond the output range.
/// This ensures each distinct value gets a unique, stable token regardless of where it
/// appears in the iteration order, and prevents `ShapeOnlyRecorder` from re-assigning
/// an intermediate output the same slot token as an external output.
pub(crate) struct CompileRecorder {
    n_plan_inputs: usize,
    /// Optional name→token map. `None` for single-node islands (positional mode).
    name_map: Option<Arc<HashMap<String, u64>>>,
    /// First token reserved for anonymous `alloc_temp` allocations, after all named outputs.
    first_temp_token: usize,
    /// Positional fallback counters — used when the name map is absent or when a name is empty.
    next_resolve: usize,
    next_bind: usize,
    next_temp: usize,
    /// Temporary scratch buffer byte sizes accumulated between translate calls and flushed
    /// into `CompiledKernel::temp_byte_sizes` on each `dispatch()` call.
    pending_temp_sizes: Vec<u64>,
    pub(crate) kernels: Vec<CompiledKernel>,
    /// Byte sizes for intermediate (inter-node) output buffers, keyed by token.
    /// Populated during static-shape translate runs from `bind_output` descriptors.
    pub(crate) intermediate_sizes: HashMap<u64, u64>,
}

impl CompileRecorder {
    /// Create a recorder for a single-node island (positional mode).
    pub(crate) fn new(n_plan_inputs: usize) -> Self {
        Self {
            n_plan_inputs,
            name_map: None,
            first_temp_token: 0,
            next_resolve: 0,
            next_bind: 0,
            next_temp: 0,
            pending_temp_sizes: Vec::new(),
            kernels: Vec::new(),
            intermediate_sizes: HashMap::new(),
        }
    }

    /// Create a recorder for a multi-node island using the pre-built name→token map.
    ///
    /// `first_temp_token` is the first token available for anonymous `alloc_temp` calls;
    /// it must be `n_plan_inputs + n_plan_outputs + n_intermediates` so that temps land
    /// in their own range and do not alias any named buffer.
    pub(crate) fn new_named(
        n_plan_inputs: usize,
        name_map: Arc<HashMap<String, u64>>,
        first_temp_token: usize,
    ) -> Self {
        Self {
            n_plan_inputs,
            name_map: Some(name_map),
            first_temp_token,
            next_resolve: 0,
            next_bind: 0,
            next_temp: 0,
            pending_temp_sizes: Vec::new(),
            kernels: Vec::new(),
            intermediate_sizes: HashMap::new(),
        }
    }

    /// Look up an input token: name-based first, then positional fallback.
    fn resolve_token(&mut self, name: &str) -> u64 {
        if let Some(map) = &self.name_map {
            if !name.is_empty() {
                if let Some(&t) = map.get(name) {
                    return t;
                }
            }
        }
        let t = self.next_resolve as u64;
        self.next_resolve += 1;
        t
    }

    /// Look up an output token and record its byte size; name-based first, then positional.
    fn bind_token(&mut self, name: &str, byte_size: u64) -> u64 {
        if let Some(map) = &self.name_map {
            if !name.is_empty() {
                if let Some(&t) = map.get(name) {
                    // Record the size so compile_impl can populate static_intermediate_byte_sizes.
                    if t >= (self.n_plan_inputs) as u64 {
                        self.intermediate_sizes.insert(t, byte_size);
                    }
                    return t;
                }
            }
        }
        let t = (self.n_plan_inputs + self.next_bind) as u64;
        self.next_bind += 1;
        t
    }

    /// Record a dynamic-shape kernel: allocate binding tokens without running the translate
    /// handler, and store the `DynKernelRecipe` for Compute-time re-run.
    ///
    /// Called from `compile_impl` when the translate handler fails due to symbolic shapes.
    /// With a name map the tokens are resolved by name; without one they are positional.
    pub(crate) fn push_dynamic_kernel(
        &mut self,
        node_desc: crate::engine::NodeDesc,
        spec: &'static crate::registry::OpSpec,
    ) {
        let mut bindings = Vec::with_capacity(node_desc.inputs.len() + node_desc.outputs.len());
        for inp in &node_desc.inputs {
            // An empty name is an *absent* optional input. It has no plan slot, so it has no
            // token, and the positional fallback would otherwise hand it token 0 — the same
            // token as the first real input, which the Compute-time patch loop would then
            // read a desc off. `NO_TOKEN` is outside every token range, so both the patch
            // loop and the binder skip it instead of resolving it to the wrong tensor.
            bindings.push(if inp.name.is_empty() {
                NO_TOKEN
            } else {
                self.resolve_token(&inp.name)
            });
        }
        for out in &node_desc.outputs {
            bindings.push(self.bind_token(&out.name, 0));
        }

        self.kernels.push(CompiledKernel {
            shader: "", // filled at Compute time by ShapeOnlyRecorder
            spec_constants: vec![],
            push_constants: vec![],
            workgroups: [0, 0, 0],
            bindings,
            temp_byte_sizes: Vec::new(), // dynamic kernels derive temp sizes at Compute time
            n_plan_inputs: self.n_plan_inputs,
            dyn_recipe: Some(Box::new(DynKernelRecipe { node_desc, spec })),
        });
    }
}

impl DispatchContext for CompileRecorder {
    fn resolve(&mut self, r: &TensorRef) -> EpResult<BufferView> {
        Ok(BufferView::from_raw(self.resolve_token(&r.name)))
    }

    fn bind_output(&mut self, o: &OutRef, desc: TensorDesc) -> EpResult<BufferView> {
        let size = desc.byte_size().unwrap_or(0) as u64;
        Ok(BufferView::from_raw(self.bind_token(&o.name, size)))
    }

    fn alloc_temp(&mut self, desc: TensorDesc) -> EpResult<BufferView> {
        let token = if self.name_map.is_some() {
            // Named mode: temps start after all named buffers.
            let t = (self.first_temp_token + self.next_temp) as u64;
            self.next_temp += 1;
            t
        } else {
            // Positional mode: temps share the bind counter.
            let t = (self.n_plan_inputs + self.next_bind) as u64;
            self.next_bind += 1;
            t
        };
        self.pending_temp_sizes
            .push(desc.byte_size().unwrap_or(0) as u64);
        Ok(BufferView::from_raw(token))
    }

    fn dispatch(&mut self, k: KernelRequest) -> EpResult<()> {
        self.kernels.push(CompiledKernel {
            shader: k.shader,
            spec_constants: k.spec_constants,
            push_constants: k.push_constants,
            workgroups: k.workgroups,
            bindings: k.bindings.iter().map(|b| b.as_raw()).collect(),
            temp_byte_sizes: std::mem::take(&mut self.pending_temp_sizes),
            n_plan_inputs: self.n_plan_inputs,
            dyn_recipe: None,
        });
        Ok(())
    }

    fn read_const_i64(&self, _r: &TensorRef) -> Option<Vec<i64>> {
        None
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ShapeOnlyRecorder — lightweight recorder for Compute-time translate re-runs
// ──────────────────────────────────────────────────────────────────────────────

/// Captures `push_constants`, `workgroups`, `spec_constants`, and output [`TensorDesc`]s from a
/// translate-handler re-run. Used by `dispatch_ort` for dynamic-shape kernels.
///
/// # Binding correction
///
/// `push_dynamic_kernel` assigns binding tokens positionally (one per input slot, one per output
/// slot) because it cannot run the translate handler at Compile time. Some translate handlers
/// produce a **different** binding sequence: e.g., `MatMulNBits` without `zero_points` runs
/// `resolve()` three times (A, B, scales) but then binds scales *again* as an inert placeholder
/// for the declared-but-unused zero-points descriptor slot, producing five binding tokens for a
/// three-input node.  If we used `kernel.bindings` from `push_dynamic_kernel` at Compute time
/// the pipeline layout would have the wrong number of entries and the output descriptor would
/// be out of range — the shader writes to binding 4 (undefined) and the output buffer stays zero.
///
/// Fix: this recorder captures `KernelRequest.bindings` from the re-run translate, giving the
/// correct per-kernel binding sequence with any extra or duplicated tokens the translate inserts.
/// `dispatch_ort` uses those captured bindings instead of `kernel.bindings` for dynamic kernels.
///
/// # Multi-node islands
///
/// For multi-node islands each node re-runs with `new_named()` so that resolve/bind_output
/// calls go through the island-wide name→token map, assigning intermediate-output tokens from
/// `n_plan_inputs + n_plan_outputs + k` rather than restarting from zero each time.
/// `output_descs` is then keyed by token rather than being a flat list.
struct ShapeOnlyRecorder {
    n_plan_inputs: usize,
    /// Optional name→token map. `None` for single-node islands (positional mode).
    name_map: Option<Arc<HashMap<String, u64>>>,
    /// First token reserved for anonymous `alloc_temp` calls (named mode only).
    first_temp_token: usize,
    /// Positional fallback counters.
    next_resolve: usize,
    next_bind: usize,
    next_temp: usize,
    /// Filled by `dispatch()` with push-constant bytes, workgroup counts, spec constants, and
    /// shader stem.  Binding tokens are in `captured_bindings`.
    #[allow(clippy::type_complexity)]
    pub captured: Option<(Vec<u8>, [u32; 3], Vec<u32>, &'static str)>,
    /// Binding tokens from the translate handler's `KernelRequest`, in descriptor-slot order.
    pub captured_bindings: Option<Vec<u64>>,
    /// Output `TensorDesc`s collected from `bind_output()`, keyed by token.
    /// Used by `dispatch_ort` to size both external ORT outputs and intermediate GPU buffers.
    pub output_desc_by_token: Vec<(u64, TensorDesc)>,
    /// Descriptors from `alloc_temp()` calls — scratch buffers not tied to ORT outputs.
    pub temp_descs: Vec<TensorDesc>,
    /// Aliased output pairs: (out_token, in_token).  Populated by `bind_aliased_output`.
    /// `dispatch_ort` uses this to borrow an input buffer for the matching output slot,
    /// avoiding a redundant device allocation for in-place KV cache updates.
    pub aliased_pairs: Vec<(u64, u64)>,
}

impl ShapeOnlyRecorder {
    /// Single-node island: positional mode.
    fn new(n_plan_inputs: usize) -> Self {
        Self {
            n_plan_inputs,
            name_map: None,
            first_temp_token: 0,
            next_resolve: 0,
            next_bind: 0,
            next_temp: 0,
            captured: None,
            captured_bindings: None,
            output_desc_by_token: Vec::new(),
            temp_descs: Vec::new(),
            aliased_pairs: Vec::new(),
        }
    }

    /// Multi-node island: name-based mode.
    fn new_named(
        n_plan_inputs: usize,
        name_map: Arc<HashMap<String, u64>>,
        first_temp_token: usize,
    ) -> Self {
        Self {
            n_plan_inputs,
            name_map: Some(name_map),
            first_temp_token,
            next_resolve: 0,
            next_bind: 0,
            next_temp: 0,
            captured: None,
            captured_bindings: None,
            output_desc_by_token: Vec::new(),
            temp_descs: Vec::new(),
            aliased_pairs: Vec::new(),
        }
    }

    fn resolve_token(&mut self, name: &str) -> u64 {
        if let Some(map) = &self.name_map {
            if !name.is_empty() {
                if let Some(&t) = map.get(name) {
                    return t;
                }
            }
        }
        let t = self.next_resolve as u64;
        self.next_resolve += 1;
        t
    }

    fn bind_token(&mut self, name: &str, desc: TensorDesc) -> u64 {
        if let Some(map) = &self.name_map {
            if !name.is_empty() {
                if let Some(&t) = map.get(name) {
                    self.output_desc_by_token.push((t, desc));
                    return t;
                }
            }
        }
        let t = (self.n_plan_inputs + self.next_bind) as u64;
        self.next_bind += 1;
        self.output_desc_by_token.push((t, desc));
        t
    }
}

impl DispatchContext for ShapeOnlyRecorder {
    fn resolve(&mut self, r: &TensorRef) -> EpResult<BufferView> {
        Ok(BufferView::from_raw(self.resolve_token(&r.name)))
    }

    fn bind_output(&mut self, o: &OutRef, desc: TensorDesc) -> EpResult<BufferView> {
        Ok(BufferView::from_raw(self.bind_token(&o.name, desc)))
    }

    /// Register an aliased output (in-place KV cache update).
    ///
    /// Records:
    /// - The output descriptor so `dispatch_ort` sets `actual_output_byte_sizes[j]` correctly.
    /// - The (out_token, in_token) pair so the output loop can borrow the input buffer
    ///   instead of allocating a new device buffer — the shader writes to the input in-place.
    ///
    /// Returns the input's token so the `KernelRequest` binding routes the shader to the
    /// right buffer (same buffer for both reads and writes).
    fn bind_aliased_output(
        &mut self,
        input: &TensorRef,
        out: &OutRef,
        desc: TensorDesc,
    ) -> EpResult<BufferView> {
        let out_token = self.bind_token(&out.name, desc);
        let in_token = self.resolve_token(&input.name);
        self.aliased_pairs.push((out_token, in_token));
        Ok(BufferView::from_raw(in_token))
    }

    fn alloc_temp(&mut self, desc: TensorDesc) -> EpResult<BufferView> {
        let token = if self.name_map.is_some() {
            let t = (self.first_temp_token + self.next_temp) as u64;
            self.next_temp += 1;
            t
        } else {
            let t = (self.n_plan_inputs + self.next_bind) as u64;
            self.next_bind += 1;
            t
        };
        self.temp_descs.push(desc);
        Ok(BufferView::from_raw(token))
    }

    fn dispatch(&mut self, k: KernelRequest) -> EpResult<()> {
        self.captured = Some((k.push_constants, k.workgroups, k.spec_constants, k.shader));
        self.captured_bindings = Some(k.bindings.iter().map(|b| b.as_raw()).collect());
        Ok(())
    }

    fn read_const_i64(&self, _r: &TensorRef) -> Option<Vec<i64>> {
        None
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Dynamic-dispatch helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Convert an ORT element data type to our `DType`.
fn dtype_from_ort_type(et: ort::ONNXTensorElementDataType) -> Option<DType> {
    match et {
        ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT => Some(DType::F32),
        ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 => Some(DType::F16),
        ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 => Some(DType::I64),
        ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 => Some(DType::I32),
        ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 => Some(DType::U8),
        ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL => Some(DType::Bool),
        _ => None,
    }
}

/// Read the dtype and concrete shape from a live ORT tensor value.
///
/// Returns `None` if any API call fails, the API function is unavailable, or the dtype is not
/// supported by this EP.
///
/// # Safety
/// `api` and `ort_val` must be live ORT pointers for the duration of this call.
unsafe fn read_tensor_desc_from_ort(
    api: *const ort::OrtApi,
    ort_val: *const ort::OrtValue,
) -> Option<TensorDesc> {
    // SAFETY: `api` is a live OrtApi; reading the immutable function table is a plain field read.
    // We check all five are present before proceeding so that no individual read fails silently.
    let (get_type_and_shape, get_elem_type, get_ndim, get_dims, release_info) = unsafe {
        match (
            (*api).GetTensorTypeAndShape,
            (*api).GetTensorElementType,
            (*api).GetDimensionsCount,
            (*api).GetDimensions,
            (*api).ReleaseTensorTypeAndShapeInfo,
        ) {
            (Some(a), Some(b), Some(c), Some(d), Some(e)) => (a, b, c, d, e),
            _ => return None,
        }
    };

    let mut info: *mut ort::OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
    // SAFETY: `ort_val` is live per the fn contract; `info` is a valid out-pointer.
    let st = unsafe { get_type_and_shape(ort_val, &mut info) };
    if !st.is_null() {
        // SAFETY: `api` and `st` are live; release_status forwards to `OrtApi::ReleaseStatus`.
        unsafe { crate::sys::release_status(api, st) };
        return None;
    }

    let mut et = ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    // SAFETY: `info` was just filled by a successful `GetTensorTypeAndShape`; `&mut et` is valid.
    let st = unsafe { get_elem_type(info, &mut et) };
    if !st.is_null() {
        // SAFETY: `api` and `st` are live.
        unsafe { crate::sys::release_status(api, st) };
        // SAFETY: `info` is a non-null, live `OrtTensorTypeAndShapeInfo` produced above.
        unsafe { release_info(info) };
        return None;
    }

    let mut ndim: usize = 0;
    // SAFETY: `info` is live and non-null; `&mut ndim` is a valid out-pointer.
    let st = unsafe { get_ndim(info, &mut ndim) };
    if !st.is_null() {
        // SAFETY: `api` and `st` are live.
        unsafe { crate::sys::release_status(api, st) };
        // SAFETY: `info` is a non-null, live `OrtTensorTypeAndShapeInfo` produced above.
        unsafe { release_info(info) };
        return None;
    }

    let mut dims = vec![0i64; ndim];
    if ndim > 0 {
        // SAFETY: `info` is live; `dims.as_mut_ptr()` points to exactly `ndim` i64 slots.
        let st = unsafe { get_dims(info, dims.as_mut_ptr(), ndim) };
        if !st.is_null() {
            // SAFETY: `api` and `st` are live.
            unsafe { crate::sys::release_status(api, st) };
            // SAFETY: `info` is a non-null, live `OrtTensorTypeAndShapeInfo` produced above.
            unsafe { release_info(info) };
            return None;
        }
    }

    // SAFETY: `info` is a non-null, live `OrtTensorTypeAndShapeInfo` produced above.
    unsafe { release_info(info) };

    let dtype = dtype_from_ort_type(et)?;
    Some(TensorDesc::new(dtype, dims))
}

// ──────────────────────────────────────────────────────────────────────────────
// VulkanSession
// ──────────────────────────────────────────────────────────────────────────────

/// Session-level Vulkan state, created once per ORT session.
///
/// **Ownership:** stored in `VulkanEp` behind `Option<Box<VulkanSession>>`. The `Box` gives a
/// stable address. `SubgraphComputeInfo` holds a `*mut VulkanSession` raw pointer; this is safe
/// because ORT guarantees the EP outlives all compiled compute infos.
///
/// **Drop order:** Rust drops fields in declaration order (top-to-bottom). Vulkan's teardown
/// contract requires child objects to be destroyed before the device, and the device before the
/// instance. The field order here encodes that contract explicitly.
pub(crate) struct VulkanSession {
    /// Per-device capabilities and metadata. Borrowed from the process-global §6.5 device owner.
    pub(crate) capable: &'static CapableDevice,
    /// Compiled pipelines. Must drop before `device` (uses device handles).
    pub(crate) pipeline_cache: PipelineCache,
    /// Command pool. Must drop before `device`.
    pub(crate) cmd_pool: CommandPool,
    /// Allocator (gpu-allocator). Must drop before `device`.
    pub(crate) alloc: Allocator,
    /// Logical device + compute queue. **Borrowed, not owned** (§6.5): one `VkDevice` per physical
    /// device per process, created by `vk::device::acquire_ep_device` and never destroyed. The
    /// session's own Vulkan children above are still destroyed first, which is what the Vulkan
    /// teardown contract requires; the device outliving them is always safe.
    pub(crate) device: &'static Device,
    /// Vulkan instance. Borrowed from the same owner; outlives the device by construction.
    pub(crate) instance: &'static Instance,
    /// Per-subgraph weight caches.  Key: subgraph ID → HashMap<(cpu_ptr, byte_size), GpuBuffer>.
    /// Populated lazily on the second inference through a subgraph; freed when the subgraph's
    /// `SubgraphComputeInfo` is released (via `release_weight_cache`).
    weight_caches: HashMap<u64, HashMap<(usize, u64), GpuBuffer>>,
    /// A permanent 4-byte device-local STORAGE_BUFFER used as a descriptor placeholder for
    /// zero-element inputs (e.g., Phi-3.5 KV-cache tensors on a first-token prefill, whose
    /// shape is `[1, H, 0, D]`).  A zero-byte VkBuffer is invalid; Vulkan also requires
    /// `VkDescriptorBufferInfo::range > 0`.  The shader never accesses this buffer because
    /// the outer dimension is 0 — it is only present to satisfy the descriptor-write constraint.
    /// Freed explicitly in `Drop` before `alloc` drops.
    zero_elem_placeholder: Option<GpuBuffer>,
}

impl VulkanSession {
    /// Create a session, selecting the best device respecting `EpOptions`.
    ///
    /// Returns `None` when no capable device is available (no Vulkan loader, no ICD, all devices
    /// fail the §7.2 gate).
    ///
    /// # Safety
    /// The Vulkan SDK dynamic library must remain loaded for the session's entire lifetime.
    pub(crate) unsafe fn create(options: &EpOptions) -> Option<Self> {
        // §6.5: the (instance, device) pair is process-global and never destroyed. The session
        // borrows it; it does not create or own it. See `vk::device::acquire_ep_device`.
        // SAFETY: the loader stays loaded for the process lifetime (the owner is leaked).
        let owner = unsafe {
            crate::vk::device::acquire_ep_device(
                options.bound_physical_index,
                options.device_index,
                options.enable_validation,
                options.force_legacy_barriers,
            )
        }?;
        let instance: &'static Instance = owner.instance;
        let device: &'static Device = &owner.device;
        let capable: &'static CapableDevice = &owner.capable;

        // SAFETY: instance and device are live; physical_device belongs to instance.
        let mut alloc =
            unsafe { Allocator::new(instance.ash(), device.physical_device(), device.ash()) }?;

        // SAFETY: device is live; compute_queue_family is valid.
        let cmd_pool = unsafe { CommandPool::new(device.ash(), device.compute_queue_family()) }?;

        // SAFETY: device is live.
        let pipeline_cache = unsafe { PipelineCache::new(device.ash(), &[]) }?;

        // §6.5: register the EP device so other components (e.g., host_device_memory) can share
        // it without creating their own VkDevice.
        // SAFETY: `device` is the process-global owner's device (`acquire_ep_device`), which is
        // leaked and therefore outlives every consumer, including the process-global
        // `HandleRegistry` / device-memory providers.
        register_ep_device(device);

        // §6.5 seam: offer the EP device context to host_device_memory so that ORT-tensor
        // allocations land on the SAME VkDevice the compute kernels use.  Called BEFORE any ORT
        // tensor is allocated — `ensure_registered` is lazy and consults OFFERED at first use, and
        // `VulkanSession::create` runs inside `CreateEp`, which precedes every ORT allocation.
        //
        // KEY (index-space, R12): `ensure_registered` is called by the allocator with the
        // *factory's advertised* device index (`HandleRegistry::set_device_index`, factory.rs) —
        // which is the physical `vkEnumeratePhysicalDevices` index, i.e. `capable.info.index`.
        // It is NOT the sorted-capables selector `idx` (the `ONNXRUNTIME_EP_VULKAN_DEVICE` value).
        // Those two agree only when enumerate order equals best-first sort order; keying the offer
        // on `idx` left `offered_device()` returning None on any desk where they diverge, so the
        // provider silently built a SECOND VkDevice (`alloc_device_frame = SPLIT-DEVICE`). Offer
        // under `capable.info.index` so the registry lookup finds it and the frame becomes SHARED.
        //
        // LIFETIME (the reason this is now unconditional): the offered context holds *borrowed*,
        // bitwise-cloned ash handles, and the consumer (`host_device_memory::PROVIDERS`) is
        // process-global. That was a use-after-free while each session created and destroyed its
        // own VkDevice — measured as STATUS_ACCESS_VIOLATION (0xC0000005) on a second session's
        // inference. It is not one now: `acquire_ep_device` makes the (instance, device) pair
        // process-global and leaks it, so (a) no `vkDestroyDevice` can run under the provider, and
        // (b) session N>0 on the same physical device runs its kernels on the *same* VkDevice the
        // provider bound its buffers on. Both halves of the old hazard are gone, so the seam is
        // closed for any session count and needs no env gate.
        let ctx = crate::vk::device::SessionSharedCtx {
            instance: instance.ash().clone(),
            ash_device: device.ash().clone(),
            physical_device: device.physical_device(),
            compute_queue: device.compute_queue(),
            compute_queue_family: device.compute_queue_family(),
            is_uma: capable.caps.is_uma,
            name: capable.info.name.clone(),
        };
        crate::vk::host_device_memory::offer_shared_device(
            capable.info.index,
            std::sync::Arc::new(ctx),
        );
        log::info!(
            "§6.5: offered the EP VkDevice for '{}' under device index {} (physical enumerate \
             index). The device-memory provider adopts it only if ORT asks for an allocator on \
             that same index; a different index means ORT selected a different EP device than \
             this session opened, and the run correctly reports SPLIT-DEVICE.",
            capable.info.name,
            capable.info.index,
        );

        // Allocate the 4-byte zero-element placeholder buffer.  A zero-element tensor maps
        // to this buffer in descriptor writes — Vulkan requires a non-null buffer handle and
        // range > 0.  The shader never touches it (outer dim = 0 at compute time).
        // SAFETY: alloc and device are live; 4 bytes is always a valid allocation size.
        let zero_elem_placeholder = unsafe { alloc.alloc_device("zero_elem_placeholder", 4) };
        if zero_elem_placeholder.is_none() {
            log::error!("VulkanSession::create: failed to allocate zero-element placeholder");
            return None;
        }

        Some(VulkanSession {
            capable,
            pipeline_cache,
            cmd_pool,
            alloc,
            device,
            instance,
            weight_caches: HashMap::new(),
            zero_elem_placeholder,
        })
    }

    /// Free all cached GPU weight buffers for a subgraph that is being released.
    ///
    /// Called from `SubgraphComputeInfo`'s `Drop` implementation in `ep.rs`.
    ///
    /// # Safety
    /// The caller must ensure that no GPU work submitted by `subgraph_id` is still in flight
    /// — i.e., the fence for the last `Compute` call has already signalled.  ORT serialises
    /// `Compute` and `ReleaseNodeComputeInfos` (the latter is never called concurrently with
    /// an active `Compute`), so this invariant holds by the ORT contract.
    pub(crate) fn release_weight_cache(&mut self, subgraph_id: u64) {
        if let Some(cache) = self.weight_caches.remove(&subgraph_id) {
            let mut buffers = 0u64;
            let mut bytes = 0u64;
            for (_, buf) in cache {
                buffers += 1;
                bytes += buf.size;
                // SAFETY: buf is owned; last Compute has completed; no outstanding GPU work.
                unsafe { self.alloc.free(buf) };
            }
            // R10 wiring artifact: this call count is produced by the call graph, not by review.
            // A run whose cache is never released leaves `weight_cache_release_calls` at 0.
            crate::counters::weights::on_cache_release(buffers, bytes);
        } else {
            // Still record the invocation: the caller reached the release path even for a subgraph
            // that never populated a cache. The call-count artifact must not depend on there being
            // buffers to free, or "never wired" and "wired but empty" become indistinguishable.
            crate::counters::weights::on_cache_release(0, 0);
        }
    }

    /// Free **every** remaining weight cache, for every subgraph.
    ///
    /// This is the session-owned backstop for the cache lifetime (Defect 1). The per-subgraph
    /// release path is `SubgraphComputeInfo::Drop → release_weight_cache`, which fires when ORT
    /// calls `ReleaseNodeComputeInfos`. That path is correct, but it makes the cache's lifetime
    /// owned by ORT's teardown order rather than by the session that allocated the buffers. If a
    /// compute-info is ever dropped out of order — or not at all before the session tears down —
    /// the device buffers would leak until the `gpu-allocator` Drop reclaimed the whole heap with a
    /// leak warning. Draining here makes the session the owner of last resort: whatever the
    /// per-subgraph path missed is freed before `alloc` drops.
    fn drain_weight_caches(&mut self) {
        let ids: Vec<u64> = self.weight_caches.keys().copied().collect();
        for id in ids {
            self.release_weight_cache(id);
        }
    }

    // ── Compute path ────────────────────────────────────────────────────────

    /// Execute a compiled subgraph using ORT kernel context tensor buffers.
    ///
    /// Inputs are staged CPU → GPU, shaders are dispatched, and outputs are copied back to
    /// ORT-allocated CPU memory. All GPU resources are allocated and freed per-call.
    ///
    /// **Memory topology note:** On UMA devices (Intel Iris Xe, mobile) device-local and
    /// host-visible heaps coincide, so staging "copies" remain within the same physical heap
    /// — they are still correct and will be optimised to direct mapped writes in M1+.
    ///
    /// **Per-dispatch facts for Niobe's harness:**
    /// - `uma`: `self.capable.caps.is_uma`
    /// - `ts_period`: `self.capable.caps.timestamp_period_ns` (raw ticks only in v0)
    /// - Tile config and shared-memory bytes: embedded in `kernel.spec_constants` and
    ///   `kernel.workgroups`; see `CompiledKernel` for the encoding.
    ///
    /// # Safety
    /// - `api` and `kernel_ctx` must be live ORT pointers for the duration of this call.
    /// - `kernels`, `input_byte_sizes`, `output_byte_sizes`, `output_shapes` must be exactly
    ///   what `compile_impl` computed for this subgraph. For static-shape subgraphs these values
    ///   are the full byte sizes / shapes baked at Compile time; for dynamic subgraphs the byte
    ///   sizes are 0 for inputs/outputs whose extents were symbolic at Compile time, and the
    ///   pre-pass below replaces them with the real values read from the ORT kernel context.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn dispatch_ort(
        &mut self,
        kernels: &[CompiledKernel],
        input_byte_sizes: &[u64],
        input_is_constant: &[bool],
        output_byte_sizes: &[u64],
        output_shapes: &[Vec<i64>],
        n_intermediates: usize,
        name_map: Option<Arc<HashMap<String, u64>>>,
        first_temp_token: usize,
        static_intermediate_byte_sizes: &[u64],
        subgraph_id: u64,
        api: *const ort::OrtApi,
        kernel_ctx: *mut ort::OrtKernelContext,
    ) -> *mut ort::OrtStatus {
        // ── Observability ──────────────────────────────────────────────────────
        // Grab the tracer once (one atomic load) and open the subgraph-level span.
        // All phase guards below are children of this span on the Chrome Trace timeline.
        // When tracing and verbose are both off, every entry point below is a no-op atomic load
        // and an early return — there is no allocation and no clock read.
        let t = trace::tracer();
        let _sg = t.subgraph_region(kernels.len());

        let n_plan_inputs = input_byte_sizes.len();
        let n_plan_outputs = output_byte_sizes.len();
        // `input_is_constant` is built from the same `plan.inputs` as `input_byte_sizes` and is
        // index-parallel to it. A short slice would silently disable the device-buffer cache
        // (the `unwrap_or(false)` below), which is safe but expensive, so make the disagreement
        // loud in debug builds rather than paying for it quietly.
        debug_assert_eq!(
            input_is_constant.len(),
            n_plan_inputs,
            "input_is_constant must be index-parallel to input_byte_sizes"
        );

        // Per-subgraph weight cache: obtain a raw pointer to break the borrow conflict
        // between `self.weight_caches` and `self.alloc` / `self.device` used later.
        // SAFETY: `weight_caches` and the other fields (`alloc`, `device`, etc.) are
        // disjoint struct fields; no two accesses below alias the same memory.
        let weight_cache_ptr: *mut HashMap<(usize, u64), GpuBuffer> =
            self.weight_caches.entry(subgraph_id).or_default() as *mut _;
        // ── Step 1: read ORT input tensor data pointers ──────────────────────
        // Also retain the OrtValue pointers for the dynamic pre-pass below.
        let mut input_cpu_ptrs: Vec<*const u8> = Vec::with_capacity(input_byte_sizes.len());
        let mut ort_values: Vec<*const ort::OrtValue> = Vec::with_capacity(input_byte_sizes.len());
        for i in 0..input_byte_sizes.len() {
            let mut ort_value: *const ort::OrtValue = std::ptr::null();
            // SAFETY: `api` is a live `OrtApi` for the duration of this call (fn contract). The
            // table is process-wide and immutable, so reading a member is a plain field read.
            let st = match unsafe { (*api).KernelContext_GetInput } {
                // SAFETY: `f` came from the live api table; `kernel_ctx` is live for this call
                // (fn contract); `i` is in range because `check_bound_counts` in `ep.rs` verified
                // the context's input count equals `input_byte_sizes.len()` before we were called;
                // `&mut ort_value` is a valid out-pointer to a live local.
                Some(f) => unsafe { f(kernel_ctx, i, &mut ort_value) },
                None => {
                    // SAFETY: `api` is a live `OrtApi` for this call; the message is a 'static
                    // NUL-terminated literal; no buffers have been allocated yet so nothing
                    // outlives the return.
                    return unsafe {
                        crate::sys::make_status(
                            api,
                            ort::OrtErrorCode_ORT_EP_FAIL,
                            "OrtApi::KernelContext_GetInput unavailable",
                        )
                    };
                }
            };
            if !st.is_null() {
                return st;
            }
            let mut data: *mut std::ffi::c_void = std::ptr::null_mut();
            // SAFETY: as above — `api` is live and the table read is a plain field read.
            let st = match unsafe { (*api).GetTensorMutableData } {
                // SAFETY: `f` came from the live api table. `ort_value` was just filled in by a
                // successful `KernelContext_GetInput`, so it is a live tensor owned by ORT for
                // the duration of this Compute call. ORT exposes only the mutable accessor, so
                // the `*const -> *mut` cast is required by the signature; we only read through
                // the pointer it returns.
                Some(f) => unsafe { f(ort_value as *mut ort::OrtValue, &mut data) },
                None => {
                    // SAFETY: `api` is a live `OrtApi` for this call; the message is a 'static
                    // NUL-terminated literal; no GPU buffers have been allocated yet.
                    return unsafe {
                        crate::sys::make_status(
                            api,
                            ort::OrtErrorCode_ORT_EP_FAIL,
                            "OrtApi::GetTensorMutableData unavailable",
                        )
                    };
                }
            };
            if !st.is_null() {
                return st;
            }
            input_cpu_ptrs.push(data as *const u8);
            ort_values.push(ort_value);
        }

        // ── Step 1.5: resolve actual byte sizes for dynamic inputs ────────────
        // `compile_impl` stores 0 for inputs/outputs whose shapes were symbolic at Compile time.
        // Replace those 0s with the sizes reported by the live tensors.
        let mut actual_input_byte_sizes: Vec<u64> = input_byte_sizes.to_vec();
        for (i, sz) in actual_input_byte_sizes.iter_mut().enumerate() {
            if *sz == 0 {
                // SAFETY: `api` and `ort_values[i]` are live for this call.
                if let Some(get_sz) = unsafe { (*api).GetTensorSizeInBytes } {
                    let mut n: usize = 0;
                    // SAFETY: `ort_values[i]` is a live tensor; `&mut n` is a valid out-pointer.
                    let st = unsafe { get_sz(ort_values[i], &mut n) };
                    if st.is_null() {
                        *sz = n as u64;
                    } else {
                        // SAFETY: `api` is live; `st` is non-null.
                        unsafe { crate::sys::release_status(api, st) };
                    }
                }
            }
        }

        // ── Step 1.6: dynamic kernel pre-pass ────────────────────────────────
        // For each kernel that has a `dyn_recipe`, re-run its translate handler with the concrete
        // ORT shapes to derive push_constants, workgroups, spec_constants, shader, output byte
        // sizes, and — critically — the correct binding token sequence.
        //
        // The binding sequence from `push_dynamic_kernel` (used at Compile time) is positional:
        // one token per input slot, one per output slot. Some translate handlers produce extra or
        // duplicate tokens (e.g. MatMulNBits without zero_points binds `scales` twice as an inert
        // placeholder for the declared-but-absent zero-points descriptor).  The ShapeOnlyRecorder
        // captures the *actual* token sequence from the translate, which is used at dispatch time
        // instead of `kernel.bindings` so the pipeline layout and descriptor writes are correct.
        //
        // Type alias for captured per-kernel dynamic dispatch data:
        //   (push_constants, workgroups, spec_constants, shader, bindings)
        //
        // The bindings vector is captured from the translate handler's KernelRequest and may differ
        // in length from kernel.bindings (which was built by push_dynamic_kernel from NodeDesc
        // input/output counts). Translate handlers can pass fewer or more bindings than there are
        // NodeDesc inputs+outputs — most notably matmul_nbits_gemv, which binds `scales` twice,
        // once as its natural slot and once as the inert zero_point placeholder. The pipeline and
        // descriptor set must be sized from this captured length, not from kernel.bindings.len(),
        // or the output binding slot falls outside the descriptor set and writes nowhere.
        type DynCaptured = (Vec<u8>, [u32; 3], Vec<u32>, &'static str, Vec<u64>);
        let mut dyn_captured: Vec<Option<DynCaptured>> = (0..kernels.len()).map(|_| None).collect();
        // Per-kernel temp buffer sizes from the ShapeOnlyRecorder (dynamic kernels only).
        let mut dyn_temp_sizes: Vec<Vec<u64>> = vec![Vec::new(); kernels.len()];
        let mut actual_output_byte_sizes: Vec<u64> = output_byte_sizes.to_vec();
        let mut actual_output_shapes: Vec<Vec<i64>> = output_shapes.to_vec();

        // For multi-node islands: intermediate buffer byte sizes, indexed by intermediate index
        // (= token - n_plan_inputs - n_plan_outputs). Starts from compile-time static sizes;
        // dynamic kernels override with Compute-time sizes from ShapeOnlyRecorder.
        let mut intermediate_byte_sizes: Vec<u64> = static_intermediate_byte_sizes.to_vec();
        if intermediate_byte_sizes.len() < n_intermediates {
            intermediate_byte_sizes.resize(n_intermediates, 0);
        }

        // TensorDesc cache for intermediate outputs produced by earlier kernels in the same island.
        // Keys are tokens in [n_plan_inputs+n_plan_outputs, first_temp_token).
        let mut computed_descs: HashMap<u64, TensorDesc> = HashMap::new();
        // Aliased output → input map: output_index → input_index.
        // Populated during the ShapeOnlyRecorder pre-pass from `bind_aliased_output` calls.
        // Used in the output allocation loop to borrow an input buffer instead of allocating
        // a new device buffer for in-place KV cache outputs.
        let mut aliased_output_to_input: HashMap<usize, usize> = HashMap::new();

        for (ki, kernel) in kernels.iter().enumerate() {
            let recipe = match &kernel.dyn_recipe {
                Some(r) => r,
                None => continue,
            };

            let n_inputs = recipe.node_desc.inputs.len();

            // Build a patched NodeDesc with concrete (non-symbolic) TensorDescs.
            // For external ORT inputs (token < n_plan_inputs): read from `ort_values`.
            // For intermediate inputs (token >= n_plan_inputs + n_plan_outputs): look up
            // in `computed_descs`, which is populated by prior kernels in the same island.
            let mut patched_inputs = recipe.node_desc.inputs.clone();
            for (slot, &binding_token) in kernel.bindings[..n_inputs].iter().enumerate() {
                if binding_token == NO_TOKEN {
                    // Absent optional input — no plan slot, nothing to patch.
                    continue;
                }
                if binding_token < n_plan_inputs as u64 {
                    // External ORT input.
                    let global_idx = binding_token as usize;
                    if global_idx < ort_values.len() {
                        // SAFETY: `api` and `ort_values[global_idx]` are live for this call.
                        let td = unsafe { read_tensor_desc_from_ort(api, ort_values[global_idx]) };
                        if let Some(td) = td {
                            patched_inputs[slot].desc = Some(td);
                        }
                    }
                } else if let Some(td) = computed_descs.get(&binding_token) {
                    // Produced by an earlier kernel in this island. Two token ranges reach here
                    // and both are ordinary:
                    //
                    // * `>= n_plan_inputs + n_plan_outputs` — a pure intermediate, consumed only
                    //   inside the island.
                    // * `[n_plan_inputs, n_plan_inputs + n_plan_outputs)` — an island *output*
                    //   that is also consumed internally. This branch previously did not exist;
                    //   the range was left symbolic under a comment calling it "theoretically
                    //   possible but unusual" and promising the translate handler would "degrade
                    //   gracefully". It is island #15's normal shape — every residual stream in
                    //   Phi-3.5 has it — and the handler does not degrade, it refuses, which is
                    //   the correct response to a `None` it cannot interpret. The result was 323
                    //   claimed nodes that executed zero times.
                    //
                    // The distinction the two ranges once carried was never about where the desc
                    // comes from; it is the same producer either way. Keying off "did a prior
                    // kernel produce this token" instead of off the token range is what removes
                    // the case that had no branch.
                    patched_inputs[slot].desc = Some(td.clone());
                }
                // A token that reaches neither arm has no producer at all. It stays `None` and
                // the translate handler refuses, which is correct: nothing here infers a desc
                // from a sibling slot. The information is restored at the point it still exists
                // (the producer, below), not reconstructed at the point it was lost.
                //
                // MEASURED 2026-08-02 (Mouse), which is how the missing case was located:
                // `/model/layers.0/input_layernorm/LayerNorm` slot 0, token 5, with
                // n_plan_inputs=5 and n_plan_outputs=2 — the `embed_tokens/Gather` result is an
                // island output AND an internal edge, and the handler refused with
                //   Unsupported("`SimplifiedLayerNormalization` input 0 has no element type at
                //   compile time")
                // so the island was dropped and the model fell back to the CPU EP.
            }

            let patched_node = NodeDesc {
                op_type: recipe.node_desc.op_type.clone(),
                domain: recipe.node_desc.domain.clone(),
                since_version: recipe.node_desc.since_version,
                name: recipe.node_desc.name.clone(),
                inputs: patched_inputs,
                outputs: recipe.node_desc.outputs.clone(),
                attributes: recipe.node_desc.attributes.clone(),
            };

            // Re-run translate through ShapeOnlyRecorder to capture push constants, workgroups,
            // and the full binding list (which may include duplicate slots not in kernel.bindings).
            // For multi-node islands use new_named so bind_output assigns the island-wide token.
            let mut sor = match name_map.as_ref() {
                Some(nm) => ShapeOnlyRecorder::new_named(
                    kernel.n_plan_inputs,
                    Arc::clone(nm),
                    first_temp_token,
                ),
                None => ShapeOnlyRecorder::new(kernel.n_plan_inputs),
            };
            // The `Err` is bound rather than discarded by `is_ok()`. R13: a broken commitment
            // whose message is "translate failed" tells a reader that something failed, which
            // they already knew from the WARN. The handler's own text is the only thing here
            // that says *which* precondition the run-time shapes violated.
            let dyn_translate = (recipe.spec.translate)(recipe.spec, &patched_node, &mut sor);
            if dyn_translate.is_ok() {
                // Update actual output byte sizes and shapes, and record intermediate descs.
                for (token, desc) in &sor.output_desc_by_token {
                    let token = *token;
                    if token >= n_plan_inputs as u64
                        && token < (n_plan_inputs + n_plan_outputs) as u64
                    {
                        // External ORT output.
                        let j = (token - n_plan_inputs as u64) as usize;
                        if let Some(sz) = desc.byte_size() {
                            actual_output_byte_sizes[j] = sz as u64;
                            actual_output_shapes[j] = desc.shape.clone();
                        }
                        // An island output can also be consumed by a later kernel *inside* this
                        // island — a residual stream is the ordinary case, not an exotic one:
                        // `SkipSimplifiedLayerNormalization` emits the skip sum as an island
                        // output and the next block reads it. Record the desc for that reader.
                        //
                        // This is the same `desc` the two lines above trust for allocation, so
                        // the reader is given exactly what the allocator was given. Nothing is
                        // inferred here: a consumer that cannot find a producer still gets
                        // `None` and still refuses, which is what keeps a genuinely absent
                        // producer distinguishable from a present one.
                        computed_descs.insert(token, desc.clone());
                    } else if token >= (n_plan_inputs + n_plan_outputs) as u64
                        && token < first_temp_token as u64
                    {
                        // Intermediate output — propagate to later kernels in this island.
                        computed_descs.insert(token, desc.clone());
                        // Also record the byte size for intermediate buffer allocation.
                        let idx = (token - (n_plan_inputs + n_plan_outputs) as u64) as usize;
                        if idx < intermediate_byte_sizes.len() {
                            if let Some(sz) = desc.byte_size() {
                                intermediate_byte_sizes[idx] = sz as u64;
                            }
                        }
                    }
                }
                if let (Some(cap), Some(cap_bi)) = (sor.captured, sor.captured_bindings) {
                    dyn_captured[ki] = Some((cap.0, cap.1, cap.2, cap.3, cap_bi));
                }
                dyn_temp_sizes[ki] = sor
                    .temp_descs
                    .iter()
                    .map(|d| d.byte_size().unwrap_or(0) as u64)
                    .collect();
                // Collect aliased output→input pairs for the output allocation loop.
                for (out_tok, in_tok) in sor.aliased_pairs {
                    if out_tok >= n_plan_inputs as u64
                        && out_tok < (n_plan_inputs + n_plan_outputs) as u64
                        && in_tok < n_plan_inputs as u64
                    {
                        let j = (out_tok - n_plan_inputs as u64) as usize;
                        aliased_output_to_input.insert(j, in_tok as usize);
                    }
                }
            } else {
                log::error!(
                    "dispatch_ort: dynamic re-run of translate for op '{}' failed: {:?}",
                    recipe.node_desc.op_type,
                    dyn_translate.as_ref().err()
                );
                // SAFETY: `api` is a live `OrtApi` for this call; the message is a 'static
                // NUL-terminated literal. No GPU buffers have been allocated yet at this point
                // in the pre-pass, so nothing outlives the return.
                return unsafe {
                    crate::sys::make_status(
                        api,
                        ort::OrtErrorCode_ORT_EP_FAIL,
                        "dynamic translate re-run failed — shapes may have changed incompatibly",
                    )
                };
            }
        }

        // ── Step 1a: bind the EP's own device buffers where ORT placed an input in them ──
        //
        // §6.5's payoff. When `alloc_device_frame` is `SHARED`, an input ORT placed in this EP's
        // device memory already lives in a `VkBuffer` on the device we are about to dispatch on.
        // Binding it directly skips a fresh `DeviceLocal` allocation and a full re-upload per
        // Compute call. `bind_target_for` declines — returning `None` — for every case it cannot
        // prove: host memory, a `SPLIT-DEVICE` frame, or an interior offset the descriptor cannot
        // express. Declining costs one upload; assuming would read a neighbouring tensor.
        //
        // Must run BEFORE Step 1b, which overwrites `input_cpu_ptrs[i]` with the *staging* address
        // and would leave nothing to classify as a handle.
        let mut bound_inputs: Vec<Option<(vk::Buffer, u64)>> = vec![None; input_cpu_ptrs.len()];
        for (i, p) in input_cpu_ptrs.iter().enumerate() {
            let want = actual_input_byte_sizes.get(i).copied().unwrap_or(0) as usize;
            bound_inputs[i] =
                crate::vk::host_device_memory::bind_target_for(p.cast_mut().cast::<u8>(), want);
        }

        // ── Step 1b: resolve any input that lives in the EP's own device memory ──
        //
        // Cross-owner note (Tank): when `ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY` is on, ORT may place
        // a subgraph input in memory *this EP* allocated, and the pointer above is then an opaque
        // handle — a reserved, inaccessible address, by design, so that mistaking it for memory
        // faults instead of reading someone else's tensor. `host_backing_for` returns `None` for
        // ordinary host memory (the default build, unchanged) and the readable bytes for a handle.
        //
        // Use `actual_input_byte_sizes` (not `input_byte_sizes`) here. For dynamic-shape inputs
        // `input_byte_sizes[i] == 0` (the compile-time signal); `actual_input_byte_sizes[i]` was
        // resolved in Step 1.5 from the live tensor. Passing 0 to `host_bytes` would skip the
        // overflow check — `len > available` is trivially false for len=0 — allowing an upload
        // of `actual_input_byte_sizes[i]` bytes from a span that may be too short.
        for (i, p) in input_cpu_ptrs.iter_mut().enumerate() {
            let want = actual_input_byte_sizes.get(i).copied().unwrap_or(0) as usize;
            match crate::transfer::host_backing_for(p.cast_mut().cast::<u8>(), want) {
                None => {}
                Some(Ok(backing)) => *p = backing as *const u8,
                Some(Err(why)) => {
                    let msg = format!(
                        "VulkanExecutionProvider: input {i} lives in device memory and its bytes \
                         are unreachable: {why}"
                    );
                    // SAFETY: `api` is a live `OrtApi` for the whole call (fn contract).
                    return unsafe {
                        crate::sys::make_status(api, ort::OrtErrorCode_ORT_EP_FAIL, &msg)
                    };
                }
            }
        }

        // ── Step 2 & 3: allocate all GPU buffers ─────────────────────────────
        // We allocate everything upfront so cleanup is a single loop on error.
        let mut gpu_inputs: Vec<GpuBuffer> = Vec::new();
        let mut staging_ups: Vec<GpuBuffer> = Vec::new();
        let mut gpu_outputs: Vec<GpuBuffer> = Vec::new();
        let mut staging_dls: Vec<GpuBuffer> = Vec::new();
        // Intermediate buffers: one per inter-node edge in multi-node islands.
        let mut gpu_intermediates: Vec<GpuBuffer> = Vec::new();
        let mut gpu_temps: Vec<GpuBuffer> = Vec::new();

        macro_rules! bail {
            ($msg:literal) => {{
                self.free_all(
                    &mut gpu_inputs,
                    &mut staging_ups,
                    &mut gpu_outputs,
                    &mut staging_dls,
                    &mut gpu_intermediates,
                    &mut gpu_temps,
                );
                // SAFETY: `api` is a live `OrtApi` for the whole call (fn contract) and `$msg` is
                // a 'static NUL-terminated literal. `free_all` above has already released every
                // buffer, so nothing owned by this frame outlives the return.
                return unsafe {
                    crate::sys::make_status(api, ort::OrtErrorCode_ORT_EP_FAIL, $msg)
                };
            }};
        }

        for (i, &sz) in actual_input_byte_sizes.iter().enumerate() {
            // Zero-element tensor (e.g., Phi-3.5 KV-cache on first-token prefill: shape
            // [1, H, 0, D]).  A zero-byte VkBuffer is invalid; Vulkan also requires
            // VkDescriptorBufferInfo::range > 0.  Bind the session placeholder (4 bytes,
            // DeviceLocal) so the descriptor write satisfies the API contract.  The shader
            // will not access this buffer — the outer KV dimension is 0, so no dispatch
            // thread ever indexes into it.  No upload is needed.
            if sz == 0 {
                let placeholder_buf = self
                    .zero_elem_placeholder
                    .as_ref()
                    .expect("zero_elem_placeholder was freed before dispatch — session bug");
                gpu_inputs.push(GpuBuffer::borrowed_ref(
                    placeholder_buf.buffer,
                    4,
                    MemClass::DeviceLocal,
                ));
                // Borrowed sentinel — upload loop skips borrowed staging buffers, barrier
                // filter skips them too (read-after-read on placeholder is fine).
                staging_ups.push(GpuBuffer::borrowed_ref(
                    vk::Buffer::null(),
                    0,
                    MemClass::DeviceLocal,
                ));
                continue;
            }
            // Bound to the EP's own device buffer in Step 1a: no allocation, no upload. The bytes
            // are already on this device — `CopyTensors` mirrors every write into a handle, and
            // `write_outputs_to_ort` mirrors the one writer that does not go through it — so
            // there is nothing to stage. `borrowed_ref` keeps `free_all` from freeing memory ORT
            // owns.
            if let Some((buf, size)) = bound_inputs[i] {
                gpu_inputs.push(GpuBuffer::borrowed_ref(buf, size, MemClass::DeviceLocal));
                staging_ups.push(GpuBuffer::borrowed_ref(
                    vk::Buffer::null(),
                    0,
                    MemClass::DeviceLocal,
                ));
                continue;
            }
            // Device-buffer cache check. The key is `(cpu_ptr, byte_size)`, but the *predicate*
            // for being cacheable at all is `input_is_constant[i]` — ORT's own answer to whether
            // this subgraph input is a graph initializer.
            //
            // MEASURED 2026-08-02: this was gated on `byte_size >= 32 KiB` instead, on the
            // reasoning that "activations are small (seq=1 in Phi-3.5 → 6 KB)". KV-cache inputs
            // are neither small nor constant: `past_key_values.N.key` is `past_len * 6144` bytes
            // and crosses 32 KiB at past_len >= 6. Every one of the 64 KV inputs was cached as
            // though it were a weight, and every inference after the first read the FIRST
            // inference's cache. `probe_kv_input_cache.py` reads STALE_CACHE: with the past
            // mutated in place between two runs, the second run returned the first run's answer
            // to the bit while the reference answer moved by 0.0157.
            //
            // An address identifies storage; it does not identify contents. Those are the same
            // question for an initializer and different questions for anything ORT may rewrite,
            // and no size threshold separates them — Phi-3.5's KV inputs are on the same side of
            // every threshold as its weights. The information that does separate them exists in
            // the caller (`GraphViewer::input_is_constant`), so the fix is to carry it here
            // rather than to infer it from something that correlates with it.
            //
            // The size floor is kept as a secondary condition only because a tiny constant is not
            // worth a cache slot; it is no longer load-bearing for correctness.
            let cacheable = input_is_constant.get(i).copied().unwrap_or(false);
            let cpu_key = (input_cpu_ptrs[i] as usize, sz);
            let cached_hit = if cacheable {
                // SAFETY: `weight_cache_ptr` is a valid, exclusive pointer to the per-subgraph
                // HashMap entry; no other code accesses `weight_caches` during this call.
                unsafe { (*weight_cache_ptr).get(&cpu_key) }
            } else {
                None
            };
            if let Some(cached) = cached_hit {
                gpu_inputs.push(GpuBuffer::borrowed_ref(
                    cached.buffer,
                    cached.size,
                    cached.mem_class,
                ));
                // Push a sentinel staging entry so the Vecs stay parallel.  The sentinel is also
                // borrowed_ref (no staging needed for cached inputs) and will be a no-op in
                // free_all.  We never write to it or submit a vkCmdCopyBuffer for it.
                staging_ups.push(GpuBuffer::borrowed_ref(
                    vk::Buffer::null(),
                    0,
                    MemClass::DeviceLocal,
                ));
                continue;
            }
            // SAFETY: `self.alloc` owns a live `VkDevice` for as long as the session exists;
            // sz > 0 is guaranteed by the zero-size guard above.
            let Some(buf) = (unsafe { self.alloc.alloc_device(&format!("ep_in_{i}"), sz) }) else {
                bail!("alloc_device failed for input buffer");
            };
            // SAFETY: same allocator, same live device, same non-zero size.
            let Some(stg) = (unsafe { self.alloc.alloc_upload(&format!("ep_stg_in_{i}"), sz) })
            else {
                // SAFETY: `buf` was returned by this allocator moments ago, has not been freed,
                // and has never been submitted, so no GPU work can be referencing it.
                unsafe { self.alloc.free(buf) };
                bail!("alloc_upload failed for input staging");
            };
            gpu_inputs.push(buf);
            staging_ups.push(stg);
        }

        for (i, &sz) in actual_output_byte_sizes.iter().enumerate() {
            // Zero-element output (e.g., GQA KV-cache on first-token prefill).
            // Same constraint as inputs: Vulkan requires a non-null buffer handle and
            // range > 0.  There is nothing to write back to ORT (0 bytes); bind the
            // session placeholder so the descriptor write is valid.
            if sz == 0 {
                let placeholder_buf = self
                    .zero_elem_placeholder
                    .as_ref()
                    .expect("zero_elem_placeholder was freed before dispatch — session bug");
                gpu_outputs.push(GpuBuffer::borrowed_ref(
                    placeholder_buf.buffer,
                    4,
                    MemClass::DeviceLocal,
                ));
                staging_dls.push(GpuBuffer::borrowed_ref(
                    vk::Buffer::null(),
                    0,
                    MemClass::Download,
                ));
                continue;
            }
            // Aliased output (in-place KV cache update): the shader writes to the input buffer
            // directly.  Borrow the input's GPU buffer for the output slot — no new device
            // allocation — and allocate only a real download staging buffer so the barrier and
            // vkCmdCopyBuffer path can flush the written data back to ORT.
            if let Some(&in_idx) = aliased_output_to_input.get(&i) {
                let in_buf = &gpu_inputs[in_idx];
                gpu_outputs.push(GpuBuffer::borrowed_ref(in_buf.buffer, sz, in_buf.mem_class));
                // SAFETY: same allocator/device; sz > 0 verified above.
                let Some(stg) =
                    (unsafe { self.alloc.alloc_download(&format!("ep_stg_out_{i}"), sz) })
                else {
                    bail!("alloc_download failed for aliased output staging");
                };
                staging_dls.push(stg);
                continue;
            }
            // Output buffers: STORAGE_BUFFER (shader writes) + TRANSFER_SRC (copy to staging).
            // SAFETY: live allocator/device as above; `sz` is the (actual) byte size, non-zero.
            let Some(buf) = (unsafe {
                self.alloc.alloc(
                    &format!("ep_out_{i}"),
                    sz,
                    MemClass::DeviceLocal,
                    vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
                )
            }) else {
                bail!("alloc failed for output buffer");
            };
            // SAFETY: same allocator, same live device, same non-zero size.
            let Some(stg) = (unsafe { self.alloc.alloc_download(&format!("ep_stg_out_{i}"), sz) })
            else {
                // SAFETY: `buf` was just allocated here, is unfreed and unsubmitted.
                unsafe { self.alloc.free(buf) };
                bail!("alloc_download failed for output staging");
            };
            gpu_outputs.push(buf);
            staging_dls.push(stg);
        }

        // Allocate intermediate buffers for multi-node islands.
        // These are STORAGE_BUFFER only — a prior kernel writes them and a later kernel reads them;
        // they never need to be transferred to/from staging.
        for (idx, &sz) in intermediate_byte_sizes.iter().enumerate() {
            if sz == 0 {
                // Zero-size intermediate: the shape was still symbolic at both Compile and Compute
                // time. This is a translate-handler defect; bail with a diagnostic.
                log::error!(
                    "dispatch_ort: intermediate buffer {idx} has zero byte size — \
                     the translate handler did not resolve the output shape at Compute time"
                );
                bail!("intermediate buffer has zero byte size — unresolved shape at Compute time");
            }
            // SAFETY: live allocator/device; sz > 0 verified above.
            let Some(buf) = (unsafe {
                self.alloc.alloc(
                    &format!("ep_inter_{idx}"),
                    sz,
                    MemClass::DeviceLocal,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                )
            }) else {
                bail!("alloc failed for intermediate buffer");
            };
            gpu_intermediates.push(buf);
        }

        // Allocate scratch buffers for kernels that called alloc_temp at translate time.
        // Dynamic kernels use dyn_temp_sizes (from the ShapeOnlyRecorder pre-pass); static kernels
        // use temp_byte_sizes baked into the CompiledKernel at Compile time.
        // Temps are appended to gpu_temps in kernel order; buf_bindings routes them by offset.
        let mut temp_starts: Vec<usize> = Vec::with_capacity(kernels.len());
        for (ki, kernel) in kernels.iter().enumerate() {
            temp_starts.push(gpu_temps.len());
            let temp_sizes: &[u64] = if dyn_captured[ki].is_some() || kernel.dyn_recipe.is_some() {
                &dyn_temp_sizes[ki]
            } else {
                &kernel.temp_byte_sizes
            };
            for (j, &sz) in temp_sizes.iter().enumerate() {
                // SAFETY: `self.alloc` is live; size comes from compiled shader metadata.
                let Some(buf) =
                    (unsafe { self.alloc.alloc_device(&format!("ep_tmp_k{ki}_{j}"), sz) })
                else {
                    bail!("alloc_device failed for temp buffer");
                };
                gpu_temps.push(buf);
            }
        }

        // ── Step 4: record command buffer ─────────────────────────────────────
        // Phase::Record wraps everything from vkBeginCommandBuffer through vkEndCommandBuffer.
        // The upload CPU-memcopy time is reported separately via record_transfer so Niobe's
        // harness can attribute it without mixing recording and copy costs.
        let _record_guard = t.phase(Phase::Record);

        // SAFETY: cmd_pool is live; no previous recording is in flight.
        let Some(recorder) = (unsafe { self.cmd_pool.begin() }) else {
            bail!("vkBeginCommandBuffer failed");
        };
        let cmd = recorder.cmd;

        // Create GPU timestamp query pool and reset all queries before any barrier or dispatch.
        // `timestamp_valid_bits == 0` means the queue does not support timestamps; skip entirely.
        let query_pool: Option<GpuQueryPool> =
            if t.wants_gpu_timestamps() && self.capable.caps.timestamp_valid_bits > 0 {
                // SAFETY: device is live; n_kernels is the dispatch count for this call.
                let qp = unsafe { GpuQueryPool::new(self.device.ash(), kernels.len()) };
                if let Some(ref qp) = qp {
                    // vkCmdResetQueryPool must precede all vkCmdWriteTimestamp calls in this CB.
                    // SAFETY: cmd is recording; qp was just created.
                    unsafe { qp.cmd_reset(cmd) };
                }
                qp
            } else {
                None
            };

        // Write CPU data into staging buffers and record staging→device copies.
        // Time the CPU memcopy portion so record_transfer can report upload bandwidth.
        // Sub-phase CmdUpload isolates this cost from the rest of Record so we can tell
        // whether upload memcpy or something else is the dominant 97% unexplained fraction.
        let _cmd_upload_guard = t.phase(Phase::CmdUpload);
        let upload_t0 = std::time::Instant::now();
        let mut uploaded_bytes: u64 = 0;
        for (i, (stg, &cpu_ptr)) in staging_ups.iter().zip(input_cpu_ptrs.iter()).enumerate() {
            // Skip cache hits — sentinel staging buffers are `borrowed` (size 0, null handle).
            if stg.borrowed {
                continue;
            }
            let sz = actual_input_byte_sizes[i] as usize;
            // SAFETY: `cpu_ptr` is the tensor data pointer ORT just gave us for input `i`; it is
            // valid for at least `actual_input_byte_sizes[i]` bytes and stays live for this call.
            let data = unsafe { std::slice::from_raw_parts(cpu_ptr, sz) };
            // SAFETY: cmd is recording; stg is Upload, gpu_inputs[i] is DeviceLocal.
            unsafe { record_upload(self.device.ash(), cmd, stg, &gpu_inputs[i], data) };
            uploaded_bytes += sz as u64;
        }
        // Record upload bytes + duration in the tracer summary.
        // record_transfer (not phase(Phase::Upload)) so the byte/bandwidth counters are emitted
        // without double-counting the duration in phase_us[Upload].
        //
        // CROSS-OWNER EDIT (Tank, declared): the `if t.active()` wrapper was removed. Persistent
        // weight residency is to be verified on BYTES, not wall time, and this is the only site
        // that knows how many bytes were staged — so gating it on the tracer meant the default
        // run recorded nothing and `alloc_device_upload_bytes` (a different copy, through the
        // provider's own VkDevice) read 0 while this loop moved ~2 GiB per inference.
        // `record_transfer` now self-guards: it takes two atomics unconditionally into
        // `counters::staging` and early-returns from all trace work when the tracer is inert.
        // Behaviour when tracing IS on is unchanged.
        t.record_transfer(Transfer::Upload, uploaded_bytes, upload_t0.elapsed());
        drop(_cmd_upload_guard);

        // Barrier: TRANSFER_WRITE → SHADER_READ on freshly-uploaded input buffers.
        // Cached inputs (borrowed staging sentinel) are already in the correct layout from their
        // last use; read-after-read across queue submissions needs no barrier.
        let up_deps: Vec<BufferDep> = staging_ups
            .iter()
            .zip(gpu_inputs.iter())
            .filter(|(stg, _)| !stg.borrowed)
            .map(|(_, b)| BufferDep {
                buffer: b.buffer,
                offset: 0,
                size: vk::WHOLE_SIZE,
                src: Access::TransferWrite,
                dst: Access::ShaderRead,
            })
            .collect();
        // SAFETY: cmd is recording; all buffers are live.
        unsafe { self.device.barriers().buffer_deps(cmd, &up_deps) };

        // Descriptor pools for each kernel dispatch.
        //
        // **Lifetime contract:** each pool must remain alive until the fence signals — i.e.,
        // until `submit_and_wait` returns.  Dropping a pool inside the recording loop frees
        // its descriptor sets while the command buffer still references them; the driver may
        // then reuse the same VkDescriptorSet handle for the next dispatch, and calling
        // vkUpdateDescriptorSets on that new set while its handle is "bound" to the recording
        // command buffer triggers validation error VUID-vkUpdateDescriptorSets-None-03047:
        //   "A descriptor set is updated while bound to a recording command buffer,
        //    without UPDATE_AFTER_BIND."
        //
        // Fix: collect every pool here and let them drop together after `submit_and_wait`.
        // This makes the class of error impossible: the set being written is never the set
        // that is bound, because each pool is fresh and its handle is distinct from every
        // previously bound set.
        let mut desc_pools: Vec<DispatchDescriptorPool> = Vec::with_capacity(kernels.len());
        // Shader name for each kernel, captured inside the loop so the GPU timestamp report can
        // label each interval. Populated in lock-step with desc_pools.
        let mut shader_names: Vec<&str> = Vec::with_capacity(kernels.len());

        // For each kernel: build pipeline + descriptor set, bind and dispatch.
        for (ki, kernel) in kernels.iter().enumerate() {
            // For dynamic kernels, use the pre-pass capture; for static, use baked values.
            // `eff_bindings` is the authoritative descriptor-slot list: for dynamic kernels it
            // comes from the translate handler's KernelRequest (which may include duplicate slots
            // such as the zero_point placeholder in matmul_nbits_gemv); for static kernels it is
            // `kernel.bindings` (recorded directly from the translate handler at Compile time, so
            // it is already correct). Both forms are in the same u64 token encoding.
            let (eff_shader, eff_spec_constants, eff_push_constants, eff_workgroups, eff_bindings): (
                &str,
                &[u32],
                &[u8],
                [u32; 3],
                &[u64],
            ) = match dyn_captured[ki].as_ref() {
                Some((pc, wg, sc, sh, bi)) => (sh, sc.as_slice(), pc.as_slice(), *wg, bi.as_slice()),
                None => (
                    kernel.shader,
                    kernel.spec_constants.as_slice(),
                    kernel.push_constants.as_slice(),
                    kernel.workgroups,
                    kernel.bindings.as_slice(),
                ),
            };

            let Some(spirv) = crate::engine::shaders::find(eff_shader) else {
                // No SPIR-V for this shader — shouldn't happen if GetCapability checked has_any().
                // SAFETY: `recorder` owns the command buffer begun above and has not
                // been submitted, so ending it cannot race any GPU work. We discard the
                // result because we are already on the error path.
                let _ = unsafe { recorder.finish() };
                self.free_all(
                    &mut gpu_inputs,
                    &mut staging_ups,
                    &mut gpu_outputs,
                    &mut staging_dls,
                    &mut gpu_intermediates,
                    &mut gpu_temps,
                );
                // SAFETY: `api` is a live `OrtApi` for the whole call (fn contract) and the
                // message is a 'static NUL-terminated literal. Every buffer allocated by
                // this frame has been released above, so nothing outlives the return.
                return unsafe {
                    crate::sys::make_status(
                        api,
                        ort::OrtErrorCode_ORT_EP_FAIL,
                        "SPIR-V not found for shader stem — was the EP built with shaders?",
                    )
                };
            };

            let n_bindings = eff_bindings.len();
            let pkey = PipelineKey {
                shader: eff_shader,
                spec_constants: eff_spec_constants.to_vec(),
            };
            // Sub-phase: pipeline cache lookup (hashmap hit) or vkCreateComputePipelines (first
            // encounter). Drops before the descriptor-alloc sub-phase so spans don't overlap.
            let _pipeline_lookup_guard = t.phase(Phase::PipelineLookup);
            // SAFETY: spirv is valid SPIR-V from build.rs; pipeline_cache and device are live.
            let Some(entry) = (unsafe {
                self.pipeline_cache
                    .get_or_create(pkey, spirv, n_bindings as u32)
            }) else {
                // SAFETY: `recorder` owns the command buffer begun above and has not
                // been submitted, so ending it cannot race any GPU work. We discard the
                // result because we are already on the error path.
                let _ = unsafe { recorder.finish() };
                self.free_all(
                    &mut gpu_inputs,
                    &mut staging_ups,
                    &mut gpu_outputs,
                    &mut staging_dls,
                    &mut gpu_intermediates,
                    &mut gpu_temps,
                );
                // SAFETY: `api` is a live `OrtApi` for the whole call (fn contract) and the
                // message is a 'static NUL-terminated literal. Every buffer allocated by
                // this frame has been released above, so nothing outlives the return.
                return unsafe {
                    crate::sys::make_status(
                        api,
                        ort::OrtErrorCode_ORT_EP_FAIL,
                        "vkCreateComputePipelines failed",
                    )
                };
            };
            drop(_pipeline_lookup_guard);

            // Sub-phase: descriptor pool creation + set allocation + descriptor writes.
            // This is the Vulkan-side cost of binding buffers to the pipeline for one dispatch.
            let _desc_alloc_guard = t.phase(Phase::DescAlloc);
            // Per-dispatch descriptor pool.  Ownership is transferred into `desc_pools` at the
            // bottom of this loop iteration and freed after `submit_and_wait` returns.
            // SAFETY: device is live; n_bindings is the correct storage-buffer count.
            let Some(desc_pool) =
                (unsafe { DispatchDescriptorPool::new(self.device.ash(), n_bindings as u32) })
            else {
                // SAFETY: `recorder` owns the command buffer begun above and has not
                // been submitted, so ending it cannot race any GPU work. We discard the
                // result because we are already on the error path.
                let _ = unsafe { recorder.finish() };
                self.free_all(
                    &mut gpu_inputs,
                    &mut staging_ups,
                    &mut gpu_outputs,
                    &mut staging_dls,
                    &mut gpu_intermediates,
                    &mut gpu_temps,
                );
                // SAFETY: `api` is a live `OrtApi` for the whole call (fn contract) and the
                // message is a 'static NUL-terminated literal. Every buffer allocated by
                // this frame has been released above, so nothing outlives the return.
                return unsafe {
                    crate::sys::make_status(
                        api,
                        ort::OrtErrorCode_ORT_EP_FAIL,
                        "DispatchDescriptorPool::new failed",
                    )
                };
            };

            // Resolve binding indices to (VkBuffer, size) pairs.
            // `eff_bindings` comes from the actual translate run (via ShapeOnlyRecorder for dynamic
            // kernels, or from `kernel.bindings` for static ones). Using it — rather than
            // `kernel.bindings` — ensures that any extra or duplicate bindings the translate
            // inserts (e.g. the scales-as-zero-points placeholder in `MatMulNBits`) are correctly
            // mapped to GPU buffers and the pipeline layout has the right number of descriptors.
            //
            // Token routing for multi-node islands:
            //   token < n_plan_inputs                                     → gpu_inputs[token]
            //   n_plan_inputs <= token < n_plan_inputs + n_plan_outputs   → gpu_outputs[j]
            //   n_plan_inputs + n_plan_outputs <= token < first_temp_token → gpu_intermediates[k]
            //   token >= first_temp_token                                  → gpu_temps[...]
            let buf_bindings: Vec<(vk::Buffer, u64)> = eff_bindings
                .iter()
                .map(|&token| {
                    if token < n_plan_inputs as u64 {
                        let b = &gpu_inputs[token as usize];
                        (b.buffer, b.size)
                    } else {
                        let j = (token - n_plan_inputs as u64) as usize;
                        let b = if j < n_plan_outputs {
                            &gpu_outputs[j]
                        } else {
                            let k = j - n_plan_outputs;
                            if k < n_intermediates {
                                &gpu_intermediates[k]
                            } else {
                                // Temp buffer.
                                let temp_idx = k - n_intermediates;
                                &gpu_temps[temp_starts[ki] + temp_idx]
                            }
                        };
                        (b.buffer, b.size)
                    }
                })
                .collect();

            // SAFETY: desc_pool is live; entry.descriptor_set_layout has exactly n_bindings
            // STORAGE_BUFFER slots.
            let Some(desc_set) = (unsafe {
                desc_pool.allocate_and_write(entry.descriptor_set_layout, &buf_bindings)
            }) else {
                // SAFETY: `recorder` owns the command buffer begun above and has not
                // been submitted, so ending it cannot race any GPU work. We discard the
                // result because we are already on the error path.
                let _ = unsafe { recorder.finish() };
                self.free_all(
                    &mut gpu_inputs,
                    &mut staging_ups,
                    &mut gpu_outputs,
                    &mut staging_dls,
                    &mut gpu_intermediates,
                    &mut gpu_temps,
                );
                // SAFETY: `api` is a live `OrtApi` for the whole call (fn contract) and the
                // message is a 'static NUL-terminated literal. Every buffer allocated by
                // this frame has been released above, so nothing outlives the return.
                return unsafe {
                    crate::sys::make_status(
                        api,
                        ort::OrtErrorCode_ORT_EP_FAIL,
                        "vkAllocateDescriptorSets failed",
                    )
                };
            };
            drop(_desc_alloc_guard);

            // Record: bind pipeline, descriptors, push constants, and dispatch.
            // SAFETY: cmd is recording; all handles are live for the duration.
            unsafe {
                self.device.ash().cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    entry.pipeline,
                );
                self.device.ash().cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    entry.pipeline_layout,
                    0,
                    &[desc_set],
                    &[],
                );
                if !eff_push_constants.is_empty() {
                    self.device.ash().cmd_push_constants(
                        cmd,
                        entry.pipeline_layout,
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        eff_push_constants,
                    );
                }
                // GPU timestamp BEFORE this dispatch — placed after push_constants so
                // the timestamp fires at COMPUTE_SHADER stage, after all prior state is set.
                if let Some(ref qp) = query_pool {
                    // SAFETY: cmd is recording; ki < n_kernels; cmd_reset was called above.
                    qp.cmd_before(cmd, ki);
                }
                let [wg_x, wg_y, wg_z] = eff_workgroups;
                if std::env::var_os("ONNXRUNTIME_EP_VULKAN_DUMP_OUTPUT_BYTES").is_some() {
                    // Decode push constants as u32 words for diagnostic.
                    let pc_words: Vec<u32> = eff_push_constants
                        .chunks_exact(4)
                        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                        .collect();
                    log::debug!(
                        "dispatch kernel[{ki}] shader={eff_shader} \
                         workgroups=[{wg_x},{wg_y},{wg_z}] push_u32={pc_words:?}",
                    );
                }
                self.device.ash().cmd_dispatch(cmd, wg_x, wg_y, wg_z);
                // GPU timestamp AFTER this dispatch.
                if let Some(ref qp) = query_pool {
                    // SAFETY: cmd is recording; ki < n_kernels.
                    qp.cmd_after(cmd, ki);
                }
            }
            // Keep this pool alive until after the fence signals.  See the `desc_pools`
            // declaration above for the full lifetime reasoning.
            shader_names.push(eff_shader);
            // §8.9.11 (Mouse, declared): the run's own record of which embedded SPIR-V module it
            // bound, so a proof-ledger entry can name the code it proved and go stale when that
            // code is replaced.
            crate::counters::record_shader_dispatched(eff_shader);
            desc_pools.push(desc_pool);

            // For multi-node islands: emit a SHADER_WRITE → SHADER_READ barrier after each
            // dispatch (except the last), so a later kernel in the same island sees this one's
            // writes.
            //
            // This is **one global memory barrier**, not one per intermediate buffer. Measured on
            // phi-3.5: the island carries 355 kernels and 417 intermediate buffers, so the
            // per-buffer form emitted 147,618 `VkBufferMemoryBarrier`s per inference — each one
            // constructed, heap-allocated into a fresh `Vec`, and walked by the driver, on the
            // host, while the GPU sat idle. That accounted for essentially all of the unnamed
            // time inside `vulkan.record`, which was costing more host time than the entire GPU
            // execution it was describing.
            //
            // The global barrier is strictly more conservative — it makes every shader write
            // visible to every shader read, a superset of the 417 named buffers — so it cannot
            // permit an overlap the per-buffer form forbade and cannot introduce a race. And it
            // gives up no real parallelism here: every kernel in the island reads what an earlier
            // one wrote, so the dependency set was effectively total already.
            if !gpu_intermediates.is_empty() && ki + 1 < kernels.len() {
                // SAFETY: cmd is recording.
                unsafe {
                    self.device
                        .barriers()
                        .memory_dep(cmd, Access::ShaderWrite, Access::ShaderRead)
                };
            }
        }

        // Barrier: SHADER_WRITE → TRANSFER_READ on all non-zero output buffers.
        // Zero-size outputs use the session placeholder (borrowed_ref) — they were never written
        // by any shader, so no barrier is needed (and we don't want to issue a SHADER_WRITE
        // barrier on the shared placeholder).
        let dl_deps: Vec<BufferDep> = gpu_outputs
            .iter()
            .zip(staging_dls.iter())
            .filter(|(_, stg)| !stg.borrowed)
            .map(|(b, _)| BufferDep {
                buffer: b.buffer,
                offset: 0,
                size: vk::WHOLE_SIZE,
                src: Access::ShaderWrite,
                dst: Access::TransferRead,
            })
            .collect();
        // SAFETY: cmd is recording; all output buffers are live.
        unsafe { self.device.barriers().buffer_deps(cmd, &dl_deps) };

        // Record output→staging downloads.  Skip zero-size outputs (borrowed staging sentinels)
        // — vkCmdCopyBuffer with size=0 is invalid, and there are no bytes to copy.
        for (i, (gpu_out, stg)) in gpu_outputs.iter().zip(staging_dls.iter()).enumerate() {
            if stg.borrowed {
                continue; // zero-size output or cached — nothing to download
            }
            // SAFETY: cmd is recording; gpu_out is DeviceLocal, stg is Download.
            unsafe {
                record_download(
                    self.device.ash(),
                    cmd,
                    gpu_out,
                    stg,
                    actual_output_byte_sizes[i],
                )
            };
        }

        // ── Submit and wait ────────────────────────────────────────────────────
        // SAFETY: `recorder` owns the command buffer begun above; recording is complete and it
        // has not been submitted, so ending it cannot race any GPU work.
        let Some(cmd_buf) = (unsafe { recorder.finish() }) else {
            self.free_all(
                &mut gpu_inputs,
                &mut staging_ups,
                &mut gpu_outputs,
                &mut staging_dls,
                &mut gpu_intermediates,
                &mut gpu_temps,
            );
            // SAFETY: `api` is a live `OrtApi` for the whole call (fn contract) and the
            // message is a 'static NUL-terminated literal. Every buffer allocated by
            // this frame has been released above, so nothing outlives the return.
            return unsafe {
                crate::sys::make_status(
                    api,
                    ort::OrtErrorCode_ORT_EP_FAIL,
                    "vkEndCommandBuffer failed",
                )
            };
        };
        // Phase::Record ends when the recording guard is dropped (before we submit).
        drop(_record_guard);

        // Bracket the GPU execution in host monotonic time for the calibration anchor.
        // host_t0 = just before queue_submit; host_t1 = just after wait_for_fences.
        // The GPU kernel(s) execute somewhere in [host_t0, host_t1]; the midpoint is the
        // anchor, and half the bracket width is the reported uncertainty.
        let host_t0 = onnx_runtime_tracer::absolute_now_us();

        // Phase::Submit — wraps only vkQueueSubmit. Measures driver bookkeeping; measures NO
        // GPU work (the call returns before any shader runs).
        let fence = {
            let _submit_guard = t.phase(Phase::Submit);
            // SAFETY: cmd_buf is in executable state; queue is idle; device is live.
            let fence_opt = unsafe {
                create_and_submit(self.device.ash(), self.device.compute_queue(), cmd_buf)
            };
            // _submit_guard drops here, ending the Submit span.
            fence_opt
        };
        let Some(fence) = fence else {
            self.free_all(
                &mut gpu_inputs,
                &mut staging_ups,
                &mut gpu_outputs,
                &mut staging_dls,
                &mut gpu_intermediates,
                &mut gpu_temps,
            );
            // SAFETY: `api` is a valid ORT API pointer for this EP invocation.
            return unsafe {
                crate::sys::make_status(api, ort::OrtErrorCode_ORT_EP_FAIL, "vkQueueSubmit failed")
            };
        };

        // Phase::FenceWait — queue latency + GPU execution + any concurrently-scheduled work.
        // This is an UPPER BOUND on kernel time, not kernel time. Real GPU time comes from the
        // VkQueryPool path below.
        let fence_ok = {
            let _fence_guard = t.phase(Phase::FenceWait);
            // SAFETY: fence was submitted above; device is live.
            let ok = unsafe { wait_fence_then_destroy(self.device.ash(), fence) };
            // _fence_guard drops here, ending the FenceWait span.
            ok
        };
        let host_t1 = onnx_runtime_tracer::absolute_now_us();

        if !fence_ok {
            self.free_all(
                &mut gpu_inputs,
                &mut staging_ups,
                &mut gpu_outputs,
                &mut staging_dls,
                &mut gpu_intermediates,
                &mut gpu_temps,
            );
            // SAFETY: `api` is a live `OrtApi` for the whole call (fn contract) and the
            // message is a 'static NUL-terminated literal. Every buffer allocated by
            // this frame has been released above, so nothing outlives the return.
            return unsafe {
                crate::sys::make_status(
                    api,
                    ort::OrtErrorCode_ORT_EP_FAIL,
                    "vkWaitForFences failed",
                )
            };
        }

        // ── GPU timestamp report ───────────────────────────────────────────────
        // Read timestamp query results and emit per-kernel GPU spans to the tracer.
        // The fence has signalled, so vkGetQueryPoolResults with WAIT_BIT is guaranteed to
        // return immediately.
        //
        // Calibration: bracketing fallback (VK_EXT_calibrated_timestamps is not used in v0).
        // The anchor places the first dispatch's begin-tick at the midpoint of the host bracket;
        // anchor_uncertainty_us = half the bracket width tells viewers how imprecise that is.
        //
        // Key invariant: the conversion reads timestamp_period_ns (52.0833 on Intel Iris Xe,
        // not 1.0) and applies the 36-valid-bit mask. Both come from caps.rs and are
        // cross-checked by bench/timestamp_audit.py against vulkaninfoSDK. If this conversion
        // is wrong, the audit exits non-zero.
        if let Some(ref qp) = query_pool {
            // SAFETY: fence has signalled; command buffer execution is complete.
            let results = unsafe { qp.read_results() };
            let device_anchor_ticks = results
                .iter()
                .flatten()
                .next()
                .map(|&(b, _)| b)
                .unwrap_or(0);
            let cal = GpuTimestampCalibration {
                timestamp_period_ns: self.capable.caps.timestamp_period_ns,
                valid_bits: self.capable.caps.timestamp_valid_bits,
                host_anchor_us: (host_t0 + host_t1) / 2,
                device_anchor_ticks,
                anchor_uncertainty_us: host_t1.saturating_sub(host_t0) / 2,
            };
            let intervals: Vec<GpuInterval> = shader_names
                .iter()
                .zip(results.iter())
                .enumerate()
                .filter_map(|(ki, (name, r))| {
                    let &(begin, end) = r.as_ref()?;
                    Some(GpuInterval {
                        label: name.to_string(),
                        begin_ticks: begin,
                        end_ticks: end,
                        node_index: Some(ki as u64),
                        flops: None, // TODO: from op spec (Mouse owns flop estimates)
                        bytes: None, // TODO: from compiled kernel metadata
                    })
                })
                .collect();
            if !intervals.is_empty() {
                t.record_gpu_intervals(&GpuTimestampReport {
                    calibration: cal,
                    queue_family: self.device.compute_queue_family(),
                    intervals,
                });
            }
        }

        // ── Step 5: write outputs back to ORT-allocated CPU memory ────────────
        // Readback: time the CPU memcopy from mapped staging_dls to ORT's output tensors.
        let readback_t0 = std::time::Instant::now();
        // SAFETY: `api` and `kernel_ctx` are live for this call (fn contract); the fence above has
        // been waited on, so every `staging_dls` buffer is mapped and its download has completed;
        // `output_byte_sizes` and `output_shapes` are the compile-time values for this subgraph.
        let status = unsafe {
            self.write_outputs_to_ort(
                api,
                kernel_ctx,
                &staging_dls,
                &actual_output_byte_sizes,
                &actual_output_shapes,
            )
        };
        // CROSS-OWNER EDIT (Tank, declared): same removal of the `if t.active()` gate as on the
        // upload path above, for the same reason — the readback byte count is the other half of
        // the per-inference boundary traffic and must exist without a tracing flag.
        let readback_bytes: u64 = actual_output_byte_sizes.iter().sum();
        t.record_transfer(Transfer::Readback, readback_bytes, readback_t0.elapsed());

        // Cleanup regardless of output-write outcome.
        //
        // Device-buffer cache insertion: for each input that was a fresh upload (non-borrowed
        // staging) *and that ORT reports as a graph initializer*, move its device buffer into
        // `weight_cache` keyed on (cpu_ptr, byte_size) so the next inference can skip the upload.
        //
        // The constancy flag is the load-bearing condition and the size floor is a convenience.
        // The prior code had it the other way round: it cached anything ≥32 KiB, reasoning that
        // "activations are small (seq=1 in Phi-3.5 → 6 KB)". That reasoning names the right
        // hazard — "caching them could serve stale data if ORT reuses the same address with
        // different bytes" — and then tests for it with a proxy that Phi-3.5's KV-cache inputs
        // fail: `past_key_values.N.key` is `past_len * 6144` bytes, i.e. ≥32 KiB from past_len 6.
        // See the lookup site for the falsifier (`probe_kv_input_cache.py` → STALE_CACHE).
        const WEIGHT_CACHE_MIN_BYTES: u64 = 32 * 1024;
        for (i, stg) in staging_ups.iter().enumerate() {
            let sz = actual_input_byte_sizes[i];
            let is_const = input_is_constant.get(i).copied().unwrap_or(false);
            if !stg.borrowed && is_const && sz >= WEIGHT_CACHE_MIN_BYTES {
                // Read fields before the mutable replace to avoid a borrow conflict.
                let (handle, cached_sz, cls) = (
                    gpu_inputs[i].buffer,
                    gpu_inputs[i].size,
                    gpu_inputs[i].mem_class,
                );
                // Take ownership of the device buffer and insert into cache.
                // Replace with a borrowed_ref so free_all treats it as a no-op.
                let owned = std::mem::replace(
                    &mut gpu_inputs[i],
                    GpuBuffer::borrowed_ref(handle, cached_sz, cls),
                );
                let key = (input_cpu_ptrs[i] as usize, sz);
                crate::counters::weights::on_cache_insert(sz);
                // If a stale entry exists for this key (e.g., pointer was reused at the same
                // size by a different tensor), free it first to avoid a GPU memory leak.
                // SAFETY: `weight_cache_ptr` is a valid, exclusive pointer to the per-subgraph
                // HashMap; no other code accesses `weight_caches` during this call.
                if let Some(old) = unsafe { (*weight_cache_ptr).insert(key, owned) } {
                    // The replaced entry leaves the cache; account for it before the device free.
                    crate::counters::weights::on_cache_evict(old.size);
                    // SAFETY: `old` is a device-local buffer owned by the cache; the fence
                    // has signalled so no GPU work references it.
                    unsafe { self.alloc.free(old) };
                }
            }
        }
        self.free_all(
            &mut gpu_inputs,
            &mut staging_ups,
            &mut gpu_outputs,
            &mut staging_dls,
            &mut gpu_intermediates,
            &mut gpu_temps,
        );
        status
    }

    /// Write each downloaded GPU output buffer back to an ORT kernel context output slot.
    ///
    /// # Safety
    /// - `api`, `kernel_ctx` must be live.
    /// - `staging_dls[i]` must be mapped (HOST_VISIBLE + HOST_COHERENT, GPU work complete).
    /// - `output_byte_sizes[i]` and `output_shapes[i]` must match the plan.
    unsafe fn write_outputs_to_ort(
        &self,
        api: *const ort::OrtApi,
        kernel_ctx: *mut ort::OrtKernelContext,
        staging_dls: &[GpuBuffer],
        output_byte_sizes: &[u64],
        output_shapes: &[Vec<i64>],
    ) -> *mut ort::OrtStatus {
        for (i, (stg, shape)) in staging_dls.iter().zip(output_shapes.iter()).enumerate() {
            // Ask ORT to allocate a CPU output buffer of the right shape.
            let mut ort_out: *mut ort::OrtValue = std::ptr::null_mut();
            // SAFETY: `api` is live per the fn contract; reading a member of the immutable,
            // process-wide api table is a plain field read.
            let st = match unsafe { (*api).KernelContext_GetOutput } {
                // SAFETY: `f` came from the live api table; `kernel_ctx` is live for this call;
                // `i` is in range because `check_bound_counts` in `ep.rs` verified the context's
                // output count equals the compiled output count; `shape` is a live local slice
                // and `&mut ort_out` a valid out-pointer.
                Some(f) => unsafe { f(kernel_ctx, i, shape.as_ptr(), shape.len(), &mut ort_out) },
                None => {
                    // SAFETY: `api` is live per the fn contract and the message is a 'static
                    // NUL-terminated literal. Buffer cleanup is the caller's — `dispatch_ort`
                    // calls `free_all` unconditionally on the way out.
                    return unsafe {
                        crate::sys::make_status(
                            api,
                            ort::OrtErrorCode_ORT_EP_FAIL,
                            "OrtApi::KernelContext_GetOutput unavailable",
                        )
                    };
                }
            };
            if !st.is_null() {
                return st;
            }

            // Get a writable pointer to ORT's output buffer.
            let mut out_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            // SAFETY: `api` is live per the fn contract; plain field read.
            let st = match unsafe { (*api).GetTensorMutableData } {
                // SAFETY: `f` came from the live api table and `ort_out` was just allocated by a
                // successful `KernelContext_GetOutput`, so it is a live tensor owned by ORT.
                Some(f) => unsafe { f(ort_out, &mut out_ptr) },
                None => {
                    // SAFETY: `api` is live per the fn contract and the message is a 'static
                    // NUL-terminated literal. Buffer cleanup is the caller's.
                    return unsafe {
                        crate::sys::make_status(
                            api,
                            ort::OrtErrorCode_ORT_EP_FAIL,
                            "OrtApi::GetTensorMutableData unavailable for output",
                        )
                    };
                }
            };
            if !st.is_null() {
                return st;
            }

            // Copy downloaded GPU output to ORT's CPU buffer.
            // Zero-element outputs have a borrowed sentinel for `stg` (no mapped memory);
            // skip the copy — there are 0 bytes to transfer.  We still called
            // KernelContext_GetOutput above so ORT's output tensor is properly allocated.
            let byte_size = output_byte_sizes[i] as usize;
            if byte_size == 0 {
                continue;
            }
            let Some(src_ptr) = stg.mapped_ptr() else {
                // SAFETY: `api` is live per the fn contract and the message is a 'static
                // NUL-terminated literal. Buffer cleanup is the caller's.
                return unsafe {
                    crate::sys::make_status(
                        api,
                        ort::OrtErrorCode_ORT_EP_FAIL,
                        "staging download buffer has no mapped pointer — allocation bug",
                    )
                };
            };
            // Cross-owner note (Tank): as for inputs — an output ORT placed in this EP's own
            // device memory is an opaque handle, not writable memory. Resolve it to its backing
            // before copying, or the write below would fault on a reserved page.
            let out_handle = out_ptr;
            match crate::transfer::host_backing_for(out_ptr.cast::<u8>(), byte_size) {
                None => {}
                Some(Ok(backing)) => out_ptr = backing.cast::<std::ffi::c_void>(),
                Some(Err(why)) => {
                    let msg = format!(
                        "VulkanExecutionProvider: output {i} lives in device memory and is not \
                         writable through this path: {why}"
                    );
                    // SAFETY: `api` is live per the fn contract. Buffer cleanup is the caller's.
                    return unsafe {
                        crate::sys::make_status(api, ort::OrtErrorCode_ORT_EP_FAIL, &msg)
                    };
                }
            }
            // Diagnostic probe: when ONNXRUNTIME_EP_VULKAN_DUMP_OUTPUT_BYTES is set, log the
            // first bytes of the staging buffer so callers can distinguish "kernel wrote zeros"
            // from "staging contents correct but copy is broken."
            if std::env::var_os("ONNXRUNTIME_EP_VULKAN_DUMP_OUTPUT_BYTES").is_some() {
                let preview_len = byte_size.min(16);
                // SAFETY: src_ptr valid for byte_size bytes (GPU work complete, HOST_COHERENT).
                let preview =
                    unsafe { std::slice::from_raw_parts(src_ptr as *const u8, preview_len) };
                let all_zero = preview.iter().all(|&b| b == 0);
                log::debug!(
                    "write_outputs_to_ort: output[{i}] byte_size={byte_size}  \
                     first_{preview_len}_bytes={:02x?}  all_zero={all_zero}",
                    preview
                );
            }

            // SAFETY: src_ptr is valid for byte_size bytes (GPU work complete, HOST_COHERENT);
            // out_ptr was allocated by ORT and is valid for byte_size bytes.
            unsafe {
                std::ptr::copy_nonoverlapping(src_ptr as *const u8, out_ptr as *mut u8, byte_size);
            }

            // Keep the device mirror in step with the write above. This is the obligation created
            // by binding device buffers for inputs: a span written here through staging and later
            // read as an input through its device buffer would be read stale, and stale-but-
            // plausible is the failure mode that survives a smoke test. `CopyTensors` already
            // mirrors every copy into a handle; this is the same duty for the one writer that
            // does not go through it. A no-op (`Ok(false)`) for ordinary host outputs.
            if let Err(why) = crate::transfer::mirror_to_device(out_handle.cast::<u8>(), byte_size)
            {
                let msg = format!(
                    "VulkanExecutionProvider: output {i} was written to host staging but its \
                     device mirror could not be updated, so a later kernel binding that span \
                     would read stale bytes: {why}"
                );
                // SAFETY: `api` is live per the fn contract. Buffer cleanup is the caller's.
                return unsafe {
                    crate::sys::make_status(api, ort::OrtErrorCode_ORT_EP_FAIL, &msg)
                };
            }
        }
        std::ptr::null_mut() // success
    }

    /// Free all GPU buffers in all five pools, draining them.
    ///
    /// Called on every error path. `GpuBuffer` has no `Drop` impl, so this must be explicit.
    fn free_all(
        &mut self,
        gpu_inputs: &mut Vec<GpuBuffer>,
        staging_ups: &mut Vec<GpuBuffer>,
        gpu_outputs: &mut Vec<GpuBuffer>,
        staging_dls: &mut Vec<GpuBuffer>,
        gpu_intermediates: &mut Vec<GpuBuffer>,
        gpu_temps: &mut Vec<GpuBuffer>,
    ) {
        // Each buffer was produced by `self.alloc` and has not been freed. Every caller reaches
        // here either before submission or after `vkWaitForFences`, so no GPU work references
        // them. `GpuBuffer` has no `Drop`, which is why the frees are explicit.
        for b in gpu_inputs.drain(..) {
            // SAFETY: as above — owned by this allocator, unfreed, not in flight.
            unsafe { self.alloc.free(b) };
        }
        for b in staging_ups.drain(..) {
            // SAFETY: as above.
            unsafe { self.alloc.free(b) };
        }
        for b in gpu_outputs.drain(..) {
            // SAFETY: as above.
            unsafe { self.alloc.free(b) };
        }
        for b in staging_dls.drain(..) {
            // SAFETY: as above.
            unsafe { self.alloc.free(b) };
        }
        for b in gpu_intermediates.drain(..) {
            // SAFETY: as above.
            unsafe { self.alloc.free(b) };
        }
        for b in gpu_temps.drain(..) {
            // SAFETY: as above.
            unsafe { self.alloc.free(b) };
        }
    }
}

impl Drop for VulkanSession {
    fn drop(&mut self) {
        // Defect 1 backstop: free any weight-cache buffers the per-subgraph release path did not
        // reclaim, before `alloc` drops. The session owns the device memory it allocated; its
        // lifetime, not ORT's teardown order, is the guarantee that the cache is released.
        self.drain_weight_caches();
        // The zero-element placeholder must be freed before `alloc` drops (field order would
        // drop `alloc` before `zero_elem_placeholder` otherwise). The explicit Drop body runs
        // before any field drop glue, so `alloc` is still live here.
        if let Some(placeholder) = self.zero_elem_placeholder.take() {
            // SAFETY: placeholder was allocated by self.alloc; no GPU work can reference it
            // at drop time (ORT serialises Compute calls and never calls Compute after EP
            // release).
            unsafe { self.alloc.free(placeholder) };
        }
        // R10 artifact timing: the observation file is otherwise flushed at data-transfer
        // teardown (transfer.rs), which ORT releases *before* the weight-cache release path
        // runs — so that snapshot records `weight_cache_release_calls = 0` and the full cache
        // still resident, hiding the release entirely. Emit one final snapshot here, after the
        // drain, so the artifact carries the post-release truth (release_calls > 0, device
        // bytes drained). Counters are cumulative atomics, so this can only add information.
        crate::counters::dump_if_requested();
    }
}
