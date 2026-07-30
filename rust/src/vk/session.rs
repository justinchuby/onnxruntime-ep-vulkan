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

use ash::vk;

use super::{
    alloc::{Allocator, GpuBuffer, MemClass, record_download, record_upload},
    barrier::{Access, BufferDep},
    cmd::{CommandPool, submit_and_wait},
    device::Device,
    instance::{CapableDevice, Instance, select_device},
    pipeline::{DispatchDescriptorPool, PipelineCache, PipelineKey},
};
use crate::{
    engine::{
        BufferView, DType, DispatchContext, EpResult, KernelRequest, NodeDesc, OutRef, TensorDesc,
        TensorRef,
    },
    ep::EpOptions,
    sys::ort,
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

/// A [`DispatchContext`] that records kernel templates without issuing Vulkan commands.
///
/// Used by `compile_impl` to run the registry's translate handlers and extract
/// [`CompiledKernel`]s. Uses *positional* counting rather than name matching so that
/// the name-equality assumption between subgraph inputs and node inputs need not hold.
pub(crate) struct CompileRecorder {
    n_plan_inputs: usize,
    next_resolve: usize,
    next_bind: usize,
    /// Temporary scratch buffer byte sizes accumulated between translate calls and flushed
    /// into `CompiledKernel::temp_byte_sizes` on each `dispatch()` call.
    pending_temp_sizes: Vec<u64>,
    pub(crate) kernels: Vec<CompiledKernel>,
}

impl CompileRecorder {
    pub(crate) fn new(n_plan_inputs: usize) -> Self {
        Self {
            n_plan_inputs,
            next_resolve: 0,
            next_bind: 0,
            pending_temp_sizes: Vec::new(),
            kernels: Vec::new(),
        }
    }

    /// Record a dynamic-shape kernel: allocate binding tokens without running the translate
    /// handler, and store the `DynKernelRecipe` for Compute-time re-run.
    ///
    /// Called from `compile_impl` when the translate handler fails due to symbolic shapes.
    /// The recorder's `next_resolve`/`next_bind` counters are advanced exactly as if the
    /// translate handler had called `resolve` and `bind_output` for each slot.
    pub(crate) fn push_dynamic_kernel(
        &mut self,
        node_desc: crate::engine::NodeDesc,
        spec: &'static crate::registry::OpSpec,
    ) {
        let n_inputs = node_desc.inputs.len();
        let n_outputs = node_desc.outputs.len();

        let mut bindings = Vec::with_capacity(n_inputs + n_outputs);
        for _ in 0..n_inputs {
            bindings.push(self.next_resolve as u64);
            self.next_resolve += 1;
        }
        for _ in 0..n_outputs {
            bindings.push((self.n_plan_inputs + self.next_bind) as u64);
            self.next_bind += 1;
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
    fn resolve(&mut self, _r: &TensorRef) -> EpResult<BufferView> {
        let idx = self.next_resolve;
        self.next_resolve += 1;
        Ok(BufferView::from_raw(idx as u64))
    }

    fn bind_output(&mut self, _o: &OutRef, _desc: TensorDesc) -> EpResult<BufferView> {
        let token = self.n_plan_inputs + self.next_bind;
        self.next_bind += 1;
        Ok(BufferView::from_raw(token as u64))
    }

    fn alloc_temp(&mut self, desc: TensorDesc) -> EpResult<BufferView> {
        let token = self.n_plan_inputs + self.next_bind;
        self.next_bind += 1;
        self.pending_temp_sizes
            .push(desc.byte_size().unwrap_or(0) as u64);
        Ok(BufferView::from_raw(token as u64))
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
struct ShapeOnlyRecorder {
    n_plan_inputs: usize,
    next_resolve: usize,
    next_bind: usize,
    /// Filled by `dispatch()` with push-constant bytes, workgroup counts, spec constants, and
    /// shader stem.  Binding tokens are in `captured_bindings`.
    #[allow(clippy::type_complexity)]
    pub captured: Option<(Vec<u8>, [u32; 3], Vec<u32>, &'static str)>,
    /// Binding tokens from the translate handler's `KernelRequest`, in descriptor-slot order.
    ///
    /// `push_dynamic_kernel` creates one token per NodeDesc input plus one per output, but some
    /// translate handlers pass a different number of slots to `dispatch` — most notably
    /// `matmul_nbits_gemv`, which binds `scales` a second time as an inert placeholder for
    /// `zero_points` when the node has no zero-point input. The descriptor set layout must have
    /// exactly this many slots or the output binding falls outside the layout and writes nowhere.
    pub captured_bindings: Option<Vec<u64>>,
    /// Output `TensorDesc`s collected from `bind_output()` calls, used to size output buffers.
    pub output_descs: Vec<TensorDesc>,
    /// Descriptors from `alloc_temp()` calls — scratch buffers not tied to ORT outputs.
    pub temp_descs: Vec<TensorDesc>,
}

impl ShapeOnlyRecorder {
    fn new(n_plan_inputs: usize) -> Self {
        Self {
            n_plan_inputs,
            next_resolve: 0,
            next_bind: 0,
            captured: None,
            captured_bindings: None,
            output_descs: Vec::new(),
            temp_descs: Vec::new(),
        }
    }
}

impl DispatchContext for ShapeOnlyRecorder {
    fn resolve(&mut self, _r: &TensorRef) -> EpResult<BufferView> {
        let idx = self.next_resolve;
        self.next_resolve += 1;
        Ok(BufferView::from_raw(idx as u64))
    }

    fn bind_output(&mut self, _o: &OutRef, desc: TensorDesc) -> EpResult<BufferView> {
        let token = self.n_plan_inputs + self.next_bind;
        self.next_bind += 1;
        self.output_descs.push(desc);
        Ok(BufferView::from_raw(token as u64))
    }

    fn alloc_temp(&mut self, desc: TensorDesc) -> EpResult<BufferView> {
        let token = self.n_plan_inputs + self.next_bind;
        self.next_bind += 1;
        self.temp_descs.push(desc);
        Ok(BufferView::from_raw(token as u64))
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
    /// Per-device capabilities and metadata. No Vulkan Drop; declared first for symmetry with
    /// create() but dropped before device handles — it holds only host data.
    pub(crate) capable: CapableDevice,
    /// Compiled pipelines. Must drop before `device` (uses device handles).
    pub(crate) pipeline_cache: PipelineCache,
    /// Command pool. Must drop before `device`.
    pub(crate) cmd_pool: CommandPool,
    /// Allocator (gpu-allocator). Must drop before `device`.
    pub(crate) alloc: Allocator,
    /// Logical device + compute queue. Must drop before `instance` (vkDestroyDevice before
    /// vkDestroyInstance is required by the Vulkan spec).
    pub(crate) device: Device,
    /// Vulkan instance. Must drop **last** — all child objects must be destroyed first.
    pub(crate) instance: Instance,
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
        let instance = Instance::create(options.enable_validation)?;
        let mut capables = instance.enumerate_capable_devices();
        if capables.is_empty() {
            log::warn!("VulkanSession::create: no devices passed the §7.2 capability gate");
            return None;
        }

        let idx = if let Some(dev_idx) = options.device_index {
            if dev_idx < capables.len() {
                dev_idx
            } else {
                log::warn!(
                    "ep.device_index={dev_idx} is out of range ({} device(s) available); \
                     using device 0",
                    capables.len()
                );
                0
            }
        } else {
            select_device(&capables).unwrap_or(0)
        };

        let capable = capables.swap_remove(idx);
        log::info!(
            "VulkanSession: selected '{}' (kind={:?} api={} uma={} subgroup_sz={} \
             ts_period={:.4}ns ts_bits={})",
            capable.info.name,
            capable.info.kind,
            capable.info.api_version,
            capable.caps.is_uma,
            capable.caps.subgroup_size,
            capable.caps.timestamp_period_ns,
            capable.caps.timestamp_valid_bits,
        );

        // SAFETY: instance is live; capable was produced by instance.enumerate_capable_devices().
        let device =
            unsafe { Device::create(instance.ash(), &capable, options.force_legacy_barriers) }?;

        // SAFETY: instance and device are live; physical_device belongs to instance.
        let alloc =
            unsafe { Allocator::new(instance.ash(), device.physical_device(), device.ash()) }?;

        // SAFETY: device is live; compute_queue_family is valid.
        let cmd_pool = unsafe { CommandPool::new(device.ash(), device.compute_queue_family()) }?;

        // SAFETY: device is live.
        let pipeline_cache = unsafe { PipelineCache::new(device.ash(), &[]) }?;

        Some(VulkanSession {
            capable,
            pipeline_cache,
            cmd_pool,
            alloc,
            device,
            instance,
        })
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
    pub(crate) unsafe fn dispatch_ort(
        &mut self,
        kernels: &[CompiledKernel],
        input_byte_sizes: &[u64],
        output_byte_sizes: &[u64],
        output_shapes: &[Vec<i64>],
        api: *const ort::OrtApi,
        kernel_ctx: *mut ort::OrtKernelContext,
    ) -> *mut ort::OrtStatus {
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

        for (ki, kernel) in kernels.iter().enumerate() {
            let recipe = match &kernel.dyn_recipe {
                Some(r) => r,
                None => continue,
            };

            let n_inputs = recipe.node_desc.inputs.len();

            // Build a patched NodeDesc with concrete (non-symbolic) TensorDescs.
            let mut patched_inputs = recipe.node_desc.inputs.clone();
            for (slot, &binding_token) in kernel.bindings[..n_inputs].iter().enumerate() {
                let global_idx = binding_token as usize; // token < n_plan_inputs → global input
                if global_idx < ort_values.len() {
                    // SAFETY: `api` and `ort_values[global_idx]` are live for this call (fn contract).
                    let td = unsafe { read_tensor_desc_from_ort(api, ort_values[global_idx]) };
                    if let Some(td) = td {
                        patched_inputs[slot].desc = Some(td);
                    }
                }
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
            let mut sor = ShapeOnlyRecorder::new(kernel.n_plan_inputs);
            if (recipe.spec.translate)(recipe.spec, &patched_node, &mut sor).is_ok() {
                // Update actual output byte sizes and shapes from the re-run.
                for (j, desc) in sor.output_descs.iter().enumerate() {
                    let out_binding_slot = n_inputs + j;
                    if out_binding_slot < kernel.bindings.len() {
                        let global_out_idx =
                            kernel.bindings[out_binding_slot] as usize - kernel.n_plan_inputs;
                        if global_out_idx < actual_output_byte_sizes.len() {
                            if let Some(sz) = desc.byte_size() {
                                actual_output_byte_sizes[global_out_idx] = sz as u64;
                                actual_output_shapes[global_out_idx] = desc.shape.clone();
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
            } else {
                log::error!(
                    "dispatch_ort: dynamic re-run of translate for op '{}' failed",
                    recipe.node_desc.op_type
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
        let mut gpu_temps: Vec<GpuBuffer> = Vec::new();

        macro_rules! bail {
            ($msg:literal) => {{
                self.free_all(
                    &mut gpu_inputs,
                    &mut staging_ups,
                    &mut gpu_outputs,
                    &mut staging_dls,
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
            // SAFETY: `self.alloc` owns a live `VkDevice` for as long as the session exists, and
            // `sz` came from `Compile`, where a zero-sized tensor was already rejected.
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
                // SAFETY: `self.alloc` is live for the whole session, and every buffer it returns
                // here is freed via the same allocator. (Tank, drive-by: restructured only to
                // satisfy `clippy::undocumented_unsafe_blocks`, which wants the comment directly
                // above the block; the code is Switch's and its behaviour is unchanged.)
                let allocated =
                    unsafe { self.alloc.alloc_device(&format!("ep_tmp_k{ki}_{j}"), sz) };
                let Some(buf) = allocated else {
                    bail!("alloc_device failed for temp buffer");
                };
                gpu_temps.push(buf);
            }
        }

        // ── Step 4: record command buffer ─────────────────────────────────────
        // SAFETY: cmd_pool is live; no previous recording is in flight.
        let Some(recorder) = (unsafe { self.cmd_pool.begin() }) else {
            bail!("vkBeginCommandBuffer failed");
        };
        let cmd = recorder.cmd;

        // Write CPU data into staging buffers and record staging→device copies.
        for (i, (stg, &cpu_ptr)) in staging_ups.iter().zip(input_cpu_ptrs.iter()).enumerate() {
            // SAFETY: `cpu_ptr` is the tensor data pointer ORT just gave us for input `i`; it is
            // valid for at least `actual_input_byte_sizes[i]` bytes and stays live for this call.
            let data =
                unsafe { std::slice::from_raw_parts(cpu_ptr, actual_input_byte_sizes[i] as usize) };
            // SAFETY: cmd is recording; stg is Upload, gpu_inputs[i] is DeviceLocal.
            unsafe { record_upload(self.device.ash(), cmd, stg, &gpu_inputs[i], data) };
        }

        // Barrier: TRANSFER_WRITE → SHADER_READ on all input buffers.
        let up_deps: Vec<BufferDep> = gpu_inputs
            .iter()
            .map(|b| BufferDep {
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
            let buf_bindings: Vec<(vk::Buffer, u64)> = eff_bindings
                .iter()
                .map(|&token| {
                    if token < kernel.n_plan_inputs as u64 {
                        let b = &gpu_inputs[token as usize];
                        (b.buffer, b.size)
                    } else {
                        let j = (token - kernel.n_plan_inputs as u64) as usize;
                        let n_ort = actual_output_byte_sizes.len();
                        let b = if j < n_ort {
                            &gpu_outputs[j]
                        } else {
                            // Temp buffer: j - n_ort indexes into this kernel's temp slice.
                            &gpu_temps[temp_starts[ki] + (j - n_ort)]
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
                // Niobe timestamp hook (BEFORE): cmd_write_timestamp(cmd, stage, ts_pool, before_idx)
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
                // Niobe timestamp hook (AFTER): cmd_write_timestamp(cmd, stage, ts_pool, after_idx)
            }
            // Keep this pool alive until after the fence signals.  See the `desc_pools`
            // declaration above for the full lifetime reasoning.
            desc_pools.push(desc_pool);
        }

        // Barrier: SHADER_WRITE → TRANSFER_READ on all output buffers.
        let dl_deps: Vec<BufferDep> = gpu_outputs
            .iter()
            .map(|b| BufferDep {
                buffer: b.buffer,
                offset: 0,
                size: vk::WHOLE_SIZE,
                src: Access::ShaderWrite,
                dst: Access::TransferRead,
            })
            .collect();
        // SAFETY: cmd is recording; all output buffers are live.
        unsafe { self.device.barriers().buffer_deps(cmd, &dl_deps) };

        // Record output→staging downloads.
        for (i, (gpu_out, stg)) in gpu_outputs.iter().zip(staging_dls.iter()).enumerate() {
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
        // SAFETY: cmd_buf is in executable state; queue is idle; device is live.
        let ok =
            unsafe { submit_and_wait(self.device.ash(), self.device.compute_queue(), cmd_buf) };
        if !ok {
            self.free_all(
                &mut gpu_inputs,
                &mut staging_ups,
                &mut gpu_outputs,
                &mut staging_dls,
                &mut gpu_temps,
            );
            // SAFETY: `api` is a live `OrtApi` for the whole call (fn contract) and the
            // message is a 'static NUL-terminated literal. Every buffer allocated by
            // this frame has been released above, so nothing outlives the return.
            return unsafe {
                crate::sys::make_status(
                    api,
                    ort::OrtErrorCode_ORT_EP_FAIL,
                    "vkQueueSubmit or vkWaitForFences failed",
                )
            };
        }

        // ── Step 5: write outputs back to ORT-allocated CPU memory ────────────
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

        // Cleanup regardless of output-write outcome.
        self.free_all(
            &mut gpu_inputs,
            &mut staging_ups,
            &mut gpu_outputs,
            &mut staging_dls,
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
            let byte_size = output_byte_sizes[i] as usize;
            // Cross-owner note (Tank): as for inputs — an output ORT placed in this EP's own
            // device memory is an opaque handle, not writable memory. Resolve it to its backing
            // before copying, or the write below would fault on a reserved page.
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
        for b in gpu_temps.drain(..) {
            // SAFETY: as above.
            unsafe { self.alloc.free(b) };
        }
    }
}
