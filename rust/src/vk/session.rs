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
    engine::{BufferView, DispatchContext, EpResult, KernelRequest, OutRef, TensorDesc, TensorRef},
    ep::EpOptions,
    sys::ort,
};

// ──────────────────────────────────────────────────────────────────────────────
// CompiledKernel
// ──────────────────────────────────────────────────────────────────────────────

/// One pre-compiled dispatch template from the Compile phase.
///
/// The `bindings` vec encodes indices into the subgraph's input/output tables rather than live
/// [`BufferView`] handles — those change per Compute call. Encoding:
/// - `token < n_plan_inputs` → GPU input buffer at index `token`
/// - `token >= n_plan_inputs` → GPU output buffer at index `token - n_plan_inputs`
///
/// For M0 (single-node subgraphs), bindings is always `[0, 1, 2]` = `[input_0, input_1, output_0]`.
/// Multi-node subgraphs with intermediate tensors are out of M0 scope.
#[derive(Debug, Clone)]
pub(crate) struct CompiledKernel {
    /// Shader stem, e.g. `"ew_binary_add_f32"`.
    pub(crate) shader: &'static str,
    /// Specialization constants baked at Compile time.
    pub(crate) spec_constants: Vec<u32>,
    /// Push-constant bytes baked from static shapes.
    pub(crate) push_constants: Vec<u8>,
    /// Workgroup counts computed at Compile time.
    pub(crate) workgroups: [u32; 3],
    /// Buffer index tokens. See struct doc for encoding.
    pub(crate) bindings: Vec<u64>,
    /// Number of subgraph-level inputs (= `plan.inputs.len()`).
    pub(crate) n_plan_inputs: usize,
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
    pub(crate) kernels: Vec<CompiledKernel>,
}

impl CompileRecorder {
    pub(crate) fn new(n_plan_inputs: usize) -> Self {
        Self {
            n_plan_inputs,
            next_resolve: 0,
            next_bind: 0,
            kernels: Vec::new(),
        }
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

    fn alloc_temp(&mut self, _desc: TensorDesc) -> EpResult<BufferView> {
        // Intermediates get tokens above input+output range. No M0 op uses alloc_temp.
        let token = self.n_plan_inputs + self.next_bind;
        self.next_bind += 1;
        Ok(BufferView::from_raw(token as u64))
    }

    fn dispatch(&mut self, k: KernelRequest) -> EpResult<()> {
        self.kernels.push(CompiledKernel {
            shader: k.shader,
            spec_constants: k.spec_constants,
            push_constants: k.push_constants,
            workgroups: k.workgroups,
            bindings: k.bindings.iter().map(|b| b.as_raw()).collect(),
            n_plan_inputs: self.n_plan_inputs,
        });
        Ok(())
    }

    fn read_const_i64(&self, _r: &TensorRef) -> Option<Vec<i64>> {
        None
    }
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
    ///   what `compile_impl` computed for this subgraph (same shapes as Compile time, which
    ///   is guaranteed for static-shape models).
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
        let mut input_cpu_ptrs: Vec<*const u8> = Vec::with_capacity(input_byte_sizes.len());
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
                    // SAFETY: `api` is live per the fn contract and the message is a 'static
                    // NUL-terminated literal; `make_status` only reads through both.
                    // SAFETY: `api` is a live `OrtApi` for the whole call (fn contract) and the
                    // message is a 'static NUL-terminated literal. Every buffer allocated by
                    // this frame has been released above, so nothing outlives the return.
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
                    // SAFETY: `api` is live per the fn contract; the message is a 'static literal.
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
        }

        // ── Step 2 & 3: allocate all GPU buffers ─────────────────────────────
        // We allocate everything upfront so cleanup is a single loop on error.
        let mut gpu_inputs: Vec<GpuBuffer> = Vec::new();
        let mut staging_ups: Vec<GpuBuffer> = Vec::new();
        let mut gpu_outputs: Vec<GpuBuffer> = Vec::new();
        let mut staging_dls: Vec<GpuBuffer> = Vec::new();

        macro_rules! bail {
            ($msg:literal) => {{
                self.free_all(
                    &mut gpu_inputs,
                    &mut staging_ups,
                    &mut gpu_outputs,
                    &mut staging_dls,
                );
                // SAFETY: `api` is a live `OrtApi` for the whole call (fn contract) and `$msg` is
                // a 'static NUL-terminated literal. `free_all` above has already released every
                // buffer, so nothing owned by this frame outlives the return.
                return unsafe {
                    crate::sys::make_status(api, ort::OrtErrorCode_ORT_EP_FAIL, $msg)
                };
            }};
        }

        for (i, &sz) in input_byte_sizes.iter().enumerate() {
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

        for (i, &sz) in output_byte_sizes.iter().enumerate() {
            // Output buffers: STORAGE_BUFFER (shader writes) + TRANSFER_SRC (copy to staging).
            // SAFETY: live allocator/device as above; `sz` is the compile-time byte size, non-zero.
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

        // ── Step 4: record command buffer ─────────────────────────────────────
        // SAFETY: cmd_pool is live; no previous recording is in flight.
        let Some(recorder) = (unsafe { self.cmd_pool.begin() }) else {
            bail!("vkBeginCommandBuffer failed");
        };
        let cmd = recorder.cmd;

        // Write CPU data into staging buffers and record staging→device copies.
        for (i, (stg, &cpu_ptr)) in staging_ups.iter().zip(input_cpu_ptrs.iter()).enumerate() {
            // SAFETY: `cpu_ptr` is the tensor data pointer ORT just gave us for input `i`; it is
            // valid for at least `input_byte_sizes[i]` bytes (that size was derived from the same
            // tensor's shape and dtype at Compile time) and stays live for this Compute call.
            let data = unsafe { std::slice::from_raw_parts(cpu_ptr, input_byte_sizes[i] as usize) };
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

        // For each kernel: build pipeline + descriptor set, bind and dispatch.
        for kernel in kernels {
            let Some(spirv) = crate::engine::shaders::find(kernel.shader) else {
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

            let n_bindings = kernel.bindings.len();
            let pkey = PipelineKey {
                shader: kernel.shader,
                spec_constants: kernel.spec_constants.clone(),
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

            // Per-dispatch descriptor pool. Freed when it goes out of scope below.
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
            let buf_bindings: Vec<(vk::Buffer, u64)> = kernel
                .bindings
                .iter()
                .map(|&token| {
                    if token < kernel.n_plan_inputs as u64 {
                        let b = &gpu_inputs[token as usize];
                        (b.buffer, b.size)
                    } else {
                        let j = (token - kernel.n_plan_inputs as u64) as usize;
                        let b = &gpu_outputs[j];
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
                if !kernel.push_constants.is_empty() {
                    self.device.ash().cmd_push_constants(
                        cmd,
                        entry.pipeline_layout,
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        &kernel.push_constants,
                    );
                }
                // Niobe timestamp hook (BEFORE): cmd_write_timestamp(cmd, stage, ts_pool, before_idx)
                let [wg_x, wg_y, wg_z] = kernel.workgroups;
                self.device.ash().cmd_dispatch(cmd, wg_x, wg_y, wg_z);
                // Niobe timestamp hook (AFTER): cmd_write_timestamp(cmd, stage, ts_pool, after_idx)
            }
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
            unsafe { record_download(self.device.ash(), cmd, gpu_out, stg, output_byte_sizes[i]) };
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
                output_byte_sizes,
                output_shapes,
            )
        };

        // Cleanup regardless of output-write outcome.
        self.free_all(
            &mut gpu_inputs,
            &mut staging_ups,
            &mut gpu_outputs,
            &mut staging_dls,
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
            // SAFETY: src_ptr is valid for byte_size bytes (GPU work complete, HOST_COHERENT);
            // out_ptr was allocated by ORT and is valid for byte_size bytes.
            unsafe {
                std::ptr::copy_nonoverlapping(src_ptr as *const u8, out_ptr as *mut u8, byte_size);
            }
        }
        std::ptr::null_mut() // success
    }

    /// Free all GPU buffers in all four pools, draining them.
    ///
    /// Called on every error path. `GpuBuffer` has no `Drop` impl, so this must be explicit.
    fn free_all(
        &mut self,
        gpu_inputs: &mut Vec<GpuBuffer>,
        staging_ups: &mut Vec<GpuBuffer>,
        gpu_outputs: &mut Vec<GpuBuffer>,
        staging_dls: &mut Vec<GpuBuffer>,
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
    }
}
