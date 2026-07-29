//! End-to-end dispatch integration test: `Add` f32 on real hardware.
//!
//! # When this test runs
//!
//! The test skips silently when any precondition is absent:
//!
//! 1. `shaders::has_any() == false` — built without `glslc` (`ALLOW_MISSING_GLSLC=1`).
//! 2. No Vulkan instance — no loader installed or `vkCreateInstance` fails.
//! 3. No capable device — all devices fail the §7.2 gate.
//!
//! CI lanes install the LunarG SDK and configure a software ICD; both Linux and Windows CI
//! lanes run this test unless neither glslc nor a device is present.
//!
//! # Validation layers
//!
//! The instance is created with `enable_validation = true` so `VK_LAYER_KHRONOS_validation`
//! is active when installed. Without a `VkDebugUtilsMessengerEXT`, the layer's default output
//! goes to stderr. Any validation error appears prefixed with `VALIDATION ERROR` in the test
//! runner output and causes the test to fail. Zero validation errors on the first clean dispatch
//! is M0 exit criterion 9 (ENGINE.md §9.0.3).
//!
//! Programmatic error capture via `VkDebugUtilsMessengerEXT` is deferred: it requires
//! `Instance::create` to accept extra instance extensions, which is a separate session's work.
//!
//! # Niobe timestamp hooks
//!
//! The dispatch path is designed so that GPU-side timestamp queries can be added without
//! restructuring the recording section. See the inline comments around `cmd_dispatch` for the
//! two injection points.

use ash::vk;

use super::{
    alloc::{Allocator, MemClass, record_download, record_upload},
    barrier::{Access, BufferDep},
    cmd::{CommandPool, submit_and_wait},
    device::Device,
    instance::Instance,
    pipeline::{DispatchDescriptorPool, PipelineCache, PipelineKey},
};
use crate::ops::common::shape_plan::ShapePlan;

// ──────────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Reinterpret a `&[f32]` as a `&[u8]` without copying.
///
/// # Safety
/// `f32` is IEEE 754 single-precision on all supported platforms. The byte representation is
/// well-defined. The returned slice is valid for exactly `std::mem::size_of_val(v)` bytes and
/// must not outlive `v`.
fn f32_as_bytes(v: &[f32]) -> &[u8] {
    // SAFETY: &[f32] has alignment 4; &[u8] requires alignment 1. Converting to a wider
    // alignment would be invalid, but narrowing is always safe. size_of_val gives exact bytes.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

/// Parse a little-endian byte sequence into a `Vec<f32>`.
fn bytes_as_f32(b: &[u8]) -> Vec<f32> {
    assert_eq!(b.len() % 4, 0, "byte slice length must be a multiple of 4");
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

// ──────────────────────────────────────────────────────────────────────────────
// Integration test
// ──────────────────────────────────────────────────────────────────────────────

/// End-to-end `Add` dispatch on real hardware with validation layers enabled.
///
/// Exercises the full pipeline:
/// 1. Instance (validation layers active) → device selection → `vkCreateDevice`.
/// 2. `gpu-allocator` backed buffers: two device-local inputs + one output, three staging
///    buffers (upload + download).
/// 3. SPIR-V pipeline creation from the embedded `ew_binary_add_f32` shader.
/// 4. Descriptor set allocation and binding.
/// 5. Single command buffer: upload copies → input barriers → dispatch → output barrier →
///    download copy.
/// 6. Single `vkQueueSubmit` + fence wait.
/// 7. Read-back and exact comparison against CPU-computed expected values.
///
/// The test uses 1 024 f32 elements (`a[i] = i as f32`, `b[i] = i * 0.5`). All values are
/// exactly representable in IEEE 754 single precision; the addition is exact (no rounding), so
/// a byte-for-byte comparison is valid.
#[test]
fn add_f32_dispatches_end_to_end() {
    // ── Guard 1: shader-less build ────────────────────────────────────────────
    if !crate::engine::shaders::has_any() {
        eprintln!(
            "[SKIP] add_f32_dispatches_end_to_end: built without shaders \
             (ALLOW_MISSING_GLSLC=1 build — no shader compiled in)"
        );
        return;
    }
    let Some(spirv) = crate::engine::shaders::find("ew_binary_add_f32") else {
        eprintln!(
            "[SKIP] add_f32_dispatches_end_to_end: ew_binary_add_f32 not compiled in \
             (check shader_variants.txt)"
        );
        return;
    };

    // ── Guard 2: Vulkan instance ──────────────────────────────────────────────
    // enable_validation=true: activates VK_LAYER_KHRONOS_validation when installed.
    // Without a VkDebugUtilsMessengerEXT, the layer's output goes to stderr.
    let Some(instance) = Instance::create(true) else {
        eprintln!("[SKIP] add_f32_dispatches_end_to_end: no Vulkan instance (no loader or ICD)");
        return;
    };

    // ── Guard 3: capable device (sorted best-first; index 0 = discrete if present) ─
    let devices = instance.enumerate_capable_devices();
    let Some(capable) = devices.into_iter().next() else {
        eprintln!("[SKIP] add_f32_dispatches_end_to_end: no capable Vulkan device");
        return;
    };

    eprintln!(
        "[RUN ] add_f32_dispatches_end_to_end: device='{}' kind={:?} api={}",
        capable.info.name, capable.info.kind, capable.info.api_version
    );

    // ── Create Vulkan objects ─────────────────────────────────────────────────

    // SAFETY: instance is live; capable was produced by instance.enumerate_capable_devices()
    // on the same instance, satisfying Device::create's safety contract.
    let device =
        unsafe { Device::create(instance.ash(), &capable, false) }.expect("vkCreateDevice failed");

    // SAFETY: instance and device are both live; physical_device is from the same instance.
    let mut alloc =
        unsafe { Allocator::new(instance.ash(), device.physical_device(), device.ash()) }
            .expect("Allocator::new failed");

    // SAFETY: device is live; compute_queue_family is valid (came from CapableDevice).
    let cmd_pool = unsafe { CommandPool::new(device.ash(), device.compute_queue_family()) }
        .expect("CommandPool::new failed");

    // SAFETY: device is live.
    let mut pipeline_cache =
        unsafe { PipelineCache::new(device.ash(), &[]) }.expect("PipelineCache::new failed");

    // ── Test data ─────────────────────────────────────────────────────────────
    const N: usize = 1024;
    let input_a: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let input_b: Vec<f32> = (0..N).map(|i| i as f32 * 0.5_f32).collect();
    let expected: Vec<f32> = input_a.iter().zip(&input_b).map(|(a, b)| a + b).collect();
    let byte_size = (N * std::mem::size_of::<f32>()) as u64;

    // ── Allocate device buffers ───────────────────────────────────────────────
    // Inputs: STORAGE_BUFFER | TRANSFER_DST (staged-in via upload, then read by the shader).
    // SAFETY: alloc is live; byte_size is non-zero.
    let buf_a =
        unsafe { alloc.alloc_device("add_in0", byte_size) }.expect("alloc_device(in0) failed");
    // SAFETY: alloc is live; byte_size is non-zero.
    let buf_b =
        unsafe { alloc.alloc_device("add_in1", byte_size) }.expect("alloc_device(in1) failed");

    // Output: STORAGE_BUFFER | TRANSFER_SRC (written by the shader, then copied to the download
    // staging buffer). alloc_device adds TRANSFER_DST only; we need TRANSFER_SRC here.
    // SAFETY: alloc is live; byte_size is non-zero.
    let buf_out = unsafe {
        alloc.alloc(
            "add_out",
            byte_size,
            MemClass::DeviceLocal,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
        )
    }
    .expect("alloc(out) failed");

    // ── Staging buffers ───────────────────────────────────────────────────────
    // SAFETY: alloc is live; byte_size is non-zero.
    let staging_a =
        unsafe { alloc.alloc_upload("staging_a", byte_size) }.expect("alloc_upload(a) failed");
    // SAFETY: alloc is live; byte_size is non-zero.
    let staging_b =
        unsafe { alloc.alloc_upload("staging_b", byte_size) }.expect("alloc_upload(b) failed");
    // SAFETY: alloc is live; byte_size is non-zero.
    let staging_out =
        unsafe { alloc.alloc_download("staging_out", byte_size) }.expect("alloc_download failed");

    // ── Shape plan → push constants + workgroup count ─────────────────────────
    let shape = vec![N as i64];
    let plan =
        ShapePlan::broadcast(&[shape.as_slice(), shape.as_slice()]).expect("broadcast failed");
    assert!(
        plan.all_identical,
        "same-shape inputs must produce all_identical == true (EW_IDENTICAL spec-const must be 1)"
    );
    let push_consts = plan.push_constants();
    let [wg_x, wg_y, wg_z] = plan.workgroups_1d(256);

    // ── Pipeline (spec: local_size_x=256, EW_IDENTICAL=1 since both shapes match) ─
    let key = PipelineKey {
        shader: "ew_binary_add_f32",
        // spec_id 0: local_size_x=256, spec_id 1: EW_IDENTICAL=1 (identical input shapes).
        spec_constants: vec![256u32, 1u32],
    };
    // SAFETY: spirv is valid SPIR-V bytes from build.rs; pipeline_cache and device are live.
    let entry = unsafe { pipeline_cache.get_or_create(key, spirv, 3) }
        .expect("vkCreateComputePipelines failed for ew_binary_add_f32");
    // Copy the raw handles (all vk::* types are Copy) to release the &mut borrow on
    // pipeline_cache via NLL — we do not need to call pipeline_cache again.
    let pipeline = entry.pipeline;
    let pipeline_layout = entry.pipeline_layout;
    let dsl = entry.descriptor_set_layout;

    // ── Descriptor pool + set ─────────────────────────────────────────────────
    // SAFETY: device is live; max_bindings=3 covers in0, in1, out.
    let desc_pool = unsafe { DispatchDescriptorPool::new(device.ash(), 3) }
        .expect("DispatchDescriptorPool::new failed");
    let buf_bindings = [
        (buf_a.buffer, byte_size),
        (buf_b.buffer, byte_size),
        (buf_out.buffer, byte_size),
    ];
    // SAFETY: desc_pool is live; dsl has exactly 3 STORAGE_BUFFER bindings.
    let desc_set = unsafe { desc_pool.allocate_and_write(dsl, &buf_bindings) }
        .expect("vkAllocateDescriptorSets failed");

    // ── Record command buffer ─────────────────────────────────────────────────
    // SAFETY: no previous recording is in flight on this pool.
    let recorder = unsafe { cmd_pool.begin() }.expect("vkBeginCommandBuffer failed");
    let cmd = recorder.cmd;

    // Step 1 — Write CPU data into staging and record staging→device copies.
    //
    // record_upload writes to staging.mapped_ptr() and emits vkCmdCopyBuffer.
    // We do not need to manually write to staging memory — the helper does it.
    //
    // SAFETY: cmd is recording; staging_{a,b} are MemClass::Upload; buf_{a,b} are DeviceLocal;
    //         byte sizes match the allocated sizes.
    unsafe {
        record_upload(
            device.ash(),
            cmd,
            &staging_a,
            &buf_a,
            f32_as_bytes(&input_a),
        );
        record_upload(
            device.ash(),
            cmd,
            &staging_b,
            &buf_b,
            f32_as_bytes(&input_b),
        );
    }

    // Step 2 — Barrier: TRANSFER_WRITE → SHADER_READ on both input buffers.
    //
    // The vkCmdCopyBuffer commands above produce TRANSFER_WRITE accesses. The dispatch below
    // reads both buffers as SHADER_READ. Without this barrier the reads are undefined.
    let upload_deps = [
        BufferDep {
            buffer: buf_a.buffer,
            offset: 0,
            size: vk::WHOLE_SIZE,
            src: Access::TransferWrite,
            dst: Access::ShaderRead,
        },
        BufferDep {
            buffer: buf_b.buffer,
            offset: 0,
            size: vk::WHOLE_SIZE,
            src: Access::TransferWrite,
            dst: Access::ShaderRead,
        },
    ];
    // SAFETY: cmd is recording; both buffers are live for the duration of the command buffer.
    unsafe { device.barriers().buffer_deps(cmd, &upload_deps) };

    // Step 3 — Dispatch.
    //
    // SAFETY: cmd is recording; all handles (pipeline, layout, descriptor set) are valid and
    //         compatible with each other (created from the same device).
    unsafe {
        device
            .ash()
            .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.ash().cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            pipeline_layout,
            0,
            &[desc_set],
            &[],
        );
        device.ash().cmd_push_constants(
            cmd,
            pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            &push_consts,
        );
        // Niobe timestamp injection point (BEFORE dispatch):
        //   device.ash().cmd_write_timestamp(cmd, stage, ts_pool, ts_before_index);
        device.ash().cmd_dispatch(cmd, wg_x, wg_y, wg_z);
        // Niobe timestamp injection point (AFTER dispatch):
        //   device.ash().cmd_write_timestamp(cmd, stage, ts_pool, ts_after_index);
    }

    // Step 4 — Barrier: SHADER_WRITE → TRANSFER_READ on the output buffer.
    //
    // The dispatch writes buf_out. The vkCmdCopyBuffer below reads it as a transfer source.
    // Without this barrier the copy would race the dispatch write.
    let output_dep = [BufferDep {
        buffer: buf_out.buffer,
        offset: 0,
        size: vk::WHOLE_SIZE,
        src: Access::ShaderWrite,
        dst: Access::TransferRead,
    }];
    // SAFETY: cmd is recording; buf_out is live.
    unsafe { device.barriers().buffer_deps(cmd, &output_dep) };

    // Step 5 — Copy result: buf_out (device-local) → staging_out (host-visible).
    // SAFETY: cmd is recording; all sizes and memory-class constraints are satisfied.
    unsafe { record_download(device.ash(), cmd, &buf_out, &staging_out, byte_size) };

    // ── Submit and wait ───────────────────────────────────────────────────────
    // SAFETY: recorder.finish() ends recording and returns the command buffer in executable
    //         state. std::mem::forget prevents the drop from running.
    let cmd_buf = unsafe { recorder.finish() }.expect("vkEndCommandBuffer failed");

    // SAFETY: cmd_buf is in executable state; the queue is idle; device is live.
    // submit_and_wait blocks until the fence signals; all GPU writes to staging_out are visible
    // to the CPU on return because staging_out uses HOST_COHERENT memory.
    let submitted = unsafe { submit_and_wait(device.ash(), device.compute_queue(), cmd_buf) };
    assert!(submitted, "vkQueueSubmit or vkWaitForFences failed");

    // ── Read back and verify ──────────────────────────────────────────────────
    // submit_and_wait returns only after the fence signals, so GPU writes to staging_out are
    // visible to the CPU at this point (staging_out uses HOST_COHERENT memory).
    let result_ptr = staging_out
        .mapped_ptr()
        .expect("staging_out must have a HOST_VISIBLE mapped pointer");
    // SAFETY: GPU work completed above; result_ptr is valid for byte_size bytes;
    //         the memory is HOST_COHERENT so no explicit cache invalidation is needed.
    let result_bytes =
        unsafe { std::slice::from_raw_parts(result_ptr as *const u8, byte_size as usize) };
    let result = bytes_as_f32(result_bytes);

    // All 1024 elements must match exactly. Both inputs use integer-representable f32 values;
    // addition of such values is exact (no rounding). Any discrepancy indicates a shader,
    // descriptor, or push-constant bug.
    assert_eq!(result.len(), N, "output element count mismatch");
    for i in 0..N {
        assert_eq!(
            result[i], expected[i],
            "Add mismatch at index {i}: got {}, expected {} (a={}, b={})",
            result[i], expected[i], input_a[i], input_b[i],
        );
    }

    eprintln!(
        "[PASS] add_f32_dispatches_end_to_end: {N} f32 elements verified on '{}'",
        capable.info.name,
    );

    // ── Explicit cleanup ──────────────────────────────────────────────────────
    //
    // GpuBuffer has no Drop impl — it must be freed through the Allocator to return the
    // sub-allocation block to gpu-allocator. Failing to do so produces a diagnostic warning
    // from gpu-allocator's own Drop.
    //
    // RAII drop order after this block (reverse declaration order):
    //   desc_pool  → vkDestroyDescriptorPool
    //   pipeline_cache → vkDestroyPipeline + layout + dsl + vkDestroyPipelineCache
    //   cmd_pool   → vkDestroyCommandPool
    //   alloc      → gpu-allocator verifies zero live allocations (all freed below)
    //   device     → vkDestroyDevice
    //   instance   → vkDestroyInstance + unloads the loader library
    //
    // All RAII destructors use their own cloned ash::Device handle so device drop order
    // (device last, before instance) is correct.
    //
    // SAFETY: each buffer was produced by `alloc` and has not been freed previously.
    unsafe {
        alloc.free(buf_a);
        alloc.free(buf_b);
        alloc.free(buf_out);
        alloc.free(staging_a);
        alloc.free(staging_b);
        alloc.free(staging_out);
    }
}
