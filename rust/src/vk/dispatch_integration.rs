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
//! When multiple devices are present the test runs on **all of them**, not just the first.
//! This exercises both a UMA (e.g. Intel Iris Xe) and a discrete (e.g. NVIDIA RTX 4060) path
//! in the same test run. See `docs/ENGINE.md §2.1` for why Intel is treated as a strictness
//! oracle rather than a second data point.
//!
//! # Validation layers
//!
//! The instance is created with `enable_validation = true` so `VK_LAYER_KHRONOS_validation`
//! is active when installed. Without a `VkDebugUtilsMessengerEXT`, the layer's default output
//! goes to stderr. Any validation error appears prefixed with `Validation Error` in the test
//! runner output.
//!
//! Programmatic error capture via `VkDebugUtilsMessengerEXT` (fail-counting) is deferred: it
//! requires extending `Instance::create` to accept additional instance extensions.
//!
//! # Device selection
//!
//! All devices that pass the §7.2 gate are exercised. The test reports each separately.
//! `ONNXRUNTIME_EP_VULKAN_DEVICE` is **not** consulted here — the intent is completeness,
//! not selection. It is consulted by the EP factory at session start.
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
    instance::{CapableDevice, Instance},
    pipeline::{DispatchDescriptorPool, PipelineCache, PipelineKey},
};
use crate::ops::common::{shape_plan::ShapePlan, templates::EW_LOCAL_SIZE};

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
// Per-device dispatch helper
// ──────────────────────────────────────────────────────────────────────────────

/// Run one `Add` f32 dispatch end-to-end on a single capable device.
///
/// Creates a logical device, allocates all buffers, records and submits one command buffer,
/// reads back the result, and verifies it against the CPU reference. All resources are freed
/// on return.
///
/// Returns `Ok(())` on success, or `Err(message)` if any step fails — allowing the caller to
/// collect per-device results rather than panicking immediately.
///
/// `spirv` must be valid SPIR-V bytes for the `ew_binary_add_f32` kernel.
///
/// # Validation
///
/// The caller creates the instance with `enable_validation = true`. Without a
/// `VkDebugUtilsMessengerEXT`, validation messages go to stderr. The `Err` path covers
/// logical Vulkan failures (vkCreateDevice failing, alloc failing, etc.); validation layer
/// messages are observable on stderr but do not automatically return `Err`.
///
/// # UMA topology
///
/// `capable.caps.is_uma` is logged. On UMA devices (Intel Iris Xe, mobile) the
/// `DEVICE_LOCAL` heap is also `HOST_VISIBLE`; the staging copies are still issued and correct
/// — they simply copy within the same physical heap rather than across heaps. A future M1+
/// optimisation may skip the copy for UMA; this path proves correctness first.
fn run_add_on_device(
    instance: &Instance,
    capable: &CapableDevice,
    spirv: &[u8],
) -> Result<(), String> {
    let dev_label = format!(
        "'{}' kind={:?} api={} uma={} subgroup_sz={} ts_period={:.4}ns ts_bits={}",
        capable.info.name,
        capable.info.kind,
        capable.info.api_version,
        capable.caps.is_uma,
        capable.caps.subgroup_size,
        capable.caps.timestamp_period_ns,
        capable.caps.timestamp_valid_bits,
    );
    eprintln!("[RUN ] run_add_on_device: {dev_label}");

    // ── Logical device ────────────────────────────────────────────────────────
    // SAFETY: instance is live; capable was produced by instance.enumerate_capable_devices().
    let device = unsafe { Device::create(instance.ash(), capable, false) }
        .ok_or_else(|| format!("vkCreateDevice failed for {}", capable.info.name))?;

    // ── Allocator ─────────────────────────────────────────────────────────────
    // SAFETY: instance and device are both live; physical_device is from the same instance.
    let mut alloc =
        unsafe { Allocator::new(instance.ash(), device.physical_device(), device.ash()) }
            .ok_or_else(|| format!("Allocator::new failed for {}", capable.info.name))?;

    // ── Command pool ──────────────────────────────────────────────────────────
    // SAFETY: device is live; compute_queue_family is valid (came from CapableDevice).
    let cmd_pool = unsafe { CommandPool::new(device.ash(), device.compute_queue_family()) }
        .ok_or_else(|| format!("CommandPool::new failed for {}", capable.info.name))?;

    // ── Pipeline cache ────────────────────────────────────────────────────────
    // SAFETY: device is live.
    let mut pipeline_cache = unsafe { PipelineCache::new(device.ash(), &[]) }
        .ok_or_else(|| format!("PipelineCache::new failed for {}", capable.info.name))?;

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
        unsafe { alloc.alloc_device("add_in0", byte_size) }.ok_or("alloc_device(in0) failed")?;
    // SAFETY: alloc is live; byte_size is non-zero.
    let buf_b =
        unsafe { alloc.alloc_device("add_in1", byte_size) }.ok_or("alloc_device(in1) failed")?;

    // Output: STORAGE_BUFFER | TRANSFER_SRC (written by the shader, then copied to the
    // download staging buffer). alloc_device adds TRANSFER_DST only; we need TRANSFER_SRC.
    // SAFETY: alloc is live; byte_size is non-zero.
    let buf_out = unsafe {
        alloc.alloc(
            "add_out",
            byte_size,
            MemClass::DeviceLocal,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
        )
    }
    .ok_or("alloc(out) failed")?;

    // ── Staging buffers ───────────────────────────────────────────────────────
    // SAFETY: alloc is live; byte_size is non-zero.
    let staging_a =
        unsafe { alloc.alloc_upload("staging_a", byte_size) }.ok_or("alloc_upload(a) failed")?;
    // SAFETY: alloc is live; byte_size is non-zero.
    let staging_b =
        unsafe { alloc.alloc_upload("staging_b", byte_size) }.ok_or("alloc_upload(b) failed")?;
    // SAFETY: alloc is live; byte_size is non-zero.
    let staging_out =
        unsafe { alloc.alloc_download("staging_out", byte_size) }.ok_or("alloc_download failed")?;

    // ── Shape plan → push constants + workgroup count ─────────────────────────
    let shape = vec![N as i64];
    let plan = ShapePlan::broadcast(&[shape.as_slice(), shape.as_slice()])
        .map_err(|e| format!("ShapePlan::broadcast failed: {e}"))?;
    let push_consts = plan.push_constants();
    let [wg_x, wg_y, wg_z] = plan.workgroups_1d(EW_LOCAL_SIZE);

    // ── Pipeline ──────────────────────────────────────────────────────────────
    let key = PipelineKey {
        shader: "ew_binary_add_f32",
        // spec_id 0: local_size_x (from EW_LOCAL_SIZE), spec_id 1: EW_IDENTICAL=1.
        spec_constants: vec![EW_LOCAL_SIZE, 1u32],
    };
    // SAFETY: spirv is valid SPIR-V bytes from build.rs; pipeline_cache and device are live.
    let entry = unsafe { pipeline_cache.get_or_create(key, spirv, 3) }
        .ok_or("vkCreateComputePipelines failed for ew_binary_add_f32")?;
    let pipeline = entry.pipeline;
    let pipeline_layout = entry.pipeline_layout;
    let dsl = entry.descriptor_set_layout;

    // ── Descriptor pool + set ─────────────────────────────────────────────────
    // SAFETY: device is live; max_bindings=3 covers in0, in1, out.
    let desc_pool = unsafe { DispatchDescriptorPool::new(device.ash(), 3) }
        .ok_or("DispatchDescriptorPool::new failed")?;
    let buf_bindings = [
        (buf_a.buffer, byte_size),
        (buf_b.buffer, byte_size),
        (buf_out.buffer, byte_size),
    ];
    // SAFETY: desc_pool is live; dsl has exactly 3 STORAGE_BUFFER bindings.
    let desc_set = unsafe { desc_pool.allocate_and_write(dsl, &buf_bindings) }
        .ok_or("vkAllocateDescriptorSets failed")?;

    // ── Record command buffer ─────────────────────────────────────────────────
    // SAFETY: no previous recording is in flight on this pool.
    let recorder = unsafe { cmd_pool.begin() }.ok_or("vkBeginCommandBuffer failed")?;
    let cmd = recorder.cmd;

    // Step 1 — Write CPU data into staging and record staging→device copies.
    // SAFETY: cmd is recording; staging_{a,b} are Upload; buf_{a,b} are DeviceLocal.
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
    // SAFETY: cmd is recording; all handles are valid and compatible.
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
    // SAFETY: cmd is in recording state; no further recording calls will follow.
    let cmd_buf = unsafe { recorder.finish() }.ok_or("vkEndCommandBuffer failed")?;

    // SAFETY: cmd_buf is in executable state; the queue is idle; device is live.
    let submitted = unsafe { submit_and_wait(device.ash(), device.compute_queue(), cmd_buf) };
    if !submitted {
        // Free all buffers before returning Err (no panic-on-drop from gpu-allocator).
        // SAFETY: all buffers were produced by alloc and have not been freed.
        unsafe {
            alloc.free(buf_a);
            alloc.free(buf_b);
            alloc.free(buf_out);
            alloc.free(staging_a);
            alloc.free(staging_b);
            alloc.free(staging_out);
        }
        return Err("vkQueueSubmit or vkWaitForFences failed".to_string());
    }

    // ── Read back and verify ──────────────────────────────────────────────────
    let result_ptr = staging_out
        .mapped_ptr()
        .ok_or("staging_out must have a HOST_VISIBLE mapped pointer")?;
    // SAFETY: GPU work completed; result_ptr is valid for byte_size bytes; HOST_COHERENT.
    let result_bytes =
        unsafe { std::slice::from_raw_parts(result_ptr as *const u8, byte_size as usize) };
    let result = bytes_as_f32(result_bytes);

    // Verify: all 1024 elements must match exactly (exact arithmetic, no rounding).
    let mut mismatch = None;
    for i in 0..N {
        if result[i] != expected[i] {
            mismatch = Some(format!(
                "Add mismatch at index {i}: got {}, expected {} (a={}, b={})",
                result[i], expected[i], input_a[i], input_b[i],
            ));
            break;
        }
    }

    // ── Explicit cleanup ──────────────────────────────────────────────────────
    // GpuBuffer has no Drop impl — must be freed via Allocator before it goes out of scope.
    // SAFETY: each buffer was produced by alloc and has not been freed previously.
    unsafe {
        alloc.free(buf_a);
        alloc.free(buf_b);
        alloc.free(buf_out);
        alloc.free(staging_a);
        alloc.free(staging_b);
        alloc.free(staging_out);
    }

    // RAII drop order (reverse declaration):
    //   desc_pool → vkDestroyDescriptorPool
    //   pipeline_cache → vkDestroyPipeline + layout + dsl + vkDestroyPipelineCache
    //   cmd_pool → vkDestroyCommandPool
    //   alloc → gpu-allocator verifies zero live allocations
    //   device → vkDestroyDevice

    match mismatch {
        Some(msg) => Err(msg),
        None => {
            eprintln!("[PASS] run_add_on_device: {N} f32 elements verified on {dev_label}");
            Ok(())
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Integration test — all capable devices
// ──────────────────────────────────────────────────────────────────────────────

/// End-to-end `Add` f32 dispatch, exercised on **every** capable device.
///
/// If more than one device passes the §7.2 gate (e.g. Intel Iris Xe + NVIDIA RTX 4060 on the
/// development machine) this test runs on both and reports per-device results. The test fails
/// if ANY device produces a wrong answer or a hard Vulkan error.
///
/// **Intel as a strictness oracle.** Intel's Vulkan driver is more spec-conformant than
/// NVIDIA's. A failure on Intel that succeeds on NVIDIA almost always means the shader or
/// barrier logic relies on undefined behaviour, not that Intel has a bug. Both must pass.
///
/// See `run_add_on_device` for the per-device mechanics.
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
    let Some(instance) = Instance::create(true) else {
        eprintln!("[SKIP] add_f32_dispatches_end_to_end: no Vulkan instance (no loader or ICD)");
        return;
    };

    // ── Guard 3: enumerate all capable devices ────────────────────────────────
    let devices = instance.enumerate_capable_devices();
    if devices.is_empty() {
        eprintln!("[SKIP] add_f32_dispatches_end_to_end: no capable Vulkan device");
        return;
    }

    eprintln!(
        "[INFO] add_f32_dispatches_end_to_end: {} capable device(s) — running on all",
        devices.len()
    );

    // ── Run on every device, collect per-device outcomes ─────────────────────
    let mut failures: Vec<String> = Vec::new();

    for capable in &devices {
        match run_add_on_device(&instance, capable, spirv) {
            Ok(()) => {}
            Err(e) => {
                eprintln!("[FAIL] device='{}': {e}", capable.info.name);
                failures.push(format!("{}: {e}", capable.info.name));
            }
        }
    }

    // Fail once at the end so all per-device output is visible before the panic.
    assert!(
        failures.is_empty(),
        "add_f32 dispatch failed on {} device(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}
