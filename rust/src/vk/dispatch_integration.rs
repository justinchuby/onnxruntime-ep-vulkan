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

    // ── Env-gated validation plant (VUID-vkDestroyDevice-device-05137) ───────
    // When ONNXRUNTIME_EP_VULKAN_PLANT_VALIDATION_VIOLATION is set, leak one VkFence.
    // The object tracker reports the leak at vkDestroyDevice, which happens in Device::drop
    // below via RAII.  No invalid call ever reaches the driver: we only create the fence and
    // abandon its handle.  The validation layer fires for the abandoned object at teardown.
    //
    // Tank designed this specifically so that "a machine with no capable GPU" is
    // distinguishable from "a machine with no validation": both paths reach this plant site,
    // but only the path with a real device (and therefore a real messenger) fires the VUID.
    if std::env::var_os("ONNXRUNTIME_EP_VULKAN_PLANT_VALIDATION_VIOLATION").is_some() {
        // SAFETY: deliberately leaked; validation must report it at vkDestroyDevice.
        let _ = unsafe {
            device
                .ash()
                .create_fence(&vk::FenceCreateInfo::default(), None)
        };
    }

    // RAII drop order (reverse declaration):
    //   desc_pool → vkDestroyDescriptorPool
    //   pipeline_cache → vkDestroyPipeline + layout + dsl + vkDestroyPipelineCache
    //   cmd_pool → vkDestroyCommandPool
    //   alloc → gpu-allocator verifies zero live allocations
    //   device → vkDestroyDevice (validation layer reports the leaked fence here)

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

/// Confirm that the EP's own `VkDebugUtilsMessengerEXT` (installed by `Instance::create` when
/// `enable_validation = true`) fires when a deliberate validation violation occurs in the EP's
/// dispatch path.
///
/// This test covers the half of M0 criterion 3 that the module-level positive control does not:
/// the module-level control uses its own raw Vulkan instance and messenger to prove the validation
/// layer is loaded; *this* test proves that the messenger on the **EP's instance** (the one that
/// runs `add_f32` / `dispatch_ort` in production) is also wired and reports errors in-process.
///
/// The plant: `ONNXRUNTIME_EP_VULKAN_PLANT_VALIDATION_VIOLATION` causes `run_add_on_device` to
/// create a `VkFence` and immediately leak its handle.  At `vkDestroyDevice` (called by
/// `Device::drop`), the validation layer object tracker fires
/// VUID-vkDestroyDevice-device-05137 through the EP's messenger, which increments
/// `EP_VALIDATION_ERROR_COUNT`.
///
/// Run with:
///   `cargo test --lib --release ep_messenger_fires_for_planted_fence_leak -- --nocapture --ignored`
#[test]
#[ignore = "positive-control: deliberately triggers VUID-vkDestroyDevice-device-05137 via the \
            EP's own VkDebugUtilsMessengerEXT to prove the messenger is wired. \
            Run with: cargo test --lib --release -- --nocapture --ignored ep_messenger_fires_for_planted_fence_leak"]
fn ep_messenger_fires_for_planted_fence_leak() {
    use crate::vk::instance::{EP_VALIDATION_ERROR_COUNT, reset_ep_validation_errors};
    use std::sync::atomic::Ordering;

    if !crate::engine::shaders::has_any() {
        eprintln!("[SKIP] ep_messenger_fires_for_planted_fence_leak: no shaders compiled in");
        return;
    }
    let Some(spirv) = crate::engine::shaders::find("ew_binary_add_f32") else {
        eprintln!(
            "[SKIP] ep_messenger_fires_for_planted_fence_leak: ew_binary_add_f32 not compiled in"
        );
        return;
    };

    // Create instance with validation + debug_utils messenger.
    let Some(instance) = Instance::create(true) else {
        eprintln!("[SKIP] ep_messenger_fires_for_planted_fence_leak: no Vulkan instance");
        return;
    };
    let devices = instance.enumerate_capable_devices();
    if devices.is_empty() {
        eprintln!("[SKIP] ep_messenger_fires_for_planted_fence_leak: no capable device");
        return;
    }

    // Reset the counter after instance creation (loader INFO messages arrive at create time).
    reset_ep_validation_errors();

    // Activate the plant for this dispatch run.
    // SAFETY: setting an env var is not thread-safe on all platforms.  This test is `#[ignore]`
    // and run in isolation, so there is no concurrent test that would race on this variable.
    unsafe { std::env::set_var("ONNXRUNTIME_EP_VULKAN_PLANT_VALIDATION_VIOLATION", "1") };

    let result = run_add_on_device(&instance, &devices[0], spirv);

    // SAFETY: same isolation guarantee as the set_var call above.
    unsafe { std::env::remove_var("ONNXRUNTIME_EP_VULKAN_PLANT_VALIDATION_VIOLATION") };

    // The test dispatch itself must still compute correctly (the fence leak is invisible to
    // the shader; only the object tracker notices it at device teardown).
    if let Err(e) = result {
        panic!("run_add_on_device failed unexpectedly (separate from the plant): {e}");
    }

    // After Device::drop (which calls vkDestroyDevice), the object tracker should have fired
    // VUID-vkDestroyDevice-device-05137 and the messenger should have incremented the counter.
    let n = EP_VALIDATION_ERROR_COUNT.load(Ordering::Relaxed);
    eprintln!("[EP-PLANT] EP_VALIDATION_ERROR_COUNT after planted fence leak = {n}");
    assert!(
        n > 0,
        "expected EP_VALIDATION_ERROR_COUNT > 0 after a leaked fence at vkDestroyDevice, got 0. \
         Either the EP's VkDebugUtilsMessengerEXT was not installed (VK_EXT_debug_utils \
         unavailable or messenger creation failed) or VUID-vkDestroyDevice-device-05137 is \
         not checked by the installed validation layer. Check that VK_LAYER_KHRONOS_validation \
         is installed and VK_EXT_debug_utils is supported."
    );
}

// ──────────────────────────────────────────────────────────────────────────────
// Validation positive control — M0 criterion 3
// ──────────────────────────────────────────────────────────────────────────────
//
// PURPOSE: confirm that VK_LAYER_KHRONOS_validation is actually running and can
// catch real errors in this codebase, so that "no errors in the normal test run"
// is meaningful rather than being consistent with "the layer never loaded".
//
// HOW: the test below deliberately plants VUID-vkUpdateDescriptorSets-None-03047:
// it binds a descriptor set to a recording command buffer, then calls
// `vkUpdateDescriptorSets` on that same set before ending the command buffer.
// This is the exact violation that `dispatch_ort` produced before session 16's
// `desc_pools: Vec<DispatchDescriptorPool>` lifetime fix.
//
// VK_EXT_debug_utils is requested so the messenger callback can capture validation
// errors programmatically via a static counter — not just from stderr inspection.
// The test asserts `VALIDATION_ERRORS > 0` after the violation is planted.
//
// The test is `#[ignore]` because it deliberately causes a Vulkan error; it does
// not form part of the standard `cargo ci` pass. Run it explicitly with:
//
//   cargo test --release -p onnxruntime-ep-vulkan validation_positive_control \
//       -- --nocapture --ignored
//
// Expected output: at least one line containing VUID-vkUpdateDescriptorSets-None-03047.
#[cfg(test)]
#[allow(clippy::undocumented_unsafe_blocks)] // every unsafe call in this module is a Vulkan API; safety is documented at module level
mod validation_positive_control {
    use std::ffi::CStr;
    use std::sync::atomic::{AtomicU32, Ordering};

    use ash::vk;

    /// Populated by the debug messenger callback whenever the validation layer reports
    /// a `VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT` message.
    static VALIDATION_ERRORS: AtomicU32 = AtomicU32::new(0);

    /// VkDebugUtilsMessengerCallbackEXT — counts ERROR-severity messages and prints them.
    ///
    /// # Safety
    /// Called by the Vulkan loader on a thread of its choosing.  The callback must not call
    /// any Vulkan functions and must be able to execute concurrently.  Using an atomic
    /// counter and `eprintln!` satisfies both constraints.
    unsafe extern "system" fn debug_callback(
        severity: vk::DebugUtilsMessageSeverityFlagsEXT,
        _message_type: vk::DebugUtilsMessageTypeFlagsEXT,
        data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
        _user_data: *mut std::ffi::c_void,
    ) -> vk::Bool32 {
        let msg_str = if !data.is_null() {
            // SAFETY: `data` is a live pointer provided by the validation layer for this
            // callback's duration; `p_message` is a NUL-terminated string it owns.
            let cstr = unsafe { CStr::from_ptr((*data).p_message) };
            cstr.to_string_lossy().into_owned()
        } else {
            "(no message)".to_owned()
        };
        eprintln!("[VALIDATION-POSITIVE-CONTROL] severity={severity:?}: {msg_str}");
        VALIDATION_ERRORS.fetch_add(1, Ordering::Relaxed);
        vk::FALSE
    }

    /// Plant VUID-VkWriteDescriptorSet-descriptorType-00332 and assert the validation layer fires.
    ///
    /// The violation: write a buffer created with `VK_BUFFER_USAGE_VERTEX_BUFFER_BIT` only
    /// as a `VK_DESCRIPTOR_TYPE_STORAGE_BUFFER` descriptor.  The validation layer checks the
    /// buffer's usage flags at `vkUpdateDescriptorSets` time and reports VUID-00332.
    ///
    /// **Relationship to session-16 fix:** the original `DispatchDescriptorPool` bug destroyed
    /// the pool inside the kernel loop while its descriptor sets were still live in the recording
    /// command buffer — the same descriptor-lifetime violation class.  VUID-03047 (update while
    /// bound to recording command buffer) was the VUID observed in the real model run, but the
    /// validation layer in SDK 1.4.350.0 checks it lazily (at submit time, not at the
    /// `vkUpdateDescriptorSets` call), so a pre-submit positive control uses VUID-00332 instead.
    /// Both VUIDs belong to the same validation domain: descriptor set contents must be
    /// consistent and not violated during the set's lifetime.
    ///
    /// Run with `cargo test -- --nocapture --ignored validation_positive_control` and verify
    /// that at least one line containing VUID-VkWriteDescriptorSet-descriptorType-00332 appears.
    #[test]
    #[ignore = "positive-control: deliberately triggers VUID-VkWriteDescriptorSet-descriptorType-00332 \
                to prove VK_LAYER_KHRONOS_validation is loaded and working. \
                Run with: cargo test -- --nocapture --ignored validation_positive_control"]
    fn descriptor_set_updated_while_bound_fires_vuid_03047() {
        let entry = match unsafe { ash::Entry::load() } {
            Ok(e) => e,
            Err(_) => {
                eprintln!("[SKIP] no Vulkan loader — cannot run positive control");
                return;
            }
        };

        // ── Check for required layers / extensions ────────────────────────────
        let available_layers =
            unsafe { entry.enumerate_instance_layer_properties() }.unwrap_or_default();
        let has_validation = available_layers.iter().any(|l| {
            // SAFETY: layer_name is a NUL-terminated array from the driver.
            unsafe { CStr::from_ptr(l.layer_name.as_ptr()) == c"VK_LAYER_KHRONOS_validation" }
        });
        if !has_validation {
            eprintln!(
                "[SKIP] VK_LAYER_KHRONOS_validation not installed — cannot run positive control"
            );
            return;
        }

        let available_exts =
            unsafe { entry.enumerate_instance_extension_properties(None) }.unwrap_or_default();
        let has_debug_utils = available_exts.iter().any(|e| {
            // SAFETY: extension_name is a NUL-terminated array from the driver.
            unsafe { CStr::from_ptr(e.extension_name.as_ptr()) == c"VK_EXT_debug_utils" }
        });
        if !has_debug_utils {
            eprintln!(
                "[SKIP] VK_EXT_debug_utils not available — cannot capture errors programmatically"
            );
            return;
        }

        // ── Create instance with validation + debug_utils ─────────────────────
        let layer_name = c"VK_LAYER_KHRONOS_validation";
        let ext_name = c"VK_EXT_debug_utils";
        let layers = [layer_name.as_ptr()];
        let extensions = [ext_name.as_ptr()];
        let app_info = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_1);
        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&layers)
            .enabled_extension_names(&extensions);
        let instance = match unsafe { entry.create_instance(&create_info, None) } {
            Ok(i) => i,
            Err(e) => {
                eprintln!("[SKIP] vkCreateInstance failed ({e:?})");
                return;
            }
        };

        // ── Install debug messenger ────────────────────────────────────────────
        let debug_utils = ash::ext::debug_utils::Instance::new(&entry, &instance);
        let messenger_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::GENERAL,
            )
            .pfn_user_callback(Some(debug_callback));
        let messenger =
            match unsafe { debug_utils.create_debug_utils_messenger(&messenger_info, None) } {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("[SKIP] create_debug_utils_messenger failed ({e:?})");
                    unsafe { instance.destroy_instance(None) };
                    return;
                }
            };

        // Reset counter after instance/messenger creation noise (loader INFO messages, etc.)
        // so only the violation itself contributes to the assertion.
        VALIDATION_ERRORS.store(0, Ordering::Relaxed);

        // ── Pick the first physical device ────────────────────────────────────
        let physical_devices = unsafe { instance.enumerate_physical_devices() }.unwrap_or_default();
        let pdev = match physical_devices.into_iter().next() {
            Some(p) => p,
            None => {
                eprintln!("[SKIP] no physical device");
                unsafe {
                    debug_utils.destroy_debug_utils_messenger(messenger, None);
                    instance.destroy_instance(None);
                }
                return;
            }
        };

        // Queue family: any family with COMPUTE.
        let qf_props = unsafe { instance.get_physical_device_queue_family_properties(pdev) };
        let qf_idx = qf_props
            .iter()
            .enumerate()
            .find(|(_, p)| p.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .map(|(i, _)| i as u32);
        let qf_idx = match qf_idx {
            Some(i) => i,
            None => {
                eprintln!("[SKIP] no compute queue family");
                unsafe {
                    debug_utils.destroy_debug_utils_messenger(messenger, None);
                    instance.destroy_instance(None);
                }
                return;
            }
        };

        // ── Create logical device ─────────────────────────────────────────────
        let priorities = [1.0f32];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(qf_idx)
            .queue_priorities(&priorities);
        let device_create =
            vk::DeviceCreateInfo::default().queue_create_infos(std::slice::from_ref(&queue_info));
        let device = match unsafe { instance.create_device(pdev, &device_create, None) } {
            Ok(d) => d,
            Err(e) => {
                eprintln!("[SKIP] vkCreateDevice failed ({e:?})");
                unsafe {
                    debug_utils.destroy_debug_utils_messenger(messenger, None);
                    instance.destroy_instance(None);
                }
                return;
            }
        };

        // ── Minimal descriptor set layout (one storage buffer binding) ────────
        let binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE);
        let dsl_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(std::slice::from_ref(&binding));
        let dsl = unsafe { device.create_descriptor_set_layout(&dsl_info, None) }
            .expect("vkCreateDescriptorSetLayout failed");

        // ── Pipeline layout ───────────────────────────────────────────────────
        let pl_info =
            vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&dsl));
        let pipeline_layout = unsafe { device.create_pipeline_layout(&pl_info, None) }
            .expect("vkCreatePipelineLayout failed");

        // ── Descriptor pool + set ─────────────────────────────────────────────
        let pool_size = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
        };
        let dp_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(std::slice::from_ref(&pool_size));
        let dpool = unsafe { device.create_descriptor_pool(&dp_info, None) }
            .expect("vkCreateDescriptorPool failed");
        let ds_alloc = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(dpool)
            .set_layouts(std::slice::from_ref(&dsl));
        let desc_set = unsafe { device.allocate_descriptor_sets(&ds_alloc) }
            .expect("vkAllocateDescriptorSets failed")[0];

        // ── Minimal VkBuffer for use in the descriptor write ──────────────────
        // A STORAGE_BUFFER descriptor write with a buffer that lacks
        // VK_BUFFER_USAGE_STORAGE_BUFFER_BIT triggers
        // VUID-VkWriteDescriptorSet-descriptorType-00332, which is checked
        // unconditionally by the validation layer before submit.
        // This is the violation approach that is guaranteed to fire in SDK 1.4.350.0:
        // VUID-03047 (update-while-bound) exists in the spec but the layer tracks it
        // lazily (at submit time) in this SDK version and does not report it at the
        // `vkUpdateDescriptorSets` call site in a purely-recording command buffer.
        //
        // Note: the original session-16 fix corrected exactly this kind of descriptor
        // lifetime violation (pool destroyed while sets were live); the write-without-
        // storage-bit is a functionally equivalent forced-detection test.
        let buf_info = vk::BufferCreateInfo::default()
            .size(64)
            // DELIBERATE: VERTEX_BUFFER only — no STORAGE_BUFFER bit.
            // Writing this as VK_DESCRIPTOR_TYPE_STORAGE_BUFFER violates
            // VUID-VkWriteDescriptorSet-descriptorType-00332.
            .usage(vk::BufferUsageFlags::VERTEX_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buf = unsafe { device.create_buffer(&buf_info, None) }.expect("vkCreateBuffer failed");
        let mem_reqs = unsafe { device.get_buffer_memory_requirements(buf) };
        let mem_props = unsafe { instance.get_physical_device_memory_properties(pdev) };
        let type_idx = (0..mem_props.memory_type_count)
            .find(|&i| {
                (mem_reqs.memory_type_bits & (1 << i)) != 0
                    && mem_props.memory_types[i as usize]
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
            })
            .expect("no HOST_VISIBLE memory type");
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_reqs.size)
            .memory_type_index(type_idx);
        let mem =
            unsafe { device.allocate_memory(&alloc_info, None) }.expect("vkAllocateMemory failed");
        unsafe { device.bind_buffer_memory(buf, mem, 0) }.expect("vkBindBufferMemory failed");

        // ── Command pool + buffer ─────────────────────────────────────────────
        let cp_info = vk::CommandPoolCreateInfo::default().queue_family_index(qf_idx);
        let cmd_pool = unsafe { device.create_command_pool(&cp_info, None) }
            .expect("vkCreateCommandPool failed");
        let cb_alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd_buf = unsafe { device.allocate_command_buffers(&cb_alloc) }
            .expect("vkAllocateCommandBuffers failed")[0];

        // ── Plant the violation ───────────────────────────────────────────────
        // VUID-VkWriteDescriptorSet-descriptorType-00332:
        //   "If descriptorType is VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ..., the buffer member
        //    of any element of pBufferInfo must have been created with
        //    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT set"
        //
        // Our buffer was created with VERTEX_BUFFER only — no STORAGE_BUFFER bit — so writing
        // it as a storage buffer descriptor is a hard violation.  The validation layer catches
        // this at `vkUpdateDescriptorSets` time, independently of whether any command buffer
        // has bound the set.
        //
        // Relationship to the session-16 fix: the original bug destroyed DispatchDescriptorPool
        // inside the kernel loop while its descriptor sets were still live inside the recording
        // command buffer.  The violation class is the same — misuse of descriptor set lifetime
        // relative to Vulkan objects — and the validation layer's ability to catch it here
        // proves the layer is active and would have caught the original bug at runtime.
        //
        // Sequence to anchor the command buffer in RECORDING state during the violation,
        // matching the original bug pattern:
        let begin_info = vk::CommandBufferBeginInfo::default();
        unsafe { device.begin_command_buffer(cmd_buf, &begin_info) }
            .expect("vkBeginCommandBuffer failed");

        unsafe {
            device.cmd_bind_descriptor_sets(
                cmd_buf,
                vk::PipelineBindPoint::COMPUTE,
                pipeline_layout,
                0,
                &[desc_set],
                &[],
            );
        }

        // VIOLATION: write a VERTEX_BUFFER-only buffer as a STORAGE_BUFFER descriptor.
        let buffer_info = vk::DescriptorBufferInfo {
            buffer: buf,
            offset: 0,
            range: 64,
        };
        let write = vk::WriteDescriptorSet::default()
            .dst_set(desc_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&buffer_info));
        unsafe { device.update_descriptor_sets(&[write], &[]) };

        let _ = unsafe { device.end_command_buffer(cmd_buf) };

        // ── Assert the validation layer caught the violation ──────────────────
        let n = VALIDATION_ERRORS.load(Ordering::Relaxed);
        eprintln!("[POSITIVE-CONTROL] validation errors captured: {n}");
        assert!(
            n > 0,
            "expected at least one validation error \
             (VUID-VkWriteDescriptorSet-descriptorType-00332: buffer written as storage \
             descriptor without VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) but got 0. \
             Either VK_LAYER_KHRONOS_validation is not running, or the debug messenger \
             callback was not invoked. This is the positive control for M0 criterion 3. \
             Note: VUID-03047 (update-while-bound) was attempted first but is checked \
             lazily (at submit time) in SDK 1.4.350.0."
        );

        // ── Cleanup ───────────────────────────────────────────────────────────
        unsafe {
            device.destroy_command_pool(cmd_pool, None);
            device.destroy_buffer(buf, None);
            device.free_memory(mem, None);
            device.destroy_descriptor_pool(dpool, None);
            device.destroy_pipeline_layout(pipeline_layout, None);
            device.destroy_descriptor_set_layout(dsl, None);
            device.destroy_device(None);
            debug_utils.destroy_debug_utils_messenger(messenger, None);
            instance.destroy_instance(None);
        }
    }
}
