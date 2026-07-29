# Project Context

- **Owner:** Justin Chu
- **Project:** onnxruntime-ep-vulkan — a cross-platform Vulkan plugin execution provider for ONNX Runtime, written in Rust.
- **Reference architecture:** `C:\Users\justinchu\dev\onnxruntime-mlx` — mirror its layout: `rust/src/{lib,ep,factory,sys,registry,engine,logging}.rs`, `rust/src/ops/*`, `rust/build.rs`, `tests/conformance/`, `bench/`, `python/`, `docs/DESIGN.md`.
- **Stack:** Rust (cdylib plugin EP), Vulkan 1.1+ compute, SPIR-V/GLSL shaders, ONNX Runtime C API, Python bindings, GitHub Actions CI.
- **Cross-platform mandate:** Windows, Linux, Android, macOS via MoltenVK; NVIDIA / AMD / Intel / Adreno / Mali; software rasterizer (lavapipe / SwiftShader) for GPU-less CI.
- **My focus:** Vulkan Compute — device/memory/sync, SPIR-V shaders, pipelines
- **Created:** 2026-07-28T17:52:04-07:00

## Learnings

<!-- SUMMARIZED by Scribe 2026-07-29T09:00:39-07:00 — full session details in decisions.md -->

### [SUMMARY] Sessions 1–11: ENGINE.md through first real dispatch (2026-07-28–2026-07-29)

**Stack chosen (session 1):** `ash` + `gpu-allocator`. Buffer-only tensor storage, one command buffer per subgraph, per-edge barriers, GLSL→SPIR-V at build time.

**Barrier abstraction (session 2) — `rust/src/vk/barrier.rs`:** Dual-backend `Barriers` enum (`Sync2Backend` / `LegacyBackend`). Backend selected once at `Device::new`. `ep.force_legacy_barriers` session option. `ONNXRUNTIME_EP_VULKAN_BACKEND_PROBE` env writes "sync2"/"legacy" for Trinity parity test. `barrier.rs` is the ONLY file allowed to name barrier types (enforced by `tests/layering.rs`).

**ash 0.38 / Rust 2024 permanent reference:** `push_next` is `#[must_use]`; `ash::khr::*` path. `ash::Device` cannot be zeroed. `entry.try_enumerate_instance_version()` not the panicking form. `c"main"` for C strings. `bytes.len().div_ceil(4)` (clippy enforces). `#![deny(unsafe_op_in_unsafe_fn)]` requires explicit `unsafe {}` + SAFETY comment even inside unsafe fns.

**Engine seams (session 4):** `TileConfig`/`PackKey`/`PrepackRequest`; `bind_aliased_output` default method (KV-cache aliasing); `VariantRow`/`parse_shader_variants` in `build.rs`; `IndirectKernelRequest`/`dispatch_indirect` default. All new methods have defaults; no existing implementors broken. llama.cpp tiling strategy + subgroup reduction shape + dequant-in-register patterns transfer (D-S4-10); no code copying.

**Device enumeration (session 5):** `Instance::enumerate_capable_devices` applies R1–R6 gate (`passes_gate` pure function, 15 unit tests). `probe_devices` sorts discrete-first. `MemClass`: `DeviceLocal`, `Upload`, `Download`, `PackedWeights`. `PipelineCache`: lazy `(shader_stem, spec_constants) → (VkPipeline, layout, dset_layout)`. Test count reached 268.

**Shader-less guard (session 7+):** `shaders::has_any()` = `SHADER_MODULES.is_empty()`. `probe_devices` + `get_capability_impl` both early-exit. OQ-4 escape hatch: `ALLOW_MISSING_GLSLC=1` produces inert artifact.

**Session 9 — ICD diagnostics + apiVersion fix:** `Instance::create` caps requested `apiVersion` to loader version before calling `vkCreateInstance` (fixes latent `ERROR_INCOMPATIBLE_DRIVER` on 1.0 loaders; use `try_enumerate_instance_version` not the panicking form). Full loader diagnostic emitted on any create failure. `epctl --probe-loader` exits 1 when no capable device passes gate.

**Session 10 — R5 removed from gate (lavapipe `supportedStages=0`):** R5 (subgroup BASIC in compute) removed from `passes_gate`. Now stored in `Capabilities::subgroup_basic_in_compute`. `assess_gate` evaluates all criteria without early exit for verbose diagnostics. 283 tests.

**Session 11 — First real dispatch on NVIDIA RTX 4060 Laptop GPU:**
- `add_f32_dispatches_end_to_end`: 1024 f32 elements, exact arithmetic, zero validation layer errors. §9.1.2 "no shader has ever executed" is now false.
- **Bug D-S11-01:** NVIDIA 1.4 doesn't export `vkCmdPipelineBarrier2KHR` (extension alias); fixed: `Sync2Backend` is now `Core(Box<ash::Device>)` / `Khr(...)`.
- **Bug D-S11-02:** `Instance::create` was requesting Vulkan 1.1 hardcoded; `vkGetDeviceProcAddr` only returns pointers up to requested version. Fixed: request `loader_version.min(1.3)`.
- **Bug D-S11-03:** Missing feature chain in `VkDeviceCreateInfo`; fixed by adding `VkPhysicalDeviceSynchronization2Features` to `pNext` chain when enabling sync2.
- Test count at end: 258 lib + 6 dump-capabilities + 26 layering + 7 portability = 297.

**Outstanding (not yet done):** GPU timestamp query implementation (Niobe D-N4/D-N5 spec → Switch `vk/cmd.rs`). `dispatch_integration.rs` has 4 `undocumented_unsafe_blocks` + rustfmt. `bind_aliased_output` seam contract with Mouse. OQ-15 `vkCmdDispatchIndirect` evaluation.

---

## Cross-agent context appended (2026-07-29T09:00:39-07:00) — first-hardware round

📌 **Local GPU facts (2026-07-29):** Vulkan SDK at `C:\VulkanSDK\1.4.350.0` — NOT on default PATH; prefix it explicitly in every shell command that calls glslc or epctl. Two devices pass §7.2 gate: **Intel Iris Xe Graphics** (Vulkan 1.4.309, UMA, 32 KiB shared, `maxSubgroupSize=32`) and **NVIDIA GeForce RTX 4060 Laptop GPU** (Vulkan 1.4.325, discrete, 48 KiB shared, `maxSubgroupSize=32`). 168 shader variants compile cleanly with SDK on PATH.

📌 **Intel Iris Xe = spec-conformance oracle (2026-07-29, Morpheus D25 + Link):** Intel's implementation is stricter than NVIDIA's on undefined behaviour and extension interactions. When the two disagree, assume Intel is correct. Use Intel results when filing bug reports or proposing spec questions.

📌 **R5 changed — subgroup BASIC in compute is now a probed capability, not a gate criterion (2026-07-29, Switch D-S10-01):** Removed from `passes_gate`; stored in `Capabilities::subgroup_basic_in_compute`. Mesa llvmpipe reports `supportedStages = 0`. Shader variants that require subgroup ops must check this capability before claiming the node, not at device-enumeration time.

📌 **`rustfmt --edition 2021` silently no-ops on this edition-2024 crate (2026-07-29, Tank D-T12):** Always use `cargo fmt --all` — the xtask `cargo ci` command does this correctly. Never invoke rustfmt directly with an edition flag.

📌 **GPU timestamp query spec (2026-07-29, Niobe D-N4/D-N5):** `ONNXRUNTIME_EP_VULKAN_TRACE_GPU=1` activates GPU spans. Niobe owns `trace.rs`; Switch must implement the `vkCmdWriteTimestamp` call sites, query pool creation/reset, `timestampPeriod` scaling, non-stalling `vkGetQueryPoolResults`, and `VK_EXT_calibrated_timestamps` path — full spec in `docs/PERF.md §3`. Return type is `GpuTimestampReport { calibration, queue_family, intervals }` with raw ticks; Niobe's `trace.rs` owns all arithmetic.

📌 **`dispatch_integration.rs` has 4 flagged `undocumented_unsafe_blocks` + rustfmt issue (2026-07-29, Niobe):** Switch must fix these before the next `cargo ci` pass.

---

## Session 12 — Multi-device dispatch; Intel oracle; caps probe fix (2026-07-29T08:13:58-07:00)

**Coordinator directive:** run dispatch on ALL capable devices with Intel as strictness oracle. Add explicit device selection via env var.

**D-S12-01 — caps probe `push_next` chain bug (FIXED):**
`let _ = props2.push_next(...)` discarded the chain link — ALL capability fields derived from
`VkPhysicalDeviceProperties2`/`Features2` chains were reading zeroed structs. Symptom: `subgroup_sz=0`
on both devices. Fix: rebind `props2 = { let p = ...push_next(...); ... }`. This is the same bug as
`DeviceFeatureChain` in session 11. After fix: `subgroup_sz=32` on both Intel Iris Xe and NVIDIA RTX 4060.

**D-S12-02 — `is_uma=true` on NVIDIA RTX 4060 Laptop is correct:**
Not a bug. ReBAR maps full VRAM as `HOST_VISIBLE`. `detect_uma` correctly identifies this.
`is_uma` means "main GPU memory is CPU-writable" not "same physical DRAM as CPU". Keep as-is.

**D-S12-03 — Intel Iris Xe: zero validation errors on Add dispatch:**
Intel passed with zero errors, same as NVIDIA. Barrier strategy, feature chain, descriptor set layout,
push constants all spec-conformant. Conformance is symmetric for this workload. Reported to Link for
driver-quirks watchlist: first hardware-derived entry (previous entries were from documentation).

**D-S12-04 — `ONNXRUNTIME_EP_VULKAN_DEVICE` env var:**
Added `select_device()` and `ENV_DEVICE_SELECTOR` in `instance.rs`. Dispatch integration test runs
ALL capable devices regardless of selector (selector is for EP factory device selection, not test scope).
`epctl --probe-loader` shows selector value and which device would be selected.

**Results:**
- Both devices PASS, zero validation errors
- `add_f32_dispatches_end_to_end` verified on Intel Iris Xe (1.4.309, UMA, subgroup_sz=32) and NVIDIA RTX 4060 (1.4.325, discrete-ReBAR, subgroup_sz=32)
- `cargo ci` green (fmt + clippy + build + test), 258 lib tests
- Fixed 3 `undocumented_unsafe_blocks` (2 in `instance.rs`, 1 in `dispatch_integration.rs`) + `rustfmt` issue

**Outstanding:** GPU timestamp query hooks for Niobe (`vk/cmd.rs`). `bind_aliased_output` seam
with Mouse. Session lifecycle (`VulkanEp` holding `Instance` + `Device`). Real `DispatchContext`.

---

## Session 12b — Cross-platform standing directive response (2026-07-29T09:39:59-07:00)

**Coordinator directive:** 要时刻注意跨平台通用性. All limits from device reports, never hardcoded. UMA is the mobile proxy.

**Changes made:**

1. **`dispatch_integration.rs` — replace `256` with `EW_LOCAL_SIZE`:** The integration test used
   `vec![256u32, 1u32]` and `plan.workgroups_1d(256)` as hardcoded constants. Replaced with
   `crate::ops::common::templates::EW_LOCAL_SIZE`. Now if the constant changes (e.g. per-device
   tuner), the integration test tracks it automatically. The dispatch still passes on both devices.

2. **`alloc.rs` — `MemClass::Download` now uses `GpuToCpu` (was `CpuToGpu`):** The original code
   used `CpuToGpu` for both Upload and Download. `Download` is GPU-writes/CPU-reads; `GpuToCpu`
   signals this to `gpu-allocator` so it can prefer cached HOST_VISIBLE memory for readback. On
   UMA devices (all same heap) it makes no difference; on discrete hardware the hint can select
   a different BAR sub-type. Unit test renamed to `mem_class_download_maps_to_gpu_to_cpu`.

3. **`ENGINE.md §3.2/§3.3` — updated to match reality:** §3.2 now shows the 4 actual
   `MemClass` entries with `MemoryLocation` hints and notes on UMA behavior. §3.3 corrects the
   "no staging needed on UMA" statement (v0 always stages; bypass is a future M1+ optimisation),
   explains the future UMA bypass path using `caps.is_uma` + `Access::HostWrite`, and notes the
   Intel Iris Xe test confirms the staging path works correctly on UMA.

4. **`ENGINE.md §2.2` — Intel oracle and UMA rationale now explicit:** The section explains the
   conformance asymmetry (correct on NVIDIA + wrong on Intel → we relied on something unspecified),
   the `ONNXRUNTIME_EP_VULKAN_DEVICE` env var for deterministic per-device testing, and the
   UMA/ReBAR distinction (discrete + ReBAR vs integrated UMA — both `is_uma=true`, different
   physical topology).

5. **`ENGINE.md §9.2` — workgroup size note:** Updated "256 threads" to refer to `EW_LOCAL_SIZE`
   and its cross-platform rationale.

**Structural rule encoded (not just documented):**
- Workgroup sizes arrive from the shared constant `EW_LOCAL_SIZE`; the shader exposes it as
  spec constant ID 0 so the runtime can override per device without recompiling GLSL.
- `MemoryLocation` hints match actual usage semantics: `CpuToGpu` = CPU writes, `GpuToCpu` = CPU reads.
- `cargo ci` green (258 lib tests, fmt, clippy).

