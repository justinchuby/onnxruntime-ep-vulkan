# Project Context

- **Owner:** Justin Chu
- **Project:** onnxruntime-ep-vulkan — a cross-platform Vulkan plugin execution provider for ONNX Runtime, written in Rust.
- **Reference architecture:** `C:\Users\justinchu\dev\onnxruntime-mlx` — mirror its layout: `rust/src/{lib,ep,factory,sys,registry,engine,logging}.rs`, `rust/src/ops/*`, `rust/build.rs`, `tests/conformance/`, `bench/`, `python/`, `docs/DESIGN.md`.
- **Stack:** Rust (cdylib plugin EP), Vulkan 1.1+ compute, SPIR-V/GLSL shaders, ONNX Runtime C API, Python bindings, GitHub Actions CI.
- **Cross-platform mandate:** Windows, Linux, Android, macOS via MoltenVK; NVIDIA / AMD / Intel / Adreno / Mali; software rasterizer (lavapipe / SwiftShader) for GPU-less CI.
- **My focus:** Vulkan Compute — device/memory/sync, SPIR-V shaders, pipelines
- **Created:** 2026-07-28T17:52:04-07:00

## Learnings

<!-- SUMMARIZED by Scribe 2026-07-28T22:28:08-07:00 — full session details in decisions.md -->

### [SUMMARY] Sessions 1–6+: ENGINE.md, barrier abstraction, seams, device/memory/pipeline (2026-07-28)

**ENGINE.md authored (session 1):**
- Reference study: llama.cpp (build-time GLSL→SPIR-V, specialization constants, per-vendor tuning, lazy pipeline creation). ExecuTorch (VK_API_VERSION_1_1, buffer-only, one-time record, weight prepacking at compile phase, yaml variant tables).
- Chosen stack: `ash` + `gpu-allocator` (not vulkano, not wgpu).
- Buffer-only tensor storage for v0. One command buffer per subgraph (no per-op submissions).
- Per data-edge barriers (`vkCmdPipelineBarrier2`), not global. GLSL→SPIR-V at build time.
- `synchronization2` and `subgroup_size_control` structurally simplify engine; baseline decision delegated to Morpheus.

**Barrier abstraction (session 2) — `rust/src/vk/barrier.rs`:**
- Dual-backend `Barriers` enum: `Sync2Backend`, `LegacyBackend`. `Access`/`Stage` closed enums (no None). Single mapping table.
- Backend selected once at `Device::new`. `ep.force_legacy_barriers` session option forces legacy.
- `ONNXRUNTIME_EP_VULKAN_BACKEND_PROBE` env var: EP writes "sync2" or "legacy" to a file during `Barriers::select`. Used by Trinity's parity test.
- Layering lint: `barrier.rs` is the ONLY file allowed to name barrier types; `BARRIER_RULES` + `SYNC2_FIELD_RULES` in `tests/layering.rs`.
- ash 0.38 notes: `push_next` is safe but `#[must_use]`; extension paths are `ash::khr::*`; `vk::DependencyInfo` uses `vk::MemoryBarrier2` for execution-only sync2 barriers.
- Rust 2024: `#![deny(unsafe_op_in_unsafe_fn)]` — unsafe fn calls inside unsafe fn still need explicit `unsafe {}` + SAFETY comment. `const { assert!(...) }` preferred form.

**Backend probe + force_legacy wiring (session 3) — `rust/src/vk/device.rs`:**
- `Device` struct owns `ash_device`, `physical_device`, `caps`, `barriers`. `Device::new` is sole call site of `Barriers::select`.
- `should_use_sync2(caps, force_legacy) -> bool` extracted for testability (ash::Device cannot be zeroed — non-nullable fn pointers → UB).
- `caps::test_caps(sync2: bool)` defined outside `mod tests` so `device.rs` tests import without touching `synchronization2` token (layering compliance).
- Total: 185 tests after session.

**Engine seams for XL kernels (session 4) — `rust/src/engine.rs`:**
- Seam 1 (prepack): `TileConfig`, `PackKey`, `PackInput`, `PackOutput`, `PrepackRequest`, `PrepackResult`, `CompileContext` trait.
- Seam 2 (KV-cache aliasing): `bind_aliased_output` default method on `DispatchContext` (returns resolved input by default).
- Seam 3 (build.rs variant table): `VariantRow`, `parse_shader_variants`, two-path compile in `build.rs`. `cargo:rerun-if-changed` for `shader_variants.txt`.
- Seam 4 (indirect dispatch): `IndirectKernelRequest`, `dispatch_indirect` default method.
- llama.cpp assessment: block format mismatch = no code copying. Tiling strategy, subgroup reduction shape, dequant-in-register patterns **do transfer**. (D-S4-10 correction of Mouse's "useless" claim.)
- Rust trait default methods returning `Err(...)` = correct pattern for stubs that concrete engine impls override. All new methods have defaults; no existing implementors broken.
- Total: 195 tests after session.

**Real device enumeration (session 5) — `rust/src/vk/instance.rs`:**
- `Instance` struct: `_entry` declared first (dropped last); `ash::Entry::new()` returns None gracefully when no loader.
- `Instance::enumerate_capable_devices()` applies R1–R6 gate (pure `passes_gate` function, 15 unit tests).
- `Capabilities::required_device_extensions(api_version)` lives in `caps.rs` (keeps `synchronization2` token out of `instance.rs`).
- `Device::create(instance, capable, force_legacy)` — logical device creation, compute queue retrieval.
- `probe_devices()` sorts discrete-first.
- glslc fallback: Switch recommended hard SDK dep + escape hatch (168 SPIR-V blobs ≈ 1–3 MiB binary weight + staleness hazard). Morpheus ruled for hard SDK dep (OQ-4 resolved).
- Total: 245 tests after session.

**Memory / command / pipeline (session 6) — `alloc.rs`, `cmd.rs`, `pipeline.rs`:**
- `MemClass`: `DeviceLocal`, `Upload`, `Download`, `PackedWeights` (maps to `GpuOnly` — enforces "no dequantized weight in VRAM" at type level).
- `CommandPool` + `CommandRecorder<'pool>`: lifetime prevents use-after-pool-drop at compile time. `Drop` logs warning if `finish()` not called.
- `submit_and_wait()`: fence-based blocking submit. V0: one submission per subgraph.
- `PipelineCache`: lazy build+cache `(shader_stem, spec_constants) → (VkPipeline, VkPipelineLayout, VkDescriptorSetLayout)`. Shader module destroyed after pipeline creation.
- `DispatchDescriptorPool`: per-dispatch pool-and-reset. V0 simple model; M2+ replaces with persistent.
- `vk::SpecializationInfo` borrows both map_entries and data — return the storage from a helper, construct in caller scope.
- Total: 265 tests after session.

**Shader-less guard (session 7+):**
- `shaders::has_any()` = `SHADER_MODULES.is_empty()`. `probe_devices()` returns `vec![]` + logs warn. `get_capability_impl()` early-returns null + logs `[built-without-shaders]`.
- Belt-and-suspenders: `probe_devices` (factory init) + `get_capability_impl` (per-session). Future refactor can skip either; both together prevent claiming.
- OQ-4 condition 3 implemented: shader-less artifact advertises zero devices, claims nothing.
- Total: 268 tests.

**Key ash 0.38 / Rust 2024 facts (permanent reference):**
- All `ash::Instance` methods are `unsafe`. `ash::khr::synchronization2::Device::new` is safe.
- `push_next` is `#[must_use]`; use `let _ = props2.push_next(...)`.
- Extension paths: `ash::khr::*` (not `ash::extensions::khr::*`).
- `ash::Device::clone()` is cheap (Arc internally).
- `gpu_allocator::vulkan::Allocator::new()` is safe.
- `c"main"` is the modern c-string literal (Rust 1.77+).
- `bytes.len().div_ceil(4)` — clippy `manual_div_ceil` enforces this.
- `ash::Device` / `ash::Instance` cannot be zeroed (non-nullable fn pointers).

**Current test count: 268 (233 lib + 6 dump-capabilities + 26 layering + 3 shader-guard). All passing.**

---

### 2026-07-29T05:17:03-07:00 — Session 9: ICD diagnostics, apiVersion fix, epctl --probe-loader

**Task:** Diagnose `ERROR_INCOMPATIBLE_DRIVER` on both CI lanes (run 30450284838). The EP loaded
and the degradation path worked (M0 criterion 5 confirmed — it advertised zero devices and let
ORT fall back to CPU). But no Vulkan device could be enumerated, so M0 cannot be verified until
the ICD issue is resolved.

**Diagnosis:**
- Root cause is environmental: lavapipe ICD either missing or its library isn't loadable on both
  CI runners. NOT a bug in our code.
- Linux: `vulkaninfo` already warned in the install step (non-fatal); EP then got
  `ERROR_INCOMPATIBLE_DRIVER` as expected. The `::warning::` failure mode masked the real problem.
- Windows: ICD JSON was found, but the mesa DLL or its dependencies (MSVC runtime) may not be
  loadable.
- Neither lane had a pre-test Vulkan availability check that FAILS CI — both runners just silently
  had no lavapipe.

**Code changes made (this session):**

1. **`vk/instance.rs` — loader diagnostic function `loader_state_lines`:**
   - Always emitted at WARN level on any `vkCreateInstance` failure.
   - Emitted at INFO level pre-creation when `ONNXRUNTIME_EP_VULKAN_VERBOSE=1`.
   - Reports: `VK_ICD_FILENAMES`, `VK_DRIVER_FILES`, `VK_INSTANCE_LAYERS` env var values;
     loader version from `vkEnumerateInstanceVersion`; layer count and names; instance extension
     count (indicator of whether any ICD loaded).

2. **`vk/instance.rs` — apiVersion fix (defensive correctness):**
   - `Instance::create` now calls `try_enumerate_instance_version` before building
     `VkApplicationInfo`.
   - If loader version is None (Vulkan 1.0) or < 1.1: return None early with clear message rather
     than hitting `ERROR_INCOMPATIBLE_DRIVER` from requesting apiVersion 1.1 against a 1.0 loader.
   - ash 0.38's `try_enumerate_instance_version()` returns `Ok(None)` for 1.0 loaders (function
     not present) and `Ok(Some(v))` for 1.1+ loaders. Never panics (vs the deprecated
     `enumerate_instance_version` which does panic on 1.0).

3. **`vk/instance.rs` — `probe_loader_report()` public fn:**
   - Standalone loader probe: loads ash, collects loader state, tries `vkCreateInstance`, applies
     §7.2 gate, returns multi-line diagnostic string. Bypasses the shader guard in
     `probe_devices()` so it works on shader-less builds. Used by epctl.

4. **`engine.rs` — `pub fn loader_probe_report()`:**
   - Thin wrapper around `vk::instance::probe_loader_report()`. Exposed as `pub` so `epctl`
     (a binary in the same crate, importing via `onnxruntime_vulkan_ep::engine`) can call it.

5. **`epctl.rs` — `--probe-loader` flag (cross-owner edit — Tank owns epctl.rs):**
   - New flag: runs `engine::loader_probe_report()`, prints to stdout.
   - Exits 1 when no capable device found (usable as a gate step in CI scripts).
   - Previously epctl was entirely static (no Vulkan, no ORT). This is the one addition that
     touches Vulkan. All existing `--dump-capabilities` behavior unchanged.

**What Link and Trinity need to do (decisions/inbox/switch-icd-diagnostics.md):**
1. Set `VK_DRIVER_FILES` alongside `VK_ICD_FILENAMES` in `ci.yml` — newer loaders may prefer it.
2. Linux: make `vulkaninfo` failure a hard `exit 1` (not `::warning::`) to catch lavapipe issues
   before tests run.
3. Linux: verify `mesa-vulkan-drivers` actually ships lavapipe on ubuntu-22.04 GitHub runners;
   consider installing from the LunarG apt repo which is already added for shaderc.
4. Windows: verify mesa DLL dependencies are loadable (check MSVC runtime availability).
5. Add `epctl --probe-loader || exit 1` as a CI step before pytest to make Vulkan availability
   explicit and named in the job log.

**ash 0.38 lesson learned:**
- `entry.try_enumerate_instance_version()` is unsafe and returns `VkResult<Option<u32>>`. The
  deprecated `entry.enumerate_instance_version()` panics on Vulkan 1.0 loaders. Always use
  `try_enumerate_instance_version`.
- `entry.enumerate_instance_layer_properties()` and `entry.enumerate_instance_extension_properties`
  are both unsafe. Both are loader-level queries that work without any ICD loaded.

**What is verified vs written-but-unexercised:**
- VERIFIED by unit tests (no ICD needed): all prior tests (272 total now); diagnostic functions
  are exercised indirectly (paths don't panic on an ICD-less machine).
- VERIFIED by real ORT run: M0 exit criterion 5 (zero-devices → CPU fallback) confirmed on both
  CI lanes by run 30450284838, before ICD fix.
- WRITTEN BUT NOT EXERCISED: `loader_state_lines` full output in the failure path (will fire on
  next CI run with new code); `epctl --probe-loader` output (needs CI with Vulkan loader present).
- UNBLOCKED: once Trinity/Link fix lavapipe, the new diagnostics will show "vkCreateInstance:
  OK" + device count, and `probe_devices()` will return real devices.

**Final state:** `cargo ci` green (rustfmt + clippy + build + test). **272 tests** (238 lib + 6
dump-capabilities + 26 layering).

---

### Session 10: R5 gate removal, `assess_gate`, and realistic device-profile tests (2026-07-28T22:28:08-07:00)

**CI run:** `30456272132` (headSha `c615f17`)  
**Problem:** `vkCreateInstance` now succeeds (session 9 `apiVersion` fix worked). But
`epctl --probe-loader` reported `0 device(s) passed the §7.2 capability gate`. `vulkaninfo`
confirmed a real Vulkan 1.3.255 lavapipe device (`llvmpipe (LLVM 15.0.7, 256 bits)`) exists.
Our own gate is rejecting lavapipe.

**Root cause:** R5 (`subgroup_props.supported_stages.contains(COMPUTE) &&
supported_operations.contains(BASIC)`) was the culprit. Mesa llvmpipe on Ubuntu 22.04 reports
`supportedStages = 0` — no stage is listed as supporting subgroup operations in this Mesa build.
R5a fails immediately.

This violates Morpheus's §7.0 governing principle verbatim: *"capability shortfalls degrade
op coverage, not device availability."* R5 is not a correctness requirement for device admission;
it is a capability that gates individual ops.

**Code changes (session 10):**

1. **`vk/instance.rs` — R5 removed from `passes_gate`:**
   - Gate now checks R1–R4, R6 only.
   - `passes_gate` signature drops the `subgroup_props` parameter.
   - `enumerate_capable_devices` no longer queries subgroup properties via `props2` chain
     (that query happens in `caps::probe` which runs after the gate).
   - Decision recorded in `switch-engine-seams.md` D-S10-01.

2. **`vk/instance.rs` — `GateCriterion` struct and `assess_gate` function:**
   - `GateCriterion` holds `label`, `requirement`, `measured` value, `passed`, and `failure_reason`.
   - `assess_gate` evaluates all five criteria without early exit, returning the full breakdown.
   - `passes_gate` is now a thin wrapper: calls `assess_gate`, returns `Err(failure_reason)` on
     first failure. Error strings are unchanged — Trinity's tests still assert on them.
   - `probe_loader_report` now iterates physical devices directly and calls `assess_gate` per
     device, showing label / requirement / measured / verdict for each criterion.
   - `enumerate_capable_devices` calls `assess_gate` at DEBUG on gate failure.

3. **`vk/caps.rs` — `subgroup_basic_in_compute: bool` added to `Capabilities`:**
   - Replaces the old R5 gate semantics.
   - Set in `caps::probe` from `subgroup_props.supported_stages.contains(COMPUTE) &&
     supported_operations.contains(BASIC)`.
   - Updated comment on `subgroup_supported_ops` (removed "BASIC is guaranteed by R5").
   - `test_caps()` and `caps_with_synchronization2()` updated.

4. **`vk/instance.rs` — test updates:**
   - Removed: `r5a_rejects_subgroup_not_in_compute_stage`, `r5b_rejects_missing_basic_subgroup_ops`
   - Removed: `good_subgroup_props()` helper (no longer needed by gate tests)
   - Added: `device_without_subgroup_compute_passes_gate` — pins R5 removal explicitly
   - Added: `lavapipe_profile_passes_gate` — synthesised from Ubuntu 22.04 Mesa properties
   - Added: `uma_integrated_gpu_passes_gate` — UMA combined DEVICE_LOCAL+HOST_VISIBLE heap
   - Added: `discrete_gpu_passes_gate` — two-heap discrete GPU with resizable-BAR type
   - Added: `assess_gate_reports_measured_values_and_identifies_failure` — verifies measured
     values are present and only the failing criterion is reported as failed

**Key technical note:**
- `subgroup_props` was queried in `enumerate_capable_devices` only for R5. After R5 removal,
  that chain is gone from the gate loop. Subgroup properties are still queried in `caps::probe`
  (which runs per-device after the gate passes), so no information is lost.
- The `probe_loader_report` verbose output now answers "which criterion rejected this device
  and what did it measure?" in a single tool invocation. This was the missing diagnostic that
  made session 9 and 10 hard to triage.

**What is verified vs written-but-unexercised:**
- VERIFIED by unit tests (no ICD needed): 283 total (up from 272). All gate logic is exercised
  including realistic lavapipe, UMA, and discrete profiles.
- WRITTEN BUT NOT EXERCISED on real hardware: `assess_gate` verbose output in
  `probe_loader_report` (will show on next CI run with lavapipe once the gate passes).
- UNBLOCKED: `enumerate_capable_devices` will now return lavapipe as a capable device on CI.
  The next milestone is Device::new becoming real.

**ash 0.38 lesson (confirmed this session):**
- `get_physical_device_properties2` and `get_physical_device_memory_properties` are both unsafe.
  Clippy's `undocumented_unsafe_blocks` lint requires a SAFETY comment on the immediately
  preceding line for each `unsafe {}` block in a loop body — a single comment before the
  first block does not cover subsequent ones.

**Final state:** `cargo ci` green. **283 tests** (241 lib + 6 dump-capabilities + 26 layering + 7 portability).