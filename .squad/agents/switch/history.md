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

## Session 12c — UMA predicate fix + timestamp seam (2026-07-29T09:47:45-07:00)

**Coordinator directive:** fix failing `mem_class_download_maps_to_cpu_to_gpu` test, fix UMA
predicate (Niobe's measurement), add timestamp seam for Niobe.

**D-S12b-01 — `is_uma` predicate corrected:**
Previous "largest DL heap also HV" returned `true` for discrete+ReBAR. New: every heap is
DEVICE_LOCAL. Result: NVIDIA RTX 4060 now `uma=false` (was wrong `true`). Intel Iris Xe
remains `uma=true`. Extracted into `is_uma_memory(mem_props)` pure function with 5 unit tests
including an explicit test that discrete+ReBAR is NOT UMA.

**D-S12b-02 — `Capabilities::timestamp_period_ns` and `timestamp_valid_bits`:**
Verified Niobe's measured values: Iris Xe=52.0833 ns/tick (ts_bits=36), RTX 4060=1.0 ns/tick
(ts_bits=64). 52× difference confirmed on hardware. Fields added to `Capabilities`, populated
in `probe()`. The "NOT converted in vk/" contract is explicit in the doc comment.

**D-S12b-03 — test fix:**
`mem_class_download_maps_to_cpu_to_gpu` renamed to `mem_class_download_maps_to_gpu_to_cpu`
and changed to assert `GpuToCpu` — was the pre-existing failing test the coordinator reported.

**Results:**
- NVIDIA RTX 4060: `uma=false`, `ts_period=1.0000ns`, `ts_bits=64`, dispatch PASS
- Intel Iris Xe: `uma=true`, `ts_period=52.0833ns`, `ts_bits=36`, dispatch PASS
- `cargo ci` green (268 lib tests, fmt, clippy)

**Outstanding:** GPU timestamp query pool hooks in `cmd.rs` (Niobe owns spec, Switch owns
`vkCmdWriteTimestamp` call sites per D-N4/D-N5). Session lifecycle. Real `DispatchContext`.

---

## Session 14 — §7.9 probe-validity; R5 re-evaluation; cargo ci green (2026-07-29T13:42:45-07:00)

**Coordinator task:** Re-check whether lavapipe's `supportedStages = 0` in session 10 was a
real device fact or the push_next probe bug. Implement §7.9 three-state probe and raw-value
audit. Correct misleading lavapipe comment.

**Finding — D-S14-01 — R5 removal premise was wrong:**
Mesa 26.1 lavapipe DOES support subgroup BASIC in compute. The `supportedStages = 0` reading
in session 10 was the push_next bug (D-S12-01 class) — a zeroed pNext chain returns all zeros
regardless of device capability. Confirmed via web search (Mesa 26.1 docs) and CI probe output
showing Mesa 26.1.3 on Windows lavapipe as Vulkan 1.4.348. The policy decision (R5 not in gate
per §7.0) remains correct for independent reasons. Morpheus needs to update §7.2's R5 rationale.

**Changes — D-S14-02 — §7.9 implementation:**

1. **`vk/caps.rs`** — added two fields to `Capabilities`:
   - `subgroup_probe_valid: bool` — false when `subgroup_size == 0` on a ≥1.1 device (§7.9 rule 1)
   - `subgroup_supported_stages: vk::ShaderStageFlags` — raw stage flags (§7.9 rule 3 audit trail)
   - `probe()` sets `subgroup_probe_valid = false` and logs WARN when size is zero
   - `subgroup_basic_in_compute` derivation gates on `probe_valid`
   - 3 new unit tests for three-state probe behavior
   - Updated all test struct literals and `test_caps()` to include new fields

2. **`vk/instance.rs`** — `probe_loader_report()` now calls `caps::probe` for each gate-passing
   device and shows raw capability values (subgroup size, stages, probe validity, is_uma,
   timestamp info) — the output that would have caught D-S12-01 immediately

3. **`vk/instance.rs`** — corrected R5 comment and `lavapipe_profile_passes_gate` test to
   document that the original `supportedStages=0` was a probe bug, not a device fact

4. **`tests/layering.rs`** — fixed pre-existing compile error (`s.op` → `s.op_type`) in
   `no_default_domain_row_carries_a_contrib_schema_baseline` test

**Decisions:** D-S14-01 and D-S14-02 in `.squad/decisions/inbox/switch-engine-seams.md`.

**State at end of session:**
- `cargo ci`: ✅ GREEN
- §7.9 three-state probe: ✅ implemented
- Raw capability values in `epctl --probe-loader`: ✅ implemented
- R5 finding documented: ✅ — premise wrong, policy correct, Morpheus update needed
- layering.rs compile error: ✅ fixed


**Context:** M0 task from coordinator: build the Compile → OrtNodeComputeInfo → Compute wire for
`Add` fp32 through ORT. Prior sessions had wired `compile_impl` / `compute_impl` (ep.rs +
session.rs), but three guard tests were still asserting the old `Staged` state for `Add`.

**Changes made this session:**

1. **`registry.rs` guard test fix** — `staged_rows_are_not_registered_for_translation` was using
   `Add` as the staged example after Add was promoted to `Live`. The test was asserting
   `spec_for(&add_desc).is_none()`, but `spec_for` now returns `Some` for a Live op. Renamed to
   `live_rows_are_registered_for_translation_and_staged_rows_are_not`; uses `Sub` (Staged) as
   the negative example; adds a positive assertion that Add IS translatable.

2. **`cargo ci --fix`** — rustfmt and clippy warnings in the rewritten test block; auto-fixed.

3. **`cargo ci` PASSED** — 258 lib tests, 6 dump-capabilities, 26 layering, 7 portability; all
   green. fmt + clippy clean.

**D-S13-01 — Vulkan drop order bug (session.rs field reordering):**
Rust drops struct fields top-to-bottom; `VulkanSession` had `instance` first. `vkDestroyInstance`
ran before `vkDestroyDevice` → STATUS_ACCESS_VIOLATION in `cdylib_load` test. Fixed by declaring
`instance` last. Rule: struct fields must be in reverse-creation order (see decisions file).

**D-S13-02 — Guard test update rule:**
When flipping an op from `Staged` to `Live`, search for every test that hard-codes the op name
against its old status. All three registry/elementwise guard tests had to change simultaneously.

**D-S13-03 — ORT wire seam:**
`compile_impl` + `compute_impl` are wired (ep.rs + session.rs). Session lifecycle gap: fresh
VulkanSession created per compile_impl call; needs persistent Arc<VulkanSession> in VulkanEp.
Trinity's ORT differential test is the M0 confirmation gate.

**State at end of session:**
- cargo ci: ✅ GREEN (258 lib tests)
- Add: ✅ Live, registered, translatable
- Guard tests: ✅ correct in both shader-present and shader-absent build modes
- compile_impl / compute_impl wire: ✅ implemented, session lifecycle gap pending
- docs/ENGINE.md: updated status table and Morpheus note
- decisions/inbox/switch-engine-seams.md: D-S13-01 through D-S13-03 appended


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

---

## Session 15 — SkipSimplifiedLayerNormalization kernel; QGemv compile error fix (2026-07-29T20:26:56-07:00)

**Coordinator task:** Implement `SkipSimplifiedLayerNormalization` — one of the three ops blocking
a real model (Phi-3.5-mini) from running on GPU (64 of 366 nodes). Mouse taking `MatMulNBits`.

**What was done:**

1. **`shaders/glsl/skip_simplified_layer_norm_f32.comp`** — new direct GLSL shader.
   - Fused residual-add (the "Skip") + RMSNorm in one dispatch, one workgroup per row
   - Three-pass: accumulate sum-of-squares while writing out3 → tree reduce → normalize
   - Pass 3 re-reads inputs (not out3) to avoid read-after-write on a `writeonly` buffer
   - LOCAL_SIZE_X = 256 (spec constant 0); shared memory = 256 × 4 bytes = 1 KiB
   - Push constants: batch_count, hidden_size, eps_bits (float bits), pad — 16 bytes total
   - Five bindings: hidden (readonly), skip (readonly), gamma (readonly), out0 (writeonly), out3
   - Cross-platform safe: 1 KiB shared memory is within the 16 KiB portability floor (§7.2)
   - Uses direct-file path: build.rs scans `shaders/glsl/` and picks it up without -D defines

2. **`ops/common/templates.rs`** — added `skip_norm()` translate handler:
   - Validates ≥3 inputs and consistent dtype
   - Extracts `batch_count` = product of dims[0..rank-2], `hidden_size` = last dim
   - `epsilon` from node attribute (default 1e-5, ONNX schema default)
   - Binds slot-3 output OR allocates temp buffer when slot 3 is absent (rare, but correct)
   - 6 unit tests: shader name, push constants, workgroups, slot-3 absent, default epsilon, rank-1

3. **`ops/elementwise.rs`** — fixed pre-existing compile error from Mouse adding `Template::QGemv`
   without updating the exhaustive match in `shader_rows_have_an_arity_their_predicate_agrees_with`.
   Added `Template::QGemv => {}` arm with comment.

4. **`cargo ci`** — ✅ GREEN after `--fix` for rustfmt. All 315 lib tests pass.

**What Mouse must do to flip SkipSimplifiedLayerNormalization to Live (documented D-S15-01):**
- `norm.rs` SkipSimplifiedLayerNormalization (Ms domain) row: `status: Live`, `translate: templates::skip_norm`
- Update `both_norms_share_one_blocker` test (currently asserts ALL 4 rows Staged)
- No `shader_variants.txt` regeneration needed (direct kernel not in manifest system)

**Key design decisions recorded (D-S15-01, D-S15-02):**
- Direct shader (not template-driven) because the norm kernel has a unique push-constant layout
  and the two outputs are not compatible with the EW template interface
- `kernel!(None)` in the op table row; translate handler names the shader stem directly
- `skip_norm_f32_shader_exists_on_disk` test compensates for the manifest system not covering it

**State at end of session:**
- `cargo ci`: ✅ GREEN (315 lib tests)
- SkipSimplifiedLayerNorm shader: ✅ written and glslc-verified
- translate handler: ✅ written and tested
- norm.rs flip to Live: ⏳ waiting on Mouse (D-S15-01)
- QGemv compile error: ✅ fixed (pre-existing, Mouse's enum addition)

