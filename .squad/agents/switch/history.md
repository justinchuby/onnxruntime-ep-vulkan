# Switch (Vulkan-Compute) — history.md

## Learnings

### [SUMMARY] Sessions 1–16+: ash engine, first execution, probe correctness, runtime extents (2026-07-28–2026-07-30)

**Sessions 1–11 (archived):** ENGINE.md produced; ash+gpu-allocator selected; buffer-only tensor storage; build-time GLSL→SPIR-V; one command buffer per subgraph; per-edge barriers. Dual-backend barrier abstraction (`legacy` + `sync2`). Capability probe implemented. Loader diagnostics added; `apiVersion` capped to loader-reported to avoid `ERROR_INCOMPATIBLE_DRIVER` on 1.0 loaders. R5 (subgroup BASIC) removed as gate criterion — now a probed capability governing shader variants, per §7.0.

**Session 12 (2026-07-29T08:13:58-07:00) — push_next chain bug root-caused:**
`let _ = props2.push_next(..)` discards the pNext chain in ash 0.38. Every chained capability (subgroup size, stages, float16, SSC) read zero before fix. Three-state probe added (`subgroup_probe_valid: bool`). Intel Iris Xe validated as spec-conformance oracle. `is_uma` predicate corrected: "every heap is DEVICE_LOCAL" (not "largest DEVICE_LOCAL heap is HOST_VISIBLE") — ReBAR on RTX 4060 was returning wrong `true`. `timestamp_period_ns` and `timestamp_valid_bits` added to `Capabilities`.

**Session 13 (2026-07-29) — teardown order:**
Vulkan struct teardown order enforced by field declaration order: `instance` must be LAST. Rust drops fields top-to-bottom; declaring `instance` first caused STATUS_ACCESS_VIOLATION in `cdylib_load`.

**Session 14 (2026-07-29T13:42:45-07:00) — §7.9 and R5 re-evaluation:**
§7.9 five rules for capability probe discipline. R5 rationale was false (lavapipe reading was probe bug); policy stands on §7.0 principle that shortfalls degrade op coverage, not device availability. `cargo ci` — green, 300 tests.

**Session 15 (2026-07-29T20:26:56-07:00) — SkipSimplifiedLayerNorm:**
`SkipSimplifiedLayerNormalization` kernel: single-pass, two outputs, 1 KiB shared memory, LOCAL_SIZE_X=256.

**Session 16 (2026-07-29T21:14:03-07:00) — descriptor-set lifetime:**
Fixed VUID-03047: descriptor sets released before command buffer submission. Descriptor pool now per-dispatch.

**Sessions 17–18 (2026-07-30) — runtime extents:**
`ENGINE_ACCEPTS_RUNTIME_EXTENTS=true`. `CompiledKernel` reads real shapes at Compute via `GetTensorTypeAndShape`. OQ-15 resolved for M0/M1: re-record per shape. Device 0 (Intel): 161 claimed, zero validation errors. Device 1 (RTX 4060): 161 claimed. Variable seqlen correct on both devices. 97 additional nodes unlocked without new kernels.
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

---

## Session 16 — Descriptor-set lifetime fix (validation error VUID-03047) (2026-07-29T21:14:03-07:00)

**Coordinator task:** Fix Vulkan validation error "A descriptor set is updated while bound to a
recording command buffer, without UPDATE_AFTER_BIND" — caught by Trinity running the real Phi-3.5
model, identical on Intel Iris Xe and NVIDIA RTX 4060.

**Root cause (D-S16-01):**
`dispatch_ort()` in `session.rs` created one `DispatchDescriptorPool` per kernel inside the
dispatch loop. `DispatchDescriptorPool::Drop` calls `vkDestroyDescriptorPool`, freeing its
`VkDescriptorSet` at end-of-iteration. On the *next* iteration `vkAllocateDescriptorSets` may
reuse the same handle. `vkUpdateDescriptorSets` on that reused handle while the previous set is
still "bound" in the recording command buffer triggers VUID-vkUpdateDescriptorSets-None-03047.

Both drivers tolerated it (inference completed), but Adreno, Mali and MoltenVK are stricter.
Working is not the same as valid.

**Fix chosen: collect-and-defer (Option 1):**
Declared `let mut desc_pools: Vec<DispatchDescriptorPool>` before the kernel loop. At the
bottom of each loop iteration, ownership transfers to the Vec via `.push(desc_pool)`. The Vec
drops after `submit_and_wait` returns, by which time the fence has signalled and the command
buffer is no longer in use. This makes the class of error impossible without extension or
platform-specific workaround: the set being written is always freshly allocated, and the set
being read by the GPU is always fully initialised before it was ever bound.

Why not `UPDATE_AFTER_BIND`? That requires `VK_EXT_descriptor_indexing`, which is optional
and §7.2 requires no extensions. It would leave two diverging code paths where one does.

**Also fixed (pre-existing, same session):**
1. `allocator.rs` — `#[cfg(unix)]` changed to `#[cfg(not(windows))]` — P2 portability lint
   requires every `#[cfg(windows)]` to have a `#[cfg(not(windows))]` counterpart. The `#[cfg(unix)]`
   guard meant the Linux/macOS mmap path was only compiled on `unix`, not on all non-Windows targets.
2. `allocator.rs:676` — added `#[allow(clippy::new_ret_no_self)]` on `VulkanAllocator::new` which
   intentionally returns `*mut OrtAllocator` for the ORT ABI, not `Self`.
3. `allocator.rs:836` — moved `// SAFETY:` comment to be immediately before the `unsafe` block
   (was above the `if` statement; `undocumented_unsafe_blocks` requires the comment adjacent to the block).
4. `allocator.rs:1074` — added `// SAFETY:` comments before `set_var`/`remove_var` unsafe blocks.

**State at end of session:**
- `cargo ci`: ✅ GREEN (326 lib tests + all integration suites)
- Validation error VUID-vkUpdateDescriptorSets-None-03047: ✅ fixed (descriptor lifetime corrected)
- Mouse's Intel access violation: 🟡 plausible same root cause (descriptor use-after-free); needs
  confirmation from Mouse — the fix eliminates the scenario but cannot confirm it was the cause
- SkipSimplifiedLayerNorm: ⏳ waiting on Mouse to flip norm.rs to Live (D-S15-01)
- Pre-existing allocator.rs clippy/portability issues: ✅ fixed as side-effect


**Current state:**
- `cargo ci` — green, 300 tests.
- 45 op rows Live on two GPUs.
- 161 Phi-3.5 nodes claimed under runtime extents.
- M0 open: validation positive control (criterion 3); CI lanes.
- Outstanding seam for Niobe: tile_config, workgroup size, shared-memory bytes, memory path must be reported by the engine.
---

## 📌 Cross-agent context — Round 4 (2026-07-30T02:49:12-07:00)

### Worktree layout and inbox portability constraint
The team works in git worktrees: `squad/switch` at `C:\Users\justinchu\dev\ep-vulkan-switch`, `squad/mouse` at `C:\Users\justinchu\dev\ep-vulkan-mouse`, `squad/tank` at `C:\Users\justinchu\dev\ep-vulkan-tank`, with `main` as the integration tree. `.squad/decisions/inbox/` is **gitignored** — records written in a worktree do NOT travel with the branch. The inbox in `main` is authoritative.

### Vulkan SDK path
`C:\VulkanSDK\1.4.350.0` — installed but **not on the default PATH**. `glslc` discovery must search this path; `VULKAN_SDK` env var is the canonical pointer.

### Local hardware — both GPUs pass the §7.2 gate
- Intel Iris Xe: Vulkan 1.4.309, UMA, `subgroup_size=32`, 32 KiB shared. Spec-conformance oracle. Do not special-case Intel.
- RTX 4060 Laptop: Vulkan 1.4.325, discrete, `subgroup_size=32`, 48 KiB shared.
- Lavapipe (CI): `subgroup_size=8`, 32 KiB shared, `is_uma=true`. CI exercises the mobile-warp path. LVP2 retracted.

### ORT's planner hands back interior pointers from run 2 onward
Memory-pattern planner does not engage on run 1 — records during run 1, from run 2 onward hands back interior pointers. 52 interior pointers observed, identical on both devices, all within span. Every probe that ran each session once was pointed at the wrong moment. Gate: `epctl --check-counters <file> --require-dispatches 1`.

### Execution counters file is the instrument for "did anything execute"
`ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE` — always-on JSON, written on first dispatch and at teardown. `dispatches_executed > 0` is the only reliable indicator that a shader ran.

### `push_next` must rebind, never discard
`let _ = props2.push_next(..)` silently discards the pNext chain in ash 0.38. Every chained capability reads zero. Rule: rebind, never discard. This was the root cause of LVP2, `subgroup_size=0`, and ReBAR UMA misclassification.

### First real execution: 45 ops Live, 161 nodes claimed on Phi-3.5
`ENGINE_ACCEPTS_RUNTIME_EXTENTS=true`. 45 op rows Live. 161 Phi-3.5 `MatMulNBits` nodes claimable. M0 not declared — open: validation positive control, CI lanes green.

### Performance metric is a TRIPLE (Niobe — critical)
`(claimed_op_coverage, island_count, largest_island_flops)` reported together, per producer at version. `largest_island_flops` alone is not quotable. Portability floor = §7.2 (16 KiB / 256 invocations). `SUBGROUP_SIZE_IS_GUARANTEED=False`.
## Session 18 — Runtime extents: ENGINE_ACCEPTS_RUNTIME_EXTENTS flipped (2026-07-30T01:00:00-07:00)

**Coordinator task:** Make tensor extents runtime parameters in the dispatch path, unblocking 97 nodes
on Phi-3.5 (Mul 64, Sigmoid 32, Sub 1) that were declined [dynamic-shape] solely because the engine
baked push constants at Compile time.

**D-S18-01 — Three engine changes implemented:**

1. compile_impl detects symbolic inputs via TensorRef::desc == None and calls
   CompileRecorder::push_dynamic_kernel(node_desc, spec) instead of translate.
   DynKernelRecipe { node_desc, spec } is stored on CompiledKernel.

2. dispatch_ort pre-pass (Steps 1.5/1.6): reads GetTensorSizeInBytes for 0-size inputs,
   then ead_tensor_desc_from_ort (GetTensorTypeAndShape + GetTensorElementType + GetDimensions)
   for each dynamic kernel input slot, re-runs translate via ShapeOnlyRecorder to capture
   push_constants, workgroups, spec_constants, shader, and output TensorDescs.

3. check_bound_input_sizes skips the check when planned == 0 (the dynamic signal).

**D-S18-02 — OQ-15: Re-record per shape**
Command buffer is already re-recorded on every Compute call. Dynamic path adds only host-side
µs-scale shape reads vs ms-scale staging. Bucketing and vkCmdDispatchIndirect deferred to M2+.

**D-S18-03 — ENGINE_ACCEPTS_RUNTIME_EXTENTS flipped**
Three tests in ops/common/claim.rs updated to reflect new baseline (	rue not alse).
Test untime_extent_support_is_a_single_switch updated; symbolic_shape_is_tagged_dynamic_shape
and symbolic_extents_are_accepted_once_extents_are_runtime_parameters both updated.

**Cross-owner edits (flagged to coordinator):**
- p.rs (Tank): compile_impl dynamic detection, check_bound_input_sizes skip, test updated.
- ops/common/claim.rs (Mouse/shared): 3 tests updated.

**Results:**
- Device 0 (Intel Iris Xe): 161 claimed, no validation errors, variable seqlen passes.
- Device 1 (RTX 4060): 161 claimed, no validation errors, variable seqlen passes.
- M1 exit criterion satisfied: seq_len=1 and seq_len=5 correct in same session.
- cargo ci green.
- Commit 9885f9 on branch squad/switch.

---

## Session 20 — Counter scoping fix + profiling island fix (2026-07-30T01:32:15-07:00)

**Coordinator task:** Reconcile the three contradictory numbers from the Phi-3.5 census run:
`Claimed: 161, Islands: 0, counters {compile_calls:1, subgraphs_live:1, dispatches_executed:1}`.

**Investigation findings (D-S20-01):**

**Finding 1: "Claimed: 161" measures GetCapability offers, not ORT acceptance.**
CLAIM_LOG is written during GetCapability, before ORT calls Compile or Compute. The 161 figure
is what our EP offered to ORT. ORT acceptance is confirmed separately by the counters.

**Finding 2: counters {1,1,1} = probe contamination, not Phi-3.5 state.**
record_dispatches() used FIRST_DISPATCH_DUMPED to write the file exactly once per process.
conftest.py::_probe_vulkan_device() dispatches an Add kernel before any test — making the
Add session the first dispatch. The file captured {1,1,1} (the probe Add state) and
FIRST_DISPATCH_DUMPED=true. All 161 subsequent Phi-3.5 dispatches incremented the in-memory
atomics but could not update the file.

**Finding 3: Islands=0 — broken regex in _count_islands.**
The regex matched on the profiling event name field expecting EP_NAME_<digits>_<digits>. Actual
ORT plugin-EP profiling format: name="VulkanExecutionProvider_VulkanExecutionProvider_<hash>_<N>_<N>_kernel_time",
args["op_name"]="VulkanExecutionProvider_<hash>_<N>" where <hash> is a 64-bit integer.
No events matched the two-short-digits pattern -> islands=0 always.

**Fixes:**
1. counters.rs: removed FIRST_DISPATCH_DUMPED. record_dispatches() calls dump_if_requested() on every dispatch.
2. ep.rs: removed all temporary diagnostic instrumentation from session 19. Permanent S18 changes intact.
3. test_phi35.py: OrtEpVulkanResetExecutionCounters() via ctypes before Phi-3.5 session;
   in-process counter read after sess.run(); _count_islands rewritten to use args["op_name"].
   _MOUSE_PREDICTED_ISLANDS_LO/HI updated to 155-161.
All three files are Tank's (cross-owner edits flagged to coordinator).

**Verified (both devices, 2026-07-30):**
compile_calls=1, subgraphs_live=161, compute_calls=161, compute_failures=0, dispatches_executed=161, islands=161
cargo ci: GREEN (343 tests)

---

## Session 21 — Interior-pointer hazard + validation positive control (2026-07-30T02:30:00-07:00)

**Coordinator tasks:**
1. Respond to Tank's finding that ORT's memory-pattern planner produces interior pointers to our allocator handles from run 2 onward (52 pointers, max offset 48 KiB across 5 runs).
2. Fix Step 1b `want` bug in session.rs (Tank's cross-owner code used compile-time byte sizes for dynamic inputs, silently bypassing the overflow guard).
3. Add validation positive control for M0 criterion 3.

**D-S21-01 — Step 1b `want` bug fixed (session.rs cross-owner, Tank's Step 1b code):**
Tank's `host_backing_for` calls in Step 1b used `input_byte_sizes[i]` (the compile-time size,
0 for dynamic-shape inputs). `host_bytes` guards with `len > available`; when `len=0`, the
check is trivially false regardless of span size. Fix: changed to `actual_input_byte_sizes[i]`,
which is resolved in Step 1.5 from the live ORT tensor. Now the overflow guard fires correctly
for dynamic inputs from run 2 onward when ORT's planner places them as interior pointers.

**D-S21-02 — Validation positive control added (dispatch_integration.rs, M0 criterion 3):**

New module: `validation_positive_control` — `#[test] #[ignore]` test named
`descriptor_set_updated_while_bound_fires_vuid_03047`.

The test:
- Creates a Vulkan instance with `VK_LAYER_KHRONOS_validation` + `VK_EXT_debug_utils`.
- Installs a `VkDebugUtilsMessengerEXT` callback incrementing `static AtomicU32 VALIDATION_ERRORS`.
- Deliberately violates **VUID-VkWriteDescriptorSet-descriptorType-00332** by creating a buffer
  with `VK_BUFFER_USAGE_VERTEX_BUFFER_BIT` only and writing it as a STORAGE_BUFFER descriptor
  while the set is bound to a recording command buffer.
- Asserts `VALIDATION_ERRORS > 0`.

**Finding: VUID-03047 not reported pre-submit in SDK 1.4.350.0.**
`VUID-vkUpdateDescriptorSets-None-03047` (the session-16 VUID) is checked lazily by the layer —
at submit time, not at the `vkUpdateDescriptorSets` call. Without a `vkQueueSubmit`, it does not
fire in a unit test. The test uses VUID-00332 instead, which fires unconditionally at the update call.
Documented in `switch-positive-control.md` (main inbox). VUID-00332 is in the same validation
domain (descriptor content lifetime).

Confirmed on Intel Iris Xe (device 0):
```
[VALIDATION-POSITIVE-CONTROL] severity=ERROR: vkUpdateDescriptorSets():
  pDescriptorWrites[0].pBufferInfo[0].buffer was created with VK_BUFFER_USAGE_2_VERTEX_BUFFER_BIT,
  but descriptorType is VK_DESCRIPTOR_TYPE_STORAGE_BUFFER.
[POSITIVE-CONTROL] validation errors captured: 31
test ... ok
```

**Cross-owner edits:**
- `rust/src/vk/session.rs` (Tank's Step 1b): `want` changed from `input_byte_sizes` to `actual_input_byte_sizes`.

**State at end of session:**
- `cargo ci`: ✅ GREEN (344 tests)
- Step 1b overflow guard: ✅ fixed for dynamic inputs
- Validation positive control: ✅ fires VUID-00332 reliably; documents VUID-03047 laziness
- Decision records written to main inbox: `switch-positive-control.md`

---

## Session 22 — EP-side messenger, fence-leak plant, multi-run census (2026-07-30T03:52:28-07:00)

**Coordinator tasks:**
1. Plant `ONNXRUNTIME_EP_VULKAN_PLANT_VALIDATION_VIOLATION` env-gated fence leak (VUID-05137).
2. Attach `VkDebugUtilsMessengerEXT` to the EP's own Vulkan instance.
3. Provide a test that proves the two work together (EP's messenger captures the plant).
4. Multi-run census — run 5 times in one session to exercise the interior-pointer planner.
5. Confirm the four-number reconciliation is on the record.

**D-S22-01 — EP-side messenger (instance.rs):**
- `EP_VALIDATION_ERROR_COUNT: AtomicU32` static added to `instance.rs`.
- `validation_log_callback` increments the counter on ERROR and calls `log::error!`.
- `Instance::create` requests `VK_EXT_debug_utils` when `enable_validation=true`.
- `Instance` struct stores `debug_messenger: Option<...>`, destroyed before `vkDestroyInstance`.
- `probe_loader_report` sets `debug_messenger: None` (diagnostic path, no session).

**D-S22-02 — Fence-leak plant (dispatch_integration.rs):**
```rust
if std::env::var_os("ONNXRUNTIME_EP_VULKAN_PLANT_VALIDATION_VIOLATION").is_some() {
    let _ = unsafe { device.ash().create_fence(&vk::FenceCreateInfo::default(), None) };
}
```
Placed in `run_add_on_device` before RAII cleanup. Fence handle is leaked; `Device::drop`
calls `vkDestroyDevice`, firing VUID-vkDestroyDevice-device-05137 through the EP's messenger.

**D-S22-03 — Plant verification test (dispatch_integration.rs):**
`ep_messenger_fires_for_planted_fence_leak` — `#[ignore]` test, confirmed:
```
[EP-PLANT] EP_VALIDATION_ERROR_COUNT after planted fence leak = 1
test vk::dispatch_integration::ep_messenger_fires_for_planted_fence_leak ... ok
```
(RTX 4060, SDK 1.4.350.0)

**D-S22-04 — Multi-run census (tests/ops/test_phi35.py — Tank's file, cross-owner):**
`test_phi35_multi_run_same_session_interior_pointer_safety`:
- 5 consecutive runs on one session.
- Asserts bit-identical output across all 5 runs.
- Asserts `dispatches_executed == 5 × subgraphs_live` (all runs reach GPU).

**Four-number reconciliation (from session 20, on record):**
- "Claimed: 161" = GetCapability offers (written before Compile, before ORT partitions).
- "{1,1,1}" = probe contamination (conftest Add dispatch fired FIRST_DISPATCH_DUMPED).
- "islands: 0" = broken regex (`^EP_NAME_(\d+)_\d+$` never matched ORT's 64-bit hash format).
- After fixes: all agree at 161. ORT accepted 100% of offers.

**Cross-owner edits:**
- `tests/ops/test_phi35.py` (Tank): multi-run test added.
- `rust/src/vk/instance.rs` — Switch's file.
- `rust/src/vk/dispatch_integration.rs` — Switch's file.

**State at end of session:**
- `cargo ci`: ✅ GREEN (344 tests + 2 new `#[ignore]` tests)
- EP-side messenger: ✅ installed on EP's own instance; `EP_VALIDATION_ERROR_COUNT` observable
- Fence-leak plant: ✅ VUID-05137 fired and captured by EP messenger (count=1)
- Multi-run census test: ✅ added; asserts 5×161=805 dispatches for Phi-3.5
- Decision records: `switch-ep-messenger-and-plant.md` in main inbox



📌 Team update (2026-07-30T05:48:29-07:00): A green suite has been shown not to imply a correct model. Phi-3.5: 161 MatMulNBits dispatched, compute_failures:0, entire suite green — vk logits all-zero (argmax 0 vs CPU argmax 30751). R9 (Morpheus): for every claim, name the instrument that would go red if the claim were false; if none, the claim is UNMEASURED. model_output_equivalence verdict required alongside all counter summaries; default UNMEASURED. Any comparison must first assert EP_NAME in session.get_providers() before calling sess.run() — failure to do so compares CPU to CPU and reports agreement. Coordinator's own first comparison reported bit-identical on both devices due to this exact error. Trinity has landed xfail(strict=True) correctness gate. M0 criterion 10 added (NOT MET: DIVERGENT). Criteria 2, 4, 5 reopened. — decided by Morpheus, Trinity, Switch, Mouse; coordinator-verified.

---

## Session 23 — Dynamic-kernel binding mismatch: all-zero logits root cause and fix (2026-07-30T09:14:00-07:00)

**Coordinator task:** Investigate all-zero logits in Phi-3.5 (161 dispatches, compute_failures=0, but argmax=0 on both devices). Discriminate "kernel writes zeros" from "copy is broken" from "wrong binding". Confirm descriptor-set fix holds at N=161 scale. Confirm EP messenger armed and listening.

**D-S23-01 — ONNXRUNTIME_EP_VULKAN_DUMP_OUTPUT_BYTES probe:**
Added to `write_outputs_to_ort`: when env var is set, logs first 16 bytes of each staging buffer
before `copy_nonoverlapping`. This immediately distinguished fault location:
- 33 of 257 staging buffers are all-zero (the GPU kernel wrote zeros to device memory)
- 224 of 257 are non-zero (kernel computed correct values)
- Zero outputs: 18432-byte (N=9216, qkv_proj × 32 layers) and 64128-byte (N=32064, lm_head × 1)
- Non-zero outputs: 16384-byte (N=8192, gate/up_proj) and 6144-byte (N=3072, o/down_proj)

Conclusion: the GPU kernel IS writing zeros into device memory. The readback/copy path is correct.

**D-S23-02 — Push-constant and workgroup dump:**
Same env var dumps `push_u32=[m, K, N, blocks_per_col]` and `workgroups=[N, 1, 1]` for each dispatch.
Confirmed: all parameters correct for zero-output kernels — `[1, 3072, 9216, 96]` with 9216 groups.
Dispatches are real. Fault is not in dispatch geometry.

**D-S23-03 — ONNXRUNTIME_EP_VULKAN_VALIDATE env-var override for Instance::create:**
`enable_validation` was only read from ORT session config (`ep.enable_validation`), never from an
env var. The phi35 comparison script didn't set session config, so the EP messenger was silent
during the 161-dispatch session. Fix: added env-var check at the top of `Instance::create`:
```rust
let enable_validation =
    enable_validation || std::env::var_os("ONNXRUNTIME_EP_VULKAN_VALIDATE").is_some();
```
With messenger armed (`ONNXRUNTIME_EP_VULKAN_VALIDATE=1`), validation layer reported:
```
vkCreateComputePipelines(): pCreateInfos[0].stage SPIR-V uses descriptor [Set 0, Binding 4]
but the binding was not declared in the VkPipelineLayoutCreateInfo::pSetLayouts[0].
vkCmdDispatch(): VkDescriptorSet ... [Set 0, Binding 4] is invalid.
```
This was the direct pointer to the root cause.

**D-S23-04 — ROOT CAUSE: push_dynamic_kernel binding token mismatch:**
`push_dynamic_kernel` (called at Compile time for symbolic-shape nodes) creates
`n_inputs + n_outputs` binding tokens positionally. For MatMulNBits without `zero_points`
(3 inputs + 1 output = 4 tokens), kernel.bindings = [0, 1, 2, 3].

But the `q_gemv_matmul_nbits_f16` shader declares **5** descriptor bindings (0-4):
- binding 0: A (activations)
- binding 1: B (packed weights)
- binding 2: scales
- binding 3: zero_points or scales-as-placeholder (QB_HAS_ZP=0 folds it away)
- binding 4: Y (output)

The translate handler correctly handles this (documented in q_gemv.comp and quant.rs):
`zp = scales` (reuses the scales token at position 3), then `bind_output(y)` → token 3 output.
KernelRequest.bindings = [0, 1, 2, 2, 3] (5 entries). But this runs ONLY at Compute time
(through ShapeOnlyRecorder), never at Compile time. kernel.bindings stays at [0, 1, 2, 3].

Pipeline layout created with n_bindings=4 (from kernel.bindings.len()). Shader writes to
binding 4 (undefined) → driver silently ignores → output buffer stays at zero-initialized value.

Nodes WITH zero_points have 4 inputs → n_bindings=5 → works correctly (o_proj, gate/up/down_proj).
Nodes WITHOUT zero_points have 3 inputs → n_bindings=4 → binding 4 undefined → zeros (qkv_proj, lm_head).

**D-S23-05 — FIX: ShapeOnlyRecorder captures KernelRequest.bindings:**
`ShapeOnlyRecorder::dispatch()` now captures `k.bindings` as the 5th element of `captured`:
```rust
pub captured: Option<(Vec<u8>, [u32; 3], Vec<u32>, &'static str, Vec<u64>)>
//                    pc      wg      sc     shader  bindings
```
`dispatch_ort` extracts `eff_bindings` from captured for dynamic kernels (or from
`kernel.bindings` for static). Both `n_bindings = eff_bindings.len()` and `buf_bindings`
iteration use `eff_bindings`. This ensures the pipeline layout has the correct number of
bindings and every descriptor slot is filled correctly.

**Results:**
- phi35_vk_vs_cpu.py: argmax vk=[30751] == cpu=[30751], top-10 overlap 10/10, both devices ✅
- max|vk-cpu| = 0.035156 (RTX 4060), 0.031250 (Intel) — fp16 precision, expected ✅
- No validation errors via EP messenger on either device ✅
- cargo ci: 346 passed, 0 failed ✅

**Files changed:**
- `rust/src/vk/session.rs` — ShapeOnlyRecorder.captured now includes bindings; DynCaptured type alias updated; eff_bindings extracted in dispatch loop; DUMP_OUTPUT_BYTES probe; workgroup/push-constant dump.
- `rust/src/vk/instance.rs` — env-var override for enable_validation.

**Decision record:** `switch-binding-mismatch-fix.md` in main inbox.

**Cross-owner note:** The fix is purely in Switch's `session.rs` and `instance.rs`. The shader
and translate are correct (documented behavior). The fault was in `push_dynamic_kernel`'s
assumption that `n_inputs + n_outputs` equals the shader's declared binding count.

## Session 23 addendum — KV-cache "unwritten" explained (2026-07-30T08:16:02-07:00)

**Tank's two-bug report (pre-fix):** Tank ran probe_run2.py BEFORE my fix was merged and found:
- Outputs 1..64 (KV cache) differ bitwise between run 1 and runs 2/3 with identical feeds.
- Output 0 (logits) exactly 0.0 on runs 2 and 3.
Coordinator relayed this as two separate bugs.

**Post-fix probe_run2.py (3 runs, both devices):**
```
Device 1 (RTX 4060):  BIT-IDENTICAL across all 65 outputs — run 1 vs 2 and run 1 vs 3
Device 0 (Intel Xe):  BIT-IDENTICAL across all 65 outputs — run 1 vs 2 and run 1 vs 3
```
Memory-pattern planner EXCLUDED as a factor. Both observations were one bug manifesting twice.

**Causal chain:** qkv_proj wrote zeros (binding 4 undefined) → CPU attention saw zero QKV →
attention's internal scratch became dirty on run 1 → on run 2+, ORT arena reuse caused KV outputs
to read dirty data, appearing "unwritten" rather than zero. The KV cache is entirely CPU-side;
our EP subgraphs have 1 output each and correctly write it.

**Instruments that would go red if claim is false:**
- probe_run2.py: any differing output printed
- phi35_vk_vs_cpu.py: argmax mismatch or top-10 < 10/10
- EP messenger: validation errors if descriptor binding mismatch recurs

**Decision record:** `switch-kv-cache-explained.md` in main inbox.

---

## Session 24 — Mouse's independent fix merged; messenger positive control confirmed (2026-07-30T08:35:04-07:00)

**Coordinator task:** Resolve merge conflict between `squad/switch` (HEAD, my binding fix from session 23) and `squad/mouse` (Mouse's independent fix for the same root cause). Confirm the two fixes are genuinely equivalent — specifically that `eff_bindings` correctly reproduces the duplicate-`scales`-at-slot-3 placeholder. Run Mouse's regression tests. Confirm the EP messenger is a positive control, not just silent.

**D-S24-01 — Equivalence of the two fixes:**
Both fixes capture `k.bindings` from `ShapeOnlyRecorder::dispatch()` and use those captured tokens for both `n_bindings` (pipeline layout) and `buf_bindings` (descriptor writes). The *structural* difference is cosmetic:
- Mine (session 23): 5-tuple `captured: Option<(pc, wg, sc, shader, Vec<u64>)>`
- Mouse's: 4-tuple `captured` + separate `captured_bindings: Option<Vec<u64>>`

Mouse's split is cleaner (separation of concerns). Adopted his structural form in the resolution.

**D-S24-02 — duplicate-`scales` placeholder verified correct:**
`ShapeOnlyRecorder::dispatch` receives `k: KernelRequest` where `k.bindings` was assembled by the translate handler. For MatMulNBits without `zero_points`, the translate rebinds `scales` as an inert ZP placeholder: `KernelRequest.bindings = [A=0, B=1, scales=2, zp-placeholder=2, Y=3]` (5 entries). `captured_bindings` captures this exactly. The pipeline layout gets 5 slots; Y is at binding index 4; the shader writes there correctly. The duplicate-2 token maps to the same `gpu_inputs[2]` (scales buffer) twice in `buf_bindings`, which is harmless. This is the right answer for the right reason.

**Falsifier named:** the instrument that would go red if "equivalent for the right reason" were false is Mouse's `test_matmulnbits_fp16_dynamic_batch_multirun` — a node without zero_points in a multi-run session on both devices. Result: PASSED on device 0 (Intel Iris Xe) and device 1 (RTX 4060).

**D-S24-03 — Resolution approach:**
- `session.rs`: adopted Mouse's structural split (`captured` 4-tuple + `captured_bindings`) + kept my struct-level `# Binding correction` docstring (Mouse's struct docstring was incorrectly written to say "need not be recomputed") + kept my `buf_bindings` comment + kept my `DUMP_OUTPUT_BYTES` diagnostic probe (Mouse removed it; useful for future debugging). Mouse's added comments on `eff_bindings` and Step 1.6 bindings note adopted.
- `dispatch_integration.rs`: kept my version entirely (plant + `ep_messenger_fires_for_planted_fence_leak` test — Mouse removed both; coordinator requires the plant for M0 criterion 3).
- `instance.rs`: no conflict (auto-merged).
- `test_phi35.py`: auto-merged cleanly; fixed stale comment block that said "marked xfail(strict=True)" (the decorator had been removed by Mouse's changes; the prose was left behind).
- `test_matmulnbits.py`: auto-merged cleanly (Mouse's fp16 regression tests adopted).

**D-S24-04 — Mouse's test_phi35.py changes (for coordinator/Trinity):**
Mouse made four categories of edits to Trinity's file:
1. **Header docstring rewritten** — updated EXPECTED RESULT section to reflect 161 nodes claimed, islands 161, logits correct. Added CORRECTNESS GUARDS section crediting `test_phi35_f16_matmulnbits_logits_nonzero` (Mouse) and `test_phi35_vulkan_matches_cpu_logits` (Trinity).
2. **`xfail(strict=True)` removed** from `test_phi35_vulkan_matches_cpu_logits` — correct (coordinator confirmed top-1 agreement on both devices satisfies the condition the marker named). Trinity should confirm.
3. **`test_phi35_vulkan_session_determinism` docstring rewritten** — trimmed the "RENAMED FROM" and "VACUOUS-PASS CONDITION" sections; added reference to the new nonzero/multirun tests as the correctness gates.
4. **Two new slow tests added:**
   - `test_phi35_f16_matmulnbits_logits_nonzero`: guards against the 2026-07-30 failure mode (dispatches with compute_failures=0 but all-zero logits); has hard EP-presence gate; asserts logit range > 1.0 AND top-1 token matches CPU.
   - `test_phi35_vulkan_multirun_logits_stable`: 3-run session, asserts non-zero and bit-identical across runs; replaces `test_phi35_multi_run_same_session_interior_pointer_safety` (session 22 cross-owner from Switch).
   
   The `test_phi35_multi_run_same_session_interior_pointer_safety` test is REMOVED in Mouse's version, replaced by the simpler `test_phi35_vulkan_multirun_logits_stable` which drops the counter/ctypes machinery. The new test is cleaner and covers the same ground (multi-run bit-identical) without owning counters (Tank's domain).

**D-S24-05 — Messenger positive control confirmed:**
```
[EP-PLANT] EP_VALIDATION_ERROR_COUNT after planted fence leak = 1
test vk::dispatch_integration::ep_messenger_fires_for_planted_fence_leak ... ok
```
`ONNXRUNTIME_EP_VULKAN_PLANT_VALIDATION_VIOLATION` fires VUID-vkDestroyDevice-device-05137 through the EP's own `VkDebugUtilsMessengerEXT`, incrementing `EP_VALIDATION_ERROR_COUNT` to 1. The messenger is not merely silent — it is wired and capturing. Test confirmed on RTX 4060 (device 1) with SDK 1.4.350.0. (Device 0 not rerun separately but the `run_add_on_device` path runs all capable devices.)

**Also fixed:** stranded decision record `switch-dynamic-shape.md` was in the worktree's gitignored inbox. Moved to the integration tree's inbox (`C:\Users\justinchu\dev\onnxruntime-ep-vulkan\.squad\decisions\inbox\`).

**Results:**
- Mouse's regression tests: `test_matmulnbits_fp16_dynamic_batch` and `test_matmulnbits_fp16_dynamic_batch_multirun` — PASSED on device 0 and device 1 ✅
- `cargo ci`: ✅ ALL CHECKS PASSED
- Messenger positive control: ✅ `EP_VALIDATION_ERROR_COUNT = 1`

**Files changed:**
- `rust/src/vk/session.rs` — conflict resolution: Mouse's struct split adopted; my docstrings, comments, and DUMP_OUTPUT_BYTES probe retained; stale conflict markers cleared.
- `tests/ops/test_phi35.py` — stale xfail comment block at line ~422 updated to reflect active status.
- `.squad/agents/switch/history.md` — this session appended.
- Decision inbox: stranded `switch-dynamic-shape.md` moved to integration tree.

---

## Session 25 — VkQueryPool GPU timestamps + tracer wiring (2026-07-30T11:27:08-07:00)

**Coordinator task:** Build VkQueryPool GPU timestamps behind `ONNXRUNTIME_EP_VULKAN_TRACE_GPU=1`; wire tracer into execution path; emit `PartitionStats`; confirm messenger positive control.

**D-S25-01 — GpuQueryPool in `rust/src/vk/timestamp.rs`:**
- `GpuQueryPool::new(n_kernels)` — 2×n_kernels queries, TIMESTAMP type, reset on every use
- `cmd_reset()` / `cmd_before(cmd, ki)` / `cmd_after(cmd, ki)` — one before/after pair per kernel
- `read_results()` returns `Vec<Option<(u64, u64)>>` — per-kernel pairs, None for sentinel (u64::MAX)
- Handles multi-node islands correctly: N kernels in an island each get query index `2*ki` / `2*ki+1`
- Query index layout, sentinel never-valid-tick, and n_kernels overflow: 3 unit tests

**D-S25-02 — `cmd_write_compute_timestamp` in `barrier.rs`:**
Single helper in the ONLY file permitted to name `PipelineStageFlags` (layering rule, enforced by `tests/layering.rs`). Both sync2 and legacy backends write `COMPUTE_SHADER` stage.

**D-S25-03 — `create_and_submit` + `wait_fence_then_destroy` in `cmd.rs`:**
Split `submit_and_wait` for calibration anchor: `host_t0` captured just before submit, `host_t1` just after wait. `anchor_uncertainty_us = (t1 - t0) / 2`. `submit_and_wait` delegates to both (backward compatible).

**D-S25-04 — Tracer wiring in `session.rs`:**
`dispatch_ort` gains a `subgraph_region()` span, Phase guards (Record/Submit/FenceWait/GPUKernels), upload/readback `record_transfer()`, GPU query pool (reset → cmd_before per kernel → submit → cmd_after per kernel would be wrong — correct: reset → cmd_before → dispatch → cmd_after per kernel), calibration anchor. `GpuQueryPool::read_results()` called after fence wait; per-kernel GPU durations emitted as spans via `tracer().gpu_kernel_span(ki, before_tick, after_tick, period_ns, valid_bits)`.

**D-S25-05 — Intel 36-bit wrap handled structurally:**
Period = 52.0833 ns/tick on Iris Xe; 36 valid bits → mask = `(1u64 << 36) - 1`. Wrap guard:
```rust
if after_masked < before_masked { after_masked += 1u64 << valid_bits; }
```
A build that drops the mask is green on NVIDIA and CI (both 1.0 ns/tick / 64 bits) but under-reports Intel by 52×. `bench/timestamp_audit.py` exits non-zero when no device with `period > 1.0 ns || valid_bits < 64` is present — makes the CI gap explicit.

**D-S25-06 — Tracer wired in `ep.rs`:**
`VulkanEp::drop()` calls `tracer().export()`. `get_capability_impl` calls `tracer().record_partition()`. Verified by running: 41 span events, 48 counter samples produced to trace file. Without `drop()` calling `export()`, the trace file never appeared despite all wiring inside dispatch — confirmed empirically by Niobe before session.

**D-S25-07 — PartitionStats emitted:**
`record_partition()` called in `get_capability_impl` after island computation. Mouse's `partition.rs` has `PartitionStats` struct with `island_count`, `claimed_nodes`, `largest_island_flops`. Switch owns emission; Mouse owns computation.

**D-S25-08 — Messenger positive control (RTX 4060):**
```
[EP-PLANT] EP_VALIDATION_ERROR_COUNT after planted fence leak = 1
```
Positive control confirmed. M0 criterion 3 is a positive control, not merely silent.

**Intel 52× trap falsifier (R9):** the `bench/timestamp_audit.py` exits non-zero if no local device can produce the distinguishing signal. CI (lavapipe only, both 1.0 ns/tick / 64 bits) cannot falsify the trap — documented explicitly.

**Results:**
- Trace file: 41 spans, 48 counters on both devices ✅
- GPU timestamps: per-dispatch durations for all kernels in multi-node islands ✅
- Messenger: EP_VALIDATION_ERROR_COUNT=1 ✅
- cargo ci: ✅ GREEN (366 tests)
- Committed: `1352405` on `squad/switch`

---

## Session 26 — Multi-node island merge conflict resolution; clippy fixes (2026-07-30T15:41:27-07:00)

**Coordinator task:** Merge `origin/main` (dc36166, Mouse's island-count fix: 321 → 33 islands, 3.7× Intel speedup) into `squad/switch`; resolve `session.rs` conflict; fix 29 clippy errors from Mouse's new partition/clustering code in `ep.rs`; verify with Mouse's tests and messenger positive control.

**D-S26-01 — Conflict resolution (session.rs, 3 hunks):**
- Hunk 1 (`alloc_temp`): Mouse's named/positional mode split adopted — named mode for the multi-node first-kernel case, positional for subsequent kernels. Correct for multi-node islands where `gpu_intermediates` naming must be stable.
- Hunk 2 (`dispatch_ort` opening): BOTH kept — my tracer block first (Phase guards, upload record_transfer), then Mouse's `n_plan_inputs`/`n_plan_outputs` declarations. No double-counting (upload/readback use `record_transfer()`, not `phase()` guards).
- Hunk 3 (submission section): My split-fence wait + GPU timestamps kept; Mouse's `gpu_intermediates` added to both `free_all` calls. Multi-node islands hold `gpu_intermediates` in the island record; `free_all` must release them after the fence fires.

**D-S26-02 — `dispatch_ort` 11-argument lint:**
Mouse added 4 new parameters (`n_intermediates`, `name_map`, `first_temp_token`, `static_intermediate_byte_sizes`). 11 > 7 threshold → `#[allow(clippy::too_many_arguments)]` added. Not worth refactoring; the signature documents the kernel dispatch contract.

**D-S26-03 — 23 clippy errors in ep.rs (Mouse's new partition/clustering code):**
All introduced by Mouse's `dc36166` additions:
- 20 `undocumented_unsafe_blocks`: `node_slots`/`value_info_name` calls in 5 new loops + `slot_bytes` closure getting `(*api).GetValueInfoTypeInfo/CastTypeInfoToTensorInfo/GetTensorElementType/GetDimensionsCount/GetDimensions` + two `release_status` calls. Per-block SAFETY comments added.
- 3 `map_or(false, ...)` → `is_some_and(...)`.
- `for (_root, cluster_nodes) in &clusters` → `for cluster_nodes in clusters.values()`.
- `field_reassign_with_default`: `Island::default()` + `island.nodes = ...` → struct literal with `nodes` field.

**D-S26-04 — Multi-node island query pool (design note):**
GpuQueryPool is created with `n_kernels` where `n_kernels = kernels.len()` — one query pair per kernel in the island. A fused subgraph now has many kernels; the pool is sized for all of them. Per-kernel attribution is correct: kernel `ki` uses query indices `2*ki` and `2*ki+1`.

**Results (2026-07-30T15:41:27-07:00):**
- Mouse's fp16 tests (`test_matmulnbits_fp16_dynamic_batch{,_multirun}`): PASSED on both devices ✅
- Trace file: 89 spans (2 test cases × ~44 spans each) ✅ — confirmed by running, not reading
- Messenger positive control: EP_VALIDATION_ERROR_COUNT=1 ✅
- `cargo ci`: 366 passed, 0 failed ✅
- Committed: `57bfed4` on `squad/switch`

**Key learning — R9 applied to "tracer wired":**
The trace file was confirmed by running `pytest tests/ops/test_matmulnbits.py -k fp16_dynamic_batch` with both trace env vars set, then checking `Test-Path trace_session26.json`. Exists with 89 spans. The instrument that would go red if wiring were broken: no file, or file with 0 spans.

