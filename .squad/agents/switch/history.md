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
   then read_tensor_desc_from_ort (GetTensorTypeAndShape + GetTensorElementType + GetDimensions)
   for each dynamic kernel input slot, re-runs translate via ShapeOnlyRecorder to capture
   push_constants, workgroups, spec_constants, shader, and output TensorDescs.

3. check_bound_input_sizes skips the check when planned == 0 (the dynamic signal).

**D-S18-02 — OQ-15: Re-record per shape**
Command buffer is already re-recorded on every Compute call. Dynamic path adds only host-side
µs-scale shape reads vs ms-scale staging. Bucketing and vkCmdDispatchIndirect deferred to M2+.

**D-S18-03 — ENGINE_ACCEPTS_RUNTIME_EXTENTS flipped**
Three tests in ops/common/claim.rs updated to reflect new baseline (	rue not alse).
Test 
untime_extent_support_is_a_single_switch updated; symbolic_shape_is_tagged_dynamic_shape
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

---

## Session 27 — Tracer end-to-end, timestamp falsifiers, hypothesis verdict, island splitters, PartitionStats fix (2026-07-30T17:22:33-07:00)

**Coordinator tasks:** (1) Verify tracer by running. (2) Timestamp 52× trap — falsifiers. (3) Per-dispatch attribution shape. (4) Fixed-per-submission hypothesis verdict with number. (5) Island splitters histogram. (6) Messenger positive control. (7) PartitionStats largest_island_flops.

**Pre-work:** Merged `origin/main` (Trinity's counter-assertion split, d9bcc9d) — 2 files changed, clean auto-merge.

**Device selection clarification (D-S27-00):**
`select_device()` uses sorted index (discrete-first). NVIDIA RTX 4060 = sorted index 0 (discrete, best-first default). Intel Iris Xe = sorted index 1. `ONNXRUNTIME_EP_VULKAN_DEVICE=0` or unset → NVIDIA. `ONNXRUNTIME_EP_VULKAN_DEVICE=1` → Intel. `epctl --probe-loader` shows Vulkan API enumeration order (Intel=0, NVIDIA=1), which differs from selector index order. This is a naming collision; selector index should be documented.

**D-S27-01 — Tracer confirmed end-to-end (running, not reading):**
`ONNXRUNTIME_EP_VULKAN_TRACE_GPU=1` + trace path + Phi-3.5 → **trace file produced**:
- **322 GPU kernel spans**, real durations (Intel Iris Xe, 52.0833 ns/tick, 36 valid bits)
- Breakdown: 161 MatMulNBits, 64 SkipNorm, 64 Mul, 32 Sigmoid, 1 Add
- One span per kernel per island — NOT one span per island (correct per-dispatch attribution)
- 34 Compute calls (33 surviving islands + 1 small single-node subgraph)
- 461 total span events, 144 counter samples

**D-S27-02 — Timestamp falsifiers (R9):**
`bench/timestamp_audit.py` exit 0 on this machine, both hazards falsifiable:
- **Period scale**: Intel Iris Xe (52.0833 ns/tick); DETECTABLE here — 52× error produces 52× wrong durations. NVIDIA (1.0) and lavapipe (1.0) cannot falsify this hazard; CI is blind to it.
- **Valid-bit mask**: Intel Iris Xe (36 valid bits, wraps in ~3579 s); EXERCISABLE here — unmasked reads would be negative or garbage. NVIDIA (64 bits) and lavapipe (64 bits) cannot falsify this.
- Both instruments agree between EP (`epctl --probe-loader`) and `vulkaninfoSDK` on both devices for both properties. Exit code 0.

**D-S27-03 — Per-dispatch attribution shape:**
`GpuQueryPool::new(kernels.len())` creates one `(cmd_before, dispatch, cmd_after)` triple per
kernel. A 10-kernel island gets 10 independent pairs. Confirmed: 322 GPU spans = one per node,
not one per island. Multi-node island attribution is correct.

**D-S27-04 — Fixed-per-submission hypothesis: verdict and deciding numbers:**

| Metric | Intel Iris Xe (UMA) | RTX 4060 (discrete) |
|--------|--------------------|--------------------|
| GPU kernel total | 784.6 ms | 48.3 ms |
| Fence-wait total | 893.8 ms | 82.7 ms |
| Non-kernel fence overhead | 109.2 ms (3.2 ms/sub) | 34.4 ms (1.0 ms/sub) |
| Record total (CPU, re-recorded) | 1340.3 ms (39.4 ms/sub) | 1316.6 ms (38.7 ms/sub) |
| GPU fraction of fence-wait | 87.8% | 58.4% |

**Verdict: CONFIRMED — but the dominant mechanism is command-buffer re-recording, not vkQueueSubmit overhead.** The deciding number: `38.7–39.4 ms/submission record time`, identical on both devices (it is CPU work), dominates over GPU execution (23.1 ms/sub Intel; 1.4 ms/sub NVIDIA). On NVIDIA, recording is 27× GPU kernel time. vkQueueSubmit itself is only 140–186 µs/submit.

The "per-island amortised figure went UP" (16.6 ms Intel / 25.7 ms NVIDIA) because total_delta/fewer_islands is the benchmark's definition — it was always measuring this ratio, not per-island overhead.

**Data that does NOT settle:** why Intel (807 ms benchmark) beats NVIDIA (1156 ms) despite being 16× slower in GPU compute. Our total trace (record+fence_wait) gives Intel=2234 ms, NVIDIA=1400 ms — NVIDIA faster — opposite of benchmark. The benchmark timing window likely excludes recording (first-run compilation) or measures GPU-only. Further instrument needed: run `bench/phi35_bench.py` with TRACE_GPU=1 and examine which phases fall inside the benchmark's timing window.

**D-S27-05 — Island splitters histogram (`ONNXRUNTIME_EP_VULKAN_CLAIM_LOG`):**
364 records, 322 claimed, 42 declined. Declined by (op, code):

| count | op | code |
|-------|-----|------|
| 32 | `com.microsoft::GroupQueryAttention` | staged |
| 2 | `Gather` | not-registered |
| 2 | `Cast` | staged |
| 1 | `SimplifiedLayerNormalization` | staged (different from SkipSimplifiedLayerNorm!) |
| 1 | `Shape` | not-registered |
| 1 | `ReduceSum` | not-registered |
| 1 | `Sub` | dtype (i64 variant) |
| 1 | `Greater` | staged |
| 1 | `If` | not-registered |

Mouse's priority: GroupQueryAttention (32 instances, primary island splitter). Implementing it
collapses up to 32 island boundaries. The 64 GQA nodes in the model split roughly 50/50 across
two execution paths (only 32 appear in this single-token inference).

**D-S27-06 — PartitionStats §10.0 triple fixed (ep.rs):**
The premature `record_partition()` call (before island computation, island_count=0) was removed.
Tracking vars added before the cluster loop (`largest_island_flops`, `largest_island_nodes`,
`total_flops`, `total_boundary_bytes`). New call placed after `counters::record_capability()`.
Confirmed in trace: `island_count: 33`, `claimed_nodes: 321`, `largest_island_nodes: 10`,
`largest_island_flops: 698351616`, `concentration: 0.031`.
Note: `boundary_bytes_per_inference` is inflated by symbolic-dim fallback (→128 for -1 dims);
treat as qualitative upper bound until shapes are available at GetCapability time.

**D-S27-07 — Messenger positive control (NVIDIA, session 27):**
`ep_messenger_fires_for_planted_fence_leak` — `EP_VALIDATION_ERROR_COUNT = 1` after planted
fence leak on NVIDIA RTX 4060. Messenger is wired and capturing. Intel not separately confirmed
in this session (dispatch_integration runs NVIDIA first as the best-score device). M0 criterion 3
remains: messenger is not merely silent.

**Device selection clarification (D-S27-00) — epctl display order vs selector order:**
A naming collision exists: `epctl --probe-loader` shows Vulkan API physical device indices
(Intel=0, NVIDIA=1), but `ONNXRUNTIME_EP_VULKAN_DEVICE` is interpreted as sorted-list index
(NVIDIA=0, Intel=1 after discrete-first sort). Setting `DEVICE=1` gives Intel, not NVIDIA,
which is the opposite of what epctl's display implies. This should be documented in ENGINE.md.

**Results:**
- Tracer: 322 GPU spans, real durations, trace file confirmed ✅
- Timestamp audit: exit 0, both falsifiers present on Intel ✅
- Per-dispatch attribution: correct (one span per kernel) ✅
- Hypothesis verdict: recording-dominated (38.7 ms/sub); GPU kernel secondary ✅
- Island splitters: GQA (32) primary, then Gather/Cast/etc. ✅
- PartitionStats: island_count=33, largest_island_flops=698M ✅
- Messenger: EP_VALIDATION_ERROR_COUNT=1 (NVIDIA) ✅
- `cargo ci`: 366 passed, 0 failed ✅
- Decision inbox: `switch-session27-trace-and-hypothesis.md`



---

## Session 29 (2026-07-30T17:19:29-07:00) — Sub-phase attribution + weight-tensor GPU buffer cache

**Coordinator assignment recap:** (1) characterize what is expensive inside `vulkan.record`, (2) confirm device-1 phase split, (3) implement and measure the fix.

### Phase breakdown — what is expensive inside `vulkan.record`

Added three sub-phase instruments: `Phase::CmdUpload`, `Phase::DescAlloc`, `Phase::PipelineLookup`. Results (NVIDIA, 3 inferences, 100 Compute calls, GPU timestamps enabled):

| Sub-phase | ms | % of record |
|---|---|---|
| `vulkan.cmd_upload` | 5204 ms | 97.1% |
| `vulkan.desc_alloc` | 83 ms | 1.6% |
| `vulkan.pipeline_lookup` | 36 ms | 0.7% |

Root cause: CPU `memcpy` of model weight tensors into staging buffers on every `Compute` call. Weight tensors (B: 4.5–13 MB, scales: 576 KB) are re-copied from the same CPU pointers every inference. 97% of record time is upload, not Vulkan API overhead.

Intel: identical pattern — cmd_upload = 95.7% of record.

### Device-1 (NVIDIA, device 0 in sorted-index) phase split confirmed

NVIDIA (100 Compute calls, 3 inferences, GPU timestamps enabled):

| Phase | ms | share |
|---|---|---|
| `vulkan.record` | ~12700 ms | 83.9% |
| `vulkan.fence_wait` | ~2100 ms | 13.9% |
| GPU kernels | ~1175 ms | 7.8% |
| `vulkan.submit` | ~75 ms | 0.5% |

NVIDIA driver quirk: with `ONNXRUNTIME_EP_VULKAN_TRACE_GPU=0`, NVIDIA record time is 7.4× slower (QueryPool presence activates a different scheduling path). All timing comparisons use GPU timestamps enabled.

### Weight-tensor GPU buffer cache — implementation and measurement

**Prediction (stated before building):** Weight tensors have stable CPU pointers across inferences. Caching tensors ≥ 32 KB should reduce upload by ~99.7% on inferences 2+. Expected: ~5-6× on NVIDIA, ~1.5-2× on Intel for 3 inferences.

**Implementation:**
- `GpuBuffer::borrowed_ref()` in `alloc.rs` — non-owning handle, `Allocator::free()` is no-op.
- `VulkanSession::weight_caches: HashMap<u64 (subgraph_id), HashMap<(usize cpu_ptr, u64 byte_size), GpuBuffer>>`.
- Cache populated after fence-signal; served as borrowed_ref on subsequent calls (skips memcpy + vkCmdCopyBuffer + barrier).
- `impl Drop for SubgraphComputeInfo` calls `release_weight_cache(subgraph_id)`.
- Borrow split: `let weight_cache_ptr: *mut _ = ... as *mut _;` (safe — disjoint fields).

**Measured results (NVIDIA RTX 4060, commit cdcc349):**

| Inference | cmd_upload total | mean/call |
|---|---|---|
| 1 (cold) | 4118 ms | 124.8 ms |
| 2 (warm) | 212 ms | 6.4 ms |
| 3 (hot) | 1.3 ms | 0.038 ms |

Net for 3 inferences: 4332 ms vs expected ~12354 ms (no cache) → **2.85× reduction in upload time**.

**Measured results (Intel Iris Xe, device 1):**

| Inference | cmd_upload total | mean/call |
|---|---|---|
| 1 (cold) | 4258 ms | 129 ms |
| 2 (warm) | 128 ms | 3.9 ms |
| 3 (hot) | 2.3 ms | 0.07 ms |

**Correctness:** `test_phi35_vulkan_multirun_logits_stable` PASSED on both devices with bit-identical logits (argmax=30751 on NVIDIA, argmax=30751 on Intel). All 420 tests pass.

**Anomaly — Intel submit inflation:** Intel `vulkan.submit` = 2594 ms for 100 calls (26 ms/call) vs coordinator baseline of 0.127 ms/call (205× higher). Hypothesis: borrowed-ref buffers keeping large device-local GpuBuffers alive across Compute calls triggers Intel driver buffer-residency tracking per vkQueueSubmit. Requires investigation.

### Messenger positive control — Intel (device 1) ✅

`a_planted_vulkan_violation_is_caught_by_the_validation_layer` with `ONNXRUNTIME_EP_VULKAN_PLANT_VALIDATION_VIOLATION=1` on device 1 (Intel): PASSED. Messenger fires on Intel as expected.

### Decision written

→ `.squad/decisions/inbox/switch-weight-cache-recording-bottleneck.md`

Includes: per-inference cache results, Intel submit anomaly, R9 falsifier for the 68% claim, courtesy note to Tank on the fence_wait lever size relative to the cache win.

### Pending

- [x] Intel submit inflation — EXPLAINED: cold-inference-only, Intel UMA synchronous vkCmdCopyBuffer during submit. Warm inferences normal (0.6ms/call). Not a bug.
- [ ] Predicted vs measured END-TO-END speedup on coordinator's 31-inference Intel run — coordinator should re-run `bench/phi35.py` on Intel device 0 to get wall-clock.
- [ ] Timestamp falsifiers — stated in session 27, should be formally confirmed.


---

## Session 30 (2026-07-30T20:34:34-07:00) — CB-caching prediction; batching feasibility; post-cache bottleneck shift

### Intel submit anomaly — explained

Submit inf1 = 77.6ms/call, inf2 = 0.6ms/call, inf3 = 0.4ms/call. Anomaly is cold-inference-only. On Intel UMA, `vkQueueSubmit` executes `vkCmdCopyBuffer` synchronously (no PCIe bus). After my weight cache: warm inferences have no large staging copies → submit is normal (0.6ms). Not a bug.

### CB-caching prediction (stated before building)

**What it would eliminate:** `vkCreateDescriptorPool` + `vkAllocateDescriptorSets` + CB begin/end overhead per warm call.

**Measured warm-call overhead (Intel post-cache):** record mean = 1.88ms/call. DescAlloc=0.17ms/call, PipelineLookup=0.12ms/call, unaccounted CB overhead ≈ 1.59ms/call.

**Why gains are small:** Most descriptor bindings (intermediates, activations) change every call. Only weight bindings are stable. Dominant warm cost = GPU execution (q_gemv ≈ 56ms/call on Intel Iris Xe).

**Prediction:** CB caching → warm record drops from 1.88ms to ~0.1ms/call.
- For 3-inference Intel: 66 warm calls × 1.78ms = 117ms savings = **0.8% of 14554ms total**
- For 31-inference extrapolation: 630 warm calls × 1.78ms = 1122ms savings → **1.12× incremental** speedup
- **Falsifier:** DescAlloc total should drop from 16ms to <1ms for 100 calls after CB caching

**Decision: do NOT implement CB caching this session.** Gain is <1% after the weight cache. GPU execution (q_gemv) dominates; CB caching doesn't address it.

### ORT calling pattern and batching feasibility

NVIDIA gaps: median=0.58ms, max=4258ms (between inferences).  
Intel gaps: median=0.65ms, max=4842ms (first GQA inference on CPU/reference executor).

**The 4842ms gap** is non-Vulkan-EP CPU work (GQA running on reference backend). Subsequent within-inference gaps are ~0.65ms.

**Batching conclusion: NOT feasible without ORT interface change.**
- Islands are data-dependent (layer-by-layer); GPU-side barriers needed between submits
- ORT does CPU work (0.65ms) between each Compute call — no inference boundary signal exposed to EP
- Non-EP GQA cost (~4842ms cold, ~2ms warm per inference) is not in our spans and cannot be batched
- Deferred-submission would need: detect inference boundary (no hook exists), accumulate submits, flush at boundary

### Post-cache bottleneck hierarchy (warm inferences)

**Intel:**
1. GPU kernel execution (q_gemv ≈ 56ms/call, 89% of fence_wait warm) ← PRIMARY
2. Fence_wait idle (driver scheduling): 14ms/call median = 462ms/inference
3. Record (post-cache): 1.88ms/call = 62ms/inference

**NVIDIA:**
1. GPU kernel execution (q_gemv dominant)
2. Record + fence_wait: both ~1ms/call (small)

**R9 falsifier for post-cache Intel claim:** if GPU execution time is not the bottleneck, a run with a faster GPU (or smaller model) would show proportionally reduced fence_wait without changing record. Falsifiable by device swap.

### Fence-wait idle decomposition (post-cache)

Coordinator's script on trace_s29_nv_cache.json:
- fence_wait=248ms, idle=103ms (41.6%), kernel=145ms, median idle=0.76ms/call

Coordinator's script on trace_s29_intel_cache.json:
- fence_wait=5981ms, idle=2054ms (34.3%), kernel=3927ms, median idle=14.21ms/call
- 256 of 964 GPU spans start OUTSIDE fence_wait (Intel UMA semi-synchronous execute during submit)

### Decision written

→ `.squad/decisions/inbox/switch-cb-cache-prediction.md`

No code changes this session. Recommend GPU kernel optimization (q_gemv MatMulNBits) as next highest-leverage step. Deferred submission is medium-term (requires ORT API change).

---

## Session 31 (2026-07-30T21:03:48-07:00) — Measurement contamination acknowledgement; quiet-machine request

**Coordinator finding:** command-buffer recording inflates 9.5× under CPU contention from concurrent agents compiling Rust on the same machine. Device 1 (NVIDIA) measured at record=65540ms vs device 0 (Intel) at 184356ms (contended) vs device 0 at 19460ms (quiet). Same device, same code, 9.5× apart. Withdrawing the device-1 comparison as unusable.

**This contaminates all timing data from sessions 28-30** (my traces: trace_s28*, trace_s29*) — all taken while multiple agents were running on this machine.

### What survives contamination

**Correctness** is unaffected by timing contamination:
- `test_phi35_vulkan_multirun_logits_stable` PASSED with bit-identical logits on both devices
- 420 tests pass (layering, portability, validation_control)
- The cache mechanism is correct; only the timing of its effect is uncertain

**Qualitative findings** (direction survives, magnitude does not):
- `cmd_upload` drops from cold (large) to warm (near-zero) — the RATIO between inf1 and inf3 is the finding. inf3 = 1.3ms vs inf1 = 4118ms is a 3168× ratio; even a 9.5× contention swing cannot make the warm calls expensive (they are doing ≤ 6 KB of uploads)
- Recording cost is fixed per-Compute, not per-dispatch — confirmed by the coordinator's pre-cache quiet-machine data and consistent with my post-cache warm-call data (warm 20-dispatch islands at 1.03ms median, much cheaper than cold regardless of contention)
- The Intel submit anomaly explanation (UMA synchronous copy) — directionally correct; absolute values unreliable

**What must be re-measured quiet:**
- Absolute wall-clock speedup from weight cache (my "2.85×" is based on contaminated data)
- Phase split fractions (68% record, 30% fence_wait) may have shifted under contention
- The CB-caching prediction (< 1% gain) is qualitatively robust but the exact percentages are not

### Request: quiet-machine measurement window

Coordinator offered to hold other agents idle. **Weight cache is committed (cdcc349), tested, and correct. Ready to measure.** When the machine is quiet, re-run `test_phi35_vulkan_multirun_logits_stable` on both devices with `ONNXRUNTIME_EP_VULKAN_TRACE_GPU=1` and produce fresh traces. The before/after comparison requires a pre-cache baseline at the same machine state — the coordinator has the pre-cache trace from the original "68%" measurement which was taken on a quiet machine (device 0 Intel). That baseline is the control; my post-cache traces need quiet conditions to match it.

### Revised weight cache claim (conservative)

- **Correctness:** confirmed, two devices, 3 inferences each, bit-identical
- **Direction:** warm-inference upload cost drops to near-zero (inf3 = 1.3ms on NVIDIA, 2.3ms on Intel)
- **Magnitude:** pending quiet-machine re-measurement; the "2.85×" figure should not be cited until then
- **Falsifier for the direction claim:** if warm-cache cmd_upload on a quiet machine exceeds 10ms/call for any warm call (inf3 on either device), the cache mechanism has a correctness defect, not just a contention artifact

Two findings apply to every agent on the team:

**(a) A mechanism that exists in a file but not in a call graph is indistinguishable from
one that does not exist.**  Verification by reading is insufficient.  Verify by running.
Five such mechanisms surfaced in this single batch: partition.rs, the GPU tracer,
model_output_equivalence, compute_failures, and should_claim_island.  In every
case the code was correct; the wiring was absent; the absence was invisible to review.

**(b) 85.9% of inference wall-time involves no GPU work** (recording 68.3%, fence-wait
idle 16.3%, submit 0.3%; GPU kernels 14.1%).  Optimising GPU kernels before the

command-buffer recording bottleneck is resolved is low-leverage.  Align work priorities
accordingly.



---

## Session 32 — Device-label fix; messenger positive control (valid); timestamp falsifiers; bench parsers (2026-07-30T19:47:00-07:00)

**Context:** Resumed from summary. Prior commit cdcc349 (weight-tensor GPU buffer cache) eliminated 97% of per-inference upload for warm inferences. Large origin/main merge (d836e0) landed Tank's device-backed allocator, transfer.rs seams, and documentation.

Critical coordinator correction: ONNXRUNTIME_EP_VULKAN_DEVICE=0 is NVIDIA (discrete-first sort); DEVICE=1 is Intel Iris Xe. All prior coordinator device labels were inverted.

**D-S32-01 — Device-label semantics in instance.rs (commit 7d66986):**
probe_loader_report() printed Device N: Name using Vulkan enumeration order while select_device() indexes a best-first sorted list. Same integers, different index spaces — a reader of the old output naturally concluded DEVICE=0 → Intel (wrong).

Fix: add [Vulkan enum index N] annotation to each device header, plus an explicit ONNXRUNTIME_EP_VULKAN_DEVICE selector index map block showing the sorted order with Vulkan enum indices beside them. Selection behavior (discrete-first sort) unchanged — compatibility outranks elegance.

`
ONNXRUNTIME_EP_VULKAN_DEVICE selector index map:
  [0] 'NVIDIA GeForce RTX 4060 Laptop GPU'  (Vulkan enum index 1)
  [1] 'Intel(R) Iris(R) Xe Graphics'  (Vulkan enum index 0)
Would select: selector index 0 → 'NVIDIA ...' (Vulkan enum index 1; best-first default)
`

Two bench parsers broke on the new format; updated regexes in 	imestamp_audit.py and nvironment.py (commit 7f734a2).

**D-S32-02 — Messenger positive control (criterion 3 — first valid readings):**
All prior "no validation errors" readings were void: the messenger was not attached (layer output went to default stderr handler, not in-process).

New valid readings:
- p_messenger_fires_for_planted_fence_leak PASS (NVIDIA device, run as --ignored): EP_VALIDATION_ERROR_COUNT = 1 after planted VkFence leak at kDestroyDevice
- _planted_vulkan_violation_is_caught_by_the_validation_layer PASS
- _clean_run_produces_no_validation_errors PASS (with ONNXRUNTIME_EP_VULKAN_REQUIRE_VALIDATION=1)

**Lane gap:** p_messenger_fires_for_planted_fence_leak is #[ignore]d (safely — it deliberately causes a Vulkan error and must not run concurrently with library tests). Morpheus: a control that must be opted into is not in the lane. Written to switch-criterion3-lane-gap.md inbox for Trinity.

**D-S32-03 — Timestamp falsifiers (both confirmed):**
Both falsifiers reside on Intel Iris Xe (selector index 1, Vulkan enum index 0):

| Hazard | Falsifier | Value |
|--------|-----------|-------|
| Period scale | Intel Iris Xe | 52.0833 ns/tick (52× error if ignored) |
| Valid-bit mask | Intel Iris Xe | 36 valid bits (unmasked → garbage/wrap) |

NVIDIA RTX 4060 (selector 0): period=1.0, validBits=64 — both hazards indistinguishable from a broken conversion. This means a build that drops period scaling and valid-bit masking is green on NVIDIA and in CI (lavapipe also 1.0/64), while under-reporting every Intel duration by 52× with potential negative durations on wrap.

ench/timestamp_audit.py exits 0. Conversion covered by unit tests in 	race.rs using the real hardware constants.

**State at end of session:**
- Device-label fix: ✅ committed 7d66986
- Bench parser fix: ✅ committed 7f734a2
- Criterion 3 positive control: ✅ first valid readings, both devices
- Timestamp falsifiers: ✅ stated formally, audit exits 0
- Lane gap: 📋 written to inbox for Trinity
- Quiet-machine cache benchmark: PENDING (coordinator offered to idle other agents — say the word)
- lloc_device_authoritative_spans: still 0 (session uses its own VkBuffers for weight cache, not ORT allocator handles — architectural constraint documented)

---

## Session 33 - Cache byte sweep confirms 2642x reduction; two-VkDevice flag (2026-07-30T22:06:01-07:00)

**Context:** Merged origin/main (692e7d0). Coordinator claimed the cache is not reducing upload, based on seeing cmd_upload 15197.8 ms with 661 subgraph spans. Task: verify via byte count (deterministic, immune to CPU contention), not wall time.

**D-S33-01 - Measurement instrument choice:**
Used tracer vulkan.transfer_bytes counter events (from t.record_transfer(Transfer::Upload, uploaded_bytes) in session.rs). The allocator counters (alloc_staged_bytes, alloc_device_upload_bytes) both read 0 on all runs - they do not observe session staging. This is a named R11 exposure (two upload accountings, one structurally blind). The tracer path is the correct instrument: uploaded_bytes directly accumulates bytes from the non-cached memcpy loop; cached entries skip the loop via `if stg.borrowed { continue; }` and do not add to uploaded_bytes.

**D-S33-02 - Per-inference upload byte sweep (NVIDIA, device 0, build cdcc349):**

Inference 0 (cold):  33 calls, 1997.596 MiB - full weight set
Inferences 1-13 (warm): 33 calls each, 0.756 MiB each - activations only

Pre-cache baseline (Tank sweep): 1/2/3 runs = 1997.60 / 3995.19 / 5992.79 MiB - exactly linear.
Post-cache:              1/2/3 runs = 1997.6 / 1998.4 / 1999.1 MiB  - flat after cold.
Reduction: 1997.596 -> 0.756 MiB per warm inference = 2642x.

The 0.756 MiB warm residual is 33 subgraphs x ~24 KB activation tensors - correct; small tensors (< 32 KB) are intentionally not cached (seq=1 inputs are runtime-variable).

Discrepancy with coordinator observation: coordinator saw cmd_upload 15197.8 ms with 661 spans - inconsistent with active cache. Likely: stale DLL from before cdcc349, or wrong library path. Cannot resolve without knowing which binary the coordinator ran.

**D-S33-03 - Two-VkDevice architectural flag (written to inbox):**
alloc_device_authoritative_spans is structurally 0: Tank's device-backed allocator uses its own VkDevice, separate from the session VkDevice. Cross-device VkBuffer access is not permitted at the Vulkan API level. Not solving unilaterally; flagged to coordinator/Morpheus for ruling. Three options: share VkDevice, external memory import (VK_KHR_external_memory), or accept the gap at M0 (weight cache captures most of the residency benefit anyway).

**State at end of session:**
- origin/main merge: clean (426 tests, 0 failures)
- Cache byte verification: 2642x reduction confirmed on NVIDIA (tracer instrument)
- Two-VkDevice flag: written to inbox
- Quiet-machine wall-clock before/after: coordinator offer still open

---

## Session 34 - Standing perf directive; MATCH + byte sweep reconfirmed (2026-07-30T22:23:35-07:00)

Context: Coordinator issued standing directive - performance is first-class, continuous. 
No new code written - directive is behavioral, not structural. Results below.

model_output_equivalence: MATCH on both devices (phi35.py --iters 1 --warmup 0).

Warm-cache byte sweep reconfirmed (trace_warm.json, phi35.py --iters 3 --warmup 1):
  Inf 0 (cold):  1997.596 MiB  <- full weight set, cache cold
  Inf 1-4 (warm):  0.756 MiB each  <- activations only, cache warm
  Pattern: NOT linear. Pre-cache was linear at 1997.60 / 3995.19 / 5992.79.
  Post-cache: 1997.6 / 1998.4 / 1999.1 MiB (flat). 2642x reduction.

Warm-cache timing (contended machine - not quotable per coordinator ruling):
  Intel Iris Xe (selector 1, UMA): 41.325 ms FASTER than CPU-only (0.9x)  <- cache + UMA
  NVIDIA RTX 4060 (selector 0, discrete): 668.399 ms slower (3.2x)

Intel faster than CPU is real: UMA + weight cache means GPU runs kernels on resident 
weights without transfer. NVIDIA still upload-limited on warm activations + fence_wait idle.

Outstanding: requested quiet-machine window for wall-clock before/after on NVIDIA.

---

## Session 35 - §6.5 VkDevice seam; byte sweep stands (2026-07-30T22:32:54-07:00)

Context: Morpheus ruled §6.5 - exactly one VkDevice per (physical device, EP instance).
Switch owns the seam. Merged origin/main (676fb94, Morpheus history + DESIGN.md §6.5/R12).

D-S35-01 - §6.5 implementation (commits e2a23b4):
Added to vk/device.rs:
  EpDeviceShare { ash_device: ash::Device, physical_device, caps, compute_queue_family }
  - ash::Device derives Clone in ash 0.38 (bitwise copy of handle + dispatch table, no Arc)
  - Cloning shares the same VkDevice handle without reference counting
  - EpDeviceShare does NOT own the VkDevice (no Drop); session::Device::drop calls vkDestroyDevice
  - Safety: VulkanSession is EP-scoped (§2.3); ORT frees tensors before destroying EP
  static EP_DEVICE: OnceLock<EpDeviceShare>
  pub(crate) fn register_ep_device(device: &Device)  <- called by VulkanSession::create
  pub(crate) fn ep_device() -> Option<&'static EpDeviceShare>  <- Tank reads this

Added to vk/session.rs:
  register_ep_device(&device) call after Device::create succeeds

Tank's work: host_device_memory.rs calls ep_device() and uses the returned handle
instead of creating its own Instance + Device. That unpins alloc_device_authoritative_spans.

The Instance concern: EpDeviceShare does not hold Instance. The session's Instance
keeps the Vulkan loader alive. Since the session outlives all device-memory usage (§2.3
+ ORT teardown order), Tank does not need to retain a separate Instance in his path.

D-S35-02 - R12 acknowledged:
alloc_device_upload_bytes = 0 (UNOBSERVABLE, frame mismatch - allocator counter cannot
see session staging). tracer vulkan.transfer_bytes is the correct instrument.
The previous reads of alloc_device_upload_bytes as "0 bytes uploaded" were frame errors.

State: 426 tests pass. Seam committed. Byte sweep from session 33/34 stands:
  cold inf 0: 1997.596 MiB; warm inf 1+: 0.756 MiB. 2642x reduction confirmed.
  model_output_equivalence = MATCH on both devices.


## Session 35 (2026-07-31) — Defects 1/2/3 closed on real Phi-3.5; offer gated

Fresh session; prior work merged. Guard D (with Trinity's `pathlib.Path`->`Path` fix) now
reports 4 real defects; 2 are mine. All measured on NVIDIA dev 0, real Phi-3.5 (2GB), one
process. Bytes are deterministic (immune to the 9.5x contention swing), so valid on a busy box.

DEFECT 1 (weight-cache device leak) — FIXED + WIRED (R10). `VulkanSession::Drop` now
`drain_weight_caches()` then dumps counters AFTER drain (the DataTransfer release path fired
before the weight-cache release, so the old dump site recorded release_calls=0 — a frame error,
not a real 0). Predict-then-measure device high-water after N sessions: predicted FLAT (one
session's peak, not linear); measured over 3 sequential sessions high_water=4,195,027,596 (~3.907
GiB) = ONE session's peak, sub-MB delta N=1->N=3 (a ~2GB/session leak would show ~6-12GB or OOM).
bytes_in_use=0 at teardown; allocs==frees=3975. R10 artifact (content varies with input):
release_calls=3 (one drain/session Drop), release_buffers=972 (326x3), release_bytes=6.28GB
(2.095GBx3). Original failure mode was SILENT CPU FALLBACK (5th instance), not a surfaced error.

DEFECT 2 (50 KV outputs never written) — FIXED. Write path is mine. Root cause from test source:
empty past [1,32,0,96] -> past_len_max=0 -> present sized 0 -> sz==0 placeholder branch -> never
written; ORT expects [1,32,1,96]. Fix in translate_gqa: gate empty_past=(past_len_max==0), bind
present as REAL bind_output [B,Nkv,seq_len,D], push kv_len=seq_len as stride; past>0 keeps
aliasing. Shader unchanged (present bindings 7,8 already separate from past 1,2). Evidence: 65
outputs bit-identical across 3 runs, kv_all_zero_run0=0, differing=0 (was 50). Logits (out 0)
never differed. Added falsifying test translate_gqa_empty_past_binds_real_present_buffers.

DEFECT 3 (compute_calls=1 after 5 runs) — DOWNSTREAM of Defect 1 (confirmed by observation).
With no OOM, compute_calls==N and compute_failures=0 in both the N-run sweep and the 3-session
run. The 1 was run-1 OOMing and later runs never reaching Compute.

offer_shared_device (§6.5): fixed index-space bug (key on capable.info.index = factory/physical
enumerate index, NOT the sorted-capables selector idx). With offer ON single-session:
alloc_device_frame SPLIT-DEVICE->SHARED, alloc_device_authoritative_spans UNOBSERVABLE->0 (the
transition IS the §6.5-closed artifact, R12). BUT offering unconditionally -> multi-session UAF:
SessionSharedCtx holds borrowed cloned ash handles (no refcount), provider is process-global &
cached, each session builds a fresh VkDevice; session 0 Drop destroys it -> STATUS_ACCESS_VIOLATION
(0xC0000005) on session 1. Real fix = sessions reuse the process-global EP device
(register_ep_device), the one-VkDevice-per-process refactor already flagged
(switch-two-vkdevice-flag.md). Gated: default OFF (safe, SPLIT-DEVICE), opt-in
ONNXRUNTIME_EP_VULKAN_OFFER_SHARED_DEVICE=1 for single-session SHARED verification. Verified: 3
sessions survive default-env, no OOM/crash.

Owed closed: Trinity — tests/validation_control.rs needs nothing further from me; #[ignore] on
ep_messenger_fires_for_planted_fence_leak justification accepted, closed. Niobe — trace.rs
"command-buffer recording" caveat fixed (false in every shipped trace; real recording 87-229ms
~1-3% wall, upload was 98% of that phase; R11).

cargo ci green (fmt+clippy+tests). Release DLL builds. Committed on squad/switch, not pushed.

## Session 36 - §6.5 CLOSED (process-global VkDevice); containment RED is Niobe's gate; record residual (2026-07-31T20:28:45-07:00)

Merged origin/main 77d5d2a into squad/switch (clean, 23 files).

TASK 1 - §6.5 closed. Root of last session's UAF: PROVIDERS/OFFERED and HandleRegistry are
process-global, the offered device was session-scoped and vkDestroyDevice'd at teardown. Fix is
"stop destroying the device", not "stop offering it". New vk::device::acquire_ep_device() returns
&'static EpDeviceOwner, one per physical device index, built once and Box::leak'd (EP_INSTANCE /
EP_DEVICES OnceLocks). VulkanSession now BORROWS &'static Device/Instance/CapableDevice; per-session
children (allocator, cmd pool, pipeline cache, weight caches) keep session lifetime and are still
destroyed first. ONNXRUNTIME_EP_VULKAN_OFFER_SHARED_DEVICE gate REMOVED - offer is unconditional,
once, from CreateEp, Arc retained for process lifetime (Tank's three obligations met).

The artifact (R10), not an assertion: bench/results/probe_sec65.py, 3 sequential sessions, device
memory ON. Selector 0 (NVIDIA): alloc_device_frame SHARED, alloc_device_authoritative_spans went
"UNOBSERVABLE" (JSON string) -> 0 (integer). That is UNOBSERVABLE -> UNWIRED; the TYPE changed,
which is the falsifier. Selector 1 (Intel): still SPLIT-DEVICE / "UNOBSERVABLE" - and that is
correct, not a bug: ONNXRUNTIME_EP_VULKAN_DEVICE indexes enumerate_capable_devices() (best-first)
while the offer is keyed by capable.info.index (raw vkEnumeratePhysicalDevices), so DEVICE=1 forces
the session onto Intel (info.index 0) while ORT asks for factory index 1 (NVIDIA). Added INFO logs
on both ensure_registered branches + offered_indices(). The two runs differing with their input is
itself the R10 evidence. UNWIRED -> measurement is Tank's: transfer::device_buffer_for must acquire
a production caller for allocator::tally::on_device_authoritative.

Census: audit_instruments.py --check said verbatim "FAIL: 1 instrument(s) got wired - good news,
update the baseline: - vk/host_device_memory.rs::offer_shared_device". Baseline regenerated; tool
also dropped ops/claim_log.rs::record from ambiguous (Mouse should confirm). Trinity's
test_census_baseline_has_no_drift green. test_phi35.py 8 passed / 1 skipped on BOTH devices.
Committed b262292.

Two defects found, both outside my files: (1) OwnedDevice::create(device_index) IGNORES its argument
and resolves via select_device, so the Intel run built an Intel device for the NVIDIA-keyed registry
- alloc_device_frame_device can disagree with its own frame key (Tank). (2) Pre-existing flake, text
not count per R13: "assertion `left == right` failed: peak depth must be the bound / left: 8 /
right: 4" at src\allocator.rs:2516 - quarantine_peak_spans is a process-wide tally shared with
parallel tests. ERROR(instrument), not a detection. Passes in isolation; 388 passed on re-run.

TASK 2 - phase_containment RED is in HER gate, not my spans. bench/phases.py::analyse computes
`siblings` then passes `attributed` (ALL spans, incl. nested cmd_upload/desc_alloc/pipeline_lookup)
into phase_containment at line 1720 - children double-counted with parent `record`. Ratios with
attributed: 1.355 / 0.946 / 0.950 / 0.963; with siblings: 0.703 / 0.928 / 0.929 / 0.930 -> red:
false. Only subgraph 0 exceeds 1.0 because the cold call emits 1412 pipeline_lookup spans, which is
exactly why the gate says "1 subgraphs". ZERO orphan spans, so the one-island/353-node structural
worry does NOT occur - the failure is arithmetic. My spans self-describe: nested_in=record on the
sub-record phases, none on record/submit/fence_wait; phase_nesting CONSISTENT; gpu_span_accounting
1412==1412; gpu_containment 0 violations. Reported, did not edit bench/.

R11 exposure in my OWN instrument: every child of `record` was named and printed, so the
decomposition LOOKED closed. Per call: cold record 1719.6ms (cmd_upload 1572.0, desc 4.8, pipe 19.4,
residual 123.4ms = 7.2%); warm 19.9/24.0/22.8ms with residual 92.6/94.0/92.5%. ~93% of warm `record`
had no span of its own. Added record_residual_us() + a "record RESIDUAL" summary row + a unit test
that asserts the residual VARIES between cold and warm regimes rather than asserting a number. Trap
I introduced and closed: the summary row is CUMULATIVE over all Compute calls (prints 10.4% here, a
mixture belonging to no single call) - row text and caveat now say so. Rewrote the Phase::Record
caveat to be regime-dependent; the old "upload is 96%, recording 1-3% of wall" is now false in the
OPPOSITE direction.

Fence-wait (rank 3): two estimators disagree by 2.4x, so nothing is quotable yet - ERROR(instrument)
per R13. overlap (intersect GPU spans with the fence_wait window on the host axis, depends on the
calibration anchor): 50.2/50.2/43.3/50.2% idle. cluster (group GPU spans into submissions by the
large gaps, alignment-free, durations only): fence_wait 69.6/93.9/94.0/71.8ms vs gpu_busy
49.4/83.8/71.0/58.5ms (n=353 each) -> 29.0/10.8/24.5/18.5% idle. I withdraw the overlap figure; the
difference measures my anchor, not the GPU. Two things surviving both: (a) between-kernel bubbles
are ~0.4ms of a 50-84ms cluster, so the idle is at the SUBMISSION EDGES (submit->first kernel,
last kernel->host wake), not between dispatches - vkQueueSubmit is still 0.2ms; (b) kernel time
itself swings 49.4->83.8->71.0->58.5ms for the SAME 353 dispatches, a 70% spread, so no idle share
from a single run is publishable regardless of estimator. Worker record had
model_output_equivalence MATCH and claimed_nodes 353 but NO executed_by key, so the run is
unattributed and I quoted no wall-clock from it. The old 53.6% and the withdrawn 3.1x/3.7x stay
withdrawn.

TASK 3 - Trinity closed. Correction for the record: ep_messenger_fires_for_planted_fence_leak is NOT
in tests/validation_control.rs; it is a --lib test at rust/src/vk/dispatch_integration.rs:482 (my
file). validation_control.rs needs nothing from me. #[ignore] stays: instance-scoped messenger +
process-wide EP_VALIDATION_ERROR_COUNT means the assertion is only sound when the test owns the
process; it is a lane declaration, not a suppression, and her subprocess wrapper enforces isolation
via the OS. Verified today post-§6.5: "[EP-PLANT] EP_VALIDATION_ERROR_COUNT after planted fence leak
= 1 ... ok". Asked her to gate the wrapper on Instance::validation_armed() (instance.rs:450) so an
unarmed machine reports ERROR(instrument) instead of green. Flagged the §6.5 consequence: the
production EP device is never destroyed now, so any "validation errors == 0 at shutdown" gate on the
production path would be UNOBSERVABLE (R12), not passing - the positive control is unaffected
because it builds its own Instance+Device.

Housekeeping: probe output goes to bench/results/ (added probe_record_residual.py, reproduces both
findings from a trace); raw traces renamed to the gitignored trace_*.json pattern so they stay as
evidence on disk without ever entering git. Nothing written to the repo root.

Decisions: switch-sec65-closed.md, switch-phase-containment-niobe.md,
switch-validation-control-trinity.md. clippy clean, cargo test --lib green, committed on
squad/switch, not pushed.

## Session 37 — 2026-08-01 — the GEMV column tile, and the Intel gap separated from the hardware

Merged origin/main (efbf18c) into squad/switch — clean, 33 files, brought in Niobe's
gpu_steady_tail, bench/exec_census.py, Mouse's ops/indexing.rs, the new gather and
simplified_layer_norm shaders. First committed the interrupted prior session's index-space WIP as
24aeb9d so the merge had a clean base.

TASK 1 — q_gemv_matmul_nbits_f16.

Built my own instrument first: bench/results/probe_gemv.py runs the traced phi35 worker,
reconstructs per-inference GPU busy from the gpu_ns device timestamps, feeds Niobe's
phases.gpu_steady_tail, and prints per-kernel totals. Three terminal states. I did NOT score
against Niobe's number — Mouse re-scored one of his own predictions yesterday and found he had
scored against a figure Morpheus gave him rather than one he measured. Measured my own baseline at
my own commit: 40.202 ms/inference, kernel 254.77 us. Niobe had 40.201 / 253.4. Independent
agreement is the only reason I trust either.

PREDICTION, stated before building: kernel 254.8 -> ~105 us; total GPU busy -> ~18 ms, range 14-25.

Rewrote the kernel around a QB_COLS column tile (spec constant id 4, max 8): one workgroup computes
8 adjacent output columns with the column loop INSIDE the activation load, so A[m][0..K) is fetched
once and reused. Four supporting changes: workgroup size now DIVIDES blocks_per_col instead of
covering it (at K=3072 the old rule took 128 invocations and idled 32 of them at all seven
barriers); scale hoisted out of the element loop; load_a2() returns both fp16 lanes from one
unpackHalf2x16; paired non-atomic store replacing atomicAnd+atomicOr. No subgroup intrinsic added,
no subgroup size baked — lavapipe reports 8 where both local GPUs report 32.

MEASURED, matched instrument (24 iters, both STEADY, both verdict MATCH):
  RTX 4060 baseline  40.390 ms/inf  STEADY n=23 RSD 0.294%  q_gemv 244.09 us
  RTX 4060 tiled     12.294 ms/inf  STEADY n=13 RSD 0.099%  q_gemv  65.36 us
  = 3.29x total GPU busy, 3.73x on the kernel.
  Iris Xe baseline   q_gemv 3804.85 us  NO_STEADY_TAIL
  Iris Xe tiled      q_gemv  468.32 us  NO_STEADY_TAIL   = 8.1x on the kernel.

I beat my own prediction (18 ms predicted, 12.294 measured). Scoring honestly: I underestimated the
tile's reuse benefit, and I had not anticipated that removing the idle-invocation barrier stalls was
worth a separate ~1.5x on its own.

Two instrument lessons, both of which changed a number I would otherwise have reported wrong.
First, at 14 iterations the tiled build reported a median of 13.610 ms; at 24 iterations the same
build reported a STEADY tail of 12.294. The faster the build the longer the ramp takes in
INFERENCES, so a fixed iteration budget under-reports fast builds. Second, I therefore re-ran the
BASELINE at 24 iterations too rather than compare across instrument settings — it was flat (40.390
vs 40.202), but the ratio I would have quoted was wrong until I checked.

TASK 2 — the 13.5x Intel gap.

Raw Intel ratios are not admissible: my two runs of the SAME build differed 2.65x (468.32 us quiet,
1240.43 us contended). But in the contended run gqa_f16 moved 217.55 -> 563.78 us and
skip_simplified_layer_norm_f16 moved 58.14 -> 117.51 us — neither is mine and neither changed.
Contention is common-mode across the frame; a design change is not. So I normalised our kernel by
an untouched control kernel measured IN THE SAME RUN. Check that it works: the contended and quiet
runs agree to ~10% on both control ratios while their raw q_gemv figures differ 2.65x.

q_gemv/gqa_f16, same run:
                                    NVIDIA   Intel   Intel excess
  baseline                            6.97   19.89      2.85x
  arithmetic + wg sizing (COLS=1)     3.71   10.64      2.87x
  + column tile (COLS=8)              1.62    2.20      1.36x

So 2.85x of the gap was OUR DESIGN, hardware absorbed by the control. The arithmetic changes were
device-neutral (2.85 -> 2.87 unchanged); the TILE is the entire portability fix. Mechanism: the
baseline re-read the whole activation row per output column and ran one full barrier + shared-memory
reduction per output column — both paid where Xe-LP is weakest relative to its ALU (68 GB/s of
shared LPDDR4x, and barrier throughput). NVIDIA had the bandwidth to hide it. That is the shape of a
kernel tuned on the machine it was written on.

Hardware bracket: 8.8x ALU, 4x bandwidth -> a memory-bound kernel belongs in [4x, 8.8x]. Baseline
raw ratio 15.6x sat OUTSIDE it; tiled ~7.2x sits INSIDE. Contention explains the Intel variance; it
cannot explain the level, because the level moved 8.1x from a pure kernel change on an equally busy
box.

NEGATIVE RESULT, and it was my leading hypothesis: forcing the global-atomic store back on Intel
moved the kernel 468.32 -> 465.18 us. Within noise. The atomics were not a bottleneck on either
device. Kept the paired store because it is free, not because it was measured to pay.

Incidental: the Intel COLS=1 ablation reported STEADY (388.943 ms, n=15, RSD 1.38%). Intel CAN
settle. The runs that would not settle were the FAST ones — host jitter is large relative to a short
frame — which points at the per-inference span reconstruction rather than the iGPU clock. Passed to
Niobe.

TASK 3.

(1) Index spaces. Selector 1 was still SPLIT-DEVICE. I first tried making ORT's binding
authoritative — and caught my own regression: it silently relocated --device 1 onto NVIDIA while
still reporting MATCH and 161 claimed nodes. ONLY THE TIMING EXPOSED IT (12.457 ms and NVIDIA-shaped
per-kernel means from a run labelled Intel). An unattributed result wearing a MATCH is worse than a
reported split frame, and Intel is the spec-conformance oracle. Reverted. The fix that works is
devices_to_advertise(): when the env var is set it is a PIN and only that device is advertised, so
ORT cannot bind another and the two spaces become ONE rather than being translated between. It can
only key off the env var because ep.device_index is read in CreateEp, after GetSupportedDevices.
Precedence shipped: explicit selector > ORT binding > best score, divergence logged naming both
spaces, SPLIT-DEVICE left able to fire. Verified (R10, content varies with input): selector 0 ->
SHARED / "NVIDIA GeForce RTX 4060 Laptop GPU", selector 1 -> SHARED / "Intel(R) Iris(R) Xe
Graphics", authoritative spans now the integer 0 on both where selector 1 was UNOBSERVABLE.
Note for the team: phi35.py sets the ep.device_index SESSION OPTION, not the env var, so the harness
does not get the pin.

(2) Leaked-device validation gate written up: the production VkDevice is never destroyed, so the
validation layer's teardown report never runs, so "0 validation errors at shutdown" cannot observe a
leak AND passes. R12 says UNOBSERVABLE, never 0; R13 says ERROR(instrument), never a detection.
Criterion 3 must not be certified by it. ep_messenger_fires_for_planted_fence_leak is unaffected —
it builds and destroys its own Instance+Device, which is the correct model.

(3) Reconciled the 70% spread with Niobe rather than re-deriving. Not a disagreement: my baseline
series shows gpu_steady_tail discarding 5 leading samples at ~49.58 ms before a step to a flat
~40.2. That ~49.6 regime is the ramp; her 0.033% describes the post-ramp regime only and the
instrument says so. My 49.4 and 58.5 were means across regimes; my 83.8 and 71.0 were the contention
window Morpheus has since shown inflated the whole suite. I withdraw the 70% spread as a statement
about kernel variability — it was ERROR(instrument) on my side, a missing regime gate, not a
detection of GPU instability. My series independently reproduces both her step and her level.

Validation: cargo fmt clean, clippy clean, cargo test --release --lib 416 passed 0 failed 2 ignored.
All probe output to bench/results/, nothing in the repo root.

Decisions: switch-gemv-column-tile.md, switch-intel-gap-separated.md,
switch-leaked-device-validation-unobservable.md, switch-kernel-spread-reconciliation-niobe.md,
switch-index-space-unified-by-single-offer.md. Committed on squad/switch, not pushed.

## Session 38 — 2026-08-01 — packed 128-bit loads, and the Intel residual closes

Merged origin/main (5eda83b — Fact Checker's docs/PERF.md and his hardware-clock decision record).

Coordinator relayed three Fact Checker findings: bandwidth predicts only 3.08x of the 13.52x Intel
gap leaving a 4.39x residual that is ours; Intel's 52.0833 ns/tick counter is reference-clock based
and trustworthy so NO_STEADY_TAIL means busy-box not broken-clock; and packed loads plus multiple
accumulators are a stronger gap than the no-subgroups constraint. He asked for a controlled A/B
rather than an assumption.

Note: the relay was timestamped 08:22 and re-listed the three "still owed" items, but all three had
been closed at 09:08-09:14 in session 37 (index spaces verified SHARED on both selectors with the
device NAME varying by selector; leaked-device UNOBSERVABLE record; Niobe reconciliation). Confirmed
rather than redone.

PREDICTION, stated before building: NVIDIA 12.294 -> 11.3 ms total (range 10.5-12.3), kernel 65.36
-> 58 us; Intel kernel 468.32 -> 310 us (range 250-420). Reasoning given in advance: NVIDIA was
already ~70% of peak bandwidth so little to get, Intel is where narrow scattered loads cost most.

CHANGE. InB is now declared uvec4[]. When the (column, block) blob is a whole number of 16-byte
units — spec constant QB_PACKED id 5, from gemv_packed(bits, block_size) — the kernel takes ONE
128-bit load per blob instead of four dependent 32-bit ones, and feeds the four components into four
INDEPENDENT accumulators instead of one serial chain. Two non-optional details: Allocator::alloc now
rounds every buffer to a multiple of 16 bytes, because a runtime-sized uvec4[] only covers
floor(size/16) elements and an unpadded buffer leaves a trailing partial element the shader must
never touch; and the activation array is filled by loops with LITERAL bounds written out per
bit-width, because my first version derived the bound from a spec constant, the driver did not
promote the array to registers, and the kernel got SLOWER. I caught that from a normalised ratio
going the wrong way (1.95 vs 1.62), not from a crash.

THE A/B. The box would not go quiet — I polled 25 minutes and never got six consecutive samples
under 20%, the same wall Niobe hit. Two runs taken under load disagreed with each other (normalised
1.95 then 1.29) with gqa_f16 inflated 4.2x, beyond the range where I had validated common-mode
cancellation; per R13 that is ERROR(instrument) and I refused to score it. Instead I added a runtime
override, ONNXRUNTIME_EP_VULKAN_GEMV_PACKED, purely so the two arms can be INTERLEAVED without a
rebuild, and ran paired reps with the untouched gqa_f16 reported per arm.

  NVIDIA, 3 pairs, controls stable 40.3-41.7 us, all MATCH:
    packed 66.44 / 64.39 / 66.09   scalar 69.19 / 72.35 / 69.22   = 1.07x
  Intel, 2 pairs:
    packed 292.34 / 298.74         scalar 404.94 / 458.91         = 1.385x / 1.327x

The gains are real and disproportionately Intel's, which is what a bandwidth-bound kernel on a
narrow memory pipe predicts. Fact Checker's hypothesis confirmed rather than merely consistent.

MEASURED, both STEADY, both MATCH, device timestamp counter:
  RTX 4060   11.567 ms/inf   STEADY n=7 RSD 0.224%   kernel 64.91 us
  Iris Xe    56.881 ms/inf   STEADY n=5 RSD 1.529%   kernel 297.15 us

First STEADY Intel figure for a fast build on this project. Predicted 11.3 (met, in range) and 310
us on Intel (met — 292-299 in the A/B, 297.15 final). On NVIDIA I predicted 58 us and got ~65: I
OVER-PREDICTED the NVIDIA gain, in exactly the direction my own roofline argument had warned. I
should have trusted the argument over the round number.

THE FINDING. Fact Checker's falsifier was "close the gap and Intel lands near 3x". It does not — it
lands at 4.58x (297.15/64.91). But in those same two runs gqa_f16, which I have never touched, lands
at 4.46x (172.86/38.80). So the bandwidth-only model has a ~1.49x blind spot that is COMMON TO ALL
KERNELS on this machine pair — hardware/driver, not our design. Our kernel is now within 3% of what
an independently written kernel achieves on the same two parts.

Design-attributable excess, q_gemv/gqa within the same run, across three commits:
                    NVIDIA   Intel   excess
  baseline            6.97   19.89    2.85x
  + column tile       1.62    2.43    1.50x
  + packed loads      1.67    1.72    1.03x
Closed, not reduced — 1.03x is inside the control's own run-to-run spread. Proposed back to Fact
Checker as a refinement: bandwidth is the right predictor, but the right FALSIFIER is a control
kernel on the same two parts, not the ratio of datasheet bandwidths, because holding a kernel to
3.08x demands 1.49x that no kernel on this machine achieves.

Cumulative over sessions 37-38: NVIDIA 40.390 -> 11.567 ms GPU busy (3.49x), kernel 244.09 -> 64.91
us (3.76x); Intel kernel 3804.85 -> 297.15 us (12.8x).

SHARED MEMORY, asked directly so answered with the number: q_gemv requests shared float red[1024] =
4096 bytes, FIXED. Sized by a literal rather than by local_size_x*QB_COLS precisely so the
requirement is a static property of the module; gemv_cols clamps wg*cols <= 1024 to keep that true.
4 KiB against Intel's 32 KiB is not an occupancy constraint and there is no 48 KiB assumption
anywhere to lose. Eliminated as a suspect.

No subgroup intrinsic added — Fact Checker's finding that packed loads are the stronger gap is why
none was needed.

Validation: fmt clean, clippy clean, cargo test --release --lib 418 passed 0 failed 2 ignored.
Decision: switch-packed-loads-residual-closed.md. Committed on squad/switch, not pushed.
📌 Team update (2026-08-01T09:53:14-07:00): The EP genuinely executes now — 3 VulkanExecutionProvider fused-node events (~355 graph nodes in one fused node) + 24 CPU per run, 65/65 outputs bit-identical, argmax 30751 matching CPU; coverage figures are execution, not offer. All wall-clock figures including 3.1x/3.7x are withdrawn under R13 pending device-clock measurement. Switch holds exclusive claim on device-clock measurement while agents run in parallel. — decided by Scribe

## Session 39 — 2026-08-01 — the allocator adopts by identity; Tank was right that selector 0 was luck

Merged origin/main (20cb57b). Relay again arrived stale (09:53) re-listing items closed at 09:08-09:14
and the packed-load work delivered at ~11:05 as 538db70; confirmed rather than redone.

TANK'S FINDING, AND MY ERROR. He showed the allocator asks for factory index 1 on BOTH selectors —
it never follows the selector at all. On selector 0 the session also offers index 1, they coincide,
frame reads SHARED. On selector 1 the session offers 0, they diverge, SPLIT-DEVICE. So my §6.5
closure on selector 0 was correct about the TYPE transition (UNOBSERVABLE -> integer 0) and wrong
about its meaning: two independent index choices agreed on this desk. Swap the GPUs and selector 0
breaks instead.

Worse, my session-37 "both selectors SHARED" verification was not the disproof it looked like. I ran
it with ONNXRUNTIME_EP_VULKAN_DEVICE set, and that pin advertises exactly one device, which FORCES
the agreement. It hid the defect on the one path the harness actually uses (ep.device_index session
option). A verification that only exercises the configuration in which the bug cannot appear is not
a verification — that is the same mistake as R11's "every child of record was named, so it looked
closed".

MECHANISM. The allocator's index is the memory-info id of whichever OrtEpDevice ORT bound — constant
across our selector. The session's is the physical index our selector opened — varies with it. No
arithmetic relates them. ensure_registered looked the offer up by the allocator's index and on a
miss STOOD UP ITS OWN SECOND VkDevice. That fallback is the defect, not the report.

FIX, by construction and not by index-swapping. §6.5 gives the EP exactly one VkDevice per (physical
device, EP instance) and acquire_ep_device makes it process-global — so when exactly one device is
on offer, a missed index is a naming disagreement between two spaces, not evidence of a second
device. New pure rule resolve_offer: Exact / SoleDevice (adopt, SHARED) / NoOffer / Ambiguous.
SPLIT-DEVICE stays reachable for the last two — a detector that can no longer fire is worth less
than the defect it reported. I deliberately did NOT chase selector 1: per Tank, "a fix that only
makes selector 1 pass on this box is the same coincidence with a different index."

FALSIFIERS. (1) a_missed_index_resolves_by_identity_in_both_directions asserts resolve_offer(1,[0])
and resolve_offer(0,[1]) resolve IDENTICALLY — his construction test written down. Any fix that
special-cases a direction passes one and fails the other. (2) The runtime artifact taken on the path
that was actually broken — env pin OFF, device chosen by session option, exactly his configuration:
  allocator_index = 1, session_devices = 0=Intel, frame = SHARED, device = Intel Iris Xe
The indices DISAGREE and the frame is SHARED anyway. That is the point: shared because identity
settled it, not because two choices agreed. Same configuration produced SPLIT-DEVICE before.

Flagged to Tank rather than edited (counters.rs is his): alloc_device_frame_sides now says "BOTH
sides are on the same VkDevice: 'Intel Iris Xe' (factory device index 1)" while the session offered
that device under index 0. Device right, parenthetical names the other space — suggest it name both,
since the whole finding is that one number cannot stand for both.

Wrote docs/ENGINE.md §2.0 "Why device indices keep going wrong here" as asked: four defects of one
shape, the structural cause — a physical device is named by four independent authorities
(vkEnumeratePhysicalDevices order, our best-first sorted list, position in the advertised list, and
ORT's bound memory-info id) and ALL FOUR are a bare usize, so the compiler cannot tell them apart
and any two compare without complaint — plus the two rules: a frame that agrees is not a frame that
is correct, and prefer resolving by identity over reconciling by arithmetic.

Validation: fmt clean, clippy clean, cargo test --release --lib 425 passed 0 failed 2 ignored.
Decision: switch-allocator-adopts-by-identity.md. Committed on squad/switch, not pushed.

Still open and mine: transfer.rs::device_buffer_for binding, so alloc_device_authoritative_spans and
alloc_device_buffer_binds can leave 0. Tank delivered the transition, not the feature, and said so.

### Session 39b — the engine binds, and pays the mirror what it now owes

Relay item 3: transfer.rs::device_buffer_for had NO production caller, so alloc_device_buffer_binds
was an honest 0 and device-backed allocation was a cost with no saving. Tank: "I delivered the
transition, not the feature, and the artifact says which."

WHAT I BUILT. New Step 1a in dispatch_ort, before the host resolution: ask
vk::host_device_memory::bind_target_for for each input; when it answers, bind that VkBuffer as a
borrowed ref and skip BOTH the allocation and the upload. Ordering matters — Step 1b overwrites
input_cpu_ptrs[i] with the staging address, so after it there is no handle left to classify.

bind_target_for declines rather than assumes, three ways: (1) the span must have a VkBuffer; (2) the
frame must be SHARED — binding across two VkDevices is undefined and would APPEAR TO WORK on a UMA
part, which is the worst way for it to fail; (3) offset must be 0 and offset+len <= size, because
vk::pipeline writes every VkDescriptorBufferInfo at offset 0, so an interior pointer cannot be
expressed and binding at 0 for one would read the neighbouring tensor the planner put at the base
of the span. Declining costs one upload. Assuming costs correctness, silently.

THE OBLIGATION, IN THE SAME CHANGE. Endpoint's doc justified staging staying authoritative BECAUSE
the session reads inputs and writes outputs through host_backing_for. Bind the inputs and that
asymmetry stops being a design note and becomes a staleness bug: a span written as an output through
staging, then read as an input through its device buffer, is read stale — and stale-but-plausible is
the failure mode that survives a smoke test. So write_outputs_to_ort now calls the new
transfer::mirror_to_device. CopyTensors already mirrors every copy into a handle; this is the same
duty for the one writer that did not go through it.

COUNTING AT THE BIND, NOT THE RESOLVE. device_buffer_for no longer tallies; it returns a
DeviceBinding that now carries the DEVICE INDEX, because a caller given only the view could bind
across devices. A resolve that is then declined is not a bind, and a counter that inflates on the
flattering side is the failure this project keeps repeating.

MEASURED — three configurations, probe_sec65.py, outputs verified per session:
  dev0 NVIDIA DEVICE_MEMORY=1  frame SHARED  binds 6  uploads 9  session_device_allocs 15  OK x3
  dev1 Intel  DEVICE_MEMORY=1  frame SHARED  binds 6  uploads 9  session_device_allocs 15  OK x3
  control     DEVICE_MEMORY off frame OFF    binds 0  uploads 0  authoritative UNOBSERVABLE  OK x3

alloc_device_buffer_binds HAS LEFT 0 — 6 = 2 device-backed inputs x 3 sessions, on BOTH devices.
Two corroborating movements make this an R10 wiring artifact rather than an incremented counter:
session_device_allocs fell 21 -> 15, exactly the 6 allocations the session no longer makes, in a
counter Tank owns and I did not touch; and alloc_device_uploads rose 6 -> 9 (6 MiB -> 9 MiB), the
three new output mirrors — the obligation is observable, not asserted.

WHAT I AM NOT CLAIMING. alloc_device_authoritative_spans is STILL 0 on both devices with 9 residency
evaluations, and I am not moving it by argument. A span stops being a mirror when it stops having
host staging, and staging is still there because every unbound path — outputs, interior pointers,
SPLIT-DEVICE, the whole default build — reads through it. Binding inputs is necessary, not
sufficient. Tank's screen is measuring the right thing and reporting 0 because 0 is the answer.
M1's residency criterion stays open; what closed is the precondition.

Regression check on the real model, NVIDIA, device memory off: verdict MATCH, gpu_steady_tail
STEADY, 12.183 ms/inference GPU busy (n=20, RSD 0.103%), q_gemv 62.18 us mean, 80.66% share. The
shipped default is provably inert — both new call sites take their early return when no registry
exists.

TESTS. an_engine_write_to_staging_is_pushed_to_the_device_mirror asserts the provider received the
bytes the engine wrote AT THE SPAN OFFSET (256), not at 0. a_host_output_is_not_mirrored_and_is_not
_an_error pins the shipped default to Ok(false) rather than an error that would fail every
inference in the configuration actually released. Both go through a new mirror_in taking an explicit
registry map, because the public entry reads factory::all_registries() which is empty under cargo
test — my first attempt passed by taking the early return and proved nothing, which is R10 biting me
in my own test. 426 passed, 0 failed.

Also fixed a duplicate #[test] attribute I left on gemv_packed_tracks_the_blob_and_not_the_block in
session 38 (clippy duplicate_macro_attributes), and its doc comment which had been copied from the
test above it.

Decision: switch-engine-binds-device-buffers.md.

### Session 39c — both clocks from one trace: the discrete GPU ignores contention, the integrated one does not

Coordinator relay: the machine cannot be quiet while agents run — it is his own orchestration, two
copilot processes at ~11,700 s CPU on a 20-core box — so wall clock is structurally unavailable. The
one finding that survived Niobe's refusal was HOST_SIDE_EXCURSIONS: host spread >= 2.0x against GPU
spread <= 1.25x for repetitions of identical work. He asked me to check whether my 70% spread was
host-side before looking for a GPU-side explanation.

BUILT probe_hostgpu.py. One trace carries BOTH clocks for the same work: vulkan.subgraph dur is the
host axis, gpu_ns is the device counter attributed ordinally (never by timestamp — the anchor
reaches 314 ms of uncertainty on Intel). Summing each per inference gives a PAIRED series over
identical work in one process, so the ratio is contention-invariant in the same sense his signature
is. Inference 0 reported separately, never averaged in — folding a known regime into a "spread"
lets one known event stand in for variability. Re-runnable over any existing trace; needs no new run,
which matters when no run can be quiet.

RESULT 1 — the cold excursion is ENTIRELY host-side. NVIDIA cold host 1619-2192 ms against cold GPU
12.2-12.8 ms with warm GPU 12.0-13.8. Intel cold host 1443-2456 ms against cold GPU 55-75 with warm
55.6-83.9. The first inference costs 1.4-2.5 SECONDS of host and essentially nothing extra on the
device. Pipeline/shader compile is host work. That is the extreme HOST_SIDE_EXCURSION, fully
accounted for on both parts.

RESULT 2 — warm spreads, host vs GPU, SAME inferences. 15 traces, 3 builds, 2 devices.
  NVIDIA host/GPU ratio: 1.13 1.10 2.07 0.93 1.52 1.34 1.43 1.23 2.27   median 1.34
  Intel  host/GPU ratio: 1.05 1.01 1.01 0.98 1.14 1.01                  median 1.01
On the discrete part the host spread EXCEEDS the GPU spread, up to 2.27x. On the integrated part
they are equal to within 1-5% in ALL SIX runs. The iGPU's device clock inherits host contention
essentially 1:1; the discrete part's does not. I independently reproduce his signature — ab_p0_r1
(host 2.260 / GPU 1.091) and bindseam (host 2.499 / GPU 1.099) both clear >=2.0 against <=1.25.
The one NVIDIA ratio below 1 is ab_p0_r2, a single isolated inference at 19.705 ms against a body of
13.83 +- 0.02 — a real one-off device excursion, not spread.

RESULT 3 — between-run reproducibility of the steady tail, same build, separate processes.
  NVIDIA baseline (2): GPU 1.0047x  host 1.017x
  NVIDIA p0 arm  (3):  GPU 1.0016x  host 1.170x
  NVIDIA p1 arm  (3):  GPU 1.107x   host 1.066x   <- contains a device step 13.33 -> 12.05 whose
                                                     ONSET INDEX VARIES (14, 10, 13). A power/clock
                                                     regime change, and exactly why gpu_steady_tail
                                                     refuses two of the three. Instrument correct.
  Intel  p0 arm  (2):  GPU 1.117x   host 1.109x
  Intel  p1+final(3):  GPU 1.027x   host 1.090x
NVIDIA steady device time reproduces ACROSS PROCESSES to 0.16-0.47% while the host number for the
same runs moves up to 17%.

CONSEQUENCE. NVIDIA device-clock numbers do not need a quiet machine; Intel ones do; wall clock
needs it on both. This is why Niobe gets 0.033% RSD on NVIDIA and NO_STEADY_TAIL on Intel from one
instrument — neither is a defect and neither needs a workaround. Fact Checker is right that the tick
is trustworthy and the work-per-tick is not; this quantifies the second half at 1.01 coupling.

THE RECONCILIATION, THIRD ASKING, NOW WITH DATA. His hypothesis is FALSIFIED for my number: 49.4/
83.8/71.0/58.5 were gpu_ns figures, not host. Worth checking anyway, because the phenomenon it names
is real and is 100% of the cold excursion. What the data does say:
 (1) 49.4 IS THE RAMP LEVEL, not a run. Two independent baseline traces both show 49.58/49.59 before
     a step to 40.2. I reproduce 49.4 to within 0.4%.
 (2) The ramp LENGTH is not reproducible: 5 inferences in one process, 3 in the other, same build,
     same box. So any whole-run mean lands somewhere on [40.2, 49.6] by where the ramp ended. That
     is an estimator property and it is what gpu_steady_tail exists to refuse.
 (3) The steady level reproduces to 0.47%. Her 0.033% within-run and my 1.0047x between-run are the
     same claim about the same regime. There was never a disagreement to settle.
 (4) 83.8 and 71.0 exceed EVERYTHING in a 15-trace corpus across three builds, two devices and two
     orders of contention. I cannot reproduce them and I will not explain them by argument. That run
     had MATCH and claimed_nodes 353 but NO executed_by key — UNATTRIBUTED under Trinity's
     _verdict.py, which now refuses that shape at construction. Terminal state INADMISSIBLE, not
     explained, and it always was.
I withdraw "70% spread" permanently. Its device-side content is a 1.233x ramp step with varying
onset; the rest is an inadmissible run and a mean taken across two regimes.

WHERE THE NEXT WIN IS, AS A BOUND. NVIDIA GPU busy is 12.18 ms/inference after tile+packed. In the
same runs the LEAST CONTENDED single inference had a host span of 28.3 ms. Host is inflated by
contention and GPU is not, so this is a bound in one direction only and I state it as one: the host
span exceeds GPU busy by AT LEAST 2.3x even in the quietest inference measured. After a 3.49x kernel
win the EP is host-bound on this machine by at least that factor, and the next order of magnitude is
not in q_gemv. This is the point at which his offer to idle the team is worth taking — not to
re-measure the kernel, which provably does not need it, but to find out whether that >=2.3x is
contention or ours.

Checked the Scribe deletion: all five of my earlier records are present in decisions.md (Switch
attributions at lines 42, 55/62, 121, 194 — including switch-leaked-device-validation-unobservable,
which is relay item 4 and is already merged, not owed). Nothing of mine was lost; nothing resubmitted.

Decision: switch-host-gpu-decoupling-measured.md.

## Session 42 — 2026-08-01 — counts over clocks; the index space closes; the A/B does not

Merged `origin/main` (`16f40ef`), which brought Niobe's `bench/device_state.py` certification gate.

**Relay #7's four asks, and where each landed.**

**1. Packed loads restated in counts — DONE (`7a1d12f`).** `bench/results/probe_gemv_counts.py`
compiles `q_gemv.comp`, freezes the spec constants per arm, optimizes, and walks the SPIR-V
def-use graph from the `inb` variable to every load reaching it. The load *type* is the claim:
`%uint` (4 B) unpacked, `%v4uint` (16 B) packed, so **4 loads per 16-byte blob become 1**. From
the ONNX graph: **161 MatMulNBits nodes** (matching the trace's 161 dispatches — two instruments
sharing no code agreeing on one integer), **116,324,352 blobs/inference**, so **465,297,408 ->
116,324,352 InB load instructions**. Byte model: weights 1775.0 MiB + scales 221.9 MiB
(irreducible) + **activations 887.5 MiB, 30.8%, ours** — total 2884.3 MiB against 9096.7 at
`QB_COLS=1`, a **3.15x** count-derived reduction where the tile measured 3.73x on the clock.
Shared memory struck off the Intel candidate list by count: 4 KiB requested, 12% of Intel's 32 KiB,
sized by a literal so it cannot move with the tile.

**I demoted half of my own `538db70` claim.** The four 32-bit loads do NOT depend on each other
and can all be in flight; only the four `bacc[c] +=` updates are serialized. Accumulator RMW per
blob 4 -> 1, serial FP adds on the critical path 4 -> 3. Real, countable, nowhere near 4x. The
byte model also predicts packed loads move **zero bytes**, so their ceiling on a bandwidth-bound
kernel is small — consistent with the 1.06x measured, and it should be shown to anyone claiming
more.

**Instrument self-defect, caught before publishing.** First version reported the same census for
both arms: glslc emits `if (QB_PACKED == 1u)` as an `OpSpecConstantOp`, which `--freeze-spec-const`
does not fold and dead-branch elimination will not look inside, so both arms compiled identically.
A perfectly stable, perfectly wrong answer — the same shape as `STEADY` at 21.4x. Fixed with
`--fold-spec-const-op-composite`; `arms_must_differ` now refuses a census whose arms agree.

**2. Certified NVIDIA A/B — ATTEMPTED THREE TIMES, NOT OBTAINED.** All three committed with
companions (`06c1242`): `ab_p0_r1` FOREIGN_GPU_WORK -> **WITHHELD**; `ab_p1_r1` and `ab_p1_long`
**SOLE_TENANT** (0% of 134 and of 327 samples) but `MARGINAL_TAIL` -> **UNCERTIFIED**. **The last
two are the finding: the box WAS quiet and the figure is still uncertifiable**, so tenancy and
certifiability are different properties and idling the squad would not have bought a number.
What refuses them is the board's clock — median SM 210 MHz, median util 0%, median power 2.9 W,
brief 2010 MHz excursions against 3105 max. Our EP is host-bound, so the board never holds boost
and the series drifts (74 -> 19.6 ms over one run), failing the coverage floor. **The duty-cycle
mechanism again, now blocking certification instead of inflating an archive.** No median quoted.

**3. What 1.03x established — my read is in the reply and in the records.** Short version: the raw
medians are UNCERTIFIED; the control-kernel ratio (`q_gemv/gqa_f16`, 1.716 NVIDIA vs 1.727 Intel)
is a *within-run* quantity that a clock change scales on both sides, so it survives without a
device-state companion. The coordinator's two Intel `NO_STEADY_TAIL` arms neither confirm nor
refute it — they are an absence.

**4. Index space by artifact on BOTH selectors — CLOSED (`06c1242`).**
`bench/results/probe_indexspace.py`. Criterion a coincidence cannot satisfy: the allocator index
must **differ** between selectors and match the session's offered index in each. Result:
selector 0 `allocator_index='1'` offered `1=NVIDIA`; selector 1 `allocator_index='0'` offered
`0=Intel`; **both SHARED; verdict ONE_INDEX_SPACE.** Pre-fix it was `'1'` on both.
**`alloc_device_buffer_binds = 6` on both selectors** — Tank's counter has left 0 and
`device_buffer_for` is invoked. `alloc_device_authoritative_spans` stays 0 by design (all 9 spans
still mirrored). Added the two-armed-artifact rule to `ENGINE.md` §2.0.

**Inadmissible today, labelled:** every GPU-busy figure I touched this session. Nothing from
`ab_p0_r1`, `ab_p1_r1` or `ab_p1_long` is quoted. Note my uncertified archive is now positively
suspect: today's flat suffix sits near 19.6 ms where the archive says 11.5, at a lower peak clock.
That distrusts the archive; it does not license quoting 19.6.

**Next step, for a reader with no memory of this session.**
1. **Do not chase a certified NVIDIA figure by waiting for a quiet box.** It is already quiet and
   still refuses. Either get `nvidia-smi --lock-gpu-clocks` + an elevated shell (blocked on
   Justin), or change the harness so the GPU holds boost — the latter is in our hands and is the
   better first move. The gate is correct in all three runs; do not touch it.
2. **The next kernel change should be chosen by the byte model, not by a clock.** Weights and
   scales are irreducible; **activations are 30.8% of traffic and ours**. Raising `QB_COLS` past 8
   needs a bigger register budget or a shared-memory activation tile — and shared memory is at
   4 KiB of a 32 KiB Intel budget, so there is room. Predict the byte reduction from
   `probe_gemv_counts.py` first; it was accurate to 18% for the column tile.
3. Decision records filed: `switch-packed-loads-in-counts.md`,
   `switch-sole-tenant-is-not-certifiable.md`, `switch-index-space-one-space.md`.

📌 Team update (2026-08-01T17:16:56-07:00): Intel device-clock figures are permanently uncertifiable on this hardware (`none_available`, no producer exists and none of the available proxies are the right kind of quantity) — attack the Intel/NVIDIA residual with counts and shapes, not clocks — decided by Niobe


📌 Team update (2026-08-01T17:16:56-07:00): All wall-clock figures remain withdrawn; only counts, bytes and certified-companion device-clock figures are quotable — decided by Switch, Morpheus, Niobe, Link


📌 Team update (2026-08-01T17:16:56-07:00): `ledger_lookup` is the last `UNWIRED` mechanism in the instrument census (criterion 11); Mouse is building it — decided by Trinity, Mouse

   `switch-sole-tenant-is-not-certifiable.md`, `switch-index-space-one-space.md`.

---

## Session 43 — 2026-08-01 — the two owed items: the leaked-device cost, and the frame reconciliation

Both delivered. Neither needed the GPU, which was the point (three agents measuring concurrently).

### 1. The 70% spread vs the steady tail — `SAME_FRAME_ORDERED_SELECTION`

Artifact: `bench/results/probe_frames.py` + `bench/results/frames.json`.

The coordinator offered two cases (different frames both right / same frame one wrong). **Neither.**

**Frames, read from source not names (R11):** Niobe's `busy_us[i]` (`phases.py:890-928`) is the sum
of `gpu_ns` over every kernel span of inference i; her tail is a **suffix** of that series. My 70%
(history.md:1401-1411) was gap-clustered per-submission sums over all kernels — **the same
quantity**, taken over the **whole** series. Frames coincide. The difference is *selection*.

**The decisive move was refusing to compare two numbers.** A single whole-RSD-vs-tail-RSD ratio is
an anecdote. Instead: *does any run publish a tail figure AND carry a large whole-series spread?*
Census over all 28 committed dev0 traces, calling Niobe's own functions:

- 9 traces with whole-series RSD >= 30% -> **0 publish**. All NO_STEADY_TAIL or MARGINAL_TAIL.
- 12 publishing traces -> max whole-series RSD **10.36%**.
- Sets disjoint, gap 10.36% -> 34.39%.

**My own 70% run is `trace_gemv_notile_dev0.json`: whole 73.22%, tail `NO_STEADY_TAIL`.** On the
exact run that produced the figure, her instrument publishes nothing. There is no number of hers to
disagree with mine. Falsifier retained: a trace with both properties returns `CONFLICT`.

**Corrected the coordinator's per-kernel hypothesis.** He said "the variance averages out". Right
direction, wrong mechanism, and the difference is useful. Steadiest trace: within one inference RSD
**37.90%**, spread 10.1x, **3 discrete duration clusters**; same ordinal across inferences RSD
**0.36%**. That is population heterogeneity across node shapes, not variance — there is nearly
none to average. Averaging predicts sum RSD 37.90%/sqrt(161) = 2.99%; observed is far below, which
is the signature of a *deterministic* spread.

**Control, and it flips:** most-disturbed trace has the same within-inference structure (35.23%, 3
clusters) but same-ordinal-across-inferences RSD **142.41%**. So **same-ordinal RSD is a per-kernel
discriminator between a clean and a disturbed run (0.36% vs 142%) that the per-inference sum
hides**, and it needs no cross-run clock comparison. Worth remembering — it may be the cheapest
in-band disturbance check we have.

**Bug I shipped and caught:** rev 1 hard-coded "heterogeneity, not variance" into the output string,
then printed it over a contended trace whose numbers said 173% variance. *A conclusion that survives
its own refutation is not a measurement.* Now derived (`HETEROGENEITY_DOMINATED` /
`VARIANCE_DOMINATED`) and run on two deliberately chosen traces rather than whichever came first.
The control above only exists because that bug forced it.

### 2. The leaked device — cost priced, recommendation: **keep it**

Mechanism was already recorded twice (decisions.md:62, and Trinity's frame split at :446-450). What
was missing was the price and an explicit decision to pay it.

**Cost is O(1) per physical device per process, not O(sessions)** — bounded in the types:
`EP_INSTANCE` is one instance, `EP_DEVICES` one owner per physical index (`vk/device.rs:154-159`).
Corroborated independently: device high-water FLAT ~3.907 GiB across 3 sessions (would be ~3x or
OOM if it scaled). **The leaked thing is a device handle, not the memory the device allocates.**

**The real cost is one lost observation window**: the layer reports leaks at `vkDestroyDevice`,
which never runs on production, so any "0 validation errors at shutdown" gate is `UNOBSERVABLE`,
never `0`. Bought back by the planted-leak positive control (owns and destroys its own device) plus
Trinity's dispatch-window frame of record. **Accepted residual**, stated rather than papered over:
a leaked object that never trips a dispatch-time VUID is not caught in-process; the compensating
instrument is device high-water across sessions, which is weaker than the layer.

**Recommendation: keep the leak.** The alternative is the use-after-free we fixed, not a cleaner
shutdown. If anyone wants the window back, do it in a subprocess that owns its own device — do not
un-leak production.

### Records filed (absolute path to main's inbox; gitignored inside worktrees)

- `switch-leaked-device-validation-cost.md`
- `switch-two-frames-one-series.md`

### Next step, for a fresh session with no memory of this one

Nothing here is blocked. The open work is unchanged and is all GPU-bound, so it waits on a window
where fewer agents are measuring:

1. **The certified NVIDIA A/B for packed loads.** Attempted 3x in session 42, never obtained —
   `SOLE_TENANT` twice and still `MARGINAL_TAIL` both times, so **tenancy is necessary but not
   sufficient for certifiability**. The board sits at 210 MHz median because our EP is host-bound
   and never holds boost. Do not quote the uncertified archive (11.5673 ms); today's flat suffix
   read ~19.6 ms at a lower peak clock, which makes the archive *positively suspect*.
2. **The packed-loads claim stands in counts** (session 42, `7a1d12f`) and does not need a clock:
   465,297,408 -> 116,324,352 InB load instructions. That is the quotable form.
3. **`gpu_steady_tail` under foreign GPU work** — still untested, still the right question, still
   needs a window.
4. Consider promoting the same-ordinal-across-inferences RSD (finding 1 above) into `phases.py` or
   a probe as a first-class disturbance check. It is cheap, in-band, and discriminates 400x.

---

## Session 44 — 2026-08-01 — the load guard, built on same-ordinal RSD

Coordinator merged session 43 as `c6cc0f3` and reproduced `probe_frames.py` independently. New
task: build the long-pending "refuse benchmarks on a contended machine" guard, using the
same-ordinal RSD I stumbled onto while reconciling the frames.

Landed: `bench/run_disturbance.py`, `ci/check_run_disturbance.py`,
`bench/test_run_disturbance.py` (14 tests), `bench/results/run_disturbance_dev0.json`.

### The threshold, and where it does NOT separate

Bimodal across all 28 dev0 traces with an EMPTY gap: 19 traces 0.624%..10.507%, then nothing, then
9 traces 35.313%..137.352%. `DISTURBANCE_RSD_MAX = 0.20` sits in the gap (1.90x above the highest
clean, 1.77x below the lowest dirty). A test pins that both arms hold for any threshold in
0.11..0.35, so no verdict rests on the constant.

**But it does NOT separate the STEADY/refused populations** — publishing 0.624%..10.507%, refused
3.694%..137.352%, substantial overlap. It predicts *stationarity*, not whether the tail publishes.
Said so in the docstring and the record rather than letting the bimodality imply more than it does.

### It adds ZERO refusals today, and I measured that rather than assuming it

`--corroborate`: 9 flagged, 9 already refused by the tail's floors, **0 that the tail would
publish**. So it corroborates, it does not protect — today. Value is elsewhere: it refuses
independently of suffix selection; it is the check that survives if a MARGINAL_TAIL's withheld
median is ever published; and two statistics over two frames agreeing on nine runs is evidence.

The coordinator's catch, worse than he put it: `contended` is the dirtiest run in the census
(137.35%) and its **tail RSD is 0.1067%, third tightest of all 28, tighter than
`baseline_certified` at 0.1163%**. This statistic is 6.9x over its bar on that run. R9 amendment 5,
third independent appearance.

It also refuses **my own** `packed`/`packed2` A/B traces at 53.4%/53.2%. Correct outcome.

### The frame — the one I was asked for does not work, and that is pinned

Asked for an in-band measurement *of the run being certified*. Tried the tail's own suffix first
because a companion ought to cover the same inferences. **No discriminating power there:**
`contended` restricted to its suffix reads **3.07%**, inside the publishing range (0.62%..3.97%).
The tail's selection has already found a quiet window. So the guard measures the WHOLE run and its
claim is scoped: *a statement about the run, not the suffix cut from it.* That is exactly its value
— it can say the suffix was carved out of a violent run, which nothing computed inside the suffix
can say. Negative result pinned by `test_the_tail_suffix_frame_would_NOT_work_...`.

### The hole is real and is now a passing test, not a caveat

A uniformly slow run PASSES. `synthetic_uniform_slowdown()` builds it and
`test_a_uniformly_slow_run_PASSES_and_that_is_the_documented_hole` asserts the pass (plus that the
run really is 2x slower, so the pass is not empty-input). **PASS means "repetitions agreed", never
"the machine was quiet"** — the CLI prints that on every pass.

### Obligation 8: complements, does not subsume

Orthogonal halves, and the hole is why. Obligation 8 catches a wrong *level* (board pinned at
210 MHz, uniform inflation) — exactly what this is blind to. This catches a non-stationary *run* at
whatever clock. We have observed a run perfectly stationary at 21.4x wrong with better RSD than the
correct one. Coverage floor is a third question again (how representative the suffix is). Three
instruments, three failure modes; no two collapse.

### Drift vs jitter — the reason this is not just whole-series RSD

`baseline` (whole 10.36%, ord 10.51%) vs `ab_p0_r2` (whole 10.97%, ord 3.82%): same drift, 2.75x
different jitter. Per-inference spread conflates them; same-ordinal RSD isolates jitter.

### Next step for a fresh session

1. Guard is landed and self-contained; no GPU needed to re-run it.
2. Consider wiring `check_run_disturbance` into whatever lane first publishes a duration — it is
   built for it (`--scan`, exit 0/1/4) but is not wired to any lane, deliberately: no lane
   publishes a timing figure today. Same rationale as `check_device_state` living in `ci/`.
3. Still open and GPU-bound: the certified NVIDIA A/B for packed loads (3 attempts, never
   obtained — tenancy is necessary but not sufficient); `gpu_steady_tail` under foreign GPU work.
4. The packed-loads claim stands in counts (`7a1d12f`) and needs no clock.

---

## Session 44b — 2026-08-01 — clippy green; index fix HELD pending Tank's MIXED

### DO NOT LAND THE DEVICE-INDEX RESOLUTION FIX YET

Coordinator-enforced sequencing from Tank. Read this before touching
`vk/host_device_memory.rs::OwnedDevice::create`.

**The defect (fourth face of the index space, mine):** `create(device_index)` takes the factory
index and then *ignores* it — line ~274 resolves via `crate::vk::instance::select_device(&capables)`,
which reads the process-global `ONNXRUNTIME_EP_VULKAN_DEVICE` and therefore returns the SAME index
for every key. So `ensure_registered(0)` and `ensure_registered(1)` stand up the same physical
device, and the `PROVIDERS` HashMap key is inert. Verified by reading, and by Tank's
`bench/results/two_device_frame_probe.txt` (both indices -> SPLIT-DEVICE on the RTX 4060).

**Why the order matters, and it is not arbitrary:** today two providers collapse onto one physical
device, so a genuine two-*device* population cannot form and the single frame label is wrong in
principle but not in fact. The moment the index actually selects, providers become genuinely
distinct and every `alloc_device_*` aggregate in a two-provider run describes a population drawn
from two `VkDevice`s while carrying one frame label chosen by registration order. That is an R12
violation shipped *by a correctness fix*. Tank's `MIXED` label must land before or with it.

**My recommendation, given to the coordinator: land together, not serialised.** Reason is R10.
Until my fix lands, `MIXED` cannot be produced end-to-end on this box — it would land with a
positive arm that is unfalsifiable in fact (synthetic unit construction only). Landing together
lets me supply the real artifact: a two-provider run on two physical devices producing `MIXED`,
with the single-provider control producing `SHARED`/`SPLIT-DEVICE` and not `MIXED`.

**Finding for Tank, from reading the only production caller:** `ensure_registered` is called from
exactly one place, `allocator.rs:595 try_attach_device_buffer`, keyed on that allocator's own
`device_index`. So two providers only register if one process stands up allocators for two
different device indices. But my §6.5 fix made `devices_to_advertise()` advertise **only** the
pinned device when `ONNXRUNTIME_EP_VULKAN_DEVICE` is set — so:

  - **pinned**: one device advertised, one index reachable, `MIXED` is correctly UNOBSERVABLE.
  - **unpinned**: all capable devices advertised, ORT may create sessions on both -> `MIXED`
    reachable, but only after my fix makes the providers genuinely distinct.

So `MIXED`'s falsifier needs an **unpinned two-session run**. Worth Tank knowing before he writes
the control, because a pinned probe can never produce it and would read as a silent pass.

### clippy, red for the life of the project, now green (`9b2a916`)

`cargo clippy --release --all-targets -- -D warnings` failed on four items, all test-profile, all
mine. Invisible because `cargo build --release` is clean — a lint gate whose failure cannot be seen
from the step beside it.

- `transfer.rs:1103` unused `DeviceMemoryProvider` import (genuinely dead — inherent methods used).
- `quant.rs:888` manual range contains -> `(1..=GEMV_MAX_COLS).contains(&cols)`, and it was
  asserting bare, so it got the failure message it never had.
- `norm.rs:270`, `indexing.rs:118` function item cast to usize -> `std::ptr::fn_addr_eq` against
  the row's own `TranslateHandler` type, which is what the comparison always meant.

**Both fn_addr_eq tests gained the negative polarity they never had**: an address comparison that
can only return true is not a check, so each now also asserts the row is NOT
`templates::unimplemented`. Both arms pass, so the comparison discriminates.

440 lib tests pass at **default parallelism** — not `--test-threads=1`. Tank's reasoning applies to
me too: serialising would have hidden a shipping defect with the identical symptom and the opposite
fix. I had used `--test-threads=1` to characterise the counter flake in session 42; that was a
diagnostic, and it should not become how the suite is run.

---

## Session 44c — 2026-08-01 — the PROVIDERS key now selects a device (`1d2a663`)

Sequencing block lifted: Tank's `MIXED` landed on main at `ca283a9`. Merged (one conflict in
`transfer.rs` — he added a function-local `DeviceMemoryProvider as _` where I had removed the
file-level import; took his side, then clippy caught that one of his two local imports was itself
unused because that test calls inherent methods on the concrete `RecordingProvider`).

### The fix

`PROVIDERS` is a `HashMap` keyed by the factory's advertised device index; `OwnedDevice::create`
resolved through `select_device` and never read its argument. Every key produced the same device.

**The previous repair was not careless and it is worth understanding why.** Indexing the best-first
sorted list *with* a physical enumerate index had put the mirror on the wrong GPU (Tank, 07-30), and
ignoring the index fixed that. But *"don't index with it"* and *"don't read it"* are different
instructions, and only the first was needed. The two spaces meet at `position_of_physical` — the
same seam `VulkanSession::create` already uses for ORT's binding — so translate, don't ignore.

Split out `resolve_provider_position` with no Vulkan in it, returning
`Translated(pos)` / `Untranslatable(pos)` rather than a bare `usize`. **The two-variant return is
the point**: the fallback position is the same number the old code always produced, so collapsing
them leaves a caller unable to distinguish a working map from an inert one — which is the state we
were in for a week.

### Scope — state this precisely, it is narrower than it sounds

`create` is reached only on `ensure_registered`'s `NoOffer`/`Ambiguous` arms. When a session has
offered a device, the §6.5 `SoleDevice` identity rule adopts it and `create` never runs. **So this
does NOT change `alloc_device_frame` on the ordinary pinned production path**, which was already
`SHARED` by identity. What it changes is the fallback path — which is exactly where Tank's probe
landed (no session ⇒ `NoOffer` on both indices).

### Falsifiers — three of four fail on the old code, and I checked rather than assumed

Restored the old semantics temporarily and re-ran:
`distinct_provider_keys_resolve_to_distinct_devices`, `a_key_that_names_no_capable_device_...` and
`agreement_between_the_two_spaces_is_permitted_but_never_relied_on` **FAIL**.

`a_pinned_offer_translates_onto_the_selectors_own_position_on_both_selectors` **PASSES on the old
code** — because in the pinned case the right answer *is* `selected`, which is what the old code
always returned. It is kept (§6.5 depends on the invariant) but its doc comment now says in as many
words that it is **not evidence for this fix**. A test whose polarity I have not checked is a
printed opinion, and this is the second time this week I have caught one of mine.

### The artifact — `bench/results/provider_key_selects_probe.txt`

`probe_provider_key_resolution` enumerates capable devices and prints the table. It creates no
logical device and dispatches nothing, so it is admissible on a contended box — counts and
identities, not timings.

```
best-first position 0 -> physical enumerate index 1 -> 'NVIDIA GeForce RTX 4060'  <- select_device
best-first position 1 -> physical enumerate index 0 -> 'Intel(R) Iris(R) Xe Graphics'
key 0: was NVIDIA (old rule) -> now Intel   [Translated(1)]
key 1: was NVIDIA (old rule) -> now NVIDIA  [Translated(0)]
2 key(s) -> 2 distinct device(s).  Old rule: 2 key(s) -> 1 device.
```

**The two index spaces are inverted on this desk in fact, not hypothetically** — the RTX is physical
1 / position 0. That is why every constructed test above uses `[1, 0]`.

And Tank's detector fires. His probe read `SPLIT-DEVICE`/`SPLIT-DEVICE` with *"this box cannot
exhibit the mixed frame"*; on the same build it now reads:

```
device_index 0: frame now "SPLIT-DEVICE", 1 declared
device_index 1: frame now "MIXED", 2 declared
FINDING: production declared 2 frames in one process.
```

**The sequencing constraint was right and this is the evidence.** `MIXED` landed first; this change
made it observable in production code within the hour rather than by unit construction. Had the
order been reversed, a two-provider run would have described a two-`VkDevice` population under a
single frame label — an R12 violation shipped *by a correctness fix*.

503 passed / 0 failed. `cargo clippy --release --all-targets -- -D warnings` green.

### Also: `ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY` default re-justified (record filed)

Morpheus ruled the default intended but not re-decided. The recorded reason ("the handle→VkBuffer
seam is not filled") **has expired** — the seam is filled, 6 binds measured on both selectors.

The real blocker is one level up and the docs conflate them: `vk::session` still binds buffers it
allocated itself and reads through `host_backing_for`, so the device buffer stays a mirror.
`alloc_device_authoritative_spans` reads **0**, and that zero is *measured* — `allocator.rs:681`
evaluates every device-backed span at free against `buffer.is_some() && staging.is_none()`, and
counts evaluations separately from outcomes, so a measured zero is distinguishable from an unwired
one.

Recommendation: **keep it off, with the reason replaced.** New reason: turning it on pays full cost
(device allocation + a bus write per copy-in) for a benefit no counter can currently report.
Second, independent reason: the leaked production device makes the shutdown-validation gate
`UNOBSERVABLE`, and device-backed allocation is precisely the path that gate would police — enabling
it by default while its safety net cannot fire is the wrong order. The two reasons should not be
collapsed; if the residency work lands, the second still stands.

Filed to main's inbox: `switch-device-memory-default-rejustified.md`. Needs Tank's half (counter
surface readiness off this desk) before M2 entry.

### Next step for whoever resumes

1. **The named change that flips the default**: make `vk::session` bind `device_buffer_for`'s buffer
   instead of allocating and re-uploading its own, then run with the flag on and read
   `alloc_device_authoritative_spans`. **Non-zero is the entire argument.** Engine-side, mine.
2. Still GPU-bound and waiting on a quiet window: the certified NVIDIA packed-loads A/B (attempted
   3x, never obtained — `SOLE_TENANT` twice yet still `MARGINAL_TAIL`), and *does foreign GPU work
   move `gpu_steady_tail()`?*. Do not quote the uncertified 11.5673 ms archive.
3. The packed-loads claim stands **in counts** (`7a1d12f`: 465,297,408 -> 116,324,352 InB load
   instructions) and needs no clock, so it is quotable as-is.
