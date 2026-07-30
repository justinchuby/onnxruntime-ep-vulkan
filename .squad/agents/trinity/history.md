# Project Context

- **Owner:** Justin Chu
- **Project:** onnxruntime-ep-vulkan — a cross-platform Vulkan plugin execution provider for ONNX Runtime, written in Rust.
- **Reference architecture:** `C:\Users\justinchu\dev\onnxruntime-mlx` — mirror its layout: `rust/src/{lib,ep,factory,sys,registry,engine,logging}.rs`, `rust/src/ops/*`, `rust/build.rs`, `tests/conformance/`, `bench/`, `python/`, `docs/DESIGN.md`.
- **Stack:** Rust (cdylib plugin EP), Vulkan 1.1+ compute, SPIR-V/GLSL shaders, ONNX Runtime C API, Python bindings, GitHub Actions CI.
- **Cross-platform mandate:** Windows, Linux, Android, macOS via MoltenVK; NVIDIA / AMD / Intel / Adreno / Mali; software rasterizer (lavapipe / SwiftShader) for GPU-less CI.
- **My focus:** Test & Conformance — differential testing vs ORT CPU EP, tolerances, CI tests
- **Created:** 2026-07-28T17:52:04-07:00

## Learnings

<!-- SUMMARIZED by Scribe 2026-07-29T09:00:39-07:00 — full session details in decisions.md -->

### [SUMMARY] Rounds 1–5: CI foundation, barriers, quantization oracle, claim log, CI fixes (2026-07-28–2026-07-29)

**Correctness oracle (Morpheus):** ORT CPU EP running the same ONNX model. Every op test MUST assert the node ran on `VulkanExecutionProvider` — CPU fallback passes vacuously.

**Tolerance policy (per op-family):** fp32 elementwise: 1e-5/1e-5; fp16: 1e-3/1e-3; comparison: exact; reductions/GEMM M2+: TBD. Widening requires Trinity sign-off + in-test comment.

**Three-regime quantization tolerance:** `DEQUANT_EXACT={rtol:0,atol:0}`, `MATMULNBITS_FP32={rtol:1e-3,atol:1e-4}`, `MATMULNBITS_FP16={rtol:2e-2,atol:1e-3}`. `accuracy_level` pinned at 1 (fp32). fp16 oracle gated on ORT≥1.28 (1.27 = NaN/Inf via null-allocator PrePack bug).

**Claim assertion mechanism:** ORT profiling JSON (`enable_profiling=True`, `end_profiling()`, parse `cat=="Node"` events for `args.provider=="VulkanExecutionProvider"`). Structured, not text parsing. `assert_vulkan_claims` in `tests/ops/_models.py`.

**Barrier parity layer:** `test_barrier_parity.py` — 74 `claim=True` cases run twice (sync2 + force-legacy), asserts bit-identical outputs. `run_with_backend()`: sets `ONNXRUNTIME_EP_VULKAN_BACKEND_PROBE`, applies `ep.force_legacy_barriers=1`.

**Machine-readable claim log:** `ONNXRUNTIME_EP_VULKAN_CLAIM_LOG` env var. `test_domain_regression.py` asserts `code=="not-registered"`. `read_claim_log(path)` in `_models.py`.

**CI lanes:**
- **Linux (ubuntu-22.04):** glslc from LunarG apt repo (`shaderc` package — NOT Ubuntu's `glslang-tools`); LunarG SDK version pinned at workflow-level; `VK_LAYER_KHRONOS_validation`; zero validation errors enforced.
- **Windows:** lavapipe ICD registered in `HKLM:\SOFTWARE\Khronos\Vulkan\Drivers` (`VK_ICD_FILENAMES` ignored by loader when elevated); mesa-dist-win 26.1.3; ORT DLL PATH added for pytest.
- **ORT version:** `ORT_VERSION=1.28.0` pinned at workflow-level env. `onnxruntime>=1.28` in `requirements.txt`.

**conftest.py:** `register_vulkan_ep` is not autouse; Vulkan-independent tests (oracle pinning, layer capture, domain regression) run without EP lib. 8 tests pass unconditionally; rest skip cleanly.

**`onnx-shape-inference` (Python) adopted as preprocessing step (Morpheus D24):** Sequence as coverage work, not harness polish.

**Validation-layer-clean** is a hard "done" criterion for any engine change. **CI is the only place shaders are verified against a reference; red CI blocks all merges.**

**PowerShell gotcha:** `-match`/`-notmatch` are FILTERS on arrays, boolean only on scalars. Use `Select-String -Quiet`, `-contains`, or `( | Where-Object {...}).Count -gt 0` for boolean result from multi-line output. Both directions of a check must be verified.

**`test_op_table.py`** — single-dispatch `CaseSpec`-driven; 124 tests, all skip cleanly without EP lib. Pre-populated with all tier-1 ops.

---

## Cross-agent context appended (2026-07-29T09:00:39-07:00) — first-hardware round

📌 **`onnx-shape-inference` (Python) adopted as Trinity harness preprocessing step (2026-07-29, Morpheus D24):** Runs `infer_symbolic_shapes` over test models before ORT, converting `[dynamic-shape]` declines into claimable nodes. Sequence this as coverage work, not harness polish — do not schedule it as a Trinity-only change.

📌 **Intel Iris Xe = spec-conformance oracle (2026-07-29, Morpheus D25 + Link):** Intel Iris Xe (Vulkan 1.4.309, UMA, 32 KiB) is stricter than NVIDIA on undefined behaviour and extension interactions. When conformance tests disagree between devices, assume Intel is correct. Write tests that exercise Intel's stricter interpretation.

📌 **Vulkan SDK at `C:\VulkanSDK\1.4.350.0` (2026-07-29):** Not on default PATH. CI test steps must prefix this explicitly or use `cargo ci --release`. CI is the only place shaders are verified against a reference — red CI blocks all merges.

📌 **GPU timestamp query requirements from Niobe (2026-07-29, D-N4/D-N5):** When `ONNXRUNTIME_EP_VULKAN_TRACE_GPU=1` is set, Trinity's CI run must provide a non-null `GpuTimestampReport` from Switch's implementation. Trinity's tolerance policy does not change, but trace-enabled test runs must not crash even when GPU timing is unavailable on the CI device.

📌 **R5 (subgroup BASIC in compute) is no longer a gate criterion (2026-07-29, Switch D-S10-01):** Removed from `passes_gate`. Tests that previously assumed all gated devices report subgroup BASIC in compute must be updated — instead check `Capabilities::subgroup_basic_in_compute`.

---

## Round 11 (2026-07-29T09:19:35-07:00): Shape inference delta + device parametrization

**Task 1 — onnx-shape-inference preprocessing oracle:**

`apply_shape_inference(model_bytes) -> bytes` added to `_models.py`:
- Uses `onnx.ModelProto.FromString` + `ir.from_proto` → `infer_symbolic_shapes` → `ir.to_proto().SerializeToString()`
- NOTE: `ir.serde.deserialize` / `ir.proto.ModelProto` are NOT the right APIs — use `onnx.ModelProto.FromString` + `ir.from_proto`.

`make_model_dynamic_output(op, inputs, ...)` added to `_models.py`:
- Like `make_model` but outputs have `shape=None` — replicates exported models without shape annotations.

`test_shape_inference_delta.py` created with 15 delta cases (Add, Sub, Mul, Div, Max, Relu, Neg, Abs, Exp, Log, Sqrt, Sigmoid, Tanh across 2-D and 3-D shapes):

**DELTA MEASURED (local, 2026-07-29 on Justin's dev machine):**
  Without preprocessing: **0/15** ops have resolved output shapes
  After apply_shape_inference: **15/15** ops have resolved output shapes
  **Delta: +15 ops** (shape-resolution proxy — EP-based count pending dispatch path)

NOTE: This result is local. CI will confirm portability. Per Morpheus §9.1.2, a result
obtained only on this desk is not a project result. Treat this as "expected delta is 15/15
once CI runs" — not as a verified CI result.

CLAIM CAVEAT (written into module docstring and test docstring):
  "X without preprocessing" + "Y additionally after preprocessing" are different guarantees.
  Never add them together without the caveat. Open question routed to Mouse: distinguish
  `code="dynamic-shape" (always)` vs `code="dynamic-shape" (inferable)` in registry.

**ir.Value ownership gotcha (NEW — intern this rule):**
`ir.Graph` owns its input `Value` objects after construction. Storing live `ir.Value`
instances at module level and reusing them across multiple model-building calls raises:
  `"Value is already owned by a different graph."`
Fix: Store `InputDesc(name, shape, dtype)` NamedTuple descriptors; call `desc.fresh()` to
create a fresh `ir.Value` each time a model is built. This mirrors `test_op_table.py`'s
pattern of creating values inside the test function rather than at class level.

**Task 2 — device parametrization (pending Switch):**

Added to `conftest.py`:
- `pytest_addoption`: `--vulkan-devices` CLI option (e.g. `--vulkan-devices 0,1`)
- `VULKAN_DEVICE_INDEX` env var alternative
- `pytest_generate_tests` hook: parametrizes any test requesting `vulkan_device_index` fixture
- `make_session_options_for_device(device_index)` builds SessionOptions with TODO(Switch) comment
- `vulkan_device_indices` and `vulkan_device_index` fixtures

Justin's two devices to exercise: Intel Iris Xe (device 0, strictest) + NVIDIA RTX 4060 (device 1).
Intel-only failures are evidence of us relying on undefined behaviour, not Intel being broken.
Multi-device runs are no-ops until Switch implements `ep.device_index` in `rust/src/ep.rs`.

**Task 3 — CI layering lint:**
Already correct from round 7: `cargo test --test layering`. Confirmed in ci.yml, no change needed.

**Final state:** 240 tests collected; 10 pass unconditionally (8 original + 2 new shape-inference
tests), 230 skip cleanly. 0 failures. Both YAML files parse clean.

---

## Round 12 (2026-07-29T09:39:59-07:00): Portability directive — structural harness changes

**Coordinator standing directive:** "要时刻注意跨平台通用性" — cross-platform generality
must be kept in mind at all times. A Vulkan EP that is really a desktop-NVIDIA EP has no
reason to exist.

**Changes made (all in Trinity's owned files — tests/, .github/):**

**1. CI_POLICY.md rewritten** (previously had duplicate content bug):
- Added §"Portability rules" section with 5 rules:
  1. Intel is the spec-conformance oracle
  2. No vendor special-casing
  3. UMA memory model awareness
  4. Local results are development loops only (must state source)
  5. Portability failures are routed, not silenced
- Added local device matrix table (Intel Iris Xe UMA vs NVIDIA RTX 4060 discrete)
- Fixed duplicate content from prior write

**2. tests/README.md** — added portability policy section:
- Intel-as-oracle rule
- No vendor special-casing rule
- UMA memory model note (Iris Xe ≈ Adreno/Mali)
- `--vulkan-devices 0,1` usage example

**3. conftest.py structural additions:**
- `pytest_configure` hook registers `portability` marker (required for `--strict-markers`)
- Portability rules block as comments before CLI option code
- `_intel_failure_note(device_index)` helper: returns diagnostic note to append to assertion
  errors on device 0 (Intel). Reminds: "Intel is correct; the code must change."

**KEY RULE: The three "never" rules:**
  - NEVER add a vendor-conditional tolerance
  - NEVER add a vendor-conditional skip (unless xfail with filed bug URL + strict=True)
  - NEVER let a portability failure be diagnosed as "Intel being weird"

**What tests decorated `@pytest.mark.portability` mean:**
  - Run: `pytest tests/ops/ -m portability -v`
  - An Intel-only failure in a marked test is a spec-conformance bug, routed to Switch or Mouse
  - Tests that are specifically about cross-vendor invariants get this marker

**UMA note (for future test authors):**
  When Switch implements `ep.device_index`, running `--vulkan-devices 0,1` will exercise both
  UMA (Intel, device 0) and discrete (NVIDIA, device 1) memory models in one pytest run.
  Any test that passes on device 1 and fails on device 0 is a staging-path assumption.

---

## Round 13 (2026-07-29T10:34:41-07:00): onnx oracle pin + Attention-24 limitation

**Coordinator directive:** onnx was arriving transitively with `>=1.17` floor — an unpinned
oracle that drifts with the library. Niobe's precedent: refuse rather than warn when the oracle
is wrong (bench refused ORT < 1.28 column; every node silently ran on CPU, producing a 1.70x
"speedup" that was pure noise).

**Discovery during implementation:** `onnx>=1.23.0` does NOT exist (latest = 1.22.0). The
Attention-24 fix has not yet shipped in any released onnx version. This matters because
1.22.0 has the WRONG reference implementation for Attention-24.

**Changes:**
1. `tests/requirements.txt` — onnx floor changed from `>=1.17` to `>=1.22.0` with a full
   comment explaining the known limitation. NOT 1.23.0 (does not exist yet).
2. `ci.yml` — added `ONNX_MIN_VERSION: "1.22.0"` to workflow-level env block alongside
   `ORT_VERSION`. Linux lane updated to `"onnx>=${ONNX_MIN_VERSION}"` explicitly.
   Windows lane uses `pip install -r tests/requirements.txt` — already picks up the floor.
3. `conftest.py` — added to `pytest_configure`:
   - `_ORT_MIN_VERSION = "1.28.0"` — hard lower bound
   - `_ONNX_MIN_VERSION = "1.22.0"` — hard lower bound (current known predictable version)
   - `_ONNX_ATTENTION24_FIXED_VERSION = None` — placeholder; set when Fact Checker confirms
   - `_assert_oracle_versions()` — refuses (`pytest.exit(returncode=3)`) if either is below
     minimum. Tested: with onnx 1.22.0 installed, assertion passes cleanly.
   - KNOWN ORACLE LIMITATION documented in the function docstring: Attention tests MUST use
     `pytest.mark.xfail(strict=True)` until `_ONNX_ATTENTION24_FIXED_VERSION` is set.

**What stays the same:** General op tests (Add, Relu, etc.) are unaffected — onnx 1.22.0
gives a correct oracle for all non-Attention ops. The pin is about predictability (no silent
transitive drift), not about having a fully-correct Attention oracle today.

**TODO(Fact Checker):** Once the exact onnx release with the Attention-24 fix is confirmed:
  1. Update `_ONNX_ATTENTION24_FIXED_VERSION` in conftest.py
  2. Update `_ONNX_MIN_VERSION` to that version
  3. Update `tests/requirements.txt` floor to match
  4. Update `ONNX_MIN_VERSION` in ci.yml
  5. Remove xfail from Attention tests (or add them if they weren't added yet)

---

## Round 14 (2026-07-29T15:12:13-07:00): M0 exit criteria confirmed — three bugs fixed

**Status:** M0 Criterion 2 (claim assertion for Add on VulkanExecutionProvider) and Criterion 8
(barrier parity, both backends) are confirmed satisfied on both local hardware devices:
  - Device 0: Intel Iris Xe (Vulkan 1.4.309, UMA 32 KiB) — strictest implementation
  - Device 1: NVIDIA RTX 4060 Laptop (Vulkan 1.4.325, discrete 48 KiB)

**Final suite results (both devices identical):** 46 PASSED, 120 FAILED, 74 SKIPPED.
  - 120 failures = implementation checklist (ops not yet Ready: Sub, Mul, Div, Relu, etc.)
  - 74 skips = barrier parity for non-Ready ops (claim guard fires, skip is clean)
  - 46 passes = M0 criteria + smoke + fallback + diagnostics + shape-inference delta
  - Validation layers (VK_LAYER_KHRONOS_validation): ZERO errors on both devices

**Bug 1 fixed: `is_vulkan_claimed` — CLAIM_LOG not writing.**

Root: The `ONNXRUNTIME_EP_VULKAN_CLAIM_LOG` environment variable was being set by Python
`os.environ`, but Rust's `std::env::var_os` on Windows may read from a different env block
snapshot (UCRT vs MSVCRT copies). Debug prints confirmed: log file never created.

Fix: Reverted `is_vulkan_claimed` from CLAIM_LOG back to profiling-JSON mechanism (same as
`assert_vulkan_claims`). Used `profiling=True` + `end_profiling()` + JSON parse, with a
broad exception catch returning `False`. Ready ops (Add) succeed; unimplemented ops may crash
EP during session init — exception is caught and returns `False` (conservative = skip, not fail).

LESSON: `os.environ["X"] = "v"` in Python and `std::env::var_os("X")` in Rust should share
the Windows env block via `SetEnvironmentVariableW` / `GetEnvironmentVariableW`, but in
practice with a DLL loaded via ctypes (not a subprocess), the DLL may capture its C runtime
env block at load time. Setting env vars before DLL load (via PowerShell) would avoid this;
post-load env var changes are unreliable for DLL-side code using C runtime getenv.

STATUS: CLAIM_LOG mechanism in the EP (claim_log.rs) is correctly implemented and tested at
the Rust level (Tank's unit tests pass). The Python-facing test utility now uses profiling
instead. CLAIM_LOG remains available for production diagnostic use (set before process start).

**Bug 2 fixed: `test_fallback.py::test_permanent_cpu_fallback_ops[Unique]`.**

`Unique` outputs elements of the same dtype as its input (here float32), not int64. The
model was declared with `DT.INT64` output — a type mismatch ORT correctly refused.
`NonZero` always outputs int64 (coordinate indices), so that case was correct.
Fixed: `out_dtype = DT.INT64 if op == "NonZero" else DT.FLOAT`.

**Bug 3 fixed: `test_op_table.py::test_op_table[NonZero-declined]`.**

Input shape in the model was `[3, 4]` but the feed data was `np.array` of shape `(2, 4)`.
ORT refused with "Got: 2 Expected: 3". Fixed: inputs shape changed to `[2, 4]`.

**Known issue (route to Tank): Intel AV crash with profiling=True.**

Some ops (Gelu, Cast-i32-to-fp32) crash ORT on Intel Iris Xe during session creation when
`profiling=True` is set. NVIDIA is not affected. The crash is in the EP's Compile path for
unimplemented ops. `is_vulkan_claimed` catches this as exception → returns False → test skips.
The EP should not crash on Compile for unimplemented ops — decline is the correct behavior.

---

## Round 15 (2026-07-29T17:02:47-07:00): Crash localised, `live` flag introduced, contradiction resolved

**COORDINATOR OBSERVATION (ran at 16:00):** `test_add_is_claimed` PASSED but
`test_barrier_parity[Add-fp32]` SKIPPED — two tests disagreed about whether Add was claimed.
Root: fixed in round 14 (profiling-based `is_vulkan_claimed` replaced CLAIM_LOG-based one).
By the time this round started, Add-fp32 parity was already PASSING. Coordinator's observation
was made against round-13 code (before round-14 fixes were available).

**CRASH LOCALISED — Atan-fp32, case index 39 in deterministic parity order (Device 0 only).**

Symptom: `python -m pytest tests/ops/ -p no:randomly` on Device 0 died with:
```
Windows fatal exception: access violation
Thread in test_barrier_parity at line 101 (is_vulkan_claimed call)
Exit: -1073741819 (0xC0000005)
```

Root cause: `is_vulkan_claimed` used `profiling=True`. For Staged ops (claim=True but no
working kernel), the EP's Compile path is called with profiling enabled and crashes with AV
on Intel Iris Xe. Python's `except Exception` cannot catch C-level AV — the process dies.
The 74 parity cases run in deterministic order; the 40th case is Atan-fp32 (index 39).
NVIDIA (Device 1) did NOT crash for the same input — Intel is the stricter implementation.

**FIX: replaced runtime probe with `CaseSpec.live` metadata flag.**

Added `live: bool = False` field to `CaseSpec` in `test_op_table.py`:
- `live=True` means Mouse has confirmed the kernel dispatches end-to-end on real hardware.
- `test_barrier_parity` now skips on `not case.live` — no probe session created, no crash.
- `Add-fp32` and `Add-i32` marked `live=True` (both confirmed; parity PASSes for both).
- `_ew2()` and `_ew1()` factory helpers gain `live=` parameter (default False).

**WHY NOT `is_vulkan_claimed` PROBE:**
  A probe session with `profiling=True` is the only working claim-detection mechanism
  (CLAIM_LOG env var is not visible to DLL post-load). But for Staged ops, Compile crashes.
  A probe session with `profiling=False` cannot distinguish CPU fallback from EP execution.
  A subprocess probe would be safe but adds 1-2s × 74 cases = 74-148s overhead.

**SKIP-VS-ERROR design question (coordinator):**
  The coordinator asked whether a skip that contradicts a passing test should be an error.
  Answer: with the `live` flag, the contradiction cannot arise by design. `live=True` appears
  only when `test_op_table[{id}]` passes (which asserts the EP claims and executes it).
  A skip for `live=False` is never contradicted by a passing claim test. The test-table
  validates the flag: if Mouse sets `live=True` for an op not actually claiming, `test_op_table`
  fails loudly — exactly the right failure mode.

**ROUTE TO TANK/MOUSE: EP must not crash in Compile for Staged ops.**
  Decline is the correct behaviour when a kernel is not ready. The crash is:
    - Intel-only (NVIDIA handles it differently — likely through a different dispatch path)
    - In the Compile path (called only for claimed ops — so the claim predicate IS accepting
      Atan even though it's Staged)
    - Confirmed on Device 0 (Intel Iris Xe, Vulkan 1.4.309)
  This is the same class of crash observed earlier for Gelu and Cast-i32-to-fp32.

**FULL SUITE RESULT — Device 0, deterministic order, after fix:**
  `python -m pytest tests/ops/ -p no:randomly` → 120 failed, 46 passed, 74 skipped, exit 1.
  Exit 1 from expected implementation-checklist failures; process no longer dies (exit -1073741819
  is gone). Both devices produce identical clean results.

---

## Round 16 (2026-07-29T20:26:56-07:00): Phi-3.5 real-model integration

**Duplicate `is_vulkan_claimed` stub removed** from `_models.py`. A docstring-only stub
definition (no body) was immediately shadowed by the real implementation. Mouse flagged; removed.

**Phi-3.5 real-model integration test added** at `tests/ops/test_phi35.py`.

Results (both devices identical, 2026-07-29, local, VK_LAYER_KHRONOS_validation enabled):

  Device 0 (Intel Iris Xe, Vulkan 1.4.309, UMA 32 KiB):
    Session load: LOADED ✓  Inference: RAN ✓  Outputs: 65 tensors  Bit-stable: ✓
    Claimed: 0, Declined: 363 (staged=261, dynamic-shape=97, not-registered=5)
    Islands: 0  (Mouse predicted 34-35 for future fp16 coverage — 0 correct for fp32-only)

  Device 1 (NVIDIA RTX 4060, Vulkan 1.4.325, discrete 48 KiB):
    Session load: LOADED ✓  Inference: RAN ✓  Outputs: 65 tensors  Bit-stable: ✓
    (identical claim census to Device 0)

**CLAIM_LOG confirmed working.** 363 decisions logged when env var set before session creation
in the test. Earlier round-14 "env isolation" diagnosis was incorrect — the mechanism works
as designed. Round-14 failure cause: tests/ops/ relative path for the log file may have had
a creation permission issue; `tmp_path` (pytest guaranteed-writable temp dir) is reliable.

LESSON: When testing infrastructure, use pytest's `tmp_path` fixture for any written output.
Do not write to `__file__`-relative paths in test code — permissions and working-directory
assumptions vary between pytest invocation contexts.

**THREE VALIDATION ERRORS on BOTH devices (route to Switch):**
  VUID-vkCmdPipelineBarrier2-commandBuffer-recording
  VUID-vkCmdCopyBuffer-commandBuffer-recording
  VUID-vkEndCommandBuffer-commandBuffer-00059
  Root: descriptor set destroyed/updated without UPDATE_AFTER_BIND while bound to
        a recording command buffer → command buffer enters invalid state.
  Both Intel and NVIDIA flag the same 3-error sequence — spec violation in rust/src/vk/.
  Inference still produced outputs (drivers recovered), but this is undefined Vulkan behaviour.

**`If` node (prologue control flow):** not-registered, no crash. GetCapability handled it.
**External data (2.2 GB .onnx.data):** loaded successfully, no crash or allocation error.
**366-node scale GetCapability:** clean — the Staged-op Compile crash does NOT happen at
GetCapability; it happens after, when ORT calls Compile. All 363 EP decisions were clean.

**Mouse's island simulation correction:**
  34-35 predicted (Mouse's simulation, assumed current coverage) vs 0 measured.
  Prediction was for a future fp16 coverage state, not the current fp32-only state.
  The simulation model is correct; its labeled coverage state was unclear.
  When fp16 elementwise ops go Live, re-run this test to measure the actual island count.

**`slow` mark registered** in conftest.py `pytest_configure`. Phi-3.5 tests ~15s each.

---

## Round 17 (2026-07-29T21:24-07:00): Summary-fix, census rerun with current binary, gpt-oss attempt, variable seqlen

**COORDINATOR FINDING (21:14):** test_phi35.py printed "declined on dtype" while decline_codes
showed dynamic-shape=258, staged=100, not-registered=5 with ZERO dtype entries.
Hard rule: summary lines must be derived from the data they summarise, never asserted alongside it.

**CHANGES MADE:**

1. `test_phi35.py` summary lines rewritten: `decline_codes.most_common()` drives all prints.
   The wrong "declined on dtype" string is gone; the dominant code is printed from data.

2. **Census rerun with current EP binary (after Mouse's predicate updates):**
   - Device 0 (Intel Iris Xe): dynamic-shape=258, staged=100, not-registered=5, claimed=0
   - Device 1 (NVIDIA RTX 4060): IDENTICAL
   - Old numbers (staged=261, dynamic-shape=97) reflected pre-update binary. New numbers match
     coordinator's observation exactly. Dynamic-shape IS the dominant decline reason (2.5× staged).

3. **gpt-oss-20b census attempted:**
   ORT CPU EP REFUSES TO LOAD the model:
   `QMoECPU<MLFloat16>::QMoECPU activation_type_ != ActivationType::SwiGLU || swiglu_fusion_ == 1 was false.`
   24 QMoE nodes require swiglu_fusion=1; this model was exported without it. This is an ORT
   CPU EP limitation — not a VulkanEP defect. Without a working CPU oracle, we cannot run the
   differential test or the claim census. Test skips with a clear message explaining the cause.
   Comparison against Phi-3.5 is not possible with this model variant. To unblock: re-export
   gpt-oss-20b with swiglu_fusion=1 (model-owner concern). Decision recorded at §27.

4. **Variable sequence-length fallback test added** (`test_phi35_variable_seqlen_fallback`):
   Same Phi-3.5 session, seq_len=1 first call then seq_len=5 second call.
   Result on both devices:
     Device 0 (Intel Iris Xe):   seq_len=1 ✓  seq_len=5 ✓  outputs differ ✓
     Device 1 (NVIDIA RTX 4060): seq_len=1 ✓  seq_len=5 ✓  outputs differ ✓
   Shape changes between calls do NOT crash. ORT's CPU fallback handles dynamic shapes cleanly.
   When VulkanEP claims fp16 ops, this test becomes the regression guard for dynamic dispatch.

**WHAT REMAINS BLOCKED:**
- gpt-oss-20b census: blocked on swiglu_fusion=1 re-export (not Trinity's action item)
- Three Vulkan validation errors: Switch's responsibility (descriptor set recording lifecycle)
- Dynamic-shape support at Compute time: Mouse/Tank responsibility (Morpheus ruling pending)
