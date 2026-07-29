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

