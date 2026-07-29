# Project Context

- **Owner:** Justin Chu
- **Project:** onnxruntime-ep-vulkan — a cross-platform Vulkan plugin execution provider for ONNX Runtime, written in Rust.
- **Reference architecture:** `C:\Users\justinchu\dev\onnxruntime-mlx` — mirror its layout: `rust/src/{lib,ep,factory,sys,registry,engine,logging}.rs`, `rust/src/ops/*`, `rust/build.rs`, `tests/conformance/`, `bench/`, `python/`, `docs/DESIGN.md`.
- **Stack:** Rust (cdylib plugin EP), Vulkan 1.1+ compute, SPIR-V/GLSL shaders, ONNX Runtime C API, Python bindings, GitHub Actions CI.
- **Cross-platform mandate:** Windows, Linux, Android, macOS via MoltenVK; NVIDIA / AMD / Intel / Adreno / Mali; software rasterizer (lavapipe / SwiftShader) for GPU-less CI.
- **My focus:** Test & Conformance — differential testing vs ORT CPU EP, tolerances, CI tests
- **Created:** 2026-07-28T17:52:04-07:00

## Learnings

<!-- Append new learnings below. Each entry is something lasting about the project. -->

📌 Team update (2026-07-28T17:59:54-07:00): Correctness oracle is ORT's own CPU EP running the same ONNX model — not numpy, not a custom reference. Every op test must also assert the node actually ran on `VulkanExecutionProvider` (not silently falling back to CPU). A test that does not assert device placement can pass vacuously even if the EP ran nothing. — decided by Morpheus

📌 Team update (2026-07-28T17:59:54-07:00): Tolerance policy — tolerances are derived and documented per op family. Widening a tolerance to turn a red test green requires Trinity's sign-off and an explanatory note inside the test. — decided by Morpheus

📌 Team update (2026-07-28T17:59:54-07:00): Validation-layer-clean (zero Vulkan validation errors) is a hard "done" criterion for any engine change, not optional. — decided by Morpheus

📌 Team update (2026-07-28T17:59:54-07:00): Software rasterizers (lavapipe, SwiftShader) both pass Vulkan 1.3 conformance and are suitable for GPU-less CI. They are a smoke test, not a correctness claim. — decided by Morpheus, verified by Fact Checker

📌 Team update (2026-07-28T17:59:54-07:00): M0 test — stock ORT loads the plugin, enumerates a Vulkan device, runs a graph with a single `Add` node, matches ORT CPU EP within tolerance, on Windows and Linux, on a software rasterizer, in CI. This is the first conformance test Trinity owns. — decided by Morpheus

📌 Team update (2026-07-28T17:59:54-07:00): Vulkan API baseline is a capability-set: ≥1.1 core + compute queue + `synchronization2` (ext or 1.3 core) + `subgroup_size_control` (ext or 1.3 core) + subgroup BASIC+ARITHMETIC + workgroup/shared-memory minimums. Tests run on any device meeting this bar (including software rasterizers). — decided by Morpheus, Switch, Link, Fact Checker

📌 Trinity work landed (2026-07-28T19:16:08-07:00): Test & CI foundation complete.
  **Claim assertion mechanism:** ORT profiling JSON (`enable_profiling=True`, `end_profiling()`, parse `cat=="Node"` events for `args.provider=="VulkanExecutionProvider"`). Structured, not text parsing. Same mechanism as onnxruntime-mlx reference. Implemented in `tests/ops/_models.py::assert_vulkan_claims`.
  **Tolerance policy:** Named constants in `_models.py` per op-family (fp32 elementwise: 1e-5/1e-5; fp32 transcendental: 1e-5/1e-5; fp16: 1e-3/1e-3; comparison: exact; reductions/GEMM M2+: TBD/OQ-10). Protocol for widening: Trinity sign-off + in-test comment.
  **CI lanes:** Linux lavapipe (ubuntu-22.04, Mesa 22.0, Vulkan 1.3, lavapipe, VK_LAYER_KHRONOS_validation, zero-errors enforced); Windows build+clippy (SwiftShader Vulkan lane TODO, documented). Layering lint wired (grep fallback until Tank's binary).
  **Files created:** `tests/README.md`, `tests/requirements.txt`, `tests/ops/conftest.py`, `tests/ops/_models.py`, `tests/ops/test_elementwise.py`, `tests/ops/test_claim_diagnostics.py`, `tests/ops/test_fallback.py`, `tests/conformance/README.md`, `tests/conformance/RESULTS.md`, `tests/conformance/claimed_ops.txt`, `tests/conformance/vulkan_runtime_wrapper.py`, `tests/conformance/test_conformance.py`, `tests/conformance/run_conformance.sh`, `.github/workflows/ci.yml`, `.github/workflows/conformance.yml`, `.squad/decisions/inbox/trinity-test-foundation.md`.
  **Verified:** Python syntax clean (py_compile). YAML syntax valid.
  **Scaffolded/blocked on Tank:** Actual test execution awaits `rust/` crate; `register_execution_provider_library` call; Mouse's Add claim predicate; Tank's validation-layer panic callback.

📌 Trinity round-2 corrections applied (2026-07-28T19:16:08-07:00):

  **CORRECTION 1 — ORT 1.28.0:** `ORT_VERSION` now pinned at workflow-level `env:` in `ci.yml` (single location for the whole file); `conformance.yml` job-level env updated; `tests/requirements.txt` updated to `onnxruntime>=1.28`. Reason: ORT 1.27 has null-allocator PrePack bug + deleter lifetime issue in plugin EP path, fixed in 1.28 (released 2026-07-24). Source: Fact Checker audit trail.

  **CORRECTION 2 — Windows Vulkan lane resolved:** Investigated mesa-dist-win (https://github.com/pal1000/mesa-dist-win). Latest release `26.1.3` (2026-06-26) ships `mesa3d-26.1.3-release-msvc.7z` (~69 MB, MSVC runtime, 7z extractable on windows-latest). This package includes lavapipe (Vulkan 1.3 CPU software rasterizer) as DLL + ICD JSON. SwiftShader rejected: Google publishes no Windows prebuilts; ~20 min build from source. `build-test-windows` CI job now installs mesa-dist-win lavapipe and runs pytest (plus always-on no-ICD fallback test). `MESA_VERSION: "26.1.3"` pinned at workflow-level env. Finding reported to Link (who owns `docs/PLATFORMS.md §7.4`).

  **ADDITION — Flat op table:** Created `tests/ops/test_op_table.py` with `CaseSpec` dataclass and single `test_op_table` dispatch. Pre-populated with all tier-1 ops from OP_COVERAGE §4.1–§4.5: 23 EW-B, 27 EW-U, 16 activations, Cast/Where, Identity/Flatten/Reshape, plus declined set (fp64, NonZero). 124 tests total; all skip cleanly without EP lib. `assert_vulkan_does_not_claim` promoted to `_models.py` top-level export; `test_fallback.py` updated. `claim=True`/`False` rows are equally easy to write — bulk op addition is one row per op.

📌 Trinity round-3: barrier parity layer (2026-07-28T19:16:08-07:00) — see decision record §8.

📌 Trinity round-4: quantization oracle + tolerance policy + domain regression (2026-07-28T22:28:08-07:00):

  **Backend probe confirmed live (commit 255f2db):** Switch's `ONNXRUNTIME_EP_VULKAN_BACKEND_PROBE`
  is in `rust/src/vk/barrier.rs`. Updated `test_barrier_parity.py` to demote the `unknown` branch
  from "TODO(Switch)" to "EP not yet built" messaging. Hard assert still fires for `legacy` value.

  **conftest.py refactor — non-Vulkan tests now run without EP:**
  `register_vulkan_ep` changed from `autouse=True` (session-wide skip) to a non-autouse fixture
  that returns bool. `vulkan_device_available` gains it as a dependency and skips if not registered.
  Result: Vulkan-independent tests (oracle pinning, layer capture, domain regression) run and pass
  without `ONNXRUNTIME_VULKAN_EP_LIB` set. 8 tests now pass unconditionally; 198 skip cleanly.

  **Quantization oracle investigation finding:**
  - MatMulNBits fp32 activations: ORT CPU EP works as differential oracle on ORT 1.28. ✓
  - MatMulNBits fp16 activations: ORT 1.27 produces NaN/Inf (PrePack bug); gated on ORT>=1.28.
  - accuracy_level pinned to 1 (fp32 accumulator) via `MATMULNBITS_ORACLE_ACCURACY_LEVEL=1`.
    Level 4 (int8 VNNI) diverges ~3.6e-3 max_abs at K=1024, N=512.
  - SimplifiedLayerNormalization not on CPU EP in 1.27; use standard LayerNormalization opset 17.

  **Three-regime tolerance policy implemented (Mouse OP_COVERAGE §10.1):**
  - `DEQUANT_EXACT = {rtol:0, atol:0}` — bit-exact vs NumPy (not CPU EP).
  - `MATMULNBITS_FP32 = {rtol:1e-3, atol:1e-4}` — fp32 vs CPU EP oracle.
  - `MATMULNBITS_FP16 = {rtol:2e-2, atol:1e-3}` — fp16 vs CPU EP oracle (ORT>=1.28).
  - Policy + justification in `_models.py` module docstring. Tests in `test_matmulnbits.py`.

  **Per-layer capture mechanism built:**
  - `with_captured_outputs(model_bytes, names)` — appends intermediates as graph outputs.
  - `compare_layers(model_bytes, feeds, names)` — runs both EPs, returns per-layer diff list.
  - Unit test `test_layer_capture_mechanism` passes without Vulkan device.

  **Morpheus C1 — domain regression test (runtime half):**
  - `tests/ops/test_domain_regression.py`: `com.microsoft::NotARealOp` → ORT raises `Fail`
    (not SystemError / EP crash). Both tests pass without Vulkan device.
  - TODO(Mouse): upgrade to machine-readable reason code once Mouse's registry API is confirmed.

  **Final test count:** 206 collected; 8 pass unconditionally, 198 skip cleanly. 0 failures.
📌 Trinity round-3: barrier-backend parity layer (2026-07-28T19:16:08-07:00):

  **New test layer:** Created `tests/ops/test_barrier_parity.py`. Reads `_CASES` from `test_op_table.py`, filters to `claim=True` (~74 cases), runs each case twice with both barrier backends, asserts bit-identical outputs.

  **Backend probe mechanism agreed with Switch:** When `ONNXRUNTIME_EP_VULKAN_BACKEND_PROBE=<path>` is set, EP writes `"sync2"` or `"legacy"` to `<path>` during `Barriers::select` in `Device::new`. File-write IPC — zero ORT API changes required. Until Switch lands this, the parity test emits `UserWarning` rather than failing; the output comparison still runs. See `_models.run_with_backend` docstring for the exact contract.

  **`run_with_backend()` added to `_models.py`:** Sets `ONNXRUNTIME_EP_VULKAN_BACKEND_PROBE`, applies `ep.force_legacy_barriers=1` when `force_legacy=True`, returns `(outputs, active_backend)`. Probe file lives in `tests/ops/` (project-relative, never `/tmp`), unique per process.

  **CI:** Both Linux and Windows pytest steps already collect `test_barrier_parity.py` (in `tests/ops/`). Explicit comments added. No extra pytest invocation needed — parity runs within the standard `pytest tests/ops` call. 198 tests total; all 198 skip cleanly without EP lib.

  **Needs from others:**
  - **Switch:** Implement `ONNXRUNTIME_EP_VULKAN_BACKEND_PROBE` in `rust/src/vk/barrier.rs` per contract in `_models.run_with_backend` docstring.
  - **Switch:** Wire `ep.force_legacy_barriers` session option to `Barriers::select` in `Device::new`.


📌 Trinity round-2 corrections applied (2026-07-28T19:16:08-07:00):

  **CORRECTION 1 — ORT 1.28.0:** `ORT_VERSION` now pinned at workflow-level `env:` in `ci.yml` (single location for the whole file); `conformance.yml` job-level env updated; `tests/requirements.txt` updated to `onnxruntime>=1.28`. Reason: ORT 1.27 has null-allocator PrePack bug + deleter lifetime issue in plugin EP path, fixed in 1.28 (released 2026-07-24). Source: Fact Checker audit trail.

  **CORRECTION 2 — Windows Vulkan lane resolved:** Investigated mesa-dist-win (https://github.com/pal1000/mesa-dist-win). Latest release `26.1.3` (2026-06-26) ships `mesa3d-26.1.3-release-msvc.7z` (~69 MB, MSVC runtime, 7z extractable on windows-latest). This package includes lavapipe (Vulkan 1.3 CPU software rasterizer) as DLL + ICD JSON. SwiftShader rejected: Google publishes no Windows prebuilts; ~20 min build from source. `build-test-windows` CI job now installs mesa-dist-win lavapipe and runs pytest (plus always-on no-ICD fallback test). `MESA_VERSION: "26.1.3"` pinned at workflow-level env. Finding reported to Link (who owns `docs/PLATFORMS.md §7.4`).

  **ADDITION — Flat op table:** Created `tests/ops/test_op_table.py` with `CaseSpec` dataclass and single `test_op_table` dispatch. Pre-populated with all tier-1 ops from OP_COVERAGE §4.1–§4.5: 23 EW-B, 27 EW-U, 16 activations, Cast/Where, Identity/Flatten/Reshape, plus declined set (fp64, NonZero). 124 tests total; all skip cleanly without EP lib. `assert_vulkan_does_not_claim` promoted to `_models.py` top-level export; `test_fallback.py` updated. `claim=True`/`False` rows are equally easy to write — bulk op addition is one row per op.



---

## Cross-agent context appended (2026-07-28T22:28:08-07:00)

📌 **`force_legacy_barriers` parity lane contract (Switch + Trinity):** `ep.force_legacy_barriers=1` session option forces the legacy `vkCmdPipelineBarrier` backend. Trinity's `test_barrier_parity.py` runs claim=True ops with both backends; bitwise-identical outputs required. `ONNXRUNTIME_EP_VULKAN_BACKEND_PROBE=<path>` env: EP writes "sync2" or "legacy" to the file during `Barriers::select`. Switch wired this in `barrier.rs`. CI must run the parity suite on both Linux and Windows lavapipe lanes.

📌 **C2 item 7: fingerprint audit CI (Morpheus §1.4):** `graph_census.py` runs in CI before any tier-3 contrib work. Trinity owns the "must NOT claim `com.microsoft::NotARealOp`" domain regression test (`test_domain_regression.py`); upgrade to machine-readable reason code when Mouse's registry API is confirmed. `[contrib-schema]` decline ≠ `[attribute]` decline — do not merge buckets.

📌 **`accuracy_level` pinning (Trinity round-4 oracle investigation):** `MatMulNBits` oracle pinned at `MATMULNBITS_ORACLE_ACCURACY_LEVEL=1`. Level 4 diverges ~3.6e-3 (would present as GPU bug). Fp16 path gated on ORT ≥ 1.28. Three tolerances defined: `DEQUANT_EXACT` (bit-exact vs NumPy), `MATMULNBITS_FP32`, `MATMULNBITS_FP16`. Dequantize bit-layout goes to NumPy (independent spec), not CPU EP.

📌 **mesa-dist-win lavapipe on Windows (Trinity + Link):** Mesa release 26.1.3 (2026-06-26), `mesa3d-26.1.3-release-msvc.7z`, includes lavapipe (Vulkan 1.3) as DLL + ICD JSON. `MESA_VERSION: "26.1.3"` pinned at workflow env. SwiftShader rejected (no Windows prebuilts, 20 min build). Link owns `docs/PLATFORMS.md §7.4` update.

📌 **ORT 1.28.0 pin (Trinity + Tank + Fact Checker):** `ORT_VERSION=1.28.0` pinned at workflow-level env in `ci.yml` and `conformance.yml`. 1.27 excluded (null-allocator PrePack bug; fp16 NaN/Inf confirmed by Trinity). `tests/requirements.txt` → `onnxruntime>=1.28`.

📌 **`bind_aliased_output` / `dispatch_indirect` seams (Switch):** Default methods on `DispatchContext` return `Err` for unimplemented seams. Trinity's `Recorder` mock does not need to implement them unless testing XL kernels. When XL kernel parity tests are added, verify both backends execute correctly through these seams.

📌 **Claim diagnostic records (Mouse turn-5):** Per-event JSON, one self-contained line per event, append-and-flush (no lifecycle hook). Hooks `claim_decision` not the ep.rs aggregator. Trinity's C1 test (`test_domain_regression.py`) can parse the JSON file directly rather than asserting "zero nodes claimed" — upgrade when Mouse's format is stable.

📌 Trinity round-5 (superseded by round-6): initial glslc fix attempt:

  **glslang-tools** provides `glslangValidator`, not `glslc`. Attempted to fix with `shaderc`
  from Ubuntu 22.04 repos — that package does not exist there. Two consecutive unverified
  claims. CI remained red.

📌 Trinity round-7: case 1/2 determination + GPG fix + layering lint (2026-07-28T22:28:08-07:00):

  **glslc root cause:** Ubuntu 22.04 does not package glslc in any of its own repos.
  The LunarG Vulkan SDK apt repository for Jammy does, via `shaderc`. Added step "Add
  LunarG Vulkan SDK apt repository" using the versioned list file and modern GPG keyring:
  key: `https://packages.lunarg.com/lunarg-signing-key-pub.asc`
  list: `https://packages.lunarg.com/vulkan/1.3.296/lunarg-vulkan-1.3.296-jammy.list`
  After `apt-get update`, `shaderc` installs `/usr/bin/glslc`. The "Verify GLSL compiler"
  precondition step runs `glslc --version` — the CI run is the proof.

  **VULKAN_SDK_VERSION:** Promoted to workflow-level env. Previously only in Windows job.

  **Windows crash:** PATH fix (round-5) did not resolve the access violation. DLL resolution
  is not the cause. Diagnostic improvements added:
  - conftest.py `register_vulkan_ep`: ctypes.CDLL pre-probe flushes to stderr before ORT
    call. "ctypes OK" in crash output → fault is EP initialization code, not dependencies.
  - `tests/ops/test_a_ep_smoke.py`: isolated ctypes + ORT-register tests, collected first.

  **Rule cemented (twice violated, now internalized):** A claim about an external package's
  contents is not usable until `package_binary --version` has run in the target environment.
  The "Verify GLSL compiler" step is that verification; it is not optional.

  **Test count:** 210 collected; 8 pass unconditionally, 200 skip cleanly, 0 failures.

  **README ownership note:** README changes must go through the coordinator (Morpheus owns
  the file). CI badge was committed by the coordinator without conflict this time.

📌 Trinity round-8: YAML syntax error + permanent parse check (2026-07-28T22:28:08-07:00):

  **Failure:** ci.yml broke YAML parsing (run 30449585950 never started). Root cause: a
  shell line continuation (`\`) inside the `run:` block scalar placed the second line at
  column 1. YAML requires all content in a block scalar to be indented past the block's
  indicator column — column 1 exits the block and breaks the document.

  **Fix:** collapsed the two-line `echo "deb ..." \ <continuation>` into a single line.
  Content (signed-by=, LunarG URL) was correct; only the YAML-breaking line split was wrong.

  **Permanent check added to `.github/CI_POLICY.md` pre-finish checklist:**
  ```
  python -c "import yaml, sys; [yaml.safe_load(open(f)) or print('OK:', f)
    for f in ['.github/workflows/ci.yml', '.github/workflows/conformance.yml']]"
  ```
  This runs locally before ending any turn that edits a workflow file. The yaml parse is
  syntax verification; `glslc --version` in-lane is tool availability verification. Both
  are now documented as non-optional pre-finish steps in CI_POLICY.md.

  **Working-practice note for YAML block scalars (`run: |`):**
  - Shell `\` continuation is fine if the next line is indented at or past the block level.
  - A continuation that outdents to column 1 exits the YAML block — the parser sees a new
    mapping key, not a continuation string.
  - Safe alternatives: single long line, or split with a shell variable:
    `URL="...long..."; echo "deb $URL" | sudo tee ...`

📌 Trinity round-9: Windows elevation fix + epctl probe + lane hardening (2026-07-28T22:28:08-07:00):

  **Windows root cause (Link §7.4.1):** LunarG loader 1.3+ silently ignores VK_ICD_FILENAMES,
  VK_DRIVER_FILES, and VK_ADD_DRIVER_FILES when the process is elevated (Administrator with
  UAC disabled). GitHub Actions Windows runners are `runneradmin`. The ICD path was correct;
  the loader never read it. Fix: register the Mesa lavapipe ICD under
  HKLM:\SOFTWARE\Khronos\Vulkan\Drivers via PowerShell. Registry-based discovery works under
  elevation.

  **VK_INSTANCE_LAYERS moved to pytest step (both lanes):** At job level it can cause
  vkCreateInstance to fail if the layer path isn't configured yet, masking lavapipe as the
  apparent problem. Now set only in the test step env.

  **VK_LOADER_DEBUG: warn added at job level (both lanes):** Previously loader failures were
  silent. Now all Vulkan-related steps emit loader diagnostics.

  **Vulkaninfo smoke-checks hardened:** Both lanes now grep for "llvmpipe" (lavapipe's device
  name) and exit 1 if not found. Linux: also added `sudo ldconfig` after package install.

  **epctl --probe-loader step added (both lanes):** Switch's standalone Vulkan probe, before
  pytest, exits non-zero if no capable device found. Turns "tests all skipped" into a named
  red step with a diagnostic.

  **YAML parse verified:** Both workflow files pass `yaml.safe_load` — checklist followed.
