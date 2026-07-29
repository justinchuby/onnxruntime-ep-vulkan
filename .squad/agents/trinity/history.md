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

