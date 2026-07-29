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
