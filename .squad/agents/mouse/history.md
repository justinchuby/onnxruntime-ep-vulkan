# Project Context

- **Owner:** Justin Chu
- **Project:** onnxruntime-ep-vulkan — a cross-platform Vulkan plugin execution provider for ONNX Runtime, written in Rust.
- **Reference architecture:** `C:\Users\justinchu\dev\onnxruntime-mlx` — mirror its layout: `rust/src/{lib,ep,factory,sys,registry,engine,logging}.rs`, `rust/src/ops/*`, `rust/build.rs`, `tests/conformance/`, `bench/`, `python/`, `docs/DESIGN.md`.
- **Stack:** Rust (cdylib plugin EP), Vulkan 1.1+ compute, SPIR-V/GLSL shaders, ONNX Runtime C API, Python bindings, GitHub Actions CI.
- **Cross-platform mandate:** Windows, Linux, Android, macOS via MoltenVK; NVIDIA / AMD / Intel / Adreno / Mali; software rasterizer (lavapipe / SwiftShader) for GPU-less CI.
- **My focus:** Op Coverage — ONNX op implementations, registry, graph partitioning
- **Created:** 2026-07-28T17:52:04-07:00

## Learnings

<!-- Append new learnings below. Each entry is something lasting about the project. -->

📌 Team update (2026-07-28T17:59:54-07:00): Vulkan API baseline is a capability-set, not a version floor. Devices must have ≥1.1 core + compute queue + `synchronization2` + `subgroup_size_control` + subgroup BASIC+ARITHMETIC + workgroup and shared-memory minimums. Everything else (`shaderFloat16`, `shaderInt8`, etc.) is optional and gates shader variants. Mouse's op handlers choose variants via `DispatchContext`; they must never hard-require an optional capability. — decided by Morpheus, Switch, Link, Fact Checker

📌 Team update (2026-07-28T17:59:54-07:00): Hard layering rule — op handlers in `rust/src/ops/` must never reference `sys::`, `Ort`, `ash`, `vk::`, or `unsafe`. CI lint enforces this and fails the build on a violation. Op handlers see only `NodeDesc`, `NodeView`, `TensorRef`, and a `DispatchContext` trait. — decided by Morpheus

📌 Team update (2026-07-28T17:59:54-07:00): M0 op is a single `Add` node. This is the first deliverable for Mouse. — decided by Morpheus

📌 Team update (2026-07-28T17:59:54-07:00): Op growth strategy — grow by family (shared shader skeleton, descriptor layout, test file). Prioritize ops that merge existing graph islands or extend an island's edge. Benchmarks must report island count and largest fused region alongside wall time. Maximizing claim rate is actively harmful; maximize fused compute volume instead. — decided by Morpheus

📌 Team update (2026-07-28T17:59:54-07:00): v1 non-goals include quantized ops, contrib ops, attention fusion, graph-level op fusion, dynamic-shape fast paths, fp64, and data-dependent output shapes. Mouse must not invest in these families for v1. — decided by Morpheus
