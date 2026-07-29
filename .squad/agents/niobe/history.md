# Project Context

- **Owner:** Justin Chu
- **Project:** onnxruntime-ep-vulkan — a cross-platform Vulkan plugin execution provider for ONNX Runtime, written in Rust.
- **Reference architecture:** `C:\Users\justinchu\dev\onnxruntime-mlx` — mirror its layout: `rust/src/{lib,ep,factory,sys,registry,engine,logging}.rs`, `rust/src/ops/*`, `rust/build.rs`, `tests/conformance/`, `bench/`, `python/`, `docs/DESIGN.md`.
- **Stack:** Rust (cdylib plugin EP), Vulkan 1.1+ compute, SPIR-V/GLSL shaders, ONNX Runtime C API, Python bindings, GitHub Actions CI.
- **Cross-platform mandate:** Windows, Linux, Android, macOS via MoltenVK; NVIDIA / AMD / Intel / Adreno / Mali; software rasterizer (lavapipe / SwiftShader) for GPU-less CI.
- **My focus:** Performance — benchmarks, profiling, regression tracking
- **Created:** 2026-07-28T17:52:04-07:00

## Learnings

<!-- Append new learnings below. Each entry is something lasting about the project. -->

📌 Team update (2026-07-28T17:59:54-07:00): Vulkan API baseline is a capability-set: ≥1.1 core + compute queue + `synchronization2` + `subgroup_size_control` + subgroup BASIC+ARITHMETIC + `maxComputeWorkGroupInvocations ≥ 256` + `maxComputeSharedMemorySize ≥ 16 KiB`. Benchmark device coverage will vary by tier (desktop 2022+ / software rasterizer / Android 2022+ mid-high / pre-2022 Android that reports these extensions). — decided by Morpheus, Switch, Link, Fact Checker

📌 Team update (2026-07-28T17:59:54-07:00): Record-once / replay-many — `Compute` hashes input shapes and replays a cached `VkCommandBuffer`; re-records only on a cache miss. Benchmarks must distinguish first-inference latency (recording path) from steady-state latency (replay path). — decided by Morpheus

📌 Team update (2026-07-28T17:59:54-07:00): Barrier model is per-edge (`vkCmdPipelineBarrier2` or `vkCmdPipelineBarrier` fallback), not global. This enables driver-side parallel dispatch scheduling. Benchmarks should profile this on real hardware to confirm the driver exploits parallelism. — decided by Switch

📌 Team update (2026-07-28T17:59:54-07:00): Benchmark reporting requirements — report island count and largest fused region alongside wall time. A change that increases claim rate but shrinks fused region size is a regression. These metrics must appear in every benchmark run. — decided by Morpheus

📌 Team update (2026-07-28T17:59:54-07:00): VkBuffer handle registry (opaque 64-bit tag → `(VkBuffer, offset)` map) — OQ-3 tracks whether the hash lookup is ever a bottleneck. Niobe's profiling will answer this; it is not expected to be hot. — decided by Morpheus

📌 Team update (2026-07-28T17:59:54-07:00): M0 benchmark — single `Add` node, software rasterizer. Numbers are not meaningful at M0; the goal is establishing the measurement infrastructure. — decided by Morpheus
