# Project Context

- **Owner:** Justin Chu
- **Project:** onnxruntime-ep-vulkan — a cross-platform Vulkan plugin execution provider for ONNX Runtime, written in Rust.
- **Reference architecture:** `C:\Users\justinchu\dev\onnxruntime-mlx` — mirror its layout: `rust/src/{lib,ep,factory,sys,registry,engine,logging}.rs`, `rust/src/ops/*`, `rust/build.rs`, `tests/conformance/`, `bench/`, `python/`, `docs/DESIGN.md`.
- **Stack:** Rust (cdylib plugin EP), Vulkan 1.1+ compute, SPIR-V/GLSL shaders, ONNX Runtime C API, Python bindings, GitHub Actions CI.
- **Cross-platform mandate:** Windows, Linux, Android, macOS via MoltenVK; NVIDIA / AMD / Intel / Adreno / Mali; software rasterizer (lavapipe / SwiftShader) for GPU-less CI.
- **My focus:** Runtime & FFI — ORT plugin EP C ABI, sys/ep/factory, build & packaging
- **Created:** 2026-07-28T17:52:04-07:00

## Learnings

<!-- Append new learnings below. Each entry is something lasting about the project. -->

📌 Team update (2026-07-28T17:59:54-07:00): Vulkan API baseline is a capability-set, not a version floor — device is advertised if it has Vulkan ≥1.1 core + a compute queue + `synchronization2` (ext or 1.3 core) + `subgroup_size_control` (ext or 1.3 core) + subgroup BASIC+ARITHMETIC + `maxComputeWorkGroupInvocations ≥ 256` + `maxComputeSharedMemorySize ≥ 16 KiB`. Tank's device-enumeration code must enforce exactly these capability checks; everything else is optional probe. `VkApplicationInfo::apiVersion = min(vkEnumerateInstanceVersion(), VK_API_VERSION_1_3)`. — decided by Morpheus, Switch, Link, Fact Checker

📌 Team update (2026-07-28T17:59:54-07:00): Vulkan crate stack is `ash` + `gpu-allocator`. Tank's `Cargo.toml` must include these; `vulkano` and `wgpu` are rejected. `gpu-allocator` is the pure-Rust VMA equivalent. — decided by Switch

📌 Team update (2026-07-28T17:59:54-07:00): `build.rs` (Tank's responsibility) must locate and invoke `glslc`, iterate `shaders/glsl/`, write SPIR-V to `OUT_DIR/spv/`, and generate `OUT_DIR/shader_modules.rs`. No runtime shader compiler in the deployed artifact. — decided by Switch

📌 Team update (2026-07-28T17:59:54-07:00): ORT Plugin EP C API (`OrtEpFactory`, `CreateEpFactories`) is experimental since ORT 1.22, no ABI stability guarantee. Strategy: pin to a specific ORT version; invest early in an FFI abstraction layer so breakages are contained. — decided by Fact Checker

📌 Team update (2026-07-28T17:59:54-07:00): M0 definition — stock ORT loads the plugin, enumerates a Vulkan device, runs a graph with a single `Add` node, matches ORT CPU EP within tolerance, on Windows and Linux, on a software rasterizer, in CI. Tank wires the ORT FFI boundary for this milestone. — decided by Morpheus

📌 Team update (2026-07-28T17:59:54-07:00): No existing Vulkan EP for ORT and no Rust crate for ORT plugin-EP bindings. We write raw FFI from scratch. — decided by Fact Checker
