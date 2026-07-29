# Project Context

- **Owner:** Justin Chu
- **Project:** onnxruntime-ep-vulkan — a cross-platform Vulkan plugin execution provider for ONNX Runtime, written in Rust.
- **Reference architecture:** `C:\Users\justinchu\dev\onnxruntime-mlx` — mirror its layout: `rust/src/{lib,ep,factory,sys,registry,engine,logging}.rs`, `rust/src/ops/*`, `rust/build.rs`, `tests/conformance/`, `bench/`, `python/`, `docs/DESIGN.md`.
- **Stack:** Rust (cdylib plugin EP), Vulkan 1.1+ compute, SPIR-V/GLSL shaders, ONNX Runtime C API, Python bindings, GitHub Actions CI.
- **Cross-platform mandate:** Windows, Linux, Android, macOS via MoltenVK; NVIDIA / AMD / Intel / Adreno / Mali; software rasterizer (lavapipe / SwiftShader) for GPU-less CI.
- **My focus:** Vulkan Compute — device/memory/sync, SPIR-V shaders, pipelines
- **Created:** 2026-07-28T17:52:04-07:00

## Learnings

<!-- Append new learnings below. Each entry is something lasting about the project. -->

### 2026-07-28T17:59:54-07:00 — ENGINE.md authored; reference implementation study complete

**Produced:** `docs/ENGINE.md` — full Vulkan runtime and shader architecture design (9 sections,
~550 lines). Decisions inbox: `.squad/decisions/inbox/switch-engine-design.md`.

**Reference study findings:**

- **llama.cpp ggml-vulkan:** `vk_device_struct` holds compute + transfer queues, vendor ID,
  UMA flag, pinned memory list, subgroup size. Build-time `vulkan-shaders-gen` compiles GLSL
  `.comp` files into SPIR-V and embeds them as C++ headers. Specialization constants tune
  workgroup/tile sizes per vendor; push constants carry per-dispatch matrix dims. Cooperative
  matrix variants (`flash_attn_cm1.comp`, `flash_attn_cm2.comp`) are behind
  `VK_KHR_cooperative_matrix` / `VK_NV_cooperative_matrix2` capability probes. Lazy pipeline
  creation with mutex-deduplication.

- **ExecuTorch ET-VK:** Vulkan 1.1 baseline. `Context` owns device, queues, command pools,
  descriptor pool, VMA allocator. `Adapter` wraps the physical device + feature detection.
  `vTensor` supports buffer or image backing; buffer-only recommended for linear ops. GLSL
  shaders compiled offline by `glslc`; yaml variant tables drive dtype/layout specialization.
  Weight prepacking at lowering time (ExecuTorch's equivalent of our Compile phase). Descriptor
  pool per context, freed and reallocated per inference graph execution.

**Key decisions made:**
- `ash` + `gpu-allocator` as Vulkan crate stack (not vulkano, not wgpu).
- Buffer-only tensor storage for v0.
- One command buffer per subgraph (no per-op submissions).
- Per data-edge barriers (`vkCmdPipelineBarrier2`), not global barriers.
- GLSL → SPIR-V at build time, embedded in cdylib; no runtime shader compiler.
- `synchronization2` and `VK_EXT_subgroup_size_control` are the only 1.3 features that
  structurally simplify engine code. Morpheus decides the final baseline.

**Unverified / follow-up:**
- MoltenVK 1.3 feature coverage completeness — Link is analyzing.
- `bufferDeviceAddress` on MoltenVK: assessed as partial; do not rely on in v0.
- `VK_KHR_cooperative_matrix` on Android: rare as of 2024–2026; capability-gated path only.
- vulkano advanced-feature gap (bindless, subgroup extensions) — assessed from issue tracker
  history, not verified against current vulkano HEAD.
