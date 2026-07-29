# Project Context

- **Owner:** Justin Chu
- **Project:** onnxruntime-ep-vulkan — a cross-platform Vulkan plugin execution provider for ONNX Runtime, written in Rust.
- **Reference architecture:** `C:\Users\justinchu\dev\onnxruntime-mlx` — mirror its layout: `rust/src/{lib,ep,factory,sys,registry,engine,logging}.rs`, `rust/src/ops/*`, `rust/build.rs`, `tests/conformance/`, `bench/`, `python/`, `docs/DESIGN.md`.
- **Stack:** Rust (cdylib plugin EP), Vulkan 1.1+ compute, SPIR-V/GLSL shaders, ONNX Runtime C API, Python bindings, GitHub Actions CI.
- **Cross-platform mandate:** Windows, Linux, Android, macOS via MoltenVK; NVIDIA / AMD / Intel / Adreno / Mali; software rasterizer (lavapipe / SwiftShader) for GPU-less CI.
- **My focus:** Platform & Hardware Support — capability detection, driver/OS matrix, cross-compilation, CI runners
- **Created:** 2026-07-28T17:52:04-07:00

## Learnings

<!-- Append new learnings below. Each entry is something lasting about the project. -->

### 2026-07-28T17:59:54-07:00 — Vulkan baseline research

- **llama.cpp ggml-vulkan requires Vulkan 1.1, not 1.3.** The user's initial proposal cited llama.cpp as a reason to target 1.3; this is incorrect. Always verify reference-project requirements before accepting user-stated baseline claims.
- **ExecuTorch Vulkan backend requires Vulkan 1.1.** Both major Vulkan ML inference references use 1.1 + extension paths.
- **Android Vulkan split is bimodal (late 2025):** ~62% at 1.1, ~26% at 1.3, ~4% at 1.0, ~7% no Vulkan. There is no significant 1.2 bulge — the installed base skips directly from 1.1 to 1.3 as devices upgrade.
- **MoltenVK reports 1.4 (MoltenVK 1.4.0, August 2025).** Portability subset limitations are Metal-imposed and must be queried regardless of reported API version. Never treat reported Vulkan version as proof of feature support on Apple platforms.
- **lavapipe (Mesa 25.0+) reports Vulkan 1.4.** Ubuntu 22.04 LTS ships Mesa 22.0 (Vulkan 1.3). Good GPU-less CI lane on Linux.
- **SwiftShader reports Vulkan 1.3.** No 1.4 support as of 2026-07-28. Useful for Windows CI fallback.
- **Adreno A1 quirk (image truncation past Y≈48) is confirmed in Qualcomm support forum.** Avoid 2D VkImage for intermediate ML tensors; use SSBOs.
- **Adreno A2 (Adreno 830 stale cache on same-layout barrier) is confirmed in Chromium issue tracker.** Insert dummy layout transitions on Adreno 830 where barriers are used.
- **Mesa 22.0 is the minimum for Vulkan 1.3 on RADV (AMD) and ANV (Intel) on Linux.** Ubuntu 22.04 LTS satisfies this.
- **Desktop Windows Vulkan 1.3 minimum drivers:** NVIDIA 472.12, AMD Adrenalin 22.1.2, Intel 30.0.101.1325. All released in early 2022; any 2022+ driver is sufficient.
- **`VP_ANDROID_baseline_2022` (Khronos) requires only Vulkan 1.1.** There is no official Khronos Android profile yet that mandates 1.3. The CTS-passing Android floor remains 1.1 as of this writing.
