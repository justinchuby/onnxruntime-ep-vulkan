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

### 2026-07-28T17:59:54-07:00 — OQ-1: Extension availability for Morpheus's capability set

**Primary-source data from vulkan.gpuinfo.org (pulled 2026-07-28):**
- `VK_KHR_synchronization2` Android coverage: **68.57%** (gap: 31.43%). Morpheus's assumption of "near-universal" is WRONG for Android.
- `VK_EXT_subgroup_size_control` Android coverage: **85.88%** (gap: 14.12%).
- `VK_KHR_synchronization2` Linux: 99.05%. Windows: 87.78%. macOS: 97.5%. iOS: 100%.
- `VK_EXT_subgroup_size_control` Linux: 98.81%. Windows: 93.33%. macOS: **100%**. iOS: **100%**.
- **macOS 100% for subgroup_size_control = extension string present, NOT feature VK_TRUE.** MoltenVK reports Vulkan 1.3 (so the extension is in core), but `subgroupSizeControl` feature is VK_FALSE because Metal cannot control SIMD group size per pipeline. The `gpuinfo.org` number counts extension string presence, not feature flag truthiness.
- `maxComputeWorkGroupInvocations` on Android: only ~1% of database reports 128. The ≥ 256 requirement is safe in practice even though the Vulkan spec minimum is 128.
- `maxComputeSharedMemorySize ≥ 16 KiB`: safe. 16,384 is the Vulkan spec minimum; universal on conformant devices.
- Subgroup BASIC: spec-guaranteed in compute on Vulkan 1.1+. Subgroup ARITHMETIC: not spec-required but >95% coverage; always query.
- `VK_LAYER_KHRONOS_synchronization2` exists (Khronos Vulkan-ExtensionLayer) and can be bundled in Android APK to provide sync2 emulation on Vulkan 1.0/1.1 devices with no dual code path in the EP. Already used by wgpu, Dawn, Godot.
- **gpuinfo.org database skews toward developer/enthusiast hardware.** Real installed-base gaps for budget Android are likely larger than shown. Always caveat database figures accordingly.
- The missing 31% of Android sync2 support is primarily: Adreno 5xx (frozen pre-2021 OEM blobs), Mali Bifrost on MediaTek with no driver update cadence, and Adreno 6xx on frozen Android 10/11 OEM drivers.

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

### 2026-07-28T19:16:08-07:00 — Frozen capability set, Option B rejected, OQ-12 specified

**CORRECTION — Option B (bundle Khronos sync2 shim layer) recommendation was wrong and the cited precedent did not exist:**
- The Khronos `VK_LAYER_KHRONOS_synchronization2` shim cannot be enabled by a plugin `.so` on retail Android. The AOSP Vulkan loader ignores `VK_LAYER_PATH`, uses no JSON manifests, and enumerates implicit layers only from the host application's `nativeLibraryDir` (set by the framework at process launch) plus `/data/local/debug/vulkan` (debuggable/userdebug builds only). We are a plugin `.so` inside someone else's process; we do not own the APK.
  - *Source: KhronosGroup/Vulkan-Loader LoaderLayerInterface.md: "The Android loader does not use manifest files"; "There is No Support For Implicit Layers on Android"*
- The claim "wgpu, Dawn, and Godot ship VK_LAYER_KHRONOS_synchronization2" was asserted without source verification and was **wrong**. All three use legacy `vkCmdPipelineBarrier` exclusively. None ships the layer.
  - Primary sources: `wgpu-hal/src/vulkan/command.rs`, `dawn/src/dawn/native/vulkan/CommandBufferVk.cpp`, `godot/drivers/vulkan/rendering_device_driver_vulkan.cpp`
- **Working-practice rule (permanent):** Qualitative precedent claims ("project X does Y") must be verified from primary source before being cited in a recommendation. The empirical gpuinfo.org measurements were excellent because they were sourced and dated. Hold qualitative claims to that same standard. Assert then check is the failure mode; check then assert is the rule.

**Frozen capability set (DESIGN.md §7.2 — Morpheus, 2026-07-28):**
- Vulkan ≥ 1.1 core + compute queue + `maxComputeWorkGroupInvocations ≥ 256` + `maxComputeSharedMemorySize ≥ 16384` + subgroup BASIC in COMPUTE + one DEVICE_LOCAL + one HOST_VISIBLE memory type.
- **No required extensions.** Governing principle: capability shortfalls degrade op coverage, not device availability.
- `synchronization2` and `subgroup_size_control` are probed into Capabilities at device-init; they drive engine strategy only.
- Switch carries a two-backend barrier abstraction (`vk/barrier.rs`). `ep.force_legacy_barriers=1` forces the legacy path. Trinity runs full suite twice per lane with bitwise-identical numerical results required.

**OQ-12 status:**
- The 31.43% Android gap is a database claim (device lacks sync2 extension), not a usability claim (device can run correct compute). Four inferences remain entirely unverified: gate pass, shader correctness, legacy-barrier backend correctness, and performance vs. own CPU.
- Decisive devices: Adreno 5xx (e.g., Snapdragon 660 on Android 8–10, no blob updates) and Mali Bifrost on MediaTek (e.g., Helio G85/G90T).
- Experiment fully specified in PLATFORMS.md §10 and `.squad/decisions/inbox/link-oq12-experiment.md`.
- Cloud device farms (Firebase Test Lab, AWS Device Farm): provide real hardware, but require APK wrapper; whether they stock the needed sync2-missing device tiers is unverified.

**Integrator note correctly placed:**
- `VK_LAYER_KHRONOS_synchronization2` is a valid deployment note for integrators who package their own APK — if they bundle the layer `.so`, our sync2 backend lights up automatically. This is an *integrator* option, not a mechanism we ship or depend on. Documented in PLATFORMS.md §8.4 with explicit scope caveat.

**RAI-003:**
- PLATFORMS.md preamble now explicitly states all physical hardware rows are untested; all verified CI is software rasterizer or desktop. README wording proposed to Morpheus per RAI-003 requirement.
