# Platform Support Matrix — onnxruntime-ep-vulkan

> **Document owner:** Link (Platform & Hardware Support Engineer)
> **Last updated:** 2026-07-28T19:16:08-07:00
> **Status:** Active — §8 reflects frozen decision (DESIGN.md §7, 2026-07-28T19:16:08-07:00)

This document is the evidence base for the project's platform support decisions. §1–§7 record the investigation and reasoning leading to the frozen capability set. §8 records what was decided, the data behind it, and the outstanding experiment needed to validate the Android half of that decision. §9 specifies the CI requirement for the dual barrier-backend parity lane. §10 documents the OQ-12 hardware validation experiment.

> **CI coverage status (RAI-003):** All physical hardware rows in §5 are marked **untested**. Every verified CI lane today runs either a software rasterizer (lavapipe on Linux, SwiftShader on Windows) or a desktop GPU build. No Android device or Apple hardware has been tested in CI. This is not hidden: it is stated explicitly here and should be reflected in the README. See §9 for the CI lanes that exist and §10 for the hardware that needs to exist.

---

## Table of Contents

1. [Vulkan Version Reality Check by Platform](#1-vulkan-version-reality-check-by-platform)
2. [What Vulkan 1.3 Actually Buys Us](#2-what-vulkan-13-actually-buys-us)
3. [The Alternative: Lower Baseline + Optional Extensions](#3-the-alternative-lower-baseline--optional-extensions)
4. [Recommendation](#4-recommendation)
5. [Platform Support Matrix Table](#5-platform-support-matrix-table)
6. [Feature & Extension Detection Strategy](#6-feature--extension-detection-strategy)
7. [Toolchain & CI Notes](#7-toolchain--ci-notes)
8. [OQ-1: Extension Coverage Data and the Frozen Decision](#8-oq-1-extension-coverage-data-and-the-frozen-decision)
9. [CI Requirement: Dual Barrier-Backend Parity Lane](#9-ci-requirement-dual-barrier-backend-parity-lane)
10. [OQ-12: Hardware Validation Experiment](#10-oq-12-hardware-validation-experiment)

---

## 1. Vulkan Version Reality Check by Platform

### 1.1 Windows — NVIDIA / AMD / Intel (Desktop Drivers)

**NVIDIA:**
Vulkan 1.3 support arrived with GeForce/Quadro driver **472.12** (released early 2022) for Maxwell 2nd-gen GPUs and newer. All WHQL-signed drivers from 2022 onward (including the current 5xx series) expose Vulkan 1.3 on supported hardware. Older Kepler/Maxwell 1st-gen cards max out at 1.2.
*Source: [NVIDIA Vulkan developer driver release notes](https://developer.nvidia.com/vulkan-driver); [Softpedia driver listing for 551.70](https://drivers.softpedia.com/get/GRAPHICS-BOARD/NVIDIA/NVIDIA-GeForce-Graphics-Vulkan-1-3-Driver-551-70-Beta-64-bit.shtml)*

**AMD:**
Adrenalin Edition **22.1.2** (January 2022) added Vulkan 1.3 for RDNA2/RDNA1/Vega/Polaris on Windows. Any driver in the 2023–2026 Adrenalin 23.x/24.x/25.x stream is sufficient.
*Source: [AMD Radeon Vulkan driver table (GPUOpen)](https://gpuopen.com/version-table/)*

**Intel:**
Driver **30.0.101.1325** (early 2022) introduced Vulkan 1.3 for 11th-gen (Tiger Lake) and newer integrated/discrete graphics (Arc). Anything ≥ 31.0.101.4255 (mid-2023) provides complete, stable Vulkan 1.3.
**⚠ Unverified:** Exact minimum driver version for Intel 10th-gen (Ice Lake) with Vulkan 1.3 is unclear — GPUInfo.org data suggests it may top out at 1.2. Mark Ice Lake **untested** until a CI lane covers it.
*Source: [Vulkan Documentation Project — Ecosystem Utilities](https://docs.vulkan.org/tutorial/latest/12_Ecosystem_Utilities_and_Compatibility.html)*

**Desktop Windows conclusion:** Vulkan 1.3 is available on all current Windows GPU drivers released since Q1 2022, which is now a 4-year-old requirement. Any user running a driver old enough to predate 1.3 is also running an unpatched, security-vulnerable driver. Requiring 1.3 on Windows imposes **zero practical regression** versus the existing 2024–2026 driver population.

---

### 1.2 Linux — Proprietary Drivers + Mesa RADV / ANV

**NVIDIA proprietary (Linux):** The same driver branch ships on Windows and Linux; Vulkan 1.3 parity is maintained. Driver 470.x+ (2021) covers most installations; anything in the 535+ LTS or 555+ production branch is fully Vulkan 1.3.

**AMD proprietary (AMDGPU-PRO):** Less commonly used in CI contexts; the open-source RADV path is preferred. AMDGPU-PRO packages from 2022+ support Vulkan 1.3 via a bundled AMDVLK or RADV.

**Mesa RADV (AMD open-source):** Vulkan 1.3 support landed in **Mesa 22.0** (March 2022) for GCN1 and newer GPUs.
*Source: [Mesa RADV documentation](https://docs.mesa3d.org/drivers/radv.html); [Phoronix: Intel Vulkan 1.3 in Mesa 22.0](https://www.phoronix.com/news/Intel-Mesa-22.0-Vulkan-1.3)*

**Mesa ANV (Intel open-source):** Vulkan 1.3 support also landed in **Mesa 22.0** for Gen7+ integrated graphics.
*Source: [Phoronix: Intel Vulkan 1.3 in Mesa 22.0](https://www.phoronix.com/news/Intel-Mesa-22.0-Vulkan-1.3)*

Ubuntu 22.04 LTS ships Mesa 22.0; Ubuntu 24.04 LTS ships Mesa 24.0. Both are Vulkan 1.3 capable. The LTS-to-LTS landscape covers Vulkan 1.3 on Linux desktops.

---

### 1.3 Android — Adreno / Mali / Xclipse

This is where Vulkan 1.3 as a hard baseline becomes a real problem.

**Installed-base distribution (late 2025, Google Play, handheld):**

| Vulkan version | Share of installed base |
|---|---|
| None | ~7% |
| 1.0 | ~4% |
| 1.1 | **~62%** |
| 1.3 | ~26% |
| 1.4 | ~1% |

*Source: [Android Developer Distribution Dashboard](https://developer.android.com/about/dashboards/) via published aggregate; cited in [Android NDK Vulkan Profiles docs](https://developer.android.com/ndk/guides/graphics/android-vulkan-profile)*

**What "Vulkan 1.1" at 62% means in practice:** Android mandated Vulkan 1.1 for all new 64-bit Android 10 devices (2019). Every mid-range and budget device shipped between 2019 and 2022 lands in this bucket — Snapdragon 660/720 with Adreno 512/618, Mali G52/G57 on MediaTek devices, all running proprietary binary blobs that the OEM froze at 1.1.

**Adreno (Qualcomm):** Adreno 6xx on Snapdragon 865+ (2020+ flagships) began reporting Vulkan 1.3. Adreno 7xx/8xx (Snapdragon 8 Gen 1 onward, 2022+) are consistently 1.3. The Mesa Turnip open-source Adreno driver reached Vulkan 1.3 for A6xx+ in late 2024.
**⚠ Unverified:** Exact minimum Adreno driver version (e.g., Qualcomm driver blob version string) at which 1.3 first reports is not confirmed by official Qualcomm documentation; community data suggests driver string v530+.

**Mali (ARM):** ARM Valhall GPUs (Mali-G77 and newer, shipping in 2020+ MediaTek Dimensity and Exynos mid-range) support Vulkan 1.3 with current OEM driver packages. Older Bifrost (G72/G76) cap at 1.1. The Panfrost open-source Mesa driver reached Vulkan 1.2+ on Bifrost; Valhall coverage in Panfrost is still catching up.

**Xclipse (Samsung/AMD RDNA2, Exynos 2200+):** Reports Vulkan 1.3 since launch (2022). RDNA2 is a modern architecture — not a concern.

**Android Vulkan Profile baseline:** The `VP_ANDROID_baseline_2022` profile (Khronos standard for CTS-passing devices as of 2022) requires only **Vulkan 1.1** core. There is no `VP_ANDROID_baseline_2024` yet that raises this to 1.3.
*Source: [Khronos Vulkan Profiles — VP_ANDROID_baseline_2022](https://github.com/KhronosGroup/Vulkan-Profiles/blob/main/profiles/VP_ANDROID_baseline_2022.json)*

**Android conclusion:** Requiring Vulkan 1.3 as a hard minimum on Android would **exclude approximately 66–73% of the global Android installed base** (the 62% at 1.1 plus the 4% at 1.0 and the 7% with no Vulkan). This is only acceptable if the project explicitly targets "high-end Android 2022+" with a documented minimum of Android 12+ on Snapdragon 8 Gen 1 or equivalent. That should be a deliberate product decision, not an accidental side-effect of an API baseline choice.

---

### 1.4 macOS and iOS via MoltenVK

**Reported Vulkan version:** MoltenVK 1.4.0 (released August 2025) reports **Vulkan 1.4** to the application, implementing it via Metal.
Prior to 1.4.0, MoltenVK 1.3.x series reported **Vulkan 1.3**.
*Source: [MoltenVK GitHub releases](https://github.com/KhronosGroup/MoltenVK/releases); [Khronos blog — MoltenVK 1.3 for Apple devices](https://www.khronos.org/news/permalink/moltenvk-1.3-released-for-vulkan-1.3-support-on-apple-devices)*

**Minimum OS requirement (latest MoltenVK):** macOS 12, iOS 15, tvOS 15.

**Portability subset limitations that matter for ML compute:**

MoltenVK exposes `VK_KHR_portability_subset` and advertises a subset of Vulkan features that Metal actually supports. Key limitations for a compute-only EP:

| Feature | Native Vulkan 1.3 | MoltenVK/Metal (2024) | Impact for EP |
|---|---|---|---|
| `shaderFloat16` / `shaderInt8` | Optional device feature | Supported on Apple Silicon; **unverified on Intel Mac** | Required for quantized inference — gate on capability query |
| 64-bit atomics in SSBOs | Yes (1.2+) | **Not supported** (Metal limitation) | Would break int64 reduction shaders — do not use |
| Buffer device address | Required in 1.3 | Partial — Metal 3 / Apple Silicon only | Avoid as a required path; use explicit binding |
| Descriptor indexing | Optional 1.2 | Partial / emulated | Do not depend on large descriptor arrays |
| Inline uniform blocks | 1.3 core | Partial or unsupported | Fall back to push constants |
| Max workgroup size | Hardware-specified | Metal enforces lower effective limit in some cases | Parameterize workgroup size via specialization constants |

**MoltenVK conclusion:** MoltenVK today reports ≥ Vulkan 1.3 (actually 1.4 in the latest release). It is *not* the limiting factor for choosing a 1.3 baseline on macOS. However, the portability subset means several 1.3 features that are "guaranteed" on Vulkan 1.3 drivers are absent or limited on Metal. The EP must query `VK_KHR_portability_subset` and gate specific features at runtime regardless of the reported API version.

---

### 1.5 Software Rasterizers for CI

**lavapipe (Mesa, CPU Vulkan):**
Mesa 22.0 → Vulkan 1.3 conformant.
Mesa 25.0 (February 2025) → reports **Vulkan 1.4** (full promoted extension coverage; conformance testing pending as of early 2025).
*Source: [Mesa 25.0 release announcement](https://lists.freedesktop.org/archives/mesa-dev/2025-February/226464.html); [Vulkanised 2025 lavapipe talk](https://vulkan.org/user/pages/09.events/vulkanised-2025/T5-Lucas-Fryzek-Igalia.pdf)*
Available on Linux CI runners without GPU hardware. Requires LLVM 10+. **Preferred CI software rasterizer.**

**SwiftShader (Google, CPU Vulkan):**
Reports **Vulkan 1.3**. No announced 1.4 support as of 2026-07-28.
*Source: [SwiftShader GitHub repository](https://github.com/google/swiftshader)*
Useful as a Windows CI fallback where lavapipe is not available, and for Android GPU-less emulator testing. Does not support all 1.3 extensions uniformly.

**CI conclusion:** Both software rasterizers cover Vulkan 1.3. Either can serve as a GPU-less correctness lane. lavapipe is preferred on Linux CI due to broader extension support.

---

## 2. What Vulkan 1.3 Actually Buys Us

These are the extensions promoted to core in Vulkan 1.3, evaluated for relevance to a **compute-only** ML execution provider:

| Feature / Promoted Extension | Relevant to Compute EP? | Notes |
|---|---|---|
| `VK_KHR_synchronization2` | **Yes — meaningful** | Cleaner barrier API; 64-bit stage/access flags. Makes pipeline barrier code less error-prone. Worth having, but the old barrier API also works. |
| `VK_KHR_dynamic_rendering` | **No — irrelevant** | Graphics-only (removes render passes/framebuffers). A compute-only EP never uses render passes. This is explicitly not a reason to require 1.3. |
| `VK_KHR_maintenance4` | Marginal | Allows querying pipeline layout memory requirements, localSizeId specialization constant. Nice QoL; not load-bearing. |
| `VK_EXT_subgroup_size_control` | **Yes — useful for ML** | Enables requesting a specific subgroup size per compute pipeline. Critical for writing portable GEMM/reduction kernels without vendor-specific hacks. Widely supported via extension even on 1.1/1.2 drivers. |
| `shaderIntegerDotProduct` (via `VK_KHR_shader_integer_dot_product`) | **Yes — useful for quantized ops** | Hardware-accelerated dot product on INT4/INT8 vectors; directly relevant for 8-bit quantized inference. However: support is per-device-optional even in 1.3; must be queried. |
| `VK_KHR_zero_initialize_workgroup_memory` | Useful for safety | Avoids reading uninitialized shared memory. Conformant 1.3 drivers must support it. Minor operational benefit. |
| `VK_EXT_pipeline_creation_cache_control` | Marginal | Allows non-blocking pipeline compilation. Nice for latency; not correctness-critical. |
| `VK_KHR_shader_terminate_invocation` | Not needed | For fragment shaders (discard). Irrelevant for compute. |
| Inline uniform blocks (core in 1.3) | Marginal | Saves a round-trip for small per-dispatch constants. Equivalent to push constants which are available everywhere. |

**Key 1.2 features that are far more load-bearing than any 1.3 addition:**

| Feature | Why it matters | Where promoted |
|---|---|---|
| `VK_KHR_shader_float16_int8` (features: `shaderFloat16`, `shaderInt8`) | FP16 and INT8 arithmetic in shaders — the core of quantized ML inference | 1.2 core, device-optional |
| `VK_KHR_8bit_storage` / `VK_KHR_16bit_storage` | Load/store FP16/INT8 from SSBOs — required for weight tensors | 1.2 core, device-optional |
| `VK_KHR_timeline_semaphore` | Required for multi-queue async compute and dependency graphs | 1.2 core |
| `VK_KHR_buffer_device_address` | GPU-side pointer arithmetic; used by some shader designs for dynamic dispatch tables | 1.2 core, mandated in 1.3 |
| `VK_EXT_descriptor_indexing` | Large descriptor arrays for weight tensors in some architectures | 1.2 core |

**Conclusion:** The genuine ML compute value of moving from 1.2 to 1.3 is: cleaner sync API (`synchronization2`) and subgroup size control in core. These are real but not dramatic wins. The most important ML features — fp16/int8 arithmetic, 8/16-bit storage, timeline semaphores — all became core in **1.2**, not 1.3.

---

## 3. The Alternative: Lower Baseline + Optional Extensions

Multiple established Vulkan ML backends chose a lower baseline:

- **ExecuTorch Vulkan backend:** Requires Vulkan **1.1**. Queries and optionally enables fp16/int8 features at runtime.
  *Source: [ExecuTorch Vulkan backend overview](https://docs.pytorch.org/executorch/stable/backends/vulkan/vulkan-overview.html)*
- **llama.cpp ggml-vulkan:** Baseline is Vulkan **1.1**. (The user cited llama.cpp as evidence for 1.3, but this is incorrect — ggml-vulkan requires 1.1 and queries extensions at device init.) Requires `VK_KHR_shader_float16_int8` (soft-required; falls back to fp32 on devices lacking it), `VK_KHR_16bit_storage`, and subgroup support.
  *Source: [ggml-vulkan Docker build recipe](https://github.com/ggml-org/llama.cpp/blob/master/.devops/vulkan.Dockerfile); [Llama.cpp ArchWiki](https://wiki.archlinux.org/title/Llama.cpp)*

### Cost of the lower-baseline approach

With a Vulkan 1.1 or 1.2 core baseline the project must:

1. **Query every optional feature explicitly** at device initialization (`VkPhysicalDeviceShaderFloat16Int8Features`, `VkPhysicalDevice16BitStorageFeatures`, `VkPhysicalDeviceSubgroupSizeControlProperties`, `VkPhysicalDeviceVulkan12Features`, etc.).
2. **Maintain two SPIR-V paths** for operators that can exploit fp16 vs. must fall back to fp32 — roughly doubling shader compilation work.
3. **Gate dispatch logic** on a device capability struct that gets passed everywhere.
4. **Handle synchronization2 absence** by using the pre-1.3 barrier API on old paths.

The complexity cost is real but well-understood and industry-proven (both ExecuTorch and llama.cpp live here). It is **not** prohibitive for a project of this scope.

---

## 4. Recommendation

> **This recommendation is advisory. The final API baseline decision belongs to Morpheus.**

### Recommended baseline: Vulkan 1.2 core + mandatory device extensions

**What:** Require the Vulkan instance to report API version ≥ 1.2. Additionally require the following device-level features (query via feature chain, fail device enumeration if absent):
- `shaderFloat16` (from `VkPhysicalDeviceShaderFloat16Int8Features`)
- `storageBuffer16BitAccess` (from `VkPhysicalDevice16BitStorageFeatures`)
- Timeline semaphore support (`timelineSemaphore` from `VkPhysicalDeviceVulkan12Features`)
- Subgroup support with `COMPUTE` stage bit and at least `BASIC` + `ARITHMETIC` operations

**What it enables:**
- All CI software rasterizers (lavapipe, SwiftShader)
- All desktop Windows/Linux drivers (2022+)
- macOS via MoltenVK (reports 1.4; portability subset queries still required)
- **Android: approximately 26% of installed base** (the fraction reporting 1.3 — plus any 1.1 devices that happen to expose the required 1.2 device extensions, which is unverified)

**Trade-off statement:** If Android support beyond high-end 2022+ devices matters to the project, the baseline should be **Vulkan 1.1 core + required extensions** instead (matching ExecuTorch). That approach reaches ~89% of Androids with Vulkan at the cost of more capability-detection code. If the project is laser-focused on desktop inference with Android as a stretch goal, **Vulkan 1.2 core** is a reasonable balance. If the team explicitly scopes out budget/mid-range Android (pre-2022 flagships), the user's proposed **Vulkan 1.3** baseline is technically justified and reduces code complexity significantly at the cost of Android market reach.

### What is explicitly NOT recommended

**Vulkan 1.3 as a hard baseline today** is not recommended **if Android coverage is a goal**, because:
1. llama.cpp — the project the user cited — does not actually require 1.3 (it requires 1.1).
2. ~73% of Android's global installed base does not report Vulkan 1.3 (late 2025 data).
3. The incremental 1.3 features that matter for compute (synchronization2, subgroup size control) can be exposed as extension paths on 1.1/1.2 drivers — they are widely available via `VK_KHR_synchronization2` and `VK_EXT_subgroup_size_control` as standalone extensions.

Vulkan 1.3 as a baseline is **acceptable** if the project explicitly targets desktop + high-end Android 2022+, documents that as a product scope decision, and does not later claim "Android support" without qualification.

---

## 5. Platform Support Matrix Table

> **Legend:** CI coverage column: `lavapipe` = CPU software rasterizer on Linux CI, `SwiftShader` = CPU software rasterizer on Windows CI or Android emulator, `real GPU` = a CI runner with physical GPU (not yet provisioned), `untested` = no CI lane covers this path.

| OS | GPU Vendor/Driver | Minimum Driver/Version | Vulkan Version Reported | Required Features | CI Coverage |
|---|---|---|---|---|---|
| Windows 10/11 | NVIDIA (GeForce/RTX/Quadro, Maxwell 2nd gen+) | 472.12 | 1.3 | fp16/int8 arith + storage, timeline semaphore, subgroups | untested (no GPU runner) |
| Windows 10/11 | AMD (RDNA1/2/3, Vega, Polaris) | Adrenalin 22.1.2 | 1.3 | fp16/int8 arith + storage, timeline semaphore, subgroups | untested (no GPU runner) |
| Windows 10/11 | Intel (Arc, 11th-gen iGPU+) | 30.0.101.1325 | 1.3 | fp16/int8 arith + storage, timeline semaphore, subgroups | untested (no GPU runner) |
| Windows 10/11 | Software (SwiftShader) | Any recent build | 1.3 | fp16/int8 arith: ⚠ check per-build; storage: yes | **SwiftShader** |
| Linux (Ubuntu 22.04+) | NVIDIA proprietary (470+ LTS) | Driver 470.x | 1.3 | fp16/int8 arith + storage, timeline semaphore, subgroups | untested (no GPU runner) |
| Linux (Ubuntu 22.04+) | AMD (Mesa RADV, Mesa 22.0+) | Mesa 22.0 | 1.3 | fp16/int8 arith + storage, timeline semaphore, subgroups | untested (no GPU runner) |
| Linux (Ubuntu 22.04+) | Intel (Mesa ANV, Mesa 22.0+, Gen11+) | Mesa 22.0 | 1.3 | fp16/int8 arith + storage, timeline semaphore, subgroups | untested (no GPU runner) |
| Linux (any distro, Mesa 22.0+) | CPU (lavapipe) | Mesa 22.0 | 1.3 (Mesa 25.0: 1.4) | All required features present | **lavapipe** |
| macOS 12+ | Apple Silicon (MoltenVK 1.3+) | MoltenVK 1.3.x | 1.3 (1.4 with MoltenVK 1.4) | portability_subset query required; no int64 atomics; buffer_device_address: Apple Silicon M1+ only | untested |
| macOS 12+ | Intel Mac (MoltenVK 1.3+) | MoltenVK 1.3.x | 1.3 | portability_subset query required; shaderFloat16 on Intel Mac: **unverified** | untested |
| iOS 15+ | Apple (MoltenVK) | MoltenVK 1.3.x | 1.3 | portability_subset query required | untested |
| Android 12+ (API 31+) | Qualcomm Adreno 730/740/750/8xx | OEM driver | 1.3 | fp16/int8 arith + storage, subgroups; ⚠ see Adreno quirks | untested |
| Android 12+ (API 31+) | ARM Mali Valhall (G77/G78/G710+) | OEM driver | 1.3 | fp16/int8 arith + storage, subgroups; ⚠ see Mali quirks | untested |
| Android 12+ (API 31+) | Samsung Xclipse (RDNA2) | OEM driver | 1.3 | fp16/int8 arith + storage, subgroups | untested |
| Android 10–11 (API 29–30) | Adreno 6xx (Snapdragon 865+) | OEM driver | 1.1–1.3 (varies) | extension paths; shaderFloat16 usually present | untested |
| Android 10–11 (API 29–30) | Adreno 5xx / Mali Bifrost (budget) | OEM driver | 1.1 | shaderFloat16: device-optional; storage: device-optional | untested |
| Android (emulator, CI) | Software (SwiftShader via AVD) | Android emulator | 1.3 | Partial extension coverage | untested (no Android CI lane) |

---

## 6. Feature & Extension Detection Strategy

### 6.1 Capability Query Architecture

All capability detection must happen at **device initialization**, before any pipeline is compiled. The output is a `DeviceCapabilities` struct (Rust) that is immutable for the lifetime of the device handle.

```
struct DeviceCapabilities {
    api_version: VkVersion,          // reported instance API version
    fp16_arithmetic: bool,           // shaderFloat16 feature
    int8_arithmetic: bool,           // shaderInt8 feature
    fp16_storage: bool,              // storageBuffer16BitAccess
    int8_storage: bool,              // storageBuffer8BitAccess
    timeline_semaphore: bool,        // timelineSemaphore (required at 1.2)
    synchronization2: bool,          // VK_KHR_synchronization2 ext OR Vulkan 1.3
    subgroup_size_control: bool,     // VK_EXT_subgroup_size_control ext OR 1.3
    subgroup_min_size: u32,
    subgroup_max_size: u32,
    subgroup_ops: VkSubgroupFeatureFlags,  // must include BASIC + ARITHMETIC
    portability_subset: bool,        // VK_KHR_portability_subset present
    shader_int_dot_product: bool,    // shaderIntegerDotProduct (1.3+)
    max_compute_workgroup_invocations: u32,
    max_push_constants_size: u32,
    max_storage_buffers_per_stage: u32,
}
```

### 6.2 Graceful Degradation Paths

| Feature missing | Fallback |
|---|---|
| `fp16_arithmetic` false | Use fp32 compute paths; log info on first use |
| `int8_arithmetic` false | Dequantize to fp32 at load time; run fp32 |
| `fp16_storage` false | Load weights as fp32 (double memory cost; warn) |
| `synchronization2` false | Use Vulkan 1.0/1.1 pipeline barrier API |
| `subgroup_size_control` false | Use fixed workgroup sizes tuned per vendor heuristic |
| `shader_int_dot_product` false | Use explicit multiply-accumulate loop in shader |
| `portability_subset` true | Enable Metal-compatible paths: no int64 atomics, no bindless |

### 6.3 Known Driver Quirks Watchlist

This is a living list. Each entry must have: symptom → affected hardware → driver version → workaround → upstream tracking link.

#### Adreno (Qualcomm)

| # | Symptom | Affected hardware | Driver version | Workaround |
|---|---|---|---|---|
| A1 | Compute write to 2D VkImage silently truncated past Y≈48 | Adreno 730/740/750 (Galaxy S23/S24) | 2150252617 | Do not use 2D images for intermediate tensors; use SSBOs exclusively |
| A2 | Stale sampled-read after storage-write (cache not flushed on same-layout barrier) | Adreno 830 | reported 2025 | Insert dummy layout transition; use explicit `TRANSFER_SRC_OPTIMAL` → `SHADER_READ_ONLY_OPTIMAL` |
| A3 | Shader miscompilation on complex control flow | Adreno 6xx various | Various | Simplify shader control flow; validate with glslang + spirv-val; test with Adreno GPU Inspector |

*Sources: [Qualcomm support forum — Adreno 730/740/750 image write truncation](https://mysupport.qualcomm.com/supportforums/s/question/0D5dK00000M2ybMSAR/); [Chromium issue tracker — Adreno 830 stale cache](https://issues.chromium.org/issues/526528807)*

#### Mali (ARM)

| # | Symptom | Affected hardware | Driver version | Workaround |
|---|---|---|---|---|
| M1 | Compute shader deadlock/freeze during async compute | Mali G52/G57/G72/G76/G77 | Various OEM blobs | Avoid simultaneous compute + texture sampling in the same command buffer on affected devices |
| M2 | Driver crash on complex SPIR-V with certain buffer/image format combos | Mali G52/G72/G76 (MediaTek/Unisoc) | Various | Simplify descriptor sets; test with Mali GPU Inspector; file OEM bug report |

*Sources: [Unity forum — compute shader freezes on mobile Vulkan](https://discussions.unity.com/t/rendering-stops-freezes-on-some-mobile-gpus-using-vulkan-compute-shaders/945225); [Unreal Engine forum — Android GPU artifacts and crashes](https://forums.unrealengine.com/t/artifacts-and-crashes-on-some-android-gpus-and-versions-when-vulkan-is-enabled/2536208)*

#### MoltenVK / Metal

| # | Symptom / Limitation | Scope | Notes |
|---|---|---|---|
| MVK1 | `VkPhysicalDevicePortabilitySubsetFeaturesKHR` must be queried and respected | All macOS/iOS | Without this query, some Vulkan calls that are valid on native drivers will crash or silently corrupt. Always enable the feature struct query unconditionally. |
| MVK2 | 64-bit atomics in SSBOs not supported (Metal limitation) | All macOS/iOS | Remove any `AtomicStore` / `AtomicCompareExchange` on int64 from compute shaders targeting Apple platforms |
| MVK3 | `bufferDeviceAddress` only on Metal 3 / Apple Silicon | Intel Mac excluded | Do not use as a required path; use explicit binding index for buffer lookups |
| MVK4 | Descriptor indexing partially emulated | All macOS/iOS | Avoid large (>128) descriptor array sizes; prefer explicit binding model |

*Sources: [MoltenVK Runtime User Guide](https://github.com/KhronosGroup/MoltenVK/blob/main/Docs/MoltenVK_Runtime_UserGuide.md); [Vulkanised 2024 — MoltenVK for Advanced Vulkan Renderers](https://vulkan.org/user/pages/09.events/vulkanised-2024/vulkanised-2024-roman-kuznetsov-meta.pdf)*

#### Older Mesa (lavapipe / ANV)

| # | Symptom | Version | Notes |
|---|---|---|---|
| LVP1 | `subgroupSize` reports 1 (subgroups not emulated in software) | lavapipe < Mesa 22.0 | CI must pin Mesa ≥ 22.0. Ubuntu 22.04 minimum. |
| ANV1 | Some specialization constant + `LocalSizeId` combos mis-handled | Mesa ANV < 23.x | Avoid `LocalSizeId` on ANV; use push constant for workgroup size on pre-23.x |

---

## 7. Toolchain & CI Notes

### 7.1 Vulkan SDK Provisioning

| OS | Preferred method | Notes |
|---|---|---|
| Linux (Ubuntu) | `apt install libvulkan-dev vulkan-validationlayers spirv-tools glslang-tools` (Ubuntu 22.04+ repos) or Vulkan SDK tarball from lunarg.com | Validation layers must be pinned to the SDK version used |
| Windows | [LunarG Vulkan SDK installer](https://vulkan.lunarg.com/sdk/home) (sets `VULKAN_SDK` env var) | Needed by ash/vulkano crate build scripts |
| macOS | Vulkan SDK tarball from LunarG, or via Homebrew `vulkan-loader` + MoltenVK | Set `VK_ICD_FILENAMES` to point to MoltenVK ICD JSON |
| Android | NDK bundles Vulkan headers; runtime loader is system-provided | Vulkan validation layers APK available from [GitHub releases](https://github.com/KhronosGroup/Vulkan-ValidationLayers/releases) |

### 7.2 Android NDK Cross-Compilation

- Target ABI: `arm64-v8a` (primary); `x86_64` for emulator (secondary).
- NDK version: r27+ for Rust NDK toolchain support and Vulkan 1.3 headers.
- Rust target: `aarch64-linux-android` — requires `cargo-ndk` or a custom `build.rs` that invokes `clang` from `$NDK_ROOT/toolchains/llvm/prebuilt/*/bin`.
- Linker: use `lld` via NDK; set `CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER`.
- Min API level: 29 (Android 10) if targeting all Vulkan 1.1 devices; 31 (Android 12) if targeting only Vulkan 1.3.

### 7.3 Validation Layers

Enable `VK_LAYER_KHRONOS_validation` unconditionally in debug builds. In CI, treat any validation error as a test failure (set `VK_EXT_debug_utils` callback to `panic!` on errors). Do not suppress validation errors — they predict real driver bugs on strict implementations.

### 7.4 GPU-less CI Lane

Confirmed working approaches for CI without physical GPU hardware:

**Linux (preferred):**
```yaml
# GitHub Actions example
- name: Install lavapipe (Mesa software Vulkan)
  run: |
    sudo apt-get install -y libvulkan-dev mesa-vulkan-drivers vulkan-tools
    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json
```
lavapipe on Ubuntu 22.04 reports Vulkan 1.3 and passes the validation layers cleanly. All compute dispatch tests can run this way.

**Windows:**
SwiftShader DLLs must be placed alongside the binary or pointed to via `VK_ICD_FILENAMES`. Google does not publish pre-built SwiftShader binaries; the CI step must either build SwiftShader from source or use a cached artifact. This is a nontrivial CI setup cost.

**Android (emulator):**
Android AVD with API 33+ uses SwiftShader-based Vulkan 1.3 by default. Available via `avdmanager` / `emulator` in the Android SDK. Useful for build + basic correctness smoke tests, but the emulator's Vulkan stack is not representative of real device driver behavior. Mark emulator tests as `ci:emulator` not `ci:android-device` in test metadata.

**What GPU-less CI does NOT cover:**
- Vendor-specific shader compilation paths
- Hardware subgroup semantics
- Driver-specific quirk workarounds (Adreno A1/A2, Mali M1/M2 above)
- Real fp16 throughput and memory bandwidth

For anything in the matrix column labeled **untested**, the project must either acquire CI access to that hardware or document the platform as "community-supported" with no CI guarantee.

---

*This document is owned by Link. Updates to the support matrix should be proposed via the decisions inbox and reviewed by the Fact Checker before merging. Hardware additions require CI coverage or explicit "untested" marking.*

---

## 8. OQ-1: Extension Coverage Data and the Frozen Decision

> **Status: RESOLVED** — `docs/DESIGN.md` §7 frozen 2026-07-28T19:16:08-07:00 by Morpheus. The evidence below drove the decision. The data and sources are preserved here because DESIGN.md cites them and they remain the authoritative measurement basis.

### 8.1 What was being measured and why

The pre-decision capability-set proposal required devices to expose either Vulkan 1.3 core or the standalone extensions `VK_KHR_synchronization2` and `VK_EXT_subgroup_size_control`. The question was whether that requirement was safe — i.e., whether those extensions are "near-universally available on 1.1/1.2 drivers", or whether requiring them would exclude a meaningful device population.

All figures below are primary-source data from **[vulkan.gpuinfo.org](https://vulkan.gpuinfo.org/)** (Sascha Willems, CC-BY 4.0), pulled **2026-07-28** directly from the page HTML. The database skews toward developer-submitted reports from newer/higher-end hardware; real installed-base fractions for budget/legacy Android are likely **worse** than shown.

### 8.2 Extension coverage data (vulkan.gpuinfo.org, 2026-07-28)

#### VK_KHR_synchronization2

Coverage = devices that expose the extension string **or** report Vulkan 1.3 (where it is core).

| Platform | Coverage | Gap |
|---|---|---|
| Windows | 87.78% | **12.22%** |
| Linux | 99.05% | 0.95% |
| Android | 68.57% | **31.43%** ← decisive |
| macOS (MoltenVK) | 97.5% | 2.5% |
| iOS (MoltenVK) | 100% | 0% |

*Source: [vulkan.gpuinfo.org — VK_KHR_synchronization2](https://vulkan.gpuinfo.org/displayextensiondetail.php?extension=VK_KHR_synchronization2), pulled 2026-07-28*

#### VK_EXT_subgroup_size_control

Coverage = devices that expose the extension string **or** report Vulkan 1.3.

| Platform | Coverage | Gap |
|---|---|---|
| Windows | 93.33% | 6.67% |
| Linux | 98.81% | 1.19% |
| Android | 85.88% | **14.12%** |
| macOS (MoltenVK) | 100% | 0% |
| iOS (MoltenVK) | 100% | 0% |

*Source: [vulkan.gpuinfo.org — VK_EXT_subgroup_size_control](https://vulkan.gpuinfo.org/displayextensiondetail.php?extension=VK_EXT_subgroup_size_control), pulled 2026-07-28*

**macOS 100% caveat — extension string ≠ feature flag.** The 100% macOS/iOS figure means the extension *string* is present in every MoltenVK 1.3+ report (Vulkan 1.3 promotes the extension to core, and MoltenVK reports ≥ 1.3). It does **not** mean `VkPhysicalDeviceSubgroupSizeControlFeatures::subgroupSizeControl = VK_TRUE`. Metal does not allow per-pipeline SIMD-group size control, so MoltenVK sets `subgroupSizeControl = VK_FALSE` while still advertising the struct for querying `minSubgroupSize`/`maxSubgroupSize`. The frozen decision (§8.3 below) requires only properties querying, not feature control — so macOS/iOS are fully in scope.
*Source: [wgpu issue #5551](https://github.com/gfx-rs/wgpu/issues/5551); [MoltenVK Runtime User Guide](https://github.com/KhronosGroup/MoltenVK/blob/main/Docs/MoltenVK_Runtime_UserGuide.md)*

#### Other capability-set limits: safety check

| Limit | Vulkan spec minimum | Android (gpuinfo.org, 2026-07-28) | Assessment |
|---|---|---|---|
| `maxComputeWorkGroupInvocations ≥ 256` | 128 | 80/8206 reports (≈ 1%) show 128; 74 show exactly 256 | **Safe.** < 2% of database reports 128; those are pre-2018 era devices. Database skew toward newer hardware means the real fraction is probably under 5%, concentrated on obsolete hardware. |
| `maxComputeSharedMemorySize ≥ 16 384 bytes` | 16 384 bytes | Guaranteed by VP_ANDROID_baseline_2022 | **Safe.** 16 KiB is the Vulkan spec floor; any conformant device passes by definition. |
| Subgroup BASIC in COMPUTE | Guaranteed by 1.1 spec | >99% | **Safe.** VK spec §34.1 mandates BASIC in all compute queues on Vulkan 1.1+. |

*Source: [gpuinfo.org — maxComputeWorkGroupInvocations Android](https://vulkan.gpuinfo.org/displaydevicelimit.php?name=maxComputeWorkGroupInvocations&platform=android), 2026-07-28; [Khronos Vulkan-Profiles VP_ANDROID_baseline_2022.json](https://github.com/KhronosGroup/Vulkan-Profiles/blob/main/profiles/VP_ANDROID_baseline_2022.json)*

---

### 8.3 What the data showed and what was decided

The measurement **disproved** the "near-universal" assumption for Android sync2: a **31.43-point Android gap** and a **12.22-point Windows gap** were measured from the database. The Windows number matters as much as the Android one — nearly one Windows device in eight would be declined.

The Android gap is *structurally* missing, not stochastically missing:
- **Adreno 500-series** (Snapdragon 625/630/636/660, Adreno 506/508/509/512): frozen OEM blobs predate the extension (published 2021); no update path exists.
- **Adreno 600-series on unupdated Android 10/11 OEM drivers**: some Samsung/Xiaomi/OPPO devices froze drivers at Android 10 launch; those drivers will not change.
- **Mali Bifrost (G52/G57/G72/G76) on MediaTek**: OEM driver update cadence is poor; the extension simply was not backported.

**⚠ Unverified per-model specifics.** The characterization above is based on gpuinfo.org database patterns and community driver changelogs. Exact minimum driver blob version strings at which sync2 appears have not been confirmed by official Qualcomm or ARM documentation. The *usability* of these devices — whether they can actually run a correct compute workload — is separately unverified; see §10.

**The frozen decision (DESIGN.md §7.2, 2026-07-28T19:16:08-07:00):**

> *Governing principle: capability shortfalls degrade op coverage, not device availability.*
>
> The device gate is six items and **no required extensions**:
> 1. Vulkan ≥ 1.1 core (instance and device)
> 2. A queue family with `VK_QUEUE_COMPUTE_BIT`
> 3. `maxComputeWorkGroupInvocations >= 256`
> 4. `maxComputeSharedMemorySize >= 16384`
> 5. Subgroup `BASIC` in the `COMPUTE` stage
> 6. At least one `DEVICE_LOCAL` and one `HOST_VISIBLE` memory type
>
> `synchronization2` and `subgroup_size_control` are probed into `vk::caps::Capabilities` and used to select an engine strategy. Neither may block device advertisement. `subgroup_size_control` is consulted as a *properties query* only — `subgroupSizeControl = VK_TRUE` is never required.

Consequences for this document:
- **`synchronization2` gap:** Switch implements a two-backend barrier abstraction (`vk/barrier.rs`). The legacy `vkCmdPipelineBarrier` path covers the 31.43% of Android and 12.22% of Windows that lack the extension. See §9 for the CI requirement.
- **`subgroup_size_control` gap:** `VkPhysicalDeviceSubgroupSizeControlProperties` is queried where available to inform workgroup size selection. No device is excluded.
- **macOS:** Fully in scope. MoltenVK reports the extension string; `subgroupSizeControl = VK_FALSE` is acceptable because we require only the properties query.

---

### 8.4 What was considered and rejected

**Requiring `VK_KHR_synchronization2` (the pre-decision proposal):** Excludes 31.43% of Android and 12.22% of Windows by measurement. Under the compatibility-first directive, indefensible.

**Option B — bundling `VK_LAYER_KHRONOS_synchronization2`:** Proposed by Link as a way to avoid a dual code path; rejected by Morpheus on two grounds, both verified from primary sources.

First, it cannot work for a plugin on retail Android: the AOSP Vulkan loader ignores `VK_LAYER_PATH`, uses no JSON manifests, and enumerates layers only from the host application's `nativeLibraryDir` (set by the framework from the installed APK via `GraphicsEnv::getAppNamespace()`) plus `/data/local/debug/vulkan`, which requires a debuggable app or a userdebug build. We do not own the APK. The Khronos `synchronization2_layer.md` states the `.so` "needs to be packaged inside the APK". Android was 100% of the motivation for proposing Option B; it is the one platform where a plugin cannot do it.
*Source: [KhronosGroup/Vulkan-Loader — LoaderLayerInterface.md](https://github.com/KhronosGroup/Vulkan-Loader/blob/main/docs/LoaderLayerInterface.md) ("The Android loader does not use manifest files"; "There is No Support For Implicit Layers on Android"); [KhronosGroup/Vulkan-ExtensionLayer — synchronization2_layer.md](https://github.com/KhronosGroup/Vulkan-ExtensionLayer/blob/main/docs/synchronization2_layer.md)*

Second, the cited prior art — that wgpu, Dawn, and Godot ship this layer — was **incorrect**. All three use legacy `vkCmdPipelineBarrier` exclusively and none ships the layer. The actual source inspection showed the precedent supports Option A, not Option B.
*Source: [wgpu-hal/src/vulkan/command.rs](https://github.com/gfx-rs/wgpu/blob/trunk/wgpu-hal/src/vulkan/command.rs); [dawn/src/dawn/native/vulkan/CommandBufferVk.cpp](https://github.com/google/dawn/blob/main/src/dawn/native/vulkan/CommandBufferVk.cpp); [godot/drivers/vulkan/rendering_device_driver_vulkan.cpp](https://github.com/godotengine/godot/blob/master/drivers/vulkan/rendering_device_driver_vulkan.cpp)*

> **Working-practice note.** The layer-shim Option B claim was asserted (wgpu/Dawn/Godot do X) rather than verified from source. It pointed the team toward a mechanism that does not work for the platform it was proposed for. Qualitative precedent claims must be verified from primary source before they are cited — the same standard applied to the gpuinfo.org measurements. This is recorded so it sticks.

**Optional integrator-side deployment note (NOT a mechanism we ship or depend on):** An Android application integrator who packages `libVkLayer_khronos_synchronization2.so` inside their own APK and enables it at `vkCreateInstance` time can cause our sync2 backend to light up automatically on otherwise sync2-missing devices, because we probe the extension at device initialization and the layer advertises it (disabling itself when native support is present). This is documented for integrators as an *optional* deployment choice. We do not ship the layer, do not depend on it, and do not test this configuration. The legacy barrier backend runs correctly without it.
*Source: [KhronosGroup/Vulkan-ExtensionLayer — synchronization2_layer.md](https://github.com/KhronosGroup/Vulkan-ExtensionLayer/blob/main/docs/synchronization2_layer.md)*

---

## 9. CI Requirement: Dual Barrier-Backend Parity Lane

**Requirement owner:** Link (specification). **Implementation owner:** Trinity (`.github/workflows/`).

**Background:** Because lavapipe and desktop hardware expose `synchronization2` (Linux 99%, Windows 88%), the legacy `vkCmdPipelineBarrier` backend — which covers 31% of Android and 12% of Windows — would never execute in any test we own without deliberate forcing. DESIGN.md §7.5 mandates a forced-legacy lane as an **M0 exit criterion** (item 8 of 8).

**What Trinity must implement:**

Every CI lane that runs the test suite must run it **twice**:

1. **Default run:** `cargo test --all-targets` (or the equivalent invocation per lane). The EP uses whichever barrier backend the device capability dictates — sync2 where available, legacy where not.
2. **Forced-legacy run:** the same invocation with the session option `ep.force_legacy_barriers=1` set (exact mechanism: set via ORT session option `"ep.force_legacy_barriers"` = `"1"` at session creation, or via env var `ONNXRUNTIME_VULKAN_EP_FORCE_LEGACY_BARRIERS=1` if Tank exposes one). This forces `vk/barrier.rs` to select `Barriers::Legacy` even on a device that has sync2.

**What both runs must assert:**
- All tests pass.
- Numerical outputs are **bitwise identical** between the default run and the forced-legacy run for every test that produces tensor output.
- Zero validation-layer errors in both runs.

**Lanes covered:** at minimum, the Linux lavapipe lane and the Windows SwiftShader lane. Both already expose sync2, so forced-legacy on these lanes is the only way the legacy `vkCmdPipelineBarrier` code path is exercised before real Android hardware is available.

**Failure mode this detects:** a bug in `LegacyBackend` — a mismatched stage mask, a missing barrier, an access flag not translated — that causes a different numerical result under the legacy path. This is the most valuable possible failure mode to catch, and it is the failure mode the parity lane was specifically designed for.

**When real Android hardware is available:** the parity run must also be executed on physical devices, including those in the sync2-missing population (see §10). At that point the "forced-legacy = bitwise identical" assertion is retired and replaced by "sync2 backend and legacy backend agree to within the op's tolerance", since the two backends may diverge at the floating-point rounding level on different hardware.

---

## 10. OQ-12: Hardware Validation Experiment

**Status: Pending hardware.** The experiment is fully specified; it can be executed the hour a device exists.

**The question:** Does carrying the legacy barrier backend (DESIGN.md §7.3) actually buy *usable* devices, or does the Adreno 5xx / Mali Bifrost population fail for reasons unrelated to barriers — driver bugs, unsupported memory limits, missing fp16, known Adreno quirks on the watchlist?

**How much of the 31.43% claim is currently unverified:** All of it, as a *usability* claim. The gpuinfo.org data proves those devices lack `VK_KHR_synchronization2`. It says nothing about whether they can run correct compute at all, whether they pass the §7.2 device gate, or whether Vulkan inference on them outperforms their own CPU. Until the experiment runs, every statement about "the legacy backend benefits the 31% Android population" is a database extrapolation, not a measurement.

**Can any of it be de-risked without physical devices?**

Cloud device farms offer real Android hardware at API level, which is better than an emulator but not free:

- **Firebase Test Lab** (Google): Provides real Android devices for automated test runs via `gcloud firebase test android`. Supports ADB-level test invocation; does NOT expose `VkQueueSubmit` or Vulkan compute dispatch from a `.so` plugin because tests run via Instrumentation APK or Robo, not via a native plugin call chain. **Not suitable for Vulkan compute validation** without wrapping the EP in an Android app.
- **AWS Device Farm**: Similar model. Real devices, JUnit/Appium test runner. Same limitation: compute validation requires an APK wrapper, not a raw plugin `.so`. Would work for end-to-end integration once an Android integration test APK exists.
- **Browserstack App Automate / LambdaTest**: Same architecture as AWS Device Farm.

**Verdict on cloud farms:** Cloud farms can de-risk *build and link* on real Android hardware (does `cargo-ndk` produce a loadable `.so`?), and with an APK wrapper they can run functional tests. They cannot replace physical devices for the gate-check + correctness + performance stages below without additional tooling investment. The minimum-cost path is two second-hand devices.

**⚠ Unverified:** Whether Firebase Test Lab or AWS Device Farm currently expose Vulkan 1.3 or even Vulkan 1.1 with compute on their device inventory, and whether sync2-missing devices (Adreno 5xx, MediaTek Mali Bifrost) are represented in their fleets, is **unconfirmed** and must be verified before committing to a cloud-farm-only strategy.

### 10.1 Devices — decisive, not representative

Two physical units of A and B are worth more than four of C and D. If only two devices can be obtained, take A and C.

| Slot | Device class | Rationale |
|---|---|---|
| A | **Adreno 5xx** — e.g. Snapdragon 660 (Adreno 512) or 636 (Adreno 509), Android 8–10, stock OEM blob | The largest sync2-missing bloc. Frozen pre-2021 drivers. If any class fails for non-barrier reasons, it is this one. |
| B | **Mali Bifrost on MediaTek** — e.g. G52 (Helio G85) or G76 (Helio G90T), stock ROM | Second bloc, second vendor. MediaTek specifically — same Mali IP on a Samsung/Exynos ROM has a different update history. |
| C | **Adreno 6xx on Android 12+** — e.g. Snapdragon 865/888 | Control: *has* sync2. Isolates "is the legacy backend correct" from "is this device usable". |
| D | **Mali Valhall on Android 12+** — e.g. G78 or G710 | Second control, second vendor. |

### 10.2 Three-stage workload

Each stage can independently fail the experiment:

**Stage 1 — Gate check (minutes).** Run the `vk/caps.rs` probe on the device and record:
- `vkEnumerateInstanceVersion` result
- `maxComputeWorkGroupInvocations`, `maxComputeSharedMemorySize`, memory types
- `VkPhysicalDeviceSubgroupProperties::supportedOperations` and `subgroupSize`
- `shaderFloat16`, `storageBuffer16BitAccess` (OQ-14 data point, separately tracked)
- `VK_KHR_synchronization2` presence in extension list

A device that fails the §7.2 gate outright (fails items 1–6 of the device gate) is the cleanest possible negative result and narrows any follow-on decision to the remaining population.

**Stage 2 — Correctness (hours).** The full M1 differential suite (elementwise/shape floor) run against ORT CPU EP reference, plus M2 set (reduction/GEMM/softmax/norm) if it exists. Run twice: once normally, once with `ep.force_legacy_barriers=1` on devices C and D. Validation layers on throughout. Record every failure with op, dtype, shape, and driver version.

**Stage 3 — Usability (hours).** One bandwidth-bound elementwise chain and one GEMM-anchored subgraph, timed against *that device's own ORT CPU EP* — not against a desktop GPU. Report wall time and Mouse's `boundary_time_fraction` (OP_COVERAGE.md §7.3).

### 10.3 Pass bar and reversal conditions

For the legacy backend to be vindicated on Android, devices A **and** B must:
1. Pass the §7.2 device gate (stage 1 green)
2. Pass the full differential suite with **zero** numerical failures and zero validation-layer errors (stage 2 green)
3. Beat their own device's ORT CPU EP by **≥ 1.5×** on the GEMM-anchored subgraph (stage 3 green)

**What reverses the DESIGN.md §7.3 decision, stated in advance:**

- **If A and B both fail stage 1 or 2:** the Android half of §7.3's justification is void. The legacy backend stays (12.22% Windows gap is independent and unchanged), but the Android §7.2 gate should be tightened by device class, and M3's Android scope narrows to the Adreno 6xx / Valhall tier. This is a scope decision recorded here, not quietly absorbed.
- **If A and B pass stages 1–2 but fail stage 3 (< 1.5×):** they are correct but not worth using. Legacy backend stays; devices remain supported (correctness is free once written); they are documented "runs, not recommended" with no tuning budget.
- **If A and B pass all three stages:** §7.3 is fully vindicated and the Android tier gets a real tuning budget in M3.
- **If C or D fail only under `ep.force_legacy_barriers=1`:** that is a `LegacyBackend` bug, not a device finding. It is the most valuable possible outcome and is exactly what the §9 parity lane exists to catch.

### 10.4 Owners and blocking

- **Link:** device acquisition, gate-check harness, PLATFORMS.md rows.
- **Trinity:** differential suite execution on-device.
- **Niobe:** stage 3, `boundary_time_fraction` numbers.
- **Morpheus:** rules on the outcome.

Blocked only on hardware, not on any other milestone. Stages 1 and 2 can run the day M1 is green.

---

*This document is owned by Link. Updates to the support matrix should be proposed via the decisions inbox and reviewed by the Fact Checker before merging. Hardware additions require CI coverage or explicit "untested" marking.*

### 8.1 Extension Coverage: Primary Data (vulkan.gpuinfo.org, pulled 2026-07-28)

#### VK_KHR_synchronization2

Coverage = devices that expose the extension string OR report Vulkan 1.3 (where it is promoted to core).

| Platform | Coverage (gpuinfo.org database) | Gap (devices lacking it) |
|---|---|---|
| Windows | **87.78%** | 12.22% |
| Linux | **99.05%** | 0.95% |
| Android | **68.57%** | **31.43%** |
| macOS (MoltenVK) | **97.5%** | 2.5% |
| iOS (MoltenVK) | **100%** | 0% |

*Source: [vulkan.gpuinfo.org — VK_KHR_synchronization2 device coverage](https://vulkan.gpuinfo.org/displayextensiondetail.php?extension=VK_KHR_synchronization2), data pulled 2026-07-28*

#### VK_EXT_subgroup_size_control

Coverage = devices that expose the extension string OR report Vulkan 1.3.

| Platform | Coverage (gpuinfo.org database) | Gap |
|---|---|---|
| Windows | **93.33%** | 6.67% |
| Linux | **98.81%** | 1.19% |
| Android | **85.88%** | **14.12%** |
| macOS (MoltenVK) | **100%** | 0% |
| iOS (MoltenVK) | **100%** | 0% |

*Source: [vulkan.gpuinfo.org — VK_EXT_subgroup_size_control device coverage](https://vulkan.gpuinfo.org/displayextensiondetail.php?extension=VK_EXT_subgroup_size_control), data pulled 2026-07-28*

