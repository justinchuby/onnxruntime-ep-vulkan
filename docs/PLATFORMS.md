# Platform Support Matrix — onnxruntime-ep-vulkan

> **Document owner:** Link (Platform & Hardware Support Engineer)
> **Last updated:** 2026-07-30T05:54:13-07:00
> **Status:** Active — §8 reflects frozen decision (DESIGN.md §7, 2026-07-28T19:16:08-07:00); both CI lanes working as of 2026-07-29T09:19:35-07:00. LVP2 **retracted** 2026-07-29T20:26:56-07:00 (instrument failure — was never a real quirk); see §6.3. Linux lavapipe **first claimed-node execution** completed 2026-07-30T07:52-07:00 (WSL, Mesa 25.2.8); see §7.5–§7.7.

This document is the evidence base for the project's platform support decisions. §1–§7 record the investigation and reasoning leading to the frozen capability set. §8 records what was decided, the data behind it, and the outstanding experiment needed to validate the Android half of that decision. §9 specifies the CI requirement for the dual barrier-backend parity lane. §10 documents the OQ-12 hardware validation experiment.

> **Standing directive — cross-platform generality (2026-07-29T09:39:59-07:00):** Cross-platform is the premise of this EP, not a porting phase. A Vulkan EP that works only on desktop NVIDIA has no reason to exist — better-supported vendor backends already do that job. Recorded at `.squad/decisions/inbox/copilot-directive-cross-platform.md`. The structural rules that enforce this:
> - **Derive workgroup sizes and memory budgets from reported device limits, never from observed constants.** A constant that fits 48 KiB on an RTX 4060 does not fit 32 KiB on an Iris Xe or ~16 KiB on Adreno 5xx.
> - **UMA is the mobile case, and we have one on this desk.** Intel Iris Xe and every Adreno/Mali are UMA — `DEVICE_LOCAL` and `HOST_VISIBLE` share the same physical memory. A staging path that assumes a discrete upload heap silently skips on half our targets.
> - **Intel is the spec oracle.** Correct on NVIDIA and wrong on Intel = EP relying on undefined behavior. Never fix this by special-casing Intel.
> - **`cfg`-gated definitions are mandatory for platform-conditional types.** The `ort::wchar_t` incident broke the Linux lane silently; `tests/portability.rs` enforces this structurally.
> - **§9.1.2 discipline:** a result on this desk is not a result this project has. CI proves portability; physical Android and macOS coverage is absent (OQ-12).

> **CI coverage status (RAI-003):** All physical hardware rows in §5 are marked **untested**. Both verified CI lanes today run lavapipe (a CPU software rasterizer) on Linux and Windows. No physical GPU, Android device, or Apple hardware has been tested in CI. This is stated explicitly here and must be reflected in the README. See §9 for the CI lanes that exist and §10 for the hardware that needs to exist.

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

> **Verification tiers — three distinct claims, kept separate throughout this table:**
> - **CI-verified** — a reproducible result from a CI runner; anyone who re-runs the workflow sees the same thing. This is the project's portability proof.
> - **Local-dev-verified** — observed on a developer's machine with `epctl --probe-loader`; confirms the device passes the gate and the EP loads correctly, but is not a portable result. Noted as local-dev (2026-07-29) in the CI Coverage column.
> - **untested** — no execution has occurred on this path, in CI or otherwise. All physical mobile hardware is in this category. Do not read rows added on 2026-07-29 as improving OQ-12 coverage — they do not.
>
> **Other column notes:** Vulkan version reported = what `apiVersion` the driver surfaces; minimum driver = first known version to pass the §7.2 device gate; required features = what the §7.2 gate checks plus what ops need for fp16/int8 paths.
>
> **A fourth claim, added 2026-08-01 and deliberately not a column here: can this platform produce a device-state record?** Passing the §7.2 gate says the EP can *run* on a device; it says nothing about whether a *timing figure* taken there is quotable. Under DESIGN.md §10.0 obligation 8 that needs a tenancy verdict and clock min/median/max against the board maximum, over the statistic's own suffix — and **exactly one row of this table has a producer for it today** (NVIDIA, via `nvidia-smi`). Everywhere else a device-clock figure is `STEADY_UNCERTIFIED`, which is not a penalty and not a waiver but a true statement about what we can currently know there. The per-platform breakdown is **§7.11.1**; it is kept out of this table because a device-state producer is an instrument we own, not a driver capability we detect.

| OS | GPU Vendor / Driver | Min Driver / Version | Vulkan Reported | **Memory** | Gate criteria (R1–R6) | fp16/int8 capability | CI Coverage |
|---|---|---|---|---|---|---|---|
| Windows 11 | Intel Iris Xe Graphics (Alder/Tiger Lake iGPU, **UMA**) | driver observed 31.0.101.5590 | **1.4.309** | **UMA** (observed) | R1–R6 all PASS (observed) | fp16 arith + storage: present (observed) | **local-dev 2026-07-29** |
| Windows 11 | NVIDIA GeForce RTX 4060 Laptop (**discrete**) | driver observed 572.x | **1.4.325** | **Discrete** (observed) | R1–R6 all PASS (observed) | fp16 arith + storage: present (observed) | **local-dev 2026-07-29** |
| Windows 10/11 | NVIDIA (GeForce/RTX/Quadro, Maxwell 2nd gen+) | 472.12 | 1.3+ | Discrete | fp16/int8 arith + storage, timeline semaphore, subgroup BASIC | fp16/int8: query required | untested (no CI GPU runner) |
| Windows 10/11 | AMD (RDNA1/2/3, Vega, Polaris) | Adrenalin 22.1.2 | 1.3+ | Discrete | fp16/int8 arith + storage, timeline semaphore, subgroup BASIC | fp16/int8: query required | untested (no CI GPU runner) |
| Windows 10/11 | Intel (Arc discrete) | 30.0.101.1325 | 1.3+ | Discrete | fp16/int8 arith + storage, timeline semaphore, subgroup BASIC | fp16/int8: query required | untested (no CI GPU runner) |
| Windows 10/11 | Intel (iGPU, 11th gen+) | 30.0.101.1325 | 1.3+ | **UMA** | fp16/int8 arith + storage, timeline semaphore, subgroup BASIC | fp16/int8: query required | untested (no CI GPU runner) |
| Windows Server 2025 | CPU (lavapipe, mesa-dist-win 26.1.3, **ICD registered in registry**) | mesa-dist-win 26.1.3 | 1.3 (llvmpipe, driverID MESA_LLVMPIPE) | **UMA** (probe: is_uma=true; all memory types HOST_VISIBLE; UMA path exercised in CI) | R1–R6 PASS; subgroup: `subgroup_probe_valid=true`, `subgroup_basic_in_compute=true`, ops=BASIC\|VOTE\|ARITHMETIC\|BALLOT\|SHUFFLE\|SHUFFLE_RELATIVE\|QUAD, size=8 | fp16/int8: unverified (probe reading provisional — taken before push_next fix; re-observation recommended) | **CI-verified (lavapipe)** |
| Linux Ubuntu 22.04 | CPU (lavapipe, Mesa 23.2.1 / LLVM 15.0.7) | Mesa 23.2.1 | **1.3.255** (apiVersion observed) | **UMA** (probe: is_uma=true; UMA path exercised in CI) | R1–R6 PASS; `deviceName = llvmpipe (LLVM 15.0.7, 256 bits)`; subgroup: `subgroup_probe_valid=true`, `subgroup_basic_in_compute=true`, ops=BASIC\|VOTE\|ARITHMETIC\|BALLOT\|SHUFFLE\|SHUFFLE_RELATIVE\|QUAD, size=8 | fp16/int8: unverified (probe reading provisional — taken before push_next fix; re-observation recommended) | **CI-verified (lavapipe)** |
| Linux Ubuntu 24.04 / WSL2 | CPU (lavapipe, Mesa 25.2.8 / LLVM 20.1.2) | Mesa 25.2.8 | **1.4.318** (observed 2026-07-30) | **UMA** (probe: is_uma=true; all memory types HOST_VISIBLE) | R1–R6 PASS; `deviceName = llvmpipe (LLVM 20.1.2, 256 bits)`; subgroup: `subgroup_probe_valid=true`, `subgroup_basic_in_compute=true`, ops=BASIC\|VOTE\|ARITHMETIC\|BALLOT\|SHUFFLE\|SHUFFLE_RELATIVE\|CLUSTERED\|QUAD\|ROTATE_KHR\|ROTATE_CLUSTERED_KHR, size=**8**; `maxComputeSharedMemorySize=32 KiB`; `maxComputeWorkGroupInvocations=1024`; `timestamp_period_ns=1.0`; `timestamp_valid_bits=64` | fp16/int8: unverified (probe does not test feature flags from chained structs on this device) | **local-dev verified 2026-07-30** (WSL2, not CI; sudo requires password for justinchu — run as root); all claimed ops execute end-to-end; 196 tests pass |
| Linux (Ubuntu 22.04+) | NVIDIA proprietary (470+ LTS or 535+) | Driver 470.x | 1.3+ | Discrete | fp16/int8 arith + storage, timeline semaphore, subgroup BASIC | fp16/int8: generally present | untested (no CI GPU runner) |
| Linux (Ubuntu 22.04+) | AMD (Mesa RADV, Mesa 22.0+) | Mesa 22.0 | 1.3+ | Discrete | fp16/int8 arith + storage, timeline semaphore, subgroup BASIC | fp16/int8: query required | untested (no CI GPU runner) |
| Linux (Ubuntu 22.04+) | Intel iGPU (Mesa ANV, Mesa 22.0+, Gen11+) | Mesa 22.0 | 1.3+ | **UMA** | fp16/int8 arith + storage, timeline semaphore, subgroup BASIC | fp16/int8: query required | untested (no CI GPU runner) |
| macOS 12+ | Apple Silicon (MoltenVK 1.3+) | MoltenVK 1.3.x | 1.3+ (1.4 with MoltenVK 1.4) | **UMA** | portability_subset required; no int64 atomics; buffer_device_address M1+ only | shaderFloat16: present on Apple Silicon | untested |
| macOS 12+ | Intel Mac (MoltenVK 1.3+) | MoltenVK 1.3.x | 1.3 | Discrete | portability_subset required | shaderFloat16: **unverified** | untested |
| iOS 15+ | Apple (MoltenVK) | MoltenVK 1.3.x | 1.3+ | **UMA** | portability_subset required | shaderFloat16: present on modern Apple | untested |
| Android 12+ (API 31+) | Qualcomm Adreno 730/740/750/8xx | OEM driver | 1.3 | **UMA** (SoC) | R1–R6 expected PASS; ⚠ Adreno quirks A1/A2/A3 | fp16 arith + storage: generally present | **untested (OQ-12 pending)** |
| Android 12+ (API 31+) | ARM Mali Valhall (G77/G78/G710+) | OEM driver | 1.3 | **UMA** (SoC) | R1–R6 expected PASS; ⚠ Mali quirks M1/M2 | fp16 arith + storage: generally present | **untested (OQ-12 pending)** |
| Android 12+ (API 31+) | Samsung Xclipse (RDNA2, SoC) | OEM driver | 1.3 | **UMA** (SoC) | R1–R6 expected PASS | fp16/int8: generally present | **untested (OQ-12 pending)** |
| Android 10–11 (API 29–30) | Adreno 6xx (Snapdragon 865+, with sync2) | OEM driver | 1.1–1.3 | **UMA** (SoC) | R1–R6 expected PASS; sync2 path probable | fp16 storage: generally present | **untested (OQ-12 pending)** |
| Android 10–11 (API 29–30) | Adreno 5xx / Mali Bifrost (sync2 missing) | OEM driver | 1.1 | **UMA** (SoC) | R1–R6 gate unknown; legacy barrier path; OQ-12 decisive devices | fp16/int8: device-optional | **untested (OQ-12 pending — decisive)** |
| Android (emulator, CI) | Software (SwiftShader via AVD) | Android emulator | 1.3 | N/A (software) | Partial; emulator not representative of real driver | fp16/int8: partial | untested (no Android CI lane) |

### 5.1 Memory architecture: UMA vs discrete — column interpretation

The **Memory** column in the table above is a first-class property, not an annotation. It determines which code paths execute:

| Architecture | Physical model | Who has it |
|---|---|---|
| **UMA** | CPU and GPU share the same DRAM. A memory type may be both `DEVICE_LOCAL` and `HOST_VISIBLE`. Zero-copy tensor mapping is possible; there is no separate upload heap. | Intel iGPU (all), Apple Silicon, **every Adreno, every Mali, every Xclipse on a mobile SoC** |
| **Discrete** | GPU has on-device VRAM (`DEVICE_LOCAL`). System RAM is `HOST_VISIBLE` only. Uploads cross PCIe / interconnect. Staging buffers are the default path. | NVIDIA desktop/laptop dGPU, AMD Radeon dGPU, Intel Arc, Intel Mac |
| **N/A (software)** | No GPU heap; lavapipe/SwiftShader present all memory as host memory. | CI software rasterizers |

**Why this matters structurally:** any upload/download path that assumes a discrete staging buffer **silently does nothing** on a UMA device — the buffer is already accessible to the GPU with the same bandwidth. An optimized UMA path that skips staging must be explicitly coded and explicitly tested. The Intel Iris Xe on Justin's desk is **the only local device that exercises the UMA path** — the RTX 4060 does not. Since all Android targets are UMA, an EP that has only been tested on the RTX 4060 has never exercised the memory model used by its primary mobile targets.

**Proposed `DeviceCapabilities` field:** `uma_memory: bool` — `true` when any reported memory type has both `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT`.

### 5.2 Intel Iris Xe as spec-conformance oracle

Intel's Windows Vulkan driver is widely regarded as the most strictly specification-conformant of the major desktop implementations. This has a specific asymmetry that must be written down explicitly:

> **If a compute shader or barrier sequence is correct on the RTX 4060 but fails or produces wrong results on the Iris Xe, the overwhelmingly likely explanation is that our code is relying on behavior the Vulkan specification does not guarantee — not that Intel has a bug.**

Consequences:
- **Do not special-case Intel.** If a test fails only on Intel, treat it as a spec-compliance bug in the EP, not a driver quirk. Special-casing Intel masks real portability problems that will surface on other strict implementations (MoltenVK, Mali strict conformance builds).
- **Intel failures are the best proxy we have for MoltenVK failures** before an Apple device is in CI. MoltenVK (Metal backend) is similarly strict about undefined Vulkan behavior.
- **Test on both devices during every local development loop.** Two gate-passing devices with different architectures (UMA vs discrete, Intel strict vs NVIDIA permissive) exercise more of the spec than running on a single GPU.

*Source for Intel strictness characterization: Justin Chu (project owner), 2026-07-29T09:19:35-07:00: "我听说 intel 的 vulkan 实现是最严格的，你可以在我们设备两个 device 都试试" — "I heard that Intel's Vulkan implementation is the strictest; try both devices on our machine."*

> **OQ-12 scope is unchanged.** The two local-dev GPU entries above do not touch the Adreno 5xx / Mali Bifrost usability question. All **untested (OQ-12 pending)** rows remain unresolved. See §10.

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

| # | Symptom | Version | Source | Notes |
|---|---|---|---|---|
| LVP1 | `subgroupSize` reports 1 (subgroups not emulated in software) | lavapipe < Mesa 22.0 | documentation | CI must pin Mesa ≥ 22.0. Ubuntu 22.04 minimum. |
| LVP2 | ~~`supportedStages = 0` — subgroup ops absent~~ **RETRACTED — instrument failure** | ~~lavapipe (all versions)~~ | ~~"observed in CI, 2026-07-29"~~ — **not valid** | **Corrected reading (CI run with fixed probe, 2026-07-29T20:26:56-07:00):** Mesa 23.2.1 lavapipe reports `subgroup_probe_valid=true`, `subgroup_basic_in_compute=true`, `subgroup_stages_raw=FRAGMENT\|COMPUTE\|TASK_EXT\|MESH_EXT`, ops=`BASIC\|VOTE\|ARITHMETIC\|BALLOT\|SHUFFLE\|SHUFFLE_RELATIVE\|QUAD`, `subgroupSize=8`. The original `supportedStages=0` reading was caused by a `push_next`/`ash` `#[must_use]` bug in Switch's `caps.rs` that silently discarded the entire `VkPhysicalDeviceProperties2` chain; every chained struct read as zero. **Mesa lavapipe does support subgroup BASIC and many other operations in compute.** The device gate removal of subgroup BASIC (DESIGN.md §7.2) stands on §7.0 grounds — capability shortfalls degrade op coverage, not device availability — not on this false observation. **A number taken with a broken instrument is not evidence merely because it was written down.** |

**Instrument-failure scope note:** The push_next bug affected ALL `VkPhysicalDeviceProperties2` chain reads taken before the probe fix. Every other quirk entry in this table (A1/A2/A3, M1/M2, MVK1–MVK4, LVP1, ANV1) is sourced from external documentation — none came from our pre-fix probe — and is unaffected. The only entry sourced from our own probe before the fix was LVP2. The gate PASS results and base-struct device limits (maxComputeWorkGroupInvocations, maxComputeSharedMemorySize, memory types) for local-dev hardware are from base `VkPhysicalDeviceProperties`, not a chained struct, and are not affected. The fp16/int8 feature readings for all devices may have been affected (VkPhysicalDeviceFeatures2 chains use the same pattern); those are marked provisional in §5.

*Sources: LVP1 — Mesa documentation; LVP2 — **RETRACTED** (original "CI-observed" reading was an instrument failure; corrected reading from fixed probe, CI run 2026-07-29T20:26:56-07:00)*

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

> **Status as of 2026-07-29T20:26:56-07:00: BOTH LANES WORKING. LVP2 RETRACTED.** The root causes documented in the previous revision are resolved. Both CI lanes now enumerate lavapipe and can create a Vulkan instance. The lavapipe `supportedStages = 0` reading (LVP2, §6.3) that appeared to cause the initial gate failure was an **instrument failure** — the probe's `push_next`/`ash` bug silently zeroed the Properties2 chain. Mesa 23.2.1 lavapipe actually supports subgroup BASIC and broader ops in compute; see §6.3. The device gate removal of subgroup BASIC (DESIGN.md §7.2) remains correct, justified on §7.0 grounds, not on any lavapipe limitation.

#### 7.4.1 Confirmed working state (as of 2026-07-29T09:19:35-07:00)

##### Linux / Ubuntu 22.04 — lavapipe (CI-verified working)

**Status: CI-verified working.** Lavapipe enumerates successfully. Observed device data:
- `deviceName = llvmpipe (LLVM 15.0.7, 256 bits)`
- `apiVersion = 1.3.255`
- `driverID = DRIVER_ID_MESA_LLVMPIPE`

The earlier Linux lane failure was a Rust compile error in `tests/mock_ort/mod.rs` (`ort::wchar_t` not found on Linux — fixed by Tank: `OrtChar` conditionalized to `c_char` on Linux). Once resolved, lavapipe works correctly under LunarG loader 1.3.296 + Mesa 23.2.1. `glslc` is available from LunarG's `shaderc` apt package (not from Ubuntu's repos).

**Subgroup capabilities (fixed probe, CI run 2026-07-29T20:26:56-07:00):** `subgroup_probe_valid=true`; `subgroup_size=8`; `subgroup_stages_raw=FRAGMENT|COMPUTE|TASK_EXT|MESH_EXT`; `subgroup_basic_in_compute=true`; ops=`BASIC|VOTE|ARITHMETIC|BALLOT|SHUFFLE|SHUFFLE_RELATIVE|QUAD`. The earlier reading of `supportedStages=0` (LVP2) was an instrument failure — the push_next bug zeroed the Properties2 chain. Mesa 23.2.1 lavapipe supports subgroup operations including arithmetic. CI now exercises the subgroup arithmetic path, in software.

**UMA confirmed:** probe reports `is_uma=true`. The UMA memory path is exercised in this CI lane.

**fp16/int8:** probe reading taken before the push_next fix is provisional. Re-observation recommended once fp16 capability probing is confirmed fixed.

##### Windows / Windows Server 2025 — lavapipe (mesa-dist-win 26.1.3, CI-verified working)

**Status: CI-verified working.** After Trinity registered the Mesa lavapipe ICD in the Windows registry, the loader enumerates it correctly:
- `driverUUID` decoded = `llvmpipe` (confirmed from CI log)
- `apiVersion = 1.3.x` (lavapipe MSVC build, mesa-dist-win 26.1.3)

**Root cause of prior failure (archived):** LunarG loader 1.3+ silently ignores `VK_ICD_FILENAMES` / `VK_DRIVER_FILES` / `VK_ADD_DRIVER_FILES` when the process runs with elevated privileges. GitHub Actions Windows runners are elevated (`runneradmin`, Administrators group, UAC disabled). **The fix is registry registration — env var override cannot work on elevated processes.** This is a permanent constraint for any new Windows ICD setup on GitHub Actions.
*Source: [KhronosGroup/Vulkan-Loader LoaderDriverInterface.md v1.3.274](https://chromium.googlesource.com/external/github.com/KhronosGroup/Vulkan-Loader/+/refs/tags/v1.3.274/docs/LoaderDriverInterface.md); [actions/runner-images discussions #6557](https://github.com/actions/runner-images/discussions/6557)*

**Permanent fix (applied by Trinity):**
```powershell
$icdPath = (Resolve-Path "$env:GITHUB_WORKSPACE\mesa3d\x64\lvp_icd.x86_64.json").Path
New-Item -Path "HKLM:\SOFTWARE\Khronos\Vulkan\Drivers" -Force | Out-Null
New-ItemProperty -Path "HKLM:\SOFTWARE\Khronos\Vulkan\Drivers" -Name $icdPath -Value 0 -PropertyType DWord -Force | Out-Null
```

**Subgroup capabilities (Windows lane):** Consistent with Linux lavapipe — same Mesa lavapipe driver (mesa-dist-win 26.1.3, same LLVM backend). Subgroup ops are supported; LVP2 was an instrument failure in both lanes. See Linux entry above for observed values.

**UMA confirmed:** is_uma=true (same as Linux lane). UMA path exercised in CI.

---

#### 7.4.2 Current CI lane state

| Lane | Primary ICD | Vulkan version | Validation layers | Parity lane | Claimed-node execution | Subgroup |
|---|---|---|---|---|---|---|
| Linux / Ubuntu 22.04 | lavapipe (Mesa 23.2.1 / `libvulkan_lvp.so`) | **1.3.255** (observed) | `VK_LAYER_KHRONOS_validation` (LunarG 1.3.296) | ✓ required (§9) | loader + probe only (CI lane does not yet run `test_elementwise`) | **subgroup_basic_in_compute=true**; ops=BASIC\|VOTE\|ARITHMETIC\|BALLOT\|SHUFFLE\|SHUFFLE_RELATIVE\|QUAD; size=8 (probe-verified, 2026-07-29T20:26:56-07:00) |
| Windows / Server 2025 | lavapipe (mesa-dist-win 26.1.3 MSVC, **registry-registered**) | 1.3 (observed) | `VK_LAYER_KHRONOS_validation` (LunarG SDK 1.3.296) | ✓ required (§9) | loader + probe only | Consistent with Linux (same Mesa/LLVM backend); LVP2 retracted |
| **WSL Ubuntu 24.04 (local-dev)** | lavapipe (Mesa 25.2.8 / LLVM 20.1.2, `lvp_icd.json`) | **1.4.318** (observed 2026-07-30) | ✗ (validation layers not loaded in this run) | ✓ confirmed — 58 passed / 28 skipped (§7.7.4) | **✅ CONFIRMED 2026-07-30** — M0 canonical `Add` (fp32) and 195 further tests; 196 total; zero lavapipe-specific failures | size=8; ops=BASIC\|VOTE\|ARITH\|BALLOT\|SHUFFLE\|SHUFFLE_REL\|CLUSTERED\|QUAD\|ROTATE_KHR\|ROTATE_CLUSTERED_KHR |

> **The WSL lane is not a CI lane.** It provides first-claimed-node evidence and a three-way capability diff. CI must be updated to run `test_elementwise` and `test_barrier_parity` on both CI lanes (Linux and Windows) to make this evidence continuous rather than one-time. The §7.5 CI lane spec reflects what the CI lanes need to become.

> **Update 2026-07-31: the "Claimed-node execution" column above is superseded by §7.4.4.** Both CI lanes run the whole of `tests/ops` and now additionally carry criterion 10's gate artifact plus a negative control. The column's readings were also, before today, unable to distinguish "executed claimed nodes" from "fell back to CPU inside `run()` and passed anyway" — which is the state the whole of §7.4.4 exists to end. Read that section for the current per-lane classification; treat this table as capability data, not as evidence of execution.

Both CI lanes expose `VK_KHR_synchronization2` and support subgroup arithmetic in compute (probe-verified on Linux lane). The §9 forced-legacy run (`ep.force_legacy_barriers=1`) is the **only** way the `vkCmdPipelineBarrier` code path is exercised before physical Android hardware is available; the subgroup arithmetic path is now exercised in both normal CI runs.

**What GPU-less CI does NOT cover:**
- Subgroup arithmetic **hardware** semantics (CI exercises the subgroup arithmetic code path via lavapipe software emulation, which is a valid correctness test but not a performance or hardware-conformance test)
- Vendor-specific shader compilation paths
- Driver-specific quirk workarounds (Adreno A1/A2, Mali M1/M2 — see §6.3)
- Real fp16 throughput and memory bandwidth

**Single-run blindness (Tank, 2026-07-30):** ORT's memory-pattern planner does not engage on the first `run()` call. It records the allocation pattern on run 1 and sub-divides arena from run 2 onward. Measured on both Windows devices: 1 run → 0 interior pointers; 2 runs → 13 interior pointers; 3 runs → 26; 5 runs → 52. **Every test helper in `tests/ops/` creates one ORT session and calls `run()` exactly once.** The 196 tests that passed on lavapipe inherit this blindness: any bug that manifests only when ORT hands back `base + offset` interior pointers is invisible in the single-run suite. This is the same class of bug that concealed the all-zero logits on run 1 (`probe_run2.py`, Tank 2026-07-30). Stating here so nobody reads the 196-pass count as covering multi-run behaviour — it does not.

For anything in the matrix column labeled **untested**, the project must either acquire CI access to that hardware or document the platform as "community-supported" with no CI guarantee.

#### 7.4.3 Where Tank's diagnostics fit

- Run `epctl --dump-capabilities` in both Windows and Linux lanes immediately after the smoke-check step. This reports device state without ORT and makes the next instance-creation failure self-diagnosing.
- Switch's EP diagnostic (`ONNXRUNTIME_VULKAN_EP_VALIDATE=1`) already logs what the loader sees before instance creation — ensure this log appears in the CI step output, not only in the test harness stderr.

#### 7.4.4 Lane classification: `operational` vs `green` — every lane, by the new definition (revised 2026-08-01T09:55-07:00)

**This is the section to read if you read only one.** As of this revision the CI lanes carry criterion 10's gate. Before it, a green lane proved that a process exited zero and **nothing else**.

##### What was wrong, stated without softening

I grepped `.github/workflows/ci.yml` for `check-verdict`, `gate_chain`, `model_output_equivalence`, `MATCH`, `UNMEASURED`, `UNATTRIBUTED`, `guard_d` and `assert_vulkan_executed`. **Zero matches.** Both lanes ran the suite and read an exit code.

That is not a theoretical hole. On 2026-07-30 the EP claimed 353 nodes, failed at run time on a zero-size allocation, and ORT printed `EP_FAIL … Falling back to CPUExecutionProvider`, re-ran the entire graph on CPU, and **raised nothing**. `get_providers()` still listed `VulkanExecutionProvider`, because the provider list is fixed at session-create time and the fallback happens inside `run()`. **The whole suite was green while zero nodes executed on Vulkan.** That is the fifth appearance of that log line on this project with every gate passing.

Morpheus's ruling is the one I am implementing: a lane that does not carry criterion 10's gate would *"measure the same silence in a new location"*, and **`operational` ≠ `green`** — a lane's pass condition must include the verdict field, so `UNMEASURED` reports `UNMEASURED` and not PASS.

##### The definitions, restated with the fourth state

| State | Definition | What may be claimed | Gate required? |
|---|---|---|---|
| **`operational`** | The lane exists, builds, runs, and reports. It may even execute claimed nodes. | That the lane is up, and that it is a **prerequisite** for running criterion 10 anywhere but a development desk. | No |
| **`green`** | The lane's pass condition includes a `model_output_equivalence` verdict of `MATCH` **carrying an `executed_by` frame parsed from ORT profiling**, *and* the lane has demonstrated on this same run that its gate can fail. | That criterion 10's tail is satisfied **for the gate artifact only** — never for another artifact, never for fp16, never for performance. | Yes |

**A lane that cannot fail when the EP does nothing is not evidence.** That sentence is the whole classification.

##### Every lane, and what it is

| Lane | Where | Runs claimed nodes? | Carries criterion 10's gate? | Has a falsifier for its own gate? | **Classification** | Why it is not more than that |
|---|---|---|---|---|---|---|
| `format` (rustfmt) | ubuntu-latest | No — no build, no Vulkan | Not applicable | Not applicable | **`operational`, and correctly so** | It is a formatting check. It makes no claim about execution and none is expected of it. `UNOBSERVABLE`, not zero (R12). |
| `lane-checks` | ubuntu-latest | No — no Vulkan by design | It **is** the gate's falsifier | Yes — 21 tests, two polarities each, synthesised inputs | **`green` for the claim it makes**, which is *"the lane checks work"* and nothing about the EP | Deliberately GPU-less: an instrument test that needed the subject healthy could say nothing on the day the subject is sick. |
| `build-test-linux` (Ubuntu 22.04, lavapipe) | GitHub-hosted | Yes | **Yes** — vocabulary preflight → `gate_chain_fp32` → `ci/check_verdict.py` → `epctl --check-counters` → fatal-log grep | **Yes, two of them** — ICD-removal, *and* the loader-independent decline probe | **`operational` today; `green` on the first run in which all steps pass** | Not yet observed on a runner. I will not classify a lane `green` from reading its YAML — that is precisely R10, and I would be doing to my own work what R10 forbids. |
| `build-test-windows` (Server 2025, lavapipe via mesa-dist-win) | GitHub-hosted | Yes | **Yes** — same steps | **Yes — and only one of the two actually fires here, and until 2026-08-01 *neither* did.** The ICD-removal control cannot be relied on: the LunarG loader silently ignores `VK_DRIVER_FILES`/`VK_ICD_FILENAMES` in elevated processes (§7.4.1) and GitHub's Windows runners are elevated. Worse, the guard that was supposed to *notice* that was a constant — it matched a phrase the loader probe prints on every run (§7.11.3) — so the whole step short-circuited on every run in both directions. Now `ci/check_icd_suppression.py` parses the device count and reports `ERROR(instrument=icd_suppression_ineffective)` with a record. The decline probe is the falsifier this lane relies on. | **`operational` today; `green` on the first passing run** | Same reason — plus, until today, this lane's only negative control was one that provably never fired. |
| `conformance` (onnx-tests, `workflow_dispatch`) | ubuntu-22.04 | Yes | Gate steps + preflight; the conformance step itself remains `continue-on-error` | **Yes, as of today** — the decline-probe control was added; it had none of its own before | **`operational` — and its conformance table is *not evidence*** | The census step is a diagnostic by design. It carries no verdict per row, so under §10.0 it says nothing about this EP. Labelled as such in the step summary. |
| WSL Ubuntu 24.04 / lavapipe (local dev, §7.7) | My desk | Yes — 196 tests, `subgroup_size = 8`, barrier parity 58/0 as a third independent implementation | No | No | **`operational`. Not `green`.** | This is the one I most want to promote and will not. It was a real result and it is not evidence for criterion 10: it is not a CI lane, it ran once, it had no verdict, and it had no falsifier. Its value is the capability diff (§7.5) and the third barrier-parity implementation, which are claims it *can* support. |
| Local Windows / RTX 4060 (selector 0) | My desk | Yes | Gate runs here (that is where it was developed) | Yes — **three** polarities verified on this hardware today: PASS, `FAIL` with no ICD, `FAIL` on a declined artifact with the loader healthy | **`operational`** | Not a lane. A development desk is not a lane no matter what it proves, because nothing re-runs it. |
| Local Windows / Intel Iris Xe (selector 1) | My desk | Yes | Same | Same — PASS and decline-probe `FAIL` both verified today | **`operational`** | Same. Also: **selector 1 is `SPLIT-DEVICE`** — the env var indexes the best-first sorted list while the offer is keyed by raw enumeration index (§6.5), so I make **no claim** that selector 1 selected a different physical device. |
| Android / Adreno / Mali | Nowhere | — | — | — | **`untested`** | No hardware. OQ-12 unchanged. lavapipe is not Adreno or Mali and never becomes it. |
| macOS / MoltenVK | Nowhere | — | — | — | **`untested`** | No runner. |

##### What the EP now demonstrably does — and what that does *not* upgrade

Recorded here because it is the strongest execution evidence this project has, and because
its limits are what decide the classifications above rather than its size:

- **Three consecutive runs, both local devices, from ORT profiling: 3 `VulkanExecutionProvider` node events (one fused node covering ~355 graph nodes) + 24 `CPUExecutionProvider`.** All 65 outputs bit-identical across runs; argmax 30751, matching CPU. Mouse's census: **355 claimed / 1 island / 8 permanent declines with recorded reasons.**
- **What that establishes:** the EP executes, at scale, on real hardware, and its outputs are attributable and reproducible. Every `UNATTRIBUTED` finding on this project is now a finding about a *specific* run, not about whether the EP works at all.
- **What it does not establish, and no amount of it will:** that any *lane* is `green`. Those runs happened on my desk. A lane is `green` when **the lane** carries a verdict and **the lane** has shown it can fail, and neither of those is a property of how well the EP runs anywhere else. Execution evidence and lane evidence are different claims with different falsifiers; folding the first into the second is exactly the move §7.4.4 exists to refuse.
- Nor does it license a timing figure. **Re-qualified 2026-08-01T13:19-07:00 and this replaces what this bullet said this morning.** Niobe's NVIDIA figure is now **"≤ 40.201 ms/inference GPU busy, device state unrecorded"** — not withdrawn, re-qualified, and those are different outcomes. Morpheus rejected the regime-separation rescue: there are **not two clock regimes**, the board moved 210 → 2490 MHz *within one run*, a governor is continuous, and *"the two states I sampled don't overlap"* had been promoted into a claim about the device. The real margin is **6.1×, not 21×**. RSD 0.033% keeps its descriptive role and loses its certifying one — a run held at a low clock is *more* steady, not less (§10.0.1 R9 amendment 5). Intel remains withheld as `NO_STEADY_TAIL`. Attribution was necessary and is no longer sufficient: §10.0 obligation 8 additionally requires a device-state record over the statistic's own suffix.

##### What still is not `green`, plainly

1. **No CI lane is `green` yet.** All three now *carry* the gate and its falsifier; none has been observed to pass it on a runner. The correct word for a wired-but-unobserved lane is `operational`.
2. **The WSL lavapipe result is not `green` and I am not going to launder it into one.** 196 passing tests with no verdict is 196 assertions about outputs whose executor was never established.
3. **`green` is per artifact.** When these lanes go green they are green for `gate_chain_fp32` — a 2-node fp32 `Add → Relu` on 256 elements. Not for Phi-3.5, not for `MatMulNBits`, not for fp16 (no `storageBuffer16BitAccess` confirmation on lavapipe; those claims remain `UNMEASURED`).
4. **The Phi-3.5 execution evidence does not make any lane `green`.** 355 claimed nodes in one island, three reproducible runs and 65 bit-identical outputs are a claim about the EP, on my desk. They are not a claim about a lane, and no quantity of them becomes one.
5. **Single-run blindness is not fixed by the gate** (§7.4.2). The gate artifact also runs once.
6. **No timing figure is quotable from a run whose verdict is not an attributed `MATCH` — and as of today that is necessary but no longer sufficient.** Every earlier wall-clock ratio on this project, including 3.1× and 3.7×, is **withdrawn**. §10.0 obligation 8 adds a second interlock: a device-clock figure also needs a **device-state record covering the statistic's own suffix**, carrying a tenancy verdict and clock min/median/max against the board maximum. Without it the figure is `STEADY_UNCERTIFIED`. Niobe's NVIDIA number is re-qualified to **≤ 40.201 ms, device state unrecorded**; Intel is still withheld as `NO_STEADY_TAIL`. **No CI lane can supply that record**, for the structural reason in §7.11.
7. **The Windows lane's ICD-removal control has never been shown to fire — and as of today I know why, and it is worse than "elevated runner".** The control's guard tested `-match 'passed the §7.2 capability gate'`, and `engine::loader_probe_report()` emits `"{n} device(s) passed the §7.2 capability gate."` on **every** run, with `n = 0` when the suppression *did* take. The match therefore succeeded unconditionally and the step short-circuited to `exit 0` on **every run in both directions**. A detector that fires on every input is not a detector, it is a constant. Replaced by `ci/check_icd_suppression.py`, which parses the count, has two-polarity tests, and writes a record whose content varies with its input (§7.11.3). The decline probe, which does not depend on the loader, remains what that lane's falsifier claim actually rests on.
8. **No lane may publish a duration.** Not a convention — a step (§7.11.2). The first lane step that writes a millisecond into an artifact or into the job summary turns `check_device_state` red on the same run.

##### The measurement, both polarities, both devices — 2026-08-01, on merged `main` (5eda83b)

Positive pole. A verdict record with a real `executed_by`, not a green from a skipped check:

```
RTX 4060 (selector 0)                     Intel Iris Xe (selector 1)
GATE: PASS                                GATE: PASS
  verdict = MATCH                           verdict = MATCH
  executed_by = {'VulkanExecutionProvider': 1}   (same)
  attribution_source = ort_profile          (same)
  counters_dispatches_executed = 2          (same)
  profile_node_events = 1                   (same)
  max_abs_diff = 0                          (same)
  artifact = gate_chain_fp32@ci-gate-v1 sha256:aba0cd3847ec28ac  (same digest — same artifact)
ci/check_verdict.py       → VERDICT-CHECK: PASS   (exit 0, separate process, separate parser)
epctl --check-counters    → PASS, 2 dispatch(es) executed (required 1)   (exit 0)
```

Negative pole 1 — **the EP cannot start** (`VK_DRIVER_FILES` → a nonexistent ICD, everything else identical):

```
GATE: FAIL(condition=UNATTRIBUTED)                                        exit 1
  executed_by = {'CPUExecutionProvider': 2}   own count 0
  permits_triple_and_ratio = false
  "the comparison ran and this EP did not run … the model was not wrong, the subject was"
```

Negative pole 2 — **the EP starts and executes nothing**, loader untouched, driver healthy, device passing the §7.2 gate (`--artifact decline_probe`: one `Det` node, an op this EP does not implement):

```
RTX 4060 (selector 0)                     Intel Iris Xe (selector 1)
GATE: FAIL(condition=UNATTRIBUTED)        GATE: FAIL(condition=UNATTRIBUTED)      exit 1
  executed_by = {'CPUExecutionProvider': 1}    (same)
  own_provider_execution_count = 0             (same)
  profile_node_events = 0 of 1 total           (same)
  permits_triple_and_ratio = false             (same)
ci/check_verdict.py → VERDICT-CHECK: FAIL(condition=UNATTRIBUTED)   exit 1
```

The second negative is the one that matters. The first proves the gate notices a missing
driver; the failure that was actually live on 2026-07-30 was a **healthy** EP that claimed
and executed nothing, and until today no control on this project reproduced that state. It
is also the only negative control available on an elevated Windows runner, where the ICD
cannot be removed by environment variable at all.

Neither negative reported `DIVERGENT`, and both were checked for it. The comparison agreed
— of course it did, both sides were CPU. **`UNATTRIBUTED` is not `DIVERGENT`**: the model
was not wrong, the subject was absent. Different owners, different fixes, different next
questions.

##### `ERROR(instrument)` must not become the lane's normal state — and how a maintainer tells

The gate imports its entire vocabulary from `tests/ops/_verdict.py` and defines no token of
its own, so when that module cannot be imported **every verdict-carrying step reports an
instrument outage at once**:

```
GATE: ERROR(instrument=verdict_vocabulary_unavailable)
```

That is the honest report and it is a hazard, because two different situations produce the
identical line: **(a)** this checkout legitimately does not contain the module — an older
branch, a bisect, a PR that predates it, a sparse checkout — and **(b)** the lane is broken:
the file is right there and the job cannot load it. Reported the same way, an instrument
outage becomes the weather, and a signal that is always on is not a signal.

`ci/check_vocabulary.py` runs **before** the gate in every lane and answers that question
from the repository's own state, giving each case its own token:

| Exit | Token | What it means | Who owns it |
|---|---|---|---|
| 0 | `VOCAB: PASS` | present, importable; prints path, **sha256**, byte count, git-tracked status, commit, Python version, and the verdict tokens the module defines | — and this is the load-bearing part: **any later `verdict_vocabulary_unavailable` in the same job is now a lane fault by elimination**, because the module imported in this interpreter, in this checkout, moments earlier |
| 4 | `VOCAB: ERROR(instrument=verdict_vocabulary_absent_from_checkout)` | the file is not in the tree | **repository state, not a lane defect.** The lane is still red — a lane that cannot emit a verdict cannot be green — but red for a reason no CI change will fix, and the message says which file has to arrive |
| 4 | `VOCAB: ERROR(instrument=verdict_vocabulary_broken)` | the file is in the tree and does not import | **lane or source defect**, with the exception text quoted in full (R13: quote the text, never the count) |

Same exit code for the two outages, because both are outages and neither is a detection.
**Deliberately not the same token**, because the token is what a maintainer greps.

The distinguishing rule, stated as a procedure rather than a feeling:

- **Every lane on this commit says `absent_from_checkout` →** the commit does not carry the vocabulary. Repository state. No CI change fixes it; the fix is the file landing.
- **One lane says `PASS` and another says `unavailable` →** that second lane is broken, and the difference between the two jobs is the fault. This comparison is possible only because the preflight prints the module's sha256 and git-tracked status on every path.
- **Every lane says `broken` with the same sha256 →** the module itself, not the lanes.

And it is surfaced rather than buried: with `--github-summary` the outcome is written to
`$GITHUB_STEP_SUMMARY` and emitted as an annotation whose **title differs per token**, so
the distinction is visible on the run's summary page without opening a log. A caveat that
lives in a different artifact from the thing it qualifies is not attached to it.

Verified locally, all three states, exit codes included: `PASS` (exit 0),
`absent_from_checkout` (exit 4), `broken` (exit 4, `SyntaxError` quoted). Four of the 21
`ci/test_lane_checks.py` tests assert these, one of them asserting only that the two outage
tokens are **not equal** — stated as its own test because it is the property, not a side
effect.

##### Verified state of `main` at the time of this revision

`32 failed / 272 passed` on Intel with `test_wiring_census` excluded (it timed out under
three-agent contention). **No wall-clock threshold appears anywhere in the gate, and none
will be added**: the same suite took 708 s under load and 161 s quiet — 4.4× — and I read
the resulting timeouts as a "68 failed" regression that did not exist. That is R13's second
clause with a stopwatch attached: a count without its text, from an instrument measuring
the machine's other tenants.

**The exclusion is lifted as of 2026-08-01 and the rule above is now enforced rather than
promised.** The census carried the last wall-clock timeout in the suite; it has been
removed, not raised. Raising it was the one repair R9 amendment 5 forbids — a threshold
fires the same way for "the box is loaded" and "the census hung", so it moves with the
reader's confidence and cannot be repaired by moving it, in either direction. Widening it
enough to survive the 4.4× above (9.5× on the `record` step) makes it wider than the hang
it exists to catch.

What replaces it is a **stall budget denominated in work the machine actually did**
(`tests/ops/_watchdog.py`). A background thread completes a fixed reference computation
over and over; each completion is one *work unit*. A step's budget is a number of units it
may spend producing no output and reaching no result. Contention lowers units-per-second,
so the same budget is automatically a wider window in wall time on a loaded box; a hang
leaves the machine producing units while the step produces nothing, so the budget is
exhausted in bounded work whether the box is busy or idle. The unit is CPU-bound, so it
tracks CPU contention exactly and GPU/disk contention only by correlation — every run
records `stall_detector.observed_units` per step in `bench/results/wiring_census-dev{N}.json`
so that margin is auditable rather than asserted.

Both arms are demonstrated, because a detector shown only to pass has been shown nothing:
`tests/ops/probe_stall_guard.py` runs the real census in all four cells of
{healthy, hung} × {quiet, loaded} and writes `bench/results/stall_guard_arms-dev{N}.json`
with `arms_must_differ`. The load-bearing cell is **hung + loaded**: a timeout loose enough
to survive this host's inflation passes there, and the work-unit budget fails there.
`tests/ops/test_stall_guard.py` carries the same property as 13 always-on deterministic
tests against a fake clock, including the crisp form — the *same* wall-clock silence trips
the guard on a fast clock and does not trip it on a slow one.

Terminal states follow R13 and depend on *whose* stall it is, never on how long it took: a
stalled mechanism under census is `FAIL(condition=NO_PROGRESS)` (a mechanism that never
returns has produced no observation of itself, which is criterion 12's subject), a stalled
toolchain step such as a `cargo` compile is `ERROR(instrument)`. `STALLED` is a distinct
census token from `UNWIRED` — "ran and produced nothing" and "never came back" are
different facts.

##### The second attribution witness — recorded, not required (2026-08-01, round 26)

A re-run of `bench/results/criterion10-dev1.json` recorded
`"counters_dispatches_executed": null` inside a passing `AGREE` / `MATCH`. Reading the
code back, the witness **did not participate in the verdict at all**:
`read_counters_dispatches()` collapsed five distinct causes into one bare `None`,
`split_frame` returned `False` for `None`, and the verdict consulted nothing else. So
`split_frame == False` meant both *"two witnesses were read and they agree"* and *"there
was only one witness"*, and `MATCH` emerged either way. Under R10 that witness could have
been absent on every run ever taken and no verdict would have moved — which is
indistinguishable from never having wired it.

The repair is **not** to make `MATCH` require the counters. The counters live inside the
frame whose existence is in question; making the correctness verdict depend on them would
move the check with the reader's confidence rather than with its subject (R9 A5). What was
silently skipped is the **split-frame check**, so that is what gains a state:

- `witness_agreement` ∈ {`AGREE`, `DISAGREE`, `UNOBSERVABLE`} replaces the boolean.
  `split_frame` survives as `witness_agreement == DISAGREE` so `ci/gate_chain_fp32.py` and
  `epctl` are unaffected.
- `counters_dispatches_executed` is an int or the string `"UNOBSERVABLE"` — **never
  `null`** (R12) — beside a `counters_witness_reason` naming which of the five absences it
  was.
- Every verdict record now says which witnesses actually spoke:
  `attribution_witnesses_present` is `["ort_profile"]` or
  `["ort_profile", "ep_counters"]`, and `explain()` on the one-witness form says the MATCH
  *"rests on ONE instrument … must not be quoted as one"*.
- `assert_closes_criterion_10(require_second_witness=True)` — default on — raises
  `InstrumentError` when the check was never performable. The canonical M0 record may not
  come from a run in which the split-frame check could not run. The requirement is
  evaluated **last**, after run count, verdict and uniformity, so that a genuine CPU
  fallback is never masked by an instrument complaint; there is a falsifier pinning that
  ordering.

A one-witness run still reads `MATCH` — the comparison agreed and the primary witness
attributed it — but it is now a *labelled* one-witness MATCH. The downgrade is in the
weight of the witness list, not in the verdict.

Both arms, live on Intel Iris Xe, same binary and same test, differing only in whether
`ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE` was set before the process started:

| arm | lane | `counters_dispatches_executed` | `witness_agreement` | `witnesses_present` |
|---|---|---|---|---|
| armed | `PASS` | `1066` | `AGREE` | `ort_profile`, `ep_counters` |
| unarmed | `ERROR(instrument)` | `"UNOBSERVABLE"` | `UNOBSERVABLE` | `ort_profile` |

`bench/results/criterion10_witness_arms-dev1-{armed,unarmed}.json`. The unarmed arm
reproduces the artifact under investigation exactly, which is also the answer to why it
went null: that run was invoked without the counters variable. On Windows the DLL caches
the variable at load time, so it cannot be armed from inside a test that has already
imported `onnxruntime` — the arming is a property of how the process was started, and
nothing in the old record said so.

##### The union defect — a required parameter, and a caller that was not on my branch (2026-08-01, round 27)

`squad/trinity` added a required keyword-only `guard` to `_run_counters_child()`.
`squad/mouse` added `test_ledger_lookup_wired`, which calls that helper, to the same file.
Both branches were green. Their union was not:

```
TypeError: _run_counters_child() missing 1 required keyword-only argument: 'guard'
tests/ops/test_wiring_census.py::test_ledger_lookup_wired  — both devices
```

This is the third instance of the shape in one day (Tank's residency screen moving a
process-global counter; Niobe's `sys.path.insert` in `bench/` rebinding imports for Link's
`ci/` checks; this). In none of them did an author do anything wrong locally, and **no
command either author could have run on their own branch would have shown it**. Our test
discipline verifies branches; these defects live in unions.

**The parameter now defaults to `None`, and `None` means *build one*, not *run unguarded*.**
That distinction is the whole decision. `guard=None → unguarded` would satisfy "callable
without a guard" while silently undoing round 25: every future caller would opt out of
stall detection with nothing to notice. An ambient guard loses only **bookkeeping** — its
costs and max-silences do not land in the census-wide guard's ledger, so they are absent
from `observed_units` in the census artifact — never protection. There must be no unguarded
path through the module. Both halves carry falsifiers, because either alone is satisfiable
by the wrong fix: *callable without a guard* is satisfied by deleting the guard, and *the
call is guarded* is satisfied by keeping the argument required. Only the pair pins it
(`tests/ops/test_stall_guard.py`, 16 always-on deterministic tests).

The same shape sat one line down: `_BUDGET_UNITS[label]` raised `KeyError` for any tag
added elsewhere. It is now `_budget_for(label)` with a generous default. A loose default is
safe here in a way it would **not** be for a wall clock, because the unit is *work*: a loose
work budget detects a hang later in work, never not at all, and contention cannot stretch it.

**The union check — `tests/union_check.py`.** Three tiers, and only one of them is a gate:

| tier | question | verdict class |
|---|---|---|
| 1 | which files were edited on **both** sides (`base...head` ∩ `head...base`) | precondition |
| 2 | was a `sys.path.insert(0, …)` added **in a directory that collides** | precondition |
| 2b | which module basenames are importable from more than one directory | precondition |
| 3 | trial-merge into a scratch worktree and **run** the union | **gate** |

Tiers 1, 2 and 2b are preconditions, never gates (R9 amendment 5): a union can be broken
with no intersecting file and no scanned side effect, so their silence is a statement about
the tool rather than about the union. The script therefore never prints a bare `PASS` for
them — it prints `PRECONDITION(tiers=1,2; tier 3 not run)`. It is named `union_check.py`
and not `check_union.py` because the `check_*` names in `ci/` are gates. Exit codes follow
R13: `0` PASS, `1` FAIL(condition), `4` ERROR(instrument).

Tier 2 was 21 findings on its first run and useless; `sys.path.insert(0, …)` is an ordinary
idiom here. What made Niobe's insert dangerous was not the insert but the **population of
colliding module basenames** — so tier 2b computes that population from the working tree
(an existing collision is as dangerous as a new one and more likely forgotten), and tier 2
names only inserts made in a directory that participates in one. Run against the pre-fix
history it reports `COLLISION device_state.py <- bench, ci` and names `bench/device_state.py`
and `ci/check_lane_inventory.py` — i.e. it retrodicts the Niobe×Link incident, which is the
R10 falsifier for the tool itself: its output varies with its input.

Tier 3 was demonstrated with **both arms**, on real history, same target, same device:

| arm | `--base` / `--head` | verdict | exit |
|---|---|---|---|
| broken union | `2fee5ef` / `c55a389` | `FAIL(condition=union_red)` — quoting the `TypeError` above | 1 |
| repaired union | current `main` / `squad/trinity` | `PASS` | 0 |

A conflict that is confined to regenerated artefacts under `bench/results/` is resolved to
HEAD's copy under `--resolve-artifacts`; a conflict anywhere else stays
`ERROR(instrument=merge_conflict)`, because an instrument that cannot construct its subject
has not observed it.

**Recommended standing obligation:** before reporting a branch done, merge `main` into it
and run the lanes named by tiers 1–2 in the merged tree. `python tests/union_check.py --run`
does exactly that and picks the targets itself. The full suite is the stronger form; the
targeted form is the one people will actually run, and it caught this defect.

Seven steps per lane, five processes, five different failure modes, no `continue-on-error`
on any of them:

0. **`ci/check_vocabulary.py`** runs first and decides *which kind* of instrument outage a
   vocabulary failure would be, so that the six steps below cannot collapse into one
   indistinguishable red. See the preceding subsection.
1. **`ci/gate_chain_fp32.py`** builds the §7.8.1 artifact, writes the verdict record to disk as `UNMEASURED` **before opening any session**, runs it under ORT profiling against a CPU-only run of the same artifact, and constructs the verdict through Trinity's `EquivalenceVerdict.from_comparison()` — which takes a parsed `ExecutionAttribution` as a *required* argument and therefore **cannot emit `MATCH` at a zero own-provider count**. It emits `UNATTRIBUTED` instead, and says which providers did execute.
2. **`ci/check_verdict.py`** re-reads the record **in a separate process**. A missing record is `FAIL(condition=UNMEASURED)`, not a skip. A `MATCH` with an empty `executed_by`, a zero own-count, or an `attribution_source` that is not `ort_profile` is rejected as `UNATTRIBUTED` — a gate that trusts its input is a gate that trusts whatever replaced its input.
3. **`epctl --check-counters <file> --require-dispatches 1`** reads the verdict spliced into the counters snapshot. Tank's exit codes: `DIVERGENT` → 1, `UNMEASURED`/absent → 3.
4. **`ci/check_fatal_log.py`** greps the captured suite output for `Falling back to CPUExecutionProvider`. R13 obligation 3: *a grep cannot `NameError`, and a guard cannot be silenced by a log format change.* The logs are captured with `2>&1` because ORT writes that line from C++ to fd 2 — a tee of stdout alone would scan a log in which its own subject cannot appear, and would agree with everything.
5. **Negative control 1 — the ICD is removed**, the *same* artifact runs through the *same* script, and the step **requires** `FAIL(condition=UNATTRIBUTED)` with a non-zero exit. If the gate ever passes with no ICD present, the step fails the lane with the sentence *"every green in this lane is uninterpretable"* — which it would be. **On Windows this control first checks whether the suppression actually took**, because the LunarG loader silently ignores `VK_DRIVER_FILES`/`VK_ICD_FILENAMES` in elevated processes (§7.4.1) and GitHub's Windows runners are elevated. If the ICD is still there the step reports `ERROR(instrument=icd_suppression_ineffective)` and asserts nothing — rather than blaming the gate for a control that never fired, which would route a finding to the wrong owner.
6. **Negative control 2 — the loader-independent one**, and the one the Windows lane actually rests on. `--artifact decline_probe` is a single `Det` node: an op this EP does not implement and is not going to. The EP loads, the driver is present, the device passes the §7.2 gate, capability detection succeeds — and the EP claims **nothing**, so the graph runs on CPU and the lane must report `FAIL(condition=UNATTRIBUTED)`. It reproduces the failure that was actually live on 2026-07-30 — a *healthy* EP executing nothing — which the ICD-removal control does not. If the EP ever claims `Det`, this control stops being one; the step's error text says so explicitly and tells the maintainer to re-point it at an op that is still declined, rather than letting the control quietly turn into a positive.

The three polarities measured on local hardware are recorded above ("The measurement, both
polarities, both devices — 2026-08-01"), superseding the 2026-07-31 two-polarity record on
the same mechanism. Note what neither negative case reported: `DIVERGENT`. The comparison agreed — of course it did, both sides were CPU. **`UNATTRIBUTED` is not `DIVERGENT`**: the model was not wrong, the subject was absent. They have different owners (`UNATTRIBUTED` routes to whoever owns run-time fallback; `DIVERGENT` routes to the kernel authors), different fixes, and different next questions, and a lane that printed one red for both would have R13's defect.

##### R13 in the lane's terminal states

Every check in `ci/` has three terminal states, three exit codes, and three distinct printed tokens:

| Exit | Token | Meaning |
|---|---|---|
| 0 | `PASS` | the check reached its observation and the observation is good |
| 1 | `FAIL(condition=<name>)` | the check reached its observation and the observation is the thing it exists to detect |
| 4 | `ERROR(instrument=<name>)` | the check **did not reach its observation** |

Exit 4 and not 3, because `epctl` already spends 3 on "the lane did not report" and two meanings on one code is exactly the defect. **An instrument error never counts as a detection**, and a lane with one is not a lane that ran. Every failure path prints the **text** it observed — the `Falling back` line in full, the `executed_by` map, the max-abs-diff — and no failure path reports a count without the text beside it. I read `5 failed` as a working guard on 2026-07-31 when the guard was raising `NameError`; a CI lane that reports only a count is that mistake at scale.

One ordering bug of exactly this shape was found and fixed while writing the gate: the verdict-splice into the counters snapshot fails when the EP executed nothing, **because** the EP never dispatched and so never wrote a snapshot. Reported naively that came out as `ERROR(instrument=counters_snapshot_unwritable)` — a real detection wearing an outage's costume, which is R13 with the polarity reversed and is worse, because it routes a finding to the harness owner. The splice outcome is now recorded and may only be terminal when the verdict is `MATCH`.

##### Vocabulary — Trinity's, and only Trinity's

`ci/gate_chain_fp32.py`, `ci/check_verdict.py` and `ci/check_fatal_log.py` **import `tests/ops/_verdict.py`** and define no verdict tokens, no marker strings and no comparison outcomes of their own. `MATCH`, `DIVERGENT`, `UNMEASURED`, `UNATTRIBUTED`, `SPLIT-FRAME`, `AGREE`/`DISAGREE`/`NOT_PERFORMED`, `FATAL_LOG_MARKERS` — all hers. If that module is absent the lane checks fail collection with `ERROR(instrument=...)` rather than skipping, because a skipped test reports the same green as a passing one.

**Supersession note.** My §7.8.2 design brief called the gate `epctl --check-verdict`. That name does not exist and will not: Morpheus assigned the verdict gate to `epctl --check-counters` (Tank), which already reads `model_output_equivalence` and now fails on `UNATTRIBUTED` and on a missing `executed_by`. One vocabulary, one flag. `ci/check_verdict.py` is a *second reader with a different parser*, not a second gate — it exists so the decision survives the producer and so a lane whose `epctl` predates the fourth state still refuses.

---

---

## 7.5 Three-Way Capability Diff (lavapipe WSL 25.2.8 × Intel Iris Xe × RTX 4060 Laptop)

**Measurement date:** 2026-07-30T07:52-07:00. All values are from `epctl --probe-loader` on the same binary built from `squad/link`. All three readings were taken with the corrected probe (push_next rebind bug corrected, see §6.3). LVP2 retraction applies only to readings taken *before* that fix; these readings are sound.

| Property | lavapipe (Ubuntu 24.04 WSL, Mesa 25.2.8 / LLVM 20.1.2) | Intel Iris Xe (Win 11, driver 31.0.101.5590) | RTX 4060 Laptop (Win 11, driver 572.x) |
|---|---|---|---|
| `deviceName` | llvmpipe (LLVM 20.1.2, 256 bits) | Intel(R) Iris(R) Xe Graphics | NVIDIA GeForce RTX 4060 Laptop GPU |
| `apiVersion` | 1.4.318 | 1.4.309 | 1.4.325 |
| `subgroup_size` | **8** | **32** | **32** |
| `subgroup_ops` | BASIC\|VOTE\|ARITH\|BALLOT\|SHUFFLE\|SHUFFLE_REL\|CLUSTERED\|QUAD\|ROTATE_KHR\|ROTATE_CLUSTERED_KHR | same | same **+ PARTITIONED_NV** |
| `subgroup_stages (compute)` | ✅ | ✅ | ✅ |
| `subgroup_stages (non-compute)` | FRAGMENT\|TASK\|MESH (no VERTEX/TESS/GEOM) | + VERTEX\|TESS\|GEOM | + VERTEX\|TESS\|GEOM\|RAY_*\|TASK\|MESH |
| `is_uma` | true | true | **false** |
| `maxComputeSharedMemorySize` | 32 KiB | 32 KiB | **48 KiB** |
| `maxComputeWorkGroupInvocations` | 1024 | 1024 | 1024 |
| `timestamp_period_ns` | 1.0 | **52.0833** | 1.0 |
| `timestamp_valid_bits` | 64 | **36** | 64 |
| §7.2 gate result | PASS | PASS | PASS |

**Notable differences and portability implications:**

1. **`subgroup_size: 8` on lavapipe vs 32 on both Windows GPUs.** Confirmed. Any shader that baked 32 would produce wrong results on lavapipe and on Android devices with `subgroupSize < 32`. See §7.6 for the shader-variant audit.

2. **`PARTITIONED_NV` on RTX 4060 only.** No EP shader uses this. If any future shader used `subgroupPartitionNV` it would fail on lavapipe and on 100% of non-NVIDIA hardware. Flagged for Switch.

3. **`maxComputeSharedMemorySize: 48 KiB` on RTX 4060 vs 32 KiB on the other two.** Current shaders allocate at most 1 KiB (`shared float red[256]` in `q_gemv.comp`). Switch and Mouse must not exceed 32 KiB in new shaders without a capability guard.

4. **`timestamp_period_ns: 52.0833` on Intel Iris Xe vs 1.0 ns on others.** Already documented in `trace.rs`; no portability issue in current code.

5. **`timestamp_valid_bits: 36` on Intel Iris Xe vs 64 on others.** Already documented. `trace.rs` handles masking.

6. **All three pass the §7.2 gate.** No property in the three-way diff causes one device to pass and another to fail.

---

## 7.6 Subgroup Size: Audit of Affected Shader Variants

**Measured fact:** lavapipe reports `subgroup_size = 8`; both Windows development GPUs report `subgroup_size = 32`. First direct confirmation from a non-Windows platform (2026-07-30).

**Audit scope:** All shader templates in v0.28.0 (build from `squad/link`, 2026-07-30):

| Shader template | Variant count | Subgroup ops? | Uses `gl_SubgroupSize`? | Affected by subgroup_size diff? |
|---|---|---|---|---|
| `ew_unary.comp` | ~92 | **No** | No | **No** |
| `ew_binary.comp` | ~66 | **No** | No | **No** |
| `ew_select.comp` | ~10 | **No** | No | **No** |
| `q_gemv.comp` | ~16 | **No** | No | **No** |
| `skip_simplified_layer_norm_f32.comp` | 1 | **No** | No | **No** |

**Audit result: zero variants affected.**

All shaders use shared-memory tree reductions (`shared float red[256]; barrier();`) rather than subgroup intrinsics. `q_gemv.comp` (lines 9–12) explicitly documents this:

> "No subgroup operations. Both development GPUs report a subgroup size of 32, which is the strongest possible invitation to bake 32 in and pass every local test. `VkPhysicalDeviceSubgroupProperties::subgroupSize` is not guaranteed to be anything, so the cross-workgroup reduction is a shared-memory tree sized by `gl_WorkGroupSize.x`."

The test suite confirms analytically: all MatMulNBits and elementwise tests pass on lavapipe with `subgroup_size = 8`, zero numerical differences.

**Subgroup size 8 is a portability risk this codebase currently avoids.** An Android device with any `subgroup_size < 32` would be silently broken by any shader that assumed `subgroupSize == 32`. The fact that it does not bite today depends on every shader being written defensively. This must be maintained as new shaders are added.

**Notice to Switch and Mouse:** Any new shader using `subgroupBroadcast`, `subgroupAdd`, `subgroupOR`, `gl_SubgroupSize`, or any other subgroup intrinsic must be authored to handle `subgroupSize` in `[4, 128]`. The `q_gemv.comp` design comment is the template. Do not assume 32 even when both development GPUs report 32.

---

## 7.7 Linux / lavapipe Execution Record (2026-07-30)

**Context:** First execution of a claimed node end-to-end on a Linux Vulkan stack with lavapipe. Previous CI-verified entries (§7.4.1) confirmed lavapipe enumerates and the probe passes; they did not confirm dispatch of a real compute pipeline.

### 7.7.1 Build chain

| Step | Result |
|---|---|
| OS | Ubuntu 24.04.1 LTS, WSL2 |
| Rust | installed via `rustup` into `/root/.cargo/` (toolchain: stable-x86_64-unknown-linux-gnu) |
| `glslc` | version 2023.8 — Ubuntu 24.04 `glslc` package (**note:** CI Ubuntu 22.04 must use LunarG `shaderc` apt repo; Ubuntu 24.04 ships `glslc` directly) |
| `libclang` | `llvm-18-dev` (LLVM 18.1.3) |
| ORT headers | vendored at `third_party/onnxruntime/include/` — no `ORT_INCLUDE_DIR` override needed |
| `CARGO_TARGET_DIR` | `/root/ep-build` (persistent across WSL invocations — avoids systemd private-tmp recycling) |
| Build status | **CLEAN** — zero warnings, zero errors |
| Artifacts | `libonnxruntime_vulkan_ep.so` (1.78 MB), `epctl` (904 KB) |

**WSL note:** `sudo` requires a password for the `justinchu` user. All root-requiring operations must be run as `wsl -d Ubuntu -u root`. The elevated-runner ICD-enumeration trap (§7.4.1 Windows note) does not apply here — WSL bash sessions are not elevated; `VK_ICD_FILENAMES` works as expected.

### 7.7.2 §7.2 Gate check on lavapipe (epctl --probe-loader)

```
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json epctl --probe-loader

Device 0: llvmpipe (LLVM 20.1.2, 256 bits) [Vulkan 1.4.318]
  R1  Vulkan API version (req. >= 1.1)              1.4.318          PASS
  R2  compute queue family                           family 0         PASS
  R3  maxComputeWorkGroupInvocations (req. >= 256)   1024             PASS
  R4  maxComputeSharedMemorySize (req. >= 16384 B)   32768 B (32 KiB) PASS
  R6a DEVICE_LOCAL memory heap                       heap 0           PASS
  R6b HOST_VISIBLE memory type                       type 0           PASS
  subgroup_size: 8  |  subgroup_basic_in_compute: true
  is_uma: true  |  timestamp_period_ns: 1.0  |  timestamp_valid_bits: 64
  Gate result: PASS
```

### 7.7.3 Barrier path selection on lavapipe

`ONNXRUNTIME_EP_VULKAN_BACKEND_PROBE=/root/backend_probe.txt` set before session creation. After the first ORT session: file contains **`sync2`**.

Derivation: lavapipe Vulkan 1.4.318 → `VK_KHR_synchronization2` promoted to core at 1.3 → `caps.synchronization2 = true`, `caps.synchronization2_is_core = true` → `force_legacy = false` (default) → `Barriers::select` → `Sync2Backend::Core`. This is the expected and correct result.

The forced-legacy path (`ep.force_legacy_barriers=1`) is exercised by the barrier parity test (§7.7.4).

### 7.7.4 Test suite results on lavapipe (2026-07-30)

Environment: `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json`, `ONNXRUNTIME_VULKAN_EP_LIB=/root/ep-build/release/libonnxruntime_vulkan_ep.so`; validation layers not loaded.

| Test file | Passed | Failed | Skipped | Notes |
|---|---|---|---|---|
| `test_elementwise.py` | 33 | 3 | 0 | 3 failures = staged ops (Min, Max, Clip-no-bounds) — platform-independent |
| `test_op_table.py` | 61 | 30 | 0 | 30 failures = staged ops — platform-independent |
| `test_matmulnbits.py` | 29 | 1 | 0 | 1 failure = staged DequantizeLinear — platform-independent |
| `test_barrier_parity.py` | **58** | **0** | 28 | **58 = 29 live ops × 2 paths (sync2 + forced-legacy)**; bit-exact; 28 skipped = staged. Prior Windows: 46/28 — increase from newly-landed ops, not a platform difference. |
| **Full suite** (excl. `test_shape_inference_delta.py`\*) | **196** | **34** | **32** | All 34 failures = staged ops ("EP did not execute any node"); zero lavapipe-specific failures; zero numerical failures |

\* `test_shape_inference_delta.py` has a collection-time import error unrelated to Vulkan. Same error on Windows. Not investigated here.

**M0 canonical test:** `test_binary_elementwise[Add-fp32]` **PASSED** — the first execution of a claimed node on a Linux Vulkan stack with lavapipe.

**Provider assertion:** `VulkanExecutionProvider in session.get_providers()` confirmed true for all 196 passing tests. The `assert_vulkan_claims` conftest guard triggered correctly on all 34 staged-op failures. No silent-fallback false-positives.

**Barrier parity detail:** 29 live ops ran sync2 then forced-legacy. Outputs bit-identical in all 58 cases. This is the **third independent implementation** (after Intel Iris Xe and RTX 4060) agreeing on barrier semantic equivalence.

### 7.7.5 What this does and does not say about OQ-12

**What it says:**
- The EP's claim predicates, shader SPIR-V, memory staging, and barrier logic are correct on a CPU-software Vulkan stack with `subgroup_size = 8`.
- The shared-memory reduction design in `q_gemv.comp` is portable to devices with `subgroup_size = 8`.
- The forced-legacy barrier path produces bit-identical results to the sync2 path on a UMA device — the strongest pre-hardware confirmation of barrier parity on UMA topology.
- The EP loads correctly on Linux and shared library dependencies resolve correctly.

**What it does not say:**
- lavapipe is not an Adreno or Mali driver. Its UMA topology matches, but its ISA, cache hierarchy, command-submission model, and driver bugs are entirely different from a real Android GPU. lavapipe results cannot be quoted as Android evidence.
- The ~32.67% sync2-lacking Android fraction (as of 2026-07-30; 2026-07-28 pull: 31.43%; see §10.0.1 for provenance and §10.0.2 for error direction — this figure is **simultaneously a ceiling** on the legacy-path usability benefit, because some gap devices also fail the §7.2 gate, **and a floor** on the gap population, because gpuinfo.org under-represents budget Android; it is not a measured usability value in either direction, and it is moving) remains **entirely unverified** as a usability claim. A lavapipe pass does not de-risk Adreno 5xx / Mali Bifrost memory access patterns, cache coherence, or the device-specific bugs in §6.3 (A1, A2, A3, M1, M2).
- lavapipe does not exercise `storageBuffer16BitAccess`, `shaderFloat16`, or any fp16-specific code path. fp16/int8 capability flags are not confirmed on lavapipe.
- The `synchronization2_is_core = true` path (Vulkan 1.4 core, exercised here) is different from the Vulkan 1.1/1.2 + `VK_KHR_synchronization2` extension path that some Android devices would use.

**OQ-12 is unchanged.** No mobile hardware has been tested.

### 7.7.6 Lane classification: `operational` vs `green` (DESIGN.md §8.9 ruling, 2026-07-30T06:32:18-07:00)

Morpheus ruled on these two states directly, because without the ruling they resolve to the same word and the distinction is silently lost:

| State | Definition | What Link may claim | Gate required? |
|---|---|---|---|
| **`operational`** | The lane exists, executes claimed nodes, and reports results | That the lane is up; that it is a prerequisite for running criterion 10 anywhere but the development desk | No |
| **`green`** | The lane's result is admissible as evidence, satisfies an M0 criterion, or is quotable in a status report | That the lane satisfies criterion 10's tail | Yes — gate artifact with `MATCH` verdict |

**The lavapipe WSL lane is `operational` as of 2026-07-30.** It is not `green`. This is not a demotion — `operational` was a prerequisite for `green`, and the prerequisite was not met before today. The path from `operational` to `green` is §7.8 (gate artifact) wired into CI by Trinity.

> **Superseded in part by §7.4.4 (2026-07-31).** The gate is wired, by Link, into both CI lanes and the conformance lane; the two-state table above gains a third requirement in §7.4.4 (the lane must also demonstrate that its gate can fail) and a fourth verdict state (`UNATTRIBUTED`). **The WSL lane's classification is unchanged: `operational`, not `green`.** Read §7.4.4 for the current per-lane table.

The mechanism that makes `green` structurally non-accidental: a lane's pass condition includes the verdict field. A run that does not measure produces `UNMEASURED`. `UNMEASURED` ≠ PASS and ≠ FAIL — it is §7.9's third state in the CI lane. A lane can be accidentally silent; it cannot be accidentally green.

---

## 7.8 Gate Artifact Design for Criterion 10 (lavapipe lane)

> **STATUS: IMPLEMENTED AND WIRED, 2026-07-31T22:15-07:00.** This section is the design brief and is preserved as written. What actually shipped, where it lives, and the two corrections the brief needed are in **§7.4.4**. In short:
>
> - The artifact is `ci/gate_chain_fp32.py`, built exactly to §7.8.1 (2-node `Add → Relu`, fp32, `[256]`, `X = linspace(-1, 1, 256)`, `Y = -0.5` so the sum crosses zero inside the tensor and the `Relu` clamp is exercised on real data).
> - **Correction 1 — the flag name in §7.8.2 was wrong.** `epctl --check-verdict` does not exist. The verdict gate is `epctl --check-counters <file> --require-dispatches 1` (Tank), which already reads `model_output_equivalence`. One flag, not two.
> - **Correction 2 — the three-verdict table in §7.8.2 is now four**, per DESIGN.md §10.0's third amendment: `UNATTRIBUTED` joins `MATCH` / `DIVERGENT` / `UNMEASURED`, and `SPLIT-FRAME` is a fifth when the two witnesses disagree. The vocabulary is Trinity's `tests/ops/_verdict.py`, imported rather than restated.
> - The lane debugging flag named at the end of §7.8.2 is not spelled anywhere in `.github/workflows/`, and a grep of those files for it finds nothing. That was the point of naming it.

**Context:** DESIGN.md §8.9 ruling (2026-07-30T06:32:18-07:00): *"Each lane carries a gate artifact: the smallest real producer-at-version model that (a) claims a non-zero node count on that lane, (b) contains at least one island of two or more nodes, and (c) exercises at least one proof key in every dtype that lane claims. Trinity chooses and pins it; Link wires it into the lanes."*

**This section is Link's design brief.** Trinity implements the verdict mechanism; Link specifies what must be satisfied for the lavapipe lane.

### 7.8.1 Gate artifact specification

**Artifact name:** `gate_chain_fp32` (provisional until Trinity assigns the canonical artifact name)

**Structure:** a 2-node sequential graph, both nodes in a single island:

```
Input X [fp32, shape: 256]  ──┐
Input Y [fp32, shape: 256]  ──┴── Add ── Relu ── Output Z [fp32, shape: 256]
```

This is the minimal graph that satisfies all three criteria:
- **(a) Claims non-zero nodes:** both `Add` and `Relu` are Live ops with fp32 proof keys; both are claimed on lavapipe.
- **(b) Island of 2+ nodes:** `Add → Relu` is one island of 2 nodes with no CPU fallback between them.
- **(c) Proof keys exercised:** `(ai.onnx, Add, 7+, F32×F32→F32, ew_binary, static, {})` and `(ai.onnx, Relu, 6+, F32→F32, ew_unary, static, {})`.

**fp16 proof keys** are not included in this artifact because `storageBuffer16BitAccess` and `shaderFloat16` are unconfirmed on lavapipe (§7.7.4). When fp16 is confirmed on lavapipe (separate OQ), a second artifact `gate_chain_fp16` must be added. Until then, the lavapipe gate artifact is fp32-only, and the lane's claims on fp16 ops are `UNMEASURED`.

**Correctness oracle:** ORT CPU EP run of the same session on the same inputs. The comparison uses `FP32_ELEMENTWISE` tolerances (`rtol=1e-5, atol=1e-5`). Since `Add` and `Relu` are IEEE-754 elementwise ops with no accumulation, bit-exact agreement is expected and any divergence is a correctness bug, not a tolerance gap.

**Feed values (for reproducibility and non-triviality):** must include at least one negative value (to exercise the `Relu` clamp path) and at least one zero. Suggested: `X = linspace(-1.0, 1.0, 256, dtype=fp32)`, `Y = ones(256, dtype=fp32)`.

**Lavapipe-specific latency note:** first-session artifact compilation takes ~200 ms on lavapipe WSL (SPIR-V JIT via LLVM 20). The artifact runs in < 1 second end-to-end including session creation. This is within any reasonable CI runner budget.

### 7.8.2 Verdict mechanism (coordinate with Trinity)

The lavapipe gate artifact must emit one of three verdicts, consistent with Trinity's `model_output_equivalence` vocabulary:

| Verdict | Meaning | Lane status |
|---|---|---|
| `MATCH` | Vulkan EP output agrees with CPU EP within `FP32_ELEMENTWISE` tolerances; `VulkanExecutionProvider` confirmed in session providers; at least 2 nodes dispatched | `green` (criterion 10 tail satisfied for this lane) |
| `DIVERGENT` | Any output disagrees; or provider assertion fails | Failure — lane is broken |
| `UNMEASURED` | The comparison step was not reached (crash, timeout, skip, or gate step not run) | Default; lane remains `operational` |

**`UNMEASURED` must be the initial state.** The verdict file (`ONNXRUNTIME_EP_VULKAN_VERDICT_FILE` or a separate lane-gate env var — coordinate with Trinity on naming) must be created with `{"verdict": "UNMEASURED"}` before any session is opened. If the process exits without reaching the comparison step, the file remains `UNMEASURED`. A CI step that does not find the verdict file must produce `UNMEASURED`, not absence.

**`epctl --check-verdict` (or Trinity's equivalent check):** exits 0 only on `MATCH`. Exits 1 on `UNMEASURED` or `DIVERGENT`. A CI lane step that succeeds while this check fails is a broken CI step.

**`--allow-unmeasured` flag:** available for local development (debugging build failures where the EP doesn't load yet). Must be explicitly passed; must be absent from the CI step definition. A CI lane cannot be green if it needs this flag.

**Coordination point with Trinity (2026-07-30):** Trinity is implementing the `model_output_equivalence` gate for the Windows/Linux criterion-10 run. The lavapipe artifact is a different artifact (smaller, lavapipe-pinned) but must use the same verdict vocabulary and the same file format. Do not invent a parallel vocabulary — one verdict type, one file schema, two artifact sizes.

### 7.8.3 What the gate artifact does and does not measure

**Measures:**
- That the `Add → Relu` fp32 island dispatches on lavapipe and produces correct values
- That `VulkanExecutionProvider` is actually executing (not silently falling back to CPU)
- That the 2-node island boundary (memory hand-off between ops) is correct

**Does not measure:**
- Multi-run arena reuse correctness (§7.4.2 single-run blindness — this artifact is also run once)
- fp16 paths (explicitly absent from artifact scope)
- MatMulNBits or any kernel outside the elementwise template family
- Barrier correctness (that is the parity suite's job)

**The single-run blindness is NOT fixed by the gate artifact.** It is documented here so it is visible alongside the artifact. The gate artifact satisfies criterion 10's structural requirement (a `model_output_equivalence` verdict exists); it does not claim to be a complete correctness oracle. For the multi-run failure mode, the instrument is `probe_run2.py` (Tank's multi-run discriminator), which is a local-dev tool, not yet wired into CI.

---

## 7.9 `is_uma` Predicate Verification

**The question (coordinator, 2026-07-30):** Confirm lavapipe's `is_uma = true` is arrived at by the corrected predicate ("every heap is DEVICE_LOCAL"), not by the old bug ("largest DEVICE_LOCAL heap is also HOST_VISIBLE") agreeing by coincidence. A broken predicate that happens to be right on this device is worse than one that is wrong, because it will be cited as evidence the predicate works.

**Predicate in `rust/src/vk/caps.rs` (lines 488–500):**

```rust
fn is_uma_memory(mem_props: &vk::PhysicalDeviceMemoryProperties) -> bool {
    let heap_count = mem_props.memory_heap_count as usize;
    // True UMA: no heap lacks DEVICE_LOCAL. A discrete GPU always has a system-RAM heap
    // without DEVICE_LOCAL; an integrated GPU's single heap always has DEVICE_LOCAL.
    (0..heap_count).all(|i|
        mem_props.memory_heaps[i]
            .flags
            .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
    )
}
```

The doc comment (lines 477–486) explicitly names the old predicate and why it was wrong:

> *"The previous predicate ('largest DEVICE_LOCAL heap is also HOST_VISIBLE') incorrectly returned `true` for discrete GPUs with ReBAR enabled — the VRAM heap is both DEVICE_LOCAL and HOST_VISIBLE via ReBAR, while the system-RAM heap has no DEVICE_LOCAL, so the old predicate agreed by coincidence that the VRAM heap was host-visible, while missing the non-DEVICE_LOCAL heap entirely."*

**Unit tests (caps.rs, lines 652–696)** cover four cases explicitly:
1. Single DEVICE_LOCAL heap → UMA: `true`
2. Two heaps: DEVICE_LOCAL + ∅ (classic discrete GPU) → UMA: `false`
3. Two heaps: DEVICE_LOCAL|HOST_VISIBLE + ∅ (ReBAR discrete) → UMA: `false` ← this was the bug
4. Two DEVICE_LOCAL heaps → UMA: `true` (hypothetical cached/uncached UMA variant)

**lavapipe heap structure:**  lavapipe is a CPU software rasterizer. All device memory is system RAM. Mesa lavapipe presents one heap with `DEVICE_LOCAL | HOST_VISIBLE` flags (the heap that satisfied both R6a and R6b in the gate check: "R6a DEVICE_LOCAL memory heap: heap 0 PASS; R6b HOST_VISIBLE memory type: type 0 PASS"). Heap count = 1. The corrected predicate evaluates: *every* heap has DEVICE_LOCAL (heap 0 does → true). 

**Verdict:** lavapipe's `is_uma = true` is arrived at by the corrected predicate AND is genuinely true. The old bug would have also returned `true` on lavapipe, because the ReBAR false-positive requires two heaps (one without DEVICE_LOCAL), and lavapipe has only one. This is NOT "the bug agreeing by coincidence" — a coincidence requires the wrong predicate to be active. The active predicate is the corrected one (unit test case 3 verifies the ReBAR case returns `false`), so lavapipe's `true` is from the right predicate, for the right reason: it is a CPU renderer with a single unified heap.

**Summary:** the coordinator's concern is satisfied. The predicate is correct; lavapipe's `is_uma` value is sound; the unit tests prove the corrected predicate is active and that the ReBAR false-positive is closed.

---

## 7.10 Subgroup-32 Red Instrument: Does the lavapipe Lane Already Catch Baked-32 Shaders?

**The question (coordinator, 2026-07-30, R9 framing):** Is executing on `subgroup_size = 8` sufficient by construction to catch any shader that assumes `subgroupSize == 32`? Or does the lane need an explicit shader-source assertion? If the former, say so explicitly — the instrument already exists and the risk should stop being treated as open.

**Answer: YES — executing on `subgroup_size = 8` is sufficient by construction, IF the lane runs numerical correctness tests.**

The mechanism:
1. A shader that bakes `gl_SubgroupSize == 32` uses 32 as a hardcoded reduction width. On lavapipe with `subgroup_size = 8`, `gl_SubgroupSize` is 8. A hardcoded 32 would cause the shader to:
   - In a subgroup intrinsic call: query 32 elements but only 8 participate → wrong reduction → wrong numerical output
   - In a workgroup-level tree with baked iteration count: read from uninitialised lanes → wrong output
2. "Wrong numerical output" → `assert_matches_cpu` diverges → test RED → build/CI fails

The lavapipe lane already runs `test_elementwise.py` (33 fp32 cases with `assert_matches_cpu`), `test_matmulnbits.py` (29 cases), and the barrier parity suite (58 cases). All use `assert_matches_cpu` or equivalent. If any current or future shader bakes `gl_SubgroupSize == 32` and is exercised by one of these tests, the test will fail on lavapipe and only on lavapipe (both Windows devices report 32 and would not catch the bug).

**The falsifier exists.** The instrument is: the numerical correctness suite running on lavapipe with `subgroup_size = 8`. It satisfies R9's red-instrument test:
> *"Name the instrument that would go red if the claim were false."*
> Claim: "no shader assumes subgroupSize == 32"
> Falsifier: `test_elementwise.py` on lavapipe — a baked-32 shader would produce wrong reduction outputs, diverging from CPU reference, failing the test.

**Condition for the falsifier to remain valid:**
- The lavapipe lane must continue running numerical correctness tests (not just dispatch-existence tests). A `test_add_is_claimed`-only lane would not catch this.
- The gate artifact (§7.8) alone is also insufficient — `Add → Relu` does not exercise any reduction path. The falsifier is `test_elementwise.py` and `test_matmulnbits.py`, which exercise all currently-compiled shader families.
- When new shader templates are added (beyond the 5 currently in the codebase), they must be exercised by a numerical correctness test before the lane's falsifier coverage extends to them.

**What the lavapipe lane does NOT catch:**
- A shader that uses `gl_SubgroupSize` correctly (reading the actual value rather than baking 32) but has a different subgroup-related bug that only manifests at size 8. That would still fail numerically, but the fault localisation would need the Vulkan validation layer's subgroup debugging extensions, not just the lane's presence.
- A new shader added to the codebase that is NOT exercised by any lavapipe test yet. If it ships with a baked-32 assumption before tests cover it, the falsifier gap is open until a test is added.

**Recommended maintenance rule (previously stated as advice, now stated as a mechanism):** Any new shader template added to `rust/shaders/glsl/templates/` must have at least one test case in the lavapipe numerical correctness suite before its op is moved from `Staged` to `Ready`. This connects the baked-32 falsifier to the op registration lifecycle automatically — an op that has no lavapipe correctness test is not `Ready`.

**Risk status: not open — instrument exists.** The note in §7.6 and history.md that "Switch and Mouse must not bake 32" is still correct advice. But the standing risk item can be closed: the lavapipe lane, running the elementwise and MatMulNBits suites, is a working red instrument against baked-32 assumptions in any shader that those suites exercise. The risk becomes open again only when a new shader template is added without a lavapipe test.

### 7.10.1 Re-statement of the argument end to end, after R10–R13 (2026-07-31T22:15-07:00)

The risk stays closed. But the argument was stated before R10 and R13 existed, and it has a link in it that those rules make visible. Stating the chain as a chain, because a decomposition that appears to close is the hardest kind of wrong (R11):

| # | Link in the chain | Status |
|---|---|---|
| 1 | A shader that bakes `32` computes wrong reductions where `subgroupSize` is not 32. | **Holds by construction.** Arithmetic. |
| 2 | lavapipe reports `subgroup_size = 8`. | **Holds — measured**, WSL Mesa 25.2.8, and independently on both CI lanes' Mesa builds. |
| 3 | A wrong reduction produces outputs that differ from the ORT CPU oracle. | **Holds** for every op whose kernel reduces. Elementwise ops do not reduce; the falsifier for those is not this one. |
| 4 | The numerical suite runs on lavapipe. | **Holds** — 196 tests, §7.7.4, and both CI lanes run `tests/ops`. |
| 5 | Those tests execute **on the GPU**, rather than on CPU via a silent fallback that makes them pass anyway. | **THIS WAS THE WEAK LINK, AND IT WAS UNSECURED UNTIL TODAY.** |
| 6 | Therefore a baked-32 shader turns a lane red. | **Holds — now**, and only because link 5 does. |

**Link 5 is the whole point of the missing gate.** A run-time fallback makes `assert_matches_cpu` compare CPU output against CPU output; it agrees, the test passes, and a shader baking 32 sails through untouched because it was never executed. On 2026-07-30 that state persisted for most of a day. The falsifier was not weak — it was *not connected to its subject*, which is R10 exactly: a mechanism in the source tree and not in the call graph is indistinguishable from one never written, and here the "call graph" is the execution path of the graph itself.

**What closes link 5, and it is not the gate artifact alone.** `gate_chain_fp32` is `Add → Relu` — elementwise, no reduction. It cannot itself falsify a baked-32 reduction. What it does is establish, per lane and per run, that **this EP executed at run time on this lane**, with an attribution from ORT's profiling. Combined with the fatal-log grep, which fails the lane on `Falling back` anywhere in the captured suite output — including during the numerical tests themselves — the lane can no longer run the reduction suite on CPU and report green. That is what links 4 and 5 together: the gate proves execution is possible on this lane and the grep proves it did not silently stop being so during the suite.

**So the chain holds end to end, with two conditions I am naming rather than assuming:**

1. **It holds only where the suite actually covers the shader.** Zero of 168+ compiled variants use subgroup intrinsics today (§7.6), so today the chain is a guarantee about an empty set — which is a real state and is worth saying plainly rather than presenting as coverage (R12: `UNOBSERVABLE`, not zero). The `Staged → Ready` rule above is what keeps it non-empty as templates arrive.
2. **It holds only on runs where the fatal-log grep had an input.** A lane that does not capture ORT's stderr scans a log in which `Falling back` cannot appear and reports a clean scan. Both lanes now tee with `2>&1`, and `ci/check_fatal_log.py` reports `ERROR(instrument=log_not_captured)` — never a pass — when the log is missing. That is the falsifier for the falsifier.

**What would break it again**, so it is written down rather than rediscovered: removing the `2>&1` from a tee; adding a lane that runs the numerical suite without the gate steps; a new reducing shader template reaching `Ready` without a lavapipe test; or ORT changing the wording of its fallback line — which is why the grep is a *second* witness to Guard D's profile parse and not a replacement for it. Each of those is a specific, checkable thing, which is the only kind of caveat that survives.

### 7.10.2 Closing the subgroup-32 argument (2026-08-01T09:55-07:00)

Link 5 is now secured on my desk in all three polarities (§7.4.4) and wired into all three
CI lanes with a falsifier each, including one that does not depend on the loader. The chain
therefore stands as stated, and this is the last time I intend to re-open it.

**The independent support, which is what makes it closeable rather than merely unrefuted.**
Fact Checker verified that llama.cpp — a Vulkan compute backend far ahead of this one in
device coverage — ships a **subgroup-free shared-memory tree reduction as its fallback path,
with subgroup variants gated behind capability queries**, not baked in. Two things follow,
and only the second is about us:

1. Our constraint is not eccentric. The most-deployed Vulkan inference backend in the field
   made the same choice on the same reasoning, which is what a decomposition that is *right*
   rather than merely internally consistent tends to look like from outside (R11).
2. **Our no-subgroups constraint is cheap.** Fact Checker's stronger finding is that the
   dominant performance gap in that codebase comes from **packed loads and multiple
   accumulators, not from subgroup operations**. So the thing we gave up by refusing to bake
   32 costs us much less than the thing we have not yet done — which relocates the
   optimisation argument to memory access patterns, where the Intel/NVIDIA residual already
   points (bandwidth predicts only 3.08× of the measured 13.52× kernel gap; the 4.39×
   residual is our design, not the hardware's).

**And the falsifier is still the same one, by construction.** lavapipe reports
`subgroup_size = 8` on every lane we run it in. A shader that assumed 32 computes a wrong
reduction there and nowhere else in our matrix — both Windows devices report 32 and cannot
catch it. That is not an argument that improves with agreement; it is arithmetic plus one
measured capability value, and the only way it fails is if the suite that would notice never
executes on the GPU. That was link 5, and link 5 is what the gate and its two negative
controls now hold.

**What is still an empty set, said plainly rather than dressed as coverage (R12).** Zero of
168+ compiled shader variants use subgroup intrinsics today. The chain is currently a
guarantee about nothing, correctly constructed and waiting for a subject. The
`Staged → Ready` rule (§7.10) is what keeps it non-empty as templates arrive; the day a
reducing template lands without a lavapipe correctness test, the guarantee is open again and
this section is wrong until a test is added.

**Status: closed.** Re-open on any of the four break conditions listed above, or on a new
reducing shader template reaching `Ready` without lavapipe coverage. Not on general unease.

---

## 7.11 Device-State Records Across the Platform Matrix (§10.0 obligation 8) — added 2026-08-01T14:20-07:00

**The finding, and why it lands in my file rather than Niobe's.** Switch showed that
`gpu_steady_tail()` is a **variance test over a suffix and therefore cannot see a bias**:

```
soloA      [SOLE_TENANT]              STEADY   11.525 ms   RSD 0.8098%
contended3 truncated to 20 inferences STEADY  126.647 ms   RSD 0.9103%   ← 10.99× wrong
contended3 truncated to 28 inferences STEADY  126.647 ms   RSD 0.8035%
```

**The wrong number carried the better RSD.** The RTX 4060 idles at 210 MHz against a
3105 MHz boost, and a run held at idle clock is *perfectly steady*, so it earns the gate's
most confident verdict. A low clock does not raise RSD; it lowers it. Morpheus ruled it
**R9 amendment 5 — the anti-correlated falsifier**: *ask which way a check moves when its
subject is wrong; if it moves with the reader's confidence it cannot be repaired by
tightening,* and it is demoted from gate to precondition.

§10.0 obligation 8 is the replacement, and **amendment 1 is what makes it mine**: the
obligation is stated as a *record, never as a tool*. `nvidia-smi` is one vendor's
implementation; this project is cross-platform by mandate (§1.1). The obligation names the
**content** — a tenancy verdict, clock min/median/max against the board's own advertised
maximum, over the statistic's own suffix — and any platform that can produce that content
satisfies it. Which platforms those are is a support-matrix question.

### 7.11.1 Producer coverage by platform — and the row that matters is the empty one

Registry lives in code (`ci/device_state.py :: PRODUCERS`) so it is testable; this table is
its documentation, not a second copy of it.

| Platform | Telemetry that could produce the content | Producer written? | A device-clock figure taken here is… |
|---|---|---|---|
| **NVIDIA** (Windows/Linux, discrete + laptop) | `nvidia-smi` — `clocks.sm`, `clocks.max.sm`, compute-app table | **Yes** — `bench/results/probe_gpustate.py` (Niobe) | **quotable**, given the record covers the statistic's own suffix |
| **AMD** | `rocm-smi` — `sclk`, process table | No | `STEADY_UNCERTIFIED` |
| **Intel iGPU** | `intel_gpu_top` (Linux) / Level Zero sysman (Windows) | No | `STEADY_UNCERTIFIED` |
| **Apple / MoltenVK** | `powermetrics` GPU frequency + residency (**needs root**) | No | `STEADY_UNCERTIFIED` |
| **Adreno** | `/sys/class/kgsl/kgsl-3d0/{gpuclk,max_gpuclk,gpubusy}` | No — no hardware (OQ-12) | `STEADY_UNCERTIFIED` |
| **Mali** | `/sys/class/devfreq/*.mali/{cur_freq,max_freq}` | No — no hardware (OQ-12) | `STEADY_UNCERTIFIED` |
| **lavapipe / llvmpipe / SwiftShader** | **none — there is no device clock** | Structurally impossible | `STEADY_UNCERTIFIED`, **permanently** — see §7.11.4 |

**Absence is never a waiver, and the direction it cuts is the point.** Obligation 8
amendment 2: the cheapest way to satisfy the obligation as first worded is to take the
measurement on a platform with no clock telemetry, where the requirement is vacuous and
the figure comes out unqualified. Morpheus named the **Intel Iris Xe** as that loophole's
biggest beneficiary, and he is right in a way that bites this file specifically: §5.2 makes
Intel our *spec-conformance oracle*, so it is the device we deliberately run things on —
and it shares its power budget with loaded CPU cores, so it is **more** exposed to clock
bias than the discrete board, not less. It is not exempt. It is unmeasured, and the entry
above says so in plain text, which is the same standard §5 applies to `untested` hardware.

**A CI runner with no GPU telemetry is that same loophole at scale.** Every GitHub-hosted
runner this project uses is exactly that, which is why the guard in §7.11.2 treats "no
producer on this host" as *louder* than "no record", not quieter.

### 7.11.2 The lane is now unable to publish a duration — by construction, not by luck

I grepped `ci/*.py` for `tenancy`, `SOLE_TENANT`, `sm_clock`, `STEADY_UNCERTIFIED` and
`gpu_steady`. **Zero matches.** The gate proves the EP *executed* and that the verdict is
attributed; it said nothing about the device state that produced any timing. That was
survivable only because no lane quoted a timing figure — a property no mechanism was
holding in place.

`ci/check_device_state.py` now runs in **all three lanes** (`build-test-linux`,
`build-test-windows`, `conformance`) on `always()`, because a figure published by a run
that already failed is the one most likely to be quoted later. Its terminal states are
R13's, and which is which is decided by obligation 8's amendments:

| Outcome | When | Exit |
|---|---|---|
| `PASS` | the lane published no lane-authored timing figure, **or** every one carries a certified companion | 0 |
| `FAIL(condition=STEADY_UNCERTIFIED)` | a figure was published and its record is missing or incomplete, on a host that could have produced one | 1 |
| `ERROR(instrument=device_state_producer_absent)` | a figure was published on a host with **no producer at all** | 4 |
| `ERROR(instrument=device_state_probe_failed)` | the record says the probe failed. **Never** `SOLE_TENANT` (amendment 3) | 4 |
| `ERROR(instrument=lane_evidence_absent)` | nothing to scan. R12: `UNOBSERVABLE`, never a clean lane | 4 |

Two witnesses with different failure modes (R13 obligation 3): a walk over the lane's JSON
artifacts, **and** a scan of `$GITHUB_STEP_SUMMARY` for a duration in prose — a JSON parser
cannot see `11.525 ms` in a sentence, and a figure written into a job summary is published
exactly as much as one written into an artifact.

**The record format is Niobe's, not a second one.** `ci/device_state.py` reads the shape
`bench/results/probe_gpustate.py :: summarise()` already writes (`verdict`, `sm_mhz`,
`sm_max_mhz`, …). This is the same refusal that kept the CI side on Trinity's verdict
vocabulary: a second format would be R11 in its purest form — two names for one
measurement, appearing to close. **One key is required that no producer emits yet:**
`window`, declaring the suffix the statistic was computed over, because obligation 8 says
the window is *"the suffix the statistic was computed over, not the run"* and a record
covering the whole run does not establish that. Its absence is reported by name
(`STEADY_UNCERTIFIED(reason=incomplete_record, missing=[window])`) rather than assumed
away. Raised with Niobe as a decision record rather than added unilaterally.

**Obligation 8b is implemented too** (`certifies_comparison`): two figures compare only if
their records agree — same tenancy verdict, overlapping clock during each statistic's own
window. Not satisfied by both being `STEADY`; that both are steady is the whole content of
the finding.

**What the guard found on its first real run, which is why it is not decorative.** The
lane's uploaded evidence *already* contained durations: the EP's counters snapshot carries
`session_staging_upload_us` and `session_staging_readback_us`, and ORT's profile carries a
`dur` per node event. Neither is authored by the lane, and requiring an instrument's raw
output to carry a device-state companion would fire on every healthy run — the fastest way
to teach a reader to ignore a check. So `ci/device_state.py :: INSTRUMENT_DUMPS` excuses
them, and the excuse is closed three ways:

1. It is a **code-level list with a reason per entry**, not a `--exclude` flag. There is no
   runtime switch, and a test asserts there is none. An entry costs a code change and a
   test review, which is the difference between an exemption and a waiver.
2. Excused figures are **still printed**, on every run, as
   `STEADY_UNCERTIFIED (carried, not claimed)` with their values. The scope of the excuse
   is auditable from the lane's own output rather than from my source file.
3. The moment one of them is *quoted*, the quote lands in a lane artifact or the job
   summary, where the guard sees it.

Asked as the drafting rule requires — *what is the cheapest thing that satisfies these
words without their intent?* — the answer is "call your figure an instrument dump", and
those three are what close it.

### 7.11.3 The Windows ICD-suppression probe: wired, and it was a constant

§7.4.1 documents that the LunarG loader silently ignores `VK_DRIVER_FILES` in elevated
processes and that GitHub's Windows runners are elevated, so that lane's ICD-removal
negative control may never have fired. I added a guard that was supposed to *say so*
instead of blaming the gate. **It was written, it was wired, and it was a constant.** It
tested:

```powershell
if ($probe -match 'passed the §7\.2 capability gate') { ...ineffective...; exit 0 }
```

`engine::loader_probe_report()` emits `"{n} device(s) passed the §7.2 capability gate."` on
**every** run, and `n` is `0` when the suppression *did* take. The match succeeded
unconditionally: the step short-circuited to `exit 0` on every run, in both directions.
**A detector that fires on every input is not a detector, it is a constant** — the same
sentence `probe_gpustate.py` uses about its own ancestry check, and the same defect one
layer up.

Replaced by `ci/check_icd_suppression.py`: it parses the **count**, has three terminal
states, six two-polarity tests, and writes a JSON record whose content varies with its
input — which is R10's falsifier for "this probe is wired" and is precisely what a
`::warning` annotation is not. It is wired into the **Linux lane as well**, where the
suppression is expected to work: expected is not observed, and the Windows lane believed
"expected" for weeks.

**A second bug, found only by running it.** My first draft read the capability-gate line
and nothing else. Measured on real hardware, both polarities, with the real binary:

```
loader healthy    "2 device(s) passed the §7.2 capability gate."                 epctl exit 0
ICD suppressed    "FAIL: vkCreateInstance returned ERROR_INCOMPATIBLE_DRIVER."   epctl exit 3
                  ...and the capability-gate line is never printed at all
```

A real suppression **never reaches** the line the first draft was parsing, so it would have
classified every successful suppression as `probe_report_unreadable` and short-circuited
the control on every run — reproducing the exact bug it was written to fix, one draft
later. Both shapes are now recognised, the epctl exit code is carried as a second witness
with a different failure mode, and a disagreement between the two witnesses is itself an
instrument state rather than resolved in the convenient direction. The lesson is more
durable than the regex: **run the instrument in both polarities before believing a
parser.**

### 7.11.4 What a device-state record means on lavapipe — the ruling, written down now

This is the case where *"no telemetry, therefore no requirement"* is most tempting and most
wrong, so it gets an answer in prose rather than a silence in a table. Two readings are
available and only one is honest.

**The tempting reading.** *"A software rasteriser has no device clock to be biased, so
obligation 8 does not apply and the figure is unqualified."* This is the waiver amendment 2
forbids, and it is worse here than anywhere else in the matrix: lavapipe runs on the **host
CPU**, which is the single most contended resource on a shared CI runner. A figure taken on
lavapipe is not immune to contention bias; it is *maximally* exposed to it, and the
exposure is invisible because the usual instrument is pointed at a GPU that is not there.

**The honest reading, and this project's ruling.** The obligation's content is *the state
of the device that produced the timing*. On a CPU renderer that device is the host, so the
corresponding record carries host quiescence, CPU frequency min/median/max against the
package maximum, and a host-tenancy verdict — which is `bench/`'s **machine-quiescence
verdict**, an instrument that already exists for wall clock, and not a GPU probe.

> **Ruling: lavapipe can never certify a *device-clock* figure, and this is permanent
> rather than pending.** There is no device clock on a CPU renderer — the quantity does not
> exist, so no instrument can be built to record it. What *is* pending is a weaker and
> different claim: a host-state record for a *wall-clock* figure. If that producer is ever
> written it certifies wall clock and still never certifies device clock. **The two must
> not be allowed to trade names.**

This is consistent with the M1 interlock amendment's own closing sentence — *wall clock
carries the quiescence verdict, device clock carries the device-state record, and there is
no third surface to retreat to.* On lavapipe the second surface is simply not present.

**The consequence for the lane matrix, stated so nobody has to derive it.** All three CI
lanes run lavapipe. **Therefore no CI lane can ever publish a certified device-clock
figure.** That is not a gap a better probe closes; it is closed by a GPU runner or not at
all, and it belongs in the matrix in plain text next to `untested`. It also means the
`PASS` this guard prints on those lanes is always the first kind — *"published nothing
timed"* — and the check says so in its own output rather than letting a reader infer the
stronger one.

### 7.11.5 Criterion 5's denominator attack — is my gate artifact exposed?

Morpheus named the attack: **run at idle clock, inflate the total, watch the share
collapse.** M1 criterion 5 is *steady-state recording share below 5%*, and a share can be
satisfied by inflating its denominator: device time inflates ~21×, host recording time does
not, the share collapses far below 5%, the series is perfectly steady, and every gate
reports its most confident verdict. Not hypothetical — it is `246.720 ms at RSD 0.1163%`
with the arithmetic run backwards.

**Answer for my lanes: the gate artifact is not exposed, and the reason is structural
rather than lucky.** `gate_chain_fp32`'s verdict is a function of **counts and an exact
comparison only** — `own_provider_execution_count`, `counters_dispatches_executed`,
`profile_node_events`, `max_abs_diff` — with no duration, no total and therefore no
denominator anywhere in it. There is no share to collapse. A runner pinned at its idle
clock produces the same `MATCH`, the same `executed_by` frame and the same
`max_abs_diff=0`; it produces them more slowly, and the verdict does not have a slot for
that. Both negative controls are equally clock-invariant: an idle clock cannot turn
`FAIL(condition=UNATTRIBUTED)` into `MATCH`, because what they detect is *nothing executed*,
which no clock rate reaches.

That is §10.0.4's invariance preference paying out on the CI side, in the same shape it
paid out for Niobe: **counts survived what clocks could not** (147,618 → 354 barriers;
1997.6 → 0.756 MiB upload). The gate was written on counts for a different reason — a
timing threshold in CI is flaky on shared runners — and it is immune to this attack as a
consequence rather than as a design goal. I record that distinction rather than claim
foresight I did not have.

**Where the lanes *would* be exposed, which is the part worth guarding.** The attack does
not need my gate; it needs a lane that publishes a share or a duration at all. A future
`bench` lane reporting criterion 5's recording share, or a regression lane reporting
ms/inference, lands the attack squarely here — and on a GitHub runner it lands with the
worst possible combination: a software rasteriser whose "device clock" cannot be recorded,
on a host whose CPU contention is invisible to us and moves the denominator directly.
§7.11.2 is what makes that lane impossible to add quietly, and §7.11.4 is why the answer
for such a lane is *"not on this runner"* rather than *"with a better probe"*.

**And §10.0.4's named abuse applies to this section itself:** *do not hand the reader a
count and let them supply the clock.* The gate's counts say the EP executed and the outputs
agreed. They say nothing whatever about how long it took, and no reader should be able to
leave §7.11 believing otherwise.

---

## 7.12 `operational` vs `green`: what each lane would catch, and what it silently misses — added 2026-08-01T18:40-07:00

An **operational** lane runs. A **green** lane runs *and would go red if the thing it watches broke*. Only the second is worth having. The three CI lanes have been classified `operational` since 2026-07-31, and this section is the work of closing that gap — check by check, with the failing arm produced rather than assumed.

The classification is not prose. It lives in **`ci/lane_inventory.py`** as data, is published as a lane artifact (`lane-inventory.json`) by `ci/check_lane_inventory.py`, and every entry must carry:

* what the check watches;
* the **mutation** that produces its failing arm, written down so the demonstration can be repeated — an unrepeatable demonstration is a memory;
* both arms as observed, quoting the failure **text** and never a count (R13);
* the **`misses`** column, which the validator refuses to let anyone leave blank. Every check misses something; a blank column means nobody looked, not that nothing is missed.

Four statuses, and three of them are not "pass":

| Status | Meaning |
|---|---|
| `DEMONSTRATED` | Both arms observed. The only status that supports the word *green*. |
| `UNDEMONSTRATED` | Runs and passes; nobody has ever seen it fail. **May be a constant.** A candidate for evidence, not evidence. |
| `IMPOSSIBLE_HERE` | The failing arm cannot be produced here, for a stated structural reason. A closed question with an answer, not a lesser `UNDEMONSTRATED`. |
| `RED_NOW` | The failing arm *is* the current tree. The strongest possible falsification, and never a problem with the check. |

`ci/check_lane_inventory.py` also refuses to let the inventory rot: **every gate step in `ci.yml` must have an entry.** An unclassified step is precisely the state the Windows ICD negative control lived in for weeks — running, reported, and never once falsified. The reverse direction is deliberately not checked, because "stale entry" and "silently deleted step" want opposite responses, and guessing between them from YAML is the R10 mistake in miniature.

### 7.12.1 The finding that changed the shape of this work

**No CI lane has ever run the crate's unit tests.** The Linux lane ran exactly one cargo test target (`--test layering`); the Windows lane ran none at all. That left **440 unit tests and four integration targets** — `cdylib_load`, `dump_capabilities`, `host_registration`, `portability` — unexecuted since the project began.

Wiring them in immediately produced two live findings and one flake, none of which any lane could previously see:

1. **`cargo clippy --release --all-targets -- -D warnings` is red**, on the exact CI command and the same unpinned `stable` channel CI installs (rustc 1.97.1): an unused `crate::engine::DeviceMemoryProvider` import, a manual `RangeInclusive::contains`, and two `direct cast of function item into an integer`. All four are in `--all-targets` (test-profile) code, which is why they went unnoticed — `cargo build --release` is clean, so the lane looked healthy from the build step. Confirmed failing on both device lanes of the 2026-08-01 `main` run.
2. **The P1 portability lint is red** on `src/ep.rs:2457`: `_file: *const ort::wchar_t` named without a `#[cfg(windows)]` gate. bindgen only emits `wchar_t` when targeting Windows, so **`cargo test --lib` cannot compile on Linux at all**. The remedy is the cfg-selected `OrtChar` alias that `tests/mock_ort/mod.rs:155/158` already defines. The lint's own message names this class: *"the class of bug that only shows up on a CI lane for another OS, hours later, while blocking everything behind it."* It has existed in `rust/tests/portability.rs` and had never been run.
3. **A flake**: `counters::tests::a_pinned_authoritative_counter_reports_unobservable_and_never_zero` failed once in three full runs, writing to a fixed path under a process-global env var while other tests run in parallel. Recorded, not masked. The suite is wired at **default parallelism deliberately** — `--test-threads=1` would hide cross-test state leakage, which is a defect and not a test-runner setting.

None of these are mine to fix (`rust/src/` is Switch's, `src/ep.rs` and `rust/tests/` are Tank's) and none is fixed here. They are routed, and the lanes that would have caught them are now wired.

### 7.12.2 The software-rasteriser blind spot, stated plainly

**lavapipe reports `timestampPeriod` 1.0 — exactly as NVIDIA does. Intel Iris Xe reports 52.0833.**

At a period of 1.0 the conversion is the **identity**. A build that drops it is *numerically indistinguishable* from a build that performs it, on every device CI can reach, while under-reporting the Iris Xe by 52.0833×. **No amount of executing on the CI device can see this**, because the CI device cannot tell the two builds apart. It is the anti-correlated shape from R9 amendment 5 in a new place: the defect is invisible precisely where we look hardest, and a lane that runs more op tests on lavapipe gets no closer to catching it.

**So: lavapipe cannot catch it, and no configuration of the lavapipe lane ever will.** Saying so is the point of this subsection. What *does* catch it is a check that never touches a device:

> `rust/src/trace.rs` unit tests construct a **synthetic calibration with a non-unit period** — `cal(40.0, 64).ticks_to_ns(100, 1100) == Some(40_000.0)` — and are therefore host-independent by construction.

**Falsified 2026-08-01.** Dropping both period multiplies in `trace.rs` (`Some(span as f64 * f64::from(self.timestamp_period_ns))` → `Some(span as f64)`, and the same in `ticks_to_axis_us`) turns four tests red on a Windows host whose own period is 1.0:

```
trace::tests::treating_intel_ticks_as_nanoseconds_is_wrong_by_fifty_two_times
  panicked at src/trace.rs:1680: period scaling not applied: 100000 ns for 100000 ticks
trace::tests::timestamp_period_is_applied_and_is_not_assumed_to_be_one
trace::tests::undefined_upper_bits_on_a_thirty_six_bit_counter_are_masked_away
trace::tests::an_intel_counter_wrap_does_not_produce_a_negative_or_absurd_duration
```

Arms differ; the clean tree gives `440 passed`. **The tests already existed and had simply never been run.** The blind spot was not a missing instrument, it was an unwired one — which is the same failure mode as the ICD control, one layer up.

The same argument covers `timestampValidBits`: lavapipe and NVIDIA report 64, Iris Xe reports **36** and wraps roughly hourly. A 64-bit assumption is correct on every device CI can reach and wrong on the one device we own that is a spec-conformance oracle. Two of the four tests above are its falsifier.

**The mirror image is still open and is recorded as such.** These tests prove the *arithmetic*; they cannot prove the conversion is **called at the real call sites**. Execution would prove the call site — and lavapipe's period of 1.0 makes a dropped call indistinguishable from a performed one, which is where this started. `ci/lane_inventory.py :: BLIND_SPOTS['conversion_call_sites']` carries it with substitute `None` and status `IMPOSSIBLE_HERE`, because nothing in this repository currently catches it.

### 7.12.3 Blind spots no lane catches

| Blind spot | Substitute | Status |
|---|---|---|
| `timestamp_period_52x` — period conversion dropped | `trace.rs` synthetic-period unit tests, now wired | `DEMONSTRATED` |
| `timestamp_valid_bits_36` — masking / wrap | the same tests, with 36-bit calibrations | `DEMONSTRATED` |
| `device_clock_state` — a figure taken at idle clock | **nothing.** No GPU in any lane, therefore no clock | `IMPOSSIBLE_HERE` |
| `concurrency_and_barriers` — a missing barrier | barrier-count parity (structural) + real hardware (not reproducible) | `UNDEMONSTRATED` |
| `vendor_driver_behaviour` — UMA/discrete, subgroup width, fp16/int8, ReBAR | local-dev observation; a GPU runner the project does not have | `IMPOSSIBLE_HERE` |
| `conversion_call_sites` — correct arithmetic, never invoked | **`ci/check_tick_conversions.py`**, a static source screen (§7.13) | `DEMONSTRATED` (2026-08-01) |
| `composed_workflow` — two correct branches, a broken union | **`--union-with`** on the lane inventory (§7.14) — one shape, one file | `DEMONSTRATED` (2026-08-02, replayed) |
| `census_denominator` — a census complete by construction, 12 out of its own 12 | **`ci/check_census_completeness.py`** (§7.15) — the whole enumerated from production Rust | `DEMONSTRATED` (2026-08-02) |

The CI matrix has **one device in it wearing two operating systems**. That is a portability result about our code and not a hardware-coverage result, and it is why OQ-12's **~32.67% as of 2026-07-30** is simultaneously a ceiling and a floor.

### 7.12.4 What was closed here

| Check | Was | Now | Mutation that produced the red arm |
|---|---|---|---|
| `build.rust_unit_tests` | not run at all | `DEMONSTRATED` | drop both period multiplies in `trace.rs` |
| `build.portability_lint` | not run at all | `RED_NOW` | none needed — the tree is the failing arm |
| `build.layering_lint` | `UNDEMONSTRATED` | `DEMONSTRATED` | plant `use ash::vk as _;` in `src/ops/norm.rs` |
| `build.clippy` | `UNDEMONSTRATED` | `RED_NOW` | none needed — reproduced locally on the CI command |
| `device.fatal_log_line` | `UNDEMONSTRATED` | `DEMONSTRATED` | feed it a log containing ORT's `EP_FAIL ... Falling back` line |
| `device.op_correctness` | `UNDEMONSTRATED` | `UNDEMONSTRATED` | **not performed** — needs a kernel edit (Switch) and an expectation (Trinity); named, not claimed |
| `build.integration_targets` | not run at all | `UNDEMONSTRATED` | being wired is the precondition for falsifying; it is not the falsification |

**A scope fact found while falsifying the layering lint, recorded rather than fixed:** the lint scopes to `src/ops/` only. The same `use ash::vk as _;` planted in `src/trace.rs` passed all 26 of its tests — despite the archived decision that put the timestamp arithmetic in `trace.rs` specifically to keep it *"on the right side of the layering lint (no `ash`)"*. **The rule that decision relied on does not exist.** `rust/tests/` is not mine.

Both device lanes therefore remain **`operational`**, and the inventory names exactly why: `device.op_correctness` and `build.integration_targets` have never been observed to fail. `lane-checks` is **`green`** — every check in it has a demonstrated failing arm.

### 7.12.5 Two defects found in my own `always()` steps

Reading the real `main` run rather than my own YAML (R10) turned up two, both mine:

1. **Three device-state tests were host-dependent.** They asserted `FAIL(condition=STEADY_UNCERTIFIED)` and passed on a developer machine with `nvidia-smi`; the runner, having none, correctly produced `ERROR(instrument=device_state_producer_absent)`. Both are red, so the *guard* was never wrong — **the tests were**, and a test whose outcome depends on which machine ran it has one arm wearing two coats. Closed by extending `ONNXRUNTIME_EP_CI_DEVICE_STATE_PRODUCERS` with `simulate:<name>`, so both polarities are reachable on any host. That is a switch on a guard, which is the shape a waiver takes, so its safety property is **asserted rather than argued**: `test_no_producer_override_can_ever_yield_a_pass` checks that an unrecorded duration is red under *every* accepted value. The override selects which red, never green — a `PASS` is decided by `certify()` reading the record, not by asking the host what it can measure.
2. **Both `always()` steps added a second red to a lane that had already died.** The Linux lane failed at Clippy, never reached the step that creates `bench/results/ci-lane`, and `check_device_state.py` and `check_fatal_log.py` each reported a true instrument error about a subject that had never existed. Noise on a red lane is how a real finding gets scrolled past. Closed with a **lane marker** (`.lane-reached`) written by the lane's own evidence producer: marker absent ⇒ `ERROR(instrument=lane_did_not_reach_evidence)`, printed loudly as *not a pass*, with a `::warning` annotation and exit 0; marker **present** and evidence absent ⇒ the original exit 4, because that is a genuine instrument failure. The direction that matters is that **the marker cannot be absent on a run that did publish**, and `test_the_marker_cannot_excuse_a_published_duration` holds it: evidence that exists is scanned exactly as before, marker or no marker.

### 7.12.6 Not mine, blocking, and routed

* `cargo fmt --check` is red on `rust/src/allocator.rs` (×2), `rust/src/ops/indexing.rs`, `rust/src/ops/norm.rs`, `rust/src/ops/partition.rs` (×3). Not touched — those files are being edited by concurrently-running agents.
* The clippy and portability failures in §7.12.1.
* Closing the duty-cycle mechanism still needs `nvidia-smi --lock-gpu-clocks` and an elevated shell.

---

## 7.13 `conversion_call_sites`: a third instrument family, because neither of the first two can see this — added 2026-08-01T21:30-07:00

### 7.13.1 The residual, stated as it was stated against me

§7.12.3 listed `conversion_call_sites` as a blind spot with substitute **nothing**. The reason was in my own report and was not papered over:

> those tests prove the conversion is correct *where called*; they cannot prove every call site calls it.

That is the 52× defect class one layer up. `build.rust_unit_tests` closes **"is the arithmetic right"** — it constructs a synthetic calibration with a non-unit period and four tests go red if the period multiply is dropped. It says nothing whatever about **"does every path use it"**, because a path that skips the conversion never appears in a test *of* the conversion.

And no device lane can close it either, for the reason that makes this class nasty:

> `timestampPeriod` is **1.0** on lavapipe, on SwiftShader and on NVIDIA. At a period of 1.0 the conversion is the **identity**. A build that skips it is byte-identical to a build that performs it on every device this CI can reach.

So the defect is invisible to **both** instrument families the project runs. That is rare. Almost every other gap here is "we lack a runner"; this one would survive having every runner, because the runners we would buy are the ones where the bug does not manifest. Only Intel Iris Xe (**52.0833 ns/tick, 36 valid bits**) exposes it, and it is a laptop on a desk, not a lane.

**A defect class invisible to every existing instrument is R9's argument for a new one**, not for more runs of the old ones.

### 7.13.2 The instrument: static, because the period's value is what blinds the others

`ci/check_tick_conversions.py` decides the question from **source text**, where `timestampPeriod` has no value at all and therefore cannot be 1.0. It needs no GPU, no Vulkan loader, no ORT, and — deliberately — **no `tests/ops/_verdict.py`**: it emits no *verdict* about the EP, only a finding about source, so a vocabulary outage that stops every other lane check leaves this one running.

Three rules, each decidable from the text:

| Rule | Claim | Condition token when red |
|---|---|---|
| **R-A** arithmetic monopoly | A tick may be moved, stored, compared and logged anywhere; it may be **scaled** only inside `GpuTimestampCalibration`'s converters. A by-hand `ticks * period` is a bypass *even though it uses the period*, because it skips the mask and the wrap recovery — on Intel that is a 36-bit counter read as if it were 64. | `tick_conversion_bypassed` |
| **R-B** single producer | Raw, unmasked ticks enter the program at exactly one site — `TimestampPool::read_results` — and that site's enclosing function constructs the calibration. **This is the arm that addresses "does every path use it."** A second reader anywhere in the tree has to justify itself. | `raw_tick_producer_not_unique` |
| **R-C** allowlist integrity | Every exemption still matches a live line. An exemption that has lost its site does not expire quietly; it becomes a blanket over whatever moved into its place. | `allowlist_entry_without_a_site` |

Result on the tree as it stands: **41 `.rs` files, 33 tick-bearing production lines, and exactly three sites in the whole codebase that scale a tick** — `ticks_to_ns`, `ticks_to_axis_us`, `mask_ticks`, all inside `GpuTimestampCalibration`. Each is recorded in `ci/tick_conversion_allowlist.json` with a reason and an owner. `trace.rs` **has no roster owner**; the allowlist says so rather than inventing one.

### 7.13.3 It is conservative on purpose, and the allowlist is the design

A lane script cannot do sound Rust semantic analysis, and one that *looked* like it could would be **R11 exactly** — a decomposition that appears to close is the hardest kind of wrong. So the screen over-reports, and every over-report must be either fixed or entered in the allowlist **with a recorded reason and an owner**. Adding a bypass then requires a visible edit to an allowlist file *in the same diff as the bypass*. That is a thing a reviewer can see; a missing function call is not.

Two rules were tightened rather than allowlisted, because they were decidably wrong rather than debatable: `->` in a signature is not arithmetic, and `timestamp_period_ns` propagating from the driver limits into the calibration **under its own name** is a move, not a scale. Filling the allowlist with entries like those would teach reviewers the file is noise, which is the failure mode that matters most for a file whose whole value is that someone reads it.

### 7.13.4 Both arms, demonstrated — `ci/negative_control_tick_conversions.py`

A screen that has only ever been observed passing is one step from a screen that cannot fail. Each defect is injected into a **scratch copy** of `rust/src` under `bench/results/`; the checked-out source is never written to.

| Injection | Result |
|---|---|
| baseline, unmodified copy | `PASS` — without this, no red below is attributable to its injection |
| `let bypass_ns = end_ticks - begin_ticks;` — the 52× defect itself | `FAIL(condition=tick_conversion_bypassed)`, line quoted |
| `let raw_span = end_ticks;` — the same defect laundered through a rename | `FAIL(condition=tick_conversion_bypassed)`, line quoted |
| `let hand_ns = iv.begin_ticks as f64 * 1.0;` — period applied by hand, mask skipped | `FAIL(condition=tick_conversion_bypassed)`, line quoted |
| a second `read_results()` caller | `FAIL(condition=raw_tick_producer_not_unique)` |
| an allowlist entry whose pinned text no longer exists | `FAIL(condition=allowlist_entry_without_a_site)` |
| allowlist deleted | `ERROR(instrument=allowlist_unreadable)`, exit 4 |
| source tree emptied | `ERROR(instrument=no_tick_sites_found)`, exit 4 — R12: finding nothing is `UNOBSERVABLE`, never a clean tree |

The rename case is not decoration. A name-based screen is defeated by `let raw = end_ticks;` followed by arithmetic on `raw`, which carries no tick token. Rebinding a tick to a non-tick name is therefore **itself** the finding, caught at the boundary where it is still visible.

**Two defects in the screen were found by its own controls, not by review:**

1. The first draft's injection anchor did not exist in `session.rs`. The control reported `ERROR(instrument=anchor_not_found)` and refused to report a pass — a negative control that cannot inject its defect is asserting nothing, and it said so instead of going green.
2. Producer detection skipped any line containing `fn `, to avoid matching the declaration. A one-line `fn other() { p.read_results(); }` was therefore **invisible** — a real evasion, found by the test that injects exactly that. Now scoped to `\bfn\s+read_results\b`.

### 7.13.5 What it claims, and what it does not — the column that has to stay honest

**Claims:** no source path in `rust/src` scales a device tick without the period and the valid-bit mask.

**Does not claim, and these are named rather than closed:**

* **That the conversion is arithmetically correct.** A wrong formula inside a sanctioned converter passes this screen untouched. That is `trace.rs`'s unit tests' question. The two arms are complementary and **neither substitutes for the other** — which is the same mistake, in reverse, that created this blind spot.
* **That a value cannot escape.** The screen decides from names. It catches the *first hop* out of a tick-named binding, which is where laundering must start; it does not follow the value further.
* **That an exemption is right.** It guarantees the judgement is written down and pinned to a line, not that the judgement is sound.
* **`#[cfg(test)]` bodies.** Out of frame by design (a test ships nothing), and reported as held out with a line count rather than silently dropped — R12 applied to the screen's own frame.

### 7.13.6 Lane status after this

`lane-checks` remains **`green`**: both new checks have a demonstrated failing arm, executed on every run rather than imagined. `device.op_correctness` and `build.integration_targets` remain **`UNDEMONSTRATED`** and both device lanes therefore remain **`operational`**. They have now run clean for a while, and **running a while is not the falsifier** — a check that has never been red is not yet known to be a check. They stay named, not claimed, until someone produces the mutation.

> **Superseded 2026-08-02 (§7.14.2).** `lane-checks` is now **`operational`**, not `green`: Switch's tautological-assertion screen landed in it and is `UNDEMONSTRATED`. The paragraph above is left standing because it was true when written and because the thing that changed it is the point.

---

## 7.14 Two corrections to this table, both of which demote my own work — added 2026-08-02T00:40-07:00

### 7.14.1 `composed_workflow`: the defect that lives in the union and in neither branch

`ci/test_lane_checks.py::test_the_real_workflow_has_no_unclassified_gate_steps` went red at the `f4ed9ce` merge. My inventory did not classify a step that arrived from Switch's branch:

```yaml
- name: Tautological-assertion screen (no GPU, whole tree)
  run: python ci/check_tautological_assertions.py
```

**Both merges were correct.** My inventory was complete for the workflow I could see; his step was complete on the branch he could see; the workflow they compose is unclassified. This was the **fifth** instance in one day, across four subsystems and three languages — a lock correct until wiring an instrument grew the population needing it; two `device_state.py` fighting over `sys.modules`, each file correct; a signature and its caller correct on their own branches; new code reintroducing a class clippy had just cleared; and this one.

Five is not a coincidence, and the shared property is sharper than "merges are risky":

> Nobody did anything wrong locally in any of the five, and **no command any of those authors could have run would have shown it.** Our discipline verifies branches. The defects live in unions.

**My call, since it is my file:** the lanes should verify the **composed** workflow, not the branch's view of it. `ci/check_lane_inventory.py` takes `--union-with <ref>` and classifies the union of step names from both sides, so a step classified on neither is red *before* the merge. It is wired into `lane-checks` with **`--union-required`**, because a reference it cannot read leaves it silently back in the branch-only view it exists to replace — green for precisely the reason it was written to stop being green for. On a `pull_request` event GitHub already checks out the merge commit and this is belt-and-braces; on a push to a branch it is the only thing that looks at the other side at all.

**It is deliberately not a merge.** A three-way merge can conflict, and a conflict is a different conversation. The union of two name lists answers "does a step exist on either side that nobody has classified" without needing the merge to succeed — and is correct in the case that actually bit us, where both sides touched different regions and merged cleanly.

**Falsified on the real event, replayed.** Using the two actual pre-merge blobs (`.github/workflows/ci.yml` at `0cd6c99` and at `main`) with the tautological entry removed to reconstruct what I knew at the time:

```
branch-only view  -> GREEN
union view        -> ['Tautological-assertion screen (no GPU, whole tree)']
```

The inputs are the real ones; only the clock is wrong. **That is a replay, not a live catch,** and it is recorded as such. Both polarities plus the outage arm also hold on a synthesised two-branch git repository.

**Scope, stated so it is not over-read:** this covers exactly one shape — workflow step names — in exactly one file. **The other four instances of 2026-08-02 remain uncovered.** A general union check is with Trinity.

### 7.14.2 Switch's tautological-assertion screen: `UNDEMONSTRATED`, and so is most of what I called green

Classified from what its own docstring says about itself, which is unusually honest and should be quoted rather than paraphrased:

* **1,056 comparison assertions scanned (rs=614, py=442), 0 detections.**
* **Neither of the two assertion defects that actually occurred here is within its reach.** One compared two *different* expressions that both evaluated to `0.0` — textually the sides differ. The other asserted a predicate true repeatedly and never once false, which is a property of a test *function*, not of a line.
* Scoped by its author to **regression, not discovery**. Its evidence that it works is **entirely planted**.

So it is `UNDEMONSTRATED`, and one `UNDEMONSTRATED` check holds the whole lane: **`lane-checks` is now `operational`.**

**But applying that standard honestly indicts my own work, and the table now says so.** My tick screen's falsifier is *also* planted — I injected bypasses into a scratch copy. So is the layering lint's, the fatal-log check's, and the criterion-10 gate's. Marking Switch's screen down for a planted falsifier while calling mine demonstrated would have been the same failure in a different direction.

The inventory therefore carries a **second axis**, orthogonal to status:

| | meaning |
|---|---|
| `PLANTED` | somebody wrote a defect on purpose and checked the screen caught it. Proves the scanner works **on the shape it was written for**. Says nothing about whether that shape occurs in real code — i.e. does not show the check is load-bearing. |
| `OBSERVED` | the arm was produced by a defect that actually happened, or by the tree as it stands. The check has caught something nobody planted for it. |

Current census, printed next to every lane verdict:

| Lane | planted / with a failing arm | observed |
|---|---|---|
| `build-test-linux` | 6 of 8 | `build.portability_lint`, `build.clippy` — both `RED_NOW` |
| `build-test-windows` | 5 of 7 | the same two |
| `lane-checks` | 4 of 5 | `hostfree.tick_screen_negative_control` |

**Most of what this project calls green rests on planted falsifiers.** The only `OBSERVED` arms outside my negative control are two checks that are *currently red*. A `PLANTED` arm is real evidence and it is the weaker kind, and a table that did not distinguish them was letting the word `green` carry more than it earned.

A planted falsifier does **not** demote a check to `UNDEMONSTRATED` — "somebody performed the mutation" and "nobody ever has" are genuinely different states, and `validate()` now refuses any green check that does not say which it is. It is recorded, surfaced, and left for the reader to weigh.

### 7.14.3 `rust/src/trace.rs` has an owner

Assigned to **Niobe** on 2026-08-02: it is timestamp calibration and trace-event arithmetic — measurement — and she owns the certification instruments that consume it. Tank has the stronger claim on counters and FFI, and the coordinator will move it if either of them thinks it is backwards. Recorded in `.squad/team.md`; `ci/tick_conversion_allowlist.json` now names her instead of `unassigned`.

---

## 7.15 Criterion 11(c): showing `ledger_hits` moves with its input — added 2026-08-02T03:10-07:00

Morpheus's discharge condition (c) is not "show the counter is non-zero." `ledger_hits=6 proven_key_lookups=6` reads identically whether the ledger was genuinely consulted or **derived from the same enumeration that produced the claims**. An identity whose two sides come from one source is a falsifier that cannot fire. (c) is therefore: *show the reading changes when the input changes*, arms asserted to differ, in the census lane rather than behind `#[ignore]`.

### 7.15.1 The arm table (measured, Intel Iris Xe device 1; re-run on RTX 4060 device 0)

The census child was parameterised by model (`ONNXRUNTIME_EP_VULKAN_CENSUS_MODEL`) and by on-disk ledger (`ONNXRUNTIME_EP_VULKAN_LEDGER_FILE`). Nine arms were **probed before anything was asserted** — an assertion written ahead of the reading is a test of the author's guess, not of the mechanism.

| arm | input varied | `proven_key_lookups` | `ledger_hits` | `ledger_gate` | `ledger_miss` | `claimed_nodes` |
|---|---|---|---|---|---|---|
| chain (default) | — | 6 | 6 | ALL-PROVEN | HIT | 6 |
| dtype proven | `mul_f32` | 1 | 1 | ALL-PROVEN | HIT | 1 |
| dtype unproven | `mul_f16_unproven` | 1 | **0** | ALL-DECLINED | KEY-ABSENT | 0 |
| **shape runtime** | dynamic-extent `Mul` f32 | 1 | **0** | ALL-DECLINED | KEY-ABSENT | 0 |
| optin zp | `matmulnbits_f16_scales_zp` | 1 | 1 | ALL-PROVEN | HIT | 1 |
| optin no-zp | `matmulnbits_f16_scales` | 1 | 1 | ALL-PROVEN | HIT | 1 |
| digest identical | `LEDGER_FILE` = real | 1 | 1 | ALL-PROVEN | HIT | 1 |
| digest drifted | `LEDGER_FILE` = truncated | 1 | **0** | **FAULTED** | LEDGER-FAULTED | 0 |
| digest absent | `LEDGER_FILE` = missing | 1 | **0** | **FAULTED** | LEDGER-FAULTED | 0 |

### 7.15.2 Which arm is load-bearing, and one that is not

The **shape-class** arm is the strongest, and it is not the one that was suggested. Same op, same dtype, same optional-input set, same one-node enumeration — the only thing that changes is one component of the proof key (`static` → `runtime-extent`). Because the enumeration is *byte-for-byte the same work* in both arms, a counter derived from the enumeration could not move; this one does, `1 → 0`.

The **MatMulNBits `scales` / `scales+zero_points` pair does not move `ledger_hits`** — both forms are proven, so both read `HIT`. Reporting that pair as an 11(c) mover would be R11's "decomposition that appears to close." What the pair actually establishes is that the two forms are **two distinct keys**, which is asserted on the artifact rather than on the counter: over the six `/`-separated key components (`domain::op / opset / dtypes / variant / shape_class / opt_inputs`), `differing == [2, 5]`. Index 5 is the populated-optional-input set, whose absence produced the 2026-07-30 all-zero-logits defect, so this doubles as that regression's guard.

The **digest-identical arm is the control that makes the other two digest arms detections** rather than a check that rejects everything. It is asserted separately, and its failure is classified `ERROR(instrument)`, not a detection.

### 7.15.3 Both arms of the demonstration

`tests/ops/probe_ledger_mutations.py` breaks exactly one property per mutation and requires each to be caught. All three CAUGHT on **both** devices:

- **M1** — the two dtype arms fed the same model: `arms_must_differ FAILED (control 1, dtype)`.
- **M2** — the identical-file control pointed at a drifted ledger: `the CONTROL arm faulted … so this arm failing makes the whole test ERROR(instrument) rather than a detection`.
- **M3** — the optional-input key component dropped: `the two MatMulNBits ledger keys are identical … One entry would answer for both forms.`

`ledger_lookup` is now in `_MANDATORY_WIRED` and `_KNOWN_UNWIRED_M0` is empty. A ledger that stops being consulted now fails the census lane instead of being excused by a baseline.

### 7.15.4 A defect the 11(c) arms introduced, found by their own witness artifacts

The tracer witness path is **shared across census arms**: each non-injected arm unlinks it and re-arms the tracer. A faulted-ledger arm dispatches nothing *by construction*, so it deletes the previous arm's tracer file and writes none. Because the tracer check reads that path, an arm with no business exercising the tracer could have turned a passing lane into a **false `UNWIRED` purely by running later**. It was invisible in the lane result — every test passed — and visible only as a tracked artifact missing from `git status` after a green run.

Fixed with `_run_counters_child(..., trace=None)`, defaulting to the historical behaviour so no caller on another branch changes meaning (the round-27 `guard` lesson: a required keyword-only argument breaks callers it cannot see). The 11(c) arms pass `trace=False`. Both arms of the demonstration: before, the witness was absent after a green dev1 census; after, it survives on both devices.

## 7.16 Which `model_output_equivalence` is of record — added 2026-08-02T03:10-07:00

`bench/results/phi35-certified-dev0.json` carries `results[0].model_output_equivalence = MATCH` beside `results[0].counters.model_output_equivalence = UNMEASURED`. **These are not two sources disagreeing.** `rust/src/counters.rs` states that the EP has no access to a CPU oracle and that the verdict is written by the Python harness; `to_json()` defaults the field to `UNMEASURED`. The nested value is *a field nobody set*.

**The mechanical tell, requiring no judgement:** `write_equivalence_record` writes the token and `model_output_equivalence_record` **in one call**. A counters document carrying the token with no record beside it was never written by a comparison. In the phi35 artifacts the record key is absent.

The rule therefore keys off **the record's presence, not the token's value**. Keying off `token == "UNMEASURED"` would answer the question by reading the very field whose trustworthiness is in doubt, and would mislabel a genuine comparison that legitimately concluded `UNMEASURED`. Both polarities are asserted in `tests/ops/test_equivalence_authority.py`.

`agreement` is **`UNOBSERVABLE` with a reason**, not `AGREE`/`DISAGREE` (R12): two values one of which nobody wrote have not agreed about anything. The tempting `agreement = (outer == inner)` yields `DISAGREE` here and sends a reader hunting a contradiction that does not exist.

**Why it "went null" between two runs: it did not go anywhere.** The phi35 bench path never calls `write_equivalence_verdict`, so the counters copy has always been the default. The outer verdict is computed by `bench/phi35.py` from the sibling `outputs` evidence. The profile side stayed intact because it has a different writer — which is the informative part: the two attribution sources can differ about whether they observed anything, so the record must say **which witnesses it actually had**.

**Scope (R9 amendment 5 applied to my own check).** Gating every historical bench artifact would leave ~20 frozen, unowned, non-regenerable records permanently red, and a permanently red gate gets loosened or ignored. The **gate** is the certified set only (4 files); the other 13 records print `PRECONDITION(equivalence authority): … Not a gate`. The one thing asserted everywhere: two *genuine* readings that contradict each other is a finding wherever it appears.

Routed to Niobe (owner of `bench/`): `bench/phi35.py` should stamp `model_output_equivalence_authority` at write time so it survives regeneration. The four existing certified artifacts were stamped by hand; her code was not edited.

## 7.17 Criterion 12: the three things the census cannot supply about itself — added 2026-08-02T03:30-07:00

Morpheus's ruling is the one to keep in front of you:

> The census answers whether a mechanism ran; a criterion answers whether a claim is false-able. **A census line can never discharge a criterion.**

`unwired: []` on both devices is a true sentence about a mechanism list. Row 12 asks for three further things, and none of them can be answered from inside the census, because each one needs a fact the census does not own. `ci/check_census_completeness.py` supplies them, with `ci/census_surface_map.json` as its map and `ci/negative_control_census_completeness.py` as its falsifier. **It does not close row 12.** Trinity owns that tally, as she owns criterion 11's, and supplying the artifact and closing the row must not be the same act.

### 7.17.1 The whole: twelve out of *what*

If the denominator comes from the same list that produced the numerator, `12/12` is true by construction, can never fail, and reads as coverage while asserting nothing — R11's hardest kind of wrong, and the shape criterion 11 was refused on. So the whole is enumerated from **production Rust that the census does not write**:

| Source | Owner | Surfaces |
|---|---|---|
| `rust/src/counters.rs` — C ABI counter fields | Tank | 14 |
| `rust/src/trace.rs` — `Phase` variants | Niobe | 10 |
| `rust/src/**` — `ONNXRUNTIME_EP_VULKAN_*` switches | Switch / Mouse / Tank | 26 |
| **total** | | **50** |

Frame, stated rather than assumed: 29,269 production lines read, **11,376 lines held out as `#[cfg(test)]` — UNOBSERVABLE by frame, not zero findings.** One switch (`ALLOW_MISSING_GLSLC`) is named only in prose and never in a line of production code, which the report says out loud.

Against those 50 surfaces the census's twelve mechanisms account for:

| Disposition | Count | Meaning |
|---|---|---|
| `censused` | 33 | a census mechanism observes it |
| **`uncensused`** | **12** | instrumented, and **no** census mechanism observes it |
| `out_of_frame` | 3 | the census's own transport or lane parameter |
| `not_a_mechanism` | 2 | ABI header fields; nothing runs |

**The answer is not 12/12.** The twelve gaps are recorded with owners — `compile_calls`, `compute_calls`, `subgraphs_stub` (Mouse); `DEVICE_MEMORY`, `VA_RESERVE_MIB`, `QUARANTINE_SPANS` (Tank); `GEMV_PACKED`, `CLAIM_LOG`, `CLAIM_DEBUG` (Mouse); `DUMP_OUTPUT_BYTES`, `BACKEND_PROBE`, `VERBOSE` (Switch). `GEMV_PACKED` is the one to look at first: it selects a *different kernel*, and no census line reports whether it was in force.

Numerator and denominator now have different authors, in different files, in a different language. That is the property that lets the count be wrong — and a count that cannot be wrong is not a measurement.

### 7.17.2 Extent: how much of each mechanism the observation covers

| Mechanism | Extent | Not named by the observation |
|---|---|---|
| `partitioner` | 1/2 | `dispatches_executed` |
| `net_benefit_gate` | 0/6 | all six cost-model switches |
| `gpu_tracer` | **1/12** | ten of ten `Phase` variants, and `TRACE_GPU` |
| `retain_viable` | 0/1 | `viable_islands_retained` — its own namesake counter |
| `ledger_lookup` | 4/7 | `unproven_forms_claimed`, `LEDGER_FILE`, `CLAIM_UNPROVEN` |
| `validation_messenger` | 0/3 | `VALIDATE`, `REQUIRE_VALIDATION`, `PLANT_VALIDATION_VIOLATION` |
| `broken_commitment_warn` | 0/2 | `compute_failures`, `FORCE_COMPUTE_FAILURE` |
| `partition_identity_check`, `model_output_equivalence`, `layering_lint`, `device_state_guard`, `instrument_census` | `UNOBSERVABLE` | no surface in the independent whole — the denominator would have to be self-supplied |

Two disciplines here. First, an extent is an **upper bound**: the numerator counts identifiers the observation happens to mention, which is the weakest evidence of coverage that can be checked at all, so 6/6 would mean "named six strings", never "covered six things". Second, the last row reports `UNOBSERVABLE`, **never 0/0** — a ratio of zero over zero presented as coverage is exactly the identity defect this screen exists to refuse (R12).

`gpu_tracer` at 1/12 is the headline. The tracer line reports event counts, one-character event *types* and a distinct-name count; it names none of the ten phases. A phase that never fires is not distinguishable from one that does — and one of those ten is `Record`.

### 7.17.3 Name against content, and why `Phase::Record` would pass

Tank's vocabulary already has the terminal state: `misnamed`. The standing specimen is `Phase::Record` — wired, invoked, correct, input-varying, and **wrong by 50× in what it was called**. A census that verifies a mechanism ran and never asks whether its name describes what it did will certify that specimen.

The decidable form is R10 turned on the census's own output: across the six census artifacts on record, did the observation's *content* ever move?

| State | Mechanisms |
|---|---|
| `VARIES` | `ledger_lookup`, `validation_messenger`, `device_state_guard` |
| `INVARIANT` — certified on presence alone | the other **nine**, including `gpu_tracer`, `partitioner` and `broken_commitment_warn` |
| `UNOBSERVABLE` — fewer than two arms | none in this set |

The third state is the point: with one arm a mechanism is *unmeasured*, not invariant, and calling that invariant would be reporting 0 where the event could not occur. And the honest limit on the second: the arms are the census runs that exist, not arms designed to vary each mechanism, so `INVARIANT` means *no arm on record distinguished it* — weaker than "no arm could", stronger than nothing.

Every mechanism now carries a **name claim** in `census_surface_map.json`: what the name asserts, and the discriminator — the observation that would differ if the name were wrong. All twelve are recorded `name_verified: false`, because none has been. The screen goes red if anyone records one as verified while the arms never varied, and red if a mechanism joins the census with no claim at all.

### 7.17.4 Both arms, demonstrated — `ci/negative_control_census_completeness.py`

Twelve arms, all fired 2026-08-02, every mutation against a **scratch copy**; nothing in the repository is modified.

| Arm | Expected |
|---|---|
| baseline, the real tree | `PASS` |
| a counter field planted in `counters.rs` | `FAIL(unmapped_surface)`, naming it |
| a `Phase` variant planted in `trace.rs` | `FAIL(unmapped_surface)`, naming it |
| an env switch planted in `allocator.rs` | `FAIL(unmapped_surface)`, naming it |
| the map claims a surface that no longer exists | `FAIL(surface_map_rot)` |
| **the census drops a mechanism whose surfaces are still instrumented** | `FAIL(mechanism_not_enumerated)` |
| a mechanism joins the census with no name claim | `FAIL(unclaimed_mechanism_name)` |
| a name recorded verified against arms that never varied | `FAIL(name_claim_contradicted)` |
| map absent / artifacts absent / extractor anchor moved / empty tree | `ERROR(instrument=…)`, exit 4 |

The sixth is the arm the criterion actually needs: **a census that misses a mechanism, caught by the independent whole.** The last row is the one that keeps the screen honest in the other direction — a denominator that silently shrinks is worse than no denominator, so a moved extractor anchor is an outage, never a smaller whole.

### 7.16.5 What this does not claim

- It does not close row 12, and a `PASS` from it does not mean the census covers the tree. A `PASS` means every surface is **accounted for** — 33 observed, 12 recorded as gaps. The gaps are the evidence that criterion 12 is **not met**.
- Its whole is the *instrumented* surface of the EP, not the EP. A mechanism touching no counter, no phase and no switch is as invisible to the denominator as to the census.
- It never decides a name is wrong. `Phase::Record` reads `INVARIANT` here, which is the flag, not the verdict.
- Its own falsifier is `PLANTED` (§7.14.2). Every arm above is a mutation I wrote. Nobody has yet added a mechanism to the Rust tree and been caught by this screen in the field.

### 7.16.6 Lane status after this

`lane-checks` stays **`operational`**, not `green`: `hostfree.tautological_assertions` is still `UNDEMONSTRATED` and one undemonstrated check holds the lane. The two new checks — `hostfree.census_completeness` and `hostfree.census_completeness_negative_control` — are `DEMONSTRATED` / `PLANTED`. The blind-spot table gains `census_denominator`, substitute `DEMONSTRATED`.

---

## 7.17 Two guards the census called `unfalsified`, and why it was right — added 2026-08-02T11:40-07:00

Tank's instrument census failed `test_census_baseline_has_no_drift` with two new unfalsified instruments:

```
tests/ops/_models.py::assert_ep_owns_whole_graph        calls=5 reject_polarity=0 accept_polarity=0
tests/ops/_models.py::assert_no_cpu_fallback_is_live    calls=2 reject_polarity=0 accept_polarity=0
```

This looks wrong, because `test_no_cpu_fallback.py` already calls both in both polarities. It is not wrong. **Every one of those calls sits behind `require_vulkan`, and the screen counts a polarity only from a test that is not GPU-gated** — a polarity nobody can observe without hardware has never been observed on the machines where the census runs. `calls>0` says the guards have callers; `reject=0 accept=0` says nothing has watched them *disagree*, so a guard that always passes, always raises, or has inverted polarity would look exactly like a working one.

`tests/ops/test_no_cpu_fallback_screen.py` supplies the missing polarities in the always-on lane. Both instruments are now `SCREENED` (`reject=2 accept=2`) and the census verdict is `PASS`.

### 7.17.1 The extent of the screen, stated

Only ORT is substituted — `_models.ort.InferenceSession`, plus `_make_session_options` replaced by a recorder so the test can read back which session-config entries the code under test actually set. `_no_cpu_fallback_options`, the key, `ep_only_session_or_refusal`'s three-way classification of ORT's text, the fp64 canary graph, and both guards' own logic are the real code.

**This file cannot tell you that ORT honours the key.** Only the hardware lane can, and it does. What it can tell you is that our side produces two different answers for two different worlds — which is the question `reject=0 accept=0` asked.

### 7.17.2 The trap in screening a falsifier

`assert_no_cpu_fallback_is_live` *is* the falsifier for a silently-swallowed config key, so its self-test must separate "the option takes effect and the check says so" from "the check would say so regardless." A screen built only from the first arm certifies the second. Two arms answer it:

- the recorded entries are asserted to contain `session.disable_cpu_ep_fallback = "1"` — asserted on the recorder, not the return value, because the return value is exactly what a check that never asked would still produce;
- the key is mutated to a typo against a fake that honours only the correct spelling (ORT's measured behaviour: unknown keys are accepted silently) — the 2026-07-30 specimen reproduced without hardware. The fake snapshots the honoured key at construction; a fake that re-read it would honour whatever the test just misspelled and the arm would screen nothing.

### 7.17.3 Both arms of the screen itself

`tests/ops/probe_fallback_screen_mutations.py` mutates `_models.py`, one property per mutation, and requires the screen to catch each. All three CAUGHT:

- **M1** precondition never asks ORT (a guard that always passes) → `DID NOT RAISE CpuFallbackRefused`;
- **M2** refusal raised unconditionally (a guard that rejects everything) → `DID NOT RAISE InstrumentError`;
- **M3** the key is never armed (silently inert) → `ORT created a session … while session.disable_cpu_ep_fallback=1 was set`.

The probe restores `_models.py`, verifies it hashes back to the original, clears `__pycache__` between arms and runs the child with `-B` — R12 generalisation 4 applies to Python: a restored source served from stale bytecode is the same frame error as a stale DLL.

### 7.17.4 Known scope gap, not mine to fix

The census scans `rust/src` and `tests/ops` only — **never `bench/`**. Every instrument under `bench/` is outside its frame, so "the census is clean" currently says nothing about them. Found by Niobe, routed to Tank; recorded here so nobody reads the verdict wider than it reaches.

---

## 7.18 A lost device that exits 0 — added 2026-08-02T17:20-07:00

### 7.18.1 The incident, and why it is not shaped like a failure

Tank, measuring KV bytes at context 512:

```
vkWaitForFences failed: The logical device has been lost
  -> CPU fallback -> EXIT = 0
```

Both of his ctx-512 points were truncated by it, and the consequence is the part worth keeping:

> **Differencing the two truncated points produced an apparent 6.7% KV saving that was an observation ending early.**

Every other gate on this project keys on something a failure changes — an exit code, a raised exception, a verdict token. A runtime device loss changes none of them. `get_providers()` still lists the EP. The harness still writes its artifact. The status is still 0. What changes is that the run is **shorter**, and a shorter run does not read as a failure. **It reads as a smaller number.** That is the same shape as every silent-CPU-fallback incident this project has had, arriving through a new mechanism, and it is the seventh time this class has cost us something.

The **cause** is Switch's (a TDR on Windows; the GQA attention loop's per-token work grows with context). The **reporting defect is mine and survives the cause being fixed**: a lost device that exits 0 is a lane that cannot go red on a class of failure it will encounter again — different kernel, different context, same silence.

### 7.18.2 The signal, and why not the exit code

**The exit status is not one of this check's inputs**, and the check prints that on every run. The defect *is* an exit status of 0, so accepting one as a filter would be accepting the defect as a filter.

Signals in preference order — Tank named the strong ones himself in `bench/results/ctx512_device_lost.txt`:

| # | Signal | Why it ranks here |
|---|---|---|
| 1 | **Structural**: an artifact declared `iters` and observed fewer `compute_calls`; or `uploads == readbacks + 1` | Arithmetic on the producer's own declared expectation. **Needs no text at all** and survives any log-format change, any locale, any vendor. |
| 2 | The EP's device-lost text (`The logical device has been lost`) | Vulkan **specification** language. Stable across vendors and driver versions in a way no ORT-internal wording is. |
| 3 | The EP's `BROKEN COMMITMENT` warning | Ours, so we control it — but it only prints when the EP notices. |
| 4 | ORT's fallback announcement, matched form-tolerantly | Weakest: it is ORT's internal wording and it has already changed under us — see §7.18.5. |

There is **no EP counter for device loss**. A monotonic `device_lost` counter would outrank every signal above, because it is structural *and* produced by the mechanism itself. That is an ask to Tank, not a change I make in his file.

### 7.18.3 Both arms, with the provenance stated

`ci/negative_control_device_loss.py`, 14 arms, all fired 2026-08-02. It counts **LIVE / REPLAYED / PLANTED** separately rather than reporting "14 passed", because those three are worth very different amounts and "we tested it" reads the same either way:

| Provenance | Count | What it evidences |
|---|---|---|
| **LIVE** | 1 | The screen found it on a file it was not written against, unprompted. |
| **REPLAYED** | 3 | A real artifact of a real incident, fed to the screen after the fact. Proves it *would have* caught it; does not prove it fires during a live run. |
| **PLANTED** | 10 | Synthesised to exercise one rule. Proves the rule fires. Proves nothing about whether the rule's event occurs in reality. |

**The red arm on Tank's artifact is REPLAYED, not live.** I did not induce a device loss. Inducing one deliberately is the arm still owed, and it would also answer a question the replay cannot: whether the EP's own text prints at all when the loss is hard enough.

**The LIVE arm is the one to read.** Pointed at the whole artifact tree, the screen found a **second, earlier device loss nobody had reported**:

```
bench/results/trinity-suite-dev1.log:3216
[vulkan-ep] ERROR: vkQueueSubmit failed: The logical device has been lost.
  2026-07-31 10:00:04 — Intel Iris Xe (device 1) — inside tests/ops/test_phi35.py:784
```

It surfaced only as an `AssertionError` and was read as a test failure, not as a lost device. It is **two days earlier than Tank's**, on a **different device**, through a **different call site** (`vkQueueSubmit`, not `vkWaitForFences`). So the class is not one kernel's bug, and the reporting defect outlives whatever Switch fixes. That is the argument for this check made by the check rather than by me.

The green arm is the whole tree today: **284 artifacts read, PASS**, with what it did *not* look at printed in the same breath.

### 7.18.4 Three mechanisms, three extents — never one guarantee

The coordinator's instinct was right and this is the ruling. These are **three mechanisms with different reaches**, and per Morpheus's rule that *two gates whose extents differ compose to the weaker extent and the stronger name*, they must be stated separately and never quoted as "we detect CPU fallback."

| Mechanism | Owner | Sees | Cannot see |
|---|---|---|---|
| `disable_cpu_ep_fallback` | Trinity | **Planned** fallback: ORT refuses at **session creation** when nodes are assigned to CPU | Anything after session creation. A device loss happens on a session ORT has **already accepted**, so this flag is structurally blind to it. |
| `ci/check_fatal_log.py` | Link | **ORT's announcement** of a runtime fallback, in a **captured** log with stderr merged | A loss the EP reports and ORT never announces; any artifact that is not the teed pytest log; and — today — the real line itself (§7.18.5). |
| `ci/check_device_loss.py` | Link | **The loss itself** in any artifact, plus **structural truncation** with no text at all | A loss whose run wrote nothing; truncation in an artifact that declared no expectation (158 of 284 artifacts were undecidable); and three of its five conditions unless the caller **names** the file as one run's evidence. |

That last row's middle column is why a second check exists at all rather than a wider marker list, and the difference is **demonstrated, not argued**: the negative control feeds both checks a log carrying the EP's device-lost line and no ORT announcement. `check_device_loss` is red on it; `check_fatal_log` is green on it. If that arm ever stops firing, the second check has no reach of its own and should be deleted.

### 7.18.5 A finding in the shared vocabulary, routed not patched

While checking whether the existing instrument would have caught Tank's run:

```
tests/ops/_verdict.py::FATAL_LOG_MARKERS =
    ("Falling back to CPUExecutionProvider", "Falling back to CPU")

what ORT actually prints:
    Falling back to ['CPUExecutionProvider'] and retrying.
```

A **list repr**. Neither marker is a substring of it. `_verdict.find_fatal_log_lines()` returns **0 hits** on Tank's artifact, which announces the fallback **twice** — so `ci/check_fatal_log.py` reads that log as clean, and it has been cited as the second witness for five incidents on the strength of a match it cannot make.

`_verdict.py` is Trinity's, and it is **the** vocabulary. So this is reported as its own condition (`marker_list_misses_real_line`) and **not patched here**: a second private marker list in `ci/` is exactly the two-dialect failure the shared vocabulary exists to prevent. The regression test in `ci/test_lane_checks.py` is written to go green when she fixes it.

### 7.18.6 The exclusion list, which is the dangerous part

`bench/results/ctx512_device_lost.txt` **is** a device loss — kept deliberately, as evidence. A directory scan finds it forever, so without an exclusion the check would be permanently red on an incident it did not catch live, and **a check that is always red is a check nobody reads**. `ci/device_loss_incident_records.json` names seven such files. It is an exclusion list, so it is the easiest place on this project to hide a defect. Four rules hold it honest:

1. Every entry carries a **reason, an owner and a date**. An exclusion nobody can review is an exclusion nobody will review.
2. Excluded files are **counted and printed in the frame line of every run**, with their reasons. The check never says "clean" without also saying what it did not look at.
3. An entry naming a file that **no longer exists** is itself a finding (`incident_record_rot`), never a silent no-op.
4. Naming a file **explicitly** on the command line overrides its exclusion, so the red arm can still read them.

Two entries carry work owed rather than history: `kv_bytes_earned-armed.json` and `-default.json` still hold the truncated ctx-512 points under `points`. The roll-up correctly moved them to `rejected_points`; **the per-lane files did not**, so a reader differencing those files gets the ghost 6.7% again. Owed by Tank.

The screen also does **not** count a truncated point the producer already filed under `rejected_*`. Counting it would make the one artifact that did the right thing look like the defective one.

### 7.18.7 What this does not claim

* It says nothing about whether the EP **executed**. A run can be device-loss-free and still be pure CPU output; that is the verdict's job.
* Its structural rule is **UNOBSERVABLE**, not clean, on any artifact that declares no expectation.
* Three of its conditions are **UNOBSERVABLE on a tree scan** by design, because controls on this project emit those texts on purpose; they need `--run-log`.
* It reads artifacts **after the fact**. It cannot stop a run, and it cannot see a loss whose run wrote nothing.

### 7.18.8 `GEMV_PACKED` — investigated, not closed

Of the twelve instrumented Rust surfaces no census mechanism observed, `ONNXRUNTIME_EP_VULKAN_GEMV_PACKED` was the one to take first, because it **selects a different kernel** and so every kernel observation we hold is silent about which kernel it observed.

What the source says: the switch resolves in `rust/src/ops/quant.rs:345` and enters the program as **specialization constant index 5** of `q_gemv.comp`; `rust/src/vk/pipeline.rs` keys its pipeline cache on `(shader_stem, spec_constants)`. So the two settings genuinely are two different pipelines.

What no artifact says: **nothing we produce records a pipeline key or a spec constant.** `trace.rs`'s `Phase` set has `PipelineLookup` with no payload, and no counter names the variant.

So the obvious closure is wrong. A host-side record of the env var is **not** an R10 observation: R10 asks for an artifact the mechanism **produced** whose content varies with its input, and reading `env::var` from a harness proves nothing about what the EP did with it. Trinity's new `flag_frame` mechanism can **disclose** the nine switches, which is real and worth having, but disclosure is not the falsifier. **Closing this requires an EP-side change** — emit the resolved `gemv_packed` value, or the spec-constant vector, into the trace or the counters. Owed by Mouse (`ops/`) with Switch (`vk/pipeline.rs`). Recorded in `ci/census_surface_map.json`.

### 7.18.9 A sixth union defect, caught by my own screen

Merging `main` at `4b5d46b` turned `ci/check_census_completeness.py` red: Trinity had added two census mechanisms (`ep_entrypoints`, `flag_frame`) for which my surface map had **no name claim**. Both branches were complete; the composition was not. Sixth instance of the shape, and the first one caught by an instrument rather than by a person. Name claims added, with `name_verified: false` and a real discriminator each — including the one for `flag_frame` that records why `GEMV_PACKED` cannot have a discrimination arm today.

I have **not** re-dispositioned the twelve gaps on the strength of Trinity's declaration. A mechanism that is declared is not a mechanism that has been observed, and moving a gap to `censused` because the code now names it would be closing a row on a declaration — the exact mistake Morpheus ruled against on criterion 11. They move when an artifact shows them covered.

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
| Android | 68.57% → **~67.33%** (2026-07-30) | **31.43% → ~32.67%** ← decisive, and **moving** |
| macOS (MoltenVK) | 97.5% | 2.5% |
| iOS (MoltenVK) | 100% | 0% |

*Source: [vulkan.gpuinfo.org — VK_KHR_synchronization2](https://vulkan.gpuinfo.org/displayextensiondetail.php?extension=VK_KHR_synchronization2), pulled 2026-07-28*

> **The Android row is a reading with a date, not a constant.** Fact Checker re-sourced it
> on **2026-07-30** at **~67.33% coverage / ~32.67% gap** and rated the earlier 68.57% ❌
> **incorrect *as a stable constant*** — it was correct on 2026-07-28 and is not a
> property of the world. Every quotation of it carries the date. The database is live and
> the number moves with every submitted device report.
>
> **Error direction, both ways at once (§10.0.2):** the gap figure is **simultaneously a
> ceiling and a floor**. A *ceiling* on the legacy-barrier path's usability benefit,
> because some gap devices also fail the §7.2 device gate and would not run this EP with
> or without sync2. A *floor* on the gap population, because gpuinfo.org skews toward
> developer-submitted reports from newer, higher-end hardware and under-represents budget
> Android. It is not a measured usability value in either direction and must never be
> quoted as one. The other four rows in this table were pulled on 2026-07-28 and have
> **not** been re-sourced since; treat them the same way.

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

The measurement **disproved** the "near-universal" assumption for Android sync2: an Android gap of **31.43 points as pulled 2026-07-28, re-sourced to ~32.67 points as of 2026-07-30** and a **12.22-point Windows gap** were measured from the database. The Windows number matters as much as the Android one — nearly one Windows device in eight would be declined. Neither figure is a constant; see §10.0.1 for provenance and §10.0.2 for the two error directions (ceiling on usability benefit, floor on population size).

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
- **`synchronization2` gap:** Switch implements a two-backend barrier abstraction (`vk/barrier.rs`). The legacy `vkCmdPipelineBarrier` path covers the **~32.67% of Android as of 2026-07-30** (31.43% as pulled 2026-07-28 — the figure moves, and it is simultaneously a ceiling and a floor; §10.0.1, §10.0.2) and 12.22% of Windows that lack the extension. See §9 for the CI requirement.
- **`subgroup_size_control` gap:** `VkPhysicalDeviceSubgroupSizeControlProperties` is queried where available to inform workgroup size selection. No device is excluded.
- **macOS:** Fully in scope. MoltenVK reports the extension string; `subgroupSizeControl = VK_FALSE` is acceptable because we require only the properties query.

---

### 8.4 What was considered and rejected

**Requiring `VK_KHR_synchronization2` (the pre-decision proposal):** Excludes **~32.67% of Android as of 2026-07-30** (31.43% as pulled 2026-07-28; a moving snapshot of a live sample, not a constant — §10.0.1) and 12.22% of Windows by measurement. Under the compatibility-first directive, indefensible.

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

**Local-dev lavapipe result (2026-07-30):** `test_barrier_parity.py` ran on WSL Ubuntu 24.04 lavapipe (Mesa 25.2.8, Vulkan 1.4). 29 live ops executed on both sync2 and forced-legacy paths. **58 passed / 0 failed / 28 skipped.** Outputs bit-identical between both paths. This is the third independent implementation (after Intel Iris Xe and RTX 4060 on Windows) to confirm barrier semantic equivalence. CI must replicate this in both CI lanes by adding `test_barrier_parity.py` to the CI test invocation.

**Failure mode this detects:** a bug in `LegacyBackend` — a mismatched stage mask, a missing barrier, an access flag not translated — that causes a different numerical result under the legacy path. This is the most valuable possible failure mode to catch, and it is the failure mode the parity lane was specifically designed for.

**When real Android hardware is available:** the parity run must also be executed on physical devices, including those in the sync2-missing population (see §10). At that point the "forced-legacy = bitwise identical" assertion is retired and replaced by "sync2 backend and legacy backend agree to within the op's tolerance", since the two backends may diverge at the floating-point rounding level on different hardware.

---

## 10. OQ-12: Hardware Validation Experiment

**Status: Pending hardware.** The experiment is fully specified; it can be executed the hour a device exists.

**The question:** Does carrying the legacy barrier backend (DESIGN.md §7.3) actually buy *usable* devices, or does the Adreno 5xx / Mali Bifrost population fail for reasons unrelated to barriers — driver bugs, unsupported memory limits, missing fp16, known Adreno quirks on the watchlist?

### 10.0 Fact-check: OQ-12 figures (2026-07-30T07:05:09-07:00)

> **Checked by:** Fact Checker — Verification mode. Applies R9 (commit `4ff4595`, §10.0.1): for every claim, name the instrument that would go red if the claim were false.

#### 10.0.1 Source and currency of the Android sync2 figure

> ### The canonical form of this figure — quote it this way or not at all
>
> **~32.67% of Android devices in the vulkan.gpuinfo.org sample lacked `VK_KHR_synchronization2` as of 2026-07-30** (coverage ~67.33%), re-sourced by Fact Checker on 2026-07-31. It is **simultaneously a ceiling** on the legacy-path usability benefit — some gap devices fail the §7.2 device gate anyway and the legacy path buys them nothing — **and a floor** on the gap-population size, because the database under-represents budget Android hardware.
>
> The earlier 68.57% / 31.43% pair (pulled 2026-07-28) is rated **❌ incorrect *as a stable constant*.** It was never wrong as a snapshot; it is wrong the moment it is quoted without its date. The figure moved 1.2 points in two days.
>
> **Three obligations on any citation of it, anywhere in this repository:** carry the **date**, carry **both error directions**, and never state it as a bare percentage. A number without its date is a claim that the world stopped.
>
> **And a standing refusal that is not negotiable: lavapipe is not Adreno and is not Mali.** No lavapipe result — not 196 passing tests, not barrier parity 58/0, not a `green` lane — may be cited as Android evidence, in this document or any other. Software rasterisation on x86 shares no driver, no memory system, no cache hierarchy and no OEM blob with the devices this figure is about. The Android column of §5 says **untested**, and it says so until something runs on Android hardware.

**Source (correctly identified in §8.2):** [vulkan.gpuinfo.org](https://vulkan.gpuinfo.org/) — VK_KHR_synchronization2, Sascha Willems, CC-BY 4.0. The §8.2 pull was dated **2026-07-28**. A web query against the same source on 2026-07-30 returned **~67.33%** Android coverage (gap ~32.67%).

> **Re-sourced 2026-07-31 (Fact Checker, recorded here by Link):** the current reading is **~67.33% Android coverage as of 2026-07-30**, gap **~32.67%**. The older 68.57% / 31.43% pair is rated **❌ incorrect *as a stable constant*** — not wrong as a 2026-07-28 snapshot, wrong as a number quoted without its date. Error direction is unchanged and is stated in §10.0.2: the gap figure is **simultaneously a ceiling** (on legacy-path usability benefit — some gap devices fail the §7.2 gate anyway) **and a floor** (on gap-population size — the database under-represents budget hardware). Any citation of this figure must carry its date and both error directions. Same correction applied at §7.7.5 and §8.2.

**Rating: ⚠️ Unverified as a current figure.** The gpuinfo.org page is JavaScript-rendered and cannot be fetched directly; the 67.33% figure comes from a web-indexed rendering, not a live page read. The direction is consistent with what would be expected if budget or legacy devices were submitted to the database in the interval: the coverage decreased (more sync2-lacking devices entered the sample), meaning the gap *grew* by roughly 1.2 points in two days. The exact current value cannot be confirmed without direct page access.

**The finding that matters:** the figure IS moving. The database is live; it changes as developers submit device reports. A number pulled 2026-07-28 is not identical to the number on 2026-07-30 or in six months. The correct posture is: **treat the gpuinfo.org figure as a lower bound on the sync2-lacking fraction of the real Android installed base** (the database over-represents newer/higher-end hardware), and expect it to drift as submissions accumulate. Do not treat 31.43% as a fixed constant — it is a snapshot of a live sample.

**Falsifiability instrument:** A direct read of `https://vulkan.gpuinfo.org/displayextensiondetail.php?extension=VK_KHR_synchronization2&platform=android` on any given date yields the current coverage percentage. If this value ever crosses 99%, the gap has closed to the point where the legacy-path decision can be revisited. Until then the figure is bounded but not pinned.

#### 10.0.2 Error direction: is the gap figure the right reading?

The §8.2 measurement already correctly defines coverage as "devices that expose the extension string **or** report Vulkan 1.3 (where sync2 is core)." The 1.3 promotion is therefore already captured; there is no undercounting from devices that support sync2 natively without the extension string. This part of the claim is sound.

However, "~32.67% of Android devices lack sync2 (2026-07-30)" and "~32.67% of Android devices are reachable via the legacy barrier path" are not the same claim. Two error directions exist:

**Overcounting the benefit (ceiling):** Devices that lack sync2 may *also* fail the §7.2 device gate (Vulkan < 1.1, no compute queue, insufficient `maxComputeWorkGroupInvocations`, etc.). Those devices are rejected before the barrier backend is even selected; the legacy path buys them nothing. The gpuinfo.org figure is silent on whether sync2-lacking devices pass §7.2. **The gap figure is therefore a ceiling on the legacy-path benefit, not a measured value.** How much below it the real benefit falls cannot be determined without the OQ-12 experiment.

**Undercounting the real population (floor):** The database skews toward developer-submitted reports from newer and higher-end hardware. Budget Android devices — which are exactly the ones most likely to lack sync2 and to be running obsolete OEM blobs — are under-represented in the submission pool. The real installed-base fraction lacking sync2 is likely *higher* than the database figure of the day, not lower. This means the legacy path potentially benefits a larger fraction of real users than the number suggests, but the usability of those devices is the unknown.

**Net position per R9:** the gap figure — ~32.67% as of 2026-07-30 — is simultaneously a ceiling on usability-benefit (some gap devices fail the gate) and a lower bound on the gap-population size (database skew). The two errors partially offset but the direction cannot be resolved without the experiment. **A single gpuinfo.org reading names a database-sample fraction, not a device-market fraction, not a usability fraction.** The legacy path is justified by the existence of a non-negligible gap population, not by the precision of this number.

#### 10.0.3 Conditions required to drop the legacy barrier path

The dual-backend architecture (DESIGN.md §7.3) exists to serve two independent gaps:

| Gap | Current figure — **always with its date; never a constant** | Drop condition |
|---|---|---|
| Android sync2 coverage | ~67.33% (gap: ~32.67%) as of 2026-07-30; 2026-07-28 pull: 68.57% (gap: 31.43%); see §10.0.1 — figure is moving | Database coverage ≥ 99% on Android **and** OQ-12 confirms gap devices fail §7.2 for other reasons |
| Windows sync2 coverage | 87.78% (gap: 12.22%) | Database coverage ≥ 99% on Windows |

**Both conditions must hold simultaneously to justify removing the legacy path.** Android coverage at 99% does not close the Windows gap; Windows coverage at 99% does not close the Android gap. Neither is currently close.

There is a weaker sufficient condition: if OQ-12 (Stage 1) shows that **all** sync2-lacking devices in slots A and B also fail the §7.2 gate, the legacy path buys no Android devices. Even in that case, the Windows gap (12.22%) independently justifies the dual-backend architecture unless Windows coverage also reaches near-universality.

**Public data as of 2026-07-30:** the legacy barrier path is justified. The conditions for removal are not met on either platform.

**Falsifiability instrument:** Repeat the gpuinfo.org read monthly. If Android coverage crosses 99% on the database *and* OQ-12 Stage 1 yields all-fail for sync2-lacking devices, the Android justification is void. If the database coverage for Windows crosses 99%, the Windows justification is void. If both happen, the legacy path becomes a pure maintenance cost with no benefit and the decision can be revisited.

---

**How much of the gap claim is currently unverified:** All of it, as a *usability* claim. The gpuinfo.org data proves those devices lack `VK_KHR_synchronization2`. It says nothing about whether they can run correct compute at all, whether they pass the §7.2 device gate, or whether Vulkan inference on them outperforms their own CPU. Until the experiment runs, every statement about "the legacy backend benefits the 31% Android population" is a database extrapolation, not a measurement.

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

