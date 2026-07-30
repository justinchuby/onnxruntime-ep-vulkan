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

### 2026-07-28T22:28:08-07:00 — CI Vulkan loader failure: confirmed root causes

**CONFIRMED ROOT CAUSE — Windows `ERROR_INCOMPATIBLE_DRIVER`:**

The LunarG Vulkan loader 1.3+ silently ignores `VK_ICD_FILENAMES`, `VK_DRIVER_FILES`, and `VK_ADD_DRIVER_FILES` when the calling process is running with elevated privileges. This is an explicit security restriction documented in the Khronos Loader source (LoaderDriverInterface.md v1.3.274): *"For security reasons, VK_ICD_FILENAMES, VK_DRIVER_FILES, and VK_ADD_DRIVER_FILES are all ignored if running the Vulkan application with elevated privileges."* GitHub Actions Windows runners run as `runneradmin` (Administrators group, UAC disabled). The loader falls back to registry-based ICD discovery; Mesa lavapipe is not registered there; no ICD found → `ERROR_INCOMPATIBLE_DRIVER`. Fix: register `mesa3d\x64\lvp_icd.x86_64.json` in `HKLM:\SOFTWARE\Khronos\Vulkan\Drivers` (requires admin rights, which the runner has). Setting `VK_DRIVER_FILES` instead of `VK_ICD_FILENAMES` does NOT fix this — it is subject to the same restriction.

*Sources verified: LoaderDriverInterface.md (Khronos), actions/runner-images #6557, run 30450284838 CI log.*

**CONFIRMED ROOT CAUSE — Linux Clippy failure (NOT a Vulkan issue):**

`tests/mock_ort/mod.rs` uses `ort::wchar_t` unconditionally. This type is only generated by bindgen on Windows (`OrtChar = wchar_t`); on Linux `OrtChar = char` and bindgen emits `c_char`. The file was authored on Windows and never compiled on Linux. Fix is in Tank's code. Trinity cannot fix this.

*Source: run 30450284838 Clippy step output.*

**OPERATIONAL RULE — reading CI env vars is not enough:**

When investigating a CI failure involving environment variables, check whether the loader/runtime has a privilege-based override for those variables before assuming they take effect. The LunarG loader (1.3+), Vulkan validation layers, and similar tools all have security-gating logic that makes env var setup invisible when running as Administrator. The CI log shows `VK_ICD_FILENAMES` correctly set and the ICD JSON correctly found; only the loader's elevation check explains why that information was ignored.

**OPERATIONAL RULE — ICD registration vs env var:**

On Windows, `VK_ICD_FILENAMES` and `VK_DRIVER_FILES` are correct for non-elevated development. For CI (elevated), the correct mechanism is always the registry: `HKLM:\SOFTWARE\Khronos\Vulkan\Drivers` for ICDs, `HKLM:\SOFTWARE\Khronos\Vulkan\ExplicitLayers` for explicit layers. The LunarG SDK installer already uses the registry for its own validation layer; CI setup for third-party ICDs must do the same.

**OPEN ITEM — Linux secondary lavapipe failure:**

The lavapipe smoke-check fires a warning (non-blocking) after LunarG upgrades `libvulkan1` to 1.3.296.0~rc1. Root cause is UNVERIFIED because it is masked by the compile error. Must re-evaluate after Tank's fix lands. Package state confirmed: Mesa 23.2.1, LunarG 1.3.296, validation layers 1.3.296. The ICD path `/usr/share/vulkan/icd.d/lvp_icd.x86_64.json` is installed by Ubuntu's `mesa-vulkan-drivers` (not the LunarG repo); on Linux the runner is non-elevated so `VK_ICD_FILENAMES` is NOT ignored. The failure is something else — most likely a library resolution or layer interaction issue with the upgraded loader. Not diagnosable until builds succeed.

### 2026-07-29T09:19:35-07:00 — First execution-derived hardware data; lavapipe LVP2 quirk; Intel as oracle

**Both CI lanes now working.** Linux: `deviceName = llvmpipe (LLVM 15.0.7, 256 bits)`, `apiVersion = 1.3.255`. Windows: lavapipe enumerated after Trinity registered ICD in `HKLM:\SOFTWARE\Khronos\Vulkan\Drivers`. The elevation-based env-var bypass was the Windows root cause; registry registration is the permanent correct mechanism.

**NEW QUIRK — LVP2 (lavapipe, CI-observed):** `VkPhysicalDeviceSubgroupProperties::supportedStages = 0` on lavapipe. Subgroup operations are not emulated in software. Switch removed subgroup BASIC from the §7.2 device gate so lavapipe passes. Any code path using subgroup operations must check `supportedStages` and degrade to scalar — this is correct behavior for any conformant Vulkan 1.1 device, not a lavapipe-specific workaround. LVP1 (subgroupSize=1 on old lavapipe) is superseded by the more precise LVP2.

**Local hardware probed (Justin's machine, `epctl --probe-loader`, 2026-07-29):**
- Intel Iris Xe Graphics: Vulkan 1.4.309, gate PASS, UMA (DEVICE_LOCAL+HOST_VISIBLE same type), maxComputeWorkGroupInvocations=1024, maxComputeSharedMemorySize=32768
- NVIDIA RTX 4060 Laptop: Vulkan 1.4.325, gate PASS, discrete (DEVICE_LOCAL heap 0 ≠ HOST_VISIBLE type 2), maxComputeSharedMemorySize=49152

**UMA vs discrete is now a first-class platform distinction.** Intel Xe is UMA — same physical memory for CPU and GPU. RTX 4060 is discrete — separate VRAM and system RAM. Adreno and Mali are also UMA. The Intel Xe on Justin's desk is the closest available proxy for mobile memory behavior.

**Intel as spec-conformance oracle.** Intel's Vulkan implementation is the strictest of major desktop vendors (owner-stated, Justin Chu 2026-07-29; consistent with industry knowledge). Rule: if a shader is correct on the RTX 4060 but wrong on the Iris Xe, the problem is an EP spec-compliance bug, not an Intel quirk. Do not special-case Intel; doing so masks portability problems that will reappear on MoltenVK and strict Android drivers. Both devices must be used in every local development loop.

**Vulkan 1.4 in the wild.** Both desktop devices report Vulkan 1.4 (not 1.3). The §1 baseline discussion was conservative. DESIGN.md §7's device gate of ≥ 1.1 is unaffected — but the 1.3-or-bust framing of §1 is more outdated than it appeared.

**OQ-12 unchanged.** Desktop GPUs do not touch the Adreno 5xx / Mali Bifrost question. All Android rows remain untested.

### 2026-07-29T09:39:59-07:00 — Standing directive: cross-platform generality is structural, not a review step

**Source:** `.squad/decisions/inbox/copilot-directive-cross-platform.md` (Justin Chu, via Copilot coordinator)

**The rules, stated concisely so they are used rather than remembered:**

1. **Derive from reported limits, never from observed constants.** Workgroup sizes, tile shapes, shared-memory budgets must come from the device's own `VkPhysicalDeviceLimits`, not from a constant that happens to fit the RTX 4060's 48 KiB. A constant that fits 48 KiB will silently fail on Iris Xe (32 KiB) and mobile (often 16–32 KiB).

2. **UMA is the mobile case, and we have one on the desk.** Intel Iris Xe, all Adreno, all Mali, all Xclipse (SoC) are UMA — `DEVICE_LOCAL` and `HOST_VISIBLE` may be the same physical type. A staging path that assumes a discrete upload heap silently skips on all of these. The Iris Xe is the only local device that exercises this path; the RTX 4060 never does.

3. **Intel is the spec oracle (repeated because it must not be forgotten).** Correct on RTX 4060 and wrong on Iris Xe = EP relying on undefined/unspecified behavior, not an Intel bug. Never fix this by special-casing Intel. Intel failures predict MoltenVK and strict Android driver failures.

4. **Every `cfg` is a portability hazard — `tests/portability.rs` enforces it.** A target-conditional binding may only be named by a `cfg`-gated definition. The `ort::wchar_t` incident (Windows-only bindgen type used unconditionally, silently broke Linux lane) is the canonical example of what this rule prevents.

5. **§9.1.2 discipline.** A result on this desk is not a result this project has. CI proves portability; physical Android and macOS coverage is absent (OQ-12). The three verification tiers — CI-verified, local-dev-verified, untested — must be kept distinct in PLATFORMS.md and never blurred.

**What changed in PLATFORMS.md as a result:**
- Document preamble now states the standing directive and its five rules explicitly.
- §5 matrix table now has a `Memory` column (UMA / Discrete / N/A) as a first-class property in every row. Reading the table no longer requires inferring memory architecture from prose.
- §5.1 rewritten to explain the UMA column and its consequences, rather than being a post-table footnote.
- Intel iGPU rows split from Intel Arc rows so UMA/Discrete is visible at row level.

### 2026-07-29T20:26:56-07:00 — LVP2 retracted; instrument-failure discipline; lavapipe actually supports subgroup ops

**LVP2 WAS AN INSTRUMENT FAILURE — RETRACTED.**

The "observed in CI" reading of `VkPhysicalDeviceSubgroupProperties::supportedStages = 0` on Mesa
lavapipe was caused by the `ash` `push_next` / `#[must_use]` bug (Switch's `caps.rs`): the
`push_next` method returns `&mut Self` and discarding that return value silently discards the entire
extension chain. The driver received a Properties2 call with no extension structs; every chained
struct read back as its zero-initialized default. The reading was taken before Switch fixed this.

**Corrected reading (fixed probe, CI run 2026-07-29T20:26:56-07:00, Linux lane, Mesa 23.2.1):**

```
subgroup_probe_valid      : true
subgroup_size             : 8
subgroup_stages_raw       : FRAGMENT | COMPUTE | TASK_EXT | MESH_EXT
subgroup_basic_in_compute : true
subgroup_ops              : BASIC | VOTE | ARITHMETIC | BALLOT | SHUFFLE | SHUFFLE_RELATIVE | QUAD
is_uma                    : true
```

Mesa 23.2.1 lavapipe supports subgroup BASIC — and arithmetic, ballot, shuffle, and quad — in
compute. LVP2 removed from the watchlist; retraction documented in PLATFORMS.md §6.3. The device
gate removal of subgroup BASIC (DESIGN.md §7.2) stands, justified by §7.0 not by lavapipe.

**Secondary finding from re-observation:** `is_uma=true` on lavapipe. The UMA memory path is now
exercised in CI (both lanes). This is a useful structural property — the UMA code path has
software-rasterizer CI coverage even before Android hardware.

**Also notable:** CI now exercises the subgroup arithmetic shader path in software via lavapipe.
The false LVP2 reading had suggested CI validated only a scalar fallback; the correct reading shows
CI validates subgroup arithmetic too (in software emulation).

**INSTRUMENT-FAILURE DISCIPLINE — permanent rule:**

> A number taken with a broken instrument is not evidence merely because it was written down.

When any probe or measurement tool is known to have had a bug:
1. Identify every reading sourced from that probe.
2. Re-observe each one with the fixed probe before citing it.
3. If re-observation contradicts the original, retract the entry explicitly and completely — do not
   leave a correction note as a footnote on a false claim (Morpheus's R6 rule: a correction must
   propagate through every document that cited the false reading, not stop at the file where it was
   noticed).
4. The `subgroup_probe_valid: true` flag from Switch's three-state probe is the signal that a
   Properties2 reading is trustworthy.

**Audit scope of the push_next bug:**
- All entries in §6.3 other than LVP2 are sourced from external documentation — unaffected.
- Gate PASS data and base device limits (maxComputeWorkGroupInvocations, shared memory size, memory
  types) for local hardware come from base `VkPhysicalDeviceProperties`, not a chained struct —
  unaffected.
- fp16/int8 feature readings for all devices may have been affected (VkPhysicalDeviceFeatures2 also
  uses push_next chaining); marked provisional in §5 pending re-observation.
- Only LVP2 was an entry in the watchlist sourced from a broken chained probe.


📌 **Windows ICD mechanism: registry registration, not env var (2026-07-29, Link + Trinity):** For non-elevated processes, `VK_ICD_FILENAMES` / `VK_DRIVER_FILES` work. For elevated processes (CI runner = `runneradmin`), the LunarG loader ignores those env vars — ICD must be registered in `HKLM:\SOFTWARE\Khronos\Vulkan\Drivers`. Future CI setup for any Windows Vulkan resource must use the registry, not the env var.

📌 **Lavapipe on Linux CI now functional (2026-07-29, Trinity + Link):** glslc installed from LunarG apt repo (`shaderc` package, not Ubuntu's `glslang-tools`). `VK_LAYER_KHRONOS_validation` enabled; zero validation errors enforced. The lavapipe smoke-check warning under upgraded loader is UNVERIFIED — recheck after Tank's `ort::wchar_t` fix lands and Linux builds succeed.

📌 **Local GPU hardware: Intel Iris Xe (Vulkan 1.4.309, UMA, 32 KiB) and RTX 4060 Laptop GPU (Vulkan 1.4.325, discrete, 48 KiB) (2026-07-29):** Both pass §7.2 gate. Intel is stricter — use it as spec oracle. SDK at `C:\VulkanSDK\1.4.350.0` — not on default PATH; must be prefixed explicitly in CI steps.
