# Fact Checker Audit Trail

> Append-only evidence log. Entries are succinct — verdict + citation, never raw source material.

<!-- Fact Checker appends findings below -->

---

## Audit Entry — Vulkan Baseline Verification
**Date:** 2026-07-28T17:59:54-07:00
**Requested by:** Justin Chu (coordinator)
**Scope:** Claims supporting the Vulkan 1.3 baseline decision for onnxruntime-ep-vulkan

### Claim 1 — llama.cpp Vulkan 1.3 baseline
**Verdict:** ❌ Contradicted

Primary source: `ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp` (ggml-org/llama.cpp, SHA 3e6b395):
```cpp
std::string target_env = (name.find("_cm2") != std::string::npos) 
    ? "--target-env=vulkan1.3" 
    : "--target-env=vulkan1.2";
```
The base shaders target **Vulkan 1.2**. Only cooperative-matrix-2 (`_cm2`) shaders — a NVIDIA Ampere+/Ada-specific optimisation path — target Vulkan 1.3. The CMakeLists.txt test-shader invocations use `--target-env=vulkan1.3` only for probing extension availability, not as the default.

Also confirmed: ExecuTorch `Runtime.cpp` sets `VK_API_VERSION_1_1` at instance creation (pytorch/executorch, SHA 8001512).

### Claim 2 — ExecuTorch Vulkan 1.3 baseline
**Verdict:** ❌ Contradicted

Primary source: `backends/vulkan/runtime/vk_api/Runtime.cpp` (pytorch/executorch, SHA 8001512):
```cpp
VK_API_VERSION_1_1, // apiVersion
```
ExecuTorch targets **Vulkan 1.1**. Device version check in `Device.cpp`: feature queries branch at `>= VK_API_VERSION_1_1`. `Allocator.cpp` initialises VMA with `VK_API_VERSION_1_0` as the VMA vulkanApiVersion. Image-based tensor storage confirmed: official ExecuTorch docs state tensors are stored as Vulkan images.

### Claim 3 — MoltenVK Vulkan 1.3 support
**Verdict:** ✅ Verified (with caveats)

MoltenVK 1.3.0 released (2025); advertises Vulkan 1.3 on macOS/iOS/tvOS/visionOS via Metal.
Sources: phoronix.com/news/MoltenVK-1.3-Released, khronos.org permalink.
Notable compute gaps for inference workloads:
- `VK_KHR_buffer_device_address`: partial/emulated, not full parity
- Descriptor indexing / bindless: limited by Metal resource model
- Some indirect dispatch forms unsupported
- `VK_KHR_portability_subset` is always advertised; callers MUST enumerate it

### Claim 4 — Android Vulkan 1.3 availability
**Verdict:** ⚠️ Unverified (plausible but not fully confirmed from authoritative primary source)

Best available data (Android Distribution Dashboard, Nov 2025, via web search):
- Vulkan 1.3: ~26% of active handheld Android devices
- Vulkan 1.1: ~62%
- No Vulkan: ~7.4%
Vulkan 1.3 is standard on 2022+ mid/high-end Snapdragon and recent ARM Mali. Budget and legacy devices lag.
Android CDD does not mandate Vulkan 1.3 for any API level as of Android 15.
Note: this figure is from web-search-aggregated data, not a direct read of the Android distribution dashboard.

### Claim 5 — lavapipe and SwiftShader Vulkan 1.3
**Verdict:** ✅ Verified

Web search confirms:
- lavapipe (Mesa 24.x+, 2024): Vulkan 1.3 supported
- SwiftShader (Google, 2024): Vulkan 1.3 supported
Both are suitable for GPU-less CI. Neither provides hardware-parity performance but both pass Vulkan 1.3 API conformance. Confirmed adequate for headless CI.

### Claim 6 — ORT Plugin EP C API status
**Verdict:** ✅ Verified (experimental, introduced ORT 1.22/1.23)

Primary source: `include/onnxruntime/core/session/onnxruntime_ep_c_api.h` (microsoft/onnxruntime, SHA 6835283):
- Functions tagged `\since Version 1.22` and `\since Version 1.23` (1.24 for newer additions)
- Entry point: plugin shared library exports `CreateEpFactories()` → returns `OrtEpFactory*[]`
- `OrtEpFactory` provides: `GetSupportedDevices`, `CreateEp`, `CreateAllocator`, `CreateDataTransfer`
- Still marked experimental; backward compatibility not guaranteed for all functions
- Qualcomm QNN EP is currently the first production-grade user of this API (announced 2026-05)

### Claim 7 — Existing Vulkan EP / Rust bindings for ORT Plugin EP
**Verdict:** ✅ Verified (no Vulkan EP exists; no Rust plugin-EP crate exists)

- No official Microsoft Vulkan EP for ORT. Feature request open: github.com/microsoft/onnxruntime/issues/21917
- WebGPU EP (preview) uses Vulkan internally on Linux/Android, but is NOT a selectable "Vulkan EP"
- No mature community/third-party Vulkan EP plugin library found
- Rust bindings: `ort` crate wraps built-in EPs only; `onnxruntime-sys` has raw FFI but no plugin-EP glue. Plugin EP would require manual FFI via `libloading` + raw C ABI. No crate for `OrtEpFactory` as of mid-2026.
