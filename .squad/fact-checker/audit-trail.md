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

---

## Audit Entry — ORT 1.28 API Verification (CreateExternalResourceImporter)
**Date:** 2026-07-28T18:51:35-07:00
**Requested by:** Justin Chu / Coordinator (second pass)
**Scope:** ORT 1.28 release existence + `CreateExternalResourceImporterForDeviceImpl` symbol + implications for OQ-3

### Claim 1 — ORT 1.28 exists and is a stable release
**Verdict:** ✅ Verified

ORT v1.28.0 released **2026-07-24** as a stable release. Confirmed from GitHub releases page and PyPI. Not a pre-release. Major breaking changes: ONNX 1.22.0 and protobuf 6.33.5 upgrade.

### Claim 2 — `CreateExternalResourceImporterForDeviceImpl` exists as a public API
**Verdict:** ❌ Contradicted (name is wrong; API exists under a different, corrected name)

The exact symbol `CreateExternalResourceImporterForDeviceImpl` does NOT exist as a public API. It appears only as a local implementation function name in test/example code (`onnxruntime/test/autoep/library/example_plugin_ep/ep_factory.h`).

Real public names — two sides of the same feature:

EP side (OrtEpFactory, since ORT 1.24):
  `ORT_API2_STATUS(CreateExternalResourceImporterForDevice, _In_ OrtEpFactory* this_ptr, _In_ const OrtEpDevice* ep_device, _Outptr_result_maybenull_ OrtExternalResourceImporterImpl** out_importer);`

Caller side (OrtInteropApi, since ORT 1.24):
  `ORT_API_STATUS_IMPL(CreateExternalResourceImporterForDevice, _In_ const OrtEpDevice* ep_device, _Outptr_result_maybenull_ OrtExternalResourceImporter** out_importer);`

Sources: include/onnxruntime/core/session/onnxruntime_ep_c_api.h (SHA 6835283, `\since Version 1.24.`); onnxruntime/core/session/interop_api.h (SHA 92c8873); ep_factory_internal.cc (SHA fb8a90f, code comment "added in ORT 1.24").

**Added in ORT 1.24, NOT 1.28.**

### Claim 3 — What does it do?
**Verdict:** ✅ Verified (fully characterized)

(a) API surface: Part of plugin EP C API. OrtEpFactory implements it; callers use OrtInteropApi. Full pipeline (all since 1.24):
  CreateExternalResourceImporterForDevice → CanImportMemory → ImportMemory → CreateTensorFromMemory
  CanImportSemaphore → ImportSemaphore → WaitSemaphore / SignalSemaphore
  ReleaseExternalMemoryHandle / ReleaseExternalSemaphoreHandle / ReleaseExternalResourceImporter

(b) Imports externally-allocated device resources without host round-trip:
Vulkan memory handle types confirmed in nv_vulkan_test.cc (SHA 918e137):
  ORT_EXTERNAL_MEMORY_HANDLE_TYPE_VK_MEMORY_WIN32 (Windows, HANDLE from vkGetMemoryWin32HandleKHR)
  ORT_EXTERNAL_MEMORY_HANDLE_TYPE_VK_MEMORY_OPAQUE_FD (Linux, fd from vkGetMemoryFdKHR)
  ORT_EXTERNAL_MEMORY_HANDLE_TYPE_DMABUF_FD (Linux DMA-BUF)
  ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE / _D3D12_HEAP (from NV TRT-RTX EP)
No host copy occurs.

(c) Ownership: Caller allocates VkDeviceMemory with VkExportMemoryAllocateInfo, exports via OS handle, hands it to ImportMemory. EP creates derived OrtExternalMemoryHandle. Caller calls ReleaseExternalMemoryHandle when done. CreateTensorFromMemory creates a view — does not take ownership of the memory handle.

(d) In-tree users: NV TensorRT RTX EP (nv_provider_factory.cc SHA 23a5378) is the primary user — supports D3D12 + Vulkan memory import on Windows/Linux. Example plugin EP has a minimal D3D12-only demo.

### Claim 4 — Relevance to OQ-3 and zero-copy IO binding
**Verdict:** Nuanced

OQ-3 (allocator pointer problem): NOT resolved. OrtExternalResourceImporter is caller-driven, not EP-allocator-driven. Our provisional answer (opaque-handle registry or BDA) remains correct for ORT-managed tensor allocation.

Zero-copy IO binding (caller-owned Vulkan buffers): FULLY ADDRESSED. Callers who allocated VkDeviceMemory with export flags can: export → ImportMemory → CreateTensorFromMemory → bind as graph I/O with no host copy. This is tested by the NV Vulkan test. Critical constraint: VkDeviceMemory MUST have been allocated with VkExportMemoryAllocateInfo (export flags set at allocation time). Cannot retrofit existing non-exported allocations.

Our EP must implement OrtEpFactory::CreateExternalResourceImporterForDevice to enable this path. The importer then does: OS handle → vkImportMemoryWin32HandleKHR / vkImportMemoryFdKHR → VkDeviceMemory → our internal buffer wrapper.

### Claim 5 — ORT 1.28 changes affecting plugin EPs
**Verdict:** ✅ Verified

Critical bug fix: "Fixed a null allocator passed to plugin EP kernel PrePack, and plugin EP allocator deleter lifetime" — would have hit us in 1.27.
New features: Model Package Phase 2 (schema versioning), crypto/IO callbacks for EPs, name-based partitioning, Linux NPU sysfs discovery.
OrtModelPackageApi moved to experimental C API — experimental surface growing.
Plugin EP API status: STILL experimental as of 1.28. No graduation from experimental noted in release notes.
OrtEpFactory vtable: no breaking signature changes noted in 1.28.

### Claim 6 — ORT version to pin
**Verdict:** Advisory

Minimum for CreateExternalResourceImporter: ORT 1.24.
Minimum to avoid known plugin EP allocator bugs: ORT 1.28.
Recommendation: compile and ship against ORT 1.28; declare minimum ORT 1.24+ in documentation; use ort_version_supported field for capability gating; isolate FFI behind abstraction layer to contain future vtable additions.
