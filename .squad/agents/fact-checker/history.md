# Project Context

- **Project:** onnxruntime-ep-vulkan
- **Created:** 2026-07-29

## Core Context

Agent Fact Checker initialized and ready for work.

## Recent Updates

📌 Team initialized on 2026-07-29

## Learnings

Initial setup complete.

---

## Audit: ORT 1.28 API Verification — 2026-07-28T18:51:35-07:00

**Task:** Verify Justin's claim that ORT 1.28 exists and exposes `CreateExternalResourceImporterForDeviceImpl`.

**Key learnings:**

1. **ORT 1.28 is real and stable** — released July 24, 2026. Not a pre-release.

2. **The exact symbol name `CreateExternalResourceImporterForDeviceImpl` does not exist as a public API.** The `Impl` suffix is used only in test/example code for a local static function. The real public name is `CreateExternalResourceImporterForDevice` — both in `OrtEpFactory` (EP side) and `OrtInteropApi` (caller side).

3. **This API was added in ORT 1.24, not 1.28.** The code comment `"OrtEpFactory::CreateExternalResourceImporterForDevice was added in ORT 1.24"` in `interop_api.cc` and `ep_factory_provider_bridge.h` is authoritative.

4. **Vulkan-specific memory handle types are already defined:** `ORT_EXTERNAL_MEMORY_HANDLE_TYPE_VK_MEMORY_WIN32` (Windows) and `ORT_EXTERNAL_MEMORY_HANDLE_TYPE_VK_MEMORY_OPAQUE_FD` (Linux). These export `VkDeviceMemory` via OS-level handles, NOT raw VkBuffer pointers.

5. **The API does NOT solve OQ-3 (allocator pointer ABI).** It is orthogonal — it is for callers to import their own Vulkan memory into ORT, not for our EP's allocator to satisfy ORT's pointer-based `Alloc()`.

6. **The API IS the answer for zero-copy IO binding** — callers with externally-allocated `VkDeviceMemory` (with export flags) can import it as a tensor without a host copy. The NV TensorRT RTX EP uses this for Vulkan↔CUDA interop.

7. **ORT 1.28 includes important plugin EP bug fixes** (null allocator in PrePack, allocator deleter lifetime). Pin to 1.28 for development.

8. **Plugin EP API is still experimental in 1.28.** No stability guarantee.

**Methodology notes:**
- GitHub code search worked well for finding the 15 files containing the real symbol name.
- The `ort_version_supported < 24` guard in `ep_factory_provider_bridge.h` was the decisive evidence for the ORT 1.24 claim.
- Used PowerShell grep on downloaded temp files to extract Vulkan-specific handle_type usage from `nv_vulkan_test.cc`.
- ORT 1.28 release notes fetched directly from GitHub releases page.

---

## Audit: Vulkan Baseline Verification — 2026-07-28T17:59:54-07:00

**Task:** Verify claims underlying Justin's Vulkan 1.3 baseline proposal.

**Key learnings:**

1. **llama.cpp targets Vulkan 1.2 by default, not 1.3.** The popular claim is inaccurate. Only the `_cm2` (cooperative matrix 2) shaders target `vulkan1.3`. Source: `vulkan-shaders-gen.cpp`. This was verified by fetching the actual source from GitHub — GitHub code search could not index it (file too large at 987KB).

2. **ExecuTorch targets Vulkan 1.1.** Confirmed from `Runtime.cpp` (`VK_API_VERSION_1_1`). Their VMA is initialized at `VK_API_VERSION_1_0`. They do use VMA and image-based tensors.

3. **MoltenVK 1.3 is real** but has compute portability caveats (buffer device address, descriptor indexing). Always emits `VK_KHR_portability_subset`.

4. **Vulkan 1.3 is ~26% of Android devices (Nov 2025).** Not a majority. This is a meaningful constraint for mobile targets.

5. **lavapipe and SwiftShader support Vulkan 1.3.** Both viable for CI without a GPU.

6. **ORT plugin EP API introduced in ORT 1.22/1.23, still experimental.** Entry point is `CreateEpFactories`. API has been revised multiple times.

7. **No existing Vulkan EP for ORT.** We are first-movers. No Rust plugin-EP crate exists.

**Methodology notes:**
- GitHub code search failed to index the 987KB `ggml-vulkan.cpp` file. Used direct raw URL fetch + offset navigation instead.
- Used `vulkan-shaders-gen.cpp` as ground truth for shader target environment.
- ORT EP header was too large for direct API read; used PowerShell grep on temp file for version tags.
- ExecuTorch verified directly via GitHub code search (`VK_API_VERSION` in `backends/vulkan`).

**Output files:**
- `.squad/fact-checker/audit-trail.md` — appended
- `.squad/decisions/inbox/fact-checker-vulkan-baseline-verification.md` — created
