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
