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

## Audit: OQ-12 figure currency, legacy-path justification, opset-26 reality, ORT 1.28 ceiling, no-bump class — 2026-07-30T07:05:09-07:00

**Task:** Verify two load-bearing claims: (1) the 31.43% Android figure behind the OQ-12 legacy barrier path decision; (2) the ONNX opset-26 claim and associated no-bump errata.

**Key learnings:**

1. **The 68.57% Android sync2 figure is a live database snapshot, not a fixed constant.** Source: vulkan.gpuinfo.org, pulled 2026-07-28. A web query on 2026-07-30 returned ~67.33% (gap ~32.67%). The figure IS moving; the direction suggests new legacy/budget device submissions are pulling coverage down. Treat it as a floor on the complement (real installed-base gap is likely higher than the database shows).

2. **31.43% is a ceiling on legacy-path benefit, not a measured value.** Sync2-lacking devices may also fail the §7.2 device gate (Vulkan < 1.1, no compute queue, etc.). The legacy path only benefits devices that BOTH lack sync2 AND pass the gate. The database says nothing about the intersection. Per R9: a single gpuinfo.org reading names a database-sample fraction; the "usability fraction" is unfalsifiable without OQ-12.

3. **The gpuinfo.org coverage calculation correctly accounts for Vulkan 1.3 devices** (where sync2 is core). There is no undercounting from the 1.3 promotion. The 1.3 error direction does not dominate.

4. **Drop conditions for the legacy path are clearly unmet.** Android database coverage would need to reach ≥99% AND OQ-12 would need to confirm gap devices fail §7.2 for other reasons. Windows coverage (87.78%) has an independent 12.22% gap. Both gaps must close. Neither is close today.

5. **Opset 26 is real and released.** ONNX 1.21.0 (April 27, 2026) introduced opset 26 (BitCast, CumProd, 2-bit types). ONNX 1.22.0 (June 15, 2026) introduced opset 27. ONNX 1.23.0 is unreleased as of 2026-07-30. Justin's ruling ("support up to opset 26") is consistent with available packages.

6. **ORT 1.28 opset ceiling is opset 27, not 24.** ORT 1.28 upgraded to ONNX 1.22.0, which supports opset 27. The onnxruntime.ai compatibility table is stale (last row: ORT 1.20 = opset 21). Any claim based on that table is wrong. "Supporting opset 26" is fully exercisable against ORT 1.28.

7. **The no-bump correction class grew: three new instances since onnx 1.22.0 shipped.** The most critical: **onnx#8182 (merged 2026-07-12, unreleased)** — Q/DQ-23 and Q/DQ-25 reference implementations were not registered in `_op_list.py`. ReferenceEvaluator silently fell back to opset-21 behavior. Using `output_dtype` (Q/DQ-23) or `precision` (Q/DQ-25) with onnx ≤ 1.22.0 raises TypeError (detectable) or silently uses the wrong implementation (C2 blind spot). Directly affects our Q/DQ `21 ..= 25` window. Also: onnx#8099 (ScatterND min/max, not in our plan), onnx#8194 (TopK sorted=0, not in our plan).

8. **The no-bump table needs a maintainer, not a snapshot.** As of 2026-07-30, at least 9 instances of the class are known (6 in existing table + 3 new). The class grew by 3 instances in 45 days. Someone (Mouse or Trinity) must own a recurring sweep.

**Methodology notes:**
- vulkan.gpuinfo.org page is JavaScript-rendered; could not be fetched directly. Web search consistently returned 67.33% citing gpuinfo.org. Treat this as approximate.
- ORT 1.28 ceiling: confirmed via primary source (ORT 1.28 release notes on GitHub, stating "Upgraded to ONNX 1.22.0"). Compatibility table on onnxruntime.ai is stale; ignored it.
- No-bump class: enumerated via `gh api "repos/onnx/onnx/pulls?state=closed..."` filtered to merged > 2026-06-15. Three qualifying PRs found; #8182 confirmed as in-plan impact via PR body read.
- ONNX version timeline: LF AI & Data blog 2026-04-27 for v1.21.0; GitHub releases page for v1.20.0; prior history entries for v1.22.0 date. Consistent across sources.

**Output files:**
- `docs/PLATFORMS.md` §10 — added §10.0 (Fact-check: OQ-12 figures) with figure currency, error direction, drop conditions, and R9 falsifiability instruments
- `.squad/decisions/inbox/fact-checker-oq12-sync2-opset.md` — full decision record; routes #8182 to Mouse+Trinity, no-bump sweep to owner TBD

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

---

## Audit: ONNX Attention-24 Errata Verification — 2026-07-29T09:47:45-07:00

**Task:** Verify Mouse's finding that the ONNX reference implementation of `ai.onnx::Attention`-24 was wrong for `nonpad_kv_seqlen`, and characterise it for Trinity (oracle pin) and Justin (upstream reporting).

**Key learnings:**

1. **ONNX 1.22.0 (June 15, 2026) introduced Attention-24 with the bug — not a hypothetical.** The bug exists in every released stable version of the onnx package that contains Attention-24. Onnx 1.23.0 is the fix, but it is unreleased (dev builds only as of 2026-07-29).

2. **Three distinct bugs, not one.** Wrong causal alignment (top-left instead of bottom-right), NaN for fully-masked rows, and mode-3 qk_matmul_output precision + zeroing. Only the causal alignment bug requires nonpad_kv_seqlen != q_sequence_length; the NaN and mode-3 bugs can manifest even with nonpad_kv_seqlen == q_sequence_length if fully-masked rows occur.

3. **ORT 1.28 is already fixed; the disagreement period is NOW.** ORT 1.28 (July 24, 2026) has correct kernels via PR #28958, but onnx 1.22.0 reference is still wrong. If Trinity runs conformance tests using the onnx Python reference against ORT 1.28 for the affected inputs, she will see false failures. This is time-sensitive.

4. **The oracle pin lower bound suffices, but the pinned version is unreleased.** `onnx >= 1.23` is correct, but 1.23 is not out yet. Immediate workaround: use onnx-weekly dev build (available), or implement expected outputs directly from the spec formula to bypass the library entirely.

5. **No-opset-bump decision is correct per ONNX policy.** The spec text was unambiguous; the reference was simply buggy. Justin (as ONNX owner) has also independently confirmed the no-bump decision.

6. **This is the leading cited example of in-place behavioral correction in ONNX history.** Web research found no comparable precedents of prior scope. The Attention-24 errata note is itself the canonical reference for this class of correction.

7. **opset-based version checks cannot detect this class of defect by construction.** The opset stays 24; the model carries no signal. The only reliable detection is a pinned onnx library version in the test environment. This is a known limitation of C2 fingerprinting that should be documented explicitly.

**Methodology notes:**
- Primary source for errata text: `defs.cc` SHA 424bd61 via GitHub code search (fragment indexing).
- ONNX version timeline confirmed via: web search (LF AI Data blog, June 30 2026 for 1.22.0 release date), ORT 1.28 release notes referencing onnx 1.22.0, PyPI onnx-weekly dev build timestamps.
- ORT PR #28958 merge in 1.28 confirmed via web search.
- Fixed reference implementation (`op_attention.py` SHA 48e988e) confirmed via GitHub code search from earlier pass.
- GitHub releases page showed pagination artifacts (only v1.20.0 visible in first fetch) — resolved via targeted web searches for release dates.

**Output files:**
- `.squad/fact-checker/audit-trail.md` — Claims 1-5 appended
- `.squad/decisions/inbox/fact-checker-onnx-attention24-oracle.md` — created (Section A: Trinity/Mouse oracle pin; Section B: Justin upstream-ready summary)
---

## Audit: ONNX Attention Errata — Pass 4 (Focused) — 2026-07-29T10:34:41-07:00

**Task:** Pin, blast radius, C2 blind spot, and ORT divergence — upstream summary dropped per coordinator directive.

**Key learnings:**

1. **The fix commit (2816da65) is dated June 20, 2026 — five days after ONNX 1.22.0 released June 15.** The GitHub commit API gives precise merge dates. Always check the commit timestamp against the release date to establish exactly which release contains a fix.

2. **`onnx >= 1.23` is a lower bound, not an exact pin.** The oracle drifts in one direction. An exact pin would block future onnx upgrades unnecessarily.

3. **The class of no-bump behavioral corrections recurs.** The PR commit message itself cites #7297 (Resize) and #7867 (Attention softcap) as precedents. When evaluating whether a given fix is a one-off, check the PR's own "Why no opset bump" rationale — if it cites prior cases, the class is confirmed recurring.

4. **The PR also patches opset-23 (old.cc).** Checking `old.cc` is mandatory when a fix touches `defs.cc` — the same bug often exists in the previous version's frozen function body.

5. **ORT hard-rejected the path before 1.28.** The wrong kernel existed in ORT 1.22–1.27 but never triggered in production. The active false-failure window is test-harness-specific, not model-execution-specific.

6. **The commit message "Why no opset bump" section is the best primary source for the policy rationale.** It cites specific prior PRs and explains the exact condition under which ONNX allows in-place corrections. Always read this section when evaluating errata.

**Methodology:** Fetched commit SHA via GitHub commits API (path=onnx/reference/ops/op_attention.py, per_page=5). Full commit message retrieved via SHA lookup. old.cc SHA from initial GitHub code search. PR precedents #7297 and #7867 confirmed via web search.

**Output files:**
- `.squad/fact-checker/audit-trail.md` — Pass 4 entry appended
- `.squad/decisions/inbox/fact-checker-onnx-attention24-oracle.md` — overwritten with focused content (upstream summary removed; C2 blind-spot section added)