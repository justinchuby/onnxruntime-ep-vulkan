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

---

## Audit Entry — ONNX Attention-24 `nonpad_kv_seqlen` Errata
**Date:** 2026-07-29T09:47:45-07:00
**Requested by:** Mouse / coordinator
**Scope:** Verify Mouse's finding that the ONNX reference implementation of `ai.onnx::Attention`-24 was wrong for `nonpad_kv_seqlen`, and characterise the defect, fix, version impact, and opset-bump question.

---

### Claim 1 — The Defect
**Verdict:** ✅ Verified

Primary source: `onnx/defs/nn/defs.cc` (onnx/onnx, SHA 424bd61) — formal Errata annotation embedded in the opset-24 Attention spec:

> "Errata (in-place behavioral correction, no opset bump): the reference implementation and backend tests were incorrect when `nonpad_kv_seqlen != q_sequence_length` (nonzero bottom-right offset, top-left instead of bottom-right causal alignment) and produced `NaN` for fully-masked rows; corrected in version 1.23. This fixed three behaviors described above: external-cache bottom-right causal alignment (`offset = nonpad_kv_seqlen - q_sequence_length`), zero (non-`NaN`) output for fully-masked rows including the mode-`3` `qk_matmul_output`, and the mode-`3` `qk_matmul_output` precision (`T1`)."

Three distinct bugs in the pre-1.23 reference implementation (`op_attention.py`):

1. **Wrong causal alignment (top-left instead of bottom-right).** The causal mask applied `j <= i` (top-left lower-triangular). The spec requires `j <= i + offset` where `offset = nonpad_kv_seqlen[b] - q_sequence_length` per batch element. For `nonpad_kv_seqlen = q_sequence_length` the offset is zero and the two formulas coincide, so this bug is silent in the no-offset case and only manifests when `nonpad_kv_seqlen != q_sequence_length`.

2. **NaN output for fully-masked rows.** When all keys are masked out (the negative-offset case), the raw `_softmax` over all-`-inf` logits produced `NaN` rather than a zero vector. The fixed implementation guards against this explicitly.

3. **mode-3 `qk_matmul_output` bugs.** The debug output was emitted at the wrong precision (not `T1`) and was not zeroed for fully-masked rows.

The spec text in `defs.cc` unambiguously defines the correct behaviour — the reference was simply wrong.

---

### Claim 2 — The Fix: Version Range and Commit
**Verdict:** ✅ Verified

- **Introduced (buggy):** ONNX 1.22.0 — released **June 15, 2026** — first stable release containing `ai.onnx::Attention`-24. Source: LF AI Data blog post (lfaidata.foundation/blog/2026/06/30/).
- **Fixed:** ONNX 1.23.0 — **NOT YET RELEASED** as of 2026-07-29. Dev builds (`onnx-weekly 1.23.0.dev20260727`) available on PyPI since July 26, 2026. Errata text in current `defs.cc` main branch states "corrected in version 1.23."
- **Fix PR:** `onnx/onnx#8068`; tracking issue `onnx/onnx#8054` ("Attention op: support offset-aware causal masking for KV-cache").
- **Fixed reference implementation:** `onnx/reference/ops/op_attention.py` (SHA 48e988e) — current main branch shows correct `_apply_causal` with per-batch offsets and NaN-safe `_softmax`.

**Impact on version range:** The bug exists in **every released stable version of the ONNX Python package** (1.22.0 is the only stable release containing Attention-24, and it is buggy). The fix exists only in dev builds and the forthcoming 1.23.0 stable release.

**ORT kernel timeline:**
- ORT 1.22–1.27: CPU and CUDA Attention-24 kernels implemented **top-left** alignment — agreeing with the buggy reference.
- ORT 1.28.0 (released 2026-07-24): PR `microsoft/onnxruntime#28958` fixed both CPU and CUDA kernels to bottom-right alignment. ORT 1.28 kernel is **CORRECT**; the ONNX 1.22.0 reference is **WRONG**. There is therefore a current period where running ORT 1.28's Attention-24 against an onnx 1.22.0 reference oracle will produce false failures.

---

### Claim 3 — Was an Opset Bump Warranted?
**Verdict:** ✅ No bump warranted; ONNX's classification is correct

ONNX's convention: operator *semantics* are versioned by opset; the reference implementation is a Python artifact versioned by ONNX library release. A bump is warranted when the *spec text* is ambiguous and the correction resolves that ambiguity in a way that changes previously-valid behaviour. When the spec is unambiguous and the reference simply implemented it wrong, a bump only adds fragmentation.

Evidence that the spec was unambiguous: the `defs.cc` spec text explicitly states:
- `offset = nonpad_kv_seqlen - q_sequence_length` (per batch)
- "A fully-masked query row … produces a zero output row, not `NaN`"
- "mode-`3` `qk_matmul_output` is emitted at the operator's output precision (`T1`)"

All three correct behaviours are stated in the spec text. The reference violated them without spec ambiguity to hide behind. ONNX's "in-place behavioral correction" classification is consistent with its policy.

Comparable precedent: this Attention-24 case is itself the leading cited example of a no-bump reference correction in ONNX history. Web research found no earlier precedent of comparable scope. Justin (as ONNX owner) has independently confirmed the no-bump decision (see `copilot-directive-attention24-no-bump.md`).

---

### Claim 4 — Blast Radius for Consumers
**Verdict:** ✅ Verified; three distinct affected surfaces

Affected configurations:
- **All three bugs:** `nonpad_kv_seqlen` present + `is_causal=1` + `nonpad_kv_seqlen != q_sequence_length` — wrong expected outputs for `Y`, `present_key`, `present_value`, and mode-3 `qk_matmul_output`.
- **NaN bug only:** any configuration with fully-masked rows (including offset < 0, i.e. `nonpad_kv_seqlen < q_sequence_length`) — produces NaN in onnx 1.22 reference, zero in correct implementation.
- **mode-3 bug only:** mode-3 `qk_matmul_output` in any configuration — wrong precision in onnx 1.22 reference.

Not affected: any Attention-24 usage where `nonpad_kv_seqlen` is absent or `is_causal=0` and no fully-masked rows occur.

ORT-specific divergence period:
- Before ORT 1.28 + onnx 1.22: both ORT kernel and reference are wrong — they agree, so conformance tests pass (wrongly).
- ORT 1.28 + onnx 1.22 (current situation): ORT kernel is correct; reference is wrong — conformance tests will show false failures for the affected configurations.
- ORT 1.28 + onnx 1.23 (target): both correct — tests pass for the right reason.

---

### Claim 5 — Other Opset-24 Ops with Same Pattern
**Verdict:** ⚠️ No additional cases found; narrow search

Mouse already verified `RMSNormalization` and `RotaryEmbedding` as unrevised through opset 27. The 19 float8e8m0 type-constraint additions are type-only changes (no behavioral content), confirmed clean.

Web research found no additional post-release reference corrections for other newly-added opset-24 operators. The Attention-24 errata is currently the only documented case of this class in ONNX 1.22.x. This does NOT mean other ops are safe — it means no errata have been documented. This class of defect is invisible to opset-based version checks by construction.

---
---

## Audit Entry — ONNX Attention-24 Errata, Pass 4 (Focused)
**Date:** 2026-07-29T10:34:41-07:00
**Requested by:** Coordinator (Justin ruled no opset bump; upstream summary dropped; focus on oracle pin, blast radius, C2 blind spot, ORT divergence)

### Claim 1 — Exact Pin
**Verdict:** ✅ Verified. `onnx >= 1.23`. Lower bound suffices.

Fix commit: SHA 2816da65 (`onnx/onnx#8068`), merged 2026-06-20 — five days after ONNX 1.22.0 release (June 15, 2026). ONNX 1.22.0 = bug present. ONNX 1.23.0 = fix present; NOT YET RELEASED as stable (dev builds `onnx-weekly 1.23.0.dev20260727` available July 26, 2026). No regression found in any later version. Lower bound is sufficient; exact pin is unnecessary and would block future upgrades.

### Claim 2 — Blast Radius
**Verdict:** ✅ Verified. Four distinct bugs in one commit. Precisely scoped:

Bug A (causal alignment): `nonpad_kv_seqlen` present + `past_key` absent + `is_causal=1` + `nonpad_kv_seqlen != q_sequence_length`. Top-left mask applied where bottom-right is required.
Visual: For `q_sequence_length=2`, `valid_kv_length=4`: top-left allows q2→k0 only; correct allows q2→{k0,k1,k2}.

Bug B (allowed-cell NaN): `is_causal=1` + boolean `attn_mask`. `(1 - attn_mask) * -inf` produces `0 * -inf = NaN` at allowed cells.

Bug C (fully-masked-row NaN): any config where all keys disallowed for a row (including negative offset or all-False attn_mask row). Softmax over all-`-inf` → NaN instead of 0.

Bug D (mode-3 precision): `qk_matmul_output_mode=3` → wrong precision (not T1), not zeroed for fully-masked rows.

Not affected: no `nonpad_kv_seqlen`, `is_causal=0`, float attn_mask or none, no fully-masked rows, mode-3 not requested.

Code fix key lines (from commit message):
- Bug A: `causal_mask = np.tril(ones, k=0)` → per-batch `offset[b] = nonpad_kv_seqlen[b] - q_sequence_length`; mask `j <= i + offset[b]`
- Bug B: `(1 - attn_mask) * -inf` → `np.where(attn_mask, 0.0, -np.inf)`
- Bug C: added `np.isinf(bias).all(axis=-1)` guard → `Where(fully_masked, 0, probs)` before `P @ V`

### Claim 3 — Same Shape Elsewhere (C2 Blind Spot)
**Verdict:** ✅ Verified. Class recurs. C2 has a confirmed blind spot.

The PR #8068 commit message explicitly cites two prior no-bump behavioral corrections as precedent:
- `onnx/onnx#7297`: Resize operator — in-place behavioral correction
- `onnx/onnx#7867`: Attention softcap ordering fix — no opset bump

The same PR also patches opset-23 Attention (`old.cc` SHA 3518a419) in-place — errata note added for "fully-masked query row NaN → 0" and prose reconciliation (old.cc previously stated `is_causal` and `attn_mask` are mutually exclusive; corrected without opset bump). This is patching a released frozen opset.

Four confirmed cases total. C2's fingerprint machinery (opset-number-based) is blind to all of them by construction. Documented in decision file for Mouse to add to C2's known-limitation log.

### Claim 4 — ORT vs Reference Divergence Period
**Verdict:** ✅ Verified. ORT 1.22–1.27 matched buggy reference; ORT 1.28 now correct; reference still wrong.

Nuance from commit message: ORT "hard-rejected or never emitted" the `nonpad_kv_seqlen` external-cache path in production before 1.28. The wrong kernel existed but was never triggered by valid models. ORT 1.28 both fixes the kernel and enables the path. The active false-failure window (ORT 1.28 correct vs onnx 1.22.0 wrong) is real but affects only test harnesses that run the onnx Python reference against ORT 1.28 on these inputs.

---