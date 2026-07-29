# Project Context

- **Owner:** Justin Chu
- **Project:** onnxruntime-ep-vulkan — a cross-platform Vulkan plugin execution provider for ONNX Runtime, written in Rust.
- **Reference architecture:** `C:\Users\justinchu\dev\onnxruntime-mlx` — mirror its layout: `rust/src/{lib,ep,factory,sys,registry,engine,logging}.rs`, `rust/src/ops/*`, `rust/build.rs`, `tests/conformance/`, `bench/`, `python/`, `docs/DESIGN.md`.
- **Stack:** Rust (cdylib plugin EP), Vulkan 1.1+ compute, SPIR-V/GLSL shaders, ONNX Runtime C API, Python bindings, GitHub Actions CI.
- **Cross-platform mandate:** Windows, Linux, Android, macOS via MoltenVK; NVIDIA / AMD / Intel / Adreno / Mali; software rasterizer (lavapipe / SwiftShader) for GPU-less CI.
- **My focus:** Lead / EP Architect — architecture, design docs, scope, review
- **Created:** 2026-07-28T17:52:04-07:00

## Learnings

<!-- SUMMARIZED by Scribe 2026-07-28T22:28:08-07:00 — original entries compressed; decisions.md is the canonical record for all rulings. -->

### [SUMMARY] Sessions 1–6: architecture, baseline, contrib, OQ rulings (2026-07-28)

**DESIGN.md authored (session 1):**
- MLX pipeline shape transfers (factory, vtable, registry, convex clustering, repo layout). Memory ownership does NOT transfer — Vulkan requires OrtAllocator + OrtDataTransferImpl + staging + barriers.
- ORT allocator is pointer-based; VkBuffer is not — opaque tagged-handle registry resolving to `(VkBuffer, offset)` chosen.
- llama.cpp base shaders target vulkan1.2, ExecuTorch targets VK_API_VERSION_1_1. "Requires 1.3" claim was wrong; verified by Fact Checker.
- Claim rate is a bad metric; **fused-region compute volume** (`largest_island_flops`) is the metric of record. Island count + largest fused region must appear in every benchmark.
- Record-once / replay-many (Compile→Compute). ExecuTorch model, not llama.cpp's per-eval re-record.
- Device test must assert `VulkanExecutionProvider` node placement — CPU fallback vacuously passes.

**Baseline frozen — OQ-1 (session 2, after Link's measurements):**
- Reversed provisional `synchronization2`+`subgroup_size_control` hard requirements. Link: 31.43% Android gap on sync2; MoltenVK reports the extension but `subgroupSizeControl=VK_FALSE`.
- **Frozen gate:** Vulkan ≥1.1, compute queue, `maxComputeWorkGroupInvocations ≥ 256`, `maxComputeSharedMemorySize ≥ 16384`, subgroup BASIC, one DEVICE_LOCAL + one HOST_VISIBLE memory type. No required extensions.
- Rule: a requirement that excludes the machines you test on has not been tested. Capability shortfalls degrade op coverage, not device availability.
- Khronos layer shim (Link's Option B) rejected: AOSP loader searches only APK owner's nativeLibraryDir; we are a plugin. Precedent was false.
- Dual-backend barrier seam: backend selected once at device init. `ep.force_legacy_barriers` session option forces legacy path in CI.

**OQ-11 ratification + contrib domain reversal (session 2):**
- `ai.onnx`-only was wrong: ORT GenAI emits `com.microsoft` ops (GQA, RotaryEmbedding, MatMulNBits, LinearAttention) directly for Qwen graphs. For scope questions: read the exporter, not the standard.
- Admitted as **named ops** (not a domain predicate). `if domain == "com.microsoft"` is forbidden. Registry key is the allowlist; graph census in CI is the drift alarm.
- OQ-12 experiment defined: §11.1 fixes devices, pass bar (≥1.5× vs phone's own CPU EP, zero numerical failures), and all four outcome consequences in advance.

**OQ-3 ruling (session 3 — Tank's proposal adopted):**
- BDA is a second shader architecture (requires `GL_EXT_buffer_reference`), not an optimization. Does not remove the side table. MoltenVK support Apple-Silicon-only. **No BDA at all.**
- **Reserved VA registry:** `VirtualAlloc(MEM_RESERVE, PAGE_NOACCESS)` on Windows, `mmap(PROT_NONE)` on POSIX. Real unique spans. Stray dereference = MMU fault, not silent corruption.
- Rule: prefer designs that make a hazard impossible by construction.

**OQ-4 ruling (session 4 — code was right):**
- **Hard Vulkan SDK build dependency.** No checked-in SPIR-V fallback.
- Checked-in `.spv` that drifts from `.comp` is silent wrong-numbers in the build system. Freshness-hash defeats the purpose. Same shape as layer shim and BDA — under-exercised second path.
- `ALLOW_MISSING_GLSLC=1` escape hatch must produce an inert artifact (zero devices, zero claims), not subtly broken. No release artifact from escape-hatch builds.

**Oracle validation + accuracy_level pinning (session 4):**
- CPU EP works as oracle for quantized path (MatMulNBits fp32). `accuracy_level` pinned at 1 — level 4 diverges ~3.6e-3; fp16 NaN/Inf on ORT 1.27 (null-allocator PrePack bug).
- Bit-layout correctness (dequantize) goes to NumPy (independent spec), not CPU EP — shared misreading passes both sides.
- Oracles that change with the machine are not oracles. Pin all CPU-sniffed knobs.

**llama.cpp accelerant + OQ-M6:**
- Rai 🟢 Green. No obligation for reading/learning. Obligation attaches only on substantial source adaptation.
- Block format mismatch = no code copying. Tiling strategy, subgroup reduction shape, dequant-in-register patterns **do transfer** (Switch confirmed). Budget algorithm study time.
- Mouse's "useless" claim was too strong — he answered "can this be copied?", Switch answered "does reading save time?". Both right. Adjudicating on Mouse alone would have been wrong.

**Key process lessons:**
- Mark unverified claims as unverified *in the document* — a lead's wrong entry propagates into everyone's assumptions.
- When two owners appear to disagree, check whether they are answering the same question.
- Pre-commit the conditions under which you will widen before the data arrives. Write the reversal conditions in the document.
- Every time you rely on the team remembering something, write a test instead.
- "Performed a fusion" and "implemented a fused node" differ: GQA arrives as one node; decomposing it materializes a [B,H,S,S] score in VRAM. Implementing is conservative; decomposing is reckless.
- A positive result on a named risk: bank the conditions (pinned `accuracy_level`), not just the headline.

**Milestone status (as of 2026-07-28):**
- M0: ORT loads plugin, enumerates Vulkan device, runs Add node, matches CPU EP, on lavapipe CI.
- M1 gate: template infrastructure before op #1, reported ops-per-kernel ratio ≥ 8.
- M2: device allocator, reserved-VA registry. Gated on M2: LLM path (KV cache cross-subgraph boundary).
- M3: Android tuning (budget only if A+B devices pass all three OQ-12 stages).

**Open questions (as of 2026-07-28):**
- OQ-12: hardware experiment (Adreno 5xx + Mali Bifrost devices).
- OQ-14: fp16 device share on Android (product-scope question).
- OQ-15: shape-agnostic dispatch / `vkCmdDispatchIndirect` (Switch).
- OQ-16: LinearAttention/CausalConvWithState schema stabilization (T5a gated on upstream).
- OQ-13: zero-copy IO binding (Tank, post-M2).
