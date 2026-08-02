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

<!-- SUMMARIZED by Scribe 2026-08-01T20:39:12-07:00 -- older entries condensed below; full text lives in git history -->

### [SUMMARY] Compressed entries (condensed 2026-08-01T20:39:12-07:00)

- **Audit: ORT 1.28 API Verification — 2026-07-28T18:51:35-07:00** — **Task:** Verify Justin's claim that ORT 1.28 exists and exposes `CreateExternalResourceImporterForDeviceImpl`.
- **Audit: Vulkan Baseline Verification — 2026-07-28T17:59:54-07:00** — **Task:** Verify claims underlying Justin's Vulkan 1.3 baseline proposal.
- **Audit: ONNX Attention Errata — Pass 4 (Focused) — 2026-07-29T10:34:41-07:00** — **Task:** Pin, blast radius, C2 blind spot, and ORT divergence — upstream summary dropped per coordinator directive.

## Audit: GEMV hardware gap, iGPU timestamps, subgroup-free Vulkan — 2026-08-01

**Task:** Verify whether the 13.5x RTX 4060 Laptop / Iris Xe int4 GEMV gap is hardware-predicted,
whether Intel device timestamps remain trustworthy under CPU load, and whether the subgroup-free
MatMulNBits structure is current.

**Key learnings:**

1. The local Intel system is i7-13800H with LPDDR5 configured at 5200 MT/s. Its theoretical system
   memory ceiling is 83.2 GB/s versus the RTX 4060 Laptop's 256 GB/s: 3.08x, not 13.5x.
2. The measured 3425.9 / 253.4 us ratio is 13.52x, leaving a 4.39x residual over the bus ratio.
   A 3–5x well-tuned expectation is plausible but remains unverified without DRAM counters or a
   matched known-good kernel.
3. Current shared-memory and workgroup limits are noise: q_gemv declares 1 KiB, while devices report
   32/48 KiB; it uses 128/256 invocations while both devices report 1024.
4. Intel UMA/package contention is real and can vary achieved GT frequency and DRAM service, but no
   source supports assigning it a stable 2–3x multiplier. That number needs a controlled load sweep.
5. The Intel timestamp tick remains valid. 52.0833 ns is a 19.2 MHz reference timer; CPU load changes
   work per tick, not the tick. `NO_STEADY_TAIL` means valid-but-variable durations, not bad scaling.
6. llama.cpp does not require subgroups. It builds shared-tree, hybrid, and subgroup-only reductions
   and selects the shared fallback when subgroup arithmetic is absent.
7. llama.cpp's subgroup-independent advantages are packed/vector loads, register dequantisation,
   manual unrolling, and multiple register accumulators per thread/workgroup. Our kernel has only the
   first half of that structure: register dequantisation plus a shared tree, but scalar activation
   loads and one output accumulator.
8. A lavapipe-safe fast path must retain baseline SPIR-V and use subgroup capability queries,
   `gl_NumSubgroups`, and subgroup IDs—never a literal width 32.

**Methodology notes:**
- Did not benchmark or issue device work. Used existing trace/device metadata, CIM hardware identity,
  public specifications, Vulkan sources, Linux i915 clock source, and current llama.cpp source.
- Corrected PERF.md's claim that Intel's device clock itself is not contention-immune.
- Per R9, every number in the report names a falsifying observation; unmeasured expected speedups are
  explicitly ⚠️ rather than carried as constants.

**Output files:**
- `docs/PERF.md` §§13.4.4–13.4.6
- `.squad/decisions/inbox/fact-checker-gemv-hardware-clock.md`

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
