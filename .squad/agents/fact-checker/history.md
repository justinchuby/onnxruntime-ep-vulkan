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

## Devil's Advocate Audit: M0 criterion 10 closure — 2026-08-02

**Task:** Try to break the claim that criterion 10 closed on three consecutive attributed `MATCH`
runs, with special attention to CPU fallback, one-session provenance, witness independence, and
the previously unwritten KV outputs.

**Finding:** ❌ **The closure is not supported by the criterion's written scope.** The artifacts
truthfully report the harness's derived `MATCH`, and the constructor correctly makes `MATCH`
unrepresentable when ORT reports zero Vulkan executions. But `_compare_run_to_cpu` compares only
output 0 (logits). Outputs 1–64 are compared only between Vulkan runs. A deterministic stale,
zeroed, or consistently wrong KV cache therefore passes all three runs and is labeled
`model_output_equivalence = MATCH` without ever being compared to the CPU oracle.

**Verification details:**

1. ✅ `verdict = MATCH` is the same canonical verdict vocabulary written under
   `model_output_equivalence`; it is not a coincidentally named lane field.
2. ❌ `outputs_compared = 65` means `len(run)`, not 65 Vulkan-vs-CPU comparisons. CPU agreement is
   argmax equality, top-10 overlap ≥5, and non-zero Vulkan logits; `max_abs_diff` is recorded but
   does not gate the verdict.
3. ⚠️ The three ORT Vulkan node events are session-aggregate. They are compatible with one island
   per run but carry no run IDs. The general `uniformly_attributed` test cannot distinguish
   distributions such as `[2,1,0]`.
4. ✅ Current test source creates one `InferenceSession` and calls it three times. ⚠️ The persisted
   artifact does not bind source commit, binary digest, session identity, or raw profile, so that
   provenance is trusted rather than independently recoverable.
5. ✅ A zero-attribution forged CPU-only profile cannot reach `MATCH`; existing adversarial tests
   enforce `UNATTRIBUTED`.
6. ⚠️ ORT's provider-tagged node profile event is emitted by `KernelScope` destruction before
   `ExecuteKernel` checks the returned status. It can record an attempted failed kernel. EP
   counters count successfully completed dispatch calls, but are process-global, are not reset by
   this criterion test, and do not prove that a complete island supplied the returned outputs.
7. Two devices are useful against device-specific defects but are not independent against the
   shared comparison scope, harness, parser, model, or constructor.
8. The fixed-in-advance condition blocks post-hoc hardening, but cannot convert a narrow
   measurement into discharge of broader criterion words.

**Settlement evidence:** Independently regenerate from a rebuilt binary; CPU-compare all 65
outputs on every run with declared per-output tolerances; record per-run completed-island
attribution; retain the raw profile; reset/snapshot counters around the series; and bind model,
source commit, ORT/EP binaries, and session identity into the artifact. Add a longer persistent
session as a standing cache-exhaustion falsifier, not as a retroactive replacement criterion.

**Output:** `.squad/decisions/inbox/fact-checker-criterion10-devils-advocate.md`

---

## Verification and Devil's Advocate Audit: Morpheus's six declines — 2026-08-02

**Task:** Verify Morpheus's “six declines” tally, determine whether the declined rules survived as
unnumbered obligations, test for later loss/re-derivation, and attack the `PROVEN-ELSEWHERE` ruling.

**Finding:** The audit does **not fully clear** the policy claim.

1. ✅ Six self-declared tally transitions are recoverable: R9 anti-correlation; R13 defaulting
   lookup; classifier scope under R13; criterion-12 witness/discharge under R11; R9's dual; and the
   proof-ledger device-frame ruling.
2. ❌ Six is not a reproducible count of all deliberate non-numberings. “Coverage does not compose”
   is explicitly “DELIBERATELY NOT NUMBERED,” adds a universal two-gate/two-extent obligation, and
   leaves the tally unchanged at four. The sixth ruling also adds a separate drafting obligation.
3. The declines are mostly honest **top-level deduplication by remedy**, but several are not
   restraint from normative growth: R9 gained rule 5, R13 gained amendment 1, and
   `PROVEN-ELSEWHERE` adds claim semantics, promotion, counting, and disclosure.
4. ⚠️ No principle was observed to disappear. The unnumbered text is in `DESIGN.md` and gets reused.
   Re-derivation did occur: anti-correlation returns as R9's dual, and differing instrument extents
   return in the expensive-proof/cheap-port composition. That is a navigability cost.
5. ✅ `LedgerEntry.device` is currently inert: the claim path asks only whether `lookup_key` is a
   `Hit`. The diagnosis and the need to name cross-device extrapolation are verified.
6. ⚠️ `PROVEN-ELSEWHERE` is prose at `d28a04a`; no predicate, counter, disclosure artifact, or
   planted control implements the promised guard. A claimable warning state can become an ignored
   default.
7. ❌ A model-level ULP series cannot promote every per-form ledger entry. Trinity's independent
   two-device ULP measurement removes the same-author circularity, but promotion must be limited to
   exact proof keys attributed as executed in that run. Otherwise the expensive proof and cheap
   invariant compose to their weaker shared extent.
8. ⚠️ Governance is structurally single-writer. Adversarial review has corrected Morpheus before,
   but the register has no formal second-reader gate for binding unnumbered obligations.

**Settlement:** Mechanically index all normative obligations, not only `R#` headings; require a
non-Morpheus reviewer for additions; implement three ledger states with per-run counts and exact
keys; plant a mismatched-device control; and bind ULP promotion to device, attribution, and exact
exercised keys.

**Correction captured:** A spawned agent cannot observe the coordinator canary because coordinator
agent instructions are not propagated into its context. Absence is therefore `UNOBSERVABLE`, not
evidence of truncation. Recorded separately for the coordinator/skill owner.

**Outputs:**
- `.squad/decisions/inbox/fact-checker-six-declines-audit.md`
- `.squad/decisions/inbox/fact-checker-subagent-canary-instrument.md`

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

## 2026-08-03 — Round 10: register derivation, coordinator claim audit, rigour-ratio DA

**Task:** Three items from the coordinator. (1) Take over Morpheus's decline tally and **derive** the
rule register from citations rather than from his self-count — he ruled that the count must leave his
hands after I found it "measures numbering, not register growth". (2) Apply the same instrument to the
coordinator's own claims across the session, on his request, because he is not neutral on his own
tallies either. (3) Devil's Advocate: steelman that the measurement discipline is not worth its cost,
with a 30-day pre-mortem.

**Finding:** Register derives to **13 numbered + 8 unnumbered-but-binding**; **3 of Morpheus's 8
declines survive**. Coordinator's derived error count is **21 against 5 self-reported classes (33%
recovery)**, rate **0.15/user-facing turn**, **95% corrected, 0 defended**. Apparatus:implementation
ratio is **4.55 : 1** (2.73 : 1 stripped of docs and squad state).

**Key learnings:**

1. **Fix the definition before looking at the results, and say so in the file.** I wrote the four
clauses — externality by path ownership, bindingness, recoverability, non-authorship — into §1 of the
published register before running the instrument. A definition chosen after seeing results is not a
measurement, and the whole authority of a derived count rests on that ordering being visible.

2. **Separate the instrument from the verdict.** `derive_register.ps1` emits owner-tagged citation
hits and **no verdicts**; the judgement lives in the prose. That makes a disagreement about method
distinguishable from a disagreement about result — which was the coordinator's stated reason for
wanting the count derived at all.

3. **Git author metadata is useless here — every commit is authored "Justin Chu".** Agent attribution
has to come from path ownership (`team.md` / `routing.md`), branch names, and `**By:**` blocks in
decision records. Any citation instrument on this repo must be built on the ownership map, and it
inherits that map's errors.

4. **Three destinations for a declined principle, and only two are declines.** Nowhere → survives.
Into an existing numbered rule as an amendment → survives (on-book growth is navigable). Into
unnumbered prose that a non-author later cites as binding → **does not survive**, because that is
exactly the off-book growth the count was built to detect. Five of eight landed in the third bucket.

5. **The shadow register — the finding I did not expect.** `R1`–`R13` are cited ~1,337 times outside
Morpheus's surfaces; **`§8.9.x` is cited 339 times**, 80 of them in `registry.rs`. This **refutes the
navigability diagnosis Morpheus accepted from me last round.** Agents were never lost; they were using
a second namespace with different semantics — `R#` names an obligation, `§8.9.x` names a location, and
a location can be re-cut under a stable citation. §8.9.19 already restates §8.9.17 over one word.
Lesson: before accepting "people cannot find it", check whether they found something else.

6. **Self-reported error lists fail in a direction nobody predicts.** I expected the coordinator's
list to be missing his worst misses. It was missing his own **catches** — the 5 he reported are the 5
adjudicated by a named agent in writing; the 13 he missed are ones he found himself mid-turn. Memory
of one's errors is indexed by who ruled on them, not by severity.

7. **Score the epistemic claim separately from the value.** `weight_reread_amplification = 1.000000`
was quoted as "exact, not approximate" over four literals and an `x/x` ratio — wrong as a claim about
status — and Niobe's later real measurement returned exactly 1.000000. ⚠️ mis-scoped, not ❌ false.
Collapsing those two would have been unfair in one direction and useless in the other.

8. **Persistence matters more than rate, and persistent errors have a shape.** The three long-lived
ones all involved a number or a green token that **arrived pre-formed from an instrument and was
re-quoted rather than re-derived**. That is R13's and R9's family, arriving from a different direction.

9. **A clearing verification is worth as much as a damning one — state both at the same volume.**
`R14` appears exactly once in the tree (in my own prior text): no phantom rule, no lost principle, the
register is under-counted rather than under-populated. Likewise 95% correction and zero denials go in
the headline, not the footnotes.

10. **Refuse the flattering denominator even when it serves your own argument.** Apparatus vs GLSL
alone gives 45:1 and a much sharper Devil's Advocate brief. It is a denominator error of precisely the
kind this project catalogues, so I used 4.55:1 and 2.73:1 and said why.

11. **Get the competitor number from primary source.** `gh api` on llama.cpp's Vulkan backend returned
164 shader entries and a 997,161-byte backend file. The web-search summary of its op parity went in as
⚠️ unverified and was not leaned on anywhere the argument load-bears.

12. **The absence of a counter is a finding in a repo that counts everything.** There is no counter for
the marginal cost of adding an op, in a project with counters for nearly all else — which is why the
per-item cost could rise every round without anyone seeing it.

**Methodology notes:**
- Task 2 evidence base: local session store (`session_store_sql`, `source: "local"`), session
  `c6bec1a7-ab4c-46df-bfbf-1305f9a28366`, turns 0–163, 136 carrying a response. Timestamps are UTC.
- The coordinator's user-facing summaries are **in Chinese**; anchors had to include 我错 / 更正 /
  撤回 / 推翻了我 / 逐位 / 放大 / 哈希 as well as English.
- **Hard observability limit, stated in the published file:** his *routing prompts* are retained
  nowhere I can read. Two of five self-reported items survive only because Morpheus quoted them.
  21 is a floor whose height is set by other agents' note-taking. Direction of error: undercount.
- `git --no-pager` is not valid in this environment. `Select-String` over `docs/DESIGN.md` (663 KB,
  very long lines) floods output — use `git grep -l`/`-n` with narrow paths and the `view` tool with
  `view_range`. Always exclude `':!*.json' ':!*.log'` from tree-wide greps.
- Worked in worktree `../ep-vulkan-fact-checker` on `squad/fact-checker` from `8ac1172`; did **not**
  switch the shared main worktree, which holds other agents' uncommitted work.
- `.squad/decisions/inbox/` is gitignored **and** worktree-local, so inbox records written inside an
  agent worktree are invisible to Scribe. All three were written into the **main** worktree's inbox.

**Output files:**
- `.squad/fact-checker/rule-register-derived.md` — Task 1, the register Morpheus does not edit
- `.squad/fact-checker/derive_register.ps1`, `.squad/fact-checker/derive-hits-8ac1172.txt` — instrument + preserved hit list
- `.squad/fact-checker/coordinator-claim-audit.md` — Task 2
- `.squad/fact-checker/devils-advocate-rigour-ratio.md` — Task 3
- `.squad/fact-checker/audit-trail.md` — three entries appended
- `.squad/decisions/inbox/fact-checker-derived-rule-register.md`, `-coordinator-claim-audit.md`, `-rigour-ratio-premortem.md` (main worktree)
