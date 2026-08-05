# Project Context

- **Project:** onnxruntime-ep-vulkan
- **Created:** 2026-07-29

<!-- CONDENSED-AT: 909a2df46ce38f2774abb57ef9a64c6d984124c7 -->

## Core Context

Agent Fact Checker initialized and ready for work.

## Learnings

## [SUMMARY] Compressed entries

<!-- SUMMARIZED by Scribe 2026-08-04T20-25-00-07-00 -->

- **[SUMMARY] Setup + Round 1-2 audits (2026-07-28–2026-07-29)** — Verified ORT 1.28 API exists and exposes `CreateExternalResourceImporterForDeviceImpl`; verified claims underlying the Vulkan 1.3 baseline proposal; ONNX Attention errata pass 4 (pin, blast radius, C2 blind spot, ORT divergence).
- **[SUMMARY] Audit: GEMV hardware gap, iGPU timestamps, subgroup-free Vulkan (2026-08-01)** — Measured 13.52x RTX4060/Iris-Xe GEMV ratio vs 3.08x memory-bus-predicted ratio, leaving an unverified 4.39x residual (no DRAM counters/matched kernel). Shared-memory/workgroup limits are noise (kernel under-declares vs device limits). Intel device timestamp tick (52.0833ns/19.2MHz) remains valid under CPU load — `NO_STEADY_TAIL` means variable, not wrong. llama.cpp does not require subgroups (shared-tree/hybrid/subgroup-only paths exist); our kernel has only half that structure (register dequant + shared tree, but scalar activation loads, one accumulator). A lavapipe-safe fast path must use capability queries, never a literal subgroup width 32.
- **[SUMMARY] Devil's Advocate: M0 criterion 10 closure (2026-08-02)** — Broke the "three consecutive MATCH runs" closure: `_compare_run_to_cpu` only compares output 0 (logits); outputs 1–64 (KV cache) are compared only Vulkan-to-Vulkan, never to the CPU oracle, so a deterministic stale/zeroed/wrong KV cache passes as `MATCH`. `outputs_compared=65` means `len(run)`, not 65 CPU comparisons. Artifact does not bind source commit/binary digest/session identity — provenance trusted, not recoverable. Settlement: CPU-compare all 65 outputs with declared per-output tolerances, bind commit/binary/session into the artifact, reset counters around the series. (This is the same class of defect later independently found and fixed via Trinity's model-scale oracle and Morpheus's §8.9.24/25 rulings — see decisions.md Round 13.)
- **[SUMMARY] Verification/DA: Morpheus's "six declines" tally (2026-08-02)** — Six self-declared transitions recoverable, but six is not a reproducible count of all deliberate non-numberings — a seventh ("coverage does not compose") exists deliberately unnumbered; several "declines" actually grew normative surface (R9 rule 5, R13 amendment 1, `PROVEN-ELSEWHERE` semantics). No principle observed to fully disappear; some re-derived under new names (navigability cost, not information loss). `LedgerEntry.device` confirmed inert (claim path only checks `lookup_key` Hit). `PROVEN-ELSEWHERE` was prose with no predicate/counter/disclosure/planted-control at time of audit. Model-level ULP promotion must bind to exact proof keys attributed as executed, not promote every per-form entry. Governance is structurally single-writer — no formal second-reader gate. Also corrected: a spawned agent cannot observe the coordinator canary (instructions aren't propagated into its context) — absence is `UNOBSERVABLE`, not evidence of truncation.
- **[SUMMARY] Audit: OQ-12 figures, opset-26, ORT 1.28 ceiling, no-bump class (2026-07-30)** — Android sync2 coverage figure is a moving database snapshot (68.57%→~67.33% in two days), not a constant; 31.43% legacy-path benefit is a ceiling, not the usable intersection with the §7.2 device gate (unfalsifiable without OQ-12). Opset 26 (ONNX 1.21.0) and opset 27 (ONNX 1.22.0) are both released; ORT 1.28 opset ceiling is 27, not 24 (onnxruntime.ai's compat table is stale). No-bump correction class grew by 3 instances in 45 days (9 total), most critical: onnx#8182 (Q/DQ-23/25 reference impls unregistered, silent opset-21 fallback) — directly affects our Q/DQ 21..=25 window; needs a maintainer/recurring sweep, not a snapshot.
- **[SUMMARY] Audit: ONNX Attention-24 errata (2026-07-29)** — Confirmed three distinct bugs in ONNX 1.22.0's Attention-24 reference (wrong causal alignment, NaN on fully-masked rows, mode-3 precision/zeroing); ORT 1.28 already ships the fix (PR #28958) while the onnx package reference is still wrong — a live disagreement window. Oracle pin `onnx >= 1.23` is correct but unreleased at audit time; opset-based version checks cannot detect this class by construction (opset stays 24, model carries no signal) — only a pinned reference-library version works.


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

📌 Team update (2026-08-03T19:55:00-07:00): Trinity found "nobody has run it" was itself an unchecked claim — 	est_gqa.py had G=4 all along. You (Fact Checker) wrote the Nq/Nkv caveat repeatedly in audits without checking whether the suite already answered it. Worth folding into future audits of repeatedly-requoted caveats: a caveat that is never re-verified against the artifact is the same shape as the pre-formed-number defect named in this round's coordinator audit. — decided by Trinity
