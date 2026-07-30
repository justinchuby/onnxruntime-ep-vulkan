# Morpheus (Lead-Architect) — history.md

## Learnings

### [SUMMARY] Sessions 1–22: design, OQ rulings, contrib admission, M0 assessment, R6/R7/R8, §8.8 (2026-07-28–2026-07-30)

**Sessions 1–6 (archived):** DESIGN.md and README.md produced. Capability set baseline: Vulkan ≥1.1 + sync2 + subgroup_size_control (either 1.3 core or as extensions). Milestones M0–M3. `com.microsoft` contrib domain admitted under constraints C1–C7. OQ-3 ruled: registry only, no buffer device address. OQ-4 ruled: hard Vulkan SDK build dependency (supersedes checked-in SPIR-V fallback). §9.1.2 established.

**Sessions 7–9 — coverage is producer-relative; §8.5:**
T3 first kernel defined. §8.5 second strengthening: "model file is the fact, builder source is intent." Metric of record: `(claimed_op_coverage, island_count, largest_island_flops)`. Producer+version required. §7.2 device gate frozen: five requirements, no required extensions.

**Session 10 — first dispatch and disclosure failure mode:**
§9.1.2 rewritten after first dispatch: three qualifiers (one kernel, one OS, no ORT-mediated path yet). Disclosure risk inverted — from overclaiming to understating.

**Session 11 — the file vs the builder:**
§8.5 third strengthening: "builder source is intent; the model file is the fact." Producer revision required in every claim. C2 opset-based checking cannot see behavioural corrections shipped without opset bump — documented blind spot.

**Session 12 — manufactured evidence and probe failure:**
R6: a decision can be right, reasonably reached, and rest on manufactured evidence. Three rules for load-bearing reasons in decision records. §7.9 capability probe discipline (five rules). Corrected: `push_next` chain bug (D-S12-01), `is_uma` predicate (D-S12b-01).

**Session 13 — R7 and the skip contradiction:**
R7: instruments fabricate negatives; "absence of instrument must not read as success." Three-layer skip contradiction: OnceLock dead, profiling JSON crashed Intel, per-op `live` flag vacuous pass. "Derive, do not declare."

**Session 14 — criteria and the standard:**
M0 criterion 3 NOT met: "no errors surfaced" = layer not loaded. M0 criterion 8 MET: legacy barrier backend executed (46/28 bit-exact, both devices, both backends). M0 criterion 9: PLATFORMS.md LVP2 needs retraction (done). Six criteria met, one partial, one not met. A standard that yields the first time it costs something was never a standard.

**Session 15 — first-match ceiling:**
§8.7: template evidence (similarity is not a measurement). The 100 staged nodes had never reached the shape check — shape viability unknown. R8: a decline code names the first failing check, not the only one.

**Session 16 — §8.8 dynamic shapes ahead of kernels:**
§8.8 RULING: dynamic-shape support is a claim-path capability, moves ahead of kernels. `REQUIRE_STATIC_SHAPES` → `ENGINE_ACCEPTS_RUNTIME_EXTENTS` (inverted). OQ-15: re-record per shape for M1. M1 gains second-token criterion. §10.0.1 R8 added.

**Current state:**
M0 open: validation positive control (criterion 3) + CI lanes. All architectural rulings current. §8.8 governs the next phase. Risks R5 (rationale corrected), R6, R7, R8 all in §10.0.1. C2 blind-spot re: no-opset-bump behavioural corrections documented in §8.3.
---

## 📌 Cross-agent context — Round 4 (2026-07-30T02:49:12-07:00)

### Worktree layout and inbox portability constraint
The team works in git worktrees: `squad/switch` at `C:\Users\justinchu\dev\ep-vulkan-switch`, `squad/mouse` at `C:\Users\justinchu\dev\ep-vulkan-mouse`, `squad/tank` at `C:\Users\justinchu\dev\ep-vulkan-tank`, with `main` as the integration tree. `.squad/decisions/inbox/` is **gitignored** — records written in a worktree do NOT travel with the branch. The inbox in `main` is authoritative.

### Vulkan SDK path
`C:\VulkanSDK\1.4.350.0` — installed but **not on the default PATH**. `glslc` discovery must search this path; `VULKAN_SDK` env var is the canonical pointer.

### Local hardware — both GPUs pass the §7.2 gate
- Intel Iris Xe: Vulkan 1.4.309, UMA, `subgroup_size=32`, 32 KiB shared. Spec-conformance oracle. Do not special-case Intel.
- RTX 4060 Laptop: Vulkan 1.4.325, discrete, `subgroup_size=32`, 48 KiB shared.
- Lavapipe (CI): `subgroup_size=8`, 32 KiB shared, `is_uma=true`. CI exercises the mobile-warp path. LVP2 retracted.

### ORT's planner hands back interior pointers from run 2 onward
Memory-pattern planner does not engage on run 1. From run 2 onward hands back interior pointers. 52 observed, `pointers_in_guard_band=0`. Gate: `epctl --check-counters <file> --require-dispatches 1`.

### Execution counters file is the instrument for "did anything execute"
`ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE` — always-on JSON. `dispatches_executed > 0` is the only reliable indicator.

### `push_next` must rebind, never discard
`let _ = props2.push_next(..)` silently discards pNext chain. Rebind, never discard.

### First real execution: 45 ops Live, 161 nodes claimed on Phi-3.5
`ENGINE_ACCEPTS_RUNTIME_EXTENTS=true`. M0 not declared — open: validation positive control, CI lanes green.

### Performance metric is a TRIPLE
`(claimed_op_coverage, island_count, largest_island_flops)` per producer at version. Portability floor = §7.2. `SUBGROUP_SIZE_IS_GUARANTEED=False`.

---

## Session 23 — R9, the correctness gate, and reopening met criteria (2026-07-30T05:48:29-07:00)

### The event
Coordinator ran the comparison nobody had run: real 2.2 GB Phi-3.5, VulkanEP vs CPU-only, both devices.
`cpu argmax 30751` / `vk argmax 0`, top-10 overlap 0/10, `vk logits [0.0000, 0.0000]`. Output #64
(KV-cache) differs by 25.27 and is NOT zero — the session is not uniformly zeroed, the logits path is
dead. 161 MatMulNBits claimed AND accepted by ORT. `compile_calls:1, subgraphs_live:161,
compute_calls:161, compute_failures:0, dispatches_executed:161, islands:161`. Identical on Iris Xe and
RTX 4060 → deterministic logic fault. **Test suite entirely green.**

### R9 — the rule (DESIGN.md §10.0.1)
> A set of individually sound instruments can be jointly silent on the property that matters, and their
> agreement raises confidence without raising evidence. Therefore: **for every claim, name the instrument
> that would go red if the claim were false.** If no such instrument exists, the claim is not evidenced,
> however much telemetry surrounds it.

Named **the red-instrument test**. Key insight to carry forward: **confidence scales with the number of
agreeing instruments; evidence scales only with the number of falsifying ones.** A set with zero
falsifiers has zero evidential weight no matter how large. R9 gets WORSE as telemetry gets BETTER —
Switch's counter repairs made the false conclusion more persuasive. R6/R7 are defeated by
corroboration; **R9 is not** — the second device agreed and both were right about the wrong question.
Also: every instrument has a **silence set**, recorded when the instrument is added.
`test_matmulnbits.py` mentions f16 exactly twice; Phi-3.5 is entirely fp16 — the suite's silence set
contained the defect.

### §9.1.3 — `compute_failures` ruled
Execution-status counter. Licensed reading is exactly: *no dispatch reported an error it was able to
detect.* Not licensed for correctness. **Prose cannot close this** (R7: derive, do not declare; R6: a
written rationale carried a false number for weeks). Mechanism: a `model_output_equivalence` verdict
emitted next to the counters, default `UNMEASURED`; no counters summary quotable without it. **Counter
NOT renamed** — `VulkanEpCounters` is published C ABI used by epctl / probe_allocator.py / test_phi35.py;
compatibility outranks API elegance. Precedent named: `Compute` returning `null` = SUCCESS to ORT —
absence of a report IS the success report, same defect one layer down.

### Metric of record — gated, not extended (§10.0)
Triple stays a triple. Gated on `model_output_equivalence` ∈ {MATCH, DIVERGENT, UNMEASURED}, default
UNMEASURED. Rejected making it a quadruple: **a wrong answer does not discount the other three numbers,
it voids them**; a fourth column invites the trade the triple exists to prevent. Coverage 0→161 and
islands 0→161 while the model went correct-via-CPU → wrong-via-GPU.

### M0 criteria — reopened
Applied my own drafting rule to the MET rows. **M0 as written could be fully met by an EP that computes
zeros on every real model.** Defect in the criteria, not the engineering, and mine.
- **Criterion 10 ADDED** — model-level correctness at producer-at-version. NOT MET (DIVERGENT).
- **Criterion 2 REOPENED** — only correctness criterion, bottoms out in a single `Add`.
- **Criteria 4 & 5 REOPENED** — negative-space, no positive control; an always-broken EP passes both.
- **Criterion 8 relabelled** — parity only; two backends agreeing on a wrong value satisfies it.
- Criterion 7 untouched — the only M0 criterion with a falsifier from day one; now the pattern.
- **4 met / 4 partial / 2 not met**, down from 7/1/1. Said plainly: M0 is further away than yesterday,
  and none of that movement is a code regression.

### Sequencing
Criterion 10 outranks the M0 tail (Windows+Linux, software rasteriser, CI) **in order, not as a gate**.
Tail stays in M0's sentence unsoftened. Reasons: certifying a defect onto three more platforms; a green
CI lane is itself an R9 composite (adds agreeing instruments to a falsifier-free set); the engineers are
already on criterion 10. Link NOT blocked — lane bring-up is a prerequisite for running criterion 10 off
this desk. When the lanes run they must carry criterion 10's gate.

### Carry forward
- Ask the red-instrument question before quoting any number.
- `UNMEASURED` is first-class everywhere and is always the default.
- Do not rename published C ABI to fix a documentation problem.

📌 Team update (2026-07-30T05:48:29-07:00): A green suite has been shown not to imply a correct model. Phi-3.5: 161 MatMulNBits dispatched, compute_failures:0, entire suite green — vk logits all-zero (argmax 0 vs CPU argmax 30751). R9 (Morpheus): for every claim, name the instrument that would go red if the claim were false; if none, the claim is UNMEASURED. model_output_equivalence verdict required alongside all counter summaries; default UNMEASURED. Any comparison must first assert EP_NAME in session.get_providers() before calling sess.run() — failure to do so compares CPU to CPU and reports agreement. Coordinator's own first comparison reported bit-identical on both devices due to this exact error. Trinity has landed xfail(strict=True) correctness gate. M0 criterion 10 added (NOT MET: DIVERGENT). Criteria 2, 4, 5 reopened. — decided by Morpheus, Trinity, Switch, Mouse; coordinator-verified.
