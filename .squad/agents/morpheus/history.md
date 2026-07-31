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

---

## Session 24 — correctness-gated claiming (2026-07-30T06:32:18-07:00)

### The situation ruled on
`main` at `557bf24` shipped an EP that claims 161 nodes on Phi-3.5 and computes zeros. Before this
week that was impossible: the EP declined everything, so it was useless but never wrong. Coverage
work crossed a line §7.0 does not describe.

### §7.0.1 — the third category
§7.0 (frozen) contemplates ops we CANNOT run. Third category: ops we CAN dispatch and have NOT
proven. Companion rule, does NOT touch the frozen gate or §7.2:
> Evidence shortfalls degrade op coverage, not device availability, and they degrade it identically
> to capability shortfalls. An op we have not proven correct on a form is, for claiming purposes, an
> op we cannot run on that form.

### §8.9 — the ruling
**Yes, claiming is gated on proof.** Argument that decided it: a decline is LOUD (claim-rate drop,
island-count rise, CPU-fallback line, voided triple); a wrong claim is SILENT by construction (R5).
A fast wrong answer is more dangerous than a slow right one because it does not announce itself.
Compatibility ruling REQUIRES the gate rather than merely permitting it — silently-wrong output is
the most severe compatibility failure available, since the only contract a user has is "ORT computes
this graph". §1.3 said all this on 2026-07-28 and we shipped its exact failure anyway: **a prose
commitment without a mechanism is not a commitment.**

### Mechanism — `Live` stops being written down
Hand-written `OpStatus::Live` is a duplicate of a machine-known fact → R7 "derive, do not declare".
Table declares only source facts: `Staged(why)` (no kernel) / `Ready` (kernel exists). Claimability
DERIVED per form from a harness-generated **proof ledger**. No entry ⇒ decline `[unproven]` naming
the missing key. Ledger never hand-edited; regeneration check in CI. Promotion AND demotion
automatic — a `DIVERGENT` model verdict demotes every participating form. epctl JSON extended
ADDITIVELY (`status` unchanged, new `claimable` + `proof`) — compat outranks elegance.

**Proof key** = (domain, op_type, opset_bucket, every input/output dtype, kernel_variant_key incl.
code-changing spec constants, shape_class, populated_optional_input_set). The point: it makes §8.7
MECHANICAL — **an expression difference leaves the key equal; a path difference changes the key** —
so an f32 proof can never be returned for an f16 node. Lookup is by key, no judgement call left.

**Two tiers:** Tier 1 per-form op proof gates CLAIMING; Tier 2 per-producer-at-version model proof
gates REPORTING and can RETRACT Tier 1. Op-level proof would likely not have caught a defect that
reproduces at N=161 descriptors/readbacks.

### Escape hatch — C1's shape
`CLAIM_UNPROVEN` takes **a list of proof keys and nothing else**. No boolean, no `1`, no `*`. A
parser that can express "everything" MUST NOT EXIST — enforced as a test (planted `*`, `1`, bare
op-type all rejected). Default safe, requires no act. Three disclosures so no build is silently
unsafe: WARN at session creation naming keys; `unproven_forms_enabled` in counters; `epctl
--check-counters` FAILS on non-empty without `--allow-unproven`. Available in release builds —
availability is not the risk, silence is; a feature gate would fork the shipped artifact from the
tested one. Bootstrapping answered by construction: the ledger comes from the ordinary differential
run, so unproven→proven IS the dev loop.

### Cost — stated first, not afterwards
Phi-3.5 claimed count **161 → 0**. Regression in the reported number, not the code. And per my own
§10.0 gate the 161 was ALREADY void (`DIVERGENT` voids the triple). Honest number was already zero;
the ruling only makes behaviour agree with reporting. Rather an honest zero than a dishonest 161.

### M0
Criterion 11 added (ledger + the three planted controls). **4 met / 4 partial / 3 not met.** M0 got
worse twice in one day; none of it a code regression. Sequencing: **the gate goes FIRST, ahead of
fixing the fp16 defect** — fixing the kernel removes today's defect, the gate stops the next one
shipping.

### Link's lanes — precondition, precisely scoped
Split the word: **operational** (exists, executes, reports — Link may declare this without the gate;
it is a prerequisite for running criterion 10 off this desk) vs **green** (admissible as evidence /
satisfies a criterion / quoted — requires the gate). Made unrepresentable: a lane's pass condition
includes the verdict field; `UNMEASURED` reports UNMEASURED, not PASS, not FAIL. Feasibility: the
gate is the MECHANISM not the model — per-lane **gate artifact** = smallest real
producer-at-version model that claims non-zero, has an island of >=2 nodes, and exercises >=1 proof
key per dtype that lane claims. Not Phi-3.5 on a rasteriser.

### Rai
Pre-recorded independence: ruling does NOT depend on his RAI verdict. If he agrees, load-bearing
reason stays the engineering one (R6 rule 1). If he disagrees, ruling stands and the disagreement is
recorded, not compromised.

### Carry forward
- Anything hand-written in the registry that the harness already knows is an R7 fork waiting to drift.
- When designing a switch, ask whether it can express "everything" — if yes, it will.
- State the cost of a ruling on the day the number goes down, not afterwards.

### Addendum — Rai converged, and RAI-009 named a gap I missed
Rai returned RAI-008 CRITICAL (class: architecture permits silently-wrong output with no disclosure)
and RAI-007 ADVISORY (the fp16 kernel instance) — correctly splitting instance from class. Converged
INDEPENDENTLY: his load-bearing reason is autoregressive amplification (one zeroed-logit dispatch →
unbounded stream of fluent wrong tokens, indistinguishable from "bad model"); mine is the
claim/decline asymmetry + compatibility. Per R6 rule 1 the load-bearing reason stays the engineering
one. **Two independent arguments are worth more than either only because they are DIFFERENT
arguments** — R9 read the right way round: a second reading is evidence only if it could have come
out differently.

**RAI-009 was a real gap in my ruling.** §8.9.4 discloses only when the escape hatch is on; §9.1.3's
verdict lives in a counters file no user sees. Closed as §8.9.7: at session creation, one INFO line
per claimed form naming its proof key and backing ledger entry; WARN if any claimed form is
UNMEASURED; explicit INFO naming top decline codes when the EP claims zero. Same mechanism at two
severities, not a second thing to maintain. **Disclosure, not a gate — a log line is an instrument
with no red state (R9) and may never substitute for the ledger.** Folded into M0 criterion 11.

---

## Session 24 — the day the model became correct, and the failure class that was invisible to review (2026-07-30T19:05:03-07:00)

Coordinator brief: the all-zero-logits defect is fixed, partition.rs was wired (3.7x), GPU
timestamps landed, and the EP is 3.1x/3.7x slower than CPU with `model_output_equivalence = MATCH`.
Asked for: an honest M0 update in both directions, a ruling on a performance criterion, a rule for
the unwired-mechanism class, whether a correct claim can be a wrong claim, and sequencing.

### The defect vindicated the key, not the reasoning
Root cause was **binding arity**, not dtype — everyone including me reasoned toward the f16 kernel.
`push_dynamic_kernel` built a 4-entry pipeline layout for `MatMulNBits`-without-`zero_points`; the
shader wrote binding 4; **both drivers silently dropped the write.** `populated_optional_input_set`
is a component of the §8.9 proof key, so with/without `zero_points` are **different keys and a proof
of one could never be returned for the other.** The key was granular on principle before the
incident. That is the argument for granularity no amount of arguing produces — and the reason
criterion 11 does not get deprioritised now that the model is right.

### M0: six met / four partial / two not met, of twelve
Closed on a **measurement**: criterion 10 `DIVERGENT` -> `MATCH` (argmax 30751 == CPU, top-10 10/10,
max diff 0.031/0.035 — **the non-identity is the correct answer**; a bit-identical result would have
been evidence of CPU fallback). Criterion 2 closed on the promise I made when reopening it — adding
a condition after seeing the result is the fault I reopened it for, in reverse.
**Criteria 4 and 5 stay partial.** A correct model does not retroactively give an unknown-polarity
check a polarity. Criterion 3 moved **both ways**: the messenger fired on a live unplanted violation
and printed the root cause in one line (the polarity proof), which also convicts the earlier
"no errors surfaced" as a silent instrument that sat on the worst defect for its whole life.
Criterion 12 added (wiring census). First forward movement this week; the table also got longer.

### R10 — the rule
> A mechanism's existence is a claim about the **call graph**, not about the source tree. The
> falsifier for "X is wired" is an observation of an artifact X produced whose content varies with
> X's input — never a reading of X's code, never a flag its author set.

Three amendments: (1) the artifact must **vary with the input** or a hardcoded banner passes;
(2) uninvoked reports **`UNWIRED`**, distinct from empty — §7.9's third state, sixth appearance —
because *a partitioner that never runs is indistinguishable from a graph with no islands*;
(3) **the identity case is an explicit red state**. `island_count == claimed_count` was one line,
true for the defect's whole life, and nobody had written down that it must be false. Generalised:
every transform carries an assertion relating input to output, and the no-op case is red — because
**doing nothing is exactly what not being called looks like.**
Sub-rule: **wiring is per entry point, not per file** — verified in-tree, `evaluate` is called from
`GetCapability`, `retain_viable` only from `#[cfg(test)]`, while everyone believed "partition.rs is
wired". Review rule: **not complete until the reviewer has seen an artifact the mechanism produced.**
Correction to the brief: **`compute_failures` is not R10** — it IS called; it is R9's silence set.
The remedies differ (a different instrument vs the same instrument invoked) and conflating them
means writing a second instrument for a gap a correct first one already covers.

### §7.0.2 — yes, a correct claim can be a wrong claim
A claim is a **scheduling decision**, not a capability statement. Net benefit is a property of the
op **in a graph at a coverage level**. Lives in the partitioner, never the registry — a net-negative
result must never demote a row to `Staged`, that would put a graph-dependent fact in a
graph-independent place. Own decline code (never folded into `staged`/`dtype`, R8 reads that
histogram). **It is the only discretionary decline in the system**, so it is the only one needing a
guard against itself: measured per artifact, re-measured when the neighbourhood changes — SkipNorm
flipped sign with no kernel change. Hazard recorded: `GetCapability` bypasses `evaluate` for a
single cluster because "there is no competing partition" — **the competing partition is always CPU
fallback**, and that is the shape a fully-claimed graph converges to.

### Performance criterion — no, and the reasoning is the deliverable
Slowness is loud (self-reporting, monotone, we noticed 3.1x without a criterion); wrongness is
silent. Criteria exist for failures that hide. M0 is not a release; M2 already carries the criterion
the counter-argument wants. And my own drafting rule kills it: **the cheapest way to pass a ratio
criterion is always to do less GPU work.**
M0 gains a **§10.0 disclosure obligation** instead — the end-to-end CPU ratio may never be omitted,
especially above 1.0. **A figure nobody may hide is not a threshold nobody may fail**, and inventing
a gate to look rigorous would be softening in the other direction.
M1 gains: a **counter before a clock** (`command_buffer_records` < `compute_calls`; identity case
red), recording share < 5%, ratio reported. Non-gameable because **the denominator is the whole
model in wall clock — an EP that claims nothing scores exactly 1.0 and never better**, so declining
can defend a ratio and never improve one. Plus `MATCH`-only admissibility and `REGRESSED-COVERAGE`.
M2 keeps the threshold, sharpened: every device in the matrix reported, not just the winner.

### Sequencing
Tail stands, unsoftened, and is now the front of the queue. The 68.3% recording cost does not enter
M0 and does not outrank the tail — **but it was never a sequencing conflict**: it is Tank's M1
`recorded.rs` work and items 1-6 are Mouse/Trinity/Switch/Link. **Sequencing governs declarations,
not calendars** (third time I have had to say it). One hard constraint the other way: caching means
**a binding table computed once and reused**, which is today's defect generalised to every kernel —
so it lands only through criterion 10's gate.

### Carry forward
- Ask R10's question before R9's: *does it run?* precedes *would it go red?*
- When someone says a file is wired, ask which entry point.
- The no-op case of any transform must be an explicit failure state, written down on day one.
- On a good day, check which rows the good news does **not** touch, and say so first.
