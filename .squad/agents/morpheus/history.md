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


<!-- SUMMARIZED by Scribe 2026-08-01T20:39:12-07:00 -- older entries condensed below; full text lives in git history -->

### [SUMMARY] Compressed entries (condensed 2026-08-01T20:39:12-07:00)

- **📌 Cross-agent context — Round 4 (2026-07-30T02:49:12-07:00)** — ### Worktree layout and inbox portability constraint The team works in git worktrees: `squad/switch` at `C:\Users\justinchu\dev\ep-vulkan-switch`, `squad/mouse` at `C:\Users\justinchu\dev\ep-vulkan-mouse`, `squad/tank` at `C:\Users\justinchu\dev\ep-vulkan-tank`, with `main` as the integration tree.
- **Session 23 — R9, the correctness gate, and reopening met criteria (2026-07-30T05:48:29-07:00)** — ### The event Coordinator ran the comparison nobody had run: real 2.2 GB Phi-3.5, VulkanEP vs CPU-only, both devices.
- **Session 24 — correctness-gated claiming (2026-07-30T06:32:18-07:00)** — ### The situation ruled on `main` at `557bf24` shipped an EP that claims 161 nodes on Phi-3.5 and computes zeros.
- **Session 24 — the day the model became correct, and the failure class that was invisible to review (2026-07-30T19:05:03-07:00)** — Coordinator brief: the all-zero-logits defect is fixed, partition.rs was wired (3.7x), GPU timestamps landed, and the EP is 3.1x/3.7x slower than CPU with `model_output_equivalence = MATCH`.
- **Session 25 — 2026-07-30T20:58:11-07:00 — R11: the instrument that is called, correct, and misnamed** — **Dispatched four hours after R10, by a specimen R10 certifies clean.** Tank found that the 68.3% "command-buffer recording" figure — which I had built an M1 criterion on and the coordinator had broadcast to the whole team — **is upload**.
- **Session 26 — 2026-07-30T22:13:37-07:00 — A standing directive, and two VkDevices nobody chose** — **Justin's directive, recorded as standing:** 「要确保我们性能是非常高 一致向高性能推进」 — *ensure performance is very high; push toward high performance continuously.* Recorded at the head of §10 alongside the compatibility directive.
- **Session 27 — 2026-07-31T07:45:10-07:00 — The verdict that certified a run we did not execute, and the guard whose crash I called a catch** — The EP executed Phi-3.5 on the GPU today for the first time — **354 of 364 nodes in one fused island, 10 on CPU matching Mouse's declines exactly, `argmax 30751` == CPU, read from ORT's own profiler.** Persistent residency landed on bytes: **1997.6 MiB → 0.756 MiB per inference.*
- **2026-08-01T18:59:38-07:00 — §6.5 closed a conditional, and I had not said which lane armed it** — ### The report came from scoring old predictions, not from new work That is the part worth keeping.

- **2026-08-01T13:19:00 — `STEADY` is not `QUOTABLE`; R9 gets its fifth rule rather than a new number:** Ruled that `gpu_steady_tail`'s bias-in-level failure (RSD certifies a confident-but-wrong reading) is R9, not R11/R14 — a rule is what its obligations require, and R9 already obliges recording an instrument's silence, which was never done for this one. New mechanism inside R9 (Rule 5): an instrument whose confidence is *anti-correlated* with its error cannot be fixed by tightening the threshold — a tighter bound admits more of the failure; precision ≠ accuracy. Also ruled the device-clock companion requirement must state "absence is not a waiver" (else the cheapest pass is measuring on a telemetry-free platform — exactly Intel, which is also the most power-budget-exposed) and that a share-of-total criterion (recording share <5%) is satisfiable by inflating the total (board stuck at idle clock inflates device time 21×, share collapses, series looks perfectly steady). Withdrew his own claim "contention inflates host work but cannot touch the GPU clock" — false twice over (foreign GPU work inflates device-busy directly; the governor itself varies 14.8× with nothing foreign running). Ruled Switch's regime-separation rescue of Niobe's 40.201ms is unsound (a boost governor ranged 210→2490MHz within one run — "regimes don't overlap" was really "the two clock states I sampled don't overlap") but the figure is *re-qualified* not withdrawn: every catalogued perturbation has a non-negative sign on time, so it survives as a sound upper bound — withdraw and re-qualify are different outcomes. Named a fifth instance this week of "confirmed a hypothesis and stopped" (solo/hog agreed 0.08%) — mechanical fix adopted: when a check agrees with me, ask which way it moves if I'm wrong.
- **2026-08-01T18:59:38 — §6.5 closed a conditional nobody had said which lane armed:** Scoring six old standing predictions against artifacts (no new measurement) found one refuted by a *third state*: predicted `SHARED` xor `SPLIT-DEVICE`, counter returned `OFF` — a binary prediction met by a third token is an unavoidable refutation, the whole payoff of the multi-state-token discipline. Ruled the flag's OFF default is not a re-decided default: its documented reason ("seam not yet filled") had expired (the seam was filled) and nobody went back to re-check it — R12's third generalisation: counter→device, verdict→executor, rationale→**date** (a default defended by an expired reason is indistinguishable from one still needed). Confirmed `authoritative=0` in Niobe's indexspace artifact was a real, contingent zero (type discipline + companion counters + ceiling arithmetic all agreed) but the *probe's extract* had dropped the two keys that make the zero readable — R11 appearing in a **selection** rather than a name (now seen in a phase table, a ledger, and a probe).
- **2026-08-01T20:39:12 — the phantom key is R13, not R11:** `probe_sec65.py` requested a counter key (`alloc_device_spans`) that never existed anywhere in source; the defaulting `.get(k, '<absent>')` printed a plausible-looking absent value on every run without ever raising. Ruled this is **R13** (two opposite diagnoses, one silent token) not R11 (R11 needs two relata — a reported quantity vs. its name; this is a one-sided reader-side request with nothing on the writer's side to compare against) — R13 amendment 1: the "dangling reference" class — any defaulting lookup (`dict.get`, `unwrap_or`, `?? fallback`, `getattr`) where the key set is knowable and absence is mis-read as a value. Ordered a key census (Tank+Niobe) using the existing 5-state vocabulary rather than minting a 6th. Explicitly declined to reopen M0 criterion 12 over this (a bad probe isn't grounds for a new milestone obligation). Also caught a defect in Niobe's own `span_accounting()`: `NOT_A_NUMBER` fired on the same defaulted-value confusion it was meant to classify.
- **2026-08-01T22:02:39 — the anchor exemption is the deciding term, §5.4.1:** Verified in `partition.rs:475` that the anchor exemption is an *early return* placed above the byte-estimator's economics arithmetic — so on any anchor-bearing island (MatMul/Gemm/Conv/Attention/MatMulNBits/GQA/MHA/QMoE/etc.) the economics model isn't outvoted, it's never evaluated. This is design working as intended (every real transformer island has an anchor), not a scandal — but ruled three genuine problems: (1) the exemption's warrant is asserted not measured, live falsifier is a small-MatMul/edge-shape graph; (2) R9's silence-set rule now generalises to a *policy term*, not just an instrument (the exemption's silence covers "the byte estimator is broken" — it was, by 104,116×); (3) `Verdict::Claim` is R11 at the value level (one field wearing three findings). Named the "cheapest satisfaction" drafting-rule's second live example: an unconditional early return *inside* a gate satisfies "always evaluated, no branch in front of it" (RAI-011) while being exactly the thing RAI-011 exists to catch — the two are observably indistinguishable.
- **2026-08-01T22:25:29 — the estimator's first half, and declining a false concurrence:** Mouse's fix (internal island edges no longer charged to the boundary) cut the byte-estimator error 89.2GB→13.9GB (6.4×); declined to read this as the economics arm *independently concurring* with the anchor exemption, since both were fed the same fabricated input — agreement between two things sharing one bad input isn't a second opinion. Found something stronger instead: `transfer_ns` is monotone in bytes and the gate claims even at the inflated 13.9GB, so it claims *a fortiori* at the true measured boundary (856,720B) — a bound survives a 16,268× adversarial inflation of the term opposing it. Renamed `MEASURED_PHI35_DEV0` (still an estimate wrong by 6.4×) vs. `MEASURED_PHI35_DEV0_REAL_BYTES` (the actual measurement) so the name stops outliving the doc comment's disclosure. General rule minted: **a criterion is discharged by an observable that changes when the claim is false, never by one that's true whatever happens.**
- **2026-08-01T23:36:43 — ratifying the coordinator's merge; two generalisations:** Row 11 (ledger criterion) kept Morpheus's "not yet MET" over Mouse's "MET" — a row closed by the agent who supplied the artifact, in the change that supplied it, is an identity whose two sides share one source; recorded explicitly that Mouse's evidence is neither rejected nor lost, only the *tally* is withheld, never the work. Row 12 (census criterion) kept Mouse's over Morpheus's stale text, with reconciling language: a census answers whether a mechanism ran, a criterion answers whether a claim is falsifiable — neither substitutes for the other. Minted **R9's third generalisation**: a criterion is discharged by an observable that changes when false, never one true regardless (three same-day specimens: RAI-011's early return, Link's screen on an undefined variable, Switch's 0.0==0.0 assertion). Named the **dangling-reference class** formally under R13 amendment 1 (phantom key + Morpheus's own stale line-number citation + Link's undefined-variable screen are one failure; a broken URL fails loudly and is merely broken — the class is references that *resolve anyway*). Found his own §10.0 `attribution_witnesses` doc example was stale (showed 2 keys, record now emits 6) — regenerated from a real artifact rather than from memory.

📌 Team update (2026-08-01T17:16:56-07:00): Intel device-clock figures are permanently uncertifiable on this hardware (`none_available`, no producer exists and none of the available proxies are the right kind of quantity) — attack the Intel/NVIDIA residual with counts and shapes, not clocks — decided by Niobe

📌 Team update (2026-08-01T17:16:56-07:00): All wall-clock figures remain withdrawn; only counts, bytes and certified-companion device-clock figures are quotable — decided by Switch, Morpheus, Niobe, Link

📌 Team update (2026-08-01T17:16:56-07:00): `ledger_lookup` is the last `UNWIRED` mechanism in the instrument census (criterion 11); Mouse is building it — decided by Trinity, Mouse

📌 Team update (2026-08-01T20:39:12-07:00): Link found the layering lint (`tests/layering.rs`) scopes to `src/ops/` only — planting `use ash::vk as _;` in `src/ops/norm.rs` reds it, but the identical line in `src/trace.rs` passes all 26 of its tests. The archived decision that placed timestamp arithmetic in `trace.rs` specifically to stay "on the right side of the layering lint (no `ash`)" was justified by a rule that does not exist. That archived rationale is invalidated by this finding — it was never wrong to put the arithmetic in `trace.rs`, but the stated reason for doing so was never true. — decided by Link

---

## 2026-08-02T01:42:02-07:00 — Restoring one sentence, and declining a rule I would have enjoyed minting

**The restoration.** A merge dropped the sentence I most wanted kept from the criterion 11 ruling —
what is withheld is the tally, not the work. The coordinator declined to splice my prose and asked
me to re-add it, which is right. I put it back and went further than three lines: I named the three
of Mouse's constructions that meet the standard I set, in the row itself rather than in a decision
file nobody re-reads.

Two of them are better than what I asked for. "Absent is treated exactly like zero, and a quoted
count exactly like absent, because a writer that stringified its counters did not read a counter" —
I did not think to require that. And "NeverAttempted is derived and never counted, since recording
it would be a lookup, which is exactly what it asserts did not happen" is the cleanest statement of
R13's instrument/subject boundary anyone here has written, including me.

I also wrote down that a lead who can only ever withhold is running a different instrument from the
one he thinks he is. I need that sentence more than the team does.

**The decline.** The coordinator brought me his own error: having named "union defects" as a
pattern, he read a clippy run into it and reported four cases that were not. Mouse checked each
against origin/main and found four of five predated the merge entirely.

It is a real failure mode and I would have enjoyed minting it. It is R13's second clause with
nothing added — quote the failure text, never the failure count — and Mouse's remedy was that
clause performed literally. So: declined, with the citation, plus one sentence for the genuinely
new scope, which is that the mis-reporting mechanism here is a person rather than an instrument.

A newly named pattern begins attracting cases that do not belong to it. Writing that down while
also declining to give it a number is the only self-consistent way to write it down at all — a
register that grows by one entry per named pattern is a register attracting cases to its own new
categories. I said so in the text.

**The generalisation I did take.** Copy-Item preserves LastWriteTime, so cargo does not rebuild
after a restore-from-backup and the mutation harness re-runs the mutated binary. Mouse nearly
"fixed" the resulting persistent false failure by weakening a correct assertion — the most
expensive outcome available. Together with his earlier contaminated build from the shared
worktree, that is two specimens of one thing: for a test result the frame is the binary that ran
it. R12's fourth generalisation, remedy unchanged.

The detail worth keeping is that the failure arrived disguised as the thing we most want. A check
that goes red is the scarce good on this project, and the one we are least likely to interrogate.

**Carry forward**
- Row 11 closes on (c), Trinity's, and on nothing else. Not by me, not by Mouse.
- The mutation-harness rebuild assertion is owed by whoever next writes one; it is a
  cross-platform note, not a Windows anecdote.
- The bound's narrow half is now a test with both polarities mutation-tested. Prose falsifiers
  should become tests wherever they can; I should stop being pleased when one stays prose.
- Register status after this session: two amendments, three generalisations, three rules declined.
  If the next finding also lands as a generalisation, check whether I have simply found a softer
  way to decline.

## 2026-08-02T02:02:23-07:00 — Criterion 10 closes on the bar I set in advance; criterion 12 does not close on a witness

The coordinator brought two things: a correction to his own reporting, and evidence that criterion 10's
advance-stated closing condition is met. He explicitly declined to close the row himself, on the grounds
that he produced the artifacts, and offered to have it re-run by someone else.

**I closed it, and I overruled his objection to his own evidence.** I verified the artifacts myself first:
both devices MATCH/AGREE, three consecutive runs of one session, per-run all AGREE, executed_by showing
3 VulkanExecutionProvider island executions against 24 CPU from ORT's own profiler, both attribution
witnesses present and agreeing, dispatches 1066/1186, argmax 30751 every run, and cross_run_identical_to_run1
true on all three — which is precisely the cross-run divergence that reopened the row.

The independence objection does not apply. The shape I have refused all session is the party who supplies
the artifact also moving the tally. He supplied and declined to close; the verdict logic is Trinity's; the
attribution instrument is not ours; the tally is mine. The separation is already where it needs to be.

The part I want to remember: **I wrote that condition in advance specifically to bind me, and it binds me
when the news is good.** Adding a re-run requirement after seeing a passing result is hardening a criterion
because it is about to pass — the exact mirror of the rescue argument I rejected on the 40.201 ms figure,
and it is no better for pointing in the direction of rigour. I recorded the re-run as a standing falsifier
instead, which costs nothing and keeps the row falsifiable after closure.

I also fenced the closure: Defect 2's KV write path and the arena-lifetime item are NOT covered. Folding
them in would have been the new condition I promised not to add; dropping them would have lost them. They
keep their own owners. And I recorded Switch's and Trinity's delivery in the row, because the row was
reopened on their work being incomplete and a reopening reason that vanishes silently is not a record.

**Criterion 12 stays open.** He had told the team it was closed, having run the census himself and got
`unwired: []`. That is a witness. The row is a conjunction of four: census, declared extent, the
decomposition identity against an independently-measured whole, the name-content check. Three are open.
I enumerated them in the cell — a conjunctive criterion whose parts are only recoverable from prose invites
being closed on whichever part the reader happens to be holding.

Diagnosis: R11's first obligation, turned on the reader. Declare the extent of what you are reporting. One
conjunct verified, the conjunction reported — a decomposition presented as closed, R11's own sentence
arriving in a status report rather than a measurement. His own aggravating form is the better sentence and
I kept it: *the thing I verified myself was the thing I over-weighted.*

**No rule minted. Second decline tonight, third this session.** That is now four declines against two
amendments and three generalisations, and my earlier self-check applies: if the next finding also lands as
"an existing obligation, walked past," I should check whether I have found a comfortable way of never being
wrong about the register's shape. I do not think that has happened yet — the remedy here genuinely is
R11 obligation 1 and I can point at the sentence — but the tally is the kind of thing one notices too late.

**Carry forward:**
- Criterion 10 is MET; its standing falsifier is the next independently produced artifact. If it diverges,
  reopen the same day with no argument from me.
- Criterion 12 needs (ii) extent, (iii) the decomposition identity, (iv) the name-content check. The
  16,268x boundary-byte residual is a live instance of (iii); `MEASURED_PHI35_DEV0` is an outstanding
  specimen of (iv), and Mouse's rename is what closes that one.
- Two instances in eight hours of witness-vs-discharge from the same person, in opposite directions
  (over-closing 12, under-claiming 10). The remedy is enumerated conjuncts in the cells, not a new rule.
  If it recurs a third time, the defect is in how the table is written, not in who reads it.
- I have now twice declined to grow the register in the same session I amended it twice. Watch for a
  softer way of declining.

## 2026-08-02T04:30:29-07:00 — I reopened criterion 10 three hours after closing it, and the closure was my error

The coordinator put my closure to Fact Checker in Devil's Advocate mode, precisely because he had
supplied the evidence. Fact Checker found that `model_output_equivalence` compares one output out of
sixty-five. I verified it in source before ruling: `_compare_run_to_cpu` takes `vk_out[0]` and
`cpu_out[0]` and derives argmax, top10 and max_abs_diff from the logits alone. `test_phi35.py` is the
same shape behind a structural length assertion. Nothing in the tree compares a KV output to CPU.

The all-65 gate is `outputs_bit_equal` — cross-run identity. It proves determinism. It cannot prove
correctness, because a deterministically wrong write passes it by being consistently wrong.

**The thing I found that neither of them brought, and it is what settled it for me:** `test_phi35.py`
Guard 1 already documents this exact mechanism in this codebase — an output outside the descriptor set
"is never written... zero-initialised by both Intel Iris Xe and NVIDIA drivers for security, reads back
as all-zero" — and the guard built against it is applied to output 0, the one tensor that already has an
oracle. The row was reopened on 50 KV outputs never written, where the symptom was cross-run divergence.
Divergence is the symptom of a *dirty* arena. On a clean one the same defect is stable and everything is
green. So the closure certified that the symptom is gone and never established the defect is fixed.

**I refused the escape I was offered.** "The criterion's words were always about logits" would require
renaming the measurement to `logits_equivalence` after seeing that the broad reading fails. That is
narrowing a criterion because it has just failed — the exact mirror of what I refused three hours ago in
the same cell when I declined to add a re-run requirement because the news was good. I wrote the
symmetric form into the row: **a criterion may not be hardened because it is about to pass, nor narrowed
because it has just failed.** If I had taken the escape it would have been the cheaper ruling and it
would have been the same failure I have been grading other people on all session.

**My own error, recorded plainly and in the register rather than only in the table.** The artifact carries
`outputs_compared: 65` in the same per-run dict as `argmax_cpu`, `top10_overlap` and `max_abs_diff`. Every
neighbour is an oracle fact; that one is a cross-run count. I read 65 and understood sixty-five oracle
comparisons, and I quoted `max_abs_diff = 0.0625` into a criteria row without stating over what — R11
obligation 1, three hours after I diagnosed the coordinator for that same obligation in criterion 12. I
wrote "the thing I verified myself was the thing I over-weighted" into row 12 and then did it in row 10.

**No new rule. Recorded as a fourth specimen under R9's red-instrument test, deliberately unnumbered.** I
ran the self-check I put in the register — if the next finding also lands as a generalisation, look for a
softer way of declining. The remedy here is R9's remedy unchanged, a different instrument, so it is not an
amendment and not a generalisation. The content that is genuinely new is written anyway: two gates whose
extents differ compose to the weaker extent and the stronger name; a record with two gates owes two
extents.

**Drafting rule applied to the remedy.** The coordinator's proposal was right and I sharpened it by asking
what the cheapest satisfaction is: an all-65 oracle is satisfied perfectly by 64 pairs of all-zero tensors,
so the non-triviality guard is not optional — an oracle that passes on the absence of data is Switch's
`0.0 == 0.0` in a fourth costume. And the planted control must be wrong *and stable*; an unstable plant is
caught by cross-run identity and proves nothing new.

**On method, the sentence I want to keep.** I verified every field of that artifact and closed wrongly
anyway. The coordinator arranged an adversary because he had supplied the evidence, and the adversary
found what my verification did not. Content verification by the ruling party is weaker than adversarial
review by a party with no stake. My standing-falsifier clause fired in three hours, which is the clause
working, not the clause being circumvented.

**Carry forward:**
- Criterion 10 discharge is now four arms, stated in full so nothing can be added later: all-65 oracle with
  justified per-output tolerances and two named extent keys; a wrong-and-stable planted control (all-zero);
  a non-triviality guard on both sides; existing attribution re-emitted not re-argued.
- Owners I named: Trinity (comparison/verdict constructors), Switch (whether the KV write path writes at
  all — still unwitnessed, and the `kv_cross_run` prediction has been UNSCORED all session).
- Fact Checker's session-aggregate attribution argument is OPEN and NOT a condition. It needs an artifact:
  plant a failing island execution, observe whether a Node event is still emitted.
- `outputs_compared` is a live R11 obligation-4 specimen and belongs in criterion 12's conjunct (iv)
  alongside `MEASURED_PHI35_DEV0`.
- Four declines now, and this one I checked hardest because reopening was the ruling I wanted.

## 2026-08-02T15:15:12-07:00 — The accumulation question had a false premise; the residual is ULPs

I was asked to make a cost ruling: should f16 kernels accumulate in f32, knowing it would invalidate
all 74 freshly re-proved ledger entries. Mouse declined to make it himself and the coordinator was
right to honour that boundary.

**I checked the kernels before ruling on the economics, and every f16 kernel already accumulates in
fp32.** `q_gemv.comp` says so in terms — "Accumulation is fp32 regardless of storage, which is also
what ORT's SQNBIT_CompFp32 path does". Both layer-norm f16 kernels say it. `gqa_f16.comp` declares
`float acc[128]`, `float dot`, and runs the online softmax in float, converting on load. fp16 is a
storage format in this EP and never was an accumulation format.

So the ruling is: no change, no cost, no ledger invalidation. **Had I reasoned about the trade as it
was posed — registers, occupancy, bandwidth-bound decode at ctx 0 — I would have produced a careful
and completely wasted analysis, and authorised a real cost to obtain a property we already have.**
That is the lesson I want to keep from this one: the question arrived with an embedded factual claim
and the claim was the thing to check first. I nearly reasoned about the economics because the
economics were what I was asked about.

**Then the residual.** I dumped all 65 per-output diffs. Sixty-four are exact negative powers of two
and the sixty-fifth is 3 x 2^-9. They are small integer multiples of the fp16 ULP. KV activation
magnitude grows with depth in a transformer, the ULP grows with it, and the absolute residual rises
with depth for a perfectly correct implementation. **The "monotone accumulation curve" is a plot of
tensor magnitude.**

So the tolerance argument is wrong for a reason nobody in the thread had reached: atol is an absolute
bound applied to tensors of growing scale. Section 10.0.4's "prefer the ratio", arriving as a defect
rather than as advice. The unit is wrong, not the number — and I made a point of writing that fixing
the unit may make the gate *tighter*, because otherwise this ruling reads as a relaxation and it is
not one.

I gave the replacement a prediction before it exists: residual in ULPs, predicted flat at 1-3 across
all 32 layers. Flat means no defect. A step means a located defect. It is better in both outcomes and
unlike the current gate it can be wrong.

**A correction to the finding I had to make carefully.** Mouse said the curve rises monotonically. The
absolute series broadly does; max_rel_diff does not — layer 2's key is 0.4559, above every layer from
3 to 30, on an unremarkable absolute residual, because max_rel is attained at near-zero elements. He
had just corrected one wrong denominator in his own instrument and there was a second one left. I
tried to say this in a way that credits the correction he made rather than scoring the one he missed.

**R9's dual, and I declined to number it again — fifth decline, second consecutive unnumbered
finding.** I ran the self-check in the open this time because two in a row is exactly when I should
suspect I have found a comfortable way of never growing the register. My honest conclusion: the dual
is the same rule with the same remedy. Everything this session was a reading that does not move when
its subject is wrong; this is a reading that moves when its subject is fine. Both are readings of
something other than the claim. That the *dual* keeps arriving argues R9 is the home, not that a new
number is owed.

**I declined the comfort of argmax=30751 and top10=10/10, and said the reason is arithmetic rather
than scepticism: it is one token.** The coordinator asked directly whether it was reassuring or
misleading and said he did not know which. It is neither. The rank invariant is the right invariant —
10.0.4 says so — and N=1 is not a stated N.

**Carry forward:**
- Criterion 10 stays open on the unit alone. Its reopening ground (all-zero KV) is measured absent:
  degenerate 0, 65/65 compared, planted control refuses. verdict=DIVERGENT is honest and must not be
  flipped by moving atol.
- The ULP series is the closing artifact. Prediction on record: flat, 1-3, all 32 layers.
- GQA's 1.37x margin is untouched by this ruling and stays open. Its proposed remedy turned out to be
  already in place, which is not a reason to close it.
- The CPU EP is not ground truth; it is a second fp16 implementation. Elementwise disagreement between
  two correct fp16 implementations grows with depth by construction. Worth repeating to anyone who
  frames this as "accumulated error".
- Five declines now. If a sixth arrives I should ask someone else whether the register is under-growing.
