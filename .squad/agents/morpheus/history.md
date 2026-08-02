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

## 2026-08-01T13:19:00-07:00 — `STEADY` is not `QUOTABLE`, and the register did not need to grow

### I was asked whether this is R11 and I had already decided it was
That is the part to keep. I wrote the brief arguing R11, and R11 is wrong — because R11's *epigram*
fits and R11's *obligations* do not. Run them against `gpu_steady_tail`: no decomposition, no flat
table, no inclusive parent, and name-content agreement **passes** — "RSD over the steady tail" is
exactly an RSD over the steady tail. **All four certify the specimen.** That is the same test I used
on 2026-07-30 to refuse folding R11 into R10; it was right then and it disqualifies my own reading
now. **A rule is what its obligations require, not what its best sentence suggests.**

Nor is it R14. It is **R9**: bias in a series' level sits in a dispersion statistic's silence set,
and R9 already obliges us to record an instrument's silence when we add it. We never did for
`gpu_steady_tail`. **The register individuates by remedy** — R10 observe invocation, R11 an
independent whole, R12 frame identity, R13 three tokens, R9 a different instrument — and this remedy
is already spoken for. A second name for one failure class is two names for one measurement,
appearing to close.

What *is* new is a mechanism inside R9. R9 describes plural instruments **jointly silent**. This is
one instrument whose confidence is **anti-correlated with the error**: the further the level is from
truth, the steadier the device that produced it. **Silence is neutral; this is worse than silence**,
and the consequence R9 did not state is that **you cannot fix it by tightening the threshold — a
tighter bound admits more of the failure.** Rule 5. And: precision is not accuracy, and this
register had never had to say so.

### The cheapest satisfaction, asked three times and it paid three times
Switch's companion requirement was right and each tightening came from my own drafting question.
Stated as a *tool*, it binds NVIDIA and exempts everyone else. Stated without "absence is not a
waiver", **the cheapest pass is to measure on a platform with no telemetry** — and the Intel iGPU,
which shares its power budget with loaded CPU cores, is the platform most exposed and most rewarded
by that loophole. And criterion 5 — recording share below 5% — has a live attack I could not have
seen a day ago: **run on a board stuck at idle clock.** Device time inflates 21x, host recording
does not, the share collapses, the series is perfectly steady, every gate goes to its most confident
verdict. **A share-of-a-total criterion is satisfiable by inflating the total.**

M1 needed no restating. **Criteria 1, 2 and 4 were untouched, and that is the finding** — bytes and
counts, the only criteria that survived a week in which every timing figure was withdrawn twice.

### I withdrew a sentence of mine, and it was load-bearing
*Contention inflates host work but cannot touch the GPU clock.* I said that when Niobe and I moved
the performance criteria onto the device clock. It is false twice: foreign GPU work inflates
device-busy directly, and the board's own governor varies it **14.8x** with nothing foreign running.
The device clock was a better surface than wall clock and I treated "better" as "immune". **There is
no third surface to retreat to now, and that is the honest state.**

### The rescue argument, and the asymmetry inside it
Switch's regime-separation rescue of Niobe's 40.201 ms fails, and it fails on his own evidence: the
board ranged **210 -> 2490 MHz within a single run**. A boost governor is continuous. "The two
regimes do not overlap" is *"the two clock states I sampled do not overlap"* promoted to a claim
about the device. Also: the margin protecting 40.201 is **6.1x, not the 21x quoted**, and it sits at
the top edge of the band; and the rescue argues about clock while contention inflates continuously
and that run has no tenancy verdict.

But the figure is **re-qualified, not withdrawn** — every perturbation we have catalogued has a
non-negative sign on time, so it is a sound **upper bound**, and deleting it would be hardening a
criterion to punish a bad week exactly as certifying it would be softening one. **Withdraw and
re-qualify are different outcomes and I have been sloppy about the difference.**

The thing I want to remember: Switch held **his own** before/after to the strict standard (⛔, "and
probably sound is not the standard") and accepted a rescue for **Niobe's** figure that he did not
hold his own numbers to. Generous instinct, real asymmetry — and the same one I have, pointed the
other way, when I audit my own work harder on a bad day. **Asymmetric standards are invisible from
inside them.**

### Fifth time this week
I confirmed a hypothesis and stopped — solo and hog agreed to 0.08%, so the instrument held. The
count is now five, and the pattern is always the same shape: **the confirming result gets no second
look precisely because it agreed.** R13's second clause is mine and I keep failing it. The
mechanical form I will actually use, because attitude does not survive being tired: **when a check
agrees with me, ask which way it moves if I am wrong.** Ten seconds, and it would have caught this
one, the Guard D one, and the phase table.

### Carry forward
- A rule is what its obligations require. Run them against the specimen before assigning it.
- Ask which way a check moves when its subject is wrong. If it moves with me, it is not evidence.
- Any statistic of *shape* is silent about *level*. Write the silence down beside the instrument.
- Two bounds on the same side do not bound a difference. Declare the sign, every time.
- Withdraw != re-qualify. An upper bound is worth keeping and is not a certification.
- Prefer the invariant that survives the contended machine — and do not hand the reader a count and
  let them supply the clock.

📌 Team update (2026-08-01T17:16:56-07:00): Intel device-clock figures are permanently uncertifiable on this hardware (`none_available`, no producer exists and none of the available proxies are the right kind of quantity) — attack the Intel/NVIDIA residual with counts and shapes, not clocks — decided by Niobe


📌 Team update (2026-08-01T17:16:56-07:00): All wall-clock figures remain withdrawn; only counts, bytes and certified-companion device-clock figures are quotable — decided by Switch, Morpheus, Niobe, Link


📌 Team update (2026-08-01T17:16:56-07:00): `ledger_lookup` is the last `UNWIRED` mechanism in the instrument census (criterion 11); Mouse is building it — decided by Trinity, Mouse


---

📌 Team update (2026-08-01T20:39:12-07:00): Link found the layering lint (`tests/layering.rs`) scopes to `src/ops/` only — planting `use ash::vk as _;` in `src/ops/norm.rs` reds it, but the identical line in `src/trace.rs` passes all 26 of its tests. The archived decision that placed timestamp arithmetic in `trace.rs` specifically to stay "on the right side of the layering lint (no `ash`)" was justified by a rule that does not exist. That archived rationale is invalidated by this finding — it was never wrong to put the arithmetic in `trace.rs`, but the stated reason for doing so was never true. — decided by Link
## 2026-08-01T18:59:38-07:00 — §6.5 closed a conditional, and I had not said which lane armed it

### The report came from scoring old predictions, not from new work
That is the part worth keeping. Six standing predictions scored against artifacts: three confirmed,
one UNSCORABLE for frame mismatch, one UNSCORED for having no artifact, and **one refuted by a third
state**. No new measurement was needed. **Scoring what we already said against what we already have
found something that six agents running instruments did not.** I should schedule that, not wait for
it to happen.

### The instrument declined to pick one of my two options
The prediction was `SHARED` xor `SPLIT-DEVICE`. The counter returned `OFF`. Both `OFF` and
`SPLIT-DEVICE` are "not SHARED", and if we had ever collapsed them the prediction would have scored
a **clean pass** and the scope gap would still be invisible. **A binary prediction met by a third
token is a refutation you cannot talk yourself out of** — and that is the whole return on the
family discipline (`UNMEASURED`, `UNWIRED`, `UNOBSERVABLE`, `SPLIT-DEVICE`, `UNATTRIBUTED`, `OFF`).
Every one of those tokens costs an argument at the time it is added. This is what they buy.

### Neither of the two options I was offered was the answer
Asked: intended, or the `offer_shared_device` gap? Neither. **Intended — and its recorded reason has
expired.** The source says the transfer "cannot be written until the handle->VkBuffer seam is
filled"; the seam is filled, `CreateDataTransfer` is registered, armed sessions complete on the real
model. **The condition the switch was waiting for was met and nobody went back to the switch.**

R12 with a **date** as the frame. Third generalisation of that rule now: counter -> device, verdict
-> executor, rationale -> date. And it has `retain_viable`'s shape exactly: **a default whose stated
reason has expired is indistinguishable from one still needed.**

I did **not** rule that the flag should flip. There is a live reason for OFF that the source does
not give — it buys host memory wearing a device handle, risk with no measured benefit. But **a
default defended by a reason its own documentation does not give is a default nobody has
re-decided**, and saying "it is probably still right" is the move I ruled against yesterday.

### The zero was fine, and the artifact that showed it was not
`authoritative=0` with `backed=9, evaluations=9` looked like R12 and is not: the counter is an
**int** not a string (three-state type discipline answering before it was asked), its unconditional
twin moved 9 (so it is nine measured negatives), and `ceiling = backed - staged = 0`, so zero is the
only value it could take. **A zero at a zero ceiling is contingent; UNOBSERVABLE would be a stronger
and false claim.**

But the *probe's extract* dropped `alloc_staged_spans` and `alloc_device_authoritative_ceiling` —
the two keys that make the zero readable — so a careful reader **correctly** could not tell. That is
**R11 in a selection rather than in a name**: a set of numbers published as though it closed. The
counter was honest and the artifact was not, and I have now seen that failure in a phase table, in a
ledger, and in a probe.

### Carry forward
- State a closure **with its lane**. A closure without its lane is a different sentence, not a
  shorter one.
- Score predictions only against artifacts from the lane they described. Wrong lane and no artifact
  are both non-passes. **The denominator never shrinks to flatter the numerator.**
- When a precondition lands, go back and re-read the sentence that named it. Reasons expire.
- Ask of every probe: does its extract contain the keys that make its own numbers interpretable?
- Never collapse two "not X" states into one. The third token is where the refutation lives.

## 2026-08-01T20:39:12-07:00 — The phantom key: R13, not R11, and the census I did not want to owe

Third ruling today, and the second in a row where the coordinator brought me a finding
pre-diagnosed and asked me not to mint a rule to reward it. Good instinct; wrong diagnosis,
narrowly.

**The specimen.** `bench/results/probe_sec65.py:89` requests `alloc_device_spans`. I grepped the
whole repo: the string occurs exactly once, at the line that requests it. No emitter, never was
one. The read is `data.get(k, '<absent>')`, so it has printed `'<absent>'` on every run since it
was written and nothing has ever thrown.

**Why not R11, which is where the coordinator put it.** I ran my own individuation test: a rule is
what its obligations require. R11's four obligations cannot even be *evaluated* here — extent of
what, no parts to decompose, no table, and name-content agreement needs content. R11 governs a
reported quantity, on the writer's side of the artifact. This is a request, on the reader's side.
A mismatch needs two relata and this has one. A name that means nothing is not the extreme case
of a name that means the wrong thing; it is a different failure on the other side of the seam.

**Why R13.** His own sentence is R13 verbatim: two opposite diagnoses with opposite fixes, one
token. R13's costume, R10's face. And everything that makes it frightening — longest latency in
the register, the hole *filled* rather than left open, the look of evidence of absence — follows
from the token, not from the name. Three tokens would have caught it on run one.

**What is new is the surface, so: amendment 1, the defaulting lookup.** Every prior R13 specimen
failed loudly and was mis-rendered downstream. This one has no exception anywhere, manufactured
by a construct whose whole purpose is not to fail. `dict.get`, `unwrap_or`, `?? fallback`,
`getattr` — where the key set is knowable, the default is not a value and absence is not a
reading.

I wrote the not-minting-R14 paragraph explicitly, because I declined a rule yesterday too and a
habit of declining is its own defect. Remedy-identity cuts both ways.

**The key census.** Two tiers, runtime and static; exact string match; owner Tank with Niobe,
importing `audit_instruments.py`'s five states rather than minting a sixth vocabulary. I named
four cheapest-satisfactions — the fuzzy matcher is the one that worries me, because
`alloc_device_spans` is one word from `alloc_device_backed_spans` and a lax matcher would
*certify the specimen*. Planted-phantom positive control required. I explicitly did **not**
reopen M0 criterion 12: no milestone claim rests on `probe_sec65.py`, and bolting a probe
obligation onto a milestone because a bad probe turned up today is hardening a criterion to
punish a bad week.

**Niobe's `span_accounting()`.** Upheld the report-without-judging call — after `gpu_steady_tail`
the case against letting describers move verdicts writes itself. But "feeds no check" is not "has
no teeth", so I gave it attachment instead of authority: the classification travels in the same
artifact as every span count it describes, per the `executed_by` lesson.

**And I found a defect in it.** `NOT_A_NUMBER` fires on `not isinstance(auth, int)` while the
extract still reads `data.get(k, "<absent>")` — so a phantom or missing key lands there and is
described as *"a string state and not a count; the type is the answer"*. False, and reassuring
in exactly the wrong direction. She inherited the defaulting read; it is not hers to carry. It is
the whole argument for fixing the lookup rather than the classifier: one fix at the defect site,
or N fixes plus a new one every time someone adds a consumer.

Three sightings in one day of an instrument-side absence rendered as a subject-side state. That
recurrence is why I signed the obligation. Any one of them alone would have been an anecdote.

**Carry forward**
- Key census is Tank's, with Niobe. Watch for it landing static-tier-only, or with a fuzzy matcher.
- `alloc_device_spans` must be classified wanted-and-non-existent vs typo *before* deletion.
- `NOT_A_NUMBER` must split until the census lands; unresolvable key is `ERROR(instrument)`.
- Niobe's `71610cd` still awaiting merge; my ruling assumes it lands.
- Still owed from earlier today: Tank + Switch re-justify `ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY`
  default-off by M2 entry; Link's device-state survey for non-NVIDIA platforms.
- I have now declined to mint a rule twice and amended twice. If the next finding also lands as
  an amendment, check whether I am protecting the register's shape rather than reading it.

## 2026-08-01T22:02:39-07:00 — The anchor exemption is the deciding term (§5.4.1)

Fourth ruling today. The coordinator brought this as a §7.12 finding and I think it is a §5.4
finding — which is to say it is mine, not Mouse's.

**What I verified before ruling.** `partition.rs:475`. The exemption is an *early return*, above
`transfer_ns` and `compute_ns`. So on an anchor-bearing island the economics arithmetic is not
outvoted, it is not evaluated. Stage 3 is a constant function on our graph: nothing about the
island can change its answer. That is a sharper statement than "the predicate claimed it via one
arm" and it is the one the artifact supports.

Then I read `is_anchor` — MatMul, Gemm, Conv, ConvTranspose, Attention, MatMulNBits, GQA, MHA,
QMoE, LinearAttention — and the diagnosis inverted. Every non-trivial transformer island contains
an anchor. So "the economics model does not decide our partition" is not an accident of Phi-3.5
and not a defect. It is the design working. 3c was written to kill anchor-free elementwise
scatter and our island is not that. The doc comment says so.

I had to resist writing this up as a scandal. It is not one. But three things are genuinely
wrong and I wrote them without inflation:

1. The exemption's warrant is asserted, not measured, and is now the sole term deciding every
   production partition. Falsifier is a *future* exposure: a small MatMul inside large boundary
   traffic. Small models, edge shapes. Generality is the constraint I am told to check
   continuously and this is where it bites.
2. The exemption's silence set includes "the byte estimator is broken." The 104,116× is why 3c
   declines us when allowed to answer. R9's silence-set rule applies to a *policy term*, not only
   to an instrument. That generalisation is the part of this ruling I expect to reuse.
3. `Verdict::Claim` is three findings wearing one name. R11 at the value level. Mouse's fix is
   right and his refusal to re-derive the arm at the call site is righter — that is RAI-011
   reappearing inside the fix for its own sibling.

**On whether §7.12 misleads.** It does not. It says the thing, in those words. What failed is
propagation: the sentence sat under a subsection about calibrating a parameter already shown not
to matter, and never reached §5.4's stage list or the M1 ordering's rank 2. Both were mine. Both
are fixed. I am getting a taste for findings that turn out to be located in my own document.

Rank 2 was worse than under-qualified — it credited `retain_viable` with the 321 → 33 collapse,
which §10.0.1's own R10 table attributes correctly to wiring the clustering. Two mis-attributions
in one row. Position withdrawn, row kept as the record.

**The drafting rule got its second live example and this one bothers me.** RAI-011's criterion is
"always evaluated, no branch in front of it". The cheapest satisfaction is an unconditional early
return *inside* the gate — every word true, `bypasses` 0 forever. 3b is not that; it predates
RAI-011 and lives in the right module. But RAI-011's observables cannot tell them apart. That is
the whole argument for item 3.

**What I refused.** Removing the exemption to let the model decide. Deferring to a model measured
wrong by five orders of magnitude is not rigour, it is ceremony, and it loses M0.

The worktree collision is the coordinator's and he recorded it as his; I noted in the decision
that Mouse caught both consequences himself, including a false ALL-DECLINED he nearly wrote up.
An agent that catches its own contaminated build is worth more than the hours it cost.

**Carry forward**
- Mouse owes `Verdict::Claim` carrying its reason, in `partition.rs`, once `mouse-1` clears.
  Until then "the exemption decided this" stays an inference, and I have written it as one.
- The byte estimator (104,116×) is a correctness item ahead of any nanosecond calibration, and
  ahead of the optimisation rank 2 used to hold.
- Named falsifier to keep live: an anchor-bearing island that should be declined. First small
  model or edge-shape graph we touch, look for it.
- Fourth ruling in a row where the reported rule was wrong but the finding was real. The pattern
  is that people diagnose correctly and file incorrectly; that is a cheap error and I should stop
  treating the filing as the claim.

## 2026-08-01T22:25:29-07:00 — The estimator's first half, and why I would not take the concurrence

Fifth ruling today. The coordinator verified my §5.4.1 himself against the code before accepting
it, which is the second time today someone has read the source rather than take my word, and I
should say plainly that this is the reason any of this holds.

**What landed.** `mouse-1` fixed the first half of the estimator defect: internal island edges
were being charged to the boundary. 89.2 GB → 13.9 GB, 6.4× gone, and with the exemption off the
gate now claims Phi-3.5 on its own economics. Verified on `squad/mouse` before ruling — the
consumer map in `ep.rs`, the new constant, `symbolic_boundary_slots`, the doc comment.

**And I declined the conclusion offered with it.** The invitation was to read this as the
economics arm *concurring* with the exemption rather than being masked by it. Two problems. The
128-for-every-unknown-dim substitution is untouched and the residual is 16,268×. And more
importantly, agreement between two things fed the same fabricated input is not a second opinion —
which is the sentence this whole register is built around. A verdict that flipped because its
input moved 6.4× while staying 16,268× wrong flipped for a reason unrelated to the proposition.

**But the same fact supports something stronger, and finding it is the part of today I am
actually pleased with.** `transfer_ns` is monotone in bytes. The gate claims at 13.9 GB. The
measured boundary is 856,720 B, which is smaller. So it claims a fortiori on the true bytes: the
claim survives a 16,268× adversarial inflation of the term opposing it. That is a bound, not an
estimate, taken from a number I do not trust in the one direction where not trusting it is safe.
Third form of the invariance preference: prefer the count, prefer the ratio, prefer the bound you
can sign.

I wrote the licence tightly because this is precisely the shape that has failed here when it
favoured us. Monotone, sign established by independent measurement for that window, used only in
the licensed direction. Absent the sign it is a guess with a confident tone. And the sign is not
general — 128 over-counts on our window and under-counts on a long prefill, where the bound does
not weaken, it evaporates. That falsifier goes beside the small-MatMul one.

**The naming call.** `MEASURED_PHI35_DEV0` holds an estimate wrong by 6.4×, next to
`MEASURED_PHI35_DEV0_REAL_BYTES` which holds the measurement. Rename. Mouse's doc comment is the
best disclosure I have read on this project — he volunteered that parking the total in
`output_bytes` biases every test towards claiming, i.e. against his own conclusions — and it is
still not enough, because names outlive doc comments. That is the coordinator's sentence and I
took it into the register verbatim. Keeping the old constant beside the new is right; only the
name is wrong.

**Line numbers.** Mine went stale within the hour. A line number is a reference that decays
without failing — it points at something else rather than erroring, which is `'<absent>'` in a
different costume. Cite the symbol.

**Criterion 11.** He asked that the RAI-011 cheapest-satisfaction observation survive into the
discharge language, and it should, so I wrote the discharge language now, while the row is still
open — which costs nothing and retracts nothing, the same reasoning as criterion 12 on 07-30.
The cheapest satisfaction of "no form claimed without a ledger entry" is a ledger generated from
the claim table: true by construction, 6/6 forever, identical under both readings. Four
conditions, none of them "the ledger exists".

The general sentence I want to keep: a criterion is discharged by an observable that changes when
the claim is false, never by one that is true whatever happens.

**Carry forward**
- Mouse: three changes now bundled — `Verdict::Claim` carries its reason, the rename, the test
  rename. All in `partition.rs`, all once the worktree is clear.
- The 128 substitution is the remaining half and it is a correctness item, not an optimisation.
  Calibrating `fixed_ns` before it is fixed is still polishing the wrong parameter.
- Falsifier list for §5.4.1 now has two: an anchor-bearing island that should be declined, and a
  boundary tensor whose real extent exceeds 128.
- Criterion 11's tally is Trinity's; the discharge conditions are written and I should not be the
  one to declare the row met.
- Five rulings today. Four of them relocated a finding from the file it was reported against to
  the file that actually carried the defect, and three of those were mine. That is the pattern to
  watch, not the rule count.

## 2026-08-01T23:36:43-07:00 — Ratifying a merge I did not make, and two generalisations the day earned

The coordinator resolved a conflict in my criteria table and told me exactly what he chose and
why, which is the correct way to touch someone else's file.

**Row 11.** He kept mine over Mouse's "MET", on the ground I had written into the cell: a row
closed by the agent who supplied the artifact, in the change that supplied it, is an identity
whose two sides come from one source. I ratified rather than reversed, and I wrote into the cell
that Mouse's evidence is neither rejected nor lost — what is withheld is the tally, not the work.
That distinction is worth being explicit about; I do not want the register to become a way of
declining people's findings.

**Row 12.** He kept Mouse's, against my stale tail, because he ran the census himself on both
devices and got `unwired: []`. He was right and my text was wrong. But he flagged that rows 11
and 12 now read as a contradiction, and he was right about that too, so I wrote the reconciliation
into row 12: the census answers whether a mechanism ran, a criterion answers whether a claim is
false-able, and neither can do the other's job. A wired mechanism beside an undischarged criterion
is the normal state of a row being taken seriously.

**Two generalisations, and I checked both against the no-minting rule before writing them.**

*R9's third generalisation.* The sentence he is now briefing agents with — a criterion is
discharged by an observable that changes when the claim is false, never by one that is true
whatever happens — is the red-instrument test with "criterion" substituted for "claim". Same
remedy, wider scope. So a generalisation, not an amendment, and I said so in the text. Three
specimens in one day: RAI-011's early return, Link's screen on a variable nothing defines, and
Switch's assertion comparing 0.0 with 0.0. Two green, one negative — the class does not care about
polarity, which is our own sentence about the census arriving as a property.

*The dangling reference.* He is right that the phantom key, my stale line number and Link's
undefined variable are one failure. I named the class under R13 amendment 1. The line number is
the worst of the three because there is no lookup to fail — the reader performs it by hand and
receives a plausible statement. I also wrote where the class stops: a broken URL fails loudly and
is merely broken. The class is references that resolve anyway.

**Trinity's finding.** The §10.0 `attribution_witnesses` example showed two keys where the record
emits considerably more. She did not edit my document, which is correct. I regenerated the example
from `criterion10-dev0.json` rather than from memory, and wrote down why: a schema example is a
claim about the record's extent, and R11's extent obligation binds a document's example exactly as
it binds a producer's output. Two live keys were missing for a day and the defect was mine.

**Carry forward**
- Mouse still owes three things in `partition.rs`, all blocked on the shared worktree: the
  `Verdict::Claim` reason field, the `MEASURED_PHI35_DEV0` rename, the test rename.
- Row 11 closes when Trinity's tally moves against the four conditions. Not before, and not by me.
- The prefill falsifier for the bound (real extents above 128) is now a standing item on the
  coordinator's list, which is where I wanted it.
- Regenerate document examples from artifacts. I have now been caught by the same class of defect
  I ruled on twice today, in my own file, in a code fence.

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
