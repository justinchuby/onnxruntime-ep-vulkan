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
