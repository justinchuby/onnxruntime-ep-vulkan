# Trinity (Test-Conformance) — history.md

## Learnings

## Round 27 — the union defect: a required parameter and a caller that was not on my branch (2026-08-02)

**Branch:** `squad/trinity` @ `1ccbfef`, then merged `main` @ `79674d6`. Not pushed.

### What broke
`TypeError: _run_counters_child() missing 1 required keyword-only argument: 'guard'` in
`tests/ops/test_wiring_census.py::test_ledger_lookup_wired`, on both devices. I added the
required keyword-only `guard` on my branch; Mouse added the caller on his. Both green alone.

### The fix, and the decision inside it
`guard=None`, where **None means build one, not run unguarded**. `None -> unguarded`
would satisfy "callable without a guard" while silently undoing round 25: every future caller
opts out of stall detection with nothing to notice. An ambient guard loses only *bookkeeping*
(its costs never reach the census-wide ledger, so they are absent from `observed_units`),
never protection. Both halves carry falsifiers because either alone is satisfiable by the
wrong fix. `test_stall_guard.py`: **16 passed**, deterministic, no GPU.

Same shape one line down: `_BUDGET_UNITS[label]` -> `_budget_for(label)` with a generous
default. Safe because the unit is *work*: a loose work budget detects a hang later in work,
never not at all, and contention cannot stretch it. That is not true of a wall clock.

### tests/union_check.py — and only tier 3 is a gate
Tiers 1 (file intersection), 2 (`sys.path.insert(0,)` in a colliding directory) and 2b
(module-basename collisions, computed from the working tree) are **preconditions**, never
gates — R9 A5: a union can break with no intersecting file and no scanned side effect, so
their silence is a statement about the tool. It prints
`PRECONDITION(tiers=1,2; tier 3 not run)`, never a bare PASS.

Tier 2 was 21 findings and useless at first. The signal was never the insert — it was the
**population of colliding basenames**. Against pre-fix history tier 2b reports
`COLLISION device_state.py <- bench, ci` and names `bench/device_state.py` and
`ci/check_lane_inventory.py`: it retrodicts the Niobe x Link incident. Against current main
(after Niobe's rename) it reports `none`. Same code, different tree, different report —
the R10 falsifier for the tool itself.

Tier 3, both arms, real history, same target, same device:
- broken union `2fee5ef..c55a389` -> `FAIL(condition=union_red)`, exit 1, quoting the TypeError
- repaired union `main..squad/trinity` -> `PASS`, exit 0, `ERROR(instrument): 0`

Conflicts confined to `bench/results/` artefacts resolve to HEAD under
`--resolve-artifacts`; anything else stays `ERROR(instrument=merge_conflict)` — an
instrument that cannot construct its subject has not observed it. Scratch worktree lives
under the repo's parent (not Niobe's `bench/results/`) and is removed in a `finally`;
`git worktree list` shows no leak.

### The near-miss I want on the record
My first post-merge census read `proven_key_lookups == 0` / `ledger_entries=0` and I
almost filed "the S8.9 ledger is UNWIRED". The cause was a **stale
`rust/target/release/onnxruntime_vulkan_ep.dll`** predating Mouse's ledger code. After
`cargo build --release` it passed. This is Mouse's exact failure mode from earlier today,
and in a shared worktree (`trinity` and `trinity-1` map to the same directory) it is the
default state after any merge. **Rebuild after merging** is now part of my standing
obligation, not an optional step.

### Verification on the fully merged tree
- `test_stall_guard.py` — 16 passed.
- `test_wiring_census.py` dev1 (Intel) — 3 passed, `ERROR(instrument): 0`.
- `test_wiring_census.py` dev0 (NVIDIA) — 3 passed, `ERROR(instrument): 0`.

No timing figure is quoted as a claim of record (S10.0 obligation 8).

### Recommendation to the coordinator
Standing obligation: **merge `main`, rebuild, then `python tests/union_check.py --run`**
before reporting a branch done. Targeted rather than full-suite, deliberately: the full suite
is stronger but the targeted form is the one people will actually run, and it caught this.

### Routed, not edited
`docs/DESIGN.md` S10.0 (~line 2781) shows an `attribution_witnesses` example with two keys
where the record now emits six. Morpheus's file.

Decision record: `.squad/decisions/inbox/trinity-union-check-and-guard-default.md`.

📌 Team update (2026-08-02T02:03:46-07:00): Morpheus named R12's fourth generalisation — for a test result, the frame is the binary that ran it — from two of Mouse's self-caught near-misses this session: a build in a shared worktree that linked a sibling's in-flight file (nearly reported as a false ALL-DECLINED finding), and Copy-Item preserving LastWriteTime, which let cargo silently keep running a mutated binary after a restore-from-backup. Your own union-check work independently reproduced the same failure shape (a stale DLL in a shared worktree nearly read as an UNWIRED §8.9 ledger) — worth checking any mutation harness you build touches or hashes a restored file and asserts the rebuild happened before reading a result as a control. — decided by Scribe

---

## Round 28 — criterion 11(c), and which `model_output_equivalence` is of record (2026-08-02T03:10-07:00)

### Standing obligation, executed first
Fetched, merged `main`, then **rebuilt and hashed either side** — R12 generalisation 4, which
Morpheus minted from my own round-27 stale-DLL scare. `3B0C0ACD7AA7...` -> `8D07173F8AE5...`,
`rebuilt=True`. No reading was treated as a control before that assertion passed.
The merge aborted on regenerated `bench/results/wiring_census-dev{0,1}.json` and an untracked
`bench/results/census/`; both are outputs, discarded, merge clean.

### Criterion 11(c): `ledger_hits` moves with its input
Three tests in the census lane, not behind `#[ignore]`:
`test_ledger_hits_moves_with_its_input`, `test_ledger_key_discriminates_optional_inputs`,
`test_ledger_digest_refusal_is_in_the_lane`.

I **probed nine arms before writing a single assertion**. An assertion written ahead of the
reading is a test of my guess, not of the mechanism — and it mattered here, because my brief
was wrong about which control moves the counter.

The **shape-class arm is load-bearing**: static `mul_f32` vs dynamic-extent `mul_f32` is the
same op, dtype, optional-input set and one-node enumeration; only the key's `shape_class`
changes, and `ledger_hits` goes `1 -> 0`. The enumeration is identical work in both arms, so a
counter derived from the enumeration could not have moved. That is the falsifier.

The **MatMulNBits `scales` / `scales+zero_points` pair does NOT move `ledger_hits`** — both
proven, both `HIT`. Reporting it as a mover would have been R11 exactly. Its real content is
that they must be two keys, asserted on the key string: `differing == [2, 5]` over the six
components; index 5 is the populated-optional-input set (the 2026-07-30 all-zero-logits defect).

The **digest-identical arm is the control**; without it every other assertion would pass
against a check that rejects everything. Its failure classifies `ERROR(instrument)`.

**Both arms demonstrated** — `probe_ledger_mutations.py`, three mutations, all CAUGHT on
**both** devices. Quoting failure text, not counts (R13).

`ledger_lookup` promoted into `_MANDATORY_WIRED`; `_KNOWN_UNWIRED_M0` emptied (kept as a seam).
A ledger that stops being consulted now fails the lane.

### `model_output_equivalence`: `MATCH` beside `UNMEASURED`
Not two sources disagreeing. The EP has no CPU oracle; `to_json()` defaults the field. The
nested value is **a field nobody set**. The mechanical tell needs no judgement:
`write_equivalence_record` writes the token and the record in one call, so a token with no
record beside it was never written by a comparison — and in the phi35 artifacts the record key
is absent. The rule keys off **record presence, not token value**: keying off
`token == "UNMEASURED"` would read the field under suspicion, and would mislabel a genuine
comparison that legitimately concluded `UNMEASURED`. Both polarities asserted.

It did not "go" null — the phi35 bench path never calls `write_equivalence_verdict`, so that
copy has always been the default. Per R12 the reconciliation is `UNOBSERVABLE` **with a reason**;
`agreement = (outer == inner)` would have read `DISAGREE` and sent a reader after a
contradiction that does not exist.

**R9 A5 turned on my own check.** My first run flagged ~20 historical artifacts. Frozen,
unowned, non-regenerable — a permanently red gate gets loosened, so I demoted them to a printed
`PRECONDITION` and scoped the gate to the 4 certified files. Two *genuine* contradicting
readings stay a finding everywhere.

### Verified
Intel: census + equivalence + stall-guard 26 passed, `ERROR(instrument): 0`.
NVIDIA: census 6 passed, `ERROR(instrument): 0`. Mutation probe PASS both devices.
Stale-stamp falsifier fired with the right text; artifact restored.
`audit_instruments.py --check`: `CENSUS VERDICT: PASS`.

### Routed, not edited
`bench/phi35.py` should stamp `model_output_equivalence_authority` at write time so it survives
regeneration — **Niobe's file**. I stamped the four existing certified artifacts by hand.
`docs/DESIGN.md` S10.0 attribution_witnesses example still shows two keys where six are emitted
— Morpheus's file, still open.

Decision record: `.squad/decisions/inbox/trinity-criterion11c-and-equivalence-authority.md`.

### Addendum — a defect the 11(c) arms introduced, caught by their own artifacts
The tracer witness path is shared across census arms. A faulted-ledger arm dispatches
nothing by construction, so it unlinked the clean arm's tracer file and wrote none — a
**false UNWIRED available purely by running later**, invisible in a green lane and visible
only as a tracked artifact missing from `git status` after six passing tests. Fixed with
`trace=None` defaulting to the historical behaviour (no required kwarg — round 27's `guard`
lesson). Both arms shown: absent before, present on both devices after.

### union_check: FAIL(condition=union_red), and the condition is not mine
153 failed / 277 passed on the merged tree. Attribution before alarm: `git diff main --
rust/ evidence/ ops/` is empty, and the same subset scores **21 failed / 18 passed
identically** in main's own checkout with main's own DLL. The decline text is
`[unproven] no proof ledger entry for ...`. So main is red on its own — the ledger gate is
enforcing against a 9-entry ledger — and that is Mouse's ground, reported not touched.
R13: quote the failure text, and an inherited condition is not a detection about my branch.

Commit `e39bdd6` on `squad/trinity`, not pushed.

---


## [SUMMARY] Compressed entries

<!-- SUMMARIZED by Scribe 2026-08-02T15:26:03.394019 -->

- **[SUMMARY] Rounds 1–17: CI, barriers, oracle, claim log, Phi-3.5, env var finding (2026-07-28–2026-07-30)** — **Rounds 1–5 (archived):** Differential harness (EP vs CPU). Profiling-JSON claim assertion. Barrier parity layer. Two lavapipe CI lanes (Linux and Windows). ORT CPU EP quantised oracle with `accuracy_level=1`; fp16 gate
- **[SUMMARY] Compressed entries (condensed 2026-08-01T20:39:12-07:00)** — - **Round 19 (2026-07-30T20:33:50-07:00) — paired controls, validation lane, wiring census** — **Trigger:** Morpheus's ruling: criteria 4 and 5 "partially met — unchanged, and deliberately untouched by today," on five co
- **Round 29 — criterion 12: the twelve surfaces no census mechanism watched (2026-08-02)** — `unwired: []` was measured against a denominator the census supplied itself. Link's
- **Round 30 — per-output attribution: the fifth costume (2026-08-02)** — The coordinator's concern, and it was right. `attributed` was a property of the SESSION;
- **Round 31 — ORT refuses the session we spent the session learning to detect (2026-08-02)** — **Handed to me:** the user asked whether ORT has a flag preventing EP fallback. It does:
- **Round 32 — the specimen stopped being a specimen (2026-08-02, merged `a0bd22d`)** — Merged `origin/main` (18 artifact conflicts, all regenerable — took main's), **rebuilt**:
- **Round 29 — the two unfalsified guards, screened in the always-on lane (2026-08-02T11:40-07:00)** — Merged `main` (`4e70601`), rebuilt, hashed either side: `D45B3A8C8C2B...` -> `918E8FF56B2E...`,

## Round 33 — 2026-08-02 — criterion 10's residual unit: ULPs, and the median

**Task:** Morpheus's ruling (merged `abf3b3e`): `atol` is an absolute bound on tensors of
growing scale; express the residual in ULPs; prediction *flat at 1-3 across 32 layers*
recorded before measuring. Do not move `atol`, do not flip `DIVERGENT`.

**Binary:** merged `origin/main`, rebuilt, DLL `523A07C1...` -> `7F55C0C1...` (hashed either
side; a stale binary told this project the wrong story three times today).

**What I built.** `ulp_residual()` and `ulp_outliers()` in `_models.py`;
`compare_all_outputs_to_cpu` now emits `median_ulp_diff` / `p99_ulp_diff` /
`max_ulp_diff` / `ulp_cancellation_elements` / `ulp_basis` per output;
`test_criterion10.py` records the full `ulp_curve` and `ulp_outliers`;
`test_criterion10_ulp.py` is a 15-arm GPU-free instrument lane.

**Three things I got wrong and the file now asserts.**

1. I demoted `max_rel_diff` for *unboundedness*. It divides by `atol + |b|`, so it is
   atol-floored and does not diverge. The complaint against it is **non-monotonicity**.
   Asserting the stronger claim would have made the file allege something the code does
   not do.
2. My docstring claimed ULP cannot blow up near zero. **It can** — a cancellation element
   reads 16384 ULP in fp16, verified. So the headline moved from `max_ulp_diff` to
   **`median_ulp_diff`**: swapping one max for another would have reinstated the artefact
   in a fresh unit (R11). The real record proves it — output 0's max reads 255658 while
   its median reads 12.
3. My specimen for max_rel_diff's non-monotonicity used a 1-ULP residual everywhere and
   came out flat — correctly, because the relative error of a 1-ULP residual is
   scale-free. **The non-monotonicity needs a cancellation element to exist at all**, and
   finding that out is what located the right headline.

**The measurement** (3 runs, 65 outputs, **byte-identical on RTX 4060 and Iris Xe**):
KV cache flat at 0-3 ULP, baseline 1, over 62 of 64 outputs — the ruling's claim, measured,
against an absolute curve that rises monotonically. Outputs 63-64 read 4: a smooth
accumulation drift, no discontinuity, reported as an exceedance rather than tuned away.
**Output 0 (logits) reads median 12 ULP, 12x baseline, both vendors, all three runs.**

**The finding.** Under `max_abs_diff` the logits were already the worst output and that read
as "largest tensor, largest residual". In ULPs magnitude is in the denominator and output 0
is **still** an order clear. Not a big tensor — **a located defect, in the head not the
layers**, and vendor-independent, therefore arithmetic in our kernels. Candidate: the final
vocab projection (3072 -> 32064). Owner: Mouse/Switch.

**The predicate is his number, not mine.** My first exceedance rule was "3x the observed
baseline", written before I saw the curve; on real data it flagged 63-64 and my instinct was
to widen it until it did not. That instinct is the defect. The predicate is now the
prediction, and the overshoots stand in the record.

**R13 on a confirming result.** The KV curve confirms the prediction, so it got the harder
look, and the second reading found what the first missed: *flat at 1-3 across all 32 layers*
is true of the layers and **false of the model**, because the model has a 65th output that is
not a layer.

**Also:** `argmax`/top-10 now carry `argmax_sample_size: 1` and a caveat (N=1 can falsify
agreement, not establish it); `_verdict.py`'s `of_record_source` names its sibling list's
weaknesses instead of listing three statistics unqualified.

**Untouched, deliberately:** `atol`, the `np.allclose` gate, the `DIVERGENT` verdict (both
devices), GQA's 1.37x margin. Criterion 10 stays open. No row closed.

**Verification:** `test_criterion10_ulp` + verdict + output_attribution (+hw) + island +
r13_lane + no_cpu_fallback = **133 passed / 0 FAIL / 0 ERROR on dev0 and dev1**.

## Round 31 — 2026-08-02 — criterion 3(a) on a run that genuinely executes Phi-3.5

**Discharged, both devices.** `bench/results/criterion3a_phi35-dev{0,1}.json`.
355 claimed nodes, one retained island, `device_losses: 0`, `in_frame_vuid_count: 0`.

**The load-bearing part is not the zero.** The EP messenger subscribes to ERROR|WARNING, so a
healthy run is silent whether the callback is live or dead: a bare `0` here is `UNOBSERVABLE`,
not a measurement. No existing falsifier reaches the ORT-session frame (the Rust plant is on a
path the session never takes and its VUID is teardown; `epctl` proves arming in *another*
process — R12 gen-4). Used `VK_LAYER_ENABLES=VK_VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT`:
WARNING-severity best-practices messages ride the EP's own messenger in-process and in-frame,
and are prefixed `BestPractices-` not `VUID-`, so they prove liveness without contaminating the
count. **14 in-frame messenger lines in the liveness arm, 0 in the clean arm, on both devices**,
with the arms asserted to differ.

**Lesson worth keeping: the device selector is a request, not an identity.** Selector 0 ran on
`1=NVIDIA GeForce RTX 4060 Laptop GPU`; selector 1 on `0=Intel(R) Iris(R) Xe Graphics`. The
allocator index is not the selector. "Both devices covered" from the env var would have been a
claim about what I asked for. The lane now reads the device name off the run and refuses a run
that cannot name its device.

**Corrections to the brief** (recorded, not adopted): `ledger_gate` is `MIXED` not `ALL-PROVEN`
(2 unproven declines); `dispatches_executed` is 355 not 8875 (per claimed node, not per
`vkCmdDispatch`) — I quoted no 8875; criterion 10 is attributed but DIVERGENT and 3(a) is
recorded against that rather than waiting on it.

**Files:** `tests/ops/probe_validation_phi35.py` (probe, `run_arm` shared with the lane),
`tests/ops/test_validation_phi35.py` (4 lane tests), `tests/ops/test_validation_phi35_frame_split.py`
(18 device-free polarity tests screening `split_frame`, `_gate_attribution`, `device_name`, so
they do not land as `unfalsified` in Tank's census). `audit_instruments.py --check`: PASS.

**Routed:** Switch — in-frame on both devices, `vkCmdDispatch(): Pipeline uses a push constant
range with offset 0 and size 128, but 104 bytes were never set` (also 88/72/36/20/4). Not a VUID,
does not fail 3(a); reading unwritten push-constant bytes is undefined.

**Two reds attributed away from this branch:** `test_r13_lane.py::test_fallback_line_produces_a_lane_failure`
is a **26-line uncommitted edit by the sibling instance in the shared worktree** (absent from
`git status` minutes earlier); main's checkout passes it 37/37. And 41 `test_op_table` failures
(`VulkanExecutionProvider did not execute any node`) reproduce in main's own checkout with main's
own DLL. Attribute before reporting — twice in one round it mattered.


---

## Round 34 — the second witness had been agreeing with us in our own words

**Routed by Link, not found by me:** `_verdict.FATAL_LOG_MARKERS` did not match the line ORT
prints. ORT emits `Falling back to ['CPUExecutionProvider'] and retrying.` -- a **list repr**.
Five incidents had cited `check_fatal_log` as second witness.

**It was not returning silence, which is the part worth remembering.** Measured across the three
real logs before touching anything: Tank's `ctx512_device_lost.txt` -- 2 announcements, **0**
hits. `trinity-suite-dev0.log` -- 5 announcements, **9** hits. `dev1` -- 1 announcement, **3**
hits. **All twelve hits were `test_phi35.py`'s own docstring** echoed into the captured log by
pytest. The marker never once matched ORT. Per R9 the five corroborations added nothing, and
they made those incidents look better witnessed than they were.

Its only positive test built the fiction and asserted it was found -- green for exactly as long
as the witness was blind.

**Two properties I would not have found from the source.** The announcement **wraps** (Tank's log
breaks the EP Error line mid-list), so matching had to move off `splitlines()`; and ORT's C++
sink writes **UTF-16LE** into an otherwise UTF-8 file, so decoded as UTF-8 the message is
NUL-separated and unfindable by substring. `dev1.log` happens to carry it in both encodings --
had it carried only the wide form, a substring search would have seen an empty log. Read the
artifacts, not the source.

**Remedy is liveness, not a better string:** a marker list never shown to fire is
indistinguishable from one that cannot. `assert_fatal_log_check_is_live()` classifies a verbatim
committed positive control red and the extent-boundary control green, before any verdict is
trusted; `check_fatal_log` reports `ERROR(instrument=witness_not_live)`, never a pass. Lineage:
`assert_no_cpu_fallback_is_live()`. One arm monkeypatches the **old broken marker** back and
requires the liveness check to raise.

**The restraint that mattered:** I did not widen the patterns to catch `The logical device has
been lost`, though it would have looked like an improvement. Link's negative control proves
`check_device_loss` has reach of its own by requiring `check_fatal_log` to stay **green** there.
Widening buys coverage by destroying a demonstration. Extent stated in PLATFORMS 7.18.4/7.18.5.1.

**Verified:** red on Tank's artifact (the file it read as clean), quoting all three lines;
negative control 14/14, 1 LIVE / 3 REPLAYED / 10 PLANTED; `ci/test_lane_checks.py` 5 red -> 3
(exactly Link's known census reds); lane set **140 PASS / 0 FAIL / 0 ERROR on both devices**.
Edited two of Link's files as direct consequences -- flagged for his review.

## Round 35 — 2026-08-02 — criterion 10 after `872d739`: the route, the axis, the run that fails it

**What surprised me, first:** the premise I was handed was false. The brief said Switch's
device-authoritative KV spans had changed the path the 64 KV outputs take under criterion 10.
The counters said `outputs_device_bound = 0`, `outputs_host_resident = 196`. The new route is
behind `ONNXRUNTIME_EP_VULKAN_BIND_OUTPUTS`, which ships OFF — a second path appeared **beside**
the criterion, not underneath it. Forced onto it (`outputs_device_bound = 196`,
`alloc_device_authority_grants = 196`), all 65 per-output residuals are **identical** on both
vendors. Switch's writeback delivers the same bytes. The route is now in the record, read off
the counters the run emitted, never off the env var that requested it: Step 1c unbinds on
refusal, so a declined bind would be recorded as a route that was taken.

**What surprised me, second, and it nearly cost this project a fictitious defect.** I had a
drafted finding placing the residual peak at **layer 9** — smooth, plausible, reproducible,
and identical on both vendors. It was an artifact of reading output names out of the record,
which is serialised `sort_keys=True`, so `present.10` precedes `present.2`. `present.9.value`
landed at index 64. Caught by asking the **session** for its order instead of the file:
`sess.get_outputs()` returns depth order, and this model's true order is not its own sort —
which is exactly what makes the refusal cheap and sound. Now three defences: names in session
order in the record, `assert_names_are_session_order()`, and a test that pins both orders
against the same medians and asserts they reach **different conclusions**.
R12 one level out: *the frame of a name is the run that produced it, not the file that stored it.*

**The measurement.** 65 compared, 62 within tolerance, 0 degenerate, failing `[0, 63, 64]`,
per-output verdicts and never an aggregate. Morpheus's prediction — flat at 1–3 ULPs across
all 32 layers — is **confirmed for the KV cache**: largest layer-to-layer step 1 ULP, no
discontinuity, only exceedance layer 31 key and value at 4. The step is output 0, the logits,
at 12 ULP, and it is **not a layer**. "Flat at 1–3 across all 32 layers" is true of the layers
and false of the model.

**The run that would fail it, named before recording anything as met, and reachable:**
criterion 10 with `BIND_OUTPUTS=1` on `872d739^` returns all 65 outputs all-zero with
`cross_run_identical_to_run1 = True` — wrong **and stable**, the exact shape the criterion was
reopened for. Guard fires: 65 degenerate → `NOT_PERFORMED` → `UNMEASURED`. Unplanted, from the
project's own history. Caveat stated in the record: logits-only catches this *instance* by luck
of zeros; the *class* stays covered by the synthetic KV-only plant.

**`atol` untouched. Verdict still `DIVERGENT` on both devices and both routes. Criterion 10
stays open.** The tolerance is genuinely mis-specified — absolute bound, growing scale — but
that is a change to what the criterion measures, so it went to Morpheus as an argument with
old-vs-new and what each catches that the other misses. The moment to fix a tolerance is not
the moment when fixing it turns two of three failing outputs green, and not by the person whose
measurement made them red.

**Verified:** 61 GPU-free falsifiers green; five real Phi-3.5 runs (2 routes × 2 devices + the
pre-Switch control); `cargo test --lib` 492 passed; clippy exit 0, no warnings.
Reported not fixed: one intermittent `counters::tests` failure (1 in 6, no rust diff on this
branch, failure text not kept — said so anyway); `bench/phi35.py` still owes Niobe's
`model_output_equivalence_authority` stamp.

---

## Round 36 — the logits step is inherited, and the intermittent was real

**The premise I was handed was false again, and in the same direction as last round.** The
brief described the logits as "the top of a smooth climb" up the 32 layers. **There is no
climb.** Tapping the residual stream at every block gives medians
`[0,0,1,2,2,...,2,1,3,3]` — flat at ~2, largest step 2, and **not monotone** (it dips at layer
29). The number only moves in the last two hops: stream L31 `3` → final RMSNorm out `6` →
logits `12`.

**Named the distinguishing observation before running either arm.** `H_proj` predicts the 12
ULP survives isolation of `lm_head`; `H_depth` predicts it vanishes. Isolated `lm_head` on
bit-identical fp16 input both EPs: **median 0.0 ULP**, attributed, both vendors. `H_proj`
refuted. The five-order accumulation envelope is median 0.0 — fp16 storage rounding swamps
fp32 order over K=3072, so order cannot manufacture this residual at all. My own named
falsifier did not fire.

**First float64 reference the criterion has ever had.** At `lm_head`: `neither (equal)`. At
the final RMSNorm: **Vulkan is bit-exact and the CPU EP is the 1-ULP one.** "The Vulkan EP is
wrong until proven otherwise" is, at that node, backwards. Worth carrying: `divergent` has
never once been asked *which* of the two is wrong.

**I built and nearly shipped a false located defect of exactly the Round 35 shape.** My first
arm F compared a float64 reference built from the *CPU EP's* tapped inputs against the Vulkan
EP's *in-situ* output — arm A's 6 ULP wearing a reference's clothes, reporting
`which_is_further_from_true: "vulkan"` **by construction**, and it would have named a node the
EP executes bit-exactly. The rule that caught it: **isolation means identical inputs on both
sides or it means nothing.** Kept the discarded version in a comment rather than deleting it.
Same family: `claimed_nodes` is process-cumulative (arm A 355, arm B 356 for one node), so
attribution is now a **delta**, and delta 0 marks the arm UNATTRIBUTED — an isolated run the EP
declined is CPU-vs-CPU and reports 0 ULP, the most convincing possible way to measure nothing.

**`atol` untouched again.** If a relative or ULP bound is correct that is Morpheus's ruling;
I filed the measurement that makes it arguable and applied nothing.

**The intermittent was three defects across three modules, and one had a false-pass mode.**
Isolated arm 8/20 failed, full arm ~0% — **the narrower filter was the more sensitive
instrument**, because a 21-test pool aligns on the contended statics far more often than a
505-test pool. Four different tests at four different assertion sites → a race. Three
`counters.rs` tests and two `logging.rs` tests touched process-global statics / the ORT logger
pointers without `ledger::test_lock()`; the logging one made a WARN that *did* reach ORT record
as `warn_reached_ort_sink: false`. A reader is as much a party to a race as a writer. My first
auditor had a hand-written two-file list and was structurally blind to the third module —
Mouse's anti-pattern; it now scans all 43 sources, resolves lock aliases, and has a selftest
that proves it can go red. After: **25/25 isolated + 15/15 full green.**

**Verified:** 12 new device-free falsifiers green (one of which I had to *make* able to go red
— my first separating case used a 2²⁰ pivot whose fp32 spacing is 0.0625, far too small to
absorb unit terms); criterion 10 lane 62 pass / 1 fail, the fail being the known open
`DIVERGENT` verdict, unchanged; `cargo test --lib` **501 passed, 0 failed**; clippy exit 0;
auditor 0/0 with selftest 3/3.

📌 Team update (2026-08-03T04-55-00-07-00): Link retired his own Session-13 method of quoting a rebuilt-DLL hash as evidence a binary changed — six builds of an unchanged tree produced six distinct Windows DLL hashes, so a hash witnesses nothing about content. Where your quoted DLL hashes (e.g. in the fatal-log and push-constant liveness work) are being used to argue "this is a different binary," they need a different witness (a real diff, a digest of the compiled artifact's semantic content, or a behavioral control) — a hash alone is not evidence of a code change. — decided by Link
