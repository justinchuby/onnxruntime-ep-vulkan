# Trinity (Test-Conformance) — history.md

## Learnings

### [SUMMARY] Rounds 1–17: CI, barriers, oracle, claim log, Phi-3.5, env var finding (2026-07-28–2026-07-30)

**Rounds 1–5 (archived):** Differential harness (EP vs CPU). Profiling-JSON claim assertion. Barrier parity layer. Two lavapipe CI lanes (Linux and Windows). ORT CPU EP quantised oracle with `accuracy_level=1`; fp16 gated on ORT ≥1.28. Windows ICD registration: `HKLM\SOFTWARE\Khronos\Vulkan\Drivers` (loader ignores `VK_ICD_FILENAMES` under elevation). PowerShell array-operator bug fixed. `glslc` via LunarG apt repo (`shaderc` not in Ubuntu's own).

**Round 11 (2026-07-29T09:19:35-07:00) — shape inference and device parametrization:**
`apply_shape_inference(model_bytes)` in all harness paths. Device parametrization across indices wired (blocked on Switch `ep.device_index`; now available as `--vulkan-devices 0,1` or `VULKAN_DEVICE_INDEX=0,1`).

---


<!-- SUMMARIZED by Scribe 2026-08-01T20:39:12-07:00 -- older entries condensed below; full text lives in git history -->

### [SUMMARY] Compressed entries (condensed 2026-08-01T20:39:12-07:00)

- **Round 19 (2026-07-30T20:33:50-07:00) — paired controls, validation lane, wiring census** — **Trigger:** Morpheus's ruling: criteria 4 and 5 "partially met — unchanged, and deliberately untouched by today," on five consecutive days.
- **📌 Cross-agent context — Round 4 (2026-07-30T02:49:12-07:00)** — ### Worktree layout and inbox portability constraint The team works in git worktrees: `squad/switch` at `C:\Users\justinchu\dev\ep-vulkan-switch`, `squad/mouse` at `C:\Users\justinchu\dev\ep-vulkan-mouse`, `squad/tank` at `C:\Users\justinchu\dev\ep-vulkan-tank`, with `main` as the integration tree.
- **Round 18 (2026-07-30T09:14:00-07:00) — False-premise test audit + correctness gate** — **Trigger:** Switch's runtime-extents merge moved 258 dynamic-shape declines to 0.
- **Round 21 (2026-07-31T07:20:03-07:00) — Guard D proven to fire; harness joins the census** — **Trigger:** Guard D (`assert_vulkan_executed_runtime`) had raised `NameError: name 'pathlib' is not defined` at its first statement since the day it landed and had **never read a single profiling event**.
- **Round 22 (2026-07-31T21:08:53-07:00) — the verdict becomes a record; MATCH made unrepresentable** — **Task:** implement §10.0's third metric amendment and §10.0.1 R13.

- **Round 20 (2026-07-31T00:26:22-07:00) — Guard D: runtime-fallback vacuous-pass closed:** `Allocator::alloc(size=0)` returned `None` for KV-cache `[1,32,0,96]` inputs; ORT printed `EP_FAIL`, fell back to CPU silently, `get_providers()` still showed the EP, and `test_phi35_vulkan_matches_cpu_logits` PASSED (CPU-vs-CPU) — fourth fallback-trap instance. Fixed via `count_vulkan_executions_from_profile`/`assert_vulkan_executed_runtime` (Guard D) wired into 4 phi35 tests; device labels corrected (selector 0=NVIDIA, selector 1=Intel). Needed from Switch: zero-size alloc must return a valid zero-length buffer, not `None`.
- **Round 23 (2026-08-01) — R13 reaches the subprocesses; criterion 3's frame gate closed both devices:** `epctl --probe-validation` → `classify_validation_probe()` → ARMED/LAYER-ABSENT/NO-LOADER/PROBE-ERROR (exit 0 with no ARMED line is PROBE-ERROR, not green); `require_validation_armed()` raises `InstrumentError` so an unarmed machine is neither green nor a false detection (R12: UNOBSERVABLE, never 0) — verified by a real test that strips `VK_LAYER_PATH` and requires the classifier's answer to change. Added `run_subprocess_checked(quiet_seconds=N)` with a 6.0× contention multiplier (Niobe's measured 4.4× worst case + headroom) and 120s floor, turning timeouts into `InstrumentError` rather than false UNWIRED. Discovered `XPASS` (an `xfail(strict)` that passed) was being misclassified into the outage bucket — R13's corollary that a result contradicting a prediction deserves scrutiny means good news shouldn't be buried there either; gave it its own token. `audit_instruments.py` extended to see `ops/_verdict.py` (12 instruments now). Replaced a stale pinned `"353"` fusion-ratio literal with a property assertion (actual count moved 355→over the session as Mouse's claims grew, then CPU count fell 30→24 for a good reason — a suite pinning literals would have gone red on progress). Post-merge results: `test_criterion10.py`/`test_validation.py`/`test_wiring_census.py` all passing both devices; full `tests/ops` **351 passed / 32 failed** in 1018s — R13 splits the 32 into 31 FAIL(condition, policy/expectation mismatch on §8.9 evidence gating, not mine to patch) + 1 XPASS; no wall-clock assertion anywhere in the harness.

📌 Team update (2026-08-01T17:16:56-07:00): Intel device-clock figures are permanently uncertifiable on this hardware (`none_available`, no producer exists and none of the available proxies are the right kind of quantity) — attack the Intel/NVIDIA residual with counts and shapes, not clocks — decided by Niobe


📌 Team update (2026-08-01T17:16:56-07:00): All wall-clock figures remain withdrawn; only counts, bytes and certified-companion device-clock figures are quotable — decided by Switch, Morpheus, Niobe, Link


📌 Team update (2026-08-01T17:16:56-07:00): `ledger_lookup` is the last `UNWIRED` mechanism in the instrument census (criterion 11); Mouse is building it — decided by Trinity, Mouse


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

## Round 29 — criterion 12: the twelve surfaces no census mechanism watched (2026-08-02)

`unwired: []` was measured against a denominator the census supplied itself. Link's
independent whole enumerates 50 surfaces from production Rust; twelve were instrumented and
observed by nothing. Two new mechanisms in `tests/ops/test_wiring_census.py`:

* `ep_entrypoints` — `compile_calls` / `compute_calls` / `subgraphs_stub`. New state
  `ENTERED-NO-DISPATCH`: a `Compute()` that entered and dispatched nothing. The census
  inferred execution from `dispatches_executed`, so a broken kernel path and a partitioner
  that claimed nothing were the same number.
* `flag_frame` — the nine uncensused env switches, two arms (`flags_a`, `flags_b`).
  6 MOVED / 3 UNOBSERVABLE-with-a-reason on **both** selectors. Every segment carries
  `census_frame=` (disclosure, cannot fail) beside a state token (discrimination, can).

Extent moved, which is the criterion's real content: `gpu_tracer` 1/12 -> 12/12,
`ledger_lookup` 4/7 -> 7/7, `net_benefit_gate` 0/6 -> 6/6, `validation_messenger` 0/3 -> 3/3,
`broken_commitment_warn` 0/2 -> 2/2, `partitioner` 1/2 -> 2/2, `retain_viable` 0/1 -> 1/1.
The tracer now names WHICH of the ten `Phase` variants the trace carried (6/10) rather than
a count of distinct names — the reading that would have certified `Phase::Record` again.

**A crash.** `DEVICE_MEMORY=1` + `VA_RESERVE_MIB=64` = STATUS_ACCESS_VIOLATION on a six-node
chain, deterministic, both devices; either flag alone is clean, 1024 is clean.
`factory.rs:897` declines to publish a device allocator, says so, and the device-memory path
runs anyway. Pinned as `test_device_memory_with_small_va_reservation`, xfail(strict).
Found on the flag frame's first run — the whole argument for censusing what nobody watches.

**Two findings against myself.**

1. The first tracer matcher looked for the Rust identifier and reported `0/10 Phase variants
   emitted`. The trace spells them `vulkan.record` etc. per `Phase::as_str`; six were in the
   artifact I was reading. I went looking for a gap and got the biggest one available on the
   first try, and it was my bug. R13's second corollary, earned again.
2. `probe_stall_guard.py` wrote its four cells as `wiring_census-dev0-<cell>.json`, matching
   Link's arm glob. His screen counted frozen snapshots as census arms, so any text
   improvement read as `VARIES`. Renamed to `stall_guard_census-*`. The honest number over
   the two real arms is **2 VARIES / 12 INVARIANT**, worse than the 3/9 reported. Moving it
   needs arms designed to vary each mechanism — named, not faked.

Verified: `tests/ops/test_wiring_census.py` 6 passed + 1 xfailed on dev0 (56 s) and dev1
(95 s), DLL rebuilt after merging `origin/main` (hash 8D07173F -> 1A802D09).
`test_r13_lane.py` + `test_verdict.py` 83 passed. `audit_instruments.py --check` PASS.

Row 12 **not closed**; `closes_row: false` written into the census artifact with the reason.
Link's screen now reports `FAIL(condition=unclaimed_mechanism_name)` for the two new
mechanisms, which is his screen working — the two `mechanism_names` claims and the twelve
re-dispositions are his file and are spelled out in
`.squad/decisions/inbox/trinity-criterion12-flag-frame.md`.

---

## Round 30 — per-output attribution: the fifth costume (2026-08-02)

The coordinator's concern, and it was right. `attributed` was a property of the SESSION;
the oracle comparison is a property of each OUTPUT. At `own_count == 0` they compose.
They stop composing as the ledger fills: EP claims some nodes, attribution says yes,
`MATCH` becomes representable, and the outputs whose producers still decline are compared
CPU-against-CPU under a verdict that says the EP ran. Morpheus's condition (c) is a
non-triviality guard on both sides; a guard on constancy cannot see a comparison whose two
sides are the SAME COMPUTATION. Fifth costume.

**Probed the data before designing anything** (`bench/results/probe_per_output_attribution.py`).
ORT names every CPU-executed node with its graph node name; a fused island arrives as ONE
event naming no constituent. So the derivation is a complement, sound in one direction:
`CPU-ONLY` (every ancestor is not ours) is a refusal and holds — an optimiser can delete
an event, never invent one. `EP-COVERED` is the weak side and only ever WITHHOLDS `MATCH`.

**The reading that made it real.** Trace-only, against Phi-3.5 on dev0: **65/65 outputs
`EP-COVERED` at an own-count of ZERO.** 459 named events against 363 graph nodes — ORT's
optimisers delete node events wholesale, and the complement is nearly uninformative on a
real model. I would have shipped a mechanism that moved the gap count and not the
coverage; my own criterion-12 lesson, one week old.

Fixed with a second source used only where it accuses us: the claim log's `node`/`claimed`
pairs. Inside the frame, so it may not grant anything — but "we did not claim the node
producing output k" is a self-accusation, and a lying claim log can only make us withhold.
Same run now reads `oracle_outputs_attributed = 0`, `oracle_outputs_vacuous = 65`,
`verdict = UNATTRIBUTED`. Identical on dev1. First time the artifact names its own vacuity
per output.

No sixth verdict token. A comparison that cannot be attributed to this EP is what
`UNATTRIBUTED` already meant, so Link's and Niobe's exhaustive branches stand.
`UNATTRIBUTED` now has two causes with two owners and `explain()` separates them.
`MATCH` is NOT refused for every partial session — Phi-3.5 declines 8 nodes and
"all outputs or nothing" never closes — but the record carries `outputs_reaching_this_ep`
and `outputs_cpu_only` so 65 agreements are never quoted as 65 pieces of evidence.

Falsifiers, because a refusal can be manufactured too: a `CPU-ONLY` output that disagrees
refutes THIS instrument (`refuted_by`, raises `InstrumentError` in the lane, 0 refuted on
both devices); a claim log joining zero graph nodes is `ERROR(instrument)`, because
"every output is vacuous" is exactly the result I went looking for and R13's second
corollary applies; a missing coverage reading is `not-computed`, never clearance.

Verified: 25 GPU-free arms + 2 hardware arms pass on both selectors; 126 passed +
1 xfailed across `test_r13_lane` / `test_verdict` / `test_criterion10_oracle` /
`test_wiring_census` / both new modules on dev0. Criterion 10 still FAIL(condition)
`UNATTRIBUTED` on both devices — inherited, correct, and the bar moved UP not down.
Criterion 10 not closed.

## Round 31 — ORT refuses the session we spent the session learning to detect (2026-08-02)

**Handed to me:** the user asked whether ORT has a flag preventing EP fallback. It does:
`session.disable_cpu_ep_fallback`. Never used here; the string appeared nowhere in `tests/`,
`bench/` or `rust/src/`.

**Probed before designing** (`bench/results/probe_disable_cpu_fallback.py`, both selectors,
ORT 1.28.0). Two things the brief did not have, either of which would have made a naive
wiring actively harmful:

1. With `CPUExecutionProvider` in the providers list — which `EP_PROVIDERS` has — the flag
   is a **configuration conflict**, `INVALID_ARGUMENT`, on **every** graph including one the
   EP claims in full. That refusal looks like a detection and is not. Wiring it into
   `EP_PROVIDERS` sessions would have made a working EP read as broken on every test.
2. A **misspelled config key is accepted silently**. A precondition on a typo is inert and
   green — the specimen shape of this entire session, one level up. Hence
   `assert_no_cpu_fallback_is_live()`: an fp64 `Add` ORT must refuse, run once per process
   **before** first use, not after.

With `providers=[EP_NAME]`: claimed single-op -> created; declined -> refused; partially
claimed -> refused. The first row is load-bearing — ORT plants no CPU nodes of its own at
single-op scale, so the precondition is exact where it is proposed and useless nowhere.

**Built:** `ORT_DISABLE_CPU_FALLBACK_KEY`, `CpuFallbackRefused` (an `AssertionError` —
FAIL(condition), asserted not assumed), `ep_only_session_or_refusal`,
`assert_ep_owns_whole_graph`, `assert_no_cpu_fallback_is_live` in `_models.py`;
`tests/ops/test_no_cpu_fallback.py`, 9 arms, **9 passed on dev0 and dev1**. `check()` gains
an opt-in precondition behind `ONNXRUNTIME_EP_VULKAN_STRICT_NO_CPU_FALLBACK`, off by default.

**Both mechanisms kept, extents written down.** This precondition reaches only graphs the EP
must claim entirely and fires before any number exists; Guard D reaches any graph and fires
after the run. The flag is wrong for Phi-3.5 — ten legitimate declines — which is exactly
where Guard D is indispensable. Two gates whose extents differ compose to the weaker extent
and the stronger name, so I stated them separately in code and artifact rather than letting
a reader infer one gate reaching everything.

**Red-text migration measured**, `test_elementwise.py`, both selectors:
`25 failed / 11 passed` -> `25 failed / 11 passed`, **failing set identical on both devices**.
The count does not improve and was never meant to; the text moves from our
`AssertionError: Vulkan EP claimed 0 nodes` to ORT's own refusal (R13). The identical set is
the better result — it is the falsifier for over-firing, and not one healthy test went red.

**Regression:** verdict + output-attribution + hw + r13 + no-cpu-fallback lanes,
116 passed / 0 FAIL / 0 ERROR on dev0.

**The lesson, and it is not about fallback.** We spent a session building detectors for a
state the runtime would have refused to enter. The user found it. Next time "does the
runtime already refuse this?" comes before "how do we detect this?".

Closes no row.
