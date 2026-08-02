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

## Round 20 (2026-07-31T00:26:22-07:00) -- Guard D: runtime-fallback vacuous-pass closed

Trigger: Allocator::alloc(size=0) returns None for KV-cache [1,32,0,96] inputs. ORT prints EP_FAIL, falls back to CPU silently. get_providers() still shows EP. test_phi35_vulkan_matches_cpu_logits PASSED (CPU-vs-CPU). Fourth fallback trap instance.

Fixes (commit 5b3518b):
- _models.py: count_vulkan_executions_from_profile, assert_vulkan_executed_runtime (Guard D)
- test_phi35.py: Guard D in 4 tests (matches_cpu, cross_run, f16_logits, multirun_stable)
- test_wiring_census.py: model_output_equivalence validity condition documented
- Device labels fixed (commit 00c576c): selector 0=NVIDIA, selector 1=Intel

Needed from Switch:
1. alloc.rs: zero-size alloc must return valid zero-length buffer, not None
2. Remove #[ignore] from ep_messenger_fires_for_planted_fence_leak

---

## Round 23 (2026-08-01) — R13 reaches the subprocesses; criterion 3's frame gate

Merged `origin/main@efbf18c` first (fast-forward; the branch was 10 commits behind and every
number below is post-merge). Round 22's work — `_verdict.py`, `test_criterion10.py`,
`test_verdict.py`, `test_r13_lane.py`, the `counters.rs`/`epctl.rs` record splice — was still
uncommitted in the worktree; it survived the merge, was re-verified against the rebuilt DLL,
and is committed here together with this round.

### 1. Criterion 3's last open item — closed, both devices

Switch: *gate the wrapper on `Instance::validation_armed()` so an unarmed machine reports
ERROR(instrument) rather than green.* Done, and the `#[ignore]` stands — his reason is right:
a process-wide `EP_VALIDATION_ERROR_COUNT` with an instance-scoped messenger is only sound
when the test owns the process, so subprocess isolation is the mechanism, not a workaround.

`epctl --probe-validation` is the in-lane predicate. `classify_validation_probe()` ->
ARMED / LAYER-ABSENT / NO-LOADER / PROBE-ERROR, and **exit 0 with no ARMED line is
PROBE-ERROR** — exit 0 alone is not the observation. `require_validation_armed()` raises
`InstrumentError` for everything else, so an unarmed machine is neither green (the old
`pytest.skip`) nor a detection (which would accuse Switch's messenger of a defect the
machine made unobservable). R12: UNOBSERVABLE, never 0.

**Not a code reading.** `test_the_armed_gate_changes_its_answer_when_the_layer_is_removed`
strips `VK_LAYER_PATH` on a real epctl process here and requires the classifier to change its
answer: ARMED -> LAYER-ABSENT, gate refuses. If a loader still armed under the redirection the
test raises `InstrumentError` rather than passing — a simulation that did not take establishes
nothing.

Both devices: `EP_VALIDATION_ERROR_COUNT = 1`, frame ARMED checked first. 3 passed each.

`classify_plant_run()` splits the control's own transcript three ways. The branch that
matters: **no artifact line at all is ERROR**, because a control that crashed before the plant
and a control that watched the plant fail both exit non-zero — a bare `assert returncode == 0`
scores the crash as a detection, which is the Guard D `NameError` verbatim. A count of `0` IS
an observation (armed frame checked first) and is the only genuine FAIL in the table.

### 2. The census timeout — and outages leave the detection channel

`test_wiring_census` shelled out on 60 s / 120 s budgets calibrated for a quiet machine.
`run_subprocess_checked(quiet_seconds=N)` now inflates every budget by 6.0 (Niobe's measured
worst case is 4.4×; the extra is headroom, because over-waiting costs minutes and under-waiting
costs a fabricated regression the whole team then investigates) with a 120 s floor and a
`$ONNXRUNTIME_EP_VULKAN_TIMEOUT_SCALE` override that ignores junk values rather than falling
to zero.

`TimeoutExpired` / `FileNotFoundError` / `OSError` become `InstrumentError`. An unobservable
mechanism records `INSTRUMENT-ERROR (...)` and **never enters `failures`**, so a timed-out
mechanism is not reported as UNWIRED. Any instrument error then raises after the
mandatory-wired assertion — ordering deliberate: a real UNWIRED outranks an outage elsewhere,
because it was actually observed.

Selector 1: **2 passed, 1 xfailed in 468 s** — nearly 4× the old 120 s budget for one of its
two subprocesses, which is exactly why it had to be `--ignore`d. Selector 0: 45 s warm, census
line now carries the frame: `verdict=MATCH executed_by={'CPUExecutionProvider': 24,
'VulkanExecutionProvider': 3} source=ort_profile permits_triple=True`.

No assertion anywhere in this harness compares a wall-clock duration to a threshold. A timeout
is a ceiling on waiting, not a measurement.

### 3. XPASS is a fourth token, and it was in the outage bucket

The full run classified `test_gqa_present_kv_shape[0]` — an `xfail(strict)` that PASSED — as
ERROR(instrument), i.e. into the bucket labelled *none of these is evidence about the EP*. It
is evidence about the EP: **the condition the xfail recorded has been fixed.** R13's third
corollary is that a result contradicting a prediction deserves more scrutiny than one
confirming it, and burying somebody else's good news in the outage bucket guarantees it gets
none. `XPASS(stale expect)` is now its own token with its own falsifier and a paired control
that fails if the other three states stop being reachable.

**Switch:** `test_gqa_present_kv_shape[0]` XPASSes on selector 1. Your zero-size-alloc fix
appears to have landed; the xfail reason (`absent optional inputs produce size=0 alloc
requests; EP falls back to CPU`) is stale for that parametrisation only. Yours to remove.

### 4. The harness screen could not see the guards this project now rests on

`audit_instruments.py` scanned `ops/_models.py` alone, so it reported *9 harness instruments*
while everything §10.0's third amendment and R13 rest on sat outside its frame — a census
reporting a number about a world it has not surveyed. Added `ops/_verdict.py` and the
`classify_` prefix; 12 instruments now, `require_validation_armed` **SCREENED**.

**Stated limit rather than a silent exclusion:** `classify_validation_probe` and
`classify_plant_run` read UNFALSIFIED and the screen is wrong about them. Its polarity model
is raise-based; these are total functions that return the token instead of raising it. The
generalisation is mechanically decidable and not implemented in this pass — **value-polarity:
screened iff a non-gated test asserts two different return values for two different inputs.**
In `hand.harness_notes`. Baseline rewritten; it also absorbs `offer_shared_device` becoming
wired (Switch, §6.5), which I looked at rather than ignored.

### 5. A stale literal that would have punished progress

`test_verdict.py` pinned `"353"` as the Phi-3.5 fusion ratio. It is 355 of 363 since Mouse
claimed `SimplifiedLayerNormalization` and `Gather`, so that assertion would have gone red for
the EP doing MORE work. Replaced with the property: the description must carry
`<claimed> of <total> graph nodes` whose claimed figure exceeds 100× the island count.

Same discipline in `test_criterion10.py`, and it paid: the CPU count fell **30 -> 24** on both
devices this round, for exactly that good reason. A suite that pinned `30` would be red.

### Results (post-merge, DLL rebuilt from this tree)

| lane | selector 0 (RTX 4060) | selector 1 (Iris Xe) |
|---|---|---|
| `test_criterion10.py` | 2 passed — 3 VulkanEP / 24 CPU, argmax 30751 ×3, identical, MATCH | 2 passed — identical figures |
| `test_validation.py` | 3 passed (3a/3b/3c) | 3 passed + 3d gate falsifier |
| `test_wiring_census.py` | 2 passed, 1 xfailed, 45 s | 2 passed, 1 xfailed, 469 s |
| full `tests/ops` | — | **351 passed / 32 failed** in 1018 s, census INCLUDED |

R13 splits that 32 into **31 FAIL(condition) + 1 XPASS**, and the 31 are one cause: §8.9
evidence gating declines `Min`/`Max`/`Cast`/comparisons/bitwise as `[staged]` against a suite
that still asserts they are claimed. Policy/expectation mismatch, still not mine to patch, and
I am still not loosening 31 assertions to make a suite green. Justin's baseline was 32 failed
/ 272 passed with the census `--ignore`d: **same failures, 79 more passes, census back in.**

GPU-free: 95 passed (`test_r13_lane` 37, `test_verdict` 44, `test_harness_census` 7,
`test_guard_d` 12 — ERROR(instrument) 0). Rust: epctl 13, counters 10,
`validation_control` 3. `bench/test_contention.py` 62.

### Owed to others
- **Switch:** the stale GQA xfail (above). `counters.rs` additive splice from Round 22 stands.
- **Niobe:** vocabulary decision from Round 22 plus this round's timeout wrapper — if a bench
  probe shells out, `_verdict.run_subprocess_checked` gives it a contention-tolerant budget
  and turns a hang into ERROR(instrument) instead of a fabricated red.
- **Tank:** `audit_instruments.py` extended again, additively (two entries in the file list
  and one regex alternative). Say the word and I move the harness domain out of your file.
- **Me, next:** value-polarity in the harness screen; then `assert_matches_cpu`, still
  UNFALSIFIED and still the correctness oracle; then wire `assert_qdq_reference_oracle_safe`.

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
