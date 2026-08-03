# Link (Platform-Support) — history.md

## Learnings

### [SUMMARY] Sessions 1–6: extension availability, capability set, CI root causes, hardware matrix, LVP2 retraction (2026-07-28–2026-07-30)

**Sessions 1–2 (archived):** OQ-1 measured: `VK_KHR_synchronization2` at 68.57% on Android, `VK_EXT_subgroup_size_control` at 85.88%. Option B (Khronos layer shim) rejected; retained only as optional integrator deployment note. Wgpu/Dawn/Godot claim of Vulkan 1.2 requirement found false — these use extensions, not 1.2 core. OQ-12 hardware experiment defined.

**Session 3 (archived) — CI root causes:**
Windows CI: `VK_ICD_FILENAMES` env var ignored under elevation — must register ICD in `HKLM\SOFTWARE\Khronos\Vulkan\Drivers`. Linux failure was a compile error (not a Vulkan problem); `glslc` arrives via LunarG apt repo.

**Session 4 (2026-07-29T09:19:35-07:00) — hardware matrix:**
First execution-derived hardware data. Both CI lanes CI-VERIFIED (Linux llvmpipe, Windows lavapipe). Both local GPUs local-dev-verified. Intel Iris Xe is the spec-conformance oracle — do not special-case Intel failures. UMA is a first-class platform column (Intel Iris Xe and mobile are UMA; results on Iris Xe are a closer mobile proxy for memory model than RTX 4060). Memory model column added to platform matrix. LVP2 initially recorded (lavapipe `supportedStages=0`) — but see retraction below.

**Session 5 (2026-07-29T09:39:59-07:00) — cross-platform standing directive:**
Cross-platform generality is structural, not a review step. Derive from reported limits. Every `cfg` is a portability hazard (`tests/portability.rs`). Intel is the spec oracle; Intel failures predict MoltenVK failures. No required extensions per §7.2.

**Session 6 (2026-07-29T20:26:56-07:00) — LVP2 RETRACTED:**
LVP2 retracted. Mesa 23.2.1 lavapipe supports subgroup BASIC+ARITHMETIC+BALLOT+SHUFFLE+SHUFFLE_RELATIVE+QUAD in compute, `subgroup_size=8`. The original `supportedStages=0` was the discarded `push_next` chain — our bug, not a device fact. PLATFORMS.md LVP2 entry updated to retraction notice. Lavapipe is UMA (`is_uma=true`, single DEVICE_LOCAL heap), `maxComputeSharedMemorySize=32 KiB`. CI now exercises the mobile-warp path (lavapipe `subgroup_size=8` vs local 32).

**Current state:**
PLATFORMS.md current. Both CI lanes verified. Hardware matrix up to date. M0 criterion 9 met (LVP2 retracted). OQ-12 experiment design defined; hardware borrow needed after M0 for Adreno/Mali. Intel Iris Xe is the spec oracle; every Intel failure is a real portability signal.
---


<!-- SUMMARIZED by Scribe 2026-08-01T20:39:12-07:00 -- older entries condensed below; full text lives in git history -->

### [SUMMARY] Compressed entries (condensed 2026-08-01T20:39:12-07:00)

- **📌 Cross-agent context — Round 4 (2026-07-30T02:49:12-07:00)** — ### Worktree layout and inbox portability constraint The team works in git worktrees: `squad/switch` at `C:\Users\justinchu\dev\ep-vulkan-switch`, `squad/mouse` at `C:\Users\justinchu\dev\ep-vulkan-mouse`, `squad/tank` at `C:\Users\justinchu\dev\ep-vulkan-tank`, with `main` as the integration tree.
- **Session 8 (2026-07-30T08:21-07:00) — Gate artifact design, is_uma verification, subgroup red instrument, lane classification** — ### Context received from coordinator - DESIGN.md §8.9 ruling (b7c2305): `operational` vs `green` distinction.
- **Session 9 (2026-07-31T21:46—22:40-07:00) — Criterion 10's gate wired into the lanes; UNATTRIBUTED; lane re-classification** — **Assignment:** wire criterion 10's gate into the CI lanes; handle the new fourth verdict state `UNATTRIBUTED`; classify every lane `operational` vs `green`; re-state the subgroup-32 falsifier chain end to end.
- **Session 10 (2026-08-01T09:00—10:05-07:00) — the gate closes the loop; ERROR(instrument) made readable:** Merged Trinity's uncommitted `_verdict.py` (copied, not committed — hers) and ran both polarities on both devices on freshly-built EP: positive `GATE: PASS`/`verdict=MATCH` identical digest on RTX 4060 and Intel Iris Xe; negative 1 (no ICD) `FAIL(UNATTRIBUTED)`; negative 2 (new) — a `decline_probe` node the EP doesn't implement — also `FAIL(UNATTRIBUTED)` on both devices, proving "EP could not start" and "EP started and did nothing" are different failures needing different controls. Found the Windows negative control (`VK_ICD_FILENAMES` removal) was likely never effective under elevation (LunarG loader ignores it elevated) — third instance of an instrument outage wearing a detection's costume; added `ci/check_vocabulary.py` preflight with distinct `ERROR(instrument=...)` tokens for "absent from checkout" vs "broken" so lanes are diffable. `ci/test_lane_checks.py` 16→21 passing. §7.4.4 republished (execution evidence does not upgrade lane status — different claims, different falsifiers); §7.10.2 subgroup-32 closed (llama.cpp's subgroup-free fallback + packed-loads/multi-accumulator design confirms lavapipe's subgroup_size=8 constraint is cheap, not costly); §10.0.1 canonical Android sync2 figure (~32.67%, dated); no wall-clock figure anywhere in the gate (708s loaded vs 161s quiet, 4.4× — the source of the "68 failed" false regression). Learnings: a negative control depending on a documented-unreliable mechanism isn't a control; one vocabulary concentrates outage risk, fix is making the outage self-describing not adding a second vocabulary; strongest execution evidence still upgrades zero lanes.

📌 Team update (2026-08-01T09:53:14-07:00): The EP genuinely executes now — 3 VulkanExecutionProvider fused-node events (~355 graph nodes in one fused node) + 24 CPU per run, 65/65 outputs bit-identical, argmax 30751 matching CPU; coverage figures are execution, not offer. All wall-clock figures including 3.1x/3.7x are withdrawn under R13 pending device-clock measurement. Switch holds exclusive claim on device-clock measurement while agents run in parallel. — decided by Scribe

---

## Session 12 — 2026-08-02 — the composed workflow, and planted vs observed falsifiers

**Asked:** classify the `Tautological-assertion screen` step that arrived from Switch's
branch and became unclassified only in the *union* of two correct merges; record union
blindness as a lane-structure property; say whether lanes should verify the composed
workflow. Also: `rust/src/trace.rs` assigned to Niobe.

**Done — commit `596f545` on `squad/link` (not pushed).**

1. `hostfree.tautological_assertions` registered `UNDEMONSTRATED`. Not green. Switch's own
   first paragraph says neither assertion defect that actually occurred here is within its
   reach; 1,056 scanned / 0 detections is a regression barrier, not evidence it works.
   Registering it correctly demoted the whole `lane-checks` lane green -> operational.

2. **Went further than asked, deliberately.** Marking Switch down while calling my own tick
   screen `DEMONSTRATED` would apply a stricter standard to another agent than to myself —
   my screen's red arms are all injections I wrote. `DEMONSTRATED` was conflating "somebody
   planted a defect and the check caught it" with "the check caught something nobody
   planted". Added an orthogonal axis, `PLANTED` / `OBSERVED`, required on every green check
   by `validate()`. Census: build-test-linux 6/8 planted, build-test-windows 5/7,
   lane-checks 4/5. Only `build.portability_lint`, `build.clippy` (both RED_NOW) and
   `hostfree.tick_screen_negative_control` are OBSERVED. **My tick screen is PLANTED.**
   A planted arm does not demote to UNDEMONSTRATED — "somebody performed the mutation" and
   "nobody ever has" are different states — but it must not be read as load-bearing.

3. **My call: lanes verify the composed workflow, not the branch's view of it.**
   `ci/check_lane_inventory.py --union-with <ref> [--union-required]`. Deliberately not a
   merge: a three-way merge can conflict, and a conflict is a different conversation; the
   union of two step-name lists answers the question without needing the merge to succeed,
   and is correct in the case that bit us (different regions, clean merge). `--union-required`
   exists because an unreadable ref silently returns the check to the branch-only view it
   was written to replace. Wired into `lane-checks` after `git fetch --depth=1 origin main`.
   Both polarities hold on a synthesised two-branch repo plus an outage arm, and the real
   merge was **replayed** on the two actual pre-merge blobs: branch-only GREEN, union view
   naming exactly Switch's step. Real inputs, wrong clock — a replay, not a live catch, and
   the code and PLATFORMS.md 7.14.1 both say so.

4. Scope limit, stated so nobody over-quotes it: it covers one shape (workflow step names)
   in one file. The other four instances of 2026-08-02 are untouched; a general union check
   is with Trinity.

5. `trace.rs` owner recorded as Niobe in `ci/tick_conversion_allowlist.json` (all 3 entries)
   and in PLATFORMS.md 7.14.3.

**Verification:** `pytest ci/test_lane_checks.py` 80 passed (65 -> 80, +5 this session);
tick screen PASS; negative control PASS (all 7 arms + baseline); `check_lane_inventory
--union-with main` PASS. Staged per-path — `link-1` shares this worktree.

📌 Team update (2026-08-02T02:03:46-07:00): Niobe's sys.modules correction — a cross-file pytest failure diagnosed as a sys.path leak was actually a module-*name* collision: two files both named device_state.py (your ci/device_state.py and her ench/device_state.py, since renamed to ench/device_companion.py) resolved through sys.modules before sys.path was ever consulted, so the prescribed sys.path fix alone would have left the bug live. ci/device_state.py and ench/device_companion.py cover overlapping subject matter (both carry R9-amendment-5 material on device-state companions) — she flagged this rather than actioning it; it is your file and your call. — decided by Scribe

## Session 13 — 2026-08-02 — criterion 12: extent, an independent whole, name vs content

**Asked:** criterion 12 is NOT met. The census (`unwired: []`) answers whether a mechanism
ran; row 12 also asks for extent, a decomposition identity against an independent whole,
and a name-vs-content check. Build the evidence; **do not close the row** — Trinity owns it.

**Done — commit on `squad/link` (not pushed).** Merged `origin/main` (`57d7018`) first.

1. `ci/check_census_completeness.py` + `ci/census_surface_map.json`. The whole is
   enumerated from production Rust the census does not write: 14 counter fields, 10 trace
   `Phase` variants, 26 `ONNXRUNTIME_EP_VULKAN_*` switches = **50 surfaces**. Numerator and
   denominator now have different authors, different files, different language — which is
   the only reason the count can be wrong. Frame published: 29,269 production lines read,
   **11,376 held out as `#[cfg(test)]`, UNOBSERVABLE by frame**.
   **The answer is not 12/12: 33 censused, 12 instrumented-and-uncensused, 3 out of frame,
   2 not mechanisms.** `GEMV_PACKED` selects a different kernel and nothing observes it.

2. Extent, per mechanism, as an explicit upper bound. `gpu_tracer` **1/12** — it names none
   of the ten phases, one of which is `Record`. `retain_viable` 0/1, its own namesake
   counter. Five host-side mechanisms report **`UNOBSERVABLE`, never 0/0**.

3. Name vs content = R10 on the census's own output. Across six artifacts: 3 `VARIES`,
   **9 `INVARIANT`** (certified on presence alone), and a third state `UNOBSERVABLE` so one
   arm is never called invariant. Every mechanism carries a name claim + discriminator, all
   `name_verified: false`. `Phase::Record` reads INVARIANT — the flag, not the verdict.

4. Twelve-arm negative control. The arm that matters: **the census drops a mechanism whose
   surfaces are still instrumented, and the independent whole names it.** Three planting
   arms prove the denominator is independent. Four outage arms — a denominator that shrinks
   silently is worse than none. The screen also refuses to narrow silently without
   `--no-artifacts` (the `--union-required` lesson, second application).

5. Registered `hostfree.census_completeness` (+ its control) as `DEMONSTRATED` / **PLANTED**;
   new blind spot `census_denominator`; two workflow steps; 8 tests (80 → 88);
   PLATFORMS.md §7.15.

**Discipline I held to:** did not touch `tests/` (Trinity's, and shared with `trinity-1`),
did not move row 12, did not let a PASS from my screen read as census coverage.

**Verification:** screen PASS; negative control 12/12 arms; `check_lane_inventory
--union-with main` PASS; `python tests/union_check.py --run` **PASS** (145 passed, 0 FAIL,
0 ERROR); `pytest ci/test_lane_checks.py` 88 passed. Suite took 768 s under contention
against ~55 s quiet — a 14x spread, and the reason no gate here has a wall clock in it.

---

## Session 15 — 2026-08-02T17:20-07:00 — the device-loss reporting defect

**Task.** Tank hit `vkWaitForFences failed: The logical device has been lost -> CPU fallback -> EXIT = 0` at ctx 512; both his measurement points were truncated and differencing them produced an apparent 6.7% KV saving that was an observation ending early. The cause is Switch's; the reporting defect is mine and survives the cause being fixed.

**Delivered.** `ci/check_device_loss.py`, `ci/negative_control_device_loss.py` (14 arms), `ci/device_loss_incident_records.json`, three lane-inventory checks, one blind spot, four workflow steps, eleven tests, `PLATFORMS.md` 7.18. Commit `93b67ff` on `squad/link`. 98 lane tests pass, union inventory PASS, `tests/union_check.py` PASS. DLL rebuilt after merging `4b5d46b`: `7D3DA69C32DD8BC9` -> `F7E07BE84F278BFC`.

**The LIVE catch.** Pointed at the artifact tree, the screen found a second, earlier device loss nobody had reported: `bench/results/trinity-suite-dev1.log:3216`, 2026-07-31, Intel, `vkQueueSubmit` inside `test_phi35.py:784`, surfaced only as an `AssertionError`. Two days before Tank's, different device, different call site. So the class is not one kernel's bug and the reporting defect outlives whatever Switch fixes. That is the strongest argument for the check and it was made by the check.

**Design decisions worth keeping.**
- The exit status is not an input. The defect IS an exit status of 0, so accepting one would be accepting the defect as a filter. Printed on every run.
- Two tiers, because a tree-wide text scan is red on files that are supposed to contain the text: `broken-commitment-control.json` and the criterion-4/5 witnesses induce those failures deliberately. Tree-wide carries only what no control emits on purpose — the Vulkan spec text and the arithmetic rule. The rest is UNOBSERVABLE unless the caller names the file as one run's evidence.
- Not counting a truncation the producer already filed under `rejected_*`. Counting it would make the honest artifact look like the defective one.
- The exclusion list is the dangerous part, so: reason/owner/date required, excluded files printed every run, rot is a finding, explicit naming overrides.

**Ruling on question 3.** Three mechanisms, three extents, never one guarantee. `disable_cpu_ep_fallback` is planned fallback at session creation and is structurally blind to a loss on a session ORT has accepted. Demonstrated rather than argued: the reach arm feeds both checks a log the EP reports and ORT does not; mine is red, `check_fatal_log` is green.

**Found in someone else's file, routed not patched.** `_verdict.FATAL_LOG_MARKERS` does not match the line ORT actually prints (a list repr), so `check_fatal_log` reads Tank's log — which announces the fallback twice — as clean, having been cited as a second witness for five incidents. Trinity's file, Trinity's fix; the regression test is written to go green when she lands it.

**GEMV_PACKED.** Investigated, not closed. It enters as specialization constant 5 of `q_gemv.comp` and the pipeline cache keys on `(shader_stem, spec_constants)`, so the two settings really are two pipelines — but nothing we produce records a pipeline key or a spec constant. A host-side record of the env var is not R10 evidence. Needs an EP-side emission; owed by Mouse with Switch.

**Sixth union defect, first caught by an instrument.** Merging main turned my census screen red because Trinity had added two mechanisms my map had no name claim for. Both branches complete, composition not. I did not re-disposition the twelve gaps on the strength of her declaration — a declared mechanism is not an observed one.

---

## Session 16 — 2026-08-02 — Cross-platform: the Linux answer

**Branch:** `squad/link` @ `aa6e5b8`. Merged `origin/main` `d375a4d`, rebuilt, DLL hashed
either side (`F7E07BE84F278BFC` -> `A9898AE483110CFF`).

**WSL Ubuntu 24.04 is available on this box** and is what made the task answerable for
real rather than by inference: cargo 1.97.1, Mesa `lvp_icd.json`, python3.12 with
onnxruntime 1.28.0. Two gotchas: `$HOME` inside `wsl -d Ubuntu -- bash -lc` inherits the
Windows value and breaks `LD_LIBRARY_PATH` (use absolute `/home/justinchu/...`), and
`CARGO_TARGET_DIR` must point inside WSL's own filesystem or it clobbers the Windows
`rust/target` and races `link-1`.

**Q1: builds clean. `cargo test --lib` does not compile** — 11 `i32`/`u32` errors on
`ort::OrtLoggingLevel_*` in `rust/src/ep.rs`. Routed, not patched.

**The CI "Clippy" step is a portability defect with a lint's name.** That is why it sat
low-priority for a day. Second `misnamed` specimen after `Phase::Record`.

**A red step skips the rest of its job — seven Linux steps have never run.** My table
said `UNDEMONSTRATED`, which flatters. New status `GATED_NEVER_RUN`, and I deleted the
`observed` date from `device.op_correctness`. *Running a while is not the falsifier* has
an other half: **a check that has never run is not yet a check.**

**Q2: lavapipe passes the gate (`subgroup_size = 8`, measured). The EP claims 0/1 and
exits 0.**

**Q3, the one that matters: the ledger is keyed per TOOLCHAIN.** `shader_digest` hashes
the embedded SPIR-V, so Ubuntu's glslc faults all 74 entries with no kernel change; and
`device` is on 74 of 75 entries and never read by the predicate. On Windows device1 the
EP prints *"proven ... on device0"* and claims the form anyway. Digest disagreeing fails
safe; digest agreeing on an unproven device fails **open**, and nothing watches it.
Blind spot `ledger_device_provenance`, substitute **NONE**. Cross-platform is an
architecture problem — **it would survive buying every runner, because a bought runner
brings its own glslc.**

**The reporting defect is mine.** conftest calls the EP's decision "No Vulkan device
available" on a box whose gate just passed; the run ends `2 passed, 36 skipped`. Fixing
clippy alone would have made op-correctness green having asserted nothing — the
narrowing I was told not to do, reached without anyone narrowing anything.

`ci/check_ledger_portability.py` + `ci/negative_control_ledger_portability.py`:
**3 LIVE / 0 REPLAYED / 8 PLANTED**, all three conditions with a live arm. It never reads
an exit status and a test asserts the source never mentions `returncode` — the defect is
an exit status of 0.

13 new lane tests, 111 passing, 3 known census reds. `PLATFORMS.md` §7.20 (§7.19 was
taken by a merge). Decision record:
`.squad/decisions/inbox/link-ledger-toolchain-not-device.md`.

**Method note to self:** I bypassed the pytest fixture on my first probe and got a false
"EP not registered". Caught it by re-running under pytest and seeing `get_ep_devices()`
list the EP. Confirm the harness's own path before reading a probe as a finding.

---

## Session 13 — 2026-08-02 — Review of Trinity's marker fix: her call upheld, two defects of mine exposed by it

Merged `origin/main` at `57a7f62` (`2e1f133`). Diff was **Python-only**; DLL SHA-256 `A9898AE4…B7379` identical either side of the rebuild — the falsifiable form of "no Rust changed".

### The ruling I was asked to confirm rather than assume

Trinity declined to widen `FATAL_LOG_MARKERS` to `The logical device has been lost`, on the grounds that my negative control proves `check_device_loss` has its own reach by requiring `check_fatal_log` to stay green there. **Confirmed, and it is executed rather than commented** — an arm plants an EP-reports-loss/ORT-silent log, requires `check_device_loss` red, then runs `check_fatal_log` on the same file and requires exit 0. Widening would have flipped it and reddened the control, correctly: `check_device_loss` would then have no *demonstrated* reach beyond `check_fatal_log`, and two checks would rest on an argument. **Coverage bought that way deletes the evidence for the thing buying it.** Upheld, and now asserted from the other side in `ci/test_lane_checks.py` so it cannot be widened quietly in either file.

Her liveness check's placement is also right, and deliberately so: it runs ahead of my `--lane-marker` exit-0 branch. The marker branch avoids a second red about a *subject* that never existed; a blind witness is not a fact about the subject at all but a standing defect invalidating every green the check ever produced. Unconditional is correct.

### Two defects of mine, the same shape as hers, one layer up

1. **`marker_cross_check()` compared a physical line to a matched span by string equality.** Fine while `find_fatal_log_lines` returned whole lines; a form-tolerant matcher returns the *span*, so `...and retrying` never equals a line carrying `...and retrying.`. It had become a test of punctuation, and kept emitting `marker_list_misses_real_line` — *"so check_fatal_log reads this log as clean"* — about a file `check_fatal_log` now exits 1 on, quoting three lines. **And an arm of my own control required that stale finding**, so anyone repairing the cross-check would have reddened my control and been told the repair was the defect. R9 amendment 5 inside a control written to enforce R9 amendment 5.
   → **An arm must assert the property we want, never the defect we currently have.** A red arm written against a known defect becomes a lock on the broken state the moment it is fixed.
2. **`check_device_loss.py` scanned un-normalised text — blind to UTF-16LE, including for device-lost text, its own primary condition.** Found by a *new* arm on that arm's first run, not by reading. On a wide-only log it would have reported a clean run and the cross-check would have reported *agreement*, because both scanners were blind. Two blind scanners agree perfectly. Both sides now go through `_verdict.normalise_log_text` — delegated, not copied; if it is unavailable the raw text is returned rather than a private substitute, because a wrong answer from a copy looks maintained.

### A vacuous test, from a habit to drop

`test_the_shared_marker_list_still_misses_the_real_ort_line` was written to go green when Trinity fixed the markers. It did — **by skipping its only branch and asserting nothing.** `xfail`-on-absence goes *quiet* on repair rather than *strict*. Rewritten to assert the agreement in all three forms a real capture arrives in (list repr, wrapped, UTF-16LE) plus the extent deliberately not covered.

### State

`negative_control_device_loss.py`: **18 arms, 1 LIVE / 4 REPLAYED / 13 PLANTED**, all fired (was 14) — the count rose because the fix added falsifiers, which is the only reason a rising count means anything. `ci/` 113 passed, 3 failed (the known census reds, not mine). `check_lane_inventory.py` PASS. Stale arm counts corrected in `lane_inventory.py`, and one exclusion record in `device_loss_incident_records.json` whose stated *reason* was the now-repaired defect.

Committed `d8fce9f` on `squad/link`, staged by explicit path. Not pushed. No wall-clock figure quoted.

**Worktree hazard, reported:** the other `link` instance committed to `ci/test_lane_checks.py` between two of my commands, so my saved diff of its uncommitted state came back empty. No work lost, and only because their commit landed first. The failure mode is silent — an empty patch reads exactly like a clean file.

📌 Team update (2026-08-02T22:37:04-07:00): Trinity's `trinity-fatal-log-witness-was-blind` finding — `_verdict.FATAL_LOG_MARKERS` never matched ORT's real list-repr fallback text; all twelve historical hits were `test_phi35.py`'s own docstring echoed back by pytest. `ci/check_fatal_log.py` was cited as second witness alongside your device-loss work for five incidents on the strength of a match it could not make — those five should be re-read as single-witness (yours) until her liveness-gated fix is reviewed. She deliberately did not widen the markers to also catch device-lost text, because doing so would have reddened your own negative control (the arm proving `check_device_loss` has reach `check_fatal_log` does not). — decided by Trinity


---

## Session 14 — 2026-08-02 — The eleven errors repaired, the seven steps run, and a method of mine retired

Merged `main` (fast-forward to `6ef62bb`, Switch's `872d739` device-authoritative KV spans included).

### The defect was a binding's width, not a signedness bug — and that determined the shape of the fix

Read both generated `ort.rs` files rather than inferring: MSVC emits
`OrtLoggingLevel = c_int`, GCC emits `c_uint`, because a C enum with no negative
enumerator is signed for one and unsigned for the other. Values `0..=4` on both. No
arithmetic anywhere, and the value arrives typed as the alias from ORT's own callback.
**Representational, not a real signedness bug.** So: three declarations in `ep.rs`'s test
tree now carry `ort::OrtLoggingLevel` instead of a spelled `i32`. Eleven errors, three
lines. **I rejected the eleven-cast form on principle** — `as i32` compiles on both
platforms while preserving the exact assumption that caused the bug — and then made that
rejection executable: portability rule **P3** has an arm that goes red on the cast form
specifically. P1 catches *a name that is not there*; P3 catches *a name that is there and
is a different width*. Portability lint 8 → 14 tests, green on both platforms.

P3's scope is a `mod` block, not a file, and the limit is stated in the source rather than
hidden: `logging.rs` and `mock_ort/mod.rs` handle severities at top level and also carry a
vendor id and a line number as `u32`, and **a lint that reports those trains people to
ignore it.** P3 is an early warning. The decisive second-platform check is the new compile
step.

### The split, and the class

`cargo build --release` compiles the **lib only**, so clippy was the lane's first
`--all-targets` invocation and therefore the first thing that ever compiled the tests.
`Compile all targets (cargo check --all-targets — compile errors, not lints)` now runs
ahead of it on both lanes, deliberately without `-D warnings` so the two reds are
different findings. Inventory entry `build.compile_all_targets` added — and the lane
checker caught the missing entry itself, which is the mechanism working.

The general form, worth more than the incident: **the audit question is not "is this step
named well" but "what is this step the *first* thing in its job to do".** Both names were
defensible in isolation; the gap between them was not.

### The seven steps: run, and mostly still red — which is the finding

PASS: layering (26), portability (14), the four integration targets, `epctl
--probe-loader`, `check_fatal_log`. FAIL: `cargo test --lib` 481/492, op-correctness 50
failed / 272 passed / **292 skipped** (Windows skips 30), Criterion 10 + both verdict
readers `UNATTRIBUTED` with 0 dispatches, and — correctly — `check_ledger_portability`
`FAIL(condition=claimed_nothing)`.

**All of it is the ledger digest, and I demonstrated that rather than arguing it:**
perturbing one GLSL template *on Windows* so the digest stops matching reproduces the same
eleven `cargo test --lib` names Linux shows, plus one — a superset. 48 of the 50 pytest
failures do not occur on Windows. A platform-specific symptom with a platform-independent
cause, and the ledger-keying decision is now the *only* thing between Linux and a
meaningful op-correctness result.

### A method of mine was unsound, and I had written it down as evidence

Session 13: *"DLL SHA-256 identical either side of the rebuild — the falsifiable form of
'no Rust changed'."* Measured: the Linux `.so` is byte-identical across four forced
rebuilds; the Windows `.dll` gave **six distinct hashes from six builds of an unchanged
tree.** On Windows an identical hash means **cargo did not relink** — which is exactly what
a fingerprint-fresh tree produces, and reads exactly like "the bytes are the same". Same
shape as the empty patch. The no-shipped-code-changed claim is now structural instead:
every edited line is inside `#[cfg(test)] mod tests` (`ep.rs:2681`), which the cdylib does
not contain. I found this only because I went to hash the two artifacts as the brief
asked; had the brief not asked, I would have quoted the old method again.

### Surprises

* **Clippy was never the problem and is green on Linux today** — it had nothing to say
  about a crate that could not compile.
* **A comment-only shader edit does not change the digest** (comments are stripped), so my
  first attempt at faulting the ledger silently proved nothing. Needed a semantically inert
  *statement*.
* **One optional Python dependency zeroes the whole op-correctness step.**
  `test_shape_inference_delta.py` raises `ImportError` at **collection**, so pytest reports
  `Interrupted: 1 error during collection` and the directory asserts nothing. CI installs
  `tests/requirements.txt` and never sees it.
* `vk::barrier::tests::backend_probe_writes_legacy_token` failed **once in nine** full
  Linux runs, passes in isolation. Order-dependent. Recorded, not diagnosed.

### State

Windows `cargo test --lib` 492/0/4 and `clippy --all-targets -D warnings` exit 0, before
and after. Linux `cargo check --all-targets` and clippy both exit 0. `ci/` 113 passed, 3
failed (the same known census reds). `check_lane_inventory.py` PASS with the union against
`main`. `PLATFORMS.md` §7.21. Reproducers under `ci/link-linux-repro/` with a README
stating the WSL constraints (`HOME`, `CARGO_TARGET_DIR`, LF endings), explicitly **not**
lane checks. Two decision records: `link-misnamed-is-a-defect-class`,
`link-ortlogginglevel-width-and-the-seven-steps`. Committed on `squad/link`, staged by
explicit path, not pushed. No wall-clock figure quoted.

**Worktree note:** `squad/link` carried eleven untracked `bench/results/*.json` files from
a sibling on arrival. Left untouched and unstaged rather than committed or cleaned — an
untracked artifact of unknown provenance is somebody's evidence until they say otherwise.

---

## Session 17 — 2026-08-03

**Task:** close the six open lanes from the sweep, re-examine the one accepted, build the
general fix for `BUILD_SKIPPED`, characterise the 12 Windows-only failures, re-run Linux
after Mouse's two-digest merge, and make an intermittent visible to someone merging.

### What surprised me

* **"Windows-only" was a category error and it was mine.** The 12 became **1**, and the
  survivor is Windows-only because **Windows is the only lane with a device that executes
  phi-3.5** — lavapipe declines to CPU. It meant *only observable on Windows*, never
  *broken on Windows*. The survivor is a deterministic fp16 accumulation finding: ULP
  rises monotonically L0 `0.0` → L31 `4.0`, only layer 31 key/value over the 3.0 band.
  Not an instrument error, not intermittent. Routed to Mouse/Switch.
* **My own screen convicted my own previous decision.** BP2 (`dead_guard`) fired **38
  times on my tree**. Session 16 left the `BUILD_SKIPPED` guards in "so the change reads
  as one deletion rather than thirty". A guard whose writer does not exist is not inert,
  it is **dormant** — it reads like a live guard and one added line re-arms all 38. Deleted.
* **The negative control caught a bug in the screen on the control's first run.**
  `screen()` returned exit 1 and never printed its `FAIL(condition=...)` token; `_fail()`
  existed and was never called. A red with no condition name is the exact defect R13
  exists to prevent.
* **The flake I went hunting has vanished.** `backend_probe_writes_legacy_token` was
  1-in-9 last round. **40/40 green** at `d46327b` (p ≈ 0.009 under the old rate). Somebody
  fixed it and nobody recorded fixing it.
* **The aggregate floor would have passed the LIVE arm.** With `cdylib_load` `#[cfg]`-ed
  out, 10 executed against a floor of 10. Only `target_ran_nothing` caught it.

### Numbers

Linux `23 failed / 592 passed / 45 skipped / 3 xfailed` (was 50/272/292); `no proof ledger
entry for` **42 → 3** (all the new `Cast` op). Windows `8 failed / 624 passed / 28 skipped
/ 3 xfailed` (was 14/568/30). Both lanes still mix kinds: Linux 23 = 8 `FAIL(condition)` +
15 `ERROR(instrument)`; Windows 8 = 6 + 2.

### Built

`ci/check_build_precondition.py` (BP1/BP2/BP3) + 16-arm control (1 LIVE / 2 REPLAYED / 13
PLANTED, one REPLAYED reading the real defect out of `607056a`).
`ci/check_flake_witness.py` (annotate the name where truncation cannot reach it; join
across runs at one commit) + 13-arm control (1 LIVE / 1 REPLAYED / 11 PLANTED).
`check_suite_productivity.py` gained per-target parsing, `target_ran_nothing`,
`min_target_blocks`/`targets_below_floor`. Six new floor steps, floors measured on Windows
**and confirmed on Linux**. Conformance census floor, not `continue-on-error`.

### State

Windows `cargo test --lib` **521/0/4** and `clippy --all-targets -D warnings` exit 0.
`ci/test_lane_checks.py` **132 passed, 3 failed** (the same known census reds; was 116
passed). `check_lane_inventory.py` PASS. `check_build_precondition.py` PASS over both
workflows. `PLATFORMS.md` §7.23. Five decision records in the inbox. Committed on
`squad/link`, staged by explicit path, **not pushed**. No wall-clock figure quoted.

**Mistake worth recording:** `[IO.File]::WriteAllText` with a relative path resolves
against the **process** CWD, which is the *main* repo, not PowerShell's `cd`. I wrote into
`onnxruntime-ep-vulkan/.github/workflows/ci.yml` by accident, caught it with `git status`
and reverted. Always pass absolute paths to .NET file APIs from this harness.

**Worktree note:** 23 sibling untracked artifacts present on arrival, left untouched.

## Session 18 — three checks that were failing into a void

Trinity handed back three reds that reproduce in `main`'s own checkout:
`audit_instruments --check` FAIL(drift), `ci/test_lane_checks.py` 3 red,
`tests/union_check.py --run` 5 red. All three were already printing their
failure in full. None was wired to anything that reads an exit code.

**The thing that surprised me was in my own reporting.** For four sessions I
wrote the lane-check suite up as "132 passed, 3 failed (the known census reds)".
After merging `main` it was **4**, and the fourth was a real regression of mine:
the merge reintroduced four `if: env.BUILD_SKIPPED != '1'` guards whose writer
my own previous commit had deleted, arriving on Trinity's new contention-gate
steps. My own build-precondition screen caught it. Reading the node ids found
it; reading the count would not have — and "the known reds" is a count.

**An accepted red and a new red are indistinguishable when the only record of
the acceptance is a number in someone's head.**

Built `ci/check_open_reds.py` + `ci/open_reds.json`: every declared guard states
the colour it is expected to be, both colours falsifiable. `unaccounted_red`
(new failure), `stale_acceptance` (good news — delete the entry, and this is the
arm that stops the register rotting), `signature_changed` (the acceptance does
not stretch), `lease_expired` (`review_by`; acceptance is a lease, not a grant,
and yes it is deliberately a time bomb). Accepted reds are annotated into the
merge UI with owner and closing condition — check-run metadata, which a
truncated log cannot eat.

Its first run convicted the shipped tree: `unaccounted_red lane_checks_suite`.
So the entry is **narrowed** — suite expected green with the three census-extent
tests `--deselect`ed, those three their own accepted-red entry — because
accepting the suite whole would absorb every future red in 135 tests. Trinity's
principle: narrowing is the amplifier, not a cost saving. A test asserts the
deselected set equals the selected set, so nothing falls out of both.

Five accepted reds, none of them mine: Mouse's nine uninvoked two-digest
accessors (twice, through two gates), the 12 census-extent surfaces
(Mouse/Switch/Tank), Switch's kv caller-bind reading where the tracked artifact
now says `OK` and the test asserts `CUDA`, and Mouse's proof-ledger writer
regression. Writing the third down is the first time anyone has had to say whose
the "known census reds" are.

`union_check`'s five are declared one at a time, because
`FAIL(condition=union_red)` sums 3 FAIL(condition) with 2 ERROR(instrument) —
the same incommensurable-sum defect as the "48". The device-only one is declared
**absent**, because it is green-by-skip host-free and an entry a skip can
satisfy is an acceptance granted by an absence.

One half of the census red was mine after all: the frame arm fired on two
undeclared `bench/` modules (`spirv_simt.py`, `test_weight_reread.py`). Declared
with reasons; `spirv_simt.py` held out as capture on its own docstring's
evidence.

Evidence: `ci/negative_control_open_reds.py` 39/39 (2 LIVE, 3 REPLAYED, 34
PLANTED, ratio printed); REPLAYED takes the real `ci.yml` at `133b9fe` and
requires an unaccounted red while today's bytes stay green. 18 new two-polarity
tests (150 passed, 3 accounted red). Windows `cargo test --lib` 527 passed 0
failed, clippy clean. All four negative controls green. Lane inventory PASS.
