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
