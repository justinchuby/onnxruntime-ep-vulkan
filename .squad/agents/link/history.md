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
- **Session 10 — 2026-08-01 — Obligation 8 lands in the lanes; a negative control turns out to have been a constant** — Merged `origin/main` to `ef58b4a` first (the merge aborted once on a stale untracked `tests/ops/_verdict.py` I had been carrying — Trinity's file, deleted, remerged clean).

## Session 10 (2026-08-01T09:00—10:05-07:00) — The gate closes the loop; ERROR(instrument) made readable

**Assignment:** merge Trinity's `_verdict.py` and run the gate on both devices, passing for
the right reason and failing when the EP does nothing; say how a maintainer tells a
legitimate vocabulary outage from a broken lane; publish the `operational`/`green` table for
every lane; close the subgroup-32 argument; keep OQ-12 honest. Worktree `squad/link`,
fast-forwarded to `origin/main` (5eda83b). Committed 304e26d. Not pushed.

### Merge-order reality

`tests/ops/_verdict.py` is **still uncommitted in Trinity's worktree** — `git log --all`
has no commit touching it. I copied her working file into my worktree to run against, and
did **not** commit it (it is hers; `git add` was per-path, never `-A`). Everything below
was measured against that file at sha256:7ff8f55e5b3da82b, 64663 bytes. The dependency is
real and unchanged: until it lands, my lanes report
`ERROR(instrument=verdict_vocabulary_unavailable)` — which is now, at least, a *readable*
outage rather than an ambient one.

### Both polarities, both devices, on merged main

Built the EP fresh in my own worktree (8m33s) rather than borrowing another agent's DLL.

- **Positive, RTX 4060 (selector 0) and Intel Iris Xe (selector 1):** `GATE: PASS`,
  `verdict=MATCH`, `executed_by={'VulkanExecutionProvider': 1}`,
  `attribution_source=ort_profile`, `counters_dispatches_executed=2`,
  `profile_node_events=1`, `max_abs_diff=0`, artifact digest
  `sha256:aba0cd3847ec28ac` identical on both. `ci/check_verdict.py` → `VERDICT-CHECK:
  PASS` (separate process, separate parser); `epctl --check-counters` → PASS.
  Re-verified unchanged after the artifact refactor — same digest.
- **Negative 1 (no ICD):** `FAIL(condition=UNATTRIBUTED)`, exit 1,
  `executed_by={'CPUExecutionProvider': 2}`, `permits_triple_and_ratio=false`.
- **Negative 2 — new, and the one that matters:** `--artifact decline_probe`, a single
  `Det` node the EP does not implement. Loader untouched, driver healthy, device passing
  the §7.2 gate, EP loaded — and the EP claims nothing.
  `FAIL(condition=UNATTRIBUTED)` on **both** devices, exit 1, `profile_node_events = 0 of
  1 total`. `check_verdict.py` agrees.

Neither negative reported `DIVERGENT`, and both were checked for it.

**Why negative 2 exists.** Negative 1 proves the gate notices a missing *driver*. The
failure actually live on 2026-07-30 was a **healthy** EP that executed nothing, and no
control on this project reproduced that state until today. "The EP could not start" and
"the EP started and did nothing" are different failures with different owners.

### The Windows defect I found while building it

The Windows lane's only negative control removed the ICD via `VK_DRIVER_FILES` /
`VK_ICD_FILENAMES`. **§7.4.1 of my own document records that the LunarG loader silently
ignores both in elevated processes**, and GitHub's Windows runners are elevated — which is
precisely why that lane registers lavapipe in the *registry*. So that step may never have
suppressed anything on any run it ever made; and if it did not, the gate executed, passed,
and the step reported `NEGATIVE CONTROL FAILED: the gate passed with no Vulkan ICD
present` — blaming the gate for a control that never fired. An instrument outage wearing a
detection's costume, third instance of that shape on this project (after the splice
ordering and the timing "68 failed"). It now probes the loader first and reports
`ERROR(instrument=icd_suppression_ineffective)` while asserting nothing. The decline probe
is loader-independent and is what that lane's falsifier claim now actually rests on.

### `ERROR(instrument)` must not become the weather — `ci/check_vocabulary.py`

The hazard is in my own design: one vocabulary means one point of total outage. Two very
different situations print the identical line — (a) this checkout does not carry
`_verdict.py`, (b) the file is there and the job cannot load it. Preflight, before the gate,
in every lane:

| Exit | Token | Meaning |
|---|---|---|
| 0 | `VOCAB: PASS` | present and importable — **so any later `verdict_vocabulary_unavailable` in this job is a lane fault by elimination** |
| 4 | `ERROR(instrument=verdict_vocabulary_absent_from_checkout)` | repository state; no CI change fixes it |
| 4 | `ERROR(instrument=verdict_vocabulary_broken)` | lane or source defect, exception text quoted |

Same exit code (both are outages, neither is a detection); **deliberately different
tokens**, because the token is what a maintainer greps. Prints path, sha256, byte count,
git-tracked status, commit and Python version on every path, so two lanes can be *diffed*:
all lanes `absent` on one commit → repository state; one lane `PASS` and another
`unavailable` → that lane is broken. `--github-summary` writes it to
`$GITHUB_STEP_SUMMARY` with a per-token annotation title, so the difference is on the
summary page and not buried in a log. All three states verified locally with their exit
codes.

`ci/test_lane_checks.py`: 16 → **21 tests, 21 passing**, including one asserting only that
the two outage tokens are not equal — its own test because it is the property, not a side
effect.

### Docs

- **§7.4.4 republished** (2026-08-01) — `operational`/`green` for every lane, seven "still
  not green" items. Added: the Phi-3.5 execution evidence (3 VulkanEP node events, one
  fused node ≈355 graph nodes, +24 CPUEP; 65 outputs bit-identical across three runs;
  argmax 30751; Mouse: 355 claimed / 1 island / 8 permanent declines) **does not upgrade any
  lane** — execution evidence and lane evidence are different claims with different
  falsifiers. Selector 1 row now says `SPLIT-DEVICE` (§6.5) rather than the old
  `ep.device_index` TODO.
- **§7.10.2 — subgroup-32 closed.** llama.cpp ships a subgroup-free shared-tree fallback
  with capability-gated subgroup variants; packed loads and multiple accumulators, not
  subgroup ops, are the stronger performance gap. So our constraint is cheap and lavapipe's
  `subgroup_size = 8` remains the falsifier by construction. Still a guarantee about an
  empty set (0 of 168+ variants use subgroup intrinsics) — said plainly, not dressed as
  coverage. Re-open only on the four named break conditions.
- **§10.0.1 — one canonical form for the Android sync2 figure**: ~32.67% as of 2026-07-30,
  simultaneously a ceiling and a floor, never a bare percentage. Every remaining 31.43%
  citation (§8.3, §8.4, the drop-condition table, §10.0.2) now carries its date. Standing
  refusal restated in the canonical block: **lavapipe is not Adreno and is not Mali**; no
  lavapipe result may be cited as Android evidence.

### No timing figure

None added, none quoted, and no wall-clock threshold anywhere in the gate. The suite ran
708 s loaded and 161 s quiet — 4.4× — and I once read the resulting timeouts as a "68
failed" regression that did not exist. That is R13's second clause with a stopwatch on it.

### Learnings

- A negative control that depends on a mechanism documented elsewhere in my own file as
  unreliable is not a control. Check that the suppression *took* before asserting on what
  it suppressed.
- One vocabulary is right and it concentrates outage risk into one place; the fix is not
  two vocabularies, it is making the one outage self-describing.
- The strongest execution evidence on the project still upgrades zero lanes. Resisting that
  substitution is most of the job.

📌 Team update (2026-08-01T09:53:14-07:00): The EP genuinely executes now — 3 VulkanExecutionProvider fused-node events (~355 graph nodes in one fused node) + 24 CPU per run, 65/65 outputs bit-identical, argmax 30751 matching CPU; coverage figures are execution, not offer. All wall-clock figures including 3.1x/3.7x are withdrawn under R13 pending device-clock measurement. Switch holds exclusive claim on device-clock measurement while agents run in parallel. — decided by Scribe

---
