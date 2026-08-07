"""Lane inventory: what each CI lane runs, what it would catch, and what it silently misses.

`operational` and `green` are different claims and this project has been careful to keep them
apart. An **operational** lane runs to completion. A **green** lane runs to completion *and
would go red if the thing it watches broke*. Only the second is worth having, and the only
evidence that distinguishes them is **two arms that differ**: the check observed passing on
a healthy input and observed failing on a broken one.

This module is that evidence, as data rather than prose. Every check a lane performs has an
entry. Every entry carries:

* what the check watches — the defect class it exists to catch;
* how it was falsified — the exact mutation applied to produce the failing arm;
* both arms' observed results, with the date they were observed;
* what it **silently misses** — the last column, and the valuable one.

The status vocabulary is deliberately four-valued, and three of the four are not "pass":

``DEMONSTRATED``
    Both arms observed. The check went red on a broken input and green on a healthy one.
    This is the only status that supports the word *green*.

``UNDEMONSTRATED``
    The check runs and passes. Nobody has ever seen it fail. It may be a constant — the ICD
    negative control was wired for weeks and matched a phrase printed on every run, so it
    short-circuited to success in both directions and *had never once executed*. An
    ``UNDEMONSTRATED`` check is not evidence; it is a candidate for evidence.

``IMPOSSIBLE_HERE``
    The falsifying arm cannot be produced in this environment, for a stated structural
    reason. This is not a lesser ``UNDEMONSTRATED``: it is a closed question with an answer.
    A lane carrying an ``IMPOSSIBLE_HERE`` entry is honestly incomplete rather than
    dishonestly finished.

``RED_NOW``
    The failing arm is the current state of the tree: the check is red on real input, on a
    real defect, right now. This is the strongest possible falsification evidence — the
    check is not merely capable of failing, it *is* failing — and it must never be read as
    a problem with the check.

THE SECOND AXIS: WHERE THE FAILING ARM CAME FROM
------------------------------------------------
Status alone conflates two things that are not the same, and the conflation was found on
2026-08-02 while classifying Switch's tautological-assertion screen. Every ``DEMONSTRATED``
check carries a second, orthogonal field:

``FALSIFIER_PLANTED``
    The failing arm exists because somebody *wrote a defect on purpose* and checked the
    screen caught it. This proves the scanner works **on the shape it was written for**.
    It proves nothing whatever about whether that shape occurs in real code — that is, it
    does not show the check is load-bearing. A planted falsifier is real evidence and it
    is the weaker kind.

``FALSIFIER_OBSERVED``
    The failing arm was produced by a defect that actually happened, or by the tree as it
    stands. The check has caught something nobody planted for it.

This axis was added because the honest answer indicts my own work as much as anyone's:
``hostfree.tick_conversion_screen`` is ``PLANTED``. So are the layering lint and the
fatal-log check. Only the portability lint and clippy are ``OBSERVED``, and they are
``OBSERVED`` because they are currently red. **Most of what this lane calls green rests on
planted falsifiers**, and a table that did not say so was letting the word `green` carry
more than it earned. Applying a stricter standard to another agent's screen than to my own
would have been the same failure in a different direction.

A ``PLANTED`` falsifier does *not* demote a check to ``UNDEMONSTRATED``: the distinction
between "somebody performed the mutation" and "nobody ever has" is real and worth keeping.
It is recorded, surfaced in the render, and left for the reader to weigh.

Nothing here is inferred from the workflow YAML. Classifying a lane from my own YAML is
what R10 forbids: the falsifier for "X is wired" is an artifact it produced whose content
varies with its input, and a YAML file produces no artifact at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Iterable

# ──────────────────────────────────────────────────────────────────────────────
# Vocabulary
# ──────────────────────────────────────────────────────────────────────────────

DEMONSTRATED = "DEMONSTRATED"
UNDEMONSTRATED = "UNDEMONSTRATED"
IMPOSSIBLE_HERE = "IMPOSSIBLE_HERE"
RED_NOW = "RED_NOW"

#: A check that has NEVER EXECUTED, because an earlier step in its own job failed and CI
#: skips the remainder. This is not ``UNDEMONSTRATED`` and the difference is the whole
#: reason it exists.
#:
#: ``UNDEMONSTRATED`` says: *this check runs, and nobody has performed the mutation that
#: would prove it can go red.* ``GATED_NEVER_RUN`` says: *this check has not run at all,
#: so we do not know it can even start.* I had the Linux lane's `device.op_correctness`
#: and `build.integration_targets` at ``UNDEMONSTRATED`` all session. That was wrong, and
#: it was wrong in the flattering direction: it read as "runs, never observed to fail",
#: which is one demonstration away from green. The truth is that the Linux job fails at
#: `Clippy (all warnings as errors)` and GitHub Actions marks every subsequent step
#: **skipped** — seven of them — and a skipped step reports nothing at all.
#:
#: A skipped step is the same shape as the defect this whole session has been about: an
#: instrument that exists, is cited in a table, and never runs. Distinguishing it in the
#: vocabulary is the only way the table stops overclaiming for a lane nobody has watched.
GATED_NEVER_RUN = "GATED_NEVER_RUN"

#: Statuses that support calling the lane containing them `green` for that check.
GREEN_STATUSES = frozenset({DEMONSTRATED, RED_NOW})

#: Statuses that are an honest recorded gap rather than an unexamined one.
RECORDED_GAP_STATUSES = frozenset({IMPOSSIBLE_HERE, GATED_NEVER_RUN})

ALL_STATUSES = frozenset(
    {DEMONSTRATED, UNDEMONSTRATED, IMPOSSIBLE_HERE, RED_NOW, GATED_NEVER_RUN}
)

#: Orthogonal to status: where the failing arm came from. See the module docstring.
FALSIFIER_PLANTED = "PLANTED"
FALSIFIER_OBSERVED = "OBSERVED"
ALL_FALSIFIERS = frozenset({FALSIFIER_PLANTED, FALSIFIER_OBSERVED})

# ──────────────────────────────────────────────────────────────────────────────
# Lanes
# ──────────────────────────────────────────────────────────────────────────────

LANE_LINUX = "build-test-linux"
LANE_WINDOWS = "build-test-windows"
LANE_HOSTFREE = "lane-checks"

#: The three lanes, and the one fact about each that governs everything else in it.
LANES = {
    LANE_LINUX: {
        "runner": "ubuntu-latest",
        "device": "lavapipe (Mesa software rasteriser, LunarG SDK package)",
        "governing_fact": (
            "The device is a CPU renderer. It has no clock, no tenancy, no discrete "
            "memory and no vendor driver. Everything it proves is about code paths, "
            "never about silicon."
        ),
    },
    LANE_WINDOWS: {
        "runner": "windows-latest",
        "device": "lavapipe (mesa-dist-win)",
        "governing_fact": (
            "Same CPU renderer as Linux, plus one thing Linux does not have: the runner "
            "process is ELEVATED, and the LunarG loader silently ignores VK_DRIVER_FILES "
            "and VK_ICD_FILENAMES in elevated processes (PLATFORMS.md 7.4.1). Every "
            "negative control built on ICD suppression is therefore unarmable here."
        ),
    },
    LANE_HOSTFREE: {
        "runner": "ubuntu-latest",
        "device": "none — no Vulkan, no GPU, no ORT",
        "governing_fact": (
            "It has no device at all, which is its strength: nothing it proves can be an "
            "accident of one machine. It is also the only lane whose failures are never "
            "ERROR(instrument)."
        ),
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# Entries
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Check:
    """One thing a lane does, and whether anyone has ever seen it fail."""

    id: str
    lane: str
    step: str
    watches: str
    status: str
    #: The mutation that produces the failing arm. Required for DEMONSTRATED/RED_NOW.
    mutation: str | None = None
    #: PLANTED or OBSERVED. Required for DEMONSTRATED/RED_NOW. A planted falsifier proves
    #: the scanner works on the shape it was written for and says nothing about whether
    #: that shape occurs in real code.
    falsifier: str | None = None
    #: What was observed on a healthy input.
    arm_healthy: str | None = None
    #: What was observed on a broken input. Quote the failure TEXT, never a count (R13).
    arm_broken: str | None = None
    observed: str | None = None
    #: Why the failing arm cannot be produced here. Required for IMPOSSIBLE_HERE.
    reason: str | None = None
    #: The valuable column. What passes this check while still being broken.
    misses: tuple[str, ...] = field(default_factory=tuple)

    def is_green(self) -> bool:
        return self.status in GREEN_STATUSES


CHECKS: tuple[Check, ...] = (
    # ── host-free lane ────────────────────────────────────────────────────────
    Check(
        id="hostfree.lane_check_tests",
        falsifier=FALSIFIER_PLANTED,
        lane=LANE_HOSTFREE,
        step="Two-polarity tests for ci/ lane checks",
        watches="The lane checks in ci/ themselves — every one has a passing and a failing case.",
        status=DEMONSTRATED,
        mutation=(
            "Each test IS the mutation: the suite asserts on both polarities of every ci/ "
            "check, so the failing arm is executed on every run rather than imagined."
        ),
        arm_healthy="suite green",
        arm_broken=(
            "removing NO_ICD_RE from check_icd_suppression.py reproduces "
            "'probe_report_unreadable' on a genuine suppression, and the suite says so"
        ),
        observed="2026-08-01",
        misses=(
            "That a check is correct does not mean it is WIRED. This suite exercises the "
            "Python; only an artifact from a real lane run shows the step ran (R10).",
            "It cannot see a defect in a check that was never written. Absence of a test "
            "for a defect class is invisible to a test suite.",
        ),
    ),
    Check(
        id="hostfree.tick_conversion_screen",
        falsifier=FALSIFIER_PLANTED,
        lane=LANE_HOSTFREE,
        step="Tick-conversion screen (static, no GPU, no vocabulary)",
        watches=(
            "Every source site that could scale a device tick into a duration, asserting "
            "each goes through GpuTimestampCalibration's converters — and that raw, "
            "unmasked ticks enter the program at exactly one place."
        ),
        status=DEMONSTRATED,
        mutation=(
            "Inject `let bypass_ns = end_ticks - begin_ticks;` into a scratch copy of "
            "rust/src. Also: launder it through a rename (`let raw_span = end_ticks;`), "
            "multiply by the period by hand while skipping the valid-bit mask, and add a "
            "second caller of read_results()."
        ),
        arm_healthy="every scaling site is a sanctioned converter or a recorded exemption",
        arm_broken=(
            "TICK-SCREEN: FAIL(condition=tick_conversion_bypassed), quoting the injected "
            "line and its file:line — verified on all four injections 2026-08-01 by "
            "ci/negative_control_tick_conversions.py"
        ),
        observed="2026-08-01",
        misses=(
            "It decides from NAMES, so it sees a tick only while the word survives. The "
            "rename rule catches the first hop out of a tick-named binding, which is where "
            "the laundering must start; it does not follow the value further.",
            "It says nothing about whether the conversion is arithmetically CORRECT. A "
            "wrong formula inside a sanctioned converter passes this screen untouched — "
            "that is trace.rs's unit tests' question, and neither arm substitutes for the "
            "other.",
            "An exemption in ci/tick_conversion_allowlist.json is a human's judgement, not "
            "a machine's. The screen guarantees the judgement is written down and pinned "
            "to a line, not that it is right.",
            "#[cfg(test)] bodies are out of frame by design. A test that encodes a wrong "
            "expectation about ticks is invisible here.",
        ),
    ),
    Check(
        id="hostfree.tick_screen_negative_control",
        falsifier=FALSIFIER_OBSERVED,
        lane=LANE_HOSTFREE,
        step="Tick-conversion screen negative control (inject the defect, demand red)",
        watches=(
            "The screen above. It injects each defect the screen claims to detect into a "
            "scratch COPY of rust/src and fails if the screen stays green."
        ),
        status=DEMONSTRATED,
        mutation=(
            "Its own baseline arm is the mutation in reverse: if the unmodified copy is "
            "not green, it reports ERROR(instrument=baseline_not_green) rather than "
            "attributing a red to an injection that did not cause it."
        ),
        arm_healthy="every injection produces the expected condition token AND the quoted line",
        arm_broken=(
            "anchor drift is reported as ERROR(instrument=anchor_not_found) — observed "
            "2026-08-01, when the first draft's anchor did not exist in session.rs and the "
            "control refused to report a pass"
        ),
        observed="2026-08-01",
        misses=(
            "It proves the screen detects the defects SOMEONE THOUGHT OF. A defect shape "
            "absent from CASES is as invisible to the control as it is to the screen.",
        ),
    ),
    Check(
        id="hostfree.census_completeness",
        falsifier=FALSIFIER_PLANTED,
        lane=LANE_HOSTFREE,
        step="Census extent and independent whole (criterion 12 evidence, no GPU)",
        watches=(
            "The three things DESIGN.md §10 row 12 asks for that the census line cannot "
            "supply about itself: how much of each mechanism's surface the observation "
            "covers; what the census's twelve is twelve OF, enumerated from production "
            "Rust the census does not write; and whether any observation's CONTENT ever "
            "moved, which is the only thing that can distinguish a name from a wrong name."
        ),
        status=DEMONSTRATED,
        mutation=(
            "Plant a counter field, a trace Phase variant and an env switch in a scratch "
            "copy of rust/src; drop a mechanism from a scratch copy of the census "
            "artifacts; rot the map; record a name as verified against arms that never "
            "varied; replay issue #58's stale absence claim and its true-absence "
            "counter-control; plant a scratch production symbol reachable only from a "
            "#[cfg(test)] mod and demand the absence_claims arm naming it stays green "
            "(issue #61). 15 arms in "
            "ci/negative_control_census_completeness.py, all fired 2026-08-07."
        ),
        arm_healthy=(
            "50 instrumented surfaces enumerated (14 counter fields, 10 trace phases, 26 "
            "env switches) against the census's 12 mechanisms: 33 censused, 12 "
            "instrumented-and-uncensused gaps with named owners, 3 out of frame, 2 not "
            "mechanisms"
        ),
        arm_broken=(
            "CENSUS-EXTENT: FAIL(condition=unmapped_surface) quoting "
            "rust/src/counters.rs:449 and the planted field text — the arm that proves "
            "the denominator is not derived from the numerator"
        ),
        observed="2026-08-02",
        misses=(
            "It does NOT close row 12. Trinity owns that tally, and supplying the "
            "artifact and closing the row must not be the same act (Morpheus, on "
            "criterion 11). A PASS here means every surface is accounted for, not that "
            "the census covers it — the twelve recorded gaps are precisely the evidence "
            "that criterion 12 is not met.",
            "Its whole is the INSTRUMENTED surface of the EP, not the EP. A mechanism "
            "that touches no counter, no trace phase and no env switch is invisible to "
            "the denominator as surely as to the census.",
            "Extent is an UPPER bound and a weak one: the numerator counts identifiers "
            "the observation happens to mention. A mechanism reading 6/6 has been shown "
            "to name six strings, nothing more.",
            "INVARIANT means no recorded arm distinguished the observation, not that no "
            "arm could. The arms are the census runs that exist; nobody designed one to "
            "vary each mechanism.",
            "It never decides a name is WRONG. Phase::Record — wired, invoked, correct, "
            "input-varying, wrong by 50x in what it was called — would read INVARIANT "
            "here, which is the flag, not the verdict.",
        ),
    ),
    Check(
        id="hostfree.device_loss_screen",
        falsifier=FALSIFIER_OBSERVED,
        lane=LANE_HOSTFREE,
        step="Device-loss screen (artifacts, no GPU)",
        watches=(
            "A run that lost the Vulkan device, fell back to the CPU EP and exited 0. "
            "The exit code is not one of its inputs, because the exit code IS the defect: "
            "a lost device that exits 0 does not look like a failure, it looks like a "
            "smaller number. Tank's ctx-512 KV points were truncated by one and "
            "differencing them produced an apparent 6.7% saving that was an observation "
            "ending early."
        ),
        status=DEMONSTRATED,
        mutation=(
            "ci/negative_control_device_loss.py, 27 arms, all fired 2026-08-07: red on "
            "Tank's real artifact (REPLAYED); red on trinity-suite-dev1.log, a second "
            "device loss from 2026-07-31 nobody had reported (LIVE — a file the screen "
            "was not written against); red on a synthesised iters=25/compute_calls=9 "
            "artifact with no log text at all; green on the same truncation once the "
            "producer has moved it under rejected_points; green on a clean log; and an "
            "instrument error, never a pass, when it is given nothing to read. Since "
            "issue #24 it also arms the exclusion list itself: red when an exclusion "
            "covers no finding, red when Tank's real ctx-4096 capture carries one more "
            "loss than its shipped record accounts for (REPLAYED), green on the same "
            "file once the record accounts for both, and an instrument error, never a "
            "pass, on a record that declares no witness at all."
        ),
        arm_healthy=(
            "746 artifacts read across bench/results, 18 excluded by name as records of "
            "known incidents with reason/owner/date and the finding(s) each accounts "
            "for, 290 decidable by the structural rule"
        ),
        arm_broken=(
            "DEVICE-LOSS: FAIL(condition=device_lost_reported) quoting "
            "'[vulkan-ep] ERROR: vkQueueSubmit failed: The logical device has been lost' "
            "from bench/results/trinity-suite-dev1.log:3216"
        ),
        observed="2026-08-07",
        misses=(
            "Its structural rule needs the producer to declare what it expected. An "
            "artifact carrying no iters/compute_calls pair is UNOBSERVABLE to it, not "
            "clean — 456 of 746 artifacts were undecidable on the run above.",
            "Three of its conditions (broken_commitment_reported, "
            "runtime_fallback_announced, marker_list_misses_real_line) are only decided "
            "on files the caller NAMES as one run's evidence. Controls on this project "
            "produce those texts deliberately, so a tree-wide scan reports them "
            "UNOBSERVABLE rather than red. Point --run-log at the lane's own log or the "
            "reach is not there.",
            "It reads artifacts after the fact. It cannot stop a run, and it cannot see "
            "a device loss whose run wrote nothing.",
            "ci/device_loss_incident_records.json is an exclusion list, and every "
            "exclusion list is a place to hide a defect. Its mitigations — reason, owner "
            "and date required; excluded files printed every run; an entry naming a "
            "missing file is itself a finding; and, since issue #24, a mandatory witness "
            "naming the finding(s) the entry accounts for, re-read through the same "
            "reader every run so an exclusion cannot cover a finding it never declared — "
            "reduce that risk and do not remove it. What remains uncovered: a witness is "
            "a count of distinct finding LINES, so a second loss whose text is identical "
            "to the forgiven one is still invisible.",
            "The witness binds a COUNT per condition, not the content behind it. "
            "Replacing an excluded artifact's body with the same number of DIFFERENT "
            "device-loss lines — a frame substitution, not a widening — passes both the "
            "screen and the witness-equality test unchanged, because nothing compares "
            "the lines themselves across runs, only how many there are per condition. "
            "Bounded by the proof/frame machinery elsewhere (this check touches no "
            "bench/results/** file), not by this screen; flagged by Morpheus's review of "
            "PR #65 as the sibling of the de-duplication limit above.",
            "It says nothing about whether the EP executed. A run can be device-loss-free "
            "and still be pure CPU output; that is the verdict's job, not this check's.",
        ),
    ),
    Check(
        id="hostfree.device_loss_negative_control",
        falsifier=FALSIFIER_PLANTED,
        lane=LANE_HOSTFREE,
        step="Device-loss screen negative control (demand red)",
        watches=(
            "The screen above. It is the only reason that screen may be called a "
            "detector rather than a step that has been observed passing."
        ),
        status=DEMONSTRATED,
        mutation=(
            "Its own provenance line: it counts LIVE, REPLAYED and PLANTED arms "
            "separately and prints that a PLANTED arm evidences nothing about whether "
            "the event occurs in reality. 1 LIVE, 6 REPLAYED, 20 PLANTED on 2026-08-07."
        ),
        arm_healthy="all 27 arms fired 2026-08-07",
        arm_broken=(
            "a missing bench/results/ctx512_device_lost.txt is reported as an arm that "
            "DID NOT FIRE — an outage in the control, never a pass"
        ),
        observed="2026-08-07",
        misses=(
            "Twenty of its twenty-seven arms are PLANTED. They prove each rule fires on "
            "an input built to make it fire; only the LIVE and REPLAYED arms evidence "
            "that the event occurs.",
            "It has never induced a real device loss. Inducing one deliberately (a TDR) "
            "would make the red arm live rather than replayed, and would also tell us "
            "whether the EP's own text prints at all when the loss is hard enough.",
        ),
    ),
    Check(
        id="hostfree.census_completeness_negative_control",
        falsifier=FALSIFIER_PLANTED,
        lane=LANE_HOSTFREE,
        step="Census extent negative control (plant a mechanism, demand red)",
        watches=(
            "The screen above. A completeness check that has only ever been observed "
            "passing reads as coverage and asserts nothing, which is the exact defect "
            "that screen exists to find."
        ),
        status=DEMONSTRATED,
        mutation=(
            "Its own baseline arm: if the unmutated tree is not green, no red from a "
            "later arm can be attributed to the injection that was supposed to cause it."
        ),
        arm_healthy="all 15 arms fired 2026-08-07, each naming the surface it planted",
        arm_broken=(
            "a mutation that cannot be performed is reported "
            "ERROR(instrument=anchor_not_found), never as a pass — the same discipline "
            "that caught the tick control's anchor drift on 2026-08-01"
        ),
        observed="2026-08-02",
        misses=(
            "It proves the screen detects the defect shapes SOMEBODY THOUGHT OF. A "
            "mechanism that arrives in a form none of the three extractors reads — "
            "neither counter, nor trace phase, nor env switch — is invisible to the "
            "control exactly as it is to the screen.",
        ),
    ),
    Check(
        id="hostfree.readme_usage",
        falsifier=FALSIFIER_OBSERVED,
        lane=LANE_HOSTFREE,
        step="README usage screen (does the documented import exist?)",
        watches=(
            "Every module imported by a fenced python block in README.md. The README told "
            "readers to `import onnxruntime_ep_vulkan` for months while no such package "
            "existed anywhere in the tree — no pyproject.toml, no setup.py, no "
            "__init__.py. Nothing could have noticed: the test suite imports what exists, "
            "so documentation drift is invisible to it by construction, and the first "
            "thing any new user does produced ModuleNotFoundError."
        ),
        status=DEMONSTRATED,
        mutation=(
            "ci/negative_control_readme_usage.py, 7 arms. The strongest is not planted: "
            "it runs the screen against the real README at 8a851f8 — the commit before "
            "the shim landed — and demands exit 1. A defect that actually shipped, not a "
            "shape somebody imagined."
        ),
        arm_healthy=(
            "README-USAGE: PASS — 2 distinct import(s) across 2 python block(s), all "
            "resolvable (onnxruntime installed/declared, onnxruntime_ep_vulkan "
            "first-party at python/src/)"
        ),
        arm_broken=(
            "README-USAGE: FAIL — 1 documented import(s) name nothing that exists: "
            "onnxruntime_ep_vulkan"
        ),
        observed="2026-08-04",
        misses=(
            "It does not EXECUTE the blocks. Executing them needs a GPU, a built artifact "
            "and a model; a check that can only run on one desk is not a check. So a "
            "README whose imports all exist but whose calls are wrong passes here.",
            "Import-resolvability is satisfied by a DECLARED dependency, not only an "
            "installed one, so that the no-GPU lane (which installs pytest/onnx/numpy and "
            "not onnxruntime) does not report a property of the lane as a finding about "
            "the README. A dependency declared but unpublishable would pass.",
            "It reads README.md only. Every other document in docs/ can drift freely.",
            "A python block that does not parse is skipped, not reported. That is "
            "deliberate — prose in a python fence is not this screen's business — but it "
            "does mean a syntactically broken example is invisible here.",
        ),
    ),
    Check(
        id="hostfree.readme_usage_negative_control",
        falsifier=FALSIFIER_PLANTED,
        lane=LANE_HOSTFREE,
        step="README usage screen negative control (replay the shipped defect)",
        watches=(
            "The screen above. It is the only reason that screen may be called a "
            "detector rather than a step that has been observed passing."
        ),
        status=DEMONSTRATED,
        mutation=(
            "Its own instrument arms: a README with no python block, and a python block "
            "importing nothing, must both report ERROR(instrument=...) and not PASS — a "
            "check that verified nothing must not read like a check that passed."
        ),
        arm_healthy="7/7 arms, 2026-08-04, 4 of them PLANTED and counted as such",
        arm_broken=(
            "the real README at 8a851f8 judged against an empty tree: exit 1, naming "
            "onnxruntime_ep_vulkan"
        ),
        observed="2026-08-04",
        misses=(
            "The historical arm needs the ref in the clone. actions/checkout defaults to "
            "depth 1, so on a shallow checkout that arm reports UNOBSERVED and the "
            "remaining four are PLANTED. The summary says so rather than reporting 6/6.",
            "Four of seven arms are PLANTED. They prove each rule fires on an input built "
            "to make it fire.",
        ),
    ),
    Check(
        id="hostfree.hardcoded_foundry_paths",
        falsifier=FALSIFIER_OBSERVED,
        lane=LANE_HOSTFREE,
        step="Hardcoded Foundry cache path screen (static, whole tree)",
        watches=(
            "Every *.py file for a literal Foundry cache directory fragment "
            "(...\\.foundry\\cache\\models...). tests/ops/test_phi35.py hardcoded this "
            "shape and went stale under us with no code change on either side when "
            "Foundry Local's own catalog revision moved (issue #11); PR #15 fixed that one "
            "site with an identity-keyed resolver, and issue #19 found ~31 more sites in "
            "tools, probes and archived benchmark scripts PR #15 had not reached. A "
            "migration with no standing guard erodes: the next probe written under time "
            "pressure pastes the pattern back in, because that is what every existing "
            "probe in the tree looked like until this pass."
        ),
        status=DEMONSTRATED,
        mutation=(
            "ci/negative_control_hardcoded_foundry_paths.py, 5 arms. The strongest is not "
            "planted: it runs the screen against the real bench/exec_census.py at ea427fd "
            "— the commit immediately before issue #19's migration — placed outside the "
            "archival allowlist exactly as it shipped, and demands exit 1."
        ),
        arm_healthy=(
            "FOUNDRY-PATHS: PASS — 33 allowlisted occurrence(s), 0 outside the allowlist "
            "(24 archival bench/results/*.py scripts + rust/tools/foundry_discovery.py's "
            "own defect-documentation (1) + the screen's own files naming the pattern in "
            "prose/fixtures — ci/check_hardcoded_foundry_paths.py (3), "
            "ci/negative_control_hardcoded_foundry_paths.py (2), ci/test_lane_checks.py "
            "(2) and ci/lane_inventory.py (1))"
        ),
        arm_broken=(
            "FOUNDRY-PATHS: FAIL — 1 hardcoded Foundry cache path(s) outside the "
            "allowlist: bench/exec_census.py:29: ..."
        ),
        observed="2026-08-05",
        misses=(
            "It matches literal source text, not identity strings or pathlib joins built "
            "from separate segments (Path.home() / '.foundry' / 'cache' / 'models' is "
            "invisible to it by design, since that shape cannot go stale the way a single "
            "hardcoded literal does). A hardcode assembled via string concatenation or "
            "f-string interpolation at a non-literal offset would also be invisible.",
            "The allowlist is a directory prefix (bench/results/) plus two named files. A "
            "genuinely new archival script placed outside bench/results/ — or a live tool "
            "placed inside it to dodge the screen — is not distinguished from the cases "
            "the allowlist is meant to cover; only the location is checked, not intent.",
            "It scans *.py only. A hardcoded path pasted into a shell script, a notebook, "
            "or a non-Python tool would not be caught.",
        ),
    ),
    Check(
        id="hostfree.hardcoded_foundry_paths_negative_control",
        falsifier=FALSIFIER_PLANTED,
        lane=LANE_HOSTFREE,
        step="Hardcoded Foundry path screen negative control (replay the shipped defect)",
        watches=(
            "The screen above. It is the only reason that screen may be called a "
            "detector rather than a step that has been observed passing."
        ),
        status=DEMONSTRATED,
        mutation=(
            "Its own instrument arm: an archival script planted under bench/results/ with "
            "the identical literal must stay green (PASS), proving the allowlist is a "
            "directory rule and not merely 'the files that happen to exist today'."
        ),
        arm_healthy="5/5 arms, 2026-08-05, 3 of them PLANTED and counted as such",
        arm_broken=(
            "the real bench/exec_census.py at ea427fd judged against an empty tree: exit "
            "1, naming rust/tools/probe_new_thing.py as the injected live-tool violation "
            "and the replayed bench/exec_census.py as the historical one"
        ),
        observed="2026-08-05",
        misses=(
            "The historical arm needs ea427fd in the clone. actions/checkout defaults to "
            "depth 1, so on a shallow checkout that arm reports UNOBSERVED and the "
            "remaining four are PLANTED. The summary says so rather than reporting 4/4.",
            "Three of five arms are PLANTED. They prove each rule fires on an input built "
            "to make it fire, not that the shape recurs unprompted.",
        ),
    ),
    Check(
        id="hostfree.tautological_assertions",
        lane=LANE_HOSTFREE,
        step="Tautological-assertion screen (no GPU, whole tree)",
        watches=(
            "Assertions whose two compared sides are the same source text, or are both "
            "literal constants — across Rust and Python, whole tree."
        ),
        status=UNDEMONSTRATED,
        mutation=(
            "Not performed on real input. Its falsifier is entirely PLANTED: "
            "ci/test_lane_checks.py runs it over deliberately-written tautologies. That "
            "shows the scanner catches the shape it was written for; it does not show the "
            "shape occurs in this tree, and 1,056 assertions scanned with 0 detections is "
            "the evidence that it currently does not."
        ),
        arm_healthy="1,056 comparison assertions scanned (rs=614, py=442), 0 detections",
        arm_broken=None,
        observed="2026-08-01 (Switch)",
        misses=(
            "Switch's own first paragraph, and the reason this is UNDEMONSTRATED rather "
            "than green: NEITHER of the two assertion defects that actually occurred here "
            "is within its reach. `test_localise_inherits_the_level_blindness_hole` "
            "compared two DIFFERENT expressions that both evaluated to 0.0 — textually the "
            "sides differ. The `fn_addr_eq` tests asserted a predicate true repeatedly and "
            "never once false — a property of a test FUNCTION, not of a line.",
            "It is scoped by its author to REGRESSION, not discovery. It found nothing; it "
            "exists so the form cannot arrive later unnoticed. A future run reporting zero "
            "is the expected state and is not evidence of health.",
            "Its own development produced three instances of the failure it hunts, "
            "including reporting PASS over a language it had not read (89 Python files "
            "yielded zero assertions and the Rust total covered for it). Fixed with a "
            "per-language coverage outage — but it is the third screen on this project to "
            "have been green while blind, so the class is not hypothetical.",
        ),
    ),
    Check(
        id="hostfree.verdict_vocabulary",
        falsifier=FALSIFIER_PLANTED,
        lane=LANE_HOSTFREE,
        step="Verdict vocabulary preflight",
        watches="Divergence between the verdict words the EP emits and the words readers accept.",
        status=DEMONSTRATED,
        mutation="Introduce a verdict string no reader knows.",
        arm_healthy="vocabulary agrees",
        arm_broken="preflight names the unknown token and the lane stops before producing a verdict it cannot read",
        observed="2026-07-31",
        misses=(
            "Agreement on WORDS is not agreement on MEANING. Two components can share the "
            "token UNATTRIBUTED and disagree about which situations produce it.",
        ),
    ),
    # ── build-level, both device lanes ────────────────────────────────────────
    Check(
        id="build.rust_unit_tests",
        falsifier=FALSIFIER_PLANTED,
        lane="both",
        step="cargo test --lib (440 unit tests)",
        watches=(
            "Every piece of pure arithmetic in the crate — and critically the "
            "timestampPeriod / timestampValidBits conversions, which are the ONLY "
            "host-independent falsifier for the 52x defect class described below."
        ),
        status=DEMONSTRATED,
        mutation=(
            "Drop both period conversions in rust/src/trace.rs: replace "
            "`Some(span as f64 * f64::from(self.timestamp_period_ns))` with "
            "`Some(span as f64)`, and the same multiply in ticks_to_axis_us."
        ),
        arm_healthy="440 passed; 2 ignored",
        arm_broken=(
            "trace::tests::treating_intel_ticks_as_nanoseconds_is_wrong_by_fifty_two_times "
            "panicked at src/trace.rs:1680: 'period scaling not applied: 100000 ns for "
            "100000 ticks' — together with timestamp_period_is_applied_and_is_not_assumed_"
            "to_be_one, undefined_upper_bits_on_a_thirty_six_bit_counter_are_masked_away, "
            "and an_intel_counter_wrap_does_not_produce_a_negative_or_absurd_duration"
        ),
        observed="2026-08-01",
        misses=(
            "It cannot see whether the conversion is CALLED at the real call sites. A "
            "correct ticks_to_ns that nobody invokes passes every one of these tests.",
            "A flake lived here: counters::tests::a_pinned_authoritative_counter_reports_"
            "unobservable_and_never_zero failed once in three full runs on 2026-08-01, "
            "writing to a fixed path under a process-global env var while other tests run "
            "in parallel. That test now holds ledger::test_lock() and the class is watched "
            "by build.contention_gate — but this step STILL cannot see an intermittent, "
            "because it runs once. Its green is a statement about one draw.",
        ),
    ),
    Check(
        id="build.rust_unit_tests_productivity",
        falsifier=FALSIFIER_PLANTED,
        lane="both",
        step="Rust unit tests asserted something (productivity floor, libtest)",
        watches=(
            "That `cargo test --lib` actually RAN its tests. libtest prints "
            "`running 0 tests` / `test result: ok.` and exits ZERO, so the step above "
            "cannot distinguish 510 passing tests from none at all."
        ),
        status=DEMONSTRATED,
        mutation=(
            "Feed ci/check_suite_productivity.py --harness libtest a log reading "
            "`running 0 tests` / `test result: ok. 0 passed; 0 failed; ...`, and a second "
            "reading `3 passed; ...; 507 filtered out` — the shape a stray filter leaves."
        ),
        arm_healthy=(
            "SUITE-PRODUCTIVITY: PASS — 510 executed, floor 500 (real Windows run, "
            "bench/results/link-suite-productivity/cargo-test-lib-windows.log)"
        ),
        arm_broken=(
            "SUITE-PRODUCTIVITY: FAIL(condition=asserted_nothing) on the zero-test log, and "
            "FAIL(condition=executed_below_floor) '3 test(s) executed ... floor is 500' on "
            "the filtered one"
        ),
        observed="2026-08-03",
        misses=(
            "It counts outcomes; it cannot see a test that ran and asserted something "
            "vacuous. That is ci/check_tautological_assertions.py's extent.",
            "GAP CLOSED 2026-08-03 (session 17). The six integration-target steps that "
            "this line used to name as a NAMED open gap — layering, portability, "
            "cdylib_load, dump_capabilities, host_registration, validation_control — are "
            "now covered by build.layering_lint_productivity, "
            "build.portability_lint_productivity and "
            "build.integration_targets_productivity. The last of those three is the "
            "interesting one: four targets share ONE step, so a per-block rule "
            "(min_target_blocks / target_ran_nothing) was needed. An aggregate floor "
            "cannot see one of four targets go silent.",
        ),
    ),
    Check(
        id="build.model_runner",
        falsifier=FALSIFIER_PLANTED,
        lane="both",
        step="Model runner (cargo test -p ort-model-runner, no device needed)",
        watches=(
            "The Rust-native real-model runner's host-free half: SHA-256 against NIST "
            "vectors, JSON round-trips, the pinned deterministic input stream, the "
            "per-model tolerance policy, the comparator's NaN asymmetry, ONNX Runtime "
            "library arbitration (absent / ambiguous / version-gated), model-pin refusal "
            "on both size and hash, and the rule that a missing counters snapshot is "
            "ABSENT rather than zero. Also that the crate compiles and lints at all: "
            "rust/modelrunner is a workspace member and NOT a default member, so the "
            "`Compile all targets` and `Clippy` steps in the same job do not reach it."
        ),
        status=DEMONSTRATED,
        mutation=(
            "In rust/modelrunner/src/evidence.rs, make `Counters::read` report "
            "`present: true` when the snapshot file does not exist — i.e. collapse "
            "\"the instrument did not report\" into \"the EP dispatched nothing\", "
            "which is the exact shape that lets a run with no EP loaded read as a run "
            "that legitimately did no work. WHEN RESTORING, UPDATE THE FILE'S MTIME "
            "(`git checkout -- <file>` then touch it, or edit in place — do NOT restore "
            "by moving a copy back, because Copy-Item/cp -p preserve the original "
            "timestamp). cargo's freshness check is mtime-based, so a restored file that "
            "is older than the planted build's artifact is treated as unchanged and the "
            "next `cargo test` re-runs the PLANTED binary. Observed 2026-08-06: a restore "
            "by Move-Item left a green tree reporting 87 passed / 1 failed."
        ),
        arm_healthy=(
            "106 passed; 0 failed (88 lib + 18 integration), clippy -D warnings clean"
        ),
        arm_broken=(
            "evidence::tests::a_missing_counters_file_is_absent_not_zero FAILED — "
            "panicked at modelrunner/src/evidence.rs:218 'assertion failed: !c.present'; "
            "test result: FAILED. 87 passed; 1 failed"
        ),
        observed="2026-08-06",
        misses=(
            "Everything that needs a device. This step never opens a Vulkan queue, never "
            "loads ONNX Runtime and never runs a model, so it cannot see the guard the "
            "runner exists for — that ORT's profile attributed executed nodes to "
            "VulkanExecutionProvider. That claim is only made by a real "
            "`--check-model-agreement` run on a machine with a device; the evidence for "
            "it lives in bench/results/rust-model-runner/ with an artifact frame, and it "
            "is a READING, not a lane.",
            "It cannot see the Linux half of the loader/path code any better than the "
            "Windows half sees the Windows one — each lane compiles and exercises only "
            "its own #[cfg] branch. The cfg-gated arms of ortlib.rs and ortapi.rs are "
            "each proven on exactly one of the two lanes.",
            "It could not, before 2026-08-06, distinguish a lane that passes from a "
            "lane that passes on one particular machine. Issue #39: the Windows arm of "
            "this step failed on every clean runner and passed on every developer box, "
            "because one integration test read an ambient model cache it had not "
            "created. The tests are now hermetic (each builds its own cache, its own "
            "stand-in library, and clears the variables discovery reads), so a green "
            "here is a green anywhere — but note that the property 'the tests are "
            "hermetic' is itself not watched by anything except review.",
            "It watched the Linux crate compile only from the Linux lane, which meant "
            "an MSVC-only type assumption reached main and went red for everyone at "
            "once (issue #39: `None => -1` on an enum that is `unsigned` under GCC and "
            "`int` under MSVC). `cargo ci --cross` now compiles this crate for "
            "x86_64-unknown-linux-gnu from a Windows host and catches that before a "
            "push, but it is a developer command, not a lane, and nothing forces it.",
        ),
    ),
    Check(
        id="build.layering_lint_productivity",
        falsifier=FALSIFIER_PLANTED,
        lane="both",
        step="Layering lint asserted something (productivity floor, libtest)",
        watches=(
            "That `cargo test --test layering` actually RAN its tests. The lint step "
            "above it exits ZERO on `running 0 tests`, so a target that compiles to "
            "nothing reports the same green as one that proved ash stays inside "
            "rust/src/vk/."
        ),
        status=DEMONSTRATED,
        mutation=(
            "#[cfg(any())] every test fn in rust/tests/layering.rs, run the real cargo "
            "command, and check the real log. Not a synthesised log — the subject "
            "genuinely absent."
        ),
        arm_healthy=(
            "SUITE-PRODUCTIVITY: PASS — 26 executed in 1 target block, floor 24 "
            "(bench/results/link-suite-productivity/cargo-test-layering-windows.log)"
        ),
        arm_broken=(
            "SUITE-PRODUCTIVITY: FAIL(condition=asserted_nothing) — and `cargo` itself "
            "exited 0 on that same run, which is the whole reason this step exists "
            "(bench/results/link-suite-productivity/ARM-LIVE-layering-empty.log)"
        ),
        observed="2026-08-03",
        misses=(
            "The floor is 24 against a measured 26, so two layering tests could be "
            "deleted without changing this step's colour. A floor is a lower bound on "
            "work, and any lower bound below the current value has slack by "
            "construction; the alternative is a floor that turns every legitimate "
            "deletion into a red step and gets raised by whoever is inconvenienced.",
            "It cannot see a layering test that runs and asserts something vacuous.",
        ),
    ),
    Check(
        id="build.portability_lint_productivity",
        falsifier=FALSIFIER_PLANTED,
        lane="both",
        step="Portability lint asserted something (productivity floor, libtest)",
        watches=(
            "That `cargo test --test portability` ran. Same failure shape as layering: "
            "libtest exits zero on an empty target."
        ),
        status=DEMONSTRATED,
        mutation=(
            "Run the real cargo command with a filter that matches 2 of the 14 tests — "
            "the shape a stray `--` argument or a renamed test module leaves behind, "
            "which is NOT `running 0 tests` and so escapes the asserted_nothing rule."
        ),
        arm_healthy=(
            "SUITE-PRODUCTIVITY: PASS — 14 executed in 1 target block, floor 12 "
            "(bench/results/link-suite-productivity/cargo-test-portability-windows.log)"
        ),
        arm_broken=(
            "SUITE-PRODUCTIVITY: FAIL(condition=executed_below_floor) — 2 executed, "
            "floor 12, cargo exit 0 "
            "(bench/results/link-suite-productivity/ARM-LIVE-portability-filtered.log)"
        ),
        observed="2026-08-03",
        misses=(
            "12 of 14 is the floor, so the filtered arm had to drop to 2 to be caught. "
            "A filter that left 12 running would pass. The rule catches collapse, not "
            "erosion.",
        ),
    ),
    Check(
        id="build.integration_targets_productivity",
        falsifier=FALSIFIER_PLANTED,
        lane="both",
        step="Integration targets asserted something (productivity floor, libtest)",
        watches=(
            "That ALL FOUR of cdylib_load, dump_capabilities, host_registration and "
            "validation_control ran — not that their sum cleared a number. One step "
            "invokes four targets, and a sum cannot see one of four go silent."
        ),
        status=DEMONSTRATED,
        mutation=(
            "#[cfg(any())] the single test in rust/tests/cdylib_load.rs — the only test "
            "that proves the shipped cdylib can be dlopen'd — and run the real command."
        ),
        arm_healthy=(
            "SUITE-PRODUCTIVITY: PASS — 11 executed across 4 target blocks "
            "(cdylib_load 1, dump_capabilities 6, host_registration 1, "
            "validation_control 3), floor 10, min_target_blocks 4 "
            "(bench/results/link-suite-productivity/cargo-test-integration-windows.log)"
        ),
        arm_broken=(
            "SUITE-PRODUCTIVITY: FAIL(condition=target_ran_nothing) naming "
            "`tests\\cdylib_load.rs`. THE AGGREGATE WOULD HAVE PASSED: 10 executed "
            "against a floor of 10. The per-target rule is the load-bearing one and "
            "this arm is the proof "
            "(bench/results/link-suite-productivity/ARM-LIVE-integration-cdylib-empty.log)"
        ),
        observed="2026-08-03",
        misses=(
            "If a whole target stops being COMPILED — removed from Cargo.toml rather "
            "than emptied — `Running tests/x.rs` never appears and the block count "
            "drops. That is what min_target_blocks=4 is for, and it is the arm not yet "
            "run live: targets_below_floor is demonstrated on a synthesised log only.",
            "It counts blocks and outcomes. host_registration still talks to "
            "tests/mock_ort, so proving it ran proves nothing about the real ORT host.",
        ),
    ),
    Check(
        id="hostfree.verification_subjects",
        falsifier=FALSIFIER_OBSERVED,
        lane=LANE_HOSTFREE,
        step="Verification-subject screen (does a check verify itself?)",
        watches=(
            "Whether any `--check` mode in this tree verifies its own output rather than "
            "an independent subject. `gen_proof_ledger.py --check` compared the proof "
            "ledger against the proof ledger — header digest against body, declared "
            "count against lines present — so it PASSed on a file that described a "
            "binary nobody had built."
        ),
        status=DEMONSTRATED,
        mutation=(
            "None needed: it was written because the defect was found. Its own first run "
            "classified `gen_proof_ledger.py --check` as SELF; that check was then "
            "repaired to ask the built artifact through OrtEpVulkanGetLedgerIdentity, "
            "and the screen now reports CHECKED 22, classified 22, FOUND 0 SELF "
            "(13 ARTIFACT, 9 EXTERNAL)."
        ),
        arm_healthy="CHECKED 22, classified 22, FOUND 0 SELF",
        arm_broken="the pre-2832526 tree, where gen_proof_ledger.py --check was SELF",
        observed="2026-08-03",
        misses=(
            "It arrived in ci/ AT 2832526 IN NO WORKFLOW STEP and was wired here a few "
            "hours later. ci/check_lane_inventory.py could not have caught that: it asks "
            "whether every workflow step has a Check, not whether every check has a "
            "step. An unwired tool is invisible to the coverage census by construction, "
            "and the register (ci/open_reds.json) is the screen that closes that gap, "
            "because the register RUNS things.",
            "0 SELF is a statement about the 22 subjects it enumerated, not about the "
            "tree. A check it did not enumerate is not a check it cleared.",
        ),
    ),
    Check(
        id="hostfree.ledger_census",
        falsifier=FALSIFIER_OBSERVED,
        lane=LANE_HOSTFREE,
        step="Proof-ledger census (has a proof ever gone missing?)",
        watches=(
            "Whether a key that was once committed to evidence/proof_ledger.jsonl has "
            "left it without a retirement record. `gen_proof_ledger.py --check` asks "
            "whether every entry AGREES with the build; it cannot ask about an entry "
            "that is no longer there, and the shrinking-write guard covers writes by "
            "the tool, which a merge is not."
        ),
        status=DEMONSTRATED,
        mutation=(
            "None needed: the arm that convicts is a real revision. "
            "`--at eb84364` — a merge in this repository — reports 3 VANISHED and names "
            "the three Cast forms 26fd93f proved. `--at 26fd93f` is green, so the screen "
            "is not a constant."
        ),
        arm_healthy="the working tree: 115 ever proven = 115 present + 0 retired + 0 VANISHED",
        arm_broken="eb84364: 106 ever proven = 103 present + 0 retired + 3 VANISHED",
        observed="2026-08-03",
        misses=(
            "It rules on the KEY SET only. An entry whose digests were rewritten in the "
            "same merge is present, so this screen says nothing about it; that is "
            "`gen_proof_ledger.py --check`'s question and the two must stay separate.",
            "It cannot see a form that was never proven. 'Which forms ought to be in the "
            "ledger' is probe_model_op_census.py's question, not this one.",
            "Its denominator is git history, so it is blind in a shallow clone. A CI "
            "checkout with `fetch-depth: 1` would report a smaller N and PASS — the same "
            "shape as the defect it detects, one level down.",
        ),
    ),
    Check(
        id="hostfree.ledger_census_negative_control",
        falsifier=FALSIFIER_OBSERVED,
        lane=LANE_HOSTFREE,
        step="Proof-ledger census negative control",
        watches=(
            "Whether ci/check_ledger_census.py still convicts what it is supposed to "
            "convict and acquits what it is supposed to acquit: 13 arms, 2 LIVE / 5 "
            "REPLAYED / 6 PLANTED, the ratio printed."
        ),
        status=DEMONSTRATED,
        mutation=(
            "Its own first run convicted the screen: without `--full-history` the replay "
            "arm reported PASS on eb84364, the merge it exists to convict, because git's "
            "default history simplification hides 26fd93f from the ledger's own log — 13 "
            "revisions simplified against 55 with the flag. One arm now asserts that gap "
            "by the numbers, so the flag cannot be removed as noise."
        ),
        arm_healthy="13/13 arms passed (2 LIVE, 5 REPLAYED, 6 PLANTED)",
        arm_broken="the same screen with --full-history dropped: 12/13, the REPLAYED conviction lost",
        observed="2026-08-03",
        misses=(
            "The PLANTED arms build synthetic repositories, so they prove the rule and "
            "not the wiring. Only the REPLAYED arms touch this repository's real history, "
            "and they are 5 of 13 — which is why the ratio is printed rather than the "
            "total.",
        ),
    ),
    Check(
        id="hostfree.landing_simulation",
        falsifier=FALSIFIER_PLANTED,
        lane=LANE_HOSTFREE,
        step="Landing simulation (would this branch survive the squash that lands it?)",
        watches=(
            "Whether this branch's ledger/register declaration screens the same colour "
            "under every landing GitHub can build from it — squash, two-parent merge, "
            "rebase, and one unrelated commit after each. The colour of "
            "`ci/check_ledger_census.py` on the PR checkout is not that question: on a "
            "`pull_request` event the checkout IS the two-parent merge ref, and a "
            "`rewitness/3` cause is green on it while the squash that lands it is red, "
            "because the origins walk resolves the moved value to a real branch commit "
            "whose tree does not carry main's concurrent edit."
        ),
        status=DEMONSTRATED,
        mutation=(
            "PLANTED, and the honest word is planted: the main-side edit to "
            "`rust/shaders/glsl/templates/q_gemv.comp` was appended by me, not observed "
            "landing. What is not planted is the mechanism — base `bb09871` plus that one "
            "line, PR `8f12b32`: the branch head is exit 0, `git merge` of the same "
            "main-side commit into the head is exit 0, and `git merge --squash` is exit 1 "
            "`FAIL(condition=uncorroborated_rewitness_cause)`. Nobody has yet landed such a "
            "collision on this repository, which is why this reads PLANTED and why the step "
            "exists before somebody does. The two polarities are held by "
            "ci/negative_control_landing_simulation.py's REPLAYED arms."
        ),
        arm_healthy=(
            "on a branch that moves nothing a corroboration reads: VERDICT: NOT-REQUIRED, "
            "with the reason printed; on one that does: three landings green, both inner "
            "controls firing"
        ),
        arm_broken=(
            "squash: FAIL(condition=uncorroborated_rewitness_cause) — cause path "
            "'rust/shaders/glsl/templates/q_gemv.comp' declares new=5f043e3aa6888dc6… but "
            "the tree the witness moved in has 3dad6cad1ac4b754…"
        ),
        observed="2026-08-07",
        misses=(
            "It is GATED. On a branch that writes no declaration and touches nothing in a "
            "live rewitness/3 corroboration's closure it decides NOT-REQUIRED and runs no "
            "landing at all. The decision is sound — every input to the landing-sensitive "
            "part of the census is then byte-identical across the merge base, the base tip "
            "and the branch — but it is a decision, and a bug in it is a step that silently "
            "stops running. That is why the negative control below is unconditional.",
            "IT IS ENFORCED, AND ONLY AT JOB GRAIN. Checked against the API on 2026-08-07, "
            "not assumed: ruleset 20479180 (`main`, active) now carries `deletion`, "
            "`non_fast_forward` and `required_status_checks` with "
            "`strict_required_status_checks_policy: true` over four contexts, of which this "
            "step's job — `Lane-check self-test (two polarities, no GPU)` — is one. "
            "`branches/main/protection` is still 404: the classic API is unused and the "
            "ruleset is the whole of the policy. So the base can no longer move after this "
            "step goes green and before the merge button: GitHub marks the pull request "
            "BEHIND and refuses the merge until the lane has re-run against the moved base. "
            "The residual that remains is the GRAIN: a required context is a JOB, not a step, "
            "so what is required is that this job went green, and a landing gate that stopped "
            "running inside a green job would satisfy the rule. That is what the "
            "unconditional negative control below, and the run's zero-skipped-steps count, "
            "are for. The other residual is scope: the rule is on `~DEFAULT_BRANCH` only.",
            "The enforcement sequence is `ci/landing_enforcement_followup.json` and it is now "
            "spent, not pending: `main` green at fbbb898 (run 31181293838, four jobs, zero "
            "skipped steps) -> ruleset 20479180 gained the strict required-check rule -> base "
            "movement demonstrably invalidated a green landing check on a disposable pull "
            "request. What is NOT done is the `main_is_green` open red, which stays open on "
            "purpose: it censuses the LAST TEN pushes and three of those are still the "
            "historical reds of issue #24, so it cannot be flipped to expect=green without "
            "lying about the window. That flip belongs to #24's lane, not to this one.",
            "It only simulates. A landing GitHub refuses to build (a real merge conflict) is "
            "reported as SIM INVALID and is a failure of this step, not a verdict about the "
            "declaration.",
        ),
    ),
    Check(
        id="hostfree.landing_simulation_negative_control",
        falsifier=FALSIFIER_OBSERVED,
        lane=LANE_HOSTFREE,
        step="Landing-simulation negative control (both polarities, the real q_gemv collision)",
        watches=(
            "Whether the gated step above still fires when it should and still acquits when "
            "it should: 10 arms, 2 STRUCTURAL / 4 PLANTED / 4 REPLAYED, the ratio printed. "
            "The step is skipped on most PRs by design, and a gate that has stopped firing "
            "looks identical from the outside to one that correctly found nothing."
        ),
        status=DEMONSTRATED,
        mutation=(
            "Its own contrast arm convicts the shape it replaced. `overlay_squash_is_blind` "
            "builds the squash the way ci/simulate_squash_rewitness.py used to — "
            "`git checkout <pr> -- .`, an overlay of the branch's paths onto the base — and "
            "requires the census to be GREEN on it, because an overlay drops a base-side "
            "edit to a file the branch also touched. Wiring that into the lane would have "
            "produced a green step on the exact defect the step exists for."
        ),
        arm_healthy="10/10 arms passed (2 STRUCTURAL, 4 PLANTED, 4 REPLAYED)",
        arm_broken=(
            "with the overlay squash restored: the REPLAYED red arm reports exit=0 and "
            "`the real squash is RED on the concurrent q_gemv cause-path edit` fails"
        ),
        observed="2026-08-07",
        misses=(
            "The REPLAYED arms pin ONE historical scenario — bb09871..ca61252 and the one "
            "rewitness/3 record that exists. A second record, or a cause in a closure with "
            "an `#include` graph the first does not have, is not covered by them; that is "
            "ci/negative_control_ledger_census.py's planted block's job.",
            "It asserts the gate's DECISION and the simulator's COLOUR. It does not assert "
            "that the workflow step's shell would propagate a non-zero exit — that is "
            "ci/check_verdict.py's question, and on this `run:` it is a plain `python` call "
            "whose status the runner consumes directly.",
        ),
    ),
    Check(
        id="hostfree.ledger_loss_probe",
        falsifier=FALSIFIER_PLANTED,
        lane=LANE_HOSTFREE,
        step="Ledger-loss probe (the census invariant's falsifier, and it writes nothing)",
        watches=(
            "Whether `gen_proof_ledger.check_no_proof_went_missing` — the invariant the "
            "census above is built on — still fires on a deletion, still exempts a SIGNED "
            "retirement, still refuses an unsigned one, and still reports "
            "ERROR(instrument) rather than 'nothing is missing' when the attempt log is "
            "absent. Seven arms. It also watches its own blast radius: the probe writes "
            "nothing into a tracked path, so running it can never dirty a checkout."
        ),
        status=DEMONSTRATED,
        mutation=(
            "Arm 3 is not a mutation at all: it is `eb84364`'s two artifacts as they "
            "really stood, and it convicts them naming all three Cast keys. The planted "
            "arms sit either side of it — arm 2 deletes one live entry and requires it "
            "named, arm 4 retires that same key with owner/date/reason and requires "
            "silence, arm 5 leaves the retirement in place with the key still present and "
            "requires a failure, arm 5b strips owner and date and requires the PRODUCER to "
            "refuse rather than the census alone. CLASSIFIED PLANTED, and the classification "
            "is the conservative one on purpose: the failing arm this entry cites is a "
            "SYNTHETIC checkout with no attempt log, which I constructed. The probe has "
            "twice been genuinely red on real input — 2/6 on `main` on 2026-08-05, when "
            "arms 1/2/4/5 were reading an empty register against a tree carrying 43 real "
            "retirements, and again the same day when running it dirtied `main` — but "
            "neither of those reds is reproducible on demand today, and an entry that "
            "claims OBSERVED on the strength of a red nobody can re-run is exactly the "
            "overclaim this axis exists to prevent."
        ),
        arm_healthy="the working tree: 7/7 arms, exit 0, `git status` byte-identical before and after",
        arm_broken=(
            "a checkout carrying a ledger and a register but no proof_attempts.jsonl: "
            "1/7, exit 1, and the ONE arm that stays green is arm 6, which predicted the "
            "outage (ci/test_lane_checks.py::test_ledger_loss_probe_fails_loud_when_its_"
            "subject_is_not_there)"
        ),
        observed="2026-08-06",
        misses=(
            "Its denominator is the ledger as it stands, so it says nothing about a form "
            "that was never proven — that is probe_model_op_census.py's question.",
            "Arm 3 needs `git show eb84364:...`. In a shallow clone that arm cannot reach "
            "its subject; the probe reports it as a failure rather than dropping it, so "
            "the lane goes red for a checkout reason. `fetch-depth: 0` on this job is "
            "therefore load-bearing for this step, exactly as it is for the census "
            "controls above.",
            "It rules on the KEY SET only, like the census. An entry whose digests moved "
            "is present and this probe is silent about it.",
            "The destination policy consults `git check-ignore` and nothing else. A "
            "destination outside any repository is trusted without further question, "
            "which is the right boundary here but is a boundary.",
        ),
    ),
    Check(
        id="hostfree.open_reds",
        falsifier=FALSIFIER_OBSERVED,
        lane=LANE_HOSTFREE,
        step="Open reds (every guard is the colour the register declares)",
        watches=(
            "The COLOUR of every declared guard against ci/open_reds.json, in both "
            "directions. A check the register expects green that goes red is an "
            "unaccounted red and fails the lane. A check the register accepts as red "
            "that goes GREEN is a stale acceptance and also fails the lane — that is "
            "the arm that stops the register rotting, because an allowlist which only "
            "ever suppresses grows monotonically and nobody is ever asked to prune it. "
            "An accepted red whose output no longer contains its declared `signature` "
            "is a DIFFERENT red and fails: the acceptance does not stretch. And every "
            "entry carries a `review_by` after which it is a lease_expired failure — "
            "acceptance here is a lease, not a grant."
        ),
        status=DEMONSTRATED,
        mutation=(
            "None needed. It was written against three checks that were red in main's "
            "own checkout on 2026-08-03 (audit_instruments --check, ci/test_lane_checks."
            "py, tests/union_check.py --run) and it convicted the shipped tree on its "
            "first run: the register expected ci/test_lane_checks.py green and it was "
            "red, which is the arm working on real bytes."
        ),
        arm_healthy=(
            "OPEN-REDS: PASS — 8 check(s) are the colour the register declares "
            "(5 accepted red, each named with an owner and a closing condition)"
        ),
        arm_broken=(
            "OPEN-REDS: FAIL — FAIL(condition=unaccounted_red) lane_checks_suite, on "
            "the real tree, before the census-extent trio was split out into its own "
            "narrowed entry. Also REPLAYED against 133b9fe's real ci.yml bytes (the "
            "merge that reintroduced four dormant BUILD_SKIPPED guards): "
            "FAIL(condition=unaccounted_red)."
        ),
        observed="2026-08-03",
        misses=(
            "It rules on COLOUR, not on coverage. A guard that is in no lane and in no "
            "register is exactly as invisible to it as it was before; that is "
            "ci/lane_inventory.py's question and the two are deliberately kept as two "
            "tools over one tree, which is the failure rust/tools/audit_instruments.py "
            "names as 'two censuses over one tree'.",
            "It cannot observe a colour that depends on a device. "
            "tests/ops/test_matmulnbits.py::test_layer_capture_mechanism is red on a "
            "lane with a GPU and green-by-skip host-free, so it is named in the "
            "register's `not_declared_here` block rather than declared — an entry that "
            "a skip can satisfy is an acceptance granted by an absence.",
            "`review_by` makes this a deliberate time bomb. That is the design, not an "
            "oversight: a red nobody has re-read in three months is not accepted, it is "
            "forgotten, and the inconvenience is the only thing that makes anyone "
            "re-read the entry.",
            "It shells out rather than importing, so it is as slow as the checks it "
            "runs. Importing would screen a different thing from the one that is "
            "failing — the argv, __main__ and exit-code paths would all go unexercised, "
            "and the exit code is the entire subject.",
        ),
    ),
    Check(
        id="hostfree.open_reds_negative_control",
        falsifier=FALSIFIER_PLANTED,
        lane=LANE_HOSTFREE,
        step="Open-reds negative control (demand each rule red on purpose)",
        watches=(
            "That every rule of the open-reds screen still fires, including the two "
            "that only ever fire on good news (stale_acceptance) or on the calendar "
            "(lease_expired) and would otherwise go years without being observed."
        ),
        status=DEMONSTRATED,
        mutation=(
            "39 arms: 2 LIVE (the shipped register must be the colour it declares, and "
            "must name an owner for every accepted red), 3 REPLAYED (the real ci.yml at "
            "133b9fe must be an unaccounted red, the same rule over today's bytes must "
            "be green — so it is not a constant), 34 PLANTED. The lease arm is driven "
            "in BOTH polarities off one knob (OPEN_REDS_TODAY the day before and the "
            "day after review_by), because 'any date makes it red' would pass a "
            "one-sided test."
        ),
        arm_healthy="39/39 arms fire as specified, exit 0",
        arm_broken=(
            "Each condition arm is red by construction with its defect genuinely "
            "present: a green-expected check that exits 1, a red-expected check that "
            "exits 0, a red-expected check that fails for a different reason, an "
            "expired lease, seven partial entries, a blanket acceptance with no "
            "signature, a missing register, a missing command, a check that never "
            "finished."
        ),
        observed="2026-08-03",
        misses=(
            "34 of 39 arms are PLANTED and the control prints that ratio itself. A "
            "planted arm proves the rule fires on the shape it was written for; it does "
            "not show the rule is load-bearing. The LIVE and REPLAYED arms are the ones "
            "that do.",
            "The LIVE arm runs the whole register, so this control is as expensive as "
            "the screen. It is in the host-free lane for that reason.",
        ),
    ),
    Check(
        id="hostfree.gh_auth_screen",
        falsifier=FALSIFIER_OBSERVED,
        lane=LANE_HOSTFREE,
        step="GH-invocation auth screen (every API-using `gh` call has a token path)",
        watches=(
            "Every workflow step that reaches the GitHub API through `gh` — directly, "
            "by running a `gh <subcommand>` that talks to the network (`run`, `api`, "
            "`pr`, `issue`, `release`, `workflow`, `repo`, `gist`, `org`, `project`, "
            "`search`, `secret`, `variable`, `cache`, `auth status`; NOT `--version`, "
            "`help`, `completion`, `config`, `extension`), or indirectly, by invoking a "
            "`ci/*.py` script that itself shells out to `gh`, or by being one of the two "
            "scripts independently verified to run `ci/open_reds.json`'s real, default "
            "register end to end while that register still has a live entry pointing at "
            "a `gh`-shelling script — must have `GH_TOKEN` or `GITHUB_TOKEN` declared in "
            "an `env:` block it can see (its own, its job's, or the workflow's), in "
            "either supported YAML shape: block (`env:` then indented keys, quoted or "
            "unquoted) or inline/flow (`env: {KEY: value}`, on one physical line or "
            "split across several), classified by a YAML-STRUCTURAL parser — an "
            "indentation-based stack of frames giving exact ancestor-path visibility — "
            "rather than by line shape or indent thresholds (issue #21; frame-stack "
            "rewrite issue #25). A duplicate key inside one `env:` mapping is an "
            "unsupported/ambiguous construct and is ERROR, never a silent last-wins "
            "guess, and so is a duplicate SIBLING `env:` key at the same scope (two "
            "`env:` blocks under one job) — the parser also rejects a `steps:` frame "
            "unless its ancestor path is EXACTLY `jobs.<job>.steps` (not any frame "
            "merely named `steps`, e.g. a `with: steps:` action input), an anchor/tag/"
            "alias value followed by more-indented content, a multi-document `---` "
            "file, and a nested flow collection or quoted-brace value inside `env: "
            "{...}` (PR #27 review of issue #25)."
        ),
        status=DEMONSTRATED,
        mutation=(
            "It was written against, and first convicted, the real defect: PR #13 run "
            "31052604259's `Open-reds negative control` step called `gh run list` (via "
            "ci/negative_control_open_reds.py -> ci/check_open_reds.py -> the real "
            "register's main_is_green entry -> ci/check_main_is_green.py) with no "
            "GH_TOKEN anywhere in scope. REPLAYED against the real ci.yml at b1886d99, "
            "that exact step is convicted; the same rule over today's fixed bytes is "
            "green, so it is not a constant. Issue #21 review of PR #17 found two "
            "further blind spots, both closed here: the inline `env: {GH_TOKEN: ...}` "
            "form (the exact remediation text) was being FALSELY CONVICTED because the "
            "block-only line-shape check required nothing to follow `env:` on the same "
            "line; and a zero-`gh`-reaching-subject frame (this screen pointed at the "
            "wrong scope, a subdirectory, or a stale path) printed a silent `PASS — 0 "
            "gh-reaching step(s)`, indistinguishable on the page from never having run "
            "at all. Both are now covered: inline `env:` is parsed the same as block "
            "`env:`, and 0 subjects is ERROR(instrument=zero_gh_reaching_subjects) by "
            "default, PASS only with the documented `--allow-empty-frame` opt-in. The "
            "screen is also now wired against the whole `.github/workflows` directory "
            "(expanded recursively), not two files named on a command line, so a new "
            "workflow — or one moved into a subdirectory — is screened without anyone "
            "having to remember to add its name. An adversarial review of PR #22 (issue "
            "#25) then found that the two-indent-threshold approach behind issue #21's "
            "fix could still be fooled: text resembling `env:`/`GH_TOKEN:` inside a "
            "`run: |` block scalar (a step's own shell script) was structurally "
            "indistinguishable from a real declaration by indent alone, and a real "
            "`env` mapping nested under `services.<id>` or `with:` (one level deeper "
            "than job/step) would have been misattributed to job scope by the same "
            "two-threshold logic. `parse_workflow()` was rewritten around a generic "
            "indentation-based stack of frames (exact ancestor-path visibility at any "
            "depth) rather than two fixed thresholds: a `run: |` (or `>`) block-scalar "
            "frame's body is never re-interpreted as YAML structure; `env:` is only "
            "attributed to workflow/job/step scope when the frame stack matches one of "
            "those three exact ancestor paths, so `services.<id>.env`, `with: env:`, "
            "and `container: env:` are all excluded generically rather than by special "
            "case. Quoted block keys (`\"GH_TOKEN\":`), multi-line flow mappings split "
            "across several physical lines, and trailing `# comment`s after an inline "
            "`env: {...}` (which the old anchored-regex approach could not match) are "
            "now all correctly recognised, and a duplicate key within one `env:` "
            "mapping (block or flow form) raises ERROR(instrument="
            "unsupported_yaml_construct) rather than silently picking a winner. PR #27's "
            "review of that rewrite then found five more blind spots IN the frame-stack "
            "parser itself, each fixed the same way — raise rather than guess: (R1) a "
            "`with: steps:` action input (a legal frame whose leaf key merely happens "
            "to be \"steps\") reset step-minting the same as a real `jobs.<job>.steps` "
            "list, orphaning the real step and silently dropping whatever ran after the "
            "trap from body capture — a false PASS by omission, closed by tagging a "
            "`steps:` frame `is_real_steps` only when its exact ancestor path is "
            "`jobs.<job>.steps`; (R2) a scalar value this parser does not structurally "
            "understand (`&anchor`, `!!tag`, `*alias`) pushed no frame at all, so more-"
            "indented content that followed it (e.g. a nested `env:`) was silently "
            "attributed to the grandparent frame instead — closed by raising when the "
            "next content line is more indented than such a value; (R3a) a multi-"
            "document `---` file was read straight through as if it were one document; "
            "(R3b) a nested flow collection as an `env: {...}` value (`{FOO: {GH_TOKEN: "
            "x}}`) and a quoted value containing a literal brace (`{FOO: '{\"GH_TOKEN\": "
            "\"1\"}'}`) both risked a misread rather than a refusal — `_flow_mapping_"
            "key_list` now tracks quote state and nesting depth char-by-char and raises "
            "on either shape, while still recognising `${{ github.token }}`-style GH "
            "Actions expressions as internally-balanced opaque text, not nesting; (N1) "
            "two sibling `env:` keys at the same scope (two `env:` blocks under one "
            "job) were silently unioned via `set.update()`; closed with `claim_env_slot"
            "()`, which raises on a second `env:` at a scope already claimed; (N2) a "
            "dash-inline `- env:` block's redispatch used a synthetic `dash_indent + 1` "
            "column that undershot `env:`'s true column, letting a later true sibling "
            "field at that real column be swallowed as one of `env:`'s own children — "
            "closed by computing the real column from the actual whitespace after the "
            "dash. Morpheus's re-review of PR #27 then found a sixth blind spot, this "
            "time in the frame-POP logic rather than a value it could not read: the "
            "generic `indent <= frame.indent` pop that correctly discards a mapping "
            "key's frame on a same-or-shallower-indent sibling line was applied "
            "unconditionally to sequence-item (`- ...`) lines too, even though YAML "
            "permits a list's items to sit at the EXACT SAME column as the key that "
            "introduces them (the \"compact\"/zero-indent block-sequence form, e.g. "
            "`steps:\\n- name: x`, valid for any list-valued key, not only `steps:`) — "
            "so that form's `steps:` frame was popped before its own first dash was "
            "ever recognised as ITS child, silently dropping the step (R4). Closed by "
            "giving sequence-item lines their own pop rule: pop frames strictly "
            "DEEPER than the dash first, then pop at most one sibling sequence-item "
            "frame already sitting at this exact indent, but never the owning key's "
            "frame itself, whether that key was opened at the item's exact indent "
            "(compact form) or shallower (the ordinary indented form). A related "
            "buf/flush ordering defect surfaced in the same fix: appending a new "
            "step's own dash line to the PREVIOUS step's body buffer before flushing "
            "it (rather than after) corrupted both steps' captured text whenever the "
            "new step's list sat deeper than the old step's — now flush happens before "
            "anything is appended to the new step's buffer."
        ),
        arm_healthy=(
            "GH-AUTH: PASS — N `gh`-reaching step(s) across every workflow file under "
            ".github/workflows, every one with GH_TOKEN or GITHUB_TOKEN declared in "
            "scope. The file/subject counts are derived at check-time from the real "
            "directory listing and the real YAML, not hardcoded in this table — as of "
            "2026-08-05 that reads `PASS — 2 gh-reaching step(s) across 6 workflow "
            "file(s)`, but the exact figures are expected to drift as workflows are "
            "added; read the check's own stdout for the current count, not this line. "
            "3 ci/*.py script(s) classified as reaching `gh` (directly or through "
            "ci/open_reds.json)."
        ),
        arm_broken=(
            "GH-AUTH: FAIL(condition=missing_token_path) — REPLAYED against the real "
            "ci.yml at b1886d99, naming the `Open-reds negative control` step exactly; "
            "or GH-AUTH: ERROR(instrument=zero_gh_reaching_subjects) when the screen is "
            "pointed at a scope (e.g. conformance.yml alone) with no real `gh`-reaching "
            "step in it."
        ),
        observed="2026-08-05",
        misses=(
            "It reads key NAMES out of `env:` blocks and never a value, so it cannot "
            "tell a real token from an empty string assigned to the right name — that "
            "is a provisioning defect (the secret was never set), not an authoring "
            "defect (the step forgot to ask for one), and the two want different fixes.",
            "It does not check that the token has enough SCOPE for the call (`actions: "
            "read` vs `contents: write` etc.); that is a human judgement made once per "
            "step and reviewed in the diff, not a thing derivable from the YAML text.",
            "It does not follow `uses:` steps (third-party actions), which authenticate "
            "however their own inputs say to — a different surface from `run:` shelling "
            "out to `gh` directly.",
            "The indirect-detection table (KNOWN_REAL_REGISTER_RUNNERS) is a short, "
            "named, hand-verified allowlist rather than a generic call graph, because a "
            "generic 'file A's text mentions file B's name' graph cannot distinguish "
            "ci/negative_control_open_reds.py's real run of the register from "
            "ci/test_lane_checks.py's synthetic one without producing exactly that "
            "false positive.",
            "The inline/flow-mapping key scan (`_flow_mapping_key_list`) is a quote-"
            "and-depth-aware char-by-char scan over flow syntax, not a real YAML flow-"
            "mapping parser. As of the PR #27 review fix, it no longer merely 'could in "
            "principle' be confused by a nested flow collection as a value (`{FOO: "
            "{GH_TOKEN: x}}`) or a quoted value containing a literal brace (`{FOO: "
            "'{\"GH_TOKEN\": \"1\"}'}`): both now raise ERROR(instrument="
            "unsupported_yaml_construct) rather than being silently read either "
            "correctly or incorrectly — the prior prose describing these as merely "
            "theoretical risks was itself inaccurate once the char-by-char rewrite "
            "landed, since the rewrite could tell them apart correctly but the review "
            "asked for a loud refusal over a silent (even if correct) resolution.",
            "Issue #25 fixed three named blind spots (`run: |` block-scalar text, "
            "`services.<id>.env`, `with: env:`) plus quoted block keys, duplicate-key "
            "detection, and multi-line flow mappings. The same exact-ancestor-path "
            "mechanism also now excludes `container: env:` as a side effect, though "
            "that shape is not separately named in either issue and has no dedicated "
            "negative-control arm of its own.",
            "The parser is a constrained, hand-rolled indentation walker, not a "
            "general-purpose YAML implementation (deliberately — this repository's "
            "lane-checks CI job installs only `pytest onnx numpy`, no PyYAML, and "
            "PyYAML's own default duplicate-key behaviour is a silent last-wins rather "
            "than an error, which would have reopened exactly the ambiguity this fix "
            "closes). YAML anchors/aliases (`&`/`*`), tag directives (`!!tag`), and "
            "multi-document files (`---`) now reliably raise ERROR(instrument="
            "unsupported_yaml_construct) (PR #27 review of issue #25) rather than being "
            "silently misread or misattributed — this is a loud, deliberate refusal, "
            "not merely 'unsupported and either misread or raised' as earlier prose "
            "here claimed before that fix landed.",
        ),
    ),
    Check(
        id="hostfree.gh_auth_negative_control",
        falsifier=FALSIFIER_PLANTED,
        lane=LANE_HOSTFREE,
        step="GH-invocation auth negative control (demand red without a token, never a skip)",
        watches=(
            "That the gh-auth screen still fires on a missing token path, AND — the arm "
            "issue #16 specifically asked for — that ci/check_main_is_green.py itself, "
            "run with every GitHub credential stripped from its environment, reports "
            "ERROR(instrument=github_unreachable) and exits 4: never exit 0, never the "
            "word PASS, never a skip. An absent instrument must read as absent, not as a "
            "quiet green. Issue #21 added: inline vs. block `env:` both satisfy the "
            "check; a wrongly-named env key does not; a zero-gh-reaching-subject frame "
            "is ERROR by default and PASS only with --allow-empty-frame; an empty or "
            "mis-scoped directory is ERROR; and directory expansion recurses into "
            "subdirectories so a nested workflow cannot hide from a broader invocation. "
            "Issue #25 added: text resembling `env:`/`GH_TOKEN:` inside a `run: |` "
            "block scalar, and a real `env` mapping nested under `services.<id>` or "
            "`with:`, must NOT satisfy the check; a quoted block key and a multi-line "
            "flow mapping MUST satisfy it; a duplicate key within one `env:` mapping "
            "(either YAML shape) is ERROR(instrument=unsupported_yaml_construct); and "
            "ci.yml's real production invocation is pinned to the `.github/workflows` "
            "directory form, not two files named on a command line. PR #27's review of "
            "the frame-stack parser added: a `with: steps:` action input does not "
            "orphan the real step's own trailing command (R1); an anchor/tag/alias "
            "value followed by more-indented content is ERROR, not attributed to the "
            "grandparent frame (R2); a multi-document `---` file is ERROR (R3a); a "
            "nested flow collection or quoted-brace value inside `env: {...}` is "
            "ERROR, while a `${{ github.token }}` GH Actions expression in the same "
            "position is NOT mistaken for either (R3b); a duplicate SIBLING `env:` key "
            "at the same scope is ERROR (N1); and a dash-inline `- env:` block does not "
            "swallow a true sibling field at its real column (N2). Morpheus's "
            "re-review of PR #27 added: a sequence item whose dash sits at the EXACT "
            "SAME column as its owning key (the equally-valid compact/zero-indent "
            "block-sequence form YAML permits for any list-valued key, not only "
            "`steps:`) still mints its step rather than being popped off the frame "
            "stack before it is ever recognised as that key's child (R4) — in both a "
            "single compact-form job, and a compact-form job followed by a "
            "differently-indented job in the same file (the exact false-PASS-by-"
            "omission Morpheus's reproducer demonstrated)."
        ),
        status=DEMONSTRATED,
        mutation=(
            "44 arms: 8 LIVE (today's workflows and the whole workflows directory pass "
            "the screen; screening only conformance.yml — genuinely 0 gh-reaching steps "
            "— is ERROR(instrument=zero_gh_reaching_subjects), never a silent PASS; "
            "check_main_is_green.py with GH_TOKEN/GITHUB_TOKEN/GH_ENTERPRISE_TOKEN "
            "removed and GH_CONFIG_DIR pointed at an empty throwaway directory must "
            "exit 4 with ERROR(instrument=github_unreachable) and never PASS or SKIP; "
            "ci.yml's own text is read and its real check_gh_auth.py invocation is "
            "asserted to be the single `.github/workflows` directory form), "
            "2 REPLAYED (the real ci.yml at b1886d99 convicts on `Open-reds negative "
            "control`; the same rule over today's bytes is green), 34 PLANTED (no-token "
            "API call; non-API `gh --version` with no token of its own is NOT counted; "
            "a token in a different job does not satisfy this one; a job-level token "
            "satisfies every step in that job; block-form and inline-form `env:` at "
            "step level both satisfy the check; a wrongly-named env key does not; a "
            "workflow with steps but zero gh-reaching subjects is ERROR by default and "
            "PASS only with --allow-empty-frame; an empty/mis-scoped directory is "
            "ERROR; directory expansion recurses into subdirectories; indirect reach "
            "through the real open-reds register is still caught; a missing workflow "
            "file is ERROR(instrument), not a red; an empty workflow is "
            "ERROR(instrument=no_steps_parsed); a comment that merely quotes `gh run "
            "list` as prose is not mistaken for the command; env-like text inside a "
            "`run: |` block scalar does not satisfy the check; `services.<id>.env` and "
            "`with: env:` do not satisfy job/step scope; a quoted block key satisfies "
            "the check exactly like an unquoted one; a duplicate key in block-form or "
            "flow-form `env:` is ERROR, never a silent guess; a trailing comment after "
            "an inline `env: {...}` does not hide it; a flow mapping split across "
            "several physical lines is still read as one declaration; a `with: steps:` "
            "list does not orphan the real step (R1); `with: &a` and `with: !!map` "
            "followed by nested content are ERROR (R2, two arms); a multi-document "
            "`---` file is ERROR (R3a); a nested flow-collection value and a quoted-"
            "brace value inside `env: {...}` are each ERROR (R3b, two arms); a "
            "`${{ ... }}` GH Actions expression inside `env: {...}` is NOT mistaken "
            "for the R3b shape (regression guard); a duplicate sibling `env:` key at "
            "job scope is ERROR (N1); a dash-inline `- env:` block does not swallow a "
            "true sibling field (N2); a single-job compact-form (equal-indent) "
            "`steps:` list mints its step and convicts an untokened `gh api` call "
            "inside it rather than silently dropping it (R4a); and a compact-form job "
            "followed by an indented-form job in the same file does not silently omit "
            "the compact job's untokened step (R4b, Morpheus's exact reproducer))."
        ),
        arm_healthy="44/44 arms fire as specified, exit 0",
        arm_broken=(
            "Each arm is red by construction with its defect genuinely present: the "
            "screen convicts the real pre-fix ci.yml bytes; check_main_is_green.py with "
            "no credentials exits 4 and names github_unreachable rather than exiting 0 "
            "or printing PASS/SKIP; each planted shape trips the rule it targets."
        ),
        observed="2026-08-05",
        misses=(
            "34 of 44 arms are PLANTED and this control prints that ratio itself. A "
            "planted arm proves the rule fires on the shape it was written for; it does "
            "not show the rule is load-bearing. The LIVE and REPLAYED arms are the ones "
            "that do.",
            "The 'no credential leak' property rests on never having a real token to "
            "leak in the first place (the arm deletes GH_TOKEN/GITHUB_TOKEN/"
            "GH_ENTERPRISE_TOKEN before invoking check_main_is_green.py) rather than on "
            "scanning a real secret out of captured output — the stronger version of "
            "this arm would need a real token deliberately withheld from logging, which "
            "this host-free lane cannot provision.",
        ),
    ),
    Check(
        id="hostfree.build_precondition",
        falsifier=FALSIFIER_OBSERVED,
        lane=LANE_HOSTFREE,
        step="Build-precondition screen (a lane that could not build must not report)",
        watches=(
            "Workflow steps that convert a failed or skipped BUILD into a green lane. "
            "BP1: a script that writes a gated env name AND exits 0 in the same body "
            "(`BUILD_SKIPPED=1; exit 0` — one missing tracked Cargo.toml turned thirty "
            "steps green on both device lanes). BP2: a dormant `if: env.X != '1'` guard "
            "whose writer no longer exists — inert today, re-armed by one added line, "
            "and indistinguishable in review from a live one. BP3: a build step that "
            "publishes an artifact path without asserting the artifact exists."
        ),
        status=DEMONSTRATED,
        mutation=(
            "None needed for the primary arm. BP2 FIRED ON THE REAL TREE on its first "
            "run: 38 dormant guards left behind by my own 2026-08-02 decision to keep "
            "them 'so the change reads as one deletion'. BP1 is falsified by REPLAY — "
            "`git show 607056a:.github/workflows/ci.yml`, the real bytes that carried "
            "the defect on main."
        ),
        arm_healthy=(
            "BUILD-PRECONDITION: PASS — 99 steps across 2 workflow files, 0 findings "
            "(after the 38 guards were deleted and conformance.yml's Build step was "
            "taught to assert its .so)"
        ),
        arm_broken=(
            "BUILD-PRECONDITION: FAIL(condition=dead_guard) ×38 on my own tree, and "
            "FAIL(condition=skip_flag_with_exit_zero) on the replayed 607056a bytes, and "
            "FAIL(condition=build_step_does_not_verify_its_artifact) on conformance.yml"
        ),
        observed="2026-08-03",
        misses=(
            "It is a STATIC screen over YAML text with no YAML parser, by choice: the "
            "lane-checks job installs only pytest/onnx/numpy, and a screen skipped "
            "because an import failed is a screen that does not exist. The cost is that "
            "if the block structure it keys on ever stops matching it reports "
            "ERROR(instrument=no_steps_parsed) rather than PASS — UNOBSERVABLE is not "
            "zero — but it cannot reason about `uses:` actions or composite steps at "
            "all.",
            "BP1 is a conjunction (writes a gated name AND exits 0 in one script). A "
            "build that writes the flag in step A and exits 0 in step B is invisible to "
            "it. Two PLANTED arms assert each half alone stays clean, because a rule "
            "that reddens ordinary provisioning steps trains people to ignore it.",
            "BP3 knows one artifact-shaped thing: a step named `Build ` that exports a "
            "path. It does not know what every lane's build is supposed to produce.",
        ),
    ),
    Check(
        id="hostfree.build_precondition_negative_control",
        falsifier=FALSIFIER_PLANTED,
        lane=LANE_HOSTFREE,
        step="Build-precondition negative control (demand red on each rule)",
        watches=(
            "That the build-precondition screen still goes RED on each of its three "
            "rules. A screen that has stopped firing is indistinguishable from a clean "
            "tree, and this repo now has a clean tree, so the screen's own silence is "
            "no longer evidence of anything."
        ),
        status=DEMONSTRATED,
        mutation=(
            "16 arms: 1 LIVE (the current tree must be clean), 2 REPLAYED (real ci.yml "
            "bytes from 607056a, asserting BP1 and NOT BP2 catches them), 13 PLANTED "
            "(each rule, plus each half of BP1's conjunction alone, which must stay "
            "clean)."
        ),
        arm_healthy="16/16 arms fire as specified, exit 0",
        arm_broken=(
            "Caught a real bug in the screen on its first run: screen() returned exit 1 "
            "without ever printing the R13 FAIL(condition=...) token — _fail() existed "
            "and was never called. Four arms went red. A red step with no condition name "
            "is the exact defect the R13 vocabulary exists to prevent, and the screen "
            "had it."
        ),
        observed="2026-08-03",
        misses=(
            "REPLAYED arms pin a commit hash. If 607056a is ever unreachable (a squashed "
            "history, a fresh shallow clone) those two arms cannot run. They "
            "ERROR(instrument=...) rather than passing quietly, but that still leaves "
            "BP1's only non-planted evidence unavailable.",
        ),
    ),
    Check(
        id="hostfree.powershell_exit_status",
        falsifier=FALSIFIER_OBSERVED,
        lane=LANE_HOSTFREE,
        step="PowerShell exit-status screen (a verdict step must consume its own code)",
        watches=(
            "Issue #49 (PS1): a Windows `run:` step that captures `$LASTEXITCODE` into "
            "a named variable to brand or print its OWN verdict, but whose success path "
            "ends without an explicit `exit`. GitHub's generated `pwsh` wrapper appends "
            "`if ((Test-Path variable:\\LASTEXITCODE)) { exit $LASTEXITCODE }` after the "
            "script body, so the step's real exit code stays pinned to whatever native "
            "command ran LAST before the fall-through -- on a step built around an "
            "intentionally-failing command (a negative control), that is never the "
            "step's own verdict. Both Windows 'Gate negative control' steps carried this "
            "shape; the second is what issue #49 was filed against, the first is a "
            "second, proven-coupled instance that had simply never been exercised "
            "because ICD suppression never used to take on an elevated Windows runner. "
            "Issue #55 (PS2): the capture was never what made the verdict lie -- a "
            "step that invokes a native command (`&`, a bare tool name, or a bare "
            "`*.exe`) ANYWHERE in its body and ends on a `Write-Host` verdict print with "
            "no intervening `exit` is relying on the SAME implicit wrapper, just without "
            "ever having named a variable for it. 'Install Mesa lavapipe' carried this "
            "exact shape -- `vulkaninfoSDK.exe`'s own exit status, unread, behind a "
            "final `Write-Host \"lavapipe smoke-check: OK\"` with no `exit` -- and PS1's "
            "`$name = $LASTEXITCODE` regex never reached it."
        ),
        status=DEMONSTRATED,
        mutation=(
            "REPLAYED (PS1): the real ci.yml bytes at the commit issue #49 was filed "
            "against, which carry the defect on BOTH Windows negative-control steps. "
            "REPLAYED (PS2): the real ci.yml bytes at d2bf65f, the commit issue #55 was "
            "filed against (PR #50 merged, PS1's two steps already fixed), where "
            "'Install Mesa lavapipe' still carries the no-capture sibling shape -- and a "
            "companion arm confirming PS1 alone would have left that tree PASS. "
            "PLANTED: a renamed step/variable exercising PS1's shape and PS2's shape "
            "each under a different tool/step name, plus arms proving both rules stay "
            "clean on every adjacent, correct idiom (`exit $LASTEXITCODE` directly, an "
            "explicit `exit 0`, a two-command accumulate-then-exit pattern, a bare "
            "`*.exe` invocation, a known bare tool name, a step with no native call at "
            "all) and a dedicated regression guard proving a `.exe` substring inside a "
            "quoted URL/path VALUE (never invoked) does not false-positive PS2."
        ),
        arm_healthy=(
            "POWERSHELL-EXIT-STATUS: PASS -- 2 workflow files, 121 named steps scanned, "
            "0 findings (after both Windows Gate steps and 'Install Mesa lavapipe' all "
            "gained an explicit `exit 0`)"
        ),
        arm_broken=(
            "POWERSHELL-EXIT-STATUS: FAIL(condition=stale_exit_code_after_native_capture) "
            "naming both `.github/workflows/ci.yml` Gate-negative-control steps by line "
            "and their last (non-exiting) script line, OR "
            "FAIL(condition=native_exit_stale_at_verdict_print) naming 'Install Mesa "
            "lavapipe' by line and its last (non-exiting) Write-Host line"
        ),
        observed="2026-08-07",
        misses=(
            "It is a STATIC screen over YAML text with no YAML parser, for the same "
            "reason check_build_precondition.py is: the lane-checks job installs only "
            "pytest/onnx/numpy, and a screen skipped because an import failed is a "
            "screen that does not exist. It reports ERROR(instrument=no_steps_parsed) "
            "rather than PASS if its block-structure assumption ever stops matching.",
            "PS2's native-call detection is a finite, explicit allowlist (`&`, and the "
            "bare names cargo/rustc/rustup/python/pip/git/7z/gh/npm/node, or a bare "
            "`*.exe` token), not an interpreter -- a native tool invoked through some "
            "OTHER PowerShell idiom (`Start-Process`, `$?`, a `try`/`catch` around a "
            "native call, or a tool name outside that list) is outside both named "
            "rules. This is the honest remainder of what used to read '... through some "
            "OTHER PowerShell idiom' before issue #55 closed the specific no-capture "
            "gap that phrase was covering for.",
            "It only asks whether the LAST line of the script is a consuming `exit` "
            "(PS1) or a Write-Host verdict with a native call somewhere before it "
            "(PS2). A step whose success path passes through a captured value (or a "
            "native call) and consumes/reports on it several statements before "
            "genuinely new, unrelated native output follows would be a different, "
            "unproven shape neither rule reaches.",
        ),
    ),
    Check(
        id="hostfree.powershell_exit_status_negative_control",
        falsifier=FALSIFIER_PLANTED,
        lane=LANE_HOSTFREE,
        step="PowerShell exit-status negative control (demand red on the real shape)",
        watches=(
            "That the PowerShell exit-status screen still goes RED on both real "
            "defects and on both general shapes (PS1 and PS2), not just on the three "
            "specific steps this repository already fixed. A screen that has stopped "
            "firing is indistinguishable from a clean tree."
        ),
        status=DEMONSTRATED,
        mutation=(
            "20 arms: 1 LIVE (today's ci.yml is clean under both rules), 4 REPLAYED "
            "(the real bytes at the commit issue #49 was filed against, asserting the "
            "PS1 condition token AND that both known offending step names are "
            "reported; and separately the real bytes at d2bf65f, the commit issue #55 "
            "was filed against, asserting the PS2 condition token names 'Install Mesa "
            "lavapipe' AND that PS1 alone would have missed it), 15 PLANTED (each "
            "rule's shape under a different name, each adjacent correct idiom staying "
            "clean under both rules, a dedicated regression guard for the `.exe`-in-a-"
            "quoted-string false positive, and both instrument paths)."
        ),
        arm_healthy="20/20 arms fire as specified, exit 0",
        arm_broken=(
            "Caught a real bug in the screen on its first run: the FAIL branch printed "
            "the human-readable report but never the R13 `FAIL(condition=...)` token "
            "itself -- the exact defect class `check_build_precondition.py`'s own "
            "negative control found in that screen on 2026-08-03, in a different "
            "screen written four days later. Issue #55's PS2 widening separately caught "
            "its own would-be false positive during authoring: an early draft of "
            "PS2's native-call regex matched a `.exe` substring inside 'Install LunarG "
            "Vulkan SDK''s quoted download URL, which was never invoked as a command; "
            "the regression guard arm above pins that case shut."
        ),
        observed="2026-08-07",
        misses=(
            "The REPLAYED arms each pin a commit hash. If either commit is ever "
            "unreachable (a squashed history, a fresh shallow clone) that pair of arms "
            "ERROR(instrument=...) rather than passing quietly, but that still leaves "
            "the corresponding rule's only non-planted evidence unavailable.",
        ),
    ),
    Check(
        id="hostfree.cleanroom_index_url_privacy",
        falsifier=FALSIFIER_PLANTED,
        lane=LANE_HOSTFREE,
        step="Cleanroom index-URL privacy tests (nothing echoes or persists a credential)",
        watches=(
            "Issue #55: every surface on which `python/verify_cleanroom.py` could expose "
            "the `--index-url` it was handed. A private mirror's URL can carry userinfo "
            "(`user:pass@`, a bare username, percent-encoded credentials), a query "
            "credential (`?token=`, `?api_key=`, `?<whatever the vendor called it>=`) or "
            "a fragment. The real value must reach pip's argv and NOTHING else: not the "
            "`$ ...` progress echo, not the persisted "
            "`bench/results/cleanroom_install_dev0.json`, not an exception's rendered "
            "text, not the final summary print. The tests drive production `main()` with "
            "a shell-free fake `subprocess` and a redirected repository root, then assert "
            "on the bytes that production code actually printed, raised and wrote."
        ),
        status=DEMONSTRATED,
        mutation=(
            "REPLAYED from committed, content-addressed fixtures rather than from a "
            "commit ref: `ci/fixtures/cleanroom-redaction/verify_cleanroom.rejected-r1"
            ".pysrc` and `...rejected-r2.pysrc` are the exact module bytes PR #57 was "
            "rejected at twice, each pinned by a sha256 in that directory's "
            "`manifest.json` and verified before use, so the evidence survives a squash "
            "landing that deletes the branch. r1 gated `_echo_cmd` on `\"://\" in s` and "
            "truncated before scrubbing; r2 echoed a schemeless `--index-url` raw and "
            "mangled ordinary `@`-bearing Windows paths. PLANTED: twelve surgical "
            "reintroductions of one defect each into today's module -- the `://` echo "
            "gate, the echo losing the run's own URL, loss of argument-context "
            "recognition, the `@`-guessing scanner, removal of the raw-derived literal "
            "pass, truncate-before-scrub, a denylist-shaped query redaction, removal of "
            "the record-wide scrub at the write seam, re-chaining the exception so the "
            "raw message stays reachable through `__context__`, widening `except "
            "Exception` back to `BaseException`, swallowing the traceback instead of "
            "scrubbing and recording it, and fragment pass-through. Each mutation "
            "asserts its anchor occurs exactly once before applying, so a drifted anchor "
            "is ERROR(instrument=anchor_drift), never a silent walkover. See "
            "ci/negative_control_cleanroom_redaction.py."
        ),
        arm_healthy=(
            "365 tests pass; `pip_index_url` renders as `https://REDACTED@mirror.example/"
            "pypi/simple?token=REDACTED#REDACTED` while the unredacted URL is present in "
            "exactly one observed argv, pip's own"
        ),
        arm_broken=(
            "on either rejected fixture's bytes the suite goes RED, and the same "
            "synthetic sentinel is produced directly by the shipped functions: "
            "`sentinel-pass` by r1's `_echo_cmd` for a schemeless and a scheme-relative "
            "URL and by r1's `_scrub_text(stderr[-1500:], url)` for a URL straddling the "
            "truncation edge; `sentinel-token-value` by r2's `_echo_cmd` for three "
            "schemeless spellings; and r2 rewrites "
            "`C:\\Users\\justin.chu@contoso.example\\...` to "
            "`REDACTED@contoso.example\\...`"
        ),
        observed="2026-08-07",
        misses=(
            "A credential embedded in the URL PATH (`/t/<token>/simple`, as some signing "
            "mirrors do) is deliberately NOT redacted: after userinfo, query and fragment "
            "are gone the path is the only provenance left, so it is kept. This is a "
            "policy decision recorded in README.md and pinned by its own test, not an "
            "oversight -- but it is a real leak for anyone whose mirror signs that way.",
            "The scrub covers only what THIS module echoes, persists or raises. pip's own "
            "logs (`pip.log`, `~/.cache/pip`), the OS process table, and any proxy access "
            "log are outside its reach, and no test here can see them.",
            "The redaction is textual. A credential that never appears in a URL-shaped "
            "span -- an environment variable pip reads on its own, a netrc entry, a "
            "keyring lookup -- is not a shape this module can recognise, so it is neither "
            "redacted nor detected.",
            "UNDER-FIRE class, and the price of fixing issue #55 R3: the general scan now "
            "requires an explicit `//` authority marker, so a FOREIGN schemeless "
            "credential URL -- one this run was never handed, appearing only in text pip "
            "or the OS produced, e.g. `other.example/simple?token=...` -- is not redacted "
            "by shape. The run's OWN URL is still covered in every spelling, by value and "
            "by argument position, and its credential literals are redacted even when "
            "quoted back without their URL. The alternative was corrupting every ordinary "
            "`C:\\Users\\first.last@corp\\...` path in the record, which is what revision "
            "2 did.",
            "OVER-FIRE class, recorded here because a screen's misses prose has until now "
            "only carried false NEGATIVES: the general scanner will still redact an "
            "e-mail address and a `#sha256=` fragment inside a genuine `//`-anchored URL "
            "in `pip freeze` output, because both are indistinguishable from a credential "
            "by shape alone. That is chosen (the wheel digest is recorded verbatim and "
            "independently as `wheel_sha256`) and pinned by a test, but it does cost "
            "readability in the record. A bare e-mail address outside a URL is no longer "
            "touched, and a 16-entry never-mangle table pins that.",
        ),
    ),
    Check(
        id="hostfree.cleanroom_index_url_privacy_control",
        falsifier=FALSIFIER_OBSERVED,
        lane=LANE_HOSTFREE,
        step="Cleanroom index-URL privacy negative control (demand red on the real bytes)",
        watches=(
            "That the privacy suite above is load-bearing rather than merely present. A "
            "suite that has only ever been seen green cannot distinguish 'nothing leaks' "
            "from 'nothing is asserted' -- and that is not hypothetical here: the "
            "PREVIOUS version of that suite was green on bytes that echoed a raw "
            "`user:pass@host` to stdout and wrote a password fragment into a tracked "
            "artifact, because it re-implemented `main()`'s argv construction locally "
            "instead of observing production code."
        ),
        status=DEMONSTRATED,
        mutation=(
            "29 arms: 3 INTEGRITY (each replay fixture matches the sha256 declared in "
            "`ci/fixtures/cleanroom-redaction/manifest.json`, and a deliberately tampered "
            "fixture is REFUSED -- a stale or edited fixture fails loudly rather than "
            "replaying the wrong bytes), 1 LIVE (today's module is green), 12 REPLAYED "
            "(today's suite against each rejected fixture demanding RED, plus each "
            "measured defect reproduced in-process: B1 for the schemeless and "
            "scheme-relative spellings, B2 for the truncation straddle, B3 for a query "
            "credential, R2 for three schemeless spellings, R3 for a corrupted Windows "
            "profile path and a corrupted e-mail address, and N1 for non-idempotence -- "
            "each asserting the SENTINEL is present in what the shipped function "
            "returned, so a 'did NOT reproduce' arm is itself a failure), 12 PLANTED (one "
            "defect surgically reintroduced into today's module per arm), 1 LANDING (the "
            "control re-runs itself against a copy of the tree with NO `.git` at all and "
            "demands the replay arms still fire there)."
        ),
        arm_healthy="29/29 arms fire as specified, exit 0",
        arm_broken=(
            "NEGATIVE-CONTROL: FAIL(condition=arm_did_not_fire) naming the arm and "
            "whether the suite was GREEN or RED against what was expected; a mutation "
            "whose anchor text has drifted is "
            "ERROR(instrument=anchor_drift) naming the mutation, never a quiet pass"
        ),
        observed="2026-08-07",
        misses=(
            "The replay fixtures are bytes, not history. They prove what the module DID "
            "at each rejected head and that today's suite is red on it; they cannot prove "
            "the fixture is what that head really contained to anyone who does not trust "
            "the commit that added it. The `origin` field in the manifest is provenance "
            "prose, deliberately NOT an arm -- when the ref is still reachable the "
            "INTEGRITY arms report that it matches, and when a squash landing has made it "
            "unreachable they say so and pass on the digest alone. Substituting a "
            "flattering fixture would take a reviewed commit, which is the same trust "
            "boundary as the tests themselves.",
            "OVER-FIRE class: the digest is over CRLF-normalised bytes, so a fixture that "
            "differs only in line endings still verifies. That is deliberate on a repo "
            "with `core.autocrlf=true` (the alternative is a control that fails on every "
            "Windows checkout), but it means the arms cannot detect a line-ending-only "
            "tamper -- which for Python source cannot change behaviour.",
            "The PLANTED arms are exact-substring surgery on the module's current text. "
            "They prove each named mechanism is load-bearing; they say nothing about a "
            "leak mechanism nobody has thought of, which is precisely how B1 survived the "
            "first round and R2 the second.",
            "The LANDING arm proves the replay arms survive the ABSENCE of history. It "
            "does not exercise a real squash-merge conflict, and it cannot prove the "
            "fixture files themselves were carried across the landing -- only that "
            "nothing in the control needs `git` once they are.",
        ),
    ),
    Check(
        id="device.flake_witness",
        falsifier=FALSIFIER_OBSERVED,
        lane="both",
        step="Flake witness (name the failure where truncation cannot reach it)",
        watches=(
            "Two things a single run cannot see. (1) A failing test NAME that does not "
            "survive the transport: the coordinator's merge gate went red once in seven "
            "on 2026-08-03 and the name was lost to a truncated log tail, leaving a red "
            "with no subject — a signal that teaches people to press re-run. (2) An "
            "INTERMITTENT: one test id that both fails and does not fail at the SAME "
            "commit, in the same suite, on the same lane. That fact exonerates the "
            "commit and indicts the test, and it is the fact the person staring at one "
            "red and six greens actually needs."
        ),
        status=DEMONSTRATED,
        mutation=(
            "Flip one outcome in the ledger built from 42 real captured `cargo test "
            "--lib` logs and demand red; plus eleven planted arms covering the shapes "
            "that LOOK like a flake and are not (a regression across commits, a "
            "portability difference across lanes, a test that stopped running)."
        ),
        arm_healthy=(
            "FLAKE-WITNESS: PASS — 42 real Linux run logs at d46327b, 0 intermittent "
            "(ci/negative_control_flake_witness.py LIVE arm)"
        ),
        arm_broken=(
            "FLAKE-WITNESS: FAIL(condition=intermittent) naming the id, the commit, the "
            "lane, and both sets of run ids — 'THE COMMIT IS EXONERATED AND THE TEST IS "
            "NOT'"
        ),
        observed="2026-08-03",
        misses=(
            "AN INTERMITTENT NEEDS TWO OBSERVATIONS AND THIS BUYS THE SECOND ONE CHEAPLY; "
            "it does not buy the first. A 1-in-40 needs roughly forty runs at one commit "
            "before the join can see it, so on a hosted runner with no cache across runs "
            "the ledger has ONE run in it and this check can only annotate, never "
            "conclude. --require-history exists so a lane can refuse to pretend "
            "otherwise, and no lane sets it yet.",
            "pytest names only its failures, so the complement is INFERRED (NOT_FAILED, "
            "not PASSED) and NOT_FAILED includes skipped and deselected. Where two runs' "
            "executed counts differ by more than the tolerance it says INCOMPARABLE "
            "rather than claiming a flake — a test that stopped running is "
            "check_suite_productivity's defect class, not this one.",
            "It names WHICH test is intermittent, never WHY. "
            "vk::barrier::tests::backend_probe_* was 1-in-9 on Linux a round ago because "
            "backend_probe_* is a PROCESS-GLOBAL env var the tests race for; that is "
            "Trinity's env-var auditor. NOTE: 40 consecutive runs at d46327b produced "
            "ZERO failures, so that particular flake no longer reproduces here "
            "(p ~= 0.009 under the old 1-in-9 rate). Something fixed it and nobody "
            "recorded fixing it.",
            "It cannot see an intermittent that differs BETWEEN commits rather than "
            "within one. Keying on the commit is the whole point — it is what separates "
            "'your change broke it' from 'it does that' — and it is also the limit.",
        ),
    ),
    Check(
        id="hostfree.flake_witness_negative_control",
        falsifier=FALSIFIER_PLANTED,
        lane=LANE_HOSTFREE,
        step="Flake-witness negative control (demand red on an intermittent)",
        watches=(
            "That the flake witness still fires. It is silent on a repo with no "
            "reproducing flake, and silence from a check is indistinguishable from "
            "silence from no check."
        ),
        status=DEMONSTRATED,
        mutation="13 arms: 1 LIVE, 1 REPLAYED (a flipped outcome in the real ledger), 11 PLANTED.",
        arm_healthy="13/13 arms fire as specified, exit 0",
        arm_broken="exit 1 with the misfiring arm's full output, per arm",
        observed="2026-08-03",
        misses=(
            "The LIVE arm's subject is two TRACKED logs under ci/fixtures/flake-witness/ "
            "plus whatever untracked runs happen to be present. Without the tracked pair "
            "the control would quietly become all-planted on a hosted runner, which is "
            "the exact defect it exists to catch.",
        ),
    ),
    Check(
        id="build.contention_gate",
        falsifier=FALSIFIER_OBSERVED,
        lane="both",
        step="Contention gate (repeat the process-global families; keep the whole red)",
        watches=(
            "Intermittent failures caused by tests sharing a process-global — the counters "
            "statics, the ORT logger pointers, or an environment variable — which the single "
            "`cargo test --lib` above cannot see. A 1-in-40 per-run rate passes 97.5% of "
            "merges, and on 2026-08-03 exactly that happened: one red, six greens, pushed."
        ),
        status=DEMONSTRATED,
        mutation=(
            "None planted. Remove the three `let _g = crate::allocator::ledger::test_lock();` "
            "lines from the `backend_probe_*` tests in rust/src/vk/barrier.rs — i.e. restore "
            "the tree as it stood at main@d46327b — and the gate goes red on its own."
        ),
        arm_healthy=(
            "CONTENTION GATE: GREEN — 5 pools, 20 reps each, 0/20 red on every pool, 38 s "
            "wall (Windows, 2026-08-03, post-fix tree)"
        ),
        arm_broken=(
            "CONTENTION GATE: RED — 3 of 20 reps of pool env:backend_probe, naming "
            "vk::barrier::tests::backend_probe_writes_sync2_token at src/vk/barrier.rs:918:9 "
            "and, in rep 16, backend_probe_writes_legacy_token at src/vk/barrier.rs:899:51 in "
            "the SAME run; full capture retained per red repetition"
        ),
        observed="2026-08-03",
        misses=(
            "Its pools come from audit_counter_test_lock, whose stated extent is test "
            "BODIES, not call graphs. A test that reaches a process-global only through a "
            "helper defined outside its own body is in neither the auditor nor this gate.",
            "It repeats; it does not control scheduling. A race whose window is narrower "
            "than this machine's scheduling jitter can still pass 20 reps. The observed "
            "rate is printed per pool so a decreasing-but-nonzero rate is visible rather "
            "than rounding to green.",
            "20 reps detect a 0.025 per-run rate with probability 0.397 — under half. Its "
            "power comes entirely from the pools being NARROW; a defect that only "
            "manifests in the full 525-test population is outside it, and the full-suite "
            "step above remains the only lane that runs that population at all.",
            "Race conditions between the EP and ORT, or between two processes, are not "
            "test-pool contention and are invisible here.",
        ),
    ),
    Check(
        id="hostfree.test_lock_auditor",
        falsifier=FALSIFIER_OBSERVED,
        lane=LANE_HOSTFREE,
        step="Test-lock auditor (process-global families, static, no GPU)",
        watches=(
            "A #[test] that touches a process-global — counters statics, ORT logger "
            "pointers, or a CONTENDED environment variable — without holding "
            "crate::allocator::ledger::test_lock(); and a test holding some OTHER mutex, "
            "which is indistinguishable from a correct guard by eye."
        ),
        status=DEMONSTRATED,
        mutation=(
            "None planted for the positive arm: at main@d46327b the tool reports three real "
            "ENV-UNGUARDED findings (the backend_probe_* family). --selftest additionally "
            "carries nine specimens: unguarded, foreign-lock, contended-env, guarded-env, "
            "sole-writer, #[ignore]d writer, and a top-level #[test] in a whole-file test "
            "module — the last of which the pre-Round-37 scanner could not see at all."
        ),
        arm_healthy="0 unguarded, 0 wrong-lock, 0 unguarded env over 3 contended variables",
        arm_broken=(
            "ENV-UNGUARDED vk/barrier.rs:882 backend_probe_writes_legacy_token "
            "[ONNXRUNTIME_EP_VULKAN_BACKEND_PROBE] (+2 more), exit 1"
        ),
        observed="2026-08-03",
        misses=(
            "Bodies, not call graphs — see build.contention_gate. This is why the repeat "
            "gate exists beside it rather than instead of it.",
            "It gates only CONTENDED variables (>=2 live tests, >=1 writer). A single test "
            "mutating a variable that production code reads on another thread is REPORTED "
            "as a sole writer and deliberately not failed, because deciding it needs the "
            "call graph this tool does not have.",
            "It cannot see a global that is neither a counters call, a logger attach, nor "
            "an env access — a fourth family would be invisible exactly as the environment "
            "was until 2026-08-03, and nothing in a green run would say so.",
        ),
    ),
    Check(
        id="hostfree.contention_gate_extractor",
        falsifier=FALSIFIER_OBSERVED,
        lane=LANE_HOSTFREE,
        step="Contention-gate failure extractor (replay a real red, demand names)",
        watches=(
            "The half of the 2026-08-03 incident that was not the flake: a gate that goes "
            "red and cannot say WHAT failed. The failing test name was lost to a "
            "`Select-Object -Last 2`, so the incident produced a red with no name."
        ),
        status=DEMONSTRATED,
        mutation=(
            "None planted. rust/tools/contention_gate_red_fixture.txt is a verbatim capture "
            "of a real red repetition from this tree (rep 16, two tests failing in one run)."
        ),
        arm_healthy=(
            "selftest 5/5 — two names and a panic site recovered from the real red, a green "
            "capture stays silent, and the tail-2 slice of that same red yields NOTHING, "
            "which is the incident reproduced as an assertion"
        ),
        arm_broken=(
            "Delete or truncate the fixture, or narrow the extractor to the `failures:` "
            "block alone, and the selftest reports the missing names by name"
        ),
        observed="2026-08-03",
        misses=(
            "It certifies the EXTRACTOR, not the gate's ability to provoke a failure. That "
            "is build.contention_gate's arm, and it needs a Rust toolchain.",
            "One fixture, one failure shape (libtest panic). A harness abort with no "
            "`test ... FAILED` lines and no panic line would yield no names and this "
            "selftest would not notice.",
        ),
    ),
    Check(
        id="hostfree.lane_check_productivity",
        falsifier=FALSIFIER_PLANTED,
        lane=LANE_HOSTFREE,
        step="Two-polarity suite asserted something (productivity floor)",
        watches=(
            "That the suite which certifies every other check in ci/ collected and ran "
            "its 116 tests, rather than reporting success for collecting none."
        ),
        status=DEMONSTRATED,
        mutation=(
            "Feed ci/check_suite_productivity.py a log reading `400 passed, 100 skipped` "
            "against the tests/ops floor, and `no tests ran in 0.01s` against the "
            "single-test floor."
        ),
        arm_healthy="SUITE-PRODUCTIVITY: PASS — 116 executed (3 failed, 113 passed), floor 100",
        arm_broken=(
            "SUITE-PRODUCTIVITY: FAIL(condition=collected_below_floor) and "
            "FAIL(condition=no_tests_ran) — see ci/negative_control_suite_productivity.py"
        ),
        observed="2026-08-03",
        misses=(
            "A floor is a lower bound on WORK, never on correctness. 116 vacuous tests "
            "clear it exactly as well as 116 real ones.",
        ),
    ),
    Check(
        id="hostfree.suite_productivity_negative_control",
        falsifier=FALSIFIER_OBSERVED,
        lane=LANE_HOSTFREE,
        step="Suite-productivity negative control (demand red on a suite that did nothing)",
        watches=(
            "ci/check_suite_productivity.py itself — every condition and every instrument "
            "path, including the two LIVE arms from a real environment with the optional "
            "dependency really removed."
        ),
        status=DEMONSTRATED,
        mutation=(
            "The control IS the mutation: 25 arms, each asserting a specific terminal "
            "state. Deleting any condition branch from check_suite_productivity.py turns "
            "its arm red on the next run."
        ),
        arm_healthy="NEGATIVE-CONTROL: PASS — every arm fired as declared (3 LIVE / 1 REPLAYED / 21 PLANTED)",
        arm_broken=(
            "The LIVE pre-fix arm: a scratch venv built without onnx-shape-inference "
            "reproduced `Interrupted: 1 error during collection` over all 665 tests, and "
            "the check reported FAIL(condition=collection_error). The REPLAYED arm is "
            "stronger: bench/results/linux_lavapipe_optests.txt is a real green run "
            "reading `2 passed, 36 skipped`, and it now trips collected_below_floor."
        ),
        observed="2026-08-03",
        misses=(
            "The LIVE arms are replayed logs from a real run, not a live re-run. If "
            "pytest's summary format changes, the arms keep passing against text that no "
            "longer occurs — which is why summary_not_found and "
            "unrecognised_outcome_word are ERROR(instrument) rather than a shrug.",
        ),
    ),
    Check(
        id="device.op_correctness_productivity",
        falsifier=FALSIFIER_OBSERVED,
        lane="both",
        step="Op-correctness step asserted something (productivity floor)",
        watches=(
            "That the op-correctness step collected and executed tests at all. Until "
            "2026-08-03 one missing OPTIONAL Python dependency aborted collection of the "
            "whole tests/ops directory, and an all-skipped run exits ZERO."
        ),
        status=DEMONSTRATED,
        mutation=(
            "Build a Python environment without `onnx-shape-inference` and run "
            "`pytest tests/ops`. Before the 2026-08-03 fix this produced "
            "`Interrupted: 1 error during collection`; the floors also fire on any log "
            "whose accounted total drops below 660 or whose executed count drops below "
            "300."
        ),
        arm_healthy=(
            "SUITE-PRODUCTIVITY: PASS — 665 collected, 316 executed in the WEAKEST "
            "possible environment (no EP library, no ICD, optional dep absent)"
        ),
        arm_broken=(
            "FAIL(condition=collection_error) quoting "
            "'ERROR collecting tests/ops/test_shape_inference_delta.py' and "
            "'Interrupted: 1 error during collection'; and, on a real historical log, "
            "FAIL(condition=collected_below_floor) '38 test(s) collected, floor is 660'"
        ),
        observed="2026-08-03",
        misses=(
            "The per-lane executed floors are set from a developer machine, NOT from "
            "inside build-test-linux or build-test-windows. Nobody has measured either "
            "runner's executed count, and ci/suite_floor.json says so rather than "
            "carrying two invented numbers that would read as measured.",
            "It cannot see a suite that runs its floor's worth of tests against the CPU "
            "EP. That is ci/check_verdict.py and the Criterion 10 gate.",
        ),
    ),
    Check(
        id="device.no_icd_step_productivity",
        falsifier=FALSIFIER_PLANTED,
        lane=LANE_WINDOWS,
        step="No-ICD fallback step asserted something (productivity floor)",
        watches=(
            "A step whose entire content is ONE pytest node id. A renamed test, a moved "
            "file or a new skip makes it run nothing and say nothing."
        ),
        status=DEMONSTRATED,
        mutation=(
            "Feed the check a `no tests ran in 0.01s` log against the single-test floor "
            "of 1."
        ),
        arm_healthy="SUITE-PRODUCTIVITY: PASS on `1 passed in 0.40s`",
        arm_broken="SUITE-PRODUCTIVITY: FAIL(condition=no_tests_ran)",
        observed="2026-08-03",
        misses=(
            "It cannot tell a test that ran under a REAL absent ICD from one that ran "
            "with the environment override silently ineffective. That is "
            "ci/check_icd_suppression.py's subject.",
        ),
    ),
    Check(
        id="device.ledger_loss_windows_namespace_regression",
        falsifier=FALSIFIER_OBSERVED,
        lane=LANE_WINDOWS,
        step="Windows-namespace destination-policy regression (probe_ledger_loss)",
        watches=(
            "That `ci/test_lane_checks.py`'s `probe_ledger_loss` subset — including the "
            "parametrized `\\\\?\\C:\\...` extended-length and `\\\\localhost\\C$\\...` "
            "admin-share regressions added for the PR #51 review — actually executes "
            "against a real Windows filesystem, not only against the `ubuntu-latest` "
            "`lane-checks` job (where every Windows-only case in that file is "
            "`skipif(sys.platform != 'win32')` and never runs at all)."
        ),
        status=DEMONSTRATED,
        mutation=(
            "Reverted `rust/tools/probe_ledger_loss.py` to the rejected PR #51 head "
            "(9c15fdf) in place, keeping the hardened test file, and ran this step's own "
            "command on a real Windows checkout."
        ),
        arm_healthy=(
            "31 passed, 268 deselected — the fixed `classify_destination()` in place "
            "(local Windows run, 2026-08-06)"
        ),
        arm_broken=(
            "9 failed, 22 passed — including both literal bypass regressions "
            "(`...refuses_every_tracked_destination...[extended-length-prefix-\\\\?\\]` "
            "and `[localhost-admin-share-\\\\localhost\\C$]`) against the unmodified "
            "9c15fdf source, on the same real Windows checkout (local run, 2026-08-06)"
        ),
        observed="2026-08-06",
        misses=(
            "It is not wired into ci/check_suite_productivity.py or "
            "ci/check_flake_witness.py: `ci/test_lane_checks.py` already names those "
            "under the `lane-checks` job, and a second, differently-scoped floor entry "
            "for the same file under `build-test-windows` would be a second answer to "
            "how much of it must run. A silent all-skip here (e.g. every "
            "`_WINDOWS_ONLY` guard misfiring) would still exit 0 and this table would not "
            "catch that on its own — the step's own log is the only witness.",
        ),
    ),
    Check(
        id="build.portability_lint",
        falsifier=FALSIFIER_OBSERVED,
        lane="both",
        step="cargo test --test portability",
        watches=(
            "Platform-conditional bindings named without a cfg gate — 'the class of bug "
            "that only shows up on a CI lane for another OS, hours later', in the lint's "
            "own words."
        ),
        status=RED_NOW,
        mutation="None needed. The tree is the failing arm.",
        arm_healthy="not currently observable — see arm_broken",
        arm_broken=(
            "PORTABILITY LINT FAILED - 1 violation(s) of rule P1. src/ep.rs:2457 "
            "'_file: *const ort::wchar_t' is named without a #[cfg(windows)] gate on the "
            "item that defines it. bindgen only emits wchar_t when targeting Windows, so "
            "cargo test --lib cannot compile on Linux. Remedy: the cfg-selected OrtChar "
            "alias, as tests/mock_ort/mod.rs:155/158 already does."
        ),
        observed="2026-08-01",
        misses=(
            "It is a source lint, not a build. It sees names, not linkage: a symbol that "
            "exists on both platforms but MEANS something different passes it.",
        ),
    ),
    Check(
        id="build.integration_targets",
        lane="both",
        step="cargo test --test cdylib_load/dump_capabilities/host_registration/validation_control",
        watches=(
            "That the built cdylib loads, that capability dumping works, that the EP "
            "registers with a mock ORT host, and that validation-layer control behaves. "
            "Four integration targets that existed in rust/tests/ and had never been run "
            "by CI."
        ),
        status=UNDEMONSTRATED,
        arm_healthy="cdylib_load 1, dump_capabilities 6, host_registration 1, validation_control 3 — all pass (local, 2026-08-01)",
        observed="2026-08-01",
        misses=(
            "Nobody has watched any of these go red, so each is a candidate for evidence "
            "rather than evidence. They are wired now, which is the precondition for "
            "falsifying them; being wired is not itself the falsification.",
            "host_registration talks to tests/mock_ort, not to ORT. It proves the shape "
            "of the registration call, not that the real host accepts it.",
            "ON THE LINUX LANE THIS HAS NEVER EXECUTED AT ALL (added 2026-08-02). The "
            "`observed` date above is a Windows local run. The Linux job dies at clippy "
            "and skips the remaining seven steps including this one, so the status is "
            "UNDEMONSTRATED on Windows and GATED_NEVER_RUN on Linux. The entry carries "
            "the weaker of the two rather than splitting, and this line is why. A "
            "per-lane status is the right fix if a second such entry appears.",
            "GATE REMOVED 2026-08-02, STILL NOT OBSERVED IN CI. The eleven compile "
            "errors are fixed and `cargo check --all-targets` now runs as its own step "
            "ahead of clippy, so nothing in the workflow blocks this step any more. Run "
            "by hand on WSL Ubuntu the four targets PASS. That is a local run, not a "
            "lane: no GitHub Actions job has executed since, so the Linux side of this "
            "entry stays GATED_NEVER_RUN until an actual run says otherwise. A gate "
            "removed in YAML is not a check that has run.",
        ),
    ),
    Check(
        id="build.layering_lint",
        falsifier=FALSIFIER_PLANTED,
        lane="both",
        step="cargo test --test layering",
        watches="ash types leaking out of rust/src/vk/ into modules that must stay portable.",
        status=DEMONSTRATED,
        mutation="Plant `use ash::vk as _;` among the imports of rust/src/ops/norm.rs.",
        arm_healthy="26 passed",
        arm_broken=(
            "layering violations in src/ops/ — 'src/ops/norm.rs:30: layering rule 2 (no "
            "raw Vulkan) violated by `ash` — op code must not depend on the Vulkan "
            "bindings; express intent via DispatchContext'"
        ),
        observed="2026-08-01",
        misses=(
            "Its scope is src/ops/ ONLY. The same `use ash::vk as _;` planted in "
            "src/trace.rs passed all 26 tests on 2026-08-01, despite the archived "
            "decision that put the timestamp arithmetic in trace.rs specifically to keep "
            "it 'on the right side of the layering lint (no ash)'. The rule the decision "
            "relied on does not exist. Recorded rather than fixed: rust/tests/ is not "
            "mine.",
            "It reads source text, not the dependency graph. A module that reaches ash "
            "through a re-export it does not name is invisible to it.",
        ),
    ),
    Check(
        id="build.compile_all_targets",
        falsifier=FALSIFIER_OBSERVED,
        lane="both",
        step="cargo check --release --all-targets",
        watches=(
            "That the crate's test, bench and example targets COMPILE on this lane. "
            "`cargo build --release` compiles the lib only; everything else was first "
            "compiled by clippy, because clippy was the lane's first --all-targets "
            "invocation."
        ),
        status=RED_NOW,
        mutation=(
            "None needed for the observed arm — the tree WAS the failing arm until "
            "2026-08-02. Eleven bindgen-signedness errors in rust/src/ep.rs "
            "(`ort::OrtLoggingLevel` is `c_int` under MSVC and `c_uint` under GCC, both "
            "verified by reading the two generated `ort.rs` files) made `cargo test "
            "--lib` refuse to compile on Linux while Windows stayed green. To re-arm it: "
            "declare a severity carrier as `i32` in any module of ep.rs's test tree."
        ),
        arm_healthy=(
            "Linux `cargo test --lib --no-run` compiles (WSL Ubuntu 24.04, cargo 1.97.1, "
            "2026-08-02, after the alias fix). NOT YET OBSERVED IN CI — this step is new "
            "and the workflow has not run since."
        ),
        arm_broken=(
            "error[E0308]: mismatched types x6 and error[E0277]: can't compare `i32` with "
            "`u32` x5 in src/ep.rs, at lines 2769, 2813, 2825, 2856, 2914 and 3000 — "
            "reproduced on WSL Ubuntu before the fix, and the reason the seven steps "
            "behind it had never executed."
        ),
        observed="2026-08-02",
        misses=(
            "It is `cargo check`, not `cargo build`: it type-checks and borrow-checks "
            "every target but does not codegen or link them. A defect that only appears "
            "at link time on one platform is invisible here and is caught, if at all, by "
            "the build step and the integration targets.",
            "It says nothing about whether anything is CORRECT. It is the weakest claim "
            "in the lane on purpose — 'this compiles here' — and its whole value is that "
            "it is the claim clippy's name was accidentally making.",
            "It cannot see a platform whose lane does not exist. macOS and Android are "
            "in PLATFORMS.md and in no job, so nothing compiles them either.",
        ),
    ),
    Check(
        id="build.clippy",
        falsifier=FALSIFIER_OBSERVED,
        lane="both",
        step="cargo clippy --release -D warnings",
        watches="Lint regressions.",
        status=RED_NOW,
        mutation="None needed. The tree is the failing arm.",
        arm_healthy="not currently observable — see arm_broken",
        arm_broken=(
            "cargo clippy --release --all-targets -- -D warnings, the exact CI command, "
            "on the same `stable` channel CI installs (rustc 1.97.1): 'error: unused "
            "import: crate::engine::DeviceMemoryProvider', 'error: manual "
            "RangeInclusive::contains implementation', 'error: direct cast of function "
            "item into an integer' x2, 'could not compile onnxruntime-ep-vulkan (lib "
            "test)'. Reproduced locally and observed failing on both device lanes of the "
            "2026-08-01 main run."
        ),
        observed="2026-08-01",
        misses=(
            "Everything that is not a lint. Clippy has never had an opinion about whether "
            "a kernel computes the right number.",
            "The failures are all in --all-targets (test-profile) code, which is why they "
            "went unnoticed: `cargo build --release` is clean. A lane that builds but "
            "cannot lint its own tests looks healthy from the build step.",
            "UNTIL 2026-08-02 IT ALSO MISSED THE DIFFERENCE BETWEEN A LINT AND A COMPILE "
            "ERROR, and reported both under this name. It was the lane's first "
            "--all-targets invocation, so every compile error in test code arrived here "
            "wearing a lint's name — which is how eleven Linux-only type errors were "
            "triaged as low-priority style for a day while seven steps behind them never "
            "ran. Split: `build.compile_all_targets` now runs first, and this step's red "
            "means a lint again. The class is recorded in decisions as `misnamed`, second "
            "specimen after `Phase::Record`.",
        ),
    ),
    # ── device lanes: execution ───────────────────────────────────────────────
    Check(
        id="device.op_correctness",
        lane="both",
        step="pytest tests/ops (lavapipe)",
        watches="Op outputs against a CPU-EP reference, bit-exact or within a stated tolerance.",
        status=GATED_NEVER_RUN,
        mutation=(
            "NOT YET PERFORMED, and — corrected 2026-08-02 — IT CANNOT BE, BECAUSE THIS "
            "STEP HAS NEVER EXECUTED IN CI. The Linux job fails at `Clippy (all warnings "
            "as errors)` and GitHub Actions marks the remaining seven steps *skipped*, "
            "this one among them. I had it at UNDEMONSTRATED all session, which reads as "
            "'runs, never observed to fail' and is one demonstration away from green. It "
            "is not: it has not started. `epctl --probe-loader` on lavapipe reports gate "
            "PASS, so the device is reachable — the step is gated, not impossible.\n\n"
            "AND IT WOULD NOT ASSERT ANYTHING IF THE GATE WERE LIFTED TODAY. Run by hand "
            "on lavapipe at d375a4d (WSL Ubuntu 24.04): `2 passed, 36 skipped`, every "
            "skip reading 'No Vulkan device available'. The cause is not the device. "
            "Every ledger entry faults on Linux because `shader_digest` is over the "
            "embedded SPIR-V bytes and Ubuntu's glslc (shaderc 2023.8) emits different "
            "bytes than the Vulkan SDK's, so the EP claims 0/1 nodes and the harness's "
            "`_probe_vulkan_device` reports that decision as an absent device. Fixing "
            "clippy alone would turn this step GREEN HAVING ASSERTED NOTHING. See "
            "PLATFORMS.md §7.19 and `ci/check_ledger_portability.py`, which goes red on "
            "exactly that run.\n\n"
            "UPDATE 2026-08-02 (Link): THE GATE IS GONE AND THE STEP IS STILL NOT GREEN, "
            "which is the point. The eleven errors are fixed, `cargo check "
            "--all-targets` runs as its own named step ahead of clippy, and the whole "
            "Linux job was then run by hand on WSL Ubuntu. This step: 50 failed, 272 "
            "passed, 292 skipped. 48 of the 50 do not fail on Windows, and all 48 are "
            "downstream of the one cause above — the EP claims 0/1 nodes because every "
            "ledger entry faults, so `epctl --check-counters` reports 0 dispatches and "
            "the Criterion 10 verdict is UNATTRIBUTED. It remains GATED_NEVER_RUN "
            "because no CI job has run since; a gate removed in YAML is not a check that "
            "has run. See PLATFORMS.md §7.21."
        ),
        arm_healthy="op suite green on lavapipe — NOT OBSERVED; the step has never run in CI",
        observed=None,
        misses=(
            "Everything lavapipe implements differently from silicon: subgroup width, "
            "fp16 and int8 arithmetic paths, cooperative-matrix paths, and any race a "
            "serial software rasteriser cannot expose. A missing barrier is invisible on "
            "a device that does not run the two dispatches concurrently.",
            "Anything about SPEED. Correct output on a CPU renderer says nothing about "
            "whether the GPU path is fast, and there is no producer here that could say.",
        ),
    ),
    Check(
        id="device.ledger_portability",
        falsifier=FALSIFIER_OBSERVED,
        lane="both",
        step="Proof-ledger portability screen (named run artifacts)",
        watches=(
            "That a run declared to be a device lane actually claimed nodes: no proof-"
            "ledger fault, no `claims 0/N nodes`, and no 'No Vulkan device available' "
            "reported by a harness on a box where the loader gate passed."
        ),
        status=DEMONSTRATED,
        mutation=(
            "Not planted — OBSERVED. Built the EP on Linux (WSL Ubuntu 24.04) at d375a4d "
            "and ran it against lavapipe. Every ledger entry faulted, the session claimed "
            "0/1 nodes, all work went to the CPU EP, and the process exited 0. The same "
            "probe on Windows/NVIDIA at the same commit claims 1 proven form and is "
            "clean. Two live artifacts, opposite polarities, same commit and same ledger."
        ),
        arm_healthy="windows_nvidia_probe_control.txt → PASS",
        arm_broken=(
            "linux_lavapipe_probe.txt → FAIL(condition=ledger_fault); "
            "linux_lavapipe_optests.txt → FAIL(condition=device_absence_misnamed) on a "
            "run whose own summary line reads '2 passed, 36 skipped'"
        ),
        observed="2026-08-02",
        misses=(
            "It takes named artifacts and does not scan the tree. bench/results holds "
            "artifacts that quote ledger faults on purpose, and unlike the device-loss "
            "screen there is no text here that no negative control would ever emit. A "
            "run nobody names is a run this screen reports UNOBSERVABLE for, not a run "
            "it reports clean.",
            "`claimed_nothing` and `device_absence_misnamed` are UNOBSERVABLE unless the "
            "caller supplies --device-lane and --loader-artifact respectively. Both are "
            "printed as UNOBSERVABLE rather than silently passing, but a caller who "
            "forgets them gets a narrower check that still says PASS on the last line.",
            "It says nothing about WHY the digests differ. It reports that this build's "
            "modules do not match what was proven; establishing that the cause is the "
            "glslc version rather than a real kernel edit took a separate comparison and "
            "is written up rather than checked.",
            "It cannot see the deeper defect it exposed: the ledger's predicate is the "
            "shader digest alone, and `device` is recorded in every entry and never "
            "read. This screen fires when the digest disagrees. Nothing fires when the "
            "digest agrees and the device is one no proof run ever touched.",
        ),
    ),
    Check(
        id="device.criterion10_gate",
        falsifier=FALSIFIER_PLANTED,
        lane="both",
        step="Criterion 10 gate artifact + independent reader + epctl --check-counters",
        watches=(
            "That the EP actually EXECUTED the graph and that the verdict is attributed "
            "to it — counts of provider executions, dispatches and profile node events, "
            "plus one exact numerical comparison."
        ),
        status=DEMONSTRATED,
        mutation="Feed the reader a declined artifact (the 'declined' negative control).",
        arm_healthy="PASS with matching counts",
        arm_broken="UNATTRIBUTED — the reader refuses to credit the EP for a graph it declined",
        observed="2026-07-31",
        misses=(
            "It is entirely count-based, which is why it is immune to clock manipulation "
            "(no total, therefore no denominator to inflate). The same property means it "
            "cannot notice a device running at its idle clock, and it never claims to.",
            "Counts are exactly what DESIGN.md 10.0.4 warns about handing a reader "
            "unaccompanied: quoting '354 barriers, 33 dispatches' invites 'so how fast is "
            "it?', and the reader answers with a clock nobody recorded.",
        ),
    ),
    Check(
        id="device.icd_negative_control",
        falsifier=FALSIFIER_PLANTED,
        lane=LANE_LINUX,
        step="Gate negative control - no ICD must produce UNATTRIBUTED",
        watches="That the criterion-10 gate can fail at all: with no ICD there is nothing to attribute.",
        status=DEMONSTRATED,
        mutation="VK_DRIVER_FILES pointed at a nonexistent ICD manifest.",
        arm_healthy="gate PASS with the ICD present",
        arm_broken="vkCreateInstance returned ERROR_INCOMPATIBLE_DRIVER; the gate reports UNATTRIBUTED",
        observed="2026-08-01",
        misses=(
            "It proves the gate CAN fail, which is not the same as proving it fails for "
            "the right reasons. Removing the driver is the crudest possible defect.",
        ),
    ),
    Check(
        id="device.icd_negative_control_windows",
        lane=LANE_WINDOWS,
        step="Gate negative control - no ICD must produce UNATTRIBUTED",
        watches="The same thing, on the elevated Windows runner.",
        status=IMPOSSIBLE_HERE,
        reason=(
            "GitHub's Windows runners run elevated, and the LunarG loader silently ignores "
            "VK_DRIVER_FILES and VK_ICD_FILENAMES for elevated processes. The suppression "
            "does not take, so the control cannot be armed. Until 2026-08-01 the lane hid "
            "this behind a guard that matched a phrase printed on EVERY run, so it "
            "short-circuited to success in both directions and reported 'the gate cannot "
            "fail' — blaming the gate for the instrument. ci/check_icd_suppression.py now "
            "reports ERROR(instrument=icd_suppression_ineffective) instead, which is an "
            "instrument error and never a detection (R13)."
        ),
        observed="2026-08-01",
        misses=(
            "The whole of it. This control has never executed on Windows and, absent a "
            "non-elevated Windows runner, never will.",
        ),
    ),
    Check(
        id="device.fatal_log_line",
        falsifier=FALSIFIER_PLANTED,
        lane="both",
        step="Known-fatal log line is a lane failure (R13 second witness)",
        watches="A fatal EP log line that a green exit code would otherwise swallow.",
        status=DEMONSTRATED,
        mutation=(
            "Feed the checker a captured log containing ORT's fallback announcement: "
            "'[E:onnxruntime:, sequential_executor.cc:516] EP_FAIL : Falling back to "
            "CPUExecutionProvider'."
        ),
        arm_healthy="FATAL-LOG-CHECK: PASS, exit 0",
        arm_broken=(
            "FATAL-LOG-CHECK: FAIL(condition=runtime_fallback_announced_by_ort), exit 1, "
            "quoting the matching line in full — 'ORT abandoned this EP at run time and "
            "re-executed on CPU without raising. Every assertion in the lane may have "
            "passed; they were checking CPU output.'"
        ),
        observed="2026-08-01",
        misses=(
            "It matches known strings. A new fatal condition with new wording is invisible "
            "to it, and the lane stays green.",
            "It needs the log to have been captured with stderr merged. Until 2026-08-01 "
            "it reported ERROR(instrument=log_not_captured) on any lane that died before "
            "the tee step, which is true but adds a second red to a failure it did not "
            "cause; the lane marker now separates those.",
            "Its marker list DOES NOT MATCH the real line. tests/ops/_verdict.py::"
            "FATAL_LOG_MARKERS looks for 'Falling back to CPUExecutionProvider'; ORT "
            "actually prints \"Falling back to ['CPUExecutionProvider'] and retrying.\" — "
            "a list repr, so neither marker is a substring. Found 2026-08-02 by "
            "ci/check_device_loss.py on bench/results/ctx512_device_lost.txt, where this "
            "check reads 0 hits on a log that announces the fallback twice. The file is "
            "Trinity's and the fix is hers; a second private marker list in ci/ would be "
            "the two-dialect failure the shared vocabulary exists to prevent. Until it "
            "lands, this check has been cited as a second witness for five incidents on "
            "the strength of a match it cannot make.",
        ),
    ),
    Check(
        id="device.device_loss_screen",
        falsifier=FALSIFIER_OBSERVED,
        lane="both",
        step="Device-loss screen on this lane's own evidence",
        watches=(
            "A device loss during the lane's own run, named as a run log so the full "
            "condition set applies. Distinct from device.fatal_log_line in EXTENT, and "
            "the two must never be quoted as one guarantee: that check reads ORT's "
            "announcement in a captured log, this one reads the EP's own device-lost "
            "text and a structural truncation rule that needs no text at all."
        ),
        status=DEMONSTRATED,
        mutation=(
            "The reach arm in ci/negative_control_device_loss.py: a log carrying the EP's "
            "device-lost line and no ORT announcement. This check is red on it; "
            "ci/check_fatal_log.py is green on the same file. That difference is the "
            "reason both exist and it is demonstrated rather than argued."
        ),
        arm_healthy="DEVICE-LOSS: PASS on a lane whose device survived",
        arm_broken=(
            "DEVICE-LOSS: FAIL(condition=device_lost_reported) quoting '[vulkan-ep] "
            "ERROR: vkWaitForFences failed: The logical device has been lost'"
        ),
        observed="2026-08-02",
        misses=(
            "It has never run on a lane that lost the device LIVE. Both device losses on "
            "record were caught by replaying artifacts after the fact; inducing a TDR "
            "deliberately is the arm that is still owed.",
            "It cannot see a device loss whose run wrote nothing to either named log.",
            "Trinity's disable_cpu_ep_fallback is a THIRD extent, not a superset: it "
            "makes ORT refuse at session creation on node assignment, so it catches "
            "PLANNED fallback and cannot see a loss that happens on a session ORT has "
            "already accepted. Three mechanisms, three extents, stated separately in "
            "docs/PLATFORMS.md 7.16.",
        ),
    ),
    Check(
        id="device.device_state_guard",
        falsifier=FALSIFIER_PLANTED,
        lane="all",
        step="No duration without a device-state record (10.0 obligation 8)",
        watches=(
            "Any lane-authored duration published without a device-state record carrying a "
            "tenancy verdict and clock min/median/max against the board maximum."
        ),
        status=DEMONSTRATED,
        mutation="Plant a gpu_busy_ms field in a lane artifact with no device_state companion.",
        arm_healthy="PASS - no lane-authored figure, or every figure carries a certified companion",
        arm_broken="FAIL(condition=STEADY_UNCERTIFIED), exit 1",
        observed="2026-08-01",
        misses=(
            "It constrains the ANSWER, not the invitation. It cannot stop a lane "
            "publishing counts that lead a reader to supply their own clock.",
            "On every lane in this project it can only ever return PASS-by-absence or "
            "ERROR(instrument): see the lavapipe entry below. It is a guard against a "
            "future lane, not a measurement of the present one.",
        ),
    ),
)

# ──────────────────────────────────────────────────────────────────────────────
# Blind spots — defect classes NO lane catches, stated once, in one place
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BlindSpot:
    """A real defect class that every CI lane passes.

    A blind spot with a named substitute is a managed risk. A blind spot without one is a
    hole, and this file exists so the difference is legible without reading the YAML.
    """

    id: str
    defect: str
    why_ci_is_blind: str
    #: What WOULD catch it. `None` means nothing in this repository currently would.
    substitute: str | None
    substitute_status: str


BLIND_SPOTS: tuple[BlindSpot, ...] = (
    BlindSpot(
        id="runtime_device_loss_exits_zero",
        defect=(
            "The Vulkan device is lost mid-run; ORT re-executes the fused subgraph on the "
            "CPU EP; the process finishes its remaining work as CPU output and exits 0."
        ),
        why_ci_is_blind=(
            "Every other gate we have keys on something a failure changes: an exit code, a "
            "raised exception, a verdict token. A runtime device loss changes none of "
            "them. get_providers() still lists the EP; the harness still writes an "
            "artifact; the exit status is still 0. What changes is that the run is "
            "SHORTER, and a shorter run does not read as a failure — it reads as a "
            "smaller number. Tank's two ctx-512 points were truncated by exactly this and "
            "differencing them yielded an apparent 6.7% KV saving that was an observation "
            "ending early. Trinity's disable_cpu_ep_fallback cannot see it either: her "
            "flag makes ORT refuse at SESSION CREATION on node assignment, and a device "
            "loss happens long after that, on a session ORT has already accepted."
        ),
        substitute=(
            "ci/check_device_loss.py, on two signals neither of which is an exit code: "
            "the EP's own device-lost text, which is Vulkan specification language and so "
            "stable across vendors and versions; and a structural rule that compares what "
            "an artifact DECLARED it would observe (iters) against what it observed "
            "(compute_calls), plus uploads == readbacks + 1, an inference caught in "
            "flight. The structural rule needs no text at all and survives any log-format "
            "change. Falsified 2026-08-02 by 18 arms including a LIVE catch: a second, "
            "earlier device loss in bench/results/trinity-suite-dev1.log (2026-07-31, "
            "Intel, vkQueueSubmit) that had been read as a test failure and never "
            "reported as a lost device. Re-falsified 2026-08-07 (issue #24) by 27 arms, "
            "which now include the exclusion list itself: an entry that accounts for no "
            "finding, or for fewer findings than its file carries, is red."
        ),
        substitute_status=DEMONSTRATED,
    ),
    BlindSpot(
        id="timestamp_period_52x",
        defect=(
            "A build that drops the timestampPeriod conversion, using raw GPU ticks as "
            "nanoseconds."
        ),
        why_ci_is_blind=(
            "lavapipe reports timestampPeriod = 1.0, exactly as NVIDIA does. At a period "
            "of 1.0 the conversion is the IDENTITY, so a build that omits it is "
            "numerically indistinguishable from a build that performs it — green on "
            "lavapipe, green on NVIDIA, and under-reporting Intel Iris Xe by 52.0833x. No "
            "amount of executing on the CI device can see this, because the CI device "
            "cannot tell the two builds apart. This is the anti-correlated shape again: "
            "the defect is invisible precisely where we look hardest."
        ),
        substitute=(
            "rust/src/trace.rs unit tests, which construct a SYNTHETIC calibration with a "
            "non-unit period (cal(40.0, 64).ticks_to_ns(100, 1100) == 40_000.0) and "
            "therefore do not depend on the host device at all. Falsified on 2026-08-01: "
            "dropping both conversions turns four of them red, naming the 52x case "
            "explicitly. Wired into both device lanes as `cargo test --lib` on the same "
            "day — before which the crate's 440 unit tests had NEVER run in CI."
        ),
        substitute_status=DEMONSTRATED,
    ),
    BlindSpot(
        id="timestamp_valid_bits_36",
        defect=(
            "Failing to mask raw timestamps to the queue's valid-bit width, or "
            "mishandling counter wrap."
        ),
        why_ci_is_blind=(
            "lavapipe reports 64 valid bits, as NVIDIA does. Intel Iris Xe reports 36, "
            "which wraps roughly hourly. A 64-bit assumption is correct on every device "
            "CI can reach and wrong on the one device we own that is a spec-conformance "
            "oracle."
        ),
        substitute=(
            "The same trace.rs unit tests, with synthetic 36-bit calibrations. "
            "undefined_upper_bits_on_a_thirty_six_bit_counter_are_masked_away and "
            "an_intel_counter_wrap_does_not_produce_a_negative_or_absurd_duration both "
            "went red under the 2026-08-01 mutation."
        ),
        substitute_status=DEMONSTRATED,
    ),
    BlindSpot(
        id="census_denominator",
        defect=(
            "A wiring census that is complete by construction: twelve mechanisms out of "
            "twelve, where the twelve in the denominator is the same list that produced "
            "the numerator. It reads as coverage and can never go red."
        ),
        why_ci_is_blind=(
            "No amount of running the census can see this. The census is the numerator; "
            "asking it for the denominator is asking a decomposition to certify itself, "
            "which is R11's hardest kind of wrong and the shape criterion 11 was refused "
            "on. Nor can a device lane see it: every mechanism the census enumerates does "
            "run, on both devices, so the lane is green and the question is never asked."
        ),
        substitute=(
            "ci/check_census_completeness.py enumerates the whole from production Rust "
            "the census does not write — C ABI counter fields, trace.rs Phase variants "
            "and ONNXRUNTIME_EP_VULKAN_* switches. Numerator and denominator then have "
            "different authors in different files in a different language, so the count "
            "can be wrong, which is the only reason it can be evidence. Falsified "
            "2026-08-02: a counter field, a trace phase and an env switch planted in a "
            "scratch copy of rust/src are each named by the screen. The measured answer "
            "is not 12/12: it is 50 surfaces, 33 censused, 12 instrumented and observed "
            "by nothing."
        ),
        substitute_status=DEMONSTRATED,
    ),
    BlindSpot(
        id="device_clock_state",
        defect=(
            "A timing figure taken while the board sat at its idle clock — the finding "
            "that produced obligation 8, where the wrong figure carried the better RSD."
        ),
        why_ci_is_blind=(
            "There is no GPU in any lane, therefore no clock, therefore no device-state "
            "record is possible. Absence of telemetry is not a waiver: obligation 8 "
            "amendment 2 exists precisely because the cheapest pass would otherwise be a "
            "platform with no instrumentation, and a CI runner with no GPU telemetry is "
            "that loophole at scale."
        ),
        substitute=None,
        substitute_status=IMPOSSIBLE_HERE,
    ),
    BlindSpot(
        id="concurrency_and_barriers",
        defect="A missing or mis-scoped pipeline barrier between dependent dispatches.",
        why_ci_is_blind=(
            "A software rasteriser does not execute dispatches concurrently the way "
            "silicon does, so the hazard a barrier exists to prevent frequently cannot "
            "occur there. The lane's barrier-parity tests check that the EXPECTED "
            "barriers were recorded — a structural claim — not that omitting one produces "
            "a wrong answer, which on lavapipe it often would not."
        ),
        substitute=(
            "Barrier-count parity against a recorded expectation (in-lane, structural) "
            "plus execution on real hardware, which only happens on developer machines "
            "and is therefore not reproducible evidence."
        ),
        substitute_status=UNDEMONSTRATED,
    ),
    BlindSpot(
        id="vendor_driver_behaviour",
        defect=(
            "Anything a real vendor driver does differently: UMA vs discrete memory "
            "placement, subgroup width, fp16/int8 arithmetic, ReBAR heap reporting, "
            "driver-specific validation errors."
        ),
        why_ci_is_blind=(
            "Every lane runs the same software rasteriser. The CI matrix has one device "
            "in it wearing two operating systems, which is a portability result about "
            "OUR code and not a hardware coverage result. OQ-12's ~32.67% as of "
            "2026-07-30 is simultaneously a ceiling and a floor for this reason."
        ),
        substitute=(
            "Local-dev runs on RTX 4060 and Iris Xe, which are observations rather than "
            "reproducible evidence, and a GPU runner, which the project does not have."
        ),
        substitute_status=IMPOSSIBLE_HERE,
    ),
    BlindSpot(
        id="ledger_device_provenance",
        defect=(
            "A proven form claimed on a device no proof run ever touched. Every one of "
            "the 74 ledger entries records `\"device\": \"device0\"` — the RTX 4060 — and "
            "the verification predicate is the shader digest alone. `device` is written "
            "into the record and never read back."
        ),
        why_ci_is_blind=(
            "The demonstration is one command: run the harness probe on Windows with "
            "ONNXRUNTIME_EP_VULKAN_DEVICE=1 (Intel Iris Xe). The EP claims the form and "
            "prints its own provenance while doing it — 'proven by "
            "evidence/cases/matmulnbits_f16_scales.onnx ON DEVICE0'. The banner states "
            "that the proof came from a different device and claims the form anyway. No "
            "lane can see this, because no lane compares the two. Not one form in this "
            "project has ever been proven on Iris Xe, and not one on lavapipe.\n\n"
            "The digest constrains the wrong axis. It is a hash of the embedded SPIR-V "
            "bytes, so it moves when the glslc that compiled them changes — which is why "
            "every entry faults on Linux, where nothing about the kernel differs — and it "
            "does not move when the device changes, which is the only axis a correctness "
            "proof about a GPU kernel actually varies along. So the ledger over-constrains "
            "the build machine and under-constrains the device, and both halves are "
            "failures of the same design fact rather than two separate bugs."
        ),
        substitute=(
            "NONE. ci/check_ledger_portability.py covers only the loud half — the digest "
            "disagreeing — which is the half that fails safe: the form is declined and "
            "the work goes to the CPU EP, which is always right. The silent half is the "
            "dangerous one and nothing watches it: digest agrees, device never proven, "
            "form claimed. Closing it needs the ledger's predicate to include the device "
            "(making every entry per-device and demanding Linux/lavapipe proof runs of "
            "its own), or an explicit written statement that shader-level correctness is "
            "held to be device-independent — which would be a claim about every driver we "
            "have never run on. That is a design decision for Morpheus and Mouse, not a "
            "screen for me to add. Recorded here so the choice is made rather than "
            "inherited.\n\n"
            "Do NOT resolve it with `gen_proof_ledger.py --reprove` on each platform. "
            "That makes the digest a per-machine build fingerprint, deletes the one thing "
            "it does do (a kernel edit invalidating its proofs), and the flag's own "
            "destructive default rewrote the ledger from 74 entries to 1 while printing "
            "PASS."
        ),
        substitute_status=UNDEMONSTRATED,
    ),
    BlindSpot(
        id="conversion_call_sites",
        defect=(
            "A conversion that is correct in isolation and never invoked at the real call "
            "site — the mirror image of the 52x defect, and not covered by the fix for it."
        ),
        why_ci_is_blind=(
            "Unit tests prove the arithmetic. Execution on lavapipe would prove the call "
            "site, but lavapipe's period of 1.0 makes a dropped call indistinguishable "
            "from a performed one, which is where this started. Neither of the two "
            "instrument families this project runs — unit tests and device lanes — can "
            "see this class, which is what made it worth a third."
        ),
        substitute=(
            "ci/check_tick_conversions.py, a STATIC source screen, added 2026-08-01. It "
            "decides the question from source text, where the period's value is "
            "irrelevant, so the identity-at-1.0 problem that blinds every device lane does "
            "not apply to it. Three rules: tick arithmetic only inside the sanctioned "
            "converters; raw ticks enter the program at exactly one site "
            "(TimestampPool::read_results) whose enclosing function builds the "
            "calibration; and every exemption in ci/tick_conversion_allowlist.json still "
            "matches a live line. Falsified on 2026-08-01 by "
            "ci/negative_control_tick_conversions.py, which injects the raw delta, the "
            "same defect laundered through a rename, a by-hand period multiply that skips "
            "the valid-bit mask, and a second reader of raw ticks — into a scratch COPY of "
            "rust/src, never the tree — and demands the screen go red quoting the line, on "
            "all four. Its residual is named rather than closed: it decides from names and "
            "cannot follow a value past the first rename, and it would pass over a wrong "
            "formula inside a sanctioned converter."
        ),
        substitute_status=DEMONSTRATED,
    ),
    BlindSpot(
        id="composed_workflow",
        defect=(
            "A defect that exists in no branch and only in their union: two correct "
            "merges composing into a broken whole."
        ),
        why_ci_is_blind=(
            "Every lane verifies the branch it is handed. On 2026-08-02 this shape "
            "occurred FIVE times in one day, across four subsystems and three languages — "
            "a lock that was correct until wiring an instrument grew the population "
            "needing it; two device_state.py fighting over sys.modules, each file correct; "
            "a signature and its caller correct on their own branches; new code "
            "reintroducing a class clippy had just cleared; and my own lane inventory, "
            "complete for the workflow I could see, meeting Switch's step, complete on the "
            "branch he could see. Nobody did anything wrong locally in any of the five, "
            "and NO COMMAND ANY OF THOSE AUTHORS COULD HAVE RUN would have shown it. That "
            "is not five mistakes; it is one property of verifying branches instead of "
            "unions."
        ),
        substitute=(
            "Partial, and it covers my file only. ci/check_lane_inventory.py now takes "
            "--union-with <ref>: it classifies the UNION of step names from the working "
            "tree and from a reference, so a step that exists on either side and is "
            "classified on neither goes red BEFORE the merge rather than after. Wired into "
            "lane-checks with --union-required, because a reference it cannot read leaves "
            "it silently back in the branch-only view it exists to replace. "
            "FALSIFIED ON THE REAL EVENT, replayed: the two actual pre-merge blobs "
            "(.github/workflows/ci.yml at 0cd6c99 and at main) with the tautological entry "
            "removed from the inventory to reconstruct what I knew at the time — "
            "branch-only view GREEN, union view ['Tautological-assertion screen (no GPU, "
            "whole tree)'], exactly the step that broke at f4ed9ce. The inputs are the real "
            "ones; only the clock is wrong, so it is a replay rather than a live catch. "
            "Both polarities also hold on a synthesised two-branch repository, including "
            "the outage arm. It covers exactly one shape, workflow step names, in exactly "
            "one file. The other four instances of 2026-08-02 remain uncovered; a general "
            "union check is with Trinity."
        ),
        substitute_status=DEMONSTRATED,
    ),
)


# ──────────────────────────────────────────────────────────────────────────────
# Queries
# ──────────────────────────────────────────────────────────────────────────────


def checks_for_lane(lane: str) -> list[Check]:
    """Every check that runs in `lane`, including the ones marked 'both' and 'all'."""
    out = []
    for c in CHECKS:
        if c.lane == lane:
            out.append(c)
        elif c.lane == "all":
            out.append(c)
        elif c.lane == "both" and lane in (LANE_LINUX, LANE_WINDOWS):
            out.append(c)
    return out


def lane_classification(lane: str) -> tuple[str, str]:
    """`green` or `operational`, and the sentence that justifies it.

    A lane is `green` only when every check in it has a demonstrated failing arm or a
    recorded structural reason it cannot have one. One `UNDEMONSTRATED` check is enough to
    hold the whole lane at `operational`, because an undemonstrated check may be a constant
    and a lane containing a constant cannot be trusted to fail.
    """
    cs = checks_for_lane(lane)
    unproven = [c for c in cs if c.status == UNDEMONSTRATED]
    if not cs:
        return ("UNKNOWN", f"no checks recorded for {lane}")
    if unproven:
        names = ", ".join(c.id for c in unproven)
        return (
            "operational",
            f"{lane} runs, but these checks have never been observed to fail: {names}. "
            f"An undemonstrated check may be a constant.",
        )
    gaps = [c for c in cs if c.status in RECORDED_GAP_STATUSES]
    if gaps:
        names = ", ".join(c.id for c in gaps)
        return (
            "green (with recorded gaps)",
            f"{lane}: every check has a demonstrated failing arm except {names}, which "
            f"cannot have one here for a stated structural reason.",
        )
    return ("green", f"{lane}: every check has a demonstrated failing arm.")


def falsifier_census(lane: str) -> tuple[int, int, str]:
    """How many of a lane's failing arms were planted, and how many actually happened.

    Surfaced next to the lane verdict because it is the thing most likely to be
    over-read. `green` says somebody performed the mutation; it does not say the check
    has ever caught anything nobody wrote for it.
    """
    cs = [c for c in checks_for_lane(lane) if c.is_green()]
    planted = [c for c in cs if c.falsifier == FALSIFIER_PLANTED]
    observed = [c for c in cs if c.falsifier == FALSIFIER_OBSERVED]
    if not cs:
        return (0, 0, f"{lane} has no check with a failing arm at all.")
    note = (
        f"{lane}: {len(planted)} of {len(cs)} failing arms are PLANTED — the mutation was "
        f"written on purpose to make the check go red. That proves each check works on the "
        f"shape it was written for; it does not show the check is load-bearing."
    )
    if observed:
        note += (
            f" {len(observed)} {'is' if len(observed) == 1 else 'are'} OBSERVED (arm "
            f"produced by a defect nobody planted): "
            + ", ".join(c.id for c in observed)
            + "."
        )
    else:
        note += (
            " NONE are OBSERVED: no check in this lane has ever caught a defect nobody "
            "planted for it."
        )
    return (len(planted), len(observed), note)


def validate() -> list[str]:
    """Structural problems with the inventory itself. Empty list means well-formed."""
    problems: list[str] = []
    seen: set[str] = set()
    for c in CHECKS:
        if c.id in seen:
            problems.append(f"{c.id}: duplicate id")
        seen.add(c.id)
        if c.status not in ALL_STATUSES:
            problems.append(f"{c.id}: unknown status {c.status!r}")
        if c.status in GREEN_STATUSES and not c.arm_broken:
            problems.append(
                f"{c.id}: status {c.status} claims a failing arm but records no arm_broken. "
                f"A status is not evidence; the observed failure text is."
            )
        if c.status == DEMONSTRATED and not c.mutation:
            problems.append(
                f"{c.id}: DEMONSTRATED without a mutation. If the mutation is not written "
                f"down the demonstration cannot be repeated, and an unrepeatable "
                f"demonstration is a memory."
            )
        if c.status in GREEN_STATUSES and c.falsifier not in ALL_FALSIFIERS:
            problems.append(
                f"{c.id}: status {c.status} claims a failing arm but does not say whether "
                f"that arm was PLANTED or OBSERVED. A planted falsifier proves the check "
                f"works on the shape somebody wrote for it; it does not show the check is "
                f"load-bearing. Leaving the reader to guess which one they are reading is "
                f"how `green` comes to mean more than it earned."
            )
        if c.status == IMPOSSIBLE_HERE and not c.reason:
            problems.append(
                f"{c.id}: IMPOSSIBLE_HERE without a reason. That is UNDEMONSTRATED wearing "
                f"a better name."
            )
        if not c.misses:
            problems.append(
                f"{c.id}: no `misses` recorded. Every check misses something; a blank "
                f"column means nobody looked, not that nothing is missed."
            )
    for b in BLIND_SPOTS:
        if b.substitute is None and b.substitute_status not in (
            IMPOSSIBLE_HERE,
            UNDEMONSTRATED,
        ):
            problems.append(
                f"{b.id}: no substitute, but status {b.substitute_status} implies one"
            )
    return problems


def as_dict() -> dict:
    """The whole inventory as JSON-serialisable data, for the lane artifact."""
    return {
        "lanes": {
            name: {
                **meta,
                "classification": lane_classification(name)[0],
                "why": lane_classification(name)[1],
            }
            for name, meta in LANES.items()
        },
        "checks": [
            {
                "id": c.id,
                "lane": c.lane,
                "step": c.step,
                "watches": c.watches,
                "status": c.status,
                "falsifier": c.falsifier,
                "mutation": c.mutation,
                "arm_healthy": c.arm_healthy,
                "arm_broken": c.arm_broken,
                "observed": c.observed,
                "reason": c.reason,
                "misses": list(c.misses),
            }
            for c in CHECKS
        ],
        "blind_spots": [
            {
                "id": b.id,
                "defect": b.defect,
                "why_ci_is_blind": b.why_ci_is_blind,
                "substitute": b.substitute,
                "substitute_status": b.substitute_status,
            }
            for b in BLIND_SPOTS
        ],
        "problems": validate(),
    }


def render(lanes: Iterable[str] | None = None) -> str:
    """Human-readable inventory: what it runs, what it catches, what it silently misses."""
    lines: list[str] = []
    for lane in lanes or LANES:
        meta = LANES[lane]
        cls, why = lane_classification(lane)
        lines.append(f"## {lane}  [{cls}]")
        lines.append(f"   runner: {meta['runner']}")
        lines.append(f"   device: {meta['device']}")
        lines.append(f"   {meta['governing_fact']}")
        lines.append(f"   {why}")
        lines.append(f"   {falsifier_census(lane)[2]}")
        lines.append("")
        for c in checks_for_lane(lane):
            lines.append(f"   - {c.id}  [{c.status}]")
            lines.append(f"     step:    {c.step}")
            lines.append(f"     watches: {c.watches}")
            if c.arm_broken:
                lines.append(f"     red arm: {c.arm_broken}")
            if c.reason:
                lines.append(f"     why not: {c.reason}")
            for m in c.misses:
                lines.append(f"     MISSES:  {m}")
            lines.append("")
    lines.append("## Blind spots — defect classes NO lane catches")
    lines.append("")
    for b in BLIND_SPOTS:
        lines.append(f"   - {b.id}  [substitute: {b.substitute_status}]")
        lines.append(f"     defect: {b.defect}")
        lines.append(f"     blind:  {b.why_ci_is_blind}")
        lines.append(f"     instead: {b.substitute or 'NOTHING IN THIS REPOSITORY'}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(render())
    print(json.dumps({"problems": validate()}, indent=2))
