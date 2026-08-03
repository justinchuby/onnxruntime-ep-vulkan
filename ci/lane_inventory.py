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

#: Statuses that support calling the lane containing them `green` for that check.
GREEN_STATUSES = frozenset({DEMONSTRATED, RED_NOW})

#: Statuses that are an honest recorded gap rather than an unexamined one.
RECORDED_GAP_STATUSES = frozenset({IMPOSSIBLE_HERE})

ALL_STATUSES = frozenset({DEMONSTRATED, UNDEMONSTRATED, IMPOSSIBLE_HERE, RED_NOW})

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
            "varied. Twelve arms in "
            "ci/negative_control_census_completeness.py, all fired 2026-08-02."
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
            "ci/negative_control_device_loss.py, 14 arms, all fired 2026-08-02: red on "
            "Tank's real artifact (REPLAYED); red on trinity-suite-dev1.log, a second "
            "device loss from 2026-07-31 nobody had reported (LIVE — a file the screen "
            "was not written against); red on a synthesised iters=25/compute_calls=9 "
            "artifact with no log text at all; green on the same truncation once the "
            "producer has moved it under rejected_points; green on a clean log; and an "
            "instrument error, never a pass, when it is given nothing to read."
        ),
        arm_healthy=(
            "284 artifacts read across bench/results, 7 excluded by name as records of "
            "known incidents with reason/owner/date, 126 decidable by the structural rule"
        ),
        arm_broken=(
            "DEVICE-LOSS: FAIL(condition=device_lost_reported) quoting "
            "'[vulkan-ep] ERROR: vkQueueSubmit failed: The logical device has been lost' "
            "from bench/results/trinity-suite-dev1.log:3216"
        ),
        observed="2026-08-02",
        misses=(
            "Its structural rule needs the producer to declare what it expected. An "
            "artifact carrying no iters/compute_calls pair is UNOBSERVABLE to it, not "
            "clean — 158 of 284 artifacts were undecidable on the run above.",
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
            "missing file is itself a finding — reduce that risk and do not remove it.",
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
            "the event occurs in reality. 1 LIVE, 3 REPLAYED, 10 PLANTED on 2026-08-02."
        ),
        arm_healthy="all 14 arms fired 2026-08-02",
        arm_broken=(
            "a missing bench/results/ctx512_device_lost.txt is reported as an arm that "
            "DID NOT FIRE — an outage in the control, never a pass"
        ),
        observed="2026-08-02",
        misses=(
            "Ten of its fourteen arms are PLANTED. They prove each rule fires on an "
            "input built to make it fire; only the LIVE and REPLAYED arms evidence that "
            "the event occurs.",
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
        arm_healthy="all twelve arms fired 2026-08-02, each naming the surface it planted",
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
            "A flake lives here: counters::tests::a_pinned_authoritative_counter_reports_"
            "unobservable_and_never_zero failed once in three full runs on 2026-08-01, "
            "writing to a fixed path under a process-global env var while other tests run "
            "in parallel. Recorded, not masked, and not mine to fix.",
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
        ),
    ),
    # ── device lanes: execution ───────────────────────────────────────────────
    Check(
        id="device.op_correctness",
        lane="both",
        step="pytest tests/ops (lavapipe)",
        watches="Op outputs against a CPU-EP reference, bit-exact or within a stated tolerance.",
        status=UNDEMONSTRATED,
        mutation=(
            "NOT YET PERFORMED. What would close it: perturb one kernel's arithmetic — a "
            "changed constant in a shader, or an off-by-one in a dispatch's workgroup "
            "count — rebuild, and confirm the op suite reds with the failure attributed "
            "to that op. This needs a shader/kernel edit (Switch) and a test-suite "
            "expectation (Trinity), so it is named here rather than claimed. Until it is "
            "done, 'the op suite passes' is a candidate for evidence."
        ),
        arm_healthy="op suite green on lavapipe",
        observed="2026-07-31",
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
            "change. Falsified 2026-08-02 by 14 arms including a LIVE catch: a second, "
            "earlier device loss in bench/results/trinity-suite-dev1.log (2026-07-31, "
            "Intel, vkQueueSubmit) that had been read as a test failure and never "
            "reported as a lost device."
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
