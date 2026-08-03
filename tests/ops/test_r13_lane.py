"""Falsification suite for the lane's R13 machinery (§10.0.1).

> **R13 — a check has at least three terminal states and must report them as three
> distinct tokens: `PASS`, `FAIL(condition)` — the condition it exists to detect — and
> `ERROR(instrument)`, in which the check did not reach its observation. A red that could
> mean either is not a signal. An instrument error is a lane failure of a different kind
> and never counts as a detection.**

THE SPECIMEN THIS FILE IS BUILT AROUND
======================================
Guard D contained a ``NameError`` and raised before it read a single profiling event.
The suite went from ``8 passed`` to ``5 failed``; the red matched the prediction that
motivated the change; it was reported to the team as the guard working.  It had crashed.

Two mechanisms answer that, and both are falsified here rather than described:

  1. ``conftest._classify_failure`` — sorts each red into ``FAIL(condition)`` or
     ``ERROR(instrument)`` **by exception type**, so the distinction is structural rather
     than a matter of how carefully anyone read the summary line.
  2. ``conftest._fallback_lane_failure`` — the second witness with a different failure
     mode.  A ``grep`` for the known-fatal ``Falling back`` line cannot ``NameError``, and
     a guard cannot be silenced by a log-format change; each covers the other's outage.

R9 discipline: a classifier that agrees with its author is not evidence.  Every claim
below is stated as a mutant that must be rejected, with a paired control proving the
protocol does not reject everything.

These tests need no GPU, no EP library and no model.
"""

from __future__ import annotations

import sys

import pytest

import conftest as lane


class _Crash:
    """Minimal stand-in for pytest's ``ExceptionChainRepr.reprcrash``."""

    def __init__(self, message: str) -> None:
        self.message = message


class _Repr:
    def __init__(self, message: str) -> None:
        self.reprcrash = _Crash(message)


class _Report:
    """A synthesised ``TestReport`` carrying only what the classifier reads."""

    def __init__(self, message: str, when: str = "call") -> None:
        self.when = when
        self.longrepr = _Repr(message)


# ---------------------------------------------------------------------------
# Claim: an instrument outage and a detection are different tokens
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("message", "expected", "why"),
    [
        (
            "AssertionError: [Guard D: fallback detected] VulkanExecutionProvider executed ZERO",
            "FAIL",
            "the guard read the profile and found zero Vulkan events — a detection",
        ),
        (
            "NameError: name 'pathlib' is not defined",
            "ERROR",
            "the historical defect: the guard raised at its first statement, having "
            "observed nothing",
        ),
        (
            "RuntimeError: [attribution instrument failure] Profiling trace not found",
            "ERROR",
            "the trace could not be read; there is no observation to be a finding about",
        ),
        (
            "Failed: [Device 1] VulkanExecutionProvider not in session.get_providers()",
            "FAIL",
            "pytest.fail is a deliberate condition failure",
        ),
        (
            "TypeError: EquivalenceVerdict.from_comparison() requires an ExecutionAttribution",
            "ERROR",
            "a harness misuse is an outage, not a finding about the EP",
        ),
        (
            "assert 3 == 1",
            "FAIL",
            "a bare rewritten assertion is still a condition failure",
        ),
    ],
)
def test_classifier_separates_detections_from_outages(message: str, expected: str, why: str) -> None:
    state, text = lane._classify_failure(_Report(message))
    assert state == expected, f"{message!r} should classify as {expected}: {why}"
    assert text, "R13 obligation 2 / second clause: the summary quotes text, not a count"
    assert text.startswith(message.split("\n")[0][:20])


def test_setup_failures_are_outages() -> None:
    """A fixture that blew up never reached the test body, so it observed nothing."""
    state, _ = lane._classify_failure(_Report("AssertionError: whatever", when="setup"))
    assert state == "ERROR"


def test_the_two_states_are_actually_reachable() -> None:
    """Paired control: a classifier that answered ``ERROR`` to everything would pass every
    outage test above and be useless.  Both states must occur."""
    states = {
        lane._classify_failure(_Report("AssertionError: x"))[0],
        lane._classify_failure(_Report("NameError: x"))[0],
    }
    assert states == {"FAIL", "ERROR"}


def test_a_two_state_classifier_fails_the_protocol() -> None:
    """Mutation control (R9): the protocol must reject a classifier with pytest's alphabet.

    ``lambda r: ("FAIL", ...)`` is exactly the vocabulary the lane had before R13 — one
    token for both a detection and an outage.  If the protocol below does not reject it,
    the tests above are agreement rather than evidence.
    """

    def two_state_classifier(report):
        return "FAIL", "failed"

    def protocol(classifier) -> None:
        detection, _ = classifier(_Report("AssertionError: found zero Vulkan events"))
        outage, _ = classifier(_Report("NameError: name 'pathlib' is not defined"))
        if detection == outage:
            raise AssertionError(
                "classifier gives one token to a detection and an outage — a red that "
                "could mean either is not a signal (R13)"
            )

    with pytest.raises(AssertionError):
        protocol(two_state_classifier)
    protocol(lane._classify_failure)  # the real one passes the same protocol


# ---------------------------------------------------------------------------
# Claim: the known-fatal log line is a lane failure, on a PASSING test
# ---------------------------------------------------------------------------

# Copied verbatim out of bench/results/ctx512_device_lost.txt, 2026-08-02, line wrap and
# all.  The previous value of this constant was our own paraphrase -- "Falling back to
# CPUExecutionProvider." -- and every arm below was green against it while the witness was
# structurally unable to see what ORT actually prints (a list repr).  Fiction testing
# fiction.  Do not "simplify" this string: its awkwardness is the evidence.
_FALLBACK_CAPTURE = (
    "2026-08-02 21:00:00 [I] session start\n"
    "EP Error: [ONNXRuntimeError] : 11 : EP_FAIL : Non-zero status code returned while "
    "running VulkanExecutionProvider_13948954645276092517_0 node.\n"
    "Name:'VulkanExecutionProvider_VulkanExecutionProvider_13948954645276092517_0_0' "
    "Status Message: vkWaitForFences failed using ['VulkanExecutionProvider',\n"
    "'CPUExecutionProvider']\n"
    "Falling back to ['CPUExecutionProvider'] and retrying.\n"
    "2026-08-02 21:00:03 [I] done\n"
)

#: The string the witness used to match: THIS repository's prose about what ORT prints.
#: It must NOT be a lane failure -- twelve of the twelve hits the old markers ever produced
#: across three real logs were this sentence, quoted back out of a captured suite log.
_OUR_OWN_PROSE_CAPTURE = (
    "E       ORT prints 'EP_FAIL ... Falling back to CPUExecutionProvider' during "
    "sess.run()\n"
)


def test_fallback_line_produces_a_lane_failure() -> None:
    message = lane._fallback_lane_failure(_FALLBACK_CAPTURE)
    assert message is not None
    assert "LANE FAILURE" in message
    assert "Falling back" in message
    assert "expects_ort_fallback" in message, (
        "the failure must tell a deliberate provoker how to opt out, or the witness will "
        "be deleted the first time it is inconvenient"
    )


def test_clean_capture_is_not_a_lane_failure() -> None:
    """Paired control: a witness that fires on everything is a witness that says nothing."""
    assert lane._fallback_lane_failure("everything was fine\nrun complete\n") is None
    assert lane._fallback_lane_failure("") is None


def test_our_own_prose_about_the_line_is_not_the_line() -> None:
    """The paired control that did not exist, and whose absence was the whole defect.

    A suite log routinely contains this repository's own description of what ORT prints,
    because our docstrings are echoed into it by pytest.  The old markers matched that
    description and nothing else: twelve hits across three real logs, all ours, zero from
    ORT.  So a witness that fires on our prose is not merely imprecise -- it produces
    confident positives that carry no information at all, which is worse than silence
    because it looks like corroboration.
    """
    assert lane._fallback_lane_failure(_OUR_OWN_PROSE_CAPTURE) is None


def test_the_witness_cannot_crash() -> None:
    """The whole value of the second witness is that its failure mode differs from a guard's.

    A guard can ``NameError``; a scan over a string must not be able to.  If this ever
    raises, the two witnesses share a failure mode and are one witness.
    """
    for hostile in ("", "\x00\x01\x02", "Falling back", "a" * 100_000, "\n" * 1000):
        lane._fallback_lane_failure(hostile)


def test_marker_is_registered(pytestconfig: pytest.Config) -> None:
    """The opt-out marker must be registered, or a deliberate provoker cannot opt out.

    A witness with no legitimate exemption is a witness that will be deleted the first
    time it is inconvenient, which is how three controls on this project died.
    """
    markers = "\n".join(pytestconfig.getini("markers"))
    assert "expects_ort_fallback" in markers


# ==============================================================================================
# R13 APPLIED TO SUBPROCESSES AND TO THE CRITERION-3 FRAME
# ==============================================================================================
# Two lane defects motivated everything below, and both were misread as findings:
#
#   * `test_wiring_census` shells out to `cargo test`.  Under four concurrent agents this
#     machine runs the same suite 4.4x slower (708 s vs 161 s, Niobe).  The 120 s budget
#     expired, `subprocess.TimeoutExpired` propagated, pytest scored a red, and the test had
#     to be `--ignore`d to get a clean run.  Nothing was observed; a detection was recorded.
#
#   * The criterion-3 planted-fence-leak control asserts `EP_VALIDATION_ERROR_COUNT > 0`.
#     On a machine with no validation layer that count CANNOT become non-zero for any state
#     of the EP.  The control then either fails (accusing Switch's messenger of a defect the
#     machine made impossible to observe) or skips (reporting green).  R12's answer is the
#     third one: UNOBSERVABLE.
#
# Everything here is pure: exit codes and strings in, tokens out.  No GPU, no cargo, no
# Vulkan, no clock.  A timing-dependent test of a contention defect would be its own joke.

import _verdict


# --- the budget ------------------------------------------------------------------------------

def test_the_budget_survives_the_measured_inflation() -> None:
    """4.4x is the measured worst case; the multiplier must exceed it, not match it."""
    assert _verdict.CONTENTION_INFLATION_FACTOR > 4.4, (
        "the inflation factor must have headroom over the 4.4x that has actually been "
        "measured on this host, or the next slightly worse day manufactures a red")
    assert _verdict.contention_tolerant_timeout(161, env={}) >= 708, (
        "the quiet 161 s suite took 708 s under load; a budget derived from the quiet "
        "number must still cover the loaded one")


def test_a_tiny_budget_is_still_floored() -> None:
    """Process start-up alone can lose tens of seconds on a contended Windows host."""
    assert _verdict.contention_tolerant_timeout(0.1, env={}) >= _verdict.CONTENTION_TIMEOUT_FLOOR_S


def test_the_scale_override_is_read_and_bad_values_are_ignored() -> None:
    assert _verdict.contention_tolerant_timeout(100, floor=0, env={"ONNXRUNTIME_EP_VULKAN_TIMEOUT_SCALE": "20"}) == 2000
    for junk in ("", "abc", "-3", "0"):
        assert _verdict.contention_tolerant_timeout(
            100, floor=0, env={"ONNXRUNTIME_EP_VULKAN_TIMEOUT_SCALE": junk}
        ) == 100 * _verdict.CONTENTION_INFLATION_FACTOR, (
            f"a junk override ({junk!r}) must fall back to the default, not to zero -- a zero "
            "budget times out instantly and every check becomes an outage")


def test_a_timeout_raises_instrument_error_and_says_it_is_not_a_detection() -> None:
    """THE MUTANT THIS REPLACES: a bare `subprocess.run(..., timeout=120)`, whose
    `TimeoutExpired` is an ordinary exception that pytest scores as a red."""
    with pytest.raises(_verdict.InstrumentError) as exc:
        _verdict.run_subprocess_checked(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            what="deliberate timeout",
            quiet_seconds=0.01,
            floor=0.5,
        )
    text = str(exc.value)
    assert "NOT A DETECTION" in text.upper()
    assert "ERROR(instrument)" in text
    assert not isinstance(exc.value, AssertionError), (
        "an InstrumentError must never be an AssertionError, or conftest's type-based "
        "classifier scores an outage as a finding")


def test_a_missing_command_is_an_outage_not_a_finding() -> None:
    with pytest.raises(_verdict.InstrumentError):
        _verdict.run_subprocess_checked(
            ["a-command-that-does-not-exist-on-any-machine-12345"],
            what="missing command", quiet_seconds=1,
        )


def test_the_wrapper_does_not_judge_the_exit_code() -> None:
    """PAIRED CONTROL. A wrapper that raised on every non-zero exit would pass every test
    above while destroying the caller's ability to state a condition at all."""
    done = _verdict.run_subprocess_checked(
        [sys.executable, "-c", "raise SystemExit(7)"],
        what="non-zero exit", quiet_seconds=5,
    )
    assert done.returncode == 7
    done_ok = _verdict.run_subprocess_checked(
        [sys.executable, "-c", "print('hi')"], what="clean exit", quiet_seconds=5,
    )
    assert done_ok.returncode == 0 and "hi" in done_ok.stdout


# --- the criterion-3 frame -------------------------------------------------------------------

@pytest.mark.parametrize(
    "code, out, expected",
    [
        (0, "VALIDATION ARMED", _verdict.VALIDATION_ARMED),
        (3, "VALIDATION LAYER ABSENT", _verdict.VALIDATION_LAYER_ABSENT),
        (3, "NO VULKAN LOADER: could not open vulkan-1.dll", _verdict.VALIDATION_NO_LOADER),
        (1, "something else went wrong", _verdict.VALIDATION_PROBE_ERROR),
        (None, "", _verdict.VALIDATION_PROBE_ERROR),
        (0, "", _verdict.VALIDATION_PROBE_ERROR),
    ],
)
def test_validation_probe_classification(code, out, expected) -> None:
    """The last row is the one that matters: exit 0 with no ARMED line is NOT armed.

    `probe_validation` returning 0 without printing its state would license every
    downstream claim on nothing at all -- the criterion-3 disease one level up.
    """
    state, reason = _verdict.classify_validation_probe(code, out)
    assert state == expected, reason
    assert reason, "every state must carry a reason a reader can act on"


def test_an_unarmed_frame_is_an_outage_and_an_armed_one_is_not() -> None:
    """BOTH POLARITIES. A gate that raised unconditionally would also reject every unarmed
    machine, and would additionally make criterion 3 unpassable."""
    _verdict.require_validation_armed(_verdict.VALIDATION_ARMED, "epctl reports ARMED")
    for bad in (_verdict.VALIDATION_LAYER_ABSENT, _verdict.VALIDATION_NO_LOADER,
                _verdict.VALIDATION_PROBE_ERROR):
        with pytest.raises(_verdict.InstrumentError) as exc:
            _verdict.require_validation_armed(bad, "synthesised")
        text = str(exc.value)
        assert "UNOBSERVABLE" in text, "R12: not 0, and not a pass"
        assert "NOT A FINDING" in text.upper()
        assert "NOT A PASS" in text.upper()


# --- the planted-violation control's three states --------------------------------------------

_PASSING_TRANSCRIPT = (
    "running 1 test\n"
    "[EP-PLANT] EP_VALIDATION_ERROR_COUNT after planted fence leak = 1\n"
    "test vk::dispatch_integration::ep_messenger_fires_for_planted_fence_leak ... ok\n"
)


@pytest.mark.parametrize(
    "code, out, expected, why",
    [
        (0, _PASSING_TRANSCRIPT, "PASS",
         "the transcript Switch actually recorded on the RTX 4060"),
        (101, "[EP-PLANT] EP_VALIDATION_ERROR_COUNT after planted fence leak = 0\n", "FAIL",
         "a count of 0 IS an observation -- the frame was checked armed first, so the event "
         "could have occurred and did not. This is the only genuine detection in the table."),
        (101, "error[E0433]: failed to resolve: use of undeclared crate\n", "ERROR",
         "a compile failure means the control never ran"),
        (0, "[SKIP] ep_messenger_fires_for_planted_fence_leak: no capable device\n", "ERROR",
         "R12: the plant's event cannot occur in this frame -- UNOBSERVABLE, not 0"),
        (101, "thread 'main' panicked at src/vk/instance.rs:1: something unrelated\n", "ERROR",
         "no artifact line at all: the control died before its observation. THIS is the "
         "branch a bare `assert returncode == 0` gets wrong -- it scores a crash as a "
         "detection, which is the Guard D NameError verbatim."),
        (101, _PASSING_TRANSCRIPT, "FAIL",
         "the plant fired but the run still failed: an observation was reached, so it is a "
         "condition failure, not an outage"),
        (None, "", "ERROR", "killed by a signal, no artifact"),
    ],
)
def test_plant_run_classification(code, out, expected, why) -> None:
    state, reason, _count = _verdict.classify_plant_run(code, out)
    assert state == expected, f"{why}\nclassifier said {state}: {reason}"
    assert reason


def test_plant_classification_reaches_all_three_states() -> None:
    """R13's own falsifier: a classifier that answered ERROR to everything would satisfy
    the timeout tests above, and one that answered FAIL to everything would satisfy the
    zero-count test. Both must be impossible."""
    states = {
        _verdict.classify_plant_run(0, _PASSING_TRANSCRIPT)[0],
        _verdict.classify_plant_run(
            101, "[EP-PLANT] EP_VALIDATION_ERROR_COUNT after planted fence leak = 0\n")[0],
        _verdict.classify_plant_run(101, "[SKIP] no ICD\n")[0],
    }
    assert states == {"PASS", "FAIL", "ERROR"}


def test_the_plant_artifact_pattern_is_the_one_the_rust_test_prints() -> None:
    """R10: the falsifier for "this parse is wired" is that its content varies with input.
    Byte-copied from rust/src/vk/dispatch_integration.rs:530."""
    for n in (0, 1, 7, 4096):
        line = f"[EP-PLANT] EP_VALIDATION_ERROR_COUNT after planted fence leak = {n}"
        assert _verdict.classify_plant_run(0, line)[2] == n
    assert _verdict.classify_plant_run(0, "EP_VALIDATION_ERROR_COUNT = 1")[2] is None, (
        "a near-miss line must NOT parse, or a rename on the Rust side silently makes this "
        "check unfalsifiable instead of loudly breaking it")


# ---------------------------------------------------------------------------
# XPASS(strict) is a fourth token, and it is the one most likely to be thrown away
#
# Found on a real run: `test_gqa_present_kv_shape[0]` xfail(strict)-PASSED on Intel and the
# classifier filed it under ERROR(instrument) -- the bucket labelled "none of these is
# evidence about the EP". It is evidence about the EP: the condition the xfail recorded had
# been fixed. R13's third corollary is that a result contradicting a prediction deserves
# MORE scrutiny than one confirming it, and burying somebody else's good news in the
# outage bucket is how it gets none.
# ---------------------------------------------------------------------------


def test_an_xpass_is_neither_a_detection_nor_an_outage() -> None:
    state, text = lane._classify_failure(
        _Report("[XPASS(strict)] GQA Compute path: absent optional inputs produce size=0 "
                "alloc requests; EP falls back to CPU. Owner: Switch.")
    )
    assert state == "XPASS", (
        "an xfail(strict) that passed reached its observation, so it is not an outage; and "
        "the observation is that a recorded expectation is stale, so it is not a detection "
        f"about the EP's current behaviour either. Got {state}: {text}")


def test_the_four_tokens_are_all_reachable() -> None:
    """PAIRED CONTROL for the test above. A classifier that answered XPASS to everything
    would satisfy it while destroying the other three states."""
    states = {
        lane._classify_failure(_Report("AssertionError: outputs disagree"))[0],
        lane._classify_failure(_Report("NameError: name 'ep_name' is not defined"))[0],
        lane._classify_failure(_Report("[XPASS(strict)] somebody fixed it"))[0],
        lane._classify_failure(_Report("AssertionError: x", when="setup"))[0],
    }
    assert states == {"FAIL", "ERROR", "XPASS"}, (
        f"three distinct classifications must be reachable from four reports; got {states}")
