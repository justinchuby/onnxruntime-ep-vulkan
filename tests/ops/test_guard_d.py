"""Two-polarity self-test for Guard D (assert_vulkan_executed_runtime).

R10 RATIONALE
=============
A guard that is never falsified is indistinguishable from one that never fires.
``assert_vulkan_executed_runtime`` is the instrument we rely on to detect runtime
fallback; if it has a defect (e.g. a missing import, wrong comparison, always-passes
body), every correctness gate it protects becomes vacuous simultaneously.

This file applies the same paired-control discipline used for criteria 4 & 5:
  - **Negative polarity**: a profile containing zero VulkanEP Node events MUST cause
    Guard D to raise ``AssertionError`` ("fallback detected").
  - **Positive polarity**: a profile containing ≥ 1 VulkanEP Node event MUST cause
    Guard D to return the event count without raising.

These tests run without the Vulkan EP, without a real model, and without a GPU.
They are **always in the lane** — they are not ``@pytest.mark.slow``, not skipped when
the EP lib is absent, and do not require ``require_vulkan``.

FAILURE MODE COVERAGE
=====================
Three distinct outcomes are tested:
  1. Zero Vulkan events → ``AssertionError`` (fallback detected)
  2. Non-zero Vulkan events → returns count (guard passes)
  3. Missing trace file → ``RuntimeError`` (instrument failure, not a fallback finding)

This directly answers the coordinator's concern (2026-07-31T03:23:12-07:00):
  "a guard that raises NameError looks identical to a guard that failed correctly,
   and no one — including me — could tell."
The three outcomes are now empirically distinct within the same test run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import _models as m

# ---------------------------------------------------------------------------
# Synthetic profiling-trace builders
# ---------------------------------------------------------------------------

def _write_profile(events: list[dict], path: Path) -> None:
    """Write *events* as a JSON array to *path* (ORT profiling trace format)."""
    with open(path, "w") as fh:
        json.dump(events, fh)


def _node_event(provider: str, op_name: str = "SomeOp") -> dict:
    """Minimal ORT Node profiling event for *provider*."""
    return {
        "cat": "Node",
        "name": op_name,
        "ph": "X",
        "ts": 0,
        "dur": 1,
        "pid": 0,
        "tid": 0,
        "args": {
            "provider": provider,
            "op_name": op_name,
        },
    }


def _non_node_event() -> dict:
    """A profiling event that is NOT a Node event (should be ignored by Guard D)."""
    return {"cat": "Session", "name": "model_loading_array_unsafe_options", "ph": "X", "ts": 0, "dur": 1, "pid": 0, "tid": 0}


# ---------------------------------------------------------------------------
# Test: negative polarity — zero Vulkan events → AssertionError
# ---------------------------------------------------------------------------

def test_guard_d_rejects_zero_vulkan_events(tmp_path: Path) -> None:
    """Guard D must raise AssertionError when the trace contains zero Vulkan Node events.

    This is the negative-polarity control: a runtime fallback to CPU produces a trace
    with only CPUExecutionProvider events.  Guard D is the instrument that catches this.
    If Guard D does NOT raise here, it cannot catch a real fallback either.

    Polarity: REJECT (AssertionError)
    """
    trace_path = tmp_path / "profile_cpu_only.json"
    _write_profile(
        [
            _non_node_event(),                             # session-level event, not a node
            _node_event("CPUExecutionProvider", "Add"),    # CPU fallback ran Add
            _node_event("CPUExecutionProvider", "MatMul"), # CPU fallback ran MatMul
        ],
        trace_path,
    )

    with pytest.raises(AssertionError) as exc_info:
        m.assert_vulkan_executed_runtime(str(trace_path))

    msg = str(exc_info.value)
    # Must say "fallback detected" (not a generic crash) and name the EP.
    assert "fallback detected" in msg or "ZERO" in msg, (
        f"AssertionError message does not identify the failure as a fallback: {msg!r}"
    )
    # Must name what DID execute, to help triage.
    assert "CPUExecutionProvider" in msg, (
        f"AssertionError message does not name the provider(s) that ran: {msg!r}"
    )
    # File must be deleted after the call (Guard D owns cleanup).
    assert not trace_path.exists(), "Guard D must delete the profile file after reading"


# ---------------------------------------------------------------------------
# Test: positive polarity — Vulkan event present → passes, returns count
# ---------------------------------------------------------------------------

def test_guard_d_accepts_vulkan_events(tmp_path: Path) -> None:
    """Guard D must return the fused-island count when Vulkan Node events are present.

    This is the positive-polarity control: a real GPU run produces at least one
    VulkanExecutionProvider Node event.  Guard D must pass without raising.

    Note: the count reflects *fused island* executions, not individual graph nodes.
    One fused island covering 354 graph nodes appears as exactly 1 Node event.
    A count of 1 is healthy; it is not evidence that only one graph node ran.

    Polarity: ACCEPT (returns count ≥ 1)
    """
    trace_path = tmp_path / "profile_vulkan_ran.json"
    _write_profile(
        [
            _non_node_event(),
            _node_event("CPUExecutionProvider", "Gather"),     # a declined op on CPU
            _node_event(m.EP_NAME, "VulkanExecutionProvider_abc123_0"),  # the fused island
        ],
        trace_path,
    )

    count = m.assert_vulkan_executed_runtime(str(trace_path))

    assert count == 1, (
        f"Expected Guard D to return 1 fused island (one VulkanEP Node event), got {count}"
    )
    # File must be deleted after the call.
    assert not trace_path.exists(), "Guard D must delete the profile file after reading"


def test_guard_d_counts_multiple_islands(tmp_path: Path) -> None:
    """Guard D must return the correct count when multiple fused islands executed.

    With Mouse's 1-island GQA partition a count of 1 is normal for Phi-3.5.
    But with earlier partitioning (161 islands) the count was 161.  Guard D must
    count correctly in both cases.
    """
    trace_path = tmp_path / "profile_multi_island.json"
    _write_profile(
        [
            _node_event(m.EP_NAME, f"VulkanExecutionProvider_abc123_{i}")
            for i in range(5)
        ],
        trace_path,
    )

    count = m.assert_vulkan_executed_runtime(str(trace_path))
    assert count == 5, f"Expected 5 fused islands, got {count}"
    assert not trace_path.exists()


# ---------------------------------------------------------------------------
# Test: missing file → RuntimeError (instrument failure, not a fallback finding)
# ---------------------------------------------------------------------------

def test_guard_d_raises_runtime_error_on_missing_file(tmp_path: Path) -> None:
    """Guard D must raise RuntimeError when the trace file does not exist.

    This distinguishes 'guard is broken' from 'fallback was detected':
      - RuntimeError means the harness could not read the trace (instrument failure).
        Route to harness maintainer; do NOT treat as an EP bug.
      - AssertionError means the trace was read but Vulkan ran nothing (real finding).
        Route to Switch / Mouse.

    A guard that raises NameError (or any other uncaught exception) is invisible to
    callers that only catch AssertionError.  RuntimeError is explicit and documented.
    """
    missing_path = tmp_path / "nonexistent_profile.json"
    assert not missing_path.exists()

    with pytest.raises(RuntimeError) as exc_info:
        m.assert_vulkan_executed_runtime(str(missing_path))

    msg = str(exc_info.value)
    assert "instrument failure" in msg or "not found" in msg.lower(), (
        f"RuntimeError message does not identify an instrument failure: {msg!r}"
    )


# ---------------------------------------------------------------------------
# Test: corrupted JSON → RuntimeError (instrument failure)
# ---------------------------------------------------------------------------

def test_guard_d_raises_runtime_error_on_corrupt_json(tmp_path: Path) -> None:
    """Guard D must raise RuntimeError when the trace file contains invalid JSON."""
    corrupt_path = tmp_path / "corrupt_profile.json"
    corrupt_path.write_text("{ not valid json !!!")

    with pytest.raises(RuntimeError) as exc_info:
        m.assert_vulkan_executed_runtime(str(corrupt_path))

    msg = str(exc_info.value)
    assert "instrument failure" in msg, (
        f"RuntimeError message does not identify an instrument failure: {msg!r}"
    )
    # File must still be deleted even on parse failure.
    assert not corrupt_path.exists(), "Guard D must delete the profile file even on JSON error"


# ---------------------------------------------------------------------------
# Test: count_vulkan_executions_from_profile unit check
# ---------------------------------------------------------------------------

def test_count_vulkan_executions_from_profile() -> None:
    """Unit check for the counting helper used by assert_vulkan_executed_runtime.

    Verifies: non-Node events ignored, CPU events ignored, Vulkan events counted.
    This is a pure-function test with no file I/O.
    """
    events = [
        {"cat": "Session", "name": "startup", "args": {}},              # not a node
        {"cat": "Node", "args": {"provider": "CPUExecutionProvider"}},  # CPU, not Vulkan
        {"cat": "Node", "args": {"provider": m.EP_NAME}},               # Vulkan #1
        {"cat": "Node", "args": {"provider": m.EP_NAME}},               # Vulkan #2
        {"cat": "Node"},                                                  # missing args
        {"cat": "Node", "args": "not-a-dict"},                           # bad args type
    ]
    count = m.count_vulkan_executions_from_profile(events)
    assert count == 2, f"Expected 2 Vulkan events, got {count}"


# ---------------------------------------------------------------------------
# Mutation controls: does the two-polarity protocol discriminate?
# ---------------------------------------------------------------------------
#
# R9: confidence scales with agreeing instruments; evidence scales only with
# falsifying ones.  The six tests above all pass against the real Guard D.  That is
# agreement, not evidence: a protocol that passes a correct guard tells us nothing
# until we have watched it FAIL a broken one.  The historical defect is the exact
# specimen — Guard D raised ``NameError`` at its first statement for its entire life,
# every test that called it went red, and the redness was read as "Guard D works".
#
# The protocol is therefore factored out and applied to deliberately broken guards.
# Each mutant below is a real failure mode we have shipped or could ship.

_MUTANT_MESSAGE = "the two-polarity protocol did not reject a guard that is broken"


def _apply_two_polarity_protocol(guard, tmp_path: Path, tag: str) -> None:
    """Run the two-polarity protocol against *guard*; raise AssertionError if it fails it.

    This is the same protocol the tests above apply to ``m.assert_vulkan_executed_runtime``,
    expressed once so it can be pointed at a mutant.  A guard passes iff:

      NEGATIVE polarity — a trace with zero VulkanEP Node events raises ``AssertionError``.
      POSITIVE polarity — a trace with one VulkanEP Node event returns ``1`` and does not raise.

    Anything else — no raise on the negative, a non-``AssertionError`` exception on either,
    a wrong count on the positive — is a failed protocol.
    """
    neg = tmp_path / f"{tag}_neg.json"
    _write_profile([_node_event("CPUExecutionProvider", "Add")], neg)
    try:
        guard(str(neg))
    except AssertionError:
        pass  # correct: fallback detected
    except BaseException as exc:  # noqa: BLE001
        raise AssertionError(
            f"NEGATIVE polarity: guard raised {type(exc).__name__}, not AssertionError. "
            f"A guard that crashes is not a guard that fired: {exc}"
        ) from None
    else:
        raise AssertionError(
            "NEGATIVE polarity: guard accepted a trace with zero VulkanEP Node events. "
            "It cannot detect a runtime fallback."
        )

    pos = tmp_path / f"{tag}_pos.json"
    _write_profile([_node_event(m.EP_NAME, "VulkanExecutionProvider_x_0")], pos)
    try:
        got = guard(str(pos))
    except BaseException as exc:  # noqa: BLE001
        raise AssertionError(
            f"POSITIVE polarity: guard raised {type(exc).__name__} on a trace that DOES "
            f"contain a VulkanEP Node event: {exc}"
        ) from None
    if got != 1:
        raise AssertionError(f"POSITIVE polarity: guard returned {got!r}, expected 1")


# --- the mutants ------------------------------------------------------------


def _mutant_nameerror(profile_path):
    """The actual historical defect: ``pathlib`` referenced but never imported.

    Fixed on main as 3ea42fd.  Before that commit Guard D raised ``NameError`` at its
    first statement, so it had never read a single profiling event in its life — and the
    suite going from 8 passed to 5 failed was read as the guard working.
    """
    path = pathlib.Path(profile_path)  # noqa: F821 — deliberately undefined
    return 1


def _mutant_always_passes(profile_path):
    """Returns a plausible count without reading anything. The vacuous-pass shape."""
    return 1


def _mutant_inverted(profile_path):
    """Polarity inverted: raises when Vulkan DID run, passes when it did not."""
    with open(profile_path) as fh:
        events = json.load(fh)
    count = m.count_vulkan_executions_from_profile(events)
    assert count == 0, "inverted"
    return 1


def _mutant_wrong_provider_key(profile_path):
    """Reads ``args['ep']`` instead of ``args['provider']`` — never matches, always 'fallback'.

    Passes the negative polarity for the wrong reason and fails the positive one.  This is
    the mutant that a negative-polarity-only test suite would certify as healthy: it is
    the reason both polarities are required, not just the interesting one.
    """
    with open(profile_path) as fh:
        events = json.load(fh)
    count = sum(
        1
        for e in events
        if e.get("cat") == "Node" and isinstance(e.get("args"), dict)
        and e["args"].get("ep") == m.EP_NAME
    )
    assert count > 0, "fallback detected"
    return count


@pytest.mark.parametrize(
    "mutant",
    [
        pytest.param(_mutant_nameerror, id="nameerror-the-historical-defect"),
        pytest.param(_mutant_always_passes, id="always-passes"),
        pytest.param(_mutant_inverted, id="polarity-inverted"),
        pytest.param(_mutant_wrong_provider_key, id="wrong-provider-key"),
    ],
)
def test_two_polarity_protocol_rejects_broken_guards(mutant, tmp_path: Path) -> None:
    """Each mutant guard MUST fail the two-polarity protocol.

    If this test passes, the protocol has been observed rejecting four distinct broken
    guards — including a byte-accurate reconstruction of the defect that shipped.  That is
    the falsifying observation R9 asks for; without it the six passing tests above are
    agreement between a guard and a test that was never shown capable of disagreeing.
    """
    with pytest.raises(AssertionError):
        _apply_two_polarity_protocol(mutant, tmp_path, tag=mutant.__name__)


def test_two_polarity_protocol_accepts_the_real_guard(tmp_path: Path) -> None:
    """The real Guard D MUST pass the same protocol the mutants fail.

    Paired control.  Without this the previous test would also pass if the protocol
    rejected everything unconditionally.
    """
    _apply_two_polarity_protocol(m.assert_vulkan_executed_runtime, tmp_path, tag="real")


def test_a_crashing_guard_is_not_an_assertion_error(tmp_path: Path) -> None:
    """A broken guard and a detected fallback must be distinguishable by exception type.

    This is the reader-level claim, stated as a test rather than as prose:

      ``AssertionError``  → a finding about the EP. Route to Switch/Mouse.
      anything else       → a finding about the guard. Route to the harness owner.

    On 2026-07-31 nobody could make that distinction unaided, because ``NameError`` and
    a correct fallback detection both presented as "the gate went red".
    """
    trace = tmp_path / "p.json"
    _write_profile([_node_event("CPUExecutionProvider", "Add")], trace)

    # The historical mutant raises NameError — NOT AssertionError.
    with pytest.raises(NameError):
        _mutant_nameerror(str(trace))

    # The real guard raises AssertionError on the same input.
    with pytest.raises(AssertionError):
        m.assert_vulkan_executed_runtime(str(trace))
