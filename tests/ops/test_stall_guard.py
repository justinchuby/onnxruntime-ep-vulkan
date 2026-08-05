"""Falsifiers for the stall guard itself.

The guard's whole claim is that it separates *hang* from *contention*.  A demonstration
that only shows it passing under load proves nothing — it would also pass if it had been
deleted.  Switch's probes carry ``arms_must_differ`` for this reason and so does this file:
every property below is asserted in **both** polarities, and the crux test
(:func:`test_same_silence_does_not_trip_when_the_machine_is_slower`) asserts that the same
wall-clock silence produces *opposite* verdicts depending only on how much work the machine
got through meanwhile.  That is the property a bigger timeout cannot have.

These run in-process against a **fake** clock, so they are deterministic, take no
measurable time, and are always on.  The end-to-end four-cell demonstration against the
real census lives in ``tests/ops/probe_stall_guard.py``.
"""

from __future__ import annotations

import inspect
import pathlib
import shutil
import sys
import threading
import time

import pytest

import _watchdog
from _watchdog import (
    KIND_MECHANISM,
    KIND_TOOLCHAIN,
    STALLED,
    ClockDead,
    StallGuard,
    Stalled,
    WorkClock,
    filesystem_progress,
    guarded_call,
    guarded_run,
    injected_stall_target,
)


class FakeClock:
    """A work clock the test drives by hand.  Same surface as :class:`WorkClock`."""

    def __init__(self) -> None:
        self.units = 0
        self.alive = True
        self._dead_reason = ""

    def advance(self, n: int = 1) -> None:
        self.units += n

    def die(self, reason: str = "thread gone") -> None:
        self.alive = False
        self._dead_reason = reason

    def seconds_since_last_unit(self) -> float:
        return 0.0

    def recent_unit_cost_s(self) -> float:
        return 0.001

    def assert_alive(self, what: str) -> None:
        if not self.alive:
            raise ClockDead(
                f"[{what} instrument failure] ERROR(instrument): work clock dead "
                f"({self._dead_reason})"
            )


# ---------------------------------------------------------------------------
# arm A — contention must not trip it
# ---------------------------------------------------------------------------


def test_slow_but_progressing_never_trips():
    """Contention: work happens between beats, but beats keep arriving.

    Ten times the budget's worth of work passes in total.  A wall-clock timeout tuned to
    the budget fires here; the guard must not, because progress never stopped.
    """
    clock = FakeClock()
    guard = StallGuard(clock=clock, what="slow-but-alive")
    budget = 10

    for step in range(20):
        clock.advance(budget - 1)  # nearly the whole budget between beats, 20 times over
        guard.raise_if_stalled(budget_units=budget, kind=KIND_MECHANISM)
        guard.beat(f"step-{step}")

    assert clock.units > budget * 10, "the fixture must outlast a naive total-time budget"
    guard.raise_if_stalled(budget_units=budget, kind=KIND_MECHANISM)


# ---------------------------------------------------------------------------
# arm B — a hang must trip it
# ---------------------------------------------------------------------------


def test_silence_trips_it():
    clock = FakeClock()
    guard = StallGuard(clock=clock, what="hung-step")
    guard.beat("last-good-mechanism")

    clock.advance(11)
    with pytest.raises(Stalled) as caught:
        guard.raise_if_stalled(budget_units=10, kind=KIND_MECHANISM)

    report = caught.value.report
    assert report.last_beat == "last-good-mechanism", "the report must name where it died"
    assert report.silent_units > report.budget_units
    # R13: the text is the evidence, not the count.
    assert "NO_PROGRESS" in caught.value.args[0]
    assert "last progress : last-good-mechanism" in caught.value.args[0]


def test_boundary_is_strict_not_inclusive():
    """Exactly-at-budget is not a stall; one over is.  Both arms of the edge."""
    clock = FakeClock()
    guard = StallGuard(clock=clock, what="edge")
    guard.beat("x")

    clock.advance(10)
    guard.raise_if_stalled(budget_units=10, kind=KIND_MECHANISM)  # must not raise

    clock.advance(1)
    with pytest.raises(Stalled):
        guard.raise_if_stalled(budget_units=10, kind=KIND_MECHANISM)


# ---------------------------------------------------------------------------
# the crux — load-invariance, stated as a difference between two runs
# ---------------------------------------------------------------------------


def test_same_silence_does_not_trip_when_the_machine_is_slower():
    """The property no wall-clock threshold can have.

    Two runs.  Identical silence in **wall** terms (the fixture holds wall time fixed by
    never sleeping); different amounts of work completed by the machine during it.  The
    guard's verdict must differ, and it must differ in the direction that says "a slow
    machine is not a hung census".
    """
    budget = 10

    fast = FakeClock()
    fast_guard = StallGuard(clock=fast, what="quiet-machine")
    fast_guard.beat("mechanism-4")
    fast.advance(40)  # a quiet box gets through a lot of reference work while silent
    with pytest.raises(Stalled):
        fast_guard.raise_if_stalled(budget_units=budget, kind=KIND_MECHANISM)

    loaded = FakeClock()
    loaded_guard = StallGuard(clock=loaded, what="loaded-machine")
    loaded_guard.beat("mechanism-4")
    loaded.advance(4)  # same wall silence, but the box only managed a tenth of the work
    loaded_guard.raise_if_stalled(budget_units=budget, kind=KIND_MECHANISM)  # must not raise

    assert fast.units != loaded.units, "arms_must_differ: the two arms must not coincide"


def test_a_hang_on_a_loaded_machine_still_trips():
    """The cell that matters: fault and load together.

    A loaded machine ticks slowly, so this takes longer in wall time — but the budget is
    not in wall time, so the fault cannot hide behind the load.  Ten times slower simply
    means ten times as long before the same verdict, not no verdict.
    """
    clock = FakeClock()
    guard = StallGuard(clock=clock, what="hung-on-a-busy-box")
    guard.beat("mechanism-7")

    for _ in range(11):
        clock.advance(1)  # one unit at a time: the loaded box, crawling but alive
        try:
            guard.raise_if_stalled(budget_units=10, kind=KIND_MECHANISM)
        except Stalled as exc:
            assert "mechanism-7" in exc.args[0]
            break
    else:
        pytest.fail("a hang must still be detected on a loaded machine, only later")


# ---------------------------------------------------------------------------
# R13 — the instrument's own failures are never detections
# ---------------------------------------------------------------------------


def test_dead_clock_is_an_instrument_error_not_a_pass():
    """A watchdog that has stopped must not read as 'nothing to report'.

    Both arms: alive clock with silence -> detection; dead clock with the same silence ->
    ERROR(instrument).  If the dead clock silently passed, every future green would be
    unfalsifiable.
    """
    clock = FakeClock()
    guard = StallGuard(clock=clock, what="census")
    guard.beat("m")
    clock.advance(50)

    with pytest.raises(Stalled):
        guard.raise_if_stalled(budget_units=10, kind=KIND_MECHANISM)

    clock.die("thread killed")
    with pytest.raises(ClockDead) as caught:
        guard.raise_if_stalled(budget_units=10, kind=KIND_MECHANISM)
    assert "instrument failure" in caught.value.args[0]
    assert not isinstance(caught.value, Stalled), "an outage must not be a detection"


def test_stall_kind_selects_the_terminal_state():
    """A toolchain stall and a mechanism stall must not print the same verdict."""
    clock = FakeClock()

    def fire(kind: str) -> str:
        c = FakeClock()
        g = StallGuard(clock=c, what="w")
        g.beat("b")
        c.advance(11)
        with pytest.raises(Stalled) as caught:
            g.raise_if_stalled(budget_units=10, kind=kind)
        return caught.value.args[0]

    mechanism_text = fire(KIND_MECHANISM)
    toolchain_text = fire(KIND_TOOLCHAIN)
    assert mechanism_text != toolchain_text, "arms_must_differ"
    assert "FAIL(condition=NO_PROGRESS)" in mechanism_text
    assert "ERROR(instrument=NO_PROGRESS)" in toolchain_text
    assert clock.units == 0


def test_stalled_token_is_distinct_from_unwired():
    """'never came back' and 'ran and produced nothing' are different facts (R12)."""
    assert STALLED == "STALLED"
    assert STALLED != "UNWIRED"


# ---------------------------------------------------------------------------
# guarded_call — the wrapper, both arms
# ---------------------------------------------------------------------------


def test_guarded_call_returns_and_records_cost():
    clock = FakeClock()
    guard = StallGuard(clock=clock, what="wrapper")

    def work() -> int:
        clock.advance(3)
        return 42

    assert guarded_call(guard, "step-a", work, budget_units=100) == 42
    assert "step-a" in guard.costs, "every step must record what it cost in work units"


def test_guarded_call_propagates_the_step_error_unchanged():
    """A step that fails for its own reasons must not be reported as a stall."""
    clock = FakeClock()
    guard = StallGuard(clock=clock, what="wrapper")

    def boom():
        raise ValueError("the mechanism itself said no")

    with pytest.raises(ValueError, match="the mechanism itself said no"):
        guarded_call(guard, "step-b", boom, budget_units=100)


def test_guarded_call_abandons_a_hung_step():
    """The in-process arm: a step that never returns must not hold the suite forever."""
    clock = FakeClock()
    release = threading.Event()
    ticking = threading.Event()

    def ticker() -> None:
        # Stands in for the real work clock: the machine keeps working while the step does
        # not.  That disagreement is the entire signal.
        while not release.is_set():
            clock.advance(1)
            ticking.set()
            time.sleep(0.001)

    pump = threading.Thread(target=ticker, daemon=True)
    pump.start()
    ticking.wait(5)

    guard = StallGuard(clock=clock, what="hung-in-process")
    try:
        with pytest.raises(Stalled) as caught:
            guarded_call(
                guard,
                "never-returns",
                lambda: release.wait(3600),
                budget_units=5,
                kind=KIND_MECHANISM,
            )
        assert "never-returns" in caught.value.args[0]
    finally:
        release.set()
        pump.join(timeout=5)


# ---------------------------------------------------------------------------
# the real clock — it must actually tick on this machine
# ---------------------------------------------------------------------------


def test_real_work_clock_ticks_and_stops():
    with WorkClock() as clock:
        assert clock.alive
        first = clock.units
        assert first > 0, "start() must not return before the clock has moved"
        deadline = time.perf_counter() + 30
        while clock.units <= first and time.perf_counter() < deadline:
            time.sleep(0.01)
        assert clock.units > first, "the work clock must advance on this machine"
    assert not clock.alive, "stop() must actually stop it"


def test_fault_injection_is_off_by_default():
    """A planted stall must never be mistakable for a suffered one, and must be opt-in."""
    assert injected_stall_target() == ""


# ---------------------------------------------------------------------------
# Round 27 — the cross-branch break, as a standing falsifier
# ---------------------------------------------------------------------------
# `_run_counters_child(*, inject, tag, guard)` had `guard` as a REQUIRED keyword-only
# argument.  `test_ledger_lookup_wired` landed in the same file from `squad/mouse` and
# called it without one.  Each branch was green alone; the union raised
# `TypeError: missing 1 required keyword-only argument: 'guard'`, which the R13 summary
# correctly refused to read as evidence about the EP.
#
# The repair is a default that PROVISIONS a guard rather than one that omits it.  Both
# halves need a falsifier, because each alone is satisfiable by the wrong fix:
#   - callable without a guard      -> satisfied by deleting the guard entirely
#   - the call is guarded           -> satisfied by keeping the argument required
# Only the pair pins the intended behaviour.

def _census_module():
    import importlib
    return importlib.import_module("test_wiring_census")


def test_counters_child_is_callable_without_a_guard():
    """Arm 1: a caller that does not know about the guard can still construct the call."""
    import inspect

    census = _census_module()
    sig = inspect.signature(census._run_counters_child)
    param = sig.parameters["guard"]
    assert param.default is None, (
        "guard must have a default, or a caller added on another branch breaks on a "
        "TypeError that no command either author could have run would have shown"
    )
    # Binding is the real property; the signature is how it is achieved.
    sig.bind(inject=False, tag="ledger")


def test_counters_child_without_a_guard_is_still_guarded(monkeypatch):
    """Arm 2: the default provisions a live guard; it never runs unguarded.

    `guard=None` meaning "run unguarded" would satisfy arm 1 and silently undo round 25 —
    every future caller opting out of stall detection with nothing to notice.
    """
    census = _census_module()
    seen = {}

    def _fake_guarded_run(cmd, *, guard, what, budget_units, **kw):
        seen["guard"] = guard
        seen["budget_units"] = budget_units
        # Liveness must be read HERE, inside the call: the ambient guard's clock is
        # stopped when the call returns, which is the point of it being ambient.
        seen["alive"] = guard.clock.alive
        guard.raise_if_stalled(budget_units=budget_units, kind=KIND_MECHANISM)
        raise RuntimeError("stop here: the child itself is not under test")

    monkeypatch.setattr(census._watchdog, "guarded_run", _fake_guarded_run)
    with pytest.raises(RuntimeError, match="stop here"):
        census._run_counters_child(inject=False, tag="ledger")

    guard = seen["guard"]
    assert isinstance(guard, StallGuard), "the default must build a guard, not omit one"
    assert seen["alive"], "the ambient guard's clock must be running, or it cannot fire"
    assert not guard.clock.alive, "and it must be stopped again, or the census leaks threads"
    assert seen["budget_units"] > 0, "an unbudgeted step is an unguarded step"


def test_unknown_step_gets_a_budget_rather_than_a_keyerror():
    """A label nobody put in the table must still be watched.

    `_BUDGET_UNITS[label]` raised `KeyError` for a tag added elsewhere — the same
    cross-branch shape one line down from the one that fired.  Generous is safe here
    because the unit is work: a loose budget detects a hang later in work, never not at
    all, and contention cannot stretch it.
    """
    census = _census_module()
    assert census._budget_for("counters_child_clean") == census._BUDGET_UNITS["counters_child_clean"]
    assert census._budget_for("a_step_from_some_other_branch") == census._DEFAULT_BUDGET_UNITS
    assert census._DEFAULT_BUDGET_UNITS > 0


# ---------------------------------------------------------------------------
# Round 38 — the toolchain progress witness, and its refusal arm
# ---------------------------------------------------------------------------
# `guarded_run`'s premise is "every line the child writes is a beat".  A cold `cargo`
# build of ONE large crate writes a single line and then nothing for minutes, so the
# premise fails on a healthy child and the census reported ERROR(instrument) against
# nothing (measured: 12015 silent units, isolated; the same command warm PASSED while
# taking LONGER in wall time, which is what proves the failing quantity was silence).
#
# The patch is a second beat source, not a bigger budget, and the difference is exactly
# what arm B below has to show: a child that writes nothing must still be caught while
# the witness is live.  A bigger budget could not pass arm B.

_SCRATCH = pathlib.Path(__file__).resolve().parents[2] / "bench" / "scratch" / "fsbeat"


def test_filesystem_progress_moves_only_when_something_is_written():
    _SCRATCH.mkdir(parents=True, exist_ok=True)
    d = _SCRATCH / "arm0"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir()
    before = filesystem_progress([d])
    assert filesystem_progress([d]) == before, "an idle directory must read the same twice"
    (d / "a.txt").write_text("x")
    assert filesystem_progress([d]) != before, "a written file must move the fingerprint"
    shutil.rmtree(d)


def test_the_witness_sees_the_depth_a_cold_rustc_actually_writes_at():
    """Depth 1 would have watched the one place a compiling rustc does not touch.

    Measured on a cold `cargo test --test layering` (311s): longest interval with no
    visible change was 195.5s at depth 1 and 18.6s at depth 3, because the churn is
    `target/debug/incremental/<hash>/s-*-working/`.  A shallow witness is not a stricter
    witness, it is a blind one — and a blind witness that never beats is indistinguishable
    from not having one, which is how the first version of this patch failed.
    """
    _SCRATCH.mkdir(parents=True, exist_ok=True)
    d = _SCRATCH / "depth"
    if d.exists():
        shutil.rmtree(d)
    deep = d / "incremental" / "crate-hash" / "s-working"
    deep.mkdir(parents=True)
    before = filesystem_progress([d])
    (deep / "x.o").write_bytes(b"0")
    assert filesystem_progress([d]) != before, (
        "a file three levels down must move the fingerprint"
    )
    assert filesystem_progress([d], depth=1) == filesystem_progress([d], depth=1)
    assert _watchdog._PROGRESS_SCAN_DEPTH >= 3
    shutil.rmtree(d)


def test_a_silent_child_that_is_writing_files_is_not_called_stalled():
    """Arm A — the healthy cold-build shape: no output, but visible production.

    Run as a PAIR, because either half alone is satisfiable by the wrong thing: the
    witness-off half proves the budget really is tight enough to fire on this child, and
    the witness-on half proves the witness is what kept it alive.  Without the first half
    "it passed" would be evidence about a generous budget and nothing else.
    """
    _SCRATCH.mkdir(parents=True, exist_ok=True)
    d = _SCRATCH / "armA"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir()
    child = (
        "import time,pathlib\n"
        f"p=pathlib.Path(r'{d}')\n"
        "for i in range(16):\n"
        "    time.sleep(0.3)\n"
        "    (p/f'f{i}.bin').write_bytes(b'0')\n"
    )

    def _budget(clock):
        u0 = clock.units
        time.sleep(1.0)
        return max(4, clock.units - u0)

    with WorkClock() as clock:
        guard = StallGuard(clock=clock, what="fs witness arm A (witness off)")
        with pytest.raises(Stalled):
            guarded_run(
                [sys.executable, "-c", child],
                guard=guard,
                what="fs witness arm A (witness off)",
                budget_units=_budget(clock),
                kind=KIND_TOOLCHAIN,
                label="silent_but_producing",
            )

    shutil.rmtree(d)
    d.mkdir()
    with WorkClock() as clock:
        guard = StallGuard(clock=clock, what="fs witness arm A")
        r = guarded_run(
            [sys.executable, "-c", child],
            guard=guard,
            what="fs witness arm A",
            budget_units=_budget(clock),
            kind=KIND_TOOLCHAIN,
            label="silent_but_producing",
            progress_paths=[d],
        )
    assert r.returncode == 0, "the same child must now be allowed to finish"
    shutil.rmtree(d)


def test_a_silent_child_that_writes_nothing_still_trips_with_the_witness_live():
    """Arm B — the refusal arm.  The witness must not be a way of never firing.

    Deliberately NOT run with the witness disabled: the point is that the patch leaves a
    genuinely wedged child exactly as catchable as it was.
    """
    _SCRATCH.mkdir(parents=True, exist_ok=True)
    d = _SCRATCH / "armB"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir()
    child = "import time,sys\ntime.sleep(600)\n"
    with WorkClock() as clock:
        guard = StallGuard(clock=clock, what="fs witness arm B")
        u0 = clock.units
        time.sleep(1.0)
        budget = max(4, clock.units - u0)
        with pytest.raises(Stalled) as ei:
            guarded_run(
                [sys.executable, "-c", child],
                guard=guard,
                what="fs witness arm B",
                budget_units=budget,
                kind=KIND_TOOLCHAIN,
                label="silent_and_wedged",
                progress_paths=[d],
            )
    assert ei.value.report.kind == KIND_TOOLCHAIN
    assert "no forward progress" in str(ei.value)
    shutil.rmtree(d)


def test_the_witness_is_not_consulted_for_mechanism_spans():
    """A mechanism stall is a DETECTION; a filesystem beat must not be able to mask one."""
    src = inspect.getsource(_watchdog.guarded_run)
    assert "kind == KIND_TOOLCHAIN" in src, (
        "the progress witness must be scoped to toolchain spans, or it can rescue a "
        "mechanism that produced no observation — which is the thing criterion 12 detects"
    )
