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

import threading
import time

import pytest

from _watchdog import (
    KIND_MECHANISM,
    KIND_TOOLCHAIN,
    STALLED,
    ClockDead,
    StallGuard,
    Stalled,
    WorkClock,
    guarded_call,
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
