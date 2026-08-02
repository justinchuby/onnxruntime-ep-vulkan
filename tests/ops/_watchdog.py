"""A stall detector whose budget is denominated in **work the machine actually did**.

Why this file exists
--------------------
`test_wiring_census` was excluded from the suite with ``--ignore`` because it timed out.
The obvious repair — multiply the timeout — is the repair R9 amendment 5 forbids:

  > ask which way a check moves when its subject is wrong.  **If it moves with the
  > reader's confidence, it cannot be repaired by tightening** and is demoted from gate to
  > precondition.

A wall-clock timeout moves the *same way* for both of the things it must separate.  A
loaded machine and a hung census both mean "the number has not arrived yet", so whichever
threshold you pick you are trading false reds against false greens along a single axis and
there is no value on that axis that is right.  Niobe measured this box at **4.4x** for the
suite (708 s loaded, 161 s quiet) and **9.5x** for the `record` step; a threshold that
survives 9.5x is ~10 minutes wide, and a census that hangs forever passes under it in the
sense that matters — nobody learns which mechanism stopped, only that the box was slow.

So the wall clock is not tightened here.  It is **removed from the gate** (R9 A5's own
prescription: demoted to a precondition) and replaced by two changes that are independent
of each other, each attacking a different half of the confusion.

1.  **The budget is on silence, not on duration.**
    A hang produces *no forward progress*.  Contention produces *slow forward progress*.
    Total elapsed time cannot tell those apart; time-since-the-last-observation can.
    Every mechanism resolved, and every line a watched subprocess writes, is a beat.

2.  **Silence is counted on a work clock, not a wall clock.**
    :class:`WorkClock` runs a background thread that performs a fixed, deterministic
    reference computation over and over.  Each completed computation advances the clock by
    one **work unit**.  Under contention a work unit costs more wall seconds, so a budget
    of *N* units automatically becomes *N x (however much slower this machine currently
    is)*.  The budget is written once and needs no per-machine tuning, no
    ``$TIMEOUT_SCALE``, and no guess about how many agents are running.

Which way it moves when its subject is wrong
--------------------------------------------
This is the property the design has to have, and it is the one thing worth checking:

* **machine loaded, census healthy** — beats keep arriving *and* the window widens.  Two
  independent reasons not to fire.  It does not fire.
* **machine quiet, census hung** — beats stop; the work clock runs at full speed; the
  window closes at its narrowest.  It fires, fast.
* **machine loaded, census hung** — beats stop; the work clock is slowed, so the window is
  wider in wall seconds.  **It still fires**, because the budget counts work done, and
  work is still being done — just not by the census.  Later in wall time, identical in
  work.

There is no combination in which the fault hides behind the load.  That is the difference
between this and a bigger number, and it is why the demonstration in
``probe_stall_guard.py`` has four cells rather than one.

What this does NOT claim
------------------------
The reference computation is CPU-bound (SHA-256 over a fixed block, which releases the
GIL).  It therefore tracks **CPU** contention exactly and tracks GPU, disk and network
contention only insofar as they correlate.  A census step that is blocked on a busy GPU
while the CPU is idle will consume more work units than the same step on a quiet GPU.
The budgets below carry margin for that, and
``bench/results/stall_guard_arms-*.json`` records the observed per-step unit costs under
both load conditions so the margin is auditable rather than asserted.

No duration in this module is ever compared to a threshold as evidence about the EP
(§10.0 obligation 8).  Wall seconds appear in exactly two places: the duty-cycle sleep,
and the liveness check on the watchdog thread itself — a check about the instrument, never
about the subject.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# The work clock
# ---------------------------------------------------------------------------

#: The reference computation.  256 KiB of SHA-256: large enough that `hashlib` releases
#: the GIL (so the clock keeps ticking while the census holds it), small enough that one
#: unit is a few milliseconds on any machine this project runs on.  Deterministic
#: instruction count — that is the whole point, it is the *unperturbable quantity*
#: (§10.0.4) against which everything else here is measured.
REFERENCE_BLOCK: bytes = bytes(range(256)) * 1024  # 256 KiB

#: Wall-clock sleep after each unit, as a multiple of the unit's own measured cost.
#: Chosen so the clock costs ~5% of one core and so that *the sleep scales with the
#: machine too* — a fixed sleep would make the clock only partially load-proportional.
DUTY_SLEEP_RATIO: float = 19.0

#: If the clock thread has produced no unit for this many wall seconds, the clock itself is
#: broken (thread dead, or a machine so wedged that nothing runs).  This is the only
#: absolute wall-clock number in the design and it guards the *instrument*, not the
#: subject: exceeding it is always ERROR(instrument) and never a finding.
CLOCK_LIVENESS_CEILING_S: float = 120.0


class WorkClock:
    """A monotonic counter of reference computations completed by this machine.

    Not a timer.  ``units`` answers "how much work has this machine got through since I
    started watching", which is the quantity a stall budget actually wants: it is
    invariant to how many other agents are running, because their contention shows up as
    fewer units per second, not as fewer units per unit of work.
    """

    def __init__(self, *, duty_sleep_ratio: float = DUTY_SLEEP_RATIO) -> None:
        self._units = 0
        self._duty = float(duty_sleep_ratio)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._started_at = 0.0
        self._last_unit_at = 0.0
        self._recent_unit_cost_s = 0.0

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> "WorkClock":
        if self._thread is not None:
            return self
        self._started_at = time.perf_counter()
        self._last_unit_at = self._started_at
        self._thread = threading.Thread(
            target=self._run, name="work-clock", daemon=True
        )
        self._thread.start()
        # Do not return until the clock has actually ticked once.  A guard that starts
        # measuring against a clock that has never moved cannot tell "no work happened"
        # from "the clock is not running", and those are opposite findings.
        deadline = time.perf_counter() + 30.0
        while self.units == 0 and time.perf_counter() < deadline:
            time.sleep(0.005)
        return self

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=5.0)

    def __enter__(self) -> "WorkClock":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -- the loop ----------------------------------------------------------
    def _run(self) -> None:
        block = REFERENCE_BLOCK
        while not self._stop.is_set():
            t0 = time.perf_counter()
            hashlib.sha256(block).digest()
            cost = time.perf_counter() - t0
            with self._lock:
                self._units += 1
                self._last_unit_at = time.perf_counter()
                self._recent_unit_cost_s = cost
            # Sleeping a multiple of the unit's own cost keeps the duty cycle fixed AND
            # keeps the whole tick proportional to machine speed, which is what makes a
            # budget in units behave like a budget in "however long that takes here".
            self._stop.wait(min(cost * self._duty, 1.0))

    # -- reading -----------------------------------------------------------
    @property
    def units(self) -> int:
        with self._lock:
            return self._units

    @property
    def alive(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def seconds_since_last_unit(self) -> float:
        with self._lock:
            return time.perf_counter() - self._last_unit_at

    def recent_unit_cost_s(self) -> float:
        """Reporting only.  Never a gate — it is a duration (§10.0 obligation 8)."""
        with self._lock:
            return self._recent_unit_cost_s

    def assert_alive(self, what: str) -> None:
        """Raise if the watchdog itself has stopped.  Who watches the watchman."""
        if not self.alive:
            raise ClockDead(
                f"[{what} instrument failure] ERROR(instrument): the work clock thread is "
                "not running, so 'no progress' and 'nothing was measured' are the same "
                "reading.  No stall verdict is available from this run."
            )
        if self.seconds_since_last_unit() > CLOCK_LIVENESS_CEILING_S:
            raise ClockDead(
                f"[{what} instrument failure] ERROR(instrument): the work clock has "
                f"produced no reference unit for over {CLOCK_LIVENESS_CEILING_S:.0f}s of "
                "wall time.  Either the thread is wedged or this machine is not running "
                "Python at all; in both cases the stall detector has no denominator and "
                "reports nothing about the census."
            )


class ClockDead(Exception):
    """The instrument died.  Always ERROR(instrument), never a detection."""


# ---------------------------------------------------------------------------
# Kinds — whose stall is it
# ---------------------------------------------------------------------------
#
# R13 forbids an instrument error from being read as a detection, so a stall has to say
# whose it is before it says anything else.  Two kinds, and they get different terminal
# states:
#
#   MECHANISM — the stalled operation IS a mechanism under census (an EP call, an ORT
#               session, a child that exists to exercise the EP).  A mechanism that never
#               returns has produced no observation, which is precisely what criterion 12
#               exists to detect.  This is a FAIL(condition).
#   TOOLCHAIN — the stalled operation is scaffolding the census needs but does not judge
#               (a cargo compile, a crate download, an interpreter start).  The census did
#               not reach its observation.  This is ERROR(instrument).
#
# Tank's ruling of 2026-08-01 stands untouched: *`subprocess.TimeoutExpired` in
# test_wiring_census.py is ERROR(instrument), never FAIL(condition), because the census is
# deterministic and byte-based, so a timeout is evidence about the box and none about the
# call graph.*  That reasoning is about a **wall-clock** timeout and it is correct about
# one: its firing depends on how loaded the box is, so its firing is evidence about load.
# A work-clock stall's firing does not depend on load — that is the property demonstrated
# in all four cells of `probe_stall_guard.py` — so the premise of the ruling is absent and
# the conclusion does not carry over.  `subprocess.TimeoutExpired` remains
# ERROR(instrument) wherever it still occurs.  Flip MECHANISM_STALL_IS_A_DETECTION to
# False to restore the old behaviour in one line if Tank disagrees.
KIND_MECHANISM = "MECHANISM"
KIND_TOOLCHAIN = "TOOLCHAIN"

MECHANISM_STALL_IS_A_DETECTION = True

#: Census line token for a mechanism that never returned.  Deliberately NOT `UNWIRED`:
#: "ran and produced nothing" and "never came back" are different facts and a census that
#: spells them the same way has lost the one it was built to report.
STALLED = "STALLED"


@dataclass
class StallReport:
    """Everything known at the moment the guard fired.  Quoted, never counted."""

    what: str
    kind: str
    budget_units: int
    silent_units: int
    last_beat: str
    beats_seen: int
    clock_units_total: int
    #: Reporting only.  Present so a reader can see the machine was alive; never compared
    #: to a threshold and never quoted as a performance figure (§10.0 obligation 8).
    wall_seconds_silent: float = 0.0

    def text(self) -> str:
        head = (
            f"[{self.what}] {'FAIL(condition=NO_PROGRESS)' if self.kind == KIND_MECHANISM and MECHANISM_STALL_IS_A_DETECTION else 'ERROR(instrument=NO_PROGRESS)'}: "
            f"no forward progress for {self.silent_units} work units "
            f"(budget {self.budget_units}); this machine completed "
            f"{self.clock_units_total} reference units in total during the watch, so it "
            "was not idle and it was not merely slow — it was doing everything except "
            "this."
        )
        body = (
            f"\n  kind          : {self.kind}"
            f"\n  last progress : {self.last_beat}"
            f"\n  beats seen    : {self.beats_seen}"
            f"\n  why this is not a contention artefact: the budget is denominated in "
            "reference computations completed by THIS machine during THIS watch, not in "
            "seconds.  Contention lowers units-per-second, which widens the window in "
            "wall time and cannot on its own exhaust it.  Exhausting it requires the "
            "watched operation to produce nothing while the machine produces something."
        )
        return head + body


class Stalled(Exception):
    """Raised by the guard.  Carries the report; the caller decides the terminal state."""

    def __init__(self, report: StallReport) -> None:
        super().__init__(report.text())
        self.report = report


@dataclass
class StallGuard:
    """Watches a work clock and a stream of beats; fires only when the two disagree."""

    clock: WorkClock
    what: str = "guarded operation"
    _last_beat_units: int = field(default=0, init=False)
    _last_beat_label: str = field(default="<start>", init=False)
    _last_beat_wall: float = field(default=0.0, init=False)
    _beats: int = field(default=0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    #: Per-step unit costs, recorded on every run so the budgets can be audited against
    #: what the census actually costs rather than against what someone guessed.
    costs: dict = field(default_factory=dict, init=False)
    #: Largest silence actually observed per step, in units.  This is the quantity the
    #: budget is compared against, so recording it is what makes the budget auditable —
    #: `costs` records the whole step including the beats, which overstates the headroom.
    max_silence: dict = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.beat("<start>")

    def beat(self, label: str) -> None:
        """Record forward progress.  Cheap enough to call per output line."""
        with self._lock:
            self._last_beat_units = self.clock.units
            self._last_beat_label = label
            self._last_beat_wall = time.perf_counter()
            self._beats += 1

    def silent_units(self) -> int:
        with self._lock:
            return max(0, self.clock.units - self._last_beat_units)

    def raise_if_stalled(self, *, budget_units: int, kind: str) -> None:
        """Raise :class:`Stalled` iff the watched thing has gone silent for the budget.

        Named `raise_if_stalled` and not `check` on purpose: `audit_instruments.py`'s
        harness screen counts instrument calls **by function name**, so a method called
        `check` here would have been tallied against `tests/ops/_models.py::check` and
        would have supplied that instrument with polarity evidence it never earned.  That
        is `misnamed` in Tank's own vocabulary, and it was caught by running his census
        rather than by reading this file.
        """
        self.clock.assert_alive(self.what)
        silent = self.silent_units()
        step = self._last_beat_label.split(":")[0]
        if silent > self.max_silence.get(step, -1):
            self.max_silence[step] = silent
        if silent <= budget_units:
            return
        with self._lock:
            report = StallReport(
                what=self.what,
                kind=kind,
                budget_units=budget_units,
                silent_units=silent,
                last_beat=self._last_beat_label,
                beats_seen=self._beats,
                clock_units_total=self.clock.units,
                wall_seconds_silent=time.perf_counter() - self._last_beat_wall,
            )
        raise Stalled(report)


# ---------------------------------------------------------------------------
# Running a step under the guard
# ---------------------------------------------------------------------------

#: How often the waiting loop wakes to consult the guard.  Not a budget — shortening it
#: does not make the detector stricter, because the detector's threshold is in units.
POLL_S: float = 0.25


def guarded_call(
    guard: StallGuard,
    label: str,
    fn,
    *,
    budget_units: int,
    kind: str = KIND_MECHANISM,
):
    """Run ``fn()`` on a worker thread and abandon it if it starves the guard.

    In-process steps cannot be interrupted from outside, so a stalled one is *abandoned*
    (the thread is a daemon and the process will not wait for it) rather than killed.  The
    abandonment is deliberate and is recorded: pretending the step was cancelled would be
    a claim about a thread this code has no way to stop.
    """
    box: dict = {}

    def _target() -> None:
        try:
            stall_here_if_injected(label)
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 — re-raised in the caller's thread
            box["error"] = exc

    thread = threading.Thread(target=_target, name=f"step-{label}", daemon=True)
    start_units = guard.clock.units
    guard.beat(f"{label}:start")
    thread.start()
    while thread.is_alive():
        thread.join(POLL_S)
        guard.raise_if_stalled(budget_units=budget_units, kind=kind)
    guard.costs[label] = guard.clock.units - start_units
    guard.beat(f"{label}:done")
    if "error" in box:
        raise box["error"]
    return box.get("value")


def guarded_run(
    cmd: "list[str]",
    *,
    guard: StallGuard,
    what: str,
    budget_units: int,
    kind: str = KIND_TOOLCHAIN,
    label: str | None = None,
    **popen_kwargs,
) -> subprocess.CompletedProcess:
    """``subprocess.run`` with the wall clock taken out of the gate.

    Every line the child writes on either stream is a beat, so a `cargo` build that is
    crawling under load is *visibly alive* and a child that has wedged is *visibly not*,
    without either of them being compared to a number of seconds.

    Returns a ``CompletedProcess`` with ``stdout``/``stderr`` kept separate (callers parse
    them separately, and merging them here would silently change what several witnesses
    read).  Raises :class:`Stalled` when the child goes silent for the budget, after
    killing it — a stalled child that is left running poisons every later step.
    """
    label = label or what
    out_lines: list[str] = []
    err_lines: list[str] = []

    if injected_stall_target() == label:
        # Planted stall.  The real command is replaced by a child that is genuinely
        # silent and genuinely never exits — the fault being modelled, not a mock of it.
        # It is announced on stderr and recorded in the artifact so a planted stall can
        # never be mistaken for a suffered one (the rule Tank applied to
        # `record_injected_compute_failure`).
        print(
            f"[STALL-INJECT] replacing step {label!r} with a silent non-terminating child "
            f"(${ENV_INJECT_STALL}={label}).  fault_injection: ACTIVE.",
            file=sys.stderr,
            flush=True,
        )
        cmd = [
            sys.executable,
            "-c",
            "import time,sys\n"
            "sys.stderr.write('planted silent child: producing no output, never exiting\\n')\n"
            "sys.stderr.flush()\n"
            "time.sleep(86400)\n",
        ]
        popen_kwargs.pop("env", None)
        popen_kwargs.pop("cwd", None)

    popen_kwargs.pop("capture_output", None)
    popen_kwargs.pop("timeout", None)
    try:
        proc = subprocess.Popen(  # noqa: S603
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            bufsize=1,
            **popen_kwargs,
        )
    except FileNotFoundError as exc:
        raise InstrumentAbsent(
            f"[{what} instrument failure] ERROR(instrument): the command does not exist "
            f"on this machine: {cmd[0]!r}.  The check never ran; this is not a finding.\n{exc}"
        ) from exc
    except OSError as exc:
        raise InstrumentAbsent(
            f"[{what} instrument failure] ERROR(instrument): could not start the "
            f"subprocess {cmd[0]!r}: {exc}.  The check never ran; this is not a finding."
        ) from exc

    def _pump(stream, sink: list[str], tag: str) -> None:
        try:
            for line in stream:
                sink.append(line)
                guard.beat(f"{label}:{tag}")
        finally:
            with __import__("contextlib").suppress(Exception):
                stream.close()

    pumps = [
        threading.Thread(target=_pump, args=(proc.stdout, out_lines, "out"), daemon=True),
        threading.Thread(target=_pump, args=(proc.stderr, err_lines, "err"), daemon=True),
    ]
    for p in pumps:
        p.start()

    start_units = guard.clock.units
    guard.beat(f"{label}:spawn")
    try:
        while proc.poll() is None:
            time.sleep(POLL_S)
            guard.raise_if_stalled(budget_units=budget_units, kind=kind)
    except Stalled:
        _kill_tree(proc)
        raise
    finally:
        for p in pumps:
            p.join(timeout=5.0)

    guard.costs[label] = guard.clock.units - start_units
    guard.beat(f"{label}:exit")
    return subprocess.CompletedProcess(
        cmd, proc.returncode, "".join(out_lines), "".join(err_lines)
    )


class InstrumentAbsent(Exception):
    """The command could not be started at all.  ERROR(instrument)."""


def _kill_tree(proc: "subprocess.Popen") -> None:
    """Kill the child and its descendants.  Best effort, never raises.

    The census spawns children that spawn children (``cargo`` -> ``rustc``,
    ``python`` -> the EP).  Killing only the direct child leaves the grandchild holding
    the pipe, and the next step then inherits a machine that is still busy with work
    nobody is waiting for — which looks exactly like contention and would be read as it.
    """
    import contextlib

    pid = proc.pid
    if sys.platform == "win32":
        with contextlib.suppress(Exception):
            subprocess.run(  # noqa: S603,S607
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
                timeout=30,
            )
    with contextlib.suppress(Exception):
        proc.kill()
    with contextlib.suppress(Exception):
        proc.wait(timeout=15)


# ---------------------------------------------------------------------------
# Fault injection — an injected stall must never read like a suffered one
# ---------------------------------------------------------------------------
#
# Same rule Tank applied to `record_injected_compute_failure`: the artifact says so.
ENV_INJECT_STALL: str = "ONNXRUNTIME_EP_VULKAN_TEST_STALL"


def injected_stall_target() -> str:
    """Name of the step to hang on purpose, or ``""``.  Read fresh on every call."""
    return os.environ.get(ENV_INJECT_STALL, "").strip()


def stall_here_if_injected(label: str) -> None:
    """Hang forever when this step is the injected target.  The falsifier's other arm."""
    target = injected_stall_target()
    if target and target == label:
        print(
            f"[STALL-INJECT] hanging deliberately in step {label!r} "
            f"(${ENV_INJECT_STALL}).  This is a planted stall, not a suffered one.",
            file=sys.stderr,
            flush=True,
        )
        while True:
            time.sleep(3600)
