"""Machine quiescence — the guard that refuses a number taken on a busy machine.

Why this exists
---------------
On 2026-07-30 the coordinator measured the same device, the same build and the same test
twice and got answers 9.5x apart::

    device 0, machine quiet          vulkan.record total    19 460 ms
    device 0, six agents compiling   vulkan.record total   184 356 ms

Nothing in ``bench/`` noticed. Every guard we had was pointed at the *program*: is the EP
loaded (``refuse_if_ep_absent``), did every island actually run (``dispatch_accounting``), is
the baseline moving (``stats.drift``). None of them can see a number that is uniformly wrong
because the machine was busy for the whole run. ``stats.drift`` in particular detects a
baseline *moving*; a uniformly loaded machine produces a baseline that is stable and wrong,
which is the worst possible failure — it looks like the good case.

This is the same defect class as the two fabricated speedups in ``test_plausible_but_wrong.py``:
a plausible number, produced by a working harness, that means something other than what it
says. The response is the same one this project has adopted every time: **refuse, do not warn.**
A number printed under a warning gets quoted without the warning.

The verdict
-----------
``quiescence(...)`` returns one of three values, and it gates performance numbers exactly the
way ``model_output_equivalence`` does in DESIGN.md §10.0:

``QUIET``
    Foreign CPU work stayed below the threshold for the whole measurement window. Numbers may
    be quoted.
``CONTENDED``
    Other processes were competing for the CPU. **No performance number may be quoted.** The
    result is reported as UNMEASURED, not as a warned-about figure.
``UNMEASURED``
    We could not tell. Untested is not quiet, by the same rule that makes ``stats.drift``
    refuse a ratio when there are too few samples to test steadiness.

Two instruments, because one can be fooled
------------------------------------------
R9 (DESIGN.md §10.0.1): confidence scales with agreeing instruments, evidence scales only with
falsifying ones. There are two here and they fail differently.

1. **The survey** (:class:`Monitor`) — an *absolute* accounting. Every sample reads the
   system-wide idle counter; busy CPU-seconds are ``cores * wall - idle``, and our own process
   tree's CPU-seconds are subtracted to leave ``foreign_cpu_s``. Divided by wall time this
   gives ``foreign_busy_cores``: the average number of cores other processes kept busy while we
   measured. It needs no reference measurement and no calibration, and because it reads a
   system-wide counter rather than iterating processes it **cannot miss a short-lived process**
   — a ``rustc`` that starts and exits between two samples still shows up.

2. **The tachometer** (:func:`occupancy_probe`) — a fixed quantity of single-threaded integer
   work, timed. This measures the thing that actually matters, which is not "is the machine
   busy" but "can this process get a core", and it is sensitive to causes the survey is blind
   to: thermal throttling, a co-tenant VM, a power-plan change, CPU affinity. It is *relative*:
   it needs a quiet-machine reference, persisted in ``machine-baseline.json``. With no
   reference it reports ``VACUOUS`` rather than "pass" — the same discipline
   ``timestamp_audit`` uses on a device whose ``timestampPeriod`` is 1.0.

They can disagree, and disagreement is informative: a quiet survey with a slow tachometer is
not a quiet machine, it is a machine that is slow for a reason we have not identified. That
combination resolves to ``CONTENDED``.

What goes red if a number from this module is false
---------------------------------------------------
* ``idle_accounting`` — per sample, ``busy + idle`` must reconstruct ``cores * wall`` within
  5%. If the idle counter is not what we think it is, this fires and the verdict is
  ``UNMEASURED`` rather than a fabricated "quiet".
* ``own_cpu_not_exceeding_busy`` — our own tree cannot have used more CPU than the machine
  did. If it does, the tree walk is double-counting and ``foreign_cpu_s`` is meaningless.
* The tachometer is itself the falsifier for the survey, and the survey for the tachometer.

Deliberate conservative bias
----------------------------
A process first seen part-way through the window has its CPU-so-far taken as its baseline, so
work it did before we noticed it is attributed to *foreign* rather than to us. This
under-counts our own CPU and over-counts foreign CPU, which makes the guard refuse more often
than strictly necessary. That is the correct direction for a guard: a false refusal costs a
re-run, a false pass costs a published number that is wrong by 9.5x.
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

try:  # pragma: no cover - exercised by the import-failure path only
    import psutil
except Exception:  # pragma: no cover
    psutil = None  # type: ignore[assignment]

QUIET = "QUIET"
CONTENDED = "CONTENDED"
UNMEASURED = "UNMEASURED"

#: Average foreign busy cores at or below which the machine counts as quiet. Idle Windows sits
#: near 0.1-0.3 cores (Defender, search indexer, telemetry); one compiler is >= 1.0. This is a
#: judgement, not a measurement, and it is stated here rather than buried in a comparison.
QUIET_BUSY_CORES = 0.5

#: A single sample above this many foreign busy cores is "loud".
LOUD_SAMPLE_CORES = 1.0

#: Fraction of loud samples tolerated inside an otherwise quiet window. A 40-minute run will
#: catch a Defender scan or an installer; a run that is loud for more than this is contended.
QUIET_LOUD_FRACTION = 0.10

#: Tachometer tolerance against the persisted quiet reference.
OCCUPANCY_TOLERANCE = 1.15

#: Sampling period, seconds.
DEFAULT_INTERVAL = 1.0

#: How often the (expensive) descendant scan is redone. See :meth:`Monitor._own_tree`.
CHILD_REFRESH_S = 15.0

#: The monitor's own CPU cost, as a fraction of one core, above which it is perturbing the
#: measurement badly enough that its own verdict is not trustworthy.
MONITOR_SELF_COST_CORES = 0.05

_BASELINE_NAME = "machine-baseline.json"


# --------------------------------------------------------------------------------------
# the survey
# --------------------------------------------------------------------------------------


@dataclass
class Sample:
    """One reading of the system-wide idle counter plus our own tree's CPU."""

    t: float
    idle: float
    own_cpu: float
    #: Foreign busy cores over the interval that *ended* at this sample. ``None`` for the first.
    foreign_busy_cores: "float | None" = None
    #: ``busy + idle`` over that interval, divided by ``cores * wall``. Should be ~1.0.
    accounting_ratio: "float | None" = None


@dataclass
class Window:
    """The survey's verdict material for one measurement window."""

    available: bool
    reason: str = ""
    cores: int = 0
    wall_s: float = 0.0
    busy_cpu_s: float = 0.0
    own_cpu_s: float = 0.0
    foreign_cpu_s: float = 0.0
    mean_foreign_busy_cores: float = 0.0
    peak_foreign_busy_cores: float = 0.0
    median_foreign_busy_cores: float = 0.0
    loud_sample_fraction: float = 0.0
    n_samples: int = 0
    top_foreign: list = field(default_factory=list)
    falsifiers: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "available": self.available,
            "reason": self.reason,
            "cores": self.cores,
            "wall_s": round(self.wall_s, 3),
            "busy_cpu_s": round(self.busy_cpu_s, 2),
            "own_cpu_s": round(self.own_cpu_s, 2),
            "foreign_cpu_s": round(self.foreign_cpu_s, 2),
            "mean_foreign_busy_cores": round(self.mean_foreign_busy_cores, 3),
            "median_foreign_busy_cores": round(self.median_foreign_busy_cores, 3),
            "peak_foreign_busy_cores": round(self.peak_foreign_busy_cores, 3),
            "loud_sample_fraction": round(self.loud_sample_fraction, 4),
            "n_samples": self.n_samples,
            "top_foreign": self.top_foreign,
            "falsifiers": self.falsifiers,
        }


#: Windows' pid-0 accumulator. It is not a process doing work; its "CPU time" *is* idle time,
#: and counting it as a foreign consumer would name the one thing on the box that is not
#: competing with us. Diagnostic-only, but a wrong name in a report gets quoted like any other
#: number.
_NOT_REAL_WORK = {"System Idle Process", "Idle"}


def _own_tree_cpu(seen: dict, procs: list) -> float:
    """CPU-seconds used by this process and its descendants, accumulated across exits.

    ``seen`` maps pid -> the largest CPU total ever observed for that pid. Children that exit
    between samples keep their last reading, so their work is not silently reattributed to
    foreign processes.
    """
    for p in procs:
        try:
            ct = p.cpu_times()
            total = float(ct.user) + float(ct.system)
        except Exception:
            continue
        if total > seen.get(p.pid, 0.0):
            seen[p.pid] = total
    return sum(seen.values())


class Monitor:
    """Samples machine load on a background thread for the duration of a measurement.

    Sampling *during* the run is the point. A machine that is quiet when the harness starts and
    busy at minute 20 is the failure case, and it is precisely what a single before-and-after
    check would miss.
    """

    def __init__(
        self,
        interval: float = DEFAULT_INTERVAL,
        name_top: int = 5,
        child_refresh_s: float = CHILD_REFRESH_S,
    ) -> None:
        self.interval = interval
        self.name_top = name_top
        self.child_refresh_s = child_refresh_s
        self.samples: list[Sample] = []
        self._stop = threading.Event()
        self._thread: "threading.Thread | None" = None
        self._own_seen: dict = {}
        self._own_baseline = 0.0
        self._proc_start: dict = {}
        self._tree: list = []
        self._tree_at = 0.0
        self._sampler_cpu = 0.0
        self._cores = psutil.cpu_count() if psutil is not None else (os.cpu_count() or 0)
        self._unavailable = "" if psutil is not None else "psutil not importable"

    # -- lifecycle -------------------------------------------------------------------
    def start(self) -> "Monitor":
        if psutil is None:
            return self
        self._own_seen = {}
        self._sampler_cpu = 0.0
        self._tree, self._tree_at = self._own_tree(force=True), time.perf_counter()
        self._own_baseline = _own_tree_cpu(self._own_seen, self._tree)
        self._proc_start = self._snapshot_processes()
        self.samples = [
            Sample(t=time.perf_counter(), idle=self._idle(), own_cpu=self._own_baseline)
        ]
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="contention")
        self._thread.start()
        return self

    def stop(self) -> Window:
        if psutil is None:
            return Window(available=False, reason=self._unavailable)
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval * 3)
        self._own_tree(force=True)
        self._take_sample()
        return self._summarise()

    def __enter__(self) -> "Monitor":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.window = self.stop()

    # -- internals -------------------------------------------------------------------
    @staticmethod
    def _idle() -> float:
        return float(psutil.cpu_times().idle)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self._take_sample()
            except Exception:  # pragma: no cover - sampling must never kill a benchmark
                pass

    def _own_tree(self, force: bool = False) -> list:
        """Our process and its descendants, re-scanned at most every ``child_refresh_s``.

        ``Process.children(recursive=True)`` walks every process on the box to build a parent
        map. Called once a second for forty minutes that is not free: the first version of this
        monitor spent 9 CPU-seconds in 12 seconds of wall clock, which is enough foreign load to
        trip its own threshold. An instrument that perturbs what it measures is not an
        instrument. Between refreshes we reuse the cached handles, which are cheap.
        """
        now = time.perf_counter()
        if not force and self._tree and (now - self._tree_at) < self.child_refresh_s:
            return self._tree
        try:
            me = psutil.Process(os.getpid())
            self._tree = [me] + me.children(recursive=True)
        except Exception:
            pass
        self._tree_at = now
        return self._tree

    def _take_sample(self) -> None:
        if psutil is None or not self.samples:
            return
        _t_enter = time.thread_time()
        now = time.perf_counter()
        idle = self._idle()
        own = _own_tree_cpu(self._own_seen, self._own_tree())
        prev = self.samples[-1]
        dt = now - prev.t
        s = Sample(t=now, idle=idle, own_cpu=own)
        if dt > 0 and self._cores:
            capacity = self._cores * dt
            d_idle = max(0.0, idle - prev.idle)
            busy = capacity - d_idle
            d_own = max(0.0, own - prev.own_cpu)
            s.foreign_busy_cores = max(0.0, (busy - d_own)) / dt
            s.accounting_ratio = (max(0.0, busy) + d_idle) / capacity if capacity else None
        self.samples.append(s)
        # CPU time on the sampling thread, not wall time. Wall time was the first attempt and it
        # is confounded with exactly the thing this guard exists to detect: on a machine at 100%
        # CPU a 5 ms sample takes 100 ms of wall clock because it spends the rest descheduled,
        # so the "am I perturbing the measurement" check went red *because the machine was busy*.
        # A falsifier that fires on the condition it is supposed to be independent of is not a
        # control.
        self._sampler_cpu += max(0.0, time.thread_time() - _t_enter)

    def _snapshot_processes(self) -> dict:
        out: dict = {}
        if psutil is None:
            return out
        own = set()
        try:
            me = psutil.Process(os.getpid())
            own = {me.pid} | {c.pid for c in me.children(recursive=True)}
        except Exception:
            pass
        for p in psutil.process_iter(["pid", "name", "cpu_times"]):
            try:
                if p.info["pid"] in own:
                    continue
                name = p.info["name"] or "?"
                if name in _NOT_REAL_WORK or p.info["pid"] == 0:
                    continue
                ct = p.info["cpu_times"]
                if ct is None:
                    continue
                out[p.info["pid"]] = (name, float(ct.user) + float(ct.system))
            except Exception:
                continue
        return out

    def _top_foreign(self) -> list:
        """Biggest foreign CPU consumers over the window, by name.

        Diagnostic only — it is *not* the measurement. Processes that started and finished
        inside the window are invisible here but are fully counted by the idle-counter
        accounting above, which is why the accounting and not this list decides the verdict.
        """
        end = self._snapshot_processes()
        by_name: dict = {}
        for pid, (name, cpu) in end.items():
            before = self._proc_start.get(pid)
            base = before[1] if before else 0.0
            delta = cpu - base
            if delta > 0.05:
                by_name[name] = by_name.get(name, 0.0) + delta
        rows = sorted(by_name.items(), key=lambda kv: -kv[1])[: self.name_top]
        return [{"name": n, "cpu_s": round(c, 2)} for n, c in rows]

    def _summarise(self) -> Window:
        deltas = [s for s in self.samples if s.foreign_busy_cores is not None]
        if len(self.samples) < 2 or not deltas:
            return Window(
                available=False,
                reason="fewer than two samples; window too short to measure load",
                cores=self._cores,
                n_samples=len(self.samples),
            )
        first, last = self.samples[0], self.samples[-1]
        wall = last.t - first.t
        capacity = self._cores * wall
        d_idle = max(0.0, last.idle - first.idle)
        busy = capacity - d_idle
        own = max(0.0, last.own_cpu - first.own_cpu)
        foreign = max(0.0, busy - own)
        vals = [float(s.foreign_busy_cores) for s in deltas]
        loud = sum(1 for v in vals if v > LOUD_SAMPLE_CORES) / len(vals)
        ratios = [s.accounting_ratio for s in deltas if s.accounting_ratio is not None]
        acct = statistics.median(ratios) if ratios else None
        monitor_cores = (self._sampler_cpu / wall) if wall else 0.0

        falsifiers = {
            "idle_accounting": {
                "red_if": "median (busy+idle)/(cores*wall) departs from 1.0 by more than 5%",
                "value": None if acct is None else round(acct, 4),
                "red": acct is None or abs(acct - 1.0) > 0.05,
            },
            "own_cpu_not_exceeding_busy": {
                "red_if": "our own process tree used more CPU than the machine reports as busy",
                "own_cpu_s": round(own, 2),
                "busy_cpu_s": round(busy, 2),
                "red": own > busy * 1.05 + 1.0,
            },
            "monitor_not_perturbing": {
                "red_if": (
                    f"the sampler's own CPU cost exceeds {MONITOR_SELF_COST_CORES} cores, at "
                    "which point the instrument is part of the load it is measuring"
                ),
                "monitor_cores": round(monitor_cores, 4),
                "red": monitor_cores > MONITOR_SELF_COST_CORES,
            },
        }
        return Window(
            available=True,
            cores=self._cores,
            wall_s=wall,
            busy_cpu_s=busy,
            own_cpu_s=own,
            foreign_cpu_s=foreign,
            mean_foreign_busy_cores=foreign / wall if wall else 0.0,
            median_foreign_busy_cores=statistics.median(vals),
            peak_foreign_busy_cores=max(vals),
            loud_sample_fraction=loud,
            n_samples=len(self.samples),
            top_foreign=self._top_foreign(),
            falsifiers=falsifiers,
        )


def sample_now(seconds: float = 2.0, interval: float = 0.5) -> Window:
    """Blocking spot-check of machine load. Used before a run to decide whether to start."""
    m = Monitor(interval=interval).start()
    time.sleep(max(seconds, interval * 2))
    return m.stop()


# --------------------------------------------------------------------------------------
# the tachometer
# --------------------------------------------------------------------------------------


def _spin(n: int) -> int:
    x = 1
    for _ in range(n):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
    return x


#: Chosen so one spin is tens of milliseconds on a contemporary core: long enough to average
#: over a scheduler quantum, short enough that ten of them cost under a second.
SPIN_ITERS = 400_000


def occupancy_probe(reps: int = 7, iters: int = SPIN_ITERS) -> float:
    """Seconds for a fixed quantity of single-threaded integer work, best of ``reps``.

    ``min`` is the right estimator: we are asking how fast this machine *can* go right now, and
    every source of error (preemption, GC, an interrupt) is one-sided and makes it slower.
    """
    best = float("inf")
    for _ in range(max(1, reps)):
        t0 = time.perf_counter()
        _spin(iters)
        best = min(best, time.perf_counter() - t0)
    return best


def machine_key() -> str:
    """Identity of the thing the tachometer reference belongs to.

    The reference is not portable across CPUs *or* across Python builds — the spin loop is
    interpreter work, so a Python upgrade invalidates it. Both go in the key so a stale
    reference is never silently reused.
    """
    cpu = platform.processor() or "?"
    return f"{platform.node()}|{cpu}|py{platform.python_version()}|{os.cpu_count()}"


def baseline_path(results_dir: "Path | None" = None) -> Path:
    base = results_dir or (Path(__file__).resolve().parent / "results")
    return base / _BASELINE_NAME


def load_baseline(results_dir: "Path | None" = None) -> "dict | None":
    p = baseline_path(results_dir)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8")).get(machine_key())
    except Exception:
        return None


def save_baseline(seconds: float, window: Window, results_dir: "Path | None" = None) -> Path:
    """Record a quiet-machine tachometer reference. Only ever called with a QUIET survey."""
    p = baseline_path(results_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        blob = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    except Exception:
        blob = {}
    prev = blob.get(machine_key()) or {}
    kept = min(seconds, prev["spin_s"]) if prev.get("spin_s") else seconds
    blob[machine_key()] = {
        "spin_s": kept,
        "spin_iters": SPIN_ITERS,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "survey_mean_foreign_busy_cores": round(window.mean_foreign_busy_cores, 3),
        "confirmations": int(prev.get("confirmations", 0)) + 1,
        "note": (
            "Fastest fixed-work spin observed on a machine whose survey said QUIET. Kept as a "
            "minimum across recordings: a later, slower reading cannot raise it, so a "
            "reference taken on a busy machine is corrected by the first quiet one."
        ),
    }
    p.write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")
    return p


def occupancy_check(results_dir: "Path | None" = None, reps: int = 7) -> dict:
    """Compare a live spin against the persisted quiet reference.

    Reports ``VACUOUS`` when no reference exists — never "pass". An instrument that cannot
    falsify anything on this machine must say so, exactly as ``timestamp_audit`` does on a
    device with ``timestampPeriod`` 1.0.
    """
    now = occupancy_probe(reps=reps)
    ref = load_baseline(results_dir)
    if not ref or not ref.get("spin_s"):
        return {
            "verdict": "VACUOUS",
            "spin_s": round(now, 6),
            "reference_s": None,
            "ratio": None,
            "red": False,
            "reason": (
                "no quiet-machine reference recorded for this host+python; the tachometer "
                "cannot falsify the survey here. Untested is not passed."
            ),
        }
    ratio = now / float(ref["spin_s"])
    return {
        "verdict": "SLOW" if ratio > OCCUPANCY_TOLERANCE else "NOMINAL",
        "spin_s": round(now, 6),
        "reference_s": round(float(ref["spin_s"]), 6),
        "reference_recorded_at": ref.get("recorded_at"),
        "ratio": round(ratio, 4),
        "tolerance": OCCUPANCY_TOLERANCE,
        "red": ratio > OCCUPANCY_TOLERANCE,
        "reason": (
            f"fixed-work spin took {ratio:.2f}x its quiet-machine reference; this process is "
            "not getting a core at full speed"
            if ratio > OCCUPANCY_TOLERANCE
            else "fixed-work spin matches the quiet-machine reference"
        ),
    }


# --------------------------------------------------------------------------------------
# the verdict
# --------------------------------------------------------------------------------------


def quiescence(window: Window, occupancy: "dict | None" = None) -> dict:
    """Combine survey and tachometer into ``QUIET`` / ``CONTENDED`` / ``UNMEASURED``.

    Resolution rules, in order:

    1. Survey unavailable or its own falsifiers red -> ``UNMEASURED``. We do not guess.
    2. Survey over threshold -> ``CONTENDED``.
    3. Survey quiet but tachometer red -> ``CONTENDED``. Two instruments disagreeing is not a
       tie to be broken in favour of the convenient answer; it is an unexplained slowdown.
    4. Survey quiet, tachometer nominal or vacuous -> ``QUIET``, with the tachometer's own
       verdict carried alongside so a vacuous check is visible in the record.
    """
    reasons: list[str] = []
    if not window.available:
        return {
            "verdict": UNMEASURED,
            "reasons": [window.reason or "load could not be sampled"],
            "survey": window.to_dict(),
            "occupancy": occupancy,
        }

    red_falsifiers = [k for k, v in window.falsifiers.items() if v.get("red")]
    if red_falsifiers:
        return {
            "verdict": UNMEASURED,
            "reasons": [f"load accounting falsifier red: {', '.join(red_falsifiers)}"],
            "survey": window.to_dict(),
            "occupancy": occupancy,
        }

    if window.mean_foreign_busy_cores > QUIET_BUSY_CORES:
        reasons.append(
            f"other processes kept {window.mean_foreign_busy_cores:.2f} cores busy on average "
            f"(threshold {QUIET_BUSY_CORES}); "
            f"{window.foreign_cpu_s:.0f} foreign CPU-seconds over {window.wall_s:.0f}s wall"
        )
    if window.loud_sample_fraction > QUIET_LOUD_FRACTION:
        reasons.append(
            f"{window.loud_sample_fraction * 100:.0f}% of samples exceeded "
            f"{LOUD_SAMPLE_CORES} foreign busy cores (threshold "
            f"{QUIET_LOUD_FRACTION * 100:.0f}%)"
        )
    if occupancy and occupancy.get("red"):
        reasons.append(f"occupancy probe: {occupancy.get('reason')}")

    verdict = CONTENDED if reasons else QUIET
    if verdict == QUIET:
        reasons.append(
            f"foreign load {window.mean_foreign_busy_cores:.2f} cores over "
            f"{window.wall_s:.0f}s, peak {window.peak_foreign_busy_cores:.2f}"
        )
    return {
        "verdict": verdict,
        "reasons": reasons,
        "survey": window.to_dict(),
        "occupancy": occupancy,
        "thresholds": {
            "quiet_busy_cores": QUIET_BUSY_CORES,
            "loud_sample_cores": LOUD_SAMPLE_CORES,
            "quiet_loud_fraction": QUIET_LOUD_FRACTION,
            "occupancy_tolerance": OCCUPANCY_TOLERANCE,
        },
    }


def gate(verdict: dict, what: str = "performance number") -> "str | None":
    """The refusal string, or ``None`` when the number may be quoted.

    Mirrors ``phi35.ratio_refusal``. The caller must substitute this for the number, not print
    it beside the number: a figure printed under a warning gets quoted without the warning.
    """
    v = verdict.get("verdict")
    if v == QUIET:
        return None
    head = (
        "machine was CONTENDED during measurement"
        if v == CONTENDED
        else "machine quiescence UNMEASURED"
    )
    return (
        f"REFUSED: {what} withheld — {head}. "
        + "; ".join(verdict.get("reasons") or ["no reason recorded"])
        + ". Same device and build measured 9.5x apart on load alone (docs/PERF.md §10)."
    )


def describe(verdict: dict) -> str:
    v = verdict.get("verdict", UNMEASURED)
    s = verdict.get("survey") or {}
    occ = verdict.get("occupancy") or {}
    mark = {QUIET: "OK", CONTENDED: "RED", UNMEASURED: "??"}.get(v, "??")
    lines = [f"[{mark}] machine_quiescence: {v}"]
    if s.get("available"):
        lines.append(
            f"      foreign load: mean {s['mean_foreign_busy_cores']:.2f} cores, "
            f"median {s['median_foreign_busy_cores']:.2f}, peak {s['peak_foreign_busy_cores']:.2f} "
            f"of {s['cores']} over {s['wall_s']:.0f}s ({s['n_samples']} samples)"
        )
        lines.append(
            f"      foreign CPU {s['foreign_cpu_s']:.0f}s vs our own {s['own_cpu_s']:.0f}s; "
            f"loud samples {s['loud_sample_fraction'] * 100:.0f}%"
        )
        if s.get("top_foreign"):
            top = ", ".join(f"{r['name']} {r['cpu_s']:.0f}s" for r in s["top_foreign"])
            lines.append(f"      busiest foreign processes: {top}")
    if occ:
        lines.append(
            f"      occupancy probe: {occ.get('verdict')} "
            f"(ratio {occ.get('ratio')}, tolerance {occ.get('tolerance')}) — {occ.get('reason')}"
        )
    for r in verdict.get("reasons") or []:
        lines.append(f"      - {r}")
    return "\n".join(lines)


def main(argv: "list[str] | None" = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Is this machine quiet enough to benchmark on?")
    ap.add_argument("--seconds", type=float, default=10.0, help="sampling duration")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--json", action="store_true")
    ap.add_argument(
        "--record-baseline",
        action="store_true",
        help="if the survey says QUIET, store the tachometer reading as this machine's "
        "quiet reference (kept as a minimum across recordings)",
    )
    a = ap.parse_args(argv)

    m = Monitor(interval=a.interval).start()
    time.sleep(max(a.seconds, a.interval * 2))
    win = m.stop()
    occ = occupancy_check()
    v = quiescence(win, occ)
    if a.record_baseline and v["verdict"] == QUIET:
        p = save_baseline(occ["spin_s"], win)
        v["baseline_written"] = str(p)
    if a.json:
        print(json.dumps(v, indent=2))
    else:
        print(describe(v))
        if v.get("baseline_written"):
            print(f"      quiet reference written to {v['baseline_written']}")
    return 0 if v["verdict"] == QUIET else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
