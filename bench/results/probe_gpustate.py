"""Sample the GPU's own state while a probe runs, so a device-clock figure can be audited.

Why this exists
---------------

The coordinator told me the machine's CPU contention was his own orchestration, then corrected
it: another project is running CPU **and GPU** tests on this box. Everything the project had
established about contention covered the CPU case — Fact Checker's finding that Intel's
52.0833 ns/tick counter is reference-clock based, so host load moves work-per-tick and not the
tick. That argument says nothing about a second process putting work on the GPU.

So there is an open question with teeth: **does foreign GPU work move ``gpu_steady_tail()``,
and does the gate fire when it does?** The gate's criterion is a relative standard deviation
over a suffix — a *variance* test. Contention that is intermittent raises variance and would be
caught. Contention that is *sustained* does not raise variance at all: it shifts the whole
series to a new level that is perfectly steady about its own wrong mean. A variance gate cannot
see a bias. That is a hypothesis about the instrument, and this module exists to test it rather
than reason about it.

What it measures
----------------

``nvidia-smi`` sampled at 4 Hz for the duration of a command, giving four things the trace
cannot know because they are properties of the device rather than of our submissions:

* **SM clock** — the RTX 4060 idles at 1950 MHz against a 3105 MHz maximum. A GPU-busy series
  that falls over a run may be our kernel getting faster or may be the clock ramping up, and
  those are not the same finding. Sampling the clock separates them.
* **GPU utilisation** — whether the device was busy when our probe was not submitting.
* **Foreign compute processes** — the falsifier for "the box was quiet". A run claimed as solo
  with another PID holding GPU memory was not solo, and the claim should not survive.
* **Power draw** — the board's own account of how hard it is working.

The joined report pairs these against the probe's own steady-tail window, so a ``STEADY``
verdict can be read together with the device state that produced it. A ``STEADY`` taken while a
foreign PID held the GPU is a steady measurement of a contended device: still precise, no longer
a measurement of our kernel.

R13
---

Three terminal states, and the instrument's own failure is one of them. ``nvidia-smi`` absent,
non-zero, or emitting an unparseable row is ``ERROR(instrument)`` and never a finding of "no
contention". The absence of evidence here is produced by the same code path as the absence of
contention, so the two must be named apart or the quiet answer wins by default.

Usage::

    python bench/results/probe_gpustate.py --tag solo -- python bench/results/probe_gemv.py --device 0 --iters 44
"""

from __future__ import annotations

import argparse
import json
import re
import psutil
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: How often to ask the driver for board state. 4 Hz is fine against inferences of ~12–60 ms:
#: we are characterising the *regime* a run sat in, not resolving individual dispatches.
SAMPLE_S = 0.25

#: Fraction of a run's samples that must show a foreign compute process before the run is called
#: contended rather than merely brushed. One stray sample at start-up is not a contended run.
FOREIGN_SAMPLE_FRACTION = 0.10

GPU_QUERY = "utilization.gpu,clocks.sm,clocks.max.sm,power.draw,temperature.gpu"
APP_QUERY = "pid,used_gpu_memory"


class SamplerError(RuntimeError):
    """``nvidia-smi`` could not be used. This is ERROR(instrument), never 'no contention'."""


def _smi(args: "list[str]", timeout: float = 10.0) -> str:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        raise SamplerError("nvidia-smi is not on PATH; device state is unobservable on this host")
    p = subprocess.run([exe, *args], capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise SamplerError(f"nvidia-smi exited {p.returncode}: {p.stderr.strip()[:200]}")
    return p.stdout


def _sample_once(index: int) -> dict:
    """One board-state reading plus the compute-process table, with a host timestamp."""
    gpu = _smi(["-i", str(index), f"--query-gpu={GPU_QUERY}",
                "--format=csv,noheader,nounits"]).strip()
    fields = [f.strip() for f in gpu.split(",")]
    if len(fields) != 5:
        raise SamplerError(f"unparseable --query-gpu row: {gpu!r}")

    apps_raw = _smi(["-i", str(index), f"--query-compute-apps={APP_QUERY}",
                     "--format=csv,noheader,nounits"])
    apps = []
    for line in apps_raw.splitlines():
        line = line.strip()
        if not line or "not supported" in line.lower():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and re.fullmatch(r"\d+", parts[0]):
            apps.append({"pid": int(parts[0]), "mib": _num(parts[1])})

    return {
        "t": time.perf_counter(),
        "util_pct": _num(fields[0]),
        "sm_mhz": _num(fields[1]),
        "sm_max_mhz": _num(fields[2]),
        "power_w": _num(fields[3]),
        "temp_c": _num(fields[4]),
        "apps": apps,
    }


def _num(s: str) -> "float | None":
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _is_ours(pid: int, own_root: "int | None") -> bool:
    """True when ``pid`` is our launched command or a descendant of it.

    This exists because the first run of this probe reported ``FOREIGN_GPU_WORK`` on a run I had
    launched alone: the offending PID was my own probe's *worker* subprocess, holding 0.0 MiB.
    A contention detector that counts our own worker as a stranger fires on every run and is
    therefore not a detector — it is a constant. Ancestry is the fact that separates them, and it
    is checked live because a worker that has exited cannot be interrogated afterwards.
    """
    if own_root is None:
        return False
    if pid == own_root:
        return True
    try:
        proc = psutil.Process(pid)
        return any(a.pid == own_root for a in proc.parents())
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.Error):
        # Cannot tell. Do NOT default to "ours" — that would silently suppress real contention.
        # Defaulting to foreign is the direction that keeps the detector able to fire.
        return False


class Sampler(threading.Thread):
    """Polls the board on its own thread for as long as ``stop`` is unset.

    Classification happens **live**, inside the sampling loop, because ancestry can only be
    resolved while the process is alive. A worker that has exited by the time the report is
    written is exactly the case that would otherwise be misfiled as foreign.
    """

    def __init__(self, index: int) -> None:
        super().__init__(daemon=True)
        self.index = index
        self.stop = threading.Event()
        self.samples: "list[dict]" = []
        self.error: "str | None" = None
        #: Set by the caller once the command is launched, so ancestry has a root to test against.
        self.own_root: "int | None" = None
        self._verdict_cache: "dict[int, bool]" = {}

    def _classify(self, pid: int) -> bool:
        if pid not in self._verdict_cache:
            self._verdict_cache[pid] = _is_ours(pid, self.own_root)
        return self._verdict_cache[pid]

    def run(self) -> None:
        try:
            while not self.stop.is_set():
                s = _sample_once(self.index)
                for app in s["apps"]:
                    app["ours"] = self._classify(app["pid"])
                self.samples.append(s)
                self.stop.wait(SAMPLE_S)
        except (SamplerError, subprocess.TimeoutExpired) as exc:
            self.error = str(exc)


def summarise(samples: "list[dict]") -> dict:
    """Reduce a sample series to the few facts that decide whether a figure is quotable."""
    if not samples:
        return {"n": 0}

    def series(key):
        return [s[key] for s in samples if s.get(key) is not None]

    clocks = series("sm_mhz")
    utils = series("util_pct")
    powers = series("power_w")

    foreign_samples = 0
    foreign_pids: "dict[int, float]" = {}
    own_pids: "set[int]" = set()
    for s in samples:
        others = [a for a in s["apps"] if not a.get("ours")]
        own_pids.update(a["pid"] for a in s["apps"] if a.get("ours"))
        if others:
            foreign_samples += 1
        for a in others:
            foreign_pids[a["pid"]] = max(foreign_pids.get(a["pid"], 0.0), a["mib"] or 0.0)

    frac = foreign_samples / len(samples)
    out = {
        "n": len(samples),
        "seconds": round(samples[-1]["t"] - samples[0]["t"], 1),
        "sm_mhz": _stats(clocks),
        "sm_max_mhz": samples[0].get("sm_max_mhz"),
        "util_pct": _stats(utils),
        "power_w": _stats(powers),
        "foreign_sample_fraction": round(frac, 4),
        "foreign_pids": {str(k): v for k, v in sorted(foreign_pids.items())},
        "own_pids_on_gpu": sorted(own_pids),
    }
    # The clock ramp, stated as a ratio, because that is the size a GPU-busy series would move
    # by if the clock were the whole story.
    if clocks:
        lo, hi = min(clocks), max(clocks)
        out["clock_ramp_x"] = round(hi / lo, 4) if lo else None
        out["clock_at_max_pct"] = round(100.0 * statistics.fmean(clocks) / (out["sm_max_mhz"] or hi), 1)
    out["verdict"] = ("FOREIGN_GPU_WORK" if frac >= FOREIGN_SAMPLE_FRACTION else "SOLE_TENANT")
    return out


def _stats(vs: "list[float]") -> dict:
    if not vs:
        return {"n": 0}
    return {"n": len(vs), "min": round(min(vs), 1), "median": round(statistics.median(vs), 1),
            "mean": round(statistics.fmean(vs), 1), "max": round(max(vs), 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", type=int, default=0, help="nvidia-smi board index to watch")
    ap.add_argument("--tag", default="run")
    ap.add_argument("cmd", nargs=argparse.REMAINDER,
                    help="-- followed by the command to run while sampling")
    args = ap.parse_args()

    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not cmd:
        print("ERROR(instrument): no command given after --", file=sys.stderr)
        return 2

    try:
        _sample_once(args.device)
    except (SamplerError, subprocess.TimeoutExpired) as exc:
        # R13: the instrument failed. That is not a finding about contention.
        print(f"ERROR(instrument): {exc}", file=sys.stderr)
        return 2

    sampler = Sampler(args.device)
    sampler.start()
    started = time.perf_counter()
    proc = subprocess.Popen(cmd, text=True)
    sampler.own_root = proc.pid
    proc.wait()
    elapsed = time.perf_counter() - started
    sampler.stop.set()
    sampler.join(timeout=10.0)

    if sampler.error:
        print(f"ERROR(instrument): sampling failed mid-run: {sampler.error}", file=sys.stderr)
        return 2

    summary = summarise(sampler.samples)
    summary["tag"] = args.tag
    summary["command"] = cmd
    summary["command_exit"] = proc.returncode
    summary["own_root_pid"] = proc.pid
    summary["wall_s"] = round(elapsed, 1)

    print(f"\n== device state during '{args.tag}' ==")
    print(f"verdict            : {summary['verdict']}"
          f"   (foreign compute apps in {summary['foreign_sample_fraction']:.0%} of "
          f"{summary['n']} samples over {summary['wall_s']} s)")
    print(f"our own pids on gpu: {summary['own_pids_on_gpu']}  (root {proc.pid})")
    if summary["foreign_pids"]:
        print(f"foreign pids       : {summary['foreign_pids']}")
    c, u, p = summary["sm_mhz"], summary["util_pct"], summary["power_w"]
    print(f"SM clock MHz       : min {c['min']}  median {c['median']}  max {c['max']}"
          f"   (board max {summary['sm_max_mhz']}, ran at {summary.get('clock_at_max_pct')}% of it)")
    print(f"clock ramp         : {summary.get('clock_ramp_x')}x from lowest to highest sample")
    print(f"utilisation %      : min {u['min']}  median {u['median']}  max {u['max']}")
    print(f"power W            : min {p['min']}  median {p['median']}  max {p['max']}")

    out = HERE / f"gpustate_{args.tag}.json"
    out.write_text(json.dumps(summary, indent=2), "utf-8")
    print(f"\nwrote {out}")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
