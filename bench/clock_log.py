"""A continuous record of what the board was doing, so a figure can be certified afterwards.

WHY
===
On this hardware a quiet window is not something to wait for -- the box is shared with another
team for the foreseeable future (`docs/PERF.md` §20). The old workflow was: notice the machine
is quiet, start a run, sample the device alongside it, certify. That workflow has a step that
never completes.

This inverts it. The sampler runs continuously and cheaply, writing an append-only record of
tenancy and SM clock. When a run happens to land in a genuinely quiet minute, the companion for
it already exists and the figure can be certified **retrospectively**, from the window the run
occupied.

The SM-clock axis is the only instrument on this project that has ever caught its failure
class. The specimen is `phi35-0baf660-dev0.json`: device-clock series STEADY, RSD 0.0717% --
tighter than all 28 census traces -- under SOLE_TENANT with zero foreign GPU work, and 20.18x
wrong. Tail and tenancy both said pass; 210.0 MHz across min, median, mean *and max* of 160
samples against a 3105 MHz boost is what refused it. Keeping that axis recording even when
nothing is being certified costs a subprocess every quarter second and is the cheapest
insurance the project holds.

WHAT IT IS NOT
==============
It is not a certification. `window()` returns the same shape `bench/device_companion.py`
consumes, and `certify()` still decides. This module only guarantees that the evidence exists
when the question is asked.

It is also **not a substitute for an in-run companion**. A retrospectively assembled window is
sampled at 4 Hz over wall time and cannot know which of its samples overlapped a submission;
an in-run companion at least shares the process. Retrospective certification is weaker and
`window()` says so in its own silence set.

Usage:
    python bench/clock_log.py --record            # run continuously (Ctrl-C to stop)
    python bench/clock_log.py --record --seconds 600
    python bench/clock_log.py --window <iso_start> <iso_end>
    python bench/clock_log.py --summary
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_RESULTS = _HERE / "results"

#: Append-only. One JSON object per line, each a single 4 Hz sample plus its wall clock.
LOG = _RESULTS / "clock_log.jsonl"

#: Matches probe_gpustate.SAMPLE_S so the two axes are directly comparable.
SAMPLE_S = 0.25

#: Retrospective windows thinner than this are not worth certifying against: at 4 Hz a
#: ten-second window is forty samples, and fewer says less about a regime than it appears to.
MIN_WINDOW_SAMPLES = 40

UNOBSERVABLE = "UNOBSERVABLE"

RETROSPECTIVE_SILENCE = (
    "Assembled retrospectively from a wall-clock window rather than sampled inside the run. It "
    "cannot know which samples overlapped a submission, so it is weaker than an in-run "
    "companion and must not be used to upgrade a figure an in-run companion refused."
)


def _probe():
    """Reuse Switch's sampler rather than writing a second one for the same axis."""
    sys.path.insert(0, str(_RESULTS))
    try:
        import probe_gpustate  # noqa: E402
    finally:
        # bench/results carries modules whose names collide with lane checks elsewhere in the
        # tree; leaving it on sys.path is the defect locked by bench/test_import_isolation.py.
        try:
            sys.path.remove(str(_RESULTS))
        except ValueError:
            pass
    return probe_gpustate


def record(seconds: "float | None" = None, index: int = 0, log: "Path | None" = None) -> int:
    """Sample the board until interrupted, appending one line per sample."""
    probe = _probe()
    path = Path(log) if log else LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    n = 0
    try:
        while seconds is None or (time.monotonic() - started) < seconds:
            t0 = time.monotonic()
            try:
                sample = probe._sample_once(index)
            except Exception as exc:  # sampler unavailable -> record the silence, not a zero
                sample = {"error": f"{type(exc).__name__}: {exc}"}
            sample["wall"] = datetime.now(timezone.utc).isoformat()
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(sample) + "\n")
            n += 1
            time.sleep(max(0.0, SAMPLE_S - (time.monotonic() - t0)))
    except KeyboardInterrupt:
        pass
    print(f"{n} samples appended to {path}")
    return 0


def load(log: "Path | None" = None) -> list:
    path = Path(log) if log else LOG
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def window(start_iso: str, end_iso: str, log: "Path | None" = None) -> dict:
    """Assemble a device-state record for a past window, in `device_companion`'s shape.

    Returns a record whose `verdict` may be `UNOBSERVABLE` -- which is not a pass. A window
    with no samples in it is not a quiet window, it is an unrecorded one, and those are
    different (R12).
    """
    probe = _probe()
    start = datetime.fromisoformat(start_iso)
    end = datetime.fromisoformat(end_iso)
    if end <= start:
        return {
            "verdict": UNOBSERVABLE,
            "reason": "ERROR(instrument): the window ends at or before it starts.",
            "n": 0,
        }

    picked = []
    for s in load(log):
        w = s.get("wall")
        if not w:
            continue
        try:
            t = datetime.fromisoformat(w)
        except ValueError:
            continue
        if start <= t <= end:
            picked.append(s)

    # A sample missing the fields `summarise` needs is a truncated or corrupt line, not a
    # quiet moment. Dropping it here keeps the refusal path from raising: a harness that dies
    # on its own error path cannot report the error it found.
    required = ("t", "sm_mhz", "apps")
    usable = [s for s in picked if "error" not in s and all(k in s for k in required)]
    malformed = len(picked) - len(usable) - sum(1 for s in picked if "error" in s)

    if len(usable) < MIN_WINDOW_SAMPLES:
        return {
            "verdict": UNOBSERVABLE,
            "reason": (
                f"only {len(usable)} usable samples in this window; {MIN_WINDOW_SAMPLES} are "
                "required. A window with too few samples is unrecorded, not quiet."
            ),
            "n": len(usable),
            "malformed_samples": malformed,
            "window": {"start": start_iso, "end": end_iso},
            "retrospective": True,
        }

    try:
        state = probe.summarise(usable)
    except (KeyError, IndexError, TypeError) as exc:
        return {
            "verdict": UNOBSERVABLE,
            "reason": f"ERROR(instrument): the sampler could not summarise this window: "
                      f"{type(exc).__name__}: {exc}",
            "n": len(usable),
            "window": {"start": start_iso, "end": end_iso},
            "retrospective": True,
        }
    state["retrospective"] = True
    state["malformed_samples"] = malformed
    state["window"] = {"start": start_iso, "end": end_iso}
    state["sampler"] = "bench/clock_log.py (continuous), sampling bench/results/probe_gpustate.py"
    state.setdefault("silence_set", []).append(RETROSPECTIVE_SILENCE)
    return state


def summary(log: "Path | None" = None) -> dict:
    samples = load(log)
    usable = [s for s in samples if "error" not in s]
    walls = sorted(s["wall"] for s in samples if s.get("wall"))
    out = {
        "path": str(Path(log) if log else LOG),
        "samples": len(samples),
        "usable": len(usable),
        "errors": len(samples) - len(usable),
        "first": walls[0] if walls else None,
        "last": walls[-1] if walls else None,
    }
    if usable:
        probe = _probe()
        out["state"] = probe.summarise(usable)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--record", action="store_true", help="sample continuously")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--window", nargs=2, metavar=("START_ISO", "END_ISO"))
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    log = Path(args.log) if args.log else None

    if args.record:
        return record(args.seconds, args.device, log)
    if args.window:
        print(json.dumps(window(args.window[0], args.window[1], log), indent=1, default=str))
        return 0
    if args.summary:
        s = summary(log)
        print(json.dumps(s, indent=1, default=str)[:3000])
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
