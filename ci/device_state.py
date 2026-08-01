#!/usr/bin/env python3
"""``device_state`` — the §10.0 obligation-8 companion, read from the CI side.

Why this file exists
--------------------

Switch showed that ``gpu_steady_tail()`` is a **variance test over a suffix**, and a
variance test cannot see a bias.  A board held at its 210 MHz idle clock against a
3105 MHz boost produces a series that is *perfectly steady about a wrong mean*, so the
wrong figure earns the gate's most confident verdict:

    soloA       [SOLE_TENANT]              STEADY   11.525 ms   RSD 0.8098%
    contended3  truncated to 20 inferences STEADY  126.647 ms   RSD 0.9103%   <- 10.99x wrong
    contended3  truncated to 28 inferences STEADY  126.647 ms   RSD 0.8035%

Morpheus ruled it R9 amendment 5 — *the anti-correlated falsifier*: ask which way a check
moves when its subject is wrong; **if it moves with the reader's confidence it cannot be
repaired by tightening**, and it is demoted from gate to precondition.  §10.0 obligation 8
is the replacement: a device-clock figure is quotable only alongside a **device-state
record over the statistic's own suffix**, carrying a tenancy verdict and clock
min/median/max against the board maximum.  Absent that record the figure is
``STEADY_UNCERTIFIED``.

What this file is, and what it is deliberately not
--------------------------------------------------

It is a **reader and an adjudicator**.  It is not a second record format and it is not a
second producer.  The record it reads is the one ``bench/`` already writes
(``bench/results/probe_gpustate.py`` → ``summarise()``), because Niobe owns ``bench/`` and
``docs/PERF.md`` and a second format would be R11 in its purest form: two names for one
measurement, appearing to close.  The same refusal that kept the CI side on Trinity's
verdict vocabulary applies here.

It is also **not a tool**, per obligation 8 amendment 1.  ``nvidia-smi`` is one vendor's
implementation.  This project is cross-platform by mandate (§1.1), so what is required
here is the record's *content* — tenancy verdict, clock min/median/max, the board's own
advertised maximum, over the statistic's own window — and any platform that can produce
that content satisfies the obligation.  :data:`PRODUCERS` is the per-platform registry of
who could produce it, and it is honest about the platforms where nobody can yet.

The loophole this closes, stated so it cannot be reopened by accident
--------------------------------------------------------------------

Obligation 8 amendment 2: **the absence of the companion is never a waiver.**  Otherwise
the cheapest pass is a platform with no telemetry, where the requirement is vacuous and
the figure comes out unqualified.  Morpheus named the Intel iGPU as that loophole's
biggest beneficiary — it shares its power budget with loaded CPU cores, so it is *more*
exposed to clock bias than the discrete board, not less.  **A CI runner with no GPU
telemetry is the same loophole at scale**, and every GitHub-hosted runner this project
uses is exactly that.

Amendment 3, and it is the one with a code path: a probe that is absent, unparseable or
times out is ``ERROR(instrument)`` and **never** a finding of ``SOLE_TENANT``.  Absence of
evidence and evidence of absence come out of one code path here.

Vocabulary
----------

Not mine.  ``SOLE_TENANT`` / ``FOREIGN_GPU_WORK`` are the producer's
(``bench/results/probe_gpustate.py``).  ``STEADY_UNCERTIFIED`` is Morpheus's, from §10.0
obligation 8.  ``PASS`` / ``FAIL(condition=...)`` / ``ERROR(instrument=...)`` are R13's.
This file contributes no token of its own.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------------------
# Vocabulary — imported in spirit, restated here only because there is no importable
# module that owns these two tokens today.  Both are quoted from their owners.
# --------------------------------------------------------------------------------------

#: Producer's tenancy verdicts (bench/results/probe_gpustate.py :: summarise()).
TENANCY_SOLE = "SOLE_TENANT"
TENANCY_FOREIGN = "FOREIGN_GPU_WORK"
TENANCY_VERDICTS = (TENANCY_SOLE, TENANCY_FOREIGN)

#: DESIGN.md §10.0 obligation 8's fourth state.  Not STEADY, because "steady" is read as
#: "quotable" and the whole finding is that it is not sufficient for that; not ERROR,
#: because the statistic did compute.
STEADY_UNCERTIFIED = "STEADY_UNCERTIFIED"

#: The state a certified record confers.  It carries the tenancy verdict through rather
#: than collapsing it: a figure taken under FOREIGN_GPU_WORK is quotable *as a contended
#: figure*, which is a different claim from a sole-tenant one and must stay different.
CERTIFIED = "CERTIFIED"


# --------------------------------------------------------------------------------------
# The cross-platform producer registry (obligation 8 amendment 1).
#
# Named by CONTENT, not by tool.  A row is `available` only when something on this host
# can emit all four required contents.  `unimplemented` and `none_structural` are both
# honest states and NEITHER is a waiver — a figure taken on such a platform is
# STEADY_UNCERTIFIED forever, and that is a true statement about what we can know there.
# --------------------------------------------------------------------------------------

#: A producer exists on this host and can emit the required content.
STATUS_AVAILABLE = "available"
#: Telemetry exists on this platform but nobody has written the producer yet. Not a waiver.
STATUS_UNIMPLEMENTED = "unimplemented"
#: There is no device clock and no GPU tenancy to observe, because the "device" is the CPU.
#: Not a waiver either — see :func:`lavapipe_note`.
STATUS_NONE_STRUCTURAL = "none_structural"

PRODUCERS = {
    "nvidia": {
        "status": STATUS_AVAILABLE,
        "probe": "nvidia-smi",
        "producer": "bench/results/probe_gpustate.py",
        "note": "The only implemented producer today. One vendor's implementation of the "
        "obligation, not the obligation.",
    },
    "amd": {
        "status": STATUS_UNIMPLEMENTED,
        "probe": "rocm-smi",
        "producer": None,
        "note": "rocm-smi exposes sclk and a process table; no producer written. Figures "
        "taken here are STEADY_UNCERTIFIED until one is.",
    },
    "intel": {
        "status": STATUS_UNIMPLEMENTED,
        "probe": "intel_gpu_top (Linux) / Level Zero sysman (Windows)",
        "producer": None,
        "note": "The iGPU shares its power budget with loaded CPU cores, so it is MORE "
        "exposed to clock bias than the discrete board. Morpheus named it as the "
        "platform the 'no telemetry means no requirement' loophole would have "
        "rewarded most. It is not exempt; it is unmeasured.",
    },
    "apple": {
        "status": STATUS_UNIMPLEMENTED,
        "probe": "powermetrics (requires root)",
        "producer": None,
        "note": "MoltenVK over Metal. powermetrics reports GPU frequency and residency; "
        "the root requirement is a real obstacle on a hosted runner.",
    },
    "adreno": {
        "status": STATUS_UNIMPLEMENTED,
        "probe": "/sys/class/kgsl/kgsl-3d0/{gpuclk,max_gpuclk,gpubusy}",
        "producer": None,
        "note": "Readable without root on most devices. OQ-12 hardware does not exist yet.",
    },
    "mali": {
        "status": STATUS_UNIMPLEMENTED,
        "probe": "/sys/class/devfreq/*.mali/{cur_freq,max_freq}",
        "producer": None,
        "note": "Same as Adreno: the telemetry exists, the hardware does not.",
    },
    "cpu_renderer": {
        "status": STATUS_NONE_STRUCTURAL,
        "probe": None,
        "producer": None,
        "note": "lavapipe / llvmpipe / SwiftShader. There is no SM clock, no board maximum "
        "and no GPU tenancy, because the device IS the host CPU. See lavapipe_note().",
    },
}

#: Substrings of ``VkPhysicalDeviceProperties::deviceName`` (or a driver id) that place a
#: device in the ``cpu_renderer`` row.  Kept as data so a new software rasteriser is one
#: line and not a new branch.
CPU_RENDERER_MARKERS = ("llvmpipe", "lavapipe", "swiftshader", "software rasterizer")


def lavapipe_note() -> str:
    """What a device-state record *means* on a CPU renderer. Written down, not discovered.

    This is the case where "no telemetry, therefore no requirement" is most tempting and
    most wrong, so it gets an answer in prose rather than a silence in a table.

    A software rasteriser has no SM clock, no board maximum and no GPU tenancy in the
    sense obligation 8 uses those words.  Two readings are available and only one of them
    is honest:

    * **The tempting reading.** "There is no device clock to be biased, so the obligation
      does not apply and the figure is unqualified."  This is the waiver amendment 2
      forbids, and it is worse here than anywhere else: lavapipe runs on the *host CPU*,
      which is the single most contended resource on a shared CI runner.  A figure taken
      on lavapipe is not immune to contention bias; it is maximally exposed to it, and
      the exposure is invisible because the usual instrument is pointed at a GPU that
      isn't there.
    * **The honest reading, and the one this project takes.** The obligation's content is
      *the state of the device that produced the timing*.  On a CPU renderer that device
      is the host, so the corresponding record would carry host quiescence, CPU frequency
      min/median/max against the package's advertised maximum, and a host-tenancy verdict
      — which is ``bench/``'s machine-quiescence verdict, not a GPU probe.  No producer
      emits that as an obligation-8 record today.

    **Therefore: lavapipe can never certify a *device-clock* figure, and this is permanent
    rather than pending.** There is no device clock on a CPU renderer; the quantity does
    not exist, so no instrument can be built to record it.  What *is* pending is the
    weaker, different claim — a host-state record for a *wall-clock* figure — and if that
    producer is ever written it certifies wall clock and still never certifies device
    clock.  The two must not be allowed to trade names.

    The practical consequence for the lanes, stated plainly: **all three CI lanes run
    lavapipe, so no CI lane can ever publish a certified device-clock figure.** That is
    not a gap to be closed by a better probe. It is closed by a GPU runner or not at all.
    """
    return lavapipe_note.__doc__ or ""


def host_producer_status() -> dict:
    """What this host can actually produce, decided by probing rather than by assertion.

    Returns a dict with ``status`` and ``rows`` — the registry rows judged available here.
    The absence of every producer is reported as an absence, not as quiet.

    ``ONNXRUNTIME_EP_CI_DEVICE_STATE_PRODUCERS=none`` forces the "no producer on this
    host" branch so it can be exercised on a desk that has one.  It can only ever *remove*
    producers — there is deliberately no value of it that adds one, because a switch that
    could assert telemetry into existence would be the waiver amendment 2 forbids, wearing
    a test harness as a disguise.
    """
    if os.environ.get("ONNXRUNTIME_EP_CI_DEVICE_STATE_PRODUCERS", "").strip().lower() == "none":
        return {
            "status": STATUS_UNIMPLEMENTED,
            "rows": {},
            "available": [],
            "platform": sys.platform,
            "forced": "producers suppressed by ONNXRUNTIME_EP_CI_DEVICE_STATE_PRODUCERS=none",
        }
    rows = {}
    if shutil.which("nvidia-smi"):
        rows["nvidia"] = PRODUCERS["nvidia"]
    if shutil.which("rocm-smi"):
        rows["amd"] = dict(PRODUCERS["amd"])
    if sys.platform.startswith("linux") and shutil.which("intel_gpu_top"):
        rows["intel"] = dict(PRODUCERS["intel"])
    available = {k: v for k, v in rows.items() if v["status"] == STATUS_AVAILABLE}
    return {
        "status": STATUS_AVAILABLE if available else STATUS_UNIMPLEMENTED,
        "rows": rows,
        "available": sorted(available),
        "platform": sys.platform,
    }


# --------------------------------------------------------------------------------------
# Certifying a record against obligation 8.
# --------------------------------------------------------------------------------------

#: The four contents obligation 8 names, mapped onto the producer's existing key names.
#: This is a *reader's* mapping. It adds no key to the producer's format except
#: ``window``, which obligation 8 states in its own words ("the suffix the statistic was
#: computed over, not the run") and which no producer emits yet — so its absence is
#: reported by name rather than assumed away.
REQUIRED_CONTENT = {
    "tenancy": "verdict",
    "clock": "sm_mhz",
    "board_maximum": "sm_max_mhz",
    "window": "window",
}

CLOCK_SUBKEYS = ("min", "median", "max")


def certify(record: "dict | None") -> dict:
    """Decide whether a device-state record makes an accompanying figure quotable.

    Three outcomes and they are R13's, not softer versions of them:

    * ``CERTIFIED`` — every required content present; the tenancy verdict is carried
      through, so a contended figure stays labelled contended.
    * ``STEADY_UNCERTIFIED`` — no record, or a record missing required content. The
      statistic computed; it is simply not quotable. This is not a failure of the
      measurement, it is the measurement having no reading yet.
    * ``ERROR(instrument=...)`` — the probe ran and failed, or the record is unparseable.
      **Never** ``SOLE_TENANT``.
    """
    if record is None:
        return {
            "state": STEADY_UNCERTIFIED,
            "reason": "no_device_state_record",
            "detail": "No device-state record accompanies this figure. §10.0 obligation 8: "
            "absent the record the figure is STEADY_UNCERTIFIED. Absence is never "
            "a waiver (amendment 2).",
        }
    if not isinstance(record, dict):
        return {
            "state": "ERROR",
            "instrument": "device_state_record_unparseable",
            "detail": f"Device-state record is {type(record).__name__}, not an object.",
        }
    if record.get("error"):
        return {
            "state": "ERROR",
            "instrument": "device_state_probe_failed",
            "detail": f"The probe reported: {record['error']!r}. Per obligation 8 "
            "amendment 3 this is ERROR(instrument) and never a finding of "
            f"{TENANCY_SOLE}.",
        }

    missing = [
        name for name, key in REQUIRED_CONTENT.items() if record.get(key) in (None, {}, "")
    ]
    if missing:
        return {
            "state": STEADY_UNCERTIFIED,
            "reason": "incomplete_record",
            "missing": missing,
            "detail": "Device-state record is missing required content: "
            + ", ".join(f"{m} ({REQUIRED_CONTENT[m]})" for m in missing)
            + ". Obligation 8 names the content, so a record that omits any of it does "
            "not make the figure quotable.",
        }

    tenancy = record.get(REQUIRED_CONTENT["tenancy"])
    if tenancy not in TENANCY_VERDICTS:
        return {
            "state": "ERROR",
            "instrument": "device_state_tenancy_unrecognised",
            "detail": f"Tenancy verdict {tenancy!r} is not one of {TENANCY_VERDICTS}. "
            "A verdict this reader cannot interpret is an instrument error, not a "
            "quiet device.",
        }

    clock = record.get(REQUIRED_CONTENT["clock"])
    if not isinstance(clock, dict) or any(clock.get(k) is None for k in CLOCK_SUBKEYS):
        return {
            "state": STEADY_UNCERTIFIED,
            "reason": "incomplete_clock_record",
            "detail": "The clock record must carry min, median and max. A clock number "
            "without its spread is the statistic all over again.",
        }

    board_max = record.get(REQUIRED_CONTENT["board_maximum"])
    try:
        board_max = float(board_max)
    except (TypeError, ValueError):
        board_max = None
    if not board_max:
        return {
            "state": STEADY_UNCERTIFIED,
            "reason": "no_board_maximum",
            "detail": "Obligation 8: a clock number without its ceiling is an index "
            "without its ordering (R11).",
        }

    out = {
        "state": CERTIFIED,
        "tenancy": tenancy,
        "clock_min_mhz": clock["min"],
        "clock_median_mhz": clock["median"],
        "clock_max_mhz": clock["max"],
        "board_max_mhz": board_max,
        "window": record.get("window"),
    }
    try:
        out["clock_at_max_pct"] = round(100.0 * float(clock["median"]) / board_max, 1)
    except (TypeError, ValueError, ZeroDivisionError):
        out["clock_at_max_pct"] = None
    return out


def certifies_comparison(a: "dict | None", b: "dict | None") -> dict:
    """§10.0 obligation 8b — two device-clock figures compare only if their records agree.

    "Agree" is: both certified, same tenancy verdict, and overlapping clock during each
    statistic's own window.  It is explicitly NOT satisfied by both figures being
    ``STEADY``; that both are steady is the whole content of the finding that steadiness
    does not carry this.
    """
    ca, cb = certify(a), certify(b)
    for side, c in (("before", ca), ("after", cb)):
        if c["state"] != CERTIFIED:
            return {
                "comparable": False,
                "reason": f"{side}_not_certified",
                "detail": f"The {side} figure's record is {c['state']}. A before/after "
                "pair whose 'before' predates the companion requirement is not a "
                "pair, and the improvement it would show is UNMEASURED until the "
                "'before' is retaken.",
            }
    if ca["tenancy"] != cb["tenancy"]:
        return {
            "comparable": False,
            "reason": "tenancy_disagrees",
            "detail": f"before={ca['tenancy']} after={cb['tenancy']}.",
        }
    lo = max(float(ca["clock_min_mhz"]), float(cb["clock_min_mhz"]))
    hi = min(float(ca["clock_max_mhz"]), float(cb["clock_max_mhz"]))
    if lo > hi:
        return {
            "comparable": False,
            "reason": "clock_ranges_disjoint",
            "detail": f"before [{ca['clock_min_mhz']}, {ca['clock_max_mhz']}] MHz and "
            f"after [{cb['clock_min_mhz']}, {cb['clock_max_mhz']}] MHz do not overlap.",
        }
    return {"comparable": True, "tenancy": ca["tenancy"], "clock_overlap_mhz": [lo, hi]}


# --------------------------------------------------------------------------------------
# Finding a published duration in a lane artifact.
#
# The point of this half is structural: a lane must be UNABLE to publish a duration
# without the record, rather than merely expected not to.  So the guard runs over the
# lane's own evidence artifacts and reports every timing-shaped quantity it finds.
# --------------------------------------------------------------------------------------

#: Key-name patterns that mark a value as a timing figure, a rate, or a share of one.
#: Criterion 5 is a *share* and a share is a timing figure with a denominator, so shares
#: and ratios are in scope: "run at idle clock, inflate the total, watch the share
#: collapse" is an attack on a share, not on a duration.
DURATION_KEY_PATTERNS = (
    r"(^|_)(ms|us|usec|ns|nsec|sec|secs|s)$",
    r"dur(ation)?",
    r"elapsed",
    r"latenc",
    r"(^|_)time($|_)",
    r"throughput",
    r"per_inference",
    r"(^|_)share($|_)",
    r"speedup",
    r"(^|_)ratio($|_)",
    r"(^|_)rsd($|_)",
    r"busy",
    r"steady",
)

_DURATION_RE = re.compile("|".join(DURATION_KEY_PATTERNS), re.IGNORECASE)

#: The closed, documented exemption set.  Exact key names only — never prefixes, because a
#: prefix exemption grows on its own and this is precisely the kind of list that becomes a
#: loophole if it is allowed to be approximate.  Each entry states why the quantity is not
#: a timing figure.
EXEMPT_KEYS = {
    # A filesystem mtime. An instant, not an interval; it is the attribution witness that
    # proves the profile read was the profile this run wrote.
    "profile_mtime_ns": "filesystem timestamp used as an attribution witness, not an interval",
    # A boolean about whether the §10.0 triple may be quoted at all. Contains "ratio"; it
    # is not one.
    "permits_triple_and_ratio": "boolean admissibility flag, not a measured ratio",
    # Prose naming what the own-provider count means.
    "own_provider_count_means": "explanatory string, not a quantity",
}

#: Keys that belong to the device-state record itself. A record describing its own
#: sampling window is not a lane publishing a duration, and requiring the record to carry
#: a record would be an infinite regress.
RECORD_INTERNAL_KEYS = {"seconds", "wall_s", "n", "clock_ramp_x", "sm_mhz", "window"}

#: Where a document must carry its companion. One key, at the top level of the document
#: that carries the figure, so "did this artifact publish a duration lawfully" is a
#: question with a mechanical answer.
COMPANION_KEY = "device_state"

#: Instrument dumps: files an *instrument* wrote, which the lane reads but does not author.
#: ORT's profiler emits a ``dur`` in microseconds for every node event; the EP's counters
#: snapshot carries two host-side staging counters in microseconds. Neither is a claim by
#: the lane, and requiring an instrument's raw output to carry a device-state companion
#: would make the guard fire on every healthy run, which is the fastest way to teach a
#: reader to ignore it.
#:
#: This is a **closed, code-level list with a reason per entry** and deliberately not a
#: command-line flag. A runtime ``--exclude`` would be a waiver anyone could reach for; an
#: entry here has to be written into this file and past this file's tests. Applying the
#: drafting rule — what is the cheapest thing that satisfies these words without their
#: intent? — the answer is "call your figure an instrument dump", and it is closed two
#: ways: the pattern list cannot be extended at runtime, and figures found inside an
#: instrument dump are **still reported**, as ``STEADY_UNCERTIFIED`` carried-not-claimed,
#: rather than passing silently. The moment a lane *quotes* one of them it appears in a
#: lane artifact or the job summary, where this guard sees it.
INSTRUMENT_DUMPS = (
    (
        "*counters*.json",
        "the EP's own counter snapshot, written by the EP and read by "
        "`epctl --check-counters`; carries session_staging_{upload,readback}_us, which "
        "this project quotes nowhere and which are STEADY_UNCERTIFIED",
    ),
    (
        "*_profile_*.json",
        "ONNX Runtime's profiler output; every node event carries a `dur` in "
        "microseconds. It is the attribution instrument, not a figure of record",
    ),
    (
        "*ort_stderr*.json",
        "captured native stderr, kept as JSON only when a harness wraps it",
    ),
)


def is_instrument_dump(path: Path) -> "str | None":
    """Return the documented reason this file is an instrument dump, or ``None``."""
    for pattern, reason in INSTRUMENT_DUMPS:
        if path.match(pattern):
            return reason
    return None


def find_durations(doc, path: str = "", inside_record: bool = False) -> "list[dict]":
    """Walk a parsed JSON document and return every timing-shaped numeric quantity.

    Only *numbers* count.  A boolean or a string whose key looks timing-shaped is not a
    published figure, and treating it as one would train readers to ignore this check.
    """
    found: "list[dict]" = []
    if isinstance(doc, dict):
        record_here = inside_record or COMPANION_KEY in path.split(".")
        for k, v in doc.items():
            child = f"{path}.{k}" if path else k
            if k == COMPANION_KEY:
                found.extend(find_durations(v, child, inside_record=True))
                continue
            if isinstance(v, (dict, list)):
                found.extend(find_durations(v, child, inside_record=record_here))
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            if k in EXEMPT_KEYS:
                continue
            if record_here and k in RECORD_INTERNAL_KEYS:
                continue
            if _DURATION_RE.search(k):
                found.append({"key": k, "path": child, "value": v})
    elif isinstance(doc, list):
        for i, v in enumerate(doc):
            found.extend(find_durations(v, f"{path}[{i}]", inside_record=inside_record))
    return found


#: A duration in free text — the second witness, with a different failure mode from the
#: JSON walk.  A step that writes "11.525 ms" into the job summary has published a figure
#: just as surely as one that writes it into an artifact, and a JSON parser cannot see it.
SUMMARY_DURATION_RE = re.compile(
    r"(?<![\w.])\d+(?:\.\d+)?\s*(?:ms|milliseconds?|µs|us|microseconds?|ns|nanoseconds?)(?![\w])"
    r"|(?<![\w.])\d+(?:\.\d+)?\s*(?:s|sec|secs|seconds?)/(?:inference|infer|token|run)",
    re.IGNORECASE,
)


def find_summary_durations(text: str) -> "list[str]":
    return SUMMARY_DURATION_RE.findall(text)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def scan_paths(paths: "list[Path]") -> "list[dict]":
    """Every JSON document under ``paths``, with its durations and its companion."""
    out: "list[dict]" = []
    for root in paths:
        files = sorted(root.rglob("*.json")) if root.is_dir() else [root]
        for f in files:
            entry: dict = {"file": str(f)}
            try:
                doc = load_json(f)
            except Exception as exc:  # noqa: BLE001
                entry["error"] = f"{type(exc).__name__}: {exc}"
                out.append(entry)
                continue
            entry["durations"] = find_durations(doc)
            entry["companion"] = doc.get(COMPANION_KEY) if isinstance(doc, dict) else None
            entry["instrument_dump"] = is_instrument_dump(f)
            out.append(entry)
    return out


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")
