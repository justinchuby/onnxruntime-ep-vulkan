"""Phase attribution from the EP's Chrome Trace JSON — where the wall time actually goes.

WHY THIS EXISTS
===============
Until 2026-07-30 every number this harness produced was host wall time with the caveat *"no GPU
kernel time is included: no ``VkQueryPool`` exists yet"*. That caveat is **false as of the
timestamp path landing** (``rust/src/vk/timestamp.rs``, ``GpuQueryPool::cmd_before/cmd_after``
around every ``vkCmdDispatch``), and a stale caveat is worse than no caveat: it is a claim about
the instrument set that a reader will trust. A caveat has to be retired the moment it stops being
true, by the same rule that says a number's caveats travel with it.

So this module reads the trace the EP already writes and splits the wall time into the phases the
EP already labels. It invents nothing: every phase name, every caveat string and every unit comes
out of ``rust/src/trace.rs``. If a phase is not in the trace, it is absent here, not zero.

WHAT KILLED A HYPOTHESIS
========================
The reason this file exists in this shape is that the phase split falsified a plausible inference
of mine. From the first Phi-3.5 measurement I observed Intel paying roughly 2× per island what
NVIDIA paid *while having no bus to cross*, and reasoned towards a fixed per-submission cost —
submit-and-wait per island. I declined to design around it and asked for the instrument. The
instrument said ``vulkan.submit`` is **0.3%** of the run. The cost is host-side command-buffer
recording at ~68%, and the GPU is idle for most of the wall clock. The inference was drawn from
real data and was about the wrong stage; nothing short of a measurement was going to say so.

UNITS, AND ONE TRAP IN THEM
===========================
Chrome Trace ``X`` events carry ``dur`` in **integer microseconds**. GPU spans additionally carry
``gpu_ns`` as a float. Several kernels here run in 2–3 µs, where integer truncation is a 15–30%
error, and there are thousands of them — so **GPU totals are summed from ``gpu_ns``**, never from
``dur``. Host phases are milliseconds-scale and use ``dur``; the truncation there is under 0.01%.

THE INSTRUMENTS THAT GO RED
===========================
Per R9 (``DESIGN.md`` §10.0.1), every number below is paired with something that fails if it is
false. These are checks, not statistics:

* :func:`phase_containment` — every phase span must lie inside a ``vulkan.subgraph`` span, and the
  phases inside one subgraph must not sum to more than the subgraph itself. Goes red if the
  nesting assumption used to attribute spans to islands is wrong.
* :func:`gpu_containment` — per submission, GPU busy time must be ≤ ``submit`` + ``fence_wait``.
  The CPU was blocked on the fence for that long; the GPU cannot have been busy longer. Goes red
  if the tick→nanosecond conversion **over**-scales, which is what a wrongly applied
  ``timestampPeriod`` looks like in the direction that inflates a result.
* :func:`timestamp_conversion_integrality` — the 52× trap, end to end and for the first time.
  ``gpu_ns`` must be an exact integer multiple of the device's ``timestampPeriod``, because it is
  a tick count times that period. A build that dropped the period scale emits raw ticks, and raw
  ticks are integers, so ``gpu_ns / 52.0833`` comes out fractional. **Decisive only where the
  period is not 1.0**, i.e. only on the Intel part — on NVIDIA and on lavapipe the check is
  vacuous and says so rather than passing.
* :func:`trace_matches_counters` — the number of ``vulkan.subgraph`` spans must equal the EP's own
  ``compute_calls``. Goes red if the trace saw a different set of executions than the counters
  did, which is the only way the phase shares could be computed over the wrong denominator.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

#: Host-side phase spans, in the order `rust/src/trace.rs` documents them.
HOST_PHASES = ("compile", "prepack", "record", "upload", "submit", "fence_wait", "readback")

#: `X` (complete) events on the host lane that bound one `Compute` call.
SUBGRAPH = "vulkan.subgraph"

#: Prefix of a device-lane span produced from `VkQueryPool` results.
GPU_PREFIX = "vulkan.gpu."

#: Relative slack allowed before :func:`phase_containment` calls a subgraph over-subscribed.
#: Phases are wall-clock siblings inside one span; they cannot legitimately exceed it. The slack
#: absorbs the microsecond truncation of `dur` across ~7 spans, nothing more.
CONTAINMENT_SLACK = 0.02

#: How close `gpu_ns / timestamp_period_ns` must be to a whole number. The tick count is an
#: integer by construction; this tolerance absorbs float32→float64 widening of the period only.
INTEGRALITY_TOL = 1e-3


def load(path: "str | Path") -> "list[dict]":
    """Read a Chrome Trace JSON array (or an object with ``traceEvents``)."""
    data = json.loads(Path(path).read_text("utf-8"))
    return data["traceEvents"] if isinstance(data, dict) else data


# ---------------------------------------------------------------------------
# Structure — subgraph spans and the phases nested inside them
# ---------------------------------------------------------------------------

def subgraph_spans(events: "list[dict]") -> "list[dict]":
    """Every ``vulkan.subgraph`` span, in start order, with its ``nodes`` count.

    ``nodes`` is the island's dispatch count — ``kernels.len()`` at the call site
    (``vk/session.rs``, ``t.subgraph_region(kernels.len())``). It is the only per-island size the
    trace carries, and it is what question "does recording scale with island size" is asked in.
    """
    out = [
        {
            "ts": e["ts"],
            "dur": e.get("dur", 0),
            "end": e["ts"] + e.get("dur", 0),
            "nodes": (e.get("args") or {}).get("nodes"),
            "tid": e.get("tid"),
        }
        for e in events
        if e.get("ph") == "X" and e.get("name") == SUBGRAPH
    ]
    out.sort(key=lambda s: s["ts"])
    for i, s in enumerate(out):
        s["index"] = i
    return out


def phase_spans(events: "list[dict]") -> "list[dict]":
    """Every ``vulkan.<phase>`` host span, in start order, carrying the EP's own caveat string."""
    names = {f"vulkan.{p}": p for p in HOST_PHASES}
    out = [
        {
            "phase": names[e["name"]],
            "ts": e["ts"],
            "dur": e.get("dur", 0),
            "end": e["ts"] + e.get("dur", 0),
            "caveat": (e.get("args") or {}).get("caveat"),
        }
        for e in events
        if e.get("ph") == "X" and e.get("name") in names
    ]
    out.sort(key=lambda s: s["ts"])
    return out


def gpu_spans(events: "list[dict]") -> "list[dict]":
    """Every device-lane span, with duration taken from ``gpu_ns`` rather than ``dur``.

    ``dur`` is integer microseconds and several of these kernels are 2–3 µs long. Summing ``dur``
    over 3000 such spans under-reports GPU time by a double-digit percentage, silently.
    """
    out = []
    for e in events:
        if e.get("ph") != "X" or not str(e.get("name", "")).startswith(GPU_PREFIX):
            continue
        a = e.get("args") or {}
        out.append({
            "kernel": e["name"][len(GPU_PREFIX):],
            "ts": e["ts"],
            "dur_us_truncated": e.get("dur", 0),
            "gpu_ns": a.get("gpu_ns"),
            "period_ns": a.get("timestamp_period_ns"),
            "valid_bits": a.get("timestamp_valid_bits"),
            "node_index": a.get("node_index"),
            "anchor_uncertainty_us": a.get("anchor_uncertainty_us"),
        })
    out.sort(key=lambda s: s["ts"])
    return out


def attribute(subgraphs: "list[dict]", phases: "list[dict]") -> "list[dict]":
    """Attach each phase span to the subgraph span whose interval contains it.

    Host phases are opened as children of the subgraph guard on the same thread, so containment by
    timestamp is exact rather than heuristic. Spans that land outside every subgraph (``compile``
    and ``prepack`` run at session build time) are returned under ``subgraph_index = None`` — they
    are real time and are not silently dropped.
    """
    bounds = [(s["ts"], s["end"], s["index"]) for s in subgraphs]
    out = []
    for p in phases:
        owner = None
        # subgraphs are disjoint and sorted; a linear scan is fine at these sizes and is
        # obviously correct, which matters more here than the log factor.
        lo, hi = 0, len(bounds) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            a, b, idx = bounds[mid]
            if p["ts"] < a:
                hi = mid - 1
            elif p["ts"] > b:
                lo = mid + 1
            else:
                owner = idx
                break
        q = dict(p)
        q["subgraph_index"] = owner
        q["nodes"] = subgraphs[owner]["nodes"] if owner is not None else None
        out.append(q)
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _summarise(values: "list[float]") -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(values),
        "total_ms": round(sum(values), 3),
        "mean_ms": round(statistics.fmean(values), 4),
        "median_ms": round(statistics.median(ordered), 4),
        "min_ms": round(ordered[0], 4),
        "max_ms": round(ordered[-1], 4),
    }


def host_phase_totals(attributed: "list[dict]") -> dict:
    by: "dict[str, list[float]]" = {}
    for p in attributed:
        by.setdefault(p["phase"], []).append(p["dur"] / 1000.0)
    return {k: _summarise(v) for k, v in by.items()}


def gpu_totals(gpus: "list[dict]") -> dict:
    by: "dict[str, list[float]]" = {}
    for g in gpus:
        ns = g.get("gpu_ns")
        if ns is None:
            continue
        by.setdefault(g["kernel"], []).append(ns / 1e6)
    out = {k: _summarise(v) for k, v in by.items()}
    allv = [ns / 1e6 for g in gpus if (ns := g.get("gpu_ns")) is not None]
    return {"per_kernel": out, "all": _summarise(allv)}


def transfer_events(events: "list[dict]") -> "list[dict]":
    """Individual host transfers, each with its bytes, its recovered duration and its timestamp.

    ``vulkan.transfer_bytes`` and ``vulkan.transfer_gib_s`` are pushed back-to-back from
    ``record_transfer`` (``trace.rs``), per transfer, **not** cumulatively. The duration is not
    written to the trace, but the bandwidth was computed as ``bytes ÷ duration``, so
    ``bytes ÷ bandwidth`` recovers exactly the duration the EP measured rather than estimating it.

    The direction matters for attribution and the two are not symmetric:

    * ``upload`` happens **inside** the ``vulkan.record`` span — ``record_upload`` is called
      between ``vkBeginCommandBuffer`` and the dispatch loop (``vk/session.rs`` step 4).
    * ``readback`` happens **after** the fence, inside the ``vulkan.subgraph`` span but outside
      every phase span.

    So a claim that "recording is the bottleneck" is not yet a claim about command-buffer
    construction: it has to say how much of ``record`` is ``vkCmd*`` calls and how much is a host
    memcpy into staging memory. That split is the difference between an optimisation Switch can
    make in the recording loop and one that requires the weights to stop being re-uploaded.
    """
    out: "list[dict]" = []
    pending: "dict[str, tuple[int, int]]" = {}
    for e in events:
        if e.get("ph") != "C":
            continue
        a = e.get("args") or {}
        if e.get("name") == "vulkan.transfer_bytes":
            for k, v in a.items():
                pending[k] = (int(v), e.get("ts", 0))
        elif e.get("name") == "vulkan.transfer_gib_s":
            for k, v in a.items():
                got = pending.pop(k, None)
                if got is None or not v:
                    continue
                n, ts = got
                out.append({
                    "direction": k,
                    "bytes": n,
                    "ts": ts,
                    "us": n / (float(v) * (1024 ** 3)) * 1e6,
                    "gib_s": float(v),
                })
    out.sort(key=lambda t: t["ts"])
    return out


def transfer_totals(transfers: "list[dict]") -> dict:
    bytes_by: "dict[str, int]" = {}
    us_by: "dict[str, float]" = {}
    n_by: "dict[str, int]" = {}
    for t in transfers:
        d = t["direction"]
        bytes_by[d] = bytes_by.get(d, 0) + t["bytes"]
        us_by[d] = us_by.get(d, 0.0) + t["us"]
        n_by[d] = n_by.get(d, 0) + 1
    return {
        d: {
            "n": n_by[d],
            "bytes": bytes_by[d],
            "mib": round(bytes_by[d] / 1024 ** 2, 1),
            "host_copy_ms": round(us_by[d] / 1000.0, 3),
            "mean_gib_s": (round(bytes_by[d] / (1024 ** 3) / (us_by[d] / 1e6), 4)
                           if us_by[d] else None),
            "inside_record_span": d == "upload",
        }
        for d in sorted(bytes_by)
    }


def partition_stats(events: "list[dict]") -> dict:
    """The EP's ``PartitionStats``, as emitted on the ``vulkan.getcapability`` instant event.

    §10.0's metric of record is the triple ``(claimed_op_coverage, island_count,
    largest_island_flops)``. The third slot is now *plumbed* — the event carries the key — but the
    EP fills it with zero, because ``CoverageReport`` never computes it. ``0`` and "not computed"
    are different states and this reports which one it is rather than passing a zero upward as if
    it were a measurement.
    """
    ev = next((e for e in events
               if e.get("ph") == "i" and e.get("name") == "vulkan.getcapability"), None)
    if ev is None:
        return {"present": False, "reason": "no vulkan.getcapability event in the trace"}
    a = dict(ev.get("args") or {})
    declined = {k[len("declined_"):]: v for k, v in a.items() if k.startswith("declined_")}
    core = {k: v for k, v in a.items() if not k.startswith("declined_")}
    flops = core.get("largest_island_flops")
    islands = core.get("island_count")
    return {
        "present": True,
        "claimed_nodes": core.get("claimed_nodes"),
        "island_count": islands,
        "largest_island_flops": flops,
        "concentration": core.get("concentration"),
        "boundary_bytes_per_inference": core.get("boundary_bytes_per_inference"),
        "boundary_time_fraction": core.get("boundary_time_fraction"),
        "declined_kinds": len(declined),
        "third_slot_state": (
            "UNPOPULATED" if not flops else "populated"),
        "third_slot_note": (
            "the key is emitted and the value is 0. CoverageReport does not compute a FLOP "
            "estimate, so this is 'not computed' wearing the appearance of 'zero FLOPs'. The "
            "metric of record's third slot is still empty and must not be quoted as 0."
            if not flops else None),
        "island_count_state": (
            "UNPOPULATED — GetCapability reports 0 islands while the EP's own subgraphs_live "
            "counter reports a non-zero count; the same struct is unfilled"
            if not islands else "populated"),
    }


# ---------------------------------------------------------------------------
# Falsifiers
# ---------------------------------------------------------------------------

def phase_containment(subgraphs: "list[dict]", attributed: "list[dict]") -> dict:
    """Every phase span inside a subgraph, and phases never summing past their subgraph."""
    orphans = [p for p in attributed
               if p["subgraph_index"] is None and p["phase"] not in ("compile", "prepack")]
    per: "dict[int, float]" = {}
    for p in attributed:
        if p["subgraph_index"] is not None:
            per[p["subgraph_index"]] = per.get(p["subgraph_index"], 0) + p["dur"]
    over = [
        {"subgraph_index": i, "phases_us": per[i], "subgraph_us": subgraphs[i]["dur"]}
        for i in per
        if subgraphs[i]["dur"] > 0 and per[i] > subgraphs[i]["dur"] * (1 + CONTAINMENT_SLACK)
    ]
    return {
        "red": bool(orphans or over),
        "orphan_phase_spans": len(orphans),
        "over_subscribed_subgraphs": len(over),
        "examples": over[:3],
        "detail": ("ok: every phase span lies inside a subgraph span and no subgraph's phases "
                   "sum past it"
                   if not (orphans or over) else
                   f"RED: {len(orphans)} phase spans outside any subgraph, "
                   f"{len(over)} subgraphs whose phases exceed their own duration. The "
                   f"attribution of phases to islands is not sound and no phase share below may "
                   f"be read."),
    }


def attribute_gpu_ordinally(subgraphs: "list[dict]", gpus: "list[dict]") -> dict:
    """Attach GPU spans to submissions **by order and count**, never by timestamp.

    A device-lane span's ``ts`` is not a host timestamp. It is a GPU tick converted to the host
    timeline through a single calibration anchor, and the trace tells you how bad that placement
    is: ``anchor_uncertainty_us`` reaches **314 ms** on the Intel part. Deciding which submission
    a 2 µs kernel belongs to by asking whose ``[ts, end]`` interval contains it is therefore a
    coin flip at the boundaries — and it produced 14 phantom "GPU busier than its own fence"
    violations on a build whose conversion arithmetic was independently proven correct.

    Ordinal attribution needs no clock. Each ``vulkan.subgraph`` dispatches exactly ``nodes``
    kernels and writes their query results in order, so walking the two lists in lockstep is
    exact — *provided* the counts line up, which :func:`gpu_span_accounting` asserts as integer
    equality before any of this is believed.
    """
    busy: "dict[int, float]" = {}
    counts: "dict[int, int]" = {}
    cursor = 0
    exhausted = False
    for s in subgraphs:
        n = s.get("nodes")
        if not isinstance(n, int) or n < 0:
            continue
        take = gpus[cursor:cursor + n]
        if len(take) < n:
            exhausted = True
        cursor += len(take)
        total = 0.0
        for g in take:
            ns = g.get("gpu_ns")
            if ns is not None:
                total += ns / 1000.0
        busy[s["index"]] = total
        counts[s["index"]] = len(take)
    return {
        "busy_us": busy,
        "spans_per_submission": counts,
        "consumed": cursor,
        "left_over": len(gpus) - cursor,
        "exhausted_early": exhausted,
    }


def gpu_span_accounting(subgraphs: "list[dict]", gpus: "list[dict]",
                        counters: "dict | None" = None) -> dict:
    """Integer equality, no tolerance: one GPU span per dispatch, for every dispatch.

    ``sum(subgraph.nodes) == len(gpu_spans)``. This is the precondition ordinal attribution rests
    on, and it is also the check that catches a kernel whose query was never written — the failure
    mode that raises nothing, leaves ``compute_failures`` at 0 (§9.1.3), and simply removes time
    from the GPU column. Where the counters file reports ``dispatches_executed`` the equality is
    extended to it, so a trace agreeing with itself is not mistaken for a trace agreeing with the
    run.
    """
    expected = sum(s["nodes"] for s in subgraphs if isinstance(s.get("nodes"), int))
    observed = len(gpus)
    dispatches = None
    if counters:
        v = counters.get("dispatches_executed")
        if isinstance(v, int):
            dispatches = v

    parts = [("sum(subgraph.nodes)", expected), ("gpu spans", observed)]
    if dispatches is not None:
        parts.append(("dispatches_executed", dispatches))
    values = {v for _, v in parts}
    red = len(values) > 1

    return {
        "red": red,
        "asserts": " == ".join(f"{k}({v})" for k, v in parts),
        "expected_from_nodes": expected,
        "observed_gpu_spans": observed,
        "dispatches_executed": dispatches,
        "detail": (
            f"ok: {' == '.join(str(v) for _, v in parts)} — every dispatch produced exactly one "
            "timestamped span, so ordinal attribution is exact"
            if not red else
            "RED: " + ", ".join(f"{k}={v}" for k, v in parts) +
            " disagree. Some dispatch produced no GPU span (or produced two); GPU time is "
            "under-reported and ordinal attribution is not trustworthy."
        ),
    }


def device_fingerprint(gpus: "list[dict]") -> dict:
    """The device identity the trace itself carries: ``timestampPeriod`` and ``timestampValidBits``.

    This exists because ``ep.device_index`` and ``vkEnumeratePhysicalDevices`` index are two
    different orderings of the same devices (``engine.rs::probe_devices`` is sorted best-first),
    so a results row labelled from the enumeration index can name the wrong GPU. The trace knows
    which device ran it; the label does not. ``devices.device_identity_check`` compares the two.
    """
    periods = sorted({g["period_ns"] for g in gpus if g.get("period_ns") is not None})
    bits = sorted({g["valid_bits"] for g in gpus if g.get("valid_bits") is not None})
    return {
        "timestamp_period_ns": periods[0] if len(periods) == 1 else None,
        "timestamp_valid_bits": bits[0] if len(bits) == 1 else None,
        "periods_seen": periods,
        "valid_bits_seen": bits,
        "consistent": len(periods) <= 1 and len(bits) <= 1,
        "detail": (
            "no GPU spans carry a timestamp fingerprint" if not periods and not bits else
            f"period={periods} bits={bits}"
            + ("" if len(periods) <= 1 and len(bits) <= 1
               else " — MORE THAN ONE DEVICE'S CALIBRATION IN ONE TRACE")
        ),
    }


def anchor_quality(gpus: "list[dict]") -> dict:
    """How far a device-lane span could be from where it is drawn. Not an error — a caveat.

    Reported so nobody reads the GPU lane's *placement* as evidence of overlap or serialisation.
    Durations are unaffected; only positions are.
    """
    unc = [g["anchor_uncertainty_us"] for g in gpus if g.get("anchor_uncertainty_us") is not None]
    if not unc:
        return {"available": False,
                "detail": "trace carries no anchor_uncertainty_us; span placement unquantified"}
    unc.sort()
    worst = unc[-1]
    return {
        "available": True,
        "max_us": round(worst, 1),
        "median_us": round(unc[len(unc) // 2], 1),
        "detail": (
            f"device-lane spans may sit up to {worst / 1000.0:.1f} ms from their true host "
            "position. Durations are unaffected; positions are not evidence. Attribution is "
            "ordinal for exactly this reason."
        ),
    }


def gpu_containment(subgraphs: "list[dict]", attributed: "list[dict]",
                    gpus: "list[dict]") -> dict:
    """GPU busy time per submission must not exceed the CPU's own block on the fence.

    ``submit + fence_wait`` is the interval during which the submission could possibly have been
    executing. Anything longer means the tick→nanosecond conversion **over**-scaled — the failure
    direction a wrongly applied ``timestampPeriod`` produces when it is applied twice, or applied
    on a device that did not need it.

    Attribution is ordinal (see :func:`attribute_gpu_ordinally`); an earlier timestamp-containment
    version of this check reported 14 violations that were entirely artefacts of a 314 ms
    calibration anchor uncertainty, on a build that passed
    :func:`timestamp_conversion_integrality` decisively.
    """
    window: "dict[int, float]" = {}
    for p in attributed:
        if p["subgraph_index"] is not None and p["phase"] in ("submit", "fence_wait"):
            window[p["subgraph_index"]] = window.get(p["subgraph_index"], 0) + p["dur"]
    ordinal = attribute_gpu_ordinally(subgraphs, gpus)
    busy = ordinal["busy_us"]
    violations = [
        {"subgraph_index": i, "gpu_us": round(busy[i], 1), "window_us": window.get(i, 0)}
        for i in sorted(busy)
        if busy[i] > window.get(i, 0) * 1.05 + 50
    ]
    return {
        "red": bool(violations),
        "attribution": "ordinal (count-based); timestamps are not used to assign spans",
        "submissions_with_gpu_spans": sum(1 for v in busy.values() if v > 0),
        "violations": len(violations),
        "examples": violations[:3],
        "detail": ("ok: GPU busy time never exceeds submit+fence_wait for the same submission"
                   if not violations else
                   f"RED: {len(violations)} submissions report more GPU time than the CPU spent "
                   f"blocked on their fence. The tick→ns conversion over-scales."),
    }


def timestamp_conversion_integrality(gpus: "list[dict]") -> dict:
    """The 52× trap, checked end to end on real ticks for the first time.

    ``gpu_ns`` is a tick count multiplied by ``timestampPeriod``. So ``gpu_ns ÷ period`` must be a
    whole number. A build that dropped the period scale reports raw ticks as nanoseconds; dividing
    those by 52.0833 gives a fraction, and this goes red.

    It is **only decisive where the period is not 1.0.** On NVIDIA and on lavapipe the period is
    exactly 1.0, every value trivially divides, and the check cannot fail — reported as
    ``VACUOUS``, never as a pass. That asymmetry is the whole reason the Intel part is the only
    instrument on this desk for this bug class and CI has none.
    """
    by_period: "dict[float, list[dict]]" = {}
    for g in gpus:
        p = g.get("period_ns")
        if p is None or g.get("gpu_ns") is None:
            continue
        by_period.setdefault(float(p), []).append(g)
    out: dict = {"periods_seen": sorted(by_period), "red": False, "decisive": False,
                 "results": []}
    for period, group in sorted(by_period.items()):
        if abs(period - 1.0) < 1e-9:
            out["results"].append({
                "period_ns": period,
                "n": len(group),
                "verdict": "VACUOUS",
                "detail": "timestampPeriod is 1.0: raw ticks and nanoseconds are numerically "
                          "identical, so dropping the scale is undetectable on this device. Not "
                          "a pass.",
            })
            continue
        bad = []
        for g in group:
            ticks = g["gpu_ns"] / period
            if abs(ticks - round(ticks)) > INTEGRALITY_TOL:
                bad.append({"gpu_ns": g["gpu_ns"], "implied_ticks": ticks,
                            "kernel": g["kernel"]})
        out["decisive"] = True
        out["red"] = out["red"] or bool(bad)
        out["results"].append({
            "period_ns": period,
            "n": len(group),
            "non_integral": len(bad),
            "examples": bad[:3],
            "verdict": "PASS" if not bad else "RED",
            "detail": (f"every one of {len(group)} GPU durations is an exact multiple of "
                       f"{period} ns/tick, so the period scale is applied end to end"
                       if not bad else
                       f"{len(bad)} of {len(group)} GPU durations are not multiples of {period} "
                       f"ns/tick. The conversion is not applying the device period — every Intel "
                       f"duration is under-reported by {period:.4g}×."),
        })
    if not out["decisive"]:
        out["detail"] = ("no device in this trace reports timestampPeriod != 1.0, so the 52× trap "
                         "cannot be falsified by this run. That is a gap in the instrument set, "
                         "not a pass — run the Intel part.")
    return out


def valid_bits_applied(gpus: "list[dict]") -> dict:
    """A masked read cannot produce a duration wider than the counter's own wrap period."""
    worst = None
    bad = []
    for g in gpus:
        vb, p, ns = g.get("valid_bits"), g.get("period_ns"), g.get("gpu_ns")
        if not vb or not p or ns is None or vb >= 64:
            continue
        wrap_ns = (1 << vb) * p
        if ns >= wrap_ns:
            bad.append({"gpu_ns": ns, "wrap_ns": wrap_ns, "kernel": g["kernel"]})
        worst = max(worst or 0.0, ns / wrap_ns)
    return {
        "red": bool(bad),
        "checked": sum(1 for g in gpus if 0 < (g.get("valid_bits") or 64) < 64),
        "max_fraction_of_wrap_period": round(worst, 8) if worst is not None else None,
        "examples": bad[:3],
        "detail": ("no device in this trace reports validBits < 64; the mask is not exercisable "
                   "here" if worst is None else
                   "ok: every duration is below the counter's wrap period, consistent with the "
                   "mask being applied" if not bad else
                   "RED: a duration exceeds the wrap period, which an unmasked read produces."),
    }


def trace_matches_counters(subgraphs: "list[dict]", counters: "dict | None") -> dict:
    """The trace and the counters must have watched the same executions.

    Integer equality, no tolerance, in the shape of ``dispatch_accounting``: if the trace holds
    fewer ``vulkan.subgraph`` spans than the EP counted ``compute_calls``, then the phase shares
    below were computed over a subset of the run and the denominator is wrong.
    """
    calls = (counters or {}).get("compute_calls")
    n = len(subgraphs)
    if calls is None:
        return {"red": None, "subgraph_spans": n, "compute_calls": None,
                "detail": "not checkable: no counters file"}
    return {
        "red": n != calls,
        "subgraph_spans": n,
        "compute_calls": calls,
        "detail": (f"ok: {n} vulkan.subgraph spans == compute_calls {calls}" if n == calls else
                   f"RED: {n} vulkan.subgraph spans != compute_calls {calls}. The trace did not "
                   f"observe the same executions the counters did; every phase share is over the "
                   f"wrong denominator."),
    }


# ---------------------------------------------------------------------------
# Question 4 — does recording scale with island size, and is the decline real?
# ---------------------------------------------------------------------------

def record_scaling(attributed: "list[dict]", subgraphs: "list[dict]",
                   transfers: "list[dict]") -> dict:
    """Two separate questions about ``vulkan.record``, kept separate because they confound.

    **(a) Does recording cost scale with island size?** Group record spans by the enclosing
    subgraph's ``nodes`` (dispatch count). Two things are needed before that comparison means
    anything, and skipping either produces a confident wrong answer:

    * **Warmup must be excluded.** The first inference's record spans are 4× the rest. Pooled over
      all inferences, the within-group variance from warmup swamps the between-group difference
      and a rank correlation comes out near zero — which reads as "does not scale" and is an
      artifact of the first inference.
    * **Bytes must be separated from dispatches.** The upload memcpy happens inside the record
      span, so a bigger island that also uploads more bytes will record slower for a reason that
      has nothing to do with ``vkCmd*`` calls. ``record_minus_upload`` is the residual — the part
      of recording that is actually command-buffer construction — and the size question is asked
      of *that*, as well as of the raw span.

    **(b) Is the decline across inferences real, or an artifact of which islands run when?**
    The pooled figure — first *k* record spans against the last *k* — cannot tell those apart,
    because the islands run in a fixed order and the first *k* spans *are* one whole inference. So
    the decline is re-measured **within each island position**: for each island, compare its own
    first occurrence against its own last, and compare per-cycle means, where every cycle contains
    exactly the same set of islands. A decline that survives that cannot be an island-mix effect.

    ``islands_per_inference`` is recovered from the run's own repeat structure rather than assumed:
    the record spans form a cycle over island node-counts, and the period is the smallest that
    repeats. Taking it from ``subgraphs_live`` would import the very thing this is used to
    cross-check.
    """
    rec = [p for p in attributed if p["phase"] == "record" and p["subgraph_index"] is not None]
    rec.sort(key=lambda p: p["ts"])
    out: dict = {"n": len(rec)}
    if not rec:
        out["usable"] = False
        return out

    # Attribute each upload to the record span whose interval contains it.
    up_us: "dict[int, float]" = {}
    up_bytes: "dict[int, int]" = {}
    starts = [p["ts"] for p in rec]
    for t in transfers:
        if t["direction"] != "upload":
            continue
        lo, hi = 0, len(rec) - 1
        owner = None
        while lo <= hi:
            mid = (lo + hi) // 2
            if t["ts"] < rec[mid]["ts"]:
                hi = mid - 1
            elif t["ts"] > rec[mid]["end"]:
                lo = mid + 1
            else:
                owner = mid
                break
        if owner is not None:
            up_us[owner] = up_us.get(owner, 0.0) + t["us"]
            up_bytes[owner] = up_bytes.get(owner, 0) + t["bytes"]
    del starts

    rows = []
    for i, p in enumerate(rec):
        total = p["dur"] / 1000.0
        upl = up_us.get(i, 0.0) / 1000.0
        rows.append({
            "i": i,
            "nodes": p["nodes"],
            "record_ms": total,
            "upload_ms": upl,
            "upload_bytes": up_bytes.get(i, 0),
            "record_minus_upload_ms": max(total - upl, 0.0),
        })

    durs = [r["record_ms"] for r in rows]
    out["usable"] = True
    out.update(_summarise(durs))
    out["upload_inside_record_ms"] = round(sum(r["upload_ms"] for r in rows), 3)
    out["upload_share_of_record"] = (round(out["upload_inside_record_ms"] / out["total_ms"], 4)
                                     if out.get("total_ms") else None)
    out["command_construction_ms"] = round(sum(r["record_minus_upload_ms"] for r in rows), 3)
    out["composition_note"] = (
        "vulkan.record is not one activity. It brackets vkBeginCommandBuffer through "
        "vkEndCommandBuffer, and the host memcpy of this island's inputs into staging buffers "
        "happens inside it. The two are reported separately because they have different fixes: "
        "the memcpy goes away if the data stops being re-copied per Compute call; the residual "
        "is the recording loop itself.")

    period = _cycle_period([p["nodes"] for p in rec])
    out["islands_per_inference_inferred"] = period
    cycles = (len(rec) // period) if period else 0
    # Steady state = everything after the first inference. Every size comparison is made here,
    # because the first inference's spans are a different population.
    steady = rows[period:] if (period and cycles >= 2) else rows
    out["steady_state_basis"] = ("excludes the first inference"
                                 if (period and cycles >= 2) else
                                 "ALL spans — fewer than two full inferences, warmup not separable")
    out["steady_state_record"] = _summarise([r["record_ms"] for r in steady])
    out["steady_state_command_construction"] = _summarise(
        [r["record_minus_upload_ms"] for r in steady])

    # (a) size scaling, on steady state, for both the raw span and the residual
    def _by(key: str) -> dict:
        g: "dict[int, list[float]]" = {}
        for r in steady:
            if r["nodes"] is not None:
                g.setdefault(int(r["nodes"]), []).append(r[key])
        return {str(k): _summarise(v) for k, v in sorted(g.items())}

    out["by_island_size_record"] = _by("record_ms")
    out["by_island_size_command_construction"] = _by("record_minus_upload_ms")
    out["by_island_size_upload_bytes"] = {
        str(k): round(statistics.fmean(v), 1)
        for k, v in sorted({
            int(r["nodes"]): [x["upload_bytes"] for x in steady if x["nodes"] == r["nodes"]]
            for r in steady if r["nodes"] is not None
        }.items())
    }
    # If recording is a byte-throughput cost rather than a per-dispatch cost, the implied
    # bandwidth is the same for every island size. That is a much stronger discriminator than a
    # rank correlation on two points, because it predicts a *value*, not an ordering.
    out["implied_record_gib_s_by_island_size"] = {}
    for k, mean_bytes in out["by_island_size_upload_bytes"].items():
        med = (out["by_island_size_record"].get(k) or {}).get("median_ms")
        if med:
            out["implied_record_gib_s_by_island_size"][k] = round(
                (mean_bytes / 1024 ** 3) / (med / 1000.0), 4)
    vals = list(out["implied_record_gib_s_by_island_size"].values())
    out["implied_bandwidth_spread"] = (round(max(vals) / min(vals), 4)
                                       if len(vals) >= 2 and min(vals) > 0 else None)
    sizes = [int(r["nodes"]) for r in steady if r["nodes"] is not None]
    out["size_rank_correlation_record"] = _spearman(
        sizes, [r["record_ms"] for r in steady if r["nodes"] is not None])
    out["size_rank_correlation_command_construction"] = _spearman(
        sizes, [r["record_minus_upload_ms"] for r in steady if r["nodes"] is not None])
    out["bytes_rank_correlation_record"] = _spearman(
        [r["upload_bytes"] for r in steady], [r["record_ms"] for r in steady])
    out["distinct_island_sizes"] = len(set(sizes))

    if out["distinct_island_sizes"] < 2:
        out["size_verdict"] = "UNDECIDABLE"
        out["size_detail"] = (
            "every island in this run has the same dispatch count, so island size does not vary "
            "and this run cannot say whether recording scales with it. Not a refutation.")
    else:
        med_r = {k: v["median_ms"] for k, v in out["by_island_size_record"].items()}
        med_c = {k: v["median_ms"] for k, v in
                 out["by_island_size_command_construction"].items()}
        ks = sorted(int(k) for k in med_r)
        span_r = (med_r[str(ks[-1])] / med_r[str(ks[0])]) if med_r[str(ks[0])] else None
        size_span = ks[-1] / ks[0] if ks[0] else None
        out["size_span"] = size_span
        out["median_ratio_largest_over_smallest"] = round(span_r, 3) if span_r else None
        rb = out["bytes_rank_correlation_record"]
        rs = out["size_rank_correlation_record"]
        # A per-dispatch cost would make the median ratio track the size ratio. A per-submission
        # cost would leave it flat. Anything in between is stated as such rather than rounded to
        # one of the two.
        if span_r is None or size_span is None:
            out["size_verdict"] = "UNDECIDABLE"
        elif span_r >= 0.7 * size_span:
            out["size_verdict"] = "SCALES_WITH_DISPATCH_COUNT"
        elif span_r <= 1.3:
            out["size_verdict"] = "DOES_NOT_SCALE_WITH_DISPATCH_COUNT"
        else:
            out["size_verdict"] = "SUBLINEAR"
        out["size_detail"] = (
            f"steady-state medians by dispatch count: record {med_r} ms, of which command "
            f"construction {med_c} ms; mean upload bytes per island size "
            f"{out['by_island_size_upload_bytes']}. Dispatch count spans {size_span:.2g}×, "
            f"record median spans {span_r:.2g}×. Spearman(dispatches, record) = {rs}; "
            f"Spearman(upload bytes, record) = {rb}. Implied record bandwidth by island size "
            f"{out['implied_record_gib_s_by_island_size']} GiB/s (spread "
            f"{out['implied_bandwidth_spread']}×). Only "
            f"{out['distinct_island_sizes']} distinct island sizes are present in this "
            f"partition, so this is a two-point comparison, not a fitted curve.")
        if (out["implied_bandwidth_spread"] is not None
                and out["implied_bandwidth_spread"] <= 1.25):
            out["size_verdict"] = "SCALES_WITH_BYTES_NOT_DISPATCHES"
            out["size_confound"] = (
                f"the implied bandwidth is the same to within "
                f"{out['implied_bandwidth_spread']}× across island sizes, while the command "
                f"construction residual spans only "
                f"{(max(v['median_ms'] for v in out['by_island_size_command_construction'].values()) / min(v['median_ms'] for v in out['by_island_size_command_construction'].values())):.2g}× "
                f"for a {size_span:.2g}× change in dispatch count. Recording cost is a "
                f"byte-throughput cost, not a per-dispatch cost: bigger islands record slower "
                f"here because they carry more input bytes, and the apparent size effect is a "
                f"bytes effect wearing a size label.")
        elif rb is not None and rs is not None and rb > rs:
            out["size_confound"] = (
                "record duration is more strongly rank-correlated with uploaded bytes than with "
                "dispatch count. The apparent size effect is at least partly a bytes effect: "
                "bigger islands here also carry more input bytes.")

    # (b) decline across inferences, controlled for island identity
    if not period or cycles < 2:
        out["decline_verdict"] = "UNDECIDABLE"
        out["decline_detail"] = ("could not recover the island cycle from the record sequence, or "
                                 "fewer than two full inferences are present.")
        return out
    pooled_first = statistics.fmean(durs[:period])
    pooled_last = statistics.fmean(durs[-period:])
    out["pooled_first_cycle_mean_ms"] = round(pooled_first, 4)
    out["pooled_last_cycle_mean_ms"] = round(pooled_last, 4)
    out["pooled_decline_ratio"] = round(pooled_last / pooled_first, 4) if pooled_first else None

    per_island_ratios = []
    for slot in range(period):
        series = durs[slot::period][:cycles]
        if len(series) >= 2 and series[0] > 0:
            per_island_ratios.append(series[-1] / series[0])
    out["per_island_last_over_first"] = {
        "n": len(per_island_ratios),
        "median": round(statistics.median(per_island_ratios), 4) if per_island_ratios else None,
        "fraction_declining": (round(sum(1 for r in per_island_ratios if r < 0.9)
                                     / len(per_island_ratios), 4)
                               if per_island_ratios else None),
    }
    cycle_means = [statistics.fmean(durs[c * period:(c + 1) * period]) for c in range(cycles)]
    out["cycle_means_ms"] = [round(x, 4) for x in cycle_means]
    out["cycle_count"] = cycles
    # The question Switch actually needs answered is not "does it decline" but "where does it
    # settle". A decline that has flattened by cycle 3 is a fixed one-off cost; one still falling
    # at the last cycle means the steady state has not been reached and the number is provisional.
    tail = cycle_means[max(1, cycles - 3):]
    out["tail_spread_ratio"] = (round(max(tail) / min(tail), 4)
                                if len(tail) >= 2 and min(tail) > 0 else None)
    out["flattened"] = (out["tail_spread_ratio"] is not None
                        and out["tail_spread_ratio"] <= 1.25)
    # "Warmup" and "gradual approach to a steady state" are not the same shape, and the
    # difference decides whether Switch is looking at a one-off cost or a curve. Count how many
    # cycles it takes to first come within 25% of the tail median: 1 means the very first
    # inference is the only expensive one and everything after it is steady state.
    tail_median = statistics.median(tail)
    out["tail_median_ms"] = round(tail_median, 4)
    cycles_to_steady = next(
        (c + 1 for c, m in enumerate(cycle_means) if abs(m - tail_median) <= 0.25 * tail_median),
        None)
    out["cycles_to_steady"] = cycles_to_steady
    out["warmup_shape"] = (
        "ONE_OFF_FIRST_INFERENCE" if cycles_to_steady == 2 else
        "NO_WARMUP" if cycles_to_steady == 1 else
        f"RAMP_OVER_{cycles_to_steady}_INFERENCES" if cycles_to_steady else "NEVER_SETTLES")
    out["warmup_shape_note"] = (
        "cycles_to_steady is the first inference whose mean record cost is within 25% of the "
        "median of the last three. A value of 2 means the cost is a one-off paid on inference 1 "
        "and everything after it is steady state — a fixed cost to amortise, not a curve to fit.")
    frac = out["per_island_last_over_first"]["fraction_declining"]
    if cycles < 3:
        out["decline_verdict"] = "UNDECIDABLE"
        out["decline_detail"] = ("fewer than three inferences; a trend cannot be separated from "
                                 "noise.")
    elif frac is not None and frac >= 0.6 and cycle_means[-1] < cycle_means[0] * 0.9:
        out["decline_verdict"] = "REAL_WARMUP"
        out["decline_detail"] = (
            f"the decline holds within island identity: {frac:.0%} of islands record faster on "
            f"their last inference than on their first, and the per-cycle means fall across "
            f"cycles that each contain exactly the same {period} islands in the same order. It "
            f"is a warmup effect, not an island-mix artifact. Shape: {out['warmup_shape']} "
            f"(steady within 25% of the tail median {out['tail_median_ms']} ms from inference "
            f"{out['cycles_to_steady']}). "
            + (f"It has flattened: the last three cycle means span "
               f"{out['tail_spread_ratio']}×, so the steady state is reached and "
               f"{out['steady_state_record']['median_ms']} ms is the median Switch is optimising "
               f"against." if out["flattened"] else
               f"It has NOT flattened: the last three cycle means still span "
               f"{out['tail_spread_ratio']}×, so the steady state is not established and the "
               f"steady-state median is provisional."))
    elif cycle_means[-1] >= cycle_means[0] * 0.9:
        out["decline_verdict"] = "NO_DECLINE"
        out["decline_detail"] = (
            f"per-cycle means {[round(x, 1) for x in cycle_means]} ms do not fall; the pooled "
            f"first-vs-last comparison was reading island identity, not time.")
    else:
        out["decline_verdict"] = "MIXED"
        out["decline_detail"] = (
            f"per-cycle means fall ({[round(x, 1) for x in cycle_means]} ms) but only "
            f"{frac:.0%} of individual islands decline, so the pooled figure is partly island "
            f"mix and partly warmup. Both effects are present and neither dominates.")
    return out


def _spearman(xs: "list[float]", ys: "list[float]") -> "float | None":
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return round(num / den, 4) if den else None


def _ranks(vs: "list[float]") -> "list[float]":
    order = sorted(range(len(vs)), key=lambda i: vs[i])
    out = [0.0] * len(vs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vs[order[j + 1]] == vs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def _cycle_period(seq: "list") -> "int | None":
    """Smallest ``p`` such that ``seq[i] == seq[i + p]`` for every valid ``i``.

    The record sequence repeats the island order once per inference, so ``p`` is the island count
    — recovered from the trace's own structure rather than taken from a counter, so that
    :func:`record_scaling` stays independent of the counter it is used to cross-check.
    """
    n = len(seq)
    if n < 2:
        return None
    for p in range(1, n // 2 + 1):
        if n % p:
            continue
        if all(seq[i] == seq[i % p] for i in range(n)):
            return p
    return None


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def analyse(events: "list[dict]", counters: "dict | None" = None) -> dict:
    """The whole phase picture for one trace, with its falsifiers attached."""
    subs = subgraph_spans(events)
    attributed = attribute(subs, phase_spans(events))
    gpus = gpu_spans(events)
    transfers = transfer_events(events)
    host = host_phase_totals(attributed)
    gpu = gpu_totals(gpus)

    # The denominator for "share of what". The subgraph spans are the EP's own view of the time
    # ORT spent inside Compute; it is not the process wall clock and is not labelled as such.
    in_compute_ms = sum(s["dur"] for s in subs) / 1000.0
    shares = {
        k: round(v["total_ms"] / in_compute_ms, 5)
        for k, v in host.items() if v.get("n") and in_compute_ms > 0
    }
    if gpu["all"].get("n") and in_compute_ms > 0:
        shares["gpu_kernels"] = round(gpu["all"]["total_ms"] / in_compute_ms, 5)

    phased_ms = sum(v["total_ms"] for v in host.values() if v.get("n"))
    scaling = record_scaling(attributed, subs, transfers)

    return {
        "subgraph_spans": len(subs),
        "time_in_compute_ms": round(in_compute_ms, 3),
        "time_in_compute_note": (
            "the sum of vulkan.subgraph spans — the EP's own view of time inside Compute. It is "
            "NOT the process wall clock: ORT's own graph execution, the CPU EP's nodes between "
            "islands, and session setup are all outside it. Phase shares below are shares of "
            "this, and may not be restated as shares of the benchmark's wall time."),
        "host_phases_ms": host,
        "unattributed_in_compute_ms": round(in_compute_ms - phased_ms, 3),
        "unattributed_note": (
            "time inside vulkan.subgraph that no phase span covers: reading ORT input pointers, "
            "buffer allocation and descriptor-pool work before recording, the readback memcpy "
            "and the writes into ORT's output tensors after the fence. It is reported rather "
            "than folded into a neighbouring phase, because a phase split whose parts do not sum "
            "to the whole should say so."),
        "gpu": gpu,
        "device_fingerprint": device_fingerprint(gpus),
        "anchor_quality": anchor_quality(gpus),
        "gpu_note": (
            "summed from the per-span gpu_ns float, not from the integer-microsecond `dur`: "
            "several of these kernels run in 2-3 us, where truncation is a 15-30% error over "
            "thousands of spans."),
        "shares_of_time_in_compute": shares,
        "transfers": transfer_totals(transfers),
        "partition_stats": partition_stats(events),
        "record_scaling": scaling,
        "falsifiers": {
            "phase_containment": phase_containment(subs, attributed),
            "gpu_span_accounting": gpu_span_accounting(subs, gpus, counters),
            "gpu_containment": gpu_containment(subs, attributed, gpus),
            "timestamp_conversion_integrality": timestamp_conversion_integrality(gpus),
            "valid_bits_applied": valid_bits_applied(gpus),
            "trace_matches_counters": trace_matches_counters(subs, counters),
        },
    }


def red_flags(report: dict) -> "list[str]":
    """Every falsifier that went red, as text. Empty means every check that *could* fail did not."""
    out = []
    for name, f in (report.get("falsifiers") or {}).items():
        if f.get("red"):
            out.append(f"{name}: {f.get('detail')}")
    ti = (report.get("falsifiers") or {}).get("timestamp_conversion_integrality") or {}
    if ti and not ti.get("decisive"):
        out.append("timestamp_conversion_integrality: NOT DECISIVE on this device — "
                   + str(ti.get("detail")))
    return out


def describe(report: dict) -> "list[str]":
    lines = ["  phase split (share of time inside Compute; "
             f"{report['time_in_compute_ms']:.1f} ms over {report['subgraph_spans']} "
             f"subgraph invocations):"]
    shares = report.get("shares_of_time_in_compute") or {}
    host = report.get("host_phases_ms") or {}
    for phase in HOST_PHASES:
        if phase in host and host[phase].get("n"):
            s = host[phase]
            lines.append(f"    vulkan.{phase:<11} {s['total_ms']:>10.2f} ms  "
                         f"{shares.get(phase, 0) * 100:5.1f}%   n={s['n']:<6} "
                         f"median {s['median_ms']:.3f} ms")
    rs = report.get("record_scaling") or {}
    if rs.get("usable") and rs.get("upload_inside_record_ms") is not None:
        lines.append(f"      \u2514 of vulkan.record: {rs['upload_inside_record_ms']:.2f} ms is the "
                     f"host upload memcpy ({(rs.get('upload_share_of_record') or 0) * 100:.1f}%), "
                     f"{rs['command_construction_ms']:.2f} ms is command construction")
    un = report.get("unattributed_in_compute_ms")
    if un is not None:
        lines.append(f"    unattributed        {un:>10.2f} ms  "
                     f"{(un / report['time_in_compute_ms'] * 100 if report['time_in_compute_ms'] else 0):5.1f}%"
                     f"   (input pointers, allocation, readback, output writes)")
    g = (report.get("gpu") or {}).get("all") or {}
    if g.get("n"):
        lines.append(f"    GPU kernels (sum)   {g['total_ms']:>10.2f} ms  "
                     f"{shares.get('gpu_kernels', 0) * 100:5.1f}%   n={g['n']}")
        for k, v in sorted((report["gpu"]["per_kernel"]).items(),
                           key=lambda kv: -kv[1]["total_ms"]):
            lines.append(f"      {k:<34} {v['total_ms']:>9.2f} ms  n={v['n']}")
    for d, t in (report.get("transfers") or {}).items():
        where = "inside vulkan.record" if t["inside_record_span"] else "after the fence, in no phase span"
        lines.append(f"    host {d:<8} memcpy {t['host_copy_ms']:>9.2f} ms  "
                     f"{t['mib']:.1f} MiB at {t['mean_gib_s']} GiB/s  n={t['n']}  ({where})")
    return lines
