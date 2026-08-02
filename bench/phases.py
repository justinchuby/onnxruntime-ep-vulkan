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

WHAT KILLED A HYPOTHESIS, AND WHAT KILLED ITS REPLACEMENT
=========================================================
The reason this file exists in this shape is that the phase split falsified a plausible inference
of mine. From the first Phi-3.5 measurement I observed Intel paying roughly 2× per island what
NVIDIA paid *while having no bus to cross*, and reasoned towards a fixed per-submission cost —
submit-and-wait per island. I declined to design around it and asked for the instrument. The
instrument said ``vulkan.submit`` is **0.3%** of the run, and the GPU is idle for most of the wall
clock. The inference was drawn from real data and was about the wrong stage.

**Then the replacement hypothesis died the same way, and it was mine to catch.** The phase table
that killed "submission is the cost" said ``vulkan.record`` was ~68% and it was read — including
by me, in this docstring — as "command-buffer recording is the bottleneck". It is not.
``Phase::Record`` opens before ``vkBeginCommandBuffer`` and closes after ``vkEndCommandBuffer``,
and **the host staging memcpy runs inside that window**, reporting through ``record_transfer``
into a ``ph:"C"`` counter that emits *no span* (deliberately, to avoid double-counting). An
aggregation over ``ph:"X"`` spans is therefore structurally incapable of seeing it.

Measured on the stored NVIDIA trace by this module: upload is **98.6%** of the ``record`` phase.
Command-buffer construction is **753 ms of 65090 ms wall — 1.2%**, not 68%. The real cost is that
the EP re-uploads the entire weight set every inference (~1997.6 MiB/inference, exactly linear in
run count).

The lesson is structural, not arithmetic: **a phase whose children are invisible to the
aggregation must never be reported as a leaf.** Summing sibling spans and reporting the largest by
name attributes a child's cost to its parent's name, and the name then travels. ``phase_totals``
now refuses to present ``record`` as a leaf, and ``phase_leaf_accounting`` goes red if it is.

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
  nesting assumption used to attribute spans to islands is wrong. It checks **two tiers
  separately** (siblings against their subgraph, children against their own ``record`` parent)
  because summing a parent together with its children against the grandparent is not a
  containment violation, it is an arithmetic mistake in the checker — and that mistake reported
  RED for a whole day. See :data:`CONTAINMENT_BASIS` and the ``ERROR`` state below.
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
HOST_PHASES = ("compile", "prepack", "record", "upload", "desc_alloc", "pipeline_lookup",
               "cmd_upload", "submit", "fence_wait", "readback")

#: Phases the EP emits *nested inside* another phase's span.
#:
#: ``trace.rs`` marks each of these with a caveat beginning ``host/sub-record:``. They are real
#: ``ph:"X"`` spans that open and close **inside** ``vulkan.record``, so summing ``HOST_PHASES``
#: as if they were siblings double-counts every microsecond of them. That is not a small error:
#: ``cmd_upload`` alone is ~98% of ``record``, so a naive sum inflates the host total by nearly
#: 2× and every share computed from it.
#:
#: This tuple is the *expectation*. It is never trusted on its own — :func:`phase_nesting`
#: re-derives parenthood from timestamp containment, which is evidence, and goes red when the two
#: disagree. A sub-phase added to ``trace.rs`` tomorrow is therefore detected rather than silently
#: double-counted.
SUB_RECORD_PHASES = ("desc_alloc", "pipeline_lookup", "cmd_upload")

#: Prefix ``trace.rs`` puts on a nested phase's caveat. The artifact declares its own structure.
SUB_PHASE_CAVEAT_PREFIX = "host/sub-record:"

#: Phases that **contain** other accounted work and are therefore NOT leaves.
#:
#: ``Phase::Record`` opens before ``vkBeginCommandBuffer`` and closes after
#: ``vkEndCommandBuffer``, and the host staging memcpy runs inside that window. Upload reports
#: through ``record_transfer`` into a ``ph:"C"`` counter and deliberately emits no span, so an
#: aggregation over ``ph:"X"`` cannot see it and silently folds it into ``record``.
#:
#: This is not a rounding problem. Measured on the stored NVIDIA trace, upload is **98.6%** of
#: ``record`` — so a table that lists ``record`` beside ``upload`` as siblings reports the single
#: largest cost in the run under a name that means something else, and points optimisation work at
#: the ``vkCmd*`` loop instead of at weight residency. Both readings of "record is 68%" made on
#: this project drew that conclusion.
#:
#: Maps parent phase -> the accounted children that live inside its span.
PHASE_CHILDREN = {
    "record": ("upload",) + SUB_RECORD_PHASES,
}


def is_leaf_phase(phase: str) -> bool:
    """True when a phase's duration is entirely its own work.

    A non-leaf phase's total is an upper bound on the activity its name describes, never a
    measurement of it. Use ``record_scaling()['command_construction_ms']`` for the leaf residual.
    """
    return phase not in PHASE_CHILDREN


#: `X` (complete) events on the host lane that bound one `Compute` call.
SUBGRAPH = "vulkan.subgraph"

#: Prefix of a device-lane span produced from `VkQueryPool` results.
GPU_PREFIX = "vulkan.gpu."

#: Relative slack allowed before :func:`phase_containment` calls a subgraph over-subscribed.
#: Phases are wall-clock siblings inside one span; they cannot legitimately exceed it. The slack
#: absorbs the microsecond truncation of `dur` across ~7 spans, nothing more.
CONTAINMENT_SLACK = 0.02

#: What set of spans :func:`phase_containment` is entitled to sum, written into the artifact.
#:
#: The containment claim is only meaningful about the spans that are actually *added together*
#: somewhere — which is ``sibling_phases()``, the same set ``host_phase_totals`` and every share
#: consume. Handing it the unfiltered list adds ``record`` to the ``cmd_upload`` that lives inside
#: ``record``, so a run in which ``cmd_upload`` is 97% of ``record`` reports the parent's time
#: nearly twice and "exceeds its own duration" with no defect present anywhere in the EP.
CONTAINMENT_BASIS = ("sibling phase spans only — nested sub-record spans are checked separately "
                     "against their own parent, never against the subgraph")

#: How close `gpu_ns / timestamp_period_ns` must be to a whole number. The tick count is an
#: integer by construction; this tolerance absorbs float32→float64 widening of the period only.
INTEGRALITY_TOL = 1e-3

#: How far a single inference's normalised host cost may sit from the run median before
#: :func:`contention_signature` calls it an excursion. Two-fold is well outside anything a
#: steady machine produces for identical repeated work, and well inside the 9.5x the coordinator
#: measured under six concurrent agents — so it catches the real case without firing on noise.
EXCURSION_FACTOR = 2.0

#: How stable an island slot's own GPU busy time must be, across repetitions of identical work,
#: for that slot's GPU time to serve as a control for its host time. Above this the two moved
#: together and the host swing is explained by the work, not by the machine.
GPU_STABLE_MAX = 1.25


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
            "nested_in": (e.get("args") or {}).get("nested_in"),
        }
        for e in events
        if e.get("ph") == "X" and e.get("name") in names
    ]
    out.sort(key=lambda s: s["ts"])
    return out


def phase_nesting(phases: "list[dict]") -> dict:
    """Derive which phases nest inside which **from timestamp containment**, and check the names.

    Two independent sources for the same fact, which is what makes this a falsifier that can fire:

    * **Evidence** — a span whose ``[ts, end)`` lies inside another phase span on the same thread
      is nested. Timestamps come from the EP's clock and do not know what the phase is called.
    * **Declaration** — ``trace.rs`` prefixes a nested phase's caveat with ``host/sub-record:``.
      That is a *name*, and R11 says a name is not a definition.

    Goes red when they disagree in either direction: a phase declared nested that is not contained
    (the caveat is wrong, or the span escaped its parent) or a phase contained that is not declared
    (a new sub-phase landed and every sibling sum since then has been double-counting).

    The second direction is the one that matters operationally. ``desc_alloc``,
    ``pipeline_lookup`` and ``cmd_upload`` were added to ``trace.rs`` after this module was
    written; without this check they would have been summed as siblings of ``record``, inflating
    the host total by ~2× and every share derived from it, with nothing raising.
    """
    by_name: "dict[str, list[dict]]" = {}
    for p in phases:
        by_name.setdefault(p["phase"], []).append(p)

    parents = [p for p in phases if p["phase"] == "record"]
    parents.sort(key=lambda p: p["ts"])

    def contained_in_record(sp: dict) -> bool:
        for par in parents:
            if par["ts"] <= sp["ts"] and sp["end"] <= par["end"] and par is not sp:
                return True
        return False

    observed, declared, mismatches = {}, {}, []
    for name, spans in sorted(by_name.items()):
        if name == "record":
            continue
        inside = sum(1 for s in spans if contained_in_record(s))
        observed[name] = {"n": len(spans), "inside_record": inside}
        says_sub = any(str(s.get("caveat") or "").startswith(SUB_PHASE_CAVEAT_PREFIX)
                       for s in spans)
        declared[name] = says_sub
        all_inside = inside == len(spans) and spans
        if says_sub and not all_inside:
            mismatches.append(
                f"{name}: caveat declares '{SUB_PHASE_CAVEAT_PREFIX}' but only {inside}/"
                f"{len(spans)} spans are contained by a vulkan.record span")
        elif all_inside and not says_sub:
            mismatches.append(
                f"{name}: every one of its {len(spans)} spans is contained by vulkan.record but "
                f"its caveat does not declare it nested — if it is summed as a sibling of "
                f"'record' the host total double-counts it")

    unexpected = sorted(n for n, d in declared.items()
                        if d and n not in PHASE_CHILDREN.get("record", ()))
    out = {
        "check": "phase_nesting",
        "asserts": "the phases summed as siblings do not overlap, so the host total counts each "
                   "microsecond once",
        "observed": observed,
        "declared_nested": sorted(n for n, d in declared.items() if d),
        "expected_nested": sorted(SUB_RECORD_PHASES),
        "unexpected_nested": unexpected,
    }
    if mismatches:
        out.update(ok=False, verdict="MISMATCH", detail="; ".join(mismatches))
    elif not parents:
        out.update(ok=True, verdict="VACUOUS",
                   detail="no vulkan.record span in this trace; nothing can nest inside it.")
    else:
        out.update(ok=True, verdict="CONSISTENT",
                   detail=(f"containment and caveats agree; nested = "
                           f"{', '.join(out['declared_nested']) or 'none'}"))
    if unexpected:
        out["detail"] += (f". NOTE: {', '.join(unexpected)} declare themselves nested but are not "
                          f"in PHASE_CHILDREN — they are being treated as children from the "
                          f"trace's own declaration, not from this module's table.")
    return out


def nested_phase_names(phases: "list[dict]") -> "set[str]":
    """Every phase name that must NOT be summed as a top-level sibling.

    Three sources, unioned, because each catches a failure the others cannot:

    * :data:`SUB_RECORD_PHASES` — this module's table. Catches a child whose trace-side
      declaration was dropped.
    * the ``host/sub-record:`` caveat prefix — prose, but prose the EP author writes deliberately.
    * the ``nested_in`` span arg — machine-readable parentage, emitted from ``Phase::nested_in()``
      whose ``match`` is exhaustive with no ``_`` arm, so a new phase cannot default to sibling.

    A union and not a vote: any one of them saying "child" is sufficient. Being wrongly excluded
    from a sum costs a line in ``nested_phases_ms``; being wrongly *included* double-counts the
    largest cost in the run, which is the error this project actually made.
    """
    nested = set(SUB_RECORD_PHASES)
    for p in phases:
        if str(p.get("caveat") or "").startswith(SUB_PHASE_CAVEAT_PREFIX):
            nested.add(p["phase"])
        parent = p.get("nested_in")
        if parent and parent != "none":
            nested.add(p["phase"])
    return nested


def sibling_phases(phases: "list[dict]") -> "list[dict]":
    """The phase spans that may legitimately be summed together — nested children removed.

    Children come from :func:`nested_phase_names`. This is the only set any total, share or
    containment sum may be computed over.
    """
    nested = nested_phase_names(phases)
    return [p for p in phases if p["phase"] not in nested]



def decomposition_identity(host: dict, gpu: dict, in_compute_ms: float,
                           independent_whole_ms: "float | None" = None,
                           whole_source: str = "") -> dict:
    """R11: does the decomposition close, and **against a whole from a different source**?

    The rule this implements was paid for. A phase table closed at 99.0% —
    ``68.3 + 16.3 + 14.1 + 0.3`` — and was wrong, because the missing 2 GB memcpy was *inside* one
    of the rows. Both sides of that identity were sums over the same tracer's spans, so the parts
    and the whole moved together and the check could not fire no matter how badly the rows were
    named. **An identity whose two sides come from the same source is a falsifier that cannot
    fire.**

    So this returns two different things and never conflates them:

    * ``internal_closure`` — parts against ``sum(vulkan.subgraph)``. Both from the EP's tracer.
      Useful for spotting unattributed time; **not evidence that the rows mean what they say**,
      and labelled as such. It is `WEAK` by construction.
    * ``external_closure`` — ``sum(vulkan.subgraph)`` against a whole measured by a *different*
      clock: the harness's own ``time.perf_counter`` around ``session.run``, which knows nothing
      about phases. This one **can** fire. If the trace claims more time than the wall clock
      contains, spans are being double-counted — which is exactly what summing nested sub-record
      phases as siblings does.

    Without an independent whole the verdict is ``UNCHECKABLE``, never ``ok``.
    """
    parts_ms = sum(v["total_ms"] for v in host.values() if v.get("n"))
    out = {
        "check": "decomposition_identity",
        "asserts": "a published decomposition closes against a whole measured by a different "
                   "instrument than the parts",
        "parts_ms": round(parts_ms, 3),
        "in_compute_ms": round(in_compute_ms, 3),
    }

    # --- internal: same source on both sides. Structurally weak, and says so. ---
    if in_compute_ms > 0:
        resid = in_compute_ms - parts_ms
        out["internal_closure"] = {
            "residual_ms": round(resid, 3),
            "residual_share": round(resid / in_compute_ms, 5),
            "over_subscribed": parts_ms > in_compute_ms * (1 + CONTAINMENT_SLACK),
            "strength": "WEAK",
            "why_weak": (
                "both sides are sums over the same tracer's spans. A cost hidden inside a row "
                "cancels from both and this check stays green — which is precisely how "
                "'68.3+16.3+14.1+0.3 = 99.0%' closed while missing a 2 GB memcpy. Never quote "
                "this as evidence that the rows are correctly named."),
        }
        if out["internal_closure"]["over_subscribed"]:
            out["internal_closure"]["detail"] = (
                f"parts ({parts_ms:.1f} ms) exceed the whole ({in_compute_ms:.1f} ms) by more "
                f"than {CONTAINMENT_SLACK:.0%}. Phases are being counted more than once — the "
                f"usual cause is summing nested sub-record spans as siblings.")

    # --- external: a whole from a clock that does not know what a phase is. ---
    if independent_whole_ms is None or independent_whole_ms <= 0:
        out.update(ok=False, verdict="UNCHECKABLE",
                   detail=("no independently measured whole was supplied, so the decomposition "
                           "cannot be checked against anything that could contradict it. Per R11 "
                           "it is not publishable in this state."))
        return out

    ratio = in_compute_ms / independent_whole_ms
    out["external_closure"] = {
        "independent_whole_ms": round(independent_whole_ms, 3),
        "whole_source": whole_source or "unspecified",
        "trace_share_of_wall": round(ratio, 5),
        "strength": "CAN FIRE",
    }
    if ratio > 1 + CONTAINMENT_SLACK:
        out.update(
            ok=False, verdict="EXCEEDS_WALL",
            detail=(f"the trace accounts for {in_compute_ms:.1f} ms inside Compute but the "
                    f"harness's own clock measured only {independent_whole_ms:.1f} ms of wall "
                    f"time for the same work ({ratio:.2f}x). Time is being counted more than "
                    f"once. Two independent clocks disagreeing is not a rounding question."))
    else:
        out.update(
            ok=True, verdict="CLOSES",
            detail=(f"trace-side time inside Compute is {ratio:.1%} of the wall time measured by "
                    f"{whole_source or 'the harness'} — a different clock that knows nothing "
                    f"about phases. The remainder is ORT graph execution and CPU-EP nodes "
                    f"between islands, which are real and outside the EP."))
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


def host_phase_totals(attributed: "list[dict]", child_ms: "dict[str, float] | None" = None,
                      child_names: "dict[str, tuple] | None" = None) -> dict:
    """Per-phase host totals, with non-leaf phases marked as such.

    Every entry carries ``is_leaf``. For a non-leaf phase the total is an **upper bound on the
    activity its name names**, not a measurement of it, and ``leaf_ms`` carries the residual once
    the accounted children are subtracted.

    This is not decoration. The one number this project got most wrong — "command-buffer recording
    is 68% of the run" — came from reading ``record``'s total as though it measured ``vkCmd*``
    calls, when 98.6% of it was the weight upload happening inside the same span. The reading was
    made twice, by two people, from a table that gave them no way to tell.

    ``child_ms`` maps a parent phase to the milliseconds of accounted child work inside it. It is
    supplied by the caller from :func:`record_scaling`, which attributes each transfer to the
    record span whose interval **contains** it. Deriving it here from transfer direction alone
    would be a second, weaker attribution of the same quantity, and two attributions that can
    disagree are worse than one that can be checked.
    """
    by: "dict[str, list[float]]" = {}
    for p in attributed:
        by.setdefault(p["phase"], []).append(p["dur"] / 1000.0)
    out = {k: _summarise(v) for k, v in by.items()}
    child_ms = child_ms or {}

    for phase, rec in out.items():
        rec["is_leaf"] = is_leaf_phase(phase)
        if rec["is_leaf"]:
            continue
        kids = (child_names or {}).get(phase) or PHASE_CHILDREN[phase]
        rec["contains"] = list(kids)
        rec["caveat"] = (
            f"NOT A LEAF: this span also contains {', '.join(kids)}. Its total is an upper bound "
            f"on {phase} itself, not a measurement of it. Quoting it as '{phase}' attributes a "
            f"child's cost to the parent's name."
        )
        known = child_ms.get(phase)
        if known is None:
            rec["leaf_ms"] = None
            rec["leaf_ms_note"] = (
                "no transfer counters in this trace, so the children cannot be subtracted and the "
                "leaf cost is UNKNOWN — not equal to the total."
            )
        else:
            rec["child_ms"] = round(known, 3)
            rec["leaf_ms"] = round(max(rec["total_ms"] - known, 0.0), 3)
            rec["child_share"] = (round(known / rec["total_ms"], 4)
                                  if rec["total_ms"] else None)
    return out


def phase_leaf_accounting(totals: dict) -> dict:
    """Falsifier: no phase total may be quotable under its own name unless it is a leaf.

    Goes **red** when a non-leaf phase is present and its children were not subtracted — i.e.
    exactly the state in which "record is 68%" is derivable from the table. Goes red *loudly* when
    a non-leaf phase is the largest phase in the run, because that is when the misreading is not
    merely possible but the natural one.
    """
    non_leaf = {k: v for k, v in totals.items() if not v.get("is_leaf", True)}
    out = {
        "check": "phase_leaf_accounting",
        "asserts": "a phase total is only quoted under its own name when that span contains no "
                   "other accounted work",
        "non_leaf_phases": sorted(non_leaf),
    }
    if not non_leaf:
        out.update(ok=True, verdict="VACUOUS",
                   detail="no non-leaf phase in this trace; nothing to mis-attribute.")
        return out

    unresolved = [k for k, v in non_leaf.items() if v.get("leaf_ms") is None]
    largest = max(totals, key=lambda k: totals[k].get("total_ms") or 0.0)
    out["largest_phase"] = largest
    out["largest_phase_is_non_leaf"] = largest in non_leaf

    if unresolved:
        out.update(
            ok=False, verdict="UNRESOLVED", unresolved=unresolved,
            detail=(f"{', '.join(unresolved)} contains other accounted work that this trace does "
                    f"not let us subtract. Its total must not be quoted under its own name."))
        if largest in unresolved:
            out["detail"] += (f" It is also the LARGEST phase here, so the natural reading of this "
                              f"table — '{largest} is the bottleneck' — is unsupported.")
        return out

    out.update(ok=True, verdict="RESOLVED",
               detail="; ".join(
                   f"{k}: {v['child_share']:.1%} of it is {'+'.join(v['contains'])}, "
                   f"leaf residual {v['leaf_ms']:.1f} ms"
                   for k, v in sorted(non_leaf.items())))
    return out


def upload_accounting(counter_ms: "float | None", span_ms: "float | None",
                      tol: float = 0.25) -> dict:
    """Falsifier: two independent accountings of the same upload must agree.

    The transfer counters (bytes, inverted through a rate into a duration) and the ``cmd_upload``
    span (a wall-clock interval) measure the same host memcpy by different means. If both exist
    and disagree, at least one is wrong and neither may be quoted -- the precedent is
    ``alloc_device_upload_bytes`` reporting 0 on a run where ``cmd_upload`` was 15.2 s: two upload
    accountings, one blind, and nothing went red.

    Goes red on disagreement beyond ``tol``. Reports VACUOUS -- never a pass -- when only one
    instrument is present, because one instrument cannot falsify itself.
    """
    out = {
        "check": "upload_accounting",
        "asserts": "the transfer-counter upload duration and the cmd_upload span agree",
        "counter_ms": None if counter_ms is None else round(counter_ms, 3),
        "span_ms": None if span_ms is None else round(span_ms, 3),
        "tolerance": tol,
    }
    if counter_ms is None or span_ms is None:
        present = "cmd_upload span" if span_ms is not None else (
            "transfer counters" if counter_ms is not None else "neither instrument")
        out.update(ok=True, verdict="VACUOUS", n_instruments=int(counter_ms is not None)
                   + int(span_ms is not None),
                   detail=f"only {present} available; a single instrument cannot falsify itself. "
                          f"This is not a pass.")
        return out
    denom = max(counter_ms, span_ms)
    rel = abs(counter_ms - span_ms) / denom if denom > 0 else 0.0
    out["relative_difference"] = round(rel, 4)
    out["n_instruments"] = 2
    if rel > tol:
        out.update(ok=False, red=True, verdict="DISAGREE",
                   detail=(f"transfer counters say {counter_ms:.1f} ms of upload inside record; "
                           f"the cmd_upload span says {span_ms:.1f} ms ({rel:.1%} apart, "
                           f"tolerance {tol:.0%}). One of the two is wrong; no upload figure from "
                           f"this trace is quotable until they are reconciled."))
        return out
    out.update(ok=True, verdict="AGREE",
               detail=(f"counters {counter_ms:.1f} ms vs span {span_ms:.1f} ms, {rel:.1%} apart. "
                       f"The span is used; the counters corroborate it."))
    return out


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

def phase_containment(subgraphs: "list[dict]", siblings: "list[dict]",
                      nested: "list[dict] | None" = None) -> dict:
    """Two tiers of containment, each checked against its own parent. R13 three-state.

    **Tier 1 — siblings against their subgraph.** The phases that get summed (``record``,
    ``submit``, ``fence_wait``, …) are wall-clock intervals opened inside one ``Compute`` call.
    They may not sum past the ``vulkan.subgraph`` span that brackets that call, and none of them
    may fall outside every subgraph.

    **Tier 2 — children against their own ``record``.** ``desc_alloc``, ``pipeline_lookup`` and
    ``cmd_upload`` are real ``ph:"X"`` spans *inside* ``vulkan.record``. Their parent is
    ``record``, not the subgraph. Checking them against the subgraph while ``record`` is also in
    the sum asks the subgraph to contain the same microseconds twice.

    # The three terminal states (R13)

    ``PASS`` — both tiers close.
    ``FAIL`` — a real containment violation in the EP or in the attribution.
    ``ERROR`` — **this checker was handed the wrong set of spans**, so it is not entitled to a
    verdict at all. Specifically: a span declaring itself nested appearing in the sibling list.
    An instrument error is not a detection, and reporting it as one costs the guard its
    authority.

    # Why the ERROR arm exists

    It exists because that is exactly what happened. ``analyse()`` computed ``siblings`` correctly
    for every total and every share, and then passed the *unfiltered* ``attributed`` list to this
    function. On a one-island Phi-3.5 run, ``record`` is 8.32 s and the ``cmd_upload`` inside it
    is 7.97 s, so the "sum" came to 16.66 s against a 13.67 s subgraph and reported

        RED: 1 subgraphs whose phases exceed their own duration

    for three consecutive days, on both devices, with nothing wrong in the EP. Every phase share
    in the project was withheld on it. Sibling-only, the same trace sums to 8.60 s inside
    13.67 s — 63%, comfortable. The failing side was the checker.

    This is R11's inverse: not a decomposition that closes falsely, but one made *not* to close by
    adding a term to it. The remedy is the same — say which set the identity is over, in the
    artifact, next to the number.
    """
    misfiled = sorted({p["phase"] for p in siblings
                       if str(p.get("caveat") or "").startswith(SUB_PHASE_CAVEAT_PREFIX)
                       or (p.get("nested_in") not in (None, "none"))})
    if misfiled:
        return {
            "red": False,
            "state": "ERROR",
            "basis": CONTAINMENT_BASIS,
            "instrument_error": (
                f"{', '.join(misfiled)} declare themselves nested inside another phase but were "
                f"handed to phase_containment as siblings. Summing a parent with its own children "
                f"against the grandparent is an arithmetic mistake in this checker, not a "
                f"containment violation in the EP. No verdict is issued."),
            "orphan_phase_spans": None,
            "over_subscribed_subgraphs": None,
            "examples": [],
            "detail": ("ERROR (instrument): phase_containment was given nested spans in the "
                       f"sibling set ({', '.join(misfiled)}). Per R13 an instrument error is not "
                       "a detection — this is neither a pass nor a failure."),
        }

    orphans = [p for p in siblings
               if p["subgraph_index"] is None and p["phase"] not in ("compile", "prepack")]
    per: "dict[int, float]" = {}
    for p in siblings:
        if p["subgraph_index"] is not None:
            per[p["subgraph_index"]] = per.get(p["subgraph_index"], 0) + p["dur"]
    over = [
        {"subgraph_index": i, "phases_us": per[i], "subgraph_us": subgraphs[i]["dur"],
         "ratio": round(per[i] / subgraphs[i]["dur"], 4)}
        for i in per
        if subgraphs[i]["dur"] > 0 and per[i] > subgraphs[i]["dur"] * (1 + CONTAINMENT_SLACK)
    ]

    # Tier 2. Parents are located by timestamp containment, not by name order: `record` spans are
    # disjoint, so the enclosing one is unique when it exists.
    kids = list(nested or [])
    parents = sorted([p for p in siblings if p["phase"] == "record"], key=lambda p: p["ts"])
    stray_children, per_parent = [], {}
    for k in kids:
        owner = next((i for i, r in enumerate(parents)
                      if r["ts"] <= k["ts"] and k["end"] <= r["end"]), None)
        if owner is None:
            stray_children.append(k)
        else:
            per_parent[owner] = per_parent.get(owner, 0) + k["dur"]
    over_parents = [
        {"record_index": i, "children_us": per_parent[i], "record_us": parents[i]["dur"],
         "ratio": round(per_parent[i] / parents[i]["dur"], 4)}
        for i in per_parent
        if parents[i]["dur"] > 0 and per_parent[i] > parents[i]["dur"] * (1 + CONTAINMENT_SLACK)
    ]

    red = bool(orphans or over or stray_children or over_parents)
    if red:
        detail = (
            f"FAIL: tier 1 — {len(orphans)} sibling phase spans outside any subgraph, "
            f"{len(over)} subgraphs whose sibling phases exceed their own duration; "
            f"tier 2 — {len(stray_children)} sub-record spans outside every vulkan.record span, "
            f"{len(over_parents)} record spans whose children exceed them. The attribution of "
            f"phases to islands is not sound and no phase share below may be read.")
    else:
        detail = (
            f"PASS: {len(siblings)} sibling spans all lie inside a subgraph and no subgraph's "
            f"siblings sum past it; {len(kids)} sub-record spans all lie inside a vulkan.record "
            f"span and no record's children sum past it. Checked over "
            f"{len(subgraphs)} subgraph spans. Basis: {CONTAINMENT_BASIS}.")

    return {
        "red": red,
        "state": "FAIL" if red else "PASS",
        "basis": CONTAINMENT_BASIS,
        "orphan_phase_spans": len(orphans),
        "over_subscribed_subgraphs": len(over),
        "stray_sub_record_spans": len(stray_children),
        "over_subscribed_record_spans": len(over_parents),
        "sibling_spans_checked": len(siblings),
        "nested_spans_checked": len(kids),
        "examples": (over + over_parents)[:3],
        "worst_subgraph_ratio": (max((o["ratio"] for o in (
            {"ratio": round(per[i] / subgraphs[i]["dur"], 4)}
            for i in per if subgraphs[i]["dur"] > 0)), default=None)),
        "detail": detail,
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


def contention_signature(attributed: "list[dict]", subgraphs: "list[dict]",
                         gpu_busy_us: "dict[int, float] | None" = None,
                         integrated_gpu: bool = False) -> dict:
    """Was the machine busy while this trace was captured? Answered from the trace alone.

    Why this can be answered after the fact
    ---------------------------------------
    ``bench/contention.py`` samples the machine *while* a benchmark runs, but every trace this
    project captured before it existed has no such record. That would normally make the question
    unanswerable — except that a Vulkan trace already contains two clocks with completely
    different exposure to host CPU load:

    * **host phase spans** (``record``, ``submit``, ``fence_wait``) are wall-clock intervals on
      a thread that must be scheduled to make progress. Take the core away and they stretch.
    * **GPU spans** are differences of the device's own timestamp counter. The GPU does not care
      how many copies of ``rustc`` are running. Take the core away and they do not move.

    So the trace carries its own control. That is the whole method.

    The statistic
    -------------
    Submissions repeat in a fixed cycle: the same islands, in the same order, once per
    inference. **Island slot ``s`` on inference ``c`` does exactly the same work as island slot
    ``s`` on inference ``c+1``.** So for each slot, take the spread (max/min) of its host record
    time across repetitions, and the spread of its *own* GPU busy time across the same
    repetitions. A slot whose host time swings while its GPU time does not has stalled on the
    host.

    ``stalled_slot_fraction`` is the share of controllable slots in that state. Three outcomes,
    and only one of them is contention:

    ============================  ==============================  ==========================
    host spread on a slot         gpu spread on the same slot     that slot is
    ============================  ==============================  ==========================
    >= ``EXCURSION_FACTOR``       < ``GPU_STABLE_MAX``            a host-side stall
    >= ``EXCURSION_FACTOR``       >= ``GPU_STABLE_MAX``           doing different work
    < ``EXCURSION_FACTOR``        anything                        steady
    ============================  ==============================  ==========================

    A secondary statistic, ``per_inference_host_factor``, collapses each inference to a single
    number to catch a run that was *uniformly* slow rather than sporadically stalled. It is
    reported, but it is not the primary: it is a median across slots, and a median across slots
    hides a stall that hits a minority of them. See the comment at the computation.

    What this does and does not establish
    -------------------------------------
    A red verdict says the host stalled in a way the device did not. **It does not name the
    cause.** Another process on the CPU is the obvious candidate — the coordinator measured the
    same device, build and test 9.5× apart on load alone — but a page-fault storm, a driver
    allocation, or a thermal event would look the same from inside the trace. What it does
    establish is the thing that matters for deciding whether to trust a stored number: the run
    was **not in a steady state**, so its mean is a mean over conditions that were not held
    constant, and it cannot be compared with a number taken under different ones.

    On an **integrated** GPU the control is weaker in one direction and must not be
    over-claimed: the iGPU shares its power and thermal budget with the CPU cores, so heavy CPU
    load can slow the device too. That cannot manufacture a false ``HOST_SIDE_EXCURSIONS`` — it
    pushes the other way — but it can manufacture a false ``WORKLOAD_VARIATION``, which is why
    that verdict carries a caveat and is not marked quotable.

    Warmup is excluded before the statistic is computed, because a genuine warmup ramp produces
    host-side excursions too, in the first cycles, for an entirely legitimate reason.
    """
    rec = [p for p in attributed if p["phase"] == "record" and p["subgraph_index"] is not None]
    rec.sort(key=lambda p: p["ts"])
    out: dict = {"n_record_spans": len(rec)}
    if len(rec) < 12:
        out.update(verdict="UNTESTABLE",
                   reason="fewer than 12 attributed record spans; no cycle structure to use")
        return out

    nodes_seq = [p.get("nodes") for p in rec]
    period = _cycle_period(nodes_seq) or (1 if len({*nodes_seq}) == 1 else None)
    # `period == 1` is the single-island graph, which is the configuration this project is
    # *trying* to reach — and the original `period < 2` guard turned the in-band control off
    # exactly there, permanently, as a side effect of success. Nothing in the statistic needs two
    # slots: slot 0 still yields one host series and one GPU series over the same repeated work,
    # which is the whole method. What is lost is the ability to see a stall hit some islands and
    # not others, so `stalled_slot_fraction` degenerates to 0.0 or 1.0 and is annotated as such.
    cycles = len(rec) // period if period else 0
    if not period or cycles < 4:
        out.update(verdict="UNTESTABLE",
                   reason=(f"no usable repeat structure: "
                           + ("the subgraph order has no repeating period"
                              if not period else
                              f"only {cycles} complete cycles of {period} island(s) "
                              f"(need >= 4, and the first is dropped as warmup)")))
        return out
    out["single_slot"] = period == 1
    cycles = len(rec) // period
    out["islands_per_inference"] = period
    out["cycles"] = cycles

    # Drop the first cycle outright: warmup excursions are host-side and legitimate, and
    # including them would let a correctly-behaving run answer "contended".
    skip = 1
    out["cycles_skipped_as_warmup"] = skip

    def _series(value_of) -> "dict[int, list[float]]":
        by_slot: "dict[int, list[float]]" = {}
        for c in range(skip, cycles):
            for s in range(period):
                p = rec[c * period + s]
                v = value_of(p)
                if v is not None:
                    by_slot.setdefault(s, []).append(v)
        return by_slot

    host_by_slot = _series(lambda p: p["dur"] / 1000.0)
    gpu_ok = bool(gpu_busy_us)
    gpu_by_slot = _series(
        (lambda p: gpu_busy_us.get(p["subgraph_index"])) if gpu_ok else (lambda p: None)
    )

    def _range_ratio(vals: "list[float]") -> "float | None":
        if len(vals) < 3 or min(vals) <= 0:
            return None
        return max(vals) / min(vals)

    # ---- primary statistic: per-slot, GPU-controlled --------------------------------
    #
    # An earlier version of this took the median across slots of each inference's normalised
    # host cost. That is the right shape for a run that is *uniformly* slow, and it is exactly
    # wrong for the real case: on the RTX 4060 trace, island slot 0 recorded 12.48, 70.19 and
    # 12.59 ms on three inferences while its GPU time was constant to 0.03%, and slot 5 went
    # 301 -> 1156 -> 374 ms. A median over 33 slots reported that run STABLE. Stalls hit some
    # islands and not others, so the statistic has to be per-slot; the median was averaging away
    # the thing it was built to find.
    slots = []
    for s in sorted(host_by_slot):
        hr = _range_ratio(host_by_slot[s])
        gr = _range_ratio(gpu_by_slot.get(s, []))
        slots.append({
            "slot": s,
            "nodes": rec[s].get("nodes"),
            "host_range_ratio": None if hr is None else round(hr, 3),
            "gpu_range_ratio": None if gr is None else round(gr, 4),
            "host_ms": [round(v, 2) for v in host_by_slot[s]],
        })
    testable = [r for r in slots
                if r["host_range_ratio"] is not None and r["gpu_range_ratio"] is not None]
    stalled = [r for r in testable
               if r["host_range_ratio"] >= EXCURSION_FACTOR
               and r["gpu_range_ratio"] < GPU_STABLE_MAX]
    moved_together = [r for r in testable
                      if r["host_range_ratio"] >= EXCURSION_FACTOR
                      and r["gpu_range_ratio"] >= GPU_STABLE_MAX]
    out["slots_testable"] = len(testable)
    out["stalled_slot_fraction"] = (
        round(len(stalled) / len(testable), 4) if testable else None)
    out["stalled_slots"] = sorted(
        stalled, key=lambda r: -(r["host_range_ratio"] or 0))[:8]
    out["slots_moving_with_gpu"] = len(moved_together)

    # ---- secondary statistic: whole-inference inflation ------------------------------
    def _factors(by_slot) -> "list[float]":
        medians = {k: statistics.median(v) for k, v in by_slot.items() if v}
        per_cycle: "list[float]" = []
        for c in range(skip, cycles):
            i = c - skip
            ratios = [by_slot[s][i] / medians[s]
                      for s in by_slot if medians.get(s) and i < len(by_slot[s])]
            if ratios:
                per_cycle.append(statistics.median(ratios))
        return per_cycle

    host = _factors(host_by_slot)
    gpu = _factors(gpu_by_slot)

    def _spread(vals: "list[float]") -> dict:
        if len(vals) < 3:
            return {"n": len(vals)}
        return {
            "n": len(vals),
            "median": round(statistics.median(vals), 4),
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
            "range_ratio": round(max(vals) / min(vals), 4) if min(vals) > 0 else None,
            "excursion_fraction": round(
                sum(1 for v in vals if v > EXCURSION_FACTOR) / len(vals), 4),
        }

    hs, gs = _spread(host), _spread(gpu)
    out["per_inference_host_factor"] = hs
    out["per_inference_gpu_factor"] = gs
    out["per_inference_host_factors"] = [round(v, 3) for v in host]
    out["per_inference_gpu_factors"] = [round(v, 3) for v in gpu]
    out["whole_run_inflation"] = bool(
        hs.get("n", 0) >= 3 and (hs.get("range_ratio") or 0) > EXCURSION_FACTOR)

    integrated = bool(integrated_gpu)
    gpu_testable = bool(testable)

    if not gpu_testable:
        verdict, reason = "HOST_EXCURSIONS_UNCONTROLLED", (
            "no GPU busy time could be paired with host record spans, so the device-clock "
            "control could not be run. Untested is not passed")
    elif stalled:
        worst = max(stalled, key=lambda r: r["host_range_ratio"])
        verdict, reason = "HOST_SIDE_EXCURSIONS", (
            f"{len(stalled)} of {len(testable)} island slots "
            f"({(len(stalled) / len(testable)) * 100:.0f}%) recorded a >= {EXCURSION_FACTOR}x "
            f"host spread across repetitions of identical work while their own GPU time stayed "
            f"within {GPU_STABLE_MAX}x. Worst: slot {worst['slot']} "
            f"({worst['nodes']} dispatches) host {worst['host_range_ratio']}x vs GPU "
            f"{worst['gpu_range_ratio']}x. The stall is on the host, not the device: this run "
            "was not in a steady state and its aggregates are means over conditions that were "
            "not held constant")
    elif out["whole_run_inflation"]:
        verdict, reason = "NOT_STEADY", (
            f"whole inferences of identical work varied {hs.get('range_ratio')}x "
            f"(min {hs.get('min')}, max {hs.get('max')} of the run median), but GPU time moved "
            f"alongside on every slot, so the control cannot isolate the cause to the host. "
            "Either way the run was not in a steady state: this is not a quotable measurement, "
            "and it is not an exoneration of the machine")
    elif moved_together:
        verdict, reason = "WORKLOAD_VARIATION", (
            f"host spread exceeded {EXCURSION_FACTOR}x on {len(moved_together)} slots but their "
            f"GPU time moved with it, so the work itself differed between inferences. That "
            "explains the host spread; it does not establish that the machine was quiet")
    elif cycles - skip < 4:
        verdict, reason = "UNDERPOWERED", (
            f"only {cycles - skip} post-warmup inferences over {len(testable)} controllable "
            f"island slots; a stall lasting less than one inference would not be visible. "
            "This is not a quiet-machine finding")
    else:
        verdict, reason = "STABLE", (
            f"no island slot recorded a {EXCURSION_FACTOR}x host spread over "
            f"{cycles - skip} repetitions of identical work; "
            f"{len(testable)} slots controlled against their own GPU time")

    out["verdict"] = verdict
    out["reason"] = reason
    if integrated:
        out["integrated_gpu_caveat"] = (
            "This device shares its power and thermal budget with the CPU cores. The control "
            "here — 'the device clock does not care how busy the host is' — is therefore weaker "
            "than on a discrete part: heavy CPU load can slow the GPU itself through DVFS, so "
            "GPU time moving *with* host time does not cleanly exonerate the machine. On an "
            "integrated device, WORKLOAD_VARIATION should be read as 'not established', not as "
            "'quiet'.")
    out["falsifier"] = {
        "name": "gpu_range_control",
        "red_if": (
            "the same island slot's GPU busy time varies as much as its host record time, which "
            "would mean the work differed between inferences and the host variation is explained"),
        "slots_stalled_host_only": len(stalled),
        "slots_moving_with_gpu": len(moved_together),
        "slots_testable": len(testable),
        "gpu_stable_max": GPU_STABLE_MAX,
    }
    out["quotable"] = verdict in ("STABLE",)
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
# Steady state — the first inference is now a different workload from the rest
# ---------------------------------------------------------------------------

#: Relative standard deviation a GPU-busy suffix must be under to be called steady.
#: The device clock does not see host load, so on a discrete part a genuinely steady tail comes
#: in around 0.03% — three orders of magnitude inside this. 2% is loose enough not to reject the
#: integrated part, where the GPU shares a power budget with the loaded CPU cores.
GPU_TAIL_RSD_MAX = 0.02

#: Fewest samples a steady tail may be declared from.
GPU_TAIL_MIN_N = 5

#: Absolute floor on the tail's length, and the floor on what fraction of the usable series it
#: must cover, before the tail's median may be *quoted*. See :func:`gpu_steady_tail` — a suffix
#: that clears the RSD bar without clearing these is ``MARGINAL_TAIL``, which is not a number.
GPU_TAIL_QUOTABLE_MIN_N = 8
GPU_TAIL_MIN_COVERAGE = 0.5


def gpu_steady_tail(busy_us: "list[float | None]") -> dict:
    """The longest stable suffix of the per-inference GPU-busy series, found from the device clock.

    # Why one cold inference is not enough warmup

    :func:`steady_state_split` drops the first *cycle*, which removes the one-time weight upload.
    It does not remove a **multi-inference ramp**, and there is one: on the RTX 4060 the GPU-busy
    series runs 48.85, 48.91, 48.87, 48.88, 47.82 and then steps to 40.19 and stays there to
    within 0.03% for ten inferences. Averaging across the step reports 42.6 ms for a machine that
    settles at 40.20 ms — an 6% overstatement produced entirely by counting the ramp.

    # Why the criterion is the GPU clock and not the host clock

    A warmup detector run on host wall time cannot tell a ramp from a contended machine; both look
    like "early samples are slower". The device timestamp counter does not see host load at all,
    so a step in *this* series is a property of the device or the driver, and finding the tail
    this way stays valid on a box that is not quiet. That is the only reason this figure is
    reportable today.

    Returns ``verdict`` ``STEADY`` with the tail's statistics, ``MARGINAL_TAIL`` when a suffix
    settles but is too short or too small a share of the run to be quoted, ``NO_STEADY_TAIL``
    when no sufficiently long suffix settles, or ``INSUFFICIENT``.

    # Why a settled suffix is not automatically a quotable one — the minimum-n floor

    *Ruling of 2026-08-01, mine, on Morpheus's question.* **The 2% RSD bar is a constraint on the
    tail's internal spread, not on its agreement with the device's true steady rate.** A suffix of
    five samples taken from a local flat stretch of a wandering series clears it exactly as easily
    as a settled device does, and reports a median that is simply wrong. Two independent
    specimens, from two different people's runs, on the same day:

    * Switch's ``contended`` row **passed** at ``n=8`` after discarding 38 — and sat **2.1% above**
      his solo figure, which the RSD said nothing about.
    * My own pre-barrier-fix A/B runs produced tails at ``n=7`` (39 discarded, median 20.06 ms)
      and ``n=5`` (38 discarded, median 37.56 ms) on a device whose two clean runs, from the same
      DLL on the same afternoon, both read **13.346 ms** to four figures. Each bad tail
      disagreed with its own run's warm-mean GPU busy by 33% and 42% respectively. Every tail with
      ``n >= 38`` in that set agreed with its run's warm mean to within 0.2%.

    The separation is clean and it is not on ``n`` alone: it is on **how much of the series the
    tail keeps**. A genuine warmup ramp is a short prefix (5-6 of 46 here); a device that never
    settled produces a short flat *suffix* (38-39 of 46 discarded). Coverage is therefore the
    floor that does the discriminating — every bad specimen above sits at 12-17% coverage, every
    good one at 83-100% — and the absolute ``n`` floor exists only to reject a series too short
    to have shown anything. Both apply, and a suffix that clears the RSD bar but fails either is
    ``MARGINAL_TAIL``: not a slower number, **no number**, with the suffix's median kept under
    ``withheld_median_ms`` so that it cannot be read as one by a later reader or an aggregation.

    This deliberately makes the instrument refuse more often. It refuses the two runs above, and
    it would have refused Switch's ``contended`` row. That is the point: those are the runs whose
    numbers were wrong, and a refusal that costs a real measurement is cheaper than a pass that
    ships a fabricated one.

    # The floor above is necessary and it is nowhere near sufficient -- amendment of 2026-08-01

    **Same day, later: the premise I ruled on was withdrawn, and the floor is not the fix.** I was
    told this gate "either lands within 0.08% of solo or refuses outright". Switch's
    ``probe_gputenancy.py`` shows the opposite, from committed artifacts, with no GPU needed:
    ``contended3`` truncated to 20, 28 and 34 inferences reports **STEADY at 126.647 ms, 10.99x
    wrong, RSD 0.79-0.91%**; and a board held at its 210 MHz idle clock against a 3105 MHz boost
    reports **STEADY at 246.735 ms, 21.4x wrong, RSD 0.1163%, nothing discarded**.

    **In both failures the wrong number carried the BETTER RSD than the right one.** That is the
    whole lesson and it is fatal to any repair from inside this function. This is a variance test
    over a suffix: **it cannot see a bias.** A uniformly wrong series is a *perfectly steady* one,
    so the gate answers a run that is entirely wrong with its most confident possible verdict. A
    low clock does not raise RSD; it lowers it.

    So: **more samples make a biased series more confident, not less.** The ``n`` and coverage
    floors above do real work -- they catch a *wandering* device, which is a different failure and
    a real one -- but they must never be presented as closing this. Under DESIGN.md **R9 rule 5**
    the remedy for an anti-correlated falsifier is **a different instrument**, and this check is
    demoted from a gate to a **precondition**. Every tail returned here is therefore born
    ``certification: UNCERTIFIED``, and only :mod:`bench.device_state` -- a tenancy verdict and an
    SM-clock record taken over the same window, from outside this series -- can lift it.
    """
    vals = [v / 1000.0 for v in busy_us if v]
    out = {"n_inferences": len(vals),
           "series_ms": [round(v, 3) for v in vals][:64],
           # R9 rule 5. Set here, unconditionally, so that *every* tail this function returns is
           # born unquotable. `analyse(device_state=...)` is the only thing that can lift it, and
           # only on the evidence of a second instrument that watched the same window. A default
           # of "quotable until someone objects" is the shape that let a 21.4x-wrong figure out.
           "certification": {"verdict": "UNCERTIFIED", "quotable": False,
                             "detail": ("no device-state companion was supplied to "
                                        "phases.analyse(). An RSD over a suffix is silent about "
                                        "the level of that suffix; see bench/device_companion.py.")}}
    if len(vals) < GPU_TAIL_MIN_N + 1:
        out.update(verdict="INSUFFICIENT",
                   detail=f"{len(vals)} usable inferences; need at least {GPU_TAIL_MIN_N + 1}.")
        return out
    best = None
    for start in range(0, len(vals) - GPU_TAIL_MIN_N + 1):
        suffix = vals[start:]
        mean = statistics.fmean(suffix)
        if mean <= 0:
            continue
        rsd = statistics.pstdev(suffix) / mean
        if rsd <= GPU_TAIL_RSD_MAX:
            best = (start, suffix, rsd)
            break
    if best is None:
        out.update(verdict="NO_STEADY_TAIL",
                   detail=(f"no suffix of >= {GPU_TAIL_MIN_N} inferences holds GPU busy time "
                           f"within {GPU_TAIL_RSD_MAX:.0%} RSD. The device never settled, so "
                           f"there is no steady-state GPU figure to quote from this run."))
        return out
    start, suffix, rsd = best
    coverage = len(suffix) / len(vals)
    stats_block = {
        "discarded_inferences": start,
        "n": len(suffix),
        "coverage": round(coverage, 4),
        "median_ms": round(statistics.median(suffix), 4),
        "mean_ms": round(statistics.fmean(suffix), 4),
        "min_ms": round(min(suffix), 4),
        "max_ms": round(max(suffix), 4),
        "rsd": round(rsd, 6),
    }
    if len(suffix) < GPU_TAIL_QUOTABLE_MIN_N or coverage < GPU_TAIL_MIN_COVERAGE:
        # Not a number. The suffix is flat, and a flat suffix that is a small piece of the series
        # is as easily a local excursion as a settled device -- see the docstring's specimens,
        # where every such tail disagreed with its own run's warm mean by 33-42%.
        out.update(stats_block)
        out.update(
            verdict="MARGINAL_TAIL",
            median_ms=None,
            withheld_median_ms=stats_block["median_ms"],
            detail=(f"a suffix of {len(suffix)} inference(s) holds within {rsd:.4%} RSD, but it "
                    f"is {coverage:.0%} of the {len(vals)} usable inferences and "
                    f"{start} were discarded to find it. Floors: n >= "
                    f"{GPU_TAIL_QUOTABLE_MIN_N} and coverage >= "
                    f"{GPU_TAIL_MIN_COVERAGE:.0%}. The RSD bar constrains the tail's internal "
                    f"spread, not its agreement with the device's true steady rate, so a short "
                    f"flat suffix passes it just as well as a settled one. No GPU figure is "
                    f"quotable from this run; the suffix's median is kept under "
                    f"`withheld_median_ms` so it cannot be read as one."),
        )
        return out
    out.update(stats_block)
    out.update(
        verdict="STEADY",
        detail=(f"GPU busy settles after {start} inference(s) and holds "
                f"{statistics.median(suffix):.3f} ms across {len(suffix)} of them "
                f"({coverage:.0%} of the series) at "
                f"{rsd:.4%} RSD. Device-clock only: this is the summed duration of the "
                f"dispatches, not the wall time of an inference, and it is NOT a substitute "
                f"for the end-to-end figure. STEADY is a precondition and not a release: this "
                f"same verdict was returned at 10.99x and at 21.4x wrong, both times with a "
                f"better RSD than the correct run. See `certification` -- until a device-state "
                f"companion certifies it, there is no quotable number here."),
    )
    return out


def steady_state_split(subgraphs: "list[dict]", siblings: "list[dict]",
                       nested: "list[dict]", busy_us: "dict[int, float] | None" = None,
                       transfers: "list[dict] | None" = None) -> dict:
    """The phase split over warm inferences only, reported beside the cold one it excludes.

    # Why this had to exist the day persistent weight residency landed

    Before residency, every inference re-uploaded the weights, so every inference cost roughly the
    same and averaging over all of them was harmless. After residency the **first** ``Compute``
    call uploads 1997.977 MiB and every later one uploads 0.387 MiB — a 5162× step, in the same
    trace, under the same span name.

    Averaged over four inferences that single upload is 91% of the ``record`` total and 99.98% of
    the ``cmd_upload`` total, so an all-inference share table reports the cost of a *fixed
    one-time transfer* as if it were a per-inference cost, and points optimisation at the
    transfer that was just eliminated. That is the record-is-68% mistake for the third time, with
    residency as the new disguise: the mean of two populations, presented under the name of one of
    them.

    The cold span is not discarded — it is reported separately, because a user pays it once and it
    is the correct place to read model-load cost.

    # How warm is decided

    From the trace's own repeat structure, not from a warmup flag the harness passes in. Islands
    execute in a fixed order once per inference, so the cycle period recovered by
    :func:`_cycle_period` is the island count, and the first cycle is the cold one. Fewer than
    three cycles and this returns ``INSUFFICIENT`` rather than a number: two warm samples cannot
    show that they agree with each other.
    """
    order = [s["nodes"] for s in subgraphs]
    period = _cycle_period(order) or (1 if len({*order}) == 1 else None)
    if period is None:
        return {"verdict": "NO_CYCLE",
                "detail": ("the subgraph order has no repeating period, so which invocations are "
                           "the first-of-each-island cannot be recovered from the trace.")}
    cycles = len(subgraphs) // period
    cold = {s["index"] for s in subgraphs[:period]}
    warm = [s for s in subgraphs[period:]]
    if cycles < 3:
        return {"verdict": "INSUFFICIENT", "cycles": cycles, "island_count": period,
                "detail": (f"{cycles} inference(s) in this trace; at least 3 are needed to drop "
                           f"the cold one and still have two warm samples that can disagree.")}

    warm_idx = {s["index"] for s in warm}
    warm_ms = sum(s["dur"] for s in warm) / 1000.0
    cold_ms = sum(s["dur"] for s in subgraphs if s["index"] in cold) / 1000.0

    def fold(spans, keep):
        out: "dict[str, float]" = {}
        for p in spans:
            if p["subgraph_index"] in keep:
                out[p["phase"]] = out.get(p["phase"], 0.0) + p["dur"] / 1000.0
        return out

    warm_sib, warm_kid = fold(siblings, warm_idx), fold(nested, warm_idx)
    cold_sib, cold_kid = fold(siblings, cold), fold(nested, cold)
    warm_gpu = sum((busy_us or {}).get(i, 0.0) for i in warm_idx) / 1000.0
    record_leaf = warm_sib.get("record", 0.0) - sum(warm_kid.values())
    fence = warm_sib.get("fence_wait", 0.0)
    accounted = sum(warm_sib.values())

    up_cold = up_warm = None
    if transfers:
        ups = [t for t in transfers if t["direction"] == "upload"]
        if ups:
            up_cold = round(ups[0]["bytes"] / 1024 ** 2, 3)
            rest = [t["bytes"] for t in ups[1:]]
            up_warm = round(statistics.fmean(rest) / 1024 ** 2, 4) if rest else None

    n = cycles - 1
    tail = gpu_steady_tail([(busy_us or {}).get(s["index"]) for s in subgraphs])
    return {
        "verdict": "OK",
        "island_count": period,
        "cycles": cycles,
        "warm_inferences": n,
        "cold_in_compute_ms": round(cold_ms, 3),
        "warm_in_compute_ms": round(warm_ms, 3),
        "warm_per_inference_ms": round(warm_ms / n, 3),
        "cold_over_warm_ratio": (round(cold_ms / (warm_ms / n), 2) if warm_ms else None),
        "warm_phases_ms": {k: round(v, 3) for k, v in sorted(warm_sib.items())},
        "warm_nested_ms": {k: round(v, 3) for k, v in sorted(warm_kid.items())},
        "cold_phases_ms": {k: round(v, 3) for k, v in sorted(cold_sib.items())},
        "cold_nested_ms": {k: round(v, 3) for k, v in sorted(cold_kid.items())},
        "warm_shares": ({
            **{k: round(v / warm_ms, 5) for k, v in warm_sib.items() if k != "record"},
            "record_INCLUDING_children": round(warm_sib.get("record", 0) / warm_ms, 5),
            "record_excl_children": round(record_leaf / warm_ms, 5),
            **{f"in_record/{k}": round(v / warm_ms, 5) for k, v in warm_kid.items()},
            "gpu_busy": round(warm_gpu / warm_ms, 5),
            "fence_wait_gpu_idle": round(max(fence - warm_gpu, 0.0) / warm_ms, 5),
            "unattributed": round((warm_ms - accounted) / warm_ms, 5),
        } if warm_ms else {}),
        "warm_gpu_busy_ms": round(warm_gpu, 3),
        "gpu_steady_tail": tail,
        "upload_mib_cold": up_cold,
        "upload_mib_warm_mean": up_warm,
        "gpu_busy_note": (
            "GPU busy overlaps submit+fence_wait and is NOT a sibling of the host phases — it is "
            "printed as a share of the same denominator so the two can be compared, never so "
            "they can be added. `fence_wait_gpu_idle` is the part of the fence wait the GPU was "
            "demonstrably not executing in, and it is the only GPU-idle figure that may be "
            "quoted."),
        "duration_caveat": (
            "these are durations and durations move with machine load. The *structure* here — "
            "that recording is paid on every inference rather than once, and that the cold "
            "inference is a different workload from the warm ones — is a count of spans and "
            "survives contention. The percentages do not."),
    }


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def analyse(events: "list[dict]", counters: "dict | None" = None,
            integrated_gpu: bool = False,
            independent_whole_ms: "float | None" = None,
            whole_source: str = "",
            device_state: "dict | None" = None) -> dict:
    """The whole phase picture for one trace, with its falsifiers attached.

    ``device_state`` is the **required companion** for any device-clock figure: a tenancy verdict
    and an SM-clock record taken by :mod:`bench.device_state` over the same window as this trace.
    Omitting it is not a shortcut -- the GPU steady tail stays ``UNCERTIFIED`` and its median is
    not a quotable number. See :func:`gpu_steady_tail`'s amendment of 2026-08-01.
    """
    subs = subgraph_spans(events)
    all_phases = phase_spans(events)
    nesting = phase_nesting(all_phases)
    attributed = attribute(subs, all_phases)
    # Only true siblings may be summed. Nested sub-record spans (desc_alloc, pipeline_lookup,
    # cmd_upload) are real X spans inside vulkan.record; adding them to the host total counts the
    # same microseconds twice. They are reported separately, as a breakdown of `record`.
    sib_spans = sibling_phases(all_phases)
    nested_names = sorted({p["phase"] for p in all_phases} - {p["phase"] for p in sib_spans})
    siblings = attribute(subs, sib_spans)
    nested_attr = attribute(subs, [p for p in all_phases if p["phase"] in nested_names])
    gpus = gpu_spans(events)
    transfers = transfer_events(events)
    scaling = record_scaling(attributed, subs, transfers)
    # record_scaling owns the containment attribution; host_phase_totals consumes it rather than
    # re-deriving it, so there is exactly one answer to "how much of record is upload".
    counter_upload_ms = (scaling["upload_inside_record_ms"]
                         if scaling.get("usable")
                         and scaling.get("upload_inside_record_ms") is not None else None)
    # On traces that carry sub-record spans, `cmd_upload` measures the same memcpy the transfer
    # counters measure. Adding both would double-count it; picking one silently would hide a
    # disagreement between two instruments. Prefer the span (it is a wall-clock interval, not a
    # rate inverted back into a duration) and record whether they agree.
    nested_ms = {n: sum(p["dur"] for p in attributed if p["phase"] == n) / 1000.0
                 for n in nested_names}
    span_upload_ms = nested_ms.get("cmd_upload")
    upload_agreement = upload_accounting(counter_upload_ms, span_upload_ms)
    resolved_upload = span_upload_ms if span_upload_ms is not None else counter_upload_ms
    child_ms = {}
    if resolved_upload is not None or nested_ms:
        child_ms["record"] = (resolved_upload or 0.0) + sum(
            v for k, v in nested_ms.items() if k != "cmd_upload")
    # Name only the children this trace actually contains, and that were actually subtracted.
    # Listing PHASE_CHILDREN wholesale would claim to have accounted for sub-phases that are not
    # in the trace at all -- the same over-claiming this module exists to prevent.
    accounted_children = tuple(n for n in PHASE_CHILDREN["record"]
                               if n in nested_names
                               or (n == "upload" and span_upload_ms is None
                                   and counter_upload_ms is not None))
    host = host_phase_totals(siblings, child_ms, {"record": accounted_children})
    nested_totals = host_phase_totals([p for p in attributed if p["phase"] in nested_names])
    gpu = gpu_totals(gpus)

    # The denominator for "share of what". The subgraph spans are the EP's own view of the time
    # ORT spent inside Compute; it is not the process wall clock and is not labelled as such.
    in_compute_ms = sum(s["dur"] for s in subs) / 1000.0
    shares = {
        k: round(v["total_ms"] / in_compute_ms, 5)
        for k, v in host.items() if v.get("n") and in_compute_ms > 0
    }
    # A share carries the same misreading risk as the total it is computed from: "record is 68%"
    # was a share. Non-leaf shares are re-emitted under an explicitly parent-shaped name and the
    # leaf residual is given its own share, so the honest number is the one closest to hand.
    for k, v in list(host.items()):
        if v.get("is_leaf", True) or k not in shares:
            continue
        shares[f"{k}_INCLUDING_{'_'.join(v['contains'])}"] = shares.pop(k)
        if v.get("leaf_ms") is not None and in_compute_ms > 0:
            shares[f"{k}_excl_{'_'.join(v['contains'])}"] = round(v["leaf_ms"] / in_compute_ms, 5)

    phased_ms = sum(v["total_ms"] for v in host.values() if v.get("n"))
    leaf_acct = phase_leaf_accounting(host)
    if gpu["all"].get("n") and in_compute_ms > 0:
        shares["gpu_kernels"] = round(gpu["all"]["total_ms"] / in_compute_ms, 5)
    ordinal = attribute_gpu_ordinally(subs, gpus)
    contention = contention_signature(attributed, subs, ordinal.get("busy_us"), integrated_gpu)

    report = {
        "subgraph_spans": len(subs),
        "time_in_compute_ms": round(in_compute_ms, 3),
        "time_in_compute_note": (
            "the sum of vulkan.subgraph spans — the EP's own view of time inside Compute. It is "
            "NOT the process wall clock: ORT's own graph execution, the CPU EP's nodes between "
            "islands, and session setup are all outside it. Phase shares below are shares of "
            "this, and may not be restated as shares of the benchmark's wall time."),
        "host_phases_ms": host,
        "nested_phases_ms": nested_totals,
        "nested_phases_note": (
            "sub-record spans, reported separately because they are INSIDE vulkan.record. They "
            "are excluded from host_phases_ms and from every share, since adding them to their "
            "own parent counts the same microseconds twice."),
        "phase_nesting": nesting,
        "decomposition_identity": decomposition_identity(
            host, gpu, in_compute_ms, independent_whole_ms, whole_source),
        "phase_leaf_accounting": leaf_acct,
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
        "steady_state": steady_state_split(subs, siblings, nested_attr,
                                           ordinal.get("busy_us"), transfers),
        "contention_signature": contention,
        "falsifiers": {
            "phase_containment": phase_containment(subs, siblings, nested_attr),
            "gpu_span_accounting": gpu_span_accounting(subs, gpus, counters),
            "gpu_containment": gpu_containment(subs, attributed, gpus),
            "timestamp_conversion_integrality": timestamp_conversion_integrality(gpus),
            "valid_bits_applied": valid_bits_applied(gpus),
            "trace_matches_counters": trace_matches_counters(subs, counters),
            "upload_accounting": upload_agreement,
        },
    }
    # R9 rule 5. The tail arrives UNCERTIFIED from `gpu_steady_tail`; only a second instrument,
    # which watched the same window from outside the series, can lift that. Passing no companion
    # leaves it UNCERTIFIED -- there is no code path here that turns absence of evidence into a
    # pass, because that is precisely how a 21.4x-wrong figure was published with a 0.12% RSD.
    from device_companion import certify as _certify
    tail = (report.get("steady_state") or {}).get("gpu_steady_tail")
    if isinstance(tail, dict):
        tail["certification"] = _certify(tail, device_state)
    report["device_state"] = device_state or {
        "verdict": "ABSENT",
        "detail": ("no tenancy verdict and no SM-clock record was taken over this trace's window. "
                   "Device-clock figures in this report are UNCERTIFIED."),
    }
    return report


def red_flags(report: dict) -> "list[str]":
    """Every falsifier that went red, as text. Empty means every check that *could* fail did not.

    R13: a check has three terminal states. ``ERROR(instrument)`` is listed here — an unchecked
    claim is still not quotable — but it is prefixed so that it is never read as a detection.
    "5 failed" was read as a working guard when the guard was throwing ``NameError``; the text of
    a failure, not its count, is the thing that carries the diagnosis.
    """
    out = []
    for name, f in (report.get("falsifiers") or {}).items():
        if f.get("state") == "ERROR":
            out.append(f"{name}: ERROR(instrument) — NOT a detection, the check did not run: "
                       f"{f.get('instrument_error') or f.get('detail')}")
        elif f.get("red"):
            out.append(f"{name}: {f.get('detail')}")
    ti = (report.get("falsifiers") or {}).get("timestamp_conversion_integrality") or {}
    if ti and not ti.get("decisive"):
        out.append("timestamp_conversion_integrality: NOT DECISIVE on this device — "
                   + str(ti.get("detail")))
    cs = report.get("contention_signature") or {}
    if cs.get("verdict") in ("HOST_SIDE_EXCURSIONS", "HOST_EXCURSIONS_UNCONTROLLED",
                             "WORKLOAD_VARIATION", "UNDERPOWERED", "UNTESTABLE"):
        out.append(f"contention_signature: {cs['verdict']} — {cs.get('reason')}")
    cert = ((report.get("steady_state") or {}).get("gpu_steady_tail") or {}).get("certification")
    if cert and not cert.get("quotable"):
        kind = cert.get("verdict")
        prefix = ("gpu_steady_tail: ERROR(instrument=device_state) — NOT a detection: "
                  if kind == "ERROR" else f"gpu_steady_tail: {kind} — ")
        out.append(prefix + str(cert.get("detail")))
    la = report.get("phase_leaf_accounting") or {}
    if la and not la.get("ok"):
        out.append(f"phase_leaf_accounting: {la.get('verdict')} — {la.get('detail')}")
    pn = report.get("phase_nesting") or {}
    if pn and not pn.get("ok"):
        out.append(f"phase_nesting: {pn.get('verdict')} — {pn.get('detail')}")
    di = report.get("decomposition_identity") or {}
    if di and not di.get("ok"):
        out.append(f"decomposition_identity: {di.get('verdict')} — {di.get('detail')}")
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
            leaf = s.get("is_leaf", True)
            key = (phase if leaf
                   else f"{phase}_INCLUDING_{'_'.join(s.get('contains') or ())}")
            mark = "" if leaf else "  <- NOT A LEAF"
            lines.append(f"    vulkan.{phase:<11} {s['total_ms']:>10.2f} ms  "
                         f"{shares.get(key, 0) * 100:5.1f}%   n={s['n']:<6} "
                         f"median {s['median_ms']:.3f} ms{mark}")
            if not leaf:
                kids = "+".join(s.get("contains") or ())
                if s.get("leaf_ms") is None:
                    lines.append(f"      ! this span also contains {kids}, which this trace does "
                                 f"not let us subtract. Do NOT quote it as '{phase}'.")
                else:
                    lines.append(
                        f"      = {s['child_ms']:.2f} ms {kids} "
                        f"({(s.get('child_share') or 0) * 100:.1f}%) + "
                        f"{s['leaf_ms']:.2f} ms actual {phase} "
                        f"({shares.get(f'{phase}_excl_{kids}', 0) * 100:.1f}% of Compute)")
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
