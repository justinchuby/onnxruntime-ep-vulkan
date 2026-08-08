"""Attribute the Vulkan-vs-CUDA gap to a *cause*, not to a name.

`bench/cuda_competition.py` answers "which is faster, and by how much, with what
confidence".  It deliberately does not answer "why", because the two questions want
different instruments and mixing them is how a benchmark turns into a story.  This
module is the "why".

# The question this module exists to answer

The baseline says Vulkan takes ~45 ms where CUDA takes ~16 ms on Phi-3.5 `prefill_1`.
There are only a few places 30 ms can be:

1. **Kernel time** — our shaders are slower than cuBLAS/cuDNN's at the same work.
2. **Host overhead** — descriptor allocation, pipeline lookup, command recording,
   submission; work the GPU never sees.
3. **Transfer** — staging host->device and device->host bytes crossing PCIe.
4. **CPU fallback** — nodes the EP declined, executed by the CPU EP inside our wall time.
5. **Serialisation** — a submit/fence-wait per island rather than per inference.

These demand *completely different* fixes, and four of the five are invisible to a
faster shader.  Guessing between them is the single most expensive mistake available
here, so this module measures the split before anything is optimised.

# How the split is measured

The Vulkan EP already carries the instrument: `rust/src/trace.rs` emits a Chrome
Trace with

* `cat == "ep"` **structural** spans — `vulkan.compute_call` (the instrumented
  success-path region opened inside `compute_impl`, absent when a call early-outs
  before that; buckets every EP span the instrumented path emits, not all of ORT's
  `Compute` entry) containing `vulkan.subgraph` (opened inside `dispatch_ort`). These
  bracket regions and are never summed into anything;
* `cat == "ep.phase"` host spans (`vulkan.record`, `vulkan.submit`, `vulkan.fence_wait`,
  `vulkan.upload`, `vulkan.readback`, `vulkan.desc_alloc`, `vulkan.pipeline_lookup`,
  `vulkan.cmd_upload`), each carrying a `nested_in` arg;
* `cat == "gpu"` device spans (`vulkan.gpu.<label>`) with real `VkQueryPool` timestamps,
  emitted only when `ONNXRUNTIME_EP_VULKAN_TRACE_GPU=1` **and** the queue reports usable
  `timestampValidBits`;
* `cat == "counter"` samples for `vulkan.transfer_bytes`.

Three properties of that instrument drive this module's design and are worth stating
because getting any of them wrong silently produces a plausible wrong answer:

**The anchor must be the outer bracket.**  Every span is bucketed into the `Compute` call
whose window contains it.  Anchor on `vulkan.subgraph` while `vulkan.compute_call` exists
and everything between the two brackets falls into the "outside any call" bucket, where no
per-call total and no steady-state median can reach it.  On Phi-3.5 `prefill_1` that hidden
term was this harness's own counters-file dump running inside every timed inference, and it
was the same order of magnitude as the inference itself.  The pre-fix run is *not* a
committed artifact, so no figure is quoted for it here.  With the dump scoped to the first
run the same term collapses to a small fraction of a millisecond.  No magnitude is quoted
on this branch either: this head carries the instrument and none of its output, so there is
no committed profile to point at, and
`test_cuda_profile.py::test_every_documented_outside_subgraph_citation_matches_the_artifact`
reads the figure out of the artifact wherever one is committed.
The anchor stays, because the region was invisible for as long as nothing bracketed it.
:func:`choose_anchor` picks the outermost available span and records which one it used;
matching is by **exact name**, never `startswith`.

**`nested_in` is load-bearing.**  `vulkan.record` is an *inclusive* bracket: it contains
`upload`, `readback`, `desc_alloc`, `pipeline_lookup` and `cmd_upload`.  An aggregator
that sums spans by name double-counts the memcpy and then reports it as command-buffer
recording.  `phase_breakdown()` therefore sums *only* spans whose `nested_in == "none"`
into the sibling total, and reports the nested ones separately as a decomposition of
their parent — never added to it.  `record_residual_us` is the part of `record` that no
child claims, i.e. the actual `vkCmd*` calls.

**Host phases are not GPU time.**  `vulkan.submit` is asynchronous and measures no GPU
work at all; `vulkan.fence_wait` is an *upper bound* on kernel time (queue latency plus
execution).  The only figure in this file that is genuinely device time is
`gpu_ns_total`, and when GPU timestamps are unavailable it is `None` — never zero, and
never silently replaced by `fence_wait`.  A run that cannot see the device says so.

# Why tracing changes the number it measures

Enabling the tracer costs host time: a clock read and a span allocation per phase, per
dispatch.  With ~3200 dispatches per inference that is not negligible.  So this module
reports **shares, not absolute times**, and it reports the traced total alongside the
untraced baseline total so the observer effect is visible rather than assumed away.
The `overhead_ratio` field is exactly that: traced wall / untraced wall.  A share
computed from a run measurably slower than the run being explained is still the right
share as long as the inflation is spread over the phases proportionally — and where it
is not (it inflates host phases, not GPU ones), the direction of the bias is *toward*
host overhead, which is the hypothesis being tested.

**The untraced arm has no device measurement, so nothing may be subtracted across the
two.**  GPU timestamps come from a query pool the tracer arms, so an untraced run emits
none, and there is no untraced device time to compare against — there cannot be, from
this instrument.  Earlier revisions papered over that by dividing the *traced* process's
device median by the *untraced* process's wall and calling the result a
"tracer-overhead-immune bound", and by subtracting the same two operands to get a
"host-side residual".  Both are cross-process arithmetic wearing the clothes of a
within-run decomposition, and both are withdrawn — see ``withdrawn_terms`` on the
report.  Subtractions inside one process (``traced wall - traced device``) remain sound
and are what ``compute_reconciliation`` uses.

# The CUDA side

ORT's own profile gives per-node `*_kernel_time` events with a `provider` arg.  For the
CUDA arm those are real device kernels, so summing them by op type gives the competitor's
kernel budget directly.  The comparison that matters is not "our total vs their total"
(that mixes five causes) but "our kernel time vs their kernel time on the ops both
providers actually execute".  `op_type_comparison()` restricts to that intersection.

Usage
-----
    python -m bench.cuda_profile --workload phi35_prefill_1 --iters 10 \
        --out bench/results/_cuda69/profile_prefill_1.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import cuda_competition as cc  # noqa: E402
import cuda_workloads  # noqa: E402
from public_paths import dump_public_json, write_public_text  # noqa: E402

SCHEMA = "cuda_profile/2"

TRACE_ENV = "ONNXRUNTIME_EP_VULKAN_TRACE"
TRACE_GPU_ENV = "ONNXRUNTIME_EP_VULKAN_TRACE_GPU"

#: Phases whose wall time may be summed.  Mirrors ``Phase::is_sibling`` in
#: ``rust/src/trace.rs``; the trace itself carries ``nested_in`` per span, so this list is
#: a cross-check rather than the source of truth.  A disagreement between the two is
#: reported as a refusal instead of being resolved silently in either direction.
#:
#: The equality between this tuple and ``trace.rs`` is asserted by
#: ``bench/test_trace_vocabulary.py``, which parses ``Phase::as_str``/``Phase::nested_in``
#: through :mod:`bench.trace_vocabulary`.  It is *not* derived at import time on purpose:
#: this module must keep reducing a trace produced by a different checkout, and a reduction
#: that imports the current ``trace.rs`` cannot read last month's artifact.  The declaration
#: stays; the drift is caught by a test rather than by a shipped refusal.
SIBLING_PHASES = ("compile", "prepack", "record", "submit", "fence_wait")
NESTED_PHASES = ("upload", "readback", "desc_alloc", "pipeline_lookup", "cmd_upload")

#: ``cat == "ep"`` structural spans.  These bracket regions; they are **not** phases, they are
#: never summed into any total, and they add no level to the phase tree.
#:
#: ``vulkan.compute_call`` is the **instrumented success-path region inside** ``compute_impl``
#: — not ORT's literal ``Compute`` bracket.  It opens after the null check, ``this_info`` and
#: the ``guard_ffi_status`` entry, and closes before ``disclose_broken_commitment``, so an
#: early-out that never reaches ``compute_impl`` produces no span at all and the FFI guard's
#: own cost is outside it.  ``vulkan.subgraph`` opens inside ``dispatch_ort`` and is contained
#: by it.  Anchoring the per-call bucketing on the inner span means every span outside it —
#: however large — lands in the ``None`` bucket and appears in no per-call or steady-state
#: total.  On Phi-3.5 ``prefill_1`` that region was the harness's own counters dump, not the
#: EP, and it was the same order of magnitude as the inference; the pre-fix run is not a
#: committed artifact, so no figure for it is quoted here.  Once the dump is scoped to the
#: first run the term collapses to a small fraction of a millisecond
#: (``outside_subgraph_ms``); the magnitude is quoted only where a committed profile backs
#: it, which is not this branch.  The bracket is what made the number sayable at all.
COMPUTE_CALL_SPAN = "vulkan.compute_call"
SUBGRAPH_SPAN = "vulkan.subgraph"

#: Anchor spans in order of preference — outermost first.
ANCHOR_PREFERENCE = (COMPUTE_CALL_SPAN, SUBGRAPH_SPAN)

#: Verdicts for the attribution itself.
GPU_TIME_MEASURED = "GPU_TIME_MEASURED"
GPU_TIME_UNAVAILABLE = "GPU_TIME_UNAVAILABLE"
TRACE_ABSENT = "TRACE_ABSENT"
GPU_TIME_WITHHELD = "GPU_TIME_WITHHELD"

#: Verdicts a reader may quote a number from without reading anything else.
GREEN_VERDICTS = frozenset({GPU_TIME_MEASURED})


def seal_verdict(out: dict) -> dict:
    """Withhold a green verdict from a record that is simultaneously refusing.

    ``GPU_TIME_MEASURED`` beside a non-empty ``refusals`` list is a record disagreeing with
    itself, and it is the *token* people read: `bench/results/_cuda69`'s profile shipped
    ``verdict: "GPU_TIME_MEASURED"`` next to a self-declared instrument disagreement and
    passed as a green result for exactly that reason.

    The gate is structural, applied on every exit from :func:`attribute` rather than at each
    of the sites that can append a refusal, because "remember to downgrade the verdict when
    you add a refusal" is the kind of obligation that holds until the next site is added.
    Whoever appends a refusal does not have to know this rule exists.
    ``test_every_exit_from_attribute_is_sealed`` reads the source and enforces it.

    Nothing is deleted: the reduction is still on the record and still readable, and
    ``withheld_because`` says exactly which refusals cost it the green token.  What changes
    is that no reader can quote the number without meeting them.  Non-green verdicts are
    left alone — ``TRACE_ABSENT`` is already not a claim, and relabelling it would erase why
    it refused.
    """
    if out.get("verdict") in GREEN_VERDICTS and out.get("refusals"):
        out["withheld_from"] = out["verdict"]
        out["verdict"] = GPU_TIME_WITHHELD
        out["withheld_because"] = list(out["refusals"])
    return out


# ---------------------------------------------------------------------------
# Chrome-trace reduction
# ---------------------------------------------------------------------------

def _ep_spans(events: list, name: str) -> list:
    """``cat == "ep"`` complete spans whose name is **exactly** *name*, in time order.

    Equality, never ``startswith``.  The previous matcher was
    ``str(e.get("name", "")).startswith("vulkan.subgraph")``, and the EP simultaneously emitted
    instants named ``vulkan.compute[REPLAY]`` while a proposed whole-``Compute`` span was named
    ``vulkan.compute`` — separable only by ``cat``/``ph``, which the matcher did not read.  A
    prefix matcher over two vocabularies that share a prefix mixes them silently.  ``trace.rs``
    now asserts that no emitted name prefixes another
    (``no_trace_name_is_a_prefix_of_another``), and this side matches exactly, so neither
    guarantee is load-bearing alone.
    """
    out = [e for e in events
           if e.get("cat") == "ep" and e.get("ph") == "X" and e.get("name") == name]
    out.sort(key=lambda e: int(e["ts"]))
    return out


def choose_anchor(events: list) -> dict:
    """Which span brackets one ORT ``Compute`` call in *this* trace, and what it can see.

    Returns ``span``, ``basis``, ``count``, ``sees_whole_compute`` and ``available``.

    A trace written by an EP built before ``vulkan.compute_call`` existed has only the inner
    span, and refusing to reduce it would discard every stored artifact.  So the fallback is
    taken — and *recorded*, with ``sees_whole_compute = False``, because a reduction anchored on
    the inner span cannot make any statement about the region outside it and must not be read as
    though it had.
    """
    available = {name: len(_ep_spans(events, name)) for name in ANCHOR_PREFERENCE}
    for name in ANCHOR_PREFERENCE:
        if available[name]:
            return {
                "span": name,
                "count": available[name],
                "available": available,
                "sees_whole_compute": name == COMPUTE_CALL_SPAN,
                "basis": (
                    "`vulkan.compute_call` brackets the instrumented success-path region inside "
                    "`compute_impl`, so every span the EP emits on that path during the call "
                    "falls inside a bucket"
                    if name == COMPUTE_CALL_SPAN else
                    "FALLBACK: this trace has no `vulkan.compute_call` span, so the anchor is "
                    "`vulkan.subgraph`, which opens inside `dispatch_ort`. Anything the Compute "
                    "callback does outside `dispatch_ort` is invisible to this reduction and is "
                    "NOT included in any per-call or steady-state total below."),
            }
    return {"span": None, "count": 0, "available": available, "sees_whole_compute": False,
            "basis": "no anchor span of any kind in this trace"}


def compute_calls(events: list, anchor: "str | None" = None) -> list:
    """The anchor spans — one per ORT ``Compute`` call — in time order.

    These are the bucketing anchors for everything else.  Without them a reduction
    can only report cumulative totals, and cumulative totals on this EP are
    actively misleading: the first ``Compute`` uploads the whole int4 weight set
    (on Phi-3.5 ``prefill_1``, almost the entire ``cmd_upload`` total lands in one
    of fourteen calls).  Dividing that total by the call
    count attributes a large per-run staging cost to a steady state that pays none, which
    inverts the conclusion: it makes a warm run look transfer-bound when it is not.
    The magnitudes are quoted with the committed profile, which is not on this branch.

    ``rust/src/trace.rs`` warns about exactly this in ``Phase::Record``'s caveat —
    "the summary prints that residual CUMULATIVELY over all calls, so it mixes the
    two regimes".  The first version of this module walked into it anyway, which is
    why the split is structural here rather than a note.

    The anchor is :data:`COMPUTE_CALL_SPAN` when the trace carries it and
    :data:`SUBGRAPH_SPAN` otherwise — see :func:`choose_anchor`.  Anchoring on the inner span
    while an outer one exists puts every span between them into the ``None`` bucket, where no
    per-call total and no steady-state median can reach it.
    """
    anchor = anchor or choose_anchor(events)["span"]
    if not anchor:
        return []
    return [{"index": i, "ts": int(e["ts"]), "dur": int(e.get("dur") or 0),
             "anchor": anchor, "nodes": (e.get("args") or {}).get("nodes")}
            for i, e in enumerate(_ep_spans(events, anchor))]


def bucket_by_call(events: list, calls: list) -> list:
    """Assign each phase/gpu/inner-subgraph span to the ``Compute`` call containing it.

    Containment is by start timestamp against ``[ts, ts+dur)``.  A span that falls
    in no call's window is kept in a ``None`` bucket rather than dropped — session
    setup, compile and prepack legitimately happen outside any ``Compute``, and
    silently discarding them would make the phase totals fail to reconcile with the
    trace they came from.

    When the anchor is ``vulkan.compute_call``, the inner ``vulkan.subgraph`` spans are
    bucketed too.  They are never summed with the phases: they are the *inner bracket*, and
    the difference between the two brackets is what :func:`_summarise_bucket` reports as
    ``outside_subgraph_us`` — measured, and attributed to nothing.
    """
    if not calls:
        return []
    anchor = calls[0].get("anchor", SUBGRAPH_SPAN)
    starts = [c["ts"] for c in calls]
    ends = [c["ts"] + c["dur"] for c in calls]
    buckets: list = [{"index": c["index"], "dur": c["dur"], "anchor": anchor,
                      "phases": [], "gpu": [], "subgraph": []} for c in calls]
    outside = {"index": None, "dur": None, "anchor": anchor,
               "phases": [], "gpu": [], "subgraph": []}
    import bisect

    for ev in events:
        cat = ev.get("cat")
        if ev.get("ph") != "X":
            continue
        if cat in ("ep.phase", "gpu"):
            key = "phases" if cat == "ep.phase" else "gpu"
        elif (cat == "ep" and anchor == COMPUTE_CALL_SPAN
              and ev.get("name") == SUBGRAPH_SPAN):
            key = "subgraph"
        else:
            continue
        ts = int(ev.get("ts") or 0)
        i = bisect.bisect_right(starts, ts) - 1
        target = buckets[i] if 0 <= i < len(calls) and ts < ends[i] else outside
        target[key].append(ev)
    return buckets + [outside]


def _summarise_bucket(bucket: dict) -> dict:
    """Phase, GPU and bracket totals for a single ``Compute`` call."""
    phases = phase_breakdown(bucket["phases"])
    gpu = gpu_breakdown(bucket["gpu"])
    call_us = bucket.get("dur")
    inner = [int(e.get("dur") or 0) for e in bucket.get("subgraph") or []]
    subgraph_us = sum(inner) if inner else None
    # The two brackets differ by whatever the Compute callback does outside `dispatch_ort`.
    # Reported, never charged: this reduction measures the size and the side of that region and
    # has no instrument that names its cause.
    outside_subgraph_us = (call_us - subgraph_us
                           if call_us is not None and subgraph_us is not None else None)
    # Time inside the *inner* bracket that no sibling phase covers — reading ORT input pointers,
    # buffer and descriptor work before recording, output tensor writes after the fence.
    covered = subgraph_us if subgraph_us is not None else call_us
    unattributed_us = (max(0, covered - phases["sibling_total_us"])
                       if covered is not None else None)
    return {
        "index": bucket["index"],
        "anchor": bucket.get("anchor"),
        "call_us": call_us,
        "subgraph_us": subgraph_us,
        "outside_subgraph_us": outside_subgraph_us,
        "unattributed_in_subgraph_us": unattributed_us,
        "sibling_total_us": phases["sibling_total_us"],
        "record_us": phases["record_us"],
        "record_residual_us": phases["record_residual_us"],
        "siblings_us": phases["siblings_us"],
        "nested_us": phases["nested_us"],
        "gpu_ns": gpu["total_ns"],
        "gpu_spans": gpu["span_count"],
        "gpu_per_label_ns": gpu["per_label_ns"],
    }


def steady_state(per_call: list) -> dict:
    """Median of the warm calls, with the cold first call reported beside it.

    The median rather than the mean: one warm call can still hit a driver hiccup on
    a shared box, and a mean lets it move the headline.  Both regimes are returned
    because the answer to "where does the time go" is genuinely different for each,
    and a reader who is told only one of them will apply it to the other.
    """
    warm = [c for c in per_call if c["index"] is not None and c["index"] > 0]
    cold = next((c for c in per_call if c["index"] == 0), None)
    if not warm:
        return {"warm_calls": 0, "detail": "no warm call to summarise",
                "cold": cold}

    def med(key):
        return float(statistics.median(c[key] for c in warm))

    def med_opt(key):
        """Median over the warm calls that carry *key*, or ``None`` when none do.

        ``None`` for "this trace's anchor cannot see that region", never ``0``: a region an
        instrument cannot see is not a region that costs nothing, and every reduction in this
        module that confused the two produced a plausible wrong answer.
        """
        vals = [c[key] for c in warm if c.get(key) is not None]
        return float(statistics.median(vals)) if vals else None

    labels: dict = defaultdict(lambda: [])
    for c in warm:
        for label, rec in c["gpu_per_label_ns"].items():
            labels[label].append(rec["ns"])
    label_med = {k: {"ns": float(statistics.median(v)), "calls": len(v)}
                 for k, v in sorted(labels.items(), key=lambda kv: -statistics.median(kv[1]))}

    sib_keys = {k for c in warm for k in c["siblings_us"]}
    nest_keys = {k for c in warm for k in c["nested_us"]}
    return {
        "warm_calls": len(warm),
        "cold": cold,
        "anchor": warm[0].get("anchor"),
        "median_call_us": med_opt("call_us"),
        "median_subgraph_us": med_opt("subgraph_us"),
        "median_outside_subgraph_us": med_opt("outside_subgraph_us"),
        "median_unattributed_in_subgraph_us": med_opt("unattributed_in_subgraph_us"),
        "median_sibling_total_us": med("sibling_total_us"),
        "median_record_us": med("record_us"),
        "median_record_residual_us": med("record_residual_us"),
        "median_gpu_ns": med("gpu_ns"),
        "median_gpu_spans": med("gpu_spans"),
        "median_siblings_us": {
            k: float(statistics.median(c["siblings_us"].get(k, {}).get("us", 0)
                                       for c in warm)) for k in sorted(sib_keys)},
        "median_nested_us": {
            k: float(statistics.median(c["nested_us"].get(k, {}).get("us", 0)
                                       for c in warm)) for k in sorted(nest_keys)},
        "median_gpu_per_label_ns": label_med,
    }


def load_trace(path: Path) -> list:
    """Read a Chrome Trace array, tolerating the trailing-comma forms tools emit."""
    text = path.read_text("utf-8", errors="replace").strip()
    if not text:
        return []
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # A crashed export can leave the array unterminated.  Salvage what parses
        # rather than discarding the whole run, but say so at the call site.
        text = text.rstrip().rstrip(",")
        if not text.endswith("]"):
            text += "]"
        return json.loads(text)


def phase_breakdown(events: list) -> dict:
    """Sum ``ep.phase`` spans, honouring ``nested_in``.

    Returns siblings and nested phases in *separate* maps.  They are never added
    together: ``record`` already contains every nested phase, so a single flat total
    would count the staging memcpy twice and attribute it to command recording.
    """
    siblings: dict = defaultdict(lambda: [0, 0])
    nested: dict = defaultdict(lambda: [0, 0])
    parents: dict = {}
    disagreements: list = []

    for ev in events:
        if ev.get("cat") != "ep.phase" or ev.get("ph") != "X":
            continue
        name = str(ev.get("name", ""))
        if not name.startswith("vulkan."):
            continue
        phase = name[len("vulkan."):]
        dur = int(ev.get("dur") or 0)
        args = ev.get("args") or {}
        nested_in = args.get("nested_in", "none")

        # Cross-check the span's own parentage claim against this module's table.
        expect_sibling = phase in SIBLING_PHASES
        if expect_sibling != (nested_in == "none"):
            msg = (f"phase {phase!r} declares nested_in={nested_in!r} but this module "
                   f"lists it as {'sibling' if expect_sibling else 'nested'}; "
                   f"trace.rs and cuda_profile.py disagree about the phase tree")
            if msg not in disagreements:
                disagreements.append(msg)

        bucket = siblings if nested_in == "none" else nested
        bucket[phase][0] += dur
        bucket[phase][1] += 1
        if nested_in != "none":
            parents[phase] = nested_in

    def _fmt(d):
        return {k: {"us": v[0], "count": v[1]} for k, v in sorted(d.items())}

    sibling_total = sum(v[0] for v in siblings.values())
    record_us = siblings.get("record", [0, 0])[0]
    # `upload` and `cmd_upload` bracket the *same* memcpy — cmd_upload is the outer of
    # the two in session.rs — so charging both to `record`'s children double-counts it.
    # Take the larger, which is the one that actually spans the work.
    upload_like = max(nested.get("upload", [0, 0])[0], nested.get("cmd_upload", [0, 0])[0])
    claimed_children = (upload_like
                        + nested.get("readback", [0, 0])[0]
                        + nested.get("desc_alloc", [0, 0])[0]
                        + nested.get("pipeline_lookup", [0, 0])[0])

    return {
        "siblings_us": _fmt(siblings),
        "nested_us": _fmt(nested),
        "nested_parent": parents,
        "sibling_total_us": sibling_total,
        "record_us": record_us,
        "record_claimed_children_us": claimed_children,
        # The vkCmd* calls themselves: what `record` costs that no child accounts for.
        # Negative would mean the children overlap in a way this reduction does not
        # model, so it is clamped and flagged rather than reported as a negative time.
        "record_residual_us": max(0, record_us - claimed_children),
        "record_residual_underflow": record_us < claimed_children,
        "phase_tree_disagreements": disagreements,
    }


def gpu_breakdown(events: list) -> dict:
    """Sum ``cat == "gpu"`` device spans by kernel label.

    These are the only genuine device times in the trace.  If the list is empty the
    caller must report :data:`GPU_TIME_UNAVAILABLE` — an absent measurement is not a
    measurement of zero, and substituting ``fence_wait`` here would silently convert an
    upper bound into a claim.
    """
    per_label: dict = defaultdict(lambda: [0, 0])
    for ev in events:
        if ev.get("cat") != "gpu" or ev.get("ph") != "X":
            continue
        name = str(ev.get("name", ""))
        label = name[len("vulkan.gpu."):] if name.startswith("vulkan.gpu.") else name
        args = ev.get("args") or {}
        ns = args.get("gpu_ns")
        if ns is None:
            # Fall back to the microsecond duration on the span, which is the same
            # number rounded; note the loss rather than dropping the span.
            ns = int(ev.get("dur") or 0) * 1000
        per_label[label][0] += int(ns)
        per_label[label][1] += 1
    total_ns = sum(v[0] for v in per_label.values())
    return {
        "per_label_ns": {k: {"ns": v[0], "count": v[1]}
                         for k, v in sorted(per_label.items(), key=lambda kv: -kv[1][0])},
        "total_ns": total_ns,
        "span_count": sum(v[1] for v in per_label.values()),
    }


def transfer_counters(events: list) -> dict:
    """Extract ``vulkan.transfer_bytes`` counter samples."""
    per_dir: dict = defaultdict(lambda: [0, 0])
    for ev in events:
        if ev.get("ph") != "C" or ev.get("name") != "vulkan.transfer_bytes":
            continue
        for key, value in (ev.get("args") or {}).items():
            per_dir[key][0] += int(value)
            per_dir[key][1] += 1
    return {k: {"bytes": v[0], "samples": v[1]} for k, v in sorted(per_dir.items())}


def record_paths(events: list) -> dict:
    """Count ``vulkan.path[...]`` instants — first-record vs replay vs re-record.

    A "steady state" that is re-recording its command buffer every call is not a steady
    state, and the ratio here is how you find that out.

    Matched on ``cat == "ep.path"``, not on the name.  The instants were called
    ``vulkan.compute[...]`` until this revision, which made them indistinguishable under a
    prefix matcher from a whole-``Compute`` span named ``vulkan.compute``; ``trace.rs`` now
    forbids any emitted name from prefixing another and these are ``vulkan.path[...]``.
    Reading the category rather than the name keeps this function correct for stored traces
    written under either name.
    """
    counts: dict = defaultdict(int)
    for ev in events:
        if ev.get("cat") != "ep.path":
            continue
        path = (ev.get("args") or {}).get("path")
        if path:
            counts[str(path)] += 1
    return dict(counts)


def rerecord_evidence(per_call: list) -> dict:
    """Decide whether warm calls re-record their command buffer, without ``ep.path``.

    ``record_paths`` reads ``cat == "ep.path"`` instants, which this EP build does not
    emit — so on the trace that matters it returns ``{}``, and an empty dict is not
    evidence of replay.  Reading it as "no re-recording seen" would be the same
    unmeasured-is-zero error the fallback fractions were fixed for.

    There is a direct substitute.  ``desc_alloc`` and ``pipeline_lookup`` fire once per
    dispatch *while recording*.  A replayed command buffer records nothing, so a warm
    call that replays must show zero of them; a warm call showing the same count as the
    cold call is demonstrably rebuilding the whole command stream every inference.  This
    infers from work actually observed rather than from an absent marker.
    """
    warm = [c for c in per_call if c["index"] is not None and c["index"] > 0]
    cold = next((c for c in per_call if c["index"] == 0), None)
    if not warm or cold is None:
        return {"verdict": "UNKNOWN", "detail": "need a cold and at least one warm call"}

    def n(call, key):
        return int((call.get("nested_us") or {}).get(key, {}).get("count", 0))

    warm_desc = [n(c, "desc_alloc") for c in warm]
    warm_pipe = [n(c, "pipeline_lookup") for c in warm]
    cold_desc, cold_pipe = n(cold, "desc_alloc"), n(cold, "pipeline_lookup")
    med_desc = float(statistics.median(warm_desc)) if warm_desc else 0.0

    if med_desc == 0:
        verdict = "REPLAYED"
        detail = ("warm calls allocate no descriptors, consistent with replaying a "
                  "prebuilt command buffer.")
    elif cold_desc and med_desc >= 0.9 * cold_desc:
        verdict = "RERECORDED_EVERY_CALL"
        detail = (f"warm calls allocate a median of {med_desc:.0f} descriptor sets "
                  f"against {cold_desc} on the cold call — the command buffer is "
                  f"rebuilt from scratch on every inference, so per-dispatch host cost "
                  f"is paid every time rather than once.")
    else:
        verdict = "PARTIALLY_RERECORDED"
        detail = (f"warm calls allocate a median of {med_desc:.0f} descriptor sets "
                  f"against {cold_desc} cold — some but not all recording is reused.")

    return {
        "verdict": verdict,
        "detail": detail,
        "cold_desc_alloc": cold_desc,
        "cold_pipeline_lookup": cold_pipe,
        "warm_desc_alloc_median": med_desc,
        "warm_pipeline_lookup_median": (float(statistics.median(warm_pipe))
                                        if warm_pipe else 0.0),
        "basis": "desc_alloc/pipeline_lookup fire once per dispatch during recording only",
    }


# ---------------------------------------------------------------------------
# ORT-profile reduction (the CUDA side)
# ---------------------------------------------------------------------------

def op_kernel_times(profile_path: Path) -> dict:
    """Per-op-type kernel time and node count from an ORT profile, by provider.

    Only ``cat == "Node"`` events whose name ends ``_kernel_time`` are counted, matching
    :func:`cuda_competition.read_profile_partition`.  ``op_name`` is the ONNX op type;
    ORT emits it as an arg, and a node whose arg is missing is bucketed as ``UNKNOWN``
    rather than dropped (dropping shrinks the denominator and flatters the result).
    """
    per_op: dict = defaultdict(lambda: {"us": 0, "runs": 0, "nodes": set(), "provider": None})
    try:
        events = json.loads(profile_path.read_text("utf-8", errors="replace"))
    except Exception:
        return {}
    if isinstance(events, dict):
        events = events.get("traceEvents", [])
    for ev in events:
        if ev.get("cat") != "Node":
            continue
        name = str(ev.get("name", ""))
        if not name.endswith("_kernel_time"):
            continue
        args = ev.get("args") or {}
        op = str(args.get("op_name") or "UNKNOWN")
        provider = str(args.get("provider") or "UNATTRIBUTED")
        key = (op, provider)
        rec = per_op[key]
        rec["us"] += int(ev.get("dur") or 0)
        rec["runs"] += 1
        rec["nodes"].add(name[: -len("_kernel_time")])
        rec["provider"] = provider
    return {f"{op}|{prov}": {"op_type": op, "provider": prov, "us": v["us"],
                             "runs": v["runs"], "nodes": len(v["nodes"])}
            for (op, prov), v in per_op.items()}


def op_type_comparison(vulkan_ops: dict, cuda_ops: dict) -> dict:
    """Compare per-op budgets, restricted to op types **both** providers executed.

    Comparing a Vulkan op against a CUDA op that ran on the CPU EP, or against an op the
    other provider fused away, is not a kernel comparison — it is a partitioning
    comparison wearing a kernel comparison's clothes.  The intersection is the only set
    where "their kernel vs our kernel" is a meaningful sentence.
    """
    def _by_type(ops: dict, want_gpu: bool) -> dict:
        out: dict = defaultdict(lambda: {"us": 0, "nodes": 0})
        for rec in ops.values():
            on_gpu = rec["provider"] not in ("CPUExecutionProvider", "UNATTRIBUTED")
            if on_gpu != want_gpu:
                continue
            out[rec["op_type"]]["us"] += rec["us"]
            out[rec["op_type"]]["nodes"] += rec["nodes"]
        return dict(out)

    v = _by_type(vulkan_ops, True)
    c = _by_type(cuda_ops, True)
    shared = sorted(set(v) & set(c))
    rows = []
    for op in shared:
        vu, cu = v[op]["us"], c[op]["us"]
        rows.append({"op_type": op, "vulkan_us": vu, "cuda_us": cu,
                     "vulkan_nodes": v[op]["nodes"], "cuda_nodes": c[op]["nodes"],
                     "ratio_vulkan_over_cuda": (vu / cu) if cu else None})
    rows.sort(key=lambda r: -(r["vulkan_us"] - r["cuda_us"]))
    return {
        "shared_op_types": rows,
        "vulkan_only_op_types": sorted(set(v) - set(c)),
        "cuda_only_op_types": sorted(set(c) - set(v)),
        # Vulkan fuses its island into one profile node, so this will usually be empty
        # on an LLM graph.  That is a finding about the instrument, not a bug: it is
        # exactly why the GPU-timestamp lane exists.
        "comparable": bool(rows),
    }


# ---------------------------------------------------------------------------
# Driving a traced Vulkan run
# ---------------------------------------------------------------------------

def run_traced_vulkan(workload, *, iters: int, warmup: int, scratch: Path,
                      device: int, seed: int, gpu_timestamps: bool = True,
                      timeout: int = 3600) -> dict:
    """Run the Vulkan arm with the tracer on, in its own process.

    Reuses :func:`cuda_competition.dispatch_arm`'s worker entry point rather than
    re-implementing session setup, so the traced run and the benchmarked run are the
    same code path with one environment variable changed.  Anything else would be
    profiling a program that is not the one being measured.
    """
    scratch.mkdir(parents=True, exist_ok=True)
    trace_path = scratch / f"trace_{workload.key}.json"
    if trace_path.exists():
        trace_path.unlink()

    prev = {k: os.environ.get(k) for k in (TRACE_ENV, TRACE_GPU_ENV)}
    os.environ[TRACE_ENV] = str(trace_path)
    if gpu_timestamps:
        os.environ[TRACE_GPU_ENV] = "1"
    else:
        os.environ.pop(TRACE_GPU_ENV, None)
    try:
        rec = cc.dispatch_arm(cc.ARM_VULKAN, workload, iters=iters, warmup=warmup,
                              scratch=scratch, device=device, seed=seed, timeout=timeout)
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    rec["trace_path"] = str(trace_path)
    return rec


def attribute(traced: dict, untraced: dict | None, trace_path: Path) -> dict:
    """Reduce one traced Vulkan run into a cause attribution."""
    out: dict = {
        "arm": cc.ARM_VULKAN,
        "workload": traced.get("workload"),
        "traced_verdict": traced.get("verdict"),
        "refusals": list(traced.get("refusals") or []),
        "instrument_errors": list(traced.get("instrument_errors") or []),
    }
    # This reduction is itself a Vulkan record, and the regime its inputs were measured
    # under is the difference between a clean median and a counter-inflated one, which on
    # the measured build was close to a factor of two.  No figure is quoted: the artifact
    # that carried the inflated side is no longer committed, and a number the tree cannot
    # back is the defect this branch removes.  Inheriting it
    # rather than leaving it absent is what stops the reduction from being the one Vulkan
    # artifact in the tree that does not say which regime it describes.
    out["counters_scope"] = traced.get("counters_scope")
    untraced_scope = (untraced or {}).get("counters_scope")
    if untraced is not None and untraced_scope != out["counters_scope"]:
        # `overhead_ratio` divides one by the other, so two different regimes make it a
        # ratio of two things that were not measured the same way.
        out["refusals"].append(
            f"the traced arm was measured with counters_scope={out['counters_scope']!r} and "
            f"the untraced arm with {untraced_scope!r}. Every term that relates the two "
            f"(overhead_ratio above all) compares different instrumentation regimes.")
    if not trace_path.is_file():
        out["verdict"] = TRACE_ABSENT
        out["refusals"].append(
            f"the EP wrote no trace to {trace_path}. Without it there is no phase split, "
            f"and a phase split guessed from a total is not a measurement.")
        return seal_verdict(out)

    events = load_trace(trace_path)
    out["trace_events"] = len(events)
    phases = phase_breakdown(events)
    gpu = gpu_breakdown(events)
    out["phases"] = phases
    out["gpu"] = gpu
    out["transfer_bytes"] = transfer_counters(events)
    out["record_paths"] = record_paths(events)
    out["refusals"].extend(phases["phase_tree_disagreements"])

    # Per-call split.  Cumulative phase totals on this EP are dominated by the cold
    # first Compute (99.7% of cmd_upload lands there), so a steady-state claim built
    # from them is wrong by more than the gap being chased.
    anchor = choose_anchor(events)
    out["anchor"] = anchor
    calls = compute_calls(events, anchor["span"])
    out["compute_calls"] = len(calls)
    if calls:
        buckets = bucket_by_call(events, calls)
        per_call = [_summarise_bucket(b) for b in buckets]
        out["per_call"] = per_call
        out["steady"] = steady_state(per_call)
        out["rerecord"] = rerecord_evidence(per_call)
        if not anchor["sees_whole_compute"]:
            # Not a refusal: the reduction below is sound about the region it can see. It is a
            # bounded claim, and the bound is stated on the artifact rather than left to the
            # reader to infer from a missing field.
            out.setdefault("scope_limits", []).append(anchor["basis"])
    else:
        out["per_call"] = []
        out["steady"] = None
        out["refusals"].append(
            "the trace has no `vulkan.compute_call` or `vulkan.subgraph` spans, so phases "
            "cannot be split by Compute call. Only cumulative totals are available, and on "
            "this EP those are dominated by the cold first call — do not read them as "
            "steady state.")

    traced_median = traced.get("median_ms")
    out["traced_median_ms"] = traced_median
    untraced_median = (untraced or {}).get("median_ms")
    out["untraced_median_ms"] = untraced_median
    out["overhead_ratio"] = (traced_median / untraced_median
                             if traced_median and untraced_median else None)

    if gpu["span_count"] == 0:
        out["verdict"] = GPU_TIME_UNAVAILABLE
        out["refusals"].append(
            "no GPU timestamp spans in the trace: either the queue reports "
            "timestampValidBits == 0 or ONNXRUNTIME_EP_VULKAN_TRACE_GPU was not set. "
            "fence_wait is an UPPER BOUND on kernel time and is not substituted here.")
        out["gpu_ms_total"] = None
        out["gpu_share_traced"] = None
        return seal_verdict(out)

    out["verdict"] = GPU_TIME_MEASURED
    # Total inferences the traced process actually ran: the compile/first run, the
    # warmups, and the steady-state iterations.  Derived from the sample lists rather
    # than from the requested counts, because a run that refused partway through has
    # fewer samples than it was asked for and dividing by the request would understate
    # per-run device time.
    runs = 1 + len(traced.get("warmup_ms") or []) + len(traced.get("steady_ms") or [])
    out["traced_runs"] = runs
    out["gpu_ms_total"] = gpu["total_ns"] / 1e6
    # Prefer the median of the WARM calls.  total/runs is an average over a population
    # that contains one cold outlier, and on the host-phase axis that outlier is three
    # orders of magnitude larger than its neighbours; the same reduction applied to the
    # device axis for consistency, so every per-run number in this report describes the
    # same regime.  The naive average is kept beside it so the two can be compared.
    steady = out.get("steady") or {}
    steady_gpu_ms = (steady.get("median_gpu_ns") / 1e6
                     if steady.get("median_gpu_ns") is not None else None)
    out["gpu_ms_per_run_mean_all_calls"] = gpu["total_ns"] / 1e6 / max(1, runs)
    gpu_ms_per_run = steady_gpu_ms if steady_gpu_ms is not None else out["gpu_ms_per_run_mean_all_calls"]
    out["gpu_ms_per_run"] = gpu_ms_per_run
    out["gpu_ms_per_run_basis"] = ("warm_call_median" if steady_gpu_ms is not None
                                   else "mean_over_all_calls")
    out["gpu_share_traced"] = (gpu_ms_per_run / traced_median) if traced_median else None
    # `gpu_share_untraced_bound` and `host_ms_per_run_residual` were reported here by
    # cuda_profile/1 and /2 and are WITHDRAWN, not corrected.  Both divided or subtracted
    # `gpu_ms_per_run` -- a device median that only the TRACED process can produce, because
    # the tracer is what arms the query pool -- against `untraced_median`, the wall clock of
    # a SECOND, SEPARATE process.  A quantity built across two processes cannot be a
    # within-run decomposition, and calling one of them a "bound that no amount of tracer
    # overhead can inflate" asserted equality of device time across two runs that this
    # instrument has no way to check: the untraced arm emits no device timestamps at all.
    #
    # They are withdrawn rather than renamed because this revision has no eligible author
    # for a replacement metric and will not invent one.  A reader who needs the host-side
    # split can subtract within one process: `traced_median - gpu_ms_per_run`, both warm-call
    # medians from the traced run, which `compute_reconciliation` already reports properly.
    out["withdrawn_terms"] = {
        "gpu_share_untraced_bound": (
            "= gpu_ms_per_run / untraced_median_ms. Withdrawn: numerator is the traced "
            "process's device median, denominator is a different process's wall. Not a "
            "bound, and not tracer-overhead-immune -- it assumes cross-run device time is "
            "identical, which this instrument cannot verify because the untraced arm "
            "produces no device timestamps."),
        "host_ms_per_run_residual": (
            "= untraced_median_ms - gpu_ms_per_run. Withdrawn: a subtraction across two "
            "processes wearing the clothes of a within-run decomposition. Whatever "
            "run-to-run variation separated the two processes is charged to 'host' by "
            "construction."),
        "both_operands_remain_published": (
            "gpu_ms_per_run and untraced_median_ms are unchanged and still on the record, "
            "so any reader who disagrees with this withdrawal can recompute the old fields "
            "exactly and see what they rest on."),
        "no_replacement_in_this_revision": (
            "Naming a sound cross-run quantity, and any re-measurement it would need, is "
            "deliberately left open. Nothing here was re-run."),
    }
    out["compute_reconciliation"] = compute_reconciliation(
        steady, traced_median, out.get("overhead_ratio"), anchor)
    return seal_verdict(out)


def compute_reconciliation(steady: dict, traced_median_ms, overhead_ratio,
                           anchor: dict) -> dict:
    """Split one warm inference into named, measured, non-overlapping regions.

    Every term is a **warm-call median from the traced run**, so they are on one axis and may
    be subtracted from each other.  They may **not** be subtracted from the untraced wall:
    that is a different process, tracing costs host time (``overhead_ratio`` here), and
    mixing the two scales is how a decomposition closes falsely.  The withdrawn
    ``host_ms_per_run_residual`` was exactly that mistake; see ``withdrawn_terms`` on the
    report.

    The regions, outermost first::

        traced wall
        └── vulkan.compute_call          instrumented success-path region, inside compute_impl
            ├── vulkan.subgraph          opened inside dispatch_ort
            │   ├── sibling phases       record + submit + fence_wait (+ compile/prepack)
            │   └── unattributed         no phase span covers it
            └── outside_subgraph         measured; NOT attributed by this instrument

    ``outside_subgraph`` is the region the previous revision of this module could not see at
    all, because it anchored on the inner span.  It is reported as a number and a side, and
    nothing else: this reduction has no instrument that can name its cause, and the last claim
    made about it (``Phase::BindCheck``, "the binding checks") was withdrawn — it accounted for
    a small fraction of a region orders of magnitude larger.  Neither the claim's magnitude nor
    the region's is quoted here: the run that produced them is not a committed artifact.

    **These medians do not algebraically partition.**  ``sibling_phases_ms`` +
    ``unattributed_in_subgraph_ms`` does not equal ``subgraph_ms``, and it is not meant to:
    each is an independently-taken median over the warm calls, and the median of a sum is not
    the sum of the medians unless every call splits the same way.  The residual is small and
    is an artefact of the statistic, not unaccounted time; the arithmetic is quoted with the
    committed profile that exhibits it, which is not on this branch.  ``partition_note`` on
    the returned dict carries the same statement computed from the run in hand, so a reader
    of the JSON alone cannot mistake the gap for a leak.
    """
    if not steady or not steady.get("warm_calls"):
        return {"available": False,
                "detail": "no warm call to reconcile; one call is not a steady state"}

    def ms(us):
        return None if us is None else us / 1000.0

    call = ms(steady.get("median_call_us"))
    sub = ms(steady.get("median_subgraph_us"))
    outside = ms(steady.get("median_outside_subgraph_us"))
    sibling = ms(steady.get("median_sibling_total_us"))
    unattr = ms(steady.get("median_unattributed_in_subgraph_us"))
    out_of_call = (traced_median_ms - call
                   if traced_median_ms is not None and call is not None else None)
    partition_note = None
    if sibling is not None and unattr is not None and sub is not None:
        partition_note = (
            f"these are independently-taken medians, not a partition: "
            f"sibling_phases_ms + unattributed_in_subgraph_ms = {sibling + unattr:.3f} ms "
            f"against subgraph_ms {sub:.3f} ms, a residual of {sub - sibling - unattr:+.3f} ms. "
            f"The median of a sum is not the sum of the medians unless every warm call splits "
            f"the same way, so this residual is a property of the statistic and NOT "
            f"unaccounted-for time. Do not report it as a gap.")
    return {
        "available": True,
        "axis": "traced-run warm-call medians, in milliseconds",
        "anchor": anchor.get("span"),
        "sees_whole_compute": anchor.get("sees_whole_compute"),
        "traced_wall_ms": traced_median_ms,
        "compute_call_ms": call,
        "subgraph_ms": sub,
        "outside_subgraph_ms": outside,
        "sibling_phases_ms": sibling,
        "unattributed_in_subgraph_ms": unattr,
        "outside_compute_call_ms": out_of_call,
        "partition_note": partition_note,
        "outside_subgraph_attribution": (
            "MEASURED, UNATTRIBUTED. This instrument reports the size of the region and which "
            "side of `vulkan.subgraph` it falls on. It does not name its cause."
            if outside else None),
        "do_not": ("subtract these traced-axis terms from `untraced_median_ms`; the traced run "
                   f"is {overhead_ratio:.3f}x the untraced one"
                   if overhead_ratio else
                   "subtract these traced-axis terms from an untraced wall"),
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render(report: dict) -> str:
    # A rendered report is what people read instead of the JSON, so the withheld verdict
    # has to arrive before the numbers, not in a REFUSAL footnote under them.
    seal_verdict(report)
    lines: list = ["# Vulkan vs CUDA — gap attribution", ""]
    lines.append(f"workload: `{report.get('workload')}`  ")
    lines.append(f"verdict: **{report.get('verdict')}**")
    if report.get("verdict") == GPU_TIME_WITHHELD:
        lines.append("")
        lines.append(f"> **{report.get('withheld_from')} is withheld.** This reduction "
                     f"carries {len(report.get('withheld_because') or [])} unresolved "
                     f"refusal(s), listed at the end of this report. The numbers below "
                     f"are printed so they can be checked, not so they can be quoted.")
    lines.append("")

    if report.get("untraced_median_ms"):
        lines.append(f"- untraced median: {report['untraced_median_ms']:.4f} ms")
    if report.get("traced_median_ms"):
        lines.append(f"- traced median: {report['traced_median_ms']:.4f} ms")
    if report.get("overhead_ratio"):
        lines.append(f"- tracer overhead ratio: {report['overhead_ratio']:.3f}x "
                     f"(host phases inflate, device time does not)")
    if report.get("cuda_median_ms"):
        lines.append(f"- CUDA median: {report['cuda_median_ms']:.4f} ms")
    lines.append("")

    gpu_ms = report.get("gpu_ms_per_run")
    steady = report.get("steady") or {}
    rec = report.get("compute_reconciliation") or {}
    if rec.get("available"):
        lines.append("## Where one warm inference goes")
        lines.append("")
        lines.append(f"Anchor: `{rec.get('anchor')}`"
                     + ("" if rec.get("sees_whole_compute")
                        else " — **inner span only; the region outside it is invisible here**"))
        lines.append("")
        lines.append("| region | ms | inside |")
        lines.append("|---|---:|---|")
        rows = [
            ("traced wall (one `session.run`)", rec.get("traced_wall_ms"), "—"),
            ("outside `vulkan.compute_call` (ORT + harness)",
             rec.get("outside_compute_call_ms"), "traced wall"),
            ("`vulkan.compute_call` (instrumented success-path region, inside `compute_impl`)",
             rec.get("compute_call_ms"), "traced wall"),
            ("`vulkan.subgraph` (inside `dispatch_ort`)",
             rec.get("subgraph_ms"), "`vulkan.compute_call`"),
            ("sibling phases (record+submit+fence_wait)",
             rec.get("sibling_phases_ms"), "`vulkan.subgraph`"),
            ("unattributed inside `vulkan.subgraph`",
             rec.get("unattributed_in_subgraph_ms"), "`vulkan.subgraph`"),
            ("**outside `vulkan.subgraph`, inside the callback**",
             rec.get("outside_subgraph_ms"), "`vulkan.compute_call`"),
        ]
        for label, value, parent in rows:
            lines.append(f"| {label} | {'—' if value is None else f'{value:.3f}'} | {parent} |")
        lines.append("")
        lines.append(f"All terms are {rec.get('axis')}. Do not {rec.get('do_not')}.")
        if rec.get("outside_subgraph_attribution"):
            lines.append("")
            lines.append(f"`outside_subgraph`: {rec['outside_subgraph_attribution']}")
        if rec.get("partition_note"):
            lines.append("")
            lines.append(f"**Read the table as terms, not as a sum.** {rec['partition_note']}")
        lines.append("")
    for limit in report.get("scope_limits") or []:
        lines.append(f"> SCOPE LIMIT: {limit}")
        lines.append("")
    if gpu_ms is not None:
        basis = report.get("gpu_ms_per_run_basis")
        lines.append(f"## Device time")
        lines.append("")
        lines.append(f"- GPU kernel time per run: **{gpu_ms:.4f} ms** "
                     f"(basis: {basis}, {steady.get('warm_calls', 0)} warm calls, "
                     f"{report.get('gpu', {}).get('span_count', 0)} timestamped spans "
                     f"across all calls)")
        if report.get("withdrawn_terms"):
            lines.append(f"- `gpu_share_untraced_bound` and `host_ms_per_run_residual` are "
                         f"**withdrawn**: both combined this traced-process device median "
                         f"with the untraced process's wall clock, which is a subtraction "
                         f"across two runs, not a decomposition of one. Their operands are "
                         f"still published above and in `untraced_median_ms`. No "
                         f"replacement is offered here and nothing was re-measured.")
        lines.append("")
        per_med = steady.get("median_gpu_per_label_ns")
        if per_med:
            lines.append("| kernel | device ms (warm-call median) | dispatches/call |")
            lines.append("|---|---:|---:|")
            for label, rec in list(per_med.items())[:20]:
                cnt = ((steady.get("cold") or {}).get("gpu_per_label_ns", {})
                       .get(label, {}).get("count"))
                lines.append(f"| `{label}` | {rec['ns'] / 1e6:.3f} | {cnt if cnt else '?'} |")
            lines.append("")
        else:
            lines.append("| kernel | device ms (all calls) | dispatches |")
            lines.append("|---|---:|---:|")
            per = report.get("gpu", {}).get("per_label_ns", {})
            for label, rec in list(per.items())[:20]:
                lines.append(f"| `{label}` | {rec['ns'] / 1e6:.3f} | {rec['count']} |")
            lines.append("")

    if steady.get("warm_calls"):
        cold = steady.get("cold") or {}
        lines.append("## Host phases per Compute call — cold vs steady")
        lines.append("")
        lines.append("Cumulative totals are not shown as a per-run figure: on this EP "
                     "the first `Compute` uploads the whole weight set and carries "
                     "almost the entire `cmd_upload` cost, so `total / calls` describes "
                     "no regime that actually occurs.")
        lines.append("")
        lines.append("| phase | cold call 0 (ms) | warm median (ms) |")
        lines.append("|---|---:|---:|")
        keys = list(steady.get("median_siblings_us", {}))
        for k in keys:
            c = (cold.get("siblings_us") or {}).get(k, {}).get("us", 0) / 1000
            w = steady["median_siblings_us"][k] / 1000
            lines.append(f"| `{k}` | {c:.3f} | {w:.3f} |")
        lines.append(f"| **sibling total** | "
                     f"{cold.get('sibling_total_us', 0) / 1000:.3f} | "
                     f"{steady.get('median_sibling_total_us', 0) / 1000:.3f} |")
        lines.append("")
        lines.append("| nested phase | cold call 0 (ms) | warm median (ms) | inside |")
        lines.append("|---|---:|---:|---|")
        parents = (report.get("phases") or {}).get("nested_parent") or {}
        for k in list(steady.get("median_nested_us", {})):
            c = (cold.get("nested_us") or {}).get(k, {}).get("us", 0) / 1000
            w = steady["median_nested_us"][k] / 1000
            lines.append(f"| `{k}` | {c:.3f} | {w:.3f} | `{parents.get(k, '?')}` |")
        lines.append(f"| `record` residual (vkCmd* calls) | "
                     f"{cold.get('record_residual_us', 0) / 1000:.3f} | "
                     f"{steady.get('median_record_residual_us', 0) / 1000:.3f} "
                     f"| `record` |")
        lines.append("")
        lines.append(f"- GPU device time, cold call 0: "
                     f"{cold.get('gpu_ns', 0) / 1e6:.3f} ms")
        lines.append(f"- GPU device time, warm median: "
                     f"{steady.get('median_gpu_ns', 0) / 1e6:.3f} ms")
        lines.append("")

    ph = report.get("phases") or {}
    if ph:
        lines.append("## Host phases, cumulative over every call (provenance only)")
        lines.append("")
        lines.append("| phase | ms | calls |")
        lines.append("|---|---:|---:|")
        for name, rec in (ph.get("siblings_us") or {}).items():
            lines.append(f"| `{name}` | {rec['us'] / 1000:.3f} | {rec['count']} |")
        lines.append(f"| **sibling total** | "
                     f"{ph.get('sibling_total_us', 0) / 1000:.3f} | |")
        lines.append("")
        lines.append("| nested phase | ms | calls | inside |")
        lines.append("|---|---:|---:|---|")
        for name, rec in (ph.get("nested_us") or {}).items():
            parent = (ph.get("nested_parent") or {}).get(name, "?")
            lines.append(f"| `{name}` | {rec['us'] / 1000:.3f} | {rec['count']} "
                         f"| `{parent}` |")
        lines.append(f"| `record` residual (vkCmd* calls) | "
                     f"{ph.get('record_residual_us', 0) / 1000:.3f} | | `record` |")
        lines.append("")

    tb = report.get("transfer_bytes") or {}
    if tb:
        lines.append("## Transfers")
        lines.append("")
        for direction, rec in tb.items():
            lines.append(f"- {direction}: {rec['bytes'] / 1024 ** 2:.1f} MiB "
                         f"over {rec['samples']} transfer(s)")
        lines.append("")

    rp = report.get("record_paths") or {}
    if rp:
        lines.append(f"## Command-buffer paths: {rp}")
        lines.append("")

    rr = report.get("rerecord") or {}
    if rr:
        lines.append(f"## Command-buffer reuse: **{rr.get('verdict')}**")
        lines.append("")
        lines.append(rr.get("detail", ""))
        lines.append("")
        if not rp:
            lines.append("_(`ep.path` instants are absent from this EP build, so this is "
                         "inferred from per-dispatch recording work rather than read from "
                         "a marker; an absent marker is not evidence of replay.)_")
            lines.append("")

    cmp_ = report.get("op_comparison") or {}
    if cmp_.get("shared_op_types"):
        lines.append("## Per-op-type, restricted to ops both providers ran on GPU")
        lines.append("")
        lines.append("| op | vulkan ms | cuda ms | ratio |")
        lines.append("|---|---:|---:|---:|")
        for row in cmp_["shared_op_types"][:20]:
            ratio = row["ratio_vulkan_over_cuda"]
            lines.append(f"| `{row['op_type']}` | {row['vulkan_us'] / 1000:.3f} "
                         f"| {row['cuda_us'] / 1000:.3f} "
                         f"| {ratio:.2f}x |" if ratio else
                         f"| `{row['op_type']}` | {row['vulkan_us'] / 1000:.3f} "
                         f"| {row['cuda_us'] / 1000:.3f} | – |")
        lines.append("")
    elif cmp_:
        lines.append("## Per-op-type comparison unavailable")
        lines.append("")
        lines.append("The Vulkan EP fuses its claimed island into a single ORT profile "
                     "node, so ORT's profile has no per-op Vulkan rows to intersect with "
                     "CUDA's. Per-kernel Vulkan time comes from the GPU-timestamp table "
                     "above instead.")
        lines.append("")

    for r in report.get("refusals") or []:
        lines.append(f"> REFUSAL: {r}")
    for e in report.get("instrument_errors") or []:
        lines.append(f"> INSTRUMENT ERROR: {e}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workload", default="prefill_1")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260807)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", default=None)
    ap.add_argument("--no-gpu-timestamps", action="store_true")
    ap.add_argument("--reanalyse", action="store_true",
                    help="re-derive the attribution from the trace referenced by an "
                         "existing --out report, without re-running the benchmark")
    ap.add_argument("--skip-cuda", action="store_true",
                    help="skip the CUDA reference arm (attribution of our own time only)")
    a = ap.parse_args(argv)

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(a.scratch) if a.scratch else out_path.parent / "scratch_profile"
    scratch.mkdir(parents=True, exist_ok=True)

    workloads = {w.key: w for w in cuda_workloads.all_workloads()}
    workload = workloads.get(a.workload)
    if workload is None:
        print(f"unknown workload {a.workload!r}; known: {sorted(workloads)}",
              file=sys.stderr)
        return 3

    if a.reanalyse:
        # Re-derive the attribution from evidence already on disk.  The reductions in
        # this module have been wrong twice; a re-run costs minutes and, on a shared
        # box, is not the same measurement.  Re-reducing the stored trace changes only
        # the analysis, which is what is actually under revision.
        prior = json.loads(out_path.read_text("utf-8"))
        traced = prior["traced_record"]
        untraced = prior.get("untraced_record")
        report = attribute(traced, untraced, Path(traced["trace_path"]))
        for k in ("schema", "iters", "warmup", "untraced_record", "traced_record",
                  "cuda_record", "cuda_median_ms", "op_comparison", "gap_ms",
                  "speedup_vulkan_over_cuda"):
            if k in prior:
                report[k] = prior[k]
        report["reanalysed_from"] = str(out_path)
        # The schema names the *reduction*, and the reduction is this module's, not the stored
        # report's.  Copying the old value across would label a `cuda_profile/2` document
        # `cuda_profile/1` and make a reader who checks the schema read the wrong field list.
        report["schema"] = SCHEMA
        report["reanalysed_from_schema"] = prior.get("schema")
        dump_public_json(report, out_path)
        md = render(report)
        write_public_text(md, out_path.with_suffix(".md"))
        print(md)
        return 0

    print(f"[profile] untraced baseline: vulkan/{workload.key}", file=sys.stderr)
    untraced = cc.dispatch_arm(cc.ARM_VULKAN, workload, iters=a.iters, warmup=a.warmup,
                               scratch=scratch / "untraced", device=a.device, seed=a.seed)

    print(f"[profile] traced run: vulkan/{workload.key}", file=sys.stderr)
    traced = run_traced_vulkan(workload, iters=a.iters, warmup=a.warmup,
                               scratch=scratch / "traced", device=a.device, seed=a.seed,
                               gpu_timestamps=not a.no_gpu_timestamps)

    report = attribute(traced, untraced, Path(traced["trace_path"]))
    report["schema"] = SCHEMA
    report["ep_provenance"] = cc.ep_provenance()
    report["iters"] = a.iters
    report["warmup"] = a.warmup
    report["untraced_record"] = untraced
    report["traced_record"] = traced

    if not a.skip_cuda:
        print(f"[profile] cuda reference: cuda/{workload.key}", file=sys.stderr)
        cuda = cc.dispatch_arm(cc.ARM_CUDA, workload, iters=a.iters, warmup=a.warmup,
                               scratch=scratch / "cuda", device=a.device, seed=a.seed)
        report["cuda_record"] = cuda
        report["cuda_median_ms"] = cuda.get("median_ms")
        cuda_prof = cuda.get("profile_path")
        vk_prof = untraced.get("profile_path")
        if cuda_prof and vk_prof:
            report["op_comparison"] = op_type_comparison(
                op_kernel_times(Path(vk_prof)), op_kernel_times(Path(cuda_prof)))
        if report.get("cuda_median_ms") and report.get("untraced_median_ms"):
            report["gap_ms"] = report["untraced_median_ms"] - report["cuda_median_ms"]
            report["speedup_vulkan_over_cuda"] = (
                report["cuda_median_ms"] / report["untraced_median_ms"])

    dump_public_json(report, out_path)
    md = render(report)
    write_public_text(md, out_path.with_suffix(".md"))
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
