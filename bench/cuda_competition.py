"""Vulkan EP vs ORT CUDA EP, on the same device, same bytes (issue #69). Owner: Niobe.

WHAT THIS IS
------------
Issue #69 says: compete with, and beat, ORT's CUDA EP.  That is a comparative
claim, and a comparative claim needs proof that the losing arm was actually in
the race (PR #45's standing ruling).  This harness produces that proof or
refuses to produce a number.

THE ARMS AND WHY THERE ARE FOUR
-------------------------------
The Vulkan EP is an ORT **plugin EP**; the vendored headers declare
``ORT_API_VERSION 28``, so it requires an ORT 1.28 runtime.  On this box the ORT
1.28 CUDA build cannot run at all: it is built against CUDA 13.0, and the
installed driver (573.44, r570 branch) tops out at CUDA 12.8.  The newest
onnxruntime-gpu wheel built against CUDA 12 is **1.26.0** (CUDA 12.8 — exactly
the driver's line).  So the two GPU arms are unavoidably on different ORT
versions.

That is a confound, and disclosing it is not the same as measuring it.  So the
suite runs **four** arms:

===================  ===========  ==================================================
arm                  runtime      what it is for
===================  ===========  ==================================================
``vulkan``           ORT 1.28     the subject
``cuda``             ORT 1.26     the competitor
``cpu_host``         ORT 1.28     bridge: CPU EP in the Vulkan arm's runtime
``cpu_cuda_rt``      ORT 1.26     bridge: CPU EP in the CUDA arm's runtime
===================  ===========  ==================================================

``cpu_host`` / ``cpu_cuda_rt`` are the *same provider* running the *same bytes*
in the two runtimes.  Their ratio is an upper bound on how much of any
Vulkan-vs-CUDA difference could be ORT-version rather than provider.  Without
them the headline ratio is uninterpretable; with them it is bracketed.  This is
the reason the CPU arms exist — they are a control, not a "reference arm" in the
decorative sense.

WHAT MAKES A NUMBER ADMISSIBLE
------------------------------
Every arm must clear all four gates before its timings may be quoted:

1. **Provider held.**  ``session.get_providers()`` must still contain the arm's
   provider after construction.  ORT silently drops a provider whose library
   fails to load and falls through to the next one.
2. **Nodes claimed.**  The ORT profile must show the arm's provider owning at
   least one executed node.  For the Vulkan arm the claim log and the C-ABI
   dispatch counter must both be non-zero.
3. **No unattributed fallback.**  The fraction of executed nodes assigned to
   ``CPUExecutionProvider`` is measured and reported.  An arm above
   ``FALLBACK_SPLIT_THRESHOLD`` is labelled ``SPLIT_FRAME`` — its number
   describes a hybrid execution, not the provider named on the arm — and is
   excluded from the headline ratio while remaining in the record.
4. **Output equivalence.**  Logits must agree with a CPU reference of the same
   model on the same feeds, within a budget stated per dtype.

A refusal at any gate is a *finding* and is recorded as one.  An instrument
failure (harness crashed, profile unreadable) is recorded separately and never
counted as a finding — R13.

PROCESS ISOLATION
-----------------
Each (arm, workload) pair runs in its own subprocess, launched with the
interpreter belonging to that arm's runtime.  Two ORT versions cannot coexist in
one process, and a 2.3 GB model left resident from a previous arm changes the
allocator state the next arm starts from.  The parent never imports onnxruntime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import bench_models  # noqa: E402
from public_paths import dump_public_json, public_path, write_public_text  # noqa: E402
import cuda_workloads  # noqa: E402
from cuda_workloads import Workload  # noqa: E402

# --- arm identity -----------------------------------------------------------

ARM_VULKAN = "vulkan"
ARM_CUDA = "cuda"
ARM_CPU_HOST = "cpu_host"
ARM_CPU_CUDA_RT = "cpu_cuda_rt"

ARMS = (ARM_VULKAN, ARM_CUDA, ARM_CPU_HOST, ARM_CPU_CUDA_RT)

#: Which runtime each arm needs.  ``host`` = the interpreter that has ORT 1.28 and
#: the Vulkan plugin; ``cuda_rt`` = the interpreter that has onnxruntime-gpu on CUDA 12.
ARM_RUNTIME = {
    ARM_VULKAN: "host",
    ARM_CPU_HOST: "host",
    ARM_CUDA: "cuda_rt",
    ARM_CPU_CUDA_RT: "cuda_rt",
}

ARM_PROVIDER = {
    ARM_VULKAN: "VulkanExecutionProvider",
    ARM_CUDA: "CUDAExecutionProvider",
    ARM_CPU_HOST: "CPUExecutionProvider",
    ARM_CPU_CUDA_RT: "CPUExecutionProvider",
}

#: Interpreter for the CUDA-12 runtime.  An env var, not a guess: the venv layout is
#: a property of the box, and a harness that hunts for a python is a harness that can
#: silently pick the wrong ORT.
CUDA_PYTHON_ENV = "ONNXRUNTIME_EP_VULKAN_BENCH_CUDA_PYTHON"
EP_LIB_ENV = "ONNXRUNTIME_VULKAN_EP_LIB"
CLAIM_LOG_ENV = "ONNXRUNTIME_EP_VULKAN_CLAIM_LOG"
COUNTERS_ENV = "ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE"
# Escape hatch for the A/B that established the artifact: set to 1 to keep the counters
# file live across the timed region, reproducing the inflated measurement on purpose.
KEEP_COUNTERS_ENV = "ONNXRUNTIME_EP_VULKAN_BENCH_KEEP_COUNTERS"


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")

# --- verdict vocabulary -----------------------------------------------------

ADMISSIBLE = "ADMISSIBLE"
SPLIT_FRAME = "SPLIT_FRAME"
PROVIDER_ABSENT = "PROVIDER_ABSENT"
NOTHING_CLAIMED = "NOTHING_CLAIMED"
NOT_EQUIVALENT = "NOT_EQUIVALENT"
UNMEASURED = "UNMEASURED"
INSTRUMENT_ERROR = "INSTRUMENT_ERROR"

#: Verdicts that read as "this number may be quoted". A record carrying one of these
#: beside a non-empty ``refusals`` list is disagreeing with itself, which is the shape
#: `test_no_committed_record_calls_itself_admissible_while_refusing` screens for.
GREEN_VERDICTS = frozenset({ADMISSIBLE})

#: Declared vocabulary for :attr:`ArmResult.counters_scope`.
#:
#: The EP rewrites its counters JSON after every ``Compute``. Left on for the whole run
#: that file write lands inside every timed inference and inflates the Vulkan median --
#: 60.519 ms against 31.192 ms for the same arm and workload on the same machine and the
#: same build, a 1.94x inflation. Both figures are committed:
#: ``bench/results/_cuda69/counters_ab_inflated.json`` is the A/B taken with the dump left
#: live on purpose (via ``KEEP_COUNTERS_ENV``), and ``baseline_postgqa.json`` is the clean
#: arm. The
#: two numbers are not interchangeable, so the regime is part of the record rather than
#: something a reader is expected to remember. Empty means "this arm has no Vulkan
#: counters", which is every non-Vulkan arm; it is *not* a permitted value for a Vulkan
#: record, and the committed-artifact screen enforces that.
COUNTERS_SCOPE_FIRST_RUN = "first_run_only"
COUNTERS_SCOPE_ALL_RUNS = "all_runs_INFLATES_TIMING"
COUNTERS_SCOPES = frozenset({COUNTERS_SCOPE_FIRST_RUN, COUNTERS_SCOPE_ALL_RUNS})

#: Above this share of *kernel time* on the CPU EP, the arm is a hybrid.
#:
#: **Why time and not node count.**  The first version of this gate counted profile
#: nodes, and it was wrong for exactly the arm it mattered most for.  The Vulkan EP
#: is a *fusing* EP: it claims a contiguous island of the graph and ORT replaces the
#: whole island with one fused node.  On Phi-3.5 it claimed 355 of 363 nodes and the
#: profile showed **one** ``VulkanExecutionProvider_..._0`` node beside 8 residual
#: host nodes — a node-count reading of 8/9 = 88.9% "CPU fallback" for a graph that
#: is 97.8% on the GPU.  The CUDA EP does not fuse, so the same metric read 0.9% for
#: it.  A metric that reports a 40x different fallback for two arms doing the same
#: thing is measuring fusion, not fallback.
#:
#: Kernel-time share is fusion-invariant and is what the word "fallback" is actually
#: reaching for: how much of the run happened somewhere other than the named
#: provider.  Node-count and graph-level shares are still recorded, because a reader
#: comparing partitions needs all three, but only this one gates admissibility.
#:
#: 0.05 is not tuned: it is "essentially all of the work happened on the named
#: provider", with room for the shape/mask/rope preprocessing ORT keeps on the host
#: in every partitioning of this graph.  A 30%-CPU CUDA arm is not a slow CUDA arm;
#: it is not a CUDA arm.
FALLBACK_SPLIT_THRESHOLD = 0.05

#: Output-equivalence budget, in **ULP measured at the reference tensor's peak
#: magnitude**, derived from the arms' declared compute precision.
#:
#: Three attempts got here, and both dead ends are recorded because each looks
#: correct until it is run.
#:
#: **Attempt 1 — absolute/relative tolerance.** Unusable: on Phi-3.5 the CUDA EP's
#: logits differ from the CPU EP's by 0.078 absolute and the Vulkan EP's by 0.039.
#: Any absolute budget tight enough to be meaningful rejects *the production
#: provider we are competing against*; any budget loose enough to admit it was
#: chosen by looking at it, which is fitting the instrument to the result.
#:
#: **Attempt 2 — raw bit-pattern ULP.** Rejected every GPU arm at ~18,000 ULP
#: against a 32-ULP budget, and the reason is instructive: *every* worst-case
#: element was a near-zero logit that changed sign (``ref=-0.0245``,
#: ``sub=+0.0170`` — 0.04 absolute, on a tensor whose peak is 14.77).  Raw ULP
#: measures distance in units of the *local* spacing, which near zero is
#: vanishingly small, so a physically irrelevant difference reads as an enormous
#: one.  The metric was not wrong about the bits; it was wrong about what a logit
#: tensor's consumer cares about.
#:
#: **What is used — ULP at the tensor's peak.**  The consumer of a logit tensor is
#: softmax/argmax, and of an embedding is a dot product.  Both are sensitive to
#: differences *relative to the tensor's scale*, not relative to each element's own
#: magnitude.  So the unit is ``spacing(max|reference|)`` in the output dtype: the
#: same physical difference gets the same score wherever in the tensor it lands.
#: Under this unit the observed numbers become legible — Vulkan 4.5-36, CUDA 6-113
#: on Phi-3.5 — instead of a uniform five-digit rejection.
#:
#: The budget itself is ``ROUNDING_DEPTH_BOUND x 2^(out_mantissa - compute_mantissa)``
#: and is computed per arm from the precision that arm declares it computes in.
#: Nothing here is fitted: both factors come from format definitions and a stated
#: model-depth bound, and the same formula is applied to every arm.
#:
#: **This makes the TF32 arm's budget loose, on purpose and in the open.**  The CUDA
#: EP defaults to TF32 for fp32 MatMul/Conv, i.e. a 10-bit mantissa where the graph
#: says 23.  Measured on MobileNetV2 that is 18,060 peak-ULP of error against the CPU
#: reference where the Vulkan EP has 23 — a 785x precision difference on identical
#: model bytes, and *not* an ORT-version artifact (``cpu_cuda_rt``, the same ORT
#: build's CPU EP, reads 0).  Granting CUDA a budget that matches TF32 is the only
#: way to compare it at all; pretending the budget is the same for both would either
#: disqualify the competitor or silently license a precision loss the Vulkan EP does
#: not take.  The regime is recorded per arm and printed beside every result.
ROUNDING_DEPTH_BOUND = 128

#: Explicit mantissa bits per compute regime.  ``tf32`` is NVIDIA's 19-bit format
#: with 10 explicit mantissa bits, used by default for fp32 MatMul/Conv on Ampere
#: and later; ``bf16`` is included for completeness.
MANTISSA_BITS = {
    "float16": 10,
    "bfloat16": 7,
    "tf32": 10,
    "float32": 23,
    "float64": 52,
}

#: Retained for the record so a reader can see the physical size of a difference and
#: not only its ULP count.  **Not** a gate — see above.
EQUIVALENCE_BUDGET = {
    "float16": {"atol": 2e-2, "rtol": 2e-2},
    "float32": {"atol": 1e-3, "rtol": 1e-3},
}
#: Task equivalence, and the gate that actually matters.  A model whose top-10
#: ordering is preserved is doing the user's job identically regardless of the last
#: mantissa bit; a model whose argmax moved is not, however small the ULP count.
#:
#: **Conditioned, because an unconditioned version has no power on this input.**
#: The feeds are pseudo-random, so many rows' top-k logits are separated by less
#: than the numerical budget — statistically tied.  Requiring tied entries to keep
#: their order measures the random number generator, not the provider: on
#: ``decode_past128`` both GPU arms scored 0.0 agreement, which says nothing about
#: either.  A row is therefore *resolvable* only when the reference's own gap
#: between adjacent ranks exceeds the budget, and only resolvable rows are gated.
#: When no row is resolvable the check reports ``UNRESOLVED`` and abstains — a task
#: check with no power must not be allowed to read as a pass.
TOP_K = 10


def compute_regime(arm: str, provider_options: "dict | None" = None) -> str:
    """The narrowest precision this arm's provider computes fp32 ops in.

    Read from the live provider options where ORT exposes them, so a run against a
    build with a different default records what that build actually did rather than
    what this file remembers.  Falls back to the documented default only when the
    option is absent.
    """
    if arm != ARM_CUDA:
        return "float32"
    opts = provider_options or {}
    raw = opts.get("use_tf32")
    if raw is None:
        # ORT's CUDA EP has defaulted use_tf32=1 since 1.17.  Recorded as the
        # documented default rather than assumed silently.
        return "tf32"
    return "tf32" if str(raw) not in ("0", "False", "false") else "float32"


def equivalence_budget_ulp(out_dtype: str, regime: str) -> "int | None":
    """Peak-ULP budget for one output dtype under one compute regime."""
    out_bits = MANTISSA_BITS.get(out_dtype)
    comp_bits = MANTISSA_BITS.get(regime)
    if out_bits is None or comp_bits is None:
        return None
    # A chain of roundings at relative 2^-comp_bits, expressed in units of the
    # output format's own ULP.  Never below 1: the output rounding alone costs that.
    return int(ROUNDING_DEPTH_BOUND * max(1, 2 ** (out_bits - comp_bits)))


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class ArmResult:
    arm: str
    workload: str
    model_key: str
    verdict: str = UNMEASURED
    provider_requested: str = ""
    providers_held: "list[str]" = field(default_factory=list)
    #: node -> provider partition read from the ORT profile
    partition: dict = field(default_factory=dict)
    kernel_time_us: dict = field(default_factory=dict)
    executed_nodes: int = 0
    cpu_fallback_nodes: int = 0
    #: three fallback shares, deliberately not collapsed into one number
    cpu_fallback_fraction_nodes: "float | None" = None
    cpu_fallback_fraction_time: "float | None" = None
    cpu_fallback_fraction_graph: "float | None" = None
    #: the one that gates admissibility (== ``cpu_fallback_fraction_time``)
    cpu_fallback_fraction: "float | None" = None
    fallback_node_names: "list[str]" = field(default_factory=list)
    #: Vulkan-only
    vulkan_claimed_nodes: "int | None" = None
    vulkan_probed_nodes: "int | None" = None
    vulkan_dispatches: "int | None" = None
    vulkan_fused_nodes: "int | None" = None
    #: timings
    warmup_ms: "list[float]" = field(default_factory=list)
    steady_ms: "list[float]" = field(default_factory=list)
    session_create_ms: "float | None" = None
    first_run_ms: "float | None" = None
    #: derived
    median_ms: "float | None" = None
    p05_ms: "float | None" = None
    p95_ms: "float | None" = None
    rsd_pct: "float | None" = None
    tokens_per_s: "float | None" = None
    #: provenance
    ort_version: str = ""
    ort_build_cuda: "str | None" = None
    python: str = ""
    feed_digest: str = ""
    model_bundle_sha256: str = ""
    device_identity: dict = field(default_factory=dict)
    peak_bytes: "int | None" = None
    #: where the ORT profile for this arm landed, so a downstream attribution pass
    #: (``bench/cuda_profile.py``) can read per-op kernel times without re-running.
    profile_path: str = ""
    #: The precision this arm's provider actually computes fp32 ops in, read from the
    #: live provider options.  The CUDA EP defaults to TF32 (10 mantissa bits) for
    #: fp32 MatMul/Conv, which is a real and material difference from the graph's
    #: declared fp32 — it is both why that arm needs a wider equivalence budget and
    #: part of why it is fast.  Recorded, never inferred at read time.
    compute_regime: str = ""
    provider_options: dict = field(default_factory=dict)
    #: Whether the EP's counters-file dump was live during the timed region.  The dump
    #: rewrites a JSON document after every Compute, so "all_runs" means the reported
    #: median includes this harness's own instrumentation and is not a clean measurement.
    counters_scope: str = ""
    #: equivalence
    equivalence: dict = field(default_factory=dict)
    #: findings vs instrument failures — never merged
    refusals: "list[str]" = field(default_factory=list)
    instrument_errors: "list[str]" = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _quantile(xs: "list[float]", q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def summarise(samples: "list[float]") -> dict:
    if not samples:
        return {"median_ms": None, "p05_ms": None, "p95_ms": None, "rsd_pct": None, "n": 0}
    med = statistics.median(samples)
    sd = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return {
        "median_ms": med,
        "p05_ms": _quantile(samples, 0.05),
        "p95_ms": _quantile(samples, 0.95),
        "rsd_pct": (sd / med * 100.0) if med else None,
        "n": len(samples),
    }


def bootstrap_ratio_ci(a: "list[float]", b: "list[float]", *,
                       draws: int = 4000, seed: int = 20260807,
                       level: float = 0.95) -> dict:
    """Confidence interval on ``median(b) / median(a)`` by percentile bootstrap.

    Unpaired, because the two arms live in different processes with different
    runtimes and cannot be interleaved sample-for-sample.  §24 of docs/PERF.md
    already established that *pairing* on this box introduces a device-axis
    asymmetry larger than the effects being chased, so the unpaired form is not a
    concession — it is the form whose failure mode is understood.
    """
    import random

    if len(a) < 3 or len(b) < 3:
        return {"ratio": None, "lo": None, "hi": None, "draws": 0,
                "note": "fewer than 3 samples in an arm; no interval is quotable"}
    rng = random.Random(seed)
    point = statistics.median(b) / statistics.median(a)
    ratios: "list[float]" = []
    for _ in range(draws):
        ra = statistics.median([a[rng.randrange(len(a))] for _ in a])
        rb = statistics.median([b[rng.randrange(len(b))] for _ in b])
        if ra > 0:
            ratios.append(rb / ra)
    ratios.sort()
    if not ratios:
        return {"ratio": point, "lo": None, "hi": None, "draws": 0}
    tail = (1.0 - level) / 2.0
    return {
        "ratio": point,
        "lo": ratios[int(tail * (len(ratios) - 1))],
        "hi": ratios[int((1 - tail) * (len(ratios) - 1))],
        "draws": len(ratios),
        "level": level,
    }


# ---------------------------------------------------------------------------
# Provider attribution (the mandatory part)
# ---------------------------------------------------------------------------

def read_profile_partition(profile_path: Path) -> dict:
    """Count executed graph nodes per provider from an ORT profile.

    Only ``cat == "Node"`` events whose name ends in ``_kernel_time`` are counted:
    those are the per-node kernel executions, one per node per run.  ``fence`` and
    ``session``-category events are bookkeeping and would inflate the count.
    """
    with profile_path.open("r", encoding="utf-8") as fh:
        events = json.load(fh)

    partition: "dict[str, int]" = {}
    per_provider_us: "dict[str, float]" = {}
    node_provider: "dict[str, str]" = {}
    op_provider: "dict[str, dict[str, int]]" = {}

    for ev in events:
        if not isinstance(ev, dict) or ev.get("cat") != "Node":
            continue
        name = ev.get("name", "")
        if not name.endswith("_kernel_time"):
            continue
        args = ev.get("args") or {}
        provider = args.get("provider") or "UNATTRIBUTED"
        node = name[: -len("_kernel_time")]
        # One entry per *node*, not per execution: with N runs profiled, every node
        # appears N times, and counting executions would report N x the graph size.
        node_provider[node] = provider
        per_provider_us[provider] = per_provider_us.get(provider, 0.0) + float(ev.get("dur", 0) or 0)
        op = args.get("op_name") or "?"
        op_provider.setdefault(provider, {})
        op_provider[provider][op] = op_provider[provider].get(op, 0) + 1

    for provider in node_provider.values():
        partition[provider] = partition.get(provider, 0) + 1

    return {
        "partition": partition,
        "kernel_time_us": per_provider_us,
        "op_counts_by_provider": op_provider,
        "node_provider": node_provider,
        "executed_nodes": len(node_provider),
    }


def classify_fallback(partition: dict, kernel_time_us: dict,
                      provider_requested: str) -> dict:
    """Three fallback shares for one arm's profile.  None of them is optional.

    ``nodes``
        CPU-EP profile nodes / all profile nodes.  Fusion-blind — see
        :data:`FALLBACK_SPLIT_THRESHOLD`.  Kept because it is what a reader
        eyeballing an ORT profile will compute, and the record should show the same
        number they will, next to the reason it is not the gate.
    ``time``
        CPU-EP kernel microseconds / all kernel microseconds.  Fusion-invariant.
        **This is the gate.**
    ``graph``
        Set by the caller for a fusing EP from its own claim log: unclaimed original
        graph nodes / probed original graph nodes.  ``None`` for non-fusing EPs,
        because for them it is the same quantity as ``nodes`` and a duplicated
        number invites a reader to treat two views as two witnesses.

    A ``None`` share is never a zero.  0/0 is "no measurement", and the distinction
    is the whole point: an arm that executed nothing has an unmeasured fallback, not
    an absent one.
    """
    executed = sum(partition.values())
    total_us = sum(kernel_time_us.values())
    cpu_nodes = partition.get("CPUExecutionProvider", 0)
    cpu_us = kernel_time_us.get("CPUExecutionProvider", 0.0)

    if provider_requested == "CPUExecutionProvider":
        # The CPU arm cannot fall back to itself.  Reporting 0.0 rather than None is
        # correct here and is not the 0/0 case: the provider ran, and the share of
        # its work that happened somewhere else is genuinely zero.
        return {"executed_nodes": executed, "cpu_fallback_nodes": 0,
                "nodes": 0.0 if executed else None,
                "time": 0.0 if total_us else None,
                "graph": None}

    return {
        "executed_nodes": executed,
        "cpu_fallback_nodes": cpu_nodes,
        "nodes": (cpu_nodes / executed) if executed else None,
        "time": (cpu_us / total_us) if total_us else None,
        "graph": None,
    }


# ---------------------------------------------------------------------------
# Equivalence
# ---------------------------------------------------------------------------

def peak_ulp_spacing(reference) -> "float | None":
    """The size of one ULP at the reference tensor's peak finite magnitude.

    This is the unit the equivalence budget is denominated in.  Using the *peak*
    rather than each element's own magnitude is the whole point: a logit tensor is
    consumed by softmax/argmax and an embedding by a dot product, and both are
    sensitive to differences relative to the tensor's scale.  Per-element spacing
    makes a 0.04 difference on a 0.02-magnitude logit look like 19,000 ULP while the
    same 0.04 on the tensor's 14.77 peak looks like 5 — same physical error, two
    answers differing by four orders of magnitude, and only one of them predicts
    whether the model's output changed.

    Computed by hand from ``frexp`` rather than via ``np.spacing``: Trinity found
    (2026-08-04) that ``np.spacing`` returns ``inf`` at fp16's largest finite value,
    so a tensor that peaks there would get an infinite budget — every error would
    pass.  A metric that cannot fail at one end of its range is not a metric.
    """
    import numpy as np

    ref = np.asarray(reference)
    bits = MANTISSA_BITS.get(str(ref.dtype))
    if bits is None:
        return None
    finite = ref[np.isfinite(ref)]
    if finite.size == 0:
        return None
    peak = float(np.abs(finite.astype(np.float64)).max())
    if peak == 0.0:
        # An all-zero reference has no scale of its own; fall back to the dtype's
        # smallest normal so "equal to zero" stays the only passing answer.
        return float(np.finfo(ref.dtype).tiny)
    _, exp = np.frexp(peak)
    return float(2.0 ** (exp - 1 - bits))


def ulp_distance(subject, reference):
    """Element-wise raw ULP distance between two arrays of the same float dtype.

    Computed by reinterpreting the bit patterns as sign-magnitude integers and
    mapping them to a monotone two's-complement ordering, so the distance across
    zero is correct and does not need a special case.

    Deliberately **not** ``np.spacing``: Trinity found (2026-08-04) that
    ``np.spacing`` returns ``inf`` at fp16's largest finite value, which made a
    504-unit error read back as 0.0 ULP — an instrument that makes a wrong residual
    look sound.  The bit-pattern form has no such boundary.

    **This is reported but no longer gates.**  Raw ULP is the right metric for values
    of comparable magnitude and the wrong one near zero, where the spacing collapses
    and a physically irrelevant difference scores in the tens of thousands.  It is
    kept because the *ratio* between raw and peak-scaled ULP localises where an arm's
    error lives: a large raw and small scaled figure means the error is in the noise
    floor, and the reverse means it is on the values that matter.
    """
    import numpy as np

    ref = np.asarray(reference)
    sub = np.asarray(subject)
    if ref.dtype == np.float16:
        int_dtype = np.int16
        offset = np.int16(-32768)
    elif ref.dtype == np.float32:
        int_dtype = np.int32
        offset = np.int32(-2147483648)
    elif ref.dtype == np.float64:
        int_dtype = np.int64
        offset = np.int64(-9223372036854775808)
    else:
        return None

    def monotone(a):
        bits = a.astype(ref.dtype).view(int_dtype).astype(np.int64)
        neg = bits < 0
        # Negative floats descend as their bit pattern ascends; reflect them so the
        # whole range is one monotone integer line through zero.
        return np.where(neg, np.int64(offset) - bits, bits)

    return np.abs(monotone(sub) - monotone(ref))


def conditioned_top_k(subject_f64, reference_f64, *, k: int, resolution: float) -> dict:
    """Task agreement, gated only on rows whose ranking the observed noise can resolve.

    ``resolution`` is the largest reference gap that the two tensors' disagreement
    could have inverted: twice a robust estimate of their element-wise noise.  If
    they differ by at most ``d``, a reference gap greater than ``2d`` provably
    survives — rank *i* can fall by at most ``d`` and rank *i+1* can rise by at most
    ``d``.  The caller supplies ``d`` as a high quantile rather than the maximum,
    because the maximum lets a sparse plant manufacture its own excuse; see
    :func:`compare_outputs` for that derivation.

    **Why conditioning is necessary here, and what it costs.**  The feeds are
    pseudo-random token ids over a random KV cache, so the model has no coherent
    context and emits a near-flat distribution: on ``phi35_prefill_1`` the reference's
    top-1 to top-2 gap is **0.031**, smaller than either GPU arm's own worst-case
    error (0.039 Vulkan, 0.078 CUDA).  Which of those two tokens comes first is
    genuinely undecidable at this precision.  The unconditioned check scored 0.0
    agreement for *both* GPU arms on ``decode_past128``, which is a fact about the
    random number generator and not about either provider.

    The cost is real and is reported rather than hidden: where no row resolves, this
    check has **no power**, returns ``UNRESOLVED``, and must not be read as a pass.
    ``resolvable_rows`` is the coverage figure; a reader who sees 0 there knows the
    task claim rests entirely on the numeric budget.

    Three views, in decreasing order of how well conditioned they are:

    * **argmax** — gated on rows where the top-1/top-2 gap survives.  This is the
      decision a greedy decoder actually makes.
    * **top-k set membership** — gated on rows where the rank-k/rank-k+1 gap
      survives.  Which tokens are in the candidate set, ignoring their order.
    * **top-k ordering** — reported, never gated.  It is the most fragile view and
      it is not the task: a sampler does not care whether ranks 7 and 8 swapped.
    """
    import numpy as np

    s2 = subject_f64.reshape(-1, subject_f64.shape[-1])
    r2 = reference_f64.reshape(-1, reference_f64.shape[-1])
    k = int(min(k, r2.shape[-1]))
    if k < 1 or r2.shape[0] == 0:
        return {"verdict": "UNRESOLVED", "detail": "no rows or no vocabulary axis"}

    s_top = np.argsort(-s2, axis=-1, kind="stable")[:, :k]
    r_order = np.argsort(-r2, axis=-1, kind="stable")
    r_top = r_order[:, :k]

    kk = min(k + 1, r2.shape[-1])
    top_vals = np.take_along_axis(r2, r_order[:, :kk], axis=-1)
    gaps = top_vals[:, :-1] - top_vals[:, 1:]

    n_rows = int(r2.shape[0])
    zeros = np.zeros(n_rows, dtype=bool)
    argmax_resolvable = gaps[:, 0] > resolution if gaps.shape[-1] >= 1 else zeros
    # The top-k *set* is determined when the rank-k/rank-k+1 boundary survives; the
    # ordering inside the set is irrelevant to membership.
    set_resolvable = gaps[:, -1] > resolution if gaps.shape[-1] >= 1 else zeros
    order_resolvable = (gaps > resolution).all(axis=-1) if gaps.size else zeros

    exact = (s_top == r_top).all(axis=-1)
    setwise = np.array([len(set(a.tolist()) & set(b.tolist())) == k
                        for a, b in zip(s_top, r_top)]) if n_rows else zeros
    argmax_ok = s_top[:, 0] == r_top[:, 0]

    out: dict = {
        "top_k": k,
        "rows": n_rows,
        "resolution_abs": float(resolution),
        "argmax_resolvable_rows": int(argmax_resolvable.sum()),
        "set_resolvable_rows": int(set_resolvable.sum()),
        "order_resolvable_rows": int(order_resolvable.sum()),
        "unconditioned_argmax_agreement": float(argmax_ok.mean()),
        "unconditioned_set_agreement": float(setwise.mean()),
        "unconditioned_order_agreement": float(exact.mean()),
        "median_argmax_gap": float(np.median(gaps[:, 0])) if gaps.shape[-1] else None,
    }
    if argmax_resolvable.any():
        out["argmax_agreement"] = float(argmax_ok[argmax_resolvable].mean())
    if set_resolvable.any():
        out["set_agreement"] = float(setwise[set_resolvable].mean())
    if order_resolvable.any():
        out["order_agreement"] = float(exact[order_resolvable].mean())

    if not argmax_resolvable.any() and not set_resolvable.any():
        out["verdict"] = "UNRESOLVED"
        out["detail"] = (
            f"no row's reference ranking is separated by more than {resolution:.4g}, the "
            f"largest gap the two tensors' measured disagreement could have inverted. The "
            f"median top-1/top-2 gap is {out['median_argmax_gap']}. No ordering claim is "
            f"testable on this input, so this check abstains rather than passing — it is a "
            f"property of pseudo-random feeds through a model with no coherent context, not "
            f"of the provider.")
        return out

    ok = True
    if argmax_resolvable.any():
        ok = ok and out["argmax_agreement"] == 1.0
    if set_resolvable.any():
        ok = ok and out["set_agreement"] == 1.0
    out["verdict"] = "MATCH" if ok else "DIVERGENT"
    return out


def compare_outputs(subject, reference, output_meta, *, top_k: int = TOP_K,
                    regime: str = "float32") -> dict:
    """Compare an arm's outputs against a CPU reference under a derived ULP budget.

    ``regime`` is the arm's declared compute precision (see :func:`compute_regime`);
    it widens the budget for a provider that has told us it computes in a narrower
    format than the graph declares.  It is recorded in the result so no reader can
    mistake a wide budget for a tight one.

    Reports the evidence unconditionally, not only on failure: a passing check whose
    peak-ULP distance is 0 for every output is a different event from one at 90% of
    budget, and only the numbers distinguish them.
    """
    import numpy as np

    rec: dict = {"verdict": "MATCH", "outputs": [], "compute_regime": regime,
                 "rounding_depth_bound": ROUNDING_DEPTH_BOUND,
                 "physical_budget_reported_not_enforced": EQUIVALENCE_BUDGET}
    if subject is None or reference is None:
        return {"verdict": UNMEASURED, "outputs": [],
                "detail": "one side of the comparison was not produced"}
    if len(subject) != len(reference):
        return {"verdict": "DIVERGENT", "outputs": [],
                "detail": f"output arity differs: {len(subject)} vs {len(reference)}"}

    worst = "MATCH"
    for idx, (s, r) in enumerate(zip(subject, reference)):
        name = output_meta[idx] if idx < len(output_meta) else f"output_{idx}"
        s = np.asarray(s)
        r = np.asarray(r)
        entry: dict = {"name": name, "dtype": str(r.dtype), "shape": list(r.shape)}
        if s.shape != r.shape:
            entry["verdict"] = "DIVERGENT"
            entry["detail"] = f"shape {list(s.shape)} != {list(r.shape)}"
            worst = "DIVERGENT"
            rec["outputs"].append(entry)
            continue

        sf = s.astype(np.float64)
        rf = r.astype(np.float64)
        finite = np.isfinite(sf) & np.isfinite(rf)
        entry["nonfinite_subject"] = int((~np.isfinite(sf)).sum())
        entry["nonfinite_reference"] = int((~np.isfinite(rf)).sum())
        if not finite.any():
            entry["verdict"] = "DIVERGENT"
            entry["detail"] = "no finite element pairs to compare"
            worst = "DIVERGENT"
            rec["outputs"].append(entry)
            continue
        # A non-finite element on either side that its counterpart does not share is a
        # divergence in itself, not something to be excluded from the comparison.
        if entry["nonfinite_subject"] != entry["nonfinite_reference"]:
            entry["verdict"] = "DIVERGENT"
            entry["detail"] = ("non-finite element counts differ; one side produced inf/NaN "
                               "where the other did not")
            worst = "DIVERGENT"
            rec["outputs"].append(entry)
            continue

        diff = np.abs(sf[finite] - rf[finite])
        denom = np.maximum(np.abs(rf[finite]), 1e-6)
        entry["max_abs_diff"] = float(diff.max())
        entry["max_rel_diff"] = float((diff / denom).max())
        entry["mean_abs_diff"] = float(diff.mean())
        entry["reference_abs_max"] = float(np.abs(rf[finite]).max())

        # An arm that returned a constant where the reference varies is the failure
        # mode this project has actually shipped, and it can hide under any tolerance
        # that is stated in units of the reference.  Check it directly.
        if float(np.abs(sf[finite]).max()) == 0.0 and entry["reference_abs_max"] > 0.0:
            entry["verdict"] = "DIVERGENT"
            entry["detail"] = "subject is identically zero where the reference is not"
            worst = "DIVERGENT"
            rec["outputs"].append(entry)
            continue

        spacing = peak_ulp_spacing(r)
        budget = equivalence_budget_ulp(str(r.dtype), regime)
        raw = ulp_distance(s, r)
        if raw is not None:
            entry["max_raw_ulp"] = int(raw[finite].max())

        if spacing is None or budget is None:
            entry["peak_ulp"] = None
            ok = bool(np.array_equal(s, r))
            entry["detail"] = ("dtype has no ULP model; falling back to exact equality, which "
                               "is the only defensible budget for an integral or unknown type")
            resolution = 0.0
        else:
            scaled = diff / spacing
            entry["peak_ulp_spacing"] = spacing
            entry["max_peak_ulp"] = float(scaled.max())
            entry["mean_peak_ulp"] = float(scaled.mean())
            entry["peak_ulp_budget"] = budget
            entry["over_budget_elements"] = int((scaled > budget).sum())
            ok = bool(entry["max_peak_ulp"] <= budget)
            # --- resolution: how large a reference gap the noise could invert ----
            #
            # Two arrays differing by at most `d` cannot inverta reference gap wider
            # than `2d`.  The tempting choice for `d` is the tensor-wide maximum,
            # and it is wrong in a way that took a planted control to expose: an
            # arm that swaps two adjacent ranks produces, *as its only error*,
            # exactly the disagreement that then declares the swap unresolvable.
            # The control excuses itself.
            #
            # The quantity actually wanted is the *typical* rounding noise between
            # the two tensors, which a sparse plant does not move and distributed
            # rounding does.  A high quantile gives that: at p99 a plant touching
            # under 1% of elements leaves the estimate at the noise floor, so every
            # row stays resolvable and the plant is caught, while genuine
            # element-wise rounding raises it and tied rows correctly abstain.
            #
            # A plant large enough to move p99 is by construction large enough for
            # the numeric gate above to see, so the two gates cover each other's
            # blind spot.  That is the reason both exist and neither is redundant.
            noise = float(np.quantile(diff, 0.99)) if diff.size else 0.0
            entry["disagreement_p99"] = noise
            resolution = min(2.0 * noise, budget * spacing)

        if "logits" in name and rf.ndim >= 1:
            tk = conditioned_top_k(sf, rf, k=top_k, resolution=resolution)
            entry["task"] = tk
            # Task equivalence gates in both directions where it has power, and
            # abstains where it does not.  It can fail an arm whose ULP is fine, and
            # it cannot rescue one whose ULP is not.
            if tk.get("verdict") == "DIVERGENT":
                ok = False

        entry["verdict"] = "MATCH" if ok else "DIVERGENT"
        if not ok:
            worst = "DIVERGENT"
        rec["outputs"].append(entry)

    rec["verdict"] = worst
    return rec


# ---------------------------------------------------------------------------
# Worker — one arm, one workload, one process
# ---------------------------------------------------------------------------

def _session_options(ort, profile_prefix: Path):
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.enable_profiling = True
    opts.profile_file_prefix = str(profile_prefix)
    # Identical across arms on purpose: a thread-count difference between arms is a
    # difference in the experiment, not in the provider.
    opts.intra_op_num_threads = 0
    opts.inter_op_num_threads = 0
    return opts


def _register_vulkan(ort, rec: ArmResult) -> bool:
    lib = os.environ.get(EP_LIB_ENV)
    if not lib or not Path(lib).is_file():
        rec.refusals.append(
            f"{EP_LIB_ENV} is unset or does not point at a file; there is no Vulkan EP to "
            f"measure, so no Vulkan number exists for this workload")
        rec.verdict = PROVIDER_ABSENT
        return False
    try:
        ort.register_execution_provider_library(ARM_PROVIDER[ARM_VULKAN], str(Path(lib).resolve()))
    except Exception as exc:
        if "already registered" not in str(exc):
            rec.refusals.append(f"Vulkan EP registration failed: {exc}")
            rec.verdict = PROVIDER_ABSENT
            return False
    return True


def _read_claim_log(path: Path) -> "tuple[int | None, int | None]":
    if not path.exists():
        return None, None
    claimed = 0
    total = 0
    for line in path.read_text("utf-8", "replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        total += 1
        if entry.get("claimed"):
            claimed += 1
    return (claimed, total) if total else (None, None)


def _read_counters(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text("utf-8", "replace"))
    except json.JSONDecodeError:
        return {}


def run_arm(arm: str, workload: Workload, model: bench_models.ResolvedModel,
            *, iters: int, warmup: int, scratch: Path,
            device_index: int = 0, seed: int = 20260807) -> ArmResult:
    """Execute one arm on one workload.  Correctness before timing, always."""
    import numpy as np  # noqa: F401
    import onnxruntime as ort

    rec = ArmResult(arm=arm, workload=workload.key, model_key=workload.model_key,
                    provider_requested=ARM_PROVIDER[arm])
    rec.ort_version = ort.__version__
    rec.python = sys.version.split()[0]
    rec.model_bundle_sha256 = model.bundle_sha256 or ""
    try:
        import re as _re
        m = _re.search(r"CUDA_VERSION\s*=\s*([0-9.]+)", ort.get_build_info())
        rec.ort_build_cuda = m.group(1) if m else None
    except Exception:
        rec.ort_build_cuda = None

    if model.status != bench_models.MODEL_OK:
        rec.refusals.append(f"model {model.key} is {model.status}: {model.detail[:400]}")
        rec.verdict = UNMEASURED
        return rec

    scratch.mkdir(parents=True, exist_ok=True)
    tag = f"{arm}_{workload.key}_{os.getpid()}"

    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception as exc:
            rec.instrument_errors.append(f"preload_dlls raised: {exc!r}")

    claim_log = scratch / f"claim_{tag}.jsonl"
    counters_file = scratch / f"counters_{tag}.json"
    if arm == ARM_VULKAN:
        if not _register_vulkan(ort, rec):
            return rec
        claim_log.unlink(missing_ok=True)
        counters_file.unlink(missing_ok=True)
        os.environ[CLAIM_LOG_ENV] = str(claim_log)
        os.environ[COUNTERS_ENV] = str(counters_file)
    opts = _session_options(ort, scratch / f"profile_{tag}")
    if arm == ARM_VULKAN:
        opts.add_session_config_entry("ep.device_index", str(device_index))

    providers: "list" = []
    if arm == ARM_VULKAN:
        providers = [ARM_PROVIDER[ARM_VULKAN], "CPUExecutionProvider"]
    elif arm == ARM_CUDA:
        providers = [("CUDAExecutionProvider", {"device_id": device_index}),
                     "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    t0 = time.perf_counter()
    try:
        sess = ort.InferenceSession(model.path, opts, providers=providers)
    except Exception as exc:
        rec.refusals.append(f"session creation failed for arm {arm}: {exc!r}"[:1500])
        rec.verdict = PROVIDER_ABSENT
        return rec
    finally:
        os.environ.pop(CLAIM_LOG_ENV, None)
    rec.session_create_ms = (time.perf_counter() - t0) * 1000.0

    rec.providers_held = list(sess.get_providers())
    try:
        rec.provider_options = {k: dict(v) for k, v in
                                (sess.get_provider_options() or {}).items()}
    except Exception as exc:
        rec.instrument_errors.append(f"get_provider_options unavailable: {exc!r}")
    rec.compute_regime = compute_regime(
        arm, rec.provider_options.get(ARM_PROVIDER[arm]))
    if ARM_PROVIDER[arm] not in rec.providers_held:
        rec.refusals.append(
            f"ORT dropped {ARM_PROVIDER[arm]} during session creation and fell through to "
            f"{rec.providers_held}; this session is not a {arm} session and its timings would "
            f"be another provider's timings wearing this arm's label")
        rec.verdict = PROVIDER_ABSENT
        sess.end_profiling()
        return rec

    if arm == ARM_VULKAN:
        claimed, total = _read_claim_log(claim_log)
        rec.vulkan_claimed_nodes = claimed
        rec.vulkan_probed_nodes = total
        if not claimed:
            rec.refusals.append(
                "the Vulkan EP was held by the session but claimed zero nodes; every node ran "
                "elsewhere, so nothing here is a Vulkan measurement")
            rec.verdict = NOTHING_CLAIMED
            sess.end_profiling()
            return rec

    try:
        feeds = cuda_workloads.build_feeds(sess, workload, seed=seed)
    except Exception as exc:
        rec.instrument_errors.append(f"feed construction failed: {exc!r}")
        rec.verdict = INSTRUMENT_ERROR
        sess.end_profiling()
        return rec
    rec.feed_digest = feeds.digest

    # --- first run: compile/warmup boundary, measured separately -------------
    t0 = time.perf_counter()
    try:
        first_out = sess.run(None, feeds.arrays)
    except Exception as exc:
        rec.refusals.append(f"first inference failed: {exc!r}"[:1500])
        rec.verdict = UNMEASURED
        sess.end_profiling()
        return rec
    rec.first_run_ms = (time.perf_counter() - t0) * 1000.0

    # The counters file is dispatch evidence, and collecting it must not become part of
    # what is being measured.  `counters::record_dispatches` calls `dump_if_requested`,
    # which READS AND REWRITES the whole counters JSON after every Compute.  Left set for
    # the timed region that rewrite lands inside every Vulkan inference — an artifact the
    # CUDA arm has no equivalent of, so it would be charged to the Vulkan EP and inflate
    # exactly the number this harness exists to report.  The first run above has already
    # written the file, so the evidence is banked; the env var comes off before anything
    # that is timed.  `_read_counters` still reads it below.
    if arm == ARM_VULKAN and not _env_flag(KEEP_COUNTERS_ENV):
        os.environ.pop(COUNTERS_ENV, None)
        rec.counters_scope = COUNTERS_SCOPE_FIRST_RUN
    elif arm == ARM_VULKAN:
        rec.counters_scope = COUNTERS_SCOPE_ALL_RUNS

    # --- warmup, kept and reported, never silently discarded -----------------
    for _ in range(warmup):
        t0 = time.perf_counter()
        sess.run(None, feeds.arrays)
        rec.warmup_ms.append((time.perf_counter() - t0) * 1000.0)

    # --- steady state --------------------------------------------------------
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(None, feeds.arrays)
        rec.steady_ms.append((time.perf_counter() - t0) * 1000.0)

    summary = summarise(rec.steady_ms)
    rec.median_ms = summary["median_ms"]
    rec.p05_ms = summary["p05_ms"]
    rec.p95_ms = summary["p95_ms"]
    rec.rsd_pct = summary["rsd_pct"]
    if rec.median_ms:
        # Tokens produced per second at this shape.  For prefill that is seq_len
        # tokens ingested; for decode it is the one token emitted.
        rec.tokens_per_s = (workload.seq_len * 1000.0) / rec.median_ms

    profile = Path(sess.end_profiling())
    rec.profile_path = str(profile)
    try:
        attrib = read_profile_partition(profile)
    except Exception as exc:
        rec.instrument_errors.append(f"profile at {profile} unreadable: {exc!r}")
        attrib = None

    if attrib is not None:
        rec.partition = attrib["partition"]
        rec.kernel_time_us = attrib["kernel_time_us"]
        shares = classify_fallback(attrib["partition"], attrib["kernel_time_us"],
                                   ARM_PROVIDER[arm])
        rec.executed_nodes = shares["executed_nodes"]
        rec.cpu_fallback_nodes = shares["cpu_fallback_nodes"]
        rec.cpu_fallback_fraction_nodes = shares["nodes"]
        rec.cpu_fallback_fraction_time = shares["time"]
        if arm == ARM_VULKAN and rec.vulkan_probed_nodes:
            unclaimed = rec.vulkan_probed_nodes - (rec.vulkan_claimed_nodes or 0)
            rec.cpu_fallback_fraction_graph = unclaimed / rec.vulkan_probed_nodes
            rec.vulkan_fused_nodes = rec.partition.get(ARM_PROVIDER[ARM_VULKAN], 0)
        rec.cpu_fallback_fraction = rec.cpu_fallback_fraction_time
        rec.fallback_node_names = sorted(
            n for n, p in attrib["node_provider"].items() if p == "CPUExecutionProvider"
        )[:200]
        rec.partition_detail = attrib["op_counts_by_provider"]  # type: ignore[attr-defined]

    if arm == ARM_VULKAN:
        counters = _read_counters(counters_file)
        for key in ("dispatches_executed", "dispatches", "dispatch_count"):
            if key in counters:
                rec.vulkan_dispatches = int(counters[key])
                break
        if rec.vulkan_dispatches is not None and rec.vulkan_dispatches == 0:
            rec.refusals.append(
                "the Vulkan EP claimed nodes but executed zero dispatches; the GPU did no work")
            rec.verdict = NOTHING_CLAIMED
            return rec

    # --- gate 2: did the arm's own provider execute anything? ----------------
    own = rec.partition.get(ARM_PROVIDER[arm], 0)
    if rec.executed_nodes and own == 0 and arm != ARM_VULKAN:
        rec.refusals.append(
            f"{ARM_PROVIDER[arm]} was held by the session but owns zero executed nodes in the "
            f"profile; the partition is {rec.partition}")
        rec.verdict = NOTHING_CLAIMED
        return rec

    # --- gate 3: split frame — on kernel-time share, not node count ----------
    if rec.cpu_fallback_fraction is not None and \
            rec.cpu_fallback_fraction > FALLBACK_SPLIT_THRESHOLD and \
            ARM_PROVIDER[arm] != "CPUExecutionProvider":
        rec.verdict = SPLIT_FRAME
        rec.refusals.append(
            f"{rec.cpu_fallback_fraction:.1%} of profiled kernel time ran on the CPU EP, above "
            f"the {FALLBACK_SPLIT_THRESHOLD:.0%} split threshold; this is a hybrid execution "
            f"and its timing is not a {ARM_PROVIDER[arm]} timing. Node-count share was "
            f"{rec.cpu_fallback_fraction_nodes}, graph share "
            f"{rec.cpu_fallback_fraction_graph}, partition {rec.partition}")
    elif rec.cpu_fallback_fraction is None and rec.executed_nodes == 0:
        rec.verdict = INSTRUMENT_ERROR
        rec.instrument_errors.append(
            "the profile contained no kernel events, so the partition is unmeasured; this is "
            "the instrument failing to look, not a finding about the provider (R13)")
    else:
        rec.verdict = ADMISSIBLE

    rec.first_outputs = first_out  # type: ignore[attr-defined]
    rec.output_names = [o.name for o in sess.get_outputs()]  # type: ignore[attr-defined]
    return rec


# ---------------------------------------------------------------------------
# Worker entry point
# ---------------------------------------------------------------------------

def _npy_dump(outputs, names, path: Path) -> "list[dict]":
    """Persist an arm's outputs so the parent can compare across processes.

    Full tensors, not summaries: an equivalence check computed from summaries is a
    check of the summariser.  Phi-3.5's 65 outputs at past 2048 are large, so only
    the first output (``logits`` in every model here) is written in full and the KV
    outputs are recorded by shape and digest.
    """
    import numpy as np

    path.mkdir(parents=True, exist_ok=True)
    manifest: "list[dict]" = []
    for idx, (arr, name) in enumerate(zip(outputs, names)):
        arr = np.asarray(arr)
        entry = {"name": name, "dtype": str(arr.dtype), "shape": list(arr.shape)}
        if idx == 0:
            f = path / "out0.npy"
            np.save(f, arr)
            entry["file"] = str(f)
        else:
            import hashlib
            entry["sha256"] = hashlib.sha256(arr.tobytes()).hexdigest()
        manifest.append(entry)
    return manifest


def _worker_main(argv: "list[str]") -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--workload", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260807)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump-outputs", default=None)
    a = ap.parse_args(argv)

    scratch = Path(a.scratch)
    workloads = {w.key: w for w in cuda_workloads.all_workloads()}
    workload = workloads.get(a.workload)
    if workload is None:
        dump_public_json({
            "arm": a.arm, "workload": a.workload, "verdict": INSTRUMENT_ERROR,
            "instrument_errors": [f"unknown workload {a.workload!r}"]}, Path(a.out), indent=None)
        return 3

    model = bench_models.resolve(a.model, allow_download=False)
    try:
        rec = run_arm(a.arm, workload, model, iters=a.iters, warmup=a.warmup,
                      scratch=scratch, device_index=a.device, seed=a.seed)
    except Exception as exc:
        import traceback
        rec = ArmResult(arm=a.arm, workload=a.workload, model_key=a.model,
                        verdict=INSTRUMENT_ERROR)
        rec.instrument_errors.append(f"{exc!r}\n{traceback.format_exc()[:3000]}")

    outputs = getattr(rec, "first_outputs", None)
    names = getattr(rec, "output_names", [])
    manifest = []
    if outputs is not None and a.dump_outputs:
        try:
            manifest = _npy_dump(outputs, names, Path(a.dump_outputs))
        except Exception as exc:
            rec.instrument_errors.append(f"output dump failed: {exc!r}")
    payload = rec.to_dict()
    payload["outputs_manifest"] = manifest
    payload["partition_detail"] = getattr(rec, "partition_detail", {})
    dump_public_json(payload, Path(a.out))
    return 0 if rec.verdict in (ADMISSIBLE, SPLIT_FRAME) else 2


# ---------------------------------------------------------------------------
# Parent
# ---------------------------------------------------------------------------

def _interpreter_for(runtime: str) -> "str | None":
    if runtime == "host":
        return sys.executable
    return os.environ.get(CUDA_PYTHON_ENV)


def _decode(raw) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    for enc in ("utf-8", "utf-16-le", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace")


def dispatch_arm(arm: str, workload: Workload, *, iters: int, warmup: int,
                 scratch: Path, device: int, seed: int,
                 timeout: int = 3600) -> dict:
    """Run one arm in its own process with its own interpreter."""
    interp = _interpreter_for(ARM_RUNTIME[arm])
    out = scratch / f"result_{arm}_{workload.key}.json"
    dump = scratch / f"outputs_{arm}_{workload.key}"
    if interp is None or not Path(interp).is_file():
        return {"arm": arm, "workload": workload.key, "model_key": workload.model_key,
                "verdict": UNMEASURED,
                "refusals": [f"no interpreter for runtime {ARM_RUNTIME[arm]!r}; set "
                             f"{CUDA_PYTHON_ENV} to the python of the CUDA-12 venv. "
                             f"Without it this arm is absent, not slow."]}

    cmd = [interp, str(Path(__file__).resolve()), "--worker",
           "--arm", arm, "--workload", workload.key, "--model", workload.model_key,
           "--iters", str(iters), "--warmup", str(warmup),
           "--device", str(device), "--seed", str(seed),
           "--scratch", str(scratch), "--out", str(out), "--dump-outputs", str(dump)]

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = str(_HERE) + os.pathsep + env.get("PYTHONPATH", "")

    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"arm": arm, "workload": workload.key, "model_key": workload.model_key,
                "verdict": INSTRUMENT_ERROR,
                "instrument_errors": [f"arm timed out after {timeout}s"]}

    if not out.is_file():
        return {"arm": arm, "workload": workload.key, "model_key": workload.model_key,
                "verdict": INSTRUMENT_ERROR,
                "instrument_errors": [
                    f"worker exited {proc.returncode} without writing a record. "
                    f"stderr tail: {_decode(proc.stderr).strip()[-1500:]}"]}
    rec = json.loads(out.read_text("utf-8"))
    rec["worker_returncode"] = proc.returncode
    rec["outputs_dir"] = str(dump)
    return rec


def cross_arm_equivalence(records: "list[dict]", reference_arm: str = ARM_CPU_HOST) -> dict:
    """Compare each GPU arm's first output against the CPU reference's, across processes."""
    import numpy as np

    by_arm = {r["arm"]: r for r in records}
    ref = by_arm.get(reference_arm)
    result: dict = {"reference_arm": reference_arm, "arms": {}}
    if ref is None or not ref.get("outputs_manifest"):
        result["detail"] = (f"reference arm {reference_arm} produced no outputs; "
                            f"equivalence is UNMEASURED, not passing")
        return result

    ref_entry = ref["outputs_manifest"][0]
    ref_file = ref_entry.get("file")
    if not ref_file or not Path(ref_file).is_file():
        result["detail"] = "reference output file missing"
        return result
    ref_arr = np.load(ref_file)

    for rec in records:
        arm = rec["arm"]
        if arm == reference_arm:
            continue
        manifest = rec.get("outputs_manifest") or []
        if not manifest or not manifest[0].get("file"):
            result["arms"][arm] = {"verdict": UNMEASURED,
                                   "detail": "arm produced no comparable output"}
            continue
        f = manifest[0]["file"]
        if not Path(f).is_file():
            result["arms"][arm] = {"verdict": UNMEASURED, "detail": f"missing {f}"}
            continue
        arr = np.load(f)
        # The budget is widened for an arm that has declared it computes in a
        # narrower format than the graph asks for.  Read from the arm's own record
        # where present; a record predating that field falls back to the *documented*
        # default for its provider and is flagged as inferred, so nobody reads an
        # assumption as a measurement.
        regime = rec.get("compute_regime")
        inferred = not regime
        if inferred:
            regime = compute_regime(arm, None)
        result["arms"][arm] = compare_outputs(
            [arr], [ref_arr], [ref_entry["name"]], regime=regime)
        result["arms"][arm]["compute_regime"] = regime
        result["arms"][arm]["compute_regime_inferred"] = inferred
    return result


#: Sources that decide what the Vulkan attention path *does*.  A suite record pins
#: their digests so that a result cannot outlive the code it measured: change one of
#: these without re-running, and the staleness screen fails instead of letting the old
#: ranking pose as current.  Paths are repo-relative on purpose -- they are identity,
#: not location.
EP_PINNED_SOURCES = (
    "rust/shaders/glsl/gqa_f16.comp",
    "rust/src/ops/attention.rs",
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def ep_provenance() -> dict:
    """Identity of the Vulkan EP build under measurement.

    Two independent handles, because they fail differently.  ``lib_sha256`` is the
    binary that actually ran -- exact, but not committed, so nothing in the tree can
    be compared against it later.  ``pinned_sources`` *is* committed, which is what
    lets a test assert that a stored result still describes the current tree.
    """
    repo = _HERE.parent
    prov: dict = {"pinned_sources": {}, "pinned_source_set": list(EP_PINNED_SOURCES)}

    lib = os.environ.get(EP_LIB_ENV)
    if lib and Path(lib).is_file():
        prov["lib_sha256"] = _sha256_file(Path(lib))
        prov["lib_bytes"] = Path(lib).stat().st_size
    else:
        prov["lib_sha256"] = None
        prov["lib_unavailable_because"] = (
            f"{EP_LIB_ENV} is unset or does not point at a file")

    for rel in EP_PINNED_SOURCES:
        p = repo / rel
        prov["pinned_sources"][rel] = _sha256_file(p) if p.is_file() else None

    try:
        head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=30)
        prov["git_commit"] = head.stdout.strip() or None
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                               capture_output=True, text=True, timeout=60)
        # Recorded, not hidden: a measurement taken on a dirty tree is still a
        # measurement, but the reader is entitled to know the commit under-describes it.
        prov["git_tree_dirty"] = bool(dirty.stdout.strip())
        # The narrower and more load-bearing question.  Edits to the harness do not
        # change what the EP computes; edits to the pinned sources do.  A commit
        # reference is only a fair label for a measurement when *these* were clean.
        pinned = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain", "--"] + list(EP_PINNED_SOURCES),
            capture_output=True, text=True, timeout=60)
        prov["pinned_sources_dirty"] = bool(pinned.stdout.strip())
    except Exception as exc:
        prov["git_commit"] = None
        prov["git_error"] = repr(exc)
    return prov


def device_facts() -> dict:
    """Selected device identity + host facts, recorded once per suite run."""
    facts: dict = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    try:
        sys.path.insert(0, str(_HERE))
        import cuda_probe

        facts["nvidia"] = asdict(cuda_probe.nvidia_smi_facts())
    except Exception as exc:
        facts["nvidia"] = {"producer": "none_available", "error": repr(exc)}
    return facts


def run_suite(workloads: "list[Workload]", arms: "list[str]", *, iters: int, warmup: int,
              scratch: Path, device: int, seed: int, repeats: int = 1) -> dict:
    scratch.mkdir(parents=True, exist_ok=True)
    suite: dict = {
        "schema": "cuda_competition/1",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device_facts": device_facts(),
        "ep_provenance": ep_provenance(),
        "iters": iters, "warmup": warmup, "seed": seed, "repeats": repeats,
        "arms": list(arms),
        "fallback_split_threshold": FALLBACK_SPLIT_THRESHOLD,
        "results": [],
        "equivalence": {},
        "comparisons": [],
    }
    models = {}
    for w in workloads:
        if w.model_key not in models:
            models[w.model_key] = bench_models.resolve(w.model_key, allow_download=False).to_dict()
    suite["models"] = models

    for w in workloads:
        per_workload: "list[dict]" = []
        for rep in range(repeats):
            rep_scratch = scratch / f"rep{rep}"
            for arm in arms:
                rec = dispatch_arm(arm, w, iters=iters, warmup=warmup, scratch=rep_scratch,
                                   device=device, seed=seed)
                rec["repeat"] = rep
                per_workload.append(rec)
                suite["results"].append(rec)
                status = rec.get("verdict")
                med = rec.get("median_ms")
                med_s = f"{med:.4f} ms" if isinstance(med, (int, float)) else "-"
                print(f"  {w.key:<20} {arm:<14} {status:<16} {med_s}")
        rep0 = [r for r in per_workload if r.get("repeat") == 0]
        eq = cross_arm_equivalence(rep0)
        suite["equivalence"][w.key] = eq
        suite["comparisons"].append(compare_workload(w, per_workload, equivalence=eq))

    return suite


def compare_workload(workload: Workload, records: "list[dict]",
                     equivalence: "dict | None" = None) -> dict:
    """Vulkan-vs-CUDA for one workload, admissible arms only, with an interval.

    An arm whose outputs did not match the CPU reference is excluded from the ratio.
    A faster arm that computes something else is not faster; it is wrong.  The
    exclusion is recorded so the reader sees why the row is empty.
    """
    equivalence = equivalence or {}
    eq_arms = equivalence.get("arms") or {}
    disqualified = {arm for arm, res in eq_arms.items()
                    if res.get("verdict") not in ("MATCH", None)}

    def samples(arm: str) -> "list[float]":
        out: "list[float]" = []
        for r in records:
            if r.get("arm") == arm and r.get("verdict") == ADMISSIBLE:
                out.extend(r.get("steady_ms") or [])
        return out

    vk = samples(ARM_VULKAN)
    cu = samples(ARM_CUDA)
    cpu_h = samples(ARM_CPU_HOST)
    cpu_c = samples(ARM_CPU_CUDA_RT)

    entry: dict = {
        "workload": workload.key,
        "family": workload.family,
        "seq_len": workload.seq_len,
        "past_len": workload.past_len,
        "n_vulkan": len(vk), "n_cuda": len(cu),
        "equivalence_disqualified_arms": sorted(disqualified),
        "verdict": UNMEASURED,
    }
    if disqualified & {ARM_VULKAN, ARM_CUDA}:
        entry["detail"] = (
            f"arm(s) {sorted(disqualified & {ARM_VULKAN, ARM_CUDA})} did not match the CPU "
            f"reference within the derived ULP budget; a speed ratio between arms computing "
            f"different answers is not a performance result")
        entry["verdict"] = NOT_EQUIVALENT
        return entry
    if not vk or not cu:
        missing = [name for name, s in (("vulkan", vk), ("cuda", cu)) if not s]
        entry["detail"] = (f"no admissible samples for {', '.join(missing)}; a missing arm is "
                           f"not a losing arm and no ratio is quotable")
        return entry

    entry["vulkan"] = summarise(vk)
    entry["cuda"] = summarise(cu)
    # ratio > 1 means CUDA takes longer, i.e. Vulkan is faster.
    entry["speedup_vulkan_over_cuda"] = bootstrap_ratio_ci(vk, cu)
    if cpu_h and cpu_c:
        entry["runtime_version_confound"] = {
            "note": ("CPU EP in each runtime on identical bytes. Any Vulkan-vs-CUDA ratio "
                     "within this bracket is not distinguishable from the ORT version "
                     "difference between the two arms."),
            "cpu_host_ort": summarise(cpu_h),
            "cpu_cuda_rt_ort": summarise(cpu_c),
            "ratio_cpu_cudart_over_cpu_host": bootstrap_ratio_ci(cpu_h, cpu_c),
        }

    ci = entry["speedup_vulkan_over_cuda"]
    if ci.get("lo") is not None and ci["lo"] > 1.0:
        entry["verdict"] = "VULKAN_FASTER"
    elif ci.get("hi") is not None and ci["hi"] < 1.0:
        entry["verdict"] = "CUDA_FASTER"
    else:
        entry["verdict"] = "INDISTINGUISHABLE"
    return entry


def render(suite: dict) -> str:
    lines: "list[str]" = []
    lines.append("# Vulkan EP vs ORT CUDA EP — issue #69")
    lines.append("")
    nv = (suite.get("device_facts") or {}).get("nvidia") or {}
    lines.append(f"device      : {nv.get('device_names')} driver={nv.get('driver_version')} "
                 f"cuda_driver={nv.get('cuda_driver_version')}")
    lines.append(f"iters/warmup: {suite['iters']}/{suite['warmup']}  repeats={suite['repeats']}")
    lines.append("")
    lines.append("## Arm admissibility")
    lines.append("")
    lines.append("Fallback shares: `time` gates admissibility (fusion-invariant); `nodes` is the "
                 "fusion-blind profile-node share; `graph` is unclaimed/probed original graph "
                 "nodes and exists only for the fusing Vulkan EP.")
    lines.append("")
    lines.append("| workload | arm | verdict | ORT | profile nodes | fb-time | fb-nodes | "
                 "fb-graph | median ms | RSD% |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in suite["results"]:
        def pct(v):
            return "-" if v is None else f"{v:.1%}"
        med = r.get("median_ms")
        med_s = "-" if med is None else f"{med:.4f}"
        rsd = r.get("rsd_pct")
        rsd_s = "-" if rsd is None else f"{rsd:.2f}"
        lines.append(f"| {r['workload']} | {r['arm']} | {r.get('verdict')} | "
                     f"{r.get('ort_version','-')} | {r.get('executed_nodes','-')} | "
                     f"{pct(r.get('cpu_fallback_fraction_time'))} | "
                     f"{pct(r.get('cpu_fallback_fraction_nodes'))} | "
                     f"{pct(r.get('cpu_fallback_fraction_graph'))} | {med_s} | {rsd_s} |")
    lines.append("")
    lines.append("## Output equivalence (vs CPU EP reference)")
    lines.append("")
    lines.append("Budget is in **ULP at the reference tensor's peak magnitude** — see "
                 "`EQUIVALENCE`/`ROUNDING_DEPTH_BOUND` in this module for why raw ULP and "
                 "absolute tolerance were both rejected. `regime` is the precision the arm's "
                 "provider declares it computes fp32 ops in; the CUDA EP defaults to **TF32** "
                 "(10 mantissa bits, not 23), which both widens its budget and is part of why "
                 "it is fast. `top-k` is conditioned on rows whose reference ranking the "
                 "numerics can resolve; `UNRESOLVED` means the check had no power on this "
                 "input and abstained rather than passing.")
    lines.append("")
    lines.append("| workload | arm | verdict | regime | max peak-ULP | budget | raw ULP | "
                 "max abs diff | top-k | resolvable rows |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for wl, eq in (suite.get("equivalence") or {}).items():
        for arm, res in (eq.get("arms") or {}).items():
            outs = res.get("outputs") or []
            first = outs[0] if outs else {}
            task = first.get("task") or {}
            pk = first.get("max_peak_ulp")
            pk_s = "-" if pk is None else f"{pk:.2f}"
            agree = task.get("row_agreement")
            agree_s = task.get("verdict", "-") if agree is None else f"{agree:.3f}"
            lines.append(
                f"| {wl} | {arm} | {res.get('verdict')} | {res.get('compute_regime','-')} | "
                f"{pk_s} | {first.get('peak_ulp_budget','-')} | "
                f"{first.get('max_raw_ulp','-')} | "
                f"{first.get('max_abs_diff','-')} | {agree_s} | "
                f"{task.get('resolvable_rows','-')}/{task.get('rows','-')} |")
    lines.append("")
    lines.append("## Vulkan vs CUDA")
    lines.append("")
    lines.append("| workload | vulkan med ms | cuda med ms | speedup (vk over cuda) | 95% CI | ORT-version bracket | verdict |")
    lines.append("|---|---|---|---|---|---|---|")
    for c in suite["comparisons"]:
        if "vulkan" not in c:
            lines.append(f"| {c['workload']} | - | - | - | - | - | {c['verdict']} |")
            continue
        ci = c["speedup_vulkan_over_cuda"]
        lo = ci.get("lo")
        hi = ci.get("hi")
        ci_s = "-" if lo is None else f"[{lo:.3f}, {hi:.3f}]"
        conf = (c.get("runtime_version_confound") or {}).get("ratio_cpu_cudart_over_cpu_host")
        conf_s = "-"
        if conf and conf.get("lo") is not None:
            conf_s = f"[{conf['lo']:.3f}, {conf['hi']:.3f}]"
        lines.append(f"| {c['workload']} | {c['vulkan']['median_ms']:.4f} | "
                     f"{c['cuda']['median_ms']:.4f} | "
                     f"{(ci.get('ratio') or float('nan')):.3f} | {ci_s} | {conf_s} | "
                     f"{c['verdict']} |")
    return "\n".join(lines)


def reanalyse(suite: dict) -> dict:
    """Re-derive equivalence and comparisons from a stored suite's own records.

    The timings, the dumped output tensors and the provider attribution are the
    *evidence*; the equivalence verdicts and the speed comparisons are *conclusions
    drawn from it*.  Separating the two means a reviewer can re-run the analysis
    against a committed record without a GPU, and — the reason this exists — an
    improvement to the equivalence instrument can be applied to a measurement that
    already happened instead of silently invalidating half an hour of benchmarking.

    Re-running the arms instead would confound an analysis change with a fresh set
    of samples on a shared box, which is precisely the confound this suite spends
    two CPU arms bracketing.
    """
    out = dict(suite)
    out["reanalysed"] = True
    out["equivalence"] = {}
    out["comparisons"] = []
    by_workload: dict = {}
    for rec in suite.get("results", []):
        by_workload.setdefault(rec["workload"], []).append(rec)
    workloads = {w.key: w for w in cuda_workloads.all_workloads()}
    for key, recs in by_workload.items():
        w = workloads.get(key)
        if w is None:
            continue
        rep0 = [r for r in recs if r.get("repeat", 0) == 0]
        eq = cross_arm_equivalence(rep0)
        out["equivalence"][key] = eq
        out["comparisons"].append(compare_workload(w, recs, equivalence=eq))
    return out


def main(argv: "list[str] | None" = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--worker" in argv:
        return _worker_main(argv)

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arms", nargs="*", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--workloads", nargs="*", default=None,
                    help="workload keys; default is every workload")
    ap.add_argument("--family", nargs="*", default=None,
                    help="restrict to workload families (prefill/decode/vision/encoder)")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260807)
    ap.add_argument("--scratch", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--reanalyse", default=None, metavar="SUITE_JSON",
                    help="re-derive equivalence and comparisons from a stored suite "
                         "record instead of running anything")
    a = ap.parse_args(argv)

    if a.reanalyse:
        src = Path(a.reanalyse)
        suite = reanalyse(json.loads(src.read_text("utf-8")))
        out = Path(a.out) if a.out else src.with_name(src.stem + "_reanalysed.json")
        dump_public_json(suite, out)
        report = render(suite)
        write_public_text(report, out.with_suffix(".md"))
        print(report)
        print()
        print(f"record: {out}")
        return 0

    every = cuda_workloads.all_workloads()
    if a.workloads:
        wanted = set(a.workloads)
        selected = [w for w in every if w.key in wanted]
    elif a.family:
        fams = set(a.family)
        selected = [w for w in every if w.family in fams]
    else:
        selected = every
    if not selected:
        print("no workloads selected")
        return 3

    scratch = Path(a.scratch) if a.scratch else _HERE / "results" / "_cuda69" / "scratch"
    suite = run_suite(selected, a.arms, iters=a.iters, warmup=a.warmup, scratch=scratch,
                      device=a.device, seed=a.seed, repeats=a.repeats)

    out = Path(a.out) if a.out else _HERE / "results" / "_cuda69" / "suite.json"
    dump_public_json(suite, out)
    report = render(suite)
    write_public_text(report, out.with_suffix(".md"))
    print()
    print(report)
    print()
    print(f"record: {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
