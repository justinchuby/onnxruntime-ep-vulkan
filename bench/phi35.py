"""Model-level benchmark for Phi-3.5 — the project's first measurement of a real model.

WHY THIS FILE IS SEPARATE FROM ``bench.py``
===========================================
``bench.py`` benchmarks *op graphs* we build ourselves. This benchmarks an *artifact somebody
else produced*, and the difference is not size — it is what may be concluded. A synthetic
``MatMulNBits`` graph is a statement about a kernel. Phi-3.5 at a pinned producer-at-version is
a statement about a model, and `DESIGN.md` §10.0 attaches obligations to that which do not
apply to op graphs: the metric of record (``claimed_op_coverage``, ``island_count``,
``largest_island_flops``) may only be reported when the same run has established
``model_output_equivalence == MATCH``.

WHAT THIS FILE REFUSES TO DO
============================
Three refusals, all structural (exit 2, no numbers printed), each named after the fabricated
result it exists to prevent. This project has produced three wrong-but-plausible numbers, and
every one of them would have been caught by a gate on the *effect* rather than a gate on a
precondition:

1. **1.70×** — measured through an EP that could not load under ORT 1.27. ORT *printed*
   ``API version [28] is not available`` and did not raise, so every node ran on the CPU EP
   wearing our provider's name. → :func:`refuse_if_ep_absent`, plus ``bench.MIN_ORT``.
2. **1.45×** — measured through an EP that loads correctly and declines every node. The CPU
   did all the work; the variance between two CPU runs became a speedup. →
   :func:`refuse_if_nothing_claimed`.
3. **argmax 0** — 161 nodes dispatched, ``compute_failures: 0``, 300 tests green, and the
   logits were all zero, because of a write to an undeclared descriptor binding that both
   drivers silently dropped. Nothing in the instrument set went red. →
   :func:`classify_outputs` and :func:`refuse_if_not_match`, which run the CPU comparison *in
   the same process, on the same artifact, in the same run that is timed*.

The third is the important one and it is why this module does its own correctness comparison
rather than reading a verdict somebody else recorded earlier. §10.0: "*No CPU-only comparison
was performed on this artifact in this run*" is precisely ``UNMEASURED``, and a verdict from a
previous run of a previous build is exactly that. The vocabulary — the field name
``model_output_equivalence`` and the values ``MATCH`` / ``DIVERGENT`` / ``UNMEASURED`` — is
Trinity's and Switch's per §10.0; this module consumes it and does not invent a parallel one.

WHAT THIS FILE WILL NOT LET YOU CONCLUDE
========================================
Even on a clean ``MATCH`` this measures **one configuration**, and the configuration is
currently pathological in a way that must travel with the number:

* Every tensor is host-staged (device-backed allocation is behind
  ``ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY``, which is off by default).
  ``epctl --check-counters --require-device-memory`` exits 1 on this state. A number measured
  here is a number about *staging*, and is labelled ``staging-bound`` in the result record so
  that it cannot be quoted later as "what the Vulkan EP does".
* Coverage is 257 of 363 nodes probed, in **257 separate islands** (the EP's own
  ``subgraphs_live`` counter), so the graph crosses the device boundary 257 times per
  inference and buys back only the GEMV. The expected result is a *slowdown*, and a slowdown
  honestly measured is the most useful number available today: it prices the boundary, which
  is what tells us which op to write next.
* There is **no GPU kernel time** in this measurement on any device, because no ``VkQueryPool``
  exists yet — the timestamp hooks in the recording path are still comments
  (``vk/session.rs`` and ``vk/dispatch_integration.rs``). Everything here is host wall time.
  See :mod:`timestamp_audit` for what *is* verified about the timestamp path.

Usage::

    python bench/phi35.py --device 0 --iters 20 --out bench/results/phi35-dev0.json
    python bench/phi35.py --all-devices --iters 20 --out bench/results/phi35.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import devices as device_mod  # noqa: E402
import environment  # noqa: E402
import producers  # noqa: E402
import stats as stats_mod  # noqa: E402
from stats import Sample  # noqa: E402

EP_NAME = "VulkanExecutionProvider"
EP_LIB_ENV = "ONNXRUNTIME_VULKAN_EP_LIB"
CLAIM_LOG_ENV = "ONNXRUNTIME_EP_VULKAN_CLAIM_LOG"
COUNTERS_ENV = "ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE"
MODEL_ENV = "ONNXRUNTIME_EP_VULKAN_PHI35_ONNX"

#: §10.0 verdict vocabulary. Owned by the metric-of-record ruling, not by this file.
MATCH = "MATCH"
DIVERGENT = "DIVERGENT"
UNMEASURED = "UNMEASURED"

#: Top-k agreement required for ``MATCH``. §10.0 says "argmax and top-k agree"; the strictest
#: reading is used, because the failure this gate exists to catch (all-zero logits) produced
#: 0/10 and a laxer bar would still have caught it — but a *partially* wrong kernel would not,
#: and that is the next failure, not the last one.
TOP_K = 10

#: The canonical artifact and its feeds live in Trinity's `tests/ops/test_phi35.py`. They are
#: imported rather than copied: the artifact definition is hers, and two definitions of "the
#: Phi-3.5 benchmark input" that drift apart would make her correctness verdict and my timing
#: numbers describe different runs while appearing to describe the same one.
_TESTS_OPS = _HERE.parent / "tests" / "ops"


class ArtifactUnavailable(RuntimeError):
    """The pinned Phi-3.5 artifact or its definition is not available on this machine."""


def _trinity_module():
    """Import `tests/ops/test_phi35.py` for the artifact path and the canonical feeds."""
    if str(_TESTS_OPS) not in sys.path:
        sys.path.insert(0, str(_TESTS_OPS))
    try:
        import test_phi35  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ArtifactUnavailable(
            f"could not import tests/ops/test_phi35.py ({exc}). That module owns the artifact "
            "path and the canonical feeds; this benchmark deliberately has no copy of them."
        ) from exc
    return test_phi35


def model_path() -> Path:
    """The pinned artifact. ``MODEL_ENV`` overrides for a machine with a different cache root."""
    override = os.environ.get(MODEL_ENV)
    if override:
        p = Path(override)
    else:
        p = Path(_trinity_module()._ONNX_FILE)
    if not p.is_file():
        raise ArtifactUnavailable(
            f"Phi-3.5 artifact not found at {p}. Set {MODEL_ENV} to its .onnx path. "
            "The model is never committed to this repo."
        )
    return p


def phi35_producer() -> producers.Producer:
    """Provenance for the artifact.

    §4.18 / OP_COVERAGE: op coverage is relative to a producer, not to a model architecture —
    and a benchmark artifact is relative to its producer too. This build is the ORT GenAI
    builder's ``com.microsoft`` contrib graph (``MatMulNBits``, ``GroupQueryAttention``,
    ``SkipSimplifiedLayerNormalization``); Justin's ``mobius`` builder emits
    ``ai.onnx::Attention`` @23 and ``RMSNormalization`` from the same weights and would be a
    different artifact with a different island structure and a different number.
    """
    p = producers.ort_genai_builder(
        "Phi-3.5-mini-instruct/cuda-int4-rtn-block-32",
        opsets={"com.microsoft": 1, "ai.onnx": 14},
        model_family="phi",
    )
    # Self-enforcing: this module is *named* after a model family, so the producer must be one
    # that actually built that family at a recorded version. If someone repoints this benchmark
    # at a differently-produced artifact without updating the provenance, construction raises
    # rather than the report quietly acquiring a family name it did not earn.
    producers.assert_family_label_is_earned("phi35_model_benchmark", ["phi"], p)
    return p


# ---------------------------------------------------------------------------
# The verdict — §10.0
# ---------------------------------------------------------------------------

def classify_outputs(vk_out, cpu_out, *, top_k: int = TOP_K) -> dict:
    """Compare a Vulkan run against a CPU-only run of the same artifact → §10.0 verdict.

    Returns a record carrying ``model_output_equivalence`` and, always, the evidence — §10.0
    requires a ``DIVERGENT`` run to report *which* outputs, max-abs-diff per output, and
    argmax/top-k agreement, and there is no reason to withhold the same evidence on a ``MATCH``:
    a ``MATCH`` whose max-abs-diff is 0.0000 on every output is a different (and more
    suspicious) event than one whose diff is 0.03, and only the recorded number distinguishes
    them.

    ``UNMEASURED`` is returned when there is nothing to compare — never as a silence, and never
    degraded into a soft ``MATCH``.
    """
    import numpy as np

    if not vk_out or not cpu_out:
        return {
            "model_output_equivalence": UNMEASURED,
            "reason": "one or both runs produced no outputs",
            "outputs": [],
        }
    if len(vk_out) != len(cpu_out):
        return {
            "model_output_equivalence": DIVERGENT,
            "reason": f"output count differs: vk={len(vk_out)} cpu={len(cpu_out)}",
            "outputs": [],
        }

    details: "list[dict]" = []
    divergent = False
    for idx, (a, b) in enumerate(zip(vk_out, cpu_out)):
        va = np.asarray(a).astype(np.float32)
        vb = np.asarray(b).astype(np.float32)
        entry: dict = {"index": idx, "shape": list(va.shape)}
        if va.shape != vb.shape:
            entry["disagrees"] = True
            entry["reason"] = f"shape {list(va.shape)} vs {list(vb.shape)}"
            details.append(entry)
            divergent = True
            continue

        finite = bool(np.isfinite(va).all())
        entry["all_finite"] = finite
        entry["vk_max_abs"] = float(np.abs(va).max()) if va.size else 0.0
        entry["cpu_max_abs"] = float(np.abs(vb).max()) if vb.size else 0.0
        entry["max_abs_diff"] = float(np.abs(va - vb).max()) if va.size else 0.0
        denom = entry["cpu_max_abs"] or 1.0
        entry["max_rel_diff"] = entry["max_abs_diff"] / denom

        # The all-zero guard is stated as its own field rather than folded into the diff,
        # because all-zero output was the actual failure and it is worth being able to grep for.
        entry["vk_is_effectively_zero"] = entry["vk_max_abs"] <= 1e-3 < entry["cpu_max_abs"]

        if va.ndim >= 2 and va.shape[-1] >= top_k:
            flat_a = va.reshape(-1, va.shape[-1])[0]
            flat_b = vb.reshape(-1, vb.shape[-1])[0]
            entry["argmax_vk"] = int(flat_a.argmax())
            entry["argmax_cpu"] = int(flat_b.argmax())
            entry["argmax_agrees"] = entry["argmax_vk"] == entry["argmax_cpu"]
            ta = set(np.argsort(-flat_a)[:top_k].tolist())
            tb = set(np.argsort(-flat_b)[:top_k].tolist())
            entry["top_k"] = top_k
            entry["top_k_overlap"] = len(ta & tb)
            entry["top_k_agrees"] = entry["top_k_overlap"] == top_k
            agrees = entry["argmax_agrees"] and entry["top_k_agrees"]
        else:
            # Not logits-shaped: fall back to a value comparison. §9.1's tolerance policy for
            # fp16 accumulation chains; 1e-2 relative is generous and is stated as such.
            entry["argmax_agrees"] = None
            agrees = entry["max_rel_diff"] <= 1e-2

        entry["disagrees"] = (not finite) or entry["vk_is_effectively_zero"] or not agrees
        divergent = divergent or entry["disagrees"]
        details.append(entry)

    return {
        "model_output_equivalence": DIVERGENT if divergent else MATCH,
        "reason": None if not divergent else "at least one output disagrees with the CPU oracle",
        "outputs": details,
    }


# ---------------------------------------------------------------------------
# The refusals
# ---------------------------------------------------------------------------

def refuse_if_ep_absent(used_providers: "list[str]") -> "str | None":
    """The 1.70× refusal. ORT does not raise when a plugin EP fails to load; it omits the name."""
    if EP_NAME in used_providers:
        return None
    return (
        f"{EP_NAME} is not in session.get_providers() ({used_providers}). ORT falls back to CPU "
        "silently — timing this session would time the CPU EP under our provider's name. This is "
        "the shape of the 1.70x result (ORT 1.27 printed 'API version [28] is not available' and "
        "did not raise). Refusing to time."
    )


def refuse_if_nothing_claimed(claimed_nodes: "int | None") -> "str | None":
    """The 1.45× refusal. An EP that loads and claims nothing leaves CPU-vs-CPU jitter behind."""
    if claimed_nodes is None:
        return (
            "claimed-node count is unknown (no claim log was produced). A timing harness that "
            "cannot tell whether the EP did any work is not a harness. Refusing to time."
        )
    if claimed_nodes <= 0:
        return (
            f"the EP claimed {claimed_nodes} nodes. Every node ran on CPU; any difference "
            "measured would be CPU-vs-CPU variance wearing the EP's name. This is the shape of "
            "the 1.45x result. Refusing to time."
        )
    return None


def refuse_if_not_match(verdict: str) -> "str | None":
    """The §10.0 gate. ``DIVERGENT`` and ``UNMEASURED`` both void the numbers; neither is soft."""
    if verdict == MATCH:
        return None
    if verdict == DIVERGENT:
        return (
            "model_output_equivalence = DIVERGENT. The EP computed a different answer from the "
            "CPU oracle on this artifact in this run. Per DESIGN.md §10.0 a wrong answer voids "
            "the metric rather than discounting it, and a speed measurement on a wrong result is "
            "not a weak result — it is a fabricated one. Refusing to report timings."
        )
    return (
        "model_output_equivalence = UNMEASURED. No CPU-only comparison was performed on this "
        "artifact in this run. Per DESIGN.md §10.0 UNMEASURED is the default and is not a soft "
        "MATCH. Refusing to report timings."
    )


# ---------------------------------------------------------------------------
# Configuration labelling
# ---------------------------------------------------------------------------

DEVICE_MEMORY_ENV = "ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY"


def staging_label(counters: "dict | None") -> dict:
    """Classify the memory configuration the number was measured in.

    Two independent grounds, and the weaker one is not allowed to overrule the stronger:

    * **By configuration.** Device-backed allocation lives behind ``ONNXRUNTIME_EP_VULKAN_
      DEVICE_MEMORY`` (``factory.rs``). With that unset, every tensor is host-staged *by
      construction* — this is not an observation that could come out otherwise, and it holds
      even when the counters file carries no allocator keys.
    * **By observation.** ``alloc_device_backed_spans == 0`` with a non-zero
      ``alloc_staged_spans``. That is the state ``epctl --check-counters
      --require-device-memory`` exits 1 on.

    The label travels *inside* the result record so that it cannot be separated from the number
    by a copy-paste into a summary. A latency measured in this configuration is a measurement of
    staging; it is not "what the Vulkan EP does".
    """
    env_on = os.environ.get(DEVICE_MEMORY_ENV, "") not in ("", "0")
    out: dict = {
        "device_memory_env": DEVICE_MEMORY_ENV,
        "device_memory_enabled": env_on,
    }
    c = counters or {}
    staged = c.get("alloc_staged_spans")
    backed = c.get("alloc_device_backed_spans")
    out["staged_spans"] = staged
    out["staged_bytes"] = c.get("alloc_staged_bytes")
    out["device_backed_spans"] = backed

    if not env_on:
        out["configuration"] = "staging-bound"
        out["basis"] = "configuration"
        out["note"] = (
            f"{DEVICE_MEMORY_ENV} is not set, so device-backed allocation is off and every "
            "tensor is host-staged by construction. This number measures a staging-bound "
            "configuration and may not be quoted as 'what the Vulkan EP does'."
        )
        return out
    if staged is None or backed is None:
        out["configuration"] = "unknown"
        out["basis"] = "none"
        out["note"] = (
            f"{DEVICE_MEMORY_ENV} is set but the counters file carries no allocator span "
            "counters, so the configuration is unknown — which is not the same as device-backed."
        )
        return out
    out["configuration"] = ("staging-bound" if (backed == 0 and staged > 0)
                            else "device-backed" if (staged == 0 and backed > 0) else "mixed")
    out["basis"] = "observation"
    out["note"] = {
        "staging-bound": "every tensor is host-staged; epctl --check-counters "
                         "--require-device-memory exits 1 on this state.",
        "device-backed": "no staged spans observed.",
        "mixed": "both staged and device-backed spans observed; attribution is ambiguous.",
    }[out["configuration"]]
    return out


def boundary_cost(vk_median_ms: float, cpu_median_ms: float,
                  island_count: "int | None") -> dict:
    """Price the device-boundary crossings, honestly and with the direction of the bound stated.

    With ``island_count`` single-node islands, the graph leaves and re-enters the device once per
    island per inference. The whole host-side difference is attributed to those crossings::

        per_island_ms = (vk_median_ms - cpu_median_ms) / island_count

    That is a **lower** bound on the true per-crossing cost, not an estimate of it, and the
    reason is worth stating because the sign is counter-intuitive: the same difference also
    contains whatever the GPU *saved* on the GEMV it took over. If the GPU kernel is faster than
    the CPU's, the saving offsets part of the boundary cost, so the true boundary cost is larger
    than this figure — the arithmetic below cannot separate them and does not pretend to.

    Separating them needs GPU kernel time, which needs ``VkQueryPool`` timestamps, which do not
    exist yet (see :mod:`timestamp_audit`). That is recorded here as the missing instrument
    rather than approximated.

    ``island_count`` is ``None`` when the EP did not report ``subgraphs_live``. In that case no
    per-island figure is produced at all — dividing by an assumed island count is exactly how a
    plausible-but-wrong number gets made, and this project has three of those already.
    """
    delta = vk_median_ms - cpu_median_ms
    per_island = (delta / island_count) if island_count else None
    return {
        "vk_median_ms": vk_median_ms,
        "cpu_median_ms": cpu_median_ms,
        "delta_ms": delta,
        "island_count": island_count,
        "island_count_source": "EP counter `subgraphs_live`",
        "per_island_ms_lower_bound": per_island,
        "bound_direction": "lower",
        "why_lower": "the host-side delta nets the boundary cost against whatever the GPU saved "
                     "on the work it took over; the two cannot be separated without GPU kernel "
                     "time, which requires VkQueryPool timestamps that do not exist yet",
        "separating_instrument": "VkQueryPool timestamps around each dispatch "
                                 "(docs/PERF.md §3, routed to Switch; not implemented)",
    }


def dispatch_accounting(counters: "dict | None", island_count: "int | None",
                        inference_count: int) -> dict:
    """Check that every island actually ran on every inference. A falsifier, not a statistic.

    If the EP reports ``subgraphs_live`` islands and the harness performed ``inference_count``
    inferences, then ``compute_calls`` must be exactly their product. Anything less means some
    island was not executed — silently, because a subgraph that is never invoked reports no
    error, raises nothing, and leaves ``compute_failures`` at zero (§9.1.3: that counter is an
    execution-status counter and never a correctness signal).

    This is the instrument that goes red if the island count used to price the boundary is wrong,
    or if the EP short-circuited work the timing was supposed to include. It is cheap and exact —
    integer equality, no tolerance — which is the whole reason it is worth having.
    """
    out: dict = {
        "inference_count": inference_count,
        "island_count": island_count,
        "compute_calls": (counters or {}).get("compute_calls"),
        "dispatches_executed": (counters or {}).get("dispatches_executed"),
        "compute_failures": (counters or {}).get("compute_failures"),
        "compute_failures_note": "an execution-status counter, never a correctness signal "
                                 "(DESIGN.md §9.1.3). It is recorded, not relied on.",
    }
    if island_count is None or out["compute_calls"] is None:
        out["consistent"] = None
        out["detail"] = "not checkable: island count or compute_calls unavailable"
        return out
    expected = island_count * inference_count
    out["expected_compute_calls"] = expected
    out["consistent"] = out["compute_calls"] == expected
    out["detail"] = (
        f"compute_calls {out['compute_calls']} == {island_count} islands x {inference_count} "
        f"inferences" if out["consistent"] else
        f"compute_calls {out['compute_calls']} != expected {expected}: some island did not run on "
        f"every inference, or the island count is not what the counter says. The timing above "
        f"does not measure what it claims to."
    )
    return out


# ---------------------------------------------------------------------------
# Worker — one device, one subprocess
# ---------------------------------------------------------------------------

def _time_run(sess, feeds, iters: int, warmup: int, name: str) -> Sample:
    for _ in range(warmup):
        sess.run(None, feeds)
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(None, feeds)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return Sample(name=name, samples=samples)


def _read_claim_log(path: Path) -> "list[dict]":
    if not path.exists():
        return []
    out = []
    for line in path.read_text("utf-8", "replace").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def measure(device_index: int, iters: int, warmup: int, scratch: Path) -> dict:
    """Run the whole gated measurement for one device, in this process.

    Ordering matters and is not incidental: correctness is established **before** anything is
    timed, on the same session objects that are then timed. A harness that times first and
    checks afterwards can still emit a number if the check crashes.
    """
    import numpy as np  # noqa: F401  (imported for its side effect on ORT's dtype handling)
    import onnxruntime as ort

    t35 = _trinity_module()
    path = model_path()
    feeds = t35._build_phi35_feeds()

    lib = os.environ.get(EP_LIB_ENV)
    record: dict = {
        "artifact": str(path),
        "device_index": device_index,
        "iters": iters,
        "warmup": warmup,
        "ort_version": ort.__version__,
        "producer": phi35_producer().to_dict(),
        "model_output_equivalence": UNMEASURED,
        "refusals": [],
    }
    if not lib or not Path(lib).is_file():
        record["refusals"].append(
            f"{EP_LIB_ENV} is not set or does not point at a file; there is no EP to measure.")
        return record
    try:
        ort.register_execution_provider_library(EP_NAME, str(Path(lib).resolve()))
    except Exception as exc:
        if "already registered" not in str(exc):
            record["refusals"].append(f"EP registration failed: {exc}")
            return record

    claim_log = scratch / f"phi35_claim_dev{device_index}_{os.getpid()}.jsonl"
    claim_log.unlink(missing_ok=True)
    os.environ[CLAIM_LOG_ENV] = str(claim_log)

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.add_session_config_entry("ep.device_index", str(device_index))
    vk_sess = ort.InferenceSession(str(path), opts, providers=[EP_NAME, "CPUExecutionProvider"])
    os.environ.pop(CLAIM_LOG_ENV, None)

    used = list(vk_sess.get_providers())
    record["providers"] = used
    refusal = refuse_if_ep_absent(used)
    if refusal:
        record["refusals"].append(refusal)
        return record

    claims = _read_claim_log(claim_log)
    claim_log.unlink(missing_ok=True)
    claimed = [c for c in claims if c.get("claimed")]
    record["claimed_nodes"] = len(claimed) if claims else None
    record["total_nodes_probed"] = len(claims) if claims else None
    refusal = refuse_if_nothing_claimed(record["claimed_nodes"])
    if refusal:
        record["refusals"].append(refusal)
        return record

    cpu_opts = ort.SessionOptions()
    cpu_opts.log_severity_level = 3
    cpu_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    cpu_sess = ort.InferenceSession(str(path), cpu_opts, providers=["CPUExecutionProvider"])

    # --- correctness first, on the sessions that are about to be timed ---
    vk_out = vk_sess.run(None, feeds)
    cpu_out = cpu_sess.run(None, feeds)
    verdict = classify_outputs(vk_out, cpu_out)
    record.update(verdict)
    refusal = refuse_if_not_match(record["model_output_equivalence"])
    if refusal:
        record["refusals"].append(refusal)
        return record

    # --- only now may anything be timed ---
    vk = _time_run(vk_sess, feeds, iters, warmup, f"phi35.vulkan.dev{device_index}")
    cpu = _time_run(cpu_sess, feeds, iters, warmup, "phi35.cpu")
    record["vulkan"] = vk.to_dict()
    record["cpu"] = cpu.to_dict()
    # Raw samples are kept, not just the summary. A median with a p05 at a quarter of it is a
    # different event from a tight distribution with the same median, and only the samples
    # distinguish them. They are small and they are the evidence.
    record["vulkan_samples_ms"] = [round(x, 4) for x in vk.samples]
    record["cpu_samples_ms"] = [round(x, 4) for x in cpu.samples]
    record["vulkan_drift"] = stats_mod.drift(vk.samples)
    record["cpu_drift"] = stats_mod.drift(cpu.samples)
    # Every inference this process performed through the Vulkan session: one for the correctness
    # comparison, then the warmup, then the timed runs. Used by :func:`dispatch_accounting`.
    record["vk_inference_count"] = 1 + warmup + iters
    return record


def _worker_main(argv: "list[str]") -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--device", type=int, required=True)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", required=True)
    a = ap.parse_args(argv)
    scratch = Path(a.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        rec = measure(a.device, a.iters, a.warmup, scratch)
    except ArtifactUnavailable as exc:
        rec = {"device_index": a.device, "refusals": [str(exc)],
               "model_output_equivalence": UNMEASURED}
    except Exception as exc:  # pragma: no cover - environment dependent
        rec = {"device_index": a.device, "refusals": [f"measurement raised: {exc!r}"],
               "model_output_equivalence": UNMEASURED}
    Path(a.out).write_text(json.dumps(rec, indent=2), "utf-8")
    return 0 if not rec.get("refusals") else 2


# ---------------------------------------------------------------------------
# Parent — orchestrates devices, never blends them
# ---------------------------------------------------------------------------

def _run_device(device_index: int, iters: int, warmup: int, scratch: Path) -> dict:
    """Run one device in a subprocess.

    A subprocess per device, not a loop in one process, for two reasons that are both about the
    number being attributable: the EP's counters file is only complete at teardown (they are
    written from a process-exit hook, and a process that has torn down cannot print into ours),
    and ORT's EP registration and device binding are process-global, so a second session on a
    different device in the same process is not obviously a clean slate.
    """
    scratch.mkdir(parents=True, exist_ok=True)
    out = scratch / f"phi35_dev{device_index}.json"
    counters = scratch / f"phi35_counters_dev{device_index}.json"
    out.unlink(missing_ok=True)
    counters.unlink(missing_ok=True)
    env = dict(os.environ)
    env[COUNTERS_ENV] = str(counters)
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--device",
           str(device_index), "--iters", str(iters), "--warmup", str(warmup),
           "--out", str(out), "--scratch", str(scratch)]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if out.exists():
        rec = json.loads(out.read_text("utf-8"))
    else:
        rec = {"device_index": device_index, "model_output_equivalence": UNMEASURED,
               "refusals": [f"worker produced no result (exit {proc.returncode}): "
                            f"{proc.stderr.strip()[-800:]}"]}
    if counters.exists():
        try:
            rec["counters"] = json.loads(counters.read_text("utf-8"))
        except json.JSONDecodeError:
            rec["counters"] = None
    rec["memory_configuration"] = staging_label(rec.get("counters"))
    _derive(rec)
    rec["worker_stderr_tail"] = proc.stderr.strip()[-2000:] or None
    return rec


def _derive(rec: dict) -> None:
    """Compute everything that needs the counters file, which only exists after teardown.

    The island count is taken from the EP's own ``subgraphs_live`` counter and never from the
    claimed-node count. They happen to be equal on this artifact — every claimed node is its own
    single-node island — but *equal* and *the same number* are different statements, and the
    first fabricated number in this project came from treating an agreement as a definition.
    """
    counters = rec.get("counters") or {}
    islands = counters.get("subgraphs_live")
    rec["island_count"] = islands
    rec["dispatch_accounting"] = dispatch_accounting(
        counters, islands, rec.get("vk_inference_count") or 0)

    if rec.get("refusals") or not rec.get("vulkan"):
        return

    rec["boundary"] = boundary_cost(rec["vulkan"]["median_ms"], rec["cpu"]["median_ms"], islands)
    claimed = rec.get("claimed_nodes")
    rec["metric_of_record"] = {
        "claimed_op_coverage": claimed,
        "nodes_probed": rec.get("total_nodes_probed"),
        "island_count": islands,
        "island_count_source": "EP counter `subgraphs_live`",
        "islands_equal_claimed_nodes": (islands == claimed) if islands is not None else None,
        "islands_equal_claimed_nodes_note":
            "two independent counters agreeing that every claimed node is its own single-node "
            "island. Per R9 (DESIGN.md §10.0.1) agreement raises confidence, not evidence; the "
            "falsifier is `dispatch_accounting`, which goes red if the islands did not all run.",
        "largest_island_flops": None,
        "largest_island_flops_note":
            "not emitted by the EP (rust/src/trace.rs PartitionStats is unfilled). Recorded as "
            "null rather than guessed. Note that with every island a single node, the largest "
            "island is one MatMulNBits — the metric is not currently discriminating.",
        "gated_on": rec.get("model_output_equivalence"),
    }


def baseline_disagreement(results: "list[dict]", factor: float = 2.0) -> "str | None":
    """Warn when the CPU baseline moved between workers.

    Each worker times its Vulkan session and a CPU-only session back to back in one process, so
    each device's *delta* is internally consistent. But the CPU baselines from two workers are
    measured minutes apart on a machine that has just paged a 2.2 GB model, and if they disagree
    by more than ``factor`` then the machine's state changed between them. The per-device deltas
    remain valid; what is not valid is reading anything into how they compare — which the
    cross-device refusal already forbids, and this says out loud rather than leaving to be
    inferred.
    """
    medians = [(r.get("device_index"), (r.get("cpu") or {}).get("median_ms"))
               for r in results if r.get("cpu")]
    medians = [(i, m) for i, m in medians if m]
    if len(medians) < 2:
        return None
    lo, hi = min(m for _, m in medians), max(m for _, m in medians)
    if hi / lo < factor:
        return None
    detail = ", ".join(f"device {i}: {m:.1f} ms" for i, m in medians)
    return (
        f"the CPU-only baseline differs by {hi / lo:.1f}x between workers ({detail}). Same CPU, "
        f"same artifact, same feeds — so the machine's state changed between the runs. Each "
        f"device's own vulkan-vs-cpu delta was measured back to back in one process and stands; "
        f"nothing may be read across the two runs."
    )


def repeat_spread(records: "list[dict]") -> dict:
    """Summarise the run-to-run spread of whole-process repeats.

    Within-run spread (MAD, rsd) answers "how much does one iteration differ from the next in
    this process". It says nothing about the question a reader actually has, which is "if you
    ran this again tomorrow, what would you get". Those differ here by an order of magnitude:
    with a ten-iteration warmup the within-run rsd on the discrete part is a few percent, while
    two whole runs minutes apart produced medians of 2029 ms and 1462 ms — a 28% shift that no
    within-run statistic can see, because each run is internally perfectly consistent.

    So repeats are a separate instrument with a separate number, and the spread reported to a
    reader is the *larger* of the two. Reporting only the within-run figure would be the same
    error as quoting a tight standard error from a biased estimator.
    """
    med = [r["vulkan"]["median_ms"] for r in records if r.get("vulkan")]
    cpu = [r["cpu"]["median_ms"] for r in records if r.get("cpu")]
    out: dict = {"repeats": len(med)}
    if not med:
        out["usable"] = False
        return out
    out["usable"] = True
    out["vulkan_medians_ms"] = [round(x, 3) for x in med]
    out["cpu_medians_ms"] = [round(x, 3) for x in cpu]
    out["vulkan_median_of_medians_ms"] = round(statistics.median(med), 3)
    out["cpu_median_of_medians_ms"] = round(statistics.median(cpu), 3) if cpu else None
    out["vulkan_run_to_run_ratio"] = round(max(med) / min(med), 4)
    out["cpu_run_to_run_ratio"] = round(max(cpu) / min(cpu), 4) if len(cpu) > 1 else None
    if len(med) < 3:
        out["note"] = ("fewer than three repeats: the run-to-run spread is not established, "
                       "only bounded below by whatever these runs happened to differ by")
    return out


def _merge_repeats(records: "list[dict]") -> dict:
    """Fold N repeats of one device into a single record, keeping the widest honest spread."""
    usable = [r for r in records if r.get("vulkan") and not r.get("refusals")]
    base = dict(usable[-1] if usable else records[-1])
    base["repeat_records"] = [
        {k: r.get(k) for k in ("vulkan", "cpu", "vulkan_drift", "cpu_drift", "refusals",
                               "model_output_equivalence", "claimed_nodes")}
        for r in records
    ]
    base["repeat_spread"] = repeat_spread(usable)
    if usable and len(usable) > 1:
        spread = base["repeat_spread"]
        base["vulkan"] = dict(base["vulkan"])
        base["vulkan"]["median_ms"] = spread["vulkan_median_of_medians_ms"]
        base["cpu"] = dict(base["cpu"])
        base["cpu"]["median_ms"] = spread["cpu_median_of_medians_ms"]
        base["boundary"] = boundary_cost(base["vulkan"]["median_ms"], base["cpu"]["median_ms"],
                                         base.get("island_count"))
    return base


def _describe(rec: dict, facts: "device_mod.DeviceFacts | None") -> "list[str]":
    lines: "list[str]" = []
    name = facts.name if facts else f"device {rec.get('device_index')}"
    lines.append("")
    lines.append(f"### {name} (index {rec.get('device_index')})")
    if facts:
        lines.append(f"  driver {facts.driver_name} {facts.driver_version}  "
                     f"api {facts.api_version}  "
                     f"transfer class: {'UMA' if facts.uma else 'discrete'}  "
                     f"timestampPeriod {facts.timestamp_period_ns} ns/tick  "
                     f"validBits {facts.timestamp_valid_bits}")
    lines.append(f"  model_output_equivalence : {rec.get('model_output_equivalence')}")
    lines.append(f"  claimed nodes            : {rec.get('claimed_nodes')} of "
                 f"{rec.get('total_nodes_probed')} probed")
    lines.append(f"  islands (subgraphs_live) : {rec.get('island_count')}")
    da = rec.get("dispatch_accounting") or {}
    mark = {True: "ok", False: "RED", None: "n/a"}[da.get("consistent")]
    lines.append(f"  dispatch accounting      : {mark} — {da.get('detail')}")
    cfg = rec.get("memory_configuration") or {}
    lines.append(f"  memory configuration     : {cfg.get('configuration')} "
                 f"(basis: {cfg.get('basis')})")
    if rec.get("refusals"):
        for r in rec["refusals"]:
            lines.append(f"  ⛔ REFUSED: {r}")
        lines.append("  → no timing is reported for this device.")
        return lines
    vk, cpu, b = rec.get("vulkan"), rec.get("cpu"), rec.get("boundary") or {}
    lines.append(f"  vulkan   median {vk['median_ms']:.3f} ms   "
                 f"p05-p95 {vk['p05_ms']:.3f}-{vk['p95_ms']:.3f}   "
                 f"mad {vk['mad_ms']:.3f}   rsd {vk['rsd']:.1%}   n={vk['n']}"
                 f"{'   ** NOISY **' if vk['noisy'] else ''}")
    lines.append(f"  cpu-only median {cpu['median_ms']:.3f} ms   "
                 f"p05-p95 {cpu['p05_ms']:.3f}-{cpu['p95_ms']:.3f}   "
                 f"mad {cpu['mad_ms']:.3f}   rsd {cpu['rsd']:.1%}   n={cpu['n']}"
                 f"{'   ** NOISY **' if cpu['noisy'] else ''}")
    delta = b.get("delta_ms", 0.0)
    verb = "slower" if delta > 0 else "faster"
    lines.append(f"  vulkan is {abs(delta):.3f} ms {verb} than CPU-only on this artifact "
                 f"({(rec['vulkan']['median_ms'] / rec['cpu']['median_ms']):.1f}x)")
    for tag in ("vulkan", "cpu"):
        d = rec.get(f"{tag}_drift") or {}
        if d.get("steady") is False:
            lines.append(f"  ⚠ {tag} sample is NOT steady: {d['reason']}")
    rs = rec.get("repeat_spread") or {}
    if rs.get("usable") and rs.get("repeats", 0) > 1:
        lines.append(f"  run-to-run ({rs['repeats']} whole-process repeats): vulkan medians "
                     f"{rs['vulkan_medians_ms']} ms (spread {rs['vulkan_run_to_run_ratio']:.2f}x), "
                     f"cpu {rs['cpu_medians_ms']} ms")
        lines.append("  the run-to-run spread is wider than the within-run spread and is the one "
                     "a reader should carry.")
        if rs.get("note"):
            lines.append(f"  ⚠ {rs['note']}")
    per = b.get("per_island_ms_lower_bound")
    if per is not None:
        lines.append(f"  amortised over {b['island_count']} islands: {per * 1000:.1f} us per "
                     f"island boundary (LOWER bound — {b['why_lower']})")
    else:
        lines.append("  no per-island figure: the island count is unavailable, and dividing by "
                     "an assumed one is how a plausible-but-wrong number gets made.")
    lines.append(f"  {cfg.get('note', '')}")
    lines.append("  no GPU kernel time is included: no VkQueryPool exists yet, so every number "
                 "above is host wall time.")
    return lines


def main(argv: "list[str] | None" = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--worker" in argv:
        return _worker_main(argv)

    ap = argparse.ArgumentParser(description="Gated Phi-3.5 model benchmark.")
    ap.add_argument("--device", type=int, action="append",
                    help="device index; repeat for several. Default: every gated device.")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=10,
                    help="Iterations discarded before timing. Ten, not three: the Intel part "
                         "takes roughly six inferences to reach steady state and it gets "
                         "*slower*, not faster, on the way there (see stats.drift).")
    ap.add_argument("--repeats", type=int, default=3,
                    help="Whole-process repeats per device. Within-run spread cannot see "
                         "run-to-run variation; on this machine that variation is the larger "
                         "of the two.")
    ap.add_argument("--out", help="write the full JSON record here")
    a = ap.parse_args(argv)

    facts, source = device_mod.probe()
    by_index = {f.index: f for f in facts}
    indices = a.device if a.device else sorted(by_index) or [0]

    results = []
    for i in indices:
        reps = [_run_device(i, a.iters, a.warmup, _HERE / "_scratch")
                for _ in range(max(1, a.repeats))]
        results.append(_merge_repeats(reps))

    print("=" * 78)
    print("Phi-3.5 model benchmark — DESIGN.md §10.0 gated")
    print("=" * 78)
    print(f"device facts from: {source}")
    try:
        print(f"artifact         : {model_path()}")
    except ArtifactUnavailable as exc:
        print(f"artifact         : UNAVAILABLE — {exc}")
    print(f"producer         : {phi35_producer().summary()}")
    for rec, idx in zip(results, indices):
        for line in _describe(rec, by_index.get(idx)):
            print(line)

    print("")
    print("-" * 78)
    warn = baseline_disagreement(results)
    if warn:
        print(f"⚠ {warn}")
        print("")
    if len(results) > 1:
        print("The two devices are NOT compared. They differ in transfer class (UMA vs discrete),")
        print("shared-memory budget and timestampPeriod; a single figure spanning them would")
        print("describe neither. See bench/compare.py's cross-device refusal.")
    print("Nothing above is a speedup claim. It is one configuration (staging-bound, every")
    print("claimed node its own island, no GPU kernel time) of one artifact at one")
    print("producer-at-version.")

    payload = {
        "kind": "phi35",
        "environment": environment.capture(),
        "devices": device_mod.capture(),
        "producer": phi35_producer().to_dict(),
        "baseline_disagreement": warn,
        "results": results,
    }
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(payload, indent=2), "utf-8")
        print(f"wrote {a.out}")
    return 0 if any(not r.get("refusals") for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
