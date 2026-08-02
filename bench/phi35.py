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
* GPU kernel time **is** measured, as of the ``VkQueryPool`` path landing
  (``rust/src/vk/timestamp.rs``; ``GpuQueryPool::cmd_before/cmd_after`` around every
  ``vkCmdDispatch``). This module reports the phase split next to the wall time — see
  :mod:`phases`. The previous caveat here, *"no GPU kernel time is included: no VkQueryPool
  exists yet"*, was **retired on 2026-07-30** the day it stopped being true. It is recorded
  rather than deleted because a stale caveat is the defect class this project produced five
  times in one day: a true statement about the instrument set, kept past the point where the
  instrument set changed, is read by every later reader as current.
* The phase split comes from a **separate traced pass**, never from the timed run. Timestamp
  queries are pipeline-stage writes inside the command buffer plus a query-pool reset per
  recording, and the tracer itself allocates and takes a clock reading per span. Reporting a
  wall time measured with the instrument switched on, as if it were the wall time without it,
  is the same error one level up. The overhead is measured and reported as
  ``tracing_overhead_ratio`` rather than assumed small.

Usage::

    python bench/phi35.py --device 0 --iters 20 --out bench/results/phi35-dev0.json
    python bench/phi35.py --all-devices --iters 20 --out bench/results/phi35.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import devices as device_mod  # noqa: E402
import admissible  # noqa: E402
import contention  # noqa: E402
import device_companion as device_state  # noqa: E402
import environment  # noqa: E402
import phases as phases_mod  # noqa: E402
import producers  # noqa: E402
import stats as stats_mod  # noqa: E402
from stats import Sample  # noqa: E402

EP_NAME = "VulkanExecutionProvider"
EP_LIB_ENV = "ONNXRUNTIME_VULKAN_EP_LIB"
CLAIM_LOG_ENV = "ONNXRUNTIME_EP_VULKAN_CLAIM_LOG"
COUNTERS_ENV = "ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE"
TRACE_ENV = "ONNXRUNTIME_EP_VULKAN_TRACE"
TRACE_GPU_ENV = "ONNXRUNTIME_EP_VULKAN_TRACE_GPU"
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


#: Tank's `rust/tools/probe_broken_commitment.py`. Imported, never re-implemented — see
#: :func:`_decode_child_stream`.
_RUST_TOOLS = _HERE.parent / "rust" / "tools"


def _tank_decoder():
    """Return Tank's ``decode_both``, or ``None`` if his module is not importable.

    **Why an import and not four lines of my own.** The worker's stderr is a single OS handle
    that two writers share: our own narrow Python lines, and ORT's default logging sink, which
    on Windows writes **UTF-16LE**. Reading it as UTF-8 raises ``UnicodeDecodeError`` in
    ``subprocess``'s reader thread — which is how this harness died on 2026-08-01, at
    ``0xa7`` in position 1006 — and reading it as UTF-16LE alone loses our own lines and any
    wide line whose alignment our odd-length line shifted by a byte.

    Tank already solved this channel and wrote down what each naive version got wrong: his
    ``decode_both`` reads the same bytes four ways. R12 is the reason to borrow rather than
    rewrite: his instrument and mine are each correct about a different world (his channel is
    UTF-16LE, mine assumed UTF-8), and the repair for that is *one* description of the channel,
    not two. A second decoder here would be a second dialect for the same stream — the thing
    Link refused to create for the verdict vocabulary — and the two would drift.

    Returning ``None`` rather than falling back to a private decode is deliberate: with no
    decoder the tail is unreadable, and an unreadable tail is ``ERROR(instrument)`` under R13,
    not a quietly mangled string that later reads as evidence.
    """
    if str(_RUST_TOOLS) not in sys.path:
        sys.path.insert(0, str(_RUST_TOOLS))
    try:
        from probe_broken_commitment import decode_both  # type: ignore
    except Exception:  # pragma: no cover - environment dependent
        return None
    return decode_both


def _decode_child_stream(raw: "bytes | str | None") -> str:
    """Decode one captured child stream. Never raises, never returns ``None``.

    ``None`` is a real value here — a stream that was not captured, or a ``subprocess.run`` that
    never got far enough to fill one in — and the ``AttributeError`` at the old line 722 was this
    function's absence: ``proc.stderr.strip()`` on ``None``. A harness that dies on its own error
    path cannot report the error it found, so every caller below goes through here.
    """
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    decode_both = _tank_decoder()
    if decode_both is None:
        return (f"ERROR(instrument=stream_decoder): {len(raw)} bytes were captured and not "
                f"decoded. rust/tools/probe_broken_commitment.py::decode_both is not importable "
                f"from {_RUST_TOOLS}, and this harness deliberately has no second decoder for a "
                f"channel that is already described there.")
    try:
        return decode_both(raw)
    except Exception as exc:  # pragma: no cover - defensive; decode_both uses errors="replace"
        return f"ERROR(instrument=stream_decoder): decode_both raised {exc!r} on {len(raw)} bytes"


def _stream_tail(raw: "bytes | str | None", limit: int) -> str:
    """The last ``limit`` characters of a decoded stream, or ``""``. Never raises."""
    return _decode_child_stream(raw).strip()[-limit:]


def _run_worker(cmd: "list[str]", env: dict,
                on_start=None) -> "tuple[subprocess.CompletedProcess | None, str]":
    """Run a worker with **bytes** capture. Returns ``(proc, instrument_error)``.

    ``text=True`` is the defect: it hands the decode to ``subprocess``'s reader thread, where a
    failure surfaces as a traceback out of ``buffer.append(fh.read())`` — a crash in the
    measuring apparatus wearing the costume of a measurement failure (R13). Bytes cannot fail to
    be read. Whatever the child wrote is decoded later, by an instrument that is allowed to say
    it could not read it.

    ``on_start`` is called with the child's PID **as soon as it exists and before it is waited
    on**, because a companion that only learns our PID after the child has exited cannot tell our
    own worker from a stranger. That is not a hypothetical: wiring it the other way made the
    device-state companion report ``FOREIGN_GPU_WORK`` in 93% of samples against a single PID
    holding 0.0 MiB — our own worker — on a run with nothing else on the board. A detector that
    fires on every run is not a detector, it is a constant.
    """
    try:
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as exc:  # pragma: no cover - environment dependent
        return None, (f"ERROR(instrument=worker_capture): the worker subprocess could not be run "
                      f"or captured: {exc!r}. This is the harness failing, not a property of the "
                      f"EP: no verdict about the run may be drawn from it.")
    if on_start is not None:
        try:
            on_start(proc.pid)
        except Exception:
            pass
    try:
        out, err = proc.communicate()
    except Exception as exc:  # pragma: no cover - environment dependent
        proc.kill()
        proc.communicate()
        return None, (f"ERROR(instrument=worker_capture): the worker's output could not be "
                      f"collected: {exc!r}. This is the harness failing, not a property of the "
                      f"EP: no verdict about the run may be drawn from it.")
    return subprocess.CompletedProcess(cmd, proc.returncode, out, err), ""


def _note_instrument_error(rec: dict, message: str) -> None:
    """Record an ``ERROR(instrument=...)`` on a result record. **Never** a refusal.

    R13: a refusal is a detection — a condition the harness found and named. An instrument error
    is the harness not having looked. They are stored in different fields so that no reader, and
    no later aggregation, can count one as the other.
    """
    if not message:
        return
    rec.setdefault("instrument_errors", []).append(message)
    rec["instrument_error_note"] = (
        "R13: these are failures of this harness, not detections about the EP. They are not "
        "refusals and must never be counted as findings; a run carrying one has an unmeasured "
        "property, not a failing one.")


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
    """The host-side delta, and an explicit retirement of the per-island figure derived from it.

    **``per_island_ms_lower_bound`` is retired as of 2026-07-30.** It was defined as::

        per_island_ms = (vk_median_ms - cpu_median_ms) / island_count

    and described as a lower bound on the cost of one device-boundary crossing. The phase split
    (:mod:`phases`) shows that is not what it measures and never was. The delta is dominated by
    host-side work inside ``Compute`` — command-buffer recording and, inside that, the staging
    memcpy — which is a **per-inference, per-byte** cost, not a per-crossing one.

    The falsification was clean and is worth recording, because the statistic did not merely fail
    to be useful — it moved the wrong way. When Mouse wired ``partition.rs`` into
    ``GetCapability``, islands collapsed 321 → 33 and the Intel wall time fell 2954.6 → 807.2 ms:
    a large real improvement. The per-island figure went **up**, from 8.5 to 16.6 ms, because the
    denominator fell faster than a numerator that was never proportional to it. A statistic that
    rises when the thing it purports to measure improves is not a noisy statistic; it is measuring
    something else.

    What replaces it is not a redefinition of the same quotient. It is two directly measured
    numbers from the trace, neither of which is an attribution of a residual:

    * ``phases.host_phases_ms['record']`` — the per-island ``Record`` phase. **Corrected
      2026-07-30: this is not "the recording cost".** The span brackets the host staging memcpy,
      which is 98.6% of it, so the entry now carries ``is_leaf: false`` and the leaf residual is
      ``leaf_ms`` / ``record_scaling['command_construction_ms']``. Quoting the total under the name
      "recording" is the mistake this project made twice; ``phases.phase_leaf_accounting`` goes red
      on it. See ``docs/PERF.md`` §11.2.
    * ``phases.record_scaling`` — whether that cost scales with island size at all, asked of the
      upload-free residual rather than of the total, because a bigger island uploads more bytes and
      would otherwise "scale" for a reason unrelated to ``vkCmd*`` calls.

    ``delta_over_island_count_ms`` is still emitted, because deleting a number that has been
    quoted leaves a reader unable to reconcile older reports. It carries ``is_per_island_cost:
    false`` and its own note, and nothing downstream prints it as a per-island cost.
    """
    delta = vk_median_ms - cpu_median_ms
    quotient = (delta / island_count) if island_count else None
    return {
        "vk_median_ms": vk_median_ms,
        "cpu_median_ms": cpu_median_ms,
        "delta_ms": delta,
        "island_count": island_count,
        "island_count_source": "EP counter `subgraphs_live`",
        "per_island_ms_lower_bound": None,
        "per_island_ms_lower_bound_status": "RETIRED 2026-07-30",
        "per_island_ms_lower_bound_retirement_reason":
            "it was total-host-delta over island count, and the phase split shows the delta is "
            "dominated by per-inference host work (command-buffer recording, and inside it the "
            "staging memcpy) that is not proportional to island count. It rose 8.5 -> 16.6 ms "
            "(Intel) across a change that collapsed islands 321 -> 33 and cut wall time "
            "2954.6 -> 807.2 ms. Replaced by the directly measured vulkan.record median and by "
            "phases.record_scaling.",
        "delta_over_island_count_ms": quotient,
        "delta_over_island_count_is_per_island_cost": False,
        "delta_over_island_count_note":
            "an arithmetic restatement of delta_ms, kept only so older reports can be "
            "reconciled. It is NOT a per-island or per-boundary cost and must not be quoted as "
            "one.",
        "separating_instrument": "bench/phases.py, over the EP's own Chrome Trace JSON "
                                 "(vulkan.record / submit / fence_wait / vulkan.gpu.*)",
    }


def ratio_refusal(vk_drift: "dict | None", cpu_drift: "dict | None") -> "str | None":
    """Refuse to emit a vulkan-vs-cpu ratio when either sample is not steady.

    A ratio has two operands and inherits the worse of them. On the last recorded run the Vulkan
    absolutes were solid while the CPU baseline drifted 872.8 → 331.5 ms within a single process
    (rsd 14.4% Intel, 56.5% NVIDIA), and the harness printed the ratio anyway with a warning
    above it. **A warned-about number still gets quoted** — it travels into a summary, then into a
    document, and the warning does not travel with it. This project has already lost two numbers
    that way.

    So the ratio is not emitted at all when ``stats.drift`` says either sample is unsteady. The
    absolutes still are: each is a real measurement of its own session and neither depends on the
    other. What is withheld is the derived quantity whose meaning depends on both being stable.
    ``None`` means "emit it".
    """
    for tag, d in (("vulkan", vk_drift), ("cpu", cpu_drift)):
        if not d:
            continue
        if d.get("steady") is False:
            return (
                f"no vulkan/cpu ratio is reported: the {tag} sample is not steady "
                f"({d.get('reason')}). A ratio inherits the instability of its worse operand, "
                f"and a ratio printed under a warning is quoted without the warning. The two "
                f"absolutes above stand on their own and are unaffected.")
        if d.get("steady") is None:
            return (
                f"no vulkan/cpu ratio is reported: steadiness of the {tag} sample could not be "
                f"tested ({d.get('reason')}). Untested is not steady.")
    return None


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
    t_all = time.perf_counter()
    for _ in range(warmup):
        sess.run(None, feeds)
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(None, feeds)
        samples.append((time.perf_counter() - t0) * 1000.0)
    s = Sample(name=name, samples=samples)
    # The wall time of the whole warmup+timed loop, from a clock that knows nothing about phases.
    # This is the independent whole the trace's decomposition is checked against (R11): the trace
    # covers exactly these inferences, and every span it contains happened inside this interval.
    s.loop_wall_ms = (time.perf_counter() - t_all) * 1000.0
    return s


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
    # The timed pass runs with tracing OFF. Timestamp queries are pipeline-stage writes plus a
    # query-pool reset per recording, and every span costs an allocation and a clock read; a wall
    # time measured through the instrument is not the wall time without it.
    env.pop(TRACE_ENV, None)
    env.pop(TRACE_GPU_ENV, None)
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--device",
           str(device_index), "--iters", str(iters), "--warmup", str(warmup),
           "--out", str(out), "--scratch", str(scratch)]
    mon = contention.Monitor().start()
    proc, capture_error = _run_worker(cmd, env)
    window = mon.stop()
    if out.exists():
        rec = json.loads(out.read_text("utf-8"))
    else:
        returncode = proc.returncode if proc is not None else "unknown"
        tail = _stream_tail(proc.stderr if proc is not None else None, 800)
        rec = {"device_index": device_index, "model_output_equivalence": UNMEASURED,
               "refusals": [f"worker produced no result (exit {returncode}): "
                            f"{tail or '<no stderr captured>'}"]}
    _note_instrument_error(rec, capture_error)
    # The load survey covers the worker's whole lifetime, and the worker is a child of this
    # process, so its own CPU is subtracted rather than counted as competition.
    rec["machine_quiescence"] = contention.quiescence(window, contention.occupancy_check())
    if counters.exists():
        try:
            rec["counters"] = json.loads(counters.read_text("utf-8"))
        except json.JSONDecodeError:
            rec["counters"] = None
    rec["memory_configuration"] = staging_label(rec.get("counters"))
    _derive(rec)
    rec["worker_stderr_tail"] = _stream_tail(proc.stderr if proc is not None else None, 2000) or None
    return rec


def _is_integrated(facts: "device_mod.DeviceFacts | None") -> bool:
    """Does this device share its power budget with the CPU?

    Only used to weaken — never to strengthen — the GPU control in
    :func:`phases.contention_signature`. On an integrated part, heavy CPU load can slow the
    device too, so "GPU time moved as well" stops being an exoneration.
    """
    if facts is None:
        return False
    kind = str(getattr(facts, "kind", "") or "").lower()
    return "integrated" in kind or "cpu" in kind or "virtual" in kind


def _run_trace_pass(device_index: int, iters: int, warmup: int, scratch: Path,
                    timed: dict, integrated: bool = False) -> dict:
    """A second, instrumented process whose only product is the phase split.

    Separate from the timed pass on purpose. What comes back is *where the time goes*, in
    proportions; the absolute totals of this pass are inflated by the instrument and are labelled
    as such. ``tracing_overhead_ratio`` is the traced median over the untimed median — measured,
    not assumed small, because if the instrument doubled the run then the proportions it reports
    are proportions of a different run.
    """
    scratch.mkdir(parents=True, exist_ok=True)
    out = scratch / f"phi35_trace_dev{device_index}.json"
    counters = scratch / f"phi35_trace_counters_dev{device_index}.json"
    trace = scratch / f"phi35_trace_dev{device_index}.trace.json"
    for p in (out, counters, trace):
        p.unlink(missing_ok=True)
    env = dict(os.environ)
    env[COUNTERS_ENV] = str(counters)
    env[TRACE_ENV] = str(trace)
    env[TRACE_GPU_ENV] = "1"
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--device",
           str(device_index), "--iters", str(iters), "--warmup", str(warmup),
           "--out", str(out), "--scratch", str(scratch)]
    mon = contention.Monitor().start()
    # R9 rule 5. The device-state companion samples the *same window* as the trace, because a
    # record taken at another time is a record about another run. Without it the GPU steady tail
    # comes back UNCERTIFIED, and that is the intended behaviour rather than a degraded mode.
    companion = device_state.Companion(board_index=device_index).start()
    proc, capture_error = _run_worker(cmd, env, on_start=companion.own_root)
    dev_state = companion.stop()
    window = mon.stop()
    rep: dict = {
        "iters": iters,
        "warmup": warmup,
        "trace_file": str(trace),
        "machine_quiescence": contention.quiescence(window, contention.occupancy_check()),
        "device_state": dev_state,
        "device_state_note": ("the tenancy verdict and SM-clock record for this window. It is a "
                              "required companion, not a diagnostic: `gpu_steady_tail` is a "
                              "variance test over a suffix and cannot see a bias, and it has "
                              "reported STEADY at 10.99x and 21.4x wrong -- both times with a "
                              "BETTER RSD than the correct run. See bench/device_companion.py."),
        "note": ("a separate instrumented process. Proportions are the product; the absolute "
                 "totals here are inflated by the tracer and the query pool and are not the "
                 "benchmark's numbers."),
    }
    _note_instrument_error(rep, capture_error)
    if not trace.exists():
        returncode = proc.returncode if proc is not None else "unknown"
        tail = _stream_tail(proc.stderr if proc is not None else None, 400)
        rep["refusal"] = (f"no trace file was written (worker exit {returncode}). No phase "
                          f"split is reported: {tail or '<no stderr captured>'}")
        return rep
    traced_rec = json.loads(out.read_text("utf-8")) if out.exists() else {}
    cnt = None
    if counters.exists():
        try:
            cnt = json.loads(counters.read_text("utf-8"))
        except json.JSONDecodeError:
            cnt = None
    try:
        # R11: the decomposition is only publishable if it can be checked against a whole from an
        # instrument that is not the tracer. The worker's perf_counter around the warmup+timed
        # loop is that instrument -- it knows nothing about phases and cannot be inflated by the
        # same bug that would inflate a phase sum.
        whole = (traced_rec.get("vulkan") or {}).get("loop_wall_ms")
        rep["analysis"] = phases_mod.analyse(
            phases_mod.load(trace), cnt, integrated_gpu=integrated,
            independent_whole_ms=whole,
            whole_source=("the traced worker's own perf_counter around the warmup+timed run loop"
                          if whole else ""),
            device_state=dev_state)
    except Exception as exc:  # pragma: no cover - environment dependent
        rep["refusal"] = f"the trace could not be analysed: {exc!r}"
        return rep
    rep["red_flags"] = phases_mod.red_flags(rep["analysis"])
    tv = (traced_rec.get("vulkan") or {}).get("median_ms")
    uv = (timed.get("vulkan") or {}).get("median_ms")
    rep["traced_vulkan_median_ms"] = tv
    rep["untraced_vulkan_median_ms"] = uv
    rep["tracing_overhead_ratio"] = round(tv / uv, 4) if (tv and uv) else None
    rep["tracing_overhead_note"] = (
        "traced median / untraced median, both from this machine minutes apart. Above ~1.5 the "
        "phase proportions describe a run the benchmark did not measure and should be read as "
        "indicative only."
        if rep["tracing_overhead_ratio"] else
        "not computable: one of the two passes produced no median.")
    rep["traced_model_output_equivalence"] = traced_rec.get("model_output_equivalence")
    return rep


def measurement_validity(rec: dict) -> dict:
    """Two independent contention instruments, combined into one gate on the numbers.

    ``bench/contention.py`` watches the machine from **outside** the process: how many cores
    other processes kept busy while the worker ran. ``phases.contention_signature`` reads it from
    **inside** the trace: whether identical repeated work took wildly different amounts of host
    time while the device's own clock said it did not. They share no inputs — one is a system
    idle counter, the other is a Vulkan query pool — so agreement raises confidence and either
    one going red is evidence (R9, DESIGN.md §10.0.1).

    The out-of-band survey is the primary gate because it covers the *timed* pass, which is the
    pass the published number comes from. The in-band signature covers only the separate traced
    pass, so it cannot by itself condemn the timed number — but it is the only instrument that
    works on a trace captured before any of this existed, and every stored number in
    ``docs/PERF.md`` depends on it.

    Verdict is the worst of the two. ``QUIET`` requires the survey to say so *and* the in-band
    signature not to contradict it.
    """
    survey = rec.get("machine_quiescence") or {}
    sig = (((rec.get("phase_pass") or {}).get("analysis") or {})
           .get("contention_signature") or {})
    sig_pass = (rec.get("phase_pass") or {}).get("machine_quiescence") or {}

    reasons: "list[str]" = []
    verdict = survey.get("verdict") or contention.UNMEASURED
    if verdict != contention.QUIET:
        reasons.append(f"timed pass: out-of-band load survey says {verdict} — "
                       + "; ".join(survey.get("reasons") or ["no detail"]))
    if sig_pass.get("verdict") and sig_pass["verdict"] != contention.QUIET:
        reasons.append(f"traced pass: out-of-band load survey says {sig_pass['verdict']}")
    sv = sig.get("verdict")
    if sv and sv != "STABLE":
        reasons.append(f"traced pass: in-band trace signature says {sv} — {sig.get('reason')}")
        if verdict == contention.QUIET:
            # The survey saw a quiet machine and the trace saw stalls anyway. That is two
            # instruments disagreeing, which is not a tie to break in favour of the convenient
            # answer; something slowed the host that the survey does not account for.
            verdict = contention.UNMEASURED
    return {
        "verdict": verdict,
        "reasons": reasons or ["both contention instruments agree the machine was quiet"],
        "out_of_band_survey": survey.get("verdict"),
        "in_band_trace_signature": sv,
        "refusal": contention.gate({"verdict": verdict, "reasons": reasons},
                                   "phi-3.5 timing"),
    }


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
    # The ratio is a derived quantity gated on both operands being steady. See ratio_refusal.
    refusal = ratio_refusal(rec.get("vulkan_drift"), rec.get("cpu_drift"))
    rec["ratio_refusal"] = refusal
    rec["vulkan_over_cpu_ratio"] = (
        None if refusal or not rec["cpu"]["median_ms"]
        else round(rec["vulkan"]["median_ms"] / rec["cpu"]["median_ms"], 4))
    claimed = rec.get("claimed_nodes")
    validity = measurement_validity(rec)
    rec["measurement_validity"] = validity
    ps = ((rec.get("phase_pass") or {}).get("analysis") or {}).get("partition_stats") or {}
    rec["metric_of_record"] = {
        "claimed_op_coverage": claimed,
        "nodes_probed": rec.get("total_nodes_probed"),
        "island_count": islands,
        "island_count_source": "EP counter `subgraphs_live`",
        "islands_equal_claimed_nodes": (islands == claimed) if islands is not None else None,
        "islands_equal_claimed_nodes_note":
            "two independent counters. Per R9 (DESIGN.md §10.0.1) agreement raises confidence, "
            "not evidence; the falsifier is `dispatch_accounting`, which goes red if the islands "
            "did not all run.",
        "largest_island_flops": None,
        "largest_island_flops_emitted_value": ps.get("largest_island_flops"),
        "largest_island_flops_state": ps.get("third_slot_state", "NOT_OBSERVED"),
        "largest_island_flops_note":
            "the EP now *emits* the key on its `vulkan.getcapability` trace event, but emits 0: "
            "`CoverageReport` computes no FLOP estimate, and `PartitionStats.island_count` is 0 "
            "on the same event while `subgraphs_live` is not. The slot is plumbed and unfilled. "
            "It is reported as null here rather than as 0, because 'not computed' and 'zero "
            "FLOPs' are different states and only one of them is a measurement.",
        "gated_on": rec.get("model_output_equivalence"),
        "machine_quiescence": validity["verdict"],
        "quiescence_note":
            "a second gate, alongside model_output_equivalence. The same device, build and test "
            "measured 9.5x apart on host recording time depending only on what else was running "
            "on the machine, so a timing taken on a contended box is not a slower measurement of "
            "the same thing — it is a measurement of a different machine.",
    }


def baseline_disagreement(results: "list[dict]",
                          factor: "float | None" = None) -> "str | None":
    """Warn when the CPU baseline moved between workers.

    Each worker times its Vulkan session and a CPU-only session back to back in one process, so
    each device's *delta* is internally consistent. But the CPU baselines from two workers are
    measured minutes apart on a machine that has just paged a 2.2 GB model, and if they disagree
    by more than ``factor`` then the machine's state changed between them. The per-device deltas
    remain valid; what is not valid is reading anything into how they compare — which the
    cross-device refusal already forbids, and this says out loud rather than leaving to be
    inferred.

    The threshold is :data:`admissible.BASELINE_TOL`, **imported and not restated**. This function
    used to carry its own ``factor=2.0`` while ``admissible.baseline_comparability`` used
    ``tol=0.25``: one claim, two constants, eight-fold apart. The 2026-07-31 run's CPU baseline
    moved 291.8 → 228.7 ms (1.276×) between its two device passes and this printed nothing, while
    the same movement is ``BASELINE_MOVED`` by the other file's rule. The looser constant wins
    wherever it happens to be the one that runs, which makes the strict one decorative.
    """
    if factor is None:
        factor = 1.0 + admissible.BASELINE_TOL
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
        f"the CPU-only baseline differs by {hi / lo:.3f}x between workers ({detail}), beyond the "
        f"{factor:.2f}x this project allows a control to move. Same CPU, same artifact, same "
        f"feeds — so the machine's state changed between the runs, and it changed by more than "
        f"most of the effects we are trying to measure. Each device's own vulkan-vs-cpu delta was "
        f"measured back to back in one process and stands; nothing may be read across the two "
        f"runs."
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


def _resolve_device_identity(rec: dict, facts: "list[device_mod.DeviceFacts]") -> dict:
    """Name the device from the trace's own timestamp fingerprint, not from the index.

    ``ep.device_index`` indexes a **best-first** list (``engine.rs::probe_devices``); ``probe()``
    and ``vulkaninfo`` return **enumeration** order. On a laptop with an iGPU and a dGPU those two
    orderings are reversed, so labelling a row with ``probe()[ep_index]`` prints the wrong GPU's
    name, driver, transfer class and ``timestampPeriod`` over the numbers. This resolves the name
    from the trace instead and records whether the two agreed.
    """
    idx = rec.get("device_index")
    fp = (((rec.get("phase_pass") or {}).get("analysis") or {})
          .get("device_fingerprint") or {})
    chk = device_mod.device_identity_check(
        facts, idx, fp.get("timestamp_period_ns"), fp.get("timestamp_valid_bits"))
    device = chk.pop("device", None)
    rec["device_identity"] = chk
    return {"check": chk, "device": device}


def _describe(rec: dict, facts: "device_mod.DeviceFacts | None") -> "list[str]":
    lines: "list[str]" = []
    ident = rec.get("device_identity") or {}
    if ident.get("verdict") == "UNVERIFIED":
        # A plausible wrong name is worse than no name. Two orderings of the same two devices
        # exist on this machine; without the trace's fingerprint we cannot say which one ran.
        name = f"UNIDENTIFIED DEVICE (ep.device_index={rec.get('device_index')})"
        facts = None
    else:
        name = facts.name if facts else f"device {rec.get('device_index')}"
    lines.append("")
    lines.append(f"### {name} (ep.device_index {rec.get('device_index')})")
    if ident:
        mark = {True: "ok", False: "RED", None: "UNVERIFIED"}[ident.get("ok")]
        lines.append(f"  device identity [{mark}]      : {ident.get('detail')}")
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
    mv = rec.get("measurement_validity") or {}
    lines.append(f"  machine_quiescence       : {mv.get('verdict')} "
                 f"(survey {mv.get('out_of_band_survey')}, "
                 f"trace signature {mv.get('in_band_trace_signature')})")
    for r in mv.get("reasons") or []:
        lines.append(f"      {r}")
    if mv.get("refusal"):
        # Withheld, not annotated. Printing the medians under a contention warning is exactly
        # how the 9.5x-inflated figure would enter a document: the number travels, the warning
        # does not. What is still printed below is structural — counts, identity, accounting —
        # which contention cannot corrupt.
        lines.append(f"  ⛔ {mv['refusal']}")
        lines.append(f"      withheld: vulkan median, cpu median, their delta and ratio "
                     f"({vk['n']} + {cpu['n']} samples were collected and are in the JSON "
                     f"record under `vulkan`/`cpu`, marked non-quotable).")
        lines.append("      re-run when the machine is quiet: "
                     "`python bench/contention.py --seconds 20` must exit 0 first.")
        return lines + _describe_structure(rec)
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
    lines.append(f"  vulkan is {abs(delta):.3f} ms {verb} than CPU-only on this artifact")
    if rec.get("vulkan_over_cpu_ratio") is not None:
        lines.append(f"  vulkan / cpu ratio       : {rec['vulkan_over_cpu_ratio']:.2f}x")
    else:
        lines.append(f"  ⛔ {rec.get('ratio_refusal')}")
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
    lines.append(f"  {b.get('per_island_ms_lower_bound_status')}: per-island boundary cost — "
                 f"{b.get('per_island_ms_lower_bound_retirement_reason')}")
    lines.append(f"  {cfg.get('note', '')}")

    pp = rec.get("phase_pass") or {}
    if pp.get("refusal"):
        lines.append(f"  ⛔ no phase split: {pp['refusal']}")
    elif pp.get("analysis"):
        an = pp["analysis"]
        lines.append("")
        lines.append(f"  --- phase split (separate traced pass, {pp['iters']} iters; "
                     f"tracing overhead {pp.get('tracing_overhead_ratio')}x) ---")
        lines.extend(phases_mod.describe(an))
        sc = an.get("record_scaling") or {}
        if sc.get("usable"):
            lines.append(f"    record vs island size   : {sc.get('size_verdict')} — "
                         f"{sc.get('size_detail')}")
            if sc.get("size_confound"):
                lines.append(f"    confound                : {sc['size_confound']}")
            lines.append(f"    record across inferences: {sc.get('decline_verdict')} — "
                         f"{sc.get('decline_detail')}")
        ps = an.get("partition_stats") or {}
        lines.append(f"    largest_island_flops    : {ps.get('third_slot_state')} — "
                     f"{ps.get('third_slot_note') or 'populated'}")
        for f in pp.get("red_flags") or []:
            lines.append(f"    ⛔ {f}")
        if not pp.get("red_flags"):
            lines.append("    every falsifier that could go red did not: phase containment, GPU "
                         "containment, timestamp integrality, valid-bit mask, trace-vs-counters.")
    else:
        lines.append("  no phase split was requested (--no-phases).")
    return lines


def _describe_structure(rec: dict) -> "list[str]":
    """The part of a contended run that is still worth printing.

    Contention stretches durations. It does not change how many islands the partitioner made,
    how many dispatches ran, whether every GPU span was accounted for, or whether the timestamp
    conversion is arithmetically sound. Those are counts and integer identities, and they remain
    valid evidence from a run whose timings do not. Withholding them alongside the timings would
    throw away the falsifiers that cost the most to collect.
    """
    pp = rec.get("phase_pass") or {}
    an = pp.get("analysis") or {}
    if not an:
        return []
    out = ["", "  --- structural results from the traced pass (contention-independent) ---"]
    fs = an.get("falsifiers") or {}
    for name in ("gpu_span_accounting", "phase_containment", "trace_matches_counters",
                 "timestamp_conversion_integrality", "valid_bits_applied"):
        f = fs.get(name) or {}
        if not f:
            continue
        state = f.get("state")
        if state == "ERROR":
            mark = "ERROR"
        elif f.get("red"):
            mark = "RED"
        elif not f.get("decisive", True):
            mark = "VACUOUS"
        else:
            mark = "ok"
        out.append(f"    [{mark:^7}] {name}: "
                   f"{f.get('detail') or f.get('reason') or f.get('verdict')}")
    ps = an.get("partition_stats") or {}
    out.append(f"    largest_island_flops    : {ps.get('third_slot_state')} — "
               f"{ps.get('third_slot_note') or 'populated'}")
    cs = an.get("contention_signature") or {}
    if cs:
        out.append(f"    in-band trace signature : {cs.get('verdict')} — {cs.get('reason')}")
        for s in (cs.get("stalled_slots") or [])[:4]:
            out.append(f"        slot {s['slot']:2d} ({s['nodes']} dispatches): host "
                       f"{s['host_range_ratio']}x vs its own GPU {s['gpu_range_ratio']}x — "
                       f"{s['host_ms']} ms")
        if cs.get("integrated_gpu_caveat"):
            out.append(f"        caveat: {cs['integrated_gpu_caveat']}")
    out.append("    the phase split, record-scaling and warmup verdicts are WITHHELD: every one "
               "of them is a statement about durations, and durations are what contention "
               "moves.")
    return out


def _preserve_traces(results: "list[dict]", out: Path) -> None:
    """Copy each device's trace next to the result artifact it justifies.

    ``_run_trace_pass`` writes to a deterministic scratch path, so the next run on the same
    device silently destroys the evidence for the last published number. That happened: a
    three-iteration smoke test overwrote the trace behind a section of ``docs/PERF.md``, and the
    verdict for that device had to be transcribed by hand because it was no longer derivable.

    Traces are the expensive artifact here — a two-device run is forty minutes, and both defects
    found in the previous session were fixed by re-analysing stored traces with no re-run at all.
    Keeping them beside the JSON costs half a megabyte each.
    """
    dest = out.parent / "traces"
    for rec in results:
        src = ((rec.get("phase_pass") or {}).get("trace_file")) or ""
        if not src or not Path(src).is_file():
            continue
        target = dest / f"{out.stem}-dev{rec.get('device_index')}.trace.json"
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        rec["phase_pass"]["trace_preserved_at"] = str(target)
        print(f"preserved trace -> {target}")


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
    ap.add_argument("--trace-iters", type=int, default=5,
                    help="Iterations for the separate instrumented pass that produces the phase "
                         "split. Small on purpose: it answers a proportions question, and its "
                         "absolute totals are inflated by the instrument.")
    ap.add_argument("--no-phases", action="store_true",
                    help="Skip the traced pass. The wall time is then reported with no statement "
                         "about where it goes.")
    ap.add_argument("--out", help="write the full JSON record here")
    ap.add_argument("--require-quiet", action="store_true",
                    help="Refuse to start unless the machine is already quiet. A full run costs "
                         "~40 minutes and produces nothing quotable if another process is "
                         "compiling through it, so failing in ten seconds is cheaper than "
                         "failing in forty minutes.")
    ap.add_argument("--quiet-check-seconds", type=float, default=15.0,
                    help="How long --require-quiet samples the machine before deciding.")
    a = ap.parse_args(argv)

    if a.require_quiet:
        pre = contention.quiescence(
            contention.sample_now(a.quiet_check_seconds), contention.occupancy_check())
        print(contention.describe(pre))
        if pre["verdict"] != contention.QUIET:
            print("⛔ refusing to start: --require-quiet was given and the machine is not quiet.")
            print("   Nothing was measured. Quiesce the machine and re-run.")
            return 2

    facts, source = device_mod.probe()
    ep_order = device_mod.ep_selection_order(facts)
    indices = a.device if a.device else list(range(len(ep_order))) or [0]

    results = []
    identified: "list[device_mod.DeviceFacts | None]" = []
    for i in indices:
        reps = [_run_device(i, a.iters, a.warmup, _HERE / "_scratch")
                for _ in range(max(1, a.repeats))]
        merged = _merge_repeats(reps)
        if not a.no_phases and not merged.get("refusals"):
            cand = device_mod.by_ep_index(facts, i)
            merged["phase_pass"] = _run_trace_pass(
                i, a.trace_iters, a.warmup, _HERE / "_scratch", merged,
                integrated=_is_integrated(cand))
            _derive(merged)
        resolved = _resolve_device_identity(merged, facts)
        identified.append(resolved["device"] or device_mod.by_ep_index(facts, i))
        results.append(merged)

    print("=" * 78)
    print("Phi-3.5 model benchmark — DESIGN.md §10.0 gated")
    print("=" * 78)
    print(f"device facts from: {source}")
    print(f"ep.device_index order (best-first, engine.rs::probe_devices): "
          + ", ".join(f"{n}={d.name}" for n, d in enumerate(ep_order)))
    try:
        print(f"artifact         : {model_path()}")
    except ArtifactUnavailable as exc:
        print(f"artifact         : UNAVAILABLE — {exc}")
    print(f"producer         : {phi35_producer().summary()}")
    for rec, dev in zip(results, identified):
        for line in _describe(rec, dev):
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
    print("claimed node its own island) of one artifact at one producer-at-version. The phase")
    print("split comes from a separate instrumented pass; its proportions are the product and")
    print("its absolute totals are not the benchmark's numbers.")

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
        _preserve_traces(results, Path(a.out))
        Path(a.out).write_text(json.dumps(payload, indent=2), "utf-8")
        print(f"wrote {a.out}")
    return 0 if any(not r.get("refusals") for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
