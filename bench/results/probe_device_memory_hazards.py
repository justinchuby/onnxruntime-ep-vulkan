"""What is `ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY` still protecting against?

`BIND_OUTPUTS` ships ON since `d985240`. `DEVICE_MEMORY` ships OFF, so ORT does not allocate
fused-node outputs through this EP, `bind_target_for` declines all 195 of them
(`outputs_bind_attempted = 195`, `outputs_device_bound = 0`, measured), and **no user gets the
393,216 -> 0 B/past-token result, nor the ctx-4096 capability**. This probe exists to decide
whether that outer flag can move, by running the family of callers a *memory* flag exposes.

# The family, and why the lanes we already own do not cover it

Every lane this project has built is a caller who binds deliberately or declines deliberately.
`probe_default_bind_outputs.py` closed the "binds nothing" case for the inner flag. The case an
*allocator* flip exposes is different: **a caller who does something unusual with memory.** Four
of those, each of which has a named mechanism that could break it:

* `alloc_first`   — the allocator is asked for **before the session exists**. Measured earlier:
                    this builds a second `VkDevice` (`alloc_device_frame = SPLIT-DEVICE`) whose
                    buffers no dispatch can bind. The question is not whether the frame splits —
                    it does — but whether a split frame **returns different bytes**.
* `two_sessions`  — two sessions on one device, interleaved. One `HandleRegistry` is shared; a
                    span carved by one session is freed by the other's allocator release.
* `outlive`       — an `OrtValue` in our device memory read **after its session is gone**. The
                    allocator that owns the span is released at session teardown; the counters
                    `alloc_frees_after_release` and `alloc_failed_lookups` exist for exactly this.
* `budget`        — **the device allocation fails partway through the run.** First-class case, not
                    an edge case: the shipping path has been measured dying at ctx 4096 on an 8 GB
                    discrete GPU (`alloc failed for output buffer`, 0 dispatches). A device-memory
                    path that cannot allocate must *degrade*, and the degradation must be readable
                    off the counters rather than off a log nobody kept.

# How each lane is judged

Correctness first, counters second — the standing rule minted when a byte count in
`probe_kv_chain_phi35.py` was measuring the wrong tensor and only a bitwise comparison against an
identical computation caught it. Every lane runs the same two inferences on the same seeded
inputs as the `ship` lane (the shipping path, `DEVICE_MEMORY` unset) and **all 65 outputs are
compared byte for byte**, never an aggregate and never just `logits`.

`budget` additionally has to be seen *failing*: `ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY_BUDGET_MB`
caps the provider so `alloc_device_attach_failures` is forced positive. A guard never observed in
its positive state has no demonstrated positive state — so the run that shows the degradation is
paired with `armed`, the same lane uncapped, which must show `attach_attempts > 0` and
`attach_failures == 0`. Without that pair, "it degraded gracefully" is a sentence about a code
path nobody entered.

# What is deliberately not claimed

No wall-clock: the box is permanently contended (`PERF.md` §20). Counts, bytes, verdicts. The
device name is read off the run, never off the selector. `Nq/Nkv = 1.00` on Phi-3.5-mini, so a
grouping defect is invisible here; it is 4x on Llama-3 8B.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent

# ARCHIVAL: pinned to the exact Foundry cache layout this investigation measured against
# (the pre-2026-08-05 "...-cuda-gpu/cuda-int4-rtn-block-32/..." catalog revision, issue
# #11). Intentionally NOT auto-resolved against the live cache: a live resolver could
# silently pick a *different* cached revision than the one this result was measured
# against, which would misattribute a new run to an old artifact. Override PHI35_MODEL to
# replay against a different artifact explicitly (issue #19).
ONNX_FILE = pathlib.Path(
    os.environ.get(
        "PHI35_MODEL",
        r"C:\Users\justinchu\.foundry\cache\models\Microsoft"
        r"\Phi-3.5-mini-instruct-cuda-gpu\cuda-int4-rtn-block-32"
        r"\phi-3.5-mini-instruct-cuda-int4-rtn-block-32.onnx",
    )
)

# Result-identity contract (issue #19 follow-up, Morpheus review on PR #31): the resolved
# model path and its exact content hash are stamped into every output record below,
# computed lazily (only once the model has already been used successfully) so a
# PHI35_MODEL override or a stale/wrong cached file can never be silently absorbed into the
# evidence. Reuses the streaming SHA-256 helper `model_provenance.sha256_of` rather than a
# 23rd divergent hasher.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "rust" / "tools"))
import model_provenance as _model_provenance  # noqa: E402


def _result_identity() -> dict:
    return {
        "onnx_file": str(ONNX_FILE),
        "onnx_sha256": _model_provenance.sha256_of(ONNX_FILE),
    }
EP_NAME = "VulkanExecutionProvider"
COUNTERS_ENV = "ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE"

LAYERS = 32
KV_HEADS = 32
HEAD_DIM = 96
VOCAB = 32064
SEED_PAST = 4

# Lanes that arm the outer flag. `ship` is the shipping path and is the oracle for every one.
ARMED_LANES = ("armed", "alloc_first", "two_sessions", "outlive", "budget")
ALL_LANES = ("ship",) + ARMED_LANES

COUNTER_KEYS = (
    "compute_calls",
    "compute_failures",
    "device_losses",
    "dispatches_executed",
    "outputs_bind_attempted",
    "outputs_bind_declined",
    "outputs_device_bound",
    "outputs_host_resident",
    "alloc_allocations",
    "alloc_device_backed_spans",
    "alloc_staged_spans",
    "alloc_device_attach_attempts",
    "alloc_device_attach_failures",
    "alloc_device_attach_unavailable",
    "alloc_device_memory_budget_bytes",
    "alloc_device_frame",
    "alloc_device_frame_device",
    "alloc_frees_after_release",
    "alloc_failed_lookups",
    "alloc_device_download_bytes",
    "alloc_device_upload_bytes",
    "session_staging_readback_bytes",
    "session_staging_upload_bytes",
)


def _lib() -> str:
    return os.environ.get(
        "ONNXRUNTIME_VULKAN_EP_LIB",
        str(REPO / "rust" / "target" / "release" / "onnxruntime_vulkan_ep.dll"),
    )


def _counters(path: pathlib.Path) -> dict:
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return doc.get("counters", doc)


# --------------------------------------------------------------------------- worker


def _digest(arr) -> str:
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def _seed(np):
    rng = np.random.default_rng(20260803)
    past = {}
    for layer in range(LAYERS):
        for kind in ("key", "value"):
            past[f"past_key_values.{layer}.{kind}"] = (
                rng.standard_normal((1, KV_HEADS, SEED_PAST, HEAD_DIM)).astype(np.float16) * 0.02
            )
    return past


def _feeds(np, past, step):
    past_len = SEED_PAST + step
    feeds = {
        "input_ids": np.array([[1 + step]], dtype=np.int64),
        "attention_mask": np.ones((1, past_len + 1), dtype=np.int64),
    }
    feeds.update(past)
    return feeds


def _make_session(ort, providers):
    so = ort.SessionOptions()
    return ort.InferenceSession(
        str(ONNX_FILE),
        so,
        providers=providers,
        free_dimension_overrides_by_name={"batch_size": "1", "sequence_length": "1"},
    )


def _worker(lane: str, out_path: pathlib.Path) -> int:  # noqa: C901
    import numpy as np
    import onnxruntime as ort

    doc: dict = {"lane": lane, "ort_version": ort.__version__}
    counters_path = pathlib.Path(os.environ[COUNTERS_ENV]) if COUNTERS_ENV in os.environ else None
    if counters_path is not None:
        counters_path.unlink(missing_ok=True)

    def bail(why: str, verdict: str = "ERROR(instrument)") -> int:
        doc["verdict"] = verdict
        doc.setdefault("why", []).append(why)
        out_path.write_text(json.dumps({**doc, **_result_identity()}, indent=2), encoding="utf-8")
        return 2

    try:
        ort.register_execution_provider_library(EP_NAME, _lib())
    except Exception as exc:  # noqa: BLE001
        if "already registered" not in str(exc):
            raise
    ep_device = next((d for d in ort.get_ep_devices() if d.ep_name == EP_NAME), None)
    if ep_device is None:
        return bail("the Vulkan EP is not among ORT's EP devices")
    doc["ep_device"] = {
        k: ep_device.ep_metadata.get(k)
        for k in ("vulkan.device_name", "vulkan.device_index", "vulkan.vendor_id")
    }

    providers = [EP_NAME, "CPUExecutionProvider"]
    past = _seed(np)

    # ── the one lane that inverts the ordering ────────────────────────────────────────────────
    # Asking for the allocator BEFORE the session exists stands the device-memory provider up on
    # a VkDevice of its own. Everything about this lane is that one line being early.
    early_mi = None
    early_ov = None
    if lane == "alloc_first":
        early_mi = ep_device.memory_info(ort.OrtDeviceMemoryType.DEFAULT)
        if early_mi is None:
            return bail(
                "the EP advertised no DEFAULT allocator, so the outer flag is not armed and this "
                "lane never put its question"
            )
        # Force a real allocation through it, before any session: a memory_info that is merely
        # *asked for* may not stand the provider up, and a lane that did not build the second
        # device is not this lane.
        early_ov = ort.OrtValue.ortvalue_from_shape_and_type(
            [1, KV_HEADS, SEED_PAST, HEAD_DIM], np.float16, memory_info=early_mi
        )
        early_ov.update_inplace(past["past_key_values.0.key"])
        doc["allocated_before_session"] = True

    sess = _make_session(ort, providers)
    if EP_NAME not in sess.get_providers():
        return bail(f"{EP_NAME} absent from {sess.get_providers()}")
    names = [o.name for o in sess.get_outputs()]
    doc["output_count"] = len(names)
    logits_dtype = np.float16 if "float16" in sess.get_outputs()[0].type else np.float32
    doc["logits_dtype"] = str(np.dtype(logits_dtype))
    # One-element box so `_outlive_read` can drop the ONLY reference to the session while the
    # caller's device `OrtValue`s are still alive. A `del` of a parameter would drop a copy of the
    # reference and prove nothing — which is the same class of mistake as a control that shares a
    # binary with its subject.
    sess_box = [sess]
    del sess

    sess2 = None
    if lane == "two_sessions":
        # A second session on the same device, sharing one HandleRegistry. Built before either
        # runs, so both are live across both inferences and neither teardown is sequenced away.
        sess2 = _make_session(ort, providers)
        if EP_NAME not in sess2.get_providers():
            return bail(f"second session: {EP_NAME} absent from {sess2.get_providers()}")

    digests: dict[str, str] = {}
    logits_stats: list[dict] = []
    steps: list[dict] = []

    for step in range(2):
        feeds = _feeds(np, past, step)
        if lane == "two_sessions" and step == 1:
            # Interleave: step 1 runs on the OTHER session, fed the first session's outputs. If
            # the two sessions' allocators are not interchangeable, this is where it shows.
            outs = sess2.run(None, feeds)
        elif lane == "outlive":
            outs, ovs = _run_bound(
                ort, np, sess_box[0], ep_device, feeds, names, step, logits_dtype
            )
            doc.setdefault("bound_outputs", []).append(len(ovs))
        else:
            outs = sess_box[0].run(None, feeds)
        got = dict(zip(names, outs))
        for n in names:
            digests[f"step{step}/{n}"] = _digest(np.asarray(got[n]))
        flat = np.asarray(got["logits"], dtype=np.float64).reshape(-1)[-VOCAB:]
        logits_stats.append(
            {
                "step": step,
                "argmax": int(np.argmax(flat)),
                "nonzero": int(np.count_nonzero(flat)),
                "distinct": int(np.unique(flat).size),
                "max": float(flat.max()),
                "sum": float(flat.sum()),
            }
        )
        past = {
            f"past_key_values.{layer}.{kind}": np.asarray(got[f"present.{layer}.{kind}"])
            for layer in range(LAYERS)
            for kind in ("key", "value")
        }
        steps.append({"step": step, "counters": _counters(counters_path) if counters_path else {}})

    doc["digests"] = digests
    doc["logits_stats"] = logits_stats
    # Land the document BEFORE the risky part. `outlive` deliberately reads memory whose session
    # is gone; if that takes the process down, the evidence up to that point must survive, and a
    # crash then reads as a hazard rather than as a probe that produced nothing.
    out_path.write_text(json.dumps({**doc, **_result_identity()}, indent=2), encoding="utf-8")

    if lane == "outlive":
        rc = _outlive_read(ort, np, sess_box, ep_device, past, names, doc, out_path, logits_dtype)
        if rc:
            return rc

    sess_box[0] = None
    if sess2 is not None:
        del sess2
    if early_ov is not None:
        # Freed after the session that never owned it. The allocator this span came from is a
        # different one; `alloc_failed_lookups` is the counter that would notice.
        del early_ov
    final = _counters(counters_path) if counters_path else {}
    doc["final_counters"] = {k: final.get(k) for k in COUNTER_KEYS}
    doc["verdict"] = "LANE_RAN"
    out_path.write_text(json.dumps({**doc, **_result_identity()}, indent=2), encoding="utf-8")
    return 0


def _run_bound(ort, np, sess, ep_device, feeds, names, step, logits_dtype):
    """Run with all 65 outputs bound to device `OrtValue`s the caller keeps."""
    mi = ep_device.memory_info(ort.OrtDeviceMemoryType.DEFAULT)
    if mi is None:
        raise RuntimeError("no DEFAULT allocator: the outer flag is not armed")
    b = sess.io_binding()
    for k, v in feeds.items():
        b.bind_cpu_input(k, v)
    past_len = SEED_PAST + step
    ovs = {}
    for n in names:
        if n == "logits":
            ov = ort.OrtValue.ortvalue_from_shape_and_type(
                [1, 1, VOCAB], logits_dtype, memory_info=mi
            )
        else:
            ov = ort.OrtValue.ortvalue_from_shape_and_type(
                [1, KV_HEADS, past_len + 1, HEAD_DIM], np.float16, memory_info=mi
            )
        b.bind_ortvalue_output(n, ov)
        ovs[n] = ov
    sess.run_with_iobinding(b)
    return [ovs[n].numpy() for n in names], ovs


def _outlive_read(ort, np, sess_box, ep_device, past, names, doc, out_path, logits_dtype) -> int:
    """Read device `OrtValue`s after their session is gone.

    The span belongs to a `HandleRegistry` whose `VulkanAllocator` ORT releases at session
    teardown. `alloc_frees_after_release` and `alloc_failed_lookups` were written for this event;
    this is the first caller that produces it on purpose.
    """
    mi = ep_device.memory_info(ort.OrtDeviceMemoryType.DEFAULT)
    kept = {}
    sess = sess_box[0]
    b = sess.io_binding()
    # The extent comes off the tensors, not off a step counter. The first cut of this lane derived
    # it from the loop index, disagreed with the arrays by one token, and ORT refused the run —
    # a probe defect, and the fourth of its kind in this file's family. It is written this way so
    # the number cannot be re-derived wrongly.
    past_len = int(past["past_key_values.0.key"].shape[2])
    feeds = {
        "input_ids": np.array([[3]], dtype=np.int64),
        "attention_mask": np.ones((1, past_len + 1), dtype=np.int64),
    }
    feeds.update(past)
    for k, v in feeds.items():
        b.bind_cpu_input(k, v)
    for n in names:
        if n == "logits":
            ov = ort.OrtValue.ortvalue_from_shape_and_type(
                [1, 1, VOCAB], logits_dtype, memory_info=mi
            )
        else:
            ov = ort.OrtValue.ortvalue_from_shape_and_type(
                [1, KV_HEADS, past_len + 1, HEAD_DIM], np.float16, memory_info=mi
            )
        b.bind_ortvalue_output(n, ov)
        kept[n] = ov
    sess.run_with_iobinding(b)
    before = {n: _digest(kept[n].numpy()) for n in names}

    import gc

    del b
    del sess
    sess_box[0] = None
    gc.collect()
    doc["session_deleted_with_ortvalues_live"] = True
    out_path.write_text(json.dumps({**doc, **_result_identity()}, indent=2), encoding="utf-8")

    try:
        after = {n: _digest(kept[n].numpy()) for n in names}
    except Exception as exc:  # noqa: BLE001
        doc["outlive"] = {"read_after_teardown": "raised", "error": str(exc)[:400]}
        out_path.write_text(json.dumps({**doc, **_result_identity()}, indent=2), encoding="utf-8")
        return 0
    changed = [n for n in names if before[n] != after[n]]
    doc["outlive"] = {
        "read_after_teardown": "returned",
        "outputs": len(names),
        "changed_after_teardown": changed,
    }
    out_path.write_text(json.dumps({**doc, **_result_identity()}, indent=2), encoding="utf-8")
    return 0


# --------------------------------------------------------------------------- driver


def _run_lane(lane: str, budget_mb: int) -> dict:
    out = HERE / f"devmem_hazards_{lane}.json"
    counters = HERE / f"devmem_hazards_{lane}.counters.json"
    out.unlink(missing_ok=True)
    counters.unlink(missing_ok=True)
    env = dict(os.environ)
    env[COUNTERS_ENV] = str(counters)
    env["ONNXRUNTIME_VULKAN_EP_LIB"] = _lib()
    env.pop("ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY_BUDGET_MB", None)
    if lane == "ship":
        env.pop("ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY", None)
    else:
        env["ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY"] = "1"
    if lane == "budget":
        env["ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY_BUDGET_MB"] = str(budget_mb)
    # `BIND_OUTPUTS` is left UNSET everywhere: since d985240 the default is ON, and a lane that
    # asks for the default by name cannot tell you what a caller who asks for nothing gets.
    env.pop("ONNXRUNTIME_EP_VULKAN_BIND_OUTPUTS", None)
    cmd = [sys.executable, str(pathlib.Path(__file__).resolve()),
           "--worker", "--lane", lane, "--out", str(out)]
    proc = subprocess.run(cmd, env=env, capture_output=True)
    doc = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else {}
    doc["exit_code"] = proc.returncode
    if proc.returncode != 0:
        doc.setdefault("verdict", "LANE_DIED")
        doc["stderr_tail"] = (proc.stderr or b"").decode("utf-8", "replace")[-2000:]
    return doc


def _score(doc: dict, lanes: dict) -> int:  # noqa: C901
    why: list[str] = []
    ship = lanes.get("ship", {})
    if ship.get("exit_code") or not ship.get("digests"):
        doc["verdict"] = "ERROR(instrument)"
        doc["why"] = ["the `ship` oracle lane did not produce 65 outputs; nothing can be compared"]
        return 2

    # Degeneracy guard, read before anything else: an all-zero or constant output agrees with
    # everything, so agreement over a degenerate oracle is not agreement.
    st = ship.get("logits_stats") or []
    if not st or min(s["nonzero"] for s in st) < VOCAB // 2 or min(s["distinct"] for s in st) < 1000:
        doc["verdict"] = "ERROR(instrument)"
        doc["why"] = [f"the oracle's logits are degenerate: {st}"]
        return 2

    comparison: dict[str, dict] = {}
    separated: list[str] = []
    died: list[str] = []
    for lane in ARMED_LANES:
        d = lanes.get(lane)
        if not d:
            continue
        if d.get("exit_code"):
            died.append(lane)
            comparison[lane] = {"status": "DIED", "exit_code": d["exit_code"]}
            continue
        mine = d.get("digests") or {}
        keys = sorted(ship["digests"])
        missing = [k for k in keys if k not in mine]
        diff = [k for k in keys if k in mine and mine[k] != ship["digests"][k]]
        comparison[lane] = {
            "compared": len(keys) - len(missing),
            "identical": len(keys) - len(missing) - len(diff),
            "differing": diff[:8],
            "missing": missing[:8],
            "counters": d.get("final_counters", {}),
        }
        if diff or missing:
            separated.append(lane)

    doc["comparison"] = comparison

    # The budget lane's whole worth is that the failure it reports actually happened, and that the
    # same lane uncapped does not report it. Neither half alone is evidence.
    armed = (lanes.get("armed") or {}).get("final_counters") or {}
    budget = (lanes.get("budget") or {}).get("final_counters") or {}
    fault = {
        "armed_attach_attempts": armed.get("alloc_device_attach_attempts"),
        "armed_attach_failures": armed.get("alloc_device_attach_failures"),
        "budget_bytes": budget.get("alloc_device_memory_budget_bytes"),
        "budget_attach_attempts": budget.get("alloc_device_attach_attempts"),
        "budget_attach_failures": budget.get("alloc_device_attach_failures"),
        "budget_staged_spans": budget.get("alloc_staged_spans"),
        "budget_compute_failures": budget.get("compute_failures"),
        "budget_device_losses": budget.get("device_losses"),
    }
    doc["allocation_failure"] = fault
    if "budget" in lanes and "armed" in lanes and "budget" not in died and "armed" not in died:
        if not (fault["budget_attach_failures"] or 0) > 0:
            why.append(
                "the capped lane recorded zero attach failures, so the degradation path was never "
                "entered and 'it degrades' is a claim about code nobody ran"
            )
        if (fault["armed_attach_failures"] or 0) != 0:
            why.append(
                "the UNCAPPED lane also recorded attach failures, so the counter does not "
                "separate the injected fault from the ordinary run"
            )
        if not (fault["armed_attach_attempts"] or 0) > 0:
            why.append(
                "the uncapped armed lane never attempted a device attach, so this run says "
                "nothing about the device-memory path at all"
            )
        if (fault["budget_compute_failures"] or 0) != 0:
            why.append(
                f"the capped lane failed {fault['budget_compute_failures']} Compute call(s): the "
                "path DIED rather than degraded"
            )

    # Non-triviality: every armed lane must have executed dispatches on the EP, or its agreement
    # with `ship` is the agreement of two runs of the CPU EP.
    for lane in ARMED_LANES:
        if lane not in lanes or lane in died:
            continue
        c = (lanes.get(lane) or {}).get("final_counters") or {}
        if not (c.get("dispatches_executed") or 0) > 0:
            why.append(f"{lane}: dispatches_executed = 0, so the EP never ran")
        if (c.get("device_losses") or 0) != 0:
            why.append(f"{lane}: device_losses = {c.get('device_losses')}, verdict void")

    doc["why"] = why
    if died:
        doc["verdict"] = "HAZARD_LANE_DIED"
        doc["died"] = died
        return 1
    if separated:
        doc["verdict"] = "DEVICE_MEMORY_HAZARD_SEPARATES"
        doc["separated"] = separated
        return 1
    if why:
        doc["verdict"] = "ERROR(instrument)"
        return 2
    doc["verdict"] = "NO_HAZARD_LANE_SEPARATES"
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--lane", default="ship")
    ap.add_argument("--out", default="")
    ap.add_argument("--lanes", default=",".join(ALL_LANES))
    ap.add_argument("--budget-mb", type=int, default=8)
    ap.add_argument("--device", default=os.environ.get("ONNXRUNTIME_EP_VULKAN_DEVICE", "0"))
    ap.add_argument("--json", default=str(HERE / "device_memory_hazards.json"))
    args = ap.parse_args()

    # Selector only: which device actually ran is read off `vulkan.device_name` in every lane's
    # own document. `DEVICE=0` has run on `1=NVIDIA` on this box.
    os.environ["ONNXRUNTIME_EP_VULKAN_DEVICE"] = str(args.device)

    if args.worker:
        return _worker(args.lane, pathlib.Path(args.out))

    doc: dict = {
        "probe": "what is ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY still protecting against?",
        "model": str(ONNX_FILE),
        "lib": _lib(),
        "budget_mb": args.budget_mb,
    }
    if not ONNX_FILE.is_file():
        doc["verdict"] = "ERROR(instrument)"
        doc["why"] = [f"the model is missing at {ONNX_FILE}"]
        print(json.dumps(doc, indent=2))
        return 2

    lanes = {}
    for lane in args.lanes.split(","):
        lane = lane.strip()
        if lane:
            lanes[lane] = _run_lane(lane, args.budget_mb)
    doc["lanes"] = lanes
    doc["device"] = (lanes.get("ship") or {}).get("ep_device")
    rc = _score(doc, lanes)
    pathlib.Path(args.json).write_text(json.dumps({**doc, **_result_identity()}, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in doc.items() if k != "lanes"}, indent=2))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
