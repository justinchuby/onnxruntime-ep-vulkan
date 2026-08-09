"""Is the model actually faster now? Current `main` against the exact pre-GQA parent, one process
per (workload, arm, repeat), on real models.

WHY THIS EXISTS SEPARATELY FROM EVERY OTHER PROBE IN THIS DIRECTORY (issue #69)
==============================================================================
Every clocked real-model instrument this repository has compares **arms inside one build**:
`probe_real_model_latency.py` compares `vulkan_tiled` / `vulkan_untiled` / `cpu` from a single
`.dll`, and `probe_gqa_local_size.py` sweeps a specialisation constant inside a single `.dll`.
Both are the right shape for "does this knob pay". Neither can answer the question issue #69
actually asks, which is a question about **two builds**:

    is the model faster on the EP we ship today than on the EP we shipped before PR #72?

A knob sweep is not that answer. `ONNXRUNTIME_EP_VULKAN_GQA_LOCAL_SIZE=1` reproduces the
*geometry* of the old kernel inside the new binary, but it does not reproduce the old binary:
the shader source differs (a specialisation constant exists at all), the host rule differs, and
an in-build kill switch can only ever measure the difference the switch was built to make. The
only honest instrument for "before vs after" builds **both** libraries and runs both.

WHAT THAT COSTS, AND HOW IT IS PAID
-----------------------------------
ORT registers an execution-provider library **process-globally**, so one process can hold exactly
one EP `.dll`. Therefore:

* the unit of measurement is an **OS process**, one per `(workload, arm, repeat)`;
* the two arms cannot be compared inside one process, so cross-arm output agreement is
  established by **digest** — each process hashes the exact bytes of every output tensor it
  produced, and the digests are compared afterwards;
* every process independently establishes its own correctness against the CPU EP, in that same
  process, on the same session object that is then timed. Nothing is inherited.

WHAT MAKES A NUMBER ADMISSIBLE HERE
-----------------------------------
A latency is written into this artifact only when all of the following hold for that exact
process. Anything else is a **refusal**, and a refusal record carries no `speed` key at all —
not a null, not a zero, absent — so no reader and no downstream tool can average it in.

1.  **Correctness.** `real_model.classify_outputs` returns MATCH against a CPU EP reference run
    in the same process on the same feeds.
2.  **An outputs digest.** Present and non-empty. A missing digest is not "unknown", it is a
    refusal: it is the field that lets the two arms be shown to have computed the same thing.
3.  **A path witness.** The EP's own counters file, written from that process's exit hook,
    naming `pipeline_variants` — the specialisation vectors handed to `vkCreateComputePipelines`,
    recorded at dispatch time by `vk::session`, not by anything this file says.
4.  **The device ran it.** `compute_failures == 0`, the Vulkan EP is in the session's provider
    list, and at least one dispatch executed.
5.  **The model is the model.** The sha256 of the resolved graph (and of its external weights)
    equals the pinned or independently recorded digest.
6.  **The witness separates the arms, wherever a speedup is claimed.** The candidate's GQA
    dispatch passes a specialisation constant, so its pipeline key is `gqa_f16:<local>`; the
    baseline passes none, so its key is `gqa_f16:`. Two arms that produce the *same* witness on a
    workload claimed faster have not been shown to be two different code paths, and the claim is
    refused. On a workload claimed **neutral** an identical witness is corroboration, not a
    defect — which is why this rule is stated per-verdict rather than globally.

WHAT IT REFUSES TO CONCLUDE
---------------------------
This compares Vulkan to Vulkan. A gain here is **not** a CUDA win, and says nothing about any
other execution provider. The CPU EP appears only as the correctness reference; its latency is
recorded for context and is never the claim.

Usage::

    python bench/results/probe_crossbuild_gqa_landing.py \
        --candidate-lib <path to candidate onnxruntime_vulkan_ep.dll> \
        --baseline-lib  <path to baseline  onnxruntime_vulkan_ep.dll> \
        --candidate-sha 85fbda2... --baseline-sha c96e7d9... \
        --out real_model_crossbuild_gqa_landing.json
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
from pathlib import Path

_BENCH = Path(__file__).resolve().parents[1]
_ROOT = _BENCH.parent
for _p in (str(_BENCH), str(_ROOT), str(_ROOT / "rust" / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import real_model as rm  # noqa: E402

SCHEMA = "real_model_crossbuild_gqa_landing/1"
ISSUE = 69

EP_LIB_ENV = "ONNXRUNTIME_VULKAN_EP_LIB"
COUNTERS_ENV = "ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE"

#: Fixed by the pre-registration (embedded in the artifact under `preregistration`, sha256
#: `a17c39ce…69e63b`) before the first timed iteration. Not tunable here:
#: the whole point of a declared band is that it predates the numbers.
REPEATS = 3
WARMUPS = 5
ITERS = 20
#: The floor of the band. Widened at analysis time to the null control's own half-range if that
#: is larger — a rule, fixed in advance, never a number chosen after seeing a result.
BAND_FLOOR = 0.05

# --------------------------------------------------------------------------------------------
# MiniLM — pinned here, self-contained, because `real_model.MODELS` does not carry it on
# `origin/main` (issue #78 owns the shared pin and is not landed at the SHA under test). The pin
# is complete or it does not exist: repo + immutable commit + file + digest + size.
# --------------------------------------------------------------------------------------------

MINILM_KEY = "all-MiniLM-L6-v2-onnx"
MINILM_REPO = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MINILM_FILE = "onnx/model.onnx"
MINILM_SHA256 = "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452"
MINILM_CACHE_NAME = "all-MiniLM-L6-v2-onnx-1110a243.onnx"


def minilm_url() -> str:
    """Built from the validated triple at point of use; `/resolve/main/` is unconstructible."""
    return (f"https://huggingface.co/{MINILM_REPO}/resolve/"
            f"{MINILM_REVISION}/{MINILM_FILE}")


def resolve_minilm() -> dict:
    path = rm.repo_cache_dir() / MINILM_CACHE_NAME
    if not path.is_file():
        raise rm.ModelUnavailable(
            f"{MINILM_KEY}: {path.name} is absent from the model cache. Fetch it from "
            f"{minilm_url()} — the pinned revision, not `main`.")
    digest = rm.sha256_file(path)
    if digest != MINILM_SHA256:
        raise rm.ModelUnavailable(
            f"{MINILM_KEY}: cached bytes hash {digest}, the pin says {MINILM_SHA256}. "
            f"Refusing to benchmark substitute bytes under a pinned model's name.")
    return {
        "key": MINILM_KEY,
        "family": "encoder",
        "path": str(path),
        "sha256": digest,
        "bytes": path.stat().st_size,
        "provenance": "huggingface-immutable-pin",
        "resolver": "pinned-download",
        "recorded_sha256": MINILM_SHA256,
        "agrees_with_recorded_provenance": True,
        "source": {"repo": MINILM_REPO, "revision": MINILM_REVISION, "file": MINILM_FILE,
                   "url": minilm_url()},
        "external_data": {"scanned": True, "files": [],
                          "reason": "single-file export; the .onnx hash covers the weights"},
        "weights_bytes": 0,
        "note": "sentence-transformers/all-MiniLM-L6-v2 ONNX export at an immutable commit. "
                "Fragmentation control: a BERT encoder with no GroupQueryAttention, so the "
                "landed change cannot touch it and its arms must agree.",
    }


# --------------------------------------------------------------------------------------------
# Workloads
# --------------------------------------------------------------------------------------------

#: `(key, phase, m, past)`. Order is the order they are measured in.
WORKLOADS = (
    (rm.PHI35.key, "prefill", 1, 0),
    (rm.PHI35.key, "prefill", 32, 0),
    (rm.PHI35.key, "prefill", 64, 0),
    (rm.PHI35.key, "prefill", 128, 0),
    (rm.PHI35.key, "decode", 1, 128),
    (rm.PHI35.key, "decode", 1, 1024),
    (rm.MOBILENETV2.key, "batch", 1, 0),
    (rm.MOBILENETV2.key, "batch", 16, 0),
    (MINILM_KEY, "encode", 128, 0),
    (MINILM_KEY, "encode", 384, 0),
)

#: Workloads whose GQA dispatch geometry the candidate's rule actually changes. Derived from
#: `ops::attention::gqa_local_size` (largest power of two <= 64 leaving >= 32 workgroups) applied
#: to `B * Nq * S = 32 * S` — written out so the expectation is falsifiable by the witness rather
#: than asserted after the fact.
EXPECTED_GQA_LOCAL = {
    (rm.PHI35.key, "prefill", 1, 0): 1,
    (rm.PHI35.key, "prefill", 32, 0): 32,
    (rm.PHI35.key, "prefill", 64, 0): 64,
    (rm.PHI35.key, "prefill", 128, 0): 64,
    (rm.PHI35.key, "decode", 1, 128): 1,
    (rm.PHI35.key, "decode", 1, 1024): 1,
}


def workload_label(key: str, phase: str, m: int, past: int) -> str:
    if phase == "batch":
        return f"{key}/batch/N{m}"
    if phase == "encode":
        return f"{key}/encode/S{m}"
    return f"{key}/{phase}/M{m}/past{past}"


def make_case(key: str, phase: str, m: int, past: int) -> rm.Case:
    if phase == "batch":
        return rm.Case(key, "batch", m, 0, tokens=None, unit="images")
    if phase == "encode":
        return rm.Case(key, "encode", m, 0, tokens=m, unit="tokens")
    return rm.Case(key, phase, m, past,
                   tokens=(m if phase == "prefill" else 1), unit="tokens")


def minilm_feeds(case: rm.Case, np):
    """Its own int64 token feed. Never an image tensor, and never another model's feed builder.

    Deterministic: a fixed id range and an all-ones mask, so two processes feed identical bytes
    and `feeds_digest` can prove it. `token_type_ids` is all zeros — a single-segment sentence,
    which is what a sentence-transformer is for.
    """
    ids = (np.arange(1, case.m + 1, dtype=np.int64) % 30000).reshape(1, case.m)
    return {
        "input_ids": ids,
        "attention_mask": np.ones((1, case.m), dtype=np.int64),
        "token_type_ids": np.zeros((1, case.m), dtype=np.int64),
    }


def build_feeds(case: rm.Case, np):
    if case.model_key == MINILM_KEY:
        return minilm_feeds(case, np)
    return rm.build_feeds(case, np)


def resolve(key: str) -> dict:
    if key == MINILM_KEY:
        return resolve_minilm()
    return rm.resolve_model(rm.MODELS[key])


def outputs_digest(outputs, np) -> str:
    """sha256 over every output's dtype, shape and exact bytes, in order.

    This is what makes two *processes* — and therefore two builds — comparable at all. Shape and
    dtype are hashed alongside the bytes because a reinterpreted buffer of the same length is not
    the same tensor.
    """
    h = hashlib.sha256()
    for i, arr in enumerate(outputs):
        a = np.ascontiguousarray(arr)
        h.update(str(i).encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------------------------
# The worker: one (workload, arm, repeat), in its own process
# --------------------------------------------------------------------------------------------


def _session(ort, path: Path, providers, device_index: int):
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if rm.EP_NAME in providers:
        opts.add_session_config_entry("ep.device_index", str(device_index))
    return ort.InferenceSession(str(path), opts, providers=list(providers))


def worker(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--model", required=True)
    ap.add_argument("--phase", required=True)
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--past", type=int, default=0)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--repeat", type=int, required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--warmup", type=int, default=WARMUPS)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    label = workload_label(a.model, a.phase, a.m, a.past)
    rec: dict = {
        "workload": label, "arm": a.arm, "repeat": a.repeat,
        "model_key": a.model, "phase": a.phase, "m": a.m, "past": a.past,
        "pid": os.getpid(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    def bail(reason: str, **extra) -> int:
        """Write a refusal. Structurally free of speed fields: `speed` is never added."""
        rec["admissible"] = False
        rec["refusal"] = {"reason": reason, **extra}
        rec.pop("speed", None)
        Path(a.out).write_text(json.dumps(rec, indent=2), encoding="utf-8")
        return 3

    import numpy as np
    import onnxruntime as ort

    lib = os.environ.get(EP_LIB_ENV)
    if not lib or not Path(lib).is_file():
        return bail(f"{EP_LIB_ENV} unset or missing", lib=lib)
    rec["ep_library_sha256"] = rm.sha256_file(lib)
    try:
        ort.register_execution_provider_library(rm.EP_NAME, str(Path(lib).resolve()))
    except Exception as exc:
        if "already registered" not in str(exc):
            return bail(f"EP registration failed: {exc}")

    try:
        model_rec = resolve(a.model)
    except Exception as exc:
        return bail(f"model unresolvable: {exc}")
    rec["model"] = {k: v for k, v in model_rec.items() if k != "path"}
    rec["model_resolved_from"] = model_rec["resolver"]

    case = make_case(a.model, a.phase, a.m, a.past)
    feeds = build_feeds(case, np)
    rec["feeds_sha256"] = rm.feeds_digest(feeds)

    path = Path(model_rec["path"])

    # -- correctness first, in this process, on the session that is then timed ----------------
    try:
        ref_sess = _session(ort, path, (rm.CPU_EP,), a.device)
        t0 = time.perf_counter()
        reference = ref_sess.run(None, feeds)
        rec["cpu_reference_ms"] = (time.perf_counter() - t0) * 1000.0
        rec["cpu_reference_outputs_sha256"] = outputs_digest(reference, np)
        del ref_sess
    except Exception as exc:
        return bail(f"CPU reference failed: {exc}")

    try:
        t0 = time.perf_counter()
        sess = _session(ort, path, (rm.EP_NAME, rm.CPU_EP), a.device)
        build_ms = (time.perf_counter() - t0) * 1000.0
    except Exception as exc:
        return bail(f"Vulkan session build failed: {exc}")
    rec["providers"] = list(sess.get_providers())
    if rm.EP_NAME not in rec["providers"]:
        return bail("Vulkan EP absent from the session's provider list")

    try:
        t0 = time.perf_counter()
        got = sess.run(None, feeds)
        rec["first_run_ms"] = (time.perf_counter() - t0) * 1000.0
    except Exception as exc:
        return bail(f"Vulkan first run failed: {exc}")

    rec["session_build_ms"] = build_ms
    rec["outputs_sha256"] = outputs_digest(got, np)
    if not rec["outputs_sha256"]:
        return bail("empty outputs digest")
    equivalence = rm.classify_outputs(case, got, reference, np)
    rec["equivalence"] = equivalence
    if equivalence.get("verdict") != rm.MATCH:
        return bail("equivalence DIVERGENT", verdict=equivalence.get("verdict"))

    # -- timed pass: same session object, tracing off ----------------------------------------
    for _ in range(a.warmup):
        sess.run(None, feeds)
    samples = []
    for _ in range(a.iters):
        t = time.perf_counter()
        sess.run(None, feeds)
        samples.append((time.perf_counter() - t) * 1000.0)
    # The digest is taken again AFTER the timed pass: an arm that agreed on its verification run
    # and then drifted under repetition would otherwise pass on a verdict its timed iterations
    # never earned.
    post = sess.run(None, feeds)
    rec["outputs_sha256_post_timing"] = outputs_digest(post, np)
    if rec["outputs_sha256_post_timing"] != rec["outputs_sha256"]:
        return bail("outputs changed between the verification run and the timed pass",
                    before=rec["outputs_sha256"], after=rec["outputs_sha256_post_timing"])
    del sess

    rec["speed"] = {
        "samples_ms": [round(x, 4) for x in samples],
        **rm.latency_stats(samples),
        "throughput": rm.throughput(case, statistics.median(samples)),
    }
    rec["admissible"] = True
    rec["refusal"] = None
    rec["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    Path(a.out).write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return 0


# --------------------------------------------------------------------------------------------
# The parent
# --------------------------------------------------------------------------------------------


def gqa_witness(counters: "dict | None") -> dict:
    """Pull the `gqa_f16` pipeline key out of a counters file.

    `pipeline_variants` is written by `vk::session` at pipeline-creation time from the resolved
    specialisation vector, so it is a witness about what the device was handed, not about what
    this script believes. The candidate passes `vec![local]`; the baseline passes `vec![]`. Their
    keys are therefore `gqa_f16:<local>` and `gqa_f16:` and cannot be confused.
    """
    out = {"present": False, "gqa_keys": [], "local_size": None, "all_variants": [],
           "shaders_dispatched": []}
    if not counters:
        return out
    out["present"] = True
    variants = counters.get("pipeline_variants") or []
    out["all_variants"] = list(variants)
    out["shaders_dispatched"] = list(counters.get("shaders_dispatched") or [])
    keys = [v for v in variants if v.split(":")[0] == "gqa_f16"]
    out["gqa_keys"] = keys
    sizes = []
    for k in keys:
        tail = k.split(":", 1)[1]
        sizes.append(int(tail.split(",")[0]) if tail else None)
    out["local_size"] = sizes[0] if len(sizes) == 1 else sizes or None
    out["dispatches_executed"] = counters.get("dispatches_executed")
    out["compute_calls"] = counters.get("compute_calls")
    out["compute_failures"] = counters.get("compute_failures")
    out["claimed_nodes"] = counters.get("claimed_nodes")
    out["viable_islands_retained"] = counters.get("viable_islands_retained")
    out["running_device_names"] = counters.get("running_device_names")
    return out


def admissibility_gate(rec: dict) -> dict:
    """Re-derive admissibility from the record alone, and strip speed from anything refused.

    The worker already refuses in-process, but the worker is the thing being trusted. This is a
    second, *pure* pass over the written record: it re-checks the claims a speed number depends
    on — equivalence, an outputs digest, a model digest that agrees with the pin, a path witness,
    a device that actually ran something — using nothing but the record. Being pure is the point:
    it is the surface the mutation tests drive, and a mutated record cannot reach a timing.

    Returns the same dict, mutated in place. A refused record has no ``speed`` key at all.
    """
    def refuse(reason: str, **extra) -> dict:
        rec["admissible"] = False
        rec["refusal"] = {"reason": reason, **extra}
        rec.pop("speed", None)
        return rec

    if not rec.get("admissible"):
        rec.pop("speed", None)
        return rec

    eq = rec.get("equivalence")
    if not eq or not eq.get("verdict"):
        return refuse("no output equivalence verdict on this (arm, repeat)")
    if eq.get("verdict") != rm.MATCH:
        return refuse("equivalence not MATCH", verdict=eq.get("verdict"))

    if not rec.get("outputs_sha256"):
        return refuse("missing or empty outputs digest")
    if rec.get("outputs_sha256_post_timing") != rec.get("outputs_sha256"):
        return refuse("outputs digest changed across the timed pass")

    model = rec.get("model") or {}
    recorded = model.get("recorded_sha256")
    if not model.get("sha256"):
        return refuse("no model digest recorded for this process")
    if recorded and model["sha256"] != recorded:
        return refuse("model digest does not match the recorded/pinned provenance",
                      measured=model["sha256"], recorded=recorded)
    if model.get("agrees_with_recorded_provenance") is False:
        return refuse("model provenance disagreement reported by the resolver",
                      measured=model.get("sha256"), recorded=recorded)

    w = rec.get("path_witness") or {}
    if not w.get("present"):
        return refuse("no path witness: the EP wrote no counters file")
    if w.get("compute_failures"):
        return refuse("compute_failures > 0", compute_failures=w.get("compute_failures"))
    if not w.get("dispatches_executed"):
        return refuse("no dispatch executed on the device")

    return rec


def run_one(args, key, phase, m, past, arm, repeat, scratch: Path) -> dict:
    label = workload_label(key, phase, m, past)
    tag = f"{label}_{arm}_r{repeat}".replace("/", "_").replace(".", "_")
    out_path = scratch / f"rec_{tag}.json"
    counters_path = scratch / f"counters_{tag}.json"
    for p in (out_path, counters_path):
        p.unlink(missing_ok=True)

    env = dict(os.environ)
    env[EP_LIB_ENV] = args.candidate_lib if arm == "candidate" else args.baseline_lib
    env[COUNTERS_ENV] = str(counters_path)
    # Production defaults on both arms: no tuning variable is set by this driver, and any
    # inherited one is stripped so an operator's shell cannot silently become part of the result.
    for k in list(env):
        if k.startswith("ONNXRUNTIME_EP_VULKAN_") and k not in (COUNTERS_ENV,):
            env.pop(k)

    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker",
           "--model", key, "--phase", phase, "--m", str(m), "--past", str(past),
           "--arm", arm, "--repeat", str(repeat), "--device", str(args.device),
           "--iters", str(args.iters), "--warmup", str(args.warmup), "--out", str(out_path)]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    wall = time.perf_counter() - t0

    if out_path.exists():
        rec = json.loads(out_path.read_text(encoding="utf-8"))
    else:
        rec = {"workload": label, "arm": arm, "repeat": repeat, "admissible": False,
               "refusal": {"reason": f"worker wrote no record (exit {proc.returncode})",
                           "stderr": (proc.stderr or "")[-2000:]}}
    rec["worker_exit"] = proc.returncode
    rec["worker_wall_s"] = round(wall, 3)
    rec["ep_library_role"] = arm

    counters = None
    if counters_path.exists():
        try:
            counters = json.loads(counters_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            counters = None
    rec["path_witness"] = gqa_witness(counters)

    admissibility_gate(rec)

    assert rec.get("admissible") or "speed" not in rec, \
        "a refused record must not carry speed fields"
    return rec


def paired(records, label) -> dict:
    """Per-repeat baseline/candidate ratios, and the refusal accounting that gates them."""
    cand = {r["repeat"]: r for r in records
            if r["workload"] == label and r["arm"] == "candidate"}
    base = {r["repeat"]: r for r in records
            if r["workload"] == label and r["arm"] == "baseline"}
    refusals = [{"arm": r["arm"], "repeat": r["repeat"], "refusal": r["refusal"]}
                for r in records if r["workload"] == label and not r.get("admissible")]
    out = {"workload": label, "refusals": refusals, "repeats_paired": 0}
    ratios, rows = [], []
    for rep in sorted(set(cand) & set(base)):
        c, b = cand[rep], base[rep]
        if not (c.get("admissible") and b.get("admissible")):
            continue
        cm = c["speed"]["median_ms"]
        bm = b["speed"]["median_ms"]
        if not cm:
            continue
        ratios.append(bm / cm)
        rows.append({"repeat": rep, "candidate_median_ms": round(cm, 4),
                     "baseline_median_ms": round(bm, 4), "ratio": round(bm / cm, 6),
                     "candidate_rsd": round(c["speed"].get("rsd", 0), 5),
                     "baseline_rsd": round(b["speed"].get("rsd", 0), 5)})
    out["per_repeat"] = rows
    out["repeats_paired"] = len(ratios)
    if ratios:
        out["ratio_median"] = statistics.median(ratios)
        out["ratio_min"] = min(ratios)
        out["ratio_max"] = max(ratios)
        out["ratios"] = [round(r, 6) for r in ratios]
        out["candidate_median_ms"] = statistics.median(
            [r["candidate_median_ms"] for r in rows])
        out["baseline_median_ms"] = statistics.median(
            [r["baseline_median_ms"] for r in rows])
    # Cross-arm output agreement: two builds computing the same answer, by digest.
    digests = {"candidate": sorted({c["outputs_sha256"] for c in cand.values()
                                    if c.get("outputs_sha256")}),
               "baseline": sorted({b["outputs_sha256"] for b in base.values()
                                   if b.get("outputs_sha256")})}
    out["outputs_sha256"] = digests
    out["cross_arm_bitwise_identical"] = (
        bool(digests["candidate"]) and digests["candidate"] == digests["baseline"]
        and len(digests["candidate"]) == 1)
    # Witness separation: do the two arms demonstrably bind different pipelines?
    wit = {"candidate": sorted({tuple(c["path_witness"]["gqa_keys"]) for c in cand.values()}),
           "baseline": sorted({tuple(b["path_witness"]["gqa_keys"]) for b in base.values()})}
    out["gqa_pipeline_keys"] = {k: [list(t) for t in v] for k, v in wit.items()}
    out["witness_distinguishes_arms"] = wit["candidate"] != wit["baseline"]
    return out


def verdict_for(p: dict, band: float) -> str:
    if p.get("refusals"):
        return "REFUSED"
    if p.get("repeats_paired", 0) < REPEATS:
        return "REFUSED"
    med, lo, hi = p.get("ratio_median"), p.get("ratio_min"), p.get("ratio_max")
    if med is None:
        return "REFUSED"
    if lo > 1 + band:
        return "FASTER"
    if med < 1 - band:
        return "SLOWER"
    if (1 - band) <= med <= (1 + band) and (1 - 2 * band) <= lo and hi <= (1 + 2 * band):
        return "NEUTRAL"
    return "INDETERMINATE"


def environment_record(args, device) -> dict:
    import onnxruntime as ort

    return {
        "onnxruntime": ort.__version__,
        "python": sys.version.split()[0],
        "os": f"{platform.system()} {platform.release()} {platform.version()}",
        "machine": platform.machine(),
        "device": {
            "index": device.index, "name": device.name,
            "uuid": getattr(device, "uuid", None),
            "luid": getattr(device, "luid", None),
            "pci": getattr(device, "pci", None),
            "driver": getattr(device, "driver_version", None),
            "transfer_class": getattr(device, "transfer_class", None),
            "subgroup_size": getattr(device, "subgroup_size", None),
            "timestamp_period_ns": getattr(device, "timestamp_period", None),
            "max_compute_shared_memory": getattr(device, "max_compute_shared_memory", None),
        },
        "methodology": {
            "unit_of_measurement": "one OS process per (workload, arm, repeat)",
            "repeats": args.repeats, "iters_per_repeat": args.iters,
            "warmup_per_session": args.warmup, "warmups_discarded": True,
            "arm_order": "the two arms of one workload run adjacent; order flips per repeat",
            "tracing_during_timed_pass": False,
            "tuning_env_stripped": True,
            "equivalence": "CPU EP reference in the same process, on the same session object "
                           "that is then timed, for every (workload, arm, repeat)",
        },
        "power_and_affinity": {
            "assumption": "stock Windows power plan, no CPU affinity mask, no GPU clock lock",
            "thread_affinity_set": False, "gpu_clocks_locked": False,
            "note": "the box is shared indefinitely (PERF.md 20); arms are interleaved at the "
                    "workload level precisely because machine state cannot be assumed quiet",
        },
    }


def sanitize_exclusivity(record: dict) -> dict:
    """Strip the runtime channel out of the GPU-exclusivity proof before it goes public.

    The proof needs to say *that* the lock was held, how long, that nothing was killed, and what
    else was on the device — none of which requires an operator's home directory or the absolute
    path of a lock file. Process listings are reduced to executable base names; the lock path is
    reduced to its file name. What survives is auditable; what is dropped is somebody's disk.
    """
    out = dict(record)
    lock = out.get("lock_path")
    if lock:
        out["lock_path"] = Path(lock).name
        out["lock_path_note"] = ("session-local, deliberately outside every worktree; the "
                                 "directory is a runtime detail and is not published")
    for k in list(out):
        if not k.startswith("gpu_compute_apps"):
            continue
        names = []
        for line in out[k] or []:
            parts = [p.strip() for p in str(line).split(",")]
            exe = parts[1] if len(parts) > 1 else str(line)
            names.append(Path(exe).name if ("\\" in exe or "/" in exe) else exe)
        counted = {}
        for n in names:
            counted[n] = counted.get(n, 0) + 1
        out[k] = [f"{n} x{c}" if c > 1 else n for n, c in sorted(counted.items())]
    return out


def finalize(argv) -> int:
    """Stamp a finished artifact with the *released* exclusivity record, sanitized.

    The probe embeds the lock record as it stood at acquire time, because that is the only state
    that exists while the run is still going. This re-embeds the terminal record — held duration,
    release time, the device's process list at release — so the published proof covers the whole
    measurement window rather than its first instant. It touches nothing else: an assertion below
    fails if any measurement byte moves.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--finalize", required=True, help="artifact to stamp")
    ap.add_argument("--exclusivity", required=True, help="terminal (RELEASED) lock record")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    art = json.loads(Path(a.finalize).read_text(encoding="utf-8"))
    excl = json.loads(Path(a.exclusivity).read_text(encoding="utf-8"))
    if excl.get("state") != "RELEASED":
        print(f"[finalize] refusing: lock record state is {excl.get('state')!r}, not RELEASED",
              file=sys.stderr)
        return 2
    before = json.dumps({k: v for k, v in art.items() if k != "exclusivity"}, sort_keys=True)
    art["exclusivity"] = sanitize_exclusivity(excl)
    after = json.dumps({k: v for k, v in art.items() if k != "exclusivity"}, sort_keys=True)
    assert before == after, "finalize must not touch any measurement field"

    out = Path(a.out or a.finalize)
    out.write_text(json.dumps(art, indent=2), encoding="utf-8")
    print(f"[finalize] exclusivity stamped RELEASED (held {excl.get('held_seconds')}s) -> {out}")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--worker" in argv:
        return worker(argv)
    if "--finalize" in argv:
        return finalize(argv)

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate-lib", required=True)
    ap.add_argument("--baseline-lib", required=True)
    ap.add_argument("--candidate-sha", required=True)
    ap.add_argument("--baseline-sha", required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--warmup", type=int, default=WARMUPS)
    ap.add_argument("--out", default="real_model_crossbuild_gqa_landing.json")
    ap.add_argument("--runtime-out", default=None,
                    help="where the UNREDACTED runtime record goes; kept out of the public "
                         "artifact so absolute paths and the public channel stay separate")
    ap.add_argument("--scratch", default=str(_BENCH / "results" / "_crossbuild_scratch"))
    ap.add_argument("--exclusivity", default=None,
                    help="path to a JSON record proving exclusive GPU use was held")
    ap.add_argument("--prereg", default=None)
    ap.add_argument("--only", default=None, help="comma-separated workload substrings")
    args = ap.parse_args(argv)

    for name, p in (("candidate", args.candidate_lib), ("baseline", args.baseline_lib)):
        if not Path(p).is_file():
            print(f"[xbuild] refusing: {name} library {p} does not exist", file=sys.stderr)
            return 2
    if rm.sha256_file(args.candidate_lib) == rm.sha256_file(args.baseline_lib):
        print("[xbuild] refusing: the two arms are the same binary. A/B of one build is not "
              "an A/B.", file=sys.stderr)
        return 2

    import bench as bench_mod
    import devices as device_mod

    os.environ[EP_LIB_ENV] = args.candidate_lib
    if not bench_mod.register_ep():
        print("[xbuild] refusing: no Vulkan EP to measure.", file=sys.stderr)
        return 2
    facts, _src = device_mod.probe()
    device = bench_mod.select_device(facts, args.device)
    print(f"[xbuild] device {device.index}: {device.name}", flush=True)

    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)

    wanted = WORKLOADS
    if args.only:
        needles = [s for s in args.only.split(",") if s]
        wanted = tuple(w for w in WORKLOADS
                       if any(n in workload_label(*w) for n in needles))

    models_seen = {}
    unavailable = []
    for key in dict.fromkeys(w[0] for w in wanted):
        try:
            models_seen[key] = resolve(key)
        except Exception as exc:
            print(f"[xbuild] MODEL UNAVAILABLE: {key}: {exc}", file=sys.stderr)
            unavailable.append({"model": key, "error": str(exc)})

    records = []
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    for rep in range(args.repeats):
        arms = ("candidate", "baseline") if rep % 2 == 0 else ("baseline", "candidate")
        for (key, phase, m, past) in wanted:
            if key not in models_seen:
                continue
            for arm in arms:
                rec = run_one(args, key, phase, m, past, arm, rep, scratch)
                records.append(rec)
                label = rec["workload"]
                if rec.get("admissible"):
                    print(f"[xbuild] r{rep} {label:<52} {arm:<9} "
                          f"{rec['speed']['median_ms']:9.2f} ms  "
                          f"gqa={rec['path_witness']['gqa_keys']}", flush=True)
                else:
                    print(f"[xbuild] r{rep} {label:<52} {arm:<9} REFUSED: "
                          f"{rec['refusal']['reason']}", flush=True)

    labels = list(dict.fromkeys(r["workload"] for r in records))
    prelim = {lab: paired(records, lab) for lab in labels}

    # The band, resolved by the rule declared in advance: 5% or the null control's own
    # half-range, whichever is larger.
    null_label = workload_label(rm.PHI35.key, "prefill", 1, 0)
    null = prelim.get(null_label, {})
    null_half_range = None
    if null.get("ratio_min") and null.get("ratio_max"):
        null_half_range = max(abs(null["ratio_max"] - 1.0), abs(1.0 - null["ratio_min"]))
    band = max(BAND_FLOOR, null_half_range or 0.0)

    workloads_out = []
    for lab in labels:
        p = prelim[lab]
        p["verdict"] = verdict_for(p, band)
        expect = next((EXPECTED_GQA_LOCAL[w] for w in EXPECTED_GQA_LOCAL
                       if workload_label(*w) == lab), None)
        p["expected_candidate_gqa_local_size"] = expect
        if expect is not None:
            got = p["gqa_pipeline_keys"]["candidate"]
            p["candidate_gqa_local_size_as_expected"] = all(
                keys == [f"gqa_f16:{expect}"] for keys in got) if got else False
            p["baseline_gqa_unspecialised"] = all(
                keys == ["gqa_f16:"] for keys in p["gqa_pipeline_keys"]["baseline"]
            ) if p["gqa_pipeline_keys"]["baseline"] else False
        if p["verdict"] == "FASTER" and not p["witness_distinguishes_arms"]:
            p["verdict"] = "REFUSED"
            p.setdefault("refusals", []).append({
                "reason": "a workload claimed FASTER whose path witness does not distinguish "
                          "the two arms has not been shown to have run two different code paths"})
        workloads_out.append(p)

    public = {
        "schema": SCHEMA,
        "issue": ISSUE,
        "question": "Does current origin/main improve end-to-end real-model Vulkan wall-clock "
                    "latency versus the exact Vulkan parent immediately before the landed GQA "
                    "local-size optimisation (PR #72 / 0cfa362)?",
        "not_a_claim_about": "any other execution provider. This is Vulkan against prior "
                             "Vulkan. It is not a CUDA comparison and no CUDA number appears "
                             "in it.",
        "arms": {
            "candidate": {"commit": args.candidate_sha,
                          "ep_library_sha256": rm.sha256_file(args.candidate_lib),
                          "ep_library_bytes": Path(args.candidate_lib).stat().st_size,
                          "role": "origin/main at the time of measurement"},
            "baseline": {"commit": args.baseline_sha,
                         "ep_library_sha256": rm.sha256_file(args.baseline_lib),
                         "ep_library_bytes": Path(args.baseline_lib).stat().st_size,
                         "role": "first parent of 0cfa362 (PR #72) — the exact Vulkan tree "
                                 "immediately before the GQA local-size change"},
        },
        "band": {
            "floor": BAND_FLOOR,
            "null_control": null_label,
            "null_control_half_range": null_half_range,
            "applied": band,
            "rule": "max(5%, the null control's own half-range). Fixed before the run; the "
                    "null control is prefill M=1, where gqa_local_size(32) == 1 and the two "
                    "builds dispatch identical geometry.",
        },
        "started_at": started,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "environment": environment_record(args, device),
        "models": {k: {kk: vv for kk, vv in v.items() if kk != "path"}
                   for k, v in models_seen.items()},
        "unavailable_models": unavailable,
        "workloads": workloads_out,
        "records": [{k: v for k, v in r.items() if k != "model"} for r in records],
    }
    if args.exclusivity and Path(args.exclusivity).is_file():
        public["exclusivity"] = sanitize_exclusivity(
            json.loads(Path(args.exclusivity).read_text(encoding="utf-8")))
    if args.prereg and Path(args.prereg).is_file():
        public["preregistration"] = {
            "sha256": rm.sha256_file(args.prereg),
            "bytes": Path(args.prereg).stat().st_size,
            "text": Path(args.prereg).read_text(encoding="utf-8"),
        }

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (Path(__file__).with_name(args.out) if len(Path(args.out).parts) == 1
                    else Path(args.out).resolve())
    out_path.write_text(json.dumps(public, indent=2), encoding="utf-8")

    if args.runtime_out:
        runtime = {
            "note": "RUNTIME CHANNEL. Absolute paths live here and never in the public "
                    "artifact.",
            "candidate_lib": str(Path(args.candidate_lib).resolve()),
            "baseline_lib": str(Path(args.baseline_lib).resolve()),
            "model_paths": {k: v["path"] for k, v in models_seen.items()},
            "scratch": str(scratch.resolve()),
            "interpreter": sys.executable,
        }
        Path(args.runtime_out).write_text(json.dumps(runtime, indent=2), encoding="utf-8")

    print(f"\n  band applied: {band:.4f}")
    print(f"  {'workload':<52} {'cand ms':>9} {'base ms':>9} {'ratio':>7}  verdict")
    for p in workloads_out:
        if p.get("ratio_median"):
            print(f"  {p['workload'][:52]:<52} {p['candidate_median_ms']:>9.2f} "
                  f"{p['baseline_median_ms']:>9.2f} {p['ratio_median']:>6.3f}x  "
                  f"{p['verdict']}")
        else:
            print(f"  {p['workload'][:52]:<52} {'-':>9} {'-':>9} {'-':>7}  {p['verdict']}")
    print(f"\n  wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
