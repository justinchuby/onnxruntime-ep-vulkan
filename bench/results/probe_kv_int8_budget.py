"""What does an int8 KV cache cost, **in ULPs**, on the outputs criterion 10 already measures?

THE QUESTION, AND WHY IT COMES BEFORE A KERNEL
==============================================
The KV ledger's next lever is int8.  But int8 is a *correctness* change wearing a bandwidth
change's clothes, and this project has a specific reason to be careful about that here:
Trinity established that at the final RMSNorm **Vulkan is bit-exact against float64 while
ORT's CPU EP is the 1-ULP side**, and criterion 10 is open with three failing outputs at
**12 ULP on the logits and 4 ULP on layer 31's key and value**.  Quantising the KV cache
moves exactly those tensors.

So the error budget is established *before* a kernel exists, in the unit criterion 10 already
uses, with **Trinity's instrument** (``tests/ops/_models.ulp_residual`` plus the
median/p99/max/cancellation distribution her consumer records beside it).  Not a second one:
a second ULP instrument would be a second answer nobody could reconcile.

Every number this probe can produce was written down first, in
``bench/results/kv-int8-budget-prediction.md``.

WHAT IS SIMULATED, AND WHAT IS THEREFORE NOT CLAIMED
====================================================
The cache is quantised at the **host** boundary: each decode step's newly produced KV token is
round-tripped int8 (or int4) and fed back as the next step's ``past``.  That models the
*storage* error of an int8 cache **exactly** — a real kernel dequantises on read and
accumulates in fp32, which is what fp16 storage already does today (``gqa_f16.comp`` carries
``float acc[128]``).  It models **nothing** about a kernel's own rounding on write, nor any
change to accumulation order.

**So every residual here is a LOWER BOUND on a real int8 kernel's residual, never an estimate
of it.**  It is a budget: if int8 is unaffordable at the lower bound, no kernel makes it
affordable.

THE GRANULARITY LADDER IS THE EXPERIMENT
========================================
ULP is scale-free *per element*; quantisation noise is scale-free *per group*.  An element well
below its group's max pays the group's absolute error against its own much finer spacing, so
the whole budget is the ratio ``max|x| / |x|`` within a group.  Granularity is defined on the
newest token's slice of one layer's K or V, ``(1, 32 kv_heads, 1, 96)`` = 3072 values, because
that is what a real kernel quantises once and keeps:

    per_tensor    1 scale   / 3072 values
    per_head     32 scales  /   96 values   <- the per-head-group scale of the general case
    per_block32  96 scales  /   32 values   <- matches this model's own block-32 weight convention

``Nq/Nkv = 1.00`` on Phi-3.5 and **4x on Llama-3**, so ``per_head`` here is a per-head-group
scale over a group of one query head.  Nothing in this probe exercises a non-unit grouping and
no number in it may be quoted for a 4x model.  Said in the verdict, not only here.

NO TOLERANCE IS CHOSEN HERE, DELIBERATELY
=========================================
A criterion may not be hardened because it is about to pass, nor narrowed because it has just
failed — and the mirror of that rule forbids picking a tolerance that the result under test
happens to pass.  This probe picks none.  It measures and hands the ruling to the tolerance
owner.

CONTROLS
========
* **Inputs shared exactly.**  One rng seed, one fixed token sequence (never argmax-driven, or
  the lanes would decode different sentences and the residual would be a plot of that).  The
  seed KV is hashed and the hash must agree across every lane.
* **Correctness read before the byte count**, as in the two rounds before this one.
* **Liveness.**  A Vulkan lane's residual is not believed unless ``dispatches_executed`` moved
  and ``compute_failures``/``device_losses`` are zero.  This project has already been handed
  one "saving" that was an observation ending early.
* **Degeneracy.**  A constant tensor compares equal to any tolerance.  Both sides are screened
  with Trinity's ``_is_degenerate``.
* **Past-prefix preservation.**  ``present[:, :, :past_len, :]`` is asserted bitwise equal to
  the ``past`` that was fed.  This is the aliasing fact the arena rests on, checked here on the
  CPU EP as well — and it is what makes "quantise the newest token only" a faithful model
  rather than a repeated re-quantisation.
* **No clock.**  The box is permanently contended.  Counts, bytes, slopes, ULPs.  The device
  name is read off the run, never off the selector.
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
SCRATCH = HERE / "_scratch"
OPS = REPO / "tests" / "ops"

ONNX_FILE = pathlib.Path(
    r"C:\Users\justinchu\.foundry\cache\models\Microsoft"
    r"\Phi-3.5-mini-instruct-cuda-gpu\cuda-int4-rtn-block-32"
    r"\phi-3.5-mini-instruct-cuda-int4-rtn-block-32.onnx"
)
EP_NAME = "VulkanExecutionProvider"
COUNTERS_ENV = "ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE"

LAYERS = 32
KV_HEADS = 32
HEAD_DIM = 96
F16 = 2
BYTES_PER_PAST_TOKEN_F16 = LAYERS * 2 * KV_HEADS * HEAD_DIM * F16  # 393,216 — Niobe's slope

# Fixed, never argmax-driven: identical input_ids in every lane, so any divergence is the cache.
# The first 15 are the hand-written list the shallow lanes ran on and are frozen so their record
# stays reproducible; beyond that the sequence extends deterministically, because a deep lane
# needs more than a hand-written list and a list that ran out would silently start repeating one
# token — a different chain wearing the same name.
_TOKENS_HEAD = [1, 450, 4996, 310, 278, 3186, 338, 263, 1472, 322, 4234, 5828, 393, 8465, 2645]


def _tokens(n: int) -> list[int]:
    import random as _r
    if n <= len(_TOKENS_HEAD):
        return _TOKENS_HEAD[:n]
    g = _r.Random(20260803)
    return _TOKENS_HEAD + [g.randrange(100, 32000) for _ in range(n - len(_TOKENS_HEAD))]


TOKENS = _tokens(15)

# Measured, elsewhere and earlier, and quoted as the non-KV term of the footprint model:
# the arena's ctx-8192 footprint (bench/results/kv_arena_chain-A8192.json).
ARENA_CTX8192_FOOTPRINT_B = 5_512_528_520

GRANS = {
    # name          -> group size within the newest token's (H*D = 3072)-value slice
    "per_tensor": 3072,
    "per_head": HEAD_DIM,
    "per_block32": 32,
}

LANES = [
    ("cpu_fp16", "cpu", 0, None),
    ("cpu_i8_per_tensor", "cpu", 8, "per_tensor"),
    ("cpu_i8_per_head", "cpu", 8, "per_head"),
    ("cpu_i8_per_block32", "cpu", 8, "per_block32"),
    ("cpu_i4_per_head", "cpu", 4, "per_head"),
    ("cpu_i4_per_block32", "cpu", 4, "per_block32"),
    ("vk_fp16", "vk", 0, None),
    ("vk_i8_per_head", "vk", 8, "per_head"),
    ("vk_i8_per_block32", "vk", 8, "per_block32"),
]


def _lib() -> str:
    return os.environ.get(
        "ONNXRUNTIME_VULKAN_EP_LIB",
        str(REPO / "rust" / "target" / "release" / "onnxruntime_vulkan_ep.dll"),
    )


def _counters(path: pathlib.Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


# ---------------------------------------------------------------------------
# The quantiser.  Symmetric, round-to-nearest-even, scale stored fp16.
# ---------------------------------------------------------------------------


def qdq_token(np, slice_, bits: int, group: int):
    """Round-trip one token's KV slice through ``bits``-bit symmetric quantisation.

    ``slice_`` is ``(1, H, 1, D)`` fp16 — one layer's newest key or value.  Groups are laid out
    along the flattened ``(H, D)`` axis in memory order, so ``group == D`` is exactly per-head
    and ``group == 32`` is exactly the block-32 convention this model's own weights use.

    The scale is **stored fp16** because that is what a real cache would hold, and a scale kept
    in fp32 in a simulation would understate the error of the thing being simulated.

    Returns ``(dequantised fp16 array, number of scales)``.
    """
    a = slice_.astype(np.float32).reshape(-1)
    assert a.size % group == 0, (a.size, group)
    g = a.reshape(-1, group)
    qmax = (1 << (bits - 1)) - 1  # 127 for int8, 7 for int4 (symmetric, 2*qmax+1 levels)
    amax = np.abs(g).max(axis=1, keepdims=True)
    scale = (amax / qmax).astype(np.float16).astype(np.float32)
    # An all-zero group has no scale; leaving it at 0 would divide by zero and a 1.0 there
    # dequantises back to exactly zero, which is the right answer for that group.
    scale = np.where(scale == 0.0, np.float32(1.0), scale)
    q = np.clip(np.rint(g / scale), -qmax, qmax)
    return (q * scale).reshape(slice_.shape).astype(np.float16), int(scale.size)


def scale_bytes_per_token(group: int) -> int:
    """fp16 scales for one past token across the whole cache, at this group size."""
    per_tensor_slice = KV_HEADS * HEAD_DIM
    return LAYERS * 2 * (per_tensor_slice // group) * F16


def value_bytes_per_token(bits: int) -> int:
    return LAYERS * 2 * KV_HEADS * HEAD_DIM * bits // 8


# ---------------------------------------------------------------------------
# Worker — one lane, one chain
# ---------------------------------------------------------------------------


def _worker(lane: str, backend: str, bits: int, gran: str | None, steps: int,
            seed_past: int, out_json: pathlib.Path, out_npz: pathlib.Path) -> int:
    import numpy as np
    import onnxruntime as ort

    doc: dict = {
        "lane": lane, "backend": backend, "bits": bits, "granularity": gran,
        "steps": steps, "seed_past": seed_past, "ort_version": ort.__version__,
    }
    group = GRANS[gran] if gran else None

    counters_path = pathlib.Path(os.environ[COUNTERS_ENV]) if COUNTERS_ENV in os.environ else None
    if counters_path is not None:
        counters_path.unlink(missing_ok=True)

    if backend == "vk":
        try:
            ort.register_execution_provider_library(EP_NAME, _lib())
        except Exception as exc:  # noqa: BLE001
            if "already registered" not in str(exc):
                raise
        ep_device = next((d for d in ort.get_ep_devices() if d.ep_name == EP_NAME), None)
        if ep_device is None:
            doc["verdict"] = "ERROR(instrument)"
            doc["why"] = ["the Vulkan EP is not among ORT's EP devices"]
            out_json.write_text(json.dumps(doc, indent=2), encoding="utf-8")
            return 2
        # Read the device off the run, never off the selector: DEVICE=0 has run on 1=NVIDIA.
        doc["ep_device"] = {
            k: ep_device.ep_metadata.get(k)
            for k in ("vulkan.device_name", "vulkan.device_index", "vulkan.vendor_id")
        }

    providers = ["CPUExecutionProvider"] if backend == "cpu" else [EP_NAME, "CPUExecutionProvider"]
    sess = ort.InferenceSession(
        str(ONNX_FILE),
        ort.SessionOptions(),
        providers=providers,
        free_dimension_overrides_by_name={"batch_size": "1", "sequence_length": "1"},
    )
    if backend == "vk" and EP_NAME not in sess.get_providers():
        doc["verdict"] = "ERROR(instrument)"
        doc["why"] = [f"{EP_NAME} absent from {sess.get_providers()}"]
        out_json.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        return 2

    # The seed past: identical bytes in every lane (same seed, same draw order), so the nine
    # chains are the same chain and any divergence is the cache, not the input.
    rng = np.random.default_rng(20260803)
    seed_kv = [
        rng.standard_normal((1, KV_HEADS, seed_past, HEAD_DIM)).astype(np.float16) * 0.02
        for _ in range(LAYERS * 2)
    ]
    doc["seed_kv_sha256"] = hashlib.sha256(
        b"".join(a.tobytes() for a in seed_kv)
    ).hexdigest()[:16]
    doc["tokens_sha256"] = hashlib.sha256(
        json.dumps(TOKENS[:steps]).encode()
    ).hexdigest()[:16]

    past = {}
    for layer in range(LAYERS):
        past[f"past_key_values.{layer}.key"] = seed_kv[2 * layer]
        past[f"past_key_values.{layer}.value"] = seed_kv[2 * layer + 1]

    # The seed is already "in cache", so it is quantised too — otherwise the first steps would
    # read an fp16 past that no int8 cache could have held.
    n_scales_seed = 0
    if group is not None:
        for name, arr in past.items():
            out = np.empty_like(arr)
            for t in range(seed_past):
                dq, ns = qdq_token(np, arr[:, :, t:t + 1, :], bits, group)
                out[:, :, t:t + 1, :] = dq
                n_scales_seed += ns
            past[name] = out
    doc["seed_scales"] = n_scales_seed

    names = [o.name for o in sess.get_outputs()]
    saved: dict[str, "np.ndarray"] = {}
    per_step: list[dict] = []
    prefix_ok_all = True

    for step in range(steps):
        past_len = seed_past + step
        feeds = {
            "input_ids": np.array([[TOKENS[step]]], dtype=np.int64),
            "attention_mask": np.ones((1, past_len + 1), dtype=np.int64),
        }
        feeds.update(past)
        outs = sess.run(None, feeds)
        got = dict(zip(names, outs))

        # The aliasing fact the arena rests on, checked on this backend too: the model copies
        # `past` into `present` unchanged, so quantising only the newest token is faithful and
        # not a repeated re-quantisation of values already quantised.
        prefix_ok = True
        for layer in range(LAYERS):
            for kind in ("key", "value"):
                p = np.asarray(got[f"present.{layer}.{kind}"])
                q = past[f"past_key_values.{layer}.{kind}"]
                if p.shape[2] < past_len or not np.array_equal(p[:, :, :past_len, :], q):
                    prefix_ok = False
        prefix_ok_all = prefix_ok_all and prefix_ok

        saved[f"s{step}:logits"] = np.asarray(got["logits"])
        new_past = {}
        n_scales = 0
        for layer in range(LAYERS):
            for kind in ("key", "value"):
                arr = np.asarray(got[f"present.{layer}.{kind}"]).copy()
                if group is not None:
                    dq, ns = qdq_token(np, arr[:, :, past_len:past_len + 1, :], bits, group)
                    arr[:, :, past_len:past_len + 1, :] = dq
                    n_scales += ns
                # Saved AFTER quantisation: this is what a caller reading the cache would get,
                # and it is what the next step reads.  Saving the pre-quantisation tensor would
                # measure a cache nobody has.
                saved[f"s{step}:present.{layer}.{kind}"] = arr
                new_past[f"past_key_values.{layer}.{kind}"] = arr
        past = new_past

        per_step.append({
            "step": step,
            "past_len": past_len,
            "token": TOKENS[step],
            "logits_argmax": int(np.asarray(got["logits"]).reshape(-1).argmax()),
            "scales_written": n_scales,
            "past_prefix_preserved": prefix_ok,
        })

    doc["per_step"] = per_step
    doc["past_prefix_preserved_all_steps"] = prefix_ok_all
    doc["counters"] = _counters(counters_path)
    np.savez(out_npz, **saved)
    doc["npz"] = str(out_npz)
    doc["verdict"] = "LANE_RAN"
    out_json.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return 0


# ---------------------------------------------------------------------------
# Parent — comparison with Trinity's instrument, and the verdict
# ---------------------------------------------------------------------------


def _stats(np, m, vk, cpu) -> dict:
    """One output's residual, in Trinity's unit and with Trinity's distribution.

    EDITED 2026-08-03 by Trinity — flagged for Switch's review.
    ==========================================================
    This function used to re-implement the distribution beside ``m.ulp_residual``, which
    is what the docstring below warns against ("a second ULP instrument would be a second
    answer nobody could reconcile") — and it then drifted exactly as predicted.  Its
    ``cancellation_elements`` counted **exact zeros** and its
    ``near_zero_reference_elements`` counted **subnormals**; on the logits both read 0
    while ``max_ulp`` read 337,178.  0.1953125 / 337178 = 5.79e-7, which is fp16's
    spacing at a reference of ~4.9e-4 — eight times *above* the smallest normal, so a
    subnormal test cannot see it and neither counter could ever have explained that max.

    The distribution now comes from ``m.ulp_distribution``, which is the single
    implementation.  Both of the narrower counts are still published under their original
    key names so no number already quoted from this probe's artifacts moves; what changes
    is that ``cancellation_elements`` is now the scale-relative count that actually
    accounts for the max, and the at-scale basis and the two-basis disagreement verdict
    are added.
    """
    d = m.ulp_distribution(vk, cpu)
    b = cpu.astype(np.float64)
    subnormal_ref = d["subnormal_reference_elements"]
    return {
        "near_zero_reference_elements": subnormal_ref,
        "near_zero_reference_fraction": (subnormal_ref / b.size) if b.size else 0.0,
        "exact_zero_reference_elements": d["exact_zero_reference_elements"],
        "tensor_scale": d["tensor_scale"],
        "one_ulp_at_scale": d["one_ulp_at_scale"],
        "median_ulp": d["median_ulp"],
        "p99_ulp": d["p99_ulp"],
        "max_ulp": d["max_ulp"],
        # Use THIS to rank lanes against the same oracle.  `max_ulp` above is attained at
        # whichever element has the smallest reference, and on this probe's own logits it
        # ranked the shipping fp16 path (337,178) 47x worse than a simulated int4 KV cache
        # (7,110).  At the tensor's scale the same residuals rank fp16 145x better
        # (6.25 vs 908).  Blind where `max_ulp` is sharp; read both.
        "median_ulp_at_scale": d["median_ulp_at_scale"],
        "p99_ulp_at_scale": d["p99_ulp_at_scale"],
        "max_ulp_at_scale": d["max_ulp_at_scale"],
        "max_abs": d["max_abs"],
        "cancellation_elements": d["cancellation_elements"],
        "ulp_basis_ratio": d["ulp_basis_ratio"],
        "ulp_basis_verdict": d["ulp_basis_verdict"],
        "bitwise_equal": bool(np.array_equal(vk, cpu)),
        "vk_degenerate": bool(m._is_degenerate(vk)),
        "cpu_degenerate": bool(m._is_degenerate(cpu)),
        "ulp_basis": d["ulp_basis"],
        "ulp_at_scale_basis": d["ulp_at_scale_basis"],
    }


def _compare(np, m, oracle_npz, lane_npz, steps: int) -> dict:
    o = np.load(oracle_npz)
    v = np.load(lane_npz)
    per_output: list[dict] = []
    degenerate = 0
    top1_agree = 0
    top1_ranks: list[int] = []
    for step in range(steps):
        for key in ["logits"] + [
            f"present.{layer}.{kind}" for layer in range(LAYERS) for kind in ("key", "value")
        ]:
            k = f"s{step}:{key}"
            a, b = v[k], o[k]
            if a.shape != b.shape or a.dtype != b.dtype:
                per_output.append({"step": step, "output": key,
                                   "status": "SHAPE_OR_DTYPE_MISMATCH"})
                continue
            s = _stats(np, m, a, b)
            if s["vk_degenerate"] or s["cpu_degenerate"]:
                degenerate += 1
            s.update({"step": step, "output": key})
            per_output.append(s)
        la, lb = v[f"s{step}:logits"], o[f"s{step}:logits"]
        agree = int(la.reshape(-1).argmax() == lb.reshape(-1).argmax())
        top1_agree += agree
        # Where the oracle's chosen token landed in this lane's ranking.  A rank of 2 is a very
        # different failure from a rank of 900, and "top-1 agreement" alone cannot tell them
        # apart — which matters because candidate ruling shape 2 is built on this observable.
        order = np.argsort(-lb.reshape(-1).astype(np.float64))
        rank_of_oracle_top1 = int(
            np.where(np.argsort(-la.reshape(-1).astype(np.float64)) == int(order[0]))[0][0]
        )
        top1_ranks.append(rank_of_oracle_top1)

    def _roll(pred) -> dict:
        rows = [r for r in per_output if r.get("median_ulp") is not None and pred(r["output"])]
        if not rows:
            return {}
        med = sorted(r["median_ulp"] for r in rows)
        return {
            "outputs": len(rows),
            "median_of_median_ulp": med[len(med) // 2],
            "max_of_median_ulp": med[-1],
            "max_p99_ulp": max(r["p99_ulp"] for r in rows),
            "max_ulp": max(r["max_ulp"] for r in rows),
            "max_abs": max(r["max_abs"] for r in rows),
            "cancellation_elements": sum(r["cancellation_elements"] for r in rows),
            "near_zero_reference_elements": sum(r["near_zero_reference_elements"] for r in rows),
            "max_near_zero_reference_fraction": max(
                r["near_zero_reference_fraction"] for r in rows
            ),
            "bitwise_equal_outputs": sum(1 for r in rows if r["bitwise_equal"]),
        }

    # The depth question: is the residual flat in past_len, or does it compound?
    by_step = []
    for step in range(steps):
        rows = [r for r in per_output
                if r.get("step") == step and r.get("median_ulp") is not None
                and r["output"] != "logits"]
        if rows:
            med = sorted(r["median_ulp"] for r in rows)
            by_step.append({"step": step, "median_of_median_ulp": med[len(med) // 2]})
    slope = None
    if len(by_step) >= 2:
        xs = np.array([r["step"] for r in by_step], dtype=np.float64)
        ys = np.array([r["median_of_median_ulp"] for r in by_step], dtype=np.float64)
        slope = float(np.polyfit(xs, ys, 1)[0])

    # A rising per-step median has two candidate causes and they mean opposite things:
    #   (a) COMPOUNDING — a token born at step k inherits the divergence of every step before
    #       it, so later-born token POSITIONS carry more error than earlier-born ones;
    #   (b) the newest token is simply noisier and the mix shifts.
    # These are separated by reading the FINAL step's tensor along its token axis.  A rising
    # profile in token position is (a); a flat profile with a spike at the end is (b).
    last = steps - 1
    by_token_pos = []
    keys = [f"s{last}:present.{layer}.{kind}"
            for layer in range(LAYERS) for kind in ("key", "value")]
    for pos in range(v[keys[0]].shape[2]):
        meds = []
        for k in keys:
            u, _ = m.ulp_residual(v[k][:, :, pos, :], o[k][:, :, pos, :])
            fin = u[np.isfinite(u)]
            if fin.size:
                meds.append(float(np.median(fin)))
        meds.sort()
        by_token_pos.append({"token_pos": pos, "median_of_median_ulp": meds[len(meds) // 2]})

    return {
        "kv_median_ulp_slope_per_step": slope,
        "kv_median_ulp_by_token_position_at_last_step": by_token_pos,
        "kv": _roll(lambda n: n != "logits"),
        "logits": _roll(lambda n: n == "logits"),
        "layer31": _roll(lambda n: n.startswith("present.31.")),
        "degenerate_pairs": degenerate,
        "top1_token_agreement": f"{top1_agree}/{steps}",
        "top1_token_agreement_fraction": top1_agree / steps if steps else 0.0,
        "rank_of_oracle_top1_per_step": top1_ranks,
        "worst_rank_of_oracle_top1": max(top1_ranks) if top1_ranks else None,
        "kv_median_ulp_by_step": by_step,
        "worst_outputs": sorted(
            [r for r in per_output if r.get("median_ulp") is not None],
            key=lambda r: -r["median_ulp"],
        )[:5],
    }


def _footprint_model(bits: int, gran: str | None) -> dict:
    """Class MODEL, never quotable as a measurement (Niobe's provenance rule)."""
    if gran is None:
        kv = BYTES_PER_PAST_TOKEN_F16
        scales = 0
    else:
        kv = value_bytes_per_token(bits)
        scales = scale_bytes_per_token(GRANS[gran])
    per_token = kv + scales
    non_kv = ARENA_CTX8192_FOOTPRINT_B - 8192 * BYTES_PER_PAST_TOKEN_F16
    total = non_kv + 8192 * per_token
    return {
        "provenance": "MODEL",
        "bytes_per_past_token": per_token,
        "value_bytes_per_past_token": kv,
        "scale_bytes_per_past_token": scales,
        "predicted_footprint_ctx8192_B": total,
        "ratio_vs_fp16_arena": ARENA_CTX8192_FOOTPRINT_B / total,
        "non_kv_term_B": non_kv,
        "non_kv_term_provenance": (
            "MEASUREMENT — the arena's measured ctx-8192 footprint 5,512,528,520 B minus "
            "8192 * 393,216; see bench/results/kv_arena_chain-A8192.json"
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed-past", type=int, default=4)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--lanes", default=None, help="comma-separated lane names")
    ap.add_argument("--keep-npz", action="store_true")
    ap.add_argument("--worker", default=None)
    args = ap.parse_args()

    SCRATCH.mkdir(parents=True, exist_ok=True)

    if args.worker:
        name, backend, bits, gran = next(l for l in LANES if l[0] == args.worker)
        return _worker(
            name, backend, bits, gran, args.steps, args.seed_past,
            SCRATCH / f"kv_int8_{name}.json", SCRATCH / f"kv_int8_{name}.npz",
        )

    sys.path.insert(0, str(OPS))
    import numpy as np
    import _models as m  # Trinity's instrument.  Deliberately hers, not a second one.

    if not ONNX_FILE.exists():
        print(f"model not found: {ONNX_FILE}", file=sys.stderr)
        return 2

    wanted = args.lanes.split(",") if args.lanes else [l[0] for l in LANES]
    doc: dict = {
        "probe": "kv_int8_error_budget",
        "question": (
            "what does an int8 KV cache cost, in ULPs, on the outputs criterion 10 already "
            "measures?"
        ),
        "prediction_record": "bench/results/kv-int8-budget-prediction.md",
        "instrument": "tests/ops/_models.ulp_residual (Trinity's), used unmodified",
        "steps": args.steps,
        "seed_past": args.seed_past,
        "tokens": TOKENS[:args.steps],
        "nq_over_nkv": 1.0,
        "untested": [
            "Nq/Nkv = 1.00 on Phi-3.5 and 4x on Llama-3; per_head here is a per-head-group "
            "scale over a group of ONE query head, so nothing here exercises a non-unit "
            "grouping and no number here may be quoted for a 4x model",
            "quantisation is simulated at the host boundary: storage error is modelled "
            "exactly, kernel write rounding and accumulation order are not modelled at all, "
            "so every residual below is a LOWER BOUND on a real int8 kernel's",
        ],
        "lanes": {},
        "why": [],
    }

    rc = 0
    for name, backend, bits, gran in LANES:
        if name not in wanted:
            continue
        env = dict(os.environ)
        cjson = SCRATCH / f"kv_int8_{name}.counters.json"
        env[COUNTERS_ENV] = str(cjson)
        if backend == "vk":
            env["ONNXRUNTIME_EP_VULKAN_DEVICE"] = str(args.device)
        cmd = [sys.executable, str(pathlib.Path(__file__).resolve()),
               "--worker", name, "--steps", str(args.steps),
               "--seed-past", str(args.seed_past)]
        print(f"[lane] {name} ...", flush=True)
        p = subprocess.run(cmd, env=env)
        lane_json = SCRATCH / f"kv_int8_{name}.json"
        if p.returncode != 0 or not lane_json.exists():
            doc["lanes"][name] = {"verdict": "ERROR(instrument)",
                                  "returncode": p.returncode}
            doc["why"].append(f"lane {name} did not complete")
            rc = 2
            continue
        doc["lanes"][name] = json.loads(lane_json.read_text(encoding="utf-8"))

    oracle = "cpu_fp16"
    if oracle not in doc["lanes"] or doc["lanes"][oracle].get("verdict") != "LANE_RAN":
        doc["verdict"] = "ERROR(instrument)"
        doc["why"].append("the fp16 CPU-EP oracle lane did not run; nothing can be compared")
        _write(doc, args)
        return 2

    onpz = SCRATCH / f"kv_int8_{oracle}.npz"
    seed_hashes = set()
    for name, lane in doc["lanes"].items():
        if lane.get("verdict") != "LANE_RAN":
            continue
        seed_hashes.add((lane["seed_kv_sha256"], lane["tokens_sha256"]))
        if not lane.get("past_prefix_preserved_all_steps"):
            doc["why"].append(
                f"lane {name}: present did NOT preserve the past prefix bitwise — the "
                "quantise-the-newest-token-only model is not faithful on this backend"
            )
            rc = 2
        if lane["backend"] == "vk":
            c = lane.get("counters") or {}
            live = int(c.get("dispatches_executed") or 0)
            lane["liveness"] = {
                "dispatches_executed": live,
                "compute_failures": int(c.get("compute_failures") or 0),
                "device_losses": int(c.get("device_losses") or 0),
                "subgraphs_live": int(c.get("subgraphs_live") or 0),
            }
            if live == 0 or lane["liveness"]["compute_failures"] or lane["liveness"]["device_losses"]:
                doc["why"].append(
                    f"lane {name}: liveness failed ({lane['liveness']}); its residual is not "
                    "believed — a small residual because nothing ran is not a small residual"
                )
                rc = 2
        if name != oracle:
            lane["vs_cpu_fp16"] = _compare(
                np, m, onpz, SCRATCH / f"kv_int8_{name}.npz", args.steps
            )
        lane["footprint"] = _footprint_model(lane["bits"], lane["granularity"])
        lane.pop("npz", None)

    if len(seed_hashes) != 1:
        doc["why"].append(f"lanes did not share inputs exactly: {sorted(seed_hashes)}")
        rc = 2
    doc["inputs_shared_exactly"] = len(seed_hashes) == 1

    # ---- the verdict ------------------------------------------------------
    control = doc["lanes"].get("vk_fp16", {}).get("vs_cpu_fp16", {})
    ctrl_kv = (control.get("kv") or {}).get("max_of_median_ulp")
    int8_lanes = {n: l for n, l in doc["lanes"].items()
                  if l.get("bits") == 8 and "vs_cpu_fp16" in l}
    int8_kv = {n: l["vs_cpu_fp16"]["kv"]["max_of_median_ulp"] for n, l in int8_lanes.items()}
    doc["summary"] = {
        "fp16_control_kv_max_of_median_ulp": ctrl_kv,
        "int8_kv_max_of_median_ulp": int8_kv,
        "int4_kv_max_of_median_ulp": {
            n: l["vs_cpu_fp16"]["kv"]["max_of_median_ulp"]
            for n, l in doc["lanes"].items() if l.get("bits") == 4 and "vs_cpu_fp16" in l
        },
        "granularity_monotone_per_block32_le_per_head_le_per_tensor": None,
    }
    cpu8 = {g: doc["lanes"][f"cpu_i8_{g}"]["vs_cpu_fp16"]["kv"]["max_of_median_ulp"]
            for g in GRANS if f"cpu_i8_{g}" in doc["lanes"]
            and "vs_cpu_fp16" in doc["lanes"][f"cpu_i8_{g}"]}
    if len(cpu8) == 3:
        doc["summary"]["granularity_monotone_per_block32_le_per_head_le_per_tensor"] = bool(
            cpu8["per_block32"] <= cpu8["per_head"] <= cpu8["per_tensor"]
        )

    if rc == 0:
        # The question is whether ANY ULP band admits int8 while still catching an fp16-path
        # defect.  The band would have to be at least the int8 residual, which is compared to
        # the fp16 path's own residual.  No tolerance is CHOSEN here: the ratio is reported and
        # the ruling is filed with the tolerance owner.
        #
        # The best (smallest) int8 residual is the fairest case for int8, so it is the one the
        # question is put on.  Any int8 lane counts — an earlier version read only the CPU
        # granularity ladder and returned ERROR(instrument) on a device-only lane subset that
        # had in fact run cleanly.
        cpu8_best = min(int8_kv.values()) if int8_kv else None
        doc["summary"]["best_int8_lane"] = (
            min(int8_kv, key=int8_kv.get) if int8_kv else None
        )
        cpu_fp16_floor = 0.0  # the oracle is itself, by construction
        doc["summary"]["headroom_ratio_int8_over_fp16_control"] = (
            (cpu8_best / ctrl_kv) if (cpu8_best and ctrl_kv) else None
        )
        doc["summary"]["fp16_control_floor"] = cpu_fp16_floor
        if cpu8_best is not None and ctrl_kv is not None and cpu8_best > 3.0 * ctrl_kv:
            doc["verdict"] = "NO_ULP_BAND_ADMITS_INT8_AND_STILL_CATCHES_FP16"
        elif cpu8_best is not None:
            doc["verdict"] = "PREDICTION_FAILED_INT8_SITS_INSIDE_THE_FP16_BAND"
        else:
            doc["verdict"] = "ERROR(instrument)"
            rc = 2
        doc["tolerance_ruling"] = (
            "NOT MADE HERE, DELIBERATELY. A criterion may not be hardened because it is about "
            "to pass nor narrowed because it has just failed, and the mirror of that rule "
            "forbids choosing a tolerance the result under test happens to pass. Filed with "
            "the tolerance owner (Morpheus) with three candidate shapes; see "
            "bench/results/kv-int8-budget-prediction.md section 4."
        )
    else:
        doc["verdict"] = "ERROR(instrument)"

    _write(doc, args)
    if not args.keep_npz:
        for f in SCRATCH.glob("kv_int8_*.npz"):
            f.unlink(missing_ok=True)
    return rc


def _write(doc: dict, args) -> None:
    out = pathlib.Path(args.out) if args.out else HERE / f"kv_int8_budget-dev{args.device}.json"
    out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"\n{doc.get('verdict')}  ->  {out}")


if __name__ == "__main__":
    raise SystemExit(main())
