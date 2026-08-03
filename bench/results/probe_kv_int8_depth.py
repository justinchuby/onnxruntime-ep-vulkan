"""How far does the int8 KV residual compound with depth?  A slope over 8 steps cannot say.

WHY THIS IS A SEPARATE PROBE
============================
``probe_kv_int8_budget.py`` established the int8 error budget at 8 decode steps and returned
``NO_ULP_BAND_ADMITS_INT8_AND_STILL_CATCHES_FP16``.  It also **falsified its own prediction**:
the residual was predicted flat in ``past_len`` and it is not — it rises, and the per-token
-position profile separates the two candidate causes cleanly:

    cpu_i8_per_head, median ULP by token position at the last step:
        [11, 11, 11, 11 | 37, 30, 22, 35, 37, 35, 39, 39]
         ^ the 4 seed tokens   ^ the 8 model-produced tokens

The seed tokens were quantised once from a value the oracle also holds, so **11 ULP is the pure
storage error**.  Model-produced tokens sit at 2-4x that and rise with position: the cache's own
error perturbs attention, which perturbs the next token's KV **before** it is quantised.  It
compounds.

**A slope measured over 8 steps does not bound a 8192-token context, and this project does not
extrapolate slopes.**  So the question the shallow probe cannot answer is put here directly:
run the chain deep and watch the curve's *shape*.  Linear, sub-linear (the error saturates as
attention averages over more tokens) and super-linear are three different verdicts and only one
of them is survivable.

DESIGN
======
Oracle and quantised lane run **in one process, in lockstep, on the same input_ids**, and the
comparison is made per step and thrown away.  That is not a stylistic choice: at 128 steps the
saved tensors would be ~6.6 GB per lane, so the shallow probe's save-then-compare structure does
not reach the depth the question needs.  Memory here is bounded by two live KV caches.

Same controls as the shallow probe: inputs shared exactly (one seed, one token sequence, both
hashed), Trinity's ``ulp_residual`` used unmodified, distribution rather than max, degeneracy
screened, no clock.  CPU EP on both sides — this measures the **arithmetic** of a quantised
cache, and the shallow probe already showed the Vulkan lane agrees with the CPU lane to within
1-2 ULP at every granularity, on both devices.

WHAT IT STILL CANNOT SAY
========================
``Nq/Nkv = 1.00`` here and 4x on Llama-3; nothing below exercises a non-unit head grouping.
Quantisation is simulated at the host boundary, so this is a **lower bound** on a real int8
kernel's residual, never an estimate of it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
OPS = REPO / "tests" / "ops"

sys.path.insert(0, str(HERE))
from probe_kv_int8_budget import (  # noqa: E402
    GRANS,
    HEAD_DIM,
    KV_HEADS,
    LAYERS,
    ONNX_FILE,
    _tokens,
    qdq_token,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--seed-past", type=int, default=4)
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--granularity", default="per_head", choices=sorted(GRANS))
    ap.add_argument("--every", type=int, default=4, help="record a comparison every N steps")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(OPS))
    import numpy as np
    import onnxruntime as ort

    import _models as m  # Trinity's instrument, unmodified.

    if not ONNX_FILE.exists():
        print(f"model not found: {ONNX_FILE}", file=sys.stderr)
        return 2

    group = GRANS[args.granularity]
    tokens = _tokens(args.steps)
    doc: dict = {
        "probe": "kv_int8_depth",
        "question": "does the int8 KV residual compound with depth, and in what shape?",
        "instrument": "tests/ops/_models.ulp_residual (Trinity's), used unmodified",
        "bits": args.bits,
        "granularity": args.granularity,
        "group_size": group,
        "steps": args.steps,
        "seed_past": args.seed_past,
        "ort_version": ort.__version__,
        "backend": "CPUExecutionProvider (both lanes)",
        "tokens_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest()[:16],
        "untested": [
            "Nq/Nkv = 1.00 on Phi-3.5 and 4x on Llama-3; no non-unit head grouping is exercised",
            "quantisation is simulated at the host boundary: a LOWER BOUND on a real kernel's "
            "residual, never an estimate of it",
        ],
    }

    def _sess():
        return ort.InferenceSession(
            str(ONNX_FILE), ort.SessionOptions(),
            providers=["CPUExecutionProvider"],
            free_dimension_overrides_by_name={"batch_size": "1", "sequence_length": "1"},
        )

    s_ref, s_q = _sess(), _sess()
    names = [o.name for o in s_ref.get_outputs()]

    rng = np.random.default_rng(20260803)
    seed_kv = [
        rng.standard_normal((1, KV_HEADS, args.seed_past, HEAD_DIM)).astype(np.float16) * 0.02
        for _ in range(LAYERS * 2)
    ]
    doc["seed_kv_sha256"] = hashlib.sha256(
        b"".join(a.tobytes() for a in seed_kv)
    ).hexdigest()[:16]

    past_ref, past_q = {}, {}
    for layer in range(LAYERS):
        for j, kind in enumerate(("key", "value")):
            n = f"past_key_values.{layer}.{kind}"
            past_ref[n] = seed_kv[2 * layer + j]
            a = seed_kv[2 * layer + j].copy()
            for t in range(args.seed_past):
                a[:, :, t:t + 1, :], _ = qdq_token(np, a[:, :, t:t + 1, :], args.bits, group)
            past_q[n] = a

    series: list[dict] = []
    top1_agree = 0
    first_disagreement = None
    degenerate_seen = 0

    for step in range(args.steps):
        past_len = args.seed_past + step
        base = {
            "input_ids": np.array([[tokens[step]]], dtype=np.int64),
            "attention_mask": np.ones((1, past_len + 1), dtype=np.int64),
        }
        fr = dict(base)
        fr.update(past_ref)
        fq = dict(base)
        fq.update(past_q)
        gr = dict(zip(names, s_ref.run(None, fr)))
        gq = dict(zip(names, s_q.run(None, fq)))

        nref, nq = {}, {}
        meds = []
        for layer in range(LAYERS):
            for kind in ("key", "value"):
                k = f"present.{layer}.{kind}"
                ar = np.asarray(gr[k]).copy()
                aq = np.asarray(gq[k]).copy()
                aq[:, :, past_len:past_len + 1, :], _ = qdq_token(
                    np, aq[:, :, past_len:past_len + 1, :], args.bits, group
                )
                nref[f"past_key_values.{layer}.{kind}"] = ar
                nq[f"past_key_values.{layer}.{kind}"] = aq
                if step % args.every == 0 or step == args.steps - 1:
                    if m._is_degenerate(ar) or m._is_degenerate(aq):
                        degenerate_seen += 1
                        continue
                    u, _b = m.ulp_residual(aq, ar)
                    fin = u[np.isfinite(u)]
                    if fin.size:
                        meds.append(float(np.median(fin)))
        past_ref, past_q = nref, nq

        lr = np.asarray(gr["logits"]).reshape(-1)
        lq = np.asarray(gq["logits"]).reshape(-1)
        agree = int(lr.argmax() == lq.argmax())
        top1_agree += agree
        if not agree and first_disagreement is None:
            first_disagreement = step

        if meds:
            meds.sort()
            u, _b = m.ulp_residual(lq.astype(np.float16), lr.astype(np.float16)) \
                if lr.dtype == np.float16 else m.ulp_residual(lq, lr)
            fin = u[np.isfinite(u)]
            series.append({
                "step": step,
                "past_len": past_len,
                "kv_median_of_median_ulp": meds[len(meds) // 2],
                "kv_worst_median_ulp": meds[-1],
                "logits_median_ulp": float(np.median(fin)) if fin.size else 0.0,
                "logits_max_abs": float(np.abs(lq.astype(np.float64)
                                               - lr.astype(np.float64)).max()),
                "top1_agree_cumulative": top1_agree,
            })
            print(f"  step {step:4d} past_len {past_len:5d}  kv median {meds[len(meds)//2]:8.1f} "
                  f"ULP  logits {series[-1]['logits_median_ulp']:7.1f} ULP  "
                  f"top1 {top1_agree}/{step + 1}", flush=True)

    # The final tensor read along its token axis: storage error (the seed tokens, quantised once
    # from a value the oracle also holds) against compounded error (model-produced tokens).
    keys = [f"present.{layer}.{kind}" for layer in range(LAYERS) for kind in ("key", "value")]
    prof = []
    extent = past_q[f"past_key_values.0.key"].shape[2]
    stride = max(1, extent // 64)
    for pos in range(0, extent, stride):
        mm = []
        for layer in range(LAYERS):
            for kind in ("key", "value"):
                n = f"past_key_values.{layer}.{kind}"
                u, _b = m.ulp_residual(past_q[n][:, :, pos, :], past_ref[n][:, :, pos, :])
                fin = u[np.isfinite(u)]
                if fin.size:
                    mm.append(float(np.median(fin)))
        mm.sort()
        prof.append({"token_pos": pos, "median_of_median_ulp": mm[len(mm) // 2]})

    doc["series"] = series
    doc["final_profile_by_token_position"] = prof
    doc["seed_tokens_storage_only_median_ulp"] = (
        float(np.median([p["median_of_median_ulp"] for p in prof
                         if p["token_pos"] < args.seed_past])) if args.seed_past else None
    )
    doc["top1_token_agreement"] = f"{top1_agree}/{args.steps}"
    doc["first_top1_disagreement_step"] = first_disagreement
    doc["degenerate_comparisons_skipped"] = degenerate_seen

    # The shape of the curve is the verdict, and it is read off the data rather than assumed.
    xs = np.array([s["past_len"] for s in series], dtype=np.float64)
    ys = np.array([s["kv_median_of_median_ulp"] for s in series], dtype=np.float64)
    if xs.size >= 4:
        lin = float(np.polyfit(xs, ys, 1)[0])
        # Fit y = a * x**b.  b ~ 1 linear, b < 1 saturating, b > 1 super-linear.
        ok = (xs > 0) & (ys > 0)
        b_exp = float(np.polyfit(np.log(xs[ok]), np.log(ys[ok]), 1)[0]) if ok.sum() >= 4 else None
        half, rest = ys[: len(ys) // 2], ys[len(ys) // 2:]
        doc["curve"] = {
            "linear_slope_ulp_per_token": lin,
            "power_law_exponent": b_exp,
            "first_half_mean": float(half.mean()),
            "second_half_mean": float(rest.mean()),
            "shape": (
                "SATURATING" if (b_exp is not None and b_exp < 0.8)
                else "LINEAR" if (b_exp is not None and b_exp < 1.2)
                else "SUPER_LINEAR"
            ),
        }
        doc["verdict"] = f"INT{args.bits}_KV_RESIDUAL_COMPOUNDS_{doc['curve']['shape']}"
    else:
        doc["verdict"] = "ERROR(instrument)"

    doc["not_claimed"] = (
        f"measured to past_len {int(xs.max()) if xs.size else 0}. The operating point every "
        "roofline number in this project is quoted at is ctx 8192, which this run does not "
        "reach, and a fitted exponent is not a licence to extrapolate to it. What would make "
        "ctx 8192 knowable: the same lockstep run at that depth, which costs one CPU-EP oracle "
        "chain of 8192 steps and nothing else — the arena already reaches 8192 on the device side."
    )

    out = pathlib.Path(args.out) if args.out else (
        HERE / f"kv_int8_depth-i{args.bits}-{args.granularity}-n{args.steps}.json"
    )
    out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"\n{doc['verdict']}  ->  {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
