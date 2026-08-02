"""What does island fragmentation actually cost, and in what currency?

The record from early in this project is 355 of 363 nodes in ONE fused island.
Today's as-shipped reading is 323 claimed across 33 islands. The question put to
me was whether 33 islands is measurably worse than 1, and in what currency --
because Switch's byte model of intermediates (9.52 MiB against 1996.8 MiB of
weights, 0.47%) does not cover synchronisation, pipeline drain, descriptor
rebinding or allocator round-trips, and his diagnosis of his own error was
"a large count of small things is not a large thing, and I keep reaching for
the count."

THE FIRST ANSWER I GOT WAS WRONG, AND ITS SHAPE IS THE REASON THIS FILE EXISTS.

I divided `session_staging_upload_bytes` by the inference count in each of two
committed records and reported that fragmentation had REDUCED staging traffic
from 78.43 to 43.59 MiB/inference. It had not. That counter is cumulative over
the session and is dominated by the one-time ~2184.7 MiB weight upload, and the
two records had different iteration counts (28 vs 51). Dividing one fixed cost
by two different denominators produces a ratio of 51/28 = 1.82x out of nothing
at all; I measured 1.78x and read it as an improvement. The giveaway was that
the "improvement" equalled the iteration ratio.

R11: a decomposition that appears to close is the hardest kind of wrong.

THE FIX IS A SLOPE, NOT A QUOTIENT. Run each configuration at two iteration
counts and difference them; the fixed session cost cancels exactly and what
survives is the genuine per-inference term:

    bytes(n) = fixed + slope * n        slope = (b2 - b1) / (n2 - n1)

and the recovered `fixed` is then a check, not an output -- it should agree
between the two configurations, because the weights are the same weights.

BOTH ARMS RUN ON ONE BINARY. Forcing the fused arm with a rebuild of an old
commit would have confounded fusion with every other change in between. Instead
both arms are the CURRENT binary and the fused arm is produced by handing the
EP the proof key of the form that declines:

    ONNXRUNTIME_EP_VULKAN_CLAIM_UNPROVEN=com.microsoft::GroupQueryAttention/...

That variable takes proof KEYS. Setting it to `1` -- which I tried first -- is
rejected by the EP's own WARN ("wildcard key '*/all/1/true/yes' is not a valid
proof key. The entire list is ignored"), and both arms of that first attempt
were byte-identical. An A/B in which the two arms agree perfectly is not a
null result, it is ERROR(instrument), and it was the EP's warning that caught
it rather than anything of mine.

This probe reads only counters. Counters do not care whether the box is busy,
so every number here is contention-independent and none of it is a timing.
The timing arm of this A/B is NOT here and cannot be taken on a contended box.

Usage:  python bench/results/probe_island_boundary_cost.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
ROOT = BENCH.parent

MiB = 1024 * 1024

# The four counter records, committed beside this file. Two configurations,
# two iteration counts each. Produced with --no-phases: the traced pass answers
# a proportions question and is not needed for counters.
ARMS = {
    "fused_1_island": ("islandab_slope_fused_n5.json", "islandab_slope_fused_n25.json"),
    "shipped_33_islands": ("islandab_slope_frag_n5.json", "islandab_slope_frag_n25.json"),
}

# Independently derived from the model's declared output shapes, NOT from any
# counter in these records. This is what makes the closure below a check rather
# than an identity: if the recovered readback slope equals this, two unrelated
# sources agree.
N_KV_TENSORS = 64          # present.0.key .. present.31.value
KV_SHAPE = (1, 32, 1, 96)  # f16
F16 = 2
KV_BYTES = N_KV_TENSORS * KV_SHAPE[0] * KV_SHAPE[1] * KV_SHAPE[2] * KV_SHAPE[3] * F16
LOGITS_BYTES = 1 * 1 * 32064 * F16

# Switch's figure, clock-free, for scale only.
WEIGHT_BYTES = int(1996.8 * MiB)

COUNTERS = (
    "session_staging_upload_bytes",
    "session_staging_uploads",
    "session_staging_readback_bytes",
    "session_staging_readbacks",
    "session_device_allocs",
    "dispatches_executed",
)


def read_arm(path: Path) -> dict:
    rec = json.loads(path.read_text(encoding="utf-8"))["results"][0]
    c = rec["counters"]
    islands = c["subgraphs_live"]
    compute_calls = c["compute_calls"]
    if compute_calls % islands:
        raise SystemExit(
            f"ERROR(instrument): {path.name} compute_calls {compute_calls} is not a "
            f"whole multiple of {islands} islands; the inference count is not recoverable."
        )
    return {
        "file": path.name,
        "islands": islands,
        "inferences": compute_calls // islands,
        "high_water_bytes": c["session_device_high_water_bytes"],
        **{k: c[k] for k in COUNTERS},
    }


def solve(lo: dict, hi: dict) -> dict:
    dn = hi["inferences"] - lo["inferences"]
    if dn <= 0:
        raise SystemExit("ERROR(instrument): the two points share an inference count; no slope exists.")
    slope = {k: (hi[k] - lo[k]) / dn for k in COUNTERS}
    fixed = {k: lo[k] - slope[k] * lo["inferences"] for k in COUNTERS}
    return {"slope": slope, "fixed": fixed, "dn": dn}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(HERE / "island_boundary_cost.json"))
    args = ap.parse_args()

    out: dict = {
        "question": "is 33 islands measurably worse than 1, and in what currency?",
        "method": "two iteration counts per configuration; the fixed session cost cancels in the slope",
        "both_arms_same_binary": True,
        "arms": {},
    }

    print("=" * 78)
    print("ISLAND FRAGMENTATION -- what it costs, per inference, counters only")
    print("=" * 78)

    for name, (a, b) in ARMS.items():
        lo, hi = read_arm(HERE / a), read_arm(HERE / b)
        if lo["islands"] != hi["islands"]:
            raise SystemExit(f"ERROR(instrument): {name} arms disagree on island count.")
        sol = solve(lo, hi)
        out["arms"][name] = {"points": [lo, hi], **sol}
        s, f = sol["slope"], sol["fixed"]
        print(f"\n{name}: {lo['islands']} island(s), points at n={lo['inferences']} and n={hi['inferences']}")
        print(f"  fixed session upload        {f['session_staging_upload_bytes'] / MiB:10.2f} MiB   (the weight upload)")
        print(f"  per-inference upload        {s['session_staging_upload_bytes'] / MiB:10.4f} MiB in {s['session_staging_uploads']:5.1f} round-trips")
        print(f"  per-inference readback      {s['session_staging_readback_bytes'] / MiB:10.4f} MiB in {s['session_staging_readbacks']:5.1f} round-trips")
        print(f"  per-inference device allocs {s['session_device_allocs']:10.1f}")
        print(f"  per-inference dispatches    {s['dispatches_executed']:10.1f}")

    fu = out["arms"]["fused_1_island"]
    fr = out["arms"]["shipped_33_islands"]

    # Check 1: the recovered fixed cost is the same weights in both arms.
    fixed_gap = abs(fu["fixed"]["session_staging_upload_bytes"] - fr["fixed"]["session_staging_upload_bytes"])
    fixed_rel = fixed_gap / fu["fixed"]["session_staging_upload_bytes"]
    out["fixed_cost_agreement"] = {
        "fused_MiB": fu["fixed"]["session_staging_upload_bytes"] / MiB,
        "shipped_MiB": fr["fixed"]["session_staging_upload_bytes"] / MiB,
        "relative_gap": fixed_rel,
        "verdict": "AGREE" if fixed_rel < 0.01 else "DISAGREE",
    }

    print("\n" + "-" * 78)
    print("CHECK 1 -- the recovered fixed cost should be the same weights in both arms")
    print(f"  fused {fu['fixed']['session_staging_upload_bytes'] / MiB:.2f} MiB vs shipped "
          f"{fr['fixed']['session_staging_upload_bytes'] / MiB:.2f} MiB -> "
          f"{out['fixed_cost_agreement']['verdict']} ({fixed_rel * 100:.3f}% apart)")

    # Check 2: the fused arm's readback slope should be exactly the model's
    # declared outputs. This number comes from the ONNX shapes, not from us.
    declared = LOGITS_BYTES + KV_BYTES
    got = fu["slope"]["session_staging_readback_bytes"]
    out["declared_output_closure"] = {
        "declared_bytes": declared,
        "measured_bytes": got,
        "residual_bytes": got - declared,
        "verdict": "EXACT" if abs(got - declared) < 1 else "RESIDUAL",
    }
    print("\nCHECK 2 -- the 1-island readback slope against the model's declared outputs")
    print(f"  declared {declared} B (logits {LOGITS_BYTES} + {N_KV_TENSORS} KV x {KV_BYTES // N_KV_TENSORS} B)")
    print(f"  measured {got:.0f} B -> {out['declared_output_closure']['verdict']}")

    # The finding.
    d_up = fr["slope"]["session_staging_upload_bytes"] - fu["slope"]["session_staging_upload_bytes"]
    d_rb = fr["slope"]["session_staging_readback_bytes"] - fu["slope"]["session_staging_readback_bytes"]
    tot_f = fu["slope"]["session_staging_upload_bytes"] + fu["slope"]["session_staging_readback_bytes"]
    tot_r = fr["slope"]["session_staging_upload_bytes"] + fr["slope"]["session_staging_readback_bytes"]

    out["finding"] = {
        "round_trips_per_inference": {"fused": 2.0, "shipped": fr["slope"]["session_staging_uploads"] + fr["slope"]["session_staging_readbacks"]},
        "staging_bytes_per_inference": {"fused": tot_f, "shipped": tot_r, "ratio": tot_r / tot_f},
        "marginal_upload_bytes": d_up,
        "marginal_readback_bytes": d_rb,
        "kv_tensor_bytes": KV_BYTES,
        "marginal_is_one_kv_round_trip": abs(d_rb - KV_BYTES) < 1,
        "upload_residual_bytes": d_up - KV_BYTES,
        "overhead_vs_weight_read_pct": (d_up + d_rb) / WEIGHT_BYTES * 100.0,
        "device_allocs_ratio": fr["slope"]["session_device_allocs"] / fu["slope"]["session_device_allocs"],
        "dispatches_ratio": fr["slope"]["dispatches_executed"] / fu["slope"]["dispatches_executed"],
        "high_water_ratio": fr["points"][0]["high_water_bytes"] / fu["points"][0]["high_water_bytes"],
    }

    print("\n" + "-" * 78)
    print("FINDING -- 33 islands against 1, per inference, same binary")
    print(f"  host round-trips        {2.0:8.0f}  ->  {out['finding']['round_trips_per_inference']['shipped']:8.0f}   {out['finding']['round_trips_per_inference']['shipped'] / 2.0:6.2f}x  WORSE (count)")
    print(f"  staging bytes      {tot_f / MiB:11.4f}  -> {tot_r / MiB:11.4f} MiB {tot_r / tot_f:6.2f}x  WORSE (bytes)")
    print(f"  device allocs      {fu['slope']['session_device_allocs']:11.1f}  -> {fr['slope']['session_device_allocs']:11.1f}     {out['finding']['device_allocs_ratio']:6.2f}x  (not worse)")
    print(f"  dispatches         {fu['slope']['dispatches_executed']:11.1f}  -> {fr['slope']['dispatches_executed']:11.1f}     {out['finding']['dispatches_ratio']:6.2f}x  (not worse)")
    print(f"  device high-water  {fu['points'][0]['high_water_bytes'] / MiB:11.1f}  -> {fr['points'][0]['high_water_bytes'] / MiB:11.1f} MiB {out['finding']['high_water_ratio']:6.3f}x  (not worse)")
    print(f"\n  the marginal traffic is one extra host round-trip of the {N_KV_TENSORS} KV tensors:")
    print(f"    out {d_up:.0f} B, back {d_rb:.0f} B, against {KV_BYTES} B of KV -- readback exact, upload {d_up - KV_BYTES:+.0f} B")
    print(f"  and it is {out['finding']['overhead_vs_weight_read_pct']:.4f}% of the weight read, so in BYTES it is negligible.")

    out["ruling"] = (
        "Fragmentation is worse in exactly two currencies and neither is the one that was "
        "assumed. (1) HOST ROUND-TRIP COUNT: 2 -> 66 per inference, 33x, which is 32 extra "
        "submissions, fence waits and pipeline drains. Their TIME COST IS UNMEASURED and "
        "cannot be taken on a contended box. (2) STAGING BYTES: 1.92x, but the absolute "
        "marginal figure is one extra host round-trip of the 64 KV tensors, 786,424 B, which "
        "is 0.0376% of the weight read -- negligible against Switch's roofline, and his shape "
        "holds: a large count of small things is not a large thing. Allocations, dispatch "
        "count and device high-water are all FLAT OR SLIGHTLY BETTER, so the allocator "
        "round-trip and descriptor-rebinding hypotheses are falsified rather than unmeasured. "
        "THE COST THAT IS NEITHER OF THESE, AND IS PROBABLY THE LARGE ONE, IS THAT THE 32 "
        "DECLINED GroupQueryAttention NODES NOW EXECUTE ON CPU. That is an execution-location "
        "cost, not a boundary cost, it is not in this table, and it is the next thing to measure."
    )
    print("\n" + "-" * 78)
    print("RULING")
    for line in out["ruling"].split(". "):
        if line.strip():
            print("  " + line.strip() + ".")

    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
