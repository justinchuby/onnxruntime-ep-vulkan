#!/usr/bin/env python3
"""The bandwidth roofline for Phi-3.5 int4 decode, derived from the artifact. No clock.

WHY A ROOFLINE IS ADMISSIBLE WHEN A MEASUREMENT IS NOT
======================================================
Every wall-clock figure this project holds is withdrawn or uncertified.  That has been read
as blocking performance work.  It does not, because the quantity that governs a
bandwidth-bound decode is **bytes**, and bytes are a count.

A roofline is a *bound*, and Morpheus's addition today — prefer the bound you can sign —
applies exactly: the floor is monotone, its direction is known, and no arrangement of
kernels can go under it.  It cannot be inflated by a contended box, because contention costs
seconds and this is bytes over bytes-per-second where the numerator is read off the graph and
the denominator off the part's spec sheet.

WHAT THIS ESTABLISHES AND WHAT IT REFUSES TO
============================================
Establishes: the weight-stream floor from the graph, the achieved fraction of it, and the
size and owner of the gap.

**Two byte counts, and they are not the same quantity.** `byte_model` counts the bytes named
by *load instructions*. For the weight stream that equals DRAM traffic -- 2 GB streamed once
against an 8 MB L2 cannot be anything else. For the activation row it does not: the row is
6 KiB at K=3072 and 16 KiB at K=8192, so the `ceil(N / QB_COLS)` re-reads are overwhelmingly
cache hits, exactly as `q_gemv.comp`'s own header says ("the bytes hit L1 but the instructions
still issue"). Conflating the two is a real error and this probe made it before it caught it:
charging the activation term to DRAM puts achieved bandwidth at 248.2 GB/s, **97.0% of spec
peak**, which no GDDR6 controller reaches and which should have been read as a refutation of
the model rather than a triumph. Against the weight stream alone the figure is 171.8 GB/s,
**67.1% of peak** -- an ordinary number, and the one that is true.

So the two counts answer different questions and both are reported:

* **weight stream** -> DRAM roofline -> is the part's bandwidth the limit? (No: 67.1%.)
* **activation reads** -> load-issue and L1/L2 pressure -> what else could the limit be?

Run:  python bench/results/probe_roofline.py
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench" / "results"))

from probe_gemv_counts import byte_model, matmulnbits_census  # noqa: E402

# ARCHIVAL: pinned to the exact Foundry cache layout this investigation measured against
# (the pre-2026-08-05 "...-cuda-gpu/cuda-int4-rtn-block-32/..." catalog revision, issue
# #11). Intentionally NOT auto-resolved against the live cache: a live resolver could
# silently pick a *different* cached revision than the one this result was measured
# against, which would misattribute a new run to an old artifact. Override PHI35_MODEL to
# replay against a different artifact explicitly (issue #19).
MODEL = pathlib.Path(
    os.environ.get(
        "PHI35_MODEL",
        r"C:\Users\justinchu\.foundry\cache\models\Microsoft"
        r"\Phi-3.5-mini-instruct-cuda-gpu\cuda-int4-rtn-block-32"
        r"\phi-3.5-mini-instruct-cuda-int4-rtn-block-32.onnx",
    )
)

#: NVIDIA GeForce RTX 4060 **Laptop** GPU, confirmed by nvidia-smi on this box.
#: 128-bit GDDR6 at 16 Gbps = 128/8 * 16e9 = 256.0 GB/s peak, decimal GB.
#: This is the *spec* peak.  Real GDDR6 controllers sustain roughly 75-85% of it on a pure
#: stream, so the achievable floor is HIGHER than the spec floor -- which moves the result in
#: the conservative direction and is reported as a band rather than a point.
BUS_BITS = 128
MEM_GBPS = 16.0
PEAK_BYTES_PER_S = BUS_BITS / 8 * MEM_GBPS * 1e9
ACHIEVABLE_FRACTION = (0.75, 0.85)

#: The quotable achieved figure. `bench/results/phi35-certified-dev0.json` records
#: `certification.quotable = true`, tail_verdict STEADY, n=41 at 82% coverage, 1.4959% RSD, sole
#: tenant over 51 samples, board at 2010 of 3105 MHz. Reproduced independently in
#: `phi35-certified-dev0-b.json` at 12.1869 ms -- two runs agreeing to 0.02%.
#:
#: This is GPU-busy time on the **device timestamp counter**, which is why it survives when the
#: wall-clock figures do not. What R13 withdrew was the wall-clock *speedup ratios* (3.1x/3.7x),
#: taken during CPU fallback, which compared the CPU EP against itself. That withdrawal does not
#: reach this number: different instrument, different quantity, its own companion attached. Quoted
#: with the companion named, as the certification requires.
ACHIEVED_GPU_BUSY_MS = 12.1847

MIB = 1024 * 1024


def roofline(total_weight_stream_bytes: int) -> dict:
    floor_at_peak = total_weight_stream_bytes / PEAK_BYTES_PER_S
    return {
        "bytes": total_weight_stream_bytes,
        "MiB": total_weight_stream_bytes / MIB,
        "GB_decimal": total_weight_stream_bytes / 1e9,
        "peak_bytes_per_s": PEAK_BYTES_PER_S,
        "peak_GB_per_s": PEAK_BYTES_PER_S / 1e9,
        "floor_ms_at_spec_peak": floor_at_peak * 1e3,
        "floor_ms_at_75pct_achievable": floor_at_peak / ACHIEVABLE_FRACTION[0] * 1e3,
        "floor_ms_at_85pct_achievable": floor_at_peak / ACHIEVABLE_FRACTION[1] * 1e3,
    }


def main() -> int:
    if not MODEL.is_file():
        print(f"ERROR(instrument=model_absent): {MODEL}", file=sys.stderr)
        return 4

    census = matmulnbits_census(MODEL)
    bm = byte_model(census["nodes"])

    weights = bm["weight_bytes"]
    scales = bm["scale_bytes"]
    acts = bm["activation_bytes"]
    irreducible = weights + scales

    rl = roofline(irreducible)


    achieved_s = ACHIEVED_GPU_BUSY_MS / 1e3
    dram_bytes_per_s = irreducible / achieved_s
    report = {
        "probe": "bandwidth_roofline_phi35_int4_decode",
        "device": "NVIDIA GeForce RTX 4060 Laptop GPU",
        "model": str(MODEL),
        "matmulnbits_nodes": census["count"],
        "bytes": {
            "int4_weights": weights,
            "fp16_scales": scales,
            "irreducible_weight_stream": irreducible,
            "activation_load_bytes_mostly_cache_hits": acts,
            "activation_share_of_load_bytes": bm["activation_share"],
            "total_load_bytes": bm["total_bytes"],
        },
        "roofline_dram": rl,
        "achieved": {
            "gpu_busy_ms": ACHIEVED_GPU_BUSY_MS,
            "source": "phi35-certified-dev0.json, certification.quotable=true, STEADY n=41",
            "reproduced_ms": 12.1869,
            "effective_dram_GB_per_s": dram_bytes_per_s / 1e9,
            "fraction_of_spec_peak": dram_bytes_per_s / PEAK_BYTES_PER_S,
            "fraction_of_dram_roofline": rl["floor_ms_at_spec_peak"] / ACHIEVED_GPU_BUSY_MS,
            "headroom_x": ACHIEVED_GPU_BUSY_MS / rl["floor_ms_at_spec_peak"],
        },
        "gap_owner": (
            "NOT DRAM bandwidth: the weight stream occupies 67.1% of spec peak, so a third of "
            "the part's bandwidth is idle while the kernel runs. The kernel is therefore "
            "limited by something other than the memory it is waiting on, and the largest "
            "candidate that is a count rather than a guess is activation load issue -- "
            f"{acts / MIB:.1f} MiB of load-instruction bytes, {bm['activation_share']:.1%} of "
            "all bytes named by loads, that move almost no DRAM traffic and exist only because "
            "each workgroup re-reads the row. Halving them is the tile widening in this commit."
        ),
    }

    out = ROOT / "bench" / "results" / "roofline_phi35.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("BANDWIDTH ROOFLINE — Phi-3.5 int4, batch-1 decode, RTX 4060 Laptop GPU")
    print("=" * 78)
    print(f"  MatMulNBits nodes in graph          {census['count']}")
    print()
    print("  DRAM — every weight byte is read once per token and cannot be cached")
    print(f"    int4 weights                      {weights / MIB:10.1f} MiB")
    print(f"    fp16 scales (one per 32 weights)  {scales / MIB:10.1f} MiB"
          f"   ({scales / irreducible:.1%} of the stream)")
    print(f"    -------------------------------------------------")
    print(f"    weight stream                     {irreducible / MIB:10.1f} MiB"
          f"  = {irreducible / 1e9:.3f} GB")
    print()
    print(f"  Peak bandwidth ({BUS_BITS}-bit @ {MEM_GBPS} Gbps) "
          f"{PEAK_BYTES_PER_S / 1e9:8.1f} GB/s")
    print(f"    floor at spec peak                {rl['floor_ms_at_spec_peak']:10.2f} ms")
    print(f"    floor at 85% achievable           "
          f"{rl['floor_ms_at_85pct_achievable']:10.2f} ms")
    print()
    print(f"  ACHIEVED (quotable, device clock)   {ACHIEVED_GPU_BUSY_MS:10.4f} ms")
    print(f"    effective DRAM bandwidth          {dram_bytes_per_s / 1e9:10.1f} GB/s"
          f"   ({dram_bytes_per_s / PEAK_BYTES_PER_S:.1%} of spec peak)")
    print(f"    fraction of DRAM roofline         "
          f"{rl['floor_ms_at_spec_peak'] / ACHIEVED_GPU_BUSY_MS:10.1%}")
    print(f"    headroom                          "
          f"{ACHIEVED_GPU_BUSY_MS / rl['floor_ms_at_spec_peak']:10.2f}x")
    print()
    print("  LOAD ISSUE — the activation row, re-read once per workgroup, mostly cache hits")
    print(f"    activation load bytes             {acts / MIB:10.1f} MiB"
          f"   ({bm['activation_share']:.2%} of load bytes)")
    print(f"    total named by loads              {bm['total_bytes'] / MIB:10.1f} MiB")
    print()
    print("  GAP OWNER: not DRAM bandwidth — a third of it is idle. See `gap_owner`.")
    print(f"\n  record: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
