#!/usr/bin/env python3
"""The whole decode island in bytes: weights, KV cache, intermediates. No clock.

THE QUESTION THIS ANSWERS
=========================
The roofline in `probe_roofline.py` assumes **each weight byte is read exactly once per
token**.  If the GEMV re-read weights -- poor blocking, a tile that misses cache -- actual
bytes would exceed the 2.09 GB the floor is built on, we would be further from roofline than
67%, and the fix would be blocking rather than anything else.

**It does not re-read them.  The amplification is 1.000000.**

    InB load instructions per inference   116,324,352   (SPIR-V def-use walk, packed path)
    load width                                    16 B  (the type is %v4uint, measured)
    product                            1,861,189,632 B  = 1775.0 MiB
    int4 weight bytes from the graph   1,861,189,632 B  = 1775.0 MiB

**Why that is evidence and not an identity, because it is nearly one.**  `blobs x blob_bytes
= weight_bytes` *is* an identity -- a blob is defined as one (column, block) of packed
weights.  Two things in the product are not identities and both were measured:

* **loads per blob = 1.**  The def-use walk finds a single `%v4uint` load where the unpacked
  path issues four `%uint` loads.  Had the shader issued a load per *element* the count would
  be 8x higher for the same blobs.
* **each blob is touched by exactly one workgroup.**  `col0 = gl_WorkGroupID.x * QB_COLS` and
  `ucol[c] = col0 + c` partition columns across workgroups; invocations stride blocks within
  one.  So no two workgroups read the same weight, and a load instruction is a DRAM read
  rather than a cache hit.  The one path that would break this is the tail-tile redirect
  (`ucol[c] = ... : col0`), which re-reads column `col0`'s blobs -- and it is unreachable
  here because all five Phi-3.5 `N` values are divisible by the 16-wide tile.

Remove either and the product stops matching.  That is what makes 1.000000 a result.

WHAT THE ISLAND IS MADE OF
==========================
So weight streaming is optimal and the remaining budget is everything that is *not* weights.
That budget is dominated by a term nobody has costed, and it is not intermediates.

Run:  python bench/results/probe_island_bytes.py
"""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]

MIB = 1024 * 1024
PEAK_BYTES_PER_S = 256.0e9

# -- Phi-3.5-mini-instruct, read off the graph in `probe_island_census` --------------------
LAYERS = 32
HIDDEN = 3072
FFN = 8192
VOCAB = 32064
#: `past_key_values.N.key` is [batch, 32, past_sequence_length, 96]. Thirty-two KV heads for
#: thirty-two attention heads: **the op is named `GroupQueryAttention` but this model does no
#: grouping**, so the cache is four to eight times what a genuinely grouped model of the same
#: size would carry. A model property, not ours, but it is what makes the term dominate.
KV_HEADS = 32
HEAD_DIM = 96
FP16 = 2

#: Measured by `probe_roofline.py` from the same graph.
WEIGHT_STREAM_BYTES = 2_093_838_336


def kv_bytes(past_len: int) -> int:
    """Bytes of KV cache `GroupQueryAttention` must read to decode one token at `past_len`.

    Every layer reads its whole K and V history. Linear in context and unbounded, which is
    what distinguishes it from every other term here.
    """
    per_token_per_layer = KV_HEADS * HEAD_DIM * FP16
    return LAYERS * 2 * past_len * per_token_per_layer


def intermediate_bytes() -> tuple[int, list[tuple[str, int]]]:
    """The minimum activation traffic between dispatches, at batch 1, one token.

    Every tensor crossing a dispatch boundary, counted once written and once read. This is the
    quantity a dequant+GEMV+activation fusion would remove, so it is the size of that prize.
    """
    items: list[tuple[str, int]] = []

    # Five MatMulNBits per layer. Input row read once, output row written once -- the minimum,
    # ignoring the workgroup re-read amplification, which is a cache-hit term and is accounted
    # separately in `probe_roofline.py`.
    mm = [("qkv", HIDDEN, 3 * HIDDEN), ("o", HIDDEN, HIDDEN),
          ("gate", HIDDEN, FFN), ("up", HIDDEN, FFN), ("down", FFN, HIDDEN)]
    per_layer = sum(i + o for _, i, o in mm) * FP16
    items.append((f"MatMulNBits activations ({len(mm)}/layer)", per_layer * LAYERS))

    # Two SkipSimplifiedLayerNormalization per layer: reads input and skip, writes normalised
    # output and the skip sum the next block consumes.
    items.append(("SkipSimplifiedLayerNormalization (2/layer)",
                  2 * LAYERS * 4 * HIDDEN * FP16))
    # SwiGLU: Sigmoid over the gate, then two Muls.
    items.append(("Sigmoid (1/layer)", LAYERS * 2 * FFN * FP16))
    items.append(("Mul (2/layer)", LAYERS * 2 * 3 * FFN * FP16))
    # GroupQueryAttention's own non-cache traffic: q/k/v in, context out.
    items.append(("GroupQueryAttention q/k/v/out", LAYERS * 4 * HIDDEN * FP16))
    # Head: final norm, lm_head output, embedding gather of one row.
    items.append(("final norm + lm_head output + embed gather",
                  (2 * HIDDEN + VOCAB + HIDDEN) * FP16))
    return sum(v for _, v in items), items


def main() -> int:
    inter, breakdown = intermediate_bytes()
    rows = []
    for past in (0, 128, 512, 2048, 4096, 8192):
        kv = kv_bytes(past)
        total = WEIGHT_STREAM_BYTES + kv + inter
        rows.append({
            "past_sequence_length": past,
            "weight_MiB": WEIGHT_STREAM_BYTES / MIB,
            "kv_cache_MiB": kv / MIB,
            "intermediates_MiB": inter / MIB,
            "total_MiB": total / MIB,
            "kv_share": kv / total,
            "intermediate_share": inter / total,
            "floor_ms_at_spec_peak": total / PEAK_BYTES_PER_S * 1e3,
        })

    report = {
        "probe": "whole_island_bytes_phi35_decode",
        "weight_reread_amplification": {
            "inb_load_instructions_per_inference": 116_324_352,
            "load_width_bytes": 16,
            "product_bytes": 1_861_189_632,
            "int4_weight_bytes_from_graph": 1_861_189_632,
            "amplification": 1.0,
            "verdict": "each weight byte is read exactly once; blocking is not the defect",
        },
        "intermediate_breakdown_bytes": dict(breakdown),
        "by_context_length": rows,
    }
    out = ROOT / "bench" / "results" / "island_bytes_phi35.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("WEIGHT RE-READ CHECK — the assumption the whole roofline rests on")
    print("=" * 78)
    print("  116,324,352 InB loads x 16 B  =  1,861,189,632 B  = 1775.0 MiB")
    print("  int4 weight bytes from graph  =  1,861,189,632 B  = 1775.0 MiB")
    print("  amplification                 =  1.000000")
    print("  -> each weight byte is read exactly once. Blocking is NOT the defect,")
    print("     and the 67%-of-roofline figure stands.")
    print()
    print("WHOLE ISLAND — bytes that must move to decode one token")
    print("=" * 78)
    print("  intermediates (the fusion prize), by producer:")
    for name, v in sorted(breakdown, key=lambda kv: -kv[1]):
        print(f"    {name:44s} {v / MIB:8.2f} MiB")
    print(f"    {'TOTAL intermediates':44s} {inter / MIB:8.2f} MiB")
    print()
    print(f"  {'past_len':>9} {'weights':>10} {'KV cache':>10} {'inter':>8} "
          f"{'total':>10} {'KV%':>7} {'inter%':>8} {'floor ms':>9}")
    for r in rows:
        print(f"  {r['past_sequence_length']:>9} {r['weight_MiB']:>10.1f} "
              f"{r['kv_cache_MiB']:>10.1f} {r['intermediates_MiB']:>8.2f} "
              f"{r['total_MiB']:>10.1f} {r['kv_share']:>7.1%} "
              f"{r['intermediate_share']:>8.3%} {r['floor_ms_at_spec_peak']:>9.2f}")
    print()
    print("  The fusion prize is the `inter%` column. The KV cache is the `KV%` column.")
    print(f"\n  record: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
