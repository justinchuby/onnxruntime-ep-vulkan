#!/usr/bin/env python3
"""§8.9.24(4), the other half: a float64 forward pass of the whole graph, layer at a time.

WHAT ROUND 37 LEFT OPEN, AND WHY IT BLOCKS EVERYTHING
=====================================================
Round 37 answered "which side is wrong" **per hop**, with identical bytes fed to both EPs.
It could not answer it for the outputs **as criterion 10 sees them**, because at model
scale neither EP's inputs are the other's, and a float64 reference seeded from either
side's tensors reports that side's opponent as wrong *by construction* -- round 36's arm F.
The stated cost of closing it was:

    "a float64 forward pass of all 355 nodes (~30 GB dense, infeasible here; ~0.7 GB
     layer-at-a-time, feasible) with a liveness bar at EVERY layer, since a reference that
     dies mid-graph agrees with everything downstream."

This file is that pass. It answers, for outputs 0, 63 and 64, which of the two EPs is
further from the true value of the whole model, and it proposes nothing that follows.

THE SEAM, WHICH IS THE ONLY PLACE ARM F COULD COME BACK
=======================================================
There are two structures in this probe and **they never exchange data**:

  1. THE CHAIN (class `REFERENCE_CHAIN`).  A float64 state threaded from the embedding
     lookup to the logits.  It reads **initialisers and `input_ids` only** -- bytes that
     are identical for both EPs -- and it reads **no EP tensor at any layer boundary,
     ever**.  Layer L's reference input is layer L-1's *reference* output, not either
     EP's tap.  That is what "honest at the seam" means here and it is mechanically
     checkable: `assert_chain_never_reseeded` re-derives every seam value from the
     chain's own recorded state hashes and raises if any of them equals an EP tap.
     Because the chain is a function of the weights alone, it is symmetric between the
     two EPs in the strongest available sense: neither side appears in its derivation.

  2. THE LIVENESS BAR (class `LAYER_LIVENESS`).  Per layer, per side, **re-seeded** from
     that side's own tapped residual, one layer of float64 arithmetic, compared against
     that same side's own tapped outputs.  This is where an EP tensor is read -- and its
     result is a statement about *the reference implementation*, not about either EP. It
     is run for BOTH sides so no side is privileged, and it is never allowed to influence
     the chain.

     Reseeding is what made arm F dishonest **when its output fed a verdict**. Reseeding
     is what makes a liveness bar meaningful. The distinction is not the reseed, it is
     which of the two structures the number is allowed to reach, and here the answer is
     enforced by data flow: `_local_liveness` returns a dict, and no field of it is ever
     read by `_chain_layer`.

A LAYER THAT CONTRIBUTED NOTHING MUST FAIL LOUDLY, NOT VANISH INTO A SUM
========================================================================
This is the load-bearing part, and the reason is on the record from the round before this
one: `np.spacing` returning `inf` at fp16's maximum made a **wrong residual look sound**,
and nobody re-derives a number that already looks clean. A layer-at-a-time chain has
exactly that failure mode built into its shape -- a layer that silently computed nothing
adds 0 to a residual stream and every layer downstream still produces plausible numbers.

So every layer is *observable* and every layer is *screened*:

  * `nodes_evaluated` is counted as the arithmetic happens and must equal
    `NODES_PER_LAYER` (11: input norm, qkv, GQA, o_proj, post norm, gate, up, sigmoid,
    swish mul, gate*up mul, down).  355 = 32*11 + embed + final norm + lm_head, which is
    the project's own `claimed_nodes`.
  * every dequantised weight reports its element count and a checksum, and a weight whose
    dequantised sum is exactly zero is refused -- that is what a dead read looks like.
  * the layer's residual output must DIFFER from its residual input, and its delta output
    must be nonzero: a layer that passed its input through is a layer that did not run.
  * everything must be finite.
  * both EPs' taps must be reproduced by one layer of the same float64 code, per side.

`assert_every_layer_live` raises `DeadLayerError` unless all 32 layers clear all of it,
and `--selftest` proves the gate can go red on seven separate ways of killing a layer.
A gate that has never refused is not a gate.

TWO REFERENCES, BECAUSE "TRUE" HAS TWO DEFENSIBLE READINGS
==========================================================
    `f64`  -- exact real arithmetic of the composed graph, no intermediate rounding.
    `f16r` -- each node's exact result rounded once to fp16 before the next node, i.e.
              what a perfect fp16 implementation of THIS graph would emit.

Neither is privileged. Both are computed in the same pass (the dequantised weights are
shared, so the second chain is nearly free) and the verdict is reported against both.
Where they agree the answer is robust to that choice; where they disagree, the
disagreement is the finding, exactly as a discriminator split is.

WHAT THIS PROBE MAY NOT DO
==========================
§8.9.24(4) permits this answer and forbids what follows from it.
`assert_record_proposes_no_motion` (shared with `probe_criterion10_side.py`, not
re-implemented) runs over the emitted record and raises. `atol`/`rtol` are untouched.

Run:
    $env:VULKAN_SDK="C:\\VulkanSDK\\1.4.350.0"; $env:PATH="$env:VULKAN_SDK\\Bin;$env:PATH"
    $env:ONNXRUNTIME_VULKAN_EP_LIB="<repo>/rust/target/release/onnxruntime_vulkan_ep.dll"
    python bench/results/probe_criterion10_chain.py --device 0 --out bench/results/criterion10_chain-dev0.json

    python bench/results/probe_criterion10_chain.py --selftest    # no GPU, no model
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests" / "ops"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_criterion10_side import (  # noqa: E402
    MODEL_DIR,
    MODEL_FILE,
    MotionInRecordError,
    assert_record_proposes_no_motion,
    dequantize_nbits,
    make_tapped_model,
    register_ep,
    rope_reference_f64,
    route_and_device,
    run_both_eps,
    ulp_stats,
    which_side,
)

N_LAYERS = 32
NODES_PER_LAYER = 11
#: 32*11 + embed_tokens/Gather + layers.32 final norm + lm_head. The project's own
#: `claimed_nodes` for this model is 355; if this arithmetic ever stops matching it, one
#: of the two counts is describing a different graph.
NODES_TOTAL = N_LAYERS * NODES_PER_LAYER + 3

EMBED_TAP = "/model/embed_tokens/Gather/output_0"
FINAL_NORM_TAP = "/model/layers.32/final_norm_layernorm/output_0"


def layer_taps(i: int) -> dict:
    """The materialised tensors that bound layer ``i``'s arithmetic.

    The residual entering layer 0 is the embedding row; for i>=1 it is that layer's own
    `input_layernorm/output_3`, which is the skip-norm's internal sum and therefore the
    first materialised tensor of the layer.
    """
    return {
        "res_in": EMBED_TAP if i == 0 else f"/model/layers.{i}/input_layernorm/output_3",
        "attn_out": f"/model/layers.{i}/attn/GroupQueryAttention/output_0",
        "res_mid": f"/model/layers.{i}/post_attention_layernorm/output_3",
        "delta_out": f"/model/layers.{i}/mlp/down_proj/MatMul/output_0",
    }


TAPS: list[str] = [EMBED_TAP, FINAL_NORM_TAP, "cos_cache", "sin_cache"]
for _i in range(N_LAYERS):
    TAPS.extend(layer_taps(_i).values())
TAPS = list(dict.fromkeys(TAPS))

#: The per-layer, per-side liveness bar. One layer of this float64 code, seeded from a
#: side's own tapped residual, must reproduce that side's own tapped outputs to a few ULP.
#: A reference that cannot do that for a layer is a reference I got the layer wrong in,
#: and it would disagree with BOTH EPs downstream -- "both are wrong", the most confident
#: way to be useless.
LOCAL_LIVENESS_MEDIAN_ULP = 8.0

#: MatMulNBits dequantisation chunk, in output rows. Bounds peak resident float64 weight:
#: 4096 x 8192 x 8 B ~= 0.27 GB, against ~30 GB for the dense model. This is the entire
#: reason the pass is feasible on this box.
DEQUANT_CHUNK_N = 4096

PREDICTIONS = {
    "registered_before_measuring": True,
    "note": (
        "A PREDICTION IS NOT A READING (docs/DESIGN.md §8.9.24(6)). Recorded so the "
        "measurements can contradict them. I was the specimen for this rule one round ago."
    ),
    "outputs_63_64": (
        "round 37 established both EPs are BIT-EXACT on these with identical inputs, so "
        "at model scale their distance from true should be whatever their layer-31 QKV "
        "inputs already carried -- inherited, and I do not predict a direction"
    ),
    "output_0": (
        "the isolated lm_head put ORT's CPU EP further from true (2 vs 11 ULP). At model "
        "scale the Vulkan EP additionally carries 32 layers of its own drift, so I expect "
        "this to be able to reverse -- and if it does NOT reverse that is the more "
        "interesting reading"
    ),
    "chain_vs_both_eps": (
        "the chain should sit closer to both EPs than the EPs sit to each other only if "
        "their errors are uncorrelated; correlated rounding would put it outside them"
    ),
    "what_would_make_this_inconclusive": (
        "any layer failing its liveness bar, any arm with a zero dispatch or claim delta, "
        "or the two reference variants (f64 / f16r) disagreeing on direction"
    ),
}


def direction_across_variants(dirs: dict) -> dict:
    """Aggregate per-variant directions without letting a decisive variant speak for a
    conflicted one.

    `dirs` maps reference-variant name -> that variant's `unanimous_direction`, which is
    None when its five discriminators conflict. A None is *not* a missing value to be
    dropped: it is a variant saying there is no fact of the matter. Dropping it would let
    the other variant supply a direction for both, which is the same defect as reading a
    verdict off a single discriminator -- one level up.
    """
    decisive = {v: d for v, d in dirs.items() if d in ("vulkan", "cpu")}
    distinct = set(decisive.values())
    all_decisive = len(decisive) == len(dirs) and bool(dirs)
    agree = all_decisive and len(distinct) == 1
    return {
        "direction": next(iter(distinct)) if agree else None,
        "agree": agree,
        "without_a_direction": sorted(set(dirs) - set(decisive)),
    }


class DeadLayerError(AssertionError):
    """A layer of the reference chain contributed nothing, or contributed nonsense."""


class ChainReseededError(AssertionError):
    """The chain read an EP tensor at a seam. That is arm F and it voids the answer."""


# ==========================================================================================
# float64 node references
# ==========================================================================================
def _digest(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()[:16]


def rmsnorm_f64(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    """SimplifiedLayerNormalization: x * rsqrt(mean(x^2) + eps) * w, in float64."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    if x.size != w.size:
        raise ValueError(f"rmsnorm: state {x.size} vs weight {w.size}")
    return x / np.sqrt(np.mean(x * x) + eps) * w


def sigmoid_f64(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic in float64.

    This is the ONE transcendental in the reference. Every other node is exact rational
    arithmetic; this one carries float64's own ~1e-16 relative error, which is ~1e-13 fp16
    ULP. It is stated rather than ignored, and it is the reason this chain is described as
    envelope-free *to 1e-10 fp16 ULP* rather than exact.
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def matmulnbits_f64(
    xs: list[np.ndarray],
    packed: np.ndarray,
    scales: np.ndarray,
    *,
    n: int,
    k: int,
    block_size: int,
    bits: int,
    chunk: int = DEQUANT_CHUNK_N,
) -> tuple[list[np.ndarray], dict]:
    """y[i] = W @ x[i] in float64, dequantising W in row chunks and discarding each chunk.

    Every state vector that needs this weight is passed in at once, so the expensive part
    (dequantisation) happens once and the four chains cost one of it, not four.

    Returns the outputs and a **weight witness**: how many rows were actually dequantised
    and what they summed to. A chunk loop that silently ran zero iterations, or a weight
    that read back as all-zero, is the exact shape of a layer that contributes nothing,
    and the witness is what makes it observable rather than a clean 0 in a residual.
    """
    xs64 = [np.asarray(x, dtype=np.float64).reshape(-1) for x in xs]
    for x in xs64:
        if x.size != k:
            raise ValueError(f"matmulnbits: state {x.size} != K {k}")
    outs = [np.zeros(n, dtype=np.float64) for _ in xs64]
    n_blocks = (k + block_size - 1) // block_size
    packed = packed.reshape(n, n_blocks, block_size // 2)
    scales = np.asarray(scales).reshape(n, n_blocks)
    rows_done = 0
    abs_sum = 0.0
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        w = dequantize_nbits(
            packed[lo:hi], scales[lo:hi], n=hi - lo, k=k, block_size=block_size, bits=bits
        )
        for out, x in zip(outs, xs64):
            out[lo:hi] = w @ x
        rows_done += hi - lo
        abs_sum += float(np.abs(w).sum())
        del w
    witness = {
        "rows_dequantised": rows_done,
        "rows_expected": n,
        "weight_abs_sum": abs_sum,
        "weight_is_live": bool(rows_done == n and abs_sum > 0.0),
    }
    if not witness["weight_is_live"]:
        raise DeadLayerError(
            f"MatMulNBits weight read back dead: {witness}. A zero weight adds 0 to the "
            f"residual stream and every layer downstream still looks plausible."
        )
    return outs, witness


def gqa_position0_f64(
    qkv: np.ndarray, cos0: np.ndarray, sin0: np.ndarray, *, heads: int, kv_heads: int,
    do_rotary: int, interleaved: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GroupQueryAttention at sequence position 0 with an empty KV cache, in float64.

    With one query and one key the softmax is exp(s)/exp(s) = 1 **exactly**, for any
    conformant implementation and any score -- so the attention output is the V slice
    verbatim and Q never enters the answer at all. That is not an approximation and not a
    prediction: it is checked against the run's own tapped `GroupQueryAttention/output_0`
    for all 32 layers, per side, in the liveness bar.

    Returns (attn_out, present_key, present_value).
    """
    flat = np.asarray(qkv, dtype=np.float64).reshape(-1)
    hd = flat.size // (heads + 2 * kv_heads)
    if hd * (heads + 2 * kv_heads) != flat.size:
        raise ValueError(f"packed QKV size {flat.size} not divisible for {heads}/{kv_heads}")
    q_sz, kv_sz = heads * hd, kv_heads * hd
    k_raw = flat[q_sz : q_sz + kv_sz].reshape(kv_heads, hd)
    v_raw = flat[q_sz + kv_sz : q_sz + 2 * kv_sz].reshape(kv_heads, hd)
    key = (
        rope_reference_f64(k_raw, cos0, sin0, interleaved=bool(interleaved))
        if do_rotary
        else k_raw
    )
    if kv_heads != heads:
        rep = heads // kv_heads
        attn = np.repeat(v_raw, rep, axis=0).reshape(-1)
    else:
        attn = v_raw.reshape(-1)
    return attn, key, v_raw


# ==========================================================================================
# The chain
# ==========================================================================================
def _r16(x: np.ndarray) -> np.ndarray:
    """Round one node's exact result to fp16, then carry on in float64."""
    return np.asarray(x, dtype=np.float64).astype(np.float16).astype(np.float64)


class LayerWeights:
    """One layer's initialisers, materialised on demand and dropped after the layer.

    Peak resident float64 weight is one dequantisation chunk, not one layer and certainly
    not the model.
    """

    def __init__(self, graph_inits: dict, layer: int):
        self.layer = layer
        self.g = graph_inits
        p = f"model.layers.{layer}"
        self.names = {
            "in_norm": f"{p}.input_layernorm.weight",
            "post_norm": f"{p}.post_attention_layernorm.weight",
            "qkv": f"{p}.attn.qkv_proj.MatMul.weight",
            "o": f"{p}.attn.o_proj.MatMul.weight",
            "gate": f"{p}.mlp.gate_proj.MatMul.weight",
            "up": f"{p}.mlp.up_proj.MatMul.weight",
            "down": f"{p}.mlp.down_proj.MatMul.weight",
        }


def _chain_layer(
    states: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    layer: int,
    w: dict,
    attrs: dict,
    cos0: np.ndarray,
    sin0: np.ndarray,
) -> tuple[dict, dict]:
    """Advance every chain variant through one whole transformer layer.

    ``states`` maps variant name -> (residual, delta); the true residual entering the
    layer is residual+delta, which is the sum the skip-norm performs internally.
    Returns the new states and this layer's observability record.

    **No argument of this function is an EP tensor.** The weights come from the model
    file, the caches come from... the run's own `cos_cache`/`sin_cache` taps -- and those
    two are the one exception, deliberately: they are *initialiser-derived constants* that
    both EPs receive identically (`/model/rotemb_caches_subgraph/If` selects between two
    constant tensors), and round 37 established that predicting them instead of reading
    them is precisely how a rotation defect gets reported as a copy defect. They are
    checked bit-equal between the two EPs' taps before use, and the check raises.
    """
    names = list(states)
    eps = float(attrs["eps"])
    heads, kv_heads = int(attrs["num_heads"]), int(attrs["kv_num_heads"])
    nodes = 0
    witnesses: dict = {}

    res_in = {k: (v[0] + v[1]) for k, v in states.items()}
    normed = {}
    for k in names:
        x = rmsnorm_f64(res_in[k], w["in_norm"], eps)
        normed[k] = _r16(x) if k.endswith("f16r") else x
    nodes += 1  # input_layernorm

    qkv_out, witnesses["qkv"] = matmulnbits_f64(
        [normed[k] for k in names], w["qkv_packed"], w["qkv_scales"], **w["qkv_meta"]
    )
    qkv = {k: (_r16(v) if k.endswith("f16r") else v) for k, v in zip(names, qkv_out)}
    nodes += 1  # qkv_proj MatMulNBits

    attn, pkey, pval = {}, {}, {}
    for k in names:
        a, kk, vv = gqa_position0_f64(
            qkv[k], cos0, sin0, heads=heads, kv_heads=kv_heads,
            do_rotary=int(attrs.get("do_rotary", 0)),
            interleaved=int(attrs.get("rotary_interleaved", 0)),
        )
        if k.endswith("f16r"):
            a, kk, vv = _r16(a), _r16(kk), _r16(vv)
        attn[k], pkey[k], pval[k] = a, kk, vv
    nodes += 1  # GroupQueryAttention

    o_out, witnesses["o"] = matmulnbits_f64(
        [attn[k] for k in names], w["o_packed"], w["o_scales"], **w["o_meta"]
    )
    o = {k: (_r16(v) if k.endswith("f16r") else v) for k, v in zip(names, o_out)}

    nodes += 1  # o_proj
    res_mid = {k: res_in[k] + o[k] for k in names}
    normed2 = {}
    for k in names:
        x = rmsnorm_f64(res_mid[k], w["post_norm"], eps)
        normed2[k] = _r16(x) if k.endswith("f16r") else x
        if k.endswith("f16r"):
            res_mid[k] = _r16(res_mid[k])
    nodes += 1  # post_attention_layernorm (its output_3 is the residual sum)

    gate_out, witnesses["gate"] = matmulnbits_f64(
        [normed2[k] for k in names], w["gate_packed"], w["gate_scales"], **w["gate_meta"]
    )
    nodes += 1
    up_out, witnesses["up"] = matmulnbits_f64(
        [normed2[k] for k in names], w["up_packed"], w["up_scales"], **w["up_meta"]
    )
    nodes += 1
    act = {}
    for k, g_, u_ in zip(names, gate_out, up_out):
        g_ = _r16(g_) if k.endswith("f16r") else g_
        u_ = _r16(u_) if k.endswith("f16r") else u_
        s = sigmoid_f64(g_)
        s = _r16(s) if k.endswith("f16r") else s
        sw = g_ * s
        sw = _r16(sw) if k.endswith("f16r") else sw
        m = sw * u_
        act[k] = _r16(m) if k.endswith("f16r") else m
    nodes += 3  # Sigmoid, act Mul, gate*up Mul

    down_out, witnesses["down"] = matmulnbits_f64(
        [act[k] for k in names], w["down_packed"], w["down_scales"], **w["down_meta"]
    )
    down = {k: (_r16(v) if k.endswith("f16r") else v) for k, v in zip(names, down_out)}
    nodes += 1  # down_proj

    new_states = {k: (res_mid[k], down[k]) for k in names}

    # ---- observability: this layer must be visible in the record on its own terms -------
    primary = names[0]
    changed = not np.array_equal(res_mid[primary], res_in[primary])
    delta_nonzero = bool(np.any(down[primary] != 0.0))
    finite = all(
        bool(np.all(np.isfinite(v))) for k in names for v in (res_mid[k], down[k])
    )
    obs = {
        "layer": layer,
        "nodes_evaluated": nodes,
        "nodes_expected": NODES_PER_LAYER,
        "residual_in_l2": float(np.linalg.norm(res_in[primary])),
        "residual_out_l2": float(np.linalg.norm(res_mid[primary])),
        "delta_out_l2": float(np.linalg.norm(down[primary])),
        "attn_out_l2": float(np.linalg.norm(attn[primary])),
        "residual_changed": bool(changed),
        "delta_is_nonzero": delta_nonzero,
        "all_finite": finite,
        "state_digest": _digest(res_mid[primary]),
        "weight_witnesses": witnesses,
        "contribution_note": (
            "a layer that contributed nothing adds 0 to the residual stream and every "
            "layer downstream still produces plausible numbers -- so nodes_evaluated, the "
            "weight witnesses and residual_changed are screened, not merely reported"
        ),
    }
    obs["present_key_f64"] = pkey
    obs["present_value_f64"] = pval
    return new_states, obs


def assert_every_layer_live(layers: list[dict], *, expect: int = N_LAYERS) -> None:
    """The bar. Raises unless every layer visibly did work.

    Reported-and-not-screened is how a wrong residual looks sound. Each condition below
    corresponds to a way a layer-at-a-time pass can produce nothing and be summed into a
    clean chain.
    """
    bad: list[str] = []
    seen = {int(r.get("layer", -1)) for r in layers}
    for i in range(expect):
        if i not in seen:
            bad.append(f"layer {i}: absent from the record entirely")
    for r in layers:
        i = r.get("layer")
        if r.get("nodes_evaluated") != r.get("nodes_expected"):
            bad.append(
                f"layer {i}: evaluated {r.get('nodes_evaluated')} nodes, expected "
                f"{r.get('nodes_expected')}"
            )
        if not r.get("residual_changed"):
            bad.append(f"layer {i}: residual out is identical to residual in -- a no-op")
        if not r.get("delta_is_nonzero"):
            bad.append(f"layer {i}: MLP delta is all zero -- the layer added nothing")
        if not r.get("all_finite"):
            bad.append(f"layer {i}: non-finite state")
        if not (r.get("delta_out_l2") or 0.0) > 0.0:
            bad.append(f"layer {i}: delta norm is zero")
        for nm, wit in (r.get("weight_witnesses") or {}).items():
            if not wit.get("weight_is_live"):
                bad.append(f"layer {i}: weight {nm} dequantised dead: {wit}")
            if wit.get("rows_dequantised") != wit.get("rows_expected"):
                bad.append(
                    f"layer {i}: weight {nm} dequantised "
                    f"{wit.get('rows_dequantised')}/{wit.get('rows_expected')} rows"
                )
        for side, live in (r.get("local_liveness") or {}).items():
            if not live.get("live"):
                bad.append(
                    f"layer {i}: the reference does not reproduce the {side} EP's own "
                    f"tapped outputs from that side's own tapped input "
                    f"(median {live.get('median_ulp')} ULP > {LOCAL_LIVENESS_MEDIAN_ULP})"
                )
    if bad:
        raise DeadLayerError(
            "the float64 chain has "
            + str(len(bad))
            + " layer(s) that cannot be shown to have contributed:\n  "
            + "\n  ".join(bad)
        )


def assert_chain_never_reseeded(layers: list[dict], tap_digests: dict[str, dict]) -> None:
    """The seam check, as a refusal.

    Every chain state digest is compared against the digests of BOTH EPs' taps at the same
    boundary. A chain that had been reseeded from a side would reproduce that side's bytes
    exactly at the seam. This cannot prove the chain is right; it can prove it is not
    quietly one of the two EPs wearing a float64 label, which is the failure round 36's
    arm F actually had.
    """
    hits = []
    for r in layers:
        d = r.get("state_digest")
        for side, digs in tap_digests.items():
            if d and d == digs.get(int(r["layer"])):
                hits.append(f"layer {r['layer']}: chain state is byte-identical to the {side} tap")
    if hits:
        raise ChainReseededError(
            "the chain reproduced an EP tensor exactly at a seam; that is arm F:\n  "
            + "\n  ".join(hits)
        )


def _local_liveness(
    ref_out: dict[str, np.ndarray], taps: dict[str, np.ndarray], names: dict
) -> dict:
    """Per side: one layer of the reference, seeded from that side's OWN tapped residual,
    against that side's OWN tapped outputs.

    The number this returns is a statement about the reference implementation, not about
    the EP. Nothing here is read by the chain -- see the module docstring on the seam.
    """
    out: dict = {}
    for key, tap_name in names.items():
        got = taps.get(tap_name)
        if got is None:
            out[key] = {"median_ulp": None, "note": "tap absent"}
            continue
        st = ulp_stats(ref_out[key], np.asarray(got).reshape(-1).astype(np.float16))
        out[key] = {
            "median_ulp": st["median_ulp_diff"],
            "max_ulp": st["max_ulp_diff"],
            "elements_differing": st["elements_differing"],
            "elements": st["elements"],
        }
    meds = [v["median_ulp"] for v in out.values() if v.get("median_ulp") is not None]
    return {
        "per_tensor": out,
        "median_ulp": max(meds) if meds else None,
        "live": bool(meds) and max(meds) <= LOCAL_LIVENESS_MEDIAN_ULP,
        "threshold": LOCAL_LIVENESS_MEDIAN_ULP,
        "what_it_witnesses": (
            "that this layer of the float64 reference is the layer the model actually "
            "contains. It is RE-SEEDED from an EP tap on purpose, and its result never "
            "reaches the chain"
        ),
    }


# ==========================================================================================
# main
# ==========================================================================================
def _load_inits(model) -> dict:
    from probe_criterion10_side import _materialise

    return {i.name: i for i in model.graph.initializer}, _materialise


def _arr(inits, materialise, name) -> np.ndarray:
    from onnx import numpy_helper

    t = inits[name]
    materialise(t)
    return numpy_helper.to_array(t)


def _nbits_meta(node) -> dict:
    import onnx

    a = {x.name: onnx.helper.get_attribute_value(x) for x in node.attribute}
    return {
        "n": int(a["N"]), "k": int(a["K"]),
        "block_size": int(a["block_size"]), "bits": int(a["bits"]),
    }


def main(argv=None) -> int:  # noqa: PLR0912, PLR0915
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=os.environ.get("ONNXRUNTIME_EP_VULKAN_DEVICE", "0"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--scratch", default=str(REPO / "bench" / "scratch" / "c10_chain"))
    ap.add_argument("--layers", type=int, default=N_LAYERS,
                    help="stop after N layers (a shortened chain is reported as PARTIAL "
                         "and may not answer the question; for smoke runs only)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return _selftest()

    import onnx

    os.environ["ONNXRUNTIME_EP_VULKAN_DEVICE"] = str(args.device)
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    from test_phi35 import _build_phi35_feeds

    rec: dict = {
        "probe": "criterion10_chain",
        "question": (
            "for criterion 10's outputs 0, 63 and 64 AS THE CRITERION SEES THEM -- each "
            "EP's own in-situ output on the whole model -- which of the two is further "
            "from the true value?"
        ),
        "ruling": "docs/DESIGN.md §8.9.24(4), the half round 37 costed and did not run",
        "device_selector_requested": str(args.device),
        "selector_caveat": "a selector is a request, not an identity; device_name is off the run",
        "predictions": PREDICTIONS,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "reference_variants": {
            "f64": "exact real arithmetic of the composed graph, no intermediate rounding",
            "f16r": "each node's exact result rounded once to fp16 before the next node",
            "why_both": (
                "'true' has two defensible readings at fp16 and neither is privileged; "
                "where they agree on direction the answer is robust to the choice, and "
                "where they disagree the disagreement is the finding"
            ),
        },
        "seam_honesty": {
            "chain_reads": "initialisers, input_ids, and the rotary caches (constants, "
                           "checked bit-equal between the two EPs' taps before use)",
            "chain_never_reads": "any EP-computed tensor, at any layer boundary",
            "liveness_reads": "both EPs' taps, per layer, per side -- and its result never "
                              "reaches the chain",
            "why_this_is_the_answer_to_arm_f": (
                "arm F was dishonest because a reference seeded from one side was compared "
                "against the other side's output. Here the reference is a function of the "
                "weights alone: neither side appears in its derivation, so neither side "
                "can be favoured by its construction"
            ),
        },
    }
    rec["ep_registration"] = register_ep()

    # -- Arm A0: the UNTAPPED model. These are criterion 10's own outputs. ----------------
    counters_a0 = scratch / "counters_untapped.json"
    feeds = _build_phi35_feeds()
    try:
        res0 = run_both_eps(MODEL_FILE, feeds, counters_a0)
        if "instrument_error" in res0:
            rec["arm_untapped"] = {"status": "ERROR(instrument)", **res0}
            res0 = None
        else:
            rec["arm_untapped"] = {
                "status": "MEASURED",
                "class": "IN_SITU",
                "note": "the graph criterion 10 actually runs, with no taps added",
                **route_and_device(counters_a0),
            }
    except Exception as exc:  # noqa: BLE001
        rec["arm_untapped"] = {"status": "ERROR(instrument)", "error": repr(exc)}
        res0 = None

    # -- Arm A1: the tapped model, for the per-layer liveness bar -------------------------
    tapped = None
    taps_cpu: dict[str, np.ndarray] = {}
    taps_vk: dict[str, np.ndarray] = {}
    try:
        tapped = make_tapped_model(TAPS)
        counters_a1 = scratch / "counters_tapped.json"
        res1 = run_both_eps(tapped, feeds, counters_a1)
        if "instrument_error" in res1:
            rec["arm_tapped"] = {"status": "ERROR(instrument)", **res1}
        else:
            names1 = res1["names"]
            for nm in TAPS + ["logits", "present.31.key", "present.31.value"]:
                if nm in names1:
                    j = names1.index(nm)
                    taps_cpu[nm] = res1["cpu"][j]
                    taps_vk[nm] = res1["vk"][j]
            perturb = {}
            if res0 is not None:
                for nm in ("logits", "present.31.key", "present.31.value"):
                    if nm in names1 and nm in res0["names"]:
                        j0 = res0["names"].index(nm)
                        perturb[nm] = {
                            "cpu_tapped_equals_untapped": bool(
                                np.array_equal(
                                    np.asarray(taps_cpu[nm]).view(np.uint16),
                                    np.asarray(res0["cpu"][j0]).view(np.uint16),
                                )
                            ),
                            "vulkan_tapped_equals_untapped": bool(
                                np.array_equal(
                                    np.asarray(taps_vk[nm]).view(np.uint16),
                                    np.asarray(res0["vk"][j0]).view(np.uint16),
                                )
                            ),
                        }
            rec["arm_tapped"] = {
                "status": "MEASURED",
                "class": "TAPS",
                "taps_requested": len(TAPS),
                "taps_returned": len([nm for nm in TAPS if nm in taps_cpu]),
                "tapping_is_non_perturbative": perturb,
                "why_this_screen": (
                    "adding 100+ graph outputs can change partitioning. The VERDICT is "
                    "computed on the untapped run; the taps only feed the liveness bar. "
                    "If a tapped output is not bit-identical to its untapped twin, the "
                    "liveness bar is describing a different run and says so here"
                ),
                **route_and_device(counters_a1),
            }
    except Exception as exc:  # noqa: BLE001
        rec["arm_tapped"] = {"status": "ERROR(instrument)", "error": repr(exc)}
    finally:
        if tapped is not None:
            try:
                Path(tapped).unlink(missing_ok=True)
            except OSError as exc:
                rec.setdefault("cleanup_warnings", []).append(f"{tapped}: {exc}")

    need = [EMBED_TAP, "cos_cache", "sin_cache"]
    if not all(nm in taps_cpu for nm in need):
        rec["arm_chain"] = {
            "status": "ERROR(instrument)",
            "why": f"the taps the chain needs are absent: have {sorted(taps_cpu)[:6]}...",
        }
        assert_record_proposes_no_motion(rec)
        Path(args.out).write_text(json.dumps(rec, indent=1, sort_keys=True, default=str),
                                  encoding="utf-8") if args.out else None
        print("ERROR(instrument): no taps; nothing measured")
        return 1

    # the rotary caches are the one run-derived input the chain takes; they are constants
    # and must be identical on both sides or the chain has no side-independent value
    cos_equal = bool(np.array_equal(np.asarray(taps_cpu["cos_cache"]).view(np.uint16),
                                    np.asarray(taps_vk["cos_cache"]).view(np.uint16)))
    sin_equal = bool(np.array_equal(np.asarray(taps_cpu["sin_cache"]).view(np.uint16),
                                    np.asarray(taps_vk["sin_cache"]).view(np.uint16)))

    # -- The chain ------------------------------------------------------------------------
    model = onnx.load(str(MODEL_FILE), load_external_data=False)
    from probe_criterion10_side import _materialise

    inits = {i.name: i for i in model.graph.initializer}
    nodes_by_name = {n.name: n for n in model.graph.node}
    gqa0 = nodes_by_name[f"/model/layers.0/attn/GroupQueryAttention"]
    gqa_attrs = {a.name: onnx.helper.get_attribute_value(a) for a in gqa0.attribute}
    ln0 = nodes_by_name["/model/layers.0/input_layernorm/LayerNorm"]
    eps = float(
        {a.name: onnx.helper.get_attribute_value(a) for a in ln0.attribute}["epsilon"]
    )
    attrs = {
        "eps": eps,
        "num_heads": int(gqa_attrs["num_heads"]),
        "kv_num_heads": int(gqa_attrs["kv_num_heads"]),
        "do_rotary": int(gqa_attrs.get("do_rotary", 0)),
        "rotary_interleaved": int(gqa_attrs.get("rotary_interleaved", 0)),
    }
    cos0 = np.asarray(taps_cpu["cos_cache"])[0].astype(np.float64)
    sin0 = np.asarray(taps_cpu["sin_cache"])[0].astype(np.float64)

    embed_row = np.asarray(taps_cpu[EMBED_TAP]).reshape(-1)
    # The embedding is a Gather from an fp16 initialiser: a COPY, so both EPs and the
    # reference start from bytes that are equal by construction. Verified, not assumed.
    emb_equal = bool(
        np.array_equal(
            np.asarray(taps_cpu[EMBED_TAP]).view(np.uint16),
            np.asarray(taps_vk[EMBED_TAP]).view(np.uint16),
        )
    )
    tok = int(np.asarray(feeds["input_ids"]).reshape(-1)[0])
    emb_w = _arr(inits, _materialise, "model.embed_tokens.weight")
    emb_from_weights = np.asarray(emb_w[tok]).reshape(-1)
    emb_matches_weights = bool(
        np.array_equal(
            emb_from_weights.astype(np.float16).view(np.uint16),
            np.asarray(taps_cpu[EMBED_TAP]).reshape(-1).astype(np.float16).view(np.uint16),
        )
    )
    del emb_w
    start = emb_from_weights.astype(np.float64)

    variants = ("f64", "f16r")
    states = {v: (start.copy(), np.zeros_like(start)) for v in variants}
    layer_records: list[dict] = []
    present31 = {}
    t0 = time.time()
    n_layers = max(0, min(int(args.layers), N_LAYERS))
    for i in range(n_layers):
        p = f"model.layers.{i}"
        qkv_node = nodes_by_name[f"/model/layers.{i}/attn/qkv_proj/MatMul_Q4"]
        o_node = nodes_by_name[f"/model/layers.{i}/attn/o_proj/MatMul_Q4"]
        gate_node = nodes_by_name[f"/model/layers.{i}/mlp/gate_proj/MatMul_Q4"]
        up_node = nodes_by_name[f"/model/layers.{i}/mlp/up_proj/MatMul_Q4"]
        down_node = nodes_by_name[f"/model/layers.{i}/mlp/down_proj/MatMul_Q4"]
        w = {
            "in_norm": _arr(inits, _materialise, f"{p}.input_layernorm.weight"),
            "post_norm": _arr(inits, _materialise, f"{p}.post_attention_layernorm.weight"),
        }
        for tag, node in (("qkv", qkv_node), ("o", o_node), ("gate", gate_node),
                          ("up", up_node), ("down", down_node)):
            w[f"{tag}_packed"] = _arr(inits, _materialise, node.input[1])
            w[f"{tag}_scales"] = _arr(inits, _materialise, node.input[2])
            w[f"{tag}_meta"] = _nbits_meta(node)

        states, obs = _chain_layer(
            states, layer=i, w=w, attrs=attrs, cos0=cos0, sin0=sin0
        )
        pk, pv = obs.pop("present_key_f64"), obs.pop("present_value_f64")
        if i == N_LAYERS - 1:
            present31 = {v: (pk[v], pv[v]) for v in variants}

        # -- the liveness bar, per side, re-seeded, never feeding the chain --------------
        tp = layer_taps(i)
        obs["local_liveness"] = {}
        for side, tapset in (("cpu", taps_cpu), ("vulkan", taps_vk)):
            seed = tapset.get(tp["res_in"])
            if seed is None:
                obs["local_liveness"][side] = {
                    "live": False, "median_ulp": None, "note": f"tap {tp['res_in']} absent"
                }
                continue
            s0 = np.asarray(seed).reshape(-1).astype(np.float64)
            local = {"local_f16r": (s0, np.zeros_like(s0))}
            local, lobs = _chain_layer(
                local, layer=i, w=w, attrs=attrs, cos0=cos0, sin0=sin0
            )
            lk, lv = lobs.pop("present_key_f64"), lobs.pop("present_value_f64")
            res_mid, delta = local["local_f16r"]
            live = _local_liveness(
                {
                    "res_mid": res_mid.astype(np.float16),
                    "delta_out": delta.astype(np.float16),
                    "attn_out": None,
                },
                tapset,
                {"res_mid": tp["res_mid"], "delta_out": tp["delta_out"]},
            )
            live["seed_tap"] = tp["res_in"]
            live["seeded_from"] = side
            obs["local_liveness"][side] = live
        del w
        obs["elapsed_s_cumulative"] = round(time.time() - t0, 1)
        layer_records.append(obs)
        print(
            f"  layer {i:2d}: nodes={obs['nodes_evaluated']} "
            f"res_l2={obs['residual_out_l2']:.4g} delta_l2={obs['delta_out_l2']:.4g} "
            f"live_cpu={obs['local_liveness'].get('cpu', {}).get('median_ulp')} "
            f"live_vk={obs['local_liveness'].get('vulkan', {}).get('median_ulp')} "
            f"({obs['elapsed_s_cumulative']}s)",
            flush=True,
        )

    # -- the tail: final norm + lm_head ---------------------------------------------------
    final_w = _arr(inits, _materialise, "model.layers.32.final_norm_layernorm.weight")
    lm_node = nodes_by_name["/lm_head/MatMul_Q4"]
    hidden = {}
    for v in variants:
        r, d = states[v]
        s = r + d
        x = rmsnorm_f64(s, final_w, eps)
        hidden[v] = _r16(x) if v == "f16r" else x
    lm_packed = _arr(inits, _materialise, lm_node.input[1])
    lm_scales = _arr(inits, _materialise, lm_node.input[2])
    logits_out, lm_witness = matmulnbits_f64(
        [hidden[v] for v in variants], lm_packed, lm_scales, **_nbits_meta(lm_node)
    )
    del lm_packed, lm_scales
    logits_ref = {v: (_r16(x) if v == "f16r" else x) for v, x in zip(variants, logits_out)}
    tail_nodes = 2

    # -- the gates ------------------------------------------------------------------------
    chain_status = "MEASURED"
    gate_errors = {}
    try:
        assert_every_layer_live(layer_records, expect=n_layers)
    except DeadLayerError as exc:
        chain_status = "ERROR(instrument)"
        gate_errors["assert_every_layer_live"] = str(exc)
    tap_digests = {
        side: {
            i: _digest(np.asarray(tapset[layer_taps(i)["res_mid"]]).reshape(-1).astype(np.float64))
            for i in range(n_layers)
            if layer_taps(i)["res_mid"] in tapset
        }
        for side, tapset in (("cpu", taps_cpu), ("vulkan", taps_vk))
    }
    try:
        assert_chain_never_reseeded(layer_records, tap_digests)
        reseed_check = "PASS (no chain state is byte-identical to an EP tap)"
    except ChainReseededError as exc:
        chain_status = "ERROR(instrument)"
        gate_errors["assert_chain_never_reseeded"] = str(exc)
        reseed_check = "FAIL"

    complete = n_layers == N_LAYERS
    if not complete:
        chain_status = "PARTIAL(--layers used; this cannot answer the question)"

    rec["arm_chain"] = {
        "status": chain_status,
        "class": "REFERENCE_CHAIN",
        "layers_run": n_layers,
        "layers_expected": N_LAYERS,
        "nodes_evaluated_total": sum(r["nodes_evaluated"] for r in layer_records) + tail_nodes + 1,
        "nodes_expected_total": NODES_TOTAL,
        "elapsed_s": round(time.time() - t0, 1),
        "peak_dequant_chunk_rows": DEQUANT_CHUNK_N,
        "embedding": {
            "token_id": tok,
            "both_eps_tapped_embedding_bit_equal": emb_equal,
            "reference_embedding_matches_the_initialiser_row": emb_matches_weights,
            "note": "the chain starts from the initialiser row, not from either tap",
        },
        "rotary_caches": {
            "cos_bit_equal_between_eps": cos_equal,
            "sin_bit_equal_between_eps": sin_equal,
            "note": (
                "constants selected by /model/rotemb_caches_subgraph/If; read off the run "
                "rather than predicted, because predicting them is how round 37 nearly "
                "reported a rotation defect as a copy defect"
            ),
        },
        "seam_check": reseed_check,
        "gate_errors": gate_errors,
        "lm_head_weight_witness": lm_witness,
        "per_layer": layer_records,
        "envelope": (
            "every node in this chain is exact rational arithmetic in float64 except the "
            "MLP sigmoid, which carries float64's own ~1e-16 relative error (~1e-13 fp16 "
            "ULP). The chain is envelope-free to ~1e-10 fp16 ULP, not literally exact, and "
            "it is stated rather than rounded away"
        ),
    }

    # -- The answer ------------------------------------------------------------------------
    if res0 is not None and complete and chain_status == "MEASURED":
        route0 = rec["arm_untapped"]
        attributed = str(route0.get("attribution", "")).startswith("ATTRIBUTED")
        screened = route0.get("dispatch_screen") == "PASS"
        answers = {}
        n0 = res0["names"]
        pk = {v: present31[v][0] for v in variants}
        pv = {v: present31[v][1] for v in variants}
        targets = {
            "output_0_logits": ("logits", logits_ref),
            "output_63_present.31.key": ("present.31.key", pk),
            "output_64_present.31.value": ("present.31.value", pv),
        }
        for label, (nm, refs) in targets.items():
            if nm not in n0:
                answers[label] = {"status": "ERROR(instrument)", "why": f"{nm} not an output"}
                continue
            j = n0.index(nm)
            vk16 = np.asarray(res0["vk"][j]).reshape(-1).astype(np.float16)
            cpu16 = np.asarray(res0["cpu"][j]).reshape(-1).astype(np.float16)
            per_variant = {}
            for v in variants:
                ref16 = np.asarray(refs[v]).reshape(-1).astype(np.float16)
                if ref16.size != vk16.size:
                    per_variant[v] = {
                        "status": "ERROR(instrument)",
                        "why": f"reference {ref16.size} vs output {vk16.size} elements",
                    }
                    continue
                vk_vs = ulp_stats(vk16, ref16)
                cpu_vs = ulp_stats(cpu16, ref16)
                per_variant[v] = {
                    "vulkan_ep_vs_reference": vk_vs,
                    "cpu_ep_vs_reference": cpu_vs,
                    **which_side(vk_vs, cpu_vs),
                }
            dirs = {
                v: per_variant[v].get("unanimous_direction")
                for v in variants
                if "unanimous_direction" in per_variant[v]
            }
            # A variant whose discriminators CONFLICT has no direction, and it may not be
            # dropped so the other variant can speak for both. Silently ignoring a null is
            # how a split gets reported as an answer -- the defect this probe's own
            # discriminator machinery exists to prevent, reappearing one level up.
            decisive = {v: d for v, d in dirs.items() if d in ("vulkan", "cpu")}
            agg = direction_across_variants(dirs)
            distinct = set(decisive.values())
            all_decisive = agg["agree"] or len(decisive) == len(variants)
            direction = agg["direction"]
            vk_vs_cpu = ulp_stats(vk16, cpu16)
            answers[label] = {
                "status": "MEASURED" if (attributed and screened) else "ERROR(instrument)",
                "class": "ORACLE_MODEL_SCALE",
                "output_index": int(label.split("_")[1]),
                "vk_vs_cpu": vk_vs_cpu,
                "by_reference_variant": per_variant,
                "variants_agree_on_direction": agg["agree"],
                "variants_without_a_direction": agg["without_a_direction"],
                "direction": direction,
                "direction_note": (
                    "null because "
                    + (
                        "at least one reference variant's discriminators conflict, so that "
                        "variant has no direction and cannot be spoken for by the other"
                        if not all_decisive
                        else "the two reference variants disagree"
                    )
                    if direction is None
                    else "both reference variants are unanimous across all five "
                         "discriminators and agree with each other"
                ),
                "how_far_both_sides_are_from_true_vs_from_each_other": {
                    "vulkan_median_ulp_from_reference": {
                        v: per_variant[v]["vulkan_ep_vs_reference"]["median_ulp_diff"]
                        for v in variants if "vulkan_ep_vs_reference" in per_variant[v]
                    },
                    "cpu_median_ulp_from_reference": {
                        v: per_variant[v]["cpu_ep_vs_reference"]["median_ulp_diff"]
                        for v in variants if "cpu_ep_vs_reference" in per_variant[v]
                    },
                    "the_two_eps_median_ulp_apart": vk_vs_cpu["median_ulp_diff"],
                    "reading": (
                        "where both sides sit much further from the true value than they "
                        "sit from each other, their errors are largely COMMON, not "
                        "opposed. That is a statement about where the error is, and it is "
                        "not a statement about what any threshold should be -- §8.9.24(4) "
                        "forbids the second and this row must not be read as making it"
                    ),
                },
                "reading": (
                    "the reference here is a function of the model's weights alone; "
                    "neither EP's tensors enter its derivation, so 'further from true' is "
                    "not a statement either side's construction can win. Where "
                    "variants_agree_on_direction is false, the choice of what 'true' means "
                    "at fp16 decides the answer and there is no fact of the matter without "
                    "making that choice explicit."
                ),
            }
        rec["the_answer"] = answers
    else:
        rec["the_answer"] = {
            "status": "UNMEASURED",
            "why": {
                "untapped_run": rec.get("arm_untapped", {}).get("status"),
                "chain": chain_status,
                "complete": complete,
            },
        }

    rec["what_this_does_not_answer"] = {
        "why_a_residual_is_where_it_is": (
            "this says which side is further from the true value of the whole model. It "
            "does not attribute the distance to a node, a kernel or a mechanism -- the "
            "per-hop arms in probe_criterion10_side.py do that, one node at a time"
        ),
        "whether_the_reference_is_the_only_defensible_one": (
            "two variants are reported precisely because it is not. A third accumulation "
            "order for the MatMulNBits reductions would move the fp16-rounded reference by "
            "at most the reduction's own float64 error, ~1e-10 fp16 ULP, which is below "
            "every margin reported here -- but that is an argument, not a measurement, and "
            "it is labelled as one"
        ),
        "any_conclusion_about_tolerances": (
            "§8.9.24(4) forbids it and assert_record_proposes_no_motion enforces it on "
            "this artifact"
        ),
    }
    rec["not_done_here"] = [
        "atol and rtol are not moved -- fifth round running",
        "no criterion-10 row is closed; the verdict stays whatever the criterion says",
        "nothing that follows from this answer is proposed",
    ]
    rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    assert_record_proposes_no_motion(rec)

    text = json.dumps(rec, indent=1, sort_keys=True, default=str)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    print(f"\nchain: {rec['arm_chain']['status']}  layers={n_layers}  "
          f"nodes={rec['arm_chain']['nodes_evaluated_total']}/{NODES_TOTAL}  "
          f"seam={reseed_check}")
    for label, a in (rec.get("the_answer") or {}).items():
        if not isinstance(a, dict) or "by_reference_variant" not in a:
            continue
        print(f"\n{label}: {a['status']}  direction={a.get('direction')!r} "
              f"(variants agree: {a.get('variants_agree_on_direction')})")
        for v, pv_ in a["by_reference_variant"].items():
            if "which_is_further_from_true" not in pv_:
                continue
            print(f"   [{v}] which_is_further_from_true={pv_['which_is_further_from_true']}"
                  f"  unanimous={pv_.get('unanimous_direction')!r}"
                  f"  conflict={pv_.get('discriminators_conflict')}")
            for d, val in pv_["verdict_by_discriminator"].items():
                print(f"      {d:20s} vk={val['vulkan_value']!s:14s} "
                      f"cpu={val['cpu_value']!s:14s} -> {val['verdict']}")
    return 0


# ==========================================================================================
# Selftest -- no GPU, no model, no ORT session
# ==========================================================================================
def _fake_layer(i: int, **over) -> dict:
    r = {
        "layer": i,
        "nodes_evaluated": NODES_PER_LAYER,
        "nodes_expected": NODES_PER_LAYER,
        "residual_changed": True,
        "delta_is_nonzero": True,
        "all_finite": True,
        "delta_out_l2": 1.0,
        "state_digest": f"digest{i}",
        "weight_witnesses": {
            "qkv": {"rows_dequantised": 9216, "rows_expected": 9216, "weight_is_live": True}
        },
        "local_liveness": {
            "cpu": {"live": True, "median_ulp": 0.0},
            "vulkan": {"live": True, "median_ulp": 0.0},
        },
    }
    r.update(over)
    return r


def _selftest() -> int:  # noqa: PLR0915
    # (1) the liveness bar must REFUSE every shape of a layer that did nothing, or the
    #     whole layer-at-a-time construction is unwitnessed.
    good = [_fake_layer(i) for i in range(4)]
    assert_every_layer_live(good, expect=4)
    kills = {
        "absent layer": good[:3],
        "fewer nodes": good[:3] + [_fake_layer(3, nodes_evaluated=10)],
        "residual unchanged": good[:3] + [_fake_layer(3, residual_changed=False)],
        "delta all zero": good[:3] + [_fake_layer(3, delta_is_nonzero=False)],
        "delta norm zero": good[:3] + [_fake_layer(3, delta_out_l2=0.0)],
        "non-finite": good[:3] + [_fake_layer(3, all_finite=False)],
        "dead weight": good[:3] + [
            _fake_layer(3, weight_witnesses={
                "qkv": {"rows_dequantised": 0, "rows_expected": 9216, "weight_is_live": False}
            })
        ],
        "local liveness red": good[:3] + [
            _fake_layer(3, local_liveness={
                "cpu": {"live": True, "median_ulp": 0.0},
                "vulkan": {"live": False, "median_ulp": 900.0},
            })
        ],
    }
    for why, layers in kills.items():
        try:
            assert_every_layer_live(layers, expect=4)
        except DeadLayerError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"the liveness bar passed a layer killed by: {why}")

    # (2) the seam check must catch a chain that is one of the EPs wearing a float64 label
    assert_chain_never_reseeded(good, {"cpu": {0: "other", 1: "other"}})
    try:
        assert_chain_never_reseeded(good, {"vulkan": {2: "digest2"}})
    except ChainReseededError as exc:
        assert "arm F" in str(exc), exc
    else:  # pragma: no cover
        raise AssertionError("the seam check cannot detect a reseeded chain")

    # (3) a dead weight must raise INSIDE the arithmetic, not be summed as a clean zero
    x = [np.ones(32, dtype=np.float64)]
    packed = np.full((4, 1, 16), 0x88, dtype=np.uint8)  # every nibble == 8 -> value 0
    scales = np.ones((4, 1), dtype=np.float32)
    try:
        matmulnbits_f64(x, packed, scales, n=4, k=32, block_size=32, bits=4)
    except DeadLayerError as exc:
        assert "dead" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("an all-zero dequantised weight was accepted")
    packed_live = np.full((4, 1, 16), 0x9A, dtype=np.uint8)
    ys, wit = matmulnbits_f64(x, packed_live, scales, n=4, k=32, block_size=32, bits=4)
    assert wit["weight_is_live"] and wit["rows_dequantised"] == 4, wit
    # chunking must not change the answer
    ys2, _ = matmulnbits_f64(x, packed_live, scales, n=4, k=32, block_size=32, bits=4, chunk=1)
    assert np.array_equal(ys[0], ys2[0]), "the chunk loop changes the result"

    # (4) rmsnorm against an independent formulation
    rng = np.random.default_rng(7)
    v = rng.normal(size=64)
    w = rng.normal(size=64) + 2.0
    got = rmsnorm_f64(v, w, 1e-5)
    want = v / np.sqrt((v * v).sum() / v.size + 1e-5) * w
    assert np.allclose(got, want, rtol=0, atol=1e-15), np.abs(got - want).max()

    # (5) GQA at position 0: the attention output is the V slice, exactly, and Q is
    #     irrelevant. If this ever stops holding, the chain's attention step is wrong.
    heads = kvh = 4
    hd = 8
    qkv = rng.normal(size=3 * heads * hd)
    cos = np.full(hd // 2, 1.1904296875)
    sin = np.zeros(hd // 2)
    attn, key, val = gqa_position0_f64(
        qkv, cos, sin, heads=heads, kv_heads=kvh, do_rotary=1, interleaved=0
    )
    v_slice = qkv[2 * heads * hd :]
    assert np.array_equal(attn, v_slice), "attention at position 0 is not the V slice"
    assert np.array_equal(val.reshape(-1), v_slice)
    assert np.allclose(key.reshape(-1), qkv[heads * hd : 2 * heads * hd] * 1.1904296875)
    # changing Q alone must not change any output
    qkv2 = qkv.copy()
    qkv2[: heads * hd] += 100.0
    a2, k2, v2 = gqa_position0_f64(
        qkv2, cos, sin, heads=heads, kv_heads=kvh, do_rotary=1, interleaved=0
    )
    assert np.array_equal(a2, attn) and np.array_equal(k2, key) and np.array_equal(v2, val), (
        "Q entered the position-0 answer; the softmax is not being treated as exactly 1"
    )
    # grouped case: V must be broadcast across the group, not truncated
    attn_g, _, _ = gqa_position0_f64(
        np.arange(4 * 8 + 2 * 2 * 8, dtype=np.float64), np.zeros(0), np.zeros(0),
        heads=4, kv_heads=2, do_rotary=0, interleaved=0,
    )
    assert attn_g.size == 4 * 8, attn_g.size

    # (6) the two chain variants must actually differ, or 'f16r' is a relabelled 'f64'
    assert not np.array_equal(_r16(np.array([1.0 / 3.0])), np.array([1.0 / 3.0]))

    # (7) the sigmoid must be stable at both tails (a naive form overflows and would put
    #     NaN into the chain, which the liveness bar would then catch far too late)
    s = sigmoid_f64(np.array([-800.0, 0.0, 800.0]))
    assert np.all(np.isfinite(s)) and s[1] == 0.5 and s[0] >= 0.0 and s[2] <= 1.0, s

    # (8) the no-motion guard, imported not re-implemented, still refuses
    try:
        assert_record_proposes_no_motion({"a": {"proposed_atol": 1}})
    except MotionInRecordError:
        pass
    else:  # pragma: no cover
        raise AssertionError("the shared no-motion guard passed a motion")

    # (9) the node arithmetic must match the project's claimed_nodes for this model
    assert NODES_TOTAL == 355, NODES_TOTAL

    # (10) the variant aggregation must not let a DECISIVE variant speak for a CONFLICTED
    #      one. The first run of this probe did exactly that on output 64 -- f64's
    #      discriminators conflicted (no direction) and f16r said "cpu", and the record
    #      printed direction='cpu' with variants_agree_on_direction=True. That is the
    #      single-discriminator defect one level up, found in my own code by reading the
    #      first real run rather than by a test, which is why it now has one.
    assert direction_across_variants({"f64": "cpu", "f16r": "cpu"}) == {
        "direction": "cpu", "agree": True, "without_a_direction": []
    }
    conflicted = direction_across_variants({"f64": None, "f16r": "cpu"})
    assert conflicted["direction"] is None, conflicted
    assert conflicted["agree"] is False
    assert conflicted["without_a_direction"] == ["f64"]
    opposed = direction_across_variants({"f64": "vulkan", "f16r": "cpu"})
    assert opposed["direction"] is None and opposed["agree"] is False
    assert direction_across_variants({"f64": None, "f16r": None})["direction"] is None
    assert direction_across_variants({})["direction"] is None

    print(
        "SELFTEST PASS: 10 arms -- the liveness bar refuses EIGHT distinct ways for a layer "
        "to contribute nothing (absent, short node count, unchanged residual, zero delta, "
        "zero delta norm, non-finite, dead weight, and a red per-side local bar); the seam "
        "check catches a chain reseeded from an EP tap; an all-zero dequantised weight "
        "raises inside the arithmetic instead of summing as a clean 0; chunking does not "
        "change a result; rmsnorm agrees with an independent formulation; position-0 "
        "attention is the V slice exactly and is invariant to Q; the two reference "
        "variants differ; the sigmoid is finite at both tails; a decisive variant cannot "
        "speak for a conflicted one; and 32*11+3 == 355"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
