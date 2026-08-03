"""probe_gqa_grouping.py — does the EP survive `Nq/Nkv != 1`?

WHY THIS EXISTS
===============
Every verdict in this project since 2026-07-30 carries the sentence "`Nq/Nkv = 1.00`
here, 4x on Llama-3".  Switch has written it five times.  It is a caveat, and the team
answered it with instruments rather than with a run: **documenting a risk is not
mitigating it**.  This probe runs the grouped case.

WHICH QUESTION THIS ANSWERS
===========================
This is a **node-level correctness** answer, not an end-to-end one.  It runs synthetic
single-node `com.microsoft.GroupQueryAttention` graphs at Llama-3 8B's attention shape
(Nq=32, Nkv=8, D=128, group size 4) against ORT's CPU EP, with a float64 second reference
where the two disagree.  It does **not** answer:

  * whether a full Llama-3 8B export loads, partitions, and fits in 8 GB (it will not —
    Switch measured the shipping KV lane OOM at past 4096 on this box);
  * whether grouped models are end-to-end token-identical over a generation loop.

Say which question you answered.  This one is: *does `gqa_f16.comp` compute the right
answer, and stay uncorrupted, when several query heads share one KV head?*

THE SPECIFIC HAZARD, NAMED BEFORE RUNNING
=========================================
`attention.rs` (the arena block) proves `present`-aliases-`past` disjointness with:

    read set  = { t : t < past_len }
    write set = { tok_pos = past_len + s_local }   =>  tok_pos >= past_len
    both at base (b*Nkv + kv_h) * stride * D, one common stride

Read that argument again with grouping in mind.  Neither the base, nor the read set, nor
the write set mentions `h` or `Nq/Nkv`; `kv_h` appears identically on both sides.  So the
disjointness of *addresses* is invariant in the group size -- H_DISJOINT below.

What grouping *does* change is the **multiplicity of writers per address**.  At G = 1,
`kv_h = h` and the map (h, s_local) -> (kv_h, tok_pos) is injective: exactly one
invocation writes each `present` slot.  At G = 4 it is 4-to-1: four unordered
invocations perform a non-atomic `atomicAnd` + `atomicOr` pair on the same half-word.
That is a write-write concurrency that **does not exist at G = 1 and is therefore
untested by every run this project has made** -- H_DUP below.

HYPOTHESES, WITH THE OBSERVATION THAT SEPARATES THEM, WRITTEN BEFORE THE FIRST RUN
=================================================================================
H_DISJOINT_HOLDS  the aliasing argument survives grouping unchanged.
    Predicts: arena arms at G=4 agree with the CPU EP to the same per-output ULP
    distribution as the G=1 arms, and are bit-identical across repeated runs.
H_DISJOINT_FAILS  grouping breaks the argument; the arena corrupts.
    Predicts: arena G=4 arms diverge from CPU, *and* -- the signature that distinguishes
    corruption from an arithmetic difference -- the divergence is in the `t < past_len`
    region of `present`, i.e. in rows the dispatch was only supposed to read.
H_DUP_BENIGN  the G duplicate writers write bit-identical values, so every interleaving
    of the And/Or pairs lands on the same bits.
    Predicts: repeated runs of the same grouped arm are **bit-identical**, and equal to
    the G=1 answer for the same KV head.
H_DUP_RACES  the duplicate writers are a real race.
    Predicts: cross-run instability -- the one observable a single-run comparison against
    CPU cannot produce, which is why every arm here runs three times.

Note H_DUP_RACES is falsifiable by an instrument that is *not* the CPU comparison.  Two
observables that can disagree loudly: `cpu_agreement` and `cross_run_identical`.  A race
whose values happen to agree with CPU on one run shows up only in the second.

REPORTING
=========
Per-output verdicts, never an aggregate (R: criterion 10).  `divergent` is symmetric --
where Vulkan and CPU disagree the float64 reference says **which** of the two is further
from true, because this project read `divergent` asymmetrically for six rounds.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _models as m  # noqa: E402

# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------

# Llama-3 8B attention block: 32 query heads, 8 KV heads, head_dim 128 -> Nq/Nkv = 4.
LLAMA3 = dict(nq=32, nkv=8, d=128, rot=128)
# A cheaper grouped shape, same ratio, for the arms that sweep many configurations.
SMALL_G4 = dict(nq=8, nkv=2, d=32, rot=16)
# The G = 1 control at the same total width, so "grouped vs not" is the only difference.
SMALL_G1 = dict(nq=8, nkv=8, d=32, rot=16)
LLAMA3_G1 = dict(nq=32, nkv=32, d=128, rot=128)

MAX_SEQ = 512


def _cos_sin(max_seq: int, rot_dim: int) -> tuple[np.ndarray, np.ndarray]:
    pos = np.arange(max_seq, dtype=np.float32)[:, None]
    freq = 1.0 / (10000 ** (np.arange(0, rot_dim, 2, dtype=np.float32) / rot_dim))
    angles = pos * freq
    return np.cos(angles).astype(np.float16), np.sin(angles).astype(np.float16)


def build_model(
    *,
    nq: int,
    nkv: int,
    d: int,
    rot: int,
    past_stride: int,
    seq_len: int,
    present_declared: bool,
    present_len: int | None = None,
    batch: int = 1,
    symbolic_extents: bool = False,
) -> bytes:
    """One GQA node.

    ``present_declared=False`` leaves the present extent symbolic, which is the *only*
    case the EP will treat as an arena (`declared_present_len.is_none()` in
    `attention.rs`).  Declaring it is how a graph asks for the growing convention in
    writing.

    ``symbolic_extents=True`` additionally makes the **sequence and past extents of the
    inputs** symbolic, as Phi-3.5's real export has them.  MEASURED, and it is not a
    detail: a graph whose every input extent is a literal classifies `static`
    (`ops/common/claim.rs::classify_shapes`), the proof key ends `/static/`, and the
    ledger has no entry for it -- so the node is declined `unproven` and the arena is
    never reached.  A synthetic arena test with static inputs measures nothing.
    """
    packed_dim = (nq + 2 * nkv) * d
    scale = float(d ** -0.5)

    sd = (lambda name, lit: name) if symbolic_extents else (lambda name, lit: lit)
    seq_d = sd("S", seq_len)
    past_d = sd("P", past_stride)

    in_types = {
        "packed_qkv": (TensorProto.FLOAT16, [batch, seq_d, packed_dim]),
        "past_key": (TensorProto.FLOAT16, [batch, nkv, past_d, d]),
        "past_value": (TensorProto.FLOAT16, [batch, nkv, past_d, d]),
        "seqlens_k": (TensorProto.INT32, [batch]),
        "total_seq": (TensorProto.INT32, []),
        "cos_cache": (TensorProto.FLOAT16, [MAX_SEQ, rot // 2]),
        "sin_cache": (TensorProto.FLOAT16, [MAX_SEQ, rot // 2]),
    }
    if present_declared:
        pres_shape = [batch, nkv, present_len, d]
    else:
        pres_shape = [batch, nkv, "T", d]

    inputs = [helper.make_tensor_value_info(n, t, s) for n, (t, s) in in_types.items()]
    outputs = [
        helper.make_tensor_value_info("attn_out", TensorProto.FLOAT16,
                                      [batch, seq_d, nq * d]),
        helper.make_tensor_value_info("present_key", TensorProto.FLOAT16, pres_shape),
        helper.make_tensor_value_info("present_value", TensorProto.FLOAT16, pres_shape),
    ]
    node = helper.make_node(
        "GroupQueryAttention",
        inputs=["packed_qkv", "", "", "past_key", "past_value", "seqlens_k", "total_seq",
                "cos_cache", "sin_cache"],
        outputs=["attn_out", "present_key", "present_value"],
        domain="com.microsoft",
        name="gqa_0",
        num_heads=nq,
        kv_num_heads=nkv,
        scale=scale,
        local_window_size=-1,
        do_rotary=1,
        rotary_interleaved=0,
        smooth_softmax=0,
    )
    graph = helper.make_graph([node], "gqa_grouping", inputs, outputs)
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17),
                       helper.make_opsetid("com.microsoft", 1)],
    )
    return model.SerializeToString()


def make_feeds(
    *, nq: int, nkv: int, d: int, rot: int, past_stride: int, seq_len: int,
    past_len: int, batch: int = 1, seed: int = 0,
) -> dict[str, np.ndarray]:
    """Inputs.

    ``past_len`` is the *true* past length carried by `seqlens_k`; ``past_stride`` is the
    declared extent of the past buffer.  Under the growing convention they are equal.
    Under the arena the stride is a capacity and the tail beyond ``past_len`` is filled
    with a **poison value**, not zeros: a corrupted read of the tail that happened to
    read zeros would be invisible against a zero-filled tail.
    """
    rng = np.random.default_rng(seed)
    packed_dim = (nq + 2 * nkv) * d
    packed = (rng.standard_normal((batch, seq_len, packed_dim)) * 0.1).astype(np.float16)
    pk = (rng.standard_normal((batch, nkv, past_stride, d)) * 0.1).astype(np.float16)
    pv = (rng.standard_normal((batch, nkv, past_stride, d)) * 0.1).astype(np.float16)
    if past_stride > past_len:
        pk[:, :, past_len:, :] = np.float16(0.5)
        pv[:, :, past_len:, :] = np.float16(-0.25)
    cos, sin = _cos_sin(MAX_SEQ, rot)
    return {
        "packed_qkv": packed,
        "past_key": pk,
        "past_value": pv,
        # ORT: seqlens_k[b] == total_sequence_length - 1 == past_len + seq_len - 1
        "seqlens_k": np.full((batch,), past_len + seq_len - 1, dtype=np.int32),
        "total_seq": np.array(past_len + seq_len, dtype=np.int32),
        "cos_cache": cos,
        "sin_cache": sin,
    }


# ---------------------------------------------------------------------------
# float64 second reference
# ---------------------------------------------------------------------------

def gqa_reference_f64(
    feeds: dict[str, np.ndarray], *, nq: int, nkv: int, d: int, rot: int,
    seq_len: int, past_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plain float64 GQA.  Independent of both EPs.

    Deliberately written from the ONNX operator contract, not transcribed from
    `gqa_f16.comp` -- a reference transcribed from the implementation under test agrees
    with it by construction and proves nothing (the Round-36 arm-F mistake).
    Everything is float64 from the inputs' exact fp16 values onward; no online softmax,
    no rescaling, so an accumulation-order difference cannot hide in the reference.

    Returns (attn_out, present_key, present_value) with `present` in the **growing**
    layout [B, Nkv, past_len + S, D]; callers slice it for the arena.
    """
    g = nq // nkv
    rh = rot // 2
    packed = feeds["packed_qkv"].astype(np.float64)
    pk = feeds["past_key"].astype(np.float64)
    pv = feeds["past_value"].astype(np.float64)
    cos = feeds["cos_cache"].astype(np.float64)
    sin = feeds["sin_cache"].astype(np.float64)
    b_n = packed.shape[0]
    scale = float(d ** -0.5)

    def rope(vec: np.ndarray, pos: int) -> np.ndarray:
        out = vec.copy()
        c, s = cos[pos, :rh], sin[pos, :rh]
        x, y = vec[:rh], vec[rh:2 * rh]
        out[:rh] = x * c - y * s
        out[rh:2 * rh] = y * c + x * s
        return out

    attn = np.zeros((b_n, seq_len, nq * d), dtype=np.float64)
    total = past_len + seq_len
    pres_k = np.zeros((b_n, nkv, total, d), dtype=np.float64)
    pres_v = np.zeros((b_n, nkv, total, d), dtype=np.float64)

    for b in range(b_n):
        pres_k[b, :, :past_len, :] = pk[b, :, :past_len, :]
        pres_v[b, :, :past_len, :] = pv[b, :, :past_len, :]
        # New K/V rows, rotated, one per (kv_head, s).
        for kv_h in range(nkv):
            for s in range(seq_len):
                pos = past_len + s
                k_new = packed[b, s, nq * d + kv_h * d: nq * d + (kv_h + 1) * d]
                v_new = packed[b, s, (nq + nkv) * d + kv_h * d:
                                     (nq + nkv) * d + (kv_h + 1) * d]
                pres_k[b, kv_h, pos, :] = rope(k_new, pos)
                pres_v[b, kv_h, pos, :] = v_new
        for h in range(nq):
            kv_h = h // g
            for s in range(seq_len):
                pos = past_len + s
                q = rope(packed[b, s, h * d:(h + 1) * d], pos)
                keys = pres_k[b, kv_h, :pos + 1, :]
                vals = pres_v[b, kv_h, :pos + 1, :]
                sc = (keys @ q) * scale
                sc -= sc.max()
                e = np.exp(sc)
                w = e / e.sum()
                attn[b, s, h * d:(h + 1) * d] = w @ vals
    return attn, pres_k, pres_v


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

def _sess(model: bytes, providers, device_index: int | None = None) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    if device_index is not None and providers != ["CPUExecutionProvider"]:
        opts.add_session_config_entry("ep.device_index", str(device_index))
    return ort.InferenceSession(model, opts, providers=providers)


DEVICE_INDEX: int | None = None


def run_arm(arm: dict, repeats: int = 3) -> dict:
    shape = arm["shape"]
    nq, nkv, d, rot = shape["nq"], shape["nkv"], shape["d"], shape["rot"]
    seq_len, past_len = arm["seq_len"], arm["past_len"]
    past_stride = arm["past_stride"]
    declared = arm["present_declared"]

    model = build_model(
        nq=nq, nkv=nkv, d=d, rot=rot, past_stride=past_stride, seq_len=seq_len,
        present_declared=declared,
        present_len=(past_len + seq_len) if declared else None,
        symbolic_extents=arm.get("symbolic_extents", not declared),
    )
    feeds = make_feeds(nq=nq, nkv=nkv, d=d, rot=rot, past_stride=past_stride,
                       seq_len=seq_len, past_len=past_len, seed=arm.get("seed", 0))

    rec: dict = {
        "arm": arm["name"],
        "group_size": nq // nkv,
        "nq": nq, "nkv": nkv, "head_dim": d, "rotary_dim": rot,
        "seq_len": seq_len, "past_len": past_len, "past_stride": past_stride,
        "present_declared": declared,
        "symbolic_extents": arm.get("symbolic_extents", not declared),
        "kv_arena_env": os.environ.get("ONNXRUNTIME_EP_VULKAN_KV_ARENA", ""),
        "regime": "prefill" if seq_len > 1 else "decode",
    }
    dev = _ep_device()
    if dev is not None:
        rec["device_name"] = dev.ep_metadata.get("vulkan.device_name")

    # Vacuous-pass guard: an unclaimed node is CPU-vs-CPU and agrees perfectly.
    try:
        claimed = m.is_vulkan_claimed(model, feeds)
    except Exception as exc:  # pragma: no cover - environment dependent
        rec["claim_error"] = repr(exc)
        claimed = None
    rec["vulkan_claimed"] = claimed
    if claimed is not True:
        rec["verdict"] = "UNMEASURED(node_not_claimed)"
        return rec

    try:
        vk_sess = _sess(model, m.EP_PROVIDERS, DEVICE_INDEX)
        runs = [vk_sess.run(None, feeds) for _ in range(repeats)]
        cpu_out = _sess(model, ["CPUExecutionProvider"]).run(None, feeds)
    except Exception as exc:
        rec["run_error"] = repr(exc)
        rec["verdict"] = "UNMEASURED(run_failed)"
        return rec

    vk_out = runs[0]
    rec["vk_shapes"] = [list(a.shape) for a in vk_out]
    rec["cpu_shapes"] = [list(a.shape) for a in cpu_out]

    # -- Observable 1: cross-run stability.  A race between the G duplicate writers is
    # visible here and nowhere else.
    identical, differing = True, []
    for i in range(1, repeats):
        eq, idx = m.outputs_bit_equal(runs[0], runs[i])
        if not eq:
            identical = False
            differing.extend(idx)
    rec["cross_run_identical"] = identical
    rec["cross_run_differing_outputs"] = sorted(set(differing))
    rec["repeats"] = repeats

    # -- Observable 2: per-output agreement with the CPU EP.
    outcome, facts = m.compare_all_outputs_to_cpu(list(vk_out), list(cpu_out))
    rec["cpu_comparison"] = outcome
    rec["per_output"] = facts["per_output"]
    rec["oracle_failing_indices"] = facts["oracle_failing_indices"]
    rec["oracle_degenerate_indices"] = facts["oracle_degenerate_indices"]

    # -- Observable 3 (only where 1 and 2 leave a question): the float64 reference.
    # `divergent` is symmetric; this says WHICH side is further from true.
    ref = gqa_reference_f64(feeds, nq=nq, nkv=nkv, d=d, rot=rot,
                            seq_len=seq_len, past_len=past_len)
    names = ["attn_out", "present_key", "present_value"]
    f64: list[dict] = []
    for i, name in enumerate(names):
        r = ref[i]
        v, c = vk_out[i], cpu_out[i]
        if i > 0:
            # present: compare only the rows the contract defines -- [0, past_len+seq_len).
            # Under the arena the buffer is longer (a capacity); its tail is poison and is
            # not part of any contract.  Sliced explicitly and said so, rather than
            # letting a shape mismatch masquerade as a disagreement.
            total = past_len + seq_len
            v = v[:, :, :total, :]
            c = c[:, :, :total, :] if c.shape[2] >= total else c
            r = r[:, :, :total, :]
        if v.shape != r.shape or c.shape != r.shape:
            f64.append({"output": name, "status": "SHAPE_MISMATCH",
                        "vk": list(v.shape), "cpu": list(c.shape), "ref": list(r.shape)})
            continue
        dv = np.abs(v.astype(np.float64) - r)
        dc = np.abs(c.astype(np.float64) - r)
        further = "neither (equal)"
        if float(dv.max()) > float(dc.max()):
            further = "vulkan"
        elif float(dc.max()) > float(dv.max()):
            further = "cpu"
        f64.append({
            "output": name,
            "vk_max_abs_from_f64": float(dv.max()),
            "cpu_max_abs_from_f64": float(dc.max()),
            "vk_median_abs_from_f64": float(np.median(dv)),
            "cpu_median_abs_from_f64": float(np.median(dc)),
            "which_is_further_from_true": further,
        })
    rec["f64_reference"] = f64

    # -- The corruption signature, tested separately from the tolerance question.
    # A broken alias writes into the region the dispatch was only supposed to READ,
    # i.e. present rows t < past_len must equal past rows t < past_len bit-for-bit.
    total = past_len + seq_len
    pk_in = feeds["past_key"][:, :, :past_len, :]
    pv_in = feeds["past_value"][:, :, :past_len, :]
    pk_out = vk_out[1][:, :, :past_len, :]
    pv_out = vk_out[2][:, :, :past_len, :]
    past_intact = bool(np.array_equal(pk_out.view(np.uint16), pk_in.view(np.uint16))
                       and np.array_equal(pv_out.view(np.uint16), pv_in.view(np.uint16)))
    rec["past_region_of_present_bit_intact"] = past_intact
    rec["past_region_rows_checked"] = int(past_len)
    if past_len == 0:
        rec["past_region_note"] = (
            "past_len == 0: nothing to preserve, this check is vacuous here"
        )
    # Arena tail: rows at or beyond total must still hold the poison value.  This is the
    # out-of-contract region; a write there is an overrun even though no oracle covers it.
    tail_intact = None
    if vk_out[1].shape[2] > total:
        tail_k = vk_out[1][:, :, total:, :]
        tail_v = vk_out[2][:, :, total:, :]
        tail_intact = bool(np.all(tail_k == np.float16(0.5))
                           and np.all(tail_v == np.float16(-0.25)))
    rec["arena_tail_poison_intact"] = tail_intact

    # -- Verdict, per output, never an aggregate.
    rec["verdict"] = derive_verdict(rec)
    return rec


def _ep_device():
    devs = [d for d in ort.get_ep_devices() if d.ep_name == m.EP_NAME]
    if not devs:
        return None
    idx = DEVICE_INDEX or 0
    return devs[idx] if idx < len(devs) else devs[0]


def run_arena_arm(arm: dict, repeats: int = 3) -> dict:
    """The arena lane: `present` is bound to the same OrtValue as `past`.

    The EP refuses the arena unless the caller *actually* aliases the two -- it checks
    that ORT's output tensor is the buffer the dispatch wrote and raises otherwise.  So
    this arm needs io-binding onto device memory, which needs
    ``ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY=1`` and ``ONNXRUNTIME_EP_VULKAN_BIND_OUTPUTS=1``
    in the process environment before the session is built.  A run without them is not a
    weaker arena test, it is a *growing-convention* test wearing the arena's name -- the
    EP declines and ORT falls back, and the comparison would pass.
    """
    shape = arm["shape"]
    nq, nkv, d, rot = shape["nq"], shape["nkv"], shape["d"], shape["rot"]
    seq_len, past_len, past_stride = arm["seq_len"], arm["past_len"], arm["past_stride"]

    rec: dict = {
        "arm": arm["name"], "lane": "arena",
        "group_size": nq // nkv, "nq": nq, "nkv": nkv, "head_dim": d, "rotary_dim": rot,
        "seq_len": seq_len, "past_len": past_len, "past_stride": past_stride,
        "regime": "prefill" if seq_len > 1 else "decode",
        "kv_arena_env": os.environ.get("ONNXRUNTIME_EP_VULKAN_KV_ARENA", ""),
        "device_memory_env": os.environ.get("ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY", ""),
        "bind_outputs_env": os.environ.get("ONNXRUNTIME_EP_VULKAN_BIND_OUTPUTS", ""),
    }
    if rec["kv_arena_env"] != "1" or rec["device_memory_env"] != "1" \
            or rec["bind_outputs_env"] not in ("", "1"):
        rec["verdict"] = "UNMEASURED(arena_env_not_set)"
        return rec

    dev = _ep_device()
    if dev is None:
        rec["verdict"] = "UNMEASURED(no_ep_device)"
        return rec
    rec["device_name"] = dev.ep_metadata.get("vulkan.device_name")

    model = build_model(nq=nq, nkv=nkv, d=d, rot=rot, past_stride=past_stride,
                        seq_len=seq_len, present_declared=False, symbolic_extents=True)
    feeds = make_feeds(nq=nq, nkv=nkv, d=d, rot=rot, past_stride=past_stride,
                       seq_len=seq_len, past_len=past_len, seed=arm.get("seed", 0))

    sess = _sess(model, m.EP_PROVIDERS, DEVICE_INDEX)
    if m.EP_NAME not in sess.get_providers():
        rec["verdict"] = "UNMEASURED(ep_absent)"
        return rec
    mi = dev.memory_info(ort.OrtDeviceMemoryType.DEFAULT)
    if mi is None:
        rec["verdict"] = "UNMEASURED(no_default_allocator)"
        return rec

    runs: list[list[np.ndarray]] = []
    try:
        for _ in range(repeats):
            ovs = {}
            for name, arr in (("key", feeds["past_key"]), ("value", feeds["past_value"])):
                ov = ort.OrtValue.ortvalue_from_shape_and_type(
                    list(arr.shape), np.float16, memory_info=mi)
                ov.update_inplace(np.ascontiguousarray(arr))
                ovs[name] = ov
            b = sess.io_binding()
            for n in ("packed_qkv", "seqlens_k", "total_seq", "cos_cache", "sin_cache"):
                b.bind_cpu_input(n, np.ascontiguousarray(feeds[n]))
            b.bind_ortvalue_input("past_key", ovs["key"])
            b.bind_ortvalue_input("past_value", ovs["value"])
            # `get_outputs()` returns bound outputs in **binding order**, not in the
            # graph's output order.  Binding attn_out last and then reading index 0 gave
            # present_key wearing attn_out's name and a SHAPE_OR_DTYPE_MISMATCH that
            # looked like a divergence.  attn_out is bound first and read by index 0.
            b.bind_output("attn_out")
            # The declaration: same OrtValue in and out.
            b.bind_ortvalue_output("present_key", ovs["key"])
            b.bind_ortvalue_output("present_value", ovs["value"])
            sess.run_with_iobinding(b)
            attn = np.asarray(b.get_outputs()[0].numpy())
            assert attn.shape == (feeds["packed_qkv"].shape[0], seq_len, nq * d), (
                f"binding-order guard: attn_out read back as {attn.shape}"
            )
            runs.append([attn, np.asarray(ovs["key"].numpy()),
                         np.asarray(ovs["value"].numpy())])
    except Exception as exc:
        rec["run_error"] = repr(exc)
        rec["verdict"] = "UNMEASURED(arena_refused)"
        return rec

    cpu_model = build_model(nq=nq, nkv=nkv, d=d, rot=rot, past_stride=past_stride,
                            seq_len=seq_len, present_declared=False,
                            symbolic_extents=True)
    cpu_out = _sess(cpu_model, ["CPUExecutionProvider"]).run(None, feeds)

    vk_out = runs[0]
    rec["vk_shapes"] = [list(a.shape) for a in vk_out]
    rec["cpu_shapes"] = [list(a.shape) for a in cpu_out]
    rec["repeats"] = repeats

    identical, differing = True, []
    for i in range(1, repeats):
        eq, idx = m.outputs_bit_equal(runs[0], runs[i])
        if not eq:
            identical = False
            differing.extend(idx)
    rec["cross_run_identical"] = identical
    rec["cross_run_differing_outputs"] = sorted(set(differing))

    total = past_len + seq_len
    # The arena buffer is a capacity; the contract covers rows [0, total).
    sliced_vk = [vk_out[0], vk_out[1][:, :, :total, :], vk_out[2][:, :, :total, :]]
    sliced_cpu = [cpu_out[0],
                  cpu_out[1][:, :, :total, :] if cpu_out[1].shape[2] >= total else cpu_out[1],
                  cpu_out[2][:, :, :total, :] if cpu_out[2].shape[2] >= total else cpu_out[2]]
    outcome, facts = m.compare_all_outputs_to_cpu(sliced_vk, sliced_cpu)
    rec["cpu_comparison"] = outcome
    rec["per_output"] = facts["per_output"]
    rec["oracle_failing_indices"] = facts["oracle_failing_indices"]
    rec["oracle_degenerate_indices"] = facts["oracle_degenerate_indices"]

    ref = gqa_reference_f64(feeds, nq=nq, nkv=nkv, d=d, rot=rot,
                            seq_len=seq_len, past_len=past_len)
    f64 = []
    for i, name in enumerate(["attn_out", "present_key", "present_value"]):
        r = ref[i] if i == 0 else ref[i][:, :, :total, :]
        v, c = sliced_vk[i], sliced_cpu[i]
        if v.shape != r.shape or c.shape != r.shape:
            f64.append({"output": name, "status": "SHAPE_MISMATCH",
                        "vk": list(v.shape), "cpu": list(c.shape), "ref": list(r.shape)})
            continue
        dv = np.abs(v.astype(np.float64) - r)
        dc = np.abs(c.astype(np.float64) - r)
        further = ("neither (equal)" if float(dv.max()) == float(dc.max())
                   else ("vulkan" if float(dv.max()) > float(dc.max()) else "cpu"))
        f64.append({"output": name,
                    "vk_max_abs_from_f64": float(dv.max()),
                    "cpu_max_abs_from_f64": float(dc.max()),
                    "which_is_further_from_true": further})
    rec["f64_reference"] = f64

    pk_in = feeds["past_key"][:, :, :past_len, :]
    pv_in = feeds["past_value"][:, :, :past_len, :]
    rec["past_region_of_present_bit_intact"] = bool(
        np.array_equal(vk_out[1][:, :, :past_len, :].view(np.uint16), pk_in.view(np.uint16))
        and np.array_equal(vk_out[2][:, :, :past_len, :].view(np.uint16), pv_in.view(np.uint16))
    )
    rec["past_region_rows_checked"] = int(past_len)
    tail_intact = None
    if vk_out[1].shape[2] > total:
        tail_intact = bool(np.all(vk_out[1][:, :, total:, :] == np.float16(0.5))
                           and np.all(vk_out[2][:, :, total:, :] == np.float16(-0.25)))
    rec["arena_tail_poison_intact"] = tail_intact
    rec["verdict"] = derive_verdict(rec)
    return rec


def derive_verdict(rec: dict) -> str:
    if not rec.get("cross_run_identical", True):
        return "RACE(cross_run_unstable)"
    if not rec.get("past_region_of_present_bit_intact", True):
        return "CORRUPT(past_region_of_present_overwritten)"
    if rec.get("arena_tail_poison_intact") is False:
        return "CORRUPT(arena_tail_overwritten)"
    if rec.get("cpu_comparison") == m.COMPARISON_AGREE:
        return "AGREE"
    if rec.get("cpu_comparison") == m.COMPARISON_NOT_PERFORMED:
        return "UNMEASURED(degenerate_output)"
    return "DIVERGENT"


ARMS: list[dict] = [
    # name, shape, seq_len, past_len, past_stride, present_declared
    dict(name="g1_decode_growing", shape=SMALL_G1, seq_len=1, past_len=16,
         past_stride=16, present_declared=True),
    dict(name="g4_decode_growing", shape=SMALL_G4, seq_len=1, past_len=16,
         past_stride=16, present_declared=True),
    dict(name="g1_prefill_growing", shape=SMALL_G1, seq_len=8, past_len=16,
         past_stride=16, present_declared=True),
    dict(name="g4_prefill_growing", shape=SMALL_G4, seq_len=8, past_len=16,
         past_stride=16, present_declared=True),
    dict(name="g4_prefill_growing_frompast0", shape=SMALL_G4, seq_len=8, past_len=0,
         past_stride=0, present_declared=True),
    # Arena arms: present extent symbolic, past extent is a CAPACITY larger than the
    # true past length, tail poisoned.
    dict(name="g1_decode_arena", shape=SMALL_G1, seq_len=1, past_len=16,
         past_stride=64, present_declared=False),
    dict(name="g4_decode_arena", shape=SMALL_G4, seq_len=1, past_len=16,
         past_stride=64, present_declared=False),
    dict(name="g1_prefill_arena", shape=SMALL_G1, seq_len=8, past_len=16,
         past_stride=64, present_declared=False),
    dict(name="g4_prefill_arena", shape=SMALL_G4, seq_len=8, past_len=16,
         past_stride=64, present_declared=False),
    # Llama-3 8B's real attention shape.
    dict(name="llama3_g1_decode_growing", shape=LLAMA3_G1, seq_len=1, past_len=16,
         past_stride=16, present_declared=True),
    dict(name="llama3_g4_decode_growing", shape=LLAMA3, seq_len=1, past_len=16,
         past_stride=16, present_declared=True),
    dict(name="llama3_g4_prefill_growing", shape=LLAMA3, seq_len=8, past_len=16,
         past_stride=16, present_declared=True),
    dict(name="llama3_g4_prefill_arena", shape=LLAMA3, seq_len=8, past_len=16,
         past_stride=64, present_declared=False),
]


def register_ep() -> str:
    """Register the EP plugin. Returns the resolved library path.

    Raises rather than falling back: an unregistered EP produces a CPU-vs-CPU comparison
    that agrees perfectly, which is the most convincing possible way to measure nothing.
    """
    lib = os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB")
    if not lib or not Path(lib).is_file():
        raise SystemExit(
            "ONNXRUNTIME_VULKAN_EP_LIB is not set or not found; refusing to run a probe "
            "that would silently measure CPU against CPU."
        )
    try:
        ort.register_execution_provider_library(m.EP_NAME, str(Path(lib).resolve()))
    except Exception as exc:
        if "already registered" not in str(exc):
            raise
    return str(Path(lib).resolve())


def selftest() -> int:
    """Device-free liveness checks.  An instrument never seen in its positive state has
    no demonstrated positive state.

    Every arm below asserts that some check in this file **can go red**.  They run
    without a GPU and without the EP.
    """
    failures: list[str] = []

    def expect(name: str, got, want) -> None:
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")

    base = dict(cross_run_identical=True, past_region_of_present_bit_intact=True,
                arena_tail_poison_intact=True, cpu_comparison=m.COMPARISON_AGREE)
    expect("healthy", derive_verdict(dict(base)), "AGREE")
    expect("race", derive_verdict({**base, "cross_run_identical": False}),
           "RACE(cross_run_unstable)")
    expect("corrupt_past", derive_verdict({**base,
                                           "past_region_of_present_bit_intact": False}),
           "CORRUPT(past_region_of_present_overwritten)")
    expect("corrupt_tail", derive_verdict({**base, "arena_tail_poison_intact": False}),
           "CORRUPT(arena_tail_overwritten)")
    expect("degenerate", derive_verdict({**base,
                                         "cpu_comparison": m.COMPARISON_NOT_PERFORMED}),
           "UNMEASURED(degenerate_output)")
    expect("divergent", derive_verdict({**base,
                                        "cpu_comparison": m.COMPARISON_DISAGREE}),
           "DIVERGENT")
    # A CORRUPT verdict must outrank a passing tolerance comparison: silent corruption
    # that happens to stay inside fp16 tolerance is the failure shape this probe exists
    # for, and a verdict that reported AGREE there would be the whole point missed.
    expect("corrupt_outranks_agree",
           derive_verdict({**base, "past_region_of_present_bit_intact": False,
                           "cpu_comparison": m.COMPARISON_AGREE}),
           "CORRUPT(past_region_of_present_overwritten)")

    # The float64 reference must actually implement grouping.  If it silently ignored
    # `nkv` -- reading a per-query-head K/V slice instead of a shared one -- it would
    # agree with a G=1 implementation and could never convict a grouping defect.  So:
    # the SAME packed tensor read at G=4 and at G=1 must give DIFFERENT answers.
    nq, nkv, d, rot, S, P = 8, 2, 32, 16, 4, 3
    feeds = make_feeds(nq=nq, nkv=nkv, d=d, rot=rot, past_stride=P, seq_len=S,
                       past_len=P, seed=11)
    a_g4 = gqa_reference_f64(feeds, nq=nq, nkv=nkv, d=d, rot=rot, seq_len=S, past_len=P)[0]
    # Same bytes, read as if every query head had its own KV head.  Needs a packed
    # tensor wide enough for Nkv = Nq; pad by tiling so the read is in-bounds.
    wide = np.concatenate(
        [feeds["packed_qkv"]] * 3, axis=2
    )[:, :, : (nq + 2 * nq) * d].astype(np.float16)
    feeds_g1 = dict(feeds)
    feeds_g1["packed_qkv"] = wide
    feeds_g1["past_key"] = np.repeat(feeds["past_key"], nq // nkv, axis=1)
    feeds_g1["past_value"] = np.repeat(feeds["past_value"], nq // nkv, axis=1)
    a_g1 = gqa_reference_f64(feeds_g1, nq=nq, nkv=nq, d=d, rot=rot, seq_len=S, past_len=P)[0]
    if np.allclose(a_g4, a_g1):
        failures.append(
            "f64 reference is blind to grouping: G=4 and G=1 over the same bytes agree"
        )

    # And it must be sensitive to the KV cache at all -- a reference that ignored `past`
    # would agree with a kernel that dropped it.
    feeds_b = dict(feeds)
    pk = feeds["past_key"].copy()
    pk[:, :, 0, :] = np.float16(0.9)
    feeds_b["past_key"] = pk
    a_b = gqa_reference_f64(feeds_b, nq=nq, nkv=nkv, d=d, rot=rot, seq_len=S, past_len=P)[0]
    if np.allclose(a_g4, a_b):
        failures.append("f64 reference ignores past_key: perturbing it changed nothing")

    for f in failures:
        print("SELFTEST FAIL:", f)
    print(f"selftest: {8 + 2 - len(failures)}/{8 + 2} checks passed")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench/results/gqa_grouping.json")
    ap.add_argument("--only", default="", help="substring filter on arm name")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--selftest", action="store_true",
                    help="device-free liveness checks; no EP, no GPU")
    ap.add_argument("--device-index", type=int, default=None,
                    help="ep.device_index to pin (0 = discrete-first default)")
    args = ap.parse_args()

    global DEVICE_INDEX
    DEVICE_INDEX = args.device_index

    if args.selftest:
        return selftest()

    lib_path = register_ep()
    device_index = os.environ.get("VULKAN_DEVICE_INDEX", "")
    record: dict = {
        "probe": "gqa_grouping",
        "ep_library": lib_path,
        "question_answered": "node-level correctness of GroupQueryAttention at Nq/Nkv != 1",
        "question_not_answered": (
            "end-to-end behaviour of a full grouped model (no Llama-3 export was run; "
            "an 8 GB discrete GPU on this box will not hold one)"
        ),
        "hypotheses": {
            "H_DISJOINT_HOLDS": "arena aliasing argument is invariant in group size",
            "H_DISJOINT_FAILS": "grouped arena corrupts the t < past_len region of present",
            "H_DUP_BENIGN": "the G duplicate present writers write identical bits",
            "H_DUP_RACES": "the G duplicate present writers race; cross-run unstable",
        },
        "vulkan_device_index_env": device_index,
        "arms": [],
    }
    for arm in ARMS:
        if args.only and args.only not in arm["name"]:
            continue
        arena = not arm["present_declared"]
        prev = os.environ.get("ONNXRUNTIME_EP_VULKAN_KV_ARENA")
        if arena:
            os.environ["ONNXRUNTIME_EP_VULKAN_KV_ARENA"] = "1"
        elif prev is not None:
            del os.environ["ONNXRUNTIME_EP_VULKAN_KV_ARENA"]
        try:
            rec = run_arena_arm(arm, repeats=args.repeats) if arena \
                else run_arm(arm, repeats=args.repeats)
        finally:
            if prev is None:
                os.environ.pop("ONNXRUNTIME_EP_VULKAN_KV_ARENA", None)
            else:
                os.environ["ONNXRUNTIME_EP_VULKAN_KV_ARENA"] = prev
        record["arms"].append(rec)
        print(f"{rec['arm']:<34} G={rec.get('group_size')} "
              f"{rec.get('regime','?'):<7} verdict={rec['verdict']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Route off the counters, never off the env var that requested it.  `kv_convention`
    # is derived in the EP from push-constant fields 24 and 28 -- the same comparison the
    # shader's own `copy_leader` makes -- so it reports the convention the dispatch
    # actually ran under, not the one the caller asked for.
    cf = os.environ.get("ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE")
    if cf and Path(cf).is_file():
        try:
            counters = json.loads(Path(cf).read_text(encoding="utf-8"))
            record["kv_convention_observed"] = counters.get("kv_cache_convention")
            record["running_device_names_observed"] = counters.get("running_device_names")
            record["counters_file"] = cf
        except Exception as exc:  # noqa: BLE001
            record["kv_convention_observed"] = f"ERROR(instrument): {exc!r}"
    else:
        record["kv_convention_observed"] = "UNOBSERVED(no counters file)"
    record["device_names_seen"] = sorted(
        {a["device_name"] for a in record["arms"] if a.get("device_name")}
    )
    record["device_index_requested"] = DEVICE_INDEX
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    bad = [a["arm"] for a in record["arms"] if a["verdict"] not in ("AGREE",)]
    if bad:
        print("NOT AGREE: " + ", ".join(bad))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
