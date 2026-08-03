#!/usr/bin/env python3
"""KV-cache traffic for Phi-3.5 decode, derived from the graph and the shader. No clock.

THE QUESTION THIS ANSWERS
=========================
"At decode, each GQA node reads the entire history to produce one token. Is that irreducible
given the algorithm, or is it an artefact of how we lay the cache out?"

**The answer is: irreducible in ELEMENTS, not in BYTES, and we are currently moving ~3x the
irreducible element count anyway.** Those are three separate claims and they have three
separate owners, so they are reported separately:

  1. ELEMENTS ARE IRREDUCIBLE.  Attention at position L is a softmax-weighted sum over every
     past key and every past value.  Every one of them carries non-zero weight in general
     (softmax has no zeros), so no layout, no residency policy and no read pattern can avoid
     touching all of them.  This is a property of the operator, not of our implementation.
     Anything that claims otherwise is claiming an approximation -- and an approximation is a
     different kernel that must be proved against the reference like any other.

  2. BYTES ARE NOT.  bytes = elements x precision, and precision is a choice.  The cache is
     fp16 today.  int8 is 2x and int4 is 4x, and no layout change can match that, because
     layout cannot beat a factor of one on a stream that is already read exactly once.
     **This is the lever, and it is the only one with an order of magnitude in it.**

  3. WE ARE AT ~3x, AND THAT PART IS OURS.  Two amplifications sit on top of the irreducible
     term, both introduced by us, both measured here rather than asserted:

       (a) THE PRESENT-COPY ROUND-TRIP.  Under the growing-cache convention the graph declares
           `past` at `past_sequence_length` and `present` at `total_sequence_length` -- two
           distinct allocations -- so every token we read the whole past cache a SECOND time
           and write a whole present cache, purely to relocate bytes that did not change.
           That is +2 units on top of the 1 unit attention genuinely needs.
       (b) GROUP-SIZE AMPLIFICATION.  Our shader runs one invocation per QUERY head and each
           invocation reads the full history of its KV head, so attention issues
           `Nq x L x D` loads against a cache holding `Nkv x L x D`.  The factor is `Nq/Nkv`.

     **(b) is exactly 1.00 on Phi-3.5 and that is the trap, not the result.** Phi-3.5 declares
     `num_heads = kv_num_heads = 32`: it does no grouping at all despite the operator's name.
     A model that actually groups pays the factor.  Reporting only the Phi-3.5 number would
     certify a kernel as efficient using the one model on which its defect is invisible.

WHY THIS IS A COUNT AND NOT A MEASUREMENT
=========================================
Every figure here is read off the graph's declared shapes and the shader's loop bounds.  No
clock, no device state, no dependence on whether the box is quiet.  That is now the operating
condition, not a bad night, and per the roofline probe's argument a bound you can sign beats a
measurement you cannot.

WHAT WOULD FALSIFY EACH CLAIM
=============================
* The copy round-trip is claimed from TWO independent readings that must agree: the graph's
  declared past/present extents, and the shader's own guard `pc.present_len != pc.past_stride`.
  If the graph said `past == present` the guard would not fire and the 3x would be a 1x.  The
  probe checks both and refuses to report if they disagree.
* The group factor is `Nq/Nkv` read from the node attributes; if the shader stopped iterating
  per query head the factor would be 1 regardless of the model.  The probe reads the shader
  source to confirm the loop is still per-query-head.

Run:  python bench/results/probe_kv_traffic.py
"""

from __future__ import annotations

import json
import pathlib

import sys

import onnx

ROOT = pathlib.Path(__file__).resolve().parents[2]

MODEL = pathlib.Path(
    r"C:\Users\justinchu\.foundry\cache\models\Microsoft"
    r"\Phi-3.5-mini-instruct-cuda-gpu\cuda-int4-rtn-block-32"
    r"\phi-3.5-mini-instruct-cuda-int4-rtn-block-32.onnx"
)

SHADER = ROOT / "rust" / "shaders" / "glsl" / "gqa_f16.comp"

#: Weight stream, established by probe_roofline.py and unchanged: int4 blobs + fp16 scales.
WEIGHT_BYTES = 1_861_189_632 + 232_648_704  # 1996.8 MiB

#: RTX 4060 Laptop (AD107), 128-bit GDDR6 @ 16 Gbps.  Spec peak, decimal GB.
PEAK_GB_S = 256.0

ELEM_BYTES = {onnx.TensorProto.FLOAT16: 2, onnx.TensorProto.FLOAT: 4}


def _dims(vi) -> list:
    return [(d.dim_param or d.dim_value) for d in vi.type.tensor_type.shape.dim]


def gqa_census(model_path: pathlib.Path) -> dict:
    """Every GQA geometry fact, read off the graph.  Refuses on any inconsistency."""
    m = onnx.load(str(model_path), load_external_data=False)
    g = m.graph
    vi = {v.name: v for v in list(g.input) + list(g.output) + list(g.value_info)}

    nodes = [n for n in g.node if n.op_type == "GroupQueryAttention"]
    if not nodes:
        raise SystemExit("REFUSE: no GroupQueryAttention nodes in graph")

    shapes: set = set()
    for n in nodes:
        attrs = {a.name: a.i for a in n.attribute if a.type == onnx.AttributeProto.INT}
        past_k = vi.get(n.input[3])
        pres_k = vi.get(n.output[1])
        if past_k is None or pres_k is None:
            raise SystemExit(f"REFUSE: {n.name} past/present not in value_info")
        pd, sd = _dims(past_k), _dims(pres_k)
        if len(pd) != 4 or len(sd) != 4:
            raise SystemExit(f"REFUSE: {n.name} past/present not rank 4: {pd} {sd}")
        et = past_k.type.tensor_type.elem_type
        if et not in ELEM_BYTES:
            raise SystemExit(f"REFUSE: unhandled KV elem_type {et}")
        shapes.add(
            (
                attrs["num_heads"],
                attrs["kv_num_heads"],
                pd[1],
                pd[3],
                str(pd[2]),
                str(sd[2]),
                et,
            )
        )

    if len(shapes) != 1:
        raise SystemExit(f"REFUSE: GQA layers are not uniform: {shapes}")
    nq, nkv, heads_dim, head_dim, past_sym, present_sym, et = shapes.pop()
    if heads_dim != nkv:
        raise SystemExit(f"REFUSE: cache head count {heads_dim} != kv_num_heads {nkv}")

    return {
        "layers": len(nodes),
        "num_heads": nq,
        "kv_num_heads": nkv,
        "head_dim": head_dim,
        "elem_bytes": ELEM_BYTES[et],
        "past_extent_symbol": past_sym,
        "present_extent_symbol": present_sym,
        #: The graph's own answer to "are past and present the same allocation?".
        "growing_cache": past_sym != present_sym,
    }


def shader_facts(path: pathlib.Path) -> dict:
    """Two facts read out of the kernel itself, so the model is not asserted from the graph
    alone.  A byte model that never looks at the code that moves the bytes is a description of
    a program nobody wrote."""
    src = path.read_text(encoding="utf-8")
    return {
        #: step 3a fires only when the two extents differ -- the shader's own reading of the
        #: same fact the graph declares.  The two must agree or one of them is stale.
        "copy_guarded_on_extent_mismatch": "pc.present_len != pc.past_stride" in src,
        #: one invocation per (b, h, s) with h over QUERY heads: the group amplification.
        #: Read from the actual gid decode, not a guessed identifier -- an earlier version of
        #: this detector matched on a bound that does not exist in the file and reported
        #: False for a fact that is plainly true. A detector that can be wrong in the
        #: reassuring direction is worse than no detector, so both halves are required:
        #: the grid is sized by Nq, and the KV head is derived FROM the query head.
        "iterates_per_query_head": (
            "gid / (Nq * S)" in src and "h / (Nq / Nkv)" in src
        ),
        "local_size_x_1": "local_size_x = 1" in src,
        #: the past->present copy writes from inside the attention loop rather than from a
        #: standalone loop that re-reads the cache. Both halves are checked: the fused write
        #: is present AND the leader flag it depends on exists.
        "copy_fused_into_attention_loop": (
            "wr_presk(copy_dst + t * D + d, kv)" in src
            and "wr_presv(copy_dst + t * D + d, vv)" in src
            and "bool copy_leader" in src
        ),
    }


def kv_bytes_per_token(c: dict, past_len: int, fused_copy: bool = True) -> dict:
    """Bytes of KV touched to produce ONE decode token at context `past_len`.

    `unit` is the irreducible term: the cache itself, read once.  Everything else is a
    multiple of it, and every multiple has a named owner.

    `fused_copy` selects the two shader generations. Before fusion the past→present copy was
    a standalone loop that re-read the entire past cache; after fusion the copy leader writes
    each element at the moment step 4 loads it for the attention sum, so the copy costs no
    loads at all.  The WRITE survives fusion in both generations because ORT's contract makes
    the past region of `present` a required output, and no arrangement of ours avoids it.
    """
    unit = c["layers"] * 2 * c["kv_num_heads"] * past_len * c["head_dim"] * c["elem_bytes"]
    group = c["num_heads"] // c["kv_num_heads"]

    attention_read = unit * group  # (b) group-size amplification
    copy_read = 0 if fused_copy else (unit if c["growing_cache"] else 0)
    copy_write = unit if c["growing_cache"] else 0
    return {
        "past_len": past_len,
        "cache_resident_bytes": unit,
        "attention_read_bytes": attention_read,
        "copy_read_bytes": copy_read,
        "copy_write_bytes": copy_write,
        "kv_total_bytes": attention_read + copy_read + copy_write,
        "amplification_over_cache": (attention_read + copy_read + copy_write) / unit
        if unit
        else 0.0,
    }


def alias_feasibility(c: dict, past_len: int) -> dict:
    """Can `present` be bound onto `past`'s allocation, removing the copy WRITE too?

    This is the 2.21x lever, and it is decided by arithmetic on the graph's own declared
    extents rather than by anything about our kernel.  `bind_aliased_output` returns the
    INPUT's buffer for the output (vk/session.rs), so the alias requires `present` to fit
    inside `past`'s allocation.  On a growing cache it does not: `present` is `past + seq_len`
    and is therefore strictly larger, for every seq_len >= 1 and every past_len.

    There is no seq_len at which the two coincide, so this is not another quantity whose two
    definitions agree on the test set -- it is infeasible everywhere in this convention.
    """
    per_token = c["layers"] * 2 * c["kv_num_heads"] * c["head_dim"] * c["elem_bytes"]
    return {
        "past_bytes": per_token * past_len,
        "present_bytes_at_seq_len_1": per_token * (past_len + 1),
        "present_fits_in_past": (past_len + 1) <= past_len,  # False for all past_len
        "alias_feasible": (not c["growing_cache"]),
        "blocked_by": None
        if not c["growing_cache"]
        else "graph declares present at total_sequence_length > past_sequence_length; "
        "an alias cannot place the larger tensor in the smaller allocation",
    }


def main() -> int:
    c = gqa_census(MODEL)
    sf = shader_facts(SHADER)

    # The two independent readings of the same fact must agree, and every shader fact the
    # byte model depends on must be TRUE, or the model is describing a program nobody wrote.
    if c["growing_cache"] and not sf["copy_guarded_on_extent_mismatch"]:
        raise SystemExit(
            "REFUSE: graph declares a growing cache but the shader has no extent-mismatch "
            "guard -- one of the two readings is stale and the copy term cannot be signed"
        )
    if not sf["iterates_per_query_head"]:
        raise SystemExit(
            "REFUSE: could not confirm the kernel dispatches one invocation per query head; "
            "the group-amplification term is unsupported"
        )

    group = c["num_heads"] // c["kv_num_heads"]
    fused = sf["copy_fused_into_attention_loop"]

    print()
    print("  GQA geometry, read off the graph")
    print(f"    layers                {c['layers']}")
    print(f"    num_heads (Nq)        {c['num_heads']}")
    print(f"    kv_num_heads (Nkv)    {c['kv_num_heads']}")
    print(f"    group_size Nq/Nkv     {group}   <- 1 means this model does NO grouping")
    print(f"    head_dim              {c['head_dim']}")
    print(f"    kv element bytes      {c['elem_bytes']} (fp16)")
    print(f"    past extent symbol    {c['past_extent_symbol']!r}")
    print(f"    present extent symbol {c['present_extent_symbol']!r}")
    print(f"    growing cache         {c['growing_cache']}  (past and present are distinct)")
    print()
    print("  shader facts, read off gqa_f16.comp")
    for k, v in sf.items():
        print(f"    {k:38s} {v}")

    rows = [kv_bytes_per_token(c, L, fused) for L in (0, 512, 2048, 8192, 32768, 131072)]

    print()
    print(f"  copy fused into attention loop: {fused}")
    print("  Per decode token.  MiB.  weights = 1996.8 MiB, irreducible, amplification 1.000000")
    print(
        f"    {'past_len':>8}  {'cache':>9}  {'attn rd':>9}  {'copy rd':>9}  {'copy wr':>9}"
        f"  {'KV tot':>9}  {'KV %':>7}  {'floor ms':>9}"
    )
    for r in rows:
        tot = WEIGHT_BYTES + r["kv_total_bytes"]
        pct = 100.0 * r["kv_total_bytes"] / tot
        floor_ms = tot / (PEAK_GB_S * 1e9) * 1e3
        print(
            f"    {r['past_len']:>8}  {r['cache_resident_bytes']/2**20:>9.1f}"
            f"  {r['attention_read_bytes']/2**20:>9.1f}  {r['copy_read_bytes']/2**20:>9.1f}"
            f"  {r['copy_write_bytes']/2**20:>9.1f}  {r['kv_total_bytes']/2**20:>9.1f}"
            f"  {pct:>6.1f}%  {floor_ms:>9.2f}"
        )

    # THE LEVER THAT IS NOT AVAILABLE, and why -- reported rather than worked around.
    L = 8192
    af = alias_feasibility(c, L)
    print()
    print(f"  Can `present` alias `past`, removing the copy WRITE?  at past_len={L}")
    print(f"    past bytes                    {af['past_bytes']/2**20:9.1f} MiB")
    print(f"    present bytes (seq_len=1)     {af['present_bytes_at_seq_len_1']/2**20:9.1f} MiB")
    print(f"    present fits in past          {af['present_fits_in_past']}")
    print(f"    alias feasible                {af['alias_feasible']}")
    if af["blocked_by"]:
        print(f"    blocked by                    {af['blocked_by']}")

    # The counterfactual table.  `fused` is what we now do; the rest are named levers.
    before = kv_bytes_per_token(c, L, fused_copy=False)
    base = kv_bytes_per_token(c, L, fused)
    unit = base["cache_resident_bytes"]
    levers = [
        ("standalone copy (previous shader)", before["kv_total_bytes"]),
        ("fused copy (this shader)", base["kv_total_bytes"]),
        ("shared buffer -- NOT AVAILABLE here", unit * group),
        ("+ int8 KV cache", unit * group // 2),
        ("+ int4 KV cache", unit * group // 4),
    ]
    print()
    print(f"  Levers, at past_len={L}.  MiB per token, and the floor they imply.")
    print("  Speedups are against the previous shader, so the landed change is visible.")
    ref = WEIGHT_BYTES + before["kv_total_bytes"]
    for name, kvb in levers:
        tot = WEIGHT_BYTES + kvb
        print(
            f"    {name:36s} KV {kvb/2**20:8.1f} MiB   total {tot/2**20:8.1f} MiB"
            f"   floor {tot/(PEAK_GB_S*1e9)*1e3:6.2f} ms   {ref/tot:5.3f}x"
        )

    # ── The link axis ────────────────────────────────────────────────────────
    # Everything above is a DRAM floor: bytes moved between the device and its own memory.
    # MEASURED 2026-08-02 (`probe_kv_bytes_earned.py`, this build), the KV cache also crosses
    # the host<->device boundary every inference, on both axes and exactly:
    #
    #   readback  393,216 B per past token   ratio 1.000000, linearity spread 0.000000
    #   upload    393,216 B per past token   ratio 1.000000   <- newly observable
    #
    # The upload term read UNOBSERVABLE until today, and the reason is a defect rather than an
    # unseen path: the EP's input cache was keyed on `(cpu_ptr, byte_size)` and was skipping the
    # past-KV upload entirely, serving the first inference's KV to every later one. Correcting
    # the predicate did not add traffic; it stopped hiding a cost that was being paid in wrong
    # answers. Niobe's instrument was right and the build was wrong.
    #
    # No link rate is asserted here. This box's real host<->device rate has not been measured,
    # and the whole point of the exercise is that the DRAM figure was quoted against the wrong
    # constraint. Bytes are reported; a rate is not invented.
    link_per_past_token = 2 * 393_216
    link_bytes = link_per_past_token * L
    dram_bytes = WEIGHT_BYTES + base["kv_total_bytes"]
    print()
    print(f"  LINK AXIS at past_len={L} (measured, both directions, per inference)")
    print(f"    host->device upload           {link_bytes/2/2**20:9.1f} MiB")
    print(f"    device->host readback         {link_bytes/2/2**20:9.1f} MiB")
    print(f"    link total                    {link_bytes/2**20:9.1f} MiB")
    print(f"    DRAM total (table above)      {dram_bytes/2**20:9.1f} MiB")
    print(f"    link / DRAM                   {link_bytes/dram_bytes:9.3f}")
    print("    link rate                     UNMEASURED on this box -- no floor is quoted")
    print(
        "    The DRAM levers above are admissible at every extent and BINDING only where the"
    )
    print(
        "    link term is smaller. The fused-copy change is a DRAM reduction; it moves no byte"
    )
    print("    off this axis, so at the contexts where KV matters it is not the constraint.")


    print()
    print("  Group amplification is invisible here and is NOT invisible in general.")
    copy_units = (1 if fused else 2) if c["growing_cache"] else 0
    for gs in (1, 4, 8):
        print(
            f"    Nq/Nkv = {gs}:  attention reads {gs}x the cache, total KV traffic"
            f" {gs + copy_units}x the cache per token"
            f"  (was {gs + copy_units + 1}x, so the fusion is"
            f" {(gs + copy_units + 1) / (gs + copy_units):.3f}x there)"
            + ("   <- Phi-3.5" if gs == group else "")
        )

    out = {
        "geometry": c,
        "shader_facts": sf,
        "weight_bytes": WEIGHT_BYTES,
        "peak_gb_s": PEAK_GB_S,
        "per_token": rows,
        "answer_read_is_irreducible_in_elements": True,
        "answer_read_is_reducible_in_bytes": True,
        "our_amplification_over_cache": kv_bytes_per_token(c, 8192, fused)["amplification_over_cache"],
        "copy_fused": fused,
        "alias_feasibility": alias_feasibility(c, 8192),
        "link_axis": {
            "measured_by": "bench/results/probe_kv_bytes_earned.py",
            "upload_bytes_per_past_token": 393_216,
            "readback_bytes_per_past_token": 393_216,
            "ratio_upload": 1.0,
            "ratio_readback": 1.0,
            "link_rate_gb_s": None,
            "link_rate_state": "UNMEASURED",
            "note": (
                "The upload term read UNOBSERVABLE before 2026-08-02 because the EP's input "
                "cache was keyed on (cpu_ptr, byte_size) and skipped the past-KV upload, "
                "serving stale KV. Correcting the predicate made the cost visible; it did not "
                "create it. No link rate is asserted: this box's host<->device rate is not "
                "measured, and quoting a floor against an unmeasured constraint is the error "
                "this section exists to correct."
            ),
            "binding": (
                "The DRAM levers in `per_token` are admissible at every extent and binding only "
                "where the link term is smaller. The fused-copy change (1.377x at 8192) is a "
                "DRAM reduction and moves no byte off the link axis."
            ),
        },
    }
    rec = ROOT / "bench" / "results" / "kv_traffic.json"
    rec.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  record: {rec}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
