#!/usr/bin/env python3
"""The KV lever ledger, derived from artifacts at run time. No literals, no clock.

WHY THIS EXISTS
===============
The KV work of the last several rounds has been sequenced against a ledger quoting
**2.21x (present copy) / 3.17x (int8) / 4.06x (int4)**. Those three numbers have no derivation
anywhere in this tree. `bench/results/kv-int8-budget-prediction.md` Section 3 recorded, *before*
the first int8 run, that none of them reproduces from any artifact on any baseline that could be
constructed -- footprint, modelled stream, KV-only, with or without the present write -- and that
what the artifacts support for int8/int4 is **1.40x / 1.76x** on the footprint. They were quoted
to the user more than once. This file is the replacement, and it is a **generator** rather than a
table, for the reason Niobe established: a measurement restated as a literal under a docstring
naming its source stops moving when its source moves.

THE DEFECT THE OLD LEDGER HAD, WHICH IS NOT "WRONG ARITHMETIC"
==============================================================
A KV lever is a ratio, and a ratio needs a **named denominator**. The old ledger had one column
of "x" holding numbers from at least three different axes:

  * LINK      -- bytes crossing host<->device per past token (staging traffic)
  * DRAM      -- bytes the kernel's own loads and stores name, per token
  * FOOTPRINT -- bytes of device allocation held at a context length

A saving on one of those does not compose with a saving on another, and an elimination (the round
trip goes to *zero*) has no multiplier at all -- 393,216 -> 0 is not "N x", it is a removal, and
writing it as a finite ratio requires inventing a denominator. Every lever below therefore carries
its axis and its baseline **by name**, and levers on different axes are never multiplied together.

PROVENANCE
==========
Per Niobe's rule, every quantity carries a class:

  * SPECIFICATION -- a fact about a named part, published by its maker
  * MEASUREMENT   -- a fact about *this* graph / module / run, re-derived here every time
  * MODEL         -- an analytic construction; never quotable as a measurement

The byte figures for int8/int4 are **MODEL** and are marked so: no int8 kernel exists, so no int8
byte can be measured. The fused-copy and arena per-token DRAM figures are **MODEL** too -- they
come from `kv_traffic.json`, which is an analytic per-token account, not a counter. The
device-residency slope, the arena footprint, and the group-leader write reduction are
**MEASUREMENT**: each is read off a run or off an execution of the compiled SPIR-V.

Run:  python bench/results/probe_kv_lever_ledger.py
"""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
RES = ROOT / "bench" / "results"


class LedgerError(RuntimeError):
    """This instrument failed. Never a finding about a lever."""


def load(name: str) -> dict:
    p = RES / name
    if not p.is_file():
        raise LedgerError(
            f"artifact absent: {p}. This ledger reads its numbers out of artifacts every run; "
            "it will not fall back to a remembered value."
        )
    return json.loads(p.read_text(encoding="utf-8"))


def dig(d: dict, path: str, src: str):
    cur = d
    for part in path.split("."):
        if isinstance(cur, list):
            cur = cur[int(part)]
        elif isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            raise LedgerError(f"`{src}` has no field `{path}` (stopped at `{part}`)")
    return cur


# -- the levers ---------------------------------------------------------------------------------


def lever_device_residency() -> dict:
    """L1 -- the host<->device round trip for the KV cache. Axis: LINK. Status: LANDED."""
    src = "phi35_kv_chain-default-dev0.json"
    d = load(src)
    host = float(dig(d, "host_lane_slope_bytes_per_past_token", src))
    res = float(dig(d, "resident_lane_slope_bytes_per_past_token", src))
    declared = int(dig(d, "bytes_per_past_token_declared", src))
    verdict = dig(d, "verdict", src)
    bitexact = bool(dig(d, "resident_vs_host_numeric.bitwise_identical", src))
    return {
        "id": "L1",
        "name": "KV cache stays device-resident across run() calls",
        "axis": "LINK (bytes crossing host<->device per past token)",
        "baseline": "the shipping lane: every present_key/value downloaded to host after each "
                    "step and re-uploaded as the next step's past",
        "before": host,
        "after": res,
        "unit": "bytes per past token",
        "ratio": None,
        "ratio_is_undefined_because": (
            "the after-term is 0. This is an ELIMINATION, not a multiplier. Any finite 'Nx' "
            "written here is an invented denominator -- which is the most likely shape of the "
            "old ledger's unreproducible figures."
        ),
        "class": "MEASUREMENT",
        "source": f"{src}: host_lane_slope_bytes_per_past_token / "
                  f"resident_lane_slope_bytes_per_past_token",
        "correctness_control": f"logits bitwise identical across the two lanes on all "
                               f"{dig(d, 'steps', src)} steps: {bitexact}",
        "status": "LANDED (default since 2026-08-03; ONNXRUNTIME_EP_VULKAN_BIND_OUTPUTS=0 off)",
        "applies_when": "always; independent of grouping and of cache convention",
        "notes": [
            f"declared bytes per past token from the model's own input shapes: {declared}",
            f"verdict recorded by the probe: {verdict}",
            "does NOT reduce DRAM traffic: the kernel still reads and writes the same cache.",
        ],
    }


def lever_fused_copy() -> dict:
    """L2 -- the past-cache re-read fused into the attention loop. Axis: DRAM. LANDED."""
    src = "kv_traffic.json"
    d = load(src)
    fused = bool(dig(d, "shader_facts.copy_fused_into_attention_loop", src))
    amp_now = float(dig(d, "our_amplification_over_cache", src))
    nq = int(dig(d, "geometry.num_heads", src))
    nkv = int(dig(d, "geometry.kv_num_heads", src))
    g = nq // nkv
    # The shader comment states the rule: per-token KV traffic went from (G + 2) x cache to
    # (G + 1) x cache. At G = 1 that is 3 -> 2. The 2 is the artifact's own `amplification`.
    before = g + 2
    after = g + 1
    return {
        "id": "L2",
        "name": "past-cache copy fused into the attention loop (one load, used twice)",
        "axis": "DRAM (multiples of the resident cache, per token, growing convention)",
        "baseline": "a standalone relocation loop re-reading the whole past cache purely to "
                    "materialise it in `present`",
        "before": before,
        "after": after,
        "unit": "x resident cache per token",
        "ratio": before / after,
        "class": "MODEL",
        "source": f"{src}: shader_facts.copy_fused_into_attention_loop={fused}, "
                  f"our_amplification_over_cache={amp_now}, geometry.num_heads/kv_num_heads",
        "correctness_control": "the fused path is the shipping path; tests/ops/test_gqa*.py "
                               "gate it against the CPU EP",
        "status": "LANDED",
        "applies_when": "the GROWING cache convention only. Under the arena the copy does not "
                        "exist at all and this lever is already spent (see L3).",
        "notes": [
            f"the model in the artifact has Nq/Nkv = {g}; at G = {g} the rule gives "
            f"{before} -> {after}",
            "MODEL because `kv_traffic.json` is an analytic per-token account. Nothing here "
            "observes DRAM. It is the size of a prize, not a reading.",
        ],
    }


def lever_arena() -> dict:
    """L3 -- the shared-buffer (arena) convention. Axis: DRAM and FOOTPRINT. AVAILABLE."""
    src = "kv_traffic.json"
    d = load(src)
    per_token = dig(d, "per_token", src)
    at8192 = [r for r in per_token if r["past_len"] == 8192]
    if not at8192:
        raise LedgerError(f"`{src}` per_token has no past_len 8192 row")
    r = at8192[0]
    read = int(r["attention_read_bytes"])
    write = int(r["copy_write_bytes"])

    fsrc = "kv_arena_chain-A8192.json"
    f = load(fsrc)
    hw = dig(f, "bytes.arena.alloc_high_water_bytes", fsrc)
    conv = dig(f, "bytes.arena.kv_cache_convention", fsrc)
    grow_rc = dig(f, "bytes.grow.worker_rc", fsrc)
    return {
        "id": "L3",
        "name": "KV arena — `present` and `past` are one allocation (SHARED convention)",
        "axis": "DRAM (per-token) and FOOTPRINT (allocation at a context length)",
        "baseline": "the GROWING convention, where `present` is a separate, strictly larger "
                    "allocation whose past region must be materialised every step",
        "before": (read + write) / max(read, 1),
        "after": 1.0,
        "unit": "x resident cache per token (write side removed entirely)",
        "ratio": (read + write) / max(read, 1),
        "class": "MODEL",
        "source": f"{src}: per_token[past_len=8192] attention_read_bytes={read}, "
                  f"copy_write_bytes={write}",
        "measured_companion": {
            "quantity": "device allocation high-water at ctx 8192, arena lane",
            "value": hw,
            "unit": "bytes",
            "class": "MEASUREMENT",
            "source": f"{fsrc}: bytes.arena.alloc_high_water_bytes "
                      f"(kv_cache_convention={conv!r})",
            "note": f"the growing lane did not complete this size (worker_rc={grow_rc}); the "
                    f"arena did. A lever that runs where the baseline OOMs has no ratio against "
                    f"it at that extent, and one must not be manufactured.",
        },
        "correctness_control": "tests/ops/test_gqa_grouping.py arena lane, bit-exact against "
                               "the CPU EP at G=1 and G=4, decode and prefill",
        "status": "AVAILABLE (ONNXRUNTIME_EP_VULKAN_KV_ARENA=1); caller-side declaration, "
                  "not a shader mode",
        "applies_when": "the caller can declare a fixed KV capacity up front",
        "notes": [
            "the arena is what makes L4 worth the whole of its value: once the past copy is "
            "gone, the new-token write is 100% of `present` traffic, so the group redundancy "
            "is 100% of what is left. The two levers are not independent — see L4's arms.",
        ],
    }


def lever_group_write_leader() -> dict:
    """L4 -- one writer per KV group instead of G. Axis: DRAM (write side). THIS ROUND."""
    src = "kv_write_redundancy.json"
    d = load(src)
    controls = dig(d, "controls", src)
    for k in ("positive_G_gt_1_baseline_writers_equals_G",
              "repaired_G_gt_1_fixed_writers_equals_1",
              "negative_G_eq_1_unchanged", "all_arms_bit_exact"):
        if not controls.get(k):
            raise LedgerError(
                f"`{src}` control `{k}` did not pass; no figure from this probe is publishable")
    arms = []
    for a in dig(d, "arms", src):
        arms.append({
            "case": a["case"], "G": a["G"], "convention": a["convention"],
            "present_key_write_bytes_before": a["present_key_write_bytes_baseline"],
            "present_key_write_bytes_after": a["present_key_write_bytes_fixed"],
            "ratio": a["reduction"],
            "writers_per_new_token_word_before":
                a["baseline"]["present_key"]["writers_per_new_token_word_max"],
            "writers_per_new_token_word_after":
                a["fixed"]["present_key"]["writers_per_new_token_word_max"],
        })
    g4_arena = [a for a in arms if a["G"] == 4 and a["convention"] == "arena"]
    g1 = [a for a in arms if a["G"] == 1]
    return {
        "id": "L4",
        "name": "one `present` writer per KV group instead of G (group write leader)",
        "axis": "DRAM (write side: bytes named by stores into present_key/present_value)",
        "baseline": "the same kernel at the same shape with all G query heads of a group "
                    "writing the same `present` words",
        "before": max((a["writers_per_new_token_word_before"] for a in arms), default=0),
        "after": 1,
        "unit": "invocations writing each new-token word",
        "ratio": min((a["ratio"] for a in g4_arena), default=None),
        "class": "MEASUREMENT",
        "source": f"{src}: arms[].present_key_write_bytes_{{baseline,fixed}}, recorded by "
                  "executing the compiled SPIR-V over the whole dispatch grid "
                  "(bench/spirv_simt.py store trace)",
        "correctness_control": (
            "all arms bit-exact vs the baseline kernel on attn_output, present_key and "
            "present_value, read before any byte count; plus tests/ops/test_gqa_grouping.py "
            "against the CPU EP on both devices, arena and growing, decode and prefill"),
        "status": "THIS ROUND",
        "applies_when": "Nq/Nkv > 1 ONLY. Phi-3.5-mini has Nq/Nkv = 1 and cannot exhibit it at "
                        "all — the first performance defect in this project that the reference "
                        "model is structurally incapable of showing.",
        "arms": arms,
        "notes": [
            f"G=1 arms move by exactly 1.00x ({len(g1)} of them): the negative control.",
            "the ratio quoted is the arena/decode arm, where the new-token write is the whole "
            "of `present` traffic. Under the growing convention the mandatory past relocation "
            "dilutes it (1.33x at past_len 8, falling toward 1.0 as past_len grows), because "
            "that relocation already had exactly one writer.",
            "the write bytes are what the kernel's stores NAME. Calling them DRAM transactions "
            "is `probe_roofline.py`'s cache argument, which is an argument, not this reading.",
        ],
    }


def lever_quantised_cache() -> dict:
    """L5 -- int8/int4 KV storage. Axis: FOOTPRINT. NOT BUILT; every byte here is MODEL.

    Every figure is DERIVED here from the graph's own measured geometry and the arena's measured
    high-water. Nothing is transcribed from `kv-int8-budget-prediction.md`: that file's Section 3
    table is a *prediction record*, and quoting a prediction back as a ledger figure is how a
    number stops having a derivation -- which is the failure this whole file exists to repair.
    """
    isrc = "island_bytes_phi35.json"
    isl = load(isrc)
    layers = int(dig(isl, "graph_measurements.LAYERS", isrc))
    kvh = int(dig(isl, "graph_measurements.KV_HEADS", isrc))
    hd = int(dig(isl, "graph_measurements.HEAD_DIM", isrc))
    fp16 = int(dig(isl, "graph_measurements.FP16", isrc))

    asrc = "kv_arena_chain-A8192.json"
    arena = load(asrc)
    measured_footprint = int(dig(arena, "bytes.arena.alloc_high_water_bytes", asrc))

    ctx = 8192
    elems_per_token = layers * 2 * kvh * hd          # K and V, every layer
    fp16_bytes_per_token = elems_per_token * fp16
    # The non-KV term is the measured footprint with the measured KV term subtracted. It is the
    # only part of this that is anchored to a run.
    non_kv = measured_footprint - ctx * fp16_bytes_per_token
    if non_kv <= 0:
        raise LedgerError(
            f"non-KV footprint came out {non_kv} <= 0 from measured {measured_footprint} minus "
            f"{ctx} x {fp16_bytes_per_token}; the two artifacts disagree about the model")

    def scales_per_token(granularity: str) -> int:
        """Scale ELEMENTS per token, stored fp16. Granularity is a per-(layer, K/V) fact."""
        if granularity == "per_block32":
            if hd % 32:
                raise LedgerError(f"head_dim {hd} is not a multiple of the 32-element block")
            return layers * 2 * kvh * (hd // 32)
        if granularity == "per_head":
            return layers * 2 * kvh
        if granularity == "per_tensor":
            return layers * 2
        raise LedgerError(f"unknown granularity {granularity!r}")

    lanes = []
    for bits, gran in ((8, "per_tensor"), (8, "per_head"), (8, "per_block32"),
                       (4, "per_head"), (4, "per_block32")):
        bpt = elems_per_token * bits // 8 + scales_per_token(gran) * fp16
        footprint = non_kv + ctx * bpt
        lanes.append({
            "lane": f"int{bits}/{gran}", "bits": bits, "granularity": gran,
            "kv_bytes_per_token": bpt,
            "footprint_at_ctx_8192": footprint,
            "ratio_vs_fp16_arena": measured_footprint / footprint,
        })
    best8 = max((l for l in lanes if l["bits"] == 8), key=lambda l: l["ratio_vs_fp16_arena"])
    worst8 = min((l for l in lanes if l["bits"] == 8), key=lambda l: l["ratio_vs_fp16_arena"])
    best4 = max((l for l in lanes if l["bits"] == 4), key=lambda l: l["ratio_vs_fp16_arena"])

    rsrc = "kv_int8_budget-dev0.json"
    load(rsrc)  # presence is the claim that the residual measurement exists; fields are its own
    return {
        "id": "L5",
        "name": "quantised KV cache (int8 / int4 storage)",
        "axis": "FOOTPRINT (device allocation at a context length)",
        "baseline": f"the fp16 arena's MEASURED {measured_footprint} B high-water at ctx {ctx} "
                    f"({asrc}: bytes.arena.alloc_high_water_bytes)",
        "before": measured_footprint,
        "after": best8["footprint_at_ctx_8192"],
        "unit": "bytes of device allocation at ctx 8192",
        "ratio": best8["ratio_vs_fp16_arena"],
        "ratio_int8_best": best8["ratio_vs_fp16_arena"],
        "ratio_int8_worst": worst8["ratio_vs_fp16_arena"],
        "ratio_int4_best": best4["ratio_vs_fp16_arena"],
        "class": "MODEL",
        "derivation": {
            "elements_per_token": elems_per_token,
            "elements_per_token_from": f"{isrc}: LAYERS x 2 x KV_HEADS x HEAD_DIM "
                                       f"= {layers} x 2 x {kvh} x {hd}",
            "fp16_kv_bytes_per_token": fp16_bytes_per_token,
            "non_kv_footprint_bytes": non_kv,
            "non_kv_from": f"{measured_footprint} (MEASURED) - {ctx} x {fp16_bytes_per_token}",
            "scales_stored": "fp16, counted per (layer, K/V) at the lane's granularity",
        },
        "source": f"derived at run time from {isrc} (graph geometry, MEASUREMENT) and {asrc} "
                  f"(arena high-water, MEASUREMENT). The COMPOSITION is MODEL: no int8 kernel "
                  f"exists, so no int8 byte has ever been observed. Residuals in {rsrc} are "
                  f"MEASUREMENT; the bytes are not.",
        "correctness_control": (
            "host-boundary quantisation only: storage error modelled exactly, kernel write "
            "rounding and accumulation order not modelled at all, so every residual is a LOWER "
            "BOUND on a real int8 kernel's."),
        "status": "NOT BUILT — deliberately. The error budget gates the kernel and the budget "
                  "was measured first.",
        "applies_when": "if and when a tolerance ruling admits it; open, and Morpheus's call",
        "lanes": lanes,
        "notes": [
            "verdict NO_ULP_BAND_ADMITS_INT8_AND_STILL_CATCHES_FP16: the best granularity sits "
            "at 6-7x the fp16 path's own residual, so any band admitting int8 stops policing "
            "fp16. That survived the §8.9.22 repair of the observable.",
            "the residual SATURATES: 29 ULP by past_len ~28 for int8/per_head, measured to 259. "
            "An 8-step slope extrapolated to ctx 8192 predicted ~13,000 — wrong by ~450x.",
            "the old ledger's 3.17x (int8) and 4.06x (int4) do not reproduce; the derivation "
            "above is what the artifacts support. The disagreement was written down in "
            "kv-int8-budget-prediction.md Section 3 before the first int8 run.",
            "the FOOTPRINT ratio is the small one because the KV cache is only 60.5% of the "
            "stream at ctx 8192 and 0% of the weights; a 2x or 4x saving on 60% of a total is "
            "1.4x or 1.8x of the total, and quoting the 2x/4x as the lever is the single most "
            "likely way the old numbers got their size.",
        ],
    }


# -- the unreproducible three -------------------------------------------------------------------


def reconstruction_attempt(levers: list[dict]) -> dict:
    """What the old 2.21x / 3.17x / 4.06x could have been. Answer: nothing checkable."""
    isl = load("island_bytes_phi35.json")
    rows = {r["past_sequence_length"]: r for r in isl["by_context_length"]}
    at0, at8192 = rows[0], rows[8192]
    kv8192 = at8192["kv_cache_MiB"]
    L5 = [L for L in levers if L["id"] == "L5"][0]
    lanes = {l["lane"]: l for l in L5["lanes"]}
    fp16_bpt = L5["derivation"]["fp16_kv_bytes_per_token"]

    def stream_ratio(lane: str) -> float:
        """KV term scaled by the lane's bytes/token, at the growing convention's 1x cache."""
        kv_new = kv8192 * lanes[lane]["kv_bytes_per_token"] / fp16_bpt
        return at8192["total_MiB"] / (at0["total_MiB"] + kv_new)

    tried = {
        "stream total at 8192 / stream total at 0 (i.e. all KV removed)":
            at8192["total_MiB"] / at0["total_MiB"],
        "(non-KV + 2x KV) / (non-KV + KV)  [removing the present copy]":
            (at0["total_MiB"] + 2 * kv8192) / (at0["total_MiB"] + kv8192),
        "(non-KV + 3x KV) / (non-KV + 2x KV)  [the fused-copy lever, L2, on the stream]":
            (at0["total_MiB"] + 3 * kv8192) / (at0["total_MiB"] + 2 * kv8192),
        "KV term only, fp16 / int8 (naive, no scales)": 2.0,
        "KV term only, fp16 / int4 (naive, no scales)": 4.0,
        "footprint ratio, int8 per_block32 (derived above)":
            lanes["int8/per_block32"]["ratio_vs_fp16_arena"],
        "footprint ratio, int4 per_head (derived above)":
            lanes["int4/per_head"]["ratio_vs_fp16_arena"],
        "modelled stream ratio, int8 per_head": stream_ratio("int8/per_head"),
        "modelled stream ratio, int4 per_head": stream_ratio("int4/per_head"),
    }
    targets = {"present_copy": 2.21, "int8": 3.17, "int4": 4.06}
    nearest = {}
    for tname, tval in targets.items():
        k, v = min(tried.items(), key=lambda kv: abs(kv[1] - tval))
        nearest[tname] = {"nearest_baseline": k, "value": round(v, 4),
                          "gap": round(abs(v - tval), 4)}
    return {
        "targets": targets,
        "baselines_tried": {k: round(v, 4) for k, v in tried.items()},
        "nearest": nearest,
        "conclusion": (
            "none of the three reproduces on any baseline constructible from the artifacts in "
            "this tree. All three are RETRACTED rather than corrected: a number whose "
            "derivation cannot be found is not repaired by finding a derivation that lands "
            "nearby."
        ),
        "hypothesis_named_as_a_hypothesis": (
            "2.21 and 4.06 sit 0.21 and 0.06 from the naive KV-TERM-ONLY ratios 2.0 (fp16 -> "
            "int8, scales ignored) and 4.0 (fp16 -> int4, scales ignored). That is consistent "
            "with the old ledger quoting savings on the KV term as if they were savings on "
            "something a user waits for -- an axis error, not an arithmetic one, which is why "
            "no amount of re-deriving on the stream or the footprint could ever land on them. "
            "3.17 fits nothing, so even this does not explain all three. It is offered as a "
            "hypothesis and is not the derivation; what would settle it is the artifact the "
            "figures were read off, and no such artifact exists in this tree. Nothing in this "
            "ledger depends on it."
        ),
    }


def main() -> int:
    try:
        levers = [lever_device_residency(), lever_fused_copy(), lever_arena(),
                  lever_group_write_leader(), lever_quantised_cache()]
        recon = reconstruction_attempt(levers)
    except LedgerError as e:
        print(f"ERROR(instrument): {e}")
        return 3

    print("KV LEVER LEDGER — re-derived from artifacts, every figure classed. No clock.")
    print()
    print(f"{'id':<3} {'lever':<52} {'axis':<10} {'class':<12} {'ratio':>8}  status")
    print("-" * 118)
    for L in levers:
        axis = L["axis"].split(" ")[0]
        r = L.get("ratio")
        if r is None:
            r = L.get("ratio_int8_best")
        rs = f"{r:.2f}x" if isinstance(r, (int, float)) else "n/a"
        print(f"{L['id']:<3} {L['name'][:52]:<52} {axis:<10} {L['class']:<12} {rs:>8}  "
              f"{L['status'].split(' ')[0]}")
    print()
    print("AXES ARE NOT INTERCHANGEABLE. LINK bytes, DRAM bytes and FOOTPRINT bytes are three")
    print("different quantities; a ratio on one does not compose with a ratio on another, and")
    print("L1 has no ratio at all because its after-term is zero.")
    print()
    print("L4 arms (the lever measured this round):")
    for a in [L for L in levers if L["id"] == "L4"][0]["arms"]:
        print(f"   G={a['G']} {a['convention']:<8} writers/word {a['writers_per_new_token_word_before']}"
              f" -> {a['writers_per_new_token_word_after']}   "
              f"{a['present_key_write_bytes_before']:>8} -> "
              f"{a['present_key_write_bytes_after']:>8} B   {a['ratio']:.2f}x   {a['case']}")
    print()
    print("THE OLD LEDGER'S THREE NUMBERS — reconstruction attempt")
    for k, v in recon["baselines_tried"].items():
        print(f"   {v:>8.4f}   {k}")
    for t, n in recon["nearest"].items():
        print(f"   target {t} = {recon['targets'][t]}: nearest is {n['value']} "
              f"(gap {n['gap']}) — {n['nearest_baseline']}")
    print(f"   {recon['conclusion']}")
    print(f"   HYPOTHESIS (not a derivation): {recon['hypothesis_named_as_a_hypothesis']}")

    print()
    print("L5 lanes (derived here from measured geometry + measured arena high-water; MODEL):")
    for l in [L for L in levers if L["id"] == "L5"][0]["lanes"]:
        print(f"   {l['lane']:<20} {l['kv_bytes_per_token']:>8} B/token   "
              f"{l['footprint_at_ctx_8192']:>13} B at ctx 8192   "
              f"{l['ratio_vs_fp16_arena']:.3f}x")

    record = {
        "probe": "kv_lever_ledger",
        "PROVENANCE": {
            "SPECIFICATION": "a fact about a named part, published by its maker",
            "MEASUREMENT": "a fact about this graph/module/run, re-derived here every run",
            "MODEL": "an analytic construction; never quotable as a measurement",
        },
        "axes": {
            "LINK": "bytes crossing host<->device per past token (staging traffic)",
            "DRAM": "bytes the kernel's own loads and stores name, per token",
            "FOOTPRINT": "bytes of device allocation held at a context length",
        },
        "composition_rule": (
            "levers on different axes are never multiplied. L2 and L3 are both DRAM and are "
            "MUTUALLY EXCLUSIVE (L2 exists only under the growing convention, L3 abolishes it). "
            "L3 and L4 are both DRAM and DO compose: L3 removes the past-region write, after "
            "which L4's new-token dedup is the whole of what remains — which is why L4 measures "
            "its full G on the arena arm and only 1.33x on the growing arm."
        ),
        "levers": levers,
        "retracted": recon,
    }
    out = RES / "kv_lever_ledger.json"
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\nrecord: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
