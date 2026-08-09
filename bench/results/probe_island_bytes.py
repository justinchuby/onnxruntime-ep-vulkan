#!/usr/bin/env python3
"""The whole decode island in bytes: weights, KV cache, intermediates. No clock.

WHAT WENT WRONG HERE, AND THE RULE THAT COMES OUT OF IT
=======================================================
Until this revision the `weight_reread_amplification` block below was **five literals**::

    "inb_load_instructions_per_inference": 116_324_352,
    "load_width_bytes": 16,
    "product_bytes": 1_861_189_632,
    "int4_weight_bytes_from_graph": 1_861_189_632,
    "amplification": 1.0,

The docstring argued -- correctly, as it turns out -- that this is not an identity, because
two factors in it are contingent: **loads per blob** and **blobs per workgroup**. It said the
argument rested on a SPIR-V def-use walk. **No walk was in the tree.** The reasoning happened
once, in a head, and its conclusion was transcribed as a constant. A kernel change that made
the packed path re-read every blob eight times would not have moved the printed
`1.000000` by a digit.

The generalisation is not "do not hardcode". It is that **this file mixed two kinds of number
without saying which was which**, and once mixed, a transcribed conclusion is indistinguishable
from an observation. So every quantity here now carries a class, in `PROVENANCE`:

**SPECIFICATION** -- a fact about a part, published by whoever made it. Legitimately a literal;
    deriving it would be pretending to measure a datasheet. But a spec is a fact *about a
    named part*, so the name of the part is a separate claim and is **not** a specification --
    see `PEAK_BYTES_PER_S`, whose device name is read off the run's timestamp fingerprint.

**MEASUREMENT** -- a fact about *this* artifact: this graph, this compiled module, this run.
    Must be derived here, every time, from the artifact. A literal is the defect. The test is
    R9's: does the number change when the claim is false?

**MODEL** -- an analytic construction that is neither published nor observed: an assumption
    about what must happen, arithmetic over measurements. Legitimately code, never quotable as
    a measurement. Its *inputs* must all be measurements or specifications, and it must be
    labelled, because an unlabelled model reads exactly like a measurement -- which is how the
    amplification survived: it was a model's conclusion wearing a measurement's clothes.

THE WEIGHT RE-READ CHECK
========================
The roofline in `probe_roofline.py` assumes **each weight byte is read exactly once per
token**. That assumption is now measured, by `probe_weight_reread.py`, which executes the
compiled SPIR-V of the pipeline the run dispatched -- located by content digest against
`evidence/proof_ledger.jsonl` -- over the whole dispatch grid, and records every address the
`InB` binding is loaded from. This file only *reads* that record; it does not restate it, and
if the record is missing, stale against the ledger, or was produced without a positive control
firing, this file prints `UNWITNESSED` and no number.

WHAT THE ISLAND IS MADE OF
==========================
The remaining budget is everything that is *not* weights. That budget is dominated by a term
nobody has costed, and it is not intermediates.

Run:  python bench/results/probe_weight_reread.py --out bench/results/weight_reread_phi35.json
      python bench/results/probe_island_bytes.py
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]

MIB = 1024 * 1024

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

REREAD_RECORD = ROOT / "bench" / "results" / "weight_reread_phi35.json"

# Result-identity contract (issue #19 follow-up, Morpheus review on PR #31): the resolved model
# path and its exact content hash are stamped into the output record below, computed lazily
# (only once the model has already been opened successfully) so a PHI35_MODEL override or a
# stale/wrong cached file can never be silently absorbed into the evidence. Reuses the streaming
# SHA-256 helper `model_provenance.sha256_of` rather than a 23rd divergent hasher.
sys.path.insert(0, str(ROOT / "rust" / "tools"))
import model_provenance as _model_provenance  # noqa: E402


def _result_identity() -> dict:
    return {
        "onnx_file": str(MODEL),
        "onnx_sha256": _model_provenance.sha256_of(MODEL),
    }

#: The run whose device this file's peak bandwidth is a specification *of*. The device name is
#: read out of this record's `device_identity.observed_from_trace` -- the device whose
#: timestamp fingerprint appears in the trace -- and never off `device_index`, which is an EP
#: enumeration order and in this very record has vulkaninfo index 0 naming a different part.
RUN_RECORD = ROOT / "bench" / "results" / "phi35-certified-dev0.json"

# -- SPECIFICATIONS ----------------------------------------------------------------------------
#: NVIDIA's published memory configuration for the part named below. 128-bit GDDR6 at 16 Gbps.
#: A datasheet fact: it does not change when our code changes, so it is not derivable from
#: anything we could run, and a literal is the correct form. What is *not* a specification is
#: which part the run used; that is `device_from_run()`.
SPEC_PART = "NVIDIA GeForce RTX 4060 Laptop GPU"
BUS_BITS = 128
MEM_GBPS = 16.0
PEAK_BYTES_PER_S = BUS_BITS / 8 * MEM_GBPS * 1e9


class InstrumentError(RuntimeError):
    """The instrument cannot answer. Never degrade to a plausible number."""


PROVENANCE: dict[str, dict[str, str]] = {
    "PEAK_BYTES_PER_S": {
        "class": "SPECIFICATION",
        "source": f"{BUS_BITS}-bit GDDR6 @ {MEM_GBPS} Gbps, published for {SPEC_PART}",
        "wrong_when": "the run used a different part -- so the part name is checked against "
                      "the run's timestamp fingerprint, not assumed",
    },
    "SPEC_PART": {
        "class": "MEASUREMENT",
        "source": f"{RUN_RECORD.name} -> device_identity.observed_from_trace",
        "wrong_when": "a run on another device is quoted against this peak; the check fires",
    },
    "LAYERS, HIDDEN, FFN, VOCAB, KV_HEADS, HEAD_DIM, FP16": {
        "class": "MEASUREMENT",
        "source": "the ONNX graph: MatMulNBits K/N attributes, past_key_values.N.key shape "
                  "and elem_type, GroupQueryAttention node count",
        "wrong_when": "the model changes -- previously literals, which is why this file would "
                      "have gone on printing Phi-3.5 numbers for any other model",
    },
    "WEIGHT_STREAM_BYTES": {
        "class": "MEASUREMENT",
        "source": "sum of B and scales initializer sizes over the graph's MatMulNBits nodes",
        "wrong_when": "the quantisation or the model changes; was a literal restating "
                      "probe_roofline.py, so the two could drift apart silently",
    },
    "weight_reread_amplification": {
        "class": "MEASUREMENT",
        "source": f"{REREAD_RECORD.name}, produced by probe_weight_reread.py: a SIMT execution "
                  "of the compiled SPIR-V located by digest against the proof ledger",
        "wrong_when": "the kernel re-reads a blob, or two workgroups share one -- both are "
                      "witnessed positive by controls in that probe",
    },
    "intermediate_breakdown_bytes": {
        "class": "MODEL",
        "source": "node counts by op_type from the graph (measured) x a per-node traffic rule "
                  "(assumed): every tensor crossing a dispatch boundary is written once and "
                  "read once",
        "wrong_when": "never, by observation -- nothing here observes activation traffic. It "
                      "is the size of a prize, not a reading, and must not be quoted as one",
    },
    "kv_bytes()": {
        "class": "MODEL",
        "source": "LAYERS x 2 x past_len x KV_HEADS x HEAD_DIM x FP16 (all inputs measured)",
        "wrong_when": "never, by observation -- GroupQueryAttention runs on CPU in this EP, so "
                      "no dispatch of ours reads this cache and no counter of ours sees it",
    },
}


# -- MEASUREMENTS: the device the peak is a specification of -----------------------------------


def device_from_run(record: pathlib.Path = RUN_RECORD) -> dict:
    """The device name **observed in the run's trace**, never the one implied by the selector.

    A results row carries `device_index`, which is an EP enumeration order. In
    `phi35-certified-dev0.json` that index is 0, and vulkaninfo index 0 on this box names an
    Intel integrated part -- so reading the name off the selector would attribute a discrete
    GPU's run to an iGPU, or the reverse. `device_identity` resolves it the only way that can
    fail: by matching the timestamp-counter fingerprint in the row's own trace against the
    enumerated devices.
    """
    if not record.exists():
        raise InstrumentError(f"no run record at {record}: the device is unknown, and a peak "
                              "bandwidth with no named part is not a specification")
    doc = json.loads(record.read_text(encoding="utf-8"))
    found = []

    def walk(o):
        if isinstance(o, dict):
            ident = o.get("device_identity")
            if isinstance(ident, dict):
                found.append(ident)
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(doc.get("results"))
    if not found:
        raise InstrumentError(
            f"{record.name} has no `device_identity`; the device it ran on is not readable "
            "from the record and this probe will not read it off `device_index`"
        )
    names = {i.get("observed_from_trace") for i in found}
    if len(names) != 1 or None in names:
        raise InstrumentError(f"{record.name} attributes its rows to {names}")
    name = names.pop()
    return {
        "observed_from_trace": name,
        "reason": found[0].get("reason", ""),
        "ok": all(bool(i.get("ok")) for i in found),
        "matches_spec_part": name == SPEC_PART,
    }


# -- MEASUREMENTS: everything that is a property of this graph ---------------------------------


def _initializer_bytes(t) -> int:
    for d in t.external_data:
        if d.key == "length":
            return int(d.value)
    if t.raw_data:
        return len(t.raw_data)
    raise InstrumentError(
        f"initializer `{t.name}` has neither raw_data nor an external `length`; this probe "
        "will not infer its size from the shape"
    )


def graph_facts(model: pathlib.Path = MODEL) -> dict:
    """Every model property this file used to state as a literal, read off the graph.

    Each of these was a constant with a comment saying where it came from. A comment is not a
    derivation: it cannot be wrong in a way the program notices.
    """
    if not model.exists():
        raise InstrumentError(f"model not found: {model}")
    try:
        import onnx
    except ImportError as e:  # pragma: no cover
        raise InstrumentError(f"onnx not importable: {e}") from e

    m = onnx.load(str(model), load_external_data=False)
    init = {i.name: i for i in m.graph.initializer}

    op_counts: dict[str, int] = {}
    matmuls = []
    for n in m.graph.node:
        op_counts[n.op_type] = op_counts.get(n.op_type, 0) + 1
        if n.op_type != "MatMulNBits":
            continue
        a = {at.name: at.i for at in n.attribute if at.type == onnx.AttributeProto.INT}
        if "K" not in a or "N" not in a:
            raise InstrumentError(f"`{n.name}` is missing K/N")
        b, s = n.input[1], n.input[2]
        for name in (b, s):
            if name not in init:
                raise InstrumentError(
                    f"`{n.name}` input `{name}` is not an initializer; its bytes are not a "
                    "property of the graph and this probe will not model them"
                )
        matmuls.append({
            "name": n.name, "K": a["K"], "N": a["N"],
            "bits": a.get("bits", 4), "block_size": a.get("block_size", 32),
            "weight_bytes": _initializer_bytes(init[b]),
            "scale_bytes": _initializer_bytes(init[s]),
        })
    if not matmuls:
        raise InstrumentError("no MatMulNBits nodes in the graph")

    # KV geometry: past_key_values.N.key is [batch, kv_heads, past_sequence_length, head_dim].
    keys = [i for i in m.graph.input if i.name.startswith("past_key_values.")
            and i.name.endswith(".key")]
    if not keys:
        raise InstrumentError("no `past_key_values.N.key` graph inputs; KV geometry unknown")
    shapes = {tuple(d.dim_value for d in k.type.tensor_type.shape.dim) for k in keys}
    elems = {k.type.tensor_type.elem_type for k in keys}
    if len(shapes) != 1 or len(elems) != 1:
        raise InstrumentError(f"KV inputs disagree: shapes={shapes} elem_types={elems}")
    shape = shapes.pop()
    if len(shape) != 4:
        raise InstrumentError(f"past_key_values key rank is {len(shape)}, expected 4")
    kv_heads, head_dim = int(shape[1]), int(shape[3])
    if kv_heads <= 0 or head_dim <= 0:
        raise InstrumentError(f"KV head geometry is symbolic: {shape}")

    import onnx.helper as helper
    dt = elems.pop()
    fp = onnx.helper.tensor_dtype_to_np_dtype(dt).itemsize if hasattr(
        helper, "tensor_dtype_to_np_dtype") else 2

    layers = op_counts.get("GroupQueryAttention", 0)
    if layers != len(keys):
        raise InstrumentError(
            f"{layers} GroupQueryAttention nodes but {len(keys)} KV inputs; the layer count "
            "is ambiguous and this probe will not pick one"
        )

    # HIDDEN is the residual width: the projection that is square. FFN is the width feeding a
    # projection back down to HIDDEN. VOCAB is the widest output in the graph.
    ns = {(mm["K"], mm["N"]) for mm in matmuls}
    square = {k for k, n in ns if k == n}
    if len(square) != 1:
        raise InstrumentError(f"cannot identify the residual width; square projections: {square}")
    hidden = square.pop()
    ffn = {k for k, n in ns if n == hidden and k != hidden}
    if len(ffn) != 1:
        raise InstrumentError(f"cannot identify the FFN width; candidates: {ffn}")
    vocab = max(n for _, n in ns)

    return {
        "model": str(model),
        "op_counts": op_counts,
        "matmulnbits": matmuls,
        "LAYERS": layers,
        "HIDDEN": hidden,
        "FFN": ffn.pop(),
        "VOCAB": vocab,
        "KV_HEADS": kv_heads,
        "HEAD_DIM": head_dim,
        "FP16": fp,
        "weight_bytes": sum(mm["weight_bytes"] for mm in matmuls),
        "scale_bytes": sum(mm["scale_bytes"] for mm in matmuls),
        "WEIGHT_STREAM_BYTES": sum(mm["weight_bytes"] + mm["scale_bytes"] for mm in matmuls),
    }


# -- MEASUREMENT: the re-read amplification, read from the probe that measures it --------------


def weight_reread(record: pathlib.Path = REREAD_RECORD) -> dict:
    """Load the measured amplification, or refuse.

    Three ways this refuses, all of which the literal block could not:
    the record is absent; its module digest no longer matches the ledger the run dispatched
    from; or no positive control fired, in which case the probe has never been seen to report
    anything but 1 and its 1 means nothing.
    """
    if not record.exists():
        return {"verdict": "UNWITNESSED",
                "why": f"no {record.name}; run bench/results/probe_weight_reread.py "
                       f"--out bench/results/{record.name} first"}
    doc = json.loads(record.read_text(encoding="utf-8"))
    controls = doc.get("positive_controls", {})
    if not controls.get("witnessed"):
        return {"verdict": "UNWITNESSED",
                "why": "the probe's positive controls did not all fire; a detector never seen "
                       "in its positive state has no demonstrated positive state",
                "controls": controls.get("controls", [])}
    module = doc.get("subject", {})
    if not module.get("digest_matches_ledger"):
        return {"verdict": "UNWITNESSED",
                "why": "the walked module's digest does not match the proof ledger; the "
                       "reading is not bound to the kernel the run dispatched",
                "module": module}
    tot = doc.get("measured", {})
    den = doc.get("denominator", {})
    for k in ("inb_load_instructions_per_inference", "named_bytes_per_inference",
              "amplification"):
        if tot.get(k) is None:
            return {"verdict": "UNWITNESSED", "why": f"the record has no `{k}`"}
    if den.get("int4_weight_bytes_from_graph") is None:
        return {"verdict": "UNWITNESSED",
                "why": "the record has no graph-derived denominator"}
    return {
        "verdict": tot.get("verdict", ""),
        "inb_load_instructions_per_inference": tot["inb_load_instructions_per_inference"],
        "load_width_bytes": tot.get("load_widths_bytes_observed"),
        "product_bytes": tot["named_bytes_per_inference"],
        "int4_weight_bytes_from_graph": den["int4_weight_bytes_from_graph"],
        "amplification": tot["amplification"],
        "max_loads_naming_one_word": tot.get("max_loads_naming_one_word"),
        "words_named_by_more_than_one_workgroup": tot.get(
            "words_named_by_more_than_one_workgroup"),
        "coverage": tot.get("min_coverage_of_the_weight_tensor"),
        "module": {
            "stem": module.get("shader_stem"),
            "digest": module.get("shader_digest"),
            "digest_matches_ledger": module.get("digest_matches_ledger"),
            "module_bytes": module.get("module_bytes"),
        },
        "positive_controls": [
            {"control": c["control"], "amplification": c.get("amplification")}
            for c in controls.get("controls", [])
        ],
    }


# -- MODELS --------------------------------------------------------------------------------


def kv_bytes(f: dict, past_len: int) -> int:
    """MODEL. Bytes of KV cache `GroupQueryAttention` must read to decode one token.

    Every layer reads its whole K and V history: linear in context and unbounded, which is what
    distinguishes it from every other term here. Nothing observes this -- GQA does not run on
    this EP -- so it is a construction, and is labelled one.
    """
    per_token_per_layer = f["KV_HEADS"] * f["HEAD_DIM"] * f["FP16"]
    return f["LAYERS"] * 2 * past_len * per_token_per_layer


def intermediate_bytes(f: dict) -> tuple[int, list[tuple[str, int]]]:
    """MODEL. The minimum activation traffic between dispatches, at batch 1, one token.

    The *counts* are measured -- every multiplier below is a node count read off the graph. The
    *rule* is assumed: every tensor crossing a dispatch boundary is written once and read once.
    That rule is not observed by anything, here or elsewhere, so this total is the size of the
    fusion prize and not a reading of it.
    """
    fp, hid, ffn = f["FP16"], f["HIDDEN"], f["FFN"]
    n = f["op_counts"]
    items: list[tuple[str, int]] = []

    # Each MatMulNBits reads its input row and writes its output row: K + N elements. Summed
    # over the actual nodes, so a model with a different projection mix costs differently.
    mm = sum(x["K"] + x["N"] for x in f["matmulnbits"]) * fp
    items.append((f"MatMulNBits activations ({len(f['matmulnbits'])} nodes)", mm))

    # Reads input and skip, writes normalised output and the skip sum the next block consumes.
    items.append((f"SkipSimplifiedLayerNormalization ({n.get('SkipSimplifiedLayerNormalization', 0)})",
                  n.get("SkipSimplifiedLayerNormalization", 0) * 4 * hid * fp))
    items.append((f"Sigmoid ({n.get('Sigmoid', 0)})", n.get("Sigmoid", 0) * 2 * ffn * fp))
    items.append((f"Mul ({n.get('Mul', 0)})", n.get("Mul", 0) * 3 * ffn * fp))
    items.append((f"GroupQueryAttention q/k/v/out ({n.get('GroupQueryAttention', 0)})",
                  n.get("GroupQueryAttention", 0) * 4 * hid * fp))
    items.append((f"SimplifiedLayerNormalization ({n.get('SimplifiedLayerNormalization', 0)})",
                  n.get("SimplifiedLayerNormalization", 0) * 2 * hid * fp))
    items.append((f"Gather ({n.get('Gather', 0)}, embedding row)",
                  n.get("Gather", 0) * hid * fp))
    return sum(v for _, v in items), items


def main() -> int:
    try:
        f = graph_facts()
        dev = device_from_run()
    except InstrumentError as e:
        print(f"ERROR(instrument): {e}", file=sys.stderr)
        return 2

    reread = weight_reread()
    weight_stream = f["WEIGHT_STREAM_BYTES"]
    inter, breakdown = intermediate_bytes(f)

    rows = []
    for past in (0, 128, 512, 2048, 4096, 8192):
        kv = kv_bytes(f, past)
        total = weight_stream + kv + inter
        rows.append({
            "past_sequence_length": past,
            "weight_MiB": weight_stream / MIB,
            "kv_cache_MiB": kv / MIB,
            "intermediates_MiB": inter / MIB,
            "total_MiB": total / MIB,
            "kv_share": kv / total,
            "intermediate_share": inter / total,
            "floor_ms_at_spec_peak": total / PEAK_BYTES_PER_S * 1e3,
        })

    report = {
        **_result_identity(),
        "probe": "whole_island_bytes_phi35_decode",
        "provenance": PROVENANCE,
        "specification": {
            "part": SPEC_PART,
            "bus_bits": BUS_BITS,
            "mem_gbps": MEM_GBPS,
            "peak_bytes_per_s": PEAK_BYTES_PER_S,
            "device_observed_in_run": dev,
        },
        "graph_measurements": {
            k: f[k] for k in ("LAYERS", "HIDDEN", "FFN", "VOCAB", "KV_HEADS", "HEAD_DIM",
                              "FP16", "weight_bytes", "scale_bytes", "WEIGHT_STREAM_BYTES")
        },
        "op_counts": f["op_counts"],
        "weight_reread_amplification": reread,
        "intermediate_breakdown_bytes": dict(breakdown),
        "by_context_length": rows,
    }
    out = ROOT / "bench" / "results" / "island_bytes_phi35.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("PROVENANCE — which numbers in this file are which kind")
    print("=" * 78)
    for name, p in PROVENANCE.items():
        print(f"  [{p['class']:>13}] {name}")
        print(f"                  from: {p['source']}")
    print()

    print("WEIGHT RE-READ CHECK — the assumption the whole roofline rests on")
    print("=" * 78)
    if reread["verdict"] == "UNWITNESSED":
        print(f"  UNWITNESSED: {reread['why']}")
        print("  No amplification is quoted. The roofline's per-token weight bytes are")
        print("  unconfirmed until probe_weight_reread.py runs with its controls firing.")
    else:
        m = reread["module"]
        print(f"  module {m.get('stem')} digest {m.get('digest')} "
              f"(ledger: {m.get('digest_matches_ledger')})")
        print(f"  {reread['inb_load_instructions_per_inference']:,} InB loads at "
              f"{reread['load_width_bytes']} B  =  {reread['product_bytes']:,} B")
        print(f"  int4 weight bytes from graph   =  "
              f"{reread['int4_weight_bytes_from_graph']:,} B")
        print(f"  amplification                  =  {reread['amplification']:.6f}")
        print(f"  max loads naming one 4-B word  =  {reread['max_loads_naming_one_word']}")
        print(f"  words named by >1 workgroup    =  "
              f"{reread['words_named_by_more_than_one_workgroup']}")
        print("  positive controls: " + ", ".join(
            f"{c['control']}={c['amplification']}" if c["amplification"] is not None
            else c["control"] for c in reread["positive_controls"]))
    print()

    print("DEVICE — the part the peak bandwidth is a specification of")
    print("=" * 78)
    print(f"  observed in run trace: {dev['observed_from_trace']}")
    print(f"  spec sheet describes:  {SPEC_PART}")
    print(f"  match: {dev['matches_spec_part']}   ({dev['reason']})")
    if not dev["matches_spec_part"]:
        print("  ERROR(instrument): the peak below is a specification of a part this run did")
        print("  not use. Every floor_ms figure in this table is meaningless.")
    print()

    print("WHOLE ISLAND — bytes that must move to decode one token")
    print("=" * 78)
    print(f"  weights + scales from the graph: {weight_stream:,} B "
          f"({weight_stream / MIB:,.1f} MiB) over {len(f['matmulnbits'])} MatMulNBits nodes")
    print(f"  {f['LAYERS']} layers, hidden {f['HIDDEN']}, ffn {f['FFN']}, "
          f"vocab {f['VOCAB']}, {f['KV_HEADS']} kv heads x {f['HEAD_DIM']} "
          f"(all read off the graph)")
    print()
    print("  MODEL — intermediates (the fusion prize), by producer:")
    for name, v in sorted(breakdown, key=lambda kv: -kv[1]):
        print(f"    {name:52s} {v / MIB:8.2f} MiB")
    print(f"    {'TOTAL intermediates':52s} {inter / MIB:8.2f} MiB")
    print()
    print(f"  {'past_len':>9} {'weights':>10} {'KV cache':>10} {'inter':>8} "
          f"{'total':>10} {'KV%':>7} {'inter%':>8} {'floor ms':>9}")
    for r in rows:
        print(f"  {r['past_sequence_length']:>9} {r['weight_MiB']:>10.1f} "
              f"{r['kv_cache_MiB']:>10.1f} {r['intermediates_MiB']:>8.2f} "
              f"{r['total_MiB']:>10.1f} {r['kv_share']:>7.1%} "
              f"{r['intermediate_share']:>8.3%} {r['floor_ms_at_spec_peak']:>9.2f}")
    print()
    print("  The `inter%` and `KV%` columns are MODEL, not measurement: no dispatch of ours")
    print("  reads the KV cache, and nothing here observes activation traffic.")
    print(f"\n  record: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
