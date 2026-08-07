#!/usr/bin/env python3
"""Does the packed GEMV read each weight byte once? Executed, not asserted. No clock.

WHAT THIS REPLACES
==================
`probe_island_bytes.py` published

    inb_load_instructions_per_inference   116,324,352
    load_width_bytes                               16
    product_bytes                       1,861,189,632
    int4_weight_bytes_from_graph        1,861,189,632
    amplification                             1.000000

and **all five were literals**. `116,324,352` was `N x blocks_per_col` summed over the graph --
the *blob count* -- and `16` was `blob_bytes()`, so `product_bytes` was `blobs x blob_bytes`,
which is the definition of the weight tensor. Two constants typed to be equal, and their ratio
typed as 1.0. A kernel change that re-read every blob eight times would not have moved it.

The docstring's *argument* was not circular: the load count is only the blob count if (a) the
shader issues one load per blob and (b) no two workgroups load the same blob. Those are the
whole content of the claim -- and the script established neither. It transcribed a conclusion.

WHAT THIS DOES INSTEAD
======================
1. Finds the compiled SPIR-V module **by the digest the proof ledger already records** for
   `q_gemv_matmul_nbits_f16`, so the reading is bound to a specific compiled kernel and fails
   loudly rather than silently reading some other build's module.
2. **Runs it** (`bench/spirv_simt.py`) over the entire dispatch grid of every `MatMulNBits`
   node in the Phi-3.5 graph, at the specialization constants and push constants
   `ops::quant::matmul_nbits_gemv` resolves for that node's shape, and records the byte range
   named by every load of binding 1.
3. Takes the denominator from the **ONNX initializers** -- `len(raw_data)`, or the external-data
   `length` field -- rather than restating a number.
4. Runs the detector in its **positive state** before quoting it in its negative one. Three
   controls, all on real compiled SPIR-V: a tail tile where `QB_COLS` does not divide `N`, a
   deliberately re-reading variant of the same kernel, and the unpacked path. If none of them
   can be constructed the reading is published as `UNWITNESSED` and no amplification is quoted:
   a detector never seen in its positive state has no demonstrated positive state.

WHAT IT STILL DOES NOT ESTABLISH
================================
The trace counts **bytes named by load instructions**, not DRAM transactions. For the weight
stream those coincide -- 1.8 GB streamed once past an 8 MB L2 cannot be anything else -- and
that argument is `probe_roofline.py`'s, unchanged and still an argument. What has changed is
that the multiplicity is now measured: if two workgroups named the same blob, or one workgroup
named it twice, this probe now says so.

Run:  python bench/results/probe_weight_reread.py
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(ROOT / "bench" / "results"))

from spirv_simt import Dispatch, InstrumentError, SpirvModule  # noqa: E402

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

LEDGER = ROOT / "evidence" / "proof_ledger.jsonl"
SHADER_STEM = "q_gemv_matmul_nbits_f16"
SHADER_SRC = ROOT / "rust" / "shaders" / "glsl" / "templates" / "q_gemv.comp"
SHADER_INC = ROOT / "rust" / "shaders" / "include"

# Result-identity contract (issue #19 follow-up, Morpheus review on PR #31): the resolved model
# path and its exact content hash are stamped into the output record below, computed lazily
# (only once the model has already been used successfully) so a PHI35_MODEL override or a
# stale/wrong cached file can never be silently absorbed into the evidence. Reuses the streaming
# SHA-256 helper `model_provenance.sha256_of` rather than a 23rd divergent hasher.
sys.path.insert(0, str(ROOT / "rust" / "tools"))
import model_provenance as _model_provenance  # noqa: E402


def _result_identity() -> dict:
    return {
        "onnx_file": str(MODEL),
        "onnx_sha256": _model_provenance.sha256_of(MODEL),
    }

#: Mirrors of `ops::quant`. Checked against the Rust unit tests' own expectations in
#: `bench/test_weight_reread.py`.
GEMV_RED_WORDS = 2048
GEMV_MAX_COLS = 16
GEMV_MIN_WORKGROUPS = 64
GEMV_MIN_BLOCKS_PER_INVOCATION = 2
GEMV_MAX_ROWS = 4
GEMV_MAX_TILE = 32
GEMV_MAX_GROUPS_Y = 65535

#: Binding of `InB`, the packed weight stream, in `q_gemv.comp`.
INB_BINDING = 1

#: Prefill widths walked at `M > 1`. Small on purpose: the SPIR-V walk executes every lane of
#: every workgroup, so cost is linear in `M`, and the quantity being measured -- weight-read
#: amplification -- is `ceil(M / QB_ROWS)`, which is fully determined by two tile-crossings.
#: `M = 4` crosses a two-row tile twice and `M = 5` leaves a tail row, which is the case a tile
#: gets wrong if it gets anything wrong.
PREFILL_M = (2, 4, 5)


def fnv1a64(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def shader_digest(stem: str, spirv: bytes) -> str:
    """`registry::shader_digest_for` for a single stem, in Python.

    Reimplemented rather than shelled out to because the point is to *match* the ledger's
    number: if this drifts from the Rust the match fails and the probe refuses, which is the
    behaviour wanted. A wrapper that called the Rust could not fail that way.
    """
    inp = bytearray(stem.encode("utf-8"))
    inp.append(0)
    inp += len(spirv).to_bytes(8, "little")
    inp += spirv
    inp.append(0)
    return f"{fnv1a64(bytes(inp)):016x}"


def ledger_digest_for(stem: str) -> str:
    """The digest the proof ledger recorded for runs that dispatched exactly `stem`."""
    if not LEDGER.is_file():
        raise InstrumentError(f"proof ledger absent: {LEDGER}")
    found: set[str] = set()
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        if '"__ledger__"' in line or not line.strip():
            continue
        e = json.loads(line)
        if e.get("shaders") == [stem] and e.get("shader_digest"):
            found.add(e["shader_digest"])
    if not found:
        raise InstrumentError(
            f"no proof-ledger entry dispatched exactly [{stem}]; without one there is no "
            "witnessed digest to bind this reading to a compiled kernel"
        )
    if len(found) > 1:
        raise InstrumentError(
            f"the ledger holds {len(found)} distinct digests for [{stem}]: {sorted(found)}. "
            "Two kernels wearing one stem is exactly the state a digest exists to expose; "
            "this probe will not pick one."
        )
    return found.pop()


def locate_module(stem: str) -> tuple[pathlib.Path, bytes, str]:
    """The compiled module whose digest is the one the ledger witnessed.

    Searched by content, never by path. A build directory is a guess; a digest is not.
    """
    want = ledger_digest_for(stem)
    candidates = sorted((ROOT / "rust" / "target").rglob(f"{stem}.spv"))
    seen: dict[str, pathlib.Path] = {}
    for p in candidates:
        blob = p.read_bytes()
        d = shader_digest(stem, blob)
        seen.setdefault(d, p)
        if d == want:
            return p, blob, want
    raise InstrumentError(
        f"no compiled `{stem}.spv` under rust/target has the ledger's digest {want}. "
        f"Found {len(candidates)} candidate(s) with digests {sorted(seen)}. Build the release "
        "EP (`cargo build --release -p onnxruntime-ep-vulkan`) or regenerate the ledger; this "
        "probe will not read a module the ledger has never seen."
    )


# -- host-side dispatch geometry ---------------------------------------------------------------


def gemv_workgroup(blocks_per_col: int) -> int:
    best = None
    wg = 32
    while wg <= 256:
        if (blocks_per_col % wg == 0
                and blocks_per_col // wg >= GEMV_MIN_BLOCKS_PER_INVOCATION):
            best = wg
        wg *= 2
    if best is not None:
        return best
    wg = 32
    while wg < blocks_per_col and wg < 256:
        wg *= 2
    return wg


def gemv_cols(n: int, wg: int) -> int:
    cols = min(GEMV_MAX_COLS, max(GEMV_RED_WORDS // wg, 1))
    while cols > 1 and (n % cols != 0 or n // cols < GEMV_MIN_WORKGROUPS):
        cols //= 2
    return cols


def gemv_packed(bits: int, block_size: int) -> bool:
    return (block_size * bits // 8) % 16 == 0


def gemv_named_bytes(m: int, n: int, k: int, bits: int, a_bytes: int,
                     cols: int, rows: int) -> int:
    """Bytes the grid names, exactly as the SPIR-V walk counts them.

    A tile names its whole weight column strip once and its whole activation row strip once, so
    the weight stream is amplified by the number of row tiles and the activation stream by the
    number of column tiles. The two are independent, which is why a tile has to be chosen by a
    model over both and not by maximising `rows`.
    """
    row_tiles = -(-m // rows)
    col_tiles = -(-n // cols)
    weight = row_tiles * col_tiles * cols * k * bits // 8
    activation = row_tiles * col_tiles * rows * k * a_bytes
    return weight + activation


def gemv_tile(m: int, n: int, k: int, bits: int, a_bytes: int, wg: int) -> tuple[int, int]:
    """`ops::quant::gemv_tile`, in Python. Returns `(cols, rows)`."""
    base_cols = gemv_cols(n, wg)
    best = (base_cols, 1)
    best_bytes = gemv_named_bytes(m, n, k, bits, a_bytes, base_cols, 1)
    if m <= 1:
        return best
    rows = 2
    while rows <= GEMV_MAX_ROWS:
        cols = base_cols
        while True:
            legal = (cols * rows <= GEMV_MAX_TILE
                     and wg * cols <= GEMV_RED_WORDS
                     and n % cols == 0
                     and (cols == 1 or n // cols >= GEMV_MIN_WORKGROUPS))
            if legal:
                got = gemv_named_bytes(m, n, k, bits, a_bytes, cols, rows)
                if got < best_bytes:
                    best = (cols, rows)
                    best_bytes = got
                break
            if cols == 1:
                break
            cols //= 2
        rows *= 2
    return best


# -- the graph ---------------------------------------------------------------------------------


def initializer_bytes(t) -> int:
    """Bytes of one initializer, from the graph. External data counts its `length` field."""
    for d in t.external_data:
        if d.key == "length":
            return int(d.value)
    if t.raw_data:
        return len(t.raw_data)
    raise InstrumentError(
        f"initializer `{t.name}` has neither raw_data nor an external `length`; its size is "
        "not readable from this graph and this probe will not infer it from the shape"
    )


def census(model: pathlib.Path) -> dict:
    """Every `MatMulNBits` node with its shape, its dispatch geometry, and its B bytes."""
    try:
        import onnx
    except ImportError as e:  # pragma: no cover
        raise InstrumentError(f"onnx not importable: {e}") from e
    m = onnx.load(str(model), load_external_data=False)
    init = {i.name: i for i in m.graph.initializer}
    nodes = []
    for n in m.graph.node:
        if n.op_type != "MatMulNBits":
            continue
        a = {at.name: at.i for at in n.attribute if at.type == onnx.AttributeProto.INT}
        K, N = a.get("K"), a.get("N")
        bits, block = a.get("bits"), a.get("block_size")
        if None in (K, N, bits, block):
            raise InstrumentError(f"`{n.name}` is missing a shape attribute")
        b_name = n.input[1]
        if b_name not in init:
            raise InstrumentError(
                f"`{n.name}`'s B input `{b_name}` is not an initializer; its bytes are not a "
                "property of the graph and this probe will not model them"
            )
        bpc = K // block
        wg = gemv_workgroup(bpc)
        cols = gemv_cols(N, wg)
        nodes.append({
            "name": n.name, "K": K, "N": N, "bits": bits, "block_size": block,
            "blocks_per_col": bpc, "blob_bytes": block * bits // 8,
            "wg": wg, "cols": cols, "packed": int(gemv_packed(bits, block)),
            "groups_x": -(-N // cols),
            "b_initializer": b_name,
            "b_initializer_bytes": initializer_bytes(init[b_name]),
            "has_zero_points": len(n.input) > 3 and bool(n.input[3]),
        })
    if not nodes:
        raise InstrumentError("no MatMulNBits nodes in the graph")
    return {"count": len(nodes), "nodes": nodes}


# -- the walk ----------------------------------------------------------------------------------


def walk_shape(mod: SpirvModule, K: int, N: int, bits: int, block: int, wg: int, cols: int,
               packed: int, has_zp: int = 0, m_total: int = 1, rows: int = 1) -> dict:
    """Execute the whole grid for one node shape and reduce its `InB` trace."""
    bpc = K // block
    bb = block * bits // 8
    b_words = N * bpc * bb // 4
    row_tiles = -(-m_total // rows)
    groups_y = min(row_tiles, GEMV_MAX_GROUPS_Y)
    buffers = {
        0: np.zeros(max(1, m_total * K // 2), dtype=np.uint32),
        1: np.zeros(max(1, b_words), dtype=np.uint32),
        2: np.zeros(max(1, N * bpc // 2 + 1), dtype=np.uint32),
        3: np.zeros(max(1, N * ((bpc * bits + 7) // 8) // 4 + 2), dtype=np.uint32),
        4: np.zeros(max(1, (m_total * N + 1) // 2), dtype=np.uint32),
    }
    d = Dispatch(
        groups=(-(-N // cols), groups_y, 1),
        local_size=(wg, 1, 1),
        spec={0: wg, 1: bits, 2: block, 3: has_zp, 4: cols, 5: packed, 6: rows},
        push_constants=[m_total, K, N, bpc],
        buffers=buffers,
    )
    tr = mod.run(d, trace_binding=INB_BINDING)
    graph_bytes = N * bpc * bb
    by_width: dict[str, int] = {}
    for count, w in tr.sites.values():
        key = f"width_{w * 4}B"
        by_width[key] = by_width.get(key, 0) + count
    widths = sorted({w * 4 for _, w in tr.sites.values()})
    return {
        "K": K, "N": N, "bits": bits, "block_size": block, "blocks_per_col": bpc,
        "blob_bytes": bb, "wg": wg, "cols": cols, "rows": rows, "packed": packed,
        "has_zero_points": has_zp, "m_total": m_total,
        "row_tiles": row_tiles, "groups_y": groups_y,
        "workgroups": -(-N // cols) * groups_y,
        "load_instructions": tr.load_instructions,
        "load_widths_bytes": widths,
        "loads_by_width": by_width,
        "named_bytes": tr.named_bytes,
        "distinct_bytes_touched": tr.touched_words * 4,
        "blob_bytes_total": graph_bytes,
        "amplification": tr.named_bytes / graph_bytes,
        "coverage": tr.touched_words * 4 / graph_bytes,
        "max_loads_naming_one_word": tr.max_reads_per_word,
        "words_named_by_more_than_one_workgroup": tr.words_read_by_more_than_one_workgroup,
        "loads_per_blob": tr.load_instructions / (N * bpc),
    }


# -- positive controls -------------------------------------------------------------------------


def _tool(name: str) -> str | None:
    sdk = os.environ.get("VULKAN_SDK")
    if sdk:
        cand = pathlib.Path(sdk) / "Bin" / (name + (".exe" if os.name == "nt" else ""))
        if cand.is_file():
            return str(cand)
    return shutil.which(name)


#: The edit that makes the packed 4-bit path read every blob twice. Surgical on purpose: it
#: changes the number of loads and nothing else, so a probe that fails to see it is not seeing
#: loads.
#:
#: The literal is indented to match the `QB_ROWS == 1u` arm of `main()`, which is where the
#: packed decode loop lives since the row tile was added (issue #7). It is an exact literal
#: rather than a regex because a control that quietly matches *something* is not a control:
#: `compile_rereading_variant` returns `None` when the text moves, and `positive_controls`
#: reports the miss instead of inventing a substitute.
REREAD_FROM = """                        uint nchunk = bb >> 4u;
                        if (QB_BITS == 4u) {
                            for (uint ch = 0u; ch < nchunk; ++ch) {"""
REREAD_TO = """                        uint nchunk = (bb >> 4u) * 2u;
                        if (QB_BITS == 4u) {
                            for (uint chr = 0u; chr < nchunk; ++chr) {
                                uint ch = chr >> 1u;"""


def compile_rereading_variant() -> tuple[bytes, str] | None:
    """Compile a variant of the same shader that loads every blob twice.

    Also compiles the *unmodified* source with the build's own flags and reports whether that
    reproduces the ledger's digest. When it does, the control is exactly "the shipped kernel
    plus this one edit" rather than "some kernel that resembles it".
    """
    glslc = _tool("glslc")
    if glslc is None or not SHADER_SRC.is_file():
        return None
    src = SHADER_SRC.read_text(encoding="utf-8")
    if REREAD_FROM not in src:
        return None
    patched = src.replace(REREAD_FROM, REREAD_TO, 1)
    flags = ["-fshader-stage=compute", "--target-env=vulkan1.1", "-O",
             "-DSCALAR_T=uint", "-DDTYPE_F16", "-I", str(SHADER_INC)]
    with tempfile.TemporaryDirectory(dir=str(ROOT / "bench" / "results")) as td:
        td = pathlib.Path(td)
        outs = {}
        for name, text in (("base", src), ("reread", patched)):
            comp = td / f"q_gemv_{name}.comp"
            spv = td / f"q_gemv_{name}.spv"
            comp.write_text(text, encoding="utf-8")
            p = subprocess.run([glslc, *flags, str(comp), "-o", str(spv)],
                               capture_output=True, text=True)
            if p.returncode != 0:
                return None
            outs[name] = spv.read_bytes()
    rebuilt = shader_digest(SHADER_STEM, outs["base"])
    return outs["reread"], rebuilt


def positive_controls(mod: SpirvModule) -> dict:
    """Show the detector in its positive state, or say it has not been seen there.

    Switch's rule. Each control names, in advance, the number it expects to move and the
    direction, and the control is only a control if the observed value differs from 1.
    """
    controls = []

    # 1. A tail tile: `QB_COLS` does not divide `N`, so the shader's out-of-range redirect
    #    points the surplus columns back at `col0` and that tile re-reads column `col0`'s
    #    blobs. Real module, real specialization, a shape the host would never choose.
    tail = walk_shape(mod, K=64, N=130, bits=4, block=32, wg=32, cols=16, packed=1)
    controls.append({
        "control": "tail_tile_N_not_divisible_by_cols",
        "module": "the shipped module, at N=130 cols=16",
        "predicted": "amplification > 1: the surplus columns redirect onto col0",
        "amplification": tail["amplification"],
        "max_loads_naming_one_word": tail["max_loads_naming_one_word"],
        "positive": tail["amplification"] > 1.0,
        "detail": tail,
    })

    # 2. The same kernel, edited to load every blob twice. This is the counterfactual the
    #    original claim needed and never had: "if a kernel change made the packed path re-read
    #    every blob, would this probe say so?"
    blob = compile_rereading_variant()
    if blob is None:
        controls.append({
            "control": "deliberately_rereading_variant",
            "positive": False,
            "detail": "glslc unavailable or the shader text moved; variant not built",
        })
    else:
        spirv, rebuilt_digest = blob
        rr = SpirvModule(spirv)
        got = walk_shape(rr, K=64, N=128, bits=4, block=32, wg=32, cols=16, packed=1)
        controls.append({
            "control": "deliberately_rereading_variant",
            "module": "q_gemv.comp with the packed 4-bit chunk loop doubled",
            "predicted": "amplification == 2.0 on a shape whose amplification is 1.0 shipped",
            "unmodified_source_rebuild_digest": rebuilt_digest,
            "rebuild_reproduces_ledger_digest": rebuilt_digest == ledger_digest_for(
                SHADER_STEM),
            "amplification": got["amplification"],
            "max_loads_naming_one_word": got["max_loads_naming_one_word"],
            "positive": got["amplification"] > 1.5,
            "detail": got,
        })

    # 3. The unpacked path. Its *bytes* are the same -- four 4-byte loads in place of one
    #    16-byte one -- so this control is about `load_width_bytes`, which the old record also
    #    stated as a literal. The walk reads the width off the SPIR-V result type.
    unpacked = walk_shape(mod, K=64, N=128, bits=4, block=32, wg=32, cols=16, packed=0)
    packed = walk_shape(mod, K=64, N=128, bits=4, block=32, wg=32, cols=16, packed=1)
    controls.append({
        "control": "unpacked_path_changes_the_width_not_the_bytes",
        "predicted": "width 16 -> 4 B, loads x4, named bytes unchanged",
        "packed_widths_bytes": packed["load_widths_bytes"],
        "unpacked_widths_bytes": unpacked["load_widths_bytes"],
        "packed_loads": packed["load_instructions"],
        "unpacked_loads": unpacked["load_instructions"],
        "named_bytes_equal": packed["named_bytes"] == unpacked["named_bytes"],
        "positive": (packed["load_widths_bytes"] != unpacked["load_widths_bytes"]
                     and unpacked["load_instructions"] == 4 * packed["load_instructions"]),
    })

    # 4. The defect this probe was extended to see, and its repair, on the same module. Forcing
    #    `rows = 1` at `M = 4` reproduces the pre-tile grid exactly -- four y-workgroups each
    #    naming the whole weight strip -- and the number moves to 4.0. Selecting the host's tile
    #    at the same M moves it back to `ceil(M / rows)`. The first half is the control (a state
    #    where the answer is not 1); the second half is the measurement.
    untiled = walk_shape(mod, K=3072, N=256, bits=4, block=32, wg=32, cols=16, packed=1,
                         m_total=4, rows=1)
    tiled = walk_shape(mod, K=3072, N=256, bits=4, block=32, wg=32, cols=16, packed=1,
                       m_total=4, rows=2)
    controls.append({
        "control": "row_tile_removes_the_M_fold_weight_reread",
        "module": "the shipped module, at M=4 with QB_ROWS forced to 1 and then to 2",
        "predicted": "amplification 4.0 untiled -> 2.0 tiled; ceil(M / QB_ROWS) either way",
        "untiled_amplification": untiled["amplification"],
        "tiled_amplification": tiled["amplification"],
        "untiled_max_loads_naming_one_word": untiled["max_loads_naming_one_word"],
        "tiled_max_loads_naming_one_word": tiled["max_loads_naming_one_word"],
        "positive": (untiled["amplification"] == 4.0 and tiled["amplification"] == 2.0),
        "detail": {"untiled": untiled, "tiled": tiled},
    })

    witnessed = any(c.get("positive") for c in controls[:2])
    return {"witnessed": witnessed, "controls": controls}


# -- main --------------------------------------------------------------------------------------


def main() -> int:
    try:
        path, blob, digest = locate_module(SHADER_STEM)
        mod = SpirvModule(blob)
        if not MODEL.is_file():
            raise InstrumentError(f"model absent: {MODEL}")
        cen = census(MODEL)

        pos = positive_controls(mod)

        shapes: dict[tuple, dict] = {}
        for nd in cen["nodes"]:
            key = (nd["K"], nd["N"], nd["bits"], nd["block_size"], nd["wg"], nd["cols"],
                   nd["packed"], int(nd["has_zero_points"]))
            e = shapes.setdefault(key, {"nodes": 0, "b_bytes": 0})
            e["nodes"] += 1
            e["b_bytes"] += nd["b_initializer_bytes"]

        per_shape = []
        total_loads = 0
        total_named = 0
        max_word = 0
        multi_wg = 0
        min_coverage = 1.0
        for key, agg in sorted(shapes.items()):
            K, N, bits, block, wg, cols, packed, has_zp = key
            r = walk_shape(mod, K=K, N=N, bits=bits, block=block, wg=wg, cols=cols,
                           packed=packed, has_zp=has_zp)
            r["nodes_with_this_shape"] = agg["nodes"]
            r["b_initializer_bytes_all_nodes"] = agg["b_bytes"]
            r["blob_model_agrees_with_initializers"] = (
                r["blob_bytes_total"] * agg["nodes"] == agg["b_bytes"])
            per_shape.append(r)
            total_loads += r["load_instructions"] * agg["nodes"]
            total_named += r["named_bytes"] * agg["nodes"]
            max_word = max(max_word, r["max_loads_naming_one_word"])
            multi_wg += r["words_named_by_more_than_one_workgroup"] * agg["nodes"]
            min_coverage = min(min_coverage, r["coverage"])

        graph_bytes = sum(n["b_initializer_bytes"] for n in cen["nodes"])
        widths = sorted({w for r in per_shape for w in r["load_widths_bytes"]})

        # -- prefill: the same graph at M > 1, tiled and untiled ---------------------------
        # The decode walk above is `M = 1`, where the row tile is by construction absent. Issue
        # #7 is about `M > 1`, so the same shapes are walked again at representative prefill
        # widths, twice each: once with `QB_ROWS` forced to 1 (the geometry before this change)
        # and once with the tile `ops::quant::gemv_tile` actually selects. Both are executions of
        # the same ledger-bound module; the pair is what makes the reduction a measurement rather
        # than a subtraction of one run from a remembered number.
        prefill = []
        for m in PREFILL_M:
            m_named_untiled = 0
            m_named_tiled = 0
            for key, agg in sorted(shapes.items()):
                K, N, bits, block, wg, cols, packed, has_zp = key
                tile_cols, tile_rows = gemv_tile(m, N, K, bits, 2, wg)
                u = walk_shape(mod, K=K, N=N, bits=bits, block=block, wg=wg, cols=cols,
                               packed=packed, has_zp=has_zp, m_total=m, rows=1)
                t = walk_shape(mod, K=K, N=N, bits=bits, block=block, wg=wg, cols=tile_cols,
                               packed=packed, has_zp=has_zp, m_total=m, rows=tile_rows)
                prefill.append({
                    "m_total": m, "K": K, "N": N, "bits": bits, "block_size": block,
                    "nodes_with_this_shape": agg["nodes"],
                    "selected_cols": tile_cols, "selected_rows": tile_rows,
                    "untiled_amplification": u["amplification"],
                    "tiled_amplification": t["amplification"],
                    "predicted_tiled_amplification": -(-m // tile_rows),
                    "untiled_load_widths_bytes": u["load_widths_bytes"],
                    "tiled_load_widths_bytes": t["load_widths_bytes"],
                    "untiled_named_bytes": u["named_bytes"],
                    "tiled_named_bytes": t["named_bytes"],
                    "untiled_groups_y": u["groups_y"], "tiled_groups_y": t["groups_y"],
                    "coverage_untiled": u["coverage"], "coverage_tiled": t["coverage"],
                })
                m_named_untiled += u["named_bytes"] * agg["nodes"]
                m_named_tiled += t["named_bytes"] * agg["nodes"]
            prefill.append({
                "m_total": m, "K": None, "N": "ALL SHAPES",
                "untiled_amplification": m_named_untiled / graph_bytes,
                "tiled_amplification": m_named_tiled / graph_bytes,
                "untiled_named_bytes": m_named_untiled,
                "tiled_named_bytes": m_named_tiled,
                "weight_bytes_not_named_because_of_the_tile":
                    m_named_untiled - m_named_tiled,
            })

        if pos["witnessed"]:
            amplification = total_named / graph_bytes
            verdict = (
                "each weight byte is named by exactly one load instruction"
                if amplification == 1.0 and max_word == 1
                else f"amplification {amplification:.6f}; the weight stream is re-read"
            )
        else:
            amplification = None
            verdict = (
                "UNWITNESSED: no positive control could be constructed, so this probe has no "
                "demonstrated positive state and no amplification is quoted from it"
            )

        report = {
            **_result_identity(),
            "probe": "weight_reread_amplification_phi35_executed",
            "no_clock": "Every number here is a count of instructions or of bytes.",
            "subject": {
                "shader_stem": SHADER_STEM,
                "shader_digest": digest,
                "digest_source": "evidence/proof_ledger.jsonl, entries dispatching this stem",
                #: `locate_module` refuses unless the module's content digest equals the one
                #: the ledger records for a run that dispatched this stem, so reaching here at
                #: all means the walk is bound to the compiled kernel a run actually used.
                "digest_matches_ledger": True,
                "module_path": str(path.relative_to(ROOT)),
                "module_bytes": len(blob),
                "spec_constants": "[local_size_x, bits, block_size, has_zp, cols, packed], "
                                  "resolved per node by the ops::quant mirrors below",
            },
            "denominator": {
                "source": "ONNX initializers of each MatMulNBits B input "
                          "(raw_data length, or the external-data `length` field)",
                "int4_weight_bytes_from_graph": graph_bytes,
                "matmulnbits_nodes": cen["count"],
            },
            "positive_controls": pos,
            "measured": {
                "inb_load_instructions_per_inference": total_loads,
                "load_widths_bytes_observed": widths,
                "named_bytes_per_inference": total_named,
                "amplification": amplification,
                "max_loads_naming_one_word": max_word,
                "words_named_by_more_than_one_workgroup": multi_wg,
                "min_coverage_of_the_weight_tensor": min_coverage,
                "verdict": verdict,
            },
            "by_shape": per_shape,
            "by_shape_prefill": prefill,
            "prefill_m_values": PREFILL_M,
        }
    except InstrumentError as e:
        print(f"ERROR(instrument): {e}", file=sys.stderr)
        return 4

    out = ROOT / "bench" / "results" / "weight_reread_phi35.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("WEIGHT RE-READ — executed against the compiled kernel, not asserted")
    print("=" * 78)
    print(f"  module           {SHADER_STEM}  digest {digest}  ({len(blob):,} B)")
    print(f"  denominator      ONNX initializers over {cen['count']} MatMulNBits nodes")
    print()
    print("  POSITIVE CONTROLS — the detector seen in a state where the answer is not 1")
    for c in pos["controls"]:
        amp = c.get("amplification")
        shown = f"amplification {amp:.6f}" if isinstance(amp, float) else c.get("detail", "")
        flag = "SEEN" if c.get("positive") else "not seen"
        print(f"    [{flag:>8}] {c['control']}: {shown}")
    print()
    print("  MEASURED — whole grid of every node, every load of binding 1 recorded")
    print(f"    InB load instructions / inference   {total_loads:>15,}")
    print(f"    load widths observed (B)            {str(widths):>15}")
    print(f"    bytes named by those loads          {total_named:>15,}")
    print(f"    int4 weight bytes from the graph    {graph_bytes:>15,}")
    if amplification is None:
        print("    amplification                            UNWITNESSED")
    else:
        print(f"    amplification                       {amplification:>15.6f}")
    print(f"    max loads naming one 4-byte word    {max_word:>15,}")
    print(f"    words named by >1 workgroup         {multi_wg:>15,}")
    print(f"    coverage of the weight tensor       {min_coverage:>15.6f}")
    print()
    print("  PREFILL — the same 161 nodes at M > 1, QB_ROWS forced to 1 vs the selected tile")
    for r in prefill:
        if r["N"] != "ALL SHAPES":
            continue
        m = r["m_total"]
        print(f"    M={m:<4} amplification  untiled {r['untiled_amplification']:>9.6f}"
              f"   tiled {r['tiled_amplification']:>9.6f}"
              f"   weight bytes not named {r['weight_bytes_not_named_because_of_the_tile']:>15,}")
    print(f"\n  -> {verdict}")
    print(f"\n  record: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
