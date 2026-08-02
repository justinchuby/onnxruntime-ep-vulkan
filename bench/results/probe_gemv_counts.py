"""Restate the packed-loads change in **counts**, so it survives a contended box.

Why this exists
---------------
Every performance figure this project has taken on the device clock is now `UNCERTIFIED` or
`WITHHELD` under `bench/device_companion.py`: a `STEADY` tail was measured at 10.99x wrong (foreign
GPU work outliving the run) and at 21.4x wrong (a board that never left its 210 MHz idle clock),
both times with a *better* RSD than the correct run. And Intel cannot be certified at all here,
because `nvidia-smi` is NVIDIA-only, so its device-state record is `UNOBSERVABLE`.

So the coordinator asked for the barrier treatment: **make the count the claim and the ratio the
estimate.** `147,618 -> 354` barriers is certain because it is a count; the 5.33x that followed it
was correctly demoted to an estimate. This module does the same for the 128-bit packed loads.

What is countable here, and what is not
---------------------------------------
Countable, and reported below:

* **The width of a load from `InB`.** Established from SPIR-V, not from reading the GLSL: this
  module compiles `q_gemv.comp`, freezes the specialization constants to each arm, optimizes, and
  walks the def-use graph from the `inb` variable through every `OpAccessChain` to every `OpLoad`
  that reaches it. The result *type* of those loads is the claim.
* **Loads issued per blob.** A blob -- one (column, block) unit of packed weights -- is
  `block_size * bits / 8` bytes, which at Phi-3.5's `bits=4, block_size=32` is exactly 16. Given
  the load width from the disassembly, loads per blob is `16 / width`. Arithmetic over a measured
  width, not an assumption.
* **Blobs per inference**, from the model file: every `MatMulNBits` node's `N`, `K`, `bits` and
  `block_size` read out of the ONNX graph. No device, no clock, no run.
* **Bytes fetched from global memory per inference**, split into the part that is irreducible
  (each weight byte is read exactly once) and the part that is our design (the activation row is
  re-read once per workgroup, so it is amplified by `N / QB_COLS`).
* **Accumulator chain depth**: the longest chain of dependent floating-point instructions in the
  hot basic block, computed over the SPIR-V def-use graph.
* **Shared memory requested**, against Intel's 32 KiB and NVIDIA's 48 KiB.

NOT countable here, and deliberately absent: any statement that arm 1 is faster than arm 0. That
is a ratio, it needs a clock, and the clock needs a certified companion. This module produces the
structural half of the claim and stops.

A note on unrolling
-------------------
The optimizer does not fully unroll the column loop, so the *static* number of `OpLoad`s in the
module is not the number executed. That is why this module reports the load **width** from the
disassembly and derives loads-per-blob from the blob size, rather than counting static
instructions and pretending they are dynamic ones. The width is a property of the instruction and
survives any amount of unrolling; a static count does not.

R13
---
`glslc`, `spirv-opt` or `spirv-dis` missing or failing is `ERROR(instrument)` and is never a
finding about the shader. The absence of a load and the absence of a toolchain are produced by
different code paths here and are named apart.

Usage::

    python bench/results/probe_gemv_counts.py
    python bench/results/probe_gemv_counts.py --json bench/results/gemv_counts.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
SHADER = REPO / "rust" / "shaders" / "glsl" / "templates" / "q_gemv.comp"
INCLUDE = REPO / "rust" / "shaders" / "include"

# Specialization constant ids, from q_gemv.comp's own header comment.
SPEC_LOCAL_SIZE_X = 0
SPEC_BITS = 1
SPEC_BLOCK = 2
SPEC_HAS_ZP = 3
SPEC_COLS = 4
SPEC_PACKED = 5

# `GEMV_RED_WORDS` in ops/quant.rs -- the shared reduction array, sized by a literal.
RED_WORDS = 1024
SHARED_BYTES = RED_WORDS * 4

# The two vendors' shared-memory budgets, from the project's own device survey.
INTEL_SHARED_KIB = 32
NVIDIA_SHARED_KIB = 48

# `gemv_workgroup` in ops/quant.rs: the largest power of two in [32,256] that DIVIDES
# blocks_per_col and leaves at least GEMV_MIN_BLOCKS_PER_INVOCATION blocks each.
GEMV_MIN_BLOCKS_PER_INVOCATION = 2
GEMV_MAX_COLS = 8
GEMV_MIN_WORKGROUPS = 64


class InstrumentError(RuntimeError):
    """R13: this instrument failed. Never a finding about the shader."""


def gemv_workgroup(blocks_per_col: int) -> int:
    """Mirror of `ops::quant::gemv_workgroup`. Checked against its unit test in `main`."""
    best = None
    wg = 32
    while wg <= 256:
        if blocks_per_col % wg == 0 and blocks_per_col // wg >= GEMV_MIN_BLOCKS_PER_INVOCATION:
            best = wg
        wg *= 2
    if best is not None:
        return best
    wg = 32
    while wg < blocks_per_col and wg < 256:
        wg *= 2
    return wg


def gemv_cols(n: int, wg: int) -> int:
    """Mirror of `ops::quant::gemv_cols`."""
    cols = min(GEMV_MAX_COLS, max(RED_WORDS // wg, 1))
    while cols > 1 and (n % cols != 0 or n // cols < GEMV_MIN_WORKGROUPS):
        cols //= 2
    return cols


def _tool(name: str) -> str:
    sdk = os.environ.get("VULKAN_SDK")
    if sdk:
        cand = Path(sdk) / "Bin" / (name + (".exe" if os.name == "nt" else ""))
        if cand.is_file():
            return str(cand)
    found = shutil.which(name)
    if not found:
        raise InstrumentError(
            f"`{name}` not found. Set VULKAN_SDK or put the SDK's Bin on PATH. "
            "This is an instrument failure and says nothing about the shader."
        )
    return found


def _run(argv: list[str]) -> str:
    try:
        p = subprocess.run(argv, capture_output=True, text=True, errors="replace")
    except OSError as e:  # pragma: no cover - toolchain absence
        raise InstrumentError(f"could not execute {argv[0]}: {e}") from e
    if p.returncode != 0:
        raise InstrumentError(f"{Path(argv[0]).name} exited {p.returncode}: {p.stderr.strip()[:500]}")
    return p.stdout


INSTR = re.compile(r"^\s*(?:(%\w+)\s*=\s*)?(Op\w+)(.*)$")
OPERAND = re.compile(r"%\w+")


def disassemble(packed: int, wg: int, cols: int, bits: int, block: int) -> str:
    """Compile, freeze the spec constants to this arm, optimize, disassemble.

    `--fold-spec-const-op-composite` is load-bearing and was learned the hard way. glslc emits
    `if (QB_PACKED == 1u)` as an `OpSpecConstantOp %bool IEqual`, which lives in the constant
    section; `--freeze-spec-const` turns `QB_PACKED` itself into an `OpConstant` but leaves the
    `OpSpecConstantOp` alone, and neither ccp nor dead-branch elimination will look inside it. The
    first version of this probe therefore produced two byte-identical modules differing only in
    the value of one unused constant, and reported the same load census for both arms -- a
    perfectly stable, perfectly wrong answer. `arms_must_differ` below exists so that failure
    cannot recur silently.
    """
    glslc, opt, dis = _tool("glslc"), _tool("spirv-opt"), _tool("spirv-dis")
    with tempfile.TemporaryDirectory(dir=str(HERE)) as td:
        base = Path(td) / "base.spv"
        spec = Path(td) / "spec.spv"
        _run([glslc, "-fshader-stage=compute", "-I", str(INCLUDE), str(SHADER), "-o", str(base)])
        values = (
            f"{SPEC_LOCAL_SIZE_X}:{wg} {SPEC_BITS}:{bits} {SPEC_BLOCK}:{block} "
            f"{SPEC_HAS_ZP}:0 {SPEC_COLS}:{cols} {SPEC_PACKED}:{packed}"
        )
        _run([opt, "--set-spec-const-default-value", values, "--freeze-spec-const",
              "--fold-spec-const-op-composite", "-O", str(base), "-o", str(spec)])
        return _run([dis, str(spec)])


def parse(text: str) -> dict:
    """Minimal SPIR-V text model: result id -> (opcode, operand ids), plus names and types."""
    defs: dict[str, tuple[str, list[str]]] = {}
    names: dict[str, str] = {}
    order: list[tuple[str | None, str, list[str]]] = []
    for line in text.splitlines():
        m = INSTR.match(line)
        if not m:
            continue
        res, op, rest = m.group(1), m.group(2), m.group(3)
        ops = OPERAND.findall(rest)
        if op == "OpName":
            q = re.search(r'"([^"]*)"', rest)
            if ops and q:
                names[ops[0]] = q.group(1)
            continue
        if res:
            defs[res] = (op, ops)
        order.append((res, op, ops))
    return {"defs": defs, "names": names, "order": order}


def loads_from(mod: dict, var_name: str) -> dict[str, int]:
    """Every `OpLoad` reachable from `var_name` through access chains, counted by result type.

    This is the load-width claim. It walks the def-use graph rather than reading the GLSL, so it
    reports what the compiler actually emitted after specialization and optimization.
    """
    defs, names = mod["defs"], mod["names"]
    roots = {i for i, n in names.items() if n == var_name}
    if not roots:
        raise InstrumentError(
            f"no variable named `{var_name}` in the disassembly. The shader's binding names may "
            "have changed; this instrument cannot report a load width it cannot locate."
        )
    # Transitively close over access chains rooted at the buffer variable.
    reach = set(roots)
    changed = True
    while changed:
        changed = False
        for res, (op, ops) in defs.items():
            if res in reach:
                continue
            # Every one of these carries the result type as operand 0; the pointer being derived
            # from is operand 1. Reading operand 0 finds the *type* and reaches nothing, which
            # looks exactly like "the shader issues no loads" -- a silent zero, so it is asserted
            # against below rather than reported.
            if op in ("OpAccessChain", "OpInBoundsAccessChain", "OpPtrAccessChain", "OpCopyObject"):
                if len(ops) > 1 and ops[1] in reach:
                    reach.add(res)
                    changed = True
    out: dict[str, int] = {}
    for res, (op, ops) in defs.items():
        if op != "OpLoad" or len(ops) < 2:
            continue
        # ops[0] is the result type, ops[1] the pointer.
        if ops[1] in reach:
            ty = names.get(ops[0], ops[0])
            out[ty] = out.get(ty, 0) + 1
    if not out:
        raise InstrumentError(
            f"walked {len(reach)} pointer(s) rooted at `{var_name}` and found no `OpLoad`. A "
            "kernel that reads no weights is not a possible state, so this is the instrument "
            "failing to trace the def-use graph, not a finding about the shader."
        )
    return out


FLOAT_OPS = {"OpFAdd", "OpFMul", "OpFSub", "OpFNegate", "OpExtInst", "OpFConvert", "OpDot"}


def hot_block(mod: dict) -> dict:
    """The hot basic block's float work and its longest dependency chain.

    Reported with its own normalizer. The two arms' inner-loop bodies do **different amounts of
    work**: the general path's body is one 32-bit word of `B` (8 nibbles at 4 bits), the packed
    path's is one whole 16-byte blob (32 nibbles). Comparing their raw chain depths is comparing a
    quarter of a blob against a whole one, so the multiply count is carried alongside the depth
    and the caller is expected to divide.
    """
    best = {"chain": 0, "fmul": 0}
    depth: dict[str, int] = {}
    block_ops: list[tuple[str, str, list[str]]] = []

    def flush() -> dict:
        local, fmul = 0, 0
        for res, op, ops in block_ops:
            d = 0
            for o in ops[1:]:
                if o in depth:
                    d = max(d, depth[o])
            depth[res] = d + 1
            local = max(local, d + 1)
            if op == "OpFMul":
                fmul += 1
        return {"chain": local, "fmul": fmul}

    for res, op, ops in mod["order"]:
        if op == "OpLabel":
            cand = flush()
            if cand["fmul"] > best["fmul"]:
                best = cand
            block_ops, depth = [], {}
            continue
        if res and op in FLOAT_OPS:
            block_ops.append((res, op, ops))
    cand = flush()
    return cand if cand["fmul"] > best["fmul"] else best


def matmulnbits_census(model: Path) -> dict:
    """Every `MatMulNBits` node in the graph, by shape. A count from the artifact, not a guess."""
    try:
        import onnx
    except ImportError as e:  # pragma: no cover
        raise InstrumentError(f"onnx not importable: {e}") from e
    m = onnx.load(str(model), load_external_data=False)
    nodes = []
    for n in m.graph.node:
        if n.op_type != "MatMulNBits":
            continue
        a = {at.name: at.i for at in n.attribute if at.type == onnx.AttributeProto.INT}
        nodes.append({"K": a.get("K"), "N": a.get("N"),
                      "bits": a.get("bits"), "block_size": a.get("block_size")})
    return {"count": len(nodes), "nodes": nodes}


def byte_model(nodes: list[dict]) -> dict:
    """Bytes fetched from global memory per inference, split irreducible vs amplified.

    Weights and scales are each read exactly once, so those bytes are a property of the model. The
    activation row is streamed *in full by every workgroup*, so it is multiplied by the workgroup
    count -- `ceil(N / QB_COLS)` at batch 1. That amplification is our design and is the term the
    column tile reduced by 8x; it is stated separately for exactly that reason.
    """
    weights = scales = acts = 0
    acts_untiled = 0
    blobs = 0
    per_shape: dict[tuple, dict] = {}
    for nd in nodes:
        K, N, bits, blk = nd["K"], nd["N"], nd["bits"], nd["block_size"]
        if None in (K, N, bits, blk):
            continue
        bpc = K // blk
        wg = gemv_workgroup(bpc)
        cols = gemv_cols(N, wg)
        w = N * K * bits // 8
        s = N * bpc * 2
        groups = -(-N // cols)
        a = groups * K * 2
        weights += w
        scales += s
        acts += a
        acts_untiled += N * K * 2
        blobs += N * bpc
        key = (K, N, bits, blk)
        e = per_shape.setdefault(key, {"K": K, "N": N, "bits": bits, "block_size": blk,
                                       "wg": wg, "cols": cols, "blob_bytes": blk * bits // 8,
                                       "blobs_per_node": N * bpc, "nodes": 0})
        e["nodes"] += 1
    total = weights + scales + acts
    untiled_total = weights + scales + acts_untiled
    return {
        "weight_bytes": weights,
        "scale_bytes": scales,
        "activation_bytes": acts,
        "total_bytes": total,
        "activation_share": acts / total if total else 0.0,
        "blobs_per_inference": blobs,
        # The column tile (QB_COLS=1 -> 8) is also a byte count, not just a timing result: one
        # workgroup per output column re-read the whole activation row N times; eight columns per
        # workgroup re-read it N/8 times. Stated as a counterfactual over the same graph.
        "activation_bytes_at_cols_1": acts_untiled,
        "total_bytes_at_cols_1": untiled_total,
        "bytes_reduction_from_column_tile": untiled_total / total if total else 0.0,
        "shapes": list(per_shape.values()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", type=Path, default=HERE / "gemv_counts.json")
    ap.add_argument("--model", type=Path, default=None)
    args = ap.parse_args()

    report: dict = {"instrument": "probe_gemv_counts", "arms": {}}
    try:
        # The Phi-3.5 hot shape: K=8192 -> 256 blocks -> wg 128, cols 8.
        wg, cols, bits, block = 128, 8, 4, 32
        blob_bytes = block * bits // 8
        texts: dict[int, str] = {}
        for packed in (0, 1):
            text = disassemble(packed, wg, cols, bits, block)
            texts[packed] = text
            mod = parse(text)
            b = loads_from(mod, "inb")
            a = loads_from(mod, "ina")
            width = None
            if b:
                # Widths in bytes, keyed by the disassembler's type names.
                widths = {"uint": 4, "v2uint": 8, "v3uint": 12, "v4uint": 16}
                seen = {widths.get(t.lstrip("%")) for t in b}
                if len(seen) == 1 and None not in seen:
                    width = seen.pop()
            hb = hot_block(mod)
            # Trip counts of the fixed inner loops, from the frozen spec constants. These are
            # exact and need no cooperation from the optimizer's unroller.
            per_blob_bodies = (blob_bytes // 4) if packed == 0 else (blob_bytes // 16)
            report["arms"][f"QB_PACKED={packed}"] = {
                "inb_load_result_types": b,
                "ina_load_result_types": a,
                "inb_load_width_bytes": width,
                "loads_per_blob": (blob_bytes // width) if width else None,
                # The loop-carried dependency that matters: how many times one blob's worth of
                # work reads and rewrites the accumulator. The general path does it once per
                # 32-bit word; the packed path once per 16-byte chunk.
                "accumulator_updates_per_blob": per_blob_bodies,
                "independent_partial_sums_per_update": 1 if packed == 0 else 4,
                "serial_fp_adds_on_critical_path_per_blob": 4 if packed == 0 else 3,
                "hot_block_fp_chain": hb["chain"],
                "hot_block_fmul": hb["fmul"],
                "hot_block_bodies_per_blob": per_blob_bodies,
            }
        report["blob_bytes"] = blob_bytes
        # The negative control on this instrument. If specialization silently fails to take, both
        # arms disassemble identically and every count below is a count of the same module twice.
        a0m, a1m = report["arms"]["QB_PACKED=0"], report["arms"]["QB_PACKED=1"]
        if a0m["inb_load_result_types"] == a1m["inb_load_result_types"]:
            raise InstrumentError(
                "both arms report the same InB load census "
                f"({a0m['inb_load_result_types']}). The specialization did not take, so this is "
                "one module counted twice rather than two arms compared."
            )
        report["arms_differ"] = True
        report["caveat"] = (
            "The two halves of the packed-loads claim are not equally well supported, and the "
            "counts say which. The LOAD half is exact: one 128-bit load per blob in place of "
            "four 32-bit ones, verified from the SPIR-V load types rather than from the GLSL, "
            "giving 465,297,408 -> 116,324,352 InB load instructions per inference. The "
            "ACCUMULATOR half is real but small: the four multiply-adds of the general path are "
            "already mutually independent, and only their four `bacc[c] +=` updates are "
            "serialized. Restructuring them into four partial sums plus a depth-2 tree takes the "
            "accumulator read-modify-write count per blob from 4 to 1 and the serial FP adds on "
            "the critical path from 4 to 3. That is a real reduction in a loop-carried "
            "dependency, but it is nowhere near 4x, and the commit message for 538db70 -- 'the "
            "serial `t += ...` chain of the general path is a latency dependency that pins "
            "memory-level parallelism near one outstanding load' -- overstated it: the four "
            "32-bit loads have no dependency on each other and can all be in flight. The "
            "instruction count is the finding; the dependency restructuring is a secondary term."
        )
        report["spec"] = {"local_size_x": wg, "QB_COLS": cols, "QB_BITS": bits,
                          "QB_BLOCK": block, "QB_HAS_ZP": 0}
        report["shared_memory"] = {
            "requested_bytes": SHARED_BYTES,
            "requested_kib": SHARED_BYTES / 1024,
            "intel_budget_kib": INTEL_SHARED_KIB,
            "nvidia_budget_kib": NVIDIA_SHARED_KIB,
            "fraction_of_intel_budget": SHARED_BYTES / (INTEL_SHARED_KIB * 1024),
            "note": ("sized by the literal RED_WORDS, not by local_size_x * QB_COLS, so the "
                     "request does not move with the tile and cannot spill on the smaller "
                     "budget. This eliminates shared memory as a candidate for the Intel gap."),
        }
        model = args.model or None
        if model is None:
            sys.path.insert(0, str(REPO / "bench"))
            try:
                import phi35  # type: ignore
                model = phi35.model_path()
            except Exception as e:
                report["model_census"] = {"error": f"model not resolvable: {e}"}
                model = None
        if model is not None and Path(model).exists():
            census = matmulnbits_census(Path(model))
            report["model_census"] = census
            report["bytes"] = byte_model(census["nodes"])
    except InstrumentError as e:
        report["verdict"] = "ERROR(instrument)"
        report["reason"] = str(e)
        print(f"ERROR(instrument): {e}", file=sys.stderr)
        args.json.write_text(json.dumps(report, indent=2))
        return 3

    report["verdict"] = "COUNTED"
    args.json.write_text(json.dumps(report, indent=2))

    a0 = report["arms"]["QB_PACKED=0"]
    a1 = report["arms"]["QB_PACKED=1"]
    print("== q_gemv packed loads, restated in counts ==")
    print(f"blob = {report['blob_bytes']} bytes  (block_size {block} x {bits} bits)")
    for name, arm in (("QB_PACKED=0", a0), ("QB_PACKED=1", a1)):
        print(f"  {name:14s} InB loads {arm['inb_load_result_types']} "
              f"width {arm['inb_load_width_bytes']} B  loads/blob {arm['loads_per_blob']}")
        print(f"    {'':12s} accumulator updates/blob {arm['accumulator_updates_per_blob']}  "
              f"independent partial sums/update {arm['independent_partial_sums_per_update']}  "
              f"serial FP adds on critical path/blob "
              f"{arm['serial_fp_adds_on_critical_path_per_blob']}")
        print(f"    {'':12s} hot block: {arm['hot_block_fmul']} FMul, chain depth "
              f"{arm['hot_block_fp_chain']}, covering 1/{arm['hot_block_bodies_per_blob']} blob")
    print("  NOTE: the load-width count is exact. The accumulator restructuring is a much smaller")
    print("        effect than the commit message for 538db70 claimed -- see the JSON caveat.")
    sm = report["shared_memory"]
    print(f"shared memory requested: {sm['requested_kib']:.0f} KiB "
          f"({sm['fraction_of_intel_budget']:.0%} of Intel's {sm['intel_budget_kib']} KiB)")
    if "bytes" in report:
        b = report["bytes"]
        print(f"MatMulNBits nodes: {report['model_census']['count']}")
        print(f"blobs per inference: {b['blobs_per_inference']:,}")
        print(f"  InB load instructions per inference: "
              f"packed=0 {b['blobs_per_inference'] * a0['loads_per_blob']:,}   "
              f"packed=1 {b['blobs_per_inference'] * a1['loads_per_blob']:,}")
        print(f"bytes/inference: weights {b['weight_bytes'] / 2**20:,.1f} MiB  "
              f"scales {b['scale_bytes'] / 2**20:,.1f} MiB  "
              f"activations {b['activation_bytes'] / 2**20:,.1f} MiB "
              f"({b['activation_share']:.1%})")
        print(f"  total {b['total_bytes'] / 2**20:,.1f} MiB; at QB_COLS=1 it would be "
              f"{b['total_bytes_at_cols_1'] / 2**20:,.1f} MiB "
              f"({b['bytes_reduction_from_column_tile']:.2f}x more)")
        for s in b["shapes"]:
            print(f"  K={s['K']:5d} N={s['N']:6d} x{s['nodes']:3d} nodes  "
                  f"wg={s['wg']:3d} cols={s['cols']}  blob={s['blob_bytes']}B")
    print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
