"""Probe: what does each criterion-11(c) arm actually read?

Written before the assertions, deliberately. An assertion that encodes a guess about what
the mechanism does is a test of the guess. Output goes to bench/results/.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import test_wiring_census as census  # noqa: E402

_TOOLS = census._REPO_ROOT / "rust" / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))
from ledger_case_models import PLANTED_CONTROL_CASE  # noqa: E402

CASES = census._EVIDENCE_CASES
LEDGER = census._EVIDENCE_LEDGER
OUT = census._REPO_ROOT / "bench" / "results" / "ledger_arms_probe.json"

FIELDS = (
    "proven_key_lookups", "ledger_hits", "ledger_gate", "ledger_miss",
    "ledger_entries", "ledger_faults", "unproven_declines", "claimed_nodes",
    "dispatches_executed", "model_output_equivalence",
)


def arm(tag: str, model: pathlib.Path | None, ledger_file: pathlib.Path | None = None):
    extra = {}
    if model is not None:
        extra[census._ENV_CENSUS_MODEL] = str(model)
    if ledger_file is not None:
        extra[census._ENV_LEDGER_FILE] = str(ledger_file)
    try:
        doc, log = census._run_counters_child(inject=False, tag=tag, extra_env=extra)
    except Exception as exc:  # noqa: BLE001
        return {"tag": tag, "ERROR": f"{type(exc).__name__}: {exc}"[:600]}
    return {"tag": tag, **{k: doc.get(k) for k in FIELDS}}


def _dynamic_mul_f32(path: pathlib.Path) -> pathlib.Path:
    """`mul_f32` with symbolic extents: same op, same dtype, same optional inputs.

    Only the *shape class* component of the proof key changes (`static` -> `runtime-extent`).
    """
    import onnx_ir as ir
    from onnx_ir import DataType as DT

    a = ir.Value(name="a", type=ir.TensorType(DT.FLOAT), shape=ir.Shape(["M", "N"]))
    b = ir.Value(name="b", type=ir.TensorType(DT.FLOAT), shape=ir.Shape(["M", "N"]))
    out = ir.Value(name="out", type=ir.TensorType(DT.FLOAT), shape=ir.Shape(["M", "N"]))
    node = ir.node("Mul", [a, b], outputs=[out])
    graph = ir.Graph([a, b], [out], nodes=[node], name="dyn_mul", opset_imports={"": 21})
    path.write_bytes(ir.to_proto(ir.Model(graph, ir_version=10)).SerializeToString())
    return path


def _cast_f32_to(path: pathlib.Path, to_dtype, name: str) -> pathlib.Path:
    """`Cast` from f32 to `to_dtype`, static extents, one node.

    The pair to this is the *same* function with a different `to`. Op, input dtype, opset bucket,
    shape class and input arity are all identical; the only thing that differs is the destination
    half of the pair-keyed variant, and therefore the only thing that differs in the proof key.
    That is what makes the pair an OBSERVED arm rather than a planted one: both models are ones a
    user could hand us today, both are claimable by the shipped claim table, and the ledger is the
    only thing that separates them.
    """
    import onnx_ir as ir
    from onnx_ir import DataType as DT

    a = ir.Value(name="a", type=ir.TensorType(DT.FLOAT), shape=ir.Shape([4, 8]))
    out = ir.Value(name="out", type=ir.TensorType(to_dtype), shape=ir.Shape([4, 8]))
    node = ir.node("Cast", [a], outputs=[out], attributes={"to": int(to_dtype)})
    graph = ir.Graph([a], [out], nodes=[node], name=name, opset_imports={"": 21})
    path.write_bytes(ir.to_proto(ir.Model(graph, ir_version=10)).SerializeToString())
    return path


def main() -> int:
    tampered = census._REPO_ROOT / "bench" / "results" / "ledger_tampered.jsonl"
    lines = LEDGER.read_text(encoding="utf-8").splitlines()
    tampered.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    results = census._REPO_ROOT / "bench" / "results"
    dyn = _dynamic_mul_f32(results / "dyn_mul_f32.onnx")

    # The planted control's identity is imported, never spelled. It moved once already — from
    # `mul_f16_unproven` to `sub_f16_dyn_unproven`, because populating the op-suite ledger proved
    # the old plant's form and the control fired on its own author — and this arm was still naming
    # the old file, so it produced ERROR(instrument) and the probe exited 0 anyway. A control that
    # names its plant by hand rots the moment the plant moves, and a probe that exits 0 with an
    # errored arm hands a reader an artifact whose gap is invisible. Both are fixed here.
    from onnx_ir import DataType as _DT

    rows = [
        arm("probe_chain", None),
        arm("probe_mulf32", CASES / "mul_f32.onnx"),
        arm("probe_planted_unproven", CASES / f"{PLANTED_CONTROL_CASE}.onnx"),
        arm("probe_muldyn", dyn),
        arm("probe_nbits_zp", CASES / "matmulnbits_f16_scales_zp.onnx"),
        arm("probe_nbits_noz", CASES / "matmulnbits_f16_scales.onnx"),
        arm("probe_digest_same", CASES / "mul_f32.onnx", LEDGER),
        arm("probe_digest_drift", CASES / "mul_f32.onnx", tampered),
        arm("probe_digest_absent", CASES / "mul_f32.onnx",
            results / "no_such_ledger.jsonl"),
        # OBSERVED pair. Everything about these two models is the same but the `to` attribute.
        arm("probe_cast_proven_pair", CASES / "cast_f32_to_i32.onnx"),
        arm("probe_cast_unproven_pair",
            _cast_f32_to(results / "cast_f32_to_u8.onnx", _DT.UINT8, "cast_f32_u8")),
    ]
    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    for r in rows:
        print(json.dumps(r))
    errored = [r["tag"] for r in rows if "ERROR" in r]
    if errored:
        print(
            "ERROR(instrument): these arms reached no observation and this artifact is "
            f"incomplete: {errored}. R13 — an instrument outage is never reported as a green run."
        )
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
