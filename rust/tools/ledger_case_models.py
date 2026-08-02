#!/usr/bin/env python
"""Build the small ONNX artifacts the §8.9 proof ledger is bootstrapped from. Owner: Mouse.

These are *builders*, not committed models: the ledger records the builder that produced a case
(§8.9.2 rule 3 — "the artifact **or builder** the case came from"), and a builder that is in the
repository is reproducible in a way a binary blob in `bench/results/` is not.

Each function returns a model whose graph is one node of one form, because the ledger's unit is
the dispatchable form and a multi-node model would prove a graph rather than a form.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper


def _binary(op: str, elem: int, shape=(4, 8), opset: int = 17) -> onnx.ModelProto:
    a = helper.make_tensor_value_info("A", elem, list(shape))
    b = helper.make_tensor_value_info("B", elem, list(shape))
    c = helper.make_tensor_value_info("C", elem, list(shape))
    node = helper.make_node(op, ["A", "B"], ["C"], name=f"{op.lower()}0")
    graph = helper.make_graph([node], f"{op}_graph", [a, b], [c])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


def _unary(op: str, elem: int, shape=(4, 8), opset: int = 17) -> onnx.ModelProto:
    x = helper.make_tensor_value_info("X", elem, list(shape))
    y = helper.make_tensor_value_info("Y", elem, list(shape))
    node = helper.make_node(op, ["X"], ["Y"], name=f"{op.lower()}0")
    graph = helper.make_graph([node], f"{op}_graph", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


def _matmulnbits(
    with_zero_points: bool, K: int = 32, N: int = 32, bits: int = 4, block_size: int = 32
) -> onnx.ModelProto:
    """The pair §8.9 was written around.

    The all-zero-logits defect of 2026-07-30 was `MatMulNBits` **with** `zero_points` executed by
    a kernel proven **without** them. The two graphs below differ in exactly one thing — whether
    the optional fourth input is populated — which is why `populated_optional_input_set` is a
    component of the proof key. Their keys differ, so a proof of one can never be returned for
    the other, and the defect is unrepresentable rather than merely unlikely.

    They are also the form the whole Phi-3.5 claim rests on: MatMulNBits is the anchor op, and
    the test suite's own device probe uses it.
    """
    rng = np.random.default_rng(20260801)
    blocks_per_col = (K + block_size - 1) // block_size
    packed_bytes = block_size * bits // 8

    packed = rng.integers(0, 256, size=[N, blocks_per_col, packed_bytes], dtype=np.uint8)
    scale = rng.uniform(0.001, 0.1, size=[N * blocks_per_col]).astype(np.float16)

    inputs = ["X", "B", "scale"]
    inits = [
        onnx.numpy_helper.from_array(packed, name="B"),
        onnx.numpy_helper.from_array(scale, name="scale"),
    ]
    if with_zero_points:
        # One packed nibble-pair per block per column, per the contrib-op spec.
        zp_len = N * ((blocks_per_col + 1) // 2)
        zp = rng.integers(0, 256, size=[zp_len], dtype=np.uint8)
        inputs.append("zero_points")
        inits.append(onnx.numpy_helper.from_array(zp, name="zero_points"))

    node = helper.make_node(
        "MatMulNBits",
        inputs=inputs,
        outputs=["Y"],
        name="mmnb0",
        domain="com.microsoft",
        K=K,
        N=N,
        bits=bits,
        block_size=block_size,
        accuracy_level=1,
    )
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT16, [1, K])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [1, N])
    graph = helper.make_graph([node], "mmnb_graph", [x], [y], initializer=inits)
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
    )
    model.ir_version = 10
    return model


BUILDERS = {
    "add_f32": lambda: _binary("Add", TensorProto.FLOAT),
    "matmulnbits_f16_scales": lambda: _matmulnbits(with_zero_points=False),
    "matmulnbits_f16_scales_zp": lambda: _matmulnbits(with_zero_points=True),
    "add_f16": lambda: _binary("Add", TensorProto.FLOAT16),
    "mul_f32": lambda: _binary("Mul", TensorProto.FLOAT),
    "sub_f32": lambda: _binary("Sub", TensorProto.FLOAT),
    "div_f32": lambda: _binary("Div", TensorProto.FLOAT),
    "relu_f32": lambda: _unary("Relu", TensorProto.FLOAT),
    "sqrt_f32": lambda: _unary("Sqrt", TensorProto.FLOAT),
    # ------------------------------------------------------------------
    # THE PLANTED CONTROL — deliberately never proven.
    #
    # Rai's RAI-008(a) requires that criterion 11 be verified by a planted `[unproven]` decline,
    # not by reading the gate's source. `mul_f16` is a form the EP has a kernel for and the
    # ledger has no entry for, and the generator must never be pointed at it.
    #
    # It is the falsifier for the whole mechanism: if this case is ever claimed, the gate is
    # not gating. Its sibling `mul_f32` IS proven, so the pair also shows the gate discriminating
    # rather than declining everything — a gate that declined unconditionally would pass a
    # one-armed test and be useless.
    # ------------------------------------------------------------------
    "mul_f16_unproven": lambda: _binary("Mul", TensorProto.FLOAT16),
}


# The domain each case must sample from.
#
# An op that is a partial function is not exercised by an input distribution that leaves its
# domain: `Sqrt` fed `standard_normal` returns NaN for about half its elements on every EP, so
# the comparison is NaN-against-NaN over half the tensor and proves nothing there. Naming the
# constraint here keeps it next to the case that needs it, rather than in the runner where the
# next case would have to rediscover it.
#
# `any`      — unconstrained standard normal.
# `positive` — strictly positive (Sqrt, Log).
# `nonzero`  — bounded away from zero (divisors).
INPUT_DOMAIN: dict[str, str] = {
    "sqrt_f32": "positive",
    "div_f32": "nonzero",
}


def input_domain(name: str) -> str:
    """The sampling domain for a case, defaulting to unconstrained."""
    return INPUT_DOMAIN.get(name, "any")


def build(name: str, out_dir: pathlib.Path) -> pathlib.Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.onnx"
    onnx.save(BUILDERS[name](), str(path))
    return path


def main() -> int:
    out_dir = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("bench/results/_ledger_models")
    for name in BUILDERS:
        print(build(name, out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
