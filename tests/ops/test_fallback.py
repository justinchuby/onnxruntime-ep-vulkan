"""Fallback correctness tests for the Vulkan EP.

These tests verify the inverse of the claim assertion: for ops and attribute combinations
that the EP does NOT claim, the session must still:
  1. Produce correct outputs (matching ORT's CPU EP reference).
  2. Not crash.
  3. Not corrupt the graph topology for claimed nodes in a mixed session.

This is the ''clean CPU fallback'' guarantee from DESIGN.md §1.3 and §5.6.

A session that silently runs everything on CPU looks correct but provides no GPU acceleration
— the vacuous-pass problem. These tests assert the OPPOSITE: when the EP should NOT claim a
node, we verify it does not, and that the result is still correct.

New "must not be claimed" cases belong in tests/ops/test_op_table.py as rows with
``claim=False``. This file focuses on structural tests: mixed sessions, CPU isolation,
and permanent fallback ops that need parametrized edge-input coverage.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
from onnx_ir import DataType as DT

import onnxruntime as ort
import _models as m


# ---------------------------------------------------------------------------
# Ops that must never be claimed (permanent CPU fallback — DESIGN.md §1.2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op,feeds,comment", [
    (
        "NonZero",
        {"x": np.array([[0.0, 1.5, 0.0], [2.0, 0.0, -1.0]], dtype=np.float32)},
        "data-dependent output shape — permanent CPU fallback",
    ),
    (
        "Unique",
        {"x": np.array([3.0, 1.0, 1.0, 3.0, 2.0], dtype=np.float32)},
        "data-dependent output shape — permanent CPU fallback",
    ),
], ids=["NonZero", "Unique"])
def test_permanent_cpu_fallback_ops(op, feeds, comment, require_vulkan) -> None:
    """Ops with data-dependent output shapes must never be claimed by the Vulkan EP."""
    input_name = list(feeds.keys())[0]
    inp = feeds[input_name]
    model = m.make_model(
        op,
        [m.tensor(input_name, DT.FLOAT, list(inp.shape))],
        [m.tensor("out", DT.INT64, [-1])],
    )
    # Must not be claimed:
    m.assert_vulkan_does_not_claim(model, feeds)
    # Must still produce correct results on CPU fallback:
    m.assert_matches_cpu(model, feeds, rtol=0, atol=0)


def test_fp64_not_claimed(require_vulkan) -> None:
    """fp64 Add must not be claimed — fp64 is a permanent CPU fallback (DESIGN.md §1.2).

    Most GPUs have no usable double-precision and Vulkan makes shaderFloat64 optional.
    """
    model = m.make_model(
        "Add",
        [m.tensor("a", DT.DOUBLE, [3]), m.tensor("b", DT.DOUBLE, [3])],
        [m.tensor("out", DT.DOUBLE, [3])],
    )
    feeds = {
        "a": np.array([1.0, 2.0, 3.0], dtype=np.float64),
        "b": np.array([4.0, 5.0, 6.0], dtype=np.float64),
    }
    # Must not be claimed:
    m.assert_vulkan_does_not_claim(model, feeds)
    m.assert_matches_cpu(model, feeds, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Mixed session: claimed + unclaimed ops in one graph
# ---------------------------------------------------------------------------


def test_mixed_session_claimed_and_fallback(require_vulkan) -> None:
    """A graph with one claimed op (Add) and one fallback op (NonZero) must:
      - Run Add on VulkanExecutionProvider.
      - Run NonZero on CPUExecutionProvider.
      - Produce correct end-to-end outputs.

    This tests that ORT's partitioning inserts the correct device round-trips and that
    the EP's data-transfer path handles partition boundaries correctly.
    """
    import onnx_ir as ir

    # Build: Add(a, b) → NonZero(sum)
    a = ir.Value(name="a", type=ir.TensorType(DT.FLOAT), shape=ir.Shape([3]))
    b = ir.Value(name="b", type=ir.TensorType(DT.FLOAT), shape=ir.Shape([3]))
    add_out = ir.Value(name="add_out", type=ir.TensorType(DT.FLOAT), shape=ir.Shape([3]))
    nz_out = ir.Value(name="nz_out", type=ir.TensorType(DT.INT64), shape=ir.Shape([1, -1]))

    add_node = ir.node("Add", [a, b], outputs=[add_out])
    nz_node = ir.node("NonZero", [add_out], outputs=[nz_out])

    graph = ir.Graph(
        [a, b], [add_out, nz_out],
        nodes=[add_node, nz_node],
        name="mixed",
        opset_imports={"": 21},
    )
    model_bytes = ir.to_proto(ir.Model(graph, ir_version=10)).SerializeToString()

    feeds = {
        "a": np.array([1.0, 0.0, -1.0], dtype=np.float32),
        "b": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    }

    # Run with Vulkan EP (Add claimed) + CPU EP (NonZero fallback)
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.enable_profiling = True
    opts.profile_file_prefix = "_vulkan_mixed_probe"
    sess = ort.InferenceSession(model_bytes, opts, providers=m.EP_PROVIDERS)
    vulkan_result = sess.run(None, feeds)
    profile_path = sess.end_profiling()

    try:
        with open(profile_path) as fh:
            events = json.load(fh)
    finally:
        try:
            os.remove(profile_path)
        except OSError:
            pass

    node_providers = [
        (e.get("args", {}).get("op_name"), e.get("args", {}).get("provider"))
        for e in events
        if e.get("cat") == "Node" and isinstance(e.get("args"), dict)
    ]

    # Add must have run on VulkanExecutionProvider.
    add_providers = {prov for name, prov in node_providers if name == "Add"}
    assert m.EP_NAME in add_providers, (
        f"Add was not claimed in the mixed session: {node_providers}"
    )

    # NonZero must NOT have run on VulkanExecutionProvider.
    nz_providers = {prov for name, prov in node_providers if name == "NonZero"}
    assert m.EP_NAME not in nz_providers, (
        f"NonZero ran on {m.EP_NAME} — it must fall back to CPU: {node_providers}"
    )

    # End-to-end correctness: result must match CPU-only session.
    cpu_opts = ort.SessionOptions()
    cpu_opts.log_severity_level = 3
    cpu_sess = ort.InferenceSession(model_bytes, cpu_opts, providers=["CPUExecutionProvider"])
    cpu_result = cpu_sess.run(None, feeds)

    np.testing.assert_allclose(vulkan_result[0], cpu_result[0], rtol=0, atol=0,
                               err_msg="Add output: mixed session vs CPU-only mismatch")
    np.testing.assert_array_equal(vulkan_result[1], cpu_result[1],
                                  err_msg="NonZero output: mixed session vs CPU-only mismatch")


# ---------------------------------------------------------------------------
# CPU-only session (no EP registered / providers list excludes VulkanEP)
# ---------------------------------------------------------------------------


def test_cpu_only_session_still_works() -> None:
    """A session created with only CPUExecutionProvider must work regardless of EP registration.

    This tests that registering the Vulkan EP does not interfere with CPU-only sessions —
    no side effects on other inference sessions.
    """
    model = m.make_model(
        "Add",
        [m.tensor("a", DT.FLOAT, [3]), m.tensor("b", DT.FLOAT, [3])],
        [m.tensor("out", DT.FLOAT, [3])],
    )
    feeds = {
        "a": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "b": np.array([4.0, 5.0, 6.0], dtype=np.float32),
    }
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(model, opts, providers=["CPUExecutionProvider"])
    result = sess.run(None, feeds)
    np.testing.assert_array_equal(result[0], np.array([5.0, 7.0, 9.0], dtype=np.float32))
