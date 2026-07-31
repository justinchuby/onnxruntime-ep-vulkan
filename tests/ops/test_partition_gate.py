"""Proof that `retain_viable` / the net-benefit predicate fires at runtime.

R10 (Morpheus 2026-07-30): a mechanism that exists in the source tree and not in the call
graph is indistinguishable from one that was never written. The falsifier for "retain_viable
is wired" is an observation of an artifact it produced whose content varies with its input —
not a code reading, not a flag its author set.

This test provides that artifact. The partition economics gate applies to MULTI-CLUSTER graphs
(models where claimed nodes form two or more disjoint components). In the multi-cluster path,
partition::evaluate is called for each component; a non-anchor component below the
flops-per-transfer margin produces a [partition] decline code in the CLAIM_LOG.

Model design: two independent Sigmoid branches share no edges. ORT sees them as a graph with
two disconnected claimed clusters, each a 1-node island with no anchors (Sigmoid is not in
partition::is_anchor). partition::evaluate fires → TooSmall (1 < min_nodes=4, anchors=0) →
[partition] decline code. CLAIM_LOG records the decline.

Two guard directions (§7.0.2, as formalised by Morpheus):
  Over-declination: the anchor exemption in partition::evaluate ensures any island containing
    MatMulNBits or GQA always passes (anchors > 0 → Claim). Phi-3.5 bench checks this
    (353 claimed, 1 island after GQA, MATCH). Falsifier: bench/phi35.py → 0 claimed nodes.
  Under-declination: THIS TEST. A non-anchor two-cluster model must produce [partition]
    declines. Falsifier: claims["Sigmoid"]["code"] != "partition" when CLAIM_LOG is set,
    i.e. the EP claims both Sigmoid nodes instead of declining them.

Run::

    pytest tests/ops/test_partition_gate.py -v --no-header
"""

from __future__ import annotations

import os
import pathlib

import numpy as np
import onnx
import onnx.helper as oh
import onnxruntime as ort

import _models as m


def _two_branch_sigmoid_model() -> bytes:
    """ONNX model with two independent Sigmoid branches — two disjoint claimed clusters.

    x1 (fp16, [4]) → Sigmoid → y1 (fp16, [4])
    x2 (fp16, [4]) → Sigmoid → y2 (fp16, [4])

    x1, x2 are independent inputs. y1, y2 are independent outputs. The two Sigmoid nodes
    share no edges, so they form two disconnected claimed clusters. The partition gate
    evaluates each cluster independently and produces [partition] declines for both
    (TooSmall: 1 node, no anchors, 1 < min_nodes=4).
    """
    x1 = oh.make_tensor_value_info("x1", onnx.TensorProto.FLOAT16, [4])
    x2 = oh.make_tensor_value_info("x2", onnx.TensorProto.FLOAT16, [4])
    y1 = oh.make_tensor_value_info("y1", onnx.TensorProto.FLOAT16, [4])
    y2 = oh.make_tensor_value_info("y2", onnx.TensorProto.FLOAT16, [4])
    n1 = oh.make_node("Sigmoid", inputs=["x1"], outputs=["y1"], name="sig1")
    n2 = oh.make_node("Sigmoid", inputs=["x2"], outputs=["y2"], name="sig2")
    graph = oh.make_graph([n1, n2], "two_branch_sigmoid", [x1, x2], [y1, y2])
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 15)])
    model.ir_version = 8
    return model.SerializeToString()


def _two_branch_mul_model() -> bytes:
    """ONNX model with two independent three-node Mul chains — two 3-node clusters, no anchors.

    Each chain: x → Mul(x,x) → t1 → Mul(t1,t1) → t2 → Mul(t2,t2) → out
    Two independent chains share no edges → two disconnected clusters of 3 nodes each.
    TooSmall: 3 < min_nodes=4, anchors=0.
    """
    x1 = oh.make_tensor_value_info("xa", onnx.TensorProto.FLOAT16, [4])
    x2 = oh.make_tensor_value_info("xb", onnx.TensorProto.FLOAT16, [4])
    ya = oh.make_tensor_value_info("ya", onnx.TensorProto.FLOAT16, [4])
    yb = oh.make_tensor_value_info("yb", onnx.TensorProto.FLOAT16, [4])
    # Chain A
    na1 = oh.make_node("Mul", inputs=["xa", "xa"], outputs=["ta1"], name="ma1")
    na2 = oh.make_node("Mul", inputs=["ta1", "ta1"], outputs=["ta2"], name="ma2")
    na3 = oh.make_node("Mul", inputs=["ta2", "ta2"], outputs=["ya"], name="ma3")
    # Chain B
    nb1 = oh.make_node("Mul", inputs=["xb", "xb"], outputs=["tb1"], name="mb1")
    nb2 = oh.make_node("Mul", inputs=["tb1", "tb1"], outputs=["tb2"], name="mb2")
    nb3 = oh.make_node("Mul", inputs=["tb2", "tb2"], outputs=["yb"], name="mb3")
    graph = oh.make_graph([na1, na2, na3, nb1, nb2, nb3], "two_mul_chains", [x1, x2], [ya, yb])
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 15)])
    model.ir_version = 8
    return model.SerializeToString()


def test_two_branch_sigmoid_produces_partition_decline(
    require_vulkan, tmp_path: pathlib.Path
) -> None:
    """Two independent Sigmoid clusters must produce [partition] declines in CLAIM_LOG.

    Artifact: code == "partition" in the CLAIM_LOG record for Sigmoid.
    Falsifier: if this assertion fails, the partition economics gate is not in the call graph
    for multi-cluster models (R10 uninvoked-mechanism defect).

    The model has two disconnected claimed clusters (each a 1-node Sigmoid island, no anchors).
    partition::evaluate fires for each (only_one_cluster == False); TooSmall fires for both
    (1 < min_nodes=4, anchors=0). Both clusters produce [partition] declines and fall back to CPU.
    """
    log_path = tmp_path / "claim_partition_sigmoid.jsonl"
    model = _two_branch_sigmoid_model()
    feeds = {
        "x1": np.ones((4,), dtype=np.float16),
        "x2": np.full((4,), 2.0, dtype=np.float16),
    }

    opts = m._make_session_options()
    os.environ["ONNXRUNTIME_EP_VULKAN_CLAIM_LOG"] = str(log_path)
    try:
        session = ort.InferenceSession(model, opts, providers=m.EP_PROVIDERS)
        results = session.run(None, feeds)
    finally:
        del os.environ["ONNXRUNTIME_EP_VULKAN_CLAIM_LOG"]

    # Verify output correctness (CPU fallback must produce the right answers).
    for inp, out in zip([feeds["x1"], feeds["x2"]], results):
        expected = (1.0 / (1.0 + np.exp(-inp.astype(np.float32)))).astype(np.float16)
        np.testing.assert_allclose(out.astype(np.float32), expected.astype(np.float32),
                                   rtol=1e-3, atol=1e-3)

    # --- THE ARTIFACT ---
    # CLAIM_LOG must contain a [partition] record for Sigmoid.
    # Sequence: registry writes "claimed=true" for each Sigmoid; partition gate writes
    # "claimed=false, code=partition" for each (TooSmall). read_claim_log returns the last
    # record per op — so Sigmoid's entry has code="partition".
    claims = m.read_claim_log(log_path)
    assert "Sigmoid" in claims, (
        "No CLAIM_LOG record for Sigmoid. "
        "Either CLAIM_LOG is not written or the EP did not process Sigmoid at all."
    )
    rec = claims["Sigmoid"]
    assert rec["code"] == "partition", (
        f"Expected code='partition' for 1-node Sigmoid island (TooSmall); "
        f"got code={rec['code']!r}. "
        "If retain_viable is wired for multi-cluster graphs, TooSmall (1 < min_nodes=4, "
        "anchors=0) must fire. "
        f"Full record: {rec}"
    )
    assert rec["claimed"] is False, (
        f"Sigmoid must not be claimed after partition decline. Full record: {rec}"
    )


def test_two_branch_mul_produces_partition_decline(
    require_vulkan, tmp_path: pathlib.Path
) -> None:
    """Two independent 3-node Mul chains must produce [partition] declines.

    nodes=3 < min_nodes=4 and anchors=0 → TooSmall → [partition] for both clusters.
    Exercises the multi-node path through the island builder (distinct from 1-node test).
    """
    log_path = tmp_path / "claim_partition_mul.jsonl"
    model = _two_branch_mul_model()
    feeds = {
        "xa": np.full((4,), 2.0, dtype=np.float16),
        "xb": np.full((4,), 3.0, dtype=np.float16),
    }

    opts = m._make_session_options()
    os.environ["ONNXRUNTIME_EP_VULKAN_CLAIM_LOG"] = str(log_path)
    try:
        session = ort.InferenceSession(model, opts, providers=m.EP_PROVIDERS)
        _ = session.run(None, feeds)
    finally:
        del os.environ["ONNXRUNTIME_EP_VULKAN_CLAIM_LOG"]

    claims = m.read_claim_log(log_path)
    assert "Mul" in claims, "No CLAIM_LOG record for Mul."
    rec = claims["Mul"]
    assert rec["code"] == "partition", (
        f"Expected code='partition' for 3-node Mul chain (TooSmall); got {rec['code']!r}. "
        "TooSmall (3 < min_nodes=4, anchors=0) should fire for multi-cluster graphs."
    )
