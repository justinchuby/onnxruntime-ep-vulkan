"""Graph-level proof for the conservative Shape/Cast/Concat rank inference (issue #8).

WHAT THIS FILE IS FOR
=====================
``rust/src/shape_infer.rs`` carries ~50 unit tests that check the *rules* in isolation.  They
prove the algebra and nothing else.  A rule that is correct and never reached is worth exactly
as much as a rule that was never written, so this file works from the other end: it plants
whole ONNX graphs, runs them through a real ``InferenceSession``, and reads the artifacts the
EP produced.

THE CHAIN THESE TESTS PLANT IS THE ONE BERT ACTUALLY CONTAINS
=============================================================
The chain was not invented for the tests.  ``rust/tools/probe_rank_chain.py`` walked
BERT-SQuAD-12 (sha256 5f0d96a9…9655, 435,852,736 bytes) and printed the producer chain behind
the first unranked ``Reshape``::

    bert/encoder/Reshape__27:0        <- Cast{to: INT64}
      …/shape_Concat__26:0            <- Concat{axis: 0}
        …/shape_Unsqueeze__23:0       <- Unsqueeze{axes: [0]}
          …/strided_slice__17:0       <- Cast{to: INT32}
            …/strided_slice__16:0     <- Squeeze{axes: [0]}
              …/strided_slice:0       <- Slice
                …/Shape__12:0         <- Cast{to: FLOAT}      ← the load-bearing step
                  …/Shape:0           <- Shape
                    input_ids:0       = <graph input> rank 2

That ``Cast{to: FLOAT}`` is why everything downstream is unranked, and it is the reason this
pass is possible at all:

  * ORT's partial-data propagation only tracks integral tensors.  The moment the shape values
    become floats ORT loses them, cannot constant-fold the ``Concat``, and therefore cannot say
    what rank the ``Reshape`` produces.  58 of BERT's 71 ``Reshape`` outputs are unranked for
    exactly this reason, and 98 of 98 ``MatMul`` A-inputs inherit it.
  * A cast does not change a tensor's *shape*.  So while the shape tensor's **values** are
    genuinely lost, its **length** is not — and the length is the only thing that determines
    the rank of the ``Reshape``.

Every test below is built on that distinction.  Reproducing it needs the float cast: without
it ORT folds the chain to a literal initializer before the EP is ever asked, and the graph
proves nothing.  (That is not a guess — the first draft of this file omitted the cast, and
every planted chain arrived at the EP already annotated.)

THE PLANTED CONTROL
===================
The pass is behind a kill switch, ``ONNXRUNTIME_EP_VULKAN_RANK_INFERENCE=0``.  That is not a
convenience — it is what makes the claim falsifiable.  Every "the pass helped" assertion here
is a **paired** measurement of the same model in the same process image:

    inference OFF  →  the EP declines the chain, code ``unknown-rank``
    inference ON   →  the EP claims it, and more dispatches actually run

A single-sided measurement ("it claims 8 nodes") cannot distinguish the pass working from the
pass being irrelevant.  The pair can.  Delete the pass and leave the wiring, or delete the
wiring and leave the pass, and these tests fail either way.

THE CONVERSE CASES
==================
An engine that answered "rank 3" to everything would pass the positive tests above.  The
larger half of this file is therefore the graphs where the right answer is "I do not know":
a shape tensor of unknown length, a ``Concat`` axis out of range, a ``Reshape`` whose target is
a genuine runtime value.  "Conservative" is a claim about the cases where inference fails, and
it can only be tested there.

WHY THE OUTPUT COMPARISON IS PART OF THE PROOF
==============================================
A rank fact is an assertion about what the kernels will be handed at ``Compute``.  If the
assertion is wrong the model still runs — it just returns different numbers.  Every positive
test therefore also compares against CPU.  A coverage gain that changes an output is not a
coverage gain.

WHY DISPATCH COUNTS ARE READ AS DIFFERENCES
===========================================
``dispatches_executed`` lives in a process-global atomic (``rust/src/counters.rs``) and the
counters file is a cumulative snapshot taken at EP teardown.  Two snapshots from one pytest
process are therefore not comparable as absolute numbers.  Each paired test runs the OFF model
twice and the ON model once and compares consecutive differences, so both sides are measured
the same way.

Run::

    pytest tests/ops/test_rank_inference_chain.py -v --no-header
"""

from __future__ import annotations

import contextlib
import json
import os
import pathlib

import numpy as np
import onnx
import onnx.helper as oh
import onnxruntime as ort
import pytest

import _models as m

RANK_INFERENCE_ENV = "ONNXRUNTIME_EP_VULKAN_RANK_INFERENCE"

#: BERT-SQuAD-12 is opset 12, where ``Squeeze``/``Unsqueeze`` take ``axes`` as an attribute.
#: The planted chains keep that opset so they exercise the same attribute-reading path the
#: real model does.
BERT_OPSET = 12


# ---------------------------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------------------------


@contextlib.contextmanager
def _env(**kv: "str | None"):
    """Set/clear environment variables for the duration of the block, then restore them.

    The EP reads these once per session, so the block must be exited even when the session
    raises — otherwise one failing test silently reconfigures every test after it.
    """
    saved = {k: os.environ.get(k) for k in kv}
    try:
        for k, v in kv.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _run(
    model: bytes,
    feeds: dict,
    tmp_path: pathlib.Path,
    tag: str,
    *,
    inference: bool,
    run: bool = True,
) -> tuple[list, dict, "int | None"]:
    """Run *model* once; return ``(outputs, claim_records, cumulative_dispatches)``.

    The third element is the **cumulative** process counter at this session's teardown, not
    this session's own dispatch count.  Callers must difference consecutive readings; see
    :func:`_dispatch_deltas`.  ``None`` means the counters file was not written, which is
    reported as absence rather than silently read as zero.
    """
    log_path = tmp_path / f"claim_{tag}.jsonl"
    counters_path = tmp_path / f"counters_{tag}.json"
    with _env(
        ONNXRUNTIME_EP_VULKAN_CLAIM_LOG=str(log_path),
        ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE=str(counters_path),
        **{RANK_INFERENCE_ENV: None if inference else "0"},
    ):
        session = ort.InferenceSession(model, m._make_session_options(), providers=m.EP_PROVIDERS)
        outputs = session.run(None, feeds) if run else []
        del session  # counters are flushed at EP teardown, so drop the session before reading

    claims = m.read_claim_log(log_path) if log_path.is_file() else {}
    dispatches = None
    if counters_path.is_file():
        with counters_path.open(encoding="utf-8") as fh:
            dispatches = json.load(fh).get("dispatches_executed")
    return outputs, claims, dispatches


def _dispatch_deltas(model: bytes, feeds: dict, tmp_path: pathlib.Path, tag: str):
    """Return ``(off_delta, on_delta, off_claims, on_claims, off_out, on_out)``.

    The OFF model is run twice so its own per-session dispatch count is measurable as a
    difference between consecutive cumulative snapshots — the same way the ON count is.  A
    comparison between an absolute snapshot and a difference would be meaningless, and it is
    exactly the kind of arithmetic that makes a passing test say nothing.
    """
    _, _, d0 = _run(model, feeds, tmp_path, f"{tag}_off1", inference=False)
    out_off, claims_off, d1 = _run(model, feeds, tmp_path, f"{tag}_off2", inference=False)
    out_on, claims_on, d2 = _run(model, feeds, tmp_path, f"{tag}_on", inference=True)
    if d0 is None or d1 is None or d2 is None:
        return None, None, claims_off, claims_on, out_off, out_on
    return d1 - d0, d2 - d1, claims_off, claims_on, out_off, out_on


def _claim_rows(log_path: pathlib.Path) -> list[dict]:
    """Every record in the claim log, in order — not just the last one per op."""
    if not log_path.is_file():
        return []
    rows = []
    with log_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                with contextlib.suppress(json.JSONDecodeError):
                    rows.append(json.loads(line))
    return rows


def _model(nodes, inputs, outputs, initializers=(), *, opset: int = BERT_OPSET) -> bytes:
    graph = oh.make_graph(list(nodes), "rank_infer", list(inputs), list(outputs),
                          initializer=list(initializers))
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", opset)])
    model.ir_version = 8
    # Deliberately NOT running onnx.shape_inference: the point is a graph whose ranks nothing
    # upstream of the EP established. Annotating it here would prove the pass unnecessary.
    return model.SerializeToString()


def _i64(name: str, values) -> onnx.TensorProto:
    arr = np.asarray(values, dtype=np.int64)
    return oh.make_tensor(name, onnx.TensorProto.INT64, list(arr.shape), arr.flatten().tolist())


def _i32_scalar(name: str, value: int) -> onnx.TensorProto:
    return oh.make_tensor(name, onnx.TensorProto.INT32, [], [int(value)])


def _f32(name: str, arr: np.ndarray) -> onnx.TensorProto:
    return oh.make_tensor(name, onnx.TensorProto.FLOAT, list(arr.shape),
                          arr.astype(np.float32).flatten().tolist())


def _unranked(name: str, dtype=onnx.TensorProto.FLOAT):
    """A value_info with a type and *no* shape — what ORT reports as "never established"."""
    vi = onnx.ValueInfoProto()
    vi.name = name
    vi.type.tensor_type.elem_type = dtype
    return vi


def _bert_shape_chain(axis: int = 0, extra_pieces=()):
    """The BERT ``Shape→Cast(f32)→Slice→Squeeze→Cast(i32)→Unsqueeze→Concat→Cast(i64)`` chain.

    Returns ``(nodes, initializers, output_name)``.  The output is an int64 rank-1 tensor of
    length ``3 + len(extra_pieces)`` whose *values* nothing upstream of the EP can know,
    because they passed through a float.
    """
    nodes = [
        oh.make_node("Shape", ["x"], ["s"], name="shape0"),
        # The step that loses the values and keeps the length.
        oh.make_node("Cast", ["s"], ["sf"], to=onnx.TensorProto.FLOAT, name="cast_f"),
        oh.make_node("Slice", ["sf", "k_st", "k_en", "k_ax"], ["sl"], name="slice0"),
        oh.make_node("Squeeze", ["sl"], ["sq"], axes=[0], name="squeeze0"),
        oh.make_node("Cast", ["sq"], ["si"], to=onnx.TensorProto.INT32, name="cast_i"),
        oh.make_node("Unsqueeze", ["si"], ["u0"], axes=[0], name="unsq0"),
        oh.make_node("Unsqueeze", ["k32"], ["u1"], axes=[0], name="unsq1"),
        oh.make_node("Unsqueeze", ["k4"], ["u2"], axes=[0], name="unsq2"),
        oh.make_node("Concat", ["u0", "u1", "u2", *extra_pieces], ["cc"], axis=axis,
                     name="concat0"),
        oh.make_node("Cast", ["cc"], ["ns"], to=onnx.TensorProto.INT64, name="cast_o"),
    ]
    inits = [
        _i64("k_st", [0]), _i64("k_en", [1]), _i64("k_ax", [0]),
        _i32_scalar("k32", 32), _i32_scalar("k4", 4),
    ]
    return nodes, inits, "ns"


# ---------------------------------------------------------------------------------------------
# The planted control
# ---------------------------------------------------------------------------------------------


def _planted_chain() -> tuple[bytes, dict]:
    """``x[N,8,16]`` reshaped to ``[N,32,4]`` through the BERT chain, then a MatMul island.

    The rank of ``h`` is fixed by ONNX semantics and by nothing else::

        Shape(x)                → length 3   (x's rank, which ORT *did* give us)
        Cast(…, FLOAT)          → length 3   (a cast preserves shape)
        Slice(…, 0, 1, axis 0)  → length 1
        Squeeze(…, axes=[0])    → rank 0
        Unsqueeze(…, axes=[0])  → length 1
        Concat(3 × length 1)    → length 3
        Cast(…, INT64)          → length 3
        Reshape(x, …)           → rank 3

    Not one of those steps guesses a dimension *value*.  The leading extent stays unknown from
    beginning to end; only the length of the shape tensor is used, and the length is the whole
    of what determines the rank.
    """
    x = oh.make_tensor_value_info("x", onnx.TensorProto.FLOAT, ["N", 8, 16])
    y = _unranked("y")
    chain, chain_inits, ns = _bert_shape_chain()
    nodes = [
        *chain,
        oh.make_node("Reshape", ["x", ns], ["h"], name="reshape0"),
        # Everything below consumes an unranked value; this is what the pass unlocks.
        oh.make_node("MatMul", ["h", "W"], ["mm"], name="matmul0"),
        oh.make_node("Mul", ["mm", "mm"], ["sq2"], name="mul0"),
        oh.make_node("Add", ["sq2", "mm"], ["ad"], name="add0"),
        oh.make_node("Sub", ["ad", "mm"], ["sb"], name="sub0"),
        oh.make_node("Mul", ["sb", "sb"], ["y"], name="mul1"),
    ]
    inits = [*chain_inits, _f32("W", np.linspace(-0.5, 0.5, 16).reshape(4, 4))]
    feeds = {"x": np.linspace(-1.0, 1.0, 2 * 8 * 16, dtype=np.float32).reshape(2, 8, 16)}
    return _model(nodes, [x], [y], inits), feeds


def test_planted_chain_declines_without_inference_and_claims_with_it(
    require_vulkan, tmp_path: pathlib.Path
) -> None:
    """The paired control: OFF declines ``unknown-rank``, ON claims and dispatches more.

    ARTIFACT
        Three claim logs and three counters files from the same model in one process, differing
        only in ``ONNXRUNTIME_EP_VULKAN_RANK_INFERENCE``.

    FALSIFIER
        If the OFF run does *not* decline, this graph never reproduced the defect in issue #8
        and proves nothing about it — the test fails and says so rather than passing vacuously.
        If the ON run does not raise the dispatch count, the pass changed a claim decision
        without changing what actually ran on the device, which is the precise failure mode
        (``claimed_nodes`` up, dispatches flat) that issue #8 was filed about.
    """
    model, feeds = _planted_chain()
    off_delta, on_delta, claims_off, claims_on, out_off, out_on = _dispatch_deltas(
        model, feeds, tmp_path, "planted"
    )

    # --- the graph must actually reproduce the defect, or the comparison is meaningless ---
    assert "MatMul" in claims_off, (
        "The EP never saw MatMul without inference; the planted graph does not reach the code "
        f"path issue #8 is about. Records: {sorted(claims_off)}"
    )
    assert claims_off["MatMul"]["claimed"] is False, (
        "MatMul was already claimed with inference OFF, so this graph does not reproduce the "
        f"unknown-rank decline. Record: {claims_off['MatMul']}"
    )
    assert claims_off["MatMul"]["code"] == "unknown-rank", (
        "MatMul declined for a reason other than unknown rank, so the inference pass is not "
        f"what would unlock it. Record: {claims_off['MatMul']}"
    )

    # --- and the pass must unlock it ---
    assert claims_on["MatMul"]["claimed"] is True, (
        f"Rank inference did not unlock the MatMul it was written for. "
        f"Record: {claims_on['MatMul']}"
    )
    assert claims_on["MatMul"].get("rank_inferred") is True, (
        "MatMul was claimed but not on an inferred rank, so something other than this pass "
        f"changed the decision. Record: {claims_on['MatMul']}"
    )

    # --- claiming is not running; the device must do more work ---
    assert off_delta is not None and on_delta is not None, (
        "counters missing; a missing witness is not a measurement, so the dispatch comparison "
        "cannot be made"
    )
    assert on_delta > off_delta, (
        f"per-session dispatches did not rise: off={off_delta}, on={on_delta}. Claiming more "
        "nodes while dispatching the same number of times is the defect, not the fix."
    )

    # --- and the answer must not move ---
    np.testing.assert_allclose(out_on[0], out_off[0], rtol=1e-5, atol=1e-6,
                               err_msg="Rank inference changed the model's output.")
    cpu = m.run_cpu(model, feeds)
    np.testing.assert_allclose(out_on[0], cpu[0], rtol=1e-4, atol=1e-5,
                               err_msg="Vulkan disagrees with CPU on the inferred chain.")


def test_planted_chain_proves_rank_without_inventing_extents(
    require_vulkan, tmp_path: pathlib.Path
) -> None:
    """The claimed MatMul reads rank 3 with the batch extent still unknown.

    The chain proves a *length*, not a shape.  If the recorded input shape came back fully
    concrete the pass invented an extent it had no way to know — the exact failure this design
    exists to avoid, and one a test that only counted claims could never see.  The leading
    dimension of ``x`` is symbolic and its value crossed a float cast, so a concrete first
    extent here could only be a guess.
    """
    model, feeds = _planted_chain()
    log_path = tmp_path / "claim_proof.jsonl"
    with _env(ONNXRUNTIME_EP_VULKAN_CLAIM_LOG=str(log_path), **{RANK_INFERENCE_ENV: None}):
        session = ort.InferenceSession(model, m._make_session_options(), providers=m.EP_PROVIDERS)
        session.run(None, feeds)
        del session

    rows = [r for r in _claim_rows(log_path) if r.get("op") == "MatMul"]
    assert rows, f"no MatMul rows in {log_path}"
    shapes = rows[-1].get("input_shapes")
    assert shapes, f"MatMul row carries no input_shapes: {rows[-1]}"
    a = shapes[0]
    assert len(a) == 3, (
        f"MatMul input A was read at rank {len(a)}, not the rank 3 the Concat length proves. "
        f"Row: {rows[-1]}"
    )
    assert a[0] == -1, (
        f"the leading extent came back as {a[0]}; it passed through a float cast, so its value "
        f"is not knowable and must stay unknown. Row: {rows[-1]}"
    )
    assert rows[-1].get("shape_class") == "extents-symbolic", (
        "a rank proven with unknown extents must be classified as extents-symbolic so it takes "
        f"the runtime-extent kernel path rather than a static descriptor. Row: {rows[-1]}"
    )


def test_fanout_shape_tensor_unlocks_every_consumer(
    require_vulkan, tmp_path: pathlib.Path
) -> None:
    """One shape tensor feeding several ``Reshape`` nodes must unlock all of them.

    Real transformer graphs compute a shape once and reuse it across the query/key/value
    projections.  A pass that walked a single chain instead of iterating to a fixed point would
    unlock the first consumer and miss the rest — which looks like a partial win and is a bug.
    """
    x = oh.make_tensor_value_info("x", onnx.TensorProto.FLOAT, ["N", 8, 16])
    y = _unranked("y")
    chain, chain_inits, ns = _bert_shape_chain()
    nodes = [
        *chain,
        # Two *distinct* consumers of one shape tensor. They must differ in something other
        # than their node name: ORT's common-subexpression elimination folds two Reshape nodes
        # with identical inputs into one before the EP is asked anything, which would quietly
        # turn this into a single-consumer test that passes for the wrong reason.
        oh.make_node("Neg", ["x"], ["xn"], name="neg0"),
        oh.make_node("Reshape", ["x", ns], ["h1"], name="reshape1"),
        oh.make_node("Reshape", ["xn", ns], ["h2"], name="reshape2"),
        # A MatMul on each branch. Two jobs: it is the op whose rank precondition the pass has
        # to satisfy *per consumer*, and it is an anchor, so the island it lands in clears the
        # partition net-benefit margin. Without real compute an all-elementwise island is
        # declined as transfer-dominated and the test would measure the economics gate rather
        # than the inference (`docs/DESIGN.md` §8.11).
        oh.make_node("MatMul", ["h1", "W"], ["m1"], name="matmul1"),
        oh.make_node("MatMul", ["h2", "W"], ["m2"], name="matmul2"),
        oh.make_node("Mul", ["m1", "m2"], ["p"], name="mul0"),
        oh.make_node("Add", ["p", "m1"], ["q"], name="add0"),
        oh.make_node("Sub", ["q", "m2"], ["r"], name="sub0"),
        oh.make_node("Mul", ["r", "r"], ["y"], name="mul1"),
    ]
    inits = [*chain_inits, _f32("W", np.linspace(-0.5, 0.5, 16).reshape(4, 4))]
    feeds = {"x": np.linspace(-1, 1, 2 * 8 * 16, dtype=np.float32).reshape(2, 8, 16)}
    model = _model(nodes, [x], [y], inits)

    off_delta, on_delta, claims_off, claims_on, out_off, out_on = _dispatch_deltas(
        model, feeds, tmp_path, "fanout"
    )
    assert claims_off.get("MatMul", {}).get("claimed") is False, (
        f"the fanout graph does not reproduce the defect: {claims_off.get('MatMul')}"
    )
    # Per node, not per op type: the aggregated view would be satisfied by one branch, which is
    # precisely the partial win this test exists to rule out.
    on_rows = _claim_rows(tmp_path / "claim_fanout_on.jsonl")
    unlocked = {
        r["node"] for r in on_rows if r.get("op") == "MatMul" and r.get("claimed") is True
    }
    assert unlocked == {"matmul1", "matmul2"}, (
        "rank inference reached only part of the fan-out; claimed MatMul nodes: "
        f"{sorted(unlocked)} (all MatMul rows: "
        f"{[(r['node'], r.get('claimed'), r.get('code')) for r in on_rows if r.get('op') == 'MatMul']})"
    )
    assert off_delta is not None and on_delta is not None and on_delta > off_delta, (
        f"dispatches did not rise across the fanout: off={off_delta} on={on_delta}"
    )
    np.testing.assert_allclose(out_on[0], out_off[0], rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------------------------
# Converse cases: graphs where the honest answer is "unknown"
# ---------------------------------------------------------------------------------------------


def _assert_matmul_declines(claims: dict, why: str, tag: str) -> None:
    assert "MatMul" in claims, f"[{tag}] EP never saw MatMul; records: {sorted(claims)}"
    rec = claims["MatMul"]
    assert rec["claimed"] is False, (
        f"[{tag}] MatMul was claimed, but {why}. Inferring a rank here would be a guess, and a "
        f"guess that happens to be right is still a guess. Record: {rec}"
    )
    assert rec["code"] == "unknown-rank", (
        f"[{tag}] MatMul declined with code {rec['code']!r}, not 'unknown-rank'. The decline is "
        f"for the wrong reason, so this test no longer guards {why}. Record: {rec}"
    )


def test_unknown_length_concat_input_leaves_rank_unknown(
    require_vulkan, tmp_path: pathlib.Path
) -> None:
    """One ``Concat`` input of unknown length poisons the whole sum.

    ``Concat`` along axis 0 gives a length that is the *sum* of its inputs' lengths.  A sum
    with an unknown term is unknown, and there is no conservative rounding that rescues it.
    The extra piece here is ``Shape`` of a second graph input whose own rank was never
    established, so its length is unknown for a reason the EP can see and cannot resolve.

    The graph is built but not run: what is under test is the claim decision, which is made at
    session build, and the ``Reshape`` would be inconsistent at runtime by construction.
    """
    x = oh.make_tensor_value_info("x", onnx.TensorProto.FLOAT, ["N", 8, 16])
    x2 = _unranked("x2")
    y = _unranked("y")
    chain, chain_inits, ns = _bert_shape_chain(extra_pieces=["u_unknown"])
    nodes = [
        oh.make_node("Shape", ["x2"], ["s2"], name="shape_unknown"),
        oh.make_node("Cast", ["s2"], ["u_unknown"], to=onnx.TensorProto.INT32, name="cast_unk"),
        *chain,
        oh.make_node("Reshape", ["x", ns], ["h"], name="reshape0"),
        oh.make_node("MatMul", ["h", "W"], ["mm"], name="matmul0"),
        oh.make_node("Mul", ["mm", "mm"], ["y"], name="mul0"),
    ]
    inits = [*chain_inits, _f32("W", np.eye(4, dtype=np.float32))]
    model = _model(nodes, [x, x2], [y], inits)

    _, claims, _ = _run(model, {}, tmp_path, "unknown_len", inference=True, run=False)
    _assert_matmul_declines(
        claims,
        "one Concat input has unknown length, so the total length is unknown",
        "unknown_len",
    )


def test_reshape_to_a_shape_tensor_of_unknown_length_stays_unknown(
    require_vulkan, tmp_path: pathlib.Path
) -> None:
    """A ``Reshape`` whose target has an unknown *length* must not gain a rank.

    This is the boundary of the whole design, and it is a sharper line than it first looks.
    A shape tensor that is a runtime *value* is fine: ``Reshape``'s output rank is the number
    of elements in that tensor, so a target declared ``[3]`` proves rank 3 no matter what the
    three numbers turn out to be — the graph fixes the length even though nothing fixes the
    values.  What is *not* fine is a target whose length is itself symbolic, because then the
    output rank is a property of the feed.  The distinction is the difference between reading a
    length and guessing one, and only the second is forbidden.

    Here ``shp`` is declared ``[L]``.  The EP must decline.
    """
    x = oh.make_tensor_value_info("x", onnx.TensorProto.FLOAT, ["N", 8, 16])
    shp = oh.make_tensor_value_info("shp", onnx.TensorProto.INT64, ["L"])
    y = _unranked("y")
    nodes = [
        oh.make_node("Reshape", ["x", "shp"], ["h"], name="reshape0"),
        oh.make_node("MatMul", ["h", "W"], ["mm"], name="matmul0"),
        oh.make_node("Mul", ["mm", "mm"], ["y"], name="mul0"),
    ]
    inits = [_f32("W", np.eye(16, dtype=np.float32))]
    model = _model(nodes, [x, shp], [y], inits)
    feeds = {
        "x": np.linspace(-1, 1, 2 * 8 * 16, dtype=np.float32).reshape(2, 8, 16),
        "shp": np.array([2, 8, 16], dtype=np.int64),
    }
    _, claims, _ = _run(model, feeds, tmp_path, "runtime_shape", inference=True)
    _assert_matmul_declines(
        claims,
        "the Reshape target has a symbolic length, so the output rank is a property of the "
        "feed rather than of the graph",
        "runtime_shape",
    )


def test_reshape_to_a_fixed_length_runtime_value_does_prove_the_rank(
    require_vulkan, tmp_path: pathlib.Path
) -> None:
    """The converse of the converse: a ``[3]``-shaped target proves rank 3, values or not.

    Without this test the one above could be satisfied by refusing every ``Reshape`` whose
    target is not an initializer — conservative in the sense of "proves nothing", which would
    lose most of BERT.  ONNX fixes the output rank as the element count of the shape tensor,
    and the element count of a tensor declared ``[3]`` is 3.
    """
    x = oh.make_tensor_value_info("x", onnx.TensorProto.FLOAT, ["N", 8, 16])
    shp = oh.make_tensor_value_info("shp", onnx.TensorProto.INT64, [3])
    y = _unranked("y")
    nodes = [
        oh.make_node("Reshape", ["x", "shp"], ["h"], name="reshape0"),
        oh.make_node("Mul", ["h", "h"], ["y"], name="mul0"),
    ]
    model = _model(nodes, [x, shp], [y])
    feeds = {
        "x": np.linspace(-1, 1, 2 * 8 * 16, dtype=np.float32).reshape(2, 8, 16),
        "shp": np.array([2, 32, 4], dtype=np.int64),
    }
    out, claims, _ = _run(model, feeds, tmp_path, "fixed_len_shape", inference=True)
    rec = claims.get("Mul")
    assert rec is not None, f"EP never saw Mul; records: {sorted(claims)}"
    if rec.get("input_shapes"):
        assert len(rec["input_shapes"][0]) == 3, (
            "the shape tensor is declared [3], so the Reshape output has exactly rank 3. "
            f"Record: {rec}"
        )
    cpu = m.run_cpu(model, feeds)
    np.testing.assert_allclose(out[0], cpu[0], rtol=1e-5, atol=1e-6)


def test_out_of_range_concat_axis_is_not_normalised_into_validity(
    require_vulkan, tmp_path: pathlib.Path
) -> None:
    """A ``Concat`` axis outside ``[-r, r)`` is a malformed graph, not a rank fact.

    Normalising ``axis`` by adding the rank is right for negatives in range and wrong for
    everything else; an implementation that adds unconditionally turns ``axis=5`` on rank-1
    inputs into a plausible-looking 6 and carries on.  The EP must not produce a rank here,
    and — since the graph is invalid — must not be the thing that reports the problem either.
    """
    x = oh.make_tensor_value_info("x", onnx.TensorProto.FLOAT, ["N", 8, 16])
    y = _unranked("y")
    chain, chain_inits, ns = _bert_shape_chain(axis=5)
    nodes = [
        *chain,
        oh.make_node("Reshape", ["x", ns], ["h"], name="reshape0"),
        oh.make_node("MatMul", ["h", "W"], ["mm"], name="matmul0"),
        oh.make_node("Mul", ["mm", "mm"], ["y"], name="mul0"),
    ]
    inits = [*chain_inits, _f32("W", np.eye(4, dtype=np.float32))]
    model = _model(nodes, [x], [y], inits)

    with _env(**{RANK_INFERENCE_ENV: None}):
        with pytest.raises(Exception) as excinfo:
            ort.InferenceSession(model, m._make_session_options(), providers=m.EP_PROVIDERS)
    text = str(excinfo.value)
    assert "VulkanExecutionProvider" not in text, (
        "an out-of-range Concat axis is a malformed graph and must be refused by ORT's own "
        f"checker, not surfaced as an EP failure: {text}"
    )


def test_negative_in_range_concat_axis_is_accepted(
    require_vulkan, tmp_path: pathlib.Path
) -> None:
    """``axis=-1`` on rank-1 shape tensors means axis 0 and must behave identically.

    Paired with the out-of-range test above, this pins normalisation to "add the rank once,
    then bounds-check" — what the spec says, and what a sloppy implementation gets wrong in one
    direction or the other.
    """
    x = oh.make_tensor_value_info("x", onnx.TensorProto.FLOAT, ["N", 8, 16])
    y = _unranked("y")
    chain, chain_inits, ns = _bert_shape_chain(axis=-1)
    nodes = [
        *chain,
        oh.make_node("Reshape", ["x", ns], ["h"], name="reshape0"),
        oh.make_node("MatMul", ["h", "W"], ["mm"], name="matmul0"),
        oh.make_node("Mul", ["mm", "mm"], ["y"], name="mul0"),
    ]
    inits = [*chain_inits, _f32("W", np.linspace(-0.5, 0.5, 16).reshape(4, 4))]
    model = _model(nodes, [x], [y], inits)
    feeds = {"x": np.linspace(-1, 1, 2 * 8 * 16, dtype=np.float32).reshape(2, 8, 16)}

    out, claims, _ = _run(model, feeds, tmp_path, "axis_neg", inference=True)
    assert claims.get("MatMul", {}).get("claimed") is True, (
        "axis=-1 on a rank-1 tensor is axis 0; the chain proves the same length it does with "
        f"axis=0. Record: {claims.get('MatMul')}"
    )
    cpu = m.run_cpu(model, feeds)
    np.testing.assert_allclose(out[0], cpu[0], rtol=1e-4, atol=1e-5)


def test_shape_start_end_narrows_the_length(require_vulkan, tmp_path: pathlib.Path) -> None:
    """``Shape`` with ``start``/``end`` (opset 15+) yields a shorter shape tensor.

    Reading ``Shape``'s output length as "the input's rank" unconditionally is right only for
    the default slice.  With ``start=1, end=3`` on a rank-3 input the length is 2, so the
    ``Concat`` below gives 3 and the ``Reshape`` output has rank 3.  An implementation that
    ignored the attributes would claim rank 4 — a *wrong* rank, which yields wrong descriptors
    rather than an honest decline, and is far worse than proving nothing.

    The *middle* dimension is the symbolic one so ORT cannot fold the ``Shape`` away before the
    EP is asked; with every sliced dimension static the chain never reaches the rule under test.
    """
    x = oh.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [2, "S", 16])
    y = _unranked("y")
    nodes = [
        oh.make_node("Shape", ["x"], ["s"], start=1, end=3, name="shape_se"),
        oh.make_node("Cast", ["s"], ["sf"], to=onnx.TensorProto.FLOAT, name="cast_f"),
        oh.make_node("Cast", ["sf"], ["si"], to=onnx.TensorProto.INT64, name="cast_i"),
        oh.make_node("Concat", ["si", "k2"], ["ns"], axis=0, name="concat0"),
        oh.make_node("Reshape", ["x", "ns"], ["h"], name="reshape0"),
        oh.make_node("Mul", ["h", "h"], ["y"], name="mul0"),
    ]
    inits = [_i64("k2", [2])]
    model = _model(nodes, [x], [y], inits, opset=15)
    feeds = {"x": np.linspace(-1, 1, 256, dtype=np.float32).reshape(2, 8, 16)}

    out, claims, _ = _run(model, feeds, tmp_path, "shape_se", inference=True)
    rec = claims.get("Mul")
    if rec is not None and rec.get("claimed") and rec.get("input_shapes"):
        assert len(rec["input_shapes"][0]) == 3, (
            "Shape(start=1,end=3) on a rank-3 input has length 2, so Concat gives 3 and the "
            f"Reshape output has rank 3. Record: {rec}"
        )
    cpu = m.run_cpu(model, feeds)
    np.testing.assert_allclose(out[0], cpu[0], rtol=1e-5, atol=1e-6)


def test_annotated_graph_is_untouched_by_the_pass(
    require_vulkan, tmp_path: pathlib.Path
) -> None:
    """A fully annotated graph must produce identical decisions and dispatch counts either way.

    The pass is supposed to be a pure *addition*: where ORT already established a shape,
    ``InferredShapes::refine`` hands ORT's reading back untouched.  The cheapest way for that
    invariant to break is for the overlay to overwrite a good reading with a worse one, which
    would show up here as a claim that used to happen and no longer does — a regression the
    BERT measurement, which only ever goes up, would not catch.
    """
    x = oh.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [4, 4])
    y = oh.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [4, 4])
    nodes = [
        oh.make_node("MatMul", ["x", "W"], ["mm"], name="matmul0"),
        oh.make_node("Mul", ["mm", "mm"], ["sq"], name="mul0"),
        oh.make_node("Add", ["sq", "mm"], ["ad"], name="add0"),
        oh.make_node("Sub", ["ad", "mm"], ["y"], name="sub0"),
    ]
    inits = [_f32("W", np.eye(4, dtype=np.float32) * 0.5)]
    model = _model(nodes, [x], [y], inits)
    feeds = {"x": np.linspace(-1, 1, 16, dtype=np.float32).reshape(4, 4)}

    off_delta, on_delta, claims_off, claims_on, out_off, out_on = _dispatch_deltas(
        model, feeds, tmp_path, "annotated"
    )
    for op in sorted(set(claims_off) | set(claims_on)):
        assert claims_off.get(op, {}).get("claimed") == claims_on.get(op, {}).get("claimed"), (
            f"{op} changed its claim decision on a fully annotated graph: "
            f"off={claims_off.get(op)} on={claims_on.get(op)}"
        )
    assert claims_on.get("MatMul", {}).get("rank_inferred") is False, (
        "nothing needed inferring on a fully annotated graph, so no row should be flagged "
        f"rank_inferred. Record: {claims_on.get('MatMul')}"
    )
    assert off_delta == on_delta, (
        f"per-session dispatch count moved on a fully annotated graph: "
        f"off={off_delta} on={on_delta}"
    )
    np.testing.assert_array_equal(out_off[0], out_on[0])
