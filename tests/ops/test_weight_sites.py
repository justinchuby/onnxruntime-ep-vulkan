"""The anchor table, audited against the schemas it claims to be derived from.

Issue #73. ``partition::is_anchor`` used to match the bare string ``"MatMul"``, so every
``MatMul`` in a graph was an anchor and every anchor-bearing island took the exemption in
``partition::evaluate`` — bypassing both the ``min_nodes`` gate and the transfer-dominated
economics. On MiniLM-L6-v2 that claimed six one-node islands, each the attention ``AV`` batched
product with **both** operands runtime activations: 983,040 B in and 196,608 B out to buy
0.013 GFLOP. The exemption's warrant has always been *a resident weight is uploaded once and
read once per output element, so it amortises the boundary by construction*, and two activations
amortise nothing.

The predicate is now a node property: **a heavy-op node carrying a resident initializer at a
schema-designated weight site**. `rust/src/ops/partition.rs` holds the designation table.

# What this file is for, and why it is not in `rust/`

The Rust side pins the table against a hand-maintained held-out list
(`partition::tests::the_weight_site_table_matches_its_held_out_pin`) and proves that pin
load-bearing with eight mutations. That is a **change detector**: it catches an edit, but both
halves of it are written by the same hand, so it cannot catch a *derivation* error.

This file is the independent half. It reads the pinned schemas themselves — `onnx.defs` for the
default domain and `onnxruntime`'s own schema registry for `com.microsoft` — and requires the
shipped table's family membership, operand order, operand names and operand **count** to equal
them exactly. Neither the table nor the held-out list is an input to that comparison.

It reads the table out of the **built binary** via ``epctl --dump-weight-sites --json``, never
out of `partition.rs`'s source. A source scraper certifies a table that no build consumed;
issue #73's acceptance criteria call that a lexical seam and forbid it.

Run::

    pytest tests/ops/test_weight_sites.py -v --no-header
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess

import numpy as np
import onnx
import onnx.defs
import onnx.helper as oh
import onnx.numpy_helper as onh
import onnxruntime as ort
import pytest

import _models as m

# ---------------------------------------------------------------------------
# Reading the shipped table out of the binary
# ---------------------------------------------------------------------------


def _epctl_path() -> pathlib.Path | None:
    """`epctl` beside the EP library under test, or None if it was not built.

    `ONNXRUNTIME_VULKAN_EP_LIB` is the library the whole suite loads, so its directory is the
    build whose behaviour every other test in this package is describing. Taking `epctl` from
    anywhere else would audit a different build than the one under test.
    """
    lib = os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB")
    if not lib:
        return None
    exe = "epctl.exe" if os.name == "nt" else "epctl"
    candidate = pathlib.Path(lib).parent / exe
    return candidate if candidate.is_file() else None


def _shipped_sites() -> list[dict]:
    exe = _epctl_path()
    if exe is None:
        pytest.skip(
            "epctl was not found beside ONNXRUNTIME_VULKAN_EP_LIB; `cargo build --release` "
            "produces it. Skipping rather than reading partition.rs's source, which would "
            "audit a table no build consumed."
        )
    out = subprocess.run(
        [str(exe), "--dump-weight-sites", "--json"],
        capture_output=True,
        text=True,
        check=True,
        encoding="utf-8",
    )
    doc = json.loads(out.stdout)
    rows = doc["weight_sites"]
    assert rows, "epctl reported an empty weight-site table"
    return rows


def _by_family(rows: list[dict]) -> dict[str, list[dict]]:
    families: dict[str, list[dict]] = {}
    for row in rows:
        families.setdefault(row["op"], []).append(row)
    return families


# ---------------------------------------------------------------------------
# Reading the pinned schemas — the independent side of the comparison
# ---------------------------------------------------------------------------


def _split_qualified(qualified: str) -> tuple[str, str]:
    """`domain::op` → `(domain, op)`; the default domain has no prefix."""
    if "::" in qualified:
        domain, op = qualified.split("::", 1)
        return domain, op
    return "", qualified


def _schema_input_names(qualified: str) -> list[str] | None:
    """Ordered input names from the *installed* onnx / onnxruntime packages.

    The default domain comes from `onnx.defs`, which is the normative source. `com.microsoft`
    has no published schema file, so it comes from onnxruntime's own registry — the same
    registry the runtime consults when it validates a node, which is exactly the authority the
    EP's claim predicates are written against.
    """
    domain, op = _split_qualified(qualified)
    if domain == "":
        try:
            schema = onnx.defs.get_schema(op, "")
        except Exception:
            return None
        return [p.name for p in schema.inputs]

    from onnxruntime.capi._pybind_state import get_all_operator_schema

    for schema in get_all_operator_schema():
        if schema.domain == domain and schema.name == op:
            return [p.name for p in schema.inputs]
    return None


# ---------------------------------------------------------------------------
# The audit
# ---------------------------------------------------------------------------


def test_every_family_matches_its_pinned_schema_exactly() -> None:
    """Family by family: same operand count, same order, same names.

    This is the assertion the previous revision of this work did not have. A table compared
    only against a copy of itself stays green when two names are swapped; this one does not,
    because the other side of the comparison is `onnx.defs` / onnxruntime.

    Falsifier: swap two `name` fields in `WEIGHT_SITES`, or delete a row, or add one, and this
    test reds while every internal consistency check stays green.
    """
    shipped = _by_family(_shipped_sites())
    assert shipped, "no families in the shipped table"

    unresolved: list[str] = []
    for family, rows in sorted(shipped.items()):
        expected = _schema_input_names(family)
        if expected is None:
            unresolved.append(family)
            continue

        assert len(rows) == len(expected), (
            f"{family}: the shipped table has {len(rows)} operand(s), the pinned schema declares "
            f"{len(expected)}. A table that is not the same length as its schema is misclassifying "
            f"whatever now sits at the indices past the end.\n"
            f"  shipped: {[r['name'] for r in rows]}\n"
            f"  schema:  {expected}"
        )
        for i, (row, name) in enumerate(zip(rows, expected)):
            assert row["index"] == i, (
                f"{family}: row {i} declares index {row['index']}; the table's index must be the "
                f"operand's position in the pinned schema or it is a label, not an offset"
            )
            assert row["name"] == name, (
                f"{family} operand {i}: the shipped table calls it {row['name']!r}, the pinned "
                f"schema calls it {name!r}"
            )

    assert not unresolved, (
        f"these families are in the shipped anchor table and have no schema in the installed "
        f"onnx / onnxruntime packages: {unresolved}. Either the op was renamed upstream or the "
        f"table names an op that does not exist; both are defects and neither may be silent."
    )


def test_the_designated_sites_are_the_ones_the_derivation_names() -> None:
    """The per-family designated operand *names*, stated independently of the table's order.

    Written as names rather than indices on purpose: an index list would re-derive from the same
    ordering the table supplies, and the point is to pin the semantic choice. These names are
    read off the pinned schemas' declared shapes — a designated site is one whose extent scales
    with the op's reduction dimension, so a resident value there is reused across every output
    element.
    """
    expected: dict[str, set[str]] = {
        # Either factor of a product may be the resident one.
        "MatMul": {"A", "B"},
        "Gemm": {"A", "B"},  # not C, which is a bias
        "Conv": {"W"},  # not X, not B
        "ConvTranspose": {"W"},
        # Fused SDPA over already-projected Q/K/V: no weight input exists.
        "Attention": set(),
        "com.microsoft::Attention": {"weights"},
        "com.microsoft::MatMulNBits": {"B", "scales", "zero_points", "g_idx"},
        # THE 193 -> 161 CORRECTION. Every GQA input is an activation, a KV cache, a length, a
        # per-head scalar or an O(head_size) norm gain.
        "com.microsoft::GroupQueryAttention": set(),
        "com.microsoft::MultiHeadAttention": set(),
        # Nine. Not four: packed weights + scales + zero points, for each of fc1/fc2/fc3.
        "com.microsoft::QMoE": {
            "fc1_experts_weights",
            "fc2_experts_weights",
            "fc3_experts_weights",
            "fc1_scales",
            "fc2_scales",
            "fc3_scales",
            "fc1_zero_points",
            "fc2_zero_points",
            "fc3_zero_points",
        },
        "com.microsoft::LinearAttention": set(),
    }

    shipped = _by_family(_shipped_sites())
    assert set(shipped) == set(expected), (
        f"family set changed: shipped {sorted(set(shipped) - set(expected))} unexpected, "
        f"{sorted(set(expected) - set(shipped))} missing"
    )
    for family, names in expected.items():
        actual = {r["name"] for r in shipped[family] if r["designated"]}
        assert actual == names, (
            f"{family}: designated {sorted(actual)}, expected {sorted(names)}"
        )

    total = sum(len(v) for v in expected.values())
    assert total == 20, "twenty designated sites over eleven families"
    all_rows = [r for rows in shipped.values() for r in rows]
    assert len(all_rows) == 84
    assert sum(1 for r in all_rows if r["designated"]) == 20


def test_qmoe_designates_nine_sites_and_none_of_them_is_an_activation_scale() -> None:
    """The QMoE row-block, spelled out, because it has been miscounted before.

    Four of QMoE's twenty-one inputs are activation-side scales (17-20) and one is a
    token-shaped tensor named ``router_weights`` (14). A reading that keys on names gets twelve
    sites; the correct answer is nine.
    """
    rows = _by_family(_shipped_sites())["com.microsoft::QMoE"]
    assert len(rows) == 21
    designated = [r for r in rows if r["designated"]]
    assert len(designated) == 9, [r["name"] for r in designated]
    assert {r["index"] for r in designated} == {2, 3, 5, 6, 8, 9, 11, 12, 13}

    by_name = {r["name"]: r for r in rows}
    assert not by_name["router_weights"]["designated"], (
        "router_weights is `(num_tokens, num_experts)` — its extent scales with tokens, so it "
        "is an activation however it is named"
    )
    for name in ("fc1_act_scale", "fc2_act_scale", "fc1_act_block_scale", "fc2_act_block_scale"):
        assert not by_name[name]["designated"], f"{name} is activation-side by the schema's word"
    for name in ("fc1_global_scale", "fc2_global_scale"):
        assert not by_name[name]["designated"], (
            f"{name} is `(num_experts,)` — weight-side in origin and far too small to amortise a "
            f"boundary; the packed weights it scales are the site that does"
        )


def test_group_query_attention_designates_nothing_including_its_norm_gains() -> None:
    """GQA is the reason Phi-3.5-mini-int4 has 161 anchors and not 193.

    Its 32 nodes remain *claimed*; `docs/OP_COVERAGE.md` reports the anchor count and the
    claimed-node count separately because they are different questions.
    """
    rows = _by_family(_shipped_sites())["com.microsoft::GroupQueryAttention"]
    assert len(rows) == 16
    assert not any(r["designated"] for r in rows), [
        r["name"] for r in rows if r["designated"]
    ]
    by_name = {r["name"]: r for r in rows}
    # The two that most look like weights, called out by name so a future edit has to argue.
    for name in ("q_norm_weight", "k_norm_weight"):
        assert by_name[name]["kind"] == "bias_or_gain", (
            f"{name} is an O(head_size) RMS-norm gain: learned and resident, but one multiply "
            f"per element with no reuse across outputs"
        )
    for name in ("cos_cache", "sin_cache"):
        assert by_name[name]["kind"] == "precomputed_table", (
            f"{name} is resident on every rotary model; designating it would make the anchor "
            f"property vacuous for every LLM"
        )


def test_every_row_carries_a_justification_and_a_kind() -> None:
    """No silent designations, and no kind outside the closed set."""
    kinds = {
        "factor",
        "quant_payload",
        "activation",
        "bias_or_gain",
        "per_group_scalar",
        "precomputed_table",
        "cached_state",
        "mask_or_length",
    }
    designating = {"factor", "quant_payload"}
    for row in _shipped_sites():
        assert row["kind"] in kinds, row
        assert row["reason"].strip(), f"{row['op']}[{row['index']}] has no justification"
        assert row["designated"] == (row["kind"] in designating), (
            f"{row['op']}[{row['index']}] designation disagrees with its kind: {row}"
        )


# ---------------------------------------------------------------------------
# The other half: what the EP actually does with it
# ---------------------------------------------------------------------------
#
# Everything above is about a table. These two exercise the shipped predicate through a real
# `GetCapability` call on a real ORT session, which is the only evidence that the table is wired
# to anything. The two models differ in exactly one respect — whether the MatMul's B operand is
# a graph input or a graph initializer — so nothing else can account for a difference in verdict.
#
# Both models have TWO disjoint one-node clusters, so the sole-island override in
# `partition::gate_islands` is not the term deciding either of them.


_K, _N = 32, 16


def _two_branch_matmul(*, weight_resident: bool) -> bytes:
    """Two independent one-node `MatMul` clusters.

    `weight_resident=False`: both operands of both nodes are graph inputs — the MiniLM attention
    shape. Each cluster is a one-node island with no anchor, so the size gate declines it.

    `weight_resident=True`: operand B is a graph initializer. Each cluster is a one-node island
    with one anchor, and the exemption claims it.

    `MatMul` is registered fp32-only (`rust/src/ops/matmul.rs`) and its claim predicate needs a
    fully static rank-2 B, which both variants satisfy.
    """
    rng = np.random.default_rng(73)
    a1 = oh.make_tensor_value_info("a1", onnx.TensorProto.FLOAT, [4, _K])
    a2 = oh.make_tensor_value_info("a2", onnx.TensorProto.FLOAT, [4, _K])
    y1 = oh.make_tensor_value_info("y1", onnx.TensorProto.FLOAT, [4, _N])
    y2 = oh.make_tensor_value_info("y2", onnx.TensorProto.FLOAT, [4, _N])

    n1 = oh.make_node("MatMul", ["a1", "b1"], ["y1"], name="mm1")
    n2 = oh.make_node("MatMul", ["a2", "b2"], ["y2"], name="mm2")

    inputs = [a1, a2]
    initializer = []
    for name in ("b1", "b2"):
        arr = rng.standard_normal((_K, _N)).astype(np.float32)
        if weight_resident:
            initializer.append(onh.from_array(arr, name=name))
        else:
            inputs.append(oh.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [_K, _N]))

    graph = oh.make_graph(
        [n1, n2], "two_matmul", inputs, [y1, y2], initializer=initializer
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 21)])
    model.ir_version = 8
    return model.SerializeToString()


def _feeds(model_bytes: bytes) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    model = onnx.load_from_string(model_bytes)
    return {
        i.name: rng.standard_normal(
            [d.dim_value for d in i.type.tensor_type.shape.dim]
        ).astype(np.float32)
        for i in model.graph.input
    }


def _run_and_read_claims(model_bytes: bytes, log_path: pathlib.Path) -> dict[str, dict]:
    feeds = _feeds(model_bytes)
    opts = m._make_session_options()
    os.environ["ONNXRUNTIME_EP_VULKAN_CLAIM_LOG"] = str(log_path)
    try:
        session = ort.InferenceSession(model_bytes, opts, providers=m.EP_PROVIDERS)
        results = session.run(None, feeds)
    finally:
        del os.environ["ONNXRUNTIME_EP_VULKAN_CLAIM_LOG"]

    # Correctness holds either way — a decline means CPU fallback, not a wrong answer.
    expected1 = feeds["a1"] @ (feeds["b1"] if "b1" in feeds else _initializer(model_bytes, "b1"))
    np.testing.assert_allclose(results[0], expected1, rtol=1e-4, atol=1e-4)
    return m.read_claim_log(log_path)


def _initializer(model_bytes: bytes, name: str) -> np.ndarray:
    model = onnx.load_from_string(model_bytes)
    for init in model.graph.initializer:
        if init.name == name:
            return onh.to_array(init)
    raise AssertionError(f"no initializer {name!r}")


def test_an_activation_only_matmul_island_is_declined_by_the_partition_gate(
    require_vulkan, tmp_path: pathlib.Path
) -> None:
    """The six MiniLM islands, reproduced as a two-cluster model and declined.

    Artifact: ``code == "partition"`` in the CLAIM_LOG for `MatMul`.

    Falsifier, and it is the whole issue: if `is_anchor` goes back to matching the op name, this
    node becomes an anchor, takes the exemption in `partition::evaluate`, and is claimed — and
    this assertion is what says so.
    """
    model = _two_branch_matmul(weight_resident=False)
    claims = _run_and_read_claims(model, tmp_path / "claim_matmul_activation.jsonl")

    assert "MatMul" in claims, (
        "no CLAIM_LOG record for MatMul; either the log is not written or the EP never saw the "
        "node, and in neither case does this test mean anything"
    )
    rec = claims["MatMul"]
    assert rec["code"] == "partition", (
        f"a one-node island of a MatMul whose operands are both runtime activations must be "
        f"declined by the partition gate; got code={rec['code']!r}. Full record: {rec}"
    )
    assert rec["claimed"] is False, rec


def test_a_weight_bearing_matmul_island_of_the_same_shape_is_still_claimed(
    require_vulkan, tmp_path: pathlib.Path
) -> None:
    """The negative control for the test above, and the guard against over-declining.

    The model differs from it in exactly one bit: B is an initializer. If narrowing the anchor
    predicate had cost us the case it was written for — a lone weight-bearing `MatMul` at a
    model's tail, which is MobileNetV2-12's `Gemm` — this is what would catch it.
    """
    model = _two_branch_matmul(weight_resident=True)
    claims = _run_and_read_claims(model, tmp_path / "claim_matmul_weight.jsonl")

    assert "MatMul" in claims, "no CLAIM_LOG record for MatMul"
    rec = claims["MatMul"]
    assert rec["claimed"] is True, (
        f"a one-node island whose MatMul carries a resident weight at operand B must still be "
        f"claimed — it is exactly the case the anchor exemption exists for. Full record: {rec}"
    )
    assert rec.get("code") in (None, "", "ok"), rec
