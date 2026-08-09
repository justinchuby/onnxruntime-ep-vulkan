"""Independent (Python-side) proof for issue #73's weight-site audit. Owner: Link.

WHY THIS FILE EXISTS
====================
``rust/src/ops/partition.rs`` carries the shipped instrument (``HEAVY_OP_FAMILIES``,
``WeightOperand``, ``WeightSiteReason``, ``SCHEMA_SOURCES``, ``EXPECTED_INPUT_COUNTS``,
``WEIGHT_SITE_AUDIT``, ``weight_sites``, ``classify_weight_operand``, ``is_anchor``) and ~30
``#[test]`` functions that check it in isolation, in Rust, against Rust's own copy of the data.
A completeness rule proven only by comparing a table to itself (or to a second table generated
from the same source in the same language, by the same author, in the same compiler run) cannot
rule out the two having drifted together. This file is the other half of MANDATORY fix #3's
requirement — "deleting the final QMoE row (site 20) must fail both Rust/Python relevant
guards" — implemented as a **second, independent transcription-and-check pipeline**, written
in a different language, run by a different toolchain, that re-derives the same facts from the
same committed source text rather than importing Rust's own answer.

What "independent" means here, precisely: this file's regexes are its own, not a call into
``rust/tools/probe_island_counterfactual.py`` (which only ever parses ``HEAVY_OP_FAMILIES``,
not ``WEIGHT_SITE_AUDIT`` or ``EXPECTED_INPUT_COUNTS``). If a future edit desynchronises the
Rust tables from each other, both this file and the Rust suite must independently notice —
neither is allowed to depend on the other having already caught it.

WHAT IS -- AND IS NOT -- PROVEN HERE
=====================================
Proven: the shipped source text, read fresh by a second parser, is internally consistent (row
counts match the pinned expected-arity table; QMoE's trailing two inputs are present and
catalogued as non-weight; GQA designates no weight site) and that the completeness contract
would visibly break under the specific mutation MANDATORY fix #3 names.
Also proven: the exact evidence figures this repository's decision records depend on --
355 total claimed nodes, 161 ``com.microsoft::MatMulNBits``, 32 ``com.microsoft::
GroupQueryAttention`` -- re-derived directly from the exact committed
``bench/results/_claim_log_phi35_r15_after.jsonl``, not carried forward from memory or from a
prior PR's prose.
NOT proven, and not claimed: that the transcription is a *semantically correct* reading of the
pinned upstream ``.cc`` schema. That is manual, disclosed, human transcription (see
``SCHEMA_SOURCES`` and the doc comment on ``WEIGHT_SITE_AUDIT`` in ``partition.rs``) and no tool
in this repository — this file included — fetches ONNX Runtime source over the network or
parses its C++.

Run::

    pytest tests/ops/test_weight_site_audit.py -v --no-header
"""

from __future__ import annotations

import json
import pathlib
import re

import numpy as np
import onnx
import onnx.helper as oh
import onnxruntime as ort
import pytest

import _models as m

# ---------------------------------------------------------------------------
# Independent Rust-source parsing (own regexes; see module docstring on why this is not a call
# into rust/tools/probe_island_counterfactual.py).
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_PARTITION_RS = _REPO_ROOT / "rust" / "src" / "ops" / "partition.rs"

# Reasons that make an input a candidate resident-weight site. This is a second, independent
# transcription of `WeightSiteReason::designates_weight` (partition.rs) — not read out of the
# Rust source, because the point of this file is to re-derive facts, not to parse the very
# predicate it is trying to check from the outside.
_WEIGHT_DESIGNATING_REASONS = frozenset({"Weight", "WeightScale", "WeightZeroPoint", "Positional"})


def _partition_rs_text() -> str:
    if not _PARTITION_RS.is_file():
        raise RuntimeError(
            f"ERROR(instrument): {_PARTITION_RS} does not exist -- cannot independently "
            "re-derive the weight-site audit tables. Refusing to skip or fall back to a cached "
            "copy that could silently drift from the shipped source."
        )
    return _PARTITION_RS.read_text(encoding="utf-8")


def parse_heavy_op_families(text: str) -> tuple[str, ...]:
    """Re-derive `HEAVY_OP_FAMILIES` from the Rust source text, independently of the Rust tests."""
    match = re.search(r"pub const HEAVY_OP_FAMILIES:\s*&\[&str\]\s*=\s*&\[(.*?)\];", text, re.DOTALL)
    if not match:
        raise RuntimeError(
            "ERROR(instrument): could not find `HEAVY_OP_FAMILIES` in partition.rs -- its "
            "declaration shape has changed and this parser has not been updated to match."
        )
    names = re.findall(r'"([^"]*)"', match.group(1))
    if not names:
        raise RuntimeError(
            "ERROR(instrument): `HEAVY_OP_FAMILIES` parsed to zero entries."
        )
    return tuple(names)


def parse_expected_input_counts(text: str) -> dict[str, int]:
    """Re-derive `EXPECTED_INPUT_COUNTS` (family -> pinned expected input arity)."""
    match = re.search(
        r"pub const EXPECTED_INPUT_COUNTS:\s*&\[\(&str,\s*usize\)\]\s*=\s*&\[(.*?)\];",
        text,
        re.DOTALL,
    )
    if not match:
        raise RuntimeError(
            "ERROR(instrument): could not find `EXPECTED_INPUT_COUNTS` in partition.rs -- its "
            "declaration shape has changed and this parser has not been updated to match."
        )
    rows = re.findall(r'\(\s*"([^"]*)"\s*,\s*(\d+)\s*\)', match.group(1))
    if not rows:
        raise RuntimeError(
            "ERROR(instrument): `EXPECTED_INPUT_COUNTS` parsed to zero entries."
        )
    return {family: int(count) for family, count in rows}


def parse_weight_site_audit(text: str) -> list[dict]:
    """Re-derive `WEIGHT_SITE_AUDIT` (one dict per `SchemaInput` row) from the Rust source text."""
    match = re.search(
        r"pub const WEIGHT_SITE_AUDIT:\s*&\[SchemaInput\]\s*=\s*&\[(.*?)\n\];",
        text,
        re.DOTALL,
    )
    if not match:
        raise RuntimeError(
            "ERROR(instrument): could not find `WEIGHT_SITE_AUDIT` in partition.rs -- its "
            "declaration shape has changed and this parser has not been updated to match."
        )
    row_pattern = re.compile(
        r"SchemaInput\s*\{\s*"
        r'family:\s*"([^"]*)"\s*,\s*'
        r"index:\s*(\d+)\s*,\s*"
        r'name:\s*"([^"]*)"\s*,\s*'
        r"reason:\s*WeightSiteReason::(\w+)\s*,\s*"
        r"\}",
        re.DOTALL,
    )
    rows = [
        {"family": family, "index": int(index), "name": name, "reason": reason}
        for family, index, name, reason in row_pattern.findall(match.group(1))
    ]
    if not rows:
        raise RuntimeError(
            "ERROR(instrument): `WEIGHT_SITE_AUDIT` parsed to zero rows -- the outer regex "
            "matched the constant but the inner row pattern found nothing inside it."
        )
    return rows


@pytest.fixture(scope="module")
def partition_rs_text() -> str:
    return _partition_rs_text()


@pytest.fixture(scope="module")
def heavy_op_families(partition_rs_text: str) -> tuple[str, ...]:
    return parse_heavy_op_families(partition_rs_text)


@pytest.fixture(scope="module")
def expected_input_counts(partition_rs_text: str) -> dict[str, int]:
    return parse_expected_input_counts(partition_rs_text)


@pytest.fixture(scope="module")
def weight_site_audit(partition_rs_text: str) -> list[dict]:
    return parse_weight_site_audit(partition_rs_text)


# ---------------------------------------------------------------------------
# MANDATORY fix #1 — QMoE has 21 inputs (0-20); 19/20 are activation block scales, not weights.
# ---------------------------------------------------------------------------


def test_qmoe_audit_has_exactly_21_rows_indices_0_through_20(weight_site_audit: list[dict]) -> None:
    """Independently-parsed row count for QMoE must be 21 (indices 0..20 inclusive, no gaps)."""
    qmoe_rows = [r for r in weight_site_audit if r["family"] == "com.microsoft::QMoE"]
    assert len(qmoe_rows) == 21, (
        f"Expected 21 QMoE audit rows (pinned schema has inputs 0-20); parsed {len(qmoe_rows)}. "
        "Full parsed indices: " + repr(sorted(r["index"] for r in qmoe_rows))
    )
    indices = sorted(r["index"] for r in qmoe_rows)
    assert indices == list(range(21)), f"QMoE indices must be exactly 0..20 with no gap; got {indices}"


def test_qmoe_inputs_19_and_20_are_activation_block_scales_not_weights(
    weight_site_audit: list[dict],
) -> None:
    """MANDATORY fix #1: `fc1_act_block_scale`/`fc2_act_block_scale` are audited, named, typed
    correctly, and explicitly excluded from the weight-designating reason set."""
    by_index = {
        r["index"]: r for r in weight_site_audit if r["family"] == "com.microsoft::QMoE"
    }
    assert 19 in by_index and 20 in by_index, "QMoE inputs 19 and 20 must both be catalogued"
    row19, row20 = by_index[19], by_index[20]
    assert row19["name"] == "fc1_act_block_scale", row19
    assert row20["name"] == "fc2_act_block_scale", row20
    assert row19["reason"] == "ActivationBlockScale", row19
    assert row20["reason"] == "ActivationBlockScale", row20
    assert row19["reason"] not in _WEIGHT_DESIGNATING_REASONS, (
        "input 19 (fc1_act_block_scale) must NOT be classified as a resident weight site"
    )
    assert row20["reason"] not in _WEIGHT_DESIGNATING_REASONS, (
        "input 20 (fc2_act_block_scale) must NOT be classified as a resident weight site"
    )


# ---------------------------------------------------------------------------
# GQA designates no weight sites (PRESERVE/RECONSTRUCT requirement).
# ---------------------------------------------------------------------------


def test_gqa_audit_has_16_rows_and_designates_zero_weight_sites(
    weight_site_audit: list[dict], expected_input_counts: dict[str, int]
) -> None:
    gqa_rows = [
        r for r in weight_site_audit if r["family"] == "com.microsoft::GroupQueryAttention"
    ]
    assert expected_input_counts["com.microsoft::GroupQueryAttention"] == 16, (
        "pinned expected input count for GQA must be 16"
    )
    assert len(gqa_rows) == 16, f"GQA audit must have 16 rows; parsed {len(gqa_rows)}"
    weight_rows = [r for r in gqa_rows if r["reason"] in _WEIGHT_DESIGNATING_REASONS]
    assert weight_rows == [], (
        f"GQA must designate zero weight sites; found: {weight_rows}. GQA's arithmetic runs "
        "over already-projected runtime activations, caches, and small per-head scale terms — "
        "none of that is a resident constant weight."
    )


# ---------------------------------------------------------------------------
# MANDATORY fix #3 — a real per-family expected-input-count contract, and its mutation proof.
# ---------------------------------------------------------------------------


def test_expected_input_counts_match_audit_row_counts_per_family(
    weight_site_audit: list[dict], expected_input_counts: dict[str, int], heavy_op_families
) -> None:
    """Every family's independently-pinned expected arity must equal the (independently
    re-parsed) audit row count for that family — and every heavy family must have a pinned
    count at all."""
    counts_by_family: dict[str, int] = {}
    for row in weight_site_audit:
        counts_by_family[row["family"]] = counts_by_family.get(row["family"], 0) + 1

    for family, expected in expected_input_counts.items():
        got = counts_by_family.get(family, 0)
        assert got == expected, (
            f"family {family}: audit has {got} rows, pinned schema says {expected} "
            "(re-derived independently in Python)"
        )

    for family in heavy_op_families:
        assert family in expected_input_counts, (
            f"family {family} is heavy but has no pinned expected input count"
        )


def test_python_guard_detects_deleted_trailing_qmoe_row(
    weight_site_audit: list[dict], expected_input_counts: dict[str, int]
) -> None:
    """MANDATORY: 'deleting the final QMoE row (site 20) must fail both Rust/Python relevant
    guards.' This is the Python guard's half of that proof, independent of the Rust test
    `mutation_proof_deleting_qmoe_input_20_would_fail_the_count_contract` (same partition.rs,
    same required behaviour, parsed and checked by a second toolchain).

    The mutation is simulated on the parsed rows -- the shipped table on disk is never touched
    -- and the assertion is exactly the completeness check the previous PR's contiguity-from-zero
    logic could not perform: this desyncs `EXPECTED_INPUT_COUNTS` from a trailing-truncated audit.
    """
    mutated_rows = [
        r
        for r in weight_site_audit
        if not (r["family"] == "com.microsoft::QMoE" and r["index"] == 20)
    ]
    mutated_qmoe_count = sum(1 for r in mutated_rows if r["family"] == "com.microsoft::QMoE")
    pinned_expected = expected_input_counts["com.microsoft::QMoE"]

    assert mutated_qmoe_count == 20, "sanity: simulated deletion should drop QMoE to 20 rows"
    assert pinned_expected == 21, "sanity: pinned schema count for QMoE must still read 21"
    assert mutated_qmoe_count != pinned_expected, (
        "a trailing-row deletion from WEIGHT_SITE_AUDIT must desynchronise from the "
        "independently-pinned EXPECTED_INPUT_COUNTS entry -- if this assertion ever fails "
        "(i.e. the counts still match after the deletion), the count contract has regressed "
        "back into contiguity-from-zero pseudo-completeness and can no longer catch a trailing "
        "omission."
    )


# ---------------------------------------------------------------------------
# Exact evidence subjects/counts — 355 claimed / 161 MatMulNBits / 32 GQA, re-derived from the
# exact committed collection (bench/results/_claim_log_phi35_r15_after.jsonl), not carried
# forward from prose.
# ---------------------------------------------------------------------------

_CLAIM_LOG_PHI35 = _REPO_ROOT / "bench" / "results" / "_claim_log_phi35_r15_after.jsonl"


def _last_record_per_node(path: pathlib.Path) -> dict[str, dict]:
    if not path.is_file():
        raise RuntimeError(
            f"ERROR(instrument): {path} does not exist -- cannot re-derive the claimed-node "
            "evidence figures from the exact committed collection."
        )
    last: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            last[rec["node"]] = rec
    if not last:
        raise RuntimeError(f"ERROR(instrument): {path} parsed to zero records.")
    return last


def test_claim_log_re_derives_exact_355_161_32(recwarn) -> None:
    """Re-derive the exact evidence counts straight from the committed JSONL, independent of
    any prior PR's prose. Last record per unique node name wins (a node's final claim
    decision — later stages, e.g. partition, can overwrite an earlier registry claim)."""
    last = _last_record_per_node(_CLAIM_LOG_PHI35)
    total_claimed = sum(1 for r in last.values() if r.get("claimed") is True)
    matmulnbits_claimed = sum(
        1
        for r in last.values()
        if r.get("claimed") is True and r.get("op") == "com.microsoft::MatMulNBits"
    )
    gqa_claimed = sum(
        1
        for r in last.values()
        if r.get("claimed") is True and r.get("op") == "com.microsoft::GroupQueryAttention"
    )
    assert len(last) == 363, f"expected 363 unique node records; got {len(last)}"
    assert total_claimed == 355, f"expected 355 total claimed nodes; got {total_claimed}"
    assert matmulnbits_claimed == 161, f"expected 161 claimed MatMulNBits nodes; got {matmulnbits_claimed}"
    assert gqa_claimed == 32, f"expected 32 claimed GroupQueryAttention nodes; got {gqa_claimed}"
    assert matmulnbits_claimed + gqa_claimed == 193, (
        "161 MatMulNBits + 32 GQA must equal 193 -- the anchor total this PR's docs cite"
    )


# ---------------------------------------------------------------------------
# Behavioural proof: the name-only exemption bug (issue #73) is closed at the partition layer.
#
# Mirrors test_partition_gate.py's two-disjoint-cluster strategy: a single-cluster graph takes
# the "single cluster" exemption in ep.rs and never reaches partition::evaluate at all, so a
# valid behavioural test needs at least two disjoint (non-edge-sharing) claimed clusters.
#
# `MatMul`'s claim predicate (rust/src/ops/matmul.rs::matmul) does not distinguish an
# activation-only node from a weight-bearing one by constant-ness at all -- both clusters below
# are claimed identically at the registry stage. The divergence this test proves lives entirely
# at the partition stage: `classify_weight_operand` reads `NodeView::input_is_constant` on
# `weight_sites("MatMul") == &[1]` (input B), which is `true` only when B is a graph
# initializer. That is the actual "node property, not a name test" fix for issue #73.
# ---------------------------------------------------------------------------


def _two_cluster_matmul_model() -> bytes:
    """Two disjoint single-node MatMul clusters, one activation-only and one weight-bearing.

    Cluster A ("mm_activation"): a (fp32, [4,4]) x b (fp32, [4,4]) -> y_act. Both operands are
    graph inputs -- `input_is_constant` reads `false` for B -- so `classify_weight_operand`
    returns `Absent` and `is_anchor` is `false`. The island has 1 node and 0 anchors: TooSmall
    (1 < min_nodes=4) fires and the node declines to CPU with code "partition".

    Cluster B ("mm_weight"): c (fp32, [4,4]) x W (fp32, [4,4] constant initializer) -> y_w. B
    is a graph initializer -- `input_is_constant` reads `true` -- so `classify_weight_operand`
    returns `Present` and `is_anchor` is `true`. The anchor exemption (partition::evaluate)
    unconditionally claims the island regardless of its size.

    The two clusters share no nodes or edges, so ORT partitions them as separate connected
    components and `partition::evaluate` is invoked once per island (this is NOT the
    single-cluster exemption path).
    """
    a = oh.make_tensor_value_info("a", onnx.TensorProto.FLOAT, [4, 4])
    b = oh.make_tensor_value_info("b", onnx.TensorProto.FLOAT, [4, 4])
    y_act = oh.make_tensor_value_info("y_act", onnx.TensorProto.FLOAT, [4, 4])
    c = oh.make_tensor_value_info("c", onnx.TensorProto.FLOAT, [4, 4])
    y_w = oh.make_tensor_value_info("y_w", onnx.TensorProto.FLOAT, [4, 4])

    n_act = oh.make_node("MatMul", inputs=["a", "b"], outputs=["y_act"], name="mm_activation")
    n_w = oh.make_node("MatMul", inputs=["c", "W"], outputs=["y_w"], name="mm_weight")

    w_init = oh.make_tensor(
        "W", onnx.TensorProto.FLOAT, [4, 4], (np.eye(4, dtype=np.float32) * 0.5).flatten().tolist()
    )

    graph = oh.make_graph(
        [n_act, n_w],
        "two_cluster_matmul",
        [a, b, c],
        [y_act, y_w],
        initializer=[w_init],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 15)])
    model.ir_version = 8
    return model.SerializeToString()


def _last_record_per_node_from_text(text: str) -> dict[str, dict]:
    last: dict[str, dict] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        last[rec["node"]] = rec
    return last


def test_activation_only_matmul_declines_weight_bearing_matmul_claims(
    require_vulkan, tmp_path
) -> None:
    """Issue #73's actual falsifier: a name-only `is_anchor("MatMul")` would treat both clusters
    identically (both would either be exempted or both declined). This test proves they diverge
    by weight presence alone, which is only possible if `is_anchor` is a node property.

    Falsifier: if `mm_activation`'s final CLAIM_LOG record is `claimed=True` (the pre-#73 bug —
    any MatMul-named node exempted regardless of whether it carries a weight), or if
    `mm_weight`'s final record is declined with code "partition" (the anchor exemption not
    firing for an actual weight-bearing node), this test fails.
    """
    log_path = tmp_path / "claim_weight_site_audit.jsonl"
    model = _two_cluster_matmul_model()
    rng = np.random.default_rng(73)
    feeds = {
        "a": rng.standard_normal((4, 4)).astype(np.float32),
        "b": rng.standard_normal((4, 4)).astype(np.float32),
        "c": rng.standard_normal((4, 4)).astype(np.float32),
    }

    opts = m._make_session_options()
    import os

    os.environ["ONNXRUNTIME_EP_VULKAN_CLAIM_LOG"] = str(log_path)
    try:
        session = ort.InferenceSession(model, opts, providers=m.EP_PROVIDERS)
        results = session.run(None, feeds)
    finally:
        del os.environ["ONNXRUNTIME_EP_VULKAN_CLAIM_LOG"]

    out_names = [o.name for o in session.get_outputs()]
    outputs = dict(zip(out_names, results))
    w = (np.eye(4, dtype=np.float32) * 0.5)
    np.testing.assert_allclose(outputs["y_act"], feeds["a"] @ feeds["b"], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(outputs["y_w"], feeds["c"] @ w, rtol=1e-4, atol=1e-4)

    claims = _last_record_per_node_from_text(log_path.read_text(encoding="utf-8"))
    assert "mm_activation" in claims, "no CLAIM_LOG record for the activation-only MatMul node"
    assert "mm_weight" in claims, "no CLAIM_LOG record for the weight-bearing MatMul node"

    act_rec = claims["mm_activation"]
    w_rec = claims["mm_weight"]

    assert act_rec["claimed"] is False, (
        "activation-only MatMul (no constant weight operand) must NOT be exempted from the "
        f"partition size gate; full record: {act_rec}"
    )
    assert act_rec["code"] == "partition", (
        "activation-only MatMul's 1-node, 0-anchor island must decline with code 'partition' "
        f"(TooSmall); full record: {act_rec}"
    )
    assert w_rec["claimed"] is True, (
        "weight-bearing MatMul (constant B) must be exempted via the anchor rule even as a "
        f"1-node island; full record: {w_rec}"
    )
