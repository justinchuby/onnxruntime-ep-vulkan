"""Every anchor number in this repository must name an artifact field or say it is a model.

WHY THIS FILE EXISTS
--------------------
Issue #73 accumulated three revisions' worth of numbers that read as measurements and were not:

* ``anchors: 193`` was carried in ``Island`` constants documented as "read out of
  ``PartitionStats``". **``PartitionStats`` has no anchor field.** The number came from applying
  a name-only predicate to claim-log op names. It was a recomputation — a MODEL — wearing a
  measurement's clothes, and the doc comment beside it asserted the opposite.
* ``docs/OP_COVERAGE.md`` asserted "225 anchors" on Phi-3.5, a figure with no witness at all.
* "no censused model contains a rank-2 ``MatMul`` with a runtime ``B``" was asserted while
  ``bench/results/matmul_shape_space_bert.json`` contains exactly one.

The pattern is the same each time: a number that can only be *derived* is stated as though it
were *read*, and then the derivation is forgotten while the number is inherited. So this file
does not check prose for tone. It re-derives each surviving number from the artifact it claims
to come from, and it fails if the source stops supporting it.

R10 applies to numbers as much as to mechanisms: the falsifier for "this figure is real" is the
artifact field it was read from. Where there is no such field, the figure must be labelled
MODEL and be reproducible from fields that do exist — which is what these tests reproduce.

Run::

    pytest tests/ops/test_anchor_claims_are_witnessed.py -v --no-header
"""

from __future__ import annotations

import json
import pathlib
import re

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[2]
_PARTITION_RS = _REPO / "rust" / "src" / "ops" / "partition.rs"
_MATMUL_RS = _REPO / "rust" / "src" / "ops" / "matmul.rs"
_OP_COVERAGE = _REPO / "docs" / "OP_COVERAGE.md"
_PHI35_CLAIM_LOG = _REPO / "bench" / "results" / "_claim_log_phi35_r15_after.jsonl"
_BERT_SHAPE_SPACE = _REPO / "bench" / "results" / "matmul_shape_space_bert.json"


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# ---------------------------------------------------------------------------------------
# 1. The Phi-3.5 anchor figure
# ---------------------------------------------------------------------------------------


def test_no_artifact_in_this_repository_carries_an_anchor_count():
    """The premise. If this ever becomes false, the MODEL labels should be replaced by a read.

    Scans every JSON/JSONL artifact under ``bench/results`` and ``evidence`` for a field named
    ``anchors`` holding a number. The only ``anchors`` keys that exist hold *lists of op names*
    (the island-counterfactual probe's ranking set), which is a configuration echo, not a count.
    """
    numeric_anchor_fields: list[str] = []
    roots = [_REPO / "bench" / "results", _REPO / "evidence"]
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.json")):
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue

            def walk(node, where):
                if isinstance(node, dict):
                    for k, v in node.items():
                        if k == "anchors" and isinstance(v, (int, float)) and not isinstance(
                            v, bool
                        ):
                            numeric_anchor_fields.append(f"{where}:{k}={v}")
                        walk(v, f"{where}/{k}")
                elif isinstance(node, list):
                    for i, v in enumerate(node):
                        walk(v, f"{where}[{i}]")

            walk(doc, str(path.relative_to(_REPO)).replace("\\", "/"))

    assert not numeric_anchor_fields, (
        "an artifact now carries a numeric anchor count: "
        f"{numeric_anchor_fields}. The MODEL labels in partition.rs and OP_COVERAGE.md were "
        "written because none existed; if one does now, read it instead of deriving it."
    )


def test_the_phi35_anchor_figure_is_derivable_from_the_claim_log():
    """161 is a recomputation over ``op`` and ``claimed``, and here is the recomputation.

    ``com.microsoft::MatMulNBits`` inputs 1 (``B``) and 2 (``scales``) are **required** by the
    contrib schema and are initializers by the format's definition, so every claimed node of
    that op anchors under the shipped rule without a fresh per-input census. Nothing else in
    this claim log can be established as anchoring from op names alone.
    """
    rows = _read_jsonl(_PHI35_CLAIM_LOG)
    claimed = [r for r in rows if r.get("claimed")]
    nbits = [r for r in claimed if r["op"] == "com.microsoft::MatMulNBits"]
    assert len(nbits) == 161, f"claim log now has {len(nbits)} claimed MatMulNBits nodes"

    text = _PARTITION_RS.read_text(encoding="utf-8")
    assert "anchors: 161," in text, "the Island constants no longer carry the derived figure"
    assert "anchors: 193" not in text, "the withdrawn name-only figure is back"


def test_the_anchor_figure_is_labelled_a_model_and_not_a_reading():
    """The number is allowed to exist. Calling it measured is not.

    Mutation guard: reinstating the old sentence — that nodes, *anchors* and FLOPs were read out
    of the verbose ``PartitionStats`` summary — turns this red.
    """
    text = _PARTITION_RS.read_text(encoding="utf-8")
    assert "MODEL, not a measured field" in text
    assert "`PartitionStats` has no anchor field" in text
    assert "nodes, anchors and FLOPs are unaffected" not in text, (
        "partition.rs again claims the anchor figure was read out of PartitionStats"
    )


def test_op_coverage_does_not_assert_an_unwitnessed_anchor_count():
    """``225 anchors`` had no witness of any kind and must not return."""
    text = _OP_COVERAGE.read_text(encoding="utf-8")
    assert "225 anchors" not in text
    for m in re.finditer(r"(\d+)\s+anchors\b", text):
        pytest.fail(
            f"docs/OP_COVERAGE.md asserts a bare anchor count {m.group(0)!r}; no artifact "
            "carries one, so it must be labelled MODEL and derived, or omitted"
        )


# ---------------------------------------------------------------------------------------
# 2. The BERT rank-2 runtime-B population
# ---------------------------------------------------------------------------------------


def test_the_rank2_runtime_b_matmul_population_is_exactly_one_on_the_cited_artifact():
    """One node, named, on a named subject — not "none exists" and not an aggregate.

    This is the population of ``MatMul`` nodes that this EP's kernel row would claim and that
    the issue #73 anchor rule declines to exempt: ``B`` is a fully-static rank 2 (so the kernel
    accepts it) and ``B`` is not an initializer (so it is not a resident weight).
    """
    doc = json.loads(_BERT_SHAPE_SPACE.read_text(encoding="utf-8"))
    assert doc["op"] == "MatMul"
    assert doc["model"].endswith("bertsquad-12.onnx"), doc["model"]
    assert doc["total"] == 98, doc["total"]

    runtime_rank2 = {
        k: v
        for k, v in doc["shape_classes"].items()
        if "rank_b=2" in k and "b_init=False" in k
    }
    assert sum(runtime_rank2.values()) == 1, runtime_rank2

    # And the node itself, by name, so the claim is checkable without re-deriving the class key.
    hits = [
        n
        for n in doc["nodes"]
        if n["b"] is not None and len(n["b"]) == 2 and not n["b_is_initializer"]
    ]
    assert len(hits) == 1, hits
    assert hits[0]["node"] == "MatMul"
    assert hits[0]["b"] == [768, 2]


def test_the_98_node_subject_is_not_combined_with_the_95_node_census():
    """Two subjects, two counts. Arithmetic across them would be arithmetic across two graphs.

    ``matmul_shape_space_bert.json`` walks a 98-node ``MatMul`` population; ``matmul.rs`` also
    cites a separately optimized 95-node census of the same model. The module doc must state the
    98 with its own subject and must warn against mixing them.

    Mutation guard: deleting the "Do not combine" paragraph, or restating the one runtime-``B``
    node as a fraction of 95, turns this red.
    """
    text = _MATMUL_RS.read_text(encoding="utf-8")
    assert "matmul_shape_space_bert.json" in text
    assert "`total: 98`" in text
    assert "Do not combine that count with the 95-node census" in text
    # Scoped to the module doc: the rest of the file legitimately discusses other 95-node
    # readings (e.g. rank inference on 94 of 95 nodes), which are a different subject again.
    module_doc = text[: text.index("\nuse ")] if "\nuse " in text else text
    for bad in ("1 of 95", "1/95", "one of BERT's 95"):
        assert bad not in module_doc, f"matmul.rs mixes the two subjects: {bad!r}"


def test_no_source_claims_the_runtime_b_population_is_empty():
    """The withdrawn claim, pinned by its shape rather than its wording.

    "No censused model contains one" was false against an artifact already in this repository.
    """
    for path in (_MATMUL_RS, _PARTITION_RS):
        text = path.read_text(encoding="utf-8").lower()
        for bad in (
            "no censused model contains",
            "b is an initializer on every graph we have censused, and a runtime",
            "no such node exists",
        ):
            assert bad not in text, f"{path.name} reinstates the withdrawn emptiness claim"


# ---------------------------------------------------------------------------------------
# 4. The production chain
# ---------------------------------------------------------------------------------------


_EP_RS = _REPO / "rust" / "src" / "ep.rs"


def test_the_island_builder_counts_anchors_by_weight_and_flops_by_family():
    """`ep.rs` must consult the weight oracle, and must *not* key FLOPs on the anchor rule.

    This is a source guard and says so. The behavioural guard is the Rust test
    ``new_anchor_semantics_never_newly_claim_over_the_production_chain``, which sweeps the
    shipped ``evaluate`` over islands built the way this function builds them — but it builds
    them in a test helper, so a rollback confined to ``ep.rs`` would leave it green. That is the
    gap this closes.

    Two separate properties:

    * ``anchors`` is incremented under ``is_anchor(&qual, weights)`` where ``weights`` comes from
      ``classify_weight_operand`` over ``has_input(i) && input_is_constant(i)``. Reverting to
      ``is_anchor(&qual)`` — the retired name-only call — turns this red.
    * ``flops`` is added under ``is_heavy_op_family(&qual)``. Keying it on ``is_anchor`` instead
      would silently drop ``2 * 3072 * 3072`` from every weightless heavy node and change every
      model's FLOP estimate, which is a different decision and must not ride along.
    """
    text = _EP_RS.read_text(encoding="utf-8")

    assert "partition::classify_weight_operand(&qual, |i| {" in text, (
        "ep.rs no longer derives a WeightOperand from the node view"
    )
    assert "view.has_input(i) && view.input_is_constant(i)" in text, (
        "ep.rs no longer asks whether the designated site holds a resident initializer"
    )
    assert "if partition::is_anchor(&qual, weights) {" in text, (
        "ep.rs no longer counts anchors by the weight-aware predicate"
    )
    assert "if partition::is_heavy_op_family(&qual) {" in text, (
        "ep.rs no longer keys the FLOP estimate on the heavy-family set"
    )
    # The retired one-argument form must not come back anywhere.
    assert not re.search(r"is_anchor\(&qual\)\s*\{", text), "ep.rs calls the name-only is_anchor"


def test_the_flop_estimate_is_identical_to_the_base_commit_arithmetic():
    """The two constants that make up the estimate are unchanged, in the same two branches.

    Issue #73 tightened *who anchors*. It must not have moved a single FLOP, because a FLOP
    change would confound any later partition measurement with this one.
    """
    text = _EP_RS.read_text(encoding="utf-8")
    assert "island.flops.saturating_add(2 * 3072 * 3072)" in text
    assert "island.flops.saturating_add(out_bytes / 2)" in text


# ---------------------------------------------------------------------------------------
# 5. LinearAttention
# ---------------------------------------------------------------------------------------


def test_linear_attention_designates_no_weight_site_in_the_source():
    """``decay`` and ``beta`` are B×T runtime gates, so the family designates no site.

    ORT v1.28.0 ``bert_defs.cc:2402-2431``: ``decay`` is ``(B, T, H_kv * d_k)`` or
    ``(B, T, H_kv)``; ``beta`` is ``(B, T, H_kv)`` or ``(B, T, 1)`` and is described as a
    "sigmoid output". Both carry batch and time extents. A learned parameter does not.

    Mutation guard: adding ``"com.microsoft::LinearAttention" => &[4, 5],`` to ``weight_sites``
    turns this red. The Rust unit test ``linear_attention_designates_no_weight_site`` is the
    behavioural half; this is the source half, and it is here because the two mutations differ
    (one changes behaviour, one changes only the cited justification).
    """
    text = _PARTITION_RS.read_text(encoding="utf-8")
    match = re.search(
        r'"com\.microsoft::LinearAttention"\s*=>\s*&\[([^\]]*)\]', text
    )
    assert match is None, (
        "weight_sites now designates a site for LinearAttention: "
        f"&[{match.group(1) if match else ''}]. The pinned schema shows only B×T runtime gates."
    )
    assert "B\u00d7T runtime gates" in text or "runtime gates" in text
    assert "sigmoid output" in text
