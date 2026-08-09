"""The anchor weight-site table is held to the pinned upstream schemas, and the guard is proven.

WHY THIS EXISTS
===============
Issue #73: `is_anchor` was a list of op names. A `MatMul` multiplying two *activations* — the two
batched matmuls at the heart of every attention block — matched the name and was granted the
anchor exemption, so six single-node islands in MiniLM were claimed by the very gate that exists
to reject them. The repair makes anchor status a property of the node: a heavy op carrying a
resident initializer at an input site the pinned schema declares to be a contracted parameter.

That repair moves the load onto a table. `rust/src/ops/partition.rs::ANCHOR_OP_SCHEMAS` enumerates
every declared input of every anchor-eligible op. A table like that goes wrong quietly, and it has
already gone wrong here in two distinct ways worth naming, because this lane exists to make both
loud:

1. **A trailing omission.** A 21-input `QMoE` was tabulated with 19 rows. Every structural check
   passed: indices 0..=18 were present, contiguous, correctly named. Contiguity cannot see a
   missing tail. The repair is an independently written `declared_inputs` total plus this lane's
   comparison against a machine extract of the real schema.
2. **A rule that did not generate its table.** The prose claimed the designated set was derived
   from declared extents. It was not — a 1D learned parameter was omitted while two sites with no
   declared extents were designated, and two sequence-indexed tables were designated while a
   genuine expert weight matrix was left out. The repair is that the roles are stated as an
   audited semantic reading, and that the audited set is pinned *by site name* in the extract so
   a silent re-judgement is a test failure rather than an editorial decision.

WHAT IS CHECKED, AND BY WHOM
============================
The comparison itself lives in `rust/tools/ort_weight_sites.py::check_table`. That is deliberate:
this file must not re-implement the rule it is testing, or a laxer copy in the test body would
pass while the shipped checker was broken. Every test below — the real check and every mutant —
calls the same production function.

`rust/tools/ort_weight_sites.json` holds the machine extract: the declared inputs of every op,
pulled from the ONNX Runtime contrib defs at the commit `third_party/onnxruntime/PROVENANCE.md`
pins (recorded with per-file sha256) and from the `onnx` package's own schema objects.

POSITIVE CONTROLS
=================
`test_*_is_caught` plants each historical failure and asserts the checker names the offending
site. A guard never seen failing is not a guard. The mutations are applied to in-memory copies of
the parsed table; nothing on disk is touched.
"""

from __future__ import annotations

import copy
import json
import pathlib
import re
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "rust" / "tools"))
import ort_weight_sites as ows  # noqa: E402

RUST_TABLE_PATH = REPO / "rust" / "src" / "ops" / "partition.rs"
EXTRACT_PATH = REPO / "rust" / "tools" / "ort_weight_sites.json"
DESIGN = REPO / "docs" / "DESIGN.md"
OP_COVERAGE = REPO / "docs" / "OP_COVERAGE.md"


@pytest.fixture(scope="module")
def extract() -> dict:
    return json.loads(EXTRACT_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rust_source() -> str:
    return RUST_TABLE_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def rust_table(rust_source: str) -> dict:
    return ows.parse_rust_table(rust_source)


# =============================================================================================
# The extract is real
# =============================================================================================


def test_the_extract_pins_its_upstream_sources_by_hash(extract: dict) -> None:
    """A table checked against an unpinned extract is checked against nothing."""
    prov = extract["provenance"]
    assert prov["onnxruntime_commit"] == "da9b5e364c465de65c49d91e696cd6485270757f"
    assert prov["onnxruntime_tag"] == "v1.28.0"
    assert re.fullmatch(r"\d+\.\d+\.\d+", prov["onnx_version"])
    assert set(prov["files"]) == {"bert_defs.cc", "contrib_defs.cc"}
    for name, meta in prov["files"].items():
        assert re.fullmatch(r"[0-9a-f]{64}", meta["sha256"]), name
        assert int(meta["bytes"]) > 0, name
        assert prov["onnxruntime_commit"] in meta["url"], name


def test_the_pinned_commit_is_the_one_the_repository_vendors() -> None:
    """The schema pin and the header pin must be the same commit or the table is off-version."""
    provenance = (REPO / "third_party" / "onnxruntime" / "PROVENANCE.md").read_text(
        encoding="utf-8"
    )
    assert ows.PINNED_ORT_COMMIT in provenance, (
        "rust/tools/ort_weight_sites.py pins an ONNX Runtime commit that "
        "third_party/onnxruntime/PROVENANCE.md does not name"
    )


def test_the_extract_covers_every_op_the_partitioner_can_anchor(extract: dict, rust_table: dict) -> None:
    assert set(extract["ops"]) == set(rust_table)
    assert len(extract["ops"]) == 11


def test_qmoe_has_twenty_one_declared_inputs_including_the_two_block_scales(extract: dict) -> None:
    """The exact shape whose truncation shipped once."""
    sites = extract["ops"]["com.microsoft::QMoE"]["sites"]
    assert len(sites) == 21
    assert [s["name"] for s in sites[-2:]] == [
        "fc1_act_block_scale",
        "fc2_act_block_scale",
    ]


def test_group_query_attention_has_sixteen_declared_inputs(extract: dict) -> None:
    sites = extract["ops"]["com.microsoft::GroupQueryAttention"]["sites"]
    assert len(sites) == 16
    assert sites[11]["name"] == "head_sink"
    assert [s["name"] for s in sites[7:9]] == ["cos_cache", "sin_cache"]
    assert [s["name"] for s in sites[12:14]] == ["k_scale", "v_scale"]


# =============================================================================================
# The shipped table matches — and the checker can tell when it does not
# =============================================================================================


def test_the_shipped_table_matches_the_pinned_schemas(rust_table: dict, extract: dict) -> None:
    findings = ows.check_table(rust_table, extract)
    assert findings == [], "\n".join(findings)


def test_the_parser_reads_the_shipped_table_and_not_a_guess(rust_table: dict) -> None:
    """If the parser silently found nothing, every check above would pass vacuously."""
    assert len(rust_table) == 11
    assert sum(len(v["sites"]) for v in rust_table.values()) == 84
    qmoe = rust_table["com.microsoft::QMoE"]
    assert qmoe["declared_inputs"] == 21
    assert len(qmoe["sites"]) == 21
    # Roles are resolved through the `use SiteRole::X as ALIAS` aliases, not read as aliases.
    roles = {s["role"] for v in rust_table.values() for s in v["sites"]}
    assert roles <= set(ows.load_extract()["roles"]), roles
    assert "ContractedParameter" in roles


def _mutate(table: dict) -> dict:
    return copy.deepcopy(table)


def test_a_trailing_omission_is_caught(rust_table: dict, extract: dict) -> None:
    """The historical defect, replanted: drop QMoE's last two sites and its total with them.

    Contiguity still holds after the truncation, which is exactly why the extract comparison and
    the independent `declared_inputs` total both exist.
    """
    mutant = _mutate(rust_table)
    qmoe = mutant["com.microsoft::QMoE"]
    qmoe["sites"] = qmoe["sites"][:19]
    qmoe["declared_inputs"] = 19
    findings = ows.check_table(mutant, extract)
    assert findings, "a 21-input schema tabulated with 19 rows must be caught"
    assert any("fc1_act_block_scale" in f for f in findings), findings
    assert any("fc2_act_block_scale" in f for f in findings), findings


def test_a_declared_total_that_lies_about_its_own_rows_is_caught(rust_table: dict, extract: dict) -> None:
    """Half the defect: the rows are dropped but the total is left honest — still caught."""
    mutant = _mutate(rust_table)
    mutant["com.microsoft::QMoE"]["sites"] = mutant["com.microsoft::QMoE"]["sites"][:19]
    findings = ows.check_table(mutant, extract)
    assert any("21" in f and "19" in f for f in findings), findings


def test_a_renamed_site_is_caught(rust_table: dict, extract: dict) -> None:
    mutant = _mutate(rust_table)
    mutant["com.microsoft::MatMulNBits"]["sites"][2]["name"] = "scale"
    findings = ows.check_table(mutant, extract)
    assert any("MatMulNBits[2]" in f and "scales" in f for f in findings), findings


def test_a_reordered_site_list_is_caught(rust_table: dict, extract: dict) -> None:
    """Same index set, same names, wrong order — the residency read is index-parallel."""
    mutant = _mutate(rust_table)
    sites = mutant["com.microsoft::GroupQueryAttention"]["sites"]
    sites[7], sites[8] = sites[8], sites[7]
    sites[7]["index"], sites[8]["index"] = 7, 8
    findings = ows.check_table(mutant, extract)
    assert any("GroupQueryAttention[7]" in f for f in findings), findings


def test_a_flipped_optionality_is_caught(rust_table: dict, extract: dict) -> None:
    mutant = _mutate(rust_table)
    mutant["Gemm"]["sites"][2]["optional"] = False
    findings = ows.check_table(mutant, extract)
    assert any("Gemm[2]" in f and "optional" in f for f in findings), findings


def test_a_phantom_site_is_caught(rust_table: dict, extract: dict) -> None:
    mutant = _mutate(rust_table)
    mm = mutant["MatMul"]
    mm["sites"].append({"index": 2, "name": "C", "optional": True, "role": "ElementwiseParameter"})
    mm["declared_inputs"] = 3
    findings = ows.check_table(mutant, extract)
    assert any("MatMul[2]" in f for f in findings), findings


def test_designating_a_learned_elementwise_parameter_is_caught(rust_table: dict, extract: dict) -> None:
    """`head_sink` is 1D `(num_heads)` and a real learned parameter — and still not an anchor.

    Designating it would make a GQA node with a resident `head_sink` claim the exemption, which is
    issue #73 through a smaller door: an `O(num_heads)` vector has nothing to amortise a host
    round trip against.
    """
    mutant = _mutate(rust_table)
    gqa = mutant["com.microsoft::GroupQueryAttention"]
    assert gqa["sites"][11]["name"] == "head_sink"
    gqa["sites"][11]["role"] = "ContractedParameter"
    findings = ows.check_table(mutant, extract)
    assert any("GroupQueryAttention" in f and "designates" in f for f in findings), findings
    assert any("head_sink" in f for f in findings), findings


def test_designating_a_rope_table_is_caught(rust_table: dict, extract: dict) -> None:
    mutant = _mutate(rust_table)
    mutant["com.microsoft::GroupQueryAttention"]["sites"][7]["role"] = "ContractedParameter"
    findings = ows.check_table(mutant, extract)
    assert any("cos_cache" in f for f in findings), findings


def test_designating_a_cache_scale_is_caught(rust_table: dict, extract: dict) -> None:
    mutant = _mutate(rust_table)
    mutant["com.microsoft::GroupQueryAttention"]["sites"][12]["role"] = "ContractedParameter"
    findings = ows.check_table(mutant, extract)
    assert any("k_scale" in f for f in findings), findings


def test_undesignating_an_expert_weight_matrix_is_caught(rust_table: dict, extract: dict) -> None:
    """The other half of the historical D2 finding: `fc3_experts_weights` was omitted."""
    mutant = _mutate(rust_table)
    qmoe = mutant["com.microsoft::QMoE"]
    assert qmoe["sites"][8]["name"] == "fc3_experts_weights"
    qmoe["sites"][8]["role"] = "QuantisationCompanion"
    findings = ows.check_table(mutant, extract)
    assert any("fc3_experts_weights" in f for f in findings), findings


def test_designating_an_activation_is_caught(rust_table: dict, extract: dict) -> None:
    """No activation may ever be designated. This is the whole of issue #73."""
    mutant = _mutate(rust_table)
    mutant["com.microsoft::MatMulNBits"]["sites"][0]["role"] = "ContractedParameter"
    findings = ows.check_table(mutant, extract)
    assert any("MatMulNBits" in f and "designates" in f for f in findings), findings


def test_an_unknown_role_is_caught(rust_table: dict, extract: dict) -> None:
    mutant = _mutate(rust_table)
    mutant["Conv"]["sites"][1]["role"] = "WeightIsh"
    findings = ows.check_table(mutant, extract)
    assert any("unknown role" in f for f in findings), findings


def test_a_dropped_op_row_is_caught(rust_table: dict, extract: dict) -> None:
    mutant = _mutate(rust_table)
    del mutant["com.microsoft::LinearAttention"]
    findings = ows.check_table(mutant, extract)
    assert any("LinearAttention" in f and "absent" in f for f in findings), findings


def test_an_unaudited_op_row_is_caught(rust_table: dict, extract: dict) -> None:
    mutant = _mutate(rust_table)
    mutant["com.microsoft::Invented"] = {
        "source": "nowhere",
        "declared_inputs": 1,
        "sites_const": "X",
        "sites": [{"index": 0, "name": "x", "optional": False, "role": "ContractedParameter"}],
    }
    findings = ows.check_table(mutant, extract)
    assert any("Invented" in f for f in findings), findings


# =============================================================================================
# The Rust source says what the checker assumes it says
# =============================================================================================


def test_no_name_only_anchor_call_survives_anywhere_in_the_tree() -> None:
    """`is_anchor` must be impossible to call by name alone.

    The signature is the enforcement, so this is a belt-and-braces scan of every Rust source: a
    single-argument `is_anchor("Op")` would not compile, but a *new* name-only predicate wearing
    the same meaning would, and that is the shape issue #73 is about.
    """
    offenders = []
    for path in sorted((REPO / "rust" / "src").rglob("*.rs")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for m in re.finditer(r"is_anchor\(\s*(&?\w+|\"[^\"]*\")\s*\)", text):
            line = text[: m.start()].count("\n") + 1
            offenders.append(f"{path.relative_to(REPO).as_posix()}:{line}: {m.group(0)}")
    assert offenders == [], "name-only is_anchor call(s):\n" + "\n".join(offenders)


def test_the_flop_estimator_and_the_anchor_predicate_are_different_calls() -> None:
    """FLOP-neutrality is structural: `ep.rs` must charge FLOPs on `is_heavy_op`, not `is_anchor`.

    If the FLOP branch keyed on anchor status, `GroupQueryAttention` ceasing to be an anchor would
    have moved 32 Phi-3.5 nodes from the heavy rate to the elementwise rate and silently shifted
    every recorded `total_flops`.
    """
    ep = (REPO / "rust" / "src" / "ep.rs").read_text(encoding="utf-8")
    assert "partition::is_heavy_op(&qual)" in ep
    assert "partition::is_anchor(&qual, &resident_inputs)" in ep
    # The FLOP constant must sit under the heavy-op branch, not the anchor branch.
    heavy_at = ep.index("partition::is_heavy_op(&qual)")
    anchor_at = ep.index("partition::is_anchor(&qual, &resident_inputs)")
    flop_at = ep.index("2 * 3072 * 3072", heavy_at)
    assert heavy_at < flop_at
    assert anchor_at < heavy_at, "the anchor tally must be decided before the FLOP branch"
    assert "island.anchors += 1" in ep


def test_residency_is_read_as_present_and_constant(rust_source: str) -> None:
    """A present-but-not-constant input is traffic, not a weight. Both conjuncts must be there."""
    ep = (REPO / "rust" / "src" / "ep.rs").read_text(encoding="utf-8")
    assert "view.has_input(i) && view.input_is_constant(i)" in ep


def test_the_weight_operand_has_exactly_two_states(rust_source: str) -> None:
    """No `Unknown`: a third state is a place for a caller to guess."""
    m = re.search(r"pub enum WeightOperand \{(.*?)\n\}", rust_source, re.S)
    assert m, "WeightOperand not found"
    variants = re.findall(r"^\s{4}(\w+),", m.group(1), re.M)
    assert variants == ["Present", "Absent"], variants


def test_only_one_site_role_designates(rust_source: str) -> None:
    m = re.search(r"pub fn designates\(&self\) -> bool \{(.*?)\n    \}", rust_source, re.S)
    assert m, "SiteRole::designates not found"
    assert "matches!(self, SiteRole::ContractedParameter)" in m.group(1)


# =============================================================================================
# Documentation guards
# =============================================================================================


def test_design_section_5_5_exists_and_is_where_its_citations_point() -> None:
    """§5.5 was silently deleted once. It is normative and four live citations resolve to it."""
    design = DESIGN.read_text(encoding="utf-8")
    heading = "### 5.5 `Compile` — plan build and prepacking"
    assert heading in design, "DESIGN.md §5.5 heading is missing"
    body_at = design.index(heading) + len(heading)
    assert "For each fused subgraph, in order:" in design[body_at : body_at + 400], (
        "§5.5 must sit immediately before its enumeration"
    )
    # No normative 5.x section is orphaned: the numbering is contiguous through §5.6.
    numbers = re.findall(r"^### (5\.\d+) ", design, re.M)
    assert numbers == ["5.1", "5.2", "5.3", "5.4", "5.5", "5.6"], numbers


def test_every_design_section_5_5_citation_resolves() -> None:
    """Counted against the exact field it binds: citations that name **DESIGN.md's** §5.5.

    `OP_COVERAGE.md` has a §5.5 of its own (the dtype variant matrix) and `claim.rs:626` cites
    *that* one. Both headings exist; conflating them is how a citation count goes wrong.
    """
    design_citations = []
    op_coverage_citations = []
    for path in [DESIGN, OP_COVERAGE, REPO / "rust" / "src"]:
        files = sorted(path.rglob("*.rs")) if path.is_dir() else [path]
        for f in files:
            text = f.read_text(encoding="utf-8", errors="replace")
            for m in re.finditer(r"§5\.5", text):
                line = text[: m.start()].count("\n") + 1
                window = text[max(0, m.start() - 120) : m.start() + 40]
                rel = f.relative_to(REPO).as_posix()
                if "OP_COVERAGE.md" in window:
                    op_coverage_citations.append(f"{rel}:{line}")
                else:
                    design_citations.append(f"{rel}:{line}")

    design_text = DESIGN.read_text(encoding="utf-8")
    op_coverage_text = OP_COVERAGE.read_text(encoding="utf-8")
    assert "### 5.5 `Compile` — plan build and prepacking" in design_text
    assert "### 5.5 The dtype variant matrix" in op_coverage_text

    assert len(design_citations) == 4, design_citations
    assert len(op_coverage_citations) == 1, op_coverage_citations
    assert "rust/src/ops/common/claim.rs" in op_coverage_citations[0]


#: The one file permitted to hold a pre-#73 anchor count, and the retraction it must carry.
#:
#: Modelled on `test_counters_abi_singleton.py`'s permitted-file registry rather than on a
#: proximity window: a window is a guess about how far a caveat reaches, and the three Phi-3.5
#: constants sit further from their shared retraction than any window worth writing.
STALE_ANCHOR_COUNT_PERMITTED = {
    "rust/src/ops/partition.rs": "`anchors: 193` is a historical reading",
    # This file quotes the figure in its own prose and in the guard below.
    "tests/ops/test_weight_site_schema.py": "`anchors: 193` is a historical reading",
    # The decision note that tells teammates the figure is dead has to name the figure to do so,
    # and so does the file the Scribe merges it into.
    ".squad/decisions/inbox/niobe-schema-designated-weight-anchor.md": "pre-#73 reading",
    ".squad/decisions.md": "pre-#73 reading",
}

#: Directories holding captured run output, or a frozen dated record of a past round. Both are a
#: record of what was printed or decided on a date; rewriting either to match a later predicate is
#: falsifying evidence, not correcting a claim. `.squad/decisions-archive/` holds exactly the same
#: kind of frozen, dated record as `bench/results/` — Scribe-rotated history, not a live claim.
STALE_ANCHOR_COUNT_EXEMPT_DIRS = (
    "bench/results/",
    "ci/fixtures/",
    "evidence/",
    "third_party/",
    ".squad/decisions-archive/",
)

#: Numeric band that has carried a stale anchor total: `193` (161 `MatMulNBits` + 32
#: `GroupQueryAttention`, a pre-#73 reading) or the still-older `225`. A band, not the two bare
#: literals, so a one-digit typo/variant of either does not slip past the guard unnoticed.
_STALE_COUNT = r"(?:19[0-9]|22[0-9])"

#: Phrases that, found near a match, show the number is being disclosed as a family/heavy-op-node
#: count (or an explicitly labelled historical reading) rather than asserted as a live anchor
#: total. Any one of these defuses a match; none of them appeared near either defect this guard
#: was tightened to catch (see `test_the_stale_anchor_guard_catches_the_pr113_review_findings`).
_STALE_ANCHOR_COUNT_DISCLOSURE = re.compile(
    r"heavy-op-family|heavy op family|family[- ]node|non-anchor|used to read|"
    r"historical reading|pre-#73|never an anchor|not an anchor",
    re.I,
)

#: How far (characters) a disclosure phrase may sit from a match and still cover it. Wide enough
#: to span one markdown hard-wrap (this repo wraps prose near column 90) plus a short clause;
#: not wide enough to reach into an unrelated paragraph three sentences away.
_STALE_ANCHOR_COUNT_WINDOW = 220


def _stale_anchor_count_patterns() -> list[re.Pattern[str]]:
    """Every textual shape that treats a family-node count as an anchor total.

    Five shapes, not two. The original pair only matched the number glued directly onto the word
    "anchor(s)" (`anchors: 193`, `193 anchors`) — real PR #113 review findings show two ways past
    that: prose where the number and the word share a clause but are not adjacent (`32 / 193` —
    the declined anchors over the total anchors`), and a bare ratio that repeats the same figure
    with no word "anchor" anywhere nearby at all (`32/193` exactly`, relying on an equally
    uncaptioned earlier paragraph in the same document to supply the meaning — this is exactly how
    that second finding went stale without ever containing the word "anchor").
    """
    return [
        # "anchors: 193" / "anchors 193" / "anchors = 225" -- word immediately decorating number.
        re.compile(rf"anchors?[^A-Za-z0-9]{{0,4}}{_STALE_COUNT}\b", re.I),
        # "193 anchors" / "225 anchors" -- number immediately decorating word.
        re.compile(rf"\b{_STALE_COUNT}\s+anchors?\b", re.I),
        # number ... anchors, same clause (bounded by a sentence-ending period so a disclosure two
        # sentences later cannot retroactively excuse a live claim in this one).
        re.compile(rf"\b{_STALE_COUNT}\b[^.]{{0,60}}?\banchors?\b", re.I),
        # anchors ... number, same clause.
        re.compile(rf"\banchors?\b[^.]{{0,60}}?\b{_STALE_COUNT}\b", re.I),
        # The specific 32/193 (or 193/32) ratio. In this codebase that ratio has never meant
        # anything but "32 declined GroupQueryAttention nodes over 193 heavy-op-family nodes";
        # left uncaptioned it reads as an anchor fraction to anyone who has not memorised the
        # distinction, which is exactly what happened to the PR #113 predictions-table finding.
        re.compile(r"\b32\s*/\s*193\b|\b193\s*/\s*32\b"),
    ]


def test_no_stale_anchor_count_is_reasserted() -> None:
    """The 193- and 225-anchor figures were produced by the predicate this change replaces.

    `193 = 161 MatMulNBits + 32 GroupQueryAttention` is a **pre-#73 reading**: GQA designates no
    weight site, so its 32 nodes are not anchors under the shipped predicate, and the
    `MatMulNBits` term is not re-asserted either because that run never asked whether each node's
    `B` was resident.

    Such a figure may be *recorded as history* — `partition.rs` does exactly that, dated, labelled
    and pointing at all three constants that carry it. It may not appear anywhere else, in any
    tense, **unless the same passage discloses it as a family/heavy-op-node count** (see
    `_stale_anchor_count_patterns` and `_STALE_ANCHOR_COUNT_DISCLOSURE`). A permitted file must
    still carry its retraction, so deleting the label turns this red rather than turning the guard
    off.
    """
    patterns = _stale_anchor_count_patterns()
    suffixes = {".rs", ".py", ".md", ".json", ".toml", ".yml", ".yaml"}
    skip_dirs = {".git", "target", "__pycache__", ".venv", "node_modules"}

    offenders = []
    scanned = 0
    for path in sorted(REPO.rglob("*")):
        if not path.is_file() or path.suffix not in suffixes:
            continue
        rel = path.relative_to(REPO).as_posix()
        if any(part in skip_dirs for part in path.relative_to(REPO).parts):
            continue
        if rel.startswith(STALE_ANCHOR_COUNT_EXEMPT_DIRS):
            continue
        scanned += 1
        text = path.read_text(encoding="utf-8", errors="replace")
        hits = [(m, p) for p in patterns for m in p.finditer(text)]
        if not hits:
            continue
        if rel in STALE_ANCHOR_COUNT_PERMITTED:
            assert STALE_ANCHOR_COUNT_PERMITTED[rel] in text, (
                f"{rel} still states a pre-#73 anchor count but has lost its retraction"
            )
            continue
        for m, _ in hits:
            start = max(0, m.start() - _STALE_ANCHOR_COUNT_WINDOW)
            end = min(len(text), m.end() + _STALE_ANCHOR_COUNT_WINDOW)
            if _STALE_ANCHOR_COUNT_DISCLOSURE.search(text[start:end]):
                continue
            line = text[: m.start()].count("\n") + 1
            offenders.append(f"{rel}:{line}: {m.group(0)!r}")

    assert scanned > 100, f"the scan found only {scanned} files; it is not looking at the tree"
    assert offenders == [], "un-retracted anchor counts:\n" + "\n".join(offenders)


def test_the_stale_anchor_guard_is_not_vacuous(tmp_path, monkeypatch) -> None:
    """Positive control: the patterns above must actually match the shapes they forbid."""
    patterns = _stale_anchor_count_patterns()
    for planted in [
        "anchors: 193,",
        "anchors = 225",
        "Phi-3.5 has 193 anchors",
        "353 claimed, 1 island, 225 anchors",
        "anchors 193",
        # The two real PR #113 review findings: number and word share a clause but are not
        # adjacent, and a bare ratio with no word "anchor" anywhere near it.
        "16.58% is `32 / 193` \u2014 the declined anchors over the total anchors.",
        "it is `32/193` exactly",
    ]:
        assert any(p.search(planted) for p in patterns), planted
    for benign in [
        "anchors: 0",
        "0 anchors",
        "anchors: 161",
        "1930 anchors",
        # unrelated ratios sharing the numeric band must not become candidates on their own
        "195/195 device-resident",
        "0/195",
        "1 / 229",
    ]:
        assert not any(p.search(benign) for p in patterns), benign


def test_the_stale_anchor_guard_catches_the_pr113_review_findings() -> None:
    """Mutation evidence: the exact two PR #113 review-finding sentences must fail this guard.

    These are the two stale claims named in the PR #113 revision-N+1 review: `docs/OP_COVERAGE.md`
    described the FLOP estimator's `32/193` figure as "the declined anchors over the total
    anchors" (near the roofline-split analysis) and, separately, restated the same figure as
    exactly `32/193` in a predictions table with no word "anchor" anywhere nearby to explain it.
    Both treat `193` — a heavy-op-family node count (161 `MatMulNBits` + 32 `GroupQueryAttention`)
    — as an anchor total, which is exactly what issue #73 retired. This test proves the tightened
    guard rejects both sentences verbatim, and that the corrected replacement text actually
    shipped in `docs/OP_COVERAGE.md` passes.
    """
    patterns = _stale_anchor_count_patterns()

    def is_caught(text: str) -> bool:
        for p in patterns:
            for m in p.finditer(text):
                start = max(0, m.start() - _STALE_ANCHOR_COUNT_WINDOW)
                end = min(len(text), m.end() + _STALE_ANCHOR_COUNT_WINDOW)
                if not _STALE_ANCHOR_COUNT_DISCLOSURE.search(text[start:end]):
                    return True
        return False

    stale_finding_1 = (
        "16.58% at every context and does not move at all.** 16.58% is `32 / 193` \u2014 the "
        "declined anchors\nover the total anchors. The estimator scores every anchor with the "
        "constant `2 * 3072 * 3072` and\neverything else with `out_bytes / 2` under a substituted "
        "dim"
    )
    stale_finding_2 = (
        "| 5 | the EP estimator is flat in ctx | **held** \u2014 16.58% at every ctx, and it is "
        "`32/193` exactly |"
    )
    assert is_caught(stale_finding_1), "the guard regressed: it no longer catches finding 1"
    assert is_caught(stale_finding_2), "the guard regressed: it no longer catches finding 2"

    # The corrected text that actually ships must not itself trip the guard.
    op_coverage_text = OP_COVERAGE.read_text(encoding="utf-8")
    assert not is_caught(
        "16.58% at every context and does not move at all.** 16.58% is `32 / 193` \u2014 the 32 "
        "CPU-declined\n`GroupQueryAttention` nodes over **193 heavy-op-family nodes** (161 "
        "`MatMulNBits` + 32\n`GroupQueryAttention`), **never an anchor total**"
    )
    assert "declined anchors\nover the total anchors" not in op_coverage_text
    assert "the anchor ratio of claimed-to-total node counts" not in op_coverage_text
    assert (
        "32 CPU-declined\n`GroupQueryAttention` nodes over **193 heavy-op-family nodes**"
        in op_coverage_text
    )
    assert (
        "32 declined `GroupQueryAttention` over 193 heavy-op-family nodes scored by the FLOP "
        "estimator" in op_coverage_text
    )


def test_the_design_rule_and_the_shipped_table_agree(rust_table: dict) -> None:
    """DESIGN.md §5.4.2's table must name the same designated sites the code designates.

    A normative rule that does not generate the shipped table is the D2 blocker restated; this
    binds the prose to the artifact rather than to a reviewer's memory.
    """
    design = DESIGN.read_text(encoding="utf-8")
    assert "#### 5.4.2" in design, "the weight-site ruling is missing from DESIGN.md"
    section = design[design.index("#### 5.4.2") :]
    section = section[: section.index("\n### ")] if "\n### " in section else section

    for op, row in sorted(rust_table.items()):
        designated = [s["name"] for s in row["sites"] if s["role"] == "ContractedParameter"]
        short = op.split("::")[-1]
        assert re.search(rf"\b{re.escape(short)}\b", section), f"{op} is not in §5.4.2"
        for name in designated:
            assert f"`{name}`" in section, f"{op} designates {name}, which §5.4.2 does not name"

    # The zero-designating rows must be stated as such, since "audited and designates nothing" is
    # the fact that carries the GQA result.
    for op in [
        "com.microsoft::GroupQueryAttention",
        "com.microsoft::MultiHeadAttention",
        "com.microsoft::LinearAttention",
    ]:
        assert rust_table[op]["sites"], op
        assert not [s for s in rust_table[op]["sites"] if s["role"] == "ContractedParameter"], op
    assert "designates no site" in section or "designate no site" in section


def test_the_design_prose_does_not_claim_a_dimensional_derivation() -> None:
    """The D2 blocker was a rule that claimed to derive a table it did not derive.

    The §5.4.2 prose must say plainly that the roles are an audited reading. The check is written
    as a **required affirmative disclosure plus a ban on the affirmative claim**, not as a
    substring ban: a bare substring ban is tripped by the negation that satisfies it ("it is *not*
    mechanically derived"), which would make the correct prose fail and train the next author to
    delete the disclaimer.
    """
    design = DESIGN.read_text(encoding="utf-8")
    section = design[design.index("#### 5.4.2") :]
    section = section[: section.index("\n### ")] if "\n### " in section else section
    # Markdown emphasis and hard-wrapping are formatting, not content: a phrase must be findable
    # whether or not the author bolded it or a line break landed inside it.
    lowered = re.sub(r"\s+", " ", section.replace("*", "").replace("`", "")).lower()

    # The disclosure must be there, in words, not implied by absence.
    assert "audited semantic reading" in lowered, "§5.4.2 must state what the roles are"
    assert "not a dimensional test" in lowered, "§5.4.2 must say what the roles are not"

    # And the affirmative claim must not be. `(?<!not )` and friends keep the negated form legal.
    forbidden = re.compile(
        r"(?<!not )(?<!is not )(?<!are not )"
        r"(?:mechanically derived|derived from the declared extents|"
        r"follows from the declared shape)",
        re.I,
    )
    hits = []
    for m in forbidden.finditer(section):
        window = section[max(0, m.start() - 40) : m.start()]
        if re.search(r"\bnot\b\s*$|\bnot\b\s+\w+\s*$", window, re.I):
            continue
        line = design[: design.index("#### 5.4.2") + m.start()].count("\n") + 1
        hits.append(f"docs/DESIGN.md:{line}: {m.group(0)!r}")
    assert hits == [], "§5.4.2 claims a derivation it does not perform:\n" + "\n".join(hits)


def test_the_derivation_guard_is_not_vacuous() -> None:
    """Positive control: the affirmative form must be caught and the negated form must not."""
    forbidden = re.compile(
        r"(?<!not )(?<!is not )(?<!are not )"
        r"(?:mechanically derived|derived from the declared extents|"
        r"follows from the declared shape)",
        re.I,
    )
    assert forbidden.search("the designated set is mechanically derived from the schema")
    assert forbidden.search("each role is derived from the declared extents of its site")
    assert not forbidden.search("it is not mechanically derived from declared extents")


# =============================================================================================
# End to end
# =============================================================================================


def test_the_production_checker_runs_clean_from_its_own_entry_point(capsys) -> None:
    """The CLI the lane will call, exercised as the lane calls it."""
    rc = ows.main(["--check"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "OK: 11 ops, 84 declared input sites, 11 designated" in out


def test_the_onnx_rederivation_agrees_or_declares_itself_skipped(capsys) -> None:
    """A version mismatch is an environment fact, not a table defect — and must say so."""
    status, findings = ows.verify_onnx_rows(ows.load_extract())
    assert findings == [], findings
    assert status.startswith("verified") or status.startswith("skipped"), status
