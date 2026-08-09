"""The weight-site audit in `partition.rs` must agree with the schemas it says it read.

WHY THIS EXISTS
---------------
Issue #73's anchor rule turns on a table of *which input index of which op holds a weight*. A
previous revision of this work stated the admission rule dimensionally — "no batch or sequence
extent" — and shipped a table that rule does not generate: `GroupQueryAttention`'s rank-1
`head_sink` was omitted though that rule admits it, its extent-less `k_scale`/`v_scale` were
designated though that rule cannot evaluate them, its sequence-sized RoPE caches were designated
though that rule excludes them, and `QMoE`'s third expert weight matrix was missing entirely.

The repair was to stop claiming dimensional exactness and commit the audit itself:
`WEIGHT_SITE_AUDIT` holds one row per input per heavy family, with the extents the pinned schema
declares and the reason the site was admitted or excluded, and `weight_sites()` is read off it.
The Rust side pins `weight_sites() == the designated rows`. This file pins the part Rust cannot
check: that the **default-domain** rows say what the ONNX operator schemas actually say, and that
the table's internal structure holds.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It does not fetch the pinned ONNX Runtime source at test time. `com.microsoft` rows carry their
provenance as file-and-line-range strings in `SCHEMA_SOURCES`, and those were read at
`da9b5e364c465de65c49d91e696cd6485270757f` by hand; a test that downloaded them would be a
network dependency in the unit lane and would fail closed for the wrong reason. What this file
checks about the contrib rows is *structure* — completeness, contiguity, no activation
designated, provenance present — which is what an audit table can be wrong about silently.

Run::

    pytest tests/ops/test_anchor_weight_sites.py -v --no-header
"""

from __future__ import annotations

import pathlib
import re

import onnx.defs

REPO = pathlib.Path(__file__).resolve().parents[2]
PARTITION_RS = REPO / "rust" / "src" / "ops" / "partition.rs"

SITE_RE = re.compile(
    r'site\(\s*"(?P<op>[^"]*)"\s*,\s*(?P<index>\d+)\s*,\s*"(?P<name>[^"]*)"\s*,'
    r'\s*"(?P<shape>[^"]*)"\s*,\s*(?P<designated>true|false)\s*,\s*(?P<reason>\w+)\s*,?\s*\)',
    re.DOTALL,
)

#: Which ONNX opset each default-domain row was audited against. The EP's claimed opsets; the
#: input *names and order* of these four ops have been stable across all of them.
DEFAULT_DOMAIN_OPS = {
    "MatMul": ("A", "B"),
    "Gemm": ("A", "B", "C"),
    "Conv": ("X", "W", "B"),
    "ConvTranspose": ("X", "W", "B"),
}


def _source() -> str:
    return PARTITION_RS.read_text(encoding="utf-8")


def _audit() -> list[dict]:
    text = _source()
    body = text.split("pub const WEIGHT_SITE_AUDIT: &[SchemaInput] = &[", 1)[1].split("\n];", 1)[0]
    rows = [m.groupdict() for m in SITE_RE.finditer(body)]
    assert rows, "WEIGHT_SITE_AUDIT parsed empty — the audit is the derivation and must be readable"
    for r in rows:
        r["index"] = int(r["index"])
        r["designated"] = r["designated"] == "true"
    return rows


def _weight_sites() -> dict[str, list[int]]:
    """The shipped `weight_sites()` match arms, parsed."""
    text = _source()
    body = text.split("pub fn weight_sites(qualified_op: &str) -> &'static [usize] {", 1)[1]
    body = body.split("\n}", 1)[0]
    out: dict[str, list[int]] = {}
    for line in body.splitlines():
        m = re.match(r'\s*((?:"[^"]+"\s*\|\s*)*"[^"]+")\s*=>\s*&\[([0-9,\s]*)\]', line)
        if not m:
            continue
        indices = [int(x) for x in re.findall(r"\d+", m.group(2))]
        for op in re.findall(r'"([^"]+)"', m.group(1)):
            out[op] = indices
    assert out, "weight_sites() parsed empty"
    return out


def _heavy_families() -> list[str]:
    body = _source().split("pub const HEAVY_OP_FAMILIES: &[&str] = &[", 1)[1].split("];", 1)[0]
    return re.findall(r'"([^"]+)"', body)


def test_weight_sites_is_exactly_the_designated_audit_rows():
    """The same invariant the Rust test pins, checked from the other side of the source.

    Two independent parsers agreeing is worth more than one: the Rust test compiles the constants
    and could not catch a row that fails to parse, and this one cannot catch a row the compiler
    would reject. Between them the table cannot be both readable and inconsistent.
    """
    audit = _audit()
    sites = _weight_sites()
    for family in _heavy_families():
        designated = sorted(r["index"] for r in audit if r["op"] == family and r["designated"])
        assert sites.get(family, []) == designated, (
            f"{family}: weight_sites() says {sites.get(family, [])}, the audit says {designated}"
        )


def test_every_heavy_family_is_audited_contiguously_from_zero():
    audit = _audit()
    for family in _heavy_families():
        rows = [r for r in audit if r["op"] == family]
        assert rows, f"{family} has no audit rows, so its anchor behaviour is underived"
        assert [r["index"] for r in rows] == list(range(len(rows))), (
            f"{family}: audit indices {[r['index'] for r in rows]} are not the schema's own "
            f"declaration order contiguous from 0 — a missing input must be visible as a gap"
        )
        assert len({r["name"] for r in rows}) == len(rows), f"{family} has duplicate input names"


def test_the_default_domain_rows_match_the_installed_onnx_schemas():
    """Re-read from `onnx` rather than trusted: these four are the positional-rule families."""
    audit = _audit()
    for op, expected_names in DEFAULT_DOMAIN_OPS.items():
        schema = onnx.defs.get_schema(op)
        actual = tuple(i.name for i in schema.inputs)
        assert actual == expected_names, (
            f"ONNX {op} now declares inputs {actual}; the audit was written against "
            f"{expected_names}. The positional weight rule (input 1) depends on this order."
        )
        rows = sorted((r for r in audit if r["op"] == op), key=lambda r: r["index"])
        assert [r["name"] for r in rows] == list(expected_names), (
            f"{op}: audit names {[r['name'] for r in rows]} != schema names {list(expected_names)}"
        )
        designated = [r["index"] for r in rows if r["designated"]]
        assert designated == [1], (
            f"{op} designates {designated}; the positional rule designates exactly input 1, the "
            f"right-hand operand"
        )


def test_onnx_attention_designates_nothing_and_its_inputs_are_all_runtime():
    """The default-domain `Attention` is the ONNX-side statement of what issue #73 is about."""
    audit = _audit()
    rows = sorted((r for r in audit if r["op"] == "Attention"), key=lambda r: r["index"])
    assert rows, "ONNX Attention is not audited"
    assert not any(r["designated"] for r in rows), (
        "ONNX Attention consumes pre-projected Q/K/V and owns no weight matrix"
    )
    try:
        schema = onnx.defs.get_schema("Attention")
    except Exception:  # pragma: no cover - older onnx without Attention-23
        return
    actual = [i.name for i in schema.inputs]
    assert [r["name"] for r in rows] == actual, (
        f"ONNX Attention audit names {[r['name'] for r in rows]} != schema {actual}"
    )


def test_no_designated_site_reads_as_a_runtime_tensor():
    """The single thing this table must never do, checked lexically."""
    forbidden = (
        "query",
        "key",
        "value",
        "input",
        "mask",
        "past",
        "present",
        "cache",
        "seqlens",
        "position",
        "sequence",
        "bias",
        "router",
        "act_scale",
        "decay",
        "beta",
        "sink",
    )
    for row in _audit():
        if not row["designated"]:
            continue
        low = row["name"].lower()
        hit = [w for w in forbidden if w in low]
        assert not hit, (
            f"{row['op']} input {row['index']} `{row['name']}` is designated but reads as a "
            f"runtime tensor ({hit})"
        )


def test_the_four_audited_corrections_survive_in_the_source():
    """Seraph's D2 findings against the previous table, each pinned by name and shape text."""
    audit = {(r["op"], r["index"]): r for r in _audit()}
    gqa = "com.microsoft::GroupQueryAttention"

    head_sink = audit[(gqa, 11)]
    assert head_sink["name"] == "head_sink"
    assert head_sink["shape"] == "(num_heads)"
    assert not head_sink["designated"]
    assert head_sink["reason"] == "X_PER_CHANNEL"

    for index, name in ((12, "k_scale"), (13, "v_scale")):
        row = audit[(gqa, index)]
        assert row["name"] == name
        assert row["shape"] == "", "the pinned schema declares no extents for these"
        assert not row["designated"]
        assert row["reason"] == "X_CACHE_SCALE"

    for index, name in ((7, "cos_cache"), (8, "sin_cache")):
        row = audit[(gqa, index)]
        assert row["name"] == name
        assert "max_sequence_length" in row["shape"]
        assert not row["designated"]
        assert row["reason"] == "X_POSITIONAL_TABLE"

    fc3 = audit[("com.microsoft::QMoE", 8)]
    assert fc3["name"] == "fc3_experts_weights"
    assert fc3["designated"] and fc3["reason"] == "W_WEIGHT"
    assert _weight_sites()["com.microsoft::QMoE"] == [2, 3, 5, 6, 8, 9, 11, 12, 13]


def test_the_attention_families_designate_nothing():
    sites = _weight_sites()
    for op in (
        "Attention",
        "com.microsoft::MultiHeadAttention",
        "com.microsoft::GroupQueryAttention",
        "com.microsoft::LinearAttention",
    ):
        assert sites.get(op, []) == [], f"{op} designates {sites.get(op)}; it owns no weight matrix"


def test_every_family_carries_provenance_at_the_pinned_revision():
    body = _source().split("pub const SCHEMA_SOURCES: &[(&str, &str)] = &[", 1)[1].split("\n];", 1)[0]
    pairs = re.findall(r'\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,?\s*\)', body, flags=re.DOTALL)
    provenance = dict(pairs)
    families = _heavy_families()
    assert set(provenance) == set(families), (
        f"provenance covers {sorted(provenance)}, families are {sorted(families)}"
    )
    for family in families:
        if family.startswith("com.microsoft::"):
            src = provenance[family]
            assert "ORT@da9b5e3" in src, f"{family} provenance names no pinned revision: {src}"
            assert re.search(r"\.cc:\d+-\d+$", src), (
                f"{family} provenance carries no file:line range: {src}"
            )


EP_RS = REPO / "rust" / "src" / "ep.rs"


def _anchor_call_site() -> str:
    """The `is_anchor(...)` call in `ep.rs`'s island builder, source text."""
    text = EP_RS.read_text(encoding="utf-8")
    start = text.index("partition::is_anchor(")
    return text[start : start + 500]


def test_the_production_call_site_asks_for_residency_and_not_merely_presence():
    """The oracle must be `has_input(i) && input_is_constant(i)`, both conjuncts.

    Dropping `input_is_constant` compiles cleanly, passes every unit test that reasons about
    `classify_weight_operand` in isolation, and silently restores the defect: a node anchors as
    soon as *anything* is bound at a designated site, weight or activation. There is no Vulkan
    device and no ORT graph fixture in this lane to catch that behaviourally, so it is caught
    lexically and the limitation is stated rather than hidden.
    """
    call = _anchor_call_site()
    assert "view.has_input(i)" in call, f"the anchor oracle does not check input presence:\n{call}"
    assert "view.input_is_constant(i)" in call, (
        "the anchor oracle does not check that the input is a resident initializer, so a runtime "
        f"activation at a designated site would anchor the node:\n{call}"
    )
    assert "classify_weight_operand" in call, (
        "the call site must go through `classify_weight_operand` rather than re-deriving which "
        "indices to consult — a second copy of that reasoning at the call site is RAI-011 "
        f"reappearing inside the fix for its own sibling:\n{call}"
    )


def test_the_flop_estimate_keys_off_the_family_and_not_the_anchor_predicate():
    """FLOPs must stay bit-identical to main: the arithmetic did not change, the exemption did."""
    text = EP_RS.read_text(encoding="utf-8")
    assert "partition::is_heavy_op_family(&qual)" in text, (
        "`ep.rs` no longer branches its FLOP estimate on the op family. If the FLOP branch is "
        "gated on `is_anchor` instead, every FLOP-derived number in the repo moves — including "
        "OP_COVERAGE.md's 16.58% roofline split — for a change that was supposed to move none."
    )
    flop_branch = text[text.index("partition::is_heavy_op_family(&qual)") :][:400]
    assert "2 * 3072 * 3072" in flop_branch, (
        "the heavy-family FLOP constant has moved out from under the family branch"
    )


def test_the_rule_does_not_claim_dimensional_exactness():
    """The prose repair, guarded: the doc must not restate the rule the table does not follow."""
    text = _source()
    block = text.split("pub const WEIGHT_SITE_AUDIT", 1)[0]
    marker = "/// The per-input audit"
    assert marker in block, "the audit's doc comment is missing"
    block = block[block.index(marker) :]
    # Doc comments wrap, so the prose is normalised before phrases are looked for — otherwise the
    # guard would fire on a reflow, which is a false positive that gets guards deleted.
    flat = " ".join(line.lstrip().removeprefix("///").strip() for line in block.splitlines())
    flat = re.sub(r"\s+", " ", flat)
    lowered = flat.lower()
    assert "not a dimensional test" in lowered, (
        "the audit's doc comment must say plainly that the criterion is an audited semantic "
        "reading rather than a dimensional rule — the previous revision's defect was a sentence "
        "that promised exactness the table did not deliver"
    )
    assert "neither a batch nor a sequence dimension" in lowered, (
        "the withdrawn dimensional rule must be quoted where it is withdrawn, so a reader who "
        "remembers it can find out what replaced it"
    )
    for correction in ("head_sink", "k_scale", "cos_cache", "fc3_experts_weights"):
        assert correction in flat, (
            f"the doc comment does not name `{correction}`, one of the four rows the previous "
            f"table got wrong; a correction nobody can locate is not a correction"
        )
