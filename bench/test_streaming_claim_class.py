"""Locks `docs/PERF.md` §26.4.1's published figures to their class and to their derivation.

What this file is for, and why it is not part of `test_perf_claims.py`
---------------------------------------------------------------------
`test_perf_claims.py` checks that §26's *values* agree with committed artifacts. This file checks
something upstream of that: that every figure §26.4.1 publishes is **classified**, that the
classification is not something prose can move, and that a `MODEL` figure — a byte count computed
from geometry — is never able to read as a measurement.

That failure mode is specific and it has happened. A guard written as "the word MODEL must appear
near the number" is defeated three ways at once, all of which a careless edit produces by accident
and a careless reviewer waves through:

* delete the figure's own label and let an unrelated `**MODEL**` elsewhere in the same sentence
  satisfy the proximity window;
* introduce a figure the ledger never heard of — `(8, 4) measured 2.3 GB/s` — and have the scanner
  miss it because it splits clauses on `.` and the decimal point ends the clause before the unit;
* publish a bandwidth at all, in a subsection whose entire content is byte counts.

None of those are reachable here, because none of the mechanisms they exploit exist here. There is
no proximity window, no clause splitting, and no lexical class detection. Instead:

1. **The ledger is the only source of class.** A fenced ```claim-ledger``` block in §26.4.1 has one
   row per published figure: `id class value unit source`. Prose cannot add, remove, or reclassify
   a row.
2. **Every prose figure must cite a row id, syntactically bound to itself.** The required form is
   ``  `VALUE UNIT` (row.id)  `` — the citation is the token immediately after the figure's code
   span, with only whitespace between. Any number carrying a performance unit in this subsection
   that is not written that way fails, decimals included.
3. **Every `MODEL` row is recomputed here**, from the named-bytes formula and the committed
   SPIR-V-walk witness, and must match to the digit. Nothing is transcribed.
4. **Every `MEASURED` row must name a committed artifact and a JSON pointer**, and the value at
   that pointer must be the published one. There are currently no `MEASURED` rows; the path is
   implemented so that adding one is mechanical rather than a matter of prose.

Frame, stated because a guard whose scope is unstated is a guard nobody can audit: this file rules
on **§26.4.1 only**. It is the subsection issue #81 authored and the only one that carries a claim
ledger. The rest of §26 is checked by `test_perf_claims.py` against its own artifacts.

No GPU, no EP, no model file, no network.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

_BENCH = Path(__file__).resolve().parent
_REPO = _BENCH.parent
PERF = _REPO / "docs" / "PERF.md"
RESULTS = _BENCH / "results"

#: The subsection this file rules on. Stated once.
SECTION = "26.4.1"

#: The witness the MODEL rows are recomputed against.
WITNESS = "weight_reread_phi35.json"

#: Units that make a number a performance figure rather than a count of something else.
#:
#: `B`, `GB` and `%` are here because §26.4.1's whole content is byte counts and their ratio; the
#: time and rate units are here because publishing one in this subsection is exactly the mistake
#: being guarded against, and an unregisterable figure has to fail rather than pass unnoticed.
PERF_UNITS = ("GB/s", "MB/s", "TB/s", "GB", "MB", "KB", "B", "ms", "us", "s", "%", "x", "×")

#: The Phi-3.5 projection shape every MODEL row is derived at.
N = 3072
K = 3072
BITS = 4
A_BYTES = 2
#: Column tile the selector picks for this shape; `wg = 32` at `K/32 = 96` blocks.
SELECTED_COLS = 16
SELECTED_ROWS = 2


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------

def _perf_text() -> str:
    if not PERF.is_file():
        pytest.skip("docs/PERF.md is not present in this tree")
    return PERF.read_text(encoding="utf-8")


def _subsection(text: str, number: str) -> str:
    """The body of `### <number> ...` up to the next heading of the same or higher level.

    Structural: it keys on the heading syntax, not on any word in the prose. Fenced blocks are
    tracked so that a `#` comment inside one — the claim ledger's own header line is exactly that —
    cannot be mistaken for a heading and truncate the section.
    """
    start = re.search(rf"^#{{2,4}}\s+{re.escape(number)}\s", text, re.M)
    assert start, f"docs/PERF.md has no §{number} heading"
    level = len(text[start.start():start.end()].split()[0])
    body: list[str] = []
    fenced = False
    for line in text[start.end():].splitlines(keepends=True):
        if line.startswith("```"):
            fenced = not fenced
        elif not fenced and re.match(rf"#{{1,{level}}}\s", line):
            break
        body.append(line)
    return "".join(body)


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------

class Row:
    __slots__ = ("id", "cls", "value", "unit", "source")

    def __init__(self, rid: str, cls: str, value: str, unit: str, source: str):
        self.id = rid
        self.cls = cls
        self.value = value
        self.unit = unit
        self.source = source

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"Row({self.id!r}, {self.cls!r}, {self.value!r}, {self.unit!r})"

    @property
    def number(self) -> float:
        return float(self.value)


def _ledger(section: str) -> dict[str, Row]:
    blocks = re.findall(r"^```claim-ledger\n(.*?)^```", section, re.M | re.S)
    assert len(blocks) == 1, (
        f"§{SECTION} must carry exactly one ```claim-ledger``` block, found {len(blocks)}"
    )
    rows: dict[str, Row] = {}
    for line in blocks[0].splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        assert len(parts) == 5, f"ledger row must be `id class value unit source`: {line!r}"
        rid, cls, value, unit, source = parts
        assert cls in ("MODEL", "MEASURED"), f"unknown class {cls!r} in {line!r}"
        assert rid not in rows, f"duplicate ledger id {rid!r}"
        assert unit in PERF_UNITS, f"ledger row {rid!r} uses unit {unit!r} the scanner cannot see"
        float(value)  # a non-numeric value is a malformed row, not a passing one
        rows[rid] = Row(rid, cls, value, unit, source)
    assert rows, "the claim ledger is empty"
    return rows


# ---------------------------------------------------------------------------
# The model — an independent implementation of the named-bytes formula
# ---------------------------------------------------------------------------

def named_bytes(m: int, n: int, k: int, bits: int, a_bytes: int, cols: int, rows: int) -> int:
    """Weight + activation bytes the dispatch grid names, for one node.

    Written from the geometry rather than ported: `ceil(m/rows) * ceil(n/cols)` tiles, each naming
    `cols * k * bits / 8` weight bytes and `rows * k * a_bytes` activation bytes.
    """
    row_tiles = -(-m // rows)
    col_tiles = -(-n // cols)
    weight = row_tiles * col_tiles * cols * k * bits // 8
    activation = row_tiles * col_tiles * rows * k * a_bytes
    return weight + activation


def _witness() -> dict:
    path = RESULTS / WITNESS
    if not path.is_file():
        pytest.skip(f"{WITNESS} is not committed in this tree")
    return json.loads(path.read_text(encoding="utf-8"))


def _one_pass_bytes() -> int:
    return int(_witness()["denominator"]["int4_weight_bytes_from_graph"])


def _model_value(source: str) -> float:
    """Recompute a MODEL row from its `source` expression.

    The grammar is deliberately tiny and total: anything it does not recognise is a failure, so a
    row cannot be added with a hand-written number and an unrecognised provenance string.
    """
    if source.startswith("weight_reread_phi35.json#"):
        pointer = source.split("#", 1)[1]
        node = _witness()
        for part in [p for p in pointer.split("/") if p]:
            node = node[part]
        return float(node)

    m = re.fullmatch(r"formula:(tiled|untiled)\((\d+)\)", source)
    if m:
        which, width = m.group(1), int(m.group(2))
        passes = -(-width // SELECTED_ROWS) if which == "tiled" else width
        return passes * _one_pass_bytes()

    m = re.fullmatch(r"formula:removed_share\((\d+)\)", source)
    if m:
        width = int(m.group(1))
        tiled = -(-width // SELECTED_ROWS)
        return 100.0 * (width - tiled) / width

    m = re.fullmatch(r"formula:named\((\d+),(\d+),(\d+)\)", source)
    if m:
        width, cols, rows = (int(g) for g in m.groups())
        return float(named_bytes(width, N, K, BITS, A_BYTES, cols, rows))

    raise AssertionError(f"unrecognised MODEL provenance {source!r}")


#: How a published value is scaled to the unit it is published in.
_UNIT_SCALE = {"B": 1.0, "KB": 1e3, "MB": 1e6, "GB": 1e9, "%": 1.0}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_every_model_row_recomputes_from_the_formula_and_the_witness():
    """A MODEL figure is a computation, so it is computed here rather than transcribed."""
    section = _subsection(_perf_text(), SECTION)
    rows = _ledger(section)
    modelled = [r for r in rows.values() if r.cls == "MODEL"]
    assert modelled, "the ledger classifies nothing as MODEL, which cannot be right here"
    for row in modelled:
        computed = _model_value(row.source)
        scale = _UNIT_SCALE.get(row.unit)
        assert scale is not None, f"{row.id}: unit {row.unit!r} has no scale for a MODEL row"
        published = row.number * scale
        # Published figures are rounded to the digits they show. Compare at the published
        # precision rather than at full precision, and derive that precision from the text of the
        # value itself so a row cannot loosen its own tolerance without visibly losing digits.
        decimals = len(row.value.split(".")[1]) if "." in row.value else 0
        tolerance = 0.5 * scale * (10 ** -decimals)
        assert abs(published - computed) <= tolerance, (
            f"{row.id}: §{SECTION} publishes {row.value} {row.unit} "
            f"({published:.6g} in base units); {row.source} computes {computed:.6g}"
        )


def test_the_witness_agrees_that_the_tiled_arm_makes_ceil_m_over_two_passes():
    """The pass count the subsection is built on, read off the committed artifact.

    This is the assertion that would have caught the withdrawn `ceil(M/4)`. It does not consult
    the prose at all — it consults the SPIR-V walk.
    """
    doc = _witness()
    rows = [r for r in doc.get("by_shape_prefill", []) if r.get("N") == "ALL SHAPES"]
    assert rows, f"{WITNESS} carries no ALL SHAPES prefill rows"
    one_pass = _one_pass_bytes()
    for row in rows:
        m = int(row["m_total"])
        assert row["tiled_amplification"] == -(-m // SELECTED_ROWS), (
            f"M={m}: the witness reports tiled amplification {row['tiled_amplification']}, "
            f"not ceil(M/{SELECTED_ROWS})"
        )
        assert row["untiled_amplification"] == m, f"M={m}: untiled must be M passes"
        assert row["tiled_named_bytes"] == -(-m // SELECTED_ROWS) * one_pass
        assert row["untiled_named_bytes"] == m * one_pass


def test_the_16x2_and_8x4_pair_ties_exactly_on_the_derived_residues():
    """The tie condition, derived here independently and exhausted.

    `total(cols, rows) = ceil(M/rows) * (W + rows * (N/cols) * K * a_bytes)` with
    `W = N*K*bits/8`. Halving `cols` doubles the activation term, so the bracket ratio is 2 exactly
    when `bits == 2 * a_bytes`; given that, the tie is `ceil(M/2) == 2*ceil(M/4)`, i.e.
    `M mod 4 in {0, 3}`.
    """
    for m in range(2, 4097):
        wide = named_bytes(m, N, K, BITS, A_BYTES, 16, 2)
        deep = named_bytes(m, N, K, BITS, A_BYTES, 8, 4)
        assert (wide == deep) is (m % 4 in (0, 3)), f"M={m}: {wide} vs {deep}"
        if wide != deep:
            assert wide < deep, f"M={m}: the wide tile must never name more bytes"
    # M=2 is a factor of two apart in a stated direction, not a tie.
    assert named_bytes(2, N, K, BITS, A_BYTES, 8, 4) == 2 * named_bytes(
        2, N, K, BITS, A_BYTES, 16, 2
    )
    # The ratio condition, not just the residue condition: fp32 activations never tie.
    for m in range(2, 513):
        assert named_bytes(m, N, K, BITS, 4, 16, 2) != named_bytes(m, N, K, BITS, 4, 8, 4)


def test_every_published_figure_is_bound_to_a_ledger_row_by_syntax():
    """No unregistered performance figure may appear in §26.4.1, in any spelling.

    The scan is over the raw text with prose stripped of nothing. A figure is recognised as
    `` `VALUE UNIT` `` in a code span; the binding is `(row.id)` immediately after it, separated by
    whitespace only. There is no window, no sentence, and no clause: the citation either is the
    next token or it is not.
    """
    section = _subsection(_perf_text(), SECTION)
    rows = _ledger(section)
    # The ledger block itself carries the same numbers by construction; excluding it is not a
    # loophole because its rows are what everything else is checked against.
    prose = re.sub(r"^```claim-ledger\n.*?^```", "", section, flags=re.M | re.S)

    units = "|".join(re.escape(u) for u in sorted(PERF_UNITS, key=len, reverse=True))
    figure = re.compile(rf"(?<![\w.])(\d+(?:\.\d+)?)\s*({units})(?![\w/])")
    bound = re.compile(
        rf"`\s*(\d+(?:\.\d+)?)\s*({units})\s*`\s*\(`?([a-z][a-z0-9_.]*)`?\)"
    )

    cited: dict[tuple[int, int], str] = {}
    for m in bound.finditer(prose):
        cited[(m.start(1), m.end(2))] = m.group(3)

    unbound = []
    for m in figure.finditer(prose):
        rid = cited.get((m.start(1), m.end(2)))
        if rid is None:
            line = prose[: m.start()].count("\n") + 1
            unbound.append(f"{m.group(0)!r} (line {line} of §{SECTION})")
            continue
        assert rid in rows, (
            f"§{SECTION} publishes {m.group(0)!r} citing {rid!r}, which the ledger does not list"
        )
        row = rows[rid]
        assert row.unit == m.group(2), (
            f"§{SECTION} publishes {m.group(0)!r} citing {rid!r}, whose unit is {row.unit!r}"
        )
        assert math.isclose(float(m.group(1)), row.number, rel_tol=0, abs_tol=1e-9), (
            f"§{SECTION} publishes {m.group(0)!r} citing {rid!r}, whose value is {row.value}"
        )

    assert not unbound, (
        f"§{SECTION} publishes performance figures with no ledger citation: {unbound}. "
        f"Every figure must be written as `VALUE UNIT` (row.id)."
    )


def test_no_prose_word_can_change_how_a_figure_is_classified():
    """The anti-decoy property, asserted as a property of this checker rather than of the text.

    Class is read from the ledger's `class` column and from nowhere else. This test proves that by
    construction: it runs the classifier over a synthetic section whose prose says the opposite of
    its ledger, and checks the ledger wins.
    """
    synthetic = (
        "#### x\n"
        "This paragraph is emphatically **MEASURED** and was measured on real hardware, "
        "measured **MODEL** measured.\n"
        "```claim-ledger\n"
        "# id            class   value  unit  source\n"
        "model.one_pass_bytes MODEL 1861189632 B weight_reread_phi35.json#/denominator/int4_weight_bytes_from_graph\n"
        "```\n"
    )
    rows = _ledger(synthetic)
    assert rows["model.one_pass_bytes"].cls == "MODEL", (
        "prose changed the class, which is the exact defect this checker exists to prevent"
    )
    # And the reverse: a MEASURED row stays MEASURED however loudly the prose says MODEL.
    flipped = synthetic.replace(" MODEL 1861189632", " MEASURED 1861189632")
    assert _ledger(flipped)["model.one_pass_bytes"].cls == "MEASURED"


def test_every_measured_row_names_an_artifact_and_a_pointer_that_carries_its_value():
    """The path a real measurement would take, implemented and enforced now.

    There are no MEASURED rows today and the subsection says so. Implementing the check before it
    is needed is the point: the first person to publish a measurement gets a mechanical
    requirement, not a judgement call about whether prose is enough.
    """
    section = _subsection(_perf_text(), SECTION)
    rows = _ledger(section)
    measured = [r for r in rows.values() if r.cls == "MEASURED"]
    for row in measured:
        assert "#" in row.source, (
            f"{row.id}: a MEASURED row must cite `artifact.json#/json/pointer`, got {row.source!r}"
        )
        name, pointer = row.source.split("#", 1)
        path = RESULTS / name
        assert path.is_file(), f"{row.id}: cites {name}, which is not committed"
        node = json.loads(path.read_text(encoding="utf-8"))
        for part in [p for p in pointer.split("/") if p]:
            node = node[part]
        scale = _UNIT_SCALE.get(row.unit, 1.0)
        assert math.isclose(float(node) , row.number * scale, rel_tol=1e-3), (
            f"{row.id}: publishes {row.value} {row.unit}; {row.source} carries {node}"
        )
    if not measured:
        assert "no `MEASURED` row" in section, (
            "the subsection has no MEASURED rows and must say so, so that a reader is not left "
            "to infer that a byte count was timed"
        )


def test_the_withdrawn_figures_are_gone_from_the_section_that_carried_them():
    """The withdrawn conclusions must not survive anywhere in §26.4.

    They were derived from `ceil(M/4)` and no replacement has been measured, so their reappearance
    — in any of the spellings the draft used — is a regression rather than an edit.
    """
    section = _subsection(_perf_text(), "26.4")
    for withdrawn in ("199.7", "244.5", "242.8", "238.8", "222.5", "217.4", "323 ms", "9.4–11.5"):
        assert withdrawn not in section, (
            f"§26.4 still carries the withdrawn figure {withdrawn!r}"
        )
    assert "ceil(M/4)" not in section or "wrong pass count" in section, (
        "§26.4 must not assert ceil(M/4) except when describing the correction"
    )
