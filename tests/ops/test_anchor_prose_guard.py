"""Documentation & prose guard for the issue #73 anchor model.

WHY THIS FILE EXISTS
====================
The Rust checker (``test_anchor_weight_sites_schema.py``) proves the *code* anchors on a
weight-site residency fact, not an op name. But the defect issue #73 fixes has a second home:
**prose**. A design doc that says "193 anchors", a comment that calls `32/193` an "anchor ratio",
or a sentence that treats `GroupQueryAttention` as an anchor re-teaches the wrong model to the
next reader even when the code is correct. This guard reads the shipped docs, comments and tests
and fails on any of those framings.

The correct model, which this guard enforces by exclusion:
  * A node **anchors** only when it is heavy-family AND carries a resident initializer at a
    schema-designated weight site.
  * ``GroupQueryAttention`` is heavy-family but designates **no** weight site → never anchors.
  * Phi-3.5: **161 ``MatMulNBits`` anchors** plus **32 non-anchor ``GroupQueryAttention`` nodes**.
  * **193 is the heavy-family node count — never an anchor count and never an anchor ratio.**

It also asserts the structural guarantee at the source level: ``is_anchor`` takes residency facts
(``resident_inputs: &[bool]``), so a name-only anchor call cannot even be written.

No GPU, no model, no EP. Always in the lane.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]

# Files whose *prose* is normative about the anchor model. Kept explicit so the guard scans the
# live documents rather than transient scratch under bench/ or third_party vendored trees.
_SCAN = [
    _REPO / "docs" / "DESIGN.md",
    _REPO / "docs" / "OP_COVERAGE.md",
    _REPO / "rust" / "src" / "ops" / "partition.rs",
    _REPO / "rust" / "src" / "ops" / "matmul.rs",
    _REPO / "rust" / "src" / "ep.rs",
]

# A line matching one of these has stated the forbidden framing, UNLESS it also carries an explicit
# negation (so corrected prose that says "193 is NOT an anchor count" is allowed to name the error
# it is correcting).
_FORBIDDEN = [
    (re.compile(r"\b193\b[^\n]{0,45}\banchor", re.I), "193 presented as an anchor quantity"),
    (re.compile(r"\banchor[^\n]{0,45}\b193\b", re.I), "an anchor quantity equated with 193"),
    (re.compile(r"\b225\b[^\n]{0,25}\banchor", re.I), "225 presented as an anchor count"),
    (re.compile(r"\b32\s*/\s*193\b[^\n]{0,70}\banchor", re.I), "32/193 framed as an anchor ratio"),
    (re.compile(r"\banchor[^\n]{0,70}\b32\s*/\s*193\b", re.I), "32/193 framed as an anchor ratio"),
    (re.compile(r"total\s+anchors", re.I), "'total anchors' treats heavy-family count as anchors"),
    (re.compile(r"declined\s+anchors", re.I), "'declined anchors' framing (GQA is non-anchor)"),
    (re.compile(r"\bGQA\b[^\n]{0,20}\bis\s+an?\s+anchor", re.I), "GQA described as an anchor"),
    (re.compile(r"GroupQueryAttention[^\n]{0,20}\bis\s+an?\s+anchor", re.I), "GQA as an anchor"),
    (re.compile(r"(?:GQA|GroupQueryAttention)[^\n]{0,20}anchors\s+(?:its|an|the)", re.I),
     "GQA described as anchoring an island"),
    (re.compile(r"containing\s+MatMulNBits\s+or\s+GQA\s+is\s+always\s+claimed", re.I),
     "island exempted merely for containing GQA"),
]

_NEGATION = re.compile(
    r"not\s+an?\s+anchor|never\s+an?\s+anchor|non-anchor|isn't\s+an\s+anchor|"
    r"not\s+the\s+anchor|no\s+longer[^\n]{0,20}anchor",
    re.I,
)


def _iter_lines():
    for path in _SCAN:
        if not path.exists():
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            yield path, n, line


def test_no_forbidden_anchor_framing():
    violations = []
    for path, n, line in _iter_lines():
        if _NEGATION.search(line):
            continue
        for rx, why in _FORBIDDEN:
            if rx.search(line):
                violations.append(f"{path.name}:{n}: {why}\n    {line.strip()[:160]}")
    assert not violations, (
        "forbidden anchor framing found (193/225 as an anchor count, 32/193 as an anchor ratio, "
        "or GQA treated as an anchor):\n" + "\n".join(violations)
    )


def test_is_anchor_takes_residency_facts_not_a_name():
    """Structural #73 guarantee: a name-only anchor decision does not typecheck."""
    part = (_REPO / "rust" / "src" / "ops" / "partition.rs").read_text(encoding="utf-8")
    sig = re.search(r"pub fn is_anchor\(([^)]*)\)", part)
    assert sig, "is_anchor not found in partition.rs"
    params = sig.group(1)
    assert "resident_inputs" in params and "&[bool]" in params, (
        f"is_anchor must take `resident_inputs: &[bool]`; got is_anchor({params})"
    )
    # No call site may invoke is_anchor with only the op name.
    for rs in (_REPO / "rust" / "src").rglob("*.rs"):
        text = rs.read_text(encoding="utf-8")
        for m in re.finditer(r"is_anchor\(\s*&?\s*[A-Za-z_][A-Za-z0-9_.]*\s*\)", text):
            raise AssertionError(f"name-only is_anchor call in {rs.name}: {m.group(0)}")


def test_phi35_framing_is_present_and_correct():
    """The corrected 161-anchor / 32-non-anchor / 193-heavy-family framing must be documented."""
    design = (_REPO / "docs" / "DESIGN.md").read_text(encoding="utf-8")
    cov = (_REPO / "docs" / "OP_COVERAGE.md").read_text(encoding="utf-8")
    blob = design + "\n" + cov
    assert re.search(r"161\s+`?MatMulNBits`?\s+anchor", blob, re.I), (
        "docs must state Phi-3.5 has 161 MatMulNBits anchors"
    )
    assert re.search(r"non-anchor\b[^\n]{0,40}GroupQueryAttention|"
                     r"GroupQueryAttention[^\n]{0,40}non-anchor|"
                     r"32\s+non-anchor", blob, re.I), (
        "docs must state the 32 GroupQueryAttention nodes are non-anchor"
    )
    assert re.search(r"193[^\n]{0,60}not an anchor count", blob, re.I), (
        "docs must state 193 is the heavy-family node count, not an anchor count"
    )
