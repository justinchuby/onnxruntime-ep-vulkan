"""Locks issue #81's withdrawn-streaming-claim narrative in `docs/PERF.md` (S25.3/S26.4) to a
per-claim label, not a wide-window substring search.

PR #92's `bench/test_withdrawn_streaming_claim.py` (rejected, and **not present on `main`** --
this file is a fresh build, not a copy of it) asserted only that the bare substring `"MODEL"`
occurred somewhere within +/-700 characters of an `M = 128` anchor. That window was wide enough
to also catch the section's own nearby glossary prose (the word "MODEL" used generically, twice,
elsewhere in the same neighbourhood) regardless of whether the *specific* claim next to the
number the window was meant to guard was itself labelled correctly, and a planted mutation that
swapped that claim's own MODEL label for a measured/witnessed one did not turn the test red
(Niobe, PR #92 review, comment 5229765153 -- issue #81 finding B2). The docstring claimed mutation
coverage; the mutation did not turn red.

This file binds each of the two headline byte figures issue #81's B3 finding corrected -- the
real, artifact-witnessed `(16, 2)` weight-byte reading at M=4, and the MODEL-only `(8, 4)` figure
at the same M -- to the single clause (bounded by the `.`/`;` the prose itself already uses to
separate its two half-sentences) that contains it, never a fixed character radius around an
unrelated anchor. Two mutation tests below plant the exact class of mutation Niobe's review
demonstrated (a label swapped for its opposite) and assert, in-process, that the corresponding
label assertion now raises -- proof this guard is mutation-sensitive, not just aimed near the
words "MODEL" and "witnessed".

The two figures, the equality condition between `(16, 2)` and `(8, 4)`, and the three witnessed
ratios are all independently re-derived here from `bench/results/probe_weight_reread.py`'s own
`gemv_named_bytes` (the same formula the production tile search compares) and from the committed
`weight_reread_phi35.json` artifact -- never copied from `docs/PERF.md`'s own prose -- and only
then checked against what the document states.

No GPU, no EP, no model file.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

_BENCH = Path(__file__).resolve().parent
_REPO = _BENCH.parent
RESULTS = _BENCH / "results"
if str(RESULTS) not in sys.path:
    sys.path.insert(0, str(RESULTS))

import probe_weight_reread as pwr  # noqa: E402  (reuse the canonical gemv_named_bytes formula)

PERF = _REPO / "docs" / "PERF.md"
ARTIFACT = "weight_reread_phi35.json"

#: The Phi-3.5 shape issue #81's own probe range and Niobe's review comment both cite.
K = N = 3072
BITS = 4
A_BYTES = 2  # fp16 activations


# ---------------------------------------------------------------------------
# Loading and clause/bullet extraction
# ---------------------------------------------------------------------------

def _perf() -> str:
    if not PERF.is_file():
        pytest.skip("docs/PERF.md is absent")
    return PERF.read_text(encoding="utf-8")


def _clause_at(text: str, idx: int) -> str:
    """The single clause containing offset `idx`, bounded by the nearest `.`/`;` on each side --
    not a fixed character window.

    This is the fix for issue #81's B2 finding: the rejected guard's +/-700 character window
    could, and did, include unrelated prose. A clause bounded by the punctuation the prose itself
    already uses to separate its two half-sentences cannot.
    """
    left = max(text.rfind(".", 0, idx), text.rfind(";", 0, idx))
    start = 0 if left == -1 else left + 1
    right_dot = text.find(".", idx)
    right_semi = text.find(";", idx)
    ends = [e for e in (right_dot, right_semi) if e != -1]
    end = min(ends) if ends else len(text)
    return text[start:end + 1]


def _bullet_at(text: str, idx: int) -> str:
    """The `* ...` bullet containing offset `idx`, bounded by adjacent bullets/blank lines."""
    start = text.rfind("\n* ", 0, idx)
    start = 0 if start == -1 else start + 1
    end_bullet = text.find("\n* ", idx)
    end_blank = text.find("\n\n", idx)
    ends = [e for e in (end_bullet, end_blank) if e != -1]
    end = min(ends) if ends else len(text)
    return text[start:end]


def _clause_containing(text: str, needle: str) -> str:
    return _clause_at(text, text.index(needle))


def _bullet_containing(text: str, needle: str) -> str:
    return _bullet_at(text, text.index(needle))


def _assert_clause_label(text: str, number_literal: str, expected: str, forbidden: str) -> None:
    clause = _clause_containing(text, number_literal)
    assert re.search(rf"\*\*{re.escape(expected)}\*\*", clause), (
        f"{number_literal}'s clause is not labelled {expected}: {clause!r}"
    )
    assert not re.search(rf"\*\*{re.escape(forbidden)}\*\*", clause, re.IGNORECASE), (
        f"{number_literal}'s clause is wrongly (also) labelled {forbidden}: {clause!r}"
    )


# ---------------------------------------------------------------------------
# Independent re-derivation (not copied from docs/PERF.md)
# ---------------------------------------------------------------------------

def _witnessed_m4_weight_bytes() -> int:
    """The real `(16, 2)` weight-byte figure at M=4, read off the committed artifact."""
    path = RESULTS / ARTIFACT
    if not path.is_file():
        pytest.skip(f"{ARTIFACT} is not committed in this tree")
    doc = json.loads(path.read_text(encoding="utf-8"))
    for row in doc["by_shape_prefill"]:
        if row["K"] == K and row["N"] == N and row["m_total"] == 4:
            return row["tiled_named_bytes"]
    raise AssertionError(f"{ARTIFACT} carries no K=N={K}, M=4 row")


def _model_m4_8x4_weight_bytes() -> int:
    """The MODEL-only `(8, 4)` weight-byte figure at the same M, N, K -- from the same
    `gemv_named_bytes` formula the production selector itself compares (`a_bytes=0` isolates the
    weight-only term), reused from `probe_weight_reread.py` rather than re-typed here."""
    return pwr.gemv_named_bytes(4, N, K, BITS, 0, 8, 4)


def _model_total_named_bytes(m: int, cols: int, rows: int) -> int:
    """Total named bytes (weight + activation) for the same shape, via the canonical formula."""
    return pwr.gemv_named_bytes(m, N, K, BITS, A_BYTES, cols, rows)


# ---------------------------------------------------------------------------
# B2: per-claim MODEL/witnessed labelling, clause-bound (not a wide substring window)
# ---------------------------------------------------------------------------

def test_the_witnessed_m4_weight_bytes_are_labelled_witnessed_not_model():
    """THE B2 CONTROL, half 1. The real, artifact-witnessed `(16, 2)` figure at M=4 must be
    labelled **witnessed**, not **MODEL**, in the clause it actually sits in."""
    _assert_clause_label(_perf(), "9,437,184", "witnessed", "MODEL")


def test_the_unreachable_8x4_weight_bytes_are_labelled_model_not_witnessed():
    """THE B2 CONTROL, half 2. The `(8, 4)` figure at the same M is never the tile the selector
    reaches -- it is a computed MODEL value, not a witnessed one -- and must be labelled that way
    in the clause it actually sits in."""
    _assert_clause_label(_perf(), "4,718,592", "MODEL", "witnessed")


def test_a_mislabel_mutation_of_the_8x4_figure_is_caught():
    """THE B2 MUTATION PROOF, forward direction. Niobe's review of PR #92 demonstrated that
    swapping a claim's own MODEL label for a measured/witnessed one did not turn the guard red,
    because the guard searched a wide window rather than the claim's own clause. Planting the
    same class of mutation here -- MODEL -> witnessed, on the `(8, 4)` figure's own clause -- must
    turn `_assert_clause_label` red, or this guard has the same defect.
    """
    text = _perf()
    needle = "and this second figure is a **MODEL** value"
    assert needle in text, "the exact phrase this test mutates must still be present verbatim"
    mutated = text.replace(needle, "and this second figure is a **witnessed** value", 1)
    with pytest.raises(AssertionError):
        _assert_clause_label(mutated, "4,718,592", "MODEL", "witnessed")


def test_a_rollback_mutation_of_the_witnessed_figure_is_caught():
    """THE B2 MUTATION PROOF, reverse direction: relabelling the real, witnessed M=4 figure as a
    MODEL value (a rollback of the correction, not just a forward mislabel) must also turn the
    corresponding assertion red."""
    text = _perf()
    needle = "is the **witnessed** `(16, 2)` figure"
    assert needle in text, "the exact phrase this test mutates must still be present verbatim"
    mutated = text.replace(needle, "is the **MODEL** `(16, 2)` figure", 1)
    with pytest.raises(AssertionError):
        _assert_clause_label(mutated, "9,437,184", "witnessed", "MODEL")


# ---------------------------------------------------------------------------
# The two figures are re-derived, not copied
# ---------------------------------------------------------------------------

def test_the_witnessed_figure_matches_the_committed_artifact():
    m = re.search(r"\*\*(9,437,184) B\*\*", _perf())
    assert m, "docs/PERF.md no longer publishes the witnessed M=4 weight-byte figure verbatim"
    published = int(m.group(1).replace(",", ""))
    assert published == _witnessed_m4_weight_bytes() == 9_437_184


def test_the_model_figure_matches_the_canonical_formula():
    m = re.search(r"=\s*(4,718,592) B", _perf())
    assert m, "docs/PERF.md no longer publishes the MODEL (8,4) weight-byte figure verbatim"
    published = int(m.group(1).replace(",", ""))
    assert published == _model_m4_8x4_weight_bytes() == 4_718_592


def test_the_witnessed_figure_is_exactly_double_the_model_figure_at_m4():
    """Ties the two figures together independently of either literal: at M=4, `(16, 2)` reads 2x
    the weight bytes of `(8, 4)` -- `ceil(4/2) = 2` versus `ceil(4/4) = 1` passes over the same
    column strip."""
    assert _witnessed_m4_weight_bytes() == 2 * _model_m4_8x4_weight_bytes()


# ---------------------------------------------------------------------------
# B3: the equality condition, independently re-derived (not copied from the document's prose)
# ---------------------------------------------------------------------------

def test_the_total_named_bytes_equality_condition_is_m_congruent_0_or_3_mod_4():
    """Re-derives the tie condition straight from the canonical `gemv_named_bytes` formula --
    not by parsing the claim out of docs/PERF.md and trusting it -- and only then checks the
    document states the same residues."""
    assert BITS == 2 * A_BYTES, "the tie condition's load-bearing sub-condition for this shape"
    residues_that_tie = set()
    for m in range(1, 201):
        a = _model_total_named_bytes(m, 16, 2)
        b = _model_total_named_bytes(m, 8, 4)
        if a == b:
            residues_that_tie.add(m % 4)
    assert residues_that_tie == {0, 3}, residues_that_tie

    text = _perf()
    assert "M \\equiv 0" in text, "docs/PERF.md must state the residue condition"
    assert re.search(r"3\s*\(mod 4\)", text), "docs/PERF.md must state the full residue pair"


def test_the_witnessed_m2_m4_m5_ratios_match_the_documents_own_numbers():
    """The three M this probe actually witnesses (M=2, 4, 5): only M=4 ties; M=2 is 2.00x and M=5
    is 1.33x -- exactly the numbers issue #81's B3 finding cites, re-derived here rather than
    quoted from the document."""
    def ratio(m: int) -> float:
        return _model_total_named_bytes(m, 8, 4) / _model_total_named_bytes(m, 16, 2)

    assert abs(ratio(2) - 2.0) < 1e-9
    assert abs(ratio(4) - 1.0) < 1e-9
    assert abs(ratio(5) - 4 / 3) < 1e-9

    text = _perf()
    assert "2.00" in text and "1.33" in text, (
        "docs/PERF.md must state both witnessed non-tying ratios (M=2 and M=5)"
    )


def test_m128_ties_and_is_not_generalised_from_m2_or_m5():
    """M=128 (Phi-3.5's own prefill width) is the one `M` an equal-total A/B may legitimately
    use; the document must say so without implying it generalises to the witnessed M=2 or M=5,
    which do not tie."""
    assert _model_total_named_bytes(128, 16, 2) == _model_total_named_bytes(128, 8, 4)
    text = _perf()
    bullet = _bullet_containing(text, "M = 128")
    assert "128 mod 4 == 0" in bullet


# ---------------------------------------------------------------------------
# (8, 4) stays UNMEASURED; the withdrawal is preserved in full
# ---------------------------------------------------------------------------

def test_84_is_explicitly_labelled_unmeasured_at_least_once():
    text = _perf()
    positions = [m.start() for m in re.finditer(re.escape("(8, 4)"), text)]
    assert positions, "docs/PERF.md no longer mentions (8, 4) at all"
    assert any("UNMEASURED" in _bullet_at(text, p) for p in positions), (
        "no bullet mentioning (8, 4) carries the UNMEASURED label"
    )


def test_no_84_clause_carries_a_bandwidth_or_time_figure():
    """A clause-scoped (not bullet-scoped) check that `(8, 4)` itself is never given a live
    performance number: the withdrawal bullet in S26.4 legitimately quotes withdrawn GB/s/ms/%
    figures nearby, in the sentence that withdraws them -- but never in the same clause as an
    `(8, 4)` mention itself.
    """
    unit = re.compile(r"\d[\d,]*(?:\.\d+)?\s*(?:GB/s|ms\b|%)")
    text = _perf()
    positions = [m.start() for m in re.finditer(re.escape("(8, 4)"), text)]
    assert positions, "docs/PERF.md no longer mentions (8, 4) at all"
    for pos in positions:
        clause = _clause_at(text, pos)
        assert not unit.search(clause), f"a clause mentioning (8, 4) carries a number: {clause!r}"


def test_the_withdrawn_bandwidth_series_is_preserved_verbatim():
    """The withdrawal must still name every figure it withdraws (issue #81: preserve the
    withdrawal in full) -- a silent deletion of the disclosure, as much as a silent replacement
    number, would lose the record of what was wrong and why."""
    text = _perf()
    for literal in ("199.7", "217.4", "200\u2013245", "227", "323 ms", "11%"):
        assert literal in text, f"the withdrawn figure {literal!r} is no longer in docs/PERF.md"
    bullet = _bullet_containing(text, "199.7")
    assert "withdrawn in full" in bullet and "not replaced" in bullet
