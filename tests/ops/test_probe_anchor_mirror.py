"""The counterfactual probe's anchor set must mirror the production heavy-op family list.

Issue #73, Fact Checker finding. `rust/tools/probe_island_counterfactual.py` carried a
hand-maintained 6-tuple::

    DEFAULT_ANCHORS = ("Conv", "Gemm", "MatMul", "MatMulNBits", "Attention",
                       "GroupQueryAttention")

against a production list of eleven families in ``ops::partition``. Four were missing:
``ConvTranspose``, ``MultiHeadAttention``, ``QMoE`` and ``LinearAttention``. Nothing failed.
A probe that under-counts anchors under-counts retained islands, which makes every
counterfactual delta it prints look *better* than the change really is -- drift in the one
direction that flatters the author. The list had no falsifier, so it drifted.

This test is the falsifier. It asserts set equality between the two sides **and mutates each
side to prove the assertion can fail**, because a set-equality assertion that both sides derive
from one source is a tautology (§7.0.2 / RAI-011): the probe now parses the Rust constant, so
the interesting question is not "do they agree" -- they must -- but "would this test notice if
someone re-froze the probe's list, or added a family to Rust that the parse missed".

Both arms are exercised:

* **omission** -- drop a family from the Rust source the probe parses; the derived set must
  shrink and stop matching the production set;
* **addition** -- add a family; the derived set must grow and stop matching the frozen set.

NO CLOCK. No model, no GPU, no network: this is a source-text agreement check.

Run::

    pytest tests/ops/test_probe_anchor_mirror.py -v --no-header
"""

from __future__ import annotations

import importlib.util
import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
PROBE = REPO / "rust" / "tools" / "probe_island_counterfactual.py"
PARTITION_RS = REPO / "rust" / "src" / "ops" / "partition.rs"


def _load_probe():
    spec = importlib.util.spec_from_file_location("_probe_island_counterfactual", PROBE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def probe():
    return _load_probe()


@pytest.fixture(scope="module")
def rust_source() -> str:
    return PARTITION_RS.read_text(encoding="utf-8")


def _families_used_by_is_heavy_op_family(source: str) -> set[str]:
    """Bare op_types the *production predicate* accepts.

    Read independently of the probe's own parser, from the declaration `is_heavy_op_family`
    actually consults, so that agreement between the two is evidence rather than restatement.
    """
    m = re.search(r"pub const HEAVY_OP_FAMILIES:\s*&\[&str\]\s*=\s*&\[(.*?)\];", source, re.S)
    assert m is not None, "HEAVY_OP_FAMILIES declaration not found in partition.rs"
    body = m.group(1)
    # The predicate must consult this constant and nothing else, or the mirror is meaningless.
    assert re.search(
        r"pub fn is_heavy_op_family\(qualified_op: &str\) -> bool \{\s*"
        r"HEAVY_OP_FAMILIES\.contains\(&qualified_op\)\s*\}",
        source,
    ), (
        "is_heavy_op_family no longer reads HEAVY_OP_FAMILIES verbatim; the probe's derived "
        "anchor set is no longer a mirror of the shipped predicate"
    )
    return {n.rsplit("::", 1)[-1] for n in re.findall(r'"([^"]+)"', body)}


def test_probe_default_anchors_equal_the_production_heavy_families(probe, rust_source):
    production = _families_used_by_is_heavy_op_family(rust_source)
    derived = set(probe.DEFAULT_ANCHORS)
    assert derived == production, (
        f"probe anchor set disagrees with production heavy-op families; "
        f"missing={sorted(production - derived)} extra={sorted(derived - production)}"
    )


def test_the_four_families_the_frozen_list_omitted_are_present_now(probe):
    """Named explicitly, so a regression reads as a regression and not as a count change."""
    for family in ("ConvTranspose", "MultiHeadAttention", "QMoE", "LinearAttention"):
        assert family in probe.DEFAULT_ANCHORS, (
            f"{family} is a production heavy-op family and was one of the four the frozen "
            "DEFAULT_ANCHORS tuple omitted; its absence understates retained islands"
        )


def test_default_anchors_is_not_accidentally_the_old_frozen_tuple(probe):
    frozen = {"Conv", "Gemm", "MatMul", "MatMulNBits", "Attention", "GroupQueryAttention"}
    assert set(probe.DEFAULT_ANCHORS) != frozen, "the stale six-name list is back"


def test_an_omitted_family_makes_the_mirror_fail(probe, rust_source):
    """Mutation arm 1: the guard notices a family the probe fails to pick up."""
    production = _families_used_by_is_heavy_op_family(rust_source)
    mutated = rust_source.replace('    "com.microsoft::QMoE",\n', "", 1)
    assert mutated != rust_source, "mutation did not apply; the fixture text has moved"
    derived = probe.production_heavy_op_families(source=mutated)
    assert "QMoE" not in derived
    assert derived != production, (
        "dropping a family from the parsed source did not change the derived set -- the "
        "derivation is not actually reading the source and this guard is vacuous"
    )


def test_an_extra_family_makes_the_mirror_fail(probe):
    """Mutation arm 2: the guard notices a family present in Rust and absent from the probe."""
    frozen = set(probe.DEFAULT_ANCHORS)
    mutated = PARTITION_RS.read_text(encoding="utf-8").replace(
        '    "MatMul",\n', '    "MatMul",\n    "com.microsoft::FusedConv",\n', 1
    )
    derived = probe.production_heavy_op_families(source=mutated)
    assert "FusedConv" in derived
    assert derived != frozen, "an added production family left the mirror satisfied"


def test_the_derivation_refuses_rather_than_guessing(probe):
    """No silent fallback to a literal: an unparseable source must raise.

    A default here would recreate the exact defect -- a list that keeps working after it stops
    being true.
    """
    with pytest.raises(RuntimeError, match="HEAVY_OP_FAMILIES"):
        probe.production_heavy_op_families(source="// no declaration here\n")
    with pytest.raises(RuntimeError, match="empty"):
        probe.production_heavy_op_families(
            source="pub const HEAVY_OP_FAMILIES: &[&str] = &[];\n"
        )


def test_the_probe_states_that_its_anchor_set_is_a_ceiling(probe):
    """The set mirrors *families*, not `is_anchor`. That gap must be written down.

    Production anchor status also requires a resident weight (issue #73), which the claim log
    cannot answer. Saying so is what keeps the probe's delta a ranking signal instead of a
    prediction -- the confusion that put "32 GQA nodes are anchors" into the design doc.
    """
    doc = PROBE.read_text(encoding="utf-8")
    assert "ceiling" in doc
    assert "is_anchor" in doc
