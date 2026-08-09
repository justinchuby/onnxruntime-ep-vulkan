"""`probe_island_counterfactual.py`'s anchor model must mirror the Rust source it claims to.

WHY THIS EXISTS
---------------
The probe hard-coded a six-name tuple described as *"mirrors `ops::partition::is_anchor` closely
enough to rank with"*. It did not. By the time issue #73 was opened the Rust list held eleven
names, and the probe was missing `ConvTranspose`, `MultiHeadAttention`, `QMoE` and
`LinearAttention` and spelled the contrib ops unqualified. Nothing failed: the probe kept
ranking, and the ranking kept being wrong for four families.

This is `DESIGN.md` §5.4.1(a)'s citation ruling in a second costume — **a reference that decays
without failing**. A transcription of a list in another file is a citation. The repair is the
same: parse the source, raise when you cannot, and pin the two together with a test.

WHAT IT ASSERTS
---------------
1. The probe parses `HEAVY_OP_FAMILIES` out of `partition.rs` and gets the eleven names.
2. The probe's ranking set is exactly those names' `op_type` forms — an ONNX `NodeProto.op_type`
   is unqualified, so `com.microsoft::QMoE` must become `QMoE` and nothing may be lost or gained
   in the conversion.
3. The parse *raises* on an unreadable source rather than falling back to a guess.
4. The probe's report discloses that its anchor set is a ceiling, not the shipped rule.

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
    spec = importlib.util.spec_from_file_location("probe_island_counterfactual", PROBE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _families_from_rust() -> tuple[str, ...]:
    """Independent re-read of the Rust literal, so the test does not trust the parser it checks."""
    text = PARTITION_RS.read_text(encoding="utf-8")
    body = text.split("pub const HEAVY_OP_FAMILIES: &[&str] = &[", 1)[1].split("];", 1)[0]
    return tuple(re.findall(r'"([^"]+)"', body))


def test_the_probe_reads_the_rust_list_rather_than_a_transcription():
    probe = _load_probe()
    assert probe.heavy_op_families() == _families_from_rust()


def test_the_mirrored_list_is_the_full_eleven_in_source_order():
    """Order matters as well as membership: `ep.rs`'s FLOP branch iterates this list."""
    assert _families_from_rust() == (
        "MatMul",
        "Gemm",
        "Conv",
        "ConvTranspose",
        "Attention",
        "com.microsoft::MatMulNBits",
        "com.microsoft::GroupQueryAttention",
        "com.microsoft::MultiHeadAttention",
        "com.microsoft::Attention",
        "com.microsoft::QMoE",
        "com.microsoft::LinearAttention",
    )


def test_the_ranking_set_is_the_unqualified_form_and_loses_nothing():
    """`NodeProto.op_type` carries no domain, so the conversion must be exact, not lossy."""
    probe = _load_probe()
    families = probe.heavy_op_families()
    ranking = probe.anchor_op_types(families)
    assert ranking == {f.rsplit("::", 1)[-1] for f in families}
    # `Attention` exists in both the default and the `com.microsoft` domain; collapsing them is
    # correct here and must not silently shrink the set below the number of distinct bare names.
    assert len(ranking) == len({f.rsplit("::", 1)[-1] for f in families})
    for required in ("ConvTranspose", "MultiHeadAttention", "QMoE", "LinearAttention"):
        assert required in ranking, (
            f"{required} is in HEAVY_OP_FAMILIES but not in the probe's ranking set — this is the "
            f"exact drift the hard-coded six-name tuple had"
        )


def test_an_unreadable_source_raises_rather_than_guessing(tmp_path):
    """A silent fallback is how the old transcription stayed wrong. Raising is the fix."""
    probe = _load_probe()

    missing = tmp_path / "no_such_partition.rs"
    missing.write_text("// nothing here\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="HEAVY_OP_FAMILIES"):
        probe.heavy_op_families(missing)

    unterminated = tmp_path / "unterminated.rs"
    unterminated.write_text(
        'pub const HEAVY_OP_FAMILIES: &[&str] = &[\n    "MatMul",\n', encoding="utf-8"
    )
    with pytest.raises(RuntimeError):
        probe.heavy_op_families(unterminated)

    empty = tmp_path / "empty.rs"
    empty.write_text("pub const HEAVY_OP_FAMILIES: &[&str] = &[\n];\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="empty"):
        probe.heavy_op_families(empty)


def test_the_probe_discloses_that_its_anchor_set_is_a_ceiling():
    """The probe has no per-node residency, so it over-counts anchors. It must say so."""
    text = PROBE.read_text(encoding="utf-8")
    assert "ceiling" in text.lower(), (
        "the probe models the family half of the anchor rule only, which over-states retained "
        "island size; a reader who takes its anchor set for the shipped rule is misled"
    )
    assert "anchors_are" in text, (
        "the emitted report must carry the same disclosure as the docstring — a caveat that lives "
        "in a different artifact from the number it qualifies is not attached to it"
    )
