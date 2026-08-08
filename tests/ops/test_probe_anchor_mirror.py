"""The ranking probe's anchor set must mirror the partitioner's heavy-family list.

WHY THIS FILE EXISTS
--------------------
``rust/tools/probe_island_counterfactual.py`` ranks candidate op registrations by how much
island mass they would add, and it applies an anchor exemption while doing so. Its anchor set
used to be a hand-maintained six-tuple::

    ("Conv", "Gemm", "MatMul", "MatMulNBits", "Attention", "GroupQueryAttention")

The partitioner's list had grown to eleven names. ``ConvTranspose``, ``MultiHeadAttention``,
``QMoE`` and ``LinearAttention`` were missing, silently, for as long as it took anyone to
compare two files by eye — which is to say, indefinitely. The instrument was ranking against a
partitioner that no longer existed, and nothing was red.

The probe now derives its set from ``ops::partition::HEAVY_OP_FAMILIES`` by parsing the Rust
source. This file is the falsifier for that derivation: it re-parses the Rust list
independently, compares the two as sets, and — crucially — checks that the derivation *notices*
a divergence, by mutating a copy of the source and requiring the parse to change.

WHAT IS AND IS NOT CLAIMED
--------------------------
The probe's set is the **ceiling** on anchoring, not the shipped rule. Since issue #73,
``ops::partition::is_anchor`` requires a heavy family *and* a resident weight at a
schema-designated input; a claim log records op names, not per-input constancy, so the probe
cannot evaluate the second half. This file asserts that the probe mirrors the family list and
that its docstring says which half it models. It does not assert the probe agrees with the
partitioner's verdicts, because it does not.

Run::

    pytest tests/ops/test_probe_anchor_mirror.py -v --no-header
"""

from __future__ import annotations

import importlib.util
import pathlib
import re
import sys

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[2]
_PARTITION_RS = _REPO / "rust" / "src" / "ops" / "partition.rs"
_PROBE_PY = _REPO / "rust" / "tools" / "probe_island_counterfactual.py"


def _load_probe():
    """Import the probe module by path, without requiring `rust/tools` on `sys.path`."""
    spec = importlib.util.spec_from_file_location("_probe_island_counterfactual", _PROBE_PY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _rust_families(text: str) -> list[str]:
    """Independently re-parse `HEAVY_OP_FAMILIES` — a second implementation, on purpose."""
    marker = "pub const HEAVY_OP_FAMILIES: &[&str] = &["
    start = text.index(marker)
    end = text.index("];", start)
    return re.findall(r'"([^"]+)"', text[start + len(marker) : end])


def _bare(names) -> set[str]:
    """ONNX `op_type` carries no domain, so `com.microsoft::X` compares as `X`."""
    return {n.rsplit("::", 1)[-1] for n in names}


def test_the_probe_anchor_set_is_the_partitioner_heavy_family_list():
    families = _rust_families(_PARTITION_RS.read_text(encoding="utf-8"))
    assert families, "HEAVY_OP_FAMILIES parsed as empty — the source moved"
    probe = _load_probe()
    assert set(probe.DEFAULT_ANCHORS) == _bare(families), (
        f"probe DEFAULT_ANCHORS={sorted(probe.DEFAULT_ANCHORS)} does not mirror "
        f"HEAVY_OP_FAMILIES={sorted(_bare(families))}"
    )


def test_the_four_names_that_were_missing_are_present():
    """The specific historical defect, named so a reader knows what went wrong before.

    A pure set-equality assertion would also pass if *both* sides regressed together. These
    four are asserted against a literal because their absence is the bug that motivated the
    derivation.
    """
    probe = _load_probe()
    for op in ("ConvTranspose", "MultiHeadAttention", "QMoE", "LinearAttention"):
        assert op in probe.DEFAULT_ANCHORS, f"{op} is in the partitioner's list and not here"


@pytest.mark.parametrize(
    "mutation,expect",
    [
        # Add a name to the Rust list: the derived set must grow.
        (('    "MatMul",\n', '    "MatMul",\n    "SyntheticFamily",\n'), "SyntheticFamily"),
        # Omit one: the derived set must shrink.
        (('    "ConvTranspose",\n', ""), "ConvTranspose"),
    ],
)
def test_the_derivation_tracks_the_rust_list_under_mutation(tmp_path, mutation, expect):
    """Non-vacuity: the parser must *respond* to the source, not merely agree with it today.

    A hard-coded tuple would pass the equality test above on the day it was written and fail
    silently thereafter — which is exactly what happened. Here the Rust source is copied,
    mutated, and re-parsed; the derived set is required to change accordingly.
    """
    original = _PARTITION_RS.read_text(encoding="utf-8")
    old, new = mutation
    assert old in original, f"mutation anchor {old!r} not found — the source moved"
    mutated = original.replace(old, new, 1)
    assert mutated != original

    before = _bare(_rust_families(original))
    after = _bare(_rust_families(mutated))
    assert before != after, "the mutation did not change the parsed set"
    if new:
        assert expect in after and expect not in before
    else:
        assert expect in before and expect not in after


def test_the_probe_refuses_to_guess_when_the_list_is_gone(tmp_path, monkeypatch):
    """A missing list must be a loud failure, not a silent fallback to a stale literal.

    The whole hazard is a mirror that keeps working after the thing it mirrors moves.
    """
    probe = _load_probe()
    fake_root = tmp_path / "rust"
    (fake_root / "src" / "ops").mkdir(parents=True)
    (fake_root / "src" / "ops" / "partition.rs").write_text(
        "// no families here\n", encoding="utf-8"
    )
    (fake_root / "tools").mkdir()
    shim = fake_root / "tools" / "probe_island_counterfactual.py"
    shim.write_text(_PROBE_PY.read_text(encoding="utf-8"), encoding="utf-8")

    spec = importlib.util.spec_from_file_location("_probe_shim_missing_list", shim)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with pytest.raises(SystemExit) as excinfo:
        spec.loader.exec_module(module)
    assert "HEAVY_OP_FAMILIES" in str(excinfo.value)


def test_the_probe_documents_that_it_models_only_the_family_half():
    """The probe must not read as though it reproduces the shipped anchor rule.

    It models the heavy-family half. The resident-weight half needs per-input constancy, which
    a claim log does not carry. Saying so in the file is what keeps a reader from treating its
    anchor-exempt islands as the partitioner's verdicts.
    """
    src = _PROBE_PY.read_text(encoding="utf-8")
    assert "ceiling" in src
    assert "resident weight" in src
