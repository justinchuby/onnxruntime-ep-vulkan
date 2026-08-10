"""Pytest lane for ``ci/check_anchor_weight_sites.py`` — the schema/source agreement guard.

This runs the reproducible anchor weight-site checker under pytest so the pinned schema
provenance (``rust/tools/anchor_weight_sites.json``), the shipped Rust tables in
``rust/src/ops/partition.rs``, and the live ``onnx.defs`` schema library are asserted to
agree on every CI run — not only when someone remembers to run the checker by hand.

The checker needs no GPU, no model and no EP, so it is always in the lane. It does need the
``onnx`` package for the live standard-domain extraction; if that is unavailable the checker
reports the missing extraction as a failure rather than passing vacuously, and this test
surfaces it.

Two polarities, per R10 (a screen only trusted when it is shown it can go red):
  - the real repository must PASS;
  - a scratch copy whose Rust weight-site table has been mutated to reintroduce name-only
    anchoring must FAIL.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_CHECKER = _REPO / "ci" / "check_anchor_weight_sites.py"


def _run(cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(cwd / "ci" / "check_anchor_weight_sites.py"), "--check"],
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )


def test_checker_exists():
    assert _CHECKER.exists(), f"missing anchor weight-site checker at {_CHECKER}"


def test_real_repo_passes():
    """The shipped table, JSON provenance and live onnx schema must agree."""
    p = _run(_REPO)
    assert p.returncode == 0, f"checker went red on the real repo:\n{p.stdout}\n{p.stderr}"
    assert "PASS" in p.stdout


def _mirror(tmp: Path) -> Path:
    """Copy just enough of the repo for the checker to run against a scratch tree."""
    for rel in ("ci", "rust/src/ops", "rust/tools", "third_party/onnxruntime"):
        src = _REPO / rel
        dst = tmp / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)
    return tmp


def test_name_only_anchoring_is_caught(tmp_path):
    """Reintroduce the issue #73 defect in a scratch copy; the checker must go red.

    Mutation: give GroupQueryAttention a weight site in the Rust table (the exact name-only
    over-anchoring the fix removed). The checker cross-checks the JSON — which pins GQA to no
    weight site — and must fail.
    """
    tree = _mirror(tmp_path)
    part = tree / "rust" / "src" / "ops" / "partition.rs"
    text = part.read_text(encoding="utf-8")
    mutated = text.replace(
        '"com.microsoft::MatMulNBits" => &[1],',
        '"com.microsoft::MatMulNBits" => &[1],\n        '
        '"com.microsoft::GroupQueryAttention" => &[14],',
        1,
    )
    assert mutated != text, "mutation anchor not found; update the test to match partition.rs"
    part.write_text(mutated, encoding="utf-8")

    p = _run(tree)
    assert p.returncode != 0, (
        "checker PASSED after GQA was given a bogus weight site — it does not actually detect "
        f"name-only anchoring:\n{p.stdout}\n{p.stderr}"
    )
    assert "GroupQueryAttention" in (p.stdout + p.stderr)
