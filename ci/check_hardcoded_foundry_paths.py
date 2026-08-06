#!/usr/bin/env python3
"""``check_hardcoded_foundry_paths`` — no *new* live code may hardcode a Foundry cache path.

THE DEFECT THIS EXISTS FOR
==========================
Foundry Local's on-disk cache layout is versioned by its own CLI's internal catalog
revision, not by anything this repository controls. ``tests/ops/test_phi35.py`` hardcoded

    C:\\Users\\...\\.foundry\\cache\\models\\Microsoft\\Phi-3.5-mini-instruct-cuda-gpu
                                              \\cuda-int4-rtn-block-32\\...\\.onnx

and the path went stale under us with no code change on either side: Foundry Local 0.10.2
downloads the identical model to ``Phi-3.5-mini-instruct-cuda-gpu-2\\v2\\...`` instead, and
the only reason anyone noticed was a manually-created directory junction bridging the two
(issue #11). PR #15 landed ``rust/tools/foundry_discovery.py`` — a resolver keyed on model
*identity* (variant name + execution provider), not on a path — and migrated the tests. Issue
#19 migrated the ~31 remaining hardcodes in tools, probes and archived benchmark scripts that
PR #15 had not reached.

A migration with no standing guard is a migration that erodes: the next probe anyone writes
under time pressure will paste a hardcoded path back in, because that is what every existing
probe in the tree looked like until this pass. This screen is the guard.

WHAT THIS CHECKS
================
Every ``*.py`` file in the tree is scanned for the literal path fragment that names a
Foundry cache directory directly (``...\\.foundry\\cache\\models...`` or the POSIX spelling
``.../.foundry/cache/models/...``), in *either* a real path-shaped separator to catch drift
in either style. A hit is permitted only inside an explicit allowlist of files/directories
that are immutable archival record or the resolver's own documentation of the defect it
fixes — never in a live lookup.

    ALLOWED   bench/results/**    — archived one-off investigation scripts. Their default
                                    path is deliberately the exact historical artifact they
                                    were measured against (see issue #19); each accepts an
                                    explicit ``PHI35_MODEL`` env override rather than
                                    auto-resolving against whatever is cached today, because
                                    a live resolver could silently swap in a *different*
                                    cached revision than the one the archived result names.
    ALLOWED   rust/tools/foundry_discovery.py — the resolver itself; the fragment appears
                                    only in its own docstring, naming the defect it exists
                                    to prevent.
    ALLOWED   tests/ops/test_foundry_discovery.py — synthetic fixtures use a `C:\\fake\\...`
                                    prefix and never this literal fragment, but the file is
                                    allowlisted defensively since it is the resolver's test
                                    surface.
    ALLOWED   ci/test_lane_checks.py, ci/lane_inventory.py — this screen's own pytest
                                    coverage plants literal example hardcodes as PLANTED
                                    fixtures, and the evidence ledger's entry for this very
                                    check quotes the fragment to describe what it rejects;
                                    neither is a live lookup.
    REJECTED  everything else. A hit anywhere else is either a new live tool that skipped
                                    the resolver, or an archival script placed outside
                                    bench/results/ where this screen cannot tell it apart
                                    from a live lookup.

This is a **static, source-text** screen: it does not import or execute anything, so it runs
without a GPU, without Foundry Local installed, and without the model being cached.

VERDICTS
--------
    0  FOUNDRY-PATHS: PASS  — no hardcoded cache path outside the allowlist
    1  FOUNDRY-PATHS: FAIL  — a live file hardcodes a Foundry cache path; each hit is named
    4  FOUNDRY-PATHS: ERROR(instrument=...) — the check could not read the tree

Run:  python ci/check_hardcoded_foundry_paths.py [--root PATH]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

#: The literal fragment that names a Foundry cache directory, in either path-separator
#: style. Deliberately narrow: it does not match a model *identity* string (e.g.
#: "Phi-3.5-mini-instruct-cuda-gpu" alone, used as a resolver key) or a pathlib join whose
#: segments are separate string literals (e.g. ``Path.home() / ".foundry" / "cache"``) —
#: only a single literal that already spells out the on-disk hierarchy, which is exactly
#: the shape a hardcoded lookup takes.
_PATTERN = re.compile(r"\.foundry[\\/]+cache[\\/]+models", re.IGNORECASE)

#: Directories whose *.py contents are the archival record itself: their default path is
#: deliberately the exact historical artifact, per issue #19's classification.
_ALLOWED_DIRS = (
    "bench/results/",
)

#: Individual files allowed to name the pattern: the resolver's own defect-documentation,
#: its test surface, and this screen's own docstring (which quotes the pattern to describe
#: what it rejects).
_ALLOWED_FILES = (
    "rust/tools/foundry_discovery.py",
    "tests/ops/test_foundry_discovery.py",
    "ci/check_hardcoded_foundry_paths.py",
    "ci/negative_control_hardcoded_foundry_paths.py",
    "ci/test_lane_checks.py",
    "ci/lane_inventory.py",
)

#: Directories never scanned: not source, or third-party / build output that is not ours
#: to screen.
_EXCLUDED_DIRS = (".git", ".venv", "venv", "target", "node_modules", "__pycache__", ".squad")


def is_allowlisted(rel_posix: str) -> bool:
    if rel_posix in _ALLOWED_FILES:
        return True
    return any(rel_posix.startswith(d) for d in _ALLOWED_DIRS)


def iter_python_files(root: Path):
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root)
        if any(part in _EXCLUDED_DIRS for part in rel.parts):
            continue
        yield path, rel.as_posix()


def scan(root: Path) -> tuple[list[tuple[str, int, str]], list[tuple[str, int, str]]]:
    """Return (violations, allowlisted_hits): file/line/quoted-line pairs, each a match."""
    violations: list[tuple[str, int, str]] = []
    allowlisted: list[tuple[str, int, str]] = []
    for path, rel_posix in iter_python_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if _PATTERN.search(line):
                entry = (rel_posix, lineno, line.strip())
                (allowlisted if is_allowlisted(rel_posix) else violations).append(entry)
    return violations, allowlisted


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=REPO, help="repository root to scan")
    args = ap.parse_args(argv)

    if not args.root.is_dir():
        print(f"FOUNDRY-PATHS: ERROR(instrument=root_absent) {args.root}")
        return 4

    violations, allowlisted = scan(args.root)

    for rel_posix, lineno, line in allowlisted:
        print(f"  allowlisted  {rel_posix}:{lineno}  {line}")

    if violations:
        print(f"\nFOUNDRY-PATHS: FAIL — {len(violations)} hardcoded Foundry cache path(s) "
              f"outside the allowlist:")
        for rel_posix, lineno, line in violations:
            print(f"  {rel_posix}:{lineno}: {line}")
        print(
            "\nA hardcoded Foundry cache path goes stale silently whenever Foundry Local's "
            "own catalog revision changes (issue #11) and ties archival reproducibility to "
            "one cache layout (issue #19). Live lookups must resolve by identity via "
            "rust/tools/foundry_discovery.py:resolve_model_path(). Archival investigation "
            "scripts belong under bench/results/ with an explicit env-var override, not a "
            "silent live guess."
        )
        return 1

    print(
        f"FOUNDRY-PATHS: PASS — {len(allowlisted)} allowlisted occurrence(s), "
        f"0 outside the allowlist"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
