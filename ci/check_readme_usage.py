#!/usr/bin/env python3
"""``check_readme_usage`` — every module the README tells a user to import must exist.

THE DEFECT THIS EXISTS FOR
==========================
For most of this project's life the README's usage block read:

    import onnxruntime_ep_vulkan
    onnxruntime_ep_vulkan.register_execution_provider_library()

and that package did not exist. No ``pyproject.toml``, no ``setup.py``, no ``__init__.py``,
and ``git ls-files | grep onnxruntime_ep_vulkan`` returned nothing. The block was honestly
labelled *"Intended usage"* — but a reader does not run a label, they run the code, and the
first thing anyone does with this project produced ``ModuleNotFoundError``.

The section was wrong for months and nothing could have noticed, because **no check reads
the README as executable claims**. Documentation drift is not caught by the test suite by
construction: the test suite imports what exists.

WHAT THIS CHECKS
================
Every ``import X`` and ``from X import ...`` inside a fenced ``python`` code block in
``README.md`` must be importable from a checkout — either a third-party package this
project depends on, or a first-party module that is actually in the tree.

It does **not** execute the blocks. Executing them needs a GPU, a built artifact and a
model; a check that can only run on one desk is not a check. Import-resolvability is the
part that is decidable from the tree alone, and it is the part that was wrong.

VERDICTS
--------
    0  README-USAGE: PASS  — every imported module resolves
    1  README-USAGE: FAIL  — a documented import names something that does not exist
    4  README-USAGE: ERROR(instrument=...) — the check could not read the README

Run:  python ci/check_readme_usage.py [--readme PATH]
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

#: Where a first-party module may live. A documented import is satisfied by any of these.
FIRST_PARTY_ROOTS = ("python/src", "bench", "tests/ops", "ci")

#: Files that *declare* third-party dependencies. A documented import of a declared
#: dependency resolves even when that dependency is not installed in the interpreter
#: running this check — the no-GPU lane installs pytest/onnx/numpy and not onnxruntime,
#: and failing there would be a false positive about the lane rather than a finding about
#: the README.
DEPENDENCY_MANIFESTS = ("python/pyproject.toml", "tests/requirements.txt")

_REQ_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def declared_dependencies(tree: Path) -> dict[str, str]:
    """Distribution names declared anywhere in *tree*, mapped to the file declaring them."""
    found: dict[str, str] = {}
    for rel in DEPENDENCY_MANIFESTS:
        path = tree / rel
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        names: list[str] = []
        if path.suffix == ".toml":
            try:
                import tomllib  # noqa: PLC0415

                data = tomllib.loads(text)
                names = list(data.get("project", {}).get("dependencies", []))
            except Exception:  # pragma: no cover - tomllib is stdlib on 3.11+
                names = re.findall(r'"([^"]+)"', text.split("dependencies")[-1])
        else:
            names = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
        for raw in names:
            m = _REQ_NAME.match(raw)
            if m:
                # Import name vs distribution name differ by - -> _ for every dependency
                # this project has; record both spellings rather than guess one.
                dist = m.group(1)
                found.setdefault(dist, rel)
                found.setdefault(dist.replace("-", "_"), rel)
    return found

#: Fenced blocks tagged as python. ```python ... ``` and ```py ... ```.
_BLOCK = re.compile(r"^```(?:python|py)\s*$(.*?)^```\s*$", re.MULTILINE | re.DOTALL)


def python_blocks(text: str) -> list[str]:
    return [m.group(1) for m in _BLOCK.finditer(text)]


def imported_roots(block: str) -> set[str]:
    """Top-level module names imported by *block*, or an empty set if it does not parse.

    A block that does not parse is not this check's business — it is prose in a python
    fence, or a fragment. Reporting it would make the check noisy about the one thing it
    is not measuring.
    """
    try:
        tree = ast.parse(block)
    except SyntaxError:
        return set()
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def resolves(name: str, tree: Path, declared: dict[str, str]) -> tuple[bool, str]:
    """Is *name* importable, either as an installed package, from *tree*, or declared?"""
    for root in FIRST_PARTY_ROOTS:
        base = tree / root
        if (base / name / "__init__.py").is_file():
            return True, f"first-party package {root}/{name}/"
        if (base / f"{name}.py").is_file():
            return True, f"first-party module {root}/{name}.py"
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        spec = None
    if spec is not None:
        return True, "installed"
    if name in declared:
        return True, f"declared dependency ({declared[name]})"
    return False, "NOT FOUND"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--readme", type=Path, default=REPO / "README.md")
    ap.add_argument(
        "--tree", type=Path, default=REPO,
        help="repository root that first-party imports are resolved against "
             "(the negative control points this at an empty tree so a historical README "
             "is judged against the historical state, not against today's)",
    )
    args = ap.parse_args(argv)

    if not args.readme.is_file():
        print(f"README-USAGE: ERROR(instrument=readme_absent) {args.readme}")
        return 4
    try:
        text = args.readme.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"README-USAGE: ERROR(instrument=readme_unreadable) {exc}")
        return 4

    blocks = python_blocks(text)
    if not blocks:
        # Not a pass: a README with no python block is a README this check did not check.
        print(
            "README-USAGE: ERROR(instrument=no_python_blocks) "
            f"{args.readme} has no fenced python block; nothing was verified"
        )
        return 4

    seen: dict[str, tuple[bool, str]] = {}
    declared = declared_dependencies(args.tree)
    for block in blocks:
        for name in imported_roots(block):
            seen.setdefault(name, resolves(name, args.tree, declared))

    if not seen:
        print(
            "README-USAGE: ERROR(instrument=no_imports_found) "
            f"{len(blocks)} python block(s) parsed, none imports anything"
        )
        return 4

    missing = sorted(n for n, (ok, _) in seen.items() if not ok)
    for name in sorted(seen):
        ok, why = seen[name]
        print(f"  {'ok  ' if ok else 'MISS'}  {name:<24} {why}")

    if missing:
        print(
            f"\nREADME-USAGE: FAIL — {len(missing)} documented import(s) name nothing that "
            f"exists: {', '.join(missing)}\n"
            f"The README tells a reader to run this. A reader who runs it gets "
            f"ModuleNotFoundError. Either ship the module or correct the documentation."
        )
        return 1

    print(
        f"\nREADME-USAGE: PASS — {len(seen)} distinct import(s) across "
        f"{len(blocks)} python block(s), all resolvable"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
