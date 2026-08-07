"""Regenerate `bench/public_path_legacy.json`, the declared-legacy ratchet.

Issue #69's eight evidence JSONs were not the first artifacts in this tree to
name the operator's machine — they were the first ones anybody screened. A
repository-wide screen turned on today would be red on 299 files that predate
it, and a screen that is red on day one is a screen that gets skipped.

So the screen is a **ratchet**, not a gate: every file that leaks today is
declared here with its leak count, and
`bench/test_public_paths.py::test_no_committed_evidence_file_leaks_more_than_declared`
fails if a file leaks that is not declared, if a declared file leaks *more*
than declared, or if a declared file has stopped leaking or stopped existing.
The last two matter as much as the first: a declaration nobody removes when it
becomes false is how a screen quietly stops describing the tree.

The list may only ever shrink. Regenerating it after adding a leak is not the
remedy for adding a leak — the remedy is not adding one. Run this only when
removing entries::

    python bench/gen_public_path_legacy.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import public_paths as pp  # noqa: E402

BASELINE = Path(__file__).resolve().parent / "public_path_legacy.json"

#: Extensions that carry committed evidence a reader is expected to read.
#: Defined on the boundary module so the scanner CLI and the ratchet cannot
#: drift into disagreeing about what counts as an evidence file.
EVIDENCE_SUFFIXES = pp.EVIDENCE_SUFFIXES

#: Paths under which nothing may *ever* be declared legacy. These are the trees
#: whose writers now go through `public_paths`, so a leak here is a regression
#: in a fixed path, not an inheritance.
NEVER_LEGACY = ("bench/results/_cuda69/",)


def tracked_evidence(repo: Path) -> "list[str]":
    out = subprocess.run(["git", "ls-files", "-z"], cwd=repo, check=True,
                         capture_output=True, text=True).stdout
    return sorted(f for f in out.split("\0")
                  if f and Path(f).suffix.lower() in EVIDENCE_SUFFIXES)


def survey(repo: Path) -> "dict[str, dict]":
    found: "dict[str, dict]" = {}
    for rel in tracked_evidence(repo):
        target = repo / rel
        if not target.is_file():
            continue
        hits = pp.scan(target.read_text(encoding="utf-8", errors="replace"))
        if hits:
            found[rel] = {"leaks": len(hits),
                          "kinds": sorted({kind for kind, _ in hits})}
    return found


def main(argv=None) -> int:
    repo = pp.REPO
    files = survey(repo)
    doc = {
        "schema": "public_path_legacy/1",
        "note": (
            "Committed evidence files that named a machine before "
            "bench/public_paths.py existed. This list may only shrink. Adding an "
            "entry to make a test pass is the defect the test is for."),
        "never_legacy": list(NEVER_LEGACY),
        "regenerate_with": "python bench/gen_public_path_legacy.py",
        "files": files,
    }
    BASELINE.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n",
                        encoding="utf-8")
    print(f"{BASELINE.name}: {len(files)} declared legacy file(s), "
          f"{sum(v['leaks'] for v in files.values())} leak(s)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
