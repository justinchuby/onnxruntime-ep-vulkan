"""Regenerate `bench/public_path_legacy.json`, the declared-legacy ratchet.

Issue #69's eight evidence JSONs were not the first artifacts in this tree to
name the operator's machine — they were the first ones anybody screened. A
repository-wide screen turned on today would be red on every file that predates
it, and a screen that is red on day one is a screen that gets skipped.

How many files that is, and how many leaks they carry, is deliberately not
written in this prose: it is the `files` map of `bench/public_path_legacy.json`,
which this module generates and :func:`artifact_totals` reads back. A count typed
beside the collection it summarises goes stale the first time the collection
moves, and nothing says so — this docstring carried such a number, and it was
already wrong by the time the artifact it summarised was regenerated. Any count
that IS quoted, here or in `docs/PERF.md`, is checked against the artifact by
`bench/test_public_paths.py::test_no_prose_quotes_a_file_count_the_ratchet_does_not_carry`.

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
import re
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

# ---------------------------------------------------------------------------
# COUNTS ARE READ FROM THE ARTIFACT, NEVER TYPED BESIDE IT.
#
# This module's own docstring used to open with a file count that no longer matched the
# artifact two lines of code away from it. Nothing was wrong with the artifact; the defect
# was the second copy of a number, with no mechanism that makes the two disagree out loud.
#
# `artifact_totals` is that mechanism's input: the only sanctioned source for "how many".
# Prose is free to quote a figure — a report that says "a screen would be red on N files"
# is more use to a reader than one that says "on some files" — but only a figure the
# artifact carries, and `stale_count_claims` is what makes that enforceable rather than
# aspirational. The files it is enforced over are listed in `COUNTED_PROSE`.
COUNTED_PROSE = (
    "docs/PERF.md",
    "bench/gen_public_path_legacy.py",
    "bench/public_paths.py",
    "bench/test_public_paths.py",
)

#: The quotable shapes: "<n> committed evidence files", "<n> leaks", "<n> declared files".
#: Narrow on purpose — it screens claims about THIS artifact's collections, not every
#: integer in a document. The `(?<![\d.])` guard keeps `Phi-3.5 file` out of it, and the
#: shapes are spelled without an example number for the reason this whole block exists.
#:
#: THE NOUN IS CAPTURED BECAUSE THE NOUN IS THE BINDING. See :data:`COUNT_NOUNS`.
COUNT_CLAIM = re.compile(
    r"(?<![\d.])(\d[\d,]*)\s+(?:(?:current|committed|declared|tracked|legacy|evidence)\s+)*"
    r"(files?|leaks?)\b",
    re.IGNORECASE,
)

#: Which of :func:`artifact_totals`' collections each quotable noun is a count OF.
#:
#: WHY A MAP AND NOT A SET OF ALLOWED NUMBERS.  The screen used to ask "is this integer one
#: of the numbers the artifact publishes?", and this artifact publishes two — how many
#: files are declared, and how many leaks they carry between them. So a sentence that
#: quoted the FILE count and called it a leak count passed: the digits were real, and a
#: screen that checks digits against a bag cannot tell which collection they came from.
#: Every count claim is now bound to the collection its own noun names, so a file count
#: cannot witness a leak claim and a leak count cannot witness a file claim, however many
#: of each the artifact happens to have. Two collections that coincidentally hold the same
#: number is not a licence to quote either for the other.
#:
#: No figure appears in this comment for the reason the comment is about: it is prose
#: beside a collection, and `stale_count_claims` screens this very file.
COUNT_NOUNS = {"file": "file", "files": "file", "leak": "leak", "leaks": "leak"}


def load_baseline() -> "dict":
    """The committed ratchet, as a document. One reader, so callers cannot disagree."""
    return json.loads(BASELINE.read_text(encoding="utf-8"))


def artifact_totals(doc=None) -> "dict[str, int]":
    """The two numbers this artifact may be quoted as: entries, and leaks summed over them.

    Derived from the `files` map every time it is asked, so there is no third place for a
    count to go stale.
    """
    files = (load_baseline() if doc is None else doc)["files"]
    return {"file": len(files),
            "leak": sum(rec["leaks"] for rec in files.values())}


def stale_count_claims(text: str, totals=None, name: str = "<text>") -> "list[str]":
    """Count claims in ``text`` that the artifact does not carry. Offenders, not a tally.

    Each claim is checked against the ONE collection its noun names, never against the set
    of every number the artifact publishes. A sentence quoting the declared-file count and
    calling it a leak count is an offender here and was not before: its digits were
    witnessed by a collection, just not by the collection the sentence was about.
    """
    totals = artifact_totals() if totals is None else totals
    offenders = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for match in COUNT_CLAIM.finditer(line):
            value = int(match.group(1).replace(",", ""))
            noun = match.group(2).lower()
            key = COUNT_NOUNS[noun]
            expected = totals.get(key)
            if value != expected:
                offenders.append(
                    f"{name}:{line_no}: {match.group(0)!r} — the ratchet declares "
                    f"{expected} {key}(s); {value} is not that count "
                    f"(it declares {totals['file']} file(s) and {totals['leak']} leak(s), "
                    f"and a claim about one may not be witnessed by the other)")
    return offenders


def tracked_evidence(repo: Path) -> "list[str]":
    out = subprocess.run(["git", "ls-files", "-z"], cwd=repo, check=True,
                         capture_output=True, text=True).stdout
    return sorted(f for f in out.split("\0")
                  if f and Path(f).suffix.lower() in EVIDENCE_SUFFIXES)


def survey_texts(items) -> "dict[str, dict]":
    """Count the leaks in ``(relative path, text)`` pairs. The whole survey rule, in one place.

    WHY THE SURVEY IS STRUCTURAL-ONLY, AND WHY THAT IS A CORRECTION RATHER THAN A RELAXATION.

    This ratchet is checked into the repository and compared against by every contributor's
    test run. It therefore has to be a fact about the *files*, and until now it was partly a
    fact about whoever ran the generator: `pp.scan` includes an ``account_name`` screen built
    from ``Path.home().name`` and ``$USERNAME``, so the same tree surveyed under a different
    account produced a different list.

    Three ways that went wrong, all of them on somebody else's machine:

    * On a CI account (``runner``), nearly every declared file stopped leaking and the
      ratchet failed with "a declared file has stopped leaking" — a red with no defect
      behind it, in a test whose whole job is to be believable when it is red.
    * A short or common account name (``dev``, ``ann``) matched inside ordinary words. With
      no word boundary, ``devices``/``annotation`` counted as leaks, in files with nothing
      wrong with them, and `scrub_text` rewrote those words in prose.
    * The generator recorded the current operator's account in a committed artifact, which
      is the exact thing `public_paths` exists to keep out of committed artifacts.

    Nothing is given up by dropping it. The leaks this list is a ratchet on are home
    directories, checkout names, virtualenvs and system roots — shapes, matched the same way
    on every machine, and a Windows home directory with an operator's account under it is
    still counted here on a laptop where nobody has that account. The account screen stays
    exactly where it belongs: at the *write* boundary in `dump_public_json`, judging a
    payload this process just built.
    """
    found: "dict[str, dict]" = {}
    for rel, text in items:
        hits = pp.scan_structural(text)
        if hits:
            found[rel] = {"leaks": len(hits),
                          "kinds": sorted({kind for kind, _ in hits})}
    return found


def survey(repo: Path) -> "dict[str, dict]":
    def _texts():
        for rel in tracked_evidence(repo):
            target = repo / rel
            if not target.is_file():
                continue
            yield rel, target.read_text(encoding="utf-8", errors="replace")

    return survey_texts(_texts())


def main(argv=None) -> int:
    repo = pp.REPO
    files = survey(repo)
    doc = {
        "schema": "public_path_legacy/2",
        "note": (
            "Committed evidence files that named a machine before "
            "bench/public_paths.py existed. This list may only shrink. Adding an "
            "entry to make a test pass is the defect the test is for."),
        "scan": (
            "structural kinds only (public_paths.scan_structural): every shape is decided "
            "from the file's text, so this list is the same on every machine. The "
            "account-name screen is deliberately absent — it is a fact about the process "
            "that runs it, and a ratchet nobody else can reproduce is not a ratchet."),
        "kinds": list(pp.STRUCTURAL_KINDS),
        "never_legacy": list(NEVER_LEGACY),
        "regenerate_with": "python bench/gen_public_path_legacy.py",
        "files": files,
    }
    BASELINE.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n",
                        encoding="utf-8")
    totals = artifact_totals(doc)
    print(f"{BASELINE.name}: {totals['file']} declared legacy file(s), "
          f"{totals['leak']} leak(s)")

    # Regenerating the artifact is exactly when the prose that quotes it goes stale, so
    # this is where a reader is told. Reported, not raised: the artifact on disk is now
    # correct, and the remaining work is in the documents that summarise it.
    offenders = []
    for rel in COUNTED_PROSE:
        target = repo / rel
        if target.is_file():
            offenders += stale_count_claims(
                target.read_text(encoding="utf-8", errors="replace"), totals, rel)
    if offenders:
        print("\nSTALE COUNT(S) QUOTED IN PROSE — update or remove each:", file=sys.stderr)
        for line in offenders:
            print(f"  ! {line}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
