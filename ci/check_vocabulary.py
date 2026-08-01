#!/usr/bin/env python3
"""``check_vocabulary`` — preflight that makes ``ERROR(instrument)`` readable.

THE HAZARD THIS EXISTS FOR
==========================
Every check in ``ci/`` imports its verdict vocabulary from ``tests/ops/_verdict.py`` and
defines none of its own — one vocabulary, not two.  The consequence is that when that
module cannot be imported, **every lane step reports an instrument outage at once**:

    GATE: ERROR(instrument=verdict_vocabulary_unavailable)

That is the honest report, and it is also a hazard, because two very different situations
produce the identical line:

  (a) **The checkout legitimately does not contain the module.**  An older branch, a
      bisect, a PR opened before the vocabulary landed, a sparse checkout.  Nothing is
      broken; this tree simply predates the vocabulary.  Every lane will say it, on every
      step, forever, until the tree gains the file.
  (b) **The lane is broken.**  The file is right there in the checkout and the job still
      cannot import it: wrong Python, partial checkout, a syntax error introduced
      upstream, a permissions problem, an ``ImportError`` from one of the module's own
      dependencies.

If those two are reported the same way then ``ERROR(instrument=...)`` becomes the lane's
normal state, and a genuine outage is indistinguishable from the weather.  A signal that
is always on is not a signal.

WHAT THIS STEP DOES ABOUT IT
============================
It answers the question **before** the gate runs, from the repository's own state rather
than from an import that can fail for a dozen reasons, and it gives each case its own
token so a maintainer greps a word, not a mood:

    0  VOCAB: PASS
       Present, importable, and its provenance is printed.  Every later
       `verdict_vocabulary_unavailable` in this job is therefore a **lane** fault: this
       step just proved the module imports in this interpreter, in this checkout.

    4  VOCAB: ERROR(instrument=verdict_vocabulary_absent_from_checkout)
       The file is not in the tree.  This is a **repository state, not a lane defect**.
       The lane is still red — a lane that cannot emit a verdict cannot be green — but it
       is red for a reason that no CI change will fix, and the message says so and names
       the file that has to arrive.

    4  VOCAB: ERROR(instrument=verdict_vocabulary_broken)
       The file is in the tree and does not import.  This **is** a lane or source defect,
       and the exception text is quoted in full (R13: quote the text, never the count).

The two error tokens share exit code 4 because both are instrument outages and neither is
a detection.  They do not share a **token**, and the token is what a maintainer reads.

PROVENANCE, SO TWO LANES CAN BE COMPARED
========================================
On every path this prints the module's path, its SHA-256, its size, and whether git tracks
it.  That turns "is this the weather or is this my lane?" into a diff:

  * both lanes report ``absent_from_checkout`` with the same commit → repository state (a)
  * one lane imports it and another cannot → lane (b), and the difference is right there
  * both lanes report ``broken`` with the same SHA-256 → the module itself, not the lanes

CI SURFACING
============
With ``--github-summary`` the outcome is appended to ``$GITHUB_STEP_SUMMARY`` as a table
row and emitted as a ``::error title=...::`` annotation whose **title differs per token**.
The requirement is that a maintainer sees which of the two it is from the run's summary
page, without opening a log — a caveat that lives only in a log is not attached to the
thing it qualifies.

USAGE
    python ci/check_vocabulary.py [--github-summary]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VOCAB_REL = Path("tests") / "ops" / "_verdict.py"
VOCAB_PATH = REPO_ROOT / VOCAB_REL

EXIT_PASS = 0
EXIT_ERROR_INSTRUMENT = 4

TOKEN_ABSENT = "verdict_vocabulary_absent_from_checkout"
TOKEN_BROKEN = "verdict_vocabulary_broken"


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def provenance() -> dict[str, str]:
    """Facts about the module that survive being pasted into an issue."""
    info: dict[str, str] = {
        "path": str(VOCAB_REL).replace(os.sep, "/"),
        "exists": "yes" if VOCAB_PATH.is_file() else "no",
        "commit": _git("rev-parse", "--short", "HEAD") or "<unknown>",
        "git_tracked": "yes" if _git("ls-files", "--error-unmatch", str(VOCAB_REL)) else "no",
        "python": sys.version.split()[0],
    }
    if VOCAB_PATH.is_file():
        data = VOCAB_PATH.read_bytes()
        info["sha256"] = hashlib.sha256(data).hexdigest()[:16]
        info["bytes"] = str(len(data))
        info["last_touched_by"] = _git("log", "-1", "--format=%h %s", "--", str(VOCAB_REL)) or "<unknown>"
    else:
        info["sha256"] = "<absent>"
        info["bytes"] = "<absent>"
        info["last_touched_by"] = "<absent>"
    return info


def _format_provenance(info: dict[str, str]) -> str:
    return "\n".join(f"  {k} = {v}" for k, v in info.items())


def _annotate(token: str | None, title: str, message: str, summary: bool) -> None:
    """Surface the outcome where it is read, not only where it is logged."""
    if not summary:
        return
    one_line = message.replace("\n", " ")
    if token is None:
        print(f"::notice title={title}::{one_line}", flush=True)
    else:
        print(f"::error title={title}::{one_line}", flush=True)
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(f"### Verdict vocabulary — {title}\n\n{message}\n\n")
    except OSError:
        # Failing to write the summary must not become the lane's terminal state; the
        # stdout above already carries it.
        pass


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--github-summary",
        action="store_true",
        help="also emit a ::error/::notice annotation and a $GITHUB_STEP_SUMMARY section",
    )
    args = p.parse_args(sys.argv[1:] if argv is None else argv)

    info = provenance()
    prov = _format_provenance(info)

    if not VOCAB_PATH.is_file():
        detail = (
            f"{VOCAB_REL.as_posix()} is not present in this checkout.\n{prov}\n\n"
            "This is a REPOSITORY STATE, not a lane defect. Every verdict-carrying step "
            "in this job will report ERROR(instrument=verdict_vocabulary_unavailable), "
            "and none of them is a finding about the EP or about CI. The lane is red "
            "because a lane that cannot emit a verdict cannot be green — not because "
            "anything in it is broken.\n"
            "Distinguishing rule for a maintainer: if the OTHER lanes on this same "
            "commit also report `verdict_vocabulary_absent_from_checkout`, the commit "
            "does not carry the vocabulary and no CI change will fix it. If any lane on "
            "this commit reports VOCAB: PASS, this one is broken and the difference "
            "between the two jobs is the fault."
        )
        print(f"VOCAB: ERROR(instrument={TOKEN_ABSENT})", flush=True)
        print(detail, flush=True)
        _annotate(
            TOKEN_ABSENT,
            "verdict vocabulary absent from checkout (repository state, not a lane defect)",
            f"`{VOCAB_REL.as_posix()}` is not in commit `{info['commit']}`. "
            "Every gate step in this job will report an instrument outage. This is not a "
            "finding about the EP and not a CI fault.",
            args.github_summary,
        )
        return EXIT_ERROR_INSTRUMENT

    sys.path.insert(0, str(REPO_ROOT / "tests" / "ops"))
    try:
        import _verdict  # type: ignore
    except Exception as exc:  # noqa: BLE001 - every import failure is this case
        detail = (
            f"{VOCAB_REL.as_posix()} is present in this checkout and does not import.\n"
            f"{prov}\n\n"
            f"{exc!r}\n{traceback.format_exc()}\n"
            "This IS a lane or source defect: the file is right there and this "
            "interpreter cannot load it. Quoting the exception rather than a count, "
            "because a count is what let a NameError masquerade as a detection on "
            "2026-07-31."
        )
        print(f"VOCAB: ERROR(instrument={TOKEN_BROKEN})", flush=True)
        print(detail, flush=True)
        _annotate(
            TOKEN_BROKEN,
            "verdict vocabulary present but unimportable (lane or source defect)",
            f"`{VOCAB_REL.as_posix()}` (sha256:{info['sha256']}) is in the checkout and "
            f"raised {type(exc).__name__} on import under Python {info['python']}.",
            args.github_summary,
        )
        return EXIT_ERROR_INSTRUMENT

    # Present and importable. Report the tokens it defines, so a later disagreement
    # about the vocabulary is a diff and not an argument.
    tokens = [
        getattr(_verdict, name)
        for name in dir(_verdict)
        if name.startswith("VERDICT_") and isinstance(getattr(_verdict, name), str)
    ]
    print("VOCAB: PASS", flush=True)
    print(prov, flush=True)
    print(f"  verdict tokens = {sorted(tokens)}", flush=True)
    print(
        "Because this step passed, any later ERROR(instrument="
        "verdict_vocabulary_unavailable) in this job is a LANE fault and not a "
        "repository state: the module imported here, in this interpreter, in this "
        "checkout, moments ago.",
        flush=True,
    )
    _annotate(
        None,
        "verdict vocabulary present and importable",
        f"`{VOCAB_REL.as_posix()}` sha256:{info['sha256']} imports cleanly; "
        f"{len(tokens)} verdict tokens. Later vocabulary outages in this job are lane faults.",
        args.github_summary,
    )
    return EXIT_PASS


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
