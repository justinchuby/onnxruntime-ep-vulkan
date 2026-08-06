#!/usr/bin/env python3
"""Is a committed artifact NEWER than the fix it is cited as evidence for?

RAI ASKED THE GENERAL QUESTION AND IT IS THE RIGHT ONE. She verified RAI-012 as genuinely
fixed — rebuilt the crate under WSL, ran the real op suite, 42 declines all carrying the
corrected message and zero of the old one — and then found that the three artifact files
the round cited as the evidence were committed at `ac4bd0b` (07:12) while the fix landed at
`7688356` (10:33). The claim was true; the citation was not. Nothing in this repository
made that detectable.

A committed artifact is a READING, and a reading has a FRAME. The proof ledger already
does this properly: every entry records the subject it was taken against, and
`gen_proof_ledger.py --check` compares that against the running build, so a stale proof is
detectable rather than merely wrong. No other artifact in this repository carries a frame
at all.

WHAT THIS SCREEN DOES, AND WHY IT IS THE CHEAP HALF
---------------------------------------------------
Each declared artifact carries a sidecar `<dir>/artifact-frame.json` naming, per file:

  * `produced_at_commit` — HEAD when the run was taken
  * `subject_sha256`     — the built EP the run loaded
  * `platform`           — where
  * `subject_paths`      — the source the reading is ABOUT

and this screen asks three questions:

  1. `FAIL(condition=artifact_predates_subject)` — has any commit touched `subject_paths`
     since `produced_at_commit`? This needs NO BUILD, NO DEVICE and NO SECOND PLATFORM: it
     is `git log <commit>..HEAD -- <paths>`, and it is exactly the question Rai had to ask
     by hand. It is the arm that would have caught the RAI-012 citation.
  2. `FAIL(condition=artifact_content_moved)` — do the artifact bytes still hash to what
     the frame recorded? A frame that has drifted from its file is a frame for a reading
     nobody has. Text artifacts (`.json`/`.log`/`.txt`/`.md`) are hashed through the same
     line-ending normalisation as rust/build.rs's `normalize_shader_text`: a checkout that
     rewrites LF to CRLF (`core.autocrlf=true`) does not change what the artifact SAYS, and
     a digest that moves on it reports a difference the file does not have — see
     docs/DESIGN.md 8.9.26 and mouse's "portable digest" repair. It is still exactly as
     sensitive to a real edit, on any platform.
  3. `FAIL(condition=artifact_unframed)` — does this directory hold a file the frame says
     nothing about? A file dropped in beside a stamped frame, without a re-stamp, is
     exactly as uncompared as one the frame names and cannot find; this arm makes the
     directory and the frame's file list the same set, not merely a subset check.

WHAT THIS SCREEN DELIBERATELY DOES NOT DO, AND THE COST OF MAKING IT
--------------------------------------------------------------------
It does not compare `subject_sha256` against a live build by default. That comparison is
the strong form and it is NOT cheap here, for a reason measured in this repository rather
than assumed: the Windows `.dll` is not byte-reproducible across forced rebuilds
(PLATFORMS.md §7.21.3), so digest equality would report STALE on every Windows rebuild that
changed nothing. `--subject <path>` enables it for lanes that want it — Linux, where the
`.so` IS byte-identical across rebuilds — and it is off elsewhere because a check that
cries wolf on every rebuild is a check somebody switches off.

It also does not know what an artifact is "evidence FOR". `subject_paths` is a proxy —
"the source this reading is about" — and it is the writer's declaration, not a derivation.
That is a real limit: an artifact can be stale with respect to a fix in a path nobody
listed. The honest position is that this makes a stale citation DETECTABLE in the common
case, not impossible; the general case needs each claim to name its evidence, which is a
register nobody has asked for yet and which I am not building on speculation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
FRAME_NAME = "artifact-frame.json"
REQUIRED = ("produced_at_commit", "platform", "subject_paths", "files")

# Suffixes hashed through `_normalize_text` rather than raw. Every artifact this screen has
# ever framed is one of these; an unrecognised suffix is hashed byte-for-byte so a future
# binary artifact (a `.so`/`.dll` subject, say) is never silently reinterpreted as text.
TEXT_SUFFIXES = {".json", ".log", ".txt", ".md"}


def _normalize_text(data: bytes) -> bytes:
    """CRLF and a lone CR both become LF, exactly like rust/build.rs's normalize_shader_text.

    A Windows checkout with `core.autocrlf=true` rewrites every tracked text file's line
    endings on the way to disk; the bytes committed to git — the frame's actual authoritative
    surface — never change. A digest that moves on that rewrite is reporting a difference the
    artifact does not have, which is strictly worse than the digest it would replace: see
    docs/DESIGN.md 8.9.26's 133-byte "line-ending artifact" and mouse's "portable digest"
    repair (`.squad/agents/mouse/history.md`). Deliberately not a whitespace normaliser —
    trailing spaces and blank lines stay in the digest — only the checkout-time transform is
    absorbed.
    """
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


KNOWN_LIMITS = {
    "subject_paths_is_a_writers_declaration": (
        "`subject_paths` is a list the writer typed, not a derivation from the claim the "
        "artifact is cited for. An artifact can be stale with respect to a fix in a path "
        "nobody named, and this screen will say PASS. Closing it needs every claim to name "
        "its evidence; see ci/open_reds.json id=artifact_frame_subject_paths_is_a_declaration."
    ),
}


def _git(args: list[str], repo: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=str(repo), capture_output=True, text=True, encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    if path.suffix.lower() in TEXT_SUFFIXES:
        h.update(_normalize_text(path.read_bytes()))
        return h.hexdigest()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_frame(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"no frame at {path}; the comparison input is missing, so nothing was ruled on "
            "either way — that is UNOBSERVABLE, not PASS"
        )
    doc = json.loads(path.read_text(encoding="utf-8"))
    missing = [f for f in REQUIRED if not doc.get(f)]
    if missing:
        raise ValueError(f"{path}: frame is missing {missing}")
    return doc


def stamp(repo: Path, directory: Path, subject_paths: list[str], platform: str,
          subject: Path | None, note: str) -> dict:
    head = _git(["rev-parse", "HEAD"], repo).stdout.strip()
    dirty = _git(["status", "--porcelain", "--", str(directory)], repo).stdout.strip()
    dirty_subject = _git(["status", "--porcelain", "--", *subject_paths], repo).stdout.strip()
    files = {}
    for f in sorted(directory.iterdir()):
        if f.is_file() and f.name != FRAME_NAME:
            files[f.name] = _sha256(f)
    doc = {
        "schema": 1,
        "produced_at_commit": head,
        "produced_with_uncommitted_changes_in_dir": bool(dirty),
        "produced_with_uncommitted_changes_in_subject": bool(dirty_subject),
        "platform": platform,
        "subject_sha256": _sha256(subject) if subject else "",
        "subject_path": subject.name if subject else "",
        "subject_paths": subject_paths,
        "note": note,
        "files": files,
    }
    (directory / FRAME_NAME).write_text(
        json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return doc


def check(repo: Path, directory: Path, subject: Path | None) -> int:
    doc = load_frame(directory / FRAME_NAME)
    rel = directory.relative_to(repo).as_posix()
    head = _git(["rev-parse", "HEAD"], repo).stdout.strip()
    since = doc["produced_at_commit"]

    print(f"ARTIFACT FRAME of {rel}")
    print(f"  produced at {since[:12]} on {doc['platform']}; HEAD is {head[:12]}")
    print(f"  reading is ABOUT: {doc['subject_paths']}")

    ok = _git(["rev-parse", "--verify", "--quiet", since + "^{commit}"], repo)
    if ok.returncode != 0:
        print(
            f"ERROR(instrument=frame_unresolvable): produced_at_commit {since!r} is not a "
            "commit here. An unresolvable frame makes every comparison below vacuous and "
            "the screen would print PASS having read nothing."
        )
        return 2

    if doc.get("produced_with_uncommitted_changes_in_subject"):
        print("")
        print(
            "FAIL(condition=frame_pins_an_uncommitted_tree): the reading was taken while the "
            f"subject paths {doc['subject_paths']} were dirty relative to {since[:12]}, so "
            "produced_at_commit does not name the tree that was actually measured and every "
            "arm below compares against the wrong baseline. Commit the subject, then re-stamp."
        )
        return 1

    moved = []
    for name, digest in sorted(doc["files"].items()):
        f = directory / name
        if not f.is_file():
            moved.append((name, "the frame names it and it is gone"))
        elif _sha256(f) != digest:
            moved.append((name, "bytes differ from the frame"))
    if moved:
        print("")
        print(
            f"FAIL(condition=artifact_content_moved): {len(moved)} file(s) do not match the "
            "frame that claims to describe them. A frame that has drifted from its file is a "
            "frame for a reading nobody has; re-stamp when you re-run."
        )
        for name, why in moved:
            print(f"  - {name}: {why}")
        return 1

    unframed = sorted(
        f.name for f in directory.iterdir()
        if f.is_file() and f.name != FRAME_NAME and f.name not in doc["files"]
    )
    if unframed:
        print("")
        print(
            f"FAIL(condition=artifact_unframed): {len(unframed)} file(s) sit in this "
            "directory with no entry in the frame that claims to describe it. An artifact "
            "nobody stamped is a reading nobody can compare against anything; name it in "
            "`files` (re-stamp) or remove it from the directory."
        )
        for name in unframed:
            print(f"  - {name}")
        return 1

    later = _git(
        ["log", "--format=%H %s", f"{since}..{head}", "--", *doc["subject_paths"]], repo
    ).stdout.strip()
    if later:
        rows = later.splitlines()
        hist = doc.get("historical")
        print("")
        print(
            f"{'HISTORICAL' if hist else 'FAIL(condition=artifact_predates_subject)'}: "
            f"{len(rows)} commit(s) have touched "
            "the source this reading is about since the reading was taken. The artifact may "
            "still be TRUE — but it is not evidence about the tree it is being cited in, and "
            "on 2026-08-03 exactly that gap put a pre-fix log under a post-fix claim."
        )
        for row in rows[:10]:
            print(f"  - {row[:100]}")
        if len(rows) > 10:
            print(f"  ... and {len(rows) - 10} more")
        if hist:
            # A DECLARED HISTORICAL READING IS A POSITIVE STATE, NOT AN ABSENCE.
            # The screen's own advice used to be "move it out of the cited set and say so
            # in `note`" — i.e. empty `subject_paths` and write prose. That is a DELETION:
            # the staleness stops being printed, nobody owns it, and the same file goes on
            # being cited. It is the shape that put 43 deliberate retirements into the
            # proof census as VANISHED, one register over. So a deliberately old reading is
            # declared the way a retired proof is: owner, date, reason, and the moved
            # commits still listed above every single run, so the age is visible rather
            # than excused away.
            missing = [f for f in ("owner", "date", "reason", "superseded_by") if not hist.get(f)]
            if missing:
                print("")
                print(
                    f"ERROR(instrument=incomplete_historical_declaration): `historical` is "
                    f"missing {missing}. A reading kept on purpose needs a name, a date, a "
                    "reason and the thing that now answers the question it used to — "
                    "otherwise it is not a declaration, it is the absence of one with a key."
                )
                return 2
            print("")
            print(
                f"  DECLARED HISTORICAL by {hist['owner']} on {hist['date']}: "
                f"{hist['reason']}"
            )
            print(f"  The current answer to this reading's question is: {hist['superseded_by']}")
            print(
                "\nPASS (historical): the age above is real, declared, owned and printed. "
                "This artifact is NOT evidence about HEAD and must not be cited as if it "
                "were; it is kept because deleting a reading is how a measurement stops "
                "being falsifiable."
            )
            return 0
        print(
            f"\n  Repair: re-run the producer and re-stamp "
            f"(`python ci/check_artifact_frame.py --stamp {rel} ...`). If the reading is "
            "deliberately historical, declare it: add a `historical` block to "
            f"{FRAME_NAME} with `owner`, `date`, `reason` and `superseded_by`. Declaring it "
            "keeps the age printed on every run; emptying `subject_paths` would only stop "
            "the screen asking."
        )
        return 1

    if subject:
        recorded = doc.get("subject_sha256")
        if not recorded:
            print(
                "\nERROR(instrument=no_subject_witness): --subject was given but the frame "
                "records none, so there is nothing to compare it to."
            )
            return 2
        live = _sha256(subject)
        if live != recorded:
            print("")
            print(
                f"FAIL(condition=subject_moved): the reading was taken against "
                f"{recorded[:16]} and {subject.name} now hashes to {live[:16]}."
            )
            return 1
        print(f"  subject {subject.name} still hashes to {live[:16]}")

    print("")
    print(
        f"PASS: {len(doc['files'])} artifact(s) match their frame, and nothing has touched "
        f"{doc['subject_paths']} since {since[:12]}. Read as: this reading is about THIS "
        "tree. Not read as: this reading is green, or complete, or about anything outside "
        "the paths named above."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory")
    ap.add_argument("--repo", default=str(REPO))
    ap.add_argument("--stamp", action="store_true")
    ap.add_argument("--subject-paths", nargs="*", default=[])
    ap.add_argument("--platform", default="")
    ap.add_argument("--subject", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--assert-known-limit", default="")
    args = ap.parse_args(argv)
    if args.assert_known_limit:
        name = args.assert_known_limit
        if name not in KNOWN_LIMITS:
            print(
                f"ERROR(instrument=unknown_limit): {name!r} is not a declared limit of this "
                f"screen. Declared: {sorted(KNOWN_LIMITS)}. A register entry accepting a "
                "limit the screen does not admit to is an acceptance of nothing."
            )
            return 2
        print(f"KNOWN-LIMIT {name}")
        print(f"  {KNOWN_LIMITS[name]}")
        print(
            "\nFAIL(condition=known_limit_still_open): this is a declared, owned, bounded "
            "gap and it is red on purpose. It goes green when the limit is closed, not when "
            "somebody stops looking."
        )
        return 1
    repo = Path(args.repo).resolve()
    directory = (repo / args.directory).resolve()
    subject = Path(args.subject).resolve() if args.subject else None
    if subject and not subject.is_file():
        print(f"ERROR(instrument=no_subject): --subject {subject} does not exist.")
        return 2
    if args.stamp:
        if not args.subject_paths or not args.platform:
            print("--stamp needs --subject-paths and --platform")
            return 2
        doc = stamp(repo, directory, args.subject_paths, args.platform, subject, args.note)
        print(f"stamped {directory / FRAME_NAME} at {doc['produced_at_commit'][:12]} "
              f"over {len(doc['files'])} file(s)")
        return 0
    return check(repo, directory, subject)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR(instrument=frame_unusable): {exc}")
        sys.exit(2)
