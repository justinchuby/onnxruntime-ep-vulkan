#!/usr/bin/env python3
"""Does `check_artifact_frame.py` actually fail when an artifact IS stale?

A screen that has never been seen to fail is indistinguishable from a screen that cannot
fail. Same discipline as `negative_control_ledger_census.py`: LIVE arms run against this
repository as it stands, REPLAYED arms run against a real historical event that actually
happened here and was NOT planted, and PLANTED arms build a throwaway git repo where the
defect is constructed on purpose. The REPLAYED arms are the load-bearing ones — a planted
defect proves the code path runs, a replayed one proves it would have caught the real thing.

REPLAYED arm: RAI-012, 2026-08-03. `bench/results/link-linux-downstream/` was committed at
`ac4bd0b` and cited as the evidence for a fix that landed at `7688356`, three hours later.
Rai found it by reading timestamps by hand. This arm asserts the screen finds it from a
frame alone, and that the fix commit is among the commits it names.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SCREEN = HERE / "check_artifact_frame.py"

# Loaded (not just invoked as a subprocess) so the REPLAYED arm can hash a file exactly the
# way the screen itself does -- through `_sha256`'s line-ending normalisation for text
# suffixes. Re-implementing that hash here with plain `hashlib.sha256(f.read_bytes())`
# would make this arm platform-dependent again: it would build a frame from raw CRLF bytes
# on a Windows checkout and then ask the (CRLF-normalising) screen to compare against it,
# which is the same false "moved" this file exists to prevent, just relocated into the
# control instead of the thing it controls.
_spec = importlib.util.spec_from_file_location("check_artifact_frame", SCREEN)
_check_artifact_frame = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_check_artifact_frame)
frame_sha256 = _check_artifact_frame._sha256

# The real RAI-012 citation gap. Neither sha was planted by this file.
RAI012_ARTIFACT_COMMIT = "ac4bd0b"
RAI012_FIX_COMMIT = "7688356"

RESULTS: list[tuple[str, str, bool, str]] = []


def record(kind: str, name: str, ok: bool, detail: str) -> None:
    RESULTS.append((kind, name, ok, detail))
    print(f"  [{'ok' if ok else 'XX'}] {kind:<8} {name}: {detail}")


def run_screen(directory: str, repo: Path, extra: list[str] | None = None):
    return subprocess.run(
        [sys.executable, str(SCREEN), directory, "--repo", str(repo), *(extra or [])],
        capture_output=True, text=True, encoding="utf-8",
    )


def git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True,
                          text=True, encoding="utf-8")


def build_repo(root: Path) -> None:
    git(["init", "-q", "-b", "main"], root)
    git(["config", "user.email", "nc@example.invalid"], root)
    git(["config", "user.name", "negative control"], root)
    (root / "src").mkdir()
    # `newline="\n"` pins these to LF regardless of platform: `write_text`'s default
    # newline translation would otherwise make this control's own baseline CRLF on
    # Windows, which is exactly the checkout-time transform arm 4c below constructs on
    # purpose -- the fixture must not already be in that state before the arm runs.
    (root / "src" / "lib.rs").write_text("fn a() {}\n", encoding="utf-8", newline="\n")
    (root / "art").mkdir()
    (root / "art" / "run.log").write_text("633 passed\n", encoding="utf-8", newline="\n")
    git(["add", "-A"], root)
    git(["commit", "-q", "-m", "base"], root)


def stamp(root: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCREEN), "art", "--repo", str(root), "--stamp",
         "--subject-paths", "src", "--platform", "negative control"],
        capture_output=True, text=True, encoding="utf-8",
    )


def touch_subject(root: Path, text: str) -> None:
    (root / "src" / "lib.rs").write_text(text, encoding="utf-8", newline="\n")
    git(["add", "-A"], root)
    git(["commit", "-q", "-m", "fix: the thing the artifact is cited as evidence for"], root)



def planted(tmp: Path) -> None:
    # 1. clean stamp -> PASS
    r1 = tmp / "clean"
    r1.mkdir()
    build_repo(r1)
    stamp(r1)
    out = run_screen("art", r1)
    record("PLANTED", "clean_stamp_passes", out.returncode == 0 and "PASS" in out.stdout,
           f"rc={out.returncode}")

    # 2. subject moves after the stamp -> the RAI-012 shape
    touch_subject(r1, "fn a() { b(); }\n")
    out = run_screen("art", r1)
    record("PLANTED", "subject_moved_after_reading",
           out.returncode == 1 and "artifact_predates_subject" in out.stdout,
           f"rc={out.returncode}")

    # 2b. THE ARM FOR THE EXEMPTION, WHICH IS THE ARM THAT WAS MISSING FROM THE OTHER
    # REGISTER TOO. `historical` lets an old reading be KEPT on purpose. Every arm in
    # this control plants a defect; none exercised a declared-legitimate state, and one
    # register over that is exactly how 43 deliberate proof retirements spent days being
    # reported as deletions. So: the acquittal has a positive arm, and the acquittal is
    # refused when it is not signed.
    def _historical(repo, block):
        frame = repo / "art" / "artifact-frame.json"
        doc = json.loads(frame.read_text(encoding="utf-8"))
        if block is None:
            doc.pop("historical", None)
        else:
            doc["historical"] = block
        frame.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    _historical(r1, {"owner": "link", "date": "2026-08-04",
                     "reason": "planted: kept deliberately",
                     "superseded_by": "planted: the lane that answers this now"})
    out = run_screen("art", r1)
    record("PLANTED", "a DECLARED historical reading is acquitted",
           out.returncode == 0 and "PASS (historical)" in out.stdout,
           f"rc={out.returncode}")
    record("PLANTED", "and its age is still PRINTED, not excused away",
           "commit(s) have touched" in out.stdout and "HISTORICAL:" in out.stdout,
           "the moved commits must stay on screen or the declaration is a mute button")

    _historical(r1, {"owner": "link", "date": "2026-08-04",
                     "reason": "planted: no superseded_by"})
    out = run_screen("art", r1)
    record("PLANTED", "an UNSIGNED historical declaration is refused, not honoured",
           out.returncode == 2 and "incomplete_historical_declaration" in out.stdout,
           f"rc={out.returncode}")
    _historical(r1, None)
    out = run_screen("art", r1)
    record("PLANTED", "and removing the declaration restores the conviction",
           out.returncode == 1 and "artifact_predates_subject" in out.stdout,
           f"rc={out.returncode}")

    # 3. artifact bytes edited without re-stamping
    r2 = tmp / "edited"
    r2.mkdir()
    build_repo(r2)
    stamp(r2)
    (r2 / "art" / "run.log").write_text("999 passed\n", encoding="utf-8")
    out = run_screen("art", r2)
    record("PLANTED", "artifact_content_moved",
           out.returncode == 1 and "artifact_content_moved" in out.stdout,
           f"rc={out.returncode}")

    # 4. a frame file naming a file that no longer exists
    r3 = tmp / "deleted"
    r3.mkdir()
    build_repo(r3)
    stamp(r3)
    (r3 / "art" / "run.log").unlink()
    out = run_screen("art", r3)
    record("PLANTED", "artifact_file_deleted",
           out.returncode == 1 and "artifact_content_moved" in out.stdout,
           f"rc={out.returncode}")

    # 4b. a file dropped into the evidence directory without ever being stamped. The
    # `moved` loop above only walks the frame's own keys, so a brand-new, wholly unframed
    # file was previously invisible to this screen by construction -- present in the
    # directory that is supposed to be one artifact's whole evidence, absent from the one
    # document that is supposed to describe it. Issue #20's "moved/unframed artifact" arm.
    r3b = tmp / "unframed"
    r3b.mkdir()
    build_repo(r3b)
    stamp(r3b)
    (r3b / "art" / "extra-not-in-frame.log").write_text("nobody stamped this\n", encoding="utf-8")
    out = run_screen("art", r3b)
    record("PLANTED", "artifact_unframed",
           out.returncode == 1 and "artifact_unframed" in out.stdout
           and "extra-not-in-frame.log" in out.stdout,
           f"rc={out.returncode}")

    # 4c. the converse of the two arms above: a pure line-ending rewrite of a text
    # artifact (the CRLF a Windows checkout with core.autocrlf=true actually performs)
    # must NOT be reported as content having moved. This is the arm that proves the
    # `_normalize_text` fix is real rather than merely non-crashing: content bytes that
    # really changed (4/4b above) still convict, and a checkout transform that changed
    # nothing the artifact SAYS does not.
    r3c = tmp / "crlf"
    r3c.mkdir()
    build_repo(r3c)
    stamp(r3c)
    original = (r3c / "art" / "run.log").read_bytes()
    (r3c / "art" / "run.log").write_bytes(original.replace(b"\n", b"\r\n"))
    out = run_screen("art", r3c)
    record("PLANTED", "checkout_line_ending_rewrite_is_not_content_drift",
           out.returncode == 0 and "PASS" in out.stdout,
           f"rc={out.returncode}")

    # 5. no frame at all is UNOBSERVABLE, never PASS
    r4 = tmp / "noframe"
    r4.mkdir()
    build_repo(r4)
    out = run_screen("art", r4)
    record("PLANTED", "absent_frame_is_not_pass",
           out.returncode == 2 and "frame_unusable" in out.stdout, f"rc={out.returncode}")

    # 6. an unresolvable produced_at_commit must not silently rule on nothing.
    #    This is the defect that bit check_ledger_census.py this session: a boundary that
    #    resolves to nothing made every comparison vacuous and the screen printed PASS.
    r5 = tmp / "badsha"
    r5.mkdir()
    build_repo(r5)
    stamp(r5)
    frame = r5 / "art" / "artifact-frame.json"
    doc = json.loads(frame.read_text(encoding="utf-8"))
    doc["produced_at_commit"] = "0" * 40
    frame.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out = run_screen("art", r5)
    record("PLANTED", "unresolvable_frame_is_not_pass",
           out.returncode == 2 and "frame_unresolvable" in out.stdout, f"rc={out.returncode}")

    # 7. a frame stamped while the subject was dirty pins a tree nobody has
    r6 = tmp / "dirty"
    r6.mkdir()
    build_repo(r6)
    (r6 / "src" / "lib.rs").write_text("fn a() { uncommitted(); }\n", encoding="utf-8")
    stamp(r6)
    out = run_screen("art", r6)
    record("PLANTED", "frame_pins_an_uncommitted_tree",
           out.returncode == 1 and "frame_pins_an_uncommitted_tree" in out.stdout,
           f"rc={out.returncode}")

    # 8. a frame missing a required field
    r7 = tmp / "incomplete"
    r7.mkdir()
    build_repo(r7)
    stamp(r7)
    frame = r7 / "art" / "artifact-frame.json"
    doc = json.loads(frame.read_text(encoding="utf-8"))
    doc["subject_paths"] = []
    frame.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out = run_screen("art", r7)
    record("PLANTED", "incomplete_frame_is_not_pass",
           out.returncode == 2 and "frame_unusable" in out.stdout, f"rc={out.returncode}")


def replayed(tmp: Path) -> None:
    """The real RAI-012 citation gap, from this repository's own history."""
    art = REPO / "bench" / "results" / "link-linux-downstream"
    if not art.is_dir():
        record("REPLAYED", "rai012_citation_gap", False, "artifact directory is missing")
        return
    fix = git(["rev-parse", "--verify", "--quiet", RAI012_FIX_COMMIT + "^{commit}"], REPO)
    old = git(["rev-parse", "--verify", "--quiet", RAI012_ARTIFACT_COMMIT + "^{commit}"], REPO)
    if fix.returncode != 0 or old.returncode != 0:
        record("REPLAYED", "rai012_citation_gap", False,
               "the historical commits are not present (shallow clone?) — UNOBSERVABLE")
        return
    fix_sha, old_sha = fix.stdout.strip(), old.stdout.strip()

    # Restore the exact frame the artifacts would have carried had they been stamped when
    # they were committed, and ask the screen the question Rai asked by hand.
    scratch = tmp / "replay"
    scratch.mkdir()
    saved = art / "artifact-frame.json"
    backup = scratch / "artifact-frame.json.bak"
    had_frame = saved.is_file()
    if had_frame:
        shutil.copy2(saved, backup)
    try:
        doc = {
            "schema": 1,
            "produced_at_commit": old_sha,
            "produced_with_uncommitted_changes_in_dir": False,
            "produced_with_uncommitted_changes_in_subject": False,
            "platform": "WSL Ubuntu (replayed frame for RAI-012)",
            "subject_sha256": "",
            "subject_path": "",
            "subject_paths": ["rust/src"],
            "note": "REPLAY ONLY - reconstructed frame for the 2026-08-03 citation gap",
            "files": {
                f.name: frame_sha256(f)
                for f in sorted(art.iterdir())
                if f.is_file() and f.name != "artifact-frame.json"
            },
        }
        saved.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        out = run_screen("bench/results/link-linux-downstream", REPO)
        caught = out.returncode == 1 and "artifact_predates_subject" in out.stdout
        record("REPLAYED", "rai012_citation_gap_is_caught", caught, f"rc={out.returncode}")

        # ...and the fix itself must be among the commits it names, or it caught some
        # other drift and the match is a coincidence.
        between = git(["log", "--format=%H", f"{old_sha}..{fix_sha}", "--", "rust/src"],
                      REPO).stdout.split()
        record("REPLAYED", "the_fix_is_inside_the_gap", fix_sha in between,
               f"{len(between)} commit(s) touched rust/src between the artifact and the fix")
    finally:
        if had_frame:
            shutil.copy2(backup, saved)
        elif saved.is_file():
            saved.unlink()


def live() -> None:
    art = REPO / "bench" / "results" / "link-linux-downstream"
    if not (art / "artifact-frame.json").is_file():
        record("LIVE", "declared_artifact_is_framed", False, "no frame present")
        return
    out = run_screen("bench/results/link-linux-downstream", REPO)
    record("LIVE", "declared_artifact_is_framed", out.returncode == 0,
           f"rc={out.returncode} (this arm is expected to go red when the EP changes and "
           "the Linux lane has not been re-run; that is the point)")


def main() -> int:
    print("NEGATIVE CONTROL: check_artifact_frame.py")
    print("")
    with tempfile.TemporaryDirectory(dir=str(REPO / "bench" / "results")) as td:
        tmp = Path(td)
        print(" planted arms (defect constructed on purpose):")
        planted(tmp)
        print("")
        print(" replayed arms (real events in this repository, not planted):")
        replayed(tmp)
    print("")
    print(" live arms (this repository as it stands):")
    live()

    bad = [r for r in RESULTS if not r[2]]
    kinds = {k: sum(1 for r in RESULTS if r[0] == k) for k in {r[0] for r in RESULTS}}
    print("")
    print(f"{len(RESULTS)} arm(s): " + ", ".join(f"{v} {k}" for k, v in sorted(kinds.items())))
    if bad:
        print(f"FAIL: {len(bad)} arm(s) did not behave as required:")
        for kind, name, _, detail in bad:
            print(f"  - {kind} {name}: {detail}")
        return 1
    print("PASS: every arm behaved as required. Read as: the screen fails when it should. "
          "Not read as: the screen catches every stale citation — subject_paths is a "
          "writer's declaration, so an artifact can still be stale w.r.t. a path nobody named.")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.exit(main())
