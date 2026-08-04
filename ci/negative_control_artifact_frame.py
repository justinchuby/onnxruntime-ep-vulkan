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
    (root / "src" / "lib.rs").write_text("fn a() {}\n", encoding="utf-8")
    (root / "art").mkdir()
    (root / "art" / "run.log").write_text("633 passed\n", encoding="utf-8")
    git(["add", "-A"], root)
    git(["commit", "-q", "-m", "base"], root)


def stamp(root: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCREEN), "art", "--repo", str(root), "--stamp",
         "--subject-paths", "src", "--platform", "negative control"],
        capture_output=True, text=True, encoding="utf-8",
    )


def touch_subject(root: Path, text: str) -> None:
    (root / "src" / "lib.rs").write_text(text, encoding="utf-8")
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
                f.name: __import__("hashlib").sha256(f.read_bytes()).hexdigest()
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
