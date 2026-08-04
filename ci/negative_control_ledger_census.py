#!/usr/bin/env python3
"""Negative control for `ci/check_ledger_census.py`: does it convict, and does it acquit?

A screen for a *deletion* is worth exactly what its false-negative rate is, and the failure it
hunts is one that three other instruments already missed (`--check`, the shrinking-write guard,
`check_open_reds.py`). So the arms below are weighted toward the ways it could be silently
useless, not toward the ways it could be noisy.

ARM KINDS, and the ratio is printed because a control made only of plants proves only that the
planting works:

  LIVE      run against the repository as it is now
  REPLAYED  run against a real historical revision, unmodified
  PLANTED   run against a synthetic repository built for the arm

The REPLAYED arms are the ones that matter. `eb84364` is a real merge in this repository that
really did drop three real `Cast` proofs, and nobody planted it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SCREEN = HERE / "check_ledger_census.py"

RESULTS: list[tuple[str, str, bool, str]] = []


def record(kind: str, name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((kind, name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {kind:9s} {name}" + (f" — {detail}" if detail else ""))


def run(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCREEN), *args],
        cwd=str(cwd or REPO),
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.update(
        GIT_AUTHOR_NAME="ctl", GIT_AUTHOR_EMAIL="ctl@example.invalid",
        GIT_COMMITTER_NAME="ctl", GIT_COMMITTER_EMAIL="ctl@example.invalid",
    )
    return subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, encoding="utf-8", env=env
    )


def _entry(key: str) -> str:
    return json.dumps({"key": key, "verdict": "MATCH", "shader_digest": "0" * 16}) + "\n"


def _plant_repo(tmp: Path, history: list[list[str]]) -> Path:
    """A throwaway git repo whose ledger takes each state in `history`, one commit per state."""
    repo = tmp / "planted"
    (repo / "evidence").mkdir(parents=True)
    _git(["init", "-q", "-b", "main"], repo)
    for state in history:
        (repo / "evidence" / "proof_ledger.jsonl").write_text(
            "".join(_entry(k) for k in state), encoding="utf-8"
        )
        _git(["add", "-A"], repo)
        _git(["commit", "-q", "-m", f"{len(state)} entries"], repo)
    return repo


def main() -> int:
    print("negative control: ci/check_ledger_census.py")

    # ── LIVE ────────────────────────────────────────────────────────────────────────
    r = run(["--json", os.devnull if os.name != "nt" else "NUL"])
    record("LIVE", "today's tree is green", r.returncode == 0, r.stdout.strip().splitlines()[-1][:90])

    r = run(["--list-retired"])
    record("LIVE", "--list-retired never fails on a valid file", r.returncode == 0)

    # ── REPLAYED: the real deletion, unplanted ──────────────────────────────────────
    cast_keys = {
        "ai.onnx::Cast/6+/f32>i32/ew_cast_f32_to_i32/static/n1",
        "ai.onnx::Cast/6+/i32>f32/ew_cast_i32_to_f32/static/n1",
        "ai.onnx::Cast/6+/f32>bool/ew_cast_f32_to_bool/static/n1",
    }
    r = run(["--at", "eb84364"])
    named = {k for k in cast_keys if k in r.stdout}
    record(
        "REPLAYED",
        "eb84364 (a real merge) is convicted",
        r.returncode == 1 and "proof_vanished" in r.stdout,
        f"exit={r.returncode}",
    )
    record(
        "REPLAYED",
        "and it names all three real Cast forms",
        named == cast_keys,
        f"{len(named)}/3 named",
    )
    record(
        "REPLAYED",
        "the count is 3, not 'some'",
        "3 VANISHED" in r.stdout,
    )

    # The commit that PROVED them must be clean: a screen that convicts every revision is a
    # constant, not a screen.
    r = run(["--at", "26fd93f"])
    record("REPLAYED", "26fd93f (the proving commit) is acquitted", r.returncode == 0)

    # THE ARM THAT CAUGHT THE SCREEN'S OWN DEFECT. Without `--full-history`, git's default
    # simplification hides 26fd93f from the ledger's own log and the replay above reports PASS
    # on the very merge it exists to convict. This asserts the denominator is the unsimplified
    # one, by the numbers, so the flag cannot be dropped as noise.
    simplified = _git(["rev-list", "eb84364", "--", "evidence/proof_ledger.jsonl"], REPO)
    full = _git(
        ["rev-list", "--full-history", "eb84364", "--", "evidence/proof_ledger.jsonl"], REPO
    )
    n_simpl = len(simplified.stdout.split())
    n_full = len(full.stdout.split())
    proving = _git(["rev-parse", "26fd93f"], REPO).stdout.strip()
    record(
        "REPLAYED",
        "history simplification really does hide the proving commit",
        proving not in simplified.stdout and proving in full.stdout and n_full > n_simpl,
        f"{n_simpl} simplified vs {n_full} full-history revisions",
    )

    # ── PLANTED ─────────────────────────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        repo = _plant_repo(tmp, [["a", "b", "c"], ["a", "b", "c", "d"]])
        r = run(["--repo", str(repo)], cwd=repo)
        record("PLANTED", "a ledger that only grows is green", r.returncode == 0)

        repo = _plant_repo(tmp / "1", [["a", "b", "c"], ["a", "c"]])
        r = run(["--repo", str(repo)], cwd=repo)
        record(
            "PLANTED",
            "one dropped key is convicted and named",
            r.returncode == 1 and "1 VANISHED" in r.stdout and "  - b\n" in r.stdout,
            f"exit={r.returncode}",
        )

        repo = _plant_repo(tmp / "2", [["a", "b"], []])
        r = run(["--repo", str(repo)], cwd=repo)
        record(
            "PLANTED",
            "an EMPTIED ledger is convicted (not read as 'no entries disagree')",
            r.returncode == 1 and "2 VANISHED" in r.stdout,
        )

        repo = _plant_repo(tmp / "3", [["a", "b"], ["a"], ["a", "b"]])
        r = run(["--repo", str(repo)], cwd=repo)
        record(
            "PLANTED",
            "a key removed and restored is green (this screen is not a history critic)",
            r.returncode == 0,
        )

        # A rewrite that keeps the COUNT but changes the KEYS. This is the arm that separates
        # this screen from a line counter: `entry_count` is unchanged, so a shrinking-write
        # guard sees nothing.
        repo = _plant_repo(tmp / "4", [["a", "b", "c"], ["a", "b", "z"]])
        r = run(["--repo", str(repo)], cwd=repo)
        record(
            "PLANTED",
            "same count, different keys, still convicted",
            r.returncode == 1 and "1 VANISHED" in r.stdout,
        )

        repo = _plant_repo(tmp / "5", [["a", "b"]])
        (repo / "evidence" / "proof_ledger.jsonl").unlink()
        r = run(["--repo", str(repo)], cwd=repo)
        record(
            "PLANTED",
            "a MISSING ledger is an error, not a PASS",
            r.returncode != 0 and "UNOBSERVABLE" in (r.stdout + r.stderr),
        )

    kinds = {}
    for kind, *_ in RESULTS:
        kinds[kind] = kinds.get(kind, 0) + 1
    passed = sum(1 for *_x, ok, _d in ((k, n, o, d) for k, n, o, d in RESULTS) if ok)
    total = len(RESULTS)
    print("")
    print(f"{passed}/{total} arms passed  ({', '.join(f'{v} {k}' for k, v in sorted(kinds.items()))})")
    if passed != total:
        print("FAIL — a control arm did not behave as declared")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
