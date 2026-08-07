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

TWO OF THESE ARMS ARE ABOUT A SECOND TOOL, ON PURPOSE
-----------------------------------------------------
The retirement exemption is granted by this screen AND by `rust/tools/gen_proof_ledger.py`, and
it is one exemption only if both answer from one register with one rule. They did not: one file,
two parsers, and a withdrawal with no owner was an exemption in the producer and a refusal here.
An arm that asks only this screen cannot see that, because a screen agreeing with itself is what
the divergence looked like from either side.
"""

from __future__ import annotations

import hashlib
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
    # PYTHONIOENCODING is not decoration. The retirement register carries `§` in its reasons, and
    # on a Windows runner the child writes its stdout in the console codepage while this decodes
    # utf-8 — the harness died in `_readerthread` before arm 1 recorded anything, which is a
    # negative control that reports nothing rather than a screen that reports wrong. Pinning both
    # ends to utf-8 makes the arms run on every lane instead of on the lanes with a friendly
    # locale.
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    return subprocess.run(
        [sys.executable, str(SCREEN), *args],
        cwd=str(cwd or REPO),
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
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


def _retire(repo: Path, rows: list[dict]) -> None:
    """Write the ONE retirement register into a planted repo, and commit it.

    Committed and not merely written, because the census walks committed history: a
    register left in the worktree would be read but its arrival would not be, and the
    difference matters for the same reason the frame-witness arms exist.
    """
    (repo / "evidence" / "retired_proof_keys.json").write_text(
        json.dumps({"retired": rows}, indent=2), encoding="utf-8"
    )
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "retire by name and reason"], repo)


def _write_rewitness(repo: Path, doc: dict) -> None:
    (repo / "evidence" / "proof_rewitness.json").write_text(
        json.dumps(doc, indent=2), encoding="utf-8"
    )
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "rewitness register"], repo)


def _declare(repo: Path, revision: str, owner: str, reason: str) -> None:
    _write_rewitness(
        repo,
        {
            "schema": 1,
            "screened_since": "",
            "rewitness": [
                {
                    "revision": revision,
                    "field": "source_digest",
                    "owner": owner,
                    "date": "2026-08-04",
                    "reason": reason,
                }
            ],
        },
    )


def _v2(field: str, caused_by: str, transitions: list[tuple[str, str, str]], **extra) -> dict:
    """A schema `rewitness/2` record: content-addressed, with no `revision` to erase."""
    rec = {
        "schema": "rewitness/2",
        "field": field,
        "owner": "switch",
        "date": "2026-08-06",
        "reason": "planted",
        "caused_by": caused_by,
        "transitions": [{"key": k, "old": o, "new": n} for k, o, n in transitions],
    }
    rec.update(extra)
    return rec


def _squash_merge(repo: Path, ledger: dict[str, str], decl: dict | None) -> str:
    """Replay the real squash-merge shape and return the ERASED branch head.

    On a branch: commit the ledger move, then commit the declaration describing it (an
    author cannot name a commit that does not exist yet, which is the whole problem). Then
    land the branch TREE on main as one brand-new commit and delete the branch, exactly as
    `gh pr merge --squash` does. The branch shas still RESOLVE — the objects are there —
    but they are no longer ancestors of anything, which is precisely the state that turned
    a correct #35 declaration into `stale_rewitness_declaration` + `undeclared_witness_move`.
    """
    base = _git(["rev-parse", "HEAD"], repo).stdout.strip()
    _git(["checkout", "-q", "-b", "feature"], repo)
    (repo / "evidence" / "proof_ledger.jsonl").write_text(
        "".join(
            json.dumps({"key": k, "verdict": "MATCH", "shader_digest": "0" * 16,
                        "source_digest": v, "toolchain": "planted"}) + "\n"
            for k, v in ledger.items()
        ),
        encoding="utf-8",
    )
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "branch: move the witness"], repo)
    branch_head = _git(["rev-parse", "HEAD"], repo).stdout.strip()
    if decl is not None:
        doc = decl(branch_head) if callable(decl) else decl
        (repo / "evidence" / "proof_rewitness.json").write_text(
            json.dumps({"schema": 1, "screened_since": "", "rewitness": [doc]}, indent=2),
            encoding="utf-8",
        )
        _git(["add", "-A"], repo)
        _git(["commit", "-q", "-m", "branch: declare the move"], repo)
    tree = _git(["rev-parse", "HEAD"], repo).stdout.strip()
    _git(["checkout", "-q", "main"], repo)
    _git(["checkout", "-q", tree, "--", "."], repo)
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "squash merge (new sha, branch history erased)"], repo)
    _git(["checkout", "-q", "--detach", "HEAD"], repo)
    _git(["branch", "-q", "-D", "feature"], repo)
    _git(["checkout", "-q", "main"], repo)
    return branch_head or base


def _plant_frames(tmp: Path, history: list[dict[str, str]]) -> Path:
    """A repo whose ledger keeps the SAME KEYS and moves only their frame witnesses.

    The key census reads 0 VANISHED on every state here, which is the whole point: the loss
    these arms are about is inside surviving entries, where every other screen is blind.
    """
    repo = tmp / "framed"
    (repo / "evidence").mkdir(parents=True)
    _git(["init", "-q", "-b", "main"], repo)
    for state in history:
        (repo / "evidence" / "proof_ledger.jsonl").write_text(
            "".join(
                json.dumps(
                    {
                        "key": k,
                        "verdict": "MATCH",
                        "shader_digest": "0" * 16,
                        "source_digest": v,
                        "toolchain": "planted",
                    }
                )
                + "\n"
                for k, v in state.items()
            ),
            encoding="utf-8",
        )
        _git(["add", "-A"], repo)
        _git(["commit", "-q", "-m", f"frames {sorted(state.items())}"], repo)
    return repo


T_COMP = "rust/shaders/glsl/templates/t.comp"
U_COMP = "rust/shaders/glsl/templates/u.comp"
HELPER = "rust/shaders/include/helper.glsl"
VARIANTS = "rust/src/ops/shader_variants.txt"
UNRELATED = "rust/src/registry.rs"


def _cid(text: str) -> str:
    """sha256 of the normalised bytes of `text` — computed HERE, not imported from the screen.

    Deliberately a second implementation. If these arms called `check_ledger_census.content_id`
    they would agree with the screen by construction, including on the day somebody changes
    what "normalised" means; two independent computations of the same number disagree the
    moment one of them drifts, which is the only version of this arm worth having.
    """
    data = text.encode("utf-8").replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def _sources(t_body: str, helper_body: str, defines_a: str = "D=1", **extra: str) -> dict:
    """A minimal but REAL shader tree: a variant manifest, two templates, one shared include.

    Real-shaped because the corroboration is not a string comparison — it re-derives the
    source closure from the tree (manifest row -> template -> `#include` graph) exactly as
    `rust/build_support/shader_source_digest.rs` does. An arm planted against a fake layout
    would exercise the record parser and none of the reasoning that makes the record mean
    anything.
    """
    files = {
        VARIANTS: (
            "# stem\tsource\tdefines\n"
            f"stem_a\ttemplates/t.comp\t{defines_a}\n"
            "stem_b\ttemplates/u.comp\tD=2\n"
        ),
        T_COMP: t_body,
        U_COMP: '#version 450\nvoid main() { /* unrelated to stem_a */ }\n',
        HELPER: helper_body,
        UNRELATED: "pub fn register() {}\n",
    }
    files.update(extra)
    return files


def _write_sourced(repo: Path, files: dict, ledger: dict, newline: str = "\n") -> None:
    """Write a source tree + a ledger whose entries name the stems they were proven from."""
    for rel, body in files.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(body.replace("\n", newline).encode("utf-8"))
    (repo / "evidence").mkdir(parents=True, exist_ok=True)
    (repo / "evidence" / "proof_ledger.jsonl").write_text(
        "".join(
            json.dumps({
                "key": k, "verdict": "MATCH", "shader_digest": "0" * 16,
                "source_digest": digest, "toolchain": "planted", "shaders": list(stems),
            }) + "\n"
            for k, (digest, stems) in ledger.items()
        ),
        encoding="utf-8",
    )


def _plant_sourced(tmp: Path, files: dict, ledger: dict) -> Path:
    repo = tmp
    repo.mkdir(parents=True)
    _git(["init", "-q", "-b", "main"], repo)
    _write_sourced(repo, files, ledger)
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "base state"], repo)
    return repo


def _land_sourced(repo: Path, mode: str, files: dict, ledger: dict, records,
                  newline: str = "\n") -> str:
    """Move source + witness on a branch, declare it, and land it the way `mode` lands.

    -> the branch tip, which every mode but `merge` ERASES from the graph. `main` is advanced
    by one unrelated commit first, because that is both the realistic case (main moves while a
    PR is open) and the only way the rebase arm means anything: replaying a commit onto its own
    unchanged parent reproduces the identical sha, so a rebase with no divergence is a rebase
    that erased nothing and an arm that tested nothing. It silently passed exactly once.

    The landing is then ASSERTED rather than assumed: the witness must really have moved at
    HEAD, and the branch tip must really be unreachable. A control harness whose plant quietly
    failed to plant reports green for the same reason the defect it hunts does.
    """
    base = _git(["rev-parse", "HEAD"], repo).stdout.strip()
    _git(["checkout", "-q", "-b", "feature"], repo)
    _write_sourced(repo, files, ledger, newline)
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "branch: move the source and the witness together"], repo)
    if records is not None:
        recs = records if isinstance(records, list) else [records]
        _write_rewitness(repo, {"schema": 1, "screened_since": "", "rewitness": recs})
    tip = _git(["rev-parse", "HEAD"], repo).stdout.strip()

    _git(["checkout", "-q", "main"], repo)
    (repo / "NOTES.md").write_text("main advanced while the PR was open\n", encoding="utf-8")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "main: an unrelated commit, landed first"], repo)

    if mode == "squash":
        _git(["checkout", "-q", tip, "--", "."], repo)
        _git(["add", "-A"], repo)
        _git(["commit", "-q", "-m", "squash landing (new sha, branch history erased)"], repo)
        _git(["branch", "-q", "-D", "feature"], repo)
    elif mode == "merge":
        r = _git(["merge", "-q", "--no-ff", "--no-edit", "-m", "merge landing", "feature"], repo)
        assert r.returncode == 0, f"merge landing failed: {r.stdout}{r.stderr}"
    elif mode == "rebase":
        r = _git(["cherry-pick", f"{base}..feature"], repo)
        assert r.returncode == 0, f"rebase landing failed: {r.stdout}{r.stderr}"
        _git(["branch", "-q", "-D", "feature"], repo)
    else:
        raise AssertionError(f"unknown landing {mode!r}")

    shipped = (repo / "evidence" / "proof_ledger.jsonl").read_text(encoding="utf-8")
    for key, (digest, _stems) in ledger.items():
        assert f'"{digest}"' in shipped, f"{mode} landing did not carry {key}'s new witness"
    if mode != "merge":
        assert _git(["merge-base", "--is-ancestor", tip, "HEAD"], repo).returncode != 0, (
            f"the {mode} landing left the branch tip reachable; it erased nothing, so any "
            "arm about erasure would pass without testing anything"
        )
    return tip


def _v3(transitions: list[tuple[str, str, str]], paths: list[tuple[str, str, str]],
        **extra) -> dict:
    """A schema `rewitness/3` record: the cause is CONTENT, so no landing can erase it."""
    rec = {
        "schema": "rewitness/3",
        "field": "source_digest",
        "owner": "tank",
        "date": "2026-08-06",
        "reason": "planted",
        "caused_by_content": {
            "kind": "same_change",
            "paths": [{"path": p, "old": o, "new": n} for p, o, n in paths],
        },
        "transitions": [{"key": k, "old": o, "new": n} for k, o, n in transitions],
    }
    rec.update(extra)
    return rec


def _repath(rec: dict, path: str, old: str | None = None, new: str | None = None) -> dict:
    """A copy of `rec` with its FIRST cause path rewritten — one planted lie per arm."""
    out = json.loads(json.dumps(rec))
    p = out["caused_by_content"]["paths"][0]
    p["path"] = path
    if old is not None:
        p["old"] = old
    if new is not None:
        p["new"] = new
    return out


def _dup_path(rec: dict) -> dict:
    out = json.loads(json.dumps(rec))
    paths = out["caused_by_content"]["paths"]
    paths.append(json.loads(json.dumps(paths[0])))
    return out


def _both_readers():
    """The two tools that grant the retirement exemption, imported in process.

    `check_ledger_census.py` is the consumer and `rust/tools/gen_proof_ledger.py` is the
    producer, and the exemption is only ONE exemption if they answer from one module. Imported
    rather than shelled out because the claim under test is *which code reads the register* —
    a subprocess could only show that the two agree on one input, which two divergent parsers
    also do until the input separates them.
    """
    sys.path.insert(0, str(REPO / "ci"))
    sys.path.insert(0, str(REPO / "rust" / "tools"))
    import gen_proof_ledger  # noqa: PLC0415
    import proof_retirement  # noqa: PLC0415

    return gen_proof_ledger, proof_retirement


def _synthetic_pr_merge(tmp: Path) -> tuple[Path, str]:
    """Build a repo whose HEAD is a GitHub-style two-parent `pull/N/merge` preview:
    the current tip of `main` merged with a PR head, full history reachable from HEAD.

    THIS IS ISSUE #28. A real `refs/pull/N/merge` ref is GitHub's synthetic merge of a PR
    branch INTO THE CURRENT TIP OF THE BASE BRANCH, so its first parent is always exactly
    whatever `origin/main` resolves to at fetch time. `git fetch --depth=1 origin main`,
    run later in the same CI job for an unrelated lookup, grafts that exact commit —
    regardless of the job's own `fetch-depth: 0` checkout already holding its full
    history — because a depth-limited fetch marks its own boundary unconditionally. Since
    the graft point is one of HEAD's own two parents, `git rev-parse
    --is-shallow-repository` goes true for the WHOLE repository, not just that one ref.

    Returns `(repo, main_tip)` so a caller can replay the exact poisoning fetch against
    `main_tip` and prove two things at once: a genuine, unpoisoned synthetic merge is
    OBSERVABLE and green (the merge shape itself is not the defect); the identical tree,
    incidentally shallow-marked the way the real CI job shallow-marks it, is refused
    rather than guessed at (the shallow guard is not weakened to accommodate the fix).
    """
    repo = _plant_repo(tmp, [["a"], ["a", "b"]])  # c1: [a]; c2 (main tip): [a, b]
    main_tip = _git(["rev-parse", "HEAD"], repo).stdout.strip()
    _git(["checkout", "-q", "-b", "prhead", "HEAD~1"], repo)
    (repo / "evidence" / "proof_ledger.jsonl").write_text(
        "".join(_entry(k) for k in ["a", "c"]), encoding="utf-8"
    )
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "pr proves c"], repo)
    pr_head = _git(["rev-parse", "HEAD"], repo).stdout.strip()
    _git(["checkout", "-q", "main"], repo)
    _git(
        ["merge", "-q", "--no-ff", "--no-edit", "-m", f"Merge {pr_head} into {main_tip}", "prhead"],
        repo,
    )
    return repo, main_tip


def main() -> int:
    print("negative control: ci/check_ledger_census.py")

    # ── LIVE ────────────────────────────────────────────────────────────────────────
    r = run(["--json", os.devnull if os.name != "nt" else "NUL"])
    record(
        "LIVE",
        "today's tree is green",
        r.returncode == 0,
        ((r.stdout or "").strip().splitlines() or ["<no output>"])[-1][:90],
    )

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

    # THE FRAME-WITNESS ARM, REPLAYED ON A REAL EVENT. `eee65aa` really did put 115
    # withdrawn source digests back, nobody planted it, and no key went missing — so the
    # census arms above are all green at that revision and every other screen in the
    # repository read clean. The screen must still be RULING on that revision: a record is
    # not a deletion, so the declaration makes it legible rather than silent.
    r = run(["--at", "eee65aab6cc8940bfdbaeda562106f5aaa519f30"])
    record(
        "REPLAYED",
        "eee65aa's 115 witness moves are named, and no key vanished there",
        "DECLARED ACCIDENT: eee65aab6cc8" in r.stdout
        and "115 entr(ies)" in r.stdout
        and "0 VANISHED" in r.stdout,
        f"exit={r.returncode}",
    )

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

        # ── the retirement arms ────────────────────────────────────────────────────
        # THE GAP THIS CLOSES. Every arm above and below plants or replays a DELETION.
        # Not one of them exercised the EXEMPTION: key gone, withdrawal recorded, screen
        # stays green and names it. So `evidence/retired_proof_keys.json` had 43 entries
        # that no green run had ever read — and when the screen was pointed at a register
        # filename that did not exist, all 43 came back as VANISHED and nothing here
        # noticed, because nothing here had a positive state for a retirement. A branch
        # with no positive state, in the tool built to have positive states.
        repo = _plant_repo(tmp / "5r", [["a", "b", "c"], ["a", "c"]])
        _retire(repo, [{"key": "b", "owner": "link", "date": "2026-08-04",
                        "reason": "planted: the form was withdrawn on purpose"}])
        r = run(["--repo", str(repo)], cwd=repo)
        record(
            "PLANTED",
            "a retirement ACQUITS the same deletion arm 1 convicts",
            r.returncode == 0 and "0 VANISHED" in r.stdout and "1 retired" in r.stdout,
            f"exit={r.returncode}",
        )
        record(
            "PLANTED",
            "and the acquitted key is NAMED, not silently absorbed",
            "- b  [link 2026-08-04:" in r.stdout,
        )

        # A retirement is not a suppression list: it must carry the same three fields an
        # accepted red carries. A withdrawal nobody has to sign is a blanket exemption. The
        # register is present and unreadable, which is ERROR(instrument) and never a colour:
        # degrading it to "nothing is retired" would report every deliberate withdrawal in it
        # as VANISHED, and that is the exact 43-key false positive this screen once produced.
        repo = _plant_repo(tmp / "5s", [["a", "b", "c"], ["a", "c"]])
        _retire(repo, [{"key": "b", "reason": "planted: no owner, no date"}])
        r = run(["--repo", str(repo)], cwd=repo)
        record(
            "PLANTED",
            "a retirement with no owner is REFUSED, not honoured",
            r.returncode == 2
            and "unreadable_retirement_register" in r.stdout
            and "missing" in r.stdout
            and "ever proven" not in r.stdout,
            f"exit={r.returncode}",
        )

        # ── the ONE-REGISTER arms ──────────────────────────────────────────────────
        # Pointing both tools at one FILENAME is not one register. Until 2026-08-05 this
        # screen and `rust/tools/gen_proof_ledger.py` — the ledger's producer, and the other
        # tool that grants this exemption — each parsed that file with its own rule: this one
        # required `owner`/`date`/`reason`, the producer required only `reason`. A withdrawal
        # nobody signed was therefore an exemption in the producer and a refusal here, which is
        # the same "two registers, one fact" shape as the 43 false positives, arriving through
        # the schema instead of through the path. These arms plant ONE file and ask BOTH
        # readers, because that is the only question that can tell one register from two.
        producer, census_reader = _both_readers()
        record(
            "LIVE",
            "the producer and this screen read the SAME register module",
            producer.proof_retirement is census_reader
            and Path(producer.RETIRED) == (REPO / census_reader.RETIRED_REL),
            f"{Path(producer.RETIRED).name}",
        )

        signed = tmp / "one_register_signed.json"
        signed.write_text(
            json.dumps({"retired": [{"key": "b", "owner": "tank", "date": "2026-08-05",
                                     "reason": "planted: withdrawn on purpose"}]}),
            encoding="utf-8",
        )
        keys, err = producer._retired_keys(signed)
        record(
            "PLANTED",
            "a signed withdrawal is honoured by BOTH readers",
            not err and set(keys) == {"b"} and set(census_reader.load(signed)) == {"b"},
            err or "b exempt in both",
        )

        unsigned = tmp / "one_register_unsigned.json"
        unsigned.write_text(
            json.dumps({"retired": [{"key": "b", "reason": "planted: no owner, no date"}]}),
            encoding="utf-8",
        )
        keys, err = producer._retired_keys(unsigned)
        refused_by_census = False
        try:
            census_reader.load(unsigned)
        except census_reader.RetirementError:
            refused_by_census = True
        record(
            "PLANTED",
            "an UNSIGNED withdrawal is refused by BOTH, and exempts nothing in either",
            bool(err) and "owner" in err and not keys and refused_by_census,
            err[:70] or "the producer honoured it",
        )

        # And the producer's refusal must be an ERROR, never an empty exemption set: `{}` there
        # would turn every withdrawal in a malformed register into a lost proof, which is the
        # 43-VANISHED report with the polarity flipped. The census arm for the same register is
        # `5s` above — one file, one rule, asserted from both ends.

        # A retirement that agrees with nothing is a false record of the file's contents.
        repo = _plant_repo(tmp / "5t", [["a", "b"], ["a", "b"]])
        _retire(repo, [{"key": "b", "owner": "link", "date": "2026-08-04",
                        "reason": "planted: withdrawn, and yet here it is"}])
        r = run(["--repo", str(repo)], cwd=repo)
        record(
            "PLANTED",
            "a key recorded as withdrawn and STILL PRESENT is convicted",
            r.returncode == 1 and "retired_but_present" in r.stdout,
            f"exit={r.returncode}",
        )

        # Two registers for one fact is the defect the 43 false positives came from, and
        # it is an instrument failure rather than a finding: with both files present the
        # screen cannot say which one is the record, so it has observed nothing.
        repo = _plant_repo(tmp / "5u", [["a", "b", "c"], ["a", "c"]])
        _retire(repo, [{"key": "b", "owner": "link", "date": "2026-08-04",
                        "reason": "planted"}])
        (repo / "evidence" / "proof_retired.json").write_text(
            json.dumps({"retired": {"b": {"owner": "someone", "date": "2026-08-04",
                                          "reason": "the register that never existed"}}}),
            encoding="utf-8",
        )
        _git(["add", "-A"], repo)
        _git(["commit", "-q", "-m", "second register"], repo)
        r = run(["--repo", str(repo)], cwd=repo)
        record(
            "PLANTED",
            "TWO retirement registers is an instrument ERROR, never a colour",
            r.returncode == 2 and "two_retirement_registers" in r.stdout,
            f"exit={r.returncode}",
        )

        # A retirement written TODAY cannot govern a deletion that happened BEFORE it.
        # This is the arm that made the replay honest: with the register read from the
        # worktree, `--at eb84364` exited 1 for `retired_but_present` — today's 43
        # withdrawals applied to a revision where those keys were still present — and
        # returned before the three real vanished Cast forms were ever printed. Convicting
        # the right revision for the wrong reason is the same defect as acquitting it, so
        # `--at` now reads the register at the same revision as the ledger, exactly as
        # `screened_since` refuses to rule on history a screen predates.
        repo = _plant_repo(tmp / "5w", [["a", "b", "c"], ["a", "c"]])
        before = _git(["rev-parse", "HEAD"], repo).stdout.strip()
        _retire(repo, [{"key": "b", "owner": "link", "date": "2026-08-04",
                        "reason": "planted: withdrawn AFTER the revision under test"}])
        r = run(["--repo", str(repo)], cwd=repo)
        record(
            "PLANTED",
            "the retirement acquits the revision that FOLLOWS it",
            r.returncode == 0 and "0 VANISHED" in r.stdout,
            f"exit={r.returncode}",
        )
        r = run(["--repo", str(repo), "--at", before], cwd=repo)
        record(
            "PLANTED",
            "and does NOT reach back to acquit the revision that precedes it",
            r.returncode == 1 and "proof_vanished" in r.stdout
            and "retired_but_present" not in r.stdout,
            f"exit={r.returncode}",
        )
        record(
            "PLANTED",
            "and the replay SAYS which moment's register it read",
            f"read at {before} (0 record(s))" in r.stdout,
        )

        # ── the frame-witness arm ──────────────────────────────────────────────────
        # Same key set throughout, so the census above reads 0 VANISHED on every arm here:
        # the whole point is that the loss is INSIDE surviving entries.
        repo = _plant_frames(tmp / "6", [{"a": "X", "b": "X"}, {"a": "X", "b": "X"}])
        r = run(["--repo", str(repo)], cwd=repo)
        record("PLANTED", "witnesses that never move are green", r.returncode == 0)

        repo = _plant_frames(tmp / "7", [{"a": "X"}, {"a": "Y"}])
        r = run(["--repo", str(repo)], cwd=repo)
        record(
            "PLANTED",
            "an UNDECLARED witness move is convicted (0 keys vanished)",
            r.returncode == 1
            and "undeclared_witness_move" in r.stdout
            and "0 VANISHED" in r.stdout,
            f"exit={r.returncode}",
        )

        repo = _plant_frames(tmp / "8", [{"a": "X"}, {"a": "Y"}])
        head = _git(["rev-parse", "HEAD"], repo).stdout.strip()
        _declare(repo, head, "link", "planted: a deliberate re-witness")
        r = run(["--repo", str(repo)], cwd=repo)
        record(
            "PLANTED",
            "the SAME move, declared, is green — the arm is about the record, not the value",
            r.returncode == 0,
            f"exit={r.returncode}",
        )

        # The reversion shape that actually happened, and the reason this arm does not ask
        # "did the value go back": X -> Y -> X is symmetric, and only the declaration
        # distinguishes the repair from the regression.
        repo = _plant_frames(tmp / "9", [{"a": "X"}, {"a": "Y"}, {"a": "X"}])
        r = run(["--repo", str(repo)], cwd=repo)
        record(
            "PLANTED",
            "X->Y->X (the real 2026-08-04 shape) is convicted for BOTH undeclared moves",
            r.returncode == 1 and "undeclared_witness_move" in r.stdout,
        )

        repo = _plant_frames(tmp / "10", [{"a": "X"}, {"a": "X"}])
        _declare(repo, _git(["rev-parse", "HEAD"], repo).stdout.strip(), "link", "planted")
        r = run(["--repo", str(repo)], cwd=repo)
        record(
            "PLANTED",
            "a declaration matching no move is convicted (the register does not rot)",
            r.returncode == 1 and "stale_rewitness_declaration" in r.stdout,
        )

        repo = _plant_frames(tmp / "11", [{"a": "X"}, {"a": "Y"}])
        _write_rewitness(repo, {"schema": 1, "screened_since": "deadbeef" * 5, "rewitness": []})
        r = run(["--repo", str(repo)], cwd=repo)
        record(
            "PLANTED",
            "an unresolvable screened_since is an instrument ERROR, not a PASS",
            r.returncode == 2 and "does not resolve" in r.stdout,
            f"exit={r.returncode}",
        )

        # THE ARM FOR THE DEFECT THIS SCREEN HAD ON ITS FIRST RUN. A boundary that does not
        # resolve puts every transition out of frame; the first version printed PASS.
        assert "0 UNDECLARED" not in r.stdout, "an unusable boundary must not report a colour"

        repo = _plant_frames(tmp / "12", [{"a": "X"}, {"a": "Y"}])
        _git(["clone", "-q", "--depth", "1", "--no-local", repo.as_uri(), str(tmp / "shallow")], tmp)
        r = run(["--repo", str(tmp / "shallow")], cwd=tmp / "shallow")
        record(
            "PLANTED",
            "a SHALLOW clone is an instrument ERROR, not a PASS",
            r.returncode == 2 and "SHALLOW" in r.stdout,
            f"exit={r.returncode}",
        )

        # ── the squash-safe (rewitness/2) declaration arm ──────────────────────────
        # Every arm below is about ONE claim: a declaration must survive the merge that
        # lands it. v1 could not, and these arms hold both polarities of that.
        #
        # These arms plant REAL-SHAPED digests (16 lowercase hex) rather than the "X"/"Y"
        # placeholders above, because v2 validates digest shape: a malformed digest can
        # never match a real transition, so accepting one would let a record declare
        # nothing while looking like a declaration.
        HX, HY, HZ, HW = "a" * 16, "b" * 16, "c" * 16, "d" * 16
        HP, HQ, HM, HN = "e" * 16, "f" * 16, "0" * 16, "1" * 16

        # THE DEFECT, REPRODUCED. A correct, timely v1 declaration naming the branch
        # commit that carries the move — erased by the squash, so the register now reports
        # BOTH failures for a move that was declared in advance and in good faith.
        repo = _plant_frames(tmp / "15", [{"a": HX}])
        _squash_merge(
            repo, {"a": HY},
            lambda b: {"revision": b, "field": "source_digest", "owner": "link",
                       "date": "2026-08-06", "reason": "planted: declared before the merge"},
        )
        r = run(["--repo", str(repo)], cwd=repo)
        record(
            "PLANTED",
            "a v1 declaration naming its own branch commit SELF-INVALIDATES on squash",
            r.returncode == 1
            and "undeclared_witness_move" in r.stdout
            and "stale_rewitness_declaration" not in r.stdout,
            f"exit={r.returncode}",
        )

        # THE FIX, UNDER THE IDENTICAL SHAPE. Same repo shape, same squash, same erased
        # branch — the only difference is that the declaration is content-addressed, so
        # there is nothing for the squash to erase.
        repo = _plant_frames(tmp / "16", [{"a": HX}])
        base = _git(["rev-parse", "HEAD"], repo).stdout.strip()
        _squash_merge(repo, {"a": HY},
                      lambda _b: _v2("source_digest", base, [("a", HX, HY)]))
        r = run(["--repo", str(repo)], cwd=repo)
        record(
            "PLANTED",
            "the SAME squash with a rewitness/2 declaration is green — no re-point needed",
            r.returncode == 0 and "0 UNDECLARED" in r.stdout,
            f"exit={r.returncode}",
        )

        # A v2 record must not become a wildcard. Each arm plants exactly one wrong thing.
        for name, trans, expect in [
            ("a WRONG `old` declares a move that never happened", [("a", HW, HY)],
             "undeclared_witness_move"),
            ("a WRONG `new` is caught the same way", [("a", HX, HZ)],
             "undeclared_witness_move"),
        ]:
            repo = _plant_frames(tmp / f"17-{expect}-{trans[0][1][:2]}", [{"a": HX}])
            base = _git(["rev-parse", "HEAD"], repo).stdout.strip()
            _squash_merge(repo, {"a": HY}, lambda _b, t=trans: _v2("source_digest", base, t))
            r = run(["--repo", str(repo)], cwd=repo)
            record("PLANTED", name, r.returncode == 1 and expect in r.stdout,
                   f"exit={r.returncode}")

        # A MISSING key: two moves, one declared. The declared one must not vouch for the
        # other — which a bare `keys: 2` count would have let it do.
        repo = _plant_frames(tmp / "18", [{"a": HX, "b": HP}])
        base = _git(["rev-parse", "HEAD"], repo).stdout.strip()
        _squash_merge(repo, {"a": HY, "b": HQ},
                      lambda _b: _v2("source_digest", base, [("a", HX, HY)]))
        r = run(["--repo", str(repo)], cwd=repo)
        record(
            "PLANTED",
            "declaring 1 of 2 moves convicts the OTHER one (enumeration, not a count)",
            r.returncode == 1
            and "undeclared_witness_move" in r.stdout
            and "moved on 1 entr" in r.stdout
            and "e.g. b\n" in r.stdout,
            f"exit={r.returncode}",
        )

        # AN EXTRA key: everything real is declared, plus one row that matches nothing.
        # This is the arm a bare count cannot have, and it is the one that catches a row
        # copied in from another move.
        repo = _plant_frames(tmp / "19", [{"a": HX}])
        base = _git(["rev-parse", "HEAD"], repo).stdout.strip()
        _squash_merge(
            repo, {"a": HY},
            lambda _b: _v2("source_digest", base, [("a", HX, HY), ("ghost", HM, HN)]),
        )
        r = run(["--repo", str(repo)], cwd=repo)
        record(
            "PLANTED",
            "an OVER-declared transition inside a matching record is convicted",
            r.returncode == 1 and "overdeclared_witness_move" in r.stdout and "ghost" in r.stdout,
            f"exit={r.returncode}",
        )

        # `caused_by` is the one revision v2 still reads, so it gets the treatment v1's
        # `revision` could not survive: it must be LANDED. A branch-only sha resolves and
        # must still fail, or the schema would reintroduce its own defect through the
        # single field it kept.
        repo = _plant_frames(tmp / "20", [{"a": HX}])
        erased = _squash_merge(repo, {"a": HY}, None)
        _write_rewitness(repo, {"schema": 1, "screened_since": "",
                                "rewitness": [_v2("source_digest", erased, [("a", HX, HY)])]})
        r = run(["--repo", str(repo)], cwd=repo)
        record(
            "PLANTED",
            "a BRANCH-ONLY `caused_by` resolves but is convicted as not landed",
            r.returncode == 1 and "unlanded_rewitness_cause" in r.stdout,
            f"exit={r.returncode}",
        )

        repo = _plant_frames(tmp / "21", [{"a": HX}, {"a": HY}])
        _write_rewitness(repo, {"schema": 1, "screened_since": "",
                                "rewitness": [_v2("source_digest", "b" * 40, [("a", HX, HY)])]})
        r = run(["--repo", str(repo)], cwd=repo)
        record(
            "PLANTED",
            "a `caused_by` that does not resolve at all is convicted, not ignored",
            r.returncode == 1 and "unlanded_rewitness_cause" in r.stdout,
            f"exit={r.returncode}",
        )

        # Schema errors are UNOBSERVABLE (exit 2), never a colour: a register the checker
        # could not parse has ruled on nothing, and printing PASS over it is the failure
        # this whole file exists to prevent.
        head = None
        for name, rec_doc, needle in [
            ("an UNKNOWN schema is refused, never treated as v1",
             {"schema": "rewitness/99", "field": "source_digest", "owner": "x",
              "date": "d", "reason": "r", "caused_by": "HEAD",
              "transitions": [{"key": "a", "old": "0" * 16, "new": "1" * 16}]},
             "does not know"),
            ("a v2 record MISSING `caused_by` is refused",
             {"schema": "rewitness/2", "field": "source_digest", "owner": "x",
              "date": "d", "reason": "r",
              "transitions": [{"key": "a", "old": "0" * 16, "new": "1" * 16}]},
             "is missing"),
            ("a `keys` count disagreeing with the enumeration is refused",
             {"schema": "rewitness/2", "field": "source_digest", "owner": "x",
              "date": "d", "reason": "r", "caused_by": "HEAD", "keys": 7,
              "transitions": [{"key": "a", "old": "0" * 16, "new": "1" * 16}]},
             "enumerates"),
            ("a transition with old == new is refused (it declares a non-event)",
             {"schema": "rewitness/2", "field": "source_digest", "owner": "x",
              "date": "d", "reason": "r", "caused_by": "HEAD",
              "transitions": [{"key": "a", "old": "0" * 16, "new": "0" * 16}]},
             "old == new"),
            ("a malformed digest is refused rather than left unmatchable",
             {"schema": "rewitness/2", "field": "source_digest", "owner": "x",
              "date": "d", "reason": "r", "caused_by": "HEAD",
              "transitions": [{"key": "a", "old": "nope", "new": "1" * 16}]},
             "16-hex-digit"),
            ("an extra field inside a transition is refused, not ignored",
             {"schema": "rewitness/2", "field": "source_digest", "owner": "x",
              "date": "d", "reason": "r", "caused_by": "HEAD",
              "transitions": [{"key": "a", "old": "0" * 16, "new": "1" * 16, "why": "?"}]},
             "exactly key/old/new"),
        ]:
            repo = _plant_frames(tmp / f"22-{needle[:6]}", [{"a": HX}, {"a": HY}])
            head = _git(["rev-parse", "HEAD"], repo).stdout.strip()
            doc = json.loads(json.dumps(rec_doc).replace('"HEAD"', json.dumps(head)))
            _write_rewitness(repo, {"schema": 1, "screened_since": "", "rewitness": [doc]})
            r = run(["--repo", str(repo)], cwd=repo)
            record("PLANTED", name,
                   r.returncode == 2 and needle in r.stdout,
                   f"exit={r.returncode}")

        # A DUPLICATE transition, across two records. One real move must not be able to
        # consume two declarations, because the second one then vouches for nothing while
        # looking like it vouches for something.
        repo = _plant_frames(tmp / "23", [{"a": HX}, {"a": HY}])
        head = _git(["rev-parse", "HEAD"], repo).stdout.strip()
        dup = _v2("source_digest", head, [("a", HX, HY)])
        _write_rewitness(repo, {"schema": 1, "screened_since": "",
                                "rewitness": [dup, json.loads(json.dumps(dup))]})
        r = run(["--repo", str(repo)], cwd=repo)
        record(
            "PLANTED",
            "a DUPLICATE transition (across records) is refused",
            r.returncode == 2 and "more than once" in r.stdout,
            f"exit={r.returncode}",
        )

        # BACKWARDS COMPATIBILITY. Every record already in the real register is v1, so a
        # v1 record that still matches must stay green, and v1 and v2 must coexist in one
        # file. Migration is opt-in per record, not a flag day.
        repo = _plant_frames(tmp / "24", [{"a": HX, "b": HP}, {"a": HY, "b": HQ}])
        head = _git(["rev-parse", "HEAD"], repo).stdout.strip()
        _write_rewitness(repo, {
            "schema": 1, "screened_since": "",
            "rewitness": [
                {"revision": head, "field": "source_digest", "owner": "link",
                 "date": "2026-08-06", "reason": "planted v1, still matching"},
            ],
        })
        r = run(["--repo", str(repo)], cwd=repo)
        record(
            "PLANTED",
            "an UNMIGRATED v1 record that still matches stays green (no flag day)",
            r.returncode == 0,
            f"exit={r.returncode}",
        )

        repo = _plant_frames(tmp / "25", [{"a": HX, "b": HP}, {"a": HY, "b": HQ}])
        head = _git(["rev-parse", "HEAD"], repo).stdout.strip()
        parent = _git(["rev-parse", "HEAD~1"], repo).stdout.strip()
        _write_rewitness(repo, {
            "schema": 1, "screened_since": "",
            "rewitness": [
                {"revision": head, "field": "source_digest", "owner": "link",
                 "date": "2026-08-06", "reason": "planted v1 covering the revision"},
                _v2("source_digest", parent, [("b", HP, HQ)]),
            ],
        })
        r = run(["--repo", str(repo)], cwd=repo)
        record(
            "PLANTED",
            "v1 and v2 records coexist in one register without either shadowing the other",
            r.returncode == 0,
            f"exit={r.returncode}",
        )

        # ── rewitness/3: a same-change cause, corroborated against production source ──
        #
        # WHY THERE IS A THIRD SCHEMA, IN ONE PARAGRAPH. `rewitness/2` fixed v1's erasure
        # problem by content-addressing the TRANSITIONS, but it still names the cause with a
        # commit sha, and it requires that sha to be landed. For a witness that moves because
        # of an edit in the SAME change, no such sha exists at authoring time: the only
        # honest answer is the branch commit, which a squash and a rebase both erase. PR #53
        # was green on its branch, green under a merge commit, and red on `main` after a
        # squash. `rewitness/3` names the cause as CONTENT — the exact old/new of the shader
        # source the witness is a hash of — which every landing preserves because every
        # landing preserves the tree.
        #
        # These arms are the reason to believe that. The green ones are run under all three
        # landings; the red ones are each ONE planted lie, because a screen that convicts a
        # tree with nine defects in it has shown nothing about which defect it can see.
        T1 = '#version 450\n#include "helper.glsl"\nvoid main() { tile(TILE); }\n'
        T2 = '#version 450\n#include "helper.glsl"\nvoid main() { tile(TILE); prefill(); }\n'
        U2 = '#version 450\nvoid main() { /* edited, and nothing to do with stem_a */ }\n'
        H1, H2 = "#define TILE 1\n", "#define TILE 4\n"
        BEFORE = _sources(T1, H1)
        AFTER = _sources(T2, H1)
        LED_BEFORE = {"a": (HX, ["stem_a"]), "b": (HP, ["stem_b"])}
        LED_AFTER = {"a": (HY, ["stem_a"]), "b": (HP, ["stem_b"])}
        GOOD_TRANS = [("a", HX, HY)]
        GOOD_PATHS = [(T_COMP, _cid(T1), _cid(T2))]

        for mode in ("squash", "merge", "rebase"):
            repo = _plant_sourced(tmp / f"26-{mode}", BEFORE, LED_BEFORE)
            _land_sourced(repo, mode, AFTER, LED_AFTER, _v3(GOOD_TRANS, GOOD_PATHS))
            r = run(["--repo", str(repo)], cwd=repo)
            record(
                "PLANTED",
                f"a rewitness/3 same-change cause is green under a {mode.upper()} landing",
                r.returncode == 0 and "0 uncorroborated" in r.stdout,
                f"exit={r.returncode}",
            )

        # THE DEFECT THIS SCHEMA REMOVES, IN THREE ROWS. The same true declaration, written
        # the only way v2 allows — naming the branch commit that really made the change.
        # It is green under one landing and red under the other two, which is a screen whose
        # colour is decided by a maintainer's choice of merge button.
        for mode, expect_rc in (("squash", 1), ("rebase", 1), ("merge", 0)):
            repo = _plant_sourced(tmp / f"27-{mode}", BEFORE, LED_BEFORE)
            tip = _land_sourced(repo, mode, AFTER, LED_AFTER, None)
            _write_rewitness(repo, {"schema": 1, "screened_since": "",
                                    "rewitness": [_v2("source_digest", tip, GOOD_TRANS)]})
            r = run(["--repo", str(repo)], cwd=repo)
            record(
                "PLANTED",
                f"the SAME cause written as v2 (branch sha) is "
                + ("REFUSED" if expect_rc else "accepted")
                + f" under a {mode.upper()} landing — landing-dependence, reproduced",
                r.returncode == expect_rc
                and (expect_rc == 0 or "unlanded_rewitness_cause" in r.stdout),
                f"exit={r.returncode}",
            )

        # Each arm below plants exactly ONE wrong thing into an otherwise correct record,
        # and lands it with a squash, the landing this repository actually uses.
        for name, files_after, led_after, trans, paths, needle in [
            (
                "a WRONG `new` content id is convicted (the cause is not in the tree)",
                AFTER, LED_AFTER, GOOD_TRANS, [(T_COMP, _cid(T1), _cid(T2 + "//x"))],
                "the tree the witness moved in",
            ),
            (
                "a WRONG `old` content id is convicted (the cause did not start here)",
                AFTER, LED_AFTER, GOOD_TRANS, [(T_COMP, _cid(T1 + "//x"), _cid(T2))],
                "did not start from this tree",
            ),
            (
                "a WRONG PATH — a real edit, to a file this key is not hashed from — is "
                "convicted",
                _sources(T1, H1, **{U_COMP: U2}), LED_AFTER, GOOD_TRANS,
                [(U_COMP, _cid(_sources(T1, H1)[U_COMP]), _cid(U2))],
                "is not in the source closure",
            ),
            (
                "a MISSING content transition (witness moves, source does not) is convicted",
                BEFORE, LED_AFTER, GOOD_TRANS, GOOD_PATHS,
                "no production source in the closure",
            ),
            (
                "a GENERATED/EVIDENCE-only cause path is refused before it is even read",
                AFTER, LED_AFTER, GOOD_TRANS,
                [("evidence/proof_ledger.jsonl", "a" * 64, "b" * 64)],
                "generated evidence or lane machinery",
            ),
            (
                "an OVERBROAD cause (a real source edit outside the closure) is convicted",
                _sources(T2, H1, **{UNRELATED: "pub fn register() { /* edited */ }\n"}),
                LED_AFTER, GOOD_TRANS,
                GOOD_PATHS + [(UNRELATED, _cid("pub fn register() {}\n"),
                               _cid("pub fn register() { /* edited */ }\n"))],
                "is not in the source closure",
            ),
            (
                "an UNDER-DECLARED cause (a second edit inside the closure) is convicted",
                _sources(T2, H2), LED_AFTER, GOOD_TRANS, GOOD_PATHS,
                "and the record does not declare it",
            ),
            (
                "one key's real cause CANNOT vouch for a key it shares no shader with",
                AFTER, {"a": (HY, ["stem_a"]), "b": (HQ, ["stem_b"])},
                [("a", HX, HY), ("b", HP, HQ)], GOOD_PATHS,
                "authorised by no declared cause path in its own stems",
            ),
            (
                "a moved SUBJECT (the entry's `shaders` changed) is convicted, not excused",
                AFTER, {"a": (HY, ["stem_a", "stem_b"]), "b": (HP, ["stem_b"])},
                GOOD_TRANS, GOOD_PATHS,
                "The SUBJECT moved",
            ),
        ]:
            slug = "".join(c for c in name if c.isalnum())[:24]
            repo = _plant_sourced(tmp / f"28-{slug}", BEFORE, LED_BEFORE)
            _land_sourced(repo, "squash", files_after, led_after, _v3(trans, paths))
            r = run(["--repo", str(repo)], cwd=repo)
            record(
                "PLANTED", name,
                r.returncode == 1
                and "uncorroborated_rewitness_cause" in r.stdout
                and needle in r.stdout,
                f"exit={r.returncode}",
            )

        # REPLAY AFTER LANDING. The cause is real and already landed; a LATER commit moves
        # the witness again with no source change and points at that same, spent cause. This
        # is the shape a hand-edited digest takes when somebody reaches for the nearest
        # plausible-looking edit, and it is the one a naive "did this content ever exist?"
        # check would wave through.
        repo = _plant_sourced(tmp / "29-replay", BEFORE, LED_BEFORE)
        _land_sourced(repo, "squash", AFTER, LED_AFTER, _v3(GOOD_TRANS, GOOD_PATHS))
        _write_sourced(repo, AFTER, {"a": (HZ, ["stem_a"]), "b": (HP, ["stem_b"])})
        _git(["add", "-A"], repo)
        _git(["commit", "-q", "-m", "later: move the witness again, touching no source"], repo)
        _write_rewitness(repo, {"schema": 1, "screened_since": "", "rewitness": [
            _v3(GOOD_TRANS, GOOD_PATHS),
            _v3([("a", HY, HZ)], GOOD_PATHS, reason="planted: replay of a spent cause"),
        ]})
        r = run(["--repo", str(repo)], cwd=repo)
        record(
            "PLANTED",
            "REPLAYING a cause that already landed, for a later move, is convicted",
            r.returncode == 1
            and "uncorroborated_rewitness_cause" in r.stdout
            and "already landed without the witness move" in r.stdout,
            f"exit={r.returncode}",
        )

        # The green arms that keep the corroboration from being merely strict. Each is a
        # REAL way `source_digest` moves, and a screen that convicted them would push authors
        # straight back to prose.
        for name, files_after, paths in [
            (
                "an INCLUDE-only edit corroborates (the closure is transitive)",
                _sources(T1, H2), [(HELPER, _cid(H1), _cid(H2))],
            ),
            (
                "a VARIANT-MANIFEST edit (the `-D` defines) corroborates",
                _sources(T1, H1, defines_a="D=8"),
                [(VARIANTS, _cid(_sources(T1, H1)[VARIANTS]),
                  _cid(_sources(T1, H1, defines_a="D=8")[VARIANTS]))],
            ),
        ]:
            slug = "".join(c for c in name if c.isalnum())[:20]
            repo = _plant_sourced(tmp / f"30-{slug}", BEFORE, LED_BEFORE)
            _land_sourced(repo, "squash", files_after, LED_AFTER, _v3(GOOD_TRANS, paths))
            r = run(["--repo", str(repo)], cwd=repo)
            record("PLANTED", name, r.returncode == 0, f"exit={r.returncode}")

        # LINE ENDINGS. `core.autocrlf` is a per-clone setting, so a content id taken over raw
        # bytes would make the verdict depend on which machine checked out the tree — green on
        # the author's Linux box, red on the Windows lane, for identical source. The screen
        # normalises exactly as `normalize_shader_text` in the Rust does; this arm lands the
        # after-tree with CRLF everywhere and requires the LF-computed declaration to hold.
        repo = _plant_sourced(tmp / "31-crlf", BEFORE, LED_BEFORE)
        _land_sourced(repo, "squash", AFTER, LED_AFTER, _v3(GOOD_TRANS, GOOD_PATHS),
                      newline="\r\n")
        r = run(["--repo", str(repo)], cwd=repo)
        record(
            "PLANTED",
            "a CRLF checkout of the same source still corroborates an LF-computed cause",
            r.returncode == 0 and "0 uncorroborated" in r.stdout,
            f"exit={r.returncode}",
        )

        # Structural refusals are exit 2 — UNOBSERVABLE, never a colour. A register the
        # checker could not read has ruled on nothing, and a v3 record it half-read would
        # authorise whatever it names.
        for name, mutate, needle in [
            ("a v3 record carrying BOTH `caused_by` and `caused_by_content` is refused",
             lambda r: r | {"caused_by": "0" * 40}, "also carries `caused_by`"),
            ("an UNKNOWN cause `kind` is refused, not read as same_change",
             lambda r: r | {"caused_by_content": r["caused_by_content"] | {"kind": "vibes"}},
             "does not know"),
            ("a v3 record for a field other than `source_digest` is refused",
             lambda r: r | {"field": "toolchain"}, "declares field="),
            ("an ABSOLUTE cause path is refused",
             lambda r: _repath(r, "/etc/passwd"), "repository-relative POSIX path"),
            ("a cause path containing `..` is refused",
             lambda r: _repath(r, "rust/../../secrets.comp"), "repository-relative POSIX path"),
            ("a WINDOWS cause path is refused",
             lambda r: _repath(r, "C:\\shaders\\t.comp"), "repository-relative POSIX path"),
            ("a DUPLICATED cause path is refused",
             lambda r: _dup_path(r), "declared twice"),
            ("a cause path with old == new is refused (it declares a non-event)",
             lambda r: _repath(r, T_COMP, old=_cid(T2)), "declares old == new"),
            ("a malformed (non-sha256) content id is refused",
             lambda r: _repath(r, T_COMP, old="nope"), "64-hex-digit sha256"),
            ("a v3 record MISSING `caused_by_content` is refused",
             lambda r: {k: v for k, v in r.items() if k != "caused_by_content"}, "is missing"),
        ]:
            slug = "".join(c for c in name if c.isalnum())[:22]
            repo = _plant_sourced(tmp / f"32-{slug}", BEFORE, LED_BEFORE)
            _land_sourced(repo, "squash", AFTER, LED_AFTER,
                          mutate(_v3(GOOD_TRANS, GOOD_PATHS)))
            r = run(["--repo", str(repo)], cwd=repo)
            record("PLANTED", name, r.returncode == 2 and needle in r.stdout,
                   f"exit={r.returncode}")

        # All three schemas in ONE register. Migration is per record and opt-in: the six
        # transitions PR #53 migrated sit beside v1 and v2 records nobody rewrote, and the
        # screen must apply each record's own rule rather than the newest one it knows.
        repo = _plant_sourced(tmp / "33-coexist", BEFORE, LED_BEFORE)
        landed_before = _git(["rev-parse", "HEAD"], repo).stdout.strip()
        _land_sourced(repo, "squash", AFTER, {"a": (HY, ["stem_a"]), "b": (HQ, ["stem_b"])}, [
            _v3(GOOD_TRANS, GOOD_PATHS),
            _v2("source_digest", landed_before, [("b", HP, HQ)]),
        ])
        r = run(["--repo", str(repo)], cwd=repo)
        record(
            "PLANTED",
            "v2 and v3 records coexist in one register, each judged by its own rule",
            r.returncode == 0 and "0 uncorroborated" in r.stdout and "0 unlanded" in r.stdout,
            f"exit={r.returncode}",
        )


        #
        # The walk used `--all`. Minutes after a teammate pushed an in-progress branch the
        # live screen reported 28 VANISHED proofs "first proven in 18ddece" — a commit on
        # `squad/mouse` that is not an ancestor of HEAD. It was convicting this branch for
        # not containing somebody else's unmerged draft, and the sentence it printed
        # ("committed to this ledger and no longer in it") was simply false.
        #
        # Two arms, because the fix must not cost the screen its reason for existing:
        # a sibling branch's proofs are NOT the denominator, and a proof dropped inside a
        # merge still is — both merge parents are reachable from HEAD.
        repo = _plant_repo(tmp / "13", [["a", "b"]])
        _git(["checkout", "-q", "-b", "sibling"], repo)
        (repo / "evidence" / "proof_ledger.jsonl").write_text(
            "".join(_entry(k) for k in ["a", "b", "z"]),
            encoding="utf-8",
        )
        _git(["add", "-A"], repo)
        _git(["commit", "-q", "-m", "sibling proves z, never merged"], repo)
        _git(["checkout", "-q", "main"], repo)
        r = run(["--repo", str(repo)], cwd=repo)
        record(
            "PLANTED",
            "an UNMERGED sibling branch's proofs are not this branch's denominator",
            r.returncode == 0 and "0 VANISHED" in r.stdout,
            f"exit={r.returncode}",
        )

        _git(["merge", "-q", "--no-commit", "--no-ff", "sibling"], repo)
        (repo / "evidence" / "proof_ledger.jsonl").write_text(
            "".join(_entry(k) for k in ["a", "b"]),
            encoding="utf-8",
        )
        _git(["add", "-A"], repo)
        _git(["commit", "-q", "-m", "merge sibling, resolving z away"], repo)
        r = run(["--repo", str(repo)], cwd=repo)
        record(
            "PLANTED",
            "but a proof dropped INSIDE a merge is still convicted after the scope fix",
            r.returncode == 1 and "proof_vanished" in r.stdout and "\n  - z\n" in r.stdout,
            f"exit={r.returncode}",
        )

        # ISSUE #28: a valid `pull/N/merge` synthetic preview, distinguished from the same
        # tree incidentally shallow-marked the way the real CI job shallow-marked it.
        repo, main_tip = _synthetic_pr_merge(tmp / "14")
        r = run(["--repo", str(repo)], cwd=repo)
        record(
            "PLANTED",
            "a genuine GitHub pull/N/merge preview (full history) is OBSERVABLE and green",
            r.returncode == 0 and "0 VANISHED" in r.stdout,
            f"exit={r.returncode}",
        )

        # The exact mechanism from the workflow: an UNRELATED depth-1 fetch of a ref that
        # happens to be one of HEAD's own merge parents (`origin/main`'s tip is always the
        # base parent of a real `pull/N/merge`). This must not become a false PASS: the
        # census cannot tell that the graft is incidental, so it must still refuse to
        # answer rather than trust a walk it cannot prove is complete.
        _git(["fetch", "--no-tags", "--depth=1", repo.as_uri(), main_tip], repo)
        r = run(["--repo", str(repo)], cwd=repo)
        record(
            "PLANTED",
            "issue #28: the SAME tree, shallow-marked by an unrelated fetch of its own "
            "base parent, still refuses rather than guesses",
            r.returncode == 2 and "SHALLOW" in r.stdout,
            f"exit={r.returncode}",
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
