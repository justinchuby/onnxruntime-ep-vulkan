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

        # THE ARM FOR THE THIRD FRAMING DEFECT OF THE SESSION, AND THE ONE THAT SHIPPED.
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
