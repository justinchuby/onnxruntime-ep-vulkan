#!/usr/bin/env python3
"""Negative control for ``ci/check_gh_auth.py`` — every arm shown in its POSITIVE state,
plus the arm issue #16 actually asked for: proof that missing `gh` authentication reports
an instrument error, never a PASS and never a skip. Extended for issue #21 with the two
blind spots PR #17's review found: inline `env: {GH_TOKEN: ...}` being falsely convicted,
and a zero-`gh`-reaching-subject frame (e.g. the screen pointed at the wrong scope, a
subdirectory, or a file that genuinely has nothing to check) reporting a silent PASS.

A screen that has only ever been observed green is indistinguishable from a constant that
returns green. Every rule below is therefore exercised with the defect GENUINELY PRESENT,
labelled with how it got there:

  LIVE      the real screen (or the real check_main_is_green.py) against real bytes/a real
            environment, right now.
  REPLAYED  real workflow bytes taken from this repository's own history — a defect that
            actually shipped, not one imagined for the occasion.
  PLANTED   a mutation written on purpose. Proves the rule fires on the shape it was
            written for. Does NOT prove the rule is load-bearing, and the count of
            PLANTED arms is printed so nobody reads it as if it did.

Run:  python ci/negative_control_gh_auth.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SCREEN = HERE / "check_gh_auth.py"
MAIN_IS_GREEN = HERE / "check_main_is_green.py"
CI_YML = REPO / ".github" / "workflows" / "ci.yml"
CONFORMANCE_YML = REPO / ".github" / "workflows" / "conformance.yml"
WORKFLOWS_DIR = REPO / ".github" / "workflows"

# The commit immediately before this fix landed: `Open-reds negative control` has no
# GH_TOKEN in its `env:`, at any scope — the real bytes PR #13 run 31052604259 hit.
HISTORICAL_REF = "b1886d9976108b4723ca3df79e237060f239f250"

EXIT_PASS = 0
EXIT_FAIL_CONDITION = 1
EXIT_ERROR_INSTRUMENT = 4

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str, str]] = []


def record(kind: str, name: str, ok: bool, note: str = "") -> None:
    results.append((kind, name, PASS if ok else FAIL, note))
    mark = "ok  " if ok else "FAIL"
    print(f"  [{mark}] {kind:<8} {name}" + (f"  — {note}" if note and not ok else ""))


def run_screen(*paths: Path, extra: list[str] | None = None) -> subprocess.CompletedProcess:
    e = dict(os.environ)
    e.setdefault("PYTHONIOENCODING", "utf-8")
    return subprocess.run(
        [sys.executable, str(SCREEN), *(str(p) for p in paths), *(extra or [])],
        capture_output=True, encoding="utf-8", errors="replace", cwd=str(REPO), timeout=300,
    )


def sanitized_env() -> dict:
    """The process environment with every credential `gh` might pick up removed, so the
    LIVE arm below tests "no auth", not "an auth this box happened to have lying around".
    """
    e = dict(os.environ)
    for key in list(e):
        if key in ("GH_TOKEN", "GITHUB_TOKEN", "GH_ENTERPRISE_TOKEN"):
            del e[key]
    e.setdefault("PYTHONIOENCODING", "utf-8")
    # Force gh into "I am running inside Actions" mode regardless of the box this runs
    # on, so the arm exercises the same refusal path as a real workflow step — a `gh`
    # that is NOT told GITHUB_ACTIONS=true may fall back to a locally cached `gh auth
    # login` on a developer machine instead of refusing, which would test the wrong
    # thing here.
    e["GITHUB_ACTIONS"] = "true"
    # Point gh at an empty, throwaway config directory so a developer's real `gh auth
    # login` session (if any) cannot leak into this arm and turn "no auth" into "an
    # auth this box happened to have".
    tmp_gh_config = tempfile.mkdtemp(prefix="gh-auth-control-config-")
    e["GH_CONFIG_DIR"] = tmp_gh_config
    return e


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="gh-auth-control-", dir=str(REPO / "ci")))
    try:
        print("\nLIVE — the real screen against the real workflows")
        r = run_screen(CI_YML, CONFORMANCE_YML)
        record(
            "LIVE", "today's ci.yml + conformance.yml pass the screen", r.returncode == 0,
            (r.stdout or "")[-1500:] + (r.stderr or "")[-1500:],
        )
        record(
            "LIVE", "and it names the two real `gh`-reaching steps as checked",
            "PASS" in r.stdout and "gh`-reaching step" in r.stdout,
        )

        print("\nLIVE — issue #21: the real .github/workflows DIRECTORY, not two named files")
        # Naming files one at a time is exactly how a new workflow keeps going unscreened
        # after it is added — nobody remembers to extend the command line. Pointing the
        # screen at the directory instead must find the same two real subjects (and every
        # other workflow file in the tree, whether or not it happens to call `gh`).
        r = run_screen(WORKFLOWS_DIR)
        record(
            "LIVE", "the whole workflows directory, expanded recursively, still passes",
            r.returncode == EXIT_PASS and "gh`-reaching step" in r.stdout,
            (r.stdout or "")[-800:],
        )

        print("\nLIVE — issue #21: wrong-scope invocation, the real conformance.yml alone")
        # conformance.yml genuinely has zero `gh`-reaching steps of its own. Screening it
        # alone is exactly what a mis-scoped CI caller (pointed at a subdirectory, the
        # wrong file, or run from the wrong working directory) would do by accident, and
        # the pre-#21 screen would have printed `PASS — 0 gh-reaching step(s)` for it —
        # indistinguishable from a screen that never actually read anything.
        r = run_screen(CONFORMANCE_YML)
        record(
            "LIVE",
            "screening only conformance.yml (0 real gh-reaching steps) is a loud "
            "ERROR(instrument=zero_gh_reaching_subjects), never a silent PASS",
            r.returncode == EXIT_ERROR_INSTRUMENT and "zero_gh_reaching_subjects" in r.stdout,
            (r.stdout or "")[-600:],
        )

        print("\nLIVE — the exact arm issue #16 asked for: absent auth, never a PASS")
        # This is ci/check_main_is_green.py — "the exact open-reds process that uses
        # `gh run list`" — invoked with every credential stripped. Whether `gh` itself is
        # on PATH or not, whether it would otherwise be logged in or not, the answer must
        # collapse to the same polarity: ERROR(instrument=github_unreachable), exit 4,
        # never 0, never the word PASS, never a skip.
        env = sanitized_env()
        r = subprocess.run(
            [sys.executable, str(MAIN_IS_GREEN), "--branch", "main", "--limit", "5"],
            capture_output=True, encoding="utf-8", errors="replace", cwd=str(REPO),
            env=env, timeout=90,
        )
        out = (r.stdout or "") + (r.stderr or "")
        record(
            "LIVE", "no GH_TOKEN/GITHUB_TOKEN -> exit 4, never 0",
            r.returncode == EXIT_ERROR_INSTRUMENT, f"returncode={r.returncode}\n{out[-1500:]}",
        )
        record(
            "LIVE", "the token is named as the failure, in R13 vocabulary",
            "ERROR(instrument=github_unreachable)" in out,
            out[-800:],
        )
        record(
            "LIVE", "the answer is never spelled PASS or a skip token",
            "PASS" not in out and "SKIP" not in out.upper(),
            out[-800:],
        )

        print("\nREPLAYED — this repository's own ci.yml one commit before this fix")
        old = subprocess.run(
            ["git", "show", f"{HISTORICAL_REF}:.github/workflows/ci.yml"],
            cwd=str(REPO), capture_output=True, encoding="utf-8", errors="replace",
        )
        if old.returncode != 0:
            record("REPLAYED", f"{HISTORICAL_REF} readable", False, old.stderr.strip())
        else:
            victim = tmp / f"ci-at-{HISTORICAL_REF[:7]}.yml"
            victim.write_text(old.stdout, encoding="utf-8")
            r = run_screen(victim)
            record(
                "REPLAYED",
                "the real pre-fix ci.yml convicts (missing_token_path, Open-reds negative control)",
                r.returncode == EXIT_FAIL_CONDITION
                and "missing_token_path" in r.stdout
                and "Open-reds negative control" in r.stdout,
                (r.stdout or "")[-1500:],
            )
            r2 = run_screen(CI_YML)
            record(
                "REPLAYED",
                "the same rule over today's bytes is PASS, so it is not a constant",
                r2.returncode == EXIT_PASS,
                (r2.stdout or "")[-800:],
            )

        print("\nPLANTED — synthetic workflows, each isolating one shape")

        def plant(name: str, text: str) -> subprocess.CompletedProcess:
            p = tmp / name
            p.write_text(text, encoding="utf-8")
            return run_screen(p)

        r = plant(
            "no-token-api-call.yml",
            "name: p\non: push\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
            "    steps:\n      - name: s\n        run: gh api repos/x/y\n",
        )
        record(
            "PLANTED", "an API call with no token anywhere is convicted",
            r.returncode == EXIT_FAIL_CONDITION and "missing_token_path" in r.stdout,
            (r.stdout or "")[-600:],
        )

        r = plant(
            "non-api-call-no-token.yml",
            # A second, real subject (`gh api`, token-satisfied) keeps this file's total
            # gh-reaching count above zero -- otherwise issue #21's zero-subject rule
            # would fire here for an unrelated reason and this arm would stop isolating
            # the one shape it exists to check: that `gh --version` itself is not
            # mistaken for an API call.
            "name: p\non: push\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
            "    env:\n      GH_TOKEN: x\n"
            "    steps:\n      - name: s\n        run: gh --version\n"
            "      - name: real subject\n        run: gh api repos/x/y\n",
        )
        record(
            "PLANTED",
            "a non-API `gh --version` call with no token of its own is NOT counted as "
            "a subject (only the file's one real `gh api` call is)",
            r.returncode == EXIT_PASS and "PASS — 1 `gh`-reaching step" in r.stdout,
            (r.stdout or "")[-600:],
        )

        r = plant(
            "token-in-wrong-job.yml",
            "name: p\non: push\njobs:\n"
            "  has_token:\n    runs-on: ubuntu-latest\n    env:\n"
            "      GH_TOKEN: x\n    steps:\n      - name: s1\n        run: echo hi\n"
            "  no_token:\n    runs-on: ubuntu-latest\n    steps:\n"
            "      - name: s2\n        run: gh pr list\n",
        )
        record(
            "PLANTED", "a token declared in a DIFFERENT job does not satisfy this one",
            r.returncode == EXIT_FAIL_CONDITION and "no_token" in r.stdout,
            (r.stdout or "")[-600:],
        )

        r = plant(
            "job-level-token-satisfies.yml",
            "name: p\non: push\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
            "    env:\n      GH_TOKEN: x\n    steps:\n"
            "      - name: s1\n        run: gh issue list\n"
            "      - name: s2\n        run: gh release list\n",
        )
        record(
            "PLANTED", "a job-level token satisfies every step in that job",
            r.returncode == EXIT_PASS,
            (r.stdout or "")[-600:],
        )

        r = plant(
            "block-env-step-level.yml",
            "name: p\non: push\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
            "    steps:\n      - name: s\n        env:\n          GH_TOKEN: x\n"
            "        run: gh api repos/x/y\n",
        )
        record(
            "PLANTED",
            "block-form `env:` (key on its own indented line below) at STEP level "
            "satisfies the token check",
            r.returncode == EXIT_PASS,
            (r.stdout or "")[-600:],
        )

        r = plant(
            "inline-env-step-level.yml",
            "name: p\non: push\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
            "    steps:\n      - name: s\n"
            "        env: {GH_TOKEN: ${{ github.token }}}\n"
            "        run: gh api repos/x/y\n",
        )
        record(
            "PLANTED",
            "issue #21's false-conviction fix: inline `env: {GH_TOKEN: ...}` on one "
            "line at STEP level satisfies the token check exactly like the block form "
            "above -- this is the exact remediation text quoted in this screen's own "
            "FAIL message, and it must never itself be convicted",
            r.returncode == EXIT_PASS,
            (r.stdout or "")[-600:],
        )

        r = plant(
            "wrong-token-name.yml",
            "name: p\non: push\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
            "    steps:\n      - name: s\n        env: {TOKEN: x}\n        run: gh pr list\n",
        )
        record(
            "PLANTED",
            "a wrongly-named env key (TOKEN, not GH_TOKEN/GITHUB_TOKEN) does not "
            "satisfy the check, in either YAML form",
            r.returncode == EXIT_FAIL_CONDITION and "missing_token_path" in r.stdout,
            (r.stdout or "")[-600:],
        )

        r = plant(
            "zero-subjects.yml",
            "name: p\non: push\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
            "    steps:\n      - name: s\n        run: echo hi\n",
        )
        record(
            "PLANTED",
            "issue #21: a workflow with real steps but ZERO gh-reaching subjects is a "
            "loud ERROR(instrument=zero_gh_reaching_subjects) by default, never a "
            "silent PASS",
            r.returncode == EXIT_ERROR_INSTRUMENT and "zero_gh_reaching_subjects" in r.stdout,
            (r.stdout or "")[-600:],
        )

        r = run_screen(tmp / "zero-subjects.yml", extra=["--allow-empty-frame"])
        record(
            "PLANTED",
            "--allow-empty-frame is the only documented way to turn that same "
            "zero-subject frame into a PASS, and it says so explicitly in its output",
            r.returncode == EXIT_PASS and "allow-empty-frame" in r.stdout,
            (r.stdout or "")[-400:],
        )

        empty_dir = tmp / "empty_subdir"
        empty_dir.mkdir()
        r = run_screen(empty_dir)
        record(
            "PLANTED",
            "issue #21: a directory scope with no *.yml/*.yaml file under it at all "
            "(the wrong-working-directory/wrong-subdirectory shape) is ERROR, never a "
            "silent pass",
            r.returncode == EXIT_ERROR_INSTRUMENT and "empty_workflow_directory" in r.stdout,
            (r.stdout or "")[-400:],
        )

        nested = tmp / "nested_dir" / "sub"
        nested.mkdir(parents=True)
        (nested / "bad.yml").write_text(
            "name: p\non: push\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
            "    steps:\n      - name: s\n        run: gh api repos/x/y\n",
            encoding="utf-8",
        )
        r = run_screen(tmp / "nested_dir")
        record(
            "PLANTED",
            "directory expansion recurses into subdirectories, so a workflow nested "
            "under a --workflows-dir-style invocation cannot hide from it -- checking "
            "only a subdirectory of a broader scope cannot pass vacuously",
            r.returncode == EXIT_FAIL_CONDITION and "missing_token_path" in r.stdout,
            (r.stdout or "")[-600:],
        )

        r = plant(
            "indirect-via-open-reds.yml",
            "name: p\non: push\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
            "    steps:\n      - name: s\n        run: python ci/check_open_reds.py\n",
        )
        record(
            "PLANTED",
            "a step reaching `gh` only through the real open-reds register is still caught",
            r.returncode == EXIT_FAIL_CONDITION and "check_open_reds.py" in r.stdout,
            (r.stdout or "")[-600:],
        )

        r = run_screen(tmp / "does-not-exist.yml")
        record(
            "PLANTED", "a workflow path that does not exist -> ERROR(instrument), not a red",
            r.returncode == EXIT_ERROR_INSTRUMENT and "workflow_not_found" in r.stdout,
            (r.stdout or "")[-400:],
        )

        r = plant("empty.yml", "name: p\non: push\n")
        record(
            "PLANTED", "a workflow with no steps -> ERROR(instrument=no_steps_parsed)",
            r.returncode == EXIT_ERROR_INSTRUMENT and "no_steps_parsed" in r.stdout,
            (r.stdout or "")[-400:],
        )

        r = plant(
            "comment-mentions-gh-run.yml",
            # Same reasoning as the `gh --version` arm above: a second real subject
            # keeps the total above zero so this isolates only the comment-matching
            # shape, not issue #21's separate zero-subject rule.
            "name: p\non: push\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
            "    env:\n      GH_TOKEN: x\n"
            "    steps:\n      - name: s\n"
            "        # this step does NOT call `gh run list`, it only talks about it\n"
            "        run: echo 'no gh here'\n"
            "      - name: real subject\n        run: gh api repos/x/y\n",
        )
        record(
            "PLANTED",
            "a comment that merely quotes `gh run list` is not mistaken for the "
            "command (only the file's one real `gh api` call is counted)",
            r.returncode == EXIT_PASS and "PASS — 1 `gh`-reaching step" in r.stdout,
            (r.stdout or "")[-600:],
        )

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    live = sum(1 for k, _, _, _ in results if k == "LIVE")
    replayed = sum(1 for k, _, _, _ in results if k == "REPLAYED")
    planted = sum(1 for k, _, _, _ in results if k == "PLANTED")
    failed = [r for r in results if r[2] == FAIL]

    print(
        f"\n{len(results)} arm(s): {live} LIVE / {replayed} REPLAYED / {planted} PLANTED. "
        f"PLANTED arms prove the rule fires on the shape it was written for; they do not "
        f"show it is load-bearing. The LIVE and REPLAYED arms are the ones that do."
    )
    if failed:
        print(f"\n{len(failed)} arm(s) FAILED:")
        for kind, name, _, note in failed:
            print(f"  [{kind}] {name}")
            if note:
                print(f"      {note[:500]}")
        return 1
    print(f"\n{len(results)}/{len(results)} arms fire as specified, exit 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
