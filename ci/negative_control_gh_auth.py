#!/usr/bin/env python3
"""Negative control for ``ci/check_gh_auth.py`` — every arm shown in its POSITIVE state,
plus the arm issue #16 actually asked for: proof that missing `gh` authentication reports
an instrument error, never a PASS and never a skip. Extended for issue #21 with the two
blind spots PR #17's review found: inline `env: {GH_TOKEN: ...}` being falsely convicted,
and a zero-`gh`-reaching-subject frame (e.g. the screen pointed at the wrong scope, a
subdirectory, or a file that genuinely has nothing to check) reporting a silent PASS.
Extended again for issue #25 with the YAML-STRUCTURAL blind spots an adversarial review
of #22 found: text resembling `env:`/`GH_TOKEN:` inside a `run: |` block scalar, and a
real `env` mapping nested under something else entirely (`services.<id>.env`, `with:
env:`) — both of which a purely line-oriented "more indented than X" check cannot tell
apart from a real declaration. Plus: quoted block keys and multi-line flow mappings must
still be RECOGNISED as valid declarations, and a duplicate key in one `env:` mapping must
fail loudly rather than silently pick a winner.

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
import re
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

        print("\nPLANTED — issue #25: YAML-structural shapes a line-oriented scan cannot tell apart")

        r = plant(
            "run-block-scalar-trap.yml",
            # Text that resembles a real `env:`/`GH_TOKEN:` declaration, but sitting
            # inside a `run: |` block scalar -- a step's own shell script, not YAML
            # structure. A line-oriented "is this line more indented than X" check
            # cannot tell this apart from a real declaration; a structural parser must.
            "name: p\non: push\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
            "    steps:\n      - name: s\n        run: |\n"
            "          echo \"this text is not yaml:\"\n"
            "          echo \"env:\"\n"
            "          echo \"  GH_TOKEN: fake\"\n"
            "          gh api repos/x/y\n",
        )
        record(
            "PLANTED",
            "text resembling `env:`/`GH_TOKEN:` inside a `run: |` block scalar is "
            "literal script text, not a real token declaration -- still convicted",
            r.returncode == EXIT_FAIL_CONDITION and "missing_token_path" in r.stdout,
            (r.stdout or "")[-600:],
        )

        r = plant(
            "services-env-trap.yml",
            # `services.<id>.env` is a real YAML mapping named `env`, but it is the
            # SERVICE CONTAINER's environment, not the job's/step's -- a nested
            # unrelated map, not a token path any step here can see.
            "name: p\non: push\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
            "    services:\n      redis:\n        image: redis\n"
            "        env:\n          GH_TOKEN: fake\n"
            "    steps:\n      - name: s\n        run: gh api repos/x/y\n",
        )
        record(
            "PLANTED",
            "`services.<id>.env` does not satisfy the check -- it is the service "
            "container's environment, not the job's",
            r.returncode == EXIT_FAIL_CONDITION and "missing_token_path" in r.stdout,
            (r.stdout or "")[-600:],
        )

        r = plant(
            "with-env-trap.yml",
            # `with: env:` is a real YAML mapping named `env`, but it is an ACTION
            # INPUT that happens to be called `env` -- not the step's own execution
            # environment.
            "name: p\non: push\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
            "    steps:\n      - name: s\n        uses: some/action@v1\n"
            "        with:\n          env:\n            GH_TOKEN: fake\n"
            "      - name: real\n        run: gh api repos/x/y\n",
        )
        record(
            "PLANTED",
            "`with: env:` does not satisfy the check -- it is an action input named "
            "`env`, not the step's own environment",
            r.returncode == EXIT_FAIL_CONDITION and "missing_token_path" in r.stdout,
            (r.stdout or "")[-600:],
        )

        r = plant(
            "quoted-block-key.yml",
            "name: p\non: push\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
            "    steps:\n      - name: s\n        env:\n"
            '          "GH_TOKEN": ${{ github.token }}\n'
            "        run: gh api repos/x/y\n",
        )
        record(
            "PLANTED",
            "a quoted block-form key (`\"GH_TOKEN\":`) satisfies the check exactly "
            "like an unquoted one",
            r.returncode == EXIT_PASS,
            (r.stdout or "")[-600:],
        )

        r = plant(
            "duplicate-block-key.yml",
            "name: p\non: push\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
            "    steps:\n      - name: s\n        env:\n"
            "          GH_TOKEN: one\n          GH_TOKEN: two\n"
            "        run: gh api repos/x/y\n",
        )
        record(
            "PLANTED",
            "a key declared twice in one BLOCK-form `env:` mapping is an "
            "unsupported/ambiguous construct -- ERROR, never a silent guess",
            r.returncode == EXIT_ERROR_INSTRUMENT
            and "unsupported_yaml_construct" in r.stdout
            and "declared twice" in r.stdout,
            (r.stdout or "")[-600:],
        )

        r = plant(
            "duplicate-flow-key.yml",
            "name: p\non: push\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
            "    steps:\n      - name: s\n"
            "        env: {GH_TOKEN: one, GH_TOKEN: two}\n"
            "        run: gh api repos/x/y\n",
        )
        record(
            "PLANTED",
            "a key declared twice in one FLOW-form `env: {...}` mapping is the same "
            "unsupported/ambiguous construct -- ERROR, never a silent guess",
            r.returncode == EXIT_ERROR_INSTRUMENT
            and "unsupported_yaml_construct" in r.stdout
            and "declared twice" in r.stdout,
            (r.stdout or "")[-600:],
        )

        r = plant(
            "trailing-comment-on-inline-env.yml",
            "name: p\non: push\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
            "    steps:\n      - name: s\n"
            "        env: {GH_TOKEN: ${{ github.token }}}  # ci: token lives here\n"
            "        run: gh api repos/x/y\n",
        )
        record(
            "PLANTED",
            "a trailing `# comment` after a real inline `env: {...}` declaration "
            "does not stop it from being read as a declaration",
            r.returncode == EXIT_PASS,
            (r.stdout or "")[-600:],
        )

        r = plant(
            "multiline-flow-mapping.yml",
            "name: p\non: push\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
            "    steps:\n      - name: s\n        env: {\n"
            "          GH_TOKEN: ${{ github.token }},\n"
            "          OTHER: value\n        }\n"
            "        run: gh api repos/x/y\n",
        )
        record(
            "PLANTED",
            "a flow mapping `env: {...}` split across several physical lines is "
            "still read as one declaration, not missed the way a single-line-only "
            "regex would miss it",
            r.returncode == EXIT_PASS,
            (r.stdout or "")[-600:],
        )

        print("\nPLANTED — issue #25 (PR #27 review): R1/R2/R3/N1/N2 fixes")

        r = plant(
            "r1-with-steps-trap.yml",
            # A legal action input happens to be named `steps` and holds its own list
            # of maps (`with: steps: [...]`, e.g. a matrix-generating action). Before
            # R1, ANY frame whose key is literally "steps" reset step-minting, so this
            # nested list's own dash orphaned the REAL step: its `run: gh api ...`
            # line (after the `with:` block) was silently dropped from body capture
            # and never became a checked subject at all -- a false PASS by omission,
            # not by misjudging a token that was actually there. A second, unrelated,
            # correctly-tokened `gh` step keeps the file's total gh-reaching count
            # above zero, so this isolates the omission itself rather than tripping
            # issue #21's separate zero-subject rule.
            "name: p\non: push\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
            "    steps:\n      - name: s\n        uses: some/action@v1\n"
            "        with:\n          steps:\n            - name: fake\n"
            "              run: echo hi\n"
            "        run: gh api repos/x/y\n"
            "      - name: real\n        env:\n          GH_TOKEN: x\n"
            "        run: gh api repos/a/b\n",
        )
        record(
            "PLANTED",
            "R1: a `with: steps:` list does not reset/orphan the real step -- its "
            "own `run: gh api ...` (untokened) is still convicted, not silently "
            "dropped from the count",
            r.returncode == EXIT_FAIL_CONDITION
            and "missing_token_path" in r.stdout
            and "PASS — 1 `gh`-reaching step" not in r.stdout,
            (r.stdout or "")[-700:],
        )

        r = plant(
            "r2-anchor-with-nested-env.yml",
            # `with: &a` is a scalar-ish value this parser does not structurally
            # understand (an anchor). Before R2, no frame was pushed for it at all,
            # so the indented `env:` mapping that actually belongs to the anchored
            # `with:` value got silently attributed to the enclosing step frame
            # instead -- exactly the false "step-level env" reading R2 forbids.
            "name: p\non: push\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
            "    steps:\n      - name: s\n        uses: some/action@v1\n"
            "        with: &anchor\n          env:\n            GH_TOKEN: x\n"
            "        run: gh api repos/x/y\n",
        )
        record(
            "PLANTED",
            "R2: an anchor/tag/alias value (`with: &a`) followed by more-indented "
            "content is ERROR(instrument), not silently attributed to the "
            "grandparent frame",
            r.returncode == EXIT_ERROR_INSTRUMENT
            and "unsupported_yaml_construct" in r.stdout
            and "anchor, tag, or alias" in r.stdout,
            (r.stdout or "")[-700:],
        )

        r = plant(
            "r2-tag-with-nested-env.yml",
            "name: p\non: push\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
            "    steps:\n      - name: s\n        uses: some/action@v1\n"
            "        with: !!map\n          env:\n            GH_TOKEN: x\n"
            "        run: gh api repos/x/y\n",
        )
        record(
            "PLANTED",
            "R2: the same shape with an explicit `!!tag` instead of an anchor is "
            "caught the same way",
            r.returncode == EXIT_ERROR_INSTRUMENT
            and "unsupported_yaml_construct" in r.stdout
            and "anchor, tag, or alias" in r.stdout,
            (r.stdout or "")[-700:],
        )

        r = plant(
            "r3a-multi-document.yml",
            # A `---` document-separator line. This screen reads one YAML document
            # per file; silently continuing past it would misattribute the second
            # document's steps/env to whatever frame was open at the end of the first.
            "name: p\non: push\n---\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
            "    steps:\n      - name: s\n        run: gh api repos/x/y\n",
        )
        record(
            "PLANTED",
            "R3a: a multi-document `---` separator is ERROR(instrument), never read "
            "past silently",
            r.returncode == EXIT_ERROR_INSTRUMENT
            and "unsupported_yaml_construct" in r.stdout
            and "document-separator" in r.stdout,
            (r.stdout or "")[-700:],
        )

        r = plant(
            "r3b-nested-flow-value.yml",
            # `env: {FOO: {GH_TOKEN: x}}` -- GH_TOKEN is nested INSIDE FOO's own
            # value, not a sibling key of this mapping. The prior known-limits prose
            # claimed this "could in principle" confuse the screen; it must now be a
            # loud refusal, not a silent maybe-right guess either way.
            "name: p\non: push\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
            "    steps:\n      - name: s\n"
            "        env: {FOO: {GH_TOKEN: x}}\n        run: gh api repos/x/y\n",
        )
        record(
            "PLANTED",
            "R3b: a nested flow collection as an `env: {...}` value "
            "(`{FOO: {GH_TOKEN: x}}`) is ERROR(instrument), not a guess at whether "
            "the nested key counts",
            r.returncode == EXIT_ERROR_INSTRUMENT
            and "unsupported_yaml_construct" in r.stdout
            and "nested flow collection" in r.stdout,
            (r.stdout or "")[-700:],
        )

        r = plant(
            "r3b-quoted-brace-value.yml",
            # `env: {FOO: '{"GH_TOKEN": "1"}'}` -- a quoted value that itself
            # contains literal braces. The quote-tracking scan can in fact tell this
            # is opaque text, but the review asked for a loud refusal here too rather
            # than trusting its own reading of an ambiguous-looking shape.
            "name: p\non: push\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
            "    steps:\n      - name: s\n"
            '        env: {FOO: \'{"GH_TOKEN": "1"}\'}\n        run: gh api repos/x/y\n',
        )
        record(
            "PLANTED",
            "R3b: a quoted value containing a literal brace inside `env: {...}` is "
            "ERROR(instrument), not silently trusted as inert text",
            r.returncode == EXIT_ERROR_INSTRUMENT
            and "unsupported_yaml_construct" in r.stdout
            and "quoted value" in r.stdout,
            (r.stdout or "")[-700:],
        )

        r = plant(
            "r3b-gh-expression-still-passes.yml",
            # Guard against a regression the R3b fix could easily introduce: a
            # `${{ github.token }}` GitHub Actions expression is internally balanced
            # `{`/`}` text, not YAML flow-collection nesting, and must NOT be
            # mistaken for the R3b nested-value shape above -- this is, after all,
            # the exact remediation text this screen's own FAIL message recommends.
            "name: p\non: push\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
            "    steps:\n      - name: s\n"
            "        env: {GH_TOKEN: ${{ github.token }}, OTHER: ${{ secrets.X }}}\n"
            "        run: gh api repos/x/y\n",
        )
        record(
            "PLANTED",
            "R3b regression guard: a `${{ ... }}` GH Actions expression inside "
            "`env: {...}` is NOT mistaken for nested flow-collection value braces",
            r.returncode == EXIT_PASS,
            (r.stdout or "")[-700:],
        )

        r = plant(
            "n1-duplicate-sibling-block-env.yml",
            # Two SIBLING `env:` block keys at the same job scope. Before N1 these
            # were silently unioned (`set.update`), so a job with two `env:` blocks
            # -- one of which happens to declare GH_TOKEN -- satisfied the check even
            # though this is itself invalid/ambiguous YAML (a duplicate mapping key)
            # that no single reading should be trusted to resolve.
            "name: p\non: push\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
            "    env:\n      GH_TOKEN: x\n    env:\n      GITHUB_TOKEN: y\n"
            "    steps:\n      - name: s\n        run: gh api repos/x/y\n",
        )
        record(
            "PLANTED",
            "N1: a duplicate sibling `env:` key at job scope is ERROR(instrument), "
            "not silently unioned with the first one",
            r.returncode == EXIT_ERROR_INSTRUMENT
            and "unsupported_yaml_construct" in r.stdout
            and "second `env:` key" in r.stdout,
            (r.stdout or "")[-700:],
        )

        r = plant(
            "n2-dash-inline-env-sibling-field.yml",
            # `- env:` opens a block-form env inline with the dash. Before N2, the
            # redispatch used a synthetic `dash_indent + 1` column for `env:` itself,
            # which undershoots its TRUE column (`dash_indent + 2`, one space after
            # the dash) -- so a later TRUE SIBLING field at that same real column
            # (here a bogus step-level `GH_TOKEN:` key that is NOT actually inside
            # `env:`) was wrongly swallowed as one of env's own children, letting a
            # coincidentally-named sibling satisfy the token check it should not.
            "name: p\non: push\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
            "    steps:\n      - env:\n          NOT_A_TOKEN: x\n"
            "        GH_TOKEN: this-is-not-really-an-env-declaration\n"
            "        run: gh api repos/x/y\n",
        )
        record(
            "PLANTED",
            "N2: a dash-inline `- env:` block correctly stops at its own real "
            "column -- a step-level sibling key that merely LOOKS like a token "
            "declaration does not satisfy the check",
            r.returncode == EXIT_FAIL_CONDITION and "missing_token_path" in r.stdout,
            (r.stdout or "")[-700:],
        )

        print("\nLIVE — issue #25: ci.yml's production invocation is pinned to the directory form")
        # The wiring concern issue #25 (and #21 before it) exists to close: naming
        # files one at a time on the command line is exactly how a new/relocated
        # workflow keeps going unscreened. Read the real step's `run:` text and prove
        # it is the single-directory form, not two (or more) named files -- reverting
        # this wiring is exactly the regression this arm exists to catch.
        ci_text = CI_YML.read_text(encoding="utf-8")
        invocations = re.findall(r"run: python ci/check_gh_auth\.py ([^\n]+)", ci_text)
        record(
            "LIVE",
            "the real ci.yml invokes check_gh_auth.py with the whole "
            ".github/workflows directory, not named files",
            len(invocations) == 1 and invocations[0].strip() == ".github/workflows",
            f"invocation(s) found: {invocations!r}",
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
