#!/usr/bin/env python3
"""``check_gh_auth`` — every API-using `gh` invocation in these workflows has a token path.

WHY THIS EXISTS
===============
PR #13 run 31052604259: the test-lock auditor succeeded, and the very next step —
`Open-reds negative control` — exited 4 at `gh run list` with `gh: To use GitHub CLI in
a GitHub Actions workflow, set the GH_TOKEN environment variable`. `ci/check_main_is_green
.py` did exactly what it was built to do (ERROR(instrument=github_unreachable), never a
false PASS); the defect was one level up, in a workflow step that called it with no token
in scope. Main's run before #13 never reached that step at all, so the gap could not
surface until #6 made the lane observable and #13 made the step before it stop eating the
failure.

This is the general form, screened statically so the NEXT missing token is a pull-request
diff rather than a run log: **which workflow steps reach the GitHub API through `gh`, and
does every one of them have GH_TOKEN or GITHUB_TOKEN declared somewhere it can see?**

THE TWO WAYS A STEP REACHES `gh`
=================================
DIRECT   its own script contains a `gh <subcommand>` where the subcommand talks to the
         API — `run`, `api`, `pr`, `issue`, `release`, `workflow`, `repo`, `gist`, `org`,
         `project`, `search`, `secret`, `variable`, `cache`, `auth status`. NOT `--version`,
         `help`, `completion`, `config`, `extension` — none of those touch the network and
         flagging them would teach a reader to ignore this screen's output.

INDIRECT its script invokes a `ci/*.py` tool that itself shells out to `gh` (detected the
         same way — the head of a subprocess argv is `"gh"`), or is one of the two scripts
         independently verified to run `ci/open_reds.json`'s REAL, default register end to
         end while that register still has a live entry pointing at a `gh`-shelling
         script. This is exactly the PR #13 chain, two steps removed from the literal
         string `gh`: `negative_control_open_reds.py` runs `check_open_reds.py` against
         the real register, whose `main_is_green` entry runs `check_main_is_green.py`,
         which calls `gh run list`. See `KNOWN_REAL_REGISTER_RUNNERS` below for why this
         is a short, named, hand-verified list rather than a generic "does file A's text
         mention file B's name" graph: `ci/test_lane_checks.py` also spawns
         `check_open_reds.py` as a subprocess, but always against a synthetic register it
         builds in a tmp dir, so it never reaches `gh` — and a generic textual graph
         cannot tell those two shapes apart without producing that false positive.

WHAT "has a token path" MEANS
==============================
`GH_TOKEN` or `GITHUB_TOKEN` is declared as a key in an `env:` block whose scope covers
the step: the step's own `env:`, its job's `env:`, or the workflow's top-level `env:`.
Declared, not valued — this screen reads key NAMES out of `env:` blocks and never reads,
prints, or evaluates a value, so it cannot leak a token even if one were hardcoded (which
would be its own, different, defect). Both supported YAML shapes for that block are read
the same way, semantically, not by line-shape guessing:

    env:                              env: {GH_TOKEN: ${{ github.token }}}
      GH_TOKEN: ${{ github.token }}

The block form spans multiple lines under `env:`; the inline (flow-mapping) form puts
`{key: value, ...}` on the `env:` line itself. Issue #21: a step remediated with the
inline form was being FALSELY CONVICTED of having no token path, because the block-only
line-shape check saw `env: {...}` and, since nothing followed on the NEXT line the way a
block mapping requires, read the step as having declared no keys at all — the exact
remediation text read as if it were the defect it fixes.

ZERO SUBJECTS
=============
If nothing across the screened files reaches the GitHub API through `gh`, that is either
an honestly `gh`-free set of workflows, or this screen has been pointed at the wrong
scope — a subdirectory, a stale path, the wrong working directory — and is reading
nothing. Issue #21: `PASS — 0 gh-reaching step(s)` is indistinguishable, on the page, from
"this screen was never actually run", so a zero-subject frame is `ERROR(instrument=
zero_gh_reaching_subjects)` by default, not a PASS. Pass `--allow-empty-frame` if an empty
frame is genuinely the intended one (e.g. deliberately screening a workflow directory
known to contain no `gh` calls yet, as a forward-looking regression barrier) — that mode
is opt-in and named on the command line, never the silent default.

A directory may be given in place of individual files: every `*.yml`/`*.yaml` file under
it (recursively) is screened, so a caller that names a whole `.github/workflows/`
directory keeps covering every workflow in it as files are added, rather than only the
ones somebody remembered to list by name — the concrete way a check "over a subdirectory"
would otherwise pass vacuously by construction.

WHAT IT DELIBERATELY DOES NOT DO
=================================
It does not run `gh`, and it does not need network access or a real token to answer its
question — it is a read of the YAML and the `ci/` tree, nothing else. It does not check
that the token has enough SCOPE for the call (`actions: read` vs `contents: write` etc.);
that is a human judgement about least privilege, made once per step and reviewed in the
diff, not a thing this screen can derive from the text. It does not follow `uses:` steps
(third-party actions) — those authenticate however their own inputs say to, which is a
different surface.

Terminal states (R13):

    0  GH-AUTH: PASS
    1  GH-AUTH: FAIL(condition=missing_token_path)
    2  usage
    4  GH-AUTH: ERROR(instrument=...)   workflow_not_found | empty_workflow_directory |
                                        no_steps_parsed | zero_gh_reaching_subjects

USAGE
    python ci/check_gh_auth.py .github/workflows/ci.yml .github/workflows/conformance.yml
    python ci/check_gh_auth.py .github/workflows
    python ci/check_gh_auth.py --allow-empty-frame .github/workflows/docs-only.yml
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CI_DIR = REPO_ROOT / "ci"
DEFAULT_REGISTER = CI_DIR / "open_reds.json"

EXIT_PASS = 0
EXIT_FAIL_CONDITION = 1
EXIT_USAGE = 2
EXIT_ERROR_INSTRUMENT = 4

TOKEN_NAMES = ("GH_TOKEN", "GITHUB_TOKEN")

#: `gh <subcommand>` invocations that talk to the GitHub API. Word-boundary on `gh` so this
#: does not fire on `ghost`, `high`, or a path containing the letters. Deliberately excludes
#: --version/help/completion/config/extension: those are local, and flagging them would
#: teach a reader that this screen's FAILs are noise.
_GH_API_SUBCOMMAND_RE = re.compile(
    r"""(?<![\w./-])gh\s+
        (run|api|pr|issue|release|workflow|repo|gist|org|project|search|
         secret|variable|cache|label|ruleset|attestation|codespace)\b
        (?!\s+(?:--version|help))
    """,
    re.VERBOSE,
)
#: `gh auth status` reaches the API to validate the token; `gh auth login`/`gh auth setup-git`
#: do not read repository data the way this screen cares about, so only `status` is included.
_GH_AUTH_STATUS_RE = re.compile(r"(?<![\w./-])gh\s+auth\s+status\b")

_PY_SCRIPT_RUN_RE = re.compile(r"(?:^|[\s;&|])python3?\s+(?P<path>ci/[A-Za-z0-9_./-]+\.py)")

_STEP_START_RE = re.compile(r"^(?P<indent>\s*)-\s+name:\s*(?P<name>.+?)\s*$")
_STEP_USES_ONLY_RE = re.compile(r"^(?P<indent>\s*)-\s+uses:\s*(?P<uses>\S+)")
_JOB_RE = re.compile(r"^  (?P<job>[A-Za-z0-9_.-]+):\s*$")
#: Block form: `env:` alone on its line, keys follow indented on the lines under it.
_ENV_BLOCK_RE = re.compile(r"^(?P<indent>\s*)env:\s*$")
#: Inline (flow-mapping) form: `env: {KEY: value, ...}` on the one line. YAML permits
#: either shape and they are semantically identical; a screen that only recognises the
#: first teaches a step author that the fix it just applied (the inline form is exactly
#: the remediation text quoted in this screen's own FAIL message) is the defect (#21).
_ENV_INLINE_RE = re.compile(r"^(?P<indent>\s*)env:\s*(?P<inline>\{.*\})\s*$")
_ENV_KEY_RE = re.compile(r"^(?P<indent>\s*)(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*:")
#: Key names inside a YAML flow mapping's `{...}` text. A key starts right after the
#: opening brace or a top-level comma — the two positions flow-style permits — optionally
#: quoted. Values are never inspected (this screen's own non-disclosure rule): they may
#: contain `${{ ... }}` expressions, which this repository's usage never puts a bare
#: `identifier:` inside, so scanning for that shape finds keys without needing a real
#: YAML flow-mapping parser.
_FLOW_MAPPING_KEY_RE = re.compile(r"""(?:\{|,)\s*(['"]?)(?P<key>[A-Za-z_][A-Za-z0-9_]*)\1\s*:""")


def _flow_mapping_keys(inline: str) -> set[str]:
    """Key names declared in a YAML inline mapping, e.g. ``{GH_TOKEN: ${{ github.token }}}``."""
    return {m.group("key") for m in _FLOW_MAPPING_KEY_RE.finditer(inline)}


@dataclass
class Step:
    name: str
    job: str
    file: str
    line: int
    body: str = ""
    #: Token names visible to this step: its own env:, its job's env:, the workflow's env:.
    token_names: set[str] = field(default_factory=set)

    @property
    def where(self) -> str:
        return f"{self.file}:{self.line} [{self.job}] {self.name!r}"

    @property
    def code_body(self) -> str:
        """`body` with comment-only lines removed.

        Most steps in this repository carry multi-line `#`-prefixed prose directly under
        `- name:`, before `env:`/`run:` — pure YAML-level documentation, never executed.
        This screen's own comments quote example commands (including `gh run list`
        itself, as prose), so matching against the raw body would let a step's
        *description* of another step's defect trip its *own* detector. A shell `#`
        comment inside a real `run:` block does not execute `gh` either, so excluding
        comment-only lines is correct on both sides of that boundary.
        """
        return "\n".join(
            line for line in self.body.splitlines() if not line.strip().startswith("#")
        )


def parse_workflow(path: Path) -> list[Step]:
    """Extract every named step with its `run:` body and its VISIBLE token names.

    No PyYAML, same reasoning as ci/check_build_precondition.py: this runs in the
    lane-checks job, which installs pytest/onnx/numpy and nothing else, and a screen
    skipped because an import failed is a screen that does not exist.

    Scoping is by relative indentation, in one pass: an `env:` block is WORKFLOW-scoped
    if it sits at or above the current job's own indent (or there is no job yet),
    JOB-scoped if it sits deeper than the job's indent but we are between steps, and
    STEP-scoped if it sits deeper than the current step's own `- name:`/`- uses:` marker.
    That is what GitHub Actions' nesting means by construction, regardless of how many
    spaces any one file happens to use per level.
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    file_str = str(path).replace("\\", "/")

    steps: list[Step] = []
    workflow_env: set[str] = set()
    job_env: dict[str, set[str]] = {}

    job = "<top-level>"
    job_indent: int | None = None
    current: Step | None = None
    current_indent: int = 0
    buf: list[str] = []

    # (indent-of-the-`env:`-line, scope) — scope in {"workflow", "job", "step"}.
    env_ctx: tuple[int, str] | None = None

    def flush() -> None:
        nonlocal current, buf
        if current is not None:
            current.body = "\n".join(buf)
            steps.append(current)
        current = None
        buf = []

    def env_scope(env_indent: int) -> str:
        """Which frame an `env:` line at this indent belongs to, by nesting alone —
        deeper than the current step's marker is STEP, deeper than the job's is JOB,
        otherwise WORKFLOW. Read at call time (closes over `current`/`job_indent`,
        which change as the file is walked), not fixed at definition time."""
        if current is not None and env_indent > current_indent:
            return "step"
        if job_indent is not None and env_indent > job_indent:
            return "job"
        return "workflow"

    def record(scope: str, keys: set[str]) -> None:
        if scope == "workflow":
            workflow_env.update(keys)
        elif scope == "job":
            job_env[job].update(keys)
        elif current is not None:
            current.token_names.update(keys)

    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        indent = len(line) - len(line.lstrip())

        if env_ctx is not None and indent <= env_ctx[0]:
            env_ctx = None

        if stripped.startswith("#"):
            if current is not None and indent > current_indent:
                buf.append(line)
            continue

        m = _JOB_RE.match(line)
        if m:
            flush()
            job = m.group("job")
            job_indent = indent
            job_env.setdefault(job, set())
            env_ctx = None
            continue

        m = _ENV_INLINE_RE.match(line)
        if m:
            env_indent = len(m.group("indent"))
            record(env_scope(env_indent), _flow_mapping_keys(m.group("inline")) & set(TOKEN_NAMES))
            if current is not None and env_indent > current_indent:
                buf.append(line)
            continue

        m = _ENV_BLOCK_RE.match(line)
        if m:
            env_indent = len(m.group("indent"))
            env_ctx = (env_indent, env_scope(env_indent))
            if current is not None and env_indent > current_indent:
                buf.append(line)
            continue

        if env_ctx is not None and indent > env_ctx[0]:
            ek = _ENV_KEY_RE.match(line)
            if ek and ek.group("key") in TOKEN_NAMES:
                record(env_ctx[1], {ek.group("key")})
            if current is not None and indent > current_indent:
                buf.append(line)
            continue

        m = _STEP_START_RE.match(line)
        if m:
            flush()
            current = Step(
                name=m.group("name").strip().strip("\"'"),
                job=job,
                file=file_str,
                line=idx,
            )
            current_indent = len(m.group("indent"))
            continue
        m = _STEP_USES_ONLY_RE.match(line)
        if m:
            flush()
            current = Step(
                name=f"<uses {m.group('uses')}>",
                job=job,
                file=file_str,
                line=idx,
            )
            current_indent = len(m.group("indent"))
            continue

        if current is not None:
            if indent <= current_indent:
                flush()
                continue
            buf.append(line)

    flush()

    for step in steps:
        step.token_names |= workflow_env
        step.token_names |= job_env.get(step.job, set())
    return steps


def gh_direct_scripts(ci_dir: Path) -> dict[str, str]:
    """``ci/*.py`` files whose source shells out to the literal `gh` binary."""
    found: dict[str, str] = {}
    pat = re.compile(r'''\[\s*(?:"gh"|'gh')\s*,''')
    if not ci_dir.is_dir():
        return found
    for f in sorted(ci_dir.glob("*.py")):
        text = f.read_text(encoding="utf-8", errors="replace")
        m = pat.search(text)
        if m:
            line_no = text.count("\n", 0, m.start()) + 1
            found[f.name] = f"{f.name}:{line_no} shells out to `gh` directly"
    return found


#: A register `cmd` that carries one of these flags is not going to call `gh` for real no
#: matter what script it names — `--from-json` (check_main_is_green.py) and `--assert-
#: known-limit` (a fixed prose branch, no subprocess at all) are the two bypasses that
#: exist in this tree today. A cmd with neither is a LIVE invocation.
_REGISTER_CMD_BYPASS_FLAGS = ("--from-json", "--assert-known-limit")

#: Scripts independently verified — by reading them, not by a generic call-graph walk —
#: to run ci/open_reds.json's REAL, default register end to end, as opposed to a synthetic
#: one built in a tmp dir. This is the exact chain PR #13 run 31052604259 hit:
#: `negative_control_open_reds.py`'s LIVE arm calls `check_open_reds.py` against the real
#: file, whose `main_is_green` entry runs `check_main_is_green.py`, which calls `gh run
#: list`. It is a closed, hand-verified list rather than "does this file's text mention
#: that filename" for a concrete reason: `ci/test_lane_checks.py` ALSO spawns
#: `check_open_reds.py` as a subprocess (see its `_open_reds()` helper), but every one of
#: its call sites builds a fresh synthetic register in a tmp dir first and passes
#: `--register <tmpfile>`, so it never reaches the real `main_is_green` entry and never
#: calls `gh` — a textual-mention graph cannot tell those two shapes apart; a human
#: reading both files can.
KNOWN_REAL_REGISTER_RUNNERS: dict[str, str] = {
    "check_open_reds.py": (
        "loads --register (default ci/open_reds.json, the real file) and runs every "
        "entry's cmd via subprocess — see its run_entry()."
    ),
    "negative_control_open_reds.py": (
        'its LIVE arm calls run_screen(REGISTER) where REGISTER = HERE / "open_reds.json"'
        " — the real file, no --register override — specifically to prove the shipped "
        "register is the colour it declares."
    ),
}


def register_live_gh_entries(register_path: Path, direct: dict[str, str]) -> list[str]:
    """Entries in `register_path` whose `cmd` really calls a `direct` (gh-shelling) script
    — i.e. carries none of `_REGISTER_CMD_BYPASS_FLAGS`. Empty if the register does not
    exist, is unreadable, or every such entry has been given a bypass flag."""
    if not register_path.exists():
        return []
    try:
        doc = json.loads(register_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    hits = []
    for entry in list(doc.get("checks", [])) + list(doc.get("known_limits", [])):
        cmd = [str(t) for t in entry.get("cmd", [])]
        names = {Path(t).name for t in cmd}
        if names & set(direct) and not any(flag in cmd for flag in _REGISTER_CMD_BYPASS_FLAGS):
            hits.append(
                f"{register_path.name}: entry {entry.get('id', '?')!r} runs a LIVE "
                f"{', '.join(sorted(names & set(direct)))}"
            )
    return hits


def scripts_needing_gh(ci_dir: Path, register_path: Path) -> dict[str, str]:
    """Every ``ci/*.py`` filename that reaches `gh` for real, with a one-line reason:
    scripts that shell out to `gh` directly, plus the closed set of scripts independently
    verified to run the real open-reds register end to end while it still has a live
    entry pointing at one of those direct scripts."""
    direct = gh_direct_scripts(ci_dir)
    reasons: dict[str, str] = dict(direct)

    live_entries = register_live_gh_entries(register_path, direct)
    if live_entries:
        for name, why in KNOWN_REAL_REGISTER_RUNNERS.items():
            if (ci_dir / name).is_file():
                reasons[name] = f"{why} ({'; '.join(live_entries)})"
    return reasons


def step_gh_reason(step: Step, needing: dict[str, str]) -> str | None:
    """Why (if at all) this step reaches the GitHub API through `gh`."""
    code = step.code_body
    m = _GH_API_SUBCOMMAND_RE.search(code)
    if m:
        return f"runs `gh {m.group(1)}` directly"
    if _GH_AUTH_STATUS_RE.search(code):
        return "runs `gh auth status` directly"
    for m in _PY_SCRIPT_RUN_RE.finditer(code):
        name = Path(m.group("path")).name
        if name in needing:
            return f"runs {m.group('path')}, which {needing[name]}"
    return None


def _fail(condition: str, *lines: str) -> int:
    print(f"GH-AUTH: FAIL(condition={condition})", flush=True)
    for line in lines:
        print(line, flush=True)
    return EXIT_FAIL_CONDITION


def _error(instrument: str, *lines: str) -> int:
    print(f"GH-AUTH: ERROR(instrument={instrument})", flush=True)
    for line in lines:
        print(line, flush=True)
    return EXIT_ERROR_INSTRUMENT


def _workflow_files_under(dir_path: Path) -> list[Path]:
    """Every ``*.yml``/``*.yaml`` file under `dir_path`, recursively, sorted.

    Lets a caller name a whole workflows directory instead of listing files one at a
    time, so a workflow added later is screened from the day it exists rather than the
    day someone remembers to add its name to a command line — the concrete way a screen
    "over a subdirectory" (issue #21) would otherwise keep passing on a shrinking view
    of the tree without anybody changing a line of YAML.
    """
    return sorted({*dir_path.rglob("*.yml"), *dir_path.rglob("*.yaml")})


def screen(
    paths: list[Path], ci_dir: Path, register_path: Path, *, allow_empty_frame: bool = False
) -> int:
    missing = [p for p in paths if not p.exists()]
    if missing:
        return _error(
            "workflow_not_found",
            f"{[str(p) for p in missing]} do(es) not exist. A screen over files that "
            "are not there would pass, and a pass from a screen that read nothing is "
            "not an observation.",
        )

    expanded: list[Path] = []
    for p in paths:
        if p.is_dir():
            files = _workflow_files_under(p)
            if not files:
                return _error(
                    "empty_workflow_directory",
                    f"{p} is a directory with no `*.yml`/`*.yaml` file under it. "
                    "Screening a scope that contains no workflow files at all is the "
                    "wrong-working-directory/wrong-scope failure this screen exists to "
                    "surface, not a clean tree.",
                )
            expanded.extend(files)
        else:
            expanded.append(p)
    paths = sorted(set(expanded))

    all_steps: list[Step] = []
    for p in paths:
        all_steps.extend(parse_workflow(p))

    if not all_steps:
        return _error(
            "no_steps_parsed",
            f"No `- name:`/`- uses:` steps were found in {[str(p) for p in paths]}. "
            "Either these are not workflow files or the block structure this screen "
            "relies on has changed. UNOBSERVABLE is not zero.",
        )

    needing = scripts_needing_gh(ci_dir, register_path)

    offenders: list[str] = []
    checked = 0
    for step in all_steps:
        reason = step_gh_reason(step, needing)
        if reason is None:
            continue
        checked += 1
        if not (step.token_names & set(TOKEN_NAMES)):
            offenders.append(
                f"  {step.where}\n"
                f"    {reason}, and neither GH_TOKEN nor GITHUB_TOKEN is declared in an "
                f"`env:` block this step can see (its own, its job's, or the workflow's)."
            )

    if offenders:
        return _fail(
            "missing_token_path",
            "",
            "A step that reaches the GitHub API through `gh` with no token in scope. "
            "Without one, `gh` refuses locally with \"set the GH_TOKEN environment "
            "variable\" — the exact failure in PR #13 run 31052604259 — and whatever "
            "the step wraps (ci/check_main_is_green.py included) correctly reports "
            "ERROR(instrument), which then fails for a reason that has nothing to do "
            "with the rule it was checking.",
            *offenders,
            "",
            "Fix: add `env: {GH_TOKEN: ${{ github.token }}}` to the step (or its job), "
            "scoped to the step/job that actually calls `gh` — not to the whole "
            "workflow, which would hand every job in this file a token none of the "
            "others need.",
        )

    if checked == 0 and not allow_empty_frame:
        return _error(
            "zero_gh_reaching_subjects",
            f"0 `gh`-reaching step(s) across {len(paths)} workflow file(s): "
            f"{[str(p) for p in paths]}.",
            "That is either an honestly `gh`-free set of workflows, or this screen has "
            "been pointed at the wrong scope — a subdirectory, a stale path, the wrong "
            "working directory — and is reading nothing. `PASS — 0 gh-reaching "
            "step(s)` would be indistinguishable, on the page, from a screen that was "
            "never actually run (issue #21); a zero-subject frame is therefore an "
            "instrument error by default, not a pass.",
            "If an empty frame is genuinely the intended one, pass --allow-empty-frame "
            "to say so explicitly on the command line — never the silent default.",
        )

    print(
        f"GH-AUTH: PASS — {checked} `gh`-reaching step(s) across {len(paths)} workflow "
        f"file(s), every one with GH_TOKEN or GITHUB_TOKEN declared in scope. "
        f"{len(needing)} ci/*.py script(s) classified as reaching `gh` (directly or "
        "through ci/open_reds.json)."
        + (
            " (--allow-empty-frame: 0 gh-reaching subjects explicitly accepted.)"
            if checked == 0
            else ""
        ),
        flush=True,
    )
    return EXIT_PASS


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "workflows", nargs="*", help="workflow YAML files (or directories) to screen"
    )
    ap.add_argument("--ci-dir", default=str(CI_DIR))
    ap.add_argument("--register", default=str(DEFAULT_REGISTER))
    ap.add_argument(
        "--allow-empty-frame",
        action="store_true",
        help=(
            "accept 0 gh-reaching step(s) as PASS instead of ERROR(instrument="
            "zero_gh_reaching_subjects). Documented opt-in only — the default refuses "
            "an empty frame because it cannot be told apart from this screen having "
            "been pointed at the wrong scope (issue #21)."
        ),
    )
    if not argv:
        print(__doc__, flush=True)
        return EXIT_USAGE
    args = ap.parse_args(argv)
    if not args.workflows:
        print(__doc__, flush=True)
        return EXIT_USAGE
    return screen(
        [Path(p) for p in args.workflows],
        Path(args.ci_dir),
        Path(args.register),
        allow_empty_frame=args.allow_empty_frame,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
