#!/usr/bin/env python3
"""``check_build_precondition`` — a lane that could not build must be unable to report.

WHY THIS IS NOT A LANE DEFECT BUT A DEFECT CLASS
------------------------------------------------

On 2026-08-03 both device lanes carried this, verbatim, in their ``Build Vulkan EP``
step::

    if [ ! -f rust/Cargo.toml ]; then
      echo "BUILD_SKIPPED=1" >> "$GITHUB_ENV"
      exit 0
    fi

and **thirty** downstream steps across the two jobs carried ``if: env.BUILD_SKIPPED !=
'1'``. One missing tracked file — a sparse-checkout filter, a bad path, a partial clone —
and both jobs would have reported SUCCESS having compiled nothing, run nothing and
asserted nothing. Every individual step behaved exactly as written. The defect is not in
any of them.

That instance was closed by making the step ``exit 1``. **Closing an instance is not
closing the class.** The next one will not be spelled ``BUILD_SKIPPED``, will not be in
``Build Vulkan EP``, and will be introduced by someone doing something locally reasonable
— tolerating a runner that sometimes lacks a GPU, letting a nightly job pass while a
dependency is being landed. So this screen does not look for that string. It looks for
the *shape*, and it is a static screen over the workflow files so that it fires on the
pull request that introduces the next one rather than on the incident that reveals it.

THE THREE RULES, AND WHY EACH IS DECIDABLE
------------------------------------------

**BP1 — ``skip_flag_with_exit_zero``.** A step that writes ``NAME=<value>`` into
``$GITHUB_ENV`` *and* contains a success exit in the same script, where ``NAME`` is read
by any ``if:`` expression in the same workflow. That conjunction is the mechanism
exactly: a step converts a failure into a variable, exits green, and the variable then
silently deletes every step that depends on it. Neither half is a defect alone —
provisioning steps legitimately publish ``ORT_HOME``, and plenty of steps legitimately
``exit 0`` — which is why the rule is the conjunction and why nobody noticed for a week.

**BP2 — ``dead_guard``.** An ``if:`` expression reading ``env.NAME`` where nothing in the
workflow ever writes ``NAME`` — not to ``$GITHUB_ENV``, not in a workflow/job/step
``env:`` block. The guard cannot fire, so it is not doing what it appears to do, and it
is *re-armable by anyone*: the next person who adds a writer for ``NAME`` arms thirty
gates in one line, in a diff that shows one line. This is the rule that fires on the
tree as this file was written — my own fix left the thirty guards in place deliberately,
so the change would "read as one deletion rather than thirty". That was the wrong call
and this rule is how I found out.

**BP3 — ``build_step_does_not_verify_its_artifact``.** A step whose name begins with
``Build`` must, in its own script, assert the existence of the artifact it exists to
produce and fail non-zero if it is absent. ``cargo build`` exiting 0 is not evidence that
a cdylib was produced: a ``[lib] crate-type`` edit, a renamed target or a cached
no-op-with-a-stale-target all exit 0. This is the positive form of the same requirement:
a build step is not permitted to be silent about its own output, because everything
downstream of it is a claim about that output.

WHAT IT DOES NOT CLAIM
----------------------

That the lane is correct, that the build is right, or that the guards that DO exist are
the right guards. It answers one question — *can a lane report on work it did not do,
because a precondition step converted its own failure into a green?* — and it answers it
about the file, not about a run.

Terminal states (R13):

    0  BUILD-PRECONDITION: PASS
    1  BUILD-PRECONDITION: FAIL(condition=skip_flag_with_exit_zero)
    1  BUILD-PRECONDITION: FAIL(condition=dead_guard)
    1  BUILD-PRECONDITION: FAIL(condition=build_step_does_not_verify_its_artifact)
    2  usage
    4  BUILD-PRECONDITION: ERROR(instrument=...)

USAGE
    python ci/check_build_precondition.py .github/workflows/ci.yml [more.yml ...]
    python ci/check_build_precondition.py --allowlist ci/build_precondition_allowlist.json ...
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ALLOWLIST = REPO_ROOT / "ci" / "build_precondition_allowlist.json"

EXIT_PASS = 0
EXIT_FAIL_CONDITION = 1
EXIT_USAGE = 2
EXIT_ERROR_INSTRUMENT = 4

#: `echo "NAME=value" >> $GITHUB_ENV` (bash) and `echo "NAME=value" >> $env:GITHUB_ENV`
#: (PowerShell), plus the `>> "$GITHUB_ENV"` quoted form the Linux lane uses.
_ENV_WRITE_RE = re.compile(
    r"""(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=  # NAME=
        [^\n]*?                                # value, on one line
        >>\s*"?\$(?:env:)?\{?GITHUB_ENV\}?"?   # >> $GITHUB_ENV / $env:GITHUB_ENV
    """,
    re.VERBOSE,
)

#: A multi-line `run:` body can also write via `printf`/`Add-Content`; those are matched
#: by the same trailing redirect, so only the NAME= capture differs. Kept as a second
#: pattern rather than a cleverer single one: a regex that matches both by being loose is
#: a regex that matches neither reliably.
_ENV_WRITE_ADDCONTENT_RE = re.compile(
    r"Add-Content\s+[^\n]*GITHUB_ENV[^\n]*?[\"']?(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=",
)

#: `env.NAME` inside any expression — `if:` conditions and `${{ }}` interpolations.
_ENV_READ_RE = re.compile(r"\benv\.(?P<name>[A-Za-z_][A-Za-z0-9_]*)")

#: A success exit. `exit 0` covers bash and PowerShell; `return 0` and a bare `exit` in a
#: bash conditional are the same act. `exit $LASTEXITCODE` is NOT this — it propagates.
_SUCCESS_EXIT_RE = re.compile(r"^\s*(?:exit\s+0|return\s+0)\s*(?:#.*)?$", re.MULTILINE)

#: Assertions that a produced file exists, with a non-zero exit behind them.
_ARTIFACT_ASSERT_RE = re.compile(
    r"(test\s+-[fesx]\s|Test-Path|\[\s+-[fesx]\s)", re.IGNORECASE
)

_STEP_START_RE = re.compile(r"^(?P<indent>\s*)-\s+name:\s*(?P<name>.+?)\s*$")
_USES_STEP_RE = re.compile(r"^(?P<indent>\s*)-\s+uses:\s*(?P<uses>\S+)")
_JOB_RE = re.compile(r"^  (?P<job>[A-Za-z0-9_.-]+):\s*$")


@dataclass
class Step:
    name: str
    job: str
    file: str
    line: int
    body: str = ""
    if_expr: str = ""

    @property
    def where(self) -> str:
        return f"{self.file}:{self.line} [{self.job}] {self.name!r}"


@dataclass
class Workflow:
    path: Path
    text: str
    steps: list[Step] = field(default_factory=list)
    #: Every `env:` key declared at workflow, job or step level, wherever it appears.
    declared_env: set[str] = field(default_factory=set)


def parse_workflow(path: Path) -> Workflow:
    """Extract steps, their `run:` bodies and their `if:` expressions, without PyYAML.

    Deliberately no YAML dependency, for the reason ``check_lane_inventory.py`` gives:
    the lane-checks job installs pytest/onnx/numpy and nothing else, and a screen that
    is skipped on the runner because an import failed is a screen that does not exist.
    The block structure needed here is shallow — a step is a `- name:` line and every
    following line indented further than it — and indentation is the one thing YAML
    guarantees.
    """
    text = path.read_text(encoding="utf-8")
    wf = Workflow(path=path, text=text)
    lines = text.splitlines()

    job = "<top-level>"
    current: Step | None = None
    current_indent = 0
    buf: list[str] = []
    in_env_block = False
    env_block_indent = 0

    def flush() -> None:
        nonlocal current, buf
        if current is not None:
            current.body = "\n".join(buf)
            wf.steps.append(current)
        current = None
        buf = []

    for idx, line in enumerate(lines, start=1):
        m = _JOB_RE.match(line)
        if m and not line.strip().startswith("#"):
            flush()
            job = m.group("job")
            in_env_block = False

        stripped = line.strip()
        indent = len(line) - len(line.lstrip())

        # `env:` blocks anywhere: their keys are *declared* names, so a guard reading one
        # of them is live even though nothing writes it at run time.
        if stripped == "env:":
            in_env_block = True
            env_block_indent = indent
            continue
        if in_env_block:
            if stripped and indent <= env_block_indent:
                in_env_block = False
            elif stripped and not stripped.startswith("#"):
                key = stripped.split(":", 1)[0].strip()
                if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                    wf.declared_env.add(key)

        m = _STEP_START_RE.match(line)
        if m:
            flush()
            current = Step(
                name=m.group("name").strip().strip("\"'"),
                job=job,
                file=str(path).replace("\\", "/"),
                line=idx,
            )
            current_indent = len(m.group("indent"))
            continue
        m = _USES_STEP_RE.match(line)
        if m and current is None:
            continue
        if current is not None:
            if stripped and indent <= current_indent and not stripped.startswith("#"):
                flush()
                continue
            buf.append(line)
            if re.match(r"^\s*if:\s*", line):
                current.if_expr += line.split("if:", 1)[1].strip() + " "
    flush()

    # `if:` on `- uses:` steps and job-level `if:` are read from the whole file rather
    # than per step: BP2 asks whether a NAME is ever read, not by which step.
    return wf


def env_writes(text: str) -> set[str]:
    names = {m.group("name") for m in _ENV_WRITE_RE.finditer(text)}
    names |= {m.group("name") for m in _ENV_WRITE_ADDCONTENT_RE.finditer(text)}
    return names


def env_reads_in_conditions(wf: Workflow) -> dict[str, list[str]]:
    """`env.NAME` occurrences in `if:` expressions, mapped to where they were read.

    Only `if:` — an `env.NAME` inside a `run:` body is a value being used, which is not
    the shape. The shape is a *gate*.
    """
    out: dict[str, list[str]] = {}
    for line_no, line in enumerate(wf.text.splitlines(), start=1):
        if not re.match(r"^\s*(if:|-\s+if:)", line):
            continue
        for m in _ENV_READ_RE.finditer(line):
            out.setdefault(m.group("name"), []).append(
                f"{str(wf.path).replace(chr(92), '/')}:{line_no}: {line.strip()}"
            )
    return out


def load_allowlist(path: Path) -> dict:
    if not path.exists():
        return {"names": {}, "steps": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("allowlist must be a JSON object")
    data.setdefault("names", {})
    data.setdefault("steps", {})
    return data


def _fail(condition: str, *lines: str) -> int:
    print(f"BUILD-PRECONDITION: FAIL(condition={condition})", flush=True)
    for line in lines:
        print(line, flush=True)
    return EXIT_FAIL_CONDITION


def _error(instrument: str, *lines: str) -> int:
    print(f"BUILD-PRECONDITION: ERROR(instrument={instrument})", flush=True)
    for line in lines:
        print(line, flush=True)
    return EXIT_ERROR_INSTRUMENT


def screen(workflows: list[Workflow], allow: dict) -> tuple[int, str, list[str]]:
    """Apply BP1..BP3. Returns (exit code, condition token, report lines)."""
    report: list[str] = []
    allow_names: dict = allow.get("names", {})
    allow_steps: dict = allow.get("steps", {})

    all_written: set[str] = set()
    for wf in workflows:
        all_written |= env_writes(wf.text)
    all_declared: set[str] = set()
    for wf in workflows:
        all_declared |= wf.declared_env

    report.append(
        f"BUILD-PRECONDITION: frame — {len(workflows)} workflow file(s), "
        f"{sum(len(w.steps) for w in workflows)} named step(s); "
        f"{len(all_written)} name(s) written to GITHUB_ENV; "
        f"{len(all_declared)} name(s) declared in env: blocks."
    )

    # ---- BP1 ----------------------------------------------------------------
    bp1: list[str] = []
    for wf in workflows:
        gated = set(env_reads_in_conditions(wf))
        for step in wf.steps:
            written = env_writes(step.body)
            if not written:
                continue
            if not _SUCCESS_EXIT_RE.search(step.body):
                continue
            offending = sorted(n for n in written if n in gated)
            if not offending:
                continue
            if allow_steps.get(step.name):
                report.append(
                    f"BP1 allowlisted: {step.where} — {allow_steps[step.name]}"
                )
                continue
            bp1.append(
                f"  {step.where}\n"
                f"    writes {offending} to GITHUB_ENV and exits 0 in the same script.\n"
                f"    Those name(s) gate other steps via `if: env.<NAME>`, so this step "
                f"can convert its own failure into the SILENT DELETION of every step "
                f"that reads them, while reporting success itself."
            )
    if bp1:
        return EXIT_FAIL_CONDITION, "skip_flag_with_exit_zero", report + [
            "",
            "BP1 — a step that turns its own failure into a variable and exits green:",
            *bp1,
            "",
            "This is the exact form both device lanes carried until 2026-08-03: "
            "`if [ ! -f rust/Cargo.toml ]; then echo \"BUILD_SKIPPED=1\" >> $GITHUB_ENV; "
            "exit 0; fi`, with thirty downstream steps gated on the flag. One missing "
            "tracked file, thirty green steps, both lanes.",
            "If the condition really is tolerable, make the step SAY SO IN ITS OWN "
            "RESULT — a red step with a clear reason, or an allowlist entry in "
            f"{DEFAULT_ALLOWLIST.name} naming the step and the reason. What is not "
            "available is a green step whose greenness means nothing ran.",
        ]

    # ---- BP2 ----------------------------------------------------------------
    bp2: list[str] = []
    for wf in workflows:
        for name, sites in sorted(env_reads_in_conditions(wf).items()):
            if name in all_written or name in all_declared:
                continue
            if name in allow_names:
                report.append(f"BP2 allowlisted: env.{name} — {allow_names[name]}")
                continue
            bp2.append(
                f"  env.{name}: read by {len(sites)} `if:` expression(s), written by "
                f"NOTHING.\n"
                + "\n".join(f"      {s}" for s in sites[:4])
                + (f"\n      ... and {len(sites) - 4} more" if len(sites) > 4 else "")
            )
    if bp2:
        return EXIT_FAIL_CONDITION, "dead_guard", report + [
            "",
            "BP2 — a gate nothing can open or close:",
            *bp2,
            "",
            "A guard whose writer does not exist is not inert, it is DORMANT. It reads in "
            "review exactly like a live guard, and the next person who adds a single line "
            "writing that name arms every one of these at once — a one-line diff that "
            "changes the meaning of every step listed above. Delete them, or write the "
            "producer, or record the name in the allowlist with the reason.",
        ]

    # ---- BP3 ----------------------------------------------------------------
    bp3: list[str] = []
    for wf in workflows:
        for step in wf.steps:
            if not re.match(r"^Build\s", step.name):
                continue
            if "run:" not in step.body:
                continue
            if allow_steps.get(step.name):
                report.append(f"BP3 allowlisted: {step.where} — {allow_steps[step.name]}")
                continue
            if not _ARTIFACT_ASSERT_RE.search(step.body):
                bp3.append(
                    f"  {step.where}\n"
                    f"    is a build step whose script never asserts that its own output "
                    f"exists.\n"
                    f"    `cargo build` exiting 0 is not evidence that a cdylib was "
                    f"produced: a [lib] crate-type edit, a renamed target, or a cached "
                    f"no-op against a stale target directory all exit 0. Everything "
                    f"downstream of this step is a claim about an artifact this step "
                    f"never looked at."
                )
    if bp3:
        return EXIT_FAIL_CONDITION, "build_step_does_not_verify_its_artifact", report + [
            "",
            "BP3 — a build step that does not verify what it built:",
            *bp3,
        ]

    report.append("")
    report.append(
        "BP1: no step writes a gating env name and exits 0 in the same script.\n"
        "BP2: every `env.NAME` read by an `if:` has a writer or a declaration.\n"
        "BP3: every `Build *` step asserts the existence of its own artifact.\n"
        "What this claims: no lane in these files can report on work a precondition step "
        "silently skipped. What it does not claim: that the lanes are correct, that the "
        "guards which DO exist guard the right things, or that a step cannot fail to do "
        "its work for reasons that never touch GITHUB_ENV — that is "
        "ci/check_suite_productivity.py's question, and it reads runs rather than files."
    )
    return EXIT_PASS, "", report


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(add_help=True, description=__doc__)
    ap.add_argument("workflows", nargs="*", help="workflow YAML files to screen")
    ap.add_argument("--allowlist", default=str(DEFAULT_ALLOWLIST))
    ap.add_argument("--summary", default="", help="append the report to this file too")
    if not argv:
        print(__doc__, flush=True)
        return EXIT_USAGE
    args = ap.parse_args(argv)
    if not args.workflows:
        print(__doc__, flush=True)
        return EXIT_USAGE

    paths = [Path(p) for p in args.workflows]
    missing = [p for p in paths if not p.exists()]
    if missing:
        return _error(
            "workflow_not_found",
            f"{[str(p) for p in missing]} do(es) not exist. A screen over files that are "
            "not there would pass, and a pass from a screen that read nothing is the "
            "defect this file exists to end, one level up.",
        )

    try:
        allow = load_allowlist(Path(args.allowlist))
    except Exception as exc:  # noqa: BLE001
        return _error(
            "allowlist_unreadable",
            f"Could not read {args.allowlist}: {exc!r}\n"
            "An unreadable allowlist is not an empty allowlist: reading it as empty would "
            "turn every recorded human judgement into a fresh red, and the fastest way "
            "past that is to delete the judgement.",
        )

    workflows = [parse_workflow(p) for p in paths]
    empty = [str(w.path) for w in workflows if not w.steps]
    if empty:
        return _error(
            "no_steps_parsed",
            f"No `- name:` steps were found in {empty}. Either the file is not a workflow "
            "or the block structure this screen relies on has changed. UNOBSERVABLE is "
            "not zero.",
        )

    code, condition, report = screen(workflows, allow)
    if code == EXIT_PASS:
        print("BUILD-PRECONDITION: PASS", flush=True)
    else:
        print(f"BUILD-PRECONDITION: FAIL(condition={condition})", flush=True)
    for line in report:
        print(line, flush=True)
    if args.summary:
        try:
            with open(args.summary, "a", encoding="utf-8") as fh:
                fh.write("\n".join(report) + "\n")
        except OSError:
            pass
    return code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
