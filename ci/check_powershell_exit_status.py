#!/usr/bin/env python3
"""``check_powershell_exit_status`` — a verdict step must consume its OWN exit status.

WHY THIS IS A DEFECT CLASS AND NOT ONE STEP (issue #49)
--------------------------------------------------------

The Windows copy of "Gate negative control — a declined artifact must produce
UNATTRIBUTED" (``.github/workflows/ci.yml``) captured ``ci/gate_chain_fp32.py``'s
INTENTIONALLY non-zero exit code into ``$code``, printed the human-readable verdict
with ``Write-Host``, and then simply ended the script. ``Write-Host`` never touches
``$LASTEXITCODE``. GitHub Actions' generated wrapper for a `pwsh` step
(``pwsh -command ". '{0}'"``) appends an implicit::

    if ((Test-Path variable:\\LASTEXITCODE)) { exit $LASTEXITCODE }

after the script body — so the step's real exit code was still ``$code``, the gate's
deliberately non-zero exit, even though the step's own log said ``NEGATIVE CONTROL
PASSED``. The sibling "no ICD" control (same file) has the identical shape: it too
captures the gate's exit code into ``$code`` and falls through a `Write-Host` with no
trailing `exit`. Two steps, one root cause, discovered because PR #45 finally let the
Windows lane run far enough to reach the second of them.

Closing the two instances by hand does not close the class: the shape is "a PowerShell
step captures `$LASTEXITCODE` into a named variable, uses it to print or brand its OWN
verdict, and reaches a success path whose last statement is not an explicit `exit`" —
and nothing stops a future step from doing that with a different variable name, in a
different job, for a different reason. This is a static screen over the workflow text so
it fires on the pull request that introduces the next one, the same way
``check_build_precondition.py`` does for the ``BUILD_SKIPPED`` shape.

THE ONE RULE, AND WHY IT IS DECIDABLE
--------------------------------------

**PS1 — ``stale_exit_code_after_native_capture``.** A `run:` body that assigns
``$LASTEXITCODE`` to a named PowerShell variable (``$code = $LASTEXITCODE``, any name)
must end — as the LAST non-blank, non-comment line of the script — with a statement that
itself sets the step's exit status: `` exit <int>``, ``exit $LASTEXITCODE``, ``exit
$<name>`` (any variable), or ``throw``. A script that captures the code and then ends on
anything else (a `Write-Host`, a bare expression, a comment) is relying on the implicit
wrapper to propagate `$LASTEXITCODE`, and that value is whatever native command ran LAST
— which, on a step built around an intentionally-failing command, is never the step's
own verdict.

This is deliberately narrow: a script that never captures `$LASTEXITCODE` into a variable
is not in scope (there is nothing to grow stale — the implicit trailer already carries
the exit code of whatever ran last, which is correct when nothing has second-guessed
it). Bash has no such implicit trailer (its own exit status IS the last command's own
status by construction), so the rule keys on the PowerShell-only spelling
`$LASTEXITCODE` rather than on `runs-on: windows-*` — it is precise about the mechanism,
not the platform, and no `shell:` override in this repository changes a Windows `run:`
step away from `pwsh`.

WHAT IT DOES NOT CLAIM
-----------------------

That a step's checks are individually correct, or that every exit path is reachable.
It answers one question — *can this step's own printed verdict and its actual exit
status disagree because a later, non-native line looked at a variable instead of
setting one?* — and it answers it about the file, not about a run.

Terminal states (R13):

    0  POWERSHELL-EXIT-STATUS: PASS
    1  POWERSHELL-EXIT-STATUS: FAIL(condition=stale_exit_code_after_native_capture)
    2  usage
    4  POWERSHELL-EXIT-STATUS: ERROR(instrument=...)

USAGE
    python ci/check_powershell_exit_status.py .github/workflows/ci.yml [more.yml ...]
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

EXIT_PASS = 0
EXIT_FAIL_CONDITION = 1
EXIT_USAGE = 2
EXIT_ERROR_INSTRUMENT = 4

_STEP_START_RE = re.compile(r"^(?P<indent>\s*)-\s+name:\s*(?P<name>.+?)\s*$")
_USES_STEP_RE = re.compile(r"^(?P<indent>\s*)-\s+uses:\s*(?P<uses>\S+)")
_JOB_RE = re.compile(r"^  (?P<job>[A-Za-z0-9_.-]+):\s*$")

#: `$name = $LASTEXITCODE`, anywhere in the body (including inside a one-line `if { }`
#: block, e.g. `if ($LASTEXITCODE -ne 0) { $rc = $LASTEXITCODE }`) — that shape captures
#: the code into a variable just as much as a bare assignment does.
_CAPTURE_RE = re.compile(r"\$(?P<name>\w+)\s*=\s*\$LASTEXITCODE\b")

#: A line that itself sets the step's own exit status, so anything captured earlier is
#: consumed rather than left to the implicit wrapper.
_CONSUMING_EXIT_RE = re.compile(
    r"^\s*(?:exit\s+(?:\$LASTEXITCODE|\$\w+|-?\d+)|throw\b.*)\s*(?:#.*)?$"
)


@dataclass
class Step:
    name: str
    job: str
    file: str
    line: int
    body: str = ""

    @property
    def where(self) -> str:
        return f"{self.file}:{self.line} [{self.job}] {self.name!r}"


@dataclass
class Workflow:
    path: Path
    text: str
    steps: list[Step] = field(default_factory=list)


def parse_workflow(path: Path) -> Workflow:
    """Extract steps and their `run:` bodies, without a YAML parser.

    Same rationale as ``check_build_precondition.py``: the lane-checks job installs
    only pytest/onnx/numpy, and a screen skipped because an import failed is a screen
    that does not exist. A step is a `- name:` line and everything indented further
    than it, and indentation is the one structural guarantee YAML gives without a
    parser.
    """
    text = path.read_text(encoding="utf-8")
    wf = Workflow(path=path, text=text)
    lines = text.splitlines()

    job = "<top-level>"
    current: Step | None = None
    current_indent = 0
    buf: list[str] = []

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

        stripped = line.strip()
        indent = len(line) - len(line.lstrip())

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
    flush()
    return wf


def _last_content_line(body: str) -> str | None:
    """The last non-blank, non-comment line of a `run:` body.

    A line that is `# comment` only — not a trailing `# comment` on a real statement,
    which `_CONSUMING_EXIT_RE` already tolerates on the exit line itself.
    """
    for line in reversed(body.splitlines()):
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            continue
        return line
    return None


def offending_steps(wf: Workflow) -> list[tuple[Step, str]]:
    """Steps that capture `$LASTEXITCODE` and do not end by consuming it explicitly."""
    out: list[tuple[Step, str]] = []
    for step in wf.steps:
        if "run:" not in step.body and not step.body.strip():
            continue
        captures = sorted({m.group("name") for m in _CAPTURE_RE.finditer(step.body)})
        if not captures:
            continue
        last = _last_content_line(step.body)
        if last is not None and _CONSUMING_EXIT_RE.match(last):
            continue
        out.append((step, last or "<empty script>"))
    return out


def _error(instrument: str, *lines: str) -> int:
    print(f"POWERSHELL-EXIT-STATUS: ERROR(instrument={instrument})", flush=True)
    for line in lines:
        print(line, flush=True)
    return EXIT_ERROR_INSTRUMENT


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(add_help=True, description=__doc__)
    ap.add_argument("workflows", nargs="*", help="workflow YAML files to screen")
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
            f"{[str(p) for p in missing]} do(es) not exist. A screen over files that "
            "are not there would pass, and a pass from a screen that read nothing is "
            "the defect this file exists to end, one level up.",
        )

    workflows = [parse_workflow(p) for p in paths]
    empty = [str(w.path) for w in workflows if not w.steps]
    if empty:
        return _error(
            "no_steps_parsed",
            f"No `- name:` steps were found in {empty}. Either the file is not a "
            "workflow or the block structure this screen relies on has changed. "
            "UNOBSERVABLE is not zero.",
        )

    report: list[str] = [
        f"POWERSHELL-EXIT-STATUS: frame — {len(workflows)} workflow file(s), "
        f"{sum(len(w.steps) for w in workflows)} named step(s) scanned."
    ]

    offenders: list[tuple[Step, str]] = []
    for wf in workflows:
        offenders.extend(offending_steps(wf))

    if offenders:
        lines = [
            "",
            "PS1 — a step captures $LASTEXITCODE into a variable and its success path "
            "does not consume it:",
        ]
        for step, last in offenders:
            lines.append(
                f"  {step.where}\n"
                f"    last line of the script is: {last.strip()!r}\n"
                f"    Without an explicit `exit`, GitHub's implicit pwsh wrapper runs "
                f"`if ((Test-Path variable:\\LASTEXITCODE)) {{ exit $LASTEXITCODE }}` "
                f"after this line, reading whatever native command ran LAST — which, on "
                f"a step built around an intentionally-failing command, is never this "
                f"step's own verdict (issue #49)."
            )
        lines.append(
            ""
        )
        lines.append(
            "This is the exact shape both Windows 'Gate negative control' steps carried "
            "until issue #49: `$code = $LASTEXITCODE`, several checks that each `exit 1` "
            "on failure, and a final `Write-Host \"... PASSED ...\"` with no `exit 0` — "
            "so the step's real exit code stayed pinned to the gate's deliberately "
            "non-zero $code even when every one of its own checks agreed the verdict "
            "was PASS."
        )
        report += lines
        print(f"POWERSHELL-EXIT-STATUS: FAIL(condition=stale_exit_code_after_native_capture)", flush=True)
        print("\n".join(report), flush=True)
        if args.summary:
            try:
                with open(args.summary, "a", encoding="utf-8") as fh:
                    fh.write(
                        "POWERSHELL-EXIT-STATUS: "
                        "FAIL(condition=stale_exit_code_after_native_capture)\n"
                        + "\n".join(report) + "\n"
                    )
            except OSError:
                pass
        return EXIT_FAIL_CONDITION

    report.append("")
    report.append(
        "PS1: every step that captures $LASTEXITCODE into a variable ends its script "
        "by explicitly consuming it (`exit <code>`, `exit $LASTEXITCODE`, `exit "
        "$<name>`, or `throw`).\n"
        "What this claims: no step in these files can report its own verdict while "
        "silently exiting with an earlier, unrelated command's stale status. What it "
        "does not claim: that the checks inside a step are individually correct, or "
        "that a step which never captures the code at all is exiting on the right "
        "command's status — that is a question about the step's whole body, not this "
        "one shape."
    )
    print("POWERSHELL-EXIT-STATUS: PASS", flush=True)
    print("\n".join(report), flush=True)
    if args.summary:
        try:
            with open(args.summary, "a", encoding="utf-8") as fh:
                fh.write("POWERSHELL-EXIT-STATUS: PASS\n" + "\n".join(report) + "\n")
        except OSError:
            pass
    return EXIT_PASS


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
