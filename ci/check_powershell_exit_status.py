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

ISSUE #55: THE CAPTURE WAS NEVER WHAT MADE THE VERDICT LIE
------------------------------------------------------------

PS1, as filed against issue #49, is narrower than the class it is named after. The
defect is *a step whose printed verdict and its actual exit status can disagree*; the
named-variable capture is not what causes that, the fall-through to an implicit,
uninspected `$LASTEXITCODE` is. `ci.yml`'s **"Install Mesa lavapipe (mesa-dist-win …)"**
step has the identical fall-through with the capture removed: it calls `7z` and then
`vulkaninfoSDK.exe` directly (the call operator `&`, not a cmdlet — both set
`$LASTEXITCODE`), checks the captured *text* of the second call's output with
`Select-String`, and ends on `Write-Host "lavapipe smoke-check: OK (llvmpipe
enumerated)"` with no `exit` at all. PS1's own regex requires `$name = $LASTEXITCODE`
literally, so it never reaches this step — the step is green today only because
`vulkaninfoSDK.exe` happens to exit 0, which is latent, not proven, and indistinguishable
from an actually-verified verdict.

THE TWO RULES, AND WHY EACH IS DECIDABLE
-------------------------------------------

**PS1 — ``stale_exit_code_after_native_capture``.** A `run:` body that assigns
``$LASTEXITCODE`` to a named PowerShell variable (``$code = $LASTEXITCODE``, any name)
must end — as the LAST non-blank, non-comment line of the script — with a statement that
itself sets the step's exit status: `` exit <int>``, ``exit $LASTEXITCODE``, ``exit
$<name>`` (any variable), or ``throw``. A script that captures the code and then ends on
anything else (a `Write-Host`, a bare expression, a comment) is relying on the implicit
wrapper to propagate `$LASTEXITCODE`, and that value is whatever native command ran LAST
— which, on a step built around an intentionally-failing command, is never the step's
own verdict.

**PS2 — ``native_exit_stale_at_verdict_print``.** A `run:` body that invokes a native
command anywhere in its script (the call operator `& …`, or a bare invocation of a known
external tool — `cargo`, `rustc`, `rustup`, `python`, `pip`, `git`, `7z`, `gh`, `npm`,
`node`, or any `*.exe`) — and whose LAST non-blank, non-comment line is a `Write-Host`
verdict print with no intervening explicit `exit`/`throw` anywhere after that native
call, is relying on the SAME implicit wrapper as PS1, just without ever having named a
variable for it. The two rules share one mechanism (an un-consumed, implicit
`$LASTEXITCODE` trailer) and differ only in whether the script bothered to look at the
value before falling through — PS2 exists because "never looked" is not "safe", it is
"undecided until the native command's own exit happens to agree with the printed text".

Together PS1 and PS2 are still narrow on purpose: a script that never invokes a native
command at all is out of scope for both (there is nothing whose exit status could go
stale — the implicit trailer, if it fires, is empty or carries whatever ran in a PRIOR
step, and nothing in THIS script's text asserted otherwise). A script whose last
statement IS the native invocation itself (no verdict print follows it) is also out of
scope: the implicit trailer then correctly reports that exact command's own status,
which is what "correct by construction" actually requires — not merely "no `$name =
$LASTEXITCODE` appears here", which is the over-general shorthand issue #55 flagged in
this file's own negative control.

Bash has no such implicit trailer (its own exit status IS the last command's own status
by construction), so both rules key on PowerShell-only spellings (`$LASTEXITCODE`, the
call operator, or a small set of native tool names) rather than on `runs-on:
windows-*` — precise about the mechanism, not the platform, and no `shell:` override in
this repository changes a Windows `run:` step away from `pwsh`.

WHAT IT DOES NOT CLAIM
-----------------------

That a step's checks are individually correct, or that every exit path is reachable.
It answers one question — *can this step's own printed verdict and its actual exit
status disagree because a later, non-native line looked at (or never looked at) a value
instead of setting one?* — and it answers it about the file, not about a run. PS2's own
native-command list is a finite allowlist, not an interpreter: a step invoking a native
tool outside that list, or through some other PowerShell idiom (`$?`, `try`/`catch`
around a native call, `Start-Process -Wait -PassThru`), is outside both named rules
(recorded honestly in ``ci/lane_inventory.py``'s ``misses`` for this check).

Terminal states (R13):

    0  POWERSHELL-EXIT-STATUS: PASS
    1  POWERSHELL-EXIT-STATUS: FAIL(condition=stale_exit_code_after_native_capture)
    1  POWERSHELL-EXIT-STATUS: FAIL(condition=native_exit_stale_at_verdict_print)
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

#: PS2's native-invocation allowlist: the call operator (`& …`, which runs whatever
#: follows as a native command whether or not the result is assigned to a variable —
#: `$out = & "tool.exe" args` still sets `$LASTEXITCODE` to that tool's own exit code),
#: a short list of external tools this repository's workflows actually invoke by bare
#: name, and any direct `*.exe` invocation. This is an allowlist, not an interpreter: a
#: native tool outside it, or a native call reached through some other PowerShell idiom
#: (`Start-Process`, `$?`), is a documented miss (see the header and lane_inventory.py),
#: not a silent false negative this rule claims to close.
_NATIVE_CALL_RE = re.compile(
    r"^\s*(?:\$\w+(?:\.\w+)*\s*=\s*)?"  # optional `$var = ` / `$var.prop = ` prefix
    r"(?:&\s+\S"  # the call operator, e.g. `& "$env:...\tool.exe" --summary`
    r"|(?:cargo|rustc|rustup|python3?|pip3?|git|7z|gh|npm|node)\b"
    # A bare (unquoted) `*.exe` token used as the invoked command itself, e.g.
    # `rust\target\release\epctl.exe --probe-loader`. The character class
    # deliberately excludes `"` and whitespace: without it, a line that merely
    # ASSIGNS a URL or path string ending in `...Installer.exe"` (never invoked —
    # see "Install LunarG Vulkan SDK", which downloads that installer via
    # `Invoke-WebRequest`/`Start-Process`, neither of which sets `$LASTEXITCODE`)
    # would false-positive as a native call it never made.
    r"|[\w.\\/:$-]+\.exe\b)"
)

#: The last line of a step's script is a plain human-readable verdict announcement —
#: not a check, not an exit, just text for the log. `Write-Host` never sets
#: `$LASTEXITCODE`, so a native call earlier in the same body leaves whatever it set
#: sitting there, unconsumed and unmentioned, for the implicit wrapper to pick up.
_VERDICT_PRINT_RE = re.compile(r"^\s*Write-Host\b")


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


def _has_native_call(body: str) -> bool:
    return any(_NATIVE_CALL_RE.match(line) for line in body.splitlines())


def _last_content_index(body: str) -> int | None:
    """Line index (0-based, into `body.splitlines()`) of `_last_content_line`'s match."""
    lines = body.splitlines()
    for idx in range(len(lines) - 1, -1, -1):
        s = lines[idx].strip()
        if not s or s.startswith("#"):
            continue
        return idx
    return None


def offending_steps_ps1(wf: Workflow) -> list[tuple[Step, str]]:
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


def offending_steps_ps2(
    wf: Workflow, already_flagged: set[tuple[str, int]]
) -> list[tuple[Step, str]]:
    """Steps with an earlier native call whose LAST line is a verdict print, no exit.

    Skips anything PS1 already flagged (identified by `(file, line)`) so the same step
    is not reported twice under two condition names — PS1's capture-based shape is a
    strict subset of what PS2 also covers whenever both are present in one step.
    """
    out: list[tuple[Step, str]] = []
    for step in wf.steps:
        if (step.file, step.line) in already_flagged:
            continue
        if "run:" not in step.body and not step.body.strip():
            continue
        last = _last_content_line(step.body)
        if last is None or not _VERDICT_PRINT_RE.match(last.strip()):
            continue
        last_idx = _last_content_index(step.body)
        prior_lines = step.body.splitlines()[:last_idx] if last_idx is not None else []
        if not any(_NATIVE_CALL_RE.match(line) for line in prior_lines):
            continue
        out.append((step, last))
    return out


def audit_candidates(wf: Workflow) -> list[tuple[Step, str]]:
    """Every step this file's rules can reach at all, with its own disposition.

    Per issue #55: "audit every named candidate and record per-step dispositions rather
    than auto-inserting exits". A candidate is any step that either captures
    `$LASTEXITCODE` into a variable (PS1's trigger) or invokes a recognised native
    command (PS2's trigger) — i.e. every step for which this screen's rules have an
    opinion, whether or not that opinion is a finding.
    """
    out: list[tuple[Step, str]] = []
    for step in wf.steps:
        if "run:" not in step.body and not step.body.strip():
            continue
        captures = bool(_CAPTURE_RE.search(step.body))
        native = _has_native_call(step.body)
        if not captures and not native:
            continue
        last = _last_content_line(step.body)
        last_display = (last or "<empty script>").strip()
        if last is not None and _CONSUMING_EXIT_RE.match(last):
            out.append((
                step,
                f"SAFE — ends by explicitly consuming its exit status ({last_display!r})",
            ))
            continue
        if captures:
            out.append((
                step,
                f"FLAGGED PS1 — captures $LASTEXITCODE and does not end by consuming it "
                f"(last line: {last_display!r})",
            ))
            continue
        # native, no capture: either the native call IS the last statement (fine, the
        # implicit trailer then reports exactly that call's own status), or something
        # non-native follows it without an exit.
        if last is not None and _NATIVE_CALL_RE.match(last.strip()):
            out.append((
                step,
                "SAFE — last statement is the native invocation itself, so the "
                "implicit trailer reports that call's own status, not a stale one",
            ))
            continue
        if last is not None and _VERDICT_PRINT_RE.match(last.strip()):
            out.append((
                step,
                f"FLAGGED PS2 — a native call earlier in the body is followed by a "
                f"Write-Host verdict with no intervening exit (last line: "
                f"{last_display!r})",
            ))
            continue
        out.append((
            step,
            f"SAFE — last statement ({last_display!r}) is neither a verdict print nor "
            f"the native call itself; no known rule's shape applies",
        ))
    return out


def _error(instrument: str, *lines: str) -> int:
    print(f"POWERSHELL-EXIT-STATUS: ERROR(instrument={instrument})", flush=True)
    for line in lines:
        print(line, flush=True)
    return EXIT_ERROR_INSTRUMENT


def _emit(report: list[str], condition: str | None, summary_path: str) -> None:
    verdict = (
        "PASS" if condition is None else f"FAIL(condition={condition})"
    )
    print(f"POWERSHELL-EXIT-STATUS: {verdict}", flush=True)
    print("\n".join(report), flush=True)
    if summary_path:
        try:
            with open(summary_path, "a", encoding="utf-8") as fh:
                fh.write(f"POWERSHELL-EXIT-STATUS: {verdict}\n" + "\n".join(report) + "\n")
        except OSError:
            pass


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

    # ---- per-step audit (every named candidate, decided honestly) ------------------
    # Issue #55: record a disposition for every step either rule can reach, rather than
    # silently deciding only the ones that happen to fail today.
    audit: list[tuple[Step, str]] = []
    for wf in workflows:
        audit.extend(audit_candidates(wf))
    report.append("")
    report.append(
        f"AUDIT: {len(audit)} named candidate step(s) — a step is a candidate if it "
        "either captures $LASTEXITCODE into a variable (PS1's trigger) or invokes a "
        "recognised native command (PS2's trigger):"
    )
    for step, disposition in audit:
        report.append(f"  {step.where}\n    {disposition}")

    # ---- PS1 -------------------------------------------------------------------------
    ps1: list[tuple[Step, str]] = []
    for wf in workflows:
        ps1.extend(offending_steps_ps1(wf))
    if ps1:
        lines = [
            "",
            "PS1 — a step captures $LASTEXITCODE into a variable and its success path "
            "does not consume it:",
        ]
        for step, last in ps1:
            lines.append(
                f"  {step.where}\n"
                f"    last line of the script is: {last.strip()!r}\n"
                f"    Without an explicit `exit`, GitHub's implicit pwsh wrapper runs "
                f"`if ((Test-Path variable:\\LASTEXITCODE)) {{ exit $LASTEXITCODE }}` "
                f"after this line, reading whatever native command ran LAST — which, on "
                f"a step built around an intentionally-failing command, is never this "
                f"step's own verdict (issue #49)."
            )
        lines.append("")
        lines.append(
            "This is the exact shape both Windows 'Gate negative control' steps carried "
            "until issue #49: `$code = $LASTEXITCODE`, several checks that each `exit 1` "
            "on failure, and a final `Write-Host \"... PASSED ...\"` with no `exit 0` — "
            "so the step's real exit code stayed pinned to the gate's deliberately "
            "non-zero $code even when every one of its own checks agreed the verdict "
            "was PASS."
        )
        report += lines
        _emit(report, "stale_exit_code_after_native_capture", args.summary)
        return EXIT_FAIL_CONDITION

    # ---- PS2 -------------------------------------------------------------------------
    flagged_ps1 = {(step.file, step.line) for step, _ in ps1}
    ps2: list[tuple[Step, str]] = []
    for wf in workflows:
        ps2.extend(offending_steps_ps2(wf, flagged_ps1))
    if ps2:
        lines = [
            "",
            "PS2 — a native command runs earlier in the body and the step's success "
            "path ends on a Write-Host verdict print with no intervening exit:",
        ]
        for step, last in ps2:
            lines.append(
                f"  {step.where}\n"
                f"    last line of the script is: {last.strip()!r}\n"
                f"    No `$name = $LASTEXITCODE` capture ever names the value, so PS1 "
                f"does not reach this step — but the SAME implicit `pwsh` trailer still "
                f"runs, still reads whatever the last native command (`&`, `cargo`, "
                f"`python`, a `*.exe`, ...) set, and this step's own printed verdict "
                f"never inspected it either. Issue #55: 'Install Mesa lavapipe' carried "
                f"this exact shape — `vulkaninfoSDK.exe`'s own exit status, unread, "
                f"behind a `Write-Host \"lavapipe smoke-check: OK\"` with no `exit`."
            )
        lines.append("")
        lines.append(
            "This rule does not require a capture at all: 'never looked at the value' "
            "is not 'safe', it is 'undecided until the native command's own exit "
            "happens to agree with the printed text' — latent, not proven."
        )
        report += lines
        _emit(report, "native_exit_stale_at_verdict_print", args.summary)
        return EXIT_FAIL_CONDITION

    report.append("")
    report.append(
        "PS1: every step that captures $LASTEXITCODE into a variable ends its script "
        "by explicitly consuming it (`exit <code>`, `exit $LASTEXITCODE`, `exit "
        "$<name>`, or `throw`).\n"
        "PS2: every step that invokes a recognised native command (`&`, `cargo`, "
        "`rustc`, `rustup`, `python`, `pip`, `git`, `7z`, `gh`, `npm`, `node`, or a bare "
        "`*.exe`) either ends its script on that invocation itself, or ends on something "
        "other than an un-consumed Write-Host verdict print.\n"
        "What this claims: no step in these files can report its own verdict — captured "
        "into a variable or not — while silently exiting with an earlier, unrelated "
        "native command's stale status. What it does not claim: that the checks inside "
        "a step are individually correct, or that every native invocation is reached "
        "through a spelling this file's allowlist recognises — a native call via "
        "`Start-Process`, `$?`, or a tool name outside PS2's list is a documented miss, "
        "not a silent pass (see ci/lane_inventory.py)."
    )
    _emit(report, None, args.summary)
    return EXIT_PASS


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))

