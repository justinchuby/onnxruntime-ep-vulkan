#!/usr/bin/env python3
"""Negative control for ``check_powershell_exit_status`` — arms counted by provenance.

Same discipline as ``negative_control_build_precondition.py``: a screen over *this
repository's own workflow files* is unusually easy to write in a way that has never
been seen failing, because the tree it runs on is, by construction, the tree its author
just fixed. Provenance:

* ``LIVE``     — this repository's own workflow file as it stands right now.
* ``REPLAYED`` — the real defect, read out of this repository's history with ``git
                 show``. Not text written to make the screen fire: the exact bytes that
                 were on ``origin/main`` at the commit issue #49 was filed against, where
                 both Windows "Gate negative control" steps carried the stale-exit-code
                 shape; and separately, the exact bytes at the commit issue #55 was filed
                 against (``d2bf65f``, PR #50 merged), where "Install Mesa lavapipe"
                 still carried the no-capture sibling shape PS1 could not reach.
* ``PLANTED``  — text written to exercise a path. Proves the path is wired; proves
                 nothing about whether that shape occurs in the wild.

The REPLAYED arms are the ones that matter: they are the defect as it actually shipped,
retrieved rather than reconstructed. If a future refactor of this screen stops catching
either, that arm goes red for a reason nobody can argue with.

Exit 0 when every arm fired as declared, 1 otherwise.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

CI_DIR = Path(__file__).resolve().parent
REPO_ROOT = CI_DIR.parent
SCRIPT = CI_DIR / "check_powershell_exit_status.py"

EXIT_PASS = 0
EXIT_FAIL_CONDITION = 1
EXIT_USAGE = 2
EXIT_ERROR_INSTRUMENT = 4

#: The commit both Windows "Gate negative control" steps still carried the bug at —
#: origin/main's HEAD when issue #49 was filed, the commit this fix branches from.
HISTORICAL_REF = "f8cfbf4"
#: The commit issue #55 was filed against (PR #50, "fix(ci): Windows gate negative
#: controls must consume their own exit status", merged). PS1 closed both Gate
#: negative-control steps there, but "Install Mesa lavapipe" — same fall-through
#: mechanism, no named-variable capture — still ended on an un-consumed Write-Host.
HISTORICAL_REF_PS2 = "d2bf65f"
WORKFLOW_REL = ".github/workflows/ci.yml"

MINIMAL_HEAD = """name: control
on: [push]
jobs:
  demo:
"""


def run(args: list[str]) -> tuple[int, str]:
    # Same UTF-8 pin as negative_control_build_precondition.py: this process's own
    # PYTHONIOENCODING does not govern how THIS subprocess.run() call decodes a child's
    # captured stdout (falls back to locale.getpreferredencoding(), cp1252 on a default
    # Windows shell), and the CHILD independently picks its own stdout encoding the same
    # way unless PYTHONIOENCODING is present in ITS environment. Both sides are pinned.
    child_env = dict(os.environ)
    child_env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(REPO_ROOT),
        env=child_env,
    )
    return proc.returncode, proc.stdout + proc.stderr


def write(tmp: Path, name: str, text: str) -> Path:
    path = tmp / name
    path.write_text(text, encoding="utf-8")
    return path


def historical_workflow(ref: str = HISTORICAL_REF) -> str | None:
    """The real ci.yml at `ref`.

    Returns None rather than raising: a shallow clone legitimately may not have the
    object, and a control that cannot read its subject must say UNOBSERVABLE rather
    than quietly dropping an arm.
    """
    proc = subprocess.run(
        ["git", "show", f"{ref}:{WORKFLOW_REL}"],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(REPO_ROOT),
    )
    if proc.returncode != 0:
        return None
    return proc.stdout


def main() -> int:
    arms: list[tuple[str, str, bool, str]] = []  # (provenance, name, ok, note)

    with tempfile.TemporaryDirectory(prefix="pwshexit-") as td:
        tmp = Path(td)

        # ---- LIVE --------------------------------------------------------------
        rc, out = run([str(REPO_ROOT / WORKFLOW_REL)])
        arms.append((
            "LIVE",
            "this repository's own ci.yml passes (PS1's two steps and PS2's "
            "'Install Mesa lavapipe' all now `exit 0`)",
            rc == EXIT_PASS and "POWERSHELL-EXIT-STATUS: PASS" in out,
            "the green arm: a screen that only ever reddens is a lock on a broken state",
        ))

        # ---- REPLAYED: the defect as it really shipped --------------------------
        hist = historical_workflow()
        if hist is None:
            arms.append((
                "REPLAYED",
                f"the real stale-exit-code shape at {HISTORICAL_REF} goes red",
                False,
                f"could not `git show {HISTORICAL_REF}:{WORKFLOW_REL}` — UNOBSERVABLE, "
                "and an unreadable subject is not a passing arm",
            ))
        else:
            path = write(tmp, "historical-ci.yml", hist)
            rc, out = run([str(path)])
            arms.append((
                "REPLAYED",
                f"the real stale-exit-code shape at {HISTORICAL_REF} goes red",
                rc == EXIT_FAIL_CONDITION
                and "stale_exit_code_after_native_capture" in out,
                "PS1 on the exact bytes that shipped until issue #49",
            ))
            arms.append((
                "REPLAYED",
                "and it names BOTH known offending steps, not just one",
                "no ICD must produce UNATTRIBUTED" in out
                and "a declined artifact must produce UNATTRIBUTED" in out,
                "issue #49 was filed against the second step only; the first carries "
                "the identical shape and would have gone red the moment ICD "
                "suppression actually took on an elevated Windows runner",
            ))

        # ---- REPLAYED: issue #55's PS2 defect, as it really shipped --------------
        hist_ps2 = historical_workflow(HISTORICAL_REF_PS2)
        if hist_ps2 is None:
            arms.append((
                "REPLAYED",
                f"the real no-capture lavapipe shape at {HISTORICAL_REF_PS2} goes red",
                False,
                f"could not `git show {HISTORICAL_REF_PS2}:{WORKFLOW_REL}` — "
                "UNOBSERVABLE, and an unreadable subject is not a passing arm",
            ))
        else:
            path = write(tmp, "historical-ci-ps2.yml", hist_ps2)
            rc, out = run([str(path)])
            arms.append((
                "REPLAYED",
                f"the real no-capture 'Install Mesa lavapipe' shape at "
                f"{HISTORICAL_REF_PS2} goes red under PS2",
                rc == EXIT_FAIL_CONDITION
                and "native_exit_stale_at_verdict_print" in out
                and "Install Mesa lavapipe" in out,
                "PS2 on the exact bytes PR #50 shipped (issue #49's fix, before issue "
                "#55's PS2 widening) -- PS1 alone leaves this tree PASS, because "
                "'Install Mesa lavapipe' never names a variable for $LASTEXITCODE",
            ))
            arms.append((
                "REPLAYED",
                "and PS1 alone would have missed it (the gap issue #55 was filed over)",
                "stale_exit_code_after_native_capture" not in out,
                "confirms the residual: no `$name = $LASTEXITCODE` capture appears in "
                "this step, so PS1's own condition token never fires on it -- only "
                "PS2's wider, capture-free rule does",
            ))

        # ---- PLANTED: the shape, on a different variable name and step name -----
        rc, out = run([str(write(tmp, "planted-basic.yml", MINIMAL_HEAD + """    steps:
      - name: Some other verdict step
        run: |
          $out = python some_gate.py 2>&1 | Out-String
          $rc = $LASTEXITCODE
          Write-Host $out
          if ($rc -eq 0) {
            Write-Error "unexpected pass"
            exit 1
          }
          Write-Host "PASSED: gate reported the expected failure and exited $rc."
"""))])
        arms.append((
            "PLANTED",
            "PS1 fires on a different variable name and a different step name",
            rc == EXIT_FAIL_CONDITION and "stale_exit_code_after_native_capture" in out,
            "the rule is the shape (capture, then a non-exiting success path), not the "
            "literal spelling `$code` or the literal step name",
        ))

        # ---- PLANTED: the fix — explicit exit 0 on the success path clears it ---
        rc, out = run([str(write(tmp, "planted-fixed.yml", MINIMAL_HEAD + """    steps:
      - name: Some other verdict step
        run: |
          $out = python some_gate.py 2>&1 | Out-String
          $rc = $LASTEXITCODE
          Write-Host $out
          if ($rc -eq 0) {
            Write-Error "unexpected pass"
            exit 1
          }
          Write-Host "PASSED: gate reported the expected failure and exited $rc."
          exit 0
"""))])
        arms.append((
            "PLANTED",
            "an explicit `exit 0` on the success path clears the same step",
            rc == EXIT_PASS,
            "this is the actual fix applied to both Windows steps for issue #49",
        ))

        # ---- PLANTED: `exit $LASTEXITCODE` directly at the end also clears it ---
        rc, out = run([str(write(tmp, "planted-passthrough.yml", MINIMAL_HEAD + """    steps:
      - name: A pass-through verdict step
        run: |
          python some_gate.py
          $rc = $LASTEXITCODE
          Write-Host "gate exited $rc"
          exit $LASTEXITCODE
"""))])
        arms.append((
            "PLANTED",
            "ending on `exit $LASTEXITCODE` directly (propagate, not brand) is clean",
            rc == EXIT_PASS,
            "PS1 does not require exit 0 specifically — only that the step's own last "
            "line is the thing that sets its exit status, not an earlier stale one",
        ))

        # ---- PLANTED: a step with no capture and no dangling verdict print stays
        # clean under BOTH rules — the narrow, uncontroversial case ----------------
        rc, out = run([str(write(tmp, "planted-no-capture.yml", MINIMAL_HEAD + """    steps:
      - name: Ordinary build step
        run: |
          cargo build --release
          if (-not (Test-Path "target\\release\\x.dll")) {
            Write-Error "artifact missing"
            exit 1
          }
"""))])
        # Corrected per issue #55: the note this arm carried before was true of THIS
        # example (its last statement is the `if (Test-Path ...)` artifact check
        # itself, not a `Write-Host` quoting a conclusion reached earlier) but
        # over-general as written -- it read as a blanket amnesty for "no `$name =
        # $LASTEXITCODE` capture appears here", and 'Install Mesa lavapipe' has exactly
        # that (no capture) while NOT being correct by construction: its last
        # statement IS a Write-Host verdict print sitting downstream of
        # vulkaninfoSDK.exe's unread exit code. PS2 exists for that shape; this arm
        # only proves the narrower case where no verdict print follows the native call
        # at all, so there is nothing for the implicit trailer to get right or wrong.
        arms.append((
            "PLANTED",
            "a step that never captures $LASTEXITCODE and never prints a dangling "
            "verdict after its last native call is not flagged",
            rc == EXIT_PASS,
            "true only because this step's LAST statement is the artifact check "
            "itself, with no separate Write-Host verdict downstream of it -- not "
            "because it 'never captures the code' in general (see PS2 and the "
            "REPLAYED lavapipe arms above, which never capture anything either)",
        ))

        # ---- PLANTED: accumulate-then-exit ($rc pattern) stays clean ------------
        rc, out = run([str(write(tmp, "planted-accumulate.yml", MINIMAL_HEAD + """    steps:
      - name: Two-part check
        run: |
          $rcAll = 0
          python check_a.py
          if ($LASTEXITCODE -ne 0) { $rcAll = $LASTEXITCODE }
          python check_b.py
          if ($LASTEXITCODE -ne 0) { $rcAll = $LASTEXITCODE }
          exit $rcAll
"""))])
        arms.append((
            "PLANTED",
            "accumulating into a variable across two commands and exiting it is clean",
            rc == EXIT_PASS,
            "this is the real shape of the Windows lane's own flake-witness step — "
            "captures are fine as long as the last line consumes one",
        ))

        # ---- PLANTED: a bare comment as the last line still counts as unconsumed
        rc, out = run([str(write(tmp, "planted-trailing-comment.yml", MINIMAL_HEAD + """    steps:
      - name: Verdict with a trailing comment only
        run: |
          python some_gate.py
          $code = $LASTEXITCODE
          if ($code -eq 0) { Write-Error "bad"; exit 1 }
          Write-Host "PASSED"
          # done
"""))])
        arms.append((
            "PLANTED",
            "a trailing comment does not itself consume the captured code",
            rc == EXIT_FAIL_CONDITION and "stale_exit_code_after_native_capture" in out,
            "the check must look past trailing comments to the last REAL statement, "
            "not stop at whatever the last line of text happens to be",
        ))

        # ==== PS2 arms: the no-capture sibling shape (issue #55) ===================

        # ---- PLANTED: a bare `&`-called tool, then a dangling Write-Host verdict --
        rc, out = run([str(write(tmp, "planted-ps2-basic.yml", MINIMAL_HEAD + """    steps:
      - name: Some other smoke-check step
        run: |
          $probe = & "C:\\tools\\probeToolSDK.exe" --summary 2>&1
          Write-Host $probe
          if (-not ($probe | Select-String -Quiet -SimpleMatch "ready")) {
            Write-Error "probe did not report ready"
            exit 1
          }
          Write-Host "probe smoke-check: OK (ready)"
"""))])
        arms.append((
            "PLANTED",
            "PS2 fires on the shape under a different tool, path, and step name",
            rc == EXIT_FAIL_CONDITION and "native_exit_stale_at_verdict_print" in out,
            "the rule is the shape (a native call whose exit is never read, followed "
            "by an unrelated Write-Host verdict with no exit), not the literal tool "
            "name `vulkaninfoSDK.exe` or the literal step name",
        ))

        # ---- PLANTED: the fix — explicit exit 0 on the same shape clears it ------
        rc, out = run([str(write(tmp, "planted-ps2-fixed.yml", MINIMAL_HEAD + """    steps:
      - name: Some other smoke-check step
        run: |
          $probe = & "C:\\tools\\probeToolSDK.exe" --summary 2>&1
          Write-Host $probe
          if (-not ($probe | Select-String -Quiet -SimpleMatch "ready")) {
            Write-Error "probe did not report ready"
            exit 1
          }
          Write-Host "probe smoke-check: OK (ready)"
          exit 0
"""))])
        arms.append((
            "PLANTED",
            "an explicit `exit 0` on the same shape clears PS2, same as PS1",
            rc == EXIT_PASS,
            "this is the actual fix applied to 'Install Mesa lavapipe' for issue #55",
        ))

        # ---- PLANTED: a bare *.exe invocation (no `&`) reaches PS2 too -----------
        rc, out = run([str(write(tmp, "planted-ps2-bare-exe.yml", MINIMAL_HEAD + """    steps:
      - name: Probe the loader directly
        run: |
          rust\\target\\release\\probeCtl.exe --probe-loader
          Write-Host "loader probe finished"
"""))])
        arms.append((
            "PLANTED",
            "a bare, unquoted `*.exe` invocation (no call operator) also reaches PS2",
            rc == EXIT_FAIL_CONDITION and "native_exit_stale_at_verdict_print" in out,
            "PS2's native-call detection is not limited to the `&` call operator: a "
            "path ending in `.exe` invoked directly, as this repo's own 'Probe Vulkan "
            "loader' step does, sets $LASTEXITCODE exactly the same way",
        ))

        # ---- PLANTED: a known bare tool name (git) earlier reaches PS2 too -------
        rc, out = run([str(write(tmp, "planted-ps2-known-tool.yml", MINIMAL_HEAD + """    steps:
      - name: Tag check step
        run: |
          git fetch --tags
          Write-Host "tag fetch complete"
"""))])
        arms.append((
            "PLANTED",
            "a known bare tool name (git, cargo, python, ...) earlier also reaches PS2",
            rc == EXIT_FAIL_CONDITION and "native_exit_stale_at_verdict_print" in out,
            "the allowlist of bare tool names is deliberately explicit and small (see "
            "_NATIVE_CALL_RE) -- this proves it is wired for a tool this repository "
            "actually invokes by bare name (`git`), not just for `&`-prefixed calls",
        ))

        # ---- PLANTED (false-positive guard): a *.exe substring inside a quoted URL
        # or path VALUE -- never invoked -- must NOT trip PS2 ----------------------
        rc, out = run([str(write(tmp, "planted-ps2-url-string.yml", MINIMAL_HEAD + """    steps:
      - name: Install some other SDK
        run: |
          $url = "https://example.invalid/download/SomeSdk-Installer.exe"
          Write-Host "Downloading SDK from $url"
          Invoke-WebRequest -Uri $url -OutFile "$env:TEMP\\SomeSdk.exe"
          Start-Process -Wait -FilePath "$env:TEMP\\SomeSdk.exe" -ArgumentList '/quiet'
          Write-Host "SDK installed"
"""))])
        arms.append((
            "PLANTED",
            "a `.exe` substring inside a quoted URL/path VALUE is not a native call "
            "and must not trip PS2 (regression guard for the false positive this "
            "screen's own first draft produced on 'Install LunarG Vulkan SDK')",
            rc == EXIT_PASS,
            "`Invoke-WebRequest` and `Start-Process` are cmdlets, not the call "
            "operator -- neither sets $LASTEXITCODE, and a `*.exe` token that only "
            "ever appears inside quotes (as a URL or a -FilePath argument) was never "
            "invoked as a command in its own right; PS2's native-call regex requires "
            "an unquoted `*.exe` token or the `&` call operator, not a bare substring "
            "match anywhere on the line",
        ))

        # ---- PLANTED (false-positive guard): a verdict print with NO earlier
        # native call at all must not trip PS2 --------------------------------------
        rc, out = run([str(write(tmp, "planted-ps2-no-native.yml", MINIMAL_HEAD + """    steps:
      - name: Pure cmdlet step
        run: |
          $items = Get-ChildItem -Recurse -Path "C:\\some\\dir"
          New-Item -ItemType Directory -Force -Path "C:\\some\\other" | Out-Null
          Write-Host "housekeeping complete: $($items.Count) item(s)"
"""))])
        arms.append((
            "PLANTED",
            "a Write-Host verdict with no native call anywhere earlier is not flagged",
            rc == EXIT_PASS,
            "nothing in this step's body could have left a stale native exit code "
            "behind, so there is nothing for the implicit trailer to disagree with",
        ))

        # ---- PLANTED: instrument paths -------------------------------------------
        rc, out = run([str(tmp / "does-not-exist.yml")])
        arms.append((
            "PLANTED",
            "a workflow that is not there is ERROR(instrument), never a pass",
            rc == EXIT_ERROR_INSTRUMENT and "workflow_not_found" in out,
            "a screen that read nothing must not report a clean tree",
        ))

        rc, out = run([str(write(tmp, "empty.yml", "name: nothing\non: [push]\n"))])
        arms.append((
            "PLANTED",
            "a file with no steps is ERROR(instrument=no_steps_parsed)",
            rc == EXIT_ERROR_INSTRUMENT and "no_steps_parsed" in out,
            "UNOBSERVABLE is not zero — this screen has no YAML parser and must say so "
            "when the block structure it relies on stops matching",
        ))

        rc, out = run([])
        arms.append((
            "PLANTED",
            "no arguments prints usage and exits 2, never 0",
            rc == EXIT_USAGE,
            "a screen invoked with nothing must not report a clean tree",
        ))

    width = max(len(name) for _, name, _, _ in arms)
    failures = 0
    for prov, name, ok, note in arms:
        mark = "ok  " if ok else "FAIL"
        suffix = f"   ({note})" if note else ""
        print(f"  [{prov:<8}] {mark}  {name.ljust(width)}{suffix}")
        if not ok:
            failures += 1

    counts = {"LIVE": 0, "REPLAYED": 0, "PLANTED": 0}
    for prov, _, _, _ in arms:
        counts[prov] += 1
    print()
    print(
        f"NEGATIVE-CONTROL: {counts['LIVE']} LIVE / {counts['REPLAYED']} REPLAYED / "
        f"{counts['PLANTED']} PLANTED."
    )
    print(
        "NEGATIVE-CONTROL: the REPLAYED arms are the load-bearing ones. Every PLANTED "
        "arm is a defect written by the person who wrote the rule that catches it; the "
        f"REPLAYED arms are the defect as it really stood at {HISTORICAL_REF}, "
        "retrieved with `git show` rather than reconstructed from memory."
    )
    if failures:
        print(f"NEGATIVE-CONTROL: FAIL(condition=arm_did_not_fire) — {failures} arm(s).")
        return 1
    print("NEGATIVE-CONTROL: PASS — every arm fired as declared.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
