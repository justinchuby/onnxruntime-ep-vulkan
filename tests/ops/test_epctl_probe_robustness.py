"""Issue #1 hardening: the ICD/layer-removal witnesses must survive real machines.

WHY THIS FILE EXISTS
====================
Real CI (run 31094738484, Windows job 92593900456, main before the modelrunner
regression) failed ``test_criterion4_icd_polarity_witness`` with BOTH suppression arms
reporting ``probe_report_unreadable`` — not ``icd_suppression_ineffective``, which is
what a genuinely-still-present ICD reads as.  ``ci/check_icd_suppression.classify``'s
gate-verdict regex requires the literal character ``§`` (U+00A7), which
``instance.rs`` prints as UTF-8 (bytes ``0xC2 0xA7``).  The child invocations in
``test_criterion_4_5_witness.py``, ``test_validation.py`` and
``tests/ops/_verdict.py::run_subprocess_checked`` all called ``subprocess.run(...,
text=True)`` with **no explicit encoding** — which decodes with
``locale.getpreferredencoding()``, not UTF-8, on an English Windows runner.  Under
``cp1252`` those two bytes decode to ``"Â§"``, which matches neither
``GATE_VERDICT_RE`` nor ``NO_ICD_RE`` — so a perfectly healthy *or* perfectly suppressed
report is misclassified as unreadable, on every run, regardless of whether suppression
actually worked.

This is exactly Link's own 2026-08-01 finding one layer down: "a detector that fires on
every input is not a detector, it is a constant" — here the constant is the *decode*,
not the regex.

Three things are hardened here, per issue #1's brief:

1. a static, deterministic **wiring test**: every ``subprocess.run``/
   ``run_subprocess_checked`` call site that captures an ``epctl`` child's output must
   name an explicit encoding (never a bare ``text=True``), so this class of bug cannot
   silently return;
2. a **behavioural** round-trip test proving the fix actually preserves non-ASCII bytes
   across a real child-process boundary, not just that a keyword argument is present;
3. **paths with spaces** — an EP library installed under a directory whose name contains
   a space (e.g. a `Program Files`-style path) must not change which ``epctl`` binary is
   located or how it is invoked, since these tests pass the binary path as an argv
   element, never through a shell.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

# The three files known (2026-08-05, issue #1) to spawn an `epctl` child and parse its
# output for non-ASCII markers such as "§7.2 capability gate".
_EPCTL_CALLER_FILES = (
    HERE / "test_criterion_4_5_witness.py",
    HERE / "test_validation.py",
    HERE / "_verdict.py",
)


def _subprocess_run_calls(path: Path) -> "list[ast.Call]":
    """Every `subprocess.run(...)` call node in *path*, source-parsed, not string-grepped.

    AST rather than a regex: a call spread across several lines (as every call site here
    is, for readability) must not be missed just because the keyword landed on a
    different physical line than `subprocess.run`.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_subprocess_run = (
            isinstance(func, ast.Attribute)
            and func.attr == "run"
            and isinstance(func.value, ast.Name)
            and func.value.id == "subprocess"
        )
        if is_subprocess_run:
            calls.append(node)
    return calls


def test_every_epctl_subprocess_call_names_an_explicit_encoding():
    """The static drift check: a bare `text=True` must never come back.

    Asserts on the AST, not on `grep "encoding="` somewhere in the file — a call site
    could gain an unrelated second `subprocess.run` elsewhere in the same file with no
    encoding, and a substring search would not catch which call that was. This walks
    every call individually.
    """
    offenders: list[str] = []
    total_calls = 0
    for path in _EPCTL_CALLER_FILES:
        assert path.is_file(), f"expected source file missing: {path}"
        for call in _subprocess_run_calls(path):
            total_calls += 1
            kw_names = {kw.arg for kw in call.keywords if kw.arg is not None}
            has_explicit_encoding = "encoding" in kw_names
            bare_text_true = False
            for kw in call.keywords:
                if kw.arg == "text" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                    bare_text_true = True
            if bare_text_true and not has_explicit_encoding:
                offenders.append(f"{path.name}:{call.lineno}")
            elif not has_explicit_encoding and (kw_names & {"capture_output", "stdout", "stderr"}):
                # Captures output but names neither `text=True` nor `encoding=` — bytes
                # mode. Not itself the bug this file guards against (no implicit locale
                # decode happens), so it is not an offender, but every current call site
                # in these three files does capture text, and a switch to bytes mode
                # here would silently defeat this test's purpose. Recorded, not asserted,
                # because a bytes-mode call is legitimate elsewhere in this codebase.
                pass

    assert total_calls >= 3, (
        f"expected to find subprocess.run call sites in {[p.name for p in _EPCTL_CALLER_FILES]}; "
        f"found {total_calls}. This test's file list is stale if the epctl-invocation code moved."
    )
    assert not offenders, (
        "subprocess.run(..., text=True) with no explicit encoding= at "
        f"{offenders}. On Windows this decodes with the platform locale, not UTF-8, and "
        "silently corrupts non-ASCII markers such as the loader's '§7.2 capability gate' "
        "line -- see run 31094738484 / issue #1. Use encoding=\"utf-8\", errors=\"replace\"."
    )


def test_every_epctl_subprocess_call_uses_utf8_not_some_other_encoding():
    """A drift the first test cannot see: `encoding="cp1252"` would also pass it.

    Names the value, not just its presence, so a future edit cannot "fix" the first test
    by pinning the wrong codec.
    """
    for path in _EPCTL_CALLER_FILES:
        for call in _subprocess_run_calls(path):
            for kw in call.keywords:
                if kw.arg != "encoding":
                    continue
                assert isinstance(kw.value, ast.Constant) and kw.value.value == "utf-8", (
                    f"{path.name}:{call.lineno} sets encoding={ast.dump(kw.value)!r}, "
                    'expected the literal "utf-8".'
                )


def test_utf8_encoding_survives_a_real_child_process_boundary(tmp_path):
    """Behavioural half: prove the fix, not just the keyword argument.

    Spawns a real child (a throwaway script, standing in for `epctl --probe-loader`)
    that writes the exact byte sequence `instance.rs` writes for the gate-verdict line,
    and asserts the parent's `encoding="utf-8"` round-trips it exactly. Run without an
    explicit encoding on a non-UTF-8-locale Windows host, this is precisely the
    assertion that fails.
    """
    # Writes raw UTF-8 *bytes* to the stdout buffer directly, deliberately bypassing
    # Python's own text-mode stdout (whose encoding would itself depend on the console
    # codepage). This mirrors what `instance.rs`'s `println!`/`eprintln!` actually do on
    # Windows: Rust's stdio writer emits the string's UTF-8 bytes as-is, with no
    # console-codepage translation on the write side either -- so the only place a
    # locale can corrupt this text is the *read* side, which is exactly what this test
    # is isolating.
    script = tmp_path / "fake_epctl.py"
    script.write_text(
        "import sys\n"
        "sys.stdout.buffer.write(\n"
        "    '2 device(s) passed the \\u00a77.2 capability gate.\\n'.encode('utf-8')\n"
        ")\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    assert "passed the \u00a77.2 capability gate" in result.stdout
    # And the classifier this feeds must actually observe it as readable.
    sys.path.insert(0, str(REPO / "ci"))
    import check_icd_suppression as icd  # type: ignore

    verdict = icd.classify(result.stdout)
    assert verdict["state"] == icd.STATE_INEFFECTIVE
    assert verdict["devices_passing_gate"] == 2


def test_utf8_mis_decode_is_exactly_the_documented_failure_mode():
    """Names the bug precisely, so a future reader does not have to re-derive it.

    UTF-8 bytes for U+00A7 (`0xC2 0xA7`) decoded as `cp1252` produce `"Â§"`, two
    characters where the source had one -- which is neither the healthy gate-verdict
    line nor the ICD-absent failure text, and is exactly why real CI misclassified a
    readable report as `probe_report_unreadable`.
    """
    healthy_utf8_bytes = "1 device(s) passed the \u00a77.2 capability gate.".encode("utf-8")
    correctly_decoded = healthy_utf8_bytes.decode("utf-8")
    mis_decoded = healthy_utf8_bytes.decode("cp1252")

    # The regex this feeds (`ci/check_icd_suppression.GATE_VERDICT_RE`) requires "the"
    # immediately followed by a space and the section sign; cp1252 decodes the UTF-8
    # lead byte 0xC2 into its own character ("Â") rather than dropping it, so the
    # corruption inserts a character rather than deleting one -- "§7.2" alone is still a
    # substring of the mis-decode (0xA7 maps to the same code point in both cp1252 and
    # UTF-8's decoded form), but "the §7.2" (contiguous) is not, which is what actually
    # breaks the match.
    exact_phrase = "the \u00a77.2"
    assert exact_phrase in correctly_decoded
    assert exact_phrase not in mis_decoded, (
        "this test's premise (cp1252 corrupts the 'the §7.2' phrase's byte layout) no "
        "longer holds; re-derive which locale actually reproduces the real-CI failure "
        "before trusting the encoding fix to explain it."
    )

    sys.path.insert(0, str(REPO / "ci"))
    import check_icd_suppression as icd  # type: ignore

    assert icd.classify(correctly_decoded)["state"] == icd.STATE_INEFFECTIVE
    assert icd.classify(mis_decoded)["state"] == icd.STATE_UNREADABLE


def test_epctl_for_resolves_correctly_when_the_ep_library_path_has_spaces(tmp_path):
    """`test_criterion_4_5_witness.py::_epctl_for` under a `Program Files`-style path."""
    import test_criterion_4_5_witness as witness  # type: ignore

    spaced_dir = tmp_path / "Program Files (x86)" / "onnxruntime ep vulkan" / "release"
    spaced_dir.mkdir(parents=True)
    lib = spaced_dir / "onnxruntime_vulkan_ep.dll"
    lib.write_bytes(b"")
    epctl_name = "epctl.exe" if sys.platform == "win32" else "epctl"
    epctl = spaced_dir / epctl_name
    epctl.write_bytes(b"")

    resolved = witness._epctl_for(lib)
    assert resolved == epctl
    assert " " in str(resolved)
    assert resolved.is_file()


def test_validation_epctl_path_resolves_correctly_when_the_ep_library_path_has_spaces(
    tmp_path, monkeypatch
):
    """`test_validation.py::_epctl_path` under a `Program Files`-style path."""
    import test_validation as tv  # type: ignore

    spaced_dir = tmp_path / "Program Files (x86)" / "onnxruntime ep vulkan" / "release"
    spaced_dir.mkdir(parents=True)
    lib = spaced_dir / "onnxruntime_vulkan_ep.dll"
    lib.write_bytes(b"")
    epctl_name = "epctl.exe" if sys.platform == "win32" else "epctl"
    epctl = spaced_dir / epctl_name
    epctl.write_bytes(b"")

    monkeypatch.setenv("ONNXRUNTIME_VULKAN_EP_LIB", str(lib))
    resolved = tv._epctl_path()
    assert resolved == epctl
    assert " " in str(resolved)


def test_epctl_invocation_argv_list_is_never_shell_joined():
    """Every `subprocess.run` in the three caller files passes an argv list, never a
    shell string -- the property that makes the space-in-path tests above meaningful.
    A future edit to `shell=True` would silently reopen a quoting hazard these tests
    cannot otherwise see.
    """
    for path in _EPCTL_CALLER_FILES:
        for call in _subprocess_run_calls(path):
            for kw in call.keywords:
                if kw.arg == "shell":
                    assert not (
                        isinstance(kw.value, ast.Constant) and kw.value.value is True
                    ), f"{path.name}:{call.lineno} sets shell=True"
            if call.args:
                first = call.args[0]
                assert not isinstance(first, ast.Constant) or not isinstance(
                    first.value, str
                ), (
                    f"{path.name}:{call.lineno} passes a single command string to "
                    "subprocess.run's first argument; expected an argv list."
                )
