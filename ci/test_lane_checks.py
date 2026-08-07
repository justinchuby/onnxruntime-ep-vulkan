#!/usr/bin/env python3
"""Two-polarity tests for the lane checks in ``ci/``.

R9: *evidence scales only with falsifying instruments, not agreeing ones.*  A check that
has only ever been observed to pass is not known to be a check.  Every test here is a
pair: the artifact that must pass, and the artifact that must fail — with the **token**
asserted, not the exit code alone, because R13's whole finding is that two different
things spelled the same way are not a signal.

No GPU, no Vulkan and no ORT are required: the inputs are synthesised records and logs.
That is deliberate — this suite proves the *checks* work, and it must be runnable on a
lane where the thing they check is broken.

    python -m pytest ci/test_lane_checks.py -q
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

CI_DIR = Path(__file__).resolve().parent
REPO_ROOT = CI_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "tests" / "ops"))

try:
    import _verdict  # type: ignore
except ImportError as exc:  # pragma: no cover - collection-time outage
    # Deliberately NOT `pytest.importorskip`. A skipped test reports the same green as a
    # passing one, and the absence of the vocabulary these checks speak is an instrument
    # outage, not a clean lane (R12: a check whose subject cannot occur in its frame
    # reports UNOBSERVABLE, never 0). Failing collection is the honest state.
    raise RuntimeError(
        "tests/ops/_verdict.py (Trinity) is the single vocabulary the ci/ lane checks "
        "speak, and it could not be imported: "
        f"{exc!r}. This is ERROR(instrument), not a passing lane, and it must not be "
        "skipped into a green."
    ) from exc

EXIT_PASS = 0
EXIT_FAIL_CONDITION = 1
EXIT_USAGE = 2
EXIT_ERROR_INSTRUMENT = 4


def run_check(script: str, *args: str) -> subprocess.CompletedProcess:
    # Complete encoding pair, same reasoning as
    # negative_control_build_precondition.py's run(): pinning only the PARENT-side
    # decode (encoding="utf-8" below) is not enough on its own, because the CHILD
    # screen picks its OWN stdout/stderr encoding from locale.getpreferredencoding()
    # -- cp1252 on a default Windows shell -- unless PYTHONIOENCODING is present in
    # its environment. Several of these screens print literal Unicode (em-dashes,
    # arrows) in their frame/report lines; under a stock Windows env with neither
    # PYTHONIOENCODING nor PYTHONUTF8 set, the child's own print() raises
    # UnicodeEncodeError on those characters before ever emitting the assertion text
    # this suite's tests check for (e.g. check_tick_conversions.py's `->` arrow),
    # turning a real screen PASS/FAIL into an unrelated Windows-local instrument
    # crash. Forcing PYTHONIOENCODING=utf-8 into the child's env makes its own
    # encode side independent of the invoking shell; encoding="utf-8" below makes
    # this process's decode side independent of it too.
    child_env = dict(os.environ)
    child_env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, str(CI_DIR / script), *args],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=child_env,
    )


def record(**over) -> dict:
    base = {
        "verdict": _verdict.VERDICT_MATCH,
        "comparison": _verdict.COMPARISON_AGREE,
        "executed_by": {_verdict.EP_NAME: 1},
        "attribution_source": _verdict.ATTRIBUTION_SOURCE_PROFILE,
        "artifact": "gate_chain_fp32@ci-gate-v1 sha256:deadbeefdeadbeef",
        "device_index": "0",
        "detail": "max_abs_diff=0",
    }
    base.update(over)
    return base


def write(tmp_path: Path, name: str, doc) -> Path:
    p = tmp_path / name
    p.write_text(
        doc if isinstance(doc, str) else json.dumps(doc, indent=2), encoding="utf-8"
    )
    return p


# ---------------------------------------------------------------------------
# check_verdict.py
# ---------------------------------------------------------------------------


def test_attributed_match_passes(tmp_path):
    """The positive pole. Without it the negatives below prove only that it always fails."""
    p = write(tmp_path, "v.json", record())
    r = run_check("check_verdict.py", str(p))
    assert r.returncode == EXIT_PASS, r.stdout
    assert "VERDICT-CHECK: PASS" in r.stdout


@pytest.mark.parametrize(
    "over,expected_token",
    [
        # The 2026-07-30 specimen: agreed, and about a different world.
        ({"verdict": "UNATTRIBUTED", "executed_by": {"CPUExecutionProvider": 11}},
         "UNATTRIBUTED"),
        # A MATCH whose frame is missing is treated as UNATTRIBUTED, not trusted.
        ({"executed_by": {}}, "UNATTRIBUTED"),
        # A MATCH whose own count is zero cannot be a MATCH, whatever it says.
        ({"executed_by": {"CPUExecutionProvider": 3}}, "UNATTRIBUTED"),
        # Attribution from an instrument we own is not attribution (clause 1).
        ({"attribution_source": "counters"}, "UNATTRIBUTED"),
        ({"verdict": "DIVERGENT", "comparison": "DISAGREE"}, "DIVERGENT"),
        ({"verdict": "UNMEASURED", "executed_by": {}}, "UNMEASURED"),
        ({"verdict": "SPLIT-FRAME"}, "SPLIT-FRAME"),
    ],
)
def test_every_non_green_state_fails_as_itself(tmp_path, over, expected_token):
    """Each state fails, and each one prints **its own** token.

    Clause 4: ``UNATTRIBUTED`` must never be folded into ``DIVERGENT``.  A lane that
    prints one red for both has R13's defect, so the token is what is asserted here.
    """
    p = write(tmp_path, "v.json", record(**over))
    r = run_check("check_verdict.py", str(p))
    assert r.returncode == EXIT_FAIL_CONDITION, r.stdout
    assert f"VERDICT-CHECK: FAIL(condition={expected_token})" in r.stdout


def test_unattributed_is_not_reported_as_divergent(tmp_path):
    """The distinction the whole fourth state exists to preserve."""
    p = write(
        tmp_path,
        "v.json",
        record(verdict="UNATTRIBUTED", executed_by={"CPUExecutionProvider": 11}),
    )
    r = run_check("check_verdict.py", str(p))
    assert "UNATTRIBUTED" in r.stdout
    assert "FAIL(condition=DIVERGENT)" not in r.stdout
    assert "the model was not wrong, the subject was" in r.stdout.lower()


def test_absent_record_is_unmeasured_and_fails(tmp_path):
    """Absence of a check is a refusal, not a default green."""
    r = run_check("check_verdict.py", str(tmp_path / "does-not-exist.json"))
    assert r.returncode == EXIT_FAIL_CONDITION, r.stdout
    assert "FAIL(condition=UNMEASURED)" in r.stdout


def test_unparseable_record_is_an_instrument_error_not_a_detection(tmp_path):
    """R13: an instrument error never counts as a detection, so it never shares its code."""
    p = write(tmp_path, "v.json", "{not json")
    r = run_check("check_verdict.py", str(p))
    assert r.returncode == EXIT_ERROR_INSTRUMENT, r.stdout
    assert "ERROR(instrument=" in r.stdout
    assert "FAIL(condition=" not in r.stdout


def test_unknown_token_is_an_instrument_error(tmp_path):
    """Writer and reader disagreeing about the vocabulary says nothing about the EP."""
    p = write(tmp_path, "v.json", record(verdict="GREEN"))
    r = run_check("check_verdict.py", str(p))
    assert r.returncode == EXIT_ERROR_INSTRUMENT, r.stdout
    assert "verdict_token_unknown" in r.stdout


# ---------------------------------------------------------------------------
# check_fatal_log.py — the second witness, which must itself have two polarities
# ---------------------------------------------------------------------------


def test_clean_log_passes(tmp_path):
    p = write(tmp_path, "suite.log", "196 passed, 34 failed\nsome ordinary output\n")
    r = run_check("check_fatal_log.py", str(p))
    assert r.returncode == EXIT_PASS, r.stdout
    assert "FATAL-LOG-CHECK: PASS" in r.stdout


def test_falling_back_line_fails_the_lane_and_is_quoted(tmp_path):
    """The line ORT actually prints — not the one we spent three days believing it prints.

    2026-08-02, Trinity: this arm planted ``"Falling back to CPUExecutionProvider."`` and
    was green, because `_verdict.FATAL_LOG_MARKERS` was written from the same paraphrase.
    Fiction testing fiction. ORT emits a **list repr**, so the real announcement never
    matched and this check read Tank's artifact — which announces the fallback twice — as
    clean, while being cited as second witness for five incidents.

    The text below is copied verbatim from bench/results/ctx512_device_lost.txt.
    """
    p = write(
        tmp_path,
        "suite.log",
        "2026-08-02 ... EP Error: [ONNXRuntimeError] : 11 : EP_FAIL : Non-zero status "
        "code returned while running VulkanExecutionProvider_0 node.\n"
        "Falling back to ['CPUExecutionProvider'] and retrying.\n196 passed\n",
    )
    r = run_check("check_fatal_log.py", str(p))
    assert r.returncode == EXIT_FAIL_CONDITION, r.stdout
    assert "FAIL(condition=runtime_fallback_announced_by_ort)" in r.stdout
    # R13 second clause: quote the failure text, never the failure count.
    assert "Falling back to ['CPUExecutionProvider'] and retrying" in r.stdout


def test_the_lane_check_is_not_fooled_by_our_own_prose_about_it(tmp_path):
    """The defect, kept executable in the lane that missed it.

    Across three real logs the old markers produced twelve hits and every one was this
    repository's own sentence describing what we believed ORT prints, quoted back out of a
    captured suite log. A witness that matches our description of its subject, in logs that
    routinely contain our description of its subject, reports hits that mean nothing.
    """
    p = write(
        tmp_path,
        "suite.log",
        "E       ORT prints 'EP_FAIL ... Falling back to CPUExecutionProvider' "
        "during sess.run()\n196 passed\n",
    )
    r = run_check("check_fatal_log.py", str(p))
    assert r.returncode == EXIT_PASS, r.stdout


def test_an_uncaptured_log_is_unobservable_not_zero_hits(tmp_path):
    """R12: a counter whose event cannot occur in its frame reports UNOBSERVABLE."""
    r = run_check("check_fatal_log.py", str(tmp_path / "never-written.log"))
    assert r.returncode == EXIT_ERROR_INSTRUMENT, r.stdout
    assert "ERROR(instrument=log_not_captured)" in r.stdout


# ---------------------------------------------------------------------------
# gate_chain_fp32.py — the producer's own instrument states, without a GPU
# ---------------------------------------------------------------------------


def test_gate_writes_unmeasured_before_it_can_fail(tmp_path):
    """The initialisation ordering that makes a crash unable to look like a pass.

    Run with no EP library: the gate must bail with ``ERROR(instrument=...)`` **and**
    leave a readable ``UNMEASURED`` record behind, which ``check_verdict.py`` then fails
    the lane on.  Two states, in the right order, from one broken run.
    """
    out = tmp_path / "verdict.json"
    r = subprocess.run(
        [
            sys.executable,
            str(CI_DIR / "gate_chain_fp32.py"),
            "--verdict-out",
            str(out),
            "--workdir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        env={
            **{k: v for k, v in __import__("os").environ.items()},
            "ONNXRUNTIME_VULKAN_EP_LIB": str(tmp_path / "no-such.dll"),
            "ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE": "",
        },
    )
    assert r.returncode == EXIT_ERROR_INSTRUMENT, r.stdout + r.stderr
    # The SUBJECT of this test is the ordering — UNMEASURED reaches disk before anything
    # can fail — not which instrument error wins the race to be reported first. On a lane
    # that has ORT installed the gate reaches the library check and says
    # `ep_library_not_found`; on the host-free lane it stops one step earlier at
    # `python_dependency_missing`. Asserting only the first made this test pass on a
    # developer machine and fail on the runner, which is one arm wearing two coats.
    assert "GATE: ERROR(instrument=" in r.stdout, r.stdout
    assert (
        "ERROR(instrument=ep_library_not_found)" in r.stdout
        or "ERROR(instrument=python_dependency_missing)" in r.stdout
    ), r.stdout
    assert out.exists(), "the UNMEASURED record must be on disk before anything can fail"
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["verdict"] == _verdict.VERDICT_UNMEASURED
    assert doc["executed_by"] == {}

    follow_up = run_check("check_verdict.py", str(out))
    assert follow_up.returncode == EXIT_FAIL_CONDITION
    assert "FAIL(condition=UNMEASURED)" in follow_up.stdout


# ---------------------------------------------------------------------------
# check_vocabulary.py — the step that keeps ERROR(instrument) from becoming the
# lane's normal state.  Its three outcomes are exercised in a synthesised repository
# so this suite never mutates the real one.
# ---------------------------------------------------------------------------


def _fake_repo(tmp_path: Path, vocab_body: str | None) -> Path:
    """A minimal tree with ci/check_vocabulary.py and an optional tests/ops/_verdict.py."""
    (tmp_path / "ci").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tests" / "ops").mkdir(parents=True, exist_ok=True)
    shutil.copy2(CI_DIR / "check_vocabulary.py", tmp_path / "ci" / "check_vocabulary.py")
    if vocab_body is not None:
        (tmp_path / "tests" / "ops" / "_verdict.py").write_text(
            vocab_body, encoding="utf-8"
        )
    return tmp_path / "ci" / "check_vocabulary.py"


def _run_vocab(script: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True
    )


def test_vocabulary_present_and_importable_passes(tmp_path):
    """The positive pole, and the one that makes the other two mean something.

    Its message is load-bearing: after it passes, any later
    ``verdict_vocabulary_unavailable`` in the same job is a lane fault by elimination.
    """
    script = _fake_repo(tmp_path, 'VERDICT_MATCH = "MATCH"\nVERDICT_UNMEASURED = "UNMEASURED"\n')
    r = _run_vocab(script)
    assert r.returncode == EXIT_PASS, r.stdout + r.stderr
    assert "VOCAB: PASS" in r.stdout
    assert "is a LANE fault" in r.stdout


def test_absent_vocabulary_is_a_repository_state_with_its_own_token(tmp_path):
    """Case (a): the checkout does not carry the module.

    Red, because a lane that cannot emit a verdict cannot be green — but red under a
    token no CI change will clear, and the text says so.  This is the whole mechanism
    that stops ``ERROR(instrument)`` from becoming the weather.
    """
    script = _fake_repo(tmp_path, None)
    r = _run_vocab(script)
    assert r.returncode == EXIT_ERROR_INSTRUMENT, r.stdout + r.stderr
    assert "ERROR(instrument=verdict_vocabulary_absent_from_checkout)" in r.stdout
    assert "ERROR(instrument=verdict_vocabulary_broken)" not in r.stdout
    assert "REPOSITORY STATE, not a lane defect" in r.stdout


def test_present_but_unimportable_vocabulary_is_a_lane_defect_with_a_different_token(
    tmp_path,
):
    """Case (b): the file is right there and this interpreter cannot load it.

    Same exit code as case (a) — both are instrument outages and neither is a detection
    — and deliberately **not** the same token, because the token is what a maintainer
    greps and the two cases have different owners and different fixes.
    """
    script = _fake_repo(tmp_path, "def broken(:\n")
    r = _run_vocab(script)
    assert r.returncode == EXIT_ERROR_INSTRUMENT, r.stdout + r.stderr
    assert "ERROR(instrument=verdict_vocabulary_broken)" in r.stdout
    assert "ERROR(instrument=verdict_vocabulary_absent_from_checkout)" not in r.stdout
    assert "SyntaxError" in r.stdout


def test_the_two_vocabulary_outages_do_not_share_a_token(tmp_path):
    """Stated as its own assertion because it is the property, not a side effect."""
    absent = _run_vocab(_fake_repo(tmp_path / "a", None)).stdout
    broken = _run_vocab(_fake_repo(tmp_path / "b", "def broken(:\n")).stdout
    absent_token = [l for l in absent.splitlines() if l.startswith("VOCAB: ")][0]
    broken_token = [l for l in broken.splitlines() if l.startswith("VOCAB: ")][0]
    assert absent_token != broken_token


# ---------------------------------------------------------------------------
# gate_chain_fp32.py — the loader-independent negative control artifact
# ---------------------------------------------------------------------------


def test_decline_probe_is_a_distinct_artifact_built_from_an_unclaimed_op():
    """The negative control must not be the gate artifact wearing a different name.

    ``VK_DRIVER_FILES``/``VK_ICD_FILENAMES`` are silently ignored by the LunarG loader in
    elevated processes (PLATFORMS.md §7.4.1), and GitHub's Windows runners are elevated —
    so the ICD-removal negative control cannot be relied on there.  This artifact makes
    the EP execute nothing with the loader untouched.
    """
    sys.path.insert(0, str(CI_DIR))
    import gate_chain_fp32 as gate  # type: ignore

    assert set(gate.ARTIFACTS) == {"chain_fp32", "decline_probe"}
    assert gate.ARTIFACTS["decline_probe"][0] != gate.ARTIFACTS["chain_fp32"][0]

    onnx = pytest.importorskip("onnx")
    np = pytest.importorskip("numpy")

    decline_bytes, decline_feeds = gate.build_decline_probe_fp32(onnx, np)
    chain_bytes, _ = gate.build_gate_chain_fp32(onnx, np)
    assert decline_bytes != chain_bytes

    model = onnx.load_from_string(decline_bytes)
    op_types = [n.op_type for n in model.graph.node]
    assert op_types == [gate.DECLINE_OP]
    # If this op ever becomes claimable the probe stops being a negative control; the
    # lane step catches that by requiring UNATTRIBUTED, and this assertion names it.
    assert gate.DECLINE_OP not in {"Add", "Relu", "Mul", "Sub", "MatMulNBits"}
    # Non-singular by construction, so a CPU-vs-CPU comparison cannot agree on zeros.
    dets = np.linalg.det(decline_feeds["A"].astype(np.float64))
    assert np.all(np.abs(dets) > 1e-3)


# ---------------------------------------------------------------------------
# check_icd_suppression.py — the control that could not fire, and could not say so
#
# The Windows lane's ICD-removal negative control tested
# ``$probe -match 'passed the §7.2 capability gate'``.  The report line reads
# ``"{n} device(s) passed the §7.2 capability gate."`` and n is 0 when the
# suppression DID take, so the match succeeded on every input and the control
# short-circuited on every run.  A detector that fires on every input is a
# constant, not a detector.  These are the tests that would have caught it.
# ---------------------------------------------------------------------------

PROBE_SUPPRESSED = "0 device(s) passed the \u00a77.2 capability gate.\n  \u2192 No physical devices found."
#: The shape a REAL suppression takes, measured with the real binary: the loader has no
#: usable ICD, vkCreateInstance fails, and the capability-gate line is never printed.
PROBE_SUPPRESSED_NO_INSTANCE = (
    "=== Vulkan Loader Probe ===\nVulkan library loaded.\n"
    "  VK_DRIVER_FILES = C:\\nonexistent\\lvp_icd.json\n"
    "FAIL: vkCreateInstance returned ERROR_INCOMPATIBLE_DRIVER.\n"
)
PROBE_PRESENT = (
    "Vulkan 1.3.296 loader\n  llvmpipe (LLVM 15.0.7)\n"
    "1 device(s) passed the \u00a77.2 capability gate."
)


def test_icd_suppression_separates_a_suppressed_icd_from_a_present_one(tmp_path):
    """Two polarities, one string apart, and the tokens must differ."""
    ok = write(tmp_path, "p0.txt", PROBE_SUPPRESSED)
    bad = write(tmp_path, "p1.txt", PROBE_PRESENT)

    good = run_check("check_icd_suppression.py", str(ok))
    assert good.returncode == EXIT_PASS
    assert "ICD-SUPPRESSION: PASS" in good.stdout

    fired_not = run_check("check_icd_suppression.py", str(bad))
    assert fired_not.returncode == EXIT_ERROR_INSTRUMENT
    assert "ERROR(instrument=icd_suppression_ineffective)" in fired_not.stdout
    # Never a FAIL: this check has no condition of its own to detect. It decides
    # whether a *different* check can run, so every non-PASS is an instrument state.
    assert "FAIL(condition=" not in fired_not.stdout


def test_the_old_substring_test_would_have_passed_both_polarities(tmp_path):
    """The regression, asserted directly, so it cannot come back as a 'simplification'.

    Both reports contain the phrase the old check matched. That is the whole bug.
    """
    marker = "passed the \u00a77.2 capability gate"
    assert marker in PROBE_SUPPRESSED and marker in PROBE_PRESENT

    import check_icd_suppression as icd  # type: ignore

    assert icd.classify(PROBE_SUPPRESSED)["state"] == icd.STATE_SUPPRESSED
    assert icd.classify(PROBE_PRESENT)["state"] == icd.STATE_INEFFECTIVE


def test_icd_suppression_reworded_report_is_unreadable_not_suppressed(tmp_path):
    """R13: 'I could not read the report' is not 'the ICD is gone'."""
    p = write(tmp_path, "px.txt", "Vulkan looks fine on this machine.\n")
    r = run_check("check_icd_suppression.py", str(p))
    assert r.returncode == EXIT_ERROR_INSTRUMENT
    assert "ERROR(instrument=probe_report_unreadable)" in r.stdout


def test_icd_suppression_missing_report_is_an_instrument_state(tmp_path):
    r = run_check("check_icd_suppression.py", str(tmp_path / "nope.txt"))
    assert r.returncode == EXIT_ERROR_INSTRUMENT
    assert "ERROR(instrument=probe_report_unreadable)" in r.stdout


def test_icd_suppression_writes_a_record_whose_content_varies_with_its_input(tmp_path):
    """R10's falsifier for 'this probe is wired' — an artifact, not an annotation.

    The step this replaced emitted a ``::warning`` and nothing else. A warning leaves
    no artifact, so 'did the control fire on run 4132?' had no answer.
    """
    a_out = tmp_path / "a.json"
    b_out = tmp_path / "b.json"
    run_check(
        "check_icd_suppression.py",
        str(write(tmp_path, "p0.txt", PROBE_SUPPRESSED)),
        "--record-out",
        str(a_out),
    )
    run_check(
        "check_icd_suppression.py",
        str(write(tmp_path, "p1.txt", PROBE_PRESENT)),
        "--record-out",
        str(b_out),
    )
    a, b = json.loads(a_out.read_text()), json.loads(b_out.read_text())
    assert a["devices_passing_gate"] == 0 and b["devices_passing_gate"] == 1
    assert a["state"] != b["state"]


def test_icd_suppression_witnesses_that_disagree_are_an_instrument_state(tmp_path):
    """Two readers of one report, and picking the convenient one is how a lane lies."""
    p = write(tmp_path, "p0.txt", PROBE_SUPPRESSED)
    r = run_check("check_icd_suppression.py", str(p), "--exit-code", "0")
    assert r.returncode == EXIT_ERROR_INSTRUMENT
    assert "ERROR(instrument=probe_report_unreadable)" in r.stdout
    agree = run_check("check_icd_suppression.py", str(p), "--exit-code", "1")
    assert agree.returncode == EXIT_PASS


def test_a_real_suppression_never_reaches_the_capability_gate_line(tmp_path):
    """The bug this file was one draft away from repeating.

    Measured on real hardware, both polarities: when the ICD really is gone,
    ``vkCreateInstance`` returns ``ERROR_INCOMPATIBLE_DRIVER`` and the "N device(s)
    passed" line is **never printed**. Reading only that line would have classified every
    successful suppression as ``probe_report_unreadable`` — short-circuiting the negative
    control on every run, which is precisely the defect this check replaced. The lesson,
    and the reason this test exists: run the instrument in both polarities before
    believing a parser.
    """
    p = write(tmp_path, "p.txt", PROBE_SUPPRESSED_NO_INSTANCE)
    r = run_check("check_icd_suppression.py", str(p), "--exit-code", "3")
    assert r.returncode == EXIT_PASS
    assert "ICD-SUPPRESSION: PASS" in r.stdout


def test_icd_suppression_reports_the_three_states_apart(tmp_path):
    """Three inputs, three tokens. A check with one reachable outcome is a constant."""
    import check_icd_suppression as icd  # type: ignore

    states = {
        icd.classify(PROBE_SUPPRESSED)["state"],
        icd.classify(PROBE_SUPPRESSED_NO_INSTANCE)["state"],
        icd.classify(PROBE_PRESENT)["state"],
        icd.classify("Vulkan looks fine.")["state"],
    }
    assert states == {icd.STATE_SUPPRESSED, icd.STATE_INEFFECTIVE, icd.STATE_UNREADABLE}


#: A duplicate/conflicting-manifest report: two ICDs registered at once (e.g. a
#: leftover HKLM entry *and* an env-var search path both resolving), so the loader's
#: own line about the first ICD it tried is followed by a second, different verdict
#: from a second ICD. This is issue #1's "duplicate ICD manifest" scenario: the report
#: is readable and the loader's own count is the discriminator, not the number of
#: manifests behind it.
PROBE_DUPLICATE_MANIFEST = (
    "=== Vulkan Loader Probe ===\nVulkan library loaded.\n"
    "  VK_ICD_FILENAMES = C:\\a\\lvp_icd.json;C:\\b\\lvp_icd.json\n"
    "2 device(s) passed the \u00a77.2 capability gate."
)


def test_icd_suppression_with_duplicate_icd_manifests_is_still_ineffective(tmp_path):
    """Two manifests naming the same/different ICDs must not confuse the count parse.

    Whether one ICD is registered or several, the classifier reads the loader's own
    tally, never the number of manifest paths on the command line — the loader already
    did that reconciliation. A duplicate-manifest report where devices still pass the
    gate is exactly as ineffective as a single-manifest one; the classifier must say so,
    not go quiet on the unfamiliar shape.
    """
    import check_icd_suppression as icd  # type: ignore

    verdict = icd.classify(PROBE_DUPLICATE_MANIFEST)
    assert verdict["state"] == icd.STATE_INEFFECTIVE
    assert verdict["devices_passing_gate"] == 2


def test_icd_suppression_missing_manifest_path_is_readable_as_suppressed(tmp_path):
    """A manifest path that names a file which no longer exists is a real suppression.

    ``ci/negative_control_build_precondition.py``'s scenario: the ICD json was deleted
    (or, in the criterion-4 witness, pointed at ``does_not_exist/no_such_icd.json``) and
    the loader could not resolve any driver behind it. That is
    ``ERROR_INCOMPATIBLE_DRIVER`` from the loader's own mouth — a readable, positive
    reading, not an instrument outage.
    """
    import check_icd_suppression as icd  # type: ignore

    report = (
        "=== Vulkan Loader Probe ===\nVulkan library loaded.\n"
        "  VK_ICD_FILENAMES = C:\\does_not_exist\\no_such_icd.json\n"
        "FAIL: vkCreateInstance returned ERROR_INCOMPATIBLE_DRIVER: "
        "the loader found no usable ICD or the ICD library is not loadable.\n"
    )
    verdict = icd.classify(report)
    assert verdict["state"] == icd.STATE_SUPPRESSED
    assert verdict["devices_passing_gate"] == 0


# ---------------------------------------------------------------------------
# check_device_state.py — §10.0 obligation 8 at the lane level
#
# `gpu_steady_tail()` is a variance test over a suffix and cannot see a bias:
# a board held at its 210 MHz idle clock is perfectly steady about a wrong mean,
# and the wrong figure carried the BETTER RSD.  R9 amendment 5 demotes that check
# from gate to precondition; obligation 8 puts a device-state record in its place.
# These tests are what stop a future lane from publishing a duration without one.
# ---------------------------------------------------------------------------

NO_PRODUCERS = {"ONNXRUNTIME_EP_CI_DEVICE_STATE_PRODUCERS": "none"}

#: A host WITH telemetry, deterministically, on a runner that has none.  Without this the
#: three "an unrecorded duration reds the lane" tests were host-dependent: they asserted
#: FAIL(condition=STEADY_UNCERTIFIED) and passed on a desk with nvidia-smi while the CI
#: runner, having none, correctly produced ERROR(instrument=device_state_producer_absent).
#: Both are red, so the guard was never wrong — the TESTS were, and a test whose outcome
#: depends on which machine ran it has one arm, not two.
WITH_PRODUCER = {"ONNXRUNTIME_EP_CI_DEVICE_STATE_PRODUCERS": "simulate:nvidia"}


def _certified_record():
    """Niobe's record shape (bench/results/probe_gpustate.py :: summarise), not a new one."""
    return {
        "verdict": "SOLE_TENANT",
        "sm_mhz": {"n": 50, "min": 2010.0, "median": 2400.0, "max": 2490.0},
        "sm_max_mhz": 3105.0,
        "window": {"kind": "suffix", "index_from": 11, "index_to": 43, "n": 33},
    }


def run_ds(*args, env_extra=None):
    import os

    env = dict(os.environ)
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, str(CI_DIR / "check_device_state.py"), *args],
        capture_output=True,
        text=True,
        env=env,
    )


def test_device_state_passes_when_the_lane_publishes_no_duration(tmp_path):
    write(tmp_path, "verdict.json", record())
    r = run_ds("--scan", str(tmp_path))
    assert r.returncode == EXIT_PASS
    assert "DEVICE-STATE: PASS" in r.stdout


def test_device_state_fails_a_duration_with_no_record(tmp_path):
    """The polarity that matters: adding a millisecond to a lane artifact reds the lane."""
    write(tmp_path, "timing.json", {"mean_ms": 11.525, "rsd_pct": 0.8098})
    r = run_ds("--scan", str(tmp_path), env_extra=WITH_PRODUCER)
    assert r.returncode == EXIT_FAIL_CONDITION
    assert "FAIL(condition=STEADY_UNCERTIFIED)" in r.stdout
    assert "mean_ms=11.525" in r.stdout


def test_device_state_passes_a_duration_that_carries_a_certified_record(tmp_path):
    write(
        tmp_path,
        "timing.json",
        {"mean_ms": 11.525, "device_state": _certified_record()},
    )
    r = run_ds("--scan", str(tmp_path))
    assert r.returncode == EXIT_PASS
    assert "SOLE_TENANT" in r.stdout


def test_absent_telemetry_is_an_instrument_error_and_never_a_pass(tmp_path):
    """Obligation 8 amendment 2, which is the whole reason this lives in ci/.

    The cheapest way to satisfy obligation 8 as first worded is to measure where the
    requirement is vacuous. A CI runner with no GPU telemetry is that loophole at scale,
    so 'no producer' must be louder than 'no record', not quieter.
    """
    write(tmp_path, "timing.json", {"mean_ms": 11.525})
    r = run_ds("--scan", str(tmp_path), env_extra=NO_PRODUCERS)
    assert r.returncode == EXIT_ERROR_INSTRUMENT
    assert "ERROR(instrument=device_state_producer_absent)" in r.stdout
    assert "tenancy=SOLE_TENANT" not in r.stdout
    assert "DEVICE-STATE: PASS" not in r.stdout


def test_a_failed_probe_is_never_reported_as_sole_tenant(tmp_path):
    """Obligation 8 amendment 3. Absence of evidence and evidence of absence come out
    of one code path in a probe, so they must be named apart here."""
    write(
        tmp_path,
        "timing.json",
        {"mean_ms": 11.525, "device_state": {"error": "nvidia-smi exited 9"}},
    )
    r = run_ds("--scan", str(tmp_path))
    assert r.returncode == EXIT_ERROR_INSTRUMENT
    assert "ERROR(instrument=device_state_probe_failed)" in r.stdout


def test_an_incomplete_record_does_not_certify(tmp_path):
    """A record without its board maximum is a clock number without its ceiling — an
    index without its ordering (R11)."""
    import copy

    for drop in ("sm_max_mhz", "window", "verdict", "sm_mhz"):
        rec = copy.deepcopy(_certified_record())
        rec.pop(drop)
        d = tmp_path / drop
        d.mkdir()
        write(d, "timing.json", {"mean_ms": 11.525, "device_state": rec})
        r = run_ds("--scan", str(d), env_extra=WITH_PRODUCER)
        assert r.returncode == EXIT_FAIL_CONDITION, drop
        assert "STEADY_UNCERTIFIED" in r.stdout, drop


def test_a_duration_in_prose_is_published_too(tmp_path):
    """A JSON parser cannot see a figure written into the job summary. Second witness,
    different failure mode (R13 obligation 3)."""
    write(tmp_path, "verdict.json", record())
    summary = write(tmp_path, "summary.md", "Steady-state GPU busy 11.525 ms/inference\n")
    r = run_ds("--scan", str(tmp_path), "--summary", str(summary), env_extra=WITH_PRODUCER)
    assert r.returncode == EXIT_FAIL_CONDITION
    assert "11.525 ms" in r.stdout


def test_no_producer_override_can_ever_yield_a_pass(tmp_path):
    """The override's safety property, asserted rather than argued.

    ``ONNXRUNTIME_EP_CI_DEVICE_STATE_PRODUCERS`` exists so both polarities can be reached
    on any host. That makes it a switch on a guard, which is exactly the shape a waiver
    takes, so the constraint is checked here for every accepted value: an unrecorded
    duration is red under all of them. The override selects WHICH red, never green.
    """
    write(tmp_path, "timing.json", {"mean_ms": 11.525})
    for value in ("none", "simulate:nvidia", "simulate:intel", "simulate:cpu_renderer", ""):
        r = run_ds(
            "--scan",
            str(tmp_path),
            env_extra={"ONNXRUNTIME_EP_CI_DEVICE_STATE_PRODUCERS": value},
        )
        assert r.returncode != EXIT_PASS, value
        assert "DEVICE-STATE: PASS" not in r.stdout, value


def test_a_certified_record_still_passes_with_no_producer_on_the_host(tmp_path):
    """The complement, and the reason the override is safe in the other direction too.

    A PASS is decided by reading the RECORD, not by asking the host what it can measure.
    A record certified elsewhere and carried into a telemetry-less lane still certifies —
    otherwise obligation 8 would be unsatisfiable on exactly the machines that most need
    to quote a figure they did not take themselves.
    """
    write(tmp_path, "timing.json", {"mean_ms": 11.525, "device_state": _certified_record()})
    r = run_ds("--scan", str(tmp_path), env_extra=NO_PRODUCERS)
    assert r.returncode == EXIT_PASS
    assert "SOLE_TENANT" in r.stdout


def test_missing_lane_evidence_is_unobservable_not_clean(tmp_path):
    """R12: a check whose subject cannot occur in its frame reports UNOBSERVABLE, never 0.
    'The lane published no duration' and 'the lane produced nothing' are different."""
    r = run_ds("--scan", str(tmp_path / "does-not-exist"))
    assert r.returncode == EXIT_ERROR_INSTRUMENT
    assert "ERROR(instrument=lane_evidence_absent)" in r.stdout


def test_instrument_dumps_are_reported_rather_than_silently_skipped(tmp_path):
    """The exemption that would otherwise be the loophole.

    ORT's profile and the EP's counters snapshot carry microsecond fields the lane
    reads but does not author. They are excused from needing a companion — and they are
    still PRINTED, as STEADY_UNCERTIFIED carried-not-claimed, so the scope of the
    excuse is auditable from the lane's own output rather than from this file.
    """
    write(tmp_path, "counters-linux.json", {"session_staging_upload_us": 68})
    r = run_ds("--scan", str(tmp_path))
    assert r.returncode == EXIT_PASS
    assert "carried, not claimed" in r.stdout
    assert "session_staging_upload_us=68" in r.stdout


def test_the_instrument_dump_list_is_closed_and_has_a_reason_per_entry():
    """There is no runtime flag that adds to this list. An entry costs a code change and
    a test, which is the difference between an exemption and a waiver."""
    sys.path.insert(0, str(CI_DIR))
    import device_state as ds  # type: ignore

    assert "--exclude" not in (Path(CI_DIR / "check_device_state.py").read_text("utf-8"))
    for pattern, reason in ds.INSTRUMENT_DUMPS:
        assert reason and len(reason) > 40, pattern
    assert ds.is_instrument_dump(Path("bench/results/ci-lane/verdict-linux.json")) is None
    assert ds.is_instrument_dump(Path("bench/results/ci-lane/counters-linux.json"))


def test_obligation_8b_two_figures_compare_only_if_their_records_agree():
    """8b is not satisfied by both figures being STEADY — that is the whole finding."""
    sys.path.insert(0, str(CI_DIR))
    import copy

    import device_state as ds  # type: ignore

    a = _certified_record()
    assert ds.certifies_comparison(a, copy.deepcopy(a))["comparable"] is True
    # A 'before' that predates the companion requirement is not half of a pair.
    assert ds.certifies_comparison(None, a)["reason"] == "before_not_certified"
    contended = copy.deepcopy(a)
    contended["verdict"] = "FOREIGN_GPU_WORK"
    assert ds.certifies_comparison(a, contended)["reason"] == "tenancy_disagrees"
    idle = copy.deepcopy(a)
    idle["sm_mhz"] = {"n": 50, "min": 210.0, "median": 210.0, "max": 300.0}
    assert ds.certifies_comparison(a, idle)["reason"] == "clock_ranges_disjoint"


def test_lavapipe_ruling_is_written_down_not_discovered_later():
    """A CPU renderer has no device clock, so it can never certify a device-clock figure.
    That answer has to exist in prose, because 'no telemetry therefore no requirement' is
    most tempting exactly here and it is the waiver amendment 2 forbids."""
    sys.path.insert(0, str(CI_DIR))
    import device_state as ds  # type: ignore

    note = ds.lavapipe_note()
    assert "never certify" in note
    assert ds.PRODUCERS["cpu_renderer"]["status"] == ds.STATUS_NONE_STRUCTURAL
    # Not a tool registry with one row: the obligation is cross-platform by mandate.
    assert {"nvidia", "amd", "intel", "apple", "adreno", "mali"} <= set(ds.PRODUCERS)
    assert ds.PRODUCERS["intel"]["status"] != ds.STATUS_AVAILABLE


# ---------------------------------------------------------------------------
# The lane marker — telling "the lane published no duration" apart from
# "the lane died before it could publish anything".
#
# Observed on the 2026-08-01 main run: both device lanes failed at Clippy,
# never reached the step that creates bench/results/ci-lane, and the two
# always() checks downstream each added a second red for a subject that had
# never existed. Both reports were true. Both were noise on top of a failure
# they did not cause, and noise on a red lane is how a real finding gets
# scrolled past.
# ---------------------------------------------------------------------------


def test_no_evidence_and_no_marker_is_not_a_pass_but_does_not_add_a_red(tmp_path):
    r = run_ds(
        "--scan",
        str(tmp_path / "never-created"),
        "--lane-marker",
        str(tmp_path / "never-written"),
    )
    assert r.returncode == EXIT_PASS
    # It must still say, in its own output, that this is not a pass.
    assert "ERROR(instrument=lane_did_not_reach_evidence)" in r.stdout
    assert "NOT a pass" in r.stdout
    assert "DEVICE-STATE: PASS" not in r.stdout


def test_no_evidence_WITH_a_marker_is_still_an_instrument_error(tmp_path):
    """The direction that matters: the marker can only be written by the lane's own
    evidence producer, so marker-present-and-evidence-absent is a real instrument
    failure and keeps its exit 4."""
    marker = tmp_path / "marker"
    marker.write_text("", encoding="utf-8")
    r = run_ds("--scan", str(tmp_path / "never-created"), "--lane-marker", str(marker))
    assert r.returncode == EXIT_ERROR_INSTRUMENT
    assert "ERROR(instrument=lane_evidence_absent)" in r.stdout
    assert "did reach evidence production" in r.stdout


def test_the_marker_cannot_excuse_a_published_duration(tmp_path):
    """The abuse to ask about: can the marker be used to publish a figure quietly?

    No. The marker is only consulted when there is no evidence at all. Evidence that
    exists is scanned exactly as before, marker or no marker.
    """
    write(tmp_path, "timing.json", {"mean_ms": 11.525})
    r = run_ds(
        "--scan",
        str(tmp_path),
        "--lane-marker",
        str(tmp_path / "never-written"),
        env_extra=WITH_PRODUCER,
    )
    assert r.returncode == EXIT_FAIL_CONDITION
    assert "FAIL(condition=STEADY_UNCERTIFIED)" in r.stdout


def test_fatal_log_marker_has_the_same_two_polarities(tmp_path):
    absent = tmp_path / "no-such.log"
    r = run_check(
        "check_fatal_log.py",
        f"--lane-marker={tmp_path / 'never-written'}",
        str(absent),
    )
    assert r.returncode == EXIT_PASS
    assert "ERROR(instrument=lane_did_not_reach_evidence)" in r.stdout
    assert "NOT a pass" in r.stdout

    marker = tmp_path / "marker"
    marker.write_text("", encoding="utf-8")
    r = run_check("check_fatal_log.py", f"--lane-marker={marker}", str(absent))
    assert r.returncode == EXIT_ERROR_INSTRUMENT
    assert "ERROR(instrument=log_not_captured)" in r.stdout


def test_fatal_log_still_detects_with_a_marker_present(tmp_path):
    """A marker must never suppress a detection."""
    marker = tmp_path / "marker"
    marker.write_text("", encoding="utf-8")
    log = tmp_path / "run.log"
    log.write_text(
        "INFO: ok\n"
        "EP Error: [ONNXRuntimeError] : 11 : EP_FAIL : Non-zero status code returned.\n"
        "Falling back to ['CPUExecutionProvider'] and retrying.\n",
        encoding="utf-8",
    )
    r = run_check("check_fatal_log.py", f"--lane-marker={marker}", str(log))
    assert r.returncode == EXIT_FAIL_CONDITION
    assert "runtime_fallback_announced_by_ort" in r.stdout


# ---------------------------------------------------------------------------
# ci/lane_inventory.py — operational vs green, as data.
# ---------------------------------------------------------------------------


def test_the_inventory_is_well_formed():
    """Every entry must carry a `misses` column and, if it claims a failing arm, the
    observed failure TEXT rather than a status word."""
    import importlib

    inv = importlib.import_module("lane_inventory")
    assert inv.validate() == []


def test_an_undemonstrated_check_holds_its_lane_at_operational():
    """The classification rule, exercised in both directions on a synthetic lane."""
    import importlib

    inv = importlib.import_module("lane_inventory")
    for lane in inv.LANES:
        cls, why = inv.lane_classification(lane)
        assert cls in ("green", "green (with recorded gaps)", "operational")
        unproven = [c for c in inv.checks_for_lane(lane) if c.status == inv.UNDEMONSTRATED]
        if unproven:
            assert cls == "operational", lane
            for c in unproven:
                assert c.id in why


def test_a_green_status_without_an_observed_failure_is_refused(monkeypatch):
    """The inventory must not let a status word stand in for evidence — that is the
    whole shape of the defect it exists to record."""
    import importlib

    inv = importlib.import_module("lane_inventory")
    bogus = inv.Check(
        id="bogus.no_evidence",
        lane=inv.LANE_HOSTFREE,
        step="a step",
        watches="something",
        status=inv.DEMONSTRATED,
        misses=("something",),
    )
    monkeypatch.setattr(inv, "CHECKS", inv.CHECKS + (bogus,))
    problems = inv.validate()
    assert any("bogus.no_evidence" in p and "arm_broken" in p for p in problems)
    assert any("bogus.no_evidence" in p and "mutation" in p for p in problems)


def test_the_52x_blind_spot_is_recorded_with_a_named_substitute():
    """Task from 2026-08-01: if lavapipe cannot catch the timestampPeriod defect class,
    say so plainly and name what would. A documented blind spot beats a lane that
    implies coverage it does not have."""
    import importlib

    inv = importlib.import_module("lane_inventory")
    spot = next(b for b in inv.BLIND_SPOTS if b.id == "timestamp_period_52x")
    assert "1.0" in spot.why_ci_is_blind
    assert "identity" in spot.why_ci_is_blind.lower()
    assert spot.substitute is not None
    assert "trace.rs" in spot.substitute
    assert spot.substitute_status == inv.DEMONSTRATED


def test_a_blind_spot_with_no_substitute_says_so_rather_than_going_quiet():
    import importlib

    inv = importlib.import_module("lane_inventory")
    spot = next(b for b in inv.BLIND_SPOTS if b.id == "device_clock_state")
    assert spot.substitute is None
    assert spot.substitute_status == inv.IMPOSSIBLE_HERE
    assert "NOTHING IN THIS REPOSITORY" in inv.render()


def test_an_unclassified_lane_step_reds_the_checker(tmp_path):
    """R10 for the inventory itself: it must go red when its input changes in the way
    it exists to notice."""
    wf = tmp_path / "wf.yml"
    wf.write_text(
        "jobs:\n  x:\n    steps:\n"
        "      - name: Install Rust toolchain\n"
        "      - name: Something Nobody Classified\n",
        encoding="utf-8",
    )
    r = run_check("check_lane_inventory.py", "--workflow", str(wf))
    assert r.returncode == EXIT_FAIL_CONDITION
    assert "FAIL(condition=unclassified_lane_step)" in r.stdout
    assert "Something Nobody Classified" in r.stdout
    # ...and green when every step is either classified or provisioning.
    wf.write_text(
        "jobs:\n  x:\n    steps:\n"
        "      - name: Install Rust toolchain\n"
        "      - name: Clippy (all warnings as errors)\n",
        encoding="utf-8",
    )
    r = run_check("check_lane_inventory.py", "--workflow", str(wf))
    assert r.returncode == EXIT_PASS
    assert "PASS" in r.stdout


def test_the_real_workflow_has_no_unclassified_gate_steps():
    r = run_check(
        "check_lane_inventory.py",
        "--workflow",
        str(REPO_ROOT / ".github" / "workflows" / "ci.yml"),
    )
    assert r.returncode == EXIT_PASS, r.stdout

# ---------------------------------------------------------------------------------------
# ci/check_gh_auth.py — issue #21: semantic env parsing (block AND inline) and a
# zero-gh-reaching-subject frame that fails loudly by default instead of a silent PASS.
# ---------------------------------------------------------------------------------------


def test_inline_env_form_is_not_falsely_convicted(tmp_path):
    """The exact PR #17 review finding: `env: {GH_TOKEN: ...}` on one line is the
    remediation text, not the defect. Paired with the block form, which must satisfy
    the check identically."""
    inline = tmp_path / "inline.yml"
    inline.write_text(
        "name: p\non: push\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
        "    steps:\n      - name: s\n"
        "        env: {GH_TOKEN: ${{ github.token }}}\n"
        "        run: gh api repos/x/y\n",
        encoding="utf-8",
    )
    r = run_check("check_gh_auth.py", str(inline))
    assert r.returncode == EXIT_PASS, r.stdout

    block = tmp_path / "block.yml"
    block.write_text(
        "name: p\non: push\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
        "    steps:\n      - name: s\n        env:\n          GH_TOKEN: ${{ github.token }}\n"
        "        run: gh api repos/x/y\n",
        encoding="utf-8",
    )
    r = run_check("check_gh_auth.py", str(block))
    assert r.returncode == EXIT_PASS, r.stdout


def test_inline_env_at_job_and_workflow_scope_is_visible_to_its_steps(tmp_path):
    job_scope = tmp_path / "job.yml"
    job_scope.write_text(
        "name: p\non: push\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
        "    env: {GH_TOKEN: x}\n    steps:\n      - name: s\n        run: gh issue list\n",
        encoding="utf-8",
    )
    r = run_check("check_gh_auth.py", str(job_scope))
    assert r.returncode == EXIT_PASS, r.stdout

    workflow_scope = tmp_path / "workflow.yml"
    workflow_scope.write_text(
        "name: p\non: push\nenv: {GITHUB_TOKEN: x}\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
        "    steps:\n      - name: s\n        run: gh issue list\n",
        encoding="utf-8",
    )
    r = run_check("check_gh_auth.py", str(workflow_scope))
    assert r.returncode == EXIT_PASS, r.stdout


def test_a_wrongly_named_env_key_is_still_convicted(tmp_path):
    """Neither YAML shape should let a key that merely resembles GH_TOKEN pass."""
    wf = tmp_path / "wrong.yml"
    wf.write_text(
        "name: p\non: push\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
        "    steps:\n      - name: s\n        env: {TOKEN: x}\n        run: gh pr list\n",
        encoding="utf-8",
    )
    r = run_check("check_gh_auth.py", str(wf))
    assert r.returncode == EXIT_FAIL_CONDITION
    assert "FAIL(condition=missing_token_path)" in r.stdout


def test_zero_gh_reaching_subjects_fails_loudly_by_default(tmp_path):
    """Issue #21's second review finding: a frame with real steps but nothing that
    reaches `gh` must not read the same as a healthy screen. It is ERROR(instrument),
    unless the caller explicitly asks for --allow-empty-frame."""
    wf = tmp_path / "no-gh.yml"
    wf.write_text(
        "name: p\non: push\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
        "    steps:\n      - name: s\n        run: echo hi\n",
        encoding="utf-8",
    )
    r = run_check("check_gh_auth.py", str(wf))
    assert r.returncode == EXIT_ERROR_INSTRUMENT
    assert "ERROR(instrument=zero_gh_reaching_subjects)" in r.stdout

    r = run_check("check_gh_auth.py", "--allow-empty-frame", str(wf))
    assert r.returncode == EXIT_PASS
    assert "allow-empty-frame" in r.stdout


def test_a_directory_argument_is_expanded_and_an_empty_one_still_errors(tmp_path):
    """A caller may name a whole directory instead of individual files; every
    `*.yml`/`*.yaml` under it (recursively) is screened, and a directory with none in
    it at all is the wrong-scope/wrong-working-directory failure this screen exists to
    surface — never a silent pass over nothing."""
    nested = tmp_path / "workflows" / "sub"
    nested.mkdir(parents=True)
    (nested / "reachable.yml").write_text(
        "name: p\non: push\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
        "    env: {GH_TOKEN: x}\n    steps:\n      - name: s\n        run: gh api repos/x/y\n",
        encoding="utf-8",
    )
    r = run_check("check_gh_auth.py", str(tmp_path / "workflows"))
    assert r.returncode == EXIT_PASS, r.stdout
    assert "1 `gh`-reaching step" in r.stdout

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    r = run_check("check_gh_auth.py", str(empty_dir))
    assert r.returncode == EXIT_ERROR_INSTRUMENT
    assert "ERROR(instrument=empty_workflow_directory)" in r.stdout


def test_a_directory_scope_does_not_let_a_missing_token_hide_in_a_subdirectory(tmp_path):
    """The wiring concern issue #21 raises directly: screening a broader directory must
    still catch an offending step nested under it, not stop at the top level."""
    nested = tmp_path / "workflows" / "sub"
    nested.mkdir(parents=True)
    (nested / "offender.yml").write_text(
        "name: p\non: push\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
        "    steps:\n      - name: s\n        run: gh api repos/x/y\n",
        encoding="utf-8",
    )
    r = run_check("check_gh_auth.py", str(tmp_path / "workflows"))
    assert r.returncode == EXIT_FAIL_CONDITION
    assert "FAIL(condition=missing_token_path)" in r.stdout


def test_the_real_workflows_directory_passes_via_directory_expansion():
    """The wiring change itself: ci.yml now points the screen at the whole
    .github/workflows directory rather than two named files, so a workflow added
    later — or moved into a subdirectory — is screened without anyone extending a
    command line."""
    r = run_check("check_gh_auth.py", str(REPO_ROOT / ".github" / "workflows"))
    assert r.returncode == EXIT_PASS, r.stdout
    assert "gh`-reaching step" in r.stdout


def test_screening_only_the_gh_free_conformance_workflow_errors_not_passes():
    """conformance.yml genuinely has no `gh`-reaching step of its own. Screening it
    alone is exactly the wrong-scope/subdirectory shape issue #21 exists to catch, and
    it must never read as a clean PASS."""
    r = run_check(
        "check_gh_auth.py", str(REPO_ROOT / ".github" / "workflows" / "conformance.yml")
    )
    assert r.returncode == EXIT_ERROR_INSTRUMENT
    assert "ERROR(instrument=zero_gh_reaching_subjects)" in r.stdout


# ---------------------------------------------------------------------------------------
# ci/check_gh_auth.py — issue #25: YAML-structural env parsing. A purely line/indent
# heuristic cannot tell a real workflow/job/step `env:` apart from text that merely looks
# like one (inside a `run: |` block scalar) or a real `env` mapping that belongs to
# something else entirely (`services.<id>.env`, `with: env:`). These tests plant each
# shape and assert the checker gets it right — never a false PASS and never a false FAIL.
# ---------------------------------------------------------------------------------------


def test_env_like_text_inside_a_run_block_scalar_is_not_a_real_declaration(tmp_path):
    """A `run: |` block scalar is literal shell script text handed to the runner, not
    YAML structure. Lines inside it that happen to read `env:` / `GH_TOKEN:` must never
    be mistaken for an actual token declaration."""
    wf = tmp_path / "block-scalar-trap.yml"
    wf.write_text(
        "name: p\non: push\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
        "    steps:\n      - name: s\n        run: |\n"
        "          echo \"env:\"\n"
        "          echo \"  GH_TOKEN: fake\"\n"
        "          gh api repos/x/y\n",
        encoding="utf-8",
    )
    r = run_check("check_gh_auth.py", str(wf))
    assert r.returncode == EXIT_FAIL_CONDITION, r.stdout
    assert "FAIL(condition=missing_token_path)" in r.stdout


def test_services_env_is_not_the_jobs_env(tmp_path):
    """`services.<id>.env` is a real YAML mapping literally named `env`, but it belongs
    to a service container, not the job. It must not satisfy the job-scope token check
    for the job's own steps."""
    wf = tmp_path / "services-env-trap.yml"
    wf.write_text(
        "name: p\non: push\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
        "    services:\n      redis:\n        image: redis\n"
        "        env:\n          GH_TOKEN: fake\n"
        "    steps:\n      - name: s\n        run: gh api repos/x/y\n",
        encoding="utf-8",
    )
    r = run_check("check_gh_auth.py", str(wf))
    assert r.returncode == EXIT_FAIL_CONDITION, r.stdout
    assert "FAIL(condition=missing_token_path)" in r.stdout


def test_with_env_is_not_the_steps_env(tmp_path):
    """`with: env:` is a real YAML mapping literally named `env`, but it is an action
    input, not the step's own execution environment."""
    wf = tmp_path / "with-env-trap.yml"
    wf.write_text(
        "name: p\non: push\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
        "    steps:\n      - name: s\n        uses: some/action@v1\n"
        "        with:\n          env:\n            GH_TOKEN: fake\n"
        "      - name: real\n        run: gh api repos/x/y\n",
        encoding="utf-8",
    )
    r = run_check("check_gh_auth.py", str(wf))
    assert r.returncode == EXIT_FAIL_CONDITION, r.stdout
    assert "FAIL(condition=missing_token_path)" in r.stdout


def test_a_quoted_block_key_satisfies_the_check(tmp_path):
    """`"GH_TOKEN":` (quoted) is the same key as `GH_TOKEN:` (unquoted); quoting a
    block-form mapping key must not blind the parser to it."""
    wf = tmp_path / "quoted-key.yml"
    wf.write_text(
        "name: p\non: push\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
        "    steps:\n      - name: s\n        env:\n"
        '          "GH_TOKEN": ${{ github.token }}\n'
        "        run: gh api repos/x/y\n",
        encoding="utf-8",
    )
    r = run_check("check_gh_auth.py", str(wf))
    assert r.returncode == EXIT_PASS, r.stdout


@pytest.mark.parametrize(
    "env_text",
    [
        "        env:\n          GH_TOKEN: one\n          GH_TOKEN: two\n",
        "        env: {GH_TOKEN: one, GH_TOKEN: two}\n",
    ],
    ids=["block-form", "flow-form"],
)
def test_a_duplicate_key_in_one_env_mapping_is_an_unsupported_construct(tmp_path, env_text):
    """A key declared twice in the SAME `env:` mapping (block or flow form) is
    ambiguous/unsupported input, not something to silently resolve by picking a
    winner -- it must raise a loud instrument error."""
    wf = tmp_path / "dup-key.yml"
    wf.write_text(
        "name: p\non: push\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
        "    steps:\n      - name: s\n" + env_text + "        run: gh api repos/x/y\n",
        encoding="utf-8",
    )
    r = run_check("check_gh_auth.py", str(wf))
    assert r.returncode == EXIT_ERROR_INSTRUMENT, r.stdout
    assert "unsupported_yaml_construct" in r.stdout
    assert "declared twice" in r.stdout


def test_a_trailing_comment_after_inline_env_does_not_hide_the_declaration(tmp_path):
    """A trailing `# comment` after `env: {GH_TOKEN: ...}` on the same physical line
    must not stop the token from being recognised -- the old anchored-regex approach
    would silently fail to match here."""
    wf = tmp_path / "trailing-comment.yml"
    wf.write_text(
        "name: p\non: push\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
        "    steps:\n      - name: s\n"
        "        env: {GH_TOKEN: ${{ github.token }}}  # ci: token lives here\n"
        "        run: gh api repos/x/y\n",
        encoding="utf-8",
    )
    r = run_check("check_gh_auth.py", str(wf))
    assert r.returncode == EXIT_PASS, r.stdout


def test_a_full_line_comment_mentioning_env_is_never_treated_as_structure(tmp_path):
    """A full-line comment that happens to read `env:`/`GH_TOKEN:` is not YAML
    structure at all and must never satisfy the token check."""
    wf = tmp_path / "comment-only.yml"
    wf.write_text(
        "name: p\non: push\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
        "    steps:\n      - name: s\n"
        "        # env:\n        #   GH_TOKEN: fake\n"
        "        run: gh api repos/x/y\n",
        encoding="utf-8",
    )
    r = run_check("check_gh_auth.py", str(wf))
    assert r.returncode == EXIT_FAIL_CONDITION, r.stdout
    assert "FAIL(condition=missing_token_path)" in r.stdout


def test_a_multiline_flow_mapping_is_read_as_one_declaration(tmp_path):
    """A flow mapping `env: {...}` split across several physical lines is still valid
    YAML and must be recognised as a single declaration, not missed the way a
    single-physical-line-only regex would miss it."""
    wf = tmp_path / "multiline-flow.yml"
    wf.write_text(
        "name: p\non: push\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
        "    steps:\n      - name: s\n        env: {\n"
        "          GH_TOKEN: ${{ github.token }},\n"
        "          OTHER: value\n        }\n"
        "        run: gh api repos/x/y\n",
        encoding="utf-8",
    )
    r = run_check("check_gh_auth.py", str(wf))
    assert r.returncode == EXIT_PASS, r.stdout


def test_a_with_steps_list_does_not_orphan_the_real_step(tmp_path):
    """`with: steps:` is a legal action input that happens to be named `steps` and
    holds its own nested list. Before R1, ANY frame whose key was literally "steps"
    reset step-minting, so this nested list's own dash orphaned the REAL step: its
    `run: gh api ...` line (after the `with:` block) was silently dropped from body
    capture and never became a checked subject at all -- a false PASS by omission. A
    second, unrelated, correctly-tokened `gh` step keeps the file's total gh-reaching
    count above zero, isolating the omission from issue #21's separate zero-subject
    rule."""
    wf = tmp_path / "with-steps-trap.yml"
    wf.write_text(
        "name: p\non: push\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
        "    steps:\n      - name: s\n        uses: some/action@v1\n"
        "        with:\n          steps:\n            - name: fake\n"
        "              run: echo hi\n"
        "        run: gh api repos/x/y\n"
        "      - name: real\n        env:\n          GH_TOKEN: x\n"
        "        run: gh api repos/a/b\n",
        encoding="utf-8",
    )
    r = run_check("check_gh_auth.py", str(wf))
    assert r.returncode == EXIT_FAIL_CONDITION, r.stdout
    assert "FAIL(condition=missing_token_path)" in r.stdout
    assert "PASS — 1 `gh`-reaching step" not in r.stdout


@pytest.mark.parametrize(
    "with_value",
    ["&anchor", "!!map"],
    ids=["anchor", "tag"],
)
def test_an_unrecognised_scalar_value_followed_by_deeper_content_is_an_error(tmp_path, with_value):
    """When a key's value is a bare scalar this parser does not structurally
    understand (`&anchor`, `!!tag`, `*alias`), and the NEXT line is MORE indented
    (its real nested payload), that content must not be silently attributed to the
    grandparent frame -- it must raise, because the grandparent's own `env:` scope is
    NOT actually where that nested `env:` lives."""
    wf = tmp_path / "unrecognised-scalar-trap.yml"
    wf.write_text(
        "name: p\non: push\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
        "    steps:\n      - name: s\n        uses: some/action@v1\n"
        f"        with: {with_value}\n          env:\n            GH_TOKEN: x\n"
        "        run: gh api repos/x/y\n",
        encoding="utf-8",
    )
    r = run_check("check_gh_auth.py", str(wf))
    assert r.returncode == EXIT_ERROR_INSTRUMENT, r.stdout
    assert "unsupported_yaml_construct" in r.stdout
    assert "anchor, tag, or alias" in r.stdout


def test_a_multi_document_separator_is_an_error(tmp_path):
    """A `---` YAML document-separator line. This screen reads one YAML document per
    file; silently continuing past it would misattribute the second document's
    steps/env to whatever frame was still open at the end of the first."""
    wf = tmp_path / "multi-doc.yml"
    wf.write_text(
        "name: p\non: push\n---\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
        "    steps:\n      - name: s\n        run: gh api repos/x/y\n",
        encoding="utf-8",
    )
    r = run_check("check_gh_auth.py", str(wf))
    assert r.returncode == EXIT_ERROR_INSTRUMENT, r.stdout
    assert "unsupported_yaml_construct" in r.stdout
    assert "document-separator" in r.stdout


def test_a_nested_flow_collection_as_an_env_value_is_an_error(tmp_path):
    """`env: {FOO: {GH_TOKEN: x}}` -- GH_TOKEN is nested INSIDE FOO's own value, not a
    sibling key of the mapping itself. This must never be silently read either way;
    it is ambiguous input that raises."""
    wf = tmp_path / "nested-flow-value.yml"
    wf.write_text(
        "name: p\non: push\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
        "    steps:\n      - name: s\n"
        "        env: {FOO: {GH_TOKEN: x}}\n        run: gh api repos/x/y\n",
        encoding="utf-8",
    )
    r = run_check("check_gh_auth.py", str(wf))
    assert r.returncode == EXIT_ERROR_INSTRUMENT, r.stdout
    assert "unsupported_yaml_construct" in r.stdout
    assert "nested flow collection" in r.stdout


def test_a_quoted_value_containing_a_literal_brace_is_an_error(tmp_path):
    """`env: {FOO: '{"GH_TOKEN": "1"}'}` -- a quoted value that itself contains
    literal braces. The quote-tracking scan can in fact tell this is opaque text, but
    this screen deliberately refuses rather than trusting its own reading of an
    ambiguous-looking shape."""
    wf = tmp_path / "quoted-brace-value.yml"
    wf.write_text(
        "name: p\non: push\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
        "    steps:\n      - name: s\n"
        '        env: {FOO: \'{"GH_TOKEN": "1"}\'}\n        run: gh api repos/x/y\n',
        encoding="utf-8",
    )
    r = run_check("check_gh_auth.py", str(wf))
    assert r.returncode == EXIT_ERROR_INSTRUMENT, r.stdout
    assert "unsupported_yaml_construct" in r.stdout
    assert "quoted value" in r.stdout


def test_a_gh_expression_inside_flow_env_is_not_mistaken_for_nested_value_braces(tmp_path):
    """Regression guard for the nested-flow-collection check above: a
    `${{ github.token }}` GitHub Actions expression is internally-balanced `{`/`}`
    text, not YAML flow-collection nesting, and must not itself be convicted -- this
    is, after all, the exact remediation text this screen's own FAIL message
    recommends."""
    wf = tmp_path / "gh-expression-in-flow-env.yml"
    wf.write_text(
        "name: p\non: push\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
        "    steps:\n      - name: s\n"
        "        env: {GH_TOKEN: ${{ github.token }}, OTHER: ${{ secrets.X }}}\n"
        "        run: gh api repos/x/y\n",
        encoding="utf-8",
    )
    r = run_check("check_gh_auth.py", str(wf))
    assert r.returncode == EXIT_PASS, r.stdout


def test_a_duplicate_sibling_env_key_at_job_scope_is_an_error(tmp_path):
    """Two SIBLING `env:` block keys at the same job scope. Before N1 these were
    silently unioned (`set.update`), so a job with two `env:` blocks -- one of which
    happens to declare GH_TOKEN -- satisfied the check even though a duplicate
    mapping key is itself invalid/ambiguous YAML that no single reading should
    resolve silently."""
    wf = tmp_path / "dup-sibling-env.yml"
    wf.write_text(
        "name: p\non: push\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
        "    env:\n      GH_TOKEN: x\n    env:\n      GITHUB_TOKEN: y\n"
        "    steps:\n      - name: s\n        run: gh api repos/x/y\n",
        encoding="utf-8",
    )
    r = run_check("check_gh_auth.py", str(wf))
    assert r.returncode == EXIT_ERROR_INSTRUMENT, r.stdout
    assert "unsupported_yaml_construct" in r.stdout
    assert "second `env:` key" in r.stdout


def test_a_dash_inline_env_block_does_not_swallow_a_true_sibling_field(tmp_path):
    """`- env:` opens a block-form env inline with the dash. Before N2, the
    redispatch used a synthetic `dash_indent + 1` column, which undershoots `env:`'s
    TRUE column (`dash_indent + 2`, one space after the dash) -- so a later TRUE
    SIBLING field at that real column (here a bogus step-level `GH_TOKEN:` key that
    is NOT actually inside `env:`) was wrongly swallowed as one of env's own
    children, letting a coincidentally-named sibling satisfy the token check it
    should not."""
    wf = tmp_path / "dash-inline-env-sibling.yml"
    wf.write_text(
        "name: p\non: push\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
        "    steps:\n      - env:\n          NOT_A_TOKEN: x\n"
        "        GH_TOKEN: this-is-not-really-an-env-declaration\n"
        "        run: gh api repos/x/y\n",
        encoding="utf-8",
    )
    r = run_check("check_gh_auth.py", str(wf))
    assert r.returncode == EXIT_FAIL_CONDITION, r.stdout
    assert "FAIL(condition=missing_token_path)" in r.stdout


def test_a_compact_form_steps_list_mints_its_step(tmp_path):
    """`steps:` and its FIRST item's dash sit at the exact same column -- the
    equally-valid "compact"/zero-indent block-sequence form YAML permits for any
    list-valued key, not only `steps:`. Before this fix, the generic `indent <=
    frame.indent` pop popped the `steps:` frame itself before this dash was ever
    recognised as ITS child (both frames sit at the same indent), so the step --
    and its untokened `gh api` call -- was silently dropped from every subject this
    screen ever sees: a false PASS by omission, reproducible in a single job with no
    second job required (Morpheus's re-review of PR #27)."""
    wf = tmp_path / "compact-single-job.yml"
    wf.write_text(
        "name: p\non: push\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
        "    steps:\n    - name: s\n      run: gh api repos/x/y\n",
        encoding="utf-8",
    )
    r = run_check("check_gh_auth.py", str(wf))
    assert r.returncode == EXIT_FAIL_CONDITION, r.stdout
    assert "FAIL(condition=missing_token_path)" in r.stdout
    assert "PASS" not in r.stdout


def test_a_compact_form_job_before_an_indented_form_job_does_not_silently_omit_its_step(tmp_path):
    """Morpheus's exact reproducer: job `a` uses the compact (equal-indent) `steps:`
    form and its one step is untokened; job `b` uses the ordinary indented form and
    its step IS tokened. Before this fix, job `a`'s step was silently dropped (see
    the single-job arm above), so the file's only COUNTED subject was job `b`'s
    already-tokened step -- a false PASS. This also exercises the buf/flush ordering
    fix alongside the frame-pop fix: job `b`'s own dash line (indent 6) sits deeper
    than job `a`'s step's `current_indent` (4); appending that dash line to the OLD
    step's buffer before flushing (the order this screen used previously) would
    splice job `b`'s marker onto job `a`'s body instead of starting a fresh step."""
    wf = tmp_path / "compact-then-indented.yml"
    wf.write_text(
        "name: p\non: push\njobs:\n"
        "  a:\n    runs-on: ubuntu-latest\n"
        "    steps:\n    - name: untokened\n      run: gh api repos/x/y\n"
        "  b:\n    runs-on: ubuntu-latest\n"
        "    steps:\n      - name: tokened\n        env: {GH_TOKEN: x}\n"
        "        run: gh api repos/a/b\n",
        encoding="utf-8",
    )
    r = run_check("check_gh_auth.py", str(wf))
    assert r.returncode == EXIT_FAIL_CONDITION, r.stdout
    assert "FAIL(condition=missing_token_path)" in r.stdout
    assert "untokened" in r.stdout
    assert "PASS — 1 `gh`-reaching step" not in r.stdout


def test_ci_yml_production_invocation_is_pinned_to_the_directory_form():
    """The wiring concern issue #21 (and #25 after it) exists to close: a real
    regression here is exactly reverting `.github/workflows` back to two named files,
    which is how a newly added or relocated workflow would silently stop being
    screened. This asserts on ci.yml's own text, not just synthetic behaviour."""
    ci_text = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    invocations = re.findall(r"run: python ci/check_gh_auth\.py ([^\n]+)", ci_text)
    assert len(invocations) == 1, invocations
    assert invocations[0].strip() == ".github/workflows", invocations


def test_lane_checks_job_has_no_depth_limited_fetch_before_the_ledger_census_issue_28():
    """ISSUE #28. `git fetch --depth=1 origin main`, run earlier in the SAME job as
    `ci/check_ledger_census.py`, grafted the shared repository — even though the job's
    own checkout above already declares `fetch-depth: 0` — because a depth-limited fetch
    marks its own boundary regardless of what history the repository already holds. A
    real `pull/N/merge` preview's base parent IS `origin/main`'s tip, so that graft point
    was always one of the checked-out merge's own two parents, and
    `git rev-parse --is-shallow-repository` went true for the whole checkout.
    `ci/check_ledger_census.py`'s history-completeness guard then (correctly) refused to
    answer, and `ci/open_reds.json` reported `ledger_census`/`ledger_census_negative_control`
    `FAIL(condition=unaccounted_red)` on every PR.

    The fix drops the unnecessary `--depth` from that one lookup (the job already has
    full history and gains nothing from shortening this one already-known ref). This
    asserts on ci.yml's own text within the `lane-checks` job specifically, so a
    `--depth`-limited `git fetch ... origin` reintroduced anywhere ahead of the census
    step in this job fails on the change that reintroduces it.
    """
    ci_text = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    lane_start = ci_text.index("\n  lane-checks:\n")
    next_job = re.search(r"\n  [a-zA-Z][\w-]*:\n", ci_text[lane_start + 1 :])
    lane_end = lane_start + 1 + next_job.start() if next_job else len(ci_text)
    lane_body = ci_text[lane_start:lane_end]

    assert "check_ledger_census.py" in lane_body, "this job must still run the census"
    fetch_lines = re.findall(r"git fetch[^\n]*origin[^\n]*", lane_body)
    assert fetch_lines, "expected at least the main-tip lookup fetch in this job"
    for line in fetch_lines:
        assert "--depth" not in line, (
            f"a depth-limited fetch in the lane-checks job grafts the WHOLE repository "
            f"(shared .git/shallow), not just the fetched ref, poisoning the census's "
            f"history-completeness guard for the rest of the job: {line!r}"
        )


def test_history_reading_op_test_jobs_check_out_full_history_issue_24():
    """ISSUE #24. `tests/ops/test_proof_ledger.py`'s landing-safety simulation lands
    HEAD's tree onto `origin/main` and screens the result, and
    `rust/tools/probe_ledger_loss.py`'s default run reads a named historical commit.
    Both need history. On the default depth-1 shallow clone they produce
    `ERROR(instrument=truncated_history)` -- an outage from an incomplete checkout, not
    a defect in the register -- and that is exactly how `main` was red at 5113a0a on run
    31164215291: both op-test jobs failed on the same single test, for a checkout reason.

    The repair is to supply the history, not to widen the test's unobservable branch: a
    screen that skips itself on every push is a screen nobody is running. This asserts on
    ci.yml's own text so a checkout that quietly loses `fetch-depth: 0` fails on the
    change that loses it, rather than on the next push to main.
    """
    ci_text = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    job_bounds = [
        (m.group(1), m.start())
        for m in re.finditer(r"\n  ([a-zA-Z][\w-]*):\n", ci_text)
    ]
    bodies = {}
    for i, (name, start) in enumerate(job_bounds):
        end = job_bounds[i + 1][1] if i + 1 < len(job_bounds) else len(ci_text)
        bodies[name] = ci_text[start:end]

    needs_history = [
        name
        for name, body in bodies.items()
        if "test_proof_ledger" in body
        or "probe_ledger_loss" in body
        or "check_ledger_census.py" in body
        or "pytest tests/ops" in body
        or "pytest tests\\ops" in body
    ]
    assert needs_history, (
        "no job reads history any more — either the screens moved or this guard is "
        "matching on stale names, and a guard matching nothing is not a green"
    )
    for name in needs_history:
        first_checkout = bodies[name].index("uses: actions/checkout@")
        window = bodies[name][first_checkout : first_checkout + 200]
        assert "fetch-depth: 0" in window, (
            f"job {name!r} runs a history-reading screen but its checkout does not "
            f"declare fetch-depth: 0. On a depth-1 clone that screen reports "
            f"ERROR(instrument=truncated_history) and the lane goes red for a checkout "
            f"reason (issue #24, run 31164215291)."
        )


# ---------------------------------------------------------------------------------------
# ISSUE #24 -- a real import-closure contract, not a string search.
#
# Morpheus's round-N review of PR #32 (APPROVE WITH REQUIRED FIXES) rejected the first
# version of this guard on two counts: (1) the fix it guarded hand-copied a SECOND
# package list into the lane-checks job instead of using tests/requirements.txt, the one
# authoritative floor list already used by the Windows build-test job; (2) the guard
# itself only asserted that two literal substrings ("onnx_ir", "onnxruntime") appeared
# somewhere in the step's text, which a stray comment mentioning either word would also
# satisfy, and which says nothing about whether tests/ops/conftest.py's actual import
# list is still fully covered.
#
# What follows derives the CLAIM (which third-party names conftest.py needs) from the
# source file itself via `ast`, derives what tests/requirements.txt PROMISES via its own
# text, and asserts the two agree -- plus that the lane-checks job installs from that one
# file. Three buckets, none silent: every module-scope import root is stdlib, a local
# sibling file, or a third-party name that must appear (after PEP 503 normalization) in
# tests/requirements.txt.
# ---------------------------------------------------------------------------------------

CONFTEST_PATH = REPO_ROOT / "tests" / "ops" / "conftest.py"
REQUIREMENTS_PATH = REPO_ROOT / "tests" / "requirements.txt"

#: Import root name -> PyPI distribution name, for cases where PEP 503 normalization of
#: the import name does not equal the normalized distribution name on its own. `onnx_ir`
#: imports as `onnx_ir` but its distribution is `onnx-ir` (`pip show onnx_ir` in the repo
#: .venv reports `Name: onnx-ir`) -- normalization alone already bridges this particular
#: one (both sides fold to `onnx-ir`), but it is listed explicitly so the mapping is a
#: real, present, fail-loud table rather than something relying on a coincidence between
#: two independent naming schemes. A genuinely divergent case (`import cv2` -> the
#: `opencv-python` distribution, `import yaml` -> `PyYAML`) would have to be added here
#: explicitly, or this check would wrongly report the import as uncovered.
IMPORT_TO_DISTRIBUTION: dict[str, str] = {
    "onnx_ir": "onnx-ir",
}


def _extract_module_scope_import_roots(py_file: Path) -> set[str]:
    """Root package names of every *unconditional, module-scope* import in py_file.

    Only direct children of the module body (`tree.body`) are considered: an import
    guarded by `if TYPE_CHECKING:`, wrapped in `try/except ImportError:`, or nested in a
    function/class body is conditional or deferred, and does not reproduce issue #24's
    failure mode -- collection only fails because these particular names are imported
    UNCONDITIONALLY at collection time, for every test under the directory, regardless of
    which one is selected. `ast`, not text search, so a docstring or comment containing
    the word "import" cannot be mistaken for one.
    """
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    roots: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue  # `from . import x` -- always local, never third-party.
            if node.module:
                roots.add(node.module.split(".")[0])
    return roots


def _local_module_roots(directory: Path) -> set[str]:
    """Names importable as a sibling of a file in `directory` purely by file presence:
    `<name>.py` or `<name>/__init__.py`. This is how pytest's rootdir-relative import
    makes `tests/ops/_verdict.py` reachable as `import _verdict` from
    `tests/ops/conftest.py` -- a real local module, not a third-party package that
    happens to be missing from tests/requirements.txt."""
    names: set[str] = set()
    for entry in directory.iterdir():
        if entry.is_file() and entry.suffix == ".py":
            names.add(entry.stem)
        elif entry.is_dir() and (entry / "__init__.py").exists():
            names.add(entry.name)
    return names


def _classify_import_roots(
    roots: set[str], local_dir: Path
) -> tuple[set[str], set[str], set[str]]:
    """Partition import roots into (stdlib, local, third_party). Every root lands in
    exactly one bucket -- there is no silent fourth category. `local` is checked first:
    a name that is both a local sibling file and a stdlib name is a real local import in
    this directory (shadowing), so local wins over stdlib on purpose."""
    local_names = _local_module_roots(local_dir)
    stdlib_names = set(sys.stdlib_module_names)
    stdlib, local, third_party = set(), set(), set()
    for root in roots:
        if root in local_names:
            local.add(root)
        elif root in stdlib_names:
            stdlib.add(root)
        else:
            third_party.add(root)
    return stdlib, local, third_party


def _normalize_distribution_name(name: str) -> str:
    """PEP 503 normalization: case-fold, collapse runs of `-_.` into a single `-`."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirements_txt_distribution_names(requirements_path: Path) -> set[str]:
    """Normalized distribution names declared in a requirements.txt file, ignoring
    comments (whole-line or trailing) and version specifiers."""
    names: set[str] = set()
    for line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        m = re.match(r"^([A-Za-z0-9_.-]+)", line)
        if m:
            names.add(_normalize_distribution_name(m.group(1)))
    return names


def _uncovered_third_party_imports(
    conftest_path: Path, requirements_path: Path
) -> list[str]:
    """Third-party import roots at conftest_path's module scope that are NOT covered by
    a normalized distribution name in requirements_path (after IMPORT_TO_DISTRIBUTION).
    Empty list means the import closure is fully covered."""
    roots = _extract_module_scope_import_roots(conftest_path)
    _stdlib, _local, third_party = _classify_import_roots(roots, conftest_path.parent)
    declared = _requirements_txt_distribution_names(requirements_path)
    uncovered = []
    for root in sorted(third_party):
        dist = IMPORT_TO_DISTRIBUTION.get(root, root)
        if _normalize_distribution_name(dist) not in declared:
            uncovered.append(root)
    return uncovered


def test_conftest_third_party_imports_are_covered_by_tests_requirements_txt_issue_24():
    """ISSUE #24, the real contract: every third-party name `tests/ops/conftest.py`
    imports unconditionally at module scope must have a matching, normalized entry in
    `tests/requirements.txt` -- the file the lane-checks job (after this fix) actually
    installs from. This is derived from both files' own content via `ast` and PEP 503
    normalization, not asserted as a hardcoded pair of literal strings, so it keeps
    holding if conftest.py's import list changes in either direction."""
    uncovered = _uncovered_third_party_imports(CONFTEST_PATH, REQUIREMENTS_PATH)
    assert not uncovered, (
        f"tests/ops/conftest.py imports {uncovered!r} unconditionally at module scope, "
        f"but tests/requirements.txt declares no matching distribution (after PEP 503 "
        f"normalization and the IMPORT_TO_DISTRIBUTION table) for at least one of them. "
        f"Any job that installs only from tests/requirements.txt will hit "
        f"ModuleNotFoundError at collection time for every test under tests/ops/ -- "
        f"issue #24's exact failure mode."
    )


def test_conftest_import_closure_check_fails_on_a_planted_uncovered_import(tmp_path):
    """Positive control for the CHECK's own sensitivity (not for conftest.py): a fixture
    conftest.py importing a package genuinely absent from a fixture requirements.txt must
    be reported as uncovered. Without this, the test above could pass for the wrong
    reason -- e.g. a `_uncovered_third_party_imports` that always returns an empty list."""
    fixture_dir = tmp_path / "ops"
    fixture_dir.mkdir()
    (fixture_dir / "conftest.py").write_text(
        "import definitely_not_a_stdlib_or_declared_package\n", encoding="utf-8"
    )
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("numpy>=1.24\npytest>=8.0\n", encoding="utf-8")

    uncovered = _uncovered_third_party_imports(fixture_dir / "conftest.py", requirements)
    assert uncovered == ["definitely_not_a_stdlib_or_declared_package"], uncovered


def test_conftest_import_closure_check_fails_when_a_real_requirement_is_removed():
    """Negative-polarity fixture built from conftest.py's REAL import list (the actual
    regression subject, not a synthetic stand-in), paired with a requirements.txt that
    has had `onnx_ir` removed -- simulating exactly the historical bug: a job whose
    install step forgot one of the packages tests/ops/conftest.py needs. This must be
    caught even though conftest.py itself is untouched."""
    real_roots = _extract_module_scope_import_roots(CONFTEST_PATH)
    _stdlib, _local, real_third_party = _classify_import_roots(
        real_roots, CONFTEST_PATH.parent
    )
    assert "onnx_ir" in real_third_party  # sanity: still true of the real file today

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        fixture_dir = tmp_path / "ops"
        fixture_dir.mkdir()
        # `_verdict.py` must be present alongside the copied conftest.py, or `_verdict`
        # would misclassify as third-party (it is a real local sibling, not the subject
        # of this test) and pollute the result with an unrelated finding.
        shutil.copy(CONFTEST_PATH, fixture_dir / "conftest.py")
        shutil.copy(
            REPO_ROOT / "tests" / "ops" / "_verdict.py", fixture_dir / "_verdict.py"
        )

        full_requirements = REQUIREMENTS_PATH.read_text(encoding="utf-8")
        trimmed = "\n".join(
            line for line in full_requirements.splitlines() if "onnx_ir" not in line
        )
        requirements = tmp_path / "requirements.txt"
        requirements.write_text(trimmed, encoding="utf-8")

        uncovered = _uncovered_third_party_imports(
            fixture_dir / "conftest.py", requirements
        )
    assert uncovered == ["onnx_ir"], uncovered


def test_conftest_actually_collects_with_tests_requirements_txt_dependencies_issue_24():
    """Supplements, but does not replace, the static AST-based coverage check above: a
    subprocess collection of `harness_census_drift`'s own registered command, using the
    CURRENT interpreter (which has tests/requirements.txt's packages installed in this
    repo's .venv). This is real evidence that, given those dependencies, collection
    genuinely succeeds. It does not by itself prove the CI job installs them -- that is
    `test_lane_checks_job_installs_from_tests_requirements_txt_issue_24` below -- and it
    says nothing about interpreters other than this one."""
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            str(REPO_ROOT / "tests" / "ops" / "test_harness_census.py"),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert "ModuleNotFoundError" not in r.stdout, r.stdout
    assert re.search(r"\b\d+ tests? collected\b", r.stdout), r.stdout


def _lane_checks_job_body(ci_text: str) -> str:
    """Extract the `lane-checks:` job's body from ci.yml's raw text -- the same job-
    boundary convention `...issue_28` uses, factored out so both tests share it."""
    lane_start = ci_text.index("\n  lane-checks:\n")
    next_job = re.search(r"\n  [a-zA-Z][\w-]*:\n", ci_text[lane_start + 1 :])
    lane_end = lane_start + 1 + next_job.start() if next_job else len(ci_text)
    return ci_text[lane_start:lane_end]


def _step_body(job_body: str, step_name: str) -> str:
    """Every line belonging to a named step (from just after its `- name:` line up to,
    but not including, the next step at the same indentation, or the job's end), with
    whole-line `#` comments stripped -- so a check for what a step actually RUNS cannot
    be satisfied by a comment merely mentioning the right words."""
    m = re.search(rf"(?P<indent>[ \t]*)- name:\s*{re.escape(step_name)}\s*\n", job_body)
    assert m, f"expected a step named {step_name!r} in this job"
    indent = len(m.group("indent"))
    rest = job_body[m.end() :]
    next_step = re.search(rf"^[ \t]{{{indent}}}-\s", rest, re.MULTILINE)
    body = rest[: next_step.start()] if next_step else rest
    kept = [
        line
        for line in body.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    return "\n".join(kept)


def test_lane_checks_job_installs_from_tests_requirements_txt_issue_24():
    """ISSUE #24 (Morpheus review of PR #32, required fix 1). The lane-checks job's
    prior fix hand-copied its own package list (`onnxruntime==... onnx>=... numpy
    onnx_ir pytest`) -- a SECOND, independently-maintained answer to "what does
    tests/ops/ need to import", diverging from tests/requirements.txt (the one
    authoritative floor list, already used by the Windows build-test job's own "Install
    Python test dependencies" step) the moment either one is edited without the other.
    This asserts the job installs from that one file instead of a hand-copied list."""
    lane_body = _lane_checks_job_body(
        (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    )
    step_body = _step_body(lane_body, "Install test dependencies")
    assert re.search(r"-r\s+tests/requirements\.txt", step_body), (
        f"the lane-checks job's 'Install test dependencies' step must install from "
        f"tests/requirements.txt (`pip install -r tests/requirements.txt`), the one "
        f"authoritative floor list, not a hand-copied second (or third) package list "
        f"(comments mentioning the filename do not count -- this reads the step's "
        f"non-comment lines only):\n{step_body}"
    )


def test_lane_checks_job_install_step_check_fails_on_the_prior_hand_copied_list():
    """Negative-polarity fixture: this job's OWN prior form (issue #24's first pass,
    approved-with-required-fixes by Morpheus) -- a hand-copied inline package list, not
    sourced from tests/requirements.txt -- must be reported as NOT satisfying the
    requirement above, proving this test's own sensitivity rather than a check that
    would pass against anything."""
    fixture_ci_yml = (
        "name: p\non: push\njobs:\n"
        "  lane-checks:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - name: Install test dependencies\n"
        "        run: |\n"
        '          python -m pip install --upgrade "onnxruntime==1.28.0" '
        '"onnx>=1.22.0" numpy onnx_ir pytest\n'
        "  next-job:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps: []\n"
    )
    lane_body = _lane_checks_job_body(fixture_ci_yml)
    step_body = _step_body(lane_body, "Install test dependencies")
    assert not re.search(r"-r\s+tests/requirements\.txt", step_body), step_body


# ---------------------------------------------------------------------------------------
# ci/check_tautological_assertions.py
#
# This screen reports 0 detections over the real tree, so every scrap of evidence that it
# is a screen at all comes from planted violations. R9: a check observed only to pass is
# not known to be a check. Each test below is a pair - the form that must be caught, and
# the neighbouring form that must NOT be, because three of this screen's first four
# detections were false positives and an unscoped screen asserts things rather than
# merely missing them.
# ---------------------------------------------------------------------------------------

TAUT = "check_tautological_assertions.py"


def _taut(tmp_path):
    return run_check(TAUT, "--root", str(tmp_path))


def test_identical_operands_under_equality_are_caught_in_rust(tmp_path):
    (tmp_path / "a.rs").write_text(
        "#[test]\nfn t() {\n    let x = compute();\n    assert_eq!(x, x);\n}\n",
        encoding="utf-8",
    )
    r = _taut(tmp_path)
    assert r.returncode == EXIT_FAIL_CONDITION, r.stdout + r.stderr
    assert "IDENTICAL_OPERANDS" in r.stderr
    assert "a.rs:4" in r.stderr


def test_identical_operands_under_equality_are_caught_in_python(tmp_path):
    (tmp_path / "a.py").write_text(
        "def test_t():\n    x = compute()\n    assert x.mean() == x.mean()\n",
        encoding="utf-8",
    )
    r = _taut(tmp_path)
    assert r.returncode == EXIT_FAIL_CONDITION, r.stdout + r.stderr
    assert "IDENTICAL_OPERANDS" in r.stderr


def test_both_literals_are_caught_at_either_polarity(tmp_path):
    """Also pins the precedence: identical text wins the label when both rules apply.

    `assert_eq!(0.0, 0.0)` is both identical and both-literal; it reports as
    IDENTICAL_OPERANDS. `assert_eq!(1u32, 1)` is the equality case that only BOTH_LITERAL
    catches - different text, both constant, always passes reading nothing.
    """
    (tmp_path / "a.rs").write_text(
        "fn t() {\n"
        "    assert_eq!(0.0, 0.0);\n"
        "    assert_ne!(1, 2);\n"
        "    assert_eq!(1u32, 1);\n"
        "}\n",
        encoding="utf-8",
    )
    r = _taut(tmp_path)
    assert r.returncode == EXIT_FAIL_CONDITION, r.stdout + r.stderr
    assert "tautological_assertions=3" in r.stderr, r.stderr
    assert r.stderr.count("BOTH_LITERAL") == 2, r.stderr
    assert r.stderr.count("IDENTICAL_OPERANDS") == 1, r.stderr


def test_the_nan_idiom_is_not_reported(tmp_path):
    """`x != x` is the NaN test, and it was this screen's first false positive.

    The hazard is *passing without reading the subject*, not sameness. Identical operands
    under inequality either always fail - which is safe, a permanently red assertion gets
    fixed on its first run - or are a deliberate NaN probe.
    """
    (tmp_path / "a.py").write_text(
        "def test_t():\n    assert empty.median != empty.median\n",
        encoding="utf-8",
    )
    (tmp_path / "b.rs").write_text("fn t() {\n    assert_ne!(v.median, v.median);\n}\n", "utf-8")
    r = _taut(tmp_path)
    assert r.returncode == EXIT_PASS, r.stdout + r.stderr


def test_distinct_string_subscripts_are_not_identical(tmp_path):
    """The scanner's own defect 2, pinned so blanking cannot come back.

    Neutralising string literals by blanking them turns `frame["a"] == frame["b"]` into a
    term compared to itself. Three of the first four detections were this shape and all
    three were correct code.
    """
    (tmp_path / "a.py").write_text(
        'def test_t():\n    assert frame["dispatched_devices"] == frame["capable_devices"]\n',
        encoding="utf-8",
    )
    r = _taut(tmp_path)
    assert r.returncode == EXIT_PASS, r.stdout + r.stderr


def test_an_assertion_inside_a_string_literal_is_invisible(tmp_path):
    """The other direction, which is why blanking existed in the first place.

    rust/tests/layering.rs carries `assert_eq!(1, 1)` inside a string as fixture text for
    the layering lint. Neutralising must hide that without merging distinct operands - the
    two requirements pull opposite ways and both have a test.
    """
    (tmp_path / "a.rs").write_text(
        'fn t() {\n    let planted = "fn innocent() { assert_eq!(1, 1); }";\n'
        "    assert_eq!(scan(planted).len(), 1);\n}\n",
        encoding="utf-8",
    )
    r = _taut(tmp_path)
    assert r.returncode == EXIT_PASS, r.stdout + r.stderr


def test_a_commented_out_tautology_is_invisible(tmp_path):
    (tmp_path / "a.rs").write_text(
        "fn t() {\n    // assert_eq!(x, x);\n    assert_eq!(got, want);\n}\n",
        encoding="utf-8",
    )
    r = _taut(tmp_path)
    assert r.returncode == EXIT_PASS, r.stdout + r.stderr


def test_a_language_that_yields_no_assertions_is_an_outage_not_a_pass(tmp_path):
    """The screen's own defect 1, which shipped for one revision.

    A leading-whitespace bug meant 89 Python files contributed zero assertions and the run
    still said PASS, because Rust's count carried the total. A total another language paid
    for is not coverage, so coverage is asserted per language.
    """
    (tmp_path / "a.rs").write_text("fn t() {\n    assert_eq!(got, want);\n}\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    r = _taut(tmp_path)
    assert r.returncode == EXIT_ERROR_INSTRUMENT, r.stdout + r.stderr
    assert "language_scanned_nothing" in r.stderr


def test_a_tree_with_no_sources_is_an_outage(tmp_path):
    (tmp_path / "notes.md").write_text("nothing here\n", encoding="utf-8")
    r = _taut(tmp_path)
    assert r.returncode == EXIT_ERROR_INSTRUMENT, r.stdout + r.stderr
    assert "no_sources_found" in r.stderr


def test_the_real_repository_passes_and_says_what_it_does_not_cover(tmp_path):
    """The PASS arm, and the scope sentence is asserted as part of it.

    A detection with a false description is worse than no description (D-T85). The two
    assertion defects that actually occurred in this repository - two different
    expressions that always evaluate equal, and a test whose assertions are all one
    polarity - are both outside this screen. If that disclaimer is ever trimmed out of the
    output, this test fails.
    """
    r = run_check(TAUT)
    assert r.returncode == EXIT_PASS, r.stdout + r.stderr
    assert "ASSERTIONS: PASS(" in r.stdout
    assert "does NOT detect" in r.stdout
    assert "all one polarity" in r.stdout

# ---------------------------------------------------------------------------
# check_tick_conversions.py — the static screen for the tick->duration defect class.
#
# This is the only check here that decides its question from SOURCE TEXT rather than
# from a run, because the defect class it attacks is invisible to both of the other
# families: unit tests prove the conversion correct where it is called and say nothing
# about whether every path calls it, and no device lane can see a skipped conversion
# because timestampPeriod is 1.0 on every device CI can reach, which makes the
# conversion the identity there.
# ---------------------------------------------------------------------------

TICK_SANCTIONED = """\
pub struct GpuTimestampCalibration {
    pub timestamp_period_ns: f32,
    pub valid_bits: u32,
}
impl GpuTimestampCalibration {
    pub fn ticks_to_ns(&self, begin_ticks: u64, end_ticks: u64) -> Option<f64> {
        let span = mask_ticks(end_ticks, self.valid_bits);
        Some(span as f64 * f64::from(self.timestamp_period_ns))
    }
}
fn mask_ticks(raw_ticks: u64, valid_bits: u32) -> u64 {
    raw_ticks & ((1u64 << valid_bits) - 1)
}
fn consume(cal: &GpuTimestampCalibration, pool: &Pool) {
    let results = unsafe { pool.read_results() };
    let cal2 = GpuTimestampCalibration { timestamp_period_ns: 1.0, valid_bits: 64 };
    let _ns = cal.ticks_to_ns(0, 1);
}
"""

TICK_ALLOWLIST = {
    "sanctioned_sites": [
        {
            "file": "rust/src/trace.rs",
            "function": "ticks_to_ns",
            "contains": "Some(span as f64 * f64::from(self.timestamp_period_ns))",
            "reason": "the conversion itself",
            "owner": "test",
        },
        {
            "file": "rust/src/trace.rs",
            "function": "mask_ticks",
            "contains": "raw_ticks & ((1u64 << valid_bits) - 1)",
            "reason": "arithmetic is on the mask, not on the tick",
            "owner": "test",
        },
    ]
}


def _tick_tree(tmp_path: Path, body: str, allowlist=None) -> tuple[Path, Path]:
    """A minimal source tree the screen can be pointed at, plus its allowlist."""
    src = tmp_path / "rust" / "src"
    src.mkdir(parents=True, exist_ok=True)
    (src / "trace.rs").write_text(body, encoding="utf-8")
    al = tmp_path / "allowlist.json"
    al.write_text(
        json.dumps(TICK_ALLOWLIST if allowlist is None else allowlist), encoding="utf-8"
    )
    return tmp_path, al


def _run_tick(root: Path, al: Path) -> subprocess.CompletedProcess:
    return run_check("check_tick_conversions.py", "--root", str(root), "--allowlist", str(al))


def test_tick_screen_passes_when_every_scale_is_sanctioned(tmp_path):
    root, al = _tick_tree(tmp_path, TICK_SANCTIONED)
    r = _run_tick(root, al)
    assert r.returncode == EXIT_PASS, r.stdout
    assert "TICK-SCREEN: PASS" in r.stdout
    # A pass that does not say what it did not check is a pass nobody can size.
    assert "What it does not claim" in r.stdout


@pytest.mark.parametrize(
    "bypass",
    [
        "fn bad(begin_ticks: u64, end_ticks: u64) -> u64 { let d = end_ticks - begin_ticks; d }",
        "fn bad(end_ticks: u64) -> f64 { let elapsed_ns = end_ticks as f64; elapsed_ns }",
        "fn bad(end_ticks: u64) -> u64 { let launder = end_ticks; launder }",
    ],
)
def test_tick_screen_reds_on_an_injected_bypass_and_quotes_the_line(tmp_path, bypass):
    """R10 for the screen itself: an artifact whose content varies with its input.

    Three shapes of the same defect — a raw delta, a raw cast, and the rename that
    defeats a naive name-based screen.
    """
    root, al = _tick_tree(tmp_path, TICK_SANCTIONED + "\n" + bypass + "\n")
    r = _run_tick(root, al)
    assert r.returncode == EXIT_FAIL_CONDITION, r.stdout
    assert "FAIL(condition=tick_conversion_bypassed)" in r.stdout
    # R13: quote the failure text, never the failure count.
    assert "fn bad" in r.stdout


def test_tick_screen_reds_on_a_second_reader_of_raw_ticks(tmp_path):
    """The arm that addresses 'does every path use it' rather than 'is it right'."""
    root, al = _tick_tree(
        tmp_path,
        TICK_SANCTIONED + "\nfn other(p: &Pool) { let _r = unsafe { p.read_results() }; }\n",
    )
    r = _run_tick(root, al)
    assert r.returncode == EXIT_FAIL_CONDITION, r.stdout
    assert "FAIL(condition=raw_tick_producer_not_unique)" in r.stdout
    assert "is called from 2 sites" in r.stdout


def test_tick_screen_reds_when_an_allowlist_entry_has_lost_its_site(tmp_path):
    """An exemption that no longer matches anything is a blanket, not an exemption."""
    rotted = json.loads(json.dumps(TICK_ALLOWLIST))
    rotted["sanctioned_sites"].append(
        {
            "file": "rust/src/trace.rs",
            "function": "gone",
            "contains": "a line that no longer exists",
            "reason": "stale",
            "owner": "test",
        }
    )
    root, al = _tick_tree(tmp_path, TICK_SANCTIONED, allowlist=rotted)
    r = _run_tick(root, al)
    assert r.returncode == EXIT_FAIL_CONDITION, r.stdout
    assert "FAIL(condition=allowlist_entry_without_a_site)" in r.stdout
    assert "lost its site" in r.stdout


def test_tick_screen_holds_test_modules_out_of_frame_and_says_so(tmp_path):
    """R12: the frame's exclusions are reported, not silently applied.

    A #[cfg(test)] module may build ticks freely; it ships nothing. But a screen that
    quietly drops lines is a screen whose green covers an unknown amount of code.
    """
    body = TICK_SANCTIONED + (
        "\n#[cfg(test)]\nmod tests {\n"
        "    fn t() { let d = end_ticks - begin_ticks; }\n"
        "}\n"
    )
    root, al = _tick_tree(tmp_path, body)
    r = _run_tick(root, al)
    assert r.returncode == EXIT_PASS, r.stdout
    assert "held out as #[cfg(test)]" in r.stdout
    assert "UNOBSERVABLE by frame, not zero findings" in r.stdout


def test_tick_screen_missing_allowlist_is_an_instrument_error_not_a_detection(tmp_path):
    root, _ = _tick_tree(tmp_path, TICK_SANCTIONED)
    r = _run_tick(root, tmp_path / "no-such-file.json")
    assert r.returncode == EXIT_ERROR_INSTRUMENT, r.stdout
    assert "ERROR(instrument=allowlist_unreadable)" in r.stdout
    assert "NOT a detection" in r.stdout


def test_tick_screen_empty_frame_is_unobservable_not_a_pass(tmp_path):
    """A screen pointed at the wrong tree finds nothing, and finding nothing is not clean."""
    root, al = _tick_tree(tmp_path, "fn unrelated() -> u32 { 1 + 1 }\n")
    r = _run_tick(root, al)
    assert r.returncode == EXIT_ERROR_INSTRUMENT, r.stdout
    assert "ERROR(instrument=no_tick_sites_found)" in r.stdout


def test_the_real_source_tree_has_no_unsanctioned_tick_arithmetic():
    """The screen against the tree it ships with. This is the one that can regress."""
    r = run_check("check_tick_conversions.py")
    assert r.returncode == EXIT_PASS, r.stdout
    assert "TICK-SCREEN: PASS" in r.stdout


# ---------------------------------------------------------------------------
# --union-with: the composed-workflow blind spot.
#
# On 2026-08-02 the same shape occurred five times in one day: two correct branches
# composing into a broken whole, with no command either author could have run that would
# have shown it. My instance was a lane step that was complete on Switch's branch and a
# lane inventory that was complete on mine; the union was unclassified. These tests build
# a real two-branch git repository and check that the union view goes red where the
# branch-only view is green.
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=str(repo), capture_output=True, text=True, check=True
    )


def _two_branch_repo(tmp_path: Path, mine: str, theirs: str) -> Path:
    """A repo whose `main` carries `theirs` and whose working tree carries `mine`."""
    repo = tmp_path / "repo"
    (repo / ".github" / "workflows").mkdir(parents=True)
    _git(repo.parent, "init", "-q", "repo")
    _git(repo, "config", "user.email", "link@squad.test")
    _git(repo, "config", "user.name", "link")
    wf = repo / ".github" / "workflows" / "ci.yml"
    wf.write_text("jobs:\n  x:\n    steps:\n" + theirs, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "theirs")
    _git(repo, "branch", "-M", "main")
    _git(repo, "checkout", "-q", "-b", "squad/link")
    wf.write_text("jobs:\n  x:\n    steps:\n" + mine, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "mine")
    return repo


CLASSIFIED_STEP = "      - name: Clippy (all warnings as errors)\n"
OTHER_CLASSIFIED_STEP = "      - name: Portability lint (cargo test --test portability)\n"
UNKNOWN_STEP = "      - name: A Step Nobody On My Branch Can See\n"


def test_union_reds_on_a_step_that_only_exists_on_the_other_branch(tmp_path):
    """The exact defect: my branch is complete, theirs is complete, the union is not."""
    repo = _two_branch_repo(
        tmp_path, mine=CLASSIFIED_STEP, theirs=OTHER_CLASSIFIED_STEP + UNKNOWN_STEP
    )
    wf = repo / ".github" / "workflows" / "ci.yml"

    # Branch-only view — green. This is what every author could run, and it is why
    # nobody saw it coming.
    r = run_check("check_lane_inventory.py", "--workflow", str(wf))
    assert r.returncode == EXIT_PASS, r.stdout

    # Union view — red, naming the step and which side it came from.
    r = run_check(
        "check_lane_inventory.py", "--workflow", str(wf), "--union-with", "main"
    )
    assert r.returncode == EXIT_FAIL_CONDITION, r.stdout
    assert "FAIL(condition=unclassified_lane_step)" in r.stdout
    assert "A Step Nobody On My Branch Can See" in r.stdout
    assert "union of this tree and main" in r.stdout


def test_union_is_green_when_both_sides_are_classified(tmp_path):
    """The other polarity: --union-with must not simply be red all the time."""
    repo = _two_branch_repo(
        tmp_path, mine=CLASSIFIED_STEP, theirs=OTHER_CLASSIFIED_STEP
    )
    wf = repo / ".github" / "workflows" / "ci.yml"
    r = run_check(
        "check_lane_inventory.py", "--workflow", str(wf), "--union-with", "main"
    )
    assert r.returncode == EXIT_PASS, r.stdout
    assert "1 step(s) present only there" in r.stdout


def test_an_unreadable_union_reference_is_an_outage_not_a_pass(tmp_path):
    """A missing ref degrades the check to the branch-only view it exists to replace.

    Silently returning to that view would be the worst outcome available: green, for the
    reason the check was written to stop being green for.
    """
    repo = _two_branch_repo(tmp_path, mine=CLASSIFIED_STEP, theirs=CLASSIFIED_STEP)
    wf = repo / ".github" / "workflows" / "ci.yml"
    r = run_check(
        "check_lane_inventory.py",
        "--workflow",
        str(wf),
        "--union-with",
        "no/such/ref",
        "--union-required",
    )
    assert r.returncode == EXIT_ERROR_INSTRUMENT, r.stdout
    assert "ERROR(instrument=union_reference_unreadable)" in r.stdout
    assert "branch's own view" in r.stdout


def test_a_green_check_must_say_whether_its_falsifier_was_planted_or_observed():
    """The distinction found while classifying Switch's screen, applied to everything.

    A planted falsifier proves the check works on the shape somebody wrote for it. It
    does not show the check is load-bearing. Most of this project's green rests on
    planted arms -- including my own tick screen -- and the table has to say so.
    """
    import importlib

    inv = importlib.import_module("lane_inventory")
    assert inv.validate() == []
    for c in inv.CHECKS:
        if c.is_green():
            assert c.falsifier in inv.ALL_FALSIFIERS, c.id
    # And the honest self-assessment: my own screen is PLANTED, not OBSERVED.
    tick = next(c for c in inv.CHECKS if c.id == "hostfree.tick_conversion_screen")
    assert tick.falsifier == inv.FALSIFIER_PLANTED
    planted, observed, note = inv.falsifier_census(inv.LANE_HOSTFREE)
    assert planted > observed
    assert "PLANTED" in note


def test_switchs_tautological_screen_is_not_admitted_as_green():
    """It scanned 1,056 assertions and found 0, and neither assertion defect that
    actually occurred here is within its reach -- by its own first paragraph."""
    import importlib

    inv = importlib.import_module("lane_inventory")
    c = next(x for x in inv.CHECKS if x.id == "hostfree.tautological_assertions")
    assert c.status == inv.UNDEMONSTRATED
    assert not c.is_green()
    assert c.arm_broken is None
    assert any("NEITHER" in m for m in c.misses)
    # One UNDEMONSTRATED check holds the whole lane at `operational`.
    cls, _ = inv.lane_classification(inv.LANE_HOSTFREE)
    assert cls == "operational"


# ---------------------------------------------------------------------------
# Criterion 12: the three things a census line cannot supply about itself.
# ci/check_census_completeness.py + ci/census_surface_map.json.
# ---------------------------------------------------------------------------


def _census_map() -> dict:
    return json.loads((CI_DIR / "census_surface_map.json").read_text(encoding="utf-8"))


def test_the_whole_is_not_derived_from_the_census():
    """R11's falsifier-that-cannot-fire, stated as a test.

    If the denominator came from the same list as the numerator, 12/12 would be true by
    construction. The whole is enumerated from production Rust -- counters.rs fields,
    trace.rs Phase variants, ONNXRUNTIME_EP_VULKAN_* switches -- none of which the census
    writes. The check that this stays true is that the screen's own source never reads
    the census's mechanism list from anywhere but the artifact the census PRODUCED.
    """
    src = (CI_DIR / "check_census_completeness.py").read_text(encoding="utf-8")
    assert "_MECHANISMS" not in src, (
        "the screen must not read the census's own mechanism list; R10 says the "
        "falsifier is an artifact the mechanism produced, never a reading of its code"
    )
    assert '"tests"' not in src and "'tests'" not in src, (
        "the screen must not open anything under tests/ — its numerator is the artifact "
        "the census wrote, and its denominator is production Rust"
    )
    r = run_check("check_census_completeness.py")
    assert r.returncode == EXIT_PASS, r.stdout
    assert "the census's twelve is twelve OF" in r.stdout
    # The denominator is bigger than the numerator, and by a lot. That is the finding.
    assert "50 instrumented surfaces" in r.stdout or "instrumented surfaces" in r.stdout


def test_a_new_instrumented_surface_the_census_does_not_know_about_goes_red():
    """The arm that makes the completeness claim falsifiable at all."""
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "src"
        shutil.copytree(REPO_ROOT / "rust" / "src", src)
        counters = src / "counters.rs"
        text = counters.read_text(encoding="utf-8")
        anchor = "    pub abi_version: u32,"
        assert anchor in text, "anchor moved; this test would silently stop testing"
        counters.write_text(
            text.replace(anchor, anchor + "\n    pub planted_by_a_test: u64,", 1),
            encoding="utf-8",
        )
        r = run_check("check_census_completeness.py", "--rust-src", str(src))
    assert r.returncode == EXIT_FAIL_CONDITION, r.stdout
    assert "FAIL(condition=unmapped_surface)" in r.stdout
    assert "planted_by_a_test" in r.stdout


def test_extent_reports_unobservable_rather_than_zero_over_zero():
    """R12 applied to the screen's own coverage.

    A host-side mechanism has no surface in the independent whole. Reporting it as 0/0 --
    or worse, as complete -- is the identity defect the screen exists to refuse.
    """
    import importlib

    mod = importlib.import_module("check_census_completeness")
    rows = mod.extent_table(
        {"has_surfaces": [{"id": "a_counter"}]},
        ["has_surfaces", "host_side_only"],
        [{"name": "x.json", "observations": {"has_surfaces": "", "host_side_only": "PASS"}}],
    )
    by_mech = {r["mechanism"]: r for r in rows}
    assert by_mech["host_side_only"]["extent"] == "UNOBSERVABLE"
    assert by_mech["has_surfaces"]["extent"] == "0/1"
    assert not any(r["extent"] == "0/0" for r in rows)

    r = run_check("check_census_completeness.py")
    assert r.returncode == EXIT_PASS, r.stdout
    line = next(
        ln for ln in r.stdout.splitlines() if ln.strip().startswith("layering_lint")
    )
    assert "UNOBSERVABLE" in line


def test_name_content_has_three_states_and_never_calls_one_arm_invariant():
    import importlib

    mod = importlib.import_module("check_census_completeness")
    arts = [
        {"name": "a.json", "observations": {"m": "ARMED"}},
    ]
    rows = mod.name_content(["m"], arts)
    assert rows[0]["state"] == mod.NAME_UNOBSERVABLE, (
        "one arm is unmeasured, not invariant -- reporting it invariant is reporting 0 "
        "where the event could not occur"
    )
    arts.append({"name": "b.json", "observations": {"m": "ARMED"}})
    assert mod.name_content(["m"], arts)[0]["state"] == mod.NAME_INVARIANT
    arts.append({"name": "c.json", "observations": {"m": "DISARMED"}})
    assert mod.name_content(["m"], arts)[0]["state"] == mod.NAME_VARIES


def test_every_censused_mechanism_has_a_name_claim_and_none_claims_verified():
    """The Phase::Record guard: a name is a claim, and no name here is yet verified."""
    doc = _census_map()
    names = doc["mechanism_names"]
    assert names, "no name claims on record"
    for entry in names:
        assert entry["discriminator"], entry["mechanism"]
        assert entry["name_verified"] is False, (
            f"{entry['mechanism']} records its name as verified; the screen requires the "
            "observation to have varied across arms before that is admissible"
        )
    rec = next(e for e in names if e["mechanism"] == "gpu_tracer")
    assert "TRACE_GPU" in rec["discriminator"] or "phases" in rec["discriminator"]


def test_the_surface_map_records_the_standing_gaps_rather_than_hiding_them():
    """`uncensused` exists so an unmapped surface can stay red without the standing
    gaps making the screen permanently red and therefore unread."""
    doc = _census_map()
    gaps = [s for s in doc["surfaces"] if s["disposition"] == "uncensused"]
    assert gaps, "the census covers every instrumented surface -- verify before believing"
    for gap in gaps:
        assert gap["owner"] not in ("", "unassigned"), gap["id"]
        assert len(gap["reason"]) > 20, gap["id"]
    assert doc["not_a_closure"]


def test_the_screen_will_not_narrow_to_source_only_silently():
    """The --union-required lesson, applied again: a check that quietly drops half its
    input when the input is missing returns to the view it was written to replace."""
    with tempfile.TemporaryDirectory() as td:
        r = run_check("check_census_completeness.py", "--artifacts", td)
        assert r.returncode == EXIT_ERROR_INSTRUMENT, r.stdout
        assert "census_artifacts_unavailable" in r.stdout
        r2 = run_check("check_census_completeness.py", "--artifacts", td, "--no-artifacts")
        assert r2.returncode == EXIT_PASS, r2.stdout
        assert "UNOBSERVABLE in this frame" in r2.stdout


def test_census_completeness_is_registered_and_does_not_close_row_12():
    import importlib

    inv = importlib.import_module("lane_inventory")
    c = next(x for x in inv.CHECKS if x.id == "hostfree.census_completeness")
    assert c.falsifier == inv.FALSIFIER_PLANTED
    assert any("row 12" in m or "Trinity" in m for m in c.misses)


# ---------------------------------------------------------------------------
# Issue #61: `_defined_symbols()` / `check_stale_absence_claims()` must hold out
# `#[cfg(test)]` spans using `test_line_span()`, exactly as `extract_env_switches()`
# already does — a symbol defined only inside a test module is not a production
# definition. These are the reviewer's twelve probes, reproduced directly against the
# real oracle, plus the nested-module/malformed/cfg_attr/macro edge cases the fix must
# not regress.
# ---------------------------------------------------------------------------


def _write_rust_tree(tmp_path: Path, files: dict) -> Path:
    src = tmp_path / "rust_src"
    src.mkdir()
    for name, text in files.items():
        (src / name).write_text(text, encoding="utf-8")
    return src


def _stale_claims_for_multi(tmp_path: Path, rust_text: str, symbols: list) -> tuple:
    import importlib

    mod = importlib.import_module("check_census_completeness")
    src = _write_rust_tree(tmp_path, {"probe.rs": rust_text})
    mapping = {
        "surfaces": [
            {
                "kind": "counter",
                "id": f"probe_entry_{i}",
                "disposition": "uncensused",
                "reason": "probe entry for issue #61's oracle tests",
                "owner": "Link",
                "absence_claims": [symbol],
            }
            for i, symbol in enumerate(symbols)
        ]
    }
    return mod.check_stale_absence_claims(src, mapping)


def _stale_claims_for(tmp_path: Path, rust_text: str, symbol: str) -> tuple:
    return _stale_claims_for_multi(tmp_path, rust_text, [symbol])


def test_probe_01_real_production_fn_is_flagged(tmp_path):
    stale, _frame = _stale_claims_for(tmp_path, "fn scratch_symbol() {}\n", "scratch_symbol")
    assert any(c["symbol"] == "scratch_symbol" for c in stale), stale


def test_probe_02_genuinely_absent_symbol_is_not_flagged(tmp_path):
    stale, _frame = _stale_claims_for(tmp_path, "fn something_else() {}\n", "scratch_symbol")
    assert not stale, stale


def test_probe_03_named_only_in_a_line_comment_is_not_flagged(tmp_path):
    stale, _frame = _stale_claims_for(
        tmp_path, "// fn scratch_symbol() {}\nfn other() {}\n", "scratch_symbol"
    )
    assert not stale, stale


def test_probe_04_named_only_in_a_doc_comment_is_not_flagged(tmp_path):
    stale, _frame = _stale_claims_for(
        tmp_path, "/// fn scratch_symbol() {}\nfn other() {}\n", "scratch_symbol"
    )
    assert not stale, stale


def test_probe_05_named_only_inside_a_string_literal_is_not_flagged(tmp_path):
    stale, _frame = _stale_claims_for(
        tmp_path, 'fn other() { let s = "fn scratch_symbol() {}"; }\n', "scratch_symbol"
    )
    assert not stale, stale


def test_probe_06_named_only_in_a_block_comment_is_not_flagged(tmp_path):
    stale, _frame = _stale_claims_for(
        tmp_path, "/* fn scratch_symbol() {} */\nfn other() {}\n", "scratch_symbol"
    )
    assert not stale, stale


def test_probe_07_defined_only_inside_cfg_test_mod_is_not_flagged(tmp_path):
    """The hole issue #61 found: a symbol reachable only from `#[cfg(test)] mod` is not
    a production definition, and must not convict an `absence_claims` entry naming it."""
    rust = (
        "pub fn production_untouched() {}\n"
        "\n"
        "#[cfg(test)]\n"
        "mod tests {\n"
        "    use super::*;\n"
        "\n"
        "    fn scratch_symbol() {}\n"
        "\n"
        "    #[test]\n"
        "    fn a_test() {\n"
        "        scratch_symbol();\n"
        "    }\n"
        "}\n"
    )
    stale, frame = _stale_claims_for(tmp_path, rust, "scratch_symbol")
    assert not stale, stale
    assert frame["defined_symbol_lines_held_out"] > 0, frame


def test_probe_08_static_definition_is_flagged(tmp_path):
    stale, _frame = _stale_claims_for(
        tmp_path, "static SCRATCH_SYMBOL: u32 = 1;\n", "SCRATCH_SYMBOL"
    )
    assert any(c["symbol"] == "SCRATCH_SYMBOL" for c in stale), stale


def test_probe_09_generic_fn_definition_is_flagged(tmp_path):
    stale, _frame = _stale_claims_for(
        tmp_path, "fn scratch_symbol<T: Clone>(x: T) -> T { x }\n", "scratch_symbol"
    )
    assert any(c["symbol"] == "scratch_symbol" for c in stale), stale


def test_probe_10_prose_only_reason_text_is_never_mined(tmp_path):
    """The screen only rules on an explicit `absence_claims` entry. A `reason` string
    asserting an absence in prose, with no `absence_claims`, must never be checked --
    text-mining free-form prose is exactly the defect class this project's own
    SOURCE-COSMETIC/digest incidents warn against."""
    import importlib

    mod = importlib.import_module("check_census_completeness")
    src = _write_rust_tree(tmp_path, {"probe.rs": "fn scratch_symbol() {}\n"})
    mapping = {
        "surfaces": [
            {
                "kind": "counter",
                "id": "probe_entry",
                "disposition": "uncensused",
                "reason": "no counter names the variant scratch_symbol",
                "owner": "Link",
            }
        ]
    }
    stale, _frame = mod.check_stale_absence_claims(src, mapping)
    assert not stale, stale


def test_probe_11_non_circular_never_reads_its_own_bookkeeping_or_a_census_artifact():
    """The oracle greps the tree directly; it must never read the map's own
    `censused`/`uncensused` bookkeeping or a wiring-census artifact to decide a claim's
    truth. Reading either would make the check circular -- it could never disagree with
    the very thing it exists to check against."""
    src = (CI_DIR / "check_census_completeness.py").read_text(encoding="utf-8")
    start = src.index("def check_stale_absence_claims")
    body = src[start : src.index("\n\n\n", start)]
    assert "disposition" not in body, body
    assert "observations" not in body, body
    assert "load_census_artifacts" not in body, body


def test_probe_12_nested_modules_and_braces_hold_out_correctly(tmp_path):
    """A `#[cfg(test)] mod` containing further nested `mod`/`match`/block braces must be
    held out IN FULL -- the brace counter must not close early on an inner block's `}`,
    and production code textually AFTER the test module must still be read."""
    rust = (
        "pub fn production_untouched() {}\n"
        "\n"
        "#[cfg(test)]\n"
        "mod tests {\n"
        "    mod nested {\n"
        "        fn scratch_symbol() {\n"
        "            match 1 {\n"
        "                1 => { let _x = { 2 }; }\n"
        "                _ => {}\n"
        "            }\n"
        "        }\n"
        "    }\n"
        "\n"
        "    #[test]\n"
        "    fn a_test() {}\n"
        "}\n"
        "\n"
        "fn after_test_mod_is_still_production() {}\n"
    )
    stale, _frame = _stale_claims_for_multi(
        tmp_path, rust, ["scratch_symbol", "after_test_mod_is_still_production"]
    )
    flagged = {c["symbol"] for c in stale}
    assert "scratch_symbol" not in flagged, stale
    assert "after_test_mod_is_still_production" in flagged, stale


def test_unterminated_cfg_test_block_holds_out_to_end_of_file_not_a_crash(tmp_path):
    """A `#[cfg(test)] mod` with no closing brace (a malformed/truncated file) must not
    crash the oracle -- `test_line_span()`'s brace counter runs to EOF and holds out
    every remaining line, the conservative-safe direction (a symbol in the unterminated
    span reads as held-out, never as a confirmed production definition), and the
    held-out count in the published frame reflects it rather than silently reporting a
    smaller number."""
    rust = (
        "pub fn production_untouched() {}\n"
        "\n"
        "#[cfg(test)]\n"
        "mod tests {\n"
        "    fn scratch_symbol() {\n"
        "        let _x = 1;\n"
    )
    stale, frame = _stale_claims_for(tmp_path, rust, "scratch_symbol")
    assert not stale, stale
    assert frame["defined_symbol_lines_held_out"] >= 3, frame


def test_cfg_attr_test_combinations_share_test_line_span_s_existing_limit(tmp_path):
    """`test_line_span()` recognises the literal `#[cfg(test)]` attribute text, not
    `#[cfg_attr(test, ...)]` or `#[cfg(all(test, ...))]` -- the same limit
    `extract_env_switches()` already lives with (both import the one `test_line_span`).
    This oracle inherits that limit rather than growing a bespoke Rust attribute parser
    (R11: a decomposition that appears to close every case is the hardest kind of
    wrong): a `#[cfg(all(test, ...))]`-gated module is NOT held out today, so a symbol
    defined only inside one still reads as production. Documented here as a known,
    shared limit rather than discovered later as a surprise."""
    rust = (
        '#[cfg(all(test, feature = "x"))]\n'
        "mod tests {\n"
        "    fn scratch_symbol() {}\n"
        "}\n"
    )
    stale, _frame = _stale_claims_for(tmp_path, rust, "scratch_symbol")
    assert any(c["symbol"] == "scratch_symbol" for c in stale), (
        "expected the documented parser limit (only the literal #[cfg(test)] text is "
        "recognised) to read this as production; if this now passes, test_line_span's "
        "recognition grew and this test's premise needs revisiting, not deleting"
    )


def test_macro_templated_fn_name_is_invisible_to_the_regex_oracle(tmp_path):
    """Known, shared limit: every extractor in this file is a regex over literal source
    text, never a Rust parser (R11). A function name generated from a `macro_rules!`
    template (`fn $name() {}`) never appears as the literal identifier after `fn` in the
    source text, so it is invisible to `_defined_symbols()` -- consistent with, not a
    regression from, this fix."""
    rust = (
        "macro_rules! make_fn {\n"
        "    ($name:ident) => {\n"
        "        fn $name() {}\n"
        "    };\n"
        "}\n"
        "make_fn!(scratch_symbol);\n"
    )
    stale, _frame = _stale_claims_for(tmp_path, rust, "scratch_symbol")
    assert not stale, (
        "the literal identifier 'scratch_symbol' never appears after 'fn' in the "
        "source text, so the oracle correctly (if conservatively) does not see it"
    )


# ---------------------------------------------------------------------------
# Issue #47: the two ways this screen went blind, each pinned by its own arm.
# ---------------------------------------------------------------------------
#
# ONNXRUNTIME_EP_VULKAN_RANK_INFERENCE reached main unmapped (PR #46 / issue #8) and
# the census screen was correctly red about it for ten days. Neither of the arms
# below would have prevented THAT -- the screen already named the surface, and CI
# simply never ran because of an Actions outage. They exist because chasing that
# regression surfaced two separate defects in the instruments themselves, and in both
# the instrument reported an OUTAGE where a reader would read COVERAGE.


def test_the_screen_survives_non_ascii_in_the_map_s_own_prose():
    """Found while writing the RANK_INFERENCE entry (#47): a U+2192 arrow in a
    `reason` string made the screen exit ERROR(instrument=screen_raised) with
    UnicodeEncodeError, on a Windows console, AFTER it had done its work correctly.

    The map's prose is meant to be written by humans explaining a gap, and this file's
    own entries already carry section signs and em-dashes. A screen that dies on a
    character in its own input turns a real unmapped surface into an instrument
    outage, which DESIGN.md 10.0.1 R13 says is explicitly NOT a detection -- so the
    gap would be reported as "the screen could not answer" instead of as the finding
    it is.

    This arm does NOT use run_check(): run_check forces PYTHONIOENCODING=utf-8 into
    the child, which is precisely the condition under which the bug cannot happen, so
    an arm built on it would pass whether or not the screen is fixed. The shipping
    invocation has no such pin -- the ci.yml step is a bare `python
    ci/check_census_completeness.py`, and a developer's shell is barer still. So this
    pins the child to a narrow codepage on purpose. cp1252 is chosen because it is
    what a default Windows shell actually hands the screen; pinning it explicitly
    makes the arm reproduce that condition on Linux CI too, rather than passing
    vacuously there because the locale happened to be UTF-8."""
    doc = json.loads((CI_DIR / "census_surface_map.json").read_text(encoding="utf-8"))
    # Inject into an EXISTING uncensused entry rather than appending a new one. The
    # screen only prints a `reason` for surfaces it actually reports, so a fabricated
    # id -- which matches nothing in production Rust -- would have its prose skipped
    # and the arm would pass without ever exercising the encode path.
    victims = [
        s
        for s in doc["surfaces"]
        if s.get("kind") == "env_switch" and s.get("disposition") == "uncensused"
    ]
    assert victims, "no uncensused env_switch to plant into; this arm has lost its subject"
    # An arrow, a CJK glyph and an emoji: all outside cp1252.
    victims[0]["reason"] = (
        "planted \u2192 non-ascii \u6f22 \U0001f9ea prose for issue #47. "
        + victims[0]["reason"]
    )
    with tempfile.TemporaryDirectory() as td:
        m = Path(td) / "map.json"
        m.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        env = {**os.environ, "PYTHONIOENCODING": "cp1252"}
        env.pop("PYTHONUTF8", None)
        proc = subprocess.run(
            [sys.executable, str(CI_DIR / "check_census_completeness.py"), "--map", str(m)],
            capture_output=True,
            env=env,
        )
        out = proc.stdout.decode("utf-8", errors="replace") + proc.stderr.decode(
            "utf-8", errors="replace"
        )
        assert "screen_raised" not in out, out[-3000:]
        assert "UnicodeEncodeError" not in out, out[-3000:]
        assert "CENSUS-EXTENT:" in out, out[-3000:]
        # It reached an observation, and specifically it got far enough to render the
        # prose that used to kill it.
        assert "planted" in out, out[-3000:]
        assert proc.returncode in (EXIT_PASS, EXIT_FAIL_CONDITION), out[-3000:]


def test_the_census_negative_control_can_actually_read_the_screen():
    """The arm that should have caught #47 was blind. `_run` decoded the child as
    UTF-8 while letting the child pick cp1252 for itself, so on Windows every one of
    the twelve arms -- including "a new EP env switch appears and nobody tells the
    census" -- came back with empty output and reported `arm_did_not_fire`.

    `arm_did_not_fire` and "the harness could not read the child" are opposite
    findings that looked identical. This asserts the harness reads real text, so a
    future `arm_did_not_fire` means the screen was silent and nothing else."""
    import importlib

    nc = importlib.import_module("negative_control_census_completeness")
    code, out = nc._run(nc.RUST_SRC, nc.MAP, nc.ARTIFACTS)
    assert out.strip(), "the control read nothing back from the screen"
    assert "CENSUS-EXTENT:" in out, out[:2000]
    assert code == nc.EXIT_PASS, out[-2000:]
    # The decode is lossless on the real tree, not merely non-empty: a report full of
    # U+FFFD would still satisfy the assertions above while hiding the surface names
    # the arms match on.
    assert "\ufffd" not in out, "the control decoded the screen lossily"


# ---------------------------------------------------------------------------
# The device-loss screen (ci/check_device_loss.py)
# ---------------------------------------------------------------------------
#
# A lost device that exits 0 does not look like a failure; it looks like a smaller
# number. These assert on the printed CONDITION, never on a count of findings (R13).


DEVICE_LOSS_LINE = (
    "[vulkan-ep] ERROR: vkWaitForFences failed: The logical device has been lost.\n"
)


def test_device_loss_screen_is_red_on_the_ep_s_own_text():
    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "run.log"
        log.write_text(DEVICE_LOSS_LINE + "EXIT = 0\n", encoding="utf-8")
        r = run_check("check_device_loss.py", str(log))
        assert r.returncode == EXIT_FAIL_CONDITION, r.stdout
        assert "FAIL(condition=device_lost_reported)" in r.stdout
        assert "The logical device has been lost" in r.stdout


def test_device_loss_screen_does_not_take_exit_zero_as_evidence():
    """The defect IS an exit status of 0, so accepting one would be accepting the
    defect as a filter. The screen must say so where a reader will see it."""
    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "run.log"
        log.write_text(DEVICE_LOSS_LINE, encoding="utf-8")
        r = run_check("check_device_loss.py", str(log))
        assert "no exit status was read" in r.stdout


def test_device_loss_structural_rule_needs_no_log_text_at_all():
    """An artifact that declared 25 inferences and observed 9 is a short run, whatever
    the log says or fails to say. This is the arm that survives a log-format change."""
    with tempfile.TemporaryDirectory() as td:
        art = Path(td) / "points.json"
        art.write_text(
            json.dumps({"points": [{"iters": 25, "compute_calls": 9}]}), encoding="utf-8"
        )
        r = run_check("check_device_loss.py", str(art))
        assert r.returncode == EXIT_FAIL_CONDITION, r.stdout
        assert "FAIL(condition=observation_ended_early)" in r.stdout
        assert "observation ending early rather than a smaller quantity" in r.stdout


def test_device_loss_screen_does_not_punish_a_producer_who_reported_it():
    """The same truncation under a rejected_* key is the producer reporting a short
    run, which is the behaviour we want. Counting it would make the honest artifact
    look like the defective one."""
    with tempfile.TemporaryDirectory() as td:
        art = Path(td) / "points.json"
        art.write_text(
            json.dumps({"rejected_points": [{"iters": 25, "compute_calls": 9}]}),
            encoding="utf-8",
        )
        r = run_check("check_device_loss.py", str(art))
        assert r.returncode == EXIT_PASS, r.stdout


def test_device_loss_screen_reports_unobservable_not_clean_for_named_run_conditions():
    """R12 on the screen's own coverage: three conditions are only decidable on a file
    the caller declares is one run's evidence, because controls here emit those texts
    on purpose. A tree scan must say it did not look, not that it found nothing."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "arts"
        d.mkdir()
        (d / "a.log").write_text("nothing interesting\n", encoding="utf-8")
        r = run_check("check_device_loss.py", str(d))
        assert r.returncode == EXIT_PASS, r.stdout
        assert "UNOBSERVABLE in this run" in r.stdout
        assert "runtime_fallback_announced" in r.stdout
        assert "not zero findings" in r.stdout


def test_device_loss_screen_reports_an_outage_rather_than_a_pass_when_given_nothing():
    r = run_check("check_device_loss.py")
    assert r.returncode == EXIT_ERROR_INSTRUMENT, r.stdout
    assert "ERROR(instrument=no_paths_given)" in r.stdout
    assert "NOT a detection" in r.stdout


def test_device_loss_exclusion_list_entries_all_carry_a_reason_owner_and_date():
    """The exclusion list is the most dangerous file in this check. An entry without a
    reason, an owner and a date is an exclusion nobody can review."""
    doc = json.loads(
        (CI_DIR / "device_loss_incident_records.json").read_text(encoding="utf-8")
    )
    assert doc["records"], "an empty list would make the exclusion machinery untested"
    for rec in doc["records"]:
        for field in ("file", "reason", "owner", "date"):
            assert rec.get(field), rec
        assert len(rec["reason"]) > 40, rec["file"]


def test_device_loss_exclusion_list_has_no_rot():
    """An entry naming a file that is gone is an exclusion still in force over nothing
    anyone can inspect."""
    doc = json.loads(
        (CI_DIR / "device_loss_incident_records.json").read_text(encoding="utf-8")
    )
    for rec in doc["records"]:
        assert (REPO_ROOT / rec["file"]).exists(), rec["file"]


def test_device_loss_exclusion_entries_declare_the_findings_they_account_for():
    """Issue #24. Before 2026-08-07 an entry silenced its file wholesale and forever and
    said nothing about what it silenced, so a NEW loss appended to an already-excluded
    file was invisible. An exclusion now names the finding(s) it accounts for."""
    import importlib

    mod = importlib.import_module("check_device_loss")
    doc = json.loads(
        (CI_DIR / "device_loss_incident_records.json").read_text(encoding="utf-8")
    )
    for rec in doc["records"]:
        witness = rec.get("witness")
        assert isinstance(witness, dict) and witness, rec["file"]
        for condition, count in witness.items():
            assert condition in mod.ARTIFACT_TREE_WIDE, (rec["file"], condition)
            assert isinstance(count, int) and count > 0, (rec["file"], condition)


def test_device_loss_exclusion_witness_counts_are_what_the_reader_actually_sees():
    """The number in the file is only worth something if it is the number the check
    holds the exclusion to. Re-read every excluded artifact through the check's own
    reader and compare. A drift here is a silenced finding."""
    import importlib

    mod = importlib.import_module("check_device_loss")
    doc = json.loads(
        (CI_DIR / "device_loss_incident_records.json").read_text(encoding="utf-8")
    )
    for rec in doc["records"]:
        path = REPO_ROOT / rec["file"]
        scan = mod.scan_artifact(path, named=False)
        assert not scan.error, (rec["file"], scan.error)
        observed = {c: len(v) for c, v in scan.hits.items() if v}
        assert observed == rec["witness"], (rec["file"], observed, rec["witness"])


def test_device_loss_exclusion_that_covers_nothing_is_red():
    """An exclusion over a file with no finding silences nothing today and can only
    blind the screen tomorrow. Green here would let an exclusion be parked in advance."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "bystander.log").write_text("a run that kept its device\n", encoding="utf-8")
        quiet = d / "quiet.log"
        quiet.write_text("nothing happened\n", encoding="utf-8")
        recs = d / "records.json"
        recs.write_text(
            json.dumps(
                {
                    "records": [
                        {
                            "file": str(quiet),
                            "witness": {"device_lost_reported": 1},
                            "reason": "planted by test_lane_checks to prove the audit fires",
                            "owner": "link",
                            "date": "2026-08-07",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        r = run_check("check_device_loss.py", "--incident-records", str(recs), str(d))
        assert r.returncode == EXIT_FAIL_CONDITION, r.stdout
        assert "incident_record_covers_nothing" in r.stdout


def test_device_loss_exclusion_is_red_when_the_file_says_more_than_it_accounts_for():
    """The whole point of the witness: a second loss recorded in an already-forgiven
    file must reach a human, not inherit the first loss's forgiveness."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "bystander.log").write_text("a run that kept its device\n", encoding="utf-8")
        loss = d / "loss.log"
        loss.write_text(
            DEVICE_LOSS_LINE.rstrip("\n") + " after 1775 dispatches\n"
            + DEVICE_LOSS_LINE.rstrip("\n") + " after 0 dispatches\n",
            encoding="utf-8",
        )
        entry = {
            "file": str(loss),
            "witness": {"device_lost_reported": 1},
            "reason": "planted by test_lane_checks to prove the audit bounds the exclusion",
            "owner": "link",
            "date": "2026-08-07",
        }
        recs = d / "records.json"
        recs.write_text(json.dumps({"records": [entry]}), encoding="utf-8")
        r = run_check("check_device_loss.py", "--incident-records", str(recs), str(d))
        assert r.returncode == EXIT_FAIL_CONDITION, r.stdout
        assert "incident_record_widened" in r.stdout

        # ...and green once it accounts for both, so the audit bounds exclusions rather
        # than abolishing them.
        entry["witness"] = {"device_lost_reported": 2}
        recs.write_text(json.dumps({"records": [entry]}), encoding="utf-8")
        ok = run_check("check_device_loss.py", "--incident-records", str(recs), str(d))
        assert ok.returncode == EXIT_PASS, ok.stdout


def test_device_loss_exclusion_without_a_witness_is_an_outage_not_a_pass():
    """A record the check cannot bound is an unreadable instrument. Accepting it would
    restore exactly the unbounded exclusion this machinery replaced."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        loss = d / "loss.log"
        loss.write_text(DEVICE_LOSS_LINE, encoding="utf-8")
        recs = d / "records.json"
        recs.write_text(
            json.dumps(
                {
                    "records": [
                        {
                            "file": str(loss),
                            "reason": "an exclusion over everything this file will ever say",
                            "owner": "link",
                            "date": "2026-08-07",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        r = run_check("check_device_loss.py", "--incident-records", str(recs), str(d))
        assert r.returncode == EXIT_ERROR_INSTRUMENT, r.stdout
        assert "incident_record_file_unreadable" in r.stdout


def test_device_loss_screen_and_fatal_log_have_different_extents():
    """They are two mechanisms and must never be quoted as one guarantee. The evidence
    is a file one is red on and the other is green on."""
    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "ep_only.log"
        log.write_text(DEVICE_LOSS_LINE, encoding="utf-8")
        mine = run_check("check_device_loss.py", str(log))
        theirs = run_check("check_fatal_log.py", str(log))
        assert mine.returncode == EXIT_FAIL_CONDITION, mine.stdout
        assert theirs.returncode != EXIT_FAIL_CONDITION, theirs.stdout


def test_the_shared_marker_list_matches_the_real_ort_line():
    """Was `..._still_misses_the_real_ort_line`, an xfail guard on a live finding written
    to go green when Trinity fixed `_verdict.FATAL_LOG_MARKERS`. She fixed it on
    2026-08-02 — at which point the test passed by *skipping its only branch* and
    asserted nothing at all. A guard that goes quiet when its subject is repaired cannot
    tell repair from a second regression, so it now asserts the agreement it was waiting
    for, in the three forms a real capture arrives in.
    """
    real = "Falling back to ['CPUExecutionProvider'] and retrying."
    assert _verdict.find_fatal_log_lines(real + "\n"), (
        "The shared vocabulary no longer matches the line ORT actually prints. Between "
        "2026-07-31 and 2026-08-02 it did not, ci/check_fatal_log.py read a log "
        "announcing the fallback twice as clean, and every positive it produced in that "
        "window was test_phi35.py's own docstring echoed by pytest. Do not restore a "
        "plain substring here."
    )

    # It wraps. Matching had to move off splitlines() for this reason.
    wrapped = "Falling back to\n  ['CPUExecutionProvider'] and\n  retrying."
    assert _verdict.find_fatal_log_lines(wrapped), "a wrapped announcement must match"

    # ORT's C++ sink writes UTF-16LE into an otherwise UTF-8 file; decoded as UTF-8 the
    # message is NUL-separated and unfindable by substring.
    wide = (real + "\n").encode("utf-16-le").decode("utf-8", errors="replace")
    assert _verdict.find_fatal_log_lines(wide), (
        "the wide-encoded form must match; trinity-suite-dev1.log happens to carry both "
        "encodings, and had it carried only this one a substring search would have seen "
        "an empty log"
    )

    # And the extent that is deliberately NOT covered, asserted so that widening the
    # markers cannot happen silently: see ci/negative_control_device_loss.py's reach arm,
    # which requires check_fatal_log to stay green on EP-reported device loss.
    assert not _verdict.find_fatal_log_lines(
        "[vulkan-ep] ERROR: vkQueueSubmit failed: The logical device has been lost.\n"
    ), (
        "check_fatal_log's extent is ORT's announcement; the EP's own device-lost text is "
        "ci/check_device_loss.py's. Widening this list buys coverage by destroying the "
        "demonstration that the two checks have separate reach."
    )


def test_device_loss_checks_are_registered_with_honest_reach():
    import importlib

    inv = importlib.import_module("lane_inventory")
    screen = next(x for x in inv.CHECKS if x.id == "hostfree.device_loss_screen")
    assert screen.falsifier == inv.FALSIFIER_OBSERVED
    assert any("UNOBSERVABLE" in m for m in screen.misses)
    assert any("exclusion list" in m for m in screen.misses)
    # PR #65 review (Morpheus): the witness binds a count per condition, not the
    # content behind it, so a frame substitution (same count, different lines) is
    # invisible to both the screen and the witness-equality test. This is the sibling
    # of the de-duplication limit already recorded, and must not silently disappear.
    assert any(
        "COUNT per condition" in m and "frame substitution" in m for m in screen.misses
    ), "the count-not-content / frame-substitution residual must stay named in misses"
    control = next(
        x for x in inv.CHECKS if x.id == "hostfree.device_loss_negative_control"
    )
    assert any("PLANTED" in m for m in control.misses)
    lane = next(x for x in inv.CHECKS if x.id == "device.device_loss_screen")
    assert any("disable_cpu_ep_fallback" in m for m in lane.misses)
    spot = next(
        b for b in inv.BLIND_SPOTS if b.id == "runtime_device_loss_exits_zero"
    )
    assert "exits 0" in spot.defect


_LANDING_READING_FORBIDDEN_ARGS = ("--follow", "-M", "-C", "--find-renames", "--find-copies")


def _first_commit_adding_current_path(rel_path: str) -> str:
    """Short SHA of the first commit that added `rel_path` **at exactly this path**.

    THIS is the source of truth for every "landed at <sha>" claim a doc or a record
    file makes, and it is deliberately read WITHOUT `git log --follow`.

    Issue #24 item (g) (Morpheus): `--follow` answers a *different* question — "where
    does the content now at this path first appear, across renames" — and for the two
    `*-devunset-liveness.json` artifacts it answers `b669eb9` / `b262292`, the
    pre-rename ancestors, rather than `fb5f0b2` / `1f41095`, the commits that actually
    introduced the paths `docs/PLATFORMS.md` §7.18.11 cites. Both readings are true
    about the repository; only one of them is the answer to *"when did this file, under
    this name, land"*. A landing claim checked against the lineage reading silently
    accepts a pre-rename SHA, so the two readings are separate functions here rather
    than one function with a comment.

    Fail-closed in three ways, each of them load-bearing rather than documentary:

    1. The argv is asserted free of every rename-following flag at call time, so
       re-adding `--follow` to this helper fails every caller instead of quietly
       changing what "landed" means.
    2. `rel_path` must still be tracked at HEAD. A landing claim about a path this
       checkout does not have is an instrument outage, not a pass.
    3. An empty result is an assertion (shallow clone), never a permissive default.

    For the rename lineage, call `_first_commit_in_rename_lineage()` — never this.
    """
    argv = ["git", "log", "--diff-filter=A", "--format=%h", "--", rel_path]
    assert not any(bad in argv for bad in _LANDING_READING_FORBIDDEN_ARGS), (
        "the landing reading must not cross a rename: with --follow this helper "
        "returns the pre-rename ancestor (b669eb9 / b262292 for the two "
        "*-devunset-liveness.json artifacts) instead of the commit that introduced "
        "the current path (fb5f0b2 / 1f41095). Use _first_commit_in_rename_lineage() "
        "if the lineage is what you actually want."
    )

    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", rel_path],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert tracked.returncode == 0, (
        f"{rel_path} is not tracked at HEAD, so 'the commit that added this path' is "
        f"not a question this checkout can answer: {tracked.stderr.strip()!r}"
    )

    proc = subprocess.run(argv, capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert proc.returncode == 0, (rel_path, proc.stderr)
    shas = [s for s in proc.stdout.splitlines() if s.strip()]
    assert shas, f"no commit adds {rel_path} in this checkout's history (shallow clone?)"
    return shas[-1]  # oldest -> the introduction of *this* path


def _first_commit_in_rename_lineage(rel_path: str) -> str:
    """Short SHA of the oldest commit in `rel_path`'s **content lineage**, i.e. the
    `git log --follow` reading, which crosses renames.

    This is NOT a landing commit and must never be used to check, or to author, a
    "landed at <sha>" claim in a doc or a record — see
    `_first_commit_adding_current_path()` for that. It exists so the distinction is
    exercised by tests rather than asserted by a comment: the two helpers demonstrably
    disagree on the `*-devunset-liveness.json` artifacts, and the tests below pin both
    answers so neither can drift into the other.
    """
    argv = [
        "git",
        "log",
        "--diff-filter=A",
        "--format=%h",
        "--follow",
        "--",
        rel_path,
    ]
    assert "--follow" in argv, (
        "the lineage reading is defined by --follow; without it this is just a second, "
        "confusable copy of the landing reading"
    )
    proc = subprocess.run(argv, capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert proc.returncode == 0, (rel_path, proc.stderr)
    shas = [s for s in proc.stdout.splitlines() if s.strip()]
    assert shas, f"no commit adds {rel_path} in this checkout's history (shallow clone?)"
    return shas[-1]


def _sha_claims_match(claimed: str, actual: str) -> bool:
    """True when two abbreviated SHAs are the same commit at their common length."""
    n = min(len(claimed), len(actual))
    return claimed[:n] == actual[:n]


def test_device_loss_incident_records_landing_commits_match_git_history_issue_65():
    """PR #65's review (Morpheus) found two landing-commit citations were wrong:
    `device_loss_gate-BOTHLANES.json` was attributed to `5353886` (that is really the
    commit for the armA_* capture files) when it actually landed at `1fc2ab6`. This
    reads the claim straight out of ci/device_loss_incident_records.json and checks it
    against real git history instead of trusting the prose, so a future transcription
    error here fails a test instead of surviving on review-comment memory alone.

    Issue #24 item (g): the reading is `_first_commit_adding_current_path()`, the
    non-`--follow` one. Seven of this file's records name paths whose lineage reading
    differs from their landing reading, so which helper this test calls decides whether
    a pre-rename ancestor would be accepted as a landing claim. It would not."""
    doc = json.loads(
        (CI_DIR / "device_loss_incident_records.json").read_text(encoding="utf-8")
    )
    checked = 0
    diverging = 0
    for rec in doc["records"]:
        landing = _first_commit_adding_current_path(rec["file"])
        lineage = _first_commit_in_rename_lineage(rec["file"])
        if not _sha_claims_match(landing, lineage):
            diverging += 1
        m = re.search(r"landed committed at ([0-9a-f]{7,40})", rec["reason"])
        if not m:
            # Fail-closed for the case Morpheus flagged as "nothing is wrong *yet*":
            # a record that gains a landing claim later is checked by the loop above,
            # and a record without one may not smuggle the lineage SHA into its prose.
            if not _sha_claims_match(landing, lineage):
                assert lineage not in rec["reason"], (
                    f"{rec['file']}: its prose names {lineage!r}, which is the "
                    f"pre-rename ancestor, not the commit that introduced this path "
                    f"({landing!r})"
                )
            continue
        claimed_sha = m.group(1)
        assert _sha_claims_match(claimed_sha, landing), (
            f"{rec['file']}: reason claims it landed at {claimed_sha!r}, but git's own "
            f"history says the first commit adding this path is {landing!r}"
        )
        assert not (
            not _sha_claims_match(landing, lineage)
            and _sha_claims_match(claimed_sha, lineage)
        ), (
            f"{rec['file']}: reason claims {claimed_sha!r}, which is this file's "
            f"pre-rename ancestor rather than the commit that introduced its current "
            f"path ({landing!r})"
        )
        checked += 1
    assert checked >= 1, "no record in this file makes a landing-commit claim any more"
    assert diverging >= 1, (
        "this test is only meaningful while at least one recorded artifact has a "
        "rename in its lineage; if that stops being true, the guard below "
        "(test_landing_reading_does_not_cross_renames_issue_24) is the one that must "
        "still hold"
    )


def test_landing_reading_does_not_cross_renames_issue_24():
    """Issue #24 item (g), the behavioural half.

    `_first_commit_adding_path()` used `--follow` and therefore resolved the two
    `*-devunset-liveness.json` artifacts to their pre-rename ancestors. The repair is
    two named readings, and this pins both of them on the exact pair Morpheus named, so
    the distinction cannot be collapsed back into one helper without a red test:

    * current path introduced  -> `fb5f0b2` / `1f41095`  (what §7.18.11 cites)
    * content lineage reaches  -> `b669eb9` / `b262292`  (true, but not a landing)
    """
    probe = "bench/results/validation_phi35_probe-devunset-liveness.json"
    counters = "bench/results/validation_phi35_counters-devunset-liveness.json"

    assert _sha_claims_match("fb5f0b2", _first_commit_adding_current_path(probe))
    assert _sha_claims_match("1f41095", _first_commit_adding_current_path(counters))

    assert _sha_claims_match("b669eb9", _first_commit_in_rename_lineage(probe))
    assert _sha_claims_match("b262292", _first_commit_in_rename_lineage(counters))

    # The whole point: the two readings disagree here. If they ever agreed, this test
    # would be passing for free, so say so out loud.
    for path in (probe, counters):
        assert not _sha_claims_match(
            _first_commit_adding_current_path(path),
            _first_commit_in_rename_lineage(path),
        ), f"{path} no longer exercises the rename distinction this test exists for"

    # And the substitution is rejected in the direction that matters: a pre-rename SHA
    # is not an acceptable answer to "where did this path land".
    for path, pre_rename in ((probe, "b669eb9"), (counters, "b262292")):
        assert not _sha_claims_match(
            pre_rename, _first_commit_adding_current_path(path)
        ), f"{path}: the landing reading returned the pre-rename ancestor {pre_rename}"

    # docs/PLATFORMS.md §7.18.11 cites fb5f0b28 / 1f41095 for this pair. Those are the
    # landing reading and demonstrably *not* the lineage reading, which is the whole of
    # Morpheus's item (g): a doc claim checked against --follow would have been called
    # wrong while being right. The positive half of this citation check lives in
    # test_platforms_doc_7_18_11_incident_counts_are_eight_and_ninth, which is kept
    # free of the lineage helper on purpose.
    for cited, path in (("fb5f0b28", probe), ("1f41095", counters)):
        assert not _sha_claims_match(cited, _first_commit_in_rename_lineage(path)), (
            f"{path}: the doc's citation {cited} now also matches the lineage reading, "
            "so this pair no longer distinguishes the two helpers"
        )


def test_landing_reading_is_fail_closed_and_the_mutant_is_visible_issue_24():
    """Issue #24 item (g), the mutant half — Morpheus asked for the distinction to be
    enforced rather than commented, so each way of erasing it is shown to be red.

    Mutant 1: put `--follow` back into the landing reading. Run here directly against
    git, and shown to produce the pre-rename ancestor, i.e. exactly the wrong answer
    `test_landing_reading_does_not_cross_renames_issue_24` would then fail on.

    Mutant 2: sneak a rename-following flag into the landing helper's argv. The helper
    asserts its own argv, so this is red at call time rather than at review time.

    Mutant 3: point the landing reading at a path that is not in this checkout. That is
    an instrument outage and must assert, not return a stale ancestor.
    """
    import inspect
    import textwrap

    probe = "bench/results/validation_phi35_probe-devunset-liveness.json"

    mutant = subprocess.run(
        ["git", "log", "--diff-filter=A", "--format=%h", "--follow", "--", probe],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert mutant.returncode == 0, mutant.stderr
    mutant_answer = [s for s in mutant.stdout.splitlines() if s.strip()][-1]
    assert _sha_claims_match("b669eb9", mutant_answer), (
        "the --follow mutant no longer reproduces; re-derive this test's fixtures "
        "before trusting it"
    )
    assert not _sha_claims_match(
        mutant_answer, _first_commit_adding_current_path(probe)
    ), "the landing reading has silently become the --follow reading"

    # Mutant 2 -- the guard is in the helper, not in a comment above it. Read the
    # helper's own AST so a rename-following flag cannot hide in its argv while the
    # docstring still explains why it should not be there.
    landing_ast = ast.parse(
        textwrap.dedent(inspect.getsource(_first_commit_adding_current_path))
    ).body[0]
    argv_assign = next(
        n
        for n in ast.walk(landing_ast)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "argv" for t in n.targets)
    )
    argv_literals = [
        e.value
        for e in ast.walk(argv_assign)
        if isinstance(e, ast.Constant) and isinstance(e.value, str)
    ]
    for bad in _LANDING_READING_FORBIDDEN_ARGS:
        assert bad not in argv_literals, (
            f"{bad} is back in the landing reading's argv; it would resolve renamed "
            "artifacts to their pre-rename ancestors"
        )
    landing_src = inspect.getsource(_first_commit_adding_current_path)
    assert "_LANDING_READING_FORBIDDEN_ARGS" in landing_src, (
        "the landing helper must assert its own argv, so re-adding --follow is a "
        "failing test rather than a silent change of meaning"
    )
    for bad in ("--follow", "-M", "-C", "--find-renames", "--find-copies"):
        assert bad in _LANDING_READING_FORBIDDEN_ARGS

    lineage_ast = ast.parse(
        textwrap.dedent(inspect.getsource(_first_commit_in_rename_lineage))
    ).body[0]
    lineage_argv = next(
        n
        for n in ast.walk(lineage_ast)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "argv" for t in n.targets)
    )
    assert "--follow" in [
        e.value
        for e in ast.walk(lineage_argv)
        if isinstance(e, ast.Constant) and isinstance(e.value, str)
    ], (
        "the lineage helper is defined by --follow; if it loses the flag the two "
        "helpers become indistinguishable and the separation is decorative"
    )

    # The ambiguous original must not come back under its old, unqualified name.
    assert not hasattr(sys.modules[__name__], "_first_commit_adding_path"), (
        "_first_commit_adding_path() was the ambiguous helper issue #24 (g) removed; "
        "callers must choose _first_commit_adding_current_path() (landing, source of "
        "truth for doc/record claims) or _first_commit_in_rename_lineage() (ancestry)"
    )

    # Mutant 3 -- fail-closed on a path this checkout does not carry.
    with pytest.raises(AssertionError, match="not tracked at HEAD"):
        _first_commit_adding_current_path(
            "bench/results/this-path-has-never-existed-issue-24.json"
        )


def test_doc_and_record_claims_never_call_the_rename_lineage_reading_issue_24():
    """Issue #24 item (g): the separation is only worth anything if the helper used for
    doc and record claims cannot silently become the lineage one. Reads this module's
    own AST and requires every call to `_first_commit_in_rename_lineage()` to sit in a
    test that exists to *demonstrate* the difference — never in one that checks a claim
    made by `docs/PLATFORMS.md` or by `ci/device_loss_incident_records.json`."""
    tree = ast.parse((CI_DIR / "test_lane_checks.py").read_text(encoding="utf-8"))

    # Tests allowed to read the lineage, and why: each of them asserts the two readings
    # *differ*. No test that checks a doc or record claim appears here.
    lineage_demonstrations = {
        "test_device_loss_incident_records_landing_commits_match_git_history_issue_65",
        "test_landing_reading_does_not_cross_renames_issue_24",
    }

    lineage_callers: set[str] = set()
    landing_callers: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Call) or not isinstance(sub.func, ast.Name):
                continue
            if sub.func.id == "_first_commit_in_rename_lineage":
                lineage_callers.add(node.name)
            elif sub.func.id == "_first_commit_adding_current_path":
                landing_callers.add(node.name)

    assert lineage_callers <= lineage_demonstrations, (
        f"{sorted(lineage_callers - lineage_demonstrations)} call the --follow reading. "
        "If any of them checks a landing claim, it will accept a pre-rename ancestor; "
        "use _first_commit_adding_current_path() instead."
    )
    # Both helpers must actually be in use, or this guard is guarding nothing.
    assert lineage_callers, "no test exercises the lineage reading any more"
    assert (
        "test_platforms_doc_7_18_11_incident_counts_are_eight_and_ninth"
        in landing_callers
    ), "the §7.18.11 doc guard must read landing commits, not lineage"
    assert (
        "test_device_loss_incident_records_landing_commits_match_git_history_issue_65"
        in landing_callers
    ), "the record guard must read landing commits, not lineage"


def test_platforms_doc_7_18_11_incident_counts_are_eight_and_ninth():
    """PR #65's review (Morpheus): docs/PLATFORMS.md §7.18.11 said 'Nine artifacts
    produced findings' and then contradicted its own sentence and its own table by
    saying 'Six of them are ... real' and 'The seventh' is derived — the table lists
    eight real-incident rows and one derived row. Locks the corrected wording so it
    cannot silently regress back to the miscount.

    Issue #24 item (f): a fourth instance survived that review — the paragraph heading
    *'Why "just add the seven names" was the wrong repair'*. Eight real artifacts plus
    the derived ninth is **nine** names, so that number is pinned here too, by phrase
    and not merely by absence."""
    text = (REPO_ROOT / "docs" / "PLATFORMS.md").read_text(encoding="utf-8")
    section_start = text.index("### 7.18.11")
    next_heading = text.index("\n## ", section_start)
    section = text[section_start:next_heading]

    assert "Six of them" not in section, "the eight/nine miscount must not come back"
    assert "The seventh" not in section, "the eight/nine miscount must not come back"
    assert "Eight of them are artifacts" in section
    assert "The ninth," in section
    assert "eight real incidents" in section.splitlines()[0]

    # Issue #24 (f). Positive and negative, because 'seven' merely being absent would
    # also be satisfied by deleting the sentence that carries the argument.
    assert 'just add the nine names' in section, (
        "§7.18.11's 'why not just add the names' paragraph must survive, and must "
        "count the same nine artifacts its own table lists"
    )
    names_phrase = re.search(r"just add the (\w+) names", section)
    assert names_phrase, "the 'just add the ... names' paragraph is gone entirely"
    assert names_phrase.group(1) == "nine", (
        f"§7.18.11 counts {names_phrase.group(1)!r} names, but its table is eight real "
        "artifacts plus the derived ninth"
    )
    assert "seven names" not in section, "the eight/nine miscount must not come back"

    # §7.18.3 is a *dated* reading of the negative control as it stood on 2026-08-02,
    # and §7.18.11 records the move to 27 arms on 2026-08-07. It is correct history and
    # explicitly not an instance of the miscount -- pinned so a future sweep for stale
    # numbers cannot 'fix' evidence that is not stale.
    s3_start = text.index("### 7.18.3")
    s3 = text[s3_start : text.index("\n### 7.18.4", s3_start)]
    assert "14 arms, all fired 2026-08-02" in s3, (
        "§7.18.3's dated 14-arm reading is historical evidence, not a stale count; "
        "the current 27-arm state is recorded separately in §7.18.11"
    )

    # And the two provenance corrections from the same review: the table's commit
    # citations must match real git history, not just internally-consistent prose.
    bothlanes_row = next(
        line for line in section.splitlines() if "device_loss_gate-BOTHLANES.json" in line
    )
    assert "1fc2ab6" in bothlanes_row
    assert "5353886" not in bothlanes_row
    kv_chain_row = next(
        line
        for line in section.splitlines()
        if "phi35_kv_chain-ctx4096-BOTH-dev0.json" in line
    )
    assert "0db65fe" in kv_chain_row
    assert "1fc2ab6" not in kv_chain_row

    assert _sha_claims_match(
        "1fc2ab6",
        _first_commit_adding_current_path(
            "bench/results/device_loss_gate-BOTHLANES.json"
        ),
    )
    assert _sha_claims_match(
        "0db65fe",
        _first_commit_adding_current_path(
            "bench/results/phi35_kv_chain-ctx4096-BOTH-dev0.json"
        ),
    )

    # Issue #24 (g): the row Morpheus said the helper would disagree with. §7.18.11
    # cites fb5f0b28 / 1f41095 -- the commits that introduced these paths. Under the
    # old --follow reading this assertion would have read b669eb9 / b262292 and the
    # doc would have looked wrong while being right. The complementary demonstration
    # that those citations do NOT match the lineage reading lives in
    # test_landing_reading_does_not_cross_renames_issue_24, so that this test -- a
    # doc-claim test -- never touches the lineage helper at all.
    devunset_row = next(
        line
        for line in section.splitlines()
        if "validation_phi35_probe-devunset-liveness.json" in line
    )
    for path, cited in (
        ("bench/results/validation_phi35_probe-devunset-liveness.json", "fb5f0b28"),
        ("bench/results/validation_phi35_counters-devunset-liveness.json", "1f41095"),
    ):
        assert cited in devunset_row, f"§7.18.11 no longer cites {cited}"
        assert _sha_claims_match(cited, _first_commit_adding_current_path(path)), (
            f"§7.18.11 cites {cited} for {path}, but the commit that introduced that "
            f"path is {_first_commit_adding_current_path(path)}"
        )



def test_ci_yml_device_loss_negative_control_comment_matches_real_arm_count():
    """PR #65's review (Morpheus): the *Device-loss screen negative control* step's
    comment in ci.yml still said '14 arms' after the tool it describes was raised to
    27. Reads the real arm count out of a live run of the tool rather than a second
    hand-typed number, so the comment and the tool cannot drift apart silently again."""
    ci_text = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    step_start = ci_text.index("Device-loss screen negative control (demand red)")
    window = ci_text[step_start : step_start + 400]
    m = re.search(r"(\d+) arms", window)
    assert m, "comment no longer states an arm count at all"
    documented_count = int(m.group(1))
    assert "14 arms" not in window

    proc = subprocess.run(
        [sys.executable, str(CI_DIR / "negative_control_device_loss.py")],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    out = proc.stdout + proc.stderr
    real_m = re.search(r"of (\d+) arms\.", out)
    assert real_m, f"could not read the real arm count from the tool's own output: {out!r}"
    assert documented_count == int(real_m.group(1)), (
        f"ci.yml's comment says {documented_count} arms but the tool itself ran "
        f"{real_m.group(1)} — the comment has drifted from the tool again"
    )


def test_ci_yml_windows_ledger_loss_probe_comment_is_not_stale():
    """PR #65's review (Morpheus, item c): the Windows ledger-loss-probe step's comment
    claimed 'This job's own actions/checkout@v4 above has no fetch-depth override (a
    1-commit shallow clone)' and 'without changing the faster shallow checkout' -- both
    false since the fetch-depth: 0 repair landed for issue #24. Asserts the stale claim
    is gone and the job's checkout really does declare fetch-depth: 0, so the comment
    and the checkout cannot silently diverge again."""
    ci_text = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    step_start = ci_text.index("Windows-namespace destination-policy regression")
    next_step = ci_text.index("\n      - name:", step_start + 1)
    step_body = ci_text[step_start:next_step]

    assert "no fetch-depth override" not in step_body
    assert "above has no fetch-depth override" not in step_body
    assert "without changing the faster shallow checkout" not in step_body

    # The job's own checkout, above this step, must actually declare fetch-depth: 0 --
    # otherwise the corrected comment would itself become the new stale claim.
    job_start = ci_text.rindex("\n  build-test-windows:\n", 0, step_start)
    checkout_start = ci_text.index("uses: actions/checkout@", job_start)
    checkout_window = ci_text[checkout_start : checkout_start + 150]
    assert "fetch-depth: 0" in checkout_window




# ──────────────────────────────────────────────────────────────────────────────
# check_ledger_portability — a run that proves nothing must not be a run that passes
#
# Added 2026-08-02 after building the EP on Linux (WSL Ubuntu 24.04) at d375a4d and
# pointing it at lavapipe. Every proof-ledger entry faulted, the session claimed 0/1
# nodes, all work went to the CPU EP, and the process exited 0. The op harness then
# reported that EP decision as "No Vulkan device available" and skipped 36 tests on a
# box where `epctl --probe-loader` had just printed gate PASS.
#
# The arms below use the two REAL artifacts wherever they exist, because a screen whose
# only inputs are ones I wrote is a screen I have only ever proved runs.
# ──────────────────────────────────────────────────────────────────────────────

LEDGER_SCREEN = "check_ledger_portability.py"

_FAULT = (
    '[vulkan-ep] WARN: [VulkanEP] proof ledger fault: ledger entry for '
    '"ai.onnx::Tanh/6+/f32>f32/ew_unary_tanh_f32/static/n1" was proven against shader '
    "digest 16a64dbeb2dbf63d but this build's modules hash to 8f87214a7ca41ca9."
)
_CLAIMS_NOTHING = "[§8.9.7] this session claims 0/1 nodes; all work runs on the CPU EP."
_CLAIMS_ONE = "session claims 1 proven form(s) [§8.9.7]: com.microsoft::MatMulNBits x1"
_ABSENT = (
    "SKIPPED [1] tests/ops/test_elementwise.py:231: No Vulkan device available — either "
    "no ICD is installed or all devices failed the capability gate."
)
_GATE_PASS = "Device 0: llvmpipe (LLVM 20.1.2, 256 bits) [Vulkan 1.4.318]  — gate PASS"

LIVE_LINUX_PROBE = REPO_ROOT / "bench" / "results" / "linux_lavapipe_probe.txt"
LIVE_LINUX_TESTS = REPO_ROOT / "bench" / "results" / "linux_lavapipe_optests.txt"
LIVE_LOADER = REPO_ROOT / "bench" / "results" / "linux_lavapipe_loader_probe.txt"
LIVE_WINDOWS = REPO_ROOT / "bench" / "results" / "windows_nvidia_probe_control.txt"


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_ledger_screen_reports_instrument_error_when_given_no_run():
    """'I was not given a run' is not 'the run was clean' (R12)."""
    proc = run_check(LEDGER_SCREEN)
    assert proc.returncode == EXIT_ERROR_INSTRUMENT, proc.stdout
    assert "no_run_named" in proc.stdout
    assert "PASS" not in proc.stdout.splitlines()[0]


def test_ledger_screen_reports_instrument_error_for_a_missing_artifact(tmp_path):
    proc = run_check(LEDGER_SCREEN, "--run-log", str(tmp_path / "nope.txt"))
    assert proc.returncode == EXIT_ERROR_INSTRUMENT, proc.stdout
    assert "artifact_unreadable" in proc.stdout
    assert "NOT a detection" in proc.stdout


def test_ledger_screen_goes_red_on_a_ledger_fault(tmp_path):
    p = _write(tmp_path, "run.txt", _FAULT + "\n" + _CLAIMS_NOTHING + "\n")
    proc = run_check(LEDGER_SCREEN, "--run-log", p, "--device-lane")
    assert proc.returncode == EXIT_FAIL_CONDITION, proc.stdout
    assert "FAIL(condition=ledger_fault)" in proc.stdout


def test_ledger_screen_quotes_the_failure_text_not_a_count(tmp_path):
    """R13: a detector reports the specimen, so a reader can check it."""
    p = _write(tmp_path, "run.txt", (_FAULT + "\n") * 9)
    proc = run_check(LEDGER_SCREEN, "--run-log", p, "--device-lane")
    assert proc.returncode == EXIT_FAIL_CONDITION
    assert "ew_unary_tanh_f32" in proc.stdout
    assert "9 " not in proc.stdout


def test_claims_nothing_is_unobservable_without_a_declared_device_lane(tmp_path):
    """A build-only or CPU-only run claims nothing correctly. Say so; do not fail it."""
    p = _write(tmp_path, "run.txt", _CLAIMS_NOTHING + "\n")
    proc = run_check(LEDGER_SCREEN, "--run-log", p)
    assert proc.returncode == EXIT_PASS, proc.stdout
    assert "UNOBSERVABLE" in proc.stdout
    assert "claimed_nothing" in proc.stdout


def test_claims_nothing_fires_once_the_lane_is_declared(tmp_path):
    p = _write(tmp_path, "run.txt", _CLAIMS_NOTHING + "\n")
    proc = run_check(LEDGER_SCREEN, "--run-log", p, "--device-lane")
    assert proc.returncode == EXIT_FAIL_CONDITION, proc.stdout
    assert "FAIL(condition=claimed_nothing)" in proc.stdout


def test_device_absence_is_unobservable_without_a_loader_artifact(tmp_path):
    """Without a gate PASS on record, an absent device may simply be absent."""
    p = _write(tmp_path, "run.txt", _ABSENT + "\n")
    proc = run_check(LEDGER_SCREEN, "--run-log", p, "--device-lane")
    assert proc.returncode == EXIT_PASS, proc.stdout
    assert "device_absence_misnamed" in proc.stdout
    assert "UNOBSERVABLE" in proc.stdout


def test_device_absence_fires_when_the_loader_gate_passed(tmp_path):
    """The two statements cannot both be true, and only one of them silences tests."""
    p = _write(tmp_path, "run.txt", _ABSENT + "\n")
    g = _write(tmp_path, "loader.txt", _GATE_PASS + "\n")
    proc = run_check(
        LEDGER_SCREEN, "--run-log", p, "--device-lane", "--loader-artifact", g
    )
    assert proc.returncode == EXIT_FAIL_CONDITION, proc.stdout
    assert "FAIL(condition=device_absence_misnamed)" in proc.stdout


def test_a_clean_device_run_stays_green_with_every_condition_armed(tmp_path):
    p = _write(tmp_path, "run.txt", _CLAIMS_ONE + "\n")
    g = _write(tmp_path, "loader.txt", _GATE_PASS + "\n")
    proc = run_check(
        LEDGER_SCREEN, "--run-log", p, "--device-lane", "--loader-artifact", g
    )
    assert proc.returncode == EXIT_PASS, proc.stdout


def test_the_screen_never_reads_an_exit_status(tmp_path):
    """The defect IS an exit status of 0.

    Accepting one as a filter would be accepting the defect as the filter. The run that
    produced the Linux artifact exited 0 with 151 lines of ledger faults in it.
    """
    src = (CI_DIR / LEDGER_SCREEN).read_text(encoding="utf-8")
    body = src.split('"""', 2)[-1]
    for forbidden in ("returncode", "check_call", "CalledProcessError"):
        assert forbidden not in body, (
            f"{LEDGER_SCREEN} refers to `{forbidden}` outside its docstring. This screen "
            "must not take a producing process's exit status as an input."
        )


@pytest.mark.skipif(
    not LIVE_LINUX_PROBE.exists() or not LIVE_LOADER.exists(),
    reason="live Linux/lavapipe artifacts absent from this checkout",
)
def test_live_red_arm_the_real_linux_lavapipe_run(tmp_path):
    """LIVE, not planted: the EP built on Linux, ran, claimed nothing, and exited 0."""
    proc = run_check(
        LEDGER_SCREEN,
        "--run-log",
        str(LIVE_LINUX_PROBE),
        "--device-lane",
        "--loader-artifact",
        str(LIVE_LOADER),
    )
    assert proc.returncode == EXIT_FAIL_CONDITION, proc.stdout
    assert "ledger_fault" in proc.stdout


@pytest.mark.skipif(
    not LIVE_LINUX_TESTS.exists() or not LIVE_LOADER.exists(),
    reason="live Linux/lavapipe artifacts absent from this checkout",
)
def test_live_red_arm_the_run_whose_summary_line_says_two_passed():
    """The artifact this fires on ends '2 passed, 36 skipped'. That is the whole point.

    Nothing in that summary looks like a failure. A lane reading it would call the
    op-correctness step green, and the step would have asserted nothing at all.
    """
    proc = run_check(
        LEDGER_SCREEN,
        "--run-log",
        str(LIVE_LINUX_TESTS),
        "--device-lane",
        "--loader-artifact",
        str(LIVE_LOADER),
    )
    assert proc.returncode == EXIT_FAIL_CONDITION, proc.stdout
    assert "device_absence_misnamed" in proc.stdout


@pytest.mark.skipif(
    not LIVE_WINDOWS.exists(), reason="live Windows control artifact absent"
)
def test_live_green_arm_the_real_windows_control():
    """Same commit, same ledger file, different glslc. Green.

    This is what makes the red arm a portability finding rather than a stale ledger:
    both artifacts were produced from one tree at one commit, minutes apart.
    """
    proc = run_check(LEDGER_SCREEN, "--run-log", str(LIVE_WINDOWS), "--device-lane")
    assert proc.returncode == EXIT_PASS, proc.stdout


def test_the_negative_control_counts_arms_by_provenance_not_as_a_total():
    """'11 passed' would hide that only 3 arms came from a run nobody staged."""
    proc = subprocess.run(
        [sys.executable, str(CI_DIR / "negative_control_ledger_portability.py")],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "LIVE" in proc.stdout and "PLANTED" in proc.stdout
    assert "only the LIVE arms are evidence" in proc.stdout


def test_gated_never_run_is_not_the_same_status_as_undemonstrated():
    """A check that never started and a check that never went red are different facts.

    `device.op_correctness` sat at UNDEMONSTRATED all session, which reads as "runs,
    never observed to fail" — one demonstration from green. In truth the Linux job dies
    at clippy and GitHub Actions skips the remaining seven steps, this one among them.
    """
    import importlib

    inv = importlib.import_module("lane_inventory")
    assert inv.GATED_NEVER_RUN not in inv.GREEN_STATUSES
    assert inv.GATED_NEVER_RUN in inv.RECORDED_GAP_STATUSES
    check = {c.id: c for c in inv.CHECKS}["device.op_correctness"]
    assert check.status == inv.GATED_NEVER_RUN
    assert check.observed is None, (
        "an `observed` date on a step that has never executed is the overclaim this "
        "status exists to remove"
    )


def test_the_ledger_device_provenance_blind_spot_is_recorded_and_not_substituted():
    """The loud half is screened; the silent half is not, and must not read as covered.

    Digest disagrees → form declined → CPU EP, which is always right: fails safe.
    Digest agrees, device never proven → form claimed: nothing watches this, and all 74
    entries were proven on device0 alone.
    """
    import importlib

    inv = importlib.import_module("lane_inventory")
    spot = {b.id: b for b in inv.BLIND_SPOTS}["ledger_device_provenance"]
    assert spot.substitute_status == inv.UNDEMONSTRATED
    assert "NONE" in spot.substitute
    assert "--reprove" in spot.substitute, (
        "the blind spot must carry the warning against per-platform re-proving, which "
        "is the obvious wrong fix and is also currently destructive"
    )


# ===========================================================================
# check_suite_productivity.py — a step that asserted nothing must not pass
#
# The two polarities are unusually cheap here because the check's whole input is
# text, so both arms are exact captured summary lines rather than paraphrases.
# ===========================================================================

SUITE_PRODUCTIVITY = CI_DIR / "check_suite_productivity.py"


def _productivity(*args: str) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, str(SUITE_PRODUCTIVITY), *args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    return proc.returncode, proc.stdout + proc.stderr


def _log(tmp_path: Path, text: str) -> str:
    p = tmp_path / "run.log"
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_a_productive_suite_passes_and_an_all_skipped_one_does_not(tmp_path):
    """The pair. Same step, same exit code from pytest, opposite verdicts here."""
    rc, out = _productivity(
        "--suite", "tests/ops", "--lane", "build-test-linux",
        _log(tmp_path, "50 failed, 272 passed, 343 skipped in 300.00s\n"),
    )
    assert rc == 0 and "SUITE-PRODUCTIVITY: PASS" in out

    rc, out = _productivity(
        "--suite", "tests/ops", "--lane", "build-test-linux",
        _log(tmp_path, "665 skipped in 300.00s\n"),
    )
    assert rc == 1, "an all-skipped suite exits ZERO from pytest and must not pass here"
    assert "FAIL(condition=asserted_nothing)" in out


def test_a_collection_error_is_a_hard_fail_and_quotes_the_line(tmp_path):
    """R13's second clause: no count without its text."""
    rc, out = _productivity(
        "--suite", "tests/ops",
        _log(
            tmp_path,
            "ERROR collecting tests/ops/test_shape_inference_delta.py\n"
            "!!!!!! Interrupted: 1 error during collection !!!!!!\n"
            "1 error in 4.90s\n",
        ),
    )
    assert rc == 1
    assert "FAIL(condition=collection_error)" in out
    assert "test_shape_inference_delta.py" in out, (
        "the failing file must be quoted; a count of collection errors is exactly the "
        "shape that let a NameError masquerade as a detection"
    )


def test_the_collected_floor_is_environment_independent_and_the_executed_floor_is_not(tmp_path):
    """Why there are two floors and not one.

    A full collection with a collapsed executed count is a different finding from a
    collection that shrank, and they have different causes.
    """
    rc, out = _productivity(
        "--suite", "tests/ops", "--lane", "build-test-linux",
        _log(tmp_path, "400 passed, 100 skipped in 30.00s\n"),
    )
    assert rc == 1 and "FAIL(condition=collected_below_floor)" in out

    rc, out = _productivity(
        "--suite", "tests/ops", "--lane", "build-test-linux",
        _log(tmp_path, "200 passed, 465 skipped in 30.00s\n"),
    )
    assert rc == 1 and "FAIL(condition=executed_below_floor)" in out


def test_an_unobserved_input_is_an_instrument_error_and_never_a_pass(tmp_path):
    """UNOBSERVABLE is not zero (R12), in all three of its forms here."""
    rc, out = _productivity("--suite", "tests/ops", str(tmp_path / "absent.log"))
    assert rc == 4 and "ERROR(instrument=log_not_captured)" in out

    rc, out = _productivity(
        "--suite", "tests/ops", _log(tmp_path, "ORT chatter and nothing else\n")
    )
    assert rc == 4 and "ERROR(instrument=summary_not_found)" in out

    rc, out = _productivity(
        "--suite", "tests/nowhere", _log(tmp_path, "665 passed in 30.00s\n")
    )
    assert rc == 4 and "ERROR(instrument=suite_has_no_floor)" in out, (
        "an unclassified suite must not pass by default — that is the state the "
        "op-correctness step lived in"
    )


def test_the_lane_marker_distinguishes_a_dead_lane_from_a_lost_log(tmp_path):
    """Borrowed from check_fatal_log, and for the same reason.

    A lane that died before the test step is already red; adding a second red about a
    subject that never existed makes the first one harder to read.
    """
    marker = tmp_path / ".lane-reached"
    rc, out = _productivity(
        "--suite", "tests/ops", f"--lane-marker={marker}", str(tmp_path / "absent.log")
    )
    assert rc == 0 and "lane_did_not_reach_evidence" in out

    marker.write_text("", encoding="utf-8")
    rc, out = _productivity(
        "--suite", "tests/ops", f"--lane-marker={marker}", str(tmp_path / "absent.log")
    )
    assert rc == 4 and "ERROR(instrument=log_not_captured)" in out


def test_libtest_zero_tests_is_the_same_defect_and_is_caught(tmp_path):
    """`cargo test` on an empty target prints `ok.` and exits 0. There is no --strict."""
    empty = (
        "running 0 tests\n\n"
        "test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out;"
        " finished in 0.00s\n"
    )
    rc, out = _productivity(
        "--suite", "cargo test --lib", "--harness", "libtest", _log(tmp_path, empty)
    )
    assert rc == 1 and "FAIL(condition=asserted_nothing)" in out

    healthy = (
        "running 510 tests\n\n"
        "test result: ok. 510 passed; 0 failed; 4 ignored; 0 measured; 0 filtered out;"
        " finished in 11.79s\n"
    )
    rc, out = _productivity(
        "--suite", "cargo test --lib", "--harness", "libtest", _log(tmp_path, healthy)
    )
    assert rc == 0 and "SUITE-PRODUCTIVITY: PASS" in out


def test_no_flag_can_lower_a_floor(tmp_path):
    """A floor that a command line can relax is a waiver with a flag.

    Asserted rather than commented, because the obvious way to unblock a red lane is to
    add exactly such a flag, and the person adding it will not read this docstring.
    """
    rc, _ = _productivity(
        "--suite", "tests/ops", "--min-collected", "1",
        _log(tmp_path, "10 passed in 1.00s\n"),
    )
    assert rc == 2, "argparse must reject it; lowering a floor is a tracked-file edit"


def test_every_floor_states_where_its_number_came_from():
    """A floor without provenance is a number somebody remembered."""
    data = json.loads((CI_DIR / "suite_floor.json").read_text(encoding="utf-8"))
    for name, entry in data["suites"].items():
        assert entry.get("provenance"), f"{name}: floor with no provenance"
        assert entry.get("min_collected") or entry.get("min_executed_by_lane"), (
            f"{name}: an entry with no floor at all passes everything"
        )


def test_the_shape_inference_module_no_longer_imports_its_optional_dep_at_module_scope():
    """The structural half of the dependency fix, asserted rather than trusted.

    The repair was to make the report lazy. Nothing stops a future edit from moving it
    back to module scope, and the symptom would be a directory-wide collection abort in
    an environment nobody on this team runs.
    """
    src = (REPO_ROOT / "tests" / "ops" / "test_shape_inference_delta.py").read_text(
        encoding="utf-8"
    )
    module_level_calls = [
        line
        for line in src.splitlines()
        if line.startswith(("_REPORT =", "_REPORT=")) and "_DeltaReport(" in line
    ]
    assert not module_level_calls, (
        "_REPORT must not be built at module scope: apply_shape_inference imports the "
        "optional onnx-shape-inference package, and at module scope that import runs "
        "during COLLECTION, taking all 665 tests in tests/ops with it"
    )
    assert "def _report()" in src, "the lazy accessor is the repair; keep it named"


# ===========================================================================
# ci/check_flake_witness.py and ci/check_build_precondition.py — two polarities
# each, plus the one bug a negative control found in a screen on its first run.
# ===========================================================================

FLAKE_WITNESS = CI_DIR / "check_flake_witness.py"
BUILD_PRECONDITION = CI_DIR / "check_build_precondition.py"


def _witness(*args: str) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, str(FLAKE_WITNESS), *args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    return proc.returncode, proc.stdout + proc.stderr


_LIBTEST_GREEN = (
    "     Running unittests src/lib.rs (target/debug/deps/x-1)\n"
    "\n"
    "running 2 tests\n"
    "test vk::barrier::tests::backend_probe_writes_legacy_token ... ok\n"
    "test ops::norm::tests::other ... ok\n"
    "\n"
    "test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out;"
    " finished in 0.10s\n"
)

_LIBTEST_RED = _LIBTEST_GREEN.replace(
    "test vk::barrier::tests::backend_probe_writes_legacy_token ... ok",
    "test vk::barrier::tests::backend_probe_writes_legacy_token ... FAILED",
).replace(
    "test result: ok. 2 passed; 0 failed;",
    "test result: FAILED. 1 passed; 1 failed;",
)


def test_one_id_failing_and_not_failing_at_one_commit_is_the_flake_and_two_commits_is_not(tmp_path):
    """The pair that IS the mechanism.

    Same two logs, same two outcomes, same test id. The only difference is whether the
    two observations sit at one commit or two, and that difference is exactly the
    difference between "it does that" and "your change broke it".
    """
    red = tmp_path / "red.log"
    red.write_text(_LIBTEST_RED, encoding="utf-8")
    green = tmp_path / "green.log"
    green.write_text(_LIBTEST_GREEN, encoding="utf-8")

    same = tmp_path / "same.jsonl"
    _witness("--harness", "libtest", "--suite", "lib", "--lane", "l", "--commit", "AAAA",
             "--run-id", "1", "--ledger", str(same), str(red))
    rc, out = _witness("--harness", "libtest", "--suite", "lib", "--lane", "l",
                       "--commit", "AAAA", "--run-id", "2", "--ledger", str(same), str(green))
    assert rc == 1, "both polarities at ONE commit is an intermittent and must be red"
    assert "FAIL(condition=intermittent)" in out
    assert "backend_probe_writes_legacy_token" in out, "a red with no subject is the defect"
    assert "THE COMMIT IS EXONERATED AND THE TEST IS NOT" in out

    across = tmp_path / "across.jsonl"
    _witness("--harness", "libtest", "--suite", "lib", "--lane", "l", "--commit", "AAAA",
             "--run-id", "1", "--ledger", str(across), str(red))
    rc, out = _witness("--harness", "libtest", "--suite", "lib", "--lane", "l",
                       "--commit", "BBBB", "--run-id", "2", "--ledger", str(across), str(green))
    assert rc == 0, "the same two outcomes at TWO commits is a regression that got fixed"
    assert "FLAKE-WITNESS: PASS" in out


def test_the_failing_name_is_printed_last_so_a_truncated_head_cannot_lose_it(tmp_path):
    """The coordinator's actual failure: one red, six greens, and no name."""
    red = tmp_path / "red.log"
    red.write_text(_LIBTEST_RED, encoding="utf-8")
    rc, out = _witness("--harness", "libtest", "--suite", "lib", "--lane", "linux",
                       "--commit", "CCCC", "--ledger", str(tmp_path / "l.jsonl"), str(red))
    assert rc == 0
    tail = out.strip().splitlines()[-6:]
    assert any("backend_probe_writes_legacy_token" in line for line in tail), (
        "the name must be in the TAIL. GitHub truncates the middle of a long step log, "
        "and a name that does not survive the transport is a red with no subject."
    )


def test_an_annotation_is_emitted_where_log_truncation_cannot_reach_it(tmp_path, monkeypatch):
    """::error:: lines become check-run annotations, which are not log bytes."""
    red = tmp_path / "red.log"
    red.write_text(_LIBTEST_RED, encoding="utf-8")
    env = dict(os.environ, FLAKE_WITNESS_FORCE_ANNOTATE="1")
    proc = subprocess.run(
        [sys.executable, str(FLAKE_WITNESS), "--harness", "libtest", "--suite", "lib",
         "--lane", "linux", "--commit", "DDDD", "--ledger", str(tmp_path / "l.jsonl"), str(red)],
        capture_output=True, text=True, cwd=str(REPO_ROOT), env=env,
    )
    assert "::error title=FAILED on linux::" in proc.stdout
    assert "backend_probe_writes_legacy_token" in proc.stdout


def test_a_not_failed_from_a_much_smaller_run_is_incomparable_and_not_a_flake(tmp_path):
    """NOT_FAILED includes skipped. A test that stopped running is a different defect."""
    tiny = tmp_path / "tiny.log"
    tiny.write_text(
        "     Running unittests src/lib.rs (target/debug/deps/x-1)\n\n"
        "running 1 test\n"
        "test vk::barrier::tests::backend_probe_writes_legacy_token ... FAILED\n\n"
        "test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 500 filtered out;"
        " finished in 0.01s\n",
        encoding="utf-8",
    )
    green = tmp_path / "green.log"
    green.write_text(_LIBTEST_GREEN, encoding="utf-8")
    led = tmp_path / "l.jsonl"
    _witness("--harness", "libtest", "--suite", "lib", "--lane", "l", "--commit", "EEEE",
             "--run-id", "1", "--ledger", str(led), str(tiny))
    rc, out = _witness("--harness", "libtest", "--suite", "lib", "--lane", "l",
                       "--commit", "EEEE", "--run-id", "2", "--ledger", str(led), str(green))
    assert rc == 0
    assert "INCOMPARABLE" in out
    assert "check_suite_productivity" in out, "it must name whose defect class that is"


def test_a_log_it_could_not_parse_is_unobservable_and_never_a_pass(tmp_path):
    """The repo's standing rule: UNOBSERVABLE is not zero."""
    p = tmp_path / "nothing.log"
    p.write_text("collecting ...\ntests/ops/test_x.py ....\n", encoding="utf-8")
    rc, out = _witness("--harness", "pytest", "--suite", "ops",
                       "--ledger", str(tmp_path / "l.jsonl"), str(p))
    assert rc == 4
    assert "ERROR(instrument=log_unparsed)" in out


def test_a_join_over_too_few_runs_refuses_a_verdict_rather_than_giving_a_green(tmp_path):
    """A 1-in-40 is invisible in one run BY CONSTRUCTION; a green from one run is noise."""
    green = tmp_path / "green.log"
    green.write_text(_LIBTEST_GREEN, encoding="utf-8")
    rc, out = _witness("--harness", "libtest", "--suite", "lib", "--lane", "l",
                       "--commit", "FFFF", "--ledger", str(tmp_path / "l.jsonl"),
                       "--require-history", "5", str(green))
    assert rc == 4
    assert "ERROR(instrument=history_too_short)" in out


def test_the_flake_witness_names_no_counts_in_its_ledger(tmp_path):
    """R13: no count without its text. The ledger's key is a NAME, never a number.

    EXTENDED 2026-08-05 (trinity), when the ledger gained a whole-run marker so a green
    run leaves a trace and the pytest complement can be reconstructed. The marker is a
    record with no test behind it, which is exactly the shape this arm exists to refuse —
    so it is filed under a NAME (`RUN_SEEN_ID`), and this arm now says so out loud rather
    than being quietly satisfied by it.
    """
    red = tmp_path / "red.log"
    red.write_text(_LIBTEST_RED, encoding="utf-8")
    led = tmp_path / "l.jsonl"
    _witness("--harness", "libtest", "--suite", "lib", "--lane", "l", "--commit", "GGGG",
             "--ledger", str(led), str(red))
    records = [json.loads(line) for line in led.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert records, "the ledger must be written"
    assert all(r["test_id"] for r in records)
    assert not any(str(r["test_id"]).strip().isdigit() for r in records)
    assert any(r["outcome"] == "FAILED" for r in records)
    markers = [r for r in records if r["outcome"] == "RUN_SEEN"]
    assert len(markers) == 1, "one parsed log, one run marker"
    assert markers[0]["test_id"].startswith("<"), markers[0]


def _precondition(*args: str) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, str(BUILD_PRECONDITION), *args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    return proc.returncode, proc.stdout + proc.stderr


def test_build_precondition_is_green_on_this_tree_and_red_on_the_bytes_it_was_written_for(tmp_path):
    """The pair, with the red arm REPLAYED out of this repository's own history."""
    rc, out = _precondition(
        str(REPO_ROOT / ".github" / "workflows" / "ci.yml"),
        str(REPO_ROOT / ".github" / "workflows" / "conformance.yml"),
    )
    assert rc == 0, out
    assert "BUILD-PRECONDITION: PASS" in out

    bad = tmp_path / "bad.yml"
    bad.write_text(
        "jobs:\n"
        "  x:\n"
        "    steps:\n"
        "      - name: Build the thing\n"
        "        run: |\n"
        "          if [ ! -f rust/Cargo.toml ]; then\n"
        "            echo \"BUILD_SKIPPED=1\" >> $GITHUB_ENV\n"
        "            exit 0\n"
        "          fi\n"
        "          cargo build --release\n"
        "      - name: Test\n"
        "        if: env.BUILD_SKIPPED != '1'\n"
        "        run: cargo test\n",
        encoding="utf-8",
    )
    rc, out = _precondition(str(bad))
    assert rc == 1, "one missing tracked file must not be able to turn a lane green"
    assert "FAIL(condition=skip_flag_with_exit_zero)" in out


def test_the_build_precondition_screen_prints_its_condition_token_when_it_goes_red(tmp_path):
    """The bug its own negative control caught on the control's FIRST run.

    ``screen()`` returned exit 1 and never printed ``FAIL(condition=...)``: ``_fail()``
    existed and was never called. A red step with no condition name is precisely what the
    R13 vocabulary exists to prevent, and the screen enforcing it had the defect.
    """
    bad = tmp_path / "dead.yml"
    bad.write_text(
        "jobs:\n"
        "  x:\n"
        "    steps:\n"
        "      - name: Test\n"
        "        if: env.NOBODY_WRITES_THIS != '1'\n"
        "        run: cargo test\n",
        encoding="utf-8",
    )
    rc, out = _precondition(str(bad))
    assert rc == 1
    assert re.search(r"FAIL\(condition=\w+\)", out), (
        "a non-zero exit with no R13 condition token is a red with no subject"
    )
    assert "dead_guard" in out


def test_a_dormant_guard_is_not_inert_and_the_screen_says_why(tmp_path):
    """My own 2026-08-02 decision, corrected by the screen on its first run over this tree."""
    bad = tmp_path / "dormant.yml"
    bad.write_text(
        "jobs:\n"
        "  x:\n"
        "    steps:\n"
        "      - name: Test\n"
        "        if: env.BUILD_SKIPPED != '1'\n"
        "        run: cargo test\n",
        encoding="utf-8",
    )
    rc, out = _precondition(str(bad))
    assert rc == 1
    assert "dead_guard" in out
    assert "BUILD_SKIPPED" in out


# ---------------------------------------------------------------------------
# ci/negative_control_build_precondition.py's REPLAYED arms — is 607056a actually
# reachable from THIS checkout, and do BOTH arms pinned to it actually execute?
# Issue #1: `actions/checkout@v4` defaults to a depth-1 shallow clone, which makes
# `git show 607056a:<path>` fail, and the negative control correctly counts that as a
# failed arm (UNOBSERVABLE, never a silent pass) rather than skipping it — but a
# checkout that cannot reach its own history turns that correct behaviour into a
# permanent, uninformative red. The fix is `fetch-depth: 0` on the `lane-checks` job's
# checkout step; these two tests are the positive/negative pair that proves the fix is
# real rather than assumed: one shows the subject is reachable in a normal (unshallowed)
# checkout and both REPLAYED arms fire, the other reproduces the CI defect in a scratch
# shallow clone and requires the control to still report it loudly, never green.
#
# NOTE ON THE FIRST CUT OF THIS TEST (fixed after a real CI red, PR #6): it called the
# `_precondition()` helper above, which runs `check_build_precondition.py` — the
# production screen `negative_control_build_precondition.py` is a meta-test *of*. That
# produces a `BUILD-PRECONDITION: PASS` line, never the `NEGATIVE-CONTROL: ... N LIVE /
# M REPLAYED / K PLANTED` summary this test actually needs, so the count assertion could
# never pass no matter how many REPLAYED arms fired. The fix is to invoke
# `negative_control_build_precondition.py` itself (it takes no CLI arguments — see
# ``main()``), and to assert the REPLAYED arm *count* parsed out of its own summary line
# rather than a full literal string, so a change to the number of PLANTED arms (which
# says nothing about REPLAYED reachability) cannot make this test's real subject —
# "did both REPLAYED arms execute" — silently untestable again.
# ---------------------------------------------------------------------------

def _run_negative_control_build_precondition(repo_root: Path) -> tuple[int, str]:
    """Invoke ``<repo_root>/ci/negative_control_build_precondition.py`` (no CLI args).

    The script computes its own ``REPO_ROOT`` from ``Path(__file__).resolve().parent.
    parent`` -- it does NOT read the process's current working directory -- so running
    it against a *different* checkout (e.g. the scratch shallow clone below) means
    invoking the copy of the script that lives inside that checkout, not this
    worktree's copy with a different ``cwd``. Passing the wrong ``repo_root`` here would
    silently re-run this worktree's own (unshallowed) script and could never observe the
    shallow clone's REPLAYED arm at all -- the exact defect this helper exists to avoid.
    """
    script = repo_root / "ci" / "negative_control_build_precondition.py"
    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True, text=True, cwd=str(repo_root),
    )
    return proc.returncode, proc.stdout + proc.stderr


def test_the_replayed_subjects_are_reachable_from_this_checkout():
    """POSITIVE arm: this checkout can `git show` every commit a REPLAYED arm depends on,
    and both of ``negative_control_build_precondition.py``'s own REPLAYED arms execute.

    If the ``git show`` loop below is red, no REPLAYED arm anywhere in ci/ can fire here
    -- a shallow clone is exactly the shape that makes `git show <ref>:<path>` fail,
    which is indistinguishable, one arm at a time, from the defect itself having gone
    missing. The count assertion below is the control's *own* declared inventory, not a
    number chosen here: ``negative_control_build_precondition.py`` appends exactly one
    REPLAYED arm when its historical ref is unreadable (UNOBSERVABLE) and exactly two
    when it is readable (the exact historical bytes, and that BP1 -- not BP2 -- is what
    catches them). Asserting `== 2` is asserting "every declared REPLAYED arm for this
    control actually executed", the real invariant issue #1 asks for, not a guess.
    """
    for ref, rel_path in (
        ("607056a", ".github/workflows/ci.yml"),  # negative_control_build_precondition.py
        ("133b9fe", ".github/workflows/ci.yml"),  # negative_control_open_reds.py
        ("8a851f8", "README.md"),  # negative_control_readme_usage.py
        ("eb84364", "evidence/proof_ledger.jsonl"),  # negative_control_ledger_census.py
        ("26fd93f", "evidence/proof_ledger.jsonl"),  # negative_control_ledger_census.py
        ("ea427fd", "bench/exec_census.py"),  # negative_control_hardcoded_foundry_paths.py
    ):
        proc = subprocess.run(
            ["git", "show", f"{ref}:{rel_path}"],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
        assert proc.returncode == 0, (
            f"`git show {ref}:{rel_path}` failed in this checkout -- every REPLAYED arm "
            f"pinned to {ref} is UNOBSERVABLE here, not merely untested.\n"
            f"stderr: {proc.stderr.strip()}"
        )

    rc, out = _run_negative_control_build_precondition(REPO_ROOT)
    assert rc == 0, out
    assert "NEGATIVE-CONTROL: PASS" in out, (
        "every declared arm must fire once its historical refs are reachable\n" + out
    )
    counts = re.search(r"(\d+) LIVE / (\d+) REPLAYED / (\d+) PLANTED", out)
    assert counts is not None, f"could not find the control's own arm-count line:\n{out}"
    replayed = int(counts.group(2))
    assert replayed == 2, (
        f"only {replayed} of the control's 2 declared REPLAYED arms fired -- a checkout "
        "that can reach 607056a but still reports fewer than both REPLAYED arms has "
        f"regressed silently, not gone shallow.\n{out}"
    )


def test_negative_control_reports_unobservable_not_a_pass_on_a_shallow_clone(tmp_path):
    """NEGATIVE arm: reproduce the actual CI defect in a scratch clone, on purpose.

    A `--depth 1` clone of this repository cannot reach 607056a (it long predates HEAD),
    so the REPLAYED arm must FAIL loudly -- UNOBSERVABLE, per
    ``negative_control_build_precondition.historical_workflow`` -- and the control's exit
    code must stay non-zero. It must NOT skip the arm and it must NOT report PASS: a
    negative control that goes quietly green the moment its subject becomes unreachable
    is indistinguishable, from the outside, from one that was never wired at all.
    """
    shallow = tmp_path / "shallow"
    clone = subprocess.run(
        ["git", "clone", "-q", "--depth", "1", "--no-local",
         REPO_ROOT.as_uri(), str(shallow)],
        capture_output=True, text=True,
    )
    assert clone.returncode == 0, f"scratch shallow clone failed: {clone.stderr}"

    show = subprocess.run(
        ["git", "show", "607056a:.github/workflows/ci.yml"],
        capture_output=True, text=True, cwd=str(shallow),
    )
    assert show.returncode != 0, (
        "a depth-1 clone that CAN reach 607056a does not reproduce the CI defect this "
        "test exists to replay -- widen the clone or pick a more recent HISTORICAL_REF"
    )

    rc, out = _run_negative_control_build_precondition(shallow)
    assert rc == 1, (
        f"a shallow checkout must fail the control, not pass it silently.\nOutput:\n{out}"
    )
    assert "UNOBSERVABLE" in out, "the reason must be named, not just a bare non-zero exit"
    assert "NEGATIVE-CONTROL: FAIL(condition=arm_did_not_fire)" in out
    assert "NEGATIVE-CONTROL: PASS" not in out, (
        "an unreachable REPLAYED subject must never be reported as every arm firing"
    )


def _run_negative_control_build_precondition_with_env(extra_env: dict) -> tuple[int, str]:
    """Like ``_run_negative_control_build_precondition``, but with a caller-controlled
    parent environment instead of this test process's own inherited one.

    ``PYTHONIOENCODING``, ``PYTHONUTF8`` and ``PYTHONLEGACYWINDOWSSTDIO`` are stripped
    from the base environment before ``extra_env`` is applied, so a value this pytest
    process happens to have picked up (from a CI job env, a dev shell profile, etc.)
    cannot quietly stand in for the polarity a given call is trying to exercise.
    """
    env = dict(os.environ)
    for var in ("PYTHONIOENCODING", "PYTHONUTF8", "PYTHONLEGACYWINDOWSSTDIO"):
        env.pop(var, None)
    env.update(extra_env)
    script = REPO_ROOT / "ci" / "negative_control_build_precondition.py"
    proc = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True,
        cwd=str(REPO_ROOT), env=env,
    )
    return proc.returncode, proc.stdout + proc.stderr


def test_negative_control_build_precondition_passes_under_both_encoding_polarities():
    """``run()``'s child-encoding pin must make the control's own PASS independent of
    the *parent* shell that launched ``negative_control_build_precondition.py``, not
    merely reproduce whatever this dev machine already has configured.

    Before this fix, only the PARENT-side decode was pinned (``encoding="utf-8"`` on
    ``subprocess.run``); the CHILD (``check_build_precondition.py``) still picked its
    own stdout encoding from ``locale.getpreferredencoding()``, so which BP arm's
    assertion tripped depended on whatever the invoking shell had (or had not) set --
    exactly the "parent-only pin flips which BP arm fails depending on shell" defect.
    ``run()`` now forces ``PYTHONIOENCODING=utf-8`` into the CHILD's own environment
    unconditionally, so both polarities below must PASS identically:

    * a stock/legacy parent env -- neither ``PYTHONIOENCODING`` nor ``PYTHONUTF8`` set,
      the shape of a default Windows shell (cp1252-preferred-encoding) and, on POSIX,
      forced further with ``LC_ALL=C``/``LANG=C`` to remove the usual UTF-8 locale
      safety net so the same "nothing declares UTF-8" condition is reproduced there too;
    * an explicit UTF-8 parent env -- proving the fix isn't a coincidence of whichever
      polarity this box already happened to have.
    """
    polarities = {
        "stock/legacy (no PYTHONIOENCODING/PYTHONUTF8, C locale)": {
            "LC_ALL": "C", "LANG": "C",
        },
        "explicit UTF-8": {
            "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "LC_ALL": "en_US.UTF-8",
        },
    }
    for label, extra_env in polarities.items():
        rc, out = _run_negative_control_build_precondition_with_env(extra_env)
        assert rc == 0, f"[{label}] must PASS regardless of the parent shell's own encoding:\n{out}"
        assert "NEGATIVE-CONTROL: PASS" in out, f"[{label}]:\n{out}"
        # This is the one arm whose `ok` is computed from `"BP1 \u2014" in out` inside
        # negative_control_build_precondition.py's own check_build_precondition.py
        # subprocess call -- the exact literal substring match the mojibake bug broke.
        # The raw captured text isn't echoed verbatim by the outer script, so the arm's
        # own printed verdict ("ok " vs "FAIL") is the observable proxy for whether that
        # decode round-tripped correctly under this parent env.
        assert "[REPLAYED] ok    and it is BP1 that catches it, not BP2 by accident" in out, (
            f"[{label}] the em-dash-dependent REPLAYED arm did not fire cleanly -- the "
            f"child-encoding pin did not hold under this parent env:\n{out}"
        )


# ===========================================================================
# ci/check_powershell_exit_status.py -- issue #49. A Windows `run:` step captures
# $LASTEXITCODE into a variable to brand its own verdict, and its success path ends
# without an explicit `exit`, so GitHub's implicit pwsh wrapper reads whatever native
# command ran LAST -- the gate's deliberately non-zero exit, never the step's own
# printed PASS. Two polarities plus the negative control's own two-polarity coverage.
# ===========================================================================

POWERSHELL_EXIT_STATUS = CI_DIR / "check_powershell_exit_status.py"


def _pwsh_exit(*args: str) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, str(POWERSHELL_EXIT_STATUS), *args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    return proc.returncode, proc.stdout + proc.stderr


def test_powershell_exit_status_is_green_on_this_tree(tmp_path):
    """The pair: this repository's own workflows are clean after the issue #49 fix."""
    rc, out = _pwsh_exit(
        str(REPO_ROOT / ".github" / "workflows" / "ci.yml"),
        str(REPO_ROOT / ".github" / "workflows" / "conformance.yml"),
    )
    assert rc == 0, out
    assert "POWERSHELL-EXIT-STATUS: PASS" in out


def test_powershell_exit_status_reds_on_the_real_issue_49_shape(tmp_path):
    """A `$code = $LASTEXITCODE` capture whose success path falls through to a bare
    `Write-Host` -- the exact shape both Windows 'Gate negative control' steps carried
    until issue #49 -- must be caught, and it must name the R13 condition token.
    """
    bad = tmp_path / "stale.yml"
    bad.write_text(
        "jobs:\n"
        "  build-test-windows:\n"
        "    steps:\n"
        "      - name: Gate negative control -- a declined artifact must produce UNATTRIBUTED\n"
        "        run: |\n"
        "          $out = python ci\\gate_chain_fp32.py --artifact decline_probe 2>&1 | Out-String\n"
        "          $code = $LASTEXITCODE\n"
        "          Write-Host $out\n"
        "          if ($code -eq 0) {\n"
        "            Write-Error \"NEGATIVE CONTROL FAILED\"\n"
        "            exit 1\n"
        "          }\n"
        "          Write-Host \"NEGATIVE CONTROL PASSED (loader untouched): exited $code.\"\n",
        encoding="utf-8",
    )
    rc, out = _pwsh_exit(str(bad))
    assert rc == 1, "a step whose own printed verdict is PASS must not silently exit red-only"
    assert "FAIL(condition=stale_exit_code_after_native_capture)" in out
    assert "Gate negative control" in out


def test_powershell_exit_status_clears_on_an_explicit_exit_0(tmp_path):
    """The actual fix applied for issue #49: an explicit `exit 0` on the success path."""
    fixed = tmp_path / "fixed.yml"
    fixed.write_text(
        "jobs:\n"
        "  build-test-windows:\n"
        "    steps:\n"
        "      - name: Gate negative control -- a declined artifact must produce UNATTRIBUTED\n"
        "        run: |\n"
        "          $out = python ci\\gate_chain_fp32.py --artifact decline_probe 2>&1 | Out-String\n"
        "          $code = $LASTEXITCODE\n"
        "          Write-Host $out\n"
        "          if ($code -eq 0) {\n"
        "            Write-Error \"NEGATIVE CONTROL FAILED\"\n"
        "            exit 1\n"
        "          }\n"
        "          Write-Host \"NEGATIVE CONTROL PASSED (loader untouched): exited $code.\"\n"
        "          exit 0\n",
        encoding="utf-8",
    )
    rc, out = _pwsh_exit(str(fixed))
    assert rc == 0, out
    assert "POWERSHELL-EXIT-STATUS: PASS" in out


def test_powershell_exit_status_does_not_flag_a_step_that_never_captures_the_code(tmp_path):
    """A step whose implicit exit already carries the last real command's own status
    (nothing has second-guessed it) is out of scope: flagging it would be noise.
    """
    clean = tmp_path / "no-capture.yml"
    clean.write_text(
        "jobs:\n"
        "  build-test-windows:\n"
        "    steps:\n"
        "      - name: Compile all targets\n"
        "        run: |\n"
        "          cargo check --release --manifest-path rust\\Cargo.toml --all-targets\n"
        "          exit $LASTEXITCODE\n",
        encoding="utf-8",
    )
    rc, out = _pwsh_exit(str(clean))
    assert rc == 0, out


def test_powershell_exit_status_prints_its_condition_token_on_a_planted_arm(tmp_path):
    """R13 discipline for this new screen, checked directly rather than only through the
    negative control -- a red step with no condition token is a red with no subject.
    """
    bad = tmp_path / "dead.yml"
    bad.write_text(
        "jobs:\n"
        "  x:\n"
        "    steps:\n"
        "      - name: Some verdict step\n"
        "        run: |\n"
        "          python some_gate.py\n"
        "          $rc = $LASTEXITCODE\n"
        "          Write-Host \"done, rc=$rc\"\n",
        encoding="utf-8",
    )
    rc, out = _pwsh_exit(str(bad))
    assert rc == 1
    assert re.search(r"FAIL\(condition=\w+\)", out), (
        "a non-zero exit with no R13 condition token is a red with no subject"
    )


def test_powershell_exit_status_workflow_not_found_is_an_instrument_error(tmp_path):
    rc, out = _pwsh_exit(str(tmp_path / "does-not-exist.yml"))
    assert rc == 4
    assert "ERROR(instrument=workflow_not_found)" in out


def test_powershell_exit_status_no_args_is_usage_not_a_pass():
    rc, out = _pwsh_exit()
    assert rc == 2


# ---------------------------------------------------------------------------
# PS2 -- issue #55. The capture was never what made the verdict lie: a step that
# invokes a native command anywhere in its body (the call operator, a bare tool name,
# or a bare `*.exe`) and ends its script on a `Write-Host` verdict print with no
# intervening `exit` is relying on the SAME un-consumed implicit wrapper as PS1, just
# without ever having named a variable for it. "Install Mesa lavapipe" carried this
# shape and PS1's `$name = $LASTEXITCODE` regex never reached it.
# ---------------------------------------------------------------------------

def test_powershell_exit_status_reds_on_the_real_issue_55_shape(tmp_path):
    """The real 'Install Mesa lavapipe' shape: a native call (`&`), no capture into a
    named variable, and a final Write-Host verdict with no `exit` -- must be caught
    under the PS2 condition token, distinct from PS1's.
    """
    bad = tmp_path / "lavapipe-like.yml"
    bad.write_text(
        "jobs:\n"
        "  build-test-windows:\n"
        "    steps:\n"
        "      - name: Install Mesa lavapipe (mesa-dist-win)\n"
        "        run: |\n"
        "          $vulkInfo = & \"$env:VULKAN_SDK\\Bin\\vulkaninfoSDK.exe\" --summary 2>&1\n"
        "          Write-Host $vulkInfo\n"
        "          if (-not ($vulkInfo | Select-String -Quiet -SimpleMatch \"llvmpipe\")) {\n"
        "            Write-Error \"lavapipe (llvmpipe) not found\"\n"
        "            exit 1\n"
        "          }\n"
        "          Write-Host \"lavapipe smoke-check: OK (llvmpipe enumerated)\"\n",
        encoding="utf-8",
    )
    rc, out = _pwsh_exit(str(bad))
    assert rc == 1, "a step whose own printed verdict is OK must not silently exit on a stale native code"
    assert "FAIL(condition=native_exit_stale_at_verdict_print)" in out
    assert "stale_exit_code_after_native_capture" not in out, (
        "PS1's token must not fire here -- no `$name = $LASTEXITCODE` capture exists "
        "in this step; only PS2's wider, capture-free rule should"
    )
    assert "Install Mesa lavapipe" in out


def test_powershell_exit_status_ps2_clears_on_an_explicit_exit_0(tmp_path):
    """The actual fix applied for issue #55: an explicit `exit 0` on the success path
    of a step that has no $LASTEXITCODE capture at all.
    """
    fixed = tmp_path / "lavapipe-fixed.yml"
    fixed.write_text(
        "jobs:\n"
        "  build-test-windows:\n"
        "    steps:\n"
        "      - name: Install Mesa lavapipe (mesa-dist-win)\n"
        "        run: |\n"
        "          $vulkInfo = & \"$env:VULKAN_SDK\\Bin\\vulkaninfoSDK.exe\" --summary 2>&1\n"
        "          Write-Host $vulkInfo\n"
        "          if (-not ($vulkInfo | Select-String -Quiet -SimpleMatch \"llvmpipe\")) {\n"
        "            Write-Error \"lavapipe (llvmpipe) not found\"\n"
        "            exit 1\n"
        "          }\n"
        "          Write-Host \"lavapipe smoke-check: OK (llvmpipe enumerated)\"\n"
        "          exit 0\n",
        encoding="utf-8",
    )
    rc, out = _pwsh_exit(str(fixed))
    assert rc == 0, out
    assert "POWERSHELL-EXIT-STATUS: PASS" in out


def test_powershell_exit_status_ps2_reaches_a_bare_exe_invocation(tmp_path):
    """PS2's native-call detection is not limited to the `&` call operator: a bare,
    unquoted `*.exe` path invoked directly also sets $LASTEXITCODE.
    """
    bad = tmp_path / "bare-exe.yml"
    bad.write_text(
        "jobs:\n"
        "  build-test-windows:\n"
        "    steps:\n"
        "      - name: Probe something\n"
        "        run: |\n"
        "          rust\\target\\release\\probeCtl.exe --probe-loader\n"
        "          Write-Host \"probe finished\"\n",
        encoding="utf-8",
    )
    rc, out = _pwsh_exit(str(bad))
    assert rc == 1
    assert "FAIL(condition=native_exit_stale_at_verdict_print)" in out


def test_powershell_exit_status_ps2_does_not_flag_an_exe_substring_inside_a_quoted_url(tmp_path):
    """Regression guard: a `.exe` substring appearing only inside a quoted URL/path
    VALUE -- never invoked as a command -- must not trip PS2. An early draft of PS2's
    native-call regex matched this shape on 'Install LunarG Vulkan SDK'-like steps
    (`$url = "https://.../Installer.exe"`, downloaded with `Invoke-WebRequest` and run
    via `Start-Process`, neither of which sets $LASTEXITCODE).
    """
    clean = tmp_path / "url-string.yml"
    clean.write_text(
        "jobs:\n"
        "  build-test-windows:\n"
        "    steps:\n"
        "      - name: Install some other SDK\n"
        "        run: |\n"
        "          $url = \"https://example.invalid/download/SomeSdk-Installer.exe\"\n"
        "          Write-Host \"Downloading SDK from $url\"\n"
        "          Invoke-WebRequest -Uri $url -OutFile \"$env:TEMP\\SomeSdk.exe\"\n"
        "          Start-Process -Wait -FilePath \"$env:TEMP\\SomeSdk.exe\" -ArgumentList '/quiet'\n"
        "          Write-Host \"SDK installed\"\n",
        encoding="utf-8",
    )
    rc, out = _pwsh_exit(str(clean))
    assert rc == 0, out
    assert "POWERSHELL-EXIT-STATUS: PASS" in out


def test_powershell_exit_status_ps2_does_not_flag_a_verdict_with_no_native_call(tmp_path):
    """A Write-Host verdict with no native call anywhere earlier has nothing for the
    implicit trailer to disagree with -- cmdlets (Get-ChildItem, New-Item, ...) never
    set $LASTEXITCODE.
    """
    clean = tmp_path / "no-native.yml"
    clean.write_text(
        "jobs:\n"
        "  build-test-windows:\n"
        "    steps:\n"
        "      - name: Pure cmdlet step\n"
        "        run: |\n"
        "          $items = Get-ChildItem -Recurse -Path \"C:\\some\\dir\"\n"
        "          New-Item -ItemType Directory -Force -Path \"C:\\some\\other\" | Out-Null\n"
        "          Write-Host \"housekeeping complete: $($items.Count) item(s)\"\n",
        encoding="utf-8",
    )
    rc, out = _pwsh_exit(str(clean))
    assert rc == 0, out


NEGATIVE_CONTROL_POWERSHELL_EXIT_STATUS = CI_DIR / "negative_control_powershell_exit_status.py"


def test_negative_control_powershell_exit_status_all_arms_fire():
    """The production negative control itself -- every LIVE/REPLAYED/PLANTED arm must
    fire as declared, the same discipline `negative_control_build_precondition.py` is
    held to.
    """
    proc = subprocess.run(
        [sys.executable, str(NEGATIVE_CONTROL_POWERSHELL_EXIT_STATUS)],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(REPO_ROOT),
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    assert "NEGATIVE-CONTROL: PASS" in out
    assert "REPLAYED" in out and "PLANTED" in out and "LIVE" in out


# ---------------------------------------------------------------------------
# ci/negative_control_open_reds.py's _live_failure_note() -- the helper that anchors a
# failing LIVE arm's diagnostic on check_open_reds.py's own state-table header instead
# of a blind [-1500:] stdout tail (see its own docstring for the CI-log-truncation bug
# this replaced). Two polarities: the marker is present (anchor on it, don't blindly
# truncate before it), and the marker is absent/malformed (fall back to a bounded tail
# that still carries content -- fail-loud, never silently empty).
# ---------------------------------------------------------------------------

def _load_live_failure_note():
    import importlib
    mod = importlib.import_module("negative_control_open_reds")
    return mod._live_failure_note


def test_live_failure_note_anchors_on_the_state_table_when_the_marker_is_present():
    """POSITIVE: with the marker present, the note starts at the marker, not at a fixed
    byte offset that could land anywhere relative to it."""
    live_failure_note = _load_live_failure_note()
    preamble = "pytest collection noise\n" * 400  # long enough to defeat a [-1500:] tail
    table = "OPEN-REDS: frame\nstate                            check\nFAIL(...) something_unaccounted\n"
    note = live_failure_note(preamble + table, "")
    assert note.startswith("OPEN-REDS: frame"), (
        f"note must anchor on the table header, not a blind tail:\n{note[:200]!r}"
    )
    assert "something_unaccounted" in note, "the named failing check must survive into the note"
    assert "pytest collection noise" not in note, (
        "the preamble before the marker must not leak into the anchored note"
    )


def test_live_failure_note_falls_back_to_a_bounded_tail_when_the_marker_is_missing():
    """NEGATIVE: marker absent (e.g. an older/malformed check_open_reds.py that never
    printed the header, or output truncated before the header was ever written) must
    still produce a non-empty, bounded diagnostic -- never blank, and never raise --
    so a missing marker degrades to "the same tail as before" rather than to nothing."""
    live_failure_note = _load_live_failure_note()
    stdout = "z" * 5000  # no "OPEN-REDS: frame" anywhere
    stderr = "w" * 3000
    note = live_failure_note(stdout, stderr)
    assert note == stdout[-4000:] + stderr[-1500:], "must fall back to the bounded tail exactly"
    assert note, "a missing marker must still produce a non-empty, fail-loud diagnostic"
    assert len(note) <= 4000 + 1500


# ---------------------------------------------------------------------------
# ci/check_hardcoded_foundry_paths.py -- static screen against reintroducing issue #11's
# defect class (a Foundry cache path spelled out literally, going stale silently whenever
# Foundry Local's own catalog revision changes) in any live tool issue #19 migrated to the
# rust/tools/foundry_discovery.py resolver, while still allowing the bench/results/
# archival scripts issue #19 gave an explicit PHI35_MODEL override instead.
# ---------------------------------------------------------------------------

def _write_tree(root: Path, files: dict) -> None:
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")


def test_hardcoded_foundry_paths_passes_on_a_clean_tree(tmp_path):
    """POSITIVE: no occurrence of the pattern anywhere -> PASS."""
    _write_tree(tmp_path, {"bench/tool.py": "print('nothing to see here')\n"})
    proc = run_check("check_hardcoded_foundry_paths.py", "--root", str(tmp_path))
    assert proc.returncode == EXIT_PASS, proc.stdout + proc.stderr
    assert "FOUNDRY-PATHS: PASS" in proc.stdout


def test_hardcoded_foundry_paths_reds_on_a_live_tool_outside_the_allowlist(tmp_path):
    """NEGATIVE: a fresh live tool pasting the hardcoded pattern back in must be named and
    must fail, not silently pass -- this is the actual defect class issue #11/#19 exist
    for."""
    _write_tree(tmp_path, {
        "rust/tools/probe_x.py": (
            'MODEL = r"C:\\Users\\x\\.foundry\\cache\\models\\Microsoft\\Foo"\n'
        ),
    })
    proc = run_check("check_hardcoded_foundry_paths.py", "--root", str(tmp_path))
    assert proc.returncode == EXIT_FAIL_CONDITION, proc.stdout + proc.stderr
    assert "rust/tools/probe_x.py" in proc.stdout


def test_hardcoded_foundry_paths_allows_archival_scripts_under_bench_results(tmp_path):
    """The bench/results/ allowlist exists specifically so an archived investigation
    script's deliberately-preserved historical default path is not mistaken for a live
    hardcode."""
    _write_tree(tmp_path, {
        "bench/results/probe_x.py": (
            'MODEL = r"C:\\Users\\x\\.foundry\\cache\\models\\Microsoft\\Foo"\n'
        ),
    })
    proc = run_check("check_hardcoded_foundry_paths.py", "--root", str(tmp_path))
    assert proc.returncode == EXIT_PASS, proc.stdout + proc.stderr
    assert "FOUNDRY-PATHS: PASS" in proc.stdout


def test_hardcoded_foundry_paths_missing_root_is_an_instrument_error(tmp_path):
    """An absent --root is an instrument outage, not a clean lane: ERROR(instrument), not
    0."""
    missing = tmp_path / "does-not-exist"
    proc = run_check("check_hardcoded_foundry_paths.py", "--root", str(missing))
    assert proc.returncode == EXIT_ERROR_INSTRUMENT, proc.stdout + proc.stderr
    assert "ERROR(instrument=root_absent)" in proc.stdout


def test_hardcoded_foundry_paths_does_not_match_identity_strings_or_pathlib_joins(tmp_path):
    """NEGATIVE (of the negative): a resolver-style model *identity* string and a pathlib
    join built from separate literal segments are exactly the shape the migrated live
    tools now use, and must not be mistaken for the hardcoded-path shape this screen
    exists to catch."""
    _write_tree(tmp_path, {
        "rust/tools/ok.py": (
            'import pathlib\n'
            'variant_name = "Phi-3.5-mini-instruct-cuda-gpu"\n'
            'root = pathlib.Path.home() / ".foundry" / "cache" / "models"\n'
        ),
    })
    proc = run_check("check_hardcoded_foundry_paths.py", "--root", str(tmp_path))
    assert proc.returncode == EXIT_PASS, proc.stdout + proc.stderr


def test_the_real_source_tree_has_no_new_hardcoded_foundry_paths():
    """LIVE: run the screen against the real repository tree (default --root), not a
    synthesized fixture -- this is what actually gates the lane."""
    proc = run_check("check_hardcoded_foundry_paths.py")
    assert proc.returncode == EXIT_PASS, proc.stdout + proc.stderr


def test_hardcoded_foundry_paths_negative_control_all_arms_pass():
    """The negative control's own REPLAYED/PLANTED/LIVE arms must all be green; a FAIL
    here means the screen no longer catches the historical defect it was written for."""
    proc = run_check("negative_control_hardcoded_foundry_paths.py")
    assert proc.returncode == EXIT_PASS, proc.stdout + proc.stderr
    assert "/5 arms pass" in proc.stdout


# ---------------------------------------------------------------------------
# Result-identity contract (issue #19 follow-up, Morpheus review on PR #31) -- every
# PHI35_MODEL-reading probe stamps the resolved ONNX model path and its exact SHA-256
# into its own output record, so a PHI35_MODEL override -- or a stale/corrupted cached
# file silently sitting at the historical default path -- can never be silently absorbed
# into the evidence: the record always names the exact bytes it was computed from.
#
# Discovery is repo-wide and SEMANTIC (ci/phi35_identity_audit.py), not a source-text
# regex. The regex screen this replaces was rejected on PR #31 for three failures that a
# grep over source text cannot avoid, all of them found by Morpheus:
#
#   * `subprocess\.run\(\s*\[\s*sys\.executable` recognises ONE spelling. Both real
#     spawners build their argv into a variable first
#     (`cmd = [sys.executable, str(PROBE), ...]; subprocess.run(cmd, env=env)`), so the
#     screen walked past rust/tools/device_loss_gate.py and
#     bench/results/probe_device_memory_kv.py -- the two files that were actually writing
#     unattributed records.
#   * It matched its own source text: the pattern's own literal, and any prose quoting it,
#     read as a hit, so a file could be "discovered" for containing a description of the
#     defect rather than the defect.
#   * `dict(os.environ)` as the test for inheritance misses `env=os.environ.copy()` and,
#     worse, treats the STRONGEST inheritance -- passing no `env=` at all -- as no
#     inheritance.
#
# The audit module parses each file and reasons over the tree: environment reads through
# any alias (`os.environ.get`, `os.environ[...]`, `os.getenv`, `from os import environ`,
# `import os as o`), argv/env built inline or through variables, script targets named
# through module-level path constants, JSON *records* distinguished from `json.dumps`
# printed to stdout, and the reachability closed to a fixed point rather than one hop.
# ---------------------------------------------------------------------------

sys.path.insert(0, str(CI_DIR))
import phi35_identity_audit as _identity_audit  # noqa: E402


def _repo_facts() -> dict:
    """AST facts for every Python file in the repository, computed once per session."""
    global _REPO_FACTS_CACHE
    if _REPO_FACTS_CACHE is None:
        _REPO_FACTS_CACHE = _identity_audit.analyze_tree(REPO_ROOT)
    return _REPO_FACTS_CACHE


_REPO_FACTS_CACHE = None


def _iter_all_python_files() -> list[Path]:
    return [REPO_ROOT / rel for rel in sorted(_repo_facts())]


def _phi35_model_direct_readers() -> list[Path]:
    """Every *.py file anywhere in the repo that reads PHI35_MODEL from the environment
    directly -- by AST, so an alias spelling (`os.getenv`, `environ[...]`, an aliased
    `import os as o`) counts and a mention inside a docstring does not."""
    return [REPO_ROOT / rel for rel, f in sorted(_repo_facts().items()) if f.reads_model_env]


def _phi35_subprocess_inheritors(direct_readers: list[Path]) -> list[Path]:
    """A file need not read PHI35_MODEL itself to inherit the same silent-substitution
    gap: spawning `sys.executable` against a model-bearing script while handing it the
    parent environment passes that child whatever override is set, with no code in the
    parent ever naming PHI35_MODEL. Reachability is closed to a FIXED POINT, so a wrapper
    of a wrapper is included by construction rather than by luck of being one hop away."""
    direct = {p.relative_to(REPO_ROOT).as_posix() for p in direct_readers}
    facts = _repo_facts()
    reached = _identity_audit.model_bearing_scripts(facts)
    out = []
    for rel, f in sorted(facts.items()):
        if rel in direct:
            continue
        for spawn in f.spawns:
            named = [s for s in spawn.scripts if s in reached and reached[s] != rel]
            if spawn.inherits_env and named:
                out.append(REPO_ROOT / rel)
                break
    return out


def _writes_json_output(path: Path) -> bool:
    """Writes a JSON *record* -- a file on disk. `json.dumps(...)` printed to stdout is a
    report, not an artifact anybody replays, and conflating the two is how the old regex
    counted `print(json.dumps(doc))` as an evidence write."""
    rel = path.relative_to(REPO_ROOT).as_posix()
    return _repo_facts()[rel].writes_json_record


def test_phi35_model_reader_discovery_is_repo_wide_not_bench_results_only():
    """The exhaustiveness bound: a repo-wide scan must find the ~23 bench/results/
    archival scripts AND tests/ops/probe_validation_phi35.py AND its sibling skip-gating
    test tests/ops/test_push_constants_written.py -- the two files a bench/results/-only
    glob structurally cannot reach, which is exactly what Morpheus's review found."""
    readers = _phi35_model_direct_readers()
    names = {p.relative_to(REPO_ROOT).as_posix() for p in readers}
    assert "tests/ops/probe_validation_phi35.py" in names, sorted(names)
    assert "tests/ops/test_push_constants_written.py" in names, sorted(names)
    bench_results_readers = [n for n in names if n.startswith("bench/results/")]
    assert len(bench_results_readers) >= 20, (
        f"expected the ~23 archival PHI35_MODEL scripts issue #19 migrated, found "
        f"{len(bench_results_readers)}: {sorted(bench_results_readers)}"
    )
    assert len(readers) >= 25, (
        "a repo-wide scan finding fewer readers than the bench/results/-only glob alone "
        "used to find would mean the wider walk itself is broken"
    )


def test_every_phi35_model_reader_that_writes_json_stamps_result_identity():
    """Every repo-wide PHI35_MODEL reader that also writes a JSON record must stamp
    `onnx_file`/`onnx_sha256` into it -- either by defining `_result_identity()` itself or
    by importing one from a module that does (tests/ops/test_push_constants_written.py
    reads PHI35_MODEL only to decide a pytest skip and never writes JSON itself, so it is
    exempt from this one; discovery still finds it above, it just is not a writer)."""
    readers = _phi35_model_direct_readers()
    facts = _repo_facts()
    missing = []
    for path in readers:
        rel = path.relative_to(REPO_ROOT).as_posix()
        if not facts[rel].writes_json_record:
            continue
        if not facts[rel].names_the_model:
            missing.append(rel)
    assert not missing, (
        f"{len(missing)} PHI35_MODEL reader(s) write JSON but do not stamp "
        f"onnx_file/onnx_sha256: {missing}"
    )


def test_phi35_model_subprocess_inheritors_are_discovered_and_stamp_result_identity():
    """The subprocess-inheritance regression Morpheus's review named:
    bench/results/probe_push_constants_written.py never reads PHI35_MODEL itself, only
    passes it through `dict(os.environ)` into a subprocess of probe_validation_phi35.py.
    This must both be discovered by the repo-wide scan and, since it writes its own JSON
    (push_constants_written.json / push_constants_sensitivity.json), carry a
    `_result_identity()` reference of its own -- a fix that touched only the direct reader
    and missed this wrapper would leave the exact gap Morpheus found standing."""
    readers = _phi35_model_direct_readers()
    inheritors = _phi35_subprocess_inheritors(readers)
    facts = _repo_facts()
    names = {p.relative_to(REPO_ROOT).as_posix() for p in inheritors}
    assert "bench/results/probe_push_constants_written.py" in names, sorted(names)
    # One of the two the regex screen walked past: it builds argv into a variable before
    # spawning -- the exact shape Morpheus's rejection of PR #31 named.
    assert "bench/results/probe_device_memory_kv.py" in names, sorted(names)
    missing = []
    for path in inheritors:
        rel = path.relative_to(REPO_ROOT).as_posix()
        if not facts[rel].writes_json_record:
            continue
        if not facts[rel].names_the_model:
            missing.append(rel)
    assert not missing, (
        f"{len(missing)} subprocess-inheritor(s) write JSON but never name the model: "
        f"{missing}"
    )


def test_every_archival_phi35_probe_reuses_the_shared_hasher():
    """The stamping helper must reuse `model_provenance.sha256_of` (already a streaming
    SHA-256, already exercised by tests/ops/test_small_model_provenance.py) rather than
    each reader/inheritor defining its own divergent hasher for this field.

    Scoped by AST to files that DEFINE `_result_identity()` themselves. A file that only
    PROPAGATES a child's identity (rust/tools/device_loss_gate.py's `gate_identity`,
    bench/results/probe_device_memory_kv.py's `lane_identity`) must NOT re-hash: the child
    is the process that opened the file, so its hash is the hash of the bytes that were
    actually executed, not of whatever is at that path once the parent finishes. Scoping
    this by source text -- the previous spelling -- also caught any file that merely
    MENTIONED `_result_identity` in a comment, which is the same defect as the regex
    discovery this suite replaced."""
    facts = _repo_facts()
    readers = _phi35_model_direct_readers()
    inheritors = _phi35_subprocess_inheritors(readers)
    missing = []
    for p in (*readers, *inheritors):
        rel = p.relative_to(REPO_ROOT).as_posix()
        if not facts[rel].defines_result_identity:
            continue
        text = p.read_text(encoding="utf-8")
        if '"onnx_file"' in text and "_model_provenance.sha256_of" not in text:
            missing.append(rel)
    assert not missing, (
        f"{len(missing)} file(s) define _result_identity and stamp a hash without "
        f"reusing model_provenance.sha256_of: {missing}"
    )



def test_archival_probe_result_identity_reflects_a_phi35_model_override(tmp_path):
    """FUNCTIONAL: pointing PHI35_MODEL at a different file must change the stamped
    onnx_file AND onnx_sha256 in lockstep -- an override that changed the resolved path
    but left a stale hash behind would be exactly the silent-substitution failure mode
    this contract exists to close."""
    import importlib.util

    probe_path = REPO_ROOT / "bench" / "results" / "probe_kv_depth.py"
    fake_model = tmp_path / "fake_phi35.onnx"
    fake_model.write_bytes(b"not a real onnx file, just needs to exist and hash")

    old_env = os.environ.get("PHI35_MODEL")
    os.environ["PHI35_MODEL"] = str(fake_model)
    try:
        spec = importlib.util.spec_from_file_location("probe_kv_depth_under_test", probe_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        identity = mod._result_identity()
    finally:
        if old_env is None:
            os.environ.pop("PHI35_MODEL", None)
        else:
            os.environ["PHI35_MODEL"] = old_env

    assert identity["onnx_file"] == str(fake_model)

    import hashlib
    expected = hashlib.sha256(fake_model.read_bytes()).hexdigest()
    assert identity["onnx_sha256"] == expected


def test_archival_probe_result_identity_detects_silent_content_substitution(tmp_path):
    """FUNCTIONAL: a file swapped for a DIFFERENT one at the exact same path (the
    silent-substitution scenario Morpheus's review named -- a stale/corrupted re-download
    landing at the historical default path with no PHI35_MODEL override at all) must
    produce a different onnx_sha256. A hash that failed to change would mean the stamp
    is decorative rather than a detector."""
    import importlib.util

    probe_path = REPO_ROOT / "bench" / "results" / "probe_kv_depth.py"
    model_path = tmp_path / "phi35.onnx"
    model_path.write_bytes(b"artifact version A")

    old_env = os.environ.get("PHI35_MODEL")
    os.environ["PHI35_MODEL"] = str(model_path)
    try:
        spec = importlib.util.spec_from_file_location(
            "probe_kv_depth_under_test_2", probe_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        before = mod._result_identity()["onnx_sha256"]

        # Silent substitution: same path, different bytes -- no override change at all.
        model_path.write_bytes(b"artifact version B (substituted)")
        after = mod._result_identity()["onnx_sha256"]
    finally:
        if old_env is None:
            os.environ.pop("PHI35_MODEL", None)
        else:
            os.environ["PHI35_MODEL"] = old_env

    assert before != after, (
        "onnx_sha256 did not change when the file at the same path was substituted -- "
        "the stamp would not have revealed the substitution"
    )


def test_live_phi35_tools_no_longer_bypass_the_resolver_with_a_pre_check_override():
    """`probe_phi35_claim_reading.py`, `probe_silent_cpu_rebuild.py` and
    `roofline_split.py` used to check PHI35_MODEL BEFORE calling
    foundry_discovery.resolve_model_path(), which let an override silently skip the
    resolver's exact variant+execution-provider validation (Morpheus, PR #31 review).
    These are LIVE tools, not archival replay scripts, so -- unlike bench/results/ -- they
    must resolve by identity every time, with no override path at all."""
    live_tools = (
        REPO_ROOT / "rust" / "tools" / "probe_phi35_claim_reading.py",
        REPO_ROOT / "rust" / "tools" / "probe_silent_cpu_rebuild.py",
        REPO_ROOT / "rust" / "tools" / "roofline_split.py",
    )
    functional_override = re.compile(r'os\.environ\.get\(\s*["\']PHI35_MODEL["\']')
    for path in live_tools:
        text = path.read_text(encoding="utf-8")
        # A comment naming PHI35_MODEL to explain why there is deliberately no override is
        # fine and expected; an actual `os.environ.get("PHI35_MODEL", ...)` read is the
        # bypass this test exists to reject.
        assert not functional_override.search(text), (
            f"{path.relative_to(REPO_ROOT).as_posix()} still reads PHI35_MODEL from the "
            f"environment; a live tool must resolve the model by identity "
            f"unconditionally, not accept a pre-resolver override"
        )
        assert "_resolve_model(" not in text, (
            f"{path.relative_to(REPO_ROOT).as_posix()} still defines/calls the old "
            f"_resolve_model() override helper"
        )
        assert "_foundry_discovery.resolve_model_path(_PHI35_SPEC)" in text, (
            f"{path.relative_to(REPO_ROOT).as_posix()} does not call the shared resolver "
            f"directly"
        )


def test_live_phi35_tools_still_fail_loud_on_an_unresolvable_model():
    """The 3 corrected live tools must still exit non-zero with ERROR(instrument) when
    the model cannot be resolved at all -- the fix removes the override, it must not
    also remove the fail-loud contract every other live tool already has."""
    live_tools = (
        "probe_phi35_claim_reading.py",
        "probe_silent_cpu_rebuild.py",
        "roofline_split.py",
    )
    for name in live_tools:
        path = REPO_ROOT / "rust" / "tools" / name
        text = path.read_text(encoding="utf-8")
        assert "except _foundry_discovery.FoundryDiscoveryError as exc:" in text
        assert "ERROR(instrument): Phi-3.5 model not resolvable" in text


# ---------------------------------------------------------------------------
# The semantic audit itself (ci/phi35_identity_audit.py) -- independent revision of PR #31
# after rejection (Tank, 2026-08-05).
#
# The arms below are PLANTED: each one is the smallest source that carries a shape, so a
# green here says the rule fires on the shape rather than on the repository happening to
# be clean today. The two shapes the rejected regex walked past are replayed first, and
# every one of them is checked BOTH ways -- the old pattern must miss it and the audit
# must catch it -- because "the new screen is green" is also what the broken screen said.
# ---------------------------------------------------------------------------

#: The exact pattern that shipped in the rejected revision, kept only as a control.
_REJECTED_INLINE_ONLY_REGEX = re.compile(r'subprocess\.run\(\s*\[\s*sys\.executable')

_PLANTED_READER = '''\
import json, os, pathlib, sys
ONNX_FILE = pathlib.Path(os.environ.get("PHI35_MODEL", "default.onnx"))
def _result_identity():
    return {"onnx_file": str(ONNX_FILE), "onnx_sha256": _model_provenance.sha256_of(ONNX_FILE)}
def main():
    pathlib.Path("child.json").write_text(json.dumps({**{}, **_result_identity()}))
'''


def _plant(**sources: str) -> dict:
    """Analyze planted sources with the production analyzer, keyed by relative path."""
    return {
        rel.replace("__", "/") + ".py": _identity_audit.analyze_source(
            rel.replace("__", "/") + ".py", src
        )
        for rel, src in sources.items()
    }


def test_planted_variable_built_argv_is_caught_and_the_rejected_regex_misses_it():
    """THE rejection finding. `cmd = [sys.executable, str(PROBE), ...]` then
    `subprocess.run(cmd, env=env)` is the shape both rust/tools/device_loss_gate.py and
    bench/results/probe_device_memory_kv.py actually use, and the inline-only regex
    cannot see it. Both halves are asserted: the old pattern misses, the audit catches."""
    spawner = '''\
import json, os, pathlib, subprocess, sys
PROBE = pathlib.Path(__file__).parent / "probe_reader.py"
def go():
    env = dict(os.environ)
    cmd = [sys.executable, str(PROBE), "--out", "x.json"]
    subprocess.run(cmd, env=env, capture_output=True)
    pathlib.Path("gate.json").write_text(json.dumps({"reps": 1}))
'''
    assert not _REJECTED_INLINE_ONLY_REGEX.search(spawner), (
        "the planted source must be one the REJECTED regex misses, or this arm proves "
        "nothing about the defect"
    )
    facts = _plant(probe_reader=_PLANTED_READER, tools__gate=spawner)
    violations, found = _identity_audit.violations_in(facts)
    assert "tools/gate.py" in found, found
    assert [v.rel for v in violations] == ["tools/gate.py"], found


def test_planted_inline_argv_is_still_caught_after_the_regex_is_gone():
    """The shape the old regex DID catch must not regress: replacing a screen is only an
    improvement if it still covers what the old one covered."""
    spawner = '''\
import json, os, pathlib, subprocess, sys
def go():
    subprocess.run([sys.executable, "probe_reader.py"], env=dict(os.environ))
    pathlib.Path("wrap.json").write_text(json.dumps({"n": 1}))
'''
    assert _REJECTED_INLINE_ONLY_REGEX.search(spawner)
    facts = _plant(probe_reader=_PLANTED_READER, wrap=spawner)
    violations, found = _identity_audit.violations_in(facts)
    assert "wrap.py" in found
    assert [v.rel for v in violations] == ["wrap.py"]


@pytest.mark.parametrize("env_kwarg, why", [
    ("env=env", "environment copied into a variable first"),
    ("env=os.environ.copy()", "the .copy() spelling `dict(os.environ)` never matched"),
    ("env=dict(os.environ)", "the one spelling the old screen knew"),
    ("", "NO env= at all -- the child inherits everything, the strongest case, which the "
         "old screen scored as no inheritance"),
])
def test_planted_env_inheritance_spellings_all_count_as_inheritance(env_kwarg, why):
    spawner = f'''\
import json, os, pathlib, subprocess, sys
def go():
    env = dict(os.environ)
    cmd = [sys.executable, "probe_reader.py"]
    subprocess.run(cmd, {env_kwarg})
    pathlib.Path("w.json").write_text(json.dumps({{"n": 1}}))
'''
    facts = _plant(probe_reader=_PLANTED_READER, w=spawner)
    violations, _ = _identity_audit.violations_in(facts)
    assert [v.rel for v in violations] == ["w.py"], why


def test_planted_env_explicitly_scrubbed_is_not_inheritance():
    """The other polarity: a child handed a hand-built environment does NOT inherit
    PHI35_MODEL, and calling that a violation would make the screen cry wolf until it is
    switched off."""
    spawner = '''\
import json, os, pathlib, subprocess, sys
def go():
    subprocess.run([sys.executable, "probe_reader.py"], env={"PATH": os.environ["PATH"]})
    pathlib.Path("w.json").write_text(json.dumps({"n": 1}))
'''
    facts = _plant(probe_reader=_PLANTED_READER, w=spawner)
    violations, found = _identity_audit.violations_in(facts)
    assert "w.py" not in found, found
    assert not violations


@pytest.mark.parametrize("read_line", [
    'M = os.environ.get("PHI35_MODEL", "d.onnx")',
    'M = os.environ["PHI35_MODEL"]',
    'M = os.getenv("PHI35_MODEL", "d.onnx")',
])
def test_planted_direct_read_alias_spellings_are_all_discovered(read_line):
    """`os.environ.get(...)` was the only spelling the old regex knew. Subscript and
    `os.getenv` read the same variable and were invisible."""
    src = f'''\
import json, os, pathlib
{read_line}
def go():
    pathlib.Path("o.json").write_text(json.dumps({{"m": M}}))
'''
    facts = _plant(r=src)
    violations, found = _identity_audit.violations_in(facts)
    assert "r.py" in found, found
    assert [v.rel for v in violations] == ["r.py"]


@pytest.mark.parametrize("preamble, read_line", [
    ("import os as o", 'M = o.environ.get("PHI35_MODEL", "d.onnx")'),
    ("from os import environ", 'M = environ.get("PHI35_MODEL", "d.onnx")'),
    ("from os import getenv", 'M = getenv("PHI35_MODEL", "d.onnx")'),
    ("from os import environ as E", 'M = E["PHI35_MODEL"]'),
])
def test_planted_import_alias_variants_are_discovered(preamble, read_line):
    """Import aliasing defeats any pattern anchored on the literal text `os.environ`."""
    src = f'''\
import json, pathlib
{preamble}
{read_line}
def go():
    pathlib.Path("o.json").write_text(json.dumps({{"m": M}}))
'''
    facts = _plant(r=src)
    violations, _ = _identity_audit.violations_in(facts)
    assert [v.rel for v in violations] == ["r.py"], read_line


def test_planted_child_output_field_discarding_is_the_violation_not_the_absence_of_a_read():
    """bench/results/probe_device_memory_kv.py's exact shape: it READ the child record --
    it takes its byte totals from there -- and dropped `onnx_file`/`onnx_sha256` while
    writing its own. Having the identity in hand and discarding it is the failure; the
    fixed shape, which copies those two fields through, is clean."""
    discards = '''\
import json, os, pathlib, subprocess, sys
def go():
    cmd = [sys.executable, "probe_reader.py", "--out", "c.json"]
    env = dict(os.environ)
    subprocess.run(cmd, env=env)
    record = json.loads(pathlib.Path("c.json").read_text())
    pathlib.Path("mine.json").write_text(json.dumps({"bytes": record["bytes"]}))
'''
    propagates = '''\
import json, os, pathlib, subprocess, sys
def go():
    cmd = [sys.executable, "probe_reader.py", "--out", "c.json"]
    env = dict(os.environ)
    subprocess.run(cmd, env=env)
    record = json.loads(pathlib.Path("c.json").read_text())
    ident = {"onnx_file": record.get("onnx_file"), "onnx_sha256": record.get("onnx_sha256")}
    pathlib.Path("mine.json").write_text(json.dumps({"bytes": record["bytes"], **ident}))
'''
    bad = _plant(probe_reader=_PLANTED_READER, mine=discards)
    good = _plant(probe_reader=_PLANTED_READER, mine=propagates)
    assert [v.rel for v in _identity_audit.violations_in(bad)[0]] == ["mine.py"]
    assert not _identity_audit.violations_in(good)[0]


def test_a_json_report_printed_to_stdout_is_not_an_evidence_record():
    """`print(json.dumps(doc))` leaves nothing behind to be replayed or contradicted, and
    counting it as a record write is how a screen accumulates the false positives that get
    it deleted. Only a write to a file is an artifact."""
    prints_only = '''\
import json, os
M = os.environ.get("PHI35_MODEL", "d.onnx")
def go():
    print(json.dumps({"m": M}))
'''
    facts = _plant(p=prints_only)
    _, found = _identity_audit.violations_in(facts)
    assert "p.py" not in found, found


def test_the_audit_does_not_match_source_that_only_describes_the_defect():
    """The rejected regex matched the text of its own pattern, so a file could be
    'discovered' for DESCRIBING the defect rather than containing it.

    Arm 1 re-analyzes ci/phi35_identity_audit.py under an invented path -- which takes its
    declared exclusion out of play -- and it must be clean: every occurrence of the shapes
    it hunts is prose or a string, and prose is not code.

    Arm 2 plants a file whose entire body is a docstring plus a string constant holding a
    perfect copy of a violating program. A screen over source text calls that a violation;
    an analyzer over the tree sees a string.

    ci/test_lane_checks.py itself is deliberately NOT asserted clean here: unlike the
    audit module it really does read PHI35_MODEL (the functional override arms above set
    and restore it) and really does write JSON files (synthetic registers in tmp_path), so
    it is a declared exclusion with a stated reason rather than a file that happens to
    look innocent."""
    audit_src = (CI_DIR / "phi35_identity_audit.py").read_text(encoding="utf-8")
    facts = {"elsewhere/copy_of_audit.py": _identity_audit.analyze_source(
        "elsewhere/copy_of_audit.py", audit_src)}
    _, found = _identity_audit.violations_in(facts)
    assert not found, f"the audit reported source that only DESCRIBES the defect: {found}"

    prose_only = '"""' + "\n" + _PLANTED_READER + '\n"""\n' + "SAMPLE = " + repr(
        'import os, json, pathlib, subprocess, sys\n'
        'M = os.environ.get("PHI35_MODEL", "d.onnx")\n'
        'pathlib.Path("o.json").write_text(json.dumps({"m": M}))\n'
    ) + "\n"
    _, found_prose = _identity_audit.violations_in(_plant(doc=prose_only))
    assert not found_prose, found_prose

    assert "ci/test_lane_checks.py" in _identity_audit.NOT_A_PRODUCER

    # ...while the same shape planted as real code IS reported, so the arms above are not
    # green merely because the analyzer reports nothing at all.
    planted = _plant(probe_reader=_PLANTED_READER, real=(
        'import json, os, pathlib, subprocess, sys\n'
        'def go():\n'
        '    cmd = [sys.executable, "probe_reader.py"]\n'
        '    subprocess.run(cmd, env=dict(os.environ))\n'
        '    pathlib.Path("r.json").write_text(json.dumps({"n": 1}))\n'
    ))
    assert [v.rel for v in _identity_audit.violations_in(planted)[0]] == ["real.py"]


def test_reachability_is_a_fixed_point_not_one_hop():
    """A wrapper of a wrapper. The old screen looked exactly one hop from a direct reader,
    and the two files it missed happened to be one hop away -- which made a hard limit look
    like a scoping decision. Two hops must be reported."""
    mid = '''\
import json, os, pathlib, subprocess, sys
def go():
    cmd = [sys.executable, "probe_reader.py"]
    subprocess.run(cmd, env=dict(os.environ))
    pathlib.Path("mid.json").write_text(json.dumps({"onnx_file": "x", "onnx_sha256": "y"}))
'''
    outer = '''\
import json, os, pathlib, subprocess, sys
def go():
    cmd = [sys.executable, "mid.py"]
    subprocess.run(cmd, env=dict(os.environ))
    pathlib.Path("outer.json").write_text(json.dumps({"n": 1}))
'''
    facts = _plant(probe_reader=_PLANTED_READER, mid=mid, outer=outer)
    violations, found = _identity_audit.violations_in(facts)
    assert "outer.py" in found, found
    assert [v.rel for v in violations] == ["outer.py"]


def test_the_pre_fix_shape_of_device_loss_gate_is_reported():
    """REPLAYED: `rust/tools/device_loss_gate.py` as it stood at 60f0ae7 -- reduced to the
    lines that carry the shape, verbatim in structure: build env, build cmd, spawn the
    model-bearing probe, read the child's record for counters, write the gate record with
    no identity in it."""
    pre_fix = '''\
import json, os, pathlib, subprocess, sys, time
REPO = pathlib.Path(__file__).resolve().parent.parent.parent
PROBE = REPO / "bench" / "results" / "probe_reader.py"
def one_rep(i, steps, lane):
    env = dict(os.environ)
    env["ONNXRUNTIME_VULKAN_EP_LIB"] = "x"
    cmd = [sys.executable, str(PROBE), "--worker", "--steps", str(steps)]
    proc = subprocess.run(cmd, env=env, capture_output=True)
    doc = json.loads(pathlib.Path("rep.json").read_text())
    return {"rep": i, "exit_code": proc.returncode, "steps_recorded": len(doc["per_step"])}
def main():
    reps = [one_rep(i, 2, "resident") for i in range(8)]
    doc = {"gate": "device_loss_gate", "reps": reps}
    (REPO / "bench" / "results" / "device_loss_gate.json").write_text(json.dumps(doc, indent=2))
'''
    facts = _plant(probe_reader=_PLANTED_READER, rust__tools__gate=pre_fix)
    violations, found = _identity_audit.violations_in(facts)
    assert [v.rel for v in violations] == ["rust/tools/gate.py"], found
    assert "argv from variable:cmd" in found["rust/tools/gate.py"]


def test_the_real_source_tree_has_no_unattributed_phi35_evidence_producer():
    """LIVE: the whole repository, by AST. This is the arm that actually gates the lane;
    every planted arm above exists to show that a green here is a fact about the tree and
    not about the screen being blind."""
    violations, found, _ = _identity_audit.audit(REPO_ROOT)
    assert not violations, "\n".join(f"{v.rel}: {v.why} ({v.reached_via})" for v in violations)
    assert len(found) >= 25, (
        f"the audit found only {len(found)} producer(s); a discovery set that shrank is "
        f"how a screen goes quiet without anyone deciding it should: {sorted(found)}"
    )


def test_the_two_tools_named_in_the_rejection_now_name_their_model():
    """The specific files Morpheus's rejection named must be discovered as producers by
    the audit AND must name the model. Named literally so a future refactor that stops
    reaching them fails here rather than reporting a smaller clean set."""
    violations, found, facts = _identity_audit.audit(REPO_ROOT)
    for rel in ("rust/tools/device_loss_gate.py", "bench/results/probe_device_memory_kv.py"):
        assert rel in found, sorted(found)
        assert facts[rel].names_the_model, rel
    assert not violations


def test_the_two_fixed_tools_record_an_explicit_identity_error_rather_than_a_blank():
    """A success path that writes `onnx_file: null` and says nothing else is worse than no
    field at all: it looks stamped. Both tools must carry an explicit
    `onnx_identity_error` on every path where the identity could not be established, and
    the gate must exit non-zero rather than publish an unattributable rate."""
    gate = (REPO_ROOT / "rust" / "tools" / "device_loss_gate.py").read_text(encoding="utf-8")
    kv = (REPO_ROOT / "bench" / "results" / "probe_device_memory_kv.py").read_text(
        encoding="utf-8")
    for text, name in ((gate, "device_loss_gate.py"), (kv, "probe_device_memory_kv.py")):
        assert "onnx_identity_error" in text, name
        assert "child_record_carried_no_model_identity" in text, name
    assert "ERROR(instrument=model_identity_unknown)" in gate
    assert "children_disagree" in gate
    assert "the lanes consumed different models" in kv


def test_device_loss_gate_propagates_one_identity_and_refuses_disagreement():
    """FUNCTIONAL: the gate's aggregation rule, exercised directly. Repetitions of two
    different models do not pool into one loss rate, and saying so in the record is the
    only thing that makes the pooling falsifiable."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "device_loss_gate_under_test", REPO_ROOT / "rust" / "tools" / "device_loss_gate.py")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)  # type: ignore[union-attr]

    agreed = gate.gate_identity([
        {"onnx_file": "m.onnx", "onnx_sha256": "aa"},
        {"onnx_file": "m.onnx", "onnx_sha256": "aa"},
    ])
    assert agreed == {"onnx_file": "m.onnx", "onnx_sha256": "aa"}

    disagreed = gate.gate_identity([
        {"onnx_file": "m.onnx", "onnx_sha256": "aa"},
        {"onnx_file": "other.onnx", "onnx_sha256": "bb"},
    ])
    assert "children_disagree" in disagreed["onnx_identity_error"]
    assert disagreed["onnx_file"] is None
    assert len(disagreed["onnx_identities_seen"]) == 2

    silent = gate.gate_identity([{"rep": 0}, {"rep": 1}])
    assert silent["onnx_file"] is None
    assert "no_child_record_named_a_model" in silent["onnx_identity_error"]


def test_device_memory_kv_refuses_lanes_that_measured_different_models():
    """FUNCTIONAL: the two lanes differ by exactly one environment variable on purpose. If
    they also differ by the model, the readback-slope delta is not a measurement of the
    flag -- the same argument the existing DLL check makes about the binary."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "probe_device_memory_kv_under_test",
        REPO_ROOT / "bench" / "results" / "probe_device_memory_kv.py")
    kv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kv)  # type: ignore[union-attr]

    lanes_same = [
        {"identity": {"onnx_file": "m.onnx", "onnx_sha256": "aa"}},
        {"identity": {"onnx_file": "m.onnx", "onnx_sha256": "aa"}},
    ]
    assert kv.agreed_identity(lanes_same, ran_lanes=True) == {
        "onnx_file": "m.onnx", "onnx_sha256": "aa"}

    lanes_differ = [
        {"identity": {"onnx_file": "m.onnx", "onnx_sha256": "aa"}},
        {"identity": {"onnx_file": "other.onnx", "onnx_sha256": "bb"}},
    ]
    with pytest.raises(SystemExit) as exc:
        kv.agreed_identity(lanes_differ, ran_lanes=True)
    assert "different models" in str(exc.value)

    # A live run that produced no identity at all is an instrument fault, not a record.
    with pytest.raises(SystemExit):
        kv.agreed_identity([{"identity": kv.lane_identity({})}], ran_lanes=True)

    # --reuse of records written before stamping says so instead of inventing one.
    reused = kv.agreed_identity([{"identity": kv.lane_identity({})}], ran_lanes=False)
    assert "reused_records_named_no_model" in reused["onnx_identity_error"]
    assert reused["onnx_file"] is None


def test_the_audit_fails_loud_on_an_unparseable_source_instead_of_skipping_it():
    """A file the analyzer cannot parse is an instrument outage, not a clean file. Silently
    skipping it is how a screen reports green over the one file nobody could read."""
    with pytest.raises(_identity_audit.AuditError):
        _identity_audit.analyze_source("broken.py", "def (:\n")


def test_the_one_record_derived_reader_propagates_the_gate_identity():
    """The audit's stated limit, backed by a check rather than left as a caveat.
    bench/results/probe_lane_logits_identity.py is derived entirely from the device-loss
    gate's record -- a relation this module does not screen for -- so its propagation is
    asserted here explicitly. A derived comparison that cannot name its model is exactly as
    unfalsifiable as an undeclared one."""
    text = (REPO_ROOT / "bench" / "results" / "probe_lane_logits_identity.py").read_text(
        encoding="utf-8")
    assert 'doc["onnx_file"] = rec.get("onnx_file")' in text
    assert 'doc["onnx_sha256"] = rec.get("onnx_sha256")' in text
    assert "source_record_named_no_model" in text
    assert "STATED LIMITS" in (CI_DIR / "phi35_identity_audit.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# ci/check_open_reds.py — the register that tells an accepted red from a new one.
#
# These tests never point the screen at the SHIPPED register. ci/check_open_reds.py
# runs ci/test_lane_checks.py, so a test in this file that ran the real register would
# recurse until something ran out. Every case below builds a synthetic register whose
# checks are `python -c` one-liners; the shipped register's own shape is asserted by
# reading the JSON, not by executing it.
# ---------------------------------------------------------------------------

OPEN_REDS = CI_DIR / "check_open_reds.py"
OPEN_REDS_REGISTER = CI_DIR / "open_reds.json"
OPEN_REDS_REPO = CI_DIR.parent


def _entry(**over):
    e = {
        "id": "e",
        "cmd": ["python", "-c", "print('hello')"],
        "expect": "green",
        "owner": "link",
        "opened": "2026-01-01",
        "review_by": "2099-01-01",
        "reason": "test entry",
        "closes_when": "n/a",
    }
    e.update(over)
    if e.get("expect") == "red" and "extent" not in e and not e.pop("_no_extent", False):
        # Red entries now require an extent (Switch's substring finding). Tests that are
        # not about the extent arm get one that matches nothing, so the set is trivially
        # equal and the arm is a no-op for them.
        e["extent"] = {"pattern": r"^MEMBER (\S+)", "members": []}
    e.pop("_no_extent", None)
    return e


def _open_reds(tmp_path, entries, extra=None, env=None, doc_over=None):
    reg = tmp_path / "reg.json"
    doc = {
        "schema": 1,
        "purpose": "t",
        "checks": entries,
        "subjects": [e["id"] for e in entries if "id" in e],
        "retired": {},
    }
    if doc_over:
        doc.update(doc_over)
    reg.write_text(json.dumps(doc), encoding="utf-8")
    e = dict(os.environ)
    e.pop("OPEN_REDS_TODAY", None)
    e.pop("OPEN_REDS_FORCE_ANNOTATE", None)
    e.setdefault("PYTHONIOENCODING", "utf-8")
    if env:
        e.update(env)
    r = subprocess.run(
        [sys.executable, str(OPEN_REDS), "--register", str(reg), "--repo", str(OPEN_REDS_REPO),
         *(extra or [])],
        capture_output=True, encoding="utf-8", errors="replace", env=e, cwd=str(OPEN_REDS_REPO),
        timeout=300,
    )
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def test_open_reds_passes_when_every_check_is_the_declared_colour(tmp_path):
    rc, out = _open_reds(tmp_path, [
        _entry(id="g"),
        _entry(id="r", expect="red", signature="the known red",
               cmd=["python", "-c", "import sys; print('the known red'); sys.exit(1)"],
               closes_when="when somebody fixes it"),
    ])
    assert rc == 0, out
    assert "ACCOUNTED" in out
    # And PASS must not be readable as "the tree is clean".
    assert "does NOT mean the tree is clean" in out


def test_open_reds_fails_on_a_red_nobody_declared(tmp_path):
    """The defect this file exists for: a NEW red must not hide behind known ones."""
    rc, out = _open_reds(tmp_path, [
        _entry(id="g", cmd=["python", "-c", "import sys; print('surprise'); sys.exit(1)"]),
    ])
    assert rc == 1, out
    assert "unaccounted_red" in out


def test_open_reds_fails_when_an_accepted_red_went_green(tmp_path):
    """The arm that stops the register rotting. An allowlist that only ever suppresses
    grows monotonically, because nothing ever asks anyone to prune it."""
    rc, out = _open_reds(tmp_path, [
        _entry(id="r", expect="red", signature="was failing",
               cmd=["python", "-c", "print('fixed')"],
               closes_when="the day the accessors get callers"),
    ])
    assert rc == 1, out
    assert "stale_acceptance" in out
    assert "the day the accessors get callers" in out


def test_open_reds_acceptance_does_not_stretch_to_a_different_red(tmp_path):
    rc, out = _open_reds(tmp_path, [
        _entry(id="r", expect="red", signature="9 NEW uninvoked instrument(s)",
               cmd=["python", "-c", "import sys; print('10 NEW uninvoked'); sys.exit(1)"]),
    ])
    assert rc == 1, out
    assert "signature_changed" in out


def test_open_reds_lease_expires_and_the_day_before_it_does_not(tmp_path):
    """Both polarities off one knob. 'Any date makes it red' would pass a one-sided test."""
    entry = _entry(id="r", expect="red", signature="still red", review_by="2026-06-01",
                   cmd=["python", "-c", "import sys; print('still red'); sys.exit(1)"])
    rc_before, _ = _open_reds(tmp_path, [entry], env={"OPEN_REDS_TODAY": "2026-05-31"})
    rc_after, out_after = _open_reds(tmp_path, [entry],
                                     env={"OPEN_REDS_TODAY": "2026-06-02"})
    assert rc_before == 0
    assert rc_after == 1 and "lease_expired" in out_after


def test_open_reds_refuses_an_acceptance_with_no_signature(tmp_path):
    e = _entry(id="r", expect="red",
               cmd=["python", "-c", "import sys; sys.exit(1)"])
    rc, out = _open_reds(tmp_path, [e])
    assert rc == 2, out
    assert "signature" in out


def test_open_reds_notices_a_second_file_joining_an_accepted_red(tmp_path):
    """SWITCH'S FINDING, CLOSED. A substring cannot see a set grow.

    The signature still matches — that is the whole point. Only the membership moved, and
    before `extent` the acceptance absorbed the newcomer in silence.
    """
    cmd = ["python", "-c",
           "import sys; print('MEMBER a'); print('MEMBER b'); print('CENSUS-EXTENT'); sys.exit(1)"]
    e = _entry(id="r", expect="red", signature="CENSUS-EXTENT", cmd=cmd,
               extent={"pattern": r"^MEMBER (\S+)", "members": ["a"]})
    rc, out = _open_reds(tmp_path, [e])
    assert rc == 1, out
    assert "extent_widened" in out
    assert "'b'" in out and "signature` matched" in out


def test_open_reds_reports_an_accepted_red_that_shrank(tmp_path):
    """The other polarity, and it is good news rather than a regression — same argument as
    `stale_acceptance`. An acceptance that has stopped covering something must be re-read,
    not left covering it in principle."""
    cmd = ["python", "-c", "import sys; print('MEMBER a'); print('SIG'); sys.exit(1)"]
    e = _entry(id="r", expect="red", signature="SIG", cmd=cmd,
               extent={"pattern": r"^MEMBER (\S+)", "members": ["a", "b"]})
    rc, out = _open_reds(tmp_path, [e])
    assert rc == 1, out
    assert "extent_narrowed" in out and "'b'" in out


def test_open_reds_accepts_a_red_whose_membership_is_unchanged(tmp_path):
    """The green polarity of the same knob: `extent` must not fail a red it does cover, or
    it is just a second way to say red."""
    cmd = ["python", "-c",
           "import sys; print('MEMBER a'); print('MEMBER b'); print('SIG'); sys.exit(1)"]
    e = _entry(id="r", expect="red", signature="SIG", cmd=cmd,
               extent={"pattern": r"^MEMBER (\S+)", "members": ["a", "b"]})
    rc, out = _open_reds(tmp_path, [e])
    assert rc == 0, out
    assert "ACCOUNTED" in out


def test_open_reds_refuses_an_acceptance_with_no_extent(tmp_path):
    """An acceptance with no declared extent is UNCOLOURED — and only that entry is.

    REWRITTEN 2026-08-05 (trinity) BECAUSE THE RULE MOVED, NOT BECAUSE IT WAS DROPPED.
    The refusal used to be a parse-time raise, which is exit 2 over the WHOLE FILE. That
    is how ci/open_reds_device.json spent from 69ac222 onward: unloadable, so `--list`,
    `--only` and a plain run all exited 2 having measured nothing, and ten entries — one
    of them a lease that could have expired unnoticed — were reported on by a usage error
    about an eleventh. The refusal is now per-entry: ERROR(instrument=extent_undeclared),
    never accepted, never green, exit 4. Both halves are asserted here, and the second is
    the one the old shape could not have: the OTHER entry in the same register is still
    ruled on.
    """
    bad = _entry(id="r", expect="red", signature="x", _no_extent=True,
                 cmd=["python", "-c", "import sys; print('x'); sys.exit(1)"])
    good = _entry(id="g", expect="green", cmd=["python", "-c", "print('hello')"])
    rc, out = _open_reds(tmp_path, [good, bad])
    assert rc == 4, out
    assert "extent_undeclared" in out
    assert "extent" in out and "SUBSTRING" in out
    # The acceptance is not granted: the word ACCOUNTED must not appear against this id.
    assert "ACCOUNTED   r" not in out, out
    # ...and the sibling was measured rather than swallowed by the other entry's defect.
    assert "PASS" in out and "g" in out, out


def test_open_reds_refuses_an_extent_pattern_with_no_capture_group(tmp_path):
    """Zero groups makes `findall` return whole lines, and a set of whole lines is a
    signature again — the arm would look like it was running and be measuring nothing.

    Same move as the arm above (trinity, 2026-08-05): the refusal is now that ENTRY's
    ERROR(instrument), exit 4, rather than a parse error over the whole register. What is
    asserted is unchanged in substance — the malformed extent is refused and the reason
    is named — plus that it is refused for the entry that carries it.
    """
    e = _entry(id="r", expect="red", signature="x", extent={"pattern": "^x$", "members": []},
               cmd=["python", "-c", "import sys; print('x'); sys.exit(1)"])
    rc, out = _open_reds(tmp_path, [e])
    assert rc == 4, out
    assert "capture group" in out
    assert "extent_undeclared" in out
    assert "ACCOUNTED" not in out, out


@pytest.mark.parametrize("field", ["owner", "reason", "closes_when", "review_by", "opened"])
def test_open_reds_refuses_a_partial_entry(tmp_path, field):
    e = _entry(id="r", expect="red", signature="x",
               cmd=["python", "-c", "import sys; sys.exit(1)"])
    e.pop(field)
    rc, out = _open_reds(tmp_path, [e])
    assert rc == 2, out
    assert field in out


def test_open_reds_reports_an_unobserved_colour_as_an_instrument_error(tmp_path):
    """UNOBSERVABLE is not green. A check that could not be run has no colour."""
    rc, out = _open_reds(tmp_path, [_entry(cmd=["definitely-not-a-real-binary-xyz"])])
    assert rc == 4, out
    assert "command_absent" in out
    assert "not a check that passed" in out


def test_open_reds_annotates_an_accepted_red_with_its_owner(tmp_path):
    """A truncated log ate a failing test name on a real merge gate. Annotations are
    check-run metadata and cannot be truncated by log volume."""
    rc, out = _open_reds(tmp_path, [
        _entry(id="r", expect="red", signature="known", owner="mouse",
               cmd=["python", "-c", "import sys; print('known'); sys.exit(1)"]),
    ], env={"OPEN_REDS_FORCE_ANNOTATE": "1"})
    assert rc == 0, out
    assert "::warning title=open red: r::" in out
    assert "mouse" in out


def test_open_reds_list_does_not_claim_a_colour_it_did_not_observe(tmp_path):
    rc, out = _open_reds(tmp_path, [_entry()], extra=["--list"])
    assert rc == 0
    assert "no check was run" in out
    assert "PASS —" not in out


def test_open_reds_has_no_flag_that_suppresses_a_failure(tmp_path):
    """The same rule as suite_floor's absent --relax: a guard that can be turned off by
    an argument is a waiver with a flag."""
    entries = [_entry(cmd=["python", "-c", "import sys; sys.exit(1)"])]
    for flag in ("--relax", "--allow-fail", "--warn-only", "--soft", "--ignore"):
        rc, out = _open_reds(tmp_path, entries, extra=[flag])
        assert rc == 2, f"{flag}: {out}"
        assert "unrecognized arguments" in out


def test_the_shipped_register_narrows_rather_than_accepting_a_whole_suite():
    """Trinity's principle: narrowing is the amplifier, not a cost saving. Accepting
    ci/test_lane_checks.py as red WHOLE would absorb every future red in 135+ tests —
    which is exactly the defect that let a 4th red hide behind '3 known reds'.

    Generalised 2026-08-05 (issue #33): this used to hardcode a single narrow-red
    entry (`lane_checks_census_extent`), which is now `retired` because the three
    node ids it named went green for real and a second, narrower census over the
    same subject is the failure `rust/tools/audit_instruments.py` names. The
    invariant this test actually protects is not that ENTRY's existence, it is that
    every test id `lane_checks_suite` deselects is picked up by SOME other
    expect=red entry over the same file — so a deselected test is always observed
    by exactly one entry in the register, never zero. With no accepted reds over
    ci/test_lane_checks.py currently open, both sides of that equality are empty,
    which is the correct state, not a vacuous one: replanting a red here (see the
    negative-control sibling test) must make the assertion fail again.
    """
    doc = json.loads(OPEN_REDS_REGISTER.read_text(encoding="utf-8"))
    by_id = {c["id"]: c for c in doc["checks"]}
    suite = by_id["lane_checks_suite"]
    assert suite["expect"] == "green"
    deselected = {suite["cmd"][i + 1] for i, a in enumerate(suite["cmd"]) if a == "--deselect"}
    selected: set[str] = set()
    for c in doc["checks"]:
        if c["id"] == "lane_checks_suite" or c["expect"] != "red":
            continue
        selected |= {a for a in c["cmd"] if a.startswith("ci/test_lane_checks.py::")}
    assert deselected == selected, (
        "every test deselected from the green entry must be selected by some red entry, "
        "or a test falls out of both and is observed by neither"
    )


def test_a_deselected_lane_checks_test_with_no_red_entry_is_caught():
    """Negative control for the generalisation above: plant exactly the defect the
    prior, hardcoded version of this test could no longer detect once its one named
    entry retired -- a --deselect with nothing accepting the red it hides."""
    doc = json.loads(OPEN_REDS_REGISTER.read_text(encoding="utf-8"))
    by_id = {c["id"]: c for c in doc["checks"]}
    suite = dict(by_id["lane_checks_suite"])
    suite["cmd"] = suite["cmd"] + [
        "--deselect",
        "ci/test_lane_checks.py::test_planted_with_no_accepted_red",
    ]
    deselected = {suite["cmd"][i + 1] for i, a in enumerate(suite["cmd"]) if a == "--deselect"}
    selected: set[str] = set()
    for c in doc["checks"]:
        if c["id"] == "lane_checks_suite" or c["expect"] != "red":
            continue
        selected |= {a for a in c["cmd"] if a.startswith("ci/test_lane_checks.py::")}
    assert deselected != selected, "the planted defect must be visible to the comparison"
    assert (
        "ci/test_lane_checks.py::test_planted_with_no_accepted_red" in (deselected - selected)
    )


def test_every_accepted_red_in_the_shipped_register_names_who_can_close_it():
    doc = json.loads(OPEN_REDS_REGISTER.read_text(encoding="utf-8"))
    for c in doc["checks"]:
        if c["expect"] != "red":
            continue
        assert c["owner"] and c["owner"] != "link", (
            f"{c['id']}: an accepted red owned by nobody but the person who accepted it "
            "is not owned"
        )
        assert len(c["closes_when"]) > 40, f"{c['id']}: closes_when must name an event"
        assert len(c["signature"]) >= 8, f"{c['id']}: signature too short to be specific"


def test_every_known_limit_is_admitted_by_the_screen_that_owns_it():
    """A limit that lives only in a register is prose with punctuation.

    Two limitations of my own tooling sat in module docstrings this quarter — Switch's
    substring finding and the census's shallow-clone blindness — and both stayed open
    until somebody tripped over them. `known_limits` is checked, not listed: each entry
    names a command that makes the screen ITSELF admit the limit by token and exit 1.
    """
    doc = json.loads(OPEN_REDS_REGISTER.read_text(encoding="utf-8"))
    limits = doc.get("known_limits")
    assert limits, "the register must have a known_limits section, even if empty on purpose"
    ids = [lim["id"] for lim in limits]
    assert len(set(ids)) == len(ids), "duplicate known_limit id"
    for lim in limits:
        for field in ("id", "screen", "cmd", "admits", "owner", "opened", "review_by"):
            assert lim.get(field), f"{lim.get('id', '?')}: known limit is missing {field!r}"
        assert len(lim["limit"]) > 120, f"{lim['id']}: a limit needs saying, not naming"
        assert len(lim["closes_when"]) > 80, (
            f"{lim['id']}: a limit with no closing condition is a limit nobody can retire, "
            "which is how the two prose ones survived"
        )
        assert (REPO_ROOT / lim["screen"]).is_file(), f"{lim['id']}: screen {lim['screen']} missing"


def test_a_known_limit_is_not_filed_as_an_accepted_red():
    """The two categories are different in kind and the register must keep them apart.

    An accepted red is a failing check somebody OTHER than its acceptor closes — the test
    above this one enforces `owner != link` for exactly that reason. A known limit is a
    bounded gap in a screen held by the screen's own author, which is the case that rule
    forbids. Filing one as the other would have meant either weakening that rule or
    writing an `owner` I did not mean.
    """
    doc = json.loads(OPEN_REDS_REGISTER.read_text(encoding="utf-8"))
    limit_ids = {lim["id"] for lim in doc.get("known_limits", [])}
    check_ids = {c["id"] for c in doc["checks"]}
    assert not (limit_ids & check_ids), (
        f"{limit_ids & check_ids} appear in both checks and known_limits; one subject, two "
        "registers, and the arithmetic in each is wrong about the other"
    )


def test_the_register_says_what_it_did_not_look_at():
    """R12 applied to the register: a check omitted on purpose and a check nobody thought
    of look identical unless the omission is written down."""
    doc = json.loads(OPEN_REDS_REGISTER.read_text(encoding="utf-8"))
    assert doc["not_declared_here"], "the register must state its own exclusions"
    for subject, why in doc["not_declared_here"].items():
        assert len(why) > 60, f"{subject}: an exclusion needs a reason, not a shrug"


def test_open_reds_refuses_a_subject_that_left_the_register(tmp_path):
    """THE DEFECT THIS SCREEN'S FIRST REAL USER FOUND IN IT.

    Mouse repaired three of five accepted reds and deleted their entries, which the file
    told him to do. One of the three was not actually green -- seven of eight uninvoked
    accessors had been wired and an eighth had not -- so deleting the entry removed the
    CHECK, not just the acceptance. The denominator went 8 -> 5 and the screen printed
    PASS with a red check in the tree it had stopped looking at.

    Same shape as check_suite_productivity's target_ran_nothing: a sum cannot see one of
    its terms go silent.
    """
    rc, out = _open_reds(
        tmp_path, [_entry(id="still_here")],
        doc_over={"subjects": ["still_here", "quietly_deleted"]},
    )
    assert rc == 2, out
    assert "quietly_deleted" in out
    assert "does not leave this register by being deleted" in out


def test_open_reds_accepts_a_subject_that_was_retired_on_purpose(tmp_path):
    """The other polarity: a deliberate retirement must not be a failure, or the rule
    above would just forbid ever removing anything."""
    rc, out = _open_reds(
        tmp_path, [_entry(id="still_here")],
        doc_over={
            "subjects": ["still_here", "gone_on_purpose"],
            "retired": {"gone_on_purpose": {
                "owner": "mouse", "date": "2026-08-03", "reason": "fixed at 2832526",
            }},
        },
    )
    assert rc == 0, out
    assert "RETIRED gone_on_purpose" in out


def test_open_reds_refuses_a_retirement_with_no_reason(tmp_path):
    rc, out = _open_reds(
        tmp_path, [_entry(id="still_here")],
        doc_over={
            "subjects": ["still_here", "gone"],
            "retired": {"gone": {"owner": "mouse", "date": "2026-08-03"}},
        },
    )
    assert rc == 2, out
    assert "reason" in out


def test_open_reds_refuses_a_check_missing_from_subjects(tmp_path):
    """Append-only in the other direction: a check may not be added without being
    recorded, or `subjects` stops being the record of what this screen is responsible
    for and the arithmetic above becomes decorative."""
    rc, out = _open_reds(
        tmp_path, [_entry(id="declared"), _entry(id="sneaked_in")],
        doc_over={"subjects": ["declared"]},
    )
    assert rc == 2, out
    assert "sneaked_in" in out


def test_open_reds_refuses_a_subject_that_is_both_live_and_retired(tmp_path):
    rc, out = _open_reds(
        tmp_path, [_entry(id="both")],
        doc_over={
            "subjects": ["both"],
            "retired": {"both": {"owner": "x", "date": "2026-08-03", "reason": "y"}},
        },
    )
    assert rc == 2, out
    assert "both live and retired" in out


def test_open_reds_frame_states_the_arithmetic_it_is_checking(tmp_path):
    rc, out = _open_reds(
        tmp_path, [_entry(id="a"), _entry(id="b")],
        doc_over={
            "subjects": ["a", "b", "c"],
            "retired": {"c": {"owner": "x", "date": "2026-08-03", "reason": "y"}},
        },
    )
    assert rc == 0, out
    assert "3 subject(s) ever declared = 2 ruled on now" in out
    assert "1 retired" in out


def test_the_shipped_register_tells_you_to_flip_not_delete():
    """The instruction that produced the defect is now the instruction that prevents it."""
    doc = json.loads(OPEN_REDS_REGISTER.read_text(encoding="utf-8"))
    assert "DO NOT DELETE" in doc["how_to_remove_an_entry"]
    live = {c["id"] for c in doc["checks"]}
    assert set(doc["subjects"]) == live | set(doc.get("retired", {})), (
        "every subject must be live or retired"
    )


# ---------------------------------------------------------------------------
# check_main_is_green.py
#
# THE SECOND-ORDER ARM. Everything else in this file screens a lane; this screens
# whether anybody READ a lane. CI was red on `main` for ten consecutive pushes while
# every merge report quoted a green local gate set, because nothing in the workflow
# asked GitHub. `--from-json` exists so both polarities are exercised with no network:
# a screen that can only be tried against the real branch has exactly one polarity on
# any given day, which is the condition it was built to abolish.
# ---------------------------------------------------------------------------


def _runs(*rows) -> list:
    out = []
    for i, (status, conclusion) in enumerate(rows):
        out.append({
            "status": status,
            "conclusion": conclusion,
            "headSha": f"{i:040x}",
            "displayTitle": f"run {i}",
            "url": f"https://example.invalid/runs/{i}",
            "workflowName": "CI",
            "createdAt": "2026-08-04T00:00:00Z",
        })
    return out


def _main_green(tmp_path, rows, *args: str):
    p = write(tmp_path, "runs.json", rows)
    r = run_check("check_main_is_green.py", "--from-json", str(p), *args)
    return r.returncode, r.stdout + r.stderr


def test_main_green_all_succeeded_passes(tmp_path):
    """The positive pole. A screen with no green state is a screen nobody can satisfy."""
    rc, out = _main_green(tmp_path, _runs(("completed", "success"), ("completed", "success")))
    assert rc == EXIT_PASS, out
    assert "0 RED" in out
    assert "PASS: every completed run" in out


def test_main_green_one_failure_is_a_condition_and_names_the_url(tmp_path):
    """The red must arrive with a URL: an unlocatable red is the one nobody opens."""
    rc, out = _main_green(tmp_path, _runs(("completed", "success"), ("completed", "failure")))
    assert rc == EXIT_FAIL_CONDITION, out
    assert "FAIL(condition=main_is_red)" in out
    assert "https://example.invalid/runs/1" in out


@pytest.mark.parametrize("conclusion", ["failure", "timed_out", "cancelled", "startup_failure"])
def test_main_green_every_non_success_conclusion_is_red(tmp_path, conclusion):
    """`cancelled` is not `success`. A screen that only knows the word `failure` is an
    allowlist of one word, and the next word through is invisible."""
    rc, out = _main_green(tmp_path, _runs(("completed", conclusion)))
    assert rc == EXIT_FAIL_CONDITION, out
    assert conclusion in out


def test_main_green_in_progress_is_not_counted_red_but_is_declared(tmp_path):
    """A run that has not finished has not failed; saying so is not the same as ignoring it."""
    rc, out = _main_green(tmp_path, _runs(("completed", "success"), ("in_progress", None)))
    assert rc == EXIT_PASS, out
    assert "1 run(s) have not finished" in out
    assert "1 still running" in out


def test_main_green_an_unreadable_answer_is_an_instrument_error_not_a_green(tmp_path):
    """The whole point. `I could not ask` must never render as `the answer was yes` —
    the ten red pushes all happened to people holding a green local transcript."""
    r = run_check("check_main_is_green.py", "--from-json", str(tmp_path / "absent.json"))
    assert r.returncode == EXIT_ERROR_INSTRUMENT, r.stdout + r.stderr
    assert "ERROR(instrument=github_unreachable)" in r.stdout
    assert "UNOBSERVABLE, not" in r.stdout


def test_main_green_zero_runs_is_unscreened_not_clean(tmp_path):
    """A branch with no runs has not passed; an empty list is a denominator of zero."""
    rc, out = _main_green(tmp_path, [])
    assert rc == EXIT_ERROR_INSTRUMENT, out
    assert "no_runs_listed" in out


def test_main_green_merge_sentence_carries_the_sha_and_url_in_both_colours(tmp_path):
    """`--for-merge` exists so the sentence in a merge report cannot be written from
    memory. If it can be paraphrased without a URL, we are back where we started."""
    rc, out = _main_green(tmp_path, _runs(("completed", "success")), "--for-merge")
    assert rc == EXIT_PASS and "MERGE REPORT SENTENCE: `main` is GREEN" in out
    assert "https://example.invalid/runs/0" in out
    rc, out = _main_green(tmp_path, _runs(("completed", "failure")), "--for-merge")
    assert rc == EXIT_FAIL_CONDITION and "is RED" in out
    assert "https://example.invalid/runs/0" in out


# ══════════════════════════════════════════════════════════════════════════════════════════
# rust/tools/probe_ledger_loss.py — the ledger-loss probe's DESTINATION and PROVENANCE
# contract (issue #14)
#
# The probe's ARMS were already two-polarity; what had never been tested is where it WRITES.
# It wrote its six working files and a `result.json` straight into a tracked directory, so
# running the diagnostic during a read-only baseline dirtied `main` on 2026-08-05, and the
# committed reading — unframed, unowned, produced on somebody else's checkout — went stale
# behind a `pass=true` that no instrument in this repository could see.
#
# Every test below is a pair in the sense the module docstring means: the destination that
# must be allowed and the destination that must be REFUSED, with the token asserted rather
# than the exit code alone.
# ══════════════════════════════════════════════════════════════════════════════════════════

PROBE = REPO_ROOT / "rust" / "tools" / "probe_ledger_loss.py"

#: The directory the probe used to write into unconditionally. It is TRACKED-surface (git does
#: not ignore it) and, since issue #14, holds no committed reading at all.
CANONICAL_DIR_REL = "bench/results/_probe_ledger_loss"

#: A drive-letter path or a POSIX absolute path into a home/tmp/mount root — the same shape
#: `probe_ledger_loss._ABSOLUTE_PATH_RE` refuses, restated here on purpose. A test that imported
#: the tool's own regex would agree with it by construction and could not notice it going wrong.
_MACHINE_PATH_RE = re.compile(
    r"(?:[A-Za-z]:[\\/])|(?:(?:^|[\s\"'(=])/(?:home|Users|users|mnt|tmp|var|opt|root)/)"
)

#: PR #51's two high-confidence bypasses, restated as spelling functions rather than literal
#: strings baked into one test, so every alias-of-a-tracked-path test below can ask for either
#: spelling of any destination. Both are Windows path *namespaces* -- alternate ways the OS lets
#: a caller name a file that already has an ordinary drive-letter name -- and neither is
#: normalised by `pathlib.Path.resolve()` or `os.path.realpath()` (verified empirically against
#: this checkout at review time: both left the prefix/UNC form untouched), which is exactly why
#: `out.resolve().relative_to(repo.resolve())` used to raise `ValueError` on them and the
#: `except ValueError` branch answered the permissive question (OUTSIDE) instead of consulting
#: git at all.
_WINDOWS_ONLY = pytest.mark.skipif(
    sys.platform != "win32",
    reason="\\\\?\\ extended-length and \\\\host\\C$\\ admin-share path namespaces are Windows-only",
)


def _extended_length_path(rel: str) -> str:
    r"""The literal `\\?\` extended-length spelling of `REPO_ROOT / rel`.

    This is the exact spelling the rejected PR #51 head demonstrated as bypass #1:
    `--out \\?\C:\...\onnxruntime-ep-vulkan-14\evidence` exited 0, ran 7/7, and wrote into the
    tracked `evidence/` directory.
    """
    return "\\\\?\\" + str(REPO_ROOT / rel)


def _localhost_admin_share_path(rel: str) -> str:
    r"""The literal `\\localhost\C$\...` administrative-share spelling of `REPO_ROOT / rel`.

    This is the exact spelling the rejected PR #51 head demonstrated as bypass #2:
    `--out \\localhost\C$\...\evidence` exited 0, ran 7/7, and wrote into the tracked
    `evidence/` directory. `localhost` (rather than the machine's own hostname) is the literal
    spelling from the review -- it resolves to this same machine's loopback SMB server on any
    Windows host with File and Printer Sharing enabled, which is why the alias is dangerous: it
    needs no configuration a reviewer would notice was missing.
    """
    full = REPO_ROOT / rel
    drive_letter = full.drive.rstrip(":")
    rest = str(full)[len(full.drive) + 1 :]  # strip the "C:\" prefix, keep the rest verbatim
    return f"\\\\localhost\\{drive_letter}$\\{rest}"


def run_probe(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run the probe the way a lane does: same interpreter, utf-8 pinned on the child's side."""
    child_env = dict(os.environ)
    child_env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, str(PROBE), *args],
        cwd=str(cwd or REPO_ROOT),
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=child_env,
    )


def _porcelain(paths: list[str] | None = None) -> str:
    return subprocess.run(
        ["git", "status", "--porcelain", "--", *(paths or [])],
        cwd=str(REPO_ROOT),
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    ).stdout


def test_ledger_loss_probe_default_run_is_green_and_reports_every_arm():
    """The positive arm. Seven arms, all of them, against this checkout's real evidence."""
    r = run_probe()
    assert r.returncode == EXIT_PASS, r.stdout + r.stderr
    assert "PASS: 7/7 arms" in r.stdout
    # Arm 3 is the load-bearing one and its absence must not hide inside the total.
    assert "[PASS] 3 the real eb84364 loss is detected" in r.stdout


def test_ledger_loss_probe_default_run_leaves_the_worktree_exactly_as_it_found_it():
    """THE DEFECT ISSUE #14 IS ABOUT, as a test rather than as a story.

    Not `git status` is empty — this suite must pass in a dirty working tree — but that the
    probe changed nothing: the porcelain before and after are the same string, and the
    directory it used to write into is untouched.
    """
    before = _porcelain()
    r = run_probe()
    assert r.returncode == EXIT_PASS, r.stdout + r.stderr
    assert _porcelain() == before, (
        "the probe changed the worktree; a diagnostic that cannot run without leaving a "
        "tracked diff is a diagnostic people stop running"
    )
    assert not (REPO_ROOT / CANONICAL_DIR_REL).exists(), (
        f"{CANONICAL_DIR_REL} was recreated by an ordinary run"
    )
    assert "removed on exit" in r.stdout


@pytest.mark.parametrize(
    "dest",
    [
        CANONICAL_DIR_REL,
        "bench/results",
        "evidence",
        "ci",
        "rust/tools",
        pytest.param(
            _extended_length_path(CANONICAL_DIR_REL),
            marks=_WINDOWS_ONLY,
            id="extended-length-prefix-\\\\?\\",
        ),
        pytest.param(
            _localhost_admin_share_path(CANONICAL_DIR_REL),
            marks=_WINDOWS_ONLY,
            id="localhost-admin-share-\\\\localhost\\C$",
        ),
    ],
)
def test_ledger_loss_probe_refuses_every_tracked_destination_and_writes_nothing(dest):
    r"""The path boundary, on the canonical directory, on four others, and on two Windows path
    *namespace* aliases of the canonical one (PR #51 review).

    A rule tested only on the one path that burned us is an allowlist of one path, and a rule
    tested only on drive-letter-relative spellings cannot see a namespace bypass -- which is
    exactly how PR #51's rejected head passed this test's five original cases while
    `--out \\?\C:\...\evidence` and `--out \\localhost\C$\...\evidence` still exited 0, ran
    7/7 and wrote seven files -- three of them deletion-bearing retirement registers -- into
    the tracked `evidence/` directory. The refusal is `ERROR(instrument=...)` and not a silent
    fallback, because a probe that quietly wrote somewhere else would answer a question nobody
    asked.
    """
    before = _porcelain()
    r = run_probe("--out", dest)
    assert r.returncode == EXIT_ERROR_INSTRUMENT, r.stdout + r.stderr
    assert "ERROR(instrument=refused_tracked_destination)" in r.stdout
    assert "Nothing was written." in r.stdout
    assert _porcelain() == before
    assert not (REPO_ROOT / dest / "result.json").exists()


def test_ledger_loss_probe_refuses_the_canonical_directory_by_name_not_by_luck():
    """The accidental-canonical-write arm. The refusal must NAME the path, so a reader of a
    log knows which destination was refused rather than that `a` destination was."""
    r = run_probe("--out", CANONICAL_DIR_REL)
    assert r.returncode == EXIT_ERROR_INSTRUMENT
    assert CANONICAL_DIR_REL in r.stdout
    assert "--record" in r.stdout, "the refusal must say what the deliberate path is"


@_WINDOWS_ONLY
@pytest.mark.parametrize(
    "spelling",
    [
        pytest.param(_extended_length_path, id="extended-length-prefix-\\\\?\\"),
        pytest.param(_localhost_admin_share_path, id="localhost-admin-share-\\\\localhost\\C$"),
    ],
)
def test_ledger_loss_probe_refuses_a_nonexistent_child_of_a_tracked_alias(spelling):
    """A destination that does not exist YET, under a namespace alias of a tracked directory.

    The ordinary `--out DIR` case is `DIR` not existing until the probe creates it -- the
    identity walk must resolve through the *missing* path components (`evidence/not/made`)
    up to the first REAL ancestor before it can ask whether that ancestor is `repo`. This is
    the same walk as the plain-spelling case, but exercised through a namespace `.resolve()`
    cannot see through, so a fix that only special-cased the exact rejected `evidence/`
    leaf would not be caught by it.
    """
    before = _porcelain()
    dest = spelling("evidence/not/made/yet")
    r = run_probe("--out", dest)
    assert r.returncode == EXIT_ERROR_INSTRUMENT, r.stdout + r.stderr
    assert "ERROR(instrument=refused_tracked_destination)" in r.stdout
    assert _porcelain() == before
    assert not (REPO_ROOT / "evidence" / "not").exists(), (
        "the probe must not create ANY part of a refused destination, not even an empty parent"
    )


@_WINDOWS_ONLY
@pytest.mark.parametrize(
    "spelling",
    [
        pytest.param(_extended_length_path, id="extended-length-prefix-\\\\?\\"),
        pytest.param(_localhost_admin_share_path, id="localhost-admin-share-\\\\localhost\\C$"),
    ],
)
def test_ledger_loss_probe_record_through_a_windows_namespace_alias_still_writes(spelling):
    r"""The OTHER polarity for the same two namespaces: `--record` through a `\\?\` or
    `\\localhost\C$\` alias of a tracked directory must still be ALLOWED and must still
    write a full, correctly-classified reading.

    A destination policy that refuses a namespace alias unconditionally (rather than
    classifying it correctly and then applying the SAME `--record` escape as the
    plain-spelling case) would silently break the one deliberate-recording path the
    project actually wants, while looking like the same fix from the outside.
    """
    dest = spelling(CANONICAL_DIR_REL)
    before = _porcelain()
    try:
        r = run_probe("--out", dest, "--record")
        assert r.returncode == EXIT_PASS, r.stdout + r.stderr
        doc = json.loads((REPO_ROOT / CANONICAL_DIR_REL / "result.json").read_text("utf-8"))
        assert doc["recorded"] is True
        assert doc["pass"] is True
        assert (REPO_ROOT / CANONICAL_DIR_REL / "artifact-frame.json").is_file()
    finally:
        shutil.rmtree(REPO_ROOT / CANONICAL_DIR_REL, ignore_errors=True)
    assert _porcelain() == before, "cleanup must restore the porcelain this test itself saw"


@_WINDOWS_ONLY
@pytest.mark.parametrize(
    "spelling",
    [
        pytest.param(_extended_length_path, id="extended-length-prefix-\\\\?\\"),
        pytest.param(_localhost_admin_share_path, id="localhost-admin-share-\\\\localhost\\C$"),
    ],
)
def test_ledger_loss_probe_allows_a_git_ignored_destination_through_a_windows_namespace_alias(
    spelling,
):
    r"""`target/` is git-ignored under its plain spelling; it must STILL be git-ignored, and
    therefore still ALLOWED, when named through a `\\?\` or `\\localhost\C$\` alias -- the
    identity walk must reach the same `DEST_IGNORED` verdict `git check-ignore` would give the
    plain spelling, not merely happen to refuse less on the tracked side."""
    real_out = REPO_ROOT / "target" / "_probe_ledger_loss_alias_test_out"
    shutil.rmtree(real_out, ignore_errors=True)
    try:
        before = _porcelain()
        dest = spelling("target/_probe_ledger_loss_alias_test_out")
        r = run_probe("--out", dest)
        assert r.returncode == EXIT_PASS, r.stdout + r.stderr
        assert (real_out / "result.json").is_file()
        assert _porcelain() == before
    finally:
        shutil.rmtree(real_out, ignore_errors=True)


def test_ledger_loss_probe_record_without_a_destination_is_error_instrument_not_usage():
    """`--record` is a permission, not a path. Letting it pick one would recreate the defect
    with an extra flag in front of it.

    R13 reconciliation (PR #51 review, item 3): this print an `ERROR(instrument=...)` token
    from the start, but used to exit `2` (usage) -- a code that disagreed with its own
    vocabulary. `2` is reserved for the argument parser rejecting the command line itself; this
    is a semantic refusal the parser accepted just fine, so it is `EXIT_ERROR_INSTRUMENT`.
    """
    r = run_probe("--record")
    assert r.returncode == EXIT_ERROR_INSTRUMENT, r.stdout + r.stderr
    assert "ERROR(instrument=record_without_destination)" in r.stdout


def test_ledger_loss_probe_an_unparseable_ledger_is_error_instrument_not_a_raw_crash(tmp_path):
    """R13 reconciliation (PR #51 review, item 3): a malformed `evidence/proof_ledger.jsonl`
    used to propagate as an uncaught `JSONDecodeError` -- a Python traceback on stderr and
    the interpreter's own default `exit(1)`, indistinguishable on the exit code alone from
    `FAIL(condition)`. The probe must reach an `ERROR(instrument=...)` token and exit 4
    instead, because a crash is not a detection.

    A throwaway `--repo` (never a git checkout) is enough: `main()` raises out of
    `run_arms()` before anything downstream needs `git`.
    """
    fake_repo = tmp_path / "fake_repo"
    (fake_repo / "evidence").mkdir(parents=True)
    (fake_repo / "evidence" / "proof_ledger.jsonl").write_text("{not valid json", encoding="utf-8")
    r = run_probe("--repo", str(fake_repo))
    assert r.returncode == EXIT_ERROR_INSTRUMENT, r.stdout + r.stderr
    assert "ERROR(instrument=ledger_unreadable)" in r.stdout
    assert "Traceback" not in r.stdout and "Traceback" not in r.stderr, (
        "a raw traceback must never reach a lane's log for an ordinary parse failure"
    )


def test_ledger_loss_probe_writes_its_reading_only_under_an_explicit_out(tmp_path):
    """The caller-provided-output arm: outside the repository, so no permission is needed."""
    out = tmp_path / "scratch"
    r = run_probe("--out", str(out))
    assert r.returncode == EXIT_PASS, r.stdout + r.stderr
    doc = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert doc["pass"] is True and doc["arms_total"] == 7
    assert doc["recorded"] is False, "an --out run is a reading, not a recording"


def test_ledger_loss_probe_allows_a_git_ignored_destination_inside_the_repository(tmp_path):
    """The other side of the path boundary. `target/` is inside the checkout and git ignores
    it, so writing there cannot dirty anything — refusing it would be a rule about location
    when the rule is about the TRACKED SURFACE."""
    out = REPO_ROOT / "target" / "_probe_ledger_loss_test_out"
    shutil.rmtree(out, ignore_errors=True)
    try:
        before = _porcelain()
        r = run_probe("--out", str(out))
        assert r.returncode == EXIT_PASS, r.stdout + r.stderr
        assert (out / "result.json").is_file()
        assert _porcelain() == before
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_ledger_loss_probe_output_is_byte_deterministic_across_runs(tmp_path):
    """Two runs, two different destinations, identical bytes.

    This is what makes the record comparable at all: if the reading moved with the directory
    it was written into, a re-run would look like a change in the ledger.
    """
    first, second = tmp_path / "a", tmp_path / "b"
    assert run_probe("--out", str(first)).returncode == EXIT_PASS
    assert run_probe("--out", str(second)).returncode == EXIT_PASS
    assert (first / "result.json").read_bytes() == (second / "result.json").read_bytes()


def test_ledger_loss_probe_record_carries_subject_and_tool_provenance(tmp_path):
    """A recorded reading must say WHO took it, WITH WHAT, AT WHICH COMMIT and ABOUT WHAT.

    The committed reading this replaces had none of those four: it was `{"arms": [...],
    "pass": true}` and nothing else, so `bfdc0f1` retiring two Conv keys could not make it
    detectably stale.
    """
    out = tmp_path / "recorded"
    r = run_probe("--out", str(out), "--record")
    assert r.returncode == EXIT_PASS, r.stdout + r.stderr
    doc = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert doc["owner"] == "tank"
    assert doc["tool"] == "rust/tools/probe_ledger_loss.py"
    assert doc["recorded"] is True
    assert re.fullmatch(r"[0-9a-f]{40}", doc["produced_at_commit"]), doc["produced_at_commit"]
    for name, rel in (
        ("ledger", "evidence/proof_ledger.jsonl"),
        ("attempts", "evidence/proof_attempts.jsonl"),
        ("register", "evidence/retired_proof_keys.json"),
    ):
        assert doc["subject"][name]["path"] == rel
        assert re.fullmatch(r"[0-9a-f]{64}", doc["subject"][name]["sha256"])

    frame = json.loads((out / "artifact-frame.json").read_text(encoding="utf-8"))
    assert "result.json" in frame["files"], "a recorded reading that no frame names is unframed"
    assert "evidence/proof_ledger.jsonl" in frame["subject_paths"]
    assert "rust/tools/probe_ledger_loss.py" in frame["subject_paths"], (
        "a change to the arms changes what pass=true means, so the tool is part of the subject"
    )


def test_ledger_loss_probe_record_names_no_machine_specific_absolute_path(tmp_path):
    """The exact defect the committed reading carried: arm details naming an absolute path
    under one developer's home directory, on a machine nobody else has."""
    out = tmp_path / "recorded"
    assert run_probe("--out", str(out), "--record").returncode == EXIT_PASS
    text = (out / "result.json").read_text(encoding="utf-8")
    hit = _MACHINE_PATH_RE.search(text)
    assert hit is None, f"absolute path in the recorded reading: {text[hit.start():hit.end() + 60]!r}"
    assert str(tmp_path) not in text and str(REPO_ROOT) not in text
    assert "<out>" in text, "the scratch paths in the arm details must be scrubbed, not deleted"


def test_ledger_loss_probe_record_bytes_are_lf_ascii_and_platform_independent(tmp_path):
    """Windows-path/encoding control.

    Two things break a record on Windows and neither is visible in a `json.loads` of it: a
    CRLF newline translation on the way to disk, and a non-ASCII arm detail encoded in the
    shell's cp1252. Both are asserted at the BYTE level here, because both survive parsing.
    """
    out = tmp_path / "recorded"
    assert run_probe("--out", str(out), "--record").returncode == EXIT_PASS
    raw = (out / "result.json").read_bytes()
    assert b"\r" not in raw, "platform newline translation reached the record"
    raw.decode("ascii")  # ensure_ascii=True: no cp1252/utf-8 ambiguity can enter the bytes
    assert raw.endswith(b"\n")


def test_ledger_loss_probe_survives_a_destination_with_spaces_and_non_ascii(tmp_path):
    """A Windows path is allowed to contain spaces and non-ASCII, and the reading must not
    change because of it — the scrubber replaces the destination root, so the bytes are the
    same as a plain ASCII destination's."""
    plain = tmp_path / "plain"
    awkward = tmp_path / "scratch dir ünïcode"
    assert run_probe("--out", str(plain)).returncode == EXIT_PASS
    r = run_probe("--out", str(awkward))
    assert r.returncode == EXIT_PASS, r.stdout + r.stderr
    assert (awkward / "result.json").read_bytes() == (plain / "result.json").read_bytes()


def test_ledger_loss_probe_fails_loud_when_its_subject_is_not_there(tmp_path):
    """The negative control for the probe ITSELF: it must be able to go red.

    A checkout carrying a ledger and a retirement register but NO attempt log is exactly the
    state arm 6 exists for, and every arm that needs the attempt log must report the outage
    rather than 'nothing is missing'. Note what stays green: arm 6, which PREDICTS the
    outage. A run in which everything failed would prove only that the probe crashed.
    """
    fake = tmp_path / "checkout"
    (fake / "evidence").mkdir(parents=True)
    for rel in ("evidence/proof_ledger.jsonl", "evidence/retired_proof_keys.json"):
        shutil.copy2(REPO_ROOT / rel, fake / rel)
    r = run_probe("--repo", str(fake))
    assert r.returncode == EXIT_FAIL_CONDITION, r.stdout + r.stderr
    assert "[FAIL] 1 current tree is clean" in r.stdout
    assert "[PASS] 6 a missing attempt log is ERROR(instrument)" in r.stdout
    assert "FAIL: 1/7 arms" in r.stdout


def test_ledger_loss_probe_reports_an_unreachable_ledger_as_an_outage_not_a_verdict(tmp_path):
    """No evidence at all is UNOBSERVABLE. A probe that answered `0 arms passed` there would
    be reporting about a repository it never read."""
    r = run_probe("--repo", str(tmp_path))
    assert r.returncode == EXIT_FAIL_CONDITION, r.stdout + r.stderr
    assert "ERROR(instrument)" in r.stdout
    assert "evidence/proof_ledger.jsonl" in r.stdout


def test_ledger_loss_probe_classify_destination_never_reads_uncertainty_as_outside(monkeypatch):
    """UNIT-LEVEL control, direct-import, complementing (not replacing) the subprocess tests
    above -- it asserts the exact PROPERTY the PR #51 review named: "never infer outside-repo
    safety from ValueError", generalised to "never infer it from any unresolvable `stat`".

    `os.stat` is monkeypatched to raise `PermissionError` -- a real, if rare, Windows failure
    mode (an ACL that denies traversal partway down `--out`'s path) that is emphatically NOT
    "this path segment does not exist yet". `classify_destination` must refuse closed
    (`DEST_TRACKED_SURFACE`), never answer `DEST_OUTSIDE`, when it cannot complete the walk.
    """
    sys.path.insert(0, str((REPO_ROOT / "rust" / "tools")))
    import probe_ledger_loss as pll  # type: ignore  # noqa: PLC0415

    real_stat = os.stat
    denied_marker = REPO_ROOT / "evidence" / "some_deep" / "unreadable" / "child"

    def _flaky_stat(path, *a, **kw):
        if str(path) == str(denied_marker):
            raise PermissionError(13, "Permission denied", str(path))
        return real_stat(path, *a, **kw)

    monkeypatch.setattr(pll.os, "stat", _flaky_stat)
    result = pll.classify_destination(REPO_ROOT, denied_marker)
    assert result.kind == pll.DEST_TRACKED_SURFACE, (
        f"a stat the instrument could not complete must refuse closed, got {result.kind!r} "
        "-- this is the exact substitution ('unresolvable' read as 'safe') PR #51 was rejected "
        "for, restated for a permission failure instead of a namespace ValueError"
    )


def test_ledger_loss_probe_classify_destination_resolves_windows_namespace_aliases_by_identity():
    """UNIT-LEVEL companion to the subprocess-level parametrized refusal test: asserts the
    PROPERTY (file identity survives a namespace rewrite) the subprocess tests exercise
    end-to-end, directly against the function PR #51's review named.
    """
    sys.path.insert(0, str((REPO_ROOT / "rust" / "tools")))
    import probe_ledger_loss as pll  # type: ignore  # noqa: PLC0415
    import pathlib as _pathlib

    plain = pll.classify_destination(REPO_ROOT, REPO_ROOT / "evidence")
    assert plain.kind == pll.DEST_TRACKED_SURFACE

    if sys.platform == "win32":
        ext = pll.classify_destination(
            REPO_ROOT, _pathlib.Path(_extended_length_path("evidence"))
        )
        unc = pll.classify_destination(
            REPO_ROOT, _pathlib.Path(_localhost_admin_share_path("evidence"))
        )
        assert ext.kind == pll.DEST_TRACKED_SURFACE and ext.repo_relative == "evidence"
        assert unc.kind == pll.DEST_TRACKED_SURFACE and unc.repo_relative == "evidence"



def test_the_ledger_loss_probe_leaves_no_tracked_reading_behind():
    """The ownership model, asserted rather than described.

    ONE model: the probe is EXECUTED (host-free lane, `ledger_loss_probe` in
    ci/open_reds.json), not committed. A tracked reading reappearing here is the second,
    staler answer to a question already asked live on every push — and it is precisely what
    went unnoticeably out of date.
    """
    tracked = subprocess.run(
        ["git", "ls-files", "--", CANONICAL_DIR_REL],
        cwd=str(REPO_ROOT),
        capture_output=True,
        encoding="utf-8",
    ).stdout.strip()
    assert tracked == "", (
        f"{CANONICAL_DIR_REL} has tracked files again: {tracked!r}. If a recorded reading is "
        "genuinely wanted, it needs an artifact-frame entry in ci/open_reds.json and an owner, "
        "not a bare commit."
    )


@pytest.mark.parametrize(
    "rel",
    [
        "evidence/retired_proof_keys.json",
        "evidence/proof_ledger.jsonl",
        "evidence/proof_attempts.jsonl",
        CANONICAL_DIR_REL + "/result.json",
    ],
)
def test_deletion_bearing_evidence_is_never_union_merged(rel):
    """Union merge cannot represent a DELETION, and every file here carries one.

    Retiring a proof key removes a claim's exemption; union-merging the register resurrects
    the key the next time a branch forked before the retirement lands, with nobody's
    signature on it. That is the `squad-history` defect one directory over
    (.gitattributes), and the probe writes register-shaped files, so its output path is
    asserted here too rather than left to be someone's assumption.
    """
    out = subprocess.run(
        ["git", "check-attr", "merge", "--", rel],
        cwd=str(REPO_ROOT),
        capture_output=True,
        encoding="utf-8",
    ).stdout
    assert "union" not in out, f"{rel} is union-merged: {out.strip()!r}"


def test_ledger_loss_probe_is_declared_in_the_register_the_lane_and_the_inventory():
    """The ownership model has three halves and all three must exist, or the probe is back to
    being a tool nobody runs.

    An unwired tool is invisible to the coverage census by construction
    (ci/check_verification_subjects.py's own note), which is why the register is the screen
    that closes the gap: the register RUNS things.
    """
    reg = json.loads((CI_DIR / "open_reds.json").read_text(encoding="utf-8"))
    entry = next((c for c in reg["checks"] if c["id"] == "ledger_loss_probe"), None)
    assert entry is not None, "ledger_loss_probe is not in ci/open_reds.json"
    assert entry["expect"] == "green" and entry["owner"] == "tank"
    assert entry["cmd"] == ["python", "rust/tools/probe_ledger_loss.py"], (
        "the register must run the probe in its DEFAULT mode — that is the mode the "
        "non-mutating contract is about"
    )
    assert "ledger_loss_probe" in reg["subjects"], "`subjects` is the append-only record"

    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "rust/tools/probe_ledger_loss.py" in workflow

    inventory = (CI_DIR / "lane_inventory.py").read_text(encoding="utf-8")
    assert 'id="hostfree.ledger_loss_probe"' in inventory


# ===========================================================================
# ci/with_icd_suppressed.py -- issue #1 / PR #45 applied to the lane itself. The Windows
# ICD-removal negative control armed itself with VK_ICD_FILENAMES/VK_DRIVER_FILES, which
# a High-integrity process drops (PLATFORMS.md 7.4.1), so on every hosted run it printed
# ERROR(instrument=icd_suppression_ineffective) and exited 0: a step that executed, was
# green, and asserted NOTHING. This wrapper tries every declared mechanism -- including
# the registry one, which is not gated by process integrity -- and runs the child only
# INSIDE an arm whose own loader probe reported the ICD gone.
#
# Two polarities on the property that matters: an arm that took must carry its
# environment all the way to the probe (the dead-arm defect Morpheus rejected PR #45
# for), and a run where no arm took must NOT run the child and must NOT report a pass.
# ===========================================================================

WITH_ICD_SUPPRESSED = CI_DIR / "with_icd_suppressed.py"

# A stand-in for `epctl --probe-loader`. It answers from its own os.environ, so the only
# way it can report the ICD gone is if the arm's environment genuinely reached it.
_FAKE_PROBE = """\
import os, sys
sys.stdout.reconfigure(encoding="utf-8")
gone = os.environ.get("VK_DRIVER_FILES", "").endswith("no_such_icd.json")
if os.environ.get("FAKE_PROBE_ALWAYS_PRESENT"):
    gone = False
n = 0 if gone else 2
print("[vulkan-ep] INFO:   %d device(s) passed the \\u00a77.2 capability gate." % n)
sys.exit(1 if n == 0 else 0)
"""

# A stand-in for `ci/gate_chain_fp32.py` on a suppressed host: it must be able to prove
# it ran, and it exits non-zero on purpose, exactly like the real gate does.
_FAKE_CHILD = """\
import os, sys
open(os.environ["FAKE_CHILD_MARKER"], "w").write(os.environ.get("VK_DRIVER_FILES", ""))
print("gate: FAIL(condition=UNATTRIBUTED)")
sys.exit(9)
"""


def _icd_wrapper(tmp_path, *extra, arms="driver_search_path,loader_driver_filter", child=True):
    probe = tmp_path / "fake_probe.py"
    probe.write_text(_FAKE_PROBE, encoding="utf-8")
    marker = tmp_path / "child_ran.txt"
    argv = [
        sys.executable,
        str(WITH_ICD_SUPPRESSED),
        "--probe",
        sys.executable,
        str(probe),
        "--probe-report",
        str(tmp_path / "probe.txt"),
        "--record-out",
        str(tmp_path / "record.json"),
        *extra,
    ]
    if arms:
        argv += ["--arms", arms]
    if child:
        kid = tmp_path / "fake_child.py"
        kid.write_text(_FAKE_CHILD, encoding="utf-8")
        argv += ["--", sys.executable, str(kid)]
    else:
        argv += ["--", sys.executable, "-c", "pass"]
    env = dict(os.environ, FAKE_CHILD_MARKER=str(marker))
    proc = subprocess.run(argv, capture_output=True, text=True, cwd=str(REPO_ROOT), env=env)
    record = {}
    if (tmp_path / "record.json").exists():
        record = json.loads((tmp_path / "record.json").read_text(encoding="utf-8"))
    return proc, record, marker


def test_with_icd_suppressed_runs_the_child_inside_the_arm_that_took(tmp_path):
    """The pair, green polarity: an arm removes the ICD, the child runs under it, and the
    stamp names WHICH mechanism -- never 'something worked'."""
    proc, record, marker = _icd_wrapper(tmp_path)
    out = proc.stdout + proc.stderr
    assert proc.returncode == EXIT_PASS, out
    assert record["suppression_mechanism"] == "driver_search_path", record
    assert "[ICD-SUPPRESSED] mechanism=driver_search_path child_exit=9" in out, out
    assert marker.exists(), "the child never ran, but the wrapper reported PASS"
    assert record["child_exit_code"] == 9, record


def test_with_icd_suppressed_delivers_the_arms_environment_to_the_child(tmp_path):
    """The defect Morpheus rejected PR #45 for, screened here: entering an arm without
    binding its environment spawns an unmodified child. The marker file records what the
    CHILD actually read out of its own environment, so a dropped arm cannot pass."""
    proc, record, marker = _icd_wrapper(tmp_path)
    assert proc.returncode == EXIT_PASS, proc.stdout + proc.stderr
    assert marker.read_text(encoding="utf-8").endswith("no_such_icd.json"), (
        "the child ran without the arm's environment: the wrapper suppressed nothing"
    )
    took = [a for a in record["suppression_attempts"] if a["mechanism"] == "driver_search_path"][0]
    assert took["env_applied"].get("VK_DRIVER_FILES", "").endswith("no_such_icd.json"), took


def test_with_icd_suppressed_red_polarity_no_arm_took_is_not_a_pass(tmp_path):
    """The red polarity, and the whole reason this exists: when nothing removes the ICD
    the child must NOT run and the wrapper must NOT exit 0. The old step exited 0 here."""
    proc, record, marker = _icd_wrapper(tmp_path, "--nonexistent-icd", "x", child=True)
    out = proc.stdout + proc.stderr
    assert proc.returncode == EXIT_ERROR_INSTRUMENT, out
    assert "ERROR(instrument=no_mechanism_suppressed_the_icd)" in out, out
    assert not marker.exists(), "the child ran even though the ICD was still present"
    assert record["suppression_mechanism"] is None, record
    assert record["child_exit_code"] is None, record


def test_with_icd_suppressed_attempts_every_declared_arm_before_giving_up(tmp_path):
    """A mechanism can only be said to have failed if it was tried. When none takes, the
    attempted list must equal the declared list -- including the registry arm, whatever
    this host lets it do."""
    proc, record, _ = _icd_wrapper(
        tmp_path,
        "--nonexistent-icd",
        "x",
        arms="driver_search_path,loader_driver_filter,registry_disable",
    )
    assert proc.returncode == EXIT_ERROR_INSTRUMENT, proc.stdout + proc.stderr
    attempted = [a["mechanism"] for a in record["suppression_attempts"]]
    assert attempted == record["arms_declared"], record
    registry = [a for a in record["suppression_attempts"] if a["mechanism"] == "registry_disable"]
    assert registry and registry[0]["state"] in {
        "mechanism_unavailable",
        "icd_suppression_ineffective",
        "probe_report_unreadable",
        "suppressed",
    }, registry


def test_with_icd_suppressed_disagreeing_witnesses_are_unreadable_not_suppressed(tmp_path):
    """`0 devices passed` printed by a probe that exited 0 is two readers disagreeing.
    That is an instrument state; taking the convenient one is how a lane learns to lie."""
    probe = tmp_path / "liar.py"
    probe.write_text(
        "import sys\n"
        'sys.stdout.reconfigure(encoding="utf-8")\n'
        'print("[vulkan-ep] INFO:   0 device(s) passed the \\u00a77.2 capability gate.")\n'
        "sys.exit(0)\n",
        encoding="utf-8",
    )
    marker = tmp_path / "child_ran.txt"
    kid = tmp_path / "kid.py"
    kid.write_text(_FAKE_CHILD, encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            str(WITH_ICD_SUPPRESSED),
            "--probe",
            sys.executable,
            str(probe),
            "--probe-report",
            str(tmp_path / "probe.txt"),
            "--record-out",
            str(tmp_path / "record.json"),
            "--arms",
            "driver_search_path",
            "--",
            sys.executable,
            str(kid),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=dict(os.environ, FAKE_CHILD_MARKER=str(marker)),
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode == EXIT_ERROR_INSTRUMENT, out
    assert not marker.exists(), "the child ran on a report two readers could not agree on"
    record = json.loads((tmp_path / "record.json").read_text(encoding="utf-8"))
    assert record["suppression_attempts"][0]["state"] == "probe_report_unreadable", record


def test_with_icd_suppressed_a_probe_that_cannot_launch_is_not_a_suppression(tmp_path):
    """'The binary is missing' and 'the ICD is gone' produce the same silence. Only one
    of them means the control reached its subject."""
    proc, record, marker = _icd_wrapper(
        tmp_path, arms="driver_search_path"
    )
    assert proc.returncode == EXIT_PASS  # sanity: the pair's other polarity
    proc = subprocess.run(
        [
            sys.executable,
            str(WITH_ICD_SUPPRESSED),
            "--probe",
            str(tmp_path / "no_such_epctl.exe"),
            "--probe-report",
            str(tmp_path / "probe2.txt"),
            "--record-out",
            str(tmp_path / "record2.json"),
            "--arms",
            "driver_search_path",
            "--",
            sys.executable,
            "-c",
            "pass",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == EXIT_ERROR_INSTRUMENT, proc.stdout + proc.stderr
    record2 = json.loads((tmp_path / "record2.json").read_text(encoding="utf-8"))
    assert record2["suppression_attempts"][0]["state"] == "probe_report_unreadable", record2


def test_with_icd_suppressed_no_child_is_a_usage_error_not_a_silent_pass(tmp_path):
    """A wrapper with nothing to wrap must not report that something ran."""
    proc = subprocess.run(
        [
            sys.executable,
            str(WITH_ICD_SUPPRESSED),
            "--probe",
            sys.executable,
            "--probe-report",
            str(tmp_path / "probe.txt"),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "usage" in (proc.stdout + proc.stderr)


# ---------------------------------------------------------------------------
# The landing simulation and its gate — issue #60.
#
# The residual these cover: a `rewitness/3` record is corroborated against the two trees
# the ledger moved between, which is why it survives squash, merge and rebase. It does not
# survive somebody landing an edit to a declared `caused_by_content` path BETWEEN the
# declaration and the merge — the branch head passes, GitHub's two-parent merge ref passes,
# and only the SQUASH is red, which is a red that arrives on main after the merge button.
# ---------------------------------------------------------------------------


def _landsim():
    import importlib

    sys.path.insert(0, str(CI_DIR))
    return importlib.import_module("check_landing_simulation")


def _simulator():
    import importlib

    sys.path.insert(0, str(CI_DIR))
    return importlib.import_module("simulate_squash_rewitness")


def _sim_git(args, cwd, **kw):
    env = dict(os.environ)
    env.update(
        GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t",
        GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t",
    )
    return subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True,
        encoding="utf-8", errors="replace", env=env, **kw
    )


def _two_sided_repo(tmp_path):
    """base and pr both edit `cause.txt`; only the base edits `other.txt`.

    -> (repo, base_sha, pr_sha). This is issue #60's shape reduced to two files.
    """
    repo = tmp_path / "twosided"
    repo.mkdir()
    _sim_git(["init", "-q", "-b", "main"], repo)
    (repo / "cause.txt").write_text("original\n", encoding="utf-8")
    (repo / "other.txt").write_text("original\n", encoding="utf-8")
    _sim_git(["add", "-A"], repo)
    _sim_git(["commit", "-q", "-m", "merge base"], repo)
    _sim_git(["branch", "pr"], repo)

    (repo / "cause.txt").write_text("original\nfrom the base\n", encoding="utf-8")
    (repo / "other.txt").write_text("moved on the base only\n", encoding="utf-8")
    _sim_git(["add", "-A"], repo)
    _sim_git(["commit", "-q", "-m", "the concurrent landing"], repo)
    base = _sim_git(["rev-parse", "HEAD"], repo).stdout.strip()

    _sim_git(["checkout", "-q", "pr"], repo)
    (repo / "cause.txt").write_text("from the branch\noriginal\n", encoding="utf-8")
    _sim_git(["add", "-A"], repo)
    _sim_git(["commit", "-q", "-m", "the branch's edit"], repo)
    pr = _sim_git(["rev-parse", "HEAD"], repo).stdout.strip()
    _sim_git(["checkout", "-q", "main"], repo)
    return repo, base, pr


def test_the_simulated_squash_is_the_merge_result_not_an_overlay(tmp_path):
    """The pin that stops the blindness from coming back.

    `git checkout <pr> -- .` + `git add -A` produces a DIFFERENT TREE from `git merge
    --squash` whenever the base has moved, and the difference is exactly the base's edit
    being dropped. This asserts the landing the simulator builds has the same tree as an
    honest `git merge`, and — the other polarity — that the overlay does not, so the
    assertion above cannot be satisfied by both.
    """
    sim = _simulator()
    repo, base, pr = _two_sided_repo(tmp_path)

    landing, why = sim._land(repo, base, pr, "squash")
    assert landing and not why, why
    built = _sim_git(["rev-parse", f"{landing}^{{tree}}"], repo).stdout.strip()

    _sim_git(["checkout", "-q", "-B", "honest", base], repo)
    m = _sim_git(["merge", "-q", "--no-ff", "-m", "honest", pr], repo)
    assert m.returncode == 0, m.stdout + m.stderr
    honest = _sim_git(["rev-parse", "HEAD^{tree}"], repo).stdout.strip()
    assert built == honest, (
        "the simulated squash does not carry the merge result. That is issue #60's "
        "blindness: a landing shape that drops the base's edit is green on the very "
        "collision the simulation exists to catch"
    )

    _sim_git(["checkout", "-q", "-B", "overlay", base], repo)
    _sim_git(["checkout", pr, "--", "."], repo)
    _sim_git(["add", "-A"], repo)
    _sim_git(["commit", "-q", "-m", "overlay"], repo)
    overlay = _sim_git(["rev-parse", "HEAD^{tree}"], repo).stdout.strip()
    assert overlay != honest, (
        "the fixture does not distinguish the two landing shapes, so the assertion above "
        "would pass for the wrong reason"
    )


def test_the_simulated_squash_has_one_parent_and_unancestors_the_branch(tmp_path):
    """A squash is one commit with the base as sole parent — that IS the mechanism. If the
    branch stayed an ancestor, `witness_transitions` would still resolve `after` to a branch
    commit and the simulation would reproduce the merge, not the squash."""
    sim = _simulator()
    repo, base, pr = _two_sided_repo(tmp_path)
    landing, why = sim._land(repo, base, pr, "squash")
    assert landing and not why, why
    parents = _sim_git(["rev-list", "--parents", "-n", "1", landing], repo).stdout.split()
    assert len(parents) == 2 and parents[1] == base, parents
    assert _sim_git(["merge-base", "--is-ancestor", pr, landing], repo).returncode != 0


def test_a_landing_that_drops_a_path_the_base_moved_is_refused_not_screened(tmp_path):
    """Non-vacuity, stated as a refusal, and ASSERTED ON THE REFUSAL.

    The first version of this test built an overlay landing and checked only that the overlay
    dropped `other.txt` — a property of `git checkout -- .`, not of this module. It never
    called the guard, so the branch it is named after could have been deleted outright and it
    would still have passed. It now runs the guard on both landing shapes: the overlay must be
    REFUSED and must name the dropped path, and the real squash must be accepted.
    """
    sim = _simulator()
    repo, base, pr = _two_sided_repo(tmp_path)

    _sim_git(["checkout", "-q", "-B", "overlay_landing", base], repo)
    _sim_git(["checkout", "-q", pr, "--", "."], repo)
    _sim_git(["add", "-A"], repo)
    _sim_git(["commit", "-q", "-m", "overlay"], repo)
    overlay = _sim_git(["rev-parse", "HEAD"], repo).stdout.strip()
    _sim_git(["checkout", "-q", "main"], repo)

    assert sim.lost_base_edits(repo, base, pr, overlay) == ["other.txt"], (
        "the overlay was expected to drop the base-only edit; if it no longer does, the "
        "non-vacuity guard has nothing to guard and this test is measuring nothing"
    )
    refusal = sim._refuse_if_vacuous(repo, base, pr, overlay, "squash")
    assert refusal and "other.txt" in refusal and "dropped 1 path" in refusal, refusal

    honest, why = sim._land(repo, base, pr, "squash")
    assert honest and not why, why
    assert sim.lost_base_edits(repo, base, pr, honest) == []
    assert sim._refuse_if_vacuous(repo, base, pr, honest, "squash") == ""


def test_the_refusal_is_wired_into_land_and_produces_no_landing(tmp_path, monkeypatch):
    """The guard existing is not the guard running. `_land` must return ("", why) — no
    landing, no census, no colour — when the guard refuses, because screening a tree nobody
    will ever have is worse than not screening."""
    sim = _simulator()
    repo, base, pr = _two_sided_repo(tmp_path)
    monkeypatch.setattr(sim, "_refuse_if_vacuous", lambda *a, **k: "planted refusal")
    landed, why = sim._land(repo, base, pr, "squash")
    assert landed == "" and why == "planted refusal"


def test_the_lost_set_is_vacuous_on_a_merge_ref_and_real_on_the_branch_head(tmp_path):
    """ISSUE #60's SECOND BLOCKER, as a measurement.

    `refs/pull/N/merge` has the base as its FIRST parent, so `merge-base(base, mergeref)` is
    the base itself, `moved_on_base` is empty and the merge-window guard evaluates ZERO paths
    on the one event it exists for. The engine used to take `git rev-parse HEAD`, which on a
    `pull_request` event is that merge ref. Same repository, same landing, same guard — the
    only difference is which commit is called the head.
    """
    sim = _simulator()
    repo, base, pr = _two_sided_repo(tmp_path)

    _sim_git(["checkout", "-q", "-B", "gh_merge_ref", base], repo)
    m = _sim_git(["merge", "-q", "--no-ff", "-m", "Merge pull request", pr], repo)
    assert m.returncode == 0, m.stdout + m.stderr
    merge_ref = _sim_git(["rev-parse", "HEAD"], repo).stdout.strip()

    _sim_git(["checkout", "-q", "-B", "overlay_landing", base], repo)
    _sim_git(["checkout", "-q", pr, "--", "."], repo)
    _sim_git(["add", "-A"], repo)
    _sim_git(["commit", "-q", "-m", "overlay"], repo)
    overlay = _sim_git(["rev-parse", "HEAD"], repo).stdout.strip()
    _sim_git(["checkout", "-q", "main"], repo)

    assert sim.lost_base_edits(repo, base, merge_ref, overlay) == [], (
        "the merge ref is expected to make the guard vacuous — if it no longer does, this "
        "test has stopped demonstrating the defect it pins"
    )
    assert sim.lost_base_edits(repo, base, pr, overlay) == ["other.txt"], (
        "with the REAL branch head the same landing is refused. That is the whole of the "
        "--head plumbing: the gate resolves this commit and the engine must use it"
    )


def test_the_simulator_does_not_mutate_a_repository_it_was_handed(tmp_path):
    """`_land` checks branches out and used to delete `pr` and `replay` unconditionally.

    Deleting them is right in a fan-out clone — while those refs exist the census's revision
    walk can still reach the un-ancestored branch commits — and it is a destructive edit to
    whatever repository the function is given. It is now opt-in, and refused outright on a
    dirty tree.
    """
    sim = _simulator()
    repo, base, pr = _two_sided_repo(tmp_path)
    landed, why = sim._land(repo, base, pr, "squash")
    assert landed and not why, why
    assert _sim_git(["rev-parse", "--verify", "-q", "pr"], repo).returncode == 0, (
        "the default cleanup deleted a ref in the caller's repository"
    )

    (repo / "cause.txt").write_text("uncommitted work\n", encoding="utf-8")
    landed2, why2 = sim._land(repo, base, pr, "squash")
    assert landed2 == "" and "uncommitted changes" in why2, why2
    assert (repo / "cause.txt").read_text(encoding="utf-8") == "uncommitted work\n", (
        "the refusal must come BEFORE the checkout, or it is a refusal after the damage"
    )


def test_the_gates_corroboration_inputs_are_derived_shader_sources_not_evidence():
    """What the gate's path set IS, on this repository's real register.

    Named for what it asserts: this reads `corroboration_inputs`, not `decide`, so it says
    nothing about REQUIRED/NOT-REQUIRED. The two polarities of the decision are asserted by
    `test_the_gate_decides_required_only_when_a_rule_actually_fires` below and by
    `ci/negative_control_landing_simulation.py`'s gate arms.
    """
    gate = _landsim()
    inputs, _notes = gate.corroboration_inputs(REPO_ROOT, "HEAD")
    if not inputs:
        pytest.skip("no live rewitness/3 record, so there is nothing to gate on")
    assert "rust/shaders/glsl/templates/q_gemv.comp" in inputs, sorted(inputs)
    assert not any(
        p.startswith(("evidence/", "bench/", "docs/", "ci/", ".github/", ".squad/"))
        for p in inputs
    ), f"a corroboration input reads generated evidence: {sorted(inputs)}"


def test_the_gate_derives_its_paths_from_the_register_rather_than_a_list():
    """A hard-coded path set goes stale the first time build.rs moves a directory, and a
    stale set fails OPEN: the gate declines to run on the record it exists for.

    Asserted on the CODE rather than on the file, because the prose in this module names the
    shader issue #60 was reproduced on and should keep naming it.
    """
    gate = _landsim()
    tree = ast.parse((CI_DIR / "check_landing_simulation.py").read_text(encoding="utf-8"))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc = node.body[0] if node.body else None
            if isinstance(doc, ast.Expr) and isinstance(doc.value, ast.Constant):
                docstrings.add(id(doc.value))
    named = [
        n.value for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
        and id(n) not in docstrings and "q_gemv" in n.value
    ]
    assert not named, (
        f"the gate names a specific shader in its logic ({named}), so it stops covering "
        "the next record written"
    )
    src = (CI_DIR / "check_landing_simulation.py").read_text(encoding="utf-8")
    assert "caused_by_content" in src and "source_closure" in src


def test_an_unresolvable_base_is_an_instrument_error_never_a_skip():
    """The failure the issue asked for by name: fail loudly when the base is unavailable."""
    gate = _landsim()
    with pytest.raises(gate.GateInstrumentError) as exc:
        gate.resolve_base("refs/heads/no-such-ref-a8f3", REPO_ROOT)
    assert exc.value.token == "base_unavailable"


def test_the_gate_reads_a_pull_request_merge_ref_as_the_branch_head(tmp_path):
    """On a `pull_request` event the checkout is `refs/pull/N/merge`, whose first parent IS
    the base. Taking it at face value makes `what moved on the base` empty on the one event
    the gate exists for, and the gate then never fires."""
    gate = _landsim()
    repo, base, pr = _two_sided_repo(tmp_path)
    _sim_git(["checkout", "-q", "-B", "gh-merge-ref", base], repo)
    m = _sim_git(["merge", "-q", "--no-ff", "-m", "Merge pull request", pr], repo)
    assert m.returncode == 0, m.stdout + m.stderr
    head, note = gate.resolve_head(base, repo)
    assert head == pr, f"{head} != {pr} ({note})"
    assert "merge" in note.lower()


def test_the_landing_simulation_is_declared_in_the_register_the_lane_and_the_inventory():
    """The three places a check has to appear before it counts, asserted together because
    appearing in two of them is the shape that reads as covered and is not."""
    import importlib

    sys.path.insert(0, str(CI_DIR))
    inv = importlib.import_module("lane_inventory")

    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "ci/check_landing_simulation.py" in workflow
    assert "ci/negative_control_landing_simulation.py" in workflow

    ids = {c.id for c in inv.CHECKS}
    assert "hostfree.landing_simulation" in ids
    assert "hostfree.landing_simulation_negative_control" in ids

    reds = json.loads((CI_DIR / "open_reds.json").read_text(encoding="utf-8"))
    declared = {c["id"]: c for c in reds["checks"]}
    for name in ("landing_simulation", "landing_simulation_negative_control"):
        assert name in declared, f"{name} is in the lane and in no register"
        assert declared[name]["expect"] == "green"
        assert name in reds["subjects"]


def test_the_landing_simulation_step_fetches_without_a_depth():
    """Issue #28's trap, one level down: a --depth fetch marks its own graft point even
    inside a fetch-depth: 0 checkout, and both the landing construction and the census
    denominator are then truncated at it."""
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    start = workflow.index("ci/check_landing_simulation.py")
    window = workflow[max(0, start - 1600):start]
    fetch = window.rindex("git fetch")
    assert "--depth" not in window[fetch:], (
        "the landing-simulation step fetches with a depth, which shallow-poisons the whole "
        "checkout for every screen after it"
    )


# ---------------------------------------------------------------------------
# The head the gate resolves must be the head the engine measures — issue #60, blocker B2.
#
# `check_landing_simulation.resolve_head` reads `HEAD^2` on a `pull_request` merge ref, so
# its own rule R2 is not vacuous. Until this revision it did not PASS that commit to
# `ci/simulate_squash_rewitness.py`, which took `git rev-parse HEAD` — the merge ref — and so
# rebuilt the identical vacuity one process down: `merge-base(base, HEAD)` collapses to the
# base, the merge-window `lost` guard evaluates zero paths, and the engine prints "this run
# cannot exhibit the merge-window collision" directly above the merge-window collision.
# ---------------------------------------------------------------------------


def _landing_step_block() -> str:
    """The `run:`/`env:` text of the landing-simulation step, isolated from its neighbours."""
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    start = workflow.index(
        "- name: Landing simulation (would this branch survive the squash that lands it?)"
    )
    end = workflow.index("- name: Landing-simulation negative control", start)
    return workflow[start:end]


def test_the_gate_forwards_the_head_it_resolved_to_the_engine():
    """The plumbing itself, asserted on the call rather than on prose.

    Both the flag and the VALUE matter: `--head HEAD` would satisfy a substring check and
    would reintroduce the defect exactly, because the engine resolves `HEAD` against the
    checkout — which on the event this exists for is the merge ref.
    """
    tree = ast.parse((CI_DIR / "check_landing_simulation.py").read_text(encoding="utf-8"))
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and any(
            isinstance(a, ast.Constant) and isinstance(a.value, str)
            and "simulate_squash_rewitness.py" in a.value
            for a in ast.walk(n)
        )
        and isinstance(n.func, ast.Attribute) and n.func.attr == "run"
    ]
    assert len(calls) == 1, f"expected exactly one engine invocation, found {len(calls)}"
    argv = ast.unparse(calls[0].args[0])
    assert "'--head'" in argv, f"the engine is invoked without --head: {argv}"
    assert "decision['head']" in argv.replace('"', "'"), (
        f"--head is passed a value that is not the head the gate resolved: {argv}"
    )
    assert "'--base'" in argv and "decision['base']" in argv.replace('"', "'"), argv


def test_the_engine_accepts_a_head_and_refuses_one_it_cannot_resolve():
    """`--head` is validated, never a fallback to the checkout. Exit 2, with the token."""
    sim_path = CI_DIR / "simulate_squash_rewitness.py"
    r = subprocess.run(
        [sys.executable, str(sim_path), "--head", "0" * 40, "--base", "HEAD"],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
        encoding="utf-8", errors="replace",
        env=dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1"),
    )
    assert r.returncode == 2, f"exit={r.returncode}\n{r.stdout}\n{r.stderr}"
    assert "ERROR(instrument=head_unavailable)" in r.stdout, r.stdout
    assert "--head" in subprocess.run(
        [sys.executable, str(sim_path), "--help"], capture_output=True, text=True,
    ).stdout


def test_an_unresolvable_head_is_an_instrument_error_in_the_gate_too():
    gate = _landsim()
    base = gate._rev("HEAD", REPO_ROOT)
    with pytest.raises(gate.GateInstrumentError) as exc:
        gate.resolve_head(base, REPO_ROOT, "refs/heads/no-such-head-a8f3")
    assert exc.value.token == "head_unavailable"


def test_the_gate_reads_a_direct_push_checkout_as_its_own_head(tmp_path):
    """The other topology: `push:` events check out the branch itself, and there is no second
    parent to prefer. A gate that reached for `HEAD^2` unconditionally would be wrong here."""
    gate = _landsim()
    repo, base, pr = _two_sided_repo(tmp_path)
    _sim_git(["checkout", "-q", pr], repo)
    head, note = gate.resolve_head(base, repo)
    assert head == pr, note
    assert "branch head" in note


def test_a_merge_commit_that_is_not_a_pull_request_merge_ref_is_its_own_head(tmp_path):
    """The false positive the merge-ref rule must not have.

    Merging the base INTO the branch — which this very PR did, rather than rebasing — also
    produces a two-parent HEAD. But its FIRST parent is the branch, not the base, so it is
    the branch head and `HEAD^2` would name the base. The rule keys on the first parent for
    exactly this reason; without that condition, updating a branch from main would make the
    gate measure main against main.
    """
    gate = _landsim()
    repo, base, pr = _two_sided_repo(tmp_path)
    _sim_git(["checkout", "-q", "-B", "branch_with_main_merged", pr], repo)
    m = _sim_git(["merge", "-q", "--no-ff", "-m", "Merge origin/main into the branch", base], repo)
    assert m.returncode == 0, m.stdout + m.stderr
    merged = _sim_git(["rev-parse", "HEAD"], repo).stdout.strip()
    head, note = gate.resolve_head(base, repo)
    assert head == merged, f"{head} != {merged} ({note})"
    assert head != base


def _decide_with_inputs(gate, monkeypatch, paths, repo, base, head=None):
    """`decide` with the corroboration path set pinned, so the RULES are what is measured.

    The real path set comes from the register and a shader `#include` closure; a fixture
    repository has neither, and building one would make this a test of the census. The rule
    logic — which paths moved on which side, and which rule that fires — is the subject.
    """
    monkeypatch.setattr(gate, "corroboration_inputs", lambda _r, _rev: (set(paths), ["pinned"]))
    return gate.decide("fixture", base, repo, head)


def test_the_gate_decides_required_only_when_a_rule_actually_fires(tmp_path, monkeypatch):
    """Both polarities of `decide`, isolated to rule R2: the same branch, the same rules, and
    only the base differs. This is the assertion the old test's name promised."""
    gate = _landsim()
    repo, base, pr = _two_sided_repo(tmp_path)

    hot = _decide_with_inputs(gate, monkeypatch, {"cause.txt"}, repo, base, pr)
    assert hot["required"] is True
    assert any(r.startswith("R2") for r in hot["reasons"]), hot["reasons"]
    assert hot["head"] == pr and hot["base"] == base
    assert hot["base_moved_paths"] == 2

    cold = _decide_with_inputs(gate, monkeypatch, {"nothing/here.txt"}, repo, base, pr)
    assert cold["required"] is False and cold["reasons"] == []
    assert cold["base_moved_paths"] == 2, (
        "the base still moved; NOT-REQUIRED here is an argument about relevance, not a claim "
        "that nothing landed"
    )


def test_the_gate_measures_the_branch_head_not_the_merge_ref_checkout(tmp_path, monkeypatch):
    """R2 on a `pull_request` checkout. With the merge ref taken at face value the base has
    moved zero paths and R2 can never fire; with `HEAD^2` it fires on the planted collision.
    Same repository, same register, same rules — only which commit is called the head."""
    gate = _landsim()
    repo, base, pr = _two_sided_repo(tmp_path)
    _sim_git(["checkout", "-q", "-B", "gh_merge_ref", base], repo)
    m = _sim_git(["merge", "-q", "--no-ff", "-m", "Merge pull request", pr], repo)
    assert m.returncode == 0, m.stdout + m.stderr
    merge_ref = _sim_git(["rev-parse", "HEAD"], repo).stdout.strip()

    resolved = _decide_with_inputs(gate, monkeypatch, {"cause.txt"}, repo, base)
    assert resolved["head"] == pr, resolved["head_note"]
    assert resolved["base_moved_paths"] == 2 and resolved["required"] is True
    assert any(r.startswith("R2") for r in resolved["reasons"]), resolved["reasons"]

    naive = _decide_with_inputs(gate, monkeypatch, {"cause.txt"}, repo, base, merge_ref)
    assert naive["base_moved_paths"] == 0, (
        "the merge ref is expected to make the base-moved set empty — if it no longer does, "
        "this test has stopped demonstrating why HEAD^2 is resolved at all"
    )
    assert not any(r.startswith("R2") for r in naive["reasons"]), naive["reasons"]


def test_a_head_already_on_the_base_says_there_is_no_landing(tmp_path, monkeypatch):
    """`push: branches: [main]`, and any re-run after the merge. `git merge --squash` would
    build an empty commit and the engine would report SIM INVALID about the topology rather
    than about the declaration, so the gate says so instead of falling through to a quiet
    NOT-REQUIRED that reads exactly like a gate that has stopped working."""
    gate = _landsim()
    repo, base, pr = _two_sided_repo(tmp_path)
    _sim_git(["checkout", "-q", "-B", "landed", base], repo)
    m = _sim_git(["merge", "-q", "--no-ff", "-m", "landed", pr], repo)
    assert m.returncode == 0, m.stdout + m.stderr
    landed = _sim_git(["rev-parse", "HEAD"], repo).stdout.strip()

    d = _decide_with_inputs(gate, monkeypatch, {"cause.txt"}, repo, landed, pr)
    assert d["head_is_ancestor_of_base"] is True
    assert d["required"] is False
    assert d["no_landing"].startswith("R0") and "R2" in d["no_landing"], d["no_landing"]

    d_same = _decide_with_inputs(gate, monkeypatch, {"cause.txt"}, repo, pr, pr)
    assert d_same["head_is_ancestor_of_base"] is True and d_same["required"] is False


def test_a_stale_base_narrows_the_screen_and_the_decision_says_which_base(tmp_path, monkeypatch):
    """The residual the workflow's fresh `git fetch` exists for, made visible.

    Screened against a base that is BEHIND the tip, `merge-base(base, head)` is the base
    itself, nothing has "moved on the base", and R2 cannot fire even though the collision is
    sitting on the real tip. The decision therefore records the base sha it used: a verdict
    about a base nobody will merge into is not a weaker verdict, it is one about a different
    question, and the operator has to be able to see which.
    """
    gate = _landsim()
    repo, base, pr = _two_sided_repo(tmp_path)
    stale = _sim_git(["merge-base", base, pr], repo).stdout.strip()

    fresh = _decide_with_inputs(gate, monkeypatch, {"cause.txt"}, repo, base, pr)
    assert fresh["required"] is True and fresh["base"] == base

    behind = _decide_with_inputs(gate, monkeypatch, {"cause.txt"}, repo, stale, pr)
    assert behind["base"] == stale and behind["merge_base"] == stale
    assert behind["base_moved_paths"] == 0
    assert not any(r.startswith("R2") for r in behind["reasons"]), behind["reasons"]
    assert any(r.startswith("R3") for r in behind["reasons"]), (
        "R3 must still fire — the branch itself touches the path, and only the base-side "
        "half of the question went blind"
    )


def test_dirty_reads_renames_copies_spaces_and_non_ascii_paths(tmp_path):
    """`--porcelain -z` emits a SECOND, bare-path record after a rename or copy, holding the
    original path. Slicing `[3:]` off every field ate three characters of it and produced a
    path that exists nowhere — silently, and on exactly the changes most likely to move a
    declared cause path out from under a record. Without `-z` the paths would be C-quoted
    instead, which is the same defect wearing quotes.
    """
    gate = _landsim()
    repo = tmp_path / "renames"
    repo.mkdir()
    _sim_git(["init", "-q", "-b", "main"], repo)
    (repo / "a file.txt").write_text("x\n", encoding="utf-8")
    (repo / "\u00e9\u4e2d.txt").write_text("y\n", encoding="utf-8")
    (repo / "plain.txt").write_text("z\n", encoding="utf-8")
    _sim_git(["add", "-A"], repo)
    _sim_git(["commit", "-q", "-m", "init"], repo)

    _sim_git(["mv", "a file.txt", "renamed to.txt"], repo)
    _sim_git(["mv", "\u00e9\u4e2d.txt", "\u00e9\u4e2d moved.txt"], repo)
    (repo / "untracked space.txt").write_text("w\n", encoding="utf-8")
    (repo / "plain.txt").write_text("modified\n", encoding="utf-8")

    dirty = gate._dirty(repo)
    for want in (
        "a file.txt", "renamed to.txt",
        "\u00e9\u4e2d.txt", "\u00e9\u4e2d moved.txt",
        "untracked space.txt", "plain.txt",
    ):
        assert want in dirty, f"{want!r} missing from {sorted(dirty)}"
    assert not any(p.startswith('"') for p in dirty), sorted(dirty)
    assert not any(p in ("ile.txt", "iled to.txt") for p in dirty), (
        f"a record was sliced as if it carried a status prefix: {sorted(dirty)}"
    )


def test_the_landing_step_passes_the_base_ref_through_env_not_interpolation():
    """`${{ }}` is a TEXT substitution performed before the shell sees the script, so a value
    an outside contributor controls — `github.base_ref` is the branch name on the pull
    request — would be pasted in as code. Git ref names permit `;`, `$`, backticks and `&`.
    Through `env:` it is expanded by the shell from the environment, quoted, as one argument.
    """
    block = _landing_step_block()
    run = block[block.index("run: |"):]
    assert "${{" not in run, (
        f"the landing-simulation `run:` block still interpolates a GitHub expression: {run}"
    )
    assert "LANDING_BASE_REF: ${{ github.base_ref || 'main' }}" in block, block
    assert 'git fetch --no-tags origin "$LANDING_BASE_REF"' in run, run
    assert '--summary "$GITHUB_STEP_SUMMARY"' in run, run


def test_the_whole_landing_step_run_block_quotes_every_expansion():
    """Every `$VAR` in that block must be double-quoted. An unquoted one word-splits, which
    turns a ref name with a space into two arguments and a `git fetch` into a fetch of
    something else."""
    run = _landing_step_block()
    run = run[run.index("run: |"):]
    bare = re.findall(r'(?<!")\$(?:\{\w+\}|\w+)', run)
    assert not bare, f"unquoted shell expansions in the landing-simulation step: {bare}"


def test_the_gate_fires_on_the_real_q_gemv_collision_and_names_it(tmp_path):
    """The REAL cause path, the real register, the real `#include` closure — end to end.

    Everything above this pins a mechanism on a two-file fixture. This one plants the issue
    #60 collision on a clone of this repository — one line appended to
    `rust/shaders/glsl/templates/q_gemv.comp`, the declared cause path of the only live
    `rewitness/3` record — and requires the gate to say REQUIRED, name R2, and name the file.
    `--explain-only` is the decision without the two-minute simulation; the full landing
    matrix on this scenario is `ci/negative_control_landing_simulation.py`'s REPLAYED arms.
    """
    gate = _landsim()
    inputs, _notes = gate.corroboration_inputs(REPO_ROOT, "HEAD")
    cause = "rust/shaders/glsl/templates/q_gemv.comp"
    if cause not in inputs:
        pytest.skip("no live rewitness/3 record naming the q_gemv cause path")

    clone = tmp_path / "collision"
    r = _sim_git(["clone", "-q", "--local", "--no-hardlinks", str(REPO_ROOT), str(clone)],
                 tmp_path)
    assert r.returncode == 0, r.stderr
    head = _sim_git(["rev-parse", "HEAD"], REPO_ROOT).stdout.strip()
    _sim_git(["fetch", "-q", "origin", head], clone)
    _sim_git(["checkout", "-q", "-B", "collided_base", head], clone)
    with open(clone / cause, "a", encoding="utf-8", newline="\n") as fh:
        fh.write("// a concurrent main-side edit, planted by the landing-simulation tests\n")
    _sim_git(["add", "-A"], clone)
    _sim_git(["commit", "-q", "-m", "main-side edit to the declared cause path"], clone)
    base = _sim_git(["rev-parse", "HEAD"], clone).stdout.strip()
    _sim_git(["checkout", "-q", "-B", "the_pr", head], clone)
    readme = clone / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8") + "\n<!-- an edit in no shader closure -->\n",
        encoding="utf-8",
    )
    _sim_git(["add", "-A"], clone)
    _sim_git(["commit", "-q", "-m", "an unrelated edit"], clone)

    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    hot = subprocess.run(
        [sys.executable, str(clone / "ci" / "check_landing_simulation.py"),
         "--explain-only", "--base", base],
        cwd=str(clone), capture_output=True, text=True, encoding="utf-8",
        errors="replace", env=env,
    )
    assert hot.returncode == 0, hot.stdout + hot.stderr
    assert "VERDICT: REQUIRED" in hot.stdout, hot.stdout
    assert "R2 " in hot.stdout and cause in hot.stdout, hot.stdout

    cold = subprocess.run(
        [sys.executable, str(clone / "ci" / "check_landing_simulation.py"),
         "--explain-only", "--base", head],
        cwd=str(clone), capture_output=True, text=True, encoding="utf-8",
        errors="replace", env=env,
    )
    assert cold.returncode == 0, cold.stdout + cold.stderr
    assert "VERDICT: NOT-REQUIRED" in cold.stdout, (
        "the SAME branch against a base that did not move the cause path must acquit, or the "
        "arm above is a constant\n" + cold.stdout
    )


def test_an_unreadable_worktree_blob_is_an_instrument_error_not_a_condition(monkeypatch):
    """`WorktreeBlobError` is raised by the census's `_blob_at` when the index and the
    filesystem describe a cause path differently. `ci/check_ledger_census.py` maps it to exit
    2; here it escaped `main()` as a traceback, which a shell reports as exit 1, which this
    lane reads as a CONDITION — a red about the declaration on a day the instrument could not
    read the checkout."""
    gate = _landsim()

    def boom(_repo, _rev):
        raise gate.census.WorktreeBlobError(
            "planted: the index says regular file and the filesystem says symlink"
        )

    monkeypatch.setattr(gate, "corroboration_inputs", boom)
    rc = gate.main(["--explain-only", "--base", "HEAD"])
    assert rc == 2, rc


def test_the_landing_simulation_declares_its_enforcement_follow_up():
    """B1. The step is ADVISORY: this repository has no required status checks at all, so
    nothing invalidates a green landing check when the base moves under it. The follow-up
    that would change that is sequenced behind `main_is_green` and recorded where it can be
    checked rather than remembered."""
    followup = json.loads(
        (CI_DIR / "landing_enforcement_followup.json").read_text(encoding="utf-8")
    )
    assert followup["enforced_today"] is False
    assert followup["blocked_by"] == "main_is_green"
    assert followup["ruleset_id"] == 20479180
    assert followup["observed"]["branch_protection"] == "404 not protected"
    assert followup["observed"]["required_status_checks"] == []

    reds = json.loads((CI_DIR / "open_reds.json").read_text(encoding="utf-8"))
    declared = {c["id"] for c in reds["checks"]}
    assert followup["blocked_by"] in declared, (
        "the follow-up is sequenced behind an open red that is not declared, so nothing "
        "will ever tell anybody it became satisfiable"
    )
    steps = followup["checklist"]
    assert len(steps) >= 4 and all(s["done"] is False for s in steps), steps
    assert any("up to date" in s["step"].lower() for s in steps), steps
    assert any("base" in s["verify"].lower() and "re-run" in s["verify"].lower()
               for s in steps), steps


def test_no_artifact_claims_the_landing_gate_is_enforced():
    """The rejection this revision answers: three artifacts said closing #60 needed
    "require branches to be up to date before merging", which the repository lacks — while
    the repository in fact has no required status checks AT ALL, so the base is unconstrained
    at merge time and GitHub has nothing to invalidate. A document whose stated purpose is
    naming residuals must not name a smaller one."""
    import importlib

    sys.path.insert(0, str(CI_DIR))
    inv = importlib.import_module("lane_inventory")

    checks = {c.id: c for c in inv.CHECKS}
    misses = " ".join(checks["hostfree.landing_simulation"].misses).lower()
    assert "advisory" in misses, misses
    assert "no required status checks" in misses, misses

    design = (REPO_ROOT / "docs" / "DESIGN.md").read_text(encoding="utf-8")
    part5b = design[design.index("A CONCURRENT EDIT TO A DECLARED CAUSE PATH"):]
    part5b = part5b[:part5b.index("§8.9.20") if "§8.9.20" in part5b else 12000]
    assert "no required status checks" in part5b.lower(), (
        "DESIGN.md still describes the missing enforcement as only the up-to-date rule"
    )
    assert "advisory" in part5b.lower(), part5b[-2000:]
