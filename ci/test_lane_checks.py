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

import json
import shutil
import subprocess
import sys
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
EXIT_ERROR_INSTRUMENT = 4


def run_check(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(CI_DIR / script), *args],
        capture_output=True,
        text=True,
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
    """The line that has now appeared five times while every gate passed."""
    p = write(
        tmp_path,
        "suite.log",
        "2026-07-31 ... [E:onnxruntime] EP_FAIL from VulkanExecutionProvider. "
        "Falling back to CPUExecutionProvider.\n196 passed\n",
    )
    r = run_check("check_fatal_log.py", str(p))
    assert r.returncode == EXIT_FAIL_CONDITION, r.stdout
    assert "FAIL(condition=runtime_fallback_announced_by_ort)" in r.stdout
    # R13 second clause: quote the failure text, never the failure count.
    assert "Falling back to CPUExecutionProvider" in r.stdout


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
    assert "ERROR(instrument=ep_library_not_found)" in r.stdout
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
    r = run_ds("--scan", str(tmp_path))
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
        r = run_ds("--scan", str(d))
        assert r.returncode == EXIT_FAIL_CONDITION, drop
        assert "STEADY_UNCERTIFIED" in r.stdout, drop


def test_a_duration_in_prose_is_published_too(tmp_path):
    """A JSON parser cannot see a figure written into the job summary. Second witness,
    different failure mode (R13 obligation 3)."""
    write(tmp_path, "verdict.json", record())
    summary = write(tmp_path, "summary.md", "Steady-state GPU busy 11.525 ms/inference\n")
    r = run_ds("--scan", str(tmp_path), "--summary", str(summary))
    assert r.returncode == EXIT_FAIL_CONDITION
    assert "11.525 ms" in r.stdout


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
