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
