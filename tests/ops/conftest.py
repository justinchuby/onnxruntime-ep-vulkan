"""Pytest configuration for the Vulkan EP op-correctness suite.

The Vulkan execution-provider plugin is registered once per test session from the
``ONNXRUNTIME_VULKAN_EP_LIB`` environment variable (the cargo-built cdylib). Running
``pytest`` without that variable **skips** the entire suite with a clear message rather
than failing, so this is safe to include in any pytest invocation.

Environment variables
---------------------
ONNXRUNTIME_VULKAN_EP_LIB
    Absolute path to the built EP cdylib
    (``rust/target/release/libonnxruntime_vulkan_ep.so`` on Linux,
    ``libonnxruntime_vulkan_ep.dylib`` on macOS,
    ``onnxruntime_vulkan_ep.dll`` on Windows).

ONNXRUNTIME_EP_VULKAN_VALIDATE
    Set to ``1`` to force validation-layer checks in every test session (adds
    ``ep.enable_validation=true`` to all SessionOptions). In CI debug lanes this
    is set automatically; locally it can be opt-in. The EP panics on any Vulkan
    validation error, which surfaces as an ORT exception and fails the test.

VULKAN_DEVICE_INDEX
    Comma-separated list of Vulkan device indices to run tests against.
    Default: "0" (first device). Set to "0,1" to run tests on both devices.
    Example: VULKAN_DEVICE_INDEX=0,1 pytest tests/ops/

    Device selection uses session config entry ``ep.device_index`` (pending
    Switch's implementation). When multiple indices are provided, op tests are
    parametrized and run once per device.

    TODO(Switch): implement ``ep.device_index`` session option in rust/src/ep.rs
    (SessionOptionsAdd hook). Without it, the EP always selects the default device
    and multi-device runs are no-ops. The test infrastructure is wired and ready.

Pytest options
--------------
--vulkan-devices <indices>
    Same as VULKAN_DEVICE_INDEX but as a CLI option. Takes precedence.
    Example: pytest tests/ops/ --vulkan-devices 0,1
"""

from __future__ import annotations

import ast
import functools
import json
import os
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest

import _verdict

EP_NAME = "VulkanExecutionProvider"


# ---------------------------------------------------------------------------
# Portability rules (standing directive 2026-07-29T09:39:59-07:00)
# ---------------------------------------------------------------------------
# Intel is the spec-conformance oracle.
#   Intel Iris Xe (Vulkan 1.4.309, UMA) is the strictest Vulkan implementation
#   available locally. It is device SELECTOR 1 (not 0 — see device mapping note
#   below; 0 is NVIDIA after the discrete-first sort. Corrected 2026-07-30).
#   A failure on Intel that passes on NVIDIA means the EP or test relied on
#   undefined behavior. Intel is correct; the code must change.
#
# Never vendor-special-case.
#   Do not add a vendor-conditional skip or wider tolerance for one GPU. The
#   permitted exception: a filed driver bug, marked with:
#     pytest.mark.xfail(reason="vendor bug: <URL>", strict=True)
#   This makes the xfail visible when the driver is fixed (strict=True).
#
# UMA memory model.
#   Iris Xe (selector 1) exposes DEVICE_LOCAL | HOST_VISIBLE memory, exactly as Adreno/Mali do.
#   --vulkan-devices 0,1 exercises both memory models (discrete vs UMA).
#
# Tests that exercise cross-vendor portability explicitly should be decorated with
# @pytest.mark.portability — this makes them easy to run as a targeted suite:
#   pytest tests/ops/ -m portability -v
#
# Portability failures are NOT harness bugs — route them:
#   - Shader UB → Switch
#   - Claim predicate → Mouse
#   - Memory model assumption → Switch
# ---------------------------------------------------------------------------


def _intel_failure_note(device_index: int) -> str:
    """Return a diagnostic note to append to assertion errors on Intel (device selector 1).

    DEVICE SELECTOR MAPPING (corrected 2026-07-30T21:23:53-07:00 — Tank's finding):
    ``enumerate_capable_devices()`` sorts best-first (discrete before integrated).
    ``select_device`` indexes that sorted list.  The local dev machine sorts as:
      Selector 0 → NVIDIA GeForce RTX 4060 Laptop GPU (discrete, less strict)
      Selector 1 → Intel Iris Xe Graphics (integrated/UMA, spec-conformance oracle)

    A failure on selector 1 (Intel) that passes on selector 0 (NVIDIA) means the EP or
    test relied on undefined behavior.  Intel is correct; the code must change.
    """
    if device_index != 1:
        return ""
    return (
        "\n\n[Portability note] This failure is on device selector 1 (Intel Iris Xe or similar UMA device)."
        "\nIntel Vulkan is the spec-conformance oracle for this project."
        "\nIf this test passes on NVIDIA (selector 0), the EP relied on something unspecified."
        "\nDo NOT widen the tolerance or add a vendor skip — route to Switch or Mouse."
    )


# ---------------------------------------------------------------------------
# Pytest CLI options
# ---------------------------------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--vulkan-devices",
        default=None,
        help=(
            "Comma-separated Vulkan device indices to run op tests against "
            "(e.g. --vulkan-devices 0,1). Overrides VULKAN_DEVICE_INDEX env var. "
            "Default: 0 (first device). "
            "TODO(Switch): requires ep.device_index session option in rust/src/ep.rs."
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "portability: marks tests that explicitly verify cross-vendor/cross-platform behavior. "
        "Run with: pytest -m portability. "
        "An Intel-only failure in a portability test is a spec-conformance bug, not a harness bug.",
    )
    config.addinivalue_line(
        "markers",
        "slow: marks tests that take >10 s (e.g., real model integration tests). "
        "Run with: pytest -m slow. Excluded from the default fast-feedback loop in CI.",
    )
    config.addinivalue_line(
        "markers",
        "expects_ort_fallback: this test deliberately provokes ORT's run-time fallback, so "
        "the lane's 'Falling back' log scan (R13 obligation 3) must not fail it. Every other "
        "test that prints that line is a lane failure regardless of its own assertions.",
    )
    _assert_oracle_versions()


# ---------------------------------------------------------------------------
# Harness truth screen (Guard E) — a claim probe without an established EP proves nothing
# ---------------------------------------------------------------------------
# Morpheus (Round 37, .squad/agents/trinity/history.md §8.9.25): "the one test able to tell
# a transpose from a relabelling lacks the require_vulkan fixture its fifteen neighbours
# carry, silently falls back to CPU, and reports 'EP did not claim' when the EP was never
# loaded." ``is_vulkan_claimed()`` is a deliberately non-asserting probe — its own docstring
# in ``_models.py`` says it returns False BOTH when the EP declines a node form AND when the
# EP is absent or crashes during session creation, because telling those two apart needs a
# claim ATTEMPT to have actually happened. A test that reads a False from it and calls
# ``pytest.skip("EP did not claim ...")`` without first establishing that the EP is even
# registered (``require_vulkan`` / ``vulkan_device_available``) cannot tell a decline from
# an absence, and reports the absence as though it were the more interesting finding.
#
# THIS IS A STRUCTURAL SCREEN, NOT FIVE DECORATORS. Five call sites had exactly this shape
# when it was found (three in test_gemm_and_pool.py, two in test_conv.py); patching those
# five and stopping there would leave the next one silent. So this walks every collected
# test module's OWN source at collection time, and finds every function — test or helper —
# whose body calls ``is_vulkan_claimed`` either directly or through a chain of calls to
# OTHER functions defined in that same module (``_require_structurally_declined`` in
# test_no_cpu_fallback.py is exactly this shape: none of ITS callers call the probe
# directly, they call a helper that does — this screen has to see through that or it would
# have missed a case that happens to already be correctly gated). If a test whose call
# chain reaches the probe does not also depend on one of the two fixtures that establish
# the EP first, collection FAILS — not a skip, not a warning: R13 applies here as much as
# anywhere else in this file, and a harness precondition that quietly let its own violation
# through would be the same defect one layer up.
_CLAIM_PROBE_NAME = "is_vulkan_claimed"
_ESTABLISHING_FIXTURES = frozenset({"require_vulkan", "vulkan_device_available"})


def _called_name(call: ast.Call) -> str | None:
    """The plain name a ``Call`` node invokes, whether bare (``f(...)``) or an attribute
    (``m.f(...)``) — both are how ``is_vulkan_claimed`` is actually spelled in this suite."""
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _claim_probe_taint(tree: ast.Module) -> set[str]:
    """Every top-level function in *tree* whose body calls the claim probe, directly or
    through a chain of calls to other functions defined in the same module.

    Module-local call-graph only, by construction: it cannot see a call to
    ``is_vulkan_claimed`` that arrives through a fixture, a decorator, or a function
    defined elsewhere. That is a stated limit, not a silent one — none of those shapes
    exist for this probe in this suite today (grep-verified against every ``test_*.py``
    under this directory), and a screen that hides its blind spots is worse than none
    (the discipline ``rust/tools/audit_instruments.py`` states for its own text screen).
    """
    funcs = {
        n.name: n
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    calls: dict[str, set[str]] = {}
    tainted: set[str] = set()
    for name, fn in funcs.items():
        callees: set[str] = set()
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            called = _called_name(node)
            if called == _CLAIM_PROBE_NAME:
                tainted.add(name)
            elif called in funcs:
                callees.add(called)
        calls[name] = callees
    grew = True
    while grew:
        grew = False
        for name, callees in calls.items():
            if name not in tainted and callees & tainted:
                tainted.add(name)
                grew = True
    return tainted


@functools.lru_cache(maxsize=None)
def _claim_probe_taint_for_file(path: str) -> frozenset[str]:
    source = Path(path).read_text(encoding="utf-8")
    return frozenset(_claim_probe_taint(ast.parse(source, filename=path)))


def pytest_collection_modifyitems(config: pytest.Config, items: list) -> None:
    """Fail COLLECTION if a test reaches ``is_vulkan_claimed`` without establishing the EP.

    Runs once, over every collected item, regardless of which test file asked for
    collection — a class-level screen rather than a per-test decorator, so a sixth
    violation (in a file that does not exist yet) is caught the same way the first five
    were.
    """
    violations: list[str] = []
    for item in items:
        path = getattr(item, "path", None)
        if path is None or path.suffix != ".py":
            continue
        try:
            tainted = _claim_probe_taint_for_file(str(path))
        except (SyntaxError, OSError):
            # A file this screen cannot parse is a file it did not look at — not silence:
            # any real syntax error will also fail plain collection, loudly, elsewhere.
            continue
        base_name = getattr(item, "originalname", None) or item.name.split("[", 1)[0]
        if base_name not in tainted:
            continue
        if _ESTABLISHING_FIXTURES.isdisjoint(item.fixturenames):
            violations.append(
                f"{item.nodeid}  (reaches {_CLAIM_PROBE_NAME}() without require_vulkan "
                "or vulkan_device_available)"
            )
    if violations:
        listing = "\n".join(f"  - {v}" for v in sorted(violations))
        raise pytest.UsageError(
            "Guard E (harness truth screen, tests/ops/conftest.py): the following test(s) "
            f"call `{_CLAIM_PROBE_NAME}()` — directly or through a helper defined in the "
            "same module — without depending on `require_vulkan` or "
            "`vulkan_device_available` first. `is_vulkan_claimed()` returns False both when "
            "the EP declines a node form AND when the EP is absent or crashed during "
            "session creation (see its docstring in _models.py); without one of those two "
            "fixtures a missing EP reads as a decline. See "
            ".squad/agents/trinity/history.md §8.9.25 (Morpheus).\n"
            f"{listing}\n"
            "Fix: add `require_vulkan` (or `vulkan_device_available`) as a parameter to "
            "each test listed above, or stop calling the claim probe from it."
        )


# ---------------------------------------------------------------------------
# R13 — three terminal states, printed as three distinct tokens (§10.0.1)
# ---------------------------------------------------------------------------
# > A check has at least three terminal states and must report them as three distinct
# > tokens: PASS, FAIL(condition) — the condition it exists to detect — and
# > ERROR(instrument), in which the check did not reach its observation. A red that could
# > mean either is not a signal. An instrument error is a lane failure of a different kind
# > and never counts as a detection.
#
# The specimen is this project's own: Guard D contained a NameError and raised before it
# read a single profiling event.  `8 passed` became `5 failed`, the red matched the
# prediction, and the crash was reported to the team as the guard working.  pytest's
# summary line has a two-state alphabet — `passed` and `failed` — and no reading of it,
# however careful, separates a detection from an outage.  So the lane grows a third token.
#
# Classification is by EXCEPTION TYPE, which makes it structural rather than a matter of
# care:
#   AssertionError / pytest.fail  -> FAIL(condition).  The check reached its observation.
#   InstrumentError + everything else -> ERROR(instrument).  It did not.
# Guards in _verdict.py raise InstrumentError strictly before they have a value and
# AssertionError strictly after, so the type carries the distinction by construction.
#
# Obligation 3 — a second witness with a DIFFERENT failure mode, not a better first
# witness: the lane greps captured ORT output for the known-fatal `Falling back` line and
# fails on it.  That line has appeared five times on this project while every gate passed.
# A grep cannot NameError; a guard cannot be silenced by a log-format change; each covers
# the other's outage.

_LANE_STATE_KEY = "_trinity_r13_state"

#: Accumulated per-test outcomes for the terminal summary.
#: Lists of (nodeid, first line of the failure text).  R13 second clause: the summary
#: quotes the failure TEXT, never only the count.
_lane_condition_failures: list[tuple[str, str]] = []
_lane_instrument_errors: list[tuple[str, str]] = []
_lane_fallback_logs: list[tuple[str, str]] = []
_lane_stale_expectations: list[tuple[str, str]] = []
_lane_passes: list[str] = []


def _fallback_lane_failure(captured: str) -> str | None:
    """Return a lane-failure message if *captured* announces a run-time fallback, else None.

    Pure and total, so it can be falsified without a GPU and cannot itself become the
    silent instrument it exists to cover for (R13 obligation 3).
    """
    hits = _verdict.find_fatal_log_lines(captured)
    if not hits:
        return None
    return (
        "[LANE FAILURE — known-fatal log line, R13 obligation 3]\n"
        f"AssertionError: ORT announced a run-time fallback during this test:\n    {hits[0]}\n"
        "A known-fatal log line is a lane failure, not a log line.  This line has appeared "
        "five times on this project while every gate passed — which is exactly the shape of "
        "the UNATTRIBUTED specimen (§10.0 third amendment).\n"
        "If this test provokes the fallback deliberately, mark it "
        "@pytest.mark.expects_ort_fallback."
    )


def _classify_failure(report) -> tuple[str, str]:
    """Return ``(state, quoted text)`` for a non-passing *report*.

    ``state`` is ``FAIL`` (the condition the check exists to detect) or ``ERROR`` (the
    check did not reach its observation).  The text is quoted, not counted.

    Duck-typed on purpose: it needs ``when`` and ``longrepr`` only, so it can be falsified
    directly against synthesised reports in ``test_r13_lane.py``.  A classifier that has
    never been shown to separate a ``NameError`` from an ``AssertionError`` is exactly the
    instrument R13 was written about.
    """
    text = ""
    excinfo = getattr(getattr(report, "longrepr", None), "reprcrash", None)
    if excinfo is not None:
        text = str(excinfo.message)
    elif getattr(report, "longrepr", None) is not None:
        text = str(report.longrepr)
    first = next((ln for ln in text.splitlines() if ln.strip()), "").strip()

    # An xfail(strict) that PASSED is not an outage and not a detection: the check reached
    # its observation and the observation contradicts a recorded *expectation*.  It is news,
    # usually good news, and it is the state most likely to be thrown away — nobody
    # investigates a red that is somebody else's fix.  R13's third corollary bites here: a
    # result that contradicts a prediction recruits attention for free, so it gets its own
    # token rather than being filed under the bucket nobody reads.
    if first.startswith("[XPASS(strict)]") or first.startswith("[XPASS"):
        return "XPASS", first

    # Setup/teardown failures never reached the test body: no observation exists.
    if getattr(report, "when", "call") in ("setup", "teardown"):
        return "ERROR", first or f"{getattr(report, 'when', '?')} error"

    lowered = first.lower()
    if first.startswith("AssertionError") or first.startswith("Failed"):
        return "FAIL", first
    if "instrument failure" in lowered or first.startswith("InstrumentError"):
        return "ERROR", first
    # Any other exception type is an outage by construction: a check that raises something
    # it did not name is a check that did not reach its observation.
    if ":" in first and not first.startswith("assert"):
        return "ERROR", first
    return "FAIL", first or "no failure text captured"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    """Classify each phase into R13's three states and apply the fallback-log witness."""
    outcome = yield
    report: pytest.TestReport = outcome.get_result()

    # --- second witness: the known-fatal ORT log line ------------------------
    # Runs on PASSING tests too — that is the entire point.  The line has appeared five
    # times while every gate passed.
    if report.when == "call" and not item.get_closest_marker("expects_ort_fallback"):
        captured = (report.capstderr or "") + "\n" + (report.capstdout or "")
        message = _fallback_lane_failure(captured)
        if message is not None:
            _lane_fallback_logs.append((report.nodeid, message.splitlines()[2].strip()))
            if report.passed:
                report.outcome = "failed"
                report.longrepr = message

    if report.when == "call" and report.passed:
        _lane_passes.append(report.nodeid)
    elif report.failed:
        state, text = _classify_failure(report)
        setattr(report, _LANE_STATE_KEY, state)
        if state == "ERROR":
            _lane_instrument_errors.append((report.nodeid, text))
        elif state == "XPASS":
            _lane_stale_expectations.append((report.nodeid, text))
        else:
            _lane_condition_failures.append((report.nodeid, text))


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    """Print the lane's three counts separately, and quote the failure text.

    R13: *a harness whose only failure vocabulary is its framework's FAILED line has a
    two-state alphabet and cannot carry three states.*  And the second clause: *quote the
    failure text, never the failure count* — because a confirmation of a prediction
    spends the reader's attention while a contradiction recruits it for free.
    """
    write = terminalreporter.write_line
    write("")
    write("=== R13 LANE SUMMARY — one token per terminal state ===")
    write(f"PASS               : {len(_lane_passes)}")
    write(f"FAIL(condition)    : {len(_lane_condition_failures)}")
    write(f"ERROR(instrument)  : {len(_lane_instrument_errors)}")
    write(f"XPASS(stale expect): {len(_lane_stale_expectations)}")
    for nodeid, text in _lane_condition_failures:
        write(f"  FAIL(condition)   {nodeid}")
        write(f"                    {text[:200]}")
    for nodeid, text in _lane_instrument_errors:
        write(f"  ERROR(instrument) {nodeid}")
        write(f"                    {text[:200]}")
    for nodeid, text in _lane_stale_expectations:
        write(f"  XPASS(stale)      {nodeid}")
        write(f"                    {text[:200]}")
    if _lane_stale_expectations:
        write(
            "An xfail(strict) that PASSED is neither a detection nor an outage: the "
            "recorded expectation is now wrong, usually because somebody fixed it. It is "
            "the state most likely to be discarded, because nobody investigates a red that "
            "is somebody else's good news — so it is named rather than counted."
        )
    if _lane_fallback_logs:
        write(f"LANE FAILURE (fallback log): {len(_lane_fallback_logs)}")
        for nodeid, line in _lane_fallback_logs:
            write(f"  {nodeid}")
            write(f"    {line[:200]}")
    if _lane_instrument_errors:
        write(
            "A lane with any instrument error is not a lane that ran, whatever else it "
            "reports.  An instrument error NEVER counts as a detection: none of the "
            "ERROR(instrument) lines above is evidence about the EP."
        )
    write("=== end R13 lane summary ===")


# ---------------------------------------------------------------------------
# Oracle version assertions — refuse rather than warn
# ---------------------------------------------------------------------------
# These constants are the MINIMUM versions required for a correct differential oracle.
# They must match the floors in tests/requirements.txt and the env pins in ci.yml.
#
# Design: fail hard at pytest_configure (before any test collection or execution).
# Rationale (from coordinator): Niobe's bench refuses to emit a Vulkan column under
# ORT < 1.28, because a silently wrong oracle is worse than an absent one. A test that
# passes because our CPU-EP reference had the wrong Attention semantics is not evidence
# of correctness — it is noise dressed as green.
#
# ONNX 1.23 lower bound: onnx 1.22 contains a WRONG reference implementation for
# ai.onnx::Attention at opset 24. The fix corrects semantics without an opset bump
# (Justin Chu, ONNX owner, directive 2026-07-29). A graph using Attention-24 produces
# DIFFERENT expected outputs under 1.22 vs 1.23.
# TODO(Fact Checker): update once exact release is confirmed.
#
# ORT 1.28 lower bound: null-allocator PrePack bug (fp16 MatMulNBits → NaN/Inf), and
# deleter lifetime issue in plugin-EP path. Both fixed in 1.28.
#
# ORT 1.28 opset registration ceiling (verified 2026-07-30, Fact Checker):
#   ORT 1.28 registers ONNX opsets through 27, not through 24 as the
#   onnxruntime.ai compatibility table (last row ORT 1.20 = opset 21) implied.
#   ORT 1.28 loads and runs opset-26 and opset-27 models. The compatibility table
#   is stale; any reasoning derived from it is wrong. Justin's directive "support
#   up to opset 26" is fully exercisable, and there is headroom to opset 27.
#   Reference: onnxruntime source, ORT_OPSET_SUPPORTED_VERSION in register_onnxruntime_ops.cc.
# ---------------------------------------------------------------------------

_ORT_MIN_VERSION = "1.28.0"
# onnx 1.22.0 is the current latest as of 2026-07-29. This is the minimum for a
# predictable (pinned) oracle — it does NOT mean the oracle is correct for all ops.
# Specifically: onnx 1.22 has a WRONG Attention-24 reference implementation.
# Attention tests must pytest.mark.xfail(strict=True) until Fact Checker confirms
# the fixed release and this constant is updated.
_ONNX_MIN_VERSION = "1.22.0"

# Minimum onnx version with a CORRECT Attention-24 reference implementation.
# The fix will ship in a forthcoming onnx release; 1.23.0 is the current best
# estimate (coordinator: "different outputs under onnx 1.22 vs 1.23").
# TODO(Fact Checker): confirm exact release, then:
#   1. Update this constant.
#   2. Update _ONNX_MIN_VERSION above to match.
#   3. Update tests/requirements.txt onnx lower bound.
#   4. Remove xfail from Attention tests.
_ONNX_ATTENTION24_FIXED_VERSION: str | None = None  # unknown until Fact Checker confirms

# Q/DQ opset-23+ oracle registration status (onnx#8182, detected at session start).
# See _probe_qdq_reference_oracle() below.  This is set once at pytest_configure time
# and is readable by tests via m.QDQOPSET23_REFERENCE_SAFE.
QDQOPSET23_REFERENCE_SAFE: bool = False
QDQOPSET23_REFERENCE_STATUS: str = "not yet probed"


def _version_tuple(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in v.split(".")[:3])


def _probe_qdq_reference_oracle() -> tuple[bool, str]:
    """Detect whether onnx.reference.ReferenceEvaluator can correctly evaluate DQ-23+.

    Background (onnx#8182): the opset-23 and opset-25 reference implementations for
    QuantizeLinear / DequantizeLinear are NOT registered in onnx ≤ 1.22.0.
    ReferenceEvaluator silently falls back to the opset-21 implementation, which does
    not know the ``output_dtype`` (new in 23) or ``block_size`` (new in 23) attributes.

    Two observable fallback behaviours (Fact Checker falsifier, 2026-07-30):
    - **TypeError** — the opset-21 impl receives an unknown attribute and raises.
      Observed in onnx 1.22.0 for the ``output_dtype`` form.
      Safe: the caller will see the error rather than a plausible wrong number.
    - **Silent wrong result** — the fallback succeeds for forms compatible with opset-21
      semantics but returns the wrong type or values.  Dangerous: a caller that trusts
      the number will build its oracle on wrong expected outputs.

    This function runs the Fact Checker's falsifier:
        DequantizeLinear(opset=23, output_dtype=FLOAT16)
    and classifies the result.

    Returns (safe, status):
      safe=True  → environment raises on the affected form.  Any caller that tries to
                   use ReferenceEvaluator as a Q/DQ-23 oracle will fail visibly.
      safe=False → environment returns without error.  The fallback is live and SILENT.
                   No caller may use ReferenceEvaluator as a Q/DQ-23+ oracle without
                   first calling m.assert_qdq_reference_oracle_safe().
    """
    try:
        import onnx
        import onnx.helper as oh
        from onnx.reference import ReferenceEvaluator
        node = oh.make_node("DequantizeLinear", ["x", "s", "z"], ["y"],
                            output_dtype=onnx.TensorProto.FLOAT16)
        graph = oh.make_graph(
            [node], "dq23_probe",
            [oh.make_tensor_value_info("x", onnx.TensorProto.UINT8, [4]),
             oh.make_tensor_value_info("s", onnx.TensorProto.FLOAT, []),
             oh.make_tensor_value_info("z", onnx.TensorProto.UINT8, [])],
            [oh.make_tensor_value_info("y", onnx.TensorProto.FLOAT16, [4])],
        )
        model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 23)])
        model.ir_version = 10
        ev = ReferenceEvaluator(model)
        import numpy as _np
        ev.run(None, {
            "x": _np.array([0, 128, 200, 255], dtype=_np.uint8),
            "s": _np.float32(0.01),
            "z": _np.uint8(128),
        })
        return (
            False,
            "UNSAFE: ReferenceEvaluator evaluated DQ-23 output_dtype without error — "
            "fallback to opset-21 is SILENT in this environment. "
            "No test may use ReferenceEvaluator as a Q/DQ-23+ oracle. "
            "(onnx#8182 fix is unreleased; upgrade to onnx 1.23.0+ when available.)",
        )
    except TypeError as exc:
        return (
            True,
            f"SAFE (raises): ReferenceEvaluator raises TypeError for DQ-23 output_dtype "
            f"in this environment (onnx {__import__('onnx').__version__}). "
            f"The fallback is live but detectable. "
            f"Detail: {exc}",
        )
    except Exception as exc:
        return (
            True,
            f"SAFE (raises): ReferenceEvaluator raises {type(exc).__name__} for DQ-23 output_dtype "
            f"in this environment. Detail: {exc}",
        )


def _assert_oracle_versions() -> None:
    """Hard-fail at session start if oracle libraries are below the required minimum.

    A silently wrong oracle is worse than an absent one (coordinator directive,
    citing Niobe's ORT < 1.28 case where every node ran on CPU under our provider's
    name, producing a 1.70x 'speedup' that was pure noise).

    Raises SystemExit (via pytest.exit, returncode=3) so tests cannot run against
    a drifted oracle.

    KNOWN ORACLE LIMITATION (documented, not a pin failure):
      onnx 1.22 has a WRONG Attention-24 reference implementation. The fix will ship
      in a forthcoming onnx release (_ONNX_ATTENTION24_FIXED_VERSION, currently None).
      General op tests (Add, Relu, etc.) run correctly with onnx 1.22.
      Attention tests MUST use pytest.mark.xfail(strict=True) until Fact Checker
      confirms the fixed release and _ONNX_ATTENTION24_FIXED_VERSION is set.

    Q/DQ-23+ oracle status (onnx#8182):
      Probed at session start. Result stored in QDQOPSET23_REFERENCE_SAFE and
      QDQOPSET23_REFERENCE_STATUS (module-level globals in this file and re-exported
      via m.QDQOPSET23_REFERENCE_SAFE for test visibility). No current test uses
      ReferenceEvaluator for Q/DQ; this probe exists to detect and refuse any future
      addition that would produce silently wrong expected outputs.
    """
    import onnx as _onnx  # noqa: PLC0415

    ort_ver = ort.__version__
    onnx_ver = _onnx.__version__

    errors: list[str] = []
    if _version_tuple(ort_ver) < _version_tuple(_ORT_MIN_VERSION):
        errors.append(
            f"onnxruntime {ort_ver} < required {_ORT_MIN_VERSION}.\n"
            f"  Reason: ORT 1.27 has null-allocator PrePack bug (fp16 NaN/Inf) and deleter\n"
            f"  lifetime issue in plugin-EP path. Both fixed in 1.28.\n"
            f"  Fix: pip install --upgrade 'onnxruntime>={_ORT_MIN_VERSION}'"
        )
    if _version_tuple(onnx_ver) < _version_tuple(_ONNX_MIN_VERSION):
        errors.append(
            f"onnx {onnx_ver} < required {_ONNX_MIN_VERSION}.\n"
            f"  This is the minimum for a predictable (pinned) oracle.\n"
            f"  An unpinned onnx allows silent oracle drift between test runs.\n"
            f"  Fix: pip install -r tests/requirements.txt"
        )
    if errors:
        msg = (
            "\n\n[ORACLE VERSION ERROR] Test session refused — oracle library out of date.\n"
            + "\n".join(f"  • {e}" for e in errors)
            + "\n\nInstall all required versions: pip install -r tests/requirements.txt\n"
        )
        pytest.exit(msg, returncode=3)

    # Probe Q/DQ-23 ReferenceEvaluator registration (onnx#8182).  Not a hard failure —
    # no current test uses ReferenceEvaluator for Q/DQ — but the status must be recorded
    # so that any future test that would silently use the wrong oracle is caught at review.
    global QDQOPSET23_REFERENCE_SAFE, QDQOPSET23_REFERENCE_STATUS
    QDQOPSET23_REFERENCE_SAFE, QDQOPSET23_REFERENCE_STATUS = _probe_qdq_reference_oracle()
    print(
        f"\n[ORACLE STATUS] Q/DQ-23 ReferenceEvaluator: {QDQOPSET23_REFERENCE_STATUS}",
        file=sys.stderr, flush=True,
    )
    if not QDQOPSET23_REFERENCE_SAFE:
        print(
            "[ORACLE STATUS] WARNING: A test that uses ReferenceEvaluator as a Q/DQ-23+ "
            "oracle will produce SILENTLY WRONG expected outputs. Call "
            "m.assert_qdq_reference_oracle_safe() before any such oracle path.",
            file=sys.stderr, flush=True,
        )


def _parse_device_indices(config: pytest.Config) -> list[int]:
    """Return the list of device indices to test, from CLI option or env var."""
    raw = config.getoption("--vulkan-devices") or os.environ.get("VULKAN_DEVICE_INDEX", "0")
    indices = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            try:
                indices.append(int(part))
            except ValueError:
                raise ValueError(
                    f"Invalid device index {part!r} in --vulkan-devices / VULKAN_DEVICE_INDEX. "
                    "Expected comma-separated integers, e.g. '0' or '0,1'."
                )
    return indices or [0]


# ---------------------------------------------------------------------------
# Session-level EP registration
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def register_vulkan_ep() -> bool:
    """Register the Vulkan EP plugin once per session; return True iff registration succeeded.

    Returns False (rather than skipping immediately) so that tests not requiring Vulkan
    (e.g., NumPy oracle checks, per-layer capture unit tests) can still run without the EP.
    Tests that DO require the EP use ``vulkan_device_available`` or ``require_vulkan``,
    both of which depend on this fixture and skip appropriately.

    Diagnostic protocol (helps Tank when the EP crashes during registration):
    1. Probe with ctypes.CDLL first — tests OS-level DLL dependency resolution without
       ORT involvement. If this fails, the cause is a missing runtime DLL (e.g.
       onnxruntime.dll, Vulkan loader, MSVC runtime), NOT an EP code bug.
    2. If ctypes load succeeds, call ort.register_execution_provider_library. If THAT
       crashes with a hard access violation, the ctypes-OK message will have been
       flushed to stderr before the crash, telling the diagnoser that DLL resolution
       was not the problem — the fault is in EP initialization code.
    All messages go to stderr with flush=True so they survive a hard process crash.
    """
    import ctypes

    lib = os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB")
    if not lib:
        return False
    lib_path = Path(lib).resolve()
    if not lib_path.is_file():
        print(
            f"\n[EP smoke] ONNXRUNTIME_VULKAN_EP_LIB set but file not found: {lib_path}",
            file=sys.stderr, flush=True,
        )
        return False

    # Step 1: ctypes load — DLL dependency check, no ORT involvement.
    try:
        ctypes.CDLL(str(lib_path))
        print(
            f"\n[EP smoke] ctypes.CDLL({lib_path.name}): OK — DLL dependencies resolved.",
            file=sys.stderr, flush=True,
        )
    except OSError as exc:
        print(
            f"\n[EP smoke] ctypes.CDLL FAILED: {exc}\n"
            f"[EP smoke] Cause: missing DLL dependency (onnxruntime.dll, Vulkan loader,\n"
            f"[EP smoke] or MSVC runtime). On Windows ensure $ORT_HOME\\lib is on PATH.\n"
            f"[EP smoke] EP registration skipped.",
            file=sys.stderr, flush=True,
        )
        return False

    # Step 2: ORT registration. A hard crash here means EP initialization code,
    # not DLL resolution (step 1 above already ruled that out).
    print(
        f"[EP smoke] Calling ort.register_execution_provider_library({EP_NAME!r}) ...",
        file=sys.stderr, flush=True,
    )
    try:
        ort.register_execution_provider_library(EP_NAME, str(lib_path))
        print(
            f"[EP smoke] ort.register_execution_provider_library: OK.",
            file=sys.stderr, flush=True,
        )
    except Exception as exc:
        if "already registered" in str(exc):
            # test_a_ep_smoke.py::test_ep_ort_registers runs before this fixture
            # and may have registered the EP first. That is fine — ORT's registration
            # is process-scoped, one call is enough.
            print(
                f"[EP smoke] ort.register_execution_provider_library: already registered (OK).",
                file=sys.stderr, flush=True,
            )
        else:
            print(
                f"[EP smoke] ort.register_execution_provider_library FAILED: {exc}",
                file=sys.stderr, flush=True,
            )
            return False
    return True


# ---------------------------------------------------------------------------
# Device availability probe (session-scoped, opt-in per test)
# ---------------------------------------------------------------------------


def _probe_vulkan_device() -> bool:
    """Return True if the registered Vulkan EP claims at least one node in a probe session.

    Uses a minimal MatMulNBits (com.microsoft, 4-bit, K=32 N=32) model: MatMulNBits is an
    anchor op (partition::is_anchor), so it passes the partition economics gate regardless of
    transfer cost — the anchor exemption unconditionally claims any island containing one.
    Returns False if the EP is registered but advertises zero devices (no Vulkan ICD, or all
    devices fail the capability gate).

    A plain Add model is NOT sufficient here: a 1-node Add forms a non-anchor island and is
    correctly declined by the partition TooSmall gate (1 < min_nodes=4, anchors=0).
    """
    import onnx
    import onnx.helper as oh
    import onnx.numpy_helper as onh

    rng = np.random.default_rng(0)
    K, N, bits, block_size = 32, 32, 4, 32
    blocks_per_col = (K + block_size - 1) // block_size
    packed_bytes = block_size * bits // 8  # = 16

    packed = rng.integers(0, 256, size=[N, blocks_per_col, packed_bytes], dtype=np.uint8)
    scale = rng.uniform(0.001, 0.1, size=[N * blocks_per_col]).astype(np.float16)
    act = rng.standard_normal((1, K)).astype(np.float16)

    node = oh.make_node(
        "MatMulNBits",
        inputs=["X", "B", "scale"],
        outputs=["Y"],
        domain="com.microsoft",
        K=K, N=N, bits=bits, block_size=block_size, accuracy_level=1,
    )
    x_info = oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT16, [1, K])
    y_info = oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT16, [1, N])
    graph = oh.make_graph(
        [node],
        "probe",
        [x_info],
        [y_info],
        initializer=[onh.from_array(packed, name="B"), onh.from_array(scale, name="scale")],
    )
    model = oh.make_model(
        graph,
        opset_imports=[oh.make_opsetid("", 21), oh.make_opsetid("com.microsoft", 1)],
        ir_version=8,
    )
    model_bytes = model.SerializeToString()
    feeds = {"X": act}

    opts = ort.SessionOptions()
    opts.log_severity_level = 4  # suppress noise during probe
    opts.enable_profiling = True
    opts.profile_file_prefix = "_vulkan_probe"
    try:
        sess = ort.InferenceSession(
            model_bytes, opts, providers=[EP_NAME, "CPUExecutionProvider"]
        )
        sess.run(None, feeds)
        profile_path = sess.end_profiling()
    except Exception:
        return False

    try:
        with open(profile_path) as fh:
            events = json.load(fh)
        providers = {
            e.get("args", {}).get("provider")
            for e in events
            if e.get("cat") == "Node" and isinstance(e.get("args"), dict)
        }
        return EP_NAME in providers
    except Exception:
        return False
    finally:
        try:
            os.remove(profile_path)
        except OSError:
            pass


_DEVICE_AVAILABLE: bool | None = None  # cached per session


@pytest.fixture(scope="session")
def vulkan_device_available(register_vulkan_ep: bool) -> bool:
    """Session-scoped fixture: skip if EP not registered; return True iff a device is present."""
    if not register_vulkan_ep:
        pytest.skip(
            "ONNXRUNTIME_VULKAN_EP_LIB is not set or not found — EP not registered. "
            "Set it to the built EP cdylib path:\n"
            "  Linux:   rust/target/release/libonnxruntime_vulkan_ep.so\n"
            "  Windows: rust\\target\\release\\onnxruntime_vulkan_ep.dll\n"
            "Run: cargo build --release",
        )
    global _DEVICE_AVAILABLE
    if _DEVICE_AVAILABLE is None:
        _DEVICE_AVAILABLE = _probe_vulkan_device()
    return _DEVICE_AVAILABLE


@pytest.fixture
def require_vulkan(vulkan_device_available: bool) -> None:
    """Skip the calling test if no Vulkan device is available."""
    if not vulkan_device_available:
        pytest.skip(
            "No Vulkan device available — either no ICD is installed or all devices "
            "failed the capability gate. Tests requiring a Vulkan device are skipped. "
            "To run on CPU software rasterizer (lavapipe): "
            "export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json"
        )


# ---------------------------------------------------------------------------
# Device parametrization — multi-GPU support
# ---------------------------------------------------------------------------
# Device selector mapping (corrected 2026-07-30T21:23:53-07:00 — Tank's finding):
# enumerate_capable_devices() sorts best-first (discrete before integrated).
# select_device() indexes the SORTED list, not the raw enumeration order.
# Local dev machine, sorted order:
#   Selector 0 → NVIDIA GeForce RTX 4060 Laptop GPU [Vulkan 1.4.325]  — discrete, less strict
#   Selector 1 → Intel Iris Xe Graphics              [Vulkan 1.4.309]  — UMA, spec-conformance oracle
#
# PREVIOUS LABELS (WRONG): 0=Intel, 1=NVIDIA — all results labelled this way are re-labelled;
# measurements are unchanged, only the vendor names were swapped.
#
# CI uses --vulkan-devices 0 (lavapipe is always device selector 0 in the software rasterizer lane).
#
# TODO(Switch): ep.device_index session option must be implemented in rust/src/ep.rs for
# multi-device runs to be effective. Until it lands, all sessions use the default device
# regardless of device_index. The parametrization infrastructure is wired and ready so
# that Switch's implementation immediately makes multi-device tests live.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def vulkan_device_indices(request: pytest.FixtureRequest) -> list[int]:
    """Session-scoped fixture: return the list of device indices selected for this run.

    Reads --vulkan-devices CLI option or VULKAN_DEVICE_INDEX env var.
    Default: [0] (first device only).
    """
    return _parse_device_indices(request.config)


@pytest.fixture(
    params=["device_index"],  # placeholder; real parametrization via vulkan_device_index
    ids=lambda x: x,
)
def vulkan_device_index(request: pytest.FixtureRequest, vulkan_device_indices: list[int]) -> int:
    """Per-test fixture: the Vulkan device index for this test invocation.

    When --vulkan-devices 0,1 is passed, tests that request this fixture run twice —
    once with device 0, once with device 1.

    The device index is applied to SessionOptions via ``ep.device_index`` (see TODO above).
    """
    # Parametrize dynamically over the selected indices.
    # pytest.fixture params= is static; we use indirect parametrization via conftest
    # collect_items hook instead. For now, return the first selected index.
    # Full parametrization is activated by pytest_generate_tests below.
    idx: int = request.param  # type: ignore[assignment]
    return idx


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrize tests that request ``vulkan_device_index`` over the selected devices.

    This hook replaces the static params= on the fixture. When --vulkan-devices 0,1
    is passed, every test requesting vulkan_device_index runs twice.
    """
    if "vulkan_device_index" in metafunc.fixturenames:
        indices = _parse_device_indices(metafunc.config)
        metafunc.parametrize(
            "vulkan_device_index",
            indices,
            ids=[f"dev{i}" for i in indices],
            # indirect=False so the fixture receives the int directly, not via params=
        )


def make_session_options_for_device(device_index: int) -> ort.SessionOptions:
    """Build SessionOptions for a specific Vulkan device index.

    Applies ep.device_index=<index> to select the target device.

    TODO(Switch): This session config entry must be implemented in rust/src/ep.rs.
    Until it lands, the EP ignores this entry and all sessions use the default device.
    The test infrastructure uses this helper so multi-device support activates
    automatically when Switch's implementation lands.
    """
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    # TODO(Switch): uncomment when ep.device_index is implemented
    # opts.add_session_config_entry("ep.device_index", str(device_index))
    validate = os.environ.get("ONNXRUNTIME_EP_VULKAN_VALIDATE", "").strip()
    if validate == "1":
        opts.add_session_config_entry("ep.enable_validation", "1")
    return opts


# ---------------------------------------------------------------------------
# Session options factory (provides validation layer control)
# ---------------------------------------------------------------------------


@pytest.fixture
def session_options() -> ort.SessionOptions:
    """Standard SessionOptions for op tests: quiet logging, optional validation layers."""
    opts = ort.SessionOptions()
    opts.log_severity_level = 3  # WARNING and above only
    validate = os.environ.get("ONNXRUNTIME_EP_VULKAN_VALIDATE", "").strip()
    if validate == "1":
        # Force validation layers on even in release builds.
        # The EP panics on any validation error → test fails with ORT exception.
        opts.add_session_config_entry("ep.enable_validation", "1")
    return opts

