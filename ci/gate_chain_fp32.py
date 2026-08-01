#!/usr/bin/env python3
"""``gate_chain_fp32`` — the criterion-10 gate artifact, as a CI lane step.

WHY THIS FILE EXISTS
====================
Before it, every CI lane's pass condition was *the pytest process exited zero*.  On
2026-07-30 that condition was satisfied for most of a day while this EP executed **zero
nodes**: ORT printed ``EP_FAIL ... Falling back to CPUExecutionProvider`` inside
``run()``, re-ran the whole graph on CPU, raised nothing, and ``get_providers()`` still
listed ``VulkanExecutionProvider`` because the provider list is fixed at session-create
time.  The suite was green.  Nothing ran on Vulkan.

DESIGN.md §8.9 (2026-07-30T06:32:18-07:00) requires that **each lane carry a gate
artifact**: the smallest artifact that (a) claims a non-zero node count on that lane,
(b) contains at least one island of two or more nodes, and (c) exercises at least one
proof key in every dtype the lane claims.  ``docs/PLATFORMS.md`` §7.8 specifies it:

    X [fp32, 256] -+
                   +-- Add -- Relu -- Z [fp32, 256]
    Y [fp32, 256] -+

    proof keys: (ai.onnx, Add,  7+, F32xF32->F32, ew_binary, static, {})
                (ai.onnx, Relu, 6+, F32->F32,     ew_unary,  static, {})

It is deliberately **the mechanism, not the model** (Morpheus, 2026-07-31): a right-sized
gate artifact per lane, not Phi-3.5 on a software rasteriser.

VOCABULARY — ONE, NOT TWO
=========================
Every token this script emits comes from ``tests/ops/_verdict.py`` (Trinity).  This file
defines **no** verdict strings of its own and constructs **no** verdict by literal:
``EquivalenceVerdict.from_comparison()`` requires an ``ExecutionAttribution`` parsed from
this run's ORT profile, so ``MATCH`` is unrepresentable at a zero own-provider count and
``UNATTRIBUTED`` is what comes out instead (§10.0 third amendment, clauses 3 and 4).
``UNATTRIBUTED`` is **not** ``DIVERGENT`` — the model was not wrong, the subject was.

R13 — THREE TERMINAL STATES, THREE EXIT CODES, THREE TOKENS
===========================================================
An instrument error never counts as a detection, so it never shares an exit code with
one:

    0  GATE: PASS
    1  GATE: FAIL(condition=...)      a finding about the EP
    4  GATE: ERROR(instrument=...)    a finding about this harness, about nothing else

Exit 4 rather than 3 because ``epctl`` already spends 3 on "the lane did not report", and
two different meanings on one code is the defect R13 names.

Everything this script prints about a failure quotes the **text** of what it observed —
never a count of failures.  A count is what let a ``NameError`` masquerade as a detection
on 2026-07-31.

R10 — THE FALSIFIER FOR "THIS IS WIRED"
=======================================
*The falsifier for "X is wired" is an artifact X produced whose content varies with its
input.*  This script's artifact is the verdict record at ``--verdict-out``: it carries
``executed_by``, the profile digest, the counters witness and the artifact digest, all
computed on this run.  It is not a flag.  Delete the step and the file is absent, which
is ``UNMEASURED``, which is a lane failure.

USAGE
=====
    python ci/gate_chain_fp32.py --verdict-out <path> [--counters <path>]
                                 [--workdir <dir>] [--device N]

``--counters`` should be the same path the lane passes in
``ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE`` so the record is spliced into the snapshot
``epctl --check-counters`` reads.  A caveat that lives in a different artifact from the
number it qualifies is not attached to it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# R13 exit codes.  Named, because a bare integer at a call site is a place to put the
# wrong one.
EXIT_PASS = 0
EXIT_FAIL_CONDITION = 1
EXIT_USAGE = 2
EXIT_ERROR_INSTRUMENT = 4


# ---------------------------------------------------------------------------
# Reporting — the three tokens.  Nothing else in this file prints a terminal state.
# ---------------------------------------------------------------------------


def report_pass(detail: str) -> int:
    print(f"GATE: PASS — {detail}", flush=True)
    return EXIT_PASS


def report_fail(condition: str, detail: str) -> int:
    print(f"GATE: FAIL(condition={condition})", flush=True)
    print(detail, flush=True)
    print(
        "GATE: this is a finding about the EP. It is NOT an instrument error; the check "
        "reached its observation and the observation is quoted above.",
        flush=True,
    )
    return EXIT_FAIL_CONDITION


def report_instrument_error(instrument: str, detail: str) -> int:
    print(f"GATE: ERROR(instrument={instrument})", flush=True)
    print(detail, flush=True)
    print(
        "GATE: the check did not reach its observation, so this is NOT a detection "
        "(DESIGN.md §10.0.1 R13). Do not route it as an EP bug and do not read it as a "
        "clean lane: a lane with an instrument error is not a lane that ran.",
        flush=True,
    )
    return EXIT_ERROR_INSTRUMENT


# ---------------------------------------------------------------------------
# fd-level capture of ORT's native stderr — the R13 obligation-3 second witness.
# ORT's `Falling back` line is written by C++ to fd 2 and never passes through
# sys.stderr, so a Python-level redirect would capture an empty string and the grep
# would agree with everything.  A grep cannot NameError; that is the entire point of
# it, and it is worth nothing if it greps the wrong stream.
# ---------------------------------------------------------------------------


@contextmanager
def capture_native_stderr(sink_path: Path):
    sys.stderr.flush()
    saved = os.dup(2)
    fh = open(sink_path, "w+b")
    try:
        os.dup2(fh.fileno(), 2)
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved, 2)
        os.close(saved)
        fh.flush()
        fh.close()


# ---------------------------------------------------------------------------
# The artifact
# ---------------------------------------------------------------------------

ARTIFACT_NAME = "gate_chain_fp32"
DECLINE_ARTIFACT_NAME = "gate_decline_probe_fp32"
ELEMENTS = 256

# The op the decline probe is built from.  It is deliberately one this EP does not
# implement and has no plan to: `Det` is a batched 3x3 determinant, it is not in any op
# family in `rust/src/ops/`, and ORT's CPU provider implements it, so the graph runs and
# a comparison is still performed.  If this ever becomes claimable the probe stops being
# a negative control — see `--artifact decline_probe` below for how that is caught rather
# than absorbed.
DECLINE_OP = "Det"
DECLINE_BATCH = 8


def build_gate_chain_fp32(onnx, np):
    """Return ``(model_bytes, feeds)`` for the §7.8.1 artifact.

    Feed values are pinned by §7.8.1 and are not arbitrary: ``X`` spans negative to
    positive and ``Y`` shifts the sum so it crosses zero *inside* the tensor, so the
    ``Relu`` clamp path is exercised on real data rather than on a tensor that happens
    to be non-negative.  A kernel that dropped the clamp entirely would disagree with
    CPU here rather than agree by luck.
    """
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [ELEMENTS])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [ELEMENTS])
    z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [ELEMENTS])
    graph = helper.make_graph(
        [
            helper.make_node("Add", ["X", "Y"], ["T"], name="gate_add"),
            helper.make_node("Relu", ["T"], ["Z"], name="gate_relu"),
        ],
        ARTIFACT_NAME,
        [x, y],
        [z],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        producer_name="onnxruntime-ep-vulkan-ci-gate",
    )
    model.ir_version = 10
    onnx.checker.check_model(model)

    feeds = {
        "X": np.linspace(-1.0, 1.0, ELEMENTS, dtype=np.float32),
        "Y": np.full((ELEMENTS,), -0.5, dtype=np.float32),
    }
    return model.SerializeToString(), feeds


def build_decline_probe_fp32(onnx, np):
    """Return ``(model_bytes, feeds)`` for the **negative control** artifact.

    WHY A SECOND ARTIFACT EXISTS
    ============================
    The original negative control removed the Vulkan ICD (``VK_DRIVER_FILES`` /
    ``VK_ICD_FILENAMES``) and required the gate to go red.  That control is **not
    available on every lane it is wired into**: PLATFORMS.md §7.4.1 records that the
    LunarG loader *silently ignores* both variables when the process is elevated, and
    GitHub Actions Windows runners are elevated — which is exactly why the Windows lane
    registers lavapipe in the registry in the first place.  On that lane the "no ICD"
    step does not remove the ICD; the gate then executes normally and passes, and the
    step reports ``NEGATIVE CONTROL FAILED``.  That red says "the gate cannot fail" when
    what actually happened is "the suppression did not take" — an instrument outage
    wearing a detection's costume, which is R13 with the polarity reversed and is the
    same defect the splice-ordering bug had.

    This artifact makes the EP do nothing **without touching the loader at all**: it is
    a single ``Det`` node, an op this EP does not implement.  The EP is loaded, the
    driver is present, the device is real, capability detection succeeds — and the EP
    still claims zero nodes, so the whole graph runs on CPU and the attribution comes
    back with a zero own-provider count.  The lane must report
    ``FAIL(condition=UNATTRIBUTED)``.

    It is also the *stronger* negative: "the EP was never able to start" and "the EP
    started and executed nothing" are different failures, and only the second one is the
    one that was live on 2026-07-30.

    Feeds are pinned and non-singular (a diagonal plus a fixed off-diagonal ramp) so the
    determinants are far from zero and a CPU-vs-CPU comparison cannot agree by rounding
    to zero on both sides.
    """
    from onnx import TensorProto, helper

    a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [DECLINE_BATCH, 3, 3])
    d = helper.make_tensor_value_info("D", TensorProto.FLOAT, [DECLINE_BATCH])
    graph = helper.make_graph(
        [helper.make_node(DECLINE_OP, ["A"], ["D"], name="decline_probe_det")],
        DECLINE_ARTIFACT_NAME,
        [a],
        [d],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        producer_name="onnxruntime-ep-vulkan-ci-gate",
    )
    model.ir_version = 10
    onnx.checker.check_model(model)

    base = np.eye(3, dtype=np.float32) * 2.0
    ramp = np.linspace(0.05, 0.4, DECLINE_BATCH, dtype=np.float32)
    mats = np.stack([base + r * np.float32(0.5) for r in ramp]).astype(np.float32)
    return model.SerializeToString(), {"A": mats}


# The lane selects an artifact by name; a lane step never constructs one by literal.
ARTIFACTS = {
    "chain_fp32": (ARTIFACT_NAME, build_gate_chain_fp32),
    "decline_probe": (DECLINE_ARTIFACT_NAME, build_decline_probe_fp32),
}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def parse_args(argv):
    p = argparse.ArgumentParser(description="criterion-10 gate artifact for a CI lane")
    p.add_argument("--verdict-out", required=True)
    p.add_argument(
        "--artifact",
        choices=sorted(ARTIFACTS),
        default="chain_fp32",
        help=(
            "chain_fp32 (default) is the §7.8.1 gate artifact. decline_probe is the "
            "loader-independent negative control: a single Det node this EP does not "
            "implement, so a healthy EP still executes nothing and the lane must report "
            "FAIL(condition=UNATTRIBUTED). This script reports what it observed either "
            "way — the polarity assertion lives in the lane step, not here."
        ),
    )
    p.add_argument(
        "--counters", default=os.environ.get("ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE", "")
    )
    p.add_argument("--workdir", default="")
    p.add_argument("--device", default=os.environ.get("ONNXRUNTIME_EP_VULKAN_DEVICE", ""))
    p.add_argument(
        "--rtol", type=float, default=1e-5, help="FP32_ELEMENTWISE tolerance (§7.8.1)"
    )
    p.add_argument("--atol", type=float, default=1e-5)
    return p.parse_args(argv)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def main(argv=None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    verdict_out = Path(args.verdict_out).resolve()
    verdict_out.parent.mkdir(parents=True, exist_ok=True)
    workdir = Path(args.workdir).resolve() if args.workdir else verdict_out.parent
    workdir.mkdir(parents=True, exist_ok=True)
    artifact_name, artifact_builder = ARTIFACTS[args.artifact]

    # -- import the one vocabulary -------------------------------------------------
    # tests/ops/_verdict.py is Trinity's.  Importing it rather than restating it is the
    # difference between one vocabulary and two that drift.
    sys.path.insert(0, str(REPO_ROOT / "tests" / "ops"))
    try:
        import _verdict  # type: ignore
    except Exception as exc:  # noqa: BLE001 - any import failure is an outage
        return report_instrument_error(
            "verdict_vocabulary_unavailable",
            f"Could not import tests/ops/_verdict.py from {REPO_ROOT}: {exc!r}\n"
            "This lane step defines no verdict tokens of its own by design (one "
            "vocabulary, not two), so without that module it cannot emit a verdict at "
            "all.  It reports an instrument outage rather than a pass.",
        )

    # -- UNMEASURED before the session opens ---------------------------------------
    # §7.8.2: the verdict file is written BEFORE anything can go wrong, so that every
    # path out of this process that skips the comparison leaves UNMEASURED behind.
    # This is not belt-and-braces; it is the only reason a crash cannot look like a
    # pass.
    try:
        initial = _verdict.EquivalenceVerdict.unmeasured(
            reason=(
                "gate_chain_fp32 initialised the verdict before opening any session. "
                "If this value survives, the comparison never completed."
            ),
            artifact=artifact_name,
            device_index=str(args.device),
        )
        verdict_out.write_text(
            json.dumps(initial.to_record(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001
        return report_instrument_error(
            "verdict_file_uninitialisable",
            f"Could not write the initial UNMEASURED record to {verdict_out}: {exc!r}",
        )
    print(
        f"GATE: {verdict_out} initialised to {_verdict.VERDICT_UNMEASURED} "
        "before session open.",
        flush=True,
    )

    # -- dependencies ---------------------------------------------------------------
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
    except Exception as exc:  # noqa: BLE001
        return report_instrument_error(
            "python_dependency_missing",
            f"{exc!r}\nInstall tests/requirements.txt in this lane before this step.",
        )

    lib = os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB", "")
    if not lib or not Path(lib).exists():
        return report_instrument_error(
            "ep_library_not_found",
            f"ONNXRUNTIME_VULKAN_EP_LIB={lib!r} is unset or does not name a file.\n"
            "The gate cannot distinguish 'the EP executed nothing' from 'the EP was "
            "never offered to ORT', so it refuses to report either one.",
        )

    try:
        model_bytes, feeds = artifact_builder(onnx, np)
    except Exception as exc:  # noqa: BLE001
        return report_instrument_error(
            "artifact_build_failed",
            f"{artifact_name} could not be constructed: {exc!r}\n{traceback.format_exc()}",
        )

    digest = hashlib.sha256(model_bytes).hexdigest()[:16]
    artifact_id = f"{artifact_name}@ci-gate-v1 sha256:{digest}"
    print(f"GATE: artifact = {artifact_id} ({ELEMENTS} elements, Add -> Relu)", flush=True)

    try:
        ort.register_execution_provider_library(_verdict.EP_NAME, str(Path(lib).resolve()))
    except Exception as exc:  # noqa: BLE001
        return report_instrument_error(
            "ep_library_registration_failed",
            f"ort.register_execution_provider_library({_verdict.EP_NAME!r}, {lib!r}) "
            f"raised: {exc!r}",
        )

    profile_prefix = workdir / f"{artifact_name}_profile"
    stderr_sink = workdir / f"{artifact_name}_ort_stderr.log"
    profile_path = None
    listed: list = []
    vk_out = None
    cpu_out = None

    try:
        with capture_native_stderr(stderr_sink):
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            opts.enable_profiling = True
            opts.profile_file_prefix = str(profile_prefix)
            if args.device != "":
                opts.add_session_config_entry("ep.device_index", str(args.device))

            vk_sess = ort.InferenceSession(
                model_bytes, opts, providers=[_verdict.EP_NAME, "CPUExecutionProvider"]
            )
            listed = list(vk_sess.get_providers())
            vk_out = vk_sess.run(None, feeds)[0]
            profile_path = vk_sess.end_profiling()
            del vk_sess

            cpu_opts = ort.SessionOptions()
            cpu_opts.log_severity_level = 3
            cpu_sess = ort.InferenceSession(
                model_bytes, cpu_opts, providers=["CPUExecutionProvider"]
            )
            cpu_out = cpu_sess.run(None, feeds)[0]
            del cpu_sess
    except Exception as exc:  # noqa: BLE001
        captured = _read_text(stderr_sink)
        # A raise here is ambiguous on its face — broken harness, or genuinely failing
        # EP — so the classification is made on evidence rather than on exception type:
        # if ORT announced a fallback, that is a condition, not an outage.
        fatal = _verdict.find_fatal_log_lines(captured)
        if fatal:
            return report_fail(
                "runtime_fallback_announced_by_ort",
                "ORT announced a run-time fallback and the session then raised.\n"
                + "\n".join(f"  ORT: {line}" for line in fatal)
                + f"\n\nException: {exc!r}\n{traceback.format_exc()}",
            )
        return report_instrument_error(
            "session_or_run_raised",
            f"{exc!r}\n{traceback.format_exc()}\n"
            f"Captured ORT stderr ({stderr_sink}):\n{captured or '  <empty>'}",
        )

    captured = _read_text(stderr_sink)
    print(f"GATE: session providers (session-create time) = {listed}", flush=True)

    # -- second witness, before the guard, because it cannot fail --------------------
    # R13 obligation 3: a grep cannot NameError, and a guard cannot be silenced by a log
    # format change.  Each covers the other's outage.  This line has now appeared five
    # times on this project while every gate passed.
    fatal = _verdict.find_fatal_log_lines(captured)
    if fatal:
        return report_fail(
            "runtime_fallback_announced_by_ort",
            "ORT announced a run-time fallback in this lane. The run completed and the\n"
            "outputs may well agree with CPU — they would, they were computed by CPU.\n"
            + "\n".join(f"  ORT: {line}" for line in fatal),
        )

    # -- attribution: from an instrument we do not own -------------------------------
    if not profile_path:
        return report_instrument_error(
            "profile_not_written",
            "sess.end_profiling() returned no path, so there is no attribution to read. "
            "enable_profiling was set on SessionOptions before session creation; if the "
            "path is still empty, ORT wrote no trace and this check has no input.",
        )
    try:
        attribution = _verdict.ExecutionAttribution.from_profile(profile_path)
    except _verdict.InstrumentError as exc:
        return report_instrument_error("profile_unreadable", str(exc))
    except Exception as exc:  # noqa: BLE001
        return report_instrument_error(
            "attribution_parse_raised", f"{exc!r}\n{traceback.format_exc()}"
        )

    counters_path = args.counters or None
    attribution = attribution.with_counters_witness(
        _verdict.read_counters_dispatches(counters_path)
    )
    # R13 obligation 2: state the observation whether or not the check passes.
    print(f"GATE: attribution — {attribution.describe()}", flush=True)

    # -- comparison -------------------------------------------------------------------
    try:
        agree = bool(np.allclose(vk_out, cpu_out, rtol=args.rtol, atol=args.atol))
        max_abs = float(
            np.max(np.abs(vk_out.astype(np.float64) - cpu_out.astype(np.float64)))
        )
    except Exception as exc:  # noqa: BLE001
        return report_instrument_error(
            "comparison_raised", f"{exc!r}\n{traceback.format_exc()}"
        )

    comparison = _verdict.COMPARISON_AGREE if agree else _verdict.COMPARISON_DISAGREE
    verdict = _verdict.EquivalenceVerdict.from_comparison(
        comparison=comparison,
        attribution=attribution,
        artifact=artifact_id,
        device_index=str(args.device),
        device_name=os.environ.get("ONNXRUNTIME_EP_VULKAN_DEVICE_NAME", ""),
        detail=f"max_abs_diff={max_abs:.6g} rtol={args.rtol} atol={args.atol}",
    )

    record = verdict.to_record()
    verdict_out.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"GATE: verdict record written to {verdict_out}", flush=True)
    print(json.dumps(record, indent=2, sort_keys=True), flush=True)

    # Splice into the counters snapshot so `epctl --check-counters` — a reader that
    # never sees this stdout — gates on the same value.
    #
    # ORDERING IS LOAD-BEARING, and getting it wrong once in development is why it is
    # spelled out here.  When this EP executes nothing, it never dispatches, so it never
    # writes a counters snapshot — and the splice then fails *because of the very
    # condition the gate exists to detect*.  Reporting that as ERROR(instrument=...)
    # would dress a real detection as an outage, which is R13's defect with the polarity
    # reversed and is if anything worse: it routes a finding to the harness owner.  So
    # the splice outcome is recorded, and it is only allowed to be the terminal state
    # when the verdict itself is MATCH — the one case where a downstream reader would
    # otherwise be left reading a stale or default value and calling it green.
    splice_error: str | None = None
    if counters_path:
        try:
            token = _verdict.write_equivalence_record(counters_path, verdict)
            print(
                f"GATE: spliced {token} into {counters_path} for epctl --check-counters.",
                flush=True,
            )
        except _verdict.InstrumentError as exc:
            splice_error = str(exc)
            print(
                f"GATE: could not splice the verdict into {counters_path}:\n{exc}",
                flush=True,
            )

    if verdict.verdict != _verdict.VERDICT_MATCH:
        note = ""
        if splice_error:
            note = (
                "\n\nThe verdict could not be attached to the counters snapshot either:\n"
                f"{splice_error}\n"
                "For UNATTRIBUTED that is usually the same finding said twice — an EP "
                "that executed nothing never dispatched, so it never wrote a snapshot."
            )
        return report_fail(
            verdict.verdict,
            verdict.explain()
            + f"\n\nmax_abs_diff={max_abs:.6g} (rtol={args.rtol}, atol={args.atol})"
            + f"\nAttribution: {attribution.describe()}"
            + f"\nRecord: {verdict_out}"
            + note,
        )

    if splice_error:
        return report_instrument_error(
            "counters_snapshot_unwritable",
            f"{splice_error}\nThe verdict record at {verdict_out} is valid and says "
            f"{verdict.verdict}; it simply could not be attached to the counters "
            "snapshot that downstream readers gate on. Refusing to pass: epctl "
            "--check-counters would read a stale or default value from that file and "
            "call this lane green on it.",
        )

    if verdict.verdict == _verdict.VERDICT_MATCH:
        return report_pass(
            f"{artifact_name} verdict={verdict.verdict}; "
            f"executed_by={verdict.executed_by}; max_abs_diff={max_abs:.6g}.\n"
            "  What this claims: the outputs of a run in which this EP executed at "
            "least one fused island agree with a CPU-only run of the same artifact.\n"
            "  What it does not claim: anything about any other artifact, about fp16, "
            "about multi-run arena reuse (PLATFORMS.md §7.4.2 single-run blindness), or "
            "about performance."
        )

    # Unreachable: every non-MATCH token returned above. Kept as a refusal rather than a
    # fall-through, because a future token that reaches here must not exit zero.
    return report_instrument_error(
        "verdict_token_unhandled",
        f"verdict={verdict.verdict!r} reached the end of the gate without a terminal "
        "state. This reader and tests/ops/_verdict.py disagree about the vocabulary.",
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
