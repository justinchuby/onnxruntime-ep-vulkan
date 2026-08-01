#!/usr/bin/env python3
"""Two-polarity control for the broken-commitment WARN (RAI Ruling 2 / RAI-010 (c)).

WHAT IS UNDER TEST
==================
A node this EP **claimed** — it told ORT it would compute it — whose ``Compute()`` then returns a
non-OK status is a broken commitment at runtime.  ORT reacts by silently re-running the work on the
CPU EP; five sightings of that ``Falling back`` line on this project, twice with zero signal at any
layer a user could observe.  The mechanism (``ep.rs::disclose_broken_commitment``) emits a WARNING
through **ORT's own logging sink** — the session's ``Logger_LogMessage``, not this project's log
crate — naming the node(s), the condition, the failure text, and that CPU re-execution follows.

WHY BOTH POLARITIES, IN ONE SCRIPT
==================================
A WARN that cannot be shown *not* to fire on a good run is not a detector, it is a printed opinion.
So this runs the same model, on the same device, through the same call site, twice:

    POSITIVE   fault injection armed  -> the WARN must appear in ORT's sink
    NEGATIVE   fault injection off    -> the WARN must NOT appear, and the run must have
                                         executed dispatches, so the silence is a *result* and
                                         not the silence of an EP that never ran

The negative polarity's second condition is the one that matters and is easy to omit: a CPU-only
fallback run also emits no broken-commitment WARN, and it emits none for the wrong reason.  A quiet
run that dispatched nothing is not evidence that the detector is quiet on good runs; it is the
2026-07-30 specimen wearing a green badge.  ``broken_commitments`` is therefore published as the
JSON *string* ``"UNOBSERVABLE"`` whenever ``compute_calls == 0`` — a zero-Compute run cannot even
represent the clean-run token.

HOW "ORT'S SINK" IS DISTINGUISHED FROM OUR OWN STDERR
=====================================================
The probe reads ORT's **own logging sink** — the channel already carrying ORT's ``Falling back``
line and the one a host with ORT logging configured is already watching.  Lines that reached it
carry ORT's decoration (``[W:onnxruntime:...]``); our private stderr line (``[vulkan-ep] WARN:``)
is counted separately and never satisfies the assertion, because a WARN in this project's own log
is invisible to exactly the audience that matters.

This detour is not decoration.  The obvious witness — grep ORT's decorated line out of stderr —
looked **blind** at first: ORT's default sink on Windows writes *wide* characters, so a UTF-8 read
of the same stream renders every one of its lines as NUL-separated letters and matches nothing.
The first version of this probe reported ``FAIL`` for a WARN that had in fact been delivered.  The
stream is therefore decoded twice, as UTF-8 and as UTF-16LE, and searched in both; and the witness
carries its own control — if ORT's sink emitted no line at all in the positive run, not even ORT's
own error for the failure we planted, the verdict is ``ERROR(instrument=...)``, never ``FAIL``.

USAGE
    python rust/tools/probe_broken_commitment.py [--device N] [--out PATH]

Runs every advertised device by default.  Artifacts land in ``bench/results/``; nothing is written
to the repository root.  Exit codes follow R13:

    0  PROBE: PASS
    1  PROBE: FAIL(<condition>)          a detection — the mechanism is not what Ruling 2 requires
    4  PROBE: ERROR(instrument=<what>)   the probe could not run; never a detection
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
LIB = REPO / "rust" / "target" / "release" / "onnxruntime_vulkan_ep.dll"
RESULTS = REPO / "bench" / "results"

ENV_INJECT = "ONNXRUNTIME_EP_VULKAN_FORCE_COMPUTE_FAILURE"

# The marker the EP puts in the message; ORT's own decoration of a line *its* sink emitted; and
# our private stderr line, which never satisfies the assertion.
WARN_MARKER = "BROKEN COMMITMENT"
ORT_DECORATION = re.compile(r"\[[VIWEF]:onnxruntime:")
OUR_PRIVATE_DECORATION = "[vulkan-ep] WARN:"


# ---------------------------------------------------------------------------------------------
# child process: one polarity, one device, one session
# ---------------------------------------------------------------------------------------------
def child(device_index: int, inject: bool, counters_path: pathlib.Path) -> int:
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper

    model = REPO / "rust" / "target" / "probe_broken_commitment_model.onnx"
    model.parent.mkdir(parents=True, exist_ok=True)
    if not model.is_file():
        nodes, prev = [], "x"
        for i in range(4):
            nodes.append(helper.make_node("Add", [prev, "w"], [f"t{i}"], name=f"add{i}"))
            prev = f"t{i}"
        nodes.append(helper.make_node("Identity", [prev], ["y"], name="out"))
        w = helper.make_tensor("w", TensorProto.FLOAT, [1024], np.ones(1024, np.float32))
        graph = helper.make_graph(
            nodes,
            "chain",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1024])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1024])],
            [w],
        )
        m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
        m.ir_version = 10
        onnx.save(m, str(model))

    os.environ["ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE"] = str(counters_path)
    if inject:
        os.environ[ENV_INJECT] = "1"
    else:
        os.environ.pop(ENV_INJECT, None)

    # WARNING and above, so ORT's sink carries our message and nothing quieter drowns it.
    ort.set_default_logger_severity(2)
    ort.register_execution_provider_library("VulkanExecutionProvider", str(LIB))
    devices = [d for d in ort.get_ep_devices() if d.ep_name == "VulkanExecutionProvider"]
    if device_index >= len(devices):
        print(f"CHILD-ERROR: device {device_index} not advertised ({len(devices)} available)")
        return 4

    so = ort.SessionOptions()
    so.log_severity_level = 2
    so.add_provider_for_devices([devices[device_index]], {})
    sess = ort.InferenceSession(str(model), so)
    out = sess.run(None, {"x": np.arange(1024, dtype=np.float32)})
    expect = np.arange(1024, dtype=np.float32) + 4.0
    print(f"CHILD-OK: outputs_match_cpu={bool(np.allclose(out[0], expect))}")
    return 0


# ---------------------------------------------------------------------------------------------
# parent: run both polarities, judge the artifacts
# ---------------------------------------------------------------------------------------------
def decode_both(raw: bytes) -> str:
    """Decode a child's output as UTF-8 **and** as UTF-16LE, and return both.

    ORT's default logging sink on Windows writes wide characters to stderr, so a UTF-8 decode of
    the same stream renders its lines as NUL-separated letters and every ``grep`` over them misses.
    That is not a subtlety: the first version of this probe reported ``FAIL`` for a WARN that had
    in fact been delivered, because the witness could not read the channel it was watching. Both
    decodings are searched, so a line counts wherever it is legible.

    Each stream is decoded on its own. Concatenating two captured streams with a one-byte
    separator first shifts the second stream's UTF-16LE alignment by one byte and turns every
    line in it into mojibake — a witness that reads one stream and silently loses the other, which
    is how the second version of this probe missed a WARN that a direct run showed plainly.

    Alignment is not something this probe gets to assume even within one stream: our own narrow
    stderr line is written to the *same* handle, and if it has an odd byte length every wide line
    after it is off by one. So the stream is decoded at both alignments, and additionally with the
    NUL padding removed, which recovers wide ASCII no matter where it starts. Three readings of the
    same bytes, because a witness that can only read one alignment reports absence for a line that
    is plainly present.
    """
    return "\n".join(
        (
            raw.decode("utf-8", "replace"),
            raw.decode("utf-16le", "replace"),
            raw[1:].decode("utf-16le", "replace"),
            raw.replace(b"\x00", b"").decode("utf-8", "replace"),
        )
    )


def run_polarity(device_index: int, inject: bool) -> tuple[str, dict]:
    """Run one polarity in a fresh process and return (combined stderr+stdout, counters)."""
    tag = "positive" if inject else "negative"
    counters = RESULTS / f"broken-commitment-dev{device_index}-{tag}.json"
    counters.parent.mkdir(parents=True, exist_ok=True)
    counters.unlink(missing_ok=True)
    env = dict(os.environ)
    env.pop(ENV_INJECT, None)
    proc = subprocess.run(
        [
            sys.executable,
            str(pathlib.Path(__file__).resolve()),
            "--child",
            "--device",
            str(device_index),
            *(["--inject"] if inject else []),
            "--counters",
            str(counters),
        ],
        capture_output=True,
        env=env,
    )
    log = decode_both(proc.stdout or b"") + "\n" + decode_both(proc.stderr or b"")
    doc = json.loads(counters.read_text()) if counters.is_file() else {}
    return log, doc


def ort_sink_lines(log: str) -> list[str]:
    """Every line ORT's own logging sink emitted, whoever wrote the message."""
    return [ln for ln in log.splitlines() if ORT_DECORATION.search(ln)]


def ort_sink_warns(log: str) -> list[str]:
    return [ln for ln in ort_sink_lines(log) if WARN_MARKER in ln]


def private_only_warns(log: str) -> list[str]:
    return [
        ln
        for ln in log.splitlines()
        if ln.lstrip().startswith(OUR_PRIVATE_DECORATION) and WARN_MARKER in ln
    ]


def judge(device_index: int) -> tuple[str, list[str], dict]:
    failures: list[str] = []
    report: dict = {"device": device_index}

    # ---- POSITIVE: a planted Compute failure must reach ORT's sink -------------------------
    log, doc = run_polarity(device_index, inject=True)
    sink = ort_sink_warns(log)
    report["positive"] = {
        "ort_sink_lines_total": len(ort_sink_lines(log)),
        "ort_sink_warn_lines": len(sink),
        "first_ort_sink_warn": sink[0] if sink else None,
        "private_log_warn_lines": len(private_only_warns(log)),
        "counters": doc,
    }
    if not doc:
        return "ERROR(instrument=positive_polarity_wrote_no_counters)", failures, report
    if not ort_sink_lines(log):
        # The witness saw no line from ORT's sink at all — not even ORT's own error for the
        # failure we planted. That is a blind witness, and a blind witness never produces a
        # detection (R13).
        return "ERROR(instrument=ort_sink_not_observable_in_this_host)", failures, report
    if not sink:
        failures.append(
            "positive polarity: ORT's sink emitted "
            f"{len(ort_sink_lines(log))} line(s) and none of them carried the broken-commitment "
            f"marker. private-log lines seen: {len(private_only_warns(log))}. "
            "A WARN in our own log is invisible to a host watching ORT's channel."
        )
    if doc.get("broken_commitments") != doc.get("compute_calls"):
        failures.append(
            "positive polarity: every injected Compute failure must be disclosed; "
            f"broken_commitments={doc.get('broken_commitments')!r} "
            f"compute_calls={doc.get('compute_calls')!r}"
        )
    if doc.get("broken_commitment_warn_channel") != "ORT_SINK":
        failures.append(
            "positive polarity: broken_commitment_warn_channel="
            f"{doc.get('broken_commitment_warn_channel')!r}, expected 'ORT_SINK'"
        )
    if doc.get("fault_injection") != "ACTIVE":
        failures.append(
            "positive polarity: the artifact does not mark itself as fault-injected "
            f"(fault_injection={doc.get('fault_injection')!r}); an injected failure that reads "
            "like a suffered one is worse than no control at all"
        )

    # ---- NEGATIVE: a good run must be silent, AND must have run ----------------------------
    log, doc = run_polarity(device_index, inject=False)
    sink = ort_sink_warns(log)
    report["negative"] = {
        "ort_sink_lines_total": len(ort_sink_lines(log)),
        "ort_sink_warn_lines": len(sink),
        "offending_lines": sink[:3],
        "counters": doc,
    }
    if not doc:
        return "ERROR(instrument=negative_polarity_wrote_no_counters)", failures, report
    if sink:
        failures.append(
            f"negative polarity: {len(sink)} BROKEN COMMITMENT line(s) on a successful run — "
            f"first: {sink[0]}"
        )
    dispatches = doc.get("dispatches_executed", 0)
    if not isinstance(dispatches, int) or dispatches == 0:
        failures.append(
            "negative polarity: dispatches_executed=0, so this run's silence is the silence of an "
            "EP that never executed anything. Absence of a WARN here is not a result."
        )
    if doc.get("broken_commitments") != 0:
        failures.append(
            "negative polarity: broken_commitments="
            f"{doc.get('broken_commitments')!r}, expected the integer 0 — an in-frame measured "
            "zero, not a token"
        )
    if doc.get("fault_injection") != "NONE":
        failures.append(
            f"negative polarity: fault_injection={doc.get('fault_injection')!r}, expected 'NONE'"
        )

    return ("FAIL" if failures else "PASS"), failures, report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--inject", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--counters", default="", help=argparse.SUPPRESS)
    ap.add_argument("--device", type=int, default=-1, help="device index; default = all")
    ap.add_argument("--out", default=str(RESULTS / "broken-commitment-control.json"))
    args = ap.parse_args()

    if args.child:
        return child(args.device, args.inject, pathlib.Path(args.counters))

    if not LIB.is_file():
        print(f"PROBE: ERROR(instrument=library_absent) — {LIB} not found; cargo build --release")
        return 4

    try:
        import onnxruntime as ort  # noqa: F401
    except Exception as e:  # noqa: BLE001
        print(f"PROBE: ERROR(instrument=onnxruntime_unavailable) — {type(e).__name__}: {e}")
        return 4

    if args.device >= 0:
        devices = [args.device]
    else:
        # Enumerating in the parent would load the EP here too; ask a child instead.
        probe = subprocess.run(
            [sys.executable, "-c", ENUMERATE_SNIPPET.format(lib=str(LIB).replace("\\", "\\\\"))],
            capture_output=True,
            text=True,
        )
        m = re.search(r"DEVICE_COUNT=(\d+)", probe.stdout or "")
        if not m:
            print(
                "PROBE: ERROR(instrument=device_enumeration_failed) — "
                f"{(probe.stdout or '') + (probe.stderr or '')}"
            )
            return 4
        devices = list(range(int(m.group(1))))
        if not devices:
            print("PROBE: ERROR(instrument=no_ep_devices) — the EP advertises no device here")
            return 4

    report: dict = {"devices": []}
    verdicts: dict[int, str] = {}
    all_failures: list[str] = []
    for d in devices:
        verdict, failures, per_device = judge(d)
        per_device["verdict"] = verdict
        per_device["failures"] = failures
        report["devices"].append(per_device)
        verdicts[d] = verdict
        all_failures += [f"device {d}: {f}" for f in failures]

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nartifact: {out}")

    for d, v in verdicts.items():
        print(f"device {d}: {v}")
    if any(v.startswith("ERROR") for v in verdicts.values()):
        # R13: an instrument failure is never a detection, and gets its own exit code.
        print("PROBE: " + next(v for v in verdicts.values() if v.startswith("ERROR")))
        return 4
    if all_failures:
        print("PROBE: FAIL(broken_commitment_warn_control)")
        for f in all_failures:
            print(f"  - {f}")
        return 1
    print(
        "PROBE: PASS — on every device tested, a planted Compute failure produced a WARN through "
        "ORT's own sink and a successful run with a non-zero dispatch count produced none."
    )
    return 0


ENUMERATE_SNIPPET = (
    "import onnxruntime as ort\n"
    "ort.register_execution_provider_library('VulkanExecutionProvider', r'{lib}')\n"
    "n = len([d for d in ort.get_ep_devices() if d.ep_name == 'VulkanExecutionProvider'])\n"
    "print('DEVICE_COUNT=%d' % n)\n"
)


if __name__ == "__main__":
    sys.exit(main())
