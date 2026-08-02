"""Probe: `session.disable_cpu_ep_fallback` against our EP, at single-op scale.

The user found the flag; the coordinator verified it fires on Phi-3.5.  Before wiring it
into the op-test harness as a precondition, the questions that decide the design are:

  1. On a graph the EP **claims in full**, does the flag let the session be created — or
     does ORT plant CPU nodes of its own (Cast, MemcpyToHost, Identity) that trip it even
     in a healthy run?  If it trips, the precondition is useless at exactly the scale it
     was proposed for.
  2. On a graph the EP **declines**, what exactly does ORT raise — type and text?  R13
     says quote the text, so the harness has to know what it is quoting.
  3. Does it fire on a **partially** claimed graph?  (It should — that is the documented
     behaviour — and it is why this cannot be global.)
  4. Is the flag an ``add_session_config_entry`` key or a property, and does an unknown
     key raise or pass silently?  A precondition that silently does nothing is worse than
     no precondition.

Writes bench/results/disable_cpu_fallback_probe.json.  No wall-clock assertion.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np
import onnx_ir as ir
import onnxruntime as ort

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent / "tests" / "ops"))

import _models as m  # noqa: E402

OUT = _HERE / "disable_cpu_fallback_probe.json"
KEY = "session.disable_cpu_ep_fallback"


def _single(op: str, name: str) -> bytes:
    x = m.tensor("x", ir.DataType.FLOAT, [4, 4])
    y = m.tensor("y", ir.DataType.FLOAT, [4, 4])
    out = m.tensor("out", ir.DataType.FLOAT, [4, 4])
    inputs = [x, y] if op in ("Add", "Mul", "Sub", "Div") else [x]
    node = ir.Node("", op, inputs=inputs, outputs=[out], name=name)
    graph = ir.Graph(
        inputs=[x, y], outputs=[out], nodes=[node], name=f"single_{op}",
        opset_imports={"": 17},
    )
    return ir.to_proto(ir.Model(graph, ir_version=10)).SerializeToString()


def _mixed() -> bytes:
    x = m.tensor("x", ir.DataType.FLOAT, [4, 4])
    y = m.tensor("y", ir.DataType.FLOAT, [4, 4])
    a = m.tensor("a", ir.DataType.FLOAT, [4, 4])
    out = m.tensor("out", ir.DataType.FLOAT, [4, 4])
    nodes = [
        ir.Node("", "Add", inputs=[x, y], outputs=[a], name="claimed_add"),
        ir.Node("", "Erf", inputs=[a], outputs=[out], name="declined_erf"),
    ]
    graph = ir.Graph(
        inputs=[x, y], outputs=[out], nodes=nodes, name="mixed", opset_imports={"": 17}
    )
    return ir.to_proto(ir.Model(graph, ir_version=10)).SerializeToString()


def _try(model: bytes, *, disable: bool, providers: list[str]) -> dict:
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    if disable:
        opts.add_session_config_entry(KEY, "1")
    try:
        sess = ort.InferenceSession(model, opts, providers=providers)
    except Exception as exc:  # noqa: BLE001
        return {
            "created": False,
            "exc_type": type(exc).__name__,
            "exc_module": type(exc).__module__,
            "text": str(exc)[:400],
        }
    feeds = {
        "x": np.random.default_rng(1).standard_normal((4, 4)).astype(np.float32),
        "y": np.random.default_rng(2).standard_normal((4, 4)).astype(np.float32),
    }
    try:
        sess.run(None, feeds)
        ran = True
        run_err = ""
    except Exception as exc:  # noqa: BLE001
        ran = False
        run_err = f"{type(exc).__name__}: {exc}"[:300]
    return {
        "created": True,
        "providers": sess.get_providers(),
        "ran": ran,
        "run_error": run_err,
    }


def main() -> int:
    device = os.environ.get("ONNXRUNTIME_EP_VULKAN_DEVICE", "0")
    lib = os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB")
    if lib:
        try:
            ort.register_execution_provider_library(m.EP_NAME, str(pathlib.Path(lib).resolve()))
        except Exception as exc:  # noqa: BLE001
            if "already registered" not in str(exc):
                raise

    cases = {
        "claimed_single_add": _single("Add", "the_add"),
        "declined_single_erf": _single("Erf", "the_erf"),
        "partially_claimed": _mixed(),
    }
    record = {
        "device_index": device,
        "ort_version": ort.__version__,
        "key": KEY,
        "cases": {},
    }
    for name, model in cases.items():
        record["cases"][name] = {
            "ep_flag_off": _try(model, disable=False, providers=m.EP_PROVIDERS),
            "ep_flag_on": _try(model, disable=True, providers=m.EP_PROVIDERS),
            "cpu_only_flag_on": _try(model, disable=True, providers=["CPUExecutionProvider"]),
            # The decisive arm: CPU must NOT be in the providers list, or the flag is a
            # configuration conflict rather than a fallback check.
            "ep_only_flag_on": _try(model, disable=True, providers=[m.EP_NAME]),
            "ep_only_flag_off": _try(model, disable=False, providers=[m.EP_NAME]),
        }

    # (4) does an unknown config key raise, or pass silently?
    misspelled = ort.SessionOptions()
    misspelled.log_severity_level = 3
    try:
        misspelled.add_session_config_entry("session.disable_cpu_ep_fallbackk", "1")
        record["unknown_key_accepted_silently"] = True
    except Exception as exc:  # noqa: BLE001
        record["unknown_key_accepted_silently"] = False
        record["unknown_key_error"] = f"{type(exc).__name__}: {exc}"[:200]

    OUT.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
