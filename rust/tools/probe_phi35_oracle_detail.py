"""Per-output detail for the Phi-3.5 all-output CPU-oracle comparison.

Criterion 10's lane reports one aggregate — ``DISAGREE`` with a worst ``max_abs_diff``. That
is the right thing for a gate to report, and the wrong thing to hand to whoever has to fix
it: it names neither *which* of the 65 outputs disagree nor *by how much relative to their
own justified tolerance*. This probe reports that, and nothing else.

It computes no verdict. It grants nothing. It is a reading aid for a failure that has
already been recorded by the lane.

R11 note: the probe prints the per-output table for **every** output, not only the failing
ones. A table that lists only failures cannot distinguish "one output is wrong" from "the
comparison only looked at one output" — the same shape as the reopened row.
"""

from __future__ import annotations

import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tests" / "ops"))

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

import _models as m  # noqa: E402
from test_phi35 import _ONNX_FILE, _build_phi35_feeds  # noqa: E402


def main() -> int:
    lib = os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB")
    if not lib:
        print("ERROR(instrument): ONNXRUNTIME_VULKAN_EP_LIB not set")
        return 2
    ort.register_execution_provider_library(m.EP_NAME, lib)

    onnx_file = _ONNX_FILE
    if not pathlib.Path(onnx_file).is_file():
        print(f"ERROR(instrument): model not found at {onnx_file}")
        return 2

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    vk = ort.InferenceSession(
        onnx_file, opts, providers=[m.EP_NAME, "CPUExecutionProvider"]
    )
    if m.EP_NAME not in vk.get_providers():
        print("ERROR(instrument): EP absent from get_providers(); reading would be CPU-vs-CPU")
        return 2

    cpu = ort.InferenceSession(onnx_file, opts, providers=["CPUExecutionProvider"])

    feeds = _build_phi35_feeds()
    vk_out = vk.run(None, feeds)
    cpu_out = cpu.run(None, feeds)

    outcome, facts = m.compare_all_outputs_to_cpu(vk_out, cpu_out)
    names = [o.name for o in vk.get_outputs()]

    print(f"outcome                    {outcome}")
    for k in (
        "oracle_outputs_total",
        "oracle_outputs_compared",
        "oracle_outputs_within_tolerance",
        "oracle_outputs_degenerate",
        "oracle_max_abs_diff_over_all_outputs",
        "oracle_worst_output_index",
    ):
        print(f"{k:<27}{facts[k]}")
    print(f"{'oracle_failing_indices':<27}{facts['oracle_failing_indices']}")

    # `max_rel_diff` in the facts divides by `atol + |b|`; the pass criterion divides by
    # `atol + rtol*|b|`. Quoting the first against `rtol` reports 24.6x for an output that
    # PASSES. The `bound` column below is the ratio against the criterion actually applied,
    # so it is <= 1 exactly when the status says WITHIN_TOLERANCE and can be read as a margin.
    print()
    print(f"{'idx':>3}  {'name':<28} {'status':<18} {'max_abs':>12} {'max_rel':>12} "
          f"{'rtol':>8} {'bound':>9}")
    for e, (v, c) in zip(facts["per_output"], zip(vk_out, cpu_out)):
        i = e["index"]
        rel = e.get("max_rel_diff")
        rtol = e.get("rtol")
        atol = e.get("atol")
        bound = "-"
        if rtol is not None and atol is not None:
            a = np.asarray(v, dtype=np.float64)
            b = np.asarray(c, dtype=np.float64)
            ratio = np.abs(a - b) / (atol + rtol * np.abs(b))
            bound = f"{float(ratio.max()) if ratio.size else 0.0:.3g}x"
        print(
            f"{i:>3}  {names[i][:28]:<28} {e['status']:<18} "
            f"{e.get('max_abs_diff', float('nan')):>12.6g} "
            f"{rel if rel is not None else float('nan'):>12.6g} "
            f"{rtol if rtol is not None else float('nan'):>8.4g} {bound:>9}"
        )

    # Does the divergence change the model's decision? A logits difference that never moves
    # the argmax is a different finding from one that does, and the two must not be quoted
    # as the same fact.
    v0 = np.asarray(vk_out[0], dtype=np.float64)
    c0 = np.asarray(cpu_out[0], dtype=np.float64)
    if v0.shape == c0.shape and v0.size:
        print()
        print(f"logits argmax vk={int(v0.reshape(-1, v0.shape[-1])[-1].argmax())} "
              f"cpu={int(c0.reshape(-1, c0.shape[-1])[-1].argmax())}")
        print(f"logits abs range cpu=[{c0.min():.6g}, {c0.max():.6g}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
