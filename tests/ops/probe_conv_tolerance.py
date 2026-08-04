"""Derive `Conv`'s tolerance from measurement, because `_models.py` forbids guessing it.

`_models.py`'s module docstring reserves the accumulating ops:

    Reductions, GEMM, MatMul (M2+, OQ-10 in DESIGN.md §11):
        TBD — tolerance is accumulation-order-dependent and MUST be derived from test data
        per vendor (NVIDIA/AMD/lavapipe). ... Do not guess; do not copy from fp32 elementwise.

`Conv` is the first accumulating op to land, so it is the first one that clause applies to. This
script is the derivation it asks for: it runs every case in `_conv_cases.py` on both providers
and prints the worst observed relative and absolute residual, per case and overall.

It is a **measuring instrument, not a test**. It has no pass/fail: a number it prints is an
input to a human choosing a constant, and the constant then lives in `_models.py` with this
script named as its provenance. Re-run it on a new vendor before quoting its number there.

    python tests/ops/probe_conv_tolerance.py
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np
import onnxruntime as ort
from onnx_ir import DataType as DT

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import _conv_cases as cases  # noqa: E402
import _models as m  # noqa: E402

_RNG = np.random.default_rng(0xC0FFEE)


def _register_ep() -> None:
    """Register the EP, and refuse to proceed if it is not there.

    The first run of this script printed `max_rel=0.000e+00` for all twelve cases. It was not
    measuring agreement — ORT had answered `Unknown Provider Type: VulkanExecutionProvider`,
    fallen back to CPU, and compared the CPU against itself. A perfect number from an
    instrument that observed nothing is the exact failure this project spells
    ERROR(instrument), so this function raises instead of warning.
    """
    lib = os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB")
    if not lib or not pathlib.Path(lib).exists():
        raise SystemExit(
            "ERROR(instrument): set ONNXRUNTIME_VULKAN_EP_LIB to the built EP. Without it this "
            "script compares ORT's CPU EP against itself and reports zero divergence."
        )
    try:
        ort.register_execution_provider_library("VulkanExecutionProvider", lib)
    except Exception as exc:  # noqa: BLE001 — already-registered is the only benign case
        if "already registered" not in str(exc):
            raise


def build(case) -> tuple[bytes, dict[str, np.ndarray]]:
    (_id, n, c, mm, group, kernel, strides, dilations, pads, hw, bias) = case
    h, w = hw
    kh, kw = kernel
    oh = cases.out_extent(h, pads[0], pads[2], dilations[0], kh, strides[0])
    ow = cases.out_extent(w, pads[1], pads[3], dilations[1], kw, strides[1])
    ins = [
        m.tensor("X", DT.FLOAT, [n, c, h, w]),
        m.tensor("W", DT.FLOAT, [mm, c // group, kh, kw]),
    ]
    feeds = {
        "X": _RNG.standard_normal((n, c, h, w)).astype(np.float32),
        "W": _RNG.standard_normal((mm, c // group, kh, kw)).astype(np.float32),
    }
    if bias:
        ins.append(m.tensor("B", DT.FLOAT, [mm]))
        feeds["B"] = _RNG.standard_normal(mm).astype(np.float32)
    model = m.make_model(
        "Conv",
        ins,
        [m.tensor("Y", DT.FLOAT, [n, mm, oh, ow])],
        attributes={
            "kernel_shape": list(kernel),
            "strides": list(strides),
            "dilations": list(dilations),
            "pads": list(pads),
            "group": group,
        },
    )
    return model, feeds


def main() -> int:
    _register_ep()
    worst_rel = 0.0
    worst_atol = 0.0
    rows = []
    for case in cases.CONV_CASES:
        model, feeds = build(case)
        if not m.is_vulkan_claimed(model, feeds):
            raise SystemExit(
                f"ERROR(instrument): the EP did not claim case {case[0]!r}. Any residual "
                "printed after this point would be the CPU compared against itself."
            )
        vk = m.run_vulkan(model, feeds)[0]
        cpu = m.run_cpu(model, feeds)[0]
        diff = np.abs(vk.astype(np.float64) - cpu.astype(np.float64))
        denom = np.abs(cpu.astype(np.float64))
        rel = float(np.max(diff / np.maximum(denom, 1e-30)))
        absolute = float(np.max(diff))
        worst_rel = max(worst_rel, rel)
        worst_atol = max(worst_atol, absolute)
        rows.append({"case": case[0], "max_rel": rel, "max_abs": absolute,
                     "shape": list(vk.shape)})
        print(f"  {case[0]:<18} max_rel={rel:.3e}  max_abs={absolute:.3e}  shape={vk.shape}")

    print(f"\nWORST ACROSS {len(rows)} CASE(S): max_rel={worst_rel:.3e} max_abs={worst_atol:.3e}")
    print("Quote these two numbers when pinning FP32_CONV in _models.py; do not round them down.")
    out = pathlib.Path(__file__).resolve().parents[2] / "bench" / "results" / "conv_tolerance_derivation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"cases": rows, "worst_rel": worst_rel,
                               "worst_abs": worst_atol}, indent=2), encoding="utf-8")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
