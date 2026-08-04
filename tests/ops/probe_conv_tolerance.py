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

    python tests/ops/probe_conv_tolerance.py [--device N]

DEVICE IDENTITY IS READ OFF THE RUN, NOT OFF THE REQUEST (added 2026-08-04, Mouse)
---------------------------------------------------------------------------------
`--device N` sets `ONNXRUNTIME_EP_VULKAN_DEVICE`, which is a *request* handed to the loader.
Trinity has observed `DEVICE=0` running on `1=NVIDIA`, so a result filed under "device 1"
because that is what was asked for names nothing. This script therefore takes the EP's own
`running_device_names` counter and files its output under that, and refuses to report at all
if `dispatches_executed` is 0 — Switch found a lane with zero EP dispatches, exit 0 and no
exception, and every number such a lane prints is the CPU compared against itself.
"""

from __future__ import annotations

import argparse
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


_COUNTERS = (
    pathlib.Path(__file__).resolve().parents[2]
    / "bench"
    / "results"
    / "_conv_tolerance_counters.json"
)


def _read_run_identity() -> tuple[str, int]:
    """Return `(device_name, dispatches_executed)` as reported by the EP that just ran.

    Raises rather than returning a default: an unknown device makes the whole measurement
    unattributable, which is worse than no measurement.
    """
    if not _COUNTERS.is_file():
        raise SystemExit(
            "ERROR(instrument): the EP wrote no counters file, so the device that produced "
            "these residuals cannot be named and the numbers cannot be filed."
        )
    c = json.loads(_COUNTERS.read_text(encoding="utf-8"))
    name = (c.get("running_device_names") or "").strip()
    dispatches = int(c.get("dispatches_executed") or 0)
    if not name:
        raise SystemExit(
            "ERROR(instrument): counters carry no running_device_names; see above."
        )
    return name, dispatches


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--device",
        default=None,
        help="sets ONNXRUNTIME_EP_VULKAN_DEVICE, before the EP library is registered (the only "
        "construction that makes ORT's binding and ours the same device). Accepts an index into "
        "the *best-first sorted capables* list — which is NOT vulkaninfo's enumeration order, so "
        "0 is the discrete part on this box, not GPU0 — or a substring of the device name such "
        "as `Intel`. Either way it is a request: the name this script files its results under is "
        "read back off the run.",
    )
    args = ap.parse_args()
    if args.device is not None:
        os.environ["ONNXRUNTIME_EP_VULKAN_DEVICE"] = str(args.device)
    _COUNTERS.parent.mkdir(parents=True, exist_ok=True)
    if _COUNTERS.exists():
        _COUNTERS.unlink()
    os.environ["ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE"] = str(_COUNTERS)

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

    device_name, dispatches = _read_run_identity()
    if dispatches <= 0:
        raise SystemExit(
            f"ERROR(instrument): dispatches_executed={dispatches} on {device_name!r}. The EP "
            "claimed but never dispatched; every residual above is the CPU against itself."
        )
    print(f"\nDEVICE (read off the run): {device_name}   dispatches_executed={dispatches}")
    print(f"REQUESTED: ONNXRUNTIME_EP_VULKAN_DEVICE="
          f"{os.environ.get('ONNXRUNTIME_EP_VULKAN_DEVICE', '<unset>')}")
    print(f"WORST ACROSS {len(rows)} CASE(S): max_rel={worst_rel:.3e} max_abs={worst_atol:.3e}")
    print("Quote these two numbers when pinning FP32_CONV in _models.py; do not round them down.")
    slug = "".join(ch if ch.isalnum() else "_" for ch in device_name).strip("_").lower()
    out = (
        pathlib.Path(__file__).resolve().parents[2]
        / "bench"
        / "results"
        / f"conv_tolerance_derivation_{slug}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "device_name": device_name,
                "device_requested": os.environ.get("ONNXRUNTIME_EP_VULKAN_DEVICE", ""),
                "dispatches_executed": dispatches,
                "cases": rows,
                "worst_rel": worst_rel,
                "worst_abs": worst_atol,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
