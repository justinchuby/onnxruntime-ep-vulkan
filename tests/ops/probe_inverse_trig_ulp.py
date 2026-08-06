#!/usr/bin/env python3
"""Measure the Vulkan EP's `Asin`/`Acos` error against the ORT CPU EP, densely.

WHY THIS EXISTS
===============
`tests/ops/test_op_table.py` runs `Asin`/`Acos` on twelve random points in (-0.9, 0.9). That is
enough to *catch* the defect issue #4 describes — Mesa lavapipe's GLSL built-in `Asin`/`Acos` are
off by up to 1.56e-4, ~1e3x the fp32 elementwise contract, and twelve points found it — but it is
nowhere near enough to *characterise* the replacement. Vulkan's "Precision of GLSL.std.450
Instructions" table gives `Asin`/`Acos` **no accuracy requirement** at all, so the only defensible
posture is to stop calling them and state a measured bound for whatever replaced them. This probe
is where that number comes from.

It sweeps the closed domain [-1, 1] — endpoints included, because the reduction
`asin(a) = pi/2 - 2*asin(sqrt((1-a)/2))` degenerates exactly there — plus the reduction seam at
|x| = 1/2 and its representable neighbours, signed zero, out-of-domain inputs and NaN. It reports
max / p99 / p99.9 ULP against the CPU EP (the oracle the op tests use) and against a float64
reference (the one that does not move when ORT's libm does).

WHAT IT DOES NOT DO
===================
It does not assert a bound. `tests/ops/test_inverse_trig.py` is the assertion; this is the
instrument that produced its numbers, kept runnable so a later reader can re-derive them rather
than trust them. It writes nothing unless `--out` names a path, and it refuses to report at all if
the Vulkan EP did not execute the ops — a sweep that silently ran on the CPU EP would report
0.0 ULP for everything, which is the exact shape of a vacuous pass this project keeps finding.

USAGE
=====
    $env:ONNXRUNTIME_VULKAN_EP_LIB = "...\\onnxruntime_vulkan_ep.dll"
    python tests/ops/probe_inverse_trig_ulp.py --points 2000001 --out report.json

Set `VK_ICD_FILENAMES` to a lavapipe ICD to measure the software lane; leave it unset for the
local physical device. The device actually opened is recorded in the output, because a measurement
that does not name its device is not evidence about any device (§10.0.1 R12).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from onnx_ir import DataType as DT

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _models as m  # noqa: E402

#: Elements per dispatch. The sweep is split so a multi-million-point run does not allocate one
#: enormous buffer — the EP's arena is not the subject here.
_CHUNK = 1 << 20

#: Inputs the closed-form math is delicate at, evaluated exactly rather than sampled near.
_SEAMS = (-1.0, -0.5, -0.0, 0.0, 0.5, 1.0)

#: Inputs outside the domain, plus the non-finite ones. ONNX follows the C library here: `Asin`
#: and `Acos` of anything outside [-1, 1] — including +-inf — is NaN, and NaN in is NaN out.
_EDGE = (-1.0, -0.0, 0.0, 1.0, 1.0000001, -1.0000001, 2.0, -2.0, np.nan, np.inf, -np.inf)


def sweep(points: int) -> np.ndarray:
    """The evaluation grid: a dense linear sweep plus every point the math is delicate at."""
    dense = np.linspace(-1.0, 1.0, points, dtype=np.float64).astype(np.float32)
    extra: list[np.float32] = []
    for s in (np.float32(v) for v in _SEAMS):
        extra.append(s)
        extra.append(np.nextafter(s, np.float32(-2.0)))
        extra.append(np.nextafter(s, np.float32(2.0)))
    grid = np.unique(np.concatenate([dense, np.array(extra, dtype=np.float32)]))
    grid = grid[(grid >= -1.0) & (grid <= 1.0)]
    # `np.unique` sorts, and -0.0 == +0.0, so it drops one of the two zeros. Signed zero is part
    # of the stated contract, so put -0.0 back explicitly rather than hope it survived.
    return np.concatenate([grid, np.array([-0.0], dtype=np.float32)])


def run_both(op: str, xs: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Run `op` over `xs` on the Vulkan EP and the CPU EP. Returns (vk, cpu, chunks)."""
    vk_parts: list[np.ndarray] = []
    cpu_parts: list[np.ndarray] = []
    chunks = 0
    for start in range(0, xs.size, _CHUNK):
        chunk = np.ascontiguousarray(xs[start : start + _CHUNK])
        shape = [int(chunk.size)]
        model = m.make_model(
            op,
            [m.tensor("x", DT.FLOAT, shape)],
            [m.tensor("out", DT.FLOAT, shape)],
        )
        feeds = {"x": chunk}
        vk_parts.append(np.asarray(m.run_vulkan(model, feeds)[0], dtype=np.float32))
        cpu_parts.append(np.asarray(m.run_cpu(model, feeds)[0], dtype=np.float32))
        chunks += 1
    return np.concatenate(vk_parts), np.concatenate(cpu_parts), chunks


def ulp_vs(got: np.ndarray, ref_f32: np.ndarray) -> np.ndarray:
    """|got - ref| in ULPs of float32 at the reference's magnitude.

    The spacing basis is the reference's value and is computed by `_models.format_spacing`, the
    house instrument — reimplementing it here is how the two would drift apart, and its
    `np.spacing` -> `inf` repair at the top of the finite range is not worth re-deriving.
    Non-finite elements are not an ULP distance at all: they agree (both NaN, or bit-equal) or
    they do not, and this returns 0 or inf for them rather than a number that invites averaging.
    """
    got64 = got.astype(np.float64)
    ref64 = ref_f32.astype(np.float64)
    finite = np.isfinite(got64) & np.isfinite(ref64)
    out = np.empty_like(got64)
    spacing = m.format_spacing(ref_f32, np.float32)
    with np.errstate(invalid="ignore"):
        out[finite] = np.abs(got64[finite] - ref64[finite]) / spacing[finite]
    agree = (np.isnan(got64) & np.isnan(ref64)) | (got64 == ref64)
    out[~finite] = np.where(agree[~finite], 0.0, np.inf)
    return out


def _pct(u: np.ndarray, q: float) -> float | None:
    f = u[np.isfinite(u)]
    return float(np.percentile(f, q)) if f.size else None


def stats(op: str, vk: np.ndarray, cpu: np.ndarray, exact64: np.ndarray) -> dict:
    ref32 = exact64.astype(np.float32)
    vs_cpu = ulp_vs(vk, cpu)
    vs_exact = ulp_vs(vk, ref32)
    cpu_vs_exact = ulp_vs(cpu, ref32)
    both_finite = np.isfinite(vk.astype(np.float64)) & np.isfinite(exact64)
    abs_err = np.abs(vk.astype(np.float64)[both_finite] - exact64[both_finite])
    return {
        "op": op,
        "points": int(vk.size),
        "bit_identical_to_cpu": bool(np.array_equal(vk.view(np.uint32), cpu.view(np.uint32))),
        "max_ulp_vs_cpu": _pct(vs_cpu, 100.0),
        "p99_ulp_vs_cpu": _pct(vs_cpu, 99.0),
        "p99_9_ulp_vs_cpu": _pct(vs_cpu, 99.9),
        "max_ulp_vs_float64": _pct(vs_exact, 100.0),
        "p99_9_ulp_vs_float64": _pct(vs_exact, 99.9),
        "cpu_max_ulp_vs_float64": _pct(cpu_vs_exact, 100.0),
        "max_abs_err_vs_float64": float(np.max(abs_err)) if abs_err.size else None,
        "disagreements_at_non_finite": int(np.count_nonzero(np.isinf(vs_cpu))),
    }


def edge_cases(op: str) -> dict:
    """Endpoints, signed zero, out-of-domain and NaN — reported as raw bits, not as floats.

    `repr(np.float32(-0.0))` and `repr(np.float32(0.0))` both print `0.0`, so a signed-zero
    regression is invisible in a float-formatted report. The hex is the observation.
    """
    xs = np.array(_EDGE, dtype=np.float32)
    vk, cpu, _ = run_both(op, xs)
    agree = [
        bool(a == b) or bool(np.isnan(x) and np.isnan(y))
        for a, b, x, y in zip(vk.view(np.uint32), cpu.view(np.uint32), vk, cpu)
    ]
    return {
        "inputs_hex": [f"0x{v:08x}" for v in xs.view(np.uint32)],
        "vulkan_hex": [f"0x{v:08x}" for v in vk.view(np.uint32)],
        "cpu_hex": [f"0x{v:08x}" for v in cpu.view(np.uint32)],
        "agree": agree,
        "all_agree": all(agree),
    }


def register_ep() -> str:
    """Register the built EP with ORT. Returns "" on success, else a diagnosis.

    `conftest.register_vulkan_ep` is a pytest fixture and cannot be called from a script, so the
    three lines are repeated here exactly as every other `probe_*.py` in this directory repeats
    them. What is *not* repeated is the claim check — `executed_on_vulkan` below asks
    `_models.is_vulkan_claimed`, the house guard, rather than inferring anything from this call.
    """
    import onnxruntime as ort

    lib = os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB", "")
    if not lib:
        return "ONNXRUNTIME_VULKAN_EP_LIB is unset; nothing to measure."
    path = Path(lib).resolve()
    if not path.is_file():
        return f"ONNXRUNTIME_VULKAN_EP_LIB points at a missing file: {path}"
    try:
        ort.register_execution_provider_library(m.EP_NAME, str(path))
    except Exception as exc:  # noqa: BLE001 — already-registered is the only benign case
        if "already registered" not in str(exc):
            return f"could not register {m.EP_NAME}: {exc}"
    return ""


def device_identity() -> dict:
    """Name the device this run opened, by asking `epctl --probe-loader` next to the EP.

    A measurement that does not name its device is not evidence about any device (§10.0.1 R12),
    and none of ORT's Python surfaces expose the Vulkan device name. `epctl` is built from the
    same crate as the EP and reads the same loader environment, so it sees what this process sees.
    """
    lib = os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB", "")
    out: dict = {
        "vk_icd_filenames": os.environ.get("VK_ICD_FILENAMES", ""),
        "vk_driver_files": os.environ.get("VK_DRIVER_FILES", ""),
        "ep_lib": lib,
    }
    epctl = Path(lib).resolve().parent / ("epctl.exe" if os.name == "nt" else "epctl")
    if not epctl.is_file():
        out["devices"] = None
        out["device_probe_error"] = f"epctl not found next to the EP at {epctl}"
        return out
    try:
        proc = subprocess.run(
            [str(epctl), "--probe-loader"], capture_output=True, text=True, timeout=180
        )
    except Exception as exc:  # noqa: BLE001 — diagnostic only
        out["devices"] = None
        out["device_probe_error"] = str(exc)
        return out
    names = re.findall(r"^Device \d+ \[[^\]]*\]: (.+?) \[Vulkan ", proc.stdout, re.M)
    out["devices"] = names
    out["gate_pass"] = "gate PASS" in proc.stdout
    return out


def executed_on_vulkan(op: str) -> bool:
    """Did the EP actually claim this op, or did ORT quietly run the whole sweep on the CPU?

    Without this the probe's headline is 0.0 ULP for every op on every device, because the two
    "different" sessions ran the same CPU kernel. `_models.is_vulkan_claimed` is the house guard;
    this only picks the feeds.
    """
    xs = np.linspace(-0.9, 0.9, 64, dtype=np.float32)
    model = m.make_model(op, [m.tensor("x", DT.FLOAT, [64])], [m.tensor("out", DT.FLOAT, [64])])
    return m.is_vulkan_claimed(model, {"x": xs})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Dense ULP sweep for Asin/Acos on the Vulkan EP.")
    ap.add_argument("--points", type=int, default=2_000_001, help="dense sweep resolution")
    ap.add_argument("--out", default="", help="write the JSON report here (default: stdout only)")
    ap.add_argument(
        "--label",
        default="",
        help="free-text description of what shader path this run measures, e.g. "
        "'GLSL built-in asin/acos at 94a4bd6' — recorded verbatim in the report.",
    )
    ap.add_argument(
        "--allow-unclaimed",
        action="store_true",
        help="report even if the EP declined the ops. The verdict text changes so the result "
        "cannot be quoted as a measurement of this EP.",
    )
    args = ap.parse_args(argv)

    if problem := register_ep():
        print(f"ERROR(instrument): {problem}", file=sys.stderr)
        return 4

    ops = ("Asin", "Acos")
    claimed = {op: executed_on_vulkan(op) for op in ops}
    if not all(claimed.values()) and not args.allow_unclaimed:
        print(
            "ERROR(instrument): the Vulkan EP declined "
            f"{[o for o, c in claimed.items() if not c]}. Every residual below would be the CPU "
            "EP compared against itself and would read 0.0 ULP. Re-run with --allow-unclaimed "
            "only if that is what you want to record.",
            file=sys.stderr,
        )
        return 4

    xs = sweep(args.points)
    report: dict = {
        "verdict": "MEASURED" if all(claimed.values()) else "UNATTRIBUTED(ep-declined)",
        "label": args.label,
        "generated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "sweep_points": int(xs.size),
        "domain": "[-1, 1] closed; seams at +-1/2 and +-1 with representable neighbours; "
        "signed zero; out-of-domain and non-finite inputs in edge_cases",
        "claimed_by_vulkan_ep": claimed,
        "device": device_identity(),
        "ops": {},
    }

    for op, exact_fn in (("Asin", np.arcsin), ("Acos", np.arccos)):
        vk, cpu, chunks = run_both(op, xs)
        with np.errstate(invalid="ignore"):
            exact = exact_fn(xs.astype(np.float64))
        row = stats(op, vk, cpu, exact)
        row["dispatched_chunks"] = chunks
        row["edge_cases"] = edge_cases(op)
        report["ops"][op] = row

    try:
        import onnxruntime as ort

        probe = m.make_model(
            "Asin", [m.tensor("x", DT.FLOAT, [4])], [m.tensor("out", DT.FLOAT, [4])]
        )
        report["providers"] = list(
            ort.InferenceSession(probe, providers=m.EP_PROVIDERS).get_providers()
        )
    except Exception as exc:  # pragma: no cover - diagnostic only
        report["providers_error"] = str(exc)

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
