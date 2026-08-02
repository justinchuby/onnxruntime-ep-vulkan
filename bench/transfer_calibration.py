"""Transfer calibration — replace the MVS placeholder constants with measured ones.

``rust/src/ops/partition.rs`` decides whether an island is worth running on the GPU with

    TransferModel { fixed_ns, bytes_per_ns }   →   cost_ns(bytes) = fixed_ns + bytes / bytes_per_ns

and a ``Policy { margin: 3.0, min_nodes: 4, .. }``. The current constants are explicitly
labelled placeholders in the source, and the margin is 3× *because* the model is a guess. This
script produces the samples ``TransferModel::fit(&[(bytes, ns)])`` consumes, so the guess can be
replaced on each device we care about.

Method
------

There is no public API in this EP for "copy N bytes to the device and back" — and there should
not be one, since it would be a benchmark-only path that the real code never takes. So the
staircase is measured through the thing users actually run: a **trivial single-node graph**
(``Identity``-shaped work, one elementwise ``Add`` against a broadcast scalar) whose input size
is swept. Per size we get end-to-end host latency, which decomposes as

    latency(bytes) ≈ dispatch_overhead + upload(bytes) + kernel(bytes) + readback(bytes)

Fitting an affine model to that gives ``fixed_ns`` = dispatch overhead + fixed transfer cost,
and ``bytes_per_ns`` = the *effective* boundary bandwidth including the kernel's own trivial
read/write. That is the honest description, and it is deliberately the quantity the partition
policy needs: the policy is asking "what does it cost to push this island's boundary bytes
across and get them back", and dispatch overhead is part of that cost whether or not it is
literally a transfer.

**Two caveats, both of which must travel with the numbers.**

1. The kernel is not free, so ``bytes_per_ns`` is a lower bound on raw copy bandwidth. Run with
   ``--gpu-timestamps`` and subtract the measured kernel time to separate them; until the
   engine writes timestamp queries (see ``docs/PERF.md`` §3) that subtraction is not available
   and the fit is the composite.
2. Integrated GPUs (UMA) and discrete GPUs have completely different affine constants, which is
   exactly why ``TransferModel`` ships ``UMA`` and ``DISCRETE`` starting points. Calibrate per
   device, never once.

Output
------

JSON with the raw samples plus a least-squares fit computed here (same estimator as
``TransferModel::fit``, so the two can be checked against each other), and a ready-to-paste Rust
literal. Nothing is written into the crate automatically: a measured constant landing in source
is a decision, and it goes through review with the device it came from named.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import cases as case_mod  # noqa: E402
import devices as device_mod  # noqa: E402
import environment  # noqa: E402
from bench import EP_NAME, DeviceSelectionError, _session, register_ep, select_device  # noqa: E402
from stats import Sample  # noqa: E402

_TESTS_OPS = Path(__file__).resolve().parents[1] / "tests" / "ops"
_SYS_PATH_BEFORE = list(sys.path)
if str(_TESTS_OPS) not in sys.path:
    sys.path.insert(0, str(_TESTS_OPS))

try:
    import onnx_ir as ir  # noqa: E402

    import _models  # noqa: E402
finally:
    # Scoped: a leaked `tests/ops` entry decides every later flat import process-wide.
    sys.path[:] = _SYS_PATH_BEFORE


def _staircase_model(n_elements: int):
    """A one-node fp32 ``Add`` over ``n_elements``: the smallest graph that moves bytes."""
    a = _models.tensor("A", ir.DataType.FLOAT, [n_elements])
    b = _models.tensor("B", ir.DataType.FLOAT, [n_elements])
    c = _models.tensor("C", ir.DataType.FLOAT, [n_elements])
    model = _models.make_model("Add", [a, b], [c])
    rng = np.random.default_rng(7)
    feeds = {
        "A": rng.standard_normal(n_elements).astype(np.float32),
        "B": rng.standard_normal(n_elements).astype(np.float32),
    }
    # Bytes crossing the boundary: two inputs up, one output back.
    return model, feeds, n_elements * 4 * 3


def least_squares_fit(samples: "list[tuple[int, float]]") -> "dict | None":
    """Fit ``ns = fixed_ns + bytes / bytes_per_ns`` — the same estimator as ``TransferModel::fit``.

    Returns ``None`` when the samples are degenerate (fewer than two distinct byte counts, or a
    non-positive slope, which would describe a device that gets faster with more data).
    """
    xs = [float(b) for b, _ in samples]
    ys = [float(ns) for _, ns in samples]
    n = len(xs)
    if n < 2 or len(set(xs)) < 2:
        return None
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    if sxx <= 0:
        return None
    slope = sxy / sxx  # ns per byte
    intercept = mean_y - slope * mean_x
    if slope <= 0:
        return None
    # R^2, so a bad fit is visible rather than quietly quoted.
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {
        "fixed_ns": round(max(intercept, 0.0), 3),
        "bytes_per_ns": round(1.0 / slope, 6),
        "ns_per_byte": round(slope, 9),
        "effective_gib_s": round((1.0 / slope) * 1e9 / (1024 ** 3), 3),
        "r2": round(r2, 5),
        "n_samples": n,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--max-log2", type=int, default=24, help="largest staircase step, 2^N bytes")
    ap.add_argument("--device", type=int, default=None,
                    help="physical device index to calibrate. Required on a multi-device "
                         "machine: a UMA part and a discrete part do not share a model.")
    args = ap.parse_args()

    if not register_ep():
        print("[calibrate] the EP is not available — nothing to calibrate.", file=sys.stderr)
        return 1

    device_facts, device_note = device_mod.probe()
    try:
        selected = select_device(device_facts, args.device)
    except DeviceSelectionError as exc:
        print(f"⛔ {exc}", file=sys.stderr)
        return 2

    env_record = environment.capture()
    print(environment.describe(env_record))
    print()
    print(selected.summary())
    print()
    if selected.transfer_class == "uma":
        print(
            "[calibrate] This is a UMA device: device-local memory is the same memory the host\n"
            "            wrote. An 'upload' here may be a mapping, not a copy, so `fixed_ns`\n"
            "            will dominate and `bytes_per_ns` will look enormous. That is a real\n"
            "            property of the part, not a fast PCIe link, and the resulting model\n"
            "            MUST NOT be applied to a discrete GPU. This is also the mobile case\n"
            "            (Adreno, Mali).\n"
        )
    elif selected.transfer_class == "discrete":
        print(
            "[calibrate] This is a discrete device: boundary bytes cross PCIe. `bytes_per_ns`\n"
            "            is a real bandwidth. The resulting model MUST NOT be applied to an\n"
            "            integrated/UMA part.\n"
        )
    else:
        print(
            "[calibrate] Transfer class unknown — the fit cannot be labelled UMA or discrete\n"
            "            and so cannot be safely reused anywhere. Recording it as unknown.\n"
        )

    samples: "list[tuple[int, float]]" = []
    rows = []
    for size_bytes in case_mod.transfer_staircase(args.max_log2):
        n_elements = max(size_bytes // 4, 1)
        model, feeds, boundary_bytes = _staircase_model(n_elements)
        sess = _session(model, [EP_NAME, "CPUExecutionProvider"], device_index=selected.index)
        for _ in range(args.warmup):
            sess.run(None, feeds)
        ms = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            sess.run(None, feeds)
            ms.append((time.perf_counter() - t0) * 1000.0)
        s = Sample(name=f"staircase_{boundary_bytes}B", samples=ms)
        ns = s.median * 1e6
        samples.append((boundary_bytes, ns))
        rows.append({"boundary_bytes": boundary_bytes, "elements": n_elements, **s.to_dict()})
        print(f"  {boundary_bytes:>12} B  {s.summary()}", flush=True)

    fit = least_squares_fit(samples)
    result = {
        "environment": env_record,
        "device": selected.to_dict(),
        "device_fingerprint": selected.fingerprint,
        "transfer_class": selected.transfer_class,
        "device_note": device_note,
        "iters": args.iters,
        "warmup": args.warmup,
        "samples": rows,
        "fit": fit,
        "method": (
            "end-to-end latency of a single-node fp32 Add, swept by size. Composite of dispatch "
            "overhead, upload, a trivial kernel and readback — see this file's docstring."
        ),
        "applies_to": (
            f"{selected.transfer_class} devices only, and strictly speaking only to "
            f"{selected.fingerprint}"
        ),
    }
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")

    print()
    if fit:
        print(f"fit on {selected.name} ({selected.transfer_class}): "
              f"fixed_ns={fit['fixed_ns']} bytes_per_ns={fit['bytes_per_ns']} "
              f"({fit['effective_gib_s']} GiB/s effective, R²={fit['r2']})")
        if fit["r2"] < 0.9:
            print(
                f"⚠ R²={fit['r2']} — the affine model does not describe this data. Do not paste "
                "these constants; find out why first (thermal throttling, a size-dependent "
                "kernel path, or a staircase that crosses an allocator regime)."
            )
        print("\nRust literal (paste behind review, naming the device it came from):\n")
        print(f"    // measured on {selected.name} ({selected.transfer_class}), "
              f"driver {selected.driver_version} — {env_record['captured_at']}")
        print(f"    // Valid for {selected.transfer_class} parts only. Do NOT reuse across "
              f"transfer classes.")
        print("    TransferModel {")
        print(f"        fixed_ns: {fit['fixed_ns']},")
        print(f"        bytes_per_ns: {fit['bytes_per_ns']},")
        print("    }")
    else:
        print("no usable fit — the staircase was degenerate; nothing is claimed.")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
