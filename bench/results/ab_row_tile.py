"""Interleaved A/B of the MatMulNBits row tile (issue #7) on one device, in one process.

Why this exists rather than two `bench.py` runs
-----------------------------------------------
The row tile is a *weight-traffic* change, so the effect it is supposed to have is a wall-clock
difference between the same shape run tiled and untiled. Two separate `bench.py` invocations
cannot measure that difference honestly on a shared workstation: they are minutes apart, the GPU
clock ramps, another process may take the device, and the two arms end up separated by whatever
happened in between rather than by the code. The 2026-08-06 pair of runs showed exactly this — the
`M=1` control came out 0.303 ms untiled and 0.360 ms tiled, both flagged noisy, for a shape whose
two arms bind *the same specialisation constants* and therefore cannot differ at all.

So this script alternates the two arms `REPEATS` times inside one process, on one device, with the
arms adjacent in time. Anything that drifts over minutes drifts through both arms equally, and the
paired difference survives it. This is the same reason `ONNXRUNTIME_EP_VULKAN_GEMV_PACKED` exists
(see `ops::quant::gemv_packed`); `ONNXRUNTIME_EP_VULKAN_GEMV_MAX_ROWS` is its row-tile twin.

The `M=1` row is not a result, it is the control
------------------------------------------------
At `M = 1` `gemv_tile` returns `rows = 1` in *both* arms, so both build the identical pipeline from
the identical SPIR-V. Its measured ratio is therefore a direct read of this harness's noise floor,
and every `M > 1` ratio has to be read against it. A speedup smaller than the `M=1` spread is not a
speedup.

Usage
-----
    ONNXRUNTIME_VULKAN_EP_LIB=... python bench/results/ab_row_tile.py --out ab_row_tile.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

_BENCH = Path(__file__).resolve().parents[1]
_ROOT = _BENCH.parent
for _p in (str(_BENCH), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ROWS_ENV = "ONNXRUNTIME_EP_VULKAN_GEMV_MAX_ROWS"

# The shape is the OQ-12 anchor's K and N, so these numbers sit beside a series that already
# exists rather than starting a new one. M covers both tile heights (2 -> QB_ROWS=2, 4 and 8 ->
# QB_ROWS=4) and one partial tile (5), plus the M=1 control.
K, N = 4096, 4096
M_VALUES = (1, 2, 4, 5, 8)
REPEATS = 5


def _sample(sess, feeds, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        sess.run(None, feeds)
    xs = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(None, feeds)
        xs.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(xs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(Path(__file__).with_name("ab_row_tile.json")))
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--device", type=int, default=None, help="physical device index to pin")
    args = ap.parse_args()

    import bench as bench_mod
    import numpy as np  # noqa: F401  (imported for its side effect on the model builder)
    from tests.ops import _models as m

    if not bench_mod.register_ep():
        print("A/B needs the Vulkan EP; refusing to report a one-armed comparison.", file=sys.stderr)
        return 2

    providers = ["VulkanExecutionProvider"]
    import devices as device_mod

    facts, _source = device_mod.probe()
    try:
        chosen = bench_mod.select_device(facts, args.device)
    except Exception as exc:
        # A result that does not name the device it ran on is not a result (bench.py::select_device
        # says so at length). An A/B is worse: the two arms could have landed on two GPUs.
        print(f"[ab] refusing to run: {exc}", file=sys.stderr)
        return 2
    device = chosen.index
    device_name = chosen.name
    print(f"[ab] device {device}: {device_name} [{chosen.transfer_class}]")

    models = {}
    for m_rows in M_VALUES:
        models[m_rows] = m.make_matmulnbits_model(K=K, N=N, block_size=32, rows=m_rows)

    prev = os.environ.get(ROWS_ENV)
    # Arms are keyed by the value of the knob, not by a boolean, so the JSON says what was set.
    arms = {"untiled": "1", "tiled": "4"}
    raw: dict = {m_rows: {a: [] for a in arms} for m_rows in M_VALUES}
    try:
        for rep in range(REPEATS):
            # Alternate which arm goes first. A fixed order does not cancel: the GPU clock and the
            # page cache both drift monotonically inside a repeat, so whichever arm always runs
            # second inherits that drift as a systematic bias. The first version of this script had
            # a fixed order and reported a 0.905x "slowdown" at M=1 — a shape where both arms bind
            # identical specialisation constants and identical SPIR-V, so the only thing it could
            # have been measuring was the order.
            order = list(arms.items())
            if rep % 2:
                order.reverse()
            for arm, value in order:
                os.environ[ROWS_ENV] = value
                for m_rows in M_VALUES:
                    model, feeds = models[m_rows]
                    # A fresh session per arm is required, not incidental: the tile is chosen at
                    # translate time, so a reused session would keep the arm it was built with.
                    sess = bench_mod._session(model, providers, device_index=device)
                    raw[m_rows][arm].append(_sample(sess, feeds, args.iters, args.warmup))
                    del sess
    finally:
        os.environ.pop(ROWS_ENV, None)
        if prev is not None:
            os.environ[ROWS_ENV] = prev

    rows = []
    for m_rows in M_VALUES:
        u = raw[m_rows]["untiled"]
        t = raw[m_rows]["tiled"]
        # Paired, because the two arms of a repeat are adjacent in time and share whatever the
        # machine was doing; the median of the per-repeat ratios is the estimate that survives a
        # single disturbed repeat.
        ratios = [a / b for a, b in zip(u, t)]
        rows.append(
            {
                "m": m_rows,
                "untiled_ms_median": statistics.median(u),
                "tiled_ms_median": statistics.median(t),
                "speedup_median_of_paired_ratios": statistics.median(ratios),
                "speedup_min": min(ratios),
                "speedup_max": max(ratios),
                "untiled_ms": u,
                "tiled_ms": t,
                "control": m_rows == 1,
            }
        )

    control = next(r for r in rows if r["control"])
    report = {
        "schema": "ab_row_tile/1",
        "shape": {"K": K, "N": N, "dtype": "f32", "block_size": 32, "bits": 4},
        "repeats": REPEATS,
        "iters": args.iters,
        "warmup": args.warmup,
        "device_index": device,
        "device_name": device_name,
        "arm_order": "alternated per repeat (untiled-first on even repeats)",
        "env": {ROWS_ENV: arms},
        "rows": rows,
        # Read every M>1 speedup against this: at M=1 both arms bind identical specialisation
        # constants and identical SPIR-V, so its spread is this harness's noise floor and nothing
        # narrower than it is a measurement.
        "noise_floor": {
            "m": 1,
            "speedup_median": control["speedup_median_of_paired_ratios"],
            "speedup_min": control["speedup_min"],
            "speedup_max": control["speedup_max"],
        },
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = Path(__file__).with_name(args.out)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n  A/B row tile — {K}x{N} q4 b32, {REPEATS} interleaved repeats on {device_name} (device {device})")
    print(f"  {'M':>3}  {'untiled ms':>11}  {'tiled ms':>10}  {'speedup':>9}  {'[min,max]':>16}")
    for r in rows:
        tag = "  <- control (identical pipelines)" if r["control"] else ""
        print(
            f"  {r['m']:>3}  {r['untiled_ms_median']:>11.3f}  {r['tiled_ms_median']:>10.3f}  "
            f"{r['speedup_median_of_paired_ratios']:>8.3f}x  "
            f"[{r['speedup_min']:.3f},{r['speedup_max']:.3f}]{tag}"
        )
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
