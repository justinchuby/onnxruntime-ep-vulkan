"""Print criterion 10's per-output verdict table — 65 rows, never one aggregate.

An aggregate over 65 outputs can be dominated by one of them, and on Phi-3.5 one of them
does dominate: output 0's median residual is 12 ULP against a KV baseline of 1.  A mean or
a max over the 65 is therefore a *different claim* from the one the criterion makes, which
is that **every** output agrees with the CPU oracle.  So the record prints all 65 rows and
the reader does the reducing, knowing what was reduced.

Usage:  python tests/ops/print_criterion10_table.py bench/results/criterion10-dev0.json
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import _kv_depth  # noqa: E402


def main(argv: list[str]) -> int:
    path = pathlib.Path(argv[1])
    record = json.loads(path.read_text(encoding="utf-8"))
    run = record["per_run"][0]
    # NEVER `output_coverage.per_output` — that dict is alphabetised by sort_keys=True and
    # this model's session order is not its own sort.  Reading it as an ordering attributes
    # every KV residual to the wrong layer.
    names = record.get("output_names")
    if names:
        _kv_depth.assert_names_are_session_order(names)
    else:
        names = [
            "logits" if i == 0 else "present.{}.{}".format(*_kv_depth.layer_of_index(i))
            for i in range(len(run["ulp_curve"]))
        ]
    failing = set(run.get("oracle_failing_indices") or [])
    degenerate = set(run.get("oracle_degenerate_indices") or [])

    print(f"artifact : {record['artifact']}")
    print(f"device   : {record.get('device_name')}  (read off the run)")
    print(f"route    : {(record.get('kv_writeback_route') or {}).get('route')}")
    print(f"verdict  : {record['verdict']}")
    print(
        f"{'idx':>3} {'name':<18} {'verdict':<12} {'median_ulp':>10} "
        f"{'p99_ulp':>9} {'max_ulp':>10} {'max_abs':>12} {'cancel':>7}"
    )
    for entry in run["ulp_curve"]:
        i = entry["output_index"]
        if i in degenerate:
            verdict = "NOT_COMPARED"
        elif i in failing:
            verdict = "OUT_OF_TOL"
        else:
            verdict = "WITHIN_TOL"
        name = names[i] if i < len(names) else entry.get("name") or ""
        med, p99 = entry.get("median_ulp_diff"), entry.get("p99_ulp_diff")
        mx, ma = entry.get("max_ulp_diff"), entry.get("max_abs_diff")
        can = entry.get("ulp_cancellation_elements")
        fmt = lambda v, w, p=2: (f"{v:>{w}.{p}f}" if isinstance(v, (int, float)) else f"{'-':>{w}}")  # noqa: E731
        print(
            f"{i:>3} {name:<18} {verdict:<12} {fmt(med, 10)} {fmt(p99, 9)} "
            f"{fmt(mx, 10, 0)} {fmt(ma, 12, 8)} {can if can is not None else '-':>7}"
        )
    print(
        "\nNo aggregate over these 65 rows is printed on purpose: output 0 dominates any "
        "reduction, and 'the mean output agrees' is not the claim criterion 10 makes."
    )
    curve = _kv_depth.depth_curve(
        [c["median_ulp_diff"] for c in run["ulp_curve"]],
        names if record.get("output_names") else None,
    )
    print("\nBy layer depth (median ULP, key/value):")
    print("  " + "  ".join(f"L{r['layer']}:{r['key']}/{r['value']}" for r in curve))
    print(f"  over the predicted {_kv_depth.LAYER_PREDICTED_CEILING} ULP band: "
          f"{_kv_depth.depth_exceedances(curve) or 'none'}")
    print(f"  largest layer-to-layer step: key {_kv_depth.largest_step(curve, 'key')}, "
          f"value {_kv_depth.largest_step(curve, 'value')}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
