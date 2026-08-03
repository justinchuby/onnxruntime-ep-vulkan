"""Per-output comparison of two criterion-10 records: does the KV writeback path move the residual?

Criterion 10 compares all 65 outputs of Phi-3.5 against the CPU oracle.  Switch's
`872d739` made a directly-written device buffer authoritative, so the 64 KV `present`
tensors can now leave the fused island by a **different route** than the host staging
block they used to be read from.  A criterion that is blind to which route the bytes
took cannot notice a defect confined to one of them.

This tool answers one question with per-output granularity and no aggregate:

    for each of the 65 outputs, is the residual against the CPU oracle the same under
    both writeback routes, and is it the same as it was before the route existed?

An aggregate over 65 outputs can be dominated by one of them, and output 0 (the logits)
is exactly such an output: its median residual is 12 ULP against a KV baseline of 1.  A
mean or a max over the 65 is therefore a different claim from the one the criterion makes.

Usage:
    python tests/ops/compare_criterion10_records.py A.json B.json [--label-a X --label-b Y]

Exit status is 0 whatever it finds: this is an instrument, not a gate.  The gate is
tests/ops/test_criterion10.py.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

_KEYS = (
    "median_ulp_diff",
    "p99_ulp_diff",
    "max_ulp_diff",
    "max_abs_diff",
    "ulp_cancellation_elements",
)


def _curve(path: pathlib.Path, run: int = 0) -> tuple[dict[int, dict], dict]:
    record = json.loads(path.read_text(encoding="utf-8"))
    per_run = record["per_run"][run]
    curve = {c["output_index"]: c for c in per_run["ulp_curve"]}
    return curve, per_run


def compare(a_path: pathlib.Path, b_path: pathlib.Path) -> dict:
    a_curve, a_run = _curve(a_path)
    b_curve, b_run = _curve(b_path)

    indices = sorted(set(a_curve) | set(b_curve))
    moved: list[dict] = []
    for i in indices:
        a, b = a_curve.get(i), b_curve.get(i)
        if a is None or b is None:
            moved.append({"output_index": i, "status": "PRESENT_IN_ONE_ONLY"})
            continue
        deltas = {k: (a.get(k), b.get(k)) for k in _KEYS if a.get(k) != b.get(k)}
        if deltas:
            moved.append({"output_index": i, "deltas": deltas})

    return {
        "outputs_a": len(a_curve),
        "outputs_b": len(b_curve),
        "outputs_moved": len(moved),
        "moved": moved,
        "failing_a": a_run.get("oracle_failing_indices"),
        "failing_b": b_run.get("oracle_failing_indices"),
        "within_tolerance_a": a_run.get("oracle_outputs_within_tolerance"),
        "within_tolerance_b": b_run.get("oracle_outputs_within_tolerance"),
        "degenerate_a": a_run.get("oracle_outputs_degenerate"),
        "degenerate_b": b_run.get("oracle_outputs_degenerate"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    result = compare(pathlib.Path(args.a), pathlib.Path(args.b))
    result["label_a"] = args.label_a
    result["label_b"] = args.label_b

    print(f"{args.label_a}: {args.a}")
    print(f"{args.label_b}: {args.b}")
    print(
        f"  outputs: {result['outputs_a']} vs {result['outputs_b']}; "
        f"within tolerance {result['within_tolerance_a']} vs "
        f"{result['within_tolerance_b']}; degenerate "
        f"{result['degenerate_a']} vs {result['degenerate_b']}"
    )
    print(f"  failing indices: {result['failing_a']} vs {result['failing_b']}")
    print(f"  outputs whose per-output residual moved: {result['outputs_moved']} of 65")
    for entry in result["moved"]:
        print(f"    output {entry['output_index']}: {entry.get('deltas', entry.get('status'))}")
    if result["outputs_moved"] == 0:
        print(
            "  NOTE: a null reading. It is only admissible beside a demonstration that "
            "this comparison can produce a non-null one — see the decision record."
        )

    if args.json_out:
        pathlib.Path(args.json_out).write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"  wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
