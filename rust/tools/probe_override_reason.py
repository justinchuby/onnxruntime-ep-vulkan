"""Falsifier for `net_benefit_override_reason`: the token must name the arm that rejected.

The wiring census reported `net_benefit_sole_island_overrides: 1` on a one-node elementwise chain
with no way to learn *what* the gate had rejected. `overrides` is a count of the override; the
reason it overrode lived only in `GateOutcome::SoleIslandOverride` and died at the counter
boundary. `counters.rs` now emits it.

R10 says a wiring claim is falsified by an artifact whose content **varies with its input**, so
this probe runs two graphs whose gate verdicts differ by construction and asserts the token differs
with them:

* a single elementwise node  -> the size arm rejects (`TOO_SMALL`), the sole-island override keeps
  it, and the token must say `TOO_SMALL`;
* a graph with no claimable node at all -> no island reaches the gate, no override happens, and
  the token must be `UNOBSERVABLE` rather than a verdict-shaped default (R12).

A token that read the same in both is not wired, whatever the source says.

Output: bench/results/override_reason-dev{N}.json
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_RESULTS = _ROOT / "bench" / "results"

EP_NAME = "vulkan"
EP_LIB = pathlib.Path(
    os.environ.get(
        "ONNXRUNTIME_VULKAN_EP_LIB",
        str(_ROOT / "rust" / "target" / "release" / "onnxruntime_vulkan_ep.dll"),
    )
)

# (case name, expected token). The expectation is written here, before the run, so the artifact
# scores against a prediction rather than being read for whatever it happens to say.
CASES = {
    "one_elementwise_node": "TOO_SMALL",
    "nothing_claimable": "UNOBSERVABLE",
}


def _model(case: str, path: pathlib.Path) -> None:
    import onnx
    from onnx import TensorProto, helper

    a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [4, 4])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [4, 4])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [4, 4])
    if case == "one_elementwise_node":
        nodes = [helper.make_node("Add", ["a", "b"], ["out"], name="lone_add")]
        graph = helper.make_graph(nodes, case, [a, b], [out])
    else:
        # `Det` is not in the registry at any dtype, so nothing is claimable, no island is ever
        # offered to the gate, and the frame contains no override event to have a reason for.
        # Every node in this graph must be unclaimable — a single stray `Add` would be claimed,
        # become a one-node island, and reproduce the first case instead of contrasting with it.
        det = helper.make_tensor_value_info("det", TensorProto.FLOAT, [])
        nodes = [helper.make_node("Det", ["a"], ["det"], name="unclaimable")]
        graph = helper.make_graph(nodes, case, [a], [det])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.save(model, str(path))


def run_child(case: str) -> None:
    import numpy as np
    import onnxruntime as ort

    model_path = _RESULTS / f"_override_reason_{case}.onnx"
    _model(case, model_path)
    try:
        ort.register_execution_provider_library(EP_NAME, str(EP_LIB))
    except Exception as exc:  # noqa: BLE001
        if "already registered" not in str(exc):
            raise
    sess = ort.InferenceSession(
        str(model_path),
        sess_options=ort.SessionOptions(),
        providers=[EP_NAME, "CPUExecutionProvider"],
    )
    feeds = {
        "a": np.ones((4, 4), dtype=np.float32),
        "b": np.full((4, 4), 2.0, dtype=np.float32),
    }
    feeds = {k: v for k, v in feeds.items() if k in {i.name for i in sess.get_inputs()}}
    sess.run(None, feeds)
    del sess


def main(argv: list[str]) -> int:
    if len(argv) > 1 and argv[1] == "--child":
        run_child(argv[2])
        return 0

    device = os.environ.get("ONNXRUNTIME_EP_VULKAN_DEVICE", "0")
    _RESULTS.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for case, predicted in CASES.items():
        counters_path = _RESULTS / f"override_reason-{case}-dev{device}.counters.json"
        counters_path.unlink(missing_ok=True)
        env = os.environ.copy()
        env["ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE"] = str(counters_path)
        proc = subprocess.run(
            [sys.executable, str(_HERE), "--child", case],
            env=env,
            capture_output=True,
            text=True,
        )
        row: dict = {"case": case, "predicted": predicted, "device": device}
        if proc.returncode != 0 or not counters_path.exists():
            # R13: an instrument error is never a detection. Quote the text, not a count.
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-6:]
            row["verdict"] = "ERROR(instrument)"
            row["failure_text"] = "\n".join(tail) or f"counters file missing: {counters_path}"
            print(f"=== {case}: ERROR(instrument): {row['failure_text']}")
            rows.append(row)
            continue
        doc = json.loads(counters_path.read_text())
        observed = doc.get("net_benefit_override_reason", "<absent>")
        row["verdict"] = "PASS" if observed == predicted else "FAIL(condition)"
        row["observed"] = observed
        row["counters"] = {
            k: doc.get(k, "<absent>")
            for k in (
                "claimed_nodes",
                "islands_offered",
                "viable_islands_retained",
                "net_benefit_gate",
                "net_benefit_gate_evaluations",
                "net_benefit_gate_bypasses",
                "net_benefit_sole_island_overrides",
                "net_benefit_override_reason",
            )
        }
        rows.append(row)
        print(f"=== {case}: predicted {predicted}, observed {observed} -> {row['verdict']}")
        for k, v in row["counters"].items():
            print(f"    {k:38s} {v}")

    out_path = _RESULTS / f"override_reason-dev{device}.json"
    out_path.write_text(json.dumps({"device": device, "rows": rows}, indent=2))
    print(f"\n[probe] -> {out_path}")

    tokens = {r.get("observed") for r in rows if r["verdict"] != "ERROR(instrument)"}
    if any(r["verdict"] == "ERROR(instrument)" for r in rows):
        print("VERDICT: ERROR(instrument)")
        return 2
    if len(tokens) < 2:
        print(f"VERDICT: FAIL(the token did not vary with its input: {tokens})")
        return 1
    bad = [r for r in rows if r["verdict"] != "PASS"]
    if bad:
        print(f"VERDICT: FAIL(prediction missed: {[(r['case'], r['observed']) for r in bad]})")
        return 1
    print("VERDICT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
