#!/usr/bin/env python
"""What does this EP decline, on a *real* model, and why? Owner: Mouse.

WHY THIS EXISTS
---------------
Coverage work that starts from a list of ONNX operators completes a taxonomy. Coverage work
that starts from a graph closes a model. `probe_phi35_claim_reading.py` answers
"how many nodes did the EP claim" for one model; it does not say *which op types* were
declined, under which decline code, or how many nodes each decline costs. Selecting the next
kernel needs that breakdown, per model, keyed by op.

WHAT IT MEASURES, AND FROM WHERE
--------------------------------
Two independent readings, deliberately not one:

  graph census   the ONNX graph itself, parsed with `load_external_data=False` (an 11.8 GB
                 `.onnx.data` is not needed to count node types). This is the denominator and
                 it is a fact about the model, not about us.

  claim census   `ONNXRUNTIME_EP_VULKAN_CLAIM_LOG`, written by `registry::claim_decision` in
                 the running EP, one JSON line per node. This is the numerator and it is a
                 fact about the build. It is read, never re-derived: the registry, the ledger
                 and the shape classifier all live in the DLL and there is exactly one
                 implementation of each.

A run that produces no claim log says nothing and reports `ERROR(instrument)` rather than
falling back to predicting the answer from the registry source. Predicting it is how a census
ends up measuring the census.

NO CLOCK. Counts only — the machine is permanently contended and a node count needs no timer.

USAGE
    python rust/tools/probe_model_op_census.py --model <path.onnx> [--name tag]
                                               [--out bench/results/op_census_<tag>.json]
Requires `ONNXRUNTIME_VULKAN_EP_LIB` to point at the built EP.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]

CHILD = "--census-child"


def _child(model: str) -> int:
    import onnxruntime as ort

    ort.register_execution_provider_library(
        "VulkanExecutionProvider", os.environ["ONNXRUNTIME_VULKAN_EP_LIB"]
    )
    so = ort.SessionOptions()
    so.log_severity_level = 3
    ort.InferenceSession(
        model, so, providers=["VulkanExecutionProvider", "CPUExecutionProvider"]
    )
    return 0


def graph_census(model: pathlib.Path) -> tuple[dict[str, int], dict]:
    import onnx

    m = onnx.load(str(model), load_external_data=False)
    counts: collections.Counter[str] = collections.Counter()
    for n in m.graph.node:
        counts[f"{n.domain}::{n.op_type}" if n.domain else n.op_type] += 1
    meta = {
        "nodes": len(m.graph.node),
        "opset_import": {i.domain or "ai.onnx": i.version for i in m.opset_import},
        "ir_version": m.ir_version,
    }
    return dict(counts), meta


def claim_census(model: pathlib.Path, log: pathlib.Path) -> tuple[list[dict], str]:
    if log.exists():
        log.unlink()
    env = dict(os.environ)
    env["ONNXRUNTIME_EP_VULKAN_CLAIM_LOG"] = str(log)
    r = subprocess.run(
        [sys.executable, str(pathlib.Path(__file__).resolve()), CHILD, str(model)],
        env=env,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=7200,
    )
    if not log.is_file():
        return [], f"ERROR(instrument): no claim log written.\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}"
    rows = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        return [], "ERROR(instrument): claim log is empty; this run says nothing."
    return rows, ""


def main() -> int:
    if len(sys.argv) > 2 and sys.argv[1] == CHILD:
        return _child(sys.argv[2])

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if "ONNXRUNTIME_VULKAN_EP_LIB" not in os.environ:
        print("ERROR(instrument): ONNXRUNTIME_VULKAN_EP_LIB is unset — this run would read nothing.")
        return 2

    model = pathlib.Path(args.model)
    tag = args.name or model.stem[:40]
    outdir = REPO / "bench" / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    out = pathlib.Path(args.out) if args.out else outdir / f"op_census_{tag}.json"

    graph, meta = graph_census(model)
    rows, err = claim_census(model, outdir / f"_claim_log_{tag}.jsonl")
    if err:
        print(err)
        return 2

    per_op: dict[str, dict] = {}
    for r in rows:
        e = per_op.setdefault(
            r["op"],
            {"seen": 0, "claimed": 0, "declined": 0, "codes": collections.Counter(),
             "proof_keys": collections.Counter(), "ledger_hits": 0},
        )
        e["seen"] += 1
        if r.get("claimed"):
            e["claimed"] += 1
        else:
            e["declined"] += 1
            e["codes"][r.get("code") or "?"] += 1
        if r.get("ledger_hit"):
            e["ledger_hits"] += 1
        if r.get("proof_key"):
            e["proof_keys"][r["proof_key"]] += 1

    # Ops present in the graph that the registry never even saw a decision for: control-flow
    # bodies and `max_claim_ops` exclusions short-circuit before `claim_decision`. Reported as
    # its own class rather than folded into "declined", because it is not a claim decision.
    unseen = {op: n for op, n in graph.items() if op not in per_op}

    claimed = sum(e["claimed"] for e in per_op.values())
    declined = sum(e["declined"] for e in per_op.values())

    record = {
        "model": str(model),
        "tag": tag,
        "graph": meta,
        "graph_op_counts": graph,
        "decisions_recorded": len(rows),
        "claimed_nodes": claimed,
        "declined_nodes": declined,
        "no_decision_nodes": sum(unseen.values()),
        "no_decision_ops": unseen,
        "per_op": {
            op: {
                "seen": e["seen"],
                "claimed": e["claimed"],
                "declined": e["declined"],
                "codes": dict(e["codes"]),
                "ledger_hits": e["ledger_hits"],
                "proof_keys": dict(e["proof_keys"]),
            }
            for op, e in sorted(per_op.items())
        },
        "ep_lib": os.environ["ONNXRUNTIME_VULKAN_EP_LIB"],
    }
    out.write_text(json.dumps(record, indent=1), encoding="utf-8")

    print(f"=== {tag}: {meta['nodes']} nodes, opsets {meta['opset_import']}")
    print(f"    claimed {claimed} / declined {declined} / no-decision {sum(unseen.values())}")
    print(f"    {'op':50s} {'seen':>5s} {'clm':>5s} {'dec':>5s}  codes")
    for op, e in sorted(per_op.items(), key=lambda kv: -kv[1]["declined"]):
        if not e["declined"]:
            continue
        print(f"    {op:50s} {e['seen']:5d} {e['claimed']:5d} {e['declined']:5d}  {dict(e['codes'])}")
    for op, n in sorted(unseen.items(), key=lambda kv: -kv[1]):
        print(f"    {op:50s} {n:5d}     -     -  NO CLAIM DECISION RECORDED")
    print(f"    -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
