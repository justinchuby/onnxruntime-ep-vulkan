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
Three independent readings, deliberately not one:

  graph census   the ONNX graph itself, parsed with `load_external_data=False` (an 11.8 GB
                 `.onnx.data` is not needed to count node types). This is the denominator and
                 it is a fact about the model, not about us.

  claim census   `ONNXRUNTIME_EP_VULKAN_CLAIM_LOG`, written by `registry::claim_decision` in
                 the running EP, one JSON line per node. This is the numerator and it is a
                 fact about the build. It is read, never re-derived: the registry, the ledger
                 and the shape classifier all live in the DLL and there is exactly one
                 implementation of each.

  counters       `ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE`, dumped by the EP at teardown:
                 `islands_offered`, `viable_islands_retained`, `dispatches_executed`.

WHY THE HEADLINE MOVED, 2026-08-04
----------------------------------
This probe used to headline `claimed_nodes` and never open the counters at all. On 2026-08-04
`claimed_nodes` was shown not to be what runs: BERT-SQuAD-12 claimed **481** nodes at
`GetCapability` and executed **four**, the other 477 dropped by the partitioner's net-benefit
gate as singletons stranded between unregistered ops. Registering `MatMul` moved the claim by
95 nodes and the dispatch count by one.

A number everybody quotes and nobody executes is a CI-red failure one layer in, so the headline
is now `dispatches_executed` and `claimed_nodes` is reported beside it as the *upper bound* it
always was. The two are reported together precisely because they disagree, and the disagreement
is the finding.

`dispatches_executed` is only meaningful if something was *run*. Session creation alone never
dispatches, so without `--run` this probe reports `not-measured` rather than `0` — the
distinction between "we looked and it was zero" and "nothing looked" is the whole point of the
change, and collapsing it would reintroduce the defect one level down.

NO CLOCK. Counts only — the machine is permanently contended and a node count needs no timer.

USAGE
    python rust/tools/probe_model_op_census.py --model <path.onnx> [--name tag] [--run]
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

# Lifted out of the counters artifact. Counts and states, never durations.
COUNTER_KEYS = (
    "claimed_nodes",
    "islands_offered",
    "viable_islands_retained",
    "dispatches_executed",
    "net_benefit_gate_clusters_seen",
)


def _feed(sess, seed: int = 0) -> dict:
    """Feeds shaped from the session's own declared inputs, with symbolic extents pinned to 1.

    Identical in intent to `probe_model_output_agreement.py`'s: this probe answers *how many
    dispatches ran*, not *were they right*, and the two questions want the same feed so that a
    reader can put the two artifacts side by side.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    feeds = {}
    for i in sess.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in i.shape]
        t = i.type
        if "int64" in t:
            feeds[i.name] = rng.integers(0, 2, size=shape, dtype=np.int64)
        elif "int32" in t:
            feeds[i.name] = rng.integers(0, 2, size=shape, dtype=np.int32)
        elif "float16" in t:
            feeds[i.name] = rng.standard_normal(shape).astype(np.float16)
        else:
            feeds[i.name] = rng.standard_normal(shape).astype(np.float32)
    return feeds


def _child(model: str, run: bool) -> int:
    import onnxruntime as ort

    ort.register_execution_provider_library(
        "VulkanExecutionProvider", os.environ["ONNXRUNTIME_VULKAN_EP_LIB"]
    )
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(
        model, so, providers=["VulkanExecutionProvider", "CPUExecutionProvider"]
    )
    # The vacuous-pass guard `probe_model_output_agreement.py` needed: a session that silently
    # fell back to CPU would report `dispatches_executed 0` and read as a coverage regression
    # rather than as an instrument failure.
    if "VulkanExecutionProvider" not in sess.get_providers():
        print(f"CHILD-ERROR: the Vulkan EP is not in the providers: {sess.get_providers()}")
        return 3
    if run:
        sess.run(None, _feed(sess))
    del sess
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


def claim_census(
    model: pathlib.Path, log: pathlib.Path, counters: pathlib.Path, run: bool
) -> tuple[list[dict], dict, str]:
    if log.exists():
        log.unlink()
    counters.unlink(missing_ok=True)
    env = dict(os.environ)
    env["ONNXRUNTIME_EP_VULKAN_CLAIM_LOG"] = str(log)
    env["ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE"] = str(counters)
    r = subprocess.run(
        [sys.executable, str(pathlib.Path(__file__).resolve()), CHILD, str(model),
         "1" if run else "0"],
        env=env,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=7200,
    )
    if not log.is_file():
        return [], {}, (
            f"ERROR(instrument): no claim log written.\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}"
        )
    rows = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        return [], {}, "ERROR(instrument): claim log is empty; this run says nothing."
    if r.returncode != 0:
        return rows, {}, (
            f"ERROR(instrument): the child exited {r.returncode}; the claim log is readable but "
            f"the run is not.\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}"
        )
    # Counters are reported as read, and their *absence* is reported as absence. A missing
    # counters file is not zero dispatches.
    if not counters.is_file():
        return rows, {"_status": "counters-file-never-written"}, ""
    doc = json.loads(counters.read_text(encoding="utf-8"))
    got = {k: doc.get(k, "<absent>") for k in COUNTER_KEYS}
    got["_status"] = "measured" if run else "session-only"
    if not run:
        # Session creation never dispatches. Saying `0` here would be true of the number and
        # false of the claim it invites; `not-measured` is the honest word.
        got["dispatches_executed"] = "not-measured (no inference was run; pass --run)"
    return rows, got, ""


def main() -> int:
    if len(sys.argv) > 2 and sys.argv[1] == CHILD:
        return _child(sys.argv[2], len(sys.argv) > 3 and sys.argv[3] == "1")

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", default="")
    ap.add_argument("--out", default="")
    ap.add_argument(
        "--run",
        action="store_true",
        help="execute one inference so `dispatches_executed` is measured rather than absent",
    )
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
    rows, counters, err = claim_census(
        model,
        outdir / f"_claim_log_{tag}.jsonl",
        outdir / f"_counters_{tag}.json",
        args.run,
    )
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
        # The headline, first in the record as well as first in the printout. `claimed_nodes` is
        # the upper bound; `dispatches_executed` is what ran.
        "dispatches_executed": counters.get("dispatches_executed", "<no counters>"),
        "counters": counters,
        "inference_run": bool(args.run),
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
    print(f"    DISPATCHES EXECUTED  {record['dispatches_executed']}")
    print(
        f"    claimed {claimed} (upper bound) / declined {declined} / "
        f"no-decision {sum(unseen.values())}"
    )
    if counters:
        for k in COUNTER_KEYS:
            if k in counters and k != "dispatches_executed":
                print(f"      {k:34s} {counters[k]}")
        if counters.get("_status") != "measured":
            print(f"      counters status: {counters['_status']}")
    else:
        print("      counters: <none written> — dispatches_executed is unknown, not zero")
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
