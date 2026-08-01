"""Where do the eight permanent declines sit — at the edge of the island, or inside it?

Mouse, 2026-08-01.

`bench/island_attribution.py` answers "how many island *cuts* does each declined op create".
That was the right question when there were 33 islands. With one island it is not sufficient,
because a declined node can sit **between two claimed nodes that are in the same island** and
create zero cuts while still being a node the EP hands back mid-graph.

Nobody had asked that question of the current claim set, and the claim set moved on 2026-07-31
(`SimplifiedLayerNormalization` and `Gather` are now claimed). So this classifies every declined
node by its *position* relative to the claimed set:

* ``EDGE_ENTRY``    — no claimed producer feeds it; it may feed claimed nodes (prologue).
* ``EDGE_EXIT``     — no claimed consumer; it is fed by claimed nodes (epilogue).
* ``DETACHED``      — neither side claimed.
* ``INTERIOR``      — claimed producers **and** claimed consumers. This is the category that
  changes the answer: with a single island it would mean the fused node feeds a CPU node that
  feeds the fused node back, which is a cycle ORT cannot fuse. Finding one is a falsifier for
  ``island_count == 1``, not a backlog item.

Reads the CLAIM_LOG produced by ``bench/island_attribution.py`` (or any run with
``ONNXRUNTIME_EP_VULKAN_CLAIM_LOG`` set) — it does not run the model itself, so it costs nothing
and cannot perturb what it measures.

Usage::

    python rust/tools/probe_decline_position.py [claim_log.jsonl]
"""

from __future__ import annotations

import json
import pathlib
import sys
from collections import defaultdict

_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
_RESULTS = _ROOT / "bench" / "results"
_DEFAULT_LOG = _RESULTS / "claim_log_attribution.jsonl"
_ONNX = pathlib.Path(
    r"C:\Users\justinchu\.foundry\cache\models"
    r"\Microsoft\Phi-3.5-mini-instruct-cuda-gpu"
    r"\cuda-int4-rtn-block-32"
    r"\phi-3.5-mini-instruct-cuda-int4-rtn-block-32.onnx"
)


def main() -> int:
    log_path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_LOG
    if not log_path.exists():
        # R13: an instrument that cannot run is an ERROR, never a clean bill of health.
        print(f"ERROR(instrument): CLAIM_LOG not found at {log_path}")
        return 2

    import onnx

    records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    claimed = {r["node"] for r in records if r.get("claimed") and r.get("node")}
    declined = [r for r in records if not r.get("claimed") and r.get("node")]

    graph = onnx.load(str(_ONNX), load_external_data=False).graph
    producer: dict[str, str] = {}
    consumers: dict[str, list[str]] = defaultdict(list)
    nodes: dict[str, dict] = {}
    for n in graph.node:
        op = f"{n.domain}::{n.op_type}" if n.domain not in ("", "ai.onnx") else n.op_type
        nodes[n.name] = {"op": op, "inputs": list(n.input), "outputs": list(n.output)}
        for o in n.output:
            if o:
                producer[o] = n.name
        for i in n.input:
            if i:
                consumers[i].append(n.name)

    # Union-find over claimed nodes → island membership.
    parent = {n: n for n in claimed}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for n in claimed:
        info = nodes.get(n)
        if not info:
            continue
        for o in info["outputs"]:
            for c in consumers.get(o, []):
                if c in claimed:
                    a, b = find(n), find(c)
                    if a != b:
                        parent[a] = b

    islands = {find(n) for n in claimed}
    print(f"claimed={len(claimed)}  declined={len(declined)}  islands={len(islands)}")

    rows = []
    for r in declined:
        info = nodes.get(r["node"])
        if not info:
            continue
        preds = {producer[i] for i in info["inputs"] if i in producer and producer[i] in claimed}
        succs = {
            c
            for o in info["outputs"]
            for c in consumers.get(o, [])
            if c in claimed
        }
        if preds and succs:
            category = "INTERIOR"
        elif succs:
            category = "EDGE_ENTRY"
        elif preds:
            category = "EDGE_EXIT"
        else:
            category = "DETACHED"
        pred_islands = {find(p) for p in preds}
        succ_islands = {find(s) for s in succs}
        cuts = sum(1 for a in pred_islands for b in succ_islands if a != b)
        rows.append(
            {
                "node": r["node"],
                "op": r.get("op", info["op"]),
                "code": r.get("code"),
                "category": category,
                "claimed_preds": sorted(preds),
                "claimed_succs": sorted(succs),
                "cuts": cuts,
            }
        )

    width = max((len(r["op"]) for r in rows), default=10)
    for r in sorted(rows, key=lambda r: (r["category"], r["op"])):
        print(
            f"  {r['category']:<11} {r['op']:<{width}}  [{r['code']}]  "
            f"preds={len(r['claimed_preds'])} succs={len(r['claimed_succs'])} cuts={r['cuts']}"
        )

    out = _RESULTS / f"decline_position-{log_path.stem}.json"
    out.write_text(
        json.dumps(
            {
                "claimed": len(claimed),
                "declined": len(declined),
                "islands": len(islands),
                "rows": rows,
            },
            indent=2,
        )
    )
    print(f"[probe] → {out}")

    total_cuts = sum(r["cuts"] for r in rows)
    interior = [r for r in rows if r["category"] == "INTERIOR"]
    ok = True
    if total_cuts != 0:
        print(f"FAIL(cut-creating declines): {total_cuts} cut-instance(s)")
        ok = False
    if interior and len(islands) == 1:
        print(
            "FAIL(interior decline in a single island): "
            + ", ".join(r["op"] for r in interior)
            + " — a claimed→declined→claimed path inside one island is a fusion cycle"
        )
        ok = False
    print("VERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
