#!/usr/bin/env python
"""If this EP registered one more op, how much would it actually *run*? Owner: Mouse.

WHY THIS EXISTS
---------------
The op census ranks candidate kernels by the node count of the largest unregistered op. On
BERT-SQuAD-12 that ranking says `MatMul` x95. The ranking is measured off `claimed_nodes`, and
on 2026-08-04 `claimed_nodes` was shown not to be what runs:

    claimed_nodes 480   islands_offered 3   viable_islands_retained 3   dispatches_executed 3

**BERT executes three nodes.** The other 477 claims are claimed at `GetCapability` and then
dropped by the partitioner's net-benefit gate -- `net_benefit_gate_clusters_seen 145` -- because
they are singletons and pairs stranded between unregistered ops. A kernel selected to maximise
*claims* can therefore add 95 claims and zero dispatches.

So the criterion has to be stated over what executes: **which single op registration most
increases the total size of the retained islands**, where an island is a connected run of
claimable nodes in the graph. That is a question about the graph's topology and the set of ops
we claim, and it can be answered *before* writing any kernel.

WHAT IT MEASURES
----------------
The claimable set is read off a real claim log (`probe_model_op_census.py` writes one), never
re-derived from the registry source -- there is one registry and it lives in the DLL.
An op is "claimable" if the log shows at least one node of that op type claimed, or the op is
named in `--add`.

Islands are connected components over the graph's data edges, restricted to claimable nodes.
The `--min-nodes` floor mirrors `ops::partition`'s minimum-island rule; `--anchors` mirrors its
anchor exemption. Neither is the partitioner -- this is a *ranking* instrument, and it says so:
it reports island structure, and the partitioner's cost model is the thing that finally decides.
The number to read is the **delta**, which is much more robust than either absolute.

NO CLOCK.

USAGE
    python rust/tools/probe_island_counterfactual.py --model <path.onnx>
        --claim-log bench/results/_claim_log_<tag>.jsonl
        [--add MatMul --add Transpose ...] [--top 12] [--out <json>]
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib

# Mirrors `ops::partition::is_anchor` closely enough to rank with. Not authoritative; the
# partitioner in the DLL is. Kept short and named so a reader can check it against that list.
DEFAULT_ANCHORS = ("Conv", "Gemm", "MatMul", "MatMulNBits", "Attention", "GroupQueryAttention")


def islands(nodes, claimable, producer) -> list[list[str]]:
    """Connected components of claimable nodes, joined by a data edge between two of them."""
    idx = {n.name or f"#{i}": i for i, n in enumerate(nodes)}
    keep = [i for i, n in enumerate(nodes) if n.op_type in claimable]
    keepset = set(keep)
    adj: dict[int, set[int]] = {i: set() for i in keep}
    for i in keep:
        for inp in nodes[i].input:
            p = producer.get(inp)
            if p is not None and p in keepset and p != i:
                adj[i].add(p)
                adj[p].add(i)
    seen: set[int] = set()
    out = []
    for i in keep:
        if i in seen:
            continue
        stack, comp = [i], []
        seen.add(i)
        while stack:
            j = stack.pop()
            comp.append(j)
            for k in adj[j]:
                if k not in seen:
                    seen.add(k)
                    stack.append(k)
        out.append([nodes[j].op_type for j in comp])
    assert idx is not None
    return out


def retained(comps, min_nodes: int, anchors: set[str]) -> tuple[int, int]:
    """(nodes in retained islands, number of retained islands) under the min-island rule."""
    n = 0
    k = 0
    for c in comps:
        if len(c) >= min_nodes or any(op in anchors for op in c):
            n += len(c)
            k += 1
    return n, k


def main() -> int:
    import onnx

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--claim-log", required=True)
    ap.add_argument("--add", action="append", default=[])
    ap.add_argument("--min-nodes", type=int, default=2)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--out")
    args = ap.parse_args()

    rows = [
        json.loads(line)
        for line in pathlib.Path(args.claim_log).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        print("ERROR(instrument): the claim log is empty; nothing was measured")
        return 2
    claimed_ops = {r["op"].split("::")[-1] for r in rows if r["claimed"]}
    all_ops = collections.Counter(r["op"].split("::")[-1] for r in rows)
    unregistered = {
        r["op"].split("::")[-1] for r in rows if r.get("code") == "not-registered"
    }

    m = onnx.load(args.model, load_external_data=False)
    nodes = list(m.graph.node)
    producer: dict[str, int] = {}
    for i, n in enumerate(nodes):
        for o in n.output:
            producer[o] = i

    anchors = set(DEFAULT_ANCHORS)
    base_comps = islands(nodes, claimed_ops, producer)
    base_n, base_k = retained(base_comps, args.min_nodes, anchors)
    print(f"baseline claimable ops: {len(claimed_ops)}")
    print(f"baseline islands: {base_k} retained covering {base_n} nodes "
          f"(of {len(base_comps)} components, {sum(len(c) for c in base_comps)} claimable nodes)")

    candidates = sorted(unregistered, key=lambda o: -all_ops[o])
    if args.add:
        candidates = args.add
    results = []
    for op in candidates:
        n, k = retained(islands(nodes, claimed_ops | {op}, producer), args.min_nodes, anchors)
        results.append(
            {"op": op, "graph_count": all_ops.get(op, 0), "retained_nodes": n,
             "retained_islands": k, "delta_nodes": n - base_n}
        )
    results.sort(key=lambda r: -r["delta_nodes"])

    print(f"\n{'op':<20}{'in graph':>10}{'retained':>10}{'islands':>9}{'delta':>8}")
    for r in results[: args.top]:
        print(f"{r['op']:<20}{r['graph_count']:>10}{r['retained_nodes']:>10}"
              f"{r['retained_islands']:>9}{r['delta_nodes']:>+8}")

    # The cumulative reading: adding the top op alone, then the top two together, and so on.
    # Fragmentation is not additive, and a set of ops can be worth far more than the sum of the
    # ops taken one at a time. Reporting only the singles is how "MatMul x95" got its ranking.
    cum = []
    have = set(claimed_ops)
    for r in results[: args.top]:
        have = have | {r["op"]}
        n, k = retained(islands(nodes, have, producer), args.min_nodes, anchors)
        cum.append({"added": sorted(have - claimed_ops), "retained_nodes": n,
                    "retained_islands": k, "delta_nodes": n - base_n})
    print("\ncumulative, greedy in the order above:")
    for c in cum:
        print(f"  +{','.join(c['added']):<58} {c['retained_nodes']:>5} nodes "
              f"in {c['retained_islands']:>3} islands  ({c['delta_nodes']:+})")

    report = {
        "model": args.model,
        "claim_log": args.claim_log,
        "min_nodes": args.min_nodes,
        "anchors": sorted(anchors),
        "baseline": {"retained_nodes": base_n, "retained_islands": base_k},
        "singles": results,
        "cumulative": cum,
    }
    if args.out:
        p = pathlib.Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
