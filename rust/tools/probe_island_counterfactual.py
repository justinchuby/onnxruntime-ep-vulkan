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

**It reports a bracket, not a number, and the bracket is the point (repaired 2026-08-04).**
The first version of this instrument ranked *op types*: it assumed that registering `Reshape`
made every `Reshape` node claimable. The EP claims *nodes*. Measured:

    MatMul   95 nodes in BERT, ranked +135 -- registered, claimed 1  (94 declined unknown-rank)
    Reshape  59 nodes in BERT, ranked +167 -- registered, claimed 0  (53 unknown-rank, 4 dtype,
                                                                      2 shape)

Both predictions were made by this file and both were wrong by two orders of magnitude, in the
same direction, for the same reason. So it now reports two readings of every candidate:

* **optimistic** -- every node of that op type becomes claimable. This is the old number, kept
  because it is the ceiling and a ceiling is worth knowing.
* **gated** -- only those nodes whose own claim-log row shows ORT resolved a rank for every
  operand *and* for the output. That is not the EP's gate (dtype, attributes and per-op rules
  are not modelled here) so it is not a floor either -- but it is the one precondition that
  every op in this crate shares, and it is what both misses above were made of.

The baseline is per-node too: a node counts as claimable only if the log says *that node* was
claimed. The old baseline treated all 364 `Add` nodes as claimable when 182 were.

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

# Mirrors `ops::partition::is_heavy_op_family` — the op-name-only question — not
# `ops::partition::is_anchor`, which since issue #73 also requires a resident weight operand
# (`WeightOperand::Present`) on top of the op name. This list is the **retired**, name-only anchor
# policy: it silently over-counts anchors relative to the DLL's actual behaviour whenever a listed
# op's arithmetic-heavy operand is a runtime activation rather than a constant (a `Gemm`/`MatMul`
# with a non-initializer `B`), because this tool's claim log carries no per-input constancy data
# and therefore cannot ask the question `is_anchor` now asks. Treat a count produced from this list
# as an **optimistic ceiling** on anchors, not a prediction — the partitioner in the DLL is
# authoritative and may claim fewer islands than this tool ranks. Kept short and named so a reader
# can check it against `is_heavy_op_family`'s list; upgrading it to track `is_anchor` exactly would
# require this tool's claim log to record, per input, whether the producing value was a graph
# initializer, which it does not do today.
DEFAULT_ANCHORS = ("Conv", "Gemm", "MatMul", "MatMulNBits", "Attention", "GroupQueryAttention")


def islands(nodes, keepset: set[int], producer) -> list[list[str]]:
    """Connected components of claimable nodes, joined by a data edge between two of them."""
    keep = sorted(keepset)
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


def ranks_resolved(row: dict) -> bool:
    """Did ORT establish a rank for every operand and for the output of this node?

    This is the one precondition every op in this crate shares, and it is what both of the
    instrument's historical misses were made of: `claim::check_shape` is reached only with an
    `EdgeType` that has dims, and `[]` -- which ORT emits both for a genuine scalar and for a
    rank it never established -- fails every downstream rule that needs to index.

    A missing key is not treated as resolved. An absent reading is not a passing reading; the
    node is counted as *not* gated-claimable and the caller reports how many rows were absent.
    """
    ins = row.get("input_shapes")
    outs = row.get("output_shapes")
    if not isinstance(ins, list) or not isinstance(outs, list) or not outs:
        return False
    return all(isinstance(s, list) and len(s) > 0 for s in ins + outs)


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

    # Per node, not per op type. A node re-offered in a later pass keeps the *best* verdict it
    # ever got, because that is the one the partitioner acted on.
    claimed_node: dict[str, bool] = {}
    gated_node: dict[str, bool] = {}
    for r in rows:
        name = r.get("node")
        if not name:
            continue
        claimed_node[name] = claimed_node.get(name, False) or bool(r.get("claimed"))
        gated_node[name] = gated_node.get(name, False) or ranks_resolved(r)

    all_ops = collections.Counter(r["op"].split("::")[-1] for r in rows)
    unregistered = {r["op"].split("::")[-1] for r in rows if r.get("code") == "not-registered"}

    m = onnx.load(args.model, load_external_data=False)
    nodes = list(m.graph.node)
    producer: dict[str, int] = {}
    for i, n in enumerate(nodes):
        for o in n.output:
            producer[o] = i

    # How much of the graph the log can actually speak about. ORT runs its own graph
    # transformers before `GetCapability`, so a node in the file need not be a node in the log.
    # Reported rather than assumed away: a low match rate makes every delta below suspect, and
    # an instrument that hides that is the "clean because it is not looking" failure again.
    matched = sum(1 for n in nodes if n.name in claimed_node)
    print(
        f"claim log covers {matched}/{len(nodes)} graph nodes by name "
        f"({100.0 * matched / max(1, len(nodes)):.1f}%); unmatched nodes are never claimable here"
    )
    if matched == 0:
        print(
            "ERROR(instrument): no graph node name appears in the claim log. The delta columns "
            "would all be zero and would read as 'this op is worthless' rather than 'this "
            "instrument could not see'."
        )
        return 2

    anchors = set(DEFAULT_ANCHORS)
    base_keep = {i for i, n in enumerate(nodes) if claimed_node.get(n.name)}
    base_comps = islands(nodes, base_keep, producer)
    base_n, base_k = retained(base_comps, args.min_nodes, anchors)
    print(
        f"baseline islands: {base_k} retained covering {base_n} nodes "
        f"(of {len(base_comps)} components, {len(base_keep)} claimed nodes)"
    )

    candidates = sorted(unregistered, key=lambda o: -all_ops[o])
    if args.add:
        candidates = args.add

    def add_set(op: str, gated: bool) -> set[int]:
        out = set()
        for i, n in enumerate(nodes):
            if n.op_type != op:
                continue
            if gated and not gated_node.get(n.name, False):
                continue
            if not gated and n.name not in claimed_node:
                continue
            out.add(i)
        return out

    results = []
    for op in candidates:
        opt_keep = base_keep | add_set(op, gated=False)
        gat_keep = base_keep | add_set(op, gated=True)
        opt_n, opt_k = retained(islands(nodes, opt_keep, producer), args.min_nodes, anchors)
        gat_n, gat_k = retained(islands(nodes, gat_keep, producer), args.min_nodes, anchors)
        results.append(
            {
                "op": op,
                "graph_count": all_ops.get(op, 0),
                "nodes_rank_resolved": len(add_set(op, gated=True)),
                "retained_nodes": opt_n,
                "retained_islands": opt_k,
                "delta_nodes": opt_n - base_n,
                "gated_retained_nodes": gat_n,
                "gated_retained_islands": gat_k,
                "gated_delta_nodes": gat_n - base_n,
            }
        )
    results.sort(key=lambda r: (-r["gated_delta_nodes"], -r["delta_nodes"]))

    print(
        f"\n{'op':<20}{'in graph':>9}{'ranked':>8}{'optimistic':>12}{'gated':>8}"
        f"{'islands':>9}"
    )
    for r in results[: args.top]:
        print(
            f"{r['op']:<20}{r['graph_count']:>9}{r['nodes_rank_resolved']:>8}"
            f"{r['delta_nodes']:>+12}{r['gated_delta_nodes']:>+8}"
            f"{r['gated_retained_islands']:>9}"
        )
    print(
        "  'ranked' = nodes of that op whose every operand and output has a rank in the claim "
        "log.\n  'optimistic' is the ceiling; 'gated' is what survives the one precondition "
        "every op shares.\n  Neither is a promise: dtype, attribute and per-op rules are not "
        "modelled here."
    )

    # The cumulative reading: adding the top op alone, then the top two together, and so on.
    # Fragmentation is not additive, and a set of ops can be worth far more than the sum of the
    # ops taken one at a time. Reporting only the singles is how "MatMul x95" got its ranking.
    cum = []
    opt_have = set(base_keep)
    gat_have = set(base_keep)
    added: list[str] = []
    for r in results[: args.top]:
        added.append(r["op"])
        opt_have |= add_set(r["op"], gated=False)
        gat_have |= add_set(r["op"], gated=True)
        opt_n, opt_k = retained(islands(nodes, opt_have, producer), args.min_nodes, anchors)
        gat_n, gat_k = retained(islands(nodes, gat_have, producer), args.min_nodes, anchors)
        cum.append(
            {
                "added": sorted(added),
                "retained_nodes": opt_n,
                "retained_islands": opt_k,
                "delta_nodes": opt_n - base_n,
                "gated_retained_nodes": gat_n,
                "gated_retained_islands": gat_k,
                "gated_delta_nodes": gat_n - base_n,
            }
        )
    print("\ncumulative, greedy in the order above (optimistic | gated):")
    for c in cum:
        print(
            f"  +{','.join(c['added']):<44} "
            f"{c['retained_nodes']:>5}n/{c['retained_islands']:>3}i ({c['delta_nodes']:+})"
            f"  |  {c['gated_retained_nodes']:>5}n/{c['gated_retained_islands']:>3}i "
            f"({c['gated_delta_nodes']:+})"
        )

    report = {
        "model": args.model,
        "claim_log": args.claim_log,
        "min_nodes": args.min_nodes,
        "anchors": sorted(anchors),
        "graph_nodes": len(nodes),
        "claim_log_name_matches": matched,
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
