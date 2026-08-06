"""Probe: what does ORT's own shape inference know about a model, and what would a
conservative `Shape`/`Cast`/`Concat` chain reader add?

This is the *reproduction* instrument for issue #8. It reads a model with
`onnx.shape_inference`, tabulates how many value_infos carry no rank at all, and then walks
the `Shape`/`Cast`/`Concat`/`Gather`/`Unsqueeze`/`Slice` chains that feed `Reshape` targets to
show which of those targets are recoverable from graph structure alone.

It asserts nothing and claims nothing. It prints what is in the file.

Usage:
    python rust/tools/probe_rank_chain.py <model.onnx> [--json out.json]
"""

from __future__ import annotations

import argparse
import collections
import json
import sys

import onnx
from onnx import TensorProto


def _rank_of(vi) -> int | None:
    t = vi.type.tensor_type
    if not t.HasField("shape"):
        return None
    return len(t.shape.dim)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--json")
    args = ap.parse_args()

    m = onnx.load(args.model)
    try:
        m = onnx.shape_inference.infer_shapes(m, strict_mode=False, data_prop=False)
    except Exception as exc:  # noqa: BLE001 - instrument, report and continue
        print(f"shape_inference failed: {exc}", file=sys.stderr)

    g = m.graph
    known: dict[str, int | None] = {}
    for vi in list(g.input) + list(g.output) + list(g.value_info):
        known[vi.name] = _rank_of(vi)
    inits = {i.name: i for i in g.initializer}
    for name, init in inits.items():
        known[name] = len(init.dims)

    producer = {}
    for n in g.node:
        for o in n.output:
            producer[o] = n

    op_counts = collections.Counter(n.op_type for n in g.node)

    # How many node edges have no rank at all?
    edges_total = 0
    edges_norank = 0
    per_op_norank = collections.Counter()
    for n in g.node:
        for e in list(n.input) + list(n.output):
            if not e:
                continue
            edges_total += 1
            if known.get(e) is None:
                edges_norank += 1
                per_op_norank[n.op_type] += 1

    # Reshape targets: what produces them?
    target_chain = collections.Counter()
    reshape_out_norank = 0
    for n in g.node:
        if n.op_type != "Reshape":
            continue
        if known.get(n.output[0]) is None:
            reshape_out_norank += 1
        t = n.input[1]
        if t in inits:
            target_chain["initializer"] += 1
        elif t in producer:
            target_chain[producer[t].op_type] += 1
        else:
            target_chain["<graph input>"] += 1

    # Walk one representative Reshape-target chain to the leaves.
    def walk(name: str, depth: int = 0, seen=None) -> list[str]:
        seen = seen or set()
        if name in seen or depth > 12:
            return [f"{'  ' * depth}{name} <cycle/depth>"]
        seen.add(name)
        if name in inits:
            init = inits[name]
            vals = list(onnx.numpy_helper.to_array(init).reshape(-1)[:8])
            return [f"{'  ' * depth}{name} = initializer dims={list(init.dims)} {vals}"]
        n = producer.get(name)
        if n is None:
            return [f"{'  ' * depth}{name} = <graph input> rank={known.get(name)}"]
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
        lines = [f"{'  ' * depth}{name} <- {n.op_type}{attrs if attrs else ''}"]
        for i in n.input:
            lines += walk(i, depth + 1, set(seen))
        return lines

    sample = []
    for n in g.node:
        if n.op_type == "Reshape" and n.input[1] not in inits:
            sample = walk(n.input[1])
            sample.insert(0, f"reshape node: {n.name} -> out rank {known.get(n.output[0])}")
            break

    # Chain-op inventory: how many Shape/Cast/Concat nodes, and do their inputs have ranks?
    chain_ops = ("Shape", "Cast", "Concat", "Gather", "Unsqueeze", "Squeeze", "Slice")
    chain_stats = {}
    for op in chain_ops:
        nodes = [n for n in g.node if n.op_type == op]
        with_rank_in = sum(
            1 for n in nodes if all(known.get(i) is not None for i in n.input if i)
        )
        chain_stats[op] = {
            "nodes": len(nodes),
            "all_inputs_ranked": with_rank_in,
            "outputs_ranked": sum(1 for n in nodes if known.get(n.output[0]) is not None),
        }

    # MatMul operand ranks, the headline decline in issue #8.
    mm = [n for n in g.node if n.op_type == "MatMul"]
    mm_a_norank = sum(1 for n in mm if known.get(n.input[0]) is None)
    mm_b_norank = sum(1 for n in mm if known.get(n.input[1]) is None)

    out = {
        "model": args.model,
        "nodes": len(g.node),
        "op_counts": dict(op_counts.most_common()),
        "edges_total": edges_total,
        "edges_without_rank": edges_norank,
        "edges_without_rank_by_op": dict(per_op_norank.most_common(20)),
        "reshape_target_producer": dict(target_chain),
        "reshape_outputs_without_rank": reshape_out_norank,
        "chain_ops": chain_stats,
        "matmul_nodes": len(mm),
        "matmul_A_without_rank": mm_a_norank,
        "matmul_B_without_rank": mm_b_norank,
        "sample_chain": sample,
    }
    print(json.dumps(out, indent=1))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
