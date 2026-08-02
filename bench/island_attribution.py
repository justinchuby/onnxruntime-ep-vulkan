"""Island-splitter histogram with island attribution.

Mouse's assignment (2026-07-30):
  1. Histogram declined nodes by (op, code) from CLAIM_LOG.
  2. Attribute cuts: for each declined node, does claiming it hypothetically merge two
     currently-separate islands? Count how many island cuts each decline op type is
     responsible for.
  3. The op responsible for the most cuts is the next work item.

Usage::

    python bench/island_attribution.py

Requires ONNXRUNTIME_EP_VULKAN_CLAIM_LOG to be set, or runs Phi-3.5 to produce it.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent
# Probe output belongs under bench/results/, never the repo root. An earlier version wrote
# island_attribution.json and claim_log_attribution.jsonl into the working directory, which put
# them in front of `git add` and got them committed.
_RESULTS = _HERE / "results"
_RESULTS.mkdir(parents=True, exist_ok=True)

if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

_MODEL_DIR = pathlib.Path(
    r"C:\Users\justinchu\.foundry\cache\models"
    r"\Microsoft\Phi-3.5-mini-instruct-cuda-gpu"
    r"\cuda-int4-rtn-block-32"
)
_ONNX_FILE = _MODEL_DIR / "phi-3.5-mini-instruct-cuda-int4-rtn-block-32.onnx"

EP_LIB = pathlib.Path(
    r"C:\Users\justinchu\dev\ep-vulkan-mouse\rust\target\release\onnxruntime_vulkan_ep.dll"
)


# ---------------------------------------------------------------------------
# Step 1: run ORT with CLAIM_LOG to collect per-node decisions
# ---------------------------------------------------------------------------

def collect_claim_log(log_path: pathlib.Path) -> list[dict]:
    """Run Phi-3.5 under the EP and return parsed CLAIM_LOG records."""
    import onnxruntime as ort

    # Register the EP plugin (process-scoped; safe to call multiple times).
    EP_NAME = "VulkanExecutionProvider"
    try:
        ort.register_execution_provider_library(EP_NAME, str(EP_LIB))
    except Exception as exc:
        if "already registered" not in str(exc):
            raise

    opts = ort.SessionOptions()
    opts.enable_profiling = False

    env_before = os.environ.copy()
    os.environ["ONNXRUNTIME_EP_VULKAN_CLAIM_LOG"] = str(log_path)

    device_id = int(os.environ.get("ONNXRUNTIME_EP_VULKAN_DEVICE", "0"))

    try:
        sess = ort.InferenceSession(
            str(_ONNX_FILE),
            sess_options=opts,
            providers=[EP_NAME, "CPUExecutionProvider"],
            # Device is selected via ONNXRUNTIME_EP_VULKAN_DEVICE env var.
            free_dimension_overrides_by_name={"batch_size": "1", "sequence_length": "1"},
        )
        # One inference — we only need the claim log, not outputs.
        feeds: dict[str, np.ndarray] = {
            "input_ids": np.array([[1]], dtype=np.int64),
            "attention_mask": np.array([[1]], dtype=np.int64),
        }
        empty_kv = np.empty((1, 32, 0, 96), dtype=np.float16)
        for layer in range(32):
            feeds[f"past_key_values.{layer}.key"] = empty_kv
            feeds[f"past_key_values.{layer}.value"] = empty_kv
        sess.run(None, feeds)
    finally:
        os.environ.clear()
        os.environ.update(env_before)

    records: list[dict] = []
    if log_path.exists():
        with open(log_path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return records


# ---------------------------------------------------------------------------
# Step 2: load graph topology from ONNX model
# ---------------------------------------------------------------------------

def load_graph_topology(onnx_path: pathlib.Path):
    """Return (nodes_by_name, value_to_producers, value_to_consumers).

    nodes_by_name: {node_name -> {op, inputs: [value_name], outputs: [value_name]}}
    value_to_producers: {value_name -> node_name}   (one producer per value in ONNX)
    value_to_consumers: {value_name -> [node_name]}
    """
    import onnx

    # Load without external data — we only need the graph structure, not weights.
    model = onnx.load(str(onnx_path), load_external_data=False)
    graph = model.graph

    nodes_by_name: dict[str, dict] = {}
    value_to_producers: dict[str, str] = {}
    value_to_consumers: dict[str, list[str]] = defaultdict(list)

    unnamed_idx = 0
    for node in graph.node:
        name = node.name
        if not name:
            name = f"__unnamed_{unnamed_idx}"
            unnamed_idx += 1

        op = node.op_type
        domain = node.domain
        if domain and domain not in ("", "ai.onnx"):
            op = f"{domain}::{op}"

        inputs = list(node.input)
        outputs = list(node.output)

        nodes_by_name[name] = {"op": op, "inputs": inputs, "outputs": outputs}

        for out in outputs:
            if out:
                value_to_producers[out] = name
        for inp in inputs:
            if inp:
                value_to_consumers[inp].append(name)

    return nodes_by_name, value_to_producers, dict(value_to_consumers)


# ---------------------------------------------------------------------------
# Step 3: island attribution
# ---------------------------------------------------------------------------

def compute_island_attribution(
    claim_records: list[dict],
    nodes_by_name: dict,
    value_to_producers: dict,
    value_to_consumers: dict,
) -> dict:
    """
    Compute:
      1. Decline histogram: Counter of (op, code) -> count.
      2. Island cuts caused by each declined op type.
      3. Number of cuts attributed to each op.

    A declined node D *creates a cut* if there exist two claimed nodes A and B such that:
      - A is a predecessor of D (via A → D edge)
      - B is a successor of D (via D → B edge)
      - A and B are in different islands

    "Island" here means: connected component of claimed nodes in the graph
    (two claimed nodes are in the same island if they are directly adjacent with no
    unclaimed node between them).

    The number of cuts *this instance of D* creates is the number of (island_of_A, island_of_B)
    pairs it bridges. Summed over all instances of op type T, that is T's attribution.
    """
    # Build node→claimed map from claim log.
    # Use node name as key; fall back to the "node" field in the log.
    name_to_record: dict[str, dict] = {}
    for rec in claim_records:
        node_name = rec.get("node", "")
        if node_name:
            name_to_record[node_name] = rec

    claimed_names: set[str] = set()
    for rec in claim_records:
        if rec.get("claimed"):
            node_name = rec.get("node", "")
            if node_name:
                claimed_names.add(node_name)

    # Build island membership: union-find over claimed nodes.
    # Two claimed nodes are in the same island if directly connected (output of one feeds
    # input of other, with no unclaimed gap).
    parent: dict[str, str] = {n: n for n in claimed_names}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # For each edge A→value→B where both A and B are claimed: union them.
    for node_name in claimed_names:
        node_info = nodes_by_name.get(node_name)
        if not node_info:
            continue
        for out_val in node_info["outputs"]:
            if not out_val:
                continue
            for consumer in value_to_consumers.get(out_val, []):
                if consumer in claimed_names:
                    union(node_name, consumer)

    # Count distinct islands.
    island_roots = {find(n) for n in claimed_names}
    island_count = len(island_roots)

    # Decline histogram: (op, code) → count of nodes.
    decline_histogram: Counter = Counter()
    for rec in claim_records:
        if not rec.get("claimed"):
            op = rec.get("op", "unknown")
            code = rec.get("code") or "unknown"
            decline_histogram[(op, code)] += 1

    # Island attribution: for each declined node, count island cuts it bridges.
    # A cut is when a predecessor's island ≠ a successor's island (and both are claimed).
    cut_attribution: Counter = Counter()  # op_type → cuts
    instance_cuts: list[dict] = []        # per-instance detail

    for rec in claim_records:
        if rec.get("claimed"):
            continue
        node_name = rec.get("node", "")
        op = rec.get("op", "unknown")
        node_info = nodes_by_name.get(node_name)
        if not node_info:
            continue

        # Claimed predecessors: nodes that produce inputs to this declined node.
        claimed_predecessors: set[str] = set()
        for inp_val in node_info["inputs"]:
            if not inp_val:
                continue
            producer = value_to_producers.get(inp_val)
            if producer and producer in claimed_names:
                claimed_predecessors.add(producer)

        # Claimed successors: nodes that consume outputs of this declined node.
        claimed_successors: set[str] = set()
        for out_val in node_info["outputs"]:
            if not out_val:
                continue
            for consumer in value_to_consumers.get(out_val, []):
                if consumer in claimed_names:
                    claimed_successors.add(consumer)

        # Count distinct (island_of_predecessor, island_of_successor) pairs that differ.
        # Each such pair is a cut that claiming this node would eliminate.
        cuts = 0
        pred_islands = {find(p) for p in claimed_predecessors}
        succ_islands = {find(s) for s in claimed_successors}
        # A node creates a cut if it stands between two claimed nodes in different islands,
        # OR between a claimed predecessor and claimed successor at all (since a path through
        # it could merge their islands). But to be precise: count the number of unique
        # (pred_island, succ_island) pairs where pred_island ≠ succ_island.
        for pi in pred_islands:
            for si in succ_islands:
                if pi != si:
                    cuts += 1

        # Also count: a claimed predecessor with no claimed successor (or vice versa) creates
        # an island boundary even if no cut across. But the "cut" we care about is merging
        # existing islands — a declined node at the graph edge just extends an island's boundary,
        # not merges two. So we count only pred ≠ succ cases.

        if cuts > 0:
            cut_attribution[op] += cuts
            instance_cuts.append({
                "node": node_name,
                "op": op,
                "code": rec.get("code"),
                "cuts": cuts,
                "pred_islands": len(pred_islands),
                "succ_islands": len(succ_islands),
            })

    return {
        "claimed_count": len(claimed_names),
        "island_count": island_count,
        "decline_histogram": decline_histogram,
        "cut_attribution": cut_attribution,
        "instance_cuts": instance_cuts,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not _ONNX_FILE.exists():
        print(f"ERROR: model not found at {_ONNX_FILE}")
        sys.exit(1)

    log_path = _RESULTS / "claim_log_attribution.jsonl"
    log_path.unlink(missing_ok=True)

    print(f"[island_attribution] Collecting claim log → {log_path}")
    records = collect_claim_log(log_path)
    print(f"[island_attribution] {len(records)} records in CLAIM_LOG")

    print(f"[island_attribution] Loading graph topology from {_ONNX_FILE.name}")
    nodes_by_name, value_to_producers, value_to_consumers = load_graph_topology(_ONNX_FILE)
    print(f"[island_attribution] {len(nodes_by_name)} nodes in graph")

    result = compute_island_attribution(
        records, nodes_by_name, value_to_producers, value_to_consumers
    )

    print(f"\n=== CLAIMED / ISLANDS ===")
    print(f"Claimed nodes : {result['claimed_count']}")
    print(f"Islands       : {result['island_count']}")

    print(f"\n=== DECLINE HISTOGRAM (op, code) → count ===")
    hist = result["decline_histogram"]
    for (op, code), count in sorted(hist.items(), key=lambda x: -x[1]):
        print(f"  {count:4d}  {op:<55s}  [{code}]")

    print(f"\n=== ISLAND CUT ATTRIBUTION (op → cuts) ===")
    attr = result["cut_attribution"]
    total_cuts = sum(attr.values())
    for op, cuts in sorted(attr.items(), key=lambda x: -x[1]):
        print(f"  {cuts:4d} cuts  {op}")
    print(f"  ----")
    print(f"  {total_cuts:4d} total cut-instances")

    print(f"\n=== TOP CUT-CREATING INSTANCES ===")
    top = sorted(result["instance_cuts"], key=lambda x: -x["cuts"])[:20]
    for inst in top:
        print(
            f"  {inst['cuts']} cuts  {inst['op']:<50s}  [{inst['code']}]  "
            f"pred_islands={inst['pred_islands']} succ_islands={inst['succ_islands']}"
            f"  node={inst['node']}"
        )

    # Save JSON
    out = {
        "claimed_count": result["claimed_count"],
        "island_count": result["island_count"],
        "decline_histogram": [
            {"op": op, "code": code, "count": count}
            for (op, code), count in sorted(hist.items(), key=lambda x: -x[1])
        ],
        "cut_attribution": [
            {"op": op, "cuts": cuts}
            for op, cuts in sorted(attr.items(), key=lambda x: -x[1])
        ],
    }
    out_path = _RESULTS / "island_attribution.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[island_attribution] Full result → {out_path}")


if __name__ == "__main__":
    main()
