"""Island-splitter histogram with island attribution.

Mouse's assignment (2026-07-30): given that 321 nodes are claimed in 33 islands,
attribute cuts to the declined ops that cause them, to determine the next op to implement.

A declined node D creates a cut if there exist claimed nodes A and B such that:
  - A is a predecessor of D (directly adjacent)
  - B is a successor of D (directly adjacent)
  - A and B are in different islands

The op type responsible for the most cuts is the next work item.

Run::

    pytest tests/ops/test_island_attribution.py -v -s --no-header

Results are written to island_attribution.json in the repo root.
"""

from __future__ import annotations

import json
import os
import pathlib
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import pytest

import _models as m

_MODEL_DIR = pathlib.Path(
    r"C:\Users\justinchu\.foundry\cache\models"
    r"\Microsoft\Phi-3.5-mini-instruct-cuda-gpu"
    r"\cuda-int4-rtn-block-32"
)
_ONNX_FILE = _MODEL_DIR / "phi-3.5-mini-instruct-cuda-int4-rtn-block-32.onnx"
_RESULT_FILE = pathlib.Path(__file__).parents[2] / "island_attribution.json"


def _build_phi35_feeds() -> dict[str, np.ndarray]:
    feeds: dict[str, np.ndarray] = {
        "input_ids": np.array([[1]], dtype=np.int64),
        "attention_mask": np.array([[1]], dtype=np.int64),
    }
    empty_kv = np.empty((1, 32, 0, 96), dtype=np.float16)
    for layer in range(32):
        feeds[f"past_key_values.{layer}.key"] = empty_kv
        feeds[f"past_key_values.{layer}.value"] = empty_kv
    return feeds


def _load_graph_topology(onnx_path: pathlib.Path) -> tuple[dict, dict, dict]:
    """Return (nodes_by_name, value_to_producer, value_to_consumers).

    - nodes_by_name: {name -> {"op": str, "inputs": [...], "outputs": [...]}}
    - value_to_producer: {value_name -> node_name}   (single producer in ONNX DAG)
    - value_to_consumers: {value_name -> [node_name]}
    """
    model = onnx.load(str(onnx_path), load_external_data=False)
    graph = model.graph

    nodes_by_name: dict[str, dict] = {}
    value_to_producer: dict[str, str] = {}
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
                value_to_producer[out] = name
        for inp in inputs:
            if inp:
                value_to_consumers[inp].append(name)

    return nodes_by_name, value_to_producer, dict(value_to_consumers)


def _compute_attribution(
    claim_records: list[dict],
    nodes_by_name: dict,
    value_to_producer: dict,
    value_to_consumers: dict,
) -> dict:
    """Compute decline histogram and island cut attribution."""
    # Parse claimed/declined from claim log.
    name_to_record: dict[str, dict] = {}
    for rec in claim_records:
        node_name = rec.get("node", "")
        if node_name:
            name_to_record[node_name] = rec

    claimed_names: set[str] = {
        rec["node"] for rec in claim_records
        if rec.get("claimed") and rec.get("node")
    }

    # Union-Find: compute island membership (connected components of claimed nodes).
    parent: dict[str, str] = {n: n for n in claimed_names}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pa] = pb

    # Union claimed nodes that share a direct edge (no unclaimed node between them).
    for node_name in claimed_names:
        ni = nodes_by_name.get(node_name)
        if not ni:
            continue
        for out_val in ni["outputs"]:
            if not out_val:
                continue
            for consumer in value_to_consumers.get(out_val, []):
                if consumer in claimed_names:
                    union(node_name, consumer)

    island_roots = {find(n) for n in claimed_names}
    island_count = len(island_roots)

    # Decline histogram by (op, code).
    decline_histogram: Counter = Counter()
    for rec in claim_records:
        if not rec.get("claimed"):
            op = rec.get("op", "unknown")
            code = rec.get("code") or "unknown"
            decline_histogram[(op, code)] += 1

    # Island cut attribution: for each declined node, count how many island boundary merges
    # it would enable if claimed. A "cut" = a (pred_island, succ_island) pair with pred ≠ succ.
    cut_attribution: Counter = Counter()
    instance_cuts: list[dict] = []

    for rec in claim_records:
        if rec.get("claimed"):
            continue
        node_name = rec.get("node", "")
        op = rec.get("op", "unknown")
        code = rec.get("code") or "unknown"
        ni = nodes_by_name.get(node_name)
        if not ni:
            continue

        pred_islands: set[str] = set()
        for inp_val in ni["inputs"]:
            if not inp_val:
                continue
            producer = value_to_producer.get(inp_val)
            if producer and producer in claimed_names:
                pred_islands.add(find(producer))

        succ_islands: set[str] = set()
        for out_val in ni["outputs"]:
            if not out_val:
                continue
            for consumer in value_to_consumers.get(out_val, []):
                if consumer in claimed_names:
                    succ_islands.add(find(consumer))

        # Count distinct cross-island (pred, succ) pairs.
        cuts = sum(1 for pi in pred_islands for si in succ_islands if pi != si)
        if cuts > 0:
            cut_attribution[op] += cuts
            instance_cuts.append({
                "node": node_name, "op": op, "code": code, "cuts": cuts,
                "pred_islands": len(pred_islands), "succ_islands": len(succ_islands),
            })

    return {
        "claimed_count": len(claimed_names),
        "island_count": island_count,
        "decline_histogram": decline_histogram,
        "cut_attribution": cut_attribution,
        "instance_cuts": sorted(instance_cuts, key=lambda x: -x["cuts"]),
    }


@pytest.mark.slow
def test_island_attribution(require_vulkan, tmp_path: pathlib.Path) -> None:
    """Produce the island-splitter histogram with cut attribution.

    Falsifier: if the attribution names a different op than GQA as the top cutter,
    we implement that op first (data decides, not the brief's guess).
    """
    if not _ONNX_FILE.exists():
        pytest.skip(f"Phi-3.5 model not found at {_ONNX_FILE}")

    device_index = os.environ.get("ONNXRUNTIME_EP_VULKAN_DEVICE", "0")
    claim_log_path = tmp_path / f"phi35_claim_log_attr_dev{device_index}.jsonl"

    # Set claim log BEFORE session creation.
    old = os.environ.get("ONNXRUNTIME_EP_VULKAN_CLAIM_LOG")
    os.environ["ONNXRUNTIME_EP_VULKAN_CLAIM_LOG"] = str(claim_log_path)

    try:
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.enable_profiling = False
        sess = ort.InferenceSession(
            str(_ONNX_FILE),
            opts,
            providers=m.EP_PROVIDERS,
            free_dimension_overrides_by_name={"batch_size": "1", "sequence_length": "1"},
        )
        feeds = _build_phi35_feeds()
        sess.run(None, feeds)
    finally:
        if old is not None:
            os.environ["ONNXRUNTIME_EP_VULKAN_CLAIM_LOG"] = old
        else:
            os.environ.pop("ONNXRUNTIME_EP_VULKAN_CLAIM_LOG", None)

    # Parse claim log.
    claim_records: list[dict] = []
    if claim_log_path.exists():
        with open(claim_log_path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        claim_records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    assert len(claim_records) > 0, (
        "CLAIM_LOG was empty — the claim log was not visible to the EP. "
        "On Windows, env vars set mid-process may not be visible to the DLL "
        "if the DLL read the path at init time. "
        f"Claim log path: {claim_log_path}"
    )

    print(f"\n[island_attribution] {len(claim_records)} records in CLAIM_LOG")

    # Load graph topology.
    print(f"[island_attribution] Loading graph topology from {_ONNX_FILE.name}")
    nodes_by_name, value_to_producer, value_to_consumers = _load_graph_topology(_ONNX_FILE)
    print(f"[island_attribution] {len(nodes_by_name)} nodes in graph")

    result = _compute_attribution(
        claim_records, nodes_by_name, value_to_producer, value_to_consumers
    )

    # Print results.
    print(f"\n=== CLAIMED / ISLANDS (device {device_index}) ===")
    print(f"Claimed nodes : {result['claimed_count']}")
    print(f"Islands       : {result['island_count']}")

    print(f"\n=== DECLINE HISTOGRAM (op, code) → count ===")
    hist = result["decline_histogram"]
    for (op, code), count in sorted(hist.items(), key=lambda x: -x[1]):
        print(f"  {count:4d}  {op:<55s}  [{code}]")

    print(f"\n=== ISLAND CUT ATTRIBUTION (op type → cuts) ===")
    attr = result["cut_attribution"]
    total_cuts = sum(attr.values())
    for op, cuts in sorted(attr.items(), key=lambda x: -x[1]):
        print(f"  {cuts:4d} cuts  {op}")
    print(f"  ---- total: {total_cuts}")

    print(f"\n=== TOP CUT-CREATING INSTANCES ===")
    for inst in result["instance_cuts"][:15]:
        print(
            f"  {inst['cuts']} cuts  {inst['op']:<50s}  [{inst['code']}]"
            f"  pred={inst['pred_islands']} succ={inst['succ_islands']}"
            f"  node={inst['node']}"
        )

    # Save JSON for external consumption.
    out_data = {
        "device": device_index,
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
        "top_cut_instances": result["instance_cuts"][:20],
    }
    _RESULT_FILE.write_text(json.dumps(out_data, indent=2))
    print(f"\n[island_attribution] Written → {_RESULT_FILE}")

    # Structural assertion: claimed nodes in log > 0.
    assert result["claimed_count"] > 0, (
        "CLAIM_LOG shows 0 claimed nodes — something is wrong with the claim log or the EP."
    )
