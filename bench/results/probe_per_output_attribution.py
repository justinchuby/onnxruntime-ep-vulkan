"""Probe: does ORT's profile support *per-output* attribution?

The question the coordinator asked (2026-08-02): attribution is a binary property of the
session while the oracle comparison is per-output.  At ``own_count == 0`` they compose
safely.  As Mouse fills the ledger, a session will claim *some* nodes: attribution says
yes, ``MATCH`` becomes representable, and the outputs whose producing nodes still decline
are still CPU-against-CPU.

This probe asks what the *data* supports, before any mechanism is designed:

  1. What does an ORT ``Node`` event carry?  Names, ``args`` keys, providers.
  2. Do CPU-executed node names match graph node names in the ONNX file?
  3. Is the fused node's name enough to recover which graph nodes it absorbed?
  4. Given (2)/(3), can each *graph output* be labelled by the provider(s) upstream of it?

Writes bench/results/per_output_attribution_probe.json.  No wall-clock assertion; no
threshold of any kind.  Run under both selectors.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np
import onnx_ir as ir
import onnxruntime as ort

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent / "tests" / "ops"))

import _models as m  # noqa: E402

OUT = _HERE / "per_output_attribution_probe.json"


def _mixed_model() -> bytes:
    """Two independent branches to two outputs.

    ``out_claimed``  <- Add, Mul        (proven forms; the EP is expected to claim these)
    ``out_declined`` <- Sin, Cos, Erf   (expected to decline; if the EP claims them the
                                         probe says so rather than assuming)

    Independent branches on purpose: a shared ancestor would make every output downstream
    of every node and the question unanswerable by topology.
    """
    x = m.tensor("x", ir.DataType.FLOAT, [4, 4])
    y = m.tensor("y", ir.DataType.FLOAT, [4, 4])
    a = m.tensor("a", ir.DataType.FLOAT, [4, 4])
    b = m.tensor("out_claimed", ir.DataType.FLOAT, [4, 4])
    s = m.tensor("s", ir.DataType.FLOAT, [4, 4])
    c = m.tensor("c", ir.DataType.FLOAT, [4, 4])
    e = m.tensor("out_declined", ir.DataType.FLOAT, [4, 4])

    nodes = [
        ir.Node("", "Add", inputs=[x, y], outputs=[a], name="claimed_add"),
        ir.Node("", "Mul", inputs=[a, y], outputs=[b], name="claimed_mul"),
        ir.Node("", "Sin", inputs=[x], outputs=[s], name="declined_sin"),
        ir.Node("", "Cos", inputs=[s], outputs=[c], name="declined_cos"),
        ir.Node("", "Erf", inputs=[c], outputs=[e], name="declined_erf"),
    ]
    graph = ir.Graph(
        inputs=[x, y], outputs=[b, e], nodes=nodes, name="mixed", opset_imports={"": 17}
    )
    return ir.to_proto(ir.Model(graph, ir_version=10)).SerializeToString()


def _graph_nodes(model_bytes: bytes) -> dict:
    model = ir.from_proto(__import__("onnx").load_model_from_string(model_bytes))
    g = model.graph
    producer = {}
    node_inputs = {}
    for n in g:
        for o in n.outputs:
            if o is not None and o.name:
                producer[o.name] = n.name
        node_inputs[n.name] = [i.name for i in n.inputs if i is not None]
    return {
        "producer": producer,
        "node_inputs": node_inputs,
        "outputs": [o.name for o in g.outputs],
        "node_names": [n.name for n in g],
        "node_ops": {n.name: n.op_type for n in g},
    }


def _ancestors(topo: dict, output_name: str) -> set[str]:
    """Every graph node upstream of *output_name*, inclusive of its producer."""
    seen: set[str] = set()
    stack = [output_name]
    while stack:
        value = stack.pop()
        node = topo["producer"].get(value)
        if node is None or node in seen:
            continue
        seen.add(node)
        stack.extend(topo["node_inputs"].get(node, ()))
    return seen


def main() -> int:
    device = os.environ.get("ONNXRUNTIME_EP_VULKAN_DEVICE", "0")
    lib = os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB")
    if lib:
        try:
            ort.register_execution_provider_library(m.EP_NAME, str(pathlib.Path(lib).resolve()))
        except Exception as exc:  # noqa: BLE001
            if "already registered" not in str(exc):
                raise
    model = _mixed_model()
    topo = _graph_nodes(model)
    feeds = {
        "x": np.random.default_rng(11).standard_normal((4, 4)).astype(np.float32),
        "y": np.random.default_rng(12).standard_normal((4, 4)).astype(np.float32),
    }

    prefix = _HERE / f"per_output_attr_dev{device}"
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.enable_profiling = True
    opts.profile_file_prefix = str(prefix)
    sess = ort.InferenceSession(model, opts, providers=m.EP_PROVIDERS)
    providers = sess.get_providers()
    sess.run(None, feeds)
    profile_path = sess.end_profiling()

    events = json.loads(pathlib.Path(profile_path).read_text(encoding="utf-8"))
    pathlib.Path(profile_path).unlink(missing_ok=True)
    node_events = [e for e in events if isinstance(e, dict) and e.get("cat") == "Node"]

    by_provider: dict[str, list[str]] = {}
    arg_keys: set[str] = set()
    samples = []
    for ev in node_events:
        args = ev.get("args") or {}
        arg_keys.update(args.keys())
        prov = args.get("provider", "<none>")
        by_provider.setdefault(prov, []).append(ev.get("name", ""))
        if len(samples) < 6:
            samples.append({"name": ev.get("name"), "args": args})

    # (2) do the CPU event names match graph node names?
    def _strip(n: str) -> str:
        for suffix in ("_kernel_time", "_fence_before", "_fence_after"):
            if n.endswith(suffix):
                return n[: -len(suffix)]
        return n

    cpu_named = {_strip(n) for n in by_provider.get("CPUExecutionProvider", [])}
    ep_named = {_strip(n) for n in by_provider.get(m.EP_NAME, [])}
    graph_names = set(topo["node_names"])

    # (3) complement: nodes the graph has that no CPU event names
    unnamed_by_cpu = sorted(graph_names - cpu_named)

    # (4) per-output labelling, by two independent routes
    per_output = {}
    for out_name in topo["outputs"]:
        anc = _ancestors(topo, out_name)
        per_output[out_name] = {
            "ancestor_nodes": sorted(anc),
            "ancestor_ops": sorted({topo["node_ops"][n] for n in anc}),
            "ancestors_named_by_cpu": sorted(anc & cpu_named),
            "ancestors_not_named_by_cpu": sorted(anc - cpu_named),
            "ancestors_named_by_ep_directly": sorted(anc & ep_named),
        }

    record = {
        "device_index": device,
        "providers_in_session": providers,
        "graph": {
            "node_count": len(topo["node_names"]),
            "node_ops": topo["node_ops"],
            "outputs": topo["outputs"],
        },
        "profile": {
            "node_event_count": len(node_events),
            "arg_keys_seen": sorted(arg_keys),
            "events_by_provider": {k: sorted(v) for k, v in by_provider.items()},
            "sample_events": samples,
        },
        "question_2_cpu_names_are_graph_names": sorted(cpu_named & graph_names),
        "question_2_cpu_names_that_are_not_graph_nodes": sorted(cpu_named - graph_names),
        "question_3_ep_event_names": sorted(ep_named),
        "question_3_graph_nodes_no_cpu_event_names": unnamed_by_cpu,
        "question_4_per_output": per_output,
    }
    OUT.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
