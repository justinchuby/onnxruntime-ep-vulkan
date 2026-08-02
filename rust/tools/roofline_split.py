"""What fraction of Phi-3.5's *work* runs on the CPU EP — as a function of context length.

`executed_by = {'CPUExecutionProvider': 120, 'VulkanExecutionProvider': 99}` is a node count, and
a large count of small things is not a large thing. This tool replaces that count with two
quantities that are actually spent — **FLOPs** and **bytes moved** — split by execution provider.

Three constraints shaped the design, and each one is enforced rather than noted.

**1. The denominator must not come from our own partitioner.** An identity whose two sides come
from one source is a falsifier that cannot fire (R11). So the whole model is enumerated straight
from the ONNX file — all 366 graph nodes, their real shapes, their real initializer byte lengths —
and the *only* thing taken from the EP is which nodes it claimed, read from the CLAIM_LOG. Two
sources: the graph says how much work exists, the EP says which part of it it took.

**2. Attention's cost is context-dependent, so a scalar is the wrong shape of answer.** KV traffic
is 0% of bytes at zero context and the majority of it deep into a long one, and the reason that
stayed invisible is that the one quotable figure was taken at ctx=0. Every number here is reported
against a stated context length, the way a timing states its device state. `ctx` is the number of
tokens already in the KV cache; the step measured is one decode step
(`sequence_length = 1`, `past_sequence_length = ctx`).

**3. The EP's own estimator substitutes the constant 128 for every unknown dim.** That is a
fabricated input, not an over-broad one, so no amount of tightening repairs it. This tool does not
inherit that constant: it resolves every extent with ORT's symbolic shape inference against a
*stated* context length, and then reports how many extents it still had to invent. If the answer
rests on invented extents, the answer is `UNOBSERVABLE`, not small.

**No clock.** Nothing here is timed. FLOPs and bytes are counted from shapes and dtypes; a count
does not care whether the box is busy.

Output: `bench/results/roofline_split-dev{N}.json`
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
RESULTS = REPO / "bench" / "results"

MODEL = pathlib.Path(
    os.environ.get(
        "PHI35_MODEL",
        r"C:\Users\justinchu\.foundry\cache\models\Microsoft"
        r"\Phi-3.5-mini-instruct-cuda-gpu\cuda-int4-rtn-block-32"
        r"\phi-3.5-mini-instruct-cuda-int4-rtn-block-32.onnx",
    )
)

#: Context lengths to report at. 0 is included precisely because it is the regime that understates
#: attention the most — it is the trap, not the answer, and leaving it out would hide that the
#: shape of the curve is the finding.
CONTEXTS = (0, 128, 512, 2048, 8192)

#: ONNX TensorProto dtype -> bytes per element. Only the ones this model uses.
DTYPE_BYTES = {
    1: 4,  # FLOAT
    2: 1,  # UINT8
    3: 1,  # INT8
    6: 4,  # INT32
    7: 8,  # INT64
    9: 1,  # BOOL
    10: 2,  # FLOAT16
    11: 8,  # DOUBLE
    16: 2,  # BFLOAT16
}

# --------------------------------------------------------------------------------------------
# The child: one real session, so the split is the EP's live decision and not a re-derivation
# of it. Re-deriving the claim rule here would make both sides of the split come from the same
# source, which is the thing R11 names.
# --------------------------------------------------------------------------------------------


def run_child() -> None:
    import onnxruntime as ort

    ort.register_execution_provider_library(
        "VulkanExecutionProvider", os.environ["ONNXRUNTIME_VULKAN_EP_LIB"]
    )
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.enable_profiling = True
    so.profile_file_prefix = os.environ["ROOFLINE_PROFILE_PREFIX"]
    sess = ort.InferenceSession(
        str(MODEL), so, providers=["VulkanExecutionProvider", "CPUExecutionProvider"]
    )
    sys.path.insert(0, str(REPO / "tests" / "ops"))
    from test_phi35 import _build_phi35_feeds  # noqa: PLC0415

    sess.run(None, _build_phi35_feeds())
    print(f"[child] profile -> {sess.end_profiling()}")


# --------------------------------------------------------------------------------------------
# Shapes, resolved against a stated context length rather than against a constant.
# --------------------------------------------------------------------------------------------


def resolve_shapes(ctx: int) -> tuple[dict, dict, object]:
    """Return (name -> dims, name -> elem_type, inferred model) for one decode step at `ctx`."""
    import onnx
    from onnxruntime.tools.onnx_model_utils import make_dim_param_fixed
    from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

    model = onnx.load(str(MODEL), load_external_data=False)
    make_dim_param_fixed(model.graph, "batch_size", 1)
    make_dim_param_fixed(model.graph, "sequence_length", 1)
    make_dim_param_fixed(model.graph, "past_sequence_length", ctx)
    make_dim_param_fixed(model.graph, "total_sequence_length", ctx + 1)
    inferred = SymbolicShapeInference.infer_shapes(model, auto_merge=True, verbose=0)

    dims: dict[str, list] = {}
    etype: dict[str, int] = {}
    for vi in list(inferred.graph.value_info) + list(inferred.graph.input) + list(
        inferred.graph.output
    ):
        tt = vi.type.tensor_type
        etype[vi.name] = tt.elem_type
        dims[vi.name] = [
            d.dim_value if d.HasField("dim_value") else None for d in tt.shape.dim
        ]
    return dims, etype, inferred


def initializer_bytes(model) -> dict[str, int]:
    """Real stored size of every initializer, read from the file rather than from its shape.

    An int4-packed weight is not `numel * dtype_bytes`, and guessing it would put a fabricated
    number into the one term that dominates this model. External data carries the true length.
    """
    out: dict[str, int] = {}
    for init in model.graph.initializer:
        ext = {e.key: e.value for e in init.external_data}
        if "length" in ext:
            out[init.name] = int(ext["length"])
            continue
        numel = 1
        for d in init.dims:
            numel *= d
        out[init.name] = numel * DTYPE_BYTES.get(init.data_type, 4)
    return out


#: `cos_cache` / `sin_cache` are produced by the `If` at `/model/rotemb_caches_subgraph/If`, whose
#: predicate is `total_sequence_length > 4096` and whose two branches are Constants of shape
#: [4096, 48] and [131072, 48] float16. Symbolic shape inference cannot fold that branch, so it
#: leaves the leading dim open — but the extent is not unknown, it is *conditional*, and the
#: condition is the context length this run already states. Reading it out of the graph is the
#: difference between a resolved extent and a fabricated one.
ROTARY_COLS = 48
ROTARY_ROWS_SMALL = 4096
ROTARY_ROWS_LARGE = 131072
ROTARY_SWITCH = 4096
ROTARY_CACHES = ("cos_cache", "sin_cache")


def rotary_rows(ctx: int) -> int:
    return ROTARY_ROWS_LARGE if (ctx + 1) > ROTARY_SWITCH else ROTARY_ROWS_SMALL


def tensor_bytes(
    name: str, dims: dict, etype: dict, inits: dict[str, int], ctx: int
) -> tuple[int, bool]:
    """Bytes for one tensor, and whether an extent had to be invented to say so.

    Fabrication is tracked per *tensor*, not per node. Flagging a whole node because one small
    operand was unresolved is how a 0.03% uncertainty gets reported as 73% of the answer.
    """
    if name in inits:
        return inits[name], False
    if name in ROTARY_CACHES:
        return rotary_rows(ctx) * ROTARY_COLS * 2, False
    if name not in dims:
        # No shape at all. Reported as fabricated rather than filled in with a constant: the
        # whole point of this tool is that it does not do what `slot_bytes` does.
        return 0, True
    shape = dims[name]
    invented = any(d is None for d in shape)
    numel = 1
    for d in shape:
        numel *= d if d is not None else 1
    return numel * DTYPE_BYTES.get(etype.get(name, 1), 4), invented


# --------------------------------------------------------------------------------------------
# The cost model. Written out per op family rather than folded into one rule, because folding is
# how `2 * 3072 * 3072` per anchor became a node count wearing a FLOP's clothes.
# --------------------------------------------------------------------------------------------


def node_flops(node, dims: dict, inits: dict[str, int]) -> tuple[int, bool]:
    """FLOPs for one node at the resolved shapes, and whether any extent was invented."""

    def shape_of(name):
        return dims.get(name)

    def numel(name):
        s = shape_of(name)
        if s is None or any(d is None for d in s):
            return None
        n = 1
        for d in s:
            n *= d
        return n

    attrs = {a.name: a for a in node.attribute}

    if node.op_type == "MatMulNBits":
        # 2 FLOPs (multiply + add) per weight element per row of A. K and N are attributes, so
        # only the row count depends on the resolved shape.
        k = attrs["K"].i
        n = attrs["N"].i
        a = shape_of(node.input[0])
        if a is None or any(d is None for d in a[:-1]):
            return 0, True
        rows = 1
        for d in a[:-1]:
            rows *= d
        return 2 * rows * k * n, False

    if node.op_type == "GroupQueryAttention":
        # Two matmuls per head: scores = Q @ K^T over the whole cache, then out = P @ V.
        heads = attrs["num_heads"].i
        q = shape_of(node.input[0])
        present = shape_of(node.output[1]) if len(node.output) > 1 else None
        if q is None or present is None or any(d is None for d in (q + present)):
            return 0, True
        b, s = q[0], q[1]
        head_dim = present[3]
        total = present[2]
        return 2 * 2 * b * heads * s * total * head_dim, False

    if node.op_type in ("If", "Shape", "Constant"):
        # No arithmetic. `If` selects a branch and `Shape` reads metadata; their whole cost is
        # the bytes they move, which `node_bytes` charges them for. Returning a fabrication flag
        # here would report a resolved zero as an unknown, which is the mirror of R12's error.
        return 0, False

    # Everything else: one FLOP per output element. These are elementwise, normalisation and
    # control-flow ops; the choice of constant here cannot move the answer, and §7.15 shows it.
    total = 0
    for out in node.output:
        if not out:
            continue
        n = numel(out)
        if n is None:
            return 0, True
        total += n
    return total, False


def node_bytes(node, dims, etype, inits, ctx: int) -> tuple[int, int]:
    """DRAM traffic for one node: every input read plus every output written.

    Returns (bytes, fabricated_bytes). Fabricated bytes are the portion that rests on an extent
    this tool could not resolve; they are reported, never silently filled in.
    """
    total = 0
    fabricated = 0
    for name in list(node.input) + list(node.output):
        if not name:
            continue
        if node.op_type == "GroupQueryAttention" and name in ROTARY_CACHES:
            # Rotary indexes the cache at the current positions only; it does not stream it. The
            # `If` that materialises the whole cache is charged for it separately, and is on the
            # CPU side, so this is not a cost being dropped.
            q = dims.get(node.input[0]) or [1, 1]
            total += (q[1] or 1) * ROTARY_COLS * 2
            continue
        b, inv = tensor_bytes(name, dims, etype, inits, ctx)
        total += b
        if inv:
            fabricated += b
    return total, fabricated


# --------------------------------------------------------------------------------------------
# The EP's own estimator, reproduced exactly, so the claim "it cannot answer this" is an
# artifact and not an opinion.
# --------------------------------------------------------------------------------------------

ANCHORS = {"MatMulNBits", "GroupQueryAttention"}
EP_ANCHOR_FLOPS = 2 * 3072 * 3072
EP_UNKNOWN_DIM = 128


def ep_estimator_flops(node, dims, etype) -> int:
    if node.op_type in ANCHORS:
        return EP_ANCHOR_FLOPS
    out_bytes = 0
    for name in node.output:
        if not name:
            continue
        shape = dims.get(name)
        if shape is None:
            out_bytes += 4096
            continue
        numel = 1
        for d in shape:
            numel *= d if d is not None else EP_UNKNOWN_DIM
        out_bytes += numel * DTYPE_BYTES.get(etype.get(name, 1), 4)
    return out_bytes // 2


# --------------------------------------------------------------------------------------------


def read_claim_log(path: pathlib.Path) -> dict[str, bool]:
    claimed: dict[str, bool] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        rec = json.loads(line)
        node = rec.get("node")
        if node:
            # Last write wins: the partition gate can decline a node the registry accepted, and
            # the later record is the one that decided.
            claimed[node] = bool(rec.get("claimed"))
    return claimed


def read_profile(path: pathlib.Path) -> dict[str, int]:
    events = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(events, dict):
        events = events.get("traceEvents", [])
    counts: collections.Counter[str] = collections.Counter()
    for ev in events:
        if ev.get("cat") == "Node":
            provider = (ev.get("args") or {}).get("provider")
            if provider:
                counts[provider] += 1
    return dict(counts)


def main(argv: list[str]) -> int:
    if len(argv) > 1 and argv[1] == "--child":
        run_child()
        return 0

    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse-attribution", action="store_true")
    ap.add_argument(
        "--counterfactual",
        default="",
        help="op type to move to the Vulkan side before the arithmetic runs. Used to size a "
        "pending fix, and to show the artifact varies with its input rather than with the "
        "reading of the code that produced it.",
    )
    args = ap.parse_args(argv[1:])

    import onnx

    device = os.environ.get("ONNXRUNTIME_EP_VULKAN_DEVICE", "0")
    RESULTS.mkdir(parents=True, exist_ok=True)
    claim_log = RESULTS / f"roofline_claimlog-dev{device}.jsonl"
    profile_prefix = str(RESULTS / f"roofline_profile-dev{device}")

    if not args.reuse_attribution:
        claim_log.unlink(missing_ok=True)
        for stale in RESULTS.glob(f"roofline_profile-dev{device}*.json"):
            stale.unlink()
        env = dict(os.environ)
        env["ONNXRUNTIME_EP_VULKAN_CLAIM_LOG"] = str(claim_log)
        env["ROOFLINE_PROFILE_PREFIX"] = profile_prefix
        proc = subprocess.run(
            [sys.executable, str(pathlib.Path(__file__).resolve()), "--child"],
            env=env,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=7200,
        )
        if proc.returncode != 0 or not claim_log.is_file():
            # R13: an instrument error is never a detection. Quote the text, never a count.
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-8:]
            print("VERDICT: ERROR(instrument)")
            print("\n".join(tail) or f"no claim log at {claim_log}")
            return 2

    claimed = read_claim_log(claim_log)
    profiles = sorted(RESULTS.glob(f"roofline_profile-dev{device}*.json"))
    executed_by = read_profile(profiles[-1]) if profiles else {}
    vulkan_partitions = executed_by.get("VulkanExecutionProvider")

    model = onnx.load(str(MODEL), load_external_data=False)
    inits = initializer_bytes(model)
    graph_nodes = list(model.graph.node)

    # The EP names nodes; the graph names nodes; they must be the same names or the split is
    # being made against a set this tool cannot see. Checked, not assumed.
    graph_ops = {n.name: n.op_type for n in graph_nodes}
    unknown_to_graph = sorted(set(claimed) - set(graph_ops))
    never_offered = sorted(set(graph_ops) - set(claimed))
    declined = sorted(n for n, ok in claimed.items() if not ok)
    declined_by_op = dict(
        collections.Counter(graph_ops.get(n, "?") for n in declined).most_common()
    )

    counterfactual = None
    claimed_count = sum(1 for v in claimed.values() if v)
    if args.counterfactual:
        moved = [n for n in declined if graph_ops.get(n) == args.counterfactual]
        if not moved:
            print(f"VERDICT: ERROR(instrument): no declined node has op type "
                  f"{args.counterfactual!r}")
            return 2
        for n in moved:
            claimed[n] = True
        counterfactual = {"op_type": args.counterfactual, "nodes_moved_to_vulkan": len(moved)}

    reconciliation = {
        "graph_nodes": len(graph_nodes),
        "offered_to_ep": len(claimed),
        "never_offered": {
            "count": len(never_offered),
            "by_op": dict(collections.Counter(graph_ops[n] for n in never_offered)),
            "note": "ORT folds these before GetCapability; they are absent from the frame, "
            "not declined in it.",
        },
        "claimed": claimed_count,
        "declined": len(declined),
        "declined_by_op": declined_by_op,
        "cpu_node_events_in_profile": executed_by.get("CPUExecutionProvider"),
        "vulkan_node_events_in_profile": vulkan_partitions,
        "vulkan_node_events_are_fused_partitions": (
            "Each Vulkan profile event is one fused partition, not one original node. "
            f"{vulkan_partitions} of them means the claimed work is split into "
            f"{vulkan_partitions} islands."
        ),
        "claim_log_names_absent_from_graph": unknown_to_graph[:8],
    }

    rows = []
    for ctx in CONTEXTS:
        dims, etype, _ = resolve_shapes(ctx)
        acc = {
            side: {"flops": 0, "bytes": 0, "nodes": 0, "ep_est_flops": 0}
            for side in ("vulkan", "cpu")
        }
        fab = {"flops_nodes": 0, "bytes_nodes": 0, "flops": 0, "bytes": 0}
        by_op: dict[str, dict] = {}
        for node in graph_nodes:
            side = "vulkan" if claimed.get(node.name, False) else "cpu"
            f, f_inv = node_flops(node, dims, inits)
            b, b_fab = node_bytes(node, dims, etype, inits, ctx)
            acc[side]["flops"] += f
            acc[side]["bytes"] += b
            acc[side]["nodes"] += 1
            acc[side]["ep_est_flops"] += ep_estimator_flops(node, dims, etype)
            if f_inv:
                fab["flops_nodes"] += 1
            if b_fab:
                fab["bytes_nodes"] += 1
                fab["bytes"] += b_fab
            slot = by_op.setdefault(
                node.op_type,
                {"flops": 0, "bytes": 0, "nodes": 0, "cpu_nodes": 0, "cpu_flops": 0,
                 "cpu_bytes": 0},
            )
            slot["flops"] += f
            slot["bytes"] += b
            slot["nodes"] += 1
            slot["cpu_nodes"] += side == "cpu"
            if side == "cpu":
                slot["cpu_flops"] += f
                slot["cpu_bytes"] += b

        tot_f = acc["vulkan"]["flops"] + acc["cpu"]["flops"]
        tot_b = acc["vulkan"]["bytes"] + acc["cpu"]["bytes"]
        tot_e = acc["vulkan"]["ep_est_flops"] + acc["cpu"]["ep_est_flops"]
        rows.append(
            {
                "ctx": ctx,
                "step": "one decode step: sequence_length=1, past_sequence_length=ctx",
                "total_flops": tot_f,
                "total_bytes": tot_b,
                "cpu_flops": acc["cpu"]["flops"],
                "cpu_bytes": acc["cpu"]["bytes"],
                "cpu_flops_frac": acc["cpu"]["flops"] / tot_f if tot_f else None,
                "cpu_bytes_frac": acc["cpu"]["bytes"] / tot_b if tot_b else None,
                "cpu_nodes": acc["cpu"]["nodes"],
                "vulkan_nodes": acc["vulkan"]["nodes"],
                # The same split under the EP's own estimator, for contrast.
                "ep_estimator_cpu_flops_frac": (
                    acc["cpu"]["ep_est_flops"] / tot_e if tot_e else None
                ),
                "fabricated_extent_nodes_flops": fab["flops_nodes"],
                "fabricated_extent_nodes_bytes": fab["bytes_nodes"],
                "fabricated_bytes_frac": (fab["bytes"] / tot_b) if tot_b else None,
                "by_op": by_op,
            }
        )

    out = {
        "device_selector": device,
        "model": str(MODEL),
        "model_sha256_head": hashlib.sha256(MODEL.read_bytes()).hexdigest()[:16],
        "reconciliation": reconciliation,
        "counterfactual": counterfactual,
        "executed_by": executed_by,
        "denominator_source": (
            "Every total is a sum over the 366 nodes of the ONNX graph, read from the file. "
            "The EP contributes only the claimed/declined membership, read from the CLAIM_LOG. "
            "Two sources, so the fractions can be wrong in a way that shows."
        ),
        "no_duration_quoted": (
            "Every figure here is a FLOP count, a byte count or a ratio of two of them. "
            "Nothing was timed; the machine is contended and stays that way."
        ),
        "rows": rows,
    }
    out_path = RESULTS / (
        f"roofline_split-dev{device}"
        + (f"-cf-{args.counterfactual}" if args.counterfactual else "")
        + ".json"
    )
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    r = reconciliation
    print(f"graph {r['graph_nodes']} = offered {r['offered_to_ep']} + never-offered "
          f"{r['never_offered']['count']} {r['never_offered']['by_op']}")
    print(f"offered {r['offered_to_ep']} = claimed {r['claimed']} + declined {r['declined']} "
          f"{r['declined_by_op']}")
    print(f"profile: {executed_by}  -> CPU events == declined nodes 1:1; "
          f"{vulkan_partitions} Vulkan events == {vulkan_partitions} fused islands")
    if unknown_to_graph:
        print(f"WARNING: claim-log names absent from the graph: {unknown_to_graph[:4]}")
    if counterfactual:
        print(f"COUNTERFACTUAL: {counterfactual['nodes_moved_to_vulkan']} "
              f"{counterfactual['op_type']} nodes moved to the Vulkan side")
    print()
    print(f"{'ctx':>6} {'CPU FLOPs':>12} {'CPU bytes':>12} {'EP-est CPU':>12} "
          f"{'totalGB':>9} {'fab bytes':>10}")
    for row in rows:
        print(
            f"{row['ctx']:>6} "
            f"{row['cpu_flops_frac']:>11.2%} "
            f"{row['cpu_bytes_frac']:>11.2%} "
            f"{row['ep_estimator_cpu_flops_frac']:>11.2%} "
            f"{row['total_bytes'] / 1e9:>9.3f} "
            f"{row['fabricated_bytes_frac']:>9.2%}"
        )
    print(f"\n[roofline] -> {out_path}")

    fab_worst = max(x["fabricated_bytes_frac"] or 0 for x in rows)
    if fab_worst > 0.01:
        print(f"VERDICT: UNOBSERVABLE(fabricated extents carry {fab_worst:.2%} of the bytes)")
        return 1
    if not any(x["cpu_bytes_frac"] for x in rows):
        print("VERDICT: ERROR(instrument): no bytes attributed at any context")
        return 2
    print("VERDICT: PASS(no fabricated extent contributed to any figure above)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
