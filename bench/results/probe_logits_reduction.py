"""Criterion 10, round 36: is the logits' 12-ULP step chain depth, or the lm_head kernel?

WHAT ROUND 35 LEFT
==================
Per-output ULP residuals over all 65 outputs of Phi-3.5: the 32 KV layers are flat at
1 ULP layer-to-layer, topping out at 4 ULP for layer 31's key and value. The step is
**output 0, the logits, at 12 ULP median**, and it is not a layer. The three failing
outputs are exactly `[0, 63, 64]` -- logits, and layer 31's key and value.

TWO HYPOTHESES, AND THE ONE OBSERVATION THAT SEPARATES THEM
===========================================================
`H_depth`  The logits are the end of the chain. Their residual is *inherited* from the
           final hidden state, which is the deepest activation in the graph, and the
           lm_head projection merely carries it through. 12 ULP is then what a correct
           implementation looks like at that depth and the criterion needs a relative
           or ULP bound rather than an absolute one.

`H_proj`   The lm_head's own reduction -- K=3072 into N=32064, by far the longest single
           reduction in the graph -- differs from ORT's in a way the other 160
           MatMulNBits nodes do not. The residual is then manufactured *at* that node.

**Named before either arm runs, because a prediction written afterwards is a description:**

  Arm B runs the lm_head node ALONE, fed the *identical* fp16 hidden state on both EPs,
  so nothing upstream can contribute.

    * `H_proj` predicts arm B's median ULP is ~12 -- the full-model figure, reproduced
      with the chain removed.
    * `H_depth` predicts arm B's median ULP is small (0-2), the same as any other
      MatMulNBits in this graph, and that arm A shows the final hidden state already
      carrying a residual that explains the 12.

  These cannot both hold. If arm B lands between (say 3-8 ULP) neither is confirmed and
  the record says so rather than rounding to the nearer story.

THE OBSERVATION THE CRITERION HAS NEVER MADE
============================================
"DIVERGENT" has only ever meant *the CPU EP and we disagree*. It has never once asked
**which of the two is further from the true value.** Arm C computes the same lm_head
projection in float64 from the model's own int4 weights and asks exactly that.

  * If |vk - f64| ~ |cpu - f64|, both are inside the envelope of correct-but-differently-
    ordered implementations and no kernel is at fault.
  * If |vk - f64| >> |cpu - f64|, the Vulkan EP is the one that is wrong, and the shape
    is a located defect.
  * If |cpu - f64| >> |vk - f64|, the *oracle* is the further one -- and every criterion
    in this project that reads "the CPU EP is golden" inherits that.

Arm C carries its own liveness check: the float64 reference must reproduce the CPU EP's
logits to within a few ULP. A dequantisation I got wrong would otherwise disagree with
both EPs and be reported as "both are wrong", which is the most confident way to be
useless. If the self-check fails the arm reports ERROR(instrument) and no verdict.

ARM D -- WHAT A CORRECT IMPLEMENTATION LOOKS LIKE
=================================================
Five *legitimate* accumulation orders for the same reduction (sequential, reversed,
per-32-block, pairwise, and 32-lane strided -- the last being the shape a subgroup
reduction actually takes), each with an fp32 accumulator as both implementations use,
each rounded to fp16. The spread AMONG THEM is the envelope: it is how far two correct
kernels may sit apart on this tensor. A measured residual inside that envelope is not
evidence of a defect no matter how large the number is.

WHAT WOULD FALSIFY THE CONCLUSION, AND WHETHER IT IS REACHABLE
==============================================================
See `falsifier` in the emitted record.

Route is read off the counters the run emitted, never off the env var that requested it
(round 35: Step 1c unbinds on refusal, so a declined bind would otherwise be recorded as
a route that was taken). Device name is read off the run, never off the selector
(round 31: selector 0 ran on the NVIDIA part and selector 1 on the Intel one).

`atol` is not moved here and no verdict row is closed. If the conclusion is that a
relative or ULP bound is correct, that is Morpheus's ruling.

Run:
    python bench/results/probe_logits_reduction.py --device 0 --out bench/results/logits_reduction-dev0.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests" / "ops"))

MODEL_DIR = Path(
    r"C:\Users\justinchu\.foundry\cache\models\Microsoft"
    r"\Phi-3.5-mini-instruct-cuda-gpu\cuda-int4-rtn-block-32"
)
MODEL_FILE = MODEL_DIR / "phi-3.5-mini-instruct-cuda-int4-rtn-block-32.onnx"

#: The node under suspicion and the activation that feeds it.
LM_HEAD_NODE = "/lm_head/MatMul_Q4"
HIDDEN_TENSOR = "/model/layers.32/final_norm_layernorm/output_0"

#: The residual stream after each of the 32 blocks -- `output_3` of the block's
#: SkipSimplifiedLayerNormalization is the skip sum, i.e. the stream itself, not the
#: normalised copy the MLP consumes. This is the chain the logits sit at the end of, and
#: it is a *different* tensor from the 64 KV outputs the criterion already plots.
RESIDUAL_STREAM = [f"/model/layers.{i}/post_attention_layernorm/output_3" for i in range(32)]

#: The final RMSNorm and its two activation inputs, tapped so the hop can be measured
#: against a float64 reference the same way the lm_head is.
FINAL_NORM_NODE = "/model/layers.32/final_norm_layernorm/SkipLayerNorm"
FINAL_NORM_SKIP = "/model/layers.31/mlp/down_proj/MatMul/output_0"
FINAL_NORM_GAMMA = "model.layers.32.final_norm_layernorm.weight"

PREDICTIONS = {
    "registered_before_measuring": True,
    "H_depth": (
        "the lm_head node run ALONE on an identical fp16 input shows a median ULP "
        "residual of 0-2 -- like every other MatMulNBits in this graph -- and the 12 ULP "
        "seen in the full model is inherited from the final hidden state"
    ),
    "H_proj": (
        "the lm_head node run ALONE on an identical fp16 input reproduces ~12 median ULP "
        "with the whole chain removed, locating the residual at that kernel's reduction"
    ),
    "discriminating_observation": (
        "arm_b_isolated_lm_head.median_ulp_diff -- the two hypotheses predict "
        "disjoint values for this single number (0-2 vs ~12)"
    ),
    "neither_if": "arm B lands in 3-8 ULP; the record says INCONCLUSIVE rather than rounding",
}


# ---------------------------------------------------------------------------------------
# MatMulNBits float64 reference
# ---------------------------------------------------------------------------------------
def dequantize_nbits(
    packed: np.ndarray, scales: np.ndarray, *, n: int, k: int, block_size: int, bits: int
) -> np.ndarray:
    """Dequantise a MatMulNBits weight to float64.

    Layout per the ONNX Runtime contrib op: `B` is `[N, n_blocks_per_col, blob_size]`
    uint8 with `blob_size = block_size * bits / 8`, low nibble first. With no zero-point
    input the quantisation is symmetric about the midpoint, which for 4 bits is 8.
    """
    if bits != 4:
        raise NotImplementedError(f"reference covers bits=4 only, got {bits}")
    n_blocks = (k + block_size - 1) // block_size
    blob = block_size * bits // 8
    b = packed.reshape(n, n_blocks, blob)
    lo = (b & 0x0F).astype(np.int16)
    hi = (b >> 4).astype(np.int16)
    # Interleave: element 2i is the low nibble of byte i, 2i+1 the high nibble.
    nib = np.empty((n, n_blocks, blob * 2), dtype=np.int16)
    nib[:, :, 0::2] = lo
    nib[:, :, 1::2] = hi
    s = scales.reshape(n, n_blocks).astype(np.float64)
    deq = (nib.astype(np.float64) - 8.0) * s[:, :, None]
    return deq.reshape(n, n_blocks * block_size)[:, :k]


def reduce_orders(a32: np.ndarray, w32: np.ndarray, block_size: int = 32) -> "dict[str, np.ndarray]":
    """The same reduction, five legitimate ways, each with an fp32 accumulator.

    `a32` is `[K]`, `w32` is `[N, K]`, both float32. Returns float32 results `[N]`.
    Every one of these is a *correct* implementation; they differ only in the order the
    partial sums are combined, which is the freedom a GPU kernel actually exercises.
    """
    k = a32.shape[0]
    prod = None  # built per-order to avoid keeping five [N, K] arrays alive at once
    out = {}

    # Sequential k = 0..K-1, the textbook loop.
    acc = np.zeros(w32.shape[0], dtype=np.float32)
    for i in range(k):
        acc += w32[:, i] * a32[i]
    out["sequential"] = acc.copy()

    # Reversed: the same terms, opposite order.
    acc = np.zeros(w32.shape[0], dtype=np.float32)
    for i in range(k - 1, -1, -1):
        acc += w32[:, i] * a32[i]
    out["reversed"] = acc.copy()

    # Per-32-block partials then a sum over blocks -- the shape a block-quantised kernel
    # naturally takes, because the scale is per block.
    prod = (w32 * a32[None, :]).astype(np.float32)
    nb = k // block_size
    blocks = prod[:, : nb * block_size].reshape(w32.shape[0], nb, block_size)
    part = blocks.sum(axis=2, dtype=np.float32)
    acc = part.sum(axis=1, dtype=np.float32)
    if k % block_size:
        acc = acc + prod[:, nb * block_size :].sum(axis=1, dtype=np.float32)
    out["per_block"] = acc

    # Pairwise / tree, which is what a vectorised host implementation does.
    out["pairwise"] = np.einsum("nk,k->n", w32, a32, dtype=np.float32).astype(np.float32)

    # 32 lanes, strided -- the shape a subgroup reduction takes: lane j accumulates
    # k = j, j+32, j+64, ... and the 32 lane totals are then reduced.
    lanes = 32
    pad = (-k) % lanes
    padded = np.pad(prod, ((0, 0), (0, pad)))
    lane_sums = padded.reshape(w32.shape[0], -1, lanes).sum(axis=1, dtype=np.float32)
    out["strided_lanes"] = lane_sums.sum(axis=1, dtype=np.float32)
    return out


def ulp_stats(vk: np.ndarray, ref: np.ndarray, dtype=np.float16) -> dict:
    """Median / p99 / max ULP of `vk` against `ref`, in ULPs of `ref`'s own spacing."""
    import _models as m

    ulps, basis = m.ulp_residual(vk.astype(dtype), ref.astype(dtype))
    flat = ulps.reshape(-1)
    return {
        "median_ulp_diff": float(np.median(flat)),
        "p99_ulp_diff": float(np.percentile(flat, 99)),
        "max_ulp_diff": float(flat.max()),
        "max_abs_diff": float(np.abs(vk.astype(np.float64) - ref.astype(np.float64)).max()),
        "elements": int(flat.size),
        "ulp_basis": basis,
    }


# ---------------------------------------------------------------------------------------
# Model surgery
# ---------------------------------------------------------------------------------------
def _materialise(t) -> None:
    """Pull one initialiser's bytes out of the 2.3 GB external blob, in place.

    `data_location` must be cleared afterwards or `numpy_helper.to_array` re-resolves the
    relative path against the *current working directory* and reports the blob as "not a
    regular file" -- which is what it did the first time this probe grew an arm.
    """
    import onnx
    from onnx.external_data_helper import load_external_data_for_tensor

    if t.HasField("data_location") and t.data_location == onnx.TensorProto.EXTERNAL:
        load_external_data_for_tensor(t, str(MODEL_DIR))
        t.ClearField("data_location")
        del t.external_data[:]


def make_tapped_model(scratch: Path, extra: "list[str] | None" = None) -> Path:
    """A copy of the real model with `extra` tensors added as graph outputs.

    Written **into the model directory** under a process-unique name and removed again by
    the caller. ONNX resolves an external-data reference relative to the model file, and
    the alternatives are both worse: copying the 2.3 GB blob is antisocial with six agents
    on this box, and hard-linking it makes `onnx.load` refuse the file outright
    (`multiple hard links, indicating a potential hardlink attack`) -- measured, not
    assumed, on the first run of this probe.
    """
    import onnx

    scratch.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(MODEL_FILE), load_external_data=False)
    names = {vi.name for vi in model.graph.output}
    for nm in [HIDDEN_TENSOR] + list(extra or []):
        if nm in names:
            continue
        vi = onnx.ValueInfoProto()
        vi.name = nm
        model.graph.output.append(vi)
        names.add(nm)
    out = MODEL_DIR / f"phi35_tapped_{os.getpid()}.onnx"
    onnx.save(model, str(out))
    return out


def make_isolated_lm_head(scratch: Path) -> "tuple[Path, dict]":
    """A standalone model holding the lm_head MatMulNBits node and nothing else.

    Only the lm_head's own two initialisers are pulled out of the 2.3 GB external blob;
    the rest of the graph is dropped before any external data is touched.
    """
    import onnx
    from onnx import numpy_helper
    from onnx.external_data_helper import load_external_data_for_tensor

    scratch.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(MODEL_FILE), load_external_data=False)
    node = next(n for n in model.graph.node if n.name == LM_HEAD_NODE)
    attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
    inits = {i.name: i for i in model.graph.initializer}
    keep = []
    for nm in node.input[1:]:
        t = inits[nm]
        _materialise(t)
        keep.append(t)

    hidden = onnx.helper.make_tensor_value_info(
        node.input[0], onnx.TensorProto.FLOAT16, [1, 1, int(attrs["K"])]
    )
    logits = onnx.helper.make_tensor_value_info(
        node.output[0], onnx.TensorProto.FLOAT16, [1, 1, int(attrs["N"])]
    )
    g = onnx.helper.make_graph([node], "lm_head_only", [hidden], [logits], initializer=keep)
    sub = onnx.helper.make_model(g, opset_imports=list(model.opset_import))
    sub.ir_version = model.ir_version
    out = scratch / "lm_head_only.onnx"
    onnx.save(sub, str(out))

    weights = {
        "packed": numpy_helper.to_array(keep[0]),
        "scales": numpy_helper.to_array(keep[1]),
        "attrs": {k: int(v) for k, v in attrs.items() if isinstance(v, int)},
        "input_name": node.input[0],
        "output_name": node.output[0],
    }
    return out, weights


# ---------------------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------------------
#: Counters are PROCESS-cumulative, not per-session: arm B's file read `claimed_nodes: 356`
#: where arm A read 355, and arm B claimed exactly one node. Reading the absolute number as
#: this arm's attribution would have credited the lm_head run with the whole model.
_COUNTER_HISTORY: "list[dict]" = []


def route_and_device(counters_path: "Path | None") -> dict:
    """Route, device name and attribution read off what the run emitted, never off env vars.

    Attribution matters more here than anywhere else in this probe: if the isolated
    single-node graph is NOT claimed by the EP, its "Vulkan" run is a CPU run and a 0-ULP
    residual is CPU-vs-CPU -- the most convincing possible way to be measuring nothing.
    """
    fact = {
        "kv_writeback_route": "UNOBSERVABLE",
        "device_name": "UNOBSERVABLE",
        "device_name_source": "counters alloc_device_frame_session_devices",
        "outputs_device_bound": "UNOBSERVABLE",
        "outputs_host_resident": "UNOBSERVABLE",
        "claimed_nodes_this_arm": "UNOBSERVABLE",
        "attribution": "UNATTRIBUTED",
    }
    if not counters_path or not Path(counters_path).exists():
        return fact
    try:
        doc = json.loads(Path(counters_path).read_text(encoding="utf-8-sig"))
    except Exception as exc:  # noqa: BLE001
        fact["counters_read_error"] = str(exc)
        return fact
    bound = doc.get("outputs_device_bound")
    host = doc.get("outputs_host_resident")
    fact["outputs_device_bound"] = bound
    fact["outputs_host_resident"] = host
    if isinstance(bound, int) and isinstance(host, int):
        fact["kv_writeback_route"] = (
            "device_authoritative" if bound > 0 else "host_staging" if host > 0 else "NEITHER"
        )
    devs = doc.get("alloc_device_frame_session_devices")
    if devs:
        fact["device_name"] = devs if isinstance(devs, str) else json.dumps(devs)

    prev = _COUNTER_HISTORY[-1] if _COUNTER_HISTORY else {}
    _COUNTER_HISTORY.append(doc)
    for key, out in (
        ("claimed_nodes", "claimed_nodes_this_arm"),
        ("dispatches_executed", "dispatches_executed_this_arm"),
    ):
        cur, old = doc.get(key), prev.get(key, 0)
        if isinstance(cur, int) and isinstance(old, int):
            fact[out] = cur - old
            fact[key + "_process_cumulative"] = cur
    claimed = fact.get("claimed_nodes_this_arm")
    fact["attribution"] = (
        "ATTRIBUTED"
        if isinstance(claimed, int) and claimed > 0
        else "UNATTRIBUTED (this arm's Vulkan session claimed no node; its residual is CPU-vs-CPU)"
    )
    fact["attribution_note"] = (
        "claimed_nodes is process-cumulative; this is the delta since the previous arm"
    )
    return fact


def run_both_eps(model_path: Path, feeds: dict, counters_path: Path) -> dict:
    import onnxruntime as ort

    import _models as m

    os.environ["ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE"] = str(counters_path)
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    vk_sess = ort.InferenceSession(str(model_path), opts, providers=m.EP_PROVIDERS)
    providers = vk_sess.get_providers()
    if m.EP_NAME not in providers:
        return {"instrument_error": f"{m.EP_NAME} absent from session: {providers}"}
    vk_out = vk_sess.run(None, feeds)
    del vk_sess

    cpu_opts = ort.SessionOptions()
    cpu_opts.log_severity_level = 3
    cpu_sess = ort.InferenceSession(
        str(model_path), cpu_opts, providers=["CPUExecutionProvider"]
    )
    cpu_out = cpu_sess.run(None, feeds)
    names = [o.name for o in cpu_sess.get_outputs()]
    del cpu_sess
    return {"vk": vk_out, "cpu": cpu_out, "names": names, "providers": providers}


def register_ep() -> dict:
    """Register the built EP cdylib, exactly as `tests/ops/conftest.py` does.

    Reports what it did rather than assuming: a probe that quietly ran CPU-vs-CPU is the
    failure mode this whole suite exists to prevent one level down.
    """
    import onnxruntime as ort

    import _models as m

    lib = os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB")
    if not lib:
        default = REPO / "rust" / "target" / "release" / "onnxruntime_vulkan_ep.dll"
        if default.is_file():
            lib = str(default)
            os.environ["ONNXRUNTIME_VULKAN_EP_LIB"] = lib
    if not lib or not Path(lib).is_file():
        return {"registered": False, "why": f"EP cdylib not found (ONNXRUNTIME_VULKAN_EP_LIB={lib})"}
    try:
        ort.register_execution_provider_library(m.EP_NAME, str(Path(lib).resolve()))
        return {"registered": True, "lib": lib}
    except Exception as exc:  # noqa: BLE001
        if "already" in str(exc).lower():
            return {"registered": True, "lib": lib, "note": "already registered"}
        return {"registered": False, "why": repr(exc), "lib": lib}


def make_isolated_final_norm(scratch: Path) -> "tuple[Path, dict]":
    """A standalone model holding the final SkipSimplifiedLayerNormalization and nothing else."""
    import onnx
    from onnx import numpy_helper

    scratch.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(MODEL_FILE), load_external_data=False)
    node = next(n for n in model.graph.node if n.name == FINAL_NORM_NODE)
    eps = float(
        next(onnx.helper.get_attribute_value(a) for a in node.attribute if a.name == "epsilon")
    )
    g_t = next(i for i in model.graph.initializer if i.name == FINAL_NORM_GAMMA)
    _materialise(g_t)

    ins = [
        onnx.helper.make_tensor_value_info(nm, onnx.TensorProto.FLOAT16, [1, 1, 3072])
        for nm in node.input[:2]
    ]
    outs = [
        onnx.helper.make_tensor_value_info(node.output[0], onnx.TensorProto.FLOAT16, [1, 1, 3072])
    ]
    # Only output_0 is kept: the node declares more outputs than this graph needs, and an
    # unused optional output is dropped rather than left dangling.
    node = onnx.helper.make_node(
        node.op_type,
        list(node.input),
        [node.output[0]],
        name=node.name,
        domain=node.domain,
        epsilon=eps,
    )
    g = onnx.helper.make_graph([node], "final_norm_only", ins, outs, initializer=[g_t])
    sub = onnx.helper.make_model(g, opset_imports=list(model.opset_import))
    sub.ir_version = model.ir_version
    out = scratch / "final_norm_only.onnx"
    onnx.save(sub, str(out))
    return out, {
        "input_names": list(node.input[:2]),
        "output_name": node.output[0],
        "epsilon": eps,
        "gamma": numpy_helper.to_array(g_t).astype(np.float64).reshape(-1),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=os.environ.get("ONNXRUNTIME_EP_VULKAN_DEVICE", "0"))
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--scratch", default=str(REPO / "bench" / "scratch" / "logits_reduction")
    )
    args = ap.parse_args(argv)

    os.environ["ONNXRUNTIME_EP_VULKAN_DEVICE"] = str(args.device)
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)

    import _models as m  # noqa: F401  (imported for its side-effect-free helpers)
    from test_phi35 import _build_phi35_feeds

    rec: dict = {
        "probe": "logits_reduction",
        "question": "is the logits' 12-ULP step chain depth, or the lm_head reduction?",
        "device_selector_requested": str(args.device),
        "predictions": PREDICTIONS,
        "lm_head_node": LM_HEAD_NODE,
        "hidden_tensor": HIDDEN_TENSOR,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    rec["ep_registration"] = register_ep()

    # -- Arm A: the full model with the final hidden state tapped out ---------------------
    tapped = None
    try:
        tapped = make_tapped_model(scratch, extra=RESIDUAL_STREAM + [FINAL_NORM_SKIP])
        feeds = _build_phi35_feeds()
        counters_a = scratch / "counters_armA.json"
        res = run_both_eps(tapped, feeds, counters_a)
        if "instrument_error" in res:
            rec["arm_a_full_model_tap"] = {"status": "ERROR(instrument)", **res}
        else:
            idx = res["names"].index(HIDDEN_TENSOR)
            li = res["names"].index("logits")
            hidden_vk, hidden_cpu = res["vk"][idx], res["cpu"][idx]

            # The depth curve on the RESIDUAL STREAM -- the chain the logits end. This is
            # not the KV curve: the 64 KV outputs are attention products, plotted already
            # and flat at 1 ULP. `H_depth` says *this* curve is the one that climbs.
            curve = []
            for layer, nm in enumerate(RESIDUAL_STREAM):
                if nm not in res["names"]:
                    continue
                j = res["names"].index(nm)
                st = ulp_stats(res["vk"][j], res["cpu"][j])
                curve.append(
                    {
                        "layer": layer,
                        "median_ulp_diff": st["median_ulp_diff"],
                        "p99_ulp_diff": st["p99_ulp_diff"],
                        "max_abs_diff": st["max_abs_diff"],
                    }
                )
            meds = [c["median_ulp_diff"] for c in curve]
            steps = [round(b - a, 6) for a, b in zip(meds, meds[1:])]
            rec["arm_a_full_model_tap"] = {
                "status": "MEASURED",
                "hidden_state": ulp_stats(hidden_vk, hidden_cpu),
                "logits": ulp_stats(res["vk"][li], res["cpu"][li]),
                "hidden_shape": list(np.shape(hidden_cpu)),
                "residual_stream_depth_curve": curve,
                "residual_stream_layers_measured": len(curve),
                "residual_stream_first_median": meds[0] if meds else None,
                "residual_stream_last_median": meds[-1] if meds else None,
                "residual_stream_largest_layer_to_layer_step": max(steps) if steps else None,
                "residual_stream_monotone_nondecreasing": (
                    all(s >= 0 for s in steps) if steps else None
                ),
                "curve_reading": (
                    "H_depth predicts this climbs with depth while the 64 KV outputs stay "
                    "flat at 1 ULP. A flat residual stream would leave the 6-ULP hidden "
                    "state unexplained and this conclusion unsupported."
                ),
                "outputs_in_session_order": res["names"][:2] + ["..."],
                **route_and_device(counters_a),
            }
            np.save(scratch / "hidden_cpu.npy", hidden_cpu)
            np.save(scratch / "hidden_vk.npy", hidden_vk)
            for nm, fn in ((RESIDUAL_STREAM[31], "stream31"), (FINAL_NORM_SKIP, "downproj31")):
                if nm in res["names"]:
                    np.save(scratch / f"{fn}_cpu.npy", res["cpu"][res["names"].index(nm)])
                    np.save(scratch / f"{fn}_vk.npy", res["vk"][res["names"].index(nm)])

            # The chain, hop by hop, so the reader sees the structure rather than my prose.
            # ULP is per-tensor-scale, so a ratio between hops is a ratio of RELATIVE
            # errors -- which is the right comparison across tensors of different
            # magnitude, and is named here rather than left to be assumed.
            hops = [
                ("residual stream, layer 0", meds[0] if meds else None),
                ("residual stream, layer 31", meds[-1] if meds else None),
                (
                    "final RMSNorm output (feeds lm_head)",
                    ulp_stats(hidden_vk, hidden_cpu)["median_ulp_diff"],
                ),
                ("logits", ulp_stats(res["vk"][li], res["cpu"][li])["median_ulp_diff"]),
            ]
            rec["arm_a_full_model_tap"]["chain_hop_by_hop_median_ulp"] = [
                {"where": w, "median_ulp_diff": v} for w, v in hops
            ]
    except Exception as exc:  # noqa: BLE001
        rec["arm_a_full_model_tap"] = {"status": "ERROR(instrument)", "error": repr(exc)}
    finally:
        # The tapped copy lives in the shared model cache only for as long as arm A needs it.
        if tapped is not None:
            try:
                Path(tapped).unlink(missing_ok=True)
            except OSError as exc:  # noqa: PERF203
                rec.setdefault("cleanup_warnings", []).append(f"{tapped}: {exc}")

    # -- Arm B: the lm_head alone, identical input on both EPs ----------------------------
    weights = None
    try:
        iso, weights = make_isolated_lm_head(scratch)
        hidden_cpu = np.load(scratch / "hidden_cpu.npy")
        feeds_b = {weights["input_name"]: hidden_cpu.astype(np.float16)}
        counters_b = scratch / "counters_armB.json"
        res = run_both_eps(iso, feeds_b, counters_b)
        if "instrument_error" in res:
            rec["arm_b_isolated_lm_head"] = {"status": "ERROR(instrument)", **res}
        else:
            rec["arm_b_isolated_lm_head"] = {
                "status": "MEASURED",
                "input_identical_on_both_eps": True,
                "note": (
                    "the SAME fp16 hidden state is fed to both EPs, so nothing upstream "
                    "of this node can contribute to the residual"
                ),
                **ulp_stats(res["vk"][0], res["cpu"][0]),
                **route_and_device(counters_b),
            }
            np.save(scratch / "logits_iso_vk.npy", res["vk"][0])
            np.save(scratch / "logits_iso_cpu.npy", res["cpu"][0])
    except Exception as exc:  # noqa: BLE001
        rec["arm_b_isolated_lm_head"] = {"status": "ERROR(instrument)", "error": repr(exc)}

    # -- Arm C: a second reference. Which of the two is further from the true value? ------
    try:
        if weights is None:
            _iso, weights = make_isolated_lm_head(scratch)
        a = weights["attrs"]
        w64 = dequantize_nbits(
            weights["packed"],
            weights["scales"],
            n=a["N"],
            k=a["K"],
            block_size=a["block_size"],
            bits=a["bits"],
        )
        hidden = np.load(scratch / "hidden_cpu.npy").reshape(-1).astype(np.float64)
        ref64 = w64 @ hidden
        ref16 = ref64.astype(np.float16)

        vk16 = np.load(scratch / "logits_iso_vk.npy").reshape(-1)
        cpu16 = np.load(scratch / "logits_iso_cpu.npy").reshape(-1)

        # Liveness: a dequantisation I got wrong disagrees with BOTH EPs and would be
        # reported as "both are wrong" -- the most confident way to be useless.
        cpu_vs_ref = ulp_stats(cpu16, ref16)
        vk_vs_ref = ulp_stats(vk16, ref16)
        live = cpu_vs_ref["median_ulp_diff"] <= 8.0
        rec["arm_c_float64_reference"] = {
            "status": "MEASURED" if live else "ERROR(instrument)",
            "reference_liveness": {
                "check": "the float64 reference must reproduce the CPU EP to a few ULP",
                "cpu_median_ulp_vs_reference": cpu_vs_ref["median_ulp_diff"],
                "threshold": 8.0,
                "live": bool(live),
                "if_not_live": (
                    "the dequantisation is wrong, not the EPs; no verdict is drawn from "
                    "this arm"
                ),
            },
            "cpu_ep_vs_float64": cpu_vs_ref,
            "vulkan_ep_vs_float64": vk_vs_ref,
            "which_is_further_from_true": (
                "UNMEASURED"
                if not live
                else "vulkan"
                if vk_vs_ref["median_ulp_diff"] > cpu_vs_ref["median_ulp_diff"]
                else "cpu"
                if cpu_vs_ref["median_ulp_diff"] > vk_vs_ref["median_ulp_diff"]
                else "neither (equal)"
            ),
        }

        # -- Arm D: the envelope of correct-but-differently-ordered implementations -------
        w32 = w64.astype(np.float32)
        a32 = hidden.astype(np.float32)
        orders = reduce_orders(a32, w32, block_size=a["block_size"])
        per_order = {
            name: ulp_stats(v.astype(np.float16), ref16) for name, v in orders.items()
        }
        pairwise_spread = {}
        keys = list(orders)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                s = ulp_stats(
                    orders[keys[i]].astype(np.float16), orders[keys[j]].astype(np.float16)
                )
                pairwise_spread[f"{keys[i]}__vs__{keys[j]}"] = {
                    "median_ulp_diff": s["median_ulp_diff"],
                    "p99_ulp_diff": s["p99_ulp_diff"],
                    "max_ulp_diff": s["max_ulp_diff"],
                }
        envelope = (
            max(v["median_ulp_diff"] for v in pairwise_spread.values())
            if pairwise_spread
            else None
        )
        envelope_max = (
            max(v["max_ulp_diff"] for v in pairwise_spread.values())
            if pairwise_spread
            else None
        )
        rec["arm_d_accumulation_order_envelope"] = {
            "status": "MEASURED",
            "what": (
                "five legitimate accumulation orders for the same reduction, each with an "
                "fp32 accumulator as both implementations use, each rounded to fp16"
            ),
            "orders": list(orders),
            "each_vs_float64": {k: v["median_ulp_diff"] for k, v in per_order.items()},
            "pairwise_median_ulp": pairwise_spread,
            "envelope_median_ulp": envelope,
            "envelope_max_ulp": envelope_max,
            "reading": (
                "a residual inside this envelope is not evidence of a defect however "
                "large the number is; it is how far two correct kernels may sit apart. "
                "An envelope of 0 at the median says something stronger and unexpected: "
                "for THIS reduction, fp16 storage rounding swamps accumulation order "
                "entirely, so order cannot manufacture a residual here at all."
            ),
        }
    except Exception as exc:  # noqa: BLE001
        rec["arm_c_float64_reference"] = {"status": "ERROR(instrument)", "error": repr(exc)}

    # -- Arm E: the falsifier this record names, actually run -----------------------------
    # "The node is clean" must be a property of the KERNEL, not of the one vector it was
    # fed. Arm B is repeated on a random fp16 vector and on the hidden state scaled by
    # 2**-4; if the median moves, arm B measured an input and not an implementation.
    try:
        iso = scratch / "lm_head_only.onnx"
        hidden_cpu = np.load(scratch / "hidden_cpu.npy").astype(np.float16)
        rng = np.random.default_rng(20260803)
        cases = {
            "random_fp16": (rng.standard_normal(hidden_cpu.shape) * 0.5).astype(np.float16),
            "hidden_scaled_2em4": (hidden_cpu.astype(np.float32) * 2.0**-4).astype(np.float16),
            "hidden_scaled_2ep4": (hidden_cpu.astype(np.float32) * 2.0**4).astype(np.float16),
        }
        name_in = weights["input_name"] if weights else None
        arm_e = {"status": "MEASURED", "cases": {}}
        for label, vec in cases.items():
            counters_e = scratch / f"counters_armE_{label}.json"
            r = run_both_eps(iso, {name_in: vec}, counters_e)
            if "instrument_error" in r:
                arm_e["cases"][label] = {"status": "ERROR(instrument)", **r}
                continue
            arm_e["cases"][label] = ulp_stats(r["vk"][0], r["cpu"][0])
        meds = [
            v.get("median_ulp_diff")
            for v in arm_e["cases"].values()
            if isinstance(v, dict) and "median_ulp_diff" in v
        ]
        base = rec.get("arm_b_isolated_lm_head", {}).get("median_ulp_diff")
        arm_e["all_medians_including_arm_b"] = ([base] if base is not None else []) + meds
        arm_e["falsifier_fired"] = bool(
            meds and base is not None and max(abs(x - base) for x in meds) > 2.0
        )
        arm_e["reading"] = (
            "falsifier_fired == true would void the conclusion: the lm_head's residual "
            "would then depend on which vector it was handed, so arm B measured an input."
        )
        rec["arm_e_falsifier_other_inputs"] = arm_e
    except Exception as exc:  # noqa: BLE001
        rec["arm_e_falsifier_other_inputs"] = {"status": "ERROR(instrument)", "error": repr(exc)}

    # -- Arm F: the final RMSNorm hop, ISOLATED, the same way the lm_head was -------------
    # Round 36's surprise: the residual stream is FLAT (0 -> 3 ULP over 32 blocks) and the
    # number doubles twice at the very end -- 3 at the stream, 6 out of the final RMSNorm,
    # 12 at the logits. Arm B accounts for the second doubling. This arm asks the same
    # question of the FIRST doubling, on a node the EP does claim.
    #
    # THE VERSION OF THIS ARM I THREW AWAY, because it is the mistake this project keeps
    # making: it compared a float64 reference built from the CPU EP's tapped inputs against
    # the Vulkan EP's IN-SITU output, which was produced from the Vulkan EP's own already
    # divergent inputs. That reports `which_is_further_from_true: vulkan` by construction --
    # it is arm A's 6 ULP wearing a reference's clothes, and it looked like a located
    # defect. Isolation means identical inputs on both sides or it means nothing.
    try:
        iso_fn, fn = make_isolated_final_norm(scratch)
        s16 = np.load(scratch / "stream31_cpu.npy").astype(np.float16)
        d16 = np.load(scratch / "downproj31_cpu.npy").astype(np.float16)
        feeds_f = {fn["input_names"][0]: s16, fn["input_names"][1]: d16}
        counters_f = scratch / "counters_armF.json"
        r = run_both_eps(iso_fn, feeds_f, counters_f)
        if "instrument_error" in r:
            rec["arm_f_isolated_final_rmsnorm"] = {"status": "ERROR(instrument)", **r}
        else:
            total = s16.astype(np.float64).reshape(-1) + d16.astype(np.float64).reshape(-1)
            ref64 = total / np.sqrt(np.mean(total * total) + fn["epsilon"]) * fn["gamma"]
            ref16 = ref64.astype(np.float16)
            cpu16 = r["cpu"][0].reshape(-1)
            vk16 = r["vk"][0].reshape(-1)
            cpu_vs = ulp_stats(cpu16, ref16)
            vk_vs = ulp_stats(vk16, ref16)
            live = cpu_vs["median_ulp_diff"] <= 8.0
            route = route_and_device(counters_f)
            attributed = str(route.get("attribution", "")).startswith("ATTRIBUTED")
            rec["arm_f_isolated_final_rmsnorm"] = {
                "status": "MEASURED" if (live and attributed) else "ERROR(instrument)",
                "node": FINAL_NORM_NODE,
                "op_type": "com.microsoft::SkipSimplifiedLayerNormalization",
                "epsilon": fn["epsilon"],
                "input_identical_on_both_eps": True,
                "vk_vs_cpu": ulp_stats(r["vk"][0], r["cpu"][0]),
                "cpu_ep_vs_float64": cpu_vs,
                "vulkan_ep_vs_float64": vk_vs,
                "reference_liveness": {
                    "cpu_median_ulp_vs_reference": cpu_vs["median_ulp_diff"],
                    "threshold": 8.0,
                    "live": bool(live),
                    "if_not_live": "the RMSNorm reference is wrong, not the EPs; no verdict here",
                },
                "which_is_further_from_true": (
                    "UNMEASURED"
                    if not (live and attributed)
                    else "vulkan"
                    if vk_vs["median_ulp_diff"] > cpu_vs["median_ulp_diff"]
                    else "cpu"
                    if cpu_vs["median_ulp_diff"] > vk_vs["median_ulp_diff"]
                    else "neither (equal)"
                ),
                **route,
            }
    except Exception as exc:  # noqa: BLE001
        rec["arm_f_isolated_final_rmsnorm"] = {"status": "ERROR(instrument)", "error": repr(exc)}

    # -- Verdict -------------------------------------------------------------------------
    b = rec.get("arm_b_isolated_lm_head", {})
    med = b.get("median_ulp_diff") if b.get("status") == "MEASURED" else None
    b_attributed = str(b.get("attribution", "")).startswith("ATTRIBUTED")
    fired = rec.get("arm_e_falsifier_other_inputs", {}).get("falsifier_fired")
    if med is None:
        rec["conclusion"] = {"verdict": "UNMEASURED", "why": "arm B did not produce a number"}
    elif not b_attributed:
        rec["conclusion"] = {
            "verdict": "UNATTRIBUTED",
            "why": (
                "arm B's Vulkan session claimed no node, so its residual is CPU-vs-CPU and "
                f"the {med} ULP means nothing. Attribution: {b.get('attribution')}"
            ),
        }
    elif fired:
        rec["conclusion"] = {
            "verdict": "VOID",
            "why": (
                "arm E fired: the lm_head's residual depends on which vector it is fed, so "
                "arm B measured an input rather than a kernel and neither hypothesis is "
                "supported by it."
            ),
        }
    elif med <= 2.0:
        rec["conclusion"] = {
            "verdict": "H_depth",
            "why": (
                f"the lm_head alone, on an identical input, is {med} median ULP -- like any "
                "other MatMulNBits here. The 12 ULP in the full model is inherited, not made "
                "at this node."
            ),
        }
    elif med >= 9.0:
        rec["conclusion"] = {
            "verdict": "H_proj",
            "why": (
                f"the lm_head alone reproduces {med} median ULP with the chain removed; the "
                "residual is manufactured at this node's reduction."
            ),
        }
    else:
        rec["conclusion"] = {
            "verdict": "INCONCLUSIVE",
            "why": (
                f"arm B is {med} median ULP -- between the two predictions. Neither "
                "hypothesis is confirmed and the record does not round to the nearer one."
            ),
        }

    rec["falsifier"] = {
        "what_would_fail_if_this_is_wrong": (
            "arm B is the whole conclusion, so the run that fails it is arm B measured "
            "against a DIFFERENT hidden state. If the residual at the lm_head depends on "
            "which input it is fed, then 'the node is clean' was a property of one vector "
            "and not of the kernel. Re-running arm B on (i) a random fp16 vector of the "
            "same shape and (ii) the hidden state scaled by 2**-4 must give the same "
            "median ULP; if it does not, this conclusion is void."
        ),
        "reachable": True,
        "also": (
            "arm C's verdict is void if the reference liveness check fails, and that check "
            "is reported rather than assumed."
        ),
    }
    rec["not_done_here"] = [
        "atol is not moved; the tolerance unit is Morpheus's ruling and was already filed",
        "no criterion-10 row is closed; the verdict stays DIVERGENT",
    ]
    rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    text = json.dumps(rec, indent=1, sort_keys=True, default=str)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    print(text[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
