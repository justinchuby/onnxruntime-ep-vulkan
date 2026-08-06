#!/usr/bin/env python3
"""§8.9.24(4): a float64 answer to **which side is wrong** on criterion 10's outputs 0, 63, 64.

THE RULING THAT MADE THIS BLOCKING
==================================
`docs/DESIGN.md` §8.9.24 ruling (4), Morpheus 2026-08-04:

    "No motion to change criterion 10's tolerance, unit, predicate or verdict structure is
     entertained until outputs 0, 63 and 64 have a float64 answer to 'which side is
     wrong'. Owner: Trinity."

`DIVERGENT` has meant *the CPU EP and we disagree* for this project's entire life. Round 36
asked which of the two was actually further from the true value for the first time, at two
nodes, and the answer at the final RMSNorm was that **we are bit-exact and ORT's CPU EP
carries the 1 ULP**. On the three outputs that decide criterion 10, nobody has asked.

THE SHAPE THIS ANSWER MUST NOT HAVE, AND THE GUARD IS MECHANICAL
=================================================================
Morpheus's warning is the reason this file has a `--check-no-motion` mode:

    "If the reference is wrong, the remedy is a different oracle -- and the argument would
     have been made in the direction of loosening, using the reference's own error as the
     budget."

So this probe **determines which side is further from the true value and proposes nothing
that follows from it.** It emits no tolerance, no unit, no predicate and no verdict. That
is not a promise in a docstring: `assert_record_proposes_no_motion` scans the emitted
record for any key or string that could carry a tolerance motion and raises, and
`tests/ops/test_criterion10_side.py` runs it as a gate. A result that cannot be quoted as
a budget cannot be laundered into one.

THE METHOD, AND THE TRAP IT IS BUILT AROUND
===========================================
My own round-36 near-miss is the whole method here. Arm F compared a float64 reference
built from the **CPU EP's** tapped inputs against the Vulkan EP's **in-situ** output --
which the Vulkan EP produced from its own, already divergent, inputs. That reports
`which_is_further_from_true: "vulkan"` **by construction**, and it named a node the EP
executes bit-exactly.

    ISOLATION MEANS IDENTICAL INPUTS ON BOTH SIDES OR IT MEANS NOTHING.

Every oracle arm below feeds **the same bytes** to both EPs -- the CPU-tapped fp16 tensor,
fed to both -- and compares each EP's output against a float64 reference computed from
**that same** tensor. The in-situ numbers are measured too, and they are labelled
`INHERITANCE`, not `ORACLE`: they answer *how much of this output's divergence arrived at
the node* and they are not an answer to which side is wrong.

WHAT THE THREE OUTPUTS ACTUALLY ARE, WHICH IS WHY THIS IS TRACTABLE AT ALL
==========================================================================
    output 0   `logits`            <- /lm_head/MatMul_Q4, K=3072 -> N=32064
    output 63  `present.31.key`    <- /model/layers.31/attn/GroupQueryAttention, out[1]
    output 64  `present.31.value`  <- the same node, out[2]

The criterion's feed is one token with an empty KV cache (`_build_phi35_feeds`), so the
GQA node runs at **sequence position 0**. Its `present` outputs are the rotary-embedded K
and the unmodified V, sliced out of the packed QKV projection.

The form of the rotation at position 0 is **measured off the tapped caches**
(`rope_form_at_position_0`), never derived from the position index, and this is not a
formality: the first draft of this probe predicted `cos_cache[0] == 1` and therefore an
exact *copy*. **That prediction was false.** Phi-3.5's long-rope attention factor is folded
into the cache, so `cos_cache[0]` is uniformly `1.1904296875` and `sin_cache[0]` is zero.
The rotation degenerates not to the identity but to a **single multiply** -- which is still
ONE correctly-rounded operation, so the float64 reference is still envelope-free and still
an oracle. The conclusion survived; the stated reason for it did not. Had the cache carried
a nonzero sine, the same unmeasured assumption would have reported a rotation defect as a
copy defect. A PREDICTION IS NOT A READING.

That gives outputs 63 and 64 something outputs generally do not have: a reference that is
**exact in float64**, with no accumulation envelope to argue about. And it relocates the
question, because a single scaling multiply cannot manufacture a residual out of equal
inputs -- so arm C isolates the `MatMulNBits` that produces the packed QKV, which is where
any residual in 63/64 has to have been made. Arm E then closes that loop mechanically:
it reconstructs each in-situ residual from the in-situ QKV tap and checks it against the
residual criterion 10 actually reports, so "inherited" is an identity rather than a story.

SCREENING, ON EVERY ARM, BECAUSE A SILENT CPU REBUILD EXITS ZERO
================================================================
Switch established this week that at ctx 4096 ORT rebuilds all 355 nodes on the CPU EP,
**exits 0, and executes zero EP dispatches** -- a run that reads as clean. Every arm here
screens `dispatches_executed > 0` and `claimed_nodes > 0` as **deltas** (the counters are
process-cumulative; round 36's arm B read 356 where arm A read 355 for one node), names
its device off the run rather than off the selector, and reports `UNATTRIBUTED` rather
than a number when the delta is zero. An isolated arm the EP declined is CPU-vs-CPU and
its 0-ULP residual is the most convincing possible way to have measured nothing.

Run:
    $env:VULKAN_SDK="C:\\VulkanSDK\\1.4.350.0"; $env:PATH="$env:VULKAN_SDK\\Bin;$env:PATH"
    $env:ONNXRUNTIME_VULKAN_EP_LIB="<repo>/rust/target/release/onnxruntime_vulkan_ep.dll"
    python bench/results/probe_criterion10_side.py --device 0 --out bench/results/criterion10_side-dev0.json

    python bench/results/probe_criterion10_side.py --selftest    # no GPU, no model
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests" / "ops"))

# ARCHIVAL: pinned to the exact Foundry cache layout this investigation measured against
# (the pre-2026-08-05 "...-cuda-gpu/cuda-int4-rtn-block-32/..." catalog revision, issue
# #11). Intentionally NOT auto-resolved against the live cache: a live resolver could
# silently pick a *different* cached revision than the one this result was measured
# against, which would misattribute a new run to an old artifact. Override PHI35_MODEL to
# replay against a different artifact explicitly (issue #19).
MODEL_DIR = Path(
    r"C:\Users\justinchu\.foundry\cache\models\Microsoft"
    r"\Phi-3.5-mini-instruct-cuda-gpu\cuda-int4-rtn-block-32"
)
MODEL_FILE = Path(
    os.environ.get(
        "PHI35_MODEL",
        str(MODEL_DIR / "phi-3.5-mini-instruct-cuda-int4-rtn-block-32.onnx"),
    )
)

LAYER = 31
GQA_NODE = f"/model/layers.{LAYER}/attn/GroupQueryAttention"
QKV_NODE = f"/model/layers.{LAYER}/attn/qkv_proj/MatMul_Q4"
QKV_IN = f"/model/layers.{LAYER}/input_layernorm/output_0"
QKV_OUT = f"/model/layers.{LAYER}/attn/qkv_proj/MatMul/output_0"
LM_HEAD_NODE = "/lm_head/MatMul_Q4"
HIDDEN_TENSOR = "/model/layers.32/final_norm_layernorm/output_0"

#: Tapped so the isolated arms can be fed the run's own tensors rather than a reconstruction.
TAPS = [QKV_IN, QKV_OUT, HIDDEN_TENSOR, "cos_cache", "sin_cache"]

#: The reference liveness bar. A float64 reference that cannot reproduce the CPU EP to a
#: few ULP is a reference I got wrong, and it would disagree with BOTH EPs -- reported as
#: "both are wrong", the most confident way to be useless.
LIVENESS_ULP = 8.0

PREDICTIONS = {
    "registered_before_measuring": True,
    "note": (
        "A PREDICTION IS NOT A READING (docs/DESIGN.md §8.9.24(6)). These are recorded so "
        "the measured values can contradict them; nothing below is quoted as a result."
    ),
    "outputs_63_64": (
        "the GQA node is fed identical packed QKV on both EPs and at position 0 its "
        "present outputs are slices of that input, so BOTH EPs should be exact and "
        "which_is_further_from_true should read 'neither (equal)'. If either EP is not "
        "exact, that is a copy/layout defect and it is located at this node."
    ),
    "where_the_in_situ_63_64_residual_comes_from": (
        "the qkv_proj MatMulNBits at layer 31 -- because a copy cannot manufacture one"
    ),
    "output_0": (
        "round 36 measured the isolated lm_head at 'neither (equal)' and the isolated "
        "final RMSNorm at 'cpu'. Both re-measured here on the merged tree; a different "
        "answer means one of the two rounds was measuring something else."
    ),
    "what_would_make_this_inconclusive": (
        "any arm whose claimed_nodes delta is 0 (the EP declined the isolated graph, so "
        "the comparison is CPU-vs-CPU), or whose reference liveness check fails"
    ),
}


# ==========================================================================================
# The no-motion guard.  §8.9.24(4) permits this answer and forbids what follows from it.
# ==========================================================================================
#: Substrings that would indicate this record had grown a tolerance motion.  Deliberately
#: broad: the cost of a false positive is renaming a key, and the cost of a false negative
#: is the laundering route the ruling exists to close.
_MOTION_TOKENS = (
    "proposed_atol",
    "proposed_rtol",
    "suggested_atol",
    "suggested_rtol",
    "recommended_tol",
    "new_atol",
    "new_rtol",
    "tolerance_budget",
    "budget_from_reference",
    "should_loosen",
    "would_pass_if",
    "recommend",
)


class MotionInRecordError(AssertionError):
    """The record grew something that could be quoted as a tolerance motion."""


def assert_record_proposes_no_motion(rec, path: str = "record") -> None:
    """§8.9.24(4), as a refusal rather than as an intention.

    The danger this closes is specific and it is named in the ruling: an oracle result
    showing the *reference* is the further side would be argued in the direction of
    loosening, using the reference's own error as the budget. A record that cannot express
    a budget cannot be quoted as one.

    Note what this does NOT do. It does not stop anyone writing that argument elsewhere --
    nothing could. It stops **this artifact** from being the thing they quote, which is
    the only part I own.
    """
    if isinstance(rec, dict):
        for k, v in rec.items():
            lk = str(k).lower()
            for tok in _MOTION_TOKENS:
                if tok in lk:
                    raise MotionInRecordError(
                        f"{path}.{k}: §8.9.24(4) -- this record answers which side is "
                        f"wrong and may propose nothing that follows from it. The key "
                        f"name matches '{tok}'."
                    )
            assert_record_proposes_no_motion(v, f"{path}.{k}")
    elif isinstance(rec, (list, tuple)):
        for i, v in enumerate(rec):
            assert_record_proposes_no_motion(v, f"{path}[{i}]")


# ==========================================================================================
# References
# ==========================================================================================
def dequantize_nbits(
    packed: np.ndarray, scales: np.ndarray, *, n: int, k: int, block_size: int, bits: int
) -> np.ndarray:
    """Dequantise a MatMulNBits weight to float64. Symmetric RTN, zero point 8.

    Shared shape with `probe_logits_reduction.dequantize_nbits`; kept separate rather than
    imported because that module runs a model at import-adjacent scope and this one must
    stay importable with no ORT session. The two are checked against each other in
    `--selftest`.
    """
    if bits != 4:
        raise ValueError(f"only 4-bit is implemented; got {bits}")
    n_blocks = (k + block_size - 1) // block_size
    b = packed.reshape(n, n_blocks, block_size // 2)
    lo = (b & 0x0F).astype(np.int16)
    hi = (b >> 4).astype(np.int16)
    nib = np.empty((n, n_blocks, block_size), dtype=np.int16)
    nib[:, :, 0::2] = lo
    nib[:, :, 1::2] = hi
    s = scales.reshape(n, n_blocks).astype(np.float64)
    deq = (nib.astype(np.float64) - 8.0) * s[:, :, None]
    return deq.reshape(n, n_blocks * block_size)[:, :k]


def rope_reference_f64(
    k_heads: np.ndarray, cos_row: np.ndarray, sin_row: np.ndarray, *, interleaved: bool
) -> np.ndarray:
    """Rotary embedding in float64 for one sequence position.

    `k_heads` is `[heads, head_dim]` in float64. The half-split (non-interleaved) form is
    what this model declares (`rotary_interleaved = 0`), and the interleaved branch is
    present so that a model that declares the other form is refused loudly rather than
    silently rotated the wrong way.
    """
    head_dim = k_heads.shape[-1]
    rot = 2 * len(cos_row)
    if rot > head_dim:
        raise ValueError(f"rotary width {rot} exceeds head_dim {head_dim}")
    out = k_heads.astype(np.float64).copy()
    x = out[:, :rot]
    if interleaved:
        x0, x1 = x[:, 0::2], x[:, 1::2]
        r0 = x0 * cos_row - x1 * sin_row
        r1 = x0 * sin_row + x1 * cos_row
        x[:, 0::2], x[:, 1::2] = r0, r1
    else:
        half = rot // 2
        x0, x1 = x[:, :half], x[:, half:rot]
        r0 = x0 * cos_row - x1 * sin_row
        r1 = x0 * sin_row + x1 * cos_row
        x[:, :half], x[:, half:rot] = r0, r1
    out[:, :rot] = x
    return out


def ulp_stats(vk: np.ndarray, ref: np.ndarray, dtype=np.float16) -> dict:
    """Residual statistics in the shared comparator's units, with §8.9.24(3)'s companions.

    `_models` is the one implementation of "the ULP distribution of this pair"; a second
    one here would be a second answer nobody could reconcile.
    """
    import _models as m

    a = np.asarray(vk).reshape(-1).astype(dtype)
    b = np.asarray(ref).reshape(-1).astype(dtype)
    dist = m.ulp_distribution(a, b)
    tol, _why = m.tolerance_for_output(a)
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    out = {
        "median_ulp_diff": dist["median_ulp"],
        "p99_ulp_diff": dist["p99_ulp"],
        "max_ulp_diff": dist["max_ulp"],
        "max_ulp_at_scale_diff": dist["max_ulp_at_scale"]
        if dist["one_ulp_at_scale"]
        else None,
        "max_abs_diff": float(diff.max()) if diff.size else 0.0,
        "bit_exact": bool(np.array_equal(a.view(np.uint16), b.view(np.uint16)))
        if a.dtype == np.float16
        else bool(np.array_equal(a, b)),
        "elements": int(a.size),
        "elements_differing": int(np.count_nonzero(diff)),
        "tensor_scale": dist["tensor_scale"],
        "one_ulp_at_scale": dist["one_ulp_at_scale"],
        "at_scale_basis_note": (
            "max_ulp_at_scale_diff is None where the reference tensor has no finite "
            "nonzero scale: a residual divided by an undefined denominator is an absence, "
            "not a 0, and a 0 there would acquit"
        ),
    }
    out.update(
        m.allowance_in_ulps_at_scale(
            np.abs(b.astype(np.float64)), tol, dist["one_ulp_at_scale"], over="every element"
        )
    )
    out.update(m.ulp_element_basis_stats(diff, b, prefix="ulp_element_basis", dtype=a.dtype))
    m.assert_ulp_at_scale_row_is_complete(out, where="ulp_stats")
    return out


#: Every statistic on which "further from the true value" can be read off a pair of
#: residual distributions. All of them are reported; none of them is allowed to be the
#: only one reported. A verdict that holds on one and reverses on another is not a
#: verdict, and the round-36 -> round-37 disagreement on `lm_head` was exactly that
#: shape: round 36 discriminated on `median_ulp_diff` and read "neither (equal)" off two
#: zeros; this probe discriminates on `max_ulp_diff` and reads "cpu". Both readings are
#: true of their own statistic. Reporting one without the other makes a choice of
#: statistic look like a result that changed.
DISCRIMINATORS = ("max_ulp_diff", "p99_ulp_diff", "median_ulp_diff", "elements_differing", "max_abs_diff")

#: The discriminator quoted in `which_is_further_from_true`. `max_ulp_diff` because on
#: these tensors most elements are bit-exact on both sides, so the median is 0 on both
#: and carries no ordering information whatever -- "neither (equal)" off two zeros is a
#: null result wearing the clothes of a measurement.
PRIMARY_DISCRIMINATOR = "max_ulp_diff"


def _verdict_on(a, b) -> str:
    if a is None or b is None:
        return "UNMEASURED"
    if a == b:
        return "neither (equal)"
    return "vulkan" if a > b else "cpu"


def which_side(vk_vs_ref: dict, cpu_vs_ref: dict, *, key: str = PRIMARY_DISCRIMINATOR) -> dict:
    """The answer, on every discriminator at once, and nothing that follows from it.

    A single statistic can only ever answer "which side is further *by this statistic*".
    This project has already published one verdict that appeared to change between rounds
    when only the statistic had changed, so the verdict is reported on all of
    `DISCRIMINATORS` and the record carries whether they agree. `unanimous_direction` is
    the only field that should ever be quoted as "which side is wrong" without a
    qualifier; where the discriminators split, the split *is* the result.
    """
    by: dict[str, dict] = {}
    for d in DISCRIMINATORS:
        a, b = vk_vs_ref.get(d), cpu_vs_ref.get(d)
        by[d] = {
            "verdict": _verdict_on(a, b),
            "vulkan_value": a,
            "cpu_value": b,
            "margin": (abs(a - b) if a is not None and b is not None else None),
        }
    directional = {v["verdict"] for v in by.values() if v["verdict"] in ("vulkan", "cpu")}
    unmeasured = [d for d, v in by.items() if v["verdict"] == "UNMEASURED"]
    primary = by.get(key, {})
    out = {
        "which_is_further_from_true": primary.get("verdict", "UNMEASURED"),
        "discriminator": key,
        "vulkan_value": primary.get("vulkan_value"),
        "cpu_value": primary.get("cpu_value"),
        "margin": primary.get("margin"),
        "verdict_by_discriminator": by,
        "discriminators_conflict": len(directional) > 1,
        "discriminators_silent": sorted(
            d for d, v in by.items() if v["verdict"] == "neither (equal)"
        ),
        "discriminators_unmeasured": sorted(unmeasured),
        "unanimous_direction": (
            next(iter(directional))
            if len(directional) == 1
            else ("neither (no discriminator separates the two sides)" if not directional else None)
        ),
        "both_bit_exact_against_the_reference": bool(
            vk_vs_ref.get("bit_exact") and cpu_vs_ref.get("bit_exact")
        ),
        "reading": (
            "a verdict of 'neither (equal)' on two zeros is a null result, not a "
            "measurement of agreement; read both_bit_exact_against_the_reference, "
            "unanimous_direction and verdict_by_discriminator before quoting this row. "
            "Where discriminators_conflict is true there is no single answer and the "
            "conflict is the finding."
        ),
    }
    if out["discriminators_conflict"]:
        out["conflict_note"] = (
            "at least one statistic puts the Vulkan EP further from the float64 value and "
            "at least one puts ORT's CPU EP further; neither side is 'the wrong one' on "
            "this arm"
        )
    return out


# ==========================================================================================
# Model surgery and runs
# ==========================================================================================
def _materialise(t) -> None:
    """Pull one initialiser's bytes out of the 2.3 GB external blob, in place.

    `data_location` must be cleared afterwards or `numpy_helper.to_array` re-resolves the
    relative path against the current working directory (round 36).
    """
    import onnx
    from onnx.external_data_helper import load_external_data_for_tensor

    if t.HasField("data_location") and t.data_location == onnx.TensorProto.EXTERNAL:
        load_external_data_for_tensor(t, str(MODEL_DIR))
        t.ClearField("data_location")
        del t.external_data[:]


def make_tapped_model(extra: list[str]) -> Path:
    """A copy of the real model with `extra` tensors added as graph outputs.

    Written **into the model directory** under a process-unique name: ONNX resolves
    external data relative to the model file, copying the 2.3 GB blob is antisocial with
    six agents on this box, and hard-linking makes `onnx.load` refuse the file outright.
    """
    import onnx

    model = onnx.load(str(MODEL_FILE), load_external_data=False)
    have = {vi.name for vi in model.graph.output}
    for nm in extra:
        if nm in have:
            continue
        vi = onnx.ValueInfoProto()
        vi.name = nm
        model.graph.output.append(vi)
        have.add(nm)
    out = MODEL_DIR / f"phi35_side_{os.getpid()}.onnx"
    onnx.save(model, str(out))
    return out


def make_isolated_node(scratch: Path, node_name: str, filename: str) -> tuple[Path, dict]:
    """A standalone model holding exactly one node of the real graph, plus its initialisers.

    Input value-infos are declared from the tapped tensors at run time rather than from the
    parent graph's shape inference, because the parent's shapes are symbolic.
    """
    import onnx

    scratch.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(MODEL_FILE), load_external_data=False)
    node = next(n for n in model.graph.node if n.name == node_name)
    inits = {i.name: i for i in model.graph.initializer}
    keep = []
    init_names = []
    for nm in node.input:
        if nm and nm in inits:
            t = inits[nm]
            _materialise(t)
            keep.append(t)
            init_names.append(nm)
    meta = {
        "node": node_name,
        "op_type": node.op_type,
        "domain": node.domain,
        "inputs": list(node.input),
        "outputs": list(node.output),
        "initializers": init_names,
        "attrs": {
            a.name: onnx.helper.get_attribute_value(a) for a in node.attribute
        },
        "_node": node,
        "_opset": list(model.opset_import),
        "_ir": model.ir_version,
        "_keep": keep,
        "_path": scratch / filename,
    }
    return meta["_path"], meta


def finish_isolated_model(meta: dict, feed_shapes: dict, out_dtypes: dict) -> Path:
    """Materialise the isolated model once the tapped tensors' shapes are known."""
    import onnx

    node = meta["_node"]
    init_names = set(meta["initializers"])
    ins = []
    for nm in node.input:
        if not nm or nm in init_names:
            continue
        shape, dt = feed_shapes[nm]
        ins.append(onnx.helper.make_tensor_value_info(nm, dt, list(shape)))
    outs = [
        onnx.helper.make_tensor_value_info(nm, out_dtypes[nm], None)
        for nm in node.output
        if nm in out_dtypes
    ]
    kept_node = onnx.helper.make_node(
        node.op_type,
        list(node.input),
        [nm for nm in node.output if nm in out_dtypes],
        name=node.name,
        domain=node.domain,
        **{a.name: onnx.helper.get_attribute_value(a) for a in node.attribute},
    )
    g = onnx.helper.make_graph(
        [kept_node], "isolated", ins, outs, initializer=meta["_keep"]
    )
    sub = onnx.helper.make_model(g, opset_imports=meta["_opset"])
    sub.ir_version = meta["_ir"]
    onnx.save(sub, str(meta["_path"]))
    return meta["_path"]


_COUNTER_HISTORY: list[dict] = []


def route_and_device(counters_path: Path | None) -> dict:
    """Route, device name, and this arm's attribution -- all read off what the run emitted.

    Attribution is a **delta**: the counters are process-cumulative, so an absolute
    `claimed_nodes` credits every arm with everything that ran before it. A zero delta
    means the EP declined this graph, the arm is CPU-vs-CPU, and its residual means
    nothing however clean it looks.

    The dispatch screen is the one Switch's ctx-4096 work made mandatory: a silent CPU
    rebuild exits 0, raises nothing, and reads as a clean EP run.
    """
    fact = {
        "device_name": "UNOBSERVABLE",
        "device_name_source": "counters alloc_device_frame_session_devices, off the run",
        "claimed_nodes_this_arm": "UNOBSERVABLE",
        "dispatches_executed_this_arm": "UNOBSERVABLE",
        "attribution": "UNATTRIBUTED",
        "dispatch_screen": "UNOBSERVABLE(no counters file)",
    }
    if not counters_path or not Path(counters_path).exists():
        return fact
    try:
        doc = json.loads(Path(counters_path).read_text(encoding="utf-8-sig"))
    except Exception as exc:  # noqa: BLE001
        fact["counters_read_error"] = repr(exc)
        return fact
    doc = doc.get("counters", doc)
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
            fact[f"{key}_process_cumulative"] = cur
    claimed = fact.get("claimed_nodes_this_arm")
    disp = fact.get("dispatches_executed_this_arm")
    fact["attribution"] = (
        "ATTRIBUTED"
        if isinstance(claimed, int) and claimed > 0
        else "UNATTRIBUTED (this arm's Vulkan session claimed no node; CPU-vs-CPU)"
    )
    fact["dispatch_screen"] = (
        "PASS"
        if isinstance(disp, int) and disp > 0
        else "ERROR(instrument=zero_ep_dispatches: a silent CPU rebuild exits 0 and reads "
        "as a clean EP run -- Switch, ctx-4096, 2026-08-04)"
    )
    return fact


def run_both_eps(model_path: Path, feeds: dict, counters_path: Path) -> dict:
    """One session per EP on the same model and the same feeds."""
    import onnxruntime as ort

    import _models as m

    os.environ["ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE"] = str(counters_path)
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    vk_sess = ort.InferenceSession(str(model_path), opts, providers=m.EP_PROVIDERS)
    providers = vk_sess.get_providers()
    if m.EP_NAME not in providers:
        return {"instrument_error": f"{m.EP_NAME} absent from session: {providers}"}
    names = [o.name for o in vk_sess.get_outputs()]
    vk_out = vk_sess.run(None, feeds)
    del vk_sess

    cpu_opts = ort.SessionOptions()
    cpu_opts.log_severity_level = 3
    cpu_sess = ort.InferenceSession(
        str(model_path), cpu_opts, providers=["CPUExecutionProvider"]
    )
    cpu_out = cpu_sess.run(None, feeds)
    del cpu_sess
    return {"vk": vk_out, "cpu": cpu_out, "names": names, "providers": providers}


def register_ep() -> dict:
    """Register the built EP cdylib, and say so.

    `ONNXRUNTIME_VULKAN_EP_LIB` unset is a refusal, not a default: without it the whole
    probe runs CPU-vs-CPU, agrees perfectly, and proves nothing.
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
        return {
            "registered": False,
            "why": f"ONNXRUNTIME_VULKAN_EP_LIB unset or not a file ({lib})",
        }
    try:
        ort.register_execution_provider_library(m.EP_NAME, str(Path(lib).resolve()))
        return {"registered": True, "lib": str(Path(lib).resolve())}
    except Exception as exc:  # noqa: BLE001
        if "already" in str(exc).lower():
            return {"registered": True, "lib": lib, "note": "already registered"}
        return {"registered": False, "why": repr(exc), "lib": lib}


# ==========================================================================================
# Arms
# ==========================================================================================
def _liveness(cpu_vs_ref: dict, what: str) -> dict:
    live = (cpu_vs_ref.get("median_ulp_diff") or 0.0) <= LIVENESS_ULP
    return {
        "check": f"the float64 reference for {what} must reproduce the CPU EP to a few ULP",
        "cpu_median_ulp_vs_reference": cpu_vs_ref.get("median_ulp_diff"),
        "threshold": LIVENESS_ULP,
        "live": bool(live),
        "if_not_live": (
            "the reference is wrong, not the EPs; it would disagree with BOTH and be "
            "reported as 'both are wrong', which is the most confident way to be useless"
        ),
    }


def _oracle_arm(label: str, vk16, cpu16, ref16, route: dict, extra: dict) -> dict:
    vk_vs = ulp_stats(vk16, ref16)
    cpu_vs = ulp_stats(cpu16, ref16)
    live = _liveness(cpu_vs, label)
    attributed = str(route.get("attribution", "")).startswith("ATTRIBUTED")
    screened = route.get("dispatch_screen") == "PASS"
    ok = live["live"] and attributed and screened
    rec = {
        "status": "MEASURED" if ok else "ERROR(instrument)",
        "class": "ORACLE",
        "isolation": (
            "identical bytes fed to both EPs, and the float64 reference is computed from "
            "those same bytes -- round 36's arm-F trap, which reported "
            "which_is_further_from_true by construction"
        ),
        "vk_vs_cpu": ulp_stats(vk16, cpu16),
        "vulkan_ep_vs_float64": vk_vs,
        "cpu_ep_vs_float64": cpu_vs,
        "reference_liveness": live,
        **extra,
        **route,
    }
    rec.update(
        which_side(vk_vs, cpu_vs)
        if ok
        else {
            "which_is_further_from_true": "UNMEASURED",
            "why_unmeasured": {
                "reference_live": live["live"],
                "attributed": attributed,
                "dispatch_screen": route.get("dispatch_screen"),
            },
        }
    )
    return rec


def main(argv=None) -> int:  # noqa: PLR0912, PLR0915
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=os.environ.get("ONNXRUNTIME_EP_VULKAN_DEVICE", "0"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--scratch", default=str(REPO / "bench" / "scratch" / "c10_side"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return _selftest()

    import onnx

    os.environ["ONNXRUNTIME_EP_VULKAN_DEVICE"] = str(args.device)
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)

    import _models as m  # noqa: F401
    from test_phi35 import _build_phi35_feeds

    rec: dict = {
        "probe": "criterion10_side",
        "question": (
            "on criterion 10's failing outputs 0, 63 and 64 -- which of the Vulkan EP and "
            "ORT's CPU EP is further from the true value?"
        ),
        "ruling": "docs/DESIGN.md §8.9.24(4)",
        "device_selector_requested": str(args.device),
        "selector_caveat": "a selector is a request, not an identity; device_name is off the run",
        "predictions": PREDICTIONS,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    rec["ep_registration"] = register_ep()

    # -- Arm A: the full model, tapped. In-situ residuals and the inheritance chain. -----
    tapped = None
    taps: dict[str, np.ndarray] = {}
    try:
        tapped = make_tapped_model(TAPS)
        feeds = _build_phi35_feeds()
        counters_a = scratch / "counters_armA.json"
        res = run_both_eps(tapped, feeds, counters_a)
        if "instrument_error" in res:
            rec["arm_a_in_situ"] = {"status": "ERROR(instrument)", **res}
        else:
            names = res["names"]
            for nm in TAPS:
                if nm in names:
                    taps[nm] = res["cpu"][names.index(nm)]
                    taps[nm + "@vk"] = res["vk"][names.index(nm)]
            # Arm E reconstructs these from the QKV tap, so they have to survive arm A.
            for nm in ("present.31.key", "present.31.value"):
                if nm in names:
                    taps[nm] = res["cpu"][names.index(nm)]
                    taps[nm + "@vk"] = res["vk"][names.index(nm)]
            per = {}
            for label, nm in (
                ("logits (output 0)", "logits"),
                ("present.31.key (output 63)", "present.31.key"),
                ("present.31.value (output 64)", "present.31.value"),
                ("qkv_proj output (feeds the GQA node)", QKV_OUT),
                ("qkv_proj input (layer 31 input_layernorm)", QKV_IN),
                ("final RMSNorm output (feeds lm_head)", HIDDEN_TENSOR),
            ):
                if nm in names:
                    j = names.index(nm)
                    per[label] = ulp_stats(res["vk"][j], res["cpu"][j], dtype=res["cpu"][j].dtype)
            rec["arm_a_in_situ"] = {
                "status": "MEASURED",
                "class": "INHERITANCE",
                "not_an_oracle_answer": (
                    "these compare the two EPs to EACH OTHER on their own in-situ inputs. "
                    "They say how much divergence had arrived by each point; they say "
                    "nothing about which side is wrong, and a float64 reference laid "
                    "against them would report 'vulkan' by construction (round 36 arm F)"
                ),
                "per_tensor": per,
                **route_and_device(counters_a),
            }
            for nm, arr in taps.items():
                np.save(scratch / (nm.replace("/", "_").replace(".", "_") + ".npy"), arr)
    except Exception as exc:  # noqa: BLE001
        rec["arm_a_in_situ"] = {"status": "ERROR(instrument)", "error": repr(exc)}
    finally:
        if tapped is not None:
            try:
                Path(tapped).unlink(missing_ok=True)
            except OSError as exc:
                rec.setdefault("cleanup_warnings", []).append(f"{tapped}: {exc}")

    have_taps = all(k in taps for k in (QKV_IN, QKV_OUT, HIDDEN_TENSOR, "cos_cache", "sin_cache"))
    if not have_taps:
        rec["arms_bcd"] = {
            "status": "ERROR(instrument)",
            "why": f"arm A did not produce the taps the isolated arms need; got {sorted(taps)}",
        }

    # -- Arm B: outputs 63 and 64, isolated GQA, EXACT float64 reference ------------------
    if have_taps:
        try:
            _p, meta = make_isolated_node(scratch, GQA_NODE, "gqa31_only.onnx")
            attrs = meta["attrs"]
            qkv = taps[QKV_OUT]
            cos, sin = taps["cos_cache"], taps["sin_cache"]
            past = np.empty((1, int(attrs["kv_num_heads"]), 0, 96), dtype=np.float16)
            seqlens_k = np.array([0], dtype=np.int32)
            total_seq = np.array(1, dtype=np.int32)
            node_in = meta["inputs"]
            feed = {
                node_in[0]: qkv.astype(np.float16),
                node_in[3]: past,
                node_in[4]: past,
                node_in[5]: seqlens_k,
                node_in[6]: total_seq,
                node_in[7]: cos,
                node_in[8]: sin,
            }
            shapes = {k: (v.shape, onnx.helper.np_dtype_to_tensor_dtype(v.dtype)) for k, v in feed.items()}
            out_dtypes = {
                meta["outputs"][0]: onnx.TensorProto.FLOAT16,
                meta["outputs"][1]: onnx.TensorProto.FLOAT16,
                meta["outputs"][2]: onnx.TensorProto.FLOAT16,
            }
            iso = finish_isolated_model(meta, shapes, out_dtypes)
            counters_b = scratch / "counters_armB.json"
            res = run_both_eps(iso, feed, counters_b)
            if "instrument_error" in res:
                rec["arm_b_isolated_gqa31"] = {"status": "ERROR(instrument)", **res}
            else:
                route = route_and_device(counters_b)
                heads = int(attrs["num_heads"])
                kv_heads = int(attrs["kv_num_heads"])
                hd = qkv.reshape(-1).size // (heads + 2 * kv_heads)
                flat = qkv.reshape(-1).astype(np.float64)
                q_sz = heads * hd
                kv_sz = kv_heads * hd
                k_raw = flat[q_sz : q_sz + kv_sz].reshape(kv_heads, hd)
                v_raw = flat[q_sz + kv_sz : q_sz + 2 * kv_sz].reshape(kv_heads, hd)
                cos0 = np.asarray(cos)[0].astype(np.float64)
                sin0 = np.asarray(sin)[0].astype(np.float64)
                identity = bool(np.all(cos0 == 1.0) and np.all(sin0 == 0.0))
                sin_is_zero = bool(np.all(sin0 == 0.0))
                cos_is_constant = bool(np.unique(cos0).size == 1)
                # Three cases, and only the first two make the reference envelope-free.
                # This model turns out to be the SECOND one, which the first draft of this
                # probe predicted was the first one -- see `rope_form_at_position_0`.
                if identity:
                    rope_form = "IDENTITY"
                elif sin_is_zero:
                    rope_form = "PURE_SCALE"
                else:
                    rope_form = "ROTATION"
                exact = rope_form in ("IDENTITY", "PURE_SCALE")
                k_ref = rope_reference_f64(
                    k_raw, cos0, sin0, interleaved=bool(attrs.get("rotary_interleaved", 0))
                ) if int(attrs.get("do_rotary", 0)) else k_raw
                key_ref16 = k_ref.astype(np.float16).reshape(1, kv_heads, 1, hd)
                val_ref16 = v_raw.astype(np.float16).reshape(1, kv_heads, 1, hd)

                names = res["names"]
                ik, iv = names.index(meta["outputs"][1]), names.index(meta["outputs"][2])
                _exact_why = {
                    "IDENTITY": (
                        "cos_cache[0] is all-ones and sin_cache[0] is all-zeros, so the "
                        "rotation is the identity: present.key is the K slice of the "
                        "packed QKV verbatim and present.value is the V slice verbatim. "
                        "The float64 reference is a copy and is EXACT -- there is no "
                        "accumulation envelope to argue about, and any residual here is a "
                        "copy or layout defect, not arithmetic"
                    ),
                    "PURE_SCALE": (
                        "sin_cache[0] is all-zeros but cos_cache[0] is NOT all-ones, so "
                        "the rotation at position 0 degenerates to a single multiply by "
                        "that constant (the cross terms are multiplied by zero). One "
                        "multiply is ONE correctly-rounded operation, so the float64 "
                        "reference rounded to fp16 is still the exact answer any conformant "
                        "implementation must produce: there is no accumulation envelope. "
                        "It is NOT a copy -- the first draft of this probe predicted a copy "
                        "and was wrong, which is why the form is measured off the tapped "
                        "caches rather than derived from the position index"
                    ),
                    "ROTATION": (
                        "sin_cache[0] is not all-zeros, so the reference is a genuine "
                        "float64 rotation: two products and a sum per element, which "
                        "carries a rounding envelope. A residual of 1-2 ULP here is NOT "
                        "evidence of a defect on either side and this arm must not be read "
                        "as an oracle"
                    ),
                }[rope_form]
                shared = {
                    "node": GQA_NODE,
                    "op_type": "com.microsoft::GroupQueryAttention",
                    "sequence_position": 0,
                    "rope_form_at_position_0": rope_form,
                    "rope_is_identity_at_position_0": identity,
                    "rope_sin_is_zero_at_position_0": sin_is_zero,
                    "rope_cos_is_constant_at_position_0": cos_is_constant,
                    "rope_scale_at_position_0": (
                        float(cos0.reshape(-1)[0]) if cos_is_constant else None
                    ),
                    "rope_check_source": (
                        "cos_cache[0] and sin_cache[0] READ OFF THE RUN (tapped as graph "
                        "outputs in arm A), not assumed from the position index"
                    ),
                    "reference_is_exact": exact,
                    "reference_note": _exact_why,
                }
                rec["arm_b_present31_key"] = _oracle_arm(
                    "present.31.key", res["vk"][ik], res["cpu"][ik], key_ref16, route,
                    {**shared, "output_index_in_criterion10": 63},
                )
                # The route/counters delta is consumed by the first call; the second arm
                # reports the same run rather than a second delta, and says so.
                route_b2 = dict(route)
                route_b2["attribution_note"] = (
                    "same session as arm_b_present31_key; the counters delta is not "
                    "double-counted"
                )
                rec["arm_b_present31_value"] = _oracle_arm(
                    "present.31.value", res["vk"][iv], res["cpu"][iv], val_ref16, route_b2,
                    {**shared, "output_index_in_criterion10": 64},
                )
        except Exception as exc:  # noqa: BLE001
            rec["arm_b_isolated_gqa31"] = {"status": "ERROR(instrument)", "error": repr(exc)}

    # -- Arm C: where a residual in 63/64 can have been made: the qkv_proj MatMulNBits ----
    if have_taps:
        try:
            _p, meta = make_isolated_node(scratch, QKV_NODE, "qkv31_only.onnx")
            a = meta["attrs"]
            x16 = np.asarray(taps[QKV_IN]).astype(np.float16)
            node_in = meta["inputs"]
            feed = {node_in[0]: x16}
            shapes = {node_in[0]: (x16.shape, onnx.TensorProto.FLOAT16)}
            iso = finish_isolated_model(
                meta, shapes, {meta["outputs"][0]: onnx.TensorProto.FLOAT16}
            )
            counters_c = scratch / "counters_armC.json"
            res = run_both_eps(iso, feed, counters_c)
            if "instrument_error" in res:
                rec["arm_c_isolated_qkv_proj31"] = {"status": "ERROR(instrument)", **res}
            else:
                from onnx import numpy_helper

                packed = numpy_helper.to_array(meta["_keep"][0])
                scales = numpy_helper.to_array(meta["_keep"][1])
                w64 = dequantize_nbits(
                    packed, scales, n=int(a["N"]), k=int(a["K"]),
                    block_size=int(a["block_size"]), bits=int(a["bits"]),
                )
                ref16 = (w64 @ x16.reshape(-1).astype(np.float64)).astype(np.float16)
                rec["arm_c_isolated_qkv_proj31"] = _oracle_arm(
                    "layer 31 qkv_proj", res["vk"][0], res["cpu"][0], ref16,
                    route_and_device(counters_c),
                    {
                        "node": QKV_NODE,
                        "op_type": "com.microsoft::MatMulNBits",
                        "K": int(a["K"]),
                        "N": int(a["N"]),
                        "why_this_node": (
                            "outputs 63 and 64 are slices of this node's output. A copy "
                            "cannot manufacture a residual, so any in-situ divergence on "
                            "them was made here or upstream of here"
                        ),
                    },
                )
        except Exception as exc:  # noqa: BLE001
            rec["arm_c_isolated_qkv_proj31"] = {"status": "ERROR(instrument)", "error": repr(exc)}

    # -- Arm D: output 0, isolated lm_head, float64 reference -----------------------------
    if have_taps:
        try:
            _p, meta = make_isolated_node(scratch, LM_HEAD_NODE, "lm_head_only.onnx")
            a = meta["attrs"]
            h16 = np.asarray(taps[HIDDEN_TENSOR]).astype(np.float16)
            node_in = meta["inputs"]
            feed = {node_in[0]: h16}
            iso = finish_isolated_model(
                meta,
                {node_in[0]: (h16.shape, onnx.TensorProto.FLOAT16)},
                {meta["outputs"][0]: onnx.TensorProto.FLOAT16},
            )
            counters_d = scratch / "counters_armD.json"
            res = run_both_eps(iso, feed, counters_d)
            if "instrument_error" in res:
                rec["arm_d_isolated_lm_head"] = {"status": "ERROR(instrument)", **res}
            else:
                from onnx import numpy_helper

                w64 = dequantize_nbits(
                    numpy_helper.to_array(meta["_keep"][0]),
                    numpy_helper.to_array(meta["_keep"][1]),
                    n=int(a["N"]), k=int(a["K"]),
                    block_size=int(a["block_size"]), bits=int(a["bits"]),
                )
                ref16 = (w64 @ h16.reshape(-1).astype(np.float64)).astype(np.float16)
                rec["arm_d_isolated_lm_head"] = _oracle_arm(
                    "lm_head", res["vk"][0], res["cpu"][0], ref16,
                    route_and_device(counters_d),
                    {
                        "node": LM_HEAD_NODE,
                        "op_type": "com.microsoft::MatMulNBits",
                        "K": int(a["K"]),
                        "N": int(a["N"]),
                        "output_index_in_criterion10": 0,
                        "round_36_reading": "neither (equal)",
                        "why_re_measured": (
                            "round 36 measured this on a different build; a result quoted "
                            "across a rebuild is a claim about a binary nobody hashed"
                        ),
                    },
                )
        except Exception as exc:  # noqa: BLE001
            rec["arm_d_isolated_lm_head"] = {"status": "ERROR(instrument)", "error": repr(exc)}

    # -- Arm E: the inheritance identity. No device, no reference -- pure reconstruction. --
    #
    # Arms B and C say a residual on outputs 63/64 cannot have been MADE at the GQA node.
    # That is an argument. Arm E turns it into an identity: take each EP's OWN in-situ
    # packed-QKV tap, apply the position-0 map in float64, round, and check the result is
    # BIT-EQUAL to that EP's own in-situ present.31.{key,value}. If it is, then outputs 63
    # and 64 are a deterministic function of the QKV tensor on both sides, and their
    # divergence is exactly the divergence already present in that tensor -- inherited,
    # with no room left for a contribution of their own. If it is not, the reconstruction
    # is wrong and arms B/C must not be read as covering these outputs.
    #
    # This arm can only ever CONSTRAIN the reading of B and C. It cannot support a verdict,
    # because it never consults a reference: a perfect score here is compatible with both
    # EPs being wrong together.
    if have_taps and "present.31.key" in taps and "present.31.value" in taps:
        try:
            _p, meta_e = make_isolated_node(scratch, GQA_NODE, "_gqa_meta_only.onnx")
            ae = meta_e["attrs"]
            cos0 = np.asarray(taps["cos_cache"])[0].astype(np.float64)
            sin0 = np.asarray(taps["sin_cache"])[0].astype(np.float64)
            heads, kvh = int(ae["num_heads"]), int(ae["kv_num_heads"])
            sides = {}
            for tag, suffix in (("vulkan", "@vk"), ("cpu", "")):
                flat = np.asarray(taps[QKV_OUT + suffix]).reshape(-1).astype(np.float64)
                hd = flat.size // (heads + 2 * kvh)
                q_sz, kv_sz = heads * hd, kvh * hd
                k_raw = flat[q_sz : q_sz + kv_sz].reshape(kvh, hd)
                v_raw = flat[q_sz + kv_sz : q_sz + 2 * kv_sz].reshape(kvh, hd)
                k_rec = (
                    rope_reference_f64(
                        k_raw, cos0, sin0,
                        interleaved=bool(ae.get("rotary_interleaved", 0)),
                    )
                    if int(ae.get("do_rotary", 0))
                    else k_raw
                ).astype(np.float16).reshape(-1)
                v_rec = v_raw.astype(np.float16).reshape(-1)
                got_k = np.asarray(taps["present.31.key" + suffix]).reshape(-1).astype(np.float16)
                got_v = np.asarray(taps["present.31.value" + suffix]).reshape(-1).astype(np.float16)
                sides[tag] = {
                    "present.31.key_reconstructs_bit_exactly": bool(
                        np.array_equal(k_rec.view(np.uint16), got_k.view(np.uint16))
                    ),
                    "present.31.value_reconstructs_bit_exactly": bool(
                        np.array_equal(v_rec.view(np.uint16), got_v.view(np.uint16))
                    ),
                    "key_elements_mismatched": int(
                        np.count_nonzero(k_rec.view(np.uint16) != got_k.view(np.uint16))
                    ),
                    "value_elements_mismatched": int(
                        np.count_nonzero(v_rec.view(np.uint16) != got_v.view(np.uint16))
                    ),
                }
            everything_reconstructs = all(
                v for s in sides.values() for k, v in s.items() if k.endswith("bit_exactly")
            )
            rec["arm_e_inheritance_identity"] = {
                "status": "MEASURED",
                "class": "CONSTRAINT",
                "not_an_oracle_answer": (
                    "no reference is consulted here; a perfect score is compatible with "
                    "both EPs being wrong together, and this arm can only bound how arms "
                    "B and C may be read"
                ),
                "per_side": sides,
                "both_sides_reconstruct_bit_exactly": everything_reconstructs,
                "reading": (
                    "TRUE means outputs 63 and 64 are, on each EP, a deterministic "
                    "float64-reproducible function of that EP's own packed-QKV tensor. "
                    "Their in-situ divergence is then exactly the divergence already "
                    "present in that tensor -- INHERITED, with no locally-made component "
                    "left to attribute. FALSE means the reconstruction is wrong and arms "
                    "B and C must NOT be read as covering these outputs."
                    if everything_reconstructs
                    else "FALSE: the reconstruction does not hold; arms B and C must NOT "
                    "be read as covering outputs 63 and 64, and this needs a new probe"
                ),
            }
        except Exception as exc:  # noqa: BLE001
            rec["arm_e_inheritance_identity"] = {
                "status": "ERROR(instrument)",
                "error": repr(exc),
            }

    # -- The honest limit, stated with what it would take ---------------------------------
    rec["what_this_does_not_answer"] = {
        "the_in_situ_question": (
            "for the outputs AS CRITERION 10 SEES THEM, neither EP's inputs are the "
            "other's. Answering 'which side is wrong' there requires the true float64 "
            "value of each output for the model as a whole, which requires a float64 "
            "forward pass of the entire 355-node graph."
        ),
        "why_it_is_not_done_here": (
            "the reference would have to dequantise every MatMulNBits weight in the model "
            "to float64. Phi-3.5-mini is ~3.8e9 parameters; at 8 bytes each that is ~30 GB "
            "of dense float64 weights for a model whose int4 form is 2.3 GB. It is not a "
            "matter of patience on this box."
        ),
        "what_it_would_take": [
            "a float64 (or at minimum float32-dense) reference implementation of the "
            "whole graph -- GQA, RoPE, SkipSimplifiedLayerNormalization, MatMulNBits, "
            "and the MLP -- driven from the same feeds, dequantising layer by layer and "
            "discarding each layer's weights before the next, so peak resident float64 "
            "weight is one layer (~3 x 3072 x 9216 x 8 B ~= 0.7 GB) rather than the model",
            "acceptance that such a reference is itself an implementation with its own "
            "accumulation order, so it answers 'which side is further from THIS reference' "
            "unless the per-hop envelope is measured too -- which is what arms B/C/D do "
            "one node at a time",
            "a liveness bar on it at every layer, not only at the end: a whole-graph "
            "reference that drifts is indistinguishable from an EP that drifts",
        ],
        "what_is_answered_instead": (
            "the same question at each node that produces one of the three outputs, with "
            "identical inputs on both sides -- which is the only form of it that can be "
            "asked honestly at this scale, and is the form that locates a defect"
        ),
        "this_is_a_complete_result": (
            "a per-hop oracle answer plus a stated limit is a result; a model-scale answer "
            "built on a reference nobody could keep live would be a number with a "
            "provenance class of PREDICTION (docs/DESIGN.md §8.9.24(6))"
        ),
    }

    rec["not_done_here"] = [
        "atol and rtol are not moved, and §8.9.24(4) forbids the motion outright",
        "no verdict is split by mechanism (§8.9.24(2))",
        "no criterion-10 row is closed; the verdict stays DIVERGENT on both devices",
        "nothing that follows from this answer is proposed -- assert_record_proposes_no_"
        "motion enforces it on this artifact",
    ]
    rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    assert_record_proposes_no_motion(rec)

    text = json.dumps(rec, indent=1, sort_keys=True, default=str)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    for key in (
        "arm_a_in_situ",
        "arm_b_present31_key",
        "arm_b_present31_value",
        "arm_c_isolated_qkv_proj31",
        "arm_d_isolated_lm_head",
        "arm_e_inheritance_identity",
    ):
        a = rec.get(key)
        if not isinstance(a, dict):
            continue
        print(
            f"\n{key}: {a.get('status')}  attribution={a.get('attribution')}  "
            f"dispatch={a.get('dispatch_screen')}  device={a.get('device_name')}"
        )
        if "which_is_further_from_true" in a:
            print(
                f"   which_is_further_from_true = {a['which_is_further_from_true']}  "
                f"(vk {a.get('vulkan_value')} vs cpu {a.get('cpu_value')} "
                f"on {a.get('discriminator')})"
            )
            print(
                f"   unanimous_direction = {a.get('unanimous_direction')!r}  "
                f"conflict={a.get('discriminators_conflict')}  "
                f"silent={a.get('discriminators_silent')}"
            )
            for d, v in (a.get("verdict_by_discriminator") or {}).items():
                print(f"      {d:20s} vk={v['vulkan_value']!s:12s} cpu={v['cpu_value']!s:12s}"
                      f" -> {v['verdict']}")
        if key == "arm_e_inheritance_identity":
            print(f"   both_sides_reconstruct_bit_exactly = "
                  f"{a.get('both_sides_reconstruct_bit_exactly')}")
            for side, s in (a.get("per_side") or {}).items():
                print(f"      {side}: {s}")
        if key == "arm_a_in_situ":
            for lbl, st in (a.get("per_tensor") or {}).items():
                print(f"   [INHERITANCE] {lbl}: median {st['median_ulp_diff']} ULP, "
                      f"max_abs {st['max_abs_diff']}, differing {st['elements_differing']}"
                      f"/{st['elements']}")
    return 0


# ==========================================================================================
# Selftest -- no GPU, no model, no ORT session
# ==========================================================================================
def _selftest() -> int:  # noqa: PLR0915
    import _models as m

    # (1) the no-motion guard must be able to REFUSE, or it witnesses nothing
    clean = {"which_is_further_from_true": "cpu", "margin": 1.0, "nested": [{"ok": 1}]}
    assert_record_proposes_no_motion(clean)
    for bad in (
        {"proposed_atol": 0.01},
        {"a": {"b": [{"tolerance_budget": 3}]}},
        {"nested": {"would_pass_if": "atol=2e-3"}},
    ):
        try:
            assert_record_proposes_no_motion(bad)
        except MotionInRecordError as exc:
            assert "§8.9.24(4)" in str(exc), exc
        else:  # pragma: no cover
            raise AssertionError(f"the no-motion guard passed {bad}; it cannot go red")

    # (2) the RoPE reference must distinguish the THREE forms position 0 can take, because
    #     the first draft of this probe asserted the first one and the model turned out to
    #     be the second. Without arms for all three, "exact" would be a property of the
    #     function rather than a reading of the caches.
    k = np.arange(2 * 8, dtype=np.float64).reshape(2, 8)
    ident = rope_reference_f64(k, np.ones(4), np.zeros(4), interleaved=False)
    assert np.array_equal(ident, k), ident
    # PURE_SCALE: sin == 0 but cos != 1 -- the real Phi-3.5 case. Not a copy; still one
    # multiply, so still exactly reproducible.
    scale = 1.1904296875
    scaled = rope_reference_f64(k, np.full(4, scale), np.zeros(4), interleaved=False)
    assert np.array_equal(scaled, k * scale), scaled
    assert not np.array_equal(scaled, k), "a pure scale must not read as the identity"
    turned = rope_reference_f64(
        k, np.full(4, np.cos(0.7)), np.full(4, np.sin(0.7)), interleaved=False
    )
    assert not np.allclose(turned, k)
    # and the two conventions must differ, or `rotary_interleaved` is not being honoured
    a = rope_reference_f64(k, np.full(4, np.cos(0.7)), np.full(4, np.sin(0.7)), interleaved=False)
    b = rope_reference_f64(k, np.full(4, np.cos(0.7)), np.full(4, np.sin(0.7)), interleaved=True)
    assert not np.allclose(a, b), "the two rotary conventions compute the same thing"
    # A pure scale is convention-independent -- the cross terms vanish. If these two ever
    # disagreed, the position-0 reading would depend on an attribute it must not depend on.
    assert np.array_equal(
        rope_reference_f64(k, np.full(4, scale), np.zeros(4), interleaved=True), k * scale
    )

    # (3) `which_side` must distinguish all three answers, must not call two zeros a
    #     measurement without saying so, and must REPORT A CONFLICT rather than resolve it
    far_vk = which_side({"max_ulp_diff": 4.0}, {"max_ulp_diff": 1.0})
    far_cpu = which_side({"max_ulp_diff": 1.0}, {"max_ulp_diff": 4.0})
    equal = which_side(
        {"max_ulp_diff": 0.0, "bit_exact": True}, {"max_ulp_diff": 0.0, "bit_exact": True}
    )
    assert far_vk["which_is_further_from_true"] == "vulkan"
    assert far_cpu["which_is_further_from_true"] == "cpu"
    assert equal["which_is_further_from_true"] == "neither (equal)"
    assert equal["both_bit_exact_against_the_reference"] is True
    assert "null result" in equal["reading"]
    assert which_side({}, {})["which_is_further_from_true"] == "UNMEASURED"

    # (3a) the round-36 vs round-37 shape: the SAME pair, read on the median, says
    #      "neither (equal)"; read on the max, says "cpu". Neither reading is wrong and
    #      the record must carry both, or a change of statistic looks like a change of
    #      result. This is the specimen the multi-discriminator field exists for.
    vk_d = {"max_ulp_diff": 2.0, "median_ulp_diff": 0.0, "p99_ulp_diff": 0.0,
            "elements_differing": 11, "max_abs_diff": 0.000244140625}
    cpu_d = {"max_ulp_diff": 11.0, "median_ulp_diff": 0.0, "p99_ulp_diff": 0.0,
             "elements_differing": 47, "max_abs_diff": 0.00390625}
    split = which_side(vk_d, cpu_d)
    assert split["which_is_further_from_true"] == "cpu"
    assert split["verdict_by_discriminator"]["median_ulp_diff"]["verdict"] == "neither (equal)"
    assert split["verdict_by_discriminator"]["max_ulp_diff"]["verdict"] == "cpu"
    assert split["verdict_by_discriminator"]["elements_differing"]["verdict"] == "cpu"
    assert split["discriminators_conflict"] is False
    assert split["unanimous_direction"] == "cpu", split["unanimous_direction"]
    assert "median_ulp_diff" in split["discriminators_silent"]

    # (3b) a genuine conflict must NOT be resolved into a verdict. If one statistic says
    #      vulkan and another says cpu, the split is the finding, and `unanimous_direction`
    #      must be None so no row can be quoted as "which side is wrong".
    conflict = which_side(
        {"max_ulp_diff": 8.0, "median_ulp_diff": 0.0, "p99_ulp_diff": 0.0,
         "elements_differing": 2, "max_abs_diff": 0.5},
        {"max_ulp_diff": 2.0, "median_ulp_diff": 0.0, "p99_ulp_diff": 0.0,
         "elements_differing": 90, "max_abs_diff": 0.1},
    )
    assert conflict["discriminators_conflict"] is True
    assert conflict["unanimous_direction"] is None
    assert "conflict_note" in conflict
    # (3c) all-silent is not unanimity for either side
    allsilent = which_side(
        {"max_ulp_diff": 0.0, "median_ulp_diff": 0.0, "p99_ulp_diff": 0.0,
         "elements_differing": 0, "max_abs_diff": 0.0, "bit_exact": True},
        {"max_ulp_diff": 0.0, "median_ulp_diff": 0.0, "p99_ulp_diff": 0.0,
         "elements_differing": 0, "max_abs_diff": 0.0, "bit_exact": True},
    )
    assert allsilent["discriminators_conflict"] is False
    assert allsilent["unanimous_direction"].startswith("neither")
    assert len(allsilent["discriminators_silent"]) == len(DISCRIMINATORS)

    # (4) the dequantiser agrees with the other implementation of it in this tree
    from probe_logits_reduction import dequantize_nbits as other

    rng = np.random.default_rng(4)
    n, k_, bs = 6, 64, 32
    packed = rng.integers(0, 256, size=(n, k_ // bs, bs // 2), dtype=np.uint8)
    scales = rng.random((n, k_ // bs)).astype(np.float32) + 0.1
    mine = dequantize_nbits(packed, scales, n=n, k=k_, block_size=bs, bits=4)
    theirs = other(packed, scales, n=n, k=k_, block_size=bs, bits=4)
    assert np.array_equal(mine, theirs), "two dequantisers, two answers"

    # (5) an unattributed arm must report UNMEASURED, not a clean number.  This is the
    #     silent-CPU-rebuild shape: exit 0, zero dispatches, a perfect residual.
    ref = np.zeros(16, np.float16)
    dead = _oracle_arm(
        "unattributed",
        ref, ref, ref,
        {"attribution": "UNATTRIBUTED (declined)", "dispatch_screen": "ERROR(instrument=zero_ep_dispatches)"},
        {},
    )
    assert dead["status"] == "ERROR(instrument)", dead
    assert dead["which_is_further_from_true"] == "UNMEASURED", dead
    live_route = {"attribution": "ATTRIBUTED", "dispatch_screen": "PASS"}
    good = _oracle_arm("attributed", ref, ref, ref, live_route, {})
    assert good["status"] == "MEASURED", good

    # (6) a dead reference must void the arm even when the run is perfectly attributed
    bad_ref = np.full(16, 100.0, np.float16)
    voided = _oracle_arm("bad ref", ref, ref, bad_ref, live_route, {})
    assert voided["status"] == "ERROR(instrument)", voided
    assert voided["reference_liveness"]["live"] is False

    # (7) ulp_stats carries §8.9.24(3)'s companions, so nothing this probe emits can be
    #     quoted as a ULP-at-scale figure without its allowance
    st = ulp_stats(np.float16([1.0, 2.0]), np.float16([1.0, 2.5]))
    m.assert_ulp_at_scale_row_is_complete(st, where="selftest ulp_stats")
    assert st["ulp_element_basis_max"] is not None

    print(
        "SELFTEST PASS: 11 arms -- the no-motion guard refuses three shapes of motion, the "
        "rotary reference distinguishes IDENTITY from PURE_SCALE from ROTATION and the two "
        "conventions from each other (and shows a pure scale is convention-independent), "
        "which_side names all three answers, flags a two-zero null, reproduces the "
        "round-36/37 median-vs-max split as agreement rather than a changed result, and "
        "REFUSES to resolve a genuine discriminator conflict into a verdict; the "
        "dequantiser agrees with the tree's other one, an unattributed or dead-reference "
        "arm reports UNMEASURED, and every emitted row carries §8.9.24(3)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
