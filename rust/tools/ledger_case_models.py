#!/usr/bin/env python
"""Build the small ONNX artifacts the §8.9 proof ledger is bootstrapped from. Owner: Mouse.

These are *builders*, not committed models: the ledger records the builder that produced a case
(§8.9.2 rule 3 — "the artifact **or builder** the case came from"), and a builder that is in the
repository is reproducible in a way a binary blob in `bench/results/` is not.

Each function returns a model whose graph is one node of one form, because the ledger's unit is
the dispatchable form and a multi-node model would prove a graph rather than a form.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper


def _binary(op: str, elem: int, shape=(4, 8), opset: int = 17) -> onnx.ModelProto:
    a = helper.make_tensor_value_info("A", elem, list(shape))
    b = helper.make_tensor_value_info("B", elem, list(shape))
    c = helper.make_tensor_value_info("C", elem, list(shape))
    node = helper.make_node(op, ["A", "B"], ["C"], name=f"{op.lower()}0")
    graph = helper.make_graph([node], f"{op}_graph", [a, b], [c])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


def _unary(op: str, elem: int, shape=(4, 8), opset: int = 17) -> onnx.ModelProto:
    x = helper.make_tensor_value_info("X", elem, list(shape))
    y = helper.make_tensor_value_info("Y", elem, list(shape))
    node = helper.make_node(op, ["X"], ["Y"], name=f"{op.lower()}0")
    graph = helper.make_graph([node], f"{op}_graph", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


def _matmulnbits(
    with_zero_points: bool, K: int = 32, N: int = 32, bits: int = 4, block_size: int = 32
) -> onnx.ModelProto:
    """The pair §8.9 was written around.

    The all-zero-logits defect of 2026-07-30 was `MatMulNBits` **with** `zero_points` executed by
    a kernel proven **without** them. The two graphs below differ in exactly one thing — whether
    the optional fourth input is populated — which is why `populated_optional_input_set` is a
    component of the proof key. Their keys differ, so a proof of one can never be returned for
    the other, and the defect is unrepresentable rather than merely unlikely.

    They are also the form the whole Phi-3.5 claim rests on: MatMulNBits is the anchor op, and
    the test suite's own device probe uses it.
    """
    rng = np.random.default_rng(20260801)
    blocks_per_col = (K + block_size - 1) // block_size
    packed_bytes = block_size * bits // 8

    packed = rng.integers(0, 256, size=[N, blocks_per_col, packed_bytes], dtype=np.uint8)
    scale = rng.uniform(0.001, 0.1, size=[N * blocks_per_col]).astype(np.float16)

    inputs = ["X", "B", "scale"]
    inits = [
        onnx.numpy_helper.from_array(packed, name="B"),
        onnx.numpy_helper.from_array(scale, name="scale"),
    ]
    if with_zero_points:
        # One packed nibble-pair per block per column, per the contrib-op spec.
        zp_len = N * ((blocks_per_col + 1) // 2)
        zp = rng.integers(0, 256, size=[zp_len], dtype=np.uint8)
        inputs.append("zero_points")
        inits.append(onnx.numpy_helper.from_array(zp, name="zero_points"))

    node = helper.make_node(
        "MatMulNBits",
        inputs=inputs,
        outputs=["Y"],
        name="mmnb0",
        domain="com.microsoft",
        K=K,
        N=N,
        bits=bits,
        block_size=block_size,
        accuracy_level=1,
    )
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT16, [1, K])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [1, N])
    graph = helper.make_graph([node], "mmnb_graph", [x], [y], initializer=inits)
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
    )
    model.ir_version = 10
    return model



# ---------------------------------------------------------------------------
# 2026-08-02 — populating the ledger for the op suite.
#
# The 153 op-suite reds were Guard D refusing to report a vacuous CPU-vs-CPU pass on forms the
# gate declines for want of a proof. The remedy is proofs, not a softer guard. The forms were
# enumerated **mechanically** — one gated run of `pytest tests/ops` with
# `ONNXRUNTIME_EP_VULKAN_CLAIM_LOG` set, taking every record whose decline codes were exactly
# `{unproven}` — never from the claim table, because a ledger derived from the enumeration that
# produces the claims makes criterion 11 true by construction (Morpheus, RAI-008).
#
# Each builder below is still one node of one form. The list is longer; the unit did not change.
# ---------------------------------------------------------------------------

_DYN = "N"


def _unary_dyn(op: str, elem: int, opset: int = 21, **attrs) -> onnx.ModelProto:
    """A one-node unary whose shape is *symbolic*, so it classifies `runtime-extent`.

    `shape_class` is a component of the proof key, so a static proof is not a proof of the
    dynamic form and this builder is not a convenience — the key would differ.
    """
    x = helper.make_tensor_value_info("X", elem, [_DYN, 8])
    y = helper.make_tensor_value_info("Y", elem, [_DYN, 8])
    node = helper.make_node(op, ["X"], ["Y"], name=f"{op.lower()}0", **attrs)
    graph = helper.make_graph([node], f"{op}_dyn_graph", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


def _binary_dyn(op: str, elem: int, opset: int = 21) -> onnx.ModelProto:
    a = helper.make_tensor_value_info("A", elem, [_DYN, 8])
    b = helper.make_tensor_value_info("B", elem, [_DYN, 8])
    c = helper.make_tensor_value_info("C", elem, [_DYN, 8])
    node = helper.make_node(op, ["A", "B"], ["C"], name=f"{op.lower()}0")
    graph = helper.make_graph([node], f"{op}_dyn_graph", [a, b], [c])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


def _typed_binary(
    op: str, in_elem: int, out_elem: int, shape=(4, 8), opset: int = 21
) -> onnx.ModelProto:
    """A one-node binary whose **output dtype differs from its input dtype**.

    The comparison ops need this and `_binary` cannot express it: `Greater` takes floats and
    returns bools. The output dtype is a key component, so getting it wrong does not produce a
    slightly-off case — it produces a proof of a form the model does not contain.
    """
    a = helper.make_tensor_value_info("A", in_elem, list(shape))
    b = helper.make_tensor_value_info("B", in_elem, list(shape))
    c = helper.make_tensor_value_info("C", out_elem, list(shape))
    node = helper.make_node(op, ["A", "B"], ["C"], name=f"{op.lower()}0")
    graph = helper.make_graph([node], f"{op}_graph", [a, b], [c])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


def _typed_unary(
    op: str, in_elem: int, out_elem: int, shape=(4, 8), opset: int = 21
) -> onnx.ModelProto:
    """A one-node unary whose output dtype differs from its input dtype (`IsNaN`)."""
    x = helper.make_tensor_value_info("X", in_elem, list(shape))
    y = helper.make_tensor_value_info("Y", out_elem, list(shape))
    node = helper.make_node(op, ["X"], ["Y"], name=f"{op.lower()}0")
    graph = helper.make_graph([node], f"{op}_graph", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


def _variadic(op: str, elem: int, n: int = 3, shape=(4, 8), opset: int = 21) -> onnx.ModelProto:
    """`Sum` / `Mean` / `Max` / `Min` with **three** inputs.

    Three, not two. With two inputs a variadic op is indistinguishable from a binary one, so a
    two-input case would prove the binary path and leave the fold that makes these ops variadic
    completely unexercised while reporting `MATCH`.
    """
    names = [chr(ord("A") + i) for i in range(n)]
    ins = [helper.make_tensor_value_info(nm, elem, list(shape)) for nm in names]
    out = helper.make_tensor_value_info("Y", elem, list(shape))
    node = helper.make_node(op, names, ["Y"], name=f"{op.lower()}0")
    graph = helper.make_graph([node], f"{op}_graph", ins, [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


def _where(elem: int, shape=(4, 8), opset: int = 21) -> onnx.ModelProto:
    cond = helper.make_tensor_value_info("C", TensorProto.BOOL, list(shape))
    x = helper.make_tensor_value_info("X", elem, list(shape))
    y = helper.make_tensor_value_info("Y", elem, list(shape))
    out = helper.make_tensor_value_info("Z", elem, list(shape))
    node = helper.make_node("Where", ["C", "X", "Y"], ["Z"], name="where0")
    graph = helper.make_graph([node], "Where_graph", [cond, x, y], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


def _prelu(elem: int, shape=(4, 8), opset: int = 21) -> onnx.ModelProto:
    """`PRelu` with a **per-channel** slope, which is the form real graphs carry.

    A slope shaped like `X` would exercise the plain elementwise path and leave the broadcast
    the op exists for untested.
    """
    x = helper.make_tensor_value_info("X", elem, list(shape))
    slope = helper.make_tensor_value_info("slope", elem, [shape[-1]])
    y = helper.make_tensor_value_info("Y", elem, list(shape))
    node = helper.make_node("PRelu", ["X", "slope"], ["Y"], name="prelu0")
    graph = helper.make_graph([node], "PRelu_graph", [x, slope], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


def _rms_norm(elem: int, shape=(4, 8), opset: int = 23) -> onnx.ModelProto:
    """`RMSNormalization`, the standard-domain norm the mobius builder emits (opset 23)."""
    x = helper.make_tensor_value_info("X", elem, list(shape))
    scale = helper.make_tensor_value_info("scale", elem, [shape[-1]])
    y = helper.make_tensor_value_info("Y", elem, list(shape))
    node = helper.make_node("RMSNormalization", ["X", "scale"], ["Y"], name="rmsnorm0", axis=-1)
    graph = helper.make_graph([node], "RMSNormalization_graph", [x, scale], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


def _clip(elem: int, opset: int = 21, bounds: str = "min+max") -> onnx.ModelProto:
    """`Clip` with **whichever** optional inputs `bounds` names populated.

    The suite's own failing case declines with `[arity] Clip has 1 inputs; this handler takes
    exactly 3` — a *different* finding from `[unproven]`, and it is not fixed by a ledger entry.
    The three-input form is the one the claim log reports as unproven.

    The four `bounds` values are **four separate proof obligations**, not one with a knob.
    `populated_input_set` is a component of the proof key, so each populated set gets its own
    entry, and that is the point: an omitted bound compiles a *different* shader body through
    `EW_SELECTOR`, so an entry for `min+max` says nothing at all about `max`-only. `max` is the
    load-bearing one — it is the form with an omitted **interior** input, where a handler that
    counted inputs instead of reading names would bind `max`'s tensor to `min`'s slot and clamp
    from below by a number meant as a ceiling.
    """
    x = helper.make_tensor_value_info("X", elem, [4, 8])
    y = helper.make_tensor_value_info("Y", elem, [4, 8])
    np_dt = np.float16 if elem == TensorProto.FLOAT16 else np.float32
    inits, names = [], ["X"]
    if "min" in bounds:
        inits.append(onnx.numpy_helper.from_array(np.array(-0.5, dtype=np_dt), name="lo"))
        names.append("lo")
    elif "max" in bounds:
        # An omitted *interior* input is named by an empty string, not by a short input list.
        names.append("")
    if "max" in bounds:
        inits.append(onnx.numpy_helper.from_array(np.array(0.5, dtype=np_dt), name="hi"))
        names.append("hi")
    node = helper.make_node("Clip", names, ["Y"], name="clip0")
    graph = helper.make_graph([node], "clip_graph", [x], [y], initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


def _cast(in_elem: int, to_elem: int, shape=(4, 8), opset: int = 21) -> onnx.ModelProto:
    """`Cast` from one element type to another.

    Every (source, destination) pair is its own proof obligation: the proof key carries both
    dtypes, and the pair chooses a **different compiled module** — `ew_cast.comp` is the one
    template whose variant space is a cross product rather than a column. A proof of `f32 -> i32`
    describes a module that the `i32 -> f32` node never runs.
    """
    x = helper.make_tensor_value_info("X", in_elem, list(shape))
    y = helper.make_tensor_value_info("Y", to_elem, list(shape))
    node = helper.make_node("Cast", ["X"], ["Y"], name="cast0", to=int(to_elem))
    graph = helper.make_graph([node], "Cast_graph", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


def _cast_dyn(in_elem: int, to_elem: int, opset: int = 21) -> onnx.ModelProto:
    """`Cast` with a *symbolic* leading extent, so it classifies `runtime-extent`.

    `shape_class` is a component of the proof key, so the static `Cast` proof does not describe
    this form — the EP reaches it through a different specialisation and the key differs.
    """
    x = helper.make_tensor_value_info("X", in_elem, [_DYN, 8])
    y = helper.make_tensor_value_info("Y", to_elem, [_DYN, 8])
    node = helper.make_node("Cast", ["X"], ["Y"], name="cast0", to=int(to_elem))
    graph = helper.make_graph([node], "Cast_dyn_graph", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


def _isinf(elem: int, opset: int = 21, **attrs) -> onnx.ModelProto:
    """`IsInf`, whose two `detect_*` attributes select four different predicates.

    Each is its own case. The attributes are a **specialisation constant**, so they are part of
    the proof key, and an entry proving the default `(1, 1)` form describes a shader body that
    the `(1, 0)` form does not run. Proving only the default would leave the selector itself —
    the whole mechanism — unexercised while the ledger read full.
    """
    x = helper.make_tensor_value_info("X", elem, [4, 8])
    y = helper.make_tensor_value_info("Y", TensorProto.BOOL, [4, 8])
    node = helper.make_node("IsInf", ["X"], ["Y"], name="isinf0", **attrs)
    graph = helper.make_graph([node], "IsInf_graph", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


def _matmulnbits_typed(
    elem: int, with_zero_points: bool, dynamic: bool = False,
    K: int = 32, N: int = 32, bits: int = 4, block_size: int = 32,
) -> onnx.ModelProto:
    """`MatMulNBits` at either activation dtype, static or runtime-extent.

    Generalises `_matmulnbits`, which stays as it is because the f16 static pair is the pair
    §8.9 was written around and renaming it would break the ledger entries that name it.
    """
    rng = np.random.default_rng(20260802)
    np_dt = np.float16 if elem == TensorProto.FLOAT16 else np.float32
    blocks_per_col = (K + block_size - 1) // block_size
    packed_bytes = block_size * bits // 8

    packed = rng.integers(0, 256, size=[N, blocks_per_col, packed_bytes], dtype=np.uint8)
    scale = rng.uniform(0.001, 0.1, size=[N * blocks_per_col]).astype(np_dt)

    inputs = ["X", "B", "scale"]
    inits = [
        onnx.numpy_helper.from_array(packed, name="B"),
        onnx.numpy_helper.from_array(scale, name="scale"),
    ]
    if with_zero_points:
        zp_len = N * ((blocks_per_col + 1) // 2)
        inputs.append("zero_points")
        inits.append(
            onnx.numpy_helper.from_array(
                rng.integers(0, 256, size=[zp_len], dtype=np.uint8), name="zero_points"
            )
        )

    node = helper.make_node(
        "MatMulNBits", inputs=inputs, outputs=["Y"], name="mmnb0", domain="com.microsoft",
        K=K, N=N, bits=bits, block_size=block_size, accuracy_level=1,
    )
    lead = _DYN if dynamic else 1
    x = helper.make_tensor_value_info("X", elem, [lead, K])
    y = helper.make_tensor_value_info("Y", elem, [lead, N])
    graph = helper.make_graph([node], "mmnb_graph", [x], [y], initializer=inits)
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
    )
    model.ir_version = 10
    return model


def _gather_dyn(elem: int) -> onnx.ModelProto:
    data = helper.make_tensor_value_info("data", elem, [_DYN, 8])
    idx = helper.make_tensor_value_info("indices", TensorProto.INT64, [2])
    out = helper.make_tensor_value_info("Y", elem, [2, 8])
    node = helper.make_node("Gather", ["data", "indices"], ["Y"], name="gather0", axis=0)
    graph = helper.make_graph([node], "gather_graph", [data, idx], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


def _simplified_layer_norm(elem: int) -> onnx.ModelProto:
    h = 32
    x = helper.make_tensor_value_info("X", elem, [_DYN, h])
    scale = helper.make_tensor_value_info("scale", elem, [h])
    y = helper.make_tensor_value_info("Y", elem, [_DYN, h])
    node = helper.make_node(
        "SimplifiedLayerNormalization", ["X", "scale"], ["Y"],
        name="sln0", axis=-1, epsilon=1e-5, stash_type=1,
    )
    graph = helper.make_graph([node], "sln_graph", [x, scale], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    model.ir_version = 10
    return model


def _skip_simplified_layer_norm(
    elem: int, with_extra_outputs: bool, static: bool = False
) -> onnx.ModelProto:
    """The output arities and shape classes the claim log reports as distinct forms.

    `f16,f16,f16>f16` and `f16,f16,f16>f16,-,-,f16` differ only in whether slot 3 (the
    input-skip sum) is requested. That is a **path** difference under §8.7 — a different set of
    bindings is written — so the keys differ and one is not a proof of the other. Slot 3 is
    exactly where the 2026-07-31 residual defect lived, which is why both are built.

    `static` selects a concrete leading dim rather than a symbolic one, which moves
    `shape_class` from `runtime-extent` to `static` — a fourth axis of the same key. The
    op-suite's SkipSLN tests all use concrete shapes; Phi-3.5 uses symbolic ones. Proving one
    has never been a proof of the other and the residual triage of 2026-08-02 showed exactly
    that: the runtime-extent forms were proven and the static forms still declined.
    """
    h = 32
    lead = 4 if static else _DYN
    x = helper.make_tensor_value_info("X", elem, [lead, h])
    skip = helper.make_tensor_value_info("skip", elem, [lead, h])
    gamma = helper.make_tensor_value_info("gamma", elem, [h])
    y = helper.make_tensor_value_info("Y", elem, [lead, h])
    outs = ["Y"]
    vis = [y]
    if with_extra_outputs:
        # Slots 1 and 2 stay unnamed — "" is how ONNX spells an unrequested optional output,
        # and it is what makes the dtype signature read `f16,f16,f16>f16,-,-,f16`.
        outs = ["Y", "", "", "S"]
        vis.append(helper.make_tensor_value_info("S", elem, [lead, h]))
    node = helper.make_node(
        "SkipSimplifiedLayerNormalization", ["X", "skip", "gamma"], outs,
        name="ssln0", domain="com.microsoft", epsilon=1e-5,
    )
    graph = helper.make_graph([node], "ssln_graph", [x, skip, gamma], vis)
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
    )
    model.ir_version = 10
    return model

_GQA_NUM_HEADS = 8
_GQA_KV_HEADS = 2
_GQA_HEAD_DIM = 32
_GQA_ROTARY_DIM = 16
_GQA_MAX_SEQ = 64
_GQA_PAST_SEQ = 4


def _group_query_attention() -> onnx.ModelProto:
    """`GroupQueryAttention` in the one form the kernel implements.

    Packed QKV (inputs 1 and 2 empty), `do_rotary=1`, `rotary_interleaved=0`,
    `local_window_size=-1`, `softcap=0` — every one of those is a positive requirement in
    `ops/attention.rs`, so any other combination declines on `[attribute]` and the ledger has
    no say in it. The batch and sequence extents stay symbolic, which is what puts this form in
    the `runtime-extent` shape class rather than `static`; the concrete values are bound at run
    time by the feed plan, because `seqlens_k` and `total_sequence_length` are indices into the
    KV cache and not free variables.

    The recipe is the one `tests/ops/test_gqa.py` already exercises. It is duplicated rather
    than imported because a case model that changes when a test file changes is a case model
    whose ledger entries silently stop describing what was proven.
    """
    packed = (_GQA_NUM_HEADS + 2 * _GQA_KV_HEADS) * _GQA_HEAD_DIM
    f16 = TensorProto.FLOAT16
    past_kv = ["B", _GQA_KV_HEADS, _GQA_PAST_SEQ, _GQA_HEAD_DIM]
    ins = [
        helper.make_tensor_value_info("packed_qkv", f16, ["B", "S", packed]),
        helper.make_tensor_value_info("past_key", f16, past_kv),
        helper.make_tensor_value_info("past_value", f16, past_kv),
        helper.make_tensor_value_info("seqlens_k", TensorProto.INT32, ["B"]),
        helper.make_tensor_value_info("total_seq", TensorProto.INT32, []),
        helper.make_tensor_value_info(
            "cos_cache", f16, [_GQA_MAX_SEQ, _GQA_ROTARY_DIM // 2]
        ),
        helper.make_tensor_value_info(
            "sin_cache", f16, [_GQA_MAX_SEQ, _GQA_ROTARY_DIM // 2]
        ),
    ]
    pres_kv = ["B", _GQA_KV_HEADS, _GQA_PAST_SEQ + 1, _GQA_HEAD_DIM]
    outs = [
        helper.make_tensor_value_info(
            "attn_out", f16, ["B", "S", _GQA_NUM_HEADS * _GQA_HEAD_DIM]
        ),
        helper.make_tensor_value_info("present_key", f16, pres_kv),
        helper.make_tensor_value_info("present_value", f16, pres_kv),
    ]
    node = helper.make_node(
        "GroupQueryAttention",
        inputs=[
            "packed_qkv", "", "", "past_key", "past_value", "seqlens_k", "total_seq",
            "cos_cache", "sin_cache",
        ],
        outputs=["attn_out", "present_key", "present_value"],
        domain="com.microsoft",
        name="gqa0",
        num_heads=_GQA_NUM_HEADS,
        kv_num_heads=_GQA_KV_HEADS,
        scale=float(_GQA_HEAD_DIM ** -0.5),
        local_window_size=-1,
        do_rotary=1,
        rotary_interleaved=0,
        smooth_softmax=0,
    )
    graph = helper.make_graph([node], "gqa_graph", ins, outs)
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)],
    )
    model.ir_version = 10
    return model


BUILDERS = {
    "add_f32": lambda: _binary("Add", TensorProto.FLOAT),
    "group_query_attention_f16": _group_query_attention,
    "matmulnbits_f16_scales": lambda: _matmulnbits(with_zero_points=False),
    "matmulnbits_f16_scales_zp": lambda: _matmulnbits(with_zero_points=True),
    "add_f16": lambda: _binary("Add", TensorProto.FLOAT16),
    "mul_f32": lambda: _binary("Mul", TensorProto.FLOAT),
    "sub_f32": lambda: _binary("Sub", TensorProto.FLOAT),
    "div_f32": lambda: _binary("Div", TensorProto.FLOAT),
    "relu_f32": lambda: _unary("Relu", TensorProto.FLOAT),
    "sqrt_f32": lambda: _unary("Sqrt", TensorProto.FLOAT),
    # ------------------------------------------------------------------
    # THE PLANTED CONTROL — deliberately never proven.
    #
    # RAI-008(a) requires criterion 11 be verified by a planted `[unproven]` decline, not by
    # reading the gate's source. This is a form the EP has a kernel for and the ledger has no
    # entry for. If it is ever claimed, the gate is not gating.
    #
    # IT MOVED ON 2026-08-02, AND WHY IT MOVED IS THE POINT.
    # ------------------------------------------------------
    # The control used to be `Mul` at f16, static. Populating the ledger for the op suite added a
    # `mul_f16` case — the *same form* — and the control fired on its own author, in the lane,
    # with "the gate is not gating". It was right, and it was the only thing that noticed.
    #
    # The lesson is not "never touch the control". A form kept permanently unprovable to protect
    # one file name is real coverage traded for a name, and the suite needs `Mul` at f16. The
    # lesson is that *nothing cheap* was checking the invariant, so it had to be discovered by a
    # three-minute lane test after a forty-minute generation run. `PLANTED_CONTROL_KEY` below
    # closes that: the generator now refuses to write this key and `--check` fails if it appears,
    # so the collision is caught in the tool that would cause it.
    #
    # The pair is now `Sub` at f16, static (PROVEN) against `Sub` at f16, runtime-extent
    # (NEVER PROVEN). They differ in exactly one key component — `shape_class` — which makes the
    # arms a sharper control than the old dtype pair: it shows the gate discriminating on the
    # component that separates decode from prefill, where §8.7 says a path difference lives.
    # ------------------------------------------------------------------
    "sub_f16_dyn_unproven": lambda: _binary_dyn("Sub", TensorProto.FLOAT16),
}



# ---------------------------------------------------------------------------
# The op-suite forms, enumerated from a gated `pytest tests/ops` run (2026-08-02).
#
# Order is deliberate: unary f32 first because that is where the reds are concentrated.
# ---------------------------------------------------------------------------

# (case name, op type) for the plain one-in one-out activations at f32.
_UNARY_F32 = [
    "Abs", "Neg", "Sign", "Floor", "Ceil", "Round", "Exp", "Erf", "Sigmoid", "Tanh",
    "HardSigmoid", "HardSwish", "LeakyRelu", "Elu", "Selu", "Celu", "Softplus", "Softsign",
    "ThresholdedRelu", "Gelu", "Mish", "Identity",
    "Sin", "Cos", "Tan", "Sinh", "Cosh", "Atan", "Asinh",
    # Partial functions — see INPUT_DOMAIN. They are in the list, not excluded: an op whose
    # domain must be constrained is still an op the suite tests.
    "Reciprocal", "Log", "Asin", "Acos", "Acosh", "Atanh",
]

_UNARY_F16 = ["Relu", "Sigmoid", "Sqrt", "Exp", "Tanh", "Gelu", "Erf"]
_BINARY_F16 = ["Mul", "Sub", "Div"]

_OP_CASE = {op: f"{op.lower()}_f32" for op in _UNARY_F32}

for _op in _UNARY_F32:
    BUILDERS[f"{_op.lower()}_f32"] = (lambda o=_op: _unary(o, TensorProto.FLOAT, opset=21))
for _op in _UNARY_F16:
    BUILDERS[f"{_op.lower()}_f16"] = (lambda o=_op: _unary(o, TensorProto.FLOAT16, opset=21))
for _op in _BINARY_F16:
    BUILDERS[f"{_op.lower()}_f16"] = (lambda o=_op: _binary(o, TensorProto.FLOAT16, opset=21))

BUILDERS.update({
    # runtime-extent forms. A static proof is not a proof of these: `shape_class` is a key
    # component, and it is the component that separates decode from prefill.
    "mul_f16_dyn": lambda: _binary_dyn("Mul", TensorProto.FLOAT16),
    "sigmoid_f16_dyn": lambda: _unary_dyn("Sigmoid", TensorProto.FLOAT16),
    "exp_f32_dyn": lambda: _unary_dyn("Exp", TensorProto.FLOAT),
    "log_f32_dyn": lambda: _unary_dyn("Log", TensorProto.FLOAT),
    "sigmoid_f32_dyn": lambda: _unary_dyn("Sigmoid", TensorProto.FLOAT),
    "tanh_f32_dyn": lambda: _unary_dyn("Tanh", TensorProto.FLOAT),

    "clip_f32": lambda: _clip(TensorProto.FLOAT),
    # The three other populated-input sets. Each is a distinct proof key and a distinct compiled
    # body; see `_clip`'s docstring for why `max`-only is the load-bearing one.
    "clip_f32_min_only": lambda: _clip(TensorProto.FLOAT, bounds="min"),
    "clip_f32_max_only": lambda: _clip(TensorProto.FLOAT, bounds="max"),
    "clip_f32_no_bounds": lambda: _clip(TensorProto.FLOAT, bounds="none"),
    # The four IsInf predicates. `detect_*` are specialisation constants and part of the key.
    "isinf_f32": lambda: _isinf(TensorProto.FLOAT),
    "isinf_f32_pos_only": lambda: _isinf(TensorProto.FLOAT, detect_negative=0),
    "isinf_f32_neg_only": lambda: _isinf(TensorProto.FLOAT, detect_positive=0),
    # The three Cast pairs the op suite exercises. Each is a different compiled module.
    "cast_f32_to_i32": lambda: _cast(TensorProto.FLOAT, TensorProto.INT32),
    "cast_i32_to_f32": lambda: _cast(TensorProto.INT32, TensorProto.FLOAT),
    "cast_f32_to_bool": lambda: _cast(TensorProto.FLOAT, TensorProto.BOOL),
    # `i64 -> i32` is the pair a real graph carries and the op suite did not: Phi-3.5 casts its
    # position/attention indices down, and the EP claimed the form in both shape classes with no
    # entry describing either. Both are listed because `shape_class` is part of the key.
    #
    # STAGED, NOT YET PROVABLE. Measured 2026-08-03: the decline that binds on these two is
    # `[dtype]`, not `[unproven]` — `ew_cast_i64_to_i32.spv` is generated but declares
    # `OpCapability Int64`, and `ENGINE_ENABLED_CAPABILITIES` does not carry it, so no pipeline can
    # be created and `gen_proof_ledger.py` correctly reports `no unlockable keys`. These cases are
    # the proof the day the three edits named in `ops/common/variants.rs` land (enable the feature
    # in the chain, probe it in `vk::caps`, decline the variant on devices that lack it); until
    # then they cost two files and answer a question that otherwise takes a second build.
    "cast_i64_to_i32": lambda: _cast(TensorProto.INT64, TensorProto.INT32),
    "cast_i64_to_i32_dyn": lambda: _cast_dyn(TensorProto.INT64, TensorProto.INT32),
    # ------------------------------------------------------------------
    # THE gpt-oss-20b FORMS (census 2026-08-03, `probe_model_op_census.py`).
    #
    # Selected from a graph, not from a list. On gpt-oss-20b the EP claimed **1 of 374 nodes**,
    # and 292 of the 370 declines were `[unproven]` — not missing kernels. Every form below is
    # a `runtime-extent` sibling of a form the ledger already proves `static`, so the shader,
    # the claim predicate and the translate handler all exist and the only thing missing is the
    # measurement. `shape_class` is a key component, which is correct and is exactly why the
    # gap was invisible: the ledger read full while a whole model declined.
    #
    # The pairing rule this taught, applied below: when a module is proven in one shape class,
    # prove the other one in the same run. The marginal cost is one comparison; the cost of not
    # doing it is a model.
    # ------------------------------------------------------------------
    # 49 nodes each on gpt-oss-20b's MoE router path (f16 activations -> f32 router -> f16).
    "cast_f16_to_f32": lambda: _cast(TensorProto.FLOAT16, TensorProto.FLOAT),
    "cast_f16_to_f32_dyn": lambda: _cast_dyn(TensorProto.FLOAT16, TensorProto.FLOAT),
    "cast_f32_to_f16": lambda: _cast(TensorProto.FLOAT, TensorProto.FLOAT16),
    "cast_f32_to_f16_dyn": lambda: _cast_dyn(TensorProto.FLOAT, TensorProto.FLOAT16),
    # 72 nodes. `Add` at f16 is proven static; every gpt-oss-20b `Add` is symbolic.
    "add_f16_dyn": lambda: _binary_dyn("Add", TensorProto.FLOAT16),
    # 73 nodes — the anchor op, in the one form gpt-oss-20b carries: zero points AND symbolic
    # extents. The ledger holds `scales`/runtime-extent and `scales+zero_points`/static; the
    # intersection of the two axes was never measured, and it is the whole model.
    "matmulnbits_f16_scales_zp_dyn": lambda: _matmulnbits_typed(
        TensorProto.FLOAT16, True, dynamic=True
    ),
    # 1 node.
    "simplified_layer_norm_f32": lambda: _simplified_layer_norm(TensorProto.FLOAT),
    # 47 + 1 nodes. The f32 pair is proven `static` only; gpt-oss-20b is symbolic throughout.
    "skip_simplified_layer_norm_f32": lambda: _skip_simplified_layer_norm(
        TensorProto.FLOAT, with_extra_outputs=False
    ),
    "skip_simplified_layer_norm_f32_slot3": lambda: _skip_simplified_layer_norm(
        TensorProto.FLOAT, with_extra_outputs=True
    ),
    # Pow is a partial function in its *first* argument: a negative base with a non-integral
    # exponent has no real value, and standard_normal supplies both. Sampled positive.
    "pow_f32": lambda: _binary("Pow", TensorProto.FLOAT, opset=21),

    "matmulnbits_f32_scales": lambda: _matmulnbits_typed(TensorProto.FLOAT, False),
    "matmulnbits_f32_scales_zp": lambda: _matmulnbits_typed(TensorProto.FLOAT, True),
    "matmulnbits_f16_scales_dyn": lambda: _matmulnbits_typed(
        TensorProto.FLOAT16, False, dynamic=True
    ),

    "gather_f16_dyn": lambda: _gather_dyn(TensorProto.FLOAT16),
    "simplified_layer_norm_f16": lambda: _simplified_layer_norm(TensorProto.FLOAT16),
    "skip_simplified_layer_norm_f16": lambda: _skip_simplified_layer_norm(
        TensorProto.FLOAT16, with_extra_outputs=False
    ),
    "skip_simplified_layer_norm_f16_slot3": lambda: _skip_simplified_layer_norm(
        TensorProto.FLOAT16, with_extra_outputs=True
    ),
    # The four static-shape / f32 SkipSLN forms the op-suite actually exercises. The
    # runtime-extent f16 pair above does not prove any of them: `shape_class` and the dtype
    # tuple are both key components, so these are four separate obligations.
    "skip_simplified_layer_norm_f16_static": lambda: _skip_simplified_layer_norm(
        TensorProto.FLOAT16, with_extra_outputs=False, static=True
    ),
    "skip_simplified_layer_norm_f16_static_slot3": lambda: _skip_simplified_layer_norm(
        TensorProto.FLOAT16, with_extra_outputs=True, static=True
    ),
    "skip_simplified_layer_norm_f32_static": lambda: _skip_simplified_layer_norm(
        TensorProto.FLOAT, with_extra_outputs=False, static=True
    ),
    "skip_simplified_layer_norm_f32_static_slot3": lambda: _skip_simplified_layer_norm(
        TensorProto.FLOAT, with_extra_outputs=True, static=True
    ),
})

# ---------------------------------------------------------------------------
# The 22 `Staged(UNEXERCISED)` rows (census: `epctl --dump-capabilities`, 2026-08-02).
#
# `UNEXERCISED` means the shader compiles and nothing has ever run it. That is the one staging
# reason the proof machinery can discharge on its own: the other four — 13 XL kernels whose
# shaders are still being written, 3 `NEEDS_PARAMS`, 2 `NEEDS_CAST_MATRIX`, 1 `NO_SHADER` — are
# missing code, not missing evidence, and no proof run can conjure a kernel.
#
# Half of these return **bool**, which is why the degeneracy guard in `gen_proof_ledger.py`
# had to land first. `Equal` on two independent normals is all-False; `IsNaN` on a finite
# tensor is all-False. Both would have reported `MATCH  worst_rel 0.0` and proven nothing.
# ---------------------------------------------------------------------------

_BOOL_BINARY = ["And", "Or", "Xor"]                     # bool  , bool   -> bool
_BITWISE_BINARY = ["BitwiseAnd", "BitwiseOr", "BitwiseXor"]   # i32, i32 -> i32
_COMPARE = ["Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual"]  # f32,f32 -> bool
_VARIADIC_F32 = ["Sum", "Mean", "Max", "Min"]

for _op in _BOOL_BINARY:
    BUILDERS[f"{_op.lower()}_bool"] = (
        lambda o=_op: _typed_binary(o, TensorProto.BOOL, TensorProto.BOOL)
    )
for _op in _BITWISE_BINARY:
    BUILDERS[f"{_op.lower()}_i32"] = (
        lambda o=_op: _typed_binary(o, TensorProto.INT32, TensorProto.INT32, opset=18)
    )
for _op in _COMPARE:
    BUILDERS[f"{_op.lower()}_f32"] = (
        lambda o=_op: _typed_binary(o, TensorProto.FLOAT, TensorProto.BOOL)
    )
for _op in _VARIADIC_F32:
    BUILDERS[f"{_op.lower()}_f32_v3"] = (lambda o=_op: _variadic(o, TensorProto.FLOAT))
    # The two-input form, which is a *different key* — arity rides the last key component, so
    # `.../n2` can never be returned for an `.../n3` node. That is what makes proving the pair
    # safe rather than a shortcut: the 3-input lowering is genuinely not written (the translate
    # handler says so), and the key prevents the 2-input proof from covering for it.
    BUILDERS[f"{_op.lower()}_f32_v2"] = (lambda o=_op: _variadic(o, TensorProto.FLOAT, n=2))

BUILDERS.update({
    # §8.9.16 — `Add`/i32 and `Mul`/i32 were unreachable, not unproven. `elementwise::EXERCISED`
    # vetoed them inside the claim predicate, which runs before a proof key is computed, so the
    # generator saw no unlockable key and the forms stayed unproven because they were unproven.
    # With the veto replaced by a loadability test they are ordinary `[unproven]` declines, which
    # is a decline a proof run can clear. Both go through the same `ew_binary` template as the
    # f32 rows and the same `_typed_binary` builder as the bitwise i32 cases already here; what
    # differs is the one-line body, which is exactly what the ledger key does not let them share.
    "add_i32": lambda: _typed_binary("Add", TensorProto.INT32, TensorProto.INT32, opset=18),
    "mul_i32": lambda: _typed_binary("Mul", TensorProto.INT32, TensorProto.INT32, opset=18),
    "not_bool": lambda: _typed_unary("Not", TensorProto.BOOL, TensorProto.BOOL, opset=21),
    "bitwisenot_i32": lambda: _typed_unary(
        "BitwiseNot", TensorProto.INT32, TensorProto.INT32, opset=18
    ),
    "isnan_f32": lambda: _typed_unary("IsNaN", TensorProto.FLOAT, TensorProto.BOOL),
    "where_f32": lambda: _where(TensorProto.FLOAT),
    "prelu_f32": lambda: _prelu(TensorProto.FLOAT),
    "rmsnormalization_f32": lambda: _rms_norm(TensorProto.FLOAT),
    # `Swish` is ai.onnx **opset 24**, and the row's window is opset 24 exactly. Whether this
    # environment can even build it is the case's first question, not an assumption.
    "swish_f32": lambda: _unary("Swish", TensorProto.FLOAT, opset=24),
})

# The domain each case must sample from.
#
# An op that is a partial function is not exercised by an input distribution that leaves its
# domain: `Sqrt` fed `standard_normal` returns NaN for about half its elements on every EP, so
# the comparison is NaN-against-NaN over half the tensor and proves nothing there. Naming the
# constraint here keeps it next to the case that needs it, rather than in the runner where the
# next case would have to rediscover it.
#
# `any`      — unconstrained standard normal.
# `positive` — strictly positive (Sqrt, Log).
# `nonzero`  — bounded away from zero (divisors).
INPUT_DOMAIN: dict[str, str] = {
    "sqrt_f32": "positive",
    "div_f32": "nonzero",
}


# Added 2026-08-02 with the op-suite cases. Every one of these is an op that is a *partial
# function*: fed `standard_normal` it returns NaN or Inf on a large fraction of elements, on
# both EPs, so the comparison would be NaN-against-NaN and would prove nothing there while
# looking like a full case. Naming the constraint is not making the test easier — an op tested
# outside its domain is not tested.
INPUT_DOMAIN.update({
    "log_f32": "positive",
    "log_f32_dyn": "positive",
    "sqrt_f16": "positive",
    "reciprocal_f32": "nonzero",
    "div_f16": "nonzero",
    # |x| <= 1 for the inverse circular functions; Atanh additionally diverges at |x| = 1.
    "asin_f32": "unit",
    "acos_f32": "unit",
    "atanh_f32": "unit",
    # Acosh is real only for x >= 1.
    "acosh_f32": "ge_one",
    "pow_f32": "positive",
})

# The 22 UNEXERCISED cases. Two of the three constraints here exist because the *reference*
# would otherwise be constant, which is a vacuous comparison rather than a lenient one.
INPUT_DOMAIN.update({
    # Full-width integers. The default `[0, 2)` exercises one bit of thirty-two, and a shader
    # that got the other thirty-one wrong would pass.
    "bitwiseand_i32": "bits",
    "bitwiseor_i32": "bits",
    "bitwisexor_i32": "bits",
    "bitwisenot_i32": "bits",
    # `Equal` on two independent normals is all-False: a constant reference.
    "equal_f32": "discrete",
    # `IsNaN` on a finite tensor is all-False, likewise.
    "isnan_f32": "withnan",
})

INPUT_DOMAIN.update({
    # `IsInf` needs both infinities *and* a NaN in the input, on every one of its three cases.
    # `withnan` is not enough: it plants `+inf` but no `-inf`, so a shader that ignored
    # `detect_negative` entirely would return all-False on the negative-only case and match a
    # reference that was also all-False. The domain that separates the four predicates has to
    # contain a value each of them decides differently.
    "isinf_f32": "withinf",
    "isinf_f32_pos_only": "withinf",
    "isinf_f32_neg_only": "withinf",
    # `Cast` to an integer truncates. A standard normal truncates to **zero** for ~68% of its
    # elements and to +/-1 for most of the rest, which is very close to a constant reference: a
    # kernel that returned zero unconditionally would score a near-match. `spread` widens the
    # draw so the truncation is actually doing something.
    "cast_f32_to_i32": "spread",
    "cast_i32_to_f32": "spread",
    # `Cast` to bool on a continuous distribution is all-True, a constant reference and exactly
    # the vacuous case `isnan_f32` gets `withnan` for. `discrete` is integer-valued and includes
    # zero, so both polarities of `x != 0` appear.
    "cast_f32_to_bool": "discrete",
    # `i64 -> i32` is a *narrowing*: the interesting failure is a kernel that reads the wrong half
    # of the 64-bit word. The default `[0, 2)` draw leaves the high word zero, where a wrong-half
    # read is invisible. `spread` includes negatives, whose high word is `0xFFFFFFFF`, so reading
    # the wrong half yields -1 and the mismatch shows. It stays inside i32, so nothing here turns
    # on out-of-range Cast, which ONNX leaves undefined.
    "cast_i64_to_i32": "spread",
    "cast_i64_to_i32_dyn": "spread",
})

def input_domain(name: str) -> str:
    """The sampling domain for a case, defaulting to unconstrained."""
    return INPUT_DOMAIN.get(name, "any")


# ---------------------------------------------------------------------------
# Feed plans — inputs that are not free variables.
#
# `input_domain` constrains a *distribution*. A feed plan pins an actual value, for inputs
# where no distribution is valid. `GroupQueryAttention` has three: `seqlens_k` and
# `total_sequence_length` are indices into the KV cache whose only correct values are
# determined by the cache extent, and the rotary caches are a table the kernel indexes by
# position. A random int32 in `total_sequence_length` is not a harsher test; it is an invalid
# model, and the CPU arm raises rather than producing an oracle — ERROR(instrument), not a
# verdict.
#
# `symbolic_dims` pins the symbolic extents so that the pinned indices and the tensor shapes
# agree. The dims stay symbolic *in the model*, which is what keeps `shape_class` at
# `runtime-extent`; only the run binds them.
# ---------------------------------------------------------------------------

def _rope_cache(max_seq: int, rot_dim: int) -> tuple[list, list]:
    import math

    cos, sin = [], []
    for p in range(max_seq):
        for i in range(0, rot_dim, 2):
            a = p / (10000 ** (i / rot_dim))
            cos.append(math.cos(a))
            sin.append(math.sin(a))
    return cos, sin


def _gqa_feed_plan() -> dict:
    cos, sin = _rope_cache(_GQA_MAX_SEQ, _GQA_ROTARY_DIM)
    half = [_GQA_MAX_SEQ, _GQA_ROTARY_DIM // 2]
    return {
        # B=1, S=1: the decode step, which is the shape every Phi-3.5 GQA node runs at after
        # prefill. `past` is the concrete cache extent baked into the model.
        "symbolic_dims": {"B": 1, "S": 1},
        # f16 attention over standard normals saturates the exponent range before the softmax
        # and both arms then agree on inf, which is a vacuous match. 0.1 keeps the logits in
        # the range the kernel actually sees.
        "float_scale": 0.1,
        "fixed_inputs": {
            "seqlens_k": {"dtype": "int32", "shape": [1], "values": [_GQA_PAST_SEQ]},
            "total_seq": {"dtype": "int32", "shape": [], "values": _GQA_PAST_SEQ + 1},
            "cos_cache": {"dtype": "float16", "shape": half, "values": cos},
            "sin_cache": {"dtype": "float16", "shape": half, "values": sin},
        },
    }


FEED_PLAN = {
    "group_query_attention_f16": _gqa_feed_plan,
}


def feed_plan(name: str) -> dict:
    """The pinned-value plan for a case, empty when every input is a free variable."""
    fn = FEED_PLAN.get(name)
    return fn() if fn else {}


def build(name: str, out_dir: pathlib.Path) -> pathlib.Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.onnx"
    onnx.save(BUILDERS[name](), str(path))
    return path


def main() -> int:
    out_dir = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("bench/results/_ledger_models")
    for name in BUILDERS:
        print(build(name, out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------------------------
# The planted control, declared once so that tools can refuse it.
#
# `gen_proof_ledger.py` refuses to write this key and fails `--check` if it is present. Without
# that, the only thing standing between a populated ledger and a disarmed control is whoever
# happens to read the case list — and on 2026-08-02 that was not enough.
# ---------------------------------------------------------------------------

PLANTED_CONTROL_CASE = "sub_f16_dyn_unproven"
PLANTED_CONTROL_KEY = "ai.onnx::Sub/7+/f16,f16>f16/ew_binary_sub_f16/runtime-extent/n2"
#: The proven sibling. The pair differs in exactly one key component: `shape_class`.
PLANTED_CONTROL_SIBLING_KEY = "ai.onnx::Sub/7+/f16,f16>f16/ew_binary_sub_f16/static/n2"
