"""test_gqa_decode_kv_parallel.py — issue #90, the KV-parallel decode kernel on the real device.

WHAT THIS FILE IS FOR
=====================

`gqa_decode_f16` splits one decode step's KV cache across `W` invocations of a single workgroup
and merges their partial softmaxes through shared memory. `tests/ops/test_gqa.py` cannot reach
it: every past length it uses (0, 4, 16) is below the selector's `W >= 2` boundary, so those
tests exercise `gqa_f16` and would keep passing if `gqa_decode_f16` were deleted.

This file is the production-path coverage for the new module, and it is written so that it
CANNOT pass on the old one. Every case asserts a **pipeline witness** — the EP's own
`pipeline_variants` record, read out of a counters snapshot the EP wrote at teardown — before
it compares a single number. A run that silently fell back to the serial kernel fails here
rather than passing quietly with the serial kernel's (correct) answers, which is the specific
way a parallel-kernel suite goes vacuous.

WHAT IS COMPARED, AND AGAINST WHAT
==================================

ORT's own CPU `GroupQueryAttention` is the oracle, on **all three** outputs:

  * `attn_out`      — the softmax-weighted value sum, which is what the merge computes;
  * `present_key`   — the KV cache after the write, which is what the fused past-copy and the
                      rotated new-key write produce;
  * `present_value` — likewise for values.

An `attn_out`-only comparison would leave the cache-write path — `kv_write_leader`, the
`lane == 0` pin on the new row, and the strided fused copy — checked by nothing, and those are
exactly the parts that changed shape when the work was split across lanes. Requirement: an
argmax or a single-output check is NOT sufficient here and is not used.

THE TOLERANCE IS PREDECLARED
============================

`TOLERANCE` below is fixed before any measurement and applies per output. It is not widened in
response to a result. `present_key`/`present_value` additionally get a **bitwise** check on the
region the kernel is supposed to copy rather than compute (see
`_assert_past_region_is_copied_bitwise`): a copy that is merely within 1e-2 of the input is not
a copy, and no float tolerance can express that.

WHAT THIS FILE DOES NOT CLAIM
=============================

Nothing here is a speed measurement. Kernel timing lives in
`bench/results/probe_gqa_decode_kv_parallel.py` and its scope limits are stated there and in
docs/PERF.md. A green run of this file means the parallel kernel agrees with the CPU kernel on
this device; it says nothing about how fast it is, and nothing about any other device than the
one the run opened.
"""

from __future__ import annotations

import ctypes
import json
import os
import pathlib
import contextlib

import numpy as np
import pytest
from onnx import TensorProto, helper

import _models as m

# ---------------------------------------------------------------------------------------------
# The selector, mirrored — deliberately, and deliberately not imported
# ---------------------------------------------------------------------------------------------
#
# These four constants and `_expected_lanes` restate `ops::attention::gqa_decode_kv_lanes` in
# Python. That is a second implementation of a rule, which is normally the mirror-drift defect
# this repo avoids by exporting the answer from the DLL. It is the right thing HERE for one
# reason: the mirror is never consulted to decide what to run, only to state what the run is
# expected to have produced. If the Rust selector drifts from this file, the pipeline witness
# assertion fails with "expected gqa_decode_f16:4, saw gqa_decode_f16:8" — the drift is the
# detection, not a silent agreement. A test that asked the EP what it was about to do and then
# checked it did that would assert nothing at all.

MAX_KV_LANES = 16
MIN_KV_PER_LANE = 32
MAX_HEAD_DIM = 128
ENV_KV_PARALLEL = "ONNXRUNTIME_EP_VULKAN_GQA_DECODE_KV_PARALLEL"

#: Past lengths that select each lane count, and the smallest one that does.
#: `past + 1 >= 32 * W` is the rule, so the boundaries are 63, 127, 255, 511.
LANE_BOUNDARY = {2: 63, 4: 127, 8: 255, 16: 511}

#: Predeclared, per output. Fixed before the first run of this file; see the module docstring.
TOLERANCE = {
    "attn_out": {"rtol": 1e-2, "atol": 1e-2},
    "present_key": {"rtol": 1e-2, "atol": 1e-2},
    "present_value": {"rtol": 1e-2, "atol": 1e-2},
}

SERIAL_STEM = "gqa_f16"
DECODE_STEM = "gqa_decode_f16"


def _expected_lanes(seq_len: int, past_len: int, head_dim: int) -> int:
    if seq_len != 1 or head_dim == 0 or head_dim > MAX_HEAD_DIM:
        return 1
    total = past_len + seq_len
    w = MAX_KV_LANES
    while w >= 2:
        if total // w >= MIN_KV_PER_LANE:
            return w
        w //= 2
    return 1


# ---------------------------------------------------------------------------------------------
# Model + feeds
# ---------------------------------------------------------------------------------------------


def _cos_sin(max_seq: int, rot_dim: int) -> tuple[np.ndarray, np.ndarray]:
    pos = np.arange(max_seq, dtype=np.float32)[:, None]
    freq = 1.0 / (10000 ** (np.arange(0, rot_dim, 2, dtype=np.float32) / rot_dim))
    angles = pos * freq
    return np.cos(angles).astype(np.float16), np.sin(angles).astype(np.float16)


def _build(
    *,
    past_seq: int,
    num_heads: int = 8,
    kv_heads: int = 2,
    head_dim: int = 32,
    seq_len: int = 1,
    rotary_dim: int = 16,
) -> bytes:
    packed_dim = (num_heads + 2 * kv_heads) * head_dim
    max_seq = past_seq + seq_len + 1
    f16 = TensorProto.FLOAT16
    ins = [
        helper.make_tensor_value_info("packed_qkv", f16, ["B", seq_len, packed_dim]),
        helper.make_tensor_value_info("past_key", f16, ["B", kv_heads, past_seq, head_dim]),
        helper.make_tensor_value_info("past_value", f16, ["B", kv_heads, past_seq, head_dim]),
        helper.make_tensor_value_info("seqlens_k", TensorProto.INT32, ["B"]),
        helper.make_tensor_value_info("total_seq", TensorProto.INT32, []),
        helper.make_tensor_value_info("cos_cache", f16, [max_seq, rotary_dim // 2]),
        helper.make_tensor_value_info("sin_cache", f16, [max_seq, rotary_dim // 2]),
    ]
    outs = [
        helper.make_tensor_value_info(
            "attn_out", f16, ["B", seq_len, num_heads * head_dim]
        ),
        helper.make_tensor_value_info(
            "present_key", f16, ["B", kv_heads, past_seq + seq_len, head_dim]
        ),
        helper.make_tensor_value_info(
            "present_value", f16, ["B", kv_heads, past_seq + seq_len, head_dim]
        ),
    ]
    node = helper.make_node(
        "GroupQueryAttention",
        inputs=[
            "packed_qkv", "", "", "past_key", "past_value", "seqlens_k", "total_seq",
            "cos_cache", "sin_cache",
        ],
        outputs=["attn_out", "present_key", "present_value"],
        domain="com.microsoft",
        name="gqa_kvpar",
        num_heads=num_heads,
        kv_num_heads=kv_heads,
        scale=float(head_dim ** -0.5),
        local_window_size=-1,
        do_rotary=1,
        rotary_interleaved=0,
        smooth_softmax=0,
    )
    model = helper.make_model(
        helper.make_graph([node], "gqa_kvpar_graph", ins, outs),
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("com.microsoft", 1),
        ],
    )
    model.ir_version = 10
    return model.SerializeToString()


def _feeds(
    *,
    past_seq: int,
    num_heads: int = 8,
    kv_heads: int = 2,
    head_dim: int = 32,
    seq_len: int = 1,
    rotary_dim: int = 16,
    batch: int = 1,
    seqlens: "list[int] | None" = None,
    scale: float = 0.1,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    packed_dim = (num_heads + 2 * kv_heads) * head_dim
    max_seq = past_seq + seq_len + 1
    lens = seqlens if seqlens is not None else [past_seq] * batch
    assert len(lens) == batch
    cos, sin = _cos_sin(max_seq, rotary_dim)
    return {
        "packed_qkv": (
            rng.standard_normal((batch, seq_len, packed_dim)) * scale
        ).astype(np.float16),
        "past_key": (
            rng.standard_normal((batch, kv_heads, past_seq, head_dim)) * scale
        ).astype(np.float16),
        "past_value": (
            rng.standard_normal((batch, kv_heads, past_seq, head_dim)) * scale
        ).astype(np.float16),
        "seqlens_k": np.array(lens, dtype=np.int32),
        "total_seq": np.array(max(lens) + seq_len, dtype=np.int32),
        "cos_cache": cos,
        "sin_cache": sin,
    }


# ---------------------------------------------------------------------------------------------
# The witness harness
# ---------------------------------------------------------------------------------------------


@contextlib.contextmanager
def _env(**kv: "str | None"):
    saved = {k: os.environ.get(k) for k in kv}
    try:
        for k, v in kv.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _reset_counters() -> bool:
    """Zero the EP's process-wide counters, including the dispatched-shader sets.

    Returns False when the DLL cannot be located or does not export the reset, which is
    reported by the caller as an instrument outage — never read as "nothing was dispatched".
    The sets are process-cumulative, so without this a case would inherit the previous case's
    `pipeline_variants` and every witness after the first would be someone else's.
    """
    lib = os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB")
    if not lib or not pathlib.Path(lib).is_file():
        return False
    try:
        dll = ctypes.CDLL(lib)
        dll.OrtEpVulkanResetExecutionCounters()
    except (OSError, AttributeError):
        return False
    return True


def _run_with_witness(
    model: bytes,
    feeds: dict,
    tmp_path: pathlib.Path,
    tag: str,
    *,
    override: "str | None" = None,
) -> tuple[list[np.ndarray], dict]:
    """Run *model* on the Vulkan EP and return its outputs plus the EP's own counters snapshot.

    The snapshot is written by the EP at teardown, so the session is dropped before the file is
    read. `pipeline_variants` and `shaders_dispatched` in the returned dict describe THIS run
    and no other, because `_reset_counters` cleared them immediately before it.
    """
    if not _reset_counters():
        pytest.skip(
            "instrument unavailable: ONNXRUNTIME_VULKAN_EP_LIB is unset or does not export "
            "OrtEpVulkanResetExecutionCounters, so a pipeline witness for this case would be "
            "the previous case's. Refusing to report a result rather than reporting an "
            "unattributable one."
        )
    counters_path = tmp_path / f"counters_{tag}.json"
    with _env(
        ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE=str(counters_path),
        **{ENV_KV_PARALLEL: override},
    ):
        import onnxruntime as ort

        session = ort.InferenceSession(
            model, m._make_session_options(), providers=m.EP_PROVIDERS
        )
        m.assert_ep_in_session(session)
        outputs = session.run(None, feeds)
        del session
    assert counters_path.is_file(), (
        f"the EP wrote no counters snapshot to {counters_path}. Without it there is no "
        f"pipeline witness, and an equivalence result with no witness cannot say which kernel "
        f"produced it."
    )
    with counters_path.open(encoding="utf-8") as fh:
        counters = json.load(fh)
    return outputs, counters


def _assert_decode_pipeline(counters: dict, lanes: int, where: str) -> None:
    """The load-bearing assertion of this file: the PARALLEL module ran, at THIS lane count."""
    variants = counters.get("pipeline_variants") or []
    stems = counters.get("shaders_dispatched") or []
    want = f"{DECODE_STEM}:{lanes}"
    assert want in variants, (
        f"{where}: the EP did not build the parallel pipeline. pipeline_variants={variants!r}, "
        f"shaders_dispatched={stems!r}, wanted {want!r}. Either the selector declined this "
        f"shape (in which case this case tests the serial kernel and proves nothing about "
        f"issue #90) or the lane count moved. Both are failures of this test, not of the "
        f"comparison below it."
    )
    assert DECODE_STEM in stems, (
        f"{where}: {DECODE_STEM} appears in pipeline_variants but not in shaders_dispatched "
        f"({stems!r}) — a pipeline was created and never dispatched."
    )
    assert SERIAL_STEM not in stems, (
        f"{where}: both kernels ran ({stems!r}). This case cannot attribute its outputs to "
        f"either one."
    )
    assert int(counters.get("dispatches_executed") or 0) >= 1, (
        f"{where}: dispatches_executed={counters.get('dispatches_executed')!r}"
    )


def _assert_serial_pipeline(counters: dict, where: str) -> None:
    """The refusal polarity: the parallel module must NOT have been built at all."""
    variants = counters.get("pipeline_variants") or []
    stems = counters.get("shaders_dispatched") or []
    assert not any(v.startswith(f"{DECODE_STEM}:") for v in variants), (
        f"{where}: the parallel pipeline was built where the selector must refuse. "
        f"pipeline_variants={variants!r}"
    )
    assert DECODE_STEM not in stems, (
        f"{where}: the parallel module was dispatched where the selector must refuse. "
        f"shaders_dispatched={stems!r}"
    )


def _compare(
    vk: list[np.ndarray],
    cpu: list[np.ndarray],
    where: str,
    names=("attn_out", "present_key", "present_value"),
    masks: "dict[str, np.ndarray] | None" = None,
) -> None:
    """Per-output comparison at the predeclared tolerance. Every output, every element.

    *masks*, when given, restricts an output to the elements the operator DEFINES. Only
    `present_key`/`present_value` under a ragged `seqlens_k` ever use it, and the elements it
    excludes are checked instead — more strictly — by `_assert_undefined_region_matches_serial`.
    A mask is never a way to drop an element that disagrees; it is a way to say which elements
    the two implementations were ever required to agree about.
    """
    for i, name in enumerate(names):
        tol = TOLERANCE[name]
        a = vk[i].astype(np.float32)
        b = cpu[i].astype(np.float32)
        assert a.shape == b.shape, f"{where}/{name}: shape {a.shape} != CPU {b.shape}"
        scope = np.ones(a.shape, dtype=bool) if masks is None else masks.get(name)
        if scope is None:
            scope = np.ones(a.shape, dtype=bool)
        finite = np.isfinite(a) & np.isfinite(b)
        assert np.array_equal(np.isfinite(a)[scope], np.isfinite(b)[scope]), (
            f"{where}/{name}: the two kernels disagree about WHICH elements are finite. "
            f"vulkan non-finite={int((~np.isfinite(a))[scope].sum())}, "
            f"cpu non-finite={int((~np.isfinite(b))[scope].sum())}. That is a policy "
            f"divergence, not a tolerance question, and no tolerance is applied to it."
        )
        bad = ~np.isclose(a, b, rtol=tol["rtol"], atol=tol["atol"]) & finite & scope
        n = int(bad.sum())
        assert n == 0, (
            f"{where}/{name}: {n} of {int(scope.sum())} defined elements outside the "
            f"PREDECLARED tolerance rtol={tol['rtol']} atol={tol['atol']}. worst |vk-cpu| = "
            f"{float(np.abs(a[bad] - b[bad]).max()):.6g} at |cpu| = "
            f"{float(np.abs(b[bad]).max()):.6g}. This tolerance is not to be widened to make "
            f"this line green."
        )


def _defined_kv_masks(shape: tuple, seqlens: list[int], seq_len: int) -> dict:
    """Which `present_key`/`present_value` elements does GroupQueryAttention DEFINE?

    `past_key` is an ALLOCATION of `past_seq` rows; `seqlens_k[b]` says how many of them hold
    real history for batch `b`. Rows at or beyond `seqlens_k[b] + seq_len` hold whatever the
    allocation held, and the operator says nothing about what a kernel leaves there. ORT's CPU
    kernel and this EP's SERIAL kernel already differ there, on `main`, with no KV split
    involved — measured on 2026-08-09 at seqlens `[127, 63]`: 3710 of 16384 `present_key`
    elements differ from CPU, identically for both EP kernels, worst |vk-cpu| = 0.437744.

    So this mask is not a way to excuse a difference the split introduced. It is a statement of
    what the comparison is entitled to ask, and every element it excludes is then held to a
    STRICTER standard than the one it escapes: `_assert_undefined_region_matches_serial`
    requires the parallel kernel to be BITWISE identical to the serial kernel there.
    """
    b, hkv, total, d = shape
    mask = np.zeros((b, hkv, total, d), dtype=bool)
    for i, klen in enumerate(seqlens):
        mask[i, :, : min(klen + seq_len, total), :] = True
    return {"present_key": mask, "present_value": mask}


def _assert_undefined_region_matches_serial(
    par: list[np.ndarray], ser: list[np.ndarray], where: str
) -> None:
    """Outside the defined region, the two EP kernels must agree BIT FOR BIT.

    This is the assertion that makes `_defined_kv_masks` safe. The operator does not define
    those elements, so CPU cannot arbitrate them — but "undefined" is not a licence for the new
    kernel to change what the shipping one wrote, and a bitwise check is the strongest
    statement available that it did not. It is also the check that would catch a strided fused
    copy that wrote the wrong rows in exactly the region no tolerance test looks at.
    """
    for i, name in ((1, "present_key"), (2, "present_value")):
        assert np.array_equal(par[i].view(np.uint16), ser[i].view(np.uint16)), (
            f"{where}/{name}: the parallel kernel and the serial kernel wrote different bytes. "
            f"Whatever the operator defines, these two must agree everywhere — the split is "
            f"supposed to reorder work, not change what is written."
        )


def _assert_past_region_is_copied_bitwise(
    vk: list[np.ndarray], feeds: dict, seqlens: list[int], where: str
) -> None:
    """The cache's past region is COPIED, not recomputed — so it must be bitwise identical.

    `gqa_decode_f16` fuses the past-copy into the same strided loop that computes attention, so
    each row `t < past_len` is written by whichever lane owns `t`. A tolerance check cannot tell
    a correct copy from a copy that lost a lane's rows and refilled them with something close;
    a bitwise check can, and it is the only assertion in this file that would catch a stride
    bug that happened to land on numerically similar data.
    """
    for out_idx, in_name in ((1, "past_key"), (2, "present_value")):
        src = feeds["past_key" if out_idx == 1 else "past_value"]
        got = vk[out_idx]
        for b, klen in enumerate(seqlens):
            if klen == 0:
                continue
            assert np.array_equal(
                got[b, :, :klen, :].view(np.uint16), src[b, :, :klen, :].view(np.uint16)
            ), (
                f"{where}/{'present_key' if out_idx == 1 else 'present_value'}: batch {b}'s "
                f"first {klen} cache rows are not a bitwise copy of the input cache. The past "
                f"region is copied, never recomputed, so any difference here is a lost or "
                f"duplicated row in the strided fused copy — not rounding."
            )
        _ = in_name


# ---------------------------------------------------------------------------------------------
# 1. The lane matrix — W = 2, 4, 8, 16 on the production path
# ---------------------------------------------------------------------------------------------


@pytest.mark.portability
@pytest.mark.parametrize("lanes", [2, 4, 8, 16])
def test_lane_matrix_matches_cpu(lanes: int, tmp_path, vulkan_device_available) -> None:
    """Each supported lane count builds its own pipeline and agrees with the CPU kernel.

    This is the B4 requirement in one test: a production dispatch that actually reaches W >= 2,
    witnessed by the EP's own record, compared on all three outputs.
    """
    past = LANE_BOUNDARY[lanes]
    assert _expected_lanes(1, past, 32) == lanes
    model = _build(past_seq=past)
    feeds = _feeds(past_seq=past, seed=lanes)

    vk, counters = _run_with_witness(model, feeds, tmp_path, f"w{lanes}")
    _assert_decode_pipeline(counters, lanes, f"W={lanes} past={past}")
    cpu = m.run_cpu(model, feeds)
    _compare(vk, cpu, f"W={lanes} past={past}")
    _assert_past_region_is_copied_bitwise(vk, feeds, [past], f"W={lanes} past={past}")


@pytest.mark.parametrize("past", [63, 64, 100, 127, 128, 200, 255, 300, 511, 512])
def test_lane_ladder_is_the_selectors_ladder(past: int, tmp_path, vulkan_device_available) -> None:
    """Across a ragged sweep of past lengths, the built pipeline is the one the rule predicts.

    A selector that quietly rounded, clamped or off-by-oned would still produce correct output
    (every lane count is correct) and would be invisible to an equivalence test. It is visible
    here, because the lane count is read out of the EP rather than assumed.
    """
    lanes = _expected_lanes(1, past, 32)
    assert lanes >= 2, "this sweep is for shapes the selector accepts"
    model = _build(past_seq=past)
    feeds = _feeds(past_seq=past, seed=past)
    vk, counters = _run_with_witness(model, feeds, tmp_path, f"ladder{past}")
    _assert_decode_pipeline(counters, lanes, f"past={past}")
    _compare(vk, m.run_cpu(model, feeds), f"past={past}")


# ---------------------------------------------------------------------------------------------
# 2. Shape coverage — head_dim, grouping, batch, ragged past
# ---------------------------------------------------------------------------------------------


@pytest.mark.portability
@pytest.mark.parametrize("head_dim", [32, 64, 128])
def test_head_dim_coverage(head_dim: int, tmp_path, vulkan_device_available) -> None:
    """head_dim 32/64/128 — the last is the module's D_MAX and its shared-memory ceiling."""
    past = 127
    model = _build(past_seq=past, head_dim=head_dim, rotary_dim=min(16, head_dim))
    feeds = _feeds(past_seq=past, head_dim=head_dim, rotary_dim=min(16, head_dim), seed=head_dim)
    vk, counters = _run_with_witness(model, feeds, tmp_path, f"hd{head_dim}")
    _assert_decode_pipeline(counters, _expected_lanes(1, past, head_dim), f"head_dim={head_dim}")
    _compare(vk, m.run_cpu(model, feeds), f"head_dim={head_dim}")


@pytest.mark.parametrize(
    "num_heads,kv_heads,label",
    [(8, 8, "MHA"), (8, 2, "GQA-4x"), (8, 1, "MQA"), (4, 2, "GQA-2x")],
)
def test_head_grouping_coverage(
    num_heads: int, kv_heads: int, label: str, tmp_path, vulkan_device_available
) -> None:
    """MHA, MQA and two GQA ratios.

    The grouping decides `kv_write_leader` — which query head's workgroup owns the cache write
    for its KV head. Getting it wrong duplicates or drops a present-KV write, which `attn_out`
    alone would not see on a single decode step.
    """
    past = 127
    model = _build(past_seq=past, num_heads=num_heads, kv_heads=kv_heads)
    feeds = _feeds(past_seq=past, num_heads=num_heads, kv_heads=kv_heads, seed=num_heads * 10 + kv_heads)
    vk, counters = _run_with_witness(model, feeds, tmp_path, f"grp{num_heads}x{kv_heads}")
    _assert_decode_pipeline(counters, _expected_lanes(1, past, 32), label)
    _compare(vk, m.run_cpu(model, feeds), label)
    _assert_past_region_is_copied_bitwise(vk, feeds, [past], label)


@pytest.mark.parametrize("batch,seqlens", [(2, [127, 127]), (2, [127, 63]), (3, [255, 100, 3])])
def test_batch_and_ragged_past(
    batch: int, seqlens: list[int], tmp_path, vulkan_device_available
) -> None:
    """Batch > 1, and batches whose real KV lengths differ from each other and from the cache.

    `seqlens_k` below the allocated past extent is how this operator spells a mask: positions
    at or beyond it must contribute nothing. A merge that let an empty lane's `m = -1e30`
    sentinel leak into the running maximum, or that divided by a zero denominator, shows up
    here and only here — the uniform-length cases cannot produce an empty lane.

    Two comparisons, because a ragged cache has two regions with two different guarantees:
    the DEFINED region is compared to ORT's CPU kernel at the predeclared tolerance, and the
    undefined tail is compared BITWISE to the serial kernel this change is replacing. See
    `_defined_kv_masks` for the measurement that motivated the split; `attn_out`, which is
    fully defined, is compared to CPU in its entirety either way.
    """
    past = max(seqlens)
    model = _build(past_seq=past)
    feeds = _feeds(past_seq=past, batch=batch, seqlens=seqlens, seed=sum(seqlens))
    where = f"batch={batch} seqlens={seqlens}"
    vk, counters = _run_with_witness(model, feeds, tmp_path, f"b{batch}_{past}")
    _assert_decode_pipeline(counters, _expected_lanes(1, past, 32), where)
    masks = _defined_kv_masks(vk[1].shape, seqlens, 1)
    _compare(vk, m.run_cpu(model, feeds), where, masks=masks)
    _assert_past_region_is_copied_bitwise(vk, feeds, seqlens, where)
    serial, c_ser = _run_with_witness(
        model, feeds, tmp_path, f"b{batch}_{past}_ser", override="1"
    )
    _assert_serial_pipeline(c_ser, f"{where} (serial arm)")
    _assert_undefined_region_matches_serial(vk, serial, where)


# ---------------------------------------------------------------------------------------------
# 3. Numerics — cancellation and the non-finite policy
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [1e-3, 0.1, 1.0])
def test_cancellation_matches_cpu(scale: float, tmp_path, vulkan_device_available) -> None:
    """The online-softmax merge across three input magnitudes, against ORT's CPU kernel.

    These are the scales at which f16 attention itself is faithful enough for the CPU kernel to
    be a usable arbiter — measured worst |vk-cpu| on `attn_out` at past 255: 7.6e-06 at 0.1 and
    1.2e-04 at 1.0, both far inside the predeclared 1e-2. The regime where f16 stops being
    faithful is the next test, and it is judged differently and deliberately.
    """
    past = 255
    model = _build(past_seq=past)
    feeds = _feeds(past_seq=past, scale=scale, seed=int(scale * 1000) + 1)
    where = f"scale={scale}"
    vk, counters = _run_with_witness(model, feeds, tmp_path, f"sc{scale}")
    _assert_decode_pipeline(counters, 8, where)
    _compare(vk, m.run_cpu(model, feeds), where)


@pytest.mark.parametrize("scale", [2.0, 4.0])
def test_extreme_magnitude_is_no_worse_than_the_kernel_it_replaces(
    scale: float, tmp_path, vulkan_device_available
) -> None:
    """Where f16 attention itself diverges from CPU, the criterion is PARITY, not tolerance.

    # Why this test is judged against the serial kernel and not against CPU

    At `scale = 4` the logits are large enough that f16 rounding inside the exponential
    dominates. Measured on 2026-08-09 at past 255, `attn_out`: max |vk-cpu| = 0.013916 — and
    that is the number for the SERIAL kernel, on `main`, with no KV split anywhere in it. The
    parallel kernel produces 0.013916 as well, and max |parallel - serial| = 0, BITWISE.

    So a vs-CPU tolerance here would not be measuring this change. It would be measuring f16,
    and it could be met only by widening the predeclared tolerance until it stopped saying
    anything — which is precisely the move requirement 7 forbids. The honest criterion in this
    regime is the one this test applies: the new kernel must not be WORSE than the kernel it
    replaces, and its residual must not exceed the serial kernel's.

    # Why this is not vacuous

    It is a two-sided comparison against an independently produced arm. The serial arm is a
    real dispatch of `gqa_f16` — `_assert_serial_pipeline` proves it — not a recomputation of
    the parallel arm, and a merge bug that changed the answer would move `max|par-cpu|` above
    `max|ser-cpu|` and fail. The `scale = 2.0` case, where both arms sit at 2.1e-03 and inside
    the predeclared tolerance, is the positive control that says the comparison is live.
    """
    past = 255
    model = _build(past_seq=past)
    feeds = _feeds(past_seq=past, scale=scale, seed=int(scale * 1000) + 1)
    where = f"scale={scale}"
    par, c_par = _run_with_witness(model, feeds, tmp_path, f"ex{scale}")
    _assert_decode_pipeline(c_par, 8, where)
    ser, c_ser = _run_with_witness(model, feeds, tmp_path, f"ex{scale}_ser", override="1")
    _assert_serial_pipeline(c_ser, f"{where} (serial arm)")
    cpu = m.run_cpu(model, feeds)
    for i, name in enumerate(("attn_out", "present_key", "present_value")):
        a = par[i].astype(np.float32)
        s = ser[i].astype(np.float32)
        c = cpu[i].astype(np.float32)
        assert np.array_equal(np.isfinite(a), np.isfinite(s)), (
            f"{where}/{name}: the two EP kernels disagree about which elements are finite"
        )
        par_worst = float(np.abs(a - c).max())
        ser_worst = float(np.abs(s - c).max())
        assert par_worst <= ser_worst + 1e-6, (
            f"{where}/{name}: the parallel kernel is WORSE than the serial kernel it replaces "
            f"— max|parallel-cpu|={par_worst:.6g} > max|serial-cpu|={ser_worst:.6g}. The KV "
            f"split is supposed to reorder the sum, not degrade it."
        )
    _assert_undefined_region_matches_serial(par, ser, where)


def test_denominator_never_collapses_at_the_longest_cache(tmp_path, vulkan_device_available) -> None:
    """W = 16 with a real cache: every lane merges, and the result is finite.

    Lane 0 always owns `t = 0`, so the running denominator is never zero no matter how the
    strided partition falls. This is the test that would go red if that stopped being true.
    """
    past = 511
    model = _build(past_seq=past)
    feeds = _feeds(past_seq=past, seed=511)
    vk, counters = _run_with_witness(model, feeds, tmp_path, "denom")
    _assert_decode_pipeline(counters, 16, "W=16 denominator")
    assert np.isfinite(vk[0].astype(np.float32)).all(), (
        "attn_out contains a non-finite element at W=16 on ordinary inputs"
    )
    _compare(vk, m.run_cpu(model, feeds), "W=16 denominator")


# ---------------------------------------------------------------------------------------------
# 4. The selector boundary — the refusal polarity
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("seq_len", [2, 3, 4, 5, 6, 7, 8])
def test_prefill_sequence_lengths_refuse_the_parallel_kernel(
    seq_len: int, tmp_path, vulkan_device_available
) -> None:
    """`seq_len == 1` is a hard boundary, not a heuristic — and 2..8 prove it.

    `gqa_decode_f16` drops the sibling-key interval `past_len <= t < tok_pos`, which is empty
    ONLY at `seq_len == 1`. A mutation that relaxed the selector's `== 1` to `<= 8` would keep
    every other test in this file green — the shapes they use all have `seq_len == 1` — and
    would silently compute wrong attention for the multi-token step. These seven cases are the
    held-out control for exactly that mutation: each asserts the parallel pipeline was NOT
    built, and then that the answer is still right, so the mutation fails on the witness rather
    than on a number that might coincidentally survive.
    """
    past = 511
    model = _build(past_seq=past, seq_len=seq_len)
    feeds = _feeds(past_seq=past, seq_len=seq_len, seed=seq_len)
    where = f"seq_len={seq_len} past={past}"
    vk, counters = _run_with_witness(model, feeds, tmp_path, f"s{seq_len}")
    _assert_serial_pipeline(counters, where)
    _compare(vk, m.run_cpu(model, feeds), where)


@pytest.mark.parametrize("past", [0, 1, 4, 16, 31, 62])
def test_short_caches_refuse_the_parallel_kernel(
    past: int, tmp_path, vulkan_device_available
) -> None:
    """Below past 63 the split is refused: two lanes over fewer than 64 tokens is not work.

    62 is the last refusal and 63 is the first acceptance
    (`test_lane_matrix_matches_cpu[2]`); the pair pins the boundary from both sides, so an
    off-by-one in either direction is red.
    """
    model = _build(past_seq=past)
    feeds = _feeds(past_seq=past, seed=past + 900)
    where = f"past={past}"
    vk, counters = _run_with_witness(model, feeds, tmp_path, f"short{past}")
    _assert_serial_pipeline(counters, where)
    _compare(vk, m.run_cpu(model, feeds), where)


def test_boundary_is_closed_at_63_and_open_at_62(tmp_path, vulkan_device_available) -> None:
    """The two adjacent shapes, in one test, so the boundary cannot be read as two facts."""
    for past, want_parallel in ((62, False), (63, True)):
        model = _build(past_seq=past)
        feeds = _feeds(past_seq=past, seed=past)
        vk, counters = _run_with_witness(model, feeds, tmp_path, f"bnd{past}")
        if want_parallel:
            _assert_decode_pipeline(counters, 2, f"past={past}")
        else:
            _assert_serial_pipeline(counters, f"past={past}")
        _compare(vk, m.run_cpu(model, feeds), f"past={past}")


# ---------------------------------------------------------------------------------------------
# 5. The switch — kill, clamp, and the fallback's exactness
# ---------------------------------------------------------------------------------------------


def test_override_1_is_an_exact_kill_switch(tmp_path, vulkan_device_available) -> None:
    """`=1` takes a shape that would split all the way back to the serial kernel.

    Not "back to one lane of the parallel kernel" — back to `gqa_f16` itself, the module that
    shipped before this change, at its own workgroup size. That is what makes the switch an
    operational fallback rather than a slower configuration of the new code, and
    `_assert_serial_pipeline` is what distinguishes the two.
    """
    past = 511
    model = _build(past_seq=past)
    feeds = _feeds(past_seq=past, seed=1)
    on, c_on = _run_with_witness(model, feeds, tmp_path, "kill_on")
    _assert_decode_pipeline(c_on, 16, "override unset")
    off, c_off = _run_with_witness(model, feeds, tmp_path, "kill_off", override="1")
    _assert_serial_pipeline(c_off, "override=1")
    assert SERIAL_STEM in (c_off.get("shaders_dispatched") or []), (
        f"override=1 dispatched {c_off.get('shaders_dispatched')!r}, not the serial module"
    )
    cpu = m.run_cpu(model, feeds)
    _compare(on, cpu, "override unset")
    _compare(off, cpu, "override=1")


@pytest.mark.parametrize("value,lanes", [("2", 2), ("4", 4), ("8", 8), ("16", 16)])
def test_override_pins_the_lane_count(
    value: str, lanes: int, tmp_path, vulkan_device_available
) -> None:
    """A pinned lane count is the A/B control, and it must reach the pipeline it names.

    past = 511 admits every lane count, so each of these four is a genuine choice rather than
    the auto rule wearing an override's name.
    """
    past = 511
    model = _build(past_seq=past)
    feeds = _feeds(past_seq=past, seed=int(value))
    vk, counters = _run_with_witness(model, feeds, tmp_path, f"ov{value}", override=value)
    _assert_decode_pipeline(counters, lanes, f"override={value}")
    _compare(vk, m.run_cpu(model, feeds), f"override={value}")


@pytest.mark.parametrize("value,lanes", [("3", 2), ("31", 16), ("999", 16), ("0", None)])
def test_override_clamps_and_floors(
    value: str, lanes: "int | None", tmp_path, vulkan_device_available
) -> None:
    """Out-of-range and non-power-of-two overrides resolve to a lane count the module has.

    The clamp is to [1, 16] and the floor is to a power of two, so 3 -> 2, 31 -> 16, 999 -> 16
    and 0 -> the serial kernel. An unclamped 999 would specialise a workgroup the shared
    allocation cannot serve; this is the test that says the host, not the driver, refuses it.
    """
    past = 511
    model = _build(past_seq=past)
    feeds = _feeds(past_seq=past, seed=hash(value) % 1000)
    vk, counters = _run_with_witness(model, feeds, tmp_path, f"cl{value}", override=value)
    if lanes is None:
        _assert_serial_pipeline(counters, f"override={value}")
    else:
        _assert_decode_pipeline(counters, lanes, f"override={value}")
    _compare(vk, m.run_cpu(model, feeds), f"override={value}")


def test_override_cannot_open_the_sequence_length_gate(tmp_path, vulkan_device_available) -> None:
    """The correctness gates are applied BEFORE the override, so no value can open them.

    An override that could force the parallel kernel onto a prefill shape would turn an
    operator convenience into a correctness hazard. It cannot: `seq_len != 1` refuses first.
    """
    past = 511
    model = _build(past_seq=past, seq_len=4)
    feeds = _feeds(past_seq=past, seq_len=4, seed=4)
    vk, counters = _run_with_witness(model, feeds, tmp_path, "gate", override="16")
    _assert_serial_pipeline(counters, "override=16 at seq_len=4")
    _compare(vk, m.run_cpu(model, feeds), "override=16 at seq_len=4")


# ---------------------------------------------------------------------------------------------
# 6. Pipeline identity — two kernels, two cache keys, no alias
# ---------------------------------------------------------------------------------------------


def test_lane_counts_are_distinct_pipelines_within_one_process(
    tmp_path, vulkan_device_available
) -> None:
    """Four lane counts in one process produce four distinct pipeline-cache entries.

    The pipeline cache is keyed on `(shader_stem, spec_constants)`. If the lane count were left
    out of that key, the second lane count in a process would silently reuse the first one's
    pipeline — correct-looking output at the wrong workgroup size, and the fastest possible way
    to make a benchmark lie. Because `_reset_counters` clears the variant set between runs,
    this test re-runs without the reset to observe the ACCUMULATED set, which is where the
    aliasing would be visible.
    """
    past = 511
    model = _build(past_seq=past)
    feeds = _feeds(past_seq=past, seed=77)
    counters_path = tmp_path / "counters_identity.json"
    assert _reset_counters(), "instrument unavailable"
    import onnxruntime as ort

    for value in ("2", "4", "8", "16"):
        with _env(
            ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE=str(counters_path),
            **{ENV_KV_PARALLEL: value},
        ):
            s = ort.InferenceSession(model, m._make_session_options(), providers=m.EP_PROVIDERS)
            s.run(None, feeds)
            del s
    with counters_path.open(encoding="utf-8") as fh:
        variants = json.load(fh)["pipeline_variants"]
    want = {f"{DECODE_STEM}:{w}" for w in (2, 4, 8, 16)}
    assert want <= set(variants), (
        f"four lane counts produced {sorted(variants)!r}; expected all of {sorted(want)!r}. A "
        f"missing entry means two lane counts shared one pipeline-cache slot."
    )


def test_the_two_kernels_never_share_a_variant_name(tmp_path, vulkan_device_available) -> None:
    """The serial and parallel modules are different stems, so no spec value can collide them."""
    past = 511
    model = _build(past_seq=past)
    feeds = _feeds(past_seq=past, seed=5)
    _, c_par = _run_with_witness(model, feeds, tmp_path, "id_par")
    _, c_ser = _run_with_witness(model, feeds, tmp_path, "id_ser", override="1")
    par = {v for v in c_par["pipeline_variants"] if v.startswith(DECODE_STEM)}
    ser = {v for v in c_ser["pipeline_variants"] if v.startswith(SERIAL_STEM)}
    assert par and ser and not (par & ser), (
        f"parallel variants {sorted(par)!r} and serial variants {sorted(ser)!r} must be "
        f"disjoint sets of names"
    )


# ---------------------------------------------------------------------------------------------
# 7. The proof key — the new form is claimed on its own entry
# ---------------------------------------------------------------------------------------------


def test_the_decode_form_claims_under_its_own_proof_key(
    tmp_path, vulkan_device_available
) -> None:
    """A shape that can reach the parallel kernel keys as `@kvpar`, not as the serial form.

    Issue #90's reviewer finding B2 was that a new module had been given the OLD module's
    ledger entry. The repair is a distinct key: `registry::variant_key` appends `@kvpar`
    whenever the node's declared extents leave the alternative reachable, so the serial entry
    keeps its own independent proof and the parallel form has to earn one. This test reads the
    key the EP actually used out of the claim log.
    """
    past = 127
    model = _build(past_seq=past)
    feeds = _feeds(past_seq=past, seed=3)
    log = tmp_path / "claims.jsonl"
    with _env(ONNXRUNTIME_EP_VULKAN_CLAIM_LOG=str(log)):
        import onnxruntime as ort

        s = ort.InferenceSession(model, m._make_session_options(), providers=m.EP_PROVIDERS)
        s.run(None, feeds)
        del s
    assert log.is_file(), "no claim log was written"
    records = [json.loads(ln) for ln in log.read_text(encoding="utf-8").splitlines() if ln.strip()]
    gqa = [r for r in records if "GroupQueryAttention" in str(r.get("proof_key", ""))]
    assert gqa, f"no GroupQueryAttention claim record in {records!r}"
    keys = {r.get("proof_key", "") for r in gqa}
    assert all("@kvpar" in k for k in keys), (
        f"a shape that reaches the parallel kernel claimed under {keys!r}, which does not carry "
        f"the dispatch-set suffix. Its evidence would be the serial kernel's."
    )
    assert all(r.get("ledger_hit") for r in gqa), (
        f"the decode form found no ledger entry: {gqa!r}. Its entry is missing or its subject "
        f"moved, and the node would decline to CPU in production."
    )
    assert all(r.get("claimed") for r in gqa), (
        f"the decode form declined: {gqa!r}."
    )


def test_the_serial_form_keeps_its_own_untouched_proof_key(
    tmp_path, vulkan_device_available
) -> None:
    """The other half of B2: a shape that CANNOT reach the parallel kernel keys as before.

    `past = 4` is statically too short for two lanes, so `dispatch_set_includes_alternative`
    can prove the alternative unreachable and the key stays the bare `gqa_f16` spelling that is
    already in `evidence/proof_ledger.jsonl`. If the suffix were applied unconditionally, every
    pre-existing GQA entry would go stale at once; this test is what says it is not.
    """
    model = _build(past_seq=4)
    feeds = _feeds(past_seq=4, seed=44)
    log = tmp_path / "claims_serial.jsonl"
    with _env(ONNXRUNTIME_EP_VULKAN_CLAIM_LOG=str(log)):
        import onnxruntime as ort

        s = ort.InferenceSession(model, m._make_session_options(), providers=m.EP_PROVIDERS)
        s.run(None, feeds)
        del s
    records = [json.loads(ln) for ln in log.read_text(encoding="utf-8").splitlines() if ln.strip()]
    gqa = [r for r in records if "GroupQueryAttention" in str(r.get("proof_key", ""))]
    assert gqa, "no GroupQueryAttention claim record"
    for r in gqa:
        assert "@kvpar" not in r["proof_key"], (
            f"a statically-unreachable shape acquired the dispatch-set suffix: {r['proof_key']!r}"
        )
        assert r["proof_key"].endswith(
            "/gqa_f16/runtime-extent/past_key+past_value+cos_cache+sin_cache"
        ), f"the serial key changed spelling: {r['proof_key']!r}"
        assert r.get("ledger_hit") and r.get("claimed"), f"serial form no longer claims: {r!r}"
