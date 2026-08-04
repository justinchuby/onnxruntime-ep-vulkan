"""`Conv` conformance — the attribute axes the proof key deliberately does not carry.

WHY THIS FILE EXISTS AND `test_op_table.py` IS NOT ENOUGH
--------------------------------------------------------
The proof ledger records `Conv` keys of the shape::

    ai.onnx::Conv/1+/f32,f32,f32>f32/conv_f32/static/n3

and `group`, `strides`, `dilations` and `pads` appear nowhere in any of them. That is a ruling,
not an oversight (§8.9.23): `conv_f32.comp` folds all four into push-constant *expressions* on
one uniform code path — `cpg = c / pc.group`, with pads/strides/dilations as index arithmetic
and bounds `continue`s every node executes — so they are expressions, not paths, and the
ProofKey contract permits the collapse. They are **disclosed** instead, as `blind_axes` on the
registry row, rendered into every `Conv` claim line with the clause that a CI-time suite speaks
for them and nothing in the reader's session does.

**This file is that suite.** The disclosure a user reads points here. If these cases stop
running, the claim line is making a promise nothing keeps.

A short-lived attempt (2026-08-04, same day) to put four boolean form bits in the key instead
was reversed: a bit separates `stride > 1` from `stride == 1` but not `stride=2` from
`stride=3`, so it moved the granularity question without answering it, while making the key
claim a distinction the kernel does not draw.

DECLINES ARE ALSO TESTED
------------------------
`Conv` declines f16, rank != 4, `auto_pad != NOTSET` and a symbolic spatial extent, each for a
stated reason. A decline that silently became a claim would be a wrong answer rather than a
missing one, so the declines are asserted, not assumed.
"""

from __future__ import annotations

import numpy as np
import pytest
from onnx_ir import DataType as DT

import _conv_cases as cases
import _models as m

_RNG = np.random.default_rng(0xC0FFEE)


def _build(case) -> tuple[bytes, dict[str, np.ndarray]]:
    (_id, n, c, mm, group, kernel, strides, dilations, pads, hw, bias) = case
    h, w = hw
    kh, kw = kernel
    oh = cases.out_extent(h, pads[0], pads[2], dilations[0], kh, strides[0])
    ow = cases.out_extent(w, pads[1], pads[3], dilations[1], kw, strides[1])
    ins = [
        m.tensor("X", DT.FLOAT, [n, c, h, w]),
        m.tensor("W", DT.FLOAT, [mm, c // group, kh, kw]),
    ]
    feeds = {
        "X": _RNG.standard_normal((n, c, h, w)).astype(np.float32),
        "W": _RNG.standard_normal((mm, c // group, kh, kw)).astype(np.float32),
    }
    if bias:
        ins.append(m.tensor("B", DT.FLOAT, [mm]))
        feeds["B"] = _RNG.standard_normal(mm).astype(np.float32)
    model = m.make_model(
        "Conv",
        ins,
        [m.tensor("Y", DT.FLOAT, [n, mm, oh, ow])],
        attributes={
            "kernel_shape": list(kernel),
            "strides": list(strides),
            "dilations": list(dilations),
            "pads": list(pads),
            "group": group,
        },
    )
    return model, feeds


@pytest.mark.parametrize("case", cases.CONV_CASES, ids=[c[0] for c in cases.CONV_CASES])
def test_conv_matches_cpu_across_the_attribute_space(case, require_vulkan) -> None:
    """Every attribute axis the proof key does not carry, against ORT's CPU EP.

    `FP32_CONV` is `gen_proof_ledger.py`'s own tolerance, derived in `_models.py` from
    `probe_conv_tolerance.py`'s measurement. It is not `FP32_ELEMENTWISE`, and the reason is at
    the top of `_models.py`: `Conv` accumulates, and accumulation order is a vendor fact.
    """
    model, feeds = _build(case)
    m.check(model, feeds, **m.FP32_CONV)


def test_depthwise_reads_only_its_own_group() -> None:
    """A depthwise convolution with one-hot weights must be a pure channel-wise copy.

    Tolerance-based comparison against ORT can hide a group-indexing error whenever the wrong
    channel happens to hold a similar value. This case makes the failure structural: channel `i`
    is a copy of channel `i` and of nothing else, so reading the wrong group produces a
    different *number*, not a slightly different one, and the assertion is exact.
    """
    c = 4
    x = _RNG.standard_normal((1, c, 5, 5)).astype(np.float32)
    w = np.ones((c, 1, 1, 1), dtype=np.float32)
    model = m.make_model(
        "Conv",
        [m.tensor("X", DT.FLOAT, [1, c, 5, 5]), m.tensor("W", DT.FLOAT, [c, 1, 1, 1])],
        [m.tensor("Y", DT.FLOAT, [1, c, 5, 5])],
        attributes={"kernel_shape": [1, 1], "strides": [1, 1], "dilations": [1, 1],
                    "pads": [0, 0, 0, 0], "group": c},
    )
    feeds = {"X": x, "W": w}
    if not m.is_vulkan_claimed(model, feeds):
        pytest.skip("EP did not claim the depthwise identity case")
    got = m.run_vulkan(model, feeds)[0]
    np.testing.assert_array_equal(got, x)


def test_zero_padding_is_skipped_accumulation_not_a_clamped_read() -> None:
    """A padded border must contribute zero, not a repeat of the edge element.

    With all-ones weights and all-ones input, every output element equals the number of taps
    that landed *inside* the tensor. A kernel that clamped out-of-range reads to the border
    instead of skipping them would return `k*k` everywhere and pass any tolerance check that
    only looked at the interior.
    """
    model = m.make_model(
        "Conv",
        [m.tensor("X", DT.FLOAT, [1, 1, 3, 3]), m.tensor("W", DT.FLOAT, [1, 1, 3, 3])],
        [m.tensor("Y", DT.FLOAT, [1, 1, 3, 3])],
        attributes={"kernel_shape": [3, 3], "strides": [1, 1], "dilations": [1, 1],
                    "pads": [1, 1, 1, 1], "group": 1},
    )
    feeds = {"X": np.ones((1, 1, 3, 3), np.float32), "W": np.ones((1, 1, 3, 3), np.float32)}
    if not m.is_vulkan_claimed(model, feeds):
        pytest.skip("EP did not claim the padding case")
    got = m.run_vulkan(model, feeds)[0]
    # corners see 4 taps, edges 6, the centre 9.
    expected = np.array([[4, 6, 4], [6, 9, 6], [4, 6, 4]], np.float32).reshape(1, 1, 3, 3)
    np.testing.assert_array_equal(got, expected)


# ---------------------------------------------------------------------------
# The declines. Each names the reason the module docstring in `src/ops/conv.rs` gives.
# ---------------------------------------------------------------------------


def test_conv_declines_f16() -> None:
    """`[dtype]`. The f32 module reads one element per word; fp16 storage here is packed."""
    model = m.make_model(
        "Conv",
        [m.tensor("X", DT.FLOAT16, [1, 2, 4, 4]), m.tensor("W", DT.FLOAT16, [3, 2, 3, 3])],
        [m.tensor("Y", DT.FLOAT16, [1, 3, 2, 2])],
        attributes={"kernel_shape": [3, 3], "strides": [1, 1], "dilations": [1, 1],
                    "pads": [0, 0, 0, 0], "group": 1},
    )
    feeds = {
        "X": _RNG.standard_normal((1, 2, 4, 4)).astype(np.float16),
        "W": _RNG.standard_normal((3, 2, 3, 3)).astype(np.float16),
    }
    m.assert_vulkan_does_not_claim(model, feeds)


def test_conv_declines_rank_3() -> None:
    """`[rank]`. 1-D convolution is different index arithmetic, not a narrower loop."""
    model = m.make_model(
        "Conv",
        [m.tensor("X", DT.FLOAT, [1, 2, 8]), m.tensor("W", DT.FLOAT, [3, 2, 3])],
        [m.tensor("Y", DT.FLOAT, [1, 3, 6])],
        attributes={"kernel_shape": [3], "strides": [1], "dilations": [1],
                    "pads": [0, 0], "group": 1},
    )
    feeds = {
        "X": _RNG.standard_normal((1, 2, 8)).astype(np.float32),
        "W": _RNG.standard_normal((3, 2, 3)).astype(np.float32),
    }
    m.assert_vulkan_does_not_claim(model, feeds)


def test_conv_declines_auto_pad() -> None:
    """`[attribute]`. SAME_UPPER derives the pads from an output extent, which is not a fact
    about the node at pipeline-build time the way an explicit `pads` list is."""
    model = m.make_model(
        "Conv",
        [m.tensor("X", DT.FLOAT, [1, 2, 8, 8]), m.tensor("W", DT.FLOAT, [3, 2, 3, 3])],
        [m.tensor("Y", DT.FLOAT, [1, 3, 8, 8])],
        attributes={"kernel_shape": [3, 3], "strides": [1, 1], "dilations": [1, 1],
                    "auto_pad": "SAME_UPPER", "group": 1},
    )
    feeds = {
        "X": _RNG.standard_normal((1, 2, 8, 8)).astype(np.float32),
        "W": _RNG.standard_normal((3, 2, 3, 3)).astype(np.float32),
    }
    m.assert_vulkan_does_not_claim(model, feeds)
