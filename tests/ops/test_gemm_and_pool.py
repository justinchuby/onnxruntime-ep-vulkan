"""`Gemm` and `GlobalAveragePool` conformance — the two ops that finish a CNN's tail.

WHY THIS FILE EXISTS
--------------------
Both ops landed on 2026-08-04 with proof-ledger entries, and for `Gemm` those entries now carry
the transpose form (`ops::common::form`), so `transA`/`transB` are separated by the key rather
than by hope. What the key still does not carry, and what this file is for:

* the **values** inside a form — `alpha`, `beta`, and `C`'s broadcast shape. None of these fork
  a code path, so none of them is a form bit, and a ledger entry proved on `alpha=0.75,
  C=[N]` says nothing about `alpha=1, C=[M,1]`.
* the **structural** properties a tolerance comparison can hide. A transposed read of a square
  matrix produces numbers of the right magnitude in the wrong places; against a random reference
  that fails, but against a nearly-symmetric one it can pass. The exact cases below make those
  failures structural rather than statistical.

DECLINES ARE ASSERTED
---------------------
An op that silently claims a form its kernel does not implement is the wrong-answer class the
op-coverage charter opens with, so every named decline is run and asserted rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pytest
from onnx_ir import DataType as DT

import _models as m

_RNG = np.random.default_rng(0x6E33)


# ---------------------------------------------------------------------------
# Gemm — the value axes no proof key carries
# ---------------------------------------------------------------------------

# (id, M, K, N, transA, transB, alpha, beta, c_shape or None)
_GEMM_CASES = [
    ("base", 3, 5, 4, 0, 0, 1.0, 1.0, [4]),
    ("no_c", 3, 5, 4, 0, 0, 1.0, 1.0, None),
    ("transA", 3, 5, 4, 1, 0, 1.0, 1.0, [4]),
    ("transB", 3, 5, 4, 0, 1, 1.0, 1.0, [4]),
    ("transAB", 3, 5, 4, 1, 1, 1.0, 1.0, [4]),
    # alpha/beta are push-constant values and deliberately not form bits; these are the cases
    # that say so out loud.
    ("alpha_only", 3, 5, 4, 0, 0, 2.5, 0.0, [4]),
    ("beta_only", 3, 5, 4, 0, 0, 0.0, 3.25, [4]),
    ("negative_alpha", 3, 5, 4, 0, 0, -1.5, 1.0, [4]),
    # every legal C broadcast shape
    ("c_scalar", 3, 5, 4, 0, 0, 1.0, 1.0, []),
    ("c_row", 3, 5, 4, 0, 0, 1.0, 1.0, [1, 4]),
    ("c_col", 3, 5, 4, 0, 0, 1.0, 1.0, [3, 1]),
    ("c_full", 3, 5, 4, 0, 0, 1.0, 1.0, [3, 4]),
    # a K of 1 makes the inner loop a single iteration; a K larger than the tile-free kernel's
    # grid is the other end of the same axis.
    ("k_one", 3, 1, 4, 0, 0, 1.0, 1.0, [4]),
    ("wide_k", 2, 257, 3, 0, 1, 1.0, 1.0, [3]),
    # MobileNetV2's own head, at batch 1.
    ("mobilenet_head", 1, 1280, 1000, 0, 1, 1.0, 1.0, [1000]),
]


def _gemm_model(case):
    (_id, mm, k, n, ta, tb, alpha, beta, c_shape) = case
    a_shape = [k, mm] if ta else [mm, k]
    b_shape = [n, k] if tb else [k, n]
    ins = [m.tensor("A", DT.FLOAT, a_shape), m.tensor("B", DT.FLOAT, b_shape)]
    feeds = {
        "A": _RNG.standard_normal(a_shape).astype(np.float32),
        "B": _RNG.standard_normal(b_shape).astype(np.float32),
    }
    attrs = {"alpha": float(alpha), "beta": float(beta), "transA": ta, "transB": tb}
    if c_shape is not None:
        ins.append(m.tensor("C", DT.FLOAT, list(c_shape)))
        feeds["C"] = _RNG.standard_normal(c_shape).astype(np.float32)
    model = m.make_model(
        "Gemm", ins, [m.tensor("Y", DT.FLOAT, [mm, n])], attributes=attrs
    )
    return model, feeds


@pytest.mark.parametrize("case", _GEMM_CASES, ids=[c[0] for c in _GEMM_CASES])
def test_gemm_matches_cpu_across_the_value_space(case, require_vulkan) -> None:
    """`FP32_CONV`'s tolerance, because `Gemm` accumulates and accumulation order is a vendor fact.

    Not `FP32_ELEMENTWISE`: `_models.py` reserves the accumulating ops explicitly, and a K-long
    inner product is the same class of summation `Conv` runs.
    """
    model, feeds = _gemm_model(case)
    m.check(model, feeds, **m.FP32_CONV)


def test_transb_is_a_transpose_and_not_a_relabelling() -> None:
    """A structural check that a tolerance comparison against random data can miss.

    `B` is square and asymmetric with distinct integer entries, `A` is the identity, and `C` is
    absent. `Y` must equal `B.T` exactly. A kernel that ignored `transB` would return `B`, whose
    entries are the same *multiset* — every statistical summary of the two is identical.
    """
    n = 4
    b = np.arange(1, n * n + 1, dtype=np.float32).reshape(n, n)
    a = np.eye(n, dtype=np.float32)
    model = m.make_model(
        "Gemm",
        [
            m.tensor("A", DT.FLOAT, [n, n]),
            m.tensor("B", DT.FLOAT, [n, n]),
            m.tensor("C", DT.FLOAT, [n]),
        ],
        [m.tensor("Y", DT.FLOAT, [n, n])],
        attributes={"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 1},
    )
    feeds = {"A": a, "B": b, "C": np.zeros(n, dtype=np.float32)}
    if not m.is_vulkan_claimed(model, feeds):
        pytest.skip("EP did not claim the transB identity case")
    got = m.run_vulkan(model, feeds)[0]
    np.testing.assert_array_equal(got, b.T)
    assert not np.array_equal(got, b), "the case must discriminate, or it proves nothing"


def test_a_broadcast_column_c_is_added_down_the_rows() -> None:
    """`C` of shape `[M, 1]` must repeat across columns, not across rows.

    The two broadcast directions are indistinguishable when `M == N`, so this case uses a
    non-square output and integer values that make the correct answer unique.
    """
    mm, k, n = 3, 2, 4
    a = np.zeros((mm, k), dtype=np.float32)
    b = np.zeros((k, n), dtype=np.float32)
    c = np.array([[10.0], [20.0], [30.0]], dtype=np.float32)
    model = m.make_model(
        "Gemm",
        [
            m.tensor("A", DT.FLOAT, [mm, k]),
            m.tensor("B", DT.FLOAT, [k, n]),
            m.tensor("C", DT.FLOAT, [mm, 1]),
        ],
        [m.tensor("Y", DT.FLOAT, [mm, n])],
        attributes={"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 0},
    )
    feeds = {"A": a, "B": b, "C": c}
    if not m.is_vulkan_claimed(model, feeds):
        pytest.skip("EP did not claim the column-broadcast case")
    got = m.run_vulkan(model, feeds)[0]
    np.testing.assert_array_equal(got, np.tile(c, (1, n)))


def test_gemm_declines_f16() -> None:
    """`[dtype]`. The f32 module reads one element per word; fp16 storage here is packed."""
    model = m.make_model(
        "Gemm",
        [m.tensor("A", DT.FLOAT16, [3, 5]), m.tensor("B", DT.FLOAT16, [5, 4])],
        [m.tensor("Y", DT.FLOAT16, [3, 4])],
        attributes={"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 0},
    )
    feeds = {
        "A": _RNG.standard_normal((3, 5)).astype(np.float16),
        "B": _RNG.standard_normal((5, 4)).astype(np.float16),
    }
    m.assert_vulkan_does_not_claim(model, feeds)


def test_gemm_declines_f64() -> None:
    """`[dtype]`. Chosen over the non-broadcastable-`C` case, which is *unreachable*.

    I wrote that case first and it failed — not because the EP claimed it, but because ORT's own
    `Gemm` kernel rejects `C=[M]` against `[M, N]` with `M != N` before any EP sees it. There is
    therefore no valid graph in which our `c_broadcast()` rejection is the thing that declines,
    and a conformance test cannot reach it; `ops::matmul`'s unit tests own that predicate
    instead. `double` is a real dtype the CPU EP runs and we do not, so the decline is live.
    """
    model = m.make_model(
        "Gemm",
        [m.tensor("A", DT.DOUBLE, [3, 5]), m.tensor("B", DT.DOUBLE, [5, 4])],
        [m.tensor("Y", DT.DOUBLE, [3, 4])],
        attributes={"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 0},
    )
    feeds = {
        "A": _RNG.standard_normal((3, 5)).astype(np.float64),
        "B": _RNG.standard_normal((5, 4)).astype(np.float64),
    }
    m.assert_vulkan_does_not_claim(model, feeds)


# ---------------------------------------------------------------------------
# GlobalAveragePool
# ---------------------------------------------------------------------------

_POOL_CASES = [
    ("mobilenet", 1, 1280, 7, 7),
    ("batch", 3, 8, 5, 4),
    ("one_by_one", 2, 6, 1, 1),
    ("tall", 1, 3, 33, 1),
    # a window longer than the workgroup, so the serial reduction is not trivially short
    ("long_window", 1, 2, 17, 19),
]


@pytest.mark.parametrize("case", _POOL_CASES, ids=[c[0] for c in _POOL_CASES])
def test_global_average_pool_matches_cpu(case, require_vulkan) -> None:
    (_id, n, c, h, w) = case
    model = m.make_model(
        "GlobalAveragePool",
        [m.tensor("X", DT.FLOAT, [n, c, h, w])],
        [m.tensor("Y", DT.FLOAT, [n, c, 1, 1])],
    )
    feeds = {"X": _RNG.standard_normal((n, c, h, w)).astype(np.float32)}
    m.check(model, feeds, **m.FP32_CONV)


def test_the_pool_reduces_its_own_channel_and_not_its_neighbour() -> None:
    """Channel `i` is filled with the constant `i`, so `Y[0, i] == i` exactly.

    A base-index error that read the neighbouring channel would return `i+1` — a different
    number, not a slightly different one, so the assertion is exact and the failure is
    structural rather than statistical.
    """
    n, c, h, w = 1, 6, 3, 4
    x = np.empty((n, c, h, w), dtype=np.float32)
    for i in range(c):
        x[0, i] = float(i)
    model = m.make_model(
        "GlobalAveragePool",
        [m.tensor("X", DT.FLOAT, [n, c, h, w])],
        [m.tensor("Y", DT.FLOAT, [n, c, 1, 1])],
    )
    feeds = {"X": x}
    if not m.is_vulkan_claimed(model, feeds):
        pytest.skip("EP did not claim the channel-identity pool case")
    got = m.run_vulkan(model, feeds)[0]
    np.testing.assert_array_equal(got.reshape(c), np.arange(c, dtype=np.float32))


def test_pool_declines_f16() -> None:
    """`[dtype]`. Same packed-`uint` argument as `Conv` and `Gemm`."""
    model = m.make_model(
        "GlobalAveragePool",
        [m.tensor("X", DT.FLOAT16, [1, 4, 3, 3])],
        [m.tensor("Y", DT.FLOAT16, [1, 4, 1, 1])],
    )
    feeds = {"X": _RNG.standard_normal((1, 4, 3, 3)).astype(np.float16)}
    m.assert_vulkan_does_not_claim(model, feeds)


def test_pool_declines_rank_3() -> None:
    """`[rank]`. ONNX allows `[N, C, D1..Dn]`; this kernel implements the 4-D case only."""
    model = m.make_model(
        "GlobalAveragePool",
        [m.tensor("X", DT.FLOAT, [1, 4, 9])],
        [m.tensor("Y", DT.FLOAT, [1, 4, 1])],
    )
    feeds = {"X": _RNG.standard_normal((1, 4, 9)).astype(np.float32)}
    m.assert_vulkan_does_not_claim(model, feeds)
