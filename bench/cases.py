"""Benchmark cases — built from the *same* model builders the correctness tests use.

Every case here calls into ``tests/ops/_models.py``. That is deliberate and it is the same rule
the MLX EP's ``bench/cases.py`` follows: a benchmark that builds its own graphs can drift from
what is tested, and then measures something no test ever checked. If a builder changes shape,
these cases change with it or fail loudly.

Case groups
-----------

``elementwise``
    Dispatch-bound by construction. On a discrete GPU these will be **slower than the CPU EP**
    at small sizes and that is not a bug: a single elementwise op pays a submit, a fence wait
    and two PCIe transfers to do work the CPU finishes in microseconds. They are here to track
    *overhead*, and the size staircase is what makes the crossover point visible.

``gemm``
    ``MatMulNBits``, the GEMM-anchored case OQ-12's ≥1.5×-over-CPU bar is defined on
    (``DESIGN.md`` §11.1). The oracle knob ``accuracy_level`` is pinned to 1 by the builder's
    default — see Trinity's rule: any knob ORT selects by sniffing the host CPU must be pinned,
    or the reference drifts across machines and the drift looks like our bug.

``transfer``
    Not an op benchmark. A staircase of byte sizes whose purpose is to feed
    ``TransferModel::fit`` in ``rust/src/ops/partition.rs`` and replace the placeholder MVS
    constants (``SAFETY = 3.0``, 64 KiB output floor) with measured ones. See
    ``transfer_calibration.py``.

Nothing in this file has ever run on a GPU. See ``DESIGN.md`` §9.1.2.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# The op-test model builders are the single source of graph truth.
_TESTS_OPS = Path(__file__).resolve().parents[1] / "tests" / "ops"
if str(_TESTS_OPS) not in sys.path:
    sys.path.insert(0, str(_TESTS_OPS))

import onnx_ir as ir  # noqa: E402  (import after sys.path fix-up)

import _models  # noqa: E402


@dataclass
class Case:
    """One benchmark case: a model, its feeds, and what it is supposed to tell us."""

    name: str
    group: str
    model: bytes
    feeds: "dict[str, np.ndarray]"
    #: Free-form note printed with the result — e.g. why a case is expected to lose to CPU.
    note: str = ""
    #: FLOPs per inference, when the case has a defensible count. Used to report achieved
    #: FLOP/s alongside latency; ``None`` when a count would be made up.
    flops: "int | None" = None
    #: Bytes that must cross the host/device boundary per inference (inputs + outputs).
    #: This is the ``boundary_bytes_per_inference`` the partition cost model is minimising.
    boundary_bytes: int = 0
    #: True for the case OQ-12's ≥1.5× bar is measured on.
    oq12_anchor: bool = False
    tags: "list[str]" = field(default_factory=list)


def _fp32(shape: "tuple[int, ...]", seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def _binary_case(op_type: str, shape: "tuple[int, ...]", *, note: str = "") -> Case:
    dims = list(shape)
    a = _models.tensor("A", ir.DataType.FLOAT, dims)
    b = _models.tensor("B", ir.DataType.FLOAT, dims)
    c = _models.tensor("C", ir.DataType.FLOAT, dims)
    model = _models.make_model(op_type, [a, b], [c])
    n = int(np.prod(dims))
    feeds = {"A": _fp32(shape, 1), "B": _fp32(shape, 2)}
    return Case(
        name=f"{op_type.lower()}_fp32_{'x'.join(str(d) for d in dims)}",
        group="elementwise",
        model=model,
        feeds=feeds,
        note=note,
        flops=n,  # one op per element; an honest, boring count
        boundary_bytes=n * 4 * 3,  # two inputs in, one output back
        tags=["dispatch-bound"] if n <= 1 << 16 else [],
    )


def _matmulnbits_case(K: int, N: int, *, block_size: int = 32, anchor: bool = False) -> Case:
    model, feeds = _models.make_matmulnbits_model(K, N, block_size=block_size)
    return Case(
        name=f"matmulnbits_q4_b{block_size}_K{K}_N{N}",
        group="gemm",
        model=model,
        feeds=feeds,
        note=(
            "OQ-12 anchor: the ≥1.5x-over-CPU bar is measured here (DESIGN.md §11.1). "
            "accuracy_level pinned to 1 by the builder."
            if anchor
            else "accuracy_level pinned to 1 by the builder (Trinity's oracle-pinning rule)."
        ),
        # [1,K] x [K,N] GEMV: one multiply-add per weight element.
        flops=2 * K * N,
        # Weights are initializers and are uploaded once, not per inference; only the
        # activation in and the result out cross the boundary each run.
        boundary_bytes=(K + N) * 4,
        oq12_anchor=anchor,
        tags=["gemm", "quantized"],
    )


def build_cases(groups: "list[str] | None" = None) -> "list[Case]":
    """Build every case, or only those in ``groups``."""
    cases: "list[Case]" = []

    # Elementwise size staircase. The interesting output is not any single row but where the
    # Vulkan line crosses the CPU line — below that size the EP should decline the work, and
    # the MVS policy in partition.rs is what has to learn that number from these measurements.
    for shape in [(1024,), (256, 256), (1024, 1024), (4096, 1024)]:
        cases.append(
            _binary_case(
                "Add",
                shape,
                note="dispatch-bound at small sizes; expected slower than CPU there",
            )
        )
    cases.append(_binary_case("Mul", (1024, 1024)))

    # GEMM-anchored quantized cases. K=4096/N=4096 is the shape a Qwen3-class decoder's
    # projections actually use; the smaller pair is where dispatch overhead still shows.
    cases.append(_matmulnbits_case(1024, 1024))
    cases.append(_matmulnbits_case(4096, 4096, anchor=True))

    if groups:
        cases = [c for c in cases if c.group in groups]
    return cases


def transfer_staircase(max_log2: int = 24) -> "list[int]":
    """Byte sizes for the transfer calibration staircase: 1 KiB … 16 MiB, doubling.

    Doubling (rather than a linear sweep) because the model being fitted is affine —
    ``fixed_ns + bytes / bytes_per_ns`` — and a log-spaced staircase gives the fit both the
    fixed-cost regime (small sizes, where ``fixed_ns`` dominates) and the bandwidth regime
    (large sizes) with the same number of samples.
    """
    return [1 << e for e in range(10, max_log2 + 1)]


if __name__ == "__main__":  # pragma: no cover - manual use
    for c in build_cases():
        print(f"{c.group:12s} {c.name:36s} flops={c.flops} boundary_bytes={c.boundary_bytes}")
