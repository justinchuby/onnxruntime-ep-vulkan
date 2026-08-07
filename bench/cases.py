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
    (``DESIGN.md`` §11.1). It is also, per ``OP_COVERAGE.md`` §4.18, **the only op both LLM
    toolchains agree on** — mobius and the ORT GenAI builder emit different attention and
    normalisation ops but the same ``MatMulNBits``. That makes it the one case whose cost carries
    across producers, which is a good reason for the bar to sit here and a bad reason to assume
    anything else does.
    The oracle knob ``accuracy_level`` is pinned to 1 by the builder's
    default — see Trinity's rule: any knob ORT selects by sniffing the host CPU must be pinned,
    or the reference drifts across machines and the drift looks like our bug.

``transfer``
    Not an op benchmark. A staircase of byte sizes whose purpose is to feed
    ``TransferModel::fit`` in ``rust/src/ops/partition.rs`` and replace the placeholder MVS
    constants (``SAFETY = 3.0``, 64 KiB output floor) with measured ones. See
    ``transfer_calibration.py``.

Producer provenance
-------------------

Every case carries a :class:`producers.Producer`. Today that is always the synthetic op builder,
which is an ``op``-kind producer and therefore **cannot** name a model family: constructing a case
called ``qwen3_decoder_layer`` from it raises :class:`producers.ProducerProvenanceError` before any
timing happens. See ``producers.py`` for why — a benchmark artefact is relative to its producer in
exactly the way op coverage is (``OP_COVERAGE.md`` §4.18), and the two producers we care about
(Justin's ``mobius`` and the ORT GenAI builder) emit *different op sets* for the same architecture.

When a real model case is added it must be added **per producer** — ``qwen3_decoder_mobius`` and
``qwen3_decoder_ortgenai`` are two cases, not one case run two ways, because they are two graphs.

Nothing in this file has ever run on a GPU. See ``DESIGN.md`` §9.1.2.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# The op-test model builders are the single source of graph truth.
# The path entry is scoped to the import and then removed: `sys.path` is process-global, and a
# leaked `tests/ops` entry silently decides every later flat import in the process. See
# `bench/test_import_isolation.py` for the screen that found this.
_TESTS_OPS = Path(__file__).resolve().parents[1] / "tests" / "ops"
_SYS_PATH_BEFORE = list(sys.path)
if str(_TESTS_OPS) not in sys.path:
    sys.path.insert(0, str(_TESTS_OPS))

try:
    import onnx_ir as ir  # noqa: E402  (import after sys.path fix-up)

    import _models  # noqa: E402
finally:
    sys.path[:] = _SYS_PATH_BEFORE

import producers  # noqa: E402


@dataclass
class Case:
    """One benchmark case: a model, its feeds, and what it is supposed to tell us."""

    name: str
    group: str
    model: bytes
    feeds: "dict[str, np.ndarray]"
    #: Who built the graph. Not optional: a timing whose graph has no known origin is not
    #: reproducible, and a case cannot be named after a model family its producer did not export.
    producer: "producers.Producer | None" = None
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

    def __post_init__(self) -> None:
        if self.producer is None:
            self.producer = producers.op_builder()
        # Fatal, at construction time, before any timing exists to be mislabelled.
        producers.assert_family_label_is_earned(self.name, self.tags, self.producer)



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


def _matmulnbits_case(
    K: int, N: int, *, block_size: int = 32, anchor: bool = False, rows: int = 1
) -> Case:
    model, feeds = _models.make_matmulnbits_model(K, N, block_size=block_size, rows=rows)
    # M=1 is decode and keeps its historical name so its series stays comparable across the
    # whole results archive; M>1 is prefill and is named separately (issue #7).
    name = f"matmulnbits_q4_b{block_size}_K{K}_N{N}"
    if rows != 1:
        name += f"_M{rows}"
    return Case(
        name=name,
        group="gemm",
        model=model,
        feeds=feeds,
        note=(
            "OQ-12 anchor: the ≥1.5x-over-CPU bar is measured here (DESIGN.md §11.1). "
            "accuracy_level pinned to 1 by the builder."
            if anchor
            else (
                "prefill: M rows share one pass over the packed weights (issue #7). "
                "accuracy_level pinned to 1 by the builder."
                if rows != 1
                else "accuracy_level pinned to 1 by the builder (Trinity's oracle-pinning rule)."
            )
        ),
        # [M,K] x [K,N] GEMM: one multiply-add per weight element per row.
        flops=2 * K * N * rows,
        # Weights are initializers and are uploaded once, not per inference; only the
        # activation in and the result out cross the boundary each run.
        boundary_bytes=(K + N) * 4 * rows,
        oq12_anchor=anchor,
        tags=["gemm", "quantized"] + (["prefill"] if rows != 1 else ["decode"]),
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

    # Prefill (issue #7). The row tile is a *weight-traffic* change, so the only way to see it in
    # wall-clock is to compare the same shape at several M: an untiled GEMV costs M passes over
    # the packed weights, a tiled one costs ceil(M/QB_ROWS). M=1 above is the control that must
    # not regress; these are the rows that should get cheaper per row. M=5 is here and not just
    # 2 and 4 because a partial tile is where a row tile is most likely to be wrong or slow.
    for m_rows in (2, 4, 5, 8):
        cases.append(_matmulnbits_case(4096, 4096, rows=m_rows))

    if groups:
        cases = [c for c in cases if c.group in groups]
    return cases


def case_producers(cases: "list[Case]") -> "list[producers.Producer]":
    """Distinct producers across ``cases``, in first-seen order.

    Recorded in the result file next to device, driver, OS and build flags, because a graph's
    origin is part of the environment a number was taken in.
    """
    seen: "list[producers.Producer]" = []
    for c in cases:
        if c.producer is not None and all(
            p.fingerprint != c.producer.fingerprint for p in seen
        ):
            seen.append(c.producer)
    return seen


def transfer_staircase(max_log2: int = 24) -> "list[int]":
    """Byte sizes for the transfer calibration staircase: 1 KiB … 16 MiB, doubling.

    Doubling (rather than a linear sweep) because the model being fitted is affine —
    ``fixed_ns + bytes / bytes_per_ns`` — and a log-spaced staircase gives the fit both the
    fixed-cost regime (small sizes, where ``fixed_ns`` dominates) and the bandwidth regime
    (large sizes) with the same number of samples.
    """
    return [1 << e for e in range(10, max_log2 + 1)]


if __name__ == "__main__":  # pragma: no cover - manual use
    built = build_cases()
    for p in case_producers(built):
        print(f"producer     {p.summary()}")
    for c in built:
        print(f"{c.group:12s} {c.name:36s} flops={c.flops} boundary_bytes={c.boundary_bytes}")
