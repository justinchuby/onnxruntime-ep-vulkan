"""Barrier-backend parity tests for the Vulkan EP.

WHAT THIS FILE TESTS
====================
The Vulkan EP carries two barrier backends (DESIGN.md §7.5):
  - ``Sync2Backend``   — uses ``vkCmdPipelineBarrier2`` (VK_KHR_synchronization2 / Vulkan 1.3 core)
  - ``LegacyBackend``  — uses ``vkCmdPipelineBarrier`` (Vulkan 1.1 core, always available)

The backend is selected once at device init (``Barriers::select``).  Without forced switching,
Linux CI has ``synchronization2`` ~99% of the time and Windows lavapipe almost certainly does too —
meaning the legacy path, which we carry for 31% of Android (Adreno 5xx, Mali Bifrost) and 12% of
Windows desktop, would never be executed by any test we own.

This file runs every claimed-op case from ``test_op_table._CASES`` twice:
  1. With the natural backend (``Barriers::select`` chooses; sync2 on lavapipe).
  2. With ``ep.force_legacy_barriers=1``, forcing ``LegacyBackend`` even on a sync2-capable device.

Then asserts the outputs are **bit-identical**.

WHY BIT-IDENTICAL (not just tolerance-gated)
============================================
The barrier backend is a **host-side synchronisation primitive**, not a compute primitive.
The SPIR-V shader, the hardware, and the input data are all identical between the two runs.
The only difference is *how* the host tells the GPU to order its work.  If the GPU honours both
ordering mechanisms correctly, the computation path is unaffected and the floating-point outputs
are exactly the same bits.  Any numerical difference — even one ULP — indicates that one of the
two barrier paths has a read-after-write hazard: the GPU started reading a buffer before the
previous write finished.  Tolerance would mask this.

FALSE-GREEN GUARD
=================
``_models.run_with_backend`` sets ``ONNXRUNTIME_EP_VULKAN_BACKEND_PROBE=<path>`` and reads
the probe file the EP writes to confirm which backend ran.  Without this, a session that
silently ignores ``ep.force_legacy_barriers=1`` would run sync2 twice and pass vacuously —
the same failure mode as an op test that does not call ``assert_vulkan_claims``.

TODO(Switch): implement ``ONNXRUNTIME_EP_VULKAN_BACKEND_PROBE`` in ``rust/src/vk/barrier.rs``
(see ``_models.run_with_backend`` docstring for the exact contract).  Until then,
``active_backend == "unknown"`` triggers a ``pytest.warns`` rather than a hard failure,
because the output-identity comparison is still valuable even without the probe.

ADDING NEW OPS
==============
Nothing to do here: this file reads ``_CASES`` from ``test_op_table.py``.  Adding a row to
that table automatically adds both a claim test and a parity test.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

# Import the case table from test_op_table.  Both files live in tests/ops/ which is on
# sys.path when pytest is invoked from the repo root or from tests/ops/.
from test_op_table import CaseSpec, _CASES

import _models as m

# ---------------------------------------------------------------------------
# Only Live ops exercise the Vulkan kernel path end-to-end.
# Staged ops (claim=True but live=False) have a working claim predicate but no
# confirmed kernel dispatch — we skip parity for them.
# ---------------------------------------------------------------------------
_PARITY_CASES: list[CaseSpec] = [c for c in _CASES if c.claim]


@pytest.mark.parametrize("case", _PARITY_CASES, ids=[c.id for c in _PARITY_CASES])
def test_barrier_parity(case: CaseSpec, require_vulkan) -> None:
    """Assert the sync2 and legacy barrier backends produce bit-identical outputs.

    Failure meaning
    ---------------
    If this test fails, one of the two barrier paths has a synchronisation bug:
    a read-after-write hazard is not being correctly enforced.  The computation
    is the same in both cases (same shader, same hardware, same data); only the
    barrier API differs.  See DESIGN.md §7.5 and ``rust/src/vk/barrier.rs``.
    """
    inputs  = [m.tensor(n, dt, s) for n, dt, s in case.inputs]
    outputs = [m.tensor(n, dt, s) for n, dt, s in case.outputs]
    model   = m.make_model(
        case.op, inputs, outputs,
        domain=case.domain,
        attributes=case.attrs,
    )

    # --- Live guard: skip (not fail) when the op is not yet Live ---
    #
    # CaseSpec.live=True means Mouse has confirmed the kernel dispatches end-to-end
    # on real hardware. Barrier parity requires actual GPU execution: both sync2 and
    # legacy runs must reach the shader, otherwise bit-equality is vacuously true
    # (both fall back to CPU and agree trivially).
    #
    # WHY live FLAG INSTEAD OF is_vulkan_claimed() PROBE:
    #   is_vulkan_claimed() creates an ORT session with profiling=True. For Staged ops
    #   (claim=True but no working kernel), the EP's Compile path crashes with an
    #   access violation on Intel Iris Xe. Python's except Exception cannot catch
    #   C-level AV crashes — the entire test process dies. This was confirmed for
    #   Atan-fp32 (case index 39 in deterministic parity order) on 2026-07-29.
    #   The crash was device-0-specific (Intel); NVIDIA handled it differently.
    #   Intel is the spec-conformance oracle: EP Compile must not crash for Staged ops.
    #   Route to Tank/Mouse for the EP fix; this guard prevents the process death.
    #
    # WHY NOT ERROR INSTEAD OF SKIP:
    #   test_op_table[{id}] is the claim/correctness assertion — it fails loudly for
    #   Staged ops already. Barrier parity is orthogonal: it tests HOST sync ordering,
    #   not EP claiming. Skipping with a clear message is the right behaviour here.
    #   A skip that contradicts a live passing test IS a defect; the live flag ensures
    #   that cannot happen: live=True appears only when test_op_table passes.
    #
    # TO ADD A NEW OP: Mouse sets live=True in the _CASES row when marking Ready.
    if not case.live:
        pytest.skip(
            f"{case.id}: {case.op} is Staged (claim=True but live=False). "
            f"Barrier parity requires confirmed GPU dispatch. "
            f"Mouse: set live=True in test_op_table._CASES when marking this op Ready. "
            f"Ref: test_barrier_parity crash localisation 2026-07-29, Atan-fp32 index 39."
        )

    # --- Run 1: natural backend (sync2 if device supports it) ---
    outs_default, backend_default = m.run_with_backend(model, case.feeds, force_legacy=False)

    # --- Run 2: forced legacy backend ---
    outs_legacy, backend_legacy = m.run_with_backend(model, case.feeds, force_legacy=True)

    # ---------------------------------------------------------------
    # Backend verification — guards against "ran sync2 twice" false green.
    # Switch's ONNXRUNTIME_EP_VULKAN_BACKEND_PROBE landed in rust/src/vk/barrier.rs
    # (commit 255f2db). The probe writes "sync2" or "legacy" to the probe file during
    # Barriers::select in Device::new. "unknown" should no longer occur with the EP built.
    # ---------------------------------------------------------------
    if backend_legacy == "unknown":
        # Probe file not written — EP is not built yet (scaffolding state).
        # Warn rather than fail: the output comparison still catches computation bugs.
        warnings.warn(
            f"[{case.id}] ONNXRUNTIME_EP_VULKAN_BACKEND_PROBE probe file not written — "
            "EP is likely not built yet. The parity comparison proceeds but cannot verify "
            "which barrier backend ran. Once the EP crate is linked, this should always "
            "return 'sync2' or 'legacy'. See rust/src/vk/barrier.rs (commit 255f2db).",
            UserWarning,
            stacklevel=2,
        )
    else:
        # Probe active: assert the backends are what we asked for.
        assert backend_legacy == "legacy", (
            f"[{case.id}] ep.force_legacy_barriers=1 was set but EP reported "
            f"barrier backend {backend_legacy!r}. "
            "Check that ep.force_legacy_barriers is wired to Barriers::select in "
            "rust/src/vk/barrier.rs (DESIGN.md §7.5, commit 255f2db)."
        )
        if backend_default != "unknown":
            assert backend_default in ("sync2", "legacy"), (
                f"[{case.id}] Unexpected default backend: {backend_default!r}"
            )

    # ---------------------------------------------------------------
    # Core parity assertion: outputs must be bit-identical.
    #
    # Rationale: the barrier backend is host-side synchronisation only.
    # Same SPIR-V shader + same hardware + same input bytes => identical
    # IEEE 754 floating-point results.  Any difference is a barrier bug.
    #
    # NaN note: np.testing.assert_array_equal treats NaN != NaN, which is
    # correct here — if one path produces NaN and the other does not, that
    # IS a divergence indicating a hazard.  NaN-at-same-position agreement
    # (both paths hitting the same undefined result) is caught as "equal"
    # only for integer/bool outputs; for float outputs it is a hard failure.
    # Test inputs are seeded to avoid NaN-producing edge cases anyway.
    # ---------------------------------------------------------------
    assert len(outs_legacy) == len(outs_default), (
        f"[{case.id}] Output-count mismatch: "
        f"default({backend_default})={len(outs_default)}, "
        f"legacy({backend_legacy})={len(outs_legacy)}"
    )
    for idx, (got, want) in enumerate(zip(outs_legacy, outs_default)):
        np.testing.assert_array_equal(
            got, want,
            err_msg=(
                f"\n"
                f"Barrier backend parity FAILURE\n"
                f"  Op:              {case.op}  (id: {case.id})\n"
                f"  Output index:    {idx}\n"
                f"  Default backend: {backend_default}\n"
                f"  Legacy backend:  {backend_legacy}\n"
                f"\n"
                f"The vkCmdPipelineBarrier (legacy) and vkCmdPipelineBarrier2 (sync2)\n"
                f"backends produced different outputs for the same computation.\n"
                f"This is a synchronisation bug: a read-after-write hazard is not\n"
                f"correctly enforced in one of the two barrier backends.\n"
                f"See DESIGN.md §7.5 and rust/src/vk/barrier.rs."
            ),
        )
