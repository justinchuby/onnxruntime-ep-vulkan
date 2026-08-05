"""Which world are we in? ORT permits a device-resident KV cache; our writeback path does not.

The question
------------
The KV cache crosses the host<->device boundary every token, and past ctx 2048 that transfer —
not DRAM — is the floor (Niobe: crossover rates 70.9 / 111.1 / 154.9 GB/s, every one above any
link on this machine).  Round 2 of this thread found that ORT *does* allocate our fused-node
outputs through this EP's device allocator (195/195 device-resident), but that binding them
EP-side returns zeros.  The open question the coordinator asked for a ruling on was whether
ORT's contract permits an EP-owned KV cache at all — with the standing instruction that
*"ORT's KV round-trip is structural for plugin EPs"* would be a valuable result, not a failure.

The answer, measured
---------------------
**It is not structural, and it is not ORT's.**  A caller can allocate an `OrtValue` in this
EP's device memory and bind it as a graph output, and the results come back bit-identical to
the same session run unbound.

===============================  =========================  =========================
lane                             caller-side bind only      + EP-side Step 1c bind
===============================  =========================  =========================
alloc_device_frame               SHARED                     SHARED
outputs_device_resident          6                          0
outputs_device_bound             0                          6
nonzero elements returned        256 / 320 / 320            0 / 0 / 0
rel vs the *unbound EP*          0.0 / 0.0 / 0.0            1.0 / 1.0 / 1.0
verdict                          KV_CAN_STAY_DEVICE_RESIDENT DEVICE_BOUND_OUTPUTS_RETURN_NOTHING
===============================  =========================  =========================

So the remaining obstacle is one named invariant and it is ours: `transfer.rs` documents that a
device-backed span's **host staging block stays authoritative and the device buffer is a
mirror**.  When the EP writes the device buffer directly (Step 1c), nothing makes it
authoritative, so the read still comes from a staging block nothing wrote.  That is an EP-side
fix in our own allocator, not a limitation of the runtime.

Three things this file exists to keep true
--------------------------------------------
1. **The criterion is `ep` vs `bound`, not `cpu` vs `bound`.**  The EP disagrees with the CPU
   reference by 0.11 on `attn_out` in this case *with or without* binding — that is fp16
   arithmetic and it is not this question.  Scoring against `cpu` conflates the two, and round 2
   passed an all-zero result doing exactly that.
2. **A relative metric needs a degeneracy guard.**  Two all-zero tensors agree perfectly.  The
   `epbind` lane above scores `1.0` — the signature of zeros against a saturating denominator
   floor — and is caught by the nonzero count, not by the score.
3. **Ordering is load-bearing.**  Asking the EP for an allocator *before* the session exists
   builds a second `VkDevice`: `alloc_device_frame = SPLIT-DEVICE`, which no compute dispatch
   can bind.  The probe refused itself for this on its first run.  Any arena inherits the
   hazard, and a caller that gets the order wrong holds bytes the kernels can never reach.

And one thing that is *not* claimed
------------------------------------
The round trip is **not** removed yet.  In the working lane the KV bytes still go
device -> host staging -> ORT's device buffer; what is established is that the caller can hold
the KV in EP device memory across `run()` calls, which is the precondition.  `readback_bytes`
is not quoted here because it is not yet expected to have moved.
"""

from __future__ import annotations

import json
import os
import pathlib

import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
CALLER = REPO / "bench" / "results" / "kv_device_residency-callerbind.json"
EPBIND = REPO / "bench" / "results" / "kv_device_residency-epbind.json"


def _load(p: pathlib.Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.mark.skipif(not CALLER.is_file(), reason="no recorded caller-bind reading")
def test_ort_permits_a_device_resident_kv_cache() -> None:
    """The ruling: 'the runtime forbids it' is refuted."""
    d = _load(CALLER)
    assert d["verdict"] == "KV_CAN_STAY_DEVICE_RESIDENT", d.get("why")
    assert d["q1_addressable"]["by_ep_memory_info"] == "OK"
    assert d["counters"]["alloc_device_frame"] == "SHARED"
    assert d["counters"]["compute_failures"] == 0
    assert d["counters"]["device_losses"] == 0


@pytest.mark.skipif(not CALLER.is_file(), reason="no recorded caller-bind reading")
def test_binding_does_not_change_the_answer() -> None:
    """`ep` vs `bound`: same kernel, same inputs, only the writeback path differs."""
    d = _load(CALLER)
    assert max(d["q4_rel_vs_unbound_ep"].values()) == 0.0
    assert all(n > 0 for n in d["nonzero_elements_returned"].values()), (
        "the degeneracy guard: an all-zero result agrees with anything under a relative "
        "metric with a denominator floor"
    )


@pytest.mark.skipif(not CALLER.is_file(), reason="no recorded caller-bind reading")
def test_the_eps_gap_against_cpu_is_not_attributed_to_binding() -> None:
    """0.11 on attn_out is fp16 arithmetic in this case and is present in both lanes.

    Recorded rather than asserted away: a probe that had scored `cpu` vs `bound` would have
    called this a binding defect, and a probe that had loosened its tolerance to absorb it
    would have absorbed a real one too.
    """
    d = _load(CALLER)
    assert d["q4_unbound_ep_rel_vs_cpu"] == d["q4_rel_vs_cpu"]
    assert d["q4_unbound_ep_rel_vs_cpu"]["present_key"] == 0.0
    assert d["q4_unbound_ep_rel_vs_cpu"]["present_value"] == 0.0


@pytest.mark.skipif(not EPBIND.is_file(), reason="no recorded ep-bind reading")
def test_the_ep_side_bind_now_returns_the_right_bytes() -> None:
    """The obstacle was ours and it is closed.

    # This assertion was inverted on 2026-08-02, deliberately, and the old text is kept

    As written the previous round this test asserted `DEVICE_BOUND_OUTPUTS_RETURN_NOTHING` and
    `all(n == 0 ...)` — it pinned the *defect*. Host staging was authoritative, the device buffer
    was a mirror, and nothing made a directly-written device buffer authoritative, so a bound
    output came back as zeros. Marking the bound span device-authoritative closed that, and this
    test went red on the fix.

    That is the trap this project has already named once: *a falsifier that asserts the exact
    value of a number it does not own goes red on a fix.* Trinity's liveness control survived my
    push-constant change because she asserted `> 0` and `!= clean` rather than `== 14`. I asserted
    the defect itself, so there was no version of this that survived. The honest repair is to
    re-point it at the property that should now hold — and to say in the file that it moved, since
    a silently rewritten assertion is indistinguishable from one that never fired.
    """
    d = _load(EPBIND)
    assert d["verdict"] == "KV_CAN_STAY_DEVICE_RESIDENT"
    assert d["counters"]["outputs_device_bound"] > 0, (
        "if the bind never fired, agreement would be about the unbound path and this lane would "
        "be measuring nothing"
    )
    assert all(n > 0 for n in d["nonzero_elements_returned"].values()), (
        "the degeneracy guard, kept: all-zero outputs used to pass this lane's score perfectly"
    )
    assert max(d["q4_rel_vs_unbound_ep"].values()) == 0.0, (
        "binding the output must not change the answer; it agreed to the digit when captured"
    )


@pytest.mark.skipif(not (CALLER.is_file() and EPBIND.is_file()), reason="need both lanes")
def test_the_two_lanes_still_differ_in_what_they_exercise() -> None:
    """A control whose reading does not move with its input is a falsifier that cannot fire.

    The lanes used to differ in their *verdict*, because one of them was broken. Now both reach
    `KV_CAN_STAY_DEVICE_RESIDENT`, which is the point of the change — so the discriminator has to
    be something that is still true: the EP-side lane binds outputs and the caller-side lane does
    not. If that ever stops differing, the two lanes are one lane run twice.
    """
    a, b = _load(CALLER), _load(EPBIND)
    assert a["counters"]["outputs_device_bound"] == 0
    assert b["counters"]["outputs_device_bound"] > 0
    assert a["counters"]["outputs_device_bound"] != b["counters"]["outputs_device_bound"]


# ci/open_reds.json ("kv_caller_bind_reading", owner Switch) tracked this claim as an
# unresolved artifact/assertion disagreement: the assertion below used to read CALLER
# unconditionally, quoting "CUDA" against whatever vendor happened to be in that file. At
# 872d739 CALLER was RE-CAPTURED on a different physical device (Intel, vendor 0x8086,
# device_index 0) than the ed48f5b capture that established the claim (NVIDIA, vendor
# 0x10de, device_index 1) -- a different `vendor_id` argument to the exact same ORT call,
# which is a different question, not a re-answer of the same one. `git diff ed48f5b 872d739
# -- bench/results/kv_device_residency-callerbind.json` shows the device metadata changing
# alongside the reading: `by_device_type_and_vendor` went from the CUDA `RuntimeError` text
# to `"OK"` in the same edit that swapped the device. The obstacle did not go away; the
# question changed under it.
#
# Resolution (evidence, not argument): both artifacts still carry a `q1_addressable` block
# from this exact ORT call, and the two vendors currently on record disagree --
# `kv_device_residency-epbind.json` was captured on the NVIDIA device (0x10de) and still
# shows the `RuntimeError`; `kv_device_residency-callerbind.json`, captured on Intel
# (0x8086), shows `"OK"`. `q1_addressable` is asked before either lane's IOBinding logic
# runs, so it is a fact about the vendor id argument, not about which lane (`caller`/`ep`)
# captured it. So the CUDA-hardcode claim is VENDOR-SPECIFIC, not the universal "regardless
# of the vendor id" this file used to say: it reproduces on NVIDIA's vendor id and does not
# reproduce on Intel's, on this ORT build. The assertion below is repaired to read whichever
# currently-committed artifact carries the vendor the claim is about, instead of hardcoding
# a filename that vendor happened to change out from under.
_NVIDIA_VENDOR_ID = "0x10de"


def _artifact_with_vendor(vendor_id: str) -> dict | None:
    """Return whichever of CALLER/EPBIND currently records *vendor_id*, or None.

    `q1_addressable` is captured before the caller-bind/ep-bind lanes diverge, so it is the
    same question regardless of which of the two files holds it -- only the vendor on the
    box that produced the reading matters.
    """
    for p in (CALLER, EPBIND):
        if not p.is_file():
            continue
        d = _load(p)
        if d.get("ep_device", {}).get("vulkan.vendor_id") == vendor_id:
            return d
    return None


def test_the_python_binding_hardcodes_gpu_to_cuda() -> None:
    """`device_type='gpu'` maps to CUDA on NVIDIA's vendor id (0x10de) -- measured, not
    assumed -- so `memory_info=` is the route for *that* vendor. It is not a universal ORT
    behaviour: see the module-level note above this test for the artifact/assertion
    disagreement this replaces and the evidence that settled it."""
    d = _artifact_with_vendor(_NVIDIA_VENDOR_ID)
    if d is None:
        pytest.skip(
            "no recorded reading carries the NVIDIA vendor id (0x10de); this claim has "
            "nothing to check against, and skipping silently would restate the old bug "
            "in a new shape, so it is named here instead"
        )
    assert "CUDA" in d["q1_addressable"]["by_device_type_and_vendor"]
    assert d["q1_addressable"]["by_ep_memory_info"] == "OK"
    # The binding labels any non-CPU OrtValue 'cuda'. Recorded, never used as evidence.
    assert d["q1_addressable"]["ortvalue_device_name_reported"] == "cuda"


def test_the_cuda_hardcode_does_not_reproduce_on_intels_vendor_id() -> None:
    """The other half of the same finding, recorded rather than discarded.

    `by_device_type_and_vendor` reading `"OK"` on Intel's vendor id (0x8086) is a real,
    current measurement -- it is what let the artifact/assertion disagreement above go
    unnoticed for a round. It is NOT evidence that the hazard is gone everywhere: `q1`
    asks nothing about whether the resulting `OrtValue` is usably addressable, only
    whether the call raised, so this records what was observed and no more.
    """
    d = _artifact_with_vendor("0x8086")
    if d is None:
        pytest.skip("no recorded reading carries Intel's vendor id (0x8086)")
    assert d["q1_addressable"]["by_device_type_and_vendor"] == "OK"
    assert d["q1_addressable"]["by_ep_memory_info"] == "OK"


@pytest.mark.skipif(
    not os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB"), reason="needs the EP library"
)
@pytest.mark.slow
def test_live_caller_bound_kv_stays_device_resident() -> None:
    """Re-take the caller-bind reading now rather than trust the artifact. Seconds, not minutes:
    the GQA evidence case, one dispatch."""
    import subprocess  # noqa: PLC0415
    import sys  # noqa: PLC0415

    env = dict(os.environ)
    env["ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY"] = "1"
    env.pop("ONNXRUNTIME_EP_VULKAN_BIND_OUTPUTS", None)
    r = subprocess.run(
        [sys.executable, str(REPO / "bench" / "results" / "probe_kv_device_residency.py")],
        env=env, capture_output=True, encoding="utf-8", errors="replace", timeout=900,
    )
    assert r.returncode == 0, (r.stdout or "")[-3000:]
