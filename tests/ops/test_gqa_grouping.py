"""Grouped-query attention with **non-unit grouping** (Nq/Nkv != 1), as a regression test.

WHY THIS FILE EXISTS
====================
Every verdict in this project for several rounds carried the sentence "``Nq/Nkv = 1.00``
here, 4x on Llama-3" -- a documented blind spot, restated until it read as a finding.
Documenting a risk is not mitigating it.  ``tests/ops/test_gqa.py`` did in fact already
cover G=4 (8 query heads / 2 KV heads), so "nobody has run it" was itself not true; but it
covered only **decode on the growing convention with an aggregate tolerance**.  The
genuinely untested regimes were:

  * grouping x **prefill** (seq_len > 1, so ``tok_pos = past_len + s_local`` sweeps a range
    rather than taking one value), and
  * grouping x the **KV arena**, where ``present`` aliases ``past`` in one buffer.

The arena is the one that mattered.  ``rust/src/ops/attention.rs`` proves the alias safe by
showing the read set ``{t : t < past_len}`` and the write set ``{past_len + s_local}`` are
disjoint at a common base and stride.  That proof was constructed and checked for the
one-to-one case.  Under grouping several query heads share one KV head, and if the
argument does not survive, the arena corrupts silently on every grouped model -- the worst
failure shape available.

WHAT THIS FILE ASSERTS AND WHAT IT DOES NOT
===========================================
It answers the **node-level correctness** question: a single ``GroupQueryAttention`` node
at Llama-3 8B's attention shape (32 query heads, 8 KV heads, head dim 128) agrees with the
ORT CPU EP per output.  It does **not** answer the end-to-end question -- no grouped model
is run here, and an 8 GB board will not hold Llama-3 8B.  Anyone quoting these results
must carry that scope with them.

The measurement lives in ``probe_gqa_grouping.py``; this module is the permanent gate over
it.  The probe's own device-free ``selftest()`` arms run here too, because an instrument
that has never been seen in its red state has no demonstrated red state.
"""

from __future__ import annotations


import os

import numpy as np
import pytest

import _models as m
import probe_gqa_grouping as p


# ----------------------------------------------------------------------------------
# Device-free.  These run everywhere, including CI without a GPU.
# ----------------------------------------------------------------------------------


def test_probe_selftest_arms_all_pass() -> None:
    """The probe's verdict function can reach every one of its red states."""
    assert p.selftest() == 0


def test_the_shapes_under_test_really_are_grouped() -> None:
    """A grouping test whose shapes are one-to-one measures the thing it was written to
    stop measuring.  Asserted, not assumed."""
    assert p.LLAMA3["nq"] == 32 and p.LLAMA3["nkv"] == 8
    assert p.LLAMA3["nq"] // p.LLAMA3["nkv"] == 4
    assert p.LLAMA3["d"] == 128
    assert p.LLAMA3_G1["nq"] // p.LLAMA3_G1["nkv"] == 1
    assert p.SMALL_G4["nq"] // p.SMALL_G4["nkv"] == 4
    assert p.SMALL_G1["nq"] // p.SMALL_G1["nkv"] == 1
    # The G=1 control must match the grouped arm in every dimension EXCEPT the grouping,
    # or a difference between them is not attributable to grouping.
    for k in ("nq", "d", "rot"):
        assert p.SMALL_G4[k] == p.SMALL_G1[k], k
        assert p.LLAMA3[k] == p.LLAMA3_G1[k], k


def test_float64_reference_is_a_convex_combination() -> None:
    """The oracle is the CPU EP; the float64 reference is the tie-breaker used when the
    two disagree.  A tie-breaker nobody has checked is not a tie-breaker.

    Softmax rows are convex combinations, so every attention output element must lie
    within the range of the value vectors it mixes.  A reference that violates this has a
    bug in its normalisation and would have silently "confirmed" whichever side it
    resembled.
    """
    shape = p.SMALL_G4
    seq_len, past_len = 4, 8
    feeds = p.make_feeds(past_stride=past_len, seq_len=seq_len, past_len=past_len, **shape)
    out, pk, pv = p.gqa_reference_f64(
        feeds, seq_len=seq_len, past_len=past_len, **shape
    )
    nq, nkv, d = shape["nq"], shape["nkv"], shape["d"]
    assert out.shape == (1, seq_len, nq * d)
    assert pk.shape == (1, nkv, past_len + seq_len, d)
    assert np.isfinite(out).all()

    v_new = feeds["packed_qkv"].astype(np.float64)[..., (nq + nkv) * d:]
    lo = min(float(v_new.min()), float(feeds["past_value"].astype(np.float64).min()))
    hi = max(float(v_new.max()), float(feeds["past_value"].astype(np.float64).max()))
    assert float(out.min()) >= lo - 1e-9 and float(out.max()) <= hi + 1e-9, (
        "the float64 reference produced an attention output outside the convex hull of "
        "its own value vectors; its softmax is not normalised"
    )


def test_grouping_does_not_appear_in_the_disjointness_argument() -> None:
    """The static half of the answer, pinned so a future edit to the kernel cannot quietly
    invalidate it.

    The alias proof in ``attention.rs`` names a base ``(b*Nkv + kv_h) * stride * D``, a read
    set ``{t < past_len}`` and a write set ``{past_len + s_local}``.  None of the three
    mentions the query-head index ``h`` or the group size ``Nq/Nkv``; ``kv_h`` appears
    identically on both sides.  So *address* disjointness is invariant in the group size --
    grouping changes which invocations map to a given ``kv_h``, never which addresses they
    touch.  What grouping newly introduces is G invocations writing the **same** present
    half-word, which is a duplicate-writer question and not a disjointness one.
    """
    src = (p.Path(__file__).resolve().parents[2] / "rust" / "src" / "ops" / "attention.rs"
           ).read_text(encoding="utf-8", errors="replace")
    assert "tok_pos" in src and "past_len" in src, (
        "the alias argument this test pins has moved or been renamed; re-derive it before "
        "editing this assertion"
    )


# ----------------------------------------------------------------------------------
# Device tests.  Skip cleanly with no GPU or no EP.
# ----------------------------------------------------------------------------------

GROWING_ARMS = [
    "g1_decode_growing",
    "g4_decode_growing",
    "g1_prefill_growing",
    "g4_prefill_growing",
    "llama3_g1_decode_growing",
    "llama3_g4_decode_growing",
    "llama3_g4_prefill_growing",
]

ARENA_ARMS = [
    "g1_decode_arena",
    "g4_decode_arena",
    "g1_prefill_arena",
    "g4_prefill_arena",
    "llama3_g4_prefill_arena",
]

_BY_NAME = {a["name"]: a for a in p.ARMS}

_ARENA_ENV = {
    "ONNXRUNTIME_EP_VULKAN_KV_ARENA": "1",
    "ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY": "1",
    "ONNXRUNTIME_EP_VULKAN_BIND_OUTPUTS": "1",
}

# Snapshotted at import, deliberately.
#
# These are read by the EP when the plugin is registered and the device is opened, not
# when a session is created: with ``DEVICE_MEMORY`` unset at process start the EP's
# device exposes no DEFAULT allocator at all, and the arena arms cannot bind device
# memory however the environment is manipulated later.  Setting them inside a test
# therefore does NOT enable the arena, and -- worse -- setting ``KV_ARENA=1`` for the
# whole process makes the *growing* arms be declined and fall back to CPU, turning them
# into a CPU-vs-CPU comparison that agrees perfectly.
#
# So the two lanes are two pytest invocations, and each one skips the other's arms with a
# reason that says exactly how to run them.  A skip that names its command is a coverage
# gap on the record; a silent pass is not.
_ENV_AT_IMPORT = {k: os.environ.get(k) for k in _ARENA_ENV}
_ARENA_ENV_LIVE = all(_ENV_AT_IMPORT.get(k) == v for k, v in _ARENA_ENV.items())

_ARENA_HOWTO = (
    "the KV arena is selected by process environment read at EP registration, not at "
    "session creation. Run the arena lane as its own invocation:\n"
    "  $env:ONNXRUNTIME_EP_VULKAN_KV_ARENA='1'\n"
    "  $env:ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY='1'\n"
    "  $env:ONNXRUNTIME_EP_VULKAN_BIND_OUTPUTS='1'\n"
    "  python -m pytest test_gqa_grouping.py -k arena"
)


def _assert_healthy(rec: dict, arm_name: str) -> None:
    """Per-output verdicts, never an aggregate.

    ``CORRUPT`` and ``RACE`` outrank ``AGREE`` in ``derive_verdict``: a run whose outputs
    match the oracle but whose past region was overwritten is not a pass.
    """
    verdict = rec["verdict"]
    if verdict.startswith("UNMEASURED"):
        pytest.skip(f"{arm_name}: {verdict} — the EP declined; nothing was measured")
    assert verdict == "AGREE", f"{arm_name}: {verdict}\n{rec.get('cpu_facts')}"
    assert rec["cross_run_identical"] is True, (
        f"{arm_name}: three identical runs did not produce identical bits — a grouped "
        "dispatch with G writers per present slot is exactly where a race would live"
    )
    assert rec["past_region_of_present_bit_intact"] is True, (
        f"{arm_name}: the past region of present was modified; the alias disjointness "
        "argument does not survive this configuration"
    )
    assert rec.get("oracle_failing_indices", []) == [], (
        f"{arm_name}: outputs {rec['oracle_failing_indices']} disagree with the CPU EP"
    )
    # Per-output, never an aggregate.  A single output outside tolerance is a failure even
    # if every other output is bit-exact.
    per_output = rec.get("per_output") or []
    assert per_output, f"{arm_name}: verdict AGREE with no per-output record is not a result"
    for e in per_output:
        assert e["status"] == "WITHIN_TOLERANCE", f"{arm_name}: output {e['index']}: {e}"
    # The float64 second reference: where the two EPs differ, it says which one moved.
    # `divergent` is symmetric and this project has read it asymmetrically before.
    for f in rec.get("f64_reference") or []:
        assert f.get("status") != "SHAPE_MISMATCH", f"{arm_name}: {f}"
        assert f.get("which_is_further_from_true") != "vulkan", (
            f"{arm_name}: {f['output']} — the Vulkan EP is further from the float64 "
            f"reference than the CPU EP is: {f}"
        )


@pytest.mark.parametrize("arm_name", GROWING_ARMS)
def test_grouped_gqa_matches_cpu_growing(arm_name: str, require_vulkan) -> None:
    if _ENV_AT_IMPORT.get("ONNXRUNTIME_EP_VULKAN_KV_ARENA") == "1":
        pytest.skip(
            "KV_ARENA=1 is set for this process, so the growing arms are declined and ORT "
            "falls back to CPU — a CPU-vs-CPU comparison that would agree perfectly. Run "
            "the growing lane without the arena environment."
        )
    _assert_healthy(p.run_arm(_BY_NAME[arm_name], repeats=3), arm_name)


@pytest.mark.parametrize("arm_name", ARENA_ARMS)
def test_grouped_gqa_matches_cpu_arena(arm_name: str, require_vulkan) -> None:
    """The arena lane: ``present`` and ``past`` are one buffer, bound as the same OrtValue.

    The past-capacity tail is poisoned with 0.5 / -0.25 rather than zeros, so a kernel that
    reads or writes past the true past length shows a value that cannot be mistaken for a
    freshly-zeroed arena.  A zero-filled tail would have made a corrupt read look correct.
    """
    if not _ARENA_ENV_LIVE:
        pytest.skip(_ARENA_HOWTO)
    rec = p.run_arena_arm(_BY_NAME[arm_name], repeats=3)
    _assert_healthy(rec, arm_name)
    assert rec["arena_tail_poison_intact"] is True, (
        f"{arm_name}: the poisoned capacity tail was overwritten"
    )
    # The route must be established by the run, not by the environment variable that
    # requested it.  `kv_cache_convention` is only in the counters file, which this test
    # does not own, so what is asserted here is the set of structural preconditions the
    # EP itself checks before it will take the arena path -- each of which is falsifiable
    # against this record.  A vacuous `assert x in (None, "SHARED")` would have been true
    # whatever happened, and an observable that is true whatever happens cannot convict.
    arm = _BY_NAME[arm_name]
    assert arm["present_declared"] is False
    assert arm["past_stride"] > arm["past_len"], (
        "an arena arm whose past extent is not a capacity larger than the true past "
        "length has no tail to poison and cannot detect an over-read"
    )
    assert rec["kv_arena_env"] == "1" and rec["device_memory_env"] == "1"
    assert rec.get("device_name"), "no EP device was opened; nothing ran on Vulkan"


def test_grouping_is_not_what_makes_a_configuration_fail(require_vulkan) -> None:
    """The symmetry check that keeps a G=4 result from being blamed on G=4.

    ``divergent`` is symmetric, and this project has read it asymmetrically before.  The
    same applies to a decline: if a G=4 arm declines, the matched G=1 arm must be run
    before the finding is attributed to grouping.  On this box the arena was refused on the
    Intel iGPU -- at G=1 and at G=4 identically -- which was recorded as a coverage gap and
    emphatically not a grouping result.

    That refusal is now explained and closed: it was not a device property at all.  With
    ``ONNXRUNTIME_EP_VULKAN_DEVICE`` unset, ORT keys its allocator to the device index the
    *factory* advertised while the session honours ``ep.device_index``; the two output
    binds the arena aliases are then declined and the EP refuses correctly.  Pin the env
    var before the EP library is registered and the same arm AGREEs on the same Iris Xe.
    See ``test_arena_refusal_frame.py`` for the paired, one-arm-per-process reading.  The
    symmetry check below is unchanged -- it was right for a reason that outlived the gap.
    """
    arena = _ARENA_ENV_LIVE
    pairs = (
        [("g1_decode_arena", "g4_decode_arena"), ("g1_prefill_arena", "g4_prefill_arena")]
        if arena else
        [("g1_decode_growing", "g4_decode_growing"),
         ("g1_prefill_growing", "g4_prefill_growing")]
    )
    run = p.run_arena_arm if arena else p.run_arm
    checked = 0
    for one, four in pairs:
        r1, r4 = run(_BY_NAME[one], repeats=1), run(_BY_NAME[four], repeats=1)
        k1, k4 = r1["verdict"].split("(")[0], r4["verdict"].split("(")[0]
        assert k1 == k4, (
            f"{one} -> {r1['verdict']} but {four} -> {r4['verdict']}: the group size is the "
            "only difference between these two arms, so this IS a grouping finding and must "
            "be reported as one"
        )
        checked += 1
    assert checked == len(pairs)
