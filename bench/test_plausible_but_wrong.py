"""Tests for readings that would be *plausible and wrong*.

The coordinator's instruction, and the standard the `Phase::Submit` test in `rust/src/trace.rs`
set: the failure mode worth defending against is not a number that is obviously absurd — nobody
ships a 4000x speedup. It is a number that looks reasonable and is wrong by a constant factor,
or wrong because two things that are not comparable were compared.

Each test below names the plausible-but-wrong reading it prevents. Several of them use real
values measured on this machine (Intel Iris Xe + NVIDIA RTX 4060 Laptop, 2026-07-29) so the
fixtures are not invented.

Run: ``python -m pytest bench/test_plausible_but_wrong.py -q``
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import pytest  # noqa: E402
import numpy as np  # noqa: E402

import compare  # noqa: E402
import devices as D  # noqa: E402
import phi35  # noqa: E402
import portability as P  # noqa: E402
import producers  # noqa: E402
import stats  # noqa: E402
import timestamp_audit  # noqa: E402
from stats import Sample  # noqa: E402

# Measured with vulkaninfo on this machine, 2026-07-29. Not invented, not rounded.
IRIS_XE = dict(
    index=0,
    name="Intel(R) Iris(R) Xe Graphics",
    device_type="VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU",
    api_version="1.4.309",
    driver_version="101.6737",
    timestamp_period_ns=52.0833,
    timestamp_valid_bits=36,
    max_compute_shared_memory=32768,
    subgroup_size=32,
    uma=True,
)
RTX_4060 = dict(
    index=1,
    name="NVIDIA GeForce RTX 4060 Laptop GPU",
    device_type="VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU",
    api_version="1.4.325",
    driver_version="591.55",
    timestamp_period_ns=1.0,
    timestamp_valid_bits=64,
    max_compute_shared_memory=49152,
    subgroup_size=32,
    uma=False,
    device_local_host_visible=True,
)


def _facts(**kw) -> D.DeviceFacts:
    return D.DeviceFacts(**kw)


# -------------------------------------------------------------------------------------------
# Wrong reading: "timestampPeriod is 1.0, so ticks are nanoseconds."
# -------------------------------------------------------------------------------------------


def test_timestamp_period_is_not_one_on_real_hardware():
    """A hardcoded 1.0 would under-report Iris Xe GPU time by 52x — and 52x too fast reads as
    a triumph, not as a bug."""
    xe, nv = _facts(**IRIS_XE), _facts(**RTX_4060)
    assert nv.timestamp_period_ns == 1.0
    assert xe.timestamp_period_ns == pytest.approx(52.0833)
    assert xe.assume_one_error_factor > 50
    # The two devices on one machine disagree, which is the whole point.
    assert xe.timestamp_period_ns != nv.timestamp_period_ns


# -------------------------------------------------------------------------------------------
# Wrong reading: "timestamps are 64-bit / 32-bit, pick one."
# -------------------------------------------------------------------------------------------


def test_valid_bits_differ_and_neither_is_thirty_two():
    """36 bits is not a number anyone would guess, and masking with the wrong width produces
    intervals that are plausible and wrong rather than obviously broken."""
    xe, nv = _facts(**IRIS_XE), _facts(**RTX_4060)
    assert xe.timestamp_valid_bits == 36
    assert nv.timestamp_valid_bits == 64
    assert xe.timestamps_usable and nv.timestamps_usable


def test_wrap_period_is_short_enough_to_matter_on_the_integrated_part():
    """~1 hour, not 'never'. A long-running session will wrap, so wrap recovery is a routine
    path and not a theoretical one."""
    xe = _facts(**IRIS_XE)
    assert 3000 < xe.timestamp_wrap_seconds < 4000
    # ... and on the 4060 it is effectively never, so a wrap bug would hide there.
    assert _facts(**RTX_4060).timestamp_wrap_seconds > 1e9


def test_zero_valid_bits_is_no_timestamps_not_zero_time():
    """The dangerous reading: validBits==0 yields tick deltas of 0, which look like
    infinitely fast kernels rather than like an unsupported feature."""
    dead = _facts(index=0, name="hypothetical", timestamp_valid_bits=0, timestamp_period_ns=1.0)
    assert dead.timestamps_usable is False
    assert dead.timestamp_wrap_seconds is None


# -------------------------------------------------------------------------------------------
# Wrong reading: "some memory type is DEVICE_LOCAL|HOST_VISIBLE, so this GPU is unified."
# -------------------------------------------------------------------------------------------

_NV_MEMORY_BLOCK = """
VkPhysicalDeviceMemoryProperties:
memoryHeaps: count = 2
\tmemoryHeaps[0]:
\t\tsize   = 8345616384
\t\tflags: count = 1
\t\t\tMEMORY_HEAP_DEVICE_LOCAL_BIT
\tmemoryHeaps[1]:
\t\tsize   = 34267721728
\t\tflags: count = 0
\t\t\tNone
memoryTypes: count = 5
\tmemoryTypes[0]:
\t\theapIndex     = 1
\t\tpropertyFlags = 0x0000: count = 0
\t\tusable for:
\t\t\tnothing
\tmemoryTypes[1]:
\t\theapIndex     = 0
\t\tpropertyFlags = 0x0001: count = 1
\t\t\tMEMORY_PROPERTY_DEVICE_LOCAL_BIT
\t\tusable for:
\t\t\tcolor images
\tmemoryTypes[4]:
\t\theapIndex     = 0
\t\tpropertyFlags = 0x0007: count = 3
\t\t\tMEMORY_PROPERTY_DEVICE_LOCAL_BIT
\t\t\tMEMORY_PROPERTY_HOST_VISIBLE_BIT
\t\t\tMEMORY_PROPERTY_HOST_COHERENT_BIT
\t\tusable for:
\t\t\tcolor images
"""

_XE_MEMORY_BLOCK = """
VkPhysicalDeviceMemoryProperties:
memoryHeaps: count = 1
\tmemoryHeaps[0]:
\t\tsize   = 34267721728
\t\tflags: count = 1
\t\t\tMEMORY_HEAP_DEVICE_LOCAL_BIT
memoryTypes: count = 3
\tmemoryTypes[0]:
\t\theapIndex     = 0
\t\tpropertyFlags = 0x0001: count = 1
\t\t\tMEMORY_PROPERTY_DEVICE_LOCAL_BIT
\t\tusable for:
\t\t\tcolor images
\tmemoryTypes[1]:
\t\theapIndex     = 0
\t\tpropertyFlags = 0x0007: count = 3
\t\t\tMEMORY_PROPERTY_DEVICE_LOCAL_BIT
\t\t\tMEMORY_PROPERTY_HOST_VISIBLE_BIT
\t\t\tMEMORY_PROPERTY_HOST_COHERENT_BIT
\t\tusable for:
\t\t\tcolor images
"""


def test_a_resizable_bar_window_is_not_unified_memory():
    """The RTX 4060 exposes DEVICE_LOCAL|HOST_VISIBLE (the BAR window). The naive test reports
    a discrete GPU as UMA — plausible in a table, and it would justify skipping the staging
    copy we specifically need to measure."""
    parsed = D._parse_memory_from_text(_XE_MEMORY_BLOCK + _NV_MEMORY_BLOCK, 2)
    assert parsed[0]["uma"] is True
    assert parsed[1]["uma"] is False
    # The BAR window is still detected — it is a real optimisation, just not unified memory.
    assert parsed[1]["device_local_host_visible"] is True


# -------------------------------------------------------------------------------------------
# Wrong reading: "both runs are on 'the GPU', so the delta is the code change."
# -------------------------------------------------------------------------------------------


def _result(dev: D.DeviceFacts, median: float, **kw) -> dict:
    producer = kw.get("producer", "tests/ops/_models.py@-#abc123def456")
    return {
        "label": "x",
        "device": dev.to_dict(),
        "device_fingerprint": dev.fingerprint,
        "barrier_backend": kw.get("backend", "device default"),
        "producers": [{"fingerprint": producer}] if producer else [],
        "cases": [
            {
                "name": "case",
                "claim": {"claimed": True},
                "tile_config": kw.get("tile"),
                "producer_fingerprint": producer,
                "vulkan": {"name": "case", "median_ms": median, "mad_ms": 0.01},
                "cpu": {"name": "case", "median_ms": 10.0, "mad_ms": 0.01},
            }
        ],
    }


def test_comparing_two_different_devices_is_refused_not_warned():
    base = _result(_facts(**IRIS_XE), 10.0)
    pr = _result(_facts(**RTX_4060), 2.0)
    refusal = compare.cross_device_refusal(base, pr)
    assert refusal is not None
    assert "different devices" in refusal
    # A 5x "improvement" is exactly what this refusal is stopping from being reported.


def test_same_device_compares_normally():
    base = _result(_facts(**RTX_4060), 10.0)
    pr = _result(_facts(**RTX_4060), 10.1)
    assert compare.cross_device_refusal(base, pr) is None


def test_an_unidentified_device_is_refused_too():
    """'We forgot to record the device' must not degrade to 'assume it is the same device'."""
    base = _result(_facts(**RTX_4060), 10.0)
    anon = _result(_facts(**RTX_4060), 10.0)
    anon["device_fingerprint"] = None
    anon["device"] = None
    assert compare.cross_device_refusal(base, anon) is not None


def test_the_same_silicon_on_a_new_driver_is_a_different_device():
    """A driver update that changes performance would otherwise be attributed to the PR."""
    old = _facts(**RTX_4060)
    new = _facts(**{**RTX_4060, "driver_version": "592.10"})
    assert old.fingerprint != new.fingerprint
    assert compare.cross_device_refusal(_result(old, 10.0), _result(new, 8.0)) is not None


# -------------------------------------------------------------------------------------------
# Wrong reading: "same device, same case, so it is the same kernel."
# -------------------------------------------------------------------------------------------


def test_a_different_tile_config_is_a_different_kernel():
    b = {"tile_config": "64x64x8"}
    p = {"tile_config": "32x32x8"}
    assert compare.tile_mismatch(b, p)


def test_unknown_tile_config_never_certifies_a_comparison():
    """Two `None`s mean *unknown*, not *equal*. An unknown must not be able to bless a
    comparison — it can only fail to disprove one."""
    assert not compare.tile_mismatch({"tile_config": None}, {"tile_config": None})
    assert not compare.tile_mismatch({"tile_config": None}, {"tile_config": "64x64x8"})


# -------------------------------------------------------------------------------------------
# Wrong reading: "a 32 KiB tile config works everywhere."
# -------------------------------------------------------------------------------------------


def test_a_tile_config_tuned_on_the_4060_may_not_fit_the_iris_xe():
    xe, nv = _facts(**IRIS_XE), _facts(**RTX_4060)
    assert nv.max_compute_shared_memory > xe.max_compute_shared_memory
    # A 48 KiB tile is simply not creatable on the Xe; a speedup that does not name its tile
    # config is therefore comparing two different kernels.
    assert 48 * 1024 <= nv.max_compute_shared_memory
    assert 48 * 1024 > xe.max_compute_shared_memory


# -------------------------------------------------------------------------------------------
# Wrong reading: "the median moved 12%, that's a regression."
# -------------------------------------------------------------------------------------------


def test_a_shift_smaller_than_the_jitter_is_not_a_finding():
    base = Sample("b", [10.0, 14.0, 7.0, 12.0, 9.0] * 6)
    pr = Sample("p", [11.2, 15.7, 7.8, 13.4, 10.1] * 6)
    from stats import relative_delta, significant

    assert relative_delta(base, pr) > 0.10
    assert not significant(base, pr, 0.10)


# -------------------------------------------------------------------------------------------
# Wrong reading: "the harness picked a GPU, so the number is about that GPU."
# -------------------------------------------------------------------------------------------


def test_a_multi_device_machine_forces_an_explicit_choice():
    from bench import DeviceSelectionError, select_device

    both = [_facts(**IRIS_XE), _facts(**RTX_4060)]
    with pytest.raises(DeviceSelectionError):
        select_device(both, None)
    assert select_device(both, 1).name.startswith("NVIDIA")
    with pytest.raises(DeviceSelectionError):
        select_device(both, 7)
    # A single-device machine needs no ceremony.
    assert select_device([_facts(**RTX_4060)], None).index == 1


# -------------------------------------------------------------------------------------------
# Wrong reading: "the provider list said VulkanExecutionProvider, so it ran on Vulkan."
# -------------------------------------------------------------------------------------------


def test_an_ort_older_than_the_ep_refuses_to_produce_vulkan_numbers(monkeypatch, tmp_path):
    """Observed for real on 2026-07-29: ORT 1.27 rejects the plugin's API version 28, ORT then
    runs every node on the CPU EP, and both columns of the table are the same code. In that
    state `matmulnbits_q4_b32_K4096_N4096` showed 1.36 ms 'vulkan' vs 2.31 ms 'cpu' — a 1.70x
    'speedup', above the OQ-12 bar, entirely composed of run-to-run noise.

    Two independent gates must stop that: the claim gate (no claimed node → no number) and this
    version gate (no loadable EP → no Vulkan column at all).
    """
    import bench as bench_mod

    lib = tmp_path / "fake_ep.dll"
    lib.write_bytes(b"not a real dll")
    monkeypatch.setenv(bench_mod.EP_LIB_ENV, str(lib))
    monkeypatch.setattr(bench_mod, "_ort_version", lambda: (1, 27, 0))
    assert bench_mod.register_ep() is False


def test_the_version_gate_lets_a_supported_ort_through(monkeypatch, tmp_path):
    import bench as bench_mod

    lib = tmp_path / "fake_ep.dll"
    lib.write_bytes(b"not a real dll")
    monkeypatch.setenv(bench_mod.EP_LIB_ENV, str(lib))
    monkeypatch.setattr(bench_mod, "_ort_version", lambda: (1, 28, 0))
    called = {}

    class _FakeOrt:
        __version__ = "1.28.0"

        @staticmethod
        def register_execution_provider_library(name, path):
            called["name"] = name

    monkeypatch.setitem(sys.modules, "onnxruntime", _FakeOrt)
    assert bench_mod.register_ep() is True
    assert called["name"] == bench_mod.EP_NAME


# ---------------------------------------------------------------------------------------------
# Producer provenance. Mouse's OP_COVERAGE.md §4.18: op coverage is relative to a producer, not
# to a model architecture. A *timing* is worse, because it has no shape to disagree about and so
# nothing fails loudly — the number just quietly describes a different graph.
# ---------------------------------------------------------------------------------------------


def test_a_synthetic_op_graph_cannot_be_named_after_a_model_family():
    """The wrong reading: "qwen3_decoder_layer: 1.4x" means something about Qwen3.

    It does not, if the graph was three hand-built ops. This must fail at construction, not be
    caught in review.
    """
    with pytest.raises(producers.ProducerProvenanceError) as e:
        producers.assert_family_label_is_earned(
            "qwen3_decoder_layer", [], producers.op_builder()
        )
    assert "qwen3" in str(e.value)


def test_a_family_word_hidden_in_a_tag_is_caught_too():
    with pytest.raises(producers.ProducerProvenanceError):
        producers.assert_family_label_is_earned("decoder_layer", ["llama"], producers.op_builder())


def test_an_ordinary_op_case_name_is_not_flagged():
    """The gate must not be so eager that it makes honest names unusable."""
    producers.assert_family_label_is_earned("matmulnbits_q4_b32_K4096_N4096", ["gemm"],
                                            producers.op_builder())
    producers.assert_family_label_is_earned("add_fp32_1024x1024", ["dispatch-bound"],
                                            producers.op_builder())
    assert producers.family_words_in("matmulnbits_q4_b32_K4096_N4096") == []


def test_a_versioned_model_exporter_earns_the_label():
    producers.assert_family_label_is_earned(
        "qwen3_decoder_mobius", [], producers.mobius("0.3.1")
    )


def test_an_unversioned_exporter_does_not_earn_the_label():
    """A family label from an exporter whose version is unknown is not reproducible."""
    anon = producers.Producer(name="somebuilder", kind=producers.KIND_MODEL,
                              version=None, model_family="qwen3")
    assert anon.can_claim_model_family is False
    with pytest.raises(producers.ProducerProvenanceError):
        producers.assert_family_label_is_earned("qwen3_decoder", [], anon)


def test_a_case_must_be_named_after_the_family_actually_built():
    with pytest.raises(producers.ProducerProvenanceError):
        producers.assert_family_label_is_earned(
            "llama_decoder", [], producers.mobius("0.3.1", model_family="qwen3")
        )


def test_mobius_and_the_ort_genai_builder_are_different_producers():
    """The finding itself: the same architecture, two op sets.

    mobius emits ai.onnx::Attention @ 23; the ORT GenAI builder emits the com.microsoft contrib
    graph. MatMulNBits is the only op both agree on, so it is the only case whose cost carries
    across them.
    """
    m, g = producers.mobius("0.3.1"), producers.ort_genai_builder("0.9.2")
    assert m.fingerprint != g.fingerprint
    assert m.opsets != g.opsets
    assert "com.microsoft" in g.opsets and "com.microsoft" not in m.opsets


def test_a_builder_edit_makes_a_different_producer():
    """Same instinct as the driver version in the device fingerprint.

    If the builder changed between base and PR, the graph changed, and attributing the delta to
    the EP would be wrong in a way that looks entirely reasonable.
    """
    a = producers.Producer(name="b", kind=producers.KIND_OP, digest="a" * 64)
    b = producers.Producer(name="b", kind=producers.KIND_OP, digest="b" * 64)
    assert a.fingerprint != b.fingerprint


def test_comparing_across_producers_is_refused_not_warned():
    base = _result(_facts(**RTX_4060), 10.0, producer="onnx-genai-models/mobius@0.3.1#aaaaaaaaaaaa")
    pr = _result(_facts(**RTX_4060), 4.0,
                 producer="onnxruntime-genai/builder.py@0.9.2#bbbbbbbbbbbb")
    refusal = compare.producer_refusal(base, pr)
    assert refusal is not None
    assert "different producers" in refusal
    # A 2.5x "improvement" that is entirely an exporter difference.


def test_same_producer_compares_normally():
    base = _result(_facts(**RTX_4060), 10.0)
    pr = _result(_facts(**RTX_4060), 10.1)
    assert compare.producer_refusal(base, pr) is None


def test_an_unrecorded_producer_is_refused_too():
    """"We do not know what built these" is not evidence that the same thing built both."""
    base = _result(_facts(**RTX_4060), 10.0)
    anon = _result(_facts(**RTX_4060), 10.0, producer=None)
    anon["cases"][0]["producer_fingerprint"] = None
    assert compare.producer_refusal(base, anon) is not None


def test_every_shipped_case_carries_a_producer():
    """Nothing may reach a result file without provenance."""
    import cases as case_mod

    built = case_mod.build_cases()
    assert built
    assert all(c.producer is not None for c in built)
    assert all(c.producer.fingerprint for c in built)
    # Today there is exactly one producer, and it is an op builder that can name no family.
    prods = case_mod.case_producers(built)
    assert len(prods) == 1
    assert prods[0].kind == producers.KIND_OP
    assert prods[0].can_claim_model_family is False


# ---------------------------------------------------------------------------------------------
# Portability. Justin's standing directive: 要时刻注意跨平台通用性 — cross-platform generality at
# all times. A Vulkan EP that is really a desktop-NVIDIA EP has no reason to exist. The wrong
# reading these guard against: "it ran fast on the hardware we own, therefore it is fast".
# ---------------------------------------------------------------------------------------------


def test_the_floor_is_the_admission_floor_not_this_desk():
    """The wrong reading: 32 KiB is the budget, because the *smaller* local GPU has 32 KiB.

    DESIGN.md §7.2 R4 admits devices with 16 KiB. §7.0 says shortfalls degrade op coverage, not
    device availability — so a 16 KiB device is one we promised to run on.
    """
    assert P.FLOOR_SHARED_MEMORY_BYTES == 16384
    assert P.FLOOR_WORKGROUP_INVOCATIONS == 256
    assert P.FLOOR_SHARED_MEMORY_BYTES < IRIS_XE["max_compute_shared_memory"]
    assert P.FLOOR_SHARED_MEMORY_BYTES < RTX_4060["max_compute_shared_memory"]


def test_a_tile_that_fits_the_smaller_local_gpu_is_still_not_portable():
    """The Iris Xe is our UMA proxy. It is not a shared-memory proxy."""
    xe_tuned = P.Configuration(name="xe", shared_memory_bytes=32768, workgroup_invocations=256)
    assert P.fits_device(xe_tuned, IRIS_XE["max_compute_shared_memory"], 1024) is True
    v = P.evaluate(xe_tuned)
    assert v.verdict == P.NEEDS_FALLBACK
    assert v.quotable_as_ep_behaviour is False


def test_a_floor_compliant_config_is_portable():
    v = P.evaluate(P.Configuration(name="floor", shared_memory_bytes=16384,
                                   workgroup_invocations=256))
    assert v.verdict == P.PORTABLE
    assert v.quotable_as_ep_behaviour is True


def test_an_unrecorded_config_is_not_portable_it_is_unknown():
    """"We did not record the tile" must not degrade into "the tile was fine"."""
    v = P.evaluate(P.Configuration(name=None))
    assert v.verdict == P.UNKNOWN
    assert v.quotable_as_ep_behaviour is False
    assert P.fits_device(P.Configuration(), 49152, 1024) is False


def test_a_4060_tile_does_not_fit_the_iris_xe_by_reported_limits():
    """Answered from the device's *reported* limits, never from a constant that happens to fit."""
    tuned = P.Configuration(name="4060", shared_memory_bytes=49152, workgroup_invocations=256)
    assert P.fits_device(tuned, RTX_4060["max_compute_shared_memory"], 1024) is True
    assert P.fits_device(tuned, IRIS_XE["max_compute_shared_memory"], 1024) is False


def test_a_subgroup_size_dependency_must_be_declared_and_is_never_assumed():
    """Both local GPUs report 32 — the coincidence most likely to bake a 32 into a kernel.

    Vulkan 1.1 guarantees subgroup BASIC in compute and nothing about the size.
    """
    assert P.SUBGROUP_SIZE_IS_GUARANTEED is False
    assert IRIS_XE["subgroup_size"] == RTX_4060["subgroup_size"] == 32
    undeclared = P.Configuration(name="x", shared_memory_bytes=16384, workgroup_invocations=256,
                                 depends_on_subgroup_size=True)
    assert P.evaluate(undeclared).verdict == P.UNKNOWN
    declared = P.Configuration(name="x", shared_memory_bytes=16384, workgroup_invocations=256,
                               depends_on_subgroup_size=True, subgroup_size=32)
    assert P.evaluate(declared).verdict == P.NEEDS_FALLBACK


def test_assuming_unified_memory_is_never_portable():
    """UMA on Iris Xe/Adreno/Mali, not on the 4060. A staging path must still exist."""
    v = P.evaluate(P.Configuration(name="uma-shortcut", shared_memory_bytes=16384,
                                   workgroup_invocations=256, assumes_unified_memory=True))
    assert v.verdict == P.NEEDS_FALLBACK
    assert any("staging" in r for r in v.reasons)


def test_transfer_models_may_not_be_blended_across_transfer_classes():
    """The coordinator's directive: a single blended model describes neither device.

    The blended constant would land plausibly *between* the two, which is why this is a refusal
    and not a warning.
    """
    refusal = P.transfer_model_merge_refusal(
        [{"transfer_class": "uma"}, {"transfer_class": "discrete"}]
    )
    assert refusal is not None and "describes neither" in refusal
    assert P.transfer_model_merge_refusal([{"transfer_class": "discrete"}]) is None


def test_a_transfer_fit_without_a_class_cannot_be_combined_with_anything():
    assert P.transfer_model_merge_refusal(
        [{"transfer_class": "discrete"}, {"r2": 0.99}]
    ) is not None


def test_the_comparison_table_says_when_its_numbers_are_desk_specific():
    """A reader scanning a table cannot otherwise tell EP behaviour from local behaviour."""
    pr = _result(_facts(**RTX_4060), 1.0)
    pr["cases"][0]["portability"] = P.evaluate(
        P.Configuration(name="4060", shared_memory_bytes=49152, workgroup_invocations=256)
    ).to_dict()
    banner = "\n".join(compare.portability_banner(pr))
    assert "above the §7.2 admission floor" in banner

    portable = _result(_facts(**RTX_4060), 1.0)
    portable["cases"][0]["portability"] = P.evaluate(
        P.Configuration(name="floor", shared_memory_bytes=16384, workgroup_invocations=256)
    ).to_dict()
    assert compare.portability_banner(portable) == []


def test_unrecorded_configs_are_called_out_rather_than_passing_silently():
    """Today every row is in this state, and the table must say so."""
    pr = _result(_facts(**RTX_4060), 1.0)
    pr["cases"][0]["portability"] = P.evaluate(P.Configuration()).to_dict()
    banner = "\n".join(compare.portability_banner(pr))
    assert "do not record the configuration" in banner



# ===========================================================================================
# Phi-3.5 model benchmark — the gates that stand between a real run and a fabricated number.
#
# Every test below is named for the *wrong reading* it prevents, not for the function it calls.
# The three fabricated results this project has already produced (1.70x through an EP that
# could not load, 1.45x through an EP that claimed nothing, and argmax 0 through 161 dispatches
# that computed zeros while `compute_failures` stayed at 0) were each individually plausible,
# and in each case the instrument that would have gone red did not exist yet.
# ===========================================================================================


def test_timing_is_refused_when_the_ep_is_not_in_get_providers():
    """The 1.70x shape: ORT prints a version complaint, does not raise, and runs on CPU."""
    assert phi35.refuse_if_ep_absent(["VulkanExecutionProvider", "CPUExecutionProvider"]) is None
    refusal = phi35.refuse_if_ep_absent(["CPUExecutionProvider"])
    assert refusal and "1.70x" in refusal


def test_timing_is_refused_when_the_ep_claimed_nothing():
    """The 1.45x shape: the EP loads, declines every node, and CPU jitter becomes a speedup."""
    assert phi35.refuse_if_nothing_claimed(257) is None
    assert phi35.refuse_if_nothing_claimed(0)
    # Unknown is not zero and is not fine: it means the harness cannot tell.
    assert phi35.refuse_if_nothing_claimed(None)


def test_a_divergent_verdict_voids_the_timing_it_does_not_discount_it():
    """DESIGN.md §10.0: a wrong answer voids the triple rather than trading against it."""
    assert phi35.refuse_if_not_match(phi35.MATCH) is None
    refusal = phi35.refuse_if_not_match(phi35.DIVERGENT)
    assert refusal and "fabricated" in refusal


def test_unmeasured_is_not_a_soft_match():
    """The default is UNMEASURED and it refuses just as hard as DIVERGENT does."""
    refusal = phi35.refuse_if_not_match(phi35.UNMEASURED)
    assert refusal and "UNMEASURED" in refusal and "not a soft MATCH" in refusal


def test_all_zero_logits_are_divergent_not_a_small_difference():
    """The actual bug: dispatches ran, no failure was reported, and the output was all zeros.

    A tolerance-based comparison could be argued into calling a zero vector "close" when the
    oracle's values are small. The explicit zero guard cannot be.
    """
    cpu = [np.linspace(-5, 13, 32064, dtype=np.float32).reshape(1, 1, 32064)]
    vk = [np.zeros((1, 1, 32064), dtype=np.float32)]
    v = phi35.classify_outputs(vk, cpu)
    assert v["model_output_equivalence"] == phi35.DIVERGENT
    assert v["outputs"][0]["vk_is_effectively_zero"] is True


def test_a_run_that_agrees_on_argmax_but_not_top_k_is_still_divergent():
    """Half-right is not right. A partially wrong kernel is the next bug, not the last one."""
    rng = np.random.default_rng(7)
    base = rng.standard_normal((1, 1, 4096)).astype(np.float32)
    cpu = [base.copy()]
    perturbed = base.copy()
    # Keep the argmax, scramble the rest of the top-k.
    top = np.argsort(-base[0, 0])[:phi35.TOP_K]
    perturbed[0, 0, top[1:]] = -50.0
    v = phi35.classify_outputs([perturbed], cpu)
    assert v["outputs"][0]["argmax_agrees"] is True
    assert v["outputs"][0]["top_k_agrees"] is False
    assert v["model_output_equivalence"] == phi35.DIVERGENT


def test_a_genuine_match_carries_its_evidence_not_just_its_verdict():
    """A MATCH with a max-abs-diff of exactly 0.0 is a different (more suspicious) event."""
    rng = np.random.default_rng(11)
    base = rng.standard_normal((1, 1, 4096)).astype(np.float32) * 5.0
    v = phi35.classify_outputs([base + 1e-4], [base])
    assert v["model_output_equivalence"] == phi35.MATCH
    assert v["outputs"][0]["max_abs_diff"] > 0.0
    assert v["outputs"][0]["top_k_overlap"] == phi35.TOP_K


def test_the_retired_per_island_cost_is_no_longer_produced_at_all():
    """The statistic was retired on 2026-07-30; a null is not enough, the reason must travel.

    It was total-host-delta over island count, described as a lower bound on the cost of one
    device-boundary crossing. The phase split showed the delta is dominated by per-inference host
    work that is not proportional to island count — and the statistic *rose* (8.5 -> 16.6 ms on
    Intel) across a change that collapsed islands 321 -> 33 and cut wall time 2954.6 -> 807.2 ms.
    A statistic that moves the wrong way when the thing it purports to measure improves is
    measuring something else.
    """
    known = phi35.boundary_cost(1465.9, 185.9, 257)
    assert known["per_island_ms_lower_bound"] is None
    assert known["per_island_ms_lower_bound_status"] == "RETIRED 2026-07-30"
    assert "not proportional to island count" in \
        known["per_island_ms_lower_bound_retirement_reason"]
    unknown = phi35.boundary_cost(1465.9, 185.9, None)
    assert unknown["per_island_ms_lower_bound"] is None


def test_the_quotient_is_kept_but_explicitly_not_a_per_island_cost():
    """Deleting a number that has been quoted leaves older reports irreconcilable.

    So the arithmetic is still emitted — under a name that says what it is, with a flag that
    forbids the reading it used to invite.
    """
    b = phi35.boundary_cost(1465.9, 185.9, 257)
    assert b["delta_over_island_count_ms"] == pytest.approx(4.9805, rel=1e-3)
    assert b["delta_over_island_count_is_per_island_cost"] is False
    assert "must not be quoted as one" in b["delta_over_island_count_note"]
    assert "phases.py" in b["separating_instrument"]
    assert b["separating_instrument"].count("not implemented") == 0


def test_no_ratio_is_emitted_when_either_baseline_is_unsteady():
    """A warned-about number still gets quoted. §item 3: refuse, do not warn.

    The Vulkan absolutes on the last recorded run were solid; the CPU baseline drifted
    872.8 -> 331.5 ms inside one process. The harness printed the ratio with a warning above it,
    and a warning does not travel into the summary that quotes the ratio.
    """
    steady = stats.drift([100.0, 101.0, 99.5, 100.5, 100.2, 99.8, 100.1, 100.3])
    assert steady["steady"] is True
    assert phi35.ratio_refusal(steady, steady) is None

    drifting = stats.drift([872.8, 700.0, 600.0, 520.0, 450.0, 400.0, 360.0, 331.5])
    assert drifting["steady"] is False
    msg = phi35.ratio_refusal(steady, drifting)
    assert msg is not None and "cpu sample is not steady" in msg
    # A drifting *vulkan* sample refuses just as hard: the ratio has two operands.
    assert phi35.ratio_refusal(drifting, steady) is not None


def test_untested_steadiness_is_not_steady():
    """Three samples cannot be tested for drift; that is a refusal, not a pass."""
    untestable = stats.drift([100.0, 101.0, 99.0])
    assert untestable["steady"] is None
    msg = phi35.ratio_refusal(untestable, untestable)
    assert msg is not None and "Untested is not steady" in msg


def test_an_island_that_never_ran_is_caught_by_dispatch_accounting():
    """`compute_failures: 0` cannot see this; integer accounting can (DESIGN.md §9.1.3)."""
    ok = phi35.dispatch_accounting({"compute_calls": 7967, "compute_failures": 0}, 257, 31)
    assert ok["consistent"] is True
    # One island silently skipped on every inference: 256 x 31.
    bad = phi35.dispatch_accounting({"compute_calls": 7936, "compute_failures": 0}, 257, 31)
    assert bad["consistent"] is False
    assert "does not measure what it claims" in bad["detail"]


def test_dispatch_accounting_refuses_to_pass_when_it_cannot_check():
    assert phi35.dispatch_accounting(None, 257, 31)["consistent"] is None
    assert phi35.dispatch_accounting({"compute_calls": 10}, None, 31)["consistent"] is None


def test_a_staging_bound_run_is_labelled_even_when_the_counters_are_silent(monkeypatch):
    """The configuration is off by default, so the label does not depend on a counter existing."""
    monkeypatch.delenv(phi35.DEVICE_MEMORY_ENV, raising=False)
    label = phi35.staging_label(None)
    assert label["configuration"] == "staging-bound"
    assert label["basis"] == "configuration"
    assert "may not be quoted" in label["note"]


def test_device_memory_on_but_uninstrumented_is_unknown_not_device_backed(monkeypatch):
    """Absence of evidence is the third state, again."""
    monkeypatch.setenv(phi35.DEVICE_MEMORY_ENV, "1")
    assert phi35.staging_label({"compute_calls": 1})["configuration"] == "unknown"
    assert phi35.staging_label(
        {"alloc_staged_spans": 2510, "alloc_device_backed_spans": 0}
    )["configuration"] == "staging-bound"


# -------------------------------------------------------------------------------------------
# Wrong reading: "the spread is small, so the number is stable."
# -------------------------------------------------------------------------------------------


def test_a_monotone_ramp_is_drift_not_jitter():
    """Measured on the Iris Xe: the first Phi-3.5 run got steadily *slower*, 724 -> 2809 ms.

    A median over these samples reports a number the device never sustains and the minimum
    reports one it reached only while cold. Neither is absurd on its face.
    """
    measured = [724, 695, 903, 1447, 2080, 2669, 2821, 2765, 2802, 2718,
                2750, 2803, 2765, 2755, 2809]
    d = stats.drift([float(x) for x in measured])
    assert d["steady"] is False
    assert d["direction"] == "slower"
    assert d["ratio"] > 1.5


def test_jitter_around_a_stable_value_is_not_called_drift():
    """The detector must not fire on an ordinary noisy-but-stationary run."""
    rng = np.random.default_rng(3)
    samples = list(1000.0 + rng.normal(0, 30, 20))
    assert stats.drift(samples)["steady"] is True


def test_drift_is_not_asserted_from_too_few_samples():
    assert stats.drift([1.0, 2.0, 3.0])["steady"] is None


def test_within_run_spread_cannot_see_run_to_run_variation():
    """Two runs, each internally tight, whose medians differ by 28%. Measured, not invented."""
    records = [
        {"vulkan": {"median_ms": 2028.8}, "cpu": {"median_ms": 186.8}},
        {"vulkan": {"median_ms": 1461.8}, "cpu": {"median_ms": 190.4}},
    ]
    spread = phi35.repeat_spread(records)
    assert spread["vulkan_run_to_run_ratio"] == pytest.approx(1.388, rel=1e-2)
    assert "fewer than three repeats" in spread["note"]


def test_a_cpu_baseline_that_moved_between_workers_is_called_out():
    """Same CPU, same artifact, minutes apart: 218 ms and 665 ms. Both were really measured."""
    results = [
        {"device_index": 0, "cpu": {"median_ms": 218.3}},
        {"device_index": 1, "cpu": {"median_ms": 664.7}},
    ]
    warn = phi35.baseline_disagreement(results)
    assert warn and "3.0x" in warn
    steady = [
        {"device_index": 0, "cpu": {"median_ms": 229.8}},
        {"device_index": 1, "cpu": {"median_ms": 185.9}},
    ]
    assert phi35.baseline_disagreement(steady) is None


# -------------------------------------------------------------------------------------------
# Wrong reading: "the timestamp conversion is fine, CI is green."
# -------------------------------------------------------------------------------------------


def test_the_period_mistake_is_invisible_on_nvidia_and_on_ci():
    """Both report 1.0 ns/tick, so treating ticks as nanoseconds is *correct* there.

    This is the whole reason the bug class survives, and it is asserted so that nobody
    concludes from a green CI lane that the conversion has been exercised.
    """
    nv = timestamp_audit.conversion_self_check(1.0, 64)
    assert nv["period_mistake_is_detectable_here"] is False
    assert nv["wrap_case_ns"] is None  # 64 valid bits: the mask cannot be exercised either


def test_the_period_mistake_is_detectable_on_the_intel_part():
    xe = timestamp_audit.conversion_self_check(52.0833, 36)
    assert xe["period_mistake_is_detectable_here"] is True
    assert xe["period_error_factor"] == pytest.approx(52.0833, rel=1e-4)
    # A wrap during a measurement must come back as the short true duration, not a negative
    # one and not an hour.
    assert xe["wrap_unmasked_would_be_negative"] is True
    assert xe["wrap_case_ns"] == pytest.approx(xe["wrap_case_expected_ns"], rel=1e-9)
    assert xe["garbage_upper_bits_masked_ns"] == pytest.approx(
        xe["garbage_upper_bits_expected_ns"], rel=1e-9)


def test_the_intel_counter_wrap_period_is_about_an_hour_which_is_reachable():
    """2^36 ticks at 52.0833 ns is ~3579 s. A long benchmark session crosses it."""
    secs = timestamp_audit.wrap_period_seconds(52.0833, 36)
    assert 3500 < secs < 3650
    assert timestamp_audit.wrap_period_seconds(1.0, 64) is None


def test_an_ep_that_misreads_the_timestamp_period_is_caught_by_the_audit():
    """The red instrument: EP-reported caps cross-checked against the SDK's own dump."""
    sdk = _facts(**IRIS_XE)
    agreeing = {"timestamp_period_ns": 52.0833, "timestamp_valid_bits": 36, "uma": True}
    assert timestamp_audit.cross_check(agreeing, sdk)["agree"] is True

    # The classic: the EP reads the field but never applies it, or reads a default of 1.0.
    wrong_period = dict(agreeing, timestamp_period_ns=1.0)
    r = timestamp_audit.cross_check(wrong_period, sdk)
    assert r["agree"] is False
    assert "timestampPeriod disagrees" in r["problems"][0]

    # A 32-bit mask on a 36-bit counter: plausible, and drops four bits of every reading.
    wrong_bits = dict(agreeing, timestamp_valid_bits=32)
    assert timestamp_audit.cross_check(wrong_bits, sdk)["agree"] is False


def test_a_uma_misclassification_is_caught_too():
    """It misfiles the whole transfer-cost model, which is fitted per transfer class."""
    sdk = _facts(**RTX_4060)
    ep = {"timestamp_period_ns": 1.0, "timestamp_valid_bits": 64, "uma": True}
    r = timestamp_audit.cross_check(ep, sdk)
    assert r["agree"] is False
    assert "UMA classification disagrees" in r["problems"][0]
