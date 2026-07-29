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

import compare  # noqa: E402
import devices as D  # noqa: E402
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
    return {
        "label": "x",
        "device": dev.to_dict(),
        "device_fingerprint": dev.fingerprint,
        "barrier_backend": kw.get("backend", "device default"),
        "cases": [
            {
                "name": "case",
                "claim": {"claimed": True},
                "tile_config": kw.get("tile"),
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
