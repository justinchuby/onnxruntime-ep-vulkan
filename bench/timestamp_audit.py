"""Audit the GPU-timestamp path against the two real devices on this desk.

WHAT THIS IS FOR
================
`rust/src/trace.rs` converts Vulkan query-pool ticks to nanoseconds. Two device properties
enter that conversion, and on this machine they differ by more than a factor of fifty:

    device                     timestampPeriod   timestampValidBits   wrap period
    Intel Iris Xe              52.0833 ns/tick   36                   ~3579 s (~1 hour)
    NVIDIA RTX 4060 Laptop      1.0    ns/tick   64                   never (in practice)
    lavapipe (WSL, CI)          1.0    ns/tick   64                   never

Both mistakes available here are *silent* and *plausible*:

1. **Ignoring ``timestampPeriod``.** Ticks are treated as nanoseconds. On NVIDIA and on
   lavapipe that is exactly right, so CI is green and the desk's discrete GPU agrees. On Intel
   every duration is reported **52× too small** — a kernel that took 5.2 ms is reported as
   100 µs. Nothing is negative, nothing is absurd, and the number is wrong by a constant.
   This is the same shape as the tracer-epoch hazard: a reading that looks reasonable and is
   off by a fixed factor is the one that survives for months.
2. **Ignoring ``timestampValidBits``.** The upper bits of a query result are *undefined* when
   ``validBits < 64``. Unmasked, garbage in bits 36..63 makes the delta enormous or negative.
   Worse, a genuine wrap during a measurement (an hour of GPU uptime on Intel is not exotic)
   yields a *negative* delta that, taken as unsigned, becomes an enormous positive duration.

WHAT THIS MODULE VERIFIES, AND WHAT EACH PART CANNOT
=====================================================
**One — the inputs to the conversion,** on real hardware, by cross-checking two independent
instruments that read them by different routes:

* ``epctl --probe-loader`` — the EP's own capability probe (``rust/src/vk/caps.rs``), i.e. the
  values the conversion will actually be handed at run time.
* ``vulkaninfoSDK`` — the SDK's dump, i.e. what the driver reports to a program that is not us.

If those disagree, the EP is reading the wrong field or the wrong queue family, and this exits
non-zero. That is the red instrument for the inputs.

**Two — the conversion end to end,** from a real trace, with ``--trace``. This became possible on
2026-07-30, when the ``VkQueryPool`` path landed (``rust/src/vk/timestamp.rs`` hands back raw,
unmasked ticks; ``trace.rs::GpuTimestampCalibration::ticks_to_ns`` applies the mask, the
single-wrap recovery and the period scale; ``vk/session.rs`` composes them). The previous text
here — *"it cannot verify the conversion end to end, because no ``VkQueryPool`` exists"* — was
**true when written and is retired as of that date**, rather than left standing to be read as
current by everyone who arrives later.

The end-to-end check is ``phases.timestamp_conversion_integrality``: an emitted ``gpu_ns`` is a
tick count times the period, so ``gpu_ns ÷ period`` must be a whole number. A build that dropped
the period scale emits raw ticks — integers — and dividing those by 52.0833 gives a fraction.
**It is decisive only where the period is not 1.0**, which on this desk means the Intel part
alone; on NVIDIA and lavapipe it is reported ``VACUOUS``, never as a pass. That asymmetry is why
the Iris Xe is the only instrument here for this bug class and CI has none.

Usage::

    python bench/timestamp_audit.py
    python bench/timestamp_audit.py --trace bench/_scratch/phi35_trace_dev1.trace.json
    python bench/timestamp_audit.py --json bench/results/timestamp-audit.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import devices as device_mod  # noqa: E402

EPCTL_ENV = "ONNXRUNTIME_EP_VULKAN_EPCTL"

#: Relative tolerance for comparing two reports of the same float property. The value is a
#: driver constant reported through two paths, so it should be bit-identical; the tolerance
#: exists only to absorb decimal formatting (``52.0833`` vs ``52.083302``), not disagreement.
PERIOD_RTOL = 1e-4

_DEVICE_HEADER = re.compile(r"^Device (?P<i>\d+): (?P<name>.+?) \[Vulkan", re.M)
_PERIOD = re.compile(r"timestamp_period_ns\s*:\s*(?P<v>[0-9.]+)")
_BITS = re.compile(r"timestamp_valid_bits\s*:\s*(?P<v>\d+)")
_UMA = re.compile(r"is_uma\s*:\s*(?P<v>true|false)")


def find_epctl() -> "Path | None":
    override = os.environ.get(EPCTL_ENV)
    if override and Path(override).is_file():
        return Path(override)
    for rel in ("rust/target/release/epctl.exe", "rust/target/release/epctl",
                "rust/target/debug/epctl.exe", "rust/target/debug/epctl"):
        p = _HERE.parent / rel
        if p.is_file():
            return p
    return None


def probe_ep(epctl: Path) -> "dict[int, dict]":
    """Parse ``epctl --probe-loader`` into per-device capability facts as the EP sees them."""
    proc = subprocess.run([str(epctl), "--probe-loader"], capture_output=True, text=True,
                          timeout=300)
    text = proc.stdout + proc.stderr
    out: "dict[int, dict]" = {}
    headers = list(_DEVICE_HEADER.finditer(text))
    for n, h in enumerate(headers):
        end = headers[n + 1].start() if n + 1 < len(headers) else len(text)
        block = text[h.start():end]
        idx = int(h.group("i"))
        entry: dict = {"index": idx, "name": h.group("name").strip()}
        if (m := _PERIOD.search(block)):
            entry["timestamp_period_ns"] = float(m.group("v"))
        if (m := _BITS.search(block)):
            entry["timestamp_valid_bits"] = int(m.group("v"))
        if (m := _UMA.search(block)):
            entry["uma"] = m.group("v") == "true"
        out[idx] = entry
    return out


# ---------------------------------------------------------------------------
# The conversion, mirrored from rust/src/trace.rs
# ---------------------------------------------------------------------------

def mask_ticks(raw: int, valid_bits: int) -> int:
    """Mirror of ``trace.rs::mask_ticks``. Bits above ``valid_bits`` are undefined, not zero."""
    if valid_bits >= 64:
        return raw & 0xFFFF_FFFF_FFFF_FFFF
    if valid_bits == 0:
        return 0
    return raw & ((1 << valid_bits) - 1)


def span_ns(start_raw: int, end_raw: int, period_ns: float, valid_bits: int) -> "float | None":
    """Mirror of ``trace.rs::GpuTimestampCalibration::span_ns``.

    Masks first, then handles wrap by adding the modulus when the end precedes the start, then
    scales by the period. All three steps are load-bearing on the Intel part and none of them is
    on the NVIDIA one, which is exactly why the audit runs on both.
    """
    if valid_bits == 0 or period_ns <= 0:
        return None
    a, b = mask_ticks(start_raw, valid_bits), mask_ticks(end_raw, valid_bits)
    span = b - a
    if span < 0:
        if valid_bits >= 64:
            return None
        span += 1 << valid_bits
    return span * period_ns


def wrap_period_seconds(period_ns: float, valid_bits: int) -> "float | None":
    if valid_bits == 0 or valid_bits >= 64 or period_ns <= 0:
        return None
    return (1 << valid_bits) * period_ns / 1e9


# ---------------------------------------------------------------------------
# The audit
# ---------------------------------------------------------------------------

def cross_check(ep: dict, sdk: "device_mod.DeviceFacts") -> dict:
    """Compare the EP's reading of a device against the SDK's. Disagreement is a failure."""
    problems: "list[str]" = []
    e_p, s_p = ep.get("timestamp_period_ns"), sdk.timestamp_period_ns
    if e_p is None or s_p is None:
        problems.append("timestampPeriod missing from one of the two instruments")
    elif abs(e_p - s_p) > PERIOD_RTOL * max(abs(s_p), 1e-9):
        problems.append(
            f"timestampPeriod disagrees: EP reports {e_p}, vulkaninfo reports {s_p}. Every GPU "
            f"duration would be wrong by a factor of {e_p / s_p:.4g}.")
    e_b, s_b = ep.get("timestamp_valid_bits"), sdk.timestamp_valid_bits
    if e_b is None or s_b is None:
        problems.append("timestampValidBits missing from one of the two instruments")
    elif e_b != s_b:
        problems.append(
            f"timestampValidBits disagrees: EP reports {e_b}, vulkaninfo reports {s_b}. The mask "
            f"would be the wrong width and wrapped or garbage-topped results would go unnoticed.")
    if ep.get("uma") is not None and sdk.uma is not None and ep["uma"] != sdk.uma:
        problems.append(
            f"UMA classification disagrees: EP says {ep['uma']}, vulkaninfo-derived says "
            f"{sdk.uma}. Transfer-cost models are fitted per transfer class, so this misfiles "
            f"the whole model.")
    return {
        "index": sdk.index,
        "name": sdk.name,
        "ep_period_ns": e_p,
        "sdk_period_ns": s_p,
        "ep_valid_bits": e_b,
        "sdk_valid_bits": s_b,
        "ep_uma": ep.get("uma"),
        "sdk_uma": sdk.uma,
        "wrap_period_s": wrap_period_seconds(s_p or 0.0, s_b or 0),
        "agree": not problems,
        "problems": problems,
    }


def conversion_self_check(period_ns: float, valid_bits: int) -> dict:
    """Exercise the conversion with this device's real constants, including the wrong readings.

    Each entry names the *wrong* answer as well as the right one, because the point is not that
    the right answer comes out — it is that the wrong answer is numerically distinguishable from
    it on this device. On the RTX 4060 the naive readings coincide with the correct one, and
    that is recorded rather than hidden: it means NVIDIA and CI cannot falsify this at all, and
    the Intel part is the only local instrument that can.
    """
    ticks = 100_000
    correct = span_ns(1_000, 1_000 + ticks, period_ns, valid_bits)
    naive_period = float(ticks)  # ticks read as nanoseconds
    checks = {
        "ticks": ticks,
        "correct_ns": correct,
        "unscaled_ns": naive_period,
        "period_error_factor": (correct / naive_period) if correct else None,
        "period_mistake_is_detectable_here": bool(correct is not None
                                                  and abs(correct - naive_period) > 1e-6),
    }
    if valid_bits < 64:
        modulus = 1 << valid_bits
        start = modulus - 1_000
        end = 500  # wrapped
        checks["wrap_case_ns"] = span_ns(start, end, period_ns, valid_bits)
        checks["wrap_case_expected_ns"] = 1_500 * period_ns
        checks["wrap_unmasked_would_be_negative"] = (end - start) < 0
        # Garbage in the undefined upper bits must not survive the mask.
        garbage = (0xDEAD << valid_bits) | 4_000
        checks["garbage_upper_bits_masked_ns"] = span_ns(1_000, garbage, period_ns, valid_bits)
        checks["garbage_upper_bits_expected_ns"] = 3_000 * period_ns
        checks["wrap_period_s"] = wrap_period_seconds(period_ns, valid_bits)
    else:
        checks["wrap_case_ns"] = None
        checks["note"] = ("valid_bits == 64: no mask and no wrap are exercisable on this device. "
                          "It cannot falsify the masking path.")
    return checks


def end_to_end(trace_path: "str | Path") -> dict:
    """Check the emitted GPU durations against the device period they claim to have used.

    Consumes a trace written by a run with ``ONNXRUNTIME_EP_VULKAN_TRACE_GPU=1``. The check
    itself lives in :mod:`phases` so that the benchmark and the audit cannot drift into two
    different definitions of "the conversion is right".
    """
    import phases  # noqa: PLC0415 - imported here so the audit still runs without a trace

    p = Path(trace_path)
    if not p.is_file():
        return {"status": "UNMEASURED",
                "reason": f"no trace at {p}. Run the benchmark with the phase pass enabled, or "
                          f"set ONNXRUNTIME_EP_VULKAN_TRACE and ONNXRUNTIME_EP_VULKAN_TRACE_GPU."}
    events = phases.load(p)
    gpus = phases.gpu_spans(events)
    if not gpus:
        return {"status": "UNMEASURED",
                "reason": f"{p} contains no vulkan.gpu.* spans: the run produced no timestamp "
                          f"results, so there is no tick to check the conversion against."}
    integral = phases.timestamp_conversion_integrality(gpus)
    mask = phases.valid_bits_applied(gpus)
    return {
        "status": ("RED" if (integral.get("red") or mask.get("red")) else
                   "VERIFIED" if integral.get("decisive") else "VACUOUS"),
        "trace": str(p),
        "gpu_spans": len(gpus),
        "integrality": integral,
        "valid_bits": mask,
        "note": ("the period scale is applied end to end on a device where dropping it would be "
                 "detectable" if integral.get("decisive") and not integral.get("red") else
                 "every device in this trace reports timestampPeriod 1.0, so this trace cannot "
                 "falsify a dropped period scale. Not a pass — run the Intel part."),
    }


def audit() -> dict:
    epctl = find_epctl()
    sdk_facts, sdk_source = device_mod.probe()
    _gpu_trace_requested = bool(os.environ.get("ONNXRUNTIME_EP_VULKAN_TRACE_GPU"))
    report: dict = {
        "epctl": str(epctl) if epctl else None,
        "vulkaninfo_source": sdk_source,
        "gpu_kernel_time_available": True,
        "gpu_kernel_time_status": "ACTIVE" if _gpu_trace_requested else "IMPLEMENTED_NOT_REQUESTED",
        "gpu_kernel_time_reason": (
            "VkQueryPool timestamps are implemented in rust/src/vk/timestamp.rs and wired in "
            "rust/src/vk/session.rs via GpuQueryPool::cmd_before/cmd_after around each "
            "vkCmdDispatch. Set ONNXRUNTIME_EP_VULKAN_TRACE_GPU=1 to enable per-dispatch "
            "GPU kernel timing. The conversion (period scaling + valid-bit masking) is covered "
            "by unit tests in trace.rs using the real hardware constants measured here."
            if not _gpu_trace_requested else
            "ONNXRUNTIME_EP_VULKAN_TRACE_GPU=1 is set; GPU timestamps will be written around "
            "each vkCmdDispatch and reported as vulkan.gpu.* spans on the device lane in the "
            "Chrome Trace JSON output."
        ),
        "devices": [],
        "problems": [],
    }
    if not epctl:
        report["problems"].append(
            "epctl was not found; the EP's own view of the timestamp properties cannot be "
            "cross-checked against the SDK's. Build it: cargo build --release.")
    if not sdk_facts:
        report["problems"].append(
            "no device facts from vulkaninfo; nothing to cross-check against.")
    if report["problems"]:
        return report

    ep_facts = probe_ep(epctl)
    for f in sdk_facts:
        ep = ep_facts.get(f.index)
        if ep is None:
            report["problems"].append(
                f"device {f.index} ({f.name}) is visible to vulkaninfo but not to epctl's probe. "
                f"The two instruments do not see the same machine.")
            continue
        entry = cross_check(ep, f)
        entry["conversion"] = conversion_self_check(f.timestamp_period_ns or 0.0,
                                                    f.timestamp_valid_bits or 0)
        report["devices"].append(entry)
        report["problems"].extend(f"device {f.index}: {p}" for p in entry["problems"])

    detectors = [d for d in report["devices"]
                 if d["conversion"].get("period_mistake_is_detectable_here")]
    report["period_mistake_detectable_on"] = [d["name"] for d in detectors]
    if not detectors:
        report["problems"].append(
            "no device on this machine has timestampPeriod != 1.0, so treating ticks as "
            "nanoseconds would be indistinguishable from the correct conversion here. That is a "
            "gap in the instrument set, not a pass.")
    maskers = [d for d in report["devices"] if (d["sdk_valid_bits"] or 64) < 64]
    report["mask_exercisable_on"] = [d["name"] for d in maskers]
    if not maskers:
        report["problems"].append(
            "no device on this machine reports timestampValidBits < 64, so an unmasked read "
            "cannot be falsified here.")
    return report


def describe(report: dict) -> str:
    out = ["=" * 78, "GPU timestamp path audit", "=" * 78,
           f"epctl        : {report.get('epctl')}",
           f"vulkaninfo   : {report.get('vulkaninfo_source')}",
           f"GPU kernel time: {report['gpu_kernel_time_status']} — "
           f"{report['gpu_kernel_time_reason']}", ""]
    for d in report.get("devices", []):
        c = d["conversion"]
        out.append(f"### {d['name']} (vkEnumeratePhysicalDevices index {d['index']})")
        # NOT `ep.device_index`. `engine.rs::probe_devices` sorts best-first, so on a laptop with
        # an iGPU and a dGPU the two orderings are reversed. See devices.ep_selection_order.
        out.append(f"  timestampPeriod   EP {d['ep_period_ns']}  vulkaninfo {d['sdk_period_ns']}"
                   f"   {'agree' if d['agree'] else 'DISAGREE'}")
        out.append(f"  timestampValidBits EP {d['ep_valid_bits']}  vulkaninfo "
                   f"{d['sdk_valid_bits']}")
        wrap = d.get("wrap_period_s")
        out.append(f"  counter wraps every: "
                   + (f"{wrap:.0f} s ({wrap / 3600:.2f} h)" if wrap else "never (64 valid bits)"))
        out.append(f"  UMA               EP {d['ep_uma']}  vulkaninfo-derived {d['sdk_uma']}")
        if c.get("correct_ns") is not None:
            out.append(f"  {c['ticks']} ticks -> {c['correct_ns'] / 1e6:.4f} ms correctly; "
                       f"{c['unscaled_ns'] / 1e6:.4f} ms if the period were ignored "
                       f"({'DETECTABLE here' if c['period_mistake_is_detectable_here'] else 'INDISTINGUISHABLE here'})")
        if c.get("wrap_case_ns") is not None:
            out.append(f"  wrapped span      {c['wrap_case_ns']:.1f} ns "
                       f"(expected {c['wrap_case_expected_ns']:.1f}); unmasked it would be "
                       f"{'negative' if c['wrap_unmasked_would_be_negative'] else 'positive'}")
            out.append(f"  garbage upper bits {c['garbage_upper_bits_masked_ns']:.1f} ns "
                       f"(expected {c['garbage_upper_bits_expected_ns']:.1f})")
        for p in d["problems"]:
            out.append(f"  ⛔ {p}")
        out.append("")
    out.append(f"period mistake is detectable on: "
               f"{report.get('period_mistake_detectable_on') or 'NO LOCAL DEVICE'}")
    out.append(f"valid-bit mask is exercisable on: "
               f"{report.get('mask_exercisable_on') or 'NO LOCAL DEVICE'}")
    if report["problems"]:
        out.append("")
        out.append("PROBLEMS:")
        out.extend(f"  - {p}" for p in report["problems"])
    return "\n".join(out)


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", help="write the full report here")
    ap.add_argument("--trace", action="append",
                    help="a Chrome Trace JSON from a run with ONNXRUNTIME_EP_VULKAN_TRACE_GPU=1; "
                         "checks the conversion end to end. Repeat for several devices.")
    a = ap.parse_args(argv)
    report = audit()
    if a.trace:
        report["end_to_end"] = [end_to_end(t) for t in a.trace]
        for e in report["end_to_end"]:
            if e["status"] == "RED":
                report["problems"].append(
                    f"end-to-end conversion check on {e['trace']} went RED: "
                    f"{e['integrality'].get('results')}")
    print(describe(report))
    for e in report.get("end_to_end", []):
        print("")
        print(f"end-to-end conversion ({e.get('trace')}): {e['status']} — {e.get('note') or e.get('reason')}")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(report, indent=2), "utf-8")
    return 1 if report["problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
