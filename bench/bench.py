"""Vulkan EP benchmark runner.

Runs every case in ``cases.build_cases()`` through the Vulkan EP and through the ORT CPU EP on
the same machine, in the same process, with the same ORT build — the only baseline
``DESIGN.md`` §9.2 accepts — and reports **medians with spread**, never a single number.

Usage (Windows PowerShell)::

    $env:ONNXRUNTIME_VULKAN_EP_LIB = "rust\\target\\release\\onnxruntime_vulkan_ep.dll"
    python bench\\bench.py --out pr.json --label PR

Usage (Linux/macOS)::

    ONNXRUNTIME_VULKAN_EP_LIB=rust/target/release/libonnxruntime_vulkan_ep.so \\
      python bench/bench.py --out pr.json --label PR

What this measures, stated exactly
----------------------------------

``session.run`` wall time is **end-to-end host latency**: upload, command-buffer record (first
run) or replay, ``vkQueueSubmit``, the fence wait, and readback. It is what a user waits for,
and it is *not* GPU kernel time. On a GPU those are different numbers and conflating them is
the single easiest way to publish a wrong speedup.

To get kernel time, run again with ``--gpu-timestamps``. That sets
``ONNXRUNTIME_EP_VULKAN_TRACE_GPU=1`` and ``ONNXRUNTIME_EP_VULKAN_TRACE=<path>``; the EP writes
``VkQueryPool`` timestamps around dispatches and exports them as ``vulkan.gpu.*`` spans on a
device lane (``rust/src/trace.rs``). This run is **not** the one to quote for latency —
timestamp writes perturb the command buffer — so the harness keeps the two runs separate and
labels which is which.

Required reporting (Mouse's contract, ``OP_COVERAGE.md`` §7.3)
--------------------------------------------------------------

Every case carries ``island_count``, ``largest_island_nodes``, ``largest_island_flops``,
``boundary_bytes_per_inference`` and the ``declined`` histogram, read from the EP's own claim
log (``ONNXRUNTIME_EP_VULKAN_CLAIM_LOG``). A speedup number without those is not accepted, so
this harness does not know how to print one.

Honesty gates
-------------

* If the EP claims **no** node of a case, the case's Vulkan number is a measurement of the CPU
  EP wearing our name. It is recorded with ``claimed: false`` and marked ``⛔`` — never
  compared, never quoted.
* ``--fail-on-unclaimed`` turns that into a non-zero exit for CI.
* No number is invented anywhere. If a run does not happen, the field is ``null``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import cases as case_mod  # noqa: E402
import devices as device_mod  # noqa: E402
import environment  # noqa: E402
from stats import Sample  # noqa: E402

EP_NAME = "VulkanExecutionProvider"
EP_LIB_ENV = "ONNXRUNTIME_VULKAN_EP_LIB"
CLAIM_LOG_ENV = "ONNXRUNTIME_EP_VULKAN_CLAIM_LOG"
TRACE_ENV = "ONNXRUNTIME_EP_VULKAN_TRACE"
TRACE_GPU_ENV = "ONNXRUNTIME_EP_VULKAN_TRACE_GPU"

#: The EP is compiled against ORT 1.28 (`ORT_API_VERSION = 28`) and the project refuses 1.27
#: outright (null-allocator `PrePack` bug → NaN/Inf on fp16). An older host loader rejects the
#: plugin at registration, and every case then runs on the CPU EP *under our provider name* —
#: the exact shape of a wrong-but-plausible result, so it is checked up front and named.
MIN_ORT = (1, 28)


def _ort_version() -> "tuple[int, ...]":
    import onnxruntime as ort

    parts = []
    for chunk in ort.__version__.split("."):
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def register_ep() -> bool:
    """Register the plugin EP. Returns False (with a printed reason) when it is unavailable."""
    import onnxruntime as ort

    lib = os.environ.get(EP_LIB_ENV)
    if not lib or not Path(lib).is_file():
        print(
            f"[bench] {EP_LIB_ENV} is not set or does not exist — the Vulkan side will be "
            f"skipped and only CPU baselines are recorded.",
            file=sys.stderr,
        )
        return False

    version = _ort_version()
    if version[:2] < MIN_ORT:
        print(
            f"[bench] ⛔ onnxruntime {ort.__version__} is older than the {MIN_ORT[0]}."
            f"{MIN_ORT[1]} this EP is built against. The plugin will not load, and every case "
            f"would silently run on the CPU EP while wearing the Vulkan provider's name. "
            f"Refusing to produce Vulkan numbers.",
            file=sys.stderr,
        )
        return False

    try:
        ort.register_execution_provider_library(EP_NAME, str(Path(lib).resolve()))
        return True
    except Exception as exc:  # pragma: no cover - environment dependent
        if "already registered" in str(exc):
            return True
        print(f"[bench] EP registration failed: {exc}", file=sys.stderr)
        return False


def _session(model: bytes, providers: "list[str]", *, device_index: "int | None" = None,
             force_legacy: bool = False):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    # Graph optimisation level is pinned rather than left to default for the same reason
    # `accuracy_level` is: it is a knob whose default ORT may change, and an unpinned knob makes
    # two runs on two ORT builds incomparable without anyone noticing.
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if device_index is not None:
        # Never let the EP pick. Its scoring prefers discrete (`ENGINE.md` §2.2), so on this
        # two-GPU machine an unpinned run silently benchmarks the 4060 and a reader assumes
        # whichever device they had in mind. The device is an input to the measurement.
        opts.add_session_config_entry("ep.device_index", str(device_index))
    if force_legacy:
        opts.add_session_config_entry("ep.force_legacy_barriers", "1")
    return ort.InferenceSession(model, opts, providers=providers)


def _time_run(sess, feeds: dict, iters: int, warmup: int, name: str) -> Sample:
    """Warm up, then time ``iters`` runs, returning a robust Sample.

    Warmup matters more here than on a CPU EP and for reasons worth naming: the first run pays
    command-buffer recording (``ENGINE.md`` §6.1 records once and replays), pipeline creation
    and first-touch allocation, and a GPU that has been idle is at a low clock and takes tens of
    milliseconds to ramp. Timing a cold GPU measures the power manager.
    """
    for _ in range(warmup):
        sess.run(None, feeds)
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(None, feeds)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return Sample(name=name, samples=samples)


def _claim_metrics(model: bytes, tmp_dir: Path, tag: str, device_index: "int | None",
                   force_legacy: bool) -> dict:
    """Create a Vulkan session with the claim log on and summarise what the EP claimed.

    Uses the EP's own machine-readable claim record (``claim_log.rs``), not stderr scraping, so
    the numbers here are the EP's decisions rather than our reading of its log lines.
    """
    log_path = tmp_dir / f"_claim_{tag}_{os.getpid()}.jsonl"
    log_path.unlink(missing_ok=True)
    os.environ[CLAIM_LOG_ENV] = str(log_path)
    try:
        _session(model, [EP_NAME, "CPUExecutionProvider"], device_index=device_index,
                 force_legacy=force_legacy)
    except Exception as exc:  # pragma: no cover - environment dependent
        os.environ.pop(CLAIM_LOG_ENV, None)
        return {"claimed": False, "error": str(exc), "declined": [], "claimed_nodes": 0}
    finally:
        os.environ.pop(CLAIM_LOG_ENV, None)

    records = []
    if log_path.exists():
        for line in log_path.read_text("utf-8", "replace").splitlines():
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        log_path.unlink(missing_ok=True)

    claimed = [r for r in records if r.get("claimed")]
    declined: "dict[str, dict]" = {}
    for r in records:
        if r.get("claimed"):
            continue
        entry = declined.setdefault(r["op"], {"op": r["op"], "count": 0, "code": r.get("code"),
                                              "reason": r.get("reason")})
        entry["count"] += 1
    return {
        "claimed": bool(claimed),
        "claimed_nodes": len(claimed),
        "total_nodes": len(records),
        # island metrics are engine-side and land here once GetCapability reports them through
        # the tracer (see rust/src/trace.rs `PartitionStats`). Recorded as null rather than
        # guessed, so a missing number is visibly missing.
        "island_count": None,
        "largest_island_nodes": None,
        "largest_island_flops": None,
        "declined": sorted(declined.values(), key=lambda d: -d["count"]),
    }


class DeviceSelectionError(RuntimeError):
    """Raised when the run would not be attributable to a single named device."""


def select_device(facts: "list[device_mod.DeviceFacts]", requested: "int | None"):
    """Resolve ``--device`` into exactly one device, or refuse to run.

    This is a structural refusal, not a convention. On a machine with an Iris Xe and an RTX 4060
    the two devices differ in `timestampPeriod` (52.0833 vs 1.0), in valid timestamp bits
    (36 vs 64), in max compute shared memory (32 KiB vs 48 KiB, so a tile config tuned on one
    may not even be *selectable* on the other) and in transfer class (UMA vs discrete). A result
    file that does not name which one it ran on is not a slightly worse result file; it is not a
    result.
    """
    if requested is None:
        if len(facts) == 1:
            return facts[0]
        if not facts:
            raise DeviceSelectionError(
                "no Vulkan device facts available (is vulkaninfo on PATH / VULKAN_SDK set?). "
                "Pass --device N --i-accept-unidentified-device to benchmark anyway; the "
                "result will be marked as unattributable."
            )
        names = "\n".join(f"    --device {d.index}   {d.name} [{d.transfer_class}]" for d in facts)
        raise DeviceSelectionError(
            f"{len(facts)} Vulkan devices are present and they are not interchangeable "
            f"(different timestampPeriod, shared-memory limit and transfer class). "
            f"Choose one explicitly:\n{names}"
        )
    for d in facts:
        if d.index == requested:
            return d
    raise DeviceSelectionError(
        f"--device {requested} is not among the devices vulkaninfo reports "
        f"({[d.index for d in facts]})"
    )


def run(args: argparse.Namespace) -> dict:
    ep_available = register_ep()
    env_record = environment.capture()
    print(environment.describe(env_record))
    print()

    device_facts, device_note = device_mod.probe()
    selected = None
    if ep_available:
        try:
            selected = select_device(device_facts, args.device)
        except DeviceSelectionError:
            if not args.i_accept_unidentified_device:
                raise
        if selected is not None:
            print(selected.summary())
            print()

    if args.gpu_timestamps:
        os.environ[TRACE_GPU_ENV] = "1"
        os.environ.setdefault(TRACE_ENV, str(Path(args.out).with_suffix(".trace.json")))
        print(
            f"[bench] GPU timestamp mode: TRACE={os.environ[TRACE_ENV]}. "
            f"Latency from this run is PERTURBED by the timestamp writes and must not be "
            f"quoted as steady-state latency.\n"
        )
        if selected is not None and selected.timestamps_usable is False:
            print(
                f"[bench] REFUSING GPU timestamps: device {selected.index} reports "
                f"timestampValidBits == 0, which means this queue family produces no GPU "
                f"timestamps at all. Zeros would be indistinguishable from instant kernels.\n"
            )
            os.environ.pop(TRACE_GPU_ENV, None)

    device_index = selected.index if selected is not None else args.device

    results = []
    for c in case_mod.build_cases(args.groups):
        claim = (
            _claim_metrics(c.model, _HERE, c.name, device_index, args.force_legacy_barriers)
            if ep_available
            else {
                "claimed": False,
                "claimed_nodes": 0,
                "declined": [],
                "note": "EP not registered",
            }
        )

        cpu_sample = _time_run(
            _session(c.model, ["CPUExecutionProvider"]), c.feeds, args.iters, args.warmup,
            f"{c.name}/cpu",
        )
        vk_sample = None
        if ep_available:
            vk_sample = _time_run(
                _session(c.model, [EP_NAME, "CPUExecutionProvider"],
                         device_index=device_index,
                         force_legacy=args.force_legacy_barriers),
                c.feeds, args.iters, args.warmup, f"{c.name}/vulkan",
            )

        speedup = None
        if vk_sample and claim.get("claimed") and vk_sample.median > 0:
            speedup = cpu_sample.median / vk_sample.median

        row = {
            "name": c.name,
            "group": c.group,
            "note": c.note,
            "tags": c.tags,
            "oq12_anchor": c.oq12_anchor,
            "flops": c.flops,
            "boundary_bytes_per_inference": c.boundary_bytes,
            "cpu": cpu_sample.to_dict(raw=args.raw),
            "vulkan": vk_sample.to_dict(raw=args.raw) if vk_sample else None,
            # Speedup is None unless the EP actually claimed the graph — a CPU-fallback
            # "speedup" of 1.0x is a lie with a decimal point.
            "speedup_end_to_end": round(speedup, 3) if speedup else None,
            "gpu_kernel_ms": None,  # filled only from trace timestamps; see --gpu-timestamps
            # A tile configuration is part of the identity of the kernel that was measured.
            # Until the engine reports it, this stays null and `compare.py` treats two nulls as
            # "unknown", never as "the same".
            "tile_config": None,
            "claim": claim,
            "measurement": "end-to-end host latency (upload+record/replay+submit+fence+readback)",
        }
        if speedup and c.flops:
            row["achieved_gflops_end_to_end"] = round(
                c.flops / (vk_sample.median * 1e6), 3
            )
        results.append(row)

        mark = "" if claim.get("claimed") else "  ⛔ NOT CLAIMED — CPU fallback, not a Vulkan number"
        sp = f"{speedup:5.2f}x" if speedup else "   —  "
        print(f"  {c.name:38s} vulkan={vk_sample.summary() if vk_sample else 'skipped':<70s}")
        print(f"  {'':38s} cpu   ={cpu_sample.summary()}")
        print(f"  {'':38s} ratio ={sp}{mark}\n", flush=True)

    return {
        "label": args.label,
        "iters": args.iters,
        "warmup": args.warmup,
        "gpu_timestamps": args.gpu_timestamps,
        "ep_available": ep_available,
        "barrier_backend": "legacy (forced)" if args.force_legacy_barriers else "device default",
        "environment": env_record,
        "device": selected.to_dict() if selected is not None else None,
        "device_fingerprint": selected.fingerprint if selected is not None else None,
        "device_note": device_note,
        "all_devices": [d.to_dict() for d in device_facts],
        "cases": results,
        "disclaimer": (
            "Vulkan numbers are end-to-end host latency, not GPU kernel time. Cases marked "
            "not-claimed ran on the CPU EP. See docs/PERF.md."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--iters", type=int, default=50, help="timed iterations per case")
    ap.add_argument("--warmup", type=int, default=15,
                    help="untimed iterations first (record path + GPU clock ramp)")
    ap.add_argument("--label", default="local")
    ap.add_argument("--device", type=int, default=None,
                    help="physical device index to pin (ep.device_index). Required when the "
                         "machine has more than one Vulkan device — they are not "
                         "interchangeable.")
    ap.add_argument("--i-accept-unidentified-device", action="store_true",
                    help="run even though the device cannot be identified; the result is "
                         "marked unattributable and must not be quoted")
    ap.add_argument("--force-legacy-barriers", action="store_true",
                    help="force the legacy vkCmdPipelineBarrier backend (ep.force_legacy_"
                         "barriers=1). Recorded in the result: the two backends are two "
                         "different programs and their timings are not interchangeable.")
    ap.add_argument("--groups", nargs="*", default=None,
                    help="restrict to case groups (elementwise, gemm)")
    ap.add_argument("--raw", action="store_true", help="include every raw sample in the JSON")
    ap.add_argument("--gpu-timestamps", action="store_true",
                    help="enable VkQueryPool timestamps + trace export (perturbs latency)")
    ap.add_argument("--fail-on-unclaimed", action="store_true",
                    help="exit non-zero if any case fell back to the CPU EP")
    ap.add_argument("--print-env", action="store_true",
                    help="print the environment record and exit without benchmarking")
    args = ap.parse_args()

    if args.print_env:
        print(environment.describe(environment.capture()))
        facts, why = device_mod.probe()
        print()
        if why:
            print(why)
        for d in facts:
            print(d.summary())
            print()
        return 0

    try:
        data = run(args)
    except DeviceSelectionError as exc:
        print(f"⛔ {exc}", file=sys.stderr)
        return 2
    Path(args.out).write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"wrote {len(data['cases'])} case(s) -> {args.out}")
    if data.get("device_fingerprint"):
        print(f"device: {data['device_fingerprint']}")
    else:
        print("⚠ device could not be identified — this result is not attributable and must "
              "not be quoted as a device measurement")

    unclaimed = [c["name"] for c in data["cases"] if not c["claim"].get("claimed")]
    if unclaimed:
        print(f"⛔ {len(unclaimed)} case(s) were not claimed by the EP: {', '.join(unclaimed)}")
        if args.fail_on_unclaimed:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
