"""What can the Windows GPU counters actually witness on the Intel adapter? — measured, with a
negative control on the other adapter.

The question this answers is a **capability** question, not a performance one. `bench/device_state.py`
requires a tenancy verdict and a clock record before a device-clock figure may be quoted, and its
only producer is `nvidia-smi`, so every Intel figure this project has ever taken is `UNCERTIFIED`.
Windows' `\\GPU Engine(*)` counters are produced by the WDDM scheduler rather than by a vendor tool,
so they are a candidate producer for the **tenancy half** on any adapter.

A candidate is not a capability until it has been shown to see the thing it claims to see, and the
way to show that is to run work we control and check that it appears — **and check that it does not
appear where it should not.** So this probe samples the target adapter *and* every other live
adapter over the same window:

* our worker's engine time must **rise on the adapter we ran on**;
* it must **stay at zero on the other one**.

Without the second arm, "our PID appeared in the counters" would be consistent with the LUID join
being wrong in a way that happens to include us — and the LUID join is a name match against a
registry key, which is exactly the kind of join that is silently wrong.

The probe reports no performance figure and never will. It answers: *is this adapter's tenancy
observable at all, and by what evidence.*

Usage::

    python bench/results/probe_wingpu.py --device 1 --iters 6 --tag intel
    python bench/results/probe_wingpu.py --device 0 --iters 6 --tag nvidia
    python bench/results/probe_wingpu.py --idle 20 --tag idle      # no workload: who else is on it
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))

import win_gpu_counters as wgc  # noqa: E402

TRACE_ENV = "ONNXRUNTIME_EP_VULKAN_TRACE"
TRACE_GPU_ENV = "ONNXRUNTIME_EP_VULKAN_TRACE_GPU"


def adapter_name_for_device(index: int) -> "tuple[str | None, str]":
    """Vulkan device name for ``ep.device_index == index``, which is what the LUID join keys on.

    **Not** ``devices.by_index``. There are two orderings on this machine and they disagree:
    ``vkEnumeratePhysicalDevices`` order (0 = Intel, 1 = NVIDIA) and the EP's best-first selection
    order (0 = NVIDIA, 1 = Intel), and ``ONNXRUNTIME_EP_VULKAN_DEVICE`` indexes the second.
    Using the first here pointed the sampler at the NVIDIA board while the workload ran on the
    Iris Xe, and the record came back a clean ``SOLE_TENANT`` — see
    ``win_gpu_counters.TENANCY_UNWITNESSED``, which exists because of that run.
    """
    try:
        import devices
        facts, _note = devices.probe()
        chosen = devices.by_ep_index(facts, index)
        if chosen:
            return chosen.name, (f"bench/devices.py ep_selection_order[{index}] "
                                 f"(enumeration index {chosen.index})")
    except Exception as exc:
        return None, f"ERROR(instrument=devices): {exc!r}"
    return None, "no device at that ep index"


def start_all(target_luid: str, interval: float) -> "dict[str, wgc.Sampler]":
    """One sampler per live adapter: the target and every control arm."""
    out: "dict[str, wgc.Sampler]" = {}
    for luid in sorted(wgc.live_luids()):
        sampler = wgc.Sampler(luid, interval=interval)
        sampler.open()
        sampler.start()
        out[luid] = sampler
    if target_luid not in out:
        raise wgc.CounterError(f"target LUID {target_luid} has no live counter instances")
    return out


def stop_all(samplers: "dict[str, wgc.Sampler]") -> "dict[str, dict]":
    for s in samplers.values():
        s.stop.set()
    for s in samplers.values():
        s.join(timeout=15.0)
    return {luid: wgc.summarise(s) for luid, s in samplers.items()}


def run_workload(device: int, iters: int, warmup: int, scratch: Path, on_start) -> dict:
    import phi35

    scratch.mkdir(parents=True, exist_ok=True)
    out = scratch / f"probe_wingpu_dev{device}.json"
    trace = scratch / f"probe_wingpu_dev{device}.trace.json"
    for p in (out, trace):
        p.unlink(missing_ok=True)
    env = dict(os.environ)
    env[TRACE_ENV] = str(trace)
    env[TRACE_GPU_ENV] = "1"
    cmd = [sys.executable, str(BENCH / "phi35.py"), "--worker", "--device", str(device),
           "--iters", str(iters), "--warmup", str(warmup),
           "--out", str(out), "--scratch", str(scratch)]
    proc, instrument_error = phi35._run_worker(cmd, env, on_start=on_start)
    rec: dict = {}
    if out.exists():
        try:
            rec = json.loads(out.read_text("utf-8"))
        except Exception:
            rec = {}
    rec["_worker_exit"] = None if proc is None else proc.returncode
    rec["_instrument_error"] = instrument_error or None
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=1, help="Vulkan device index to run on")
    ap.add_argument("--adapter", default=None, help="override the adapter description to join on")
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--idle", type=float, default=0.0,
                    help="observe for N seconds with no workload instead of running one")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--tag", default="wingpu")
    ap.add_argument("--nvsmi", type=int, default=None,
                    help="also run the NVIDIA clock+tenancy producer on this board index, so the "
                         "two independently-authored instruments can be compared (obligation 7)")
    args = ap.parse_args()

    if not wgc.available():
        print(json.dumps({"verdict": "UNOBSERVABLE",
                          "reason": "no WDDM GPU Engine counters on this host"}, indent=2))
        return 0

    name = args.adapter
    source = "--adapter"
    if name is None:
        name, source = adapter_name_for_device(args.device)
    if not name:
        print(f"could not name the adapter for device {args.device} ({source})")
        return 2
    luid = wgc.luid_for_adapter(name)

    samplers = start_all(luid, args.interval)
    nv = None
    if args.nvsmi is not None:
        import device_state
        nv = device_state.Companion(board_index=args.nvsmi, vendor_is_nvidia=True,
                                    allow_windows_tenancy=False).start()
    started = time.time()
    worker: "dict | None" = None
    if args.idle > 0:
        time.sleep(args.idle)
    else:
        def on_start(pid: int) -> None:
            for s in samplers.values():
                s.own_root = pid
            if nv is not None:
                nv.own_root(pid)

        worker = run_workload(args.device, args.iters, args.warmup,
                              BENCH / "_scratch" / f"wingpu_{args.tag}", on_start)
    records = stop_all(samplers)
    nv_record = nv.stop() if nv is not None else None

    target = records[luid]
    controls = {k: v for k, v in records.items() if k != luid}
    report = {
        "tag": args.tag,
        "seconds": round(time.time() - started, 2),
        "device_index": args.device,
        "adapter": {"name": name, "name_source": source, "luid": luid},
        "all_adapters": wgc.adapters(),
        "target": target,
        "controls": controls,
        "worker": None if worker is None else {
            "exit": worker.get("_worker_exit"),
            "instrument_error": worker.get("_instrument_error"),
            "verdict": worker.get("model_output_equivalence"),
        },
        "capability": _capability(target, controls),
    }
    if nv_record is not None:
        import device_state
        report["nvidia_companion"] = nv_record
        report["corroboration"] = {
            "instrument_a": "nvidia-smi (bench/results/probe_gpustate.py, Switch)",
            "instrument_b": "WDDM \\GPU Engine (bench/win_gpu_counters.py, Niobe)",
            "tenancy_a": nv_record.get("verdict"),
            "tenancy_b": target.get("verdict"),
            "agree": device_state._tenancy_agrees(nv_record.get("verdict"), target.get("verdict")),
            "note": ("Two independently-authored instruments on the same question over the same "
                     "window. Agreement is evidence about the tenancy axis only; neither says "
                     "anything about the other's clock reading."),
        }
    path = BENCH / "results" / f"wingpu-{args.tag}-dev{args.device}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("tag", "adapter", "capability")}, indent=2))
    print(f"\ntarget  : {target.get('verdict')}  ours={target.get('own_gpu_seconds')} "
          f"foreign={target.get('foreign_gpu_seconds')} engines={target.get('engine_types')}")
    for k, v in controls.items():
        print(f"control : {k} ours={v.get('own_gpu_seconds')} verdict={v.get('verdict')}")
    print(f"\nwritten: {path}")
    return 0


def _capability(target: dict, controls: "dict[str, dict]") -> dict:
    """State what the counters witnessed, in the form of a claim someone can falsify."""
    ours_here = sum((target.get("own_gpu_seconds") or {}).values())
    ours_elsewhere = {k: sum((v.get("own_gpu_seconds") or {}).values()) for k, v in controls.items()}
    leaked = {k: v for k, v in ours_elsewhere.items() if v > 0}
    return {
        "our_work_seen_on_target": ours_here > 0,
        "our_engine_seconds_on_target": round(ours_here, 4),
        "our_engine_seconds_on_other_adapters": {k: round(v, 4) for k, v in ours_elsewhere.items()},
        "negative_control_holds": not leaked,
        "tenancy_axis": target.get("verdict"),
        "clock_axis": (target.get("clock") or {}).get("verdict"),
        "statement": (
            "the WDDM counters witness our own submissions on this adapter and no clock"
            if ours_here > 0 and not leaked else
            "our own work was not witnessed on this adapter; the tenancy claim is unsupported here"
        ),
    }


if __name__ == "__main__":  # pragma: no cover - manual use
    raise SystemExit(main())
