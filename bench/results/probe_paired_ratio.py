"""Is a paired, interleaved A/B ratio a sound instrument on a permanently contended box?

`PERF.md` §20 makes an absolute wall-clock figure `STEADY_UNCERTIFIED` by default on this
machine and forbids any plan whose step is *"take the measurement when the box settles"* — that
step never completes.  The proposal this probe exists to test is that a **ratio** escapes the
refusal where an absolute number cannot:

    run our EP and a reference back to back, alternating, in one process, on the same inputs,
    interleaved finely enough that a contention episode lands on both arms, and publish the
    ratio with its dispersion — never the absolute.

That is the standard paired design for a noisy shared environment, and it is *measuring* under
contention rather than *waiting* for it to end, so §20 does not forbid it on its face.  **It is
also an assumption**, and this probe's first job is to attack it rather than to use it.

# The assumption, stated so it can fail

A paired ratio is unbiased by a disturbance only if the disturbance is **common-mode**: a
multiplicative factor that lands on both arms with the same gain.  Two ways that fails here, and
both are measured rather than argued:

1. **Granularity.** A foreign kernel that runs for 200 ms hits whichever arm is executing.  Our
   two arms are wildly unequal in duration — a Vulkan decode step is tens of milliseconds and a
   CPU EP decode step is hundreds — so an episode of any length is *more likely* to land on the
   CPU arm simply because that arm is executing more of the wall clock.  Interleaving at one
   decode step is the finest granularity a decode chain admits: a step is atomic, there is no
   smaller unit to alternate on, and that granularity is what it is rather than what we would
   like.

2. **Mechanism (the one that is specific to this project).**  §20.2: on this box, host contention
   arrives on the device axis **as an idle clock**, not as GPU contention.  The board sees too
   little work, does not ramp, and the specimen this project holds is 20.18× wrong while reading
   `SOLE_TENANT` and RSD 0.0717%.  A CPU arm has no such gain.  Worse, and this is the part the
   design creates for itself: **alternating means the GPU sits idle for the whole of every CPU
   step** — hundreds of milliseconds, every pair, by construction.  The interleaving that makes a
   *foreign* episode symmetric manufactures an *own* asymmetry on the device axis.  That is why
   this probe runs a `blocked` phase as well: the same arms, un-interleaved, with the clock
   recorded, so the cost of pairing itself is a measurement and not a worry.

# What is injected, and why an injection is the only honest way to answer this

An observational run cannot separate "the ratio is stable" from "nothing happened while we
watched".  So the disturbance is **applied**, with its size known, on each axis separately:

* `cpuload`  — N spinning processes.  Predicted to hit the CPU arm hard and the Vulkan arm
               through submission starvation (i.e. through the clock, §20.2).
* `gpuload`  — a second process running this same model on this same EP in a loop.  Predicted to
               hit the Vulkan arm hard and the CPU arm barely.

If the ratio is invariant across `paired`, `cpuload` and `gpuload` while the *levels* move, the
pairing assumption survived a test it could have failed.  If the ratio moves, the paired design
is not sound here and the honest output is that verdict, not a number.

**Disclosure about the injected load's tenancy.** The injected processes are our own descendants,
so `probe_gpustate`'s ancestry classifier calls them *ours* and the tenancy verdict stays
`SOLE_TENANT` while a second copy of Phi-3.5 is running on the board.  That is the classifier
working as designed (it exists so our own worker is not counted a stranger).  The witness for the
injection is therefore the launch itself plus the utilisation/clock series, **never** the tenancy
verdict, and the record says so in `injection.witness`.

# What this probe does NOT establish

Decode only, one model, one context window (`past` = 4..4+steps), two devices, the shipping
(host-KV) lane.  It says nothing about prefill, nothing about another model, nothing about long
context, and nothing about a quiet machine.  Every number it emits carries that frame in the
record, because a ratio is the single most re-quotable artifact this project could produce.

Provenance classes per §22 are emitted in the `PROVENANCE` block of the record.

Reproduce:
    $env:ONNXRUNTIME_VULKAN_EP_LIB="...\\rust\\target\\release\\onnxruntime_vulkan_ep.dll"
    python bench/results/probe_paired_ratio.py --device 0 --sweeps 10 --steps 6
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import random
import statistics
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
if str(REPO / "bench") not in sys.path:
    sys.path.insert(0, str(REPO / "bench"))

ONNX_FILE = pathlib.Path(
    r"C:\Users\justinchu\.foundry\cache\models\Microsoft"
    r"\Phi-3.5-mini-instruct-cuda-gpu\cuda-int4-rtn-block-32"
    r"\phi-3.5-mini-instruct-cuda-int4-rtn-block-32.onnx"
)
EP_NAME = "VulkanExecutionProvider"
COUNTERS_ENV = "ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE"

LAYERS = 32
KV_HEADS = 32
HEAD_DIM = 96
F16 = 2
VOCAB = 32064
BYTES_PER_PAST_TOKEN = LAYERS * 2 * KV_HEADS * HEAD_DIM * F16  # 393,216
SEED_PAST = 4

#: A ratio-of-ratios this far from 1.0 is treated as the pairing having moved. Judgement, not a
#: measurement — stated here beside the constant rather than buried in a comparison (§10.7).
PAIRING_TOLERANCE = 0.10
#: An injection that moves neither arm's level by at least this much was not felt, and the
#: invariance test it was supposed to drive is VACUOUS rather than passed.
INJECTION_MIN_LIFT = 0.15

PROVENANCE = {
    "BYTES_PER_PAST_TOKEN": "MODEL",          # ONNX declared shapes, arithmetic
    "step_ms": "MEASUREMENT",
    "ratio": "MEASUREMENT",
    "sm_mhz": "MEASUREMENT",
    "sm_max_mhz": "SPECIFICATION",            # the board's published boost ceiling
    "device_name": "MEASUREMENT",             # read off the run, never off the selector
    "coresidency_cost_x": "MEASUREMENT",
    "interleaving_cost_x": "MEASUREMENT",
    "variance_reduction_x": "MEASUREMENT",
    "pairs_for_5pct_ci": "MODEL",             # a normal-theory extrapolation from a measured sd
    "unpaired_runs_for_5pct_ci": "MODEL",
    "PAIRING_TOLERANCE": "MODEL",             # a chosen threshold, not an observation
    "INJECTION_MIN_LIFT": "MODEL",
}


# --------------------------------------------------------------------------------------- pure
def geomean(xs: "list[float]") -> float:
    return math.exp(statistics.fmean(math.log(x) for x in xs))


def pair_ratios(rows: "list[dict]") -> "list[dict]":
    """Match a Vulkan step to the CPU step of the *same sweep and step index*.

    Never pools across step index: per-step cost grows with `past_len`, so a pooled ratio would
    mix context lengths into one number and hide the axis it varies on.
    """
    by_key: dict = {}
    for r in rows:
        by_key.setdefault((r["phase"], r["sweep"], r["step"]), {})[r["arm"]] = r
    out = []
    for (phase, sweep, step), arms in sorted(by_key.items()):
        if "vk" in arms and "ref" in arms:
            out.append({
                "phase": phase, "sweep": sweep, "step": step,
                "past_len": arms["vk"]["past_len"],
                "vk_ms": arms["vk"]["ms"], "ref_ms": arms["ref"]["ms"],
                "ratio": arms["vk"]["ms"] / arms["ref"]["ms"],
            })
    return out


def summarise(pairs: "list[dict]") -> dict:
    """Central tendency and dispersion of a ratio sample, on the log scale it lives on."""
    if not pairs:
        return {"n": 0}
    rs = [p["ratio"] for p in pairs]
    logs = [math.log(r) for r in rs]
    sd = statistics.stdev(logs) if len(logs) > 1 else 0.0
    srt = sorted(rs)

    def q(f):
        return srt[min(len(srt) - 1, max(0, int(round(f * (len(srt) - 1)))))]

    return {
        "n": len(rs),
        "geomean": geomean(rs),
        "median": statistics.median(rs),
        "p10": q(0.10), "p90": q(0.90), "min": srt[0], "max": srt[-1],
        "log_sd": sd,
        # The multiplicative spread a single pair carries: exp(+-1 sd).
        "dispersion_x": math.exp(sd),
        "spread_x": srt[-1] / srt[0],
        # Standard error of the geometric mean, as a multiplicative factor.
        "geomean_ci95_x": math.exp(1.96 * sd / math.sqrt(len(rs))) if len(rs) > 1 else None,
        "vk_ms_median": statistics.median([p["vk_ms"] for p in pairs]),
        "ref_ms_median": statistics.median([p["ref_ms"] for p in pairs]),
    }


def by_step(pairs: "list[dict]") -> dict:
    out = {}
    for p in pairs:
        out.setdefault(p["step"], []).append(p)
    return {str(k): {"past_len": v[0]["past_len"], **summarise(v)} for k, v in sorted(out.items())}


def ratio_of_ratios(base: "list[dict]", other: "list[dict]", *, boot: int = 2000,
                    seed: int = 20260803) -> dict:
    """Does the paired ratio move between two phases? Bootstrap CI on the ratio of geomeans."""
    if not base or not other:
        return {"verdict": "UNOBSERVABLE", "why": "one phase produced no pairs"}
    rb = [p["ratio"] for p in base]
    ro = [p["ratio"] for p in other]
    point = geomean(ro) / geomean(rb)
    rng = random.Random(seed)
    draws = []
    for _ in range(boot):
        a = geomean([rb[rng.randrange(len(rb))] for _ in range(len(rb))])
        b = geomean([ro[rng.randrange(len(ro))] for _ in range(len(ro))])
        draws.append(b / a)
    draws.sort()
    lo = draws[int(0.025 * (boot - 1))]
    hi = draws[int(0.975 * (boot - 1))]
    return {
        "ratio_of_ratios": point,
        "ci95": [lo, hi],
        "contains_one": lo <= 1.0 <= hi,
        "within_tolerance": abs(point - 1.0) <= PAIRING_TOLERANCE,
    }


def level_lift(base: "list[dict]", other: "list[dict]") -> dict:
    """How far each arm's level moved between two phases, separately."""
    if not base or not other:
        return {}
    vk = geomean([p["vk_ms"] for p in other]) / geomean([p["vk_ms"] for p in base])
    cpu = geomean([p["ref_ms"] for p in other]) / geomean([p["ref_ms"] for p in base])
    return {"vk_lift_x": vk, "ref_lift_x": cpu, "asymmetry_x": vk / cpu}


def pairing_verdict(rr: dict, lift: dict) -> dict:
    """The verdict on the *method*, not on the EP.

    Three outcomes, and the third is not a pass:
      PAIRING_HOLDS   — the injection was felt by at least one arm and the ratio did not move.
      PAIRING_FAILS   — the ratio moved: the disturbance was not common-mode.
      VACUOUS         — the injection moved neither arm, so nothing was tested.
    """
    if "ratio_of_ratios" not in rr:
        return {"verdict": "UNOBSERVABLE", "why": rr.get("why", "no pairs")}
    felt = max(abs(lift.get("vk_lift_x", 1.0) - 1.0),
               abs(lift.get("ref_lift_x", 1.0) - 1.0)) >= INJECTION_MIN_LIFT
    if not felt:
        return {"verdict": "VACUOUS(injection_not_witnessed)",
                "why": "neither arm's level moved, so the invariance test had no disturbance "
                       "to be invariant to"}
    if rr["within_tolerance"] and rr["contains_one"]:
        return {"verdict": "PAIRING_HOLDS",
                "why": f"levels moved (vk {lift['vk_lift_x']:.3f}x, ref {lift['ref_lift_x']:.3f}x) "
                       f"and the ratio did not ({rr['ratio_of_ratios']:.4f}, "
                       f"CI {rr['ci95'][0]:.4f}..{rr['ci95'][1]:.4f})"}
    return {"verdict": "PAIRING_FAILS(not_common_mode)",
            "why": f"the ratio moved {rr['ratio_of_ratios']:.4f}x under a disturbance the arms "
                   f"did not share (vk {lift['vk_lift_x']:.3f}x vs ref {lift['ref_lift_x']:.3f}x)"}


def sample_size(pairs: "list[dict]") -> dict:
    """What the pairing buys, in the only currency that matters: how many pairs.

    §10.3 measured this machine at **2.65x** in single-threaded throughput, so an unpaired
    comparison of two runs taken minutes apart is worth very little. The question a paired design
    has to answer is not "is it better" but "how much of the arm-level dispersion does the ratio
    actually cancel", and that is `variance_reduction_x` below. It is computed from this run's own
    numbers rather than assumed from the design.
    """
    if len(pairs) < 2:
        return {"n": len(pairs)}
    lr = [math.log(p["ratio"]) for p in pairs]
    lv = [math.log(p["vk_ms"]) for p in pairs]
    lc = [math.log(p["ref_ms"]) for p in pairs]
    sd_r = statistics.stdev(lr)
    sd_v, sd_c = statistics.stdev(lv), statistics.stdev(lc)
    # Two arms measured independently (the unpaired design) add their variances.
    sd_u = math.sqrt(sd_v ** 2 + sd_c ** 2)

    def n_for(sd, pct=0.05):
        return math.ceil((1.96 * sd / math.log(1 + pct)) ** 2) if sd > 0 else 1

    return {
        "n": len(pairs),
        "log_sd_ratio_paired": sd_r,
        "log_sd_vk_arm": sd_v, "log_sd_ref_arm": sd_c,
        "log_sd_unpaired": sd_u,
        "variance_reduction_x": (sd_u / sd_r) if sd_r > 0 else None,
        "pairs_for_5pct_ci": n_for(sd_r),
        "unpaired_runs_for_5pct_ci": n_for(sd_u),
        "arm_spread_x": {"vk": math.exp(2 * sd_v), "ref": math.exp(2 * sd_c)},
    }


def clock_by_arm(samples: "list[dict]", rows: "list[dict]", phase: str) -> dict:
    """Board clock attributed to the intervals in which each arm was *actually executing*.

    A phase-wide median is dominated by whichever arm holds the wall clock longest, which on
    these two arms is a factor of five. The mechanism §20.2 names — host contention arriving on
    the device axis as an idle clock — is a statement about the board *while our dispatches are
    in flight*, so the window has to be the arm's own steps and not the phase.
    """
    out = {}
    for arm in ("vk", "ref"):
        spans = [(r["t0"], r["t"]) for r in rows
                 if r["phase"] == phase and r["arm"] == arm and "t0" in r]
        got = [s["sm_mhz"] for s in samples
               if s.get("sm_mhz") is not None and any(a <= s["t"] <= b for a, b in spans)]
        out[arm] = ({"n": len(got), "sm_mhz_median": statistics.median(got),
                     "sm_mhz_min": min(got), "sm_mhz_max": max(got)}
                    if got else {"n": 0, "verdict": "UNOBSERVABLE"})
    return out


def clock_window(samples: "list[dict]", t0: float, t1: float) -> dict:
    """Reduce the board-clock series over one phase's own wall window."""
    ins = [s for s in samples if t0 <= s["t"] <= t1 and s.get("sm_mhz") is not None]
    if not ins:
        return {"verdict": "UNOBSERVABLE", "n": 0}
    mhz = [s["sm_mhz"] for s in ins]
    util = [s["util_pct"] for s in ins if s.get("util_pct") is not None]
    mx = ins[0].get("sm_max_mhz")
    return {
        "n": len(ins), "sm_mhz_min": min(mhz), "sm_mhz_median": statistics.median(mhz),
        "sm_mhz_max": max(mhz), "sm_max_mhz": mx,
        "at_pct_of_boost": (100.0 * statistics.median(mhz) / mx) if mx else None,
        "util_pct_median": statistics.median(util) if util else None,
    }


# ------------------------------------------------------------------------------------ workers
def _lib() -> str:
    return os.environ.get(
        "ONNXRUNTIME_VULKAN_EP_LIB",
        str(REPO / "rust" / "target" / "release" / "onnxruntime_vulkan_ep.dll"),
    )


def _cpu_load_worker(stop_file: pathlib.Path) -> int:
    x = 0
    while not stop_file.exists():
        for i in range(2_000_000):
            x = (x * 1103515245 + 12345) & 0xFFFFFFFF
    return 0 if x else 0


def _gpu_load_worker(stop_file: pathlib.Path, ready_file: pathlib.Path, device: int) -> int:
    """A second, independent Phi-3.5 decode loop on the same board. Foreign work by intent."""
    import numpy as np
    import onnxruntime as ort

    os.environ["ONNXRUNTIME_EP_VULKAN_DEVICE"] = str(device)
    os.environ.pop(COUNTERS_ENV, None)
    try:
        ort.register_execution_provider_library(EP_NAME, _lib())
    except Exception as exc:  # noqa: BLE001
        if "already registered" not in str(exc):
            raise
    sess = ort.InferenceSession(
        str(ONNX_FILE), ort.SessionOptions(), providers=[EP_NAME, "CPUExecutionProvider"],
        free_dimension_overrides_by_name={"batch_size": "1", "sequence_length": "1"},
    )
    rng = np.random.default_rng(7)
    past = {}
    for layer in range(LAYERS):
        for kind in ("key", "value"):
            past[f"past_key_values.{layer}.{kind}"] = (
                rng.standard_normal((1, KV_HEADS, SEED_PAST, HEAD_DIM)).astype(np.float16) * 0.02)
    feeds = dict(past)
    feeds["input_ids"] = np.array([[1]], dtype=np.int64)
    feeds["attention_mask"] = np.ones((1, SEED_PAST + 1), dtype=np.int64)
    first = True
    while not stop_file.exists():
        sess.run(None, feeds)
        if first:
            ready_file.write_text("ready", encoding="utf-8")
            first = False
    return 0


# ---------------------------------------------------------------------------------- the run
def _counters(path: "pathlib.Path | None") -> dict:
    if path is None or not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return doc.get("counters", doc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--arm-b", choices=("cpu", "resident"), default="cpu",
                    help="the reference arm. `cpu` = ORT's CPU EP (a cross-device ratio); "
                         "`resident` = the same Vulkan session with the KV cache kept in device "
                         "memory (a same-device, same-EP ratio that prices the round trip)")
    ap.add_argument("--sweeps", type=int, default=10)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--cpu-load", type=int, default=max(2, (os.cpu_count() or 8) - 4))
    ap.add_argument("--out", default=None)
    ap.add_argument("--cpu-load-worker", default=None)
    ap.add_argument("--gpu-load-worker", default=None)
    ap.add_argument("--gpu-load-ready", default=None)
    args = ap.parse_args()

    if args.cpu_load_worker:
        return _cpu_load_worker(pathlib.Path(args.cpu_load_worker))
    if args.gpu_load_worker:
        return _gpu_load_worker(pathlib.Path(args.gpu_load_worker),
                                pathlib.Path(args.gpu_load_ready), args.device)

    import numpy as np
    import onnxruntime as ort

    import contention  # noqa: E402  (bench/ is on sys.path above)
    sys.path.insert(0, str(HERE))
    import probe_gpustate  # noqa: E402

    scratch = REPO / "bench" / "_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    counters_path = scratch / f"paired_ratio_counters_dev{args.device}.json"
    counters_path.unlink(missing_ok=True)
    os.environ[COUNTERS_ENV] = str(counters_path)
    os.environ["ONNXRUNTIME_EP_VULKAN_DEVICE"] = str(args.device)
    if args.arm_b == "resident":
        # Set before the library is registered: the factory reads these at registration, and a
        # DEFAULT allocator that does not exist by then never appears.
        os.environ["ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY"] = "1"
        os.environ["ONNXRUNTIME_EP_VULKAN_BIND_OUTPUTS"] = "1"

    doc: dict = {
        "probe": "probe_paired_ratio.py",
        "question": "is a paired interleaved A/B ratio sound on a permanently contended box, "
                    "and if so what is the decode ratio for Phi-3.5",
        "frame": {
            "model": ONNX_FILE.name,
            "lane": "arm A: shipping host-KV lane (sess.run, KV round trip paid every step). "
                    "Arm B is named in reference.arm_b and is NOT this lane.",
            "phase_of_inference": "decode only (one token per step); no prefill is measured",
            "context": f"past_len {SEED_PAST}..{SEED_PAST + args.steps - 1}",
            "ort_version": ort.__version__,
            "device_selector": args.device,
            "dll_sha256": None,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        },
        "PROVENANCE": PROVENANCE,
        "sweeps": args.sweeps, "steps": args.steps,
    }
    import hashlib
    doc["frame"]["dll_sha256"] = hashlib.sha256(
        pathlib.Path(_lib()).read_bytes()).hexdigest()[:16]

    try:
        ort.register_execution_provider_library(EP_NAME, _lib())
    except Exception as exc:  # noqa: BLE001
        if "already registered" not in str(exc):
            raise
    ep_device = next((d for d in ort.get_ep_devices() if d.ep_name == EP_NAME), None)
    if ep_device is None:
        doc["verdict"] = "ERROR(instrument)"
        doc["why"] = ["the Vulkan EP is not among ORT's EP devices"]
        print(json.dumps(doc, indent=2))
        return 2
    doc["frame"]["device_name"] = ep_device.ep_metadata.get("vulkan.device_name")
    doc["frame"]["device_index_reported"] = ep_device.ep_metadata.get("vulkan.device_index")
    doc["frame"]["vendor_id"] = ep_device.ep_metadata.get("vulkan.vendor_id")
    is_nvidia = "nvidia" in str(doc["frame"]["device_name"]).lower()

    # The reference. Saying what the ratio is a ratio *of* is part of the verdict, not a comment.
    doc["reference"] = {
        "arm_b": ("ORT CPUExecutionProvider, same process, same graph, same inputs"
                  if args.arm_b == "cpu" else
                  "the SAME Vulkan session with the 64 `present.*` outputs bound in this EP's "
                  "device memory and re-fed as the next step's `past` — same EP, same device, "
                  "same submission path; the only difference is whether the KV round trip is "
                  "paid"),
        "available_providers": list(ort.get_available_providers()),
        "second_gpu_ep": "NOT AVAILABLE — onnxruntime-directml publishes no wheel at this ORT "
                         "version (max 1.24.4 against this process's 1.28.0), and the Vulkan EP "
                         "is loaded through the 1.28 plugin-EP ABI, so DirectML and this EP "
                         "cannot be placed in one process on this machine today",
        "ratio_is_a_ratio_of": (
            "this EP on this GPU against ORT's CPU EP on this CPU — an end-to-end system ratio "
            "that confounds the EP with the device. It is NOT a comparison of two GPU EPs and "
            "may not be quoted as evidence about kernel quality."
            if args.arm_b == "cpu" else
            "the shipping host-KV lane against the device-resident KV lane, on one device, one "
            "session and one binary. It prices the KV round trip and nothing else; it is not a "
            "comparison against any other runtime and says nothing about how fast this EP is."),
    }

    logits_dtype = np.float16
    mem_info = None

    sessions = {}
    sessions["vk"] = ort.InferenceSession(
        str(ONNX_FILE), ort.SessionOptions(), providers=[EP_NAME, "CPUExecutionProvider"],
        free_dimension_overrides_by_name={"batch_size": "1", "sequence_length": "1"})
    if EP_NAME not in sessions["vk"].get_providers():
        doc["verdict"] = "ERROR(instrument)"
        doc["why"] = [f"{EP_NAME} absent from {sessions['vk'].get_providers()}"]
        print(json.dumps(doc, indent=2))
        return 2
    out_names = [o.name for o in sessions["vk"].get_outputs()]
    logits_dtype = (np.float16 if "float16" in sessions["vk"].get_outputs()[0].type
                    else np.float32)
    if args.arm_b == "resident":
        # Ordering is load-bearing: asking for the allocator before the session exists builds a
        # second VkDevice that no dispatch can reach.
        mem_info = ep_device.memory_info(ort.OrtDeviceMemoryType.DEFAULT)
        if mem_info is None:
            doc["verdict"] = "ERROR(instrument)"
            doc["why"] = ["the EP registered no DEFAULT allocator, so there is no device memory "
                          "to keep a KV cache in and the question was never put"]
            print(json.dumps(doc, indent=2))
            return 2

    rng = np.random.default_rng(20260803)
    seed_kv = [rng.standard_normal((1, KV_HEADS, SEED_PAST, HEAD_DIM)).astype(np.float16) * 0.02
               for _ in range(LAYERS * 2)]
    seed_past = {}
    for layer in range(LAYERS):
        seed_past[f"past_key_values.{layer}.key"] = seed_kv[2 * layer]
        seed_past[f"past_key_values.{layer}.value"] = seed_kv[2 * layer + 1]

    rows: "list[dict]" = []
    agreement: "list[dict]" = []

    def _step_host(sess, past, past_len):
        """One decode step through the shipping path: numpy in, numpy out, KV round trip paid."""
        feeds = {"input_ids": np.array([[1]], dtype=np.int64),
                 "attention_mask": np.ones((1, past_len + 1), dtype=np.int64)}
        feeds.update(past)
        t0 = time.perf_counter()
        outs = sess.run(None, feeds)
        t1 = time.perf_counter()
        got = dict(zip(out_names, outs))
        nxt = {f"past_key_values.{i}.{k}": np.asarray(got[f"present.{i}.{k}"])
               for i in range(LAYERS) for k in ("key", "value")}
        return t0, t1, nxt, np.asarray(got["logits"], dtype=np.float32).reshape(-1)

    def _step_resident(sess, past_dev, past_len):
        """One decode step with the 64 `present.*` outputs bound in this EP's device memory.

        Same session, same device, same submission path as the arm it is paired against — the
        *only* thing that differs is whether the KV round trip is paid. That is what makes this
        reference a same-axis one.
        """
        binding = sess.io_binding()
        binding.bind_cpu_input("input_ids", np.array([[1]], dtype=np.int64))
        binding.bind_cpu_input("attention_mask", np.ones((1, past_len + 1), dtype=np.int64))
        for name, ov in past_dev.items():
            binding.bind_ortvalue_input(name, ov)
        present = {}
        for layer in range(LAYERS):
            for kind in ("key", "value"):
                n = f"present.{layer}.{kind}"
                ov = ort.OrtValue.ortvalue_from_shape_and_type(
                    [1, KV_HEADS, past_len + 1, HEAD_DIM], np.float16, memory_info=mem_info)
                binding.bind_ortvalue_output(n, ov)
                present[n] = ov
        logits_ov = ort.OrtValue.ortvalue_from_shape_and_type([1, 1, VOCAB], logits_dtype)
        binding.bind_ortvalue_output("logits", logits_ov)
        t0 = time.perf_counter()
        sess.run_with_iobinding(binding)
        t1 = time.perf_counter()
        nxt = {f"past_key_values.{layer}.{kind}": present[f"present.{layer}.{kind}"]
               for layer in range(LAYERS) for kind in ("key", "value")}
        return t0, t1, nxt, np.asarray(logits_ov.numpy(), dtype=np.float32).reshape(-1)

    def _fresh_past(arm: str):
        if arm == "ref" and args.arm_b == "resident":
            out = {}
            for name, arr in seed_past.items():
                ov = ort.OrtValue.ortvalue_from_shape_and_type(list(arr.shape), np.float16,
                                                               memory_info=mem_info)
                ov.update_inplace(arr)
                out[name] = ov
            return out
        return {k: v.copy() for k, v in seed_past.items()}

    def run_chain(arm: str, phase: str, sweep: int, other: "str | None" = None,
                  check: bool = False):
        """One decode chain of `--steps` steps, from the same seed past, on one arm.

        When `other` is given the two chains are advanced *alternately*, one step each, which is
        the finest granularity a decode chain admits: a decode step is atomic, and there is no
        smaller unit to alternate on.
        """
        arms = [arm] if other is None else [arm, other]
        past = {a: _fresh_past(a) for a in arms}
        for step in range(args.steps):
            past_len = SEED_PAST + step
            for a in arms:
                if a == "ref" and args.arm_b == "resident":
                    t0, t1, nxt, lg = _step_resident(sessions["vk"], past[a], past_len)
                else:
                    t0, t1, nxt, lg = _step_host(sessions[a], past[a], past_len)
                past[a] = nxt
                rows.append({"phase": phase, "sweep": sweep, "step": step, "arm": a,
                             "past_len": past_len, "ms": (t1 - t0) * 1e3, "t0": t0, "t": t1})
                if check:
                    agreement.append({"arm": a, "step": step, "argmax": int(lg.argmax()),
                                      "sig": [float(lg[0]), float(lg[len(lg) // 2]),
                                              float(lg[-1])]})

    # Warmup: the first Vulkan step carries the whole weight upload (~1.7 s measured) and is not
    # a decode step. Discarded, never averaged in.
    run_chain("vk", "warmup", 0)
    warm = [r for r in rows if r["phase"] == "warmup"]
    doc["warmup_first_vk_step_ms"] = next(r["ms"] for r in warm if r["arm"] == "vk")
    rows.clear()

    sampler = None
    if is_nvidia:
        # nvidia-smi has its own board ordering and it is not the EP's selector; on this box the
        # only NVIDIA board is smi index 0. The device the figure is about is named off the run
        # above, so this index is only the producer's handle on it.
        try:
            probe_gpustate._sample_once(0)
            sampler = probe_gpustate.Sampler(0)
            sampler.own_root = os.getpid()
            sampler.start()
        except Exception as exc:  # noqa: BLE001
            doc["clock_producer"] = {"verdict": "ERROR(instrument)", "why": str(exc)}
    if sampler is None and not is_nvidia:
        doc["clock_producer"] = {
            "verdict": "NO_PRODUCER",
            "why": "PERF.md §16.3: the Intel clock axis is `none_available` on this machine, "
                   "permanently. No MHz-carrying counter set, no WMI class, no nvidia-smi "
                   "equivalent, and engine Running Time is a duration (same-source).",
        }

    monitor = contention.Monitor().start()
    phases: "list[tuple[str, float, float]]" = []
    tach: dict = {}

    def phase(name: str, body):
        tach[name + ":before"] = contention.occupancy_probe(reps=3)
        t0 = time.perf_counter()
        body()
        t1 = time.perf_counter()
        tach[name + ":after"] = contention.occupancy_probe(reps=3)
        phases.append((name, t0, t1))

    # --- phase 0: the Vulkan arm ALONE in the process, before the reference session exists.
    # A paired design requires both arms co-resident. That is apparatus, and apparatus has a
    # cost; this phase is what makes it a measurement rather than an assumption.
    def _solo_vk():
        for s in range(args.sweeps):
            run_chain("vk", "solo_vk", s)
    phase("solo_vk", _solo_vk)

    if args.arm_b == "cpu":
        sessions["ref"] = ort.InferenceSession(
            str(ONNX_FILE), ort.SessionOptions(), providers=["CPUExecutionProvider"],
            free_dimension_overrides_by_name={"batch_size": "1", "sequence_length": "1"})
    # The reference arm's own warmup is not a measured step either.
    run_chain("ref", "warmup_ref", 0)
    rows[:] = [r for r in rows if r["phase"] != "warmup_ref"]

    # --- phase 1: the paired design itself, no injection.
    def _paired():
        for s in range(args.sweeps):
            run_chain("vk", "paired", s, other="ref", check=(s == 0))
    phase("paired", _paired)

    # --- phase 2: the same, un-interleaved. The control for what pairing costs each arm.
    def _blocked():
        for s in range(args.sweeps):
            run_chain("vk", "blocked", s)
        for s in range(args.sweeps):
            run_chain("ref", "blocked", s)
    phase("blocked", _blocked)

    stop_file = scratch / "paired_ratio_stop"
    ready_file = scratch / "paired_ratio_ready"
    stop_file.unlink(missing_ok=True)
    ready_file.unlink(missing_ok=True)

    # --- phase 3: CPU injection.
    kids = [subprocess.Popen([sys.executable, str(pathlib.Path(__file__).resolve()),
                              "--cpu-load-worker", str(stop_file)])
            for _ in range(args.cpu_load)]
    time.sleep(3.0)

    def _cpuload():
        for s in range(args.sweeps):
            run_chain("vk", "cpuload", s, other="ref")
    phase("cpuload", _cpuload)
    stop_file.write_text("stop", encoding="utf-8")
    for k in kids:
        k.wait(timeout=60)
    stop_file.unlink(missing_ok=True)
    time.sleep(3.0)

    # --- phase 4: GPU injection — a second Phi-3.5 decode loop on the same board.
    gpu_kid = subprocess.Popen([sys.executable, str(pathlib.Path(__file__).resolve()),
                                "--gpu-load-worker", str(stop_file),
                                "--gpu-load-ready", str(ready_file),
                                "--device", str(args.device)])
    waited = 0.0
    while not ready_file.exists() and waited < 180.0 and gpu_kid.poll() is None:
        time.sleep(1.0)
        waited += 1.0
    gpu_ready = ready_file.exists()

    def _gpuload():
        for s in range(args.sweeps):
            run_chain("vk", "gpuload", s, other="ref")
    if gpu_ready:
        phase("gpuload", _gpuload)
    stop_file.write_text("stop", encoding="utf-8")
    try:
        gpu_kid.wait(timeout=120)
    except subprocess.TimeoutExpired:
        gpu_kid.kill()
    stop_file.unlink(missing_ok=True)
    ready_file.unlink(missing_ok=True)

    # --- phase 5: the reference arm alone, outside the apparatus. The mirror of phase 0, so
    # neither arm's co-residency cost has to be taken on faith. With the CPU reference the
    # Vulkan session is destroyed first; with the same-session resident reference there is
    # nothing to destroy and the phase measures the same session with the other arm idle.
    # Order effects are real, this phase runs last, and that is stated rather than corrected for.
    if args.arm_b == "cpu":
        del sessions["vk"]
        import gc
        gc.collect()

    def _solo_ref():
        for s in range(args.sweeps):
            run_chain("ref", "solo_ref", s)
    phase("solo_ref", _solo_ref)

    window = monitor.stop()
    samples = []
    if sampler is not None:
        sampler.stop.set()
        sampler.join(timeout=20)
        samples = sampler.samples
        if sampler.error:
            doc["clock_producer"] = {"verdict": "ERROR(instrument)", "why": sampler.error}

    # ------------------------------------------------------------------------------ analysis
    counters = _counters(counters_path)
    doc["liveness"] = {
        "dispatches_executed": counters.get("dispatches_executed"),
        "compute_calls": counters.get("compute_calls"),
        "compute_failures": counters.get("compute_failures"),
        "device_losses": counters.get("device_losses"),
        "subgraphs_live": counters.get("subgraphs_live"),
        "claimed_nodes": counters.get("claimed_nodes"),
        "nodes_offered": counters.get("nodes_offered"),
        "islands_offered": counters.get("islands_offered"),
    }
    doc["machine"] = window.to_dict() if hasattr(window, "to_dict") else str(window)
    doc["tachometer_s"] = tach
    doc["injection"] = {
        "cpu_load_processes": args.cpu_load,
        "gpu_load_started": gpu_ready,
        "witness": "the launch of the injecting processes plus the utilisation/clock series. "
                   "NOT the tenancy verdict: the injected processes are our own descendants, so "
                   "probe_gpustate's ancestry classifier calls them ours and the board still "
                   "reads SOLE_TENANT while a second copy of Phi-3.5 runs on it.",
    }

    all_pairs = pair_ratios(rows)
    per_phase = {}
    for name, t0, t1 in phases:
        ps = [p for p in all_pairs if p["phase"] == name]
        per_phase[name] = {
            "wall_s": t1 - t0,
            "pairs": summarise(ps),
            "sample_size": sample_size(ps),
            "by_step": by_step(ps),
            "clock": clock_window(samples, t0, t1) if samples else
                     {"verdict": "NO_PRODUCER" if not is_nvidia else "UNOBSERVABLE"},
            "clock_by_arm": clock_by_arm(samples, rows, name) if samples else
                     {"verdict": "NO_PRODUCER" if not is_nvidia else "UNOBSERVABLE"},
        }
    # `blocked`, `solo_vk` and `solo_ref` produce no pairs by construction — they are the
    # un-interleaved and un-co-resident controls. Each arm's level is summarised across all four
    # so the apparatus's own cost is visible per arm.
    levels = {}
    for a, solo in (("vk", "solo_vk"), ("ref", "solo_ref")):
        def g(ph):
            xs = [r["ms"] for r in rows if r["phase"] == ph and r["arm"] == a]
            return geomean(xs) if xs else None
        solo_ms, blocked_ms, paired_ms = g(solo), g("blocked"), g("paired")
        levels[a] = {
            "solo_ms_geomean": solo_ms,
            "blocked_ms_geomean": blocked_ms,
            "paired_ms_geomean": paired_ms,
            # cost of having the other arm's session merely resident in the process
            "coresidency_cost_x": (blocked_ms / solo_ms) if solo_ms and blocked_ms else None,
            # further cost of alternating with it, on top of co-residency
            "interleaving_cost_x": (paired_ms / blocked_ms) if blocked_ms and paired_ms else None,
            "apparatus_cost_x": (paired_ms / solo_ms) if solo_ms and paired_ms else None,
        }
    doc["apparatus_cost"] = {
        "arms": levels,
        "reading": "what each arm's own step time does when the pairing apparatus is applied to "
                   "it, with no foreign load added at all: first co-residency (the other arm's "
                   "session alive in the process), then interleaving (alternating with it). A "
                   "pairing that perturbs one arm and not the other has manufactured the "
                   "asymmetry it was adopted to remove, and the ratio it publishes is a ratio "
                   "of two perturbed arms rather than of two arms.",
    }
    ap_vk = levels.get("vk", {}).get("apparatus_cost_x")
    ap_cpu = levels.get("ref", {}).get("apparatus_cost_x")
    if ap_vk and ap_cpu:
        asym = ap_vk / ap_cpu
        doc["apparatus_cost"]["asymmetry_x"] = asym
        doc["apparatus_cost"]["verdict"] = (
            "APPARATUS_SYMMETRIC" if abs(asym - 1.0) <= PAIRING_TOLERANCE
            else "APPARATUS_ASYMMETRIC")
        doc["apparatus_cost"]["why"] = (
            f"applying the apparatus multiplies the Vulkan arm by {ap_vk:.3f}x and the reference "
            f"arm by {ap_cpu:.3f}x; a ratio published from the paired phase therefore carries a "
            f"factor of {asym:.3f}x that belongs to the instrument, not to the EP")

    base = [p for p in all_pairs if p["phase"] == "paired"]
    tests = {}
    for name in ("cpuload", "gpuload"):
        other = [p for p in all_pairs if p["phase"] == name]
        if not other:
            tests[name] = {"verdict": "UNOBSERVABLE", "why": f"phase {name} produced no pairs"}
            continue
        rr = ratio_of_ratios(base, other)
        lift = level_lift(base, other)
        tests[name] = {**rr, **lift, **pairing_verdict(rr, lift)}
    doc["pairing_tests"] = tests
    doc["phases"] = per_phase

    # Correctness gate: an arm that computes something else is not a comparison.
    vk_a = {(a["step"]): a for a in agreement if a["arm"] == "vk"}
    cpu_a = {(a["step"]): a for a in agreement if a["arm"] == "ref"}
    same = [s for s in vk_a if s in cpu_a and vk_a[s]["argmax"] == cpu_a[s]["argmax"]]
    doc["output_agreement"] = {
        "steps_compared": len(cpu_a),
        "argmax_agree": len(same),
        "verdict": "AGREE" if cpu_a and len(same) == len(cpu_a) else "DIVERGENT",
        "note": "argmax of logits per decode step, both arms, same seed past. A full per-output "
                "AGREE is Trinity's instrument; this is the liveness form of it.",
    }

    verdicts = [t.get("verdict", "UNOBSERVABLE") for t in tests.values()]
    live = (doc["liveness"]["dispatches_executed"] or 0) > 0 and \
        not doc["liveness"]["compute_failures"] and not doc["liveness"]["device_losses"]
    if not live or doc["output_agreement"]["verdict"] != "AGREE":
        doc["verdict"] = "ERROR(instrument)"
    elif doc["apparatus_cost"].get("verdict") == "APPARATUS_ASYMMETRIC":
        doc["verdict"] = "PAIRING_FAILS(apparatus_asymmetry)"
    elif all(v == "PAIRING_HOLDS" for v in verdicts) and verdicts:
        doc["verdict"] = "PAIRING_HOLDS"
    elif any(v.startswith("PAIRING_FAILS") for v in verdicts):
        doc["verdict"] = "PAIRING_FAILS"
    else:
        doc["verdict"] = "INCONCLUSIVE"

    doc["not_established"] = [
        "prefill — every step here is a single-token decode; nothing was measured about the "
        "prompt phase, which has a completely different arithmetic intensity",
        "any other model — Phi-3.5 mini int4 only, and its Nq/Nkv = 1.00 is the degenerate case "
        "of the grouped-attention axis",
        "long context — past_len 4..%d only; the shipping lane OOMs at past 4096 on the 8 GB "
        "discrete card, so the ratio at long context is not 'worse', it is 'does not run'"
        % (SEED_PAST + args.steps - 1),
        "a quiet machine — this is a ratio measured under contention, and it is not a prediction "
        "of what either arm would do alone",
        ("kernel quality — arm B is a CPU EP on a CPU, so the ratio confounds the EP with the "
         "device it runs on" if args.arm_b == "cpu" else
         "any comparison against another runtime — arm B is this same EP with the KV cache kept "
         "on the device, so this ratio prices one lever and nothing else"),
    ]

    out = pathlib.Path(args.out) if args.out else (
        HERE / f"paired_ratio_dev{args.device}.json")
    doc["rows"] = rows
    out.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    slim = {k: v for k, v in doc.items() if k != "rows"}
    print(json.dumps(slim, indent=2, default=str))
    print(f"\nwrote {out}")
    return 0 if doc["verdict"] in ("PAIRING_HOLDS", "PAIRING_FAILS", "INCONCLUSIVE") else 2


if __name__ == "__main__":
    raise SystemExit(main())
