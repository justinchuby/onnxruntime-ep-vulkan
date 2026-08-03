"""If the output-side device bind becomes the default, what happens to the caller who binds nothing?

Every lane this project has built for the bound-output path binds **deliberately**:
`probe_kv_chain_phi35.py` allocates 64 device `OrtValue`s and hands them to `io_binding`;
`probe_bound_output_correctness.py` does the same on the GQA case; Trinity's criterion-10 route
axis forces the route on with `BIND_OUTPUTS=1`. All of them are callers who know the path exists.

The caller a *default flip* exposes is the one who does not:

    sess = ort.InferenceSession(model, providers=["VulkanExecutionProvider", ...])
    outs = sess.run(None, feeds)          # no io_binding, no OrtValue, no memory_info
    logits = outs[0]                      # numpy, on the host, like always

That caller never binds an output, so nothing in the probe suite has ever run him against the
bound path. ORT still allocates his fused-node outputs through this EP's provider when the device
allocator is armed (measured: 195 of 195 device-resident, `probe_output_residency.py`), Step 1c
still binds them, and the bytes he eventually reads come back through ORT's own device→host copy
— `CopyTensors` — rather than through the session's staging readback. That is a *different door*
for the same bytes, and a door no lane has walked through.

# The question, and only this question

    with the output-side bind ON BY DEFAULT, does a caller who binds nothing get the same bytes
    he gets today — on every one of the 65 outputs, on prefill and on decode?

# The four lanes

    ship_off       BIND_OUTPUTS=0, device allocator OFF   — what shipped before 2026-08-03
    default        nothing set                            — the flip, for the default user
    armed_off      DEVICE_MEMORY=1, BIND_OUTPUTS=0        — the escape hatch, armed
    armed_default  DEVICE_MEMORY=1, nothing else          — the flip, armed. THE SUBJECT.

`ship_off` and `default` differ only in whether Step 1c runs. It cannot bind anything with the
allocator off (`bind_target_for` declines every pointer that is not one of our handles), so any
difference between those two lanes is the flip's *cost to a user who gets no benefit from it* —
the case that most deserves to be zero and has the least reason to be measured.

# What is compared

All 65 outputs, byte for byte (sha256 of the raw buffer), never an aggregate and never just
`logits`. `logits` is output 0 of 65; the other 64 are the KV cache, and a defect confined to
them leaves the logits bit-identical for one step and compounds thereafter. That is the exact
blindness Trinity reopened criterion 10 over, and it is not going to be re-introduced here.

Two shapes of caller, because they take different branches in Step 1c:
  * **prefill** — `past = 0`. The 64 KV *inputs* are zero-element (`[1,32,0,96]`), so Step 1c's
    `sz == 0` skip fires on nothing (the *outputs* are size 1) but the input side is degenerate.
  * **decode** — `past = 4`, then a second step fed from the numpy the first step returned. If a
    bound output came back stale, the second step consumes the staleness and the logits move.

# Guards, before any number here is read

  * **degeneracy** — an all-zero tensor agrees with another all-zero tensor perfectly. Every
    lane reports the nonzero fraction of every output and the count of distinct values in
    `logits`; a lane that returns nothing scores 1.0 on any comparison and is refused here.
  * **liveness** — `dispatches_executed > 0`, `compute_failures == 0`, `device_losses == 0`.
    A slope of zero because the run stopped is not a result.
  * **route read off the run** — `outputs_bind_attempted`, `outputs_device_bound`,
    `outputs_bind_declined`, `alloc_device_frame`, never off the env var that requested it.
    Step 1c unbinds on refusal, so a declined bind would otherwise record as a route taken.
  * **the EP actually ran** — `VulkanExecutionProvider` present in `sess.get_providers()`, and
    the device name read off the run, never off the selector.

No wall-clock. The box is permanently contended (`PERF.md` §20). Counts and bytes only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent

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
VOCAB = 32064
SEED_PAST = 4

LANES = {
    # lane            (DEVICE_MEMORY, BIND_OUTPUTS)   None = leave unset
    "ship_off": (None, "0"),
    "default": (None, None),
    "armed_off": ("1", "0"),
    "armed_default": ("1", None),
}

COUNTER_KEYS = (
    "outputs_bind_attempted",
    "outputs_device_bound",
    "outputs_bind_declined",
    "outputs_device_resident",
    "outputs_host_resident",
    "alloc_device_frame",
    "alloc_device_authority_grants",
    "alloc_device_downloads",
    "alloc_device_download_bytes",
    "alloc_device_uploads",
    "alloc_device_upload_bytes",
    "session_staging_readback_bytes",
    "session_staging_readbacks",
    "session_staging_upload_bytes",
    "dispatches_executed",
    "compute_calls",
    "compute_failures",
    "device_losses",
)


def _lib() -> str:
    return os.environ.get(
        "ONNXRUNTIME_VULKAN_EP_LIB",
        str(REPO / "rust" / "target" / "release" / "onnxruntime_vulkan_ep.dll"),
    )


def _dll_hash() -> str:
    p = pathlib.Path(_lib())
    if not p.is_file():
        return "<absent>"
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16].upper()


def _counters(path: pathlib.Path) -> dict:
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return doc.get("counters", doc)


# --------------------------------------------------------------------------- worker


def _worker(lane: str, out_path: pathlib.Path) -> int:
    import numpy as np
    import onnxruntime as ort

    rng = np.random.default_rng(20260803)
    doc: dict = {"lane": lane, "ort_version": ort.__version__, "dll_sha256_16": _dll_hash()}
    doc["env_as_set"] = {
        k: os.environ.get(k, "<unset>")
        for k in ("ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY", "ONNXRUNTIME_EP_VULKAN_BIND_OUTPUTS")
    }

    counters_path = pathlib.Path(os.environ[COUNTERS_ENV]) if COUNTERS_ENV in os.environ else None
    if counters_path is not None:
        counters_path.unlink(missing_ok=True)

    try:
        ort.register_execution_provider_library(EP_NAME, _lib())
    except Exception as exc:  # noqa: BLE001
        if "already registered" not in str(exc):
            raise
    ep_device = next((d for d in ort.get_ep_devices() if d.ep_name == EP_NAME), None)
    if ep_device is None:
        doc["verdict"] = "ERROR(instrument)"
        doc["why"] = ["the Vulkan EP is not among ORT's EP devices"]
        out_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        return 2
    # The device is read off the run. `DEVICE=0` has run on `1=NVIDIA` on this box.
    doc["ep_device"] = {
        k: ep_device.ep_metadata.get(k)
        for k in ("vulkan.device_name", "vulkan.device_index", "vulkan.vendor_id")
    }

    sess = ort.InferenceSession(
        str(ONNX_FILE),
        ort.SessionOptions(),
        providers=[EP_NAME, "CPUExecutionProvider"],
        free_dimension_overrides_by_name={"batch_size": "1", "sequence_length": "1"},
    )
    if EP_NAME not in sess.get_providers():
        doc["verdict"] = "ERROR(instrument)"
        doc["why"] = [f"{EP_NAME} absent from {sess.get_providers()}"]
        out_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        return 2

    out_names = [o.name for o in sess.get_outputs()]
    doc["output_names"] = out_names
    doc["output_names_order"] = (
        "session order, from sess.get_outputs(). NOT sorted and NOT binding order — a sorted "
        "container has discarded the only property it was being read for (R12)."
    )
    if out_names[0] != "logits" or len(out_names) != 1 + 2 * LAYERS:
        doc["verdict"] = "ERROR(instrument)"
        doc["why"] = [f"unexpected output set: {len(out_names)} names, first is {out_names[0]!r}"]
        out_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        return 2

    def _kv(past_len: int) -> dict:
        return {
            f"past_key_values.{layer}.{kind}": (
                rng.standard_normal((1, KV_HEADS, past_len, HEAD_DIM)).astype(np.float16) * 0.02
                if past_len
                else np.zeros((1, KV_HEADS, 0, HEAD_DIM), dtype=np.float16)
            )
            for layer in range(LAYERS)
            for kind in ("key", "value")
        }

    def _sig(a) -> dict:
        a = np.asarray(a)
        return {
            "shape": list(a.shape),
            "dtype": str(a.dtype),
            "sha256": hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16],
            "nonzero_fraction": float(np.count_nonzero(a) / max(a.size, 1)),
        }

    def _run(feeds: dict) -> tuple[dict, dict]:
        prev = _counters(counters_path) if counters_path else {}
        outs = sess.run(None, feeds)           # <- binds nothing. This is the whole point.
        got = dict(zip(out_names, outs))
        now = _counters(counters_path) if counters_path else {}
        delta = {
            k: (int(now.get(k) or 0) - int(prev.get(k) or 0))
            for k in COUNTER_KEYS
            if isinstance(now.get(k), (int, float)) or now.get(k) is None
        }
        return got, delta

    shapes: dict[str, dict] = {}

    # ── prefill: past = 0, the zero-extent KV inputs ─────────────────────────────────────────
    feeds = {
        "input_ids": np.array([[1]], dtype=np.int64),
        "attention_mask": np.ones((1, 1), dtype=np.int64),
    }
    feeds.update(_kv(0))
    got, delta = _run(feeds)
    shapes["prefill"] = {
        "past_len": 0,
        "outputs": {n: _sig(got[n]) for n in out_names},
        "counter_delta": delta,
        "argmax": int(np.argmax(np.asarray(got["logits"], dtype=np.float64).reshape(-1)[-VOCAB:])),
        "logits_distinct_values": int(
            np.unique(np.asarray(got["logits"]).reshape(-1)).size
        ),
    }

    # ── decode step 1: past = 4 ──────────────────────────────────────────────────────────────
    past = _kv(SEED_PAST)
    doc["seed_kv_sha256"] = hashlib.sha256(
        b"".join(np.ascontiguousarray(past[f"past_key_values.{l}.{k}"]).tobytes()
                 for l in range(LAYERS) for k in ("key", "value"))
    ).hexdigest()[:16]
    feeds = {
        "input_ids": np.array([[1]], dtype=np.int64),
        "attention_mask": np.ones((1, SEED_PAST + 1), dtype=np.int64),
    }
    feeds.update(past)
    got, delta = _run(feeds)
    tok = int(np.argmax(np.asarray(got["logits"], dtype=np.float64).reshape(-1)[-VOCAB:]))
    shapes["decode1"] = {
        "past_len": SEED_PAST,
        "outputs": {n: _sig(got[n]) for n in out_names},
        "counter_delta": delta,
        "argmax": tok,
        "logits_distinct_values": int(np.unique(np.asarray(got["logits"]).reshape(-1)).size),
    }

    # ── decode step 2: fed from step 1's returned numpy ──────────────────────────────────────
    # This is what catches a stale bound output that step 1's own comparison would not: step 1's
    # `present.*` is step 2's `past_key_values.*`, so a wrong-but-plausible KV tensor changes the
    # logits here even though it changed nothing there.
    feeds = {
        "input_ids": np.array([[tok]], dtype=np.int64),
        "attention_mask": np.ones((1, SEED_PAST + 2), dtype=np.int64),
    }
    feeds.update(
        {
            f"past_key_values.{layer}.{kind}": np.asarray(got[f"present.{layer}.{kind}"])
            for layer in range(LAYERS)
            for kind in ("key", "value")
        }
    )
    got, delta = _run(feeds)
    shapes["decode2"] = {
        "past_len": SEED_PAST + 1,
        "outputs": {n: _sig(got[n]) for n in out_names},
        "counter_delta": delta,
        "argmax": int(np.argmax(np.asarray(got["logits"], dtype=np.float64).reshape(-1)[-VOCAB:])),
        "logits_distinct_values": int(np.unique(np.asarray(got["logits"]).reshape(-1)).size),
    }

    doc["phases"] = shapes
    if counters_path:
        del sess  # the complete document lands at teardown
        final = _counters(counters_path)
        doc["final_counters"] = {k: final.get(k) for k in COUNTER_KEYS}
    out_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return 0


# --------------------------------------------------------------------------- driver


def _run_lane(lane: str, scratch: pathlib.Path) -> dict:
    out = scratch / f"default_bind_{lane}.json"
    counters = scratch / f"default_bind_{lane}.counters.json"
    out.unlink(missing_ok=True)
    counters.unlink(missing_ok=True)
    env = dict(os.environ)
    env[COUNTERS_ENV] = str(counters)
    env["ONNXRUNTIME_VULKAN_EP_LIB"] = _lib()
    dm, bo = LANES[lane]
    for name, val in (
        ("ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY", dm),
        ("ONNXRUNTIME_EP_VULKAN_BIND_OUTPUTS", bo),
    ):
        if val is None:
            env.pop(name, None)
        else:
            env[name] = val
    proc = subprocess.run(
        [sys.executable, str(pathlib.Path(__file__).resolve()),
         "--worker", "--lane", lane, "--out", str(out)],
        env=env,
        capture_output=True,
    )
    doc = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else {}
    doc["exit_code"] = proc.returncode
    if proc.returncode != 0:
        doc.setdefault("verdict", "ERROR(instrument)")
        doc["stderr_tail"] = (proc.stderr or b"").decode("utf-8", "replace")[-3000:]
    return doc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--lane")
    ap.add_argument("--out")
    ap.add_argument("--lanes", default=",".join(LANES))
    ap.add_argument("--device", default=os.environ.get("ONNXRUNTIME_EP_VULKAN_DEVICE", "0"))
    args = ap.parse_args()

    if args.worker:
        return _worker(args.lane, pathlib.Path(args.out))

    os.environ["ONNXRUNTIME_EP_VULKAN_DEVICE"] = str(args.device)
    scratch = HERE / "_default_bind_scratch"
    scratch.mkdir(exist_ok=True)
    lanes = [l for l in args.lanes.split(",") if l]

    report: dict = {
        "probe": "default_bind_outputs",
        "question": (
            "with the output-side device bind ON BY DEFAULT, does a caller who binds nothing "
            "get the same bytes on all 65 outputs, on prefill and on decode?"
        ),
        "dll_sha256_16": _dll_hash(),
        "device_selector": args.device,
        "lanes": {},
    }
    for lane in lanes:
        report["lanes"][lane] = _run_lane(lane, scratch)

    errs = [l for l in lanes if report["lanes"][l].get("verdict", "").startswith("ERROR")]
    if errs:
        report["verdict"] = "ERROR(instrument)"
        report["why"] = [f"{l}: {report['lanes'][l].get('why')}" for l in errs]
        _emit(report)
        return 2

    # Device name off the run, not the selector, and it must agree across lanes or the lanes are
    # not comparable in the first place.
    names = {l: (report["lanes"][l].get("ep_device") or {}).get("vulkan.device_name")
             for l in lanes}
    report["device_name_off_the_run"] = names
    if len(set(names.values())) != 1:
        report["verdict"] = "ERROR(instrument)"
        report["why"] = [f"lanes ran on different devices: {names}"]
        _emit(report)
        return 2

    # ── guards ───────────────────────────────────────────────────────────────────────────────
    why: list[str] = []
    for lane in lanes:
        fc = report["lanes"][lane].get("final_counters") or {}
        if not (fc.get("dispatches_executed") or 0) > 0:
            why.append(f"{lane}: dispatches_executed = {fc.get('dispatches_executed')}")
        if (fc.get("compute_failures") or 0) or (fc.get("device_losses") or 0):
            why.append(
                f"{lane}: compute_failures={fc.get('compute_failures')} "
                f"device_losses={fc.get('device_losses')}"
            )
        for phase, pd in (report["lanes"][lane].get("phases") or {}).items():
            dead = [n for n, s in pd["outputs"].items()
                    if s["nonzero_fraction"] < 0.5 and s["shape"] and 0 not in s["shape"]]
            if dead:
                why.append(
                    f"{lane}/{phase}: {len(dead)} of {len(pd['outputs'])} outputs are mostly "
                    f"zero ({dead[:3]}...) — a degenerate tensor agrees with anything"
                )
            if pd["logits_distinct_values"] < 1000:
                why.append(
                    f"{lane}/{phase}: logits has only {pd['logits_distinct_values']} distinct "
                    "values"
                )
    if why:
        report["verdict"] = "ERROR(instrument)"
        report["why"] = why
        _emit(report)
        return 2

    # ── the route each lane actually took, read off its counters ─────────────────────────────
    routes = {}
    for lane in lanes:
        fc = report["lanes"][lane].get("final_counters") or {}
        att = int(fc.get("outputs_bind_attempted") or 0)
        bnd = int(fc.get("outputs_device_bound") or 0)
        routes[lane] = {
            "outputs_bind_attempted": att,
            "outputs_device_bound": bnd,
            "outputs_bind_declined": int(fc.get("outputs_bind_declined") or 0),
            "alloc_device_frame": fc.get("alloc_device_frame"),
            "alloc_device_authority_grants": fc.get("alloc_device_authority_grants"),
            "route": (
                "STEP1C_DID_NOT_RUN" if att == 0
                else "STEP1C_RAN_NOTHING_BINDABLE" if bnd == 0
                else "STEP1C_BOUND"
            ),
        }
    report["routes"] = routes

    # ── the comparison: all 65, per output, per phase, against ship_off ──────────────────────
    base = "ship_off" if "ship_off" in lanes else lanes[0]
    report["baseline"] = base
    diffs: dict[str, dict] = {}
    for lane in lanes:
        if lane == base:
            continue
        per_phase = {}
        for phase in report["lanes"][base]["phases"]:
            b = report["lanes"][base]["phases"][phase]["outputs"]
            o = report["lanes"][lane]["phases"][phase]["outputs"]
            mismatched = [n for n in b if b[n]["sha256"] != o.get(n, {}).get("sha256")]
            per_phase[phase] = {
                "compared": len(b),
                "identical": len(b) - len(mismatched),
                "mismatched": mismatched,
                "argmax_base": report["lanes"][base]["phases"][phase]["argmax"],
                "argmax_lane": report["lanes"][lane]["phases"][phase]["argmax"],
            }
        diffs[lane] = per_phase
    report["vs_baseline"] = diffs

    # ── where the bytes are paid ─────────────────────────────────────────────────────────────
    report["byte_doors"] = {
        lane: {
            k: (report["lanes"][lane].get("final_counters") or {}).get(k)
            for k in ("session_staging_readback_bytes", "alloc_device_download_bytes",
                      "alloc_device_downloads", "alloc_device_upload_bytes")
        }
        for lane in lanes
    }

    bad = {
        lane: {p: d["mismatched"] for p, d in phases.items() if d["mismatched"]}
        for lane, phases in diffs.items()
    }
    bad = {k: v for k, v in bad.items() if v}
    if bad:
        report["verdict"] = "DEFAULT_FLIP_CHANGES_RESULTS"
        report["why"] = [
            f"{lane}: {sum(len(m) for m in phases.values())} output(s) differ from {base}: "
            f"{phases}"
            for lane, phases in bad.items()
        ]
        _emit(report)
        return 1

    report["verdict"] = "DEFAULT_FLIP_IS_TRANSPARENT"
    report["claim"] = (
        f"a caller who binds nothing gets byte-identical results on all "
        f"{len(report['lanes'][base]['phases']['decode1']['outputs'])} outputs across "
        f"{len(lanes)} lanes and 3 phases (prefill, decode, decode-fed-from-decode). "
        "This is a transparency result, not a saving: the naive caller reads every output, so "
        "he pays the same transfer through a different door (see byte_doors)."
    )
    _emit(report)
    return 0


def _emit(report: dict) -> None:
    path = HERE / "default_bind_outputs.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "lanes"}, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
