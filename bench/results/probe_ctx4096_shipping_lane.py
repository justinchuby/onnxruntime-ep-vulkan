r"""`device_memory_ctx4096_shipping_lane_cannot_run` — where does the shipping lane stop?

THE DEFECT
----------
At ctx 4096 the shipping lane (`ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY=0`) dies at

    alloc_device failed for input buffer

ORT rebuilds all 355 nodes on the CPU EP, the process exits 0, and the run reports plausible
logits with **zero EP dispatches**. That is why this file screens every arm on
`dispatches_executed > 0` before it reads any other number: it is the one screen that catches
exactly this fault, and a green exit is not evidence of anything here.

WHAT THIS PROBE MEASURES, AND WHAT IT DOES NOT
----------------------------------------------
It measures **where the boundary is** — the largest `past_len` at which the shipping lane
executes on the EP at all — and the **device-local high-water mark** at each point, split into
the resident part (the weight cache, paid once) and the transient part (allocated and freed
inside one `Compute`). It reads NO clock. Bytes and counts only. The device name is read off
the run's own `running_device_names`, never off the selector.

EVIDENCE HANDLING
-----------------
The previous generation of this instrument kept the worker's stderr **only on a non-zero
exit**, and the silent-CPU-rebuild path exits **zero** — so the one text that names the failure
(`alloc_device failed for input buffer`) was discarded on exactly the runs that had it. Every
arm here stores `stderr_tail` unconditionally. Mutation-verified: `--self-test` asserts the
field is populated on a zero-exit arm.

USAGE
-----
    $env:VULKAN_SDK="C:\VulkanSDK\1.4.350.0"; $env:PATH="$env:VULKAN_SDK\Bin;$env:PATH"
    $env:ONNXRUNTIME_VULKAN_EP_LIB="...\rust\target\release\onnxruntime_vulkan_ep.dll"
    python bench\results\probe_ctx4096_shipping_lane.py --plan 512,1024,2048,3072,4096
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
F16 = 2
BYTES_PER_PAST_TOKEN = LAYERS * 2 * KV_HEADS * HEAD_DIM * F16  # 393,216

# Every arm pins all four flags explicitly and records what it pinned. An unpinned flag is a
# lane whose identity is decided by whatever the shell inherited.
#
# `KV_PREFIX_ALIAS` is the one added this round, and it is the only flag in this EP that ships
# ON. `shipping` is therefore the *fixed* lane and `shipping_noalias` is the BEFORE control —
# the two differ in exactly one variable, which is what makes the comparison a comparison.
LANES = {
    "shipping": {
        "ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY": "0",
        "ONNXRUNTIME_EP_VULKAN_KV_ARENA": "0",
        "ONNXRUNTIME_EP_VULKAN_BIND_OUTPUTS": "1",
        "ONNXRUNTIME_EP_VULKAN_KV_PREFIX_ALIAS": "1",
    },
    "shipping_noalias": {
        "ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY": "0",
        "ONNXRUNTIME_EP_VULKAN_KV_ARENA": "0",
        "ONNXRUNTIME_EP_VULKAN_BIND_OUTPUTS": "1",
        "ONNXRUNTIME_EP_VULKAN_KV_PREFIX_ALIAS": "0",
    },
    "resident": {
        "ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY": "1",
        "ONNXRUNTIME_EP_VULKAN_KV_ARENA": "0",
        "ONNXRUNTIME_EP_VULKAN_BIND_OUTPUTS": "1",
        "ONNXRUNTIME_EP_VULKAN_KV_PREFIX_ALIAS": "1",
    },
}


def _lib() -> str:
    return os.environ.get(
        "ONNXRUNTIME_VULKAN_EP_LIB",
        str(REPO / "rust" / "target" / "release" / "onnxruntime_vulkan_ep.dll"),
    )


def _dll_hash() -> str:
    p = pathlib.Path(_lib())
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16].upper() if p.is_file() else "<absent>"


def _counters(path: pathlib.Path) -> dict:
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return doc.get("counters", doc)


# --------------------------------------------------------------------------- worker


def _worker(past: int, out_path: pathlib.Path) -> int:
    import numpy as np
    import onnxruntime as ort

    doc: dict = {
        "past": past,
        "env_pinned": {
            k: os.environ.get(k, "<unset>")
            for k in (
                "ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY",
                "ONNXRUNTIME_EP_VULKAN_KV_ARENA",
                "ONNXRUNTIME_EP_VULKAN_BIND_OUTPUTS",
                "ONNXRUNTIME_EP_VULKAN_KV_PREFIX_ALIAS",
            )
        },
        "ort_version": ort.__version__,
    }
    counters_path = pathlib.Path(os.environ[COUNTERS_ENV])
    counters_path.unlink(missing_ok=True)

    try:
        ort.register_execution_provider_library(EP_NAME, _lib())
    except Exception as exc:  # noqa: BLE001
        if "already registered" not in str(exc):
            raise

    so = ort.SessionOptions()
    sess = ort.InferenceSession(
        str(ONNX_FILE),
        so,
        providers=[EP_NAME, "CPUExecutionProvider"],
        free_dimension_overrides_by_name={"batch_size": "1", "sequence_length": "1"},
    )
    doc["session_providers"] = list(sess.get_providers())
    if EP_NAME not in sess.get_providers():
        doc["verdict"] = "ERROR(instrument)"
        doc["why"] = [f"{EP_NAME} absent from {sess.get_providers()}"]
        out_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        return 2

    # Counters right after session build: this is the RESIDENT term (weight cache), before any
    # Compute has allocated a transient byte.
    doc["counters_after_build"] = _counters(counters_path)

    rng = np.random.default_rng(20260804)
    feeds: dict = {}
    for layer in range(LAYERS):
        for side in ("key", "value"):
            feeds[f"past_key_values.{layer}.{side}"] = (
                rng.standard_normal((1, KV_HEADS, past, HEAD_DIM)).astype(np.float16) * 0.02
            )
    feeds["input_ids"] = np.array([[1]], dtype=np.int64)
    feeds["attention_mask"] = np.ones((1, past + 1), dtype=np.int64)
    if any(i.name == "position_ids" for i in sess.get_inputs()):
        feeds["position_ids"] = np.array([[past]], dtype=np.int64)
    feeds = {i.name: feeds[i.name] for i in sess.get_inputs() if i.name in feeds}

    try:
        outs = sess.run(["logits"], feeds)
        doc["ran"] = True
        doc["logits_sha256"] = hashlib.sha256(outs[0].tobytes()).hexdigest()[:16]
        doc["logits_finite"] = bool(np.isfinite(outs[0].astype(np.float32)).all())
    except Exception as exc:  # noqa: BLE001
        doc["ran"] = False
        doc["exception"] = f"{type(exc).__name__}: {exc}"[:2000]

    # -- Separating case: a chain on ONE session ------------------------------------------
    #
    # Everything above is a single `Compute` on a fresh session. Three of the separating cases
    # only exist across steps, and no single-shot arm can reach them:
    #
    #   * a session that outlives an inference,
    #   * a second `Compute` whose weight cache is already warm (the first Compute builds it
    #     *inside* the input loop, so the first Compute is also the peak),
    #   * a context that outgrows the allocation the first step made — which is the case the
    #     prefix alias is about, since the destination stride changes every step.
    #
    # The chain feeds each step's `present` back as the next step's `past`, on the host, so the
    # `past_len` genuinely grows. Per-step logits hashes are recorded, not just the last: a lane
    # that diverged at step 2 and reconverged would otherwise pass.
    steps = int(os.environ.get("PROBE_STEPS", "0"))
    if steps > 1 and doc.get("ran"):
        present_names = [
            f"present.{layer}.{side}" for layer in range(LAYERS) for side in ("key", "value")
        ]
        available = {o.name for o in sess.get_outputs()}
        chain: dict = {"steps": [], "requested": steps}
        try:
            if not set(present_names).issubset(available):
                chain["skipped"] = "present.* outputs are not exposed by this graph"
            else:
                cf = dict(feeds)
                for step in range(steps):
                    res = sess.run(["logits"] + present_names, cf)
                    logits = res[0]
                    chain["steps"].append(
                        {
                            "step": step,
                            "past_len_in": int(
                                cf["past_key_values.0.key"].shape[2]  # noqa: PD011
                            ),
                            "logits_sha256": hashlib.sha256(logits.tobytes()).hexdigest()[:16],
                            "logits_finite": bool(
                                np.isfinite(logits.astype(np.float32)).all()
                            ),
                        }
                    )
                    for name, arr in zip(present_names, res[1:]):
                        cf[name.replace("present.", "past_key_values.")] = arr
                    cf["input_ids"] = np.array([[int(logits[0, -1].argmax())]], dtype=np.int64)
                    grown = cf["past_key_values.0.key"].shape[2]
                    cf["attention_mask"] = np.ones((1, grown + 1), dtype=np.int64)
                    if "position_ids" in cf:
                        cf["position_ids"] = np.array([[grown]], dtype=np.int64)
                chain["final_past_len"] = int(cf["past_key_values.0.key"].shape[2])
        except Exception as exc:  # noqa: BLE001
            chain["exception"] = f"{type(exc).__name__}: {exc}"[:2000]
        doc["chain"] = chain

    doc["counters"] = _counters(counters_path)
    out_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return 0


# --------------------------------------------------------------------------- parent


def _run_arm(lane: str, past: int, scratch: pathlib.Path, steps: int = 0) -> dict:
    env = dict(os.environ)
    env.update(LANES[lane])
    env["ONNXRUNTIME_VULKAN_EP_LIB"] = _lib()
    env["PROBE_STEPS"] = str(steps)
    out = scratch / f"arm_{lane}_{past}.json"
    ctr = scratch / f"ctr_{lane}_{past}.json"
    out.unlink(missing_ok=True)
    ctr.unlink(missing_ok=True)
    env[COUNTERS_ENV] = str(ctr)
    env["PROBE_WORKER_OUT"] = str(out)
    proc = subprocess.run(
        [sys.executable, str(pathlib.Path(__file__).resolve()), "--worker", "--past", str(past)],
        env=env,
        capture_output=True,
        text=True,
        errors="replace",
    )
    rec: dict = {"lane": lane, "past": past, "exit": proc.returncode, "pinned": LANES[lane]}
    # UNCONDITIONAL. The silent-CPU-rebuild path exits zero and its stderr is the only place the
    # allocation failure is named. Keeping this only on non-zero exit destroyed the evidence.
    # The FULL stderr goes to a file, always. The tail is a convenience for reading the JSON.
    #
    # The tail alone is not evidence preservation. On the first AFTER run of this round the last
    # 6,000 characters were entirely `gpu_allocator` leak-report spam and the sentence naming the
    # actual failure — thousands of characters earlier — was gone. That is the same defect as
    # "keep stderr only on a non-zero exit" arriving by a different route: not discarded, but
    # crowded out. A truncation window is a filter, and a filter applied to evidence has to be
    # justified the same way a discard does. So: whole file on disk, tail for the eye, and
    # `alloc_device_failed_in_text` and friends are computed against the WHOLE text.
    log = scratch / f"stderr_{lane}_{past}.log"
    log.write_text(proc.stderr, encoding="utf-8", errors="replace")
    rec["stderr_log"] = str(log)
    rec["stderr_bytes"] = len(proc.stderr)
    rec["stderr_tail"] = proc.stderr[-6000:]
    rec["stderr_head"] = proc.stderr[:2000]
    # The interesting lines, wherever they are in the text: this survives any window.
    rec["stderr_signals"] = [
        ln[:500]
        for ln in proc.stderr.splitlines()
        if any(
            s in ln
            for s in (
                "alloc_device failed",
                "alloc_upload failed",
                "ERROR",
                "BROKEN COMMITMENT",
                "refus",
                "prefix alias",
                "DEVICE_LOST",
                "VK_ERROR",
                "panicked",
            )
        )
    ][:80]
    rec["stdout_tail"] = proc.stdout[-2000:]
    if out.is_file():
        rec["worker"] = json.loads(out.read_text(encoding="utf-8"))
    c = rec.get("worker", {}).get("counters", {}) or {}
    disp = int(c.get("dispatches_executed") or 0)
    rec["dispatches_executed"] = disp
    # THE SCREEN. Applied before any byte figure is read.
    rec["executed_on_ep"] = disp > 0
    rec["alloc_device_failed_in_text"] = "alloc_device failed" in proc.stderr
    rec["device_losses"] = int(c.get("device_losses") or 0)
    rec["running_device_names"] = c.get("running_device_names")
    for k in (
        "session_device_high_water_bytes",
        "session_device_bytes_in_use",
        "session_device_allocs",
        "session_device_frees",
        "weight_cache_bytes_resident",
        # The `closes_when` observable. `device_bytes` is the sum of the per-Compute input
        # allocations that actually happened; `reused_bytes` is the sum of the ones the prefix
        # alias made unnecessary. The defect closes on the first going to zero for the KV
        # inputs, not on a green exit.
        "transient_input_device_allocs",
        "transient_input_device_bytes",
        "transient_input_device_peak_bytes",
        "transient_input_reuses",
        "transient_input_reused_bytes",
    ):
        if k in c:
            rec[k] = c[k]
    w = rec.get("worker", {})
    # Correctness is read BEFORE any byte figure is compared, and from inputs both lanes share
    # exactly (one seed, `np.random.default_rng(20260804)`). A byte count on a lane whose answer
    # was never checked is the mistake a NaN certified once already.
    rec["logits_sha256"] = w.get("logits_sha256")
    rec["logits_finite"] = w.get("logits_finite")
    rec["env_pinned"] = w.get("env_pinned")
    ch = w.get("chain") or {}
    rec["chain_step_hashes"] = [s["logits_sha256"] for s in ch.get("steps", [])]
    rec["chain_all_finite"] = (
        all(s["logits_finite"] for s in ch.get("steps", [])) if ch.get("steps") else None
    )
    rec["chain_final_past_len"] = ch.get("final_past_len")
    rec["chain_exception"] = ch.get("exception")
    rec["chain_skipped"] = ch.get("skipped")
    build = rec.get("worker", {}).get("counters_after_build", {}) or {}
    rec["resident_high_water_after_build"] = build.get("session_device_high_water_bytes")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--past", type=int, default=4096)
    ap.add_argument("--plan", default="512,1024,2048,3072,4096")
    ap.add_argument("--lanes", default="shipping")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--out", default="ctx4096_shipping_lane.json")
    ap.add_argument(
        "--steps",
        type=int,
        default=0,
        help="run an N-step decode chain on one session after the single-shot arm",
    )
    args = ap.parse_args()

    scratch = HERE / "_ctx4096_scratch"
    scratch.mkdir(exist_ok=True)

    if args.worker:
        return _worker(args.past, pathlib.Path(os.environ["PROBE_WORKER_OUT"]))

    doc: dict = {
        "probe": "probe_ctx4096_shipping_lane",
        "dll_sha256_16": _dll_hash(),
        "bytes_per_past_token": BYTES_PER_PAST_TOKEN,
        "no_clock": True,
        "arms": [],
    }
    for lane in args.lanes.split(","):
        for past in [int(x) for x in args.plan.split(",") if x]:
            for r in range(args.repeats):
                rec = _run_arm(lane, past, scratch, steps=args.steps)
                rec["repeat"] = r
                doc["arms"].append(rec)
                print(
                    f"[{lane}] past={past} rep={r} exit={rec['exit']} "
                    f"dispatches={rec['dispatches_executed']} ep={rec['executed_on_ep']} "
                    f"alloc_fail_text={rec['alloc_device_failed_in_text']} "
                    f"hw={rec.get('session_device_high_water_bytes')}"
                )

    ran = [a for a in doc["arms"] if a["executed_on_ep"]]
    doc["largest_past_that_ran_on_ep"] = max((a["past"] for a in ran), default=None)
    doc["smallest_past_that_did_not"] = min(
        (a["past"] for a in doc["arms"] if not a["executed_on_ep"]), default=None
    )

    # -- Correctness control, read before any byte figure ----------------------------------
    #
    # Same seed, same feeds, same past length: two lanes that differ in exactly one flag must
    # produce byte-identical logits. Only pairs where BOTH lanes executed on the EP are
    # comparable — a lane that silently rebuilt on the CPU EP would otherwise "agree" with
    # itself and certify nothing. That is the failure this whole round is about, so it is
    # excluded explicitly rather than assumed away.
    by_key: dict = {}
    for a in doc["arms"]:
        if a["executed_on_ep"] and a.get("logits_sha256"):
            by_key.setdefault(a["past"], {})[a["lane"]] = a["logits_sha256"]
    identity = []
    for past, lanes in sorted(by_key.items()):
        if "shipping" in lanes and "shipping_noalias" in lanes:
            identity.append(
                {
                    "past": past,
                    "alias": lanes["shipping"],
                    "no_alias": lanes["shipping_noalias"],
                    "bit_identical": lanes["shipping"] == lanes["shipping_noalias"],
                    "both_on_ep": True,
                }
            )
    doc["bit_identity"] = identity
    doc["bit_identity_comparable_pairs"] = len(identity)
    doc["bit_identity_all_match"] = bool(identity) and all(x["bit_identical"] for x in identity)

    # The chain comparison is per-step, not just on the last step: a lane that diverged at
    # step 2 and reconverged at step 5 would pass an end-state check and be wrong throughout.
    chain_by_key: dict = {}
    for a in doc["arms"]:
        if a["executed_on_ep"] and a.get("chain_step_hashes"):
            chain_by_key.setdefault(a["past"], {})[a["lane"]] = a["chain_step_hashes"]
    chain_identity = []
    for past, lanes in sorted(chain_by_key.items()):
        if "shipping" in lanes and "shipping_noalias" in lanes:
            on, off = lanes["shipping"], lanes["shipping_noalias"]
            first = next(
                (i for i, (x, y) in enumerate(zip(on, off)) if x != y),
                None,
            )
            chain_identity.append(
                {
                    "seed_past": past,
                    "steps_compared": min(len(on), len(off)),
                    "first_divergent_step": first,
                    "bit_identical_every_step": first is None and len(on) == len(off),
                }
            )
    doc["chain_identity"] = chain_identity
    doc["logits_nonfinite_arms"] = [
        {"lane": a["lane"], "past": a["past"]}
        for a in doc["arms"]
        if a["executed_on_ep"] and a.get("logits_finite") is False
    ]

    out = HERE / args.out
    out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    for x in identity:
        verdict = "IDENTICAL" if x["bit_identical"] else "*** DIFFERS ***"
        print(f"  bit-identity past={x['past']}: {verdict}")
    for x in chain_identity:
        v = "IDENTICAL EVERY STEP" if x["bit_identical_every_step"] else "*** DIVERGES ***"
        print(
            f"  chain seed_past={x['seed_past']} steps={x['steps_compared']}: {v}"
            f" (first divergent step: {x['first_divergent_step']})"
        )
    if not identity:
        print("  bit-identity: NO COMPARABLE PAIR (run --lanes shipping,shipping_noalias)")
    print(f"largest past that ran on the EP: {doc['largest_past_that_ran_on_ep']}")
    print(f"smallest past that did not:      {doc['smallest_past_that_did_not']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
