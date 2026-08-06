"""Does an output the kernel wrote straight into ORT's device buffer reach the caller?

`probe_output_residency.py` established that ORT allocates every fused-node output through
this EP's device provider (195/195 with the allocator armed). So an output-side
`bind_target_for` has something to bind, and `ONNXRUNTIME_EP_VULKAN_BIND_OUTPUTS=1` now
dispatches straight into ORT's own buffer and skips the download and the host memcpy.

That is only a saving if the bytes are actually readable at the other end. `transfer`'s own
documentation says a device-backed span keeps its host staging block and that **the staging
block stays authoritative** — the device buffer is described as a mirror. If that is still
true for outputs, the bound lane returns whatever the host block held before, which for a
zeroed fresh allocation is zeros and for a reused one is the previous inference's answer.

Either way it would be WRONG, and wrong in the way that survives a smoke test. So the flag is
default-OFF and this probe is what decides whether it may ever be default-ON.

THE COMPARISON
--------------
Same model, same feeds, three lanes:

    cpu      CPUExecutionProvider only          the reference
    ep       Vulkan EP, device allocator armed, BIND_OUTPUTS off   (today's shipped path)
    bound    Vulkan EP, device allocator armed, BIND_OUTPUTS on    (the change)

`ep` vs `cpu` establishes the tolerance the EP already meets. `bound` vs `cpu` must meet the
same one. `bound` vs `ep` is reported too, because the interesting failure is not "slightly
different" but "identical to the digit to something that is not the answer".

NON-TRIVIALITY
--------------
Every lane asserts the EP actually ran (`compute_calls`, `dispatches_executed`) and that the
bound lane actually bound something (readback bytes must FALL). A bound lane that quietly
declined every bind would agree with `ep` perfectly and prove nothing — that is the switched-
off detector again, and it is refused rather than reported.

Usage:  python bench/results/probe_bound_output_correctness.py
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import onnxruntime as ort

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
EP_NAME = "VulkanExecutionProvider"
COUNTERS_ENV = "ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE"

# ARCHIVAL: pinned to the exact Foundry cache layout this investigation measured against
# (the pre-2026-08-05 "...-cuda-gpu/cuda-int4-rtn-block-32/..." catalog revision, issue
# #11). Intentionally NOT auto-resolved against the live cache: a live resolver could
# silently pick a *different* cached revision than the one this result was measured
# against, which would misattribute a new run to an old artifact. Override PHI35_MODEL to
# replay against a different artifact explicitly (issue #19).
PHI = pathlib.Path(
    os.environ.get(
        "PHI35_MODEL",
        r"C:\Users\justinchu\.foundry\cache\models\Microsoft"
        r"\Phi-3.5-mini-instruct-cuda-gpu\cuda-int4-rtn-block-32"
        r"\phi-3.5-mini-instruct-cuda-int4-rtn-block-32.onnx",
    )
)

LAYERS, KV_HEADS, HEAD_DIM = 32, 32, 96
PAST_LEN = 8
ITERS = 2

# Result-identity contract (issue #19 follow-up, Morpheus review on PR #31): the resolved model
# path and its exact content hash are stamped into the output record below, computed lazily
# (only once the model has already been used successfully) so a PHI35_MODEL override or a
# stale/wrong cached file can never be silently absorbed into the evidence. Reuses the streaming
# SHA-256 helper `model_provenance.sha256_of` rather than a 23rd divergent hasher.
sys.path.insert(0, str(ROOT / "rust" / "tools"))
import model_provenance as _model_provenance  # noqa: E402


def _result_identity() -> dict:
    return {
        "onnx_file": str(PHI),
        "onnx_sha256": _model_provenance.sha256_of(PHI),
    }


def feeds(past_len: int) -> dict:
    rng = np.random.default_rng(7)
    f = {
        "input_ids": np.array([[1071]], dtype=np.int64),
        "attention_mask": np.ones((1, past_len + 1), dtype=np.int64),
    }
    for layer in range(LAYERS):
        f[f"past_key_values.{layer}.key"] = (
            rng.standard_normal((1, KV_HEADS, past_len, HEAD_DIM)) * 0.1
        ).astype(np.float16)
        f[f"past_key_values.{layer}.value"] = (
            rng.standard_normal((1, KV_HEADS, past_len, HEAD_DIM)) * 0.1
        ).astype(np.float16)
    return f


def worker(lane: str, out_path: str) -> int:
    if lane == "cpu":
        sess = ort.InferenceSession(str(PHI), providers=["CPUExecutionProvider"],
                                    free_dimension_overrides_by_name={
                                        "batch_size": "1", "sequence_length": "1"})
    else:
        ort.register_execution_provider_library(
            EP_NAME, os.environ["ONNXRUNTIME_VULKAN_EP_LIB"])
        sess = ort.InferenceSession(
            str(PHI), providers=[EP_NAME, "CPUExecutionProvider"],
            free_dimension_overrides_by_name={"batch_size": "1", "sequence_length": "1"})
        if EP_NAME not in sess.get_providers():
            raise SystemExit(f"ERROR(instrument): {EP_NAME} absent")
        sess.disable_fallback()
    names = [o.name for o in sess.get_outputs()]
    f = feeds(PAST_LEN)
    outs = None
    for _ in range(ITERS):  # more than one, so a stale-by-one-inference read is visible
        outs = sess.run(None, f)
    picked = {"logits": np.asarray(outs[names.index("logits")], dtype=np.float32)}
    for name in ("present.0.key", "present.31.value"):
        if name in names:
            picked[name] = np.asarray(outs[names.index(name)], dtype=np.float32)
    np.savez(out_path, **picked)
    return 0


def run_lane(lane: str) -> dict:
    scratch = ROOT / "bench" / "_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    npz = scratch / f"boundout_{lane}.npz"
    cfile = scratch / f"boundout_{lane}.counters.json"
    npz.unlink(missing_ok=True)
    cfile.unlink(missing_ok=True)
    env = dict(os.environ)
    env[COUNTERS_ENV] = str(cfile)
    env.pop("ONNXRUNTIME_EP_VULKAN_BIND_OUTPUTS", None)
    if lane == "cpu":
        env.pop("ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY", None)
    else:
        env["ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY"] = "1"
        if lane == "bound":
            env["ONNXRUNTIME_EP_VULKAN_BIND_OUTPUTS"] = "1"
    env["ONNXRUNTIME_EP_VULKAN_TRACE"] = "1"
    proc = subprocess.run(
        [sys.executable, str(pathlib.Path(__file__).resolve()),
         "--worker", "--lane", lane, "--out", str(npz)],
        capture_output=True, text=True, env=env,
    )
    if not npz.is_file():
        raise SystemExit(
            f"ERROR(instrument): lane {lane} produced no outputs.\n"
            + (proc.stdout or "") + (proc.stderr or "")
        )
    c = {}
    if cfile.is_file():
        raw = json.loads(cfile.read_text(encoding="utf-8"))
        c = raw.get("counters", raw)
    return {"lane": lane, "npz": npz, "exit": proc.returncode, "counters": c,
            "text": (proc.stdout or "") + (proc.stderr or "")}


def worst_rel(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.maximum(np.abs(a), np.abs(b))
    denom = np.where(denom < 1e-3, 1e-3, denom)
    return float(np.max(np.abs(a - b) / denom))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--lane", default="cpu")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.worker:
        return worker(args.lane, args.out)

    lanes = {name: run_lane(name) for name in ("cpu", "ep", "bound")}
    data = {k: dict(np.load(v["npz"])) for k, v in lanes.items()}

    print()
    print("=" * 78)
    print("DOES A BOUND OUTPUT REACH THE CALLER, OR IS HOST STAGING AUTHORITATIVE?")
    print("=" * 78)

    for name in ("ep", "bound"):
        c = lanes[name]["counters"]
        print(
            f"  {name:6s} calls={c.get('compute_calls')} "
            f"dispatches={c.get('dispatches_executed')} "
            f"device_resident_outputs={c.get('outputs_device_resident')} "
            f"bound_outputs={c.get('outputs_device_bound')} "
            f"device_losses={c.get('device_losses')}"
        )

    problems = []
    for name in ("ep", "bound"):
        c = lanes[name]["counters"]
        if not (c.get("compute_calls") or 0) or not (c.get("dispatches_executed") or 0):
            problems.append(f"lane {name} executed no EP work")
        if c.get("device_losses"):
            problems.append(f"lane {name} lost the device")
        if not (c.get("outputs_device_resident") or 0) and not (
            c.get("outputs_device_bound") or 0
        ):
            problems.append(
                f"lane {name} saw no device-resident outputs — the device allocator did not "
                "take effect, so the bound lane had nothing to bind"
            )
    if not (lanes["bound"]["counters"].get("outputs_device_bound") or 0):
        problems.append(
            "the bound lane bound no outputs at all. It would then be identical to the shipped "
            "path and would 'pass' without exercising the change"
        )

    print()
    rows = []
    for key in sorted(data["cpu"]):
        r_ep = worst_rel(data["cpu"][key], data["ep"][key])
        r_bd = worst_rel(data["cpu"][key], data["bound"][key])
        r_be = worst_rel(data["ep"][key], data["bound"][key])
        identical = bool(np.array_equal(data["ep"][key], data["bound"][key]))
        rows.append((key, r_ep, r_bd, r_be, identical))
        print(f"  {key:18s} cpu~ep={r_ep:.6g}  cpu~bound={r_bd:.6g}  "
              f"ep~bound={r_be:.6g}  ep==bound_bitwise={identical}")

    worst_ep = max(r[1] for r in rows)
    worst_bound = max(r[2] for r in rows)
    worst_ep_vs_bound = max(r[3] for r in rows)

    # Degeneracy guard. An output that is entirely zero is not an output that agrees; it is an
    # output that was never written. This check exists because the first version of this scorer
    # did not have it and returned BOUND_OUTPUT_IS_SOUND on an all-zero result: `cpu~bound` was
    # 1.0 on every tensor, `cpu~ep` was 1.9958 because the logits metric saturates on near-zero
    # values, and a max-over-outputs comparison let the zero lane "win". A scoring rule that can
    # be beaten by returning nothing is not a scoring rule.
    degenerate = [
        k for k in sorted(data["bound"])
        if not np.count_nonzero(data["bound"][k]) and np.count_nonzero(data["ep"][k])
    ]

    print()
    if degenerate:
        verdict = "HOST_STAGING_IS_AUTHORITATIVE"
        detail = (
            f"the bound lane returned ALL ZEROS for {degenerate} while the shipped path returned "
            "nonzero values for the same tensors. The kernel wrote ORT's device buffer and the "
            "caller read the host staging block, which nothing wrote. The output-side bind "
            "cannot be enabled in this shape."
        )
    elif problems:
        verdict = "ERROR(instrument)"
        detail = "; ".join(problems)
    elif worst_ep_vs_bound <= 1e-6:
        # The primary criterion is ep vs bound, not cpu vs bound. Same kernel, same inputs, same
        # arithmetic — only the writeback path differs — so anything but agreement to the digit
        # is the writeback path changing the answer, and comparing to the CPU EP instead lets
        # the EP's own tolerance hide it.
        verdict = "BOUND_OUTPUT_IS_SOUND"
        detail = (
            f"the bound lane agrees with the shipped path to {worst_ep_vs_bound:.6g} and with "
            f"the CPU EP to {worst_bound:.6g} (shipped path: {worst_ep:.6g}). Writing straight "
            "into ORT's device buffer reaches the caller."
        )
    else:
        verdict = "BOUND_OUTPUT_DIVERGES"
        detail = (
            f"the bound lane differs from the shipped path by {worst_ep_vs_bound:.6g}. Only the "
            "writeback path differs between them, so the writeback path is changing the answer."
        )

    print(f"  VERDICT: {verdict}")
    print(f"    {detail}")

    rec = HERE / "bound_output_correctness.json"
    rec.write_text(json.dumps({
        **_result_identity(),
        "verdict": verdict, "detail": detail,
        "worst_rel_cpu_vs_ep": worst_ep, "worst_rel_cpu_vs_bound": worst_bound,
        "per_output": [
            {"name": k, "cpu_vs_ep": a, "cpu_vs_bound": b, "ep_vs_bound": c, "bitwise": d}
            for k, a, b, c, d in rows
        ],
        "counters": {k: v["counters"] for k, v in lanes.items()},
    }, indent=2), encoding="utf-8")
    print(f"\n  record: {rec}")
    return 0 if verdict == "BOUND_OUTPUT_IS_SOUND" else 1


if __name__ == "__main__":
    raise SystemExit(main())
