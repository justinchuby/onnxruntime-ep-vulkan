"""Does the criterion-10 flagged set depend on the prompt?

While testing the tolerance mechanism I scored four outputs under the incumbent
(atol=0.001, rtol=0.02) with my own feed and found `present.27.value` failing --
an output the criterion-10 run reported WITHIN_TOLERANCE. Two readings of the same
gate on the same binary disagreeing about which outputs are wrong is either a
difference in the input, or a defect in one of the two readings.

The cheap discriminator: hold the binary and the tolerance fixed, vary only the feed,
and count. If the flagged SET moves with the prompt, the gate is reporting a property
of the input, not a property of the EP -- and "3 of 65 outputs diverge" is then a
sentence about the tokens that happened to be fed.

This is not an argument for a looser gate. It is the reason the eventual per-output
tolerance has to be justified against something stable; and if the set is unstable,
the count needs to travel with its feed the way a timing travels with its device state.

Writes bench/results/kv_feed_sensitivity.json.
"""

import hashlib
import json
import os
import pathlib
import sys

import numpy as np
import onnxruntime as ort

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
# ARCHIVAL: pinned to the exact Foundry cache layout this investigation measured against
# (the pre-2026-08-05 "...-cuda-gpu/cuda-int4-rtn-block-32/..." catalog revision, issue
# #11). Intentionally NOT auto-resolved against the live cache: a live resolver could
# silently pick a *different* cached revision than the one this result was measured
# against, which would misattribute a new run to an old artifact. Override PHI35_MODEL to
# replay against a different artifact explicitly (issue #19).
MODEL_DIR = pathlib.Path(
    r"C:\Users\justinchu\.foundry\cache\models\Microsoft"
    r"\Phi-3.5-mini-instruct-cuda-gpu\cuda-int4-rtn-block-32"
)
ONNX_FILE = pathlib.Path(
    os.environ.get(
        "PHI35_MODEL",
        str(MODEL_DIR / "phi-3.5-mini-instruct-cuda-int4-rtn-block-32.onnx"),
    )
)
LIB = REPO / "rust" / "target" / "release" / "onnxruntime_vulkan_ep.dll"
OUT = HERE / "kv_feed_sensitivity.json"
EP = "VulkanExecutionProvider"
ATOL, RTOL = 0.001, 0.02

# Result-identity contract (issue #19 follow-up, Morpheus review on PR #31): the resolved model
# path and its exact content hash are stamped into the output record below, computed lazily
# (only once the model has already been opened successfully) so a PHI35_MODEL override or a
# stale/wrong cached file can never be silently absorbed into the evidence. Reuses the streaming
# SHA-256 helper `model_provenance.sha256_of` rather than a 23rd divergent hasher.
sys.path.insert(0, str(REPO / "rust" / "tools"))
import model_provenance as _model_provenance  # noqa: E402


def _result_identity() -> dict:
    return {
        "onnx_file": str(ONNX_FILE),
        "onnx_sha256": _model_provenance.sha256_of(ONNX_FILE),
    }


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def feeds_for(sess, ids):
    ids = np.asarray(ids, dtype=np.int64).reshape(1, -1)
    out = {}
    for i in sess.get_inputs():
        if i.name == "input_ids":
            out[i.name] = ids
            continue
        if i.name == "attention_mask":
            out[i.name] = np.ones(ids.shape, dtype=np.int64)
            continue
        dt = np.int64 if "int64" in i.type else (
            np.float16 if "float16" in i.type else np.float32)
        shp = [1 if isinstance(d, str) or d is None else d for d in i.shape]
        if "past_key_values" in i.name:
            shp[2] = 0
        out[i.name] = np.zeros(shp, dtype=dt)
    return out


def main():
    print(f"  DLL  {sha256(LIB)[:16]}   tolerance atol={ATOL} rtol={RTOL} (both held fixed)")
    try:
        ort.register_execution_provider_library(EP, str(LIB))
    except Exception as e:
        if "already registered" not in str(e):
            raise

    cpu = ort.InferenceSession(str(ONNX_FILE), providers=["CPUExecutionProvider"])
    vk = ort.InferenceSession(str(ONNX_FILE), providers=[EP, "CPUExecutionProvider"])
    names = [o.name for o in cpu.get_outputs()]

    rng = np.random.default_rng(7)
    feed_sets = {
        "A: [1,2,3,4]": [1, 2, 3, 4],
        "B: [450,3,29871,13]": [450, 3, 29871, 13],
        "C: random x4": rng.integers(1, 32000, size=4).tolist(),
        "D: random x4 (other)": rng.integers(1, 32000, size=4).tolist(),
        "E: [1,1,1,1]": [1, 1, 1, 1],
    }

    results = {}
    for label, ids in feed_sets.items():
        co = cpu.run(None, feeds_for(cpu, ids))
        vo = vk.run(None, feeds_for(vk, ids))
        failing = []
        for idx in range(len(names)):
            a = np.asarray(co[idx]).astype(np.float32)
            b = np.asarray(vo[idx]).astype(np.float32)
            if a.shape != b.shape:
                failing.append(idx)
                continue
            if not np.all(np.abs(a - b) <= ATOL + RTOL * np.abs(b)):
                failing.append(idx)
        results[label] = failing
        print(f"    {label:24s} ids={ids}  outside_tolerance = "
              f"{len(failing):2d}/65   {failing[:8]}{'...' if len(failing) > 8 else ''}")

    sets = [set(v) for v in results.values()]
    union = sorted(set().union(*sets))
    inter = sorted(set(sets[0]).intersection(*sets[1:]))
    counts = sorted({len(v) for v in results.values()})

    print(f"\n    union of flagged outputs across feeds : {len(union)}/65")
    print(f"    intersection (flagged under EVERY feed): {len(inter)}/65  {inter}")
    print(f"    distinct counts observed               : {counts}")

    stable = len(counts) == 1 and len(union) == len(inter)
    if stable:
        verdict = ("STABLE: the flagged set does not move with the prompt. "
                   "'N of 65 diverge' is a statement about the EP.")
    else:
        verdict = (f"FEED-DEPENDENT: the count ranges over {counts} and only "
                   f"{len(inter)} of {len(union)} flagged outputs are flagged under every "
                   f"feed. 'N of 65 diverge' is a statement about the EP AND the tokens; "
                   f"it must travel with its feed the way a timing travels with its "
                   f"device state.")
    print(f"\n  VERDICT: {verdict}")

    OUT.write_text(json.dumps({
        **_result_identity(),
        "dll_sha256": sha256(LIB),
        "atol": ATOL, "rtol": RTOL,
        "note": "binary and tolerance held fixed; only input_ids varied",
        "feeds": {k: {"input_ids": v, "outside_tolerance": results[k],
                      "n_outside": len(results[k])}
                  for k, v in feed_sets.items()},
        "union": union, "intersection": inter, "distinct_counts": counts,
        "stable": stable, "verdict": verdict,
    }, indent=2), encoding="utf-8")
    print(f"\n  record: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
