"""Why did present.31.* trip, and would a scale-set tolerance be worse?

Established already:
  * all 32 GroupQueryAttention nodes are DECLINED -- present.* is CPU-computed in
    both sessions, so this is not a Vulkan GQA kernel defect (probe_gqa_attribution);
  * error is ~1-2 ULP of each tensor's own scale at EVERY layer, and layer 31 ranks
    30th of 32 -- nearly the cleanest -- yet is the only one flagged (probe_kv_depth).

The incumbent tolerance is atol=0.001, rtol=0.02, justified from a tensor whose
max_abs was 3.6e-3. The KV tensors reach max_abs 25.25. At 25.25 the fp16 grid
spacing is 0.0156, so atol=0.001 asks for 1/16 of a ULP: unsatisfiable by any fp16
tensor of that magnitude, however correct.

MECHANISM, stated so it can be wrong: numpy's `atol + rtol*|b|` sets each element's
budget from that ELEMENT's magnitude, but in a summed reduction the rounding error is
set by the TENSOR's magnitude -- the accumulator visits large terms whatever the
output element turns out to be. So a small element inside a large tensor gets a small
budget and a large error, and trips. PREDICTION: the failing elements are the small
ones. If the failing elements are large, this mechanism is wrong.

Then the only comparison that licenses a change: incumbent vs proposal scored on the
SAME planted defects. A replacement that is merely prettier is not admissible; it has
to catch at least what the incumbent catches.

Writes bench/results/kv_tolerance_mechanism.json.
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
OUT = HERE / "kv_tolerance_mechanism.json"
EP = "VulkanExecutionProvider"

ATOL, RTOL = 0.001, 0.02          # incumbent, from criterion10-dev0.json
K_ULP = 6.0                        # proposal: k ULP of the tensor's own scale


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def fp16_grid(scale: float) -> float:
    if scale <= 0:
        return 6.0e-8
    return float(np.exp2(np.floor(np.log2(scale)) - 10))


def incumbent_passes(a, b):
    return bool(np.all(np.abs(a - b) <= ATOL + RTOL * np.abs(b)))


def proposal_passes(a, b, scale):
    return bool(np.max(np.abs(a - b)) <= K_ULP * fp16_grid(scale))


def build_feeds(sess):
    feeds = {}
    for i in sess.get_inputs():
        if i.name == "input_ids":
            feeds[i.name] = np.array([[1, 2, 3, 4]], dtype=np.int64)
            continue
        if i.name == "attention_mask":
            feeds[i.name] = np.ones((1, 4), dtype=np.int64)
            continue
        dt = np.int64 if "int64" in i.type else (
            np.float16 if "float16" in i.type else np.float32)
        shp = [1 if isinstance(d, str) or d is None else d for d in i.shape]
        if "past_key_values" in i.name:
            shp[2] = 0
        feeds[i.name] = np.zeros(shp, dtype=dt)
    return feeds


def main():
    print(f"  DLL  {sha256(LIB)[:16]}")
    try:
        ort.register_execution_provider_library(EP, str(LIB))
    except Exception as e:
        if "already registered" not in str(e):
            raise

    cpu = ort.InferenceSession(str(ONNX_FILE), providers=["CPUExecutionProvider"])
    names = [o.name for o in cpu.get_outputs()]
    cpu_out = cpu.run(None, build_feeds(cpu))
    vk = ort.InferenceSession(str(ONNX_FILE), providers=[EP, "CPUExecutionProvider"])
    vk_out = vk.run(None, build_feeds(vk))

    # ---- 1. the mechanism's prediction -----------------------------------
    print("\n  [1] Which elements trip the incumbent, and how big are they?")
    print("      idx  name                 scale   atol_needs  fp16_grid  "
          "n_fail  median|b| fail  median|b| all")
    mech = []
    for idx in (63, 64, 3, 56):          # two flagged, two that passed
        a = np.asarray(cpu_out[idx]).astype(np.float32)
        b = np.asarray(vk_out[idx]).astype(np.float32)
        scale = float(np.max(np.abs(a)))
        budget = ATOL + RTOL * np.abs(b)
        fail = np.abs(a - b) > budget
        nf = int(np.count_nonzero(fail))
        med_fail = float(np.median(np.abs(b[fail]))) if nf else float("nan")
        med_all = float(np.median(np.abs(b)))
        mech.append({"index": idx, "name": names[idx], "scale": scale,
                     "fp16_grid_at_scale": fp16_grid(scale), "n_fail": nf,
                     "n_elems": int(a.size),
                     "median_abs_failing": med_fail, "median_abs_all": med_all,
                     "incumbent_passes": nf == 0})
        print(f"      {idx:3d}  {names[idx]:20s} {scale:6.2f}  {ATOL:10.4f}  "
              f"{fp16_grid(scale):9.5f}  {nf:6d}  {med_fail:14.5f}  {med_all:12.5f}")

    flagged = [m for m in mech if not m["incumbent_passes"]]
    if flagged and all(m["median_abs_failing"] < m["median_abs_all"] for m in flagged):
        print("\n      PREDICTION HELD: failing elements are smaller than typical.")
        print("      The budget is set by the element; the error is set by the tensor.")
        mech_verdict = "held"
    elif flagged:
        print("\n      PREDICTION FAILED: failing elements are not the small ones.")
        print("      The stated mechanism is wrong -- do not act on it.")
        mech_verdict = "failed"
    else:
        print("\n      No output trips the incumbent in this run; mechanism untested here.")
        mech_verdict = "untested"

    # ---- 2. incumbent vs proposal on identical plants ---------------------
    ref = np.asarray(cpu_out[63]).astype(np.float32)
    scale = float(np.max(np.abs(ref)))
    rng = np.random.default_rng(0)
    plants = {
        "TRUE NEGATIVE: observed Vulkan output":
            np.asarray(vk_out[63]).astype(np.float32),
        "TRUE NEGATIVE: fp16 requantise (no-op)":
            ref.astype(np.float16).astype(np.float32),
        "DEFECT: all_zero": np.zeros_like(ref),
        "DEFECT: head 0 zeroed": None,
        "DEFECT: last row zeroed": None,
        "DEFECT: scale error 1%": ref * np.float32(1.01),
        "DEFECT: scale error 0.1%": ref * np.float32(1.001),
        "DEFECT: single sign flip": None,
        "DEFECT: 1 ULP added everywhere":
            ref + np.float32(fp16_grid(scale)),
        "DEFECT: gaussian noise 1% of scale":
            ref + rng.normal(0, 0.01 * scale, ref.shape).astype(np.float32),
    }
    h = ref.copy(); h[:, 0, :, :] = 0.0
    plants["DEFECT: head 0 zeroed"] = h
    lr = ref.copy(); lr[..., -1, :] = 0.0
    plants["DEFECT: last row zeroed"] = lr
    sf = ref.copy(); f = sf.reshape(-1); f[int(np.argmax(np.abs(f)))] *= -1.0
    plants["DEFECT: single sign flip"] = sf

    print(f"\n  [2] Incumbent (atol={ATOL}, rtol={RTOL}) vs proposal "
          f"(<= {K_ULP:g} ULP of tensor scale {scale:.2f} = "
          f"{K_ULP * fp16_grid(scale):.4f}):")
    print("      case                                     incumbent   proposal   agree")
    rows = []
    for nm, bad in plants.items():
        ip, pp = incumbent_passes(ref, bad), proposal_passes(ref, bad, scale)
        want_pass = nm.startswith("TRUE NEGATIVE")
        rows.append({"case": nm, "incumbent_passes": ip, "proposal_passes": pp,
                     "should_pass": want_pass})
        print(f"      {nm:40s} {'PASS' if ip else 'FAIL':>9s}  "
              f"{'PASS' if pp else 'FAIL':>9s}   {'yes' if ip == pp else 'NO'}")

    inc_wrong = [r for r in rows if r["incumbent_passes"] != r["should_pass"]]
    pro_wrong = [r for r in rows if r["proposal_passes"] != r["should_pass"]]
    missed_by_proposal = [r for r in rows
                          if not r["should_pass"] and r["proposal_passes"]
                          and not r["incumbent_passes"]]

    print(f"\n      incumbent wrong on {len(inc_wrong)}/{len(rows)}: "
          f"{[r['case'] for r in inc_wrong]}")
    print(f"      proposal  wrong on {len(pro_wrong)}/{len(rows)}: "
          f"{[r['case'] for r in pro_wrong]}")
    print(f"      defects the incumbent catches and the proposal MISSES: "
          f"{[r['case'] for r in missed_by_proposal] or 'none'}")

    if missed_by_proposal:
        verdict = ("REJECT the proposal: it loses detections the incumbent has. "
                   "Report the mis-scoping; do not change the threshold.")
    elif len(pro_wrong) < len(inc_wrong):
        verdict = (f"Proposal dominates on these {len(rows)} cases: it is wrong on "
                   f"{len(pro_wrong)} where the incumbent is wrong on {len(inc_wrong)}, "
                   f"and misses nothing the incumbent catches.")
    else:
        verdict = ("No improvement demonstrated. Keep the incumbent and report the "
                   "mis-scoping as an open finding.")
    print(f"\n  VERDICT: {verdict}")
    print("\n  NOTE: 'DEFECT: scale error 0.1%' is missed by BOTH. That is a pre-existing")
    print("  blind spot, not one the proposal introduces -- but it is now written down,")
    print("  because a gate's blind spots decay out of memory faster than its successes.")

    OUT.write_text(json.dumps({
        "dll_sha256": sha256(LIB),
        "incumbent": {"atol": ATOL, "rtol": RTOL,
                      "justified_for_max_abs": 3.6e-3,
                      "applied_to_max_abs": scale,
                      "atol_in_ulp_at_applied_scale": ATOL / fp16_grid(scale)},
        "proposal": {"k_ulp_of_tensor_scale": K_ULP,
                     "absolute_at_this_scale": K_ULP * fp16_grid(scale)},
        "mechanism_prediction": mech_verdict,
        "mechanism_detail": mech,
        "plants": rows,
        "defects_missed_by_proposal_but_caught_by_incumbent":
            [r["case"] for r in missed_by_proposal],
        "verdict": verdict,
    }, indent=2), encoding="utf-8")
    print(f"\n  record: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
