"""Does the KV divergence grow with depth, or is layer 31 special?

Criterion 10: present.31.key and present.31.value OUTSIDE_TOLERANCE, layers 0..30 WITHIN.

Attribution (probe_gqa_attribution.py) established that all 32 GroupQueryAttention nodes
are DECLINED -- every present.* output is computed by the CPU EP in BOTH sessions. So the
divergence cannot be a Vulkan GQA kernel defect. What differs is the *input* to each GQA
node, which comes from Vulkan-executed MatMulNBits q/k/v projections.

That makes a prediction that can be falsified: if the cause is fp16 rounding in the Vulkan
projections propagating through the residual stream, the per-layer error must grow with
depth, and layer 31 is simply the first to cross a fixed threshold. If instead layer 31 is
qualitatively different -- a step, not a slope -- the accumulation story is wrong.

A slope and a step are distinguishable in one reading, so take it before theorising.

Also reports the error in ULP at fp16, because an absolute tolerance says nothing without
the magnitude it is absolute against, and a relative tolerance divides by values that are
near zero here (which is what produced rel 24.60 on the logits at abs 0.0625).

Writes bench/results/kv_depth_profile.json.
"""

import hashlib
import json
import os
import pathlib
import re
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
OUT = HERE / "kv_depth_profile.json"
EP = "VulkanExecutionProvider"


def sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def fp16_ulp(x: np.ndarray) -> np.ndarray:
    """Spacing of the fp16 grid at each |x|. Error below 1 ULP is unrepresentable."""
    a = np.abs(x.astype(np.float32))
    out = np.full(a.shape, np.float32(6.0e-8), dtype=np.float32)
    nz = a > 0
    if np.any(nz):
        e = np.floor(np.log2(a[nz]))
        out[nz] = np.exp2(e - 10).astype(np.float32)  # 10 mantissa bits
    return np.maximum(out, np.float32(6.0e-8))


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


def main() -> int:
    if not ONNX_FILE.exists():
        print(f"model missing: {ONNX_FILE}")
        return 2
    print(f"  DLL  {sha256(LIB)[:16]}")

    try:
        ort.register_execution_provider_library(EP, str(LIB))
    except Exception as e:
        if "already registered" not in str(e):
            raise

    cpu = ort.InferenceSession(str(ONNX_FILE), providers=["CPUExecutionProvider"])
    names = [o.name for o in cpu.get_outputs()]
    feeds = build_feeds(cpu)
    cpu_out = cpu.run(None, feeds)

    vk = ort.InferenceSession(
        str(ONNX_FILE), providers=[EP, "CPUExecutionProvider"])
    vk_out = vk.run(None, build_feeds(vk))

    rows = []
    for idx, nm in enumerate(names):
        a = np.asarray(cpu_out[idx]).astype(np.float32)
        b = np.asarray(vk_out[idx]).astype(np.float32)
        if a.shape != b.shape:
            rows.append({"index": idx, "name": nm, "shape_mismatch": True})
            continue
        d = np.abs(a - b)
        ulp = fp16_ulp(a)
        m = re.search(r"present\.(\d+)\.(key|value)", nm)
        rows.append({
            "index": idx,
            "name": nm,
            "layer": int(m.group(1)) if m else None,
            "kind": m.group(2) if m else "logits",
            "max_abs": float(np.max(np.abs(a))),
            "max_abs_diff": float(np.max(d)),
            "max_ulp_diff": float(np.max(d / ulp)),
            "mean_abs_diff": float(np.mean(d)),
            "n_differing": int(np.count_nonzero(d)),
            "n_elems": int(a.size),
            "all_zero_cpu": bool(np.all(a == 0)),
            "all_zero_vk": bool(np.all(b == 0)),
        })

    kv = [r for r in rows if r.get("layer") is not None]
    print(f"\n  outputs compared: {len(rows)}   KV outputs: {len(kv)}")
    deg = [r for r in rows if r.get("all_zero_cpu") or r.get("all_zero_vk")]
    print(f"  degenerate (all-zero either side): {len(deg)}")

    print("\n  KV error by depth (max over key/value at each layer):")
    print("    layer   max_abs   max_abs_diff   max_ulp   n_diff/n")
    prof = []
    for L in range(32):
        rs = [r for r in kv if r["layer"] == L]
        if not rs:
            continue
        mad = max(r["max_abs_diff"] for r in rs)
        mul = max(r["max_ulp_diff"] for r in rs)
        mab = max(r["max_abs"] for r in rs)
        nd = sum(r["n_differing"] for r in rs)
        ne = sum(r["n_elems"] for r in rs)
        prof.append({"layer": L, "max_abs": mab, "max_abs_diff": mad,
                     "max_ulp_diff": mul, "n_differing": nd, "n_elems": ne})
        if L < 4 or L > 26:
            print(f"    {L:5d}   {mab:8.4f}   {mad:12.6f}   {mul:7.2f}   {nd}/{ne}")
        elif L == 4:
            print("      ...")

    # --- the normalisation the gate should have used -----------------------
    # An absolute tolerance is meaningless without the magnitude it is absolute
    # against, and a relative tolerance divides elementwise by values that are
    # near zero here. The scale-free quantity for "did this tensor come out
    # right" is the error against the tensor's OWN range, in units of the fp16
    # grid spacing at that range. 1.0 means one ULP: the smallest difference
    # fp16 can represent at that magnitude, i.e. a different summation order.
    for p in prof:
        grid = float(fp16_ulp(np.array([p["max_abs"]], dtype=np.float32))[0])
        p["ulp_at_tensor_scale"] = p["max_abs_diff"] / grid if grid else float("nan")
        p["rel_to_tensor_scale"] = (
            p["max_abs_diff"] / p["max_abs"] if p["max_abs"] else float("nan"))

    ranked = sorted(prof, key=lambda p: -p["ulp_at_tensor_scale"])
    print("\n  Error against each tensor's OWN scale (1.0 == one fp16 ULP):")
    print("    rank  layer   max_abs   max_abs_diff   ULP@scale   flagged?")
    flagged = {31}
    for i, p in enumerate(ranked[:6]):
        mark = "OUTSIDE" if p["layer"] in flagged else "within"
        print(f"    {i + 1:4d}  {p['layer']:5d}   {p['max_abs']:8.4f}   "
              f"{p['max_abs_diff']:12.6f}   {p['ulp_at_tensor_scale']:9.3f}   {mark}")
    worst = ranked[0]
    l31 = next(p for p in prof if p["layer"] == 31)
    print(f"\n  worst layer by ULP-at-scale : {worst['layer']} "
          f"({worst['ulp_at_tensor_scale']:.3f} ULP)  -- "
          f"{'FLAGGED' if worst['layer'] in flagged else 'NOT flagged'}")
    print(f"  layer 31 by ULP-at-scale    : {l31['ulp_at_tensor_scale']:.3f} ULP  "
          f"(rank {ranked.index(l31) + 1} of {len(ranked)})  -- FLAGGED")
    max_ulp_at_scale = worst["ulp_at_tensor_scale"]
    print(f"\n  max over all 32 layers      : {max_ulp_at_scale:.3f} ULP")
    if max_ulp_at_scale <= 2.0:
        print("  READING: every layer is within ~1 ULP of its own scale. That is the")
        print("           signature of a different summation order in fp16, not a defect.")
        print("           The gate flagged layer 31 while layers with LARGER scale-relative")
        print("           error passed -- the threshold is not measuring correctness.")

    # --- can the proposed criterion still fail? ---------------------------
    # DANGER, stated plainly: I am about to argue that a gate which flagged my
    # subsystem is mis-scoped. That is exactly the argument someone makes when
    # they want a looser gate, and "the error is only rounding" is what it
    # sounds like whether it is true or false. The claim is worth nothing
    # unless the replacement criterion is shown to still FIRE on defects.
    #
    # So: plant wrong-and-stable corruptions into the comparison and measure
    # the margin between what rounding produces and what a defect produces.
    # These are plants of the COMPARISON, not of the kernel -- GQA runs on the
    # CPU EP in both sessions, so there is no Vulkan code path to corrupt here.
    # They test the criterion's discrimination, which is the thing in question.
    ref = np.asarray(cpu_out[63]).astype(np.float32)   # present.31.key
    scale = float(np.max(np.abs(ref)))
    grid = float(fp16_ulp(np.array([scale], dtype=np.float32))[0])

    def ulp_at_scale(bad):
        return float(np.max(np.abs(ref - bad))) / grid

    rng = np.random.default_rng(0)
    plants = {
        "all_zero (Morpheus's natural plant)": np.zeros_like(ref),
        "one_head_zeroed (head 0 of 32)": ref.copy(),
        "last_row_zeroed": ref.copy(),
        "scale_error_1pct": ref * np.float32(1.01),
        "scale_error_0.1pct": ref * np.float32(1.001),
        "single_element_sign_flip": ref.copy(),
        "fp16_requantise (a true no-op)": ref.astype(np.float16).astype(np.float32),
    }
    plants["one_head_zeroed (head 0 of 32)"][:, 0, :, :] = 0.0
    plants["last_row_zeroed"][..., -1, :] = 0.0
    flat = plants["single_element_sign_flip"].reshape(-1)
    flat[int(np.argmax(np.abs(flat)))] *= -1.0

    observed = max_ulp_at_scale
    print(f"\n  Planted-defect discrimination (present.31.key, scale {scale:.3f}):")
    print(f"    observed rounding across all 32 layers : {observed:.3f} ULP@scale")
    print("    plant                                      ULP@scale   vs observed")
    plant_rows = []
    for nm, bad in plants.items():
        u = ulp_at_scale(bad)
        ratio = u / observed if observed else float("inf")
        plant_rows.append({"plant": nm, "ulp_at_scale": u, "ratio_to_observed": ratio})
        print(f"    {nm:42s} {u:9.3f}   {ratio:8.1f}x")

    detectable = [p for p in plant_rows
                  if p["plant"].startswith(("all_zero", "one_head", "last_row",
                                            "scale_error_1pct"))]
    margin = min(p["ratio_to_observed"] for p in detectable)
    print(f"\n  narrowest margin among real defects: {margin:.1f}x above observed rounding")
    if margin > 10:
        print("  A threshold anywhere in the open band between them separates rounding")
        print("  from defect with an order of magnitude to spare. The band is not empty,")
        print("  and it is not narrow -- so the current flag is not protecting anything")
        print("  that a scale-relative threshold would stop protecting.")
    else:
        print("  WARNING: the band is narrow. A scale-relative threshold would trade this")
        print("  false positive for false negatives, and should NOT be adopted on this")
        print("  evidence. Report the narrowness rather than the reframing.")
    # An empty gap is a statement about the plants I happened to choose, and it
    # decays -- the same caution as the disturbance-band result. Recorded so a
    # later plant landing inside the band is recognised as new evidence, not noise.

    # slope or step?
    diffs = [p["max_abs_diff"] for p in prof]
    last = diffs[-1]
    prev = max(diffs[:-1]) if len(diffs) > 1 else 0.0
    rho = float(np.corrcoef(np.arange(len(diffs)), np.array(diffs))[0, 1]) \
        if len(diffs) > 2 and np.std(diffs) > 0 else float("nan")
    print(f"\n  corr(layer index, max_abs_diff) = {rho:.4f}")
    print(f"  max_abs_diff layer 31           = {last:.6f}")
    print(f"  max_abs_diff layers 0..30 (max) = {prev:.6f}")
    print(f"  step ratio 31 / max(0..30)      = {last / prev if prev else float('inf'):.3f}")

    if rho > 0.7:
        verdict = ("SLOPE: error grows monotonically with depth. Layer 31 is the deepest, "
                   "not a special case -- it is the first to cross a fixed threshold.")
    elif last > 4 * prev:
        verdict = ("STEP: layer 31 is qualitatively unlike 0..30. Accumulation does not "
                   "explain it; something about the last layer does.")
    else:
        verdict = ("NEITHER CLEAN: error is not depth-ordered and layer 31 is not a large "
                   "step. The threshold, not the error, is doing the deciding.")
    print(f"\n  VERDICT: {verdict}")

    rec = {
        "dll_sha256": sha256(LIB),
        "gqa_all_declined": True,
        "gqa_attribution_note":
            "all 32 GroupQueryAttention nodes DECLINED; present.* computed on CPU in both "
            "sessions, so any difference enters through Vulkan-computed GQA *inputs*",
        "per_output": rows,
        "per_layer": prof,
        "corr_layer_vs_max_abs_diff": rho,
        "planted_defect_discrimination": plant_rows,
        "observed_rounding_ulp_at_scale": max_ulp_at_scale,
        "narrowest_defect_margin": margin,
        "verdict": verdict,
    }
    OUT.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"\n  record: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
