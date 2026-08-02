"""Reproduce the GQA divergence at my own commit, then localise it.

The ledger records worst_rel = 16.72642029784887 for
  com.microsoft::GroupQueryAttention/1+/f16,f16,f16,i32,i32,f16,f16>f16,f16,f16
        /metadata/runtime-extent/past_key+past_value+cos_cache+sin_cache

Quoting that number would give one instrument reporting twice. Reproduce it here, on
this binary, before touching anything -- and then ask the question the ledger does not:
WHICH output, WHICH elements, and is the wrongness structured?

16.7 is not rounding. At 1-2 ULP a kernel is right; at 16x relative it is a wrong
formula, layout, mask, or ordering. Structure in the error map tells them apart:
  * whole head wrong          -> head indexing / grouping
  * whole row (position) wrong-> mask or seqlens_k
  * whole tail of a row wrong -> past/present concatenation order
  * scattered                 -> reduction or dequant
  * exactly one column band   -> rotary (cos/sin) application

Writes bench/results/gqa_divergence.json.
"""

import hashlib
import json
import os
import pathlib
import sys

import numpy as np
import onnx
import onnxruntime as ort

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
CASE = REPO / "evidence" / "cases" / "group_query_attention_f16.onnx"
LIB = REPO / "rust" / "target" / "release" / "onnxruntime_vulkan_ep.dll"
OUT = HERE / "gqa_divergence.json"
EP = "VulkanExecutionProvider"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    print(f"  DLL   {sha256(LIB)[:16]}")
    print(f"  case  {CASE.name}  exists={CASE.exists()}")
    if not CASE.exists():
        return 2

    m = onnx.load(str(CASE))
    g = m.graph
    node = next(n for n in g.node if n.op_type == "GroupQueryAttention")
    print(f"\n  node {node.name!r}  {len(node.input)} inputs -> {len(node.output)} outputs")
    for i, nm in enumerate(node.input):
        print(f"    in [{i}] {nm!r}")
    for i, nm in enumerate(node.output):
        print(f"    out[{i}] {nm!r}")
    attrs = {}
    for a in node.attribute:
        v = (a.i if a.type == a.INT else
             a.f if a.type == a.FLOAT else
             a.s.decode() if a.type == a.STRING else "?")
        attrs[a.name] = v
    print(f"  attributes: {attrs}")

    def tinfo(vi):
        t = vi.type.tensor_type
        shp = [d.dim_value if d.HasField("dim_value") else d.dim_param
               for d in t.shape.dim]
        return shp
    print("\n  graph inputs:")
    for vi in g.input:
        print(f"    {vi.name:28s} {tinfo(vi)}")
    print("  graph outputs:")
    for vi in g.output:
        print(f"    {vi.name:28s} {tinfo(vi)}")

    try:
        ort.register_execution_provider_library(EP, str(LIB))
    except Exception as e:
        if "already registered" not in str(e):
            raise

    # NON-TRIVIALITY, decided before the comparison is built. GQA is DIVERGENT and
    # therefore out of the ledger, so an ordinary session DECLINES it, the CPU computes
    # both arms, and the comparison reports a perfect match. That is `0.0 == 0.0`
    # wearing a green tick -- I hit it on the first run of this probe and for a moment
    # it looked like a fixed kernel.
    #
    # Force the claim through the documented escape hatch, then PROVE from ORT's own
    # trace that a Vulkan node executed. If it did not, refuse to report a comparison.
    GQA_KEY = ("com.microsoft::GroupQueryAttention/1+/f16,f16,f16,i32,i32,f16,f16"
               ">f16,f16,f16/metadata/runtime-extent/"
               "past_key+past_value+cos_cache+sin_cache")
    os.environ["ONNXRUNTIME_EP_VULKAN_CLAIM_UNPROVEN"] = GQA_KEY

    cpu = ort.InferenceSession(str(CASE), providers=["CPUExecutionProvider"])

    # Use the harness's OWN declared feed plan and seed. A GQA case has inputs that are
    # not free variables -- `total_sequence_length` must equal past + current and
    # `seqlens_k` must agree with the cache extent -- so an ad-hoc feed produces a
    # different model, not a harsher test. My first attempt did exactly that and ORT
    # rejected it (`seqlens_k[0] = 5 is out of range [0, 5)`), which is the harness
    # being right and me being wrong.
    sys.path.insert(0, str(REPO / "rust" / "tools"))
    import ledger_case_models  # noqa: E402
    plan = ledger_case_models.feed_plan(CASE.stem) or {}
    symbolic = plan.get("symbolic_dims") or {}
    fixed = plan.get("fixed_inputs") or {}
    scale = float(plan.get("float_scale", 1.0))
    print(f"\n  harness feed_plan: symbolic={symbolic} fixed={list(fixed)} scale={scale}")

    rng = np.random.default_rng(20260801)   # gen_proof_ledger.py default seed
    feeds = {}
    for inp in cpu.get_inputs():
        shape = [
            d if isinstance(d, int) and d > 0
            else symbolic.get(d, 2) if isinstance(d, str) else 2
            for d in inp.shape
        ]
        if inp.name in fixed:
            f = fixed[inp.name]
            feeds[inp.name] = np.asarray(
                f["values"], dtype=np.dtype(f["dtype"])).reshape(f["shape"])
            continue
        if "int32" in inp.type:
            feeds[inp.name] = np.zeros(shape, dtype=np.int32)
        elif "int64" in inp.type:
            feeds[inp.name] = np.zeros(shape, dtype=np.int64)
        else:
            dt = np.float16 if "float16" in inp.type else np.float32
            feeds[inp.name] = (rng.standard_normal(shape) * scale).astype(dt)
    print("\n  feeds:")
    for k, v in feeds.items():
        print(f"    {k:28s} {v.dtype} {v.shape}"
              + (f"  = {v.ravel()[:4]}" if v.dtype in (np.int32, np.int64) else ""))

    co = cpu.run(None, feeds)

    so = ort.SessionOptions()
    so.enable_profiling = True
    so.profile_file_prefix = str(HERE / "gqa_div_prof")
    vk = ort.InferenceSession(str(CASE), sess_options=so,
                              providers=[EP, "CPUExecutionProvider"])
    vo = vk.run(None, feeds)
    prof_path = vk.end_profiling()
    ev = json.loads(pathlib.Path(prof_path).read_text(encoding="utf-8"))
    by_prov = {}
    for e in ev:
        if e.get("cat") == "Node" and e.get("args", {}).get("provider"):
            p = e["args"]["provider"]
            by_prov[p] = by_prov.get(p, 0) + 1
    pathlib.Path(prof_path).unlink(missing_ok=True)
    vk_execs = by_prov.get(EP, 0)
    print(f"\n  node executions by provider: {by_prov}")
    if vk_execs == 0:
        print("\n  REFUSING TO COMPARE: the Vulkan EP executed no node. Any agreement")
        print("  here would be the CPU EP agreeing with itself -- the vacuous case the")
        print("  escape hatch was set to avoid, and it did not take.")
        return 3
    print(f"  non-triviality: {vk_execs} execution(s) on {EP} -- the comparison is live")


    names = [o.name for o in cpu.get_outputs()]
    print("\n  per-output divergence:")
    per = []
    for idx, nm in enumerate(names):
        a = np.asarray(co[idx]).astype(np.float32)
        b = np.asarray(vo[idx]).astype(np.float32)
        d = np.abs(a - b)
        denom = np.maximum(np.abs(a), 1e-5)
        rel = d / denom
        per.append({
            "index": idx, "name": nm, "shape": list(a.shape),
            "max_abs": float(np.max(np.abs(a))),
            "max_abs_diff": float(np.max(d)),
            "worst_rel": float(np.max(rel)),
            "frac_differing": float(np.count_nonzero(d) / a.size),
            "vk_all_zero": bool(np.all(b == 0)),
        })
        print(f"    [{idx}] {nm:16s} {str(a.shape):22s} max_abs={np.max(np.abs(a)):8.4f} "
              f"max_abs_diff={np.max(d):9.5f}  worst_rel={np.max(rel):10.4f}  "
              f"frac_diff={np.count_nonzero(d) / a.size:.3f}"
              + ("   VK ALL ZERO" if np.all(b == 0) else ""))

    worst = max(per, key=lambda p: p["worst_rel"])
    print(f"\n  worst output: [{worst['index']}] {worst['name']} "
          f"worst_rel={worst['worst_rel']:.8f}")
    print(f"  ledger says : 16.72642029784887")
    close = abs(worst["worst_rel"] - 16.72642029784887) < 1e-6
    print(f"  reproduced to the digit: {close}")
    if not close:
        print("  NOT the same number. My feed differs from the harness's, so this is a")
        print("  different reading of the same defect -- localisation still valid, but")
        print("  the magnitude is not comparable to the ledger's.")

    # --- structure of the error --------------------------------------------
    i0 = worst["index"]
    a = np.asarray(co[i0]).astype(np.float32)
    b = np.asarray(vo[i0]).astype(np.float32)
    d = np.abs(a - b)
    print(f"\n  error structure for output[{i0}] shape {a.shape}:")
    struct = {}
    if a.ndim >= 2:
        for axis in range(a.ndim):
            other = tuple(x for x in range(a.ndim) if x != axis)
            per_ax = d.max(axis=other)
            nz = int(np.count_nonzero(per_ax))
            struct[f"axis{axis}_len"] = int(a.shape[axis])
            struct[f"axis{axis}_slices_with_error"] = nz
            head = np.array2string(per_ax[:8], precision=4, max_line_width=200)
            print(f"    axis {axis} (len {a.shape[axis]:4d}): "
                  f"{nz}/{a.shape[axis]} slices carry error   max_by_slice[:8]={head}")
    OUT.write_text(json.dumps({
        "dll_sha256": sha256(LIB),
        "case": CASE.name,
        "node_inputs": list(node.input),
        "node_outputs": list(node.output),
        "attributes": attrs,
        "feeds": {k: {"dtype": str(v.dtype), "shape": list(v.shape)}
                  for k, v in feeds.items()},
        "per_output": per,
        "worst": worst,
        "ledger_worst_rel": 16.72642029784887,
        "reproduced_to_the_digit": close,
        "error_structure": struct,
    }, indent=2), encoding="utf-8")
    print(f"\n  record: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

