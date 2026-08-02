"""Is the past->present copy exact?

`present = concat(past, new)`. The first `past_len` positions of `present_key` are a
straight copy of the fed `past_key` -- no arithmetic, no rotary, no accumulation. So
they are the one part of this kernel whose correct value is known *without* reference
to any other implementation: it is an input tensor.

That makes them a falsifier that does not depend on the CPU EP being right, and it
separates the two failure modes the divergence probe cannot tell apart:

  * copied region wrong  -> the copy/indexing is wrong (or nothing copies)
  * copied region exact, appended slot wrong -> RoPE or the QKV unpack is wrong

Refuses to report if the EP did not execute the node (see
`probe_gqa_divergence.py`: with GQA out of the ledger the CPU computes both arms and
every comparison reads 0.0).
"""

import json
import os
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "rust" / "tools"))

import onnxruntime as ort  # noqa: E402

import ledger_case_models  # noqa: E402

CASE = REPO / "evidence" / "cases" / "group_query_attention_f16.onnx"
LIB = REPO / "rust" / "target" / "release" / "onnxruntime_vulkan_ep.dll"
GQA_KEY = (
    "com.microsoft::GroupQueryAttention/1+/f16,f16,f16,i32,i32,f16,f16>f16,f16,f16"
    "/metadata/runtime-extent/past_key+past_value+cos_cache+sin_cache"
)
SEED = 20260801


def build_feeds():
    plan = ledger_case_models.feed_plan(CASE.stem) or {}
    symbolic = plan.get("symbolic_dims") or {}
    fixed = plan.get("fixed_inputs") or {}
    scale = float(plan.get("float_scale", 1.0))
    probe = ort.InferenceSession(str(CASE), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(SEED)
    feeds = {}
    for inp in probe.get_inputs():
        shape = [
            d
            if isinstance(d, int) and d > 0
            else symbolic.get(d, 2)
            if isinstance(d, str)
            else 2
            for d in inp.shape
        ]
        if inp.name in fixed:
            f = fixed[inp.name]
            feeds[inp.name] = np.asarray(f["values"], dtype=np.dtype(f["dtype"])).reshape(
                f["shape"]
            )
            continue
        if "int32" in inp.type:
            feeds[inp.name] = np.zeros(shape, dtype=np.int32)
        elif "int64" in inp.type:
            feeds[inp.name] = np.zeros(shape, dtype=np.int64)
        else:
            dt = np.float16 if "float16" in inp.type else np.float32
            feeds[inp.name] = (rng.standard_normal(shape) * scale).astype(dt)
    return feeds


def run(vulkan: bool, feeds):
    so = ort.SessionOptions()
    so.enable_profiling = vulkan
    if vulkan:
        ort.register_execution_provider_library("VulkanExecutionProvider", str(LIB))
        os.environ["ONNXRUNTIME_EP_VULKAN_CLAIM_UNPROVEN"] = GQA_KEY
        providers = ["VulkanExecutionProvider", "CPUExecutionProvider"]
    else:
        os.environ.pop("ONNXRUNTIME_EP_VULKAN_CLAIM_UNPROVEN", None)
        providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(CASE), so, providers=providers)
    outs = sess.run(None, feeds)
    execs = 0
    if vulkan:
        prof = sess.end_profiling()
        for ev in json.load(open(prof)):
            if ev.get("cat") == "Node" and "Vulkan" in str(
                ev.get("args", {}).get("provider", "")
            ):
                execs += 1
        os.unlink(prof)
    return outs, execs


def main():
    feeds = build_feeds()
    past_k = feeds["past_key"].astype(np.float32)
    past_v = feeds["past_value"].astype(np.float32)
    past_len = int(feeds["seqlens_k"][0])

    cpu, _ = run(False, feeds)
    vk, execs = run(True, feeds)

    print(f"  vulkan node executions: {execs}")
    if execs == 0:
        print("  REFUSING: the EP did not execute the node; both arms are the CPU's.")
        return 2
    print(f"  past_len = {past_len}, present S = {vk[1].shape[2]}")

    rec = {"vulkan_node_executions": execs, "past_len": past_len, "regions": {}}
    verdict = []
    for name, idx, src in (("present_key", 1, past_k), ("present_value", 2, past_v)):
        v = vk[idx].astype(np.float32)
        c = cpu[idx].astype(np.float32)
        copied_vk = v[:, :, :past_len, :]
        copied_cpu = c[:, :, :past_len, :]
        appended_vk = v[:, :, past_len:, :]
        appended_cpu = c[:, :, past_len:, :]

        d_src = float(np.max(np.abs(copied_vk - src)))
        d_cpu_src = float(np.max(np.abs(copied_cpu - src)))
        d_app = float(np.max(np.abs(appended_vk - appended_cpu)))

        # Is the copied region a *permutation* of the source? A wrong stride relocates
        # values rather than corrupting them, and that reads very differently from a
        # value defect.
        same_multiset = bool(
            np.allclose(np.sort(copied_vk, axis=None), np.sort(src, axis=None), atol=0)
        )
        rec["regions"][name] = {
            "copied_vs_fed_past_max_abs": d_src,
            "cpu_copied_vs_fed_past_max_abs": d_cpu_src,
            "appended_vs_cpu_max_abs": d_app,
            "copied_region_is_a_permutation_of_the_source": same_multiset,
        }
        print(f"\n  {name}")
        print(f"    cpu  present[:{past_len}] vs fed past : max_abs {d_cpu_src:.6g}"
              "   (the contract, on the reference)")
        print(f"    vk   present[:{past_len}] vs fed past : max_abs {d_src:.6g}")
        print(f"    vk   present[{past_len}:] vs cpu      : max_abs {d_app:.6g}")
        print(f"    copied region is a permutation of the fed past: {same_multiset}")
        if d_src == 0.0:
            verdict.append(f"{name}: COPY EXACT")
        elif same_multiset:
            verdict.append(f"{name}: COPY MISPLACED (stride/index defect)")
        else:
            verdict.append(f"{name}: COPY CORRUPT")

    # Where in the copied region does it go wrong? A per-position max separates "one
    # slot" from "everything".
    v = vk[1].astype(np.float32)
    per_pos = np.max(np.abs(v[:, :, :past_len, :] - past_k), axis=(0, 1, 3))
    print(f"\n  present_key copy error by position: "
          f"{np.array2string(per_pos, precision=4)}")
    rec["present_key_copy_error_by_position"] = [float(x) for x in per_pos]

    # And is any position of vk present_key equal to some *other* position of the past?
    # That names the stride defect outright.
    match = {}
    for t in range(v.shape[2]):
        for u in range(past_len):
            if np.max(np.abs(v[:, :, t, :] - past_k[:, :, u, :])) == 0.0:
                match[t] = u
    print(f"  vk present position -> identical fed past position: {match or '{}'}")
    rec["vk_present_position_equals_past_position"] = {
        str(k): val for k, val in match.items()
    }

    rec["verdict"] = verdict
    print("\n  " + " | ".join(verdict))
    out = pathlib.Path(__file__).with_name("gqa_present_copy.json")
    out.write_text(json.dumps(rec, indent=2))
    print(f"  record: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
