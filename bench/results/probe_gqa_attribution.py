"""Which provider ran each GroupQueryAttention node?

Criterion 10 reports `present.31.key` and `present.31.value` OUTSIDE_TOLERANCE while
layers 0..30 are all WITHIN. GQA is separately DIVERGENT at worst_rel 16.726.

Those two facts are only the same defect if the layer-31 GQA node ran on the Vulkan EP
AND the layer-0..30 GQA nodes did not. If every GQA node ran on Vulkan, a kernel defect
would have to explain why it is invisible in 31 of 32 instances -- which is a much
stranger claim and probably a false one.

So: attribute the GQA nodes, one by one, before touching a kernel. This reads the EP's
own claim log, not the output values, so it cannot be confounded by the thing it is
trying to explain.

Writes bench/results/gqa_attribution.json.
"""

import hashlib
import json
import os
import pathlib
import re
import sys

import numpy as np
import onnx
import onnxruntime as ort

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
MODEL_DIR = pathlib.Path(
    r"C:\Users\justinchu\.foundry\cache\models\Microsoft"
    r"\Phi-3.5-mini-instruct-cuda-gpu\cuda-int4-rtn-block-32"
)
ONNX_FILE = MODEL_DIR / "phi-3.5-mini-instruct-cuda-int4-rtn-block-32.onnx"
LIB = REPO / "rust" / "target" / "release" / "onnxruntime_vulkan_ep.dll"
CLAIM_LOG = HERE / "gqa_claim_log.jsonl"
OUT = HERE / "gqa_attribution.json"

EP = "VulkanExecutionProvider"


def sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def layer_of(name: str):
    m = re.search(r"layers\.(\d+)", name)
    return int(m.group(1)) if m else None


def main() -> int:
    if not ONNX_FILE.exists():
        print(f"model missing: {ONNX_FILE}")
        return 2

    print(f"  DLL   {sha256(LIB)[:16]}")

    # --- the graph's own GQA nodes, before ORT sees anything ---------------
    model = onnx.load(str(ONNX_FILE), load_external_data=False)
    gqa_nodes = [n for n in model.graph.node if n.op_type == "GroupQueryAttention"]
    gqa_by_layer = {}
    for n in gqa_nodes:
        gqa_by_layer[layer_of(n.name)] = n.name
    print(f"  GroupQueryAttention nodes in graph: {len(gqa_nodes)}")

    # which graph output does each GQA node feed?
    out_names = [o.name for o in model.graph.output]
    gqa_feeding_output = {}
    for n in gqa_nodes:
        fed = [o for o in n.output if o in out_names]
        if fed:
            gqa_feeding_output[layer_of(n.name)] = fed

    if CLAIM_LOG.exists():
        CLAIM_LOG.unlink()
    os.environ["ONNXRUNTIME_EP_VULKAN_CLAIM_LOG"] = str(CLAIM_LOG)
    os.environ.setdefault("ONNXRUNTIME_EP_VULKAN_DEVICE", "0")

    try:
        ort.register_execution_provider_library(EP, str(LIB))
    except Exception as e:
        if "already registered" not in str(e):
            raise

    so = ort.SessionOptions()
    so.enable_profiling = True
    so.profile_file_prefix = str(HERE / "gqa_attr_prof")
    sess = ort.InferenceSession(
        str(ONNX_FILE), sess_options=so, providers=[EP, "CPUExecutionProvider"]
    )

    # minimal single-token-ish feed
    feeds = {}
    for i in sess.get_inputs():
        shp = [1 if isinstance(d, str) or d is None else d for d in i.shape]
        if i.name == "input_ids":
            shp = [1, 4]
            feeds[i.name] = np.array([[1, 2, 3, 4]], dtype=np.int64)
            continue
        if i.name == "attention_mask":
            feeds[i.name] = np.ones((1, 4), dtype=np.int64)
            continue
        dt = np.float16 if "float16" in i.type else np.float32
        if "int64" in i.type:
            dt = np.int64
        # past_key_values.*: zero-length history
        shp = [0 if (isinstance(d, str) and "past" in str(d)) else d for d in shp]
        shp = [1 if isinstance(d, str) or d is None else d for d in shp]
        if "past_key_values" in i.name:
            shp[2] = 0
        feeds[i.name] = np.zeros(shp, dtype=dt)

    sess.run(None, feeds)
    prof = sess.end_profiling()

    # --- read the EP's own claim log --------------------------------------
    claims = []
    if CLAIM_LOG.exists():
        for line in CLAIM_LOG.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    claims.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    print(f"  claim-log records: {len(claims)}")

    gqa_claims = {}
    for rec in claims:
        nm = rec.get("node") or rec.get("name") or ""
        op = rec.get("op") or rec.get("op_type") or ""
        if "GroupQueryAttention" in op or "GroupQueryAttention" in nm or "attn/GroupQuery" in nm:
            L = layer_of(nm)
            gqa_claims[L] = {
                "node": nm,
                "claimed": rec.get("claimed", rec.get("decision")),
                "reason": rec.get("reason") or rec.get("decline") or rec.get("form"),
            }

    claimed_layers = sorted(L for L, v in gqa_claims.items()
                            if v.get("claimed") in (True, "claimed", "CLAIM"))
    declined_layers = sorted(L for L, v in gqa_claims.items() if L not in claimed_layers)

    print()
    print(f"  GQA layers with a claim record : {len(gqa_claims)}")
    print(f"  GQA layers CLAIMED by Vulkan   : {len(claimed_layers)}")
    if claimed_layers:
        print(f"    {claimed_layers}")
    print(f"  GQA layers DECLINED            : {len(declined_layers)}")
    if declined_layers:
        print(f"    {declined_layers}")

    print()
    if len(claimed_layers) == len(gqa_nodes):
        print("  READING: every GQA node claimed. Divergence localised to layer 31")
        print("           is NOT explained by provider attribution.")
    elif claimed_layers == [31]:
        print("  READING: ONLY layer 31 claimed. The divergence tracks the provider,")
        print("           not the kernel -- and GQA's 16.726 is the same defect.")
    elif not claimed_layers:
        print("  READING: NO GQA node claimed by Vulkan. present.31.* was produced by")
        print("           the CPU EP, so the divergence cannot be a Vulkan GQA defect.")
    else:
        print("  READING: partial claim. The set that diverges must be compared against")
        print("           the set that was claimed -- they are printed above.")

    rec = {
        "dll_sha256": sha256(LIB),
        "gqa_nodes_in_graph": len(gqa_nodes),
        "gqa_by_layer": {str(k): v for k, v in sorted(gqa_by_layer.items())},
        "gqa_feeding_graph_output": {str(k): v for k, v in sorted(gqa_feeding_output.items())},
        "claim_log_records": len(claims),
        "gqa_claims": {str(k): v for k, v in sorted(gqa_claims.items())},
        "claimed_layers": claimed_layers,
        "declined_layers": declined_layers,
        "profile": prof,
    }
    OUT.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"\n  record: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
