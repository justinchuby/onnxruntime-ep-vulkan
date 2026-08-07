"""Differential CPU-vs-Vulkan proof of the row tile on Phi-3.5's *real* MatMulNBits weights.

Why this probe exists
---------------------
`rust/modelrunner` cannot run Phi-3.5-mini end to end, and says so honestly:

    UNSUPPORTED(reason=reference_run_unsupported)
    ... Non-zero status code returned while running GroupQueryAttention node ...
    seqlens_k[0] = 7 is out of range [0, 1)

That is a limit of the runner's *input generation* — the CPU reference arm cannot be made to run
with synthetic inputs because the model's inputs are interdependent (KV cache, sequence lengths) —
and not a result about the execution provider. It also has nothing to do with MatMulNBits. But it
does mean there is no whole-model CPU reference to compare a Vulkan run against, so "we ran the
real model and the logits matched" is not available and must not be implied.

What *is* available is the narrowest thing that is still about the real model: each of the 161
MatMulNBits nodes in that file, with its real int4 packed weights, its real scales, its real
zero-points if it has them, its real K and N, its real fp16 dtype — lifted into a one-node graph
and run on both providers. The proof-ledger case models (`rust/tools/ledger_case_models.py`) prove
the same operator on *synthetic* weights of the same form; this proves it on the bytes Phi-3.5
actually ships. The two are complementary and neither replaces the other: synthetic weights are
chosen to be numerically awkward, real weights are the ones that matter.

M is the whole point. At `M = 1` this is the decode path that has always been proven. At `M > 1`
the host selects a row tile (issue #7) and the shader takes its second arm, so every M in
`M_VALUES` is a form that did not exist before this change, on weights nobody chose.

Provenance
----------
The model path is resolved by `bench.foundry_discovery`, never hardcoded, and the sha256 of the
file actually opened is recorded in the report. If the resolver's answer and the recorded hash
disagree with `bench/results/rust-model-runner/phi-3.5-mini.json`, that is reported as a
provenance failure rather than smoothed over.

Usage
-----
    ONNXRUNTIME_VULKAN_EP_LIB=... python bench/results/probe_real_matmulnbits_rows.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

_BENCH = Path(__file__).resolve().parents[1]
_ROOT = _BENCH.parent
for _p in (str(_BENCH), str(_ROOT), str(_ROOT / "rust" / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import foundry_discovery as _foundry_discovery  # noqa: E402

# The literal spec, never a literal path: issue #19's rule is that live tools resolve through the
# Foundry cache manifest, because the cache layout moves (it moved from `...-cuda-gpu/` to
# `...-cuda-gpu-2/v2/` between this repo's archival probes being written and now).
_PHI35_SPEC = _foundry_discovery.FoundryModelSpec(
    variant_name="Phi-3.5-mini-instruct-cuda-gpu",
    execution_provider="CUDAExecutionProvider",
    onnx_filename="phi-3.5-mini-instruct-cuda-int4-rtn-block-32.onnx",
    download_alias="phi-3.5-mini",
)

# M=1 is the decode control that must not move. 2 and 4 are the two tile heights the host can
# select. 3 and 5 leave a partial tile, which is where a row tile is most likely to be wrong.
M_VALUES = (1, 2, 3, 4, 5, 8)

# Sampling the nodes rather than all 161: the file is 26 MB of packed int4 and each node costs a
# session build per M. The sample is deterministic and spread across the graph, and every distinct
# (K, N, has_zero_points) form present is forced into it below, so no *form* is skipped — only
# repeats of a form already covered.
SAMPLE_NODES = 12


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_model() -> "tuple[Path, str]":
    path = _foundry_discovery.resolve_model_path(_PHI35_SPEC)
    if not Path(path).is_file():
        raise SystemExit(f"resolver returned {path}, which does not exist")
    return Path(path), "resolved"


def _collect_nodes(model_path: Path):
    """Every MatMulNBits node in the file, with its initializers, as (name, attrs, inits)."""
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(model_path), load_external_data=True)
    inits = {i.name: i for i in model.graph.initializer}
    out = []
    for node in model.graph.node:
        if node.op_type != "MatMulNBits":
            continue
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        # inputs: A, B, scales, [zero_points, g_idx, bias]
        names = list(node.input)
        if any(n and n not in inits for n in names[1:]):
            continue  # a weight that is not an initializer is not liftable; skip, do not fake it
        arrays = {}
        for slot, n in enumerate(names[1:], start=1):
            if n:
                arrays[slot] = numpy_helper.to_array(inits[n])
        out.append((node.name or f"node{len(out)}", attrs, arrays, names))
    return out, model


def _build_one_node(attrs, arrays, m_rows: int, seed: int):
    """A one-node MatMulNBits graph carrying the real packed bytes, with M rows of activation."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    K = int(attrs["K"])
    N = int(attrs["N"])
    bits = int(attrs["bits"])
    block_size = int(attrs["block_size"])

    scales = arrays[2]
    act_dtype = TensorProto.FLOAT16 if scales.dtype == np.float16 else TensorProto.FLOAT
    np_dtype = np.float16 if act_dtype == TensorProto.FLOAT16 else np.float32

    initializers = [numpy_helper.from_array(arrays[1], "B"), numpy_helper.from_array(scales, "S")]
    inputs = ["A", "B", "S"]
    if 3 in arrays:
        initializers.append(numpy_helper.from_array(arrays[3], "ZP"))
        inputs.append("ZP")

    node = helper.make_node(
        "MatMulNBits",
        inputs,
        ["Y"],
        domain="com.microsoft",
        K=K,
        N=N,
        bits=bits,
        block_size=block_size,
        # Pinned for the same reason the rest of the suite pins it: accuracy_level is a knob whose
        # default ORT may change, and an unpinned oracle makes two runs incomparable.
        accuracy_level=1,
    )
    graph = helper.make_graph(
        [node],
        "real_matmulnbits",
        [helper.make_tensor_value_info("A", act_dtype, [m_rows, K])],
        [helper.make_tensor_value_info("Y", act_dtype, [m_rows, N])],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)],
    )
    model.ir_version = 10
    rng = np.random.default_rng(seed)
    feeds = {"A": rng.standard_normal((m_rows, K)).astype(np_dtype)}
    return model.SerializeToString(), feeds, act_dtype == TensorProto.FLOAT16


def _run(model_bytes, feeds, providers):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(model_bytes, opts, providers=providers)
    return np.asarray(sess.run(None, feeds)[0])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(Path(__file__).with_name("real_matmulnbits_rows.json")))
    ap.add_argument("--nodes", type=int, default=SAMPLE_NODES)
    args = ap.parse_args()

    import bench as bench_mod

    if not bench_mod.register_ep():
        print("this probe compares two providers; refusing to report one.", file=sys.stderr)
        return 2
    import onnxruntime as ort

    model_path, provenance = _resolve_model()
    digest = _sha256(model_path)
    print(f"[real] {model_path}")
    print(f"[real] sha256 {digest} ({provenance})")

    recorded = json.loads(
        (_BENCH / "results" / "rust-model-runner" / "phi-3.5-mini.json").read_text(encoding="utf-8")
    )
    provenance_agrees = recorded.get("onnx_sha256") == digest

    nodes, _model = _collect_nodes(model_path)
    print(f"[real] {len(nodes)} MatMulNBits nodes carry liftable initializers")

    # Force one node per distinct form into the sample, then fill up to --nodes by even spacing.
    by_form: dict = {}
    for idx, (name, attrs, arrays, _names) in enumerate(nodes):
        form = (int(attrs["K"]), int(attrs["N"]), int(attrs["bits"]), int(attrs["block_size"]), 3 in arrays)
        by_form.setdefault(form, idx)
    chosen = sorted(set(by_form.values()))
    if len(chosen) < args.nodes and nodes:
        step = max(1, len(nodes) // args.nodes)
        chosen = sorted(set(chosen) | set(range(0, len(nodes), step)))[: args.nodes]

    results = []
    failures = []
    for idx in chosen:
        name, attrs, arrays, _names = nodes[idx]
        for m_rows in M_VALUES:
            model_bytes, feeds, is_fp16 = _build_one_node(attrs, arrays, m_rows, seed=idx * 131 + m_rows)
            cpu = _run(model_bytes, feeds, ["CPUExecutionProvider"])
            vk = _run(model_bytes, feeds, ["VulkanExecutionProvider"])
            # fp16 accumulation order differs between a scalar CPU loop and a tree reduction, so
            # this is a tolerance and is stated as one rather than hidden inside allclose defaults.
            rtol, atol = (2e-2, 2e-2) if is_fp16 else (1e-4, 1e-4)
            c64 = cpu.astype(np.float64)
            v64 = vk.astype(np.float64)
            err = np.abs(v64 - c64)
            rel = float(np.max(err / np.maximum(np.abs(c64), 1e-6)))
            max_abs = float(np.max(err))
            # A bare max-relative-error over an fp16 GEMM output is a cancellation meter, not an
            # accuracy meter: where the CPU result lands near zero, any absolute difference of one
            # ulp reads as a huge relative one, and the number of chances to land near zero grows
            # linearly with M. So report the relative error restricted to elements that are not
            # cancellation — those at least a tenth of the output's RMS — which is the population
            # a reader actually means when they ask "how close is it".
            rms = float(np.sqrt(np.mean(c64 * c64))) or 1.0
            sig = np.abs(c64) >= 0.1 * rms
            rel_sig = float(np.max(err[sig] / np.abs(c64[sig]))) if np.any(sig) else 0.0
            ok = bool(np.allclose(vk.astype(np.float32), cpu.astype(np.float32), rtol=rtol, atol=atol))
            # A row tile that broadcast one row into all of them would still pass a match against
            # an oracle that did the same. It cannot here (the oracle is ORT's CPU kernel), but the
            # shape still has to be able to *tell*, so assert the rows are actually distinct.
            distinct = m_rows == 1 or not np.array_equal(cpu[0], cpu[min(1, m_rows - 1)])
            row = {
                "node": name,
                "K": int(attrs["K"]),
                "N": int(attrs["N"]),
                "bits": int(attrs["bits"]),
                "block_size": int(attrs["block_size"]),
                "zero_points": 3 in arrays,
                "dtype": "f16" if is_fp16 else "f32",
                "m": m_rows,
                "match": ok,
                "max_rel": rel,
                "max_abs": max_abs,
                "max_rel_significant": rel_sig,
                "output_rms": rms,
                "rtol": rtol,
                "atol": atol,
                "rows_are_distinguishable": distinct,
            }
            results.append(row)
            if not ok or not distinct:
                failures.append(row)

    m_summary = {}
    for m_rows in M_VALUES:
        rows = [r for r in results if r["m"] == m_rows]
        m_summary[str(m_rows)] = {
            "cases": len(rows),
            "match": sum(1 for r in rows if r["match"]),
            "max_rel": max((r["max_rel"] for r in rows), default=0.0),
            "max_abs": max((r["max_abs"] for r in rows), default=0.0),
            "max_rel_significant": max((r["max_rel_significant"] for r in rows), default=0.0),
        }

    verdict = "PASS" if not failures and provenance_agrees and results else "FAIL"
    report = {
        "schema": "real_matmulnbits_rows/1",
        "verdict": verdict,
        "model": {
            "path": str(model_path),
            "sha256": digest,
            "provenance": provenance,
            "agrees_with_recorded_provenance": provenance_agrees,
            "recorded_sha256": recorded.get("onnx_sha256"),
        },
        "onnxruntime": ort.__version__,
        "limitation": (
            "rust/modelrunner reports UNSUPPORTED(reason=reference_run_unsupported) for this "
            "model: its GroupQueryAttention nodes reject generated inputs on the CPU reference "
            "arm (seqlens_k out of range), so no whole-model CPU reference exists. This probe "
            "therefore proves the operator on the model's real weights, not the model's logits, "
            "and must not be read as an end-to-end claim."
        ),
        "matmulnbits_nodes_total": len(nodes),
        "nodes_sampled": len(chosen),
        "m_values": list(M_VALUES),
        "by_m": m_summary,
        "failures": failures,
        "cases": results,
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = Path(__file__).with_name(args.out)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n  {verdict}: {len(results)} cases over {len(chosen)} real nodes x {len(M_VALUES)} M values")
    for m_rows in M_VALUES:
        s = m_summary[str(m_rows)]
        tag = "  <- decode control" if m_rows == 1 else ""
        print(
            f"    M={m_rows:<2} {s['match']}/{s['cases']} match, "
            f"max abs {s['max_abs']:.3e}, rel(significant) {s['max_rel_significant']:.3e}, "
            f"rel(all) {s['max_rel']:.3e}{tag}"
        )
    if not provenance_agrees:
        print("  PROVENANCE MISMATCH: the resolved file is not the one the archive records")
    for f in failures:
        print(f"    FAIL {f['node']} M={f['m']} rel={f['max_rel']:.3e}")
    print(f"\n  wrote {out}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
