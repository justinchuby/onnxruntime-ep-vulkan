"""Does the artifact say which GEMV kernel produced the reading? — R10 falsifier.

`ONNXRUNTIME_EP_VULKAN_GEMV_PACKED` selects a different kernel. It does **not** change the shader
stem: it moves specialization constant 5 of `q_gemv.comp`, and `vk/pipeline.rs` keys the pipeline
cache on `(shader_stem, spec_constants)`. So the two settings are two different pipelines wearing
one stem, and until now every kernel reading this project holds — the amplification result, the
packed-loads work, the `q_gemv` figures — was silent about which of them produced it.

**A reading whose subject is unidentified is not a reading of that subject.**

`counters.rs` now emits `pipeline_variants` (the resolved `(stem, spec_constants)` pairs the run
actually built) and `gemv_packed_spec_constant` (typed: `"0"`/`"1"`/`"UNOBSERVABLE"`/`"MIXED"`/
`"UNRECORDED"`). This probe is the falsifier for that emission.

**Predictions, written before the first run.**

1. **A1 (flag unset, 4-bit/block-32)** → `"1"`. The blob is 16 bytes, so the shape supports packing
   and the default takes it.
2. **A2 (`GEMV_PACKED=0`, same graph)** → `"0"`. Differs from A1 in the flag alone.
3. **A3 (`GEMV_PACKED=1`, same graph)** → `"1"`, identical to A1. Arming a switch that was already
   on must not change the recording; if it does, the field is reading the env var.
4. **B (flag unset, 4-bit/block-16)** → `"0"`. **This is the arm that matters.** The blob is 8
   bytes, so the packed path is off *by shape*, with the environment untouched. A recording that
   moves here is reading what the pipeline was built with; a recording that stays at `"1"` is
   reading the env var, which is a request and not an identity — the same distinction that caught
   `DEVICE=0` running on device 1.
5. **C (elementwise graph, `GEMV_PACKED=1`)** → `"UNOBSERVABLE"`, never `"0"`. No `MatMulNBits`, so
   no GEMV pipeline exists to carry the constant. R12: a count whose event cannot occur in its
   frame is not a zero. This is the census lane's own graph shape, and a `"0"` here would have
   quietly told Link's census that the unpacked kernel ran.
6. **`shaders_dispatched` is identical across A1 and A2.** That is the whole finding: the field the
   project already had cannot tell these two runs apart, and the new one can.

No clock. Every reading here is a token or a set of strings.

Output: `bench/results/gemv_kernel_identity-dev{N}.json`
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
RESULTS = REPO / "bench" / "results"
ENV_PACKED = "ONNXRUNTIME_EP_VULKAN_GEMV_PACKED"

#: (name, env value for the switch or None to leave it unset, graph spec, predicted token)
CASES = [
    ("A1_unset_block32", None, {"kind": "matmulnbits", "bits": 4, "block_size": 32}, "1"),
    ("A2_forced_off", "0", {"kind": "matmulnbits", "bits": 4, "block_size": 32}, "0"),
    ("A3_forced_on", "1", {"kind": "matmulnbits", "bits": 4, "block_size": 32}, "1"),
    ("B_unset_block16", None, {"kind": "matmulnbits", "bits": 4, "block_size": 16}, "0"),
    ("C_elementwise_armed", "1", {"kind": "elementwise"}, "UNOBSERVABLE"),
]


def run_child(spec: dict) -> None:
    import numpy as np
    import onnxruntime as ort

    sys.path.insert(0, str(REPO))
    # `_models` imports `_verdict` as a top-level module, so its own directory has to be on the
    # path as well as the repo root.
    sys.path.insert(0, str(REPO / "tests" / "ops"))
    import tests.ops._models as models  # noqa: PLC0415

    if spec["kind"] == "matmulnbits":
        model_bytes, feeds = models.make_matmulnbits_model(
            K=1024,
            N=512,
            bits=spec["bits"],
            block_size=spec["block_size"],
            with_zero_points=False,
            rows=1,
        )
    else:
        # A single-node elementwise chain — the census lane's own graph shape, and the frame in
        # which the GEMV constant cannot be resolved at all.
        import onnx_ir as ir  # noqa: PLC0415

        dt = ir.DataType.FLOAT
        model_bytes = models.make_model(
            "Add",
            [models.tensor("a", dt, [4, 4]), models.tensor("b", dt, [4, 4])],
            [models.tensor("out", dt, [4, 4])],
        )
        rng = np.random.default_rng(0)
        feeds = {
            "a": rng.random((4, 4), dtype=np.float32),
            "b": rng.random((4, 4), dtype=np.float32),
        }

    ort.register_execution_provider_library(
        "VulkanExecutionProvider", os.environ["ONNXRUNTIME_VULKAN_EP_LIB"]
    )
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(
        model_bytes, so, providers=["VulkanExecutionProvider", "CPUExecutionProvider"]
    )
    outs = sess.run(None, feeds)
    print(f"[child] ran, output0 shape {np.asarray(outs[0]).shape}")


def main(argv: list[str]) -> int:
    if len(argv) > 2 and argv[1] == "--child":
        run_child(json.loads(argv[2]))
        return 0

    device = os.environ.get("ONNXRUNTIME_EP_VULKAN_DEVICE", "0")
    RESULTS.mkdir(parents=True, exist_ok=True)
    me = str(pathlib.Path(__file__).resolve())

    observations = []
    for name, flag, spec, predicted in CASES:
        counters = RESULTS / f"gemv_kernel_identity-{name}-dev{device}.counters.json"
        counters.unlink(missing_ok=True)
        env = dict(os.environ)
        env.pop(ENV_PACKED, None)
        if flag is not None:
            env[ENV_PACKED] = flag
        env["ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE"] = str(counters)
        proc = subprocess.run(
            [sys.executable, me, "--child", json.dumps(spec)],
            env=env,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=1800,
        )
        if proc.returncode != 0 or not counters.is_file():
            # R13: an instrument error is never a detection. Quote the text, never a count.
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-6:]
            observations.append(
                {
                    "case": name,
                    "env": flag,
                    "graph": spec,
                    "predicted": predicted,
                    "observed": "ERROR(instrument)",
                    "failure_text": "\n".join(tail) or f"no counters file at {counters}",
                }
            )
            continue
        doc = json.loads(counters.read_text(encoding="utf-8"))
        observations.append(
            {
                "case": name,
                "env": flag,
                "graph": spec,
                "predicted": predicted,
                "observed": doc.get("gemv_packed_spec_constant", "KEY-ABSENT"),
                "pipeline_variants": doc.get("pipeline_variants", "KEY-ABSENT"),
                "shaders_dispatched": doc.get("shaders_dispatched", "KEY-ABSENT"),
                "dispatches_executed": doc.get("dispatches_executed"),
            }
        )

    by_case = {o["case"]: o for o in observations}
    errors = [o for o in observations if o["observed"] == "ERROR(instrument)"]

    checks = []

    def check(name: str, ok: bool | None, detail: str) -> None:
        checks.append({"check": name, "result": ok, "detail": detail})

    a1, a2, a3 = by_case["A1_unset_block32"], by_case["A2_forced_off"], by_case["A3_forced_on"]
    b, c = by_case["B_unset_block16"], by_case["C_elementwise_armed"]

    check(
        "the switch changes the recorded value",
        a1["observed"] != a2["observed"] and "ERROR(instrument)" not in (a1["observed"],
                                                                        a2["observed"]),
        f"A1={a1['observed']!r} vs A2={a2['observed']!r} — if these were equal the flag would be "
        f"inert or the emission would not be reading it",
    )
    check(
        "arming a switch that was already on changes nothing",
        a1["observed"] == a3["observed"],
        f"A1={a1['observed']!r} vs A3={a3['observed']!r}",
    )
    check(
        "the record follows the resolved constant, not the environment",
        b["observed"] == "0" and a1["observed"] == "1",
        f"the environment is untouched in both A1 and B; only the block size differs. "
        f"A1={a1['observed']!r} B={b['observed']!r}. A record that read the env var would give "
        f"the same token for both.",
    )
    check(
        "an unresolvable constant is UNOBSERVABLE, never 0",
        c["observed"] == "UNOBSERVABLE",
        f"C={c['observed']!r} with the switch armed at '1' on a graph with no MatMulNBits",
    )
    check(
        "shaders_dispatched cannot tell A1 from A2",
        a1.get("shaders_dispatched") == a2.get("shaders_dispatched"),
        f"A1={a1.get('shaders_dispatched')} A2={a2.get('shaders_dispatched')} — the field this "
        f"project already had is identical across the two kernels, which is why the new one exists",
    )
    check(
        "pipeline_variants can",
        a1.get("pipeline_variants") != a2.get("pipeline_variants"),
        f"A1={a1.get('pipeline_variants')} A2={a2.get('pipeline_variants')}",
    )

    predictions = [
        {"case": o["case"], "predicted": o["predicted"], "observed": o["observed"],
         "held": o["predicted"] == o["observed"]}
        for o in observations
    ]

    if errors:
        verdict = "ERROR(instrument)"
    elif all(c["result"] for c in checks) and all(p["held"] for p in predictions):
        verdict = "PASS"
    else:
        failed = [c["check"] for c in checks if not c["result"]]
        failed += [f"prediction {p['case']}" for p in predictions if not p["held"]]
        verdict = f"FAIL({'; '.join(failed)})"

    out = {
        "device_selector": device,
        "subject": "ONNXRUNTIME_EP_VULKAN_GEMV_PACKED -> q_gemv.comp specialization constant 5",
        "emission": "counters.rs::record_pipeline_variant, called from vk/session.rs at the "
        "PipelineKey construction site with the effective (shader, spec_constants) pair",
        "no_duration_quoted": "Every reading here is a token or a set of strings.",
        "verdict": verdict,
        "predictions": predictions,
        "checks": checks,
        "observations": observations,
    }
    (RESULTS / f"gemv_kernel_identity-dev{device}.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )

    for p in predictions:
        mark = "held" if p["held"] else "BROKEN"
        print(f"  {p['case']:22s} predicted {p['predicted']:14s} observed "
              f"{p['observed']:14s} {mark}")
    print()
    for ch in checks:
        print(f"  [{'ok' if ch['result'] else 'XX'}] {ch['check']}")
        if not ch["result"]:
            print(f"        {ch['detail']}")
    print()
    for e in errors:
        print(f"  {e['case']}: {e['failure_text']}")
    print(f"VERDICT: {verdict}")
    print(f"[gemv-identity] -> {RESULTS / f'gemv_kernel_identity-dev{device}.json'}")
    return 0 if verdict == "PASS" else (2 if errors else 1)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
