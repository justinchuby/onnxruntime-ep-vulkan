"""§8.9.20 falsifier: is the runtime specialisation outside both build-time digests, and does the
dispatch-time witness see it?

THE CLAIM UNDER TEST
--------------------
`shader_digest` hashes SPIR-V and `source_digest` hashes the source closure. Both are fixed when
the build finishes. What actually runs is a *pipeline* — `(SPIR-V, specialisation values, layout)`
— and `vk/pipeline.rs` keys its cache on `(shader_stem, spec_constants)`. So two runs that resolve
`q_gemv.comp`'s packed-load constant differently build **different kernels from identical bytes**,
and every witness the proof ledger records agrees across them.

That is the residual §7.22 named and did not close. This probe is its falsifier, and it has two
sides, both of which have to hold or the section is wrong about something:

* the **hole** — the two build-time digests must be byte-identical across a specialisation change.
  If they moved, there was no hole and §8.9.20 is unnecessary.
* the **witness** — `shaders_dispatched_spec_digest` must move across the same change, and must
  *not* move when nothing changed. A digest that always moves is a clock, not an instrument.

PREDICTIONS, WRITTEN BEFORE THE FIRST RUN
-----------------------------------------
1. A1 (flag unset, block 32) and A2 (`GEMV_PACKED=0`, same graph) report the **same**
   `shaders_dispatched_digest` and the **same** `shaders_dispatched_source_digest`.
2. A1 and A2 report **different** `shaders_dispatched_spec_digest`.
3. A1 and A3 (`GEMV_PACKED=1`, i.e. the value A1 already resolves to) report the **same**
   `shaders_dispatched_spec_digest`. Arming a switch that was already on must not move a digest.
4. The elementwise arm — which claims real forms off the shipped ledger and dispatches them —
   reports a non-empty `specialisation_unrecorded_forms`. Every entry in the shipped ledger was
   written before this field existed, so the dispatch-time audit must find them and say so. An
   empty list here means the audit never ran, which is the failure mode §7.22 warned about: a
   field no predicate reads.
5. No arm reports a `specialisation_delta_forms` row, because no shipped entry records a
   specialisation to differ from. This is the honest zero, and prediction 4 is what makes it
   readable as one.

No clock. Every reading is a digest string or a list.

Output: `bench/results/specialisation_witness-dev{N}.json`
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

#: (name, env value for the switch or None to leave it unset, graph spec)
CASES = [
    ("A1_unset_block32", None, {"kind": "matmulnbits", "bits": 4, "block_size": 32}),
    ("A2_forced_off", "0", {"kind": "matmulnbits", "bits": 4, "block_size": 32}),
    ("A3_forced_on", "1", {"kind": "matmulnbits", "bits": 4, "block_size": 32}),
    ("D_elementwise", None, {"kind": "elementwise"}),
]


def run_child(spec: dict) -> None:
    import numpy as np
    import onnxruntime as ort

    sys.path.insert(0, str(REPO))
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
    for name, flag, spec in CASES:
        counters = RESULTS / f"specialisation_witness-{name}-dev{device}.counters.json"
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
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-6:]
            observations.append(
                {
                    "case": name,
                    "env": flag,
                    "graph": spec,
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
                "observed": "ok",
                "shader_digest": doc.get("shaders_dispatched_digest", "KEY-ABSENT"),
                "source_digest": doc.get("shaders_dispatched_source_digest", "KEY-ABSENT"),
                "spec_digest": doc.get("shaders_dispatched_spec_digest", "KEY-ABSENT"),
                "pipeline_variants": doc.get("pipeline_variants", "KEY-ABSENT"),
                "specialisation_delta_forms": doc.get("specialisation_delta_forms", "KEY-ABSENT"),
                "specialisation_unrecorded_forms": doc.get(
                    "specialisation_unrecorded_forms", "KEY-ABSENT"
                ),
                "ledger_specialisation_unrecorded_entries": doc.get(
                    "ledger_specialisation_unrecorded_entries", "KEY-ABSENT"
                ),
                "claimed_nodes": doc.get("claimed_nodes"),
                "dispatches_executed": doc.get("dispatches_executed"),
            }
        )

    by_case = {o["case"]: o for o in observations}
    errors = [o for o in observations if o["observed"] == "ERROR(instrument)"]

    checks = []

    def check(name: str, ok: bool | None, detail: str) -> None:
        checks.append({"check": name, "result": bool(ok), "detail": detail})

    if not errors:
        a1, a2, a3 = by_case["A1_unset_block32"], by_case["A2_forced_off"], by_case["A3_forced_on"]
        d = by_case["D_elementwise"]
        check(
            "the SPIR-V digest is blind to the specialisation",
            a1["shader_digest"] == a2["shader_digest"],
            f"A1={a1['shader_digest']} A2={a2['shader_digest']} — if these differed there would "
            f"be no hole and §8.9.20 would be unnecessary",
        )
        check(
            "the source digest is blind to it too",
            a1["source_digest"] == a2["source_digest"],
            f"A1={a1['source_digest']} A2={a2['source_digest']}",
        )
        check(
            "the specialisation digest is not blind to it",
            a1["spec_digest"] != a2["spec_digest"]
            and "PARTIAL" not in str(a1["spec_digest"])
            and a1["spec_digest"] != "NONE-DISPATCHED",
            f"A1={a1['spec_digest']} A2={a2['spec_digest']} — the positive state: two pipelines "
            f"built from identical bytes, told apart",
        )
        check(
            "and it does not move when nothing moved",
            a1["spec_digest"] == a3["spec_digest"],
            f"A1={a1['spec_digest']} A3={a3['spec_digest']} — arming a switch that was already "
            f"on; a digest that always moves is a clock, not an instrument",
        )
        check(
            "the dispatch-time audit ran against the shipped ledger",
            isinstance(d["specialisation_unrecorded_forms"], list)
            and len(d["specialisation_unrecorded_forms"]) > 0,
            f"D claimed {d['claimed_nodes']} node(s) and dispatched "
            f"{d['dispatches_executed']}; unrecorded forms="
            f"{d['specialisation_unrecorded_forms']!r}. Empty means the audit never ran, which "
            f"is the field-nobody-reads failure this section exists to avoid",
        )
        check(
            "no delta is claimed where no entry records a specialisation",
            all(
                o.get("specialisation_delta_forms") == []
                for o in observations
                if o["observed"] == "ok"
            ),
            "every shipped entry is SPEC-UNRECORDED, so a delta row here would be invented",
        )

    if errors:
        verdict = "ERROR(instrument)"
    elif all(c["result"] for c in checks):
        verdict = "PASS"
    else:
        verdict = f"FAIL({'; '.join(c['check'] for c in checks if not c['result'])})"

    out = {
        "device_selector": device,
        "subject": "§8.9.20 dispatch-time frame witness — shaders_dispatched_spec_digest",
        "emission": "counters.rs::specialisation_digest_for over record_pipeline_variant's set; "
        "audited from vk/session.rs at pipeline creation via "
        "registry::audit_dispatch_specialisation",
        "no_duration_quoted": "Every reading here is a digest string or a list.",
        "verdict": verdict,
        "checks": checks,
        "observations": observations,
    }
    (RESULTS / f"specialisation_witness-dev{device}.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )

    for o in observations:
        print(
            f"  {o['case']:20s} spirv={str(o.get('shader_digest'))[:16]:16s} "
            f"source={str(o.get('source_digest'))[:16]:16s} "
            f"spec={str(o.get('spec_digest'))[:16]:16s}"
        )
    print()
    for ch in checks:
        print(f"  [{'ok' if ch['result'] else 'XX'}] {ch['check']}")
        if not ch["result"]:
            print(f"        {ch['detail']}")
    for e in errors:
        print(f"  {e['case']}: {e['failure_text']}")
    print(f"VERDICT: {verdict}")
    print(f"[spec-witness] -> {RESULTS / f'specialisation_witness-dev{device}.json'}")
    return 0 if verdict == "PASS" else (2 if errors else 1)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
