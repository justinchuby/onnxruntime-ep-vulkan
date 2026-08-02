"""Two-armed probe for the `session.disable_cpu_ep_fallback` guard (R11: arms_must_differ).

Subject is the planted control `sub_f16_dyn_unproven`, chosen because the generator refuses to
write its key under any circumstance, so neither arm can contaminate the ledger.

  ARM CLAIMED  — hatch open on the real key.  The EP takes the node.  The guard must stay
                 silent and the run must reach a verdict.
  ARM DECLINED — hatch open on a key that does not exist.  The EP declines, the node falls to
                 the default CPU EP, and ORT must refuse the session.

A guard that fired in both arms would be broken in the direction that costs the most: it would
make every proof run look vacuous.  A guard that fired in neither is not wired at all.
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import gen_proof_ledger as g  # noqa: E402

CASE = str(g.REPO / "evidence" / "cases" / "sub_f16_dyn_unproven.onnx")
REAL_KEY = "ai.onnx::Sub/7+/f16,f16>f16/ew_binary_sub_f16/runtime-extent/n2"
BOGUS_KEY = "ai.onnx::Sub/7+/f16,f16>f16/ew_binary_sub_f16/runtime-extent/NOT-A-REAL-SLOT-SET"

arms = {}
for name, key in (("claimed", REAL_KEY), ("declined", BOGUS_KEY)):
    verdict, detail = g.prove(CASE, [key], (1e-2, 1e-3))
    arms[name] = {
        "key_offered": key,
        "verdict": verdict,
        "ort_refused": "ort_refusal" in detail,
        "reason": detail.get("reason"),
        "refusal_text": detail.get("ort_refusal", "")[-400:],
        "worst_rel": detail.get("worst_rel"),
        "claimed_nodes": detail.get("claimed_nodes"),
        "dispatches_executed": detail.get("dispatches_executed"),
    }
    print(f"[{name}] verdict={verdict} ort_refused={arms[name]['ort_refused']}")

out = g.REPO / "bench" / "results" / "cpu_fallback_guard_probe.json"
out.write_text(json.dumps(arms, indent=2), encoding="utf-8")

failures = []
if arms["claimed"]["ort_refused"]:
    failures.append(
        "ARM CLAIMED was refused by ORT. The guard fires on a run the EP did take, which would "
        f"make every proof run look vacuous: {arms['claimed']}"
    )
if not arms["declined"]["ort_refused"]:
    failures.append(
        "ARM DECLINED was NOT refused by ORT. The guard is not wired: a run in which the EP "
        f"took nothing produced a comparison anyway: {arms['declined']}"
    )
if arms["claimed"]["verdict"] == arms["declined"]["verdict"]:
    failures.append(
        "arms_must_differ FAILED: both arms reached verdict "
        f"{arms['claimed']['verdict']!r}. A stable answer that does not move with its input is "
        "not a reading."
    )

if failures:
    print("FAIL(condition)")
    for f in failures:
        print("  " + f)
    sys.exit(1)
print(f"PASS  claimed={arms['claimed']['verdict']} declined={arms['declined']['verdict']}")
print(f"witness={out}")
