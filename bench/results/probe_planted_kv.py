#!/usr/bin/env python3
"""Plant an all-zero KV write and ask whether ANY existing gate goes red.

Morpheus asked for this before anything is built, and asked for it in this order
deliberately: **a negative result here is worth more than a new gate.** If some existing
gate catches a stable all-zero KV write, criterion 10 re-closes on evidence that already
exists and nobody has to write anything.

WHY THIS IS RUN AGAINST THE GATE LOGIC AND NOT AGAINST THE GPU
==============================================================
Two reasons, and the first is decisive:

  1. **The Phi-3.5 artifact is not on this machine.**
     ``C:\\Users\\justinchu\\.foundry\\cache\\models\\Microsoft\\Phi-3.5-mini-instruct-cuda-gpu``
     does not exist, so ``test_criterion10.py`` and ``test_phi35.py`` both hit their
     ``pytest.skip`` before reaching a single assertion.  A GPU plant is not available to
     me at any price today.
  2. **The question is about the gates, not about the GPU.**  "Does any existing gate go
     red when KV is stably zero" is a question about what the gate code reads.  Feeding a
     planted output list to the real, unmodified gate functions answers it directly, and
     answers it for *every* device rather than for whichever one happened to be free.

The gates exercised here are the production ones, imported not copied — ``m.outputs_bit_
equal``, ``m.AttributedRunSeries``, and ``test_criterion10._compare_run_to_cpu`` itself.
If any of those changes, this probe changes with it.  A copy would have let the gate drift
away from its own falsifier.

WHAT IS PLANTED
===============
Sixty-five outputs, matching the real model's arity:

* **output 0** — logits, ``[1, 1, 32064]``.  *Correct in both arms.*  This is the point:
  the defect under test does not touch the logits, which is exactly why one-output
  comparison cannot see it.
* **outputs 1..64** — KV caches.  Correct arm carries plausible non-zero values; planted
  arm is **all zero**.

The plant is **wrong and stable**: byte-identical across all three runs.  An *unstable*
plant would be caught by cross-run identity and would prove nothing about this gap
(Morpheus's discharge condition (b)).  Stability is what makes it the right control — a
dirty arena produces divergence, and divergence is the symptom the row was closed on.  A
clean arena leaves the same defect silent.

Run:  python bench/results/probe_planted_kv.py
"""

from __future__ import annotations

import json
import pathlib
import sys
from datetime import datetime, timezone

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "ops"))

import _models as m  # noqa: E402
from test_criterion10 import _compare_run_to_cpu  # noqa: E402

N_OUTPUTS = 65
VOCAB = 32064
KV_SHAPE = (1, 32, 8, 96)


def build_correct_run(seed: int = 7) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    logits = rng.normal(0.0, 4.0, size=(1, 1, VOCAB)).astype(np.float32)
    logits[0, 0, 1234] = 40.0  # an unambiguous argmax, stable across arms
    outs = [logits]
    for i in range(1, N_OUTPUTS):
        outs.append(rng.normal(0.0, 1.0, size=KV_SHAPE).astype(np.float16))
    return outs


def build_planted_run(correct: list[np.ndarray]) -> list[np.ndarray]:
    """Identical logits, all-zero KV. Wrong, and stable."""
    planted = [correct[0].copy()]
    for arr in correct[1:]:
        planted.append(np.zeros_like(arr))
    return planted


def _identical_attribution(tmp: pathlib.Path) -> "m.ExecutionAttribution":
    """A single attribution, parsed by the real mechanism, shared by BOTH arms.

    ``ExecutionAttribution()`` is private (R10 amendment 1: an attribution must be a value
    a mechanism computed, never a literal its author set), so this writes a trace and lets
    ``from_profile`` parse it — the mechanism runs.

    **The trace is fabricated, and that is stated rather than hidden.**  It is legitimate
    here for one reason only: attribution is *held identical across both arms*, so it
    cannot be the thing that discriminates them.  Whatever it says, it says equally about
    the correct run and the planted one.  This probe asks whether the *comparison* gates
    catch a stable all-zero KV write; it makes no claim about attribution, and its output
    must not be read as one.
    """
    events = [
        {
            "cat": "Node",
            "name": f"VulkanExecutionProvider_fused_{i}",
            "dur": 100,
            "args": {"provider": "VulkanExecutionProvider"},
        }
        for i in range(3)
    ]
    path = tmp / "planted_kv_probe_profile.json"
    path.write_text(json.dumps(events), encoding="utf-8")
    return m.ExecutionAttribution.from_profile(path, delete=True)


def gate_results(
    vk_runs: list[list[np.ndarray]],
    cpu_out: list[np.ndarray],
    attribution: "m.ExecutionAttribution",
) -> dict:
    """Run every gate criterion 10 actually applies, and report each one's colour."""
    comparisons: list[str] = []
    cross_run_identical: list[bool] = []
    per_run: list[dict] = []

    for run in vk_runs:
        identical, differing = m.outputs_bit_equal(vk_runs[0], run)
        outcome, facts = _compare_run_to_cpu(run, cpu_out)
        if not identical:
            outcome = m.COMPARISON_DISAGREE
            facts["cross_run_differing_outputs"] = differing[:10]
        facts["cross_run_identical_to_run1"] = identical
        facts["cross_run_outputs_compared"] = len(run)
        comparisons.append(outcome)
        cross_run_identical.append(identical)
        per_run.append(facts)

    series = m.AttributedRunSeries.from_runs(
        comparisons=comparisons,
        attribution=attribution,
        artifact="planted-kv-probe",
        device_index="n/a",
    )

    # test_phi35 Guard 1, applied to the same tensor it guards: output 0.
    logits = vk_runs[0][0].astype(np.float32)
    guard1_range = float(logits.max()) - float(logits.min())

    kv = vk_runs[0][1:]
    return {
        "comparisons": comparisons,
        "cross_run_identical": cross_run_identical,
        "series_verdict": series.verdict,
        "guard1_logit_range": guard1_range,
        "guard1_passes": guard1_range > 1.0,
        "oracle_max_abs_diff": per_run[0]["logits_max_abs_diff"],
        "outputs_compared_key": per_run[0]["cross_run_outputs_compared"],
        "kv_outputs_all_zero": bool(all(not np.any(a) for a in kv)),
        "kv_outputs_count": len(kv),
        # The arm that did not exist when this probe was written, run over the same input
        # so the before/after is one comparison rather than two runs.
        "all_output_oracle_outcome": m.compare_all_outputs_to_cpu(vk_runs[0], cpu_out)[0],
    }


def main() -> int:
    correct = build_correct_run()
    planted = build_planted_run(correct)

    results_dir = ROOT / "bench" / "results"
    clean_arm = gate_results([correct] * 3, correct, _identical_attribution(results_dir))
    planted_arm = gate_results([planted] * 3, correct, _identical_attribution(results_dir))

    red = {
        "cross_run_identity": not all(planted_arm["cross_run_identical"]),
        "cpu_oracle_comparison": m.COMPARISON_DISAGREE in planted_arm["comparisons"],
        "series_verdict": planted_arm["series_verdict"] != clean_arm["series_verdict"],
        "phi35_guard1_logit_range": not planted_arm["guard1_passes"],
    }
    new_arm_red = planted_arm["all_output_oracle_outcome"] != m.COMPARISON_AGREE

    report = {
        "probe": "planted_kv_all_zero",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "question": (
            "Does any gate that exists today go red when 64 of 65 outputs are stably "
            "all-zero?"
        ),
        "model_artifact_present_on_this_box": False,
        "plant": {
            "outputs": N_OUTPUTS,
            "corrupted": "1..64 (all KV), set to zero",
            "logits_output_0": "unmodified, identical to the oracle",
            "stable_across_runs": True,
            "why_stable": (
                "an unstable plant is caught by cross-run identity and would prove "
                "nothing about this gap"
            ),
        },
        "clean_arm": clean_arm,
        "planted_arm": planted_arm,
        "gates_that_went_red": red,
        "any_pre_existing_gate_red": any(red.values()),
        "new_all_output_oracle_red": new_arm_red,
    }

    out = ROOT / "bench" / "results" / "planted_kv_probe.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("PLANTED-KV PROBE — does any existing gate catch a stable all-zero KV write?\n")
    print("  plant: outputs 1..64 of 65 set to zero, logits untouched, stable across 3 runs")
    print(f"  planted arm KV all zero: {planted_arm['kv_outputs_all_zero']} "
          f"({planted_arm['kv_outputs_count']} outputs)\n")
    print("  gates that existed when criterion 10 was closed:")
    for gate, is_red in red.items():
        print(f"    {'RED  ' if is_red else 'green'}  {gate}")
    print()
    print(f"  clean   arm: verdict={clean_arm['series_verdict']} "
          f"comparisons={clean_arm['comparisons']}")
    print(f"  planted arm: verdict={planted_arm['series_verdict']} "
          f"comparisons={planted_arm['comparisons']}")
    print(f"  logits max_abs_diff, planted arm: {planted_arm['oracle_max_abs_diff']} "
          f"(one output out of {N_OUTPUTS})")
    print(f"  cross-run outputs compared: {planted_arm['outputs_compared_key']}")
    print()
    if any(red.values()):
        print("  RESULT: at least one pre-existing gate catches it. Morpheus is wrong and")
        print("          the row re-closes on existing evidence. No new gate is needed.")
    else:
        print("  RESULT: NO pre-existing gate goes red. A deterministically wrong KV write")
        print("          is invisible to every gate criterion 10 applied. The missing arm")
        print("          is real.")
    print()
    print("  the arm added in response:")
    print(f"    {'RED  ' if new_arm_red else 'green'}  compare_all_outputs_to_cpu -> "
          f"{planted_arm['all_output_oracle_outcome']}")
    print(f"    clean arm for contrast          -> {clean_arm['all_output_oracle_outcome']}")
    print(f"\n  record: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
