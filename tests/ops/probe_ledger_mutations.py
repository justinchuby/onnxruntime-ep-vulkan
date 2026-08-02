"""Mutation probe: do the criterion-11(c) assertions actually fire?

A green test is not evidence that a check works; it is evidence that the world is
currently the way the check expects. R10's falsifier is an artifact whose content varies
with its input, and for a *test* that means: mutate the thing it is supposed to catch, and
watch it go red. A one-armed demonstration proves nothing.

Each mutation below breaks exactly one of the properties the lane claims to pin. Every one
must produce a FAILURE. A mutation that leaves the lane green names an assertion that is
decoration.

Output: bench/results/criterion11c_mutations-dev{N}.json
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import test_wiring_census as census  # noqa: E402

CASES = census._EVIDENCE_CASES
SELECTOR = os.environ.get("ONNXRUNTIME_EP_VULKAN_DEVICE", "unset")
OUT = census._REPO_ROOT / "bench" / "results" / f"criterion11c_mutations-dev{SELECTOR}.json"


class _Tmp:
    """A stand-in for pytest's tmp_path, so the probe can call the test bodies directly."""

    def __init__(self, base: pathlib.Path):
        self.base = base
        base.mkdir(parents=True, exist_ok=True)

    def __truediv__(self, other):
        return self.base / other


def _run(name: str, fn, mutation: str) -> dict:
    try:
        fn()
    except AssertionError as exc:
        return {
            "mutation": name, "broke": mutation, "outcome": "CAUGHT",
            "failure_text": str(exc).splitlines()[0][:300],
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "mutation": name, "broke": mutation, "outcome": "ERROR(instrument)",
            "failure_text": f"{type(exc).__name__}: {exc}"[:300],
        }
    return {"mutation": name, "broke": mutation, "outcome": "MISSED"}


def main() -> int:
    tmp = _Tmp(census._REPO_ROOT / "bench" / "results" / "mutation_scratch")
    real_arm = census._ledger_arm
    real_ledger = census._EVIDENCE_LEDGER
    rows = []

    # M1 — the two dtype arms are given the SAME model. If `ledger_hits` were derived from
    # the claim enumeration rather than read from the ledger, this is what every run would
    # already look like.
    def m1():
        def stub(tag, *, model=None, ledger_file=None):
            if tag == "ledger_dtype_unproven":
                model = CASES / "mul_f32.onnx"
            return real_arm(tag, model=model, ledger_file=ledger_file)

        census._ledger_arm = stub
        try:
            census.test_ledger_hits_moves_with_its_input(None, tmp)
        finally:
            census._ledger_arm = real_arm

    rows.append(_run("M1_dtype_arms_collapsed", m1,
                     "the proven and unproven dtype arms read the same input"))

    # M2 — the *control* arm of the digest test is handed the drifted ledger. This is the
    # mutation that turns the refusal into a check that fails on everything.
    def m2():
        drifted = tmp / "m2_drifted.jsonl"
        lines = real_ledger.read_text(encoding="utf-8").splitlines()
        drifted.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

        def stub(tag, *, model=None, ledger_file=None):
            if tag == "ledger_digest_same":
                ledger_file = drifted
            return real_arm(tag, model=model, ledger_file=ledger_file)

        census._ledger_arm = stub
        try:
            census.test_ledger_digest_refusal_is_in_the_lane(None, tmp)
        finally:
            census._ledger_arm = real_arm

    rows.append(_run("M2_digest_control_drifted", m2,
                     "the identical-file control arm was given a drifted ledger"))

    # M3 — the two MatMulNBits ledger keys are collapsed onto one, which is the
    # 2026-07-30 all-zero-logits defect written back into the artifact.
    def m3():
        collapsed = tmp / "m3_collapsed_ledger.jsonl"
        lines = real_ledger.read_text(encoding="utf-8").splitlines()
        out = []
        for line in lines:
            if '"key"' in line and "MatMulNBits" in line:
                doc = json.loads(line)
                doc["key"] = (
                    "com.microsoft::MatMulNBits/1+/f16,u8,f16>f16/"
                    "q_gemv_matmul_nbits_f16/static/scales"
                )
                line = json.dumps(doc, sort_keys=True)
            out.append(line)
        collapsed.write_text("\n".join(out) + "\n", encoding="utf-8")
        census._EVIDENCE_LEDGER = collapsed
        try:
            census.test_ledger_key_discriminates_optional_inputs(None)
        finally:
            census._EVIDENCE_LEDGER = real_ledger

    rows.append(_run("M3_optional_input_component_dropped", m3,
                     "the two MatMulNBits keys were collapsed onto one"))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    missed = [r for r in rows if r["outcome"] != "CAUGHT"]
    for r in rows:
        print(json.dumps(r))
    print(f"\nwitness: {OUT}")
    if missed:
        print(f"MUTATION PROBE: FAIL(condition=mutations_missed) — {len(missed)}")
        return 1
    print("MUTATION PROBE: PASS — every mutation was caught")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
