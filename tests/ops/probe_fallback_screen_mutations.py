"""Both arms of the no-CPU-fallback screen: it must go red when the guards are broken.

A screen that has only ever been watched to pass is indistinguishable from a screen that
cannot fail.  `test_no_cpu_fallback_screen.py` passing tells you nothing on its own, so this
mutates `_models.py` — one property per mutation — and requires the screen to catch each.

The three mutations are the three ways a guard fails while looking wired, which is Tank's
`unfalsified` state spelled out:

  M1  always-passes   the precondition stops calling ORT at all
  M2  always-rejects  ORT's refusal is raised unconditionally
  M3  silently-inert  the session-config key is never armed

R12 generalisation 4 — *for a test result, the frame is the binary that ran it* — applies to
Python too: a mutated source restored to identical bytes could still be served from a stale
`__pycache__`, so this runs the child with `-B`, deletes the cache directory between arms,
and verifies the restored file hashes back to the original before reporting anything.

Writes `bench/results/fallback_screen_mutations.json`.  Run: `python probe_fallback_screen_mutations.py`
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = HERE / "_models.py"
SCREEN = HERE / "test_no_cpu_fallback_screen.py"
OUT = HERE.parent.parent / "bench" / "results" / "fallback_screen_mutations.json"

#: (name, what it breaks, old, new).  Each `old` must appear exactly once, or the mutation
#: silently applied somewhere else and the arm is not the arm it claims to be.
MUTATIONS = [
    (
        "M1_precondition_never_asks_ORT",
        "assert_ep_owns_whole_graph stops building a session — a guard that always passes",
        "    ep_only_session_or_refusal(model)\n",
        "    return None  # MUTANT M1\n",
    ),
    (
        "M2_refusal_raised_unconditionally",
        "every session is refused — a guard that rejects everything certifies nothing",
        "    try:\n        return ort.InferenceSession(\n",
        "    try:\n        raise CpuFallbackRefused('MUTANT M2 " + "fallback to CPU EP has been explicitly disabled" + "')\n        return ort.InferenceSession(\n",
    ),
    (
        "M3_key_never_armed",
        "the session-config key is not set — the option is silently inert",
        '    opts.add_session_config_entry(ORT_DISABLE_CPU_FALLBACK_KEY, "1")\n',
        "    pass  # MUTANT M3\n",
    ),
]


def _run_screen() -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, "-B", "-m", "pytest", str(SCREEN), "-q", "-p", "no:cacheprovider"],
        cwd=str(HERE), capture_output=True, text=True, timeout=600,
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _clear_cache() -> None:
    shutil.rmtree(HERE / "__pycache__", ignore_errors=True)


def main() -> int:
    original = MODELS.read_text(encoding="utf-8")
    digest = hashlib.sha256(original.encode("utf-8")).hexdigest()
    results: list[dict] = []

    _clear_cache()
    rc, log = _run_screen()
    if rc != 0:
        print("ERROR(instrument): the screen is not green before mutation; nothing below "
              "would be attributable to a mutation.\n" + log[-3000:])
        return 2
    baseline_passed = [ln for ln in log.splitlines() if " passed" in ln]

    try:
        for name, broke, old, new in MUTATIONS:
            count = original.count(old)
            if count != 1:
                results.append({"mutation": name, "outcome": "ERROR(instrument)",
                                "reason": f"anchor found {count} times, expected exactly 1"})
                continue
            MODELS.write_text(original.replace(old, new), encoding="utf-8")
            _clear_cache()
            rc, log = _run_screen()
            failing = [ln.strip() for ln in log.splitlines() if ln.startswith("FAILED ")]
            results.append({
                "mutation": name,
                "broke": broke,
                "outcome": "CAUGHT" if rc != 0 else "MISSED",
                "failing_tests": failing,
                "failure_text": next(
                    (ln.strip() for ln in log.splitlines()
                     if ln.strip().startswith(("E ", "assert", "Failed:"))), ""
                )[:300],
            })
    finally:
        MODELS.write_text(original, encoding="utf-8")
        _clear_cache()

    restored = hashlib.sha256(MODELS.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
    ok_restore = restored == digest

    _clear_cache()
    rc_after, log_after = _run_screen()

    doc = {
        "probe": "no-cpu-fallback screen, both arms",
        "baseline": baseline_passed,
        "mutations": results,
        "restored_identical": ok_restore,
        "green_after_restore": rc_after == 0,
        "note": "A mutation that is MISSED means the screen would certify that broken guard.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    for r in results:
        print(json.dumps(r))
    print(f"\nwitness: {OUT}")

    if not ok_restore or rc_after != 0:
        print("ERROR(instrument): _models.py did not come back clean — do not read the "
              f"arms above as evidence (restored_identical={ok_restore}, green={rc_after == 0})")
        print(log_after[-2000:])
        return 2
    missed = [r["mutation"] for r in results if r.get("outcome") != "CAUGHT"]
    if missed:
        print(f"SCREEN MUTATION PROBE: FAIL(condition) — MISSED {missed}")
        return 1
    print("SCREEN MUTATION PROBE: PASS — every mutation was caught")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
