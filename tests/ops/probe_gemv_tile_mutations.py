"""Both arms of the `ONNXRUNTIME_EP_VULKAN_GEMV_TILE` locks: they must go red when broken.

A test suite that has only ever been watched to pass is indistinguishable from one that cannot
fail. The tile-request tests in `rust/src/ops/quant.rs` and `rust/tests/gemv_tile_override.rs`
are the whole warrant for issue #81's claim that the `(8,4)` GEMV arm is *reachable and
measurable*, so this file breaks the thing they guard, one property per mutation, and requires
the suite to catch each one.

The six mutations are the six ways this instrument fails while still looking wired:

  M1  parser trims whitespace     `" 8,4"` becomes a tile instead of a refusal
  M2  parser accepts zero         `"0,4"` becomes a tile — a division by zero in the geometry
  M3  malformed falls back        an unreadable value silently runs the CONTROL arm
  M4  legality drops a rule       the accumulator budget stops being checked
  M5  ceiling clamps not refuses  a request above the ceiling runs a DIFFERENT tile
  M6  handler ignores the request the knob is disconnected; both arms are the control arm

M3 and M6 are the two that matter most and they are the two that a passing-only suite is
likeliest to miss, because in both of them *every dispatch still succeeds and still produces
correct output*. The only symptom is that the measurement is labelled with an arm that never
ran, which is a wrong number in a document rather than a wrong number on a device.

R12 generalisation 4 — *for a test result, the frame is the binary that ran it* — is why the
restored file is hashed back to the original before anything above is reported, and why the
suite is re-run green afterwards: a mutated source restored to identical bytes proves nothing
if the arms above were served from a stale build.

Writes `bench/results/gemv_tile_mutations.json`.
Run: `python tests/ops/probe_gemv_tile_mutations.py`
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
QUANT = ROOT / "rust" / "src" / "ops" / "quant.rs"
OUT = ROOT / "bench" / "results" / "gemv_tile_mutations.json"

#: The two suites that between them carry every claim. Both are run for every arm, because the
#: mutations deliberately split across them: M1/M2/M4/M5 are unit-level properties of the pure
#: functions, while M3 and M6 are only visible where the request meets a dispatch.
SUITES = [
    ("unit", ["test", "--lib", "ops::quant::tests::"]),
    ("live", ["test", "--test", "gemv_tile_override"]),
]

#: (name, what it breaks, old, new). Each `old` must appear exactly once in the file, or the
#: mutation landed somewhere other than where this file says it did and the arm is not the arm
#: it claims to be.
MUTATIONS = [
    (
        "M1_parser_trims_whitespace",
        "the parser accepts ' 8,4' — a value whose text is not the text the operator typed",
        "    let fields: Vec<&str> = raw.split(',').collect();\n",
        "    let fields: Vec<&str> = raw.trim().split(',').map(|s| s.trim()).collect();  // MUTANT M1\n",
    ),
    (
        "M2_parser_accepts_zero",
        "a zero field becomes a tile: cols=0 divides by zero, rows=0 is an infinite row count",
        "        if v == 0 {\n            return Err(TileSyntaxError::Zero { field });\n        }\n",
        "        // MUTANT M2: the zero check is gone\n",
    ),
    (
        "M3_malformed_falls_back_to_the_selector",
        "an unreadable value silently runs the CONTROL arm while the operator labels it as the "
        "treatment arm — every dispatch succeeds and the measurement is wrong",
        "    let (cols, rows) = parse_gemv_tile(raw).map_err(TileRefusal::Malformed)?;\n",
        "    let Ok((cols, rows)) = parse_gemv_tile(raw) else {  // MUTANT M3\n"
        "        let (cols, rows) = gemv_tile_with(m, n, k, bits, a_bytes, wg, max_rows);\n"
        "        return Ok(TileChoice::Automatic { cols, rows });\n"
        "    };\n",
    ),
    (
        "M4_legality_drops_the_accumulator_budget",
        "cols*rows stops being checked, so a request can reach the shader's out-of-bounds arm",
        "    if cols * rows > GEMV_MAX_TILE {\n        return Err(TileIllegality::TileTooLarge {\n"
        "            cols,\n            rows,\n            max: GEMV_MAX_TILE,\n        });\n    }\n",
        "    // MUTANT M4: the accumulator budget is no longer a condition\n",
    ),
    (
        "M5_ceiling_clamps_instead_of_refusing",
        "a request above the process ceiling runs a DIFFERENT tile than the one it is labelled "
        "with — the exact reading that is correct for a ceiling and wrong for an exact request",
        "    if rows > max_rows {\n        return Err(TileIllegality::RowsAboveCeiling {\n"
        "            rows,\n            ceiling: max_rows,\n        });\n    }\n",
        "    let rows = rows.min(max_rows);  // MUTANT M5\n",
    ),
    (
        "M6_handler_ignores_the_request",
        "the knob is disconnected: the parser and the legality function are perfect and never "
        "consulted, so both arms of the A/B build the same pipeline",
        "    let request = std::env::var(ENV_GEMV_TILE).ok();\n"
        "    matmul_nbits_gemv_with_request(spec, node, ctx, request.as_deref())\n",
        "    matmul_nbits_gemv_with_request(spec, node, ctx, None)  // MUTANT M6\n",
    ),
]


def _cargo() -> str:
    """`cargo`, from PATH or from the default rustup location.

    Resolved rather than assumed because a probe that reports 'every mutation was caught'
    because `cargo` was missing and every run failed would be the worst possible outcome — and
    `_run` below distinguishes a compile failure from a test failure for the same reason.
    """
    found = shutil.which("cargo")
    if found:
        return found
    fallback = Path(os.path.expanduser("~")) / ".cargo" / "bin" / (
        "cargo.exe" if os.name == "nt" else "cargo")
    if fallback.is_file():
        return str(fallback)
    raise SystemExit("ERROR(instrument): cargo is not on PATH and not at ~/.cargo/bin")


def _run(args: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        [_cargo(), *args], cwd=str(ROOT / "rust"),
        capture_output=True, text=True, timeout=1800,
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _suite() -> tuple[bool, dict]:
    """Run both suites. Returns `(all_green, per-suite detail)`.

    A **compile** failure is recorded distinctly from a **test** failure. A mutation that does
    not compile has not demonstrated that any test noticed it — it has demonstrated that rustc
    did — and counting that as CAUGHT would let this probe certify a suite that checks nothing.
    """
    detail = {}
    green = True
    for name, args in SUITES:
        rc, log = _run(args)
        compiled = "error[E" not in log and "error: could not compile" not in log
        failing = [ln.strip() for ln in log.splitlines()
                   if ln.strip().startswith("test ") and ln.strip().endswith("FAILED")]
        detail[name] = {
            "returncode": rc,
            "compiled": compiled,
            "failing_tests": failing,
            "first_failure": next(
                (ln.strip() for ln in log.splitlines()
                 if "panicked at" in ln or ln.strip().startswith("assert")), ""
            )[:300],
        }
        if rc != 0:
            green = False
    return green, detail


def main() -> int:
    original = QUANT.read_text(encoding="utf-8")
    digest = hashlib.sha256(original.encode("utf-8")).hexdigest()

    baseline_green, baseline = _suite()
    if not baseline_green:
        print("ERROR(instrument): the suite is not green before mutation; nothing below would "
              "be attributable to a mutation.")
        print(json.dumps(baseline, indent=2))
        return 2

    results: list[dict] = []
    try:
        for name, broke, old, new in MUTATIONS:
            count = original.count(old)
            if count != 1:
                results.append({
                    "mutation": name,
                    "outcome": "ERROR(instrument)",
                    "reason": f"anchor found {count} times, expected exactly 1",
                })
                continue
            QUANT.write_text(original.replace(old, new), encoding="utf-8")
            green, detail = _suite()
            compiled = all(d["compiled"] for d in detail.values())
            caught_by_a_test = any(d["failing_tests"] for d in detail.values())
            if not compiled:
                outcome = "REJECTED-BY-COMPILER"
            elif green:
                outcome = "MISSED"
            elif caught_by_a_test:
                outcome = "CAUGHT"
            else:
                outcome = "RED-BUT-NOT-BY-A-TEST"
            results.append({
                "mutation": name,
                "broke": broke,
                "outcome": outcome,
                "suites": detail,
            })
    finally:
        QUANT.write_text(original, encoding="utf-8")

    restored = hashlib.sha256(QUANT.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
    ok_restore = restored == digest
    after_green, after = _suite()

    doc = {
        "probe": "gemv tile request locks, both arms",
        "subject": "rust/src/ops/quant.rs + rust/tests/gemv_tile_override.rs",
        "no_clock": "Every outcome here is a test verdict, not a duration.",
        "baseline": baseline,
        "mutations": results,
        "restored_identical": ok_restore,
        "restored_sha256": restored,
        "green_after_restore": after_green,
        "after_restore": after,
        "note": "A mutation that is MISSED means the suite would certify that broken "
                "instrument. REJECTED-BY-COMPILER means rustc noticed and no test did, which "
                "is not the same evidence and is not counted as a pass.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    for r in results:
        print(f"  [{r.get('outcome'):>21}] {r['mutation']}")
    print(f"\nwitness: {OUT.relative_to(ROOT).as_posix()}")

    if not ok_restore or not after_green:
        print("ERROR(instrument): quant.rs did not come back clean — do not read the arms "
              f"above as evidence (restored_identical={ok_restore}, green={after_green})")
        return 2
    missed = [r["mutation"] for r in results if r.get("outcome") != "CAUGHT"]
    if missed:
        print(f"GEMV TILE MUTATION PROBE: FAIL(condition) — not caught by a test: {missed}")
        return 1
    print("GEMV TILE MUTATION PROBE: PASS — every mutation was caught by a named test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
