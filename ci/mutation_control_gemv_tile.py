"""Held-out mutation controls for the issue #81 tile-request surface.

Each row plants one defect that the change is supposed to make impossible, runs the SMALLEST
command that ought to notice, and requires it to go red. A guard nobody has seen fail is a guard
nobody has evidence for. Every mutation is reverted before the next one is planted, and the tree
is verified at the end.

Run: ``python ci/mutation_control_gemv_tile.py``. Exit 0 iff every mutation was caught; 1 if any
survived; 4 if the baseline was not green to begin with -- a mutation result read off a red
baseline says nothing about the mutation. The final digest comparison is reported rather than
enforced, because this tree is checked out with ``core.autocrlf=true`` and Python rewrites text
files with the platform line ending; the residue scan that matters is ``git diff``.
"""
from __future__ import annotations

import hashlib
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUST = ROOT / "rust"
PY = sys.executable

QUANT = "rust/src/ops/quant.rs"
COUNTERS = "rust/src/counters.rs"
PROBE = "bench/results/probe_weight_reread.py"
WITNESS = "bench/results/weight_reread_phi35.json"
CENSUS = "ci/census_surface_map.json"

CARGO_UNIT = ["cargo", "test", "--lib", "ops::quant", "--quiet"]
CARGO_INTEG = ["cargo", "test", "--test", "gemv_tile_request", "--quiet"]
PYTEST_PROBE = [PY, "-m", "pytest", "bench/test_weight_reread.py", "-q", "-x"]
CENSUS_CHECK = [PY, "ci/check_census_completeness.py"]

# (id, what the mutation removes, file, old, new, command, cwd)
MUTATIONS = [
    (
        "M1-parser-accepts-junk",
        "the parser's strictness: surrounding whitespace is repaired instead of refused",
        QUANT,
        "        if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {",
        "        let s = s.trim();\n        if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {",
        CARGO_UNIT, RUST,
    ),
    (
        "M2-legality-drops-divisibility",
        "the `N % cols == 0` rule that keeps a column tile from straddling the tensor",
        QUANT,
        "return Err(TileRefusal::ColsIndivisible { cols, n });",
        "let _ = TileRefusal::ColsIndivisible { cols, n };",
        CARGO_UNIT, RUST,
    ),
    (
        "M3-legality-drops-power-of-two",
        "the power-of-two rule the shader's paired store depends on",
        QUANT,
        "return Err(TileRefusal::NotAPowerOfTwo { cols, rows });",
        "let _ = TileRefusal::NotAPowerOfTwo { cols, rows };",
        CARGO_UNIT, RUST,
    ),
    (
        "M4-legality-allows-row-tile-at-decode",
        "the rule that a request may not move the ledger-backed decode geometry",
        QUANT,
        "return Err(TileRefusal::RowTileAtDecode { rows, m });",
        "let _ = TileRefusal::RowTileAtDecode { rows, m };",
        CARGO_UNIT, RUST,
    ),
    (
        "M5-refusal-degrades-to-automatic",
        "refusal itself: an illegal request silently falls back to the selected tile",
        QUANT,
        """            crate::counters::record_gemv_tile_refusal(refusal.form());
            return Err(EpError::Internal(format!(
                "`{}` refused a {GEMV_TILE_VAR} request before dispatch: {refusal}",
                node.op_type
            )));""",
        """            crate::counters::record_gemv_tile_refusal(refusal.form());
            let (c, r) = gemv_tile_with(
                m_rows,
                n as u64,
                k as u64,
                bits as u32,
                a_bytes,
                wg,
                gemv_max_rows(),
            );
            TileChoice::Selected { cols: c, rows: r }""",
        CARGO_INTEG, RUST,
    ),
    (
        "M6-selector-ties-displace",
        "strict improvement: a tie is allowed to displace the incumbent tile",
        QUANT,
        "if bytes < best_bytes {",
        "if bytes <= best_bytes {",
        CARGO_UNIT, RUST,
    ),
    (
        "M7-decision-counter-is-inert",
        "the observability seam: dispatches stop being recorded",
        COUNTERS,
        "pub fn record_gemv_tile_decision(surface: &str, cols: u32, rows: u32) {",
        "pub fn record_gemv_tile_decision(surface: &str, cols: u32, rows: u32) {\n    if true { let _ = (surface, cols, rows); return; }",
        CARGO_INTEG, RUST,
    ),
    (
        "M8-census-entry-removed",
        "the census registration for the new environment surface",
        CENSUS,
        '"id": "ONNXRUNTIME_EP_VULKAN_GEMV_TILE"',
        '"id": "ONNXRUNTIME_EP_VULKAN_GEMV_TILE_UNMAPPED"',
        CENSUS_CHECK, ROOT,
    ),
    (
        "M9-probe-restores-the-path-leak",
        "the redaction: `_result_identity` stamps the resolved absolute path again",
        PROBE,
        '"onnx_file": model.name,',
        '"onnx_file": str(model),',
        PYTEST_PROBE, ROOT,
    ),
    (
        "M10-probe-default-dirties-the-witness",
        "the untracked default: a plain run writes to the committed witness again",
        PROBE,
        'DEFAULT_OUT = ROOT / "bench" / "results" / "_local" / "weight_reread_phi35.json"',
        'DEFAULT_OUT = ROOT / "bench" / "results" / "weight_reread_phi35.json"',
        PYTEST_PROBE, ROOT,
    ),
    (
        "M11-publish-scrubs-instead-of-refusing",
        "fail-closed publication: a leaky record is written anyway",
        PROBE,
        "        raise _path_screen.PrivatePathLeak((why,))",
        "        kept = report",
        PYTEST_PROBE, ROOT,
    ),
    (
        "M12-witness-carries-the-leak-again",
        "the committed artifact's redaction",
        WITNESS,
        '"onnx_file": "phi-3.5-mini-instruct-cuda-int4-rtn-block-32.onnx"',
        '"onnx_file": "C:\\\\Users\\\\someone\\\\.foundry\\\\cache\\\\phi.onnx"',
        PYTEST_PROBE, ROOT,
    ),
]


def digest_tree() -> dict:
    return {
        f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()
        for f in (QUANT, COUNTERS, PROBE, WITNESS, CENSUS)
    }


def run(cmd, cwd) -> int:
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True).returncode


def main() -> int:
    before = digest_tree()

    print("BASELINE — every command must be GREEN before any mutation is planted")
    baseline_ok = True
    for cmd, cwd in [(CARGO_UNIT, RUST), (CARGO_INTEG, RUST),
                     (PYTEST_PROBE, ROOT), (CENSUS_CHECK, ROOT)]:
        rc = run(cmd, cwd)
        ok = rc == 0
        baseline_ok &= ok
        print(f"  [{'GREEN' if ok else 'RED  '}] exit {rc}  {' '.join(cmd[:4])}")
    if not baseline_ok:
        print("\nINSTRUMENT ERROR: the baseline is not green; no mutation result would mean "
              "anything.")
        return 4
    print()

    rows = []
    for mid, what, rel, old, new, cmd, cwd in MUTATIONS:
        target = ROOT / rel
        original = target.read_text(encoding="utf-8")
        if original.count(old) != 1:
            print(f"  [ERROR] {mid}: anchor appears {original.count(old)}x in {rel}")
            rows.append((mid, "INSTRUMENT-ERROR", what))
            continue
        target.write_text(original.replace(old, new), encoding="utf-8")
        try:
            rc = run(cmd, cwd)
        finally:
            target.write_text(original, encoding="utf-8")
        caught = rc != 0
        rows.append((mid, "CAUGHT" if caught else "SURVIVED", what))
        print(f"  [{'CAUGHT  ' if caught else 'SURVIVED'}] exit {rc:<3} {mid}")

    after = digest_tree()
    print()
    print("| mutation | removes | result |")
    print("|---|---|---|")
    for mid, res, what in rows:
        print(f"| `{mid}` | {what} | **{res}** |")
    print()
    survived = [r for r in rows if r[1] != "CAUGHT"]
    print(f"content digests unchanged (line endings aside): {before == after}")
    print(f"caught {len(rows) - len(survived)}/{len(rows)}")
    return 0 if not survived else 1


if __name__ == "__main__":
    sys.exit(main())
