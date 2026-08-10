# Negative-control driver for ci/check_anchor_weight_sites.py.
#
# A checker that has only ever been observed passing is one step from a checker that cannot
# fail. This script injects, into a SCRATCH COPY of the tree - never the real repository -
# each defect the anchor weight-site checker claims to detect, and asserts the checker goes
# red for it. The defects are exactly the ways issue #73 could be silently reintroduced:
#   1. name-only anchoring - give GroupQueryAttention a weight site in the Rust table;
#   2. table drift - drop MatMulNBits' weight site from the Rust table;
#   3. schema lie - claim a weight-site index that names a non-weight operand;
#   4. the substring trap - declare GQA anchor-eligible in the JSON despite its *_weight inputs;
#   5. stale provenance - move the pinned ORT commit away from PROVENANCE.md;
#   6. name-only signature - change is_anchor to take an op name alone.
#
# Output goes to bench/results/. Nothing here writes to the repository source.
#
#   python ci/negative_control_anchor_weight_sites.py

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRATCH = REPO / "bench" / "results" / "anchor-weight-sites-negative"
CHECKER_REL = "ci/check_anchor_weight_sites.py"
PARTITION_REL = "rust/src/ops/partition.rs"
JSON_REL = "rust/tools/anchor_weight_sites.json"
PROV_REL = "third_party/onnxruntime/PROVENANCE.md"

# Everything the checker reads, mirrored so it resolves paths relative to the scratch root.
MIRROR = ["ci", "rust/src/ops", "rust/tools", "third_party/onnxruntime"]

EXIT_PASS, EXIT_FAIL, EXIT_ERROR = 0, 1, 4


def fresh_scratch() -> Path:
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    SCRATCH.mkdir(parents=True)
    for rel in MIRROR:
        src = REPO / rel
        dst = SCRATCH / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)
    return SCRATCH


def run(root: Path) -> tuple[int, str]:
    p = subprocess.run(
        [sys.executable, str(root / CHECKER_REL), "--check"],
        cwd=str(root),
        capture_output=True,
        text=True,
    )
    return p.returncode, p.stdout + p.stderr


def edit(root: Path, rel: str, old: str, new: str) -> None:
    path = root / rel
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(
            f"NEGATIVE-CONTROL: ERROR(instrument=anchor_not_found) - {old!r} is not in "
            f"{rel}. The control cannot inject its defect, so it is asserting nothing."
        )
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def edit_json(root: Path, fn) -> None:
    path = root / JSON_REL
    doc = json.loads(path.read_text(encoding="utf-8"))
    fn(doc)
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def _give_gqa_a_rust_site(root: Path) -> None:
    edit(
        root,
        PARTITION_REL,
        '"com.microsoft::MatMulNBits" => &[1],',
        '"com.microsoft::MatMulNBits" => &[1],\n        '
        '"com.microsoft::GroupQueryAttention" => &[14],',
    )


def _drop_matmulnbits_site(root: Path) -> None:
    edit(root, PARTITION_REL, '"com.microsoft::MatMulNBits" => &[1],', "")


def _lie_about_matmul_site(root: Path) -> None:
    # Index 0 is "A" (an activation), not the weight "B" at index 1.
    edit_json(root, lambda d: _set_sites(d, "MatMul", [0]))


def _gqa_anchor_in_json(root: Path) -> None:
    edit_json(root, lambda d: _set_sites(d, "com.microsoft::GroupQueryAttention", [14]))


def _set_sites(doc: dict, op: str, sites: list) -> None:
    for row in doc["ops"]:
        if row["qualified_op"] == op:
            row["weight_sites"] = sites
            return
    raise SystemExit(f"NEGATIVE-CONTROL: ERROR(instrument=op_not_found) - {op} missing from JSON")


def _stale_ort_commit(root: Path) -> None:
    edit_json(root, lambda d: d["provenance"]["onnxruntime"].__setitem__(
        "commit", "0000000000000000000000000000000000000000"))


def _name_only_is_anchor(root: Path) -> None:
    edit(
        root,
        PARTITION_REL,
        "pub fn is_anchor(qualified_op: &str, resident_inputs: &[bool]) -> bool {",
        "pub fn is_anchor(qualified_op: &str) -> bool {",
    )


CASES = [
    {
        "name": "name-only anchoring - GQA given a weight site in the Rust table",
        "mutate": _give_gqa_a_rust_site,
        "expect_substr": "GroupQueryAttention",
    },
    {
        "name": "table drift - MatMulNBits' weight site dropped from the Rust table",
        "mutate": _drop_matmulnbits_site,
        "expect_substr": "MatMulNBits",
    },
    {
        "name": "schema lie - MatMul weight site pointed at operand 'A' not 'B'",
        "mutate": _lie_about_matmul_site,
        "expect_substr": "MatMul",
    },
    {
        "name": "substring trap - GQA declared anchor-eligible in the JSON",
        "mutate": _gqa_anchor_in_json,
        "expect_substr": "GroupQueryAttention",
    },
    {
        "name": "stale provenance - pinned ORT commit no longer matches PROVENANCE.md",
        "mutate": _stale_ort_commit,
        "expect_substr": "ORT commit",
    },
    {
        "name": "name-only signature - is_anchor reverts to taking an op name alone",
        "mutate": _name_only_is_anchor,
        "expect_substr": "resident_inputs",
    },
]


def main() -> int:
    results = []
    ok = True

    root = fresh_scratch()
    code, out = run(root)
    baseline_ok = code == EXIT_PASS
    results.append({"case": "baseline (unmodified copy)", "exit": code, "ok": baseline_ok})
    print(f"[{'ok' if baseline_ok else 'BAD'}] baseline scratch copy -> exit {code}")
    if not baseline_ok:
        print(out)
        print(
            "NEGATIVE-CONTROL: ERROR(instrument=baseline_not_green) - the unmodified copy "
            "is already red, so no injection below can be attributed to the injection."
        )
        return EXIT_ERROR

    for case in CASES:
        root = fresh_scratch()
        case["mutate"](root)
        code, out = run(root)
        went_red = code == EXIT_FAIL
        named = case["expect_substr"] in out
        good = went_red and named
        ok &= good
        results.append(
            {
                "case": case["name"],
                "exit": code,
                "went_red": went_red,
                "named_the_defect": named,
                "ok": good,
            }
        )
        print(
            f"[{'ok' if good else 'BAD'}] {case['name']}\n"
            f"        exit={code} red={went_red} named={named!r}"
        )
        if not good:
            print(out)

    out_path = REPO / "bench" / "results" / "anchor-weight-sites-negative-control.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    shutil.rmtree(SCRATCH, ignore_errors=True)

    print()
    if ok:
        print(
            "NEGATIVE-CONTROL: PASS - the anchor weight-site checker goes red for every way "
            "issue #73 could be silently reintroduced, and names the defect it caught."
        )
        return EXIT_PASS
    print(
        "NEGATIVE-CONTROL: FAIL(condition=checker_did_not_detect_an_injected_defect) - a "
        "case above did not go red. Until it does, the checker's green is not evidence."
    )
    return EXIT_FAIL


if __name__ == "__main__":
    raise SystemExit(main())
