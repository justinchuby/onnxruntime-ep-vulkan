# Negative-control driver for ci/check_tick_conversions.py.
#
# A gate that has only ever been observed passing is one step from a gate that cannot
# fail. This script injects each defect the screen claims to detect into a SCRATCH COPY
# of rust/src — never the real tree — and asserts the screen goes red with the right
# condition token and quotes the line it read.
#
# Output goes to bench/results/. Nothing here writes to the repository source.
#
#   python ci/negative_control_tick_conversions.py

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRATCH = REPO / "bench" / "results" / "tick-screen-negative"
SCREEN = REPO / "ci" / "check_tick_conversions.py"
ALLOWLIST = REPO / "ci" / "tick_conversion_allowlist.json"

EXIT_PASS, EXIT_FAIL, EXIT_ERROR = 0, 1, 4


def fresh_scratch() -> Path:
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    (SCRATCH / "rust").mkdir(parents=True)
    shutil.copytree(REPO / "rust" / "src", SCRATCH / "rust" / "src")
    shutil.copy2(ALLOWLIST, SCRATCH / "allowlist.json")
    return SCRATCH


def run(root: Path, allowlist: Path) -> tuple[int, str]:
    p = subprocess.run(
        [sys.executable, str(SCREEN), "--root", str(root), "--allowlist", str(allowlist)],
        capture_output=True,
        text=True,
    )
    return p.returncode, p.stdout + p.stderr


def inject(root: Path, rel: str, anchor: str, addition: str) -> None:
    path = root / rel
    text = path.read_text(encoding="utf-8")
    if anchor not in text:
        raise SystemExit(
            f"NEGATIVE-CONTROL: ERROR(instrument=anchor_not_found) — {anchor!r} is not in "
            f"{rel}. The control cannot inject its defect, so it is asserting nothing. "
            "Re-anchor it; do not read this as a passing control."
        )
    path.write_text(text.replace(anchor, anchor + "\n" + addition, 1), encoding="utf-8")


CASES = [
    {
        "name": "the 52x defect itself — a tick delta used as nanoseconds",
        "rel": "rust/src/vk/session.rs",
        "anchor": "            let results = unsafe { qp.read_results() };",
        "addition": "            let bypass_ns = end_ticks - begin_ticks;",
        "expect_condition": "tick_conversion_bypassed",
        "expect_quote": "bypass_ns = end_ticks - begin_ticks",
    },
    {
        "name": "the same defect laundered through a rename",
        "rel": "rust/src/vk/session.rs",
        "anchor": "            let results = unsafe { qp.read_results() };",
        "addition": "            let raw_span = end_ticks;",
        "expect_condition": "tick_conversion_bypassed",
        "expect_quote": "raw_span = end_ticks",
    },
    {
        "name": "by-hand period multiply that skips the valid-bit mask",
        "rel": "rust/src/trace.rs",
        "anchor": "        let mut emitted = 0usize;",
        "addition": "        let hand_ns = iv.begin_ticks as f64 * 1.0;",
        "expect_condition": "tick_conversion_bypassed",
        "expect_quote": "hand_ns = iv.begin_ticks as f64",
    },
    {
        "name": "a second reader of raw, unmasked ticks",
        "rel": "rust/src/trace.rs",
        "anchor": "        let mut emitted = 0usize;",
        "addition": "        let _second = pool.read_results();",
        "expect_condition": "raw_tick_producer_not_unique",
        "expect_quote": "is called from 2 sites",
    },
]


def main() -> int:
    results = []
    ok = True

    # Arm 0: the unmodified scratch copy must be green, or every red below proves nothing.
    root = fresh_scratch()
    code, out = run(root, root / "allowlist.json")
    baseline_ok = code == EXIT_PASS
    results.append({"case": "baseline (unmodified copy)", "exit": code, "ok": baseline_ok})
    print(f"[{'ok' if baseline_ok else 'BAD'}] baseline scratch copy → exit {code}")
    if not baseline_ok:
        print(out)
        print(
            "NEGATIVE-CONTROL: ERROR(instrument=baseline_not_green) — the unmodified copy "
            "is already red, so no injection below can be attributed to the injection."
        )
        return EXIT_ERROR

    for case in CASES:
        root = fresh_scratch()
        inject(root, case["rel"], case["anchor"], case["addition"])
        code, out = run(root, root / "allowlist.json")
        cond_ok = f"FAIL(condition={case['expect_condition']})" in out
        quote_ok = case["expect_quote"] in out
        good = code == EXIT_FAIL and cond_ok and quote_ok
        ok &= good
        results.append(
            {
                "case": case["name"],
                "injected": case["addition"].strip(),
                "exit": code,
                "condition_token_seen": cond_ok,
                "offending_line_quoted": quote_ok,
                "ok": good,
            }
        )
        print(
            f"[{'ok' if good else 'BAD'}] {case['name']}\n"
            f"        injected: {case['addition'].strip()}\n"
            f"        exit={code} condition={cond_ok} quoted={quote_ok}"
        )
        if not good:
            print(out)

    # Arm: allowlist rot. An exemption that has lost its site must be reported, or the
    # allowlist decays into a blanket over whatever moved into its place.
    root = fresh_scratch()
    doc = json.loads((root / "allowlist.json").read_text(encoding="utf-8"))
    doc["sanctioned_sites"][0]["contains"] = "a line that does not exist anywhere"
    (root / "allowlist.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    code, out = run(root, root / "allowlist.json")
    rot_ok = code == EXIT_FAIL and (
        "allowlist_entry_without_a_site" in out or "tick_conversion_bypassed" in out
    )
    ok &= rot_ok
    results.append({"case": "allowlist entry that has lost its site", "exit": code, "ok": rot_ok})
    print(f"[{'ok' if rot_ok else 'BAD'}] allowlist entry that has lost its site → exit {code}")
    if not rot_ok:
        print(out)

    # Arm: instrument outage must be an ERROR, not a FAIL and not a pass.
    root = fresh_scratch()
    code, out = run(root, root / "no-such-allowlist.json")
    outage_ok = code == EXIT_ERROR and "ERROR(instrument=allowlist_unreadable)" in out
    ok &= outage_ok
    results.append({"case": "missing allowlist → ERROR(instrument)", "exit": code, "ok": outage_ok})
    print(f"[{'ok' if outage_ok else 'BAD'}] missing allowlist → exit {code} (want 4)")
    if not outage_ok:
        print(out)

    # Arm: an empty frame must be an ERROR, not a pass. R12 — no sites found is
    # UNOBSERVABLE, never zero findings.
    root = fresh_scratch()
    for f in (root / "rust" / "src").rglob("*.rs"):
        f.write_text("// emptied by the negative control\n", encoding="utf-8")
    code, out = run(root, root / "allowlist.json")
    empty_ok = code == EXIT_ERROR and "no_tick_sites_found" in out
    ok &= empty_ok
    results.append({"case": "empty frame → ERROR(instrument)", "exit": code, "ok": empty_ok})
    print(f"[{'ok' if empty_ok else 'BAD'}] emptied source tree → exit {code} (want 4)")
    if not empty_ok:
        print(out)

    out_path = REPO / "bench" / "results" / "tick-screen-negative-control.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    shutil.rmtree(SCRATCH, ignore_errors=True)

    print()
    if ok:
        print(
            "NEGATIVE-CONTROL: PASS — the screen goes red for each defect it claims to "
            "detect, quotes the line it read, and reports an instrument outage as "
            "ERROR rather than as either a detection or a clean tree."
        )
        return EXIT_PASS
    print(
        "NEGATIVE-CONTROL: FAIL(condition=screen_did_not_detect_an_injected_bypass) — a "
        "case above did not go red. Until it does, the screen's green is not evidence."
    )
    return EXIT_FAIL


if __name__ == "__main__":
    raise SystemExit(main())
