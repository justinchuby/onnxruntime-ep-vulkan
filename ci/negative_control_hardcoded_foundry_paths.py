#!/usr/bin/env python3
"""Negative control for ci/check_hardcoded_foundry_paths.py — every arm in its POSITIVE state.

A screen only ever observed green is indistinguishable from a constant that returns green.
Each arm below runs the real screen against a tree with the defect genuinely present.

  REPLAYED  the exact ``bench/exec_census.py`` this repository shipped at commit
            ``ea427fd`` (immediately before issue #19's migration), byte for byte, placed
            outside the archival allowlist exactly as it was shipped. The defect actually
            happened; this is not a shape I invented.
  PLANTED   mutations written on purpose, to cover shapes the replay does not: a fresh file
            hardcoding the pattern, a file the allowlist directory prefix legitimately
            covers, and a clean tree with none at all.
  LIVE      the real tree in this checkout, right now, which must be green.

Run:  python ci/negative_control_hardcoded_foundry_paths.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SCREEN = HERE / "check_hardcoded_foundry_paths.py"

PASS, FAIL = "PASS", "FAIL"
UNOBS = "UNOB"
results: list[tuple[str, str, str, str]] = []


def run_screen(root: Path) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, str(SCREEN), "--root", str(root)],
        capture_output=True, text=True, cwd=str(REPO),
    )
    return proc.returncode, proc.stdout + proc.stderr


def arm(name: str, kind: str, files: dict[str, str] | None, expect_code: int, expect_text: str):
    if files is None:
        code, out = run_screen(REPO)
    else:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for rel, content in files.items():
                p = root / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content, encoding="utf-8")
            code, out = run_screen(root)
    ok = code == expect_code and expect_text in out
    results.append((name, kind, PASS if ok else FAIL, f"exit={code} expected={expect_code}"))
    if not ok:
        print(f"--- {name} output ---\n{out}\n")


def historical_file(ref: str, rel: str) -> str | None:
    """A file's real content at *ref*, or None if that ref/path is not in this clone."""
    proc = subprocess.run(
        ["git", "show", f"{ref}:{rel}"],
        capture_output=True, text=True, cwd=str(REPO),
    )
    return proc.stdout if proc.returncode == 0 else None


#: The commit immediately before issue #19's migration landed. bench/exec_census.py at this
#: ref hardcodes the Foundry cache path directly, outside any allowlisted directory.
HISTORICAL_REF = "ea427fd"
HISTORICAL_REL = "bench/exec_census.py"

_historical = historical_file(HISTORICAL_REF, HISTORICAL_REL)
if _historical is None:
    # A shallow clone (actions/checkout defaults to depth 1) does not contain the ref.
    # That is an absent observation, not a failing one: reporting FAIL would make the
    # lane red for a property of the checkout rather than a property of the screen.
    results.append((
        f"the real {HISTORICAL_REL} at {HISTORICAL_REF} (before issue #19's migration)",
        "REPLAYED", UNOBS, f"{HISTORICAL_REF} not in this clone (shallow?)",
    ))
else:
    arm(
        f"the real {HISTORICAL_REL} at {HISTORICAL_REF} (before issue #19's migration)",
        "REPLAYED",
        {HISTORICAL_REL: _historical},
        1,
        "hardcoded Foundry cache path(s) outside the allowlist",
    )

arm(
    "a fresh live tool that pastes the hardcoded pattern back in",
    "PLANTED",
    {
        "rust/tools/probe_new_thing.py": (
            'MODEL = r"C:\\Users\\someone\\.foundry\\cache\\models\\Microsoft\\Foo"\n'
        ),
    },
    1,
    "rust/tools/probe_new_thing.py",
)

arm(
    "an archival script nested under bench/results/ is allowlisted, not flagged",
    "PLANTED",
    {
        "bench/results/probe_something.py": (
            'MODEL = r"C:\\Users\\someone\\.foundry\\cache\\models\\Microsoft\\Foo"\n'
        ),
    },
    0,
    "FOUNDRY-PATHS: PASS",
)

arm(
    "a clean tree with no occurrences at all",
    "PLANTED",
    {"bench/exec_census.py": "print('nothing to see here')\n"},
    0,
    "0 allowlisted occurrence(s), 0 outside the allowlist",
)

arm("the real tree in this checkout, right now", "LIVE", None, 0, "FOUNDRY-PATHS: PASS")


width = max(len(n) for n, _, _, _ in results)
print()
for name, kind, verdict, detail in results:
    print(f"  {verdict:<4}  {kind:<9}  {name:<{width}}  {detail}")
planted = sum(1 for _, k, _, _ in results if k == "PLANTED")
unobs = sum(1 for _, _, v, _ in results if v == UNOBS)
print(f"\n{sum(1 for _, _, v, _ in results if v == PASS)}/{len(results)} arms pass "
      f"({planted} PLANTED — a planted arm proves the rule fires on the shape it was "
      f"written for, not that the rule is load-bearing"
      + (f"; {unobs} UNOBSERVED" if unobs else "") + ")")
if unobs:
    print("  NOTE: the strongest arm (the real historical file) did not run in this "
          "checkout. The remaining arms are planted and must not be read as a replay.")
sys.exit(0 if all(v != FAIL for _, _, v, _ in results) else 1)
