#!/usr/bin/env python3
"""Negative control for ci/check_readme_usage.py — every arm in its POSITIVE state.

A screen only ever observed green is indistinguishable from a constant that returns green.
Each arm below runs the real screen against a README with the defect genuinely present.

  REPLAYED  the exact usage block this repository shipped for months, byte for byte. The
            defect actually happened; this is not a shape I invented.
  PLANTED   a mutation written on purpose. Proves the rule fires on the shape it was
            written for; does NOT prove the rule is load-bearing.
  LIVE      the real README in this tree, right now, which must be green.

Run:  python ci/negative_control_readme_usage.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SCREEN = HERE / "check_readme_usage.py"

PASS, FAIL = "PASS", "FAIL"
UNOBS = "UNOB"
results: list[tuple[str, str, str, str]] = []


def run_screen(readme: Path, tree: Path) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, str(SCREEN), "--readme", str(readme), "--tree", str(tree)],
        capture_output=True, text=True, cwd=str(REPO),
    )
    return proc.returncode, proc.stdout + proc.stderr


def arm(name: str, kind: str, content: str | None, expect_code: int, expect_text: str):
    if content is None:
        code, out = run_screen(REPO / "README.md", REPO)
    else:
        with tempfile.TemporaryDirectory() as td:
            readme = Path(td) / "README.md"
            readme.write_text(content, encoding="utf-8")
            # Judged against an empty tree: a planted README must be judged on its own
            # terms, not rescued by a module that happens to exist in this checkout.
            code, out = run_screen(readme, Path(td))
    ok = code == expect_code and expect_text in out
    results.append((name, kind, PASS if ok else FAIL,
                    f"exit={code} expected={expect_code}"))
    if not ok:
        print(f"--- {name} output ---\n{out}\n")


def historical_readme(ref: str) -> str | None:
    """The README as it actually was at *ref*, or None if that ref is not in this clone."""
    proc = subprocess.run(
        ["git", "show", f"{ref}:README.md"],
        capture_output=True, text=True, cwd=str(REPO),
    )
    return proc.stdout if proc.returncode == 0 else None


#: `main` immediately before the Python shim landed. Its README documented an import of
#: `onnxruntime_ep_vulkan`, and `git ls-files | grep onnxruntime_ep_vulkan` at that commit
#: returned nothing. Real bytes, real defect.
HISTORICAL_REF = "8a851f8"

_historical = historical_readme(HISTORICAL_REF)
if _historical is None:
    # A shallow clone (actions/checkout defaults to depth 1) does not contain the ref.
    # That is an absent observation, not a failing one: reporting FAIL would make the
    # lane red for a property of the checkout rather than a property of the README. It is
    # printed loudly so nobody reads the remaining arms as if the replay had run.
    results.append((
        f"the real README at {HISTORICAL_REF} (before the shim landed)",
        "REPLAYED", UNOBS, f"{HISTORICAL_REF} not in this clone (shallow?)",
    ))
else:
    # The screen resolves first-party modules from this tree, which now HAS the package.
    # Point it at an empty tree so the historical README is judged against the historical
    # state -- otherwise this arm would pass for the wrong reason.
    import os as _os

    with tempfile.TemporaryDirectory() as _td:
        _readme = Path(_td) / "README.md"
        _readme.write_text(_historical, encoding="utf-8")
        _proc = subprocess.run(
            [sys.executable, str(SCREEN), "--readme", str(_readme), "--tree", _td],
            capture_output=True, text=True, cwd=_td,
            env={**_os.environ, "PYTHONPATH": ""},
        )
        _ok = _proc.returncode == 1 and "onnxruntime_ep_vulkan" in _proc.stdout
    results.append((
        f"the real README at {HISTORICAL_REF} (before the shim landed)",
        "REPLAYED", PASS if _ok else FAIL, f"exit={_proc.returncode} expected=1",
    ))
    if not _ok:
        print(f"--- historical arm output ---\n{_proc.stdout}{_proc.stderr}\n")


# The block this repository actually shipped. `onnxruntime_ep_vulkan` did not exist.
REPLAYED_BLOCK = """# onnxruntime-ep-vulkan

## Intended usage

```python
import onnxruntime as ort
import onnxruntime_ep_vulkan

onnxruntime_ep_vulkan.register_execution_provider_library()
sess = ort.InferenceSession(
    model,
    providers=["VulkanExecutionProvider", "CPUExecutionProvider"],
)
```
"""

arm(
    "the shipped usage block, verbatim",
    "REPLAYED",
    REPLAYED_BLOCK,
    1,
    "documented import(s) name nothing that exists",
)

arm(
    "from-import of a missing module",
    "PLANTED",
    "```python\nfrom totally_absent_module import thing\n```\n",
    1,
    "totally_absent_module",
)

arm(
    "a README with no python block verifies nothing and must not read as a pass",
    "PLANTED",
    "# title\n\n```powershell\ncargo build --release\n```\n",
    4,
    "no_python_blocks",
)

arm(
    "a python block that imports nothing verifies nothing",
    "PLANTED",
    "```python\nx = 1 + 1\n```\n",
    4,
    "no_imports_found",
)

arm(
    "a block that does not parse is ignored, not reported as missing",
    "PLANTED",
    "```python\nimport onnxruntime as ort\n```\n\n```python\n<not python at all>\n```\n",
    0,
    "PASS",
)

arm("the real README in this tree", "LIVE", None, 0, "README-USAGE: PASS")


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
    print("  NOTE: the strongest arm (the real historical README) did not run in this "
          "checkout. The remaining arms are planted and must not be read as a replay.")
sys.exit(0 if all(v != FAIL for _, _, v, _ in results) else 1)
