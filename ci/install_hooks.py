#!/usr/bin/env python3
"""Point this checkout's git hooks at the tracked `.githooks/` directory.

WHY THIS IS A SCRIPT AND NOT A README LINE
==========================================
`.git/hooks` is not versioned, so a hook that lives there is a fact about one person's
machine. `core.hooksPath` makes the hooks a tracked file — but the config itself is still
per-clone, which means the honest thing this script can do is *tell you whether it is set
right now*, and say so out loud when it is not.

That distinction is the whole subject of this session. The README already said a red badge
blocks merges "by team discipline enforced by this notice", and the notice was true, and
ten merges happened anyway. A notice is read once; a hook is read every time. But a hook
that is only installed on the machine of the person who wrote it is a notice again.

So: `--check` returns non-zero when the hooks are not installed, which makes "are the
hooks on?" a question something can ask, rather than a thing everyone assumes.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HOOKS_REL = ".githooks"


def _git(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=str(REPO), capture_output=True, text=True, encoding="utf-8"
    )


def current() -> str:
    return _git(["config", "--get", "core.hooksPath"]).stdout.strip()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="report without changing anything; non-zero if not installed")
    args = ap.parse_args(argv)

    hooks = REPO / HOOKS_REL
    installed = [p.name for p in sorted(hooks.iterdir())] if hooks.is_dir() else []
    got = current()

    if args.check:
        if got == HOOKS_REL:
            print(f"HOOKS: installed — core.hooksPath={got} ({', '.join(installed)})")
            return 0
        print(f"HOOKS: NOT installed — core.hooksPath={got or '<unset>'}")
        print(
            "  The tracked hooks in .githooks/ are not running in this clone. The "
            "pre-merge-commit hook is the thing that reads the colour of `main` before a "
            "merge is written; without it, that reading is back to being something "
            "somebody remembers to do, which is the arrangement that produced ten "
            "consecutive unremarked red pushes. Run: python ci/install_hooks.py"
        )
        return 1

    if not hooks.is_dir():
        print(f"ERROR(instrument): {hooks} does not exist")
        return 4
    r = _git(["config", "core.hooksPath", HOOKS_REL])
    if r.returncode != 0:
        print(f"ERROR(instrument): git config failed: {r.stderr.strip()}")
        return 4
    for p in sorted(hooks.iterdir()):
        try:
            p.chmod(p.stat().st_mode | 0o111)
        except OSError:
            pass
    print(f"HOOKS: core.hooksPath={HOOKS_REL} — installed: {', '.join(installed)}")
    print(
        "  `git merge` will now read the colour of `main` before it writes the merge "
        "commit. MAIN_COLOUR_ACK=red merges into a red branch on the record; "
        "MAIN_COLOUR_ACK=unread merges without having asked. Both are allowed and "
        "neither is silent."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
