#!/usr/bin/env pwsh
# Registers the `squad-history` git merge driver locally.
#
# git merge drivers are declared in .gitattributes (which file uses which
# driver) but the driver COMMAND ITSELF must live in .git/config, which is
# never versioned. Every clone/worktree (including scribe-1/scribe-2's
# shared worktree) must run this once, or `.squad/agents/*/history.md` and
# `.squad/decisions.md` will silently fall back to git's default merge
# strategy for unregistered driver names, which raises real conflicts on
# concurrent agent appends -- the exact failure `merge=union` was chosen to
# avoid. Re-running this script is safe (idempotent).
$ErrorActionPreference = "Stop"
$repoRoot = git rev-parse --show-toplevel
if (-not $repoRoot) { throw "not inside a git repository" }

git config merge.squad-history.name "Squad append-only history merge (condensation-safe)"
git config merge.squad-history.driver "python `"$repoRoot/.squad/tools/history_merge_driver.py`" %O %A %B %P"

Write-Output "Registered merge.squad-history driver in $repoRoot/.git/config"
