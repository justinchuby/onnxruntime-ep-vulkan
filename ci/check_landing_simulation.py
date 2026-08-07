#!/usr/bin/env python3
"""Run the landing simulation on the branches that can produce a landing-dependent verdict.

WHY THIS EXISTS — ISSUE #60
===========================
`ci/simulate_squash_rewitness.py` lands this branch three ways against the real base and
screens the census on each. It has existed since PR #44 and, until this file, **no workflow
ran it**: `ci.yml` ran `check_ledger_census.py` and its negative control and nothing else,
so the one instrument that can see a landing-dependent verdict was manual-only.

The residual it was needed for is narrow and real. A `rewitness/3` record declares its cause
as CONTENT — `(path, old, new)` sha256 over normalised bytes — corroborated against the two
trees the ledger moved between. That survives squash, merge, rebase and later unrelated
edits. It does **not** survive somebody landing an edit to a declared cause path on `main`
*between* the declaration being written and the declaring PR landing:

* the PR's own head is **PASS** — the `after` tree is the branch's ledger-carrying commit,
  which does not have the other edit;
* GitHub's synthetic two-parent merge ref, which is what CI checks out on a `pull_request`
  event, is **PASS** for the same reason: the `after` revision the origins walk finds is
  still a real branch commit;
* the **squash landing** is `FAIL(condition=uncorroborated_rewitness_cause)`, because it is
  one commit whose tree carries both edits, so the only revision carrying the declared `new`
  ledger value is a tree whose `q_gemv.comp` is no longer the declared `new` content.

Which means the red arrives on `main`, after the merge, in the `push: branches: [main]` job.
Reproduced on this repository: base `bb09871` plus a one-line edit to
`rust/shaders/glsl/templates/q_gemv.comp`, PR `8f12b32` — head PASS, merge PASS, squash
FAIL/exit 1.

WHY IT IS GATED, AND WHY THE GATE IS NOT A SKIP
===============================================
The simulation costs three landings, four censuses each, against a full clone. Running it on
every PR would buy nothing on the overwhelming majority of them, because the verdict CANNOT
be landing-dependent unless something the corroboration reads moved. So this decides first,
and prints the decision either way.

`REQUIRED` when any of:

  R1  the branch changes `evidence/proof_rewitness.json` or `evidence/proof_ledger.jsonl`
      — the declaration or the thing it declares about is being written in this change, and
      every landing-shape defect in this repository's history has been one of those;
  R2  the base has moved, since the merge base, any path a live `rewitness/3` record's
      corroboration reads — the declared cause paths, the source closure of the stems the
      moved entries name, or `shader_variants.txt`. **This is the #60 residual itself**;
  R3  the branch changes one of those same paths.

`NOT-REQUIRED` otherwise, and the reason is printed as an argument rather than asserted:
every input to the landing-sensitive part of the census is byte-identical in the base, the
merge base and this branch, so the squash tree, the merge tree and the branch tree agree on
all of them and no landing can disagree with another about them.

WHAT IS NEVER A SKIP
====================
* the base ref does not resolve, or has not been fetched -> `ERROR(instrument=base_unavailable)`,
  exit 2. A landing simulation with no landing target has not run;
* the checkout is shallow or partial -> `ERROR(instrument=shallow_checkout)`, exit 2. Both the
  landing construction and the census denominator are truncated at the graft boundary, so
  three greens would be three greens about a different repository. NOTE the `--depth` trap
  from issue #28: a depth-limited fetch marks its own graft point even in a `fetch-depth: 0`
  checkout, so fetch the base with no `--depth` at all;
* the register cannot be parsed -> `ERROR(instrument=register_unusable)`, exit 2, the same
  way `ci/check_ledger_census.py` treats it.

Usage:
    python ci/check_landing_simulation.py                       # decide, then run if required
    python ci/check_landing_simulation.py --base FETCH_HEAD
    python ci/check_landing_simulation.py --explain-only        # decide and print, never run
    python ci/check_landing_simulation.py --force               # run regardless of the decision
    python ci/check_landing_simulation.py --json out.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import check_ledger_census as census  # noqa: E402

PY = sys.executable

#: The two evidence files whose edit means this change is writing a declaration, or the thing
#: a declaration is about. Either one makes the landing shape a live question by itself.
DECLARATION_FILES = ("evidence/proof_rewitness.json", "evidence/proof_ledger.jsonl")


class GateInstrumentError(RuntimeError):
    """The gate could not observe its subject. Never a colour, always exit 2."""

    def __init__(self, token: str, detail: str):
        super().__init__(detail)
        self.token = token
        self.detail = detail


def _git(args: list[str], repo: Path = REPO) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=str(repo), capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )


def _rev(ref: str, repo: Path = REPO) -> str:
    r = _git(["rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"], repo)
    return r.stdout.strip()


def resolve_base(explicit: str | None, repo: Path = REPO) -> tuple[str, str]:
    """-> (the ref that was used, its sha). Raises rather than falling back to a guess.

    ORDER, AND WHY. An explicit `--base` wins, because the lane passes the ref it just
    fetched. Then `GITHUB_BASE_REF` — set only on `pull_request` events and naming the branch
    the PR would land on — tried as `origin/<ref>` and then bare. Then `origin/main`, then
    `main`. If none resolves this raises: a base that has to be guessed is a base that can be
    guessed wrong, and a simulation against the wrong tree is not a weaker simulation.
    """
    candidates: list[str] = []
    if explicit:
        candidates.append(explicit)
    else:
        env_base = os.environ.get("GITHUB_BASE_REF", "").strip()
        if env_base:
            candidates += [f"origin/{env_base}", env_base]
        candidates += ["origin/main", "main"]
    for ref in candidates:
        sha = _rev(ref, repo)
        if sha:
            return ref, sha
    raise GateInstrumentError(
        "base_unavailable",
        f"none of {candidates} resolves to a commit in {repo}. The landing target of this "
        "branch is the one thing this screen cannot infer, so it declines rather than "
        "simulate against whatever happens to be checked out. In CI, fetch it first — "
        "`git fetch --no-tags origin main` with NO --depth (issue #28) — and pass "
        "`--base FETCH_HEAD`.",
    )


def resolve_head(base_sha: str, repo: Path = REPO) -> tuple[str, str]:
    """-> (the PR head sha, how it was found).

    ON A `pull_request` EVENT THE CHECKOUT IS NOT THE BRANCH. GitHub checks out
    `refs/pull/N/merge`: a synthetic two-parent commit merging the PR head into the CURRENT
    tip of the base. Taking that as the branch head would make `merge-base(HEAD, base)` equal
    the base itself, so rule R2 — "has the base moved a path the corroboration reads" — would
    always compute an empty set and this gate would be vacuous on exactly the event it
    matters for. So when HEAD is a two-parent commit whose FIRST parent is the base (or an
    ancestor of it), the second parent is the real branch head and that is what is measured.
    """
    head = _rev("HEAD", repo)
    parents = _git(["rev-list", "--parents", "-n", "1", "HEAD"], repo).stdout.split()
    if len(parents) == 3:  # sha p1 p2
        p1, p2 = parents[1], parents[2]
        if p1 == base_sha or _git(
            ["merge-base", "--is-ancestor", p1, base_sha], repo
        ).returncode == 0:
            return p2, (
                f"HEAD {head[:12]} is a synthetic merge ref; its first parent {p1[:12]} is on "
                f"the base, so the branch head is its second parent {p2[:12]}"
            )
    return head, f"HEAD {head[:12]} is the branch head"


def _changed(a: str, b: str, repo: Path = REPO) -> set[str]:
    r = _git(["diff", "--name-only", f"{a}..{b}"], repo)
    if r.returncode != 0:
        raise GateInstrumentError(
            "diff_unavailable",
            f"`git diff --name-only {a[:12]}..{b[:12]}` failed: {r.stderr.strip()}",
        )
    return {line for line in r.stdout.splitlines() if line}


def _dirty(repo: Path = REPO) -> set[str]:
    """Paths that differ between HEAD and the working tree, tracked or not.

    The register is edited in the same change as the ledger, so a gate that could only read
    committed state would answer one commit too late — the same reason
    `simulate_squash_rewitness._prepare` commits the working tree into its clone.
    """
    r = _git(["status", "--porcelain", "-z", "--untracked-files=all"], repo)
    out: set[str] = set()
    for row in (r.stdout or "").split("\0"):
        if len(row) > 3:
            out.add(row[3:])
    return out


def corroboration_inputs(repo: Path, rev: str | None) -> tuple[set[str], list[str]]:
    """Every path a live `rewitness/3` corroboration READS, at `rev`. -> (paths, notes).

    Derived, not listed: the declared cause paths, plus the source closure of every stem the
    record's moved ledger entries name, re-resolved from the tree exactly as
    `ci/check_ledger_census.py` does (variant row -> template -> transitive `#include`), plus
    `shader_variants.txt`, whose own row is part of the closure test. Deriving it is the point
    — a hand-written list of "shader-ish paths" would go stale the first time `build.rs` moved
    a directory, and would go stale silently, which is the failure mode of every inventory in
    this repository that is not computed.
    """
    doc = census.load_rewitness(repo / "evidence" / "proof_rewitness.json")
    cache: dict = {}
    paths: set[str] = set()
    notes: list[str] = []
    n_v3 = 0
    for rec in doc.get("rewitness", []):
        if census._record_schema(rec) != census.SCHEMA_V3:
            continue
        n_v3 += 1
        for p in rec["caused_by_content"]["paths"]:
            paths.add(p["path"])
        ledger = census.present_entries(repo, rev)
        for t in rec["transitions"]:
            entry = ledger.get(t["key"])
            for stem in (entry or {}).get("shaders") or []:
                if not isinstance(stem, str):
                    continue
                closure, err = census.source_closure(repo, rev, stem, cache)
                if err:
                    notes.append(f"stem {stem!r}: {err}")
                paths |= closure
    if n_v3:
        paths.add(census.SHADER_VARIANTS_REL)
    notes.append(f"{n_v3} live-schema rewitness/3 record(s) in the register")
    return paths, notes


def decide(base_ref: str, base_sha: str, repo: Path = REPO) -> dict:
    """The whole decision, as data, so the negative control can assert on it."""
    head_sha, how = resolve_head(base_sha, repo)
    mb = _git(["merge-base", base_sha, head_sha], repo).stdout.strip()
    if not mb:
        raise GateInstrumentError(
            "no_merge_base",
            f"{head_sha[:12]} and {base_sha[:12]} have no merge base, so there is no landing "
            "to simulate and no way to tell what either side changed.",
        )
    on_base = _changed(mb, base_sha, repo)
    on_branch = _changed(mb, head_sha, repo) | _dirty(repo)
    inputs, notes = corroboration_inputs(repo, None)

    reasons: list[str] = []
    declaring = sorted(set(DECLARATION_FILES) & on_branch)
    if declaring:
        reasons.append(
            f"R1 this branch writes {declaring} — it is making or moving a declaration, and "
            "every landing-shape defect this repository has had was in one of those two files"
        )
    collided = sorted(inputs & on_base)
    if collided:
        reasons.append(
            f"R2 the base has moved {len(collided)} path(s) a live rewitness/3 corroboration "
            f"reads since the merge base: {collided[:5]}. THIS IS THE ISSUE #60 RESIDUAL — the "
            "branch head and the two-parent merge ref are both green on it and the squash is "
            "not"
        )
    touched = sorted(inputs & on_branch)
    if touched:
        reasons.append(
            f"R3 this branch changes {len(touched)} path(s) a live rewitness/3 corroboration "
            f"reads: {touched[:5]}"
        )
    return {
        "base_ref": base_ref,
        "base": base_sha,
        "head": head_sha,
        "head_note": how,
        "merge_base": mb,
        "base_moved_paths": len(on_base),
        "branch_moved_paths": len(on_branch),
        "corroboration_inputs": sorted(inputs),
        "notes": notes,
        "required": bool(reasons),
        "reasons": reasons,
    }


def _print_decision(d: dict) -> None:
    print("LANDING-SIMULATION GATE")
    print(f"  base   {d['base_ref']} = {d['base'][:12]}")
    print(f"  head   {d['head'][:12]}  ({d['head_note']})")
    print(f"  merge-base {d['merge_base'][:12]}")
    print(
        f"  {d['base_moved_paths']} path(s) moved on the base since the merge base; "
        f"{d['branch_moved_paths']} on this branch"
    )
    print(
        f"  {len(d['corroboration_inputs'])} path(s) are read by a live rewitness/3 "
        f"corroboration ({'; '.join(d['notes'])})"
    )
    if d["required"]:
        print("  VERDICT: REQUIRED")
        for r in d["reasons"]:
            print(f"    - {r}")
    else:
        print("  VERDICT: NOT-REQUIRED")
        print(
            "    Not a skip and not a budget decision: no path this screen's landing-sensitive\n"
            "    part reads differs between the merge base, the base tip and this branch, so\n"
            "    the squash tree, the merge tree and the branch tree carry identical bytes for\n"
            "    every one of them and no two landings can disagree about them. The register\n"
            "    and the ledger are unchanged here, so no declaration is being written either.\n"
            "    `ci/negative_control_landing_simulation.py` is what keeps this sentence\n"
            "    honest: it holds the same branch against a base that DID move a cause path\n"
            "    and requires this gate to say REQUIRED and the squash to go red."
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="")
    ap.add_argument("--repo", default=str(REPO))
    ap.add_argument("--explain-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--json", default="")
    ap.add_argument("--summary", default="")
    ap.add_argument(
        "--sim-arg", action="append", default=[],
        help="extra argument forwarded to ci/simulate_squash_rewitness.py",
    )
    args = ap.parse_args(argv)
    repo = Path(args.repo).resolve()

    complete, why = census.history_is_complete(repo)
    if not complete:
        print("LANDING-SIMULATION: ERROR(instrument=shallow_checkout)")
        print(f"  {why}")
        print(
            "  Both halves of this screen need the whole history: the landings are built by "
            "merging and replaying against a base, and the census that judges each landing "
            "takes its denominator from the full revision walk. A truncated history produces "
            "three greens about a repository this is not."
        )
        return 2

    try:
        base_ref, base_sha = resolve_base(args.base or None, repo)
        decision = decide(base_ref, base_sha, repo)
    except GateInstrumentError as exc:
        print(f"LANDING-SIMULATION: ERROR(instrument={exc.token})")
        print(f"  {exc.detail}")
        return 2
    except ValueError as exc:  # an unknown schema in the register
        print("LANDING-SIMULATION: ERROR(instrument=register_unusable)")
        print(f"  {exc}")
        return 2

    _print_decision(decision)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps(decision, indent=1) + "\n", encoding="utf-8"
        )
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as fh:
            fh.write(
                f"\n### Landing simulation: "
                f"{'REQUIRED' if decision['required'] else 'not required'}\n\n"
                + "".join(f"- {r}\n" for r in decision["reasons"])
            )

    if args.explain_only:
        return 0
    if not decision["required"] and not args.force:
        print("\nPASS (gate): no landing of this branch can disagree with another.")
        return 0

    print("\n-- running ci/simulate_squash_rewitness.py --base "
          f"{decision['base']} --")
    r = subprocess.run(
        [PY, str(HERE / "simulate_squash_rewitness.py"), "--base", decision["base"],
         *args.sim_arg],
        cwd=str(repo),
        env=dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1"),
    )
    if r.returncode == 0:
        print("\nPASS: every landing GitHub can build from this branch screens green.")
    elif r.returncode == 2:
        print("\nERROR(instrument): the landing simulation could not be built. Not a colour.")
    else:
        print(
            "\nFAIL(condition=landing_dependent_verdict): this branch screens differently "
            "depending on which merge button is pressed. Repair it BEFORE the merge — update "
            "the branch against the base, regenerate, and rewrite the record's `new` content "
            "ids — rather than after, on main, in a job nobody is watching."
        )
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
