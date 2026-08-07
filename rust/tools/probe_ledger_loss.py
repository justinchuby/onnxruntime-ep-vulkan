#!/usr/bin/env python
"""Does `--check` notice a proof that went *missing*? Owner: tank (was: Mouse).

WHY THIS EXISTS
---------------
`gen_proof_ledger.py --check` asks whether every entry that is present agrees with the build.
It has never been able to ask whether one went **absent**, and the difference is not academic:
on 2026-08-03 a merge deleted three proofs from `evidence/proof_ledger.jsonl` and `--check`
printed PASS over the result. It was not wrong by its own lights — 103 sound entries are 103
sound entries. The only instrument that saw the loss was the op suite, three steps downstream,
and `git log` on the file did not show it because history simplification hid the removal inside
the merge.

`check_no_proof_went_missing` closes that by comparing the ledger against the append-only
attempt log. This probe is its falsifier, and the load-bearing arm is **arm 3**: the two
artifacts as they actually stood at `eb84364`, the commit the incident produced. An invariant
that fires on a synthetic mutation and has never been shown against the real event is a guess.

ARMS (predictions written before the run)
-----------------------------------------
1. current tree                          -> PASS, 0 missing        (the invariant is not noisy)
2. one entry deleted from the ledger      -> FAIL naming that key   (it detects a deletion)
3. the real `eb84364` pair                -> FAIL naming the 3 Cast keys the merge lost
4. arm 2 + that key retired with a reason -> PASS                   (retirement is the escape)
5. arm 1 + a retirement for a key that is
   still in the ledger                    -> FAIL                   (a stale exemption fails)
5b. arm 2 + a retirement with no owner
   and no date                            -> ERROR(instrument)      (one register, ONE rule:
                                                                     the census refused this
                                                                     record while this tool
                                                                     honoured it, until the
                                                                     parse was unified)
6. attempt log absent                     -> ERROR(instrument), never PASS

NO CLOCK. Set/count comparisons only.

WHERE THIS PROBE IS ALLOWED TO WRITE, AND WHY IT IS NOT A DETAIL (issue #14)
---------------------------------------------------------------------------
Until 2026-08-06 `main()` wrote its six working files **and** a `result.json` straight into
`bench/results/_probe_ledger_loss/`, a tracked directory. Three consequences, all observed:

* Running the diagnostic during a read-only baseline dirtied `main`. A diagnostic that cannot
  be run without leaving a tracked diff is a diagnostic people stop running.
* The tracked `result.json` was a READING with no owner and no frame. It was written once, on
  a different checkout (its arm details still named an absolute path under another user's home
  directory), and `bfdc0f1` later retired two Conv metadata-form keys — so the committed
  `pass=true` went stale and *nothing* detected it, because the probe appeared in no workflow,
  in no `ci/open_reds.json` entry and under no artifact-frame check.
* The register files it wrote (`register_now.json`, `retire_one.json`, `retire_unsigned.json`)
  are **deletion-bearing**: they are retirement registers, and a retirement is the removal of a
  claim's exemption. Union-merging such a file resurrects a withdrawn key with nobody's
  signature on it, which is why `.gitattributes` gives `evidence/` no merge driver at all. A
  probe that writes register-shaped files into a tracked path is one merge away from the same
  defect one directory over.

THE CONTRACT NOW
    * **Default** — everything, including `result.json`, goes to an ephemeral scratch directory
      created outside the repository and removed on exit. The verdict is on stdout. Nothing
      inside any checkout is created, modified or deleted. This is what CI runs.
    * **`--out DIR`** — the caller owns `DIR` and it is kept. If `DIR` resolves inside the
      repository and git does not ignore it, the probe REFUSES with
      `ERROR(instrument=refused_tracked_destination)` and writes nothing: an accidental
      canonical write is not an observation, so it is never a PASS.
    * **`--out DIR --record`** — the explicit, auditable recording mode, and the ONLY way to put
      a reading into a tracked path. It is not merely permission: the recorded `result.json`
      carries `owner`, `tool`, `produced_at_commit` and a `subject` block digesting the three
      evidence files the reading is ABOUT, and the directory is stamped with an
      `artifact-frame.json` by `ci/check_artifact_frame.py`, so the reading can go stale
      DETECTABLY instead of silently.

PROVENANCE AND DETERMINISM (both are asserted, not intended)
    * Every path in the record is repo-relative with forward slashes. Absolute paths are
      scrubbed to `<repo>`/`<out>` and a leftover absolute path is a hard
      `ERROR(instrument=provenance_leaked_absolute_path)` — that is the exact defect the
      committed reading carried.
    * Bytes are deterministic: sorted keys, `ensure_ascii=True`, LF newlines written as bytes
      (no platform newline translation), no clock anywhere, and the subject digests are taken
      through the same line-ending normalisation `ci/check_artifact_frame.py` uses, so a
      Windows checkout with `core.autocrlf=true` produces the same record as a Linux one.

OWNERSHIP OF THE ARTIFACT — one model, stated once
    **This probe is EXECUTED, not committed.** Its canonical evidence is the run in the
    host-free `lane-checks` job (`.github/workflows/ci.yml`), declared `expect: green` as
    `ledger_loss_probe` in `ci/open_reds.json` and classified as `hostfree.ledger_loss_probe`
    in `ci/lane_inventory.py`. Owner: **tank**. There is deliberately NO tracked reading: a
    committed reading of a check that runs on every push is a second, staler answer to a
    question already being asked live. `--record` exists for a deliberate archival reading, and
    anything it writes must carry the frame above, so a recorded artifact can never again be an
    unowned `pass=true` nobody can date.

USAGE
    python rust/tools/probe_ledger_loss.py                        # ephemeral scratch, no tree writes
    python rust/tools/probe_ledger_loss.py --out ../probe-scratch # caller-owned, kept
    python rust/tools/probe_ledger_loss.py --out bench/results/_probe_ledger_loss --record
    python rust/tools/probe_ledger_loss.py --repo /path/to/checkout   # audit another checkout

EXIT CODES (R13)
    0 PASS   1 FAIL(condition)   2 usage   4 ERROR(instrument)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

TOOL_DIR = pathlib.Path(__file__).resolve().parent
REPO = TOOL_DIR.parents[1]
sys.path.insert(0, str(TOOL_DIR))

import gen_proof_ledger as gpl  # noqa: E402

TOOL_REL = "rust/tools/probe_ledger_loss.py"
OWNER = "tank"
RECORD_SCHEMA = 2

INCIDENT_COMMIT = "eb84364"
INCIDENT_KEYS = {
    "ai.onnx::Cast/6+/f32>bool/ew_cast_f32_to_bool/static/n1",
    "ai.onnx::Cast/6+/f32>i32/ew_cast_f32_to_i32/static/n1",
    "ai.onnx::Cast/6+/i32>f32/ew_cast_i32_to_f32/static/n1",
}

LEDGER_REL = "evidence/proof_ledger.jsonl"
ATTEMPTS_REL = "evidence/proof_attempts.jsonl"
REGISTER_REL = "evidence/retired_proof_keys.json"

#: The paths a recorded reading is ABOUT, handed to `ci/check_artifact_frame.py --stamp` so a
#: recorded artifact goes stale detectably rather than silently. The tool itself is in the list
#: on purpose: a change to the arms changes what `pass=true` means.
SUBJECT_PATHS = (LEDGER_REL, ATTEMPTS_REL, REGISTER_REL, TOOL_REL)

EXIT_PASS = 0
EXIT_FAIL_CONDITION = 1
EXIT_USAGE = 2
EXIT_ERROR_INSTRUMENT = 4

#: A drive-letter path, or a POSIX absolute path into a home/tmp/mount root. Deliberately not
#: "any string starting with /": the arm details legitimately contain repo-relative POSIX paths.
_ABSOLUTE_PATH_RE = re.compile(
    r"(?:[A-Za-z]:[\\/])|(?:(?:^|[\s\"'(=])/(?:home|Users|users|mnt|tmp|var|opt|root)/)"
)


def _normalize_text(data: bytes) -> bytes:
    """CRLF and a lone CR both become LF — the same transform `ci/check_artifact_frame.py` uses.

    A Windows checkout with `core.autocrlf=true` rewrites tracked text files on the way to disk.
    Digesting raw bytes would make this record platform-dependent and every cross-platform
    comparison of it a false difference.
    """
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def _digest(path: pathlib.Path) -> str:
    if not path.is_file():
        return ""
    return hashlib.sha256(_normalize_text(path.read_bytes())).hexdigest()


def _git(args: list[str], cwd: pathlib.Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, encoding="utf-8", errors="replace"
    )


def _ledger_keys(text: str) -> set[str]:
    keys = set()
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        rec = json.loads(raw)
        if "__ledger__" in rec:
            continue
        keys.add(rec["key"])
    return keys


def _evaluate(ledger_keys: set[str], attempts: pathlib.Path, retired: pathlib.Path):
    """Run the invariant against an explicit pair of artifacts."""
    saved_a, saved_r = gpl.ATTEMPTS, gpl.RETIRED
    gpl.ATTEMPTS, gpl.RETIRED = attempts, retired
    try:
        return gpl.check_no_proof_went_missing(ledger_keys)
    finally:
        gpl.ATTEMPTS, gpl.RETIRED = saved_a, saved_r


def _git_show(rev: str, path: str, dest: pathlib.Path, repo: pathlib.Path) -> bool:
    r = _git(["show", f"{rev}:{path}"], repo)
    if r.returncode != 0:
        return False
    dest.write_text(r.stdout, encoding="utf-8")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Destination policy — the half of this file issue #14 is actually about
# ──────────────────────────────────────────────────────────────────────────────

DEST_OUTSIDE = "outside_repository"
DEST_IGNORED = "inside_repository_but_git_ignored"
DEST_TRACKED_SURFACE = "inside_repository_tracked_surface"


def classify_destination(repo: pathlib.Path, out: pathlib.Path) -> str:
    """Where does `out` sit relative to `repo`'s tracked surface? Fails CLOSED.

    `git check-ignore` is the only authority consulted, because a hand-written path list here
    would be a second answer to "what is tracked" and the first one is what `git status` reads.
    An unresolvable answer (not a git repository, git absent) is TRACKED_SURFACE, not OUTSIDE:
    a destination policy that opens up when its instrument breaks is not a policy.
    """
    try:
        out.resolve().relative_to(repo.resolve())
    except ValueError:
        return DEST_OUTSIDE
    r = _git(["check-ignore", "-q", "--", str(out.resolve())], repo)
    if r.returncode == 0:
        return DEST_IGNORED
    return DEST_TRACKED_SURFACE


# ──────────────────────────────────────────────────────────────────────────────
# Provenance scrubbing
# ──────────────────────────────────────────────────────────────────────────────


def _scrub(text: str, repo: pathlib.Path, out: pathlib.Path) -> str:
    """Replace machine-specific absolute paths with `<repo>` / `<out>`, longest root first.

    `out` is substituted before `repo` when it is the longer string, because a `--record`
    destination sits *inside* the repository and substituting the shorter root first would leave
    `<repo>/bench/...` in one mode and `<out>` in the other — two spellings of one reading.
    """
    roots = [(str(out.resolve()), "<out>"), (str(repo.resolve()), "<repo>")]
    roots.sort(key=lambda pair: len(pair[0]), reverse=True)
    for root, token in roots:
        for variant in {root, root.replace("\\", "/"), root.replace("/", os.sep)}:
            if not variant:
                continue
            text = re.sub(re.escape(variant), token, text, flags=re.IGNORECASE)
    # Windows separators inside a scrubbed path are noise the record must not carry: two
    # platforms writing the same reading must produce the same bytes.
    return re.sub(
        r"(<repo>|<out>)((?:\\[^\s\"']*)+)",
        lambda m: m.group(1) + m.group(2).replace("\\", "/"),
        text,
    )


def _leaked_absolute_path(record: dict) -> str:
    """Return a window around the first absolute path still in the record, or "" if clean."""
    blob = json.dumps(record, sort_keys=True)
    hit = _ABSOLUTE_PATH_RE.search(blob)
    return blob[max(0, hit.start() - 30) : hit.end() + 50] if hit else ""


def _serialize(record: dict) -> bytes:
    """Deterministic bytes: sorted keys, ASCII-escaped, LF, no clock, no platform newline."""
    return (json.dumps(record, indent=1, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# The arms
# ──────────────────────────────────────────────────────────────────────────────


def run_arms(repo: pathlib.Path, scratch: pathlib.Path) -> list[tuple[str, bool, str]]:
    """Every arm, evaluated against `repo`, with every working file written under `scratch`."""
    ledger = repo / LEDGER_REL
    attempts_now = repo / ATTEMPTS_REL
    register = repo / REGISTER_REL

    if not ledger.is_file():
        # Not "0 arms passed": a probe that cannot reach its subject reports the outage rather
        # than a verdict about it (R13).
        return [
            (
                "0 the ledger this probe is about is readable",
                False,
                f"ERROR(instrument): no {LEDGER_REL} under the checkout being probed",
            )
        ]

    no_retire = scratch / "none.json"
    no_retire.write_bytes(_serialize({"retired": []}))

    # TODAY'S ARMS READ TODAY'S REGISTER, AND THAT IS NOT A CONVENIENCE.
    #
    # Arms 1, 2, 4 and 5 ask about the tree as it stands, and the tree as it stands has 43 keys
    # deliberately withdrawn by the §8.9.23 schema change. Handing those arms an EMPTY register
    # asked the invariant a question about a repository that does not exist: all 43 read as lost,
    # arm 1 ("a guard that fails on a healthy tree gets switched off") was red for days, and arms
    # 2/4/5 could no longer count their own failures. The empty register stays where it belongs —
    # arm 3, the historical replay, where nothing had been retired yet and where applying today's
    # withdrawals to yesterday's ledger would convict the right revision for the wrong reason.
    base_rows = (
        json.loads(register.read_text(encoding="utf-8"))["retired"] if register.is_file() else []
    )
    retired_now = scratch / "register_now.json"
    retired_now.write_bytes(_serialize({"retired": base_rows}))

    results: list[tuple[str, bool, str]] = []
    live_keys = _ledger_keys(ledger.read_text(encoding="utf-8"))

    # Arm 1 — the tree as it stands. A guard that fails on a healthy tree gets switched off.
    fails, _notes, err = _evaluate(live_keys, attempts_now, retired_now)
    results.append(("1 current tree is clean", not err and not fails, err or "; ".join(fails[:2])))

    # Arm 2 — delete one entry. This is the mutation the merge performed.
    victim = sorted(live_keys)[0]
    fails, _, err = _evaluate(live_keys - {victim}, attempts_now, retired_now)
    hit = not err and len(fails) == 1 and victim in fails[0]
    results.append(
        (
            f"2 a deleted entry is named ({victim.split('/')[0]}...)",
            hit,
            err or "; ".join(fails[:2]) or "no failure reported",
        )
    )

    # Arm 3 — THE REAL INCIDENT. Not a mutation of today's files: the two artifacts as they
    # stood in the commit the deleting merge produced.
    led_at = scratch / "ledger_at_incident.jsonl"
    att_at = scratch / "attempts_at_incident.jsonl"
    if _git_show(INCIDENT_COMMIT, LEDGER_REL, led_at, repo) and _git_show(
        INCIDENT_COMMIT, ATTEMPTS_REL, att_at, repo
    ):
        fails, _, err = _evaluate(
            _ledger_keys(led_at.read_text(encoding="utf-8")), att_at, no_retire
        )
        named = {k for k in INCIDENT_KEYS if any(k in f for f in fails)}
        hit = not err and named == INCIDENT_KEYS and len(fails) == len(INCIDENT_KEYS)
        detail = f"{len(fails)} failure(s), {len(named)}/3 incident keys named"
    else:
        hit, detail = False, f"ERROR(instrument): cannot read {INCIDENT_COMMIT} from this repo"
    results.append(("3 the real eb84364 loss is detected", hit, detail))

    # Arm 4 — retirement is the escape, and it costs a key, an owner, a date and a reason.
    # `owner`/`date` are not decoration here: since the register and its parse were unified into
    # `ci/proof_retirement.py`, this tool and `ci/check_ledger_census.py` require the same three
    # fields, so a record that omits them is refused by BOTH rather than exempting by one.
    retire_one = scratch / "retire_one.json"
    retire_one.write_bytes(
        _serialize(
            {
                "retired": base_rows
                + [
                    {
                        "key": victim,
                        "owner": "probe",
                        "date": "2026-08-03",
                        "reason": "probe arm 4",
                    }
                ]
            }
        )
    )
    fails, _, err = _evaluate(live_keys - {victim}, attempts_now, retire_one)
    results.append(
        ("4 a retired key is exempt", not err and not fails, err or "; ".join(fails[:2]))
    )

    # Arm 5 — and a retirement for a key that is still present must itself fail, or the file
    # becomes a place to park exemptions nobody ever removes.
    fails, _, err = _evaluate(live_keys, attempts_now, retire_one)
    hit = not err and len(fails) == 1 and "retired" in fails[0]
    results.append(("5 a stale retirement fails", hit, err or "; ".join(fails[:2]) or "no failure"))

    # Arm 5b — the polarity that the one-register unification adds: a withdrawal nobody signed is
    # refused HERE, in the producer, and not merely by the census. This arm is the falsifier for
    # "one register, one rule" — it went the other way (silently exempt) until 2026-08-05.
    unsigned = scratch / "retire_unsigned.json"
    unsigned.write_bytes(_serialize({"retired": [{"key": victim, "reason": "no owner, no date"}]}))
    fails, _, err = _evaluate(live_keys - {victim}, attempts_now, unsigned)
    hit = bool(err) and "owner" in err and not fails
    results.append(
        ("5b an unsigned retirement is refused, not honoured", hit, err or "no error reported")
    )

    # Arm 6 — no attempt log. The answer must be "I cannot tell", not "nothing is missing".
    fails, _, err = _evaluate(live_keys, scratch / "does_not_exist.jsonl", no_retire)
    results.append(("6 a missing attempt log is ERROR(instrument)", bool(err) and not fails, err))

    return results


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def _parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="probe_ledger_loss.py",
        description=(
            "Falsifier for check_no_proof_went_missing. Writes NOTHING into a tracked path "
            "unless --record is given."
        ),
    )
    p.add_argument(
        "--out",
        type=pathlib.Path,
        default=None,
        metavar="DIR",
        help=(
            "Keep the working files and result.json in DIR. Refused when DIR is inside the "
            "repository and not git-ignored, unless --record is also given. Default: an "
            "ephemeral scratch directory outside the repository, removed on exit."
        ),
    )
    p.add_argument(
        "--record",
        action="store_true",
        help=(
            "Explicit recording mode. The ONLY way to write a reading into a tracked path. "
            "Requires --out, adds the provenance block, and stamps an artifact-frame.json."
        ),
    )
    p.add_argument(
        "--repo",
        type=pathlib.Path,
        default=REPO,
        metavar="PATH",
        help="The checkout whose evidence the arms read. Default: this tool's own checkout.",
    )
    return p.parse_args(argv)


def _stamp_frame(out: pathlib.Path, repo: pathlib.Path) -> str:
    """Stamp `out` with an artifact frame, using `ci/check_artifact_frame.py` itself.

    Importing the screen rather than re-implementing `stamp()` here is the point: a second
    implementation of "what a frame is" is how the retirement register came to have two readers
    that disagreed about the same 43 keys (`ci/proof_retirement.py`'s docstring).
    """
    ci_dir = REPO / "ci"
    if not (ci_dir / "check_artifact_frame.py").is_file():
        return f"no ci/check_artifact_frame.py beside {TOOL_REL}; cannot stamp"
    sys.path.insert(0, str(ci_dir))
    try:
        import check_artifact_frame as caf  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - import outage
        return f"could not import ci/check_artifact_frame.py: {exc!r}"
    caf.stamp(
        repo=repo,
        directory=out,
        subject_paths=list(SUBJECT_PATHS),
        platform=sys.platform,
        subject=None,
        note=(
            f"Recorded reading of {TOOL_REL}, owner {OWNER}. Produced with --record; the "
            "probe's default execution writes nothing into any tracked path (issue #14)."
        ),
    )
    return ""


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    repo = args.repo.resolve()

    if args.record and args.out is None:
        print(
            "ERROR(instrument=record_without_destination): --record names no directory. "
            "Recording is deliberate by construction; it does not pick a path for you.",
            flush=True,
        )
        return EXIT_USAGE

    ephemeral = args.out is None
    if ephemeral:
        out = pathlib.Path(tempfile.mkdtemp(prefix="probe_ledger_loss_")).resolve()
    else:
        out = args.out.resolve()
        kind = classify_destination(repo, out)
        if kind == DEST_TRACKED_SURFACE and not args.record:
            rel = out.relative_to(repo).as_posix() if out.is_relative_to(repo) else str(out)
            print(
                f"ERROR(instrument=refused_tracked_destination): {rel} is inside the repository "
                "and git does not ignore it, so writing this reading there would leave a tracked "
                "diff — which is how a diagnostic stops being run and how an unowned `pass=true` "
                "goes stale unnoticed (issue #14). Nothing was written.\n"
                "  Ordinary use:  omit --out (ephemeral scratch outside the repository), or "
                "point --out at a scratch path.\n"
                "  Deliberate recording:  add --record, which stamps an artifact-frame.json and "
                "puts an owner and a subject digest on the reading.",
                flush=True,
            )
            return EXIT_ERROR_INSTRUMENT
        out.mkdir(parents=True, exist_ok=True)

    try:
        results = run_arms(repo, out)
        ok = all(hit for _, hit, _ in results)

        record = {
            "schema": RECORD_SCHEMA,
            "tool": TOOL_REL,
            "owner": OWNER,
            "recorded": bool(args.record),
            "produced_at_commit": _git(["rev-parse", "HEAD"], repo).stdout.strip(),
            "subject": {
                "attempts": {"path": ATTEMPTS_REL, "sha256": _digest(repo / ATTEMPTS_REL)},
                "ledger": {"path": LEDGER_REL, "sha256": _digest(repo / LEDGER_REL)},
                "register": {"path": REGISTER_REL, "sha256": _digest(repo / REGISTER_REL)},
            },
            "arms": [{"arm": n, "pass": h, "detail": _scrub(d, repo, out)} for n, h, d in results],
            "arms_passed": sum(1 for _, h, _ in results if h),
            "arms_total": len(results),
            "pass": ok,
        }

        leak = _leaked_absolute_path(record)
        if leak:
            print(
                "ERROR(instrument=provenance_leaked_absolute_path): the record still names a "
                f"machine-specific path near {leak!r}. A reading whose evidence is a path that "
                "exists on one box is not portable evidence — that is exactly what the committed "
                "reading carried (issue #14). Nothing was recorded.",
                flush=True,
            )
            return EXIT_ERROR_INSTRUMENT

        (out / "result.json").write_bytes(_serialize(record))

        if args.record:
            problem = _stamp_frame(out, repo)
            if problem:
                print(f"ERROR(instrument=frame_not_stamped): {problem}", flush=True)
                return EXIT_ERROR_INSTRUMENT

        for name, hit, detail in results:
            print(f"  [{'PASS' if hit else 'FAIL'}] {name}    {_scrub(detail, repo, out)[:160]}")
        print(f"{'PASS' if ok else 'FAIL'}: {sum(h for _, h, _ in results)}/{len(results)} arms")
        if ephemeral:
            print("scratch: an ephemeral directory outside the repository, removed on exit")
        else:
            rel = out.relative_to(repo).as_posix() if out.is_relative_to(repo) else str(out)
            print(f"{'recorded' if args.record else 'wrote'} result.json under {rel}")
        return EXIT_PASS if ok else EXIT_FAIL_CONDITION
    finally:
        if ephemeral:
            shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
