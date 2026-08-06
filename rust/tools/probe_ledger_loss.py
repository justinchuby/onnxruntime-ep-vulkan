#!/usr/bin/env python
"""Does `--check` notice a proof that went *missing*? Owner: Mouse.

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

USAGE
    python rust/tools/probe_ledger_loss.py
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import gen_proof_ledger as gpl  # noqa: E402

LEDGER = REPO / "evidence" / "proof_ledger.jsonl"
INCIDENT_COMMIT = "eb84364"
INCIDENT_KEYS = {
    "ai.onnx::Cast/6+/f32>bool/ew_cast_f32_to_bool/static/n1",
    "ai.onnx::Cast/6+/f32>i32/ew_cast_f32_to_i32/static/n1",
    "ai.onnx::Cast/6+/i32>f32/ew_cast_i32_to_f32/static/n1",
}


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


def _git_show(rev: str, path: str, dest: pathlib.Path) -> bool:
    r = subprocess.run(
        ["git", "show", f"{rev}:{path}"],
        cwd=REPO, capture_output=True, encoding="utf-8", errors="replace",
    )
    if r.returncode != 0:
        return False
    dest.write_text(r.stdout, encoding="utf-8")
    return True


def main() -> int:
    scratch = REPO / "bench" / "results" / "_probe_ledger_loss"
    scratch.mkdir(parents=True, exist_ok=True)
    attempts_now = REPO / "evidence" / "proof_attempts.jsonl"
    no_retire = scratch / "none.json"
    no_retire.write_text(json.dumps({"retired": []}), encoding="utf-8")

    # TODAY'S ARMS READ TODAY'S REGISTER, AND THAT IS NOT A CONVENIENCE.
    #
    # Arms 1, 2, 4 and 5 ask about the tree as it stands, and the tree as it stands has 43 keys
    # deliberately withdrawn by the §8.9.23 schema change. Handing those arms an EMPTY register
    # asked the invariant a question about a repository that does not exist: all 43 read as lost,
    # arm 1 ("a guard that fails on a healthy tree gets switched off") was red for days, and arms
    # 2/4/5 could no longer count their own failures. The empty register stays where it belongs —
    # arm 3, the historical replay, where nothing had been retired yet and where applying today's
    # withdrawals to yesterday's ledger would convict the right revision for the wrong reason.
    base_rows = json.loads(
        (REPO / "evidence" / "retired_proof_keys.json").read_text(encoding="utf-8")
    )["retired"] if (REPO / "evidence" / "retired_proof_keys.json").is_file() else []
    retired_now = scratch / "register_now.json"
    retired_now.write_text(json.dumps({"retired": base_rows}), encoding="utf-8")

    results: list[tuple[str, bool, str]] = []
    live_keys = _ledger_keys(LEDGER.read_text(encoding="utf-8"))

    # Arm 1 — the tree as it stands. A guard that fails on a healthy tree gets switched off.
    fails, notes, err = _evaluate(live_keys, attempts_now, retired_now)
    results.append(("1 current tree is clean", not err and not fails, err or "; ".join(fails[:2])))

    # Arm 2 — delete one entry. This is the mutation the merge performed.
    victim = sorted(live_keys)[0]
    fails, _, err = _evaluate(live_keys - {victim}, attempts_now, retired_now)
    hit = not err and len(fails) == 1 and victim in fails[0]
    results.append((f"2 a deleted entry is named ({victim.split('/')[0]}...)", hit,
                    err or "; ".join(fails[:2]) or "no failure reported"))

    # Arm 3 — THE REAL INCIDENT. Not a mutation of today's files: the two artifacts as they
    # stood in the commit the deleting merge produced.
    led_at = scratch / "ledger_at_incident.jsonl"
    att_at = scratch / "attempts_at_incident.jsonl"
    if _git_show(INCIDENT_COMMIT, "evidence/proof_ledger.jsonl", led_at) and _git_show(
        INCIDENT_COMMIT, "evidence/proof_attempts.jsonl", att_at
    ):
        fails, _, err = _evaluate(_ledger_keys(led_at.read_text(encoding="utf-8")), att_at, no_retire)
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
    retire_one.write_text(
        json.dumps({"retired": base_rows + [{"key": victim, "owner": "probe",
                                             "date": "2026-08-03",
                                             "reason": "probe arm 4"}]}),
        encoding="utf-8",
    )
    fails, _, err = _evaluate(live_keys - {victim}, attempts_now, retire_one)
    results.append(("4 a retired key is exempt", not err and not fails, err or "; ".join(fails[:2])))

    # Arm 5 — and a retirement for a key that is still present must itself fail, or the file
    # becomes a place to park exemptions nobody ever removes.
    fails, _, err = _evaluate(live_keys, attempts_now, retire_one)
    hit = not err and len(fails) == 1 and "retired" in fails[0]
    results.append(("5 a stale retirement fails", hit, err or "; ".join(fails[:2]) or "no failure"))

    # Arm 5b — the polarity that the one-register unification adds: a withdrawal nobody signed is
    # refused HERE, in the producer, and not merely by the census. This arm is the falsifier for
    # "one register, one rule" — it went the other way (silently exempt) until 2026-08-05.
    unsigned = scratch / "retire_unsigned.json"
    unsigned.write_text(
        json.dumps({"retired": [{"key": victim, "reason": "no owner, no date"}]}),
        encoding="utf-8",
    )
    fails, _, err = _evaluate(live_keys - {victim}, attempts_now, unsigned)
    hit = bool(err) and "owner" in err and not fails
    results.append(("5b an unsigned retirement is refused, not honoured", hit,
                    err[:80] or "no error reported"))

    # Arm 6 — no attempt log. The answer must be "I cannot tell", not "nothing is missing".
    fails, _, err = _evaluate(live_keys, scratch / "does_not_exist.jsonl", no_retire)
    results.append(("6 a missing attempt log is ERROR(instrument)", bool(err) and not fails, err[:80]))

    ok = all(hit for _, hit, _ in results)
    for name, hit, detail in results:
        print(f"  [{'PASS' if hit else 'FAIL'}] {name}    {detail}")
    print(f"{'PASS' if ok else 'FAIL'}: {sum(h for _, h, _ in results)}/{len(results)} arms")
    (scratch / "result.json").write_text(
        json.dumps(
            {"arms": [{"arm": n, "pass": h, "detail": d} for n, h, d in results], "pass": ok},
            indent=1,
        ),
        encoding="utf-8",
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
