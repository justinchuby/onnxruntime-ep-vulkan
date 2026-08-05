#!/usr/bin/env python3
"""Negative control for ci/check_open_reds.py — every arm shown in its POSITIVE state.

A screen that has only ever been observed green is indistinguishable from a constant that
returns green. Every rule below is therefore exercised with the defect GENUINELY PRESENT,
and each arm is labelled with how the defect got there:

  LIVE      the real register against the real tree, right now
  REPLAYED  real bytes taken from this repository's own history — a defect that actually
            happened, not one I imagined
  PLANTED   a mutation written on purpose. Proves the rule fires on the shape it was
            written for. Does NOT prove the rule is load-bearing, and the count of
            PLANTED arms is printed so nobody reads it as if it did.

Run:  python ci/negative_control_open_reds.py
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SCREEN = HERE / "check_open_reds.py"
REGISTER = HERE / "open_reds.json"

# The merge commit that reintroduced four `if: env.BUILD_SKIPPED != '1'` guards whose
# writer had been deleted. Real bytes, real defect, found by the real screen.
HISTORICAL_REF = "133b9fe"

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str, str]] = []


def record(kind: str, name: str, ok: bool, note: str = "") -> None:
    results.append((kind, name, PASS if ok else FAIL, note))
    mark = "ok  " if ok else "FAIL"
    print(f"  [{mark}] {kind:<8} {name}" + (f"  — {note}" if note and not ok else ""))


def run_screen(register: Path, extra: list[str] | None = None, env: dict | None = None):
    e = dict(os.environ)
    e.setdefault("PYTHONIOENCODING", "utf-8")
    e.pop("OPEN_REDS_TODAY", None)
    e.pop("OPEN_REDS_FORCE_ANNOTATE", None)
    if env:
        e.update(env)
    return subprocess.run(
        [sys.executable, str(SCREEN), "--register", str(register), "--repo", str(REPO), *(extra or [])],
        capture_output=True, encoding="utf-8", errors="replace", env=e, cwd=str(REPO), timeout=2400,
    )


def base_doc() -> dict:
    return json.loads(REGISTER.read_text(encoding="utf-8"))


def minimal_doc(entry: dict) -> dict:
    return {
        "schema": 1, "purpose": "negative control", "checks": [entry],
        "subjects": [entry["id"]] if "id" in entry else [], "retired": {},
    }


def green_entry(**over) -> dict:
    e = {
        "id": "always_green",
        "cmd": ["python", "-c", "print('nothing wrong here')"],
        "expect": "green",
        "owner": "link",
        "opened": "2026-01-01",
        "review_by": "2099-01-01",
        "reason": "control arm",
        "closes_when": "n/a",
    }
    e.update(over)
    return e


def red_entry(**over) -> dict:
    e = {
        "id": "always_red",
        "cmd": ["python", "-c", "import sys; print('the accepted red over member alpha'); sys.exit(1)"],
        "expect": "red",
        "signature": "the accepted red",
        # An expect=red entry has needed an `extent` since 69ac222, and this factory did
        # not have one -- so from that commit every planted red arm below was really
        # measuring "the register refuses to load at all" (exit 2) while its recorded
        # question was about stale_acceptance, signature_changed or lease_expired. The
        # arms did not go quiet; they went wrong, and the failure text they compared
        # against was a usage error. Trinity, 2026-08-05.
        "extent": {"pattern": r"over member (\w+)", "members": ["alpha"]},
        "owner": "link",
        "opened": "2026-01-01",
        "review_by": "2099-01-01",
        "reason": "control arm",
        "closes_when": "when the control stops needing it",
    }
    e.update(over)
    return e


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="open-reds-control-", dir=str(REPO / "ci")))
    try:
        w = lambda name, doc: (tmp / name, (tmp / name).write_text(json.dumps(doc), encoding="utf-8"))[0]  # noqa: E731

        print("\nLIVE — the real register against the real tree")
        r = run_screen(REGISTER)
        record("LIVE", "the shipped register is the colour it declares", r.returncode == 0,
               (r.stdout or "")[-1500:] + (r.stderr or "")[-1500:])
        record("LIVE", "and it names every accepted red with an owner",
               "ACCOUNTED REDS" in r.stdout and "owner" in r.stdout)

        print("\nREPLAYED — real defective bytes from this repository's history")
        old = subprocess.run(["git", "show", f"{HISTORICAL_REF}:.github/workflows/ci.yml"],
                             cwd=str(REPO), capture_output=True, encoding="utf-8",
                             errors="replace")
        if old.returncode != 0:
            record("REPLAYED", f"{HISTORICAL_REF} readable", False, old.stderr.strip())
        else:
            victim = tmp / "ci-at-133b9fe.yml"
            victim.write_text(old.stdout, encoding="utf-8")
            # The register said this check must be green. At 133b9fe it was not.
            doc = minimal_doc(green_entry(
                id="build_precondition_at_133b9fe",
                cmd=["python", "ci/check_build_precondition.py", str(victim)],
            ))
            rr = run_screen(w("replayed.json", doc))
            record("REPLAYED", "four dormant BUILD_SKIPPED guards -> unaccounted_red",
                   rr.returncode == 1 and "unaccounted_red" in (rr.stdout + rr.stderr),
                   (rr.stdout + rr.stderr)[-1200:])
            record("REPLAYED", "and the same bytes name the guard in the failure text",
                   "BUILD_SKIPPED" in (rr.stdout + rr.stderr))
            # Other polarity, same rule, same file: today's bytes are green.
            doc2 = minimal_doc(green_entry(
                id="build_precondition_today",
                cmd=["python", "ci/check_build_precondition.py", ".github/workflows/ci.yml"],
            ))
            r2 = run_screen(w("replayed-today.json", doc2))
            record("REPLAYED", "and today's bytes are green — the rule is not a constant",
                   r2.returncode == 0, (r2.stdout + r2.stderr)[-800:])

        print("\nPLANTED — the five condition arms, each with its defect really present")
        d = minimal_doc(green_entry(cmd=["python", "-c", "import sys; print('boom'); sys.exit(1)"]))
        r = run_screen(w("unaccounted.json", d))
        record("PLANTED", "expect=green + observed red -> unaccounted_red",
               r.returncode == 1 and "unaccounted_red" in (r.stdout + r.stderr))

        d = minimal_doc(red_entry(cmd=["python", "-c", "print('fixed!')"]))
        r = run_screen(w("stale.json", d))
        record("PLANTED", "expect=red + observed green -> stale_acceptance",
               r.returncode == 1 and "stale_acceptance" in (r.stdout + r.stderr))
        record("PLANTED", "and stale_acceptance quotes closes_when so it can be discharged",
               "when the control stops needing it" in (r.stdout + r.stderr))

        d = minimal_doc(red_entry(
            cmd=["python", "-c", "import sys; print('a DIFFERENT failure'); sys.exit(1)"]))
        r = run_screen(w("signature.json", d))
        record("PLANTED", "expect=red + red for another reason -> signature_changed",
               r.returncode == 1 and "signature_changed" in (r.stdout + r.stderr))

        d = minimal_doc(red_entry(review_by="2020-01-01"))
        r = run_screen(w("expired.json", d))
        record("PLANTED", "review_by in the past -> lease_expired",
               r.returncode == 1 and "lease_expired" in (r.stdout + r.stderr))

        # Both polarities of the lease, driven by the same knob, so the arm above is not
        # just "any date makes it red".
        d = minimal_doc(red_entry(review_by="2026-06-01"))
        r = run_screen(w("lease-both.json", d), env={"OPEN_REDS_TODAY": "2026-05-31"})
        record("PLANTED", "the day before review_by -> not expired", r.returncode == 0,
               (r.stdout + r.stderr)[-600:])
        r = run_screen(w("lease-both.json", d), env={"OPEN_REDS_TODAY": "2026-06-02"})
        record("PLANTED", "the day after review_by -> lease_expired",
               r.returncode == 1 and "lease_expired" in (r.stdout + r.stderr))

        print("\nPLANTED — `extent`: the members the acceptance is granted over")
        # This feature landed in 69ac222 with NO arm in this control, and the register it
        # was enforced over could not be loaded from that commit onward. Both facts are the
        # same omission: a rule with no falsifier beside it is a rule nobody has run.
        r = run_screen(w("extent-ok.json", minimal_doc(red_entry())))
        record("PLANTED", "extent members match the run -> the red is ACCOUNTED",
               r.returncode == 0 and "ACCOUNTED" in (r.stdout + r.stderr),
               (r.stdout + r.stderr)[-800:])

        d = minimal_doc(red_entry(cmd=["python", "-c", "import sys; "
                                       "print('the accepted red over member alpha'); "
                                       "print('the accepted red over member beta'); sys.exit(1)"]))
        r = run_screen(w("extent-wide.json", d))
        record("PLANTED", "a SECOND member joining the same red -> extent_widened",
               r.returncode == 1 and "extent_widened" in (r.stdout + r.stderr),
               (r.stdout + r.stderr)[-800:])
        record("PLANTED", "and extent_widened names the newcomer, not just a count",
               "beta" in (r.stdout + r.stderr))

        d = minimal_doc(red_entry(extent={"pattern": r"over member (\w+)",
                                          "members": ["alpha", "gamma"]}))
        r = run_screen(w("extent-narrow.json", d))
        record("PLANTED", "a member no longer failing -> extent_narrowed",
               r.returncode == 1 and "extent_narrowed" in (r.stdout + r.stderr),
               (r.stdout + r.stderr)[-800:])

        d = minimal_doc(red_entry(extent={"pattern": r"over member \w+", "members": ["alpha"]}))
        r = run_screen(w("extent-nogroup.json", d))
        record("PLANTED", "extent.pattern with no capture group -> refused",
               r.returncode != 0 and "capture group" in (r.stdout + r.stderr),
               (r.stdout + r.stderr)[-600:])

        # The arm this control was missing when the device register went dark. An entry
        # with no extent is UNCOLOURED — ERROR(instrument), never accepted, never green —
        # and, decisively, the OTHER entries in the same file are still ruled on.
        no_ext = {k: v for k, v in red_entry(id="undeclared_extent").items() if k != "extent"}
        doc = {"schema": 1, "purpose": "c",
               "checks": [green_entry(), no_ext],
               "subjects": ["always_green", "undeclared_extent"], "retired": {}}
        r = run_screen(w("extent-undeclared.json", doc))
        out = r.stdout + r.stderr
        record("PLANTED", "expect=red with no extent -> ERROR(instrument=extent_undeclared)",
               r.returncode == 4 and "extent_undeclared" in out, out[-900:])
        record("PLANTED", "and the OTHER entry in that file is still ruled on, not silenced",
               "always_green" in out and "PASS" in out, out[-900:])

        print("\nPLANTED — the register refuses a partial entry rather than accepting it")
        for field in ("owner", "reason", "closes_when", "review_by", "opened", "cmd", "expect"):
            e = red_entry()
            e.pop(field)
            r = run_screen(w(f"missing-{field}.json", minimal_doc(e)))
            record("PLANTED", f"entry missing `{field}` -> usage (exit 2)",
                   r.returncode == 2 and field in (r.stdout + r.stderr))

        r = run_screen(w("nosig.json", minimal_doc({k: v for k, v in red_entry().items()
                                                    if k != "signature"})))
        record("PLANTED", "expect=red with no signature -> usage; a blanket acceptance is a deletion",
               r.returncode == 2 and "signature" in (r.stdout + r.stderr))

        r = run_screen(w("badexpect.json", minimal_doc(red_entry(expect="amber"))))
        record("PLANTED", "expect=amber -> usage", r.returncode == 2)

        r = run_screen(w("baddate.json", minimal_doc(red_entry(review_by="soon"))))
        record("PLANTED", "review_by='soon' -> usage", r.returncode == 2)

        dup = {"schema": 1, "purpose": "c", "checks": [red_entry(), red_entry()],
               "subjects": ["always_red"], "retired": {}}
        r = run_screen(w("dup.json", dup))
        record("PLANTED", "duplicate id -> usage", r.returncode == 2 and "duplicate" in (r.stdout + r.stderr))

        r = run_screen(w("empty.json", {"schema": 1, "purpose": "c", "checks": [],
                                        "subjects": ["x"], "retired": {}}))
        record("PLANTED", "empty register -> usage; zero checks is not zero findings",
               r.returncode == 2)

        print("\nREPLAYED — the denominator defect, exactly as it happened on 2026-08-03")
        # Mouse repaired three of five accepted reds and deleted their entries, which the
        # file told him to do. One was not actually green (7 of 8 accessors wired, an
        # 8th missed), so the delete removed the CHECK and not just the acceptance: 8
        # subjects became 5, and the screen printed PASS with a red check in the tree.
        # These are the real ids from ci/open_reds.json at 9cff913.
        gone = ["audit_instruments_census", "harness_census_drift", "proof_ledger_writer_refuses"]
        doc = minimal_doc(green_entry(id="lane_checks_suite"))
        doc["subjects"] = ["lane_checks_suite", *gone]
        r = run_screen(w("denominator.json", doc))
        record("REPLAYED", "three subjects deleted from `checks` -> usage, not PASS",
               r.returncode == 2 and all(g in (r.stdout + r.stderr) for g in gone),
               (r.stdout + r.stderr)[-1200:])
        record("REPLAYED", "and the message says flip to green rather than delete",
               "does not leave this register by being deleted" in (r.stdout + r.stderr))
        # The other polarity, same three ids: retiring them deliberately is allowed.
        doc2 = minimal_doc(green_entry(id="lane_checks_suite"))
        doc2["subjects"] = ["lane_checks_suite", *gone]
        doc2["retired"] = {g: {"owner": "mouse", "date": "2026-08-03", "reason": "fixed"}
                           for g in gone}
        r = run_screen(w("denominator-ok.json", doc2))
        record("REPLAYED", "the same three RETIRED with a reason -> PASS",
               r.returncode == 0 and "RETIRED audit_instruments_census" in r.stdout,
               (r.stdout + r.stderr)[-800:])

        print("\nPLANTED — the rest of the subjects arithmetic")
        doc = minimal_doc(green_entry(id="a"))
        doc.pop("subjects")
        r = run_screen(w("nosubjects.json", doc))
        record("PLANTED", "a register with no `subjects` -> usage", r.returncode == 2)

        doc = minimal_doc(green_entry(id="a"))
        doc["subjects"] = ["a", "b"]
        doc["retired"] = {"b": {"owner": "x", "date": "2026-08-03"}}
        r = run_screen(w("noreason.json", doc))
        record("PLANTED", "a retirement with no reason -> usage",
               r.returncode == 2 and "reason" in (r.stdout + r.stderr))

        doc = minimal_doc(green_entry(id="a"))
        doc["retired"] = {"a": {"owner": "x", "date": "2026-08-03", "reason": "y"}}
        r = run_screen(w("bothlists.json", doc))
        record("PLANTED", "an id both live and retired -> usage",
               r.returncode == 2 and "both live and retired" in (r.stdout + r.stderr))

        doc = {"schema": 1, "purpose": "c", "subjects": ["a"], "retired": {},
               "checks": [green_entry(id="a"), green_entry(id="b")]}
        r = run_screen(w("undeclared.json", doc))
        record("PLANTED", "a check absent from `subjects` -> usage",
               r.returncode == 2 and "'b'" in (r.stdout + r.stderr))

        print("\nPLANTED — the instrument arms: an unobserved colour is not a green one")
        r = run_screen(tmp / "does-not-exist.json")
        record("PLANTED", "register absent -> ERROR(instrument) exit 4, not PASS",
               r.returncode == 4 and "register_absent" in (r.stdout + r.stderr))

        d = minimal_doc(green_entry(cmd=["definitely-not-a-real-binary-xyz"]))
        r = run_screen(w("nocmd.json", d))
        record("PLANTED", "command absent -> ERROR(instrument) exit 4, not a red",
               r.returncode == 4 and "command_absent" in (r.stdout + r.stderr))

        d = minimal_doc(green_entry(
            cmd=["python", "-c", "import time; time.sleep(30)"], timeout=1))
        r = run_screen(w("slow.json", d))
        record("PLANTED", "check that never finished -> ERROR(instrument=timeout)",
               r.returncode == 4 and "timeout" in (r.stdout + r.stderr))

        print("\nPLANTED — the annotation path, which is the whole point of the screen")
        d = minimal_doc(red_entry())
        r = run_screen(w("annot-red.json", d), env={"OPEN_REDS_FORCE_ANNOTATE": "1"})
        record("PLANTED", "an accounted red emits ::warning naming its owner",
               "::warning title=open red: always_red::" in r.stdout and "link" in r.stdout)
        d = minimal_doc(green_entry(cmd=["python", "-c", "import sys; sys.exit(1)"]))
        r = run_screen(w("annot-new.json", d), env={"OPEN_REDS_FORCE_ANNOTATE": "1"})
        record("PLANTED", "an unaccounted red emits ::error",
               "::error title=FAIL(condition=unaccounted_red)" in (r.stdout + r.stderr))
        r = run_screen(w("annot-red.json", minimal_doc(red_entry())))
        record("PLANTED", "and nothing is annotated off a runner",
               "::warning" not in r.stdout and "::error" not in r.stdout)

        print("\nPLANTED — there is no flag that turns a failure off")
        d = minimal_doc(green_entry(cmd=["python", "-c", "import sys; sys.exit(1)"]))
        reg = w("noflag.json", d)
        for flag in ("--relax", "--allow-fail", "--warn-only", "--soft"):
            r = run_screen(reg, [flag])
            record("PLANTED", f"`{flag}` is rejected, not honoured",
                   r.returncode == 2 and "unrecognized arguments" in (r.stdout + r.stderr))

        print("\nPLANTED — --list observes no colour and must not claim one")
        r = run_screen(REGISTER, ["--list"])
        record("PLANTED", "--list says it ran nothing",
               r.returncode == 0 and "no check was run" in r.stdout and "PASS —" not in r.stdout)

        print("\nPLANTED — --only cannot silently select nothing")
        r = run_screen(REGISTER, ["--only", "no_such_check"])
        record("PLANTED", "--only with an unknown id -> usage, not an empty green run",
               r.returncode == 2)

        print("\nPLANTED — the shipped register's own shape")
        doc = base_doc()
        ids = [c["id"] for c in doc["checks"]]
        record("PLANTED", "every accepted red names an owner that is not 'link' alone",
               all(c["owner"] for c in doc["checks"] if c["expect"] == "red"))
        record("PLANTED", "no accepted red uses a signature short enough to match anything",
               all(len(c["signature"]) >= 8 for c in doc["checks"] if c["expect"] == "red"))
        record("PLANTED", "the suite entry is narrowed, not accepted whole",
               any(i == "lane_checks_suite" for i in ids)
               and any("--deselect" in c["cmd"] for c in doc["checks"] if c["id"] == "lane_checks_suite"))
        record("PLANTED", "every subject is either live or retired — nothing has fallen out",
               set(doc["subjects"]) == set(ids) | set(doc.get("retired", {})))
        record("PLANTED", "the removal instructions say flip, not delete",
               "DO NOT DELETE" in doc["how_to_remove_an_entry"])

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 78)
    kinds = {}
    for kind, _, state, _ in results:
        kinds[kind] = kinds.get(kind, 0) + 1
    bad = [r for r in results if r[2] == FAIL]
    print(f"  {len(results)} arm(s): " + ", ".join(f"{v} {k}" for k, v in sorted(kinds.items())))
    print("  PLANTED arms prove the rule fires on the shape it was written for. They do NOT")
    print("  show the rule is load-bearing. The LIVE and REPLAYED arms are the ones that do,")
    print("  and the REPLAYED defect is one this repository actually shipped into a merge.")
    if bad:
        print(f"\n  {len(bad)} arm(s) FAILED:")
        for kind, name, _, note in bad:
            print(f"    {kind} {name}\n      {note}")
        return 1
    print("\nNEGATIVE-CONTROL(open-reds): PASS — every arm fired in its positive state.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
