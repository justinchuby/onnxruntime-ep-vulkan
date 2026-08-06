#!/usr/bin/env python3
"""Cross-lane identity, read off the repetitions the device-loss gate already paid for.

WHY THIS EXISTS
===============

``device_loss_gate.py --lanes resident,shipping`` writes ``repNNN.json.logits.npy`` for every
repetition.  Every repetition runs the SAME chain — same rng seed, same ``--seed-past``, same
``--steps`` — so the logits of two repetitions differ only by their lane and by whatever the
device did.  That makes the gate's own leftovers a **free** cross-lane identity check, and a
free check is one nobody has to decide to fund.

It is written as a separate reader, not folded into the gate, because a gate that also grades
correctness has two ways to be green and only one of them is about device loss.

WHAT IT REPORTS
===============

  * ``WITHIN_LANE`` — are all clean repetitions of one lane byte-identical to each other?
    A lane that is not self-consistent cannot be compared to anything, and this is the
    control that says so before any cross-lane claim is made.
  * ``CROSS_LANE``  — is the resident lane's digest the shipping lane's digest?

Repetitions that LOST the device or ran vacuously are excluded and **counted in the output**,
because an excluded repetition is a fact about the comparison, not a detail of it.

USAGE
    python bench/results/probe_lane_logits_identity.py \
        --record device_loss_gate-BOTHLANES.json --out lane_logits_identity.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent


def digest(path: pathlib.Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", default="device_loss_gate-BOTHLANES.json")
    ap.add_argument("--out", default="lane_logits_identity.json")
    args = ap.parse_args(argv)

    rec_path = REPO / "bench" / "results" / args.record
    if not rec_path.is_file():
        print(f"LANE-IDENTITY: ERROR(instrument=no_such_record) {rec_path}")
        return 4
    rec = json.loads(rec_path.read_text(encoding="utf-8"))

    by_lane: dict[str, list[dict]] = {}
    excluded: list[dict] = []
    for r in rec.get("reps", []):
        npy = REPO / "bench" / "results" / "device_loss_gate" / f"rep{r['rep']:03d}.json.logits.npy"
        d = digest(npy)
        row = {"rep": r["rep"], "lane": r["lane"], "sha256_16": (d or "")[:16],
               "dispatches": r["dispatches_executed"]}
        if r.get("lost") or r.get("vacuous") or d is None:
            row["excluded_because"] = (
                "device_lost" if r.get("lost")
                else "vacuous" if r.get("vacuous")
                else "no logits file"
            )
            excluded.append(row)
            continue
        by_lane.setdefault(r["lane"], []).append(row)

    doc: dict = {
        "probe": "lane_logits_identity",
        "source_record": args.record,
        "excluded": excluded,
        "per_lane": {},
    }
    # Propagated from the gate record this reader is entirely derived from (issue #19).
    # Found by hand rather than by ci/phi35_identity_audit.py, whose relations are
    # spawn/env/direct-read: this file consumes another producer's RECORD, and record
    # consumption is a stated limit of the audit rather than something it screens. A
    # derived comparison that cannot name the model is exactly as unfalsifiable as the
    # record it derives from, so the identity travels with it.
    doc["onnx_file"] = rec.get("onnx_file")
    doc["onnx_sha256"] = rec.get("onnx_sha256")
    if not doc["onnx_file"]:
        doc["onnx_identity_error"] = (
            "ERROR(identity=source_record_named_no_model): "
            f"{args.record} carries no onnx_file/onnx_sha256, so this cross-lane comparison "
            "cannot say which model's logits it compared. Re-run the gate."
        )
    for lane, rows in sorted(by_lane.items()):
        digests = sorted({r["sha256_16"] for r in rows})
        doc["per_lane"][lane] = {
            "reps_compared": len(rows),
            "distinct_digests": digests,
            "within_lane": "IDENTICAL" if len(digests) == 1 else "DIVERGENT",
            "reps": rows,
        }

    lanes = sorted(doc["per_lane"])
    if len(lanes) < 2:
        doc["cross_lane"] = "UNSCORABLE"
        doc["why"] = ["fewer than two lanes produced a comparable repetition"]
    elif any(doc["per_lane"][ln]["within_lane"] != "IDENTICAL" for ln in lanes):
        doc["cross_lane"] = "UNSCORABLE"
        doc["why"] = [
            "at least one lane is not self-consistent, so a cross-lane digest comparison "
            "would be comparing one of several answers to one of several answers"
        ]
    else:
        one = {ln: doc["per_lane"][ln]["distinct_digests"][0] for ln in lanes}
        same = len(set(one.values())) == 1
        doc["cross_lane"] = "IDENTICAL" if same else "DIVERGENT"
        doc["cross_lane_digests"] = one

    out = REPO / "bench" / "results" / args.out
    out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    for lane in lanes:
        p = doc["per_lane"][lane]
        print(f"LANE-IDENTITY: {lane:9s} {p['reps_compared']} rep(s) "
              f"{p['within_lane']} {p['distinct_digests']}")
    print(f"LANE-IDENTITY: CROSS_LANE {doc['cross_lane']}")
    if doc.get("onnx_identity_error"):
        print(f"  {doc['onnx_identity_error']}")
    else:
        print(f"  model: {doc['onnx_file']} sha256={(doc['onnx_sha256'] or '')[:16]}")
    for e in excluded:
        print(f"  excluded rep {e['rep']} ({e['lane']}): {e['excluded_because']}")
    print(f"  record: {out.relative_to(REPO)}")
    return 0 if doc.get("cross_lane") == "IDENTICAL" else 1


if __name__ == "__main__":
    sys.exit(main())
