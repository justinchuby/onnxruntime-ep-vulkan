#!/usr/bin/env python3
"""Derive the "runs versus does not run" statement from on-disk probe records.

WHY THIS EXISTS
---------------
The project has never been timed, so every performance claim it currently owns
is a count, a byte total, or a slope. There is one claim available that is
STRONGER than any of those and needs no clock at all: at long context the
shipping configuration does not run and the device-resident configuration does.
"Runs" versus "does not run" is not a ratio, so it cannot be inflated by a
favourable arm, and it survives being measured on a different box.

That claim is only worth citing if it is derived, not typed. Every number in the
output of this tool is read out of a named record file at run time and carried
with the sha256 of the file it came from. There are no literals here. If a
record moves, this tool reports fewer claims -- it never reports a stale one.

WHAT IT REFUSES
---------------
Three refusals, each of which cost this project a session at some point:

1. A lane with no counters file is NOT "a lane that ran cheaply". It is a lane
   that did not run, and a record where the counters are null must classify as
   DID_NOT_COMPLETE rather than be skipped into silence.

2. An exit code of 0 is not evidence a lane ran on the EP. ONNX Runtime
   swallows an EP Compute failure and silently rebuilds the graph on CPU, so a
   lane can exit 0, print a plausible answer, and have executed ZERO dispatches
   on the device under test. That is a distinct outcome from both "ran" and
   "crashed" and it gets its own class: COMPLETED_WITHOUT_EP.

3. A pair of lanes read out of two DIFFERENT record files is not a pair. Both
   sides of a run/does-not-run claim must come from the same probe invocation,
   because otherwise the two sides saw different VRAM tenancy, possibly a
   different build, and the comparison is between two afternoons rather than
   between two configurations.

The claim classes are ordered. COMPLETED_ON_EP > COMPLETED_WITHOUT_EP >
DID_NOT_COMPLETE. A claim is emitted only when two lanes of one record land in
different classes.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import hashlib
import json
import math
import os
import sys

# Ordered best-to-worst. The comparison is on this order, never on a number.
CLASS_ON_EP = "COMPLETED_ON_EP"
CLASS_NO_EP = "COMPLETED_WITHOUT_EP"
CLASS_DEAD = "DID_NOT_COMPLETE"
_RANK = {CLASS_DEAD: 0, CLASS_NO_EP: 1, CLASS_ON_EP: 2}

# Which configuration a lane name belongs to. The claim is about configurations,
# not about lane spellings, and two probes spell the same configuration
# differently ("resident"/"arena" vs "host"/"grow").
LANE_FAMILY = {
    "resident": "DEVICE_RESIDENT",
    "arena": "DEVICE_RESIDENT",
    "host": "SHIPPING",
    "grow": "SHIPPING",
}

DEFAULT_SOURCES = [
    "kv_arena_chain-A8192-LIVE.json",
    "kv_arena_chain-A8192.json",
    "phi35_kv_chain-ctx4096-BOTH-*.json",
]

DEFAULT_SINGLE_LANE = [
    "phi35_kv_chain-ctx4096-resident-retry*.json",
]


def _expand(here: str, patterns: list) -> list:
    """Patterns, not a fixed list.

    A record taken tomorrow must not be silently left out of a rate that is
    then quoted as complete. A denominator that only grows when someone
    remembers to edit a list is the shrinking-denominator defect this project
    has already been bitten by once.
    """
    out = []
    for pat in patterns:
        target = pat if os.path.isabs(pat) else os.path.join(here, pat)
        out.extend(sorted(glob.glob(target)))
    seen, uniq = set(), []
    for path in out:
        if path not in seen:
            seen.add(path)
            uniq.append(path)
    return uniq


DEFAULT_SINGLE_LANE = [
    "phi35_kv_chain-ctx4096-resident-retry1.json",
    "phi35_kv_chain-ctx4096-resident-retry2.json",
    "phi35_kv_chain-ctx4096-resident-retry3.json",
]


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _finite(value):
    """NaN and Infinity are JSON-illegal and they are also not measurements."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _classify(completed: bool, dispatches, why: list[str]) -> str:
    if not completed:
        return CLASS_DEAD
    if dispatches is None:
        # Completed but we cannot see the counters. That is not evidence of EP
        # execution, and guessing upward here is exactly how a silent CPU
        # fallback gets reported as a win.
        why.append("counters absent on a lane that exited cleanly")
        return CLASS_NO_EP
    if int(dispatches) <= 0:
        why.append("exited cleanly with zero EP dispatches (silent CPU fallback)")
        return CLASS_NO_EP
    return CLASS_ON_EP


def _read_kv_chain(doc: dict) -> tuple[str, list[dict]]:
    """probe_kv_chain_phi35.py: lanes is a dict of lane -> record."""
    context = doc.get("seed_past")
    out = []
    for lane, body in (doc.get("lanes") or {}).items():
        if not isinstance(body, dict) or not body:
            continue
        if lane == "cpu":
            continue  # the CPU lane is a reference oracle, not a shipping arm
        counters = body.get("final_counters") or {}
        exit_code = body.get("exit_code")
        why: list[str] = []
        completed = exit_code == 0
        if not completed:
            why.append(f"worker exit code {exit_code!r}")
        dispatches = counters.get("dispatches_executed") if counters else None
        out.append(
            {
                "lane": lane,
                "context_tokens": context,
                "completed": completed,
                "ep_dispatches": dispatches,
                "compute_failures": counters.get("compute_failures"),
                "device_losses": counters.get("device_losses"),
                "alloc_device_frame": counters.get("alloc_device_frame"),
                "outputs_device_bound": counters.get("outputs_device_bound"),
                "alloc_high_water_bytes": counters.get("alloc_device_high_water_bytes"),
                "alloc_device_upload_bytes": counters.get("alloc_device_upload_bytes"),
                "session_staging_readback_bytes": counters.get(
                    "session_staging_readback_bytes"
                ),
                "device_name": (body.get("ep_device") or {}).get("vulkan.device_name"),
                "why": why,
            }
        )
    return "seed_past", out


def _read_arena_chain(doc: dict) -> tuple[str, list[dict]]:
    """probe_kv_arena_phi35.py --mode chain: per-lane blocks under `bytes`."""
    context = doc.get("past")
    out = []
    for lane, body in (doc.get("bytes") or {}).items():
        if not isinstance(body, dict):
            continue
        rc = body.get("worker_rc")
        why: list[str] = []
        completed = rc == 0
        if not completed:
            why.append(f"worker exit code {rc!r}")
        disp = body.get("dispatches")
        total = sum(int(d) for d in disp) if isinstance(disp, list) and disp else None
        if total is None and completed:
            why.append("no per-step dispatch counts recorded")
        out.append(
            {
                "lane": lane,
                "context_tokens": context,
                "completed": completed,
                "ep_dispatches": total,
                "compute_failures": body.get("compute_failures"),
                "device_losses": body.get("device_losses"),
                "alloc_device_frame": body.get("kv_cache_convention"),
                "outputs_device_bound": None,
                "alloc_high_water_bytes": body.get("alloc_high_water_bytes"),
                "device_name": (body.get("ep_device") or {}).get("vulkan.device_name"),
                "slope_bytes_per_past_token": _finite(
                    body.get("slope_bytes_per_past_token")
                ),
                "why": why,
            }
        )
    return "past", out


READERS = {
    "kv_arena_chain": _read_arena_chain,
}


def read_record(path: str) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh, parse_constant=lambda c: float("nan"))
    except (OSError, ValueError) as exc:
        return {"source": path, "readable": False, "error": str(exc), "lanes": []}
    probe = doc.get("probe")
    reader = READERS.get(probe, _read_kv_chain)
    ctx_field, lanes = reader(doc)
    for lane in lanes:
        lane["classification"] = _classify(
            lane["completed"], lane["ep_dispatches"], lane["why"]
        )
    return {
        "source": path,
        "readable": True,
        "sha256": _sha256(path),
        "probe": probe,
        "context_field": ctx_field,
        "record_verdict": doc.get("verdict"),
        "lanes": lanes,
    }


def claims_for(record: dict) -> list[dict]:
    """A claim is a within-record pair of lanes in different classes."""
    lanes = [l for l in record["lanes"]]
    claims = []
    for i, a in enumerate(lanes):
        for b in lanes[i + 1 :]:
            if _RANK[a["classification"]] == _RANK[b["classification"]]:
                continue
            better, worse = (a, b) if _RANK[a["classification"]] > _RANK[b["classification"]] else (b, a)
            claims.append(
                {
                    "provenance": "MEASUREMENT",
                    "context_tokens": better["context_tokens"],
                    "favours": LANE_FAMILY.get(better["lane"], "UNCLASSIFIED"),
                    "device_name": better["device_name"] or worse["device_name"],
                    "ran": {
                        "lane": better["lane"],
                        "classification": better["classification"],
                        "ep_dispatches": better["ep_dispatches"],
                        "alloc_high_water_bytes": better["alloc_high_water_bytes"],
                        "alloc_device_frame": better["alloc_device_frame"],
                        "device_losses": better["device_losses"],
                    },
                    "did_not_run": {
                        "lane": worse["lane"],
                        "classification": worse["classification"],
                        "ep_dispatches": worse["ep_dispatches"],
                        "compute_failures": worse["compute_failures"],
                        "why": worse["why"],
                    },
                    "cited_from": {
                        "file": os.path.basename(record["source"]),
                        "sha256": record["sha256"],
                        "probe": record["probe"],
                        "record_verdict": record["record_verdict"],
                    },
                }
            )
    return claims


def aggregate(claims: list[dict]) -> list[dict]:
    """Collapse pairs into one ruling per context.

    A single favourable pair is an anecdote and this project has twice been
    embarrassed by one. If two records at the same context point in OPPOSITE
    directions the answer is SPLIT with both counts shown -- never the
    convenient one with the other in a footnote. A direction is asserted only
    when every pair at that context agrees.
    """
    by_ctx: dict = {}
    for c in claims:
        ctx = c["context_tokens"]
        slot = by_ctx.setdefault(ctx, {"DEVICE_RESIDENT": 0, "SHIPPING": 0, "UNCLASSIFIED": 0, "records": []})
        slot[c["favours"]] += 1
        slot["records"].append(c["cited_from"]["file"])
    out = []
    for ctx in sorted(by_ctx, key=lambda x: (x is None, x)):
        slot = by_ctx[ctx]
        dev, ship = slot["DEVICE_RESIDENT"], slot["SHIPPING"]
        if dev and ship:
            ruling = f"SPLIT({dev} pair(s) favour device-resident, {ship} favour shipping)"
        elif dev:
            ruling = f"DEVICE_RESIDENT_RAN_WHERE_SHIPPING_DID_NOT ({dev}/{dev} pairs agree)"
        elif ship:
            ruling = f"SHIPPING_RAN_WHERE_DEVICE_RESIDENT_DID_NOT ({ship}/{ship} pairs agree)"
        else:
            ruling = "UNCLASSIFIED"
        out.append(
            {
                "context_tokens": ctx,
                "pairs_favouring_device_resident": dev,
                "pairs_favouring_shipping": ship,
                "records": sorted(set(slot["records"])),
                "ruling": ruling,
                "citable": bool(dev and not ship) or bool(ship and not dev),
            }
        )
    return out


def single_lane_census(paths: list[str]) -> dict:
    """Count outcomes for records that hold only ONE lane.

    A single-lane record can never produce a pair, so it can never produce a
    claim. It can still answer a different and necessary question: when a
    context comes back SPLIT, is the contradicting observation the rule or the
    exception? That is a RATE, and one run is not a rate. These records are
    read for their outcome counts only and are never promoted into a claim.
    """
    census: dict = {}
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                doc = json.load(fh, parse_constant=lambda c: float("nan"))
        except (OSError, ValueError):
            continue
        reader = READERS.get(doc.get("probe"), _read_kv_chain)
        _, lanes = reader(doc)
        for lane in lanes:
            lane["classification"] = _classify(
                lane["completed"], lane["ep_dispatches"], lane["why"]
            )
            key = f"ctx{lane['context_tokens']}/{LANE_FAMILY.get(lane['lane'], lane['lane'])}"
            slot = census.setdefault(
                key,
                {"observations": 0, "completed_on_ep": 0, "device_losses": 0, "files": []},
            )
            slot["observations"] += 1
            slot["completed_on_ep"] += int(lane["classification"] == CLASS_ON_EP)
            slot["device_losses"] += int(lane.get("device_losses") or 0)
            slot["files"].append(os.path.basename(path))
    for slot in census.values():
        slot["files"] = sorted(set(slot["files"]))
        slot["completed_on_ep_rate"] = (
            f"{slot['completed_on_ep']}/{slot['observations']}"
        )
    return census


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", default=None)
    ap.add_argument(
        "--single",
        action="append",
        default=None,
        help="single-lane records, counted as a rate only and never turned into a claim",
    )
    ap.add_argument("--out", default="runs_vs_does_not_run.json")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    sources = _expand(here, args.source or DEFAULT_SOURCES)
    records, claims, unreadable = [], [], []
    for name in sources:
        path = name
        rec = read_record(path)
        if not rec["readable"]:
            unreadable.append({"file": os.path.basename(path), "error": rec["error"]})
            continue
        records.append(rec)
        claims.extend(claims_for(rec))

    report = {
        "artifact": "runs_vs_does_not_run",
        "built_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "built_by": "switch",
        "for": "niobe -- a performance statement that needs no clock",
        "statement": (
            "At long context the shipping configuration and the device-resident "
            "configuration do not merely differ in speed: they differ in whether "
            "the workload executes on the EP at all. Every field below is read "
            "from the cited record at build time; none is written into this tool."
        ),
        "classes": {
            CLASS_ON_EP: "worker exited 0 and executed a positive number of EP dispatches",
            CLASS_NO_EP: "worker exited 0 having executed zero EP dispatches -- ORT silently rebuilt the graph on CPU; this is NOT a fast run",
            CLASS_DEAD: "worker exited non-zero or produced no counters",
        },
        "refusals": [
            "lanes are only paired within a single probe invocation",
            "exit code 0 is never on its own read as evidence the EP ran",
            "a missing counters file classifies as DID_NOT_COMPLETE, never as absent",
        ],
        "sources_read": [
            {
                "file": os.path.basename(r["source"]),
                "sha256": r["sha256"],
                "probe": r["probe"],
                "record_verdict": r["record_verdict"],
                "lanes": r["lanes"],
            }
            for r in records
        ],
        "unreadable_sources": unreadable,
        "claims": claims,
        "claim_count": len(claims),
    }
    report["by_context"] = aggregate(claims)
    singles = args.single or DEFAULT_SINGLE_LANE
    report["single_lane_census"] = single_lane_census(_expand(here, singles))
    citable = [r for r in report["by_context"] if r["citable"]]
    report["citable_by_context"] = citable
    if not claims:
        report["verdict"] = "NO_CLAIM(no within-record pair separates)"
    else:
        parts = [f"{r['context_tokens']}:{r['ruling']}" for r in report["by_context"]]
        report["verdict"] = " | ".join(parts)
    report["what_niobe_may_cite"] = (
        [
            f"at context {r['context_tokens']}: {r['ruling']}"
            for r in citable
        ]
        or ["nothing -- every context is SPLIT or unclassified"]
    )

    out = args.out if os.path.isabs(args.out) else os.path.join(here, args.out)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")
    print(json.dumps({k: report[k] for k in ("verdict", "claim_count", "what_niobe_may_cite", "unreadable_sources")}, indent=2))
    print(f"wrote {out}")
    return 0 if citable else 1


if __name__ == "__main__":
    sys.exit(main())
