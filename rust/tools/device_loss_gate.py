#!/usr/bin/env python3
"""``device_loss_gate`` — a gate that can see a one-in-eight device loss, and can say what
else was in the capture when it does.

WHY THIS EXISTS
===============

Switch produced a device loss at context 4096 in the **device-resident** lane on
2026-08-03 (``bench/results/phi35_kv_chain-ctx4096-BOTH-dev0.json``), at rate 1 in 8, and
it did not reproduce in six further re-runs.  That is the exact shape Trinity's
``rust/tools/contention_gate.py`` was built for on the ``cargo test`` side:

    "A gate that runs once cannot see a rare flake."

and its companion, which is the worse one:

    "A gate whose output is truncated cannot say what failed, only that something did."

Re-running until it is green is the least sensitive instrument available and is also the
reasoning an intermittent is built to defeat.  This is the device-lane equivalent: fixed
repetitions, **detection power printed with every run including the uncomfortable
number**, full captures kept, and every fault in a capture reported rather than the first.

THE AMPLIFIER, AND WHY IT IS NOT A COST SAVING
==============================================

Trinity's amplifier was pool *narrowing*: a 21-test pool aligns on a contended global far
more often than a 505-test pool does.  The device lane has a different shape and needs a
different amplifier, derived from where the time actually goes.

MEASURED on this box (2026-08-04): one ``--lane resident --steps 2 --seed-past 4096``
observation costs ~145 s wall, of which the ORT session build over the 2.2 GB Phi-3.5
export is the overwhelming majority; the decode steps themselves are a small tail.  So
**repetitions of the whole process are the expensive axis and decode steps are the cheap
one**, and the exposure that matters — the number of submit/wait cycles the GPU actually
performs — is bought far more cheaply per second by raising ``--steps`` than by raising
``--reps``.

Consequently this gate reports the rate on **two** bases and never mixes them:

  * ``per_run`` — comparable to Switch's 1-in-8, which was counted over whole
    ``steps=2`` observations.  This is the number that may be compared to his.
  * ``per_compute`` — losses per EP ``Compute()`` call actually executed, which is the
    exposure basis and is the only one that stays meaningful when ``--steps`` changes.

A rate quoted without its basis is the mistake this file refuses to make; a ``steps=8``
run has four times the exposure of a ``steps=2`` run and its per-run rate is therefore
**not** Switch's number even when the underlying hazard is identical.

WHAT IS SCREENED, AND WHY EACH SCREEN EXISTS
============================================

  * ``dispatches_executed > 0`` — Switch found a lane that exited 0, raised nothing, and
    ran **zero** EP dispatches.  A repetition that did no EP work is not an observation of
    this hazard and is counted as ``VACUOUS``, never as a clean run.  A gate whose
    denominator includes vacuous repetitions understates the rate by construction.
  * ``compute_failures`` and ``device_losses`` from the EP's own counters — structural,
    no text, cannot be defeated by a log format change.
  * The full fault scan over the capture, via ``ci/check_device_loss.py``'s vocabulary, on
    **every** repetition — reporting *all* classes present, not the first.  A reader who
    stops at the first error in a stderr misses the second; three losses are on record
    where the reports named one.

STATES
======

    0  DEVICE-LOSS-GATE: PASS   — reps completed, no loss observed.  This is NOT "fixed";
                                  the printed detection power says what it does rule out.
    1  DEVICE-LOSS-GATE: FAIL   — at least one loss observed; capture kept and named.
    2  DEVICE-LOSS-GATE: REFUSE — every repetition was vacuous, or none ran.  A gate with
                                  no non-vacuous observation has observed nothing, and that
                                  is reported rather than rounded to green.
    4  DEVICE-LOSS-GATE: ERROR(instrument=...)

USAGE
    python rust/tools/device_loss_gate.py --reps 20
    python rust/tools/device_loss_gate.py --reps 8 --steps 2      # Switch's basis exactly
    python rust/tools/device_loss_gate.py --reps 10 --lanes resident,shipping
    python rust/tools/device_loss_gate.py --dry-run               # power table only

THE CONTROL LANE, ADDED 2026-08-04
==================================

Until the KV prefix alias landed, this gate had only one lane it could run.  The shipping
lane could not execute Phi-3.5 at ``--seed-past 4096`` at all — 355 nodes claimed, first
``Compute`` out of memory, whole graph rebuilt on CPU, **exit 0 with zero dispatches**.  A
fault sought in one lane cannot be attributed to that lane.  ``--lanes resident,shipping``
now runs both and **interleaves** them, because Tank withdrew his own ``closes_when`` on
the finding that the losses were *time-ordered, not treatment-ordered*: with lanes
alternating by repetition index, a cluster in time is visible as a cluster in time.

The record carries a ``separation`` verdict computed from the counts rather than argued in
prose — ``SEPARATES`` / ``DOES_NOT_SEPARATE`` / ``NO_LOSS_IN_EITHER_LANE`` / ``UNSCORABLE``
— and ``UNSCORABLE`` is what a lane that executed nothing produces, never a clearance.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

TOOLS = pathlib.Path(__file__).resolve().parent
REPO = TOOLS.parent.parent
PROBE = REPO / "bench" / "results" / "probe_kv_chain_phi35.py"
OUT_DIR = REPO / "bench" / "results" / "device_loss_gate"

sys.path.insert(0, str(REPO / "ci"))

TAG = "DEVICE-LOSS-GATE"

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_REFUSE = 2
EXIT_ERROR = 4

#: The two lanes the flip is scored on, and the ONE environment difference between them.
#:
#: Added 2026-08-04 (Switch). Until the KV prefix alias landed, this gate could only run
#: ``resident``: the shipping lane could not execute Phi-3.5 at seed_past 4096 at all
#: (``device_memory_ctx4096_shipping_lane_cannot_run`` — 0 dispatches, exit 0). A fault
#: sought in one lane cannot be attributed to that lane, and Tank withdrew his own
#: ``closes_when`` on exactly this ground: his losses were **time-ordered, not
#: treatment-ordered**, so a clean streak on a quiet box is produced by the box.
#:
#: ``BIND_OUTPUTS`` is pinned to its shipped value (ON) in BOTH lanes on purpose, so the
#: only pinned difference is ``DEVICE_MEMORY``. The caller-side difference — the resident
#: lane binds device ``OrtValue``s, the shipping lane feeds numpy — is not a confound to be
#: removed but the thing the flag exists to permit; it is what a user's code differs by.
LANES: dict[str, dict] = {
    "resident": {
        "probe_lane": "resident",
        "env": {
            "ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY": "1",
            "ONNXRUNTIME_EP_VULKAN_BIND_OUTPUTS": "1",
        },
    },
    "shipping": {
        "probe_lane": "host",
        "env": {
            "ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY": "0",
            "ONNXRUNTIME_EP_VULKAN_BIND_OUTPUTS": "1",
        },
    },
}

#: Rates the power table is printed against. 1/8 is Switch's measured figure; the others
#: are the rates this gate would still fail to rule out, printed so that a green is read
#: as what it is rather than as a fix.
POWER_RATES = (1 / 8, 1 / 20, 1 / 40, 1 / 100)


def _lib() -> str:
    v = os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB")
    if v:
        return v
    return str(REPO / "rust" / "target" / "release" / "onnxruntime_vulkan_ep.dll")


def _power(reps: int, rate: float) -> float:
    return 1.0 - (1.0 - rate) ** reps


def print_power(reps: int) -> None:
    print(f"\n{TAG}: detection power at {reps} repetition(s)")
    print("  rate        P(at least one loss seen)   P(NONE seen | hazard is real)")
    for r in POWER_RATES:
        p = _power(reps, r)
        print(f"  {r:8.4f}    {p:24.4f}   {1 - p:.4f}")
    print(
        "  The right-hand column is the uncomfortable number: it is the probability that\n"
        "  this gate returns PASS while the hazard is exactly as real as it was before.\n"
        "  A PASS rules out the rates whose right-hand column is small. It rules out\n"
        "  nothing about the rates whose right-hand column is not."
    )


def _fault_scan(text: str) -> dict[str, list[str]]:
    """Every fault class present in this capture, not the first one found.

    Delegates to ci/check_device_loss.py's vocabulary rather than carrying a second
    dialect — the same reason that file delegates its normalisation to
    tests/ops/_verdict.py.
    """
    try:
        import check_device_loss as cdl  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return {"instrument": [f"could not import ci/check_device_loss.py: {exc}"]}
    hits = cdl.scan_text(pathlib.Path("<capture>"), cdl.searchable_text(text))
    return {k: v for k, v in hits.items() if v}


def _counters(path: pathlib.Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def one_rep(i: int, steps: int, seed_past: int, arena: bool, budget_mb: int,
            lane: str = "resident") -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"rep{i:03d}.json"
    counters = OUT_DIR / f"rep{i:03d}.counters.json"
    capture = OUT_DIR / f"rep{i:03d}.capture.txt"
    for p in (out, counters, capture):
        p.unlink(missing_ok=True)

    spec = LANES[lane]
    env = dict(os.environ)
    env["ONNXRUNTIME_VULKAN_EP_LIB"] = _lib()
    env["ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE"] = str(counters)
    # Pinned explicitly, never inherited: the shipped default is NOT flipped, and a lane
    # that reads its configuration from whatever the parent shell happened to carry is a
    # lane whose result cannot be attributed.
    env.update(spec["env"])
    env["ONNXRUNTIME_EP_VULKAN_KV_ARENA"] = "1" if arena else "0"
    if budget_mb > 0:
        env["ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY_BUDGET_MB"] = str(budget_mb)
    else:
        env.pop("ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY_BUDGET_MB", None)

    cmd = [
        sys.executable, str(PROBE), "--worker", "--lane", spec["probe_lane"],
        "--steps", str(steps), "--seed-past", str(seed_past), "--out", str(out),
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True)
    elapsed = time.time() - t0

    text = (proc.stdout or b"").decode("utf-8", "replace") + "\n" + (
        proc.stderr or b""
    ).decode("utf-8", "replace")
    capture.write_text(text, encoding="utf-8")

    c = _counters(counters)
    doc = _counters(out)
    dispatches = int(c.get("dispatches_executed") or 0)
    rec = {
        "rep": i,
        "lane": lane,
        "exit_code": proc.returncode,
        "elapsed_s": round(elapsed, 1),
        "dispatches_executed": dispatches,
        "compute_calls": int(c.get("compute_calls") or 0),
        "compute_failures": int(c.get("compute_failures") or 0),
        "device_losses": int(c.get("device_losses") or 0),
        "capture": str(capture.relative_to(REPO)),
        "faults": _fault_scan(text),
        "steps_recorded": len(doc.get("per_step") or []),
    }
    rec["lost"] = bool(rec["device_losses"] or rec["faults"].get("device_lost_reported"))
    # A repetition that ran no EP dispatch did not observe this hazard. It is not a clean
    # run and must not enter the denominator.
    #
    # EXCEPT when it lost the device: MEASURED 2026-08-04, a repetition can lose the device
    # on its FIRST Compute, before a single dispatch is recorded (`dispatches_executed` is
    # tallied per successful Compute), so `dispatches == 0` and `device_losses == 1` co-occur.
    # Screening that out as vacuous would drop the loss from both numerator and denominator
    # and make the worst observation invisible — the exact shape of the 0-dispatch lane
    # Switch found, wearing the screen that exists to catch it.
    rec["vacuous"] = dispatches == 0 and not rec["lost"]
    return rec


def score_separation(lanes: list[str], per_lane: dict) -> dict | None:
    """The discriminator, applied mechanically instead of in prose.

    *A gate blocks the flip only if its failure SEPARATES the lanes.* Reported only when
    both lanes were actually observed non-vacuously — a lane that executed nothing
    separates nothing, and saying otherwise is how a broken control lane becomes evidence.

    ``SEPARATES`` carries its own error bar. One loss in ten repetitions of lane A and none
    in ten of lane B is *not* a separation: if B's true rate equalled A's observed 1/10, B
    would still show zero in ten runs about 35% of the time. That probability is computed
    and printed, and above 5% the verdict is downgraded to ``SEPARATES_UNDERPOWERED`` — the
    honest reading of an absence in a clean lane that was never run often enough to speak.
    """
    if len(lanes) < 2:
        return None
    if any(per_lane[ln]["nonvacuous"] == 0 for ln in lanes):
        return {
            "verdict": "UNSCORABLE",
            "why": "at least one lane produced no non-vacuous observation, so the lanes "
                   "were never compared: "
                   + ", ".join(f"{ln}={per_lane[ln]['nonvacuous']}nv" for ln in lanes),
        }
    if all(per_lane[ln]["losses"] == 0 for ln in lanes):
        return {
            "verdict": "NO_LOSS_IN_EITHER_LANE",
            "why": "no lane lost the device, so this run separates nothing in either "
                   "direction. Per the UNIFORM rule it is evidence about the mechanism "
                   "only up to the detection power printed above; it is NOT a clearance.",
        }
    if all(per_lane[ln]["losses"] > 0 for ln in lanes):
        return {
            "verdict": "DOES_NOT_SEPARATE",
            "why": "the fault occurred with the flag ON and with it OFF. Turning the flag "
                   "off does not avoid it, so it is a project defect and NOT a reason to "
                   "keep the default off.",
        }
    hit = [ln for ln in lanes if per_lane[ln]["losses"] > 0]
    clean = [ln for ln in lanes if per_lane[ln]["losses"] == 0]
    # Highest observed rate among the lanes that were hit, on the per-run basis the clean
    # lanes' repetition counts are also on.
    p_hat = max((per_lane[ln]["rate_per_run"] or 0.0) for ln in hit)
    quiet = {
        ln: (1.0 - p_hat) ** per_lane[ln]["nonvacuous"] for ln in clean
    }
    worst = max(quiet.values()) if quiet else 1.0
    verdict = "SEPARATES" if worst <= 0.05 else "SEPARATES_UNDERPOWERED"
    return {
        "verdict": verdict,
        "lanes_hit": hit,
        "lanes_clean": clean,
        "rate_per_run_in_hit_lane": p_hat,
        "p_clean_lane_silent_at_that_rate": quiet,
        "why": (
            "the fault occurred in " + ",".join(hit) + " and not in " + ",".join(clean)
            + f". If the clean lane's true rate equalled the hit lane's observed "
              f"{p_hat:.4f}, it would still show zero at probability {worst:.4f}. "
            + ("That is small enough to read the absence as a difference."
               if worst <= 0.05 else
               "That is NOT small, so the absence is what a clean lane run this few times "
               "looks like whether or not the flag matters. The blocker survives its own "
               "discriminator only in the weak sense that it has not yet been seen with "
               "the flag off.")
        ),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed-past", type=int, default=4096)
    ap.add_argument("--arena", action="store_true",
                    help="pin ONNXRUNTIME_EP_VULKAN_KV_ARENA=1 (this probe binds distinct "
                         "present OrtValues, so the EP refuses — measured 2026-08-04)")
    ap.add_argument("--budget-mb", type=int, default=0,
                    help="pin ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY_BUDGET_MB (0 = unset, "
                         "which is the shipped default and is UNCAPPED)")
    ap.add_argument("--dry-run", action="store_true", help="power table only, no runs")
    ap.add_argument("--lanes", default="resident",
                    help="comma-separated lanes from " + ",".join(LANES)
                         + ". More than one INTERLEAVES them (rep i takes lane "
                           "i %% len(lanes)), so lane assignment is orthogonal to time "
                           "order — the confound that made Tank withdraw his closes_when.")
    ap.add_argument("--record", default="device_loss_gate.json")
    ap.add_argument("--rescore", default="",
                    help="recompute `separation` from an EXISTING record's per-lane counts "
                         "and rewrite it in place. No runs. Exists so a scoring rule that "
                         "was wrong is applied to the observations already paid for, "
                         "instead of the observations being re-manufactured under the new "
                         "rule.")
    args = ap.parse_args(argv)

    if args.rescore:
        path = REPO / "bench" / "results" / args.rescore
        if not path.is_file():
            print(f"{TAG}: ERROR(instrument=no_such_record) {path}")
            return EXIT_ERROR
        doc = json.loads(path.read_text(encoding="utf-8"))
        old = doc.get("separation")
        doc["separation"] = score_separation(
            list(doc.get("arm", {}).get("lanes") or []), doc.get("per_lane") or {}
        )
        doc.setdefault("rescored", []).append({"from": old, "to": doc["separation"]})
        path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        print(f"{TAG}: rescored {path.relative_to(REPO)}")
        print(f"  was: {(old or {}).get('verdict')}")
        print(f"  now: {(doc['separation'] or {}).get('verdict')}")
        print(f"  {(doc['separation'] or {}).get('why')}")
        return EXIT_PASS

    lanes = [s.strip() for s in args.lanes.split(",") if s.strip()]
    unknown = [ln for ln in lanes if ln not in LANES]
    if not lanes or unknown:
        print(f"{TAG}: ERROR(instrument=unknown_lane) {unknown or '<empty>'}; "
              f"known lanes: {','.join(LANES)}")
        return EXIT_ERROR

    print_power(args.reps)
    if args.dry_run:
        return EXIT_PASS

    if not PROBE.is_file():
        print(f"{TAG}: ERROR(instrument=probe_missing) {PROBE}")
        return EXIT_ERROR
    if not pathlib.Path(_lib()).is_file():
        print(f"{TAG}: ERROR(instrument=ep_lib_missing) {_lib()} — "
              "ONNXRUNTIME_VULKAN_EP_LIB names nothing readable, and a lane that loads no "
              "EP will screen green having exercised nothing.")
        return EXIT_ERROR

    reps: list[dict] = []
    for i in range(args.reps):
        lane = lanes[i % len(lanes)]
        rec = one_rep(i, args.steps, args.seed_past, args.arena, args.budget_mb, lane)
        reps.append(rec)
        flag = "LOST" if rec["lost"] else ("VACUOUS" if rec["vacuous"] else "ok")
        print(
            f"  rep {i:3d}  {lane:9s} {flag:8s} exit={rec['exit_code']} "
            f"dispatches={rec['dispatches_executed']} "
            f"compute={rec['compute_calls']}/{rec['compute_failures']}f "
            f"losses={rec['device_losses']} {rec['elapsed_s']}s "
            + (" faults=" + ",".join(sorted(rec["faults"])) if rec["faults"] else ""),
            flush=True,
        )
        # Every fault class in the capture, every time — not the first, and not only on
        # the repetition that lost the device.
        for cls, lines in sorted(rec["faults"].items()):
            for ln in lines[:6]:
                print(f"        [{cls}] {ln[:220]}")

    nonvacuous = [r for r in reps if not r["vacuous"]]
    lost = [r for r in reps if r["lost"]]
    computes = sum(r["compute_calls"] for r in nonvacuous)

    per_lane = {}
    for ln in lanes:
        lreps = [r for r in reps if r["lane"] == ln]
        lnv = [r for r in lreps if not r["vacuous"]]
        llost = [r for r in lreps if r["lost"]]
        lcomp = sum(r["compute_calls"] for r in lnv)
        per_lane[ln] = {
            "env": LANES[ln]["env"],
            "probe_lane": LANES[ln]["probe_lane"],
            "reps": len(lreps),
            "nonvacuous": len(lnv),
            "vacuous": len(lreps) - len(lnv),
            "losses": len(llost),
            "computes_observed": lcomp,
            "rate_per_run": (len(llost) / len(lnv)) if lnv else None,
            "rate_per_compute": (len(llost) / lcomp) if lcomp else None,
            "loss_rep_indices": [r["rep"] for r in llost],
        }

    # The discriminator, applied mechanically instead of in prose — see `score_separation`.
    separation = score_separation(lanes, per_lane)

    doc = {
        "gate": "device_loss_gate",
        "arm": {
            "lanes": lanes,
            "interleaved": len(lanes) > 1,
            "seed_past": args.seed_past,
            "steps": args.steps,
            "env_common": {
                "ONNXRUNTIME_EP_VULKAN_KV_ARENA": "1" if args.arena else "0",
                "ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY_BUDGET_MB": (
                    str(args.budget_mb) if args.budget_mb > 0 else "<unset — UNCAPPED>"
                ),
            },
            "ep_lib": _lib(),
        },
        "per_lane": per_lane,
        "separation": separation,
        "time_order": [
            {"rep": r["rep"], "lane": r["lane"], "lost": r["lost"],
             "vacuous": r["vacuous"], "elapsed_s": r["elapsed_s"]}
            for r in reps
        ],
        "time_order_note": (
            "Printed because Tank withdrew his own closes_when on this ground: the losses "
            "he saw were time-ordered, not treatment-ordered. With lanes interleaved, a "
            "run of losses that clusters in REP INDEX rather than in LANE is the box, not "
            "the flag — and this table is what makes those two distinguishable."
        ),
        "reps_requested": args.reps,
        "reps_nonvacuous": len(nonvacuous),
        "reps_vacuous": len(reps) - len(nonvacuous),
        "losses": len(lost),
        "rate_per_run": (len(lost) / len(nonvacuous)) if nonvacuous else None,
        "rate_per_compute": (len(lost) / computes) if computes else None,
        "computes_observed": computes,
        "basis_note": (
            "rate_per_run is comparable to Switch's 1/8 ONLY at --steps 2; at any other "
            "--steps the exposure per run differs and rate_per_compute is the comparable "
            "quantity."
        ),
        "detection_power": {f"{r:.4f}": _power(len(nonvacuous), r) for r in POWER_RATES},
        "reps": reps,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rec_path = REPO / "bench" / "results" / args.record
    rec_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    print(f"\n{TAG}: {len(nonvacuous)}/{len(reps)} non-vacuous, {computes} Compute() "
          f"call(s) observed, {len(lost)} device loss(es).")
    for ln in lanes:
        p = per_lane[ln]
        print(f"    lane {ln:9s} {p['nonvacuous']}/{p['reps']} non-vacuous, "
              f"{p['computes_observed']} Compute(), {p['losses']} loss(es)"
              + (f" at reps {p['loss_rep_indices']}" if p["losses"] else ""))
    if separation:
        print(f"    SEPARATION: {separation['verdict']} — {separation['why']}")
    print(f"  record: {rec_path.relative_to(REPO)}")
    print_power(len(nonvacuous))

    if not nonvacuous:
        print(f"{TAG}: REFUSE — every repetition was vacuous (0 EP dispatches). A gate "
              "with no non-vacuous observation has observed nothing.")
        return EXIT_REFUSE
    # A requested lane that produced NO non-vacuous observation has observed nothing, even
    # when a sibling lane observed plenty. Without this, an interleaved run in which the
    # resident lane crashed at every repetition and the shipping lane ran cleanly would exit
    # PASS on the shipping lane's evidence — a green produced by the arm that was not the
    # question. MEASURED 2026-08-04: that is exactly the state main was in.
    starved = [ln for ln in lanes if per_lane[ln]["nonvacuous"] == 0]
    if starved:
        print(f"{TAG}: REFUSE — lane(s) {','.join(starved)} produced no non-vacuous "
              "observation, so the comparison this gate exists to make was never made. "
              "A lane that executed nothing is not a clean lane.")
        return EXIT_REFUSE
    if lost:
        print(f"{TAG}: FAIL — {len(lost)} loss(es) in lane(s) "
              + ",".join(sorted({r["lane"] for r in lost})) + "; captures kept at "
              + ", ".join(r["capture"] for r in lost))
        return EXIT_FAIL
    print(f"{TAG}: PASS — no loss in {len(nonvacuous)} non-vacuous repetition(s). This is "
          "not a fix and must not be quoted as one; read the power table above for what a "
          "PASS at this repetition count does and does not rule out.")
    return EXIT_PASS


if __name__ == "__main__":
    sys.exit(main())
