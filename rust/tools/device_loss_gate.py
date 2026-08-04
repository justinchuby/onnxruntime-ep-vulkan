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
    python rust/tools/device_loss_gate.py --dry-run               # power table only
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


def one_rep(i: int, steps: int, seed_past: int, arena: bool, budget_mb: int) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"rep{i:03d}.json"
    counters = OUT_DIR / f"rep{i:03d}.counters.json"
    capture = OUT_DIR / f"rep{i:03d}.capture.txt"
    for p in (out, counters, capture):
        p.unlink(missing_ok=True)

    env = dict(os.environ)
    env["ONNXRUNTIME_VULKAN_EP_LIB"] = _lib()
    env["ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE"] = str(counters)
    # Pinned explicitly, never inherited: the shipped default is NOT flipped, and a lane
    # that reads its configuration from whatever the parent shell happened to carry is a
    # lane whose result cannot be attributed.
    env["ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY"] = "1"
    env["ONNXRUNTIME_EP_VULKAN_BIND_OUTPUTS"] = "1"
    env["ONNXRUNTIME_EP_VULKAN_KV_ARENA"] = "1" if arena else "0"
    if budget_mb > 0:
        env["ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY_BUDGET_MB"] = str(budget_mb)
    else:
        env.pop("ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY_BUDGET_MB", None)

    cmd = [
        sys.executable, str(PROBE), "--worker", "--lane", "resident",
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
    ap.add_argument("--record", default="device_loss_gate.json")
    args = ap.parse_args(argv)

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
        rec = one_rep(i, args.steps, args.seed_past, args.arena, args.budget_mb)
        reps.append(rec)
        flag = "LOST" if rec["lost"] else ("VACUOUS" if rec["vacuous"] else "ok")
        print(
            f"  rep {i:3d}  {flag:8s} exit={rec['exit_code']} "
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

    doc = {
        "gate": "device_loss_gate",
        "arm": {
            "lane": "resident",
            "seed_past": args.seed_past,
            "steps": args.steps,
            "env": {
                "ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY": "1",
                "ONNXRUNTIME_EP_VULKAN_BIND_OUTPUTS": "1",
                "ONNXRUNTIME_EP_VULKAN_KV_ARENA": "1" if args.arena else "0",
                "ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY_BUDGET_MB": (
                    str(args.budget_mb) if args.budget_mb > 0 else "<unset — UNCAPPED>"
                ),
            },
            "ep_lib": _lib(),
        },
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
    print(f"  record: {rec_path.relative_to(REPO)}")
    print_power(len(nonvacuous))

    if not nonvacuous:
        print(f"{TAG}: REFUSE — every repetition was vacuous (0 EP dispatches). A gate "
              "with no non-vacuous observation has observed nothing.")
        return EXIT_REFUSE
    if lost:
        print(f"{TAG}: FAIL — {len(lost)} loss(es); captures kept at "
              + ", ".join(r["capture"] for r in lost))
        return EXIT_FAIL
    print(f"{TAG}: PASS — no loss in {len(nonvacuous)} non-vacuous repetition(s). This is "
          "not a fix and must not be quoted as one; read the power table above for what a "
          "PASS at this repetition count does and does not rule out.")
    return EXIT_PASS


if __name__ == "__main__":
    sys.exit(main())
