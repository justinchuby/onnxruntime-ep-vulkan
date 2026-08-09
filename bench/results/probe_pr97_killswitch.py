"""Issue #96, question Q2: is PR #97's KV-parallel kill switch bit-identical to its default path?

PR #97 documents `ONNXRUNTIME_EP_VULKAN_GQA_DECODE_KV_PARALLEL=1` as a way to force the decode GQA
kernel back to a single KV lane, and describes that path as producing identical results. That is a
falsifiable claim about a shipped escape hatch, so it is worth testing rather than repeating: an
online-softmax reduction that is combined across `W` lanes and one that never splits are different
floating-point summation orders, and "the kill switch is safe" is exactly the sentence somebody
will rely on when a model misbehaves in production.

What this measures
------------------
One library (the PR #97 build), two configurations of the same process:

  * `default` — no override; `W` comes from the host rule, so `W` is 1, 2, 4, 8 or 16 by length.
  * `forced1` — `ONNXRUNTIME_EP_VULKAN_GQA_DECODE_KV_PARALLEL=1`, so `W` is 1 everywhere.

For each KV length it compares `outputs_sha256`. At `past=32` the host rule already picks `W=1`,
so that length is a **positive control**: the two configurations must agree there bit-for-bit
whatever happens elsewhere, and a disagreement at `past=32` would mean the harness, not the
shader, is the source of the difference.

Two things this deliberately does not do
----------------------------------------
  * It publishes no latency. Bit-identity is the question; a speed figure taken here would invite
    a comparison between two configurations that were never timed under the sweep's protocol.
  * It does not treat a digest difference as a defect on its own. Reassociated floating-point
    summation legitimately changes low bits. The artifact records the equivalence verdict beside
    the digest so a reader can tell "different bits" from "different answer" -- those are separate
    findings and only the second is a bug.

This is the one place in the issue #96 work where an `ONNXRUNTIME_EP_VULKAN_*` variable is set on
purpose; the sweep and attribution passes strip all of them.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import probe_crossbuild_decode_window as base  # noqa: E402
import probe_pr97_third_arm as third  # noqa: E402

KILL_SWITCH = "ONNXRUNTIME_EP_VULKAN_GQA_DECODE_KV_PARALLEL"

#: Same lengths the sweep used, so a refusal here lines up with a refusal there.
LENGTHS = (32, 64, 128, 256, 512, 1024)


class ForcedKvParallel:
    """Adds exactly one variable to the child environment, and says so in the record.

    `run_one` resolves `child_env` through the module global, so this is the whole of the change.
    The frozen builder still strips every inherited `ONNXRUNTIME_EP_VULKAN_*` first -- the
    override is applied *after* that scrub, so it cannot be smuggled in from an operator's shell
    and silently attributed to this module.
    """

    def __init__(self, value: "str | None"):
        self.value = value
        self._saved = None

    def __enter__(self):
        self._saved = base.child_env
        value = self.value

        def patched(args, arm, counters_path, trace_path):
            env = self._saved(args, arm, counters_path, trace_path)
            if value is not None:
                env[KILL_SWITCH] = value
            return env

        base.child_env = patched
        return self

    def __exit__(self, *exc):
        base.child_env = self._saved
        return False


def run_config(args, config: str, past: int, rep: int, scratch: Path) -> dict:
    """One whole process on the PR #97 library, in one of the two configurations."""
    want = ("gqa_decode_f16:1" if config == "forced1"
            else f"gqa_decode_f16:{third.kv_parallel(past + 1)}")
    with third.WidenedWitness(), third.WitnessExpectation({"candidate": want}), \
            ForcedKvParallel("1" if config == "forced1" else None):
        rec = base.run_one(args, base.rm.PHI35.key, "decode", 1, past, "candidate", rep, scratch)
    rec["config"] = config
    rec["expected_gqa_witness"] = want
    rec["measured_gqa_witness"] = list((rec.get("path_witness") or {}).get("gqa_keys") or [])
    rec["kill_switch_set"] = (config == "forced1")
    return rec


def compare(records: list) -> list:
    rows = []
    for past in sorted({r["past"] for r in records}):
        got = {c: [r for r in records if r["past"] == past and r["config"] == c]
               for c in ("default", "forced1")}
        row = {"past": past,
               "host_rule_W": third.kv_parallel(past + 1),
               "is_positive_control": third.kv_parallel(past + 1) == 1}
        adm = {c: [r for r in v if r.get("admissible")] for c, v in got.items()}
        if not adm["default"] or not adm["forced1"]:
            row["verdict"] = "UNTESTED"
            row["why"] = ("at least one configuration produced no admissible record at this "
                          "length, so bit-identity was not tested here")
            row["refusals"] = {c: sorted({(r.get("refusal") or {}).get("reason")
                                          for r in v if not r.get("admissible")})
                               for c, v in got.items()}
            rows.append(row)
            continue

        digests = {c: sorted({r.get("outputs_sha256") for r in adm[c]}) for c in adm}
        row["digests"] = digests
        row["witnesses"] = {c: sorted({tuple(r["measured_gqa_witness"])[0]
                                       for r in adm[c] if r["measured_gqa_witness"]})
                            for c in adm}
        unstable = [c for c, d in digests.items() if len(d) != 1]
        if unstable:
            row["verdict"] = "NON-DETERMINISTIC"
            row["why"] = (f"configuration(s) {unstable} did not reproduce their own digest across "
                          "repeats, so no cross-configuration claim can be made")
        elif digests["default"] == digests["forced1"]:
            row["verdict"] = "BIT-IDENTICAL"
        else:
            row["verdict"] = "BITS-DIFFER"
            row["equivalence"] = {c: (adm[c][0].get("equivalence") or {}).get("verdict")
                                  for c in adm}
            row["why"] = ("the two configurations produced different output bits. Both still "
                          "passed the harness equivalence gate if `equivalence` reads MATCH "
                          "above; reassociated summation changes low bits without changing the "
                          "answer, and only the second would be a defect.")
        rows.append(row)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True, help="the PR #97 build")
    ap.add_argument("--commit", required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--iters", type=int, default=base.ITERS)
    ap.add_argument("--warmup", type=int, default=base.WARMUPS)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", default=str(_HERE / "_pr97_ks_scratch"))
    ap.add_argument("--lock", default=None)
    args = ap.parse_args(argv)

    lib = base.library_identity(args.lib)
    # `run_one` picks the library by arm name, and the gate checks the digest per arm; both arms
    # are the same file here because the treatment is the environment, not the build.
    args.candidate_lib = args.baseline_lib = args.lib
    args._lib_sha = {"candidate": lib["sha256"], "baseline": lib["sha256"]}

    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)

    lock = base.GpuLock(Path(args.lock) if args.lock else base.default_lock_path())
    lock.acquire()
    print(f"[lock] {lock.record['state']} after {lock.record['waited_seconds']}s wait", flush=True)
    records = []
    t0 = time.time()
    try:
        for rep in range(args.repeats):
            configs = ("default", "forced1") if rep % 2 == 0 else ("forced1", "default")
            for past in LENGTHS:
                for config in configs:
                    rec = run_config(args, config, past, rep, scratch)
                    records.append(rec)
                    state = ((rec.get("outputs_sha256") or "")[:16] if rec.get("admissible")
                             else "REFUSED " + str((rec.get("refusal") or {}).get("reason")))
                    print(f"[ks] r{rep} past={past:5d} {config:8s} "
                          f"{','.join(rec['measured_gqa_witness']) or '-':20s} {state}", flush=True)
    finally:
        lock.release()
        print(f"[lock] RELEASED after {lock.record.get('held_seconds')}s held "
              f"(nothing was killed)", flush=True)

    rows = compare(records)
    artifact = {
        "instrument": Path(__file__).name,
        "pass": "kill-switch-bit-identity",
        "issue": 96,
        "question": (f"Does {KILL_SWITCH}=1 produce the same output bits as PR #97's default "
                     "KV-parallel path?"),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "wall_seconds": round(time.time() - t0, 1),
        "library": {**lib, "commit": args.commit},
        "no_latency_published": ("This pass answers a correctness question. It records no timing "
                                 "and none may be quoted from it."),
        "positive_control": ("past=32, where the host rule already selects W=1, so both "
                             "configurations must agree bit-for-bit."),
        "rows": rows,
        "records": [{k: v for k, v in r.items() if k not in ("model", "speed")} for r in records],
        "counts": {"records": len(records),
                   "admissible": sum(1 for r in records if r.get("admissible"))},
        "exclusivity": base.sanitize_lock(lock.record),
    }
    Path(args.out).write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    for r in rows:
        print(f"  past={r['past']:5d} W={r['host_rule_W']:2d} {r['verdict']}")
    print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
