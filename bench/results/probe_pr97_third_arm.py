"""Issue #96, third arm: PR #97 (`fc1b163`) measured against the two frozen arms.

Runs under `prereg_pr97_third_arm.md`, whose digest is published on issue #96 before the first
process here starts. It exists as a *separate* driver for one reason: the instrument that produced
the frozen baseline-vs-candidate artifacts,
`probe_crossbuild_decode_window.py`, must not be edited. This module imports it and reuses its
worker, its gates, its lock, its pairing and its verdict arithmetic unchanged.

The one thing this driver adds is a **per-workload** expected witness. PR #97 adds a separate
`gqa_decode_f16` shader and takes it unconditionally at `seq_len == 1`, with a KV-parallel factor
`W` that varies with the declared cache capacity. The frozen gate holds one expected key per arm,
so this driver sets that expectation per `(arm, workload)` immediately before each process and
refuses any record whose witness is not exactly the predeclared string. That is a *stricter*
requirement than the frozen gate's, not a looser one: `W` is predicted here from the selection rule
alone, and a disagreement is a refusal.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_crossbuild_decode_window as base  # noqa: E402

_BENCH = Path(__file__).resolve().parents[1]

#: Cap and floor from `attention.rs`. Restated here so this module predicts `W` from the rule
#: rather than from anything PR #97 reports about itself.
GQA_DECODE_MAX_KV_PARALLEL = 16
GQA_DECODE_MIN_WORK_PER_LANE = 32

#: The arm whose witness vocabulary changed. Keyed by the commit so a mislabelled run cannot
#: quietly inherit the wrong expectation.
PR97_COMMIT = "fc1b163548a307e3b1be6d996673842203ebba25"

#: Every shader name that may legitimately occupy the GQA role in any arm of this comparison.
GQA_SHADER_NAMES = ("gqa_f16", "gqa_decode_f16")

# Bound once, at import, *before* anything can patch the frozen module. `gqa_witness_multi`
# delegates through this name rather than through `base.gqa_witness`, because `WidenedWitness`
# rebinds `base.gqa_witness` to `gqa_witness_multi` itself -- resolving the delegate late made the
# widened extractor call itself until the interpreter ran out of stack.
_BASE_GQA_WITNESS = base.gqa_witness


def kv_parallel(total_len_bound: int) -> int:
    """`gqa_decode_kv_parallel_with(total_len_bound, None)`, reimplemented from the rule.

    Largest power of two `<= GQA_DECODE_MAX_KV_PARALLEL` that still leaves at least
    `GQA_DECODE_MIN_WORK_PER_LANE` KV positions per lane. Deliberately a second implementation:
    if this and the shipped Rust disagree, the witness comparison fails and the record is refused,
    which is the outcome that should happen when two independent readings of a rule differ.
    """
    bound = max(1, int(total_len_bound))
    w = 1
    while w * 2 <= GQA_DECODE_MAX_KV_PARALLEL and bound // (w * 2) >= GQA_DECODE_MIN_WORK_PER_LANE:
        w *= 2
    return w


def expected_witness(arm_commit: str, key: str, phase: str, m: int, past: int) -> "str | None":
    """The exact `gqa_keys` entry this (arm, workload) must produce, or None for a no-GQA control.

    Pre-#97 arms keep the frozen expectation. On #97, every `seq_len == 1` workload — decode *and*
    prefill `M=1` — routes to the new shader, which is why prefill `M=1` is declared a treatment
    row for that arm in the addendum rather than a null.
    """
    if not base.has_gqa(key):
        return None
    if arm_commit.startswith(PR97_COMMIT[:7]):
        seq_len = m
        if seq_len == 1:
            return f"gqa_decode_f16:{kv_parallel(past + seq_len)}"
        return "gqa_decode_f16:PREFILL-UNREACHED"
    if arm_commit.startswith("85fbda2"):
        return "gqa_f16:1"
    if arm_commit.startswith("c96e7d9"):
        return "gqa_f16:"
    raise SystemExit(f"no declared witness expectation for arm at {arm_commit}")


def witness_table(commits: dict) -> list:
    """The whole predeclared expectation, emitted into the artifact before any timing is read."""
    rows = []
    for (key, phase, m, past, role) in base.WORKLOADS:
        label = base.workload_label(key, phase, m, past)
        rows.append({
            "workload": label,
            "role": role,
            "past": past if base.has_gqa(key) else None,
            "expected": {arm: expected_witness(commits[arm], key, phase, m, past)
                         for arm in ("baseline", "candidate")},
        })
    return rows


def gqa_witness_multi(counters: "dict | None") -> dict:
    """`base.gqa_witness`, widened to see `gqa_decode_f16` as well as `gqa_f16`.

    The frozen extractor matches `v.split(":")[0] == "gqa_f16"` exactly, so on a PR #97 build it
    finds no key at all and every GQA row refuses with "witness key is not the one this arm must
    produce". That is the frozen instrument being *correct about its own scope* — it was written
    when one GQA kernel existed — and it is why the third arm needs a driver rather than a flag.

    Only the extraction widens. The gate is untouched: it still demands the `gqa_keys` list equal
    exactly the one predeclared string for this (arm, workload), so a build that dispatched both
    kernels, the wrong `W`, or no GQA kernel at all is still refused.
    """
    out = _BASE_GQA_WITNESS(counters)
    variants = list(out.get("all_variants") or [])
    keys = [v for v in variants if v.split(":")[0] in GQA_SHADER_NAMES]
    out["gqa_keys"] = keys
    out["kv_parallel"] = None
    if len(keys) == 1 and keys[0].split(":")[0] == "gqa_decode_f16":
        tail = keys[0].split(":", 1)[1]
        out["kv_parallel"] = int(tail.split(",")[0]) if tail else None
    out["witness_extractor"] = "gqa_witness_multi (gqa_f16 + gqa_decode_f16)"
    return out


class WidenedWitness:
    """Installs `gqa_witness_multi` into the frozen module for the duration of a run.

    Patched rather than forked because `run_one` resolves `gqa_witness` through the module global
    at call time, so this is the whole of the change — the frozen file stays byte-identical and its
    own guard suite keeps passing against it.
    """

    def __enter__(self):
        if base.gqa_witness is gqa_witness_multi:
            raise RuntimeError("WidenedWitness is already installed; nesting would be a no-op")
        self._saved = base.gqa_witness
        base.gqa_witness = gqa_witness_multi
        return self

    def __exit__(self, *exc):
        base.gqa_witness = self._saved
        return False


class WitnessExpectation:
    """Sets the frozen gate's per-arm expectation for exactly one process, then restores it.

    The frozen module holds `EXPECTED_GQA_KEY` as a dict keyed by arm name. Writing it around a
    single `run_one` call is what lets a per-workload expectation reach an unmodified gate. The
    original mapping is restored on exit so a crash mid-sweep cannot leave a stale expectation
    behind for the next process.
    """

    def __init__(self, mapping: dict):
        self.mapping = mapping
        self._saved = None

    def __enter__(self):
        self._saved = dict(base.EXPECTED_GQA_KEY)
        base.EXPECTED_GQA_KEY.clear()
        base.EXPECTED_GQA_KEY.update(self.mapping)
        return self

    def __exit__(self, *exc):
        base.EXPECTED_GQA_KEY.clear()
        base.EXPECTED_GQA_KEY.update(self._saved)
        return False


def _attribution_per_length(records: list, lengths=None) -> dict:
    """Pure re-derivation of the attribution summary from the written records.

    Split out of `run_attribution` so `--resummarize` recomputes the published numbers with the
    same code that produced them, on a machine with no GPU.
    """
    if lengths is None:
        lengths = sorted({r.get("past") for r in records if r.get("past") is not None})
    per_length = {}
    for past in lengths:
        rows = [r for r in records if r.get("past") == past and r.get("admissible")]
        cand = [r for r in rows if r["arm"] == "candidate"]
        bases = [r for r in rows if r["arm"] == "baseline"]
        if not cand or not bases:
            per_length[str(past)] = {"refused": True, "why": "no admissible pair at this length"}
            continue

        def med_kernels(rs):
            names = set()
            for r in rs:
                names |= set(((r.get("trace") or {}).get("gpu_us_per_inference") or {}))
            return {n: statistics.median(
                [((r.get("trace") or {}).get("gpu_us_per_inference") or {}).get(n, 0.0)
                 for r in rs]) for n in names}

        per_length[str(past)] = kernel_share(med_kernels(cand), med_kernels(bases))
        per_length[str(past)]["n_pairs"] = min(len(cand), len(bases))
    return per_length


def run_attribution(args, libs, scratch: Path, lock, commits: dict) -> dict:
    """Per-kernel / per-host-phase attribution, tracer on, with the widened witness installed.

    The frozen `_attribution_main` cannot be reused directly: it calls `run_one` without the
    witness widening, so every PR #97 row would refuse before a trace was ever read. This is the
    same loop with the two context managers around the call, and it publishes no wall-clock
    latency -- the tracer's own cost is in every number here.
    """
    lengths = [int(x) for x in args.only.split(",")] if args.only else list(base.ATTRIBUTION_PAST)
    records = []
    t0 = time.time()
    for rep in range(args.repeats):
        arms = ("candidate", "baseline") if rep % 2 == 0 else ("baseline", "candidate")
        for past in lengths:
            for arm in arms:
                want = expected_witness(commits[arm], base.rm.PHI35.key, "decode", 1, past)
                with WidenedWitness(), WitnessExpectation({arm: want}):
                    rec = base.run_one(args, base.rm.PHI35.key, "decode", 1, past, arm, rep,
                                       scratch, traced=True)
                rec["role"] = "attribution"
                rec["expected_gqa_witness"] = want
                rec["measured_gqa_witness"] = list(
                    (rec.get("path_witness") or {}).get("gqa_keys") or [])
                records.append(rec)
                tr = rec.get("trace") or {}
                per = tr.get("gpu_us_per_inference") or {}
                gqa = per.get("gqa_decode_f16", per.get("gqa_f16"))
                state = "" if rec.get("admissible") else "REFUSED " + str(
                    (rec.get("refusal") or {}).get("reason"))
                print(f"[attrib] r{rep} past={past:5d} {arm:9s} "
                      f"gpu={tr.get('gpu_total_us_per_inference')}us gqa={gqa}us {state}",
                      flush=True)

    return {"records": records, "per_length": _attribution_per_length(records, lengths),
            "wall_seconds": round(time.time() - t0, 1)}


def run_pairing(args, libs, scratch: Path, lock, commits: dict) -> dict:
    records = []
    t0 = time.time()
    for rep in range(args.repeats):
        arms = ("candidate", "baseline") if rep % 2 == 0 else ("baseline", "candidate")
        for (key, phase, m, past, role) in base.WORKLOADS:
            for arm in arms:
                want = expected_witness(commits[arm], key, phase, m, past)
                with WidenedWitness(), WitnessExpectation({arm: want}):
                    rec = base.run_one(args, key, phase, m, past, arm, rep, scratch)
                rec["role"] = role
                rec["expected_gqa_witness"] = want
                rec["measured_gqa_witness"] = list(
                    (rec.get("path_witness") or {}).get("gqa_keys") or [])
                records.append(rec)
                med = (rec.get("speed") or {}).get("median_ms")
                state = (f"{med:9.2f} ms" if rec.get("admissible")
                         else "REFUSED: " + str((rec.get("refusal") or {}).get("reason")))
                print(f"[pr97] r{rep} {rec['workload']:44s} {arm:9s} {state}", flush=True)

    derived = base.resummarize_sweep(records, args.repeats)
    return {"records": records, "derived": derived, "wall_seconds": round(time.time() - t0, 1)}


def witness_audit(records: list) -> dict:
    """Did every admissible record produce exactly the witness the addendum predicted?"""
    disagreements = []
    for r in records:
        want, got = r.get("expected_gqa_witness"), r.get("measured_gqa_witness")
        if want is None:
            if got:
                disagreements.append({"workload": r["workload"], "arm": r["arm"],
                                      "expected": None, "measured": got})
            continue
        if got != [want]:
            disagreements.append({"workload": r["workload"], "arm": r["arm"],
                                  "expected": want, "measured": got,
                                  "admissible": bool(r.get("admissible"))})
    return {
        "predeclared_rule": ("W = largest power of two <= 16 with "
                             "(past_len_max + seq_len) // W >= 32, past_len_max = case.past"),
        "disagreements": disagreements,
        "all_agree": not disagreements,
        "note": ("A disagreement here is a refusal, not a footnote: it means this module's reading "
                 "of the selection rule and the shipped binary's reading differ, and no speed "
                 "figure from such a record may be believed."),
    }


def kernel_share(cand: dict, base_: dict) -> dict:
    """Compare a renamed kernel as a share of device time, which is all a rename permits.

    `gqa_f16` on the pre-#97 arm and `gqa_decode_f16` on #97 are different kernels; there is no
    same-name ratio to take. What is comparable is how much of each arm's own device time the
    decode-GQA kernel occupies, and how the untouched kernels moved alongside it.
    """
    def share(d: dict, names: tuple) -> "dict | None":
        total = sum(v for v in d.values() if v)
        hit = {k: v for k, v in d.items() if k in names and v}
        if not total or not hit:
            return None
        return {"kernel": sorted(hit)[0], "us": round(sum(hit.values()), 2),
                "share_of_device": round(sum(hit.values()) / total, 4),
                "device_total_us": round(total, 2)}

    gqa_names = ("gqa_f16", "gqa_decode_f16")
    c, b = share(cand, gqa_names), share(base_, gqa_names)
    out = {"candidate": c, "baseline": b,
           "comparison": "share-of-device-time (the kernel is renamed, so no same-name ratio)"}
    if c and b:
        out["share_delta"] = round(c["share_of_device"] - b["share_of_device"], 4)
        out["us_ratio_same_workload"] = round(b["us"] / c["us"], 4) if c["us"] else None
        out["us_ratio_meaning"] = ("baseline_us / candidate_us over the two DIFFERENT kernels that "
                                   "occupy the same position in the graph; > 1 means #97's kernel "
                                   "spent less device time. This is a like-for-role comparison, "
                                   "not a like-for-kernel one.")
        # Same quantity in the *control* convention below. Kept separately and used for the band
        # test because the two conventions are reciprocals: comparing a baseline/candidate ratio
        # against a candidate/baseline spread is a category error that happens to give the right
        # answer only when the effect is large.
        out["gqa_ratio_cand_over_base"] = round(c["us"] / b["us"], 4) if b["us"] else None
    controls = {}
    for k in base.UNTOUCHED_KERNELS:
        cu, bu = cand.get(k), base_.get(k)
        if cu and bu:
            controls[k] = round(cu / bu, 4)
    out["untouched_kernel_ratios"] = controls
    out["untouched_ratio_convention"] = ("candidate_us / baseline_us; > 1 means the candidate "
                                         "spent MORE device time on a kernel this change cannot "
                                         "touch. Compare against `gqa_ratio_cand_over_base`, "
                                         "never against `us_ratio_same_workload`.")
    if controls:
        vals = list(controls.values())
        out["untouched_ratio_min"] = min(vals)
        out["untouched_ratio_max"] = max(vals)
        out["untouched_ratio_median"] = statistics.median(vals)
    return out


def resummarize(artifact: dict) -> dict:
    """Recompute every derived figure in a third-arm artifact from its own `records`, GPU-free.

    `--resummarize --check` re-derives and diffs, so a reviewer with no device can confirm that
    the published summary is a function of the published records and not of anything else. It is
    also how the artifact was regenerated after the ratio-convention defect described in
    `untouched_ratio_convention` was found and fixed.
    """
    records = artifact.get("records") or []
    if artifact.get("pass") == "attribution":
        return {"per_length": _attribution_per_length(records)}
    repeats = len({r.get("repeat") for r in records if r.get("repeat") is not None}) or 3
    derived = base.resummarize_sweep(records, repeats)
    return {**derived,
            "witness_audit": witness_audit(records),
            "addendum_verdicts": verdict_rows(derived, (artifact.get("attribution_per_length")
                                                        or None))}


def _resummarize_main(args) -> int:
    import copy as _copy
    path = Path(args.resummarize)
    artifact = json.loads(path.read_text(encoding="utf-8"))
    fresh = resummarize(artifact)
    stale = {k: artifact.get(k) for k in fresh}
    kind = "attribution" if artifact.get("pass") == "attribution" else "sweep"
    n = len(artifact.get("records") or [])
    if args.check:
        same = json.dumps(stale, sort_keys=True) == json.dumps(fresh, sort_keys=True)
        print(f"kind={kind} records={n} re-derived={'+'.join(fresh)} "
              f"reproduces={'YES' if same else 'NO'}")
        if not same:
            for k in fresh:
                if json.dumps(stale.get(k), sort_keys=True) != json.dumps(fresh[k],
                                                                          sort_keys=True):
                    print(f"  differs: {k}", file=sys.stderr)
            return 1
        return 0
    out = _copy.deepcopy(artifact)
    out.update(fresh)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"kind={kind} records={n} rewrote {path.name} from its own records")
    return 0


def verdict_rows(derived: dict, attribution: "dict | None") -> list:
    """Apply the addendum's decision table. RESOLVES-P128-REGRESSION is unreachable by design."""
    rows = []
    attribution = attribution or {}
    for p in derived["workloads"]:
        v = p.get("verdict")
        row = {"workload": p["workload"], "role": p.get("role"), "past": p.get("past"),
               "ratio_median": p.get("ratio_median"), "sweep_verdict": v}
        if v == "REFUSED":
            row["addendum_verdict"] = "INCOMPARABLE"
            row["why"] = "a gate refused this row; no speed figure exists to judge"
        elif v == "FASTER":
            row["addendum_verdict"] = "WHOLE-MODEL-FASTER"
        elif v == "SLOWER":
            row["addendum_verdict"] = "WHOLE-MODEL-SLOWER"
        else:
            k = attribution.get(str(p.get("past")))
            moved = _kernel_moved_beyond_controls(k)
            if moved is True:
                row["addendum_verdict"] = "KERNEL-ONLY"
                row["why"] = ("decode-GQA device time moved beyond the untouched-control spread "
                              "while the whole model did not move outside the band")
            elif moved is False:
                row["addendum_verdict"] = "NEUTRAL"
                row["why"] = ("neither the model nor the decode-GQA kernel moved beyond what "
                              "kernels this change cannot touch also did")
            else:
                row["addendum_verdict"] = "NEUTRAL"
                row["why"] = ("whole model inside the band; no admissible attribution pair at "
                              "this length, so KERNEL-ONLY could not be tested")
        rows.append(row)
    return rows


def _kernel_moved_beyond_controls(k: "dict | None") -> "bool | None":
    """Both sides in candidate/baseline. See `untouched_ratio_convention` for why that matters."""
    if not k or k.get("gqa_ratio_cand_over_base") is None:
        return None
    lo, hi = k.get("untouched_ratio_min"), k.get("untouched_ratio_max")
    if lo is None or hi is None:
        return None
    return not (lo <= k["gqa_ratio_cand_over_base"] <= hi)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-lib")
    ap.add_argument("--candidate-lib")
    ap.add_argument("--baseline-commit")
    ap.add_argument("--candidate-commit")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--iters", type=int, default=base.ITERS)
    ap.add_argument("--warmup", type=int, default=base.WARMUPS)
    ap.add_argument("--repeats", type=int, default=base.REPEATS)
    ap.add_argument("--out")
    ap.add_argument("--scratch", default=str(_BENCH / "results" / "_pr97_scratch"))
    ap.add_argument("--prereg", default=str(_BENCH / "results" / "prereg_pr97_third_arm.md"))
    ap.add_argument("--lock", default=None)
    ap.add_argument("--attribution", action="store_true",
                    help="per-kernel/host attribution with the tracer on; publishes no wall clock")
    ap.add_argument("--only", default=None,
                    help="comma-separated KV lengths, attribution mode only")
    ap.add_argument("--attribution-json", default=None,
                    help="an attribution artifact from this driver, so the sweep's decision table "
                         "can reach the KERNEL-ONLY branch instead of recording it untested")
    ap.add_argument("--resummarize", metavar="ARTIFACT", default=None,
                    help="re-derive an artifact's summary from its own records; opens no device. "
                         "Same shape as probe_crossbuild_decode_window.py's flag.")
    ap.add_argument("--check", action="store_true",
                    help="with --resummarize, diff instead of rewrite; exit 1 on mismatch")
    args = ap.parse_args(argv)

    if args.resummarize:
        return _resummarize_main(args)

    missing = [f"--{n.replace('_', '-')}" for n in
               ("baseline_lib", "candidate_lib", "baseline_commit", "candidate_commit", "out")
               if not getattr(args, n)]
    if missing:
        ap.error("a measuring run needs " + ", ".join(missing))

    libs = {"baseline": base.library_identity(args.baseline_lib),
            "candidate": base.library_identity(args.candidate_lib)}
    if libs["baseline"]["sha256"] == libs["candidate"]["sha256"]:
        print("[pr97] refusing: both arms point at the same library", file=sys.stderr)
        return 2
    args._lib_sha = {k: v["sha256"] for k, v in libs.items()}
    commits = {"baseline": args.baseline_commit, "candidate": args.candidate_commit}

    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)

    table = witness_table(commits)
    print("[pr97] predeclared witnesses:")
    for row in table:
        print(f"    {row['workload']:44s} {row['expected']}")

    lock = base.GpuLock(Path(args.lock) if args.lock else base.default_lock_path())
    lock.acquire()
    print(f"[lock] {lock.record['state']} after {lock.record['waited_seconds']}s wait", flush=True)
    try:
        if args.attribution:
            run = run_attribution(args, libs, scratch, lock, commits)
        else:
            run = run_pairing(args, libs, scratch, lock, commits)
    finally:
        lock.release()
        print(f"[lock] RELEASED after {lock.record.get('held_seconds')}s held "
              f"(nothing was killed)", flush=True)

    audit = witness_audit(run["records"])

    if args.attribution:
        artifact = {
            "instrument": Path(__file__).name,
            "pass": "attribution",
            "issue": 96,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "wall_seconds": run["wall_seconds"],
            "pairing": f"{args.baseline_commit[:7]} (baseline) vs {args.candidate_commit[:7]} "
                       f"(candidate)",
            "caveat": ("The tracer is on in every process here, so no latency in this section is "
                       "a wall-clock result and none may be quoted as one. It answers WHERE a "
                       "difference sits, never WHETHER there is one."),
            "instrument_provenance": (
                "ONNXRUNTIME_EP_VULKAN_TRACE + _TRACE_GPU, the EP's own tracer, present unchanged "
                "in every tree compared here. No PR #94 instrumentation is used."),
            "rename_caveat": (
                "PR #97 renames the decode GQA kernel, so there is no same-name kernel ratio. "
                "`us_ratio_same_workload` compares the two DIFFERENT kernels occupying the same "
                "position in the graph -- like-for-role, not like-for-kernel."),
            "arms": {"baseline": {**libs["baseline"], "commit": args.baseline_commit},
                     "candidate": {**libs["candidate"], "commit": args.candidate_commit}},
            "predeclared_witnesses": table,
            "witness_audit": audit,
            "per_length": run["per_length"],
            "records": [{k: v for k, v in r.items() if k != "model"} for r in run["records"]],
            "counts": {"records": len(run["records"]),
                       "admissible": sum(1 for r in run["records"] if r.get("admissible"))},
            "exclusivity": base.sanitize_lock(lock.record),
        }
        Path(args.out).write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        print(f"\n  witness agreement: {audit['all_agree']}")
        print(f"  -> {args.out}")
        return 0

    artifact = {
        "instrument": Path(__file__).name,
        "issue": 96,
        "question": ("Does PR #97's decode KV-parallel kernel change whole-model Phi-3.5 decode "
                     "speed, only the GQA kernel's device time, or neither?"),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "wall_seconds": run["wall_seconds"],
        "pairing": f"{args.baseline_commit[:7]} (baseline) vs {args.candidate_commit[:7]} "
                   f"(candidate); ratio > 1 means the candidate is faster",
        "arms": {
            "baseline": {**libs["baseline"], "commit": args.baseline_commit},
            "candidate": {**libs["candidate"], "commit": args.candidate_commit},
        },
        "environment": base.environment_record(args),
        "ratio_convention": base.RATIO_CONVENTION if hasattr(base, "RATIO_CONVENTION") else
                            "baseline_median_ms / candidate_median_ms; > 1 means candidate faster",
        "predeclared_witnesses": table,
        "witness_audit": audit,
        **run["derived"],
        "counts": {
            "records": len(run["records"]),
            "admissible": sum(1 for r in run["records"] if r.get("admissible")),
            "refused": sum(1 for r in run["records"] if not r.get("admissible")),
        },
        "records": [{k: v for k, v in r.items() if k != "model"} for r in run["records"]],
        "model_identities": base._model_identities(run["records"]),
        "exclusivity": base.sanitize_lock(lock.record),
        "frozen_instrument": {
            "path": "probe_crossbuild_decode_window.py",
            "sha256": base.rm.sha256_file(str(Path(base.__file__))),
            "note": "imported unmodified; this driver adds a per-workload witness expectation only",
        },
    }
    if Path(args.prereg).is_file():
        artifact["preregistration"] = {
            "path": Path(args.prereg).name,
            "sha256": base.rm.sha256_file(args.prereg),
            "bytes": Path(args.prereg).stat().st_size,
        }
    attribution = None
    if args.attribution_json and Path(args.attribution_json).is_file():
        attribution = json.loads(
            Path(args.attribution_json).read_text(encoding="utf-8")).get("per_length")
        artifact["attribution_source"] = {
            "path": Path(args.attribution_json).name,
            "sha256": base.rm.sha256_file(args.attribution_json),
        }
    artifact["addendum_verdicts"] = verdict_rows(run["derived"], attribution)
    Path(args.out).write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"\n  band applied: {run['derived']['band']['applied']:.4f}")
    print(f"  window: {run['derived']['window'].get('claim')}")
    print(f"  witness agreement: {audit['all_agree']}")
    print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
