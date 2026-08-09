"""Real Phi-3.5 decode, instrumented: where does the *host* wall time inside `Compute` go?

WHAT THIS IS, AND WHAT IT IS NOT
================================
This probe is the real-model control for issue #88. It runs Phi-3.5 decode on the Vulkan EP with
the host-cost tracing enabled, and reports the attribution the EP itself produced.

**It makes no speedup claim and can make none.** There is one arm. Nothing here is compared to
CUDA, to CPU-as-a-competitor, or to a previous commit. The question it answers is *"of the host
wall time inside a Compute call, how much has a name?"* — and the honest answer includes a row for
the part that does not.

WHY EVERY REPEAT IS A FRESH PROCESS
===================================
Three separate reasons, and each one alone would be sufficient:

1. The EP's counters file is written from a process-exit hook. A process that has not torn down
   has not written it.
2. The tracer reads ``ONNXRUNTIME_EP_VULKAN_TRACE`` once, on first touch of a ``OnceLock``.
3. **Counters are cumulative and must not be borrowed across repeats.** If repeats shared a
   process, repeat 1's counters would contain repeat 0's work, and every repeat after the first
   would report a first-record count of zero — a shape indistinguishable from "the witness broke".
   Each repeat here owns its own counters file, written by its own process at its own teardown,
   and the artifact records ``counters_are_per_repeat: true`` so a reader does not have to take
   that on trust.

The first repeat additionally pays cold shader compilation and pipeline creation. It is reported
**separately and never averaged into the warm repeats**, because a cold call and a warm call are
different measurements wearing the same units.

WHAT THE ARTIFACT MAY BE READ FOR
=================================
- Which host phases exist, and their cumulative microseconds over the timed window.
- The unattributed remainder, as microseconds and as a share of the whole.
- The EP's own admissibility verdict. **If it refuses, the shares below it are not a
  decomposition**, and the artifact says so on its face rather than in a footnote.
- The record-path witness: how many Computes rebuilt a command buffer versus replayed one.
- Descriptor/submit churn per inference.

WHAT IT MAY NOT BE READ FOR
===========================
- Kernel time. ``submit`` is not GPU time and ``fence_wait`` is not idle time; see `docs/PERF.md`
  §1.3.
- A per-call split *of the cumulative session totals*. The EP's phase totals are cumulative over
  the whole session, so they mix the cold first call with every warm one. This probe therefore
  does not quote them: it re-derives a per-call split by assigning each phase span to the
  ``vulkan.subgraph`` span that contains it (see ``per_call_phases``), reports the cold call
  separately, and medians only the warm ones. A window that mixes call *shapes* would still
  report a mixture, so each past length is run in its own process.
- Any comparison. There is no second arm.

PATHS
=====
Nothing absolute reaches the artifact. Home, repository root, interpreter prefix, interpreter
path, model directory and the scratch directory are all replaced with stable tokens before the
document is written, and the document is then re-screened for the same roots. If a root survives,
the probe **refuses to write** rather than publishing a leak.

Run::

    python bench/results/probe_decode_host_attribution.py --device 0 --past 128,1024 \
        --warmup 3 --iters 20 --repeats 3 \
        --out bench/results/decode_host_attribution.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import platform
import statistics
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
BENCH = HERE.parent
REPO = BENCH.parent
sys.path.insert(0, str(BENCH))

import real_model as rm  # noqa: E402
import phases as ph  # noqa: E402

SCHEMA = "decode_host_attribution/1"

EP_LIB_ENV = "ONNXRUNTIME_VULKAN_EP_LIB"
COUNTERS_ENV = "ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE"
TRACE_ENV = "ONNXRUNTIME_EP_VULKAN_TRACE"
DEVICE_ENV = "ONNXRUNTIME_EP_VULKAN_DEVICE"

#: Counters this probe reports per repeat. Absence of any of them is recorded as absence.
HOST_COST_COUNTERS = (
    "dispatches_executed",
    "descriptor_pools_created",
    "descriptor_sets_allocated",
    "descriptor_writes",
    "command_buffers_recorded",
    "queue_submits",
    "record_path_first_record",
    "record_path_replay",
    "record_path_rerecord",
)


# ---------------------------------------------------------------------------------------------
# Path hygiene — one place, applied to the whole document, then verified
# ---------------------------------------------------------------------------------------------


def _roots() -> "list[tuple[str, str]]":
    """(absolute root, replacement token), longest first so nested roots resolve correctly."""
    pairs = [
        (str(REPO), "<repo>"),
        (str(pathlib.Path(sys.prefix)), "<venv>"),
        (str(pathlib.Path(sys.executable)), "<python>"),
        (str(pathlib.Path.home()), "<home>"),
    ]
    seen, out = set(), []
    for root, token in sorted(pairs, key=lambda p: -len(p[0])):
        if root and root not in seen:
            seen.add(root)
            out.append((root, token))
    return out


def publicise(text: str, extra: "list[tuple[str, str]] | None" = None) -> str:
    """Replace every absolute root with its token, in both path-separator styles."""
    for root, token in (extra or []) + _roots():
        for variant in {root, root.replace("\\", "/"), root.replace("/", "\\")}:
            if variant:
                text = text.replace(variant, token)
                text = text.replace(variant.replace("\\", "\\\\"), token)
    return text


def surviving_roots(text: str, extra: "list[tuple[str, str]] | None" = None) -> "list[str]":
    """Which absolute roots are still literally present. Empty list is the only passing state."""
    hits = []
    for root, _token in (extra or []) + _roots():
        for variant in {root, root.replace("\\", "/"), root.replace("/", "\\")}:
            if variant and (variant in text or variant.replace("\\", "\\\\") in text):
                hits.append(root)
                break
    return sorted(set(hits))


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------
# Child: one process, one session, one (past) point
# ---------------------------------------------------------------------------------------------


def worker(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--past", type=int, required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--provider", default="vulkan", choices=("vulkan", "cpu"))
    ap.add_argument("--logits-out", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    import numpy as np
    import onnxruntime as ort

    rec: dict = {"past": a.past, "provider": a.provider}
    model_rec = rm.resolve_model(rm.PHI35)

    providers = [rm.CPU_EP]
    if a.provider == "vulkan":
        lib = os.environ.get(EP_LIB_ENV)
        if not lib or not pathlib.Path(lib).is_file():
            rec["error"] = f"{EP_LIB_ENV} unset or missing — refusing to run on CPU and call it EP"
            pathlib.Path(a.out).write_text(json.dumps(rec), encoding="utf-8")
            return 2
        try:
            ort.register_execution_provider_library(rm.EP_NAME, str(pathlib.Path(lib).resolve()))
        except Exception as exc:
            if "already registered" not in str(exc):
                rec["error"] = f"registration failed: {exc}"
                pathlib.Path(a.out).write_text(json.dumps(rec), encoding="utf-8")
                return 2
        providers = [rm.EP_NAME, rm.CPU_EP]

    case = rm.Case(rm.PHI35.key, "decode", 1, a.past, tokens=1)
    feeds = rm.build_feeds(case, np)

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(str(model_rec["path"]), opts, providers=providers)
    used = list(sess.get_providers())
    rec["providers"] = used
    if a.provider == "vulkan" and rm.EP_NAME not in used:
        rec["error"] = f"ORT fell back to {used}; a CPU run must not be reported as an EP run"
        pathlib.Path(a.out).write_text(json.dumps(rec), encoding="utf-8")
        return 2

    for _ in range(a.warmup):
        outputs = sess.run(None, feeds)
    rec["warmup"] = a.warmup

    samples = []
    for _ in range(a.iters):
        t0 = time.perf_counter()
        outputs = sess.run(None, feeds)
        samples.append((time.perf_counter() - t0) * 1000.0)
    rec["iters"] = a.iters
    rec["wall_ms"] = rm.latency_stats(samples)
    rec["feeds_digest"] = rm.feeds_digest(feeds)

    if a.logits_out:
        np.save(a.logits_out, np.asarray(outputs[0]))

    del sess  # the trace and the counters are flushed at EP teardown
    pathlib.Path(a.out).write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return 0


# ---------------------------------------------------------------------------------------------
# Parent: read what the child left behind
# ---------------------------------------------------------------------------------------------


def _summary_args(events) -> dict:
    """The last ``vulkan.session_summary`` args, or ``{}`` — never a fabricated default."""
    rows = [e for e in events if e.get("name") == "vulkan.session_summary"]
    return (rows[-1].get("args") or {}) if rows else {}


def per_call_phases(events, *, warmup: int) -> dict:
    """Split the phase spans by which Compute call contained them, then separate cold from warm.

    The EP's own totals are **cumulative over the session**, so a session with one cold call and
    twenty warm ones reports a mixture whose shares belong to no single call. On the smoke run
    that mixture was 2.35 s of ``execute`` over three calls against a measured 114 ms warm wall —
    the cold call was 95% of the total and would have been read as the steady state.

    Each phase emits one span per Compute call, except ``desc_alloc`` and ``pipeline_lookup``
    which emit one per kernel; both are handled the same way, by assigning every phase span to
    the ``vulkan.subgraph`` span whose interval contains its start.

    The first ``warmup`` calls are reported separately and never folded into the warm median.
    """
    islands = ph.subgraph_spans(events)
    if not islands:
        return {"measured": False, "why": "no vulkan.subgraph spans, so calls cannot be separated"}
    spans = ph.phase_spans(events)

    per_call: "list[dict]" = [{} for _ in islands]
    unassigned = 0
    for s in spans:
        for i, island in enumerate(islands):
            if island["ts"] <= s["ts"] < island["end"]:
                row = per_call[i].setdefault(s["phase"], {"spans": 0, "microseconds": 0})
                row["spans"] += 1
                row["microseconds"] += int(s["dur"])
                break
        else:
            unassigned += 1

    calls = [
        {
            "index": i,
            "whole_us": int(islands[i]["dur"]),
            "kernels": islands[i]["nodes"],
            "phases": per_call[i],
        }
        for i in range(len(islands))
    ]
    cold = calls[:warmup]
    warm = calls[warmup:]

    out = {
        "measured": True,
        "calls_seen": len(calls),
        "cold_calls": len(cold),
        "warm_calls": len(warm),
        "phase_spans_outside_any_call": unassigned,
        "whole_note": (
            "`whole_us` is the `vulkan.subgraph` span, which brackets the same interval as "
            "Phase::Execute (the execute region opens immediately inside it). It is used as the "
            "per-call denominator because Execute deliberately emits no span of its own."
        ),
    }
    if cold:
        out["cold"] = [
            {"index": c["index"], "whole_us": c["whole_us"],
             "phases": {k: v["microseconds"] for k, v in c["phases"].items()}}
            for c in cold
        ]
    if not warm:
        out["warm"] = {"measured": False, "why": "every call was inside the warmup window"}
        return out

    names = sorted({n for c in warm for n in c["phases"]})
    summable = {p["phase"] for p in ph.sibling_phases(spans)}

    # The residual is a PER-CALL quantity and must be computed per call before it is summarised.
    # Taking `median(whole) - sum(median(phase))` is a different number, and at past=2048 it went
    # *negative* on two of three repeats — medians of separate distributions need not sum to the
    # median of their sum. That artefact is a property of the summary, not of the EP, and it is
    # not something to clamp; it is something not to compute.
    per_call_rows = []
    for c in warm:
        attributed = sum(
            v["microseconds"] for n, v in c["phases"].items() if n in summable
        )
        per_call_rows.append({
            "whole_us": c["whole_us"],
            "attributed_us": attributed,
            "unattributed_us": c["whole_us"] - attributed,
            "unattributed_share": (
                (c["whole_us"] - attributed) / c["whole_us"] if c["whole_us"] else None
            ),
            "shares": {
                n: (c["phases"].get(n, {"microseconds": 0})["microseconds"] / c["whole_us"])
                for n in names
            } if c["whole_us"] else {},
        })

    over = [r for r in per_call_rows if r["unattributed_us"] < 0]
    medians = {
        n: statistics.median([c["phases"].get(n, {"microseconds": 0})["microseconds"] for c in warm])
        for n in names
    }
    out["warm"] = {
        "measured": True,
        "calls": len(warm),
        "median_whole_us": statistics.median([r["whole_us"] for r in per_call_rows]),
        "median_phase_us": medians,
        "summable_phases": sorted(summable & set(names)),
        "nested_phases": sorted(set(names) - summable),
        "median_attributed_us": statistics.median([r["attributed_us"] for r in per_call_rows]),
        "median_unattributed_us": statistics.median([r["unattributed_us"] for r in per_call_rows]),
        "median_unattributed_share": statistics.median(
            [r["unattributed_share"] for r in per_call_rows if r["unattributed_share"] is not None]
        ) if any(r["unattributed_share"] is not None for r in per_call_rows) else None,
        "share_of_whole": {
            n: statistics.median([r["shares"][n] for r in per_call_rows if n in r["shares"]])
            for n in names
        },
        "calls_with_parts_exceeding_the_whole": len(over),
        "caveat": (
            "each figure is the median over warm calls of a quantity computed within a single "
            "call. The summable phases are wall-clock disjoint siblings; the nested ones are "
            "already inside a sibling and must not be added to it. Because each phase's median is "
            "taken independently, the medians are NOT required to sum to the median whole — read "
            "`median_unattributed_us`, which is a median of per-call residuals, and not the "
            "difference of two medians."
        ),
        "not_a_speedup_claim": (
            "these are host-cost shares of a single arm. Nothing here is compared to anything."
        ),
    }
    if over:
        out["warm"]["measured"] = False
        out["warm"]["why"] = (
            f"{len(over)} of {len(per_call_rows)} warm calls had disjoint sibling phases summing "
            "to more than the call that contains them. The siblings are supposed to be enclosed "
            "by the whole, so this is a defect in the instrumentation, not a reading. Refused "
            "rather than clamped: a clamp would turn an instrument fault into a plausible 100%."
        )
    return out


def read_attribution(trace_path: pathlib.Path, *, warmup: int) -> dict:
    """The EP's own host attribution, plus the phase spans, or a stated absence.

    The verdict is taken from the EP, not recomputed here. Two implementations of the same
    admissibility rule are two rules, and the one in the artifact would be the one nobody
    maintains.
    """
    if not trace_path.is_file():
        return {"present": False, "why": "the child wrote no trace file"}
    events = ph.load(trace_path)
    args = _summary_args(events)
    spans = ph.phase_spans(events)
    unknown = ph.unknown_phase_spans(events)

    per_phase: dict = {}
    for s in spans:
        row = per_phase.setdefault(s["phase"], {"spans": 0, "microseconds": 0})
        row["spans"] += 1
        row["microseconds"] += int(s["dur"])

    execute_us = args.get("execute_us")
    return {
        "present": True,
        "subgraph_spans": len(ph.subgraph_spans(events)),
        "execute_us": execute_us,
        "execute_calls": args.get("execute_calls"),
        "attributed_us": args.get("attributed_us"),
        "unattributed_us": args.get("unattributed_us"),
        "unattributed_share": (
            args["unattributed_us"] / execute_us
            if execute_us and args.get("unattributed_us") is not None
            else None
        ),
        "attribution_admissible": args.get("attribution_admissible"),
        "attribution_refusal": args.get("attribution_refusal"),
        "record_path_wired": args.get("record_path_wired"),
        "phases": per_phase,
        "summable_by_bench": sorted({p["phase"] for p in ph.sibling_phases(spans)}),
        "nested_by_bench": sorted(ph.nested_phase_names(spans)),
        "unknown_phase_spans": unknown,
        "cumulative_note": (
            "every figure above is CUMULATIVE over the whole session, warmup included. A session "
            "that mixes one cold call with many warm ones reports a mixture whose share belongs "
            "to no single call. Read `per_call` for the warm steady state."
        ),
        "per_call": per_call_phases(events, warmup=warmup),
    }


def read_counters(path: pathlib.Path) -> dict:
    if not path.is_file():
        return {"present": False, "why": "the child wrote no counters file"}
    doc = json.loads(path.read_text(encoding="utf-8"))
    out = {"present": True, "abi_version": doc.get("abi_version")}
    for name in HOST_COST_COUNTERS:
        out[name] = doc.get(name)
    missing = [n for n in HOST_COST_COUNTERS if doc.get(n) is None]
    out["missing_fields"] = missing
    return out


def run_repeat(scratch: pathlib.Path, *, past: int, repeat: int, device: int,
               warmup: int, iters: int, timeout: int) -> dict:
    tag = f"past{past}_r{repeat}"
    out_path = scratch / f"rec_{tag}.json"
    counters_path = scratch / f"counters_{tag}.json"
    trace_path = scratch / f"trace_{tag}.json"
    for p in (out_path, counters_path, trace_path):
        p.unlink(missing_ok=True)

    env = dict(os.environ)
    env[COUNTERS_ENV] = str(counters_path)
    env[TRACE_ENV] = str(trace_path)
    env[DEVICE_ENV] = str(device)

    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(pathlib.Path(__file__).resolve()), "--worker",
         "--past", str(past), "--device", str(device), "--warmup", str(warmup),
         "--iters", str(iters), "--out", str(out_path)],
        env=env, capture_output=True, encoding="utf-8", errors="replace",
        timeout=timeout, check=False,
    )
    elapsed_s = time.perf_counter() - t0

    rec = json.loads(out_path.read_text(encoding="utf-8")) if out_path.is_file() else {}
    row = {
        "past": past,
        "repeat": repeat,
        "cold": repeat == 0,
        "child_exit_code": proc.returncode,
        "child_seconds": round(elapsed_s, 2),
        "providers": rec.get("providers"),
        "error": rec.get("error"),
        "warmup": rec.get("warmup"),
        "iters": rec.get("iters"),
        "wall_ms": rec.get("wall_ms"),
        "feeds_digest": rec.get("feeds_digest"),
        "counters": read_counters(counters_path),
        "attribution": read_attribution(trace_path, warmup=warmup),
        "stderr_tail": publicise((proc.stderr or "")[-1500:]),
    }
    if proc.returncode != 0 or rec.get("error"):
        row["usable"] = False
        row["why_unusable"] = rec.get("error") or f"child exited {proc.returncode}"
    else:
        row["usable"] = True
    return row


def output_equivalence(scratch: pathlib.Path, *, past: int, device: int,
                       timeout: int) -> dict:
    """One Vulkan decode against one CPU decode on identical feeds, classified by `real_model`.

    This is the correctness gate, not a bonus. An attribution taken from a session that computed
    the wrong logits is an accurate account of the cost of being wrong.
    """
    import numpy as np

    def child(provider: str) -> "tuple[int, pathlib.Path]":
        tag = f"equiv_{provider}_past{past}"
        out_path = scratch / f"rec_{tag}.json"
        logits = scratch / f"logits_{tag}.npy"
        for p in (out_path, logits):
            p.unlink(missing_ok=True)
        env = dict(os.environ)
        env[COUNTERS_ENV] = str(scratch / f"counters_{tag}.json")
        env.pop(TRACE_ENV, None)
        env[DEVICE_ENV] = str(device)
        proc = subprocess.run(
            [sys.executable, str(pathlib.Path(__file__).resolve()), "--worker",
             "--past", str(past), "--device", str(device), "--warmup", "1", "--iters", "1",
             "--provider", provider, "--logits-out", str(logits), "--out", str(out_path)],
            env=env, capture_output=True, encoding="utf-8", errors="replace",
            timeout=timeout, check=False,
        )
        return proc.returncode, logits

    vk_rc, vk_logits = child("vulkan")
    cpu_rc, cpu_logits = child("cpu")
    if vk_rc != 0 or cpu_rc != 0 or not vk_logits.is_file() or not cpu_logits.is_file():
        return {
            "measured": False,
            "why": f"vulkan child exited {vk_rc}, cpu child exited {cpu_rc}; "
                   "an unmeasured equivalence is not a passing equivalence",
        }
    verdict = rm.classify_logits(np.load(vk_logits), np.load(cpu_logits), np)
    verdict["measured"] = True
    verdict["reference"] = "ORT CPU EP, identical feeds, same process image"
    return verdict


# ---------------------------------------------------------------------------------------------
# Environment record
# ---------------------------------------------------------------------------------------------


def environment_record(device: int) -> dict:
    lib = os.environ.get(EP_LIB_ENV)
    lib_path = pathlib.Path(lib) if lib else None
    return {
        "os": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "device_selector": device,
        "device_selector_note": (
            "a selector is a request, not an identity; the device the run actually opened is in "
            "the EP's own device frame, not here"
        ),
        "ep_library_present": bool(lib_path and lib_path.is_file()),
        "ep_library_sha256": sha256_file(lib_path) if lib_path and lib_path.is_file() else None,
        "trace_env": TRACE_ENV,
        "counters_env": COUNTERS_ENV,
        "serialised": True,
        "serialised_note": (
            "repeats run one at a time in this process's own subprocess loop; nothing in this "
            "probe runs two GPU children concurrently. Whether the *box* was otherwise quiet is "
            "not something this probe can witness, and it does not claim to."
        ),
    }


# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--worker" in argv:
        return worker(argv)

    ap = argparse.ArgumentParser(description="Real Phi-3.5 decode host-cost attribution (#88)")
    ap.add_argument("--past", default="128,1024",
                    help="comma-separated past_sequence_length points")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--skip-equivalence", action="store_true",
                    help="record the equivalence as unmeasured rather than running it")
    # `bench/scratch/` is `.gitignore`d (`.gitignore:65`). The per-repeat traces, counter files
    # and logit dumps this probe writes are large and rewritten by every run; the findings that
    # depend on them live in the committed `--out` JSON, which carries the numbers rather than the
    # arrays. Defaulting this next to the artifact would leave untracked junk in a committed
    # directory after every run.
    ap.add_argument("--scratch", default=str(BENCH / "scratch" / "issue88_decode_host"))
    ap.add_argument("--out", default=str(HERE / "decode_host_attribution.json"))
    a = ap.parse_args(argv)

    scratch = pathlib.Path(a.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    past_points = [int(x) for x in a.past.split(",") if x.strip()]

    try:
        model_rec = rm.resolve_model(rm.PHI35)
    except rm.ModelUnavailable as exc:
        print(f"[#88] refusing: {publicise(str(exc))}", file=sys.stderr)
        return 2

    doc: dict = {
        "schema": SCHEMA,
        "issue": 88,
        "owner": "Switch",
        "taken_at": "2026-08-08T17:27:40.427-07:00",
        "what_this_measures": (
            "the host-side breakdown of Vulkan EP Compute wall time during real Phi-3.5 decode"
        ),
        "what_this_does_not_measure": [
            "any speedup — there is one arm and no comparison",
            "kernel time — submit is not GPU time and fence_wait is not idle time",
            "the EP's cumulative session phase totals — those mix the cold first call with the "
            "warm ones and are not quoted here; every share below is re-derived per call by "
            "span containment and medianed over warm calls only",
        ],
        "model": {
            "key": model_rec["key"],
            "family": model_rec["family"],
            "sha256": model_rec["sha256"],
            "bytes": model_rec["bytes"],
            "provenance": model_rec["provenance"],
            "resolver": model_rec["resolver"],
            "recorded_sha256": model_rec["recorded_sha256"],
            "agrees_with_recorded_provenance": model_rec["agrees_with_recorded_provenance"],
            "weights_bytes": model_rec["weights_bytes"],
            "external_data_files": len(model_rec["external_data"]["files"]),
            "path_note": "resolved at run time via bench/real_model.py; not recorded here",
        },
        "environment": environment_record(a.device),
        "counters_are_per_repeat": True,
        "counters_are_per_repeat_note": (
            "every repeat is a fresh process with its own counters file, written by its own "
            "exit hook. No repeat can borrow a count from repeat 0, and repeat 0's cold "
            "compilation is never averaged into the warm repeats."
        ),
        "runs": [],
    }

    for past in past_points:
        for repeat in range(a.repeats):
            print(f"[#88] past={past} repeat={repeat} ...", flush=True)
            doc["runs"].append(
                run_repeat(scratch, past=past, repeat=repeat, device=a.device,
                           warmup=a.warmup, iters=a.iters, timeout=a.timeout)
            )

    # ---- per-past rollup, warm repeats only, and only when the EP said it was admissible ----
    rollup = []
    for past in past_points:
        rows = [r for r in doc["runs"] if r["past"] == past and r["usable"]]
        warm = [r for r in rows if not r["cold"]]
        admissible = [r for r in warm if r["attribution"].get("attribution_admissible") is True]
        entry = {
            "past": past,
            "usable_repeats": len(rows),
            "warm_repeats": len(warm),
            "admissible_warm_repeats": len(admissible),
            "cold_repeat_excluded": any(r["cold"] for r in rows),
        }
        if not admissible:
            entry["quotable"] = False
            entry["why_not_quotable"] = (
                "no warm repeat produced an admissible attribution; the refusals are on each run "
                "row. A share taken from a refused decomposition is not a share."
            )
            refusals = sorted({
                str(r["attribution"].get("attribution_refusal"))
                for r in warm if r["attribution"].get("attribution_refusal")
            })
            entry["refusals"] = refusals
        else:
            entry["quotable"] = True
            entry["median_wall_ms"] = statistics.median(
                [r["wall_ms"]["median_ms"] for r in admissible if r.get("wall_ms")]
            ) if all(r.get("wall_ms") for r in admissible) else None

            # Every figure below is taken from the WARM per-call split, never from the EP's
            # cumulative session totals. On the smoke run the cumulative `execute` was 2.35 s
            # across three calls against a 114 ms warm wall: the cold call was 95% of the
            # cumulative total, and quoting it as a steady state would have been wrong by 20x.
            warm_calls = [
                r["attribution"]["per_call"]["warm"] for r in admissible
                if r["attribution"].get("per_call", {}).get("warm", {}).get("measured")
            ]
            entry["repeats_with_a_warm_split"] = len(warm_calls)
            if not warm_calls:
                entry["quotable"] = False
                entry["why_not_quotable"] = (
                    "the session attribution was admissible but no repeat produced a usable warm "
                    "per-call split, so there is no steady-state reading to quote. The cumulative "
                    "session totals are a cold/warm mixture and are not a substitute."
                )
            else:
                entry["median_call_us"] = statistics.median(
                    [w["median_whole_us"] for w in warm_calls]
                )
                entry["median_unattributed_share"] = statistics.median(
                    [w["median_unattributed_share"] for w in warm_calls
                     if w.get("median_unattributed_share") is not None]
                ) if all(w.get("median_unattributed_share") is not None
                         for w in warm_calls) else None
                names = sorted({n for w in warm_calls for n in w["median_phase_us"]})
                entry["warm_phase_share_of_call"] = {}
                entry["warm_phase_us"] = {}
                for name in names:
                    shares = [w["share_of_whole"][name] for w in warm_calls
                              if w.get("share_of_whole") and name in w["share_of_whole"]]
                    us = [w["median_phase_us"][name] for w in warm_calls
                          if name in w["median_phase_us"]]
                    entry["warm_phase_share_of_call"][name] = (
                        {"median": statistics.median(shares), "repeats": len(shares),
                         "summable": name in warm_calls[0]["summable_phases"]}
                        if shares else None
                    )
                    entry["warm_phase_us"][name] = (
                        {"median": statistics.median(us), "repeats": len(us)} if us else None
                    )
                entry["reading_rules"] = [
                    "only the entries marked summable=true may be added together",
                    "the rest are nested inside a summable phase and are already counted there",
                    "median_unattributed_share is the part of a warm call no phase names",
                    "no entry here is a kernel time and none of them is a comparison",
                ]

            entry["per_inference_churn"] = {}
            for name in ("descriptor_sets_allocated", "descriptor_pools_created",
                         "command_buffers_recorded", "queue_submits",
                         "record_path_first_record", "record_path_rerecord",
                         "record_path_replay"):
                vals = []
                for r in admissible:
                    c = r["counters"]
                    n = (r.get("iters") or 0) + (r.get("warmup") or 0)
                    if c.get("present") and c.get(name) is not None and n:
                        vals.append(c[name] / n)
                entry["per_inference_churn"][name] = (
                    {"median_per_inference": statistics.median(vals), "repeats": len(vals)}
                    if vals else None
                )
            entry["per_inference_churn_note"] = (
                "counters are per-process totals divided by that process's own inference count "
                "(warmup + timed). They are not per-call medians and a process whose calls "
                "differed in shape would report a mixture; every call in a decode point here has "
                "the same shape."
            )
        rollup.append(entry)
    doc["per_past"] = rollup

    doc["output_equivalence"] = (
        {"measured": False, "why": "--skip-equivalence was passed"}
        if a.skip_equivalence
        else output_equivalence(scratch, past=min(past_points), device=a.device,
                                timeout=a.timeout)
    )

    # A run whose logits are wrong has no admissible attribution, whatever the tracer said.
    equiv_ok = (
        bool(doc["output_equivalence"].get("measured"))
        and doc["output_equivalence"].get("verdict") == rm.MATCH
    )
    doc["admissibility"] = {
        "output_equivalence_established": equiv_ok,
        "any_quotable_past_point": any(e.get("quotable") for e in rollup),
        "verdict": (
            "ADMISSIBLE — attribution may be quoted for the past points marked quotable"
            if equiv_ok and any(e.get("quotable") for e in rollup)
            else "NOT ADMISSIBLE — see output_equivalence and per_past[].why_not_quotable"
        ),
        "speedup_claim": "NONE. This probe has one arm and cannot support a comparison.",
    }

    text = publicise(json.dumps(doc, indent=2), extra=[(str(scratch), "<scratch>")])
    leaks = surviving_roots(text, extra=[(str(scratch), "<scratch>")])
    if leaks:
        print(
            "[#88] refusing to write: absolute roots survived redaction: "
            f"{[pathlib.Path(x).name for x in leaks]}",
            file=sys.stderr,
        )
        return 3

    out_path = pathlib.Path(a.out)
    if not out_path.is_absolute():
        out_path = (REPO / a.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"[#88] wrote {out_path.relative_to(REPO).as_posix()}")
    print(f"[#88] {doc['admissibility']['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
