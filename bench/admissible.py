"""Whether a stored performance number may be quoted at all.

Every guard this project owns runs at *measurement* time, inside the process that produced the
number. That leaves a gap this file closes: a JSON artifact sitting in ``bench/results/`` carries
no guard with it. It is read by a human, or pasted into a status report, or differenced against
another file, long after the process that wrote it exited. **Nothing re-checks it.**

Three fabricated results on this project came through that gap:

* a 1.70x speedup through an EP that could not load -- ORT *printed* the error and did not raise
* a 1.45x speedup through an EP that loaded and declined every node
* a "GQA speedup" obtained by differencing two runs whose **CPU baselines were 18x apart**

None of those raised anything. All three are visible in the stored artifact, if something looks.

The rule this file enforces is the one that governs the rest of ``bench/``: **refuse, never warn.**
A file that cannot demonstrate its own admissibility is graded ``INADMISSIBLE`` and this module
exits non-zero, rather than being annotated with a caution that gets dropped the moment the number
is copied out of it.

Admissibility is not a quality judgement. A slow honest number is admissible. A fast number whose
provenance cannot be reconstructed is not.

Run::

    python -m bench.admissible                 # grade every file in bench/results/
    python -m bench.admissible --json out.json # and store the grades

Exit codes: ``0`` every artifact admissible (or explicitly withdrawn), ``1`` at least one
inadmissible artifact is present, ``2`` nothing could be graded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"

#: Grades. ADMISSIBLE numbers may be quoted with their caveats. WITHDRAWN is an artifact whose
#: own author has already retracted it -- it is not a failure, it is the system working.
ADMISSIBLE = "ADMISSIBLE"
INADMISSIBLE = "INADMISSIBLE"
WITHDRAWN = "WITHDRAWN"
NOT_A_RESULT = "NOT_A_RESULT"


def _get(d: dict, *path, default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _records(doc: dict) -> "list[dict]":
    """The per-device **timing** records inside a result document, whatever shape it is in.

    A record with no ``vulkan.median_ms`` is not a timing; auditing it would grade a
    device-capability report or a timestamp audit against gates that were never meant for it, and
    a false red is as corrosive to a falsifier's authority as a false green.
    """
    def timings(v):
        return [r for r in v if isinstance(r, dict) and _get(r, "vulkan", "median_ms") is not None]

    for key in ("results", "devices", "records"):
        v = doc.get(key)
        if isinstance(v, list) and v:
            found = timings(v)
            if found:
                return found
    return [doc] if _get(doc, "vulkan", "median_ms") is not None else []


# ---------------------------------------------------------------------------------------------
# The gates. Each returns (ok, verdict_text). `ok=False` is fatal to the artifact.
# ---------------------------------------------------------------------------------------------

def _gate_ep_loaded(rec: dict) -> "tuple[bool, str]":
    """S10.0 / the two fabricated speedups: the EP must have loaded *and* claimed something."""
    providers = _get(rec, "providers") or _get(rec, "session_providers") or []
    claimed = _get(rec, "claimed_nodes")
    if claimed is None:
        claimed = _get(rec, "dispatch_accounting", "islands")
    if claimed is None:
        claimed = _get(rec, "islands")
    ep_present = any("Vulkan" in str(p) for p in providers) if providers else None

    if ep_present is False:
        return False, ("the EP is not in session.get_providers(); this ran on CPU under our "
                       "provider's name. This is the 1.70x defect exactly.")
    if not claimed:
        return False, (f"claimed node count is {claimed!r}. An EP that declines everything still "
                       f"produces a timing. This is the 1.45x defect exactly.")
    if ep_present is None:
        return False, ("the artifact does not record session.get_providers(), so it cannot show "
                       "the EP was ever loaded. Absence of the check is not evidence it passed.")
    return True, f"EP in providers, {claimed} node(s)/island(s) claimed"


def _gate_equivalence(rec: dict) -> "tuple[bool, str]":
    """S10.0: no performance number may be quoted beside a non-MATCH verdict.

    **Vocabulary is set by ``tests/ops/_verdict.py`` -- that module is canonical and this is a
    mirror.** Five tokens, and this gate must name the one it saw rather than collapsing them
    into "not MATCH", because Morpheus's disclosure obligation says the wall-clock ratio has to
    *publish* ``UNATTRIBUTED`` when that is the state. A ratio reported beside "non-MATCH" tells
    the reader the kernels might be wrong; a ratio reported beside ``UNATTRIBUTED`` tells them
    the kernels did not run and the number is a CPU number. Those are different retractions.

    The two distinctions that must survive this function:

    * ``UNATTRIBUTED`` is **not** ``DIVERGENT``. DIVERGENT = our kernels computed a wrong answer.
      UNATTRIBUTED = our kernels executed zero nodes, so the answer -- correct or not -- is not
      about them. Different owner, different fix.
    * ``UNATTRIBUTED`` is **not** ``UNMEASURED``. UNMEASURED = the comparison never ran.
      UNATTRIBUTED = the comparison ran, agreed, and was about the wrong world. The second is
      far more dangerous precisely because it looks like a pass.

    Since the third metric amendment the verdict is recorded twice: the bare token at
    ``model_output_equivalence`` (which ``counters.rs::extract_equivalence`` parses) and the
    full record at ``model_output_equivalence_record``. When the record is present its
    ``executed_by`` frame is the attribution, and a ``MATCH`` with no frame is refused here for
    the same reason ``epctl`` refuses it: a verdict that does not carry what executed the model
    is not a verdict about this EP.
    """
    # Mirrors tests/ops/_verdict.py. Canonical definitions live there.
    EQ_MATCH = "MATCH"
    EQ_DIVERGENT = "DIVERGENT"
    EQ_UNMEASURED = "UNMEASURED"
    EQ_UNATTRIBUTED = "UNATTRIBUTED"
    EQ_SPLIT_FRAME = "SPLIT-FRAME"

    record = _get(rec, "model_output_equivalence_record")
    v = _get(rec, "model_output_equivalence") or _get(rec, "equivalence") or EQ_UNMEASURED
    if isinstance(v, dict):
        record = record if isinstance(record, dict) else v
        v = v.get("verdict", EQ_UNMEASURED)
    token = str(v).upper()

    frame = record.get("executed_by") if isinstance(record, dict) else None

    if token == EQ_UNATTRIBUTED:
        return False, (
            "model_output_equivalence is UNATTRIBUTED: the comparison ran and agreed, but this "
            "EP executed zero nodes in the run that produced the outputs. This is NOT DIVERGENT "
            "-- nothing disagreed -- and it is NOT UNMEASURED -- the instrument did run. Any "
            "wall-clock ratio from this run is a CPU-vs-CPU ratio and must publish the token "
            f"UNATTRIBUTED beside it (S10.0 disclosure obligation). executed_by={frame!r}."
        )
    if token == EQ_SPLIT_FRAME:
        return False, (
            "model_output_equivalence is SPLIT-FRAME: ORT's profile and our own "
            "dispatches_executed counter disagree about whether this EP ran. One of the two "
            "witnesses is lying and we do not know which, so nothing may be reported from this "
            f"run -- not the triple, not the ratio. executed_by={frame!r}."
        )
    if token == EQ_DIVERGENT:
        return False, (
            "model_output_equivalence is DIVERGENT: the EP ran and produced a wrong answer. "
            "This is a confirmed correctness failure, not an attribution problem."
        )
    if token != EQ_MATCH:
        return False, (
            f"model_output_equivalence is {v}. S10.0 gates the triple on MATCH; "
            f"UNMEASURED is the default and is not a pass."
        )

    # MATCH from here. A MATCH is only admissible if it carries the frame that says what
    # executed the model -- the whole point of the third metric amendment.
    if record is None:
        return False, (
            "model_output_equivalence is MATCH but there is no "
            "model_output_equivalence_record, so the verdict carries no executed_by frame. "
            "Every MATCH recorded before 2026-07-31 had this shape and one of them was a "
            "CPU-vs-CPU comparison. An unattributed MATCH is treated as UNATTRIBUTED."
        )
    if not frame:
        return False, (
            f"model_output_equivalence is MATCH but executed_by is {frame!r} -- an empty frame "
            "cannot show this EP executed anything. Treated as UNATTRIBUTED."
        )
    own = 0
    if isinstance(frame, dict):
        own = sum(int(n) for p, n in frame.items() if "Vulkan" in str(p))
    if own <= 0:
        return False, (
            f"model_output_equivalence is MATCH with executed_by={frame!r}: zero nodes attributed "
            "to this EP. MATCH is unrepresentable at a zero own-provider count (S10.0 third "
            "metric amendment); treated as UNATTRIBUTED, which is NOT DIVERGENT."
        )
    return True, f"model_output_equivalence MATCH, executed_by={frame!r}"


def _gate_device_identity(rec: dict) -> "tuple[bool, str]":
    """The selector index is not the Vulkan enumeration index. A number labelled with the wrong
    device is not a small error -- it inverted a whole day's finding."""
    di = _get(rec, "device_identity")
    if di is None:
        return False, ("no device_identity check. The selector index is not the enumeration "
                       "index (--device 0 is the RTX 4060, --device 1 the Iris Xe); a label "
                       "nothing verified is a guess.")
    verdict = di.get("verdict") if isinstance(di, dict) else di
    if str(verdict).upper() not in ("MATCH", "NON_DECISIVE"):
        return False, f"device_identity is {verdict}: the device label is not corroborated."
    return True, f"device_identity {verdict}"


def _gate_quiescence(rec: dict) -> "tuple[bool, str]":
    """Same device, same build, same test: 19460 ms quiet vs 184356 ms under six agents."""
    q = _get(rec, "machine_quiescence")
    if q is None:
        return False, ("no machine_quiescence record. Recording inflates 9.5x under CPU "
                       "contention; a number taken on an unknown machine has an unknown "
                       "multiplier on it.")
    verdict = q.get("verdict") if isinstance(q, dict) else q
    if str(verdict).upper() != "QUIET":
        return False, (f"machine_quiescence is {verdict}. Durations from a contended machine are "
                       f"not comparable to anything, including themselves.")
    return True, "machine_quiescence QUIET"


def _gate_validity(rec: dict) -> "tuple[bool, str]":
    mv = _get(rec, "measurement_validity")
    if mv is None:
        return False, "no measurement_validity block; the harness's own self-check did not run."
    if isinstance(mv, dict) and mv.get("ok") is False:
        return False, f"measurement_validity failed: {mv.get('detail') or mv.get('verdict')}"
    return True, "measurement_validity present"


GATES = (
    ("ep_loaded", _gate_ep_loaded),
    ("model_output_equivalence", _gate_equivalence),
    ("device_identity", _gate_device_identity),
    ("machine_quiescence", _gate_quiescence),
    ("measurement_validity", _gate_validity),
)


def phase_share_admissibility(rec: dict) -> dict:
    """Whether this record's **phase split** may be quoted — scoped separately from its timings.

    Deliberately NOT one of :data:`GATES`. ``phase_containment`` is a statement about how phase
    spans were attributed to islands; it says nothing about whether the end-to-end wall clock is
    sound. Folding it into the overall grade would make a perfectly good latency number
    inadmissible because a sub-span accounting rule was off, and a false red costs a falsifier its
    authority exactly as fast as a false green does.

    Three states, per R13, and ``ERROR`` is not a detection: it means the check did not run, so
    the shares are unquotable for want of evidence rather than because a defect was found.
    """
    f = _get(rec, "phase_pass", "analysis", "falsifiers", "phase_containment")
    if not isinstance(f, dict):
        return {"quotable": False, "state": "ABSENT",
                "detail": ("no phase_containment result in this record; the attribution of phases "
                           "to islands was never checked. Absence of a check is a refusal.")}
    state = f.get("state") or ("FAIL" if f.get("red") else "PASS")
    return {
        "quotable": state == "PASS",
        "state": state,
        "detail": f.get("instrument_error") or f.get("detail"),
        "scope": ("gates the phase split and every share derived from it. It does NOT gate the "
                  "end-to-end wall clock, which does not depend on phase attribution."),
    }


# ---------------------------------------------------------------------------------------------
# Cross-artifact check: the defect that lives *between* two admissible-looking files
# ---------------------------------------------------------------------------------------------

#: How far two CPU-EP baselines may differ before they may not be differenced or compared.
#:
#: **One number, used everywhere.** `phi35.baseline_disagreement` carried its own threshold of
#: 2.0x while this module used 1.25x, so the same claim — "the control did not move" — had two
#: answers eight-fold apart, and a run whose CPU baseline moved 1.276x between its two device
#: passes printed no warning at all while being inadmissible by this file's own rule. Two
#: instruments for one claim must not carry two constants; the looser one silently wins wherever
#: it happens to be the one that runs.
#:
#: 25% and not tighter: the CPU EP baseline is a control, re-measured minutes apart on a machine
#: that has just paged a 2.2 GB model, so some movement is honest. 25% and not looser: 27.6% is
#: larger than most of the effects anyone here is trying to measure.
BASELINE_TOL = 0.25


def baseline_comparability(a: dict, b: dict, name_a: str, name_b: str,
                           tol: float = BASELINE_TOL) -> dict:
    """Falsifier: two runs may only be differenced if their **CPU baselines** agree.

    A speedup claim is a ratio of ratios, and the denominator is the CPU EP -- which no change to
    a Vulkan EP can affect. If the CPU baseline moved between the two runs, the machine moved, and
    the difference measures the machine.

    This is the check that would have caught the GQA claim: 6226.8 ms vs 345.2 ms, **18x apart**,
    with a Vulkan-only change in between. Goes red on any baseline movement beyond ``tol``.
    """
    ca = _get(a, "cpu", "median_ms")
    cb = _get(b, "cpu", "median_ms")
    out = {
        "check": "baseline_comparability",
        "asserts": "two runs being differenced share a CPU baseline, so the difference is "
                   "attributable to the change under test",
        "runs": [name_a, name_b],
        "cpu_median_ms": [ca, cb],
        "tolerance": tol,
    }
    if not ca or not cb:
        out.update(ok=True, verdict="VACUOUS",
                   detail="at least one run has no CPU baseline; no difference is computable. "
                          "This is not a pass.")
        return out
    ratio = max(ca, cb) / min(ca, cb)
    out["baseline_ratio"] = round(ratio, 3)
    if ratio > 1 + tol:
        out.update(
            ok=False, verdict="BASELINE_MOVED",
            detail=(f"the CPU baseline moved {ratio:.1f}x between {name_a} ({ca:.1f} ms) and "
                    f"{name_b} ({cb:.1f} ms). No Vulkan-EP change can make the CPU EP "
                    f"{ratio:.1f}x faster, so the machine moved. Any speedup differenced across "
                    f"these two runs measures the machine, not the change."))
        return out
    out.update(ok=True, verdict="COMPARABLE",
               detail=f"CPU baselines within {ratio:.2f}x; the runs are differenceable.")
    return out


# ---------------------------------------------------------------------------------------------

def grade(path: Path) -> dict:
    try:
        doc = json.loads(path.read_text("utf-8"))
    except Exception as exc:
        return {"file": path.name, "grade": NOT_A_RESULT, "detail": f"unreadable: {exc!r}"}
    if not isinstance(doc, dict):
        return {"file": path.name, "grade": NOT_A_RESULT, "detail": "not a JSON object"}

    if doc.get("withdrawn"):
        return {"file": path.name, "grade": WITHDRAWN,
                "detail": str(doc.get("withdrawn_reason") or "marked withdrawn by its author")}

    recs = _records(doc)
    if not recs:
        return {"file": path.name, "grade": NOT_A_RESULT,
                "detail": "no timing record in this document (audit or metadata artifact)"}

    rows = []
    for i, rec in enumerate(recs):
        failures, passes = [], []
        for name, fn in GATES:
            ok, text = fn(rec)
            (passes if ok else failures).append(f"{name}: {text}")
        rows.append({
            "record": rec.get("device_index", i),
            "grade": ADMISSIBLE if not failures else INADMISSIBLE,
            "failed_gates": failures,
            "passed_gates": passes,
            "vulkan_median_ms": _get(rec, "vulkan", "median_ms"),
            "cpu_median_ms": _get(rec, "cpu", "median_ms"),
            "phase_shares": phase_share_admissibility(rec),
        })
    worst = INADMISSIBLE if any(r["grade"] == INADMISSIBLE for r in rows) else ADMISSIBLE
    return {"file": path.name, "grade": worst, "records": rows}


def audit(results_dir: Path = RESULTS) -> dict:
    files = sorted(results_dir.glob("*.json"))
    graded = [grade(p) for p in files]
    # Cross-artifact: any two files whose names suggest a before/after pair on the same device.
    pairs = []
    docs = {}
    for p in files:
        try:
            docs[p.name] = json.loads(p.read_text("utf-8"))
        except Exception:
            pass
    for name, doc in docs.items():
        if not name.startswith("pre-"):
            continue
        post = "post-" + name[len("pre-"):]
        if post not in docs:
            continue
        ra, rb = _records(doc), _records(docs[post])
        if ra and rb:
            pairs.append(baseline_comparability(ra[0], rb[0], name, post))
    return {
        "results_dir": str(results_dir),
        "n_files": len(files),
        "graded": graded,
        "cross_artifact": pairs,
        "inadmissible": sorted(g["file"] for g in graded if g["grade"] == INADMISSIBLE),
        "note": ("admissibility is about provenance, not speed. A slow honest number passes; a "
                 "fast number whose conditions cannot be reconstructed does not."),
    }


def report(a: dict) -> "list[str]":
    out = [f"admissibility audit of {a['results_dir']} ({a['n_files']} file(s))", ""]
    for g in a["graded"]:
        out.append(f"  {g['grade']:<13} {g['file']}")
        if g.get("detail"):
            out.append(f"                  {g['detail']}")
        for r in g.get("records") or []:
            ps = r.get("phase_shares") or {}
            if r["grade"] == INADMISSIBLE:
                vk, cpu = r["vulkan_median_ms"], r["cpu_median_ms"]
                shown = (f"vulkan {vk} ms / cpu {cpu} ms" if vk else "no timing")
                out.append(f"                  record {r['record']}: {shown} -- NOT QUOTABLE")
                for f in r["failed_gates"]:
                    out.append(f"                    x {f}")
            if ps and not ps.get("quotable"):
                out.append(f"                  record {r['record']}: phase split NOT QUOTABLE "
                           f"[{ps.get('state')}] {str(ps.get('detail'))[:110]}")
    for c in a["cross_artifact"]:
        out.append("")
        out.append(f"  {c['verdict']:<13} {' vs '.join(c['runs'])}")
        out.append(f"                  {c['detail']}")
    out.append("")
    if a["inadmissible"]:
        out.append(f"  {len(a['inadmissible'])} artifact(s) inadmissible: "
                   f"{', '.join(a['inadmissible'])}")
        out.append("  No performance claim may cite them. Re-measure or mark them withdrawn.")
        out.append("  SCOPE: this gate governs the *wall-clock* record — `vulkan`/`cpu` medians, "
                   "their delta and their ratio. It does not reach the device-clock figures in the "
                   "same file (`phases.gpu_steady_tail`, the phase shares), which are gated "
                   "separately and by their own falsifiers, because the device timestamp counter "
                   "does not see host load. Saying otherwise would make this report claim an "
                   "authority it does not have, which costs a falsifier its authority as surely "
                   "as a false green does.")
    else:
        out.append("  every stored artifact can demonstrate its own provenance.")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    if not args.results.exists():
        print(f"no results directory at {args.results}", file=sys.stderr)
        return 2
    a = audit(args.results)
    print("\n".join(report(a)))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(a, indent=2), "utf-8")
    if not a["graded"]:
        return 2
    bad = bool(a["inadmissible"]) or any(not c["ok"] for c in a["cross_artifact"])
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
