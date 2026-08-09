"""The one summarizer that owns the issue #96 decode-window evidence.

Issue #96 asks a narrow question: *is the compiled `85fbda2` library slower than the compiled
`c96e7d9` library at Phi-3.5 decode, and if so over which KV lengths?* Two cross-build runs have
now answered it differently, and the job of this module is to hold both answers without letting
either one quietly become the record.

**Why one module.** The previous two attempts spread schema, admissibility, banding, verdicts,
the window claim and serialization across a probe script and a test file, and the two drifted:
a length with no admissible timing was counted as a length that had been measured and found not
slow. A gate that lives in more than one place is a gate that can be satisfied in one place and
not the other. So everything that decides anything lives here, the probe imports it, the tests
import it, and ``--resummarize --check`` re-runs *this* code against the published summary. There
is no second implementation to disagree with.

The design rules, in the order they matter:

**1. Absent is absent.** A record is ``ACCEPTED`` or it is ``REFUSED``, and a ``REFUSED`` record
contributes *nothing*: not a timing, not a "not slow", not an edge, not a neighbour, not support
for a window. It is not evidence of no effect; it is the absence of evidence. :func:`classify_record`
is the only thing in this repository allowed to say a record counts, and it says so only for a
record that carries a finite timing whose median re-derives from its own raw samples *and* the
full witness set that says which binary produced it and that the binary was numerically right.

**2. A window needs its edges.** :func:`window_verdict` will not name a length as the boundary of
anything unless that length was itself measured and accepted. The reported point in #96 is
``past = 128``; its neighbours in the pre-registered ordered universe are 64 and 256; both were
refused in the only sweep that has been run. So the honest verdict is ``INDETERMINATE``, and this
module returns ``INDETERMINATE`` rather than a claim with imaginary edges.

**3. No arm defines and judges itself.** The band that decides whether a ratio is a real
difference has to come from records that are not the records being judged. :func:`calibration`
takes A/A allocations — same binary on both sides — and refuses to build a band out of anything
that shares a process with the treatment. A band derived from the row it is about to grade is
not a band.

What that rule does *not* establish is that the calibration ran the same protocol as the records
it grades. In this artifact it did not: :func:`protocol_delta` reads the arms' own
``inference_calls`` off the records and reports that the A/A arms issued 25 inference calls per
record where the treatment records report 27, at the same 20 timed iterations. The band is
therefore a measurement of a *neighbouring* protocol in a later session, not of the treatment
session's own noise, and nothing downstream may describe it as the latter.

**4. A verdict that cannot fail is not a verdict.** :func:`verdict_for` and the window rule are
exercised against a *planted* effect in the shipped tests: the same functions, over the same
accepted records, with one side scaled by a known factor, must come back ``SLOWER``. A planted
positive is never persisted as a record — it is a transformation, not a measurement — but if the
detector cannot see one, none of its negatives mean anything.

**5. Underpowered is not negative.** :func:`log_ratio_interval` and :func:`power_at` exist so that
a null result has to declare how much of an effect it could actually have detected. Three repeats
at the dispersion these runs show cannot distinguish "no effect" from "the effect that was
reported", and the summary says so in the same breath as the ratio. :func:`power_at` is monotonically
decreasing in the band, so a power figure computed against a band that is not known to be wide
enough is an **upper bound** on this protocol's power, never a point estimate of it.

Nothing in this module measures anything. It reads records that a probe produced and turns them
into a verdict; every number it emits is re-derivable from the records by re-running it, which is
exactly what ``--resummarize --check`` does.
"""

from __future__ import annotations

import math
import re
import statistics
from typing import Any, Iterable, Mapping, Sequence

#: Serialization contract. Bump when a consumer would misread an older summary.
SCHEMA = "decode-window-evidence/2"

#: Issue this evidence answers to.
ISSUE = 96

# --------------------------------------------------------------------------------------------
# Record status
# --------------------------------------------------------------------------------------------

#: The record carries a timing that may be used.
ACCEPTED = "ACCEPTED"

#: The record carries no usable timing. It is absent from every downstream count.
REFUSED = "REFUSED"

#: Fields a record must carry, and carry non-empty, before its timing may be believed. Each one
#: answers "which binary, on which model, in which process, producing what" — drop any of them
#: and the timing stops being attributable to an arm.
REQUIRED_WITNESSES = (
    "arm",
    "workload",
    "repeat",
    "pid",
    "started_at",
    "finished_at",
    "model_key",
    "ep_library_sha256",
    "ep_library_bytes",
    "equivalence",
    "outputs_sha256",
    "path_witness",
    "inference_calls",
)

#: The published median must re-derive from the published raw samples to within this many
#: milliseconds. Not zero: the samples are rounded for legibility when serialized, so bit-exact
#: re-derivation is impossible by construction and a guard demanding it would be a guard nobody
#: could keep. 1e-3 ms is four orders of magnitude below the smallest effect under discussion.
MEDIAN_REDERIVATION_TOL_MS = 1e-3

# --------------------------------------------------------------------------------------------
# Verdicts
# --------------------------------------------------------------------------------------------

#: Candidate slower than baseline in *every* repeat, by more than the band.
SLOWER = "SLOWER"

#: Candidate faster than baseline in *every* repeat, by more than the band.
FASTER = "FASTER"

#: Every repeat inside the band.
NEUTRAL = "NEUTRAL"

#: Repeats disagree, or there is not enough accepted data to decide. Never a synonym for "no
#: effect" — it is the verdict that declines to have one.
INDETERMINATE = "INDETERMINATE"

#: ``baseline_median_ms / candidate_median_ms``. Above 1 means the candidate is faster; below 1
#: means the candidate is slower. Stated once, here, because a ratio whose direction is ambiguous
#: has caused two rejections already, and every prose surface that quotes a ratio is checked
#: against this string by ``test_decode_window_evidence.py``.
RATIO_CONVENTION = "baseline_median_ms / candidate_median_ms; > 1 means the candidate is faster"

#: The exact phrasings that get this backwards. Any document that quotes a ratio from this module
#: is grepped for them, because the inversion is invisible when only one number is on the page:
#: 0.9651 reads as a 3.5% move in either direction and only the convention says which one.
INVERTED_CONVENTION_PHRASES = (
    "candidate ÷ baseline",
    "candidate / baseline",
    "candidate over baseline",
    "candidate_median_ms / baseline_median_ms",
    "below 1 is the candidate running faster",
    "above 1 is the candidate running slower",
    "below 1 means the candidate is faster",
    "above 1 means the candidate is slower",
    "below 1 the candidate is faster",
    "above 1 the candidate is slower",
    "under 1 the candidate is faster",
    "over 1 the candidate is slower",
)

#: Any sentence that ties a side of 1.0 to a direction, whatever words it uses. The prose guard
#: matches on this rather than on a list of known-bad phrasings, so an inversion written in wording
#: nobody enumerated is still caught: every match is checked against the convention above.
CONVENTION_DIRECTION_CLAIM = re.compile(
    r"\b(above|below|over|under)\s+1(?:\.0)?\b[^.\n]{0,60}?\bcandidate\b[^.\n]{0,30}?\b(faster|slower)\b",
    re.IGNORECASE,
)

#: Below this, a band is not a band — it is the measurement noise of a box nobody controlled.
BAND_FLOOR = 0.05

#: The decode KV lengths this investigation pre-registered, in order. The window rule reads
#: adjacency off *this* tuple, so a length that was never in the plan can never become an edge.
#: ``past = 0`` is deliberately absent: it is prefill, a different phase, and treating it as the
#: length below 32 is how a decode window acquires a neighbour from another workload.
DECODE_LENGTH_UNIVERSE = (32, 64, 128, 256, 512, 1024)

#: The length issue #96 reports as regressed. Everything about the window rule is oriented at it.
REPORTED_LENGTH = 128


# --------------------------------------------------------------------------------------------
# Admissibility — rule 1
# --------------------------------------------------------------------------------------------


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def classify_record(record: Mapping[str, Any]) -> dict:
    """Return ``{"status", "reasons"}`` for one raw record.

    ``ACCEPTED`` means every one of the following holds. ``REFUSED`` means at least one does not,
    and the record is then absent from every count this module produces — it is not a slow
    length, not a fast one, not a measured one, and not a neighbour.

    * the producer did not itself refuse the record (``admissible`` / ``refusal``),
    * a ``speed`` block exists with a finite, strictly positive ``median_ms``,
    * ``samples_ms`` is non-empty, every sample finite and strictly positive, ``n`` agrees with
      its length, and the median of the samples re-derives ``median_ms``,
    * every witness in :data:`REQUIRED_WITNESSES` is present and non-empty,
    * the run's own EP-vs-CPU equivalence verdict is ``MATCH``.

    The samples clause is the one that matters most for ``--check``: a summary that reads only
    ``median_ms`` can be reproduced from records whose raw samples were scaled or deleted, which
    is precisely the hole the previous revision shipped.
    """
    reasons: list[str] = []

    if record.get("refusal") is not None:
        reason = record["refusal"]
        detail = reason.get("reason") if isinstance(reason, Mapping) else reason
        reasons.append(f"producer refused: {detail}")
    if "admissible" in record and not record.get("admissible"):
        reasons.append("producer marked the record inadmissible")

    speed = record.get("speed")
    if not isinstance(speed, Mapping):
        reasons.append("no speed block")
    else:
        median = speed.get("median_ms")
        if not _finite(median) or median <= 0:
            reasons.append("median_ms is not a finite positive number")
        samples = speed.get("samples_ms")
        if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)) or not samples:
            reasons.append("samples_ms is missing or empty")
        elif not all(_finite(s) and s > 0 for s in samples):
            reasons.append("samples_ms contains a non-finite or non-positive sample")
        else:
            declared_n = speed.get("n")
            if declared_n is not None and declared_n != len(samples):
                reasons.append(f"speed.n {declared_n} disagrees with {len(samples)} samples")
            if _finite(median):
                rederived = statistics.median(samples)
                if abs(rederived - median) > MEDIAN_REDERIVATION_TOL_MS:
                    reasons.append(
                        f"median_ms {median} does not re-derive from samples_ms "
                        f"({rederived}, tolerance {MEDIAN_REDERIVATION_TOL_MS} ms)"
                    )

    for field in REQUIRED_WITNESSES:
        if field not in record or record[field] is None or record[field] == "":
            reasons.append(f"missing witness: {field}")

    equivalence = record.get("equivalence")
    if isinstance(equivalence, Mapping):
        verdicts = _equivalence_verdicts(equivalence)
        if not verdicts:
            reasons.append("equivalence block carries no verdict")
        elif any(v != "MATCH" for v in verdicts):
            reasons.append(f"EP-vs-CPU equivalence is not MATCH: {sorted(set(verdicts))}")
    elif "equivalence" in record:
        reasons.append("equivalence is not a mapping")

    return {"status": REFUSED if reasons else ACCEPTED, "reasons": reasons}


def _equivalence_verdicts(equivalence: Mapping[str, Any]) -> list[str]:
    """Every per-output verdict in an equivalence block, flattened.

    Gate 5 requires equivalence *per repeat and over all outputs*, so a block that matched on its
    primary output and diverged on a secondary one must not read as MATCH. Any nested mapping
    carrying a ``verdict`` counts; a bare top-level ``verdict`` counts too.
    """
    verdicts: list[str] = []
    top = equivalence.get("verdict")
    if isinstance(top, str):
        verdicts.append(top)
    for value in equivalence.values():
        if isinstance(value, Mapping) and isinstance(value.get("verdict"), str):
            verdicts.append(value["verdict"])
    return verdicts


def accepted(records: Iterable[Mapping[str, Any]]) -> list[dict]:
    """The records that :func:`classify_record` accepts. Everything else never reaches a verdict."""
    return [dict(r) for r in records if classify_record(r)["status"] == ACCEPTED]


def status_counts(records: Iterable[Mapping[str, Any]]) -> dict:
    """``{"records", "accepted", "refused"}`` — the only three numbers a reader should trust."""
    seq = list(records)
    n_accepted = sum(1 for r in seq if classify_record(r)["status"] == ACCEPTED)
    return {"records": len(seq), "accepted": n_accepted, "refused": len(seq) - n_accepted}


# --------------------------------------------------------------------------------------------
# Pairing
# --------------------------------------------------------------------------------------------


def paired(
    records: Iterable[Mapping[str, Any]],
    workload: str,
    *,
    left: str = "baseline",
    right: str = "candidate",
) -> dict:
    """Pair ``left`` against ``right`` within each repeat of one workload.

    The ratio is ``left_median_ms / right_median_ms``, and the defaults make that
    ``baseline / candidate`` — :data:`RATIO_CONVENTION`, above 1 the candidate is *faster*, below 1
    the candidate is *slower*. The direction is stated here as well as at the constant because this
    is the line that computes it, and a prose surface that inverts it has already cost one review.

    Only accepted records pair. A repeat that is missing either side contributes no ratio and is
    listed in ``incomplete_repeats`` — it does *not* silently reduce the repeat count, because a
    verdict rule phrased as "every repeat" is meaningless if the number of repeats is whatever
    survived.

    The same function serves the treatment comparison and the A/A arms; an A/A call simply passes
    the two allocations of one binary as ``left`` and ``right``. That is deliberate: the band and
    the thing it grades must be computed by identical code or the comparison is not like-for-like.
    """
    usable = [r for r in accepted(records) if r.get("workload") == workload]
    by_repeat: dict[Any, dict[str, Mapping[str, Any]]] = {}
    for record in usable:
        by_repeat.setdefault(record.get("repeat"), {})[str(record.get("arm"))] = record

    ratios: list[float] = []
    per_repeat: list[dict] = []
    incomplete: list[Any] = []
    for repeat in sorted(by_repeat, key=lambda k: (k is None, k)):
        sides = by_repeat[repeat]
        if left not in sides or right not in sides:
            incomplete.append(repeat)
            continue
        left_ms = sides[left]["speed"]["median_ms"]
        right_ms = sides[right]["speed"]["median_ms"]
        ratio = left_ms / right_ms
        ratios.append(ratio)
        per_repeat.append(
            {
                "repeat": repeat,
                "left_median_ms": left_ms,
                "right_median_ms": right_ms,
                "ratio": ratio,
                "left_pid": sides[left].get("pid"),
                "right_pid": sides[right].get("pid"),
                "left_process": _process_key(sides[left]),
                "right_process": _process_key(sides[right]),
            }
        )

    result = {
        "workload": workload,
        "left": left,
        "right": right,
        "ratio_convention": RATIO_CONVENTION,
        "n_pairs": len(ratios),
        "ratios": ratios,
        "per_repeat": per_repeat,
        "incomplete_repeats": incomplete,
        "accepted_records": len(usable),
        "past": past_of(workload),
        "role": _role_of(usable),
    }
    if not ratios:
        result["reason"] = (
            f"no repeat has an admissible record on both sides ({len(usable)} accepted "
            f"record(s) for this workload)"
        )
    if ratios:
        result["geometric_mean"] = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
        result["median_ratio"] = statistics.median(ratios)
        result["min_ratio"] = min(ratios)
        result["max_ratio"] = max(ratios)
        result["half_range"] = (max(ratios) - min(ratios)) / 2
    return result


def _role_of(records: Sequence[Mapping[str, Any]]) -> Any:
    roles = {r.get("role") for r in records if r.get("role") is not None}
    return roles.pop() if len(roles) == 1 else sorted(map(str, roles)) or None


def past_of(workload: str) -> "int | None":
    """The decode KV length a workload label encodes, or ``None`` if it is not a decode row.

    Reads ``.../decode/M<m>/past<n>``. Prefill labels return ``None`` rather than 0 so that no
    prefill row can be mistaken for the decode length below 32.
    """
    if "/decode/" not in workload:
        return None
    tail = workload.rsplit("/past", 1)
    if len(tail) != 2 or not tail[1].isdigit():
        return None
    return int(tail[1])


# --------------------------------------------------------------------------------------------
# Calibration — rule 3
# --------------------------------------------------------------------------------------------


def _process_key(record: Mapping[str, Any]) -> str:
    """Identity of the OS process that produced a record.

    A bare PID is not one. Windows recycles PIDs, and these records span two sessions separated by
    hours, so the same integer genuinely appears in both: PID **36380** produced the treatment
    ``prefill/M1/past0`` baseline record of repeat 1 (started ``2026-08-09T00:05:22``) and, five
    hours later, the ``aa-candidate`` left-side record of repeat 1 (started
    ``2026-08-09T05:03:42``). Those are two different processes wearing one integer. Pairing the
    PID with the process's own start timestamp gives an identity that is stable across sessions and
    still catches the case the contamination check is actually for: the same process appearing on
    both sides. The result is a string so that it survives a JSON round-trip unchanged.

    The recycling is a fact about these records and is asserted from them, not from this docstring:
    ``test_the_recycled_pid_this_module_names_is_the_one_in_the_artifact`` re-derives the shared
    PIDs and fails if the integer named above stops being one of them.
    """
    return f"{record.get('pid')}@{record.get('started_at')}"


def _protocol_signature(record: Mapping[str, Any]) -> dict:
    """What protocol a record says it was produced under, read off the record itself.

    Two numbers, both already required witnesses or already published: how many inference calls the
    process issued in total, and how many of them were timed. Their difference is the discarded
    warmup, which is the part no record in this artifact names directly. Private on purpose: it
    renders no verdict, and a public function in this module that cannot be watched disagreeing
    would enter the census as `unfalsified`.
    """
    speed = record.get("speed") or {}
    calls = record.get("inference_calls")
    timed = speed.get("n")
    untimed = calls - timed if isinstance(calls, int) and isinstance(timed, int) else None
    return {"inference_calls": calls, "timed_iterations": timed, "untimed_calls": untimed}


def protocol_delta(records: Iterable[Mapping[str, Any]]) -> dict:
    """Did the calibration arm run under the protocol it grades, or a neighbouring one?

    :func:`calibration` enforces that the band comes from a *different process* than the treatment.
    It does not, and cannot, enforce that the band comes from the *same protocol*, and in this
    artifact it does not: the A/A arms issue 25 inference calls per record and the treatment records
    report 27, at the same 20 timed iterations. Two untimed calls per record is a small difference
    and an unmeasured one — nothing here establishes which way it moves dispersion, because the A/A
    arm was never run at the treatment's call count.

    So this returns the observation and refuses to price it. ``matched`` is ``True`` only when every
    accepted A/A record and every accepted treatment record agree on both numbers. When it is
    ``False`` the band may not be described as a measurement of the treatment session's noise, and
    the shipped prose guard in ``test_decode_window_evidence.py`` reddens if a document does.
    """
    seen: dict[str, set] = {"treatment": set(), "calibration": set()}
    for record in accepted(records):
        role = "calibration" if record.get("role") == "aa" else "treatment"
        sig = _protocol_signature(record)
        seen[role].add((sig["inference_calls"], sig["timed_iterations"]))

    def _render(pairs: set) -> list[dict]:
        return [
            {"inference_calls": c, "timed_iterations": t, "untimed_calls":
             (c - t if isinstance(c, int) and isinstance(t, int) else None)}
            for c, t in sorted(pairs, key=lambda p: (p[0] is None, p[0] or 0,
                                                     p[1] is None, p[1] or 0))
        ]

    treatment, calibration_side = _render(seen["treatment"]), _render(seen["calibration"])
    if not treatment or not calibration_side:
        return {
            "matched": None,
            "treatment": treatment,
            "calibration": calibration_side,
            "differences": [],
            "reason": "one side has no accepted record, so there is nothing to compare",
        }
    differences = []
    for field in ("inference_calls", "timed_iterations"):
        left = {row[field] for row in treatment}
        right = {row[field] for row in calibration_side}
        if left != right:
            differences.append(
                f"{field}: treatment {sorted(map(str, left))}, calibration {sorted(map(str, right))}"
            )
    return {
        "matched": not differences,
        "treatment": treatment,
        "calibration": calibration_side,
        "differences": differences,
        "reason": None if not differences else (
            "the calibration arm did not run the treatment's protocol; the band it produces is a "
            "measurement of a neighbouring protocol, not of the treatment session's noise"
        ),
    }


def calibration(aa_pairs: Sequence[Mapping[str, Any]], *, treatment_keys: Iterable[Any] = ()) -> dict:
    """Build the band from A/A arms, or refuse to build one.

    An A/A arm is the same binary allocated to both sides. Whatever spread it shows is, by
    construction, not a build difference — it is what this box does to two identical things, and
    it is the best scale available here against which a build difference can be called real.

    "Best available" is not "same protocol", and the difference is load-bearing enough to state in
    the function that builds the band: this function checks process disjointness and nothing else.
    It does not check that the A/A arm ran the treatment's iteration budget, or ran in the same
    session, and in this artifact neither is true. :func:`protocol_delta` is the instrument that
    reports it, and §27.2.1 of ``docs/PERF.md`` is where the consequence is argued.

    Two ways this returns no band, both of which force :func:`verdict_for` to
    ``INDETERMINATE`` rather than letting a comparison grade itself:

    * no A/A pair survives admissibility — there is nothing to calibrate on;
    * an A/A pair shares a process with the treatment records (``treatment_keys``) — the
      calibration is contaminated, and a band that was measured inside the run it grades is not
      an independent scale.
    """
    treatment = {k for k in treatment_keys if k is not None}
    usable: list[Mapping[str, Any]] = []
    contaminated: list[dict] = []
    for pair in aa_pairs:
        keys = {
            side
            for row in pair.get("per_repeat", ())
            for side in (row.get("left_process"), row.get("right_process"))
            if side is not None
        }
        overlap = sorted(str(k) for k in keys & treatment)
        if overlap:
            contaminated.append({"workload": pair.get("workload"), "shared_processes": overlap})
        elif pair.get("n_pairs"):
            usable.append(pair)

    if contaminated:
        return {
            "band": None,
            "rule": "A/A half-range, floored at %.2f" % BAND_FLOOR,
            "reason": "A/A calibration shares a process with the treatment records",
            "contaminated": contaminated,
            "arms": [],
        }
    if not usable:
        return {
            "band": None,
            "rule": "A/A half-range, floored at %.2f" % BAND_FLOOR,
            "reason": "no admissible A/A pair to calibrate on",
            "contaminated": [],
            "arms": [],
        }

    arms = [
        {
            "workload": p["workload"],
            "left": p["left"],
            "right": p["right"],
            "n_pairs": p["n_pairs"],
            "ratios": p["ratios"],
            "half_range": p["half_range"],
        }
        for p in usable
    ]
    widest = max(a["half_range"] for a in arms)
    return {
        "band": max(BAND_FLOOR, widest),
        "floor": BAND_FLOOR,
        "widest_aa_half_range": widest,
        "rule": (
            "max(%.2f, widest A/A half-range); the A/A arms are same-binary allocations "
            "disjoint from every treatment process" % BAND_FLOOR
        ),
        "reason": None,
        "contaminated": [],
        "arms": arms,
    }


# --------------------------------------------------------------------------------------------
# Verdict — rule 4
# --------------------------------------------------------------------------------------------


def verdict_for(pair: Mapping[str, Any], band: "float | None", *, repeats_required: int) -> dict:
    """Grade one paired comparison.

    Ratios arrive under :data:`RATIO_CONVENTION` — ``baseline / candidate`` — so a ratio *below*
    ``1 - band`` is the candidate taking longer. ``SLOWER`` therefore requires *every* repeat below
    ``1 - band``; ``FASTER``, every repeat above ``1 + band``; ``NEUTRAL``, every repeat inside.
    Anything else — including a missing band, a short repeat count, or repeats that straddle — is
    ``INDETERMINATE``, with the reason named.

    The unanimity rule is strict on purpose, and its cost is stated rather than hidden: see
    :func:`power_at`, which reports how large an effect this rule would have missed.
    """
    ratios = list(pair.get("ratios", ()))
    if band is None:
        return {"verdict": INDETERMINATE, "reason": "no band: calibration produced none", "band": None}
    if len(ratios) < repeats_required:
        return {
            "verdict": INDETERMINATE,
            "reason": f"{len(ratios)} accepted repeat(s), {repeats_required} required",
            "band": band,
        }

    low, high = 1 - band, 1 + band
    if all(r < low for r in ratios):
        outcome = SLOWER
    elif all(r > high for r in ratios):
        outcome = FASTER
    elif all(low <= r <= high for r in ratios):
        outcome = NEUTRAL
    else:
        outcome = INDETERMINATE
    return {
        "verdict": outcome,
        "reason": None if outcome != INDETERMINATE else "repeats straddle the band",
        "band": band,
        "band_low": low,
        "band_high": high,
        "worst_repeat_ratio": min(ratios),
    }


# --------------------------------------------------------------------------------------------
# Uncertainty — rule 5
# --------------------------------------------------------------------------------------------

#: Student-t two-sided 95% critical values, indexed by degrees of freedom. Tabulated rather than
#: computed so this module needs no scipy; only the small df these runs produce are listed, and
#: :func:`log_ratio_interval` refuses rather than extrapolating past the table.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def log_ratio_interval(ratios: Sequence[float]) -> dict:
    """A 95% interval for the geometric-mean ratio, computed in log space.

    Ratios are multiplicative, so their scatter is symmetric in logs and not in the raw values.
    Returns ``None`` bounds when there are fewer than two ratios — a single observation has no
    interval, and inventing one is how an underpowered run starts sounding decisive.
    """
    values = [r for r in ratios if _finite(r) and r > 0]
    if len(values) < 2:
        return {
            "n": len(values),
            "geometric_mean": None,
            "low": None,
            "high": None,
            "log_sd": None,
            "reason": f"{len(values)} admissible ratio(s); an interval needs at least two",
        }
    logs = [math.log(v) for v in values]
    mean = sum(logs) / len(logs)
    sd = statistics.stdev(logs)
    df = len(logs) - 1
    if df not in _T95:
        return {"n": len(values), "geometric_mean": math.exp(mean), "low": None, "high": None,
                "log_sd": sd, "reason": f"no tabulated t critical value for df={df}"}
    half = _T95[df] * sd / math.sqrt(len(logs))
    return {
        "n": len(values),
        "geometric_mean": math.exp(mean),
        "low": math.exp(mean - half),
        "high": math.exp(mean + half),
        "log_sd": sd,
        "level": 0.95,
    }


def power_at(true_ratio: float, log_sd: float, band: float, repeats: int) -> float:
    """P(this protocol declares ``SLOWER``) when the true effect really is ``true_ratio``.

    The rule is unanimity: every one of ``repeats`` independent ratios must fall below
    ``1 - band``. Treating the per-repeat log-ratios as independent draws with dispersion
    ``log_sd`` about ``log(true_ratio)``, that probability is the per-repeat tail probability
    raised to ``repeats``.

    This is the number that decides whether a null result means anything. A protocol with 35%
    power against the effect under dispute cannot report its null as a non-reproduction, and this
    function exists so the summary has to say so.

    It is monotonically decreasing in ``band``: a wider band pushes the unanimity threshold further
    from ``true_ratio``. So a power figure quoted against a band that is not known to be at least as
    wide as the treatment's own noise is an **upper bound** on power, not an estimate of it, and it
    must be published with that qualifier. ``test_the_power_figure_is_an_upper_bound`` holds the
    monotonicity as a property of this function rather than as a remark about it.
    """
    if log_sd <= 0 or repeats <= 0:
        return float("nan")
    threshold = math.log(1 - band)
    z = (threshold - math.log(true_ratio)) / log_sd
    per_repeat = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    return per_repeat**repeats


# --------------------------------------------------------------------------------------------
# The window rule — rule 2
# --------------------------------------------------------------------------------------------


def window_verdict(
    verdicts_by_length: Mapping[int, str],
    *,
    target: int = REPORTED_LENGTH,
    universe: Sequence[int] = DECODE_LENGTH_UNIVERSE,
) -> dict:
    """Decide what may be said about a KV-length window around ``target``.

    A window claim is a statement about *three* lengths: the target, and the named lengths on
    either side of it that bound the window. This function will not make one unless all three
    were measured and accepted.

    ``verdicts_by_length`` maps every length that was **actually measured and accepted** to its
    verdict — including :data:`INDETERMINATE`. A length that was refused, or never run, is simply
    absent from the mapping, and its absence is what makes the claim ``INDETERMINATE``. That is
    the whole fix: in the previous revision an absent length arrived carrying the verdict "not
    slow" and could be named as an edge of a window.

    The two failure modes are kept apart on purpose, because they call for different work:

    * **absent** — the length has no admissible pair. Nobody knows anything about it. Only a new
      measurement can change that.
    * **measured but undecided** — the length was measured and its repeats straddled the band.
      Something is known about it: that this protocol cannot resolve it at this repeat count.

    Both block a window claim. Only the first one means the sweep did not cover the length.

    ``target`` at the ends of ``universe`` has only one neighbour, and the missing side is
    reported as ``None`` rather than fabricated.
    """
    if target not in universe:
        return {
            "claim": INDETERMINATE,
            "target": target,
            "reason": f"{target} is not in the pre-registered ordered universe {list(universe)}",
            "measured_lengths": sorted(verdicts_by_length),
        }

    index = list(universe).index(target)
    predecessor = universe[index - 1] if index > 0 else None
    successor = universe[index + 1] if index + 1 < len(universe) else None

    measured = sorted(verdicts_by_length)
    missing = [
        name
        for name, length in (("predecessor", predecessor), ("target", target), ("successor", successor))
        if length is not None and length not in verdicts_by_length
    ]
    detail = {
        "target": target,
        "predecessor": predecessor,
        "successor": successor,
        "measured_lengths": measured,
        "unmeasured_lengths": [n for n in universe if n not in verdicts_by_length],
        "undecided_lengths": sorted(n for n, v in verdicts_by_length.items() if v == INDETERMINATE),
        "verdicts": {int(k): v for k, v in sorted(verdicts_by_length.items())},
    }

    if missing:
        named = {"predecessor": predecessor, "target": target, "successor": successor}
        return {
            "claim": INDETERMINATE,
            "reason": (
                "a window claim needs the target and its named neighbours measured; no "
                "admissible pair exists for: "
                + ", ".join(f"{n} ({named[n]})" for n in missing)
            ),
            "missing_sides": missing,
            **detail,
        }

    undecided = [
        name
        for name, length in (("predecessor", predecessor), ("target", target), ("successor", successor))
        if length is not None and verdicts_by_length[length] == INDETERMINATE
    ]
    if undecided:
        named = {"predecessor": predecessor, "target": target, "successor": successor}
        return {
            "claim": INDETERMINATE,
            "reason": (
                "every side was measured, but this protocol did not resolve: "
                + ", ".join(f"{n} ({named[n]})" for n in undecided)
                + "; a length whose repeats straddle the band is not evidence either way"
            ),
            "undecided_sides": undecided,
            "missing_sides": [],
            **detail,
        }

    target_verdict = verdicts_by_length[target]
    if target_verdict != SLOWER:
        neighbours = [verdicts_by_length[n] for n in (predecessor, successor) if n is not None]
        return {
            "claim": "NO-WINDOW-AT-TARGET",
            "reason": (
                f"{target} is {target_verdict}, not {SLOWER}, with both neighbours measured "
                f"({neighbours}); this is a statement about {target} and its neighbours only"
            ),
            **detail,
        }

    edges = [n for n in (predecessor, successor) if n is not None and verdicts_by_length[n] == SLOWER]
    if edges:
        return {
            "claim": "WINDOW",
            "reason": f"{target} is {SLOWER} and so are the measured neighbour(s) {edges}",
            "slow_lengths": sorted([target, *edges]),
            **detail,
        }
    return {
        "claim": "SINGLE-SLOW-POINT",
        "reason": (
            f"{target} is {SLOWER}; both named neighbours were measured and are not, "
            "so the effect does not extend to them"
        ),
        "slow_lengths": [target],
        **detail,
    }


# --------------------------------------------------------------------------------------------
# Identity — rule from gate 5
# --------------------------------------------------------------------------------------------


def identity_agreement(records: Iterable[Mapping[str, Any]], arms: Mapping[str, Any]) -> dict:
    """Check that every accepted record's library digest matches its arm's declared identity.

    A timing whose binary cannot be named is not evidence about a binary. Any disagreement, any
    arm carrying more than one digest, or any accepted record naming an arm the header does not
    declare, is reported here and makes the whole summary refuse.
    """
    observed: dict[str, set] = {}
    unknown_arms: set = set()
    for record in accepted(records):
        arm = str(record.get("arm"))
        if arm not in arms:
            unknown_arms.add(arm)
            continue
        observed.setdefault(arm, set()).add(
            (record.get("ep_library_sha256"), record.get("ep_library_bytes"))
        )

    disagreements: list[dict] = []
    for arm, seen in sorted(observed.items()):
        declared = (arms[arm].get("sha256"), arms[arm].get("bytes"))
        if len(seen) != 1:
            disagreements.append({"arm": arm, "problem": "records carry more than one library identity",
                                  "observed": sorted(map(str, seen))})
        elif next(iter(seen)) != declared:
            disagreements.append({"arm": arm, "problem": "record identity differs from the declared arm",
                                  "declared": list(declared), "observed": list(next(iter(seen)))})
    if unknown_arms:
        disagreements.append({"arm": sorted(unknown_arms), "problem": "accepted record names an undeclared arm"})

    return {"agrees": not disagreements, "disagreements": disagreements,
            "arms_observed": {a: sorted(map(str, s)) for a, s in sorted(observed.items())}}


# --------------------------------------------------------------------------------------------
# The summary — gate 7
# --------------------------------------------------------------------------------------------


def summarize(
    records: Sequence[Mapping[str, Any]],
    *,
    arms: Mapping[str, Any],
    repeats_required: int,
    aa_allocations: Sequence[Mapping[str, str]] = (),
    reference_effect: "float | None" = None,
) -> dict:
    """Turn raw records into the whole published verdict. The only path to a claim.

    ``aa_allocations`` names the A/A arms as ``{"workload", "left", "right"}`` triples; each one
    is paired by :func:`paired` and handed to :func:`calibration`. ``reference_effect``, when
    given, is the previously reported ratio whose detectability :func:`power_at` reports — that is
    what turns "we saw nothing" into "we saw nothing, and here is how often we would have".
    """
    identity = identity_agreement(records, arms)
    counts = status_counts(records)

    aa_pairs = [
        paired(records, a["workload"], left=a["left"], right=a["right"]) for a in aa_allocations
    ]
    treatment_workloads = sorted(
        {
            r.get("workload")
            for r in accepted(records)
            if r.get("role") != "aa" and r.get("workload")
        }
    )
    treatment_pairs = [paired(records, w) for w in treatment_workloads]
    treatment_keys = {
        side
        for pair in treatment_pairs
        for row in pair["per_repeat"]
        for side in (row["left_process"], row["right_process"])
    }

    band_block = calibration(aa_pairs, treatment_keys=treatment_keys)
    band = band_block["band"]

    rows: list[dict] = []
    verdicts_by_length: dict[int, str] = {}
    for pair in treatment_pairs:
        graded = verdict_for(pair, band, repeats_required=repeats_required)
        interval = log_ratio_interval(pair["ratios"])
        row = {**pair, **graded, "interval": interval}
        if reference_effect and interval.get("log_sd") and band is not None:
            row["power_at_reference_effect"] = {
                "true_ratio": reference_effect,
                "power": power_at(reference_effect, interval["log_sd"], band, repeats_required),
                "note": (
                    "probability this unanimity rule would have declared SLOWER if the "
                    "reference effect were real, at this row's own observed dispersion"
                ),
            }
        rows.append(row)
        past = pair["past"]
        # Every length with an admissible pair enters the mapping, INDETERMINATE included. The
        # window rule needs to tell "we measured it and could not resolve it" apart from "we have
        # nothing", and it can only do that if the undecided ones are present rather than dropped.
        if past is not None and pair["n_pairs"]:
            verdicts_by_length[past] = graded["verdict"]

    window = window_verdict(verdicts_by_length)

    return {
        "schema": SCHEMA,
        "issue": ISSUE,
        "ratio_convention": RATIO_CONVENTION,
        "counts": counts,
        "identity": identity,
        "refuses": not identity["agrees"],
        "band": band_block,
        "aa_arms": aa_pairs,
        "rows": rows,
        "window": window,
        "accepted_decode_lengths": sorted(verdicts_by_length),
        "length_universe": list(DECODE_LENGTH_UNIVERSE),
    }


# --------------------------------------------------------------------------------------------
# Reproduction — gate 7's --check
# --------------------------------------------------------------------------------------------

#: Numeric fields compare within this relative tolerance when re-derived. Serialization rounds;
#: the check is that the summary was *computed from* these records, not that floats round-trip.
CHECK_RTOL = 1e-9


def _differences(published: Any, rederived: Any, path: str = "") -> list[str]:
    if isinstance(published, Mapping) and isinstance(rederived, Mapping):
        # A JSON round-trip turns integer keys (KV lengths) into strings, so the published copy
        # and the freshly re-derived one disagree on key *type* while agreeing on everything that
        # matters. Compare under a string normalization rather than reporting a false difference.
        pub = {str(k): v for k, v in published.items()}
        red = {str(k): v for k, v in rederived.items()}
        out: list[str] = []
        for key in sorted(set(pub) | set(red)):
            if key not in pub:
                out.append(f"{path}.{key}: absent from the published summary")
            elif key not in red:
                out.append(f"{path}.{key}: not re-derivable")
            else:
                out += _differences(pub[key], red[key], f"{path}.{key}")
        return out
    if isinstance(published, (list, tuple)) and isinstance(rederived, (list, tuple)):
        if len(published) != len(rederived):
            return [f"{path}: length {len(published)} published, {len(rederived)} re-derived"]
        out = []
        for i, (a, b) in enumerate(zip(published, rederived)):
            out += _differences(a, b, f"{path}[{i}]")
        return out
    if _finite(published) and _finite(rederived):
        if published == rederived:
            return []
        scale = max(abs(published), abs(rederived), 1.0)
        if abs(published - rederived) <= CHECK_RTOL * scale:
            return []
        return [f"{path}: published {published}, re-derived {rederived}"]
    if published != rederived:
        return [f"{path}: published {published!r}, re-derived {rederived!r}"]
    return []


def check(artifact: Mapping[str, Any]) -> dict:
    """Re-derive the published summary from the published records and diff the two.

    Returns ``{"reproduces": bool, "differences": [...]}``. It fails — by design and under test —
    when raw samples are scaled or deleted, when a witness is removed, when a length's records are
    refused, when an A/A arm is contaminated by a treatment process, or when a record's library
    digest stops matching its arm. Each of those changes what :func:`summarize` computes, so each
    of them shows up here as a difference rather than as a silent pass.
    """
    records = artifact.get("records", [])
    published = artifact.get("summary")
    if published is None:
        return {"reproduces": False, "differences": ["artifact carries no summary block"]}
    rederived = summarize(
        records,
        arms=artifact.get("arms", {}),
        repeats_required=artifact.get("environment", {}).get("repeats", 3),
        aa_allocations=artifact.get("aa_allocations", ()),
        reference_effect=artifact.get("reference_effect"),
    )
    differences = _differences(published, rederived, "summary")
    return {"reproduces": not differences, "differences": differences}
