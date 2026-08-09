#!/usr/bin/env python3
"""Frozen Phi-3.5 performance evidence: the record, its identity, and the one gate that reads it.

WHY THIS MODULE EXISTS (issue #69, and the reason it is a *second* module)
=========================================================================
`bench/real_model.py` is the harness that *takes* a real-model measurement. This module is the
part that decides whether a taken measurement may be **published as a claim**, and it is separate
for one reason: every defect this file screens for was a defect of publication, not of
measurement. A correct number described as a different number, a hardware reading labelled as a
software rasteriser, a confidence band computed from the very data it is quoted about, a decode
observation quietly dropped because a later one read better — none of those are visible to a
timer.

So the split is:

    bench/real_model.py     produces samples          "what did the machine do"
    bench/phi_evidence.py   freezes and gates them    "what may be said about it, and by whom"

WHAT `load_frozen()` DOES, EXACTLY — AND WHAT IT DOES NOT
=========================================================
This is stated first because a previous description of a loader in this project claimed a
validation the loader did not perform, and a reviewer had to find that by reading the source.

``load_frozen(path)`` does **exactly two things**: it reads the bytes at ``path`` and parses them
as JSON. That is the whole contract.

It does **not**:

* recompute or check the content digest — that is :func:`verify_frozen_identity`;
* check the schema token — that is the first clause of :func:`evidence_gate`;
* check any claim, band, verdict, scope or limit — that is the rest of :func:`evidence_gate`;
* refuse a record whose bytes were edited after it was frozen — a mutated record loads happily,
  and ``test_load_frozen_does_not_validate_a_mutated_record`` proves it *behaviourally* by
  mutating the bytes and asserting the load succeeds.

The reason it is that thin is that a loader which validates has two failure modes that look
identical from the outside — "the file is not there" and "the file is not admissible" — and the
second is a *finding* that must reach a verdict record, not an exception in a reader. Validation
lives in exactly one place (:func:`evidence_gate`) so there is exactly one place to bypass, and exactly one
place the mutation battery has to attack.

THE FROZEN CONTRACT
===================
A frozen record carries ``identity.content_sha256``: the SHA-256 of the canonical JSON encoding
of the record with that one field removed. :func:`freeze` stamps it; :func:`verify_frozen_identity`
recomputes it. Any edit to any other field — a ratio, a verdict, a device name, a limitation —
changes the digest. The digest therefore says *these bytes are the bytes that were frozen*, and it
says nothing at all about whether they were true; that is the gate's job, and conflating the two
is how a self-verifying artifact gets quoted as a verified one.

WHAT THE GATE REFUSES, AND WHY EACH ONE IS HERE
===============================================
Every condition token below is a defect that was actually made in this project's issue-69 work.
The gate is the only authority: `ci/check_phi_evidence.py` is a thin process wrapper around
:func:`evidence_gate`, and `ci/negative_control_phi_evidence.py` mutates a healthy record once per token
and demands the matching red. A condition that cannot be produced by a mutation is not a guard.

``calibration_not_disjoint``
    The band a verdict is read against must be measured on subjects the verdict is not about.
    A band derived from the verdict's own cases cannot fail, because the data that would widen
    it is the data being judged.
``band_self_derived``
    Same defect one level in: a band whose ``derived_from`` names a verdict subject is circular
    even if the *case list* looks disjoint.
``equivalence_incomplete``
    Phi-3.5 has 65 outputs. Checking output 0 checks the logits and leaves 64 KV tensors
    unexamined, which is where a decode defect would live.
``production_path_unwitnessed``
    ``VulkanExecutionProvider in session.get_providers()`` is true whenever the EP was
    *requested*. The witness must come from outside the EP — ONNX Runtime's own profile, which
    attributes executed nodes to providers — and it must show Vulkan nodes actually executing.
``headline_scope_widened``
    One model, one phase family. A headline that names a second model, or sets
    ``generalises``, is claiming something no run here measured.
``claim_limit_violated``
    No compatible-CUDA result exists, so no CUDA comparison may be stated; issue #69 stays open;
    and no decode win may be implied.
``decode_observation_dropped``
    Two independent decode observations at ``past=128`` exist and disagree. Both are carried.
    Neither supersedes the other, and the strongest decode conclusion is ``INCONCLUSIVE``.
``dispersion_promoted``
    Within-arm dispersion measured offline is a diagnostic. It may be recorded; it may never be
    the basis of a verdict.
``isolation_overclaimed``
    This harness excludes *cooperating* processes. It cannot exclude a non-cooperating one, and
    it has no mechanism that grants exclusive ownership of the device.
``vulkan_implementation_mislabelled``
    A reading taken on an RTX A1000 is a hardware-Vulkan reading. Labelling it ``lavapipe``
    (or any software rasteriser) misattributes it to the CI lane's device class.
``device_identity_incomplete`` / ``identity_digest_mismatch``
    A number with no stable device identity is a number about an unnamed machine.
``proof_ledger_absent``
    The EP enforces a compiled proof ledger at session build. It is part of what ran, so it is
    part of the record.
``private_path_disclosed``
    Committed evidence must not carry a developer's home directory.
``verdict_disagrees_with_classifier``
    The recorded verdict must be the one :func:`classify_ratio` returns for the recorded numbers.
    Classification is symmetric by construction: mirroring every ratio ``r -> 1/r`` mirrors every
    verdict, so a gate that waves through an improvement cannot quietly wave through the
    corresponding regression.

USAGE
    python bench/phi_evidence.py --measure --out bench/results/phi35_evidence_v4.json
    python bench/phi_evidence.py --gate bench/results/phi35_evidence_v4.json
    python ci/check_phi_evidence.py                 # the authoritative live gate, in CI
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

_BENCH = Path(__file__).resolve().parent
_ROOT = _BENCH.parent
for _p in (str(_BENCH), str(_ROOT), str(_ROOT / "rust" / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

#: Bumped when a *meaning* changes, never when a field is added.
SCHEMA = "phi35_frozen_evidence/1"

#: The one model this evidence is about. A headline naming anything else is out of scope.
HEADLINE_MODEL = "phi-3.5-mini-instruct-cuda-int4-rtn-block-32"

#: Phi-3.5 emits logits plus 32x2 `present.*` blocks. Every one of them is compared.
PHI35_OUTPUT_COUNT = 65

#: The only isolation this harness performs, spelled the way it actually behaves.
ISOLATION_MODE = "cooperative-harness-exclusion"

#: The two independent decode-past-128 observations that exist and disagree.
#:
#: They are named here, by value, because "carry at least two observations" is not the
#: requirement — the requirement is that *these two* are never quietly dropped. A later,
#: friendlier third observation must not be allowed to stand in for a deleted one, and a count
#: alone cannot tell those two situations apart. Neither of these supersedes the other: the
#: first is a bare point estimate with no interval, the second's 95% interval spans 1 at a
#: power of 0.346, and a 0.346-powered non-result cannot overturn anything, including itself.
REQUIRED_DECODE_OBSERVATIONS = (
    {"point_estimate": 0.859, "interval": None, "power": None},
    {"point_estimate": 0.9651,
     "interval": {"level": 0.95, "lo": 0.820, "hi": 1.136}, "power": 0.346},
)

#: Phrases that claim more isolation than any mechanism here delivers.
_OVERCLAIM_RE = re.compile(
    r"exclusive (?:gpu|device) (?:ownership|access|use)|"
    r"(?:gpu|device) (?:is )?(?:exclusively|solely) (?:owned|held|reserved)|"
    r"sole (?:owner|tenant) of the (?:gpu|device)|"
    r"no other process can use the (?:gpu|device)",
    re.IGNORECASE,
)

#: Software Vulkan implementations. A reading from one of these is not a reading about silicon.
_SOFTWARE_DRIVERS = ("lavapipe", "llvmpipe", "swiftshader", "warp", "software rasteri")

#: Home-directory shapes that must never reach a committed artifact.
_PRIVATE_PATH_RE = re.compile(
    r"(?:[A-Za-z]:[\\/]+Users[\\/]+[^\\/\"]+)|(?:/home/[^/\"]+)|(?:/Users/[^/\"]+)",
    re.IGNORECASE,
)

MATCH = "MATCH"
PASS = "PASS"
FAIL = "FAIL"
ERROR = "ERROR"

IMPROVEMENT = "IMPROVEMENT"
REGRESSION = "REGRESSION"
INDETERMINATE = "INDETERMINATE"
INCONCLUSIVE = "INCONCLUSIVE"


class FrozenArtifactMissing(RuntimeError):
    """The frozen record is not on disk. A different state from "not admissible"."""


class FrozenArtifactUnreadable(RuntimeError):
    """The bytes are there and are not JSON. An instrument error, never a verdict."""


class EvidenceInstrumentError(RuntimeError):
    """This instrument was handed something it cannot observe. R13: never a detection.

    The distinction this class exists to keep is the one the whole file is organised around.
    ``FAIL`` is a statement *about the measurement* — the record was read and found wanting.
    ``EvidenceInstrumentError`` is a statement about the *instrument's* input: a band whose
    lower edge is above its upper edge, a latency series containing a negative number, a
    "record" that is not a mapping. There is no verdict to render over those, and returning
    ``INDETERMINATE`` for them would file an instrument fault as a measurement result — the
    exact collapse that lets a broken harness read as a quiet, unremarkable null.
    """


def _require_mapping(value, what: str) -> dict:
    if not isinstance(value, dict):
        raise EvidenceInstrumentError(
            f"{what} must be a mapping; got {type(value).__name__} ({value!r:.80}). "
            "There is no verdict to render over a non-record."
        )
    return value


# --------------------------------------------------------------------------------------------
# Identity: freeze, load, verify — three functions, three jobs, no overlap
# --------------------------------------------------------------------------------------------


def canonical_bytes(record: dict) -> bytes:
    """The exact bytes the digest is taken over: the record minus ``identity.content_sha256``.

    Key order is normalised and separators are pinned so that a re-serialisation with different
    whitespace is the same artifact, and a changed *value* never is.

    Raises :class:`EvidenceInstrumentError` if *record* is not a mapping: there are no bytes to
    canonicalise, which is an instrument error rather than a digest disagreement.
    """
    shallow = dict(_require_mapping(record, "record"))
    identity = dict(shallow.get("identity") or {})
    identity.pop("content_sha256", None)
    shallow["identity"] = identity
    return json.dumps(shallow, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def content_digest(record: dict) -> str:
    """sha256 over :func:`canonical_bytes`. Raises :class:`EvidenceInstrumentError` on a non-record."""
    return hashlib.sha256(canonical_bytes(_require_mapping(record, "record"))).hexdigest()


def freeze(record: dict, path: "Path | str") -> str:
    """Stamp ``identity.content_sha256`` and write the record. Returns the digest.

    Freezing is a *write* operation and it validates nothing. A record that would fail the gate
    freezes perfectly well — which is deliberate: a failing measurement must be publishable as a
    failing measurement, or the only artifacts that ever exist are the flattering ones.

    The one thing it refuses is a non-record (:class:`EvidenceInstrumentError`): there is nothing
    to stamp an identity onto.
    """
    _require_mapping(record, "record")
    record.setdefault("identity", {})
    record["identity"].pop("content_sha256", None)
    digest = content_digest(record)
    record["identity"]["content_sha256"] = digest
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(record, indent=1, sort_keys=True, ensure_ascii=False) + "\n",
                 encoding="utf-8")
    return digest


def load_frozen(path: "Path | str") -> dict:
    """Read the bytes at *path* and parse them as JSON. That is the entire contract.

    **It performs no validation of any kind.** Not the digest, not the schema, not a single
    claim. A record whose bytes were edited after freezing loads without complaint, and
    `bench/test_phi_evidence.py` proves that by doing it rather than by asserting it about the
    source text. Digest checking is :func:`verify_frozen_identity`; admissibility is
    :func:`evidence_gate`.

    The two failure modes it *does* have are both about the file rather than its content:
    absent (:class:`FrozenArtifactMissing`) and unparseable (:class:`FrozenArtifactUnreadable`).
    """
    p = Path(path)
    if not p.is_file():
        raise FrozenArtifactMissing(f"{p} does not exist")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FrozenArtifactUnreadable(f"{p}: {exc}") from exc


def verify_frozen_identity(record: dict) -> dict:
    """Recompute the content digest and report agreement. Never raises on disagreement.

    Returns ``{"verdict": MATCH|"DIVERGENT"|"UNMEASURED", "recorded": ..., "recomputed": ...}``.
    ``UNMEASURED`` is its own state: a record with no digest was never frozen, which is not the
    same finding as a record whose digest is wrong.

    A non-record is neither: :class:`EvidenceInstrumentError`.
    """
    recorded = (_require_mapping(record, "record").get("identity") or {}).get("content_sha256")
    if not recorded:
        return {"verdict": "UNMEASURED", "recorded": None, "recomputed": None,
                "note": "no content_sha256: this record was never frozen"}
    recomputed = content_digest(record)
    return {"verdict": MATCH if recomputed == recorded else "DIVERGENT",
            "recorded": recorded, "recomputed": recomputed}


# --------------------------------------------------------------------------------------------
# Statistics: paired ratios, a band measured off disjoint subjects, and a symmetric classifier
# --------------------------------------------------------------------------------------------


def paired_ratio_series(baseline_medians: "list[float]", head_medians: "list[float]") -> dict:
    """Per-repeat ``baseline / head`` ratios. Above 1 means the head is faster.

    Paired repeat-against-repeat, because the two arms of a repeat share whatever the box was
    doing. A ratio of pooled medians cannot see one displaced repeat; this can.

    Unequal-length inputs raise :class:`EvidenceInstrumentError` rather than silently pairing the
    common prefix — ``zip`` would have dropped the tail and returned a shorter, healthier-looking
    series, which is a measurement that was never taken presented as one that was.
    """
    if len(baseline_medians) != len(head_medians):
        raise EvidenceInstrumentError(
            f"unpaired input: {len(baseline_medians)} baseline repeat(s) against "
            f"{len(head_medians)} head repeat(s). Pairing the common prefix would discard the "
            "tail and report a series shorter than the run that produced it."
        )
    ratios = [b / h for b, h in zip(baseline_medians, head_medians) if h]
    if not ratios:
        return {"n": 0}
    return {
        "n": len(ratios),
        "ratios": [round(r, 6) for r in ratios],
        "median": statistics.median(ratios),
        "min": min(ratios),
        "max": max(ratios),
    }


def calibration_band(ratio_series: "list[dict]") -> dict:
    """The width of this harness's arm-to-arm asymmetry, from identical-build arm pairs.

    The input is the *calibration* subjects' ratio series — pairs of arms that are the same
    build, so their true ratio is 1 by construction and everything observed away from 1 is the
    harness and the box. The band is the full observed span, not a standard deviation: with
    three repeats per subject a parametric interval would be an assumption dressed as a
    measurement.

    Anything other than a sequence of series mappings raises :class:`EvidenceInstrumentError`.
    """
    if isinstance(ratio_series, dict) or not isinstance(ratio_series, (list, tuple)):
        raise EvidenceInstrumentError(
            f"calibration_band takes a sequence of ratio-series mappings; got "
            f"{type(ratio_series).__name__}. One series handed in where a list was expected "
            "would pool its keys as if they were subjects."
        )
    pooled: list[float] = []
    subjects: list[str] = []
    for series in ratio_series:
        _require_mapping(series, "each calibration ratio series")
        pooled.extend(series.get("ratios") or [])
        if series.get("subject"):
            subjects.append(series["subject"])
    if not pooled:
        return {"n": 0, "lo": None, "hi": None, "derived_from": subjects,
                "source": "calibration"}
    return {
        "n": len(pooled),
        "lo": min(pooled),
        "hi": max(pooled),
        "median": statistics.median(pooled),
        "derived_from": sorted(set(subjects)),
        "source": "calibration",
        "meaning": ("observed span of same-build arm-pair ratios on subjects DISJOINT from "
                    "every verdict subject; a verdict ratio inside this span is not a reading"),
    }


def classify_ratio(series: dict, band: dict) -> dict:
    """Symmetric verdict for one subject. ``IMPROVEMENT`` / ``REGRESSION`` / ``INDETERMINATE``.

    A subject earns ``IMPROVEMENT`` only when **every** repeat's ratio sits above the calibration
    band's upper edge, and ``REGRESSION`` only when every repeat sits below its lower edge.
    Anything else is ``INDETERMINATE`` — including a large median with one repeat inside the
    band, which is the shape a single lucky repeat produces.

    Symmetry is structural, not asserted: the ``REGRESSION`` clause is the ``IMPROVEMENT``
    clause with the ratios and the band both reciprocated, so mirroring an input mirrors the
    output. `test_the_classifier_is_symmetric_under_reciprocal` is the live proof.

    An inverted band (``lo > hi``) raises :class:`EvidenceInstrumentError`: no ratio can be both
    above the upper edge and below the lower one, so every subject would come back
    ``INDETERMINATE`` and the run would read as a quiet null instead of a broken band.
    """
    ratios = list(_require_mapping(series, "series").get("ratios") or [])
    lo, hi = _require_mapping(band, "band").get("lo"), band.get("hi")
    if lo is not None and hi is not None and lo > hi:
        raise EvidenceInstrumentError(
            f"inverted calibration band lo={lo!r} > hi={hi!r}; every subject would read "
            "INDETERMINATE and a broken band would be indistinguishable from a quiet result."
        )
    out = {"n": len(ratios), "band_lo": lo, "band_hi": hi}
    if not ratios or lo is None or hi is None:
        out["verdict"] = INDETERMINATE
        out["reason"] = "no ratios, or no calibration band to read them against"
        return out
    out["point_estimate"] = statistics.median(ratios)
    out["floor"] = min(ratios)
    out["ceiling"] = max(ratios)
    if min(ratios) > hi:
        out["verdict"] = IMPROVEMENT
    elif max(ratios) < lo:
        out["verdict"] = REGRESSION
    else:
        out["verdict"] = INDETERMINATE
    out["rule"] = ("IMPROVEMENT iff every repeat ratio > band_hi; REGRESSION iff every repeat "
                   "ratio < band_lo; otherwise INDETERMINATE")
    return out


def mirror_series(series: dict) -> dict:
    """``r -> 1/r`` over a ratio series. Used by the symmetry test and the mutation battery.

    A non-positive ratio raises :class:`EvidenceInstrumentError`: it cannot be reciprocated, and
    dropping it would mirror a shorter series than the one handed in, which is how a symmetry
    proof quietly stops proving symmetry.
    """
    raw = list(_require_mapping(series, "series").get("ratios") or [])
    bad = [r for r in raw if not isinstance(r, (int, float)) or r <= 0]
    if bad:
        raise EvidenceInstrumentError(
            f"cannot reciprocate non-positive ratio(s) {bad!r}; dropping them would mirror a "
            "shorter series than the one handed in."
        )
    ratios = [1.0 / r for r in raw]
    out = dict(series)
    out["ratios"] = [round(r, 6) for r in ratios]
    if ratios:
        out["median"] = statistics.median(ratios)
        out["min"] = min(ratios)
        out["max"] = max(ratios)
    return out


def mirror_band(band: dict) -> dict:
    """``[lo, hi] -> [1/hi, 1/lo]``. An inverted band is an instrument error, not a wide one."""
    out = dict(_require_mapping(band, "band"))
    lo, hi = band.get("lo"), band.get("hi")
    if lo is not None and hi is not None and lo > hi:
        raise EvidenceInstrumentError(
            f"cannot mirror an inverted band lo={lo!r} > hi={hi!r}; reciprocating it would "
            "return a band that looks well-formed and means the opposite of the input."
        )
    if lo and hi:
        out["lo"], out["hi"] = 1.0 / hi, 1.0 / lo
    return out


def within_arm_dispersion(samples: "list[float]") -> dict:
    """Offline within-arm spread. **Diagnostic only** — the gate refuses to let it decide.

    It is worth recording because a run whose own arm wandered is a run worth re-taking. It is
    not worth deciding on, because a tight arm and a true arm are different properties and this
    number cannot tell them apart.

    A non-positive or non-finite latency raises :class:`EvidenceInstrumentError`. No wall-clock
    duration is zero or negative, so such a sample is a timer fault, and a timer fault averaged
    into a dispersion figure reads as an unusually steady arm.
    """
    xs_raw = [float(x) for x in samples]
    bad = [x for x in xs_raw if not math.isfinite(x) or x <= 0]
    if bad:
        raise EvidenceInstrumentError(
            f"non-positive or non-finite latency sample(s) {bad!r}; that is a timer fault, and "
            "folding it into a dispersion figure reads as an unusually steady arm."
        )
    xs = sorted(xs_raw)
    if len(xs) < 2:
        return {"n": len(xs), "role": "diagnostic"}
    med = statistics.median(xs)
    return {
        "n": len(xs),
        "median_ms": med,
        "min_ms": xs[0],
        "max_ms": xs[-1],
        "stdev_ms": statistics.stdev(xs),
        "rsd": statistics.stdev(xs) / med if med else 0.0,
        "role": "diagnostic",
        "note": "offline within-arm dispersion; recorded, never a basis for a verdict",
    }


# --------------------------------------------------------------------------------------------
# The gate — the single authority
# --------------------------------------------------------------------------------------------

#: Every condition the gate can report, in the order it checks them. The negative control walks
#: this tuple and demands one red per token, so a token with no reachable mutation is a finding.
GATE_CONDITIONS = (
    "schema_unknown",
    "identity_digest_mismatch",
    "device_identity_incomplete",
    "vulkan_implementation_mislabelled",
    "isolation_overclaimed",
    "calibration_not_disjoint",
    "band_self_derived",
    "equivalence_incomplete",
    "production_path_unwitnessed",
    "headline_scope_widened",
    "claim_limit_violated",
    "decode_observation_dropped",
    "dispersion_promoted",
    "proof_ledger_absent",
    "private_path_disclosed",
    "verdict_disagrees_with_classifier",
)


def _fail(condition: str, detail: str) -> dict:
    return {"verdict": FAIL, "condition": condition, "detail": detail}


def _walk_strings(node, path="$"):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _walk_strings(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk_strings(v, f"{path}[{i}]")
    elif isinstance(node, str):
        yield path, node


def evidence_gate(record: dict) -> dict:
    """The authoritative admissibility verdict for a frozen Phi-3.5 evidence record.

    Returns ``{"verdict": PASS|FAIL|ERROR, "condition": token|None, "detail": str,
    "checked": [...]}``. It never raises on a *content* problem — a malformed record is an
    ``ERROR(instrument=...)``, a false one is a ``FAIL(condition=...)``, and the two are
    different findings with different owners (R13).
    """
    if not isinstance(record, dict):
        return {"verdict": ERROR, "condition": "record_not_a_mapping",
                "detail": f"expected a JSON object, got {type(record).__name__}"}
    if record.get("schema") != SCHEMA:
        return {"verdict": FAIL, "condition": "schema_unknown",
                "detail": f"schema {record.get('schema')!r} is not {SCHEMA!r}"}

    checked: list[str] = []

    # ---- identity ---------------------------------------------------------------------------
    ident = verify_frozen_identity(record)
    checked.append("identity_digest")
    if ident["verdict"] != MATCH:
        return _fail("identity_digest_mismatch",
                     f"content digest {ident['verdict']}: recorded {ident['recorded']}, "
                     f"recomputed {ident['recomputed']}")

    env = record.get("environment") or {}
    device = env.get("device") or {}
    software = env.get("software") or {}
    checked.append("device_identity")
    missing = [k for k in ("name", "uuid", "driver_version", "driver_name", "device_type",
                           "vulkan_api_version")
               if not device.get(k)]
    missing += [k for k in ("onnxruntime_version", "ep_library_sha256", "ep_source_commit",
                            "os")
                if not software.get(k)]
    if missing:
        return _fail("device_identity_incomplete",
                     f"execution-provider/adapter identity is missing {sorted(missing)}")

    checked.append("vulkan_implementation_label")
    impl = str(device.get("implementation_type") or "")
    driver_name = str(device.get("driver_name") or "")
    dev_name = str(device.get("name") or "")
    if impl not in ("hardware", "software"):
        return _fail("vulkan_implementation_mislabelled",
                     f"implementation_type {impl!r} is neither 'hardware' nor 'software'")
    looks_software = any(s in driver_name.lower() or s in dev_name.lower()
                         for s in _SOFTWARE_DRIVERS)
    if impl == "hardware" and looks_software:
        return _fail("vulkan_implementation_mislabelled",
                     f"labelled hardware but the driver/adapter reads {driver_name!r}/{dev_name!r}")
    if impl == "software" and not looks_software:
        return _fail("vulkan_implementation_mislabelled",
                     f"labelled software but the driver/adapter reads {driver_name!r}/"
                     f"{dev_name!r}: a hardware reading must not be filed as a software one")
    if device.get("device_type") == "discrete-gpu" and impl != "hardware":
        return _fail("vulkan_implementation_mislabelled",
                     "a discrete GPU reading is a hardware-Vulkan reading")

    # ---- isolation --------------------------------------------------------------------------
    checked.append("isolation_language")
    isolation = record.get("isolation") or {}
    if isolation.get("mode") != ISOLATION_MODE:
        return _fail("isolation_overclaimed",
                     f"isolation.mode {isolation.get('mode')!r} is not {ISOLATION_MODE!r}")
    for where, text in _walk_strings(record):
        if _OVERCLAIM_RE.search(text):
            return _fail("isolation_overclaimed",
                         f"{where} claims exclusive device ownership: {text[:120]!r}")

    # ---- calibration disjointness -----------------------------------------------------------
    checked.append("calibration_disjointness")
    calibration = record.get("calibration") or {}
    cal_subjects = set(calibration.get("subjects") or [])
    verdicts = record.get("verdicts") or []
    verdict_subjects = {v.get("subject") for v in verdicts}
    if not cal_subjects:
        return _fail("calibration_not_disjoint",
                     "no calibration subjects: every band would be self-derived")
    overlap = cal_subjects & verdict_subjects
    if overlap:
        return _fail("calibration_not_disjoint",
                     f"calibration and verdict subjects overlap on {sorted(overlap)}")

    # Disjoint by *input*, not merely by label. Two cases can carry different names and feed
    # byte-identical tensors — `prefill/M1/past0` and `decode/M1/past0` do — and a band that
    # counts the same input twice under two names is narrower than the evidence behind it.
    digests = (env.get("feeds_digest_by_subject") or {})
    if digests:
        missing_digest = sorted(s for s in (cal_subjects | verdict_subjects)
                                if not digests.get(s))
        if missing_digest:
            return _fail("calibration_not_disjoint",
                         f"no recorded input digest for {missing_digest}; without it, subject "
                         f"disjointness can only be checked on labels")
        cal_by_digest: dict = {}
        for subject in sorted(cal_subjects):
            cal_by_digest.setdefault(digests[subject], []).append(subject)
        collapsed = {d: s for d, s in cal_by_digest.items() if len(s) > 1}
        if collapsed:
            return _fail("calibration_not_disjoint",
                         f"calibration subjects {sorted(sum(collapsed.values(), []))} feed "
                         f"byte-identical inputs; the band counts one measurement more than "
                         f"once")
        verdict_digests = {digests[s]: s for s in sorted(verdict_subjects)}
        shared = [(s, verdict_digests[digests[s]]) for s in sorted(cal_subjects)
                  if digests[s] in verdict_digests]
        if shared:
            return _fail("calibration_not_disjoint",
                         f"calibration subject(s) feed the same input as verdict subject(s): "
                         f"{shared}; a different label is not a different subject")

    checked.append("band_provenance")
    band = calibration.get("band") or {}
    if band.get("source") != "calibration":
        return _fail("band_self_derived",
                     f"band.source is {band.get('source')!r}, not 'calibration'")
    derived = set(band.get("derived_from") or [])
    if not derived:
        return _fail("band_self_derived", "band names no subjects it was derived from")
    if derived & verdict_subjects:
        return _fail("band_self_derived",
                     f"band is derived from verdict subjects {sorted(derived & verdict_subjects)}")
    if not derived <= cal_subjects:
        return _fail("band_self_derived",
                     f"band is derived from non-calibration subjects "
                     f"{sorted(derived - cal_subjects)}")

    # ---- equivalence: every output, every compared arm ---------------------------------------
    checked.append("equivalence_completeness")
    equivalence = record.get("equivalence") or []
    if not equivalence:
        return _fail("equivalence_incomplete", "no equivalence records")
    for case in equivalence:
        total = case.get("outputs_total")
        compared = case.get("outputs_compared")
        if total != PHI35_OUTPUT_COUNT:
            return _fail("equivalence_incomplete",
                         f"{case.get('subject')}: outputs_total {total} != {PHI35_OUTPUT_COUNT}")
        if compared != total:
            return _fail("equivalence_incomplete",
                         f"{case.get('subject')}: compared {compared} of {total} outputs")
        for arm in case.get("arms") or []:
            if arm.get("self"):
                continue
            if arm.get("verdict") != MATCH:
                return _fail("equivalence_incomplete",
                             f"{case.get('subject')}/{arm.get('arm')} is "
                             f"{arm.get('verdict')}, not {MATCH}")
        independent = [a for a in (case.get("arms") or []) if not a.get("self")]
        if not independent:
            return _fail("equivalence_incomplete",
                         f"{case.get('subject')}: every arm is the reference checked "
                         f"against itself")

    # ---- production-path witness ------------------------------------------------------------
    checked.append("production_path_witness")
    for case in equivalence:
        witness = case.get("production_witness") or {}
        if witness.get("source") != "onnxruntime-profile":
            return _fail("production_path_unwitnessed",
                         f"{case.get('subject')}: witness source "
                         f"{witness.get('source')!r} is not ONNX Runtime's own profile")
        if not witness.get("vulkan_node_executions"):
            return _fail("production_path_unwitnessed",
                         f"{case.get('subject')}: ORT's profile attributes no executed node "
                         f"to the Vulkan EP")
        if witness.get("provider_requested_only"):
            return _fail("production_path_unwitnessed",
                         f"{case.get('subject')}: the EP was requested and executed nothing")

    # ---- headline scope ---------------------------------------------------------------------
    checked.append("headline_scope")
    headline = record.get("headline") or {}
    if list(headline.get("models") or []) != [HEADLINE_MODEL]:
        return _fail("headline_scope_widened",
                     f"headline names {headline.get('models')!r}; this evidence is about "
                     f"{HEADLINE_MODEL!r} and nothing else")
    if headline.get("generalises"):
        return _fail("headline_scope_widened",
                     "headline.generalises is set: one model on one box does not generalise")

    # ---- claim limits -----------------------------------------------------------------------
    checked.append("claim_limits")
    limits = record.get("claim_limits") or {}
    if limits.get("cuda_comparison") != "NONE":
        return _fail("claim_limit_violated",
                     f"cuda_comparison is {limits.get('cuda_comparison')!r}: no compatible-CUDA "
                     f"result exists, so no CUDA comparison may be stated")
    if limits.get("closes_issue_69") is not False:
        return _fail("claim_limit_violated",
                     "closes_issue_69 must be false: #69 remains open for the CUDA comparison")
    if limits.get("decode_conclusion") != INCONCLUSIVE:
        return _fail("claim_limit_violated",
                     f"decode_conclusion is {limits.get('decode_conclusion')!r}; the strongest "
                     f"decode conclusion this evidence supports is {INCONCLUSIVE}")

    # ---- both decode observations, neither superseding the other ----------------------------
    checked.append("decode_observations")
    observations = record.get("decode_observations") or []
    independent = [o for o in observations if o.get("independent")]
    if len(independent) < 2:
        return _fail("decode_observation_dropped",
                     f"{len(independent)} independent decode observation(s) carried; two "
                     f"disagreeing observations exist and both must be preserved")
    estimates = {round(float(o["point_estimate"]), 4) for o in independent
                 if o.get("point_estimate") is not None}
    if len(estimates) < 2:
        return _fail("decode_observation_dropped",
                     f"the independent decode observations collapse to {sorted(estimates)}; "
                     f"they disagree and the record must show that they disagree")
    for required in REQUIRED_DECODE_OBSERVATIONS:
        want = round(float(required["point_estimate"]), 4)
        match = next((o for o in independent
                      if o.get("point_estimate") is not None
                      and round(float(o["point_estimate"]), 4) == want), None)
        if match is None:
            return _fail("decode_observation_dropped",
                         f"the independent decode observation at {want}x is not carried; a "
                         f"count of observations cannot stand in for it, because a third "
                         f"observation is not a replacement for a deleted one")
        for field in ("interval", "power"):
            if match.get(field) != required[field]:
                return _fail("decode_observation_dropped",
                             f"the {want}x decode observation is carried with {field}="
                             f"{match.get(field)!r}; it was recorded as {required[field]!r} "
                             f"and its uncertainty is the reason it cannot be treated as a "
                             f"result")
    for obs in observations:
        if obs.get("supersedes") or obs.get("superseded_by"):
            return _fail("decode_observation_dropped",
                         f"observation {obs.get('id')!r} claims a supersession; these "
                         f"observations disagree and neither resolves the other")
        if obs.get("verdict") == IMPROVEMENT:
            return _fail("decode_observation_dropped",
                         f"observation {obs.get('id')!r} reads as a decode win; the decode "
                         f"conclusion is {INCONCLUSIVE}")

    # ---- dispersion stays diagnostic --------------------------------------------------------
    checked.append("dispersion_role")
    for v in verdicts:
        basis = v.get("basis")
        if basis != "paired-ratio-vs-calibration-band":
            return _fail("dispersion_promoted",
                         f"{v.get('subject')}: verdict basis {basis!r} is not the paired ratio")
        disp = v.get("dispersion") or {}
        if disp and disp.get("role") != "diagnostic":
            return _fail("dispersion_promoted",
                         f"{v.get('subject')}: dispersion.role is {disp.get('role')!r}")

    # ---- the compiled proof ledger is part of what ran --------------------------------------
    checked.append("proof_ledger")
    ledger = record.get("proof_ledger") or {}
    if not ledger.get("file_sha256") or not ledger.get("entries_total"):
        return _fail("proof_ledger_absent",
                     "no proof-ledger identity: the EP enforces one at session build, so it is "
                     "part of the semantic delta and part of this record")
    if ledger.get("entries_live") is None:
        return _fail("proof_ledger_absent", "proof ledger records no live-entry count")
    for case in equivalence:
        enforcement = case.get("runtime_enforcement") or {}
        if not enforcement.get("present"):
            return _fail("proof_ledger_absent",
                         f"{case.get('subject')}: the run carries no claim log, so the ledger "
                         f"is asserted rather than observed being enforced")
        if not enforcement.get("claimed"):
            return _fail("proof_ledger_absent",
                         f"{case.get('subject')}: the EP's own claim log records no claimed "
                         f"node")
        if enforcement.get("claimed_without_ledger_hit"):
            return _fail("proof_ledger_absent",
                         f"{case.get('subject')}: "
                         f"{enforcement['claimed_without_ledger_hit']} claimed node(s) had no "
                         f"ledger entry; this timing ran through an unproven kernel")

    # ---- no private paths -------------------------------------------------------------------
    checked.append("private_paths")
    for where, text in _walk_strings(record):
        m = _PRIVATE_PATH_RE.search(text)
        if m:
            return _fail("private_path_disclosed",
                         f"{where} carries a home-directory path: {m.group(0)!r}")

    # ---- every verdict is the classifier's own answer ---------------------------------------
    checked.append("verdict_agreement")
    for v in verdicts:
        recomputed = classify_ratio(v.get("series") or {}, band)
        if recomputed["verdict"] != v.get("verdict"):
            return _fail("verdict_disagrees_with_classifier",
                         f"{v.get('subject')}: recorded {v.get('verdict')!r}, classifier says "
                         f"{recomputed['verdict']!r}")

    return {"verdict": PASS, "condition": None,
            "detail": f"{len(verdicts)} verdict subject(s), {len(equivalence)} equivalence "
                      f"case(s), band from {sorted(derived)}",
            "checked": checked}


# --------------------------------------------------------------------------------------------
# Measurement — the driver that produces a record worth freezing
# --------------------------------------------------------------------------------------------

#: Verdict subjects: what the head is being compared against the pre-#72 baseline on.
VERDICT_CASES = (
    ("prefill", 32, 0),
    ("prefill", 64, 0),
    ("prefill", 128, 0),
    ("decode", 1, 128),
)

#: Calibration subjects. Deliberately DISJOINT from VERDICT_CASES — and disjoint by *input*,
#: not merely by label: an earlier draft used `("decode", 1, 0)`, which builds byte-identical
#: feeds to `("prefill", 1, 0)` and so contributed a duplicate to the band under a second name.
#: `past=16` is a genuinely different input and is still nowhere near the decode verdict point.
CALIBRATION_CASES = (
    ("prefill", 1, 0),
    ("prefill", 8, 0),
    ("decode", 1, 16),
)

ARM_HEAD = "vulkan_head"
ARM_BASELINE = "vulkan_pre72"
ARM_HEAD_B = "vulkan_head_b"
ARM_CPU = "cpu"


def subject_label(phase: str, m: int, past: int) -> str:
    """The one place a subject name is spelled. Every claim and every band edge is keyed by it.

    The shape is enforced rather than trusted: a subject label is the join between the artifact,
    the gate's disjointness screen and the published table, so a malformed one does not produce a
    wrong verdict — it produces a verdict about a subject that does not exist.
    :class:`EvidenceInstrumentError` for an unknown phase, a non-positive ``M`` or a negative
    ``past``.
    """
    if phase not in ("prefill", "decode"):
        raise EvidenceInstrumentError(
            f"unknown phase {phase!r}; a subject label is the key the gate matches on, so an "
            "unrecognised phase silently invents a subject nothing else in the record has."
        )
    if not isinstance(m, int) or isinstance(m, bool) or m < 1:
        raise EvidenceInstrumentError(f"M must be a positive integer; got {m!r}")
    if not isinstance(past, int) or isinstance(past, bool) or past < 0:
        raise EvidenceInstrumentError(f"past must be a non-negative integer; got {past!r}")
    return f"{HEADLINE_MODEL}/{phase}/M{m}/past{past}"


def _redact(text: str) -> str:
    """Replace a home-directory prefix with a stable token. Committed evidence names no user."""
    return _PRIVATE_PATH_RE.sub("<home>", str(text))


def _worker_main(argv: "list[str]") -> int:
    """One process, one arm, one case. Prints a JSON record on stdout.

    A process per point is not fastidiousness: the row-tile and GQA specialisations are chosen at
    session-build time, and the two builds under comparison are two different `.dll` files, so
    they cannot coexist in one process at all.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--past", type=int, required=True)
    ap.add_argument("--providers", required=True)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--capture-outputs", default="")
    ap.add_argument("--claim-log", default="")
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args(argv)

    import numpy as np
    import onnxruntime as ort
    import real_model as rm

    lib = os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB")
    providers = args.providers.split(",")
    if "VulkanExecutionProvider" in providers:
        if not lib:
            raise SystemExit("ONNXRUNTIME_VULKAN_EP_LIB is unset for a Vulkan arm")
        try:
            ort.register_execution_provider_library("VulkanExecutionProvider", lib)
        except Exception as exc:  # already registered is not a failure
            if "already" not in str(exc).lower():
                raise

    info = rm.resolve_model(rm.PHI35)
    case = rm.Case(rm.PHI35.key, args.phase, args.m, args.past,
                   tokens=args.m if args.phase == "prefill" else 1)
    feeds = rm.phi35_feeds(case, np)

    so = ort.SessionOptions()
    so.log_severity_level = 4
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    if args.profile:
        so.enable_profiling = True
        so.profile_file_prefix = str(scratch / "prof")

    t0 = time.perf_counter()
    sess = ort.InferenceSession(info["path"], so, providers=providers)
    build_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    outputs = sess.run(None, feeds)
    first_ms = (time.perf_counter() - t0) * 1e3

    for _ in range(args.warmup):
        sess.run(None, feeds)
    samples = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        sess.run(None, feeds)
        samples.append((time.perf_counter() - t0) * 1e3)

    witness = {"source": "onnxruntime-profile"}
    if args.profile:
        pf = sess.end_profiling()
        events = json.loads(Path(pf).read_text(encoding="utf-8"))
        counts: dict[str, int] = {}
        for ev in events:
            if ev.get("cat") == "Node":
                prov = str((ev.get("args") or {}).get("provider") or "unattributed")
                counts[prov] = counts.get(prov, 0) + 1
        witness.update({
            "provider_node_executions": counts,
            "vulkan_node_executions": counts.get("VulkanExecutionProvider", 0),
            "cpu_node_executions": counts.get("CPUExecutionProvider", 0),
            "provider_requested_only": ("VulkanExecutionProvider" in providers
                                        and counts.get("VulkanExecutionProvider", 0) == 0),
            "profile_events": len(events),
            "note": ("ORT's own profile attributes each executed node to a provider; it is "
                     "written by the runtime, not by the EP under test"),
        })
        Path(pf).unlink(missing_ok=True)

    rec = {
        "phase": args.phase, "m": args.m, "past": args.past,
        "providers_requested": providers,
        "providers_reported": list(sess.get_providers()),
        "session_build_s": build_s,
        "first_run_ms": first_ms,
        "samples_ms": samples,
        "output_count": len(outputs),
        "output_names": [o.name for o in sess.get_outputs()],
        "feeds_digest": rm.feeds_digest(feeds),
        "witness": witness,
        "model": {"sha256": info["sha256"], "bytes": info["bytes"],
                  "weights": [{"location": f["location"], "bytes": f["bytes"],
                               "sha256": f["sha256"]}
                              for f in info["external_data"]["files"]]},
    }
    if args.capture_outputs:
        np.savez(args.capture_outputs,
                 **{f"o{i}": np.asarray(a) for i, a in enumerate(outputs)})
        rec["outputs_captured"] = _redact(args.capture_outputs)
    if args.claim_log:
        rec["claim_log"] = _summarise_claim_log(Path(args.claim_log))
    print("@@RECORD@@" + json.dumps(rec))
    return 0


def _summarise_claim_log(path: Path) -> dict:
    """Fold the EP's own per-node claim log into counts. Runtime enforcement, not prose.

    ``ONNXRUNTIME_EP_VULKAN_CLAIM_LOG`` makes the compiled artifact write one line per claim
    decision, each carrying the node's §8.9 proof key and whether the baked-in proof ledger held
    an entry for it. That is the ledger *being enforced* during the very run being timed, which
    is why it belongs in this record rather than beside it: a claimed node with
    ``ledger_hit: false`` would mean the timing above was taken through an unproven kernel.
    """
    if not path.is_file():
        return {"present": False,
                "note": "the EP wrote no claim log; the run's enforcement is unobserved"}
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    claimed = [r for r in rows if r.get("claimed")]
    declines: dict[str, int] = {}
    for r in rows:
        if not r.get("claimed"):
            declines[str(r.get("code"))] = declines.get(str(r.get("code")), 0) + 1
    ops: dict[str, int] = {}
    for r in claimed:
        ops[str(r.get("op"))] = ops.get(str(r.get("op")), 0) + 1
    return {
        "present": True,
        "source": "ONNXRUNTIME_EP_VULKAN_CLAIM_LOG, written by the EP under test",
        "decisions": len(rows),
        "claimed": len(claimed),
        "claimed_by_op": dict(sorted(ops.items())),
        "claimed_with_ledger_hit": sum(1 for r in claimed if r.get("ledger_hit")),
        "claimed_without_ledger_hit": sum(1 for r in claimed if not r.get("ledger_hit")),
        "decline_codes": dict(sorted(declines.items())),
        "distinct_proof_keys_claimed": len(
            {r.get("proof_key") for r in claimed if r.get("proof_key")}),
    }


def _run_worker(phase: str, m: int, past: int, providers: str, lib: "str | None",
                scratch: Path, *, iters: int, warmup: int, capture: "Path | None" = None,
                profile: bool = False, claim_log: "Path | None" = None) -> dict:
    env = dict(os.environ)
    if lib:
        env["ONNXRUNTIME_VULKAN_EP_LIB"] = lib
    else:
        env.pop("ONNXRUNTIME_VULKAN_EP_LIB", None)
    if claim_log:
        Path(claim_log).unlink(missing_ok=True)
        env["ONNXRUNTIME_EP_VULKAN_CLAIM_LOG"] = str(claim_log)
    else:
        env.pop("ONNXRUNTIME_EP_VULKAN_CLAIM_LOG", None)
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker",
           "--phase", phase, "--m", str(m), "--past", str(past),
           "--providers", providers, "--iters", str(iters), "--warmup", str(warmup),
           "--scratch", str(scratch)]
    if capture:
        cmd += ["--capture-outputs", str(capture)]
    if claim_log:
        cmd += ["--claim-log", str(claim_log)]
    if profile:
        cmd += ["--profile"]
    proc = subprocess.run(cmd, capture_output=True, env=env, text=True, errors="replace")
    marker = "@@RECORD@@"
    for line in (proc.stdout or "").splitlines():
        if line.startswith(marker):
            return json.loads(line[len(marker):])
    raise RuntimeError(
        f"worker {phase}/M{m}/past{past} on {providers} produced no record "
        f"(rc={proc.returncode}): {(proc.stderr or '')[-800:]}"
    )


def main(argv: "list[str] | None" = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--worker":
        return _worker_main(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gate", metavar="ARTIFACT",
                    help="gate a frozen record and print the verdict")
    ap.add_argument("--measure", action="store_true", help="take a fresh measurement")
    ap.add_argument("--out", default=str(_BENCH / "results" / "phi35_evidence_v4.json"))
    ap.add_argument("--head-lib", help="the .dll built from the head under test")
    ap.add_argument("--baseline-lib", help="the .dll built from the pre-#72 baseline")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--scratch", default=str(_ROOT / "_niobe_scratch"))
    args = ap.parse_args(argv)

    if args.gate:
        record = load_frozen(args.gate)
        verdict = evidence_gate(record)
        print(json.dumps(verdict, indent=1))
        return 0 if verdict["verdict"] == PASS else 1
    if not args.measure:
        ap.print_help()
        return 2
    return _measure(args)


def _device_identity(index: int = 0) -> dict:
    """Adapter, driver and implementation class — from `bench/devices.py`, not from a guess.

    ``implementation_type`` is decided from the driver's own name and the device type rather
    than from what the run *expected* to be running on. That is the whole point: a CI lane on
    lavapipe and a workstation on an RTX A1000 both answer "Vulkan", and only one of them is a
    statement about silicon.
    """
    import devices as device_mod

    facts, _ = device_mod.probe()
    if not facts:
        return {"error": "no Vulkan device enumerated"}
    dev = facts[index] if index < len(facts) else facts[0]
    name = dev.name or ""
    driver_name = getattr(dev, "driver_name", None) or ""
    software = any(s in f"{name} {driver_name}".lower() for s in _SOFTWARE_DRIVERS)
    return {
        "index": index,
        "name": name,
        "uuid": dev.uuid,
        "luid": dev.luid,
        "pci": dev.pci,
        "device_type": (dev.device_type or "").replace("VK_PHYSICAL_DEVICE_TYPE_", "")
                       .replace("PHYSICAL_DEVICE_TYPE_", "").replace("_", "-")
                       .lower().removeprefix("vk-") or None,
        "driver_name": driver_name or None,
        "driver_version": getattr(dev, "driver_version", None),
        "vulkan_api_version": getattr(dev, "api_version", None),
        "implementation_type": "software" if software else "hardware",
        "second_device": [f.name for f in facts[1:]] or None,
        "implementation_note": (
            "classified from the Vulkan driver name and device type reported by the ICD. "
            "A hardware Vulkan reading and a software-rasteriser (lavapipe/llvmpipe) reading "
            "are different device classes and are never filed as one another."
        ),
    }


def _sha256_file(path: "Path | str") -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=str(_ROOT), capture_output=True,
                          text=True).stdout.strip()


def _compare_all_outputs(cand_npz: Path, ref_npz: Path, names: "list[str]") -> dict:
    """Every output tensor, not output 0. Logits get the logit gate; `present.*` get the KV gate.

    Phi-3.5's output 0 is the logits and outputs 1..64 are the KV blocks a decode step is
    entirely about. A gate that reads output 0 alone is blind to exactly the tensors the decode
    question depends on.
    """
    import numpy as np
    import real_model as rm

    cand = np.load(cand_npz)
    ref = np.load(ref_npz)
    per_output = []
    gates = {}
    worst = MATCH
    for i, name in enumerate(names):
        key = f"o{i}"
        c, r = cand[key], ref[key]
        kind = "logits" if i == 0 else "kv_block"
        res = (rm.classify_logits(c, r, np) if i == 0
               else rm.classify_activation(c, r, np))
        if res.get("verdict") != MATCH:
            worst = res.get("verdict")
        gates.setdefault(kind, res.get("gate"))
        row = {"index": i, "name": name, "kind": kind, "verdict": res.get("verdict"),
               "max_abs": res.get("max_abs")}
        if i == 0:
            row["argmax_candidate"] = res.get("argmax_candidate")
            row["argmax_reference"] = res.get("argmax_reference")
            row["max_prob_delta"] = res.get("max_prob_delta")
        else:
            row["elements_gross"] = res.get("elements_gross")
            row["worst_floor_multiple"] = res.get("worst_floor_multiple")
        per_output.append(row)
    return {"verdict": worst, "outputs_compared": len(per_output),
            "gates": gates, "per_output": per_output}


def _measure(args) -> int:
    """Take the whole matrix, freeze it, and print the gate's verdict on what was frozen.

    Ordering is alternated per repeat and every point is its own process. Nothing here decides
    anything: it records, freezes, and then submits the frozen bytes to the same gate CI runs.
    """
    head_lib = args.head_lib and str(Path(args.head_lib).resolve())
    base_lib = args.baseline_lib and str(Path(args.baseline_lib).resolve())
    if not head_lib or not base_lib:
        raise SystemExit("--head-lib and --baseline-lib are both required for --measure")
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)

    arms = {
        ARM_HEAD: (head_lib, "VulkanExecutionProvider,CPUExecutionProvider"),
        ARM_HEAD_B: (head_lib, "VulkanExecutionProvider,CPUExecutionProvider"),
        ARM_BASELINE: (base_lib, "VulkanExecutionProvider,CPUExecutionProvider"),
        ARM_CPU: (None, "CPUExecutionProvider"),
    }
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    runs: dict[tuple, list[dict]] = {}

    def take(arm: str, phase: str, m: int, past: int, repeat: int, **kw) -> dict:
        lib, providers = arms[arm]
        rec = _run_worker(phase, m, past, providers, lib, scratch,
                          iters=args.iters, warmup=args.warmup, **kw)
        rec["arm"] = arm
        rec["repeat"] = repeat
        runs.setdefault((phase, m, past), []).append(rec)
        print(f"  {arm:14s} {phase}/M{m}/past{past} r{repeat}: "
              f"median {statistics.median(rec['samples_ms']):.2f} ms", flush=True)
        return rec

    # ---- verdict subjects: head against the pre-#72 baseline, paired per repeat -------------
    print("verdict subjects (head vs pre-#72 baseline)", flush=True)
    for repeat in range(args.repeats):
        for phase, m, past in VERDICT_CASES:
            order = [ARM_HEAD, ARM_BASELINE] if repeat % 2 == 0 else [ARM_BASELINE, ARM_HEAD]
            for arm in order:
                take(arm, phase, m, past, repeat)

    # ---- calibration subjects: two arms of the SAME build, on disjoint cases ----------------
    print("calibration subjects (same build, both arms — disjoint from every verdict subject)",
          flush=True)
    for repeat in range(args.repeats):
        for phase, m, past in CALIBRATION_CASES:
            order = [ARM_HEAD, ARM_HEAD_B] if repeat % 2 == 0 else [ARM_HEAD_B, ARM_HEAD]
            for arm in order:
                take(arm, phase, m, past, repeat)

    # ---- equivalence + production-path witness: every output, against the CPU reference -----
    print("equivalence (all 65 outputs) and production-path witness", flush=True)
    equivalence = []
    for phase, m, past in VERDICT_CASES:
        subject = subject_label(phase, m, past)
        ref_npz = scratch / f"ref_{phase}_{m}_{past}.npz"
        ref = _run_worker(phase, m, past, "CPUExecutionProvider", None, scratch,
                          iters=1, warmup=0, capture=ref_npz, profile=True)
        arms_rec = [{"arm": ARM_CPU, "self": True, "verdict": MATCH,
                     "note": "the reference arm compared against itself by construction; "
                             "recorded so the table has no hole, and it is not evidence"}]
        witness = None
        enforcement = None
        for arm, lib in ((ARM_HEAD, head_lib), (ARM_BASELINE, base_lib)):
            cand_npz = scratch / f"{arm}_{phase}_{m}_{past}.npz"
            claim_log = scratch / f"claims_{arm}_{phase}_{m}_{past}.jsonl"
            rec = _run_worker(phase, m, past, "VulkanExecutionProvider,CPUExecutionProvider",
                              lib, scratch, iters=1, warmup=0, capture=cand_npz, profile=True,
                              claim_log=claim_log)
            cmp_ = _compare_all_outputs(cand_npz, ref_npz, rec["output_names"])
            arms_rec.append({"arm": arm, "self": False, "verdict": cmp_["verdict"],
                             "outputs_compared": cmp_["outputs_compared"],
                             "gates": cmp_["gates"],
                             "per_output": cmp_["per_output"]})
            if arm == ARM_HEAD:
                witness = rec["witness"]
                enforcement = rec.get("claim_log")
            cand_npz.unlink(missing_ok=True)
            claim_log.unlink(missing_ok=True)
            print(f"  {arm:14s} {subject}: {cmp_['verdict']} over "
                  f"{cmp_['outputs_compared']} outputs", flush=True)
        ref_npz.unlink(missing_ok=True)
        equivalence.append({
            "subject": subject,
            "outputs_total": ref["output_count"],
            "outputs_compared": ref["output_count"],
            "arms": arms_rec,
            "independent_comparisons": 2,
            "reference_self_checks": 1,
            "production_witness": witness,
            "runtime_enforcement": enforcement,
        })

    # ---- assemble -----------------------------------------------------------------------
    def medians(arm: str, key: tuple) -> "list[float]":
        rows = [r for r in runs[key] if r["arm"] == arm]
        rows.sort(key=lambda r: r["repeat"])
        return [statistics.median(r["samples_ms"]) for r in rows]

    calibration_series = []
    for phase, m, past in CALIBRATION_CASES:
        key = (phase, m, past)
        series = paired_ratio_series(medians(ARM_HEAD_B, key), medians(ARM_HEAD, key))
        series["subject"] = subject_label(phase, m, past)
        series["arms"] = [ARM_HEAD_B, ARM_HEAD]
        series["meaning"] = ("two arms of the IDENTICAL build; the true ratio is 1 by "
                             "construction and everything away from 1 is the harness and the box")
        calibration_series.append(series)
    band = calibration_band(calibration_series)

    verdicts = []
    for phase, m, past in VERDICT_CASES:
        key = (phase, m, past)
        subject = subject_label(phase, m, past)
        series = paired_ratio_series(medians(ARM_BASELINE, key), medians(ARM_HEAD, key))
        series["subject"] = subject
        series["orientation"] = "pre-#72 baseline median / head median; above 1 means the head " \
                                "is faster"
        cls = classify_ratio(series, band)
        pooled = [s for r in runs[key] if r["arm"] == ARM_HEAD for s in r["samples_ms"]]
        verdicts.append({
            "subject": subject,
            "phase": phase, "m": m, "past": past,
            "series": series,
            "verdict": cls["verdict"],
            "point_estimate": cls.get("point_estimate"),
            "floor": cls.get("floor"),
            "basis": "paired-ratio-vs-calibration-band",
            "dispersion": within_arm_dispersion(pooled),
            "head_median_ms": statistics.median(medians(ARM_HEAD, key)),
            "baseline_median_ms": statistics.median(medians(ARM_BASELINE, key)),
        })

    decode_subject = subject_label("decode", 1, 128)
    fresh_decode = next((v for v in verdicts if v["subject"] == decode_subject), None)
    decode_observations = [
        {
            "id": "prior-point-estimate",
            "independent": True,
            "subject": decode_subject,
            "point_estimate": 0.859,
            "interval": None,
            "power": None,
            "verdict": INCONCLUSIVE,
            "provenance": (
                "an independent decode-p128 observation recorded against issue #69 before this "
                "branch existed. This branch does not hold its raw samples and did not "
                "reproduce it; it is carried because it exists and disagrees, not because it "
                "is confirmed here."
            ),
            "raw_samples_held_here": False,
        },
        {
            "id": "prior-interval-estimate",
            "independent": True,
            "subject": decode_subject,
            "point_estimate": 0.9651,
            "interval": {"level": 0.95, "lo": 0.820, "hi": 1.136},
            "power": 0.346,
            "verdict": INCONCLUSIVE,
            "provenance": (
                "a second, independently taken decode-p128 observation recorded against issue "
                "#69, with its own interval and power. Its interval spans 1 and its power is "
                "0.346, so it cannot resolve the first observation and the first cannot resolve "
                "it. Neither supersedes the other."
            ),
            "raw_samples_held_here": False,
        },
    ]
    if fresh_decode:
        decode_observations.append({
            "id": "this-branch",
            "independent": True,
            "subject": decode_subject,
            "point_estimate": fresh_decode["series"].get("median"),
            "interval": {"level": None,
                         "lo": fresh_decode["series"].get("min"),
                         "hi": fresh_decode["series"].get("max"),
                         "kind": "observed span of per-repeat paired ratios, not a "
                                 "distributional interval"},
            "power": None,
            "verdict": INCONCLUSIVE if fresh_decode["verdict"] != REGRESSION
                       else fresh_decode["verdict"],
            "provenance": (
                f"measured on this branch, {args.repeats} paired repeats; raw samples are in "
                f"raw.runs of this record"
            ),
            "raw_samples_held_here": True,
        })
    decode_observations_note = (
        "Three observations of the same decode point that do not agree. No arbitration is "
        "attempted and none is available: the strongest supportable decode conclusion is "
        f"{INCONCLUSIVE}, and nothing here may be read as a decode win."
    )

    ledger_path = _ROOT / "evidence" / "proof_ledger.jsonl"
    ledger_lines = ledger_path.read_text(encoding="utf-8").splitlines() if \
        ledger_path.is_file() else []
    ledger_rows = [json.loads(x) for x in ledger_lines if x.strip().startswith("{")]
    ledger_header = next((r for r in ledger_rows if r.get("__ledger__")), {})
    ledger_entries = [r for r in ledger_rows if not r.get("__ledger__")]

    sample = next(iter(runs.values()))[0]
    record = {
        "schema": SCHEMA,
        "purpose": (
            "Independently regenerated Phi-3.5 real-model performance evidence for issue #69: "
            "the current head against the defensible pre-#72 Vulkan baseline, on one box, on "
            "one model. It answers a narrow prefill question and it does not answer #69."
        ),
        "identity": {
            "taken_started": started,
            "taken_finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "repeats": args.repeats,
            "iters_per_repeat": args.iters,
            "warmups_discarded_per_session": args.warmup,
            "method": (
                f"{args.repeats} repeats x {args.iters} timed iterations, {args.warmup} warmups "
                f"discarded, one fresh process and one fresh session per (arm, case, repeat), "
                f"arm order alternated per repeat. The published per-repeat number is the "
                f"median of that repeat's timed iterations; the published ratio is the median "
                f"of the per-repeat paired ratios. Session build time and first-run time are "
                f"recorded separately and never folded into either."
            ),
        },
        "headline": {
            "models": [HEADLINE_MODEL],
            "generalises": False,
            "scope": (
                "one model, one box, one driver, prefill and one decode point. The comparison "
                "is head vs the pre-#72 Vulkan baseline and nothing else."
            ),
        },
        "environment": {
            "device": _device_identity(0),
            "software": {
                "onnxruntime_version": __import__("onnxruntime").__version__,
                "python_version": sys.version.split()[0],
                "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
                "ep_library_sha256": _sha256_file(head_lib),
                "baseline_library_sha256": _sha256_file(base_lib),
                "ep_source_commit": _git("rev-parse", "HEAD"),
                "baseline_source_commit": "c96e7d94ff706d26ee6a1bd9bb084c0ade426820",
                "baseline_meaning": (
                    "the commit immediately before PR #72 (portable subgroup-sized GQA "
                    "workgroups) landed on main; the last tree in which the Vulkan EP still "
                    "dispatched GQA one invocation per workgroup"
                ),
                "binary_reproducibility": (
                    "this repository's Windows .dll is not byte-reproducible across forced "
                    "rebuilds, so what is claimed across the two arms is SOURCE identity "
                    "(the two commits above), never binary identity"
                ),
            },
            "model": sample["model"],
            "feeds_digest_by_subject": {
                subject_label(p, m, past): runs[(p, m, past)][0]["feeds_digest"]
                for p, m, past in VERDICT_CASES + CALIBRATION_CASES
            },
        },
        "isolation": {
            "mode": ISOLATION_MODE,
            "what_it_does": (
                "the harness serialises its own points and asks cooperating processes in this "
                "repository's harness family to stand down while a point is timed"
            ),
            "what_it_cannot_do": (
                "it cannot exclude a process that does not cooperate, and it acquires no "
                "device-level lock. Nothing here reserves the GPU, and no number below may be "
                "read as having been taken on an idle machine. Wall clock is "
                "STEADY_UNCERTIFIED by default (docs/PERF.md 20)."
            ),
            "clocks": "stock power plan, no affinity mask, no clock lock",
        },
        "calibration": {
            "subjects": [subject_label(p, m, past) for p, m, past in CALIBRATION_CASES],
            "why_disjoint": (
                "the band is measured on cases no verdict is read on. A band derived from the "
                "verdict's own repeats cannot widen on the data that would contradict the "
                "verdict, which makes it a restatement of the verdict rather than a check on it."
            ),
            "arms": [ARM_HEAD, ARM_HEAD_B],
            "series": calibration_series,
            "band": band,
        },
        "verdicts": verdicts,
        "equivalence": equivalence,
        "decode_observations": decode_observations,
        "decode_observations_note": decode_observations_note,
        "claim_limits": {
            "cuda_comparison": "NONE",
            "closes_issue_69": False,
            "decode_conclusion": INCONCLUSIVE,
        },
        "proof_ledger": {
            "path": "evidence/proof_ledger.jsonl",
            "file_sha256": _sha256_file(ledger_path) if ledger_path.is_file() else None,
            "self_declared_entry_count": ledger_header.get("entry_count"),
            "content_fnv1a64": ledger_header.get("content_fnv1a64"),
            "generator": ledger_header.get("generator"),
            "entries_total": len(ledger_entries),
            "entries_live": sum(1 for e in ledger_entries if not e.get("demoted")),
            "entries_matching_this_device": sum(
                1 for e in ledger_entries
                if str(e.get("device") or "") == str((_device_identity(0)).get("name") or "")),
            "verdicts": {
                v: sum(1 for e in ledger_entries if e.get("verdict") == v)
                for v in sorted({str(e.get("verdict")) for e in ledger_entries})
            },
            "runtime_enforcement_observed_in": "equivalence[].runtime_enforcement",
            "why_here": (
                "this ledger is baked into the artifact and re-checked per claim decision: the "
                "EP declines a Ready row whose proof key has no entry. The equivalence pass "
                "runs with ONNXRUNTIME_EP_VULKAN_CLAIM_LOG set, so every timed configuration is "
                "accompanied by the EP's own record of which nodes it claimed and whether the "
                "ledger backed them. It is part of the semantic delta of these runs, not a "
                "separate artifact that happens to sit in the tree."
            ),
        },
        "raw": {
            "note": "every timed iteration, unaggregated, per arm and repeat",
            "runs": [
                {k: v for k, v in rec.items() if k != "witness"}
                for key in runs for rec in runs[key]
            ],
        },
    }
    out = Path(args.out)
    digest = freeze(record, out)
    print(f"\nfrozen {out} content_sha256={digest}")
    verdict = evidence_gate(load_frozen(out))
    print(json.dumps(verdict, indent=1))
    return 0 if verdict["verdict"] == PASS else 1


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
