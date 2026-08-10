"""Publication gate for the Phi-3.5 real-model evidence (issue #69), revision v5.

WHY THIS FILE EXISTS
====================
Issue #69 asks one question — *is the real model faster now?* — on a box that
`docs/PERF.md` §20 declares **permanently contended**. Revision v4 (PR #117) answered it
with a table of prefill ratios labelled ``IMPROVEMENT``. The independent reviewer rejected
that head: the wall clock on this machine is ``STEADY_UNCERTIFIED`` by default (§20.1), no
``QUIET`` quiescence verdict was attached (§10.0), and no device-state companion showing the
board at boost was present (§20 / §15.2). An A/A repeatability band measures *noise*; it does
not establish *quiet, boost, or sole-GPU tenancy*, so it cannot license a speed claim.

This module is the single **publication authority** for the v5 evidence. It is written from
`docs/PERF.md`, issue #69, and the reviewer's findings only. It is deliberately independent of
`bench/phi_evidence.py` (the rejected v4 harness) in both design and code: this is a *gate over
a frozen record*, not a measuring harness, and it **suppresses every timing number when the
machine cannot certify one**.

WHAT IT ENFORCES (the reviewer's requirements, one named condition each)
=======================================================================
`GATE_CONDITIONS` is the registry the negative-control battery attacks one arm at a time.
Every condition is a pure function of the record — no clock, no GPU, no ORT session — so it
runs in the host-free lane and cannot be flattered by a busy box.

The seven requirements from the rejection map onto these conditions:

1. *Uncertified prefill conclusions become INDETERMINATE / STEADY_UNCERTIFIED* — a prefill
   subject is never published as ``IMPROVEMENT``; it is ``STEADY_UNCERTIFIED`` unless
   ``quiescence_quiet`` **and** ``device_state_companion`` both pass, which on this box they
   do not.
2. *PERF §20 companion / device-state admissibility is required* — ``device_state_companion``.
3. *Timings / ratios / speedups are suppressed on refusal* — `_suppress_timings` scrubs every
   time-bearing field out of the published record whenever timing is inadmissible, by key name
   **and** by value (a wall-clock float copied under an innocuous key is caught too).
4. *Defensible uncertainty / power qualification* — ``uncertainty_qualified``.
5. *Harness / build recipe / dirty state are bound to immutable identities* —
   ``immutable_run_binding`` (rejects a run whose source commit is the mutable checkout
   ``HEAD`` sentinel, or that is missing a DLL / build-recipe / dirty-state field).
6. *The false CI-only provenance claim is corrected* — ``provenance_claim_accurate``.
7. *Decode past=128 is kept separate and unpooled* — ``decode_p128_separate``.

The reviewer also named integrity gaps a green suite missed; they are conditions too:
``per_output_integrity`` (raw→aggregate reconciliation; all 65 outputs), ``device_identity_immutable``
and ``model_identity_provenance`` (no fabricated UUID / replaced model hash passes),
``refusal_output_sanitized`` (a refusal must not echo a matched private path) and
``digests_platform_stable`` (a digest recorded over CRLF bytes must not be accepted where the
tree stores LF).

WHAT SURVIVES A REFUSAL
=======================
Per §10.0, counts and integer identities cannot be corrupted by contention, so a refusal still
publishes island/dispatch counts, device and model identity, and -- only when a full 65-output
record is supplied -- the all-output correctness verdict. That verdict is never asserted without
such a record: the standing posture carries no record, so it claims no correctness result (see
``correctness.established``). Withholding the surviving witnesses would discard the falsifiers
that cost the most to collect. Only the wall-clock numbers are suppressed.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Callable

# --------------------------------------------------------------------------- #
# Verdict vocabulary. These strings are the source-of-truth verdict names in
# `bench/contention.py` (QUIET/CONTENDED/UNMEASURED) and `bench/device_companion.py`
# (QUOTABLE/WITHHELD/UNCERTIFIED). They are re-declared here so the gate stays
# import-light and host-free; `test_phi69_evidence.py::test_verdict_names_track_source`
# imports the real modules and fails if these drift.
# --------------------------------------------------------------------------- #
QUIET = "QUIET"
CONTENDED = "CONTENDED"
UNMEASURED = "UNMEASURED"

COMPANION_QUOTABLE = "QUOTABLE"
COMPANION_WITHHELD = "WITHHELD"
COMPANION_UNCERTIFIED = "UNCERTIFIED"

#: The verdict a wall-clock figure carries on this hardware by default (§20.1).
STEADY_UNCERTIFIED = "STEADY_UNCERTIFIED"
#: The timing verdict this project returns for #69 while isolation is unavailable.
INDETERMINATE = "INDETERMINATE"

#: The sentinel written in place of every suppressed timing value.
SUPPRESSED = "SUPPRESSED -- inadmissible timing on a non-QUIET box (see refusals)"

#: Phi-3.5 exposes one logits output plus 32 layers × (key, value) = 64 KV outputs.
EXPECTED_OUTPUTS = 65

#: The decode observations that must be preserved by name; a friendlier third reading may
#: never silently stand in for either (the reviewer's `REQUIRED_DECODE_OBSERVATIONS`).
REQUIRED_DECODE_OBSERVATIONS = ("0.859x", "0.9651x")

#: The device this project actually runs on (docs/PERF.md device evidence; confirmed by the
#: EP's own running-device UUID). A fabricated or placeholder UUID must not pass.
PINNED_DEVICE_UUID = "aadf33d4d118155fcc60c22b5c352463"

#: A source commit is a 40-char hex sha. The literal below is the mutable-checkout sentinel the
#: v4 harness recorded from `git rev-parse HEAD`; binding a run to it is not immutable.
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_MUTABLE_HEAD_SENTINELS = {"HEAD", "head", "", None}

#: Keys whose values are wall-clock quantities. Suppressed on refusal.
_TIMING_KEY = re.compile(
    r"(_ms$|^ms$|ratio|speedup|floor|median|_point$|^point$|band|throughput|latency)",
    re.IGNORECASE,
)

# Absolute user paths that a refusal detail must never re-echo.
_PRIVATE_PATH = re.compile(r"([A-Za-z]:\\Users\\[^\\\"']+|/home/[^/\"']+|/Users/[^/\"']+)")


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _get(d: Any, *path: str, default: Any = None) -> Any:
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _runs(record: dict) -> "list[dict]":
    runs = _get(record, "raw_runs", default=[])
    return runs if isinstance(runs, list) else []


def _subjects(record: dict) -> "list[dict]":
    subs = _get(record, "subjects", default=[])
    return subs if isinstance(subs, list) else []


def _lf_digest(text: str) -> str:
    """SHA-256 over LF-normalized bytes.

    A digest is only portable if it is taken over the bytes the tree stores. CRLF checkouts
    otherwise produce a different hash for identical content — the exact defect the reviewer
    found in the v4 proof-ledger digest (recorded as the CRLF hash, mismatching the Git blob).
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# conditions — each returns (ok: bool, reason: str)
# --------------------------------------------------------------------------- #
def record_wellformed(record: dict) -> "tuple[bool, str]":
    if not isinstance(record, dict):
        return False, "record is not an object"
    missing = [k for k in ("schema", "subjects", "raw_runs") if k not in record]
    if missing:
        return False, f"missing top-level fields: {', '.join(missing)}"
    if record.get("schema") != "phi69-evidence/v5":
        return False, f"unexpected schema {record.get('schema')!r}; expected phi69-evidence/v5"
    return True, "record has the v5 shape"


def immutable_run_binding(record: dict) -> "tuple[bool, str]":
    """Every raw run must be bound to an immutable build identity.

    The v4 harness recorded ``ep_source_commit`` from the current checkout HEAD and hard-coded
    the baseline commit, so a DLL from any commit could be labelled with any other. A run is
    bound iff it carries a DLL digest, a 40-hex source commit that is *not* the mutable ``HEAD``
    sentinel, a build-recipe digest, and an explicit dirty-state (with a files digest when dirty).
    """
    runs = _runs(record)
    if not runs:
        return False, "no raw runs to bind"
    commits = set()
    for i, run in enumerate(runs):
        label = run.get("arm", f"run[{i}]")
        commit = run.get("source_commit")
        if commit in _MUTABLE_HEAD_SENTINELS:
            return False, f"{label}: source_commit is the mutable checkout sentinel {commit!r}"
        if not (isinstance(commit, str) and _HEX40.match(commit)):
            return False, f"{label}: source_commit {commit!r} is not a 40-hex immutable id"
        if not run.get("dll_sha256"):
            return False, f"{label}: no dll_sha256 -- the binary that ran is unbound"
        if not run.get("build_recipe_sha256"):
            return False, f"{label}: no build_recipe_sha256 -- toolchain/build command unbound"
        if "worktree_dirty" not in run or not isinstance(run["worktree_dirty"], bool):
            return False, f"{label}: worktree_dirty must be an explicit bool"
        if run["worktree_dirty"] and not run.get("dirty_files_digest"):
            return False, f"{label}: dirty worktree without a dirty_files_digest"
        commits.add(commit)
    if len(commits) < 2:
        return False, "head and baseline arms must carry distinct source commits"
    return True, f"{len(runs)} runs bound to immutable dll/source/recipe/dirty identities"


def device_identity_immutable(record: dict) -> "tuple[bool, str]":
    dev = _get(record, "device", default={})
    uuid = (dev.get("uuid") or "").lower().replace("uuid:", "")
    for field in ("uuid", "luid", "pci_bus_id", "driver_version", "device_type"):
        if not dev.get(field):
            return False, f"device.{field} absent -- identity is not immutable"
    if uuid != PINNED_DEVICE_UUID:
        return False, (
            f"device uuid {uuid!r} does not match the pinned running device "
            f"{PINNED_DEVICE_UUID!r}; a non-empty string is not an identity check"
        )
    if dev.get("device_type") == "discrete-gpu" and re.search(
        r"lavapipe|llvmpipe|software|swiftshader", str(dev.get("driver_name", "")), re.I
    ):
        return False, "a discrete adapter is described in software (lavapipe) terms"
    return True, f"device pinned to {PINNED_DEVICE_UUID}"


def model_identity_provenance(record: dict) -> "tuple[bool, str]":
    model = _get(record, "model", default={})
    if not model.get("graph_sha256") or not model.get("weights_sha256"):
        return False, "model graph/weights digest absent"
    prov = model.get("provenance") or {}
    has_foundry = bool(prov.get("foundry_variant_id"))
    has_hf = bool(prov.get("hf_repo") and prov.get("hf_revision"))
    if not (has_foundry or has_hf):
        return False, "no Foundry variant id and no Hugging Face repo+revision provenance"
    return True, "model bound to digest and external provenance"


def quiescence_quiet(record: dict) -> "tuple[bool, str]":
    verdict = _get(record, "quiescence", "verdict", default=UNMEASURED)
    if verdict != QUIET:
        return False, (
            f"machine_quiescence={verdict}; PERF 10.0 forbids quoting any performance number "
            f"beside a non-QUIET verdict"
        )
    return True, "machine quiescence QUIET"


def device_state_companion(record: dict) -> "tuple[bool, str]":
    """PERF §20 / §15.2: a quotable figure needs the companion AND a boost device-state."""
    comp = _get(record, "device_state", default=None)
    if not comp:
        return False, "device_state ABSENT -- absence of the companion is a refusal, not a pass"
    verdict = comp.get("verdict")
    if verdict != COMPANION_QUOTABLE:
        return False, (
            f"device_state companion verdict={verdict}; only QUOTABLE (STEADY + SOLE_TENANT + "
            f"peak SM >= board boost floor) releases a number"
        )
    return True, "device-state companion QUOTABLE"


def _per_output_lists(record: dict) -> "list[tuple[str, list]]":
    out = []
    for sub in _subjects(record):
        name = sub.get("name", "?")
        for arm in sub.get("arms", []):
            out.append((f"{name}/{arm.get('arm', '?')}", arm.get("per_output", [])))
    return out


def per_output_integrity(record: dict) -> "tuple[bool, str]":
    """All 65 outputs per arm, and the aggregate must reconcile with the rows.

    The reviewer showed the v4 gate passed after truncating every ``per_output`` list to one
    row, or flipping a KV row to ``DIVERGENT`` while leaving the arm aggregate ``MATCH``. Both
    are raw→aggregate disconnects, and both fail here.
    """
    seen = False
    for label, rows in _per_output_lists(record):
        seen = True
        if len(rows) != EXPECTED_OUTPUTS:
            return False, f"{label}: {len(rows)} per-output rows, expected {EXPECTED_OUTPUTS}"
    for sub in _subjects(record):
        for arm in sub.get("arms", []):
            rows = arm.get("per_output", [])
            row_all_match = all(r.get("verdict") == "MATCH" for r in rows)
            agg = arm.get("aggregate")
            if (agg == "MATCH") != row_all_match:
                return False, (
                    f"{sub.get('name')}/{arm.get('arm')}: aggregate {agg!r} does not "
                    f"reconcile with its {len(rows)} per-output rows"
                )
    if not seen:
        return False, "no per-output rows present"
    return True, f"all arms carry {EXPECTED_OUTPUTS} reconciled per-output rows"


def all_output_equivalence(record: dict) -> "tuple[bool, str]":
    total = 0
    for label, rows in _per_output_lists(record):
        for r in rows:
            total += 1
            if r.get("verdict") != "MATCH":
                return False, f"{label}: output {r.get('index')} is {r.get('verdict')}"
    if total == 0:
        return False, "no per-output verdicts to check"
    return True, f"{total} per-output verdicts all MATCH"


def calibration_content_disjoint(record: dict) -> "tuple[bool, str]":
    feeds = _get(record, "feeds_digest_by_subject", default={})
    cal = set(_get(record, "calibration_subjects", default=[]))
    ver = set(_get(record, "verdict_subjects", default=[]))
    if not feeds or not cal or not ver:
        return False, "feeds_digest_by_subject / calibration / verdict subjects incomplete"
    cal_digests = {feeds.get(s) for s in cal}
    ver_digests = {feeds.get(s) for s in ver}
    if None in cal_digests or None in ver_digests:
        return False, "a subject has no feed digest"
    if cal_digests & ver_digests:
        return False, "calibration and verdict subjects share a feed digest (disjoint by label only)"
    return True, "calibration is content-disjoint from verdict subjects"


def decode_p128_separate(record: dict) -> "tuple[bool, str]":
    names = [s.get("name") for s in _subjects(record)]
    decode = next((s for s in _subjects(record) if s.get("name") == "decode/M1/past128"), None)
    if decode is None:
        return False, "decode/M1/past128 subject absent"
    if decode.get("pooled"):
        return False, "decode p128 is pooled -- it must stay separate and unpooled"
    if decode.get("timing_verdict") != INDETERMINATE:
        return False, f"decode p128 timing_verdict={decode.get('timing_verdict')}, expected INDETERMINATE"
    preserved = decode.get("preserved_observations", [])
    for obs in REQUIRED_DECODE_OBSERVATIONS:
        if obs not in preserved:
            return False, f"prior decode observation {obs!r} not preserved by name"
    # p128 must not be folded into a prefill pool.
    for s in _subjects(record):
        if s.get("name", "").startswith("prefill") and "decode/M1/past128" in s.get("pool", []):
            return False, "decode p128 folded into a prefill pool"
    return True, f"decode p128 separate, INDETERMINATE, {len(names)} subjects"


def provenance_claim_accurate(record: dict) -> "tuple[bool, str]":
    """The base-delta claim must not repeat the false 'CI-only / no bench source' form.

    The reviewer contradicted v4's §27.1: the delta from the measured commit to the PR base
    changed ``bench/`` tooling, docs and CI — only Rust/shader source was untouched. A text
    screen rejects the strong false form and requires the accurate one.
    """
    claim = str(_get(record, "provenance", "base_delta_claim", default=""))
    if not claim:
        return False, "no base_delta_claim recorded"
    low = claim.lower()
    false_forms = ("ci-only", "ci only", "no bench source", "touches no bench")
    if any(f in low for f in false_forms):
        return False, "base_delta_claim repeats the false CI-only / no-bench-source form"
    if not ("rust" in low and ("shader" in low or "kernel" in low)):
        return False, "base_delta_claim must scope the delta to Rust/shader source explicitly"
    return True, "base-delta provenance claim is accurately scoped"


def refusal_output_sanitized(record: dict) -> "tuple[bool, str]":
    """No published field may echo a matched absolute user path -- not only refusals.

    The reviewer found ``private_path_disclosed`` embedded the matched home path in ``detail``
    and the checker re-printed it. A detection must carry a boolean and a redacted marker only.
    The v4 fix scanned only ``refusals``; that scope is incomplete, because ``publish`` also
    echoes ``device``, ``model``, ``witnesses``, ``subjects`` and every condition reason. This
    scans the **whole record surface** so a private path hidden in any published field is caught.
    """
    def walk(node: Any, where: str) -> "str | None":
        if isinstance(node, dict):
            for k, v in node.items():
                hit = walk(v, f"{where}.{k}")
                if hit:
                    return hit
        elif isinstance(node, list):
            for i, v in enumerate(node):
                hit = walk(v, f"{where}[{i}]")
                if hit:
                    return hit
        elif isinstance(node, str) and _PRIVATE_PATH.search(node):
            return where
        return None

    hit = walk(record, "record")
    if hit:
        return False, f"a published field at {hit} echoes a private absolute path"
    return True, "no published field carries a private absolute path"


def digests_platform_stable(record: dict) -> "tuple[bool, str]":
    """Recorded content digests must equal the LF-normalized recomputation.

    Each entry of ``content_digests`` is ``{name, recorded, source_text}``; the recorded hash
    must match ``_lf_digest(source_text)`` so a CRLF checkout cannot change it.
    """
    entries = _get(record, "content_digests", default=[])
    if not entries:
        return False, "no content_digests to verify"
    for e in entries:
        name = e.get("name", "?")
        if "source_text" not in e:
            return False, f"{name}: no source_text to recompute the digest"
        want = _lf_digest(e["source_text"])
        if e.get("recorded") != want:
            return False, f"{name}: recorded digest is not the LF-normalized hash (CRLF drift)"
    return True, f"{len(entries)} content digests are platform-stable (LF)"


def no_dispersion_promotion(record: dict) -> "tuple[bool, str]":
    """Within-arm dispersion must be present and explicitly diagnostic.

    An absent block used to pass vacuously -- the condition then certified nothing, because a
    record that simply omitted the field was indistinguishable from one that proved the
    dispersion never moved a verdict. The block must exist, name ``role == "diagnostic"``, and
    carry an explicit ``moved_a_verdict`` that is false.
    """
    disp = _get(record, "within_arm_dispersion", default=None)
    if not isinstance(disp, dict) or "role" not in disp:
        return False, "within_arm_dispersion block absent -- cannot certify it moved no verdict"
    if disp.get("role") != "diagnostic":
        return False, f"within-arm dispersion role={disp.get('role')!r}, must be diagnostic"
    if "moved_a_verdict" not in disp or not isinstance(disp["moved_a_verdict"], bool):
        return False, "within-arm dispersion must carry an explicit moved_a_verdict bool"
    if disp["moved_a_verdict"]:
        return False, "within-arm dispersion was used to move a verdict"
    return True, "within-arm dispersion present and diagnostic only"


def isolation_language_cooperative(record: dict) -> "tuple[bool, str]":
    lang = str(_get(record, "isolation", "language", default="")).lower()
    if not lang:
        return False, "no isolation language recorded"
    if "exclusive" in lang and "gpu" in lang:
        return False, "isolation language claims exclusive GPU ownership"
    if "cooperat" not in lang:
        return False, "isolation must be described as cooperative process exclusion"
    return True, "isolation described as cooperative process exclusion only"


def headline_scope_not_widened(record: dict) -> "tuple[bool, str]":
    scope = _get(record, "headline_scope", default={})
    for axis in ("model", "prefill_family", "adapter", "box"):
        if scope.get(axis) != "one":
            return False, f"headline scope on {axis} is {scope.get(axis)!r}, must be 'one'"
    if scope.get("cuda_comparison") != "NONE":
        return False, "a CUDA comparison is claimed; #69 stays open without one"
    if scope.get("closes_issue_69"):
        return False, "record claims to close #69"
    return True, "headline scope is one model / family / adapter / box; #69 open"


def uncertainty_qualified(record: dict) -> "tuple[bool, str]":
    """A defensible figure carries its own uncertainty and a power/boost qualification.

    Even a refused figure must record why: an A/A band measures observed noise but does not
    establish quiet / boost / concurrent-GPU state, and that limitation travels with the record.
    """
    unc = _get(record, "uncertainty", default={})
    if not unc.get("aa_band"):
        return False, "no A/A noise band recorded"
    q = str(unc.get("power_boost_qualification", "")).lower()
    if not q:
        return False, "no power/boost qualification recorded"
    if "does not establish" not in q or ("boost" not in q and "power" not in q):
        return False, "power/boost qualification does not disclaim quiet/boost/concurrent state"
    return True, "uncertainty carries an A/A band and a power/boost disclaimer"


# --------------------------------------------------------------------------- #
# the condition registry — the negative-control battery attacks one arm each
# --------------------------------------------------------------------------- #
GATE_CONDITIONS: "dict[str, Callable[[dict], tuple[bool, str]]]" = {
    "record_wellformed": record_wellformed,
    "immutable_run_binding": immutable_run_binding,
    "device_identity_immutable": device_identity_immutable,
    "model_identity_provenance": model_identity_provenance,
    "quiescence_quiet": quiescence_quiet,
    "device_state_companion": device_state_companion,
    "per_output_integrity": per_output_integrity,
    "all_output_equivalence": all_output_equivalence,
    "calibration_content_disjoint": calibration_content_disjoint,
    "decode_p128_separate": decode_p128_separate,
    "provenance_claim_accurate": provenance_claim_accurate,
    "refusal_output_sanitized": refusal_output_sanitized,
    "digests_platform_stable": digests_platform_stable,
    "no_dispersion_promotion": no_dispersion_promotion,
    "isolation_language_cooperative": isolation_language_cooperative,
    "headline_scope_not_widened": headline_scope_not_widened,
    "uncertainty_qualified": uncertainty_qualified,
}

#: A wall-clock number may be published only when **every** condition passes. Splitting out a
#: "timing-only" subset let an anti-overclaim condition (decode still pooled, headline widened
#: to close #69, dispersion promoted, isolation claiming exclusive GPU) fail while a QUOTABLE
#: figure was still emitted. Admissibility is therefore the whole registry; the structural /
#: correctness / disclosure witnesses below are what still *publishes* under a refusal, a
#: separate question from what licenses a number.
TIMING_ADMISSIBILITY = tuple(GATE_CONDITIONS)

#: The witnesses that survive a refusal (still published when timing is withheld). Named for the
#: report and prose; not an admissibility subset -- admissibility requires all conditions.
REFUSAL_SURVIVING_WITNESSES = (
    "immutable_run_binding",
    "device_identity_immutable",
    "model_identity_provenance",
    "per_output_integrity",
    "all_output_equivalence",
    "digests_platform_stable",
)


def _timing_floats(node: Any) -> "set[float]":
    """Every float that lives under a wall-clock-bearing key, recursively.

    Timings are floats; integer counts (islands, dispatches) are not collected, so the
    value-based backstop below cannot clobber a survivable count that merely shares a magnitude.
    """
    def _floats_below(n: Any) -> "set[float]":
        out: "set[float]" = set()
        if isinstance(n, dict):
            for v in n.values():
                out |= _floats_below(v)
        elif isinstance(n, list):
            for v in n:
                out |= _floats_below(v)
        elif isinstance(n, float):
            out.add(n)
        return out

    found: "set[float]" = set()
    if isinstance(node, dict):
        for k, v in node.items():
            found |= _floats_below(v) if _TIMING_KEY.search(k) else _timing_floats(v)
    elif isinstance(node, list):
        for v in node:
            found |= _timing_floats(v)
    return found


def _all_floats(node: Any) -> "set[float]":
    out: "set[float]" = set()
    if isinstance(node, dict):
        for v in node.values():
            out |= _all_floats(v)
    elif isinstance(node, list):
        for v in node:
            out |= _all_floats(v)
    elif isinstance(node, float):
        out.add(node)
    return out


def _residual_timing_leak(record: dict, published: dict) -> "list[float]":
    """Any wall-clock float from the input that survived into the published subjects.

    This is the leak detector the checker and negative control use *instead of* a hard-coded
    list of forbidden literals: it derives the banned set from the record under test, so it
    catches a leak of any value -- including one copied under an innocuous key name -- and never
    embeds a specific measurement in the repository. It is a private transformation helper (like
    ``_suppress_timings``), exercised in both polarities by ``test_phi69_evidence.py``.
    """
    banned = _timing_floats(_get(record, "subjects", default=[]))
    present = _all_floats(_get(published, "subjects", default=[]))
    return sorted(banned & present)


def _suppress_timings(node: Any, banned: "set[float] | None" = None) -> Any:
    """Recursively replace every wall-clock-bearing value with the SUPPRESSED sentinel.

    Two passes, because key-name matching alone is not enough: a timing float copied under an
    innocuous key would otherwise leak. ``banned`` is the set of timing floats gathered from the
    node; any float equal to one is suppressed wherever it appears, on top of scrubbing every
    value under a timing-named key.

    Section 10.0: a number printed under a warning gets quoted without the warning, so on refusal
    the refusal is printed *in place of* the medians, delta and ratio -- not beside them.
    """
    if banned is None:
        banned = _timing_floats(node)
    if isinstance(node, dict):
        return {
            k: (SUPPRESSED if _TIMING_KEY.search(k) else _suppress_timings(v, banned))
            for k, v in node.items()
        }
    if isinstance(node, list):
        return [_suppress_timings(v, banned) for v in node]
    if isinstance(node, float) and node in banned:
        return SUPPRESSED
    return node


def _evaluate(record: dict) -> dict:
    """Run every condition. Returns {condition: {ok, reason}}.

    A batch runner over ``GATE_CONDITIONS``; it renders no verdict of its own — the verdict
    of each row is the condition's, and the publication verdict is ``publish``'s.
    """
    return {name: dict(zip(("ok", "reason"), fn(record))) for name, fn in GATE_CONDITIONS.items()}


def publish(record: dict) -> dict:
    """The single publication authority for the #69 v5 evidence.

    Returns a reader-facing dict. When timing is inadmissible (the standing state of this box),
    every prefill/decode timing verdict is INDETERMINATE / STEADY_UNCERTIFIED and all timing
    numbers are scrubbed out; the structural and correctness witnesses remain.
    """
    conditions = _evaluate(record)
    refusals = [
        {"condition": name, "reason": c["reason"]}
        for name, c in conditions.items()
        if not c["ok"]
    ]
    timing_admissible = all(conditions[name]["ok"] for name in TIMING_ADMISSIBILITY)

    published: dict = {
        "issue": 69,
        "revision": "v5",
        "schema": record.get("schema"),
        "timing_admissible": timing_admissible,
        "timing_verdict": None,
        "conditions": {k: v["ok"] for k, v in conditions.items()},
        "refusals": refusals,
        # Witnesses that contention cannot corrupt (§10.0): counts and identities.
        "witnesses": _get(record, "witnesses", default={}),
        "device": _get(record, "device", default={}),
        "model": _get(record, "model", default={}),
        "correctness": {
            "all_output_equivalence": conditions["all_output_equivalence"]["ok"],
            "reason": conditions["all_output_equivalence"]["reason"],
            # F7: the "all 65 outputs equivalent" witness is only *established* when the record
            # actually carries the 65 reconciled rows AND every one matches. With no record (the
            # standing posture) it is False -- the gate never claims a correctness result it did
            # not obtain.
            "established": (
                conditions["per_output_integrity"]["ok"]
                and conditions["all_output_equivalence"]["ok"]
            ),
            "outputs_expected": EXPECTED_OUTPUTS,
            "outputs_verified": sum(
                len(rows) for _, rows in _per_output_lists(record)
            ),
        },
        "subjects": _subjects(record),
    }

    if timing_admissible:
        published["timing_verdict"] = "QUOTABLE"
        return published

    # Refusal path: convert every timing verdict and suppress every number.
    published["timing_verdict"] = INDETERMINATE
    published["wall_clock_class"] = STEADY_UNCERTIFIED
    published["subjects"] = _suppress_timings(published["subjects"])
    scrubbed = []
    for s in published["subjects"]:
        if isinstance(s, dict):
            s = dict(s)
            if s.get("name", "").startswith("prefill"):
                s["timing_verdict"] = STEADY_UNCERTIFIED
            elif s.get("name", "").startswith("decode"):
                s["timing_verdict"] = INDETERMINATE
        scrubbed.append(s)
    published["subjects"] = scrubbed
    return published


def _report(published: dict) -> "list[str]":
    lines = [
        "PHI69-EVIDENCE v5 -- issue #69",
        f"  timing_admissible : {published['timing_admissible']}",
        f"  timing_verdict    : {published['timing_verdict']}",
    ]
    if not published["timing_admissible"]:
        lines.append(f"  wall_clock_class  : {published.get('wall_clock_class')}")
        lines.append("  REFUSED -- no timing/ratio/speedup published. Reasons:")
        for r in published["refusals"]:
            lines.append(f"    - {r['condition']}: {r['reason']}")
    corr = published["correctness"]
    if corr.get("established"):
        lines.append(
            f"  correctness       : all-output equivalence ESTABLISHED "
            f"({corr.get('outputs_verified')}/{corr.get('outputs_expected')} outputs MATCH)"
        )
    else:
        lines.append(
            f"  correctness       : all-output equivalence NOT ESTABLISHED "
            f"({corr.get('outputs_verified')}/{corr.get('outputs_expected')} outputs recorded) "
            f"-- no correctness result is claimed without a full record"
        )
    return lines
