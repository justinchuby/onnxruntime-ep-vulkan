"""The `model_output_equivalence` record — DESIGN.md §10.0 THIRD METRIC AMENDMENT.

WHY THIS MODULE EXISTS
======================
Morpheus ruled (2026-07-31T07:45:10-07:00) that *a verdict about output equivalence is
not a verdict about this EP unless it carries what executed the model*.  The specimen:
ORT fell back to CPU inside ``run()``, printed ``EP_FAIL ... Falling back``, raised
nothing, ``get_providers()`` still listed ``VulkanExecutionProvider`` (the provider list
is fixed at session-create time) — and the comparison gate compared a CPU run against a
CPU run and returned ``MATCH``.  Wired, invoked, correctly named, arithmetically correct,
and about a different world (R12).

Before this module, ``model_output_equivalence`` was a **string a caller chose**:

    m.write_equivalence_verdict(counters_path, m.EQUIVALENCE_MATCH)   # a literal

That is R10 amendment 1 exactly — *a value must be one a mechanism computed on this run,
never a flag its author set*.  The ruling names five binding clauses; this module is the
mechanism for clauses 1, 3, 4 and 5, and it carries the data clause 2 needs.

THE FIVE CLAUSES AND WHERE EACH ONE LIVES
=========================================
1. **Attribution comes from an instrument we do not own.**
   :meth:`ExecutionAttribution.from_profile` parses ORT's own profiling trace — one
   ``cat == "Node"`` event per executed (fused) node, carrying ``args["provider"]``.
   Our own ``dispatches_executed`` is recorded as a *corroborating* witness and is never
   the primary one: it lives inside the frame whose existence is in question.

2. **Both witnesses are recorded, and disagreement is red.**
   :meth:`ExecutionAttribution.with_counters_witness` attaches the second witness; when
   the two disagree the derived verdict is ``SPLIT-FRAME`` and no triple may be reported.

3. **``MATCH`` is unrepresentable without a non-zero own-provider count.**
   Structural, not asserted-afterwards.  Three mechanisms together:
     - ``MATCH`` is **not an input to any constructor**.  The caller supplies a
       *comparison outcome* (``AGREE`` / ``DISAGREE`` / ``NOT_PERFORMED``) — a statement
       about tensors, which is all a comparison can honestly produce — and the verdict is
       *derived*.  Passing ``"MATCH"`` raises :class:`ValueError`.
     - The derivation requires an :class:`ExecutionAttribution` instance; a dict, a
       string, or ``None`` raises :class:`TypeError`.
     - :class:`ExecutionAttribution` has a private constructor.  The only way to obtain
       one is to parse a profile **file**, whose path and mtime are recorded on the
       record so a stale profile from a previous run is visible.
   There is therefore no expression in this codebase that evaluates to a ``MATCH``
   verdict at ``own_count == 0``.  ``test_verdict.py`` proves it by trying.

4. **``UNATTRIBUTED`` is not ``DIVERGENT`` and must never be folded into it.**
   ``DIVERGENT`` says *our kernels computed the wrong answer*.  ``UNATTRIBUTED`` says
   *our kernels did not run, so the answer is not about them*.  Different owners,
   different fixes, different next questions.  Both void the triple; they are never the
   same token, never the same exit code, never the same message.

5. **The guard is an input to the verdict, not a neighbour of it.**
   Guard D's *observation* is :class:`ExecutionAttribution`; Guard D's *assertion* is
   :meth:`ExecutionAttribution.assert_executed`, which stays in the lane as a fast,
   legible failure.  The observation is load-bearing; the assertion is a convenience.
   A caveat that lives in a different artifact from the number it qualifies is not
   attached to it — so the attribution is written into the counters JSON beside the
   verdict, where ``epctl``, ``bench/`` and the census read it.

THE VOCABULARY — ONE VOCABULARY, NOT TWO
========================================
Mirrored, deliberately and by the project's existing convention, in three places:

  - here (Python: the writer)                    ``tests/ops/_verdict.py``
  - ``rust/src/counters.rs``  (Rust: the default writer and the parser)
  - ``rust/src/bin/epctl.rs`` (Rust: the gate)
  - ``bench/admissible.py``   (Python: Niobe's admissibility gate)

+-----------------+------------------------------------------------------+---------------+
| Token           | Meaning                                              | May report... |
+=================+======================================================+===============+
| ``MATCH``       | Outputs agree **and** this EP executed >= 1 node,    | the triple,   |
|                 | evidenced by an instrument we do not own             | the ratio     |
+-----------------+------------------------------------------------------+---------------+
| ``DIVERGENT``   | This EP executed, and an output disagrees            | nothing       |
+-----------------+------------------------------------------------------+---------------+
| ``UNMEASURED``  | No CPU-only comparison was performed                 | nothing       |
+-----------------+------------------------------------------------------+---------------+
| ``UNATTRIBUTED``| The comparison was performed and this EP produced none  | nothing     |
|                 | of the numbers compared — either it executed zero       |             |
|                 | nodes, or nothing it executed reaches any compared      |             |
|                 | output.  The comparison is correct and about another    |             |
|                 | world.  **Not** ``DIVERGENT``                           |             |
+-----------------+------------------------------------------------------+---------------+
| ``SPLIT-FRAME`` | The two witnesses disagree about whether this EP ran | nothing       |
+-----------------+------------------------------------------------------+---------------+

Precedence when several apply, in the order the checks run:

  1. ``SPLIT-FRAME`` — the witnesses disagree, so nothing downstream can be trusted,
     including the statement that no comparison happened.
  2. ``UNMEASURED``  — no comparison was performed; there is nothing to attribute.
  3. ``UNATTRIBUTED``— a comparison happened and none of it is about us.  This holds
     whether the outputs agreed **or** disagreed: a CPU-vs-CPU disagreement is a
     statement about nondeterminism in the oracle, not about our kernels.  The comparison
     outcome is preserved in the record's ``comparison`` field so nothing is lost.
  4. ``MATCH`` / ``DIVERGENT`` — attributed, so the arithmetic is about us.

THE FIFTH COSTUME (2026-08-02)
==============================
Clause 3 made ``MATCH`` unrepresentable at ``own_count == 0``.  ``own_count`` is a
property of the **session**; the oracle comparison is a property of each **output**.  At
zero they compose safely.  They stop composing the moment the proof ledger fills
part-way: the EP claims some nodes, attribution says yes, ``MATCH`` becomes
representable, and the outputs whose producing nodes still decline are compared
CPU-against-CPU under a verdict that says the EP ran.

Morpheus's discharge condition (c) is a non-triviality guard on both sides — "64 pairs of
zeros satisfy (a) perfectly, which is ``0.0 == 0.0`` in a fourth costume".  A guard on
constancy cannot see a comparison whose two sides are the **same computation**: those are
real, varying, input-dependent values on both sides.  :class:`OutputAttribution` is what
names that state, and :attr:`ExecutionAttribution.attributed` now carries two conditions
rather than one.  It adds **no sixth verdict token**: a comparison that cannot be
attributed to this EP is exactly what ``UNATTRIBUTED`` already meant, so Link's
``ci/check_verdict.py`` and Niobe's ``bench/admissible.py`` keep their exhaustive
branches.

R13 (§10.0.1) — THREE TERMINAL STATES
=====================================
A check has three terminal states — ``PASS``, ``FAIL(condition)`` and
``ERROR(instrument)`` — and **an instrument error never counts as a detection**.  This
module supplies the exception vocabulary the whole harness classifies against:

  - :class:`InstrumentError` (a ``RuntimeError``) — the check did not reach its
    observation.  Never a finding about the EP.
  - ``AssertionError`` — the check reached its observation and the observation is bad.
    A finding about the EP.

Every guard in this module raises :class:`InstrumentError` *before* it has a value and
``AssertionError`` *only after* it has one, so the classification is by construction
rather than by care.  ``conftest.py`` reads the same vocabulary to print the lane's three
counts separately, and quotes failure **text**, never a failure count.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

EP_NAME = "VulkanExecutionProvider"

# ---------------------------------------------------------------------------
# Vocabulary — mirrored in rust/src/counters.rs, rust/src/bin/epctl.rs,
# and bench/admissible.py.  Changing a token here is a cross-agent change.
# ---------------------------------------------------------------------------

#: Outputs agree AND this EP executed at least one node (attributed).
VERDICT_MATCH: str = "MATCH"
#: This EP executed and an output disagrees with the CPU oracle.
VERDICT_DIVERGENT: str = "DIVERGENT"
#: No CPU-only comparison was performed on this artifact in this run.
VERDICT_UNMEASURED: str = "UNMEASURED"
#: A comparison was performed and this EP executed zero nodes.  Not DIVERGENT.
VERDICT_UNATTRIBUTED: str = "UNATTRIBUTED"
#: The profile witness and the counters witness disagree about whether this EP ran.
VERDICT_SPLIT_FRAME: str = "SPLIT-FRAME"

#: Every token a reader may encounter in `model_output_equivalence`.
VERDICTS: tuple[str, ...] = (
    VERDICT_MATCH,
    VERDICT_DIVERGENT,
    VERDICT_UNMEASURED,
    VERDICT_UNATTRIBUTED,
    VERDICT_SPLIT_FRAME,
)

#: The only verdict that permits the triple, the ratio, or a correctness claim.
VERDICTS_PERMITTING_REPORT: frozenset[str] = frozenset({VERDICT_MATCH})

# --- comparison outcomes: what a comparison of tensors can honestly produce ---
# These are the constructor's inputs.  A verdict is never an input.

#: Every compared output agrees within the §9.1 tolerance policy.
COMPARISON_AGREE: str = "AGREE"
#: At least one compared output disagrees.
COMPARISON_DISAGREE: str = "DISAGREE"
#: No comparison against a CPU-only run of the same session was performed.
COMPARISON_NOT_PERFORMED: str = "NOT_PERFORMED"

COMPARISONS: tuple[str, ...] = (
    COMPARISON_AGREE,
    COMPARISON_DISAGREE,
    COMPARISON_NOT_PERFORMED,
)

#: The JSON key carrying the verdict token (string; what epctl and counters.rs parse).
EQUIVALENCE_KEY: str = "model_output_equivalence"
#: The JSON key carrying the full record (object; what bench/ and the census read).
EQUIVALENCE_RECORD_KEY: str = "model_output_equivalence_record"

#: The attribution instrument.  "ort_profile" is the only value that may support MATCH.
ATTRIBUTION_SOURCE_PROFILE: str = "ort_profile"
#: No attribution was taken (only ever paired with UNMEASURED).
ATTRIBUTION_SOURCE_NONE: str = "none"

# ---------------------------------------------------------------------------
# Per-output coverage — the fifth costume (2026-08-02)
# ---------------------------------------------------------------------------
# Attribution as built above is a property of the **session**: did this EP execute
# anything.  The oracle comparison is a property of each **output**.  At
# ``own_count == 0`` the two compose safely, because nothing is attributed and no MATCH
# is representable.  They stop composing the moment the ledger fills part-way:
#
#   The EP claims some nodes.  Attribution says yes.  MATCH becomes representable.  The
#   nodes producing outputs 1..64 are still declining, so those 64 comparisons are
#   CPU-against-CPU while the verdict says the EP ran.
#
# That is the same vacuity, passing through an attribution check that has already said
# yes, and it does not look like a defect: it looks like partial acceleration with a
# clean oracle, which is exactly what a filling ledger is supposed to look like.
#
# Morpheus's discharge condition (c) for criterion 10 is a non-triviality guard on both
# sides — "64 pairs of zeros satisfy (a) perfectly, which is 0.0 == 0.0 in a fourth
# costume".  A degeneracy guard tests each side for constancy.  It cannot see a
# comparison whose two sides are the *same computation*: those are real, varying,
# input-dependent CPU values on both sides.  This is the fifth costume, and the tokens
# below are what names it.

#: At least one node upstream of this output was not executed by any other provider —
#: it was absorbed into one of this EP's fused islands.  Qualified, not sound: see
#: :meth:`OutputAttribution.from_topology`.
OUTPUT_EP_COVERED: str = "EP-COVERED"
#: **Every** node upstream of this output was executed by some other provider.  This
#: output's oracle comparison is our-CPU against ORT's-CPU: vacuous by construction.
#: Sound in this direction — an optimiser can delete a node's event, never invent one.
OUTPUT_CPU_ONLY: str = "CPU-ONLY"
#: The question could not be put to this output in this frame (R12: never a bare 0 and
#: never a default of EP-COVERED, which would be the pass-by-absence this whole module
#: exists to make unrepresentable).
OUTPUT_UNOBSERVABLE: str = "UNOBSERVABLE"

OUTPUT_COVERAGE_TOKENS: tuple[str, ...] = (
    OUTPUT_EP_COVERED,
    OUTPUT_CPU_ONLY,
    OUTPUT_UNOBSERVABLE,
)

#: Per-output coverage was never computed on this run.  Distinct from "computed and
#: found nothing": the first is an instrument that was not run, the second is a reading.
OUTPUT_COVERAGE_NOT_COMPUTED: str = "not-computed"

# ---------------------------------------------------------------------------
# The second witness, and the three things it can say (R12)
# ---------------------------------------------------------------------------
# 2026-08-01, round 26.  An uncommitted re-run of criterion 10 on dev1 recorded
# ``"counters_dispatches_executed": null`` inside a passing ``AGREE``.  Reading the
# code back: ``split_frame`` returned ``False`` for a missing witness, and the verdict
# consults nothing else, so the witness was **recorded, never required**.  A witness
# that does not participate in the verdict is indistinguishable from one that was never
# wired (R10), and the artifact could not even say *why* it was absent — five distinct
# causes collapsed into one bare ``null`` (R12: UNOBSERVABLE with a reason, never a
# bare hole a reader fills in themselves).
#
# The fix is deliberately NOT "make MATCH depend on the counters witness".  The counters
# live inside the frame whose existence is in question; promoting them to a gate would
# move the check with the reader's confidence rather than with its subject (R9 A5).
# What the witness is *for* is the split-frame check — so the split-frame check gains a
# third state, and the criterion-10 lane (the canonical M0 record) requires that the
# check was performable at all, raising ERROR(instrument) when it was not (R13).

#: The two witnesses were both read and both say the same thing about *presence*.
WITNESS_AGREEMENT_AGREE: str = "AGREE"
#: Both read, and they disagree about presence.  This is the ``SPLIT-FRAME`` finding.
WITNESS_AGREEMENT_DISAGREE: str = "DISAGREE"
#: The second witness could not be read, so the check could not be performed.  R12: this
#: is *not* agreement, and the record must never let it read as agreement.
WITNESS_AGREEMENT_UNOBSERVABLE: str = "UNOBSERVABLE"

#: What ``counters_dispatches_executed`` says when there is no number.  A string token,
#: never ``null`` — the whole complaint is that ``null`` looks like a value that was not
#: interesting rather than an observation that did not happen.
WITNESS_UNOBSERVABLE: str = WITNESS_AGREEMENT_UNOBSERVABLE

#: Names of the witnesses, for ``attribution_witnesses_present``.
WITNESS_ORT_PROFILE: str = "ort_profile"
WITNESS_EP_COUNTERS: str = "ep_counters"

# The five distinct causes ``read_counters_dispatches`` used to collapse into one None,
# plus the two the caller can create.  Each is a sentence a reader can act on.
WITNESS_REASON_NOT_REQUESTED: str = (
    "the caller never attached a second witness (with_counters_witness was not called)"
)
WITNESS_REASON_NOT_ARMED: str = (
    "counters not armed: ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE was unset in this process. "
    "On Windows the DLL caches this variable at load time, so it CANNOT be armed from "
    "inside a test that has already imported onnxruntime -- it must be set before the "
    "process starts"
)
WITNESS_REASON_FILE_ABSENT: str = (
    "counters file was configured but does not exist: the EP never wrote a snapshot"
)
WITNESS_REASON_FILE_UNREADABLE: str = "counters file exists but could not be opened"
WITNESS_REASON_NOT_JSON: str = "counters file exists but is not parseable JSON"
WITNESS_REASON_FIELD_MISSING: str = (
    "counters file parsed, but has no numeric 'dispatches_executed' field"
)
WITNESS_REASON_UNKNOWN: str = "second witness absent for an unrecorded reason"


# ---------------------------------------------------------------------------
# R13 — instrument failure is a distinct terminal state, by construction
# ---------------------------------------------------------------------------

class InstrumentError(RuntimeError):
    """The check did not reach its observation — ``ERROR(instrument)``, never a detection.

    R13 (§10.0.1): *a check has at least three terminal states and must report them as
    three distinct tokens*.  Guard D raising ``NameError`` and Guard D correctly detecting
    a CPU fallback both presented as ``FAILED`` in the pytest summary line, and no reading
    of that line separated them.

    This subclasses ``RuntimeError`` so that every ``except RuntimeError`` written against
    Guard D before 2026-07-31 keeps working, and so that a bare ``except AssertionError``
    never swallows an instrument outage.

    An instrument error is a lane failure of a **different kind**: a lane with any
    instrument error is not a lane that ran, whatever else it reports.
    """

    def __init__(self, message: str, *, observed: Any = None) -> None:
        super().__init__(message)
        #: Whatever the instrument had managed to observe, if anything.  Usually ``None``
        #: — that is the point: an instrument error is the state in which there is no
        #: observation.
        self.observed = observed


# ---------------------------------------------------------------------------
# The observation — Guard D's reading, as a value
# ---------------------------------------------------------------------------

#: Private construction token.  ``ExecutionAttribution`` may only be built by a parse.
_PARSED = object()


class ExecutionAttribution:
    """What executed the graph, parsed from ORT's profiling trace.  Clause 1 and clause 5.

    This is Guard D's *observation* promoted to a value so it can be a constructor
    argument rather than a neighbouring assertion.  It cannot be fabricated: the
    constructor is private and :meth:`from_profile` requires a readable profile file,
    whose path, size, mtime and digest are recorded on the instance.

    Attributes
    ----------
    executed_by:
        ``{provider_name: node_event_count}`` for every provider that executed at least
        one node event.  Providers that executed nothing are absent — not zero (R12: a
        counter whose event cannot occur in its frame reports ``UNOBSERVABLE``, and a
        provider that was never in the session has no frame here at all).
    own_count:
        ``executed_by.get(EP_NAME, 0)`` — the number of **fused-island executions** this
        EP performed.  R11: this is not a graph-node count.  One island can cover
        hundreds of graph nodes (Phi-3.5: 355 of 363 in one island as of 2026-08-01, and
        that figure moves every time Mouse claims an op), so ``1`` is healthy
        and ``3`` for a three-run series is healthy.  Zero is the only conclusive value.
    """

    __slots__ = (
        "_executed_by",
        "_node_events",
        "_source",
        "_profile_path",
        "_profile_digest",
        "_profile_mtime_ns",
        "_counters_dispatches",
        "_counters_reason",
        "_output_coverage",
    )

    def __init__(
        self,
        *,
        _token: object = None,
        executed_by: Mapping[str, int],
        node_events: int,
        source: str,
        profile_path: str,
        profile_digest: str,
        profile_mtime_ns: int,
        counters_dispatches: int | None = None,
        counters_reason: str = WITNESS_REASON_NOT_REQUESTED,
        output_coverage: "OutputAttribution | None" = None,
    ) -> None:
        if _token is not _PARSED:
            raise TypeError(
                "ExecutionAttribution() is private.  R10 amendment 1 (DESIGN.md §10.0.1): "
                "an attribution must be a value a mechanism computed on this run, never a "
                "literal its author set.  Use ExecutionAttribution.from_profile(<path to "
                "the profile written by this session's end_profiling()>)."
            )
        self._executed_by = dict(executed_by)
        self._node_events = int(node_events)
        self._source = source
        self._profile_path = str(profile_path)
        self._profile_digest = profile_digest
        self._profile_mtime_ns = int(profile_mtime_ns)
        self._counters_dispatches = counters_dispatches
        self._counters_reason = (
            "" if counters_dispatches is not None else (counters_reason or WITNESS_REASON_UNKNOWN)
        )
        self._output_coverage = output_coverage

    # -- construction --------------------------------------------------------

    @classmethod
    def from_profile(
        cls,
        profile_path: "str | os.PathLike[str]",
        *,
        delete: bool = True,
    ) -> "ExecutionAttribution":
        """Parse ORT's profiling trace at *profile_path* into an attribution.

        The file is read and (by default) deleted, because ORT writes one per session and
        a leftover trace is exactly the "profile parsed from a previous run" cheat the
        ruling names as the second-cheapest way to fake an attribution.  The path, size,
        mtime and SHA-256 of the bytes actually read are recorded on the instance, so a
        reader of the artifact can tell one run's profile from another's.

        Raises
        ------
        InstrumentError
            The trace could not be read or parsed.  ``ERROR(instrument)``: this is a
            statement about the harness, never about the EP.  R13 obligation 1 — it is
            raised *before* any observation exists, so the state is by construction.
        """
        path = Path(profile_path)
        try:
            raw = path.read_bytes()
            stat = path.stat()
        except FileNotFoundError:
            raise InstrumentError(
                f"[attribution instrument failure] Profiling trace not found: {path}\n"
                "sess.end_profiling() should have created this file.  Check that\n"
                "SessionOptions.enable_profiling was True *before* session creation.\n"
                "This is an instrument outage, NOT a finding about the EP (R13)."
            ) from None
        except OSError as exc:
            raise InstrumentError(
                f"[attribution instrument failure] Could not read profiling trace {path}: {exc}\n"
                "This is an instrument outage, NOT a finding about the EP (R13)."
            ) from exc

        try:
            events = json.loads(raw.decode("utf-8", errors="replace"))
        except json.JSONDecodeError as exc:
            raise InstrumentError(
                f"[attribution instrument failure] Profiling trace at {path} is not valid "
                f"JSON: {exc}\n"
                "The file may be truncated (session closed before end_profiling) or corrupt.\n"
                "This is an instrument outage, NOT a finding about the EP (R13)."
            ) from exc
        finally:
            if delete:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass

        if not isinstance(events, list):
            raise InstrumentError(
                f"[attribution instrument failure] Profiling trace at {path} is JSON but not "
                f"a list of events (got {type(events).__name__}).  ORT writes a JSON array.\n"
                "This is an instrument outage, NOT a finding about the EP (R13)."
            )

        executed_by = tally_providers(events)
        node_events = sum(executed_by.values())
        return cls(
            _token=_PARSED,
            executed_by=executed_by,
            node_events=node_events,
            source=ATTRIBUTION_SOURCE_PROFILE,
            profile_path=str(path),
            profile_digest="sha256:" + hashlib.sha256(raw).hexdigest()[:16],
            profile_mtime_ns=getattr(stat, "st_mtime_ns", 0),
        )

    def with_counters_witness(
        self,
        dispatches_executed: int | None,
        *,
        reason: str = "",
    ) -> "ExecutionAttribution":
        """Return a copy carrying the second witness (clause 2).

        ``dispatches_executed`` comes from **our** counters JSON.  It may not be the
        primary witness — it lives inside the frame whose existence is in question — but
        recording it is what makes ``SPLIT-FRAME`` detectable.  *Two witnesses that can
        only ever agree are one witness.*

        ``None`` means the second witness could not be read, so the split-frame check
        **could not be performed**.  That is a third state, not a pass: it surfaces as
        ``witness_agreement == UNOBSERVABLE`` and as the string ``"UNOBSERVABLE"`` in the
        record, never as a bare ``null`` (R12).  Pass *reason* — one of the
        ``WITNESS_REASON_*`` constants, or whatever :func:`read_counters_witness`
        returned — so the artifact says *which* absence this was.
        """
        return ExecutionAttribution(
            _token=_PARSED,
            executed_by=self._executed_by,
            node_events=self._node_events,
            source=self._source,
            profile_path=self._profile_path,
            profile_digest=self._profile_digest,
            profile_mtime_ns=self._profile_mtime_ns,
            counters_dispatches=(None if dispatches_executed is None else int(dispatches_executed)),
            counters_reason=(reason or WITNESS_REASON_UNKNOWN),
            output_coverage=self._output_coverage,
        )

    def with_output_coverage(self, coverage: "OutputAttribution") -> "ExecutionAttribution":
        """Return a copy carrying the per-output coverage reading.

        Attribution as parsed is a property of the session; this attaches the property of
        each output.  Without it, ``attributed`` answers "did this EP execute anything"
        and the verdict silently reads that as "did this EP produce the numbers we
        compared" — the same sentence at two different scopes, which is how the fifth
        costume gets through.
        """
        if not isinstance(coverage, OutputAttribution):
            raise TypeError(
                "with_output_coverage() requires an OutputAttribution computed by "
                f"OutputAttribution.from_topology(), not {type(coverage).__name__}."
            )
        return ExecutionAttribution(
            _token=_PARSED,
            executed_by=self._executed_by,
            node_events=self._node_events,
            source=self._source,
            profile_path=self._profile_path,
            profile_digest=self._profile_digest,
            profile_mtime_ns=self._profile_mtime_ns,
            counters_dispatches=self._counters_dispatches,
            counters_reason=self._counters_reason or WITNESS_REASON_NOT_REQUESTED,
            output_coverage=coverage,
        )

    @property
    def output_coverage(self) -> "OutputAttribution | None":
        return self._output_coverage

    @property
    def coverage_state(self) -> str:
        """What the per-output instrument did on this run — a reading or an absence (R12)."""
        if self._output_coverage is None:
            return OUTPUT_COVERAGE_NOT_COMPUTED
        return self._output_coverage.describe()

    @property
    def reaches_compared_outputs(self) -> bool:
        """Does anything this EP executed reach a graph output?

        ``True`` when coverage was not computed: this property may only ever *withhold*
        MATCH on a positive reading, never grant it on a missing one.  An instrument that
        was not run is not a clearance — the record says ``not-computed`` so a reader can
        see which of the two they are looking at.
        """
        if self._output_coverage is None:
            return True
        return self._output_coverage.any_output_reaches_ep

    # -- reading -------------------------------------------------------------

    @property
    def executed_by(self) -> dict[str, int]:
        return dict(self._executed_by)

    @property
    def own_count(self) -> int:
        """Fused-island executions by this EP.  R11: NOT a graph-node count."""
        return int(self._executed_by.get(EP_NAME, 0))

    @property
    def other_providers(self) -> dict[str, int]:
        return {k: v for k, v in self._executed_by.items() if k != EP_NAME}

    @property
    def attributed(self) -> bool:
        """True iff an instrument we do not own says this EP produced numbers we compared.

        Two conditions, not one, since 2026-08-02:

          1. this EP executed at least one node (session scope), **and**
          2. at least one graph output is downstream of something it executed (output
             scope), when the per-output instrument ran at all.

        Condition 2 is what stops a partially-claiming session from certifying a
        comparison whose two sides are the same computation.  It can only ever remove an
        attribution: when coverage was not computed, it is vacuously satisfied and the
        record says ``not-computed`` rather than implying a check that did not happen.
        """
        return self.own_count > 0 and self.reaches_compared_outputs

    @property
    def source(self) -> str:
        return self._source

    @property
    def counters_dispatches(self) -> int | None:
        return self._counters_dispatches

    @property
    def counters_witness_reason(self) -> str:
        """Why there is no second witness.  ``""`` when there is one."""
        return self._counters_reason

    @property
    def second_witness_observed(self) -> bool:
        """Did the second witness produce a reading on **this** run?

        This is the falsifier for "the counters witness is wired" (R10): a value it
        computed on this run, not a flag anyone set.
        """
        return self._counters_dispatches is not None

    @property
    def witnesses_present(self) -> tuple[str, ...]:
        """The witnesses that actually spoke.  A verdict may not imply more than this."""
        names = [WITNESS_ORT_PROFILE]
        if self.second_witness_observed:
            names.append(WITNESS_EP_COUNTERS)
        return tuple(names)

    @property
    def witness_agreement(self) -> str:
        """``AGREE`` / ``DISAGREE`` / ``UNOBSERVABLE`` — the split-frame check's own state.

        Three states, three tokens (R13 applied to the check rather than to the lane).
        The boolean this replaced could not distinguish "two witnesses checked and
        agreed" from "there was only one witness", and both produced ``False``, and the
        verdict read ``False`` as clearance.
        """
        if self._counters_dispatches is None:
            return WITNESS_AGREEMENT_UNOBSERVABLE
        if (self.own_count > 0) != (self._counters_dispatches > 0):
            return WITNESS_AGREEMENT_DISAGREE
        return WITNESS_AGREEMENT_AGREE

    @property
    def split_frame(self) -> bool:
        """True iff the two witnesses disagree about whether this EP ran (clause 2).

        Disagreement is *about presence*, not about magnitude: the profile counts fused
        islands and the counters count dispatches, so ``1`` and ``354`` agree perfectly.
        What may not happen is one witness saying "ran" while the other says "did not".

        Note what this predicate deliberately does **not** say: ``False`` covers both
        ``AGREE`` and ``UNOBSERVABLE``.  Read :attr:`witness_agreement` when the
        difference matters — which is any time the answer is being reported.
        """
        return self.witness_agreement == WITNESS_AGREEMENT_DISAGREE

    @property
    def witnesses(self) -> dict[str, Any]:
        observed = self.second_witness_observed
        return {
            "profile_node_events": self.own_count,
            "profile_node_events_total": self._node_events,
            "counters_dispatches_executed": (
                self._counters_dispatches if observed else WITNESS_UNOBSERVABLE
            ),
            "counters_witness_reason": self._counters_reason,
            "witness_agreement": self.witness_agreement,
            "witnesses_present": list(self.witnesses_present),
            "profile_path": self._profile_path,
            "profile_digest": self._profile_digest,
            "profile_mtime_ns": self._profile_mtime_ns,
        }

    def describe(self) -> str:
        """Render the observation with its definition attached (R11).

        Every artifact that quotes an own-count goes through this function, so there is
        exactly one wording to keep correct — and so a reader never meets a bare ``1``
        or a bare ``3`` and supplies their own frame for it.
        """
        others = ", ".join(f"{k}={v}" for k, v in sorted(self.other_providers.items())) or "none"
        if self.own_count == 0:
            return (
                f"0 fused-island executions by {EP_NAME} — this EP ran NOTHING. "
                f"Providers that did execute: {others}."
            )
        plural = "" if self.own_count == 1 else "s"
        return (
            f"{self.own_count} fused-island execution{plural} by {EP_NAME} — NOT "
            f"{self.own_count} graph node{plural}; one island can cover hundreds of graph "
            "nodes (Phi-3.5: 355 of 363 in a single island as of 2026-08-01, so 1 per run "
            "is healthy; the claimed count moves, the island count does not). "
            f"Other providers: {others}. Presence signal only — read coverage from the "
            "counters JSON, never from this number."
        )

    def assert_executed(self) -> int:
        """Guard D's assertion.  Clause 5: a convenience beside the load-bearing observation.

        Returns the own-count on success; raises ``AssertionError`` — a **finding about
        the EP**, distinct from :class:`InstrumentError` — when this EP executed nothing.
        R13 obligation 2: the failure states what it observed, so the message is a
        detection rather than an outage.
        """
        assert self.own_count > 0, (
            f"[Guard D: fallback detected] {EP_NAME} executed ZERO fused islands at run time.\n"
            f"Observed: {self.describe()}\n"
            f"Attribution source: {self._source} ({self._profile_digest}).\n"
            "\n"
            "ORT prints 'EP_FAIL ... Falling back to CPUExecutionProvider' during sess.run()\n"
            "and re-runs the entire graph on CPU without raising.  The comparison gate then\n"
            "compares CPU output against CPU output.  Under §10.0's third amendment that run\n"
            "is UNATTRIBUTED, not MATCH, and not DIVERGENT: the model was not wrong, the\n"
            "subject was absent.\n"
            "\n"
            "Route to Switch/Mouse (allocation or dispatch failure), never to the harness."
        )
        return self.own_count

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"ExecutionAttribution(own_count={self.own_count}, "
            f"executed_by={self._executed_by!r}, source={self._source!r})"
        )


_PROFILE_NAME_SUFFIXES = ("_kernel_time", "_fence_before", "_fence_after")


def strip_profile_suffix(name: str) -> str:
    """``"claimed_add_kernel_time"`` -> ``"claimed_add"``.

    ORT decorates a node's profiling event name with the phase it timed.  The undecorated
    remainder is the **graph node name**, verified empirically on 2026-08-02 against a
    five-node mixed graph: every CPU-executed node's event name stripped back to exactly
    its graph node name, on both selectors.
    """
    for suffix in _PROFILE_NAME_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def node_providers(events: list[dict]) -> dict[str, str]:
    """``{graph_node_name: provider}`` from ORT's trace.

    Malformed events are skipped, for the same reason :func:`tally_providers` skips them.
    A node this EP fused does **not** appear here under its own name: the fused island
    arrives as one event named ``VulkanExecutionProvider_VulkanExecutionProvider_<hash>_0_0``,
    which names no constituent.  That asymmetry is the whole reason the coverage below is
    derived by complement and is sound only in one direction.
    """
    out: dict[str, str] = {}
    for ev in events:
        if not isinstance(ev, dict) or ev.get("cat") != "Node":
            continue
        args = ev.get("args")
        if not isinstance(args, dict):
            continue
        provider = args.get("provider")
        name = ev.get("name")
        if isinstance(provider, str) and provider and isinstance(name, str) and name:
            out[strip_profile_suffix(name)] = provider
    return out


class OutputAttribution:
    """Which provider's work reaches each **graph output**.  The fifth costume's instrument.

    Built by :meth:`from_topology` from two things, *neither of which this project owns*:

      * the ONNX artifact's own topology — producer edges read out of the model file;
      * ORT's profiling trace — which node names ran on which provider.

    Our claim log is deliberately not consulted.  It lives inside the frame whose
    existence is in question, and two witnesses that can only ever agree are one witness.

    **Soundness is one-directional, and the direction is the safe one.**
    ``CPU-ONLY`` is a *refusal*: it is returned only when every node upstream of the
    output carries an explicit other-provider event, and an optimiser can delete an
    event but cannot invent one, so a wrongly-``CPU-ONLY`` output is not reachable by
    node elimination.  ``EP-COVERED`` is the weaker inference — "some ancestor has no
    other-provider event, so it was absorbed into a fused island *or* eliminated before
    execution" — and it is therefore never on its own sufficient for anything.  It is
    used only to *withhold* MATCH, never to grant it beyond what the session-level
    attribution already granted.
    """

    __slots__ = ("_per_output", "_reasons", "_ep_name", "_node_provider_count", "_claim_log_join")

    def __init__(
        self,
        *,
        _token: object = None,
        per_output: Mapping[str, str],
        reasons: Mapping[str, str],
        ep_name: str,
        node_provider_count: int,
        claim_log_join: int = -1,
    ) -> None:
        if _token is not _PARSED:
            raise TypeError(
                "OutputAttribution() is private.  R10 amendment 1: a coverage reading must "
                "be a value a mechanism computed on this run, never a literal its author "
                "set.  Use OutputAttribution.from_topology(topology=..., "
                "node_providers=node_providers(events))."
            )
        self._per_output = dict(per_output)
        self._reasons = dict(reasons)
        self._ep_name = ep_name
        self._node_provider_count = int(node_provider_count)
        self._claim_log_join = int(claim_log_join)

    @classmethod
    def from_topology(
        cls,
        *,
        topology: Mapping[str, Any],
        node_providers: Mapping[str, str],
        claimed_nodes: "set[str] | frozenset[str] | None" = None,
        ep_name: str = EP_NAME,
    ) -> "OutputAttribution":
        """Label every graph output from the artifact's edges and ORT's trace.

        *topology* is the pure-data shape produced by ``_models.graph_topology()``::

            {"outputs": [name, ...],
             "producer": {value_name: node_name},
             "node_inputs": {node_name: [value_name, ...]}}

        *claimed_nodes* is optional and, when given, is the set of graph node names **this
        EP claimed**, read out of our own claim log.  It is our instrument, living inside
        the frame whose existence is in question — which would disqualify it from granting
        anything.  It is used only in the direction where it *accuses us*: an ancestor we
        did not claim is not ours, whatever the trace omits.  That can only move outputs
        from ``EP-COVERED`` to ``CPU-ONLY``, i.e. only ever withhold ``MATCH``, so a
        lying claim log cannot manufacture a pass.  Measured on Phi-3.5 (dev0,
        2026-08-02) the trace alone labelled 65/65 outputs ``EP-COVERED`` at an own-count
        of **zero**, because ORT's own graph optimisers delete node events wholesale; the
        complement is nearly uninformative on a real model without this second source.

        Raises
        ------
        InstrumentError
            *topology* is not that shape, the trace named no nodes at all, or
            *claimed_nodes* was supplied and joins against no node in the graph.  All
            three are ``ERROR(instrument)``: the reading was never reached, so there is
            nothing to report about the EP (R13).  The last matters most — a claim log
            whose node names do not join (ORT renames nodes before ``GetCapability``)
            would otherwise mark every output ``CPU-ONLY`` and manufacture a false red.
        """
        try:
            outputs = list(topology["outputs"])
            producer = dict(topology["producer"])
            node_inputs = {k: list(v) for k, v in dict(topology["node_inputs"]).items()}
        except (KeyError, TypeError, ValueError) as exc:
            raise InstrumentError(
                "[coverage instrument failure] topology is not "
                "{'outputs': [...], 'producer': {...}, 'node_inputs': {...}}: "
                f"{type(exc).__name__}: {exc}.  This is an instrument outage, NOT a "
                "finding about the EP (R13)."
            ) from exc

        if not node_providers:
            raise InstrumentError(
                "[coverage instrument failure] ORT's trace named zero nodes, so no output "
                "can be labelled.  Profiling may not have been enabled before session "
                "creation.  UNOBSERVABLE for every output would read as a coverage "
                "finding; it is an outage (R13)."
            )

        all_nodes = set(node_inputs)
        join = -1
        if claimed_nodes is not None:
            join = len(set(claimed_nodes) & all_nodes)
            if claimed_nodes and join == 0:
                raise InstrumentError(
                    "[coverage instrument failure] the claim log named "
                    f"{len(set(claimed_nodes))} claimed node(s), none of which is a node "
                    f"name in this graph ({len(all_nodes)} nodes).  ORT may rename nodes "
                    "before GetCapability, so the join is broken and every output would "
                    "be labelled CPU-ONLY — a manufactured red.  ERROR(instrument) (R13)."
                )

        other = {n for n, p in node_providers.items() if p != ep_name}
        per_output: dict[str, str] = {}
        reasons: dict[str, str] = {}
        for out_name in outputs:
            ancestors = _ancestor_nodes(out_name, producer, node_inputs)
            if not ancestors:
                per_output[out_name] = OUTPUT_UNOBSERVABLE
                reasons[out_name] = (
                    "no node in this graph produces this output (it is a graph input, an "
                    "initializer, or passed through); there is no upstream to attribute"
                )
                continue
            ours = ancestors - other
            if claimed_nodes is not None:
                ours = ours & set(claimed_nodes)
            unnamed = sorted(ours)
            if not unnamed:
                per_output[out_name] = OUTPUT_CPU_ONLY
                reasons[out_name] = (
                    f"none of the {len(ancestors)} upstream node(s) is ours — each either "
                    "carries an explicit other-provider event"
                    + (
                        " or was not claimed by this EP"
                        if claimed_nodes is not None
                        else ""
                    )
                    + "; nothing this EP ran reaches this output, so its oracle comparison "
                    "is CPU-against-CPU"
                )
            else:
                per_output[out_name] = OUTPUT_EP_COVERED
                reasons[out_name] = (
                    f"{len(unnamed)} of {len(ancestors)} upstream node(s) may be ours "
                    f"({', '.join(unnamed[:6])}{', ...' if len(unnamed) > 6 else ''}): no "
                    "other-provider event"
                    + (
                        " and claimed by this EP"
                        if claimed_nodes is not None
                        else ", so absorbed into a fused island or eliminated before "
                        "execution"
                    )
                    + " — this label withholds MATCH, it does not grant it"
                )
        return cls(
            _token=_PARSED,
            per_output=per_output,
            reasons=reasons,
            ep_name=ep_name,
            node_provider_count=len(node_providers),
            claim_log_join=join,
        )

    # -- reading -------------------------------------------------------------

    @property
    def per_output(self) -> dict[str, str]:
        return dict(self._per_output)

    @property
    def output_names(self) -> tuple[str, ...]:
        return tuple(self._per_output)

    def token_for(self, name_or_index: "str | int") -> str:
        if isinstance(name_or_index, int):
            names = self.output_names
            if not 0 <= name_or_index < len(names):
                return OUTPUT_UNOBSERVABLE
            return self._per_output[names[name_or_index]]
        return self._per_output.get(name_or_index, OUTPUT_UNOBSERVABLE)

    def reason_for(self, name: str) -> str:
        return self._reasons.get(name, "not labelled")

    def count(self, token: str) -> int:
        return sum(1 for v in self._per_output.values() if v == token)

    @property
    def ep_covered_count(self) -> int:
        return self.count(OUTPUT_EP_COVERED)

    @property
    def cpu_only_count(self) -> int:
        return self.count(OUTPUT_CPU_ONLY)

    @property
    def unobservable_count(self) -> int:
        return self.count(OUTPUT_UNOBSERVABLE)

    @property
    def cpu_only_names(self) -> tuple[str, ...]:
        return tuple(n for n, v in self._per_output.items() if v == OUTPUT_CPU_ONLY)

    @property
    def any_output_reaches_ep(self) -> bool:
        """Does *any* graph output depend on work this EP did?

        ``False`` beside a positive own-count is the state this class was written for:
        the EP ran, and every compared output is CPU-against-CPU anyway.
        """
        return self.ep_covered_count > 0

    @property
    def partial(self) -> bool:
        """Some outputs reach this EP and some do not — the intermediate-ledger state."""
        return self.ep_covered_count > 0 and self.cpu_only_count > 0

    def refuted_by(self, disagreeing_output_names: "list[str] | tuple[str, ...]") -> list[str]:
        """The falsifier for *this instrument* (R9, R10).

        A ``CPU-ONLY`` output has the same computation on both sides of the oracle, so it
        must agree bit-for-bit.  If the oracle says it disagreed, then this labelling is
        wrong — the output was not CPU-only after all, or the topology was misread.  The
        returned names are a finding about the **coverage instrument**, never about the
        EP, and the caller is expected to say so.
        """
        return [
            n for n in disagreeing_output_names
            if self._per_output.get(n) == OUTPUT_CPU_ONLY
        ]

    def describe(self) -> str:
        return (
            f"{self.ep_covered_count} output(s) reach {self._ep_name}, "
            f"{self.cpu_only_count} are CPU-only (their oracle comparison is vacuous), "
            f"{self.unobservable_count} unobservable; over "
            f"{len(self._per_output)} graph output(s), from {self._node_provider_count} "
            f"named node event(s) and {self.claim_log_state}"
        )

    @property
    def claim_log_join(self) -> int:
        """How many claimed node names joined the graph.  ``-1`` = no claim log used."""
        return self._claim_log_join

    @property
    def claim_log_state(self) -> str:
        if self._claim_log_join < 0:
            return "no claim log (trace complement only — weak on real models)"
        return f"a claim log joining {self._claim_log_join} node(s)"

    def to_record(self) -> dict[str, Any]:
        return {
            "per_output": dict(self._per_output),
            "reasons": dict(self._reasons),
            "ep_covered": self.ep_covered_count,
            "cpu_only": self.cpu_only_count,
            "unobservable": self.unobservable_count,
            "outputs_total": len(self._per_output),
            "partial": self.partial,
            "claim_log_join": (
                OUTPUT_COVERAGE_NOT_COMPUTED
                if self._claim_log_join < 0
                else self._claim_log_join
            ),
            "means": (
                "CPU-ONLY is a refusal and is sound: no upstream node is ours. EP-COVERED "
                "is the weaker inference and only ever withholds MATCH, never grants it. "
                "Without a claim log the EP-COVERED side is nearly uninformative on a real "
                "model, because ORT's own optimisers delete node events wholesale."
            ),
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"OutputAttribution(ep_covered={self.ep_covered_count}, "
            f"cpu_only={self.cpu_only_count}, unobservable={self.unobservable_count})"
        )


def _ancestor_nodes(
    output_name: str,
    producer: Mapping[str, str],
    node_inputs: Mapping[str, list[str]],
) -> set[str]:
    """Every graph node upstream of *output_name*, inclusive of its producer."""
    seen: set[str] = set()
    stack = [output_name]
    while stack:
        value = stack.pop()
        node = producer.get(value)
        if node is None or node in seen:
            continue
        seen.add(node)
        stack.extend(node_inputs.get(node, ()))
    return seen


def tally_providers(events: list[dict]) -> dict[str, int]:
    """Count ``cat == "Node"`` profiling events per ``args["provider"]``.

    Malformed events are skipped rather than raising: a trace with one bad event is still
    a trace, and the alternative — an instrument that dies on unfamiliar input — is the
    R13 outage this module exists to make impossible to confuse with a finding.
    """
    tally: dict[str, int] = {}
    for ev in events:
        if not isinstance(ev, dict) or ev.get("cat") != "Node":
            continue
        args = ev.get("args")
        if not isinstance(args, dict):
            continue
        provider = args.get("provider")
        if isinstance(provider, str) and provider:
            tally[provider] = tally.get(provider, 0) + 1
    return tally


# ---------------------------------------------------------------------------
# The verdict — a record, derived, never chosen
# ---------------------------------------------------------------------------

class EquivalenceVerdict:
    """The `model_output_equivalence` record of §10.0's third amendment.

    Construct with :meth:`from_comparison` (requires an attribution) or
    :meth:`unmeasured` (asserts nothing about execution).  ``verdict`` is a **derived**
    read-only property: there is no code path in this project that sets it.
    """

    __slots__ = ("_comparison", "_attribution", "_artifact", "_device_index", "_device_name", "_detail")

    def __init__(
        self,
        *,
        _token: object = None,
        comparison: str,
        attribution: ExecutionAttribution | None,
        artifact: str,
        device_index: str,
        device_name: str,
        detail: str,
    ) -> None:
        if _token is not _PARSED:
            raise TypeError(
                "EquivalenceVerdict() is private.  Use EquivalenceVerdict.from_comparison("
                "comparison=..., attribution=ExecutionAttribution.from_profile(...), ...) or "
                "EquivalenceVerdict.unmeasured(reason=...).  §10.0 clause 3: the verdict is "
                "derived from an attribution a mechanism computed, never supplied by a caller."
            )
        self._comparison = comparison
        self._attribution = attribution
        self._artifact = artifact
        self._device_index = device_index
        self._device_name = device_name
        self._detail = detail

    # -- construction --------------------------------------------------------

    @classmethod
    def from_comparison(
        cls,
        *,
        comparison: str,
        attribution: ExecutionAttribution,
        artifact: str,
        device_index: str = "",
        device_name: str = "",
        detail: str = "",
    ) -> "EquivalenceVerdict":
        """Derive a verdict from a comparison outcome and an execution attribution.

        Parameters
        ----------
        comparison:
            One of :data:`COMPARISON_AGREE`, :data:`COMPARISON_DISAGREE`,
            :data:`COMPARISON_NOT_PERFORMED`.  **Passing a verdict token — including
            ``"MATCH"`` — is a ``ValueError``.**  A comparison of tensors can say that
            they agree; only a comparison *plus an attribution* can say ``MATCH``.
        attribution:
            An :class:`ExecutionAttribution` obtained from
            :meth:`ExecutionAttribution.from_profile`.  Anything else is a ``TypeError``.

        Raises
        ------
        TypeError
            *attribution* is not an :class:`ExecutionAttribution`.  This is what makes
            ``MATCH`` unrepresentable at zero own-provider count: there is no way to
            reach the ``MATCH`` branch without a parsed profile in hand.
        ValueError
            *comparison* is not a comparison outcome.
        """
        if not isinstance(attribution, ExecutionAttribution):
            raise TypeError(
                "EquivalenceVerdict.from_comparison(attribution=...) requires an "
                "ExecutionAttribution parsed from this run's ORT profile, not "
                f"{type(attribution).__name__}.  §10.0 clause 3: 'the function that writes "
                "the verdict takes the parsed attribution as a required argument derived "
                "from a profile path ... a caller may not pass a literal.'  If no profile "
                "was taken, the honest verdict is UNMEASURED — use "
                "EquivalenceVerdict.unmeasured(reason=...)."
            )
        if comparison in VERDICTS:
            raise ValueError(
                f"{comparison!r} is a *verdict*, not a comparison outcome.  Verdicts are "
                f"derived here, never supplied: pass one of {COMPARISONS} describing what "
                "the tensor comparison found, and this constructor will decide whether the "
                "run earned MATCH.  (§10.0 third amendment, clause 3.)"
            )
        if comparison not in COMPARISONS:
            raise ValueError(
                f"comparison must be one of {COMPARISONS}, got {comparison!r}."
            )
        return cls(
            _token=_PARSED,
            comparison=comparison,
            attribution=attribution,
            artifact=artifact,
            device_index=str(device_index),
            device_name=str(device_name),
            detail=detail,
        )

    @classmethod
    def unmeasured(
        cls,
        *,
        reason: str,
        artifact: str = "",
        device_index: str = "",
        device_name: str = "",
    ) -> "EquivalenceVerdict":
        """The honest default: no comparison was performed, and nothing is claimed.

        Takes no attribution because it asserts nothing about execution.  ``executed_by``
        is emitted as ``{}`` with ``attribution_source: "none"`` — an absence a reader can
        see, rather than a zero they must interpret (R7/R12).
        """
        return cls(
            _token=_PARSED,
            comparison=COMPARISON_NOT_PERFORMED,
            attribution=None,
            artifact=artifact,
            device_index=str(device_index),
            device_name=str(device_name),
            detail=reason,
        )

    # -- derivation ----------------------------------------------------------

    @property
    def verdict(self) -> str:
        """The derived token.  Read-only, by construction: nothing assigns this.

        Precedence — see the module docstring for why it is this order.
        """
        att = self._attribution
        if att is not None and att.split_frame:
            return VERDICT_SPLIT_FRAME
        if att is None or self._comparison == COMPARISON_NOT_PERFORMED:
            return VERDICT_UNMEASURED
        if not att.attributed:
            # Clause 4: this is NOT DIVERGENT even when the comparison disagreed.  The
            # comparison is arithmetically correct and about a different world (R12).
            return VERDICT_UNATTRIBUTED
        if self._comparison == COMPARISON_AGREE:
            return VERDICT_MATCH
        return VERDICT_DIVERGENT

    @property
    def comparison(self) -> str:
        return self._comparison

    @property
    def attribution(self) -> ExecutionAttribution | None:
        return self._attribution

    @property
    def executed_by(self) -> dict[str, int]:
        return {} if self._attribution is None else self._attribution.executed_by

    @property
    def permits_report(self) -> bool:
        """May the triple, the wall-clock ratio, or a correctness claim be reported?"""
        return self.verdict in VERDICTS_PERMITTING_REPORT

    @property
    def voids_triple(self) -> bool:
        return not self.permits_report

    def to_record(self) -> dict[str, Any]:
        """The JSON object written beside the verdict token (§10.0 third amendment)."""
        att = self._attribution
        return {
            "verdict": self.verdict,
            "comparison": self._comparison,
            "executed_by": self.executed_by,
            "attribution_source": ATTRIBUTION_SOURCE_NONE if att is None else att.source,
            "attribution_witnesses": {} if att is None else att.witnesses,
            "attribution_witnesses_present": (
                [] if att is None else list(att.witnesses_present)
            ),
            "attribution_witness_agreement": (
                WITNESS_AGREEMENT_UNOBSERVABLE if att is None else att.witness_agreement
            ),
            "own_provider_execution_count": 0 if att is None else att.own_count,
            "own_provider_count_means": (
                "fused-island executions by this EP, NOT graph nodes; one island can cover "
                "hundreds of graph nodes"
            ),
            "output_coverage": (
                OUTPUT_COVERAGE_NOT_COMPUTED
                if att is None or att.output_coverage is None
                else att.output_coverage.to_record()
            ),
            "outputs_reaching_this_ep": (
                OUTPUT_COVERAGE_NOT_COMPUTED
                if att is None or att.output_coverage is None
                else att.output_coverage.ep_covered_count
            ),
            "outputs_cpu_only": (
                OUTPUT_COVERAGE_NOT_COMPUTED
                if att is None or att.output_coverage is None
                else att.output_coverage.cpu_only_count
            ),
            "artifact": self._artifact,
            "device_index": self._device_index,
            "device_name": self._device_name,
            "permits_triple_and_ratio": self.permits_report,
            "detail": self._detail,
        }

    def explain(self) -> str:
        """One paragraph a human can act on, naming the owner of each red state."""
        v = self.verdict
        head = f"model_output_equivalence = {v}"
        att = self._attribution
        frame = "no attribution taken" if att is None else att.describe()
        if v == VERDICT_MATCH:
            witnessed = "no attribution taken" if att is None else ", ".join(att.witnesses_present)
            if att is not None and not att.second_witness_observed:
                return (
                    f"{head} — outputs agree and the run is attributed. {frame} "
                    f"Witnesses that spoke: {witnessed}. The split-frame check is "
                    f"UNOBSERVABLE on this run ({att.counters_witness_reason}), so this "
                    "MATCH rests on ONE instrument. It is not a two-witness MATCH and "
                    "must not be quoted as one."
                )
            return (
                f"{head} — outputs agree and the run is attributed. {frame} "
                f"Witnesses that spoke: {witnessed}; they agree about presence."
            )
        if v == VERDICT_DIVERGENT:
            return (
                f"{head} — this EP executed and an output disagrees with the CPU oracle. "
                f"Our kernels computed the wrong answer. {frame} "
                "The triple and the wall-clock ratio may not be reported. Owner: Mouse/Switch."
            )
        if v == VERDICT_UNATTRIBUTED:
            if att is not None and att.own_count > 0:
                cov = att.output_coverage
                return (
                    f"{head} — this EP executed "
                    f"{att.own_count} fused island(s) and **not one graph output is "
                    "downstream of anything it executed**. Every comparison in this run "
                    "was our-CPU against ORT's-CPU: two sides of one computation, which "
                    "no degeneracy guard can see because both sides are real, varying, "
                    f"input-dependent values. {'' if cov is None else cov.describe() + '. '}"
                    "This is NOT DIVERGENT and NOT a partial success. Owner: whoever owns "
                    "the claim predicates for the ops upstream of the outputs (Mouse)."
                )
            return (
                f"{head} — the comparison ran and this EP did not run. This is NOT "
                "DIVERGENT: the model was not wrong, the subject was absent, and the "
                f"comparison was CPU-vs-CPU. {frame} The triple and the wall-clock ratio "
                "may not be reported. Owner: whoever owns the run-time fallback (Switch), "
                "not the kernel authors."
            )
        if v == VERDICT_SPLIT_FRAME:
            return (
                f"{head} — the two witnesses disagree about whether this EP ran: profile "
                f"says {0 if att is None else att.own_count} island execution(s), counters "
                f"say {None if att is None else att.counters_dispatches} dispatch(es). One "
                "of the two instruments is lying and we do not yet know which. Nothing may "
                "be reported. Owner: Trinity (parse) with Switch (counters)."
            )
        return (
            f"{head} — no CPU-only comparison was performed on this artifact in this run. "
            f"{self._detail} This is the default and it is not a soft MATCH."
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"EquivalenceVerdict({self.verdict}, comparison={self._comparison!r})"


# ---------------------------------------------------------------------------
# Writing the record into the counters artifact
# ---------------------------------------------------------------------------

def read_counters_witness(
    counters_path: "str | os.PathLike[str] | None",
) -> tuple[int | None, str]:
    """Read the second witness, returning ``(dispatches_executed, reason)``.

    ``reason`` is ``""`` on success and one of the ``WITNESS_REASON_*`` sentences
    otherwise.  This exists because :func:`read_counters_dispatches` collapsed five
    distinct causes — never armed, file absent, unopenable, not JSON, field missing —
    into one bare ``None``, and an artifact carrying that ``None`` could not tell a
    reader which of the five had happened, nor whether anyone had ever intended the
    witness to be read at all.
    """
    if not counters_path:
        return None, WITNESS_REASON_NOT_ARMED
    if not os.path.exists(str(counters_path)):
        return None, f"{WITNESS_REASON_FILE_ABSENT} (path: {counters_path})"
    try:
        with open(counters_path, encoding="utf-8") as fh:
            raw = fh.read()
    except OSError as exc:
        return None, f"{WITNESS_REASON_FILE_UNREADABLE} (path: {counters_path}, {exc})"
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"{WITNESS_REASON_NOT_JSON} (path: {counters_path}, {exc})"
    value = doc.get("dispatches_executed") if isinstance(doc, dict) else None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None, f"{WITNESS_REASON_FIELD_MISSING} (path: {counters_path})"
    return int(value), ""


def read_counters_dispatches(counters_path: "str | os.PathLike[str] | None") -> int | None:
    """Read ``dispatches_executed`` from the counters JSON — the second witness.

    Returns ``None`` when the witness could not be read.  **A missing witness is not an
    agreeing witness** — but note that this function cannot tell a caller *why* it is
    missing, which is precisely how a ``null`` came to sit inside a passing artifact.
    Prefer :func:`read_counters_witness`, which returns the reason alongside the value,
    and pass that reason to :meth:`ExecutionAttribution.with_counters_witness`.

    Kept for callers outside ``tests/`` that only want the number.
    """
    return read_counters_witness(counters_path)[0]


def write_equivalence_record(
    counters_path: "str | os.PathLike[str]",
    verdict: EquivalenceVerdict,
) -> str:
    """Write *verdict* into the counters JSON at *counters_path*; return the token written.

    Two keys are written, and both are load-bearing:

    ``model_output_equivalence``
        the token — a **string**, because ``rust/src/counters.rs::extract_equivalence``
        and ``epctl --check-counters`` parse it as one.  Changing its shape would break
        the gate, so it keeps its shape and gains two new values.
    ``model_output_equivalence_record``
        the full object of §10.0's third amendment: ``executed_by``,
        ``attribution_source``, ``attribution_witnesses``, ``artifact``, device identity.
        *A caveat that lives in a different artifact from the number it qualifies is not
        attached to it* — this is how the attribution travels to ``epctl``, ``bench/`` and
        the census instead of living only in a pytest caveat nobody downstream reads.

    Only an :class:`EquivalenceVerdict` is accepted.  Passing the string ``"MATCH"``
    raises ``TypeError``: the whole point of the amendment is that the value is one a
    mechanism computed on this run.

    Raises
    ------
    TypeError
        *verdict* is not an :class:`EquivalenceVerdict`.
    InstrumentError
        The counters file is missing or is not JSON we can splice.  This is a harness
        outage (R13) — it says nothing about the EP, so it must not be mistaken for one.
    """
    if not isinstance(verdict, EquivalenceVerdict):
        raise TypeError(
            "write_equivalence_record() takes an EquivalenceVerdict, not "
            f"{type(verdict).__name__}.  §10.0 third amendment clause 3: 'a caller may not "
            "pass a literal.'  Build one with EquivalenceVerdict.from_comparison("
            "comparison=..., attribution=ExecutionAttribution.from_profile(profile_path), "
            "...) or EquivalenceVerdict.unmeasured(reason=...)."
        )

    path = str(counters_path)
    if not os.path.exists(path):
        raise InstrumentError(
            f"write_equivalence_record: counters file not found: {path}\n"
            "Is ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE set?  Did the EP write a snapshot?\n"
            "This is an instrument outage, NOT a finding about the EP (R13)."
        )

    with open(path, encoding="utf-8") as fh:
        doc = fh.read()

    token = verdict.verdict
    record_json = json.dumps(verdict.to_record(), indent=2, sort_keys=True)
    # Re-indent the nested object so the hand-rolled counters JSON stays readable.
    record_json = "\n".join(
        ("  " + line if i else line) for i, line in enumerate(record_json.splitlines())
    )

    pattern = rf'("{re.escape(EQUIVALENCE_KEY)}")\s*:\s*"[^"]*"'
    if re.search(pattern, doc):
        doc = re.sub(pattern, lambda mo: f'{mo.group(1)}: "{token}"', doc, count=1)
    else:
        cut = doc.rfind("}")
        if cut == -1:
            raise InstrumentError(
                f"write_equivalence_record: {path} does not look like JSON (no '}}').\n"
                "This is an instrument outage, NOT a finding about the EP (R13)."
            )
        doc = doc[:cut].rstrip().rstrip(",") + f',\n  "{EQUIVALENCE_KEY}": "{token}"\n}}\n'

    # Splice (or replace) the record object.
    record_pattern = rf'\s*,?\s*"{re.escape(EQUIVALENCE_RECORD_KEY)}"\s*:\s*\{{.*?\n  \}}'
    doc = re.sub(record_pattern, "", doc, flags=re.DOTALL)
    cut = doc.rfind("}")
    if cut == -1:
        raise InstrumentError(
            f"write_equivalence_record: {path} does not look like JSON (no '}}').\n"
            "This is an instrument outage, NOT a finding about the EP (R13)."
        )
    doc = (
        doc[:cut].rstrip().rstrip(",")
        + f',\n  "{EQUIVALENCE_RECORD_KEY}": {record_json}\n}}\n'
    )

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(doc)
    return token


def read_equivalence_record(counters_path: "str | os.PathLike[str]") -> dict[str, Any]:
    """Read back the record object, for the census and for ``bench/``.

    Returns ``{}`` when absent.  Callers must treat ``{}`` as *no frame was recorded* —
    which, per criterion 12(g), means the census line reports a verdict from a world it
    has not identified and must not be read as an observation.
    """
    try:
        with open(counters_path, encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}
    record = doc.get(EQUIVALENCE_RECORD_KEY)
    return record if isinstance(record, dict) else {}


# ---------------------------------------------------------------------------
# The three-consecutive-attributed-MATCH series — M0 criterion 10
# ---------------------------------------------------------------------------

class AttributedRunSeries:
    """N consecutive runs of one session, each compared and all attributed.

    M0 criterion 10 closes on *three consecutive attributed ``MATCH`` runs* in one session
    on one artifact.  Before this class that record existed only as a throwaway script,
    and per R10 a record that is not in the lane does not exist.

    THE COUNTING TRAP, STATED WHERE THE NUMBER IS PRODUCED
    ======================================================
    ORT emits one ``Node`` profiling event per **fused-island execution**.  Phi-3.5's
    Vulkan partition is a single island covering 355 of 363 graph nodes (2026-08-01;
    the claimed count rises as Mouse claims ops, the island count has not moved), so:

        three runs  ->  3 VulkanExecutionProvider node events  ->  ``own_count == 3``

    Three is the number of *island executions*, i.e. one per run — **not** three graph
    nodes, and not a partial claim.  A future reader who meets a bare ``3`` beside a
    350-plus-node model will read a catastrophe, which is why every emitter of this number
    goes through :meth:`describe` and never prints it bare (R11).

    Attribution is taken **once per session**, because ORT writes one profile per session
    and ``end_profiling()`` cannot be restarted.  The series therefore requires:

      - ``own_count >= runs``            — every run executed at least one island, and
      - ``own_count % runs == 0``        — the per-run island count is uniform, so no run
                                           can have been silently dropped to CPU while
                                           another ran twice as many islands.

    Those two conditions together are what "consecutive" buys us from a session-scoped
    instrument.  A run that fell back mid-series lowers ``own_count`` below ``runs``.
    """

    __slots__ = ("_runs", "_comparisons", "_attribution", "_artifact", "_device_index", "_device_name")

    def __init__(
        self,
        *,
        _token: object = None,
        comparisons: list[str],
        attribution: ExecutionAttribution,
        artifact: str,
        device_index: str,
        device_name: str,
    ) -> None:
        if _token is not _PARSED:
            raise TypeError(
                "AttributedRunSeries() is private — use AttributedRunSeries.from_runs()."
            )
        self._runs = len(comparisons)
        self._comparisons = list(comparisons)
        self._attribution = attribution
        self._artifact = artifact
        self._device_index = str(device_index)
        self._device_name = str(device_name)

    @classmethod
    def from_runs(
        cls,
        *,
        comparisons: list[str],
        attribution: ExecutionAttribution,
        artifact: str,
        device_index: str = "",
        device_name: str = "",
    ) -> "AttributedRunSeries":
        if not isinstance(attribution, ExecutionAttribution):
            raise TypeError(
                "AttributedRunSeries.from_runs(attribution=...) requires an "
                "ExecutionAttribution parsed from this session's ORT profile."
            )
        if not comparisons:
            raise ValueError("a series needs at least one run")
        bad = [c for c in comparisons if c not in COMPARISONS]
        if bad:
            raise ValueError(f"not comparison outcomes: {bad!r}; expected one of {COMPARISONS}")
        return cls(
            _token=_PARSED,
            comparisons=comparisons,
            attribution=attribution,
            artifact=artifact,
            device_index=device_index,
            device_name=device_name,
        )

    @property
    def runs(self) -> int:
        return self._runs

    @property
    def own_count(self) -> int:
        return self._attribution.own_count

    @property
    def islands_per_run(self) -> float:
        return self.own_count / self._runs if self._runs else 0.0

    @property
    def uniformly_attributed(self) -> bool:
        return (
            self.own_count >= self._runs
            and self._runs > 0
            and self.own_count % self._runs == 0
        )

    @property
    def verdict(self) -> str:
        """The series verdict, derived exactly as a single run's is.

        The comparison outcome for the series is ``AGREE`` only when **every** run agreed;
        one disagreement makes the series ``DISAGREE``.  Attribution is then applied by
        :class:`EquivalenceVerdict`, so a series in which the EP ran nothing is
        ``UNATTRIBUTED`` and never ``MATCH`` — the same impossibility, one level up.
        """
        return self.as_verdict().verdict

    def as_verdict(self) -> EquivalenceVerdict:
        agreed = all(c == COMPARISON_AGREE for c in self._comparisons)
        any_not_performed = any(c == COMPARISON_NOT_PERFORMED for c in self._comparisons)
        if any_not_performed:
            comparison = COMPARISON_NOT_PERFORMED
        else:
            comparison = COMPARISON_AGREE if agreed else COMPARISON_DISAGREE
        return EquivalenceVerdict.from_comparison(
            comparison=comparison,
            attribution=self._attribution,
            artifact=self._artifact,
            device_index=self._device_index,
            device_name=self._device_name,
            detail=(
                f"{self._runs}-run series, per-run comparisons {self._comparisons}, "
                f"{self.describe()}"
            ),
        )

    def describe(self) -> str:
        return (
            f"{self.own_count} {EP_NAME} fused-island execution(s) across {self._runs} "
            f"consecutive run(s) of one session = {self.islands_per_run:g} island(s) per run. "
            f"This counts ISLAND EXECUTIONS, not graph nodes: Phi-3.5 fuses 355 of 363 graph "
            f"nodes into one island, so {self._runs} run(s) of a healthy session report "
            f"exactly {self._runs}. A reader who takes this for a graph-node count will read "
            "a catastrophe where the record says success."
        )

    def to_record(self) -> dict[str, Any]:
        rec = self.as_verdict().to_record()
        rec["series"] = {
            "runs": self._runs,
            "per_run_comparison": self._comparisons,
            "own_provider_execution_count": self.own_count,
            "islands_per_run": self.islands_per_run,
            "uniformly_attributed": self.uniformly_attributed,
            "counts_what": "fused-island executions, not graph nodes",
        }
        return rec

    def assert_closes_criterion_10(
        self,
        *,
        required_runs: int = 3,
        require_second_witness: bool = True,
    ) -> None:
        """Assert the M0 criterion-10 condition, quoting what was observed either way.

        ``AssertionError`` here is a **finding**; every path that could fail without an
        observation raises :class:`InstrumentError` upstream instead (R13).

        ``require_second_witness`` (default **True**, round 26): the canonical M0 record
        may not be produced by a run in which the split-frame check was never performable.
        Failing to arm the counters is a harness outage, so it raises
        :class:`InstrumentError` — ``ERROR(instrument)``, never a detection about the EP.
        """
        v = self.as_verdict()
        att = self._attribution
        assert self._runs >= required_runs, (
            f"[criterion 10] series has {self._runs} run(s), needs {required_runs}. "
            f"Observed: {self.describe()}"
        )
        # Verdict before uniformity: UNATTRIBUTED is the more informative red and must not
        # be reported as a uniformity complaint.  Clause 4 — the two are never folded.
        assert v.verdict == VERDICT_MATCH, (
            f"[criterion 10] series verdict is {v.verdict}, not MATCH.\n{v.explain()}\n"
            f"Per-run comparisons: {self._comparisons}"
        )
        assert self.uniformly_attributed, (
            f"[criterion 10] attribution is not uniform across the series. "
            f"Observed: {self.describe()} — expected own_count to be a positive multiple of "
            f"{self._runs}. A run that fell back to CPU mid-series shows up here as a "
            "non-multiple or as a count below the run count."
        )
        # LAST, deliberately.  A genuine red above is a finding and must never be masked by
        # an instrument complaint (R13); but a *green* may not be reported by a run in which
        # the split-frame check was never performable.  This is the only ordering in which
        # both hold.
        if require_second_witness and (att is None or not att.second_witness_observed):
            reason = WITNESS_REASON_NOT_REQUESTED if att is None else att.counters_witness_reason
            spoke = "none" if att is None else ", ".join(att.witnesses_present)
            raise InstrumentError(
                "[criterion 10 instrument outage] the second attribution witness was never "
                "read, so the split-frame check was UNOBSERVABLE on this run.\n"
                f"Reason: {reason}\n"
                f"Witnesses that spoke: {spoke}\n"
                "\n"
                "The comparison outcome is NOT in question and this is NOT a finding about "
                "the EP (R13).  What is missing is the check that the profile and the "
                "counters agree the EP ran -- with only one witness, an artifact recording "
                "MATCH is indistinguishable from one in which the counters witness was "
                "never wired at all (R10).\n"
                "\n"
                "Arm it by exporting ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE BEFORE the process "
                "starts (the DLL caches the value at load time on Windows), then re-run.",
                observed=None if att is None else att.witnesses,
            )


# ---------------------------------------------------------------------------
# R13 — the lane's known-fatal log line, as a second witness with a different
# failure mode (obligation 3).  A grep cannot NameError.
# ---------------------------------------------------------------------------

#: Substrings that mean ORT abandoned this EP at run time and re-ran on CPU.
#: A `grep` covers Guard D's outage and Guard D covers a log-format change; each is the
#: other's second witness, which is the point (R13 obligation 3).
FATAL_LOG_MARKERS: tuple[str, ...] = (
    "Falling back to CPUExecutionProvider",
    "Falling back to CPU",
)


def find_fatal_log_lines(text: str) -> list[str]:
    """Return the captured lines that announce a run-time fallback.

    Pure and total: it cannot raise on any string, which is exactly the property that
    makes it usable as the second witness for a guard that can crash.
    """
    if not text:
        return []
    hits: list[str] = []
    for line in text.splitlines():
        if any(marker in line for marker in FATAL_LOG_MARKERS):
            hits.append(line.strip())
    return hits


# ---------------------------------------------------------------------------
# R13 applied to SUBPROCESSES — a timeout is an outage, never a detection
# ---------------------------------------------------------------------------
# Niobe measured this machine: the same `tests/ops` suite took 708 s with four agents
# running and 161 s quiet.  **4.4x.**  A subprocess timeout calibrated against a quiet
# machine therefore fires on a healthy system whenever the team is working, and
# `subprocess.TimeoutExpired` propagates as a red that reads exactly like a detection.
# It has already cost this project a day: `68 failed` was read as a regression when it
# was contention, and `test_wiring_census` had to be `--ignore`d to get a clean run.
#
# Two obligations, and they are separate:
#   1. The budget must survive the inflation.  A wall-clock threshold that a loaded
#      machine cannot meet manufactures false reds for everyone.
#   2. When the budget is exceeded anyway, the terminal state is ERROR(instrument):
#      the check did not reach its observation, so it detected nothing.
#
# Note what is NOT here: no assertion anywhere in this harness compares a wall-clock
# duration to a threshold.  A timeout is a *ceiling on waiting*, not a measurement, and
# it is the only wall-clock number the conformance lane is permitted to contain.
# Device-clock measurement is Switch's exclusive claim; Niobe owns the reporting gate.
#
# 2026-08-01, Trinity — READ THIS BEFORE REACHING FOR THIS WRAPPER IN A NEW LANE.
# Obligation 1 above is not achievable and this comment was wrong to state it as a
# requirement.  A wall-clock budget fires the same way for "the box is loaded" and for
# "the child hung", so it moves with the reader's confidence and R9 amendment 5 demotes it
# from gate to precondition: no value of it separates the two cases, and the value that
# survives this host's 9.5x `record`-step inflation is wider than the hang it exists to
# catch.  `tests/ops/_watchdog.py` replaces it for the census with a budget denominated in
# reference computations the machine completed during the run — contention lowers
# units-per-second and widens the window automatically, while a hang exhausts the budget in
# bounded work on a loaded box exactly as on a quiet one.  Both arms are demonstrated in
# `tests/ops/probe_stall_guard.py` (four cells, `arms_must_differ`) and in the always-on
# `tests/ops/test_stall_guard.py`.
#
# This wrapper stays because its other callers are short, fixed-cost commands where the
# ceiling is a convenience rather than a gate, and because obligation 2 — a timeout is
# ERROR(instrument), never a detection — is correct and unchanged.  New long-running or
# hang-prone steps should use `_watchdog.guarded_run` instead.

#: Multiplier applied to a quiet-machine budget.  Niobe's measured worst case is 4.4x;
#: 6.0 is that with headroom, because the cost of over-waiting is minutes and the cost of
#: under-waiting is a fabricated regression that the whole team then investigates.
CONTENTION_INFLATION_FACTOR: float = 6.0

#: No subprocess budget is ever below this, however small its quiet-machine cost.  Process
#: start-up alone can lose tens of seconds on a contended Windows host.
CONTENTION_TIMEOUT_FLOOR_S: float = 120.0

#: Escape hatch for a lane that knows it is quiet (or knows it is much worse).  Read once
#: per call so a lane can set it in the environment rather than editing tests.
CONTENTION_BUDGET_ENV: str = "ONNXRUNTIME_EP_VULKAN_TIMEOUT_SCALE"


def contention_tolerant_timeout(
    quiet_seconds: float,
    *,
    floor: float = CONTENTION_TIMEOUT_FLOOR_S,
    env: "Mapping[str, str] | None" = None,
) -> float:
    """Inflate a quiet-machine budget into one a contended machine can still meet.

    Pure: given the same arguments and environment mapping it returns the same number, so
    it is falsifiable without a machine to contend for.

    ``quiet_seconds`` is what the command costs on an idle host.  The result is that,
    scaled by :data:`CONTENTION_INFLATION_FACTOR` (or by ``$ONNXRUNTIME_EP_VULKAN_TIMEOUT_SCALE``
    when set), and never below ``floor``.
    """
    source = os.environ if env is None else env
    scale = CONTENTION_INFLATION_FACTOR
    raw = source.get(CONTENTION_BUDGET_ENV)
    if raw:
        try:
            parsed = float(raw)
        except (TypeError, ValueError):
            parsed = 0.0
        if parsed > 0:
            scale = parsed
    return max(float(quiet_seconds) * scale, float(floor))


def run_subprocess_checked(
    cmd: "list[str]",
    *,
    what: str,
    quiet_seconds: float,
    floor: float = CONTENTION_TIMEOUT_FLOOR_S,
    **kwargs: Any,
):
    """Run *cmd* with a contention-tolerant budget; every failure to run is an outage.

    Returns the ``CompletedProcess``.  Raises :class:`InstrumentError` — never
    ``AssertionError`` — when the command could not be started, could not be found, or
    exceeded its budget.  The caller therefore *only* ever sees a process that ran to
    completion, and any assertion it then writes is necessarily about an observation.

    The exit code is deliberately NOT interpreted here: a non-zero exit is the callee's
    verdict about the world and belongs to the caller's condition, not to this wrapper's
    instrument frame.
    """
    import subprocess  # local: keeps this module importable in a stripped environment

    budget = contention_tolerant_timeout(quiet_seconds, floor=floor)
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=budget,
            **kwargs,
        )
    except subprocess.TimeoutExpired as exc:
        partial = ""
        for stream in (exc.stdout, exc.stderr):
            if stream:
                partial += stream if isinstance(stream, str) else stream.decode("utf-8", "replace")
        raise InstrumentError(
            f"[{what} instrument failure] ERROR(instrument): the subprocess exceeded its "
            f"{budget:.0f}s budget (quiet-machine estimate {quiet_seconds:.0f}s x "
            f"{budget / max(quiet_seconds, 1e-9):.1f}).\n"
            "A TIMEOUT IS NOT A DETECTION (R13).  Nothing was observed, so nothing about "
            "the EP has been established by this red.  The most likely cause on this host "
            "is machine contention: the same suite has been measured at 708 s under four "
            "concurrent agents and 161 s quiet (4.4x).  Raise "
            f"${CONTENTION_BUDGET_ENV} or run the lane on a quiet machine.\n"
            f"command: {' '.join(map(str, cmd))}\n"
            f"partial output (last 800 chars):\n{partial[-800:]}",
            observed=None,
        ) from exc
    except FileNotFoundError as exc:
        raise InstrumentError(
            f"[{what} instrument failure] ERROR(instrument): the command does not exist on "
            f"this machine: {cmd[0]!r}.  The check never ran; this is not a finding.\n{exc}"
        ) from exc
    except OSError as exc:
        raise InstrumentError(
            f"[{what} instrument failure] ERROR(instrument): could not start the subprocess "
            f"{cmd[0]!r}: {exc}.  The check never ran; this is not a finding."
        ) from exc


# ---------------------------------------------------------------------------
# CRITERION 3 — the frame the messenger control runs in (R12)
# ---------------------------------------------------------------------------
# Switch, 2026-07-31: *gate the wrapper on `Instance::validation_armed()`
# (instance.rs:450) so an unarmed machine reports ERROR(instrument) rather than green.*
#
# The whole of criterion 3 exists because "no validation errors surfaced" is exactly what
# a machine with no validation layer reports.  The planted-fence-leak control inherits the
# same disease one level down: on a machine where the EP's instance has no
# `VkDebugUtilsMessengerEXT`, `EP_VALIDATION_ERROR_COUNT` **cannot** become non-zero, so
# the control's `assert count > 0` fails for a reason that has nothing to do with the EP.
# Read as a detection it accuses Switch's messenger; read as a skip it reports green.
#
# R12 names the correct third answer: *a counter whose event cannot occur in its frame
# reports UNOBSERVABLE, never 0.*  Unarmed is UNOBSERVABLE, and UNOBSERVABLE is
# ERROR(instrument).
#
# `epctl --probe-validation` is the in-lane predicate: it creates an instance with the
# layer enabled AND a messenger attached, which is precisely the condition
# `validation_armed()` reports on the EP's own instance.  Its three states are already
# distinguished by exit code, so the classification below is a total function of
# (exit code, output) and needs no Vulkan to falsify.

#: The layer is installed, was enabled, and a messenger is receiving its output.  This is
#: the ONLY state in which the planted-violation control can observe anything.
VALIDATION_ARMED: str = "ARMED"
#: The loader is present but ``VK_LAYER_KHRONOS_validation`` is not installed.
VALIDATION_LAYER_ABSENT: str = "LAYER-ABSENT"
#: There is no Vulkan loader at all.
VALIDATION_NO_LOADER: str = "NO-LOADER"
#: The probe itself failed — it produced neither an arming nor a refusal.
VALIDATION_PROBE_ERROR: str = "PROBE-ERROR"

#: The exit code ``epctl --probe-validation`` uses for "cannot answer".
EPCTL_EXIT_VALIDATION_UNAVAILABLE: int = 3


def classify_validation_probe(returncode: "int | None", output: str) -> "tuple[str, str]":
    """Classify ``epctl --probe-validation`` into a frame state and a human reason.

    Total on every input, including ``None`` (the process was killed by a signal) and an
    empty string.  Returns ``(state, reason)`` where *state* is one of
    :data:`VALIDATION_ARMED`, :data:`VALIDATION_LAYER_ABSENT`, :data:`VALIDATION_NO_LOADER`
    or :data:`VALIDATION_PROBE_ERROR`.
    """
    text = output or ""
    upper = text.upper()
    if returncode == 0:
        if "ARMED" in upper:
            return VALIDATION_ARMED, "epctl reports VALIDATION ARMED"
        return (
            VALIDATION_PROBE_ERROR,
            "epctl --probe-validation exited 0 but never printed ARMED, so the probe did "
            "not actually report a state.  Exit 0 alone is not the observation.",
        )
    if returncode == EPCTL_EXIT_VALIDATION_UNAVAILABLE:
        if "NO VULKAN LOADER" in upper or "NO LOADER" in upper:
            return VALIDATION_NO_LOADER, "no Vulkan loader on this machine"
        return VALIDATION_LAYER_ABSENT, "VK_LAYER_KHRONOS_validation is not installed"
    if returncode is None:
        return (
            VALIDATION_PROBE_ERROR,
            "epctl --probe-validation was killed by a signal and returned no exit code",
        )
    return (
        VALIDATION_PROBE_ERROR,
        f"epctl --probe-validation exited {returncode}, which is neither armed (0) nor "
        f"unavailable ({EPCTL_EXIT_VALIDATION_UNAVAILABLE})",
    )


def require_validation_armed(state: str, reason: str) -> None:
    """Raise :class:`InstrumentError` unless the messenger frame can observe anything.

    This is the gate Switch asked for.  It raises rather than skipping on purpose: a skip
    is green, and a green criterion-3 control on a machine that cannot run it is the exact
    silence the criterion was written to remove.
    """
    if state == VALIDATION_ARMED:
        return
    raise InstrumentError(
        "[criterion 3 messenger control instrument failure] ERROR(instrument): validation "
        f"is not armed on this machine ({state} — {reason}).\n"
        "R12: a counter whose event cannot occur in its frame reports UNOBSERVABLE, never "
        "0.  With no VkDebugUtilsMessengerEXT receiving the layer's output, "
        "EP_VALIDATION_ERROR_COUNT cannot become non-zero for any state of the EP, so the "
        "planted-fence-leak control has no observation to make.\n"
        "THIS IS NOT A FINDING ABOUT THE MESSENGER, and it is NOT A PASS.  It is the "
        "third terminal state.  Install the Vulkan SDK validation layers to make this "
        "machine able to answer.",
        observed=state,
    )


# --- the planted-violation control's own three states ----------------------

#: The artifact the ``#[ignore]``d Rust control prints.  Named here so a rename on the
#: Rust side fails this parse loudly instead of silently making the check unfalsifiable.
PLANT_ARTIFACT_RE = re.compile(
    r"EP_VALIDATION_ERROR_COUNT after planted fence leak\s*=\s*(\d+)"
)

#: Substrings in the control's output that mean the *build*, not the control, failed.
_BUILD_OUTAGE_MARKERS: tuple[str, ...] = (
    "could not compile",
    "error[E",
    "error: linking with",
    "no such command",
    "error: could not find",
)


def classify_plant_run(returncode: "int | None", output: str) -> "tuple[str, str, int | None]":
    """Classify the planted-fence-leak control run into R13's three terminal states.

    Returns ``(state, reason, count)`` where *state* is ``"PASS"``, ``"FAIL"`` or
    ``"ERROR"`` and *count* is the observed ``EP_VALIDATION_ERROR_COUNT`` when one was
    printed, else ``None``.

    Total and pure — it takes an exit code and a string, so every branch below is
    falsifiable from a synthesised transcript with no Vulkan, no cargo and no GPU.  That
    is the same pattern that let Guard D be screened in both polarities.

    The ordering matters and is the substance of R13:

      * a build failure is an outage, whatever the control would have said;
      * a self-skip (no shaders, no ICD, no capable device) is an outage — R12's
        UNOBSERVABLE, not a zero;
      * **no artifact line at all is an outage**, because the control did not reach its
        observation.  This branch is why a bare ``assert returncode == 0`` is not enough:
        a control that crashed before the plant and a control that watched the plant fail
        both exit non-zero;
      * an artifact line with ``0`` IS an observation, and therefore FAIL(condition).
    """
    text = output or ""
    lowered = text.lower()

    if any(marker.lower() in lowered for marker in _BUILD_OUTAGE_MARKERS):
        return (
            "ERROR",
            "the control did not build, so it never ran.  A compile failure is an "
            "instrument outage and never a finding about the messenger.",
            None,
        )
    if "[skip]" in lowered:
        line = next(
            (ln.strip() for ln in text.splitlines() if "[SKIP]" in ln.upper()),
            "[SKIP]",
        )
        return (
            "ERROR",
            "the control skipped itself because this machine cannot host it "
            f"({line}).  R12: the plant's event cannot occur in this frame, so there is "
            "no count — UNOBSERVABLE, not 0.",
            None,
        )

    match = PLANT_ARTIFACT_RE.search(text)
    if match is None:
        return (
            "ERROR",
            "the control produced no EP_VALIDATION_ERROR_COUNT line, so it did not reach "
            "its observation.  Either it never ran, or it died before the plant.  A red "
            "here is an outage, not a detection.",
            None,
        )

    count = int(match.group(1))
    if count <= 0:
        return (
            "FAIL",
            "EP_VALIDATION_ERROR_COUNT = 0 after a deliberately leaked VkFence at "
            "vkDestroyDevice.  Validation is armed on this machine (checked before the "
            "run), so the event COULD have occurred and did not: the EP's own "
            "VkDebugUtilsMessengerEXT is not receiving VUID-vkDestroyDevice-device-05137. "
            "This IS a finding, and it is Switch's.",
            count,
        )
    if returncode != 0:
        return (
            "FAIL",
            f"the control observed EP_VALIDATION_ERROR_COUNT = {count} (> 0) but still "
            f"exited {returncode}, so something else in the control failed after the "
            "plant fired.  It reached an observation, so this is a condition failure.",
            count,
        )
    return (
        "PASS",
        f"EP_VALIDATION_ERROR_COUNT = {count} after the planted fence leak: the EP's own "
        "messenger is wired and fires on a real violation, in a frame that was checked to "
        "be armed before the run.",
        count,
    )



# ---------------------------------------------------------------------------
# Which `model_output_equivalence` is of record — round 28
# ---------------------------------------------------------------------------
# `bench/results/phi35-certified-dev0.json` — the artifact behind the only quotable
# figure — carries, adjacent to one another:
#
#     results[0].model_output_equivalence          = MATCH
#     results[0].counters.model_output_equivalence = UNMEASURED
#
# Same name, two places, disagreeing. That reads as two instruments contradicting each
# other. It is not. `rust/src/counters.rs::to_json()` emits `UNMEASURED` as its *default*
# and says so in its own header comment: "The EP has no access to the CPU oracle. The
# verdict is set by Trinity's Python harness after running a VulkanEP-vs-CPU comparison."
# So the counters copy is not a measurement that came back negative; it is a field that
# nobody in that frame ever set.
#
# The tell is mechanical and needs no judgement: `write_equivalence_record` writes the
# token and the `model_output_equivalence_record` object **together**, in one call. A
# counters document carrying the token with no record beside it was never written by a
# comparison. In the phi35 artifact the record key is absent.
#
# R12 — that state is UNOBSERVABLE, with a reason, and never a token a reader can mistake
# for a verdict. R11 — "the two sources disagree" is the decomposition that appears to
# close: it sends the reader hunting for a disagreement, and there is none to find.

#: A `model_output_equivalence` that no comparison in this frame wrote.
EQUIVALENCE_AUTHORITY_UNSET: str = "UNOBSERVABLE"
#: A `model_output_equivalence` written together with its record by a comparison.
EQUIVALENCE_AUTHORITY_MEASURED: str = "MEASURED"


def equivalence_authority(counters: "dict[str, Any] | None") -> "tuple[str, str, str]":
    """Say whether a counters document's equivalence token is a reading or a default.

    Returns ``(authority, token, reason)``.

    ``authority`` is :data:`EQUIVALENCE_AUTHORITY_MEASURED` only when the token is
    accompanied by the record object that :func:`write_equivalence_record` writes with it.
    Otherwise it is :data:`EQUIVALENCE_AUTHORITY_UNSET`: the field is the emitter's default
    and carries no information about correctness.

    The test is on the *record's presence*, not on the token's value, deliberately. Keying
    off ``token == UNMEASURED`` would answer the question by reading the very field whose
    trustworthiness is in question, and would also mislabel a genuine comparison that
    legitimately concluded ``UNMEASURED``.
    """
    if not isinstance(counters, dict):
        return (
            EQUIVALENCE_AUTHORITY_UNSET,
            VERDICT_UNMEASURED,
            "no counters document was recorded in this frame",
        )
    token = counters.get(EQUIVALENCE_KEY)
    record = counters.get(EQUIVALENCE_RECORD_KEY)
    if isinstance(record, dict) and record:
        return (
            EQUIVALENCE_AUTHORITY_MEASURED,
            str(token),
            f"written with its {EQUIVALENCE_RECORD_KEY} by a CPU-vs-EP comparison",
        )
    if token is None:
        return (
            EQUIVALENCE_AUTHORITY_UNSET,
            VERDICT_UNMEASURED,
            f"the counters document carries no {EQUIVALENCE_KEY} at all",
        )
    return (
        EQUIVALENCE_AUTHORITY_UNSET,
        str(token),
        f"{EQUIVALENCE_KEY}={token!r} stands alone with no {EQUIVALENCE_RECORD_KEY} "
        "beside it, so it is the emitter's default rather than a comparison's verdict "
        "(rust/src/counters.rs::to_json)",
    )


def reconcile_equivalence(record: "dict[str, Any]") -> "dict[str, Any]":
    """Reconcile the two `model_output_equivalence` values inside one bench result.

    *record* is one element of a bench artifact's ``results`` list: an outer
    ``model_output_equivalence`` written by the comparison harness, and a nested
    ``counters`` document whose copy may be the emitter's default.

    Returns the stamp an artifact should carry so a reader never has to guess which value
    is of record. ``agreement`` is ``AGREE``/``DISAGREE`` only when **both** sides are
    readings; when one side is a default the answer is ``UNOBSERVABLE`` and not ``AGREE``,
    because two values one of which nobody wrote have not agreed about anything.
    """
    outer = record.get(EQUIVALENCE_KEY)
    authority, counters_token, reason = equivalence_authority(record.get("counters"))
    if authority == EQUIVALENCE_AUTHORITY_MEASURED:
        agreement = (
            WITNESS_AGREEMENT_AGREE
            if counters_token == outer
            else WITNESS_AGREEMENT_DISAGREE
        )
    else:
        agreement = WITNESS_AGREEMENT_UNOBSERVABLE
    return {
        "of_record": EQUIVALENCE_KEY,
        "of_record_value": outer,
        "of_record_source": (
            "bench/phi35.py::compare — the CPU-vs-EP output comparison, whose evidence is "
            "the sibling `outputs` list (argmax, top-k overlap, max_rel_diff)"
        ),
        "counters_copy_value": counters_token,
        "counters_copy_authority": authority,
        "counters_copy_reason": reason,
        "agreement": agreement,
        "note": (
            "Two same-named fields, one verdict. The nested copy is NOT a second "
            "measurement disagreeing with the first; it is the EP's default for a field "
            "that only the Python comparison harness writes. Read the outer value."
        ),
    }
