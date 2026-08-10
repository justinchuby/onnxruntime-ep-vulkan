"""Two-polarity assertions for TOTAL instruments — the ones that return a refusal.

WHY THIS FILE EXISTS
====================
``rust/tools/audit_instruments.py`` decides whether an instrument has been *falsified* by
reading the test AST: it must see the instrument called inside ``pytest.raises(...)``
(reject polarity) and outside one (accept polarity).  Both, or the instrument is
``unfalsified`` — nothing has ever watched it disagree, so a broken one and a working one
look the same.  That model is exactly right for a guard that raises.

It is blind to a **total** instrument.  ``bench/devices.py::identify_by_uuid`` returns
``(device, why)`` and never raises: its refusal is the ``None`` in the first slot and the
sentence in the second.  Its caller ``device_identity_check`` *needs* that sentence on the
refusal path — it prints the reason a row's device name was withheld — so the refusal
cannot be moved into an exception without carrying the reason out of band.  The totality is
deliberate, not an oversight.

That left three ways to close the census finding and only one of them is honest:

1. Force an exception contract onto it, so the existing screen can see it.  That is
   changing the subject to fit the instrument, and it would make the *production* path
   worse to make the *screen* greener.  Refused.
2. Baseline it with a hand note saying it cannot be screened.  That is true of the screen
   as it was, and it converts an open question into a permanent one.  Refused.
3. Teach the screen a second polarity source that observes as strongly as
   ``pytest.raises`` does.  This file.

WHAT MAKES THIS NOT A BARE MARKER
=================================
``pytest.raises`` earns its credit because it is not an annotation: it *fails the test* if
the thing inside it does not raise.  A screen that credited a comment saying "this is the
reject polarity" would be crediting a polarity it never observed — the Guard D shape with
the sign flipped, which ``audit_instruments.py`` says out loud it will not do.

So ``refuses`` and ``selects`` enforce, at run time, the contract they name:

* ``refuses(result)`` raises unless ``result`` is ``(None, why)`` with a non-empty ``why``.
  An instrument that returned a device where refusal was due makes the test red.
* ``selects(result, expected)`` raises unless ``result`` is ``(expected, why)`` with a
  non-empty ``why``, compared by **identity** (``is``), not equality — two probed entries
  for two same-model cards compare equal on every field a naive ``==`` would look at, and
  that is precisely the confusion issue #18 exists to end.

A mutant instrument therefore cannot pass through either of them.  That is demonstrated,
not asserted, in ``bench/test_devices_identity.py``: five deliberately defective
reimplementations of ``identify_by_uuid`` are put through the identical protocol and every
one must go red.

THE BLIND SPOT, STATED
======================
This says nothing about whether the *inputs* a test feeds the instrument actually vary the
thing under test.  Neither does ``pytest.raises``.  That property is earned by the mutation
battery, the same way ``tests/ops/test_guard_d.py`` earns it for the harness domain, and it
is not claimed here.

A FOURTH SHAPE: THE VERDICT THAT WAS REACHED, AND WAS NEGATIVE
=============================================================
Added 2026-08-09 (Niobe) with ``bench/resnet.py``.  ``withholds`` above reads the refusal of
an instrument that reached NO verdict.  ``bench/resnet.py``'s classifiers have the other
half of that shape: they reach a definite verdict and it is ``DIVERGENT`` — "these two arms
disagree" — which is an answer, not an absence.  ``admissibility`` sits on the ``withholds``
side (``INDETERMINATE`` is a withheld verdict and is already in ``WITHHELD_TOKENS``), but
its sibling ``quotable`` returns a bare ``False`` and its classifiers return ``DIVERGENT``,
and neither is a withheld verdict under any honest reading.

Two ways to make the screen see them, and only one of them is honest:

1. Widen ``WITHHELD_TOKENS`` until ``DIVERGENT`` counts as "no verdict reached".  That
   destroys the distinction ``withholds`` exists to make — after it, "the instrument had no
   evidence" and "the instrument had evidence and it disagreed" print the same word.
   Refused.
2. A second verb for the second thing.  ``dissents``.

``dissents(result)`` asserts the instrument returned a NEGATIVE VERDICT **with a reason**,
and returns the record so the negative control that called it can go on to check *how* the
instrument disagreed.  A record carrying ``MATCH``/``ADMISSIBLE``, or a bare ``True``,
raises; so does a negative verdict that named no failing condition, because a refusal that
does not say which clause it failed cannot be acted on.  It is an assertion, not an
annotation, on the same footing as everything else in this file, and the planted mutants
that prove so live in ``bench/test_resnet.py``.

WHICH ONE TO REACH FOR
======================
* No verdict was reached — ``INDETERMINATE``, ``REFUSED``, an empty set, a ``None``:
  ``withholds``.
* A verdict was reached and it was negative — ``DIVERGENT``, a ``False`` gate: ``dissents``.
* The refusal is the ``None`` in a ``(value, why)`` pair: ``refuses``.
"""

from __future__ import annotations

from typing import Any


class PolarityError(AssertionError):
    """A total instrument did not honour the polarity contract it was asserted under."""


def _unpack(result: Any, polarity: str) -> "tuple[Any, Any]":
    if not isinstance(result, tuple) or len(result) != 2:
        raise PolarityError(
            f"{polarity}: a total instrument must return a 2-tuple (value, why); got "
            f"{type(result).__name__} {result!r}. Without the `why` there is no difference "
            f"between 'it refused and said so' and 'it returned nothing'."
        )
    return result[0], result[1]


def refuses(result: Any, *, because: str = "") -> str:
    """Assert a total instrument REFUSED, and return the reason it gave.

    The reject polarity.  ``result`` must be ``(None, why)`` with a non-empty ``why``: an
    instrument that declines to identify something must say which of its refusal cases it
    hit, or the caller cannot tell "no evidence" from "the evidence disagreed".
    """
    value, why = _unpack(result, "refuses")
    ctx = f" ({because})" if because else ""
    if value is not None:
        raise PolarityError(
            f"expected a REFUSAL{ctx}, but the instrument identified {value!r} and said: "
            f"{why!r}. A guess is not an identification — this is the polarity that catches "
            f"an instrument which accepts everything."
        )
    if not isinstance(why, str) or not why.strip():
        raise PolarityError(
            f"the instrument refused{ctx} but gave no reason ({why!r}). A silent refusal and "
            f"a crash are indistinguishable to every reader we have."
        )
    return why


def selects(result: Any, expected: Any, *, because: str = "") -> Any:
    """Assert a total instrument IDENTIFIED exactly ``expected``, and return it.

    The accept polarity.  ``result`` must be ``(expected, why)`` with a non-empty ``why``.
    The comparison is ``is``, not ``==``: two probed devices of the same model carry the
    same name, vendor and driver strings, so an equality test would pass on the wrong one
    and certify the collapse this instrument exists to prevent.
    """
    value, why = _unpack(result, "selects")
    ctx = f" ({because})" if because else ""
    if value is None:
        raise PolarityError(
            f"expected the instrument to identify a device{ctx}, but it refused and said: "
            f"{why!r}. This is the polarity that catches an instrument which refuses "
            f"everything."
        )
    if value is not expected:
        raise PolarityError(
            f"the instrument identified the WRONG device{ctx}: got {value!r}, expected "
            f"{expected!r} (compared by identity, not equality). It said: {why!r}."
        )
    if not isinstance(why, str) or not why.strip():
        raise PolarityError(
            f"the instrument identified a device{ctx} but gave no reason ({why!r})."
        )
    return value


# =============================================================================================
# A third polarity source: the instrument that WITHHOLDS a verdict
# =============================================================================================
#
# `refuses`/`selects` above are written for one totality contract: `(value, why)`.  Issue #96's
# summarizer has a different one.  Its instruments return a *verdict mapping* — `classify_record` returns
# `{"status": ..., "reasons": [...]}`, `calibration` returns `{"band": None, "reason": ...}`,
# `window_verdict` returns `{"claim": INDETERMINATE, "reason": ...}` — and their refusal is a
# withheld verdict rather than a `None` in slot zero.
#
# Retrofitting the `(value, why)` shape onto them to satisfy the screen is exactly the move this
# file refused two hundred lines up: changing the subject to fit the instrument.  So the honest
# option is the same one taken for `refuses` — teach the screen a source that observes as strongly
# as `pytest.raises` does, and make it an assertion rather than an annotation.
#
# `withholds` fails the test when the thing inside it did NOT withhold.  An instrument that
# accepted a record it should have refused, produced a band it had no calibration for, or graded a
# comparison it had no neighbours for cannot pass through it.  And when the withheld thing is a
# decision token or a `None`, the instrument must additionally have SAID WHY — a silent refusal
# and a crash are indistinguishable to every reader we have, which is the same standard `refuses`
# holds its subject to.

#: Verdict tokens that mean "no verdict was reached".  A withheld verdict is not a soft verdict.
WITHHELD_TOKENS = frozenset({"REFUSED", "INDETERMINATE", "INCONCLUSIVE", "UNMEASURED"})

#: Where a verdict mapping is allowed to carry its reason.  Checked in order.  `failed_checks`
#: and `failed_conditions` were added 2026-08-09 (Niobe) for `bench/resnet.py`, which names its
#: failing conditions in a list rather than in a prose `reason`; the addition is additive and
#: widens no acceptance — a record still has to say SOMETHING or neither helper credits it.
_REASON_KEYS = ("reason", "reasons", "failed_checks", "failed_conditions", "differences",
                "disagreements", "missing_sides", "contaminated", "incomplete_repeats")


def _resolve(result: Any, key: "str | None") -> "tuple[Any, Any]":
    """Return ``(value, parent_mapping)`` for a dotted *key*, or the whole result for ``None``."""
    if key is None:
        return result, result if isinstance(result, dict) else None
    parent: Any = None
    value: Any = result
    for part in key.split("."):
        if not isinstance(value, dict):
            raise PolarityError(
                f"withholds: cannot read {key!r} — {part!r} is not a field of a "
                f"{type(value).__name__}. A verdict-returning instrument must return a mapping."
            )
        if part not in value:
            raise PolarityError(
                f"withholds: the instrument returned no {part!r} field at all (keys: "
                f"{sorted(value)}). An absent verdict field is not a withheld verdict."
            )
        parent, value = value, value[part]
    return value, parent


def _is_withheld(value: Any) -> "tuple[bool, bool]":
    """``(withheld, must_explain)`` for one resolved value."""
    if value is None:
        return True, True
    if isinstance(value, bool):
        return (not value), (not value)
    if isinstance(value, str):
        if value in WITHHELD_TOKENS:
            return True, True
        return (value == ""), False
    if isinstance(value, float):
        return (value != value), False  # NaN is the only float refusal in circulation
    if isinstance(value, int):
        return (value == 0), False
    if isinstance(value, (list, tuple, set, dict)):
        return (len(value) == 0), False
    return False, False


def withholds(result: Any, key: "str | None" = None, *, because: str) -> str:
    """Assert a verdict-returning total instrument WITHHELD its verdict, and return its reason.

    The reject polarity for the issue-#96 summarizer.  *key* is a dotted path into the returned
    mapping naming the field that carries the verdict (``"status"``, ``"band"``, ``"band.band"``,
    ``"claim"``, ``"reproduces"``); ``None`` means the whole return value is the verdict, which is
    how ``past_of`` and ``power_at`` decline.

    *because* is required and must name which refusal case is under test.  It is not what earns
    the polarity credit — the assertions below are — but a test that cannot say which refusal it
    is exercising is a test nobody can maintain.
    """
    if not isinstance(because, str) or not because.strip():
        raise PolarityError(
            "withholds: `because` must name the refusal case under test. The screen credits the "
            "assertion, not the string, but an unnamed refusal case is unreviewable."
        )
    value, parent = _resolve(result, key)
    withheld, must_explain = _is_withheld(value)
    if not withheld:
        raise PolarityError(
            f"expected a WITHHELD verdict ({because}), but the instrument returned "
            f"{value!r} for {key or 'its result'}. This is the polarity that catches an "
            f"instrument which reaches a verdict it has no evidence for."
        )
    # An instrument whose whole return value is the refusal — `past_of` answering "that label is
    # not a decode row" with `None` — has nowhere to put a reason and needs none: the absence IS
    # the answer, and there is no second refusal case for a reader to confuse it with. The
    # requirement applies where there is a mapping that could have carried one and did not.
    if not must_explain or not isinstance(parent, dict):
        return ""
    reasons = []
    for source in (parent, result if isinstance(result, dict) else None):
        if not isinstance(source, dict):
            continue
        for name in _REASON_KEYS:
            candidate = source.get(name)
            if isinstance(candidate, str) and candidate.strip():
                reasons.append(candidate)
            elif isinstance(candidate, (list, tuple)) and len(candidate):
                reasons.append("; ".join(str(x) for x in candidate))
        if reasons:
            break
    if not reasons:
        raise PolarityError(
            f"the instrument withheld its verdict ({because}) but gave no reason: none of "
            f"{list(_REASON_KEYS)} is present and non-empty. A silent refusal and a crash are "
            f"indistinguishable to every reader we have."
        )
    return " | ".join(reasons)


# --------------------------------------------------------------------------------------------
# Negative-verdict polarity — see "A FOURTH SHAPE" above.
# --------------------------------------------------------------------------------------------

#: Tokens that mean "a verdict WAS reached and it was negative". Deliberately overlapping
#: `WITHHELD_TOKENS` on `INDETERMINATE` and nowhere else: an admissibility record is readable
#: both ways — no number was reached, and the gate said no — and forcing a caller to pick the
#: one true verb for it would be a distinction without a difference. `DIVERGENT` is only ever
#: the second kind, and `REFUSED`/`INCONCLUSIVE`/`UNMEASURED` are only ever the first.
#:
#: Kept as literals rather than imported from `real_model`/`resnet` so this module stays
#: importable by anything in `bench/` without a cycle, and so adding a token is a deliberate
#: edit here rather than a side effect of renaming a constant somewhere else.
NEGATIVE_VERDICTS = frozenset({"DIVERGENT", "INDETERMINATE", "FAIL"})


def _stated_reason(record: dict) -> "str | None":
    for key in _REASON_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, (list, tuple)) and len(value) > 0:
            return ", ".join(str(v) for v in value)
    checks = record.get("checks")
    if isinstance(checks, (list, tuple)):
        failed = [c for c in checks if isinstance(c, dict) and not c.get("held")]
        if failed:
            return ", ".join(str(c.get("name")) for c in failed)
    return None


def dissents(result: Any, *, because: str = "") -> Any:
    """Assert a VERDICT instrument said NO, and return the record it said it in.

    The reject polarity for a total instrument whose refusal is a record rather than a raise.
    ``result`` is either ``False`` (a boolean gate that declined) or a mapping whose
    ``verdict`` is in `NEGATIVE_VERDICTS` and which says which condition failed.

    Returns the argument unchanged rather than the reason string ``refuses`` returns, because
    the reason is one field of a record whose other fields — which rows disagreed, by how
    much, against what budget — are the substance of the negative control that called this.
    Handing back the reason alone would push every caller into taking the verdict twice, and
    the second take is the one the screen cannot see.
    """
    ctx = f" ({because})" if because else ""
    if result is False:
        return result
    if result is True:
        raise PolarityError(
            f"expected a NEGATIVE verdict{ctx}, but the gate returned True. This is the "
            f"polarity that catches an instrument which admits everything."
        )
    if not isinstance(result, dict):
        raise PolarityError(
            f"dissents{ctx}: a verdict instrument must return a record or a bool; got "
            f"{type(result).__name__} {result!r}."
        )
    verdict = result.get("verdict")
    if verdict not in NEGATIVE_VERDICTS:
        raise PolarityError(
            f"expected a NEGATIVE verdict{ctx}, but the record says verdict={verdict!r}. "
            f"Negative tokens are {sorted(NEGATIVE_VERDICTS)}; a record that agreed cannot "
            f"stand as the observation that this instrument is able to disagree."
        )
    if _stated_reason(result) is None:
        raise PolarityError(
            f"the instrument returned {verdict!r}{ctx} but named no failing condition "
            f"(looked in {list(_REASON_KEYS)} and in `checks`). A refusal that does not say "
            f"what it refused on is indistinguishable from a crash to every reader we have."
        )
    return result
