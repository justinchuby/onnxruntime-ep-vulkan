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

A THIRD SHAPE: THE VERDICT-RETURNING INSTRUMENT
==============================================
2026-08-09 (Niobe, issue #69).  ``refuses``/``selects`` model a total instrument whose
refusal is ``(None, why)``.  A *gate* has a third shape: it returns
``{"verdict": FAIL, "condition": ..., "detail": ...}``, because under R13 a detection is a
verdict and only a failure to observe is an exception.  ``bench/phi_evidence.py`` is built
that way on purpose — ``evidence_gate`` never raises about content — and the consequence for
this screen was that the most heavily attacked instrument in the tree scored
``reject_polarity=0``: nothing it does looks like ``pytest.raises``.

``convicts`` closes that the same way ``refuses`` closed the first gap, and on the same
terms: it *enforces*.  It raises ``PolarityError`` unless the result really is a refusing
verdict, and unless the condition token matches the one the caller named.  A gate mutated to
wave everything through cannot pass through it, which is what makes crediting it a polarity
this screen observed rather than one it was told about.
"""

from __future__ import annotations

from typing import Any

#: Verdict tokens that mean "this instrument declined to certify".  ``INDETERMINATE`` is here
#: for the same reason ``FAIL`` is: a classifier that can only ever say IMPROVEMENT has no
#: reject polarity, and the way that defect looks from outside is a run where everything won.
REFUSAL_VERDICTS = frozenset(
    {"FAIL", "ERROR", "INDETERMINATE", "INCONCLUSIVE", "DIVERGENT", "UNMEASURED", "REGRESSION"}
)


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


def convicts(result: Any, *, condition: str | None = None, because: str = "") -> str:
    """Assert a verdict-returning instrument DECLINED, and return the condition it named.

    The reject polarity for a gate.  ``result`` must be a mapping carrying a ``verdict`` in
    :data:`REFUSAL_VERDICTS`; when ``condition`` is given it must match exactly, so a gate that
    convicts on the wrong grounds is a red rather than a pass.  That last clause is the whole
    point: a gate with one over-eager clause fails everything, and a test that only checked
    "it said FAIL" would call that health.
    """
    ctx = f" ({because})" if because else ""
    if not isinstance(result, dict):
        raise PolarityError(
            f"convicts{ctx}: a verdict-returning instrument must return a mapping with a "
            f"`verdict` key; got {type(result).__name__} {result!r}."
        )
    verdict = result.get("verdict")
    if verdict not in REFUSAL_VERDICTS:
        raise PolarityError(
            f"expected a REFUSING verdict{ctx}, but the instrument returned {verdict!r} and "
            f"said: {result.get('detail') or result.get('reason')!r}. This is the polarity that "
            f"catches an instrument which certifies everything."
        )
    got = result.get("condition")
    if condition is not None and got != condition:
        raise PolarityError(
            f"the instrument refused{ctx} on the WRONG grounds: condition={got!r}, expected "
            f"{condition!r}. A gate that convicts on any mutation, including one it was not "
            f"built to see, has not been shown to read the thing it names."
        )
    return got if isinstance(got, str) else str(verdict)
