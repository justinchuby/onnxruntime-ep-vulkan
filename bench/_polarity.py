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
# ---------------------------------------------------------------------------
# Verdict-token instruments
# ---------------------------------------------------------------------------
#
# `refuses`/`selects` above are shaped for a `(value, why)` instrument. A verdict gate is
# total in a different shape: it takes a record and hands back the same record with the
# verdict either withheld or left standing, because the refusal has to travel WITH the
# numbers it disqualifies rather than beside them.
#
# The alternative was to make `seal_verdict` return `(None, why)` so the existing helpers
# could see it. That is option 1 from this module's header with the sign flipped -- making
# the production path worse so the screen goes green -- and the caller genuinely needs the
# record back: `attribute()` returns it and `render()` prints from it. So the second
# polarity source gets a second shape, enforced exactly as strictly.


def withholds(record: Any, *, because: str = "") -> str:
    """Assert a verdict gate WITHHELD a green verdict, and return what it withheld.

    The reject polarity. ``record`` must carry ``withheld_from`` (the verdict it took away)
    and a non-empty ``withheld_because``. A gate that let a refusing record keep its green
    token makes the test red -- which is the polarity that catches a gate wired to nothing.
    """
    ctx = f" ({because})" if because else ""
    if not isinstance(record, dict):
        raise PolarityError(
            f"withholds{ctx}: a verdict gate must return the record it sealed; got "
            f"{type(record).__name__} {record!r}.")
    withheld = record.get("withheld_from")
    if withheld is None:
        raise PolarityError(
            f"expected the verdict to be WITHHELD{ctx}, but the record still publishes "
            f"{record.get('verdict')!r} with refusals {record.get('refusals')!r}. A record "
            f"that refuses and calls itself measured is the defect this gate exists for.")
    why = record.get("withheld_because")
    if not why:
        raise PolarityError(
            f"the verdict was withheld{ctx} but the record does not say which refusals cost "
            f"it ({why!r}). A verdict withdrawn without a reason is indistinguishable from "
            f"one that was never issued.")
    return withheld


def publishes(record: Any, expected: Any, *, because: str = "") -> Any:
    """Assert a verdict gate LEFT a clean verdict standing, and return it.

    The accept polarity. A gate that withholds from everything screens nothing, and it would
    be far worse than no gate: every reader would learn to ignore the token.
    """
    ctx = f" ({because})" if because else ""
    if not isinstance(record, dict):
        raise PolarityError(
            f"publishes{ctx}: a verdict gate must return the record it sealed; got "
            f"{type(record).__name__} {record!r}.")
    if "withheld_from" in record:
        raise PolarityError(
            f"expected the verdict to STAND{ctx}, but the gate withheld "
            f"{record['withheld_from']!r} citing {record.get('withheld_because')!r}. This is "
            f"the polarity that catches a gate which fires on everything.")
    got = record.get("verdict")
    if got != expected:
        raise PolarityError(
            f"the gate left {got!r} standing{ctx}, expected {expected!r}.")
    return got
