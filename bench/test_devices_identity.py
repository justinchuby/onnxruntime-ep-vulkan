"""Two-polarity + mutation battery for ``bench/devices.py::identify_by_uuid`` (issue #18).

WHY THIS FILE EXISTS
====================
The instrument census flagged ``identify_by_uuid`` ``unfalsified``.  That was a true
statement about the *screen*, not about the tests: two polarity tests for it already lived
in ``bench/test_phases.py``, and the screen could not see them because it reads reject
polarity from ``pytest.raises`` and this instrument never raises — it returns
``(None, why)``.

There were three ways to make the census green and only one of them is honest.  Forcing an
exception contract onto a deliberately total function makes production worse to make a
screen greener.  A hand note in the baseline converts an open question into a permanent
one.  What is done instead: ``bench/_polarity.py`` gives the screen a second polarity
source that *enforces* what it labels, and this file earns the credit that source hands
out — by mutation, the same way ``tests/ops/test_guard_d.py`` earns it for the harness
domain.

WHAT IS PROVED HERE
===================
1. **Accept polarity.**  An exact UUID selects exactly the right card, *even when two
   probed devices report the identical ``deviceName``*.  This is the issue #18 crux: a
   name is a MODEL name and collapses; a UUID is an identity and does not.
2. **Reject polarity, three documented cases, each distinguished from the others by the
   sentence it returns** — absent UUID, unmatched UUID, and a UUID matching more than one
   probed device (which is a ``probe()`` defect, not a real ambiguity).
3. **The planted controls.**  Five deliberately defective reimplementations, each a defect
   that has actually shipped in code of this shape, are put through the *identical*
   protocol as the real instrument.  Every one must be caught.  Without this, "the tests
   pass" says only that the tests pass.
4. **The polarity helpers are themselves two-polarity tested**, because a
   ``refuses``/``selects`` that accepted anything would launder every credit above.

No GPU, no model, no EP.  Always in the lane.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import devices  # noqa: E402
from _polarity import PolarityError, refuses, selects  # noqa: E402

# The UUID of the card that must be picked, and of one that is not present anywhere.
_A = "aa" * 16
_B = "bb" * 16
_ABSENT = "cc" * 16

# Two cards of the SAME MODEL. Every string a human would read off them is identical; only
# the UUID differs. This is the desk issue #18 exists for, and the reason a name-keyed
# identification is not an identification.
_MODEL_NAME = "NVIDIA RTX A1000 Laptop GPU"


def _card(uuid: str, index: int) -> "devices.DeviceFacts":
    return devices.DeviceFacts(
        index=index,
        name=_MODEL_NAME,
        uuid=uuid,
    )


@pytest.fixture()
def two_identical_cards() -> "tuple[devices.DeviceFacts, devices.DeviceFacts]":
    return _card(_A, 0), _card(_B, 1)


# ---------------------------------------------------------------------------
# 1. ACCEPT POLARITY
# ---------------------------------------------------------------------------


def test_an_exact_uuid_selects_that_card_even_among_identical_names(two_identical_cards):
    """The crux: identical ``deviceName``, distinct UUIDs, and the right one is picked.

    ``selects`` compares by ``is``, not ``==``.  These two records are equal on every field
    a naive equality test would look at except the UUID, so an ``==`` assertion here would
    pass against an instrument that returned the wrong card — which is the exact collapse
    this instrument exists to prevent.
    """
    gpu0, gpu1 = two_identical_cards
    got = selects(devices.identify_by_uuid([gpu0, gpu1], _B), gpu1)
    assert got.uuid == _B

    # And the other way round, so this is not an instrument that always returns the last
    # element of the list.
    selects(devices.identify_by_uuid([gpu0, gpu1], _A), gpu0)


def test_uuid_matching_is_case_insensitive_because_vulkan_hex_case_is_not_an_identity(
    two_identical_cards,
):
    """``AA..`` and ``aa..`` are the same device; different tools print different case."""
    gpu0, _ = two_identical_cards
    selects(devices.identify_by_uuid([gpu0], _A.upper()), gpu0)
    selects(devices.identify_by_uuid([gpu0], _A.lower()), gpu0)


# ---------------------------------------------------------------------------
# 2. REJECT POLARITY — three documented refusals, each distinguishable
# ---------------------------------------------------------------------------


def test_an_absent_uuid_is_refused_and_says_the_identity_was_never_recorded(
    two_identical_cards,
):
    gpu0, gpu1 = two_identical_cards
    for missing in (None, ""):
        why = refuses(
            devices.identify_by_uuid([gpu0, gpu1], missing),
            because=f"uuid={missing!r} — nothing was recorded to compare",
        )
        assert "no device uuid" in why, why


def test_an_unmatched_uuid_is_refused_and_is_not_the_same_sentence_as_an_absent_one(
    two_identical_cards,
):
    """Stale evidence must not read as missing evidence: different worlds, different words."""
    gpu0, gpu1 = two_identical_cards
    absent_why = refuses(devices.identify_by_uuid([gpu0, gpu1], None))
    unmatched_why = refuses(
        devices.identify_by_uuid([gpu0, gpu1], _ABSENT),
        because="a real uuid that names no probed device",
    )
    assert "matches NO probed device" in unmatched_why, unmatched_why
    assert unmatched_why != absent_why


def test_a_uuid_matching_two_probed_devices_is_refused_as_a_probe_defect(
    two_identical_cards,
):
    """A UUID is unique per physical device, so two matches means ``probe()`` double-counted.

    Refusing here rather than picking one is the whole point: this is the only case in which
    the instrument's input is self-contradictory, and a guess would be indistinguishable
    from a correct identification to every downstream reader.
    """
    gpu0, _ = two_identical_cards
    doubled = replace(gpu0, index=1)
    why = refuses(
        devices.identify_by_uuid([gpu0, doubled], _A),
        because="the same uuid appears twice in the probe",
    )
    assert "more than one probed device" in why, why


def test_a_card_that_reports_no_uuid_can_never_be_selected_by_one(two_identical_cards):
    """The no-UUID fallback contract: absence on the DEVICE side is also a refusal.

    A driver that does not implement ``VK_KHR_id_properties`` reports no UUID.  Such a card
    must not be matched by anything, and in particular an empty/None device uuid must not
    compare equal to an empty/None evidence uuid and quietly identify itself.
    """
    gpu0, _ = two_identical_cards
    uuidless = replace(gpu0, uuid=None)
    why = refuses(
        devices.identify_by_uuid([uuidless], _A),
        because="the probed card reports no uuid at all",
    )
    assert "matches NO probed device" in why, why


# ---------------------------------------------------------------------------
# 3. THE PLANTED CONTROLS — the protocol above must catch a defective instrument
# ---------------------------------------------------------------------------


def _mutant_case_sensitive(devs, uuid):
    """DEFECT: compares raw hex case. Real evidence mixes ``AA..`` and ``aa..``."""
    if not uuid:
        return None, "evidence carries no device uuid"
    m = [d for d in devs if d.uuid and d.uuid == uuid]
    if len(m) == 1:
        return m[0], "matched"
    return None, "matches NO probed device"


def _mutant_first_match_wins(devs, uuid):
    """DEFECT: resolves ambiguity by picking a side instead of refusing."""
    if not uuid:
        return None, "evidence carries no device uuid"
    m = [d for d in devs if d.uuid and d.uuid.lower() == uuid.lower()]
    if m:
        return m[0], "matched"
    return None, "matches NO probed device"


def _mutant_absent_uuid_matches_anything(devs, uuid):
    """DEFECT: a missing identity selects the only card, which is the name-keyed collapse."""
    if not uuid:
        return (devs[0], "assumed the only device") if devs else (None, "no devices")
    m = [d for d in devs if d.uuid and d.uuid.lower() == uuid.lower()]
    if len(m) == 1:
        return m[0], "matched"
    return None, "matches NO probed device"


def _mutant_falls_back_to_name(devs, uuid):
    """DEFECT: the rejected 11a7c69 shape — no match, so identify by model name instead."""
    if not uuid:
        return None, "evidence carries no device uuid"
    m = [d for d in devs if d.uuid and d.uuid.lower() == uuid.lower()]
    if len(m) == 1:
        return m[0], "matched"
    by_name = [d for d in devs if d.name == _MODEL_NAME]
    if by_name:
        return by_name[0], "matched on device name"
    return None, "matches NO probed device"


def _mutant_refuses_without_saying_why(devs, uuid):
    """DEFECT: refuses correctly and silently, so no caller can print the reason."""
    if not uuid:
        return None, ""
    m = [d for d in devs if d.uuid and d.uuid.lower() == uuid.lower()]
    if len(m) == 1:
        return m[0], "matched"
    return None, ""


_MUTANTS = {
    "case_sensitive": _mutant_case_sensitive,
    "first_match_wins": _mutant_first_match_wins,
    "absent_uuid_matches_anything": _mutant_absent_uuid_matches_anything,
    "falls_back_to_name": _mutant_falls_back_to_name,
    "refuses_without_saying_why": _mutant_refuses_without_saying_why,
}


def _run_protocol(fn, gpu0, gpu1) -> None:
    """The exact protocol the real instrument passes above, as one callable.

    Kept in one place on purpose: a mutation battery that runs a *weaker* protocol than the
    real tests proves nothing about the real tests.
    """
    selects(fn([gpu0, gpu1], _B), gpu1)
    selects(fn([gpu0, gpu1], _A), gpu0)
    selects(fn([gpu0], _A.upper()), gpu0)
    refuses(fn([gpu0, gpu1], None))
    refuses(fn([gpu0, gpu1], _ABSENT))
    refuses(fn([gpu0, replace(gpu0, index=1)], _A))
    refuses(fn([replace(gpu0, uuid=None)], _A))


def test_the_real_instrument_passes_the_protocol(two_identical_cards):
    """The control's control: the battery below means nothing if the real one fails here."""
    gpu0, gpu1 = two_identical_cards
    _run_protocol(devices.identify_by_uuid, gpu0, gpu1)


@pytest.mark.parametrize("mutant_name", sorted(_MUTANTS))
def test_a_defective_identify_by_uuid_is_caught_by_this_protocol(
    mutant_name: str, two_identical_cards
):
    """Every planted defect must fail the same protocol the real instrument passes.

    This is what makes the census's ``screened`` verdict for ``identify_by_uuid`` a claim
    about the instrument rather than a claim about the shape of its test file.
    """
    gpu0, gpu1 = two_identical_cards
    with pytest.raises(PolarityError):
        _run_protocol(_MUTANTS[mutant_name], gpu0, gpu1)


# ---------------------------------------------------------------------------
# 4. THE HELPERS THEMSELVES, IN BOTH POLARITIES
# ---------------------------------------------------------------------------


def test_refuses_accepts_a_genuine_refusal_and_returns_its_reason():
    assert refuses((None, "because reasons")) == "because reasons"


def test_refuses_rejects_an_identification_a_silent_refusal_and_a_non_tuple(
    two_identical_cards,
):
    gpu0, _ = two_identical_cards
    with pytest.raises(PolarityError):
        refuses((gpu0, "matched"))
    with pytest.raises(PolarityError):
        refuses((None, "   "))
    with pytest.raises(PolarityError):
        refuses(None)


def test_selects_accepts_the_right_card_and_returns_it(two_identical_cards):
    gpu0, _ = two_identical_cards
    assert selects((gpu0, "matched"), gpu0) is gpu0


def test_selects_rejects_a_refusal_the_wrong_card_and_an_equal_but_distinct_one(
    two_identical_cards,
):
    """The ``is``-not-``==`` clause gets its own falsifier, because it is the load-bearing one."""
    gpu0, gpu1 = two_identical_cards
    with pytest.raises(PolarityError):
        selects((None, "refused"), gpu0)
    with pytest.raises(PolarityError):
        selects((gpu1, "matched"), gpu0)
    twin = replace(gpu0)
    assert twin == gpu0, "the fixture no longer produces an equal-but-distinct record"
    with pytest.raises(PolarityError):
        selects((twin, "matched"), gpu0)
