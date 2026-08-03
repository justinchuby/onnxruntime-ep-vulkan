"""Falsifiers for the depth axis of criterion 10's ULP curve. No GPU, no model, no ORT.

WHY THIS FILE EXISTS
====================
Morpheus's prediction is stated **per layer** — "flat at 1-3 across all 32 layers; flat =>
no defect, a step => a located one, and the layer index is the location".  Scoring it
requires knowing which output is which layer, and getting that wrong does not produce an
error: it produces a *different, plausible, reproducible curve* that is wrong on every
device.

I produced exactly that curve before writing these arms.  ``criterion10-dev*.json`` is
serialised with ``sort_keys=True``, so its ``output_coverage.per_output`` dict is
alphabetised; zipped against the ULP curve it puts ``present.9.value`` at index 64 and
``present.31.value`` at index 52, and the resulting reading was "the residual peaks at
layer 9 and falls back" — a located defect in the first third of the model, on both
vendors, entirely an artefact of a sorted container.  The correction came from asking the
session for its output order rather than the artifact.

These arms pin the mapping, the refusal, and the two readings' difference, so the mistake
cannot be made again silently.
"""

from __future__ import annotations

import pytest

import _kv_depth as d
import _models as m


# -- the mapping ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "index,expected",
    [
        (0, None),  # logits is not a layer
        (1, (0, "key")),
        (2, (0, "value")),
        (3, (1, "key")),
        (51, (25, "key")),
        (63, (31, "key")),
        (64, (31, "value")),
    ],
)
def test_the_output_index_maps_to_depth_because_the_session_order_is_depth_order(index, expected):
    assert d.layer_of_index(index) == expected


def test_the_name_mapping_and_the_index_mapping_agree_on_a_real_session_order():
    names = ["logits"] + [f"present.{L}.{p}" for L in range(32) for p in ("key", "value")]
    assert len(names) == 65
    for i, name in enumerate(names):
        assert d.layer_of_name(name) == d.layer_of_index(i), (i, name)


# -- the refusal ---------------------------------------------------------------------


def test_an_alphabetised_name_list_is_refused_rather_than_used():
    """The exact input that produced the fictitious layer-9 peak."""
    names = sorted(
        ["logits"] + [f"present.{L}.{p}" for L in range(32) for p in ("key", "value")]
    )
    with pytest.raises(ValueError, match="lexicographic order"):
        d.assert_names_are_session_order(names)


def test_the_true_session_order_is_accepted():
    names = ["logits"] + [f"present.{L}.{p}" for L in range(32) for p in ("key", "value")]
    assert d.assert_names_are_session_order(names) is names


def test_the_refusal_is_only_possible_because_the_two_orders_differ():
    """The falsifier's own precondition, asserted rather than assumed.

    If this model's session order happened to equal its lexicographic sort, the check
    above would reject every correct input and could not distinguish anything. It works
    for this model precisely because ``present.10`` sorts before ``present.2``.
    """
    names = ["logits"] + [f"present.{L}.{p}" for L in range(32) for p in ("key", "value")]
    assert names != sorted(names)
    assert sorted(names).index("present.10.key") < sorted(names).index("present.2.key")


def test_a_short_list_is_not_refused_because_the_tell_does_not_apply():
    """Two layers sort into their own order, so the tell carries no information there.

    Reported rather than tightened: a check that fires where it cannot discriminate is a
    red instrument, and this project has already been bitten by one.
    """
    assert d.assert_names_are_session_order(["logits", "present.0.key", "present.0.value"])


# -- the two readings are different, which is the whole point -------------------------


def test_the_sorted_reading_and_the_session_reading_disagree_on_the_same_data():
    """Ground truth for the near-miss: same medians, two orders, two different findings.

    Without this arm the refusal above proves only that a checker rejects a string list.
    Here the *conclusion* moves: the measured curve's peak lands on layer 31 read in
    session order and on layer 9 read in alphabetised order.
    """
    # The measured dev0/dev1 medians, in session (depth) order.
    medians = [12.0] + [
        v
        for L in range(32)
        for v in ((4.0, 4.0) if L == 31 else (3.0, 3.0) if L == 30 else (1.0, 1.0))
    ]
    session_names = ["logits"] + [
        f"present.{L}.{p}" for L in range(32) for p in ("key", "value")
    ]
    alphabetised = sorted(session_names)

    correct = d.depth_curve(medians, session_names)
    peak_correct = max(correct, key=lambda r: r["key"])
    assert peak_correct["layer"] == 31

    # The wrong reading, produced deliberately by bypassing the refusal.
    by_layer = {}
    for name, v in zip(alphabetised, medians):
        parsed = d.layer_of_name(name)
        if parsed:
            by_layer.setdefault(parsed[0], {})[parsed[1]] = v
    peak_wrong = max(by_layer.items(), key=lambda kv: kv[1].get("key", 0))
    assert peak_wrong[0] != 31, (
        "the alphabetised reading agreed with the session reading, so this specimen does "
        "not demonstrate the hazard and the refusal above is untested against a finding"
    )


# -- the band and the step ------------------------------------------------------------


def test_the_layer_band_is_the_same_number_as_the_output_band_and_neither_derives_the_other():
    """Two constants, equal today, deliberately not defined in terms of each other.

    Deriving one from the other means a change to a per-output threshold silently moves a
    per-layer prediction that a different person made in different units.
    """
    assert d.LAYER_PREDICTED_CEILING == m.ULP_PREDICTED_CEILING


def test_an_exceedance_reports_the_layer_because_the_layer_is_the_location():
    curve = d.depth_curve([0.0] + [1.0] * 62 + [4.0, 4.0])
    exceed = d.depth_exceedances(curve)
    assert [(e["layer"], e["part"]) for e in exceed] == [(31, "key"), (31, "value")]


def test_a_smooth_climb_and_a_step_are_told_apart_by_the_step_size_not_by_the_band():
    """Both exceed the band; only one is a located defect. The number is reported, not thresholded."""
    smooth = d.depth_curve(
        [0.0] + [v for L in range(32) for v in (min(1.0 + L / 12.0, 4.0),) * 2]
    )
    step = d.depth_curve([0.0] + [v for L in range(32) for v in ((12.0,) * 2 if L >= 20 else (1.0,) * 2)])
    assert d.depth_exceedances(smooth)
    assert d.depth_exceedances(step)
    assert d.largest_step(step, "key")["step"] > 5 * d.largest_step(smooth, "key")["step"]
    assert d.largest_step(step, "key")["to_layer"] == 20


def test_key_and_value_are_not_averaged_together():
    """A two-element mean is a reduction that can be dominated by one of its two terms.

    Same fault as an aggregate over the 65 outputs, one level down — and it would hide a
    defect confined to the value projection behind a clean key.
    """
    curve = d.depth_curve([0.0] + [1.0, 9.0] + [1.0] * 62)
    assert curve[0] == {"layer": 0, "key": 1.0, "value": 9.0}
    assert [e["part"] for e in d.depth_exceedances(curve)] == ["value"]
