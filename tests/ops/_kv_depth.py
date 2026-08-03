"""The KV residual **by layer depth** — and the trap I fell into reading it off the record.

THE AXIS IS THE OUTPUT INDEX, AND I NEARLY PUBLISHED THAT IT WASN'T
==================================================================
Phi-3.5's session output order, read off the session (``sess.get_outputs()``), is depth
order::

    0 logits, 1 present.0.key, 2 present.0.value, 3 present.1.key, ... 64 present.31.value

so output ``1 + 2*L + p`` is layer ``L``'s key (``p=0``) or value (``p=1``), and the ULP
curve indexed by output *is* a depth curve.  Morpheus's "flat at 1-3 across all 32 layers"
is therefore scored on the axis it was stated on.

**I spent an hour concluding the opposite, and the cause was in the record, not in the
model.**  ``bench/results/criterion10-dev*.json`` is serialised with
``json.dumps(..., sort_keys=True)``, so ``output_coverage.per_output`` — a dict keyed by
output name — is written **alphabetised**: ``present.0.key, present.0.value,
present.1.key, present.1.value, present.10.key, ...``.  Zipping that key order against the
ULP curve puts ``present.9.value`` at index 64 and ``present.31.value`` at index 52, which
manufactures a curve rising to a peak at "layer 9" and falling back — a located defect in
the first third of the model, on both vendors, entirely fictitious.

It was caught by asking the session for its output order instead of asking the artifact.
The general form is R12's: the frame of a name is the run that produced it, not the file
that stored it, and a sorted container has silently discarded the only property it was
being read for.  :func:`assert_names_are_session_order` is the falsifier, and it is cheap
because Phi-3.5's true order is *not* its own sort — so a name list equal to
``sorted(names)`` is a positive tell that a sorted container was read as an ordering.

WHAT THE CURVE ACTUALLY SAYS (dev0 and dev1 byte-identical, both writeback routes)
=================================================================================
::

    layer   0 .. 23 : 0-1 ULP throughout (a single 2 at layer 17, key)
    layer  24 .. 29 : 1-2
    layer  30       : 3
    layer  31       : 4   <- key and value both; the only band exceedance in the KV cache

Rising, smoothly, no discontinuity: a one-ULP overshoot of the predicted 1-3 band at the
deepest layer, not a step.  The step in this model is output 0 (the logits) at 12 ULP, and
that is not a layer at all.
"""

from __future__ import annotations

import re

_LAYER_RE = re.compile(r"^present\.(\d+)\.(key|value)$")

#: Morpheus's band, in the units he stated it in — per layer.  Deliberately a separate
#: constant from ``_models.ULP_PREDICTED_CEILING`` so that changing one cannot silently
#: move the other; that they are equal today is asserted in the tests.
LAYER_PREDICTED_CEILING = 3.0


def layer_of_name(output_name: str) -> "tuple[int, str] | None":
    """``present.9.key`` -> ``(9, 'key')``; anything else (e.g. ``logits``) -> ``None``."""
    match = _LAYER_RE.match(output_name or "")
    if not match:
        return None
    return int(match.group(1)), match.group(2)


def layer_of_index(index: int) -> "tuple[int, str] | None":
    """Layer and part for a Phi-3.5 output **index**, in session order.

    Index 0 is the logits and is not a layer.  This is the mapping the ULP curve is
    indexed by, and the one to use when the record carries no explicit ordered name list.
    """
    if index <= 0:
        return None
    zero_based = index - 1
    return zero_based // 2, ("key" if zero_based % 2 == 0 else "value")


def assert_names_are_session_order(names: list[str]) -> list[str]:
    """Refuse a name list that is really a sorted container's key order.

    Phi-3.5's session output order is **not** its own lexicographic sort (``present.10``
    sorts before ``present.2``), so ``names == sorted(names)`` is a positive tell that the
    caller read an alphabetised JSON object and took its key order for an ordering.  This
    is ERROR(instrument) territory: everything downstream is attributed to the wrong layer,
    silently and reproducibly, on every device.
    """
    layered = [n for n in names if _LAYER_RE.match(n)]
    if len(layered) > 2 and layered == sorted(layered):
        raise ValueError(
            "these output names are in lexicographic order, which is not this model's "
            "session output order — they almost certainly came from a JSON object "
            "serialised with sort_keys=True. Read the order from the run "
            "(sess.get_outputs()) or from the record's explicit output_names list."
        )
    return names


def depth_curve(
    medians: "list[float | None]", names: "list[str] | None" = None
) -> list[dict]:
    """Per-layer ULP medians in depth order, one row per layer, key and value apart.

    Key and value are kept apart rather than averaged.  They come from different
    reductions, and a two-element mean is a reduction that can be dominated by one of its
    two terms — the same fault as an aggregate over the 65 outputs, one level down.

    *names*, when given, is used for the mapping and is checked by
    :func:`assert_names_are_session_order` first; otherwise the index mapping is used.
    """
    if names is not None:
        assert_names_are_session_order(names)

    by_layer: dict[int, dict] = {}
    for i, median in enumerate(medians):
        if names is not None and i < len(names):
            parsed = layer_of_name(names[i])
        else:
            parsed = layer_of_index(i)
        if parsed is None:
            continue
        layer, part = parsed
        by_layer.setdefault(layer, {"layer": layer, "key": None, "value": None})[part] = median
    return [by_layer[k] for k in sorted(by_layer)]


def depth_exceedances(
    curve: list[dict], ceiling: float = LAYER_PREDICTED_CEILING
) -> list[dict]:
    """Layers whose key or value median exceeds the band predicted before measuring.

    The layer index is reported because it is the *location*: the prediction's own terms
    were "flat => no defect; a step => a located one", and a located one is located by
    this number.
    """
    out = []
    for row in curve:
        for part in ("key", "value"):
            v = row.get(part)
            if isinstance(v, (int, float)) and v > ceiling:
                out.append({"layer": row["layer"], "part": part, "median_ulp_diff": v})
    return out


def largest_step(curve: list[dict], part: str = "key") -> dict:
    """The biggest layer-to-layer jump, which is what "a step" means on this axis.

    Reported rather than thresholded: a curve that climbs by one ULP six times and a curve
    that jumps eight ULPs once both "exceed the band", and only the second is a located
    defect.  Nobody gets to pick the threshold after seeing the number.
    """
    seen = [
        (row["layer"], row[part])
        for row in curve
        if isinstance(row.get(part), (int, float))
    ]
    if len(seen) < 2:
        return {"step": None, "why": "fewer than two comparable layers"}
    jumps = [
        {"from_layer": a[0], "to_layer": b[0], "step": b[1] - a[1]}
        for a, b in zip(seen, seen[1:])
    ]
    return max(jumps, key=lambda j: j["step"])
