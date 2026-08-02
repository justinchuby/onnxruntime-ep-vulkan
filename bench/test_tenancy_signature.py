"""What foreign GPU work does and does not do to ``gpu_steady_tail`` — locked against the artifact.

THE HYPOTHESIS I WAS ASKED TO CHECK, AND IT IS FALSE
======================================================
    "``PER_DISPATCH`` is the signature that foreign GPU work should produce."

Offered with two traces that agree: ``contended3`` (``FOREIGN_GPU_WORK``, ``PER_DISPATCH``) and
``ab_p1_long`` (``SOLE_TENANT``, ``SUBMISSION_LEVEL``). Both hold. Nine device-0 traces carry both a
localisation and a committed tenancy companion, and over all nine the correspondence collapses:

    contended3          FOREIGN   explained_by_level  -0.2265
    base_b              SOLE                          -0.1039
    baseline_certified  SOLE                          -0.0119
    after_coldboard     SOLE                          +0.5227
    soloA               SOLE                          +0.5413
    ab_p1_r1            SOLE                          +0.6804
    ab_p0_r1            FOREIGN                       +0.8408
    contended           FOREIGN                       +0.8638
    ab_p1_long          SOLE                          +0.8895

The three witnessed-foreign traces sit at the **extreme bottom and near the extreme top**. The
classes are interleaved, so this is not a boundary that is in the wrong place — sweeping every cut
from -0.50 to +1.00 tops out at Youden's J = +0.333 (7 of 9), achieved only by catching
``contended3`` alone. ``contended`` is ``foreign_sample_fraction = 1.0`` and scores +0.8638, more
"submission-level" than four of the six sole-tenant traces.

Per the threshold episode: the sweep is published in full and **no cut is chosen**, because the
finding is about the signal and not about the boundary.

WHAT THE TAIL ACTUALLY DOES UNDER FOREIGN WORK, WHICH IS THE QUESTION ABOUT MY INSTRUMENT
===========================================================================================
Its **level moves**. It is not level-blind to contention the way it is level-blind to uniform bias:
``contended`` 11.7697 ms and ``ab_p0_r1`` 20.6159 ms against a sole-tenant reference of 11.5248 ms,
and ``contended3`` truncated reads 126.6465 ms — 10.99x.

Its **verdict does not follow**. Dispersion is what the verdict is computed from, and a foreign load
that is *sustained and steady* produces a steady wrong level:

    contended3 truncated to 20  STEADY  126.6465 ms  RSD 0.9103%
    contended3 truncated to 28  STEADY  126.6465 ms  RSD 0.8035%
    contended3 truncated to 34  STEADY  126.6758 ms  RSD 0.7915%
    contended3 full (62)        NO_STEADY_TAIL

So the refusal at full length is a property of **how long the run was**, not of the instrument's
sensitivity. This is a second failure mode beside uniform bias, and it deserves stating in its own
words:

**`gpu_steady_tail` detects foreign work only through its non-stationarity, never through its
magnitude. A steady foreign load is indistinguishable from a slower GPU.**

That is a strictly larger hole than "it cannot see a bias", because a sustained foreign tenant is a
far more ordinary condition than a board pinned at idle clock.

AND THE DIRECTION THIS GOES: THE COMPANION IS DOING MORE THAN WE CREDITED IT FOR
=================================================================================
Every one of the four ``contended3`` readings above — including the three that publish a confident
STEADY at 10.99x wrong — is refused by ``device_companion.certify``, because
``foreign_sample_fraction = 1.0`` is evidence from **outside** the series and does not care how
steady the series looks. The companion is not a diagnostic that happens to agree with the tail. On
this specimen it is the only thing between us and a certified 126 ms.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
_SYS_PATH_BEFORE = list(sys.path)
sys.path.insert(0, str(HERE))
try:
    import device_companion  # noqa: E402
    import phases  # noqa: E402
finally:
    sys.path[:] = _SYS_PATH_BEFORE

ARTIFACT = HERE / "results" / "tenancy_signature.json"


def _report() -> dict:
    if not ARTIFACT.is_file():
        pytest.skip(f"{ARTIFACT.name} not present; run bench/results/probe_tenancy_signature.py")
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_per_dispatch_does_not_identify_foreign_gpu_work_at_the_published_cut():
    """The two-trace correspondence holds and the nine-trace one does not."""
    c = _report()["correspondence"]
    assert c["n_with_both_signals"] >= 9, "the census shrank; recheck before trusting this"
    counts = c["counts"]
    assert counts["foreign_but_not_per_dispatch"] >= 2, (
        "witnessed foreign work that is not PER_DISPATCH is the falsifier; if this reaches 0 the "
        "hypothesis has become supportable and this test should be revisited, not deleted")
    assert counts["sole_but_per_dispatch"] >= 2, (
        "PER_DISPATCH also fires on sole-tenant traces, so it is not specific either")
    assert c["sensitivity"] is not None and c["sensitivity"] < 0.5


def test_no_cut_of_explained_by_level_separates_tenancy():
    """The stronger claim, and the reason no boundary needs choosing.

    Reporting failure at one cut invites "move the cut". The sweep answers that in advance.
    """
    bs = _report()["character_boundary_sweep"]
    best = bs["best_cut_by_youden_j"]
    assert best["youden_j"] <= 0.5, (
        f"best achievable separation over every cut from -0.50 to +1.00 is J={best['youden_j']} "
        f"at cut={best['cut']} ({best['correct']}/{best['of']}). If this ever exceeds 0.5 the "
        f"signal has become worth a boundary and the sweep must be republished, not tightened.")
    vals = bs["values_by_trace"]
    foreign_positions = [i for i, e in enumerate(vals)
                         if e["tenancy"] == "FOREIGN_GPU_WORK"]
    assert min(foreign_positions) == 0 and max(foreign_positions) >= len(vals) - 3, (
        "the finding is that the classes interleave: witnessed-foreign traces occupy both the "
        "bottom and the top of the ordering, which is what makes a threshold impossible")


def test_a_steady_foreign_load_publishes_a_confident_and_grossly_wrong_tail():
    """The specimen. This is the second failure mode, beside uniform bias.

    Kept as an artifact assertion rather than prose so that a future change to `gpu_steady_tail`
    that claims to fix this has to move this number.
    """
    sweep = _report()["contended3_truncation"]["sweep"]
    published = [s for s in sweep if s["verdict"] == "STEADY"]
    assert published, "contended3 truncated used to publish; if it no longer does, say why"
    for s in published:
        assert s["median_ms"] > 100.0, (
            f"published {s['median_ms']} ms at n<={s['truncated_to']} against a sole-tenant "
            f"reference of ~11.52 ms")
        assert s["rsd"] < 0.01, (
            "and it published with an RSD under 1% -- precision is not accuracy, and here the "
            "wrong number again carries the better RSD")
    refused = [s for s in sweep if s["verdict"] == "NO_STEADY_TAIL"]
    assert refused, "the full-length run refuses; the refusal is about length, not sensitivity"


def test_the_companion_refuses_every_reading_the_tail_published():
    """The direction this goes: obligation 8's companion is load-bearing, not corroborating.

    Recomputed here rather than read from the artifact, so the assertion exercises the real
    certification path against the real committed companion.
    """
    comp_path = HERE / "results" / "gpustate_contended3.json"
    if not comp_path.is_file():
        pytest.skip("gpustate_contended3.json not present")
    comp = json.loads(comp_path.read_text(encoding="utf-8"))
    assert comp["verdict"] == "FOREIGN_GPU_WORK" and comp["foreign_sample_fraction"] == 1.0

    for median, n in ((126.6465, 16), (126.6758, 30)):
        tail = {"verdict": "STEADY", "median_ms": median, "rsd": 0.008, "n": n, "coverage": 0.8}
        out = device_companion.certify(tail, comp)
        assert out["quotable"] is False
        assert out["verdict"] == device_companion.WITHHELD, (
            "a STEADY tail with an excellent RSD, refused on evidence from outside the series")


def test_the_tail_level_does_move_under_foreign_work():
    """Distinguishes this failure mode from the uniform-bias one, which the level cannot see.

    If the level did *not* move, foreign work would be invisible to the tail in every respect.
    It moves; only the verdict fails to follow. That distinction is the whole finding.
    """
    tm = _report()["tail_movement"]
    ref = tm["sole_tenant_reference_ms"]
    assert ref is not None and 11.0 < ref < 12.0
    assert tm["reference_built_from"], "the reference must name its sources"
    moved = [e for e in tm["foreign_gpu_work"]
             if e.get("x_vs_sole_tenant") and e["x_vs_sole_tenant"] > 1.01]
    assert moved, "witnessed foreign work moved no tail level at all -- that would be a new finding"


def test_the_reference_is_not_built_from_withheld_medians():
    """A withheld MARGINAL_TAIL median used as a denominator is that median published.

    The first cut of the probe took the median over all four sole-tenant levels, two of which were
    withheld, and produced a reference of 15.5159 ms that exists nowhere. ERROR(instrument).
    """
    tm = _report()["tail_movement"]
    withheld = {e["tag"] for e in tm["sole_tenant"] if e["level_is_withheld"]}
    assert withheld, "the census used to contain withheld sole-tenant tails; recheck if it does not"
    assert not (withheld & set(tm["reference_built_from"])), (
        f"a withheld median leaked into the reference: {withheld & set(tm['reference_built_from'])}")
