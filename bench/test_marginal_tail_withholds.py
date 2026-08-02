"""`MARGINAL_TAIL` must never publish a median. This file is the reason why.

Someone will eventually propose relaxing this to get more data points -- 12 of 28 traces publish,
and four more would if `MARGINAL_TAIL` reported its median. **This is what answers them.**

THE ARTIFACT
============

`trace_gemv_contended_dev0.json`, in the 28-trace dev0 census (`bench/results/frames.json`,
reproducible via `bench/results/probe_frames.py`):

===============================  ===========
whole-series RSD                 **129.51%**
whole-series spread              10.56x
tail verdict                     MARGINAL_TAIL (n=8, coverage 17.39%)
**tail RSD**                     **0.1067%**
device-state companion           **FOREIGN_GPU_WORK**
===============================  ===========

**It is the most disturbed run in the entire census, and its tail RSD is the third tightest of all
28 traces** -- tighter than `baseline_certified` (0.1163%), tighter than `solo` (0.1628%), tighter
than `warmup` (0.1474%), a run that sat within 1.01x of itself end to end.

Had `MARGINAL_TAIL` published its median, we would have certified our dirtiest run on the strength
of the tightest-looking number the project has ever produced -- and the number would have been an
8-sample flat patch inside a series that moved by a factor of ten.

WHY THIS IS NOT ORDINARY CONSERVATISM
=====================================

The withheld median is not "a number we are being careful about". **The tightness of a short tail
is evidence about the length of the window, not about the state of the device.** Any series that
moves at all contains a short stretch that does not; the more disturbed the series, the more such
stretches it contains and the flatter the flattest of them will be. **Selecting the flattest 17%
of a 10x-spread run and reporting its RSD is a search, and the RSD of a search result is not the
RSD of a measurement.**

That is why `withheld_median_ms` exists as a distinct key: the value is retained for diagnosis and
is not addressable as a median by any reader or any aggregation.

THIS IS THE THIRD INDEPENDENT APPEARANCE OF ONE PATTERN
=======================================================

1. `contended3` truncated -- STEADY, 126.647 ms, **10.99x wrong**, RSD 0.79%.
2. `baseline_certified` -- STEADY, 246.720 ms, **21.4x wrong**, RSD **0.1163%**, zero discarded.
3. `contended` here -- 129.51% whole-series, tail RSD **0.1067%**, third tightest of 28.

In every one, **the number we should least have trusted looked the most precise.** Precision is
not accuracy, and on this instrument precision is *anti-correlated* with accuracy (DESIGN.md
§10.0.1 R9 amendment 5), because the mechanisms that bias a series -- a board stuck at its idle
clock, a foreign tenant, a short selected window -- all *reduce* dispersion.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ORIGINAL_SYS_PATH = list(sys.path)
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import phases  # noqa: E402
finally:
    sys.path[:] = _ORIGINAL_SYS_PATH

CENSUS = Path(__file__).resolve().parent / "results" / "frames.json"

#: The specimen, quoted from the committed census so the test fails if the census stops agreeing.
CONTENDED = "trace_gemv_contended_dev0.json"


def _census() -> "dict[str, dict]":
    if not CENSUS.exists():
        pytest.skip("bench/results/frames.json not present")
    doc = json.loads(CENSUS.read_text(encoding="utf-8"))
    return {r["trace"]: r for r in doc.get("census", [])}


def test_the_contended_specimen_still_looks_exactly_this_dangerous():
    """If these numbers drift, every argument built on them below needs re-checking."""
    row = _census()[CONTENDED]
    assert row["whole_series_rsd"] > 1.2, row
    assert row["tail_verdict"] == "MARGINAL_TAIL"
    assert row["tail_rsd"] < 0.0011, "the tail RSD is supposed to be alarmingly tight"
    assert row["tail_coverage"] < 0.2


def test_the_dirtiest_run_has_a_tighter_tail_than_the_cleanest_certified_one():
    """The single sentence this whole file exists to preserve."""
    c = _census()
    contended = c[CONTENDED]["tail_rsd"]
    ranked = sorted(r["tail_rsd"] for r in c.values() if r["tail_rsd"] is not None)
    assert ranked.index(contended) <= 2, (
        f"contended's tail RSD {contended:.4%} is no longer among the three tightest of "
        f"{len(ranked)}; the argument in this file's docstring needs restating, not deleting")
    for cleaner in ("trace_gemv_baseline_certified_dev0.json", "trace_gemv_solo_dev0.json",
                    "trace_gemv_warmup_dev0.json"):
        if cleaner in c and c[cleaner]["tail_rsd"] is not None:
            assert contended < c[cleaner]["tail_rsd"], (
                f"{cleaner} was supposed to look LESS precise than the 129% RSD run")


def test_marginal_tail_never_exposes_a_median():
    """The enforcement itself, on the shape of the specimen rather than on the specimen."""
    # A series that moves by 10x and contains one flat patch of 8 -- `contended`'s shape.
    series = [11500.0 + (i % 5) * 22000.0 for i in range(38)] + [11500.0 + 0.4 * (i % 3)
                                                                 for i in range(8)]
    tail = phases.gpu_steady_tail(series)
    assert tail["verdict"] == "MARGINAL_TAIL", tail["verdict"]
    assert tail["rsd"] < 0.002, "the flat patch is supposed to look extremely precise"
    assert tail["median_ms"] is None, "MARGINAL_TAIL must not publish a median"
    assert tail["withheld_median_ms"] is not None, "but it must retain it for diagnosis"


def test_no_publishing_verdict_is_reachable_from_a_marginal_tail(monkeypatch):
    """Guards the downstream contract, not just the field.

    Every consumer gates on `verdict == "STEADY"`. If someone adds `MARGINAL_TAIL` to a publishing
    set to recover data points, this fails rather than quietly releasing `contended`.
    """
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parent / "results"))
    try:
        import probe_frames
    except Exception:  # pragma: no cover
        pytest.skip("probe_frames not importable")
    assert "MARGINAL_TAIL" not in probe_frames.PUBLISHING_VERDICTS
    assert "NO_STEADY_TAIL" not in probe_frames.PUBLISHING_VERDICTS


def test_a_marginal_tail_is_uncertified_even_when_the_board_is_perfect():
    """The companion cannot rescue it, and must not be asked to.

    `contended`'s companion says FOREIGN_GPU_WORK, so it is refused twice over. But the withholding
    must not *depend* on that: nine of the twelve publishing traces have no companion at all, and a
    MARGINAL_TAIL on a verified-clean board is still not a number.
    """
    import device_companion as device_state
    perfect = {"verdict": "SOLE_TENANT", "sm_mhz": {"max": 2490.0}, "sm_max_mhz": 3105.0,
               "n": 40, "seconds": 20.0}
    out = device_state.certify({"verdict": "MARGINAL_TAIL", "median_ms": None,
                                "withheld_median_ms": 11.77, "n": 8, "coverage": 0.17}, perfect)
    assert out["quotable"] is False
    assert out["verdict"] == device_state.UNCERTIFIED
    assert "11.77" not in json.dumps(out), "the withheld median must not leak into the certificate"
