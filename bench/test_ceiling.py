"""Locks for the count-based ceiling, the continuous clock record, and the screen that
stops the cumulative-counter quotient from happening again.

Three subjects, all consequences of one standing change: the box is shared with another team
and will stay contended, so wall-clock figures are `STEADY_UNCERTIFIED` by default and
performance work runs on counts (`docs/PERF.md` §20).

The screen at the bottom is the important one for the future. My fragmentation reading found
the denominator -- a cumulative counter dominated by a one-time ~2185 MiB weight upload,
divided by two different iteration counts, 51/28 = 1.82. On a permanently contended box run
lengths vary with whatever else is on the machine, so that shape is now the most likely error
in the repository. A checklist decays; a mechanical screen with a positive control does not.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parent

_SYS_PATH_BEFORE = list(sys.path)
sys.path.insert(0, str(BENCH))
try:
    import ceiling as ceiling_mod  # noqa: E402
    import clock_log  # noqa: E402
    import device_companion  # noqa: E402
finally:
    sys.path[:] = _SYS_PATH_BEFORE


@pytest.fixture(scope="module")
def ceil_():
    try:
        return ceiling_mod.Ceiling.load()
    except ceiling_mod.CeilingError as exc:
        pytest.skip(str(exc))


class TestCeilingHasAFrame:
    def test_the_frame_names_the_binary(self, ceil_):
        """A bound compared against a figure from another binary is the mistake R12's fourth
        generalisation is about: for a test result, the frame is the binary that ran it."""
        f = ceil_.frame()
        assert f["dll_sha256"] is None or len(f["dll_sha256"]) == 64
        assert f["device"]
        assert f["peak_GB_per_s"] == 256.0
        assert f["peak_source"]

    def test_the_frame_names_where_every_input_came_from(self, ceil_):
        f = ceil_.frame()
        for key in ("byte_model_source", "by_context_source", "claim_source", "model"):
            assert f.get(key), key

    def test_it_publishes_what_it_is_silent_about(self, ceil_):
        s = ceil_.silence_set()
        assert len(s) >= 4
        joined = " ".join(s).lower()
        # The bound must disclaim the currency §19.6 left unpriced.
        assert "drain" in joined or "synchronis" in joined
        assert "spec peak" in joined


class TestCeilingHasAnExtentAndRefuses:
    def test_the_only_admissible_context_is_zero_on_this_build(self, ceil_):
        """GroupQueryAttention is declined and runs on CPU, so KV-cache bytes are not GPU
        traffic. At past_len 0 the KV term is exactly zero and the question does not arise."""
        ext = ceil_.extent()
        assert ext["admissible"] == [0]
        assert 8192 in ext["grid"]

    def test_zero_context_is_a_bound(self, ceil_):
        r = ceil_.floor_ms(0)
        assert r["state"] == ceiling_mod.BOUND
        assert r["kv_cache_MiB"] == 0.0
        assert 8.0 < r["floor_ms_at_spec_peak"] < 8.5

    def test_every_other_context_is_UNOBSERVABLE_and_not_a_number(self, ceil_):
        for past in (128, 512, 2048, 4096, 8192):
            r = ceil_.floor_ms(past)
            assert r["state"] == ceiling_mod.UNOBSERVABLE, past
            assert "floor_ms_at_spec_peak" not in r, past

    def test_UNOBSERVABLE_is_not_zero_and_says_why(self, ceil_):
        """Reporting 0 would claim the KV traffic is free. It is not free, it is on the CPU."""
        r = ceil_.floor_ms(8192)
        assert "why_not_zero" in r
        assert "free" in r["why_not_zero"]
        # and it must still disclose the number it declined to publish, so the refusal is
        # auditable rather than merely opaque
        assert r["would_have_reported_ms"] > 20.0

    def test_it_will_not_extrapolate_off_its_own_grid(self, ceil_):
        r = ceil_.floor_ms(777)
        assert r["state"] == ceiling_mod.UNOBSERVABLE
        assert "grid" in r["reason"]

    def test_a_build_with_no_claim_record_has_an_empty_extent(self, tmp_path):
        """Unknown claim status is not permission. Absent evidence yields no admissible
        context at all, rather than defaulting to the whole grid."""
        c = ceiling_mod.Ceiling.load(claim_record=tmp_path / "nope.json")
        assert c.extent()["admissible"] == []
        assert c.floor_ms(0)["state"] == ceiling_mod.UNOBSERVABLE


class TestCeilingComparison:
    def test_the_quotable_figure_sits_at_the_one_admissible_context(self, ceil_):
        """12.1847 ms is zero context, one token, and zero context is the only place the
        bound holds. That is why the comparison survives."""
        cmp = ceil_.compare(12.1847, past_len=0)
        assert cmp["quotable"] is True
        assert 0.6 < cmp["fraction_of_roofline"] < 0.7
        assert 1.4 < cmp["headroom_x"] < 1.6

    def test_a_comparison_at_an_inadmissible_context_is_refused(self, ceil_):
        cmp = ceil_.compare(12.1847, past_len=8192)
        assert cmp["quotable"] is False
        assert cmp["state"] == ceiling_mod.UNOBSERVABLE

    def test_a_quotable_comparison_carries_its_context_forward(self, ceil_):
        """The roofline is not a constant, so a fraction-of-roofline without a context is
        the same defect as a timing without its device state."""
        cmp = ceil_.compare(12.1847, past_len=0)
        assert "past_sequence_length=0" in cmp["must_be_quoted_with"]
        assert "not a constant" in cmp["must_be_quoted_with"]


class TestContinuousClockRecord:
    def test_a_window_with_too_few_samples_is_UNOBSERVABLE_not_a_pass(self, tmp_path):
        log = tmp_path / "empty.jsonl"
        log.write_text("", encoding="utf-8")
        w = clock_log.window("2026-08-02T00:00:00+00:00", "2026-08-02T01:00:00+00:00", log)
        assert w["verdict"] == clock_log.UNOBSERVABLE
        assert "unrecorded, not quiet" in w["reason"]

    def test_an_inverted_window_is_an_instrument_error_not_a_verdict(self, tmp_path):
        log = tmp_path / "x.jsonl"
        log.write_text("", encoding="utf-8")
        w = clock_log.window("2026-08-02T01:00:00+00:00", "2026-08-02T00:00:00+00:00", log)
        assert w["verdict"] == clock_log.UNOBSERVABLE
        assert "ERROR(instrument)" in w["reason"]

    def test_a_populated_window_produces_a_record_device_companion_accepts(self, tmp_path):
        """The point of the log is retrospective certification, so its output must be the
        shape certify() consumes -- not a second dialect for the same channel."""
        log = tmp_path / "log.jsonl"
        rows = []
        for i in range(clock_log.MIN_WINDOW_SAMPLES + 5):
            rows.append(json.dumps({
                "wall": f"2026-08-02T12:00:{i % 60:02d}.{i:03d}000+00:00",
                "t": float(i) * 0.25, "sm_mhz": 2010.0, "sm_max_mhz": 3105.0, "util_pct": 90.0,
                "power_w": 40.0, "temperature_c": 60.0, "apps": [],
            }))
        log.write_text("\n".join(rows), encoding="utf-8")
        w = clock_log.window("2026-08-02T12:00:00+00:00", "2026-08-02T12:01:00+00:00", log)
        assert w.get("retrospective") is True
        assert "sm_mhz" in w and "verdict" in w
        tail = {"verdict": "STEADY", "median_ms": 12.0, "n": 30}
        cert = device_companion.certify(tail, w)
        assert "quotable" in cert

    def test_a_retrospective_window_declares_that_it_is_weaker(self, tmp_path):
        log = tmp_path / "log.jsonl"
        rows = [json.dumps({
            "wall": f"2026-08-02T12:00:{i % 60:02d}.{i:03d}000+00:00",
            "t": float(i) * 0.25, "sm_mhz": 2010.0, "sm_max_mhz": 3105.0, "util_pct": 90.0,
            "power_w": 40.0, "temperature_c": 60.0, "apps": [],
        }) for i in range(clock_log.MIN_WINDOW_SAMPLES + 5)]
        log.write_text("\n".join(rows), encoding="utf-8")
        w = clock_log.window("2026-08-02T12:00:00+00:00", "2026-08-02T12:01:00+00:00", log)
        assert any("retrospectively" in s for s in w.get("silence_set", []))

    def test_a_corrupt_log_line_is_refused_rather_than_raised(self, tmp_path):
        """Truncated lines happen to append-only logs. The refusal path must not itself
        raise -- a harness that dies on its own error path cannot report the error it found."""
        log = tmp_path / "corrupt.jsonl"
        rows = [json.dumps({"wall": f"2026-08-02T12:00:{i % 60:02d}.{i:03d}000+00:00",
                            "sm_mhz": 2010.0})
                for i in range(clock_log.MIN_WINDOW_SAMPLES + 5)]
        log.write_text("\n".join(rows) + "\n{not json", encoding="utf-8")
        w = clock_log.window("2026-08-02T12:00:00+00:00", "2026-08-02T12:01:00+00:00", log)
        assert w["verdict"] == clock_log.UNOBSERVABLE
        assert w["malformed_samples"] > 0

    def test_the_log_modules_do_not_leak_bench_results_onto_sys_path(self):
        """bench/results carries modules whose names collide with lane checks elsewhere;
        this is the defect locked by bench/test_import_isolation.py, re-asserted at the one
        new site that imports from there at call time."""
        before = list(sys.path)
        clock_log._probe()
        assert sys.path == before


# --------------------------------------------------------------------------------------
# The screen: cumulative counters must be differenced, never divided.
# --------------------------------------------------------------------------------------

#: Counters that accumulate over a whole session. Every one of these is dominated by a
#: one-time cost (chiefly the ~2185 MiB weight upload) and none may be divided by an
#: iteration count to obtain a per-inference figure.
CUMULATIVE = (
    "session_staging_upload_bytes",
    "session_staging_readback_bytes",
    "session_staging_uploads",
    "session_staging_readbacks",
    "session_staging_upload_us",
    "session_staging_readback_us",
    "session_device_allocs",
    "dispatches_executed",
    "compute_calls",
)

#: A denominator that turns one of the above into a per-inference figure.
_DENOM = r"(?:inference|iters?|iterations|n_inf\w*|samples|repeats)"

_QUOTIENT = re.compile(
    r"(?:" + "|".join(re.escape(c) for c in CUMULATIVE) + r")"
    r"[^\n]{0,80}?/\s*[^\n]{0,40}?" + _DENOM,
    re.IGNORECASE,
)


def _screen(text: str) -> list:
    hits = []
    for i, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith('"') or stripped.startswith("'"):
            continue
        if _QUOTIENT.search(line):
            hits.append((i, stripped))
    return hits


class TestCumulativeCounterScreen:
    def test_the_screen_has_a_positive_control(self):
        """A screen that has only ever run against clean code is a screen nobody has tested.
        This is the exact line I wrote and published a false finding from."""
        bad = 'per_inf = c["session_staging_upload_bytes"] / inferences\n'
        assert _screen(bad), "the screen failed to catch the defect it exists for"
        bad2 = "up = counters['session_device_allocs'] / n_inferences\n"
        assert _screen(bad2)

    def test_the_screen_has_a_negative_control(self):
        """It must not fire on the correct construction -- a difference across two points."""
        good = (
            'slope = (hi["session_staging_upload_bytes"] - lo["session_staging_upload_bytes"]) / dn\n'
            'total = c["session_staging_upload_bytes"]\n'
        )
        assert not _screen(good), _screen(good)

    def test_no_file_under_bench_divides_a_cumulative_counter_by_an_iteration_count(self):
        offenders = {}
        for path in sorted(BENCH.rglob("*.py")):
            if "_scratch" in path.parts or path.name == Path(__file__).name:
                continue
            hits = _screen(path.read_text(encoding="utf-8", errors="replace"))
            if hits:
                offenders[str(path.relative_to(BENCH))] = hits
        assert not offenders, (
            "a cumulative session counter is being divided by an iteration count. That counter "
            "is dominated by the one-time ~2185 MiB weight upload; dividing it by two different "
            "run lengths manufactures a ratio equal to the ratio of those lengths (51/28 = 1.82, "
            "which I published as a 1.78x improvement). Difference two points instead; see "
            "bench/results/probe_island_boundary_cost.py.\n"
            + json.dumps(offenders, indent=1)
        )
