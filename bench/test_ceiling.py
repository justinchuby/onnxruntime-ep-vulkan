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
    def test_the_dram_bound_is_now_admissible_at_every_context(self, ceil_):
        """GQA is claimed on this build -- 1 island, 355 of 363, read off the binary's own
        counters -- so the KV bytes are the device's and the by-context table describes it.
        This refusal is discharged. What replaced it is `binding_extent`."""
        ext = ceil_.extent()
        assert ext["admissible"] == ext["grid"]
        assert 8192 in ext["admissible"]

    def test_the_bound_is_the_FLOOR_at_zero_context_only(self, ceil_):
        """Admissible and binding are different questions. Discharging the structural
        refusal exposed a measured one: the present KV cache crosses device->host in full
        every inference, 393,216 B per past token, and the roofline does not model it."""
        assert ceil_.binding_extent()["binding"] == [0]

    def test_zero_context_is_a_bound(self, ceil_):
        r = ceil_.floor_ms(0)
        assert r["state"] == ceiling_mod.BOUND
        assert r["kv_cache_MiB"] == 0.0
        assert 8.0 < r["floor_ms_at_spec_peak"] < 8.5
        assert r["binding"]["binds_by"] == "DRAM"

    def test_at_zero_context_transfer_cannot_bind_on_any_link_ever_shipped(self, ceil_):
        """Settled without measuring this machine's link, which would be a timing."""
        b = ceil_.binding(0)
        assert b["state"] == "BINDS"
        assert b["link_GB_per_s_at_which_transfer_binds"] < ceiling_mod.LINK_FLOOR_GB_PER_S

    def test_long_contexts_are_transfer_bound_on_any_link_that_exists(self, ceil_):
        """The mirror argument, and it is the direction for work: at past_len 2048 and above
        the crossover exceeds the fastest consumer link ever shipped, so the DRAM roofline is
        nowhere near the floor there."""
        for past in (2048, 4096, 8192):
            b = ceil_.binding(past)
            assert b["binds_by"] == "TRANSFER", past
            assert b["state"] == ceiling_mod.UNOBSERVABLE, past
            assert b["link_GB_per_s_at_which_transfer_binds"] > ceiling_mod.LINK_CEILING_GB_PER_S

    def test_a_verdict_resting_on_extrapolation_says_so(self, ceil_):
        """Transfer was measured at 0, 128 and 512. Beyond that the readback law is applied
        rather than observed, and the caveat must travel with the verdict."""
        assert ceil_.binding(512)["caveat"] is None
        assert "extrapolated" in ceil_.binding(8192)["caveat"]

    def test_the_middle_contexts_are_undecided_rather_than_decided_by_a_tuned_constant(self, ceil_):
        """At 128 and 512 the crossover sits between the two stated link bounds. The honest
        answer is that it needs a measured link bandwidth, not a constant chosen until the
        count of awkward contexts reached zero."""
        for past in (128, 512):
            b = ceil_.binding(past)
            assert b["binds_by"] == "UNDECIDED", past
            assert b["state"] == ceiling_mod.UNOBSERVABLE, past

    def test_it_will_not_extrapolate_off_its_own_grid(self, ceil_):
        r = ceil_.floor_ms(777)
        assert r["state"] == ceiling_mod.UNOBSERVABLE
        assert "grid" in r["reason"]

    def test_the_kv_term_declares_which_half_of_it_was_earned(self, ceil_):
        """Switch had to earn amplification separately from residency for weights. For KV
        the magnitude is earned on the readback axis at ratio 1.000000; the read-side path
        and the DRAM amplification factor are not, and the bound must say so."""
        k = ceil_.floor_ms(0)["kv_term_earned"]
        assert k["magnitude"] == pytest.approx(1.0, abs=1e-9)
        assert "readback" in k["on_axis"]
        assert "amplification" in k["not_earned"]


class TestTheRefusalStillHasTeeth:
    """A refusal that has just been satisfied is exactly when it becomes decoration.

    The condition this module refused on -- GQA declined -- was discharged this round. These
    are the controls that show discharging it did not wire it open, and that the stale-record
    defect which let it be right about a build nobody was running cannot recur.
    """

    @staticmethod
    def _record(tmp_path, *, islands, claimed, sha, name="claim.json"):
        p = tmp_path / name
        p.write_text(json.dumps({
            "environment": {"build": {"sha256": sha}},
            "results": [{"claimed_nodes": claimed, "counters": {"subgraphs_live": islands}}],
        }), encoding="utf-8")
        return p

    def test_positive_control_a_declined_build_is_still_caught(self, tmp_path, ceil_):
        """The control that matters. A synthetic record that is IN FRAME -- same binary --
        but shows 33 islands must still collapse the extent to [0] with the structural
        reason. If this ever goes green-by-default the refusal has become decoration."""
        sha = ceil_.frame()["dll_sha256"]
        rec = self._record(tmp_path, islands=33, claimed=323, sha=sha)
        c = ceiling_mod.Ceiling.load(claim_record=rec)
        assert c.extent()["admissible"] == [0]
        assert "declined" in c.extent()["reason"]
        assert c.floor_ms(8192)["state"] == ceiling_mod.UNOBSERVABLE

    def test_negative_control_an_in_frame_claimed_build_widens(self, tmp_path, ceil_):
        sha = ceil_.frame()["dll_sha256"]
        rec = self._record(tmp_path, islands=1, claimed=355, sha=sha)
        c = ceiling_mod.Ceiling.load(claim_record=rec)
        assert c.extent()["admissible"] == c.extent()["grid"]

    def test_a_record_from_another_binary_is_refused_not_believed(self, tmp_path):
        """The defect this replaces: this module reported extent [0] off a record from the
        previous binary while the DLL beside it had already changed, and was confidently
        right about a build nobody was running. Artifact:
        bench/_scratch/ceiling_stale_record_artifact.txt."""
        rec = self._record(tmp_path, islands=33, claimed=323, sha="00" * 32)
        with pytest.raises(ceiling_mod.CeilingError) as exc:
            ceiling_mod.Ceiling.load(claim_record=rec)
        assert "out of frame" in str(exc.value)

    def test_a_record_that_cannot_name_its_binary_is_refused(self, tmp_path):
        """Size and mtime do not identify a build. A claim status that cannot be tied to a
        binary cannot decide an extent."""
        p = tmp_path / "old.json"
        p.write_text(json.dumps({
            "environment": {"build": {"bytes": 1909760, "mtime": "2026-08-02T10:06:06"}},
            "results": [{"claimed_nodes": 323, "counters": {"subgraphs_live": 33}}],
        }), encoding="utf-8")
        with pytest.raises(ceiling_mod.CeilingError) as exc:
            ceiling_mod.Ceiling.load(claim_record=p)
        assert "sha256" in str(exc.value)

    def test_a_missing_claim_record_raises_rather_than_silently_passing(self, tmp_path):
        """Trinity's shape: the premise asserts itself. Unknown claim status is not
        permission, and it is ERROR(instrument) rather than an empty extent that a caller
        might read as a quiet no-op."""
        with pytest.raises(ceiling_mod.CeilingError) as exc:
            ceiling_mod.Ceiling.load(claim_record=tmp_path / "nope.json")
        assert "ERROR(instrument)" in str(exc.value)

    def test_an_unbuilt_tree_is_an_instrument_error_not_a_verdict(self, tmp_path, ceil_):
        sha = ceil_.frame()["dll_sha256"]
        rec = self._record(tmp_path, islands=1, claimed=355, sha=sha)
        with pytest.raises(ceiling_mod.CeilingError) as exc:
            ceiling_mod.Ceiling.load(claim_record=rec, dll=tmp_path / "absent.dll")
        assert "could not be hashed" in str(exc.value)


class TestTheKVTermWasMeasuredNotAssumed:
    """`0` was the wrong answer while GQA was declined. It is still the wrong answer, and
    the token that replaced UNOBSERVABLE had to be a measurement.
    """

    @pytest.fixture(scope="class")
    def kv(self):
        p = BENCH / "results" / "kv_bytes_earned.json"
        if not p.is_file():
            pytest.skip("kv_bytes_earned.json absent")
        return json.loads(p.read_text(encoding="utf-8"))

    def test_the_prediction_came_from_the_graph_not_from_the_counters(self, kv):
        assert kv["predicted_bytes_per_past_token"] == 32 * 2 * 32 * 96 * 2
        assert "not from any counter" in kv["predicted_from"]

    def test_past_len_was_wired_and_the_falsifier_is_an_artifact(self, kv):
        """R10. Had the feeds been ignored, every number in that record would be a
        measurement of nothing."""
        w = kv["past_len_is_wired"]
        assert w["verdict"] == "WIRED"
        assert "30751" in w["artifact"] and "8521" in w["artifact"]

    def test_readback_matches_the_model_to_the_byte_on_both_segments(self, kv):
        for s in kv["segments"]:
            assert s["readback_ratio"] == pytest.approx(1.0, abs=1e-9)
        assert kv["readback"]["linearity_spread"] == pytest.approx(0.0, abs=1e-9)

    def test_the_flat_upload_axis_is_UNOBSERVABLE_and_never_zero(self, kv):
        """The upload counter is identical at past_len 0, 128 and 512. Reporting 0 would
        claim the read side of the KV cache is free; the counter is blind to the path by
        which it becomes resident, and its silence is not evidence."""
        assert kv["upload"]["state"] == "UNOBSERVABLE"
        assert kv["upload"]["observed_bytes_per_inference"] > 0
        assert "free" in kv["upload"]["means"]
        assert "R12" in kv["upload"]["why_not_zero"]

    def test_it_states_what_it_did_not_earn(self, kv):
        joined = " ".join(kv["does_not_earn"]).lower()
        assert "amplification" in joined
        assert "read side" in joined


class TestCeilingComparison:
    def test_the_quotable_figure_sits_at_the_one_binding_context(self, ceil_):
        """12.1847 ms is zero context, one token, and zero context is the only place the
        bound is the floor. That was true for a structural reason and is now true for a
        measured one."""
        cmp = ceil_.compare(12.1847, past_len=0)
        assert cmp["quotable"] is True
        assert cmp["floor_is_binding"] is True
        assert 0.6 < cmp["fraction_of_roofline"] < 0.7
        assert 1.4 < cmp["headroom_x"] < 1.6

    def test_the_pairing_is_stated_per_context_and_does_not_travel(self, ceil_):
        """The bound is now admissible at 128..8192, where we hold no quotable figure at
        all. Admissible in a regime where the comparison is not."""
        cmp = ceil_.compare(12.1847, past_len=8192)
        assert cmp["quotable"] is True          # the DRAM bound does describe this build
        assert cmp["floor_is_binding"] is False  # but it is not the floor here
        assert "NOT known to be the floor" in cmp["pairing"]
        assert "no quotable figure" in ceil_.compare(12.1847, past_len=0)["pairing"]

    def test_a_comparison_off_the_grid_is_refused(self, ceil_):
        cmp = ceil_.compare(12.1847, past_len=777)
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
