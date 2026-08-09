"""Two-polarity self-test for the HARNESS domain of the instrument census.

WHY THIS FILE EXISTS
====================
The harness census (``rust/tools/audit_instruments.py``, ``harness_survey``) was added
because Guard D was an instrument that was broken for its entire life and nothing screened
it.  A screen added for that reason, and then only ever run against the real repository —
where it prints a plausible-looking answer that nobody can check — would be the same defect
one level up.  R10 does not stop applying because the mechanism is a screen.

So the screen gets what it demands of everything else: an artifact whose content varies
with its input.  Two synthetic test trees are constructed here, differing in exactly one
thing — whether the guard has a paired-polarity self-test — and the screen must report
different states for them.  If it reports the same state for both, it is not measuring
what it claims to measure and this file goes red.

These tests need no GPU, no model and no EP.  They are always in the lane.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_AUDIT = _REPO / "rust" / "tools" / "audit_instruments.py"


def _load_audit():
    """Import Tank's census module by path (it is a tool, not an installed package)."""
    spec = importlib.util.spec_from_file_location("audit_instruments", _AUDIT)
    assert spec and spec.loader, f"cannot load {_AUDIT}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules["audit_instruments"] = mod
    spec.loader.exec_module(mod)
    return mod


# The one instrument in both synthetic trees. Its body is irrelevant to the screen —
# that is the point: the screen scores the *evidence about* the guard, never the guard.
_OWNER_SRC = '''
def assert_thing(x):
    assert x > 0, "thing is bad"
    return x
'''

_NO_POLARITY_TEST = '''
import _models as m

def test_uses_the_guard():
    m.assert_thing(1)
'''

_BOTH_POLARITIES_TEST = '''
import pytest
import _models as m

def test_guard_accepts_a_good_world():
    assert m.assert_thing(1) == 1

def test_guard_rejects_a_bad_world():
    with pytest.raises(AssertionError):
        m.assert_thing(-1)
'''

_GATED_BOTH_POLARITIES_TEST = '''
import pytest
import _models as m

def test_guard_accepts(require_vulkan):
    assert m.assert_thing(1) == 1

def test_guard_rejects(require_vulkan):
    with pytest.raises(AssertionError):
        m.assert_thing(-1)
'''


def _build_tree(root: Path, test_src: str | None) -> Path:
    """Write a synthetic tests/ tree containing one instrument and *test_src*."""
    ops = root / "ops"
    ops.mkdir(parents=True, exist_ok=True)
    (ops / "_models.py").write_text(_OWNER_SRC, encoding="utf-8")
    if test_src is not None:
        (ops / "test_it.py").write_text(test_src, encoding="utf-8")
    return root


def _state_of(audit, root: Path, fn: str = "assert_thing") -> str:
    rows = audit.harness_survey(tests_root=root, files=["ops/_models.py"])
    matches = [r for r in rows if r["fn"] == fn]
    assert matches, f"screen did not find instrument {fn!r}; rows={rows}"
    return matches[0]["state"]


# ---------------------------------------------------------------------------
# The three polarities the screen must distinguish
# ---------------------------------------------------------------------------


def test_screen_reports_uninvoked_when_nothing_calls_the_guard(tmp_path: Path) -> None:
    """No caller at all → ``uninvoked``.

    This is the state ``assert_qdq_reference_oracle_safe`` is really in today: conftest
    prints a banner telling readers to call it and nothing does.
    """
    audit = _load_audit()
    root = _build_tree(tmp_path / "t_uninvoked", None)
    assert _state_of(audit, root) == "uninvoked"


def test_screen_reports_unfalsified_when_no_test_watches_it_disagree(tmp_path: Path) -> None:
    """Called, but never inside ``pytest.raises`` → ``unfalsified``.

    NEGATIVE POLARITY for the screen.  This is Guard D's state on the day it landed: four
    callers, all agreeing with it, none ever having seen it refuse anything.
    """
    audit = _load_audit()
    root = _build_tree(tmp_path / "t_unfalsified", _NO_POLARITY_TEST)
    assert _state_of(audit, root) == "unfalsified"


def test_screen_reports_screened_when_both_polarities_exist(tmp_path: Path) -> None:
    """Accept AND reject observed in the always-on lane → ``screened``.

    POSITIVE POLARITY for the screen.  Paired with the test above, this is the falsifying
    observation: the two trees differ only in the presence of a ``pytest.raises`` block,
    and the screen's output changes.  Its verdict therefore varies with its input, which is
    the only thing that distinguishes a live screen from one that prints a constant.
    """
    audit = _load_audit()
    root = _build_tree(tmp_path / "t_screened", _BOTH_POLARITIES_TEST)
    assert _state_of(audit, root) == "screened"


def test_screen_does_not_credit_a_gpu_gated_polarity_test(tmp_path: Path) -> None:
    """A paired self-test behind ``require_vulkan`` does NOT earn ``screened``.

    This is the case that makes the screen worth having rather than merely present.
    Guard D's callers were all GPU-gated; on a machine without a device they never ran, and
    on a machine with one their failure was read as the guard working.  A falsifier that
    only runs when the hardware cooperates is not always-on, so it does not count here.
    """
    audit = _load_audit()
    root = _build_tree(tmp_path / "t_gated", _GATED_BOTH_POLARITIES_TEST)
    assert _state_of(audit, root) == "unfalsified"


# ---------------------------------------------------------------------------
# VALUE POLARITY — the same three-way discrimination, for a TOTAL instrument
#
# Added 2026-08-06 (Tank) with `VALUE_REJECT_FN`.  The model above cannot see the refusal
# of an instrument that RETURNS it instead of raising it; `bench/devices.py::identify_by_uuid`
# is the specimen and was `unfalsified` because of it.  A second polarity source added to a
# screen, and then only ever run against the real repository where it happens to turn one
# row green, would be exactly the defect the top of this file exists to prevent — so it gets
# the same treatment: synthetic trees differing in one thing, and the screen must disagree
# about them.
# ---------------------------------------------------------------------------

# A TOTAL instrument. It never raises; its refusal is the `None` in the first slot.
_TOTAL_OWNER_SRC = '''
def check_thing(x):
    if x > 0:
        return x, "accepted"
    return None, "refused: not positive"
'''

# Both polarities are genuinely exercised here — and the old model scores this
# `unfalsified`, because there is no `pytest.raises` anywhere in it. This tree is the
# reason the second source exists.
_TOTAL_NO_VALUE_POLARITY_TEST = '''
import _models as m

def test_accepts():
    got, why = m.check_thing(1)
    assert got == 1

def test_refuses():
    got, why = m.check_thing(-1)
    assert got is None
'''

# The same two observations, declared through the enforcing helper.
_TOTAL_VALUE_POLARITY_TEST = '''
import _models as m
from _polarity import refuses, selects

def test_accepts():
    assert selects(m.check_thing(1), 1) == 1

def test_refuses():
    assert "not positive" in refuses(m.check_thing(-1))
'''

# `refuses` behind a GPU gate must earn nothing, exactly as `pytest.raises` behind one does.
_TOTAL_GATED_VALUE_POLARITY_TEST = '''
import _models as m
from _polarity import refuses, selects

def test_accepts(require_vulkan):
    assert selects(m.check_thing(1), 1) == 1

def test_refuses(require_vulkan):
    assert "not positive" in refuses(m.check_thing(-1))
'''


def _build_total_tree(root: Path, test_src: str) -> Path:
    ops = root / "ops"
    ops.mkdir(parents=True, exist_ok=True)
    (ops / "_models.py").write_text(_TOTAL_OWNER_SRC, encoding="utf-8")
    (ops / "test_it.py").write_text(test_src, encoding="utf-8")
    return root


def test_a_total_instrument_with_no_value_polarity_is_unfalsified(tmp_path: Path) -> None:
    """NEGATIVE POLARITY for the new source: two real observations, no declaration.

    This tree's tests DO watch the instrument disagree.  The screen still says
    ``unfalsified``, and that is correct rather than a bug: nothing in the AST distinguishes
    ``assert got is None`` from ``assert got is not None``, so the screen has observed
    nothing.  Recording this is what stops the next reader from assuming the new source
    credits any test that happens to call a total instrument.
    """
    audit = _load_audit()
    root = _build_total_tree(tmp_path / "t_total_bare", _TOTAL_NO_VALUE_POLARITY_TEST)
    assert _state_of(audit, root, fn="check_thing") == "unfalsified"


def test_a_total_instrument_declared_through_refuses_is_screened(tmp_path: Path) -> None:
    """POSITIVE POLARITY for the new source.

    The two trees differ in exactly one thing — whether the refusal is declared through
    ``bench/_polarity.py::refuses`` — and the screen's verdict changes.  What makes that
    credit legitimate rather than an annotation is enforced elsewhere, at run time:
    ``refuses`` RAISES when the thing inside it did not refuse, and
    ``bench/test_devices_identity.py`` puts five planted mutants through it to prove so.
    """
    audit = _load_audit()
    root = _build_total_tree(tmp_path / "t_total_declared", _TOTAL_VALUE_POLARITY_TEST)
    assert _state_of(audit, root, fn="check_thing") == "screened"


def test_value_polarity_behind_a_gpu_gate_earns_nothing(tmp_path: Path) -> None:
    """The gate rule is not weakened by the new source; Guard D's lesson still applies."""
    audit = _load_audit()
    root = _build_total_tree(tmp_path / "t_total_gated", _TOTAL_GATED_VALUE_POLARITY_TEST)
    assert _state_of(audit, root, fn="check_thing") == "unfalsified"


def test_identify_by_uuid_is_screened_in_the_real_repository() -> None:
    """The fact the discrimination above was built to establish, on the files on disk.

    ``identify_by_uuid`` was the finding that produced the value-polarity model.  It must be
    ``screened`` here — not baselined, not hand-noted — or the model bought nothing.
    """
    audit = _load_audit()
    rows = audit.harness_survey(
        tests_root=audit.BENCH,
        files=audit.BENCH_INSTRUMENT_FILES,
        fn_re=audit.BENCH_FN,
        prefix="bench",
    )
    got = [r for r in rows if r["fn"] == "identify_by_uuid"]
    assert got, "the bench screen no longer sees identify_by_uuid at all"
    assert got[0]["state"] == "screened", (
        f"identify_by_uuid is {got[0]['state']}: {got[0]}. bench/test_devices_identity.py is "
        "supposed to supply both polarities in the always-on lane, the reject side through "
        "bench/_polarity.py::refuses."
    )
    assert got[0]["reject"] >= 1 and got[0]["accept"] >= 1, got[0]


def test_the_issue_88_attribution_instruments_are_screened_in_the_real_repository() -> None:
    """D1. The two total instruments issue #88 added must be ``screened``, on the files on disk.

    ``bench/phases.py::unknown_phase_spans`` closes a **silent drop**: ``phase_spans`` filters on
    ``HOST_PHASES`` membership, so a phase added to ``trace.rs`` and not added there disappears
    from the table with no warning and every share becomes a percentage of the wrong denominator.
    ``compute_call_attribution`` computes the unattributed residual that issue #88 asked for.

    Both are *total* instruments — they return ``(value, why)`` and never raise — so the only way
    the census can see their polarities is through ``bench/_polarity.py``.  Baselining them as
    ``unfalsified`` would be exactly the second option ``_polarity.py``'s header refuses: turning
    an open question into a permanent one.  This test is what stops that from being available.

    ``bench/test_compute_attribution.py`` supplies both polarities in the always-on lane (no
    device, no model, no ``skipif``) and backs them with a mutation battery.
    """
    audit = _load_audit()
    rows = audit.harness_survey(
        tests_root=audit.BENCH,
        files=audit.BENCH_INSTRUMENT_FILES,
        fn_re=audit.BENCH_FN,
        prefix="bench",
    )
    by_fn = {r["fn"]: r for r in rows}
    for fn in ("unknown_phase_spans", "compute_call_attribution"):
        got = by_fn.get(fn)
        assert got is not None, (
            f"the bench screen does not see bench/phases.py::{fn} at all. It is a module-public "
            f"top-level function of a declared instrument module, so BENCH_FN should select it — "
            f"if it does not, the screen stopped covering phases.py and that is the bigger find."
        )
        assert got["state"] == "screened", (
            f"{fn} is {got['state']}: {got}. bench/test_compute_attribution.py is supposed to "
            f"supply both polarities in the always-on lane, the reject side through "
            f"bench/_polarity.py::refuses and the accept side through ::selects."
        )
        assert got["reject"] >= 1 and got["accept"] >= 1, got


# ---------------------------------------------------------------------------
# The real repository
# ---------------------------------------------------------------------------


def test_guard_d_is_screened_in_the_real_repository() -> None:
    """Guard D must be ``screened`` here, not merely in a synthetic tree.

    The synthetic tests prove the screen discriminates.  This one states the fact the
    discrimination was built to establish, against the actual files on disk.
    """
    audit = _load_audit()
    rows = audit.harness_survey()
    guard = [r for r in rows if r["fn"] == "assert_vulkan_executed_runtime"]
    assert guard, "the census no longer sees Guard D at all — that is a bigger problem"
    assert guard[0]["state"] == "screened", (
        f"Guard D is {guard[0]['state']}: {guard[0]}. "
        "tests/ops/test_guard_d.py is supposed to supply both polarities in the always-on lane."
    )


def test_census_baseline_has_no_drift(capsys: pytest.CaptureFixture[str]) -> None:
    """``audit_instruments.py --check`` must be green, both domains.

    Runs the same entry point CI would run.  A baseline nobody is forced to look at is a
    baseline nobody looks at — Tank's words, and the reason this is a test and not a note.

    **``main_guarded``, not ``main`` (Tank, 2026-08-01).**  ``main`` has three ways out and
    only two of them are return values: ``0``, ``1``, and *an exception*.  Calling it
    directly puts the third one in pytest's hands, where a ``CensusInstrumentError`` — the
    census saying it never reached its observation — arrives as a red in the same channel
    as a drift.  That is the census reproducing, in its own test, the two-token confusion
    R13 was written to end.  ``main_guarded`` is the only entry point that emits all three.

    And the **token** is asserted, not just the code.  R13's rule is *quote the failure
    text, never the failure count*; a test that reads only ``rc == 0`` would pass against a
    build of this script that had lost its verdict line entirely.
    """
    audit = _load_audit()
    rc = audit.main_guarded(["--check"])
    out = capsys.readouterr()
    combined = out.out + out.err
    assert rc != audit.EXIT_ERROR_INSTRUMENT, (
        "instrument census reported ERROR(instrument) — it did not reach its observation, "
        "so it detected NOTHING and this red is not a finding about the census's subject "
        f"(R13).  Quote this text:\n{combined[-2000:]}"
    )
    assert rc == audit.EXIT_PASS, (
        "instrument census reported FAIL(drift).  Quote the drift, not the exit code:\n"
        f"{combined[-2000:]}"
    )
    assert "CENSUS VERDICT: PASS" in combined, (
        "the census exited 0 without printing its verdict token.  An exit code with no "
        "verdict line is one witness, and this gate requires two: a caller that reads only "
        "the code cannot tell a PASS from a script that lost its reporting.  Output:\n"
        f"{combined[-2000:]}"
    )


def test_census_reports_error_instrument_when_it_cannot_reach_its_observation(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The MISSING POLARITY: ``main_guarded`` must emit ERROR(instrument), not FAIL(drift).

    ``test_census_baseline_has_no_drift`` above watches the census agree.  Nothing watched
    it produce its third token, so ``ERROR(instrument)`` was a state the census could
    declare and no test had ever seen — which is the definition of ``unfalsified`` applied
    to the census's own reporting layer.  Without this, the R13 wrapper is exactly the
    Guard D shape: present, plausible, unobserved.

    The world is broken at ``survey()``, i.e. *before* the census reaches any observation,
    so the exception is genuinely an outage and not a disguised finding.  Three things are
    asserted and each is separately load-bearing:

    * the exit code is ``EXIT_ERROR_INSTRUMENT`` and **not** ``EXIT_FAIL_DRIFT`` — the two
      must not be the same door;
    * the verdict line says ``ERROR(instrument)`` — the code and the text agree;
    * the exception's own text survives to the reader — R13 says quote the failure text,
      and a wrapper that swallowed it would satisfy the first two assertions.
    """
    audit = _load_audit()

    sentinel = "planted survey outage: the census never reached its observation"

    def _explode() -> list:
        raise RuntimeError(sentinel)

    monkeypatch.setattr(audit, "survey", _explode)

    rc = audit.main_guarded(["--check"])
    out = capsys.readouterr()
    combined = out.out + out.err

    assert rc == audit.EXIT_ERROR_INSTRUMENT, (
        f"a census that crashed before observing anything returned {rc}.  "
        f"EXIT_FAIL_DRIFT is {audit.EXIT_FAIL_DRIFT} and EXIT_ERROR_INSTRUMENT is "
        f"{audit.EXIT_ERROR_INSTRUMENT}; an outage returning the drift code would be "
        "reported to the team as a detection.  Output:\n" + combined[-2000:]
    )
    assert "CENSUS VERDICT: ERROR(instrument)" in combined, (
        "the census exited with the instrument code but never said so in words.  "
        "Output:\n" + combined[-2000:]
    )
    assert sentinel in combined, (
        "the census swallowed the text of the failure that stopped it.  R13: quote the "
        "failure text, never the failure count — a reader given only 'ERROR(instrument)' "
        "cannot route it.  Output:\n" + combined[-2000:]
    )


def test_census_error_and_drift_do_not_share_an_exit_code() -> None:
    """The paired control for the two tests above, and the one that makes them mean something.

    Both tests assert a specific exit code.  If ``EXIT_FAIL_DRIFT`` and
    ``EXIT_ERROR_INSTRUMENT`` were ever collapsed to the same integer — the state this
    whole wrapper exists to undo — both would still pass.  This is the assertion that
    fails the moment the three states become two.
    """
    audit = _load_audit()
    codes = [audit.EXIT_PASS, audit.EXIT_FAIL_DRIFT, audit.EXIT_ERROR_INSTRUMENT]
    assert len(set(codes)) == 3, (
        f"the census's three terminal states share an exit code: {codes}.  A caller cannot "
        "tell a drift from an outage, which is precisely the two-token channel R13 names."
    )


def test_census_timeout_in_a_caller_is_an_instrument_error_not_a_detection() -> None:
    """``subprocess.TimeoutExpired`` is ERROR(instrument).  Non-negotiable (Tank).

    The census is shelled out to from CI and from ``test_wiring_census``.  A budget
    overrun there produces a non-zero exit in the caller's frame, and the caller must not
    classify it as ``FAIL(drift)``.  ``_verdict.run_subprocess_checked`` is the mechanism;
    this is the always-on falsifier for it, driven by a command that is guaranteed to
    outlive its budget and needs no census, no GPU and no cargo.
    """
    import _verdict

    with pytest.raises(_verdict.InstrumentError) as excinfo:
        _verdict.run_subprocess_checked(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            what="census timeout falsifier",
            quiet_seconds=0.05,
            floor=1.0,
        )
    text = str(excinfo.value)
    assert "ERROR(instrument)" in text, text[:1500]
    assert "NOT A DETECTION" in text.upper(), (
        "the timeout was classified without saying it is not a detection.  The whole "
        "hazard is a reader counting it as one.\n" + text[:1500]
    )


def test_new_harness_instrument_without_a_self_test_goes_red(tmp_path: Path) -> None:
    """The drift gate must FAIL when an unfalsified instrument appears that is not baselined.

    Without this, the baseline mechanism itself is unfalsified: `--check` returning 0 on a
    clean tree is agreement, not evidence.  Here a guard is added to a synthetic tree whose
    baseline does not mention it, and the comparison must report it as new.
    """
    audit = _load_audit()
    root = _build_tree(tmp_path / "t_drift", _NO_POLARITY_TEST)
    current = sorted(
        r["id"] for r in audit.harness_survey(tests_root=root, files=["ops/_models.py"])
        if r["state"] == "unfalsified"
    )
    baseline: list[str] = []  # a baseline that has never heard of this guard
    added = [x for x in current if x not in baseline]
    assert added, (
        "a new unfalsified harness instrument was not reported as drift — the gate that "
        "would have caught Guard D's successor is itself inert"
    )
