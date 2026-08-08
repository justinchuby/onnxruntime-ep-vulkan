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
import tempfile
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


# ---------------------------------------------------------------------------
# NAME COLLISIONS ACROSS MODULES (added 2026-08-07)
#
# The screen keyed `instruments` and `stats` by BARE FUNCTION NAME. Eight names are
# defined by two or more screened bench modules, and a dict keyed on a bare name keeps
# only the last one: `phases.py::attribute`, `devices.py::probe`,
# `win_gpu_counters.py::summarise` and `real_model.py::build_feeds` were not in the
# census at all, while it printed PASS. Worse than the missing rows was the credit:
# `bench/test_cuda_competition.py` wraps `cuda_workloads.build_feeds` in
# `pytest.raises`, and that reject polarity was recorded against `real_model.py`'s
# unrelated `build_feeds` — a `screened` verdict for an instrument that test has never
# called.
#
# Synthetic trees, differing in one thing, and the screen must disagree about them.
# ---------------------------------------------------------------------------

_COLLIDING_A = '''
def check_thing(x):
    if x > 0:
        return x
    raise ValueError("a refused it")
'''

_COLLIDING_B = '''
def check_thing(x):
    return x
'''

# Both polarities, aimed unambiguously at module A through its import.
_COLLISION_TEST_A = '''
import pytest
import _models as a

def test_accepts():
    assert a.check_thing(1) == 1

def test_refuses():
    with pytest.raises(ValueError):
        a.check_thing(-1)
'''

# The same two polarities aimed at module B, which is NOT screened here.
_COLLISION_TEST_B = '''
import pytest
import _other as b

def test_accepts():
    assert b.check_thing(1) == 1

def test_refuses():
    with pytest.raises(ValueError):
        b.check_thing(-1)
'''

# A bare call to the name: the file imports the module but calls the name unqualified, so
# nothing in it says whether this is the instrument or a local of the same name.
_COLLISION_TEST_BARE = '''
import pytest
import _models  # noqa: F401

def test_refuses():
    with pytest.raises(ValueError):
        check_thing(-1)
'''


def _build_collision_tree(root: Path, test_src: str) -> Path:
    ops = root / "ops"
    ops.mkdir(parents=True, exist_ok=True)
    (ops / "_models.py").write_text(_COLLIDING_A, encoding="utf-8")
    (ops / "_other.py").write_text(_COLLIDING_B, encoding="utf-8")
    (ops / "test_it.py").write_text(test_src, encoding="utf-8")
    return root


def _collision_rows(audit, root: Path) -> "dict[str, dict]":
    rows = audit.harness_survey(
        tests_root=root, files=["ops/_models.py", "ops/_other.py"])
    return {r["id"]: r for r in rows}


def test_two_modules_with_the_same_function_name_both_get_a_row(tmp_path: Path) -> None:
    """One row per (file, function). A dict keyed by name would have kept one of these."""
    audit = _load_audit()
    root = _build_collision_tree(tmp_path / "t_collide_rows", _COLLISION_TEST_A)
    rows = _collision_rows(audit, root)
    assert set(rows) == {"tests/ops/_models.py::check_thing",
                         "tests/ops/_other.py::check_thing"}, sorted(rows)


def test_polarity_lands_on_the_module_the_test_actually_imported(tmp_path: Path) -> None:
    """POSITIVE POLARITY: the credited row is the imported one, and only it."""
    audit = _load_audit()
    root = _build_collision_tree(tmp_path / "t_collide_a", _COLLISION_TEST_A)
    rows = _collision_rows(audit, root)
    assert rows["tests/ops/_models.py::check_thing"]["state"] == "screened", rows
    assert rows["tests/ops/_other.py::check_thing"]["state"] == "uninvoked", rows


def test_the_other_modules_row_is_not_credited_by_a_same_named_call(tmp_path: Path) -> None:
    """NEGATIVE POLARITY, and the specimen defect: `cw.build_feeds` credited `real_model`.

    The two trees here differ only in which module the test imports, and the screen's
    verdict must move with it. Under the bare-name model both trees produced the same
    answer, which is what made the census unable to tell a screened instrument from an
    unwatched one with a popular name.
    """
    audit = _load_audit()
    root = _build_collision_tree(tmp_path / "t_collide_b", _COLLISION_TEST_B)
    rows = _collision_rows(audit, root)
    assert rows["tests/ops/_other.py::check_thing"]["state"] == "screened", rows
    assert rows["tests/ops/_models.py::check_thing"]["state"] == "uninvoked", (
        "a call to another module's function credited this one; that is a borrowed "
        "polarity, and it reads exactly like coverage")


def test_an_unattributable_call_earns_nobody_anything(tmp_path: Path) -> None:
    """Fail closed on an unresolved name, and say so rather than dropping it silently."""
    audit = _load_audit()
    root = _build_collision_tree(tmp_path / "t_collide_bare", _COLLISION_TEST_BARE)
    unresolved: list = []
    rows = {r["id"]: r for r in audit.harness_survey(
        tests_root=root, files=["ops/_models.py", "ops/_other.py"],
        unresolved_out=unresolved)}
    assert rows["tests/ops/_models.py::check_thing"]["state"] == "uninvoked", rows
    assert rows["tests/ops/_other.py::check_thing"]["state"] == "uninvoked", rows
    assert [u["name"] for u in unresolved] == ["check_thing"], unresolved
    assert len(unresolved[0]["candidates"]) == 2, unresolved


def test_the_colliding_bench_instruments_are_in_the_census(tmp_path: Path) -> None:
    """The fact the discrimination above was built to establish, on the files on disk.

    Every one of these was missing from the bench screen because another module defines a
    function with the same name.
    """
    audit = _load_audit()
    rows = {r["id"]: r for r in audit.harness_survey(
        tests_root=audit.BENCH, files=audit.BENCH_INSTRUMENT_FILES,
        fn_re=audit.BENCH_FN, prefix="bench")}
    for want in ("bench/phases.py::attribute", "bench/cuda_profile.py::attribute",
                 "bench/devices.py::probe", "bench/cuda_probe.py::probe",
                 "bench/win_gpu_counters.py::summarise",
                 "bench/cuda_competition.py::summarise",
                 "bench/real_model.py::sha256_file",
                 "bench/bench_models.py::sha256_file",
                 "bench/real_model.py::build_feeds"):
        assert want in rows, f"{want} is not in the bench census at all"
        assert rows[want]["state"] != "uninvoked", (
            f"{want} reads UNINVOKED; every one of these has a production caller, and a "
            f"fabricated detection is worse than a missing row: {rows[want]}")


def test_real_model_build_feeds_is_not_credited_by_the_cuda_workloads_test() -> None:
    """The exact borrowed polarity that was in the tree, named.

    `bench/test_cuda_competition.py` does `with pytest.raises(ValueError): cw.build_feeds(...)`
    where `cw` is `cuda_workloads`, a module this census does not screen. `real_model.py`
    also defines `build_feeds`, and it was the row that collected the reject.
    """
    audit = _load_audit()
    rows = {r["id"]: r for r in audit.harness_survey(
        tests_root=audit.BENCH, files=audit.BENCH_INSTRUMENT_FILES,
        fn_re=audit.BENCH_FN, prefix="bench")}
    row = rows["bench/real_model.py::build_feeds"]
    assert row["reject"] == 0, (
        "real_model.build_feeds has reject polarity, but the only `pytest.raises` around a "
        f"`build_feeds` in bench/ is aimed at cuda_workloads: {row}")
    assert row["state"] == "unfalsified", row


# ---------------------------------------------------------------------------
# THE BASELINE MAY NOT CARRY A COUNT NOBODY DERIVED (added 2026-08-08)
#
# `instrument_census.json` carried a hand-typed bench row count — the sentence is
# `audit.STALE_SPECIMENS["bare_row"]`, quoted from there and not retyped here — beside a
# MACHINE-GENERATED `bench_unfalsified` array that had long outgrown it, and a
# hand-classified entry that stated its own row count three times and was wrong all three.
# Every array matched the tree, so all four drift comparisons passed: the census cannot
# see a summary that disagrees with the collection it summarises, because the summary was
# never an output of the census.
#
# It is one now. `counts` is derived at --write-baseline time and re-derived on every
# --check, and prose may quote a row count only if `counts` contains it. Both polarities,
# because "no offenders" is what a clean baseline looks like AND what a dead arm looks like.
#
# NO NUMERAL APPEARS IN THIS MODULE'S PROSE, and that is not stylistic. This file is in
# `audit.SOURCE_CLAIM_CORPUS`: its comments and docstrings are screened by the same
# binding rule as the artifact's, because the first repair screened the artifact alone
# and the stale figures went on living in the sentences describing them — here.
# ---------------------------------------------------------------------------

def _baseline(audit) -> dict:
    import json

    return json.loads(audit.BASELINE.read_text(encoding="utf-8"))


def test_the_committed_baseline_publishes_counts_it_derives() -> None:
    """POSITIVE POLARITY: the file on disk agrees with itself today."""
    audit = _load_audit()
    base = _baseline(audit)
    assert "counts" in base, (
        "the baseline publishes no derived `counts`; a census with no summary of its own "
        "invites a hand-typed one, which is the defect this arm exists for")
    assert base["counts"] == audit.baseline_counts(base)
    assert base["counts"]["bench_unfalsified"] == len(base["bench_unfalsified"])
    per_module = base["counts"]["bench_unfalsified_by_module"]
    assert sum(per_module.values()) == len(base["bench_unfalsified"])


def test_a_hand_edited_summary_count_is_drift() -> None:
    """NEGATIVE POLARITY: mutate the published count, and the derivation must disagree.

    This is the mutation the review actually found, applied to the mechanism that now
    stops it: a number edited in place beside an array nobody re-read.
    """
    audit = _load_audit()
    base = _baseline(audit)
    mutated = dict(base["counts"],
                   bench_unfalsified=base["counts"]["bench_unfalsified"] + 1)
    assert mutated != audit.baseline_counts(base), (
        "a summary count was changed by hand and the derivation still agreed with it; "
        "the self-count arm is inert")


def test_prose_may_quote_a_derived_row_count_and_only_that() -> None:
    """Both polarities of the prose screen, on the committed baseline."""
    audit = _load_audit()
    base = _baseline(audit)
    counts = audit.baseline_counts(base)

    assert audit.prose_row_claims(base, counts) == [], (
        "the baseline quotes a row count nothing derives")

    planted = dict(base, planted_note=f"{max(counts['bench_unfalsified'] + 1, 999)} rows")
    assert audit.prose_row_claims(planted, counts), (
        "a planted prose row count that no derived number matches was not caught")

    honest = dict(base, planted_note=f"{counts['bench_unfalsified']} rows")
    assert audit.prose_row_claims(honest, counts) == [], (
        "a row count that IS derived was reported as stale; a screen that fires on "
        "everything is no more use than one that never fires")


def test_the_census_self_test_covers_the_self_count_arm() -> None:
    """The arm is wired into `self_test`, so a broken one is ERROR(instrument), not a pass."""
    audit = _load_audit()
    assert audit.self_test() == 0


# ---------------------------------------------------------------------------
# A COUNT MUST BE BOUND TO THE COLLECTION ITS OWN SENTENCE NAMES (added 2026-08-08)
#
# The arm above asks "does SOME derived number equal this integer?", and this census
# derives one number per counted array plus one per screened bench module. That is a
# large enough bag that a wrong sentence can nearly always find a witness in it. The
# specimen the review found is `audit.STALE_SPECIMENS["module_set"]`: its number is real —
# it is one of the four modules on its own — and the set it claims to count is larger.
#
# So a claim that names a scope is bound to that scope. A module-set claim must equal the
# sum over the modules ITS OWN SCOPE declares, and its cardinality word must equal how
# many that is; a claim that names a collection must equal that collection. Only a bare
# `<n> rows`, with nothing in the sentence to bind it to, still falls back to the bag.
#
# Both polarities on every arm, and every figure below is read from the census rather
# than written out, so these tests cannot themselves become the stale figure — which is
# exactly what the sentences that used to sit here had become.
# ---------------------------------------------------------------------------

def _four_module_entry(audit, base) -> "tuple[str, dict, list[str], int]":
    """The census entry that names four bench modules, its modules, and their true sum."""
    counts = audit.baseline_counts(base)
    by_module = counts["bench_unfalsified_by_module"]
    for path, entry in audit._entries(base):
        modules = audit.entry_modules(entry)
        if len(modules) == 4 and all(m in by_module for m in modules):
            return path, entry, modules, sum(by_module[m] for m in modules)
    pytest.skip("no census entry declares four screened bench modules")


def _with_claim(audit, base, claim: str) -> dict:
    """The committed baseline with ``claim`` planted in the four-module entry."""
    import copy

    mutated = copy.deepcopy(base)
    path, _, _, _ = _four_module_entry(audit, base)
    for path2, entry in audit._entries(mutated):
        if path2 == path:
            entry["planted_claim"] = claim
            return mutated
    raise AssertionError("the planted entry could not be found in the copy")


def test_a_module_set_claim_may_not_borrow_another_modules_count() -> None:
    """THE MUTATION THE REVIEW FOUND. One module's count, wearing four modules' clothes."""
    audit = _load_audit()
    base = _baseline(audit)
    counts = audit.baseline_counts(base)
    _, _, modules, total = _four_module_entry(audit, base)
    borrowed = counts["bench_unfalsified_by_module"][modules[0]]
    assert borrowed != total, "the fixture needs a per-module count that is not the sum"

    offenders = audit.prose_row_claims(
        _with_claim(audit, base, f"{borrowed} rows across these four modules"), counts)

    assert offenders, (
        f"{borrowed} is a real count of one module and was accepted as a count of four")
    assert str(total) in offenders[0]


def test_the_true_module_set_sum_is_accepted() -> None:
    """BOTH POLARITIES. A binding that rejects the true sentence binds nothing usable."""
    audit = _load_audit()
    base = _baseline(audit)
    counts = audit.baseline_counts(base)
    _, _, _, total = _four_module_entry(audit, base)

    assert audit.prose_row_claims(
        _with_claim(audit, base, f"{total} rows across these four modules"), counts) == []


def test_a_module_set_claim_is_bound_to_the_modules_the_entry_names() -> None:
    """Mutate the SET, not the number: the same true sum stops being true.

    This is what makes the binding a binding rather than a second constant. Drop one
    module from the entry's `instrument` field and the sentence that was exact becomes an
    offender, with no digit changed anywhere.
    """
    import copy

    audit = _load_audit()
    base = _baseline(audit)
    counts = audit.baseline_counts(base)
    path, entry, modules, total = _four_module_entry(audit, base)

    mutated = copy.deepcopy(_with_claim(audit, base, f"{total} rows across these four modules"))
    for path2, ent in audit._entries(mutated):
        if path2 == path:
            ent["instrument"] = ent["instrument"].replace(modules[-1] + ", ", "")
            ent["instrument"] = ent["instrument"].replace(", " + modules[-1], "")
            break

    assert audit.entry_modules(ent) == modules[:-1]
    offenders = audit.prose_row_claims(mutated, counts)
    assert offenders, "the claim survived the module set it was a count of changing"


def test_the_cardinality_word_is_bound_too() -> None:
    """A cardinality word that miscounts the modules beside it is false with a true sum."""
    audit = _load_audit()
    base = _baseline(audit)
    counts = audit.baseline_counts(base)
    _, _, modules, total = _four_module_entry(audit, base)
    wrong = len(modules) - 1
    spelled = {v: k for k, v in audit.CARDINALS.items()}[wrong]

    offenders = audit.prose_row_claims(
        _with_claim(audit, base, f"{total} rows across these {spelled} modules"), counts)

    assert offenders and f"module(s), not {wrong}" in offenders[-1]


def test_a_labelled_row_count_is_checked_against_the_collection_it_names() -> None:
    """Wrong label / right number, and right label / wrong number, in both polarities."""
    audit = _load_audit()
    base = _baseline(audit)
    counts = audit.baseline_counts(base)
    bench = counts["bench_unfalsified"]
    uninvoked = counts["uninvoked"]
    assert bench != uninvoked, "the fixture needs two collections of different size"

    # Right number, wrong label: `bench` is real, and it is not how many `uninvoked` are.
    assert audit.prose_row_claims(dict(base, note=f"{bench} uninvoked rows"), counts)
    # Right label, wrong number.
    assert audit.prose_row_claims(dict(base, note=f"{bench + 1} bench unfalsified rows"),
                                  counts)
    # Both right, in each of the spellings the census actually uses.
    for honest in (f"{bench} bench unfalsified rows",
                   f"{bench} `counts.bench_unfalsified` rows",
                   f"{uninvoked} uninvoked rows"):
        assert audit.prose_row_claims(dict(base, note=honest), counts) == [], honest


def test_a_duplicate_number_in_another_collection_cannot_witness_a_label() -> None:
    """Two collections of the same size is exactly when a bag check has no power left.

    The census derives several per-module counts that coincide. A labelled claim that
    quotes one of them for a differently-named collection is still false, and the bag it
    used to be checked against contains it.
    """
    audit = _load_audit()
    base = _baseline(audit)
    counts = audit.baseline_counts(base)
    duplicated = sorted(
        v for v in counts["bench_unfalsified_by_module"].values()
        if list(counts["bench_unfalsified_by_module"].values()).count(v) > 1)
    if not duplicated:
        pytest.skip("no two screened modules currently hold the same row count")
    borrowed = duplicated[0]
    assert borrowed in audit.derived_numbers(counts), "the fixture must be in the old bag"
    assert borrowed != counts["harness_unfalsified"]

    offenders = audit.prose_row_claims(
        dict(base, note=f"{borrowed} harness unfalsified rows"), counts)

    assert offenders, (
        f"{borrowed} is a real per-module count and was accepted as a harness count")


def test_a_bare_row_count_with_nothing_to_bind_to_still_falls_back() -> None:
    """The scope of the change, stated: an unlabelled count keeps the old, weaker rule."""
    audit = _load_audit()
    base = _baseline(audit)
    counts = audit.baseline_counts(base)

    assert audit.prose_row_claims(
        dict(base, note=f"{counts['bench_unfalsified']} rows"), counts) == []
    assert audit.prose_row_claims(
        dict(base, note=f"{max(audit.derived_numbers(counts)) + 1000} rows"), counts)


# ---------------------------------------------------------------------------
# THE SCREEN'S CORPUS IS THE CENSUS'S WHOLE SURFACE, NOT ONE FILE (added 2026-08-08)
#
# The arms above screen the GENERATED DOCUMENT. That was the whole corpus, and it was not
# enough: at the moment those arms were written and green, the stale figures they exist to
# remove were sitting in `rust/tools/audit_instruments.py`'s own comments and in this
# module's prose, describing the defect in the spelling of the defect. `--check` passed and
# every census test passed, because neither file was ever read.
#
# So the corpus is declared (`audit.SOURCE_CLAIM_CORPUS`), the same binding rule runs over
# each declared file's comments, docstrings and string constants, and an in-frame file that
# starts making census count claims without being declared is drift in the corpus itself.
#
# The controls below plant `audit.STALE_SPECIMENS` — the exact sentences that shipped — into
# each previously-missed location, and require the screen to convict every one. The
# specimens are READ from the census script rather than retyped, which is why no numeral
# appears anywhere in this module: a control that spells the stale figure out is a fresh
# copy of it, sitting in a file this screen reads.
# ---------------------------------------------------------------------------

def _corpus_text(audit, rel: str) -> str:
    return (audit.REPO / rel).read_text(encoding="utf-8")


def test_every_stale_specimen_is_convicted_in_every_missed_location() -> None:
    """NEGATIVE POLARITY, planted into the real files that carried them.

    Cross-product on purpose: each specimen shape against each declared corpus file, in a
    comment and in a docstring, appended to that file's REAL text so the screen is judged
    on the document it actually reads. `--check` was green with these sentences in the
    tree; every cell here has to be red.
    """
    audit = _load_audit()
    counts = audit.baseline_counts(_baseline(audit))
    modules = audit.four_module_scope(_baseline(audit), counts)
    assert modules, "the census must declare a multi-module entry for the scoped specimens"
    named = " ".join(f"`{m}`" for m in modules)

    for rel in audit.SOURCE_CLAIM_CORPUS:
        real = _corpus_text(audit, rel)
        assert audit.source_claim_offenders(
            real, counts, rel,
            exempt_specimen_table=rel == audit.HERE.relative_to(audit.REPO).as_posix()) == [], (
            f"{rel} carries an unwitnessed census count claim today")
        for key, specimen in audit.STALE_SPECIMENS.items():
            for shape in (f"\n# {named} {specimen}\n",
                          f'\ndef _planted_{key}():\n    """{named} {specimen}."""\n'):
                offenders = audit.source_claim_offenders(
                    real + shape, counts, rel,
                    exempt_specimen_table=(
                        rel == audit.HERE.relative_to(audit.REPO).as_posix()))
                assert offenders, (
                    f"{rel}: the screen acquitted {specimen!r} planted as {shape!r}; this "
                    f"is the exact sentence that was in the tree while the gate was green")


def test_the_planted_specimen_reaches_the_real_gate_through_the_corpus_loop() -> None:
    """The same plant, on disk, through the function `--check` calls.

    `source_claim_offenders` is a text screen; `corpus_claim_offenders` is what the census
    actually runs, and it is the layer that decides WHICH files get read. A control that
    only exercises the first proves the screen works on a document nobody opens.
    """
    import shutil

    audit = _load_audit()
    counts = audit.baseline_counts(_baseline(audit))
    modules = audit.four_module_scope(_baseline(audit), counts)
    named = " ".join(f"`{m}`" for m in modules)
    root = Path(tempfile.mkdtemp(prefix="census_corpus_"))
    try:
        for rel in audit.SOURCE_CLAIM_CORPUS:
            dst = root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(audit.REPO / rel, dst)

        assert audit.corpus_claim_offenders(counts, repo=root) == [], (
            "an unmutated copy of the declared corpus is already red; the control below "
            "would prove nothing")

        for rel in audit.SOURCE_CLAIM_CORPUS:
            target = root / rel
            clean = target.read_text(encoding="utf-8")
            target.write_text(
                clean + f"\n# {named} {audit.STALE_SPECIMENS['module_sum']}\n",
                encoding="utf-8")
            offenders = audit.corpus_claim_offenders(counts, repo=root)
            assert any(rel in o for o in offenders), (
                f"the corpus loop did not read {rel}; a declared file nobody opens is a "
                f"declaration, not a screen")
            target.write_text(clean, encoding="utf-8")

        assert audit.corpus_claim_offenders(counts, repo=root) == [], (
            "reverting every plant did not return the corpus to green; the screen is "
            "reporting something other than what was planted")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_a_derived_or_current_sentence_survives_the_source_screen() -> None:
    """MUST-PASS POLARITY. A screen that convicts the fix is a screen nobody can satisfy.

    Three shapes a source file is allowed to state a census figure in: derived at runtime
    (no digit in the source at all), spelled out and correctly bound, and cited by field
    name. All three have to pass, or the only way to be green is to say nothing.
    """
    audit = _load_audit()
    counts = audit.baseline_counts(_baseline(audit))
    modules = audit.four_module_scope(_baseline(audit), counts)
    by_module = counts["bench_unfalsified_by_module"]
    named = " ".join(f"`{m}`" for m in modules)
    total = sum(by_module[m] for m in modules)
    spelled = {v: k for k, v in audit.CARDINALS.items()}[len(modules)]

    honest = (
        '\nX = f"{counts[\'bench_unfalsified\']} bench unfalsified rows"\n'
        f"\n# {counts['bench_unfalsified']} bench unfalsified rows\n"
        f"\n# {counts['harness_unfalsified']} harness unfalsified rows\n"
        f"\n# {named}: {total} rows across these {spelled} modules\n"
        "\n# Read `counts.bench_unfalsified` for the whole bench domain.\n"
    )

    assert audit.source_claim_offenders(honest, counts, "<honest>") == [], (
        "a derived or currently-true census sentence was reported as stale")


def test_a_delta_line_is_not_read_as_a_collection_size() -> None:
    """The other way this screen could have become useless: convicting a correct sentence.

    `--check` prints a count of what CHANGED in one run. That is not the size of a
    collection, and binding it to one would fail the census's own drift line — the string
    `ci/open_reds.json` holds as an accepted red's signature — for disagreeing with a
    number it was never about.
    """
    audit = _load_audit()
    counts = audit.baseline_counts(_baseline(audit))
    delta = counts["uninvoked"] + 1

    assert audit.source_claim_offenders(
        f"\n# never printed `{delta} NEW uninvoked instrument(s)`\n",
        counts, "<delta>") == []
    assert audit.source_claim_offenders(
        f"\n# {delta} bench unfalsified rows\n", counts, "<total>"), (
        "the delta exemption swallowed a plain labelled claim; it must apply only to the "
        "delta wording")


def test_the_specimen_table_is_still_stale_and_is_the_only_exempt_span() -> None:
    """`STALE_SPECIMENS` is the one place a stale figure may be spelled, so it owes proof.

    Two halves. Every entry must STILL be convicted by the live screen — a specimen that
    has quietly become true is a claim wearing a historical label. And the exemption is
    the assignment's own source span, found by parsing the file, so it cannot be moved
    onto a live sentence by copying a marker comment.
    """
    audit = _load_audit()
    base = _baseline(audit)
    counts = audit.baseline_counts(base)

    assert audit.specimen_offenders(base, counts) == []

    text = _corpus_text(audit, "rust/tools/audit_instruments.py")
    assert audit.source_claim_offenders(
        text, counts, "audit_instruments.py", exempt_specimen_table=False), (
        "with the specimen table unexempted the census script must be red; if it is not, "
        "the table has stopped holding the specimens and the exemption is inert")
    assert audit.source_claim_offenders(
        text, counts, "audit_instruments.py", exempt_specimen_table=True) == []


def test_the_corpus_grows_with_the_claims() -> None:
    """An in-frame file that starts making census claims must be declared, or it is drift.

    Non-vacuity in the other direction: the candidate scan has to actually nominate a real
    file, or "no undeclared candidates" would be what a dead scan says too. A DECLARED file
    is not required to be a candidate — this module states no figure at all, which is the
    outcome the screen is for — but every candidate must be declared.
    """
    audit = _load_audit()
    counts = audit.baseline_counts(_baseline(audit))
    candidates = audit.claim_corpus_candidates()

    assert candidates, (
        "the candidate scan nominates nothing anywhere in the tree; a scan that never "
        "fires cannot notice a file that starts making census claims")
    assert set(candidates) <= set(audit.SOURCE_CLAIM_CORPUS), (
        f"undeclared census claim candidates in the tree: "
        f"{sorted(set(candidates) - set(audit.SOURCE_CLAIM_CORPUS))}")

    root = Path(tempfile.mkdtemp(prefix="census_candidates_"))
    try:
        undeclared = root / "ci" / "check_planted_claim.py"
        undeclared.parent.mkdir(parents=True, exist_ok=True)
        undeclared.write_text(
            f"# bench_unfalsified: {audit.STALE_SPECIMENS['labelled_row']}\n",
            encoding="utf-8")
        for rel in audit.SOURCE_CLAIM_CORPUS:
            dst = root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text((audit.REPO / rel).read_text(encoding="utf-8"), encoding="utf-8")

        assert "ci/check_planted_claim.py" in audit.claim_corpus_candidates(root)
        offenders = audit.corpus_claim_offenders(counts, repo=root)
        assert any("check_planted_claim.py" in o for o in offenders)

        undeclared.write_text(
            "# a lane gate with no census claim in it at all\n", encoding="utf-8")
        assert "ci/check_planted_claim.py" not in audit.claim_corpus_candidates(root), (
            "the candidate scan nominates a file that makes no census claim; a rule that "
            "demands a declaration from every file is the reject-all failure mode")
    finally:
        import shutil

        shutil.rmtree(root, ignore_errors=True)


def test_the_check_path_still_runs_the_source_and_specimen_screens() -> None:
    """Source-reading wiring guard, for the reason the first repair needed a second.

    A screen that is written and not called is exactly what shipped: the binding rule
    existed, the corpus did not include the files carrying the claims, and nothing said
    so. This reads `main`'s own source and fails if either call is removed, so unwiring
    the screen cannot be a silent edit.
    """
    import ast

    audit = _load_audit()
    tree = ast.parse(_corpus_text(audit, "rust/tools/audit_instruments.py"))
    main_fn = next(n for n in tree.body
                   if isinstance(n, ast.FunctionDef) and n.name == "main")
    called = {n.func.id for n in ast.walk(main_fn)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}

    for name in ("prose_row_claims", "corpus_claim_offenders", "specimen_offenders"):
        assert name in called, (
            f"`main` no longer calls `{name}`; the screen exists and the census does not "
            f"run it, which is the state this whole section was written for")
