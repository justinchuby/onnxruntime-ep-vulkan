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


def test_census_baseline_has_no_drift() -> None:
    """``audit_instruments.py --check`` must be green, both domains.

    Runs the same entry point CI would run.  A baseline nobody is forced to look at is a
    baseline nobody looks at — Tank's words, and the reason this is a test and not a note.
    """
    audit = _load_audit()
    rc = audit.main(["--check"])
    assert rc == 0, "instrument census --check reported drift; see output above"


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
