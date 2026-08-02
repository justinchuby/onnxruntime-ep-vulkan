"""A mechanical screen for the defect class that broke `ci/test_lane_checks.py`.

WHY THIS FILE EXISTS, AND WHY IT DOES NOT SCREEN WHAT WAS PRESCRIBED
====================================================================
Reported symptom: `bench/test_marginal_tail_withholds.py` + `ci/test_lane_checks.py` in one pytest
process = 3 failures; each file passes alone. Prescribed diagnosis: two unrestored
`sys.path.insert(0, ...)` calls in the bench file, so downstream imports resolve against `bench/`
first. Prescribed fix: `monkeypatch.syspath_prepend`.

**That diagnosis is wrong about the mechanism, and the prescribed fix alone leaves the bug live.**
The falsifier is an artifact (R10). With `sys.path` restored to a byte-identical copy of the
original list (verified `sys.path clean: True`), Link's `import device_state` still resolved to
`bench/device_state.py` and `hasattr(ds, "certifies_comparison")` was still `False`. The three
failures are `AttributeError`, not `ImportError`, and that distinction is the whole diagnosis:
`import` consults `sys.modules` **before** it consults `sys.path`. Two different files were named
`device_state.py` — `bench/device_state.py` (the tenancy+SM-clock companion) and `ci/device_state.py`
(the lane-level obligation-8 registry). Whichever imported first bound the global name for the rest
of the process, and Link's own `sys.path.insert(0, CI_DIR)` could not rescue him, because by then
there was nothing left to resolve.

The prescribed fix is therefore necessary hygiene and insufficient as a repair. `sys.path` ordering
decides *which file wins the race*; it does not decide *that there is a race*. The race exists
because two files share a base name. That is the invariant this file screens.

Screened here, both with a positive control, because a checklist decays (Tank's rule):

  1. **No two importable modules in this repository share a base name.** Text-decidable, runs in
     milliseconds, needs no GPU, and is the half that actually caused the failure.
  2. **Importing every module under `bench/` leaves `sys.path` as it found it.** Not text-decidable
     — so it is executed, not grepped. This is the prescribed invariant, kept because it is real:
     a leaked `bench/` entry would decide the *next* name race in bench's favour.

     **Declared blind spot, because a screen that hides its own extent is worse than no screen.**
     To import a module by flat name the screen must itself put that module's directory on the
     path, so a module that inserts *its own directory* (guarded by `if ... not in sys.path`) is
     invisible to this check. Cross-directory insertion is fully visible, and that is the harmful
     case: the three violations this screen found on its first run were `bench/cases.py`,
     `bench/island_attribution.py` and `bench/transfer_calibration.py`, all leaving `tests/ops`
     permanently at the front of `sys.path` for every later import in the process. One of the three
     inserted it and never used it.

Neither screen is worth anything if it cannot fail, so each carries a positive control that feeds
it a known-bad synthetic input and asserts it reports the violation. An always-green screen and an
absent screen are the same artifact.

R13: a failure here is `FAIL(condition)` — a real defect in the repository's import surface. If the
screen itself cannot run (roots missing, walk errors), that is `ERROR(instrument)` and is raised as
such rather than passing by default.
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

#: Directories whose ``.py`` files are imported by flat name (no package, no ``__init__.py``), so
#: their base names all compete for one key in ``sys.modules``. This list is the screen's frame:
#: a directory absent from it is not screened, which is why the list is here and not inferred.
FLAT_IMPORT_ROOTS = ("bench", "bench/results", "ci", "tests/ops", "rust/tools")


def _modules_by_basename(roots, repo=REPO):
    """Map base name -> [paths]. Shared by the screen and its positive control."""
    seen = collections.defaultdict(list)
    for rel in roots:
        d = repo / rel
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.py")):
            seen[f.name].append(f"{rel}/{f.name}")
    return seen


def _collisions(roots, repo=REPO):
    return {k: v for k, v in _modules_by_basename(roots, repo).items() if len(v) > 1}


def test_no_two_flat_imported_modules_share_a_base_name():
    """The screen. This is the invariant whose violation produced the three AttributeErrors.

    Quote the failure text, never the failure count (R13) — the message names the colliding paths,
    because "1 collision" is not actionable and "bench/x.py vs ci/x.py" is.
    """
    collisions = _collisions(FLAT_IMPORT_ROOTS)
    assert not collisions, (
        "two files that are imported by flat name share a base name, so whichever is imported "
        "first binds sys.modules for the whole process and the second silently gets the first:\n  "
        + "\n  ".join(f"{name}: " + " vs ".join(paths) for name, paths in sorted(collisions.items()))
        + "\nsys.path ordering cannot fix this -- import consults sys.modules first. Rename one, "
          "or load it via importlib under a unique key."
    )


def test_the_collision_screen_can_actually_fail(tmp_path):
    """Positive control. Feed the screen a known collision; it must report it.

    Without this, the screen above is indistinguishable from `assert True` on the day someone
    breaks its walk. The synthetic tree is built in tmp_path, never in the repo.
    """
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "device_state.py").write_text("# planted\n", encoding="utf-8")
    (tmp_path / "b" / "device_state.py").write_text("# planted\n", encoding="utf-8")
    (tmp_path / "a" / "unique_here.py").write_text("# planted\n", encoding="utf-8")

    found = _collisions(("a", "b"), repo=tmp_path)
    assert "device_state.py" in found, "the screen failed to see a planted collision"
    assert found["device_state.py"] == ["a/device_state.py", "b/device_state.py"]
    assert "unique_here.py" not in found, "the screen invented a collision that is not there"


def _import_leak(paths):
    """Import each path by flat name and return the sys.path entries left behind.

    Returns (leaked_entries, errors). Import errors are reported separately: a module that will not
    import is not a path leak, and reporting it as one would be a detection from an instrument
    error (R13).

    Deliberately does **not** purge `sys.modules` between imports. The first cut did, and the
    interpreter died with a hard fault rather than a test failure: dropping already-imported C
    extension modules (numpy's, protobuf's) and letting them re-initialise is a documented way to
    crash CPython. That crash was ERROR(instrument) -- the screen breaking, not a defect found --
    and leaving the cache alone costs the screen nothing, because it measures `sys.path`.
    """
    leaked, errors = [], []
    for p in paths:
        before = list(sys.path)
        sys.path.insert(0, str(p.parent))
        try:
            __import__(p.stem)
        except BaseException as exc:  # noqa: BLE001 - any failure is an import failure, not a leak
            errors.append(f"{p.name}: {type(exc).__name__}: {exc}")
        finally:
            after = list(sys.path)
            try:
                after.remove(str(p.parent))
            except ValueError:
                pass
            for entry in after:
                if entry not in before and entry not in leaked:
                    leaked.append(f"{p.name} left {entry!r} on sys.path")
            sys.path[:] = before
    return leaked, errors


def test_importing_bench_modules_does_not_leak_sys_path():
    """The prescribed invariant, executed rather than grepped.

    A leaked `bench/` entry does not by itself cause the reported failure, but it decides the next
    name race in bench's favour, silently and process-wide.
    """
    bench = REPO / "bench"
    if not bench.is_dir():
        pytest.fail("ERROR(instrument): bench/ not found; the screen has no subject")
    # Test modules import their neighbours at collection time; screening them here would double-
    # import the suite. The library modules are the ones other people's code pulls in.
    paths = [p for p in sorted(bench.glob("*.py")) if not p.name.startswith("test_")]
    leaked, _errors = _import_leak(paths)
    assert not leaked, "modules under bench/ mutated sys.path and did not restore it:\n  " + \
                       "\n  ".join(leaked)


def test_the_sys_path_leak_screen_can_actually_fail(tmp_path):
    """Positive control for the leak screen: a module that deliberately leaks must be caught."""
    leaky = tmp_path / "planted_leaky_module.py"
    leaky.write_text("import sys\nsys.path.insert(0, '/planted/leak/path')\n", encoding="utf-8")
    clean = tmp_path / "planted_clean_module.py"
    clean.write_text("VALUE = 1\n", encoding="utf-8")

    leaked, errors = _import_leak([leaky, clean])
    assert not errors, f"the control's own fixtures failed to import: {errors}"
    assert any("planted_leaky_module" in m for m in leaked), "the leak screen missed a planted leak"
    assert not any("planted_clean_module" in m for m in leaked), "the leak screen cried wolf"
