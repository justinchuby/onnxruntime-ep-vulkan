#!/usr/bin/env python3
"""`bench/public_paths.py`, and the two failures it exists to make impossible.

These tests run without a GPU, without a model and without Foundry, which is the point: the
probe they guard (`bench/results/probe_weight_reread.py`) needs a 2 GB Phi-3.5 checkout and a
compiled SPIR-V module, so it cannot run in CI, so a guard that only lives inside its `main()`
is a guard nothing ever exercises. Everything under test here is either pure or shells out to
`git` in a repository that is right there.

Each guard is exercised in **both** states. A refusal that has never been seen to admit anything
is indistinguishable from a function that returns `False`, and an admission that has never been
seen to refuse is a comment.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(ROOT / "bench" / "results"))
sys.path.insert(0, str(ROOT / "rust" / "tools"))

import public_paths as pp  # noqa: E402


# -- public_path ---------------------------------------------------------------------------------


def test_a_repo_file_becomes_a_repo_relative_posix_path():
    got = pp.public_path(ROOT / "bench" / "results" / "weight_reread_phi35.json")
    assert got == "bench/results/weight_reread_phi35.json"
    assert "\\" not in got, "a record written on Windows must compare equal to one on Linux"


def test_a_home_file_keeps_its_shape_and_loses_the_account():
    model = pathlib.Path.home() / ".modelcache" / "vendor" / "m.onnx"
    got = pp.public_path(model)
    assert got == "<home>/.modelcache/vendor/m.onnx"
    assert pathlib.Path.home().name not in got, (
        "the shape of the location is public information; the account name is not"
    )


def test_a_file_outside_both_is_reduced_to_its_name(tmp_path):
    # `tmp_path` is under neither the repo nor (on most runners) the home directory. Where it
    # happens to be under home, the home branch is correct and equally leak-free, so accept
    # either — the assertion that matters is that no absolute path survives.
    got = pp.public_path(tmp_path / "some" / "deep" / "artifact.json")
    assert got in ("<external>/artifact.json",) or got.startswith("<home>/")
    pp.assert_public(got)


def test_public_path_scrubs_an_account_name_that_is_not_in_the_prefix(monkeypatch):
    """Stripping a home prefix does not strip an account name further down the path.

    `<home>/AppData/Local/Temp/pytest-of-hortensia/...` is home-relative, POSIX, and still says
    who ran it. This is not hypothetical: it is what `tmp_path` looks like on Windows, and it is
    what caught this module's first draft.
    """
    monkeypatch.setenv("USERNAME", "hortensia")
    monkeypatch.setenv("USER", "hortensia")
    home = pathlib.Path.home()
    got = pp.public_path(home / "Temp" / "pytest-of-hortensia" / "run" / "x.json")
    assert got == "<home>/Temp/pytest-of-<user>/run/x.json"
    pp.assert_public(got)


def test_the_scrub_is_case_insensitive_and_leaves_everything_else_alone(monkeypatch):
    monkeypatch.setenv("USERNAME", "Hortensia")
    monkeypatch.setenv("USER", "Hortensia")
    assert pp._scrub_account("a/HORTENSIA/b/hortensia/c") == "a/<user>/b/<user>/c"
    assert pp._scrub_account("a/petunia/b") == "a/petunia/b"
    # Two accounts' worth of tokens, one pass each, no interference.
    assert pp._scrub_account("") == ""


def test_public_path_and_assert_public_agree_by_construction(tmp_path):
    """Whatever `public_path` produces, `assert_public` admits. If that ever stops being true one
    of the two has a rule the other does not, and a probe would start failing at write time on a
    machine whose paths happen to be shaped differently."""
    for candidate in [
        ROOT / "bench" / "results" / "x.json",
        pathlib.Path.home() / ".modelcache" / "vendor" / "m.onnx",
        tmp_path / "a" / "b" / "c.json",
        ROOT,
        pathlib.Path.home(),
    ]:
        pp.assert_public(pp.public_path(candidate), field=str(candidate.name))


def test_the_repo_wins_over_home_when_the_repo_is_inside_home():
    """The repo is very often *inside* the home directory, and repo-relative is strictly more
    useful. This locks the branch order rather than the accident of where the repo happens to be
    checked out: the test only asserts anything when the two cases actually overlap."""
    try:
        ROOT.relative_to(pathlib.Path.home().resolve())
    except ValueError:
        pytest.skip("this checkout is not inside the home directory, so there is no overlap")
    got = pp.public_path(ROOT / "bench" / "public_paths.py")
    assert got == "bench/public_paths.py"


# -- assert_public: both states -------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "bench/results/weight_reread_phi35.json",
        "<home>/.modelcache/vendor/m.onnx",
        "<external>/artifact.json",
        "",
    ],
)
def test_assert_public_admits_what_public_path_produces(value):
    assert pp.assert_public(value) == value


@pytest.mark.parametrize(
    "value",
    [
        r"C:\Users\someone\.modelcache\vendor\m.onnx",
        r"\\server\share\m.onnx",
        "C:/Users/someone/m.onnx",
        "/home/someone/.modelcache/m.onnx",
        "/tmp/m.onnx",
        r"bench\results\x.json",
    ],
)
def test_assert_public_refuses_anything_that_still_looks_like_a_real_path(value):
    with pytest.raises(pp.LeakError):
        pp.assert_public(value, field="onnx_file")


def test_assert_public_catches_the_account_name_even_in_a_relative_path(monkeypatch):
    """The check the others miss.

    A path made relative to the *wrong* base can be relative, POSIX, and still spell out who ran
    it. `models/hortensia/phi.onnx` passes every structural test and is still a leak.
    """
    monkeypatch.setenv("USERNAME", "hortensia")
    monkeypatch.setenv("USER", "hortensia")
    with pytest.raises(pp.LeakError) as e:
        pp.assert_public("models/hortensia/phi.onnx", field="onnx_file")
    assert "onnx_file" in str(e.value)
    # And it is case-insensitive, because Windows paths are.
    with pytest.raises(pp.LeakError):
        pp.assert_public("models/Hortensia/phi.onnx")
    # A name that is not the account still passes: the screen discriminates.
    assert pp.assert_public("models/petunia/phi.onnx")


def test_assert_public_refuses_a_non_string():
    with pytest.raises(pp.LeakError):
        pp.assert_public(pathlib.Path("bench/x.json"))  # type: ignore[arg-type]


def test_a_two_letter_account_name_does_not_reject_everything(monkeypatch):
    """The account-name check is skipped for very short names, and that is deliberate.

    A user called `jo` would otherwise make `bench/json/...` unpublishable. The alternative — a
    guard that refuses everything — is not a stricter guard, it is a guard that gets disabled.
    """
    monkeypatch.setenv("USERNAME", "jo")
    monkeypatch.setenv("USER", "jo")
    assert pp.assert_public("bench/results/json_things.json")


# -- is_tracked: both states, and the fail-closed middle ------------------------------------------


def test_a_committed_file_reads_as_tracked():
    assert pp.is_tracked(ROOT / "bench" / "results" / "weight_reread_phi35.json") is True


def test_an_ignored_scratch_file_reads_as_untracked():
    assert pp.is_tracked(pp.SCRATCH / "weight_reread_phi35.json") is False


def test_a_path_outside_the_repository_reads_as_untracked(tmp_path):
    assert pp.is_tracked(tmp_path / "x.json") is False


def test_not_knowing_counts_as_tracked(monkeypatch):
    """The asymmetry, exercised rather than described.

    A false 'tracked' costs an operator one flag. A false 'untracked' costs committed evidence.
    So every answer that is not an unambiguous exit 0 or exit 1 is 'tracked'.
    """
    def boom(*a, **k):
        raise OSError("no git on this machine")

    monkeypatch.setattr(pp.subprocess, "run", boom)
    assert pp.is_tracked(ROOT / "bench" / "_scratch" / "x.json") is True

    class Weird:
        returncode = 129
        stdout = b""
        stderr = b"fatal: not a git repository"

    monkeypatch.setattr(pp.subprocess, "run", lambda *a, **k: Weird())
    assert pp.is_tracked(ROOT / "bench" / "_scratch" / "x.json") is True

    class Untracked:
        returncode = 1
        stdout = b""
        stderr = b""

    monkeypatch.setattr(pp.subprocess, "run", lambda *a, **k: Untracked())
    assert pp.is_tracked(ROOT / "bench" / "_scratch" / "x.json") is False


# -- resolve_out ----------------------------------------------------------------------------------


def test_the_default_destination_is_untracked_and_gitignored():
    dest = pp.resolve_out(None, "weight_reread_phi35.json")
    assert dest == pp.SCRATCH / "weight_reread_phi35.json"
    assert pp.is_tracked(dest) is False
    assert dest.parent.is_dir(), "the parent must exist so a ten-minute run cannot die on mkdir"
    # `bench/.gitignore` is what makes this safe, so assert it rather than assuming it.
    assert "_scratch/" in (ROOT / "bench" / ".gitignore").read_text(encoding="utf-8")


def test_writing_over_a_tracked_file_is_refused_by_default():
    with pytest.raises(pp.TrackedWriteRefused) as e:
        pp.resolve_out("bench/results/weight_reread_phi35.json", "weight_reread_phi35.json")
    msg = str(e.value)
    assert "bench/results/weight_reread_phi35.json" in msg
    assert "--allow-tracked" in msg, "a refusal must say what would have been allowed"
    assert "\\" not in msg, "the refusal itself must not leak an absolute path"


def test_allow_tracked_is_the_operator_saying_so():
    dest = pp.resolve_out(
        "bench/results/weight_reread_phi35.json",
        "weight_reread_phi35.json",
        allow_tracked=True,
    )
    assert dest == (ROOT / "bench" / "results" / "weight_reread_phi35.json").resolve()


def test_an_explicit_untracked_out_is_honoured(tmp_path):
    dest = pp.resolve_out(tmp_path / "deep" / "x.json", "weight_reread_phi35.json")
    assert dest == tmp_path / "deep" / "x.json"
    assert dest.parent.is_dir()


def test_a_relative_out_is_relative_to_the_repo_not_the_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dest = pp.resolve_out("bench/_scratch/x.json", "weight_reread_phi35.json")
    assert dest == (ROOT / "bench" / "_scratch" / "x.json").resolve()


# -- the probe, end to end at its edges -----------------------------------------------------------


def test_the_probe_refuses_a_tracked_out_without_running_anything():
    """`main` must refuse *before* it walks a module or resolves a model.

    Run as a subprocess with the real committed record as `--out`. If the guard were downstream
    of the work, this would take minutes and need a model; it returns in well under the timeout
    with exit 4 and touches nothing.
    """
    record = ROOT / "bench" / "results" / "weight_reread_phi35.json"
    before = record.read_bytes()
    done = subprocess.run(
        [sys.executable, "bench/results/probe_weight_reread.py",
         "--out", "bench/results/weight_reread_phi35.json"],
        cwd=ROOT, capture_output=True, text=True, timeout=180,
    )
    assert done.returncode == 4, done.stdout + done.stderr
    assert "tracked_write_refused" in done.stderr
    assert record.read_bytes() == before, "the refused run modified the record anyway"


def test_the_probe_has_no_unconditional_write_to_the_results_directory():
    """The defect this fixed, stated as a property of the source.

    A source-level assertion, and it is the weaker half of the pair — the subprocess test above
    exercises the behaviour. This one exists because the *shape* is what regresses: the original
    line was a hardcoded `ROOT / "bench" / "results" / ...` destination, and someone adding a
    second output later would reach for exactly that again.
    """
    src = (ROOT / "bench" / "results" / "probe_weight_reread.py").read_text(encoding="utf-8")
    assert 'ROOT / "bench" / "results" / "weight_reread_phi35.json"' not in src
    assert "resolve_out(" in src
    assert src.count("write_text(json.dumps(") == 1, (
        "exactly one record write, and it must stay inline in this file so "
        "ci/phi35_identity_audit.py can still see this probe as a producer"
    )


def test_the_committed_record_is_the_evidence_this_pr_did_not_re_run():
    """A record produced before this change, left exactly as it was.

    `onnx_file` in the committed record is still the old absolute path, and that is correct: this
    PR has no Phi-3.5 checkout and no compiled module, so re-running the probe to regenerate the
    record would be fabricating a measurement. The screening applies to the *next* run. This test
    says so out loud so the discrepancy reads as a decision rather than as an oversight.
    """
    record = json.loads(
        (ROOT / "bench" / "results" / "weight_reread_phi35.json").read_text(encoding="utf-8")
    )
    assert "onnx_sha256" in record, "the identifying field was always the hash, not the path"
    assert len(record["onnx_sha256"]) == 64
    # And the code that would write it today produces a public form for the same input.
    assert pp.assert_public(pp.public_path(pathlib.Path(record["onnx_file"])))
