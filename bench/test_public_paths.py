#!/usr/bin/env python3
"""The screen in ``bench/public_paths.py``, driven from both sides.

A scrub that has only ever been run on clean records proves nothing: it would pass identically if
it were ``return obj``. So every recognised absolute form is planted here and required to be
rewritten, and — separately — required to be *caught* by :func:`~bench.public_paths.assert_public`
when the scrub is bypassed. Those are two different failures (a scrub that misses a form, and a
screen that misses a leak) and they need two different tests.

The other half is the guard on the write. Its whole job is to answer "would this clobber committed
evidence?" and to answer *conservatively* when it cannot tell, so the interesting tests are the
ones where git does not answer: a corrupt repository, a missing binary, a timeout. Each of those
must come back "tracked", because the alternative is a probe that overwrites a witness on a machine
where git happened to be broken.

Every planted path here uses an invented user and an invented layout. Nothing in this file is a
real path from anybody's machine.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from public_paths import (  # noqa: E402
    REDACTED,
    PublicPathError,
    assert_public,
    dump_public_json,
    is_git_tracked,
    repo_root,
    scrub_public,
    untracked_default,
)

ROOT = repo_root()

#: Invented, one per recognised form. The Windows profile name has a space in it on purpose: that
#: is the case where a tail-matching regex stops early and leaves the second half of a username in
#: the record.
PLANTED = {
    "windows_drive": r"C:\Users\ada byron\.cache\models\vendor\m\v1\model.onnx",
    "windows_forward": "D:/Users/ada/.cache/models/vendor/m/v1/model.onnx",
    "unc": r"\\fileserver\share\models\vendor\m\v1\model.onnx",
    "posix_home": "/home/ada/.cache/models/vendor/m/v1/model.onnx",
    "posix_users": "/Users/ada/.cache/models/vendor/m/v1/model.onnx",
    "posix_mnt": "/mnt/d/models/vendor/m/v1/model.onnx",
    "ci_home": "/github/home/.cache/models/vendor/m/v1/model.onnx",
    "tilde": "~/.cache/models/vendor/m/v1/model.onnx",
}


class TestEveryRecognisedFormIsRewritten:
    @pytest.mark.parametrize("form", sorted(PLANTED))
    def test_a_planted_absolute_path_does_not_survive_the_scrub(self, form):
        out = scrub_public({"onnx_file": PLANTED[form]})
        assert out["onnx_file"] == REDACTED, f"{form} survived: {out['onnx_file']!r}"

    @pytest.mark.parametrize("form", sorted(PLANTED))
    def test_the_screen_catches_what_the_scrub_was_not_asked_to_fix(self, form):
        with pytest.raises(PublicPathError) as excinfo:
            assert_public({"onnx_file": PLANTED[form]})
        assert "onnx_file" in str(excinfo.value)

    def test_the_username_is_gone_and_not_merely_shortened(self):
        """The specific defect a tail-matching regex has: it stops at the first space."""
        out = scrub_public({"onnx_file": PLANTED["windows_drive"]})
        assert "byron" not in json.dumps(out)
        assert "ada" not in json.dumps(out)

    def test_a_key_leaks_exactly_as_much_as_a_value(self):
        out = scrub_public({PLANTED["posix_home"]: {"median_ms": 1.0}})
        assert list(out) == [REDACTED]

    def test_a_path_nested_in_a_list_of_dicts_is_reached(self):
        out = scrub_public({"runs": [{"env": {"MODEL": PLANTED["posix_home"]}}]})
        assert out["runs"][0]["env"]["MODEL"] == REDACTED

    def test_scrubbing_twice_changes_nothing_the_second_time(self):
        once = scrub_public({"onnx_file": PLANTED["unc"], "note": "ok"})
        assert scrub_public(once) == once

    def test_a_path_embedded_in_prose_is_removed_from_the_match_to_the_end(self):
        out = scrub_public({"why": "loaded from /home/ada/models/m.onnx before the run"})
        assert "ada" not in out["why"]
        assert out["why"].startswith("loaded from ")


class TestWhatMustSurviveUntouched:
    """A scrub that damages the data is not safer than one that leaks; it is differently wrong."""

    def test_an_onnx_node_name_is_not_a_filesystem_path(self):
        names = [
            "/model/layers.0/mlp/down_proj/MatMul",
            "/model/layers.31/self_attn/o_proj/MatMulNBits",
            "/lm_head/MatMul_output_0",
        ]
        assert scrub_public({"nodes": names})["nodes"] == names

    def test_a_node_name_whose_segment_collides_with_a_recognised_root_is_untouched(self):
        """``/data/`` and ``/var/`` are roots in the scrub's list *and* plausible module names.

        Anchoring the embedded search is what keeps the first fact from destroying the second.
        """
        names = ["/model/data/Gather", "/model/var/Add", "/encoder/tmp/Cast"]
        assert scrub_public({"nodes": names})["nodes"] == names

    def test_a_clean_record_is_returned_unchanged(self):
        record = {
            "schema": 3,
            "matmulnbits_nodes": 161,
            "prefill_m_values": [2, 4, 5],
            "ok": True,
            "ratio": 1.5,
            "missing": None,
        }
        assert scrub_public(record) == record
        assert_public(record)

    def test_a_relative_path_is_left_alone(self):
        record = {"out": "bench/results/weight_reread_phi35.json"}
        assert scrub_public(record) == record


class TestInsideTheRepositoryIsRelativisedNotRedacted:
    def test_a_repo_path_becomes_repo_relative_and_therefore_openable(self):
        inside = str(ROOT / "bench" / "results" / "weight_reread_phi35.json")
        assert scrub_public({"out": inside}) == {
            "out": "bench/results/weight_reread_phi35.json"
        }

    def test_the_relativised_form_passes_the_screen(self):
        assert_public(scrub_public({"out": str(ROOT / "docs" / "PERF.md")}))


class TestTheGuardAndTheWriteAnswerAboutTheSameFile:
    def test_a_relative_out_is_resolved_before_git_is_asked(self, tmp_path, monkeypatch):
        """The concrete defect: cwd-relative for the write, root-relative for the guard.

        Run from ``bench/results/`` with ``--out weight_reread_phi35.json``, an unresolved guard
        asks git about ``<root>/weight_reread_phi35.json`` — untracked — and the write lands on the
        committed witness next to the script.
        """
        monkeypatch.chdir(ROOT / "bench" / "results")
        with pytest.raises(PublicPathError, match="git tracks it"):
            dump_public_json({"ok": True}, Path("weight_reread_phi35.json"))

    def test_a_junction_or_symlink_to_a_tracked_file_is_seen_through(self, tmp_path):
        link = tmp_path / "alias.json"
        target = ROOT / "bench" / "results" / "weight_reread_phi35.json"
        if not target.exists():
            pytest.skip("committed witness absent in this tree")
        try:
            link.symlink_to(target)
        except (OSError, NotImplementedError):
            pytest.skip("this environment does not permit creating symlinks")
        assert is_git_tracked(link) is True

    def test_a_directory_junction_is_seen_through(self, tmp_path):
        """Windows junctions need no privilege, so this arm runs where symlinks cannot."""
        if os.name != "nt":
            pytest.skip("junctions are a Windows concept")
        junction = tmp_path / "j"
        done = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(junction), str(ROOT / "bench" / "results")],
            capture_output=True,
            text=True,
            check=False,
        )
        if done.returncode != 0:
            pytest.skip(f"mklink /J unavailable: {done.stdout}{done.stderr}")
        assert is_git_tracked(junction / "weight_reread_phi35.json") is True


class TestTheGuardFailsClosedWhenGitDoesNotAnswer:
    """Every one of these must say "tracked". "I could not tell" and "it is safe to overwrite" are
    different answers, and only one of them destroys evidence."""

    def test_a_corrupt_repository_is_treated_as_tracked(self, tmp_path):
        broken = tmp_path / "broken"
        (broken / ".git").mkdir(parents=True)
        (broken / ".git" / "HEAD").write_text("this is not a ref\n", encoding="utf-8")
        target = broken / "record.json"
        target.write_text("{}", encoding="utf-8")
        assert is_git_tracked(target, root=broken) is True

    def test_a_directory_that_is_not_a_repository_at_all_is_treated_as_tracked(self, tmp_path):
        plain = tmp_path / "plain"
        plain.mkdir()
        target = plain / "record.json"
        target.write_text("{}", encoding="utf-8")
        assert is_git_tracked(target, root=plain) is True

    def test_a_missing_git_binary_is_treated_as_tracked(self, tmp_path, monkeypatch):
        def explode(*_args, **_kwargs):
            raise FileNotFoundError("git")

        monkeypatch.setattr(subprocess, "run", explode)
        assert is_git_tracked(tmp_path / "record.json", root=tmp_path) is True

    def test_a_timeout_is_treated_as_tracked(self, tmp_path, monkeypatch):
        def hang(*_args, **_kwargs):
            raise subprocess.TimeoutExpired(cmd="git", timeout=30)

        monkeypatch.setattr(subprocess, "run", hang)
        assert is_git_tracked(tmp_path / "record.json", root=tmp_path) is True

    def test_a_destination_outside_the_repository_is_untracked_without_asking_git(self, tmp_path):
        """Otherwise git's 128 for an out-of-tree path would trip the fail-safe on every scratch
        file and make the safe default unusable."""
        assert is_git_tracked(tmp_path / "scratch.json") is False


class TestTheWriteItself:
    def test_a_tracked_destination_is_refused_by_default(self):
        target = ROOT / "docs" / "PERF.md"
        before = target.read_bytes()
        with pytest.raises(PublicPathError, match="git tracks it"):
            dump_public_json({"ok": True}, target)
        assert target.read_bytes() == before

    def test_allow_tracked_is_the_deliberate_escape(self, tmp_path):
        out = tmp_path / "record.json"
        written = dump_public_json({"ok": True}, out, allow_tracked=True)
        assert json.loads(written.read_text(encoding="utf-8")) == {"ok": True}

    def test_the_write_scrubs_before_it_lands(self, tmp_path):
        out = dump_public_json({"onnx_file": PLANTED["posix_home"]}, tmp_path / "r.json")
        text = out.read_text(encoding="utf-8")
        assert "ada" not in text
        assert REDACTED in text

    def test_a_form_the_scrub_missed_would_stop_the_write_rather_than_reach_disk(
        self, tmp_path, monkeypatch
    ):
        """The screen is not decoration: bypass the scrub and the write must still refuse."""
        import public_paths

        monkeypatch.setattr(public_paths, "scrub_public", lambda obj, **_kw: obj)
        out = tmp_path / "r.json"
        with pytest.raises(PublicPathError):
            public_paths.dump_public_json({"onnx_file": PLANTED["posix_home"]}, out)
        assert not out.exists()

    def test_missing_parent_directories_are_created(self, tmp_path):
        out = dump_public_json({"ok": True}, tmp_path / "a" / "b" / "r.json")
        assert out.exists()


class TestTheSafeDefault:
    def test_the_default_lands_where_git_already_ignores(self):
        out = untracked_default("weight_reread_phi35.json")
        assert out.parent == ROOT / "bench" / "_scratch"
        assert is_git_tracked(out) is False

    def test_the_ignore_rule_is_a_reused_decision_not_a_new_one(self):
        """``bench/.gitignore`` must actually contain the rule this default depends on."""
        text = (ROOT / "bench" / ".gitignore").read_text(encoding="utf-8")
        assert "_scratch/" in text


class TestThisModuleNamesNoRealPath:
    def test_the_seam_spells_no_literal_cache_path(self):
        """Asserted with the *production* screen's own pattern rather than a hand-written
        approximation, so this cannot pass while ``ci/check_hardcoded_foundry_paths.py`` fails.

        The seam is not on that screen's allowlist, and must not need to be.
        """
        sys.path.insert(0, str(ROOT / "ci"))
        from check_hardcoded_foundry_paths import _PATTERN, is_allowlisted

        text = (ROOT / "bench" / "public_paths.py").read_text(encoding="utf-8")
        assert _PATTERN.search(text) is None
        assert not is_allowlisted("bench/public_paths.py")
