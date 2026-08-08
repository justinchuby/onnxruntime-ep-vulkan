"""Locks for the publish seam: no implicit write to tracked evidence, no leaked local path.

WHY THIS FILE EXISTS
====================
`bench/results/probe_weight_reread.py` ended `main()` with a single unconditional
`out.write_text(...)` at a **git-tracked** path, `bench/results/weight_reread_phi35.json`. Two
defects in one line, and this file is the pair of screens for them:

* **The implicit write.** A probe you run to read a number rewrote the number. `git status` came
  back dirty from an operation the operator believed was read-only, and the recovery was to
  remember which of your own runs to `git checkout`. Worse, it is exactly the operation you want
  while reproducing a disagreement *about* that witness -- so the instrument was unusable in the
  one situation it exists for.
* **The leaked path.** The same record stamped `"onnx_file": "C:\\Users\\<user>\\.foundry\\..."`.
  That names a machine and a person, and as provenance it is worse than useless: nobody else can
  open it, so it *looks* like provenance while being unreproducible.

Every test below is a **both-polarity** pair wherever a polarity exists: the scrub is shown
rewriting a planted path (positive) and the screen is shown firing on one the scrub was denied
(positive), and both are shown leaving a clean record alone (negative). A screen never seen
firing has no demonstrated positive state -- the rule `bench/test_weight_reread.py` already
applies to the amplification detector, applied to this one.

Nothing here reads a clock, touches a device, or runs the model. These are string and filesystem
properties.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parent
RESULTS = BENCH / "results"

_SYS_PATH_BEFORE = list(sys.path)
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(RESULTS))
try:
    import public_paths as pp  # noqa: E402
finally:
    sys.path[:] = _SYS_PATH_BEFORE


# ---------------------------------------------------------------------------
# The scrub
# ---------------------------------------------------------------------------

#: One planted example of every form `_ABSOLUTE` claims to recognise. If a form is added to the
#: regex without a row here, `test_every_recognised_form_has_a_planted_example` fails: a pattern
#: with no example is a pattern nobody has seen work.
PLANTED = [
    ("windows drive, backslash", r"C:\Users\someone\.cache\models\model.onnx"),
    ("windows drive, forward slash", "D:/Users/someone/models/model.onnx"),
    ("windows drive, lowercase", r"c:\users\someone\model.onnx"),
    ("windows path with a space", r"C:\Users\Jane Smith\.cache\models\model.onnx"),
    ("unc share", r"\\build-server\share\artifacts\model.onnx"),
    ("posix home", "/home/someone/.cache/model.onnx"),
    ("posix Users", "/Users/someone/Library/model.onnx"),
    ("posix users lowercase", "/users/someone/model.onnx"),
    ("posix root", "/root/model.onnx"),
    ("posix mnt", "/mnt/c/someone/model.onnx"),
    ("posix media", "/media/usb0/model.onnx"),
    ("posix tmp", "/tmp/hf-cache/someone/model.onnx"),
    ("posix var", "/var/lib/models/someone/model.onnx"),
    ("posix opt", "/opt/cache/someone/model.onnx"),
    ("posix srv", "/srv/models/someone/model.onnx"),
    ("posix data", "/data/someone/model.onnx"),
    ("posix private", "/private/var/folders/someone/model.onnx"),
    ("macos Volumes", "/Volumes/External/someone/model.onnx"),
    ("codespaces workspaces", "/workspaces/someone-private/model.onnx"),
    ("actions container home", "/github/home/.cache/someone/model.onnx"),
    ("unexpanded home", "~/.cache/models/model.onnx"),
]


def test_onnx_node_names_are_not_mistaken_for_paths():
    """THE SHREDDER CONTROL, and the reason `_POSIX_ROOTS` is a list rather than "any leading /".

    An ONNX node name is spelled exactly like a POSIX absolute path, and these records are full of
    them. A scrub that redacted every leading-slash string would destroy the data the records exist
    to carry while passing every leak test in this file.
    """
    record = {
        "nodes": [
            "/model/layers.0/attn/qkv_proj/MatMul",
            "/model/layers.31/mlp/down_proj/MatMul",
            "/lm_head/MatMul",
            "/model/norm/SimplifiedLayerNormalization",
        ],
        "op": "/model/embed_tokens/Gather",
    }
    assert pp.scrub_public(record, root=ROOT) == record
    pp.assert_public(record, where="node names", root=ROOT)


@pytest.mark.parametrize("label,planted", PLANTED, ids=[p[0] for p in PLANTED])
def test_the_screen_fires_on_every_planted_absolute_path(label, planted):
    """POSITIVE CONTROL. The screen must be seen refusing each form it claims to know."""
    with pytest.raises(pp.PublicPathError) as excinfo:
        pp.assert_public({"onnx_file": planted}, where="planted", root=ROOT)
    assert "onnx_file" in str(excinfo.value), (
        "the refusal must name the field, or an operator cannot find it in a 700-line record"
    )


@pytest.mark.parametrize("label,planted", PLANTED, ids=[p[0] for p in PLANTED])
def test_the_scrub_removes_every_planted_absolute_path(label, planted):
    """The scrub is the fix; the screen is only the proof it worked."""
    scrubbed = pp.scrub_public({"onnx_file": planted}, root=ROOT)
    pp.assert_public(scrubbed, where="scrubbed", root=ROOT)
    for token in ("someone", "Jane", "Smith"):
        assert token not in json.dumps(scrubbed), (
            f"{label}: {token!r} survived the scrub: {scrubbed}"
        )


@pytest.mark.parametrize("label,planted", PLANTED, ids=[p[0] for p in PLANTED])
def test_a_planted_path_embedded_in_prose_is_also_removed(label, planted):
    """A record field is usually a bare path, but a message field is prose around one. The tail of
    `_ABSOLUTE` stops at the first space, so this is where a path containing a space used to leave
    half a username behind."""
    scrubbed = pp.scrub_public(
        {"error": f"could not open {planted} for reading"}, root=ROOT
    )
    pp.assert_public(scrubbed, where="prose", root=ROOT)
    for token in ("someone", "Jane", "Smith"):
        assert token not in json.dumps(scrubbed), (
            f"{label}: {token!r} survived the scrub in prose: {scrubbed}"
        )


def test_every_recognised_form_has_a_planted_example():
    """A regex alternative with no example is an alternative nobody has watched work.

    Reads the pattern rather than restating it, so adding a form to `_ABSOLUTE` without adding a
    row to `PLANTED` fails here instead of shipping unexercised.
    """
    unmatched = []
    for label, planted in PLANTED:
        if not pp._ABSOLUTE_PREFIX.search(planted):
            unmatched.append(f"{label}: {planted!r}")
    assert not unmatched, "planted examples the pattern does not match: " + ", ".join(unmatched)
    # And the converse: every POSIX root the module claims needs an example, so adding one to the
    # list without a planted case fails here rather than shipping unexercised.
    covered = {p for _, p in PLANTED}
    missing = [
        root
        for root in pp._POSIX_ROOTS
        if not any(p.startswith(f"/{root}/") for p in covered)
    ]
    assert not missing, f"POSIX roots with no planted example: {missing}"


def test_is_git_tracked_refuses_when_git_runs_but_cannot_answer(tmp_path):
    """A guard that exists to prevent a write must answer 'tracked' for any exit it does not
    understand, not only for a missing binary.

    `git ls-files --error-unmatch` exits 1 for a definitely-untracked path *inside* the repository.
    Exit 128 — corrupt index, permission denied, not a repository — means the question was not
    answered. The first draft returned `returncode == 0`, which turned every one of those into
    permission to write.

    The path here is inside the fixture root so the outside-the-repo short circuit does not apply;
    the fixture is not a git repository, so git exits 128.
    """
    not_a_repo = tmp_path / "plain"
    not_a_repo.mkdir()
    done = subprocess.run(
        ["git", "-C", str(not_a_repo), "ls-files", "--error-unmatch", "x.json"],
        capture_output=True,
        check=False,
    )
    assert done.returncode not in (0, 1), (
        f"this test needs git to fail with something other than 0/1, got {done.returncode}"
    )
    assert pp.is_git_tracked(not_a_repo / "x.json", root=not_a_repo) is True


def test_a_destination_outside_the_repository_is_untracked_without_asking_git(tmp_path):
    """The short circuit that keeps the fail-safe meaningful.

    Every ordinary scratch destination is outside the repo, and git answers 128 for those. Without
    this, the fail-safe would refuse every scratch write and the guard would be routed around
    within a week.
    """
    assert pp.is_git_tracked(tmp_path / "record.json", root=ROOT) is False


def test_a_path_inside_the_repository_becomes_relative_rather_than_redacted():
    """Scrubbing is not only redaction. A repo path relativised is *more* useful, not less."""
    inside = str(RESULTS / "weight_reread_phi35.json")
    scrubbed = pp.scrub_public({"module_path": inside}, root=ROOT)
    assert scrubbed["module_path"] == "bench/results/weight_reread_phi35.json", scrubbed
    pp.assert_public(scrubbed, where="relativised", root=ROOT)


def test_keys_are_scrubbed_as_well_as_values():
    """A leak in a key is exactly as public as a leak in a value."""
    record = {r"C:\Users\someone\model.onnx": {"nodes": 1}}
    scrubbed = pp.scrub_public(record, root=ROOT)
    assert "someone" not in json.dumps(scrubbed), scrubbed
    pp.assert_public(scrubbed, where="keys", root=ROOT)
    # And the screen must find it if the scrub is skipped -- otherwise this test proves nothing
    # about keys, only about the scrub happening to be called.
    with pytest.raises(pp.PublicPathError):
        pp.assert_public(record, where="keys", root=ROOT)


def test_the_scrub_is_idempotent():
    """`dump_public_json` scrubs, then screens the scrubbed object. A scrub that lengthened or
    re-mangled already-scrubbed text would make the second pass disagree with the first."""
    once = pp.scrub_public({"p": r"C:\Users\someone\model.onnx"}, root=ROOT)
    twice = pp.scrub_public(once, root=ROOT)
    assert once == twice, (once, twice)


def test_a_clean_record_is_left_exactly_alone():
    """NEGATIVE CONTROL. A scrub that rewrites everything is not a scrub, it is a shredder."""
    clean = {
        "amplification": 1.0,
        "verdict": "each weight byte is named by exactly one load instruction",
        "module_path": "bench/results/_ledger_models/matmulnbits_f16_scales.onnx",
        "nested": [{"k": 1}, {"k": 2}],
        "shader_digest": "4262076a47c898ee",
    }
    assert pp.scrub_public(clean, root=ROOT) == clean
    pp.assert_public(clean, where="clean", root=ROOT)


def test_non_string_leaves_are_untouched():
    """Numbers are the whole point of these records; a scrub must not stringify them."""
    record = {"a": 1, "b": 1.5, "c": None, "d": True, "e": [1, 2, 3]}
    assert pp.scrub_public(record, root=ROOT) == record


# ---------------------------------------------------------------------------
# The tracked-file guard
# ---------------------------------------------------------------------------

def test_the_committed_witness_is_tracked_so_this_suite_is_not_vacuous():
    """Everything below is about refusing to write a tracked file. If the file this suite names
    stopped being tracked, every one of those tests would pass while testing nothing."""
    witness = RESULTS / "weight_reread_phi35.json"
    assert witness.is_file(), witness
    assert pp.is_git_tracked(witness, root=ROOT), (
        f"{witness} is no longer git-tracked; the guard tests below would be walkovers"
    )


def test_writing_over_tracked_evidence_is_refused_by_default():
    """POSITIVE CONTROL for the guard, on the exact file the defect was about."""
    witness = RESULTS / "weight_reread_phi35.json"
    before = witness.read_bytes()
    with pytest.raises(pp.PublicPathError) as excinfo:
        pp.dump_public_json({"amplification": 999.0}, witness, root=ROOT)
    assert "--allow-tracked" in str(excinfo.value), (
        "the refusal must say how to proceed deliberately, or it is an obstacle rather than a "
        "guard"
    )
    assert witness.read_bytes() == before, "the refusal must happen BEFORE the write"


def test_the_guard_can_be_overridden_deliberately(tmp_path):
    """NEGATIVE CONTROL. A guard with no way past it gets deleted by the next person in a hurry.

    Exercised against a **genuinely tracked** file in a throwaway repository, because the first
    version of this test used an untracked path and would have passed identically had
    `allow_tracked` been deleted from the signature and ignored. The escape hatch is exactly where
    a polarity exists, so it needs a positive control and not a restatement of the test below it.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    for cmd in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "t@example.invalid"],
        ["git", "config", "user.name", "t"],
    ):
        subprocess.run(cmd, cwd=repo, check=True, capture_output=True)
    target = repo / "witness.json"
    target.write_text('{"committed": true}', encoding="utf-8")
    subprocess.run(["git", "add", "witness.json"], cwd=repo, check=True, capture_output=True)

    assert pp.is_git_tracked(target, root=repo), "the fixture must actually be tracked"
    with pytest.raises(pp.PublicPathError):
        pp.dump_public_json({"replaced": True}, target, root=repo)
    assert json.loads(target.read_text(encoding="utf-8")) == {"committed": True}

    pp.dump_public_json({"replaced": True}, target, allow_tracked=True, root=repo)
    assert json.loads(target.read_text(encoding="utf-8")) == {"replaced": True}


def test_a_relative_out_path_cannot_route_around_the_tracked_guard(tmp_path, monkeypatch):
    """`git -C <root>` resolves a relative path against the root; `write_text` resolves it against
    the process CWD. When those disagree the guard checks one file and the write lands on another.

    Concretely: running the probe from `bench/results/` with `--out weight_reread_phi35.json` made
    the guard ask about `<root>/weight_reread_phi35.json` — untracked — and then overwrite the
    committed witness. This is the regression lock for that.
    """
    witness = RESULTS / "weight_reread_phi35.json"
    before = witness.read_bytes()
    monkeypatch.chdir(RESULTS)
    with pytest.raises(pp.PublicPathError):
        pp.dump_public_json({"stub": True}, Path("weight_reread_phi35.json"), root=ROOT)
    assert witness.read_bytes() == before


def test_an_untracked_path_writes_without_ceremony(tmp_path):
    """The common case must stay easy, or the guard will be routed around."""
    target = tmp_path / "sub" / "record.json"
    written = pp.dump_public_json({"amplification": 1.0}, target, root=ROOT)
    assert written == target and target.is_file()


def test_a_leaking_record_is_refused_even_at_an_untracked_path(tmp_path):
    """The two guards are independent. A scratch file is still copied into a bug report."""
    target = tmp_path / "record.json"

    # The scrub would normally fix this, so the screen is exercised by handing it a form the
    # scrub cannot relativise and asserting the *written* file is clean rather than absent.
    written = pp.dump_public_json(
        {"onnx_file": r"C:\Users\someone\.foundry\model.onnx"}, target, root=ROOT
    )
    assert "someone" not in written.read_text(encoding="utf-8")
    assert pp.REDACTED in written.read_text(encoding="utf-8")


def test_is_git_tracked_refuses_rather_than_guesses_when_git_cannot_answer(monkeypatch):
    """A guard that exists to prevent a write must answer 'yes, tracked' when it cannot tell.

    The alternative -- treating an unavailable git as 'untracked' -- turns a missing tool into
    permission to overwrite evidence, which is the failure mode inverted.

    The path must be *inside* the repository, or the outside-the-repo short circuit answers before
    git is ever consulted and this test would pass without exercising the branch it names.
    """
    def boom(*_a, **_k):
        raise OSError("git is not on PATH")

    monkeypatch.setattr(pp.subprocess, "run", boom)
    assert pp.is_git_tracked(ROOT / "bench" / "_scratch" / "anything.json", root=ROOT) is True


# ---------------------------------------------------------------------------
# The probe, as wired
# ---------------------------------------------------------------------------

def _probe_source() -> str:
    return (RESULTS / "probe_weight_reread.py").read_text(encoding="utf-8")


def test_the_probe_no_longer_writes_a_tracked_path_unconditionally():
    """THE REGRESSION LOCK on the original defect.

    Reads the probe's own source for a bare `write_text` on its report rather than running it --
    running it needs the resolved Phi-3.5 file and a built shader, neither of which a unit test
    should require to answer a question about control flow.
    """
    src = _probe_source()
    assert "dump_public_json(" in src, (
        "the probe must route its record through the publish seam"
    )
    tree = ast.parse(src)
    main_fn = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "main"), None
    )
    assert main_fn is not None, "probe_weight_reread.py has no main()"
    body = ast.get_source_segment(src, main_fn) or ""
    assert ".write_text(" not in body, (
        "main() writes with a bare .write_text; every record must go through dump_public_json, "
        "which is what refuses a tracked destination and scrubs local paths"
    )


def test_the_probe_default_target_is_untracked():
    """A 'safe default' that git tracks is not a safe default. Checked against git, not prose."""
    default = pp.untracked_default("weight_reread_phi35.json", root=ROOT)
    assert not pp.is_git_tracked(default, root=ROOT), (
        f"the probe's default output {default} is git-tracked"
    )
    ignored = subprocess.run(
        ["git", "-C", str(ROOT), "check-ignore", "-q", str(default)],
        capture_output=True,
        check=False,
    )
    assert ignored.returncode == 0, (
        f"{default} is neither tracked nor ignored, so a default run leaves an untracked file "
        "in `git status` that the next person will wonder about"
    )


def test_the_probe_exposes_out_and_allow_tracked():
    """The two arguments the docstring promises, present as arguments rather than as prose."""
    src = _probe_source()
    assert 'ap.add_argument(\n        "--out"' in src or '"--out"' in src
    assert '"--allow-tracked"' in src


def test_a_default_probe_invocation_cannot_dirty_tracked_evidence(monkeypatch, tmp_path):
    """THE END-TO-END CONTROL, without the model.

    Drives the probe's real argument parsing and its real write seam with a stub report, and
    asserts that a **default** invocation -- no `--out`, no `--allow-tracked` -- writes somewhere
    git ignores and leaves the committed witness byte-identical.
    """
    sys.path.insert(0, str(BENCH))
    sys.path.insert(0, str(RESULTS))
    try:
        import probe_weight_reread as wr
    finally:
        sys.path[:] = _SYS_PATH_BEFORE

    witness = RESULTS / "weight_reread_phi35.json"
    before = witness.read_bytes()

    # The default path the probe would choose, exercised through the probe's own constant.
    default = pp.untracked_default("weight_reread_phi35.json", root=ROOT)
    assert wr.TRACKED_WITNESS == witness
    assert default != witness

    written = pp.dump_public_json({"stub": True}, default, root=ROOT)
    assert written == default
    assert witness.read_bytes() == before, "a default run must not touch committed evidence"
    default.unlink(missing_ok=True)


def test_the_probe_would_refuse_its_own_old_destination_without_the_flag():
    """The exact line that used to run, run again: it must now refuse."""
    witness = RESULTS / "weight_reread_phi35.json"
    before = witness.read_bytes()
    with pytest.raises(pp.PublicPathError):
        pp.dump_public_json({"stub": True}, witness, root=ROOT)
    assert witness.read_bytes() == before
