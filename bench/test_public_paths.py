"""Both polarities of the provenance sanitiser (`bench/public_paths.py`).

Issue #69 shipped eight evidence JSONs naming the operator's account, private
checkout directory, and model cache. The repair is a boundary, not a cleanup:
every committed artifact is written through `dump_public_json` /
`write_public_text`, which root the paths and then refuse to write anything
still carrying machine identity.

A screen nobody has watched fail is a decoration, so every test here comes in
two halves: the sanitiser turns a planted leak into a rooted path, **and** the
screen raises when handed the unsanitised original. `test_screen_fires_*` are
the ones that matter — if those ever pass silently, the sanitiser has been
disabled and every other assertion in this file becomes vacuous.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import gen_public_path_legacy as gen  # noqa: E402
import public_paths as pp  # noqa: E402


# ---------------------------------------------------------------------------
# public_path: the structural half
# ---------------------------------------------------------------------------

def test_repo_path_is_rooted_and_posix():
    got = pp.public_path(pp.REPO / "bench" / "results" / "_cuda69" / "smoke.json")
    assert got == "<repo>/bench/results/_cuda69/smoke.json", got
    assert "\\" not in got, "separators must not record which OS produced the file"


def test_repo_root_itself_is_the_bare_token():
    assert pp.public_path(pp.REPO) == "<repo>"


def test_model_cache_beats_home_because_the_role_is_the_useful_part():
    """`<model-cache>` and `<home>` both match; the more specific root must win.

    Sorting roots by length is what makes this true. If the order were
    insertion order, a cached model would be published as `<home>/.cache/...`,
    which hides the account name but also hides that the file came from the
    declared cache rather than from somewhere the operator improvised.
    """
    cache = pp._model_cache_root().resolve()
    got = pp.public_path(cache / "mobilenetv2_12" / "mobilenetv2-12.onnx")
    assert got == "<model-cache>/mobilenetv2_12/mobilenetv2-12.onnx", got


def test_unknown_root_keeps_only_the_basename():
    """A path under no declared root is reduced, not rewritten.

    Guessing a token for it would be a claim about provenance the module cannot
    support. Keeping the basename keeps the only part that is about the
    artifact rather than about the machine.
    """
    got = pp.public_path(Path("/definitely/not/a/declared/root/weights.onnx"))
    assert got == "<elsewhere>/weights.onnx", got


def test_foundry_cache_is_its_own_root():
    """Phi-3.5 is resolved out of the Windows AI Foundry cache, not our own.

    Publishing it as `<home>/.foundry/...` would hide the account name and also
    hide that the weights came from a cache provisioned by a different tool,
    under a different contract (issue #11: exactly one variant, or refuse).
    Which cache the bytes came from is provenance.
    """
    foundry = (Path.home() / ".foundry" / "cache" / "models").resolve()
    got = pp.public_path(foundry / "Microsoft" / "Phi-3.5" / "v2" / "model.onnx")
    assert got == "<foundry-cache>/Microsoft/Phi-3.5/v2/model.onnx", got


def test_a_path_inside_prose_gets_the_same_root_as_a_path_in_a_field():
    """One directory, one spelling — whichever channel it arrived through.

    `bench_models` writes the cache directory into `files[].path` (a field) and
    into `detail` (a sentence). If prose were only beheaded to `<home>/.cache/
    onnxruntime-ep-vulkan/bench-models/x` while the field became
    `<model-cache>/x`, a reader has no way to know those name the same place.
    """
    cache = pp._model_cache_root().resolve()
    prose = pp.scrub_text(f"cached under {cache / 'mobilenetv2_12'}")
    assert prose == "cached under <model-cache>/mobilenetv2_12", prose
    assert pp.scan(prose) == []


def test_a_path_from_another_checkout_is_rooted_at_that_checkout():
    """Artifacts are repaired from a different worktree than produced them.

    That is the situation this module exists for: the eight rejected JSONs were
    written in one checkout and sanitised from a sibling one. Falling through to
    `<home>` would publish `<home>/.../repos/<repo>/bench/x` — no account name,
    but a spelling of a repo-relative path that no reader could match against
    the tree.
    """
    got = pp.public_path(Path.home() / "src" / "checkouts" /
                         "onnxruntime-ep-vulkan-1234" / "bench" / "results" / "x.npy")
    assert got == "<foreign-repo>/bench/results/x.npy", got
    assert pp.scan(got) == []


def test_the_current_checkout_and_a_foreign_one_do_not_share_a_token():
    """`<repo>` names one tree. Two trees spelled `<repo>` is a reader following a path
    into the wrong checkout and finding a different file, or no file.

    The ambiguity was not hypothetical and it was not symmetric: a record produced in a
    worktree could contain both its own paths (rooted `<repo>` as the current checkout)
    and paths from a sibling worktree (also rooted `<repo>`) — while the *canonical*
    checkout, whose directory name carries no worktree suffix, serialised as `<home>/...`.
    So the one token named two trees and did not name the obvious one.
    """
    mine = pp.public_path(pp.REPO / "bench" / "results" / "x.npy")
    theirs = pp.public_path(Path.home() / "src" / "checkouts" /
                            "onnxruntime-ep-vulkan-9999" / "bench" / "results" / "x.npy")
    assert mine == "<repo>/bench/results/x.npy", mine
    assert theirs == "<foreign-repo>/bench/results/x.npy", theirs
    assert mine != theirs
    assert pp.scan(mine) == [] and pp.scan(theirs) == []


def test_a_checkout_name_in_prose_gets_the_same_two_tokens():
    """Prose and fields must agree about which tree is which, or the tokens buy nothing."""
    foreign = pp.scrub_text("copied out of onnxruntime-ep-vulkan-9999/bench/results")
    assert foreign == "copied out of <foreign-repo>/bench/results", foreign
    assert pp.scan(foreign) == []
    if pp._WORKTREE.fullmatch(pp.REPO.name):
        mine = pp.scrub_text(f"copied out of {pp.REPO.name}/bench/results")
        assert mine == "copied out of <repo>/bench/results", mine


def test_the_model_cache_is_not_mistaken_for_a_checkout():
    """`~/.cache/onnxruntime-ep-vulkan/bench-models` is a cache, not a source tree.

    The foreign-checkout rule keys on the *suffix*, which is what distinguishes
    `onnxruntime-ep-vulkan-1234` (a worktree) from `onnxruntime-ep-vulkan` (the
    project, and also the name of the cache directory). Rooting the cache at
    `<repo>` would say downloaded weights live in the source tree.
    """
    foreign = Path.home() / ".cache" / "onnxruntime-ep-vulkan" / "bench-models" / "m.onnx"
    assert pp._foreign_checkout_root(foreign.resolve()) is None
    got = pp.public_path(foreign)
    assert got.startswith("<model-cache>"), got


def test_sibling_prefix_is_not_swallowed_by_string_matching():
    """`C:\\Users\\ann-backup` is not under `C:\\Users\\ann`.

    `str.startswith` says it is. A false positive here is worse than no
    sanitisation, because the output *looks* rooted while attributing the file
    to a root it never came from.
    """
    roots = [("<home>", Path("/x/ann").resolve())]
    got = pp.public_path(Path("/x/ann-backup/model.onnx").resolve(), roots=roots)
    assert got.startswith(pp.ELSEWHERE), got
    assert "ann-backup" not in got or got == "<elsewhere>/model.onnx"


def test_public_path_is_idempotent():
    """Applying the rewrite twice must not produce `<repo>/<repo>/...`.

    Sanitisation happens both where a field is built and again at the write
    boundary. That redundancy is deliberate; it is only safe if the second
    application is a no-op.
    """
    once = pp.public_path(pp.REPO / "bench" / "x.json")
    assert pp.public_path(once) == once


def test_empty_and_none_survive():
    assert pp.public_path("") == ""
    assert pp.public_path(None) == ""


def test_relative_remainder_is_scrubbed_too():
    """Naming the root is not enough when the account name is *below* it.

    pytest writes `pytest-of-<user>/` under the temp directory. Rooting the
    prefix as `<tmp>` and stopping there would publish the account name one
    component later, which is the same leak with extra steps.
    """
    tmp = Path(pp.tempfile.gettempdir()).resolve()
    user = Path.home().name
    got = pp.public_path(tmp / f"pytest-of-{user}" / "run0" / "prof.json")
    assert not pp.scan(got), f"{got} still leaks: {pp.scan(got)}"
    assert got.startswith("<tmp>/")


# ---------------------------------------------------------------------------
# scan: the detector, and what it must NOT flag
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("planted", [
    r"C:\Users\someone\.cache\model.onnx",
    r"C:/Users/someone/.cache/model.onnx",
    "/home/someone/.cache/model.onnx",
    "/Users/someone/.cache/model.onnx",
    "/opt/proj/.venv/bin/python",
    r"C:\proj\.venv-cu12\Scripts\python.exe",
    "/src/onnxruntime-ep-vulkan-1234/bench/results",
])
def test_scan_finds_every_leak_shape(planted):
    assert pp.scan(planted), f"{planted!r} passed the screen"


@pytest.mark.parametrize("clean", [
    "<repo>/bench/results/_cuda69/smoke.json",
    "<model-cache>/mobilenetv2_12/mobilenetv2-12.onnx",
    "<tmp>/run0/prof.json",
    "<elsewhere>/weights.onnx",
    "onnxruntime-ep-vulkan",          # the project name is not a checkout name
    "bench/results/_cuda69/suite.md",
    "/usr/lib/python3.12/json/__init__.py",
])
def test_scan_leaves_public_values_alone(clean):
    """False positives are not free: a screen that cries wolf gets turned off."""
    assert pp.scan(clean) == [], f"{clean!r} flagged: {pp.scan(clean)}"


def test_project_name_alone_is_not_a_worktree_name():
    """`onnxruntime-ep-vulkan` is the project; `onnxruntime-ep-vulkan-1234` is a checkout.

    The suffix is what makes it private. Flagging the bare project name would
    make the screen unusable in a repository that is, unavoidably, named that.
    """
    assert pp.scan("onnxruntime-ep-vulkan") == []
    assert pp.scan("onnxruntime-ep-vulkan-1234")
    assert pp.scan("onnxruntime-ep-vulkan-1234-agent")


# ---------------------------------------------------------------------------
# scrub_text: free text nobody parses
# ---------------------------------------------------------------------------

def test_scrub_text_removes_home_and_checkout_and_account():
    dirty = (rf"loading C:\Users\{Path.home().name}\.cache\m.onnx from "
             r"onnxruntime-ep-vulkan-1234 using /opt/x/.venv-cu12/bin/python")
    assert pp.scan(pp.scrub_text(dirty)) == [], pp.scan(pp.scrub_text(dirty))


def test_scrub_text_is_a_no_op_on_clean_text():
    clean = "median 3.41 ms over 30 iters on <repo>/bench/results/_cuda69"
    assert pp.scrub_text(clean) == clean


def test_scrub_text_passes_non_strings_through():
    assert pp.scrub_text("") == ""


# ---------------------------------------------------------------------------
# THE WRITE BOUNDARY MUST BE SATISFIABLE — one specimen per LEAK_PATTERNS entry
#
# `dump_public_json` scrubs and then refuses to write if `scan` still finds something.
# A shape the detector knows and the scrubber does not makes that contract impossible to
# satisfy: the caller sanitises, the writer raises anyway, and there is no input that
# works. That was true of `windows_drive_abs` — `scrub_text` hand-indexed `LEAK_PATTERNS`
# as `[:3]` and `[4]` and stepped straight over index 3 — so the first Windows DLL-load
# error quoted into a committed record ("...could not be loaded from C:\Program Files\...")
# was unwritable. The remedy under deadline is always to delete the field, which is how a
# provenance screen ends up removing provenance.
#
# One case per pattern, at line start and embedded, plus the roundtrip stated as the
# postcondition it is: scan(scrub_text(x)) == [].
# ---------------------------------------------------------------------------

_LEAK_SPECIMENS = {
    "windows_home": [r"C:\Users\someone\.cache\model.onnx",
                     r"loaded from C:\Users\someone\.cache\model.onnx (26 MB)"],
    "posix_home": ["/home/someone/.cache/model.onnx",
                   "loaded from /home/someone/.cache/model.onnx (26 MB)"],
    "macos_home": ["/Users/someone/.cache/model.onnx",
                   "loaded from /Users/someone/.cache/model.onnx (26 MB)"],
    "windows_drive_abs": [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\cudart64_12.dll",
        r"DLL load failed: C:\Program Files\NVIDIA\bin\x.dll is not a valid image",
        r"C:\ProgramData\NVIDIA Corporation\cache\x.bin",
        r"wrote its cache into C:\ProgramData\NVIDIA Corporation\cache",
        r"D:\Program Files (x86)\Vulkan\vulkan-1.dll",
    ],
    "virtualenv": ["/opt/proj/.venv/bin/python",
                   r"ran .venv-cu12\Scripts\python.exe -m bench"],
}


@pytest.mark.parametrize("kind", [k for k, _ in pp.LEAK_PATTERNS])
def test_every_leak_pattern_has_a_specimen_in_this_file(kind):
    """A pattern with no specimen below is a pattern this file never actually screened."""
    assert _LEAK_SPECIMENS.get(kind), (
        f"{kind} is in LEAK_PATTERNS with no specimen here; add one rather than trusting "
        f"that the roundtrip below covers it")


@pytest.mark.parametrize("kind,text", [
    (kind, text) for kind, texts in _LEAK_SPECIMENS.items() for text in texts])
def test_every_leak_pattern_is_detected_then_scrubbed_clean(kind, text):
    """Both halves, on one string: the detector fires, and the scrubber satisfies it."""
    kinds = {k for k, _ in pp.scan(text)}
    assert kind in kinds, f"{text!r} was not detected as {kind}: {pp.scan(text)}"
    cleaned = pp.scrub_text(text)
    assert pp.scan(cleaned) == [], f"scrub_text left {kind} in {cleaned!r}"


@pytest.mark.parametrize("kind", list(_LEAK_SPECIMENS))
def test_every_leak_pattern_has_a_scrub_rule(kind):
    """The structural half of the same fact, so a new pattern cannot land without one."""
    assert kind in pp.LEAK_SCRUB


def test_a_pattern_with_no_scrub_rule_is_refused_at_import():
    """Falsifier for the import-time guard: the table cannot go back to being partial."""
    import importlib

    src = (Path(pp.__file__)).read_text(encoding="utf-8")
    mutant = src.replace('    "virtualenv": "/<venv>/",\n', "", 1)
    assert mutant != src, "the scrub table no longer has the entry this mutant removes"
    ns: dict = {"__name__": "public_paths_mutant", "__file__": pp.__file__}
    with pytest.raises(RuntimeError) as exc:
        exec(compile(mutant, pp.__file__, "exec"), ns)  # noqa: S102 - deliberate mutant
    assert "unsatisfiable" in str(exc.value) or "LEAK_SCRUB" in str(exc.value)
    del importlib


def test_a_system_root_keeps_its_role_and_loses_the_drive():
    """No broad path deletion: the DLL still says where it came from, minus the machine."""
    got = pp.scrub_text(r"could not load C:\Program Files\NVIDIA\bin\cudart64_12.dll")
    assert got == r"could not load <program-files>\NVIDIA\bin\cudart64_12.dll", got
    data = pp.scrub_text(r"cache at C:\ProgramData\NVIDIA Corporation\cache")
    assert data == r"cache at <program-data>\NVIDIA Corporation\cache", data


def test_a_windows_dll_load_error_can_be_written_to_a_committed_record(tmp_path):
    """The end-to-end shape of the defect: scrub, then write, without a PathLeak."""
    detail = (r"[WinError 126] The specified module could not be found. Error loading "
              r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\cudart64_12.dll "
              r"while running from C:\Users\someone\dev\onnxruntime-ep-vulkan-1234")
    out = tmp_path / "rec.json"
    pp.dump_public_json({"verdict": "INSTRUMENT_ERROR",
                         "instrument_errors": [pp.scrub_text(detail)]}, out)
    written = out.read_text(encoding="utf-8")
    assert "<program-files>" in written
    assert pp.scan(written) == []


# ---------------------------------------------------------------------------
# scan_payload / assert_public: structure-aware
# ---------------------------------------------------------------------------

def test_scan_payload_reports_the_key_path_that_holds_the_leak():
    payload = {"results": [{"profile_path": r"C:\Users\someone\prof.json"}]}
    problems = pp.scan_payload(payload)
    assert len(problems) == 1
    assert "results[0].profile_path" in problems[0], problems


def test_scan_payload_checks_dict_keys_not_only_values():
    """A leak used as a key is still in the file.

    `{"C:\\Users\\someone": 1}` serialises the account name exactly as a value
    would. A walker that only visits values would publish it.
    """
    problems = pp.scan_payload({r"C:\Users\someone\out": 1})
    assert problems and "key" in problems[0], problems


def test_assert_public_is_silent_on_a_clean_payload():
    pp.assert_public({"path": "<repo>/bench/x.json", "n": 3, "ok": True})


def test_screen_fires_on_a_dirty_payload():
    with pytest.raises(pp.PathLeak) as exc:
        pp.assert_public({"cache_root": r"C:\Users\someone\.cache"})
    assert "cache_root" in str(exc.value)


# ---------------------------------------------------------------------------
# The write boundary — the guarantee the eight rejected JSONs needed
# ---------------------------------------------------------------------------

def test_dump_public_json_roots_paths_it_is_handed(tmp_path):
    out = tmp_path / "rec.json"
    real = str(pp.REPO / "bench" / "results" / "_cuda69" / "prof.json")
    pp.dump_public_json({"profile_path": real, "median_ms": 1.5}, out)
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["profile_path"] == "<repo>/bench/results/_cuda69/prof.json"
    assert doc["median_ms"] == 1.5, "sanitising must not disturb the measurements"


def test_screen_fires_when_sanitising_is_switched_off(tmp_path):
    """The polarity that proves the screen is a screen.

    With `sanitise=False` the payload is published as-is unless the check
    stops it. If this test ever passes without raising, `dump_public_json` has
    become a plain `write_text` and every artifact written through it is
    unprotected.
    """
    out = tmp_path / "rec.json"
    with pytest.raises(pp.PathLeak):
        pp.dump_public_json({"profile_path": r"C:\Users\someone\prof.json"},
                            out, sanitise=False)
    assert not out.exists(), "a refused write must not leave a partial file"


def test_dump_public_json_catches_a_path_object_that_dodged_the_object_walk(tmp_path):
    """`default=str` stringifies *after* an object-level walk would have run.

    A raw `Path` is not a `str`, so an object-level screen skips it and
    `json.dumps` then writes it out in full. Screening the serialised text is
    what closes that gap; this test is the reason the round-trip exists.
    """
    out = tmp_path / "rec.json"
    with pytest.raises(pp.PathLeak):
        pp.dump_public_json({"profile_path": Path(r"C:/Users/someone/prof.json")},
                            out, sanitise=False)


def test_write_public_text_scrubs_then_screens(tmp_path):
    out = tmp_path / "report.md"
    pp.write_public_text(f"run from {pp.REPO}\n", out)
    assert pp.scan(out.read_text(encoding="utf-8")) == []


def test_screen_fires_on_dirty_markdown(tmp_path):
    out = tmp_path / "report.md"
    with pytest.raises(pp.PathLeak):
        pp.write_public_text(r"run from C:\Users\someone\repo", out,
                             sanitise=False)
    assert not out.exists()


# ---------------------------------------------------------------------------
# sanitise_file: repairing artifacts produced before the writers were fixed
# ---------------------------------------------------------------------------

def test_sanitise_file_repairs_json_in_place_and_keeps_the_numbers(tmp_path):
    src = tmp_path / "baseline.json"
    src.write_text(json.dumps({
        "cache_root": r"C:\Users\someone\.cache\onnxruntime-ep-vulkan\bench-models",
        "results": [{"profile_path": r"C:\Users\someone\repo\prof.json",
                     "median_ms": 2.25,
                     "outputs_manifest": [{"file": r"C:\Users\someone\repo\o0.npy"}]}],
    }, indent=2), encoding="utf-8")

    before, changed = pp.sanitise_file(src)
    assert before > 0 and changed
    doc = json.loads(src.read_text(encoding="utf-8"))
    assert pp.scan_payload(doc) == [], pp.scan_payload(doc)
    assert doc["results"][0]["median_ms"] == 2.25


def test_sanitise_file_is_idempotent(tmp_path):
    src = tmp_path / "b.json"
    src.write_text(json.dumps({"path": r"C:\Users\someone\m.onnx"}), encoding="utf-8")
    pp.sanitise_file(src)
    first = src.read_text(encoding="utf-8")
    before, changed = pp.sanitise_file(src)
    assert before == 0 and not changed
    assert src.read_text(encoding="utf-8") == first


def test_sanitise_file_handles_text_artifacts(tmp_path):
    src = tmp_path / "run.log"
    src.write_text(r"python C:\Users\someone\.venv-cu12\Scripts\python.exe -m bench",
                   encoding="utf-8")
    before, changed = pp.sanitise_file(src)
    assert before > 0 and changed
    assert pp.scan(src.read_text(encoding="utf-8")) == []


# ---------------------------------------------------------------------------
# The producers actually use the boundary
# ---------------------------------------------------------------------------

def test_bench_models_serialises_a_rooted_path_but_keeps_the_real_one_in_memory():
    """The record must stay usable: `run_arm` opens `model.path` with ORT.

    Rooting the field in place would publish a clean file and break the
    harness. Rooting at `to_dict` is what lets both be true, so this test
    asserts both halves.
    """
    import bench_models

    rec = bench_models.ResolvedModel(
        key="k", status=bench_models.MODEL_OK,
        path=str(pp.REPO / "bench" / "m.onnx"),
        files=[{"name": "m.onnx", "path": str(pp.REPO / "bench" / "m.onnx"),
                "bytes": 1, "sha256": "x"}],
        detail=f"cached under {pp.REPO}",
    )
    assert Path(rec.path).is_absolute(), "the in-memory path must remain openable"
    d = rec.to_dict()
    assert pp.scan_payload(d) == [], pp.scan_payload(d)
    assert d["path"] == "<repo>/bench/m.onnx"
    assert d["files"][0]["path"] == "<repo>/bench/m.onnx"


def test_a_scratch_directory_given_as_an_absolute_path_is_rooted(tmp_path):
    """Every committed record was produced with a *relative* `--scratch`.

    So the scratch paths in them (`outputs_dir`, `file`, `profile_path`) read
    `_bench_scratch\\rep0\\...` -- clean, but clean by accident of invocation
    rather than by anything this suite checks. The one shape that has never
    been exercised is the shape that would reintroduce the defect this issue is
    about: `--scratch C:\\Users\\<account>\\scratch`, which is a perfectly
    ordinary thing to type and would put an account name into every path field of
    an evidence record. No count or size is quoted for that record: neither is
    witnessed by a committed artifact, and the defect does not depend on how
    many fields it lands in.

    This asserts the boundary handles it, so the suite stops depending on how
    the operator happened to invoke the harness.
    """
    scratch = Path.home() / "scratch-abs-probe"
    payload = {
        "outputs_dir": str(scratch / "rep0" / "outputs_vulkan_prefill_1"),
        "profile_path": str(scratch / "rep0" / "profile.json"),
        "outputs": [{"file": str(scratch / "rep0" / "outputs" / "o0.npy")}],
    }
    assert Path(payload["outputs_dir"]).is_absolute()

    out = tmp_path / "rec.json"
    pp.dump_public_json(payload, out)
    written = out.read_text(encoding="utf-8")

    assert pp.scan(written) == [], pp.scan(written)
    # Not `str(Path.home()) not in written`: json.dumps escapes separators, so a
    # leaked `C:\Users\ann` is serialised `C:\\Users\\ann` and the naive check
    # passes on a dirty file. Compare against the escaped form too.
    assert str(Path.home()) not in written
    assert json.dumps(str(Path.home()))[1:-1] not in written
    assert "<home>/scratch-abs-probe/rep0/outputs_vulkan_prefill_1" in written


def test_a_relative_scratch_path_survives_the_boundary_unchanged(tmp_path):
    """The other half: rooting must not mangle the form already committed.

    If the boundary rewrote `_bench_scratch/rep0/...` into `<repo>/...` or
    `<elsewhere>/...`, every committed evidence record would churn on the next
    regeneration for no gain. Relative provenance is explicitly one of the two
    acceptable forms, so it must pass through untouched.
    """
    payload = {"outputs_dir": "_bench_scratch\\rep0\\outputs_vulkan_prefill_1"}
    out = tmp_path / "rec.json"
    pp.dump_public_json(payload, out)
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["outputs_dir"] == "_bench_scratch\\rep0\\outputs_vulkan_prefill_1"


def test_a_dirty_relative_path_is_rooted_or_refused_but_never_published(tmp_path):
    """The other side of preserving relative provenance.

    Leaving relative strings alone is right for `_bench_scratch/rep0/...`, but a
    relative path can still name the account -- `..\\..\\Users\\<account>\\...`
    walks out through the home directory, and a bare `..\\<account>\\x` names it
    outright. Neither is exotic; both are what a `--scratch` argument typed from
    a sibling directory looks like.

    The invariant is *sanitised or refused, never published* -- not "raises".
    Rooting a traversal that escapes the repo is a good outcome and is what
    happens; refusing is the fallback for when rooting cannot clean it. Asserting
    the exception specifically would fail on correct behaviour.
    """
    probes = [
        str(Path("..") / ".." / "Users" / Path.home().name / "scratch" / "out.npy"),
        str(Path("..") / Path.home().name / "x.npy"),
        f"run by {Path.home().name}",
    ]
    for probe in probes:
        assert pp.scan(probe), f"probe is not dirty, so it tests nothing: {probe!r}"
        out = tmp_path / "d.json"
        try:
            pp.dump_public_json({"outputs_dir": probe}, out)
        except pp.PathLeak:
            continue  # refused: also acceptable
        published = json.loads(out.read_text(encoding="utf-8"))["outputs_dir"]
        assert pp.scan(published) == [], (probe, published, pp.scan(published))


def test_every_cuda_writer_goes_through_the_boundary():
    """No committed-artifact writer may call `write_text` with `json.dumps`.

    This is the screen that stops the next writer from being added the old way.
    It reads the source rather than the behaviour on purpose: the failure it
    guards against is a *new* code path, which no behavioural test can reach
    until someone remembers to write one.
    """
    offenders = []
    for name in ("cuda_competition.py", "cuda_probe.py", "cuda_profile.py",
                 "bench_models.py"):
        src = (Path(__file__).resolve().parent / name).read_text(encoding="utf-8")
        for i, line in enumerate(src.splitlines(), 1):
            if "write_text(json.dumps" in line.replace(" ", ""):
                offenders.append(f"{name}:{i}: {line.strip()}")
    assert not offenders, (
        "these lines serialise straight to disk, bypassing public_paths and the "
        "screen that rejected issue #69's artifacts:\n  " + "\n  ".join(offenders))


# ---------------------------------------------------------------------------
# THE SURVEY MUST NOT BE A FACT ABOUT WHOEVER RAN IT
#
# `scan` builds an account screen from `Path.home().name` / `$USERNAME`. That is right at a
# write boundary — the payload being committed was produced by this process — and wrong for
# a repository-wide survey whose answer is checked in. Under the operator's own account the
# ratchet declared nearly every one of those files as `account_name`; under a CI account
# called `runner` those files are clean and the ratchet fails as "a declared file has
# stopped leaking"; under a short name
# like `dev` or `ann` the screen matched *inside ordinary words* and reported leaks in files
# that contain "devices" or "annotation" — and `scrub_text` rewrote those words.
#
# So the survey uses `scan_structural`, and these are the falsifiers for that.
# ---------------------------------------------------------------------------

_SIMULATED_ACCOUNTS = ["justinchu", "runner", "ann", "dev"]

_SURVEY_CORPUS = [
    ("a/clean.json", '{"path": "<repo>/bench/results/x.json", "devices": 2}'),
    ("a/words.json", '{"note": "annotation of devices under development", "dev_ops": 1}'),
    ("a/home.json", r'{"path": "C:\\Users\\justinchu\\dev\\ep-vulkan\\bench\\x.json"}'),
    ("a/escaped.json", r'{"detail": "C:\\\\\\\\Users\\\\\\\\justinchu\\\\\\\\dev"}'),
    ("a/worktree.json", '{"path": "/src/onnxruntime-ep-vulkan-1234/bench/results"}'),
    ("a/venv.json", '{"cmd": "/opt/p/.venv-cu12/bin/python -m bench"}'),
    ("a/system.json", r'{"detail": "load failed: C:\\Program Files\\NVIDIA\\x.dll"}'),
]


def _simulate_account(monkeypatch, name: str) -> None:
    """Run the rest of the test as though this process belonged to *name*."""
    home = Path(f"/simulated/{name}") if os.name != "nt" else Path(rf"C:\Users\{name}")
    monkeypatch.setattr(pp.Path, "home", classmethod(lambda cls: home))
    for var in ("USERNAME", "USER", "LOGNAME"):
        monkeypatch.setenv(var, name)
    monkeypatch.setattr(pp.getpass, "getuser", lambda: name)
    assert name in pp._account_names(), "the simulation did not take"


@pytest.mark.parametrize("account", _SIMULATED_ACCOUNTS)
def test_the_repo_wide_survey_is_the_same_on_every_account(account, monkeypatch):
    """One tree, one answer. The baseline is checked in, so it may not depend on the runner."""
    expected = gen.survey_texts(_SURVEY_CORPUS)
    _simulate_account(monkeypatch, account)
    assert gen.survey_texts(_SURVEY_CORPUS) == expected, (
        f"the survey changed under account {account!r}; a checked-in ratchet that only "
        f"reproduces on one machine is a record of that machine")


def test_the_survey_still_counts_a_username_that_is_in_a_path(monkeypatch):
    """Dropping the account screen may not drop the leaks it was covering.

    The structural half does this work: a home directory is a home directory whoever lives
    in it, including under an account that has never heard of the name inside it.
    """
    _simulate_account(monkeypatch, "runner")
    got = gen.survey_texts(_SURVEY_CORPUS)
    assert "a/home.json" in got and "windows_home" in got["a/home.json"]["kinds"]
    assert "a/escaped.json" in got, (
        "a doubly JSON-escaped Windows home was invisible to the structural patterns, so "
        "it was only ever caught by the account screen — i.e. only on one machine")
    assert "a/clean.json" not in got and "a/words.json" not in got


@pytest.mark.parametrize("account", ["dev", "ann"])
def test_a_short_account_name_does_not_eat_ordinary_words(account, monkeypatch):
    """`dev` is a legal account name. `devices` is not a leak, and `<user>ices` is damage."""
    _simulate_account(monkeypatch, account)
    text = "annotation of devices under development; dev_ops ran 3 iters"
    assert pp.scrub_text(text) == text, pp.scrub_text(text)
    assert [k for k, _ in pp.scan(text) if k == "account_name"] == []


@pytest.mark.parametrize("account", _SIMULATED_ACCOUNTS)
def test_the_account_screen_still_fires_where_the_account_really_is(account, monkeypatch):
    """The boundary is a boundary, not a retreat: the whole word is still caught."""
    _simulate_account(monkeypatch, account)
    for text in (rf"C:\Users\{account}\dev\x.json",
                 f"ran as {account} on this desk",
                 f"(user={account})"):
        assert any(k == "account_name" for k, _ in pp.scan(text)), text
        cleaned = pp.scrub_text(text)
        # A path loses the name by being rooted; prose loses it to REDACTED_USER. Either
        # way the published text may not contain it.
        assert not any(k == "account_name" for k, _ in pp.scan(cleaned)), (text, cleaned)
        assert pp.scan(cleaned) == [], (text, cleaned)


def test_the_ratchet_uses_the_machine_independent_scan():
    """Stated structurally too, so the survey cannot quietly go back to `scan`.

    A behavioural test can only observe the difference on an account whose name appears in
    the tree; this one holds on every machine.
    """
    import inspect

    src = inspect.getsource(gen.survey_texts)
    assert "scan_structural" in src, src
    assert "account_name" not in json.dumps(_legacy_doc()["files"]), (
        "the committed ratchet still declares account_name leaks, which are a fact about "
        "the machine that generated it")


# ---------------------------------------------------------------------------
# The repository-wide ratchet
# ---------------------------------------------------------------------------

def _legacy_doc() -> dict:
    return gen.load_baseline()


def test_no_committed_evidence_file_leaks_unless_it_is_declared():
    """Every tracked evidence file is scanned; new leaks are a failure.

    Not a repository-wide *gate*: the files that named a machine before
    `bench/public_paths.py` existed are the entry set of the `files` map in
    `bench/public_path_legacy.json` — a declaration, not a measurement, and its
    size is read from the artifact rather than restated here — and a screen that is
    red on the day it lands is a screen that gets skipped. So the declared set in
    `bench/public_path_legacy.json` is the ceiling and this test is the ratchet
    — an undeclared leak, or a declared file leaking more than declared, fails.

    The failure text names the files and the shapes, because "N files failed"
    tells a reader nothing they can act on.
    """
    doc = _legacy_doc()
    declared = doc["files"]
    current = gen.survey(pp.REPO)

    undeclared = sorted(set(current) - set(declared))
    worse = sorted(f for f in set(current) & set(declared)
                   if current[f]["leaks"] > declared[f]["leaks"])

    report = []
    for f in undeclared:
        report.append(f"{f}: {current[f]['leaks']} leak(s) {current[f]['kinds']} "
                      f"— NOT DECLARED")
    for f in worse:
        report.append(f"{f}: {declared[f]['leaks']} declared, "
                      f"{current[f]['leaks']} found — WORSE")
    assert not report, (
        "committed evidence names a machine.\n  " + "\n  ".join(report) +
        "\n\nRoute the writer through bench/public_paths.dump_public_json or "
        "write_public_text. Adding the file to bench/public_path_legacy.json is "
        "not the remedy — that list is for artifacts that predate the screen.")


def test_no_declaration_outlives_the_leak_it_declares():
    """A declared file that has stopped leaking must be undeclared.

    This is the half that keeps the ratchet honest. Without it the list is
    write-only: entries accumulate, nobody removes them, and eventually it
    describes a tree that no longer exists while still granting permission to
    every path it names.
    """
    doc = _legacy_doc()
    declared = doc["files"]
    current = gen.survey(pp.REPO)

    stale = []
    for f, rec in sorted(declared.items()):
        target = pp.REPO / f
        if not target.is_file():
            stale.append(f"{f}: declared legacy but no longer in the tree")
        elif f not in current:
            stale.append(f"{f}: declared {rec['leaks']} leak(s), now clean — "
                         f"remove the declaration")
    assert not stale, (
        "bench/public_path_legacy.json no longer describes the tree:\n  "
        + "\n  ".join(stale) +
        "\n\nRegenerate with `python bench/gen_public_path_legacy.py`.")


# ---------------------------------------------------------------------------
# NO PROSE MAY QUOTE A COUNT THE ARTIFACT DOES NOT CARRY
#
# The ratchet's own generator opened with "red on N files that predate it" while the
# artifact it generates declared a different N. The array was right; the sentence
# beside it was a second copy of a number that nothing re-read. That is the same defect
# class the whole provenance screen exists for — a figure published as a finding that no
# longer describes anything — so it gets the same treatment: a mechanism, not a habit.
#
# `gen.artifact_totals` is the only sanctioned source, and both polarities are asserted
# here: the tree must be clean, AND a planted stale figure must be caught.
# ---------------------------------------------------------------------------

#: A count no artifact will ever carry, and the two halves of a synthetic true one. Held
#: as numbers rather than written into the specimen strings so that this file — which is
#: itself screened below — carries no literal count claim of its own.
_ABSURD = 424242
_TRUE_FILES = 7
_TRUE_LEAKS = 11


def test_the_totals_are_read_from_the_artifact_and_not_from_a_second_copy():
    """The derivation itself, against the document on disk."""
    doc = _legacy_doc()
    totals = gen.artifact_totals(doc)
    assert totals["file"] == len(doc["files"])
    assert totals["leak"] == sum(rec["leaks"] for rec in doc["files"].values())
    assert gen.artifact_totals() == totals, (
        "artifact_totals() disagrees with artifact_totals(doc); there is a second reader")


def test_no_prose_quotes_a_file_count_the_ratchet_does_not_carry():
    """POSITIVE POLARITY: every counted document agrees with the artifact today.

    The files screened are `gen.COUNTED_PROSE`. A count may be quoted — a report that
    names a number is more use than one that says "some" — but only one the artifact
    carries, and this is what makes that enforceable instead of aspirational.
    """
    totals = gen.artifact_totals()
    offenders = []
    for rel in gen.COUNTED_PROSE:
        target = pp.REPO / rel
        assert target.is_file(), f"{rel} is screened for stale counts but is not in the tree"
        offenders += gen.stale_count_claims(
            target.read_text(encoding="utf-8", errors="replace"), totals, rel)
    assert not offenders, (
        "prose quotes a file/leak count the ratchet does not declare:\n  "
        + "\n  ".join(offenders) +
        "\n\nRead the number from bench/public_path_legacy.json (gen.artifact_totals) or "
        "drop it from the sentence. Regenerating the artifact does not make a sentence "
        "about it true.")


@pytest.mark.parametrize("planted", [
    f"A screen turned on today would be red on {_ABSURD} files that predate it.",
    f"The tree carries {_ABSURD} leaks.",
    f"{_ABSURD} committed evidence files carry a structural leak.",
    f"There are {_ABSURD:,} declared legacy files.",
])
def test_a_planted_stale_count_is_caught(planted):
    """NEGATIVE POLARITY, and the falsifier for the test above.

    An empty offender list is what a clean tree looks like AND what a dead screen looks
    like. These are the shapes the defect actually took, with a number no artifact will
    ever carry.

    The specimens are BUILT rather than written out, because this file is itself one of
    the documents `test_no_prose_quotes_a_file_count_the_ratchet_does_not_carry` screens —
    a literal stale count in a test about stale counts would be caught by it, correctly.
    """
    offenders = gen.stale_count_claims(planted, {"file": 1, "leak": 2}, "planted.md")
    assert offenders, f"the stale-count screen did not see {planted!r}"
    assert str(_ABSURD) in offenders[0] or f"{_ABSURD:,}" in offenders[0], offenders


@pytest.mark.parametrize("clean", [
    f"A screen turned on today would be red on {_TRUE_FILES} files that predate it.",
    f"Those files carry {_TRUE_LEAKS} leaks between them.",
    "The resolved Phi-3.5 file is read once.",   # a version, not a count
    "Twelve nodes covering every distinct form present.",
])
def test_the_screen_leaves_a_true_or_unrelated_number_alone(clean):
    """The other half: a screen that fires on everything is no more use than one that never does."""
    totals = {"file": _TRUE_FILES, "leak": _TRUE_LEAKS}
    assert gen.stale_count_claims(clean, totals, "clean.md") == []


def test_the_directory_this_issue_was_about_may_never_be_declared_legacy():
    """`bench/results/_cuda69/` gets no grandfather clause.

    Its writers were fixed by this change. A leak there is a regression in a
    repaired path, not an inheritance, and letting it be declared would make
    the ratchet able to absorb exactly the defect it was built for.
    """
    doc = _legacy_doc()
    assert list(doc["never_legacy"]) == list(gen.NEVER_LEGACY)
    smuggled = [f for f in doc["files"]
                if any(f.startswith(p) for p in gen.NEVER_LEGACY)]
    assert not smuggled, (
        "these are declared legacy under a tree that may not have legacy:\n  "
        + "\n  ".join(smuggled))


def test_the_ratchet_would_notice_a_planted_leak(tmp_path):
    """The polarity for the ratchet itself.

    `test_no_committed_evidence_file_leaks_unless_it_is_declared` passes today
    because the tree is clean, which is indistinguishable from passing because
    the survey found nothing. Planting a file with a real leak and confirming
    the survey reports it is what tells those two apart.
    """
    planted = tmp_path / "evidence.json"
    planted.write_text(json.dumps(
        {"profile_path": r"C:\Users\someone\repo\prof.json"}), encoding="utf-8")
    hits = pp.scan(planted.read_text(encoding="utf-8"))
    assert hits, "the survey's scanner does not see a planted absolute home path"
    assert {k for k, _ in hits} & {"windows_home"}, hits


def test_the_survey_covers_the_extensions_a_reader_reads():
    """A screen that skips `.md` and `.log` is not a screen over the evidence.

    Issue #69's leaks were in `.json`, but `profile_prefill_1.log` leaked too,
    and the reports a reviewer actually opens are `.md`. Enumerating the
    suffixes here means adding a new evidence format is a visible decision.

    The `_cuda69` half is asserted as a *rule* rather than as a tracked file.
    Asserting the file is tracked makes the test a statement about which branch
    it is running on -- it fails on an instrumentation-only head for a reason
    that has nothing to do with the survey. The rule ("a `_cuda69` artifact is in
    scope and can never be grandfathered") is what the test is actually for, it
    holds on every branch, and it is the thing that would have to break for the
    directory to fall out of the survey.
    """
    assert gen.EVIDENCE_SUFFIXES == frozenset(
        {".json", ".jsonl", ".md", ".log", ".txt", ".csv"})
    tracked = gen.tracked_evidence(pp.REPO)
    assert any(f.endswith(".md") for f in tracked)
    assert any(f.endswith(".log") for f in tracked)

    assert "bench/results/_cuda69/" in gen.NEVER_LEGACY, (
        "the directory this issue was about must never be grandfathered")
    for suffix in (".json", ".md", ".log"):
        probe = f"bench/results/_cuda69/probe{suffix}"
        assert Path(probe).suffix in gen.EVIDENCE_SUFFIXES, (
            f"a {suffix} artifact under _cuda69 would fall outside the survey")

    # Where the artifacts are actually present, the original, stronger form still
    # applies: they must really be enumerated, not merely eligible.
    cuda69 = [f for f in tracked if f.startswith("bench/results/_cuda69/")]
    if any(Path(pp.REPO / "bench" / "results" / "_cuda69").glob("*")):
        assert cuda69, "_cuda69 holds artifacts but the survey enumerated none of them"


def test_the_harness_source_itself_names_no_local_path():
    """The survey screens what the harness *writes*, never what the harness *is*.

    That gap is real: a hardcoded `C:\\Users\\...` default, a developer's model
    directory left in a constant, or a cache path baked into an argument default
    would all sail through, because `EVIDENCE_SUFFIXES` has no `.py` in it and
    never will -- source is not evidence. The producers are exactly the files
    with the opportunity to hardcode a path, so they get their own screen.

    It caught one thing on the way in, and the shape is worth recording: a
    `User-Agent: "onnxruntime-ep-vulkan-bench/1"` matched `worktree_name`,
    because `<repo>-<suffix>` is precisely what a sibling worktree looks like.
    A constant is not a leak, so the honest options were to narrow the pattern or
    to stop colliding with it. Narrowing is how a scanner starts reporting clean
    while being wrong -- this repo has already been bitten by four patterns that
    were too permissive -- so the token changed instead. An over-broad leak
    pattern costs a rename; an under-broad one costs a disclosure.
    """
    import public_paths as boundary

    # The modules that define, test, or record the patterns necessarily contain
    # them. Each is here because quoting a leak is its job, not an oversight.
    # `gen_public_path_legacy.py` is deliberately *not* here: it re-exports the
    # patterns from the boundary rather than restating them, so it has nothing to
    # be exempt for -- and the staleness check below is what established that.
    DEFINES_THE_PATTERNS = {
        "public_paths.py": "defines LEAK_PATTERNS",
        "test_public_paths.py": "plants leaks to prove the scanner sees them",
    }

    here = Path(__file__).resolve().parent
    offenders = {}
    for src in sorted(here.glob("*.py")):
        if src.name in DEFINES_THE_PATTERNS:
            continue
        hits = boundary.scan(src.read_text(encoding="utf-8", errors="replace"))
        if hits:
            offenders[src.name] = sorted({kind for kind, _ in hits})

    assert not offenders, (
        "harness source names a local path; a constant that merely collides with a "
        f"leak pattern should be renamed, not exempted: {offenders}")

    # The exemptions are not a hole: each exempt file must still be one that
    # really does quote leaks, or it has been left on the list after being fixed.
    for name in DEFINES_THE_PATTERNS:
        target = here / name
        if target.is_file():
            assert boundary.scan(target.read_text(encoding="utf-8", errors="replace")), (
                f"{name} is exempt from the source screen but contains no leak "
                "pattern -- the exemption is stale and should be dropped")


def test_the_scanner_cli_accepts_a_directory():
    """A run writes a directory, so "did this run leak?" is asked about a directory.

    The first cut of `main()` called `read_text` on whatever it was handed and died with
    `PermissionError: bench/results/_cuda69` -- which reads like a filesystem problem, not
    like "you must glob this yourself". A checking tool that is awkward to point at the
    thing you just produced is a checking tool people skip.
    """
    d = Path(__file__).resolve().parent / "results" / "_cuda69"
    if not d.is_dir():
        pytest.skip("committed artifacts not present in this tree")
    files = pp._expand([str(d)])
    assert files, "a directory of evidence expanded to nothing"
    assert all(f.suffix.lower() in pp.EVIDENCE_SUFFIXES for f in files)
    assert pp.main([str(d)]) == 0, "the committed artifacts must scan clean"


def test_the_scanner_and_the_ratchet_agree_on_what_evidence_is():
    """Two enumerations of "evidence file" is one enumeration that will drift.

    If the CLI screened a suffix the ratchet did not enumerate, a leaking file could pass
    the ratchet by never being surveyed at all.
    """
    assert gen.EVIDENCE_SUFFIXES is pp.EVIDENCE_SUFFIXES


# ---------------------------------------------------------------------------
# B2: `contained_child` IS TOTAL, AND IT REFUSES RATHER THAN REPAIRS
#
# It is the only operation allowed to turn a handle written by another process into a
# file this one opens, so every shape it silently fixed was a way for a record to be
# validated against a file nobody named:
#
#   * `sub/C:evil.npy` — the colon check read the FIRST component only, and joining a
#     drive-relative name discards everything before it, so the escape simply had to be
#     spelled one component to the right.
#   * `out0.npy:stream` — an NTFS alternate data stream. Same syntax, same blind spot.
#   * `out0.npy.` / `out0.npy ` — Windows strips a trailing dot or space before opening,
#     so two handles that read differently name one file.
#   * `sub\evil.npy` — a backslash was rewritten to a separator. On POSIX it is a legal
#     character in a filename, so the "repair" renames the file being pointed at.
#   * an embedded NUL — `Path` raises `ValueError`, which is not `OSError`, so the
#     function was not total: a caller checking for `None` got a traceback instead.
#   * `NUL` — a Windows device, in every directory, that opens.
#
# Both polarities throughout: the specimens below must be refused, and a legitimate
# contained file must still resolve, because a resolver that refuses everything closes
# the harness rather than the hole.
# ---------------------------------------------------------------------------

#: Every shape a handle may not take, with the reason it is not merely unusual.
_REFUSED_HANDLES = [
    ("", "empty"),
    (".", "the directory itself"),
    ("..", "the parent"),
    ("sub/..", "traversal that lands back on the base"),
    ("../out0.npy", "traversal to a sibling"),
    ("sub/../../out0.npy", "traversal spelled through a subdirectory"),
    ("/etc/passwd", "rooted at the filesystem"),
    ("//server/share/out0.npy", "a UNC path"),
    ("//./pipe/out0.npy", "a device path"),
    ("//?/C:/out0.npy", "an extended-length path"),
    ("C:out0.npy", "drive-relative"),
    ("C:/out0.npy", "absolute with a drive"),
    ("sub/C:evil.npy", "a drive spec in a component that is not the first"),
    ("sub/sub2/D:evil.npy", "the same, further right"),
    ("out0.npy:hidden", "an NTFS alternate data stream"),
    ("sub/out0.npy:$DATA", "a stream on a contained name"),
    ("<repo>/out0.npy", "a rooted token, which names a place and not a file"),
    ("<elsewhere>/out0.npy", "a token this module publishes for the unattributable"),
    ("sub//out0.npy", "an empty component"),
    ("sub/./out0.npy", "a no-op component that is still not a name"),
    ("out0.npy.", "a trailing dot Windows would strip"),
    ("out0.npy ", "a trailing space Windows would strip"),
    ("sub./out0.npy", "a trailing dot on a directory component"),
    ("out0\x00.npy", "an embedded NUL"),
    ("sub/out0\x00.npy", "an embedded NUL further right"),
    ("out0\n.npy", "an embedded control character"),
    ("NUL", "a Windows device name"),
    ("nul.npy", "a Windows device name with a suffix"),
    ("sub/COM1", "a device name in a subdirectory"),
    ("aux.txt", "another device name"),
]


@pytest.mark.parametrize("handle,why", _REFUSED_HANDLES,
                         ids=[h.replace("\x00", "NUL").replace("\n", "LF") or "empty"
                              for h, _ in _REFUSED_HANDLES])
def test_a_malformed_handle_is_refused_and_never_repaired(tmp_path, handle, why):
    """REJECT POLARITY. Refused, and refused by returning — never by raising."""
    base = tmp_path / "slot"
    base.mkdir()
    (base / "out0.npy").write_bytes(b"0")
    (base / "sub").mkdir()

    got = pp.contained_child(base, handle)

    assert got is None, f"{handle!r} ({why}) resolved to {got!r} instead of being refused"


@pytest.mark.parametrize("sep", ["sub\\out0.npy", "..\\out0.npy", "sub\\..\\..\\out0.npy"])
def test_a_backslash_handle_is_refused_rather_than_rewritten(tmp_path, sep):
    """A handle is a POSIX name by contract; rewriting a separator renames the file."""
    base = tmp_path / "slot"
    (base / "sub").mkdir(parents=True)
    (base / "sub" / "out0.npy").write_bytes(b"0")

    assert pp.contained_child(base, sep) is None


@pytest.mark.parametrize("rel", [None, 123, 4.5, True, ["out0.npy"], {"f": "out0.npy"},
                                 b"out0.npy"])
def test_a_handle_that_is_not_a_string_is_refused(tmp_path, rel):
    """`str(rel)` turned an int, a list and a `Path` into plausible names. It no longer runs."""
    base = tmp_path / "slot"
    base.mkdir()

    assert pp.contained_child(base, rel) is None


def test_a_path_object_is_refused_because_the_contract_is_a_name(tmp_path):
    """A `Path` is the caller's own resolution, not the handle another process wrote."""
    base = tmp_path / "slot"
    base.mkdir()
    (base / "out0.npy").write_bytes(b"0")

    assert pp.contained_child(base, Path("out0.npy")) is None
    assert pp.contained_child(base, "out0.npy") == (base / "out0.npy").resolve()


@pytest.mark.parametrize("base", [None, "", 0, False, 3.5])
def test_an_absent_or_non_path_base_is_refused(base):
    """`Path("")` is the current directory, so an empty base silently rooted at the CWD."""
    assert pp.contained_child(base, "out0.npy") is None


def test_a_legitimate_contained_handle_still_resolves(tmp_path):
    """ACCEPT POLARITY. Every refusal above is worthless if this stops working."""
    base = tmp_path / "slot"
    (base / "sub" / "deeper").mkdir(parents=True)
    for rel in ("out0.npy", "sub/out0.npy", "sub/deeper/profile.json",
                "trace_prefill_1.json", "result_cpu_host_wl.json", "out0.npy.gz",
                "a-b_c.1.npy"):
        target = base / Path(*rel.split("/"))
        target.write_bytes(b"0")
        assert pp.contained_child(base, rel) == target.resolve(), rel


def test_containment_survives_ntfs_case_folding(tmp_path):
    """`_relative_to` folds case on Windows, and a contained file must stay contained."""
    base = tmp_path / "Slot"
    base.mkdir()
    (base / "Out0.npy").write_bytes(b"0")

    got = pp.contained_child(base, "Out0.npy")
    assert got == (base / "Out0.npy").resolve()
    if os.name == "nt":
        assert pp.contained_child(str(base).upper(), "out0.npy") is not None


def test_contained_child_never_raises_whatever_it_is_handed(tmp_path):
    """Totality, asserted as such: a resolver that raises is a resolver a caller cannot check.

    An embedded NUL raises `ValueError` from `Path`, not `OSError`, and the old
    `except OSError` did not cover it. A caller written as
    `if contained_child(...) is None: refuse` was therefore bypassed by a traceback
    rather than told no, which is the failure this whole module exists to make impossible.
    """
    base = tmp_path / "slot"
    base.mkdir()
    for rel in [h for h, _ in _REFUSED_HANDLES] + [None, 1, Path("x"), object()]:
        try:
            pp.contained_child(base, rel)
        except Exception as exc:  # pragma: no cover - the assertion is the point
            pytest.fail(f"contained_child raised {exc!r} for {rel!r}; it must refuse")


@pytest.mark.parametrize("escape", ["out0.npy", "sub/out0.npy"])
def test_a_link_that_leaves_the_directory_is_refused_after_resolution(tmp_path, escape):
    """The guarantee is `resolve()` + `_relative_to`, not the string checks above it."""
    base = tmp_path / "slot"
    (base / "sub").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "out0.npy").write_bytes(b"0")
    link = base / Path(*escape.split("/"))
    try:
        link.symlink_to(outside / "out0.npy")
    except (OSError, NotImplementedError):
        pytest.skip("this account cannot create symlinks on this platform")

    assert link.exists()
    assert pp.contained_child(base, escape) is None


def test_the_four_handle_resolvers_refuse_every_malformed_handle(tmp_path):
    """The refusal has to be visible where the handles are actually read.

    `contained_child` is the mechanism; these four are the contract's surface, and each is
    a TOTAL `(value, why)` instrument. Both polarities: every specimen refuses WITH a
    reason, and the handle their own writer produces still resolves.
    """
    import _polarity
    import cuda_competition as cc
    import cuda_profile as cp

    scratch = tmp_path / "traced"
    dump = scratch / "outputs_vulkan_wl"
    dump.mkdir(parents=True)
    (dump / cc.OUTPUT_HANDLE).write_bytes(b"0")
    (scratch / "profile_x.json").write_text("[]", encoding="utf-8")
    rec = {"arm": cc.ARM_VULKAN,
           "outputs_dir": pp.public_path(dump), "scratch_dir": pp.public_path(scratch),
           pp.RUNTIME_ONLY_KEY: {"outputs_dir": str(dump.resolve()),
                                 "scratch_dir": str(scratch.resolve())}}

    for handle, why in _REFUSED_HANDLES:
        _polarity.refuses(cc.resolve_scratch_file(rec, handle), because=why)
        _polarity.refuses(cp.resolve_trace({**rec, "trace_rel": handle}), because=why)
        _polarity.refuses(cc.resolve_arm_output(rec, {"file_rel": handle}), because=why)
        _polarity.refuses(cc.relative_handle(handle, scratch), because=why)

    # ACCEPT POLARITY for all four, so the loop above cannot be satisfied by a resolver
    # that refuses unconditionally.
    (scratch / "trace_prefill_1.json").write_text("[]", encoding="utf-8")
    assert cc.resolve_scratch_file(rec, "profile_x.json")[0] is not None
    assert cp.resolve_trace({**rec, "trace_rel": "trace_prefill_1.json"})[0] is not None
    assert cc.resolve_arm_output(rec, {"file_rel": cc.OUTPUT_HANDLE})[0] is not None
    assert cc.relative_handle(scratch / "profile_x.json", scratch)[0] == "profile_x.json"


def test_resolve_public_path_refuses_a_malformed_remainder(tmp_path, monkeypatch):
    """The inverse shares the mechanism, so it inherits every refusal above."""
    monkeypatch.setattr(pp, "REPO", tmp_path)
    (tmp_path / "bench").mkdir()

    assert pp.resolve_public_path("<repo>/bench") == (tmp_path / "bench").resolve()
    for bad in ("<repo>/sub/C:evil.npy", "<repo>/bench/out0.npy:stream",
                "<repo>/bench/out0.npy.", "<repo>/../escaped", "<repo>/NUL"):
        assert pp.resolve_public_path(bad) is None, bad


# ---------------------------------------------------------------------------
# B4: A COUNT IS BOUND TO THE COLLECTION ITS OWN NOUN NAMES
#
# The screen used to ask "is this integer one of the numbers the artifact publishes?".
# The artifact publishes two, so a sentence could quote the file count and call it a leak
# count and pass: the digits were witnessed by a collection, just not by the one the
# sentence was about. Below: right number under the wrong label, right label with the
# wrong number, and a borrowed number that is real somewhere else.
# ---------------------------------------------------------------------------

def test_a_count_may_not_borrow_the_other_collections_number():
    """The exact mutation: the file count, spelled as a leak count, and the reverse."""
    totals = {"file": _TRUE_FILES, "leak": _TRUE_LEAKS}

    assert gen.stale_count_claims(f"{_TRUE_FILES} leaks", totals, "m.md"), (
        "the file count witnessed a leak claim")
    assert gen.stale_count_claims(f"{_TRUE_LEAKS} files", totals, "m.md"), (
        "the leak count witnessed a file claim")
    assert gen.stale_count_claims(f"{_TRUE_LEAKS} declared files", totals, "m.md")
    assert gen.stale_count_claims(f"{_TRUE_FILES} committed evidence leaks", totals, "m.md")


def test_a_count_with_the_right_label_and_the_wrong_number_is_caught():
    """The other mutation direction, so the binding is not satisfied by the noun alone."""
    totals = {"file": _TRUE_FILES, "leak": _TRUE_LEAKS}

    assert gen.stale_count_claims(f"{_TRUE_FILES + 1} files", totals, "m.md")
    assert gen.stale_count_claims(f"{_TRUE_LEAKS + 1} leaks", totals, "m.md")
    assert gen.stale_count_claims(f"{_ABSURD} files", totals, "m.md")


def test_a_count_that_names_its_own_collection_is_left_alone():
    """BOTH POLARITIES. A screen that fires on every count is one nobody can quote through."""
    totals = {"file": _TRUE_FILES, "leak": _TRUE_LEAKS}

    for clean in (f"{_TRUE_FILES} files", f"{_TRUE_LEAKS} leaks",
                  f"{_TRUE_FILES} committed evidence files",
                  f"{_TRUE_FILES:,} declared legacy files",
                  f"{_TRUE_LEAKS} leaks between them"):
        assert gen.stale_count_claims(clean, totals, "clean.md") == [], clean


def test_two_collections_holding_the_same_number_is_not_a_licence():
    """When both totals coincide, a borrowed number is invisible to a set-membership check.

    This is the state in which the old screen could not have been red for ANY sentence,
    and it is the one where the binding has to be the thing doing the work.

    The figures are BUILT rather than written out, because this file is one of the
    documents `test_no_prose_quotes_a_file_count_the_ratchet_does_not_carry` screens: a
    literal count in a test about counts would be convicted by it, correctly.
    """
    same = _TRUE_FILES
    totals = {"file": same, "leak": same}

    assert gen.stale_count_claims(f"{same} files", totals, "m.md") == []
    assert gen.stale_count_claims(f"{same} leaks", totals, "m.md") == []
    assert gen.stale_count_claims(f"{same + 1} leaks", totals, "m.md")


# ---------------------------------------------------------------------------
# B5: PROSE MAY NOT NAME A CHANNEL THE READER DOES NOT READ
#
# `public_payload`'s docstring said `cuda_profile` reads `profile_path` back off the
# record. It does not, and did not by the time the sentence was written: it reads
# `profile_rel` through `cuda_competition.resolve_scratch_file`. An architecture note that
# names the wrong field is worse than none — it is the document a reader consults to find
# out which channel is load-bearing, and this one pointed at the field the fix removed.
#
# `profile_path` still occurs in `cuda_profile.py` as a local PARAMETER name, so a guard
# that only asked "does this word appear in that file?" would have passed the stale
# sentence. The claim is therefore checked against what the module reads OUT OF A RECORD.
# ---------------------------------------------------------------------------

_READS_CLAIM = __import__("re").compile(
    r"``([a-z_][a-z0-9_]*)``\s+reads\s+``([a-z_][a-z0-9_]*)``")

#: `rec.get("x")` / `rec["x"]` — the two ways this suite takes a field off a record.
#:
#: READ WITH `ast`, NOT WITH A REGEX, AND ONLY IN LOAD POSITION.  The first cut matched the
#: text, so `record["profile_path"] = str(profile)` — an assignment — and a docstring
#: quoting ``rec["profile_path"]`` both counted as reads. That over-approximation was
#: invisible while the only question asked was "does this module read the field the prose
#: names": a claim was acquitted by a mention. It stops being invisible the moment the
#: opposite claim is screened too — "nothing opens this field" is false for every field the
#: module merely writes, if a write counts as a read.
_RECORD_READ = __import__("re").compile(r"""(?:\.get\(|\[)["']([A-Za-z_][A-Za-z0-9_]*)["']""")


def _record_reads(module: str) -> "set[str]":
    import ast as _ast

    src = (Path(__file__).resolve().parent / module).read_text(encoding="utf-8")
    try:
        tree = _ast.parse(src)
    except SyntaxError:  # pragma: no cover - a bench module that will not parse
        return set()

    def _direct(node) -> "set[str]":
        out: "set[str]" = set()
        for n in _ast.walk(node):
            if (isinstance(n, _ast.Subscript)
                    and isinstance(n.ctx, _ast.Load)
                    and isinstance(n.slice, _ast.Constant)
                    and isinstance(n.slice.value, str)):
                out.add(n.slice.value)
            elif (isinstance(n, _ast.Call)
                    and isinstance(n.func, _ast.Attribute)
                    and n.func.attr == "get"
                    and n.args
                    and isinstance(n.args[0], _ast.Constant)
                    and isinstance(n.args[0].value, str)):
                out.add(n.args[0].value)
        return out

    # A parameter USED AS A RECORD KEY makes the literals passed to it record reads too.
    # `_handle_base(rec, "scratch_dir")` is how this suite takes `scratch_dir` off a
    # record; the field name never appears in a subscript, only as an argument, and a
    # screen that cannot see that would call the base directories unread and convict the
    # corrected table for saying `--reanalyse` resolves them.
    key_params: "dict[str, set[int]]" = {}
    for fn in [n for n in _ast.walk(tree)
               if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef))]:
        names = [a.arg for a in fn.args.args]
        used: "set[int]" = set()
        for n in _ast.walk(fn):
            key = None
            if (isinstance(n, _ast.Subscript) and isinstance(n.ctx, _ast.Load)
                    and isinstance(n.slice, _ast.Name)):
                key = n.slice.id
            elif (isinstance(n, _ast.Call) and isinstance(n.func, _ast.Attribute)
                    and n.func.attr == "get" and n.args
                    and isinstance(n.args[0], _ast.Name)):
                key = n.args[0].id
            if key is not None and key in names:
                used.add(names.index(key))
        if used:
            key_params[fn.name] = used

    fields = _direct(tree)
    for n in _ast.walk(tree):
        if not isinstance(n, _ast.Call):
            continue
        name = (n.func.id if isinstance(n.func, _ast.Name)
                else n.func.attr if isinstance(n.func, _ast.Attribute) else None)
        for idx in key_params.get(name, ()):
            if idx < len(n.args) and isinstance(n.args[idx], _ast.Constant) \
                    and isinstance(n.args[idx].value, str):
                fields.add(n.args[idx].value)
    return fields


def source_contract_claims(text: str) -> "list[tuple[int, str, str]]":
    """Every ``(line, module, field)`` this text claims one module reads off a record."""
    return [(line_no, module, field)
            for line_no, line in enumerate(text.splitlines(), start=1)
            for module, field in _READS_CLAIM.findall(line)]


def source_contract_offenders(text: str, name: str = "<text>") -> "list[str]":
    """Every ``X reads Y`` claim in *text* that module ``X`` does not actually read."""
    offenders = []
    for line_no, module, field in source_contract_claims(text):
        target = f"{module}.py"
        if not (Path(__file__).resolve().parent / target).is_file():
            offenders.append(f"{name}:{line_no}: `{module}` is not a module here")
            continue
        if field not in _record_reads(target):
            offenders.append(
                f"{name}:{line_no}: claims {module} reads {field!r} off a record, and "
                f"{target} never does. A parameter or local of that name is not the "
                f"channel a reader is being pointed at")
    return offenders


def test_no_prose_names_a_channel_the_reader_does_not_read():
    """POSITIVE POLARITY: the architecture note agrees with the code today.

    The first assertion is the one that stops this being vacuous. A guard over a document
    that makes no claims is green for the same reason a guard over a correct one is, and
    the defect this is for was a *stated* channel name, so the document has to state one.
    """
    text = (Path(pp.__file__)).read_text(encoding="utf-8")
    claims = source_contract_claims(text)

    assert claims, (
        "public_paths.py names no `X reads Y` channel at all; this screen has no subject "
        "and would pass over the stale sentence being restored in any other phrasing")
    assert ("cuda_profile", "profile_rel") in [(m, f) for _, m, f in claims]
    assert source_contract_offenders(text, "public_paths.py") == []
    assert "resolve_scratch_file" in pp.public_payload.__doc__


def test_the_exact_stale_sentence_is_caught():
    """NEGATIVE POLARITY, planted as the sentence that actually shipped.

    `profile_path` is a real identifier in `cuda_profile.py` — the parameter of
    `op_kernel_times` — so this specimen is exactly the one a presence check cannot see.
    """
    planted = ("The in-memory record keeps real, openable paths — ``cuda_profile`` reads "
               "``profile_path`` back out of the very record it is about to serialise.")

    offenders = source_contract_offenders(planted, "planted.py")

    assert offenders, "the stale channel name was not caught"
    assert "profile_path" in offenders[0]
    assert "profile_path" in (Path(pp.__file__).parent / "cuda_profile.py").read_text(
        encoding="utf-8"), (
        "this control is only meaningful while `profile_path` still occurs in that module")


def test_the_true_sentence_passes_the_same_guard():
    """The polarity that stops the guard being satisfied by rejecting every claim."""
    true_claim = ("``cuda_profile`` reads ``profile_rel`` through "
                  "``cuda_competition.resolve_scratch_file``.")

    assert source_contract_offenders(true_claim, "true.py") == []


# ---------------------------------------------------------------------------
# B5b: THE ARCHITECTURE TABLE IN docs/PERF.md IS A CONTRACT, AND IT WAS WRONG
#
# The screen above reads `X reads Y` sentences out of the modules. §4.0.2 of docs/PERF.md
# states the same contract in a *markdown table*, and that is the surface a reader actually
# consults — it is the one place the field kinds are enumerated side by side. It listed
# `profile_path` and `trace_path` as rooted evidence that `--reanalyse` reads back through
# `public_paths.resolve_public_path`. It does not and never did: reanalysis resolves
# `profile_rel` and `trace_rel` through `resolve_scratch_file`, rooted at `scratch_dir`.
#
# So the document that explains the defect carried the defect. The table is now screened
# with the same binding as the docstrings, in BOTH polarities of the claim it makes:
#
#   * a row that names a READER must name only fields some module takes off a record;
#   * a row that says NOTHING OPENS a field must name only fields no module takes off one.
#
# The second half is what makes the corrected table checkable rather than merely different:
# it fails the moment somebody wires a reader to `profile_path` and leaves the row saying
# nothing does.
# ---------------------------------------------------------------------------

_PERF = Path(__file__).resolve().parents[1] / "docs" / "PERF.md"

#: A backticked identifier inside a markdown table cell.
_CELL_FIELD = __import__("re").compile(r"`([A-Za-z_][A-Za-z0-9_]*)`")

#: A backticked MECHANISM in the "who reads it" cell. Dots allowed, because that is how the
#: table spells a reader — `public_paths.resolve_public_path` — and a field-shaped pattern
#: matches none of them. With the field pattern here the stale row read as "names a role,
#: not a mechanism" and was skipped: the screen was green because it could not parse the
#: sentence it existed to convict.
_CELL_MECHANISM = __import__("re").compile(r"`([A-Za-z_][A-Za-z0-9_.]*)`")

#: Phrases a "who reads it" cell uses to say that nothing does.
_NO_READER = ("nothing opens", "nothing reads", "no reader", "never read", "in-process only")


def _bench_modules() -> "list[str]":
    here = Path(__file__).resolve().parent
    return sorted(p.name for p in here.glob("*.py")
                  if not p.name.startswith(("test_", "_")))


def _all_record_reads() -> "dict[str, set[str]]":
    """``{module: fields it takes off a record}`` for every non-test module in bench/."""
    return {name: _record_reads(name) for name in _bench_modules()}


def perf_field_table_rows(text: str) -> "list[tuple[int, list[str], str]]":
    """Every ``| fields | kind | who reads it | serialised |`` row, as (line, fields, reader).

    Read structurally rather than by section heading: a four-column table whose header
    names a *who reads it* column is the shape this contract is stated in, and a table that
    stops being one stops being screened — which is why the caller asserts it found rows.
    """
    rows: "list[tuple[int, list[str], str]]" = []
    in_table = False
    for line_no, line in enumerate(text.splitlines(), start=1):
        cells = [c.strip() for c in line.strip().strip("|").split("|")] if "|" in line else []
        if len(cells) != 4:
            in_table = False
            continue
        if "who reads it" in cells[2].lower():
            in_table = True
            continue
        if not in_table or set(cells[0]) <= set("- :"):
            continue
        fields = _CELL_FIELD.findall(cells[0])
        if fields:
            rows.append((line_no, fields, cells[2]))
    return rows


def perf_field_table_offenders(text: str, name: str = "docs/PERF.md") -> "list[str]":
    """Every field-table row whose reader column disagrees with what the modules read."""
    reads = _all_record_reads()
    everything = set().union(*reads.values()) if reads else set()
    offenders: "list[str]" = []
    for line_no, fields, reader in perf_field_table_rows(text):
        says_none = any(p in reader.lower() for p in _NO_READER)
        named = [m for m in _CELL_MECHANISM.findall(reader)]
        for field in fields:
            if says_none:
                if field in everything:
                    owners = sorted(m for m, f in reads.items() if field in f)
                    offenders.append(
                        f"{name}:{line_no}: the row says nothing opens {field!r}, and "
                        f"{owners} take(s) it off a record. A field with a live reader "
                        f"listed as inert is the same wrong answer as an inert field "
                        f"listed as live")
                continue
            if not named:
                continue  # the row names a role, not a mechanism; nothing to bind to
            if field not in everything:
                offenders.append(
                    f"{name}:{line_no}: the row points a reader ({reader!r}) at {field!r}, "
                    f"and no module in bench/ takes that field off a record. A parameter "
                    f"or local of that name is not the channel a reader is being sent to")
    return offenders


def test_the_perf_field_table_agrees_with_the_modules():
    """POSITIVE POLARITY, and the non-vacuity assertions that make it mean something."""
    text = _PERF.read_text(encoding="utf-8")
    rows = perf_field_table_rows(text)

    assert rows, (
        "docs/PERF.md states no field/kind/reader table at all; this screen has no subject "
        "and the stale row could be restored in any phrasing")
    subjects = {f for _, fields, _ in rows for f in fields}
    assert {"profile_rel", "trace_rel", "profile_path", "trace_path"} <= subjects, (
        f"the table no longer enumerates the four fields the defect was about: {subjects}")
    assert any(any(p in reader.lower() for p in _NO_READER) for _, _, reader in rows), (
        "no row states that nothing opens its field; the corrected table has to make that "
        "claim, or the negative-polarity half of this screen is checking nothing")

    assert perf_field_table_offenders(text) == []


def test_the_exact_stale_perf_row_is_caught():
    """NEGATIVE POLARITY: the row as it shipped, planted verbatim.

    `profile_path` and `trace_path` are real identifiers in `cuda_profile.py` — a parameter
    and an assignment target — so this is exactly the specimen a presence check cannot see.
    """
    planted = (
        "| Field | Kind | Who reads it | Serialised? |\n"
        "|---|---|---|---|\n"
        "| `outputs_dir`, `scratch_dir`, `profile_path`, `trace_path` | *evidence* — a "
        "rooted path per §4.0.1 | a reader; and `--reanalyse`, via "
        "`public_paths.resolve_public_path` | yes, rooted |\n"
    )

    offenders = perf_field_table_offenders(planted, "planted.md")

    assert offenders, "the stale field-table row was not caught"
    assert any("profile_path" in o for o in offenders)
    assert any("trace_path" in o for o in offenders)
    assert not any("scratch_dir" in o for o in offenders), (
        "`scratch_dir` IS resolved by `_handle_base`; convicting it would make this screen "
        "reject the corrected row too")


def test_the_corrected_perf_rows_pass_the_same_guard():
    """The polarity that stops the guard being satisfied by rejecting every row."""
    corrected = (
        "| Field | Kind | Who reads it | Serialised? |\n"
        "|---|---|---|---|\n"
        "| `outputs_dir`, `scratch_dir` | *base* | `_handle_base`, and so `--reanalyse`, "
        "via `public_paths.resolve_public_path` | yes, rooted |\n"
        "| `profile_path`, `trace_path` | *evidence* | nothing opens them | yes, rooted |\n"
    )

    assert perf_field_table_offenders(corrected, "corrected.md") == []


def test_a_field_the_table_calls_inert_but_something_reads_is_caught():
    """The other polarity of the corrected row: an inert claim about a live channel."""
    planted = (
        "| Field | Kind | Who reads it | Serialised? |\n"
        "|---|---|---|---|\n"
        "| `profile_rel`, `trace_rel` | *handle* | nothing opens them | yes |\n"
    )

    offenders = perf_field_table_offenders(planted, "planted.md")

    assert offenders and any("profile_rel" in o for o in offenders), (
        "a field with a live reader was allowed to be described as read by nothing")


def test_the_perf_prose_is_screened_by_the_source_contract_guard_too():
    """`X reads Y` in the document, not only in the modules.

    The docstring screen already covered `public_paths.py`. The same sentence shape occurs
    in prose, and prose is what a reader quotes; a corpus of one module is how the first
    version of this guard stayed green while docs/PERF.md carried the stale channel name.
    """
    text = _PERF.read_text(encoding="utf-8")

    assert source_contract_offenders(text, "docs/PERF.md") == []

    stale = text + (
        "\n\nThe in-memory record keeps real, openable paths — ``cuda_profile`` reads "
        "``profile_path`` back out of the very record it is about to serialise.\n")
    assert source_contract_offenders(stale, "docs/PERF.md"), (
        "the stale channel sentence appended to the real document was not caught")


# ---------------------------------------------------------------------------
# B5c: THE REPEAT CONTRACT IS DOCUMENTED, AND THE DOCUMENT IS TIED TO THE FIELDS
#
# Gate 4 compared repeat 0 and pooled every repeat's samples. The mechanism was fixed;
# the document said nothing, and a document that says nothing about the unit of a check is
# how "we compared the outputs" gets read as "we compared all the outputs".
#
# So §4.0.2.1 states the per-(arm, repeat) contract, and this screen ties it to the fields
# `cuda_competition` actually publishes. Both halves are load-bearing: prose naming a field
# nothing publishes is a promise the code does not keep, and a published repeat field the
# prose never mentions is the repeat-0-only description coming back by omission.
# ---------------------------------------------------------------------------

#: The evidence fields the repeat contract is made of. Each is written by
#: `workload_equivalence` or `compare_workload`, and each is what a reader would have to
#: look at to check that every repeat was compared.
_REPEAT_EVIDENCE_FIELDS = (
    "contributing_repeats",
    "equivalence_unchecked_repeats",
    "equivalence_divergent_repeats",
    "repeat_key",
    "repeats",
)


def test_the_repeat_contract_prose_names_the_fields_the_code_publishes():
    """Both directions, so neither the prose nor the code can drift alone."""
    import cuda_competition as cc

    text = _PERF.read_text(encoding="utf-8")
    section = text.split("#### 4.0.2.1")
    assert len(section) == 2, (
        "docs/PERF.md no longer carries the §4.0.2.1 repeat-contract section; the Gate-4 "
        "unit is undocumented again")
    body = section[1].split("\n## ")[0].split("\n### ")[0]

    for field in _REPEAT_EVIDENCE_FIELDS:
        assert field in body, (
            f"the repeat contract section does not mention {field!r}, which is part of the "
            f"evidence a reader needs to check that every repeat was compared")

    entry = cc.compare_workload(
        cc.Workload(key="w", model_key="m", family="f", seq_len=1, past_len=0),
        [{"arm": cc.ARM_VULKAN, "verdict": cc.ADMISSIBLE, "repeat": 0, "steady_ms": [1.0]}],
        {"arms": {}, "repeats": {}})
    for field in ("contributing_repeats", "equivalence_unchecked_repeats",
                  "equivalence_divergent_repeats"):
        assert field in entry, (
            f"{field} is documented in §4.0.2.1 and `compare_workload` does not publish it")

    assert "repeat 0" in body.lower() or "repeat `0`" in body.lower(), (
        "the section must name the defect it closed — comparing repeat 0 alone — or a "
        "reader cannot tell what changed")


def test_the_repeat_contract_prose_cannot_go_back_to_repeat_zero_only():
    """NEGATIVE POLARITY: strip the per-repeat fields from the section and it must fail.

    A guard over a document is green for a correct document and for one nobody reads. This
    plants the regression — a section that describes the fold without the per-repeat
    evidence — and requires the binding above to notice.
    """
    text = _PERF.read_text(encoding="utf-8")
    body = text.split("#### 4.0.2.1")[1].split("\n### ")[0]

    stripped = body
    for field in _REPEAT_EVIDENCE_FIELDS:
        stripped = stripped.replace(field, "the equivalence result")

    missing = [f for f in _REPEAT_EVIDENCE_FIELDS if f not in stripped]
    assert missing == list(_REPEAT_EVIDENCE_FIELDS), (
        "a repeat-0-only rewrite of this section still mentions the per-repeat evidence "
        "fields; the binding above would not have noticed it")

# ---------------------------------------------------------------------------
# B5d: THE WRITER AND THE READER MUST REFUSE THE SAME BASES
#
# `contained_child` (the reader) refused an absent or empty base. `relative_handle` (the
# writer) resolved an empty base to whatever directory the process happened to be running
# in, and wrote a handle against it. A writer that produces a name the reader will not
# accept puts an unopenable handle in a committed record and spells it exactly like an
# openable one — and the reader's refusal then arrives as "no file at this handle", which
# is a different finding from "this handle was rooted at nothing".
#
# One predicate now, `public_paths.unusable_base`, asked by both halves. Both polarities of
# both halves, over the same table, because a symmetric pair that refuses everything is
# symmetric and useless.
# ---------------------------------------------------------------------------

#: Bases neither half may accept. Each is a value a caller can actually produce: a missing
#: `scratch_dir` off a record (`None`), a field that serialised to the empty string, a
#: whitespace-only directory name, and a number where a path belongs.
_UNUSABLE_BASES = (
    (None, "an absent directory"),
    ("", "a directory field that serialised empty"),
    ("   ", "a whitespace-only directory name"),
    (0, "a number where a path belongs"),
    (False, "a boolean where a path belongs"),
)


def test_writer_and_reader_refuse_the_same_unusable_bases(tmp_path, monkeypatch):
    """NEGATIVE POLARITY, both halves, one table.

    The process is moved INTO ``tmp_path`` first. Without that, the empty base resolves to
    a current directory that does not contain the target, and the old writer refused for
    the wrong reason — the control would have passed against the defect it is named for.
    """
    import _polarity
    import cuda_competition as cc

    monkeypatch.chdir(tmp_path)
    target = tmp_path / "out0.npy"
    target.write_bytes(b"0")
    assert pp._relative_to(target.resolve(), Path("").resolve()) is not None, (
        "the target must be inside the current directory, or the empty-base row below is "
        "satisfied by containment rather than by the base rule")

    for base, why in _UNUSABLE_BASES:
        assert pp.unusable_base(base) is not None, f"{why}: the predicate accepted {base!r}"
        assert pp.contained_child(base, "out0.npy") is None, (
            f"{why}: the reader accepted a handle rooted at {base!r}")
        reason = _polarity.refuses(cc.relative_handle(target, base), because=why)
        assert reason, f"{why}: the writer refused without saying why"


def test_writer_and_reader_both_accept_a_real_base(tmp_path):
    """MUST-PASS POLARITY. A symmetric pair that refuses everything is still useless."""
    import cuda_competition as cc

    target = tmp_path / "out0.npy"
    target.write_bytes(b"0")

    assert pp.unusable_base(tmp_path) is None
    handle, why = cc.relative_handle(target, tmp_path)
    assert handle == "out0.npy", why
    assert pp.contained_child(tmp_path, handle) == target.resolve()


def test_the_writer_no_longer_roots_a_handle_at_the_current_directory(tmp_path, monkeypatch):
    """THE EXACT ASYMMETRY, reproduced as it was: an empty base and a file under CWD.

    With the process sitting in `tmp_path` and the base empty, the old writer resolved the
    base to `Path("").resolve()` — the current directory — and returned a handle the reader
    then refused. The control is meaningful only while the file really is under CWD, which
    the first assertion pins.
    """
    import cuda_competition as cc

    monkeypatch.chdir(tmp_path)
    target = tmp_path / "out0.npy"
    target.write_bytes(b"0")
    assert pp._relative_to(target.resolve(), Path("").resolve()) is not None, (
        "this control needs the target to be inside the current directory, which is the "
        "only condition under which the old writer produced a handle at all")

    handle, why = cc.relative_handle(target, "")

    assert handle is None, (
        f"the writer produced {handle!r} against an empty base; the reader refuses that "
        f"base, so the record would carry a handle nobody can resolve")
    assert "empty" in why


def test_both_halves_ask_the_same_predicate():
    """Structural, so the two cannot drift apart again by one being edited alone.

    The docstrings are stripped first. A guard that reads the whole source is satisfied by
    the sentence saying the predicate is used, which is the annotation-not-mechanism shape
    `bench/_polarity.py` was written to refuse.
    """
    import ast
    import inspect
    import textwrap

    import cuda_competition as cc

    def _body(fn) -> str:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        node = tree.body[0]
        if (node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)):
            node.body = node.body[1:]
        return ast.unparse(node)

    assert "unusable_base(" in _body(pp.contained_child)
    assert "unusable_base(" in _body(cc.relative_handle)
