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
    assert got == "<repo>/bench/results/x.npy", got
    assert pp.scan(got) == []


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
    ordinary thing to type and would put an account name into ~200 fields of a
    1.5MB evidence record.

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
# The repository-wide ratchet
# ---------------------------------------------------------------------------

def _legacy_doc() -> dict:
    return json.loads(gen.BASELINE.read_text(encoding="utf-8"))


def test_no_committed_evidence_file_leaks_unless_it_is_declared():
    """Every tracked evidence file is scanned; new leaks are a failure.

    Not a repository-wide *gate*: 299 files named a machine before
    `bench/public_paths.py` existed, and a screen that is red on the day it
    lands is a screen that gets skipped. So the declared set in
    `bench/public_path_legacy.json` is the ceiling and this test is the ratchet
    — an undeclared leak, or a declared file leaking more than declared, fails.

    The failure text names the files and the shapes, because "3 files failed"
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
