"""§8.9.19 — the `rewitness/3` cause is only checkable if it looks where the BUILD looks.

WHY THIS FILE EXISTS
--------------------
`ci/check_ledger_census.py` corroborates a same-change cause by re-deriving, from the trees
under comparison, the set of files whose bytes `source_digest_for` hashes: the variant
manifest row, the template it names, and that template's transitive `#include` closure. That
derivation is a PYTHON MIRROR of Rust that lives in `rust/build_support/shader_source_digest.rs`
and `rust/build.rs`. Two implementations of one rule is a defect waiting for someone to move a
directory, and the failure mode is silent in the worst possible direction:

  * if the mirror looks in the wrong place, `source_closure` returns an EMPTY closure, and an
    empty closure convicts every honest cause while authorising nothing — noisy, survivable;
  * if the mirror's normalisation drifts from the Rust's, a cause computed on one developer's
    checkout stops matching the same bytes on another's, and the screen's colour starts
    depending on `core.autocrlf`. That one is not survivable: it is green here, red on `main`,
    which is precisely the class of defect `rewitness/3` was written to end.

So the mirror is pinned here, against the Rust source itself rather than against a copy of what
somebody believed it said. These are cheap string and behaviour assertions on purpose: a test
that needed a built EP would not run in the lane that needs it.

WHAT IS NOT CLAIMED
-------------------
This file does not verify that the ledger's digests are the ones the build produced — that is
`gen_proof_ledger.py --check` and `test_no_entry_carries_a_stale_source_digest`, both of which
read a built artefact. It verifies only that the screen and the build agree about WHICH BYTES
are in play, which is the assumption those screens leave unchecked.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CENSUS = REPO / "ci" / "check_ledger_census.py"
BUILD_RS = REPO / "rust" / "build.rs"
DIGEST_RS = REPO / "rust" / "build_support" / "shader_source_digest.rs"
REGISTER = REPO / "evidence" / "proof_rewitness.json"


def _census():
    """Import the screen as a module, so the pins are on the code the lane actually runs."""
    if not CENSUS.is_file():
        pytest.skip("ci/check_ledger_census.py absent")
    spec = importlib.util.spec_from_file_location("_census_under_test", CENSUS)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_the_screen_looks_for_shaders_where_build_rs_puts_them():
    """`GLSL_DIR_REL` / `GLSL_INCLUDE_DIR_REL` / `SHADER_VARIANTS_REL` vs `build.rs`.

    Read out of the Rust rather than restated: `compile_shaders` joins the manifest dir with
    these components, and if it ever joins different ones the screen must go red here rather
    than quietly compute an empty closure for every stem.
    """
    if not BUILD_RS.is_file():
        pytest.skip("rust/build.rs absent")
    rs = BUILD_RS.read_text(encoding="utf-8")
    mod = _census()

    def joined(var: str) -> str:
        m = re.search(rf"let {var} = manifest((?:\.join\(\"[^\"]+\"\))+);", rs)
        assert m, f"cannot find `let {var} = manifest.join(...)` in rust/build.rs"
        return "rust/" + "/".join(re.findall(r'\.join\("([^"]+)"\)', m.group(1)))

    assert mod.GLSL_DIR_REL == joined("glsl_dir")
    assert mod.GLSL_INCLUDE_DIR_REL == joined("include_dir")
    assert mod.SHADER_VARIANTS_REL == joined("variant_table")


def test_the_include_search_order_matches_the_rust():
    """own directory, then `-I include`, then `shaders/glsl` — in that order.

    Order is not cosmetic: a name that exists in two of the three resolves to a different file
    under a different order, so a closure built with the wrong precedence can declare a cause
    corroborated by bytes the compiler never read.
    """
    if not DIGEST_RS.is_file():
        pytest.skip("shader_source_digest.rs absent")
    rs = DIGEST_RS.read_text(encoding="utf-8")
    m = re.search(r"let candidates = \[(.*?)\];", rs, re.S)
    assert m, "cannot find the include-resolution candidate list in the Rust"
    order = re.findall(r"(parent|include_dir|glsl_dir)\b", m.group(1))
    assert order[:3] == ["parent", "include_dir", "glsl_dir"], order

    src = CENSUS.read_text(encoding="utf-8")
    body = src[src.index("def source_closure"):]
    body = body[: body.index("\ndef ")]
    cands = re.search(r"for cand in \((.*?)\):", body, re.S)
    assert cands, "cannot find the candidate tuple in source_closure"
    assert re.findall(r"(parent|GLSL_INCLUDE_DIR_REL|GLSL_DIR_REL)", cands.group(1))[:3] == [
        "parent", "GLSL_INCLUDE_DIR_REL", "GLSL_DIR_REL",
    ]


@pytest.mark.parametrize(
    "name, variant, canonical",
    [
        ("CRLF line endings", b"#version 450\r\nvoid main() {}\r\n", b"#version 450\nvoid main() {}\n"),
        ("lone CR line endings", b"#version 450\rvoid main() {}\r", b"#version 450\nvoid main() {}\n"),
        ("a leading UTF-8 BOM", b"\xef\xbb\xbf#version 450\n", b"#version 450\n"),
        ("BOM and CRLF together", b"\xef\xbb\xbf#version 450\r\n", b"#version 450\n"),
    ],
)
def test_normalisation_folds_exactly_what_the_rust_folds(name, variant, canonical):
    """The four differences `glslc` cannot see must not change a content id.

    Every one of these is a real thing a checkout does to a file without anybody editing it:
    `core.autocrlf=true` on Windows, an editor that writes a BOM, a file that came through an
    old tool. If any of them moved the content id, the same source would have two ids and a
    correct declaration would be red on half the fleet.
    """
    mod = _census()
    assert mod.normalize_source_text(variant) == canonical, name
    assert mod.content_id(variant) == mod.content_id(canonical), name


def test_normalisation_does_not_fold_anything_else():
    """The other polarity: a REAL edit must still move the id, or the cause proves nothing."""
    mod = _census()
    base = b"#version 450\nvoid main() { tile(1); }\n"
    for different in (
        b"#version 450\nvoid main() { tile(2); }\n",   # a value
        b"#version 450\nvoid  main() { tile(1); }\n",  # whitespace inside a line
        b"#version 450\nvoid main() { tile(1); }",     # a trailing newline
        b"#version 450\n\nvoid main() { tile(1); }\n",  # a blank line
    ):
        assert mod.content_id(base) != mod.content_id(different), different


def test_the_rust_declares_the_same_normalisation_this_mirrors():
    """A pin on the Rust's own body, so a change there is felt here rather than on `main`.

    Three facts, each of which the mirror depends on and none of which is inferable from the
    function's name: the BOM stripped is the UTF-8 one, `\\r` is folded to `\\n`, and a CRLF
    pair collapses to ONE newline rather than two. The last is the one that would silently
    halve nothing and double everything if it drifted.
    """
    if not DIGEST_RS.is_file():
        pytest.skip("shader_source_digest.rs absent")
    rs = DIGEST_RS.read_text(encoding="utf-8")
    start = rs.index("pub fn normalize_shader_text")
    end = rs.find("\npub fn ", start + 1)
    fn = rs[start: end if end > 0 else len(rs)]
    assert "0xEF, 0xBB, 0xBF" in fn, "the Rust no longer strips a UTF-8 BOM"
    assert "b'\\r'" in fn and "b'\\n'" in fn, "the Rust no longer folds CR to LF"
    assert "body[i + 1] == b'\\n'" in fn, "the Rust no longer collapses CRLF to a single LF"

    mod = _census()
    assert mod.normalize_source_text(b"\xef\xbb\xbfa\r\nb\rc") == b"a\nb\nc"


def test_every_v3_cause_path_reads_as_declared_in_this_working_tree():
    """The live arm: the register's own cause content, re-read from the files on disk.

    `ci/check_ledger_census.py` already checks this against the two git trees the ledger moved
    between. This checks the tip of the branch as a developer sees it, with a second, dumber
    implementation of the content id — because the screen agreeing with itself is what a drifted
    mirror also looks like.
    """
    import hashlib

    if not REGISTER.is_file():
        pytest.skip("no rewitness register")
    doc = json.loads(REGISTER.read_text(encoding="utf-8"))
    v3 = [r for r in doc.get("rewitness", []) if r.get("schema") == "rewitness/3"]
    if not v3:
        pytest.skip("no rewitness/3 records yet")
    wrong = []
    for rec in v3:
        for p in rec["caused_by_content"]["paths"]:
            f = REPO / p["path"]
            if not f.is_file():
                wrong.append((p["path"], "absent from the working tree", p["new"]))
                continue
            data = f.read_bytes()
            if data.startswith(b"\xef\xbb\xbf"):
                data = data[3:]
            data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            got = hashlib.sha256(data).hexdigest()
            if got != p["new"]:
                wrong.append((p["path"], got, p["new"]))
    assert not wrong, (
        "declared cause content does not match the working tree:\n"
        + "\n".join(f"  {a}: tree has {b}, record says new={c}" for a, b, c in wrong)
    )


def test_a_v3_cause_may_not_point_at_generated_evidence():
    """The anti-tautology rule, asserted on the register rather than only in the checker.

    A cause read out of `evidence/` would be the record proving itself. The screen refuses it;
    this makes the refusal visible to anyone reading the register's tests rather than the
    checker's internals.
    """
    mod = _census()
    if not REGISTER.is_file():
        pytest.skip("no rewitness register")
    doc = json.loads(REGISTER.read_text(encoding="utf-8"))
    for rec in doc.get("rewitness", []):
        if rec.get("schema") != "rewitness/3":
            continue
        for p in rec["caused_by_content"]["paths"]:
            assert not mod._refused_cause_path(p["path"]), p["path"]
            assert p["path"].startswith(("rust/shaders/", "rust/src/ops/shader_variants.txt")), (
                f"{p['path']} is not production shader source"
            )


# ══════════════════════════════════════════════════════════════════════════════════════════
# `_blob_at(WORKTREE)` vs git tree semantics — issue #60
#
# A symlink in a git tree is a blob whose CONTENT IS THE TARGET STRING, mode 120000.
# `git show <rev>:<path>` prints that string. `Path.read_bytes()` does the opposite: it
# opens whatever is at the far end, which for a link pointing out of the repository means
# the census would hash a file that is not in this repository at all. The two branches of
# `_blob_at` must agree, and the working-tree one must never follow.
#
# EVERY TEST HERE RUNS ON A RUNNER WITH NO SYMLINK PRIVILEGE, which is this repository's
# Windows lane and every unelevated Windows developer box. Git checks a symlink out as an
# ORDINARY FILE CONTAINING THE TARGET STRING when `core.symlinks=false`, so the blob is
# identical and the assertion is the same one — the tests that genuinely need a real link on
# disk are the ones that skip, and they say what they are skipping.
# ══════════════════════════════════════════════════════════════════════════════════════════

def _git(args, cwd):
    env = dict(os.environ)
    env.update(
        GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
        GIT_COMMITTER_EMAIL="t@t",
    )
    return subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True,
        encoding="utf-8", errors="replace", env=env,
    )


def _repo_with_symlink(tmp_path: Path, target: str) -> tuple[Path, str]:
    """A one-commit repo whose `link` is a tracked symlink to `target`. -> (repo, rev).

    The entry is written with `git update-index --cacheinfo 120000`, NOT with `os.symlink`.
    That is the whole portability trick: it produces a real mode-120000 tree entry on every
    platform, including the ones where the process may not create a link, and the checkout
    then materialises it however that platform can.
    """
    repo = tmp_path / "sl"
    repo.mkdir()
    _git(["init", "-q", "-b", "main"], repo)
    (repo / "ordinary.txt").write_text("not a link\n", encoding="utf-8")
    _git(["add", "-A"], repo)
    blob = _git(["hash-object", "-w", "--stdin", "--path", "link"], repo)
    proc = subprocess.run(
        ["git", "hash-object", "-w", "--stdin"], cwd=str(repo), input=target.encode("utf-8"),
        capture_output=True,
    )
    sha = proc.stdout.decode().strip()
    assert sha, blob.stderr
    _git(["update-index", "--add", "--cacheinfo", f"120000,{sha},link"], repo)
    _git(["commit", "-q", "-m", "one symlink"], repo)
    _git(["checkout", "-q", "--", "."], repo)
    rev = _git(["rev-parse", "HEAD"], repo).stdout.strip()
    return repo, rev


@pytest.mark.parametrize(
    "target",
    [
        "ordinary.txt",                     # inside the repo
        "../outside-the-repo.txt",          # OUTSIDE it — the case that motivated this
        "/etc/passwd",                      # absolute, outside, and famously interesting
        "nothing-is-here.txt",              # broken link
    ],
    ids=["inside", "outside-relative", "outside-absolute", "broken"],
)
def test_a_symlink_blob_reads_the_same_from_the_worktree_and_from_a_rev(tmp_path, target):
    """The alignment itself: both branches of `_blob_at` return the TARGET STRING."""
    mod = _census()
    repo, rev = _repo_with_symlink(tmp_path, target)
    (tmp_path / "outside-the-repo.txt").write_text("SECRET-OUTSIDE-CONTENT\n", encoding="utf-8")

    from_rev = mod._blob_at(repo, rev, "link")
    from_worktree = mod._blob_at(repo, mod.WORKTREE, "link")
    assert from_rev == target.encode("utf-8"), (
        f"git stores the target string for a symlink; got {from_rev!r}"
    )
    assert from_worktree == from_rev, (
        "the working-tree branch and the git-rev branch of _blob_at disagree about a "
        f"symlink: worktree {from_worktree!r} vs rev {from_rev!r}"
    )
    assert mod._content_id_at(repo, None, "link", {}) == mod._content_id_at(
        repo, rev, "link", {}
    )


def test_the_worktree_branch_never_reads_what_a_symlink_points_at(tmp_path):
    """The security half, stated separately from the equality half.

    Equality could in principle be achieved by making BOTH branches follow the link. This
    asserts the direction: the bytes at the far end never appear in the answer, and the file
    outside the repository is never opened.
    """
    mod = _census()
    outside = tmp_path / "outside-the-repo.txt"
    outside.write_text("SECRET-OUTSIDE-CONTENT\n", encoding="utf-8")
    repo, rev = _repo_with_symlink(tmp_path, "../outside-the-repo.txt")

    for rev_arg in (None, mod.WORKTREE, rev):
        data = mod._blob_at(repo, rev_arg, "link")
        assert data is not None
        assert b"SECRET-OUTSIDE-CONTENT" not in data, (
            f"_blob_at({rev_arg!r}) followed a symlink out of the repository"
        )
        assert data == b"../outside-the-repo.txt"


def test_a_real_on_disk_symlink_reads_as_its_target(tmp_path):
    """The same assertion, on a link the FILESYSTEM knows about rather than only the index.

    Skips where the process cannot create one — which is this repository's Windows lane, and
    is stated as a skip rather than passed silently, because a green produced by an absence
    is the shape §8.9 spends most of its length on.
    """
    mod = _census()
    repo = tmp_path / "real"
    repo.mkdir()
    _git(["init", "-q", "-b", "main"], repo)
    (repo / "ordinary.txt").write_text("not a link\n", encoding="utf-8")
    try:
        os.symlink("ordinary.txt", repo / "link")
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"this environment cannot create a symlink ({exc}); the index-mode "
                    f"parametrisation above covers the same assertion here")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "a real link"], repo)
    rev = _git(["rev-parse", "HEAD"], repo).stdout.strip()
    mode = mod._index_mode(repo, "link")
    assert mode == mod.GIT_MODE_SYMLINK, f"git did not record a symlink; mode={mode}"
    assert mod._blob_at(repo, mod.WORKTREE, "link") == b"ordinary.txt"
    assert mod._blob_at(repo, rev, "link") == b"ordinary.txt"


def test_an_untracked_symlink_still_reads_as_its_target(tmp_path):
    """A cause path may be added and not yet staged; the index has no mode for it then."""
    mod = _census()
    repo = tmp_path / "untracked"
    repo.mkdir()
    _git(["init", "-q", "-b", "main"], repo)
    try:
        os.symlink("../elsewhere.txt", repo / "link")
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"this environment cannot create a symlink ({exc})")
    assert mod._index_mode(repo, "link") is None
    assert mod._blob_at(repo, mod.WORKTREE, "link") == b"../elsewhere.txt"


def test_the_index_and_the_filesystem_disagreeing_is_refused_not_guessed(tmp_path):
    """A path tracked as an ordinary blob that is a link on disk. Two oracles, no verdict."""
    mod = _census()
    repo = tmp_path / "disagree"
    repo.mkdir()
    _git(["init", "-q", "-b", "main"], repo)
    (repo / "real.txt").write_text("payload\n", encoding="utf-8")
    (repo / "subject.txt").write_text("subject\n", encoding="utf-8")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "two ordinary files"], repo)
    (repo / "subject.txt").unlink()
    try:
        os.symlink("real.txt", repo / "subject.txt")
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"this environment cannot create a symlink ({exc})")
    with pytest.raises(mod.WorktreeBlobError):
        mod._blob_at(repo, mod.WORKTREE, "subject.txt")


@pytest.mark.skipif(os.name != "nt", reason="directory junctions are a Windows reparse point")
def test_a_directory_junction_is_refused_rather_than_read_through(tmp_path):
    """Junctions need no privilege on Windows, so this is the reparse case that IS reachable
    on the lane. git would descend into it as a directory; this screen wants a blob, and a
    path that is a tree on one side and a blob on the other is refused."""
    mod = _census()
    repo = tmp_path / "junction"
    repo.mkdir()
    _git(["init", "-q", "-b", "main"], repo)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "payload.txt").write_text("payload\n", encoding="utf-8")
    r = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(repo / "j"), str(outside)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        pytest.skip(f"could not create a junction: {r.stdout}{r.stderr}")
    with pytest.raises(mod.WorktreeBlobError):
        mod._blob_at(repo, mod.WORKTREE, "j")


def test_an_ordinary_file_is_still_read_as_its_bytes(tmp_path):
    """The other polarity of every assertion above: nothing here has made a plain file
    unreadable, and a directory is still ABSENT rather than an error."""
    mod = _census()
    repo = tmp_path / "plain"
    repo.mkdir()
    _git(["init", "-q", "-b", "main"], repo)
    (repo / "a.txt").write_text("hello\r\nworld\n", encoding="utf-8", newline="")
    (repo / "sub").mkdir()
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "plain"], repo)
    assert mod._blob_at(repo, mod.WORKTREE, "a.txt") == b"hello\r\nworld\n"
    assert mod._blob_at(repo, mod.WORKTREE, "sub") is None
    assert mod._blob_at(repo, mod.WORKTREE, "no-such-file") is None


def test_a_directory_at_a_rev_is_absent_not_a_tree_listing(tmp_path):
    """`git show <rev>:<dir>` prints a tree listing. Hashing that as file content would make
    a cause path that is a directory corroborate whatever the listing happens to say."""
    mod = _census()
    repo = tmp_path / "dir"
    repo.mkdir()
    _git(["init", "-q", "-b", "main"], repo)
    (repo / "sub").mkdir()
    (repo / "sub" / "f.txt").write_text("x\n", encoding="utf-8")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "a directory"], repo)
    rev = _git(["rev-parse", "HEAD"], repo).stdout.strip()
    assert mod._blob_at(repo, rev, "sub") is None
    assert mod._blob_at(repo, mod.WORKTREE, "sub") is None


def test_a_submodule_path_is_absent_on_both_branches(tmp_path):
    """A gitlink has no blob. `git show <rev>:<path>` fails on one; the working tree holds a
    directory on the other. Both must say ABSENT, or the two branches disagree again."""
    mod = _census()
    outer = tmp_path / "outer"
    inner = tmp_path / "inner"
    for repo in (outer, inner):
        repo.mkdir()
        _git(["init", "-q", "-b", "main"], repo)
        (repo / "f.txt").write_text("x\n", encoding="utf-8")
        _git(["add", "-A"], repo)
        _git(["commit", "-q", "-m", "init"], repo)
    r = _git(["-c", "protocol.file.allow=always", "submodule", "add", "-q", str(inner), "sub"], outer)
    if r.returncode != 0:
        pytest.skip(f"submodules unavailable here: {r.stderr.strip()[:120]}")
    _git(["commit", "-q", "-m", "add submodule"], outer)
    rev = _git(["rev-parse", "HEAD"], outer).stdout.strip()
    assert mod._index_mode(outer, "sub") == mod.GIT_MODE_GITLINK
    assert mod._blob_at(outer, mod.WORKTREE, "sub") is None
    assert mod._blob_at(outer, rev, "sub") is None

