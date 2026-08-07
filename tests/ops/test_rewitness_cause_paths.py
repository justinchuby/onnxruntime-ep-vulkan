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
import re
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
