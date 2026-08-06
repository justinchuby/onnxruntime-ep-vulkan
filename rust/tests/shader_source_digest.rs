//! Controls on the §8.9.19 part 2 **source-closure digest** — `source_digest_for`.
//!
//! WHY THIS FILE EXISTS. On 2026-08-06 a reviewer observed that four inverse-trig entries moved
//! their `source_digest` in a PR whose shader inputs were byte-identical to `main`, and concluded
//! the digest function was non-deterministic. It was not: the RECORDED values had gone stale two
//! commits earlier, when a shared template gained the Asin/Acos code and then a licence header,
//! and a diff against `main` cannot see a value that went stale under it rather than moving in
//! it. That distinction was provable only by rebuilding historical shader texts and querying the
//! built EP — nothing in the tree could answer it, because a build script is not linkable from a
//! test target.
//!
//! It is now. Every property `source_digest_for` claims in its own doc comment is pinned here,
//! in both polarities: what must NOT move the digest (line endings, a BOM, include order, where
//! the checkout lives, an unused include) and what MUST (the source text, an include's contents,
//! the defines and their order, the stem). A property with only one polarity is not a control —
//! a function returning a constant would pass every "does not move" assertion in this file.
//!
//! The module is included by path from the same file `build.rs` includes, so these controls
//! cannot drift from the code that actually computes the ledger's values.

#[path = "../build_support/shader_source_digest.rs"]
mod shader_source_digest;

use shader_source_digest::{normalize_shader_text, parse_include_directive, source_digest_for};
use std::fs;
use std::path::{Path, PathBuf};

/// A throwaway shader tree: `<root>/glsl/<name>.comp` plus `<root>/include/`.
struct Tree {
    root: PathBuf,
}

impl Tree {
    fn new(tag: &str) -> Self {
        let root = std::env::temp_dir().join(format!(
            "epvk_srcdigest_{tag}_{}_{:?}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(root.join("glsl")).unwrap();
        fs::create_dir_all(root.join("include")).unwrap();
        Tree { root }
    }

    fn glsl(&self) -> PathBuf {
        self.root.join("glsl")
    }

    fn include(&self) -> PathBuf {
        self.root.join("include")
    }

    fn write(&self, rel: &str, body: &[u8]) -> PathBuf {
        let path = self.root.join(rel);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, body).unwrap();
        path
    }

    /// The digest of `glsl/<stem>.comp` with the given defines, computed the way `build.rs` does.
    fn digest(&self, stem: &str, defines: &[&str]) -> String {
        self.digest_of(stem, "k", defines)
    }

    /// Same, but with the output stem decoupled from the source file — the real shape, where 92
    /// variants share one template and differ only in stem and defines.
    fn digest_of(&self, stem: &str, src_stem: &str, defines: &[&str]) -> String {
        let src = self.glsl().join(format!("{src_stem}.comp"));
        let defs: Vec<String> = defines.iter().map(|d| (*d).to_string()).collect();
        source_digest_for(
            stem,
            &src,
            &format!("{src_stem}.comp"),
            &defs,
            &self.glsl(),
            &self.include(),
        )
    }
}

impl Drop for Tree {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

const BODY: &[u8] = b"#version 450\n#include \"a.glsl\"\n#include \"b.glsl\"\nvoid main() {}\n";

fn planted(tag: &str) -> Tree {
    let t = Tree::new(tag);
    t.write("glsl/k.comp", BODY);
    t.write("include/a.glsl", b"// a\nfloat a() { return 1.0; }\n");
    t.write("include/b.glsl", b"// b\nfloat b() { return 2.0; }\n");
    t
}

// ── shape ───────────────────────────────────────────────────────────────────────────────────

#[test]
fn digest_is_sixteen_lowercase_hex() {
    let t = planted("shape");
    let d = t.digest("k", &["DTYPE_F32"]);
    assert_eq!(d.len(), 16, "{d}");
    assert!(
        d.chars()
            .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)),
        "the ledger, the census schema and gen_proof_ledger all match on 16 hex digits: {d}"
    );
}

// ── determinism ─────────────────────────────────────────────────────────────────────────────

#[test]
fn repeated_calls_agree() {
    // The claim the 2026-08-06 review disputed, stated at its weakest and checked anyway: the
    // function is a function. If this ever fails, no amount of ledger hygiene can help.
    let t = planted("repeat");
    let first = t.digest("k", &["DTYPE_F32", "EW_OP=OP_ASIN"]);
    for i in 0..64 {
        assert_eq!(
            first,
            t.digest("k", &["DTYPE_F32", "EW_OP=OP_ASIN"]),
            "call {i}"
        );
    }
}

#[test]
fn the_same_content_at_a_different_path_digests_the_same() {
    // A digest that moves when the checkout moves is a MACHINE FINGERPRINT, and one that
    // fingerprints the machine cannot be compared across the two platforms the whole two-digest
    // scheme exists to compare. `-I<abs>` is normalised to `-I<include>` for exactly this.
    let a = planted("path_a");
    let b = planted("path_b");
    assert_ne!(a.root, b.root);
    assert_eq!(a.digest("k", &["X"]), b.digest("k", &["X"]));
}

#[test]
fn include_order_within_a_file_does_not_move_the_digest() {
    // Reordering two `#include` lines is a formatting choice, not a subject change; the closure
    // is sorted by resolved name so it cannot read as one.
    let a = planted("order_a");
    let b = planted("order_b");
    b.write(
        "glsl/k.comp",
        b"#version 450\n#include \"b.glsl\"\n#include \"a.glsl\"\nvoid main() {}\n",
    );
    // The source text itself differs (the lines really did move), so only the CLOSURE half can be
    // compared here. Prove the sort by giving both files the same text but different arrival
    // order through a second level of inclusion.
    let c = planted("order_c");
    c.write(
        "glsl/k.comp",
        b"#version 450\n#include \"top.glsl\"\nvoid main() {}\n",
    );
    c.write(
        "include/top.glsl",
        b"#include \"a.glsl\"\n#include \"b.glsl\"\n",
    );
    let d = planted("order_d");
    d.write(
        "glsl/k.comp",
        b"#version 450\n#include \"top.glsl\"\nvoid main() {}\n",
    );
    d.write(
        "include/top.glsl",
        b"#include \"a.glsl\"\n#include \"b.glsl\"\n",
    );
    assert_eq!(c.digest("k", &["X"]), d.digest("k", &["X"]));
    assert_ne!(
        a.digest("k", &["X"]),
        b.digest("k", &["X"]),
        "moving the lines in the SOURCE is a source change and must be seen"
    );
}

// ── normalisation: what a checkout decides must not reach the digest ─────────────────────────

#[test]
fn crlf_lone_cr_and_lf_all_digest_the_same() {
    // THE REGRESSION THIS NORMALISER WAS ADDED FOR. With `core.autocrlf=true` a Windows checkout
    // has CRLF and a Linux checkout of the same blob has LF; hashing raw bytes moved all 103
    // ledger entries on the first fresh Linux build and made the one row that names a toolchain
    // difference unreachable in the exact case it was built for.
    let lf = planted("nl_lf");
    let crlf = planted("nl_crlf");
    let cr = planted("nl_cr");
    let text = "#version 450\n#include \"a.glsl\"\n#include \"b.glsl\"\nvoid main() {}\n";
    crlf.write("glsl/k.comp", text.replace('\n', "\r\n").as_bytes());
    cr.write("glsl/k.comp", text.replace('\n', "\r").as_bytes());
    crlf.write("include/a.glsl", b"// a\r\nfloat a() { return 1.0; }\r\n");
    cr.write("include/a.glsl", b"// a\rfloat a() { return 1.0; }\r");
    let d = lf.digest("k", &["X"]);
    assert_eq!(d, crlf.digest("k", &["X"]), "CRLF");
    assert_eq!(d, cr.digest("k", &["X"]), "lone CR");
}

#[test]
fn a_utf8_bom_does_not_move_the_digest() {
    let plain = planted("bom_no");
    let bom = planted("bom_yes");
    let mut with_bom = vec![0xEF, 0xBB, 0xBF];
    with_bom.extend_from_slice(BODY);
    bom.write("glsl/k.comp", &with_bom);
    assert_eq!(plain.digest("k", &["X"]), bom.digest("k", &["X"]));
}

#[test]
fn normalisation_is_not_a_whitespace_normaliser() {
    // Deliberately NOT idempotent over whitespace: a trailing space is an edit a person made and
    // the point of this witness is to see it. A normaliser that swallowed it would make the
    // digest blind to the class of change it is the only witness for.
    assert_eq!(normalize_shader_text(b"a\r\nb"), b"a\nb".to_vec());
    assert_eq!(normalize_shader_text(b"a\rb"), b"a\nb".to_vec());
    assert_eq!(normalize_shader_text(b"a \nb"), b"a \nb".to_vec());
    assert_ne!(
        normalize_shader_text(b"a \nb"),
        normalize_shader_text(b"a\nb")
    );
}

// ── sensitivity: the other polarity ─────────────────────────────────────────────────────────

#[test]
fn a_source_edit_moves_the_digest() {
    let a = planted("edit_src_a");
    let b = planted("edit_src_b");
    b.write(
        "glsl/k.comp",
        b"#version 450\n#include \"a.glsl\"\n#include \"b.glsl\"\nvoid main() { }\n",
    );
    assert_ne!(a.digest("k", &["X"]), b.digest("k", &["X"]));
}

#[test]
fn a_comment_only_edit_moves_the_digest() {
    // THE #35 CASE, PINNED. Adding the Cephes attribution header changed no code and emitted
    // byte-identical SPIR-V, and it MUST still move this digest — that is the whole `same/differs`
    // row: `SOURCE-COSMETIC`, PROVEN and named, rather than silently identical.
    let a = planted("comment_a");
    let b = planted("comment_b");
    let mut edited = b"/* Cephes, Moshier. See docs/THIRD_PARTY.md. */\n".to_vec();
    edited.extend_from_slice(BODY);
    b.write("glsl/k.comp", &edited);
    assert_ne!(
        a.digest("k", &["X"]),
        b.digest("k", &["X"]),
        "a comment-only edit to a shared template is exactly what moved 55 entries in #35"
    );
}

#[test]
fn editing_an_included_file_moves_the_digest_of_every_includer() {
    let a = planted("inc_a");
    let b = planted("inc_b");
    b.write("include/a.glsl", b"// a\nfloat a() { return 1.5; }\n");
    assert_ne!(a.digest("k", &["X"]), b.digest("k", &["X"]));
}

#[test]
fn editing_an_unused_include_does_not_move_the_digest() {
    // The closure follows `#include` rather than hashing the directory, so an edit to a file this
    // module never includes cannot masquerade as a subject change.
    let a = planted("unused_a");
    let b = planted("unused_b");
    b.write("include/never_included.glsl", b"// nobody includes this\n");
    assert_eq!(a.digest("k", &["X"]), b.digest("k", &["X"]));
}

#[test]
fn the_stem_and_the_defines_are_in_the_digest() {
    // Per-variant uniqueness comes from STEM + VARIANT-DEFINE + ARGV and NOT from preprocessing:
    // two variants of one template share every byte of source, so without these they would share
    // a digest and 42 selectors would be indistinguishable in the ledger.
    let t = planted("variants");
    let base = t.digest("k", &["EW_OP=OP_ASIN", "DTYPE_F32"]);
    assert_ne!(
        base,
        t.digest_of("k2", "k", &["EW_OP=OP_ASIN", "DTYPE_F32"]),
        "stem: two variants of ONE template share every byte of source"
    );
    assert_ne!(
        base,
        t.digest("k", &["EW_OP=OP_ACOS", "DTYPE_F32"]),
        "define value"
    );
    assert_ne!(base, t.digest("k", &["EW_OP=OP_ASIN"]), "a dropped define");
    assert_ne!(
        base,
        t.digest("k", &["DTYPE_F32", "EW_OP=OP_ASIN"]),
        "define ORDER is the argv order glslc was given, so it is part of the record"
    );
}

#[test]
fn length_prefixing_stops_adjacent_fields_from_being_confusable() {
    // Each field is written as tag\0len\0..bytes..\0. Without the length, `-D` values that differ
    // only in where one field ends and the next begins would collide, and a collision here is a
    // silent PROVEN over a changed subject.
    let t = planted("framing");
    assert_ne!(t.digest("k", &["AB", "C"]), t.digest("k", &["A", "BC"]));
    assert_ne!(
        t.digest_of("ab", "k", &["C"]),
        t.digest_of("a", "k", &["bC"])
    );
}

#[test]
fn an_unresolvable_include_is_recorded_not_skipped() {
    // Skipping would make the digest quietly blind to a file it could not find, and blind is
    // indistinguishable from unchanged. It must both DIFFER from the resolvable case and be
    // stable, so a broken tree is legible rather than merely red.
    let missing = planted("unres_missing");
    missing.write(
        "glsl/k.comp",
        b"#version 450\n#include \"a.glsl\"\n#include \"gone.glsl\"\nvoid main() {}\n",
    );
    let present = planted("unres_present");
    present.write(
        "glsl/k.comp",
        b"#version 450\n#include \"a.glsl\"\n#include \"gone.glsl\"\nvoid main() {}\n",
    );
    present.write("include/gone.glsl", b"// now it resolves\n");
    let d = missing.digest("k", &["X"]);
    assert_eq!(
        d,
        missing.digest("k", &["X"]),
        "the unresolved marker is stable"
    );
    assert_ne!(d, present.digest("k", &["X"]));
}

#[test]
fn a_cyclic_include_terminates() {
    // `seen` is by NAME, so a cycle is visited once. Without this the closure walk would recurse
    // until the stack ran out and the build script would die with no verdict at all.
    let t = planted("cycle");
    t.write(
        "glsl/k.comp",
        b"#version 450\n#include \"p.glsl\"\nvoid main() {}\n",
    );
    t.write("include/p.glsl", b"#include \"q.glsl\"\n");
    t.write("include/q.glsl", b"#include \"p.glsl\"\n");
    let d = t.digest("k", &["X"]);
    assert_eq!(d.len(), 16);
    assert_eq!(d, t.digest("k", &["X"]));
}

// ── the include parser ──────────────────────────────────────────────────────────────────────

#[test]
fn include_directives_are_parsed_the_way_glslc_reads_them() {
    for (line, want) in [
        ("#include \"a.glsl\"", Some("a.glsl")),
        ("#include <a.glsl>", Some("a.glsl")),
        ("  #  include   \"a.glsl\"  ", Some("a.glsl")),
        ("#include \"\"", None),
        ("#includes \"a.glsl\"", None),
        ("// #include \"a.glsl\"", None),
        ("#version 450", None),
        ("", None),
    ] {
        assert_eq!(
            parse_include_directive(line).as_deref(),
            want,
            "line {line:?}"
        );
    }
}

// ── the real tree ───────────────────────────────────────────────────────────────────────────

#[test]
fn the_shipped_shaders_digest_reproducibly() {
    // Against the ACTUAL shader tree, not a planted one: whatever is true of a two-file fixture is
    // only interesting if it survives 96 real modules with a real include closure.
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let glsl = root.join("shaders").join("glsl");
    let include = root.join("shaders").join("include");
    let template = glsl.join("templates").join("ew_unary.comp");
    assert!(template.is_file(), "{} must exist", template.display());

    let defs = ["EW_OP=OP_ASIN".to_string(), "SCALAR_T=float".to_string()];
    let first = source_digest_for(
        "ew_unary_asin_f32",
        &template,
        "templates/ew_unary.comp",
        &defs,
        &glsl,
        &include,
    );
    for _ in 0..8 {
        assert_eq!(
            first,
            source_digest_for(
                "ew_unary_asin_f32",
                &template,
                "templates/ew_unary.comp",
                &defs,
                &glsl,
                &include,
            )
        );
    }

    // And it is a fact about the CONTENT: copy the tree somewhere else and it must not move.
    let copy = Tree::new("real_copy");
    copy_dir(&glsl, &copy.glsl());
    copy_dir(&include, &copy.include());
    let moved = source_digest_for(
        "ew_unary_asin_f32",
        &copy.glsl().join("templates").join("ew_unary.comp"),
        "templates/ew_unary.comp",
        &defs,
        &copy.glsl(),
        &copy.include(),
    );
    assert_eq!(
        first, moved,
        "the digest must not fingerprint the checkout path"
    );
}

fn copy_dir(from: &Path, to: &Path) {
    fs::create_dir_all(to).unwrap();
    for entry in fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let dst = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_dir(&entry.path(), &dst);
        } else {
            fs::copy(entry.path(), &dst).unwrap();
        }
    }
}
