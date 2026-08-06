//! The §8.9.19 part 2 **source-closure digest**, shared by `build.rs` and its controls.
//!
//! This lives outside `build.rs` for one reason: a digest whose determinism nobody can test is a
//! digest whose determinism nobody knows. On 2026-08-06 a reviewer read a ledger whose recorded
//! `source_digest`s had gone stale under a shared template and concluded the FUNCTION was
//! non-deterministic. It was not — but nothing in the tree could show that, because a build
//! script is not linkable from a test target. It is now: `rust/tests/shader_source_digest.rs`
//! includes this exact file and pins every property the digest claims, in both polarities.
//!
//! Included by path from both sides rather than published from the library, because this is
//! build-time machinery: the library must not gain a public API surface so the build script can
//! be tested. The file is byte-for-byte the code `build.rs` runs, so the controls cannot drift
//! from the function that computes the ledger's values.

use std::fs;
use std::path::Path;

/// FNV-1a/64, matching `registry::fnv1a64` and `rust/tools/gen_proof_ledger.py`.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Digest one module's **source closure** — §8.9.19 part 2's second digest.
///
/// **Why there are two digests and not one.** No single hash can be sensitive to the kernel and
/// blind to the compiler, because the compiler is a function whose output is the only thing that
/// actually runs. `shader_digest_for` hashes the SPIR-V, so it moves when `glslc` moves; this one
/// hashes what `glslc` was *given*, so it does not. Their **disagreement** is the instrument:
///
/// | `spirv_digest` | `source_digest` | reading |
/// |---|---|---|
/// | same | same | `PROVEN` |
/// | differs | same | frame delta `toolchain` — the Linux case, claimable and disclosed |
/// | differs | differs | `SUBJECT-CHANGED` — no claim |
/// | same | differs | `SOURCE-COSMETIC` — `PROVEN`, and named |
///
/// It COVERS, per §8.9.19 part 2:
///
/// * the `.comp` source text;
/// * **every file reachable through the `-I` include directory** — resolved by following
///   `#include` recursively rather than by hashing the directory, so that an edited include this
///   module does not use cannot masquerade as a subject change;
/// * the `shader_variants.txt` row — stem, source, and the `-D` assignments in row order;
/// * the `glslc` argv **minus the compiler binary and its version**, with the `-I` path replaced
///   by a placeholder and `-o`/the source path omitted, because those are absolute paths that
///   differ between two checkouts of the same tree and would make the digest a machine
///   fingerprint.
///
/// It is DELIBERATELY BLIND to compiler behaviour entirely — a miscompilation, an optimiser
/// difference, a codegen bug. That blindness is the point, and §8.9.19 part 2 records the pair's
/// joint residual (a compiler bug) as **disclosed rather than closed**: that is why the
/// `differs/same` row is claimed out loud instead of silently.
///
/// An `#include` that cannot be resolved is hashed as an explicit `INCLUDE-UNRESOLVED` marker
/// rather than skipped. Skipping it would make the digest quietly blind to a file it could not
/// find, which is the failure this digest exists to make impossible.
///
/// **Every text body is normalised through [`normalize_shader_text`] before it is hashed.** The
/// first version of this function hashed `fs::read` output directly, which made the digest a
/// **line-ending fingerprint**: with `core.autocrlf=true` a Windows checkout has CRLF and a Linux
/// checkout of the same blob has LF, so all 103 ledger entries read `source_digest` MOVED on the
/// first fresh Linux `.so` — and the `differs/same` row above, the only row that names a
/// toolchain difference, became unreachable in the exact case it was built for. Excluding the
/// absolute `-I` path and then hashing the bytes a checkout option chose was the same mistake one
/// level down: a digest that requires a git configuration to be comparable is a machine
/// fingerprint with extra steps.
pub fn source_digest_for(
    stem: &str,
    src: &Path,
    glsl_source_label: &str,
    defines: &[String],
    glsl_dir: &Path,
    include_dir: &Path,
) -> String {
    let mut input: Vec<u8> = Vec::new();
    let push = |input: &mut Vec<u8>, tag: &str, bytes: &[u8]| {
        input.extend_from_slice(tag.as_bytes());
        input.push(0);
        input.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        input.extend_from_slice(bytes);
        input.push(0);
    };

    push(&mut input, "STEM", stem.as_bytes());
    push(&mut input, "VARIANT-SOURCE", glsl_source_label.as_bytes());
    for d in defines {
        push(&mut input, "VARIANT-DEFINE", d.as_bytes());
    }
    // The argv the ruling names, minus the binary, its version, `-o <path>` and the source path.
    for arg in ["-fshader-stage=compute", "--target-env=vulkan1.1", "-O"] {
        push(&mut input, "ARGV", arg.as_bytes());
    }
    // `-I<abs path>` is normalised: the *presence and role* of the include directory is part of
    // the argv, its absolute location on this machine is not.
    push(&mut input, "ARGV", b"-I<include>");
    for d in defines {
        push(&mut input, "ARGV", format!("-D{d}").as_bytes());
    }

    let root = normalize_shader_text(
        &fs::read(src).unwrap_or_else(|e| panic!("cannot read {}: {e}", src.display())),
    );
    push(&mut input, "SOURCE", &root);

    let mut seen: Vec<String> = Vec::new();
    let mut closure: Vec<(String, Vec<u8>)> = Vec::new();
    collect_include_closure(src, &root, glsl_dir, include_dir, &mut seen, &mut closure);
    // Sorted by resolved name so the digest does not depend on include order within a file, which
    // is a formatting choice rather than a subject change.
    closure.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, bytes) in &closure {
        push(&mut input, "INCLUDE-NAME", name.as_bytes());
        push(&mut input, "INCLUDE-BODY", bytes);
    }

    format!("{:016x}", fnv1a64(&input))
}

/// Follow `#include` recursively from `body`, accumulating `(name, contents)` for the closure.
///
/// Resolution mirrors `glslc`'s: the including file's own directory first, then `-I<include_dir>`,
/// then `shaders/glsl/`. An unresolved include contributes an `INCLUDE-UNRESOLVED` body so the
/// digest records that it could not read something, rather than recording nothing.
pub fn collect_include_closure(
    from: &Path,
    body: &[u8],
    glsl_dir: &Path,
    include_dir: &Path,
    seen: &mut Vec<String>,
    out: &mut Vec<(String, Vec<u8>)>,
) {
    let text = String::from_utf8_lossy(body).to_string();
    for line in text.lines() {
        let Some(name) = parse_include_directive(line) else {
            continue;
        };
        if seen.iter().any(|s| s == &name) {
            continue;
        }
        seen.push(name.clone());
        let candidates = [
            from.parent().map(|d| d.join(&name)),
            Some(include_dir.join(&name)),
            Some(glsl_dir.join(&name)),
        ];
        let resolved = candidates.into_iter().flatten().find(|p| p.is_file());
        match resolved {
            Some(path) => {
                println!("cargo:rerun-if-changed={}", path.display());
                let bytes = normalize_shader_text(
                    &fs::read(&path)
                        .unwrap_or_else(|e| panic!("cannot read include {}: {e}", path.display())),
                );
                out.push((name.clone(), bytes.clone()));
                collect_include_closure(&path, &bytes, glsl_dir, include_dir, seen, out);
            }
            None => out.push((name.clone(), b"\x01INCLUDE-UNRESOLVED".to_vec())),
        }
    }
}

/// Normalise a shader text body so its digest is a fact about the **source**, not the checkout.
///
/// Two things a version-control configuration decides are removed:
///
/// * a leading UTF-8 BOM, which some editors add and `glslc` ignores;
/// * `\r\n` and lone `\r`, folded to `\n`.
///
/// Neither reaches the compiler as a semantic difference — `glslc` emits byte-identical SPIR-V
/// from a CRLF and an LF copy of the same file — so a digest that moves on them is reporting a
/// difference the artifact does not have. That is strictly worse than the SPIR-V digest it was
/// added to be more portable than.
///
/// It is deliberately **not** a whitespace normaliser: trailing spaces, indentation and blank
/// lines stay in the digest, because those are edits a person made to the source and the point of
/// this witness is to see them.
pub fn normalize_shader_text(bytes: &[u8]) -> Vec<u8> {
    let body = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]).unwrap_or(bytes);
    let mut out = Vec::with_capacity(body.len());
    let mut i = 0;
    while i < body.len() {
        if body[i] == b'\r' {
            // CRLF and a lone CR both become one LF; a lone CR is a classic-Mac ending and
            // dropping it silently would make two different files hash the same.
            out.push(b'\n');
            if i + 1 < body.len() && body[i + 1] == b'\n' {
                i += 1;
            }
        } else {
            out.push(body[i]);
        }
        i += 1;
    }
    out
}

/// Extract the file name from `#include "x"` / `#include <x>`, or `None` for any other line.
pub fn parse_include_directive(line: &str) -> Option<String> {
    let rest = line.trim_start().strip_prefix('#')?.trim_start();
    let rest = rest.strip_prefix("include")?.trim_start();
    let (open, close) = match rest.chars().next()? {
        '"' => ('"', '"'),
        '<' => ('<', '>'),
        _ => return None,
    };
    let rest = rest.strip_prefix(open)?;
    let end = rest.find(close)?;
    let name = rest[..end].trim();
    if name.is_empty() {
        None
    } else {
        Some(name.to_string())
    }
}
