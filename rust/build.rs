//! Build script for `onnxruntime-ep-vulkan`.
//!
//! Two jobs, both of which produce code into `OUT_DIR`:
//!
//! 1. **ORT plugin-EP C ABI bindings** — `bindgen` over the headers vendored at
//!    `third_party/onnxruntime/include` (see `third_party/onnxruntime/PROVENANCE.md`). Output:
//!    `$OUT_DIR/ort.rs`, included by `src/sys.rs`.
//!
//! 2. **GLSL → SPIR-V** — every `shaders/glsl/*.comp` is compiled by `glslc` to
//!    `$OUT_DIR/spv/<stem>.spv`, and `$OUT_DIR/shader_modules.rs` is generated with one
//!    `include_bytes!` constant per shader. The compiled cdylib is self-contained: there is no
//!    runtime shader compiler in the deployed artifact (ENGINE.md §4.2, decisions.md).
//!
//! We never link `libonnxruntime`. ORT is reached only through the `OrtApi` function-pointer
//! table handed to `CreateEpFactories`, so this build script emits no ORT link directives.
//! Vulkan is loaded dynamically by `ash::Entry::load()` at runtime, so it emits no Vulkan link
//! directives either — a build machine needs neither an ORT install nor a Vulkan SDK unless it is
//! compiling shaders.

use std::env;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Env var that points at an alternative ORT C-API include directory (must contain
/// `onnxruntime_c_api.h`). Overrides the vendored headers.
const ENV_ORT_INCLUDE_DIR: &str = "ORT_INCLUDE_DIR";
/// Env var pointing at an ORT release root; `$ORT_HOME/include` is used.
const ENV_ORT_HOME: &str = "ORT_HOME";
/// Set to `1` to allow a build with no `glslc` available *when there are shaders to compile*.
/// Shader constants are then emitted as empty and the engine will refuse to create pipelines.
/// Intended for doc/lint-only lanes, never for a shipped artifact.
const ENV_ALLOW_MISSING_GLSLC: &str = "ONNXRUNTIME_EP_VULKAN_ALLOW_MISSING_GLSLC";

fn main() {
    println!("cargo:rerun-if-changed=wrapper_ort.h");
    println!("cargo:rerun-if-changed=shaders/glsl");
    println!("cargo:rerun-if-env-changed={ENV_ORT_INCLUDE_DIR}");
    println!("cargo:rerun-if-env-changed={ENV_ORT_HOME}");
    println!("cargo:rerun-if-env-changed={ENV_ALLOW_MISSING_GLSLC}");
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is always set by cargo"));

    generate_ort_bindings(&out_dir);
    compile_shaders(&out_dir);
}

// ---------------------------------------------------------------------------------------------
// 1. ORT plugin-EP C ABI bindings
// ---------------------------------------------------------------------------------------------

/// Resolve the ONNX Runtime C-API include directory.
///
/// Resolution order:
///   1. `ORT_INCLUDE_DIR` — explicit include dir (highest precedence).
///   2. `$ORT_HOME/include` — the layout of an ORT release tarball.
///   3. `third_party/onnxruntime/include` — the vendored, version-pinned copy (the default, and
///      what CI uses).
///
/// Whichever wins, the presence of `onnxruntime_c_api.h` is verified here so a bad path fails
/// with a clear message instead of somewhere deep inside libclang.
fn resolve_ort_include() -> PathBuf {
    let manifest = PathBuf::from(
        env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is always set by cargo"),
    );
    let vendored = manifest
        .parent()
        .expect("rust/ always has a parent (the repo root)")
        .join("third_party")
        .join("onnxruntime")
        .join("include");

    let (dir, source) = match env::var(ENV_ORT_INCLUDE_DIR) {
        Ok(d) if !d.is_empty() => (PathBuf::from(d), ENV_ORT_INCLUDE_DIR),
        _ => match env::var(ENV_ORT_HOME) {
            Ok(h) if !h.is_empty() => (PathBuf::from(h).join("include"), ENV_ORT_HOME),
            _ => (vendored, "vendored third_party/onnxruntime/include"),
        },
    };

    let probe = dir.join("onnxruntime_c_api.h");
    if !probe.is_file() {
        panic!(
            "ONNX Runtime headers not found: '{}' does not contain onnxruntime_c_api.h \
             (resolved from {source}). Either restore third_party/onnxruntime/include (see \
             third_party/onnxruntime/PROVENANCE.md) or point {ENV_ORT_INCLUDE_DIR} at a valid \
             ORT C-API include directory.",
            dir.display()
        );
    }
    for header in [
        "onnxruntime_c_api.h",
        "onnxruntime_ep_c_api.h",
        "onnxruntime_error_code.h",
    ] {
        println!("cargo:rerun-if-changed={}", dir.join(header).display());
    }
    dir
}

fn generate_ort_bindings(out_dir: &Path) {
    let include = resolve_ort_include();

    let bindings = bindgen::Builder::default()
        .header("wrapper_ort.h")
        .clang_arg(format!("-I{}", include.display()))
        // The ORT headers are C, not C++. Saying so keeps bindgen from mangling anything and
        // keeps every generated struct plain `#[repr(C)]`.
        .clang_arg("-xc")
        .allowlist_type("Ort.*")
        .allowlist_type("ONNX.*")
        .allowlist_function("Ort.*")
        .allowlist_var("ORT_.*")
        // Every generated function-pointer call site is wrapped in `unsafe { }` by bindgen so the
        // crate can compile under Rust 2024's `unsafe_op_in_unsafe_fn`.
        .wrap_unsafe_ops(true)
        // Deriving these on the ABI structs makes the binding-shape assertions in
        // `tests/sys_abi.rs` (and future ones) cheap to write.
        .derive_debug(true)
        .derive_default(true)
        .generate_comments(false)
        .layout_tests(false)
        .generate()
        .unwrap_or_else(|e| {
            panic!(
                "bindgen failed over the ORT headers at {}: {e}. bindgen needs libclang; install \
                 LLVM and/or set LIBCLANG_PATH (see rust/README.md).",
                include.display()
            )
        });

    bindings
        .write_to_file(out_dir.join("ort.rs"))
        .expect("failed to write $OUT_DIR/ort.rs");
}

// ---------------------------------------------------------------------------------------------
// 2. GLSL → SPIR-V
// ---------------------------------------------------------------------------------------------

/// Locate `glslc`: `$VULKAN_SDK/bin/glslc` first, then bare `glslc` on `$PATH`, then an installed
/// SDK in its default location.
fn find_glslc() -> Option<PathBuf> {
    if let Ok(sdk) = env::var("VULKAN_SDK") {
        let exe = if cfg!(windows) { "glslc.exe" } else { "glslc" };
        let candidate = PathBuf::from(sdk).join("bin").join(exe);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    // Probe `$PATH` by actually running it — cheaper and more accurate than re-implementing PATH
    // resolution, and it also proves the binary is executable on this machine.
    let on_path = Command::new("glslc")
        .arg("--version")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|_| PathBuf::from("glslc"));
    if on_path.is_some() {
        return on_path;
    }
    installed_sdk_glslc()
}

/// The LunarG Windows installer does not put `glslc` on `$PATH` and does not set `VULKAN_SDK`
/// machine-wide; it installs to `C:\VulkanSDK\<version>\Bin`. Without this, a developer box with
/// the SDK *installed* still builds with zero shaders, and the local test suite then fails tests
/// that CI passes — the exact parity gap `cargo ci` exists to close.
///
/// Highest version wins, by lexicographic order on the directory name. That is not a correct
/// version sort in general, but SDK directory names are zero-padded four-part numbers of equal
/// shape, so it agrees with the numeric order in practice. Set `VULKAN_SDK` to override.
fn installed_sdk_glslc() -> Option<PathBuf> {
    if !cfg!(windows) {
        return None;
    }
    let root = Path::new("C:\\VulkanSDK");
    let mut versions: Vec<PathBuf> = fs::read_dir(root)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.join("Bin").join("glslc.exe").is_file())
        .collect();
    versions.sort();
    versions
        .pop()
        .map(|p| p.join("Bin").join("glslc.exe"))
        .inspect(|p| {
            println!(
                "cargo:warning=using glslc from an installed Vulkan SDK at {} \
                 (VULKAN_SDK was unset and glslc was not on PATH)",
                p.display()
            );
        })
}

fn compile_shaders(out_dir: &Path) {
    let manifest = PathBuf::from(
        env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is always set by cargo"),
    );
    let glsl_dir = manifest.join("shaders").join("glsl");
    let include_dir = manifest.join("shaders").join("include");
    let spv_dir = out_dir.join("spv");
    let variant_table = manifest.join("src").join("ops").join("shader_variants.txt");

    // ── Collect direct .comp sources (hand-written XL kernels, utils, etc.) ────────────
    let mut direct_sources: Vec<PathBuf> = Vec::new();
    if glsl_dir.is_dir() {
        let entries = fs::read_dir(&glsl_dir)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", glsl_dir.display()));
        for entry in entries {
            let path = entry
                .expect("cannot read a shaders/glsl directory entry")
                .path();
            if path.is_file() && path.extension() == Some(OsStr::new("comp")) {
                println!("cargo:rerun-if-changed={}", path.display());
                direct_sources.push(path);
            }
        }
    }
    direct_sources.sort();

    // ── Collect variant rows from shader_variants.txt (Seam 3) ───────────────────────
    // Format per row: <stem>\t<glsl_source>\t<comma-separated -D defines>
    // The stem is the output module name; glsl_source is a path relative to shaders/glsl/.
    println!("cargo:rerun-if-changed={}", variant_table.display());
    let variants: Vec<VariantRow> = if variant_table.is_file() {
        parse_shader_variants(&variant_table)
    } else {
        Vec::new()
    };

    if include_dir.is_dir() {
        println!("cargo:rerun-if-changed={}", include_dir.display());
    }

    // Nothing to compile: emit an empty module.
    if direct_sources.is_empty() && variants.is_empty() {
        write_shader_modules(out_dir, &[]);
        write_shader_toolchain(out_dir, None);
        return;
    }

    let glslc = match find_glslc() {
        Some(g) => g,
        None if env::var(ENV_ALLOW_MISSING_GLSLC).as_deref() == Ok("1") => {
            println!(
                "cargo:warning=glslc not found and {ENV_ALLOW_MISSING_GLSLC}=1: {} direct + {} \
                 variant shader(s) were NOT compiled. The resulting artifact cannot create any \
                 compute pipeline and must not be shipped.",
                direct_sources.len(),
                variants.len()
            );
            write_shader_modules(out_dir, &[]);
            write_shader_toolchain(out_dir, None);
            return;
        }
        None => {
            let total = direct_sources.len() + variants.len();
            panic!(
                "glslc not found but {total} shader(s) exist. Install the Vulkan SDK or put glslc \
                 on PATH. Set {ENV_ALLOW_MISSING_GLSLC}=1 to build a shader-less artifact for \
                 lint-only lanes."
            );
        }
    };

    fs::create_dir_all(&spv_dir)
        .unwrap_or_else(|e| panic!("cannot create {}: {e}", spv_dir.display()));

    let mut compiled: Vec<CompiledModule> = Vec::new();

    // ── Compile direct .comp sources ─────────────────────────────────────────────────
    for src in &direct_sources {
        let stem = src
            .file_stem()
            .and_then(OsStr::to_str)
            .unwrap_or_else(|| panic!("shader path {} has no usable stem", src.display()))
            .to_string();
        let spv = spv_dir.join(format!("{stem}.spv"));

        let mut cmd = Command::new(&glslc);
        cmd.arg("-fshader-stage=compute")
            .arg("--target-env=vulkan1.1")
            .arg("-O")
            .arg(format!("-I{}", include_dir.display()))
            .arg("-o")
            .arg(&spv)
            .arg(src);

        run_glslc(&glslc, cmd, src.display().to_string().as_str());
        let source_digest = source_digest_for(
            &stem,
            src,
            src.file_name().and_then(OsStr::to_str).unwrap_or(""),
            &[],
            &glsl_dir,
            &include_dir,
        );
        compiled.push(CompiledModule {
            stem,
            spv,
            source_digest,
        });
    }

    // ── Compile variant rows ─────────────────────────────────────────────────────────
    for row in &variants {
        let src = glsl_dir.join(&row.glsl_source);
        if !src.is_file() {
            panic!(
                "shader_variants.txt row '{}': source '{}' not found at '{}'",
                row.stem,
                row.glsl_source,
                src.display()
            );
        }
        let spv = spv_dir.join(format!("{}.spv", row.stem));

        let mut cmd = Command::new(&glslc);
        cmd.arg("-fshader-stage=compute")
            .arg("--target-env=vulkan1.1")
            .arg("-O")
            .arg(format!("-I{}", include_dir.display()));
        for define in &row.defines {
            cmd.arg(format!("-D{define}"));
        }
        cmd.arg("-o").arg(&spv).arg(&src);

        run_glslc(&glslc, cmd, &format!("{}@{}", row.glsl_source, row.stem));
        let source_digest = source_digest_for(
            &row.stem,
            &src,
            &row.glsl_source,
            &row.defines,
            &glsl_dir,
            &include_dir,
        );
        compiled.push(CompiledModule {
            stem: row.stem.clone(),
            spv,
            source_digest,
        });
    }

    // Deterministic order → reproducible builds.
    compiled.sort_by(|a, b| a.stem.cmp(&b.stem));

    write_shader_modules(out_dir, &compiled);
    write_shader_toolchain(out_dir, Some(&glslc));
}

/// The identity of the shader compiler, as a build-time constant (§8.9.19 part 3 item 4).
///
/// **This is a FRAME component, never a KEY component.** `glslc --version` is a fact about the
/// machine that produced the SPIR-V, not about the form that was proven; §8.9.19 part 1 rules
/// that putting it in the key turns "I have a proof that does not apply here" into "I have no
/// proof", and only the first of those is actionable.
///
/// The whole multi-line banner is collapsed to one line, because the second line of shaderc's
/// banner is a target-env list that moves for reasons that are not a compiler change. `UNKNOWN`
/// if `glslc --version` cannot be read at all — an unreadable version is a different fact from a
/// version of `""`, and a comparison against `""` would silently succeed against another
/// unreadable one.
///
/// **`UNKNOWN` is now warned about at build time and gated at test time.** Link's first fresh
/// Linux `.so` reported its own toolchain as `UNKNOWN` while embedding a full set of compiled
/// modules, so the one field that distinguishes "a second compiler" from "a rewritten kernel"
/// was absent in the only situation that needs it — and nothing said so. `stderr` is read as well
/// as `stdout`, and a non-zero exit no longer discards a banner that was printed anyway, because
/// both were guesses about how a version banner escapes and neither is worth a silent `UNKNOWN`.
fn write_shader_toolchain(out_dir: &Path, glslc: Option<&Path>) {
    let text = glslc
        .and_then(|g| Command::new(g).arg("--version").output().ok())
        .and_then(|o| {
            // Some builds print the banner to stderr, and some exit non-zero after printing it.
            // Take whichever stream carries a first non-empty line rather than requiring both a
            // clean exit and a particular stream.
            let stdout = String::from_utf8_lossy(&o.stdout).to_string();
            let stderr = String::from_utf8_lossy(&o.stderr).to_string();
            [stdout, stderr]
                .iter()
                .filter_map(|s| s.lines().map(str::trim).find(|l| !l.is_empty()))
                .map(str::to_string)
                .next()
        })
        .unwrap_or_else(|| "UNKNOWN".to_string());
    if text == "UNKNOWN" && glslc.is_some() {
        println!(
            "cargo:warning=this build compiled shaders but cannot name its shader toolchain \
             (`glslc --version` produced nothing readable). Every proof taken here records \
             toolchain=UNKNOWN, and an UNKNOWN frame cannot be told from a second compiler — so \
             a ledger entry proven on another machine will read as a changed kernel rather than \
             as a toolchain delta."
        );
    }
    let escaped = text.replace('\\', "\\\\").replace('"', "\\\"");
    let src = format!(
        "// @generated by build.rs — do not edit.\n\
         /// `glslc --version`, first non-empty line, from the build that produced this artifact.\n\
         pub const SHADER_TOOLCHAIN: &str = \"{escaped}\";\n"
    );
    fs::write(out_dir.join("shader_toolchain.rs"), src)
        .expect("failed to write $OUT_DIR/shader_toolchain.rs");
}

/// One compiled SPIR-V module and the two witnesses §8.9.19 requires of it.
struct CompiledModule {
    /// Output module name and the key in the shader-modules table.
    stem: String,
    /// Where the compiled SPIR-V landed.
    spv: PathBuf,
    /// FNV-1a/64 over this module's **source closure** — see [`source_digest_for`].
    source_digest: String,
}

/// FNV-1a/64, matching `registry::fnv1a64` and `rust/tools/gen_proof_ledger.py`.
fn fnv1a64(bytes: &[u8]) -> u64 {
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
fn source_digest_for(
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
fn collect_include_closure(
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
fn normalize_shader_text(bytes: &[u8]) -> Vec<u8> {
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
fn parse_include_directive(line: &str) -> Option<String> {
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

/// One parsed row from `shader_variants.txt`.
struct VariantRow {
    /// Output SPIR-V module stem (also the name key in the shader-modules table).
    stem: String,
    /// Source `.comp` filename relative to `shaders/glsl/`.
    glsl_source: String,
    /// `-D` defines, e.g. `["EW_OP=OP_ADD", "SCALAR_T=float", "DTYPE_F32"]`.
    defines: Vec<String>,
}

/// Parse `shader_variants.txt`.
///
/// Format per non-blank, non-comment line:
/// ```
/// <stem>\t<glsl_source>\t<comma-separated defines>
/// ```
/// Blank lines and lines starting with `#` are ignored. Panics if any data line is malformed.
fn parse_shader_variants(path: &Path) -> Vec<VariantRow> {
    let content =
        fs::read_to_string(path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let mut rows = Vec::new();
    for (lineno, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.splitn(3, '\t').collect();
        if parts.len() < 2 {
            panic!(
                "{}:{}: malformed row (expected at least 2 tab-separated fields): '{line}'",
                path.display(),
                lineno + 1
            );
        }
        let stem = parts[0].trim().to_string();
        let glsl_source = parts[1].trim().to_string();
        let defines = if parts.len() >= 3 && !parts[2].trim().is_empty() {
            parts[2]
                .split(',')
                .map(|d| d.trim().to_string())
                .filter(|d| !d.is_empty())
                .collect()
        } else {
            Vec::new()
        };
        if stem.is_empty() || glsl_source.is_empty() {
            panic!(
                "{}:{}: stem or glsl_source is empty: '{line}'",
                path.display(),
                lineno + 1
            );
        }
        rows.push(VariantRow {
            stem,
            glsl_source,
            defines,
        });
    }
    rows
}

/// Run a glslc command, panicking with human-readable output on failure.
fn run_glslc(glslc: &Path, cmd: Command, source_label: &str) {
    let mut cmd = cmd;
    let out = cmd
        .output()
        .unwrap_or_else(|e| panic!("failed to run {}: {e}", glslc.display()));
    if !out.status.success() {
        panic!(
            "glslc failed for {}:\n{}\n{}",
            source_label,
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

/// Emit `$OUT_DIR/shader_modules.rs`: one `pub const <STEM_UPPER>_SPV: &[u8]` per shader plus a
/// name→bytes lookup table the engine uses for variant selection.
fn write_shader_modules(out_dir: &Path, compiled: &[CompiledModule]) {
    let mut src = String::new();
    src.push_str(
        "// @generated by build.rs — do not edit.\n\
         // One entry per shaders/glsl/*.comp, compiled to SPIR-V at build time and embedded in\n\
         // the cdylib. There is no runtime shader compiler in the deployed artifact.\n\n",
    );

    for m in compiled {
        let ident = m.stem.to_uppercase().replace(['-', '.'], "_");
        // `include_bytes!` paths in a generated file are relative to that file, i.e. $OUT_DIR.
        // Use the absolute path with forward slashes so it is valid on Windows too.
        let path = m.spv.display().to_string().replace('\\', "/");
        src.push_str(&format!(
            "pub const {ident}_SPV: &[u8] = include_bytes!(\"{path}\");\n"
        ));
    }

    src.push_str("\n/// Every embedded SPIR-V module, keyed by its shader stem.\n");
    src.push_str(&format!(
        "pub static SHADER_MODULES: [(&str, &[u8]); {}] = [\n",
        compiled.len()
    ));
    for m in compiled {
        let ident = m.stem.to_uppercase().replace(['-', '.'], "_");
        src.push_str(&format!("    (\"{}\", {ident}_SPV),\n", m.stem));
    }
    src.push_str("];\n");

    src.push_str(
        "\n/// Every embedded module's **source-closure** digest, keyed by stem (§8.9.19).\n\
         ///\n\
         /// Toolchain-independent by construction: two machines with different `glslc` versions\n\
         /// compiling the same checkout produce different `SHADER_MODULES` bytes and *identical*\n\
         /// values here. That is exactly what lets a Linux run tell \"different compiler\" from\n\
         /// \"different kernel\".\n",
    );
    src.push_str(&format!(
        "pub static SHADER_SOURCE_DIGESTS: [(&str, &str); {}] = [\n",
        compiled.len()
    ));
    for m in compiled {
        src.push_str(&format!("    (\"{}\", \"{}\"),\n", m.stem, m.source_digest));
    }
    src.push_str("];\n");

    fs::write(out_dir.join("shader_modules.rs"), src)
        .expect("failed to write $OUT_DIR/shader_modules.rs");
}
