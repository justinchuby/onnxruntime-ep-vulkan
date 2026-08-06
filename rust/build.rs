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

/// The §8.9.19 part 2 source-closure digest and its helpers.
///
/// Extracted verbatim so a test target can include the same file — see the module docs.
/// `cargo test -p onnxruntime-ep-vulkan --test shader_source_digest` is the control.
#[path = "build_support/shader_source_digest.rs"]
mod shader_source_digest;

use shader_source_digest::source_digest_for;

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
