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

/// Locate `glslc`: `$VULKAN_SDK/bin/glslc` first, then bare `glslc` on `$PATH`.
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
    Command::new("glslc")
        .arg("--version")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|_| PathBuf::from("glslc"))
}

fn compile_shaders(out_dir: &Path) {
    let manifest = PathBuf::from(
        env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is always set by cargo"),
    );
    let glsl_dir = manifest.join("shaders").join("glsl");
    let include_dir = manifest.join("shaders").join("include");
    let spv_dir = out_dir.join("spv");

    let mut sources: Vec<PathBuf> = Vec::new();
    if glsl_dir.is_dir() {
        let entries = fs::read_dir(&glsl_dir)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", glsl_dir.display()));
        for entry in entries {
            let path = entry.expect("cannot read a shaders/glsl directory entry").path();
            if path.is_file() && path.extension() == Some(OsStr::new("comp")) {
                println!("cargo:rerun-if-changed={}", path.display());
                sources.push(path);
            }
        }
    }
    // Deterministic order → deterministic generated module → reproducible builds.
    sources.sort();

    if include_dir.is_dir() {
        println!("cargo:rerun-if-changed={}", include_dir.display());
    }

    // Nothing to compile (the M0 state, before Switch lands the first `.comp`): emit an empty
    // module and do not require a Vulkan SDK on the build machine.
    if sources.is_empty() {
        write_shader_modules(out_dir, &[]);
        return;
    }

    let glslc = match find_glslc() {
        Some(g) => g,
        None if env::var(ENV_ALLOW_MISSING_GLSLC).as_deref() == Ok("1") => {
            println!(
                "cargo:warning=glslc not found and {ENV_ALLOW_MISSING_GLSLC}=1: {} shader(s) were \
                 NOT compiled. The resulting artifact cannot create any compute pipeline and must \
                 not be shipped.",
                sources.len()
            );
            write_shader_modules(out_dir, &[]);
            return;
        }
        None => panic!(
            "glslc not found but {} shader(s) exist in {}. Install the Vulkan SDK (glslc is at \
             $VULKAN_SDK/bin/glslc) or put glslc on PATH. Set {ENV_ALLOW_MISSING_GLSLC}=1 to build \
             a shader-less artifact for lint-only lanes.",
            sources.len(),
            glsl_dir.display()
        ),
    };

    fs::create_dir_all(&spv_dir)
        .unwrap_or_else(|e| panic!("cannot create {}: {e}", spv_dir.display()));

    let mut compiled: Vec<(String, PathBuf)> = Vec::new();
    for src in &sources {
        let stem = src
            .file_stem()
            .and_then(OsStr::to_str)
            .unwrap_or_else(|| panic!("shader path {} has no usable stem", src.display()))
            .to_string();
        let spv = spv_dir.join(format!("{stem}.spv"));

        let mut cmd = Command::new(&glslc);
        cmd.arg("-fshader-stage=compute")
            .arg("--target-env=vulkan1.1") // DESIGN.md §7.4: default SPIR-V target.
            .arg("-O")
            .arg(format!("-I{}", include_dir.display()))
            .arg("-o")
            .arg(&spv)
            .arg(src);

        let out = cmd
            .output()
            .unwrap_or_else(|e| panic!("failed to run {}: {e}", glslc.display()));
        if !out.status.success() {
            panic!(
                "glslc failed for {}:\n{}\n{}",
                src.display(),
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
        }
        compiled.push((stem, spv));
    }

    write_shader_modules(out_dir, &compiled);
}

/// Emit `$OUT_DIR/shader_modules.rs`: one `pub const <STEM_UPPER>_SPV: &[u8]` per shader plus a
/// name→bytes lookup table the engine uses for variant selection.
fn write_shader_modules(out_dir: &Path, compiled: &[(String, PathBuf)]) {
    let mut src = String::new();
    src.push_str(
        "// @generated by build.rs — do not edit.\n\
         // One entry per shaders/glsl/*.comp, compiled to SPIR-V at build time and embedded in\n\
         // the cdylib. There is no runtime shader compiler in the deployed artifact.\n\n",
    );

    for (stem, spv) in compiled {
        let ident = stem.to_uppercase().replace(['-', '.'], "_");
        // `include_bytes!` paths in a generated file are relative to that file, i.e. $OUT_DIR.
        // Use the absolute path with forward slashes so it is valid on Windows too.
        let path = spv.display().to_string().replace('\\', "/");
        src.push_str(&format!(
            "pub const {ident}_SPV: &[u8] = include_bytes!(\"{path}\");\n"
        ));
    }

    src.push_str("\n/// Every embedded SPIR-V module, keyed by its shader stem.\n");
    src.push_str(&format!(
        "pub static SHADER_MODULES: [(&str, &[u8]); {}] = [\n",
        compiled.len()
    ));
    for (stem, _) in compiled {
        let ident = stem.to_uppercase().replace(['-', '.'], "_");
        src.push_str(&format!("    (\"{stem}\", {ident}_SPV),\n"));
    }
    src.push_str("];\n");

    fs::write(out_dir.join("shader_modules.rs"), src)
        .expect("failed to write $OUT_DIR/shader_modules.rs");
}
