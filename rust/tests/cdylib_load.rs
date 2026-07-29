//! Drive the mock ONNX Runtime host against the plugin **loaded as a shared library**.
//!
//! # Why this exists in addition to `host_registration.rs`
//!
//! ONNX Runtime does not link us. It `LoadLibraryEx`/`dlopen`s the file we ship and resolves
//! `CreateEpFactories` and `ReleaseEpFactory` **by name**. Everything between "the crate compiles"
//! and "ORT can call us" — the `cdylib` crate-type, the `#[unsafe(no_mangle)]` on the exports,
//! whether the built file has an unresolvable dependent library — is invisible to a test that
//! links the rlib. On 2026-07-29 a real ORT died inside registration while 268 local tests were
//! green, and none of them had ever loaded the file.
//!
//! So this test is the rlib driver's twin, one step further out: same mock host, same scenario,
//! but reached through the artifact we actually ship.
//!
//! # What it still cannot prove
//!
//! It is not ONNX Runtime. It checks that we honour the contracts ORT's headers document, not
//! that ORT's implementation is happy with us, and it never executes a shader or touches a
//! Vulkan device. CI's Python lane remains the only thing that proves a real ORT can drive us.

mod mock_ort;

use std::path::{Path, PathBuf};

use mock_ort::{CreateEpFactoriesFn, LogProbe, ReleaseEpFactoryFn, run_registration_scenario};

/// The plugin's file name for this platform, exactly as it lands in `target/<profile>/`.
fn library_file_name() -> &'static str {
    if cfg!(windows) {
        "onnxruntime_vulkan_ep.dll"
    } else if cfg!(target_os = "macos") {
        "libonnxruntime_vulkan_ep.dylib"
    } else {
        "libonnxruntime_vulkan_ep.so"
    }
}

/// Locate the built cdylib relative to this test binary.
///
/// Integration tests are linked into `target/<profile>/deps/`, and the cdylib is hard-linked one
/// directory up in `target/<profile>/`. Deriving it from `current_exe` rather than from
/// `CARGO_MANIFEST_DIR` means the test follows `--release`, `--target`, and a redirected
/// `CARGO_TARGET_DIR` without being told about any of them.
fn locate_library() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let deps: &Path = exe.parent()?;
    let candidates = [deps.join(library_file_name()), {
        let profile = deps.parent()?;
        profile.join(library_file_name())
    }];
    candidates.into_iter().find(|p| p.is_file())
}

#[test]
fn ort_can_load_the_shipped_library_and_resolve_its_entry_points() {
    // The driver cannot inject a record into the loaded library's private copy of the `log` crate,
    // so raise the plugin's own level instead: at `Info` it logs a "loaded" line at the end of
    // `CreateEpFactories`, after the ORT logger is attached, which forces exactly the round trip
    // that access-violated in CI.
    //
    // SAFETY: `set_var` requires that no other thread is concurrently reading the environment.
    // This is the only test in this binary, so the harness has not started any other test thread,
    // and the plugin reads the variable later, on this thread, inside `CreateEpFactories`.
    unsafe { std::env::set_var(onnxruntime_vulkan_ep::logging::ENV_VERBOSE, "1") };

    let Some(path) = locate_library() else {
        panic!(
            "could not find `{}` next to {}.\n\
             The cdylib is what ONNX Runtime actually loads, so this test needs it built:\n\
             run `cargo build` (or `cargo ci`, which builds before it tests) first.",
            library_file_name(),
            std::env::current_exe()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|_| "this test binary".into()),
        );
    };
    eprintln!("[cdylib-load] loading {}", path.display());

    // SAFETY: loading a library runs its initialisers, which is inherently unsafe; this one is our
    // own build output and has no static initialisers beyond the Rust runtime's. On Windows a
    // missing dependent DLL surfaces here as an `Err`, which is one of the faults this test is for.
    let lib = unsafe { libloading::Library::new(&path) }.unwrap_or_else(|e| {
        panic!(
            "the shipped library at {} could not be loaded: {e}\n\
             This is the failure ONNX Runtime reports as a registration error. On Windows it is \
             usually an unresolvable dependent DLL; elsewhere a missing SONAME.",
            path.display()
        )
    });

    // SAFETY: the symbol names are the exported entry points ORT resolves by name, and the
    // signatures are transcribed from `lib.rs`. A name mismatch surfaces as an `Err`, not UB.
    let (create, release) = unsafe {
        let create: libloading::Symbol<CreateEpFactoriesFn> =
            lib.get(b"CreateEpFactories\0").unwrap_or_else(|e| {
                panic!(
                    "`CreateEpFactories` is not exported from {}: {e}\n\
                     ORT resolves this symbol by name; without it registration fails outright.",
                    path.display()
                )
            });
        let release: libloading::Symbol<ReleaseEpFactoryFn> =
            lib.get(b"ReleaseEpFactory\0").unwrap_or_else(|e| {
                panic!(
                    "`ReleaseEpFactory` is not exported from {}: {e}\n\
                     Without it ORT leaks the factory on unregister.",
                    path.display()
                )
            });
        (*create, *release)
    };

    // SAFETY: `create`/`release` point into `lib`, which is leaked below and therefore stays
    // mapped for the rest of the process — the same lifetime ORT gives a registered EP library.
    unsafe { run_registration_scenario(create, release, LogProbe::Foreign) };

    // ORT does not unload an EP library while the process lives, and neither do we: the plugin has
    // installed a `log` logger and leaked process-lifetime statics inside its own image, so
    // unmapping it here would be less faithful, not more.
    std::mem::forget(lib);
}
