//! `cargo ci` — run locally exactly what CI's Rust lanes run, in the same order, and say plainly
//! what was *not* checked.
//!
//! # Why this exists
//!
//! On 2026-07-28 CI was red for four consecutive runs and nobody noticed. Every agent ran
//! `cargo build`, `cargo clippy` and `cargo test`, saw green, and reported green. The failing
//! check was `cargo fmt --check`, which was in CI and in nobody's local loop. That is a
//! verification gap, not bad luck: "green" meant "the three commands I happen to remember
//! passed", and there was no single artefact that knew what CI actually runs.
//!
//! So the contract of this binary is narrow and specific: **it is the list**. If CI gains a check,
//! it is added here in the same commit, and `cargo ci` stays the one thing to run before
//! reporting work complete.
//!
//! # What it deliberately does not do
//!
//! It does not try to *be* CI. It cannot be, and pretending otherwise is how a local green light
//! becomes a lie:
//!
//! * **No shader has ever executed on any device** (`DESIGN.md` §9.1.2). Nothing here changes
//!   that. Every check below is host-side Rust logic.
//! * On a machine without the Vulkan SDK the shaders are not even *compiled* — `build.rs` needs
//!   `glslc`. This binary detects that, sets the documented escape hatch so the build can
//!   proceed, and reports the reduced coverage rather than hiding it.
//! * It runs no pytest lane, no lavapipe lane, no dispatch, and no Linux/macOS *build*. It does
//!   offer `--cross`, which **compiles** the workspace (tests included) for
//!   `x86_64-unknown-linux-gnu` and is the only local check that catches a Linux-only type error
//!   — see `rust/ci/linux-stub-include/README.md` and issue #39. It *does* run the
//!   validation-layer positive control (`tests/validation_control.rs`), which is the only thing
//!   here that creates a `VkInstance`.
//!
//! Every one of those absences is printed at the end of a successful run. A developer who reads
//! the output cannot come away believing more was verified than was.

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

/// `build.rs`'s escape hatch for a machine with no shader compiler.
const ENV_ALLOW_MISSING_GLSLC: &str = "ONNXRUNTIME_EP_VULKAN_ALLOW_MISSING_GLSLC";

/// One CI check: what to call it, and the `cargo` arguments that run it.
struct Check {
    name: &'static str,
    /// The CI step this mirrors, so a reader can find it in `.github/workflows/ci.yml`.
    mirrors: &'static str,
    args: &'static [&'static str],
}

/// The checks, in CI's order — cheapest and most frequently broken first.
///
/// `fmt` is first on purpose. It is the fastest check and it is the one that was silently red for
/// four runs; putting it in front means the failure that actually happened is the one you see
/// first, in two seconds rather than after a three-minute build.
///
/// The layering lint has no entry of its own because it is an integration test (`tests/layering.rs`)
/// and therefore already inside `cargo test`. That was a deliberate choice when it was written —
/// a check that runs as part of something you already run cannot be forgotten. `cargo test` also
/// covers `tests/portability.rs` (the cross-platform lint) and `tests/dump_capabilities.rs`, which
/// is C2's surface.
const CHECKS: &[Check] = &[
    Check {
        name: "rustfmt",
        mirrors: "job `format` — cargo fmt --all -- --check",
        args: &["fmt", "--all", "--", "--check"],
    },
    Check {
        name: "clippy",
        mirrors: "job `build-test-{linux,windows}` — Clippy (all warnings as errors)",
        // `--workspace` is deliberately stricter than CI: it also lints this xtask, so the
        // tool that tells you CI will be green cannot itself be the thing that is dirty.
        args: &[
            "clippy",
            "--workspace",
            "--all-targets",
            "--",
            "-D",
            "warnings",
        ],
    },
    Check {
        name: "build",
        mirrors: "job `build-test-{linux,windows}` — Build Vulkan EP",
        args: &["build"],
    },
    Check {
        name: "test (incl. layering + portability lints + capability dump)",
        mirrors: "job `build-test-linux` — Layering lint, plus the crate's own test suite",
        args: &["test"],
    },
];

fn crate_root() -> PathBuf {
    // `xtask` lives at `rust/xtask`, so the crate under test is one level up. Derived rather than
    // assumed from the current directory, so `cargo ci` works from anywhere in the tree.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask must live inside the crate directory")
        .to_path_buf()
}

/// The one target `cargo ci --cross` checks.
///
/// Windows and Linux are the two lanes CI builds, so from either host there is exactly one
/// *other* platform worth proving compiles. Only the Windows→Linux direction is offered: the
/// reverse needs the MSVC CRT headers, which are not redistributable and cannot be stubbed.
const CROSS_TARGET: &str = "x86_64-unknown-linux-gnu";

/// Where the parse-only C library declarations live, relative to the crate root.
const CROSS_STUB_INCLUDE: &str = "ci/linux-stub-include";

/// Clang's own builtin header directory (`stddef.h`, `stdint.h`, `stdbool.h`).
///
/// bindgen re-targets clang correctly on its own, but when it does, clang stops finding its
/// builtin headers by the path it would have used for the host. Asking the driver is the only
/// answer that stays right across LLVM versions and installation layouts; the LIBCLANG_PATH walk
/// is the fallback for a machine that ships `libclang` without the `clang` driver.
fn clang_resource_include(libclang: Option<&Path>) -> Option<PathBuf> {
    let from_driver = Command::new("clang")
        .arg("-print-resource-dir")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| PathBuf::from(String::from_utf8_lossy(&o.stdout).trim().to_string()))
        .map(|p| p.join("include"))
        .filter(|p| p.is_dir());
    if from_driver.is_some() {
        return from_driver;
    }
    // `<llvm>/bin` or `<llvm>/lib` -> `<llvm>/lib/clang/<version>/include`.
    let lib_root = libclang
        .or_else(|| Some(Path::new(r"C:\Program Files\LLVM\bin")))?
        .parent()?
        .join("lib")
        .join("clang");
    let mut versions: Vec<PathBuf> = std::fs::read_dir(lib_root)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path().join("include"))
        .filter(|p| p.is_dir())
        .collect();
    versions.sort();
    versions.pop()
}

fn cross_target_installed() -> Result<bool, String> {
    let out = Command::new("rustup")
        .args(["target", "list", "--installed"])
        .output()
        .map_err(|e| format!("could not run rustup: {e}"))?;
    if !out.status.success() {
        return Err("`rustup target list --installed` failed".to_string());
    }
    Ok(String::from_utf8_lossy(&out.stdout)
        .lines()
        .any(|l| l.trim() == CROSS_TARGET))
}

/// **The decisive cross-platform check.** Compile the whole workspace, tests included, for the
/// other CI lane's target.
///
/// # Why this is not just the portability lint again
///
/// `tests/portability.rs` reads source as text. It cannot see the defect it was written about
/// twice over: the lines that broke the Linux lane on 2026-08-06 (issue #39) never *name* the
/// alias whose width they assume —
///
/// ```text
/// Some(get) => unsafe { get(status) },   // typed by ort::OrtErrorCode, which is not written here
/// None => -1,                            // `u32: Neg` is not satisfied, on Linux only
/// ```
///
/// A scanner cannot catch that and a compiler cannot miss it. So this runs the compiler.
///
/// # Why it is opt-in rather than part of the default run
///
/// It needs `rustup target add x86_64-unknown-linux-gnu` (~100 MB of std) and a working
/// `bindgen`. Both are one-time and both are stated plainly when absent — the failure mode this
/// avoids is a check that quietly does nothing on a machine missing a prerequisite, which is how
/// a green light becomes a lie. It refuses rather than skips.
fn run_cross_check(env: &Env) -> bool {
    println!("\n=== cross-compile check ({CROSS_TARGET}) ===");
    println!("    mirrors CI: job `build-test-linux` — the *compile* half of it, from this host");

    match cross_target_installed() {
        Ok(true) => {}
        Ok(false) => {
            eprintln!(
                "    REFUSED: the `{CROSS_TARGET}` std is not installed.\n    \
                 Run `rustup target add {CROSS_TARGET}` (one time, ~100 MB) and re-run.\n    \
                 Not skipping: a cross-check that silently does nothing is worse than no \
                 cross-check, because it reports the same green as one that ran."
            );
            return false;
        }
        Err(e) => {
            eprintln!("    REFUSED: {e}");
            return false;
        }
    }

    let stubs = env.root.join(CROSS_STUB_INCLUDE);
    if !stubs.join("stdlib.h").is_file() {
        eprintln!(
            "    REFUSED: {} does not contain the parse-only C headers this check needs.",
            stubs.display()
        );
        return false;
    }
    let Some(builtin) = clang_resource_include(env.libclang.as_deref()) else {
        eprintln!(
            "    REFUSED: could not locate clang's builtin header directory (stddef.h, \
             stdint.h).\n    Put `clang` on PATH or set LIBCLANG_PATH to an LLVM installation."
        );
        return false;
    };

    // Retargeted clang needs to be told where the C declarations are, because the host's are the
    // wrong ABI and there is no Linux sysroot here. `-ffreestanding` keeps it from assuming a
    // hosted environment it does not have.
    let clang_args = format!(
        "-ffreestanding -isystem \"{}\" -isystem \"{}\"",
        builtin.display(),
        stubs.display()
    );
    let args = [
        "check",
        "--workspace",
        "--all-targets",
        "--target",
        CROSS_TARGET,
    ];
    println!("    cargo {}", args.join(" "));
    println!("    BINDGEN_EXTRA_CLANG_ARGS={clang_args}");

    let mut cmd = Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".into()));
    cmd.current_dir(&env.root)
        .args(args)
        .env("BINDGEN_EXTRA_CLANG_ARGS", &clang_args);
    if let Some(p) = &env.libclang {
        cmd.env("LIBCLANG_PATH", p);
    }
    if env.glslc.is_none() {
        cmd.env(ENV_ALLOW_MISSING_GLSLC, "1");
    }
    cmd.env_remove("RUSTFLAGS");
    match cmd.status() {
        Ok(s) if s.success() => true,
        Ok(_) => false,
        Err(e) => {
            eprintln!("    could not run cargo: {e}");
            false
        }
    }
}

/// Is a shader compiler available? Mirrors `build.rs::find_glslc`.
fn find_glslc() -> Option<PathBuf> {
    let exe = if cfg!(windows) { "glslc.exe" } else { "glslc" };
    if let Ok(sdk) = std::env::var("VULKAN_SDK") {
        let candidate = Path::new(&sdk).join("bin").join(exe);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
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

/// Mirrors `build.rs::installed_sdk_glslc`. The LunarG Windows installer neither sets
/// `VULKAN_SDK` machine-wide nor puts `glslc` on `$PATH`, so an SDK can be installed and still
/// invisible. Reporting "NOT FOUND" in that case makes `cargo ci` build with zero shaders and then
/// fail the live-row tests that CI passes — a false red, which is worse than a false green because
/// it teaches people to ignore the tool.
fn installed_sdk_glslc() -> Option<PathBuf> {
    if !cfg!(windows) {
        return None;
    }
    let mut versions: Vec<PathBuf> = std::fs::read_dir("C:\\VulkanSDK")
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.join("Bin").join("glslc.exe").is_file())
        .collect();
    versions.sort();
    versions.pop().map(|p| p.join("Bin").join("glslc.exe"))
}

/// Does `libclang` need pointing at? `bindgen` needs it and it is not on `PATH` by default on
/// Windows. Failing here with cargo's raw error is a bad first experience for a new agent, so we
/// look in the usual place and say what we did.
fn default_libclang() -> Option<PathBuf> {
    if std::env::var_os("LIBCLANG_PATH").is_some() {
        return None;
    }
    let guesses: &[&str] = if cfg!(windows) {
        &[r"C:\Program Files\LLVM\bin"]
    } else {
        &["/usr/lib/llvm-18/lib", "/usr/lib/llvm-17/lib", "/usr/lib"]
    };
    guesses
        .iter()
        .map(PathBuf::from)
        .find(|p| p.is_dir() && std::fs::read_dir(p).is_ok_and(|mut d| d.any(|e| is_libclang(&e))))
}

fn is_libclang(entry: &std::io::Result<std::fs::DirEntry>) -> bool {
    entry.as_ref().is_ok_and(|e| {
        e.file_name()
            .to_string_lossy()
            .to_ascii_lowercase()
            .starts_with("libclang")
    })
}

struct Env {
    root: PathBuf,
    libclang: Option<PathBuf>,
    /// `None` when no shader compiler was found — shaders will not be compiled.
    glslc: Option<PathBuf>,
}

/// The crate's edition, read from `Cargo.toml` rather than hard-coded.
fn crate_edition(root: &Path) -> Option<String> {
    let manifest = std::fs::read_to_string(root.join("Cargo.toml")).ok()?;
    manifest.lines().find_map(|line| {
        let rest = line.trim().strip_prefix("edition")?;
        let rest = rest.trim_start().strip_prefix('=')?;
        Some(rest.trim().trim_matches('"').to_string())
    })
}

/// Refuse to run if this toolchain's `rustfmt` cannot parse the crate's edition.
///
/// **Why this exists.** `rustfmt --edition 2021` run against an edition-2024 crate does not fail —
/// it parses fewer constructs and quietly formats *nothing it does not understand*, so it reports
/// success while leaving the file unformatted. CI then runs the correct edition, finds a diff, and
/// goes red with no local reproduction. Niobe hit this; the cost of each of us discovering it
/// individually is the wrong distribution of that cost, so it is a hard preflight failure here.
///
/// `cargo fmt` itself passes `--edition` from the manifest, so the *normal* path is already
/// correct. What this guards is the case underneath it: a toolchain whose `rustfmt` predates the
/// edition, where `cargo fmt` would pass `--edition 2024` to a rustfmt that rejects or ignores it.
fn check_rustfmt_edition(env: &Env) -> Result<String, String> {
    let Some(edition) = crate_edition(&env.root) else {
        return Err("could not read `edition` from Cargo.toml".to_string());
    };
    let out = Command::new("rustfmt")
        .args(["--edition", &edition, "--version"])
        .output()
        .map_err(|e| format!("could not run rustfmt: {e}"))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(format!(
            "this toolchain's rustfmt does not accept `--edition {edition}`, which is the \
             edition in Cargo.toml.\n           rustfmt said: {}\n           \
             A rustfmt that does not understand the crate's edition does not fail on the code it \
             cannot parse — it silently leaves it unformatted and reports success, which is \
             exactly how CI goes red with no local reproduction. Update the toolchain \
             (`rustup update`) rather than working around this.",
            stderr.trim()
        ));
    }
    Ok(edition)
}

fn run_check(check: &Check, env: &Env, fix_fmt: bool, release: bool) -> bool {
    // `--fix` rewrites rather than reports, which is what you want the *first* time and never in
    // CI. Only `rustfmt` supports it; every other check stays read-only.
    let args: Vec<&str> = if fix_fmt && check.name == "rustfmt" {
        vec!["fmt", "--all"]
    } else {
        let mut a = check.args.to_vec();
        // CI builds and tests `--release`. Optimisation-dependent bugs are rare but real, and the
        // first crash this plugin ever had was found in a release CI build, so the option exists.
        if release && check.name != "rustfmt" {
            a.insert(1, "--release");
        }
        a
    };

    println!("\n=== {} ===", check.name);
    println!("    mirrors CI: {}", check.mirrors);
    println!("    cargo {}", args.join(" "));

    let mut cmd = Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".into()));
    cmd.current_dir(&env.root).args(&args);
    if let Some(p) = &env.libclang {
        cmd.env("LIBCLANG_PATH", p);
    }
    if env.glslc.is_none() {
        cmd.env(ENV_ALLOW_MISSING_GLSLC, "1");
    }
    // Do not inherit an outer `cargo ci` invocation's job server or target dir assumptions.
    cmd.env_remove("RUSTFLAGS");

    match cmd.status() {
        Ok(s) if s.success() => true,
        Ok(_) => false,
        Err(e) => {
            eprintln!("    could not run cargo: {e}");
            false
        }
    }
}

fn print_caveats(env: &Env, cross_ran: bool) {
    println!();
    println!("─────────────────────────────────────────────────────────────────────────────");
    println!("WHAT THIS DID *NOT* VERIFY — read this before reporting work complete");
    println!("─────────────────────────────────────────────────────────────────────────────");
    println!(
        "  * No shader has executed. DESIGN.md §9.1.2: no GLSL in this repository has ever run\n\
         \x20   on any device, real or software. Everything above is host-side Rust logic —\n\
         \x20   claim predicates, translation, layering, FFI shape. Correctness of any kernel is\n\
         \x20   entirely unverified."
    );
    if env.glslc.is_none() {
        println!(
            "  * No shader was even COMPILED. `glslc` was not found, so {ENV_ALLOW_MISSING_GLSLC}=1\n\
             \x20   was set for you and build.rs emitted empty shader constants. A GLSL syntax\n\
             \x20   error would NOT have been caught here — CI's Linux and Windows lanes are the\n\
             \x20   first thing that compiles them. Install the Vulkan SDK to close this locally."
        );
    } else {
        println!("  * Shaders were compiled to SPIR-V, but not run and not validated on a device.");
    }
    println!(
        "  * No Vulkan *device* was touched: no lavapipe lane, no dispatch, no queue submit.\n\
         \x20   `tests/validation_control.rs` is the one exception to \"no vkCreateInstance\": it\n\
         \x20   creates an instance with the validation layer and a debug messenger, plants a\n\
         \x20   deliberate violation, and fails if the layer does not catch it. That establishes\n\
         \x20   the *check* works on this machine — it does not establish that the EP's own\n\
         \x20   dispatch path is validation-clean, which needs a real session. On a machine with\n\
         \x20   no layer installed those tests SKIP loudly; set\n\
         \x20   ONNXRUNTIME_EP_VULKAN_REQUIRE_VALIDATION=1 to make the skip a failure."
    );
    println!(
        "  * No real ONNX Runtime. `tests/cdylib_load.rs` does load the shipped library and drive\n\
         \x20   registration, but against a *mock* host that checks the contracts the headers\n\
         \x20   document. \"The plugin loads\" here is not the same claim as \"ORT can load it\"."
    );
    println!(
        "  * No Python lane: `tests/ops` (op-correctness vs the ORT CPU oracle, barrier parity,\n\
         \x20   claim diagnostics, no-ICD fallback) was not run. It needs a real ONNX Runtime."
    );
    println!(
        "  * ZERO DISPATCHES EXECUTED, and that is expected here. The execution counters this\n\
         \x20   suite exercises are checked for shape, not for a non-zero value: nothing in\n\
         \x20   `cargo ci` runs a claimed node on a device. M0 criterion 8 is satisfied only by a\n\
         \x20   lane that reports a non-zero `dispatches_executed` — see\n\
         \x20   `epctl --check-counters`. A green `cargo ci` and an executed dispatch are\n\
         \x20   unrelated claims; do not report the second on the strength of the first."
    );
    if cross_ran {
        println!(
            "  * THE OTHER LANE'S TARGET WAS COMPILED, NOT RUN. `--cross` proves {CROSS_TARGET}\n\
             \x20   builds; it executes nothing, so every Linux runtime behaviour — path\n\
             \x20   handling, `dlopen` search order, the ELF loader — is still claimed only by\n\
             \x20   CI's Linux lane."
        );
    } else {
        println!(
            "  * ONLY THIS HOST'S TARGET WAS COMPILED, and this caveat has come true twice.\n\
             \x20   `tests/mock_ort/mod.rs` named a bindgen type that only exists on Windows and\n\
             \x20   broke the Linux lane for a full cycle; then `modelrunner/src/ortapi.rs`\n\
             \x20   negated a C enum that is unsigned under GCC and signed under MSVC, and broke\n\
             \x20   it again (issue #39). `tests/portability.rs` lints the cheap subset of that\n\
             \x20   class, but a lint reads text and could not see either inferred type.\n\
             \x20   Run `cargo ci --cross`: it compiles this workspace for {CROSS_TARGET} in\n\
             \x20   about a minute and needs no sysroot."
        );
    }
    println!(
        "\n  `cargo ci` green means: CI's *Rust* lanes should pass. It does not mean the EP works."
    );
    warn_if_decisions_are_stranded_in_a_worktree();
}

/// Warn when decision records exist in a *linked* worktree's inbox.
///
/// `.squad/decisions/inbox/` is gitignored by design — the Scribe merges it into `decisions.md` on
/// the integration tree. That is fine while everyone works in one checkout, and became a trap the
/// day we moved to per-agent worktrees: a record written in a worktree is invisible to `main`, to
/// the Scribe, and to every other agent, and nothing reports it. One was found stranded that way
/// within hours of the switch.
///
/// It is the same shape as this file's other caveats, and worth naming as a category: **isolation
/// that is good for code is bad for shared state, and the failure is silent.** Worktrees made edit
/// collisions impossible and quietly made lost reasoning possible. The fix for a silent failure is
/// not care, it is a noise.
///
/// This is a warning and not a failure on purpose: the check is heuristic (it infers "linked
/// worktree" from a `.git` *file* rather than a directory), and a lint that can be wrong must not
/// be able to fail a build, or people learn to ignore it — which is the failure mode we are
/// already fighting.
fn warn_if_decisions_are_stranded_in_a_worktree() {
    let Some(root) = repo_root() else {
        return;
    };
    // In a linked worktree `.git` is a file containing `gitdir: …`; in the main checkout it is a
    // directory. Only the former can strand a record.
    if root.join(".git").is_dir() {
        return;
    }
    let inbox = root.join(".squad").join("decisions").join("inbox");
    let Ok(entries) = std::fs::read_dir(&inbox) else {
        return;
    };
    let stranded: Vec<String> = entries
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "md"))
        .filter_map(|e| e.file_name().into_string().ok())
        .collect();
    if stranded.is_empty() {
        return;
    }
    println!(
        "\n  ! DECISION RECORDS ARE STRANDED IN THIS WORKTREE. `.squad/decisions/inbox/` is\n\
         \x20   gitignored, so these files will never reach `main`, the Scribe, or anyone else:"
    );
    for f in &stranded {
        println!("      - {f}");
    }
    println!(
        "\x20   Write decision records into the integration tree's inbox instead. Nothing in git\n\
         \x20   will tell you about this, which is why `cargo ci` does."
    );
}

/// The repository (or worktree) root, by walking up from the manifest directory.
fn repo_root() -> Option<std::path::PathBuf> {
    let mut dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    loop {
        if dir.join(".git").exists() {
            return Some(dir);
        }
        if !dir.pop() {
            return None;
        }
    }
}

fn usage() {
    eprintln!("cargo ci — run what CI runs, locally");
    eprintln!();
    eprintln!("USAGE:");
    eprintln!("    cargo ci            check everything (this is the pre-report command)");
    eprintln!("    cargo ci --fix      same, but let rustfmt rewrite instead of reporting");
    eprintln!("    cargo ci --list     print the checks and the CI steps they mirror");
    eprintln!("    cargo ci --release  build and test optimised, as CI does (slower)");
    eprintln!(
        "    cargo ci --cross    also compile the workspace for {CROSS_TARGET}, which is the"
    );
    eprintln!(
        "                        only local check that catches a Linux-only type error \
         (issue #39)."
    );
    eprintln!("                        Needs `rustup target add {CROSS_TARGET}` once.");
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let fix = args.iter().any(|a| a == "--fix");
    let list = args.iter().any(|a| a == "--list");
    let release = args.iter().any(|a| a == "--release");
    let cross = args.iter().any(|a| a == "--cross");

    if let Some(bad) = args
        .iter()
        .find(|a| !matches!(a.as_str(), "--fix" | "--list" | "--release" | "--cross"))
    {
        eprintln!("cargo ci: unrecognised argument `{bad}`");
        usage();
        return ExitCode::from(2);
    }

    if list {
        for c in CHECKS {
            println!("{:<44} {}", c.name, c.mirrors);
        }
        return ExitCode::SUCCESS;
    }

    let env = Env {
        root: crate_root(),
        libclang: default_libclang(),
        glslc: find_glslc(),
    };

    println!("cargo ci — CI parity check for onnxruntime-ep-vulkan");
    println!("crate: {}", env.root.display());
    if let Some(p) = &env.libclang {
        println!(
            "LIBCLANG_PATH not set; using {} (bindgen needs it)",
            p.display()
        );
    }
    match &env.glslc {
        Some(g) => println!("glslc: {}", g.display()),
        None => println!(
            "glslc: NOT FOUND — setting {ENV_ALLOW_MISSING_GLSLC}=1; shaders will not be compiled"
        ),
    }
    match check_rustfmt_edition(&env) {
        Ok(edition) => println!("rustfmt: accepts --edition {edition} (matches Cargo.toml)"),
        Err(msg) => {
            eprintln!();
            eprintln!("cargo ci: PREFLIGHT FAILED — {msg}");
            eprintln!();
            eprintln!(
                "Refusing to run rather than reporting a green formatting check that CI will \
                 contradict."
            );
            return ExitCode::from(2);
        }
    }

    let mut failed: Vec<&str> = Vec::new();
    for check in CHECKS {
        if !run_check(check, &env, fix, release) {
            failed.push(check.name);
            // Keep going. Stopping at the first failure means an agent fixes one thing, re-runs,
            // waits three minutes, and finds the next — which is exactly the slow loop that made
            // people stop running the checks in the first place.
        }
    }
    if cross && !run_cross_check(&env) {
        failed.push("cross-compile");
    }

    println!();
    if failed.is_empty() {
        println!("ALL CHECKS PASSED");
        print_caveats(&env, cross);
        if !cross {
            println!(
                "\n  ! NO OTHER PLATFORM WAS COMPILED. `cargo ci --cross` checks the workspace \
                 for\n\x20   {CROSS_TARGET}, which is the only local check that catches a \
                 Linux-only\n\x20   type error before CI does. Two red lanes have been caused by \
                 skipping it."
            );
        }
        ExitCode::SUCCESS
    } else {
        println!("FAILED: {}", failed.join(", "));
        println!("CI will be red. Fix these before reporting work complete.");
        if failed.contains(&"rustfmt") {
            println!("  (for rustfmt: `cargo ci --fix` rewrites the files for you)");
        }
        ExitCode::FAILURE
    }
}
