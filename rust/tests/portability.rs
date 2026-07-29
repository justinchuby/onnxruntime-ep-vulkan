//! **The portability lint.** The companion to `tests/layering.rs`, for a different failure class.
//!
//! # Why this exists
//!
//! On 2026-07-29 the Linux CI lane failed to compile — in a *test* file, mine:
//!
//! ```text
//! error[E0425]: cannot find type `wchar_t` in module `ort`
//!   tests/mock_ort/mod.rs:150  fn check_in_z_ortchar(p: *const ort::wchar_t, ...)
//! ```
//!
//! `ORTCHAR_T` is `wchar_t` on Windows and `char` everywhere else, so bindgen emits `ort::wchar_t`
//! only on Windows. The code compiled perfectly on the machine it was written on and could not
//! compile anywhere else, and it blocked the lane for a full CI cycle — during which no Vulkan
//! code on that lane ran at all, because this error masked everything behind it.
//!
//! The uncomfortable part is that `cargo ci` had *already* printed, in its own caveats:
//!
//! > *"Only this host's OS and toolchain. CI builds Linux and Windows; a `cfg(unix)` path that does
//! > not compile is invisible from a Windows machine and vice versa."*
//!
//! That was accurate and it was useless, because a caveat is a thing a person has to read and act
//! on. This file is the same statement as a mechanism.
//!
//! # Why not just cross-compile locally?
//!
//! Because it does not work on a Windows box, and I tried it rather than assuming. `rustup` has
//! the `x86_64-unknown-linux-gnu` and `aarch64-apple-darwin` std libraries installed here, and
//! `cargo check --target x86_64-unknown-linux-gnu` gets impressively far — every dependency
//! checks, and bindgen correctly re-targets clang. It then dies in `build.rs`:
//!
//! ```text
//! third_party/onnxruntime/include/onnxruntime_c_api.h:34:10: fatal error: 'stdlib.h' file not found
//! ```
//!
//! Clang's own builtin headers can be supplied with `BINDGEN_EXTRA_CLANG_ARGS` (that fixes
//! `stdbool.h`), but `stdlib.h` belongs to glibc, and getting it means vendoring or downloading a
//! Linux sysroot. That is infrastructure we would have to maintain, on every dev box, for one
//! lint's worth of value. Rejected — see `.squad/decisions/inbox/tank-m0-foundation.md` D-T20.
//!
//! So this lint attacks the specific, cheap, high-yield subset instead: **the ways our source can
//! name something that does not exist on another platform.** It runs in milliseconds, everywhere,
//! and needs no toolchain we do not already have.
//!
//! # The rules
//!
//! * **P1 — a platform-conditional binding may only be named by a `cfg`-gated definition.**
//!   `ort::wchar_t` exists only on Windows, so the only line allowed to name it is a
//!   `#[cfg(windows)]` item — in practice the `OrtChar` alias. Everything else uses the alias, and
//!   therefore compiles on both.
//! * **P2 — every platform fork has both arms in the same file.** A `#[cfg(windows)]` item with no
//!   `#[cfg(not(windows))]` sibling is a hole in the shape of the other platform.
//!
//! Run it locally with:
//!
//! ```text
//! cargo test --test portability
//! ```
//!
//! Like the layering lint, the scanner is itself tested against deliberately planted violations,
//! so a refactor that neuters it fails too.

use std::fs;
use std::path::{Path, PathBuf};

/// A binding that only exists on some platforms, and the gate it must live behind.
struct PlatformSymbol {
    /// The token as it appears in source.
    symbol: &'static str,
    /// The `cfg` attribute that must gate the definition naming it.
    gate: &'static str,
    /// What the portable spelling is, quoted back to whoever trips the lint.
    remedy: &'static str,
    /// Why the symbol is platform-conditional in the first place.
    why: &'static str,
}

/// Every ORT binding whose existence depends on the target.
///
/// This list is short because the ORT C API has exactly one platform-conditional type. If it
/// grows, it grows here — the point of a table is that adding an entry is cheaper than
/// rediscovering the rule from a red CI lane.
const PLATFORM_SYMBOLS: &[PlatformSymbol] = &[PlatformSymbol {
    symbol: "ort::wchar_t",
    gate: "#[cfg(windows)]",
    remedy: "use the `OrtChar` alias in `tests/mock_ort/mod.rs`, or define an equivalent \
             cfg-selected alias, so both arms exist",
    why: "`onnxruntime_c_api.h` defines `ORTCHAR_T` as `wchar_t` on _WIN32 and `char` otherwise, \
          so bindgen only emits `wchar_t` when targeting Windows",
}];

/// Directories scanned, relative to the crate root.
const ROOTS: &[&str] = &["src", "tests", "xtask/src"];

/// This file exempts itself, and only itself.
///
/// It has to name the forbidden symbols — once in the rule table, and again in the planted-
/// violation tests that prove the scanner works. It is also the one file in the crate that never
/// references the generated bindings at all (it reads source as text and imports nothing from
/// `onnxruntime_vulkan_ep`), so it cannot contain a real instance of the bug it looks for. Any
/// broader exemption list would be a way to smuggle violations past the lint; this one is a
/// single path, checked exactly.
const SELF_PATH: &str = "tests/portability.rs";

/// A violation, formatted for a human who has just seen a red test.
struct Finding {
    file: String,
    line: usize,
    text: String,
    message: String,
}

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn rust_files() -> Vec<PathBuf> {
    let root = crate_root();
    let mut out = Vec::new();
    let build_rs = root.join("build.rs");
    if build_rs.is_file() {
        out.push(build_rs);
    }
    for dir in ROOTS {
        collect(&root.join(dir), &mut out);
    }
    out.sort();
    out
}

fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

fn rel(path: &Path) -> String {
    path.strip_prefix(crate_root())
        .unwrap_or(path)
        .display()
        .to_string()
        .replace('\\', "/")
}

/// Is this line source code, as opposed to a comment or doc comment?
///
/// Deliberately crude: the crate uses `//` and `///` exclusively, and a lint that is easy to
/// reason about is worth more here than one that handles `/* */` nesting. A doc comment that
/// *mentions* a forbidden symbol — this file's own module docs do — must not trip the lint.
///
/// **Known limitation:** it does not understand string literals, so prose *inside* a string that
/// names a forbidden symbol trips the lint. That happened once, in `xtask`'s own caveat text, and
/// the fix was to reword the string rather than to teach the scanner about literals. A false
/// positive that costs one reword is cheaper than a parser, and lints that acquire escape hatches
/// stop being lints.
fn is_code(line: &str) -> bool {
    !line.trim_start().starts_with("//")
}

/// The contiguous run of attributes and doc comments immediately above `idx`.
fn attributes_above(lines: &[&str], idx: usize) -> Vec<String> {
    let mut attrs = Vec::new();
    for line in lines[..idx].iter().rev() {
        let t = line.trim();
        if t.is_empty() || t.starts_with("//") {
            continue;
        }
        if t.starts_with("#[") {
            attrs.push(t.to_string());
            continue;
        }
        break;
    }
    attrs
}

/// **P1** — every mention of a platform-conditional binding sits behind its gate.
fn scan_platform_symbols(file: &str, text: &str) -> Vec<Finding> {
    let lines: Vec<&str> = text.lines().collect();
    let mut findings = Vec::new();
    for (idx, line) in lines.iter().enumerate() {
        if !is_code(line) {
            continue;
        }
        for sym in PLATFORM_SYMBOLS {
            if !line.contains(sym.symbol) {
                continue;
            }
            if attributes_above(&lines, idx).iter().any(|a| a == sym.gate) {
                continue;
            }
            findings.push(Finding {
                file: file.to_string(),
                line: idx + 1,
                text: line.trim().to_string(),
                message: format!(
                    "`{}` is named without a `{}` gate on the item that defines it.\n      \
                     Why it matters: {}.\n      Remedy: {}",
                    sym.symbol, sym.gate, sym.why, sym.remedy
                ),
            });
        }
    }
    findings
}

/// **P2** — a `#[cfg(windows)]` item has a `#[cfg(not(windows))]` sibling in the same file.
fn scan_balanced_forks(file: &str, text: &str) -> Vec<Finding> {
    let lines: Vec<&str> = text.lines().collect();
    let count = |needle: &str| {
        lines
            .iter()
            .filter(|l| is_code(l) && l.trim() == needle)
            .count()
    };
    let windows = count("#[cfg(windows)]");
    let not_windows = count("#[cfg(not(windows))]");
    if windows == not_windows {
        return Vec::new();
    }
    let (more, less, missing) = if windows > not_windows {
        (windows, not_windows, "#[cfg(not(windows))]")
    } else {
        (not_windows, windows, "#[cfg(windows)]")
    };
    vec![Finding {
        file: file.to_string(),
        line: 0,
        text: format!("{windows} × #[cfg(windows)], {not_windows} × #[cfg(not(windows))]"),
        message: format!(
            "unbalanced platform fork: {more} item(s) on one side, {less} on the other.\n      \
             Every `#[cfg(windows)]` item needs a `{missing}` sibling, or the other platform has \
             a hole where this item should be — which is a compile error there and nowhere \
             else.\n      If the asymmetry is genuinely intended, give the missing arm an \
             explicit `compile_error!` or an empty stub so the intent is in the source.",
        ),
    }]
}

fn report(what: &str, findings: &[Finding]) {
    assert!(
        findings.is_empty(),
        "\n\nPORTABILITY LINT FAILED — {} violation(s) of {what}.\n\
         This is the class of bug that only shows up on a CI lane for another OS, hours later, \
         while blocking everything behind it.\n\n{}\n",
        findings.len(),
        findings
            .iter()
            .map(|f| {
                let where_ = if f.line == 0 {
                    f.file.clone()
                } else {
                    format!("{}:{}", f.file, f.line)
                };
                format!("  {where_}\n      {}\n      {}", f.text, f.message)
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    );
}

#[test]
fn platform_conditional_bindings_are_cfg_gated() {
    let mut findings = Vec::new();
    let mut saw_self = false;
    for path in rust_files() {
        let name = rel(&path);
        if name == SELF_PATH {
            saw_self = true;
            continue;
        }
        let Ok(text) = fs::read_to_string(&path) else {
            continue;
        };
        findings.extend(scan_platform_symbols(&name, &text));
    }
    assert!(
        saw_self,
        "the self-exemption path `{SELF_PATH}` did not match any scanned file, so the scanner's \
         path normalisation has drifted and the exemption may now be hiding a real file"
    );
    report(
        "rule P1 (platform-conditional bindings must be cfg-gated)",
        &findings,
    );
}

#[test]
fn platform_forks_have_both_arms() {
    let mut findings = Vec::new();
    for path in rust_files() {
        let Ok(text) = fs::read_to_string(&path) else {
            continue;
        };
        findings.extend(scan_balanced_forks(&rel(&path), &text));
    }
    report("rule P2 (every platform fork has both arms)", &findings);
}

/// Print the crate's entire platform-conditional surface.
///
/// Not an assertion — a review aid. The whole argument for this lint is that the cross-platform
/// surface should be small enough to read, so this prints it and makes any growth visible in the
/// test log rather than only in a diff.
#[test]
fn platform_fork_inventory_is_small_enough_to_review() {
    let mut total = 0usize;
    println!("\nPlatform-conditional surface of the crate:");
    for path in rust_files() {
        let Ok(text) = fs::read_to_string(&path) else {
            continue;
        };
        let hits: Vec<(usize, &str)> = text
            .lines()
            .enumerate()
            .filter(|(_, l)| {
                is_code(l) && l.trim().starts_with("#[cfg(") && {
                    let t = l.trim();
                    t.contains("windows") || t.contains("unix") || t.contains("target_os")
                }
            })
            .map(|(i, l)| (i + 1, l.trim()))
            .collect();
        if hits.is_empty() {
            continue;
        }
        println!("  {}", rel(&path));
        for (line, text) in &hits {
            println!("    {line:>5}  {text}");
        }
        total += hits.len();
    }
    println!("  total: {total} platform-conditional item(s)\n");

    // A soft ceiling, not a hard design rule: if the platform surface ever gets large enough that
    // nobody reads the list above, this fails and forces the conversation about consolidating it
    // behind aliases instead of spreading `cfg` through the crate.
    assert!(
        total <= 24,
        "the crate now has {total} platform-conditional items, which is more than a reviewer will \
         actually read. Consolidate them behind cfg-selected aliases (see `OrtChar`) rather than \
         forking logic at each use site, then raise this ceiling deliberately."
    );
}

// ---------------------------------------------------------------------------------------------
// The lint is itself tested, against the exact code that broke the Linux lane.
// ---------------------------------------------------------------------------------------------

#[test]
fn detects_the_ungated_binding_that_broke_the_linux_lane() {
    // Verbatim from `tests/mock_ort/mod.rs` before the fix.
    let planted =
        "fn check_in_z_ortchar(p: *const ort::wchar_t, who: &str) -> String {\n    todo!()\n}\n";
    let findings = scan_platform_symbols("planted.rs", planted);
    assert_eq!(
        findings.len(),
        1,
        "the scanner failed to catch the original Linux-lane compile error; it is not doing its job"
    );
    assert!(findings[0].message.contains("ort::wchar_t"));
}

#[test]
fn accepts_the_gated_alias_that_fixed_it() {
    let fixed = "/// `ORTCHAR_T`, as the platform defines it.\n#[cfg(windows)]\npub type OrtChar = ort::wchar_t;\n#[cfg(not(windows))]\npub type OrtChar = c_char;\n";
    assert!(
        scan_platform_symbols("fixed.rs", fixed).is_empty(),
        "the scanner rejected the correct cfg-gated alias, so it would block the fix as well as \
         the bug"
    );
    assert!(scan_balanced_forks("fixed.rs", fixed).is_empty());
}

#[test]
fn ignores_mentions_in_comments_and_docs() {
    let docs = "//! `ort::wchar_t` does not exist on Linux.\n/// See `ort::wchar_t`.\n// ort::wchar_t\nfn f() {}\n";
    assert!(
        scan_platform_symbols("docs.rs", docs).is_empty(),
        "a lint that cannot be written about is a lint nobody documents"
    );
}

#[test]
fn detects_a_planted_one_armed_fork() {
    let planted = "#[cfg(windows)]\nfn only_on_windows() {}\n";
    let findings = scan_balanced_forks("planted.rs", planted);
    assert_eq!(findings.len(), 1);
    assert!(findings[0].message.contains("#[cfg(not(windows))]"));

    let both = "#[cfg(windows)]\nfn f() {}\n#[cfg(not(windows))]\nfn f() {}\n";
    assert!(scan_balanced_forks("ok.rs", both).is_empty());
}
