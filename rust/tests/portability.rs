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
//! **This section was wrong, and being wrong cost a second red lane. It is kept, corrected, as
//! the record of what changed.**
//!
//! The original reasoning: `cargo check --target x86_64-unknown-linux-gnu` gets impressively far
//! on a Windows box — every dependency checks, and bindgen correctly re-targets clang — and then
//! dies in `build.rs`:
//!
//! ```text
//! third_party/onnxruntime/include/onnxruntime_c_api.h:34:10: fatal error: 'stdlib.h' file not found
//! ```
//!
//! Clang's own builtin headers can be supplied with `BINDGEN_EXTRA_CLANG_ARGS` (that fixes
//! `stdbool.h`), but `stdlib.h` belongs to glibc — so, the argument went, getting it means
//! vendoring or downloading a Linux sysroot, which is infrastructure we would have to maintain on
//! every dev box for one lint's worth of value. Rejected — see
//! `.squad/decisions/inbox/tank-m0-foundation.md` D-T20.
//!
//! On 2026-08-06 that conclusion cost the Linux lane again: `modelrunner/src/ortapi.rs` merged
//! with two Linux-only type errors (issue #39). Re-testing the premise rather than re-reading it
//! showed it was false. The ORT headers do not *use* glibc; they include three C library headers
//! and then declare their own types. `rust/ci/linux-stub-include` is those three headers as
//! declarations — 40 lines, no download, no sysroot — and with them
//! `cargo check --workspace --all-targets --target x86_64-unknown-linux-gnu` completes and
//! reproduces the real errors exactly. It is wired up as `cargo ci --cross`.
//!
//! **That is now the decisive check, and this lint is the cheap one.** The distinction matters,
//! because this scanner *structurally cannot* catch what broke the lane in 2026-08: the offending
//! lines never name the alias whose width they assume.
//!
//! ```text
//! Some(get) => unsafe { get(status) },   // typed by ort::OrtErrorCode -- not written here
//! None => -1,                            // `u32: Neg` is not satisfied, on Linux only
//! ```
//!
//! P3 fires on a *carrier* that spells a width in a scope that names the alias. Both of those
//! conditions were absent. So the honest division of labour is: this lint runs in milliseconds on
//! every `cargo test` and catches the carrier shape; `cargo ci --cross` runs the real front end
//! and catches everything; CI's Linux lane is the backstop that also *runs* the result.
//!
//! # The rules
//!
//! * **P1 — a platform-conditional binding may only be named by a `cfg`-gated definition.**
//!   `ort::wchar_t` exists only on Windows, so the only line allowed to name it is a
//!   `#[cfg(windows)]` item — in practice the `OrtChar` alias. Everything else uses the alias, and
//!   therefore compiles on both.
//! * **P2 — every platform fork has both arms in the same file.** A `#[cfg(windows)]` item with no
//!   `#[cfg(not(windows))]` sibling is a hole in the shape of the other platform.
//! * **P3 — a value of a width-varying ABI alias is carried as the alias, never as a spelled
//!   width.** P1's failure is *the name does not exist over there*. P3's is worse, because the
//!   name exists on both platforms and only its **width** differs: `ort::OrtLoggingLevel` is
//!   `c_int` under MSVC and `c_uint` under GCC, since a C enum whose values are all non-negative
//!   is signed for MSVC and unsigned for GCC. Code that spells `i32` where the alias belongs
//!   compiles and passes on the machine it was written on, and is 11 type errors on the other —
//!   which is exactly what happened on 2026-08-02 in `src/ep.rs`, blocking every Linux step
//!   behind it. Note that the remedy is the alias and **not** an `as` cast: a cast makes the
//!   spelled width compile on both platforms while keeping the assumption that produced it.
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

/// An ABI alias that exists on every platform but whose *width* is target-dependent.
struct WidthVaryingSymbol {
    /// The alias as it appears in source.
    alias: &'static str,
    /// The prefix of the generated constants belonging to it. A module that names one of these is
    /// a module that handles values of this alias.
    const_prefix: &'static str,
    /// The spellings that must not appear in such a module. Deliberately only the widths this
    /// alias can actually take: `c_int` / `c_uint` are the ABI's own spellings and are legitimate
    /// in a callback signature, so they are not listed.
    widths: &'static [&'static str],
    /// Why the width varies.
    why: &'static str,
    /// The portable spelling, quoted back to whoever trips the lint.
    remedy: &'static str,
}

/// Every ORT binding whose *width* depends on the target — and, deliberately, not all of them.
///
/// **The measurement.** Diffing the two generated `ort.rs` files (Windows MSVC against
/// `x86_64-unknown-linux-gnu`, both produced via `cargo ci --cross`) shows **25 of the 28** `ort`
/// enum aliases change signedness across the two targets. Only `OrtAllocatorType`, `OrtMemType`
/// and `OrtDeviceEpIncompatibilityReason` are `c_int` on both, because each carries a negative
/// enumerator and so is signed everywhere.
///
/// **Why the table lists one of the 25 and not all of them.** P3's trigger is "this scope names
/// a constant of the alias", which is a weak signal that gets weaker as the table grows. Adding
/// `ort::OrtErrorCode` was tried and reverted: it fires on `src/sys.rs`'s
/// `fn at_version(version: u32)` — an ORT *API version*, correctly `u32` on every target — merely
/// because the same test module elsewhere names `ort::OrtErrorCode_ORT_EP_FAIL`. This file's own
/// rule is that "a lint that reports unrelated integers trains people to ignore it", and that
/// applies to this lint too.
///
/// **What covers the other 24.** `cargo ci --cross`, which compiles the workspace for the other
/// lane's target and has no false positives because it is not a heuristic. That check is also the
/// only one that could have caught issue #39, whose two defective lines never named an alias at
/// all — so growing this table would have bought nothing there and cost a false positive.
/// `tests::the_narrow_p3_table_is_a_decision_and_not_an_oversight` pins that reasoning so a
/// future reader does not "fix" it by widening the table again.
const WIDTH_VARYING_SYMBOLS: &[WidthVaryingSymbol] = &[WidthVaryingSymbol {
    alias: "ort::OrtLoggingLevel",
    const_prefix: "ort::OrtLoggingLevel_",
    widths: &["i32", "u32"],
    why: "bindgen emits `OrtLoggingLevel` as `::std::os::raw::c_int` under MSVC and as \
          `::std::os::raw::c_uint` under GCC — verified by diffing both generated `ort.rs` files \
          — because `OrtLoggingLevel`'s values are 0..=4 and a C enum with no negative enumerator \
          is signed for MSVC and unsigned for GCC. The values are identical on both; only the \
          binding's type differs",
    remedy: "declare the carrier as `ort::OrtLoggingLevel` (the alias), not as a spelled width, \
             and do not reach for an `as` cast — a cast compiles on both platforms while keeping \
             the assumption that the width is knowable here. To *read* such a value, convert it \
             into `i64`, which both widths do losslessly (see `modelrunner/src/ortapi.rs::widen` \
             and `src/counters.rs`)",
}];

/// Directories scanned, relative to the crate root.
///
/// `modelrunner/` is a sibling crate rather than a module of this one, and its absence from this
/// list is why the lint said nothing while `modelrunner/src/ortapi.rs` merged with two Linux-only
/// type errors (issue #39). The lint is a property of the *workspace*'s source text, not of this
/// crate's, so the roots follow the source and not the crate boundary.
const ROOTS: &[&str] = &[
    "src",
    "tests",
    "xtask/src",
    "modelrunner/src",
    "modelrunner/tests",
];

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

/// The gate spellings that mean "this item exists on Windows".
///
/// `#[cfg(windows)]` and `#[cfg(target_os = "windows")]` are the same predicate written two ways,
/// and counting only the first is how `modelrunner/`'s forks were invisible to P2. Both are
/// listed rather than pattern-matched loosely, so a reader can see exactly what the rule covers.
const WINDOWS_POSITIVE_GATES: &[&str] = &["#[cfg(windows)]", "#[cfg(target_os = \"windows\")]"];

/// The gate spellings that mean "this item exists everywhere else".
///
/// The `not(any(...))` form is how a three-way fork (Windows / macOS / the rest) spells its
/// Windows-excluding arm, and it counts: what P2 is really asking is whether every Windows-only
/// item has something on the other side, not whether the two spellings are literal mirrors.
const WINDOWS_NEGATIVE_GATES: &[&str] = &[
    "#[cfg(not(windows))]",
    "#[cfg(not(target_os = \"windows\"))]",
];

fn is_windows_negative_gate(line: &str) -> bool {
    let t = line.trim();
    WINDOWS_NEGATIVE_GATES.contains(&t)
        || (t.starts_with("#[cfg(not(any(") && t.contains("windows"))
}

/// **P2** — a Windows-only item has a sibling for everywhere else, in the same file.
fn scan_balanced_forks(file: &str, text: &str) -> Vec<Finding> {
    let lines: Vec<&str> = text.lines().collect();
    let code: Vec<&&str> = lines.iter().filter(|l| is_code(l)).collect();
    let windows = code
        .iter()
        .filter(|l| WINDOWS_POSITIVE_GATES.contains(&l.trim()))
        .count();
    let not_windows = code.iter().filter(|l| is_windows_negative_gate(l)).count();
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
        text: format!("{windows} × Windows-only gate, {not_windows} × everywhere-else gate"),
        message: format!(
            "unbalanced platform fork: {more} item(s) on one side, {less} on the other.\n      \
             Every Windows-only item needs a `{missing}` sibling (or a \
             `#[cfg(not(any(target_os = \"windows\", ...)))]` arm), or the other platform has \
             a hole where this item should be — which is a compile error there and nowhere \
             else.\n      If the asymmetry is genuinely intended, give the missing arm an \
             explicit `compile_error!` or an empty stub so the intent is in the source.",
        ),
    }]
}

/// The half-open line span of one `mod name { ... }` block, keyed by its indentation.
///
/// Found by indentation rather than by counting braces: the crate is rustfmt-formatted, so a
/// module opened at indent *N* closes at the first later line that is exactly *N* spaces and a
/// `}`. Brace counting would have to understand string literals — `format!("{d:?}")` and the many
/// wrapped message strings in this crate — and a scanner that miscounts scopes reports violations
/// in the wrong module, which is worse than reporting none.
struct ModuleSpan {
    name: String,
    /// 0-based, the line after `mod name {`.
    start: usize,
    /// 0-based, exclusive: the closing brace's line.
    end: usize,
}

fn module_spans(text: &str) -> Vec<ModuleSpan> {
    let lines: Vec<&str> = text.lines().collect();
    let mut spans = Vec::new();
    for (idx, line) in lines.iter().enumerate() {
        if !is_code(line) {
            continue;
        }
        let trimmed = line.trim_start();
        let indent = line.len() - trimmed.len();
        let Some(rest) = trimmed
            .strip_prefix("pub(crate) mod ")
            .or_else(|| trimmed.strip_prefix("pub(super) mod "))
            .or_else(|| trimmed.strip_prefix("pub mod "))
            .or_else(|| trimmed.strip_prefix("mod "))
        else {
            continue;
        };
        let Some(name) = rest.strip_suffix(" {") else {
            continue;
        };
        let closing = format!("{}}}", " ".repeat(indent));
        let end = lines[idx + 1..]
            .iter()
            .position(|l| *l == closing)
            .map(|p| idx + 1 + p)
            .unwrap_or(lines.len());
        spans.push(ModuleSpan {
            name: name.trim().to_string(),
            start: idx + 1,
            end,
        });
    }
    spans
}

/// Is `word` present in `line` as a whole token rather than as part of a longer one?
///
/// This is what keeps the numeric literal `4u32` and the identifier `read_u32` from reading as
/// type positions. A trailing `::` is excluded for the same reason: `u32::MAX` is a path to an
/// associated constant — a *value* — and flagging it made P3 object to the very test that proves
/// the widening reader accepts both signednesses.
fn names_token(line: &str, word: &str) -> bool {
    let ident = |c: char| c.is_alphanumeric() || c == '_';
    let mut from = 0usize;
    while let Some(at) = line[from..].find(word) {
        let start = from + at;
        let end = start + word.len();
        let before_ok = start == 0 || !line[..start].chars().next_back().is_some_and(ident);
        let after_ok = end == line.len() || !line[end..].chars().next().is_some_and(ident);
        let is_value_path = line[end..].starts_with("::");
        if before_ok && after_ok && !is_value_path {
            return true;
        }
        from = end;
    }
    false
}

/// **P3** — inside a module that handles values of a width-varying ABI alias, the alias's own
/// widths may not be spelled out.
///
/// **Known limitation, stated rather than hidden:** the scope is a `mod` block. Code at a file's
/// top level is not scanned, because the two files that handle these values at top level
/// (`src/logging.rs`, `tests/mock_ort/mod.rs`) also carry unrelated `u32`s — a Vulkan vendor id,
/// a source line number — and a lint that reports those trains people to ignore it. This lint is
/// an early warning that costs milliseconds on any box; the *decisive* check that the crate
/// compiles for a second platform is the Linux `cargo test --lib --no-run` step in CI, which is
/// why that step exists under its own name.
fn scan_width_varying(file: &str, text: &str) -> Vec<Finding> {
    let lines: Vec<&str> = text.lines().collect();
    let mut findings = Vec::new();
    for span in module_spans(text) {
        let body = &lines[span.start.min(lines.len())..span.end.min(lines.len())];
        for sym in WIDTH_VARYING_SYMBOLS {
            let handles = body
                .iter()
                .any(|l| is_code(l) && (l.contains(sym.const_prefix) || l.contains(sym.alias)));
            if !handles {
                continue;
            }
            for (offset, line) in body.iter().enumerate() {
                if !is_code(line) {
                    continue;
                }
                for width in sym.widths {
                    if !names_token(line, width) {
                        continue;
                    }
                    findings.push(Finding {
                        file: file.to_string(),
                        line: span.start + offset + 1,
                        text: line.trim().to_string(),
                        message: format!(
                            "`mod {}` handles `{}` values and spells the width `{width}`.\n      \
                             Why it matters: {}.\n      Remedy: {}",
                            span.name, sym.alias, sym.why, sym.remedy
                        ),
                    });
                }
            }
        }
    }
    findings
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

#[test]
fn width_varying_abi_values_are_carried_as_the_alias() {
    let mut findings = Vec::new();
    for path in rust_files() {
        let name = rel(&path);
        if name == SELF_PATH {
            continue;
        }
        let Ok(text) = fs::read_to_string(&path) else {
            continue;
        };
        findings.extend(scan_width_varying(&name, &text));
    }
    report(
        "rule P3 (width-varying ABI values are carried as the alias)",
        &findings,
    );
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
    //
    // Raised from 24 to 32 in issue #39, deliberately and for one reason: the scan now covers the
    // `modelrunner/` crate as well as this one, and that crate contributes 13 items of its own
    // (path encoding in `ortapi.rs`, library naming and loader hints in `ortlib.rs`). The ceiling
    // is a budget for what a reviewer will read in one sitting, so it moves when the thing being
    // read gets larger — not when someone wants to add one more fork.
    assert!(
        total <= 32,
        "the workspace now has {total} platform-conditional items, which is more than a reviewer \
         will actually read. Consolidate them behind cfg-selected aliases (see `OrtChar`) rather \
         than forking logic at each use site, then raise this ceiling deliberately."
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

#[test]
fn p2_reads_the_target_os_spelling_as_the_same_fork() {
    // The spelling `modelrunner/` uses throughout. Counting only `#[cfg(windows)]` is why P2 had
    // nothing to say about that crate's forks at all (issue #39).
    let planted = "#[cfg(target_os = \"windows\")]\nfn only_on_windows() {}\n";
    let findings = scan_balanced_forks("planted.rs", planted);
    assert_eq!(
        findings.len(),
        1,
        "P2 ignored a one-armed `target_os = \"windows\"` fork"
    );

    let both = concat!(
        "#[cfg(target_os = \"windows\")]\n",
        "fn f() {}\n",
        "#[cfg(not(target_os = \"windows\"))]\n",
        "fn f() {}\n",
    );
    assert!(scan_balanced_forks("ok.rs", both).is_empty());
}

#[test]
fn p2_accepts_a_three_way_fork_whose_negative_arm_is_a_not_any() {
    // `modelrunner/src/ortlib.rs::library_file_names` — Windows, macOS, everything else. The
    // Windows-excluding arm is spelled `not(any(...))`, and reading that as "no sibling" would
    // make P2 report a correct fork as a hole, which is the fastest way to get a lint ignored.
    let three_way = concat!(
        "#[cfg(target_os = \"windows\")]\n",
        "fn f() {}\n",
        "#[cfg(target_os = \"macos\")]\n",
        "fn f() {}\n",
        "#[cfg(not(any(target_os = \"windows\", target_os = \"macos\")))]\n",
        "fn f() {}\n",
    );
    assert!(
        scan_balanced_forks("three.rs", three_way).is_empty(),
        "P2 reported a correctly-forked three-way platform split"
    );
}

#[test]
fn the_lint_actually_reaches_the_model_runner_crate() {
    // The scanner said nothing about `modelrunner/` for as long as that crate existed, because
    // its roots stopped at this crate's boundary. Two Linux-only type errors merged through the
    // gap. Assert the reach itself, not just the rules: a future refactor that narrows `ROOTS`
    // reopens the same hole silently.
    let files = rust_files();
    let names: Vec<String> = files.iter().map(|p| rel(p)).collect();
    for expected in [
        "modelrunner/src/ortapi.rs",
        "modelrunner/src/ortlib.rs",
        "modelrunner/src/run.rs",
        "modelrunner/tests/cli.rs",
    ] {
        assert!(
            names.iter().any(|n| n == expected),
            "the portability lint does not scan `{expected}`; scanned {} file(s): {names:?}",
            names.len()
        );
    }
}

#[test]
fn every_width_varying_entry_names_a_real_binding_and_its_constant_prefix() {
    // A typo'd alias in the table is a lint entry that can never fire, and it looks exactly like
    // one that is passing. The constant prefix is the alias plus `_`, by bindgen's own naming.
    for sym in WIDTH_VARYING_SYMBOLS {
        assert_eq!(
            sym.const_prefix,
            format!("{}_", sym.alias),
            "`{}`'s constant prefix does not follow bindgen's naming, so the scanner would never \
             recognise a module that handles it",
            sym.alias
        );
        assert!(sym.alias.starts_with("ort::"));
        assert_eq!(sym.widths, ["i32", "u32"]);
    }
}

#[test]
fn the_narrow_p3_table_is_a_decision_and_not_an_oversight() {
    // 25 of the 28 `ort` enum aliases change signedness between MSVC and GCC, and this table
    // lists one of them. That looks like something to go and fix, so: it was tried.
    //
    // Adding `ort::OrtErrorCode` makes P3 fire on `src/sys.rs`'s `fn at_version(version: u32)`,
    // where `u32` is an ORT *API version* and is correct on every target — flagged only because
    // the enclosing test module elsewhere names `ort::OrtErrorCode_ORT_EP_FAIL`. P3's trigger is
    // "this scope names a constant of the alias", which is a weak signal that gets weaker with
    // every entry added, and this file's own rule is that a lint reporting unrelated integers
    // trains people to ignore it.
    //
    // The 24 unlisted aliases are covered by `cargo ci --cross`, which compiles for the other
    // lane's target and is not a heuristic at all. Widening this table is the tempting fix and
    // the wrong one; if you are about to, read `rust/ci/linux-stub-include/README.md` first.
    assert_eq!(
        WIDTH_VARYING_SYMBOLS.len(),
        1,
        "the P3 table grew. That is a deliberate decision to revisit, not a routine edit — see \
         this test's comment for the false positive it costs and for what already covers the \
         aliases it would add."
    );
}

#[test]
fn p3_catches_a_spelled_width_for_each_alias_in_the_table() {
    // One planted violation per entry, so a table entry that is present but inert is visible.
    for sym in WIDTH_VARYING_SYMBOLS {
        let planted = format!(
            "    mod m {{\n        static C: Mutex<Vec<(i32, String)>> = f();\n        \
             fn g() {{ let _ = {}SOMETHING; }}\n    }}\n",
            sym.const_prefix
        );
        let findings = scan_width_varying("planted.rs", &planted);
        assert_eq!(
            findings.len(),
            1,
            "P3 said nothing about a spelled `i32` in a module that handles `{}`",
            sym.alias
        );
        assert!(findings[0].message.contains(sym.alias));
    }
}

#[test]
fn p3_does_not_read_an_associated_constant_as_a_carrier() {
    // `u32::MAX` is a *value*, and the path that names it is not a type position. Reading it as
    // one made P3 fire on `modelrunner/src/ortapi.rs`'s own portability test, which is a lint
    // objecting to the control that proves the bug is fixed.
    let values = concat!(
        "    mod m {\n",
        "        fn g() {\n",
        "            let _ = widen(u32::MAX);\n",
        "            let _ = i32::MIN;\n",
        "            let _ = ort::OrtLoggingLevel_ORT_LOGGING_LEVEL_INFO;\n",
        "        }\n",
        "    }\n",
    );
    assert!(
        scan_width_varying("values.rs", values).is_empty(),
        "P3 read an associated constant as a spelled carrier: {:?}",
        scan_width_varying("values.rs", values)
            .iter()
            .map(|f| f.text.clone())
            .collect::<Vec<_>>()
    );

    // ...and it must still catch the carrier on a line that also has a value path on it.
    let carrier = concat!(
        "    mod m {\n",
        "        fn g(level: i32) { let _ = ort::OrtLoggingLevel_ORT_LOGGING_LEVEL_INFO; }\n",
        "    }\n",
    );
    assert_eq!(
        scan_width_varying("carrier.rs", carrier).len(),
        1,
        "the value-path exemption swallowed a real carrier"
    );
}

// ---------------------------------------------------------------------------------------------
// P3's scanner, against the exact code that blocked seven Linux steps on 2026-08-02.
// ---------------------------------------------------------------------------------------------

/// Verbatim shape of `src/ep.rs::tests::broken_commitment` before the fix — a severity carrier
/// declared `i32`, in a module that compares against `ort::OrtLoggingLevel_*` constants.
const PLANTED_PRE_FIX: &str = concat!(
    "    mod broken_commitment {\n",
    "        pub(super) static CAPTURED: Mutex<Vec<(i32, String)>> = Mutex::new(Vec::new());\n",
    "        fn run_polarity(status: ort::OrtStatusPtr) -> (bool, Vec<(i32, String)>) {\n",
    "            assert_eq!(seen[0].0, ort::OrtLoggingLevel_ORT_LOGGING_LEVEL_WARNING);\n",
    "        }\n",
    "    }\n",
);

/// The same module after the fix, plus the `c_int` that legitimately belongs to ORT's own
/// callback signature and must not be flagged.
const PLANTED_POST_FIX: &str = concat!(
    "    mod broken_commitment {\n",
    "        pub(super) static CAPTURED: Mutex<Vec<(ort::OrtLoggingLevel, String)>> =\n",
    "            Mutex::new(Vec::new());\n",
    "        unsafe extern \"C\" fn fake_log_message(_line: std::os::raw::c_int) {}\n",
    "        fn run_polarity(s: ort::OrtStatusPtr)\n",
    "            -> (bool, Vec<(ort::OrtLoggingLevel, String)>) {\n",
    "            assert_eq!(seen[0].0, ort::OrtLoggingLevel_ORT_LOGGING_LEVEL_WARNING);\n",
    "        }\n",
    "    }\n",
);

#[test]
fn detects_the_spelled_width_that_blocked_the_linux_lane() {
    let findings = scan_width_varying("planted.rs", PLANTED_PRE_FIX);
    assert_eq!(
        findings.len(),
        2,
        "the scanner missed the `i32` severity carrier that cost seven Linux steps: {:?}",
        findings.iter().map(|f| f.line).collect::<Vec<_>>()
    );
    assert!(findings[0].message.contains("ort::OrtLoggingLevel"));
    assert!(findings[0].message.contains("i32"));
}

#[test]
fn accepts_the_alias_that_fixed_it_and_the_abi_s_own_c_int() {
    assert!(
        scan_width_varying("fixed.rs", PLANTED_POST_FIX).is_empty(),
        "the scanner rejected the correct fix, or flagged the `c_int` that ORT's own callback \
         signature requires — either way it would block the repair as well as the bug"
    );
}

#[test]
fn a_cast_is_not_accepted_as_a_fix() {
    // The tempting eleven-call-site repair. It compiles on both platforms and preserves exactly
    // the assumption that produced the bug, so the lint must still be red on it.
    let cast = concat!(
        "    mod m {\n",
        "        let want = ort::OrtLoggingLevel_ORT_LOGGING_LEVEL_WARNING as i32;\n",
        "    }\n",
    );
    assert_eq!(
        scan_width_varying("cast.rs", cast).len(),
        1,
        "an `as i32` cast at the call site passed the lint"
    );
}

#[test]
fn a_module_that_never_touches_the_alias_may_spell_whatever_it_likes() {
    let unrelated = concat!(
        "    mod vk {\n",
        "        fn vendor_id() -> u32 { 0 }\n",
        "    }\n",
    );
    assert!(
        scan_width_varying("unrelated.rs", unrelated).is_empty(),
        "P3 fired on a module that handles no ABI severity at all; a lint that reports unrelated \
         integers trains people to ignore it"
    );
}

#[test]
fn p3_does_not_read_literals_or_identifiers_as_type_positions() {
    let noisy = concat!(
        "    mod m {\n",
        "        const X: usize = 4;\n",
        "        fn f() { let _ = 4u32; let _ = read_i32(); }\n",
        "        fn g() -> ort::OrtLoggingLevel {\n",
        "            ort::OrtLoggingLevel_ORT_LOGGING_LEVEL_INFO\n",
        "        }\n",
        "    }\n",
    );
    assert!(
        scan_width_varying("noisy.rs", noisy).is_empty(),
        "`4u32` or `read_i32` read as a spelled width: {:?}",
        scan_width_varying("noisy.rs", noisy)
            .iter()
            .map(|f| f.text.clone())
            .collect::<Vec<_>>()
    );
}

#[test]
fn p3_module_spans_do_not_leak_into_the_next_module() {
    // If the span scanner ran off the end of `mod a`, `mod b`'s `i32` would be attributed to a
    // module that handles severities, and the finding would name the wrong file region.
    let two = concat!(
        "    mod a {\n",
        "        fn f() -> ort::OrtLoggingLevel {\n",
        "            ort::OrtLoggingLevel_ORT_LOGGING_LEVEL_INFO\n",
        "        }\n",
        "    }\n",
        "    mod b {\n",
        "        fn g() -> i32 { 0 }\n",
        "    }\n",
    );
    assert!(
        scan_width_varying("two.rs", two).is_empty(),
        "`mod a`'s span swallowed `mod b`: {:?}",
        scan_width_varying("two.rs", two)
            .iter()
            .map(|f| f.text.clone())
            .collect::<Vec<_>>()
    );
}
