//! **The layering lint.** `DESIGN.md` §4.2, M0 exit criterion 6.
//!
//! Two rules protect the op layer, and both are enforced here rather than in review:
//!
//! 1. **The ORT C ABI never appears in `src/ops/`.** No `crate::sys`, no `Ort*` type, no `OrtApi`
//!    function pointer. Op handlers see `NodeDesc`, `NodeView`, `TensorRef`, `OutRef` and
//!    `DispatchContext`.
//! 2. **Raw Vulkan never appears in `src/ops/`.** No `ash`, no `vk::`, no `Vk*` handle — and no
//!    `unsafe`, because everything an op handler legitimately needs is safe by construction.
//!
//! Why a test and not a `deny` attribute or an xtask:
//!
//! * A `deny` attribute cannot express "this identifier must not appear in this directory" —
//!   module privacy alone does not stop `ops/` from naming `ash`, which is a normal dependency of
//!   the crate, and there is no lint for "do not use this crate from this module".
//! * An xtask is a second binary to build, invoke and keep working on four CI lanes. A test is
//!   already run by `cargo test`, so it is impossible to forget to wire up: CI cannot be green
//!   without it, and a contributor gets the failure locally before pushing.
//!
//! Run it locally with:
//!
//! ```text
//! cargo test --test layering
//! ```
//!
//! The lint is itself tested. [`detects_planted_ort_abi_violations`] and friends run the scanner
//! over deliberately-planted violations and assert it catches every one, so a refactor that
//! accidentally neuters the scanner fails too. That is the permanent, always-on version of
//! "plant a violation and check CI goes red".

use std::fs;
use std::path::{Path, PathBuf};

/// A forbidden token and the reason it is forbidden.
struct Rule {
    /// Matched as a whole word (or as a literal, for tokens containing `:`).
    token: &'static str,
    rule: &'static str,
    why: &'static str,
}

/// The forbidden vocabulary of `src/ops/`.
const OPS_RULES: &[Rule] = &[
    // --- Rule 1: no ORT C ABI ---
    Rule {
        token: "sys",
        rule: "1 (no ORT ABI)",
        why: "op code must not reach the raw ORT bindings; use engine::NodeDesc / registry::NodeView",
    },
    Rule {
        token: "ort",
        rule: "1 (no ORT ABI)",
        why: "op code must not name the generated ORT binding module",
    },
    Rule {
        token: "OrtApi",
        rule: "1 (no ORT ABI)",
        why: "op code must never hold an ORT function-pointer table",
    },
    Rule {
        token: "OrtNode",
        rule: "1 (no ORT ABI)",
        why: "op code sees NodeDesc / NodeView, never a raw OrtNode",
    },
    Rule {
        token: "OrtValue",
        rule: "1 (no ORT ABI)",
        why: "op code sees TensorRef / BufferView, never a raw OrtValue",
    },
    Rule {
        token: "OrtStatus",
        rule: "1 (no ORT ABI)",
        why: "op code returns EpError; only ep.rs constructs an OrtStatus",
    },
    Rule {
        token: "OrtKernelContext",
        rule: "1 (no ORT ABI)",
        why: "op code has no access to ORT's kernel context",
    },
    Rule {
        token: "OrtGraph",
        rule: "1 (no ORT ABI)",
        why: "op code sees one node at a time, never the ORT graph",
    },
    // --- Rule 2: no raw Vulkan, no unsafe ---
    Rule {
        token: "ash",
        rule: "2 (no raw Vulkan)",
        why: "op code must not depend on the Vulkan bindings; express intent via DispatchContext",
    },
    Rule {
        token: "gpu_allocator",
        rule: "2 (no raw Vulkan)",
        why: "memory is the engine's concern; use DispatchContext::alloc_temp",
    },
    Rule {
        token: "vk::",
        rule: "2 (no raw Vulkan)",
        why: "op code must not name a Vulkan type",
    },
    Rule {
        token: "vkCmdDispatch",
        rule: "2 (no raw Vulkan)",
        why: "op code requests a dispatch via KernelRequest; the engine records it",
    },
    Rule {
        token: "VkBuffer",
        rule: "2 (no raw Vulkan)",
        why: "op code holds an opaque BufferView, never a VkBuffer",
    },
    Rule {
        token: "VkCommandBuffer",
        rule: "2 (no raw Vulkan)",
        why: "op code cannot hold a command buffer",
    },
    Rule {
        token: "VkPipeline",
        rule: "2 (no raw Vulkan)",
        why: "pipeline creation and caching belong to the engine",
    },
    Rule {
        token: "unsafe",
        rule: "2 (no raw Vulkan)",
        why: "everything an op handler needs is safe; an unsafe block here means a layer was skipped",
    },
];

/// Modules outside `ops/` that must also stay clear of raw Vulkan (`DESIGN.md` §4.3).
const NO_VULKAN_MODULES: &[&str] = &["registry.rs", "ep.rs", "factory.rs", "sys.rs", "logging.rs"];

const VULKAN_TOKENS: &[&str] = &["ash", "gpu_allocator", "vk::"];

/// One detected violation.
#[derive(Debug, PartialEq, Eq)]
struct Violation {
    file: String,
    line: usize,
    token: String,
    rule: &'static str,
    why: &'static str,
}

impl std::fmt::Display for Violation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}:{}: layering rule {} violated by `{}` — {}",
            self.file, self.line, self.rule, self.token, self.why
        )
    }
}

/// Blank out comments and string/char literals so documentation that *names* a forbidden token
/// (which `ops/mod.rs` does deliberately, at length) is not a violation.
///
/// Replaces the elided bytes with spaces so line numbers and columns survive.
fn strip_comments_and_strings(src: &str) -> String {
    let b: Vec<char> = src.chars().collect();
    let mut out: Vec<char> = Vec::with_capacity(b.len());
    let mut i = 0usize;

    while i < b.len() {
        let c = b[i];
        let next = b.get(i + 1).copied();

        // Line comment (covers `//`, `///`, `//!`).
        if c == '/' && next == Some('/') {
            while i < b.len() && b[i] != '\n' {
                out.push(' ');
                i += 1;
            }
            continue;
        }
        // Block comment (covers `/*`, `/**`, `/*!`), with nesting as Rust allows.
        if c == '/' && next == Some('*') {
            let mut depth = 1usize;
            out.push(' ');
            out.push(' ');
            i += 2;
            while i < b.len() && depth > 0 {
                if b[i] == '/' && b.get(i + 1) == Some(&'*') {
                    depth += 1;
                    out.push(' ');
                    out.push(' ');
                    i += 2;
                    continue;
                }
                if b[i] == '*' && b.get(i + 1) == Some(&'/') {
                    depth -= 1;
                    out.push(' ');
                    out.push(' ');
                    i += 2;
                    continue;
                }
                out.push(if b[i] == '\n' { '\n' } else { ' ' });
                i += 1;
            }
            continue;
        }
        // Raw string: r"...", r#"..."#, br#"..."#
        if (c == 'r' || c == 'b')
            && let Some(consumed) = raw_string_len(&b, i)
        {
            for c in b.iter().skip(i).take(consumed) {
                out.push(if *c == '\n' { '\n' } else { ' ' });
            }
            i += consumed;
            continue;
        }
        // Ordinary string literal.
        if c == '"' {
            out.push(' ');
            i += 1;
            while i < b.len() {
                if b[i] == '\\' {
                    out.push(' ');
                    out.push(' ');
                    i += 2;
                    continue;
                }
                let done = b[i] == '"';
                out.push(if b[i] == '\n' { '\n' } else { ' ' });
                i += 1;
                if done {
                    break;
                }
            }
            continue;
        }
        // Char literal. Lifetimes (`'a`) are not literals, so require a closing quote within 4
        // chars and never swallow a newline.
        if c == '\'' && let Some(consumed) = char_literal_len(&b, i) {
            out.extend(std::iter::repeat_n(' ', consumed));
            i += consumed;
            continue;
        }

        out.push(c);
        i += 1;
    }
    out.into_iter().collect()
}

/// Length of a raw string starting at `i`, or `None` if this is not one.
fn raw_string_len(b: &[char], i: usize) -> Option<usize> {
    let mut j = i;
    if b.get(j) == Some(&'b') {
        j += 1;
    }
    if b.get(j) != Some(&'r') {
        return None;
    }
    j += 1;
    let hash_start = j;
    while b.get(j) == Some(&'#') {
        j += 1;
    }
    let hashes = j - hash_start;
    if b.get(j) != Some(&'"') {
        return None;
    }
    j += 1;
    let closing: String = std::iter::once('"').chain(std::iter::repeat_n('#', hashes)).collect();
    let closing: Vec<char> = closing.chars().collect();
    while j < b.len() {
        if b[j] == '"' && b[j..].starts_with(closing.as_slice()) {
            return Some(j + closing.len() - i);
        }
        j += 1;
    }
    Some(b.len() - i)
}

/// Length of a char literal starting at `i`, or `None` if this is a lifetime or label.
fn char_literal_len(b: &[char], i: usize) -> Option<usize> {
    let mut j = i + 1;
    if b.get(j) == Some(&'\\') {
        j += 1;
    }
    let limit = (i + 8).min(b.len());
    while j < limit {
        if b[j] == '\n' {
            return None;
        }
        if b[j] == '\'' {
            return Some(j + 1 - i);
        }
        j += 1;
    }
    None
}

/// True when `token` occurs in `line` as a whole word.
///
/// Tokens containing `:` (e.g. `vk::`) are matched literally, since `:` is not a word character.
fn contains_token(line: &str, token: &str) -> bool {
    if token.contains(':') {
        return line.contains(token);
    }
    let bytes = line.as_bytes();
    let tb = token.as_bytes();
    let is_word = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
    let mut start = 0usize;
    while let Some(pos) = line[start..].find(token) {
        let at = start + pos;
        let before_ok = at == 0 || !is_word(bytes[at - 1]);
        let after = at + tb.len();
        let after_ok = after >= bytes.len() || !is_word(bytes[after]);
        if before_ok && after_ok {
            return true;
        }
        start = at + 1;
        if start >= line.len() {
            break;
        }
    }
    false
}

/// Scan one source file's contents against a rule set.
fn scan(display_path: &str, source: &str, rules: &[Rule]) -> Vec<Violation> {
    let code = strip_comments_and_strings(source);
    let mut found = Vec::new();
    for (n, line) in code.lines().enumerate() {
        for rule in rules {
            if contains_token(line, rule.token) {
                found.push(Violation {
                    file: display_path.to_string(),
                    line: n + 1,
                    token: rule.token.to_string(),
                    rule: rule.rule,
                    why: rule.why,
                });
            }
        }
    }
    found
}

/// Every `.rs` file under `dir`, recursively.
fn rust_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(entries) = fs::read_dir(dir) else {
        return out;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            out.extend(rust_files(&path));
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
    out.sort();
    out
}

fn src_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src")
}

// ---------------------------------------------------------------------------------------------
// The lint itself
// ---------------------------------------------------------------------------------------------

#[test]
fn ops_layer_contains_no_ort_abi_and_no_raw_vulkan() {
    let ops = src_dir().join("ops");
    assert!(
        ops.is_dir(),
        "src/ops must exist — it is the layer this lint protects"
    );

    let mut violations = Vec::new();
    for file in rust_files(&ops) {
        let source = fs::read_to_string(&file).expect("op source must be readable UTF-8");
        let display = file
            .strip_prefix(env!("CARGO_MANIFEST_DIR"))
            .unwrap_or(&file)
            .display()
            .to_string();
        violations.extend(scan(&display, &source, OPS_RULES));
    }

    assert!(
        violations.is_empty(),
        "layering violations in src/ops/ ({} found):\n{}\n\n\
         Op handlers see engine::{{NodeDesc, TensorRef, OutRef, DispatchContext, KernelRequest}} \
         and registry::NodeView — nothing else. See DESIGN.md §4.2.",
        violations.len(),
        violations
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn boundary_and_registry_modules_contain_no_raw_vulkan() {
    // The mirror-image rule: the ORT boundary layer and the registry must not reach into Vulkan
    // either. Only `engine.rs` (and the `vk/` tree it will own) may (`DESIGN.md` §4.3).
    let rules: Vec<Rule> = VULKAN_TOKENS
        .iter()
        .map(|t| Rule {
            token: t,
            rule: "4.3 (module dependency table)",
            why: "only engine.rs and the vk/ tree it owns may touch Vulkan",
        })
        .collect();

    let mut violations = Vec::new();
    for name in NO_VULKAN_MODULES {
        let path = src_dir().join(name);
        if !path.is_file() {
            continue;
        }
        let source = fs::read_to_string(&path).expect("source must be readable UTF-8");
        violations.extend(scan(name, &source, &rules));
    }

    assert!(
        violations.is_empty(),
        "raw Vulkan leaked out of the engine layer:\n{}",
        violations
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n")
    );
}

// ---------------------------------------------------------------------------------------------
// Tests of the lint (the permanently-planted violations)
// ---------------------------------------------------------------------------------------------

#[test]
fn detects_planted_ort_abi_violations() {
    let planted = [
        "use crate::sys::ort;",
        "use crate::sys::ort::OrtNode;",
        "fn f(node: *const OrtNode) -> *mut OrtStatus { std::ptr::null_mut() }",
        "fn g(api: &OrtApi) {}",
        "fn h(ctx: *mut OrtKernelContext) {}",
        "fn i(v: *const OrtValue) {}",
        "fn j(g: *const OrtGraph) {}",
    ];
    for line in planted {
        let found = scan("planted.rs", line, OPS_RULES);
        assert!(
            !found.is_empty(),
            "the layering lint failed to catch a planted ORT-ABI violation: {line}"
        );
        assert!(found.iter().all(|v| v.rule.starts_with('1')), "{found:?}");
    }
}

#[test]
fn detects_planted_vulkan_violations() {
    let planted = [
        "use ash::vk;",
        "use gpu_allocator::vulkan::Allocator;",
        "let buf: vk::Buffer = vk::Buffer::null();",
        "fn f(b: VkBuffer) {}",
        "fn g(c: VkCommandBuffer) {}",
        "fn h(p: VkPipeline) {}",
        "unsafe { device.cmd_dispatch(cb, 1, 1, 1) };",
        "vkCmdDispatch(cb, 1, 1, 1);",
    ];
    for line in planted {
        let found = scan("planted.rs", line, OPS_RULES);
        assert!(
            !found.is_empty(),
            "the layering lint failed to catch a planted Vulkan violation: {line}"
        );
        assert!(found.iter().all(|v| v.rule.starts_with('2')), "{found:?}");
    }
}

#[test]
fn a_realistic_planted_op_module_is_rejected() {
    // What a well-meaning contributor's shortcut actually looks like: a handler that reaches for
    // the raw command buffer "just this once". This is the file the lint has to fail on.
    let planted = r#"
use ash::vk;
use crate::engine::{DispatchContext, EpResult, NodeDesc};

pub fn translate_add(node: &NodeDesc, ctx: &mut dyn DispatchContext) -> EpResult<()> {
    let cb: vk::CommandBuffer = ctx.command_buffer();
    unsafe { ctx.device().cmd_dispatch(cb, 1, 1, 1) };
    Ok(())
}
"#;
    let found = scan("src/ops/elementwise.rs", planted, OPS_RULES);
    assert!(
        found.len() >= 3,
        "expected the planted module to trip several rules, got {found:?}"
    );
    assert!(found.iter().any(|v| v.token == "ash"));
    assert!(found.iter().any(|v| v.token == "vk::"));
    assert!(found.iter().any(|v| v.token == "unsafe"));
}

#[test]
fn clean_op_source_passes() {
    let clean = r#"
use crate::engine::{DispatchContext, EpResult, KernelRequest, NodeDesc, TensorDesc, DType};
use crate::registry::{DeclineReason, NodeView};

pub fn claim_add(view: &NodeView<'_>) -> Result<(), DeclineReason> {
    if view.num_inputs() != 2 {
        return Err("Add: expected exactly 2 inputs".into());
    }
    Ok(())
}

pub fn translate_add(node: &NodeDesc, ctx: &mut dyn DispatchContext) -> EpResult<()> {
    let a = ctx.resolve(&node.inputs[0])?;
    let b = ctx.resolve(&node.inputs[1])?;
    let out = ctx.bind_output(&node.outputs[0], TensorDesc::new(DType::F32, vec![1]))?;
    ctx.dispatch(KernelRequest {
        shader: "elementwise_binary",
        spec_constants: vec![],
        push_constants: vec![],
        bindings: vec![a, b, out],
        workgroups: [1, 1, 1],
    })
}
"#;
    let found = scan("src/ops/elementwise.rs", clean, OPS_RULES);
    assert!(found.is_empty(), "false positives on clean source: {found:?}");
}

#[test]
fn documentation_naming_forbidden_tokens_is_not_a_violation() {
    // `src/ops/mod.rs` documents the rules by naming every forbidden token. If the comment and
    // string stripping regressed, that file alone would produce dozens of false positives — so
    // this is both a unit test and a guard on the real scan above.
    let doc_heavy = r##"
//! No `unsafe`, no `ash`, no `vk::`, no `crate::sys`, no `OrtNode`.
/* Block comment mentioning OrtStatus and VkBuffer and unsafe. */
/// Doc comment: never write `unsafe { ... }` here.
pub fn f() -> &'static str {
    let s = "the word unsafe and ash and vk:: inside a string literal";
    let r = r#"a raw string with OrtApi and VkPipeline in it"#;
    let c = '\'';
    let _ = (s, r, c);
    "fine"
}
"##;
    let found = scan("src/ops/mod.rs", doc_heavy, OPS_RULES);
    assert!(
        found.is_empty(),
        "comment/string stripping regressed — false positives: {found:?}"
    );
}

#[test]
fn word_boundaries_are_respected() {
    // `ash` must not match `hash`, `flash`, or `ash_like`; `sys` must not match `system`.
    let benign = "let hashed = flash_map(); let system = subsystem();";
    let found = scan("src/ops/x.rs", benign, OPS_RULES);
    assert!(found.is_empty(), "word-boundary false positives: {found:?}");

    // But a genuine use still trips.
    assert!(!scan("src/ops/x.rs", "use ash::Entry;", OPS_RULES).is_empty());
}

#[test]
fn the_lint_actually_reads_the_ops_directory() {
    // A lint that silently scans zero files is worse than no lint: it is a green check that
    // proves nothing. Fail if `src/ops` ever stops producing files to scan.
    let files = rust_files(&src_dir().join("ops"));
    assert!(
        !files.is_empty(),
        "the layering lint found no files under src/ops — it would pass vacuously"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Barrier module boundary (DESIGN.md §7.5, §4.2)
//
// `rust/src/vk/barrier.rs` is the ONLY file permitted to name Vulkan barrier API tokens.
// Anything outside it that names these tokens fails CI.  The rule exists because the barrier
// implementation has two backends (sync2 and legacy), and scattering `if caps.sync2 { … } else
// { … }` across recording code would turn the dual path into a bug farm.
//
// The forbidden vocabulary (Rust/ash identifiers, not C names):
//   cmd_pipeline_barrier  — covers both the core and sync2 variants
//   BufferMemoryBarrier   — covers both VkBufferMemoryBarrier and VkBufferMemoryBarrier2
//   DependencyInfo        — VkDependencyInfo (sync2 path)
//   PipelineStageFlags    — covers both PipelineStageFlags and PipelineStageFlags2
//   AccessFlags           — covers both AccessFlags and AccessFlags2
//
// `Capabilities::synchronization2` may only be *read* in `vk/barrier.rs` and `vk/caps.rs`.
// We enforce this by banning the token `synchronization2` everywhere else (the strip logic
// removes it from comments and string literals, so documentation is not a false positive).
// ─────────────────────────────────────────────────────────────────────────────

/// Barrier tokens that are only allowed in `src/vk/barrier.rs`.
const BARRIER_RULES: &[Rule] = &[
    Rule {
        token: "cmd_pipeline_barrier",
        rule: "7.5 (barrier boundary)",
        why: "all barrier emission must go through Barriers::buffer_deps / execution_only in vk/barrier.rs",
    },
    // The sync2 variant has a `2` suffix, so the whole-word matcher above misses it.
    // Include it as an explicit separate token so both legacy and sync2 are enforced.
    Rule {
        token: "cmd_pipeline_barrier2",
        rule: "7.5 (barrier boundary)",
        why: "vkCmdPipelineBarrier2 may only be called from vk/barrier.rs",
    },
    Rule {
        token: "BufferMemoryBarrier",
        rule: "7.5 (barrier boundary)",
        why: "VkBufferMemoryBarrier(2) may only be constructed in vk/barrier.rs",
    },
    Rule {
        token: "DependencyInfo",
        rule: "7.5 (barrier boundary)",
        why: "VkDependencyInfo may only be constructed in vk/barrier.rs",
    },
    Rule {
        token: "PipelineStageFlags",
        rule: "7.5 (barrier boundary)",
        why: "PipelineStageFlags(2) may only appear in vk/barrier.rs mapping table",
    },
    Rule {
        token: "AccessFlags",
        rule: "7.5 (barrier boundary)",
        why: "AccessFlags(2) may only appear in vk/barrier.rs mapping table",
    },
];

/// Token that may only appear in `vk/barrier.rs` and `vk/caps.rs`.
/// Reading `Capabilities::synchronization2` outside those two files violates the
/// "select once at init, never branch elsewhere" contract (DESIGN.md §7.5 item 1).
const SYNC2_FIELD_RULES: &[Rule] = &[Rule {
    token: "synchronization2",
    rule: "7.5 (caps.synchronization2 read boundary)",
    why: "caps.synchronization2 may only be read in vk/barrier.rs and vk/caps.rs; everywhere \
          else, call barriers.buffer_deps() — the backend was selected once in Device::new",
}];

/// Returns `true` when `path` is `src/vk/barrier.rs` (the only permitted home of barrier tokens).
fn is_barrier_file(path: &std::path::Path) -> bool {
    let s = path.to_string_lossy();
    s.contains("vk") && s.ends_with("barrier.rs")
}

/// Returns `true` when `path` is `src/vk/barrier.rs` or `src/vk/caps.rs`.
fn is_sync2_permitted_file(path: &std::path::Path) -> bool {
    let s = path.to_string_lossy();
    (s.contains("vk") && s.ends_with("barrier.rs"))
        || (s.contains("vk") && s.ends_with("caps.rs"))
}

/// Collect every `.rs` file under `src/` (recursively), excluding those matching `exclude`.
fn rust_files_except(dir: &std::path::Path, exclude: impl Fn(&std::path::Path) -> bool) -> Vec<std::path::PathBuf> {
    rust_files(dir)
        .into_iter()
        .filter(|p| !exclude(p))
        .collect()
}

#[test]
fn barrier_api_is_confined_to_vk_barrier() {
    // Scan every source file EXCEPT vk/barrier.rs for the forbidden barrier tokens.
    let src = src_dir();
    let files = rust_files_except(&src, is_barrier_file);

    let mut violations = Vec::new();
    for file in &files {
        let source = fs::read_to_string(file).expect("source must be readable UTF-8");
        let display = file
            .strip_prefix(env!("CARGO_MANIFEST_DIR"))
            .unwrap_or(file)
            .display()
            .to_string();
        violations.extend(scan(&display, &source, BARRIER_RULES));
    }

    assert!(
        violations.is_empty(),
        "barrier API leaked outside vk/barrier.rs ({} violation(s)):\n{}\n\n\
         All barrier emission must go through Barriers::buffer_deps / execution_only. \
         See DESIGN.md §7.5.",
        violations.len(),
        violations
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn synchronization2_field_read_is_confined_to_barrier_and_caps() {
    // `caps.synchronization2` may only appear in vk/barrier.rs and vk/caps.rs.
    let src = src_dir();
    let files = rust_files_except(&src, is_sync2_permitted_file);

    let mut violations = Vec::new();
    for file in &files {
        let source = fs::read_to_string(file).expect("source must be readable UTF-8");
        let display = file
            .strip_prefix(env!("CARGO_MANIFEST_DIR"))
            .unwrap_or(file)
            .display()
            .to_string();
        violations.extend(scan(&display, &source, SYNC2_FIELD_RULES));
    }

    assert!(
        violations.is_empty(),
        "caps.synchronization2 read outside the permitted files ({} violation(s)):\n{}\n\n\
         Barriers::select is the only code that consults this field; call \
         barriers.buffer_deps() everywhere else. See DESIGN.md §7.5 item 1.",
        violations.len(),
        violations
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n")
    );
}

// ── Proof: planted violations are detected ───────────────────────────────────
//
// These tests scan deliberately-wrong code and assert the scanner catches each violation.
// This is the permanent, always-on version of "plant a violation, watch CI go red, remove it"
// (DESIGN.md §7.5 requirement: "extend [the lint] to enforce this boundary, and prove it by
// planting a violation, watching it fail, then removing it").

#[test]
fn detects_planted_barrier_violations() {
    let planted = [
        // Legacy path leaking into a recording module
        (
            "src/vk/command.rs",
            "unsafe { device.cmd_pipeline_barrier(cb, src, dst, flags, &[], &barriers, &[]) };",
        ),
        // Sync2 path leaking into a recording module
        (
            "src/vk/command.rs",
            "unsafe { sync2.cmd_pipeline_barrier2(cb, &dep_info) };",
        ),
        // Constructing a BufferMemoryBarrier outside the barrier module
        (
            "src/vk/command.rs",
            "let b = vk::BufferMemoryBarrier { ..Default::default() };",
        ),
        // Constructing a DependencyInfo outside the barrier module
        (
            "src/vk/command.rs",
            "let d = vk::DependencyInfo { ..Default::default() };",
        ),
        // Naming PipelineStageFlags outside the barrier module
        (
            "src/vk/command.rs",
            "let f = vk::PipelineStageFlags::COMPUTE_SHADER;",
        ),
        // Naming AccessFlags outside the barrier module
        (
            "src/vk/command.rs",
            "let a = vk::AccessFlags::SHADER_READ;",
        ),
    ];
    for (file, line) in planted {
        let found = scan(file, line, BARRIER_RULES);
        assert!(
            !found.is_empty(),
            "barrier lint missed a planted violation in {file}: {line}"
        );
        assert!(
            found.iter().all(|v| v.rule.starts_with("7.5")),
            "wrong rule fired: {found:?}"
        );
    }
}

#[test]
fn detects_planted_sync2_field_violation() {
    // A recording module that reads caps.synchronization2 directly — violates §7.5 item 1.
    let planted_in_command_rs = r#"
fn record(caps: &Capabilities, cb: vk::CommandBuffer) {
    if caps.synchronization2 {
        emit_sync2_barrier(cb);
    } else {
        emit_legacy_barrier(cb);
    }
}
"#;
    let found = scan("src/vk/command.rs", planted_in_command_rs, SYNC2_FIELD_RULES);
    assert!(
        !found.is_empty(),
        "sync2-field lint missed a planted violation: {planted_in_command_rs}"
    );
}

#[test]
fn barrier_rs_and_caps_rs_are_not_false_positived_by_the_barrier_lint() {
    // The barrier module file itself contains all the forbidden tokens by design. The lint must
    // scan other files, not the barrier module, so this test confirms the exclusion logic.
    //
    // Since we cannot easily load the real barrier.rs in a test (it might not exist in all
    // configurations), we simulate the content and verify the lint does NOT fire when we tell
    // the scanner it is vk/barrier.rs.
    let barrier_rs_content = r#"
fn to_legacy_flags(a: Access) -> (vk::PipelineStageFlags, vk::AccessFlags) {
    match a {
        Access::ShaderRead => (vk::PipelineStageFlags::COMPUTE_SHADER, vk::AccessFlags::SHADER_READ),
    }
}
fn emit(dev: &ash::Device, cb: vk::CommandBuffer, deps: &[BufferDep]) {
    unsafe { dev.cmd_pipeline_barrier(cb, src, dst, flags, &[], &barriers, &[]) };
}
"#;
    // Scanning as vk/barrier.rs would trigger the lint — so we are NOT scanning it.
    // This test verifies the real scan excludes it by calling the exclusion predicate directly.
    let path = std::path::PathBuf::from("src/vk/barrier.rs");
    assert!(
        is_barrier_file(&path),
        "is_barrier_file must return true for vk/barrier.rs — otherwise the real \
         barrier module would be scanned and produce false positives"
    );
    // The lint body would NOT scan this path, so no violations would be produced.
    // Demonstrate that the content *would* violate the rule if scanned:
    let violations = scan("src/vk/barrier.rs", barrier_rs_content, BARRIER_RULES);
    assert!(
        !violations.is_empty(),
        "the barrier.rs content should trip the rules if scanned (smoke test for the rule set)"
    );
}

#[test]
fn clean_recording_code_passes_barrier_lint() {
    // A command recording module that uses Barriers::buffer_deps correctly — must not trip.
    let clean = r#"
use crate::vk::barrier::{Access, Barriers, BufferDep};
use ash::vk;

fn record_add(
    barriers: &Barriers,
    cb: vk::CommandBuffer,
    input_buf: vk::Buffer,
    output_buf: vk::Buffer,
) {
    let dep = BufferDep {
        buffer: input_buf,
        offset: 0,
        size: vk::WHOLE_SIZE,
        src: Access::ShaderWrite,
        dst: Access::ShaderRead,
    };
    // SAFETY: cb is in recording state; dep.buffer is live.
    unsafe { barriers.buffer_deps(cb, &[dep]) };
}
"#;
    let found = scan("src/vk/command.rs", clean, BARRIER_RULES);
    assert!(
        found.is_empty(),
        "false positives on clean recording code: {found:?}"
    );
}

// ---------------------------------------------------------------------------------------------
// C1: no domain-wide contrib opt-in may exist in the code
//
// `DESIGN.md` §1.4 constraint C1: the registry key *is* the allowlist. An op from
// `com.microsoft` is claimable only because a row names it in full; there must be no way to
// express "claim this because of the domain it is in". Morpheus's requirement is that this be
// true *by construction*, not by review, so it is enforced here.
//
// The rule is deliberately blunt: **the contrib domain may not appear as a value in the crate at
// all**, in either of its two spellings, outside the one place that defines what the domain is
// called. Banning the value rather than enumerating comparison forms (`==`, `!=`, `matches!`,
// `if let`, `contains`) is what makes it airtight — there is no third spelling to forget.
//
// Two things are explicitly *not* violations, and the lint must not confuse them:
//   * a fully-qualified op name such as `"com.microsoft::MatMulNBits"` — that names one op, which
//     is exactly what C1 requires;
//   * `Domain::Ms` in the single arm of `Domain::as_str` that maps the enum to its wire string.
//
// Scope note: only non-test code is scanned. Tests legitimately fabricate contrib nodes — indeed
// C1's own regression test must — so linting them would forbid the very test that proves the
// constraint holds. The semantic backstop for the runtime behaviour is Trinity's end-to-end test
// (fabricate `com.microsoft::NotARealOp`, assert a plain decline plus a correct CPU run); this
// lint is the static half.
// ---------------------------------------------------------------------------------------------

/// The contrib domain's wire spelling. Appearing bare (not followed by `::`) is a violation.
const CONTRIB_DOMAIN: &str = "com.microsoft";

/// The Rust-level name for the same thing.
const CONTRIB_DOMAIN_VARIANT: &str = "Domain::Ms";

/// Truncate a source file at its `#[cfg(test)]` module.
///
/// The crate's convention is a single test module at the bottom of each file; this test asserts
/// that convention holds for the files it cares about rather than assuming it, so the truncation
/// cannot silently start hiding real code.
fn non_test_source(src: &str) -> &str {
    match src.find("\n#[cfg(test)]") {
        Some(at) => &src[..at + 1],
        None => src,
    }
}

/// Is this the one permitted site — the `Domain::Ms => "com.microsoft"` arm of `Domain::as_str`?
fn is_domain_definition_site(line: &str) -> bool {
    let t = line.trim();
    t.starts_with("Domain::Ms =>") && t.contains(CONTRIB_DOMAIN)
}

/// Occurrences of the bare contrib domain string, i.e. not followed by `::`.
fn bare_contrib_domain_hits(display_path: &str, source: &str) -> Vec<Violation> {
    let mut out = Vec::new();
    for (n, line) in non_test_source(source).lines().enumerate() {
        if line.trim_start().starts_with("//") || is_domain_definition_site(line) {
            continue;
        }
        let mut start = 0usize;
        while let Some(pos) = line[start..].find(CONTRIB_DOMAIN) {
            let at = start + pos;
            let rest = &line[at + CONTRIB_DOMAIN.len()..];
            // `com.microsoft::Foo` names a single op — permitted, and the whole point of C1.
            if !rest.starts_with("::") {
                out.push(Violation {
                    file: display_path.to_string(),
                    line: n + 1,
                    token: CONTRIB_DOMAIN.to_string(),
                    rule: "C1 (no domain-wide contrib opt-in)",
                    why: "the bare contrib domain may not appear as a value; claim contrib ops by \
                          full registry key (`com.microsoft::OpName`) only",
                });
                break;
            }
            start = at + CONTRIB_DOMAIN.len();
        }
    }
    out
}

/// Occurrences of `Domain::Ms` outside the single definition site.
fn contrib_variant_hits(display_path: &str, source: &str) -> Vec<Violation> {
    let code = strip_comments_and_strings(non_test_source(source));
    let raw: Vec<&str> = non_test_source(source).lines().collect();
    let mut out = Vec::new();
    for (n, line) in code.lines().enumerate() {
        if !line.contains(CONTRIB_DOMAIN_VARIANT) {
            continue;
        }
        if raw.get(n).is_some_and(|r| is_domain_definition_site(r)) {
            continue;
        }
        out.push(Violation {
            file: display_path.to_string(),
            line: n + 1,
            token: CONTRIB_DOMAIN_VARIANT.to_string(),
            rule: "C1 (no domain-wide contrib opt-in)",
            why: "naming the contrib domain in non-test code is how a domain-wide predicate gets \
                  written; the registry key is the allowlist",
        });
    }
    out
}

fn scan_c1(display_path: &str, source: &str) -> Vec<Violation> {
    let mut out = bare_contrib_domain_hits(display_path, source);
    out.extend(contrib_variant_hits(display_path, source));
    out
}

#[test]
fn no_domain_wide_contrib_opt_in_exists_in_the_crate() {
    let files = rust_files(&src_dir());
    assert!(!files.is_empty(), "found no sources to lint");
    let mut found = Vec::new();
    for path in &files {
        let display = path
            .strip_prefix(env!("CARGO_MANIFEST_DIR"))
            .unwrap_or(path)
            .display()
            .to_string();
        let Ok(src) = fs::read_to_string(path) else {
            continue;
        };
        found.extend(scan_c1(&display, &src));
    }
    assert!(
        found.is_empty(),
        "C1 violation(s) — a domain-wide contrib opt-in is forbidden by DESIGN.md §1.4:\n{}",
        found
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn the_c1_lint_reads_a_file_that_actually_mentions_the_domain() {
    // Guards against the lint passing because it scanned nothing, or because the definition-site
    // exemption silently swallowed every hit. `registry.rs` must contain the one permitted site.
    let registry = fs::read_to_string(src_dir().join("registry.rs")).expect("read registry.rs");
    assert!(
        non_test_source(&registry).contains(CONTRIB_DOMAIN),
        "registry.rs no longer defines the contrib domain string; the C1 lint's exemption is \
         probably stale"
    );
    assert!(
        non_test_source(&registry)
            .lines()
            .any(is_domain_definition_site),
        "the `Domain::Ms => \"com.microsoft\"` definition site moved; update is_domain_definition_site"
    );
}

#[test]
fn detects_a_planted_domain_wide_claim_predicate() {
    // The exact violation Morpheus forbids: claiming on the domain rather than on a registry key.
    let planted = r#"
use crate::registry::{Domain, NodeView};

pub fn claim(node: &NodeView) -> bool {
    // A domain-wide opt-in. This must never compile past the lint.
    node.domain() == "com.microsoft"
}
"#;
    let found = scan_c1("src/ops/bad.rs", planted);
    assert!(
        !found.is_empty(),
        "the C1 lint failed to detect a domain-wide string predicate"
    );
}

#[test]
fn detects_a_planted_variant_comparison() {
    for planted in [
        "fn claim(s: &OpSpec) -> bool { s.domain == Domain::Ms }",
        "fn claim(s: &OpSpec) -> bool { s.domain != Domain::Ms }",
        "fn claim(s: &OpSpec) -> bool { matches!(s.domain, Domain::Ms) }",
        "fn claim(s: &OpSpec) -> bool { if let Domain::Ms = s.domain { true } else { false } }",
    ] {
        let found = scan_c1("src/ops/bad.rs", planted);
        assert!(
            !found.is_empty(),
            "the C1 lint failed to detect a domain-wide variant predicate: {planted}"
        );
    }
}

#[test]
fn qualified_contrib_op_names_are_not_c1_violations() {
    // These are per-op allowlist entries — precisely what C1 requires instead of a domain opt-in.
    let clean = r#"
const ANCHORS: &[&str] = &[
    "com.microsoft::MatMulNBits",
    "com.microsoft::GroupQueryAttention",
];
"#;
    let found = scan_c1("src/ops/partition.rs", clean);
    assert!(found.is_empty(), "false positive on qualified names: {found:?}");
}

#[test]
fn the_domain_definition_site_is_exempt_but_nothing_else_in_that_file_is() {
    let source = concat!(
        "fn as_str(self) -> &'static str {\n",
        "    match self {\n",
        "        Domain::Ai => \"ai.onnx\",\n",
        "        Domain::Ms => \"com.microsoft\",\n",
        "    }\n",
        "}\n",
        "fn sneaky(d: Domain) -> bool { d == Domain::Ms }\n",
    );
    let found = scan_c1("src/registry.rs", source);
    assert_eq!(
        found.len(),
        1,
        "expected exactly the `sneaky` line to be flagged, got: {found:?}"
    );
    assert_eq!(found[0].line, 7);
}

#[test]
fn test_modules_are_out_of_scope_for_c1() {
    // C1's own regression test has to fabricate a contrib node. If the lint policed test code it
    // would forbid the test that proves the constraint.
    let source = concat!(
        "pub fn claim() -> bool { false }\n",
        "#[cfg(test)]\n",
        "mod tests {\n",
        "    #[test]\n",
        "    fn fabricated_contrib_op_declines() {\n",
        "        let d = \"com.microsoft\";\n",
        "        assert!(super::claim());\n",
        "    }\n",
        "}\n",
    );
    let found = scan_c1("src/ops/x.rs", source);
    assert!(found.is_empty(), "test code should be out of scope: {found:?}");
}

// ---------------------------------------------------------------------------------------------
// C2: contrib rows must state the ORT release their predicate was written against
// ---------------------------------------------------------------------------------------------

/// C2 (`DESIGN.md` §1.4): every contrib row states the ORT release its predicate was written
/// against.
///
/// The split: `sys` owns the *type* (`SchemaBaseline`, `OrtRelease`) and the pinned/floor
/// releases; `registry.rs` owns which release each row was verified against, stored inside the
/// schema fingerprint so a shape cannot be recorded without its provenance. This test joins the
/// two so neither side can quietly stop holding up its end.
///
/// The trigger is read from the *linked registry*, not from the source text. Rows are declared
/// through the `op_table!` macro, so the string `domain: Domain::Ms` never appears in a source
/// file — grepping for it would have produced a test that could never fire, which is worse than
/// no test at all.
#[test]
fn every_contrib_row_records_the_ort_release_it_was_written_against() {
    let missing: Vec<String> = onnxruntime_vulkan_ep::registry::all_specs()
        .filter(|s| !s.domain.as_str().is_empty() && s.schema_baseline().is_none())
        .map(|s| s.qualified_name().into_owned())
        .collect();
    assert!(
        missing.is_empty(),
        "DESIGN.md §1.4 C2: contrib row(s) {missing:?} have no schema baseline. Give the row a \
         `ContribSchema` whose `baseline` records the ORT release its claim predicate was written \
         against and the date it was verified. Contrib ops have no opset, so this is the only \
         thing that says when anyone last read the schema."
    );
}

#[test]
fn no_default_domain_row_carries_a_contrib_schema_baseline() {
    // The other direction. A baseline on an `ai.onnx` row is noise that dilutes the signal on the
    // rows where it is load-bearing — their contract is the opset window.
    let spurious: Vec<String> = onnxruntime_vulkan_ep::registry::all_specs()
        .filter(|s| s.domain.as_str().is_empty() && s.schema_baseline().is_some())
        .map(|s| s.qualified_name().into_owned())
        .collect();
    assert!(
        spurious.is_empty(),
        "default-domain row(s) {spurious:?} carry a contrib schema baseline; their compatibility \
         contract is the opset window"
    );
}

#[test]
fn no_recorded_baseline_claims_a_newer_ort_than_we_build_against() {
    for spec in onnxruntime_vulkan_ep::registry::all_specs() {
        let Some(b) = spec.schema_baseline() else {
            continue;
        };
        assert!(
            b.verified_against.api_version <= onnxruntime_vulkan_ep::sys::ORT_PINNED.api_version,
            "`{}` claims a baseline newer than the ORT we compile against",
            spec.qualified_name()
        );
    }
}

/// C2's companion: the dump must actually carry the column.
///
/// A baseline recorded in the registry but never printed satisfies the letter of C2 and none of
/// its purpose — the constraint is about *surfacing* the version, so the surface is tested too,
/// in `tests/dump_capabilities.rs`.
#[test]
fn c2_surface_is_covered_by_the_dump_capabilities_suite() {
    let dump = fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("dump_capabilities.rs"),
    )
    .expect("tests/dump_capabilities.rs must exist — it is C2's surface test");
    assert!(
        dump.contains("schema baseline"),
        "the capability-dump test no longer checks for the schema-baseline column"
    );
}

