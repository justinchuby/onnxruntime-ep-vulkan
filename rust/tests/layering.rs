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
