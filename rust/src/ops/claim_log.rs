//! A machine-readable record of every claim decision, for tests and for measurement.
//!
//! **Mouse owns this file.** It exists because of a specific failure mode Trinity hit: her C1
//! regression test (`tests/ops/test_domain_regression.py`) can only assert *structurally* — that
//! the EP claimed zero nodes — and a zero-node assertion cannot tell apart
//!
//! * "declined because no row is registered for it" (correct, and the thing C1 is about),
//! * "declined because the dtype was wrong" (also a decline, but proves nothing about C1), and
//! * "crashed before it ever got to the claim predicate" (a bug that the test would pass).
//!
//! [`crate::registry::DeclineCode`] already distinguishes all of those. It was simply invisible
//! from outside the process. This module is the surface that makes it visible.
//!
//! # The contract
//!
//! Set `ONNXRUNTIME_EP_VULKAN_CLAIM_LOG` to a file path. Every call to
//! [`crate::registry::claim_decision`] appends **one JSON object on one line** (JSON Lines) to
//! that file:
//!
//! ```jsonc
//! {"op":"com.microsoft::NotARealOp","node":"n0","opset":1,"claimed":false,
//!  "code":"not-registered","reason":"[not-registered] no Vulkan handler is registered for ..."}
//! ```
//!
//! | field     | type            | meaning                                                        |
//! |-----------|-----------------|----------------------------------------------------------------|
//! | `op`      | string          | domain-qualified op type, e.g. `Add` or `com.microsoft::MatMulNBits` |
//! | `node`    | string          | the node's name in the graph; `""` when ONNX did not give it one |
//! | `opset`   | number          | the node's `since_version`; `0` when ORT did not resolve one    |
//! | `claimed` | bool            | `true` iff this EP took the node                                |
//! | `code`    | string \| null  | the [`DeclineCode`](crate::registry::DeclineCode) tag; `null` when claimed |
//! | `reason`  | string \| null  | the full human-readable decline sentence; `null` when claimed   |
//!
//! Assertions this supports, which are the two Trinity asked for:
//!
//! ```python
//! assert claims["com.microsoft::NotARealOp"].code == "not-registered"   # declined, and *why*
//! assert claims["Add"].claimed                                          # and the positive case
//! ```
//!
//! # Design notes
//!
//! **JSON Lines, appended and flushed per record, rather than a report written at teardown.**
//! There is no point in the plugin-EP lifecycle where we are reliably told "the session is over
//! and you may now write your diagnostics" — `ReleaseEp` is not guaranteed to run before the host
//! inspects the file, and a test that reads a report the EP has not flushed yet is a flaky test.
//! Appending one self-contained line per decision means the file is valid and complete after every
//! single decision, and the reader needs no lifecycle knowledge at all.
//!
//! **It lives here rather than in `ep.rs`.** The aggregation loop in `ep.rs` (Tank's) already
//! collapses declines to one reason per op *type*, which is the right thing for a human reading
//! claim-debug output and the wrong thing for a test that wants a specific node. Recording inside
//! `claim_decision` instead means no change to the boundary layer is needed, and every caller of
//! the registry — the EP, `epctl`, a future Niobe measurement harness — gets the same record for
//! free.
//!
//! **Two declines do not appear here**, both by construction: `ep.rs` short-circuits nodes inside
//! a control-flow body and nodes excluded by `ep.max_claim_ops` *before* asking the registry, so
//! those never reach this module. That is deliberate — neither is a statement about op support —
//! but it means the record answers "what did the registry decide", not "what did the EP do", and a
//! test that sets `max_claim_ops` must not expect a line for the excluded nodes.
//!
//! **Nothing here can fail a run.** Every I/O error is swallowed. A diagnostic that can break
//! inference is worse than no diagnostic, and this code runs inside a C ABI callback where a panic
//! would be undefined behaviour.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use crate::registry::{DeclineCode, DeclineReason};

/// The environment variable that turns the record on, by naming the file to append to.
pub const CLAIM_LOG_ENV: &str = "ONNXRUNTIME_EP_VULKAN_CLAIM_LOG";

/// The path, read exactly once per process.
fn path() -> Option<&'static PathBuf> {
    static PATH: OnceLock<Option<PathBuf>> = OnceLock::new();
    PATH.get_or_init(|| {
        std::env::var_os(CLAIM_LOG_ENV)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    })
    .as_ref()
}

/// Whether the claim record is enabled for this process.
///
/// Exposed so that callers can skip building the strings when nobody is listening.
pub fn enabled() -> bool {
    path().is_some()
}

/// The append handle, opened lazily and kept open.
///
/// `Mutex` rather than a per-record open: ORT calls `GetCapability` from one thread today, but
/// nothing in the ABI promises that, and interleaved half-lines from two threads would produce a
/// file that is not valid JSON Lines.
fn sink() -> &'static Mutex<Option<File>> {
    static SINK: OnceLock<Mutex<Option<File>>> = OnceLock::new();
    SINK.get_or_init(|| {
        let file = path().and_then(|p| OpenOptions::new().create(true).append(true).open(p).ok());
        Mutex::new(file)
    })
}

/// Escape a string for inclusion in a JSON string literal.
///
/// Decline reasons are formatted from op names, dtype lists and attribute values, none of which
/// are guaranteed to be free of quotes or backslashes — a model author chooses node names.
fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// Render one record. Split out from [`record`] so the format can be unit-tested with no file.
pub(crate) fn line(
    qualified: &str,
    node: &str,
    opset: i32,
    outcome: Result<(), &DeclineReason>,
) -> String {
    let (claimed, code, reason) = match outcome {
        Ok(()) => (true, None, None),
        Err(why) => (
            false,
            DeclineCode::of_reason(why).map(DeclineCode::tag),
            Some(why.as_ref()),
        ),
    };
    let code = match code {
        Some(t) => format!("\"{t}\""),
        None => "null".to_string(),
    };
    let reason = match reason {
        Some(r) => format!("\"{}\"", escape(r)),
        None => "null".to_string(),
    };
    format!(
        "{{\"op\":\"{}\",\"node\":\"{}\",\"opset\":{},\"claimed\":{},\"code\":{},\"reason\":{}}}",
        escape(qualified),
        escape(node),
        opset,
        claimed,
        code,
        reason,
    )
}

/// Append one decision to the record, if the record is enabled.
///
/// Infallible by design: see the module docs.
pub fn record(qualified: &str, node: &str, opset: i32, outcome: Result<(), &DeclineReason>) {
    if !enabled() {
        return;
    }
    let text = line(qualified, node, opset, outcome);
    if let Ok(mut guard) = sink().lock()
        && let Some(file) = guard.as_mut()
    {
        // Both errors are deliberately dropped: a full disk must not fail an inference run.
        let _ = writeln!(file, "{text}");
        let _ = file.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Cow;

    fn reason(code: DeclineCode, detail: &str) -> DeclineReason {
        crate::registry::decline(code, detail)
    }

    #[test]
    fn a_claim_records_no_code_and_no_reason() {
        let l = line("Add", "n0", 14, Ok(()));
        assert_eq!(
            l,
            r#"{"op":"Add","node":"n0","opset":14,"claimed":true,"code":null,"reason":null}"#
        );
    }

    #[test]
    fn a_decline_records_the_machine_readable_code() {
        // This is exactly Trinity's C1 assertion: the decline must be attributable, not merely
        // present.
        let why = reason(DeclineCode::NotRegistered, "no Vulkan handler is registered");
        let l = line("com.microsoft::NotARealOp", "n0", 1, Err(&why));
        assert!(
            l.contains(r#""code":"not-registered""#),
            "the C1 test asserts on this exact token: {l}"
        );
        assert!(l.contains(r#""claimed":false"#));
    }

    #[test]
    fn every_decline_code_round_trips_through_the_record() {
        // The whole value of the surface is that `[contrib-schema]` and `[attribute]` stay
        // distinguishable from outside the process. If a code ever failed to render, the two
        // would silently merge into `null` and a test asserting on them would be asserting on
        // nothing.
        for code in DeclineCode::ALL {
            let why = reason(*code, "detail");
            let l = line("Some::Op", "n", 1, Err(&why));
            assert!(
                l.contains(&format!(r#""code":"{}""#, code.tag())),
                "{code:?} did not round-trip: {l}"
            );
        }
    }

    #[test]
    fn a_reason_from_outside_the_registry_records_a_null_code_not_a_wrong_one() {
        // `ep.rs` builds two reasons of its own (control-flow bodies, `max_claim_ops`) that do
        // not carry a tag. Guessing a code for them would be worse than admitting we have none.
        let why: DeclineReason = Cow::Borrowed("excluded by ep.max_claim_ops");
        let l = line("Add", "n0", 14, Err(&why));
        assert!(l.contains(r#""code":null"#), "{l}");
        assert!(l.contains("max_claim_ops"), "the detail is still recorded: {l}");
    }

    #[test]
    fn quotes_and_control_characters_are_escaped() {
        // Node names come from the model, so they are attacker-controlled in the same sense any
        // input is: a name containing `"` must not produce a file that is not valid JSON.
        let l = line("Add", "he said \"hi\"\n\tand left", 14, Ok(()));
        assert!(l.contains(r#"he said \"hi\"\n\tand left"#), "{l}");
        assert_eq!(l.matches('\n').count(), 0, "no raw newline may reach the file");
    }

    #[test]
    fn the_record_is_off_unless_the_env_var_names_a_file() {
        // The default must be zero cost and zero side effects: this runs inside every
        // `GetCapability` of every session in someone else's process.
        if std::env::var_os(CLAIM_LOG_ENV).is_none() {
            assert!(!enabled());
        }
    }
}
