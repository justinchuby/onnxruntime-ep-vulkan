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
//! | `codes`   | array\<string\> | **every** check that failed, in canonical order; `[]` when claimed |
//! | `reasons` | array\<string\> | the sentence for each entry in `codes`, same order              |
//! | `unevaluated` | array\<string\> | checks that could not run at all (only for an unregistered op) |
//! | `shape_class` | string      | `static` \| `extents-symbolic` \| `rank-unknown` \| `data-dependent` |
//! | `predicate_ok` | bool           | the row's predicate accepts this node *today*, ignoring status  |
//! | `predicate_ok_runtime_extents` | bool | the row's predicate accepts it if extents arrive at `Compute` |
//! | `proof_key` | string \| null | the §8.9 proof key for this node; `null` when the op has no registry row at all |
//! | `ledger_hit` | bool           | whether the proof ledger held an entry under that key |
//! | `input_shapes` | array \| null | per-input shape **as ORT reported it**; a negative entry is symbolic, an entry is `null` when ORT gave no shape |
//! | `output_shapes` | array \| null | the same, per output |
//!
//! # `proof_key` is what makes the ledger bootstrappable
//!
//! It is recorded for **every** node with a row, claimed or declined, `Ready` or `Staged`. A key
//! computed only for nodes that already pass is a key that can never bootstrap: the whole point
//! of the §8.9.4 escape hatch is that you enable a form *in order to prove it*, and you cannot
//! enable a key you have no way to learn. `rust/tools/gen_proof_ledger.py` reads this field from
//! a claim-log pass and turns it into the allowlist for the proving pass.
//!
//! # `code` is first-match; `codes` is the whole truth
//!
//! `code` names the **first** failing check, not the only one. That is what `DESIGN.md` R8 is
//! about, and it is not a cosmetic distinction: on Phi-3.5, `staged: 100` was read as "100 nodes
//! that only need a kernel" when in fact those nodes were rejected at the status check and never
//! reached the shape check, so their shape viability was unknown. `codes`, `shape_class` and
//! `predicate_ok_runtime_extents` exist so that no future census has to guess.
//!
//! `shape_class` is computed from the node's edges **without consulting its registry row**, which
//! is the only way a staged node's shape viability can be known at all.
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
//!
//! **The path is re-read on every decision, not latched once per process.** It used to be a
//! `OnceLock`, on the reasoning that an environment variable is set before a process starts and
//! never changes. That reasoning is wrong for the one caller this module exists to serve: a pytest
//! process loads the EP once and then runs many tests, and `_models.is_vulkan_claimed` sets the
//! variable *per call*, around a single session. With a `OnceLock` the very first claim decision
//! in the process — made by whatever fixture happened to create a session first — latched `None`,
//! and every subsequent probe found no file. Because the reader treats a missing file as "not
//! claimed" (conservatively, and reasonably), the failure did not surface as a broken diagnostic:
//! it surfaced as `test_barrier_parity` skipping with *"Add is not yet Ready — the EP did not
//! claim this node form"*, at the same time as `test_add_is_claimed` passed on the same op. Two of
//! our own tests contradicting each other is what made it visible; a plausible-sounding skip on
//! its own would have hidden indefinitely. See `docs/OP_COVERAGE.md` §9.2.2.
//!
//! Re-reading costs one `getenv` per claim decision — per *node*, during `GetCapability`, not per
//! dispatch — which is nothing next to the schema lookups the same call already performs.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use crate::registry::{DeclineCode, DeclineReason};

/// The shape reading ORT gave for one node, as `(inputs, outputs)`.
///
/// Three states are distinct and all three occur: `None` means the caller took no reading at all
/// (logging was off on the collecting path); `Some` with a `None` element means ORT reported no
/// type for that edge; `Some` with a `Some` element carries the dims, where a negative dim is
/// symbolic. An *empty* dim list is the ambiguous case — see the module docs on unknown rank.
type EdgeReading<'a> = Option<(
    &'a [Option<crate::registry::EdgeType>],
    &'a [Option<crate::registry::EdgeType>],
)>;

/// The environment variable that turns the record on, by naming the file to append to.
pub const CLAIM_LOG_ENV: &str = "ONNXRUNTIME_EP_VULKAN_CLAIM_LOG";

/// The path to append to, as of *now*.
///
/// Deliberately not memoised: see the module docs. A test that enables the record mid-process
/// must be able to, because that is the only way the record is ever used.
fn path() -> Option<PathBuf> {
    std::env::var_os(CLAIM_LOG_ENV)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

/// Whether the claim record is enabled right now.
///
/// Exposed so that callers can skip building the strings when nobody is listening.
pub fn enabled() -> bool {
    path().is_some()
}

/// The append handle, opened lazily and kept open, together with the path it belongs to.
///
/// `Mutex` rather than a per-record open: ORT calls `GetCapability` from one thread today, but
/// nothing in the ABI promises that, and interleaved half-lines from two threads would produce a
/// file that is not valid JSON Lines.
///
/// The path is stored alongside the handle so that a caller which points the variable at a
/// *different* file mid-process — which `is_vulkan_claimed` does, once per probe, using a
/// pid-stamped name — gets its own file rather than the first one we happened to open.
fn sink() -> &'static Mutex<Option<(PathBuf, File)>> {
    static SINK: OnceLock<Mutex<Option<(PathBuf, File)>>> = OnceLock::new();
    SINK.get_or_init(|| Mutex::new(None))
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

/// Render one JSON array of already-escaped-as-strings items.
fn json_str_array<'a>(items: impl Iterator<Item = &'a str>) -> String {
    let body: Vec<String> = items.map(|s| format!("\"{}\"", escape(s))).collect();
    format!("[{}]", body.join(","))
}

/// Render the edge shapes **as ORT reported them to us**.
///
/// # Why this is in the record and not derived by the reader
///
/// It was derived by the reader, once, and the reader was wrong. Sizing the `MatMul` claim
/// predicate needed BERT-SQuAD-12's `MatMul` shape space, and `rust/tools/probe_matmul_shape_space.py`
/// read it off `onnx.shape_inference`, which on that model resolves a shape for only **384 of
/// 1165** value infos and **none** of the 98 `MatMul` edges. Read that way, `MatMul` is entirely
/// unclaimable on BERT. The EP's own `shape_class` — already in this record — said **94 of 95
/// static**, because ORT's inference runs the constant-folding and data propagation that the
/// standalone pass does not. The two readings disagreed about whether a kernel was worth writing
/// at all.
///
/// `shape_class` was enough to catch the disagreement and not enough to resolve it: a class is
/// not a rank and not an extent, and a kernel is sized on extents. So the extents go in the
/// record, from the same `NodeView` the predicate is about to be asked about, and the census
/// stops having a second implementation of shape inference to be wrong in.
///
/// Three states are distinguished, because collapsing them is how "unknown" becomes "fine":
/// `null` for an edge ORT reported no `OrtValueInfo` for (an omitted optional input), `null` for
/// a shape inference produced nothing for, and a list for a shape it did — in which a **negative**
/// entry is symbolic, matching [`EdgeType`](crate::registry::EdgeType)'s own convention.
///
/// Costs nothing when the log is off: the caller only builds the vectors under `enabled()`.
fn json_shapes(edges: &[Option<crate::registry::EdgeType>]) -> String {
    let body: Vec<String> = edges
        .iter()
        .map(|e| match e.as_ref().and_then(|e| e.shape.as_ref()) {
            None => "null".to_string(),
            Some(dims) => {
                let ds: Vec<String> = dims.iter().map(i64::to_string).collect();
                format!("[{}]", ds.join(","))
            }
        })
        .collect();
    format!("[{}]", body.join(","))
}

/// Render one full-audit record.
///
/// Extends [`line`] rather than replacing it: `code` and `reason` keep first-match semantics so
/// every existing assertion still means what it meant, and the complete picture arrives in the
/// new fields. See the module docs on why first-match alone is not usable for planning.
pub(crate) fn audit_line(
    qualified: &str,
    node: &str,
    opset: i32,
    audit: &crate::registry::ClaimAudit,
    edges: EdgeReading<'_>,
) -> String {
    let base = line(
        qualified,
        node,
        opset,
        audit.primary.as_ref().map_or(Ok(()), Err),
    );
    let codes = json_str_array(
        audit
            .failures
            .iter()
            .map(|r| DeclineCode::of_reason(r).map_or("other", DeclineCode::tag)),
    );
    let reasons = json_str_array(audit.failures.iter().map(std::convert::AsRef::as_ref));
    let unevaluated = json_str_array(audit.unevaluated.iter().copied());
    let (inputs, outputs) = match edges {
        Some((i, o)) => (json_shapes(i), json_shapes(o)),
        None => ("null".to_string(), "null".to_string()),
    };
    format!(
        "{},\"codes\":{},\"reasons\":{},\"unevaluated\":{},\"shape_class\":\"{}\",\
         \"predicate_ok\":{},\"predicate_ok_runtime_extents\":{},\"proof_key\":{},\
         \"ledger_hit\":{},\"input_shapes\":{},\"output_shapes\":{}}}",
        base.trim_end_matches('}'),
        codes,
        reasons,
        unevaluated,
        audit.shape_class.tag(),
        audit.predicate_ok,
        audit.predicate_ok_with_runtime_extents,
        match &audit.proof_key {
            Some(k) => format!("\"{}\"", escape(&k.0)),
            None => "null".to_string(),
        },
        audit.ledger_hit,
        inputs,
        outputs,
    )
}

/// Append one decision to the record, if the record is enabled.
///
/// Infallible by design: see the module docs.
pub fn record(qualified: &str, node: &str, opset: i32, outcome: Result<(), &DeclineReason>) {
    let Some(want) = path() else {
        return;
    };
    record_to(want, &line(qualified, node, opset, outcome));
}

/// Append one full-audit decision to the record, if the record is enabled.
pub fn record_audit(
    qualified: &str,
    node: &str,
    opset: i32,
    audit: &crate::registry::ClaimAudit,
    edges: EdgeReading<'_>,
) {
    let Some(want) = path() else {
        return;
    };
    record_to(want, &audit_line(qualified, node, opset, audit, edges));
}

/// Append one already-rendered line to `want`, reopening the sink if the path has changed.
///
/// Split out from [`record`] so that the redirect behaviour — the half of the fix that is not
/// about environment variables — can be tested without mutating the process environment, which
/// is unsound under a threaded test runner.
fn record_to(want: PathBuf, text: &str) {
    if let Ok(mut guard) = sink().lock() {
        let stale = !matches!(guard.as_ref(), Some((have, _)) if *have == want);
        if stale {
            // A failed open leaves `None`, so the next decision retries. Retrying is the right
            // behaviour for a transient failure and costs nothing for a permanent one.
            *guard = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&want)
                .ok()
                .map(|f| (want, f));
        }
        if let Some((_, file)) = guard.as_mut() {
            // Both errors are deliberately dropped: a full disk must not fail an inference run.
            let _ = writeln!(file, "{text}");
            let _ = file.flush();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::common::claim::ShapeClass;
    use crate::registry::ClaimAudit;
    use std::borrow::Cow;

    fn reason(code: DeclineCode, detail: &str) -> DeclineReason {
        crate::registry::decline(code, detail)
    }

    fn audit(failures: Vec<DeclineReason>, shape_class: ShapeClass) -> ClaimAudit {
        ClaimAudit {
            primary: failures.first().cloned(),
            failures,
            unevaluated: Vec::new(),
            shape_class,
            predicate_ok: false,
            predicate_ok_with_runtime_extents: false,
            proof_key: None,
            ledger_hit: false,
            proof_state: crate::registry::ProofState::Unproven,
        }
    }

    #[test]
    fn the_audit_line_keeps_first_match_semantics_for_code() {
        // Every existing assertion reads `code`. It must keep meaning "the first failing check"
        // even though the record now also carries the complete set.
        let a = audit(
            vec![
                reason(DeclineCode::Staged, "kernel not written"),
                reason(DeclineCode::DynamicShape, "symbolic extent"),
            ],
            ShapeClass::ExtentsSymbolic,
        );
        let l = audit_line(
            "com.microsoft::SkipSimplifiedLayerNormalization",
            "n0",
            1,
            &a,
            None,
        );
        assert!(l.contains(r#""code":"staged""#), "{l}");
        assert!(l.contains(r#""claimed":false"#), "{l}");
    }

    #[test]
    fn the_audit_line_reports_every_failing_check_not_only_the_first() {
        // DESIGN.md R8. This is the record that makes a staged node's shape viability knowable:
        // without it the node above looks like "needs a kernel" and nothing more.
        let a = audit(
            vec![
                reason(DeclineCode::Staged, "kernel not written"),
                reason(DeclineCode::DynamicShape, "symbolic extent"),
            ],
            ShapeClass::ExtentsSymbolic,
        );
        let l = audit_line(
            "com.microsoft::SkipSimplifiedLayerNormalization",
            "n0",
            1,
            &a,
            None,
        );
        assert!(l.contains(r#""codes":["staged","dynamic-shape"]"#), "{l}");
        assert!(l.contains(r#""shape_class":"extents-symbolic""#), "{l}");
        assert!(l.contains(r#""predicate_ok":false"#), "{l}");
        assert!(l.contains(r#""predicate_ok_runtime_extents":false"#), "{l}");
    }

    #[test]
    fn a_claimed_node_records_an_empty_failure_set() {
        let a = audit(Vec::new(), ShapeClass::Static);
        let l = audit_line("Add", "n0", 14, &a, None);
        assert!(l.contains(r#""claimed":true"#), "{l}");
        assert!(l.contains(r#""code":null"#), "{l}");
        assert!(l.contains(r#""codes":[]"#), "{l}");
        assert!(l.contains(r#""shape_class":"static""#), "{l}");
    }

    /// The record carries the shapes **ORT** reported, because the census read them elsewhere and
    /// got a different answer.
    ///
    /// On BERT-SQuAD-12 a standalone `onnx.shape_inference` pass resolves no shape at all for any
    /// of the 98 `MatMul` edges, while ORT resolves 94 of 95 as fully static. A census that sizes
    /// a kernel from the first reading declines to write a kernel the second reading says is
    /// worth writing. Recording the EP's own reading is what removes the second implementation.
    #[test]
    fn the_record_carries_the_shapes_ort_reported_and_distinguishes_absent_from_unknown() {
        use crate::engine::DType;
        use crate::registry::EdgeType;

        let a = audit(Vec::new(), ShapeClass::ExtentsSymbolic);
        let inputs = vec![
            // A fully static edge.
            Some(EdgeType {
                dtype: Some(DType::F32),
                shape: Some(vec![256, 768]),
            }),
            // A symbolic leading extent, which the record spells as a negative, matching
            // `EdgeType`'s own convention rather than inventing a second one.
            Some(EdgeType {
                dtype: Some(DType::F32),
                shape: Some(vec![-1, 768]),
            }),
            // Present, typed, but shape inference produced nothing.
            Some(EdgeType {
                dtype: Some(DType::F32),
                shape: None,
            }),
            // An omitted optional input: ORT reported no `OrtValueInfo` at all.
            None,
        ];
        let outputs = vec![Some(EdgeType {
            dtype: Some(DType::F32),
            shape: Some(vec![256, 768]),
        })];
        let l = audit_line("MatMul", "n0", 9, &a, Some((&inputs, &outputs)));
        assert!(
            l.contains(r#""input_shapes":[[256,768],[-1,768],null,null]"#),
            "{l}"
        );
        assert!(l.contains(r#""output_shapes":[[256,768]]"#), "{l}");
    }

    /// A caller with no edges to offer must render `null`, not `[]`.
    ///
    /// `[]` is a claim that the node has no inputs. `null` is the absence of a reading. A census
    /// that cannot tell those apart would count every unrecorded node as a zero-input node.
    #[test]
    fn no_edge_reading_renders_null_rather_than_an_empty_list() {
        let a = audit(Vec::new(), ShapeClass::Static);
        let l = audit_line("Add", "n0", 14, &a, None);
        assert!(l.contains(r#""input_shapes":null"#), "{l}");
        assert!(l.contains(r#""output_shapes":null"#), "{l}");
        let empty: Vec<Option<crate::registry::EdgeType>> = Vec::new();
        let l2 = audit_line("Add", "n0", 14, &a, Some((&empty, &empty)));
        assert!(l2.contains(r#""input_shapes":[]"#), "{l2}");
    }

    #[test]
    fn an_unregistered_op_records_what_could_not_be_evaluated() {
        // Without a row there is no opset window, no schema, no status and no predicate. Saying
        // so is the difference between "these checks passed" and "these checks never ran", which
        // is the same distinction R8 is about, one level down.
        let mut a = audit(
            vec![reason(DeclineCode::NotRegistered, "no handler")],
            ShapeClass::Static,
        );
        a.unevaluated = vec!["opset", "contrib-schema", "status", "predicate"];
        let l = audit_line("Gather", "n0", 13, &a, None);
        assert!(
            l.contains(r#""unevaluated":["opset","contrib-schema","status","predicate"]"#),
            "{l}"
        );
    }

    #[test]
    fn the_audit_line_is_one_line_of_valid_json_lines() {
        let a = audit(
            vec![reason(
                DeclineCode::DynamicShape,
                "has a \"symbolic\" extent\nwith a newline",
            )],
            ShapeClass::ExtentsSymbolic,
        );
        let l = audit_line("Mul", "node\"with\"quotes", 14, &a, None);
        assert!(!l.contains('\n'), "a record must never span lines: {l}");
        assert_eq!(l.matches("{\"op\"").count(), 1);
        assert!(l.starts_with('{') && l.ends_with('}'), "{l}");
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
        let why = reason(
            DeclineCode::NotRegistered,
            "no Vulkan handler is registered",
        );
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
        assert!(
            l.contains("max_claim_ops"),
            "the detail is still recorded: {l}"
        );
    }

    #[test]
    fn quotes_and_control_characters_are_escaped() {
        // Node names come from the model, so they are attacker-controlled in the same sense any
        // input is: a name containing `"` must not produce a file that is not valid JSON.
        let l = line("Add", "he said \"hi\"\n\tand left", 14, Ok(()));
        assert!(l.contains(r#"he said \"hi\"\n\tand left"#), "{l}");
        assert_eq!(
            l.matches('\n').count(),
            0,
            "no raw newline may reach the file"
        );
    }

    #[test]
    fn the_record_is_off_unless_the_env_var_names_a_file() {
        // The default must be zero cost and zero side effects: this runs inside every
        // `GetCapability` of every session in someone else's process.
        if std::env::var_os(CLAIM_LOG_ENV).is_none() {
            assert!(!enabled());
        }
    }

    #[test]
    fn pointing_the_record_at_a_new_file_mid_process_redirects_it() {
        // The regression this pins cost us a day of confusion: the sink used to be opened once
        // per process from a `OnceLock` path, so the *second* consumer in a process — every
        // `is_vulkan_claimed` probe after the first session — silently wrote nothing. The reader
        // treats an absent file as "not claimed", so the broken diagnostic presented as
        // `test_barrier_parity` skipping with "Add is not yet Ready" while `test_add_is_claimed`
        // passed on the same op in the same run.
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join(format!("claim_log_test_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let first = dir.join("first.jsonl");
        let second = dir.join("second.jsonl");
        let _ = std::fs::remove_file(&first);
        let _ = std::fs::remove_file(&second);

        record_to(first.clone(), "one");
        record_to(second.clone(), "two");
        record_to(first.clone(), "three");

        let a = std::fs::read_to_string(&first).unwrap_or_default();
        let b = std::fs::read_to_string(&second).unwrap_or_default();
        assert!(a.contains("one"), "first file lost its first line: {a:?}");
        assert!(
            b.contains("two"),
            "the second path was never opened — this is the OnceLock defect: {b:?}"
        );
        assert!(
            a.contains("three"),
            "switching back must reopen, not stay on the second file: {a:?}"
        );
        assert!(!b.contains("three"), "lines leaked across files: {b:?}");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
