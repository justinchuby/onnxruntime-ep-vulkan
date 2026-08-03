//! Session-creation disclosure of claimed-form correctness evidence — DESIGN.md §8.9.7, RAI-009.
//!
//! # What this exists to prevent
//!
//! A build can claim an op whose correctness is **`UNMEASURED`** (nothing in the proof ledger
//! ever compared it against the CPU EP) or known **`DIVERGENT`** (something did, and it did not
//! match). Today both are reachable: `UNMEASURED` via the
//! `ONNXRUNTIME_EP_VULKAN_CLAIM_UNPROVEN` escape hatch, `DIVERGENT` via that same hatch over a
//! ledger line whose verdict is not `MATCH`. In both cases the session is created, every
//! claim-time check passes, and the user learns about it — if at all — from a wrong answer some
//! time later.
//!
//! Rai's RAI-008 discharge condition, stated plainly: *a user creating a session against a build
//! that would claim ops whose correctness is `UNMEASURED` or known `DIVERGENT` must be told at
//! session creation, not left to discover it from a wrong answer later.*
//!
//! # Why it goes through ORT's sink
//!
//! Same reason as [`crate::ep`]'s broken-commitment WARN (Ruling 2): the channel a user with ORT
//! logging configured is already watching is ORT's own logger. A WARN in this crate's private
//! `log` facade is invisible to exactly the audience that matters, and is additionally gated by
//! `RUST_LOG`, which the user did not set because they configured *ORT's* logging.
//!
//! # Why the disclosure is a pair, and per form
//!
//! The INFO ("these claimed forms are proven, by this artifact, on this device, against this ORT
//! build") and the WARN ("these claimed forms are not") travel down the same channel so they read
//! as a pair. Both are emitted **per distinct proof key**, never per node: `registry.rs` already
//! records that a disclosure repeated 365 times is a disclosure a reader learns to filter, and
//! the coordinator's scope ruling on the runtime WARN was explicit — no per-node noise.
//!
//! # Evidence class
//!
//! Under Link's `PLANTED`/`OBSERVED` axis this mechanism's red arm is **`PLANTED`**. Both the
//! `UNMEASURED` and the `DIVERGENT` arms are reachable only by deliberately enabling a form the
//! evidence does not cover; no production build of this repository has ever produced one. That is
//! fine — it just must not be recorded as more than it is.

use std::collections::BTreeMap;

use crate::counters;
use crate::logging;
use crate::registry::{self, LedgerEntry, ProofKey};

/// The log target both halves of the disclosure travel under.
pub const TARGET: &str = "VulkanEP";

/// What the proof ledger says about one claimed form.
///
/// The four states are deliberately not three. `Unmeasured` and `Divergent` must not collapse
/// into one another — "nothing measured this" and "something measured this and it disagreed" call
/// for different action by the reader — and `LedgerFaulted` is an **instrument** state (R13): it
/// is a statement about the ledger file, not a finding about the form. Reporting a faulted ledger
/// as `Unmeasured` would be reporting an instrument error as a detection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FormEvidence {
    /// The ledger holds a `MATCH` for this key, with provenance.
    Proven(&'static LedgerEntry),
    /// No entry under this key. Nothing has ever compared this form against the CPU EP.
    Unmeasured,
    /// An entry exists and its verdict is not `MATCH`. Carries that verdict verbatim.
    Divergent(String),
    /// The ledger itself is unusable, so nothing is known about *any* form from it.
    LedgerFaulted,
}

impl FormEvidence {
    /// The token this evidence contributes to the counters artifact.
    pub fn token(&self) -> &'static str {
        match self {
            FormEvidence::Proven(_) => "PROVEN",
            FormEvidence::Unmeasured => "UNMEASURED",
            FormEvidence::Divergent(_) => "DIVERGENT",
            FormEvidence::LedgerFaulted => "LEDGER-FAULTED",
        }
    }

    /// Whether this evidence obliges the session-creation WARN.
    pub fn warrants_warning(&self) -> bool {
        !matches!(self, FormEvidence::Proven(_))
    }
}

/// What the ledger says about `key`.
///
/// Order matters. A recorded non-`MATCH` verdict is consulted **first**, because it is the most
/// specific thing known and because any such line also faults the whole ledger — checking faults
/// first would turn every `DIVERGENT` form into a `LEDGER-FAULTED` one and lose the distinction
/// this function exists to preserve.
pub fn evidence_for(key: &ProofKey) -> FormEvidence {
    let ledger = registry::ledger();
    if let Some(verdict) = ledger.demotion_for(key) {
        return FormEvidence::Divergent(verdict.to_string());
    }
    if let Some(entry) = ledger.get(key) {
        return FormEvidence::Proven(entry);
    }
    if !ledger.faults.is_empty() {
        return FormEvidence::LedgerFaulted;
    }
    FormEvidence::Unmeasured
}

/// One distinct form this session is about to claim.
#[derive(Clone, Debug)]
pub struct ClaimedForm {
    /// Qualified op type, e.g. `ai.onnx::Add`. Named in the disclosure so a reader who does not
    /// speak proof-key can still find the op.
    pub op_type: String,
    /// The proof key the claim gate looked up. `None` for a claim granted on a path that does not
    /// produce one (control-flow parents), which is itself disclosed rather than skipped.
    pub key: Option<ProofKey>,
    /// How many nodes of this form the session claimed.
    pub nodes: usize,
}

/// What one disclosure did, returned so a test can assert on it without reading a log.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Disclosure {
    /// Distinct claimed forms whose evidence is a ledger `MATCH`.
    pub proven: usize,
    /// Distinct claimed forms with no evidence at all.
    pub unmeasured: usize,
    /// Distinct claimed forms whose recorded verdict is not `MATCH`.
    pub divergent: usize,
    /// Distinct claimed forms whose evidence could not be read because the ledger is faulted.
    pub ledger_faulted: usize,
    /// Whether a WARN was emitted at all.
    pub warned: bool,
    /// Whether that WARN reached ORT's own logger. `false` with `warned == true` means the WARN
    /// exists only on our stderr — delivered to nobody who matters.
    pub warn_reached_ort_sink: bool,
    /// Whether the proven-forms INFO was emitted at all (i.e. anything was proven).
    pub informed: bool,
    /// Whether that INFO reached ORT's own logger. Tracked for the same reason as the WARN's
    /// flag, and added later than it: until 2026-08-03 nothing recorded whether the INFO half of
    /// the §8.9.7 pair was delivered, so an artifact could show a fully-proven claim set and say
    /// nothing about whether the user was ever told what proved it.
    pub info_reached_ort_sink: bool,
}

impl Disclosure {
    /// Forms that obliged the WARN.
    pub fn unproven(&self) -> usize {
        self.unmeasured + self.divergent + self.ledger_faulted
    }
}

/// Emit the §8.9.7 session-creation disclosure for the forms this session is claiming.
///
/// Emits, through ORT's own sink:
/// * one INFO summarising the proven forms and naming the artifact each was proven by;
/// * one WARN, **only if** at least one claimed form is `UNMEASURED` / `DIVERGENT` /
///   `LEDGER-FAULTED`, naming every such form and what will happen.
///
/// The WARN's absence on an all-proven claim set is the negative polarity of the control and is
/// asserted directly in this module's tests: a WARN that cannot be shown *not* to fire is a
/// printed opinion, not a detector.
pub fn disclose_claimed_forms(forms: &[ClaimedForm]) -> Disclosure {
    let mut d = Disclosure::default();
    let mut proven_lines: Vec<String> = Vec::new();
    let mut unproven_lines: Vec<String> = Vec::new();

    for form in forms {
        let Some(key) = &form.key else {
            // No proof key was produced for this claim. That is not evidence of correctness and
            // must not be counted as proven; it is unmeasured by this instrument's own admission.
            d.unmeasured += 1;
            unproven_lines.push(format!(
                "{} x{} [UNMEASURED: the claim gate produced no proof key for this form]",
                form.op_type, form.nodes
            ));
            continue;
        };
        match evidence_for(key) {
            FormEvidence::Proven(entry) => {
                d.proven += 1;
                proven_lines.push(format!(
                    "{} x{} [{}] proven by {} on {} against {}",
                    form.op_type, form.nodes, key.0, entry.artifact, entry.device, entry.ort_build
                ));
            }
            FormEvidence::Unmeasured => {
                d.unmeasured += 1;
                unproven_lines.push(format!(
                    "{} x{} [{}] UNMEASURED: no differential proof exists for this form",
                    form.op_type, form.nodes, key.0
                ));
            }
            FormEvidence::Divergent(verdict) => {
                d.divergent += 1;
                unproven_lines.push(format!(
                    "{} x{} [{}] DIVERGENT: the recorded verdict for this form is {verdict}, \
                     not MATCH",
                    form.op_type, form.nodes, key.0
                ));
            }
            FormEvidence::LedgerFaulted => {
                d.ledger_faulted += 1;
                unproven_lines.push(format!(
                    "{} x{} [{}] LEDGER-FAULTED: the proof ledger is unusable, so this form's \
                     correctness is unknown rather than absent",
                    form.op_type, form.nodes, key.0
                ));
            }
        }
    }

    if !proven_lines.is_empty() {
        d.informed = true;
        d.info_reached_ort_sink = logging::info_through_ort_sink(
            TARGET,
            &format!(
                "session claims {} proven form(s) [§8.9.7]: {}",
                proven_lines.len(),
                proven_lines.join("; ")
            ),
        );
    }

    if !unproven_lines.is_empty() {
        // Quote the condition, never only the count (R13). A reader who is told "3 forms are
        // unproven" cannot act; a reader who is told which three, and why each, can.
        let msg = format!(
            "[§8.9.7] this session will claim {} form(s) whose correctness is NOT established: \
             {}. Results produced by these forms are not backed by any differential proof and \
             may be silently wrong. They are claimed because \
             ONNXRUNTIME_EP_VULKAN_CLAIM_UNPROVEN names them; unset it to leave them on the CPU \
             EP.",
            unproven_lines.len(),
            unproven_lines.join("; ")
        );
        d.warned = true;
        d.warn_reached_ort_sink = logging::warn_through_ort_sink(TARGET, &msg);
    }

    counters::record_session_disclosure(counters::SessionDisclosure {
        proven: d.proven,
        unmeasured: d.unmeasured,
        divergent: d.divergent,
        ledger_faulted: d.ledger_faulted,
        warned: d.warned,
        warn_reached_ort_sink: d.warn_reached_ort_sink,
        informed: d.informed,
        info_reached_ort_sink: d.info_reached_ort_sink,
    });
    d
}

/// Disclose that this session claimed **nothing**, and the leading reasons why.
///
/// Not a warning: never-claimed ops running on the CPU EP is the plan, disclosed once in
/// aggregate. It is an INFO on the same channel so that a user who sees no Vulkan activity can
/// learn why from the log they are already reading, instead of from silence.
pub fn disclose_zero_claims(
    num_nodes: usize,
    declined: &BTreeMap<String, (usize, String, Vec<String>)>,
) {
    let mut ranked: Vec<(&String, usize, &String)> = declined
        .iter()
        .map(|(op, (n, why, _))| (op, *n, why))
        .collect();
    ranked.sort_by_key(|(op, n, _)| (std::cmp::Reverse(*n), (*op).clone()));
    let top: Vec<String> = ranked
        .iter()
        .take(5)
        .map(|(op, n, why)| format!("{op} x{n} ({why})"))
        .collect();
    let detail = if top.is_empty() {
        "no per-op reason was recorded".to_string()
    } else {
        top.join("; ")
    };
    let reached = logging::info_through_ort_sink(
        TARGET,
        &format!(
            "[§8.9.7] this session claims 0/{num_nodes} nodes; all work runs on the CPU EP. \
             Leading reasons: {detail}"
        ),
    );
    // A zero-claims disclosure is still a disclosure, and it is still an INFO whose delivery can
    // fail. Counting it keeps `session_disclosure_info_channel` a statement about the channel
    // rather than about which branch of §8.9.7 happened to run.
    counters::record_session_disclosure(counters::SessionDisclosure {
        informed: true,
        info_reached_ort_sink: reached,
        ..Default::default()
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allocator::ledger::test_lock;

    /// A key that is in the baked ledger, discovered from the ledger itself rather than pasted.
    ///
    /// A hardcoded key that drifts makes the positive arm silently inert — the same failure as a
    /// planted control that stops being planted. This reads the mechanism's own data.
    fn a_proven_key() -> ProofKey {
        let l = registry::ledger();
        assert!(
            l.faults.is_empty(),
            "ERROR(instrument): the baked proof ledger is faulted, so this test cannot \
             distinguish a proven form from an unproven one. Faults: {:?}",
            l.faults
        );
        l.entries()
            .first()
            .expect(
                "ERROR(instrument): the baked proof ledger holds no entries, so 'a claimed form \
                 that is proven' is unrepresentable and the negative arm below would pass \
                 vacuously",
            )
            .key
            .clone()
    }

    /// Positive polarity: a claimed form with no proof must warn.
    #[test]
    fn an_unmeasured_claimed_form_warns() {
        let _g = test_lock();
        counters::reset();
        let key = ProofKey::parse(
            "test.planted::NeverProven/1+/f16>f16/no_such_kernel/static/unmeasured-control",
        );
        assert_eq!(
            evidence_for(&key),
            FormEvidence::Unmeasured,
            "ERROR(instrument): the planted key is not unmeasured, so this arm proves nothing"
        );
        let d = disclose_claimed_forms(&[ClaimedForm {
            op_type: "test.planted::NeverProven".to_string(),
            key: Some(key),
            nodes: 3,
        }]);
        assert!(d.warned, "an UNMEASURED claimed form did not warn: {d:?}");
        assert_eq!(d.unmeasured, 1);
        assert_eq!(d.proven, 0);
    }

    /// Negative polarity: an all-proven claim set must **not** warn.
    ///
    /// Non-vacuity is asserted first — `proven >= 1` — because a silent disclosure over an empty
    /// claim set is the silence of a session that claimed nothing, which would let this arm pass
    /// for a reason unrelated to the thing it tests.
    #[test]
    fn an_all_proven_claim_set_does_not_warn() {
        let _g = test_lock();
        counters::reset();
        let key = a_proven_key();
        let d = disclose_claimed_forms(&[ClaimedForm {
            op_type: "ai.onnx::Proven".to_string(),
            key: Some(key),
            nodes: 7,
        }]);
        assert!(
            d.proven >= 1,
            "ERROR(instrument): the 'proven' arm claimed no proven form, so its silence is \
             vacuous: {d:?}"
        );
        assert_eq!(d.unproven(), 0, "expected no unproven forms: {d:?}");
        assert!(
            !d.warned,
            "the session-creation WARN fired on an all-proven claim set — it cannot be shown NOT \
             to fire, so it is not a detector: {d:?}"
        );
    }

    /// Both polarities in one claim set: the WARN names the unproven form and not the proven one.
    #[test]
    fn a_mixed_claim_set_warns_only_about_the_unproven_form() {
        let _g = test_lock();
        counters::reset();
        let proven = a_proven_key();
        let unproven =
            ProofKey::parse("test.planted::NeverProven/1+/f16>f16/no_such_kernel/static/mixed");
        let d = disclose_claimed_forms(&[
            ClaimedForm {
                op_type: "ai.onnx::Proven".to_string(),
                key: Some(proven),
                nodes: 1,
            },
            ClaimedForm {
                op_type: "test.planted::NeverProven".to_string(),
                key: Some(unproven),
                nodes: 1,
            },
        ]);
        assert_eq!(d.proven, 1, "{d:?}");
        assert_eq!(d.unmeasured, 1, "{d:?}");
        assert!(d.warned, "{d:?}");
    }

    /// A claim granted with no proof key is unmeasured, not proven by omission.
    #[test]
    fn a_claim_without_a_proof_key_is_unmeasured() {
        let _g = test_lock();
        counters::reset();
        let d = disclose_claimed_forms(&[ClaimedForm {
            op_type: "ai.onnx::If".to_string(),
            key: None,
            nodes: 1,
        }]);
        assert_eq!(d.unmeasured, 1, "{d:?}");
        assert!(d.warned, "{d:?}");
    }

    /// `UNMEASURED` and `DIVERGENT` must not read alike.
    #[test]
    fn divergent_and_unmeasured_are_distinct_tokens() {
        assert_ne!(
            FormEvidence::Unmeasured.token(),
            FormEvidence::Divergent("DIVERGENT".to_string()).token()
        );
        assert_eq!(FormEvidence::Unmeasured.token(), "UNMEASURED");
        assert_eq!(
            FormEvidence::Divergent("DIVERGENT".into()).token(),
            "DIVERGENT"
        );
        assert_eq!(FormEvidence::LedgerFaulted.token(), "LEDGER-FAULTED");
        assert!(FormEvidence::LedgerFaulted.warrants_warning());
    }
}
