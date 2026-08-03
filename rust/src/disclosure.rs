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
    /// The ledger holds a `MATCH` for this key, with provenance, **obtained on this device**.
    Proven(&'static LedgerEntry),
    /// The ledger holds a sound `MATCH` for this key whose `device` **cannot be compared** to this
    /// run's hardware — a selector ordinal, or a run that has not opened a device. Claimed, and
    /// disclosed by name with what the entry says and what the run is on.
    ///
    /// It is its own state rather than a flavour of `Proven` for the same reason `Unmeasured` and
    /// `Divergent` are two: the reader's action differs. A divergence seen on new hardware
    /// suspects this list first, and today this list is *every* form.
    DeviceUnattributed(&'static LedgerEntry, &'static str),
    /// The ledger holds a sound `MATCH` obtained on a **named other device**. Not claimed.
    ///
    /// This is the slot §10.0.1 R12's `PROVEN-ELSEWHERE` would occupy. It declines while the
    /// instrument that would promote a per-form key onto second hardware is disputed — see
    /// `docs/OP_COVERAGE.md` §7.20.
    ProvenElsewhere(&'static LedgerEntry),
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
            FormEvidence::DeviceUnattributed(..) => "DEVICE-UNATTRIBUTED",
            FormEvidence::ProvenElsewhere(_) => "PROVEN-ELSEWHERE",
            FormEvidence::Unmeasured => "UNMEASURED",
            FormEvidence::Divergent(_) => "DIVERGENT",
            FormEvidence::LedgerFaulted => "LEDGER-FAULTED",
        }
    }

    /// Whether this evidence obliges the session-creation WARN.
    ///
    /// **`DeviceUnattributed` does not.** Every one of the 97 baked entries is in that state, so a
    /// WARN there fires on every form of every run and a WARN that always fires is one a reader
    /// learns to filter — which would cost the `UNMEASURED` WARN its audience. It is disclosed at
    /// INFO, by name, with the label the entry carries and the device the run opened.
    /// `ProvenElsewhere` **does** warn: it is a decline, and a form dropping to the CPU EP
    /// is a thing the operator must be able to see without reading a counters file.
    pub fn warrants_warning(&self) -> bool {
        !matches!(
            self,
            FormEvidence::Proven(_) | FormEvidence::DeviceUnattributed(..)
        )
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
        return match registry::device_state(&entry.device) {
            registry::ProofState::Proven => FormEvidence::Proven(entry),
            registry::ProofState::DeviceUnattributed { reason, .. } => {
                FormEvidence::DeviceUnattributed(entry, reason)
            }
            _ => FormEvidence::ProvenElsewhere(entry),
        };
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
    /// Distinct claimed forms whose evidence is a ledger `MATCH` obtained on this device.
    pub proven: usize,
    /// Distinct claimed forms admitted on a `MATCH` obtained on **another device** (R12).
    pub device_unattributed: usize,
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

    /// Forms admitted on ledger evidence, whichever device frame that evidence carries.
    ///
    /// The WARN's negative arm asserts non-vacuity on **this**, not on `proven`: every one of the
    /// 97 baked entries records a selector ordinal, so `proven` is 0 for every run today and an
    /// arm gated on it would pass because nothing was claimed rather than because nothing warned.
    pub fn proof_backed(&self) -> usize {
        self.proven + self.device_unattributed
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
    let mut unattributed_lines: Vec<String> = Vec::new();
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
            FormEvidence::DeviceUnattributed(entry, reason) => {
                d.device_unattributed += 1;
                unattributed_lines.push(format!(
                    "{} x{} [{}] DEVICE-UNATTRIBUTED: proven by {} against {}, entry-device={}, \
                     running-device={} — {}",
                    form.op_type,
                    form.nodes,
                    key.0,
                    entry.artifact,
                    entry.ort_build,
                    if entry.device.is_empty() {
                        "<absent>"
                    } else {
                        &entry.device
                    },
                    registry::running_device_names().join("; "),
                    reason,
                ));
            }
            FormEvidence::ProvenElsewhere(entry) => {
                // A decline, so it belongs in the WARN list beside UNMEASURED and DIVERGENT: the
                // operator's action is the same shape — a form they expected on the GPU is not.
                d.unmeasured += 1;
                unproven_lines.push(format!(
                    "{} x{} [{}] PROVEN-ELSEWHERE: proven by {} on {}, and this run is on \
                     {}. A proof is a property of a form on a device, so this form runs on the \
                     CPU EP. Prove it here with gen_proof_ledger.py --append",
                    form.op_type,
                    form.nodes,
                    key.0,
                    entry.artifact,
                    entry.device,
                    registry::running_device_names().join("; "),
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

    // Both INFO branches feed the same pair of fields, and the second one has to be *joined* to
    // the first rather than dropped. Before this merge each branch was written by a different
    // author: the proven-forms INFO set `informed`, the unattributed-forms INFO discarded its
    // return value. On this box every claimed form is DEVICE-UNATTRIBUTED, so `proven_lines` is
    // empty, and the artifact read `session_disclosure_info_channel: UNOBSERVABLE` on a run that
    // had in fact emitted an INFO. A channel counter that reports no traffic while traffic is
    // moving is worse than no counter, because it is cited as evidence.
    //
    // `info_reached_ort_sink` is ANDed, not ORed: it is a claim about the disclosure's INFO half
    // as a whole, and one record the threshold refused makes that half incomplete.
    let mut note_info = |reached: bool| {
        d.info_reached_ort_sink = if d.informed {
            d.info_reached_ort_sink && reached
        } else {
            reached
        };
        d.informed = true;
    };

    if !proven_lines.is_empty() {
        note_info(logging::info_through_ort_sink(
            TARGET,
            &format!(
                "session claims {} proven form(s) [§8.9.7]: {}",
                proven_lines.len(),
                proven_lines.join("; ")
            ),
        ));
    }

    if !unattributed_lines.is_empty() {
        // INFO, not WARN — see `FormEvidence::warrants_warning`. It is its own message rather
        // than a suffix on the proven one because the two make different statements and a reader
        // grepping for what this build vouched for *here* must not have to parse a joined list.
        note_info(logging::info_through_ort_sink(
            TARGET,
            &format!(
                "session claims {} form(s) whose proof frame is UNATTRIBUTED [§10.0.1 R12]: {}. \
                 The proofs are sound; what cannot be checked is whether they were obtained on \
                 the hardware this run opened, because the entry records a selector ordinal and \
                 a selector is a request rather than an identity. Re-prove with \
                 gen_proof_ledger.py --append on this device to replace the ordinal with a \
                 device name.",
                unattributed_lines.len(),
                unattributed_lines.join("; ")
            ),
        ));
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
        device_unattributed: d.device_unattributed,
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

/// **§8.9.18 obligation 1 — print the demotions, on every run.**
///
/// Per-entry demotion is only a fix if the demotion is *visible*. A ledger that silently gets
/// smaller is worse than one that faults loudly: the faulted one stops claiming, the shrinking one
/// keeps claiming and nobody is told which proofs left. So "96 of 97 proofs are live" has to be a
/// sentence the reader is handed, not a state they infer from a count they did not think to
/// compare.
///
/// Called once per session-creation disclosure, **before** the claim set is known and regardless
/// of whether this session claims anything, because the demotions are a property of the artifact
/// and not of what this model happens to touch. Returns `(live, demoted)` so a test can assert on
/// it without reading a log.
pub fn disclose_ledger_demotions() -> (usize, usize) {
    disclose_demotions_of(registry::ledger())
}

/// The body of [`disclose_ledger_demotions`], against any ledger.
///
/// Separated so both polarities are reachable: the baked ledger has no demoted entry, so a
/// function that could only be called on it would have no observable firing state — which is the
/// second obligation §8.9.18 attaches, and the same rule Niobe is held to on the amplification
/// probe.
pub fn disclose_demotions_of(ledger: &registry::Ledger) -> (usize, usize) {
    let live = ledger.len();
    let demoted = ledger.entry_faults.len();
    if demoted == 0 {
        // The negative polarity still speaks. A run that prints nothing here is
        // indistinguishable from a run whose disclosure did not happen, which is the ambiguity
        // §8.9.7's WARN pair exists to avoid.
        logging::info_through_ort_sink(
            TARGET,
            &format!("[§8.9.18] proof ledger: {live}/{live} entries live, 0 demoted"),
        );
        return (live, demoted);
    }
    // R13: quote the condition, never only the count. A reader told "1 entry demoted" cannot act;
    // a reader told which entry and why can re-prove it.
    logging::warn_through_ort_sink(
        TARGET,
        &format!(
            "[§8.9.18] proof ledger: {live}/{} entries live, {demoted} demoted. Each demoted \
             entry proves nothing and its form will decline unless something else proves it: {}. \
             Re-prove them with rust/tools/gen_proof_ledger.py --reprove",
            live + demoted,
            ledger.entry_faults.join("; ")
        ),
    );
    (live, demoted)
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
            d.proof_backed() >= 1,
            "ERROR(instrument): the 'proven' arm claimed no proof-backed form, so its silence is \
             vacuous: {d:?}"
        );
        assert_eq!(d.unproven(), 0, "expected no unproven forms: {d:?}");
        assert!(
            !d.warned,
            "the session-creation WARN fired on an all-proven claim set — it cannot be shown NOT \
             to fire, so it is not a detector: {d:?}"
        );
    }

    /// **Every INFO branch of §8.9.7 must reach the INFO counter, not just the first one.**
    ///
    /// `disclose_claimed_forms` has two INFO branches: proven-here forms and DEVICE-UNATTRIBUTED
    /// ones. They were written by different authors and only the first set `informed`; the second
    /// discarded the return of its own `info_through_ort_sink` call. That is not a corner case on
    /// this hardware — **every baked ledger entry is DEVICE-UNATTRIBUTED** (see
    /// `FormEvidence::warrants_warning`), so the unjoined branch is the *only* INFO a real session
    /// emits, and `session_disclosure_info_channel` read `UNOBSERVABLE` on runs that had just
    /// emitted one. A channel counter that reports no traffic while traffic moves is worse than
    /// no counter, because it is cited as evidence.
    ///
    /// This asserts the property (a proof-backed disclosure informs) rather than the branch, so
    /// it stays honest on hardware where the entry does attribute and the *other* branch runs.
    #[test]
    fn a_proof_backed_disclosure_informs_whichever_branch_carried_it() {
        let _g = test_lock();
        counters::reset();
        let d = disclose_claimed_forms(&[ClaimedForm {
            op_type: "ai.onnx::Proven".to_string(),
            key: Some(a_proven_key()),
            nodes: 7,
        }]);
        assert!(
            d.proof_backed() >= 1,
            "ERROR(instrument): nothing was proof-backed, so this arm cannot see an INFO: {d:?}"
        );
        assert!(
            d.informed,
            "a disclosure that emitted a proof-backed INFO reported no INFO at all. The INFO \
             half is counted per disclosure, not per branch: {d:?}"
        );
        let doc = counters::snapshot().to_json();
        let expected = if d.info_reached_ort_sink {
            "\"session_disclosure_info_channel\": \"OFFERED_TO_ORT\""
        } else {
            "\"session_disclosure_info_channel\": \"BELOW_ORT_THRESHOLD\""
        };
        assert!(
            doc.contains(expected),
            "the counters artifact and the disclosure disagree about the INFO half; expected \
             {expected} for {d:?}. Got:\n{doc}"
        );
    }

    /// Both polarities in one claim set: the WARN names the unproven form and not the proven one.
    #[test]
    fn a_mixed_claim_set_warns_only_about_the_unproven_form() {        let _g = test_lock();
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
        assert_eq!(d.proof_backed(), 1, "{d:?}");
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

    /// **§8.9.18 obligation — the demotion disclosure, seen in its firing state.**
    ///
    /// The baked ledger has no demoted entry, so a test that only called this on `registry::
    /// ledger()` would assert `0 demoted` forever and never show the path working. Both ledgers
    /// here are planted, they differ only in one entry's `shader_digest`, and the two readings
    /// must differ — otherwise the count is zero-by-construction, which is precisely what the
    /// ruling forbids.
    #[test]
    fn the_demotion_count_is_printed_and_is_not_zero_by_construction() {
        let _g = test_lock();
        const A: &str = "ai.onnx::Add/7+/f32,f32>f32/ew_binary_add_f32/static/n2";
        const B: &str = "ai.onnx::Mul/7+/f32,f32>f32/ew_binary_mul_f32/static/n2";
        let line = |key: &str, stem: &str, digest: &str| {
            format!(
                "{{\"key\":\"{key}\",\"verdict\":\"MATCH\",\"device\":\"d\",\"ort_build\":\"1\",\
                 \"tolerance\":\"t\",\"artifact\":\"a\",\"generated_at\":\"now\",\
                 \"shaders\":[\"{stem}\"],\"shader_digest\":\"{digest}\",\
                 \"claimed_nodes\":1,\"dispatches_executed\":1}}"
            )
        };
        let build = |a_digest: &str| {
            let b_digest = registry::shader_digest_for(&["ew_binary_mul_f32"])
                .expect("a stem list this build can digest");
            let body = format!(
                "{}\n{}\n",
                line(A, "ew_binary_add_f32", a_digest),
                line(B, "ew_binary_mul_f32", &b_digest)
            );
            let d = format!("{:016x}", registry::fnv1a64(body.as_bytes()));
            registry::parse_ledger(&format!(
                "{{\"__ledger__\":1,\"content_fnv1a64\":\"{d}\",\"entry_count\":2,\
                 \"generator\":\"test\"}}\n{body}"
            ))
        };

        let sound_digest = registry::shader_digest_for(&["ew_binary_add_f32"])
            .expect("a stem list this build can digest");
        let clean = build(&sound_digest);
        assert_eq!(
            disclose_demotions_of(&clean),
            (2, 0),
            "ERROR(instrument): the control ledger is already demoting something, so the arm \
             below cannot show a demotion being detected: {:?}",
            clean.entry_faults
        );

        let drifted = build("0000000000000000");
        assert_eq!(
            disclose_demotions_of(&drifted),
            (1, 1),
            "one entry's shader digest drifted and the disclosure did not say so: faults={:?} \
             entry_faults={:?}",
            drifted.faults,
            drifted.entry_faults
        );
        // The blast radius, stated where the disclosure is asserted: the sound entry survives.
        assert!(
            drifted.faults.is_empty(),
            "a located defect faulted the artifact: {:?}",
            drifted.faults
        );
        assert_ne!(
            disclose_demotions_of(&clean),
            disclose_demotions_of(&drifted),
            "both arms reported the same thing; the demotion count is not being read"
        );
    }

    /// §8.9.18's second group: damage you **cannot locate** still faults the whole artifact.
    ///
    /// Without this, the per-entry correction reads as "nothing faults the ledger any more",
    /// which is the weakening the ruling warns the correction can become.
    #[test]
    fn unlocatable_damage_still_faults_the_whole_artifact() {
        let key = "ai.onnx::Add/7+/f32,f32>f32/ew_binary_add_f32/static/n2";
        let digest = registry::shader_digest_for(&["ew_binary_add_f32"]).expect("a stem list");
        let entry = format!(
            "{{\"key\":\"{key}\",\"verdict\":\"MATCH\",\"device\":\"d\",\"ort_build\":\"1\",\
             \"tolerance\":\"t\",\"artifact\":\"a\",\"generated_at\":\"now\",\
             \"shaders\":[\"ew_binary_add_f32\"],\"shader_digest\":\"{digest}\",\
             \"claimed_nodes\":1,\"dispatches_executed\":1}}"
        );
        let good = format!("{entry}\n");
        let good_digest = format!("{:016x}", registry::fnv1a64(good.as_bytes()));

        // Control: the same body under a correct header is sound in both lists.
        let sound = registry::parse_ledger(&format!(
            "{{\"__ledger__\":1,\"content_fnv1a64\":\"{good_digest}\",\"entry_count\":1,\
             \"generator\":\"test\"}}\n{good}"
        ));
        assert!(
            sound.faults.is_empty() && sound.entry_faults.is_empty(),
            "ERROR(instrument): the control is not sound, so the arms below prove nothing: {:?} \
             {:?}",
            sound.faults,
            sound.entry_faults
        );

        // A hand-edited body: any line may be affected, so nothing is locatable.
        let hand_edited = registry::parse_ledger(&format!(
            "{{\"__ledger__\":1,\"content_fnv1a64\":\"dead0000dead0000\",\"entry_count\":1,\
             \"generator\":\"test\"}}\n{good}"
        ));
        assert!(
            !hand_edited.faults.is_empty(),
            "a header digest mismatch did not fault the artifact"
        );

        // A line that does not parse: you cannot tell what it was going to say.
        let unparseable = registry::parse_ledger(
            "{\"__ledger__\":1,\"entry_count\":1,\"generator\":\"test\"}\n{ not json at all\n",
        );
        assert!(
            !unparseable.faults.is_empty(),
            "an unparseable line was demoted as if it were one located entry, but nothing about \
             it is located: entry_faults={:?}",
            unparseable.entry_faults
        );
    }
}
