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
    /// The ledger holds a sound `MATCH` about **this exact subject**, obtained in a frame that
    /// differs in exactly the components named. **Claimed, and disclosed with its δ set.**
    ///
    /// §8.9.19 part 1. It stopped being a decline because declining it meant a Linux run declined
    /// every form and produced no op-correctness number at all — and because claiming *while
    /// naming which frame components moved* is strictly more informative than refusing.
    ProvenElsewhere(&'static LedgerEntry, Vec<registry::FrameDelta>, String),
    /// The ledger holds a sound `MATCH` whose SPIR-V is byte-identical to this build's and whose
    /// **source closure** differs. Claimed, and **named** — §8.9.19's fourth row.
    SourceCosmetic(&'static LedgerEntry, String, String),
    /// An entry exists and its **subject moved**: the kernel it proves has been replaced, or its
    /// SPIR-V differs with no `source_digest` to say which. Not claimed.
    ///
    /// Before §8.9.19 this population could not be disclosed at all, because the entry was
    /// deleted at parse time and the form read as `UNMEASURED`.
    SubjectChanged(&'static LedgerEntry, String, String, bool),
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
            FormEvidence::ProvenElsewhere(..) => "PROVEN-ELSEWHERE",
            FormEvidence::SourceCosmetic(..) => "SOURCE-COSMETIC",
            FormEvidence::SubjectChanged(..) => "SUBJECT-CHANGED",
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
    ///
    /// **`ProvenElsewhere` does not either, and that changed with §8.9.19.** It used to be a
    /// decline, so it warned per form. It is now a *claim*, and on a Linux run **every** form is
    /// toolchain-elsewhere — a per-form WARN there is the always-fires shape again. The claim is
    /// not silent: it is disclosed per form at INFO with its δ set, counted in
    /// `proven_elsewhere_claims`, and summarised in **one** WARN naming the δ set and the number
    /// of forms, which is the §8.9.7 disclosure §8.9.19 part 3 item 3 requires and is what keeps
    /// this from being the silent extrapolation §8.9.17 refused.
    ///
    /// **`SubjectChanged` does warn**: it is a decline, and a form dropping to the CPU EP because
    /// its kernel was replaced is a thing the operator must see without reading a counters file.
    pub fn warrants_warning(&self) -> bool {
        !matches!(
            self,
            FormEvidence::Proven(_)
                | FormEvidence::DeviceUnattributed(..)
                | FormEvidence::ProvenElsewhere(..)
                | FormEvidence::SourceCosmetic(..)
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
        return match registry::entry_state(entry) {
            registry::ProofState::Proven => FormEvidence::Proven(entry),
            registry::ProofState::DeviceUnattributed { reason, .. } => {
                FormEvidence::DeviceUnattributed(entry, reason)
            }
            registry::ProofState::ProvenElsewhere { deltas, detail } => {
                FormEvidence::ProvenElsewhere(entry, deltas, detail)
            }
            registry::ProofState::SourceCosmetic { recorded, current } => {
                FormEvidence::SourceCosmetic(entry, recorded, current)
            }
            registry::ProofState::SubjectChanged {
                recorded,
                current,
                source_comparable,
            } => FormEvidence::SubjectChanged(entry, recorded, current, source_comparable),
            registry::ProofState::Unproven => FormEvidence::Unmeasured,
        };
    }
    if !ledger.faults.is_empty() {
        return FormEvidence::LedgerFaulted;
    }
    FormEvidence::Unmeasured
}

/// The blind-axes caveat for one op, or the empty string when the row declares none.
///
/// §8.9.23. Two clauses, and the second is the one Rai's 🟡 was about:
///
/// 1. **what the key does not distinguish** — these axes are push constants in one uniform code
///    path, so one proof covers every value of them *by construction*, not by luck;
/// 2. **who speaks for them** — a CI-time suite, and **nothing in the reader's own session**.
///
/// Without (2) the line reads as though the session had witnessed the breadth. It had not, and it
/// cannot: the session ran the values its graph contained.
fn blind_axes_clause(op_type: &str) -> String {
    let axes = registry::blind_axes_for(op_type);
    if axes.is_empty() {
        return String::new();
    }
    // No terminal period: these clauses are spliced into a joined list that its group message
    // then continues with a sentence of its own. A period here renders as `does.. The proofs`.
    format!(
        " BLIND{{{}}}: the proof key does not distinguish these attributes — they are push \
         constants read by one uniform code path, so one proof covers every value of them by \
         construction. A CI-time suite varies them; nothing in this session does",
        axes.join(",")
    )
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
    /// Distinct claimed forms admitted on a `MATCH` obtained **out of frame** (§8.9.19).
    pub proven_elsewhere: usize,
    /// Distinct claimed forms whose source closure moved while their SPIR-V did not.
    ///
    /// This counts the **subject** axis, not a state population: a form can be
    /// `DEVICE-UNATTRIBUTED` (frame) and source-cosmetic (subject) at once, and on today's
    /// ledger every entry is device-unattributed, so a count taken off the state alone could
    /// only ever read zero.
    pub source_cosmetic: usize,
    /// Distinct claimed forms whose *only* finding is a cosmetic source move — the frame is
    /// otherwise clean. Subset of [`Self::source_cosmetic`]; this is the one that is a claim
    /// population, so it is the one `proof_backed` may add.
    pub source_cosmetic_only: usize,
    /// Distinct claimed forms declined because their subject moved (§8.9.19).
    pub subject_changed: usize,
    /// The union of every δ component seen across this session's out-of-frame claims.
    pub frame_deltas: Vec<String>,
    /// Distinct claimed forms with no evidence at all.
    pub unmeasured: usize,
    /// Distinct claimed forms whose recorded verdict is not `MATCH`.
    pub divergent: usize,
    /// Distinct claimed forms whose evidence could not be read because the ledger is faulted.
    pub ledger_faulted: usize,
    /// Distinct claimed forms whose row declares [`registry::OpSpec::blind_axes`], and whose
    /// disclosure line therefore carries the §8.9.23 breadth caveat.
    ///
    /// Counted so a test can assert the caveat was rendered without scraping a log line — the
    /// same reason every other field here exists. A caveat that is only observable by reading
    /// text is a caveat whose disappearance nothing detects.
    pub blind_axes_disclosed: usize,
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
    /// Whether that INFO was accepted by this process's stderr — the channel a console user
    /// actually reads, and on a default host (ORT threshold WARNING) the only one that carries an
    /// INFO at all.
    pub info_reached_stderr: bool,
    /// Whether the INFO was re-emitted at WARNING because no quiet channel would carry it, and
    /// that re-emission was delivered.
    pub info_escalated: bool,
}

impl Disclosure {
    /// Forms that obliged the WARN.
    pub fn unproven(&self) -> usize {
        self.unmeasured + self.divergent + self.ledger_faulted + self.subject_changed
    }

    /// Forms admitted on ledger evidence, whichever frame that evidence carries.
    ///
    /// The WARN's negative arm asserts non-vacuity on **this**, not on `proven`: every one of the
    /// 97 baked entries records a selector ordinal, so `proven` is 0 for every run today and an
    /// arm gated on it would pass because nothing was claimed rather than because nothing warned.
    pub fn proof_backed(&self) -> usize {
        self.proven + self.device_unattributed + self.proven_elsewhere + self.source_cosmetic_only
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
        // The blind-axes clause is appended to whichever line this form produces, whatever its
        // evidence state. Attaching it after the match rather than inside eight `format!`s is not
        // only shorter: a caveat that has to be remembered at eight call sites is a caveat that
        // will be missing from one of them, and the one it goes missing from will be the branch
        // nobody reads until it matters.
        let before = (
            proven_lines.len(),
            unattributed_lines.len(),
            unproven_lines.len(),
        );
        let blind = blind_axes_clause(&form.op_type);
        if !blind.is_empty() {
            d.blind_axes_disclosed += 1;
        }
        let Some(key) = &form.key else {
            // No proof key was produced for this claim. That is not evidence of correctness and
            // must not be counted as proven; it is unmeasured by this instrument's own admission.
            d.unmeasured += 1;
            unproven_lines.push(format!(
                "{} x{} [UNMEASURED: the claim gate produced no proof key for this form]{blind}",
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
                // SUBJECT AND FRAME ARE DIFFERENT AXES, AND A SINGLE TOKEN CAN ONLY CARRY ONE.
                //
                // Found by running the row-4 acceptance rather than reasoning about it: a
                // comment-only edit to `ew_binary.comp` moved every affected entry's subject to
                // SOURCE-COSMETIC, and the disclosure said `DEVICE-UNATTRIBUTED` and nothing
                // else — because the state lattice has to pick one token and the device fact
                // dominates. Every entry in today's baked ledger is device-unattributed, so the
                // named row §8.9.19 calls "the row that proves the pair does work" would have
                // been unobservable in the only ledger that ships, and `source_cosmetic` would
                // have been a counter whose only possible value is zero.
                //
                // The state stays single-valued; the subject verdict is read off the entry and
                // printed beside it, so neither axis can hide the other.
                let subject_note = match &entry.subject {
                    registry::SubjectVerdict::SourceCosmetic { recorded, current } => {
                        d.source_cosmetic += 1;
                        format!(
                            " SUBJECT={}: source closure {recorded} -> {current}, SPIR-V \
                             byte-identical.",
                            entry.subject.token()
                        )
                    }
                    registry::SubjectVerdict::Identical => String::new(),
                    other => format!(" SUBJECT={}.", other.token()),
                };
                unattributed_lines.push(format!(
                    "{} x{} [{}] DEVICE-UNATTRIBUTED: proven by {} against {}, entry-device={}, \
                     running-device={} — {}{}",
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
                    subject_note,
                ));
            }
            FormEvidence::ProvenElsewhere(entry, deltas, detail) => {
                // §8.9.19 part 3 item 3: the claim is granted, and the δ set is printed. A claim
                // granted before its disclosure exists is exactly the trade §8.9.17 refused.
                d.proven_elsewhere += 1;
                let tokens: Vec<&str> = deltas.iter().map(registry::FrameDelta::token).collect();
                for t in &tokens {
                    if !d.frame_deltas.iter().any(|s| s == t) {
                        d.frame_deltas.push((*t).to_string());
                    }
                }
                unattributed_lines.push(format!(
                    "{} x{} [{}] PROVEN-ELSEWHERE{{{}}}: proven by {} against {} — {}. The \
                     subject is this build's code; only the frame moved, so the claim stands and \
                     says so.",
                    form.op_type,
                    form.nodes,
                    key.0,
                    tokens.join(","),
                    entry.artifact,
                    entry.ort_build,
                    detail,
                ));
            }
            FormEvidence::SourceCosmetic(entry, recorded, current) => {
                d.source_cosmetic += 1;
                d.source_cosmetic_only += 1;
                unattributed_lines.push(format!(
                    "{} x{} [{}] SOURCE-COSMETIC: the source closure hashes to {current} where \
                     the entry recorded {recorded}, and the compiled SPIR-V is byte-identical. \
                     Proven by {} — claimed, and named because only the digest *pair* can tell \
                     this from a kernel change.",
                    form.op_type, form.nodes, key.0, entry.artifact,
                ));
            }
            FormEvidence::SubjectChanged(_entry, recorded, current, source_comparable) => {
                d.subject_changed += 1;
                let why = if source_comparable {
                    "both digests moved, so the kernel itself was replaced"
                } else {
                    "the entry records no source_digest, so `different compiler` and `different \
                     kernel` cannot be told apart and the fail-safe reading is the second; \
                     backfill it with gen_proof_ledger.py --backfill-frame"
                };
                unproven_lines.push(format!(
                    "{} x{} [{}] SUBJECT-CHANGED: proven against shader digest {recorded}, this \
                     build's modules hash to {current} — {why}. This form runs on the CPU EP; \
                     re-prove it with gen_proof_ledger.py --reprove",
                    form.op_type, form.nodes, key.0,
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
        if !blind.is_empty() {
            for (vec, len) in [
                (&mut proven_lines, before.0),
                (&mut unattributed_lines, before.1),
                (&mut unproven_lines, before.2),
            ] {
                if vec.len() > len {
                    let last = vec.last_mut().expect("just pushed");
                    last.push_str(&blind);
                }
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
    // as a whole, and one record the threshold refused makes that half incomplete. The stderr arm
    // is ANDed for the same reason, and it is tracked *separately* rather than folded into one
    // "delivered" bit, because RAI-013's question is which channel the user was on: on a default
    // host ORT's threshold refuses every INFO and the console carries all of them.
    let mut note_info = |reached: logging::Delivery| {
        if d.informed {
            d.info_reached_ort_sink &= reached.ort_sink;
            d.info_reached_stderr &= reached.stderr;
        } else {
            d.info_reached_ort_sink = reached.ort_sink;
            d.info_reached_stderr = reached.stderr;
        }
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
        d.warn_reached_ort_sink = logging::warn_through_ort_sink(TARGET, &msg).ort_sink;
    }

    d.info_escalated = escalate_if_unreachable(&d);

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
        info_reached_stderr: d.info_reached_stderr,
        info_escalated: d.info_escalated,
    });
    d
}

/// Re-emit the INFO half at WARNING when **no channel a default user sees** would carry it.
///
/// §8.9.7 obliges a disclosure, and a disclosure is a statement about a channel, not about an
/// emission. Three candidate repairs were weighed against Rai's standard — what the user is
/// entitled to be told — rather than against convenience:
///
/// * **Raise our severity unconditionally.** Rejected. A routine, fully-proof-backed claim set is
///   not a warning, and a WARNING that fires on every healthy session is how the WARNING that
///   matters — the `UNMEASURED` one twenty lines up — stops being read. The user is entitled to be
///   told *accurately*, and severity is part of the message.
/// * **A session-completion summary.** Rejected. The moment the user is entitled to the
///   disclosure is before the session runs anything, and a session that claims a form it cannot
///   back is by construction one that may not reach completion — the same R12 argument that makes
///   `record_session_disclosure` write the artifact at the event instead of at shutdown.
/// * **A returnable field on the EP.** Kept as a *supplement*, not as this repair: it discharges
///   nothing for a user who does not know to ask, and "we would have told you if you had called
///   the getter" is not a disclosure.
///
/// So the disclosure travels quietly on the two channels a default user has (ORT's sink when its
/// threshold admits INFO; this process's stderr, which is what a console renders), and escalates
/// to WARNING **only when both have been measured to refuse it** — loud exactly once the quiet
/// path is known to have failed, and never on the strength of a severity constant.
///
/// Returns whether the escalation was needed *and* delivered. `false` covers both "not needed"
/// and "needed and also lost"; those are separated by `session_disclosure_info_reach`, which is
/// the observable that has to carry the distinction.
fn escalate_if_unreachable(d: &Disclosure) -> bool {
    if !d.informed || d.info_reached_ort_sink || d.info_reached_stderr {
        return false;
    }
    // The escalation can fail too, and it is counted when it does — the same generalisation the
    // INFO's own channel counter rests on: this stays a statement about the channel rather than
    // about which branch happened to run.
    logging::warn_through_ort_sink(
        TARGET,
        &format!(
            "[§8.9.7] the session-creation disclosure could not be delivered on any channel a \
             user sees by default: ORT's own severity threshold excludes INFO and this process's \
             stderr refused the write. It is repeated here at WARNING because that is the one \
             severity ORT's default threshold admits — this is a delivery escalation, not a fault \
             report. This session claims {} proof-backed form(s) ({} proven here, {} \
             device-unattributed, {} proven elsewhere) and {} form(s) whose correctness is not \
             established. Set the host logger to INFO, or read \
             ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE, for the per-form detail.",
            d.proof_backed(),
            d.proven,
            d.device_unattributed,
            d.proven_elsewhere,
            d.unproven()
        ),
    )
    .reached_user()
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
    let d = Disclosure {
        informed: true,
        info_reached_ort_sink: reached.ort_sink,
        info_reached_stderr: reached.stderr,
        ..Default::default()
    };
    // The zero-claims branch is the one a user is *most* likely to be looking for an explanation
    // from — they asked for a Vulkan EP and got none of it — so it escalates on the same terms as
    // the claiming branch. Escalating one branch and not the other would make the reach token a
    // statement about which branch ran.
    let escalated = escalate_if_unreachable(&d);
    counters::record_session_disclosure(counters::SessionDisclosure {
        informed: true,
        info_reached_ort_sink: reached.ort_sink,
        info_reached_stderr: reached.stderr,
        info_escalated: escalated,
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
    let l = registry::ledger();
    disclose_ledger_faults_of(l);
    let counts = disclose_demotions_of(l);
    disclose_specialisation_frame_of(l);
    counts
}

/// **The whole-file faults, delivered to somebody who can act on them.**
///
/// Link, 2026-08-03: the counters artifact reported a faulted ledger and the log contained
/// **zero** occurrences of `proof ledger fault`, so the artifact and the log disagreed about
/// whether anything had gone wrong. The cause is a mechanism, not a wording. `registry::ledger()`
/// is a `OnceLock` initialised by the first lookup, and that lookup happens while ORT is still
/// building the EP — *before* the ORT logger is attached to our `log` facade. The `log::warn!` in
/// the initialiser is therefore emitted into a sink that does not exist yet, exactly once, and
/// nothing ever repeats it.
///
/// Per-entry demotions already avoided this by being disclosed here rather than at init
/// (§8.9.18); the whole-file faults were left behind, which is the more serious half — a faulted
/// ledger declines **every** form. Re-emitted here, on ORT's own channel, on every session.
pub fn disclose_ledger_faults_of(ledger: &registry::Ledger) -> usize {
    if ledger.faults.is_empty() {
        return 0;
    }
    logging::warn_through_ort_sink(
        TARGET,
        &format!(
            "[§8.9.18] proof ledger UNREADABLE ({} fault(s)); every form declines and the \
             decline is an instrument failure, not a finding about any form: {}. Regenerate it \
             with rust/tools/gen_proof_ledger.py",
            ledger.faults.len(),
            ledger.faults.join("; ")
        ),
    );
    ledger.faults.len()
}

/// **§8.9.20 — what the entries do and do not say about the pipeline they were proven on.**
///
/// Disclosed on every run because it is a *narrowing of what an entry means*, and a narrowing
/// nobody is told about is a quiet demotion of the proofs it applies to. An entry with no
/// `spec_digest` proves its form under a specialisation nobody recorded: the SPIR-V and the
/// source closure it names are exact, and the pipeline built from them is not pinned down. Unlike
/// a missing `source_digest` there is no repair from the tree — the value belonged to a run that
/// has ended — so these claim, and this is where they say so.
///
/// Returns `(recorded, unrecorded)`.
pub fn disclose_specialisation_frame_of(ledger: &registry::Ledger) -> (usize, usize) {
    let unrecorded = ledger.specialisation_unrecorded_entries().count();
    let recorded = ledger.len() - unrecorded;
    if unrecorded == 0 {
        logging::info_through_ort_sink(
            TARGET,
            &format!(
                "[§8.9.20] proof ledger: all {recorded} entr(ies) name the runtime specialisation \
                 they were proven under"
            ),
        );
    } else {
        logging::info_through_ort_sink(
            TARGET,
            &format!(
                "[§8.9.20] proof ledger: {unrecorded} of {} entr(ies) record NO runtime \
                 specialisation (SPEC-UNRECORDED). They prove their form's kernel exactly and say \
                 nothing about which pipeline was built from it; a specialisation constant chosen \
                 at dispatch is outside both the SPIR-V and the source digest. They claim; the \
                 only repair is rust/tools/gen_proof_ledger.py --reprove.",
                ledger.len()
            ),
        );
    }
    (recorded, unrecorded)
}

/// The body of [`disclose_ledger_demotions`], against any ledger.
///
/// Separated so both polarities are reachable: the baked ledger has no demoted entry, so a
/// function that could only be called on it would have no observable firing state — which is the
/// second obligation §8.9.18 attaches, and the same rule Niobe is held to on the amplification
/// probe.
pub fn disclose_demotions_of(ledger: &registry::Ledger) -> (usize, usize) {
    // §8.9.19 part 1 made a subject mismatch **survive** parsing, so `entries.len()` is no longer
    // the live count and `entry_faults.len()` is no longer the demoted count. Reading either
    // directly here would have silently reported every drifted entry as live from the day entry
    // survival landed — the obligation satisfied on paper while the number stopped meaning it.
    let subject_changed: Vec<&registry::LedgerEntry> = ledger.subject_changed_entries().collect();
    let live = ledger.len() - subject_changed.len();
    let demoted = ledger.demotion_count();
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
    let mut reasons: Vec<String> = ledger.entry_faults.clone();
    for e in &subject_changed {
        reasons.push(format!(
            "ledger entry for {:?} is {} — it was proven against shader digest {} and this \
             build's modules hash to something else",
            e.key.0,
            e.subject.token(),
            e.shader_digest
        ));
    }
    logging::warn_through_ort_sink(
        TARGET,
        &format!(
            "[§8.9.18] proof ledger: {live}/{} entries live, {demoted} demoted. Each demoted \
             entry proves nothing and its form will decline unless something else proves it: {}. \
             Re-prove them with rust/tools/gen_proof_ledger.py --reprove",
            live + demoted,
            reasons.join("; ")
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

    // ── §8.9.23 blind-axes disclosure ────────────────────────────────────────────────────────

    /// A row that declares blind axes must say so on its claim line, **both clauses**.
    ///
    /// Rai's 🟡 was not that the caveat was wrong — it was written down plainly in `form.rs` and
    /// in `OP_COVERAGE.md` — but that *the session-time line a user reads did not carry it*. So
    /// the assertion is on the rendered line and on the counter, not on the registry field: a
    /// field that is read by nothing discloses nothing.
    #[test]
    fn a_row_with_blind_axes_renders_both_clauses() {
        let clause = blind_axes_clause("ai.onnx::Conv");
        assert!(!clause.is_empty(), "Conv declares blind axes; the clause must not be empty");
        assert_eq!(
            clause,
            blind_axes_clause("Conv"),
            "the default domain has two spellings and the disclosure reaches here with the bare \
             one; a caveat that depends on which is used is a caveat that goes missing in the \
             live path"
        );
        for axis in ["group", "strides", "dilations", "pads"] {
            assert!(
                clause.contains(axis),
                "§8.9.23 names `{axis}` as blind and the disclosure does not: {clause}"
            );
        }
        assert!(
            clause.contains("does not distinguish"),
            "clause 1 — what the key is silent about — is missing: {clause}"
        );
        assert!(
            clause.contains("nothing in this session does"),
            "clause 2 — that no session-time evidence speaks for these axes — is the half Rai's \
             finding was about, and it is missing: {clause}"
        );
    }

    /// Negative polarity: a row that declares no blind axes must not acquire a caveat.
    ///
    /// Without this arm the test above passes for a clause that is appended unconditionally, which
    /// would make the caveat noise rather than information.
    #[test]
    fn a_row_without_blind_axes_renders_nothing() {
        assert_eq!(blind_axes_clause("ai.onnx::Add"), "");
        assert_eq!(blind_axes_clause("Add"), "");
        assert_eq!(blind_axes_clause("no.such::Op"), "");
    }

    /// The caveat reaches the line, in every evidence state, and is counted.
    #[test]
    fn the_blind_axes_caveat_travels_with_the_claim_line() {
        let _g = test_lock();
        counters::reset();
        // An UNMEASURED `Conv` form: the evidence state that produces a WARN line rather than an
        // INFO one. The caveat has to be on both, and the branches are separate `format!`s, so a
        // caveat attached per-branch is one that goes missing from whichever branch was written
        // last.
        let key = ProofKey::parse(
            "ai.onnx::Conv/1+/f32,f32>f32/conv_f32#base/static/unmeasured-blind-control",
        );
        assert_eq!(
            evidence_for(&key),
            FormEvidence::Unmeasured,
            "ERROR(instrument): this key is proven, so the arm below tests the wrong branch"
        );
        let d = disclose_claimed_forms(&[ClaimedForm {
            op_type: "ai.onnx::Conv".to_string(),
            key: Some(key),
            nodes: 7,
        }]);
        assert_eq!(d.blind_axes_disclosed, 1, "the caveat was not counted: {d:?}");

        // And a form with no key at all — the `continue` branch, which skips the code that
        // appends the caveat to every other line and therefore has to carry it itself.
        let d2 = disclose_claimed_forms(&[ClaimedForm {
            op_type: "ai.onnx::Conv".to_string(),
            key: None,
            nodes: 1,
        }]);
        assert_eq!(
            d2.blind_axes_disclosed, 1,
            "a claim with no proof key still claims; its breadth caveat is not optional: {d2:?}"
        );
    }

    /// Positive polarity: a claimed form with no proof must warn.
    #[test]
    fn an_unmeasured_claimed_form_warns() {        let _g = test_lock();
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

    /// **The positive state, observed rather than inferred (RAI-013).**
    ///
    /// A disclosure that reaches the user must be observable *as having reached them*. Before
    /// this, the only delivery observable was ORT's threshold, which refuses every INFO on a
    /// default host — so `to_ort_sink: 0` was the whole account of a channel that had in fact
    /// printed the disclosure to the console the user was looking at. This arm asserts the state
    /// a real default run is in: no ORT sink attached (the test harness), stderr accepted it,
    /// and the artifact says `REACHED_USER`.
    #[test]
    fn a_proof_backed_disclosure_reaches_the_user_on_some_channel() {
        let _g = test_lock();
        counters::reset();
        let d = disclose_claimed_forms(&[ClaimedForm {
            op_type: "ai.onnx::Proven".to_string(),
            key: Some(a_proven_key()),
            nodes: 7,
        }]);
        assert!(
            d.informed && d.proof_backed() >= 1,
            "ERROR(instrument): no INFO was due, so reachability is unobservable here: {d:?}"
        );
        assert!(
            d.info_reached_stderr,
            "the §8.9.7 INFO was emitted and this process's stderr did not accept it, on a box \
             whose stderr works. That is the channel a console user reads: {d:?}"
        );
        assert!(
            !d.info_escalated,
            "the disclosure escalated to WARNING on a run where a quiet channel carried it. The \
             escalation must fire only after quiet delivery has been *measured* to fail, or every \
             healthy session emits a warning and the warning that matters stops being read: {d:?}"
        );
        let doc = counters::snapshot().to_json();
        assert!(
            doc.contains("\"session_disclosure_info_reach\": \"REACHED_USER\""),
            "the disclosure reached the user and the artifact does not say so:\n{doc}"
        );
        assert!(
            doc.contains("\"session_disclosure_infos_to_stderr\": 1,")
                && doc.contains("\"session_disclosure_stderr_failures\": 0,"),
            "the per-channel counts do not agree with the reach token:\n{doc}"
        );
    }

    /// The escalation's own two polarities, driven through the function that decides it.
    ///
    /// The firing state cannot be produced in-process — `stderr_fault_active` is a `OnceLock` over
    /// an environment variable, deliberately, because the state it simulates (a host with no
    /// usable handle 2) is a property of the process and not of a call. So the *decision* is
    /// tested here from constructed states, and the *delivery* is tested on the shipping binary by
    /// `rust/tools/probe_disclosure_reachability.py --arm escalation`, which arms the variable in
    /// a child. Neither half stands alone and both are named here so a reader can find the other.
    #[test]
    fn the_escalation_fires_only_when_no_quiet_channel_carried_the_info() {
        let _g = test_lock();
        counters::reset();

        // Not needed: stderr carried it. This is the state every default run is in.
        let carried = Disclosure {
            informed: true,
            info_reached_stderr: true,
            ..Default::default()
        };
        assert!(
            !escalate_if_unreachable(&carried),
            "escalated a disclosure the console had already carried"
        );

        // Not needed: ORT's own sink took it (a host that asked for INFO).
        let ort_took_it = Disclosure {
            informed: true,
            info_reached_ort_sink: true,
            ..Default::default()
        };
        assert!(!escalate_if_unreachable(&ort_took_it));

        // Nothing was due: silence about a channel nothing travelled on.
        let nothing_due = Disclosure::default();
        assert!(!escalate_if_unreachable(&nothing_due));

        // Needed. With no ORT logger attached the escalation can still only reach stderr, and on
        // this box it does — which is the return value being asserted. The arm that matters is
        // that it *fired*: `info_reached_ort_sink` and `info_reached_stderr` are both false.
        let lost = Disclosure {
            informed: true,
            proven: 1,
            ..Default::default()
        };
        assert!(
            escalate_if_unreachable(&lost),
            "a disclosure no quiet channel carried was not escalated, so a user on such a host is \
             told nothing at all"
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

    /// **The whole-file fault reaches a session, not just a `OnceLock` nobody was listening to.**
    ///
    /// Link measured the artifact saying the ledger was faulted while the log held zero
    /// occurrences of the fault text. The cause is that `registry::ledger()`'s initialiser runs
    /// before ORT attaches its logger, so the one `log::warn!` it emits goes to a sink that does
    /// not exist yet and is never repeated. Both polarities here: a sound ledger reports nothing,
    /// a faulted one reports its faults, and they must differ.
    #[test]
    fn a_faulted_ledger_is_reported_on_every_session_not_once_before_anyone_is_listening() {
        let _g = test_lock();
        let key = "ai.onnx::Add/7+/f32,f32>f32/ew_binary_add_f32/static/n2";
        let digest = registry::shader_digest_for(&["ew_binary_add_f32"]).expect("a stem list");
        let entry = format!(
            "{{\"key\":\"{key}\",\"verdict\":\"MATCH\",\"device\":\"d\",\"ort_build\":\"1\",\
             \"tolerance\":\"t\",\"artifact\":\"a\",\"generated_at\":\"now\",\
             \"shaders\":[\"ew_binary_add_f32\"],\"shader_digest\":\"{digest}\",\
             \"claimed_nodes\":1,\"dispatches_executed\":1}}"
        );
        let body = format!("{entry}\n");
        let good = format!("{:016x}", registry::fnv1a64(body.as_bytes()));
        let sound = registry::parse_ledger(&format!(
            "{{\"__ledger__\":1,\"content_fnv1a64\":\"{good}\",\"entry_count\":1,\
             \"generator\":\"test\"}}\n{body}"
        ));
        assert_eq!(
            disclose_ledger_faults_of(&sound),
            0,
            "ERROR(instrument): the control ledger is already faulted, so the arm below cannot \
             show a fault being reported: {:?}",
            sound.faults
        );
        // A hand-edit: the header's digest no longer describes the body.
        let hand_edited = registry::parse_ledger(&format!(
            "{{\"__ledger__\":1,\"content_fnv1a64\":\"0000000000000000\",\"entry_count\":1,\
             \"generator\":\"test\"}}\n{body}"
        ));
        assert!(
            disclose_ledger_faults_of(&hand_edited) > 0,
            "the ledger is unreadable and every form is about to decline, and the session was \
             told nothing: {:?}",
            hand_edited.faults
        );
        assert_ne!(
            disclose_ledger_faults_of(&sound),
            disclose_ledger_faults_of(&hand_edited),
            "both arms reported the same thing; the fault list is not being read"
        );
    }

    /// **§8.9.20 — the narrowing is disclosed, and both polarities exist.**
    ///
    /// An entry that records no `spec_digest` proves its form's *kernel* exactly and says nothing
    /// about which pipeline was built from it. That is a narrowing of what an entry means, and a
    /// narrowing nobody is told about is a quiet demotion of every proof it applies to — today,
    /// all of them.
    #[test]
    fn the_unrecorded_specialisation_population_is_named_on_every_run() {
        let _g = test_lock();
        const KEY: &str = "ai.onnx::Add/7+/f32,f32>f32/ew_binary_add_f32/static/n2";
        let digest = registry::shader_digest_for(&["ew_binary_add_f32"]).expect("a stem list");
        let build = |spec: &str| {
            let entry = format!(
                "{{\"key\":\"{KEY}\",\"verdict\":\"MATCH\",\"device\":\"d\",\"ort_build\":\"1\",\
                 \"tolerance\":\"t\",\"artifact\":\"a\",\"generated_at\":\"now\",\
                 \"shaders\":[\"ew_binary_add_f32\"],\"shader_digest\":\"{digest}\",\
                 \"spec_digest\":\"{spec}\",\"claimed_nodes\":1,\"dispatches_executed\":1}}"
            );
            let body = format!("{entry}\n");
            let d = format!("{:016x}", registry::fnv1a64(body.as_bytes()));
            registry::parse_ledger(&format!(
                "{{\"__ledger__\":1,\"content_fnv1a64\":\"{d}\",\"entry_count\":1,\
                 \"generator\":\"test\"}}\n{body}"
            ))
        };
        assert_eq!(
            disclose_specialisation_frame_of(&build("")),
            (0, 1),
            "an entry naming no specialisation was reported as one that names it"
        );
        assert_eq!(
            disclose_specialisation_frame_of(&build("beefbeefbeefbeef")),
            (1, 0),
            "an entry naming its specialisation was reported as unrecorded"
        );
        // The shipped artifact's state, stated where it can be read rather than inferred.
        let baked = registry::ledger();
        let (recorded, unrecorded) = disclose_specialisation_frame_of(baked);
        assert_eq!(
            recorded + unrecorded,
            baked.len(),
            "the two populations must partition the ledger"
        );
    }

    /// §8.9.18's second group: damage you **cannot locate** still faults the whole artifact.
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
