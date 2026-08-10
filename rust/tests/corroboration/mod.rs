//! The **report model** for the live-device corroboration, with no Vulkan in it.
//!
//! This module is deliberately not a test target of its own (it lives in a subdirectory, so cargo
//! does not auto-discover it). It is `#[path]`-included by two files:
//!
//! * `tests/device_object_counter_corroboration.rs` — the `harness = false` binary that talks to a
//!   real driver and *renders* one of these reports;
//! * `tests/corroboration_report.rs` — an ordinary libtest target that exercises the rendering in
//!   **both** polarities without a device.
//!
//! # Why the rendering is separated from the probing
//!
//! The failure this guards against is a corroboration that always prints something reassuring.
//! A report renderer that lives inside the probe can only be exercised on a machine with the
//! right driver, so its "precondition not met" and "contradiction found" branches would be
//! written once and never run again — and the branch nobody runs is the branch that is wrong.
//! Splitting the model out makes every branch reachable from a host-free test.

#![allow(dead_code)] // Each includer uses a different subset; that is the point of sharing it.

/// What the driver did when asked for one more descriptor set than the pool's `maxSets`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Exhaustion {
    /// The driver refused. The refusable path the seam is written for exists on this machine.
    Refused,
    /// The driver returned success anyway. **This is allowed**: over-allocation past `maxSets` is
    /// not a spec-guaranteed failure, which is exactly why the host-free seam tests — not this
    /// binary — are the authority on the success-only rule.
    Allowed,
}

/// The outcome of one corroboration run.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Report {
    /// A precondition was not met and nothing was observed. **Not a pass.**
    Skip { reason: String },
    /// The probe ran and the driver refused an over-allocation, as the seam's `Failed` polarity
    /// assumes is possible.
    Corroborated { device: String },
    /// The probe ran and could not settle the question, because the driver was lenient.
    Inconclusive { device: String, reason: String },
    /// The probe observed something that contradicts the seam's premise. **The only failing
    /// outcome**, and the only one that exits non-zero.
    Contradicted { device: String, detail: String },
}

/// The one-line token a reader (or a grep) keys on. Stable and unique per outcome.
pub const TOKEN_SKIP: &str = "SKIP";
pub const TOKEN_CORROBORATED: &str = "CORROBORATED";
pub const TOKEN_INCONCLUSIVE: &str = "INCONCLUSIVE";
pub const TOKEN_CONTRADICTED: &str = "CONTRADICTED";

/// The sentence every report carries, whatever its outcome.
///
/// It is not optional and it is not a footnote: a reader who sees `CORROBORATED` and stops there
/// must still have been told that this binary is *not* the authority on the counter rule.
pub const AUTHORITY_NOTE: &str = "AUTHORITY: the success-only counter rule is settled by the \
    host-free seam tests in rust/src/counters.rs \
    (a_failed_vulkan_call_increments_no_device_object_counter). Descriptor-pool exhaustion past \
    maxSets is NOT spec-guaranteed, so this binary can corroborate that a refusable path exists \
    on this machine — it can never be the evidence that a refusal counts nothing.";

impl Report {
    /// The stable token for this outcome.
    pub fn token(&self) -> &'static str {
        match self {
            Report::Skip { .. } => TOKEN_SKIP,
            Report::Corroborated { .. } => TOKEN_CORROBORATED,
            Report::Inconclusive { .. } => TOKEN_INCONCLUSIVE,
            Report::Contradicted { .. } => TOKEN_CONTRADICTED,
        }
    }

    /// Process exit code. **Only a contradiction fails.**
    ///
    /// A skip does not fail, because a machine without a Vulkan 1.1 compute device has not
    /// disagreed with anything. An inconclusive result does not fail, because a lenient driver is
    /// conformant. Turning either into a red would train a reader to ignore this binary, which is
    /// the fastest way to make a corroboration worthless.
    pub fn exit_code(&self) -> i32 {
        match self {
            Report::Contradicted { .. } => 1,
            _ => 0,
        }
    }

    /// The full text written to stdout. Multi-line, and every line is legible on its own.
    pub fn render(&self) -> String {
        let body = match self {
            Report::Skip { reason } => format!(
                "{TOKEN_SKIP}(precondition not met): {reason}\n  NOTHING WAS OBSERVED. This is \
                 not a pass and must never be read as one — no descriptor pool was created, so \
                 no claim about descriptor pools was tested here."
            ),
            Report::Corroborated { device } => format!(
                "{TOKEN_CORROBORATED}: on `{device}`, vkAllocateDescriptorSets REFUSED a set \
                 beyond the pool's maxSets, and the refusal produced no descriptor-set handle.\n  \
                 The refusable path that the seam's Failed polarity is written for exists on this \
                 machine. It is not guaranteed to exist on any other."
            ),
            Report::Inconclusive { device, reason } => format!(
                "{TOKEN_INCONCLUSIVE}: on `{device}`, {reason}\n  The question this binary asks \
                 is unanswered on this machine. That is a conformant driver, not a defect, and \
                 not a failure."
            ),
            Report::Contradicted { device, detail } => format!(
                "{TOKEN_CONTRADICTED}: on `{device}`, {detail}\n  A call that reported failure \
                 also produced an object. Any counter that trusts the return code is now \
                 counting the wrong population."
            ),
        };
        format!("[device-object-counter corroboration]\n  {body}\n  {AUTHORITY_NOTE}\n")
    }
}

/// Turn an observed exhaustion outcome into a report. Pure; both branches are host-free.
pub fn classify(device: &str, observed: Exhaustion, handle_was_written: bool) -> Report {
    match (observed, handle_was_written) {
        // A failure that still handed back a live handle breaks the premise the counters rest on.
        (Exhaustion::Refused, true) => Report::Contradicted {
            device: device.to_string(),
            detail: "vkAllocateDescriptorSets returned an error AND wrote a non-null \
                     VkDescriptorSet handle"
                .to_string(),
        },
        (Exhaustion::Refused, false) => Report::Corroborated {
            device: device.to_string(),
        },
        (Exhaustion::Allowed, _) => Report::Inconclusive {
            device: device.to_string(),
            reason: "vkAllocateDescriptorSets ALLOWED a set beyond the pool's maxSets. \
                     Over-allocation is not a spec-guaranteed failure, so no refusal could be \
                     forced here"
                .to_string(),
        },
    }
}
