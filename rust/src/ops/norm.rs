//! Normalization — the RMSNorm family the LLM path runs on.
//!
//! # Two spellings of the same maths, and we need both
//!
//! An **ORT-GenAI-built** Qwen3 graph does not contain `LayerNormalization`; it contains
//! `com.microsoft.SimplifiedLayerNormalization` (RMSNorm) and
//! `com.microsoft.SkipSimplifiedLayerNormalization` (residual-add + RMSNorm fused). Both appear
//! twice per decoder layer, plus the Q/K norms, so a 28-layer model has roughly 60 of them.
//!
//! A **`onnx-genai-models` (mobius)-built** Qwen3 graph — Justin's own builder — contains none of
//! those. It emits the standard-domain **`RMSNormalization` (ai.onnx opset 23)** instead, four
//! times per decoder layer once Q/K norm is counted, and does not fuse the residual add at all.
//!
//! So the same model, built by two of our own toolchains, produces two disjoint sets of norm
//! nodes, and a registry holding only the contrib spellings declines 100% of the second. Both
//! spellings are therefore in the table. This is the general shape of the lesson recorded in
//! `OP_COVERAGE.md` §4.16: op coverage is relative to a *producer*, not to a model architecture.
//!
//! # These are the fusion the allowlist is for
//!
//! `OP_COVERAGE.md` §5.6's rule is compose-before-bespoke, with a short allowlist of fusions worth
//! writing by hand. `SkipSimplifiedLayerNormalization` is on it: decomposed, it is an `Add`, a
//! square, a mean reduction, an `rsqrt` and two multiplies — five dispatches and four round trips
//! through device memory for a bandwidth-bound operation. Fused, it is one pass. That is the
//! entire justification for a fusion and it is the same reason the exporter fused it.
//!
//! The reduction kernel is shared by all four rows here regardless of spelling, which is why the
//! standard-domain rows cost a table row each rather than a kernel each.

use crate::kernel;
use crate::ops::common::claim::{self, ClaimResult};
use crate::ops::common::dtype::FLOAT;
use crate::ops::common::templates;
use crate::registry::OpStatus::Staged;
use crate::registry::{
    ContribSchema, NodeView, OPSET_ANY, OPSET_STD_LLM, OPSET_STD_NORM_MAX, OpSpec, PINNED_BASELINE,
};
use crate::require;

/// Staging reason for rows that need the reduction template rather than a bespoke kernel.
///
/// Distinct from `NO_SHADER` on purpose: this says "one piece of shared work unblocks all of
/// these", which is a schedule fact the table should carry.
pub const NEEDS_REDUCTION: &str =
    "it needs the shared row-reduction template, which has not been written yet";

/// `com.microsoft.SimplifiedLayerNormalization` — RMSNorm.
pub static SIMPLIFIED_LAYER_NORM: ContribSchema = ContribSchema {
    baseline: PINNED_BASELINE,
    notes: "read from ContribOperators.md",
    min_inputs: 2,
    max_inputs: 2,
    min_outputs: 1,
    max_outputs: 2,
    required_attrs: &[],
    known_attrs: &["axis", "epsilon", "stash_type"],
};

/// `com.microsoft.SkipSimplifiedLayerNormalization` — residual add fused into RMSNorm.
pub static SKIP_SIMPLIFIED_LAYER_NORM: ContribSchema = ContribSchema {
    baseline: PINNED_BASELINE,
    notes: "read from ContribOperators.md",
    min_inputs: 3,
    max_inputs: 5,
    min_outputs: 1,
    max_outputs: 4,
    required_attrs: &[],
    known_attrs: &["epsilon"],
};

/// The only normalization axis the kernel implements: the last one.
///
/// Every LLM norm is over the hidden dimension, so `axis = -1` (or its positive spelling) covers
/// the target models. A middle-axis norm is a strided reduction — a different kernel, correctly
/// declined rather than mis-dispatched.
pub const fn axis_is_last(axis: i64, rank: usize) -> bool {
    axis == -1 || (rank > 0 && axis == (rank as i64) - 1)
}

fn simplified_layer_norm(view: &NodeView<'_>, spec: &OpSpec) -> ClaimResult {
    let edge = claim::input_edge(view, spec, 0)?;
    claim::check_dtype(spec, &edge, "input 0")?;
    claim::check_shape(spec, &edge, "input 0")?;
    let rank = edge.rank().unwrap_or(0);
    let axis = view.attr_int("axis").unwrap_or(-1);
    require!(
        axis_is_last(axis, rank),
        Attribute,
        "`{}` normalizes over axis {axis} of a rank-{rank} tensor; this EP implements the last \
         axis, which is what every LLM norm uses",
        spec.op_type
    );
    Ok(())
}

fn skip_simplified_layer_norm(view: &NodeView<'_>, spec: &OpSpec) -> ClaimResult {
    let edge = claim::input_edge(view, spec, 0)?;
    claim::check_dtype(spec, &edge, "input 0")?;
    claim::check_shape(spec, &edge, "input 0")?;
    // The skip tensor must be present and the same shape class as the input; the fused kernel adds
    // them in one pass.
    claim::typed_input(view, spec, 1, "skip")?;
    Ok(())
}

crate::op_table! {
    //  op                                    domain  opsets                       caps    kernel          claim                       translate                  status                    schema
    "SimplifiedLayerNormalization",           Ms,     1 ..= OPSET_ANY,             FLOAT,  kernel!(None),  simplified_layer_norm,      templates::unimplemented,  Staged(NEEDS_REDUCTION),  schema: &SIMPLIFIED_LAYER_NORM;
    "SkipSimplifiedLayerNormalization",       Ms,     1 ..= OPSET_ANY,             FLOAT,  kernel!(None),  skip_simplified_layer_norm, templates::unimplemented,  Staged(NEEDS_REDUCTION),  schema: &SKIP_SIMPLIFIED_LAYER_NORM;
    "RMSNormalization",                       Ai,     OPSET_STD_LLM ..= OPSET_STD_NORM_MAX, FLOAT,  kernel!(None),  simplified_layer_norm,      templates::unimplemented,  Staged(NEEDS_REDUCTION);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{Domain, OpStatus};

    #[test]
    fn last_axis_in_both_spellings() {
        assert!(axis_is_last(-1, 3));
        assert!(axis_is_last(2, 3), "the positive spelling of the same axis");
        assert!(!axis_is_last(1, 3));
        assert!(!axis_is_last(0, 3));
    }

    #[test]
    fn both_norms_share_one_blocker() {
        // The point of a distinct staging reason: this reads as "write the reduction template and
        // two rows go live", which is a schedule statement the table makes for you.
        for s in OPS {
            assert!(
                matches!(s.status, OpStatus::Staged(NEEDS_REDUCTION)),
                "{} is staged behind something else",
                s.op_type
            );
        }
    }

    #[test]
    fn the_skip_form_admits_the_fused_extra_outputs() {
        // ORT's skip norm optionally emits mean, inv_std_var and the pre-norm sum. Claiming only
        // the 1-output form would decline the nodes GenAI actually emits.
        assert!(SKIP_SIMPLIFIED_LAYER_NORM.max_outputs >= 3);
    }

    #[test]
    fn every_contrib_row_here_is_fingerprinted() {
        // Standard-domain rows carry no fingerprint by design: `ai.onnx` ops are versioned by
        // opset, which the row's window already expresses. Only `com.microsoft` needs C2.
        for s in OPS {
            match s.domain {
                Domain::Ms => assert!(s.schema.is_some(), "{} needs a fingerprint", s.op_type),
                Domain::Ai => assert!(
                    s.schema.is_none(),
                    "{} is standard-domain; its opset window is its version contract",
                    s.op_type
                ),
            }
        }
    }

    #[test]
    fn both_spellings_of_rmsnorm_are_registered() {
        // The finding from the 2026-07-29 crate review: ORT GenAI emits
        // `com.microsoft::SimplifiedLayerNormalization`, Justin's `onnx-genai-models` emits
        // `ai.onnx::RMSNormalization` @ opset 23, and they are the same maths. Registering only
        // one of them silently halves our coverage depending on who built the model.
        let contrib = OPS
            .iter()
            .find(|s| s.op_type == "SimplifiedLayerNormalization")
            .expect("the ORT GenAI spelling");
        let standard = OPS
            .iter()
            .find(|s| s.op_type == "RMSNormalization")
            .expect("the standard-domain spelling");
        assert_eq!(contrib.domain, Domain::Ms);
        assert_eq!(standard.domain, Domain::Ai);
        assert_eq!(
            standard.min_opset, OPSET_STD_LLM,
            "RMSNormalization does not exist before opset 23"
        );
        // Same claim predicate, and it must stay that way: they are one kernel.
        assert!(std::ptr::fn_addr_eq(contrib.claim, standard.claim));
    }

    /// `RMSNormalization` has exactly one schema version, so its window is closed at 23.
    ///
    /// Verified against onnx v1.22.0 on 2026-07-29: it is absent from the opset-24 section of
    /// `onnx/defs/operator_sets.h` and still lives in `defs.cc` rather than `old.cc`, so version 23
    /// is current at opset 27. `onnxruntime/mobius` builds at opset 24 but the *node* still resolves
    /// to schema version 23 — `Node_GetSinceVersion` reports the op's schema version, not the
    /// model's opset, which is why a closed window here does not exclude opset-24 graphs.
    ///
    /// mobius also emits this op at **rank 4** for Qwen3's per-head Q/K norm, with `axis = -1`.
    /// The shared predicate must not assume rank 3.
    #[test]
    fn rmsnorm_window_is_closed_at_its_only_schema_version() {
        let standard = OPS
            .iter()
            .find(|s| s.op_type == "RMSNormalization")
            .expect("the standard-domain spelling");
        assert_eq!((standard.min_opset, standard.max_opset), (23, 23));
        assert_ne!(
            standard.max_opset, OPSET_ANY,
            "an unread future revision must decline as [opset], never be claimed"
        );
    }
}
