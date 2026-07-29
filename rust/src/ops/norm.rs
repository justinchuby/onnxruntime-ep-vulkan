//! Normalization — the RMSNorm family the LLM path runs on.
//!
//! # Why these are contrib ops and why that is fine
//!
//! A GenAI-built Qwen3 graph does not contain `LayerNormalization`; it contains
//! `com.microsoft.SimplifiedLayerNormalization` (RMSNorm) and
//! `com.microsoft.SkipSimplifiedLayerNormalization` (residual-add + RMSNorm fused). Both appear
//! twice per decoder layer, plus the Q/K norms, so a 28-layer model has roughly 60 of them. They
//! are not XL — they are one reduction template away — but they are unavoidable.
//!
//! # These are the fusion the allowlist is for
//!
//! `OP_COVERAGE.md` §5.6's rule is compose-before-bespoke, with a short allowlist of fusions worth
//! writing by hand. `SkipSimplifiedLayerNormalization` is on it: decomposed, it is an `Add`, a
//! square, a mean reduction, an `rsqrt` and two multiplies — five dispatches and four round trips
//! through device memory for a bandwidth-bound operation. Fused, it is one pass. That is the
//! entire justification for a fusion and it is the same reason the exporter fused it.

use crate::kernel;
use crate::ops::common::claim::{self, ClaimResult};
use crate::ops::common::dtype::FLOAT;
use crate::ops::common::templates;
use crate::registry::OpStatus::Staged;
use crate::registry::{ContribSchema, PINNED_BASELINE, NodeView, OPSET_ANY, OpSpec};
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
    //  op                                    domain  opsets            caps    kernel          claim                       translate                  status                    schema
    "SimplifiedLayerNormalization",           Ms,     1 ..= OPSET_ANY,  FLOAT,  kernel!(None),  simplified_layer_norm,      templates::unimplemented,  Staged(NEEDS_REDUCTION),  schema: &SIMPLIFIED_LAYER_NORM;
    "SkipSimplifiedLayerNormalization",       Ms,     1 ..= OPSET_ANY,  FLOAT,  kernel!(None),  skip_simplified_layer_norm, templates::unimplemented,  Staged(NEEDS_REDUCTION),  schema: &SKIP_SIMPLIFIED_LAYER_NORM;
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
    fn every_row_here_is_contrib_and_fingerprinted() {
        for s in OPS {
            assert_eq!(s.domain, Domain::Ms);
            assert!(s.schema.is_some());
        }
    }
}
