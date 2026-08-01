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
use crate::registry::OpStatus::{Ready, Staged};
use crate::registry::{
    ContribSchema, NodeView, OPSET_ANY, OPSET_STD_LLM, OPSET_STD_NORM_MAX, OpSpec, PINNED_BASELINE,
    UNEXERCISED,
};
use crate::require;

/// Staging reason for rows that need the reduction template rather than a bespoke kernel.
///
/// Distinct from `NO_SHADER` on purpose: this says "one piece of shared work unblocks all of
/// these", which is a schedule fact the table should carry.
///
/// **Retired 2026-07-31.** The shared row-reduction shader exists
/// (`shaders/glsl/simplified_layer_norm_{f32,f16}.comp`), so no row is blocked on it any more.
/// The constant stays for the doc comment above and is asserted *unused by any row* — a staged
/// reason that has stopped being true is a schedule claim the table would otherwise keep making.
pub const NEEDS_REDUCTION: &str =
    "it needs the shared row-reduction template, which has not been written yet";

/// `com.microsoft.SimplifiedLayerNormalization` — RMSNorm.
///
/// **Emitted in the default domain, not `com.microsoft`.** Both Foundry Local graphs carry
/// `node.domain == ""` for this op (§4.21), so it has two rows below and the `ai.onnx` one is on
/// `registry::ORT_FUSED_IN_DEFAULT_DOMAIN`.
pub static SIMPLIFIED_LAYER_NORM: ContribSchema = ContribSchema {
    baseline: PINNED_BASELINE,
    notes: "ContribOperators.md; observed with an EMPTY domain in both Foundry Local graphs",
    min_inputs: 2,
    max_inputs: 2,
    min_outputs: 1,
    max_outputs: 2,
    required_attrs: &[],
    known_attrs: &["axis", "epsilon", "stash_type"],
};

/// `com.microsoft.SkipSimplifiedLayerNormalization` — residual add fused into RMSNorm.
///
/// **Census-confirmed** against both Foundry Local graphs (§4.21): 3 inputs, 4 *declared* outputs
/// of which only slots 0 and 3 are occupied (1 and 2 are empty strings — mean and inv-std, which
/// the builder never asks for), `epsilon = 1e-5`. Output 3 is the residual sum, and it is what
/// feeds the next block, so a kernel that produces only output 0 breaks the residual stream.
pub static SKIP_SIMPLIFIED_LAYER_NORM: ContribSchema = ContribSchema {
    baseline: PINNED_BASELINE,
    notes: "ContribOperators.md; arity confirmed against Phi-3.5-mini-int4 and gpt-oss-20b",
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
    // gamma (scale) is required and shares X's dtype; the kernel reads both through one load.
    claim::typed_input(view, spec, 1, "scale")?;
    // The optional second output (`inv_std_var`) is not written by the kernel. Declining is the
    // honest answer: writing slot 0 and leaving slot 1 unwritten is exactly the defect class that
    // produced the never-written KV outputs, and nothing in either Foundry Local graph asks for
    // it — so a variant that supported it could not be exercised.
    let n_out = view.num_outputs();
    require!(
        n_out <= 1,
        Arity,
        "`{}` declares {n_out} outputs; this kernel writes slot 0 only, and an unwritten output \
         slot is worse than a declined node",
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
    "SimplifiedLayerNormalization",           Ms,     1 ..= OPSET_ANY,             FLOAT,  kernel!(None),  simplified_layer_norm,      templates::simplified_norm, Ready,                   schema: &SIMPLIFIED_LAYER_NORM;
    "SkipSimplifiedLayerNormalization",       Ms,     1 ..= OPSET_ANY,             FLOAT,  kernel!(None),  skip_simplified_layer_norm, templates::skip_norm,      Ready,                    schema: &SKIP_SIMPLIFIED_LAYER_NORM;

    // `RMSNormalization` is the same maths under the standard-domain spelling and runs the same
    // kernel. It stays staged behind `UNEXERCISED` rather than `NEEDS_REDUCTION` because the
    // reduction shader now exists — the blocker is no longer "no kernel", it is "no device has
    // run this row". No graph on this machine emits it, so flipping it would be a claim with no
    // instrument behind it (§8.9, R9).
    "RMSNormalization",                       Ai,     OPSET_STD_LLM ..= OPSET_STD_NORM_MAX, FLOAT,  kernel!(None),  simplified_layer_norm,      templates::simplified_norm, Staged(UNEXERCISED);

    // The same op again, in the **default domain**. Not a duplicate and not defensive coding: the
    // ORT GenAI model builder writes `SimplifiedLayerNormalization` with `node.domain == ""`, and
    // both Foundry Local models on this machine do it (`OP_COVERAGE.md` §4.21). Without this row
    // the node keys as `ai.onnx::SimplifiedLayerNormalization` and declines `[not-registered]`,
    // which reads as "we have never heard of this op" when the truth is "we registered it under
    // the wrong name". It carries the *fingerprint* rather than trusting its opset window, because
    // ONNX publishes no schema for it — see `registry::ORT_FUSED_IN_DEFAULT_DOMAIN`.
    //
    // **This is the row Phi-3.5's one remaining norm node keys against**: the single
    // `/model/layers.0/input_layernorm/LayerNorm` (every other layer's input norm is fused into
    // `SkipSimplifiedLayerNormalization`).
    "SimplifiedLayerNormalization",           Ai,     1 ..= OPSET_ANY,             FLOAT,  kernel!(None),  simplified_layer_norm,      templates::simplified_norm, Ready,                   schema: &SIMPLIFIED_LAYER_NORM;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{Domain, OpStatus};

    /// RMSNorm is emitted under two different domains by the same producer.
    ///
    /// The ORT GenAI builder writes `SimplifiedLayerNormalization` with an **empty** domain, which
    /// keys as `ai.onnx::`. Registering only the `com.microsoft::` spelling makes every real graph
    /// decline it as `[not-registered]` — "we have never heard of this op" — when in fact we had
    /// registered it under a name the producer does not use. Both Foundry Local models do this;
    /// §4.21. The `ai.onnx` row keeps the fingerprint because ONNX publishes no schema for it.
    #[test]
    fn rmsnorm_is_registered_under_both_domains_the_builder_emits() {
        let rows: Vec<_> = OPS
            .iter()
            .filter(|s| s.op_type == "SimplifiedLayerNormalization")
            .collect();
        assert_eq!(
            rows.len(),
            2,
            "one row per domain the producer actually emits"
        );
        let ai = rows
            .iter()
            .find(|s| s.domain == Domain::Ai)
            .expect("the empty-domain spelling");
        assert!(
            crate::registry::ORT_FUSED_IN_DEFAULT_DOMAIN.contains(&ai.op_type),
            "an ai.onnx row with no ONNX schema must be on the hazard register"
        );
        assert!(
            ai.schema.is_some(),
            "no opset can version this op, so the fingerprint is the only drift signal"
        );
        assert!(rows.iter().any(|s| s.domain == Domain::Ms));
    }

    /// `SkipSimplifiedLayerNormalization` declares four outputs and the fourth one matters.
    ///
    /// Census: slots 1 and 2 are empty strings in all 112 nodes across both models, but slot 3 —
    /// the residual sum — is always occupied and feeds the next block. A kernel that emits only
    /// output 0 silently breaks the residual stream.
    #[test]
    fn the_skip_norm_fingerprint_admits_the_residual_output() {
        assert_eq!(SKIP_SIMPLIFIED_LAYER_NORM.max_outputs, 4);
        assert_eq!(SKIP_SIMPLIFIED_LAYER_NORM.min_inputs, 3);
        assert!(SKIP_SIMPLIFIED_LAYER_NORM.knows("epsilon"));
    }

    #[test]
    fn last_axis_in_both_spellings() {
        assert!(axis_is_last(-1, 3));
        assert!(axis_is_last(2, 3), "the positive spelling of the same axis");
        assert!(!axis_is_last(1, 3));
        assert!(!axis_is_last(0, 3));
    }

    #[test]
    fn the_reduction_blocker_is_retired_and_only_the_unexercised_row_remains() {
        // The shared row-reduction shader now exists (`simplified_layer_norm_{f32,f16}.comp`),
        // so `NEEDS_REDUCTION` is no longer a true statement about any row. The one remaining
        // staged row — `RMSNormalization`, the standard-domain spelling — is blocked on evidence,
        // not on a kernel: no graph on this machine emits it, so it has never run on a device.
        // Keeping the *reason* accurate is the point of this test. A row staged behind a retired
        // blocker is a lie the table tells about its own schedule.
        let staged: Vec<_> = OPS
            .iter()
            .filter(|s| matches!(s.status, OpStatus::Staged(_)))
            .collect();
        let live: Vec<_> = OPS.iter().filter(|s| s.is_live()).collect();
        assert!(
            !live.is_empty(),
            "the norm rows backed by a shader should be live"
        );
        for s in &staged {
            assert!(
                !matches!(s.status, OpStatus::Staged(NEEDS_REDUCTION)),
                "{} is still staged behind NEEDS_REDUCTION, but the reduction shader exists",
                s.op_type
            );
            assert!(
                matches!(s.status, OpStatus::Staged(UNEXERCISED)),
                "{} is staged behind an unexpected blocker: {:?}",
                s.op_type,
                s.status
            );
        }
    }

    /// Every row that shares the RMSNorm kernel and is emitted by a producer we test against
    /// must be live. This is the claim-side falsifier for the coverage number: if
    /// `SimplifiedLayerNormalization` silently regresses to `Staged`, Phi-3.5's last norm node
    /// goes back to the CPU and this test goes red before the model does.
    #[test]
    fn both_spellings_of_simplified_layer_norm_are_live() {
        let rows: Vec<_> = OPS
            .iter()
            .filter(|s| s.op_type == "SimplifiedLayerNormalization")
            .collect();
        assert_eq!(rows.len(), 2, "one row per domain the builders emit");
        for r in rows {
            assert!(
                r.is_live(),
                "SimplifiedLayerNormalization ({:?}) must be Ready; got {:?}",
                r.domain,
                r.status
            );
            assert_eq!(
                r.translate as usize, templates::simplified_norm as usize,
                "a live row must point at the RMSNorm handler, not `unimplemented`"
            );
        }
    }

    /// `SkipSimplifiedLayerNormalization` is now Live/Ready — verify its status.
    #[test]
    fn skip_norm_is_live_not_staged() {
        let skip = OPS
            .iter()
            .find(|s| s.op_type == "SkipSimplifiedLayerNormalization")
            .expect("SkipSimplifiedLayerNormalization must be registered");
        assert!(
            skip.is_live(),
            "SkipSimplifiedLayerNormalization must be Live or Ready; got {:?}",
            skip.status
        );
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
        // opset, which the row's window already expresses. The exception is an op ORT registers in
        // the default domain with no ONNX schema at all — there the opset says nothing and only a
        // fingerprint can detect drift. See `registry::ORT_FUSED_IN_DEFAULT_DOMAIN` and §4.21.
        for s in OPS {
            match s.domain {
                Domain::Ms => assert!(s.schema.is_some(), "{} needs a fingerprint", s.op_type),
                Domain::Ai if crate::registry::ORT_FUSED_IN_DEFAULT_DOMAIN.contains(&s.op_type) => {
                    assert!(
                        s.schema.is_some(),
                        "{} has no ONNX schema, so its fingerprint is the only version contract",
                        s.op_type
                    )
                }
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
