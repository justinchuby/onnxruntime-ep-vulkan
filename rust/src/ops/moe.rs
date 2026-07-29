//! Mixture of experts — `QMoE` first, `MoE` second.
//!
//! # Why `QMoE` before `MoE`
//!
//! `OP_COVERAGE.md` §4.12: the quantized form is what the GenAI builder emits and what ORT's
//! WebGPU EP chose to implement; the float form is commented out there and was declined outright
//! by the reference MLX project. A MoE model people actually run is int4, so the float path is the
//! generalization, not the starting point.
//!
//! # The hard part is data-dependent routing on a pre-recorded command buffer
//!
//! `DESIGN.md`'s record-once/replay-many model means the expert assignment is not known when the
//! command buffer is recorded. Two formulations survive that constraint:
//!
//! 1. **Masked dense** — every expert processes every token, masked. Correct, trivially
//!    recordable, and wastes roughly `1 - k/num_experts` of the FLOPs (7/8 at top-2-of-8).
//! 2. **Indirect dispatch** — a routing pass writes workgroup counts into a buffer and the expert
//!    pass is a `vkCmdDispatchIndirect`. The command buffer stays static while the *work* is
//!    dynamic.
//!
//! Option 2 is the reason `OP_COVERAGE.md` §3.3 argues we are better positioned here than a lazy
//! graph-executor backend: indirect dispatch is a native Vulkan capability. It needs an engine
//! seam that does not exist yet — an indirect-dispatch `KernelRequest` variant — and that is a
//! request to Switch, recorded here rather than assumed.

use crate::kernel;
use crate::ops::common::claim::{self, ClaimResult};
use crate::ops::common::dtype::FLOAT;
use crate::ops::common::templates;
use crate::registry::OpStatus::Staged;
use crate::registry::{ContribSchema, MAIN_BASELINE, NodeView, OPSET_ANY, OpSpec, XL_KERNEL};
use crate::require;

/// `com.microsoft.QMoE`.
///
/// **Fingerprint confidence: low.** The schema has two shapes in the wild — a 14-input form the
/// CPU tests still use and a wider form on main with FP4/FP8 global and activation scales. The
/// input range below spans both deliberately; the attribute set is the part that actually detects
/// drift, and an unknown attribute declines rather than guesses.
pub static QMOE: ContribSchema = ContribSchema {
    baseline: MAIN_BASELINE,
    notes: "14-input and extended forms both admitted; 11-input form with 8 occupied slots and \
            activation_alpha/beta observed in gpt-oss-20b; re-verify against a release before going live",
    min_inputs: 8,
    max_inputs: 21,
    min_outputs: 1,
    max_outputs: 1,
    required_attrs: &["k"],
    known_attrs: &[
        "activation_type",
        // Observed on every QMoE node in gpt-oss-20b (`activation_alpha = 1.702`,
        // `activation_beta = 1.0`). They were missing here, so a real node declined as
        // `[contrib-schema]` — the right answer for the wrong reason. Knowing an attribute is not
        // claiming it: the predicate still has to pin the values the kernel implements.
        "activation_alpha",
        "activation_beta",
        "k",
        "normalize_routing_weights",
        "quant_type",
        "swiglu_fusion",
        "swiglu_limit",
        "use_sparse_mixer",
        "weights_prepacked",
        "expert_weight_bits",
        "block_size",
    ],
};

/// `com.microsoft.MoE` — the float-expert form.
pub static MOE: ContribSchema = ContribSchema {
    baseline: MAIN_BASELINE,
    notes: "re-verify against a release before going live",
    min_inputs: 4,
    max_inputs: 8,
    min_outputs: 1,
    max_outputs: 1,
    required_attrs: &["k"],
    known_attrs: &[
        "activation_type",
        "k",
        "normalize_routing_weights",
        "use_sparse_mixer",
        "swiglu_fusion",
        "swiglu_limit",
    ],
};

/// Routing widths the first kernel implements.
///
/// Top-1 and top-2 cover every MoE LLM shipping today. A general `k` is a different reduction and
/// a different masked-dense cost model, so it declines until a real model needs it.
pub const fn supports_top_k(k: i64) -> bool {
    matches!(k, 1 | 2)
}

/// Expert activations this EP implements. `swiglu` is what Qwen-MoE uses.
pub const EXPERT_ACTIVATIONS: &[&str] = &["silu", "swiglu", "relu", "gelu"];

fn qmoe(view: &NodeView<'_>, spec: &OpSpec) -> ClaimResult {
    claim::typed_input(view, spec, 0, "input")?;
    let k = claim::required_int(view, spec, "k")?;
    require!(
        supports_top_k(k),
        Attribute,
        "`{}` routes top-{k}; this EP implements top-1 and top-2",
        spec.op_type
    );
    claim::attr_string_in(view, spec, "activation_type", EXPERT_ACTIVATIONS, "relu")?;
    // Sparse mixer changes the routing distribution, not just its speed.
    claim::attr_int_is(view, spec, "use_sparse_mixer", 0)?;
    // Only int weights; the FP4/FP8 expert forms are a separate unpack.
    claim::attr_string_in(view, spec, "quant_type", &["int"], "int")?;
    Ok(())
}

crate::op_table! {
    //  op        domain  opsets            caps    kernel          claim          translate                  status              schema
    "QMoE",       Ms,     1 ..= OPSET_ANY,  FLOAT,  kernel!(None),  qmoe,          templates::unimplemented,  Staged(XL_KERNEL),  schema: &QMOE;
    "MoE",        Ms,     1 ..= OPSET_ANY,  FLOAT,  kernel!(None),  claim::never,  templates::unimplemented,  Staged(XL_KERNEL),  schema: &MOE;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{Domain, OpStatus};

    #[test]
    fn routing_widths() {
        assert!(supports_top_k(1));
        assert!(supports_top_k(2), "the common MoE LLM configuration");
        assert!(!supports_top_k(4));
        assert!(!supports_top_k(0));
    }

    /// The only real MoE graph we have routes **top-4**, which we decline.
    ///
    /// gpt-oss-20b: 24 `QMoE` nodes, all with `k = 4`, `activation_type = "swiglu"`,
    /// `activation_alpha = 1.702`, `expert_weight_bits = 4`, `use_sparse_mixer = 0` (§4.21).
    /// "This EP implements top-1 and top-2" was written from schema reading and matches no model
    /// on this disk. The decline is still correct — the kernel does not exist — but the T5b scope
    /// is wrong, and this test exists so that raising the bound is a deliberate act with the
    /// evidence attached rather than a quiet edit.
    #[test]
    fn the_only_real_moe_graph_we_have_routes_top_4_and_we_decline_it() {
        assert!(
            !supports_top_k(4),
            "if this is being relaxed, the routing kernel must actually handle top-4 and \
             gpt-oss-20b must be in the conformance set"
        );
        for attr in [
            "activation_alpha",
            "activation_beta",
            "expert_weight_bits",
            "k",
        ] {
            assert!(
                QMOE.knows(attr),
                "`{attr}` is on every gpt-oss-20b QMoE node; not knowing it attributes the \
                 decline to schema drift instead of to the predicate that actually refused"
            );
        }
    }

    #[test]
    fn qmoe_comes_before_moe() {
        // Ordering in the table is documentation: the quantized form is the one that ships.
        assert_eq!(OPS[0].op_type, "QMoE");
        assert_eq!(OPS[1].op_type, "MoE");
    }

    #[test]
    fn swiglu_is_a_known_qmoe_attribute() {
        // Qwen-MoE's expert activation. Not knowing it would decline every Qwen-MoE node as drift.
        assert!(QMOE.knows("swiglu_fusion"));
        assert!(QMOE.knows("swiglu_limit"));
    }

    #[test]
    fn every_row_here_is_contrib_and_staged() {
        for s in OPS {
            assert_eq!(s.domain, Domain::Ms);
            assert!(s.schema.is_some());
            assert!(matches!(s.status, OpStatus::Staged(_)));
        }
    }
}
