//! Linear attention and SSM state — the Qwen3.5 hybrid path.
//!
//! # What this covers
//!
//! Qwen3.5 is a hybrid: most layers are linear-attention (gated delta net) rather than softmax
//! attention, with a short causal depthwise convolution carrying its own state. Both ops carry
//! recurrent state in and out — the same KV-cache-shaped problem as `GroupQueryAttention`, with the
//! same M2 dependency.
//!
//! # Two spellings, both real
//!
//! Both ops exist **twice**: as `com.microsoft` contrib ops on ORT main, and — since onnx v1.22.0
//! — as standard `ai.onnx` ops at opset 27. That is `OP_COVERAGE.md` §8.5 again: coverage is
//! relative to a producer at a version, and an exporter targeting the standard domain emits nodes
//! our contrib rows would never see. Registering only one spelling would make the hybrid path run
//! for one producer and fall back for the other, with no coverage number saying which. §4.20.
//!
//! # Why this is the hardest row in the plan
//!
//! `OP_COVERAGE.md` §11.1 item 4: `LinearAttention` with `update_rule = "gated_delta"` is a
//! genuinely novel kernel with, as far as I could find, no open reference Vulkan implementation to
//! learn from. The llama.cpp accelerant (OQ-M6) that helps with attention and quantized GEMM helps
//! least here. **Per-kernel licensing record: independent implementation; no third-party source is
//! expected to be adapted.**
//!
//! # Claim one rule at a time
//!
//! `update_rule` selects between `linear`, `gated`, `delta` and `gated_delta`. These are not
//! parameters of one kernel, they are four kernels. Claiming the op and then dispatching the wrong
//! recurrence is precisely the failure mode `OP_COVERAGE.md` §7 exists to prevent, so the
//! predicate names the one rule the kernel implements and declines the rest.

use crate::kernel;
use crate::ops::common::claim::{self, ClaimResult};
use crate::ops::common::dtype::FLOAT;
use crate::ops::common::templates;
use crate::registry::OpStatus::Staged;
use crate::registry::{
    ContribSchema, MAIN_BASELINE, NodeView, OPSET_ANY, OPSET_STD_LLM, OpSpec, XL_KERNEL,
};
use crate::require;

/// `com.microsoft.LinearAttention`.
///
/// **Fingerprint confidence: low — this op exists on ORT main, not in the 1.28 release.** That is
/// exactly the hazard the contrib-schema check was built for: there is no opset to pin, the schema
/// can change without any version we can observe, and the only defence is a fingerprint plus a
/// decline that says which release we were reading. Re-verify before this row goes live.
pub static LINEAR_ATTENTION: ContribSchema = ContribSchema {
    baseline: MAIN_BASELINE,
    notes: "NOT present in the 1.28 release at all; re-verify before going live",
    min_inputs: 3,
    max_inputs: 6,
    min_outputs: 1,
    max_outputs: 2,
    required_attrs: &[],
    known_attrs: &[
        "q_num_heads",
        "kv_num_heads",
        "update_rule",
        "scale",
        "chunk_size",
    ],
};

/// `com.microsoft.CausalConvWithState` — Qwen3.5's short causal depthwise conv with carried state.
///
/// **Fingerprint confidence: low, for the same reason as [`LINEAR_ATTENTION`].**
pub static CAUSAL_CONV_WITH_STATE: ContribSchema = ContribSchema {
    baseline: MAIN_BASELINE,
    notes: "NOT present in the 1.28 release at all; re-verify before going live",
    min_inputs: 2,
    max_inputs: 4,
    min_outputs: 1,
    max_outputs: 2,
    required_attrs: &[],
    known_attrs: &["activation", "ndim"],
};

/// The one recurrence this EP implements first.
///
/// Qwen3.5's rule. `OP_COVERAGE.md` §4.13: do this one, not the general case.
pub const FIRST_UPDATE_RULE: &str = "gated_delta";

/// Activations the fused causal conv folds in. `none` and `silu` are what the Mamba/GDN-style
/// blocks use; anything else is a different kernel epilogue.
pub const CONV_ACTIVATIONS: &[&str] = &["none", "silu"];

/// `LinearAttention` — claim `gated_delta` only.
fn linear_attention(view: &NodeView<'_>, spec: &OpSpec) -> ClaimResult {
    claim::typed_input(view, spec, 0, "query")?;
    claim::attr_string_in(
        view,
        spec,
        "update_rule",
        &[FIRST_UPDATE_RULE],
        FIRST_UPDATE_RULE,
    )?;
    let q_heads = view.attr_int("q_num_heads").unwrap_or(0);
    let kv_heads = view.attr_int("kv_num_heads").unwrap_or(0);
    require!(
        q_heads > 0 && kv_heads > 0 && q_heads % kv_heads == 0,
        Attribute,
        "`{}` has q_num_heads = {q_heads}, kv_num_heads = {kv_heads}; the kernel needs a whole \
         head ratio",
        spec.op_type
    );
    Ok(())
}

/// `CausalConvWithState` — 1-D only, with a state tensor the kernel carries.
fn causal_conv_with_state(view: &NodeView<'_>, spec: &OpSpec) -> ClaimResult {
    claim::typed_input(view, spec, 0, "input")?;
    claim::attr_int_is(view, spec, "ndim", 1)?;
    claim::attr_string_in(view, spec, "activation", CONV_ACTIVATIONS, "none")?;
    Ok(())
}

/// `ai.onnx::LinearAttention`, opset 27 — the standard-domain spelling of [`LINEAR_ATTENTION`].
///
/// Verified conservatively: the schema names an `update_rule` attribute with the same four values
/// as the contrib op, takes `query`/`key`/`value` plus optional `past_state`/`decay`/`beta`, and
/// carries the recurrent state type separately from the activation type. The *rest* of its
/// attribute vocabulary — head counts, chunking — was not read line for line, so the predicate
/// asserts only what was, and everything unread is protected by the row being staged.
fn std_linear_attention(view: &NodeView<'_>, spec: &OpSpec) -> ClaimResult {
    claim::typed_input(view, spec, 0, "query")?;
    claim::attr_string_in(
        view,
        spec,
        "update_rule",
        &[FIRST_UPDATE_RULE],
        FIRST_UPDATE_RULE,
    )?;
    Ok(())
}

/// `ai.onnx::CausalConvWithState`, opset 27 — the standard-domain spelling of the fused conv.
///
/// Weight shape is `(channels, 1, k)`: depthwise, 1-D, which is why there is no `ndim` attribute to
/// pin the way the contrib row does — the rank is fixed by the schema instead.
fn std_causal_conv_with_state(view: &NodeView<'_>, spec: &OpSpec) -> ClaimResult {
    claim::typed_input(view, spec, 0, "input")?;
    claim::attr_string_in(view, spec, "activation", CONV_ACTIVATIONS, "none")?;
    Ok(())
}

crate::op_table! {
    //  op                      domain  opsets            caps    kernel          claim                    translate                  status              schema
    "LinearAttention",          Ms,     1 ..= OPSET_ANY,  FLOAT,  kernel!(None),  linear_attention,        templates::unimplemented,  Staged(XL_KERNEL),  schema: &LINEAR_ATTENTION;
    "CausalConvWithState",      Ms,     1 ..= OPSET_ANY,  FLOAT,  kernel!(None),  causal_conv_with_state,  templates::unimplemented,  Staged(XL_KERNEL),  schema: &CAUSAL_CONV_WITH_STATE;

    // The `ai.onnx` half, new at opset 27 (onnx v1.22.0). These are not aliases of the contrib rows
    // above: they are a second producer's spelling of the same computation, and §8.5's rule —
    // coverage is relative to a producer *at a version* — means both spellings need rows or the
    // Qwen3.5 hybrid path runs only for whoever exported it. Windows are closed at exactly 27
    // because that is both the version they were introduced at and the newest that exists. §4.20.
    "LinearAttention",          Ai,     OPSET_STD_LLM ..= OPSET_STD_LLM,  FLOAT,  kernel!(None),  std_linear_attention,        templates::unimplemented,  Staged(XL_KERNEL);
    "CausalConvWithState",      Ai,     OPSET_STD_LLM ..= OPSET_STD_LLM,  FLOAT,  kernel!(None),  std_causal_conv_with_state,  templates::unimplemented,  Staged(XL_KERNEL);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{Domain, OpStatus};

    fn contrib_rows() -> impl Iterator<Item = &'static OpSpec> {
        OPS.iter().filter(|s| s.domain == Domain::Ms)
    }

    #[test]
    fn the_qwen35_rule_is_the_one_we_claim_first() {
        assert_eq!(FIRST_UPDATE_RULE, "gated_delta");
    }

    #[test]
    fn both_rows_admit_a_state_output() {
        for s in contrib_rows() {
            let schema = s.schema.expect("contrib row");
            assert_eq!(
                schema.max_outputs, 2,
                "{} carries recurrent state out; without a second output the graph cannot thread \
                 state across tokens",
                s.op_type
            );
        }
    }

    #[test]
    fn both_rows_say_they_are_unverified_against_a_release() {
        // The honesty check: these ops are main-branch only, and the fingerprint has to say so,
        // because a `[contrib-schema]` decline and `epctl --dump-capabilities` both quote it.
        for s in contrib_rows() {
            let schema = s.schema.expect("contrib row");
            assert!(
                schema.notes.contains("re-verify"),
                "{} claims a verification it does not have",
                s.op_type
            );
            assert!(
                schema.provenance().contains("main"),
                "{} must not report a release baseline it does not have",
                s.op_type
            );
        }
    }

    #[test]
    fn every_row_here_is_staged() {
        for s in OPS {
            assert!(matches!(s.status, OpStatus::Staged(_)));
        }
    }

    /// Both SSM ops exist in two domains, and both spellings must be registered.
    ///
    /// This is §8.5 applied to the hybrid path: `com.microsoft` is what ORT's own exporter emits
    /// today, `ai.onnx` at opset 27 is what onnx v1.22.0 standardised. Registering only one means
    /// the Qwen3.5 linear-attention layers run on the EP for one producer and fall back for the
    /// other, and the coverage number would not say which. §4.20.
    #[test]
    fn both_ssm_ops_are_registered_in_both_domains() {
        for op in ["LinearAttention", "CausalConvWithState"] {
            let rows: Vec<_> = OPS.iter().filter(|s| s.op_type == op).collect();
            assert_eq!(rows.len(), 2, "{op} needs a contrib row and a standard row");
            assert!(rows.iter().any(|s| s.domain == Domain::Ms));
            let std_row = rows
                .iter()
                .find(|s| s.domain == Domain::Ai)
                .unwrap_or_else(|| panic!("{op} has no ai.onnx row"));
            assert_eq!(
                (std_row.min_opset, std_row.max_opset),
                (OPSET_STD_LLM, OPSET_STD_LLM),
                "{op}: opset 27 is both the version it was introduced at and the newest that \
                 exists, so the window is closed at exactly one value"
            );
            assert!(
                std_row.schema.is_none(),
                "{op}: a standard-domain row is versioned by opset, not fingerprinted"
            );
        }
    }
}
