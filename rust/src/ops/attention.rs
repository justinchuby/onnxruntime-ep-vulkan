//! Fused attention — `GroupQueryAttention` and the rotary path.
//!
//! # The single most important op in the plan
//!
//! `OP_COVERAGE.md` §4.10 says it plainly: `GroupQueryAttention` is what a GenAI-built Qwen3 graph
//! puts at the centre of every decoder layer, and it arrives as *one node* carrying Q/K/V, the
//! past KV cache, the sequence lengths and the present KV cache out. It is not a fusion we
//! perform — it is a fusion the exporter already performed, and decomposing it back into
//! primitives would materialize a `[B, H, S, S]` score matrix in device memory, which is exactly
//! the thing flash-attention exists to avoid.
//!
//! So there is no path to "Qwen3 runs on Vulkan" that does not go through this file, and no
//! template leverage available in it (§11 risk 1). Justin's 2026-07-28 ruling makes it a committed
//! deliverable; what that changes is the schedule, not the difficulty.
//!
//! # KV cache is the reason M2 gates this
//!
//! GQA reads `past_key`/`past_value` and writes `present_key`/`present_value` every token. Under
//! M0/M1 host-visible I/O, that is a full round-trip of the entire cache per token, which makes a
//! technically-working decoder practically useless. The kernel can land before M2's device
//! allocator; the *claim* of "Qwen3 runs end-to-end" cannot.
//!
//! # Licensing note (OQ-M6, 🟢 GREEN)
//!
//! Rai authorized reading llama.cpp's MIT-licensed Vulkan attention shaders as a reference for the
//! tiling and subgroup strategy. The operative test is *"could you write this code without looking
//! at the original?"* — if yes, it is ours and nothing attaches. **Per-kernel record: the intent
//! for `GroupQueryAttention` is independent implementation after algorithm study, i.e. no
//! obligation.** If that changes and shader source is substantially adapted, the MIT header, the
//! `THIRD_PARTY_NOTICES.md` entry and the commit note all become mandatory, because our build
//! embeds the compiled SPIR-V into the cdylib and SPIR-V compiled from adapted GLSL is a derived
//! work.

use crate::kernel;
use crate::ops::common::claim::{self, ClaimResult};
use crate::ops::common::dtype::FLOAT;
use crate::ops::common::templates;
use crate::registry::OpStatus::Staged;
use crate::registry::{ContribSchema, PINNED_BASELINE, NodeView, OPSET_ANY, OpSpec, XL_KERNEL};
use crate::require;

/// `com.microsoft.GroupQueryAttention`.
///
/// **Fingerprint confidence: high, re-verified 2026-07-28.** Read from `docs/ContribOperators.md`
/// at tag `v1.28.0` and cross-checked against `onnxruntime/core/graph/contrib_ops/bert_defs.cc`
/// on main; the two are byte-identical for this operator, so the "1.28 and main differ" worry the
/// earlier note carried does not apply. The bounds below are the schema's own: **7–16 inputs,
/// 1–4 outputs**. (For contrast, v1.21 was 7–9 inputs and exactly 3 outputs — this schema moves,
/// which is the whole reason `[contrib-schema]` exists as a decline bucket.)
///
/// A wrong entry here costs a `[contrib-schema]` decline and a CPU fallback, never a wrong
/// answer — which is the direction this has to fail in, and the `[contrib-schema]` histogram
/// bucket is how we find out.
pub static GROUP_QUERY_ATTENTION: ContribSchema = ContribSchema {
    baseline: PINNED_BASELINE,
    notes: "ContribOperators.md @ v1.28.0, cross-checked against bert_defs.cc on main; identical",
    min_inputs: 7,
    max_inputs: 16,
    min_outputs: 1,
    max_outputs: 4,
    required_attrs: &["num_heads", "kv_num_heads"],
    known_attrs: &[
        "num_heads",
        "kv_num_heads",
        "scale",
        "do_rotary",
        "rotary_interleaved",
        "local_window_size",
        "softcap",
        "smooth_softmax",
        "qk_output",
        "qk_norm_epsilon",
        "k_quant_type",
        "v_quant_type",
        "kv_cache_bit_width",
    ],
};

/// `com.microsoft.MultiHeadAttention` — the non-GQA fused form ViT/BERT exports use.
pub static MULTI_HEAD_ATTENTION: ContribSchema = ContribSchema {
    baseline: PINNED_BASELINE,
    notes: "read from ContribOperators.md",
    min_inputs: 1,
    max_inputs: 8,
    min_outputs: 1,
    max_outputs: 3,
    required_attrs: &["num_heads"],
    known_attrs: &["num_heads", "scale", "mask_filter_value", "unidirectional"],
};

/// `com.microsoft.RotaryEmbedding`.
pub static ROTARY_EMBEDDING: ContribSchema = ContribSchema {
    baseline: PINNED_BASELINE,
    notes: "read from ContribOperators.md",
    min_inputs: 3,
    max_inputs: 4,
    min_outputs: 1,
    max_outputs: 1,
    required_attrs: &[],
    known_attrs: &[
        "interleaved",
        "is_packed_batching",
        "num_heads",
        "rotary_embedding_dim",
        "scale",
    ],
};

/// Is this head configuration a grouped-query one this EP can dispatch?
///
/// Pure and unit-tested: the GQA kernel assigns one KV head to `num_heads / kv_num_heads` query
/// heads, so a non-integer ratio is not a slow case, it is a different kernel.
pub const fn head_grouping_is_supported(num_heads: i64, kv_num_heads: i64) -> bool {
    num_heads > 0 && kv_num_heads > 0 && num_heads % kv_num_heads == 0
}

/// `GroupQueryAttention` — claim only the plain, unquantized-cache, non-sliding-window form first.
fn group_query_attention(view: &NodeView<'_>, spec: &OpSpec) -> ClaimResult {
    claim::typed_input(view, spec, 0, "query")?;

    let num_heads = claim::required_int(view, spec, "num_heads")?;
    let kv_num_heads = claim::required_int(view, spec, "kv_num_heads")?;
    require!(
        head_grouping_is_supported(num_heads, kv_num_heads),
        Attribute,
        "`{}` has num_heads = {num_heads} and kv_num_heads = {kv_num_heads}; the grouped-query \
         kernel needs num_heads to be a positive multiple of kv_num_heads",
        spec.op_type
    );

    // Each of these is a real numeric behaviour. Ignoring any one of them produces plausible
    // logits that are quietly wrong, which is far worse than a CPU fallback — §7's whole argument.
    claim::attr_int_is(view, spec, "local_window_size", -1)?;
    claim::attr_float_is(view, spec, "softcap", 0.0)?;
    claim::attr_int_is(view, spec, "smooth_softmax", 0)?;
    claim::attr_int_is(view, spec, "qk_output", 0)?;
    claim::attr_int_is(view, spec, "kv_cache_bit_width", 0)?;
    claim::attr_string_in(view, spec, "k_quant_type", &["NONE"], "NONE")?;
    claim::attr_string_in(view, spec, "v_quant_type", &["NONE"], "NONE")?;

    // `do_rotary` folds RoPE into the attention kernel. Both settings are implementable, but the
    // first kernel does not, and claiming what the kernel does not do is the one prohibited move.
    claim::attr_int_is(view, spec, "do_rotary", 0)?;

    // Optional inputs that change the numerics. Each must be *absent*, and each is a real trap:
    //
    // * 10 `attention_bias` — added to QK before the softmax. Ignoring it is a silent wrong answer.
    // * 11 `head_sink` — an extra term in the softmax denominator (the `smooth_softmax` path).
    // * 12/13 `k_scale`/`v_scale` — present exactly when the KV cache is quantized.
    // * 14/15 `q_norm_weight`/`k_norm_weight` — per-head RMS norm on Q and K, fused into the
    //   kernel. **This is not a hypothetical for us**: the ORT GenAI model builder sets
    //   `q_norm`/`k_norm` for every Qwen3-family decoder, and emits a 16-input GQA node whenever
    //   its fused-QK-norm path is enabled. ORT's own schema documentation is explicit that an EP
    //   which does not implement it "must reject the node when this input is set", so declining
    //   here is conformance, not conservatism.
    //
    // The consequence worth stating plainly: on the exact model this project exists to run, the
    // GQA node may arrive in a form the first kernel must refuse. Whether it does depends on
    // whether the builder's fused path is taken for our EP — see `OP_COVERAGE.md` §4.10.
    for (i, what) in [
        (10, "attention_bias"),
        (11, "head_sink"),
        (12, "k_scale"),
        (13, "v_scale"),
        (14, "q_norm_weight"),
        (15, "k_norm_weight"),
    ] {
        claim::input_absent(view, spec, i, what)?;
    }

    Ok(())
}

/// `RotaryEmbedding` — both interleaved and half-rotated layouts are in scope; packed batching is
/// not.
fn rotary_embedding(view: &NodeView<'_>, spec: &OpSpec) -> ClaimResult {
    claim::typed_input(view, spec, 0, "input")?;
    claim::attr_int_is(view, spec, "is_packed_batching", 0)?;
    let interleaved = view.attr_int("interleaved").unwrap_or(0);
    require!(
        matches!(interleaved, 0 | 1),
        Attribute,
        "`{}` has interleaved = {interleaved}; the attribute is a boolean",
        spec.op_type
    );
    Ok(())
}

crate::op_table! {
    //  op                     domain  opsets            caps    kernel          claim                   translate                  status              schema
    "GroupQueryAttention",     Ms,     1 ..= OPSET_ANY,  FLOAT,  kernel!(None),  group_query_attention,  templates::unimplemented,  Staged(XL_KERNEL),  schema: &GROUP_QUERY_ATTENTION;
    "RotaryEmbedding",         Ms,     1 ..= OPSET_ANY,  FLOAT,  kernel!(None),  rotary_embedding,       templates::unimplemented,  Staged(XL_KERNEL),  schema: &ROTARY_EMBEDDING;
    "MultiHeadAttention",      Ms,     1 ..= OPSET_ANY,  FLOAT,  kernel!(None),  claim::never,           templates::unimplemented,  Staged(XL_KERNEL),  schema: &MULTI_HEAD_ATTENTION;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{Domain, OpStatus};

    #[test]
    fn grouped_head_ratios() {
        assert!(head_grouping_is_supported(32, 8), "Qwen3-style 4:1 grouping");
        assert!(head_grouping_is_supported(16, 16), "MHA is GQA with ratio 1");
        assert!(!head_grouping_is_supported(30, 8), "not a whole ratio");
        assert!(!head_grouping_is_supported(8, 0));
        assert!(!head_grouping_is_supported(0, 8));
        assert!(!head_grouping_is_supported(-8, 8));
    }

    #[test]
    fn the_gqa_schema_knows_what_genai_writes() {
        for attr in [
            "num_heads",
            "kv_num_heads",
            "scale",
            "do_rotary",
            "rotary_interleaved",
            "local_window_size",
            "softcap",
        ] {
            assert!(GROUP_QUERY_ATTENTION.knows(attr), "`{attr}` is emitted");
        }
    }

    #[test]
    fn the_gqa_arity_is_the_schema_arity() {
        // The answer Trinity's test table needs, pinned as an assertion so it cannot rot
        // silently. `com.microsoft::GroupQueryAttention` @ ORT 1.28 and main:
        //
        //   inputs  (7 – 16): 0 query*, 1 key, 2 value, 3 past_key, 4 past_value, 5 seqlens_k*,
        //                     6 total_sequence_length*, 7 cos_cache, 8 sin_cache, 9 position_ids,
        //                     10 attention_bias, 11 head_sink, 12 k_scale, 13 v_scale,
        //                     14 q_norm_weight, 15 k_norm_weight        (* = required)
        //   outputs (1 –  4): 0 output*, 1 present_key, 2 present_value, 3 output_qk
        //
        // Only three inputs are required, but the optional ones are positional, so a node using
        // input 15 has 16 input slots with empty names in the unused positions — which is why the
        // minimum is 7 rather than 3: `seqlens_k` and `total_sequence_length` sit at 5 and 6.
        assert_eq!(GROUP_QUERY_ATTENTION.min_inputs, 7);
        assert_eq!(GROUP_QUERY_ATTENTION.max_inputs, 16);
        assert_eq!(GROUP_QUERY_ATTENTION.min_outputs, 1);
        assert_eq!(GROUP_QUERY_ATTENTION.max_outputs, 4);
    }

    #[test]
    fn every_row_here_is_contrib_and_fingerprinted() {
        for s in OPS {
            assert_eq!(s.domain, Domain::Ms, "{} is a contrib op", s.op_type);
            assert!(s.schema.is_some(), "{} needs a fingerprint", s.op_type);
            assert!(matches!(s.status, OpStatus::Staged(_)));
        }
    }

    #[test]
    fn gqa_declares_its_required_attributes() {
        // If these were not required, a node missing `kv_num_heads` would reach the predicate and
        // decline with `[attribute]` instead of `[contrib-schema]` — the wrong bucket, because the
        // cause would be a schema change rather than a value we chose not to support.
        assert!(GROUP_QUERY_ATTENTION.required_attrs.contains(&"num_heads"));
        assert!(
            GROUP_QUERY_ATTENTION
                .required_attrs
                .contains(&"kv_num_heads")
        );
    }
}
