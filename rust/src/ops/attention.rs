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

//! # Two producers, two attention ops (2026-07-29, re-derived against `onnxruntime/mobius`)
//!
//! The paragraph above is true of an **ORT-GenAI-built** graph. The `onnxruntime/mobius` builder
//! (@ `87fd878`, opset 24) emits **`ai.onnx::Attention`** instead — `op.Attention(query, key,
//! value, attn_mask, past_key, past_value, ...)` with `q_num_heads`/`kv_num_heads`/`scale`/
//! `softcap`/`is_causal` attributes and `_outputs=3`, no `_domain=` kwarg. Same model, same
//! architecture, a different node at the centre of every decoder layer.
//!
//! Two things the re-derivation changed versus the earlier reading of `justinchuby/onnx-genai-models`:
//!
//! 1. **Opset 24, not 23**, and opset 24 added a 7th input to `Attention` — `nonpad_kv_seqlen`
//!    (index 6, `int64`, `(batch,)`). mobius passes it on its *static* KV-cache path, paired with
//!    `ai.onnx::TensorScatter`. It changes the causal offset per batch element, so a kernel that
//!    ignores it computes a plausible wrong answer. It is declined explicitly below.
//! 2. **`com.microsoft::GroupQueryAttention` is still reachable from this producer.** mobius has a
//!    `RotaryAttentionToGQA` rewrite and a direct `_forward_gqa()` fast path, both gated on the
//!    target EP advertising GQA support. So the contrib row is not dead code for mobius graphs —
//!    which spelling arrives depends on how the EP is described to the builder, not on the model.
//!
//! That is a coverage fact, not a trivia item: a registry holding only the contrib spelling
//! declines every attention node in a mobius-built Qwen3 and the model runs entirely on CPU. Both
//! rows are therefore present. They are also genuinely one kernel — `Attention` with
//! `q_num_heads != kv_num_heads` *is* grouped-query attention, and the head-grouping predicate
//! below is shared verbatim. The standard-domain form is in fact the *easier* of the two: no
//! `seqlens_k` indirection, no in-place KV-cache aliasing requirement, and the rotary embedding
//! arrives as its own node rather than as a `do_rotary` attribute.
//!
//! **R1 settled for this producer:** mobius emits Q/K RMS norm as *separate* rank-4
//! `ai.onnx::RMSNormalization` nodes before the attention op and never passes norm weights into
//! the attention node, on any EP path. The 16-input fused-QK-norm GQA node is an ORT-GenAI
//! construct, not a mobius one. The inputs-14/15 decline below stays — it is still right for a
//! GenAI-built graph — but it is not on the mobius critical path.
//!
//! The scheduling consequence is in `OP_COVERAGE.md` §4.16: `ai.onnx::Attention` is the cheaper
//! first target and it unblocks a model family we can build ourselves, which makes it a better
//! T3 entry point than GQA even though GQA is the more famous op. Morpheus ratified that in
//! `DESIGN.md` §10.0.2.

use crate::kernel;
use crate::ops::common::claim::{self, ClaimResult};
use crate::ops::common::dtype::{ANY, FLOAT};
use crate::ops::common::templates;
use crate::registry::OpStatus::Staged;
use crate::registry::{
    ContribSchema, NO_SHADER, NodeView, OPSET_ANY, OPSET_STD_ATTENTION_MAX, OPSET_STD_LLM,
    OPSET_STD_NORM_MAX, OPSET_STD_TENSOR_SCATTER, OpSpec, PINNED_BASELINE, XL_KERNEL,
};
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
    notes: "ContribOperators.md @ v1.28.0, cross-checked against bert_defs.cc on main; identical. \
            12-input form with input 11 `head_sink` observed in gpt-oss-20b",
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
    //
    // **Census note (§4.21): `do_rotary = 1` on every GQA node in both Foundry Local graphs**, and
    // neither graph contains a separate `RotaryEmbedding` node at all. So this is not an exotic
    // option we can defer — it is the only form the ORT GenAI builder emits, and a GQA kernel
    // without fused rotary claims nothing. Recorded rather than relaxed: the decline is correct
    // until the kernel does it.
    claim::attr_int_is(view, spec, "do_rotary", 0)?;

    // **Packed QKV.** When inputs 1 and 2 are empty, input 0 is a single fused `[B, S, (Nq+2Nkv)*H]`
    // tensor rather than a query, and the kernel's addressing is completely different. This was a
    // permissive hole: nothing above reads inputs 1 or 2, so the node would have been claimed and
    // the kernel would have read a query tensor that is three tensors wide.
    //
    // Both Foundry Local graphs use the packed form on every layer (§4.21), so this is the common
    // case, not the corner. Declining it is the honest answer until the kernel splits the packing.
    claim::typed_input(view, spec, 1, "key (packed QKV is a different kernel)")?;
    claim::typed_input(view, spec, 2, "value (packed QKV is a different kernel)")?;

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

/// `ai.onnx::Attention` (opset 23 and 24) — the standard-domain spelling of grouped-query attention.
///
/// Inputs at 23: `Q`, `K`, `V`, `attn_mask`, `past_key`, `past_value`, the last three optional.
/// Opset 24 appends optional input 6 `nonpad_kv_seqlen`. Grouping comes from the
/// `q_num_heads`/`kv_num_heads` attribute pair rather than from the tensor shapes, exactly as in
/// the contrib form, so the head-grouping rule is shared.
///
/// Deliberately *not* sharing a claim predicate with [`group_query_attention`], despite the shared
/// kernel: the attribute names differ (`num_heads` vs `q_num_heads`), the illegal-combination set
/// differs (`is_causal` and `qk_matmul_output_mode` have no contrib equivalent) and the optional
/// inputs sit at different indices. One predicate covering both would be a predicate that is wrong
/// about one of them, and a predicate that is wrong in the permissive direction is the one failure
/// this design does not tolerate.
fn std_attention(view: &NodeView<'_>, spec: &OpSpec) -> ClaimResult {
    claim::typed_input(view, spec, 0, "Q")?;

    // Both are optional in the schema, defaulting to "infer from the shapes". When they are given
    // — which is what a builder emitting grouped-query attention does — the ratio must be one the
    // kernel can dispatch.
    if let (Some(q), Some(kv)) = (view.attr_int("q_num_heads"), view.attr_int("kv_num_heads")) {
        require!(
            head_grouping_is_supported(q, kv),
            Attribute,
            "`{}` has q_num_heads = {q} and kv_num_heads = {kv}; the grouped-query kernel needs \
             q_num_heads to be a positive multiple of kv_num_heads",
            spec.op_type
        );
    }

    // Same reasoning as the contrib row: each of these changes the numerics, and a plausible wrong
    // answer is worse than a CPU fallback.
    claim::attr_float_is(view, spec, "softcap", 0.0)?;
    claim::attr_int_is(view, spec, "qk_matmul_output_mode", 0)?;
    claim::attr_int_is(view, spec, "softmax_precision", 0)?;

    // Input 3 is `attn_mask`. The first kernel implements causal masking from the `is_causal`
    // attribute only; an explicit mask tensor is a separate binding and a separate code path.
    claim::input_absent(view, spec, 3, "attn_mask")?;

    // Input 6 is `nonpad_kv_seqlen`, **new at opset 24**. It selects the external KV-cache
    // pattern: K/V hold the full max-length cache (written by `TensorScatter`) and this tensor
    // gives the valid length per batch element. With `is_causal = 1` it moves the causal offset to
    // `nonpad_kv_seqlen[b] - q_sequence_length` per batch element, so a kernel that ignores it
    // attends to padding and produces plausible wrong logits.
    //
    // This is the concrete instance of why an open-ended opset window is a correctness bug rather
    // than a tidiness one: the row said `23 ..= OPSET_ANY`, and mobius defaults to opset 24 and
    // takes this path for static caches. `OP_COVERAGE.md` §4.19.
    //
    // The semantics are *not* in doubt. ONNX corrected the opset-24 reference implementation in
    // place — no opset bump, by user ruling 2026-07-29 — so `Attention`-24 means the corrected
    // bottom-right alignment and there is nothing here to gate on an onnx version. We decline
    // because the kernel does not implement it yet, which is the only reason this file ever
    // declines anything.
    claim::input_absent(view, spec, 6, "nonpad_kv_seqlen")?;

    Ok(())
}

/// `ai.onnx::RotaryEmbedding` (opset 23) — the standard-domain rotary op.
///
/// Same maths as the contrib row, different optional-input shape: position ids are input 3 here
/// rather than input 1, and there is no packed-batching mode to refuse.
fn std_rotary_embedding(view: &NodeView<'_>, spec: &OpSpec) -> ClaimResult {
    claim::typed_input(view, spec, 0, "X")?;
    let interleaved = view.attr_int("interleaved").unwrap_or(0);
    require!(
        matches!(interleaved, 0 | 1),
        Attribute,
        "`{}` has interleaved = {interleaved}; the attribute is a boolean",
        spec.op_type
    );
    Ok(())
}

/// `ai.onnx::TensorScatter` (opset 24) — the functional model of an in-place KV-cache write.
///
/// New at opset 24 and emitted by `onnxruntime/mobius` on its static-KV-cache path, immediately
/// upstream of the `Attention` node that consumes `nonpad_kv_seqlen`. Inputs: `past_cache`,
/// `update`, optional `write_indices` `(batch,)`. Attributes: `axis` (default -2, must not be 0)
/// and `mode` (`"linear"` | `"circular"`).
///
/// Why it is in this file rather than a shape module: it exists only to express KV-cache writes,
/// and its whole value to us is that the spec explicitly permits the backend to alias
/// `present_cache` onto `past_cache`. That is precisely the `bind_aliased_output` seam Switch
/// built for GQA, so it belongs next to the ops that need it.
///
/// `"circular"` is declined: the wrap-around write is a different index computation, not a slow
/// case of the linear one.
fn tensor_scatter(view: &NodeView<'_>, spec: &OpSpec) -> ClaimResult {
    claim::typed_input(view, spec, 0, "past_cache")?;
    claim::attr_string_in(view, spec, "mode", &["linear"], "linear")?;
    let axis = view.attr_int("axis").unwrap_or(-2);
    require!(
        axis != 0,
        Attribute,
        "`{}` has axis = 0; the schema forbids scattering along the batch dimension",
        spec.op_type
    );
    Ok(())
}

crate::op_table! {
    //  op                     domain  opsets                       caps    kernel          claim                   translate                  status              schema
    "GroupQueryAttention",     Ms,     1 ..= OPSET_ANY,             FLOAT,  kernel!(None),  group_query_attention,  templates::unimplemented,  Staged(XL_KERNEL),  schema: &GROUP_QUERY_ATTENTION;
    "RotaryEmbedding",         Ms,     1 ..= OPSET_ANY,             FLOAT,  kernel!(None),  rotary_embedding,       templates::unimplemented,  Staged(XL_KERNEL),  schema: &ROTARY_EMBEDDING;
    "MultiHeadAttention",      Ms,     1 ..= OPSET_ANY,             FLOAT,  kernel!(None),  claim::never,           templates::unimplemented,  Staged(XL_KERNEL),  schema: &MULTI_HEAD_ATTENTION;
    "Attention",               Ai,     OPSET_STD_LLM ..= OPSET_STD_ATTENTION_MAX, FLOAT,  kernel!(None),  std_attention,          templates::unimplemented,  Staged(XL_KERNEL);
    "RotaryEmbedding",         Ai,     OPSET_STD_LLM ..= OPSET_STD_NORM_MAX,      FLOAT,  kernel!(None),  std_rotary_embedding,   templates::unimplemented,  Staged(XL_KERNEL);
    "TensorScatter",           Ai,     OPSET_STD_TENSOR_SCATTER ..= OPSET_STD_TENSOR_SCATTER, ANY, kernel!(None), tensor_scatter, templates::unimplemented, Staged(NO_SHADER);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{Domain, OpStatus};

    #[test]
    fn grouped_head_ratios() {
        assert!(
            head_grouping_is_supported(32, 8),
            "Qwen3-style 4:1 grouping"
        );
        assert!(
            head_grouping_is_supported(16, 16),
            "MHA is GQA with ratio 1"
        );
        assert!(!head_grouping_is_supported(30, 8), "not a whole ratio");
        assert!(!head_grouping_is_supported(8, 0));
        assert!(!head_grouping_is_supported(0, 8));
        assert!(!head_grouping_is_supported(-8, 8));
    }

    /// The GQA fingerprint must admit the shapes real graphs emit, so the decline is attributed to
    /// the predicate rather than to phantom schema drift.
    ///
    /// Census (§4.21): Phi-3.5-mini-int4 emits 9 declared inputs / 3 outputs; gpt-oss-20b emits 12
    /// declared inputs (slot 11 = `head_sink`, the attention-sink term) / 3 outputs and carries
    /// `local_window_size` on every node — 128 on half the layers, -1 on the other half, i.e.
    /// alternating sliding-window attention.
    #[test]
    fn the_gqa_fingerprint_admits_both_real_arities() {
        for declared in [9usize, 12] {
            assert!(
                declared >= GROUP_QUERY_ATTENTION.min_inputs
                    && declared <= GROUP_QUERY_ATTENTION.max_inputs,
                "{declared}-input GQA is emitted by a real Foundry Local graph and the \
                 fingerprint rejects it"
            );
        }
        assert!(3 <= GROUP_QUERY_ATTENTION.max_outputs);
        for attr in [
            "do_rotary",
            "local_window_size",
            "softcap",
            "rotary_interleaved",
            "scale",
        ] {
            assert!(
                GROUP_QUERY_ATTENTION.knows(attr),
                "`{attr}` is on real nodes"
            );
        }
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
    fn every_contrib_row_here_is_fingerprinted() {
        // Standard-domain rows carry no fingerprint by design: `ai.onnx` versions by opset, which
        // the row's window already expresses. Only `com.microsoft` needs C2.
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
    fn both_producers_attention_ops_are_registered() {
        // The 2026-07-29 crate review found that ORT GenAI emits
        // `com.microsoft::GroupQueryAttention` while Justin's `onnx-genai-models` emits
        // `ai.onnx::Attention` @ opset 23 and never emits GQA at all. Registering one spelling
        // means every attention node of a model built by the other toolchain runs on CPU.
        let contrib = OPS
            .iter()
            .find(|s| s.op_type == "GroupQueryAttention")
            .expect("the ORT GenAI spelling");
        let standard = OPS
            .iter()
            .find(|s| s.op_type == "Attention" && s.domain == Domain::Ai)
            .expect("the standard-domain spelling");
        assert_eq!(contrib.domain, Domain::Ms);
        assert_eq!(standard.min_opset, OPSET_STD_LLM);
        // Deliberately *different* predicates over the same kernel — the attribute names and
        // optional-input indices differ, so one predicate would be wrong about one of them.
        assert!(!std::ptr::fn_addr_eq(contrib.claim, standard.claim));
    }

    #[test]
    fn rotary_embedding_is_registered_in_both_domains() {
        let rows: Vec<_> = OPS
            .iter()
            .filter(|s| s.op_type == "RotaryEmbedding")
            .collect();
        assert_eq!(rows.len(), 2, "one contrib spelling, one standard-domain");
        assert!(rows.iter().any(|s| s.domain == Domain::Ms));
        assert!(rows.iter().any(|s| s.domain == Domain::Ai));
        // And they must remain distinguishable by qualified name, or the registry lookup that
        // resolves a node to a row would be ambiguous.
        assert_ne!(rows[0].qualified_name(), rows[1].qualified_name());
    }

    #[test]
    fn nothing_here_is_live_yet() {
        for s in OPS {
            assert!(matches!(s.status, OpStatus::Staged(_)));
        }
    }

    /// Opset 24 added `Attention` input 6 `nonpad_kv_seqlen`, and `onnxruntime/mobius` defaults to
    /// opset 24. Before 2026-07-29 this row was `23 ..= OPSET_ANY`, so an opset-24 node — including
    /// the static-cache form that supplies input 6 — matched a predicate written against the
    /// opset-23 schema. That is the failure mode the whole design exists to prevent: not a decline,
    /// a *silent wrong answer* (the per-batch causal offset would be ignored).
    ///
    /// The window is now closed at the highest schema version anybody has read. A hypothetical
    /// `Attention`-25 declines as `[opset]` until someone reads it.
    #[test]
    fn attention_window_covers_23_and_24_and_stops_there() {
        let row = OPS
            .iter()
            .find(|s| s.op_type == "Attention" && s.domain == Domain::Ai)
            .expect("standard-domain Attention");
        assert_eq!(row.min_opset, 23);
        assert_eq!(row.max_opset, 24, "opset 24 is the newest schema read");
        assert_ne!(
            row.max_opset, OPSET_ANY,
            "an open-ended window over an op that gains inputs is a correctness bug, not untidiness"
        );
    }

    /// `RotaryEmbedding` has exactly one standard-domain schema version, unlike `Attention`.
    ///
    /// Verified against onnx v1.22.0: it is absent from the opset-24 section of `operator_sets.h`
    /// and still lives in `defs.cc` rather than `old.cc`, i.e. version 23 is current at opset 27.
    #[test]
    fn std_rotary_embedding_has_a_single_schema_version() {
        let row = OPS
            .iter()
            .find(|s| s.op_type == "RotaryEmbedding" && s.domain == Domain::Ai)
            .expect("standard-domain RotaryEmbedding");
        assert_eq!(row.min_opset, 23);
        assert_eq!(row.max_opset, 23);
    }

    /// `TensorScatter` is new at opset 24 and is the other half of the static-KV-cache pattern.
    #[test]
    fn tensor_scatter_is_registered_at_exactly_24() {
        let row = OPS
            .iter()
            .find(|s| s.op_type == "TensorScatter")
            .expect("the opset-24 KV-cache write");
        assert_eq!(row.domain, Domain::Ai);
        assert_eq!((row.min_opset, row.max_opset), (24, 24));
        assert!(
            row.schema.is_none(),
            "standard domain carries no fingerprint"
        );
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
