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

use crate::engine::{
    AttrValue, DType, DispatchContext, EpError, EpResult, KernelRequest, NodeDesc, TensorDesc,
};
use crate::kernel;
use crate::ops::common::claim::{self, ClaimResult};
use crate::ops::common::dtype::{ANY, F16, FLOAT};
use crate::ops::common::templates;
use crate::registry::OpStatus::{Live, Staged};
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

/// `GroupQueryAttention` — packed QKV + fused neox RoPE (do_rotary=1, rotary_interleaved=0).
///
/// This predicate was updated on 2026-07-30 based on island attribution results:
/// GQA was the sole island-splitter (32 nodes, 32 ORT topological breaks → 33 islands).
/// The kernel (`gqa_f16.comp`) now implements packed QKV + fused RoPE, so the previously
/// protective declines on those two properties are replaced with positive requirements.
///
/// **Falsifier for the predicate change**: after this lands, the CLAIM_LOG should show zero
/// GQA declines with code [attribute].  Any remaining [attribute] decline means the predicate
/// still disagrees with what Phi-3.5 actually sends.
fn group_query_attention(view: &NodeView<'_>, spec: &OpSpec) -> ClaimResult {
    // **Packed QKV — the only input form this kernel implements.**
    // Input 0 carries the fused [B, S, (Nq+2*Nkv)*D] tensor.  Inputs 1 and 2 (key, value)
    // should be absent or empty-name in the packed-QKV form.  We do NOT gate on their
    // presence here: ORT's handling of empty optional inputs varies across platforms (null
    // pointer vs non-null pointer with no type info), and an over-eager decline here would
    // silently stall an entire 32-layer model.  The translate handler reads only input 0's
    // shape, so a separate-Q/K/V node would produce a wrong head_dim and be caught at the
    // correctness gate (MATCH/FAIL verdict), not silently accepted.
    claim::typed_input(view, spec, 0, "packed_qkv")?;

    // KV cache and sequence lengths are required for the append-then-attend kernel.
    claim::typed_input(view, spec, 3, "past_key")?;
    claim::typed_input(view, spec, 4, "past_value")?;
    // seqlens_k is INT32 — dtype check via typed_input would fail against FLOAT caps.
    // Presence is guaranteed by the schema (it is a required input at slot 5), and the
    // translate handler resolves it by index without inspecting dtype.
    require!(
        view.has_input(5),
        Attribute,
        "`{}` requires seqlens_k (input 5) for per-batch past length",
        spec.op_type
    );

    let num_heads = claim::required_int(view, spec, "num_heads")?;
    let kv_num_heads = claim::required_int(view, spec, "kv_num_heads")?;
    require!(
        head_grouping_is_supported(num_heads, kv_num_heads),
        Attribute,
        "`{}` has num_heads = {num_heads} and kv_num_heads = {kv_num_heads}; the grouped-query \
         kernel needs num_heads to be a positive multiple of kv_num_heads",
        spec.op_type
    );

    // Numeric behaviours that produce silent wrong answers if ignored.
    claim::attr_int_is(view, spec, "local_window_size", -1)?;
    claim::attr_float_is(view, spec, "softcap", 0.0)?;
    // smooth_softmax: 0 means disabled; -1 is Phi-3.5's "not set" sentinel (also disabled).
    // Only smooth_softmax=1 enables the attention-sink behaviour that changes the numerics.
    {
        let smooth = view.attr_int("smooth_softmax").unwrap_or(0);
        require!(
            smooth <= 0,
            Attribute,
            "`{}` has smooth_softmax = {smooth}; this EP implements only smooth_softmax ≤ 0 \
             (disabled) so far",
            spec.op_type
        );
    }
    claim::attr_int_is(view, spec, "qk_output", 0)?;
    claim::attr_int_is(view, spec, "kv_cache_bit_width", 0)?;
    claim::attr_string_in(view, spec, "k_quant_type", &["NONE"], "NONE")?;
    claim::attr_string_in(view, spec, "v_quant_type", &["NONE"], "NONE")?;

    // **do_rotary=1 with rotary_interleaved=0 is now the claimed form.**
    //
    // The `gqa_f16.comp` shader fuses neox-style half-rotation RoPE into the attention pass.
    // Phi-3.5 CLAIM_LOG (2026-07-30): every GQA node has do_rotary=1, rotary_interleaved=0.
    // do_rotary=0 requires a different shader path (neither GQA graph we currently target emits
    // it, but we decline explicitly rather than silently ignoring the rotary application).
    // rotary_interleaved=1 (alternating pairs) requires a different index mapping in the shader.
    {
        let do_rotary = view.attr_int("do_rotary").unwrap_or(0);
        require!(
            do_rotary == 1,
            Attribute,
            "`{}` has do_rotary = {do_rotary}; this kernel fuses RoPE (do_rotary=1 only) — \
             the no-rotary variant needs a separate shader path",
            spec.op_type
        );
        claim::attr_int_is(view, spec, "rotary_interleaved", 0)?;
    }
    // cos_cache (input 7) and sin_cache (input 8) must be present when do_rotary=1.
    claim::typed_input(view, spec, 7, "cos_cache")?;
    claim::typed_input(view, spec, 8, "sin_cache")?;

    // Optional inputs that change the numerics — must be absent.
    //
    // * 10 `attention_bias` — added to QK before the softmax. Ignoring it is a silent wrong answer.
    // * 11 `head_sink` — an extra term in the softmax denominator (the `smooth_softmax` path).
    // * 12/13 `k_scale`/`v_scale` — present exactly when the KV cache is quantized.
    // * 14/15 `q_norm_weight`/`k_norm_weight` — per-head RMS norm on Q and K, fused into the
    //   kernel. The ORT GenAI builder sets these for every Qwen3-family decoder, and the ORT
    //   schema is explicit that an EP which does not implement it "must reject the node when this
    //   input is set", so declining here is conformance, not conservatism.
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

/// Translate `com.microsoft.GroupQueryAttention` (packed QKV + fused neox RoPE).
///
/// Dispatches `gqa_f16.comp` — one invocation per (batch, query_head, query_seq_pos).
/// Each invocation runs the full attention computation serially over head_dim and past tokens.
///
/// Push-constant layout (must stay byte-for-byte synchronised with the shader PC struct):
///
///   [0..4]   batch_size:   u32
///   [4..8]   seq_len:      u32
///   [8..12]  num_heads:    u32
///   [12..16] kv_num_heads: u32
///   [16..20] head_dim:     u32
///   [20..24] rotary_dim:   u32  (full head_dim for Phi-3.5; shader uses rotary_dim/2 cos/sin entries)
///   [24..28] past_len_max: u32  (KV buffer S dimension — from past_key.shape[2])
///   [28..32] scale:        f32
///
/// Binding order (must match `gqa_f16.comp` layout qualifiers):
///   0 packed_qkv, 1 past_key, 2 past_value, 3 seqlens_k,
///   4 cos_cache,  5 sin_cache,
///   6 attn_output (write), 7 present_key (rw), 8 present_value (rw)
fn translate_gqa(_spec: &OpSpec, node: &NodeDesc, ctx: &mut dyn DispatchContext) -> EpResult<()> {
    // -- Attribute helpers ----------------------------------------------------------------
    let attr_i64 = |name: &str| -> EpResult<i64> {
        match node.attributes.get(name) {
            Some(AttrValue::Int(v)) => Ok(*v),
            _ => Err(EpError::Internal(format!(
                "`{}` was claimed but has no integer attribute `{name}`",
                node.op_type
            ))),
        }
    };
    let attr_f32_opt = |name: &str| -> Option<f32> {
        match node.attributes.get(name) {
            Some(AttrValue::Float(v)) => Some(*v),
            _ => None,
        }
    };

    // -- packed_qkv shape ----------------------------------------------------------------
    // Shape: [B, S, (Nq + 2*Nkv) * D].  Claim already verified input 0 is typed.
    let qkv_desc = node.inputs[0].desc.as_ref().ok_or_else(|| {
        EpError::Unsupported(format!("`{}` packed_qkv has no shape", node.op_type))
    })?;
    if qkv_desc.shape.len() != 3 {
        return Err(EpError::InvalidGraph(format!(
            "`{}` packed_qkv must be rank 3, got rank {}",
            node.op_type,
            qkv_desc.shape.len()
        )));
    }
    let dtype = qkv_desc.dtype;
    if dtype != DType::F16 {
        return Err(EpError::Unsupported(format!(
            "`{}` only supports f16 (got {dtype:?}); an f32 GQA shader is not implemented",
            node.op_type
        )));
    }

    let batch_size = qkv_desc.shape[0] as u32;
    let seq_len = qkv_desc.shape[1] as u32;
    let packed_dim = qkv_desc.shape[2] as u32;

    let num_heads = attr_i64("num_heads")? as u32;
    let kv_num_heads = attr_i64("kv_num_heads")? as u32;
    let head_dim = packed_dim / (num_heads + 2 * kv_num_heads);

    // `rotary_embedding_dim` is optional — but its absence does not mean "full width".
    // `cos_cache` is [max_seq, rotary_dim/2], and that second dimension is the only place
    // the true rotary width is recorded, which is why ORT derives it from the cache rather
    // than defaulting it. Defaulting to head_dim is correct only when the rotary is
    // full-width; on a partial-rotary graph it makes the shader stride cos/sin rows by
    // twice their real length, so every head reads another position's angles.
    //
    // MEASURED 2026-08-02: `group_query_attention_f16` has head_dim 32 and cos_cache
    // [64, 8] — true rotary_dim 16 against a defaulted 32 — and was DIVERGENT at
    // worst_rel 16.726. Phi-3.5 has cos_cache [max_seq, head_dim/2], where the old default
    // happened to be right, which is why 32 nodes were declined by a defect none of them
    // had.
    //
    // When neither the attribute nor the cache shape is available the width is simply not
    // recoverable, so refuse. Inferring it from head_dim is what produced the defect, and
    // a handler that guesses can no longer detect that the information was missing.
    let rotary_dim = match node.attributes.get("rotary_embedding_dim") {
        Some(AttrValue::Int(v)) if *v > 0 => *v as u32,
        _ => {
            let half = node.inputs[7]
                .desc
                .as_ref()
                .and_then(|d| d.shape.get(1).copied())
                .filter(|h| *h > 0)
                .ok_or_else(|| {
                    EpError::Unsupported(format!(
                        "`{}` has no positive `rotary_embedding_dim` attribute and cos_cache \
                         has no usable second dimension, so the rotary width is not \
                         recoverable; refusing rather than assuming head_dim",
                        node.op_type
                    ))
                })?;
            (half as u32) * 2
        }
    };
    if rotary_dim > head_dim || rotary_dim % 2 != 0 {
        return Err(EpError::InvalidGraph(format!(
            "`{}` derived rotary_dim = {rotary_dim}, which is not an even value at most \
             head_dim = {head_dim}",
            node.op_type
        )));
    }

    // `scale` is optional; default to 1/sqrt(head_dim).
    let scale = attr_f32_opt("scale").unwrap_or_else(|| (head_dim as f32).sqrt().recip());

    // past_len_max from past_key shape[2] (buffer S dimension).
    //
    // This was `.unwrap_or(0)`, and 0 is not a neutral default: it is the empty-past case,
    // which takes a different branch below and sizes `present` to `seq_len`. A missing desc
    // and a genuinely empty cache produced the same number and were then indistinguishable.
    // Refuse instead — the extent is not recoverable from anything else here.
    let past_len_max = node.inputs[3]
        .desc
        .as_ref()
        .and_then(|d| d.shape.get(2).copied())
        .ok_or_else(|| {
            EpError::Unsupported(format!(
                "`{}` has no shape for past_key, so the cache extent is not recoverable; \
                 refusing rather than defaulting to an empty past",
                node.op_type
            ))
        })? as u32;

    // -- Resolve input buffers ----------------------------------------------------------
    let qkv_buf = ctx.resolve(&node.inputs[0])?;
    let past_k_buf = ctx.resolve(&node.inputs[3])?;
    let past_v_buf = ctx.resolve(&node.inputs[4])?;
    let seqlens_buf = ctx.resolve(&node.inputs[5])?;
    let cos_buf = ctx.resolve(&node.inputs[7])?;
    let sin_buf = ctx.resolve(&node.inputs[8])?;

    // -- Bind outputs ------------------------------------------------------------------
    // attn_output: [B, S, Nq*D]
    let attn_out_ref = node.outputs.first().ok_or_else(|| {
        EpError::InvalidGraph(format!("`{}` has no attn_output slot", node.op_type))
    })?;
    let attn_buf = ctx.bind_output(
        attn_out_ref,
        TensorDesc::new(
            dtype,
            vec![
                batch_size as i64,
                seq_len as i64,
                (num_heads * head_dim) as i64,
            ],
        ),
    )?;

    let pres_k_ref = node.outputs.get(1).ok_or_else(|| {
        EpError::InvalidGraph(format!("`{}` has no present_key slot", node.op_type))
    })?;
    let pres_v_ref = node.outputs.get(2).ok_or_else(|| {
        EpError::InvalidGraph(format!("`{}` has no present_value slot", node.op_type))
    })?;

    // present_key / present_value length. There are two ORT GQA cache conventions and they
    // are distinguished only by the declared `present` shape:
    //
    //  * **growing cache** — `past` is [B, Nkv, past_len, D] and `present` is
    //    [B, Nkv, past_len + seq_len, D]. `present` is a *different, larger* buffer, so it
    //    cannot alias `past`, and it must be filled with the past tokens as well as the new
    //    ones because ORT's `present` output is the whole cache.
    //  * **shared buffer** — `past` and `present` are both [B, Nkv, max_seq, D], the same
    //    allocation, updated in place at `tok_pos`.
    //
    // This handler previously assumed the shared-buffer convention unconditionally
    // (`kv_len = past_len_max`), with an `empty_past` special case that was really the one
    // growing-cache instance where the two conventions happen to agree. MEASURED
    // 2026-08-02: both graphs we target are growing — the evidence case declares past
    // [B,2,4,32] against present [B,2,5,32], and Phi-3.5 declares `past_sequence_length`
    // against `total_sequence_length`. With present bound one token short, the shader's
    // write at `tok_pos == past_len` fell outside the buffer and was dropped, so
    // `present_key`/`present_value` read back all-zero.
    //
    // Read the declared present extent rather than deriving it: the two conventions differ
    // in exactly that number, so deriving it picks one and cannot tell it picked wrong.
    let declared_present_len = pres_k_ref
        .desc
        .as_ref()
        .and_then(|d| d.shape.get(2).copied())
        .filter(|s| *s > 0)
        .map(|s| s as u32);
    let present_len = declared_present_len.unwrap_or(past_len_max + seq_len);
    if present_len < past_len_max + seq_len && present_len != past_len_max {
        return Err(EpError::InvalidGraph(format!(
            "`{}` declares present length {present_len}, which is neither the shared-buffer \
             extent ({past_len_max}) nor large enough for the growing-cache extent \
             ({past_len_max} + {seq_len})",
            node.op_type
        )));
    }
    // -- The arena --------------------------------------------------------------------
    //
    // When the caller declares the shared-buffer convention (`DispatchContext::kv_arena`),
    // `past`'s extent is a **capacity**, not a length: the true past length is carried by
    // `seqlens_k`, and `present` is the same allocation. The whole present-copy disappears —
    // not moved, not fused, gone — because the past tokens are already where `present` wants
    // them.
    //
    // Two things make this sound, and neither is an assumption about the test set:
    //
    //  1. **The regions are disjoint.** Under the alias `present_len == past_stride`, so a
    //     read of past token `t` and a write of present token `tok_pos` land at the same base
    //     `(b*Nkv + kv_h) * stride * D`. The kernel reads only `t < past_len` and writes only
    //     `tok_pos = past_len + s_local >= past_len`. No invocation, of any dispatch, reads a
    //     byte another invocation writes. (`gqa_f16.comp` step 3 vs step 4; the sibling-key
    //     read that *did* alias was removed on 2026-08-02 and recomputes from `packed_qkv`.)
    //  2. **The past extent does not come from the shape.** `past_len` is
    //     `seqlens_k[b] + 1 - seq_len` in the kernel, so an oversized `past` changes the
    //     stride and nothing else. MEASURED against ORT's own CPU GQA on the real Phi-3.5
    //     export, arena tail poisoned to 0.5: **bit-identical logits**
    //     (`bench/results/kv_arena_graph_accepts.json`).
    //
    // The declared present extent, when the graph states one, outranks this flag: a graph
    // that declares `[B, Nkv, P+S, D]` has asked for the growing convention in writing, and
    // the evidence case `group_query_attention_f16` is exactly that graph. Only a symbolic
    // (unstated) present extent — which is what Phi-3.5 has — is the EP's to decide.
    let arena = ctx.kv_arena() && declared_present_len.is_none() && past_len_max >= seq_len;
    let present_len = if arena { past_len_max } else { present_len };
    let shares_past_buffer = past_len_max > 0 && present_len == past_len_max;
    // -- The prefix alias ---------------------------------------------------------------
    //
    // Under the growing convention `present[0..past_len_max]` is, per `(b, kv_h)` block,
    // exactly `past`'s bytes — the shader's own first act (`copy_leader`) is to put them
    // there. So `past` is a device allocation whose entire contents are copied into another
    // device allocation we also made, and then discarded; both are live at once, and their
    // sum is what the peak cannot afford. MEASURED 2026-08-04
    // (`bench/results/ctx4096_BEFORE.json`): the shipping lane's device-local peak on Phi-3.5
    // is the resident weight cache (2,290,839,552 B) plus **2 x** 393,216 B per past token,
    // and it stops executing between `past_len` 2048 and 3072 — exit 0, zero EP dispatches.
    //
    // Two things make this the arena's own soundness argument rather than a new one:
    //
    //  1. **`past_stride` becomes `present_len`,** so the shader reads past token `t` at
    //     `(b*Nkv + kv_h) * present_len * D + t*D` — where the staged copy put it — and
    //     `copy_leader` (`present_len != past_stride`) goes false.
    //  2. **The read and write sets stay disjoint,** by the identical argument: reads are
    //     `t < past_len`, the write is `tok_pos = past_len + s_local >= past_len`, at a
    //     common base. That is `gqa_f16.comp` step 3 vs step 4, unchanged. No shader changes.
    //
    // Declined — falling back to the two-buffer shipping path, which is correct for any
    // allocation — whenever the relation is not provable here: the shared-buffer convention
    // (already an alias), a non-growing declared extent, or an empty past (prefill, where
    // there is nothing to stage and `past` is a zero-element placeholder).
    let prefix_alias = !shares_past_buffer
        && ctx.kv_growing_alias()
        && past_len_max > 0
        && present_len > past_len_max
        && batch_size > 0
        && kv_num_heads > 0
        && head_dim > 0;
    let kv_desc = TensorDesc::new(
        dtype,
        vec![
            batch_size as i64,
            kv_num_heads as i64,
            present_len as i64,
            head_dim as i64,
        ],
    );
    let (pres_k_buf, pres_v_buf) = if shares_past_buffer {
        // Shared fixed-size cache: present aliases past, shader updates `tok_pos` in place.
        // M2's device-backed allocator must honour the alias (see OP_COVERAGE.md §9.5 #3).
        (
            ctx.bind_aliased_output(&node.inputs[3], pres_k_ref, kv_desc.clone())?,
            ctx.bind_aliased_output(&node.inputs[4], pres_v_ref, kv_desc)?,
        )
    } else if prefix_alias {
        // Growing cache, PREFIX alias: one buffer, not two. `bind_prefix_output` hands back
        // the *present* buffer for the `past_*` slots as well; the engine stages `past`'s host
        // bytes straight into `present`'s prefix at the strides declared here, and never
        // allocates a `past` buffer at all.
        let elem = dtype.byte_size() as u64;
        let layout = crate::engine::PrefixLayout {
            outer_blocks: batch_size as u64 * kv_num_heads as u64,
            src_block_bytes: past_len_max as u64 * head_dim as u64 * elem,
            dst_block_bytes: present_len as u64 * head_dim as u64 * elem,
        };
        (
            ctx.bind_prefix_output(past_k_buf, pres_k_ref, kv_desc.clone(), layout)?,
            ctx.bind_prefix_output(past_v_buf, pres_v_ref, kv_desc, layout)?,
        )
    } else {
        // Growing cache: bind fresh present buffers. The shader copies the past tokens in
        // and appends the new ones, so the output is the whole cache ORT expects.
        (
            ctx.bind_output(pres_k_ref, kv_desc.clone())?,
            ctx.bind_output(pres_v_ref, kv_desc)?,
        )
    };
    // Under the prefix alias the shader reads `past` out of the `present` buffer, so the two
    // slots bind the same view and the past stride is the present stride. That second equality
    // is also the condition that switches the shader's own past-copy off (`copy_leader`), which
    // is correct: the copy it would have made has already happened, by `vkCmdCopyBuffer`.
    let (past_k_buf, past_v_buf) = if prefix_alias {
        (pres_k_buf, pres_v_buf)
    } else {
        (past_k_buf, past_v_buf)
    };
    let past_read_stride = if prefix_alias {
        present_len
    } else {
        past_len_max
    };
    // -- Push constants (36 bytes, matches shader PC struct) ---------------------------
    // Field 6 stays the KV *write* stride (the present buffer's S dimension); field 7 is
    // the past buffer's S dimension. They were one field while the two were assumed equal.
    let mut push = Vec::with_capacity(36);
    push.extend_from_slice(&batch_size.to_le_bytes());
    push.extend_from_slice(&seq_len.to_le_bytes());
    push.extend_from_slice(&num_heads.to_le_bytes());
    push.extend_from_slice(&kv_num_heads.to_le_bytes());
    push.extend_from_slice(&head_dim.to_le_bytes());
    push.extend_from_slice(&rotary_dim.to_le_bytes());
    push.extend_from_slice(&present_len.to_le_bytes());
    push.extend_from_slice(&past_read_stride.to_le_bytes());
    push.extend_from_slice(&scale.to_bits().to_le_bytes());

    // -- Dispatch ----------------------------------------------------------------------
    //
    // Two modules, one selector, and the same nine bindings and 36-byte push block for both —
    // so the choice below is a shader stem, a specialisation constant and a workgroup count,
    // and nothing about how the data is bound can diverge between the paths.
    //
    //  * **`gqa_f16` — one invocation per (batch, query head, query position).** The shipping
    //    kernel, byte-identical, and the path every prefill and every refusal takes. The
    //    invocation *count* is unchanged from before #56; how many of them share a workgroup is
    //    `gqa_local_size`'s call, and the shader's `b >= batch_size` guard retires the tail when
    //    the size does not divide `total`.
    //  * **`gqa_decode_f16` — one workgroup per (batch, query head), `lanes` lanes striding the
    //    KV cache.** Decode only. At `seq_len == 1` the whole dispatch is `B * Nq` invocations —
    //    32 on Phi-3.5 — and each one walks the entire cache alone, which is why `gqa_f16` holds
    //    93.5% of decode GPU time at `past = 1024` (docs/PERF.md §26.4). There is no packing left
    //    to do at 32 invocations; the parallelism has to come from the KV dimension. The grid is
    //    exact — exactly `B * Nq` workgroups — so the module's own `b >= batch_size` guard is a
    //    fail-closed backstop rather than a tail.
    //
    // `gqa_decode_kv_lanes` returns 1 to mean *refuse*, so `lanes > 1` is the only test and the
    // fallback is the untouched dispatch below it.
    let decode_lanes = gqa_decode_kv_lanes(GqaDecodeForm {
        seq_len: Some(seq_len as i64),
        head_dim: Some(head_dim as i64),
        past_extent: Some(past_len_max as i64),
    });
    let bindings = vec![
        qkv_buf,     // binding 0: packed_qkv
        past_k_buf,  // binding 1: past_key
        past_v_buf,  // binding 2: past_value
        seqlens_buf, // binding 3: seqlens_k
        cos_buf,     // binding 4: cos_cache
        sin_buf,     // binding 5: sin_cache
        attn_buf,    // binding 6: attn_output
        pres_k_buf,  // binding 7: present_key
        pres_v_buf,  // binding 8: present_value
    ];
    let (shader, spec_constants, workgroups) = if decode_lanes > 1 {
        (
            GQA_DECODE_MODULE,
            vec![decode_lanes],
            [(batch_size * num_heads).max(1), 1, 1],
        )
    } else {
        let total = (batch_size * num_heads * seq_len).max(1);
        let local = gqa_local_size(total);
        ("gqa_f16", vec![local], [total.div_ceil(local), 1, 1])
    };
    ctx.dispatch(KernelRequest {
        shader,
        spec_constants,
        push_constants: push,
        bindings,
        workgroups,
    })
}

/// Largest workgroup this EP will ask a Vulkan 1.1 implementation for.
///
/// `maxComputeWorkGroupInvocations` is **required** to be at least 128 by the Vulkan 1.1
/// specification, so 64 is inside the guaranteed floor with a factor of two to spare on every
/// conformant implementation, whatever it happens to report. A number derived from the local
/// device would make the pipeline — and therefore the arithmetic's scheduling — device-dependent
/// for no measured gain; this stays a property of the *specification*, like
/// `GEMV_MAX_GROUPS_Y` next door.
///
/// It is also where the measured curve stops paying: at `M = 128` the sweep reads 201.7 ms at 32
/// and 194.3 ms at 64 (3.7%), against 9.4x already banked between 1 and 32. There is no evidence
/// for going further and the specification floor forbids assuming it.
pub const GQA_MAX_LOCAL_SIZE: u32 = 64;

/// Overrides [`gqa_local_size`]. Clamped to `[1, GQA_MAX_LOCAL_SIZE]`, never trusted.
///
/// `=1` restores the pre-#56 geometry exactly (one invocation per workgroup, an exact grid), so
/// it is the kill switch for this change in the same sense that
/// `ONNXRUNTIME_EP_VULKAN_GEMV_MAX_ROWS=1` is the kill switch for the row tile.
pub const ENV_GQA_LOCAL_SIZE: &str = "ONNXRUNTIME_EP_VULKAN_GQA_LOCAL_SIZE";

/// How many invocations of `gqa_f16` share a workgroup.
///
/// Pure so it can be tested without a device. Two properties matter and they pull opposite ways:
///
///   * **Lane occupancy.** A workgroup smaller than the subgroup wastes the lanes it does not
///     fill, and the subgroup width is not knowable portably at translate time (32 on NVIDIA,
///     64 on AMD, 8/16/32 on Intel). A size of 64 is a multiple of every one of those, so it
///     fills whole subgroups on all three rather than being tuned to one.
///
///   * **Spread across compute units.** Workgroups are the unit the hardware distributes, so
///     packing *all* the work into one workgroup hands the kernel to one compute unit. A decode
///     step has `B * Nq * S = 32` invocations in total; at 64 that is a single workgroup and the
///     rest of the device is idle, which is why the size is capped by the work available rather
///     than being a constant.
///
/// So: fill subgroups when there is enough work to keep the machine busy anyway, and keep the
/// spread when there is not. [`GQA_MIN_GROUPS`] is the number of workgroups below which the
/// spread is judged to matter more than the packing.
///
/// MEASURED, not assumed — `bench/results/real_model_gqa_local_size.json`, Foundry Phi-3.5 int4
/// on an RTX A1000, `gqa_f16`'s own GPU milliseconds per inference, one process per point:
///
/// ```text
///   invocations   local=1     2      4      8     16     32     64    rule picks
///   32  (decode)     1.48   1.76   1.69   1.70   1.89   1.97   1.81      1  <- best
///   32  (past 512) 135.56 139.72 141.11 146.84 166.96 209.19 203.80      1  <- best
///   32  (past1024) 268.57 273.77 278.69 291.81 322.72 392.42 390.53      1  <- best
///   256 (M=8)       20.96  16.58  11.10   5.92   5.30   5.10   6.09      8
///   1024 (M=32)    154.59  90.78  60.57  49.43  35.37  25.48  25.69     32  <- best
///   4096 (M=128)  1891.19 1002.68 549.56 315.98 229.62 201.71 194.27    64  <- best
/// ```
///
/// The rule is best or tied-best at five of the six points. The one place it leaves something is
/// `M = 8`, where it picks 8 (5.92 ms) over the sweep's best 32 (5.10 ms) — 16%, on the smallest
/// absolute number in the table, and the price of a rule that keeps **32 workgroups** of spread.
/// Buying that 0.8 ms means lowering `GQA_MIN_GROUPS` to 8, which changes the 32-invocation rows
/// from 1 to 4 and costs 135.56 -> 141.11 ms and 268.57 -> 278.69 ms on the two decode rows.
/// Decode is where a generation loop spends nearly all of its time, so the trade is declined and
/// decode keeps *exactly* its pre-#56 geometry: local 1, 32 workgroups, an exact grid.
///
/// Every NON-REFERENCE point above was also checked byte-for-byte against its case's `local = 1`
/// reference: 6 cases x 6 non-reference sizes = **36 comparisons, 36 BITWISE-IDENTICAL** (the
/// reference is not compared with itself; 7 sizes x 6 cases = 42 is the count of measured points,
/// not of comparisons). The packing changes scheduling, not arithmetic, and that is a claim about
/// the source text, so it is verified as one.
pub fn gqa_local_size(total: u32) -> u32 {
    gqa_local_size_with(total, gqa_local_size_override())
}

/// Fewest workgroups a GQA dispatch keeps, so packing never starves the device of parallelism.
///
/// Same rule and the same reasoning as `ops::quant`'s `GEMV_MIN_WORKGROUPS`: a small
/// absolute number, not a multiple of anything the device reports, so the geometry cannot become
/// device-dependent.
///
/// 32 rather than 8 or 16 because Phi-3.5 decode has exactly 32 invocations, and every value
/// below 32 packs them and makes decode slower — see the table on [`gqa_local_size`].
pub const GQA_MIN_GROUPS: u32 = 32;

fn gqa_local_size_override() -> Option<u32> {
    std::env::var(ENV_GQA_LOCAL_SIZE)
        .ok()
        .and_then(|v| v.trim().parse::<u32>().ok())
        .map(|n| n.clamp(1, GQA_MAX_LOCAL_SIZE))
}

/// The body of [`gqa_local_size`] with the environment lifted out, so it is testable.
///
/// The clamp is repeated here rather than trusted from [`gqa_local_size_override`]: this function
/// is `pub(crate)` and a caller that hands it `Some(0)` must get a dispatchable geometry, not a
/// division by zero. `0` is not a hypothetical — it is what
/// `ONNXRUNTIME_EP_VULKAN_GQA_LOCAL_SIZE=0` parses to.
pub(crate) fn gqa_local_size_with(total: u32, override_size: Option<u32>) -> u32 {
    if let Some(n) = override_size {
        return n.clamp(1, GQA_MAX_LOCAL_SIZE);
    }
    let total = total.max(1);
    // The largest power of two that still leaves `GQA_MIN_GROUPS` workgroups to spread.
    let mut local = 1u32;
    while local * 2 <= GQA_MAX_LOCAL_SIZE && total / (local * 2) >= GQA_MIN_GROUPS {
        local *= 2;
    }
    local
}

// ===========================================================================================
// The decode-only KV-parallel module (#90)
// ===========================================================================================

/// The SPIR-V module `translate_gqa` dispatches when the KV-parallel selector admits a node.
///
/// Named once, here, because three places must agree on it and none of them can check the others:
/// the dispatch in `translate_gqa`, `registry::ProofKey::variant_modules` (which expands the
/// `@kvpar` suffix into the module set a key's provability is a conjunction over), and
/// `vk::session`'s KV-convention recorder.
pub const GQA_DECODE_MODULE: &str = "gqa_decode_f16";

/// Most lanes a `gqa_decode_f16` workgroup will ever have.
///
/// A property of the **specification**, like [`GQA_MAX_LOCAL_SIZE`] next door, and derived rather
/// than tuned. `gqa_decode_f16` reserves `s_max[16] + s_sum[16] + s_acc[16 * 128]` = 8,320 B of
/// shared memory, and Vulkan 1.1 guarantees `maxComputeSharedMemorySize >= 16384`. Sixteen lanes
/// is therefore the largest count that fits the guaranteed floor with room to spare on every
/// conformant implementation, whatever the local device happens to report — the merge tree's
/// accumulator is `lanes * head_dim` floats and 32 lanes at `head_dim = 128` is 16,384 B on its
/// own, i.e. the entire floor with nothing left for the running maxima and sums.
///
/// Reading the device's real limit would buy a wider tree on some machines and make the emitted
/// pipeline — and therefore the arithmetic's summation order — device-dependent. Declined for the
/// same reason `GQA_MAX_LOCAL_SIZE` declines it.
pub const GQA_DECODE_MAX_KV_LANES: u32 = 16;

/// Largest `head_dim` `gqa_decode_f16` accepts.
///
/// The module's `s_acc` row stride, and the bound on its private `q`/`k_new`/`v_new`/`acc`
/// arrays — which is the same 128 `gqa_f16.comp` already declares. A wider head is refused here
/// rather than truncated: `gqa_f16` handles it, correctly and serially.
pub const GQA_DECODE_MAX_HEAD_DIM: u32 = 128;

/// Fewest KV positions a lane must be given before splitting the walk is worth its merge.
///
/// The merge is a halving tree: `log2(lanes) <= 4` barriers, each moving `head_dim` floats
/// through shared memory. The walk is `total_len / lanes` iterations, each reading `2 * head_dim`
/// f16 elements from the cache. At 32 positions per lane the walk does ~16x the traffic of the
/// merge that follows it, so the reduction is amortised rather than dominant; below that the tree
/// starts to be the kernel. It is deliberately the same constant on both sides of the ABI —
/// `gqa_decode_f16.comp` declares `MIN_KV_PER_LANE` and
/// `the_shader_pins_the_min_kv_per_lane_constant` fails if the two drift apart.
///
/// Not measured. It is a bound derived from the traffic ratio, and the honest thing to do with an
/// underived constant is to name it as one: the real ladder is a candidate for a sweep exactly
/// like `gqa_local_size`'s, and until that sweep exists this number is an argument rather than a
/// reading. The kill switch below is what makes the argument falsifiable.
pub const GQA_DECODE_MIN_KV_PER_LANE: u32 = 32;

/// Overrides [`gqa_decode_kv_lanes`]. Clamped to `[1, GQA_DECODE_MAX_KV_LANES]` and floored to a
/// power of two, never trusted.
///
/// `=1` is the kill switch: it restores the pre-#90 dispatch exactly — `gqa_f16`, the same
/// geometry, the same SPIR-V — in the same sense that `ONNXRUNTIME_EP_VULKAN_GQA_LOCAL_SIZE=1`
/// is the kill switch for #56's packing. Larger values force a lane count for the A/B ladder.
///
/// **It cannot change which module a node's proof key names.** See
/// [`gqa_decode_module_is_reachable`]: the key is computed from the declared form with no
/// environment input at all, and this override can only move the dispatch *down* to `gqa_f16`,
/// never up onto a form the ladder refused. A key that named a set the environment could widen
/// would be a proof identity that depends on the machine it was read on.
pub const ENV_GQA_DECODE_KV_LANES: &str = "ONNXRUNTIME_EP_VULKAN_GQA_DECODE_KV_LANES";

/// The shape facts the decode selector reads, and nothing else.
///
/// `None` means *the graph did not state it*, which is a different answer from any number and is
/// why every field is an `Option`. ORT spells symbolic dimensions as negative and `EdgeType`
/// normalises them to `-1`; [`gqa_decode_form_from_declared`] maps that to `None` so no predicate
/// here has to know the encoding.
///
/// The same struct serves claim time (declared, often symbolic) and translate time (concrete),
/// which is the point: the two decisions are then demonstrably the same function of the same
/// facts, and `dispatch_is_inside_the_declared_dispatch_set` can assert it exhaustively.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct GqaDecodeForm {
    /// Query sequence length — `packed_qkv.shape[1]`.
    pub seq_len: Option<i64>,
    /// Per-head width, derived from `packed_qkv.shape[2] / (num_heads + 2 * kv_num_heads)`.
    pub head_dim: Option<i64>,
    /// The KV cache extent — `past_key.shape[2]`.
    ///
    /// **An upper bound, not a length.** Under the KV arena (`ONNXRUNTIME_EP_VULKAN_KV_ARENA`)
    /// this is a capacity the caller reserved and the true past length is carried by `seqlens_k`,
    /// which no host-side predicate can read. That ambiguity is not resolved by refusing the
    /// arena — which would decline the feature precisely where long contexts make it worth
    /// having — but by treating this number as the bound it is: it caps the lane count the
    /// pipeline is built for, and `gqa_decode_f16.comp` recomputes the *effective* lane count
    /// from `seqlens_k[b]` with the same ladder. A 4,096-slot arena holding 8 tokens therefore
    /// builds a 16-lane pipeline and runs 1 lane, which is `gqa_f16`'s serial walk with no
    /// barrier executed. The same mechanism handles ragged batches.
    pub past_extent: Option<i64>,
}

/// Read a [`GqaDecodeForm`] out of declared shapes and attributes.
///
/// Total and never failing: anything not stated, not positive where it must be, or not divisible
/// comes back `None`, and every predicate below has an explicit answer for `None`.
pub(crate) fn gqa_decode_form_from_declared(
    qkv_shape: Option<&[i64]>,
    past_key_shape: Option<&[i64]>,
    num_heads: Option<i64>,
    kv_num_heads: Option<i64>,
) -> GqaDecodeForm {
    // ORT normalises every symbolic dimension to -1, so "stated" is "not negative". Zero is
    // stated: an empty past cache is a real, and refusable, form.
    let stated = |d: i64| -> Option<i64> { (d >= 0).then_some(d) };
    let dim = |s: Option<&[i64]>, i: usize| -> Option<i64> { s?.get(i).copied().and_then(stated) };

    let head_dim = match (dim(qkv_shape, 2), num_heads, kv_num_heads) {
        (Some(packed), Some(nq), Some(nkv)) if nq > 0 && nkv > 0 => {
            let lanes = nq + 2 * nkv;
            (packed > 0 && packed % lanes == 0).then_some(packed / lanes)
        }
        _ => None,
    };
    GqaDecodeForm {
        seq_len: dim(qkv_shape, 1),
        head_dim,
        past_extent: dim(past_key_shape, 2),
    }
}

/// How many lanes `gqa_decode_f16` gets for this node — **`1` means refuse**.
///
/// One is not a lane count here, it is the refusal value: the module is never dispatched at one
/// lane, because at one lane it is `gqa_f16`'s serial walk carrying a workgroup's worth of shared
/// memory and a merge tree it will not use. So the caller's test is `lanes > 1`, and there is
/// exactly one fallback path rather than a fast one and a slow one.
///
/// # The gates come before the override
///
/// Every refusal below is a *correctness* boundary of `gqa_decode_f16.comp`, so the override is
/// read only after they have all passed and cannot reach past any of them:
///
///   * **`seq_len != 1`.** The module has no branch for `past_len <= t < tok_pos` — a key
///     belonging to a sibling query position of the same dispatch — because that range is empty
///     exactly when `seq_len == 1`. Prefill keeps `gqa_f16`, whose recompute-from-`packed_qkv`
///     is what makes that case race-free.
///   * **`head_dim` outside `1 ..= 128`.** The module's shared row stride and private arrays are
///     128 wide; a wider head would index past both.
///   * **An unrecoverable extent.** A `None` past extent is not an empty cache, and guessing
///     which it is has already cost this file one defect (see `past_len_max` in `translate_gqa`).
///   * **A ladder that lands on one lane.** Fewer than [`GQA_DECODE_MIN_KV_PER_LANE`] positions
///     per lane and the merge tree costs more than the split saves.
///
/// # The override widens nothing
///
/// `override_lanes` is clamped to `[1, GQA_DECODE_MAX_KV_LANES]` and floored to a power of two,
/// and it is consulted **only when the ladder already chose more than one lane**. So `=1` turns
/// the feature off and larger values move within a dispatch set the declared form already
/// admitted. That asymmetry is what keeps [`gqa_decode_module_is_reachable`] — and therefore the
/// proof key — a function of the graph alone; it is the property the #90 v3 artifact did not
/// have, and the reason its key moved with the environment.
///
/// The clamp is repeated here rather than trusted from the caller for the reason
/// [`gqa_local_size_with`] states: `ONNXRUNTIME_EP_VULKAN_GQA_DECODE_KV_LANES=0` parses to
/// `Some(0)`, and a `0` that reached the ladder would be a division by zero.
pub(crate) fn gqa_decode_kv_lanes_with(form: GqaDecodeForm, override_lanes: Option<u32>) -> u32 {
    if form.seq_len != Some(1) {
        return 1;
    }
    match form.head_dim {
        Some(d) if (1..=GQA_DECODE_MAX_HEAD_DIM as i64).contains(&d) => {}
        _ => return 1,
    }
    let Some(past) = form.past_extent else {
        return 1;
    };
    if past < 0 {
        return 1;
    }
    let total = past.saturating_add(1).min(u32::MAX as i64) as u32;
    let mut lanes = 1u32;
    while lanes * 2 <= GQA_DECODE_MAX_KV_LANES && total / (lanes * 2) >= GQA_DECODE_MIN_KV_PER_LANE
    {
        lanes *= 2;
    }
    if lanes == 1 {
        return 1;
    }
    match override_lanes {
        Some(n) => {
            let n = n.clamp(1, GQA_DECODE_MAX_KV_LANES);
            // Floor to a power of two: the merge is a halving tree and a lane count that is not
            // a power of two would leave a lane un-merged at the top of it.
            1u32 << (u32::BITS - 1 - n.leading_zeros())
        }
        None => lanes,
    }
}

/// [`gqa_decode_kv_lanes_with`] with the environment read for it. Dispatch-time only.
pub(crate) fn gqa_decode_kv_lanes(form: GqaDecodeForm) -> u32 {
    gqa_decode_kv_lanes_with(form, gqa_decode_kv_lanes_override())
}

fn gqa_decode_kv_lanes_override() -> Option<u32> {
    std::env::var(ENV_GQA_DECODE_KV_LANES)
        .ok()
        .and_then(|v| v.trim().parse::<u32>().ok())
        .map(|n| n.clamp(1, GQA_DECODE_MAX_KV_LANES))
}

/// Whether **any** run of a node with this declared form can dispatch `gqa_decode_f16`.
///
/// This is the proof key's question, and it is a different question from
/// [`gqa_decode_kv_lanes_with`]'s. That one asks *what will this dispatch do*, from concrete
/// numbers, at `Compute()`. This one asks *what could any dispatch of this node do*, from the
/// graph's declared shapes, at claim time — when the extents that decide it are frequently
/// symbolic and the answer therefore has to cover every value they might take.
///
/// # It takes no environment, by signature
///
/// There is no override parameter and no `std::env` call on this path, which is the whole point.
/// A proof key that moved with an environment variable would mean the checked-in ledger's
/// identity is not the identity a run computes — the same entry would prove one module on one
/// machine and a different set on another, and `--check` could not tell. The kill switch can only
/// narrow what actually runs (see [`gqa_decode_kv_lanes_with`]), so a key derived here is a
/// superset of any run's dispatch set on any machine, in every environment.
///
/// # Fail-safe, and which way "safe" points
///
/// Unknown means **reachable**. If the key omitted `@kvpar` while a run dispatched
/// `gqa_decode_f16`, the ledger would return a proof of `gqa_f16` for a module that never
/// appeared in it — a proof of one artifact accepted for another, which is the exact integrity
/// failure the ledger exists to prevent. The other direction is merely imprecise: the key names a
/// set, a given run may dispatch only part of it, and `SpecWitness::Partial` already has a word
/// for that.
///
/// So only a *stated* fact that provably excludes the decode module drops the suffix:
///
///   * a declared `seq_len` other than 1 — prefill, which the module refuses outright;
///   * a declared `head_dim` outside `1 ..= 128` — wider than the module's arrays;
///   * a declared cache extent whose whole range lands on one lane. The extent is an upper bound
///     on the length (see [`GqaDecodeForm::past_extent`]), so if the ladder refuses the bound it
///     refuses every length beneath it, and no run of this node can reach the module.
///
/// Everything else — a symbolic sequence length, a symbolic cache, an underivable head width —
/// is reachable.
pub(crate) fn gqa_decode_module_is_reachable(form: GqaDecodeForm) -> bool {
    if let Some(s) = form.seq_len {
        if s != 1 {
            return false;
        }
    }
    if let Some(d) = form.head_dim {
        if !(1..=GQA_DECODE_MAX_HEAD_DIM as i64).contains(&d) {
            return false;
        }
    }
    if let Some(past) = form.past_extent {
        // Ask the ladder the same question the dispatch will, with the two axes it cannot see
        // here filled in as "the value that does not itself refuse". If it still lands on one
        // lane, the extent alone settles it.
        let probe = GqaDecodeForm {
            seq_len: Some(1),
            head_dim: Some(1),
            past_extent: Some(past),
        };
        if gqa_decode_kv_lanes_with(probe, None) == 1 {
            return false;
        }
    }
    true
}

/// The `@kvpar` suffix `registry::variant_key` appends, or `None` for every other op.
///
/// Lives here rather than in `registry.rs` because the facts it reads — which shapes, which
/// attributes, the lane ladder — are this file's, and a copy of them next to the key builder is a
/// copy that can drift from the dispatch it is supposed to describe.
///
/// Returns `Some("kvpar")` when `gqa_decode_f16` is in the node's dispatch set, so that the key's
/// variant component names the *set of modules* the node can run rather than one of them. The
/// stem is unchanged — `ProofKey::variant_stem` strips `@`-suffixes — so `gqa_f16@kvpar` still
/// resolves to a declared, loadable module and `form_is_provable` reads the same branch it did
/// before this existed.
pub(crate) fn gqa_dispatch_set_suffix(spec: &OpSpec, view: &NodeView<'_>) -> Option<&'static str> {
    if spec.qualified_name() != "com.microsoft::GroupQueryAttention" {
        return None;
    }
    let qkv = view.input_type(0).and_then(|e| e.shape);
    let past = view.input_type(3).and_then(|e| e.shape);
    let form = gqa_decode_form_from_declared(
        qkv.as_deref(),
        past.as_deref(),
        view.attr_int("num_heads"),
        view.attr_int("kv_num_heads"),
    );
    gqa_decode_module_is_reachable(form).then_some("kvpar")
}

crate::op_table! {
    //  op                     domain  opsets                       caps    kernel          claim                   translate                  status              schema
    //
    // `caps` is F16 and not FLOAT, and that narrowing is a consequence of naming the module in the
    // row (2026-08-04, Mouse). `translate_gqa` has always refused f32 outright — *"an f32 GQA
    // shader is not implemented"* — and there is no `gqa_f32.comp`, so the previous FLOAT let an
    // f32 node pass the claim gate and fail at translate, which is a partition-compile failure
    // where a `[dtype]` decline was meant. It is the same argument `Conv`'s row already makes for
    // f16 in the other direction: a dtype with no module declines at the gate.
    //
    // `selects: "gqa_decode"` names the SECOND module this row's `translate` may dispatch (#90):
    // a decode form the KV-parallel selector admits runs `gqa_decode_f16`, everything else runs
    // `gqa_f16`. The row declares the whole dispatch set because the proof key, the provability
    // predicate and the two build-coherence tests all read the declaration and none of them reads
    // `translate_gqa`.
    "GroupQueryAttention",     Ms,     1 ..= OPSET_ANY,             F16,    kernel!(Standalone, "gqa", selects: "gqa_decode"),  group_query_attention,  translate_gqa,             Live,               schema: &GROUP_QUERY_ATTENTION;
    "RotaryEmbedding",         Ms,     1 ..= OPSET_ANY,             FLOAT,  kernel!(None),  rotary_embedding,       templates::unimplemented,  Staged(XL_KERNEL),  schema: &ROTARY_EMBEDDING;
    "MultiHeadAttention",      Ms,     1 ..= OPSET_ANY,             FLOAT,  kernel!(None),  claim::never,           templates::unimplemented,  Staged(XL_KERNEL),  schema: &MULTI_HEAD_ATTENTION;
    "Attention",               Ai,     OPSET_STD_LLM ..= OPSET_STD_ATTENTION_MAX, FLOAT,  kernel!(None),  std_attention,          templates::unimplemented,  Staged(XL_KERNEL);
    "RotaryEmbedding",         Ai,     OPSET_STD_LLM ..= OPSET_STD_NORM_MAX,      FLOAT,  kernel!(None),  std_rotary_embedding,   templates::unimplemented,  Staged(XL_KERNEL);
    "TensorScatter",           Ai,     OPSET_STD_TENSOR_SCATTER ..= OPSET_STD_TENSOR_SCATTER, ANY, kernel!(None), tensor_scatter, templates::unimplemented, Staged(NO_SHADER);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{
        BufferView, DispatchContext, EpResult, KernelRequest, NodeDesc, OutRef, TensorDesc,
        TensorRef,
    };
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
    fn gqa_is_live_everything_else_is_staged() {
        let live: Vec<_> = OPS.iter().filter(|s| s.is_live()).collect();
        assert_eq!(
            live.len(),
            1,
            "only GroupQueryAttention should be live in this module"
        );
        assert_eq!(live[0].op_type, "GroupQueryAttention");
        for s in OPS {
            if s.op_type != "GroupQueryAttention" {
                assert!(
                    matches!(s.status, OpStatus::Staged(_)),
                    "{} should still be staged",
                    s.op_type
                );
            }
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

    // ── translate_gqa tests ────────────────────────────────────────────────────────────────

    /// Minimal test recorder for translate handlers.  Mirrors `templates::tests::Recorder` but is
    /// a private copy here so the test module can assert on GQA-specific properties.
    #[derive(Default)]
    struct Recorder {
        next: u64,
        dispatches: Vec<KernelRequest>,
        outputs: Vec<TensorDesc>,
    }

    impl DispatchContext for Recorder {
        fn resolve(&mut self, _r: &crate::engine::TensorRef) -> EpResult<BufferView> {
            self.next += 1;
            Ok(BufferView::from_raw(self.next))
        }
        fn bind_output(
            &mut self,
            _o: &crate::engine::OutRef,
            desc: TensorDesc,
        ) -> EpResult<BufferView> {
            self.outputs.push(desc);
            self.next += 1;
            Ok(BufferView::from_raw(self.next))
        }
        fn alloc_temp(&mut self, desc: TensorDesc) -> EpResult<BufferView> {
            self.outputs.push(desc);
            self.next += 1;
            Ok(BufferView::from_raw(self.next))
        }
        fn dispatch(&mut self, k: KernelRequest) -> EpResult<()> {
            self.dispatches.push(k);
            Ok(())
        }
        fn read_const_i64(&self, _r: &crate::engine::TensorRef) -> Option<Vec<i64>> {
            None
        }
    }

    fn gqa_node(b: i64, s: i64, nq: i64, nkv: i64, d: i64, past_max: i64) -> NodeDesc {
        // Growing-cache convention: present is [B, Nkv, past + S, D], a different and larger
        // buffer than past. Both graphs this EP targets declare it this way.
        gqa_node_with_present(b, s, nq, nkv, d, past_max, past_max + s, d / 2)
    }

    #[allow(clippy::too_many_arguments)]
    fn gqa_node_with_present(
        b: i64,
        s: i64,
        nq: i64,
        nkv: i64,
        d: i64,
        past_max: i64,
        present_max: i64,
        rot_half: i64,
    ) -> NodeDesc {
        // packed_qkv: [B, S, (Nq+2*Nkv)*D]; cos_cache/sin_cache: [max_pos, rot_half], where
        // the second dim IS half the rotary width.
        let qkv_dim = (nq + 2 * nkv) * d;
        let mut attrs = std::collections::BTreeMap::new();
        attrs.insert("num_heads".into(), AttrValue::Int(nq));
        attrs.insert("kv_num_heads".into(), AttrValue::Int(nkv));
        attrs.insert("scale".into(), AttrValue::Float((d as f32).sqrt().recip()));
        let make_ref = |name: &str, shape: Vec<i64>| TensorRef {
            name: name.into(),
            desc: Some(TensorDesc::new(DType::F16, shape)),
            is_initializer: false,
        };
        let empty_ref = || TensorRef {
            name: String::new(),
            desc: None,
            is_initializer: false,
        };
        // seqlens_k is int32; the translate handler only resolves it by index, doesn't check dtype here
        let seqlens_ref = TensorRef {
            name: "seqlens_k".into(),
            desc: Some(TensorDesc::new(DType::I32, vec![b])),
            is_initializer: false,
        };
        let make_out = |name: &str, shape: Vec<i64>| OutRef {
            name: name.into(),
            desc: Some(TensorDesc::new(DType::F16, shape)),
        };
        NodeDesc {
            op_type: "GroupQueryAttention".into(),
            domain: "com.microsoft".into(),
            attributes: attrs,
            inputs: vec![
                make_ref("qkv", vec![b, s, qkv_dim]),          // 0 packed_qkv
                empty_ref(),                                   // 1 absent
                empty_ref(),                                   // 2 absent
                make_ref("past_k", vec![b, nkv, past_max, d]), // 3 past_key
                make_ref("past_v", vec![b, nkv, past_max, d]), // 4 past_value
                seqlens_ref,                                   // 5 seqlens_k
                empty_ref(),                                   // 6 total_seq (optional)
                make_ref("cos", vec![4096, rot_half]),         // 7 cos_cache
                make_ref("sin", vec![4096, rot_half]),         // 8 sin_cache
            ],
            outputs: vec![
                make_out("attn", vec![b, s, nq * d]),
                make_out("pres_k", vec![b, nkv, present_max, d]),
                make_out("pres_v", vec![b, nkv, present_max, d]),
            ],
            ..Default::default()
        }
    }

    #[test]
    fn translate_gqa_phi35_decode_produces_one_dispatch() {
        // Phi-3.5: B=1, S=1, Nq=32, Nkv=32, D=96, past_max=256 (example)
        let spec = crate::registry::all_specs()
            .find(|s| s.op_type == "GroupQueryAttention")
            .unwrap();
        let node = gqa_node(1, 1, 32, 32, 96, 256);
        let mut ctx = Recorder::default();
        translate_gqa(spec, &node, &mut ctx).expect("translate should succeed");

        assert_eq!(ctx.dispatches.len(), 1, "exactly one dispatch");
        let k = &ctx.dispatches[0];
        // #90: at 257 KV positions the selector admits the KV-parallel module, so this shape now
        // dispatches `gqa_decode_f16`. The push-constant block below is the reason this test
        // still lives at these shapes rather than moving to a refusing one: the two modules share
        // one block and one binding list, so the layout has to be pinned on the path that is
        // actually taken. `translate_gqa_decode_declares_the_unpacked_size` is the companion that
        // pins the refusal path.
        assert_eq!(
            k.shader, GQA_DECODE_MODULE,
            "257 KV positions is 8 lanes x 32, so decode takes the KV-parallel module"
        );
        assert_eq!(
            k.spec_constants,
            vec![8],
            "the lane count must reach the pipeline; without it the module keeps its default of \
             1 lane and the reduction it was built for never happens"
        );
        assert_eq!(
            k.workgroups,
            [32, 1, 1],
            "one workgroup per (batch, query head): B*Nq = 1*32 = 32"
        );
        assert_eq!(k.bindings.len(), 9, "9 bindings: 6 inputs + 3 outputs");
        // Phi-3.5 declares the growing-cache convention (present S = past + S), so present
        // is a fresh, larger allocation rather than an alias onto past: attn + present_key
        // + present_value are all real outputs.
        assert_eq!(
            ctx.outputs.len(),
            3,
            "growing cache: attn + present_key + present_value are real allocations"
        );

        // Verify push constants byte layout (36 bytes = 9 × u32/f32)
        let pc = &k.push_constants;
        assert_eq!(pc.len(), 36, "push constant block must be 36 bytes");
        let batch_size = u32::from_le_bytes(pc[0..4].try_into().unwrap());
        let seq_len = u32::from_le_bytes(pc[4..8].try_into().unwrap());
        let num_heads = u32::from_le_bytes(pc[8..12].try_into().unwrap());
        let kv_num_heads = u32::from_le_bytes(pc[12..16].try_into().unwrap());
        let head_dim = u32::from_le_bytes(pc[16..20].try_into().unwrap());
        let rotary_dim = u32::from_le_bytes(pc[20..24].try_into().unwrap());
        let present_len = u32::from_le_bytes(pc[24..28].try_into().unwrap());
        let past_stride = u32::from_le_bytes(pc[28..32].try_into().unwrap());
        let scale_bits = u32::from_le_bytes(pc[32..36].try_into().unwrap());
        let scale = f32::from_bits(scale_bits);

        assert_eq!(batch_size, 1);
        assert_eq!(seq_len, 1);
        assert_eq!(num_heads, 32);
        assert_eq!(kv_num_heads, 32);
        assert_eq!(head_dim, 96);
        assert_eq!(
            rotary_dim, 96,
            "rotary_dim comes from cos_cache's second dim (48) × 2"
        );
        assert_eq!(
            present_len, 257,
            "KV write stride is the PRESENT extent, past + seq_len"
        );
        assert_eq!(
            past_stride, 256,
            "past reads still stride by the past extent"
        );
        assert!(
            (scale - 96f32.sqrt().recip()).abs() < 1e-6,
            "scale = 1/sqrt(D)"
        );
    }

    // -- #56: the GQA workgroup size ------------------------------------------------------
    //
    // Every test below names the wrong reading it prevents, because a table of expected
    // outputs proves only that someone typed the same numbers twice.

    /// The whole point of the rule: **decode keeps its pre-#56 geometry, exactly.**
    ///
    /// Phi-3.5 decode is `B * Nq * S = 1 * 32 * 1 = 32` invocations, and the sweep
    /// (`bench/results/real_model_gqa_local_size.json`) says every packing above 1 makes it
    /// *slower* — 135.56 ms at local 1 against 209.19 ms at local 32, on the `past = 512` row.
    /// A refactor that "simplified" `GQA_MIN_GROUPS` away, or lowered it to catch the `M = 8`
    /// point, would silently take that regression on the case a generation loop spends its
    /// life in. This asserts local 1, not merely "small".
    #[test]
    fn gqa_decode_stays_at_one_invocation_per_workgroup() {
        assert_eq!(
            gqa_local_size_with(32, None),
            1,
            "32 invocations is Phi-3.5 decode; packing it measured SLOWER at every size"
        );
        assert_eq!(
            32u32.div_ceil(gqa_local_size_with(32, None)),
            32,
            "and the grid stays exact: 32 workgroups, no over-dispatched tail"
        );
    }

    /// The rule reproduces the sweep's own choices at the invocation counts it measured.
    ///
    /// This is the differential test against the artifact: if someone changes the constants or
    /// the loop, this fails and names the row that no longer agrees. `M = 8` is included with
    /// the value the rule picks (8), *not* the sweep's best (32), so the deliberate trade
    /// documented on `gqa_local_size` cannot be quietly reversed in either direction.
    #[test]
    fn gqa_local_size_matches_the_measured_sweep() {
        // (invocations, expected local size, what the row is)
        for (total, want, what) in [
            (32u32, 1u32, "decode / prefill M=1 — best measured"),
            (
                256,
                8,
                "prefill M=8 — rule trades 16% here to keep 32 groups of spread",
            ),
            (1024, 32, "prefill M=32 — best measured"),
            (4096, 64, "prefill M=128 — best measured"),
        ] {
            assert_eq!(
                gqa_local_size_with(total, None),
                want,
                "{total} invocations ({what})"
            );
        }
    }

    /// The cap is a Vulkan 1.1 guarantee, not a device reading, so no input can exceed it.
    ///
    /// `maxComputeWorkGroupInvocations` has a specification floor of 128; asking for more than
    /// 64 would make the pipeline creation fail on a conformant implementation that reports
    /// exactly the floor for the x dimension. Neither a huge grid nor a hostile override may
    /// produce one.
    #[test]
    fn gqa_local_size_never_exceeds_the_portable_cap() {
        for total in [4096u32, 1 << 20, u32::MAX] {
            let local = gqa_local_size_with(total, None);
            assert!(
                local <= GQA_MAX_LOCAL_SIZE,
                "{total} invocations produced local size {local}"
            );
            assert!(local.is_power_of_two(), "sizes stay powers of two");
        }
        assert_eq!(
            gqa_local_size_with(4096, Some(4096)),
            GQA_MAX_LOCAL_SIZE,
            "an override above the cap clamps to it rather than being honoured"
        );
    }

    /// **Planted control.** `ONNXRUNTIME_EP_VULKAN_GQA_LOCAL_SIZE=0` parses to `Some(0)`, and a
    /// workgroup size of zero is an undispatchable geometry *and* a division by zero in
    /// `total.div_ceil(local)`. The clamp must survive being handed the value the environment
    /// can actually produce, not just the values a well-behaved caller would.
    #[test]
    fn gqa_local_size_override_of_zero_is_clamped_not_dispatched() {
        assert_eq!(gqa_local_size_with(4096, Some(0)), 1);
        assert_eq!(gqa_local_size_with(32, Some(0)), 1);
        // And the geometry that comes out of it is dispatchable.
        let local = gqa_local_size_with(4096, Some(0));
        assert!(local >= 1, "div_ceil below would panic on 0");
        assert_eq!(4096u32.div_ceil(local), 4096);
    }

    /// The kill switch means what it says: `=1` restores the pre-#56 dispatch exactly.
    #[test]
    fn gqa_local_size_override_of_one_restores_the_original_geometry() {
        for total in [32u32, 256, 1024, 4096] {
            assert_eq!(gqa_local_size_with(total, Some(1)), 1);
            assert_eq!(
                total.div_ceil(1),
                total,
                "one workgroup per invocation, which is the geometry #56 replaced"
            );
        }
    }

    /// **Boundary.** The grid must cover every invocation and never lose one.
    ///
    /// `div_ceil` over-dispatches when the size does not divide the count, and the shader's
    /// `b >= batch_size` guard retires the tail. Under-dispatching would silently drop query
    /// positions — attention output for the last tokens would be whatever the buffer held. This
    /// walks counts that are deliberately *not* multiples of any candidate size.
    #[test]
    fn gqa_dispatch_grid_covers_every_invocation() {
        for total in [
            1u32, 2, 31, 32, 33, 255, 257, 1023, 1025, 4095, 4097, 100_003,
        ] {
            let local = gqa_local_size_with(total, None);
            let groups = total.div_ceil(local);
            let covered = groups * local;
            assert!(
                covered >= total,
                "{total} invocations: {groups} x {local} = {covered} does not cover them"
            );
            assert!(
                covered - total < local,
                "{total} invocations: tail of {} is a whole extra workgroup",
                covered - total
            );
        }
    }

    /// **Monotone, and it must never shrink as work grows.**
    ///
    /// A rule that packed *less* at a larger invocation count would mean the loop's condition is
    /// reading the wrong way round — a defect that the four spot values above could not see,
    /// because they are all powers of two and the loop's bug would be at the edges.
    #[test]
    fn gqa_local_size_never_decreases_as_work_grows() {
        let mut prev = 0u32;
        for total in (0u32..=8192).step_by(7) {
            let local = gqa_local_size_with(total, None);
            assert!(
                local >= prev,
                "local size fell from {prev} to {local} at {total} invocations"
            );
            prev = local;
        }
    }

    /// Zero invocations must still produce a dispatchable size — `translate_gqa` clamps `total`
    /// to 1, but this function is `pub(crate)` and must not depend on its caller doing that.
    #[test]
    fn gqa_local_size_of_zero_work_is_still_dispatchable() {
        assert_eq!(gqa_local_size_with(0, None), 1);
    }

    /// The prefill dispatch geometry, end to end through `translate_gqa`.
    ///
    /// The pure-function tests above cannot see a wiring mistake: passing `total` where `groups`
    /// belongs, or forgetting to put the size in `spec_constants`, would leave every one of them
    /// green while the shader ran at its default size of 1 over 1/64th of the grid. This reads
    /// the recorded dispatch.
    #[test]
    fn translate_gqa_prefill_packs_the_workgroup_and_declares_it() {
        let spec = crate::registry::all_specs()
            .find(|s| s.op_type == "GroupQueryAttention")
            .unwrap();
        // Phi-3.5 prefill at M=128: B=1, S=128, Nq=32 -> 4096 invocations.
        let node = gqa_node(1, 128, 32, 32, 96, 0);
        let mut ctx = Recorder::default();
        translate_gqa(spec, &node, &mut ctx).expect("translate should succeed");
        let k = &ctx.dispatches[0];
        // B * Nq * S, written out so the geometry is legible.
        let (b, nq, s) = (1u32, 32u32, 128u32);
        let total = b * nq * s;
        let local = gqa_local_size(total);
        assert_eq!(
            k.spec_constants,
            vec![local],
            "the size must reach the pipeline; without it the shader keeps its default of 1 \
             and the grid is 64x too small"
        );
        assert_eq!(k.workgroups, [total.div_ceil(local), 1, 1]);
        assert_eq!(
            k.workgroups[0] * local,
            total,
            "4096 divides evenly, so this grid is exact"
        );
    }

    /// Decode's *fallback* dispatch, through the same path, asserted as **unchanged**.
    ///
    /// `translate_gqa_phi35_decode_produces_one_dispatch` pins the KV-parallel path at 257
    /// positions. This pins the other side of the selector: at 33 positions the ladder refuses
    /// (33 / 2 = 16, under `GQA_DECODE_MIN_KV_PER_LANE`), and a refusal must be the pre-#90
    /// dispatch **exactly** — `gqa_f16`, specialisation constant 1, an exact 32-workgroup grid.
    /// Not "something small", not "a decode module at one lane": the same stem, so the same
    /// SPIR-V, so no behaviour anywhere in the refusal set can have moved.
    #[test]
    fn translate_gqa_decode_declares_the_unpacked_size() {
        let spec = crate::registry::all_specs()
            .find(|s| s.op_type == "GroupQueryAttention")
            .unwrap();
        let node = gqa_node(1, 1, 32, 32, 96, 32);
        let mut ctx = Recorder::default();
        translate_gqa(spec, &node, &mut ctx).expect("translate should succeed");
        let k = &ctx.dispatches[0];
        assert_eq!(
            k.shader, "gqa_f16",
            "a short cache must fall back to the shipping module, byte-identical"
        );
        assert_eq!(k.spec_constants, vec![1], "decode is not packed");
        assert_eq!(k.workgroups, [32, 1, 1], "and its grid is untouched");
    }

    // ── #90: the decode-only KV-parallel selector ─────────────────────────────────────────
    //
    // Two rejections govern every test in this block, and each one is named on the test that
    // answers it:
    //
    //   B-1  the proof key must not move with the environment;
    //   B-2  the variant-key suffix, the dispatch-set gates, that environment independence and
    //        `form_is_provable`'s conjunction must all be covered by `cargo test --lib`, which
    //        is the suite the hosted lane runs. A GPU-only test is not coverage for a lane with
    //        no GPU.

    /// The ladder, at every boundary it has.
    ///
    /// A table, but not a table of numbers someone typed twice: each row is the smallest cache
    /// on either side of a rung, so a change to `GQA_DECODE_MIN_KV_PER_LANE` or to the
    /// power-of-two step moves at least one of them. `past + 1` is the walk length, because the
    /// query token is one of the positions attended to.
    #[test]
    fn decode_lane_ladder_lands_on_its_declared_rungs() {
        let lanes = |past: i64| {
            gqa_decode_kv_lanes_with(
                GqaDecodeForm {
                    seq_len: Some(1),
                    head_dim: Some(96),
                    past_extent: Some(past),
                },
                None,
            )
        };
        // 32 positions per lane is the rung width, so a rung opens at 2*32 - 1 past tokens.
        assert_eq!(lanes(62), 1, "63 positions is under 2 x 32 — refuse");
        assert_eq!(lanes(63), 2, "64 positions is exactly 2 x 32");
        assert_eq!(lanes(126), 2, "127 positions is under 4 x 32");
        assert_eq!(lanes(127), 4, "128 positions is exactly 4 x 32");
        assert_eq!(lanes(255), 8, "256 positions is exactly 8 x 32");
        assert_eq!(lanes(511), 16, "512 positions is exactly 16 x 32");
        assert_eq!(
            lanes(1_000_000),
            GQA_DECODE_MAX_KV_LANES,
            "the cap is the 16 KiB shared-memory floor, not the cache length"
        );
        assert_eq!(lanes(0), 1, "an empty cache is one position — refuse");
    }

    /// Every refusal gate, one at a time, on a form that would otherwise be admitted.
    ///
    /// Each gate is a correctness boundary of `gqa_decode_f16.comp`, so each one is asserted
    /// against a form that differs from an accepted form in exactly that field — otherwise a
    /// test can pass because some *other* gate fired.
    #[test]
    fn decode_selector_refuses_outside_the_module_boundaries() {
        let admitted = GqaDecodeForm {
            seq_len: Some(1),
            head_dim: Some(96),
            past_extent: Some(1024),
        };
        assert_eq!(
            gqa_decode_kv_lanes_with(admitted, None),
            16,
            "the control: this form is admitted, so every refusal below is its own gate"
        );

        for s in [0i64, 2, 8, 128, -1] {
            assert_eq!(
                gqa_decode_kv_lanes_with(
                    GqaDecodeForm {
                        seq_len: Some(s),
                        ..admitted
                    },
                    None
                ),
                1,
                "seq_len {s} is not decode; the module has no branch for a sibling query key"
            );
        }
        assert_eq!(
            gqa_decode_kv_lanes_with(
                GqaDecodeForm {
                    seq_len: None,
                    ..admitted
                },
                None
            ),
            1,
            "an unstated sequence length is not a decode step at dispatch time"
        );

        for d in [0i64, -1, 129, 256] {
            assert_eq!(
                gqa_decode_kv_lanes_with(
                    GqaDecodeForm {
                        head_dim: Some(d),
                        ..admitted
                    },
                    None
                ),
                1,
                "head_dim {d} is outside the module's 128-wide arrays"
            );
        }
        assert_eq!(
            gqa_decode_kv_lanes_with(
                GqaDecodeForm {
                    head_dim: Some(128),
                    ..admitted
                },
                None
            ),
            16,
            "128 is inclusive — the module's arrays are exactly that wide"
        );
        assert_eq!(
            gqa_decode_kv_lanes_with(
                GqaDecodeForm {
                    head_dim: None,
                    ..admitted
                },
                None
            ),
            1,
            "an underivable head width is refused, not guessed"
        );

        assert_eq!(
            gqa_decode_kv_lanes_with(
                GqaDecodeForm {
                    past_extent: None,
                    ..admitted
                },
                None
            ),
            1,
            "an unrecoverable cache extent is refused; guessing it is the defect \
             `past_len_max` already has a comment about"
        );
        assert_eq!(
            gqa_decode_kv_lanes_with(
                GqaDecodeForm {
                    past_extent: Some(-1),
                    ..admitted
                },
                None
            ),
            1,
            "a symbolic extent that reached the dispatch is refused"
        );
    }

    /// The override clamps, floors to a power of two, and — the part that matters — **cannot
    /// widen the dispatch set**.
    ///
    /// `=1` is the kill switch and must restore the pre-#90 dispatch. Every other value may only
    /// move within a form the ladder already admitted; on a refused form every value, including
    /// 16, still refuses. That asymmetry is what makes `gqa_decode_module_is_reachable` — and
    /// therefore the proof key — computable without reading the environment at all.
    #[test]
    fn decode_lane_override_narrows_but_never_widens() {
        let admitted = GqaDecodeForm {
            seq_len: Some(1),
            head_dim: Some(96),
            past_extent: Some(1024),
        };
        let refused = GqaDecodeForm {
            past_extent: Some(8),
            ..admitted
        };
        assert_eq!(gqa_decode_kv_lanes_with(refused, None), 1, "the control");

        assert_eq!(
            gqa_decode_kv_lanes_with(admitted, Some(1)),
            1,
            "=1 is the kill switch: the pre-#90 dispatch, exactly"
        );
        for n in [0u32, 1] {
            assert_eq!(
                gqa_decode_kv_lanes_with(admitted, Some(n)),
                1,
                "{n} clamps to the refusal value rather than dividing by zero"
            );
        }
        assert_eq!(gqa_decode_kv_lanes_with(admitted, Some(2)), 2);
        assert_eq!(gqa_decode_kv_lanes_with(admitted, Some(4)), 4);
        assert_eq!(
            gqa_decode_kv_lanes_with(admitted, Some(6)),
            4,
            "floored to a power of two: a halving tree cannot merge a lane that has no partner"
        );
        assert_eq!(gqa_decode_kv_lanes_with(admitted, Some(16)), 16);
        assert_eq!(
            gqa_decode_kv_lanes_with(admitted, Some(4096)),
            GQA_DECODE_MAX_KV_LANES,
            "clamped to the shared-memory cap, never trusted"
        );

        for n in [0u32, 1, 2, 4, 8, 16, 64, 4096] {
            assert_eq!(
                gqa_decode_kv_lanes_with(refused, Some(n)),
                1,
                "override {n} must not reach past a gate the declared form already closed"
            );
        }
    }

    /// **B-1's falsifier.** The proof key's predicate is a function of the graph alone.
    ///
    /// Asserted twice over: by signature — `gqa_decode_module_is_reachable` takes no override and
    /// makes no `std::env` call — and by behaviour, by driving the *dispatch* selector with every
    /// override value the environment can produce and checking that none of them moves the
    /// reachability answer for any form.
    ///
    /// The rejected #90 v3 artifact keyed on `ONNXRUNTIME_EP_VULKAN_GQA_DECODE_KV_PARALLEL`, so
    /// the same checked-in ledger described a different dispatch set on a machine that had the
    /// variable set. This is the test that goes red if that comes back.
    #[test]
    fn decode_reachability_is_independent_of_the_environment() {
        let forms = [
            // (form, reachable)
            (
                GqaDecodeForm {
                    seq_len: Some(1),
                    head_dim: Some(96),
                    past_extent: Some(1024),
                },
                true,
            ),
            (
                GqaDecodeForm {
                    seq_len: None,
                    head_dim: None,
                    past_extent: None,
                },
                true,
            ),
            (
                GqaDecodeForm {
                    seq_len: Some(1),
                    head_dim: Some(32),
                    past_extent: Some(4),
                },
                false,
            ),
            (
                GqaDecodeForm {
                    seq_len: Some(128),
                    head_dim: Some(96),
                    past_extent: None,
                },
                false,
            ),
            (
                GqaDecodeForm {
                    seq_len: Some(1),
                    head_dim: Some(256),
                    past_extent: None,
                },
                false,
            ),
        ];
        for (form, expected) in forms {
            assert_eq!(
                gqa_decode_module_is_reachable(form),
                expected,
                "reachability of {form:?} is a property of the graph"
            );
            // 0, 1 and 16 are the values with meaning — off, the kill switch and the cap — and
            // the rest are the values an operator can actually type.
            for n in [0u32, 1, 2, 3, 8, 16, 17, 64] {
                assert_eq!(
                    gqa_decode_module_is_reachable(form),
                    expected,
                    "reachability moved while probing override {n}"
                );
                let lanes = gqa_decode_kv_lanes_with(form, Some(n));
                assert!(
                    lanes == 1 || expected,
                    "override {n} dispatched {lanes} lanes on a form whose key says the \
                     KV-parallel module is unreachable — the key would then be a proof of \
                     `gqa_f16` returned for `gqa_decode_f16`"
                );
            }
        }
    }

    /// Every dispatch the selector can produce is inside the set the key declares.
    ///
    /// The soundness property the two predicates have to satisfy jointly, checked exhaustively
    /// over the interesting corner of the space rather than argued: if any concrete form can
    /// dispatch the KV-parallel module under any override, then the *declared* form — the one
    /// the key is computed from, with the same stated facts — must report it reachable.
    #[test]
    fn dispatch_is_inside_the_declared_dispatch_set() {
        let overrides = [None, Some(0), Some(1), Some(2), Some(5), Some(16), Some(99)];
        for &s in &[Some(0i64), Some(1), Some(2), Some(64), None] {
            for &d in &[Some(1i64), Some(96), Some(128), Some(129), None] {
                for &p in &[Some(0i64), Some(63), Some(127), Some(4096), None] {
                    let form = GqaDecodeForm {
                        seq_len: s,
                        head_dim: d,
                        past_extent: p,
                    };
                    for ov in overrides {
                        if gqa_decode_kv_lanes_with(form, ov) > 1 {
                            assert!(
                                gqa_decode_module_is_reachable(form),
                                "{form:?} dispatches the KV-parallel module at override {ov:?} \
                                 but its key would not name it"
                            );
                        }
                    }
                }
            }
        }
    }

    /// The evidence case's declared form must stay off the `@kvpar` key, and Phi-3.5's must not.
    ///
    /// This is the ledger consequence of the suffix, asserted where it is cheap to assert.
    /// `rust/tools/ledger_case_models.py` declares `group_query_attention_f16` with a literal
    /// past extent of 4 and a symbolic sequence length; four past tokens is five positions, which
    /// the ladder refuses at every override, so **no run of that graph can reach the new module**
    /// and its checked-in ledger entry keeps describing it exactly. Phi-3.5 declares
    /// `past_sequence_length` symbolically, so its key gains the suffix and needs an entry of its
    /// own — which is what `group_query_attention_kv_parallel_f16` exists to mint.
    #[test]
    fn the_small_evidence_case_keeps_its_existing_key() {
        // group_query_attention_f16: Nq=8, Nkv=2, D=32 -> packed 384; past 4; S symbolic.
        let evidence = gqa_decode_form_from_declared(
            Some(&[-1, -1, 384]),
            Some(&[-1, 2, 4, 32]),
            Some(8),
            Some(2),
        );
        assert_eq!(evidence.head_dim, Some(32), "384 / (8 + 2*2) = 32");
        assert_eq!(evidence.past_extent, Some(4));
        assert_eq!(evidence.seq_len, None, "-1 is ORT's symbolic dimension");
        assert!(
            !gqa_decode_module_is_reachable(evidence),
            "a 4-token declared cache refuses at every length beneath it, so the existing \
             ledger entry must keep matching this graph"
        );

        // Phi-3.5: Nq=32, Nkv=32, D=96 -> packed 9216; past and sequence both symbolic.
        let phi = gqa_decode_form_from_declared(
            Some(&[-1, -1, 9216]),
            Some(&[-1, 32, -1, 96]),
            Some(32),
            Some(32),
        );
        assert_eq!(phi.head_dim, Some(96), "9216 / (32 + 2*32) = 96");
        assert!(
            gqa_decode_module_is_reachable(phi),
            "a symbolic cache can take any length, so the key must name the module a long one \
             would dispatch — fail-safe is fail-toward-naming-it"
        );

        // The kv-parallel ledger case: a literal 128-token cache, which the ladder admits.
        let minting = gqa_decode_form_from_declared(
            Some(&[-1, -1, 384]),
            Some(&[-1, 2, 128, 32]),
            Some(8),
            Some(2),
        );
        assert!(
            gqa_decode_module_is_reachable(minting),
            "`group_query_attention_kv_parallel_f16` must key on `@kvpar` or it cannot mint the \
             entry Phi-3.5 needs"
        );
        assert_eq!(
            gqa_decode_kv_lanes_with(
                GqaDecodeForm {
                    seq_len: Some(1),
                    ..minting
                },
                None
            ),
            4,
            "129 positions is 4 x 32; the minting run must actually dispatch the module"
        );
    }

    /// Malformed and hostile declared shapes produce a form, never a panic and never a guess.
    #[test]
    fn decode_form_reads_declared_shapes_totally() {
        assert_eq!(
            gqa_decode_form_from_declared(None, None, None, None),
            GqaDecodeForm::default(),
            "nothing stated is nothing known"
        );
        // Rank-2 qkv: no dimension 2 to read.
        assert_eq!(
            gqa_decode_form_from_declared(Some(&[1, 5]), Some(&[1, 2, 4, 32]), Some(8), Some(2))
                .head_dim,
            None
        );
        // A packed width that does not divide by the head count is not a head width.
        assert_eq!(
            gqa_decode_form_from_declared(Some(&[1, 1, 385]), None, Some(8), Some(2)).head_dim,
            None,
            "385 / 12 is not exact, so the width is not recoverable"
        );
        // Zero or negative head counts cannot divide anything.
        for (nq, nkv) in [(0i64, 2i64), (8, 0), (-1, 2)] {
            assert_eq!(
                gqa_decode_form_from_declared(Some(&[1, 1, 384]), None, Some(nq), Some(nkv))
                    .head_dim,
                None,
                "num_heads {nq} / kv_num_heads {nkv} is not a divisor"
            );
        }
        // A stated zero extent is stated, and refused on its merits rather than treated as
        // unknown — the distinction `past_len_max`'s refusal already turns on.
        let empty = gqa_decode_form_from_declared(
            Some(&[1, 1, 384]),
            Some(&[1, 2, 0, 32]),
            Some(8),
            Some(2),
        );
        assert_eq!(empty.past_extent, Some(0));
        assert!(!gqa_decode_module_is_reachable(empty));
    }

    /// The host's lane ladder and the shader's must be the same ladder.
    ///
    /// `gqa_decode_f16.comp` recomputes the *effective* lane count from `seqlens_k[b]`, because
    /// the extent the host reads is an upper bound under the KV arena. That only works while both
    /// sides use the same rung width, and nothing else in the build compares a `.comp` constant
    /// with a Rust one — glslc would happily compile a shader that disagrees.
    #[test]
    fn the_shader_pins_the_min_kv_per_lane_constant() {
        let src = include_str!("../../shaders/glsl/gqa_decode_f16.comp");
        assert!(
            src.contains(&format!(
                "const uint MIN_KV_PER_LANE  = {}u;",
                GQA_DECODE_MIN_KV_PER_LANE
            )),
            "gqa_decode_f16.comp must declare MIN_KV_PER_LANE = {}",
            GQA_DECODE_MIN_KV_PER_LANE
        );
        assert!(
            src.contains(&format!(
                "const uint MAX_KV_LANES     = {}u;",
                GQA_DECODE_MAX_KV_LANES
            )),
            "gqa_decode_f16.comp must declare MAX_KV_LANES = {}",
            GQA_DECODE_MAX_KV_LANES
        );
        assert!(
            src.contains(&format!(
                "const uint MAX_HEAD_DIM     = {}u;",
                GQA_DECODE_MAX_HEAD_DIM
            )),
            "gqa_decode_f16.comp must declare MAX_HEAD_DIM = {}",
            GQA_DECODE_MAX_HEAD_DIM
        );
    }

    /// The declared shared-memory footprint fits the Vulkan 1.1 guaranteed floor.
    ///
    /// 16,384 B is `maxComputeSharedMemorySize`'s specification minimum. The module's allocation
    /// is a constant rather than a function of the lane count precisely so that this is a
    /// property of the *file*, checkable here, rather than of whichever pipeline happened to be
    /// created — see the header of `gqa_decode_f16.comp`.
    #[test]
    fn decode_shared_memory_fits_the_guaranteed_floor() {
        const VULKAN11_MIN_SHARED_BYTES: u32 = 16_384;
        const VULKAN11_MIN_WORKGROUP_INVOCATIONS: u32 = 128;
        let lanes = GQA_DECODE_MAX_KV_LANES;
        let bytes = 4 * lanes                          // s_max
            + 4 * lanes                                // s_sum
            + 4 * lanes * GQA_DECODE_MAX_HEAD_DIM; // s_acc
        assert_eq!(bytes, 8_320, "the number the module's header states");
        assert!(
            bytes < VULKAN11_MIN_SHARED_BYTES,
            "{bytes} B does not fit the {VULKAN11_MIN_SHARED_BYTES} B Vulkan 1.1 floor"
        );
        assert!(
            lanes <= VULKAN11_MIN_WORKGROUP_INVOCATIONS,
            "and the workgroup must fit maxComputeWorkGroupInvocations' floor of \
             {VULKAN11_MIN_WORKGROUP_INVOCATIONS}"
        );
        assert!(
            lanes.is_power_of_two(),
            "the merge is a halving tree, so the cap must be a power of two"
        );
    }

    /// The KV-parallel module is on disk under the stem the dispatch and the key both name.
    #[test]
    fn translate_gqa_decode_shader_stem_is_in_manifest() {
        use crate::engine::shaders;
        assert_eq!(GQA_DECODE_MODULE, "gqa_decode_f16");
        assert!(
            shaders::find(GQA_DECODE_MODULE).is_some(),
            "`{GQA_DECODE_MODULE}` is not in the compiled SPIR-V manifest — did `build.rs` miss \
             the shader?"
        );
    }

    /// The refusal path's dispatch is the shipping one, field for field.
    ///
    /// Not "it dispatches `gqa_f16`" — the grid, the specialisation constant and the whole
    /// 36-byte push block, compared against a prefill dispatch of the same handler, so that a
    /// change to the shared code between them cannot pass by moving both.
    #[test]
    fn translate_gqa_prefill_is_untouched_by_the_decode_selector() {
        let spec = crate::registry::all_specs()
            .find(|s| s.op_type == "GroupQueryAttention")
            .unwrap();
        // S = 128: prefill, which the selector refuses on `seq_len` alone even though the cache
        // is long enough for 16 lanes.
        let node = gqa_node(1, 128, 32, 32, 96, 1024);
        let mut ctx = Recorder::default();
        translate_gqa(spec, &node, &mut ctx).expect("translate should succeed");
        let k = &ctx.dispatches[0];
        let total = 32u32 * 128;
        let local = gqa_local_size(total);
        assert_eq!(
            k.shader, "gqa_f16",
            "prefill never reaches the decode module"
        );
        assert_eq!(k.spec_constants, vec![local]);
        assert_eq!(k.workgroups, [total.div_ceil(local), 1, 1]);
    }

    /// One workgroup per (batch, query head), across batch sizes and groupings.
    ///
    /// The grid is exact — `B * Nq` — which is what makes the module's `b >= batch_size` guard a
    /// backstop rather than a tail. A geometry that over-dispatched would put workgroups with no
    /// work through the whole barrier sequence.
    #[test]
    fn decode_grid_is_one_workgroup_per_batch_head() {
        let spec = crate::registry::all_specs()
            .find(|s| s.op_type == "GroupQueryAttention")
            .unwrap();
        for (b, nq, nkv) in [(1i64, 32i64, 32i64), (4, 32, 8), (2, 8, 1), (3, 16, 16)] {
            let node = gqa_node(b, 1, nq, nkv, 64, 1024);
            let mut ctx = Recorder::default();
            translate_gqa(spec, &node, &mut ctx).expect("translate should succeed");
            let k = &ctx.dispatches[0];
            assert_eq!(k.shader, GQA_DECODE_MODULE);
            assert_eq!(
                k.workgroups,
                [(b * nq) as u32, 1, 1],
                "B={b} Nq={nq}: the grid is exactly B*Nq workgroups"
            );
            assert_eq!(k.spec_constants, vec![16]);
        }
    }

    /// Defect 2 falsifier (R9/R10): with an **empty** past KV (`past_max == 0`, the prefill /
    /// growing-KV feed real graphs emit), the present outputs must be bound as real, correctly
    /// sized `[B, Nkv, seq_len, D]` buffers — not aliased onto the zero-element past.  Before the
    /// fix, present was aliased onto the empty past, `dispatch_ort` sized it 0 bytes, and the KV
    /// outputs were never written.  The instrument that goes red if the fix regresses: the KV
    /// output descriptors reappear at `S == past_max == 0`, or the shader's KV write stride
    /// (`past_len_max` push field) collapses back to 0.
    #[test]
    fn translate_gqa_empty_past_binds_real_present_buffers() {
        let spec = crate::registry::all_specs()
            .find(|s| s.op_type == "GroupQueryAttention")
            .unwrap();
        // B=1, S=2 (two prefill tokens), Nq=32, Nkv=32, D=96, past_max=0 (empty KV).
        let node = gqa_node(1, 2, 32, 32, 96, 0);
        let mut ctx = Recorder::default();
        translate_gqa(spec, &node, &mut ctx).expect("translate should succeed");

        // attn_output + present_key + present_value are all real allocations now.
        assert_eq!(
            ctx.outputs.len(),
            3,
            "empty past must bind attn + present_key + present_value as real buffers"
        );
        // Present KV descriptors are sized to the present sequence length (= seq_len for prefill),
        // never to the zero-element past.
        for desc in &ctx.outputs[1..] {
            assert_eq!(
                desc.shape,
                vec![1, 32, 2, 96],
                "present KV must be [B, Nkv, seq_len, D], not the empty past shape"
            );
        }
        // The shader's KV write stride (push field 6) must equal the present S-dim, not 0.
        let k = &ctx.dispatches[0];
        let kv_stride = u32::from_le_bytes(k.push_constants[24..28].try_into().unwrap());
        assert_eq!(
            kv_stride, 2,
            "KV write stride must be seq_len for an empty past"
        );
    }

    /// Defect A falsifier. `rotary_embedding_dim` is absent here and `head_dim != 2 *
    /// cos_cache.shape[1]`, so the two candidate sources disagree and the test can tell
    /// them apart. MEASURED 2026-08-02: the `group_query_attention_f16` evidence case is
    /// exactly this shape (head_dim 32, cos_cache [64, 8]); defaulting to head_dim made it
    /// DIVERGENT at worst_rel 16.726, because the shader strides cos/sin rows by
    /// `rotary_dim / 2` and a doubled stride reads another position's angles.
    #[test]
    fn translate_gqa_rotary_dim_follows_cos_cache_not_head_dim() {
        let spec = crate::registry::all_specs()
            .find(|s| s.op_type == "GroupQueryAttention")
            .unwrap();
        // head_dim 32, cos_cache [4096, 8] → true rotary_dim 16, half the head.
        let node = gqa_node_with_present(1, 1, 8, 2, 32, 4, 5, 8);
        let mut ctx = Recorder::default();
        translate_gqa(spec, &node, &mut ctx).expect("translate should succeed");
        let rotary_dim =
            u32::from_le_bytes(ctx.dispatches[0].push_constants[20..24].try_into().unwrap());
        assert_eq!(
            rotary_dim, 16,
            "rotary_dim must be 2 * cos_cache.shape[1], not head_dim"
        );
    }

    /// Defect A falsifier, refusal arm. With no attribute *and* no cos_cache shape the
    /// rotary width is not recoverable. Guessing head_dim is what produced the defect, and
    /// a handler that guesses can no longer report that the information was missing.
    #[test]
    fn translate_gqa_refuses_when_rotary_width_is_unrecoverable() {
        let spec = crate::registry::all_specs()
            .find(|s| s.op_type == "GroupQueryAttention")
            .unwrap();
        let mut node = gqa_node(1, 1, 8, 2, 32, 4);
        node.inputs[7].desc = None;
        let mut ctx = Recorder::default();
        let err = translate_gqa(spec, &node, &mut ctx)
            .expect_err("must refuse rather than assume head_dim");
        assert!(
            matches!(err, EpError::Unsupported(_)),
            "unrecoverable rotary width is Unsupported, got {err:?}"
        );
    }

    /// Defect B falsifier. Under the growing-cache convention `present` is a *different,
    /// larger* buffer than `past`, so it must be a fresh allocation of `past + S` and the
    /// shader's write stride must be that extent. MEASURED 2026-08-02: bound at the past
    /// extent, the shader's write at `tok_pos == past_len` fell outside the buffer and was
    /// dropped, so `present_key`/`present_value` read back all-zero on Vulkan while the CPU
    /// reference returned the concatenated cache.
    #[test]
    fn translate_gqa_growing_cache_binds_present_at_past_plus_seq() {
        let spec = crate::registry::all_specs()
            .find(|s| s.op_type == "GroupQueryAttention")
            .unwrap();
        // Evidence-case shape: past 4 + S 1 = present 5, with a non-empty past.
        let node = gqa_node(1, 1, 8, 2, 32, 4);
        let mut ctx = Recorder::default();
        translate_gqa(spec, &node, &mut ctx).expect("translate should succeed");

        assert_eq!(
            ctx.outputs.len(),
            3,
            "present is larger than past, so it cannot alias it"
        );
        for desc in &ctx.outputs[1..] {
            assert_eq!(
                desc.shape,
                vec![1, 2, 5, 32],
                "present must be [B, Nkv, past + S, D]"
            );
        }
        let pc = &ctx.dispatches[0].push_constants;
        assert_eq!(
            u32::from_le_bytes(pc[24..28].try_into().unwrap()),
            5,
            "KV write stride is the present extent"
        );
        assert_eq!(
            u32::from_le_bytes(pc[28..32].try_into().unwrap()),
            4,
            "past reads keep the past extent"
        );
    }

    /// The other arm: when the graph declares `present` at the *same* extent as `past`, it
    /// is the shared fixed-size buffer convention and must still alias. The two conventions
    /// are distinguished only by that declared number, which is why it is read rather than
    /// derived.
    #[test]
    fn translate_gqa_shared_buffer_cache_still_aliases_past() {
        let spec = crate::registry::all_specs()
            .find(|s| s.op_type == "GroupQueryAttention")
            .unwrap();
        let node = gqa_node_with_present(1, 1, 8, 2, 32, 256, 256, 16);
        let mut ctx = Recorder::default();
        translate_gqa(spec, &node, &mut ctx).expect("translate should succeed");

        assert_eq!(
            ctx.outputs.len(),
            1,
            "shared buffer: present aliases past, only attn is a new allocation"
        );
        let pc = &ctx.dispatches[0].push_constants;
        assert_eq!(
            u32::from_le_bytes(pc[24..28].try_into().unwrap()),
            u32::from_le_bytes(pc[28..32].try_into().unwrap()),
            "the two strides coincide exactly when the buffer is shared — and the shader \
             uses that equality to skip the past->present copy"
        );
    }

    /// A `Recorder` that declares the **growing prefix alias** convention.
    ///
    /// Like `ArenaRecorder`, the convention is stated on the context rather than through
    /// `std::env`, so these arms cannot race the flag tests in the same binary. It records the
    /// `(present_view, past_view, layout)` triples the engine would act on, which is the only
    /// place a test can see the relation the handler declared.
    #[derive(Default)]
    struct PrefixRecorder {
        inner: Recorder,
        prefix_binds: Vec<(u64, u64, crate::engine::PrefixLayout)>,
    }

    impl DispatchContext for PrefixRecorder {
        fn resolve(&mut self, r: &crate::engine::TensorRef) -> EpResult<BufferView> {
            self.inner.resolve(r)
        }
        fn bind_output(
            &mut self,
            o: &crate::engine::OutRef,
            desc: TensorDesc,
        ) -> EpResult<BufferView> {
            self.inner.bind_output(o, desc)
        }
        fn alloc_temp(&mut self, desc: TensorDesc) -> EpResult<BufferView> {
            self.inner.alloc_temp(desc)
        }
        fn dispatch(&mut self, k: KernelRequest) -> EpResult<()> {
            self.inner.dispatch(k)
        }
        fn read_const_i64(&self, r: &crate::engine::TensorRef) -> Option<Vec<i64>> {
            self.inner.read_const_i64(r)
        }
        fn kv_growing_alias(&self) -> bool {
            true
        }
        fn bind_prefix_output(
            &mut self,
            input: BufferView,
            out: &crate::engine::OutRef,
            desc: TensorDesc,
            layout: crate::engine::PrefixLayout,
        ) -> EpResult<BufferView> {
            let view = self.inner.bind_output(out, desc)?;
            self.prefix_binds
                .push((view.as_raw(), input.as_raw(), layout));
            Ok(view)
        }
    }

    /// The prefix alias on the shape the shipping lane actually dies on: a growing cache with a
    /// non-empty past.
    ///
    /// Three claims, and they are separate. (1) `present` is still bound at `past + S` — the
    /// alias changes where `past` *lives*, not how big `present` is. (2) the two push-constant
    /// strides now **coincide**, which is the condition `gqa_f16.comp` reads to skip its own
    /// past→present copy — correct here because that copy has already happened, as a
    /// `vkCmdCopyBuffer`. (3) the `past_*` bindings are the very same buffer as `present_*`, so
    /// the engine has no reason to allocate a device buffer for `past` at all. That third one is
    /// the defect: it is the allocation whose absence closes `device_memory_ctx4096_*`.
    #[test]
    fn translate_gqa_prefix_alias_stages_past_into_present_and_binds_one_buffer() {
        let spec = crate::registry::all_specs()
            .find(|s| s.op_type == "GroupQueryAttention")
            .unwrap();
        // past 4 + S 1 = present 5, B=1, Nkv=2, D=32, f16.
        let node = gqa_node(1, 1, 8, 2, 32, 4);
        let mut ctx = PrefixRecorder::default();
        translate_gqa(spec, &node, &mut ctx).expect("translate should succeed");

        for desc in &ctx.inner.outputs[1..] {
            assert_eq!(
                desc.shape,
                vec![1, 2, 5, 32],
                "the alias must not change present's extent"
            );
        }

        let pc = &ctx.inner.dispatches[0].push_constants;
        let present_len = u32::from_le_bytes(pc[24..28].try_into().unwrap());
        let past_stride = u32::from_le_bytes(pc[28..32].try_into().unwrap());
        assert_eq!(present_len, 5);
        assert_eq!(
            past_stride, present_len,
            "past now lives inside present at present's stride, so the reads must use it — and \
             the equality is exactly what switches the shader's own copy off"
        );

        assert_eq!(ctx.prefix_binds.len(), 2, "one for K, one for V");
        let bindings = &ctx.inner.dispatches[0].bindings;
        for (out_view, _, layout) in &ctx.prefix_binds {
            // B*Nkv = 2 blocks of past_len*D*2 bytes, landing at present_len*D*2.
            assert_eq!(layout.outer_blocks, 2);
            assert_eq!(layout.src_block_bytes, 4 * 32 * 2);
            assert_eq!(layout.dst_block_bytes, 5 * 32 * 2);
            assert!(
                layout.fits(2 * 4 * 32 * 2, 2 * 5 * 32 * 2),
                "the geometry the handler declares must pass the engine's own refusal check; if \
                 these two ever disagree the engine bails and the lane stops, which is the \
                 failure mode this assertion exists to catch early"
            );
            assert!(
                bindings.iter().any(|b| b.as_raw() == *out_view),
                "the present buffer must be bound to the dispatch"
            );
        }
        // binding 1/2 are past_key/past_value, 7/8 are present_key/present_value.
        assert_eq!(
            (bindings[1].as_raw(), bindings[2].as_raw()),
            (bindings[7].as_raw(), bindings[8].as_raw()),
            "past and present must be the *same* buffer under the prefix alias — that identity \
             is the removed allocation"
        );
    }

    /// Prefill: `past_len == 0`, so there is no prefix to stage and nothing to alias.
    ///
    /// This is a separating case rather than a corner case. A zero-block `PrefixLayout` would be
    /// refused by the engine and stop the lane, so the handler must decline *before* declaring
    /// one. Prefill and decode take different branches here and only a run can tell them apart.
    #[test]
    fn translate_gqa_prefix_alias_declines_an_empty_past() {
        let spec = crate::registry::all_specs()
            .find(|s| s.op_type == "GroupQueryAttention")
            .unwrap();
        let node = gqa_node(1, 4, 8, 2, 32, 0);
        let mut ctx = PrefixRecorder::default();
        translate_gqa(spec, &node, &mut ctx).expect("translate should succeed");

        assert!(
            ctx.prefix_binds.is_empty(),
            "with no past tokens there is nothing to stage, so no alias may be declared"
        );
        let pc = &ctx.inner.dispatches[0].push_constants;
        assert_eq!(
            u32::from_le_bytes(pc[28..32].try_into().unwrap()),
            0,
            "prefill keeps the past stride at the past extent, which is zero"
        );
    }

    /// The shared-buffer (fixed-capacity) convention outranks the prefix alias.
    ///
    /// When `present` and `past` are the same extent the buffer is already shared and there is
    /// no prefix to stage: staging one would copy a buffer onto itself. The predicate excludes
    /// it, and this asserts the exclusion rather than trusting the ordering of two `if`s.
    #[test]
    fn translate_gqa_prefix_alias_defers_to_the_shared_buffer_convention() {
        let spec = crate::registry::all_specs()
            .find(|s| s.op_type == "GroupQueryAttention")
            .unwrap();
        let node = gqa_node_with_present(1, 1, 8, 2, 32, 256, 256, 16);
        let mut ctx = PrefixRecorder::default();
        translate_gqa(spec, &node, &mut ctx).expect("translate should succeed");

        assert!(
            ctx.prefix_binds.is_empty(),
            "a shared fixed-capacity cache has no prefix to stage"
        );
        assert_eq!(
            ctx.inner.outputs.len(),
            1,
            "and it still aliases present onto past the old way"
        );
    }

    /// A `Recorder` that declares the shared-buffer (arena) convention.
    ///
    /// The convention is on the context, not in the environment, so a test can state it
    /// without a process-wide variable — and so two arms can run in the same test binary
    /// without racing each other through `std::env`.
    #[derive(Default)]
    struct ArenaRecorder(Recorder);

    impl DispatchContext for ArenaRecorder {
        fn resolve(&mut self, r: &crate::engine::TensorRef) -> EpResult<BufferView> {
            self.0.resolve(r)
        }
        fn bind_output(
            &mut self,
            o: &crate::engine::OutRef,
            desc: TensorDesc,
        ) -> EpResult<BufferView> {
            self.0.bind_output(o, desc)
        }
        fn alloc_temp(&mut self, desc: TensorDesc) -> EpResult<BufferView> {
            self.0.alloc_temp(desc)
        }
        fn dispatch(&mut self, k: KernelRequest) -> EpResult<()> {
            self.0.dispatch(k)
        }
        fn read_const_i64(&self, r: &crate::engine::TensorRef) -> Option<Vec<i64>> {
            self.0.read_const_i64(r)
        }
        fn kv_arena(&self) -> bool {
            true
        }
    }

    /// Build a GQA node whose `present` extent is **unstated**, which is what Phi-3.5's
    /// symbolic `total_sequence_length` produces after `tensor_desc` drops it.
    fn gqa_node_symbolic_present(past_max: i64) -> NodeDesc {
        let mut node = gqa_node(1, 1, 8, 2, 32, past_max);
        for out in node.outputs.iter_mut().skip(1) {
            out.desc = None;
        }
        node
    }

    /// The arena: with the convention declared and the present extent unstated, `present`
    /// aliases `past` and the two strides coincide — which is the exact condition
    /// `gqa_f16.comp` reads to skip the past→present copy.
    #[test]
    fn translate_gqa_arena_aliases_present_onto_past() {
        let spec = crate::registry::all_specs()
            .find(|s| s.op_type == "GroupQueryAttention")
            .unwrap();
        let node = gqa_node_symbolic_present(8192);
        let mut ctx = ArenaRecorder::default();
        translate_gqa(spec, &node, &mut ctx).expect("translate should succeed");

        assert_eq!(
            ctx.0.outputs.len(),
            1,
            "arena: present aliases past, so attn is the only new allocation"
        );
        let pc = &ctx.0.dispatches[0].push_constants;
        let present_len = u32::from_le_bytes(pc[24..28].try_into().unwrap());
        let past_stride = u32::from_le_bytes(pc[28..32].try_into().unwrap());
        assert_eq!(present_len, 8192, "the arena extent is the write stride");
        assert_eq!(
            present_len, past_stride,
            "one stride, or the write region of head h lands inside the read region of head h+1"
        );
    }

    /// Same node, same shapes, **without** the declaration: the shipping path is untouched.
    ///
    /// This is the control that says the previous test measured the flag and not the shapes.
    #[test]
    fn translate_gqa_without_the_declaration_still_grows() {
        let spec = crate::registry::all_specs()
            .find(|s| s.op_type == "GroupQueryAttention")
            .unwrap();
        let node = gqa_node_symbolic_present(8192);
        let mut ctx = Recorder::default();
        translate_gqa(spec, &node, &mut ctx).expect("translate should succeed");

        assert_eq!(
            ctx.outputs.len(),
            3,
            "growing: present is its own allocation"
        );
        let pc = &ctx.dispatches[0].push_constants;
        assert_eq!(
            u32::from_le_bytes(pc[24..28].try_into().unwrap()),
            8193,
            "growing cache derives present as past + seq_len"
        );
    }

    /// A graph that **states** a growing `present` has asked for the growing convention in
    /// writing, and the caller's flag does not overrule it.
    ///
    /// This matters because the two graphs this project runs disagree: the evidence case
    /// `group_query_attention_f16` declares `[B,2,4,32]` against `[B,2,5,32]`, and Phi-3.5
    /// declares symbols. A flag that ignored the declaration would silently write a 5-token
    /// layout into a 4-token buffer on the case every correctness gate uses.
    #[test]
    fn translate_gqa_a_declared_growing_present_outranks_the_arena_flag() {
        let spec = crate::registry::all_specs()
            .find(|s| s.op_type == "GroupQueryAttention")
            .unwrap();
        let node = gqa_node(1, 1, 8, 2, 32, 4); // declares present = 5
        let mut ctx = ArenaRecorder::default();
        translate_gqa(spec, &node, &mut ctx).expect("translate should succeed");

        assert_eq!(
            ctx.0.outputs.len(),
            3,
            "the declaration wins: present is still its own allocation"
        );
        let pc = &ctx.0.dispatches[0].push_constants;
        assert_eq!(u32::from_le_bytes(pc[24..28].try_into().unwrap()), 5);
        assert_eq!(u32::from_le_bytes(pc[28..32].try_into().unwrap()), 4);
    }

    /// The arena needs capacity for the tokens this step writes. `past_len_max < seq_len`
    /// cannot hold them under any past length, so the alias is declined rather than taken
    /// and clamped — a clamped write is the dropped-write defect this project has already
    /// shipped once and read back as zeros.
    #[test]
    fn translate_gqa_arena_declines_a_capacity_smaller_than_one_step() {
        let spec = crate::registry::all_specs()
            .find(|s| s.op_type == "GroupQueryAttention")
            .unwrap();
        let mut node = gqa_node_symbolic_present(2);
        // seq_len 8 against an arena of 2.
        node.inputs[0].desc = Some(TensorDesc::new(DType::F16, vec![1, 8, (8 + 2 * 2) * 32]));
        let mut ctx = ArenaRecorder::default();
        translate_gqa(spec, &node, &mut ctx).expect("translate should succeed");

        assert_eq!(
            ctx.0.outputs.len(),
            3,
            "capacity below one step's worth of tokens is not an arena"
        );
        let pc = &ctx.0.dispatches[0].push_constants;
        assert_eq!(u32::from_le_bytes(pc[24..28].try_into().unwrap()), 10);
    }

    #[test]
    fn translate_gqa_shader_stem_is_in_manifest() {
        // Prove the stem the translate handler emits exists on disk (pattern (b) from registry.rs).
        // This is the companion to the `no_live_row_lacks_a_shader_or_dispatch_path` registry test.
        use crate::engine::shaders;
        assert!(
            shaders::find("gqa_f16").is_some(),
            "`gqa_f16` is not in the compiled SPIR-V manifest — did `build.rs` miss the shader?"
        );
    }
}
