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

use crate::engine::{AttrValue, DType, DispatchContext, EpError, EpResult, KernelRequest, NodeDesc, TensorDesc};
use crate::kernel;
use crate::ops::common::claim::{self, ClaimResult};
use crate::ops::common::dtype::{ANY, FLOAT};
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
fn translate_gqa(
    _spec: &OpSpec,
    node: &NodeDesc,
    ctx: &mut dyn DispatchContext,
) -> EpResult<()> {
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
    let qkv_desc = node.inputs[0]
        .desc
        .as_ref()
        .ok_or_else(|| EpError::Unsupported(format!("`{}` packed_qkv has no shape", node.op_type)))?;
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
    let seq_len    = qkv_desc.shape[1] as u32;
    let packed_dim = qkv_desc.shape[2] as u32;

    let num_heads    = attr_i64("num_heads")? as u32;
    let kv_num_heads = attr_i64("kv_num_heads")? as u32;
    let head_dim     = packed_dim / (num_heads + 2 * kv_num_heads);

    // `rotary_embedding_dim` is optional; default to full head_dim (Phi-3.5 pattern).
    let rotary_dim = match node.attributes.get("rotary_embedding_dim") {
        Some(AttrValue::Int(v)) => *v as u32,
        _                       => head_dim,
    };

    // `scale` is optional; default to 1/sqrt(head_dim).
    let scale = attr_f32_opt("scale").unwrap_or_else(|| (head_dim as f32).sqrt().recip());

    // past_len_max from past_key shape[2] (buffer S dimension).
    let past_len_max = node.inputs[3]
        .desc
        .as_ref()
        .and_then(|d| d.shape.get(2).copied())
        .unwrap_or(0) as u32;

    // -- Resolve input buffers ----------------------------------------------------------
    let qkv_buf     = ctx.resolve(&node.inputs[0])?;
    let past_k_buf  = ctx.resolve(&node.inputs[3])?;
    let past_v_buf  = ctx.resolve(&node.inputs[4])?;
    let seqlens_buf = ctx.resolve(&node.inputs[5])?;
    let cos_buf     = ctx.resolve(&node.inputs[7])?;
    let sin_buf     = ctx.resolve(&node.inputs[8])?;

    // -- Bind outputs ------------------------------------------------------------------
    // attn_output: [B, S, Nq*D]
    let attn_out_ref = node.outputs.first().ok_or_else(|| {
        EpError::InvalidGraph(format!("`{}` has no attn_output slot", node.op_type))
    })?;
    let attn_buf = ctx.bind_output(
        attn_out_ref,
        TensorDesc::new(dtype, vec![batch_size as i64, seq_len as i64, (num_heads * head_dim) as i64]),
    )?;

    // present_key / present_value alias past_key / past_value (in-place KV cache update).
    // The shader writes only the new token at tok_pos; all other positions are inherited from
    // the past buffer, which is the same allocation.  This avoids a full-cache copy per step.
    // M2's device-backed allocator must honour the alias (see OP_COVERAGE.md §9.5 #3).
    let pres_k_ref = node.outputs.get(1).ok_or_else(|| {
        EpError::InvalidGraph(format!("`{}` has no present_key slot", node.op_type))
    })?;
    let pres_k_buf = ctx.bind_aliased_output(&node.inputs[3], pres_k_ref)?;

    let pres_v_ref = node.outputs.get(2).ok_or_else(|| {
        EpError::InvalidGraph(format!("`{}` has no present_value slot", node.op_type))
    })?;
    let pres_v_buf = ctx.bind_aliased_output(&node.inputs[4], pres_v_ref)?;

    // -- Push constants (32 bytes, matches shader PC struct) ---------------------------
    let mut push = Vec::with_capacity(32);
    push.extend_from_slice(&batch_size.to_le_bytes());
    push.extend_from_slice(&seq_len.to_le_bytes());
    push.extend_from_slice(&num_heads.to_le_bytes());
    push.extend_from_slice(&kv_num_heads.to_le_bytes());
    push.extend_from_slice(&head_dim.to_le_bytes());
    push.extend_from_slice(&rotary_dim.to_le_bytes());
    push.extend_from_slice(&past_len_max.to_le_bytes());
    push.extend_from_slice(&scale.to_bits().to_le_bytes());

    // -- Dispatch: one invocation per (batch, query_head, query_seq_pos) ---------------
    let total = (batch_size * num_heads * seq_len).max(1);
    ctx.dispatch(KernelRequest {
        shader: "gqa_f16",
        spec_constants: vec![],
        push_constants: push,
        bindings: vec![
            qkv_buf,    // binding 0: packed_qkv
            past_k_buf, // binding 1: past_key
            past_v_buf, // binding 2: past_value
            seqlens_buf, // binding 3: seqlens_k
            cos_buf,    // binding 4: cos_cache
            sin_buf,    // binding 5: sin_cache
            attn_buf,   // binding 6: attn_output
            pres_k_buf, // binding 7: present_key
            pres_v_buf, // binding 8: present_value
        ],
        workgroups: [total, 1, 1],
    })
}

crate::op_table! {
    //  op                     domain  opsets                       caps    kernel          claim                   translate                  status              schema
    "GroupQueryAttention",     Ms,     1 ..= OPSET_ANY,             FLOAT,  kernel!(None),  group_query_attention,  translate_gqa,             Live,               schema: &GROUP_QUERY_ATTENTION;
    "RotaryEmbedding",         Ms,     1 ..= OPSET_ANY,             FLOAT,  kernel!(None),  rotary_embedding,       templates::unimplemented,  Staged(XL_KERNEL),  schema: &ROTARY_EMBEDDING;
    "MultiHeadAttention",      Ms,     1 ..= OPSET_ANY,             FLOAT,  kernel!(None),  claim::never,           templates::unimplemented,  Staged(XL_KERNEL),  schema: &MULTI_HEAD_ATTENTION;
    "Attention",               Ai,     OPSET_STD_LLM ..= OPSET_STD_ATTENTION_MAX, FLOAT,  kernel!(None),  std_attention,          templates::unimplemented,  Staged(XL_KERNEL);
    "RotaryEmbedding",         Ai,     OPSET_STD_LLM ..= OPSET_STD_NORM_MAX,      FLOAT,  kernel!(None),  std_rotary_embedding,   templates::unimplemented,  Staged(XL_KERNEL);
    "TensorScatter",           Ai,     OPSET_STD_TENSOR_SCATTER ..= OPSET_STD_TENSOR_SCATTER, ANY, kernel!(None), tensor_scatter, templates::unimplemented, Staged(NO_SHADER);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{BufferView, DispatchContext, EpResult, KernelRequest, NodeDesc, OutRef, TensorDesc, TensorRef};
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
        assert_eq!(live.len(), 1, "only GroupQueryAttention should be live in this module");
        assert_eq!(live[0].op_type, "GroupQueryAttention");
        for s in OPS {
            if s.op_type != "GroupQueryAttention" {
                assert!(matches!(s.status, OpStatus::Staged(_)), "{} should still be staged", s.op_type);
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
        fn bind_output(&mut self, _o: &crate::engine::OutRef, desc: TensorDesc) -> EpResult<BufferView> {
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
        // packed_qkv: [B, S, (Nq+2*Nkv)*D]
        let qkv_dim = (nq + 2 * nkv) * d;
        // cos_cache/sin_cache: [max_pos, d/2] — shape doesn't matter for translate
        let rot_half = d / 2;
        let mut attrs = std::collections::BTreeMap::new();
        attrs.insert("num_heads".into(), AttrValue::Int(nq));
        attrs.insert("kv_num_heads".into(), AttrValue::Int(nkv));
        attrs.insert("scale".into(), AttrValue::Float((d as f32).sqrt().recip()));
        let make_ref = |name: &str, shape: Vec<i64>| TensorRef {
            name: name.into(),
            desc: Some(TensorDesc::new(DType::F16, shape)),
            is_initializer: false,
        };
        let empty_ref = || TensorRef { name: String::new(), desc: None, is_initializer: false };
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
                make_ref("qkv",     vec![b, s, qkv_dim]),   // 0 packed_qkv
                empty_ref(),                                  // 1 absent
                empty_ref(),                                  // 2 absent
                make_ref("past_k",  vec![b, nkv, past_max, d]), // 3 past_key
                make_ref("past_v",  vec![b, nkv, past_max, d]), // 4 past_value
                seqlens_ref,                                  // 5 seqlens_k
                empty_ref(),                                  // 6 total_seq (optional)
                make_ref("cos",     vec![4096, rot_half]),    // 7 cos_cache
                make_ref("sin",     vec![4096, rot_half]),    // 8 sin_cache
            ],
            outputs: vec![
                make_out("attn",   vec![b, s, nq * d]),
                make_out("pres_k", vec![b, nkv, past_max, d]),
                make_out("pres_v", vec![b, nkv, past_max, d]),
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
        assert_eq!(k.shader, "gqa_f16", "must dispatch gqa_f16 shader");
        assert_eq!(k.workgroups, [32, 1, 1], "B*Nq*S = 1*32*1 = 32 invocations");
        assert_eq!(k.bindings.len(), 9, "9 bindings: 6 inputs + 3 outputs");
        // Only attn_output calls bind_output; present_key/value use bind_aliased_output (→ resolve)
        assert_eq!(ctx.outputs.len(), 1, "only attn_output has a new allocation");

        // Verify push constants byte layout (32 bytes = 8 × u32/f32)
        let pc = &k.push_constants;
        assert_eq!(pc.len(), 32, "push constant block must be 32 bytes");
        let batch_size  = u32::from_le_bytes(pc[0..4].try_into().unwrap());
        let seq_len     = u32::from_le_bytes(pc[4..8].try_into().unwrap());
        let num_heads   = u32::from_le_bytes(pc[8..12].try_into().unwrap());
        let kv_num_heads = u32::from_le_bytes(pc[12..16].try_into().unwrap());
        let head_dim    = u32::from_le_bytes(pc[16..20].try_into().unwrap());
        let rotary_dim  = u32::from_le_bytes(pc[20..24].try_into().unwrap());
        let past_len_max = u32::from_le_bytes(pc[24..28].try_into().unwrap());
        let scale_bits  = u32::from_le_bytes(pc[28..32].try_into().unwrap());
        let scale       = f32::from_bits(scale_bits);

        assert_eq!(batch_size, 1);
        assert_eq!(seq_len, 1);
        assert_eq!(num_heads, 32);
        assert_eq!(kv_num_heads, 32);
        assert_eq!(head_dim, 96);
        assert_eq!(rotary_dim, 96, "default rotary_dim == head_dim");
        assert_eq!(past_len_max, 256);
        assert!((scale - 96f32.sqrt().recip()).abs() < 1e-6, "scale = 1/sqrt(D)");
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

