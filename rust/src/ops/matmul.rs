//! Dense linear algebra — `Gemm` and `MatMul`, the classifier head and the transformer body.
//!
//! # Why this module exists
//!
//! Two censuses, not a taxonomy:
//!
//! * MobileNetV2-12, 2026-08-04: with `Conv` and `GlobalAveragePool` claimed, the single `Gemm`
//!   at the tail is the last node in the model that carries data. Its `B` operand is a resident
//!   weight initializer at the schema-designated weight site, so it is an *anchor* in
//!   `ops::partition::is_anchor` and a lone `Gemm` island is exempt from the minimum-node gate —
//!   the row was already written into the partitioner's cost model before any kernel existed. An
//!   activation-only `Gemm` (no resident weight) would not anchor; the classifier head always
//!   carries one.
//! * The registry had no `Gemm` at all, which is why every non-LLM model this project has looked
//!   at ends on the CPU regardless of how much of its body the EP claims.
//! * BERT-SQuAD-12, 2026-08-04: `MatMul` ×95 is the largest unregistered op on any censused
//!   graph, and every attention projection and feed-forward layer in the model is one.
//!
//! # `MatMul` and `Gemm` are one module and two rows
//!
//! `MatMul` dispatches `gemm_f32`. A rank-N `A` against a rank-2 `B` is `[M, K] × [K, N]` with
//! `M` the product of `A`'s leading axes, and because a row-major buffer does not change when
//! leading axes are merged, that collapse copies nothing. With `alpha=1`, `beta=0`, `has_c=0` and
//! both transposes clear, `gemm_f32` computes ONNX `MatMul` exactly. Writing a second module
//! would have added a variant component that distinguishes nothing, which is what §8.9.23 ruled
//! against and what `form.rs` was deleted for. The two rows carry two proof keys over one module.
//!
//! # What `MatMul` shares with `MatMulNBits`, and what it does not
//!
//! A reader will assume more sharing than there is, so, plainly. They share the *name* and the
//! inner-product-over-`K` idea. They share nothing else:
//!
//! * **Operand format.** `MatMulNBits` reads `B` **block-quantised to 4 bits**, packed two values
//!   per byte with a per-block `scales` tensor and an optional `zero_points` tensor. `MatMul`'s
//!   `B` is dense f32, one value per word. The dequantisation is most of `q_gemv.comp`.
//! * **Shape regime.** `q_gemv.comp` is a **GEMV** — it is written for `M = 1`, the single-token
//!   decode step of an autoregressive LLM, and parallelises over `N` with a per-row reduction.
//!   BERT's `MatMul` has `M` in the thousands and parallelises over `M × N`. A GEMV kernel run at
//!   `M = 3072` would serialise the whole model.
//! * **Provenance.** `MatMulNBits` is a `com.microsoft` contrib op with its own schema, its own
//!   `accuracy_level` attribute and its own opset line. `MatMul` is `ai.onnx` opset 1.
//!
//! There is no shared module, no shared key and no shared claim predicate, and `MatMulNBits`
//! existing says nothing about whether `MatMul` is claimable on any model.
//!
//! # What is claimed, and what is declined by name
//!
//! Claimed: f32 `Gemm` with `A` rank 2, `B` rank 2, either or both transposed, optional `C`
//! unidirectionally broadcast from rank 0, 1 or 2, any `alpha` and `beta`. `A`'s row extent (the
//! batch) may be symbolic; nothing else may.
//!
//! * **f16** — the same packed-`uint` argument as `conv.rs` and `pooling.rs`. Declines `[dtype]`.
//! * **`B` with a symbolic extent** — `B` is an initializer on every graph we have censused, and
//!   the inner-product length is the loop bound.
//! * **rank != 2** — ONNX `Gemm` is rank 2 by schema; a rank-3 "batched Gemm" is `MatMul`, which
//!   is the other row in this module.
//!
//! For `MatMul`, claimed: f32 `A` of rank >= 2 against a **fully static rank-2** `B`, with `A`'s
//! leading extents allowed to be symbolic under the runtime-extent rule. Declined by name:
//!
//! * **a rank-1 operand** — ONNX promotes it and then removes the axis again from the output, so
//!   the output rank differs from what this row's output-shape rule produces.
//! * **a rank >= 3 `B`** — the leading axes broadcast against `A`'s, which is batch indexing on
//!   *both* operands. That is a different traversal, not a different push constant, and it needs
//!   its own module and its own key. **Measured: 24 of BERT's 95 `MatMul` nodes are this form**
//!   (the attention `QKᵀ` and `AV` products), so it is a real gap and not a hypothetical one.
//! * **a symbolic `K` or `N`** — the inner-product length and the output row stride are the
//!   kernel's loop bound and its indexing arithmetic.
//! * **f16** — the same packed-`uint` argument as `Gemm`.
//! * **a `C` whose extents are symbolic** — the broadcast rule needs to know which of `C`'s axes
//!   are 1, and an axis whose extent is unknown could be either.
//!
//! # `transA`/`transB` are a blind axis, not a key component and not a selector
//!
//! They change which index strides and therefore what the kernel computes, but one module serves
//! all four combinations from a ternary on a push constant — one pipeline, one set of emitted
//! instructions. Under §8.7 that makes them **expressions, not paths**, and §8.9.23 rules that an
//! expression is not a proof-key component however much it changes the answer. They were a key
//! component for one round, on the worry that a `Gemm` proven at `transB=1` would silently grant a
//! claim to a `Gemm` at `transB=0` — a transposed answer of the right shape, which is the worst
//! failure because it is plausible. That worry is real; it is discharged by `blind_axes` on the
//! row, which prints the caveat on the claim line, and by the CI-time suite that varies the
//! transposes. It is not discharged by asserting a code-path distinction that does not exist.
//!
//! `alpha` and `beta` are blind for the same reason and were never in the key: they are multiplied
//! into one expression, so every value runs the same instructions; they ride the push-constant
//! tail, which is what `ops::common::params` is for.

use crate::engine::{
    AttrValue, DType, DispatchContext, EpError, EpResult, KernelRequest, NodeDesc, TensorDesc,
};
use crate::kernel;
use crate::ops::common::claim::{self, ClaimResult};
use crate::ops::common::dtype::F32;
use crate::registry::OpStatus::Ready;
use crate::registry::{NodeView, OPSET_ANY, OpSpec};
use crate::require;

/// Workgroup size, matching every other 1-D grid in this crate.
pub(crate) const GEMM_LOCAL_SIZE: u32 = 256;

/// Cap on dispatched workgroups; the shader is a grid-stride loop.
pub(crate) const GEMM_MAX_WORKGROUPS: u32 = 65_535;

/// How `C` broadcasts onto the `[M, N]` output, as the two extents the shader indexes with.
///
/// `1` on an axis means "read element 0 of that axis for every row/column", which is ONNX's
/// unidirectional broadcast expressed as the only two numbers the kernel needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CBroadcast {
    pub rows: i64,
    pub cols: i64,
}

/// Resolve `C`'s shape against the output `[M, N]`, or say why it cannot broadcast.
///
/// Shared by the claim predicate and the translate handler, because the two disagreeing about
/// what a broadcast is would be the claim-then-fail shape `conv_attrs` exists to prevent.
pub(crate) fn c_broadcast(c_shape: &[i64], m: i64, n: i64) -> Result<CBroadcast, String> {
    // ONNX broadcasts `C` unidirectionally to `[M, N]`: align to the right, each axis must be 1
    // or equal to the target. A scalar and a rank-1 `[N]` are the two forms real graphs emit.
    let (rows, cols) = match c_shape.len() {
        0 => (1, 1),
        1 => (1, c_shape[0]),
        2 => (c_shape[0], c_shape[1]),
        r => {
            return Err(format!(
                "input 2 (C) has rank {r}; `Gemm` broadcasts C from rank 0, 1 or 2 onto [M, N]"
            ));
        }
    };
    if rows != 1 && rows != m {
        return Err(format!(
            "input 2 (C) has {rows} rows; a unidirectional broadcast onto [{m}, {n}] needs 1 or {m}"
        ));
    }
    if cols != 1 && cols != n {
        return Err(format!(
            "input 2 (C) has {cols} columns; a unidirectional broadcast onto [{m}, {n}] needs 1 \
             or {n}"
        ));
    }
    if rows < 1 || cols < 1 {
        return Err(format!("input 2 (C) has a non-positive extent {c_shape:?}"));
    }
    Ok(CBroadcast { rows, cols })
}

/// `[M, K]` from `A`'s declared shape and `transA`.
pub(crate) fn a_extents(shape: &[i64], trans: bool) -> (i64, i64) {
    if trans {
        (shape[1], shape[0])
    } else {
        (shape[0], shape[1])
    }
}

/// `[K, N]` from `B`'s declared shape and `transB`.
pub(crate) fn b_extents(shape: &[i64], trans: bool) -> (i64, i64) {
    if trans {
        (shape[1], shape[0])
    } else {
        (shape[0], shape[1])
    }
}

/// `Gemm` — claim rank-2 f32 with an optionally broadcast `C`.
fn gemm(view: &NodeView<'_>, spec: &OpSpec) -> ClaimResult {
    let a = claim::input_edge(view, spec, 0)?;
    claim::check_dtype(spec, &a, "input 0 (A)")?;
    claim::check_shape(spec, &a, "input 0 (A)")?;
    let b = claim::input_edge(view, spec, 1)?;
    claim::check_dtype(spec, &b, "input 1 (B)")?;
    claim::check_shape(spec, &b, "input 1 (B)")?;

    require!(
        (2..=3).contains(&view.num_inputs()),
        Arity,
        "`{}` has {} inputs; it takes A, B and an optional C",
        spec.op_type,
        view.num_inputs()
    );
    require!(
        a.rank() == Some(2),
        Rank,
        "`{}` input 0 (A) has rank {:?}; ONNX `Gemm` is rank 2 and a batched product is `MatMul`",
        spec.op_type,
        a.rank()
    );
    require!(
        b.rank() == Some(2),
        Rank,
        "`{}` input 1 (B) has rank {:?}; ONNX `Gemm` is rank 2",
        spec.op_type,
        b.rank()
    );

    let trans_a = view.attr_int("transA").unwrap_or(0) != 0;
    let trans_b = view.attr_int("transB").unwrap_or(0) != 0;

    let b_shape = b.shape.as_deref().unwrap_or(&[]);
    require!(
        b_shape.len() == 2 && b_shape[0] > 0 && b_shape[1] > 0,
        DynamicShape,
        "`{}` input 1 (B) has a symbolic extent ({b_shape:?}); the inner-product length is the \
         kernel's loop bound",
        spec.op_type
    );

    // Only `A`'s *row* extent — the batch — may be symbolic. `K` is the loop bound and must
    // agree with `B`, which cannot be checked against an unknown. Same rule, same reason, as
    // `conv.rs`: the one extent that does not enter the arithmetic is the one that may float.
    let a_shape = a.shape.as_deref().unwrap_or(&[]);
    let a_inner_axis = usize::from(!trans_a);
    require!(
        a_shape.len() == 2 && a_shape[a_inner_axis] > 0,
        DynamicShape,
        "`{}` input 0 (A) has a symbolic inner extent ({a_shape:?}, transA={trans_a}); only the \
         row extent may be symbolic",
        spec.op_type
    );

    let (m, k) = a_extents(a_shape, trans_a);
    let (kb, n) = b_extents(b_shape, trans_b);
    require!(
        k == kb,
        Shape,
        "`{}` A contributes K={k} (transA={trans_a}) and B contributes K={kb} (transB={trans_b})",
        spec.op_type
    );

    if view.num_inputs() == 3 && view.has_input(2) {
        let c = claim::input_edge(view, spec, 2)?;
        claim::check_dtype(spec, &c, "input 2 (C)")?;
        let c_shape = c.shape.as_deref().unwrap_or(&[]);
        require!(
            c.rank().is_some_and(|r| r <= 2) && c_shape.iter().all(|&d| d > 0),
            DynamicShape,
            "`{}` input 2 (C) has rank {:?} / shape {c_shape:?}; the broadcast rule needs to know \
             which axes are 1 and a symbolic axis could be either",
            spec.op_type,
            c.rank()
        );
        // `m` may be symbolic (<= 0). A `C` that claims to broadcast against a symbolic row
        // extent is declined rather than assumed to match: guessing here is how a `[batch, N]`
        // bias silently becomes a `[1, N]` one.
        if let Err(why) = c_broadcast(c_shape, m, n) {
            crate::deny!(Shape, "`{}` {why}", spec.op_type);
        }
    }
    Ok(())
}

/// Collapse `A`'s leading axes into the GEMM's `M`, or say why this `MatMul` is not that shape.
///
/// ONNX `MatMul` is NumPy `matmul`: it admits rank 1 on either side (with promotion rules that
/// differ per side), and rank >= 3 on *both* sides with the leading axes broadcast against each
/// other. This function implements exactly one of those cases — `A` of rank >= 2 against a rank-2
/// `B` — and returns `Err` for the rest, by name, so the decline sentence says which.
///
/// The case it implements is the one that reduces to a plain `[M, K] x [K, N]` product **with no
/// change to the memory layout at all**: a row-major `[d0, d1, ..., K]` buffer is already a
/// row-major `[d0*d1*..., K]` buffer, so the collapse is an index reinterpretation and not a copy.
/// That is what makes it the same kernel as `Gemm` rather than a kernel that resembles it.
pub(crate) fn matmul_2d_extents(a: &[i64], b: &[i64]) -> Result<(i64, i64, i64), String> {
    if b.len() != 2 {
        return Err(format!(
            "input 1 (B) has rank {}; this row implements `A` of rank >= 2 against a rank-2 `B`. \
             A rank-1 `B` is ONNX's vector-promotion form and a rank >= 3 `B` broadcasts its \
             leading axes against `A`'s, which is batch indexing on both operands — a different \
             traversal, not a different push constant",
            b.len()
        ));
    }
    if a.len() < 2 {
        return Err(format!(
            "input 0 (A) has rank {}; ONNX promotes a rank-1 `A` to `[1, K]` and then *removes* \
             the axis again from the output, so the output rank differs from what this row's \
             output-shape rule produces",
            a.len()
        ));
    }
    let k = a[a.len() - 1];
    if k != b[0] {
        return Err(format!(
            "input 0 (A) contributes K={k} and input 1 (B) contributes K={}",
            b[0]
        ));
    }
    // The leading axes multiply into `M`. `checked_mul` rather than a plain product: the extents
    // are `i64` off the graph and a bogus one must decline rather than wrap into a small,
    // plausible-looking row count that would dispatch a fraction of the work and return a buffer
    // that is mostly whatever was there before.
    let mut m: i64 = 1;
    for &d in &a[..a.len() - 1] {
        if d < 0 {
            return Err(format!(
                "input 0 (A) has a symbolic leading extent ({a:?}); the collapsed row count is \
                 their product and a product with an unknown factor is unknown"
            ));
        }
        m = match m.checked_mul(d) {
            Some(v) => v,
            None => return Err(format!("input 0 (A) leading extents {a:?} overflow i64")),
        };
    }
    Ok((m, k, b[1]))
}

/// `MatMul` — claim f32 `A` of rank >= 2 against a fully-static rank-2 `B`.
fn matmul(view: &NodeView<'_>, spec: &OpSpec) -> ClaimResult {
    let a = claim::input_edge(view, spec, 0)?;
    claim::check_dtype(spec, &a, "input 0 (A)")?;
    claim::check_shape(spec, &a, "input 0 (A)")?;
    let b = claim::input_edge(view, spec, 1)?;
    claim::check_dtype(spec, &b, "input 1 (B)")?;
    claim::check_shape(spec, &b, "input 1 (B)")?;

    require!(
        view.num_inputs() == 2,
        Arity,
        "`{}` has {} inputs; it takes exactly A and B",
        spec.op_type,
        view.num_inputs()
    );

    // ORT reports `rank 0` for a genuine scalar **and** for a tensor whose rank shape inference
    // never established; `GetDimensionsCount` cannot tell them apart and `EdgeType` faithfully
    // records both as `Some([])`. For `MatMul` the ambiguity resolves itself: ONNX `MatMul` has
    // no rank-0 form on either operand, so a rank-0 reading here is *provably* an unresolved
    // rank rather than a scalar, and the decline says so rather than reporting `[rank]` as if the
    // graph had asked for something impossible.
    //
    // Measured on BERT-SQuAD-12, 2026-08-04: 94 of its 95 `MatMul` nodes read rank 0 for `A` at
    // `GetCapability`, because the TF converter emits `Reshape` targets computed from `Shape` and
    // neither ORT's nor `onnx`'s inference resolves those before partitioning.
    let a_shape = a.shape.as_deref().unwrap_or(&[]);
    let b_shape = b.shape.as_deref().unwrap_or(&[]);
    require!(
        !a_shape.is_empty(),
        UnknownRank,
        "`{}` input 0 (A) reads as rank 0, and ONNX `MatMul` has no rank-0 operand; ORT reports \
         rank 0 both for a scalar and for a tensor whose rank shape inference never established, \
         so this is the second. No runtime-extent handling recovers a rank",
        spec.op_type
    );
    require!(
        !b_shape.is_empty(),
        UnknownRank,
        "`{}` input 1 (B) reads as rank 0, and ONNX `MatMul` has no rank-0 operand; see input 0",
        spec.op_type
    );
    require!(
        b_shape.len() == 2 && b_shape[0] > 0 && b_shape[1] > 0,
        DynamicShape,
        "`{}` input 1 (B) is {b_shape:?}; this row needs a fully-static rank-2 `B` because the \
         inner-product length is the kernel's loop bound",
        spec.op_type
    );

    if let Err(why) = matmul_2d_extents(a_shape, b_shape) {
        // `A`'s leading extents are allowed to be symbolic *at the gate* — that is the
        // runtime-extent case §8.8 is about and the engine's dynamic-dispatch path re-runs
        // `translate` at `Compute` with the concrete shapes. Everything else is a real decline.
        let leading_symbolic =
            a_shape.len() >= 2 && a_shape[..a_shape.len() - 1].iter().any(|&d| d < 0);
        if !(leading_symbolic && claim::runtime_extents_ok()) {
            crate::deny!(Shape, "`{}` {why}", spec.op_type);
        }
        // Even under runtime extents, `K` is a loop bound and must be checkable now.
        require!(
            a_shape[a_shape.len() - 1] == b_shape[0],
            Shape,
            "`{}` input 0 (A) contributes K={} and input 1 (B) contributes K={}",
            spec.op_type,
            a_shape[a_shape.len() - 1],
            b_shape[0]
        );
    }
    Ok(())
}

crate::op_table! {
    //  op      domain  opsets            caps  kernel          claim   translate   status
    //
    // Opset window opens at 7: opsets 1-6 carried a `broadcast` attribute that made `C`'s
    // broadcasting opt-in, and opset 7 removed it in favour of the unidirectional rule this
    // module implements. A row that opened at 1 would claim a node whose `broadcast=0` means
    // "C must already be [M, N]" and read it under the opposite rule.
    // `alpha`, `beta`, `transA` and `transB` are all push constants in `gemm_f32.comp` —
    // expressions, not paths, by the same §8.9.23 argument that rules `Conv`'s four — so they are
    // `blind_axes` and not key components. (`form.rs` asserted the opposite for the transposes
    // and was deleted; see the test at the foot of this file.)
    "Gemm",   Ai,     7 ..= OPSET_ANY, F32,  kernel!(Standalone, "gemm"),  gemm,   translate,  Ready,
        blind_axes: &["alpha", "beta", "transA", "transB"];

    // `MatMul` dispatches **`gemm_f32`**, the same module `Gemm` does, and that is the design
    // rather than an economy.
    //
    // A row-major `[d0, d1, ..., K]` buffer *is* a row-major `[d0*d1*..., K]` buffer, so a
    // rank-N `A` against a rank-2 `B` reduces to `[M, K] x [K, N]` by reinterpreting indices and
    // copying nothing. Set `alpha=1`, `beta=0`, `has_c=0`, `transA=transB=0` and `gemm_f32`
    // computes ONNX `MatMul` exactly — the same pipeline emitting the same instructions.
    //
    // A separate `matmul_f32.comp` would have been a **variant component that distinguishes
    // nothing**, which is the error §8.9.23 ruled against on `Conv` and which `form.rs` was
    // deleted for committing. What differs between the two ops is the *claim* — output rank,
    // operand arity, the absence of `C` and of the transposes — and that difference lives in a
    // separate registry row with a separate proof key, which is where a difference in what is
    // claimed belongs. The keys read `.../MatMul/.../gemm_f32/...` and `.../Gemm/.../gemm_f32/...`
    // and are two distinct strings over one module, which the `<prefix>_<dtype>` variant scheme
    // already handles.
    //
    // Opset window opens at 1: `MatMul` has carried the NumPy `matmul` semantics unchanged since
    // opset 1 (opsets 9 and 13 only widened the type constraint, which the `caps` set answers).
    // There is no `broadcast`-attribute era to exclude, unlike `Gemm`.
    //
    // No `blind_axes`: `MatMul` has no attributes at all, so there is nothing for the key to be
    // silent about. That is worth stating rather than leaving as an empty field — a reader who
    // has just read `Gemm`'s four will otherwise assume an omission.
    "MatMul", Ai,     1 ..= OPSET_ANY, F32,  kernel!(Standalone, "gemm"),  matmul, translate_matmul, Ready;
}

/// Translate `MatMul` into the `Gemm` dispatch it is.
///
/// Reads the shapes off [`NodeDesc`] rather than trusting the claim: on BERT-SQuAD-12 the claim
/// gate sees rank 0 for `A` on 94 of 95 nodes and `Compile`/`Compute` see the real shape, so the
/// gate is deliberately the weaker of the two readings and this is the one that must be strict.
pub fn translate_matmul(
    _spec: &OpSpec,
    node: &NodeDesc,
    ctx: &mut dyn DispatchContext,
) -> EpResult<()> {
    let a = node
        .inputs
        .first()
        .and_then(|t| t.desc.as_ref())
        .ok_or_else(|| {
            EpError::Unsupported(format!(
                "`{}` input 0 has no shape at compile time",
                node.op_type
            ))
        })?;
    let b = node
        .inputs
        .get(1)
        .and_then(|t| t.desc.as_ref())
        .ok_or_else(|| {
            EpError::Unsupported(format!(
                "`{}` input 1 has no shape at compile time",
                node.op_type
            ))
        })?;
    if a.dtype != DType::F32 || b.dtype != DType::F32 {
        return Err(EpError::Unsupported(format!(
            "`{}` inputs are {:?}/{:?}; gemm_f32 reads one element per word",
            node.op_type, a.dtype, b.dtype
        )));
    }
    let (m, k, n) = matmul_2d_extents(&a.shape, &b.shape)
        .map_err(|why| EpError::Unsupported(format!("`{}` {why}", node.op_type)))?;
    if m <= 0 || n <= 0 || k <= 0 {
        return Err(EpError::Unsupported(format!(
            "`{}` computes a {m}x{n} product over K={k}; nothing to dispatch",
            node.op_type
        )));
    }
    if node.outputs.len() != 1 {
        return Err(EpError::Internal(format!(
            "`{}` was claimed as single-output but has {}",
            node.op_type,
            node.outputs.len()
        )));
    }

    let a_buf = ctx.resolve(&node.inputs[0])?;
    let b_buf = ctx.resolve(&node.inputs[1])?;
    // ONNX keeps `A`'s leading axes in the output: `[d0, ..., dn-1, K] x [K, N]` is
    // `[d0, ..., dn-1, N]`. The *kernel* sees `[M, N]`; the *tensor* must not, or every consumer
    // downstream reads the wrong rank. This is the one place the collapse is not free.
    let mut out_shape: Vec<i64> = a.shape[..a.shape.len() - 1].to_vec();
    out_shape.push(n);
    let out_buf = ctx.bind_output(&node.outputs[0], TensorDesc::new(DType::F32, out_shape))?;

    let total = u32::try_from(m * n).map_err(|_| {
        EpError::Unsupported(format!(
            "`{}` output element count overflows u32",
            node.op_type
        ))
    })?;

    let mut push = Vec::with_capacity(11 * 4);
    // `has_c = 0`, so the `C` read is predicated away and `beta` is never applied; `beta` is set
    // to 0.0 anyway so that a future reader of a captured push-constant block cannot mistake a
    // live `beta` for a dormant one.
    for v in [m, n, k, 0, 0, 0, 1, 1, i64::from(total)] {
        let v = u32::try_from(v).map_err(|_| {
            EpError::Unsupported(format!(
                "`{}` parameter {v} does not fit a u32",
                node.op_type
            ))
        })?;
        push.extend_from_slice(&v.to_le_bytes());
    }
    push.extend_from_slice(&1.0f32.to_le_bytes());
    push.extend_from_slice(&0.0f32.to_le_bytes());

    // `C` is absent, so its binding takes `A` as the inert placeholder — the same rule `Gemm`
    // and `Conv` use for an omitted optional input. The module declares four bindings whether or
    // not the fourth is read.
    let groups = total
        .div_ceil(GEMM_LOCAL_SIZE)
        .clamp(1, GEMM_MAX_WORKGROUPS);
    ctx.dispatch(KernelRequest {
        shader: "gemm_f32",
        spec_constants: vec![GEMM_LOCAL_SIZE],
        push_constants: push,
        bindings: vec![a_buf, b_buf, a_buf, out_buf],
        workgroups: [groups, 1, 1],
    })
}

/// Translate into one dispatch: one invocation per output element.
pub fn translate(_spec: &OpSpec, node: &NodeDesc, ctx: &mut dyn DispatchContext) -> EpResult<()> {
    let a = node
        .inputs
        .first()
        .and_then(|t| t.desc.as_ref())
        .ok_or_else(|| {
            EpError::Unsupported(format!(
                "`{}` input 0 has no shape at compile time",
                node.op_type
            ))
        })?;
    let b = node
        .inputs
        .get(1)
        .and_then(|t| t.desc.as_ref())
        .ok_or_else(|| {
            EpError::Unsupported(format!(
                "`{}` input 1 has no shape at compile time",
                node.op_type
            ))
        })?;
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return Err(EpError::Unsupported(format!(
            "`{}` was claimed with ranks {} / {}; this kernel is rank 2",
            node.op_type,
            a.shape.len(),
            b.shape.len()
        )));
    }
    if a.dtype != DType::F32 || b.dtype != DType::F32 {
        return Err(EpError::Unsupported(format!(
            "`{}` inputs are {:?}/{:?}; gemm_f32 reads one element per word",
            node.op_type, a.dtype, b.dtype
        )));
    }

    let int_attr = |name: &str| match node.attributes.get(name) {
        Some(AttrValue::Int(v)) => Some(*v),
        _ => None,
    };
    let float_attr = |name: &str| match node.attributes.get(name) {
        Some(AttrValue::Float(v)) => Some(*v),
        _ => None,
    };
    let trans_a = int_attr("transA").unwrap_or(0) != 0;
    let trans_b = int_attr("transB").unwrap_or(0) != 0;
    let alpha = float_attr("alpha").unwrap_or(1.0);
    let beta = float_attr("beta").unwrap_or(1.0);

    let (m, k) = a_extents(&a.shape, trans_a);
    let (kb, n) = b_extents(&b.shape, trans_b);
    if k != kb {
        return Err(EpError::Internal(format!(
            "`{}` was claimed with K={k} from A and K={kb} from B",
            node.op_type
        )));
    }
    if m <= 0 || n <= 0 || k <= 0 {
        return Err(EpError::Unsupported(format!(
            "`{}` computes a {m}x{n} product over K={k}; nothing to dispatch",
            node.op_type
        )));
    }

    let has_c = node.inputs.len() >= 3 && !node.inputs[2].name.is_empty();
    let bc = if has_c {
        let c = node.inputs[2].desc.as_ref().ok_or_else(|| {
            EpError::Unsupported(format!(
                "`{}` input 2 has no shape at compile time",
                node.op_type
            ))
        })?;
        c_broadcast(&c.shape, m, n)
            .map_err(|why| EpError::Unsupported(format!("`{}` {why}", node.op_type)))?
    } else {
        CBroadcast { rows: 1, cols: 1 }
    };

    let a_buf = ctx.resolve(&node.inputs[0])?;
    let b_buf = ctx.resolve(&node.inputs[1])?;
    // An absent `C` binds `A`, the same inert-placeholder rule `conv` uses for an omitted bias:
    // `has_c` is zero so the read is predicated away, and the binding still exists because the
    // module declares it.
    let c_buf = if has_c {
        ctx.resolve(&node.inputs[2])?
    } else {
        a_buf
    };

    if node.outputs.len() != 1 {
        return Err(EpError::Internal(format!(
            "`{}` was claimed as single-output but has {}",
            node.op_type,
            node.outputs.len()
        )));
    }
    let out_buf = ctx.bind_output(&node.outputs[0], TensorDesc::new(DType::F32, vec![m, n]))?;

    let total = u32::try_from(m * n).map_err(|_| {
        EpError::Unsupported(format!(
            "`{}` output element count overflows u32",
            node.op_type
        ))
    })?;

    let mut push = Vec::with_capacity(10 * 4);
    for v in [
        m,
        n,
        k,
        i64::from(trans_a),
        i64::from(trans_b),
        i64::from(has_c),
        bc.rows,
        bc.cols,
        i64::from(total),
    ] {
        let v = u32::try_from(v).map_err(|_| {
            EpError::Unsupported(format!(
                "`{}` parameter {v} does not fit a u32",
                node.op_type
            ))
        })?;
        push.extend_from_slice(&v.to_le_bytes());
    }
    // The two float parameters go last, so the integer prefix keeps the layout every other
    // hand-written kernel in this crate uses and the GLSL block reads top to bottom.
    push.extend_from_slice(&alpha.to_le_bytes());
    push.extend_from_slice(&beta.to_le_bytes());

    let groups = total
        .div_ceil(GEMM_LOCAL_SIZE)
        .clamp(1, GEMM_MAX_WORKGROUPS);
    ctx.dispatch(KernelRequest {
        shader: "gemm_f32",
        spec_constants: vec![GEMM_LOCAL_SIZE],
        push_constants: push,
        bindings: vec![a_buf, b_buf, c_buf, out_buf],
        workgroups: [groups, 1, 1],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::common::templates;
    use crate::registry::OpStatus;

    #[test]
    fn the_row_is_ready_and_points_at_this_handler() {
        let row = OPS
            .iter()
            .find(|s| s.op_type == "Gemm")
            .expect("Gemm must be registered");
        assert_eq!(row.status, OpStatus::Ready);
        assert!(std::ptr::fn_addr_eq(
            row.translate,
            translate as crate::registry::TranslateHandler
        ));
        assert!(
            !std::ptr::fn_addr_eq(
                row.translate,
                templates::unimplemented as crate::registry::TranslateHandler
            ),
            "the comparison must discriminate, or the assertion above proves nothing"
        );
    }

    #[test]
    fn f16_is_declined() {
        let row = OPS.iter().find(|s| s.op_type == "Gemm").unwrap();
        assert!(row.caps.contains(DType::F32));
        assert!(!row.caps.contains(DType::F16));
    }

    /// Opset 6 and below carried a `broadcast` attribute with the opposite default. The window
    /// opening at 7 is the whole guard against reading such a node under today's rule.
    #[test]
    fn the_opset_window_excludes_the_pre_broadcast_rule_versions() {
        let row = OPS.iter().find(|s| s.op_type == "Gemm").unwrap();
        assert_eq!(row.min_opset, 7);
    }

    /// MobileNetV2's own node: `A=[batch, 1280]`, `B=[1000, 1280]`, `transB=1`, `C=[1000]`.
    #[test]
    fn the_mobilenetv2_head_resolves_to_a_1000_wide_output() {
        let (m, k) = a_extents(&[8, 1280], false);
        let (kb, n) = b_extents(&[1000, 1280], true);
        assert_eq!((m, k), (8, 1280));
        assert_eq!((kb, n), (1280, 1000));
        assert_eq!(
            c_broadcast(&[1000], m, n).unwrap(),
            CBroadcast {
                rows: 1,
                cols: 1000
            }
        );
    }

    #[test]
    fn a_scalar_c_broadcasts_over_everything() {
        assert_eq!(
            c_broadcast(&[], 4, 5).unwrap(),
            CBroadcast { rows: 1, cols: 1 }
        );
    }

    #[test]
    fn a_full_rank_two_c_is_taken_as_written() {
        assert_eq!(
            c_broadcast(&[4, 5], 4, 5).unwrap(),
            CBroadcast { rows: 4, cols: 5 }
        );
        assert_eq!(
            c_broadcast(&[1, 5], 4, 5).unwrap(),
            CBroadcast { rows: 1, cols: 5 }
        );
        assert_eq!(
            c_broadcast(&[4, 1], 4, 5).unwrap(),
            CBroadcast { rows: 4, cols: 1 }
        );
    }

    /// A `C` that does not broadcast is refused, not reinterpreted. A rank-1 `[M]` is the
    /// tempting case — it looks like a per-row bias and ONNX aligns it to the *columns*.
    #[test]
    fn a_rank_one_c_aligns_to_the_columns_and_a_row_length_one_is_refused() {
        let err = c_broadcast(&[4], 4, 5).unwrap_err();
        assert!(err.contains("columns"), "{err}");
    }

    #[test]
    fn a_rank_three_c_is_refused() {
        let err = c_broadcast(&[1, 4, 5], 4, 5).unwrap_err();
        assert!(err.contains("rank 3"), "{err}");
    }

    /// The transposes are **blind axes, not key components** — §8.9.23, and this test is the
    /// reversal of the one it replaces.
    ///
    /// It previously asserted `transA`/`transB` were form bits in the proof key, on the reasoning
    /// that a `transB=1` proof would otherwise grant a `transB=0` claim. Morpheus ruled the
    /// general form of that question on `Conv`'s four attributes and the ruling applies here
    /// verbatim: `gemm_f32.comp` selects the index with a ternary on a push constant, which is one
    /// pipeline emitting one set of instructions, so under §8.7 a transpose is an **expression,
    /// not a path** and the key that omits it is true. Proof of one transpose *is* proof of the
    /// other by the key's own stated meaning ("two nodes whose keys are equal are dispatched by
    /// the same code with the same descriptor layout").
    ///
    /// The worry the old test encoded is real and survives — it is just not the key's to answer.
    /// It is answered by the disclosure, which names these axes as blind, and by the CI-time suite
    /// that varies them.
    #[test]
    fn the_transposes_are_declared_as_blind_axes_and_not_as_key_components() {
        let spec = crate::registry::all_specs()
            .find(|s| s.op_type == "Gemm")
            .expect("Gemm must be registered");
        assert!(spec.blind_axes.contains(&"transA"));
        assert!(spec.blind_axes.contains(&"transB"));
    }

    /// `Gemm`/`MatMul` anchor a partition **only when their weight operand `B` (index 1) is a
    /// resident initializer** — which the classifier head and every transformer projection carry.
    /// A lone weight `Gemm` at a model's tail is claimable because of that; an activation-only one
    /// would not anchor. Both polarities asserted so this is a check, not a tautology.
    #[test]
    fn gemm_is_a_partition_anchor() {
        use crate::ops::partition::is_anchor;
        // Weight at the designated site (index 1) ⇒ anchors.
        assert!(is_anchor("Gemm", &[false, true]));
        assert!(is_anchor("MatMul", &[false, true]));
        // Both operands runtime ⇒ does not anchor.
        assert!(!is_anchor("Gemm", &[false, false]));
        assert!(!is_anchor("MatMul", &[false, false]));
    }

    // ---------------------------------------------------------------------------------------
    // MatMul
    // ---------------------------------------------------------------------------------------

    #[test]
    fn the_matmul_row_is_ready_and_points_at_its_own_handler() {
        let row = OPS
            .iter()
            .find(|s| s.op_type == "MatMul")
            .expect("MatMul must be registered");
        assert_eq!(row.status, OpStatus::Ready);
        assert!(std::ptr::fn_addr_eq(
            row.translate,
            translate_matmul as crate::registry::TranslateHandler
        ));
        // The two rows must NOT share a translate handler: `Gemm`'s reads `transA`/`alpha` and
        // computes a rank-2 output, and `MatMul` has neither and does not.
        let gemm = OPS.iter().find(|s| s.op_type == "Gemm").unwrap();
        assert!(!std::ptr::fn_addr_eq(row.translate, gemm.translate));
    }

    /// One module, two rows. This is the §8.9.23 design and the test that pins it: if someone
    /// later adds a `matmul_f32.comp`, this fails and they have to argue for the second module.
    #[test]
    fn matmul_and_gemm_dispatch_the_same_module_under_different_keys() {
        let mm = OPS.iter().find(|s| s.op_type == "MatMul").unwrap();
        let gemm = OPS.iter().find(|s| s.op_type == "Gemm").unwrap();
        assert_eq!(
            format!("{:?}", mm.kernel),
            format!("{:?}", gemm.kernel),
            "MatMul must dispatch gemm_f32; a second module would be a variant component that \
             distinguishes nothing"
        );
        assert_ne!(mm.op_type, gemm.op_type, "the keys differ in component 1");
    }

    /// `MatMul` has no attributes, so it has nothing to be blind about. Asserted rather than
    /// left implicit: a reader coming from `Gemm`'s four blind axes will otherwise read the empty
    /// set as an oversight.
    #[test]
    fn matmul_declares_no_blind_axes_because_it_has_no_attributes() {
        let row = OPS.iter().find(|s| s.op_type == "MatMul").unwrap();
        assert!(row.blind_axes.is_empty());
        let gemm = OPS.iter().find(|s| s.op_type == "Gemm").unwrap();
        assert_eq!(gemm.blind_axes.len(), 4, "the comparison must discriminate");
    }

    /// The opset windows differ and the reason is a real semantic break in `Gemm`, not a habit.
    #[test]
    fn matmul_opens_at_opset_one_and_gemm_does_not() {
        let mm = OPS.iter().find(|s| s.op_type == "MatMul").unwrap();
        assert_eq!(mm.min_opset, 1);
        assert_eq!(
            OPS.iter().find(|s| s.op_type == "Gemm").unwrap().min_opset,
            7
        );
    }

    /// BERT-SQuAD-12's own shapes: the attention projections and the feed-forward layers.
    #[test]
    fn the_bert_projection_and_feed_forward_shapes_collapse_to_a_two_d_product() {
        // `[batch*seq, 768] x [768, 768]` — the Q/K/V and output projections, 45 of 95.
        assert_eq!(
            matmul_2d_extents(&[256, 768], &[768, 768]).unwrap(),
            (256, 768, 768)
        );
        // Rank 3, which is what the graph carries before the encoder's Reshape.
        assert_eq!(
            matmul_2d_extents(&[1, 256, 768], &[768, 768]).unwrap(),
            (256, 768, 768)
        );
        // The feed-forward pair, 12 each.
        assert_eq!(
            matmul_2d_extents(&[256, 768], &[768, 3072]).unwrap(),
            (256, 768, 3072)
        );
        assert_eq!(
            matmul_2d_extents(&[256, 3072], &[3072, 768]).unwrap(),
            (256, 3072, 768)
        );
        // The SQuAD head, 1 each.
        assert_eq!(
            matmul_2d_extents(&[256, 768], &[768, 2]).unwrap(),
            (256, 768, 2)
        );
    }

    /// The 24 batched attention products are declined **by name**, and the sentence says why it
    /// is a traversal difference rather than a parameter difference.
    #[test]
    fn a_batched_b_is_declined_and_the_reason_names_the_traversal() {
        let err = matmul_2d_extents(&[1, 12, 256, 64], &[1, 12, 64, 256]).unwrap_err();
        assert!(err.contains("rank 4"), "{err}");
        assert!(err.contains("batch indexing on both operands"), "{err}");
    }

    /// ONNX promotes a rank-1 operand and then removes the axis from the output again. Getting
    /// that wrong produces a right-sized buffer under a wrong rank, which every consumer then
    /// misreads — the plausible-wrong-answer failure the charter is about.
    #[test]
    fn a_rank_one_operand_is_declined_on_the_output_rank_not_on_the_arithmetic() {
        let err = matmul_2d_extents(&[768], &[768, 768]).unwrap_err();
        assert!(err.contains("rank 1"), "{err}");
        assert!(err.contains("removes"), "{err}");
        let err = matmul_2d_extents(&[256, 768], &[768]).unwrap_err();
        assert!(err.contains("vector-promotion"), "{err}");
    }

    #[test]
    fn a_mismatched_inner_length_is_declined() {
        let err = matmul_2d_extents(&[256, 768], &[512, 768]).unwrap_err();
        assert!(err.contains("K=768") && err.contains("K=512"), "{err}");
    }

    /// A symbolic leading extent cannot be multiplied into `M`. It is a *decline from this
    /// function*, which is not the same as a decline from the claim gate: the predicate lets it
    /// through under the runtime-extent rule precisely so the engine's dynamic-dispatch path can
    /// call this again at `Compute` with the concrete number. The two readings must not be
    /// collapsed, so this asserts the strict one.
    #[test]
    fn a_symbolic_leading_extent_is_not_multiplied_into_m() {
        let err = matmul_2d_extents(&[-1, 768], &[768, 768]).unwrap_err();
        assert!(err.contains("symbolic leading extent"), "{err}");
        // And the last axis is `K`, never a leading axis, so a symbolic `K` is a *different*
        // decline — it fails the equality against `B` rather than the product.
        let err = matmul_2d_extents(&[256, -1], &[768, 768]).unwrap_err();
        assert!(err.contains("K=-1"), "{err}");
    }

    /// The collapse must not wrap. An extent product that overflows would otherwise become a
    /// small, plausible row count and dispatch a fraction of the work over a buffer that keeps
    /// whatever was in it — silently wrong, which is the failure mode this module exists to avoid.
    #[test]
    fn an_overflowing_leading_product_declines_rather_than_wrapping() {
        let err = matmul_2d_extents(&[i64::MAX, 3, 768], &[768, 768]).unwrap_err();
        assert!(err.contains("overflow"), "{err}");
    }

    /// The output keeps `A`'s leading axes; only the kernel sees `[M, N]`.
    #[test]
    fn the_output_rank_follows_a_not_the_collapsed_product() {
        let a = [2i64, 3, 256, 768];
        let (m, _k, n) = matmul_2d_extents(&a, &[768, 512]).unwrap();
        assert_eq!(m, 2 * 3 * 256);
        let mut out: Vec<i64> = a[..a.len() - 1].to_vec();
        out.push(n);
        assert_eq!(out, vec![2, 3, 256, 512]);
        assert_eq!(out.iter().product::<i64>(), m * n);
    }

    #[test]
    fn matmul_declines_f16_for_the_same_reason_gemm_does() {
        let row = OPS.iter().find(|s| s.op_type == "MatMul").unwrap();
        assert!(row.caps.contains(DType::F32));
        assert!(!row.caps.contains(DType::F16));
    }
}
