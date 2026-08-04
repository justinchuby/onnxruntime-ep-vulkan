//! Dense linear algebra — `Gemm`, the classifier head and the transformer feed-forward layer.
//!
//! # Why this module exists
//!
//! Two censuses, not a taxonomy:
//!
//! * MobileNetV2-12, 2026-08-04: with `Conv` and `GlobalAveragePool` claimed, the single `Gemm`
//!   at the tail is the last node in the model that carries data. It is also an *anchor* in
//!   `ops::partition::is_anchor`, so a lone `Gemm` island is exempt from the minimum-node gate —
//!   the row was already written into the partitioner's cost model before any kernel existed.
//! * The registry had no `Gemm` at all, which is why every non-LLM model this project has looked
//!   at ends on the CPU regardless of how much of its body the EP claims.
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
//!   is a different row this module does not yet carry.
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

crate::op_table! {
    //  op      domain  opsets            caps  kernel          claim   translate   status
    //
    // Opset window opens at 7: opsets 1-6 carried a `broadcast` attribute that made `C`'s
    // broadcasting opt-in, and opset 7 removed it in favour of the unidirectional rule this
    // module implements. A row that opened at 1 would claim a node whose `broadcast=0` means
    // "C must already be [M, N]" and read it under the opposite rule.
    // `alpha` and `beta` are push constants in `gemm_f32.comp` — expressions, not paths, by the
    // same §8.9.23 argument that rules `Conv`'s four. `transA`/`transB` are **not** here: they are
    // in the key, under `form.rs`, because they change which index the loop reads.
    "Gemm",   Ai,     7 ..= OPSET_ANY, F32,  kernel!(Standalone, "gemm"),  gemm,   translate,  Ready,
        blind_axes: &["alpha", "beta", "transA", "transB"];
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

    /// `Gemm` is already an anchor in the partitioner, which is why a lone one at a model's tail
    /// is claimable at all. If that ever changed, this row would go live and never be used.
    #[test]
    fn gemm_is_a_partition_anchor() {
        assert!(crate::ops::partition::is_anchor("Gemm"));
    }
}
