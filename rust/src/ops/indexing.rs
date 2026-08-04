//! Indexing — `Gather` and the ops that read a tensor through another tensor's values.
//!
//! # Why this family is separate from elementwise
//!
//! An elementwise op's output index determines its input indices. An indexing op's do not: the
//! mapping goes through a *second tensor's values*, which is why the shared broadcasting helper
//! cannot express it and why the shader takes three flattened extents instead of a shape plan.
//!
//! # The one node that motivated it
//!
//! `/model/embed_tokens/Gather` in Phi-3.5-mini-int4 — the token embedding lookup. It is the
//! **first** node of the graph, so before it was claimed the EP's island began one node too late
//! and the embedding's `FLOAT16[batch, seq, 3072]` output crossed the host↔device boundary on
//! every inference. Claiming it moves the boundary to `input_ids`, `INT64[batch, seq]`: 8 bytes
//! per token instead of 6144.
//!
//! That is the whole argument for this op, and it is a boundary-bytes argument rather than a
//! FLOPs one. A `Gather` does no arithmetic at all.
//!
//! # What is *not* claimed, on purpose
//!
//! The same graph contains a second `Gather` —
//! `/model/attn_mask_reformat/attn_mask_subgraph/Gather` — whose data tensor is `INT64[2]`, the
//! output of a `Shape`. It declines on `[dtype]` and that is the correct outcome, not an
//! accident of the caps list: it is int64 scalar control-plane arithmetic whose result is
//! consumed by 32 `GroupQueryAttention` nodes as a length. See `OP_COVERAGE.md` §7.4.

use crate::engine::DType;
use crate::kernel;
use crate::ops::common::claim::{self, ClaimResult};
use crate::ops::common::dtype::FLOAT;
use crate::ops::common::templates;
use crate::registry::OpStatus::Ready;
use crate::registry::{NodeView, OPSET_ANY, OpSpec};
use crate::require;

/// Index dtypes the gather kernel can read.
///
/// int64 is read as its low 32-bit word rather than through `shaderInt64` — exact for every
/// index the ONNX schema defines, because a valid index is bounded by a tensor extent. See the
/// `gather_f32.comp` header.
const INDEX_DTYPES: [DType; 2] = [DType::I64, DType::I32];

/// `Gather` — claim float data with integer indices, any axis, any indices rank.
///
/// The three checks that matter, and why each is a decline rather than a widening:
///
/// * **data dtype ∈ `caps` (f16/f32).** An int64 data tensor is the control-plane case; the
///   kernel would have to do 64-bit loads to be correct and there is no arithmetic to win.
/// * **indices dtype ∈ int32/int64.** Anything else is not a legal `Gather` per the schema.
/// * **axis in range.** Normalised here so the translate handler never sees a negative axis.
///
/// Rank and extents are deliberately *not* constrained: the kernel flattens the data tensor
/// around its axis into three extents, so rank costs nothing.
fn gather(view: &NodeView<'_>, spec: &OpSpec) -> ClaimResult {
    let data = claim::input_edge(view, spec, 0)?;
    claim::check_dtype(spec, &data, "input 0 (data)")?;
    claim::check_shape(spec, &data, "input 0 (data)")?;

    let rank = data.rank().unwrap_or(0);
    require!(
        rank >= 1,
        Rank,
        "`{}` cannot gather from a rank-0 data tensor",
        spec.op_type
    );

    let indices = claim::input_edge(view, spec, 1)?;
    claim::check_shape(spec, &indices, "input 1 (indices)")?;
    let idx_dtype = indices.dtype;
    require!(
        idx_dtype.is_some_and(|d| INDEX_DTYPES.contains(&d)),
        DType,
        "`{}` indices are {:?}; this kernel reads int32 or int64 indices",
        spec.op_type,
        idx_dtype
    );

    let axis = view.attr_int("axis").unwrap_or(0);
    let normalized = if axis < 0 { axis + rank as i64 } else { axis };
    require!(
        normalized >= 0 && normalized < rank as i64,
        Attribute,
        "`{}` axis {axis} is out of range for a rank-{rank} data tensor",
        spec.op_type
    );

    Ok(())
}

crate::op_table! {
    //  op        domain  opsets            caps    kernel          claim   translate           status
    //
    // `caps` is FLOAT and that is the substance of the row, not a placeholder: it is what makes
    // the attention-mask `Gather` (int64 data) decline with `[dtype]` while the embedding
    // `Gather` (fp16 data) is claimed. One row, two correct answers, and the histogram says which
    // was which.
    //
    // The window opens at 1 because `Gather` has existed since opset 1 and its semantics have not
    // changed across 1/11/13 — the revisions widened the index dtype and clarified negative
    // indices, both of which this row handles explicitly.
    "Gather",   Ai,     1 ..= OPSET_ANY,  FLOAT,  kernel!(Standalone, "gather"),  gather, templates::gather,  Ready;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::OpStatus;

    #[test]
    fn the_gather_row_is_ready_and_points_at_the_gather_handler() {
        let row = OPS
            .iter()
            .find(|s| s.op_type == "Gather")
            .expect("Gather must be registered");
        assert_eq!(row.status, OpStatus::Ready);
        assert!(
            std::ptr::fn_addr_eq(
                row.translate,
                templates::gather as crate::registry::TranslateHandler
            ),
            "a Ready row must not point at `unimplemented`"
        );
        // The negative polarity. `fn_addr_eq` compares addresses, and an address comparison that
        // can only ever return true is not a check — this pins that the assertion above
        // discriminates rather than passing for every row.
        assert!(
            !std::ptr::fn_addr_eq(
                row.translate,
                templates::unimplemented as crate::registry::TranslateHandler
            ),
            "the comparison must distinguish handlers, or the assertion above proves nothing"
        );
        assert!(
            row.schema.is_none(),
            "`Gather` is an ai.onnx op; a contrib fingerprint here would be meaningless"
        );
    }

    /// The caps list is the decline mechanism for the control-plane `Gather`, so it is asserted
    /// rather than left implicit.
    ///
    /// If someone widens this row to `ANY` to "get one more node", the attention-mask `Gather`
    /// starts being claimed, its int64 data is read as fp16 through the packed-uint path, and the
    /// model produces a wrong `seqlens_k` — silently, because every downstream op still runs.
    /// That is the exact failure mode the charter's first line names.
    #[test]
    fn gather_claims_float_data_only() {
        let row = OPS.iter().find(|s| s.op_type == "Gather").unwrap();
        assert!(row.caps.contains(DType::F16));
        assert!(row.caps.contains(DType::F32));
        assert!(
            !row.caps.contains(DType::I64),
            "int64 data must decline: the attn-mask Gather reads a Shape output and there is no \
             arithmetic to win by moving it to the device"
        );
        assert!(!row.caps.contains(DType::I32));
    }

    #[test]
    fn index_dtypes_are_the_two_the_onnx_schema_allows() {
        assert_eq!(INDEX_DTYPES, [DType::I64, DType::I32]);
    }
}
