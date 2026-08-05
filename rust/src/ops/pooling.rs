//! Pooling — `GlobalAveragePool`, the op that turns a feature map into a feature vector.
//!
//! # Why this module exists
//!
//! `probe_model_op_census.py` against MobileNetV2-12 on 2026-08-04, after `Conv` landed:
//! **97 of 105 nodes claimed**, and the remaining eight were one contiguous tail. Two of those
//! eight carry data — `GlobalAveragePool` and `Gemm` — and six compute a two-element shape vector
//! on the host. This module is the first of the two.
//!
//! The census is the argument. `AveragePool`, `MaxPool` and the rest of the pooling taxonomy are
//! not here, because no graph this project has censused contains one: MobileNetV2 replaced them
//! with strided convolution, which is why it has 52 `Conv` nodes and exactly one pool.
//!
//! # What is claimed, and what is declined by name
//!
//! Claimed: rank-4 f32 `GlobalAveragePool`, batch extent symbolic, channel and spatial extents
//! concrete.
//!
//! * **f16** — same argument as `conv.rs`: packed-`uint` half I/O addresses two elements per
//!   word and a reduction reads single elements. Declines `[dtype]`, not silently.
//! * **rank != 4** — ONNX allows `[N, C, D1..Dn]` for any `n >= 1`. A rank-agnostic kernel needs
//!   the spatial extent product as a runtime value and the output rank as a pipeline fact; the
//!   4-D case is the one every vision graph emits and the one this kernel implements.
//! * **a symbolic channel or spatial extent** — the divisor is `H * W` and the grid is `N * C`.
//!   Batch may be symbolic, exactly as in `conv.rs`, and for the same reason: it is the only
//!   extent that does not enter the arithmetic.
//!
//! # On the accumulator
//!
//! `float`, not `double` and not a pairwise tree. MobileNetV2's pool reduces 49 elements; ORT's
//! CPU reference accumulates in f32 in loop order, and a kernel that accumulated more precisely
//! would make the differential comparison a test of two algorithms rather than of one
//! implementation. `FP32_CONV`'s derivation in `tests/ops/_models.py` makes the same call.

use crate::engine::{
    DType, DispatchContext, EpError, EpResult, KernelRequest, NodeDesc, TensorDesc,
};
use crate::kernel;
use crate::ops::common::claim::{self, ClaimResult};
use crate::ops::common::dtype::F32;
use crate::registry::OpStatus::Ready;
use crate::registry::{NodeView, OPSET_ANY, OpSpec};
use crate::require;

/// Workgroup size, matching every other 1-D grid in this crate.
pub(crate) const POOL_LOCAL_SIZE: u32 = 256;

/// Cap on dispatched workgroups; the shader is a grid-stride loop.
pub(crate) const POOL_MAX_WORKGROUPS: u32 = 65_535;

/// The one rank the kernel implements: `[N, C, H, W]`.
const POOL_RANK: usize = 4;

/// `GlobalAveragePool` — claim rank-4 f32.
fn global_average_pool(view: &NodeView<'_>, spec: &OpSpec) -> ClaimResult {
    let x = claim::input_edge(view, spec, 0)?;
    claim::check_dtype(spec, &x, "input 0 (X)")?;
    claim::check_shape(spec, &x, "input 0 (X)")?;

    require!(
        view.num_inputs() == 1,
        Arity,
        "`{}` has {} inputs; it takes exactly one",
        spec.op_type,
        view.num_inputs()
    );
    require!(
        x.rank() == Some(POOL_RANK),
        Rank,
        "`{}` input 0 has rank {:?}; this kernel implements the 4-D `[N, C, H, W]` case only",
        spec.op_type,
        x.rank()
    );

    let shape = x.shape.as_deref().unwrap_or(&[]);
    require!(
        shape.len() == POOL_RANK && shape[1] > 0 && shape[2] > 0 && shape[3] > 0,
        DynamicShape,
        "`{}` input 0 has a symbolic channel or spatial extent ({shape:?}); the divisor is H*W \
         and the grid is N*C, so only the batch extent may be symbolic",
        spec.op_type
    );
    // A zero-extent spatial window would divide by zero and ONNX does not define the mean of an
    // empty set. Declining is the only answer that is not an invention.
    require!(
        shape[2] * shape[3] > 0,
        Shape,
        "`{}` input 0 has an empty spatial window ({shape:?}); the mean of no elements is not \
         a value this EP will invent",
        spec.op_type
    );
    Ok(())
}

crate::op_table! {
    //  op                   domain  opsets           caps  kernel          claim                translate   status
    //
    // Opset window opens at 1: `GlobalAveragePool` has existed since opset 1 and has never been
    // revised. `caps` is F32 for the reason in the module docs, not FLOAT.
    "GlobalAveragePool", Ai, 1 ..= OPSET_ANY, F32, kernel!(Standalone, "global_average_pool"), global_average_pool, translate, Ready;
}

/// Translate into one dispatch: one invocation per `(n, c)` pair.
pub fn translate(_spec: &OpSpec, node: &NodeDesc, ctx: &mut dyn DispatchContext) -> EpResult<()> {
    let x = node
        .inputs
        .first()
        .and_then(|t| t.desc.as_ref())
        .ok_or_else(|| {
            EpError::Unsupported(format!(
                "`{}` input 0 has no shape at compile time",
                node.op_type
            ))
        })?;
    if x.shape.len() != POOL_RANK {
        return Err(EpError::Unsupported(format!(
            "`{}` was claimed with rank {}; this kernel is 4-D",
            node.op_type,
            x.shape.len()
        )));
    }
    if x.dtype != DType::F32 {
        return Err(EpError::Unsupported(format!(
            "`{}` input 0 is {:?}; this kernel reads one element per word",
            node.op_type, x.dtype
        )));
    }
    let (n, c, h, w) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]);
    if n <= 0 || c <= 0 || h <= 0 || w <= 0 {
        return Err(EpError::Unsupported(format!(
            "`{}` has a non-positive extent in {:?}; nothing to dispatch",
            node.op_type, x.shape
        )));
    }
    if node.outputs.len() != 1 {
        return Err(EpError::Internal(format!(
            "`{}` was claimed as single-output but has {}",
            node.op_type,
            node.outputs.len()
        )));
    }

    let x_buf = ctx.resolve(&node.inputs[0])?;
    let out_buf = ctx.bind_output(
        &node.outputs[0],
        TensorDesc::new(DType::F32, vec![n, c, 1, 1]),
    )?;

    let total = u32::try_from(n * c).map_err(|_| {
        EpError::Unsupported(format!(
            "`{}` output element count overflows u32",
            node.op_type
        ))
    })?;
    let hw = u32::try_from(h * w).map_err(|_| {
        EpError::Unsupported(format!("`{}` spatial window overflows u32", node.op_type))
    })?;

    let mut push = Vec::with_capacity(8);
    push.extend_from_slice(&hw.to_le_bytes());
    push.extend_from_slice(&total.to_le_bytes());

    let groups = total
        .div_ceil(POOL_LOCAL_SIZE)
        .clamp(1, POOL_MAX_WORKGROUPS);
    ctx.dispatch(KernelRequest {
        shader: "global_average_pool_f32",
        spec_constants: vec![POOL_LOCAL_SIZE],
        push_constants: push,
        bindings: vec![x_buf, out_buf],
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
            .find(|s| s.op_type == "GlobalAveragePool")
            .expect("GlobalAveragePool must be registered");
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
    fn f16_is_declined_not_read_through_the_f32_kernel() {
        let row = OPS
            .iter()
            .find(|s| s.op_type == "GlobalAveragePool")
            .unwrap();
        assert!(row.caps.contains(DType::F32));
        assert!(
            !row.caps.contains(DType::F16),
            "packed-uint half I/O addresses two elements per word; a reduction reads single \
             elements and needs its own module"
        );
    }

    /// `GlobalAveragePool` declares no blind axes, and that is a decision rather than an omission:
    /// it has no attributes at all, so there is nothing for a key to be blind to.
    #[test]
    fn the_op_has_no_blind_axes_because_it_has_no_attributes() {
        let spec = crate::registry::all_specs()
            .find(|s| s.op_type == "GlobalAveragePool")
            .expect("GlobalAveragePool must be registered");
        assert!(spec.blind_axes.is_empty());
    }
}
