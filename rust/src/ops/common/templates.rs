//! Shared translate handlers — the "one kernel family, dozens of ops" half of the leverage.
//!
//! A handler turns a [`NodeDesc`] into dispatches against the [`DispatchContext`] seam. There is
//! one per *template*, not one per op: the op's identity reaches the shader through the variant
//! stem in its row's [`Kernel`], and its shape logic comes from the shared
//! [`ShapePlan`](super::shape_plan::ShapePlan). `OP_COVERAGE.md` §5.1 is the argument for why that
//! is where the schedule is won.
//!
//! # The invariant
//!
//! Every handler here assumes its row's claim predicate already ran. It may therefore treat
//! resolved dtypes and static shapes as facts — but it still returns [`EpError::Unsupported`]
//! rather than panicking if an assumption fails, because a panic crossing the FFI boundary is
//! undefined behaviour and a wrong answer is worse than a slow one.

use crate::engine::{
    DType, DispatchContext, EpError, EpResult, KernelRequest, NodeDesc, TensorDesc,
};
use crate::registry::OpSpec;

use super::params;
use super::shape_plan::ShapePlan;

/// Workgroup size every elementwise template is compiled with.
///
/// 256 is the largest value guaranteed by `maxComputeWorkGroupInvocations` across the baseline
/// capability set (`ENGINE.md` §3), and it divides evenly into every vendor's preferred subgroup
/// size, so it is the one number that is never wrong. It reaches the shader as a specialisation
/// constant so a per-device tuner can override it later without recompiling GLSL.
pub const EW_LOCAL_SIZE: u32 = 256;

/// Gather the static shapes of a node's first `n` inputs.
fn input_shapes(node: &NodeDesc, n: usize) -> EpResult<Vec<Vec<i64>>> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let t = node.inputs.get(i).ok_or_else(|| {
            EpError::InvalidGraph(format!(
                "`{}` was claimed with {} inputs but input {i} is missing at compile time",
                node.op_type,
                node.inputs.len()
            ))
        })?;
        let desc = t.desc.as_ref().ok_or_else(|| {
            EpError::Unsupported(format!(
                "`{}` input {i} (`{}`) has no shape at compile time",
                node.op_type, t.name
            ))
        })?;
        out.push(desc.shape.clone());
    }
    Ok(out)
}

/// The element type a node's inputs agree on.
fn common_dtype(node: &NodeDesc, from: usize, n: usize) -> EpResult<DType> {
    let mut found: Option<DType> = None;
    for i in from..n {
        let dt = node
            .inputs
            .get(i)
            .and_then(|t| t.desc.as_ref())
            .map(|d| d.dtype)
            .ok_or_else(|| {
                EpError::Unsupported(format!(
                    "`{}` input {i} has no element type at compile time",
                    node.op_type
                ))
            })?;
        match found {
            None => found = Some(dt),
            Some(a) if a != dt => {
                return Err(EpError::Unsupported(format!(
                    "`{}` mixes element types across its inputs",
                    node.op_type
                )));
            }
            _ => {}
        }
    }
    found.ok_or_else(|| {
        EpError::Internal(format!(
            "`{}` has no inputs to take a dtype from",
            node.op_type
        ))
    })
}

/// The single output description a node was claimed for.
fn single_output(node: &NodeDesc) -> EpResult<&crate::engine::OutRef> {
    if node.outputs.len() != 1 {
        return Err(EpError::Internal(format!(
            "`{}` was claimed as single-output but has {}",
            node.op_type,
            node.outputs.len()
        )));
    }
    Ok(&node.outputs[0])
}

/// Build the dispatch for one elementwise template invocation.
///
/// This is the whole body of ~66 ops: broadcast on the host, resolve buffers, bind the output,
/// dispatch one 1-D grid. Nothing here knows what `Add` is.
fn dispatch_elementwise(
    spec: &OpSpec,
    node: &NodeDesc,
    ctx: &mut dyn DispatchContext,
    arity: usize,
    dtype_from: usize,
) -> EpResult<()> {
    let shapes = input_shapes(node, arity)?;
    let refs: Vec<&[i64]> = shapes.iter().map(Vec::as_slice).collect();
    let plan = ShapePlan::broadcast(&refs).map_err(|e| {
        EpError::Unsupported(format!("`{}` shapes cannot be planned: {e}", node.op_type))
    })?;

    let dtype = common_dtype(node, dtype_from, arity)?;
    let shader = spec.kernel.stem(dtype).ok_or_else(|| {
        EpError::Internal(format!(
            "`{}` was claimed but its row declares no shader",
            node.op_type
        ))
    })?;

    let mut bindings = Vec::with_capacity(arity + 1);
    for i in 0..arity {
        bindings.push(ctx.resolve(&node.inputs[i])?);
    }

    let out = single_output(node)?;
    let out_dtype = out.desc.as_ref().map_or(dtype, |d| d.dtype);
    let out_buf = ctx.bind_output(out, TensorDesc::new(out_dtype, plan.out_dims()))?;
    bindings.push(out_buf);

    ctx.dispatch(KernelRequest {
        shader,
        spec_constants: vec![EW_LOCAL_SIZE, u32::from(plan.all_identical)],
        push_constants: plan.push_constants_with_params(
            params::resolve(&node.op_type, node).map_err(|e| {
                // Claim already validated these, so reaching here means the claim predicate and
                // this resolver disagreed — an internal inconsistency, not a graph we should
                // have declined.
                EpError::Internal(format!("`{}` {e}", node.op_type))
            })?,
        ),
        bindings,
        workgroups: plan.workgroups_1d(EW_LOCAL_SIZE),
    })
}

/// Translate a unary elementwise op.
pub fn ew_unary(spec: &OpSpec, node: &NodeDesc, ctx: &mut dyn DispatchContext) -> EpResult<()> {
    dispatch_elementwise(spec, node, ctx, 1, 0)
}

/// Translate a binary elementwise op.
pub fn ew_binary(spec: &OpSpec, node: &NodeDesc, ctx: &mut dyn DispatchContext) -> EpResult<()> {
    dispatch_elementwise(spec, node, ctx, 2, 0)
}

/// Translate `Where` — three inputs, the first of which selects.
pub fn ew_select(spec: &OpSpec, node: &NodeDesc, ctx: &mut dyn DispatchContext) -> EpResult<()> {
    dispatch_elementwise(spec, node, ctx, 3, 1)
}

/// Translate `Clip` — three inputs that all share the value dtype, unlike `Where` whose first
/// input is `bool`. Same template, different `dtype_from`.
pub fn ew_clip(spec: &OpSpec, node: &NodeDesc, ctx: &mut dyn DispatchContext) -> EpResult<()> {
    dispatch_elementwise(spec, node, ctx, 3, 0)
}

/// Translate a variadic elementwise op by chaining the binary template.
///
/// Not yet reachable — every variadic row is staged — but written now because the *shape* of
/// "compose from primitives" has to exist before there is any temptation to write an N-input
/// shader.
pub fn ew_variadic(spec: &OpSpec, node: &NodeDesc, ctx: &mut dyn DispatchContext) -> EpResult<()> {
    let n = node.inputs.len();
    if n == 0 {
        return Err(EpError::InvalidGraph(format!(
            "`{}` has no inputs",
            node.op_type
        )));
    }
    if n <= 2 {
        return dispatch_elementwise(spec, node, ctx, n.max(1), 0);
    }
    Err(EpError::Unsupported(format!(
        "`{}` with {n} inputs needs the chained-dispatch lowering, which is not written yet",
        node.op_type
    )))
}

/// A handler that always refuses. The translate half of `claim::never`.
pub fn unimplemented(
    _spec: &OpSpec,
    node: &NodeDesc,
    _ctx: &mut dyn DispatchContext,
) -> EpResult<()> {
    Err(EpError::Unsupported(format!(
        "`{}` has no Vulkan translation yet; it should never have been claimed",
        node.op_type
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{BufferView, OutRef, TensorRef};

    /// A `DispatchContext` that records what a handler asked for.
    ///
    /// This is the piece that makes the template machinery testable with no Vulkan device, no
    /// shaders and no ORT session: the handlers are pure functions from a `NodeDesc` to a
    /// `KernelRequest`, so the request itself is the assertion target.
    #[derive(Default)]
    struct Recorder {
        next: u64,
        dispatches: Vec<KernelRequest>,
        outputs: Vec<TensorDesc>,
    }

    impl DispatchContext for Recorder {
        fn resolve(&mut self, _r: &TensorRef) -> EpResult<BufferView> {
            self.next += 1;
            Ok(BufferView::from_raw(self.next))
        }
        fn bind_output(&mut self, _o: &OutRef, desc: TensorDesc) -> EpResult<BufferView> {
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
        fn read_const_i64(&self, _r: &TensorRef) -> Option<Vec<i64>> {
            None
        }
    }

    fn tensor(name: &str, dtype: DType, shape: &[i64]) -> TensorRef {
        TensorRef {
            name: name.to_string(),
            desc: Some(TensorDesc::new(dtype, shape.to_vec())),
            is_initializer: false,
        }
    }

    fn out(name: &str, dtype: DType, shape: &[i64]) -> OutRef {
        OutRef {
            name: name.to_string(),
            desc: Some(TensorDesc::new(dtype, shape.to_vec())),
        }
    }

    fn spec_named(op: &str) -> &'static OpSpec {
        crate::registry::all_specs()
            .find(|s| s.op_type == op)
            .unwrap_or_else(|| panic!("`{op}` should be in the op table"))
    }

    #[test]
    fn a_binary_op_becomes_one_dispatch_at_the_right_variant() {
        let spec = spec_named("Add");
        let node = NodeDesc {
            op_type: "Add".into(),
            inputs: vec![
                tensor("a", DType::F32, &[2, 3, 4]),
                tensor("b", DType::F32, &[4]),
            ],
            outputs: vec![out("c", DType::F32, &[2, 3, 4])],
            ..Default::default()
        };
        let mut ctx = Recorder::default();
        ew_binary(spec, &node, &mut ctx).expect("translate");

        assert_eq!(ctx.dispatches.len(), 1);
        let k = &ctx.dispatches[0];
        assert_eq!(k.shader, "ew_binary_add_f32");
        assert_eq!(k.bindings.len(), 3, "two inputs and one output");
        assert_eq!(k.workgroups, [1, 1, 1]);
        assert_eq!(k.spec_constants[0], EW_LOCAL_SIZE);
        assert_eq!(k.spec_constants[1], 0, "broadcast, so not the fast path");
        assert_eq!(ctx.outputs[0].shape, vec![2, 3, 4]);
    }

    #[test]
    fn the_dtype_of_the_row_picks_the_variant() {
        let spec = spec_named("Mul");
        let node = NodeDesc {
            op_type: "Mul".into(),
            inputs: vec![tensor("a", DType::F16, &[8]), tensor("b", DType::F16, &[8])],
            outputs: vec![out("c", DType::F16, &[8])],
            ..Default::default()
        };
        let mut ctx = Recorder::default();
        ew_binary(spec, &node, &mut ctx).expect("translate");
        assert_eq!(ctx.dispatches[0].shader, "ew_binary_mul_f16");
        assert_eq!(
            ctx.dispatches[0].spec_constants[1], 1,
            "identical shapes take the linear fast path"
        );
    }

    #[test]
    fn a_unary_op_binds_two_buffers() {
        let spec = spec_named("Sqrt");
        let node = NodeDesc {
            op_type: "Sqrt".into(),
            inputs: vec![tensor("x", DType::F32, &[1024])],
            outputs: vec![out("y", DType::F32, &[1024])],
            ..Default::default()
        };
        let mut ctx = Recorder::default();
        ew_unary(spec, &node, &mut ctx).expect("translate");
        let k = &ctx.dispatches[0];
        assert_eq!(k.shader, "ew_unary_sqrt_f32");
        assert_eq!(k.bindings.len(), 2);
        assert_eq!(k.workgroups, [4, 1, 1]);
    }

    #[test]
    fn where_takes_its_dtype_from_the_value_inputs_not_the_condition() {
        let spec = spec_named("Where");
        let node = NodeDesc {
            op_type: "Where".into(),
            inputs: vec![
                tensor("c", DType::Bool, &[4]),
                tensor("x", DType::I32, &[4]),
                tensor("y", DType::I32, &[4]),
            ],
            outputs: vec![out("z", DType::I32, &[4])],
            ..Default::default()
        };
        let mut ctx = Recorder::default();
        ew_select(spec, &node, &mut ctx).expect("translate");
        assert_eq!(ctx.dispatches[0].shader, "ew_select_where_i32");
        assert_eq!(ctx.dispatches[0].bindings.len(), 4);
    }

    #[test]
    fn push_constants_travel_with_the_dispatch() {
        let spec = spec_named("Add");
        let node = NodeDesc {
            op_type: "Add".into(),
            inputs: vec![
                tensor("a", DType::F32, &[2, 2]),
                tensor("b", DType::F32, &[2, 2]),
            ],
            outputs: vec![out("c", DType::F32, &[2, 2])],
            ..Default::default()
        };
        let mut ctx = Recorder::default();
        ew_binary(spec, &node, &mut ctx).expect("translate");
        let pc = &ctx.dispatches[0].push_constants;
        assert!(!pc.is_empty() && pc.len() <= 128);
        assert_eq!(u32::from_le_bytes(pc[4..8].try_into().unwrap()), 4);
    }

    #[test]
    fn a_missing_shape_is_an_error_not_a_panic() {
        let spec = spec_named("Add");
        let node = NodeDesc {
            op_type: "Add".into(),
            inputs: vec![
                TensorRef {
                    name: "a".into(),
                    desc: None,
                    is_initializer: false,
                },
                tensor("b", DType::F32, &[4]),
            ],
            outputs: vec![out("c", DType::F32, &[4])],
            ..Default::default()
        };
        let mut ctx = Recorder::default();
        let err = ew_binary(spec, &node, &mut ctx).unwrap_err();
        assert!(matches!(err, EpError::Unsupported(_)), "{err}");
        assert!(
            ctx.dispatches.is_empty(),
            "nothing may be recorded on error"
        );
    }

    #[test]
    fn mixed_dtypes_are_an_error_not_a_panic() {
        let spec = spec_named("Add");
        let node = NodeDesc {
            op_type: "Add".into(),
            inputs: vec![tensor("a", DType::F32, &[4]), tensor("b", DType::I32, &[4])],
            outputs: vec![out("c", DType::F32, &[4])],
            ..Default::default()
        };
        let mut ctx = Recorder::default();
        assert!(ew_binary(spec, &node, &mut ctx).is_err());
    }

    #[test]
    fn non_broadcastable_shapes_are_an_error_not_a_panic() {
        let spec = spec_named("Add");
        let node = NodeDesc {
            op_type: "Add".into(),
            inputs: vec![
                tensor("a", DType::F32, &[2, 3]),
                tensor("b", DType::F32, &[2, 4]),
            ],
            outputs: vec![out("c", DType::F32, &[2, 3])],
            ..Default::default()
        };
        let mut ctx = Recorder::default();
        assert!(ew_binary(spec, &node, &mut ctx).is_err());
    }

    #[test]
    fn the_refusing_handler_refuses() {
        let spec = spec_named("Add");
        let node = NodeDesc {
            op_type: "Add".into(),
            ..Default::default()
        };
        let mut ctx = Recorder::default();
        assert!(unimplemented(spec, &node, &mut ctx).is_err());
    }

    #[test]
    fn every_shader_row_translates_to_a_stem_that_is_in_the_manifest() {
        // The end-to-end coherence check: table row -> handler -> variant -> build manifest.
        let manifest = super::super::variants::manifest();
        for spec in crate::registry::all_specs() {
            for d in spec.caps.iter() {
                if let Some(stem) = spec.kernel.stem(d) {
                    assert!(
                        manifest.iter().any(|v| v.stem == stem),
                        "`{}` would dispatch `{stem}`, which the build never produces",
                        spec.op_type
                    );
                }
            }
        }
    }
}
