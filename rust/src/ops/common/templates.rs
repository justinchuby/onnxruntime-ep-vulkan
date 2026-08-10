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
    AttrValue, DType, DispatchContext, EpError, EpResult, KernelRequest, NodeDesc, TensorDesc,
};
use crate::registry::OpSpec;

use super::params;
use super::selector;
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

    // The selector is pushed only for the ops that declare one. Vulkan ignores a map entry for a
    // constant_id the module does not use, so pushing it unconditionally would be legal — but it
    // would also make every elementwise pipeline key three-wide, and the key is what the counters
    // report as `pipeline_variants`. Pushing what the op actually has keeps that number readable.
    let mut spec_constants = vec![EW_LOCAL_SIZE, u32::from(plan.all_identical)];
    if selector::source_for(&node.op_type).is_some() {
        spec_constants.push(selector::resolve(&node.op_type, node).map_err(|e| {
            // Same argument as the parameter tail below: claim already validated this, so a
            // disagreement here is an internal inconsistency, not a graph we should have declined.
            EpError::Internal(format!("`{}` {e}", node.op_type))
        })?);
    }

    ctx.dispatch(KernelRequest {
        shader,
        spec_constants,
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

/// Translate `Clip` — the ternary module, whose bounds may be absent.
///
/// Unlike `Where`, whose three inputs are always present and whose first is `bool`, `Clip`'s `min`
/// and `max` are optional inputs. The module always declares three bindings, so an absent bound's
/// binding is filled with **input 0** as an inert placeholder — the same device `q_gemv` uses for
/// a missing `zero_points`. `EW_SELECTOR` tells the shader which of bindings 1 and 2 are real, and
/// the guard it compiles means the placeholder is never read.
///
/// Binding the value tensor rather than allocating a dummy is deliberate: it costs no allocation,
/// it cannot be out of range for any index the shader could compute (it is the same buffer the
/// value comes from, with the same element count), and there is no sentinel to get wrong at a
/// dtype that has no infinity.
pub fn ew_clip(spec: &OpSpec, node: &NodeDesc, ctx: &mut dyn DispatchContext) -> EpResult<()> {
    let sel = selector::resolve(&node.op_type, node)
        .map_err(|e| EpError::Internal(format!("`{}` {e}", node.op_type)))?;

    // Which node input backs each of the three bindings. An absent bound reads input 0.
    let sources = [
        0usize,
        if sel & 1 != 0 { 1 } else { 0 },
        if sel & 2 != 0 { 2 } else { 0 },
    ];

    let mut shapes = Vec::with_capacity(sources.len());
    for &i in &sources {
        let t = node.inputs.get(i).ok_or_else(|| {
            EpError::InvalidGraph(format!(
                "`{}` was claimed with selector {sel:#b} but input {i} is missing at compile time",
                node.op_type
            ))
        })?;
        let desc = t.desc.as_ref().ok_or_else(|| {
            EpError::Unsupported(format!(
                "`{}` input {i} (`{}`) has no shape at compile time",
                node.op_type, t.name
            ))
        })?;
        shapes.push(desc.shape.clone());
    }
    let refs: Vec<&[i64]> = shapes.iter().map(Vec::as_slice).collect();
    let plan = ShapePlan::broadcast(&refs).map_err(|e| {
        EpError::Unsupported(format!("`{}` shapes cannot be planned: {e}", node.op_type))
    })?;

    let dtype = node.inputs[0]
        .desc
        .as_ref()
        .map(|d| d.dtype)
        .ok_or_else(|| {
            EpError::Unsupported(format!(
                "`{}` input 0 has no element type at compile time",
                node.op_type
            ))
        })?;
    let shader = spec.kernel.stem(dtype).ok_or_else(|| {
        EpError::Internal(format!(
            "`{}` was claimed but its row declares no shader",
            node.op_type
        ))
    })?;

    let mut bindings = Vec::with_capacity(4);
    for &i in &sources {
        bindings.push(ctx.resolve(&node.inputs[i])?);
    }

    let out = single_output(node)?;
    let out_dtype = out.desc.as_ref().map_or(dtype, |d| d.dtype);
    let out_buf = ctx.bind_output(out, TensorDesc::new(out_dtype, plan.out_dims()))?;
    bindings.push(out_buf);

    ctx.dispatch(KernelRequest {
        shader,
        spec_constants: vec![EW_LOCAL_SIZE, u32::from(plan.all_identical), sel],
        push_constants: plan.push_constants_with_params(super::shape_plan::EW_PARAMS_NONE),
        bindings,
        workgroups: plan.workgroups_1d(EW_LOCAL_SIZE),
    })
}

/// The destination element type of a `Cast` node: the output edge if it is typed, else `to`.
///
/// See [`ew_cast`] for why the attribute path exists. Both present and disagreeing is refused;
/// neither present is refused with a message that names both places that were looked.
fn cast_destination(node: &NodeDesc, out: &crate::engine::OutRef) -> EpResult<DType> {
    let from_edge = out.desc.as_ref().map(|d| d.dtype);
    let from_attr = match node.attributes.get("to") {
        Some(AttrValue::Int(v)) => Some(*v),
        _ => None,
    };
    let mapped_attr = from_attr.and_then(crate::registry::dtype_from_onnx_value);

    match (from_edge, from_attr, mapped_attr) {
        (Some(edge), _, Some(attr)) if edge != attr => Err(EpError::Unsupported(format!(
            "`{}` output edge says {edge:?} and its `to` attribute says {attr:?}; the graph and \
             ONNX Runtime's shape inference disagree about this node's output type and this EP \
             will not choose between them",
            node.op_type
        ))),
        (Some(edge), _, _) => Ok(edge),
        (None, _, Some(attr)) => Ok(attr),
        // `to` present but naming a type this EP has no storage for. The claim predicate rejects
        // that case on the edge, so reaching it here means the edge was dropped *and* the type is
        // one we cannot hold — a different fact from "no type at all", and it says which.
        (None, Some(raw), None) => Err(EpError::Unsupported(format!(
            "`{}` casts to ONNX element type {raw}, which this EP has no storage for",
            node.op_type
        ))),
        (None, None, _) => Err(EpError::Unsupported(format!(
            "`{}` output has no element type: the output edge carries none (ONNX Runtime drops \
             it together with a symbolic shape) and the node has no `to` attribute",
            node.op_type
        ))),
    }
}

/// Translate `Cast` — the one template whose module is chosen by a dtype **pair**.
///
/// The destination type is read from the *output edge* when the edge has one, and from the `to`
/// attribute when it does not. They say the same thing when the graph is well formed, and ONNX
/// Runtime has already run shape inference by the time we are asked to compile, so the edge is
/// the resolved answer and `to` is the request; `claim::cast` checks the edge, and preferring
/// the edge here keeps the claim and the translate reading the same thing.
///
/// THE FALLBACK IS NOT DEFENSIVE, IT IS THE ONLY SOURCE ON A DYNAMIC GRAPH (found 2026-08-03).
/// `ep::tensor_desc` returns `None` for an edge with **any** symbolic extent, and it drops the
/// dtype with the shape. At `Compile` that costs nothing — ORT hands the claim predicate a
/// typed `OrtValueInfo` and the edge type is read from there — but the Compute-time dynamic
/// re-translate rebuilds the node from `NodeDesc`, whose outputs carry the dropped `desc`. So
/// every `runtime-extent` `Cast` was claimed and then failed its re-run with "output has no
/// element type at compile time" — a broken commitment, on 98 of gpt-oss-20b's 374 nodes. The
/// destination type of a `Cast` is a node attribute; a node attribute does not stop existing
/// because an extent is unknown, and this handler must not behave as though it does.
///
/// A disagreement between the two is refused rather than resolved. `to` and the inferred edge
/// differing means the graph and ORT's inference disagree about the node's output type, and
/// picking a winner would put an unannounced reinterpretation of the model inside a dispatch.
pub fn ew_cast(spec: &OpSpec, node: &NodeDesc, ctx: &mut dyn DispatchContext) -> EpResult<()> {
    let shapes = input_shapes(node, 1)?;
    let refs: Vec<&[i64]> = shapes.iter().map(Vec::as_slice).collect();
    let plan = ShapePlan::broadcast(&refs).map_err(|e| {
        EpError::Unsupported(format!("`{}` shapes cannot be planned: {e}", node.op_type))
    })?;

    let src = common_dtype(node, 0, 1)?;
    let out = single_output(node)?;
    let dst = cast_destination(node, out)?;

    let shader = spec.kernel.pair_stem(src, dst).ok_or_else(|| {
        EpError::Internal(format!(
            "`{}` was claimed but its row declares no pair-keyed shader",
            node.op_type
        ))
    })?;

    let bindings = vec![
        ctx.resolve(&node.inputs[0])?,
        ctx.bind_output(out, TensorDesc::new(dst, plan.out_dims()))?,
    ];

    ctx.dispatch(KernelRequest {
        shader,
        spec_constants: vec![EW_LOCAL_SIZE, u32::from(plan.all_identical)],
        push_constants: plan.push_constants_with_params(super::shape_plan::EW_PARAMS_NONE),
        bindings,
        workgroups: plan.workgroups_1d(EW_LOCAL_SIZE),
    })
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
    // The bound is read from the same constant the claim predicate uses, so the two cannot
    // drift apart. They were separate numbers until 2026-08-02 — the predicate said 8, this
    // said 2 — and the gap was an `EP_FAIL` at session creation for any 3-input node: claimed,
    // then untranslatable. A shared constant makes the invariant hold by construction rather
    // than by a test that has to remember to exist.
    if n <= crate::ops::common::claim::MAX_VARIADIC_INPUTS_LOWERED {
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

// ── Norm family ──────────────────────────────────────────────────────────────────────────────

/// Workgroup size the norm template is compiled with.
///
/// Shared with the EW constant (both 256) because the portability argument is the same: 256
/// invocations is the §7.2 floor guarantee, and the tree-reduction scratch at 256 × 4 bytes
/// (1 KiB) is safely within the 16 KiB shared-memory floor.  The local size must remain a
/// power of two — the tree-reduction algorithm depends on it.
pub const NORM_LOCAL_SIZE: u32 = 256;

/// Default epsilon for RMSNorm variants, matching the ONNX schema default.
const NORM_EPSILON_DEFAULT: f32 = 1e-5;

/// Translate `SkipSimplifiedLayerNormalization` — fused residual-add + RMSNorm.
///
/// # Outputs
///
/// The ORT schema declares four output slots; only 0 and 3 matter:
///
/// | slot | name           | always bound? |
/// |------|----------------|---------------|
/// | 0    | normalised out | yes           |
/// | 1    | mean           | almost never  |
/// | 2    | inv_std_var    | almost never  |
/// | 3    | residual sum   | yes (feeds next block) |
///
/// Slots 1 and 2 are empty strings in every node in both Phi-3.5 and gpt-oss (census
/// `OP_COVERAGE.md §4.21`).  When slot 3 is absent the translate handler allocates a scratch
/// buffer so the shader always has a place to write — the result is discarded, which is
/// wasteful but correct and avoids a shader variant.
///
/// # Shader
///
/// The direct shader (`shaders/glsl/skip_simplified_layer_norm_f32.comp`) is not wired into
/// the manifest system; `build.rs` picks it up by directory scan.  The stem
/// `skip_simplified_layer_norm_f32` is therefore a string literal here rather than coming from
/// `spec.kernel.stem(dtype)`.  A test below asserts the file exists so the mismatch surfaces as
/// a unit-test failure rather than a build panic.
pub fn skip_norm(_spec: &OpSpec, node: &NodeDesc, ctx: &mut dyn DispatchContext) -> EpResult<()> {
    // -- Validate and extract shape info --------------------------------------------------

    // We expect at least 3 inputs: hidden (0), skip (1), gamma (2).
    if node.inputs.len() < 3 {
        return Err(EpError::InvalidGraph(format!(
            "`{}` needs at least 3 inputs (hidden, skip, gamma), got {}",
            node.op_type,
            node.inputs.len()
        )));
    }

    let dtype = common_dtype(node, 0, 3)?;

    let hidden_shape = node.inputs[0]
        .desc
        .as_ref()
        .map(|d| &d.shape)
        .ok_or_else(|| {
            EpError::Unsupported(format!(
                "`{}` input 0 has no shape at compile time",
                node.op_type
            ))
        })?;

    if hidden_shape.is_empty() {
        return Err(EpError::Unsupported(format!(
            "`{}` requires a non-scalar input",
            node.op_type
        )));
    }
    let rank = hidden_shape.len();
    let hidden_size = hidden_shape[rank - 1] as u32;
    let batch_count: u32 = hidden_shape[..rank - 1].iter().map(|&d| d as u32).product();
    // product() on empty slice (rank==1) = 1, which is correct.

    // -- Select shader stem by dtype ------------------------------------------------------
    let shader: &'static str = match dtype {
        DType::F32 => "skip_simplified_layer_norm_f32",
        DType::F16 => "skip_simplified_layer_norm_f16",
        _ => {
            return Err(EpError::Unsupported(format!(
                "`{}` dtype {dtype:?} is not supported by the norm kernel",
                node.op_type
            )));
        }
    };

    // -- Epsilon attribute ----------------------------------------------------------------
    let eps: f32 = match node.attributes.get("epsilon") {
        Some(AttrValue::Float(v)) => *v,
        None => NORM_EPSILON_DEFAULT,
        Some(_) => {
            return Err(EpError::Unsupported(format!(
                "`{}` `epsilon` attribute is not a float",
                node.op_type
            )));
        }
    };

    // -- Bind inputs ----------------------------------------------------------------------
    let hidden_buf = ctx.resolve(&node.inputs[0])?;
    let skip_buf = ctx.resolve(&node.inputs[1])?;
    let gamma_buf = ctx.resolve(&node.inputs[2])?;

    // -- Bind or allocate outputs ---------------------------------------------------------
    let out_desc = TensorDesc::new(dtype, hidden_shape.clone());

    // Slot 0: normalised output — always present when the op is claimed.
    let out0 = node
        .outputs
        .first()
        .ok_or_else(|| EpError::InvalidGraph(format!("`{}` has no outputs", node.op_type)))?;
    let out0_buf = ctx.bind_output(out0, out_desc.clone())?;

    // Slot 3: residual sum (pre-norm) — feeds the next block in LLM graphs.  When the slot is
    // absent in a given node, allocate a scratch buffer so the shader always has a valid
    // binding.  The write is wasted but correct; no variant needed.
    let out3_buf = match node.outputs.get(3).filter(|o| !o.name.is_empty()) {
        Some(out3) => ctx.bind_output(out3, out_desc)?,
        None => ctx.alloc_temp(out_desc)?,
    };

    // -- Push constants -------------------------------------------------------------------
    // Layout (little-endian):
    //   offset  0: batch_count (u32)
    //   offset  4: hidden_size (u32)
    //   offset  8: eps_bits    (u32)  — uintBitsToFloat(eps_bits) == epsilon
    //   offset 12: _pad        (u32)
    let mut push = Vec::with_capacity(16);
    push.extend_from_slice(&batch_count.to_le_bytes());
    push.extend_from_slice(&hidden_size.to_le_bytes());
    push.extend_from_slice(&eps.to_bits().to_le_bytes());
    push.extend_from_slice(&0u32.to_le_bytes()); // padding

    // -- Dispatch -------------------------------------------------------------------------
    ctx.dispatch(KernelRequest {
        shader,
        spec_constants: vec![NORM_LOCAL_SIZE],
        push_constants: push,
        bindings: vec![hidden_buf, skip_buf, gamma_buf, out0_buf, out3_buf],
        workgroups: [batch_count.max(1), 1, 1],
    })
}

/// Translate `SimplifiedLayerNormalization` / `RMSNormalization` — RMSNorm, no residual fuse.
///
/// # Why this is not `skip_norm` with a zero skip buffer
///
/// It would be one fewer shader, and it would cost a full extra activation-sized read per row on
/// a kernel that is entirely bandwidth-bound. The fusion allowlist in `OP_COVERAGE.md` §5.6
/// exists because these norms are memory-traffic-limited, so paying an extra pass over hidden
/// bytes to save a shader file inverts the reason the fused form was allowlisted at all.
///
/// # Inputs and outputs
///
/// | slot | name | notes |
/// |------|------|-------|
/// | in 0 | X | `FLOAT16`/`FLOAT`, rank ≥ 1, normalised over the last axis |
/// | in 1 | gamma (scale) | same dtype as X, length = last dim |
/// | out 0 | normalised | always present |
/// | out 1 | inv_std_var | optional; **declined** by the claim predicate when requested |
///
/// The optional second output is declined rather than written to a scratch buffer because,
/// unlike `SkipSimplifiedLayerNormalization` slot 3, nothing in the target graphs asks for it —
/// so a scratch write would be pure waste on every row, and a shader variant for a case with no
/// observed instance is coverage we cannot exercise.
///
/// # Shader
///
/// `shaders/glsl/simplified_layer_norm_{f32,f16}.comp`, picked up by `build.rs`'s directory
/// scan. The stem is a literal here for the same reason `skip_norm`'s is; tests below assert
/// both files exist so a rename surfaces as a unit-test failure, not a build panic.
pub fn simplified_norm(
    _spec: &OpSpec,
    node: &NodeDesc,
    ctx: &mut dyn DispatchContext,
) -> EpResult<()> {
    if node.inputs.len() < 2 {
        return Err(EpError::InvalidGraph(format!(
            "`{}` needs at least 2 inputs (X, gamma), got {}",
            node.op_type,
            node.inputs.len()
        )));
    }

    let dtype = common_dtype(node, 0, 2)?;

    let hidden_shape = node.inputs[0]
        .desc
        .as_ref()
        .map(|d| &d.shape)
        .ok_or_else(|| {
            EpError::Unsupported(format!(
                "`{}` input 0 has no shape at compile time",
                node.op_type
            ))
        })?;

    if hidden_shape.is_empty() {
        return Err(EpError::Unsupported(format!(
            "`{}` requires a non-scalar input",
            node.op_type
        )));
    }
    let rank = hidden_shape.len();
    let hidden_size = hidden_shape[rank - 1] as u32;
    let batch_count: u32 = hidden_shape[..rank - 1].iter().map(|&d| d as u32).product();

    let shader: &'static str = match dtype {
        DType::F32 => "simplified_layer_norm_f32",
        DType::F16 => "simplified_layer_norm_f16",
        _ => {
            return Err(EpError::Unsupported(format!(
                "`{}` dtype {dtype:?} is not supported by the norm kernel",
                node.op_type
            )));
        }
    };

    let eps: f32 = match node.attributes.get("epsilon") {
        Some(AttrValue::Float(v)) => *v,
        None => NORM_EPSILON_DEFAULT,
        Some(_) => {
            return Err(EpError::Unsupported(format!(
                "`{}` `epsilon` attribute is not a float",
                node.op_type
            )));
        }
    };

    let hidden_buf = ctx.resolve(&node.inputs[0])?;
    let gamma_buf = ctx.resolve(&node.inputs[1])?;

    let out_desc = TensorDesc::new(dtype, hidden_shape.clone());
    let out0 = node
        .outputs
        .first()
        .ok_or_else(|| EpError::InvalidGraph(format!("`{}` has no outputs", node.op_type)))?;
    let out0_buf = ctx.bind_output(out0, out_desc)?;

    // Push constants: identical layout to `skip_norm`, so both norm shaders share one encoding.
    let mut push = Vec::with_capacity(16);
    push.extend_from_slice(&batch_count.to_le_bytes());
    push.extend_from_slice(&hidden_size.to_le_bytes());
    push.extend_from_slice(&eps.to_bits().to_le_bytes());
    push.extend_from_slice(&0u32.to_le_bytes());

    ctx.dispatch(KernelRequest {
        shader,
        spec_constants: vec![NORM_LOCAL_SIZE],
        push_constants: push,
        bindings: vec![hidden_buf, gamma_buf, out0_buf],
        workgroups: [batch_count.max(1), 1, 1],
    })
}

// ── Indexing family ──────────────────────────────────────────────────────────────────────────

/// Workgroup size the gather kernel is compiled with.
pub const GATHER_LOCAL_SIZE: u32 = 256;

/// Cap on the workgroup count in X.
///
/// The gather shaders use a grid-stride loop, so this is a scheduling choice rather than a
/// correctness one. It is set well below the Vulkan `maxComputeWorkGroupCount[0]` floor (65535)
/// so no device can reject the dispatch, and high enough to saturate both local GPUs.
pub const GATHER_MAX_WORKGROUPS: u32 = 4096;

/// Translate `Gather` — ONNX gather along one axis, any indices rank.
///
/// # The flattening
///
/// Every `Gather` is the same three-extent loop once the data tensor is folded around its axis:
///
/// ```text
/// outer    = prod(data.shape[:axis])
/// gathered = data.shape[axis]
/// inner    = prod(data.shape[axis+1:])
/// n_idx    = prod(indices.shape)
/// out[o, j, i] = data[o, indices[j], i]
/// ```
///
/// so one shader covers every axis and every indices rank, and the only per-node work here is
/// arithmetic. The output shape is `data.shape[:axis] ++ indices.shape ++ data.shape[axis+1:]`.
///
/// # Index dtype
///
/// `indices` is int32 or int64. Both are read through a uint buffer with a word stride of 1 or
/// 2 — see the shader header for why reading the low word of an int64 is exact for every index
/// the ONNX schema defines, and why that avoids depending on the `shaderInt64` device feature.
pub fn gather(_spec: &OpSpec, node: &NodeDesc, ctx: &mut dyn DispatchContext) -> EpResult<()> {
    if node.inputs.len() < 2 {
        return Err(EpError::InvalidGraph(format!(
            "`{}` needs 2 inputs (data, indices), got {}",
            node.op_type,
            node.inputs.len()
        )));
    }

    let data_desc = node.inputs[0].desc.as_ref().ok_or_else(|| {
        EpError::Unsupported(format!(
            "`{}` input 0 has no shape at compile time",
            node.op_type
        ))
    })?;
    let idx_desc = node.inputs[1].desc.as_ref().ok_or_else(|| {
        EpError::Unsupported(format!(
            "`{}` input 1 has no shape at compile time",
            node.op_type
        ))
    })?;

    let dtype = data_desc.dtype;
    let data_shape = &data_desc.shape;
    let rank = data_shape.len();
    if rank == 0 {
        return Err(EpError::Unsupported(format!(
            "`{}` cannot gather from a scalar",
            node.op_type
        )));
    }

    let raw_axis = match node.attributes.get("axis") {
        Some(AttrValue::Int(v)) => *v,
        None => 0,
        Some(_) => {
            return Err(EpError::Unsupported(format!(
                "`{}` `axis` attribute is not an int",
                node.op_type
            )));
        }
    };
    let axis = if raw_axis < 0 {
        raw_axis + rank as i64
    } else {
        raw_axis
    };
    if axis < 0 || axis >= rank as i64 {
        return Err(EpError::InvalidGraph(format!(
            "`{}` axis {raw_axis} is out of range for a rank-{rank} input",
            node.op_type
        )));
    }
    let axis = axis as usize;

    let outer: u32 = data_shape[..axis].iter().map(|&d| d as u32).product();
    let gathered = data_shape[axis] as u32;
    let inner: u32 = data_shape[axis + 1..].iter().map(|&d| d as u32).product();
    let n_idx: u32 = idx_desc.shape.iter().map(|&d| d as u32).product();

    let shader: &'static str = match dtype {
        DType::F32 => "gather_f32",
        DType::F16 => "gather_f16",
        _ => {
            return Err(EpError::Unsupported(format!(
                "`{}` data dtype {dtype:?} is not supported by the gather kernel",
                node.op_type
            )));
        }
    };

    let idx_stride_words: u32 = match idx_desc.dtype {
        DType::I64 => 2,
        DType::I32 => 1,
        other => {
            return Err(EpError::Unsupported(format!(
                "`{}` indices dtype {other:?} is not int32 or int64",
                node.op_type
            )));
        }
    };

    // Output shape: data.shape[:axis] ++ indices.shape ++ data.shape[axis+1:].
    let mut out_shape: Vec<i64> = Vec::with_capacity(rank - 1 + idx_desc.shape.len());
    out_shape.extend_from_slice(&data_shape[..axis]);
    out_shape.extend_from_slice(&idx_desc.shape);
    out_shape.extend_from_slice(&data_shape[axis + 1..]);

    let total = outer
        .checked_mul(n_idx)
        .and_then(|v| v.checked_mul(inner))
        .ok_or_else(|| {
            EpError::Unsupported(format!(
                "`{}` output element count overflows u32",
                node.op_type
            ))
        })?;

    let data_buf = ctx.resolve(&node.inputs[0])?;
    let idx_buf = ctx.resolve(&node.inputs[1])?;
    let out = single_output(node)?;
    let out_buf = ctx.bind_output(out, TensorDesc::new(dtype, out_shape))?;

    let mut push = Vec::with_capacity(24);
    for v in [outer, gathered, inner, n_idx, idx_stride_words, total] {
        push.extend_from_slice(&v.to_le_bytes());
    }

    // f16 packs two elements per word, so the thread count is halved.
    let work_items = if dtype == DType::F16 {
        total.div_ceil(2)
    } else {
        total
    };
    let groups = work_items
        .div_ceil(GATHER_LOCAL_SIZE)
        .clamp(1, GATHER_MAX_WORKGROUPS);

    ctx.dispatch(KernelRequest {
        shader,
        spec_constants: vec![GATHER_LOCAL_SIZE],
        push_constants: push,
        bindings: vec![data_buf, idx_buf, out_buf],
        workgroups: [groups, 1, 1],
    })
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
        //
        // Widened 2026-08-04 to cover **hand-written** modules. A `kernel!(Standalone, ..)` row
        // names a `.comp` that `build.rs` compiles straight out of `shaders/glsl` rather than a
        // variant the manifest generates, so the manifest is the wrong place to look for it — but
        // the property being asserted is the same one, and it is the property that failed when
        // `Conv` said `kernel!(None)` while dispatching `conv_f32`: *the module this row names
        // must be something the build actually produces*.
        //
        // The source file is checked rather than `SHADER_MODULES`, because `SHADER_MODULES` is
        // empty in a build with no `glslc` and the assertion would then pass by having nothing to
        // check on exactly the build it exists to catch.
        let manifest = super::super::variants::manifest();
        let glsl = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("shaders")
            .join("glsl");
        for spec in crate::registry::all_specs() {
            for d in spec.caps.iter() {
                for stem in spec.kernel.dispatch_stems(d) {
                    let generated = manifest.iter().any(|v| v.stem == stem);
                    let handwritten = glsl.join(format!("{stem}.comp")).is_file();
                    assert!(
                        generated || handwritten,
                        "`{}` would dispatch `{stem}` at {d:?}, which the build never produces: \
                         it is in neither the variant manifest nor shaders/glsl/{stem}.comp",
                        spec.op_type
                    );
                }
            }
        }
    }

    /// Every hand-written `.comp` is named by a row, and every name is a real file.
    ///
    /// The converse of the test above, and the one that keeps the `metadata` defect from coming
    /// back one row at a time: a `.comp` nobody's row names is a module the proof key can never
    /// mention, which is precisely the state `conv_f32.comp` shipped in.
    ///
    /// `dispatch_stems` rather than `stem` since #90: a row may reach a second hand-written
    /// module through its own selector, and that module must be *named by the row* for the same
    /// reason the first one is. Adding `gqa_decode_f16.comp` without declaring it would have
    /// failed here — which is the whole point of the converse direction.
    #[test]
    fn every_hand_written_shader_is_named_by_a_row() {
        let glsl = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("shaders")
            .join("glsl");
        let mut named = std::collections::BTreeSet::new();
        for spec in crate::registry::all_specs() {
            for d in spec.caps.iter() {
                for stem in spec.kernel.dispatch_stems(d) {
                    named.insert(stem);
                }
            }
        }
        for entry in std::fs::read_dir(&glsl).expect("shaders/glsl must be readable") {
            let path = entry.expect("readable dir entry").path();
            if path.extension().and_then(|e| e.to_str()) != Some("comp") {
                continue;
            }
            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .expect("a .comp file has a UTF-8 stem")
                .to_string();
            assert!(
                named.contains(stem.as_str()),
                "shaders/glsl/{stem}.comp is compiled by the build but no registry row names it, \
                 so no proof key can ever carry it"
            );
        }
    }

    // ── skip_norm tests ──────────────────────────────────────────────────────────────────────

    fn skip_norm_node_f32(batch: i64, seq: i64, hidden: i64) -> NodeDesc {
        NodeDesc {
            op_type: "SkipSimplifiedLayerNormalization".into(),
            inputs: vec![
                tensor("hidden", DType::F32, &[batch, seq, hidden]),
                tensor("skip", DType::F32, &[batch, seq, hidden]),
                tensor("gamma", DType::F32, &[hidden]),
            ],
            outputs: vec![
                out("out0", DType::F32, &[batch, seq, hidden]), // slot 0
                OutRef {
                    name: String::new(),
                    desc: None,
                }, // slot 1 (empty)
                OutRef {
                    name: String::new(),
                    desc: None,
                }, // slot 2 (empty)
                out("out3", DType::F32, &[batch, seq, hidden]), // slot 3
            ],
            ..Default::default()
        }
    }

    #[test]
    fn skip_norm_f32_produces_one_dispatch_with_correct_shader() {
        let spec = spec_named("SkipSimplifiedLayerNormalization");
        let node = skip_norm_node_f32(2, 4, 64);
        let mut ctx = Recorder::default();
        skip_norm(spec, &node, &mut ctx).expect("translate");

        assert_eq!(ctx.dispatches.len(), 1);
        let k = &ctx.dispatches[0];
        assert_eq!(k.shader, "skip_simplified_layer_norm_f32");
        assert_eq!(k.spec_constants, vec![NORM_LOCAL_SIZE]);
        assert_eq!(k.workgroups, [8, 1, 1], "batch×seq = 2×4 = 8 rows");
        assert_eq!(k.bindings.len(), 5, "hidden, skip, gamma, out0, out3");
        assert_eq!(
            ctx.outputs.len(),
            2,
            "bind_output called for slot-0 and slot-3"
        );
        assert_eq!(ctx.outputs[0].shape, vec![2, 4, 64]);
    }

    #[test]
    fn skip_norm_f32_push_constants_encode_shape_and_epsilon() {
        let spec = spec_named("SkipSimplifiedLayerNormalization");
        let mut node = skip_norm_node_f32(1, 8, 128);
        // Insert a custom epsilon attribute.
        node.attributes
            .insert("epsilon".into(), AttrValue::Float(1e-6));
        let mut ctx = Recorder::default();
        skip_norm(spec, &node, &mut ctx).unwrap();
        let pc = &ctx.dispatches[0].push_constants;

        assert_eq!(pc.len(), 16, "fixed 16-byte push constant block");
        let batch_count = u32::from_le_bytes(pc[0..4].try_into().unwrap());
        let hidden_size = u32::from_le_bytes(pc[4..8].try_into().unwrap());
        let eps_bits = u32::from_le_bytes(pc[8..12].try_into().unwrap());
        let pad = u32::from_le_bytes(pc[12..16].try_into().unwrap());
        assert_eq!(batch_count, 8, "1×8 rows");
        assert_eq!(hidden_size, 128);
        assert_eq!(f32::from_bits(eps_bits), 1e-6_f32);
        assert_eq!(pad, 0);
    }

    #[test]
    fn skip_norm_absent_slot3_uses_a_temp_buffer() {
        // When output slot 3 is not requested, the handler must allocate a scratch buffer so
        // the shader always has a valid binding.
        let spec = spec_named("SkipSimplifiedLayerNormalization");
        let mut node = skip_norm_node_f32(1, 1, 32);
        // Remove slot 3.
        node.outputs.truncate(1);
        let mut ctx = Recorder::default();
        skip_norm(spec, &node, &mut ctx).unwrap();

        let k = &ctx.dispatches[0];
        assert_eq!(k.bindings.len(), 5, "always 5 bindings even without slot-3");
        // ctx.outputs includes both bind_output and alloc_temp calls (Recorder design).
        // Check that exactly one is from slot-0 (bind_output) and one from the scratch (alloc_temp).
        assert_eq!(
            ctx.outputs.len(),
            2,
            "one bind_output (slot 0) + one alloc_temp (slot 3 scratch)"
        );
    }

    #[test]
    fn skip_norm_default_epsilon_matches_onnx_schema() {
        // A missing `epsilon` attribute should resolve to 1e-5, the ONNX schema default.
        let spec = spec_named("SkipSimplifiedLayerNormalization");
        let node = skip_norm_node_f32(1, 1, 16);
        let mut ctx = Recorder::default();
        skip_norm(spec, &node, &mut ctx).unwrap();

        let pc = &ctx.dispatches[0].push_constants;
        let eps_bits = u32::from_le_bytes(pc[8..12].try_into().unwrap());
        assert_eq!(
            f32::from_bits(eps_bits),
            NORM_EPSILON_DEFAULT,
            "default epsilon is 1e-5"
        );
    }

    #[test]
    fn skip_norm_rank1_input_yields_batch_count_one() {
        // A single 1-D row (e.g. during unit testing with [hidden_size] input).
        let spec = spec_named("SkipSimplifiedLayerNormalization");
        let node = NodeDesc {
            op_type: "SkipSimplifiedLayerNormalization".into(),
            inputs: vec![
                tensor("hidden", DType::F32, &[256]),
                tensor("skip", DType::F32, &[256]),
                tensor("gamma", DType::F32, &[256]),
            ],
            outputs: vec![out("out0", DType::F32, &[256])],
            ..Default::default()
        };
        let mut ctx = Recorder::default();
        skip_norm(spec, &node, &mut ctx).unwrap();
        assert_eq!(ctx.dispatches[0].workgroups, [1, 1, 1]);
    }

    /// The shader file that `skip_norm` names must exist on disk.
    ///
    /// `skip_norm` hardcodes the stem rather than routing through `spec.kernel.stem()`, so the
    /// manifest system never checks it.  This test fills that gap: a wrong stem here is a
    /// unit-test failure rather than a build panic on a different machine.
    #[test]
    fn skip_norm_f32_shader_exists_on_disk() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("shaders")
            .join("glsl")
            .join("skip_simplified_layer_norm_f32.comp");
        assert!(
            path.is_file(),
            "shaders/glsl/skip_simplified_layer_norm_f32.comp is missing; \
             the translate handler names it directly so it must exist at build time"
        );
    }

    #[test]
    fn skip_norm_f16_shader_exists_on_disk() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("shaders")
            .join("glsl")
            .join("skip_simplified_layer_norm_f16.comp");
        assert!(
            path.is_file(),
            "shaders/glsl/skip_simplified_layer_norm_f16.comp is missing; \
             the translate handler names it directly so it must exist at build time"
        );
    }

    #[test]
    fn skip_norm_f16_produces_one_dispatch_with_correct_shader() {
        // SkipSimplifiedLayerNormalization in fp16 — the path Phi-3.5 exercises.
        let spec = spec_named("SkipSimplifiedLayerNormalization");
        let node = NodeDesc {
            op_type: "SkipSimplifiedLayerNormalization".into(),
            inputs: vec![
                tensor("hidden", DType::F16, &[2, 4, 64]),
                tensor("skip", DType::F16, &[2, 4, 64]),
                tensor("gamma", DType::F16, &[64]),
            ],
            outputs: vec![
                out("out0", DType::F16, &[2, 4, 64]),
                OutRef {
                    name: String::new(),
                    desc: None,
                },
                OutRef {
                    name: String::new(),
                    desc: None,
                },
                out("out3", DType::F16, &[2, 4, 64]),
            ],
            ..Default::default()
        };
        let mut ctx = Recorder::default();
        skip_norm(spec, &node, &mut ctx).expect("translate f16");

        let k = &ctx.dispatches[0];
        assert_eq!(ctx.dispatches.len(), 1);
        assert_eq!(k.shader, "skip_simplified_layer_norm_f16");
        assert_eq!(k.workgroups, [8, 1, 1], "batch×seq = 2×4 = 8 rows");
        assert_eq!(k.bindings.len(), 5, "hidden, skip, gamma, out0, out3");
    }

    /// The f16 check is a *yesterday-red* test: before the f16 shader and handler path were
    /// written, `skip_norm` returned `Err(Unsupported)` for DType::F16.  This test would have
    /// failed on the unfixed code.
    #[test]
    fn skip_norm_f16_was_unsupported_before_this_commit() {
        // Verify that the f16 path is now supported (the test name documents what it caught).
        let spec = spec_named("SkipSimplifiedLayerNormalization");
        let node = NodeDesc {
            op_type: "SkipSimplifiedLayerNormalization".into(),
            inputs: vec![
                tensor("hidden", DType::F16, &[1, 1, 32]),
                tensor("skip", DType::F16, &[1, 1, 32]),
                tensor("gamma", DType::F16, &[32]),
            ],
            outputs: vec![out("out0", DType::F16, &[1, 1, 32])],
            ..Default::default()
        };
        let mut ctx = Recorder::default();
        // Must succeed — before the fix, this returned Err(Unsupported("fp16 variant is not yet written")).
        skip_norm(spec, &node, &mut ctx).expect(
            "f16 SkipSimplifiedLayerNorm must not return Unsupported after this commit; \
             if it does, the f16 shader path was not wired in",
        );
    }

    // ── simplified_norm tests ────────────────────────────────────────────────────────────────

    fn rmsnorm_node(dtype: DType, batch: i64, seq: i64, hidden: i64) -> NodeDesc {
        NodeDesc {
            op_type: "SimplifiedLayerNormalization".into(),
            inputs: vec![
                tensor("X", dtype, &[batch, seq, hidden]),
                tensor("gamma", dtype, &[hidden]),
            ],
            outputs: vec![out("Y", dtype, &[batch, seq, hidden])],
            ..Default::default()
        }
    }

    #[test]
    fn simplified_norm_f32_produces_one_dispatch_with_correct_shader() {
        let spec = spec_named("SimplifiedLayerNormalization");
        let node = rmsnorm_node(DType::F32, 2, 4, 64);
        let mut ctx = Recorder::default();
        simplified_norm(spec, &node, &mut ctx).expect("translate");

        assert_eq!(ctx.dispatches.len(), 1);
        let k = &ctx.dispatches[0];
        assert_eq!(k.shader, "simplified_layer_norm_f32");
        assert_eq!(k.spec_constants, vec![NORM_LOCAL_SIZE]);
        assert_eq!(k.workgroups, [8, 1, 1], "batch×seq = 2×4 = 8 rows");
        assert_eq!(
            k.bindings.len(),
            3,
            "X, gamma, out0 — no skip input, no residual output"
        );
        assert_eq!(ctx.outputs.len(), 1, "slot 0 only");
    }

    #[test]
    fn simplified_norm_f16_produces_one_dispatch_with_correct_shader() {
        // The path Phi-3.5's `/model/layers.0/input_layernorm/LayerNorm` exercises.
        let spec = spec_named("SimplifiedLayerNormalization");
        let node = rmsnorm_node(DType::F16, 1, 1, 3072);
        let mut ctx = Recorder::default();
        simplified_norm(spec, &node, &mut ctx).expect("translate f16");

        let k = &ctx.dispatches[0];
        assert_eq!(k.shader, "simplified_layer_norm_f16");
        assert_eq!(k.workgroups, [1, 1, 1]);
        assert_eq!(k.bindings.len(), 3);
    }

    /// `simplified_norm` must **not** allocate a temp buffer.
    ///
    /// This is the structural half of the P6 assertion (`alloc_temp` is the only route from an
    /// op handler to device memory, so counting calls proves the property for every shape at
    /// once). `skip_norm` legitimately allocates one when slot 3 is absent; RMSNorm has no
    /// optional output, so a temp here would be a leak, not a fallback.
    #[test]
    fn simplified_norm_allocates_no_scratch() {
        let spec = spec_named("SimplifiedLayerNormalization");
        let node = rmsnorm_node(DType::F16, 1, 8, 256);
        let mut ctx = Recorder::default();
        simplified_norm(spec, &node, &mut ctx).unwrap();
        assert_eq!(
            ctx.outputs.len(),
            1,
            "exactly one bind_output and zero alloc_temp calls"
        );
    }

    #[test]
    fn simplified_norm_push_constants_encode_shape_and_epsilon() {
        let spec = spec_named("SimplifiedLayerNormalization");
        let mut node = rmsnorm_node(DType::F32, 1, 8, 128);
        node.attributes
            .insert("epsilon".into(), AttrValue::Float(1e-6));
        let mut ctx = Recorder::default();
        simplified_norm(spec, &node, &mut ctx).unwrap();
        let pc = &ctx.dispatches[0].push_constants;
        assert_eq!(pc.len(), 16);
        assert_eq!(u32::from_le_bytes(pc[0..4].try_into().unwrap()), 8);
        assert_eq!(u32::from_le_bytes(pc[4..8].try_into().unwrap()), 128);
        assert_eq!(
            f32::from_bits(u32::from_le_bytes(pc[8..12].try_into().unwrap())),
            1e-6
        );
        assert_eq!(u32::from_le_bytes(pc[12..16].try_into().unwrap()), 0);
    }

    #[test]
    fn simplified_norm_default_epsilon_matches_onnx_schema() {
        let spec = spec_named("SimplifiedLayerNormalization");
        let node = rmsnorm_node(DType::F32, 1, 1, 16);
        let mut ctx = Recorder::default();
        simplified_norm(spec, &node, &mut ctx).unwrap();
        let pc = &ctx.dispatches[0].push_constants;
        assert_eq!(
            f32::from_bits(u32::from_le_bytes(pc[8..12].try_into().unwrap())),
            NORM_EPSILON_DEFAULT
        );
    }

    #[test]
    fn simplified_norm_rank1_input_yields_batch_count_one() {
        let spec = spec_named("SimplifiedLayerNormalization");
        let node = NodeDesc {
            op_type: "SimplifiedLayerNormalization".into(),
            inputs: vec![
                tensor("X", DType::F32, &[256]),
                tensor("gamma", DType::F32, &[256]),
            ],
            outputs: vec![out("Y", DType::F32, &[256])],
            ..Default::default()
        };
        let mut ctx = Recorder::default();
        simplified_norm(spec, &node, &mut ctx).unwrap();
        assert_eq!(ctx.dispatches[0].workgroups, [1, 1, 1]);
    }

    #[test]
    fn simplified_norm_f32_shader_exists_on_disk() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("shaders")
            .join("glsl")
            .join("simplified_layer_norm_f32.comp");
        assert!(
            path.is_file(),
            "shaders/glsl/simplified_layer_norm_f32.comp is missing; the translate handler \
             names it directly so it must exist at build time"
        );
    }

    #[test]
    fn simplified_norm_f16_shader_exists_on_disk() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("shaders")
            .join("glsl")
            .join("simplified_layer_norm_f16.comp");
        assert!(
            path.is_file(),
            "shaders/glsl/simplified_layer_norm_f16.comp is missing; the translate handler \
             names it directly so it must exist at build time"
        );
    }

    /// The two norm shaders must not be confused for one another.
    ///
    /// `skip_norm` binds 5 buffers and `simplified_norm` binds 3; if a future refactor points
    /// one row at the other's handler the binding count is what catches it, because the
    /// descriptor set would be built with the wrong arity and the shader would read garbage.
    #[test]
    fn the_two_norm_handlers_have_different_binding_arity() {
        let mut a = Recorder::default();
        simplified_norm(
            spec_named("SimplifiedLayerNormalization"),
            &rmsnorm_node(DType::F32, 1, 2, 32),
            &mut a,
        )
        .unwrap();
        let mut b = Recorder::default();
        skip_norm(
            spec_named("SkipSimplifiedLayerNormalization"),
            &skip_norm_node_f32(1, 2, 32),
            &mut b,
        )
        .unwrap();
        assert_eq!(a.dispatches[0].bindings.len(), 3);
        assert_eq!(b.dispatches[0].bindings.len(), 5);
        assert_ne!(a.dispatches[0].shader, b.dispatches[0].shader);
    }

    // ── gather tests ─────────────────────────────────────────────────────────────────────────

    fn gather_node(
        data_dtype: DType,
        data_shape: &[i64],
        idx_dtype: DType,
        idx_shape: &[i64],
        axis: Option<i64>,
    ) -> NodeDesc {
        let mut n = NodeDesc {
            op_type: "Gather".into(),
            inputs: vec![
                tensor("data", data_dtype, data_shape),
                tensor("indices", idx_dtype, idx_shape),
            ],
            outputs: vec![out("Y", data_dtype, &[])],
            ..Default::default()
        };
        if let Some(a) = axis {
            n.attributes.insert("axis".into(), AttrValue::Int(a));
        }
        n
    }

    fn push_u32(k: &KernelRequest, i: usize) -> u32 {
        u32::from_le_bytes(k.push_constants[i * 4..i * 4 + 4].try_into().unwrap())
    }

    /// The Phi-3.5 embedding lookup, which is the node this op exists for.
    #[test]
    fn gather_embedding_lookup_flattens_to_the_right_extents() {
        let spec = spec_named("Gather");
        let node = gather_node(DType::F16, &[32064, 3072], DType::I64, &[1, 1], Some(0));
        let mut ctx = Recorder::default();
        gather(spec, &node, &mut ctx).expect("translate");

        let k = &ctx.dispatches[0];
        assert_eq!(k.shader, "gather_f16");
        assert_eq!(push_u32(k, 0), 1, "outer = prod(shape[:0]) = 1");
        assert_eq!(push_u32(k, 1), 32064, "gathered = vocab");
        assert_eq!(push_u32(k, 2), 3072, "inner = hidden");
        assert_eq!(push_u32(k, 3), 1, "n_idx = 1x1");
        assert_eq!(push_u32(k, 4), 2, "int64 indices step two words");
        assert_eq!(push_u32(k, 5), 3072, "total output elements");
        assert_eq!(ctx.outputs[0].shape, vec![1, 1, 3072]);
        assert_eq!(ctx.outputs.len(), 1, "no scratch allocation");
    }

    /// Output shape is `data[:axis] ++ indices ++ data[axis+1:]`, which is the part of the ONNX
    /// schema most easily got wrong when the indices are not rank 1.
    #[test]
    fn gather_output_shape_interleaves_indices_at_the_axis() {
        let spec = spec_named("Gather");
        let node = gather_node(DType::F32, &[4, 5, 6], DType::I32, &[2, 3], Some(1));
        let mut ctx = Recorder::default();
        gather(spec, &node, &mut ctx).unwrap();

        assert_eq!(ctx.outputs[0].shape, vec![4, 2, 3, 6]);
        let k = &ctx.dispatches[0];
        assert_eq!(push_u32(k, 0), 4, "outer");
        assert_eq!(push_u32(k, 1), 5, "gathered");
        assert_eq!(push_u32(k, 2), 6, "inner");
        assert_eq!(push_u32(k, 3), 6, "n_idx = 2x3");
        assert_eq!(push_u32(k, 4), 1, "int32 indices step one word");
        assert_eq!(push_u32(k, 5), 4 * 6 * 6);
    }

    #[test]
    fn gather_negative_axis_is_normalized_before_dispatch() {
        let spec = spec_named("Gather");
        let node = gather_node(DType::F32, &[4, 5, 6], DType::I32, &[2], Some(-2));
        let mut ctx = Recorder::default();
        gather(spec, &node, &mut ctx).unwrap();
        let k = &ctx.dispatches[0];
        assert_eq!(push_u32(k, 0), 4);
        assert_eq!(push_u32(k, 1), 5, "-2 on rank 3 is axis 1");
        assert_eq!(push_u32(k, 2), 6);
    }

    #[test]
    fn gather_default_axis_is_zero() {
        let spec = spec_named("Gather");
        let node = gather_node(DType::F32, &[7, 3], DType::I64, &[2], None);
        let mut ctx = Recorder::default();
        gather(spec, &node, &mut ctx).unwrap();
        assert_eq!(push_u32(&ctx.dispatches[0], 1), 7);
    }

    /// The f16 path packs two elements per word, so it must dispatch half the threads.
    ///
    /// Getting this wrong in the *other* direction is invisible — the grid-stride loop simply
    /// leaves the tail unwritten, which is the never-written-output defect class again.
    #[test]
    fn gather_f16_dispatches_half_as_many_threads_as_f32() {
        let spec = spec_named("Gather");
        let mut a = Recorder::default();
        gather(
            spec,
            &gather_node(DType::F16, &[100, 1024], DType::I64, &[8], Some(0)),
            &mut a,
        )
        .unwrap();
        let mut b = Recorder::default();
        gather(
            spec,
            &gather_node(DType::F32, &[100, 1024], DType::I64, &[8], Some(0)),
            &mut b,
        )
        .unwrap();
        // 8 * 1024 = 8192 elements → 4096 words → 16 groups (f16); 8192 elements → 32 (f32).
        assert_eq!(a.dispatches[0].workgroups, [16, 1, 1]);
        assert_eq!(b.dispatches[0].workgroups, [32, 1, 1]);
    }

    /// The grid-stride loop exists so the workgroup count is bounded; assert the bound holds.
    #[test]
    fn gather_workgroup_count_is_capped_below_the_vulkan_floor() {
        let spec = spec_named("Gather");
        let node = gather_node(DType::F32, &[32064, 3072], DType::I64, &[1, 4096], Some(0));
        let mut ctx = Recorder::default();
        gather(spec, &node, &mut ctx).unwrap();
        let g = ctx.dispatches[0].workgroups[0];
        assert_eq!(g, GATHER_MAX_WORKGROUPS);
        assert!(
            g <= 65535,
            "maxComputeWorkGroupCount[0] is guaranteed to be at least 65535 and no more"
        );
    }

    #[test]
    fn gather_declines_an_index_dtype_the_schema_does_not_allow() {
        let spec = spec_named("Gather");
        let node = gather_node(DType::F32, &[4, 4], DType::F32, &[2], Some(0));
        let mut ctx = Recorder::default();
        let err = gather(spec, &node, &mut ctx).expect_err("f32 indices are not a legal Gather");
        assert!(format!("{err:?}").contains("indices"), "{err:?}");
    }

    #[test]
    fn gather_f32_shader_exists_on_disk() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("shaders")
            .join("glsl")
            .join("gather_f32.comp");
        assert!(path.is_file(), "shaders/glsl/gather_f32.comp is missing");
    }

    #[test]
    fn gather_f16_shader_exists_on_disk() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("shaders")
            .join("glsl")
            .join("gather_f16.comp");
        assert!(path.is_file(), "shaders/glsl/gather_f16.comp is missing");
    }

    // ---------------------------------------------------------------------------------------
    // `Cast` on a graph whose extents are symbolic.
    //
    // These four are one finding, and the first is the regression: `ep::tensor_desc` returns
    // `None` for an edge with any symbolic dimension and drops the **dtype** with the shape, so
    // the Compute-time dynamic re-translate hands this handler an output with no element type.
    // `claim::cast` reads the edge type off the live `OrtValueInfo` and is unaffected, so every
    // `runtime-extent` `Cast` was claimed and then broke its commitment — 98 nodes of
    // gpt-oss-20b, measured 2026-08-03 by `rust/tools/probe_model_op_census.py`.
    //
    // A unit test rather than only a ledger case because the ledger case proves the *fixed*
    // path and would go green again if someone restored the edge-only read on a build where the
    // case model happened to have concrete shapes. This one states the precondition directly.
    // ---------------------------------------------------------------------------------------

    /// ONNX `TensorProto.DataType` values, as `Cast`'s `to` attribute spells them.
    const ONNX_FLOAT: i64 = 1;
    const ONNX_INT32: i64 = 6;

    fn cast_node(src: DType, out_desc: Option<DType>, to: Option<i64>, shape: &[i64]) -> NodeDesc {
        let mut attributes = std::collections::BTreeMap::new();
        if let Some(t) = to {
            attributes.insert("to".to_string(), AttrValue::Int(t));
        }
        NodeDesc {
            op_type: "Cast".into(),
            inputs: vec![tensor("x", src, shape)],
            outputs: vec![OutRef {
                name: "y".into(),
                desc: out_desc.map(|d| TensorDesc::new(d, shape.to_vec())),
            }],
            attributes,
            ..Default::default()
        }
    }

    #[test]
    fn cast_takes_its_destination_from_to_when_the_output_edge_was_dropped() {
        let spec = spec_named("Cast");
        let node = cast_node(DType::F16, None, Some(ONNX_FLOAT), &[4, 8]);
        let mut ctx = Recorder::default();
        ew_cast(spec, &node, &mut ctx).expect(
            "a symbolic extent drops the output desc; `to` is the destination type and it is \
             still there",
        );
        assert_eq!(ctx.dispatches[0].shader, "ew_cast_f16_to_f32");
        assert_eq!(ctx.outputs[0].dtype, DType::F32);
    }

    #[test]
    fn cast_still_prefers_the_resolved_output_edge_when_it_has_one() {
        let spec = spec_named("Cast");
        let node = cast_node(DType::F32, Some(DType::I32), Some(ONNX_INT32), &[4, 8]);
        let mut ctx = Recorder::default();
        ew_cast(spec, &node, &mut ctx).expect("translate");
        assert_eq!(ctx.dispatches[0].shader, "ew_cast_f32_to_i32");
    }

    #[test]
    fn cast_refuses_when_the_edge_and_the_attribute_disagree() {
        let spec = spec_named("Cast");
        let node = cast_node(DType::F32, Some(DType::I32), Some(ONNX_FLOAT), &[4, 8]);
        let mut ctx = Recorder::default();
        let err = ew_cast(spec, &node, &mut ctx)
            .expect_err("choosing between a graph and ORT's inference is not this EP's call");
        assert!(format!("{err:?}").contains("disagree"), "{err:?}");
    }

    #[test]
    fn cast_refuses_when_neither_source_of_the_destination_type_survives() {
        let spec = spec_named("Cast");
        let node = cast_node(DType::F32, None, None, &[4, 8]);
        let mut ctx = Recorder::default();
        let err = ew_cast(spec, &node, &mut ctx).expect_err("no destination type anywhere");
        let msg = format!("{err:?}");
        assert!(msg.contains("output edge"), "{msg}");
        assert!(
            msg.contains("`to`"),
            "the message has to name both places looked: {msg}"
        );
    }
}
