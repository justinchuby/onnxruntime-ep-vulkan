//! Shape ops — the family whose output *is* the input's bytes under a different description.
//!
//! # Why this family is separate, and what the separation is actually about
//!
//! I argued on 2026-08-03 (`mouse-mobilenetv2-residual-declines.md`) that this family is "shape
//! metadata moving no float data", and attached a falsifier: *if a censused model routes real
//! tensor data through one of these, the distinction is wrong.* BERT-SQuAD-12 fired it. Its 59
//! `Reshape` nodes carry the attention tensors themselves — `[batch*seq, 768]` of f32, the whole
//! residual stream — and calling that metadata was simply false.
//!
//! The distinction that survives is narrower and is the one this module is named for: **is the
//! op's output the tensor, or a description of the tensor?** `Shape` produces an `int64[rank]`
//! *about* a tensor and its decline stands on that basis. `Reshape` produces the tensor. So
//! `Reshape` is a kernel decision, and this module contains a kernel.
//!
//! # What `Reshape` costs: a copy, and why it is not an alias
//!
//! A `Reshape` is a reinterpretation of a row-major buffer, so the *ideal* implementation binds
//! the input allocation as the output and dispatches nothing. That is not available here, and
//! the reason is in `vk/session.rs` rather than in anybody's opinion:
//!
//! * `DispatchContext::bind_aliased_output` exists and `ShapeOnlyRecorder` records the pair, but
//!   `dispatch_ort` honours a pair **only when the output is an external plan output and the
//!   input is an external plan input**. Every `Reshape` worth claiming is an *interior* edge of
//!   an island — that is precisely why it ranks in the island counterfactual — so its pair would
//!   be recorded, silently ignored, and its output buffer allocated and never written. Every
//!   downstream consumer would then read uninitialised device memory: a wrong answer of the
//!   right shape, which this crate's coverage charter puts above every other failure.
//! * Switch's KV-arena disjointness argument does not transfer. His concerns a *shader* that
//!   reads `past[t]` and writes `present[tok_pos]` in one dispatch, discharged by
//!   `tok_pos = past_len + s_local >= past_len` under a common stride. An aliased `Reshape`
//!   performs no write at all, so that hazard cannot arise — but a different one does, which his
//!   argument never had to address: two live tensors sharing one allocation with independent
//!   lifetimes, against a generation-stamped quarantine-on-free allocator. Nothing tracks that
//!   for interior tensors today. Aliasing `Reshape` is an engine change, not an op change, and
//!   writing it in `ops/` would disguise which it is.
//!
//! So the row dispatches a **copy**: one full-tensor read and one full-tensor write through
//! `ew_cast_f32_to_f32`, a module the build already produces. The copy is not free and is not
//! described as free. What pays for it is what it removes — an unclaimed `Reshape` *splits an
//! island*, and a split costs a device→host download, a CPU-EP execution and a host→device
//! upload of the same bytes. One device-local pass is strictly cheaper than that round trip.
//! This is a boundary-bytes argument, the same shape as [`crate::ops::indexing`]'s for `Gather`,
//! and not a FLOPs one: a `Reshape` does no arithmetic.
//!
//! No new `.comp`. A second `reshape_f32.comp` would be a variant component distinguishing
//! nothing — the §8.9.23 error `form.rs` was deleted for committing, and the same reason
//! `MatMul` dispatches `gemm_f32`. What differs between `Cast` and `Reshape` is the *claim*, and
//! that difference lives in a separate registry row with a separate proof key.
//!
//! # What is declined, and why it is most of them
//!
//! On BERT-SQuAD-12, **53 of 59 `Reshape` claim rows carry no output rank at all** and 48 carry
//! no input rank either: ORT's shape inference does not resolve this graph before partitioning,
//! because 58 of the 71 graph `Reshape` nodes take their target shape from a runtime `Cast`
//! output rather than an initializer. A `Reshape` whose output rank is unknown cannot be given
//! an output `TensorDesc`, cannot be sized, and cannot be handed to a consumer — so it declines,
//! and it declines at the gate rather than failing at translate and taking the island with it.
//!
//! `DispatchContext::read_const_i64` is documented for exactly this case — *"`Reshape`'s shape
//! input"* — and **every implementation in the tree returns `None`**. It is not used here, and
//! this module does not pretend it is available.

use crate::engine::{
    DType, DispatchContext, EpError, EpResult, KernelRequest, NodeDesc, TensorDesc,
};
use crate::kernel;
use crate::ops::common::claim::{self, ClaimResult};
use crate::ops::common::dtype::{F32, dtype_suffix};
use crate::ops::common::shape_plan::{EW_PARAMS_NONE, MAX_RANK, ShapePlan};
use crate::ops::common::templates::EW_LOCAL_SIZE;
use crate::registry::OpStatus::Ready;
use crate::registry::{NodeView, OPSET_ANY, OpSpec};
use crate::require;

/// Resolve the concrete output shape of a `Reshape` from the shape ORT *declared* for its output
/// plus the element count the input *actually has*.
///
/// This is deliberately not a second implementation of `Reshape`'s shape rules. ONNX's `0`
/// (copy the input dim) and `-1` (infer) are applied by ORT's shape inference before we see
/// anything; what reaches `declared` is the result, in which a `-1` means "ORT could not
/// resolve this axis", not "the graph asked for inference". The one thing we add is the
/// invariant that makes a single unresolved axis solvable at all — **a reshape conserves the
/// element count** — and that is a fact about the op, not a re-derivation of ORT's answer.
///
/// (Last round the census carried its own shape inference and was materially wrong; the probe
/// that held it was deleted. This function stays this small for that reason.)
fn resolve_out_dims(op: &str, in_shape: &[i64], declared: &[i64]) -> EpResult<Vec<i64>> {
    let refuse = |why: String| EpError::Unsupported(format!("`{op}` {why}"));

    if declared.is_empty() {
        return Err(refuse(
            "output has no rank at compile time; a reshape target cannot be invented from the \
             input alone"
                .into(),
        ));
    }
    let mut in_elems: i64 = 1;
    for &d in in_shape {
        if d < 0 {
            return Err(refuse(format!(
                "input 0 is {in_shape:?} at dispatch time; an extent is still symbolic, so the \
                 element count that would fix the output is unknown"
            )));
        }
        in_elems = in_elems
            .checked_mul(d)
            .ok_or_else(|| refuse(format!("input 0 extents {in_shape:?} overflow i64")))?;
    }

    let free: Vec<usize> = declared
        .iter()
        .enumerate()
        .filter(|(_, d)| **d < 0)
        .map(|(i, _)| i)
        .collect();
    if free.len() > 1 {
        return Err(refuse(format!(
            "output is {declared:?} with {} unresolved axes; conservation of element count fixes \
             at most one",
            free.len()
        )));
    }
    let mut known: i64 = 1;
    for &d in declared {
        if d >= 0 {
            known = known
                .checked_mul(d)
                .ok_or_else(|| refuse(format!("output extents {declared:?} overflow i64")))?;
        }
    }

    let mut out = declared.to_vec();
    if let Some(&i) = free.first() {
        if known <= 0 || in_elems % known != 0 {
            return Err(refuse(format!(
                "output {declared:?} has one unresolved axis and the resolved axes multiply to \
                 {known}, which does not divide the input's {in_elems} elements"
            )));
        }
        out[i] = in_elems / known;
    } else if known != in_elems {
        return Err(refuse(format!(
            "output {declared:?} holds {known} elements but input {in_shape:?} holds {in_elems}; \
             a reshape conserves the element count"
        )));
    }

    // Zero-element tensors are legal ONNX and are *not* dispatched: a zero-workgroup dispatch is
    // a no-op on some drivers and a validation error on others, and there is no data to move
    // either way. The CPU EP is correct and cheap for them.
    if in_elems == 0 {
        return Err(refuse(format!(
            "input {in_shape:?} has zero elements; this row does not dispatch an empty grid"
        )));
    }
    Ok(out)
}

/// Can a tensor holding `ins` elements become one shaped `declared`, at the gate?
///
/// Returns `true` when the question cannot be settled yet (a symbolic input extent), because a
/// gate that declines what it cannot check is a gate that declines every dynamic model. Returns
/// `false` only when the counts are *provably* incompatible.
///
/// **This is the rank-0 discriminator, and it is the whole reason this function exists.** ORT
/// reports rank 0 for a genuine scalar *and* for a rank shape inference never established;
/// `EdgeType::is_static` returns `true` for both, because "all dims are non-negative" is
/// vacuously true over an empty list. `MatMul` could resolve the ambiguity from the ONNX schema,
/// which admits no rank-0 operand. `Reshape` cannot — a scalar reshaped to `[1]` is legal ONNX.
///
/// So it is resolved by arithmetic instead, which is both stronger and narrower than a per-op
/// minimum-rank table: **a rank-0 input holds exactly one element.** A declared output of
/// `[-1, 768]` needs a multiple of 768, and 1 is not one, so the unresolved-rank reading
/// declines while a genuine scalar (whose output really does hold one element) still passes.
///
/// Measured on BERT-SQuAD-12, 2026-08-04: this is what declines `bert/encoder/Reshape_1` and
/// `.../layer_0/attention/self/ExpandDims`, the only two `Reshape` nodes in that graph whose
/// float dtype and resolved output shape got them past every other check. The first version of
/// this predicate claimed both, on the strength of having read nothing.
fn conservation_holds(ins: &[i64], declared: &[i64]) -> bool {
    if ins.iter().any(|d| *d < 0) {
        return true;
    }
    let elems: i64 = ins.iter().product();
    let known: i64 = declared.iter().filter(|d| **d >= 0).product();
    if declared.iter().any(|d| *d < 0) {
        known > 0 && elems % known == 0
    } else {
        known == elems
    }
}

/// `Reshape` — claim an f32 tensor whose *output rank* ORT resolved, with at most one free axis.
///
/// The gate is strict on purpose. A node claimed here and refused at translate does not fall back
/// to the CPU for that node — it fails the whole fused island, which is how 323 claimed Phi-3.5
/// nodes once executed zero times. Every condition the translate handler needs is checked here.
fn reshape(view: &NodeView<'_>, spec: &OpSpec) -> ClaimResult {
    require!(
        view.num_inputs() == 2,
        Arity,
        "`{}` has {} inputs; opset 5 onwards takes exactly `data` and `shape`",
        spec.op_type,
        view.num_inputs()
    );
    require!(
        view.num_outputs() == 1,
        Arity,
        "`{}` has {} outputs; it produces exactly 1",
        spec.op_type,
        view.num_outputs()
    );

    let data = claim::input_edge(view, spec, 0)?;
    claim::check_dtype(spec, &data, "input 0 (data)")?;
    claim::check_shape(spec, &data, "input 0 (data)")?;

    // The shape operand is never *read* by this handler — the target comes from ORT's declared
    // output — but its type is still checked, because a `Reshape` whose second operand is not an
    // int64 vector is not the op the ONNX schema describes and claiming it would mean the graph
    // is not what we think it is.
    let target = claim::input_edge(view, spec, 1)?;
    require!(
        target.dtype == Some(DType::I64),
        DType,
        "`{}` input 1 (shape) is {}; the ONNX schema makes it int64",
        spec.op_type,
        target.dtype.map_or("untyped", dtype_suffix)
    );
    require!(
        target.rank() == Some(1),
        Rank,
        "`{}` input 1 (shape) has rank {:?}; the ONNX schema makes it a 1-D vector",
        spec.op_type,
        target.rank()
    );

    // `allowzero=1` changes what a literal `0` in the shape operand means (an empty axis rather
    // than "copy the input's"). ORT applies it during inference, so the declared output this row
    // reads is already correct either way — but a row that claims an attribute it never reads
    // and never tested is the `Gemm` transpose mistake, and here there is no CI suite behind it
    // at all. Declined until a graph exists to test it on.
    claim::attr_int_is(view, spec, "allowzero", 0)?;

    let out = view.output_type_as_reported(0).ok_or_else(|| {
        crate::registry::decline(
            crate::registry::DeclineCode::UnknownRank,
            format_args!("`{}` output 0 has no type information", spec.op_type),
        )
    })?;
    require!(
        out.dtype == data.dtype,
        DType,
        "`{}` output is {} but input 0 is {}; a reshape does not convert",
        spec.op_type,
        out.dtype.map_or("untyped", dtype_suffix),
        data.dtype.map_or("untyped", dtype_suffix)
    );

    // The decisive check, and the one that declines most of BERT. ORT reports rank 0 for a
    // genuine scalar *and* for a rank it never established, and `EdgeType` faithfully records
    // both as `Some([])`. Here the ambiguity does not have to be resolved to be safe: a rank-0
    // output would require a rank-0 *input* (one element), and the shape operand would be a
    // 1-D vector of length 0, which no graph in the census emits. Either reading declines, and
    // the code says `unknown-rank` because that is what it is on every node measured.
    //
    // Read from `output_type_as_reported`, deliberately: the §8.11 rank overlay can prove this
    // output's rank from the `Shape`/`Cast`/`Concat` chain that feeds the target, and that proof
    // is sound — but it is a proof about the *graph*, not a `TensorDesc` ORT will hand `Compute()`
    // for an output whose extents it never resolved. The whole point of the check below is that
    // ORT reports no output descriptor for exactly those outputs, so it has to ask ORT.
    let declared = out.shape.clone().unwrap_or_default();
    require!(
        !declared.is_empty(),
        UnknownRank,
        "`{}` output reads as rank 0. ORT reports rank 0 both for a scalar and for a rank shape \
         inference never established; on every model censused it is the second, because the \
         shape operand is computed at run time from a `Shape`/`Concat` chain. Without an output \
         rank there is no `TensorDesc` to bind and no size to allocate, and no runtime-extent \
         handling recovers a rank",
        spec.op_type
    );
    require!(
        declared.len() <= MAX_RANK,
        Rank,
        "`{}` output has rank {}; the shared indexing helper handles at most {MAX_RANK}",
        spec.op_type,
        declared.len()
    );
    let free = declared.iter().filter(|d| **d < 0).count();
    require!(
        free <= 1,
        DynamicShape,
        "`{}` output is {declared:?} with {free} unresolved axes; conservation of element count \
         fixes at most one, so the rest would have to be guessed",
        spec.op_type
    );

    // When the input's extents are all resolved, **conservation of element count is checkable
    // now**, and it is checked in both the free-axis and the fully-resolved case.
    //
    // This is also, and mainly, the rank-0 discriminator. ORT reports rank 0 for a genuine
    // scalar *and* for a rank it never established, and `check_shape` above passes a rank-0
    // edge because `is_static()` — "all dims non-negative" — is **vacuously true over an empty
    // list**. That is the defect named on 2026-08-04 for `MatMul`, and this predicate reproduced
    // it: BERT's `bert/encoder/Reshape_1` reads input rank 0 with output `[-1, 768]` and was
    // claimed on the strength of having read nothing.
    //
    // `MatMul` could settle the ambiguity from the ONNX schema, which admits no rank-0 operand.
    // `Reshape` cannot — a scalar reshaped to `[1]` is legal. So it is settled by arithmetic
    // instead, which is stronger: a rank-0 input holds exactly **one** element, so a declared
    // output of `[-1, 768]` needs a multiple of 768 and 1 is not one. A *genuine* scalar still
    // passes, because its output really does hold one element. No per-op minimum-rank table is
    // needed and none is invented here.
    if let Some(ins) = data.shape.as_deref() {
        let elems: i64 = ins.iter().product();
        let known: i64 = declared.iter().filter(|d| **d >= 0).product();
        require!(
            conservation_holds(ins, &declared),
            Shape,
            "`{}` input {ins:?} holds {elems} elements and output {declared:?} accounts for \
             {known}; a reshape conserves the element count. (A rank-0 input reads as one \
             element — ORT reports rank 0 both for a scalar and for a rank it never \
             established, and this is what distinguishes them without a schema argument.)",
            spec.op_type
        );
    }

    // MEASURED 2026-08-04, and the reason this row does not claim a free axis at all.
    //
    // A free axis is resolvable *in principle* — conservation fixes it — but not from what
    // `Compile` hands the translate handler. Built as a ledger case (`reshape_f32_dyn`, input
    // `[N,3,4]`, target `[-1,4]`), the node claimed, then `dispatch_ort`'s dynamic re-run
    // refused with "`Reshape` output has no declared shape": **ORT reports no output
    // `TensorDesc` at all for an output whose extents it could not resolve**, and the shape
    // operand's *value* — the other way to recover the target — is unreachable, because
    // `read_const_i64` returns `None` in every implementation in this tree. Neither reading
    // exists, so the target is genuinely unknown at the only point it is needed.
    //
    // That failure mode is not a decline. It is a claimed island whose `Compute()` returns
    // non-OK — a broken commitment, the exact failure the 323-node Phi-3.5 round was about. So
    // the gate refuses here, where refusing is free, rather than at translate, where it is not.
    //
    // This check is deliberately placed *after* the conservation check above: a node like BERT's
    // `bert/encoder/Reshape_1` (`[]` -> `[-1, 768]`) is wrong for a more specific reason than
    // "has a free axis", and the more specific decline is the more useful one to read.
    require!(
        free == 0,
        DynamicShape,
        "`{}` output is {declared:?}; conservation fixes the free axis, but ORT reports no \
         output descriptor at all for an output it could not resolve, and the shape operand's \
         value is unreadable (`read_const_i64` is unimplemented). Claiming this form produces a \
         broken commitment at `Compute()`, not a fallback",
        spec.op_type
    );

    let loadable = data.dtype.is_some_and(|d| {
        spec.kernel
            .pair_stem(d, d)
            .is_some_and(crate::ops::common::variants::variant_is_loadable)
    });
    require!(
        loadable,
        DType,
        "`{}` has no loadable copy variant at {}; the module either was not generated or declares \
         a SPIR-V capability this engine does not enable. The CPU EP is correct for this node",
        spec.op_type,
        data.dtype.map_or("untyped", dtype_suffix)
    );
    Ok(())
}

/// Translate `Reshape` into the copy it is.
///
/// Reads input 0's shape from [`NodeDesc`] rather than trusting the claim, for the reason
/// `translate_matmul` does: on this class of model `Compile`/`Compute` see strictly better shape
/// information than `GetCapability`, so the gate is the weaker of the two readings and this is
/// the one that must be exact. Input 1 is deliberately not resolved — its *value* is what a
/// reshape needs, `read_const_i64` is `None` in every implementation in the tree, and its buffer
/// would be bound only to be ignored.
pub fn translate_reshape(
    spec: &OpSpec,
    node: &NodeDesc,
    ctx: &mut dyn DispatchContext,
) -> EpResult<()> {
    let data = node
        .inputs
        .first()
        .and_then(|t| t.desc.as_ref())
        .ok_or_else(|| {
            EpError::Unsupported(format!(
                "`{}` input 0 has no shape at compile time",
                node.op_type
            ))
        })?;
    if node.outputs.len() != 1 {
        return Err(EpError::Internal(format!(
            "`{}` was claimed as single-output but has {}",
            node.op_type,
            node.outputs.len()
        )));
    }
    let out = &node.outputs[0];
    let declared = out.desc.as_ref().map(|d| d.shape.clone()).ok_or_else(|| {
        EpError::Unsupported(format!(
            "`{}` output has no declared shape; the reshape target is unknown",
            node.op_type
        ))
    })?;

    let dtype = data.dtype;
    let out_dims = resolve_out_dims(&node.op_type, &data.shape, &declared)?;

    let shader = spec.kernel.pair_stem(dtype, dtype).ok_or_else(|| {
        EpError::Internal(format!(
            "`{}` was claimed but its row declares no pair-keyed shader",
            node.op_type
        ))
    })?;

    // The plan is built over the *input* shape, and that is the whole trick: a row-major
    // `[d0, ..., dn]` buffer and a row-major reshape of it are the same bytes in the same order,
    // so the copy is linear in both and the output's dimensions never enter the index walk. They
    // enter exactly once, in the `TensorDesc` below, which is what the allocator sizes from and
    // what a downstream consumer in the same island reads back out of `computed_descs`.
    let plan = ShapePlan::broadcast(&[data.shape.as_slice()]).map_err(|e| {
        EpError::Unsupported(format!("`{}` shape cannot be planned: {e}", node.op_type))
    })?;

    let bindings = vec![
        ctx.resolve(&node.inputs[0])?,
        ctx.bind_output(out, TensorDesc::new(dtype, out_dims))?,
    ];

    ctx.dispatch(KernelRequest {
        shader,
        spec_constants: vec![EW_LOCAL_SIZE, u32::from(plan.all_identical)],
        push_constants: plan.push_constants_with_params(EW_PARAMS_NONE),
        bindings,
        workgroups: plan.workgroups_1d(EW_LOCAL_SIZE),
    })
}

crate::op_table! {
    //  op         domain  opsets           caps  kernel               claim     translate           status
    //
    // The window opens at **5**, not 1. `Reshape-1` carried the target as a `shape` *attribute*
    // and had no second input; opset 5 moved it to an input, which is the form this row's arity
    // check and every graph in the census assume. A row opening at 1 would claim a node whose
    // second operand does not exist and decline it on arity — a decline that reads as "we don't
    // handle this shape" when the truth is "this row was written for a different operator".
    //
    // `caps` is F32 and that is a measurement, not a placeholder. `ew_cast_f16_to_f16` is
    // generated and would load, but **no censused model needs it**: Phi-3.5-mini-int4 contains
    // zero `Reshape` nodes in its whole graph, gpt-oss-20b's are not reachable, and BERT and
    // MobileNetV2 are f32. Widening the caps would add a second proof key with nothing to prove
    // it against, which is the shape of claim this crate declines to make. Same reasoning as the
    // f16 `Conv` decline of 2026-08-03.
    //
    // No `blind_axes`: `allowzero` is the only attribute `Reshape` has and it is *declined*
    // rather than ignored, so the key is silent about nothing.
    "Reshape",  Ai,     5 ..= OPSET_ANY, F32,  kernel!(EwCast, "x"),  reshape,  translate_reshape,  Ready;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{BufferView, OutRef, TensorRef};
    use crate::ops::common::variants::Template;
    use crate::registry::OpStatus;

    /// Minimal recorder for translate handlers. A private copy, as in `ops::attention`, so this
    /// module's tests can assert on the one property that matters here — the *output descriptor*
    /// the allocator would be handed.
    #[derive(Default)]
    struct Recorder {
        next: u64,
        dispatches: Vec<KernelRequest>,
        outputs: Vec<(String, TensorDesc)>,
    }

    impl DispatchContext for Recorder {
        fn resolve(&mut self, _r: &TensorRef) -> EpResult<BufferView> {
            self.next += 1;
            Ok(BufferView::from_raw(self.next))
        }
        fn bind_output(&mut self, o: &OutRef, desc: TensorDesc) -> EpResult<BufferView> {
            self.outputs.push((o.name.clone(), desc));
            self.next += 1;
            Ok(BufferView::from_raw(self.next))
        }
        fn alloc_temp(&mut self, desc: TensorDesc) -> EpResult<BufferView> {
            self.outputs.push((String::new(), desc));
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

    fn row() -> &'static OpSpec {
        OPS.iter()
            .find(|s| s.op_type == "Reshape")
            .expect("Reshape must be registered")
    }

    #[test]
    fn the_reshape_row_is_ready_and_points_at_its_own_handler() {
        let r = row();
        assert_eq!(r.status, OpStatus::Ready);
        assert!(std::ptr::fn_addr_eq(
            r.translate,
            translate_reshape as crate::registry::TranslateHandler
        ));
        assert!(
            !std::ptr::fn_addr_eq(
                r.translate,
                crate::ops::common::templates::unimplemented as crate::registry::TranslateHandler
            ),
            "the comparison must discriminate, or the assertion above proves nothing"
        );
        assert!(r.schema.is_none(), "`Reshape` is an ai.onnx row");
        assert!(
            r.blind_axes.is_empty(),
            "`allowzero` is declined, not ignored; the key is silent about nothing"
        );
    }

    /// The row must dispatch the module `Cast` already ships, not a second one. This is the
    /// assertion that fails if somebody "adds a reshape shader".
    #[test]
    fn reshape_dispatches_the_existing_cast_copy_module_and_no_new_one() {
        let r = row();
        assert_eq!(r.kernel.template, Template::EwCast);
        assert_eq!(
            r.kernel.pair_stem(DType::F32, DType::F32),
            Some("ew_cast_f32_to_f32"),
            "a reshape is a copy; a second module would distinguish nothing (§8.9.23)"
        );
        // And it is the *same* module `Cast`'s row names, read off that row rather than retyped.
        let cast = crate::ops::elementwise::OPS
            .iter()
            .find(|s| s.op_type == "Cast")
            .expect("Cast must be registered");
        assert_eq!(
            r.kernel.pair_stem(DType::F32, DType::F32),
            cast.kernel.pair_stem(DType::F32, DType::F32),
            "two rows, one module — the difference between them is the claim, not the shader"
        );
    }

    #[test]
    fn the_opset_window_opens_at_five_because_reshape_1_had_no_shape_input() {
        assert_eq!(row().min_opset, 5);
    }

    // ── resolve_out_dims: the only arithmetic in this module ─────────────────────────────

    #[test]
    fn a_fully_resolved_target_passes_through_unchanged() {
        assert_eq!(
            resolve_out_dims("Reshape", &[2, 3, 4], &[6, 4]).unwrap(),
            vec![6, 4]
        );
    }

    #[test]
    fn one_free_axis_is_fixed_by_conservation_of_element_count() {
        assert_eq!(
            resolve_out_dims("Reshape", &[2, 256], &[-1]).unwrap(),
            vec![512]
        );
        assert_eq!(
            resolve_out_dims("Reshape", &[3, 256], &[-1, 256, 1]).unwrap(),
            vec![3, 256, 1]
        );
    }

    #[test]
    fn two_free_axes_refuse_rather_than_guess() {
        let e = resolve_out_dims("Reshape", &[2, 256], &[-1, -1]).unwrap_err();
        assert!(format!("{e:?}").contains("at most one"), "{e:?}");
    }

    #[test]
    fn a_target_that_does_not_conserve_elements_refuses() {
        let e = resolve_out_dims("Reshape", &[2, 256], &[7, 9]).unwrap_err();
        assert!(format!("{e:?}").contains("conserves"), "{e:?}");
    }

    #[test]
    fn a_free_axis_that_does_not_divide_refuses() {
        let e = resolve_out_dims("Reshape", &[10], &[-1, 3]).unwrap_err();
        assert!(format!("{e:?}").contains("does not divide"), "{e:?}");
    }

    #[test]
    fn a_symbolic_input_extent_refuses_rather_than_treating_minus_one_as_a_dimension() {
        // The negative polarity of the conservation rule: `-1` in the *input* is not an extent,
        // and multiplying it in would produce a negative element count that divides cleanly and
        // looks plausible.
        let e = resolve_out_dims("Reshape", &[-1, 256], &[-1]).unwrap_err();
        assert!(format!("{e:?}").contains("still symbolic"), "{e:?}");
    }

    // ── conservation_holds: the rank-0 discriminator ─────────────────────────────────────

    #[test]
    fn a_rank_zero_input_is_one_element_and_cannot_become_a_768_wide_tensor() {
        // BERT's `bert/encoder/Reshape_1`, verbatim. The first version of this predicate
        // claimed it.
        assert!(!conservation_holds(&[], &[-1, 768]));
        // And `.../layer_0/attention/self/ExpandDims`.
        assert!(!conservation_holds(&[], &[-1, 1, 256, 256]));
    }

    #[test]
    fn a_genuine_scalar_still_passes_because_its_output_really_does_hold_one_element() {
        // The positive control. Without it the check above is indistinguishable from a blanket
        // "decline rank 0", which would be the per-op minimum-rank table this deliberately is
        // not.
        assert!(conservation_holds(&[], &[1]));
        assert!(conservation_holds(&[], &[1, 1, 1]));
        assert!(conservation_holds(&[], &[]));
        assert!(conservation_holds(&[], &[-1]));
    }

    #[test]
    fn a_symbolic_input_extent_defers_rather_than_declining_every_dynamic_model() {
        assert!(conservation_holds(&[-1, 768], &[-1, 12, 64]));
        assert!(conservation_holds(&[-1, 768], &[7]));
    }

    #[test]
    fn conservation_discriminates_on_resolved_shapes_in_both_directions() {
        assert!(conservation_holds(&[2, 3, 4], &[6, 4]));
        assert!(!conservation_holds(&[2, 3, 4], &[6, 5]));
        assert!(conservation_holds(&[2, 3, 4], &[-1, 4]));
        assert!(!conservation_holds(&[2, 3, 4], &[-1, 7]));
    }

    #[test]
    fn an_unranked_target_refuses() {
        let e = resolve_out_dims("Reshape", &[2, 256], &[]).unwrap_err();
        assert!(format!("{e:?}").contains("no rank"), "{e:?}");
    }

    #[test]
    fn a_zero_element_tensor_refuses_rather_than_dispatching_an_empty_grid() {
        let e = resolve_out_dims("Reshape", &[0, 256], &[0]).unwrap_err();
        assert!(format!("{e:?}").contains("zero elements"), "{e:?}");
    }

    // ── translate ────────────────────────────────────────────────────────────────────────

    fn node(in_shape: Vec<i64>, out_shape: Vec<i64>) -> NodeDesc {
        NodeDesc {
            op_type: "Reshape".into(),
            domain: String::new(),
            since_version: 13,
            name: "/r".into(),
            attributes: Default::default(),
            inputs: vec![
                TensorRef {
                    name: "data".into(),
                    desc: Some(TensorDesc::new(DType::F32, in_shape)),
                    is_initializer: false,
                },
                TensorRef {
                    name: "shape".into(),
                    desc: Some(TensorDesc::new(DType::I64, vec![out_shape.len() as i64])),
                    is_initializer: false,
                },
            ],
            outputs: vec![OutRef {
                name: "y".into(),
                desc: Some(TensorDesc::new(DType::F32, out_shape)),
            }],
        }
    }

    #[test]
    fn translate_binds_one_input_and_one_output_and_dispatches_the_copy() {
        let mut rec = Recorder::default();
        translate_reshape(row(), &node(vec![4, 256], vec![-1, 256, 1]), &mut rec).unwrap();
        assert_eq!(rec.dispatches.len(), 1, "a reshape is exactly one dispatch");
        let d = &rec.dispatches[0];
        assert_eq!(d.shader, "ew_cast_f32_to_f32");
        assert_eq!(
            d.bindings.len(),
            2,
            "data in, tensor out; the shape operand is not read"
        );
        // The output desc the allocator sizes from carries the *resolved* target, with the free
        // axis filled from the input's element count.
        let bound = rec
            .outputs
            .iter()
            .find(|(n, _)| n == "y")
            .expect("the output must be bound");
        assert_eq!(bound.1.shape, vec![4, 256, 1]);
        assert_eq!(bound.1.dtype, DType::F32);
        assert_eq!(bound.1.byte_size(), Some(4 * 256 * 4));
    }

    #[test]
    fn translate_refuses_an_unresolvable_target_instead_of_dispatching_a_wrong_size() {
        let mut rec = Recorder::default();
        let mut n = node(vec![4, 256], vec![-1]);
        n.outputs[0].desc = Some(TensorDesc::new(DType::F32, vec![-1, -1]));
        let e = translate_reshape(row(), &n, &mut rec).unwrap_err();
        assert!(format!("{e:?}").contains("at most one"), "{e:?}");
        assert!(rec.dispatches.is_empty(), "a refusal must dispatch nothing");
    }
}
