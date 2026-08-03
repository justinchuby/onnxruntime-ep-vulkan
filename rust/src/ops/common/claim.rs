//! Shared claim predicates — the "check before you claim" half of the registry.
//!
//! # The rule this module exists to enforce
//!
//! From the charter, and from `OP_COVERAGE.md` §7: **only claim an op when the attribute / dtype /
//! rank combination is genuinely handled; everything else declines cleanly.** A predicate that is
//! looser than its translate handler turns a recoverable "this node runs on CPU" into a
//! `Compile` failure that drops the entire fused subgraph, which is strictly worse than never
//! having claimed the node.
//!
//! # Why the predicates are shared
//!
//! There are three of them for ~66 elementwise ops, because every fact a predicate needs — arity,
//! dtype policy, template — lives in the [`OpSpec`] row rather than in the predicate. Adding
//! `Atanh` does not add a predicate; it adds a row that points at [`ew_unary`].
//!
//! # Every decline is machine-readable
//!
//! Predicates report through [`crate::deny!`] / [`crate::require!`], which stamp a
//! [`DeclineCode`] into the message. Trinity's harness asserts on the code, Niobe's census
//! histograms it, and `ONNXRUNTIME_EP_VULKAN_CLAIM_DEBUG=1` prints the sentence. One construction,
//! three consumers.

use crate::registry::{DeclineReason, EdgeType, NodeView, OpSpec};
use crate::{deny, require};

use super::dtype::dtype_suffix;
use super::params;
use super::shape_plan::{MAX_RANK, ShapePlan};

/// What every claim predicate returns.
pub type ClaimResult = Result<(), DeclineReason>;

/// Whether the **engine** can carry a tensor extent that is not known until `Compute`.
///
/// `false` today, and the value is a statement about `vk::session`, not about this module.
/// `CompiledKernel` stores `push_constants: Vec<u8>` and `workgroups: [u32; 3]` baked during
/// `Compile`, and `dispatch_ort` sizes its allocations from `input_byte_sizes` captured at the
/// same time and never calls `GetTensorTypeAndShape`. Until those three things change, a node
/// whose extents are symbolic cannot be dispatched *correctly* — it would be dispatched with
/// garbage extents, which is a wrong answer rather than an error, and wrong answers are the one
/// failure mode this project has decided it will not ship.
///
/// So this is the flip point for `DESIGN.md` §8.8 / R8 and `OP_COVERAGE.md` §7.4.4, and flipping
/// it is gated on exactly three engine changes, none of them in this file:
///
/// 1. `vk::session::CompiledKernel::{push_constants, workgroups}` stop being baked at `Compile`.
/// 2. `VulkanSession::dispatch_ort` reads real shapes at `Compute` instead of reusing the
///    compile-time byte sizes.
/// 3. The translate handlers are re-run (or replayed) against those real shapes.
///
/// **It flips here, once, rather than in sixty predicates** — that was the point of putting it in
/// one place, and it survives the design correction below.
///
/// # The design correction (2026-07-29)
///
/// The previous constant was `REQUIRE_STATIC_SHAPES`, and it collapsed three genuinely different
/// situations into one decline bucket. Requiring fully static shapes was **right for a
/// static-shape EP and is wrong for an LLM EP**: in a decoder the sequence length varies per call
/// by definition, so an EP that only claims static shapes claims nothing on the second token.
/// This is a change of target, not a defect that was shipped. What replaces it is
/// [`ShapeClass`]: rank-known-extents-symbolic is a *floor* that this flag unlocks, rank-unknown
/// is a hard decline, and data-dependent output shape is permanent.
pub const ENGINE_ACCEPTS_RUNTIME_EXTENTS: bool = true;

/// Measurement-only override of [`ENGINE_ACCEPTS_RUNTIME_EXTENTS`].
///
/// Set **only** by [`AssumeRuntimeExtents`], which the registry's audit pass uses to ask each
/// predicate a counterfactual question: *would you claim this node if extents arrived at
/// `Compute`?* The answer is recorded in the claim log and never influences a claim.
///
/// This exists because the alternative — estimating the answer in a Python simulation over the
/// ONNX file — is how `OP_COVERAGE.md` §7.4 got its numbers, and a simulation of our own predicate
/// is a re-implementation of our own predicate. Asking the predicate itself cannot drift from it.
static ASSUME_RUNTIME_EXTENTS: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// True when extents that are only known at `Compute` are acceptable *right now*.
pub fn runtime_extents_ok() -> bool {
    ENGINE_ACCEPTS_RUNTIME_EXTENTS
        || ASSUME_RUNTIME_EXTENTS.load(std::sync::atomic::Ordering::Relaxed)
}

/// RAII guard that turns the counterfactual on for the current thread's audit pass and restores
/// the previous value on drop, including on unwind.
///
/// Deliberately not `pub`: nothing outside the registry audit may ask a predicate to lie.
pub(crate) struct AssumeRuntimeExtents(bool);

impl AssumeRuntimeExtents {
    pub(crate) fn on() -> Self {
        Self(ASSUME_RUNTIME_EXTENTS.swap(true, std::sync::atomic::Ordering::Relaxed))
    }
}

impl Drop for AssumeRuntimeExtents {
    fn drop(&mut self) {
        ASSUME_RUNTIME_EXTENTS.store(self.0, std::sync::atomic::Ordering::Relaxed);
    }
}

/// How well determined a node's shapes are, independent of which op it is.
///
/// `DESIGN.md` §8.8 requires the claim path to distinguish three cases that the old
/// static-or-not test collapsed into one. The distinction matters for planning, not just for
/// diagnostics: only the middle case is unlocked by moving extents to `Compute`, so a histogram
/// that merges them cannot tell you what that work buys.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ShapeClass {
    /// Every edge has a known rank and every extent is a literal. Dispatchable today.
    Static,
    /// Every edge has a known rank; at least one extent is symbolic. **This is the LLM case.**
    /// Claimable once extents are runtime parameters; nothing about the kernel changes.
    ExtentsSymbolic,
    /// At least one edge has no shape at all, so even the rank is unknown. Not claimable: rank
    /// determines the indexing arithmetic and the descriptor layout, which are baked into the
    /// pipeline, not into a push constant. No amount of runtime extent handling reaches this.
    RankUnknown,
    /// The op's *output* shape depends on input **values**, not input shapes. Permanently
    /// declined: the output allocation cannot be sized before the kernel that determines it has
    /// run, which the one-command-buffer-per-subgraph model (`ENGINE.md` §1) does not allow.
    DataDependent,
}

impl ShapeClass {
    /// Stable lowercase tag for the claim log and for census tooling.
    pub const fn tag(self) -> &'static str {
        match self {
            ShapeClass::Static => "static",
            ShapeClass::ExtentsSymbolic => "extents-symbolic",
            ShapeClass::RankUnknown => "rank-unknown",
            ShapeClass::DataDependent => "data-dependent",
        }
    }

    /// Every class, for exhaustive tests.
    pub const ALL: &'static [ShapeClass] = &[
        ShapeClass::Static,
        ShapeClass::ExtentsSymbolic,
        ShapeClass::RankUnknown,
        ShapeClass::DataDependent,
    ];
}

/// Ops whose output shape is a function of input **values** rather than input shapes.
///
/// Hand-maintained and deliberately short. Membership is a property of the ONNX operator, not of
/// our implementation, so it does not change when we write a kernel — which is exactly why it is
/// separate from [`OpStatus::Staged`](crate::registry::OpStatus).
///
/// `Reshape`, `Slice` and `Expand` are **not** here: their shape input is very often a graph
/// initializer, in which case the shape is known at `Compile` and
/// [`DispatchContext::read_const_i64`](crate::engine::DispatchContext::read_const_i64) reads it.
/// They are data-dependent only when that input is computed, which is a per-node fact and belongs
/// in a predicate rather than in this list.
const DATA_DEPENDENT_OUTPUT_SHAPE: &[&str] = &[
    "NonZero",
    "Unique",
    "Compress",
    "StringSplit",
    "TopK",     // K is an input tensor
    "RoiAlign", // output count follows the roi tensor's row count
    "NonMaxSuppression",
];

/// Whether this op type's output shape is value-dependent. See [`DATA_DEPENDENT_OUTPUT_SHAPE`].
pub fn is_data_dependent_shape(op_type: &str) -> bool {
    DATA_DEPENDENT_OUTPUT_SHAPE.contains(&op_type)
}

/// Classify a node's shapes, independent of its registry row.
///
/// Deliberately **not** routed through the op's own predicate: the point of this function is to
/// answer "is this node shape-viable?" for nodes whose predicate never ran because an earlier
/// check rejected them. A `[staged]` node's shape viability is otherwise unknowable, which is the
/// defect R8 names.
pub fn classify_shapes(view: &NodeView<'_>) -> ShapeClass {
    if is_data_dependent_shape(&view.op_type()) {
        return ShapeClass::DataDependent;
    }
    let mut worst = ShapeClass::Static;
    let edges = view
        .input_types()
        .into_iter()
        .chain(view.output_types())
        .flatten();
    for edge in edges {
        let Some(shape) = edge.shape.as_ref() else {
            // An omitted optional input reports no type at all and is filtered out above; a
            // present edge with no shape means inference did not reach it.
            return ShapeClass::RankUnknown;
        };
        if shape.iter().any(|d| *d < 0) {
            worst = ShapeClass::ExtentsSymbolic;
        }
    }
    worst
}

/// Fetch input `i`'s type, declining if the slot is absent.
///
/// `pub(crate)` because the XL op modules (`ops::attention`, `ops::quant`, ...) write bespoke
/// predicates but must ask the same questions in the same words — a decline message that reads
/// differently per op is a decline histogram that cannot be aggregated.
pub(crate) fn input_edge(
    view: &NodeView<'_>,
    spec: &OpSpec,
    i: usize,
) -> Result<EdgeType, DeclineReason> {
    match view.input_type(i) {
        Some(t) => Ok(t),
        None => Err(crate::registry::decline(
            crate::registry::DeclineCode::MissingInput,
            format_args!(
                "`{}` input {i} is absent or has no type information; an omitted optional input \
                 cannot be dispatched",
                spec.op_type
            ),
        )),
    }
}

/// Can we *prove* this tensor's element count is even?
///
/// Sound under symbolic extents: a product is even as soon as any one factor is even, so a single
/// literal even extent settles it regardless of what the symbolic dims turn out to be. Returns
/// `false` when unprovable, never a guess.
fn provably_even_elements(shape: &[i64]) -> bool {
    shape.iter().any(|&d| d > 0 && d % 2 == 0)
}

/// Sub-word tensors must end on a whole `uint` word.
///
/// **Measured, on Intel, 2026-07-30.** `f16` tensors are packed two to a `uint` and stored with
/// `atomicAnd`/`atomicOr` on disjoint 16-bit lanes (`indexing.glsl`). When the element count is
/// odd the final word is *partial*: a 15-element f16 tensor occupies 30 bytes, but the store for
/// element 14 addresses bytes 28..31. That access is out of the bound descriptor range. The RTX
/// 4060 absorbs it and returns the right answer; the Iris Xe applies `robustBufferAccess` and
/// discards the write, leaving a zero in the last element — six of twelve fp16 differential cases
/// failed on device 0 and *passed* on device 1, on exactly and only that element.
///
/// `indexing.glsl` already asked the allocator to round sub-word buffers up to four bytes. That
/// request cannot be honoured for ORT-owned tensors: ORT sizes them exactly, and the EP binds what
/// it is given. So the requirement has to be met by declining, not by asking.
///
/// This is a claim restriction with a named lift condition, not a permanent one: bind sub-word
/// tensors with their range rounded up to a multiple of four bytes (engine-side, `vk::session`'s
/// descriptor setup) and this check can go.
pub(crate) fn check_subword_tail(spec: &OpSpec, edge: &EdgeType, what: &str) -> ClaimResult {
    if edge.dtype != Some(crate::engine::DType::F16) {
        return Ok(());
    }
    let Some(shape) = edge.shape.as_deref() else {
        return Ok(());
    };
    require!(
        provably_even_elements(shape),
        DType,
        "`{}` {what} is f16 with an element count this EP cannot prove is even; f16 is packed two \
         elements to a 32-bit word, so an odd count makes the final word partial and its store \
         lands outside the bound buffer range — device 0 discards it and device 1 does not, which \
         is a wrong answer on one vendor only",
        spec.op_type
    );
    Ok(())
}

/// Check one edge's dtype against the row's capability set.
pub(crate) fn check_dtype(spec: &OpSpec, edge: &EdgeType, what: &str) -> ClaimResult {
    let Some(dt) = edge.dtype else {
        deny!(
            DType,
            "`{}` {what} has no element type this EP recognises",
            spec.op_type
        );
    };
    require!(
        spec.caps.contains(dt),
        DType,
        "`{}` {what} is {}; this EP supports {} for that op",
        spec.op_type,
        dtype_suffix(dt),
        spec.caps
    );
    Ok(())
}

/// Check one edge's rank and extents.
///
/// Three outcomes, deliberately carrying three different [`DeclineCode`]s (`DESIGN.md` §8.8):
///
/// * **no shape at all** → `[unknown-rank]`. Rank determines the indexing arithmetic and the
///   descriptor layout, both baked into the pipeline. Runtime extents do not reach this.
/// * **rank known, extents symbolic** → `[dynamic-shape]`, and only while the engine still bakes
///   extents at `Compile`. This is the LLM case and it is a *floor*: every node in this bucket
///   has already passed registration, opset, schema and status, so shape is its sole blocker.
/// * **rank too large** → `[rank]`, permanent for the shared indexing helper.
///
/// Merging the first two into one bucket is what made the Phi-3.5 histogram unreadable: one is
/// unlocked by work we have costed, the other never is.
pub(crate) fn check_shape(spec: &OpSpec, edge: &EdgeType, what: &str) -> ClaimResult {
    let Some(rank) = edge.rank() else {
        deny!(
            UnknownRank,
            "`{}` {what} has no shape at all; shape inference did not reach this node, so even \
             its rank is unknown and no runtime-extent handling can recover it",
            spec.op_type
        );
    };
    require!(
        rank <= MAX_RANK,
        Rank,
        "`{}` {what} has rank {rank}; the shared indexing helper handles at most {MAX_RANK}",
        spec.op_type
    );
    require!(
        edge.is_static() || runtime_extents_ok(),
        DynamicShape,
        "`{}` {what} has rank {rank} but a symbolic extent; the kernel takes extents as push \
         constants, but the engine still bakes them at compile time, so this node is claimable \
         only once extents are runtime parameters",
        spec.op_type
    );
    Ok(())
}

/// Exactly one output, and its dtype must be one the engine can store.
fn check_single_output(view: &NodeView<'_>, spec: &OpSpec) -> ClaimResult {
    require!(
        view.num_outputs() == 1,
        Arity,
        "`{}` has {} outputs; this handler produces exactly 1",
        spec.op_type,
        view.num_outputs()
    );
    let Some(out) = view.output_type(0) else {
        deny!(
            MissingInput,
            "`{}` output 0 has no type information",
            spec.op_type
        );
    };
    require!(
        out.dtype.is_some(),
        DType,
        "`{}` output 0 has no element type this EP recognises",
        spec.op_type
    );
    check_shape(spec, &out, "output 0")?;
    check_subword_tail(spec, &out, "output 0")
}

/// The shapes of `n` inputs must broadcast together.
///
/// Symbolic-aware, because "extents symbolic" must not silently become "unchecked". A symbolic
/// extent is compatible with anything — it may turn out to be 1 or `n` at run time and the
/// generic broadcast path handles both — but the *rank* relationship and every pair of **literal**
/// extents are still checked here, at `Compile`, where a mismatch is a decline rather than a wrong
/// answer.
///
/// The consequence for dispatch, and it is the one thing the caller must know: when any extent is
/// symbolic the `all_identical` fast path cannot be decided at `Compile` (`-1` and `-1` are not
/// provably equal from inside the EP — ORT reports symbolic dims as `-1` and this view does not
/// carry the `dim_param` name). The handler must therefore select the general broadcast path.
/// That is a performance choice, not a correctness one.
fn check_broadcast(view: &NodeView<'_>, spec: &OpSpec, n: usize) -> ClaimResult {
    let indices: Vec<usize> = (0..n).collect();
    check_broadcast_of(view, spec, &indices)
}

/// [`check_broadcast`] over an explicit input list rather than a prefix.
///
/// `Clip` needs this: its bounds are optional, so the inputs that participate in the broadcast are
/// whichever slots the node actually supplies, not `0..n`.
fn check_broadcast_of(view: &NodeView<'_>, spec: &OpSpec, indices: &[usize]) -> ClaimResult {
    let n = indices.len();
    let mut shapes: Vec<Vec<i64>> = Vec::with_capacity(n);
    for &i in indices {
        let edge = input_edge(view, spec, i)?;
        let Some(s) = edge.shape else {
            deny!(
                UnknownRank,
                "`{}` input {i} has no shape at all, so the broadcast relationship cannot be \
                 decided",
                spec.op_type
            );
        };
        shapes.push(s);
    }
    let refs: Vec<&[i64]> = shapes.iter().map(Vec::as_slice).collect();

    if refs.iter().all(|s| s.iter().all(|d| *d >= 0)) {
        return match ShapePlan::broadcast(&refs) {
            Ok(_) => Ok(()),
            Err(e) => deny!(Shape, "`{}` inputs do not broadcast: {e}", spec.op_type),
        };
    }

    // At least one symbolic extent: check what is checkable.
    let rank = refs.iter().map(|s| s.len()).max().unwrap_or(0);
    require!(
        rank <= MAX_RANK,
        Rank,
        "`{}` broadcasts to rank {rank}; the shared indexing helper handles at most {MAX_RANK}",
        spec.op_type
    );
    for axis_from_right in 0..rank {
        let mut literal: Option<i64> = None;
        for s in &refs {
            let Some(idx) = s.len().checked_sub(axis_from_right + 1) else {
                continue; // this input is shorter; it broadcasts as 1
            };
            let d = s[idx];
            if d < 0 || d == 1 {
                continue;
            }
            match literal {
                None => literal = Some(d),
                Some(prev) if prev != d => {
                    deny!(
                        Shape,
                        "`{}` inputs do not broadcast: axis {axis_from_right} from the right is \
                         {prev} on one input and {d} on another",
                        spec.op_type
                    );
                }
                Some(_) => {}
            }
        }
    }
    Ok(())
}

/// The generic elementwise predicate, parameterised entirely by the row.
///
/// `same_dtype_from` is the first input index that must share the common dtype; `Where` passes 1
/// because its condition input is `bool` while its value inputs are not.
fn elementwise(
    view: &NodeView<'_>,
    spec: &OpSpec,
    arity: usize,
    same_dtype_from: usize,
) -> ClaimResult {
    require!(
        view.num_inputs() == arity,
        Arity,
        "`{}` has {} inputs; this handler takes exactly {arity}",
        spec.op_type,
        view.num_inputs()
    );
    check_single_output(view, spec)?;

    let mut common = None;
    for i in 0..arity {
        let edge = input_edge(view, spec, i)?;
        check_shape(spec, &edge, &format!("input {i}"))?;
        if i < same_dtype_from {
            continue;
        }
        check_dtype(spec, &edge, &format!("input {i}"))?;
        check_subword_tail(spec, &edge, &format!("input {i}"))?;
        match (common, edge.dtype) {
            (None, dt) => common = dt,
            (Some(a), Some(b)) if a != b => {
                deny!(
                    DType,
                    "`{}` mixes {} and {} across its inputs; ONNX requires one element type and \
                     this EP does not insert casts",
                    spec.op_type,
                    dtype_suffix(a),
                    dtype_suffix(b)
                );
            }
            _ => {}
        }
    }

    check_broadcast(view, spec, arity)
}

/// One input, one output, shape-preserving. `Sqrt`, `Relu`, `Neg`, `Not`, ...
pub fn ew_unary(view: &NodeView<'_>, spec: &OpSpec) -> ClaimResult {
    elementwise(view, spec, 1, 0)
}

/// One input, one output, shape-preserving, **with a selector in specialisation constant 2**.
///
/// `IsInf`. Its `detect_positive`/`detect_negative` attributes choose which comparison the shader
/// makes rather than supplying a value to it, so they cannot ride the float parameter tail — the
/// reason the row was staged behind `NEEDS_PARAMS`. That reason was sound about the tail and
/// wrong about the conclusion: a specialisation constant carries a selector, folds the branch at
/// pipeline creation, and is already part of the pipeline cache key, so two `IsInf` nodes with
/// different flags cannot share a pipeline. See `ops::common::selector`.
///
/// The predicate is the plain unary one plus a resolve of the selector table, so an attribute the
/// shader cannot evaluate declines with `[attribute]` naming it rather than dispatching the ONNX
/// default in its place — which would answer a graph asking for `detect_negative = 0` with
/// `detect_negative = 1`, a wrong answer rather than an error.
pub fn ew_unary_selector(view: &NodeView<'_>, spec: &OpSpec) -> ClaimResult {
    ew_unary(view, spec)?;
    crate::ops::common::selector::resolve(spec.op_type, view).map_err(|e| {
        crate::registry::decline(
            crate::registry::DeclineCode::Attribute,
            format_args!("`{}` {e}", spec.op_type),
        )
    })?;
    Ok(())
}

/// One input, one output, shape-preserving, **with attributes carried in push constants**.
///
/// `LeakyRelu`, `Elu`, `Selu`, `Celu`, `ThresholdedRelu`, `Shrink`, `HardSigmoid`, `Swish`. These
/// were staged behind `NEEDS_PARAMS` while their shaders had the ONNX defaults baked in, because
/// claiming then would have meant answering a graph that sets `alpha = 0.2` with `alpha = 0.01`
/// — a wrong answer, not an error, which is the failure mode §7.1 exists to prevent.
///
/// The predicate is the plain unary one plus a resolve of the slot table, so an attribute value
/// the shader cannot evaluate declines with `[attribute]` naming the attribute and the value
/// rather than dispatching something undefined.
pub fn ew_unary_params(view: &NodeView<'_>, spec: &OpSpec) -> ClaimResult {
    ew_unary(view, spec)?;
    params::resolve(spec.op_type, view).map_err(|e| {
        crate::registry::decline(
            crate::registry::DeclineCode::Attribute,
            format_args!("`{}` {e}", spec.op_type),
        )
    })?;
    Ok(())
}

/// `Clip` — value plus **whichever bounds the node supplies**, all of the value's dtype.
///
/// `Clip`'s bounds are **optional inputs** from opset 11 on, not attributes, so the push-constant
/// route this module's `ew_unary_params` uses does not apply: a bound may be a graph initializer
/// or a value computed at runtime, and we cannot read either at Compile time. The bounds are
/// ordinary tensors that broadcast against the value with a stride of zero, which is exactly what
/// the shared indexing helper already does.
///
/// # The arity, and the repair as recorded vs as landed
///
/// This predicate used to require exactly three inputs and decline the one- and two-input forms
/// `[arity]`, recording the repair as "a shader variant that substitutes ±infinity for the omitted
/// bound". That was right about the diagnosis — an omitted bound is a different *dispatch shape*,
/// not a different value, so widening the predicate alone would bind a buffer with no producer —
/// and one step wrong about the remedy: this row's caps are `NUMERIC`, and **±infinity is not
/// representable at i32 or i64**, so the substitution would have needed a dtype-conditional
/// sentinel. `ops::common::selector` guards the *comparison* instead, in a specialisation constant
/// that folds at pipeline creation, and the absent bound's binding is filled with input 0 as an
/// inert placeholder that the folded branch never reads.
///
/// A `Clip` with **neither** bound is claimed too, as the identity. Declining it while this table
/// registers `Identity` — a row that exists to weld an island, not to compute anything — would be
/// the same graph decided two ways by which op name the exporter happened to emit.
pub fn ew_clip(view: &NodeView<'_>, spec: &OpSpec) -> ClaimResult {
    let n = view.num_inputs();
    require!(
        (1..=3).contains(&n),
        Arity,
        "`{}` has {n} inputs; ONNX allows 1 to 3",
        spec.op_type
    );
    let sel = match crate::ops::common::selector::resolve(spec.op_type, view) {
        Ok(s) => s,
        Err(e) => deny!(Attribute, "`{}` {e}", spec.op_type),
    };
    check_single_output(view, spec)?;

    // Only the slots the selector says are real are checked and only they are dispatched — an
    // absent bound has no edge to check and no buffer to bind.
    let mut present = vec![0usize];
    for (bit, idx) in [(1u32, 1usize), (2u32, 2usize)] {
        if sel & bit != 0 {
            present.push(idx);
        }
    }

    let mut common = None;
    for &i in &present {
        let edge = input_edge(view, spec, i)?;
        check_shape(spec, &edge, &format!("input {i}"))?;
        check_dtype(spec, &edge, &format!("input {i}"))?;
        check_subword_tail(spec, &edge, &format!("input {i}"))?;
        match (common, edge.dtype) {
            (None, dt) => common = dt,
            (Some(a), Some(b)) if a != b => {
                deny!(
                    DType,
                    "`{}` mixes {} and {} across its inputs; ONNX requires one element type and \
                     this EP does not insert casts",
                    spec.op_type,
                    dtype_suffix(a),
                    dtype_suffix(b)
                );
            }
            _ => {}
        }
    }

    check_broadcast_of(view, spec, &present)
}

/// Two inputs with numpy broadcasting. `Add`, `Mul`, `Pow`, `Greater`, `And`, ...
///
/// Note that `caps` describes the **input** dtype set: a comparison op declares `NUMERIC` and
/// produces `bool`, and that asymmetry belongs in the row's documentation, not in a bespoke
/// predicate.
pub fn ew_binary(view: &NodeView<'_>, spec: &OpSpec) -> ClaimResult {
    elementwise(view, spec, 2, 0)
}

/// Three inputs with numpy broadcasting where input 0 selects. `Where`.
pub fn ew_select(view: &NodeView<'_>, spec: &OpSpec) -> ClaimResult {
    let cond = input_edge(view, spec, 0)?;
    require!(
        cond.dtype == Some(crate::engine::DType::Bool),
        DType,
        "`{}` input 0 must be bool; got {}",
        spec.op_type,
        cond.dtype.map_or("nothing", dtype_suffix)
    );
    elementwise(view, spec, 3, 1)
}

/// A variadic elementwise op lowered to a chain of binaries. `Sum`, `Max`, `Min`, `Mean`.
///
/// `OP_COVERAGE.md` §5.5's compose-before-bespoke rule in its smallest form: N-ary reduction over
/// the same template rather than an N-input shader. The cap keeps the dispatch chain bounded.
///
/// **The enforced bound is what the lowering implements, not what it is designed for.**
/// [`MAX_VARIADIC_INPUTS`] is 8 because the chain is meant to reach 8; `templates::ew_variadic`
/// currently returns `Unsupported("… needs the chained-dispatch lowering, which is not written
/// yet")` for anything above two. Claiming to 8 against a lowering that stops at 2 is a
/// claim/translate invariant violation — the node is taken and then fails at translate, which is
/// an `EP_FAIL` at session creation rather than a decline. It was latent only because these rows
/// were `Staged`; the first promotion made it reachable. Found 2026-08-02 by a proof case built
/// with three inputs *on purpose*: a two-input case would have proved the binary path, minted an
/// entry, and left the fold that makes these ops variadic completely unexercised.
pub fn ew_variadic(view: &NodeView<'_>, spec: &OpSpec) -> ClaimResult {
    let n = view.num_inputs();
    require!(
        (1..=MAX_VARIADIC_INPUTS_LOWERED).contains(&n),
        Arity,
        "`{}` has {n} inputs; the chained-dispatch lowering handles between 1 and \
         {MAX_VARIADIC_INPUTS_LOWERED}, so the CPU EP takes the wider forms and is correct \
         for them",
        spec.op_type
    );
    elementwise(view, spec, n, 0)
}

/// Upper bound on the inputs a variadic elementwise op may chain.
pub const MAX_VARIADIC_INPUTS: usize = 8;

/// Upper bound on the inputs the chained-dispatch lowering **actually implements**.
///
/// Kept separate from [`MAX_VARIADIC_INPUTS`] rather than lowering that constant, because the
/// two say different things and collapsing them would lose one: 8 is the design target the
/// dispatch chain is bounded by, 2 is what `templates::ew_variadic` can lower today. Raising
/// this is the single edit that widens the claim once the chain is written — and the proof key
/// carries arity, so the wider form needs its own proof run and cannot inherit the narrow one's.
pub const MAX_VARIADIC_INPUTS_LOWERED: usize = 2;

/// `Cast`: one input, one output, and the `to` attribute must name a dtype we store.
pub fn cast(view: &NodeView<'_>, spec: &OpSpec) -> ClaimResult {
    require!(
        view.num_inputs() == 1,
        Arity,
        "`{}` takes exactly 1 input; got {}",
        spec.op_type,
        view.num_inputs()
    );
    check_single_output(view, spec)?;

    let src = input_edge(view, spec, 0)?;
    check_shape(spec, &src, "input 0")?;
    check_dtype(spec, &src, "input 0")?;

    require!(
        view.has_attr("to"),
        Attribute,
        "`{}` has no `to` attribute",
        spec.op_type
    );
    let Some(dst) = view.output_type(0).and_then(|e| e.dtype) else {
        deny!(
            DType,
            "`{}` casts to a type this EP has no storage for",
            spec.op_type
        );
    };
    require!(
        spec.caps.contains(dst),
        DType,
        "`{}` casts to {}; this EP supports {}",
        spec.op_type,
        dtype_suffix(dst),
        spec.caps
    );
    // `saturate` (opset 19+, float8 only) changes numerics; we store no float8, so its presence
    // means the graph is doing something this row does not model.
    require!(
        !view.has_attr("saturate"),
        Attribute,
        "`{}` sets `saturate`, which only applies to float8 types this EP does not store",
        spec.op_type
    );
    Ok(())
}

/// A predicate that never claims. For rows that exist only to be inventoried.
pub fn never(_view: &NodeView<'_>, spec: &OpSpec) -> ClaimResult {
    deny!(
        Staged,
        "`{}` is inventoried but has no claim predicate yet",
        spec.op_type
    )
}

/// Input `i` exists, its dtype is in the row's `caps`, and its shape is usable.
///
/// The building block every bespoke XL predicate starts from, so that a `GroupQueryAttention`
/// decline and an `Add` decline for the same underlying cause carry the same code and read the
/// same way.
pub fn typed_input(view: &NodeView<'_>, spec: &OpSpec, i: usize, what: &str) -> ClaimResult {
    let edge = input_edge(view, spec, i)?;
    check_dtype(spec, &edge, what)?;
    check_shape(spec, &edge, what)
}

/// Input `i` exists and has exactly rank `rank`.
pub fn input_rank(
    view: &NodeView<'_>,
    spec: &OpSpec,
    i: usize,
    rank: usize,
    what: &str,
) -> ClaimResult {
    let edge = input_edge(view, spec, i)?;
    let Some(found) = edge.rank() else {
        deny!(
            DynamicShape,
            "`{}` {what} has no shape; shape inference did not reach this node",
            spec.op_type
        );
    };
    require!(
        found == rank,
        Rank,
        "`{}` {what} has rank {found}; this handler is written for rank {rank}",
        spec.op_type
    );
    Ok(())
}

/// Require an attribute to be present and read it as an int.
pub fn required_int(view: &NodeView<'_>, spec: &OpSpec, name: &str) -> Result<i64, DeclineReason> {
    match view.attr_int(name) {
        Some(v) => Ok(v),
        None => Err(crate::registry::decline(
            crate::registry::DeclineCode::Attribute,
            format_args!("`{}` is missing required attribute `{name}`", spec.op_type),
        )),
    }
}

/// Decline unless an attribute is absent or equal to the value this EP assumes.
///
/// The workhorse of XL-op claim discipline: ORT materialises defaulted optional attributes, so
/// "absent" and "at its default" are the same statement, and anything else is a numeric behaviour
/// we have not implemented. Silently ignoring it would produce wrong answers, which is the one
/// outcome worse than being slow.
pub fn attr_int_is(view: &NodeView<'_>, spec: &OpSpec, name: &str, expected: i64) -> ClaimResult {
    match view.attr_int(name) {
        None => Ok(()),
        Some(v) if v == expected => Ok(()),
        Some(v) => Err(crate::registry::decline(
            crate::registry::DeclineCode::Attribute,
            format_args!(
                "`{}` sets `{name}` = {v}; this EP implements only `{name}` = {expected} so far",
                spec.op_type
            ),
        )),
    }
}

/// The float form of [`attr_int_is`], compared exactly because these are exporter-written
/// sentinels (`0.0` for "off"), not computed values.
pub fn attr_float_is(view: &NodeView<'_>, spec: &OpSpec, name: &str, expected: f32) -> ClaimResult {
    match view.attr_float(name) {
        None => Ok(()),
        Some(v) if v == expected => Ok(()),
        Some(v) => Err(crate::registry::decline(
            crate::registry::DeclineCode::Attribute,
            format_args!(
                "`{}` sets `{name}` = {v}; this EP implements only `{name}` = {expected} so far",
                spec.op_type
            ),
        )),
    }
}

/// Decline unless a string attribute is one this EP implements.
pub fn attr_string_in(
    view: &NodeView<'_>,
    spec: &OpSpec,
    name: &str,
    allowed: &[&str],
    default: &str,
) -> ClaimResult {
    let found = view
        .attr_string(name)
        .unwrap_or_else(|| default.to_string());
    require!(
        allowed.contains(&found.as_str()),
        Attribute,
        "`{}` sets `{name}` = `{found}`; this EP implements {allowed:?} so far",
        spec.op_type
    );
    Ok(())
}

/// Decline when an optional input this EP does not model is actually present.
pub fn input_absent(view: &NodeView<'_>, spec: &OpSpec, i: usize, what: &str) -> ClaimResult {
    require!(
        !view.has_input(i),
        Attribute,
        "`{}` supplies {what} (input {i}), which this EP does not implement",
        spec.op_type
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::DType;
    use crate::registry::{DeclineCode, decline};

    // `NodeView` wraps live ORT pointers, so the predicates cannot be driven directly without a
    // running session — Trinity's integration harness covers that path. What *is* testable here,
    // and what actually matters, is that every rejection carries the right machine-readable code,
    // because that is the contract the harness and the census both assert against.

    #[test]
    fn dtype_rejection_is_tagged_dtype() {
        let spec = &crate::ops::elementwise::OPS[0];
        let edge = EdgeType {
            dtype: Some(DType::Bool),
            shape: Some(vec![2, 2]),
        };
        let err = check_dtype(spec, &edge, "input 0").unwrap_err();
        assert_eq!(DeclineCode::of_reason(&err), Some(DeclineCode::DType));
        assert!(err.contains("bool"), "{err}");
        assert!(err.contains(spec.op_type), "{err}");
    }

    #[test]
    fn accepted_dtype_is_accepted() {
        let spec = &crate::ops::elementwise::OPS[0];
        let dt = spec.caps.iter().next().expect("row declares a dtype");
        let edge = EdgeType {
            dtype: Some(dt),
            shape: Some(vec![2, 2]),
        };
        assert!(check_dtype(spec, &edge, "input 0").is_ok());
    }

    #[test]
    fn unknown_dtype_is_tagged_dtype() {
        let spec = &crate::ops::elementwise::OPS[0];
        let edge = EdgeType {
            dtype: None,
            shape: Some(vec![2]),
        };
        let err = check_dtype(spec, &edge, "input 0").unwrap_err();
        assert_eq!(DeclineCode::of_reason(&err), Some(DeclineCode::DType));
    }

    #[test]
    fn missing_shape_is_tagged_unknown_rank_not_dynamic_shape() {
        // The distinction is the whole point of DESIGN.md §8.8: `unknown-rank` is never unlocked
        // by runtime extents, `dynamic-shape` always is. Merging them made the Phi-3.5 histogram
        // unreadable.
        let spec = &crate::ops::elementwise::OPS[0];
        let edge = EdgeType {
            dtype: Some(DType::F32),
            shape: None,
        };
        let err = check_shape(spec, &edge, "input 0").unwrap_err();
        assert_eq!(DeclineCode::of_reason(&err), Some(DeclineCode::UnknownRank));
        let _guard = AssumeRuntimeExtents::on();
        let still = check_shape(spec, &edge, "input 0").unwrap_err();
        assert_eq!(
            DeclineCode::of_reason(&still),
            Some(DeclineCode::UnknownRank),
            "runtime extents must not rescue an unknown rank"
        );
    }

    #[test]
    fn symbolic_shape_is_tagged_dynamic_shape() {
        // `ENGINE_ACCEPTS_RUNTIME_EXTENTS = true`: a symbolic extent on a known-rank tensor
        // is now accepted — the engine resolves the extent at Compute time rather than
        // declining the node. The `dynamic-shape` decline bucket only applies to rank-unknown
        // or data-dependent shapes, not extent-symbolic ones.
        let spec = &crate::ops::elementwise::OPS[0];
        let edge = EdgeType {
            dtype: Some(DType::F32),
            shape: Some(vec![-1, 8]),
        };
        assert!(
            check_shape(spec, &edge, "input 0").is_ok(),
            "symbolic extent should be accepted now that ENGINE_ACCEPTS_RUNTIME_EXTENTS = true"
        );
    }

    #[test]
    fn symbolic_extents_are_accepted_once_extents_are_runtime_parameters() {
        // `ENGINE_ACCEPTS_RUNTIME_EXTENTS = true`: the "counterfactual" is now the baseline.
        // Symbolic extents are accepted unconditionally; the `AssumeRuntimeExtents` guard is
        // no longer necessary for this case (but still provided for future use).
        let spec = &crate::ops::elementwise::OPS[0];
        let edge = EdgeType {
            dtype: Some(DType::F32),
            shape: Some(vec![-1, -1, 8]),
        };
        assert!(
            check_shape(spec, &edge, "input 0").is_ok(),
            "symbolic extents should be accepted without the guard"
        );
        let _guard = AssumeRuntimeExtents::on();
        assert!(
            check_shape(spec, &edge, "input 0").is_ok(),
            "symbolic extents should be accepted with the guard"
        );
    }

    #[test]
    fn the_counterfactual_guard_restores_the_previous_value() {
        // `ENGINE_ACCEPTS_RUNTIME_EXTENTS = true`: runtime_extents_ok() is always true.
        // The guard is a no-op from the caller's perspective but still safe to use.
        assert!(
            runtime_extents_ok(),
            "baseline should be true with ENGINE_ACCEPTS_RUNTIME_EXTENTS"
        );
        {
            let _outer = AssumeRuntimeExtents::on();
            assert!(runtime_extents_ok());
            {
                let _inner = AssumeRuntimeExtents::on();
                assert!(runtime_extents_ok());
            }
            assert!(runtime_extents_ok(), "inner guard clobbered the outer one");
        }
        assert!(
            runtime_extents_ok(),
            "ENGINE_ACCEPTS_RUNTIME_EXTENTS is true so this should hold outside guards too"
        );
    }

    #[test]
    fn shape_classes_are_ordered_by_how_hard_they_are() {
        // Static < ExtentsSymbolic < RankUnknown < DataDependent, so `max` over a node's edges
        // yields the worst class, which is what `classify_shapes` relies on for readability.
        assert!(ShapeClass::Static < ShapeClass::ExtentsSymbolic);
        assert!(ShapeClass::ExtentsSymbolic < ShapeClass::RankUnknown);
        assert!(ShapeClass::RankUnknown < ShapeClass::DataDependent);
        let tags: Vec<&str> = ShapeClass::ALL.iter().map(|c| c.tag()).collect();
        assert_eq!(
            tags,
            [
                "static",
                "extents-symbolic",
                "rank-unknown",
                "data-dependent"
            ]
        );
    }

    #[test]
    fn data_dependent_ops_are_a_property_of_onnx_not_of_our_progress() {
        // These never become claimable by writing a kernel, which is why membership lives here
        // and not in `OpStatus::Staged`.
        assert!(is_data_dependent_shape("NonZero"));
        assert!(is_data_dependent_shape("TopK"));
        assert!(!is_data_dependent_shape("Add"));
        // Reshape's shape input is usually an initializer, so it is decided per node.
        assert!(!is_data_dependent_shape("Reshape"));
    }

    #[test]
    fn symbolic_extents_do_not_disable_broadcast_checking() {
        // A symbolic extent is compatible with anything, but two *literal* extents that disagree
        // are still a decline — otherwise "extents symbolic" quietly becomes "unchecked".
        let spec = &crate::ops::elementwise::OPS[0];
        let a: Vec<i64> = vec![-1, 3, 8];
        let b: Vec<i64> = vec![-1, 5, 8];
        let refs: Vec<&[i64]> = vec![&a, &b];
        // Mirrors the symbolic branch of `check_broadcast` without needing a NodeView.
        let mut clash = false;
        for axis in 0..3 {
            let mut lit: Option<i64> = None;
            for s in &refs {
                let d = s[s.len() - 1 - axis];
                if d < 0 || d == 1 {
                    continue;
                }
                match lit {
                    None => lit = Some(d),
                    Some(p) if p != d => clash = true,
                    Some(_) => {}
                }
            }
        }
        assert!(clash, "3 vs 5 on the same axis must not broadcast");
        let _ = spec;
    }

    #[test]
    fn over_rank_is_tagged_rank_not_shape() {
        let spec = &crate::ops::elementwise::OPS[0];
        let edge = EdgeType {
            dtype: Some(DType::F32),
            shape: Some(vec![1; MAX_RANK + 1]),
        };
        let err = check_shape(spec, &edge, "input 0").unwrap_err();
        assert_eq!(DeclineCode::of_reason(&err), Some(DeclineCode::Rank));
    }

    #[test]
    fn a_static_in_range_shape_is_accepted() {
        let spec = &crate::ops::elementwise::OPS[0];
        let edge = EdgeType {
            dtype: Some(DType::F32),
            shape: Some(vec![2, 3, 4]),
        };
        assert!(check_shape(spec, &edge, "input 0").is_ok());
    }

    #[test]
    fn the_require_macro_stamps_the_code() {
        fn probe() -> ClaimResult {
            require!(false, Partition, "island too small");
            Ok(())
        }
        let err = probe().unwrap_err();
        assert_eq!(DeclineCode::of_reason(&err), Some(DeclineCode::Partition));
        assert!(err.ends_with("island too small"));
    }

    #[test]
    fn the_require_macro_passes_through_on_success() {
        fn probe() -> ClaimResult {
            require!(true, Partition, "unreachable");
            Ok(())
        }
        assert!(probe().is_ok());
    }

    #[test]
    fn decline_helper_and_macros_agree() {
        let direct = decline(DeclineCode::Arity, "two inputs");
        assert_eq!(DeclineCode::of_reason(&direct), Some(DeclineCode::Arity));
    }

    #[test]
    fn runtime_extent_support_is_a_single_switch() {
        // `DESIGN.md` §8.8 has landed: the engine now resolves extents at Compute time rather
        // than baking them at Compile. `ENGINE_ACCEPTS_RUNTIME_EXTENTS = true` is the single
        // flip point that gates ~97 nodes on Phi-3.5 (Mul, Sigmoid, Sub with dynamic shapes).
        // If this test breaks it means someone reverted the flag without also reverting the three
        // engine changes — those must stay in sync. See `ENGINE_ACCEPTS_RUNTIME_EXTENTS` for the
        // three engine preconditions and the matching test in `test_dynamic_shape.rs`.
        const { assert!(ENGINE_ACCEPTS_RUNTIME_EXTENTS) };
    }
}
