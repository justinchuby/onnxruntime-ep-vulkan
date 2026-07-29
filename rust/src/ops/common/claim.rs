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
use super::shape_plan::{MAX_RANK, ShapePlan};

/// What every claim predicate returns.
pub type ClaimResult = Result<(), DeclineReason>;

/// Whether a fully static shape is required before an op may be claimed.
///
/// `true` today, and deliberately so. Push constants are filled at `Compile` time from the
/// extracted `NodeDesc`, so an op whose extents are only known at execution cannot yet be
/// dispatched correctly. Shape-agnostic recording (`OP_COVERAGE.md` OQ-M1) is what flips this, and
/// when it does it flips **here**, once, rather than in sixty predicates.
///
/// The honest consequence: until OQ-M1 lands, a decoder graph with symbolic `batch`/`seq` declines
/// with `[dynamic-shape]`. That shows up in the decline histogram as a single dominant bucket,
/// which is exactly the signal we want it to produce.
pub const REQUIRE_STATIC_SHAPES: bool = true;

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

/// Check one edge's rank and staticness.
pub(crate) fn check_shape(spec: &OpSpec, edge: &EdgeType, what: &str) -> ClaimResult {
    let Some(rank) = edge.rank() else {
        deny!(
            DynamicShape,
            "`{}` {what} has no shape; shape inference did not reach this node",
            spec.op_type
        );
    };
    require!(
        rank <= MAX_RANK,
        Rank,
        "`{}` {what} has rank {rank}; the shared indexing helper handles at most {MAX_RANK}",
        spec.op_type
    );
    if REQUIRE_STATIC_SHAPES {
        require!(
            edge.is_static(),
            DynamicShape,
            "`{}` {what} has a symbolic dimension; this EP fills push constants at compile time, \
             so extents must be known then",
            spec.op_type
        );
    }
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
    check_shape(spec, &out, "output 0")
}

/// The shapes of `n` inputs must broadcast together.
fn check_broadcast(view: &NodeView<'_>, spec: &OpSpec, n: usize) -> ClaimResult {
    if !REQUIRE_STATIC_SHAPES {
        return Ok(());
    }
    let mut shapes: Vec<Vec<i64>> = Vec::with_capacity(n);
    for i in 0..n {
        let edge = input_edge(view, spec, i)?;
        let Some(s) = edge.shape else {
            deny!(DynamicShape, "`{}` input {i} has no shape", spec.op_type);
        };
        shapes.push(s);
    }
    let refs: Vec<&[i64]> = shapes.iter().map(Vec::as_slice).collect();
    match ShapePlan::broadcast(&refs) {
        Ok(_) => Ok(()),
        Err(e) => deny!(Shape, "`{}` inputs do not broadcast: {e}", spec.op_type),
    }
}

/// The generic elementwise predicate, parameterised entirely by the row.
///
/// `same_dtype_from` is the first input index that must share the common dtype; `Where` passes 1
/// because its condition input is `bool` while its value inputs are not.
fn elementwise(view: &NodeView<'_>, spec: &OpSpec, arity: usize, same_dtype_from: usize) -> ClaimResult {
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
pub fn ew_variadic(view: &NodeView<'_>, spec: &OpSpec) -> ClaimResult {
    let n = view.num_inputs();
    require!(
        (1..=MAX_VARIADIC_INPUTS).contains(&n),
        Arity,
        "`{}` has {n} inputs; this handler chains between 1 and {MAX_VARIADIC_INPUTS}",
        spec.op_type
    );
    elementwise(view, spec, n, 0)
}

/// Upper bound on the inputs a variadic elementwise op may chain.
pub const MAX_VARIADIC_INPUTS: usize = 8;

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
pub fn input_rank(view: &NodeView<'_>, spec: &OpSpec, i: usize, rank: usize, what: &str) -> ClaimResult {
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
            format_args!(
                "`{}` is missing required attribute `{name}`",
                spec.op_type
            ),
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
    let found = view.attr_string(name).unwrap_or_else(|| default.to_string());
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
    fn missing_shape_is_tagged_dynamic_shape() {
        let spec = &crate::ops::elementwise::OPS[0];
        let edge = EdgeType {
            dtype: Some(DType::F32),
            shape: None,
        };
        let err = check_shape(spec, &edge, "input 0").unwrap_err();
        assert_eq!(DeclineCode::of_reason(&err), Some(DeclineCode::DynamicShape));
    }

    #[test]
    fn symbolic_shape_is_tagged_dynamic_shape() {
        let spec = &crate::ops::elementwise::OPS[0];
        let edge = EdgeType {
            dtype: Some(DType::F32),
            shape: Some(vec![-1, 8]),
        };
        let err = check_shape(spec, &edge, "input 0").unwrap_err();
        assert_eq!(DeclineCode::of_reason(&err), Some(DeclineCode::DynamicShape));
        assert!(err.contains("symbolic"), "{err}");
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
    fn static_shape_policy_is_a_single_switch() {
        // If this ever flips, OQ-M1 landed and the decline histogram's dominant bucket changes.
        const { assert!(REQUIRE_STATIC_SHAPES) };
    }
}
