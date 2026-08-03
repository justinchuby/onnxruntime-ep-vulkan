//! Tier-1 elementwise ops — the table that proves the machinery.
//!
//! Every row here is an entry in `OP_COVERAGE.md` §4's inventory, and every row is one line. That
//! is the whole claim of §5: three GLSL templates and one shared broadcasting helper cover roughly
//! a third of the op inventory, so the schedule is decided by how fast rows can be added, not by
//! how fast kernels can be written.
//!
//! # The elementwise f32 family is live; everything else is [`Staged`]
//!
//! The three GLSL templates exist (`shaders/glsl/ew_{unary,binary,select}.comp`) and all 168
//! variants compile. As of 2026-07-29, **35 of them have executed on both local devices and
//! matched ORT's CPU EP** through the real ORT wire — see [`EXERCISED`], which names the test for
//! each. `add_f32` went first, alone; the rest of the f32 elementwise arithmetic family followed in
//! one step via [`TEMPLATE_LIVE`], because a `Staged` row's differential test cannot compare
//! anything and so cannot produce the evidence that would justify flipping it.
//!
//! Every live row is live for **f32 only** ([`ew_binary_f32`], [`ew_unary_f32`]). Comparison and
//! logic ops are not in the set: their output dtype differs from their input, which is a different
//! store path in the shader rather than a different one-line expression. Nor are the variadic ops
//! ([`claim::ew_variadic`] issues several dispatches) or `Where` ([`claim::ew_select`]). Every
//! other row still declines and the CPU EP runs those nodes, which is always correct.
//!
//! What exists for the staged rows is the full description — opset window, dtype capabilities,
//! template, variant stems, claim predicate, translate handler, and the shader itself — all of it
//! unit tested. Flipping a row live once a differential test has executed its variant is a
//! one-word diff, plus its dtype in [`EXERCISED`].
//!
//! Three staging reasons appear below and they mean different things:
//!
//! * [`UNEXERCISED`] — the row and its shader are complete; nothing has executed it yet.
//! * [`NEEDS_PARAMS`] — the op's attribute is a **selector** (`fmod`, `direction`,
//!   `detect_negative`) rather than a coefficient: it chooses a different expression, so it needs
//!   its own shader variant and cannot ride the push-constant parameter tail that retired this
//!   blocker for the float-parameter activations. Claiming it with [`claim::never`] is the honest
//!   answer: `OP_COVERAGE.md` §7's rule is that an op is claimed only when the *attribute*
//!   combination is genuinely handled, and here it is not. See §5.1.1 for the float/selector
//!   distinction.
//! * [`NEEDS_CAST_MATRIX`] — the variant space is keyed on a dtype *pair*.

use crate::engine::DType;
use crate::kernel;
use crate::ops::common::claim::{self, ClaimResult};
use crate::ops::common::dtype::{ANY, BOOL, DTypeSet, F32, FLOAT, INT, NUMERIC, dtype_suffix};
use crate::ops::common::templates;
use crate::registry::OpStatus::{Live, Ready, Staged};
use crate::registry::{NodeView, OPSET_ANY, OPSET_STD_SWISH, OpSpec, UNEXERCISED};

/// `Equal` compares booleans as well as numbers.
const EQ_CAPS: DTypeSet = NUMERIC.union(BOOL);

/// Staging reason for ops whose attribute *selects an expression* rather than supplying a value.
///
/// The push-constant parameter tail (`ops::common::params`, `OP_COVERAGE.md` §5.1.1) carries
/// floats, which retired this blocker for `LeakyRelu`, `Elu`, `Selu`, `Celu`, `ThresholdedRelu`,
/// `Shrink`, `HardSigmoid` and `Swish`. It does nothing for `Mod`'s `fmod`, `BitShift`'s
/// `direction` or `IsInf`'s two detect flags: each picks between two different arithmetics, which
/// is a shader variant, not a value.
pub const NEEDS_PARAMS: &str = "its attribute selects a different expression rather than supplying a value, so it needs its own \
     shader variant rather than a push-constant parameter";

/// Staging reason for `Cast`, whose variant space is keyed on a dtype *pair*.
pub const NEEDS_CAST_MATRIX: &str = "its shader variant space is keyed on a source/destination dtype pair rather than a single \
     dtype, so it needs its own template and manifest column";

/// The `(op, dtype)` pairs that have actually executed on a device through the ORT wire.
///
/// This is the evidence list behind every [`Live`] row in this module, and it is deliberately a
/// *list of pairs* rather than a flag on the row. A row's `caps` describes what the kernel family
/// is written for and drives shader-variant generation; it is not a statement that every one of
/// those variants has run. `Add` declares `NUMERIC`, so `add_{f32,f16,i32,i64}` are all compiled
/// and all shape-checked — but only `add_f32` has been dispatched.
///
/// Evidence, as of 2026-07-29:
///
/// * `Add`/f32 — `add_f32_dispatches_end_to_end`, the crate's own device test: the shader executes
///   and computes the right answer on both local devices (Intel Iris Xe 1.4.309, NVIDIA RTX 4060
///   Laptop 1.4.325), validation layers on, zero errors on either. Then
///   `tests/ops/test_claim_diagnostics.py::test_add_is_claimed`, which reads ORT's profiling JSON
///   and requires the node to have run on `VulkanExecutionProvider` — verified by the coordinator
///   on each device rather than reported.
/// * **Every other pair here** — `tests/ops/test_op_table.py::test_op_table[<Op>-fp32]`, run on
///   device 0 and device 1: 39 passed on each, no numerical mismatch against the CPU EP oracle.
///   Each of those tests asserts placement on `VulkanExecutionProvider` *before* comparing, so a
///   pass is not the vacuous CPU-fallback pass.
///
/// The wire — `Compile` → `OrtNodeComputeInfo` → `VulkanSession::dispatch_ort`, with the plan built
/// from the **fused node** whose edge order `KernelContext_GetInput/GetOutput` index by — is what
/// all of these travel, and it is now carried real ORT tensors on two vendors.
///
/// Adding a pair here without a named test that ran it on a device is the same category of error
/// as widening a claim predicate to make a coverage number look better.
pub const EXERCISED: &[(&str, &str)] = &[
    // Binary arithmetic.
    ("Add", "f32"),
    ("Sub", "f32"),
    ("Mul", "f32"),
    ("Div", "f32"),
    ("Pow", "f32"),
    // Unary maths.
    ("Abs", "f32"),
    ("Neg", "f32"),
    ("Reciprocal", "f32"),
    ("Sqrt", "f32"),
    ("Exp", "f32"),
    ("Log", "f32"),
    ("Sin", "f32"),
    ("Cos", "f32"),
    ("Tan", "f32"),
    ("Asin", "f32"),
    ("Acos", "f32"),
    ("Atan", "f32"),
    ("Sinh", "f32"),
    ("Cosh", "f32"),
    ("Tanh", "f32"),
    ("Asinh", "f32"),
    ("Acosh", "f32"),
    ("Atanh", "f32"),
    ("Ceil", "f32"),
    ("Floor", "f32"),
    ("Round", "f32"),
    ("Sign", "f32"),
    ("Erf", "f32"),
    ("Identity", "f32"),
    // Attribute-free activations.
    ("Relu", "f32"),
    ("Sigmoid", "f32"),
    ("HardSwish", "f32"),
    ("Softplus", "f32"),
    ("Softsign", "f32"),
    ("Mish", "f32"),
    // Parameterised activations — attributes carried in the push-constant tail
    // (`ops::common::params`). These are listed here rather than in `TEMPLATE_LIVE` because the
    // tail is a *new code path*, not a new line of arithmetic inside an exercised one: a wrong
    // offset for `params[0]` would be invisible to every op above, all of which push zeros there
    // and read none of them. They earned their place by executing, with non-default attribute
    // values, on both devices.
    ("HardSigmoid", "f32"),
    ("LeakyRelu", "f32"),
    ("Elu", "f32"),
    ("Selu", "f32"),
    ("Celu", "f32"),
    ("ThresholdedRelu", "f32"),
    ("Shrink", "f32"),
    ("Gelu", "f32"),
    // Three-input `Clip` — the ternary template's first execution.
    ("Clip", "f32"),
    // ---- f16, 2026-07-30 ------------------------------------------------------------------
    // The dtype that actually matters: a Phi-3.5 decoder is f16 throughout, so the f32 rows above
    // are worth zero nodes on it. Evidence: `tests/ops/test_op_table.py::test_op_table[<Op>-fp16]`
    // on device 0 (Intel Iris Xe) and device 1 (NVIDIA RTX 4060), each asserting placement on
    // `VulkanExecutionProvider` before comparing against the CPU EP at `FP16_ANY` tolerance.
    //
    // These are listed as their own pairs rather than inferred from the f32 rows because f16 is a
    // **different storage path**, not a different expression: the tensors are packed two to a
    // `uint` word and stored through `atomicAnd`/`atomicOr` on disjoint 16-bit lanes. A wrong lane
    // or a wrong word index is invisible to every f32 row here. Same reasoning as the parameter
    // tail above.
    ("Add", "f16"),
    ("Sub", "f16"),
    ("Mul", "f16"),
    ("Div", "f16"),
    ("Relu", "f16"),
    ("Sigmoid", "f16"),
    ("Sqrt", "f16"),
    ("Exp", "f16"),
    ("Tanh", "f16"),
    ("Erf", "f16"),
    ("Gelu", "f16"),
];

/// Rows that are live on **template evidence** rather than on their own dispatch.
///
/// Each entry is `(op, the EXERCISED op whose dispatch stands in for it)`. This is a deliberately
/// weaker claim than [`EXERCISED`] and it is kept in a separate list so that nobody can mistake
/// one for the other — including me, six weeks from now.
///
/// **Currently empty, and the reason is the point.** Thirty-four rows sat here for the length of
/// one edit on 2026-07-29 — the f32 elementwise arithmetic family, standing on `Add` — and were
/// promoted into [`EXERCISED`] the same day, because flipping them is what allowed the differential
/// suite to execute them at all. It stays defined because that transition is the mechanism, not an
/// accident: the next family (bool-output comparisons, the variadic ops, `Where`, or f16 across the
/// board) will go through exactly the same two steps.
///
/// **What it asserts.** The ORT wire — `GetCapability` → `Compile` → `OrtNodeComputeInfo` →
/// `VulkanSession::dispatch_ort` — has carried a real ORT tensor to a real device, and every row
/// here reaches that wire through the *same* `translate` function, the *same* template, the *same*
/// descriptor layout and the *same* push-constant block as its representative. What differs
/// between `add_f32` and `mul_f32` is one line of GLSL inside a body the build pipeline generates
/// from one source.
///
/// **What it does not assert.** That the one line is right. `Div` by zero, `Pow` of a negative
/// base, `Log` of zero and the exact rounding of `Round` are each their own arithmetic, and no
/// amount of `Add` passing says anything about them. That is what Trinity's differential suite
/// against the CPU EP is for, and flipping these rows is what lets it run at all: while a row is
/// `Staged`, its test fails with *"the EP executed no node — the CPU-match check would be a
/// vacuous pass"*, which is loud but proves nothing. Those failures turn into real comparisons the
/// moment the rows go live, and if a shader is wrong the suite says so on the next run.
///
/// **Why this is not the slippery slope [`EXERCISED`] exists to prevent.** The rule is narrow and
/// checkable: a row may sit here only if (a) its representative is in [`EXERCISED`] and live, (b)
/// it goes through that representative's exact template, and (c) its claim predicate is narrowed
/// to the representative's dtype. It buys nothing for `Sub` at i64, for `Where`, for the variadic
/// ops, or for anything whose output dtype differs from its input — those are different code
/// paths, not different arithmetic, and they stay staged. And an entry that has not been promoted
/// by the next differential run is evidence that the run is not covering it. See
/// `OP_COVERAGE.md` §7.1.2.
pub const TEMPLATE_LIVE: &[(&str, &str)] = &[];

/// [`ew_binary`](claim::ew_binary), narrowed to the dtype that has executed.
///
/// `OP_COVERAGE.md` §7.1: claim an op only when the attribute/dtype/rank combination is genuinely
/// handled. The rows using this are `NUMERIC` because the *template* is, and because the variant
/// table reads `caps` to decide what to compile. Claiming on `caps` alone would mean claiming
/// `add_i64` on the strength of `add_f32` having run, which is a bet on three unexecuted variants
/// dressed up as a capability.
///
/// The decline is `[dtype]`, so an f16 decoder graph shows up in Niobe's histogram as a dtype
/// bucket rather than as a mystery. That bucket emptying is the signal that the f16 variant is
/// worth exercising next.
fn ew_binary_f32(view: &NodeView<'_>, spec: &OpSpec) -> ClaimResult {
    claim::ew_binary(view, spec)?;
    only_proved_dtypes(view, spec)
}

/// [`ew_unary`](claim::ew_unary), narrowed the same way and for the same reason.
fn ew_unary_f32(view: &NodeView<'_>, spec: &OpSpec) -> ClaimResult {
    claim::ew_unary(view, spec)?;
    only_proved_dtypes(view, spec)
}

/// The shared narrowing: input 0's dtype must be one this op has been *proved* at.
///
/// Reads [`EXERCISED`] directly rather than hardcoding f32. That is the whole mechanism: the
/// evidence list and the claim predicate can no longer disagree, because they are the same list.
/// Adding `("Mul", "f16")` after a differential run against the CPU EP is then the single act that
/// widens the claim — there is no second place to remember, and no way to widen a claim without
/// writing down the evidence that justifies it.
///
/// A [`TEMPLATE_LIVE`] row inherits its representative's proved dtypes and nothing more, which is
/// the same weaker-but-explicit claim that list has always made.
///
/// The decline is `[dtype]`, so an f16 decoder graph shows up in Niobe's histogram as a dtype
/// bucket rather than as a mystery. That bucket emptying is the signal that the variant is worth
/// exercising next.
fn only_proved_dtypes(view: &NodeView<'_>, spec: &OpSpec) -> ClaimResult {
    let edge = claim::input_edge(view, spec, 0)?;
    let dt = edge.dtype;
    let proved = dt.is_some_and(|d| proved_at(spec.op_type, d));
    crate::require!(
        proved,
        DType,
        "`{}` is live for {}; this node is {}. That variant of the elementwise shader compiles \
         but has never executed on a device, and the CPU EP is correct for it",
        spec.op_type,
        proved_list(spec.op_type),
        dt.map_or("untyped", dtype_suffix),
    );
    Ok(())
}

/// Has `op` executed at dtype `d` on a device, directly or through its template representative?
pub fn proved_at(op: &str, d: DType) -> bool {
    let suffix = dtype_suffix(d);
    let representative = TEMPLATE_LIVE
        .iter()
        .find(|(row, _)| *row == op)
        .map_or(op, |(_, rep)| *rep);
    EXERCISED
        .iter()
        .any(|(e, dt)| *e == representative && *dt == suffix)
}

/// The dtypes `op` is proved at, for the decline message.
fn proved_list(op: &str) -> String {
    let representative = TEMPLATE_LIVE
        .iter()
        .find(|(row, _)| *row == op)
        .map_or(op, |(_, rep)| *rep);
    let mut got: Vec<&str> = EXERCISED
        .iter()
        .filter(|(e, _)| *e == representative)
        .map(|(_, dt)| *dt)
        .collect();
    got.sort_unstable();
    if got.is_empty() {
        return "no dtype".to_string();
    }
    got.join("/")
}

/// `ai.onnx::Swish` (opset 24) — any `alpha`, now that the parameter tail carries it.
///
/// This row was previously pinned to `alpha = 1` (SiLU) because the unary template had no
/// push-constant slot for an attribute. It now has one, so the honest row is the general op.
/// The pin is kept in the tests as a record of *why* it existed, not as a constraint.
fn ew_unary_params_f32(view: &NodeView<'_>, spec: &OpSpec) -> ClaimResult {
    claim::ew_unary_params(view, spec)?;
    only_proved_dtypes(view, spec)
}

/// `Gelu` — claim `approximate = "none"` (the exact erf form), decline `"tanh"`.
///
/// `approximate` is a **string**, so unlike `alpha` it cannot ride the parameter tail: it selects
/// a different expression, not a different coefficient. The right answer is a second shader
/// variant, which is a shader-table change rather than a predicate one. Until then the default
/// form is claimed — and it is claimed honestly, because `"none"` is the value the shader
/// actually implements rather than a value we assumed because it was the default.
fn gelu(view: &NodeView<'_>, spec: &OpSpec) -> ClaimResult {
    claim::ew_unary(view, spec)?;
    claim::attr_string_in(view, spec, "approximate", &["none"], "none")?;
    only_proved_dtypes(view, spec)
}

/// `Clip` — the three-input form only, f32. See [`claim::ew_clip`] for why the shorter forms
/// decline rather than defaulting the omitted bound.
fn clip_f32(view: &NodeView<'_>, spec: &OpSpec) -> ClaimResult {
    claim::ew_clip(view, spec)?;
    only_proved_dtypes(view, spec)
}

crate::op_table! {
    // ---------------------------------------------------------------------------------------
    // Binary arithmetic, logic and comparison — one template, numpy broadcasting for free.
    //
    //  op            domain  opset window        caps      kernel                       claim               translate               status
    // ---------------------------------------------------------------------------------------

    // `Add` is the one row live on its own dispatch: `add_f32_dispatches_end_to_end` ran it on
    // both local devices. Live for **f32 only** — `caps` stays NUMERIC because that is what the
    // template and the variant table are for, and `ew_binary_f32` narrows the claim to the variant
    // that has actually run. See `EXERCISED`.
    //
    // The rows below it marked with `ew_binary_f32` / `ew_unary_f32` are live on *template*
    // evidence: same template, same wire, same dtype, different one-line body. See `TEMPLATE_LIVE`
    // for exactly what that does and does not assert. Comparison and logic ops are NOT in that set
    // — their output dtype differs from their input, which is a different store path in the
    // shader, not a different expression.
    "Add",            Ai,     7 ..= OPSET_ANY,    NUMERIC,  kernel!(EwBinary, "add"),    ew_binary_f32,      templates::ew_binary,   Live;
    "Sub",            Ai,     7 ..= OPSET_ANY,    NUMERIC,  kernel!(EwBinary, "sub"),    ew_binary_f32,      templates::ew_binary,   Live;
    "Mul",            Ai,     7 ..= OPSET_ANY,    NUMERIC,  kernel!(EwBinary, "mul"),    ew_binary_f32,      templates::ew_binary,   Live;
    "Div",            Ai,     7 ..= OPSET_ANY,    NUMERIC,  kernel!(EwBinary, "div"),    ew_binary_f32,      templates::ew_binary,   Live;
    "Pow",            Ai,     7 ..= OPSET_ANY,    FLOAT,    kernel!(EwBinary, "pow"),    ew_binary_f32,      templates::ew_binary,   Live;
    "Mod",            Ai,     10 ..= OPSET_ANY,   NUMERIC,  kernel!(EwBinary, "mod"),    claim::never,       templates::unimplemented, Staged(NEEDS_PARAMS);
    "And",            Ai,     7 ..= OPSET_ANY,    BOOL,     kernel!(EwBinary, "and"),    claim::ew_binary,   templates::ew_binary,   Ready;
    "Or",             Ai,     7 ..= OPSET_ANY,    BOOL,     kernel!(EwBinary, "or"),     claim::ew_binary,   templates::ew_binary,   Ready;
    "Xor",            Ai,     7 ..= OPSET_ANY,    BOOL,     kernel!(EwBinary, "xor"),    claim::ew_binary,   templates::ew_binary,   Ready;
    "BitwiseAnd",     Ai,     18 ..= OPSET_ANY,   INT,      kernel!(EwBinary, "bitand"), claim::ew_binary,   templates::ew_binary,   Ready;
    "BitwiseOr",      Ai,     18 ..= OPSET_ANY,   INT,      kernel!(EwBinary, "bitor"),  claim::ew_binary,   templates::ew_binary,   Ready;
    "BitwiseXor",     Ai,     18 ..= OPSET_ANY,   INT,      kernel!(EwBinary, "bitxor"), claim::ew_binary,   templates::ew_binary,   Ready;
    "BitShift",       Ai,     11 ..= OPSET_ANY,   INT,      kernel!(EwBinary, "bitshift"), claim::never,     templates::unimplemented, Staged(NEEDS_PARAMS);
    "Equal",          Ai,     7 ..= OPSET_ANY,    EQ_CAPS,  kernel!(EwBinary, "eq"),     claim::ew_binary,   templates::ew_binary,   Ready;
    "Greater",        Ai,     7 ..= OPSET_ANY,    NUMERIC,  kernel!(EwBinary, "gt"),     claim::ew_binary,   templates::ew_binary,   Ready;
    "GreaterOrEqual", Ai,     12 ..= OPSET_ANY,   NUMERIC,  kernel!(EwBinary, "ge"),     claim::ew_binary,   templates::ew_binary,   Ready;
    "Less",           Ai,     7 ..= OPSET_ANY,    NUMERIC,  kernel!(EwBinary, "lt"),     claim::ew_binary,   templates::ew_binary,   Ready;
    "LessOrEqual",    Ai,     12 ..= OPSET_ANY,   NUMERIC,  kernel!(EwBinary, "le"),     claim::ew_binary,   templates::ew_binary,   Ready;
    "PRelu",          Ai,     7 ..= OPSET_ANY,    FLOAT,    kernel!(EwBinary, "prelu"),  claim::ew_binary,   templates::ew_binary,   Ready;

    // ---------------------------------------------------------------------------------------
    // Unary maths — the longest run of pure table rows in the crate.
    // ---------------------------------------------------------------------------------------
    "Abs",            Ai,     6 ..= OPSET_ANY,    NUMERIC,  kernel!(EwUnary, "abs"),     ew_unary_f32,       templates::ew_unary,    Live;
    "Neg",            Ai,     6 ..= OPSET_ANY,    NUMERIC,  kernel!(EwUnary, "neg"),     ew_unary_f32,       templates::ew_unary,    Live;
    "Reciprocal",     Ai,     6 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "recip"),   ew_unary_f32,       templates::ew_unary,    Live;
    "Sqrt",           Ai,     6 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "sqrt"),    ew_unary_f32,       templates::ew_unary,    Live;
    "Exp",            Ai,     6 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "exp"),     ew_unary_f32,       templates::ew_unary,    Live;
    "Log",            Ai,     6 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "log"),     ew_unary_f32,       templates::ew_unary,    Live;
    "Sin",            Ai,     7 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "sin"),     ew_unary_f32,       templates::ew_unary,    Live;
    "Cos",            Ai,     7 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "cos"),     ew_unary_f32,       templates::ew_unary,    Live;
    "Tan",            Ai,     7 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "tan"),     ew_unary_f32,       templates::ew_unary,    Live;
    "Asin",           Ai,     7 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "asin"),    ew_unary_f32,       templates::ew_unary,    Live;
    "Acos",           Ai,     7 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "acos"),    ew_unary_f32,       templates::ew_unary,    Live;
    "Atan",           Ai,     7 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "atan"),    ew_unary_f32,       templates::ew_unary,    Live;
    "Sinh",           Ai,     9 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "sinh"),    ew_unary_f32,       templates::ew_unary,    Live;
    "Cosh",           Ai,     9 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "cosh"),    ew_unary_f32,       templates::ew_unary,    Live;
    "Tanh",           Ai,     6 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "tanh"),    ew_unary_f32,       templates::ew_unary,    Live;
    "Asinh",          Ai,     9 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "asinh"),   ew_unary_f32,       templates::ew_unary,    Live;
    "Acosh",          Ai,     9 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "acosh"),   ew_unary_f32,       templates::ew_unary,    Live;
    "Atanh",          Ai,     9 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "atanh"),   ew_unary_f32,       templates::ew_unary,    Live;
    "Ceil",           Ai,     6 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "ceil"),    ew_unary_f32,       templates::ew_unary,    Live;
    "Floor",          Ai,     6 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "floor"),   ew_unary_f32,       templates::ew_unary,    Live;
    "Round",          Ai,     11 ..= OPSET_ANY,   FLOAT,    kernel!(EwUnary, "round"),   ew_unary_f32,       templates::ew_unary,    Live;
    "Sign",           Ai,     9 ..= OPSET_ANY,    NUMERIC,  kernel!(EwUnary, "sign"),    ew_unary_f32,       templates::ew_unary,    Live;
    "Erf",            Ai,     9 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "erf"),     ew_unary_f32,       templates::ew_unary,    Live;
    "Not",            Ai,     1 ..= OPSET_ANY,    BOOL,     kernel!(EwUnary, "not"),     claim::ew_unary,    templates::ew_unary,    Ready;
    "BitwiseNot",     Ai,     18 ..= OPSET_ANY,   INT,      kernel!(EwUnary, "bitnot"),  claim::ew_unary,    templates::ew_unary,    Ready;
    "IsNaN",          Ai,     9 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "isnan"),   claim::ew_unary,    templates::ew_unary,    Ready;
    "IsInf",          Ai,     10 ..= OPSET_ANY,   FLOAT,    kernel!(EwUnary, "isinf"),   claim::never,       templates::unimplemented, Staged(NEEDS_PARAMS);
    "Identity",       Ai,     1 ..= OPSET_ANY,    ANY,      kernel!(EwUnary, "identity"), ew_unary_f32,      templates::ew_unary,    Live;

    // ---------------------------------------------------------------------------------------
    // Activations. The parameterised ones read their attributes from the push-constant tail
    // (`ops::common::params`); before that tail existed they were staged behind NEEDS_PARAMS
    // rather than claimed with the ONNX default silently substituted for whatever the graph set.
    // ---------------------------------------------------------------------------------------
    "Relu",           Ai,     6 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "relu"),    ew_unary_f32,       templates::ew_unary,    Live;
    "Sigmoid",        Ai,     6 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "sigmoid"), ew_unary_f32,       templates::ew_unary,    Live;
    "HardSwish",      Ai,     14 ..= OPSET_ANY,   FLOAT,    kernel!(EwUnary, "hardswish"), ew_unary_f32,     templates::ew_unary,    Live;
    "Softplus",       Ai,     1 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "softplus"), ew_unary_f32,      templates::ew_unary,    Live;
    "Softsign",       Ai,     1 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "softsign"), ew_unary_f32,      templates::ew_unary,    Live;
    "Mish",           Ai,     18 ..= OPSET_ANY,   FLOAT,    kernel!(EwUnary, "mish"),    ew_unary_f32,       templates::ew_unary,    Live;
    "HardSigmoid",    Ai,     6 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "hardsigmoid"), ew_unary_params_f32, templates::ew_unary, Live;
    "LeakyRelu",      Ai,     6 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "leakyrelu"), ew_unary_params_f32, templates::ew_unary, Live;
    "Elu",            Ai,     6 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "elu"),     ew_unary_params_f32, templates::ew_unary,   Live;
    "Selu",           Ai,     6 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "selu"),    ew_unary_params_f32, templates::ew_unary,   Live;
    "Celu",           Ai,     12 ..= OPSET_ANY,   F32,      kernel!(EwUnary, "celu"),    ew_unary_params_f32, templates::ew_unary,   Live;
    "ThresholdedRelu", Ai,    10 ..= OPSET_ANY,   FLOAT,    kernel!(EwUnary, "trelu"),   ew_unary_params_f32, templates::ew_unary,   Live;
    "Shrink",         Ai,     9 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "shrink"),  ew_unary_params_f32, templates::ew_unary,   Live;
    "Gelu",           Ai,     20 ..= OPSET_ANY,   FLOAT,    kernel!(EwUnary, "gelu"),    gelu,               templates::ew_unary,    Live;

    // `Swish` is new at opset 24 and is what `onnxruntime/mobius` emits for the SwiGLU gate of
    // every LLM MLP. `Swish(x) = x * sigmoid(alpha * x)`; `alpha` now rides the parameter tail,
    // so the general op is claimed rather than only SiLU. Window closed at 24 because that is
    // the only schema version that exists.
    "Swish",          Ai,     OPSET_STD_SWISH ..= OPSET_STD_SWISH, FLOAT, kernel!(EwUnary, "swish"), ew_unary_params_f32, templates::ew_unary, Staged(UNEXERCISED);

    // ---------------------------------------------------------------------------------------
    // Variadic elementwise — composed from the binary template, never an N-input shader.
    // ---------------------------------------------------------------------------------------
    "Sum",            Ai,     8 ..= OPSET_ANY,    NUMERIC,  kernel!(EwBinary, "add"),    claim::ew_variadic, templates::ew_variadic, Ready;
    "Mean",           Ai,     8 ..= OPSET_ANY,    FLOAT,    kernel!(EwBinary, "mean"),   claim::ew_variadic, templates::ew_variadic, Ready;
    "Max",            Ai,     8 ..= OPSET_ANY,    NUMERIC,  kernel!(EwBinary, "max"),    claim::ew_variadic, templates::ew_variadic, Ready;
    "Min",            Ai,     8 ..= OPSET_ANY,    NUMERIC,  kernel!(EwBinary, "min"),    claim::ew_variadic, templates::ew_variadic, Ready;

    // ---------------------------------------------------------------------------------------
    // Selection and type conversion.
    // ---------------------------------------------------------------------------------------
    "Where",          Ai,     9 ..= OPSET_ANY,    ANY,      kernel!(EwSelect, "where"),  claim::ew_select,   templates::ew_select,   Ready;
    "Clip",           Ai,     11 ..= OPSET_ANY,   NUMERIC,  kernel!(EwSelect, "clip"),   clip_f32,           templates::ew_clip,     Live;
    "Cast",           Ai,     6 ..= OPSET_ANY,    ANY,      kernel!(None),               claim::cast,        templates::unimplemented, Staged(NEEDS_CAST_MATRIX);
    "CastLike",       Ai,     15 ..= OPSET_ANY,   ANY,      kernel!(None),               claim::never,       templates::unimplemented, Staged(NEEDS_CAST_MATRIX);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{Domain, OpStatus};

    /// Every [`Live`] row in this module must be justified by [`EXERCISED`] or [`TEMPLATE_LIVE`].
    ///
    /// This is the structural half of the "no unexecuted claims" rule. It cannot check that the
    /// named test actually ran — nothing in a unit test can — but it makes going live a two-place
    /// edit where the second place demands either a device and a test name, or a named
    /// representative that has one. The cost of flipping a row on a hunch is that you have to
    /// write a sentence claiming evidence that does not exist.
    ///
    /// The evidence lists are keyed by `(op, dtype)`, so one op legitimately appears once per
    /// dtype it has been exercised at. The set of *ops* is what has to match the set of live rows;
    /// which dtypes each one carries is [`only_proved_dtypes`]'s business, not this test's.
    #[test]
    fn every_live_row_is_justified_by_one_of_the_two_evidence_lists() {
        let mut live: Vec<&str> = OPS
            .iter()
            .filter(|s| s.status == OpStatus::Live)
            .map(|s| s.op_type)
            .collect();
        let mut justified: Vec<&str> = EXERCISED
            .iter()
            .map(|(op, _)| *op)
            .chain(TEMPLATE_LIVE.iter().map(|(op, _)| *op))
            .collect();
        live.sort_unstable();
        justified.sort_unstable();
        justified.dedup();
        assert_eq!(
            live, justified,
            "a Live row has no entry in EXERCISED or TEMPLATE_LIVE, or an entry has no Live row"
        );
    }

    /// No live claim may rest on a variant no device can create.
    ///
    /// The artifact-level guard in [`variants`](crate::ops::common::variants) allows a wider
    /// capability set than this one, because generating an unloadable variant is harmless. This is
    /// the rule that actually matters: a `(op, dtype)` pair in [`EXERCISED`] is a promise that the
    /// pair *ran on a device*, so if its module declares a capability the engine never enables,
    /// either the promise is false or the evidence was recorded against something else.
    ///
    /// Live today: the `_i64` variants declare `Int64`, which needs
    /// `VkPhysicalDeviceFeatures::shaderInt64`; `vk::device` passes no `pEnabledFeatures` at all.
    /// Nothing claims i64, so this passes — and the moment somebody adds `("Sub", "i64")` on the
    /// strength of the variant existing, it fails here rather than as a device-lost on a user's
    /// machine. That is the same trap the f16 rows sat in for their whole existence, pre-sprung.
    #[test]
    fn no_live_claim_rests_on_an_unloadable_variant() {
        use crate::ops::common::variants::{ENGINE_ENABLED_CAPABILITIES, declared_capabilities};

        let modules: std::collections::HashMap<&str, &[u8]> =
            crate::engine::shaders::SHADER_MODULES
                .iter()
                .copied()
                .collect();

        let mut offenders: Vec<String> = Vec::new();
        for spec in OPS.iter().filter(|s| s.status == OpStatus::Live) {
            for d in crate::ops::common::dtype::ALL_DTYPES {
                if !proved_at(spec.op_type, d) {
                    continue;
                }
                let Some(stem) = spec.kernel.stem(d) else {
                    continue;
                };
                let Some(bytes) = modules.get(stem) else {
                    continue;
                };
                for cap in declared_capabilities(bytes) {
                    if !ENGINE_ENABLED_CAPABILITIES.contains(&cap) {
                        offenders.push(format!(
                            "`{}` is claimed at {} via `{stem}`, which declares SPIR-V capability \
                             {cap} — the engine enables no such feature, so that module cannot be \
                             created on any device",
                            spec.op_type,
                            dtype_suffix(d),
                        ));
                    }
                }
            }
        }
        assert!(
            offenders.is_empty(),
            "a live claim rests on a variant no device can create:\n  {}",
            offenders.join("\n  ")
        );
    }

    /// A template-live row is only as good as its representative, so the representative must hold.    ///
    /// Without this the weaker list could quietly outlive the stronger one: if `Add` were ever
    /// demoted — because the differential suite disproved the wire, which is exactly what flipping
    /// it is meant to allow — thirty-four rows would still be claiming on evidence that had been
    /// withdrawn. This turns that into a compile-time-adjacent failure instead of a silent one.
    #[test]
    fn every_template_live_row_stands_on_a_representative_that_is_itself_exercised_and_live() {
        for (op, representative) in TEMPLATE_LIVE {
            assert!(
                EXERCISED.iter().any(|(e, _)| e == representative),
                "{op} stands on {representative}, which is not in EXERCISED"
            );
            let rep = OPS
                .iter()
                .find(|s| s.op_type == *representative)
                .unwrap_or_else(|| panic!("{representative} has no row"));
            assert_eq!(
                rep.status,
                OpStatus::Live,
                "{op} stands on {representative}, which is not Live"
            );
        }
    }

    /// Every live row is live for f32 only, and none of them claims on `caps` alone.
    ///
    /// The narrowing is the whole point. `Add`'s `caps` is `NUMERIC` so the variant table still
    /// compiles `add_{f32,f16,i32,i64}` — but only `add_f32` has been dispatched, so only f32 is
    /// claimed. Using the bare `claim::ew_{unary,binary}` on a live row would claim every variant
    /// its `caps` allows on the strength of one that ran.
    #[test]
    fn live_rows_claim_f32_only_and_never_on_caps_alone() {
        let narrowed_predicates: &[crate::registry::ClaimPredicate] = &[
            ew_binary_f32,
            ew_unary_f32,
            ew_unary_params_f32,
            gelu,
            clip_f32,
        ];
        for spec in OPS.iter().filter(|s| s.status == OpStatus::Live) {
            let narrowed = narrowed_predicates
                .iter()
                .any(|p| std::ptr::fn_addr_eq(spec.claim, *p));
            assert!(
                narrowed,
                "{} is Live with a predicate that is not dtype-narrowed; it would claim every \
                 variant its caps allow on the strength of add_f32 having executed",
                spec.op_type
            );
        }
        assert!(
            EXERCISED.contains(&("Add", "f32")),
            "Add/f32 is the representative the whole family stood on"
        );
    }

    /// The families that are *not* a one-line body change each carry their own evidence.
    ///
    /// Deliberate, and worth a test rather than a comment. `Equal`/`Greater`/… write a bool from a
    /// float input, which is a different store path in the template, not a different expression.
    /// `Sum`/`Mean`/`Max`/`Min` issue several dispatches through `ew_variadic`. `Where` is a third
    /// template. `And`/`Or`/`BitwiseAnd` are not f32 at all, so the narrowing that justifies the
    /// live rows says nothing about them. Each is a distinct bet.
    ///
    /// 2026-08-02: each of them was taken through a proof run of its own, so the assertion moved
    /// from *status* to *evidence*. `Staged` was only ever a stand-in for "nothing has measured
    /// this"; now that a ledger exists, asserting the stand-in would let a row be flipped to
    /// `Ready` and pass while nothing had measured it — and would fail after a genuine proof,
    /// which is the wrong way round. The invariant that survives is that none of these families
    /// rides `add_f32`'s evidence: each names itself in a ledger key.
    #[test]
    fn families_that_are_not_a_one_line_body_change_carry_their_own_evidence() {
        let ledger = crate::registry::ledger();
        assert!(
            ledger.faults.is_empty(),
            "ERROR(instrument): the ledger is faulted, so this test can read nothing: {:?}",
            ledger.faults
        );
        for op in [
            "Equal",
            "Greater",
            "Less",
            "And",
            "Or",
            "Xor",
            "BitwiseAnd",
            "Not",
            "IsNaN",
            "Sum",
            "Mean",
            "Max",
            "Min",
            "Where",
            "PRelu",
        ] {
            let row = OPS.iter().find(|s| s.op_type == op).unwrap();
            assert_eq!(
                row.status,
                OpStatus::Ready,
                "{op} has been proven, so its row should be Ready and let the ledger decide"
            );
            let needle = format!("::{op}/");
            assert!(
                ledger.entries().iter().any(|e| e.key.0.contains(&needle)),
                "{op} is Ready with no ledger entry naming it; it would be claiming on the \
                 strength of another op's evidence"
            );
        }
    }

    /// `Swish` is the one row of the 2026-08-02 batch that stayed staged, and it is a finding.
    ///
    /// Its shader `ew_unary_swish_f32.spv` exists and compiles, but the row declines
    /// `[dtype] Swish is live for no dtype` — from [`EXERCISED`], a **second, hand-written**
    /// evidence list that gates independently of the proof ledger. So no proof run can reach it:
    /// the generator offers the key, the claim fails before the ledger is consulted, and the
    /// case reports no key at all. Adding `("Swish", "f32")` to `EXERCISED` by hand would make
    /// that list say something no run has shown, which is the thing the ledger exists to stop.
    ///
    /// It is not a coverage hole in practice — ORT decomposes opset-24 `Swish` into
    /// `Sigmoid` + `Mul`, both of which we claim and both of which are proven.
    #[test]
    fn swish_stays_staged_because_a_second_evidence_list_gates_it() {
        let row = OPS.iter().find(|s| s.op_type == "Swish").unwrap();
        assert_eq!(
            row.status,
            OpStatus::Staged(UNEXERCISED),
            "Swish was reverted to Staged deliberately: EXERCISED vetoes it before the ledger \
             is consulted, so it cannot be proven without changing that list by hand"
        );
        assert!(
            !EXERCISED.contains(&("Swish", "f32")),
            "EXERCISED now names Swish/f32. If a run proved it, say which one here; if it was \
             added by hand, the list is asserting something nothing measured"
        );
    }

    /// `Swish` is new at opset 24 and is the SwiGLU activation `onnxruntime/mobius` emits.
    ///
    /// It is registered at exactly its one schema version, and the predicate claims only
    /// `alpha = 1` — the shader is `x * sigmoid(x)`, and a shader that ignores `alpha` while
    /// claiming the node is the permissive failure this design forbids.
    #[test]
    fn swish_is_registered_at_exactly_24_with_alpha_pinned() {
        let row = OPS
            .iter()
            .find(|s| s.op_type == "Swish")
            .expect("Swish row");
        assert_eq!(row.domain, Domain::Ai);
        assert_eq!((row.min_opset, row.max_opset), (24, 24));
        assert_eq!(row.kernel.op, "swish");
        assert!(
            !std::ptr::fn_addr_eq(
                row.claim,
                claim::ew_unary as crate::registry::ClaimPredicate
            ),
            "Swish must not use the bare unary predicate; that would claim any alpha"
        );
    }

    #[test]
    fn the_table_is_the_size_op_coverage_promises_for_tier_1() {
        // §4's tier-1 elementwise inventory. If this shrinks, coverage regressed; if it grows,
        // update OP_COVERAGE.md §4 in the same change.
        assert!(
            OPS.len() >= 65,
            "tier-1 elementwise table has only {} rows",
            OPS.len()
        );
    }

    #[test]
    fn every_row_has_a_status_that_matches_its_shader_coverage() {
        use crate::engine::shaders;
        for s in OPS {
            match s.status {
                OpStatus::Staged(reason) => {
                    assert!(!reason.is_empty(), "{} is staged with no reason", s.op_type)
                }
                OpStatus::Live | OpStatus::Ready => {
                    // A live/ready row promises its shader compiles and has been executed on a device.
                    // Verify at minimum that the shader variant exists in the binary.
                    let any_shader = s.caps.iter().any(|d| {
                        s.kernel
                            .stem(d)
                            .is_some_and(|stem| shaders::find(stem).is_some())
                    });
                    assert!(
                        any_shader,
                        "{} is live but has no compiled shader variant in the binary",
                        s.op_type
                    );
                }
            }
        }
    }

    #[test]
    fn every_row_is_in_the_default_domain() {
        // OQ-11 is unratified: contrib rows are not ours to add yet.
        assert!(OPS.iter().all(|s| s.domain == Domain::Ai));
    }

    #[test]
    fn shader_rows_have_an_arity_their_predicate_agrees_with() {
        use crate::ops::common::variants::Template;
        for s in OPS {
            let arity = s.kernel.template.input_arity();
            match s.kernel.template {
                Template::None => assert_eq!(arity, 0),
                Template::EwUnary => assert_eq!(arity, 1),
                Template::EwBinary => assert_eq!(arity, 2),
                Template::EwSelect => assert_eq!(arity, 3),
                // QGemv rows live in quant.rs, not in this table; arity checked there.
                Template::QGemv => {}
            }
        }
    }

    #[test]
    fn variant_stems_are_lowercase_and_underscored() {
        for s in OPS {
            for d in s.caps.iter() {
                if let Some(stem) = s.kernel.stem(d) {
                    assert!(
                        stem.chars()
                            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_'),
                        "`{stem}` is not a clean file stem"
                    );
                }
            }
        }
    }

    /// The blocker `NEEDS_PARAMS` still means what it says, now that the *unary* half of it has
    /// been retired by the push-constant tail.
    ///
    /// The tail carries floats. `Mod`'s `fmod`, `BitShift`'s `direction` and `IsInf`'s two detect
    /// flags are not floats — they select a different expression, so each needs a shader variant
    /// rather than a value, and they keep the blocker. Recording that here stops the next reader
    /// from concluding that because `LeakyRelu` moved, these were simply overlooked.
    #[test]
    fn parameterised_ops_are_staged_behind_their_own_reason() {
        for op in ["Mod", "BitShift", "IsInf"] {
            let s = OPS.iter().find(|s| s.op_type == op).unwrap();
            assert_eq!(
                s.status,
                OpStatus::Staged(NEEDS_PARAMS),
                "{op}'s attribute is a selector, not a float; it cannot ride the parameter tail"
            );
        }
        for op in ["Cast", "CastLike"] {
            let s = OPS.iter().find(|s| s.op_type == op).unwrap();
            assert_eq!(s.status, OpStatus::Staged(NEEDS_CAST_MATRIX), "{op}");
        }
    }

    /// Every op with a slot-table entry must use a predicate that actually reads that table, and
    /// every op whose predicate reads it must have an entry.
    ///
    /// This is the failure the whole mechanism is exposed to: a row wired to
    /// `templates::ew_unary` with a plain predicate would be claimed, dispatched with a zeroed
    /// tail, and answer `LeakyRelu(alpha=0.2)` with `0.0 * x` for the negative half — a wrong
    /// answer, silently, on a graph we said we could run.
    #[test]
    fn every_slot_table_op_claims_through_the_parameter_predicate() {
        use crate::ops::common::params;

        for (op, _) in params::SLOTS {
            let s = OPS
                .iter()
                .find(|s| s.op_type == *op)
                .unwrap_or_else(|| panic!("`{op}` is in the slot table but not in the op table"));
            assert!(
                std::ptr::fn_addr_eq(
                    s.claim,
                    ew_unary_params_f32 as crate::registry::ClaimPredicate
                ),
                "`{op}` has parameter slots but claims through a predicate that never resolves \
                 them; it would dispatch with a zeroed tail"
            );
        }

        for s in OPS.iter().filter(|s| {
            std::ptr::fn_addr_eq(
                s.claim,
                ew_unary_params_f32 as crate::registry::ClaimPredicate,
            )
        }) {
            assert!(
                params::slots_for(s.op_type).is_some(),
                "`{}` claims through the parameter predicate but has no slots; it is either a \
                 plain unary op or its slots were forgotten",
                s.op_type
            );
        }
    }

    /// `Gelu`'s attribute is a string that selects an expression, so it does not ride the tail;
    /// the row must decline `approximate="tanh"` rather than answer it with the erf form.
    #[test]
    fn gelu_claims_only_the_form_its_shader_implements() {
        let s = OPS.iter().find(|s| s.op_type == "Gelu").unwrap();
        assert_eq!(s.status, OpStatus::Live);
        assert!(
            crate::ops::common::params::slots_for("Gelu").is_none(),
            "Gelu's `approximate` is a string; putting it in the float tail would be a category \
             error, not a shortcut"
        );
        assert!(std::ptr::fn_addr_eq(
            s.claim,
            gelu as crate::registry::ClaimPredicate
        ));
    }

    /// `Clip` rides the ternary template, not the parameter tail, because its bounds are optional
    /// *inputs*. The row must therefore use the ternary translate — using `ew_select`'s would
    /// take the dtype from input 1 (`Where`'s convention) and using `ew_unary`'s would bind one
    /// buffer where three are needed.
    #[test]
    fn clip_uses_the_ternary_template_and_not_the_parameter_tail() {
        let s = OPS.iter().find(|s| s.op_type == "Clip").unwrap();
        assert_eq!(s.status, OpStatus::Live);
        assert!(crate::ops::common::params::slots_for("Clip").is_none());
        assert!(std::ptr::fn_addr_eq(
            s.translate,
            templates::ew_clip as crate::registry::TranslateHandler
        ));
    }

    #[test]
    fn known_llm_hot_ops_are_present() {
        // OP_COVERAGE.md §3: these are what a Qwen decoder block is made of outside attention.
        for op in ["Add", "Mul", "Sigmoid", "Sqrt", "Where", "Cast", "Pow"] {
            assert!(
                OPS.iter().any(|s| s.op_type == op),
                "`{op}` is missing from the tier-1 table"
            );
        }
    }

    #[test]
    fn boolean_ops_do_not_claim_floats() {
        for op in ["And", "Or", "Xor", "Not"] {
            let s = OPS.iter().find(|s| s.op_type == op).unwrap();
            assert!(!s.caps.contains(crate::engine::DType::F32), "{op}");
            assert!(s.caps.contains(crate::engine::DType::Bool), "{op}");
        }
    }
}
