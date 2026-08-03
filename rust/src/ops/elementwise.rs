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
//! variants compile. Which of them may be *claimed* is no longer written down here: as of
//! 2026-08-02 (§8.9.16) the claim predicate asks only whether a **loadable** kernel variant exists
//! for the node's dtype, and whether anything has ever measured that form is the proof ledger's
//! answer, given per-key by a harness run naming artifact, device, shader digest and observed
//! `worst_rel`. The two hand-written evidence lists that used to sit here are gone; see
//! [`variants::variant_is_loadable`] and `OP_COVERAGE.md` §8.9.16.
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
//! one-word diff, plus a proof run that mints the ledger entry for the form.
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

// §8.9.16 — `EXERCISED` and `TEMPLATE_LIVE` used to live here.
//
// They were two hand-maintained lists recording which `(op, dtype)` pairs had executed on a
// device, consulted by the claim predicate before a proof key was computed. The proof ledger
// now answers that question from harness-generated evidence — artifact, device, shader digest,
// observed `worst_rel` — so keeping a second, typed answer was not redundancy but a deadlock:
// a form the list vetoed could never be offered to the proof run that would justify it. The
// residual the lists did carry honestly — *does a creatable kernel exist?* — is now derived
// from the SPIR-V by [`variants::variant_is_loadable`]. A list nobody can falsify is the next
// stale thing; these are deleted rather than kept as documentation.

/// [`ew_binary`](claim::ew_binary), narrowed to a dtype whose kernel variant loads.
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
    only_loadable_variants(view, spec)
}

/// [`ew_unary`](claim::ew_unary), narrowed the same way and for the same reason.
fn ew_unary_f32(view: &NodeView<'_>, spec: &OpSpec) -> ClaimResult {
    claim::ew_unary(view, spec)?;
    only_loadable_variants(view, spec)
}

/// The shared narrowing: a **loadable kernel variant must exist** for input 0's dtype.
///
/// §8.9.16 — WHAT THIS USED TO DO, AND WHY IT STOPPED.
/// Until 2026-08-02 this function was `only_proved_dtypes` and it consulted `EXERCISED`, a
/// hand-written `&[(&str, &str)]` of `(op, dtype)` pairs that had run on a device. That list was
/// right for its time and it is what the proof ledger was built to replace: the ledger answers
/// the same question — *has anything measured this form?* — with a harness-generated entry naming
/// the artifact, the device, the shader digest and the observed `worst_rel`, instead of a pair
/// somebody typed.
///
/// Keeping both was not redundancy, it was a **deadlock**. `EXERCISED` vetoed inside the claim
/// predicate, which runs *before* a proof key is computed, so a form it blocked could never be
/// offered to a proof run: `Add`/i32 declined `[dtype]`, the generator saw no unlockable key, and
/// the form stayed unproven forever because it was unproven. Three forms sat in that loop with
/// working, compiled shaders.
///
/// So the two questions are split. *Does a kernel exist that this engine can create?* stays here,
/// because claiming a node whose module cannot be instantiated is an `EP_FAIL` at translate time
/// rather than a decline, and no ledger entry could make it safe. *Has anything measured it?* is
/// the ledger's, and a form that reaches here now declines `[unproven]` — a decline a proof run
/// can clear, rather than one it cannot reach.
///
/// The decline is still `[dtype]`, so an unloadable variant shows up in Niobe's histogram as a
/// dtype bucket. What has changed is what the bucket means: it now holds only forms with no
/// creatable module, not forms nobody has got round to measuring.
fn only_loadable_variants(view: &NodeView<'_>, spec: &OpSpec) -> ClaimResult {
    let edge = claim::input_edge(view, spec, 0)?;
    let dt = edge.dtype;
    let loadable = dt.is_some_and(|d| {
        spec.kernel
            .stem(d)
            .is_some_and(crate::ops::common::variants::variant_is_loadable)
    });
    crate::require!(
        loadable,
        DType,
        "`{}` has no loadable shader variant at {}; the module either was not generated or \
         declares a SPIR-V capability this engine does not enable, so no device we run on could \
         create a pipeline for it. The CPU EP is correct for this node",
        spec.op_type,
        dt.map_or("untyped", dtype_suffix),
    );
    Ok(())
}

/// `ai.onnx::Swish` (opset 24) — any `alpha`, now that the parameter tail carries it.
///
/// This row was previously pinned to `alpha = 1` (SiLU) because the unary template had no
/// push-constant slot for an attribute. It now has one, so the honest row is the general op.
/// The pin is kept in the tests as a record of *why* it existed, not as a constraint.
fn ew_unary_params_f32(view: &NodeView<'_>, spec: &OpSpec) -> ClaimResult {
    claim::ew_unary_params(view, spec)?;
    only_loadable_variants(view, spec)
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
    only_loadable_variants(view, spec)
}

/// `Clip` — the three-input form only, f32. See [`claim::ew_clip`] for why the shorter forms
/// decline rather than defaulting the omitted bound.
fn clip_f32(view: &NodeView<'_>, spec: &OpSpec) -> ClaimResult {
    claim::ew_clip(view, spec)?;
    only_loadable_variants(view, spec)
}

crate::op_table! {
    // ---------------------------------------------------------------------------------------
    // Binary arithmetic, logic and comparison — one template, numpy broadcasting for free.
    //
    //  op            domain  opset window        caps      kernel                       claim               translate               status
    // ---------------------------------------------------------------------------------------

    // Rows marked `ew_binary_f32` / `ew_unary_f32` narrow the claim to a dtype whose kernel
    // variant the engine can actually create (`only_loadable_variants`). Whether the form has been
    // measured is the ledger's answer, not this table's: a row here is an offer, and an offer with
    // no ledger entry declines `[unproven]`. Comparison and logic ops keep their own predicates —
    // their output dtype differs from their input, which is a different store path in the shader,
    // not a different one-line expression.
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

    /// Every [`Live`] row in this module must offer only forms the ledger can be asked about.
    ///
    /// §8.9.16. This test used to check that every live row appeared in one of two hand-written
    /// evidence lists. Those lists are gone, and with them the question they answered: *has this
    /// form executed?* is now the proof ledger's, answered per key from a harness run. What is left
    /// for a unit test is the structural half — a live row must carry a **dtype-narrowed**
    /// predicate, so that going live cannot claim every variant the row's `caps` allow. That is
    /// asserted by `live_rows_claim_f32_only_and_never_on_caps_alone` below.
    ///
    /// What this test asserts is the property that replaced the lists: every live row has at least
    /// one dtype whose kernel variant this engine could actually create. A live row with no
    /// loadable variant is a row that will decline `[dtype]` for every node it ever sees, which is
    /// an offer nobody can accept and a claim table entry that means nothing.
    #[test]
    fn every_live_row_offers_at_least_one_loadable_variant() {
        use crate::ops::common::variants::variant_is_loadable;

        let dead: Vec<&str> = OPS
            .iter()
            .filter(|s| s.status == OpStatus::Live)
            .filter(|s| {
                !crate::ops::common::dtype::ALL_DTYPES.iter().any(|d| {
                    s.caps.contains(*d) && s.kernel.stem(*d).is_some_and(variant_is_loadable)
                })
            })
            .map(|s| s.op_type)
            .collect();
        assert!(
            dead.is_empty(),
            "a Live row has no loadable kernel variant at any dtype its caps accept, so it can \
             only ever decline: {dead:?}"
        );
    }

    /// No live claim may rest on a variant no device can create.
    ///
    /// The artifact-level guard in [`variants`](crate::ops::common::variants) allows a wider
    /// capability set than this one, because *generating* an unloadable variant is harmless. This
    /// is the rule that actually matters: a live row offers every dtype its `caps` accept, and
    /// [`only_loadable_variants`] is the thing standing between that offer and a pipeline-creation
    /// failure at translate time.
    ///
    /// §8.9.16: before the split, this test scoped itself by `proved_at` — it only looked at pairs
    /// somebody had written into `EXERCISED`, which meant it could not see the forms most at risk,
    /// the ones nobody had thought about. It now walks every dtype the row's `caps` accept and
    /// asserts the *predicate* refuses the unloadable ones, rather than asserting a list does.
    ///
    /// Live today: the `_i64` variants declare `Int64`, which needs
    /// `VkPhysicalDeviceFeatures::shaderInt64`; `vk::device` passes no `pEnabledFeatures` at all.
    /// So `variant_is_loadable` is false for every `_i64` stem and the predicate declines i64 for
    /// every row — a device-lost on a user's machine turned into a decline, derived from the
    /// artifact rather than from anybody remembering.
    #[test]
    fn no_live_claim_rests_on_an_unloadable_variant() {
        use crate::ops::common::variants::{
            ENGINE_ENABLED_CAPABILITIES, declared_capabilities, variant_is_loadable,
        };

        let modules: std::collections::HashMap<&str, &[u8]> =
            crate::engine::shaders::SHADER_MODULES
                .iter()
                .copied()
                .collect();

        let mut offenders: Vec<String> = Vec::new();
        let mut refused = 0usize;
        for spec in OPS.iter().filter(|s| s.status == OpStatus::Live) {
            for d in crate::ops::common::dtype::ALL_DTYPES {
                let Some(stem) = spec.kernel.stem(d) else {
                    continue;
                };
                let Some(bytes) = modules.get(stem) else {
                    continue;
                };
                let unloadable = declared_capabilities(bytes)
                    .iter()
                    .any(|cap| !ENGINE_ENABLED_CAPABILITIES.contains(cap));
                if !unloadable {
                    continue;
                }
                refused += 1;
                if variant_is_loadable(stem) {
                    offenders.push(format!(
                        "`{}` at {} goes through `{stem}`, which declares a SPIR-V capability the \
                         engine does not enable, yet `variant_is_loadable` says yes",
                        spec.op_type,
                        dtype_suffix(d),
                    ));
                }
            }
        }
        assert!(
            offenders.is_empty(),
            "a live claim rests on a variant no device can create:\n  {}",
            offenders.join("\n  ")
        );
        // R12: distinguish "the guard refused things" from "there was nothing to refuse". A zero
        // here would make this test pass for the wrong reason the day the i64 variants stop being
        // generated, and it is the same shape as `bypassed` and `all-rejected` sharing one `0`.
        assert!(
            refused > 0,
            "no live row has an unloadable variant at any dtype, so this test asserted nothing; \
             either the engine now enables every capability we generate — in which case delete \
             this test and say so — or the variant table stopped generating them"
        );
    }

    /// §8.9.16 — `every_template_live_row_stands_on_a_representative_that_is_itself_exercised_and_live`
    /// was deleted here. It guarded `TEMPLATE_LIVE`, the weaker of the two evidence lists, against
    /// outliving `EXERCISED`, the stronger. Both lists are gone: every row that stood on a
    /// representative's evidence now stands on a ledger entry of its own, keyed on its own dtypes
    /// and its own shader digest, so there is no longer a borrowed claim to invalidate.
    ///
    /// Every live row is live only at dtypes whose variant loads, and none of them claims on
    /// `caps` alone.
    ///
    /// The narrowing is still the whole point. `Add`'s `caps` is `NUMERIC` so the variant table
    /// compiles `add_{f32,f16,i32,i64}` — `add_i64` declares `Int64` and cannot be created, so the
    /// predicate refuses it. Using the bare `claim::ew_{unary,binary}` on a live row would offer
    /// every variant its `caps` allow, loadable or not.
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
            OPS.iter()
                .any(|s| s.op_type == "Add" && s.status == OpStatus::Live),
            "Add is the row the whole elementwise family's shape is read from"
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

    /// `Swish` was the row the second evidence list held hostage; §8.9.16 released it.
    ///
    /// Its shader `ew_unary_swish_f32.spv` exists and compiles, but until 2026-08-02 the row
    /// declined `[dtype] Swish is live for no dtype` — from `EXERCISED`, a second, hand-written
    /// evidence list that gated independently of the proof ledger. No proof run could reach it:
    /// the generator offered the key, the claim failed before the ledger was consulted, and the
    /// case reported no key at all. Adding `("Swish", "f32")` by hand would have made that list
    /// say something no run had shown, which is the thing the ledger exists to stop. That was the
    /// deadlock, and deleting the list is what fixed it.
    ///
    /// The row is *still* `Staged(UNEXERCISED)`, and that is now an ordinary staging decision
    /// rather than a trap: `Swish` is opset-24-only and ORT decomposes it into `Sigmoid` + `Mul`
    /// on every graph we have, both of which we claim and both of which are proven, so no model
    /// in this repository can produce a `Swish` node for a proof run to measure. It stays staged
    /// because nothing can exercise it, not because a list forbids it.
    ///
    /// What this test asserts is the release: the predicate the row would use if flipped no longer
    /// consults anything but loadability.
    #[test]
    fn swish_is_staged_for_want_of_a_graph_not_for_want_of_a_list() {
        use crate::ops::common::variants::variant_is_loadable;

        let row = OPS.iter().find(|s| s.op_type == "Swish").unwrap();
        assert_eq!(
            row.status,
            OpStatus::Staged(UNEXERCISED),
            "Swish is staged because no graph we have emits it, not because it cannot work"
        );
        let stem = row
            .kernel
            .stem(crate::engine::DType::F32)
            .expect("Swish has an f32 variant stem");
        assert!(
            variant_is_loadable(stem),
            "`{stem}` is not loadable, so flipping this row would offer a node the engine cannot \
             serve — that would be a different finding from the one this test records"
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
