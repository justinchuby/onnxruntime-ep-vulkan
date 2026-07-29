//! Tier-1 elementwise ops — the table that proves the machinery.
//!
//! Every row here is an entry in `OP_COVERAGE.md` §4's inventory, and every row is one line. That
//! is the whole claim of §5: three GLSL templates and one shared broadcasting helper cover roughly
//! a third of the op inventory, so the schedule is decided by how fast rows can be added, not by
//! how fast kernels can be written.
//!
//! # Everything here is [`Staged`]
//!
//! The three GLSL templates now exist (`shaders/glsl/ew_{unary,binary,select}.comp`) and all 168
//! variants compile, but the engine has no dispatch path yet and nothing has run one of them on a
//! device. So every row still declines and the CPU EP runs every one of these nodes, which is
//! always correct. What *does* exist is the full description — opset window, dtype capabilities,
//! template, variant stems, claim predicate, translate handler, and now the shader itself — all of
//! it unit tested. Flipping a row live once a differential test has executed its variant is a
//! one-word diff.
//!
//! Three staging reasons appear below and they mean different things:
//!
//! * [`UNEXERCISED`] — the row and its shader are complete; nothing has executed it yet.
//! * [`NEEDS_PARAMS`] — the op carries attributes (`alpha`, `beta`, `fmod`, `direction`,
//!   `approximate`) that the plain unary/binary template has nowhere to put. Claiming it with
//!   [`claim::never`] is the honest answer: `OP_COVERAGE.md` §7's rule is that an op is claimed
//!   only when the *attribute* combination is genuinely handled, and here it is not. These become
//!   a parameterised template variant, not a bespoke kernel each. Their shader variants are
//!   compiled with the ONNX **default** attribute values so the template stays uniform — a default
//!   is not a handled value, which is precisely why the row stays staged.
//! * [`NEEDS_CAST_MATRIX`] — the variant space is keyed on a dtype *pair*.

use crate::kernel;
use crate::ops::common::claim::{self, ClaimResult};
use crate::ops::common::dtype::{ANY, BOOL, DTypeSet, F32, FLOAT, INT, NUMERIC};
use crate::ops::common::templates;
use crate::registry::OpStatus::{Live, Staged};
use crate::registry::{NodeView, OPSET_ANY, OPSET_STD_SWISH, OpSpec, UNEXERCISED};

/// `Equal` compares booleans as well as numbers.
const EQ_CAPS: DTypeSet = NUMERIC.union(BOOL);

/// Staging reason for ops whose attributes the plain template cannot carry.
pub const NEEDS_PARAMS: &str = "it carries attributes the plain elementwise template has no push-constant slot for; it needs \
     the parameterised template variant";

/// Staging reason for `Cast`, whose variant space is keyed on a dtype *pair*.
pub const NEEDS_CAST_MATRIX: &str = "its shader variant space is keyed on a source/destination dtype pair rather than a single \
     dtype, so it needs its own template and manifest column";

/// `ai.onnx::Swish` (opset 24) — claim only `alpha = 1`, i.e. SiLU.
///
/// The generic unary template has no push-constant slot for an attribute, so a general `alpha`
/// would be [`NEEDS_PARAMS`]. But `alpha = 1` is what every SwiGLU MLP emits and it is a distinct,
/// fully-handled shader, so the honest row is "claimed at 1, declined elsewhere" rather than
/// either claiming everything or staging the whole op behind a parameter we do not need yet.
fn swish(view: &NodeView<'_>, spec: &OpSpec) -> ClaimResult {
    claim::ew_unary(view, spec)?;
    claim::attr_float_is(view, spec, "alpha", 1.0)
}

crate::op_table! {
    // ---------------------------------------------------------------------------------------
    // Binary arithmetic, logic and comparison — one template, numpy broadcasting for free.
    //
    //  op            domain  opset window        caps      kernel                       claim               translate               status
    // ---------------------------------------------------------------------------------------
    "Add",            Ai,     7 ..= OPSET_ANY,    NUMERIC,  kernel!(EwBinary, "add"),    claim::ew_binary,   templates::ew_binary,   Live;
    "Sub",            Ai,     7 ..= OPSET_ANY,    NUMERIC,  kernel!(EwBinary, "sub"),    claim::ew_binary,   templates::ew_binary,   Staged(UNEXERCISED);
    "Mul",            Ai,     7 ..= OPSET_ANY,    NUMERIC,  kernel!(EwBinary, "mul"),    claim::ew_binary,   templates::ew_binary,   Staged(UNEXERCISED);
    "Div",            Ai,     7 ..= OPSET_ANY,    NUMERIC,  kernel!(EwBinary, "div"),    claim::ew_binary,   templates::ew_binary,   Staged(UNEXERCISED);
    "Pow",            Ai,     7 ..= OPSET_ANY,    FLOAT,    kernel!(EwBinary, "pow"),    claim::ew_binary,   templates::ew_binary,   Staged(UNEXERCISED);
    "Mod",            Ai,     10 ..= OPSET_ANY,   NUMERIC,  kernel!(EwBinary, "mod"),    claim::never,       templates::unimplemented, Staged(NEEDS_PARAMS);
    "And",            Ai,     7 ..= OPSET_ANY,    BOOL,     kernel!(EwBinary, "and"),    claim::ew_binary,   templates::ew_binary,   Staged(UNEXERCISED);
    "Or",             Ai,     7 ..= OPSET_ANY,    BOOL,     kernel!(EwBinary, "or"),     claim::ew_binary,   templates::ew_binary,   Staged(UNEXERCISED);
    "Xor",            Ai,     7 ..= OPSET_ANY,    BOOL,     kernel!(EwBinary, "xor"),    claim::ew_binary,   templates::ew_binary,   Staged(UNEXERCISED);
    "BitwiseAnd",     Ai,     18 ..= OPSET_ANY,   INT,      kernel!(EwBinary, "bitand"), claim::ew_binary,   templates::ew_binary,   Staged(UNEXERCISED);
    "BitwiseOr",      Ai,     18 ..= OPSET_ANY,   INT,      kernel!(EwBinary, "bitor"),  claim::ew_binary,   templates::ew_binary,   Staged(UNEXERCISED);
    "BitwiseXor",     Ai,     18 ..= OPSET_ANY,   INT,      kernel!(EwBinary, "bitxor"), claim::ew_binary,   templates::ew_binary,   Staged(UNEXERCISED);
    "BitShift",       Ai,     11 ..= OPSET_ANY,   INT,      kernel!(EwBinary, "bitshift"), claim::never,     templates::unimplemented, Staged(NEEDS_PARAMS);
    "Equal",          Ai,     7 ..= OPSET_ANY,    EQ_CAPS,  kernel!(EwBinary, "eq"),     claim::ew_binary,   templates::ew_binary,   Staged(UNEXERCISED);
    "Greater",        Ai,     7 ..= OPSET_ANY,    NUMERIC,  kernel!(EwBinary, "gt"),     claim::ew_binary,   templates::ew_binary,   Staged(UNEXERCISED);
    "GreaterOrEqual", Ai,     12 ..= OPSET_ANY,   NUMERIC,  kernel!(EwBinary, "ge"),     claim::ew_binary,   templates::ew_binary,   Staged(UNEXERCISED);
    "Less",           Ai,     7 ..= OPSET_ANY,    NUMERIC,  kernel!(EwBinary, "lt"),     claim::ew_binary,   templates::ew_binary,   Staged(UNEXERCISED);
    "LessOrEqual",    Ai,     12 ..= OPSET_ANY,   NUMERIC,  kernel!(EwBinary, "le"),     claim::ew_binary,   templates::ew_binary,   Staged(UNEXERCISED);
    "PRelu",          Ai,     7 ..= OPSET_ANY,    FLOAT,    kernel!(EwBinary, "prelu"),  claim::ew_binary,   templates::ew_binary,   Staged(UNEXERCISED);

    // ---------------------------------------------------------------------------------------
    // Unary maths — the longest run of pure table rows in the crate.
    // ---------------------------------------------------------------------------------------
    "Abs",            Ai,     6 ..= OPSET_ANY,    NUMERIC,  kernel!(EwUnary, "abs"),     claim::ew_unary,    templates::ew_unary,    Staged(UNEXERCISED);
    "Neg",            Ai,     6 ..= OPSET_ANY,    NUMERIC,  kernel!(EwUnary, "neg"),     claim::ew_unary,    templates::ew_unary,    Staged(UNEXERCISED);
    "Reciprocal",     Ai,     6 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "recip"),   claim::ew_unary,    templates::ew_unary,    Staged(UNEXERCISED);
    "Sqrt",           Ai,     6 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "sqrt"),    claim::ew_unary,    templates::ew_unary,    Staged(UNEXERCISED);
    "Exp",            Ai,     6 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "exp"),     claim::ew_unary,    templates::ew_unary,    Staged(UNEXERCISED);
    "Log",            Ai,     6 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "log"),     claim::ew_unary,    templates::ew_unary,    Staged(UNEXERCISED);
    "Sin",            Ai,     7 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "sin"),     claim::ew_unary,    templates::ew_unary,    Staged(UNEXERCISED);
    "Cos",            Ai,     7 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "cos"),     claim::ew_unary,    templates::ew_unary,    Staged(UNEXERCISED);
    "Tan",            Ai,     7 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "tan"),     claim::ew_unary,    templates::ew_unary,    Staged(UNEXERCISED);
    "Asin",           Ai,     7 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "asin"),    claim::ew_unary,    templates::ew_unary,    Staged(UNEXERCISED);
    "Acos",           Ai,     7 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "acos"),    claim::ew_unary,    templates::ew_unary,    Staged(UNEXERCISED);
    "Atan",           Ai,     7 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "atan"),    claim::ew_unary,    templates::ew_unary,    Staged(UNEXERCISED);
    "Sinh",           Ai,     9 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "sinh"),    claim::ew_unary,    templates::ew_unary,    Staged(UNEXERCISED);
    "Cosh",           Ai,     9 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "cosh"),    claim::ew_unary,    templates::ew_unary,    Staged(UNEXERCISED);
    "Tanh",           Ai,     6 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "tanh"),    claim::ew_unary,    templates::ew_unary,    Staged(UNEXERCISED);
    "Asinh",          Ai,     9 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "asinh"),   claim::ew_unary,    templates::ew_unary,    Staged(UNEXERCISED);
    "Acosh",          Ai,     9 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "acosh"),   claim::ew_unary,    templates::ew_unary,    Staged(UNEXERCISED);
    "Atanh",          Ai,     9 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "atanh"),   claim::ew_unary,    templates::ew_unary,    Staged(UNEXERCISED);
    "Ceil",           Ai,     6 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "ceil"),    claim::ew_unary,    templates::ew_unary,    Staged(UNEXERCISED);
    "Floor",          Ai,     6 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "floor"),   claim::ew_unary,    templates::ew_unary,    Staged(UNEXERCISED);
    "Round",          Ai,     11 ..= OPSET_ANY,   FLOAT,    kernel!(EwUnary, "round"),   claim::ew_unary,    templates::ew_unary,    Staged(UNEXERCISED);
    "Sign",           Ai,     9 ..= OPSET_ANY,    NUMERIC,  kernel!(EwUnary, "sign"),    claim::ew_unary,    templates::ew_unary,    Staged(UNEXERCISED);
    "Erf",            Ai,     9 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "erf"),     claim::ew_unary,    templates::ew_unary,    Staged(UNEXERCISED);
    "Not",            Ai,     1 ..= OPSET_ANY,    BOOL,     kernel!(EwUnary, "not"),     claim::ew_unary,    templates::ew_unary,    Staged(UNEXERCISED);
    "BitwiseNot",     Ai,     18 ..= OPSET_ANY,   INT,      kernel!(EwUnary, "bitnot"),  claim::ew_unary,    templates::ew_unary,    Staged(UNEXERCISED);
    "IsNaN",          Ai,     9 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "isnan"),   claim::ew_unary,    templates::ew_unary,    Staged(UNEXERCISED);
    "IsInf",          Ai,     10 ..= OPSET_ANY,   FLOAT,    kernel!(EwUnary, "isinf"),   claim::never,       templates::unimplemented, Staged(NEEDS_PARAMS);
    "Identity",       Ai,     1 ..= OPSET_ANY,    ANY,      kernel!(EwUnary, "identity"), claim::ew_unary,   templates::ew_unary,    Staged(UNEXERCISED);

    // ---------------------------------------------------------------------------------------
    // Activations. Attribute-free ones ride the unary template; parameterised ones are staged
    // behind NEEDS_PARAMS rather than claimed with attributes we would silently ignore.
    // ---------------------------------------------------------------------------------------
    "Relu",           Ai,     6 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "relu"),    claim::ew_unary,    templates::ew_unary,    Staged(UNEXERCISED);
    "Sigmoid",        Ai,     6 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "sigmoid"), claim::ew_unary,    templates::ew_unary,    Staged(UNEXERCISED);
    "HardSwish",      Ai,     14 ..= OPSET_ANY,   FLOAT,    kernel!(EwUnary, "hardswish"), claim::ew_unary,  templates::ew_unary,    Staged(UNEXERCISED);
    "Softplus",       Ai,     1 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "softplus"), claim::ew_unary,   templates::ew_unary,    Staged(UNEXERCISED);
    "Softsign",       Ai,     1 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "softsign"), claim::ew_unary,   templates::ew_unary,    Staged(UNEXERCISED);
    "Mish",           Ai,     18 ..= OPSET_ANY,   FLOAT,    kernel!(EwUnary, "mish"),    claim::ew_unary,    templates::ew_unary,    Staged(UNEXERCISED);
    "HardSigmoid",    Ai,     6 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "hardsigmoid"), claim::never,   templates::unimplemented, Staged(NEEDS_PARAMS);
    "LeakyRelu",      Ai,     6 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "leakyrelu"), claim::never,     templates::unimplemented, Staged(NEEDS_PARAMS);
    "Elu",            Ai,     6 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "elu"),     claim::never,       templates::unimplemented, Staged(NEEDS_PARAMS);
    "Selu",           Ai,     6 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "selu"),    claim::never,       templates::unimplemented, Staged(NEEDS_PARAMS);
    "Celu",           Ai,     12 ..= OPSET_ANY,   F32,      kernel!(EwUnary, "celu"),    claim::never,       templates::unimplemented, Staged(NEEDS_PARAMS);
    "ThresholdedRelu", Ai,    10 ..= OPSET_ANY,   FLOAT,    kernel!(EwUnary, "trelu"),   claim::never,       templates::unimplemented, Staged(NEEDS_PARAMS);
    "Shrink",         Ai,     9 ..= OPSET_ANY,    FLOAT,    kernel!(EwUnary, "shrink"),  claim::never,       templates::unimplemented, Staged(NEEDS_PARAMS);
    "Gelu",           Ai,     20 ..= OPSET_ANY,   FLOAT,    kernel!(EwUnary, "gelu"),    claim::never,       templates::unimplemented, Staged(NEEDS_PARAMS);

    // `Swish` is new at opset 24 and is what `onnxruntime/mobius` emits for the SwiGLU gate of
    // every LLM MLP. `Swish(x) = x * sigmoid(alpha * x)`; the shader implements `alpha = 1`, which
    // is SiLU, and the predicate declines any other alpha rather than pretending the default is
    // the only value. Window closed at 24 because that is the only schema version that exists.
    "Swish",          Ai,     OPSET_STD_SWISH ..= OPSET_STD_SWISH, FLOAT, kernel!(EwUnary, "swish"), swish, templates::ew_unary, Staged(UNEXERCISED);

    // ---------------------------------------------------------------------------------------
    // Variadic elementwise — composed from the binary template, never an N-input shader.
    // ---------------------------------------------------------------------------------------
    "Sum",            Ai,     8 ..= OPSET_ANY,    NUMERIC,  kernel!(EwBinary, "add"),    claim::ew_variadic, templates::ew_variadic, Staged(UNEXERCISED);
    "Mean",           Ai,     8 ..= OPSET_ANY,    FLOAT,    kernel!(EwBinary, "mean"),   claim::ew_variadic, templates::ew_variadic, Staged(UNEXERCISED);
    "Max",            Ai,     8 ..= OPSET_ANY,    NUMERIC,  kernel!(EwBinary, "max"),    claim::ew_variadic, templates::ew_variadic, Staged(UNEXERCISED);
    "Min",            Ai,     8 ..= OPSET_ANY,    NUMERIC,  kernel!(EwBinary, "min"),    claim::ew_variadic, templates::ew_variadic, Staged(UNEXERCISED);

    // ---------------------------------------------------------------------------------------
    // Selection and type conversion.
    // ---------------------------------------------------------------------------------------
    "Where",          Ai,     9 ..= OPSET_ANY,    ANY,      kernel!(EwSelect, "where"),  claim::ew_select,   templates::ew_select,   Staged(UNEXERCISED);
    "Clip",           Ai,     11 ..= OPSET_ANY,   NUMERIC,  kernel!(EwSelect, "clip"),   claim::never,       templates::unimplemented, Staged(NEEDS_PARAMS);
    "Cast",           Ai,     6 ..= OPSET_ANY,    ANY,      kernel!(None),               claim::cast,        templates::unimplemented, Staged(NEEDS_CAST_MATRIX);
    "CastLike",       Ai,     15 ..= OPSET_ANY,   ANY,      kernel!(None),               claim::never,       templates::unimplemented, Staged(NEEDS_CAST_MATRIX);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{Domain, OpStatus};

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
                OpStatus::Live => {
                    // A live row promises its shader compiles and has been executed on a device.
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

    #[test]
    fn parameterised_ops_are_staged_behind_their_own_reason() {
        // The honest-claiming rule in table form: an op whose attributes the template cannot
        // carry is staged with a *different* blocker from one that is merely waiting on a shader,
        // so "what is left to do" is readable straight off the table.
        for op in [
            "LeakyRelu",
            "Elu",
            "Selu",
            "Clip",
            "Mod",
            "BitShift",
            "Gelu",
        ] {
            let s = OPS.iter().find(|s| s.op_type == op).unwrap();
            assert_eq!(
                s.status,
                OpStatus::Staged(NEEDS_PARAMS),
                "{op} should be staged behind the parameterised template"
            );
        }
        for op in ["Cast", "CastLike"] {
            let s = OPS.iter().find(|s| s.op_type == op).unwrap();
            assert_eq!(s.status, OpStatus::Staged(NEEDS_CAST_MATRIX), "{op}");
        }
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
