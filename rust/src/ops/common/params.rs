//! Op attributes as push-constant parameters — the one table that says which attribute goes in
//! which slot.
//!
//! # Why this module exists
//!
//! A block of ops — `LeakyRelu`, `Elu`, `Selu`, `Celu`, `ThresholdedRelu`, `Shrink`,
//! `HardSigmoid`, `Swish` — differ from the plain unary template by exactly one or two floats.
//! Their shader bodies were written months before any of them could be claimed, compiled with the
//! ONNX *default* attribute values baked in, and staged behind `NEEDS_PARAMS` because
//! `OP_COVERAGE.md` §7.1 forbids claiming an op whose attributes we have not actually handled: a
//! default is not a handled value, it is a guess that happens to be right on the graphs that use
//! the default and silently wrong on every other.
//!
//! The fix is a shared mechanism rather than eight per-op ones, which is §5.1's argument applied
//! one level down: the host reads the attributes, writes them into the push-constant tail, and
//! the shader indexes `pc.params[i]`. Adding the next parameterised op is then a table row.
//!
//! # The slot contract
//!
//! Slot order **is** the contract with the GLSL in `shaders/glsl/templates/ew_unary.comp`. Both
//! sides index by position, so a reordering here is not a compile error on either side — it is a
//! wrong answer. [`SLOTS`] is therefore the single definition, and the shader comments name their
//! slots so a reader can check the two against each other:
//!
//! | op                | params[0] | params[1] |
//! |-------------------|-----------|-----------|
//! | `HardSigmoid`     | alpha     | beta      |
//! | `LeakyRelu`       | alpha     | —         |
//! | `Elu`             | alpha     | —         |
//! | `Selu`            | alpha     | gamma     |
//! | `Celu`            | alpha     | —         |
//! | `ThresholdedRelu` | alpha     | —         |
//! | `Shrink`          | lambd     | bias      |
//! | `Swish`           | alpha     | —         |
//!
//! # Defaults are still read from the node
//!
//! A missing attribute resolves to the ONNX default *here*, on the host, where the value is then
//! pushed explicitly. That is not the same as the old baked-in default: the shader is compiled
//! once for all values, so a graph that sets `alpha` gets `alpha`, and the claim is honest for
//! every value rather than for one.

use crate::engine::{AttrValue, NodeDesc};
use crate::ops::common::shape_plan::{EW_PARAM_COUNT, EW_PARAMS_NONE};
use crate::registry::NodeView;

/// One attribute slot: which attribute fills it, and what ONNX says it defaults to.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParamSlot {
    /// ONNX attribute name.
    pub name: &'static str,
    /// The ONNX schema default, used when the node omits the attribute.
    pub default: f32,
    /// True when the shader divides by this value, so zero must be declined rather than
    /// dispatched. Only `Celu` needs it today (`exp(x / alpha)`), but the flag lives in the table
    /// so the next such op is a row rather than a special case.
    pub nonzero: bool,
}

const fn slot(name: &'static str, default: f32) -> ParamSlot {
    ParamSlot {
        name,
        default,
        nonzero: false,
    }
}

const fn nonzero_slot(name: &'static str, default: f32) -> ParamSlot {
    ParamSlot {
        name,
        default,
        nonzero: true,
    }
}

/// The slot table, keyed by default-domain op type. Ops absent from this table have no
/// parameters and push a zeroed tail.
///
/// `Selu`'s defaults are the truncated constants the ONNX schema itself states
/// (alpha 1.67326319, gamma 1.05070102), not the full-precision Klambauer values — matching the
/// schema is what makes us agree with the CPU EP, which is the oracle Trinity compares against.
pub const SLOTS: &[(&str, &[ParamSlot])] = &[
    (
        "HardSigmoid",
        &[slot("alpha", 0.2), slot("beta", 0.5)] as &[ParamSlot],
    ),
    ("LeakyRelu", &[slot("alpha", 0.01)]),
    ("Elu", &[slot("alpha", 1.0)]),
    (
        "Selu",
        &[slot("alpha", 1.673_263_2), slot("gamma", 1.050_701)],
    ),
    ("Celu", &[nonzero_slot("alpha", 1.0)]),
    ("ThresholdedRelu", &[slot("alpha", 1.0)]),
    (
        "Shrink",
        &[slot("lambd", 0.5), slot("bias", 0.0)] as &[ParamSlot],
    ),
    ("Swish", &[slot("alpha", 1.0)]),
];

/// The slots for an op, or `None` when it takes no parameters.
pub fn slots_for(op_type: &str) -> Option<&'static [ParamSlot]> {
    SLOTS
        .iter()
        .find(|(name, _)| *name == op_type)
        .map(|(_, s)| *s)
}

/// Why a node's attributes cannot be turned into a parameter tail.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamError {
    /// The attribute is present but not a float (an exporter type error, or our table naming an
    /// attribute of the wrong type).
    NotAFloat { attr: &'static str },
    /// The value is one the shader cannot evaluate — today only a zero divisor.
    Rejected { attr: &'static str, value: f32 },
}

impl std::fmt::Display for ParamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParamError::NotAFloat { attr } => {
                write!(f, "attribute `{attr}` is not a float")
            }
            ParamError::Rejected { attr, value } => write!(
                f,
                "attribute `{attr}` = {value} is outside the range this EP evaluates \
                 (it would divide by zero in the shader)"
            ),
        }
    }
}

/// A source of float attributes. Implemented for both node representations so the slot table is
/// read once by the claim predicate (which sees a [`NodeView`] borrowed from ORT) and again by
/// the translate handler (which sees an owned [`NodeDesc`]) without duplicating the table or the
/// validation. If those two ever disagreed, we would claim on one set of values and dispatch with
/// another.
pub trait FloatAttrs {
    /// The float attribute `name`, `None` when absent.
    fn float_attr(&self, name: &str) -> Option<f32>;
    /// True when `name` is present but is not a float.
    fn attr_is_non_float(&self, name: &str) -> bool;
}

impl FloatAttrs for NodeView<'_> {
    fn float_attr(&self, name: &str) -> Option<f32> {
        self.attr_float(name)
    }

    fn attr_is_non_float(&self, _name: &str) -> bool {
        // `NodeView::attr_float` already returns `None` for a wrongly-typed attribute, and the
        // ORT ABI gives no cheap way to distinguish "absent" from "present with another type".
        // Treating the ambiguity as "absent" means we would claim using the default; the
        // translate side sees the owned `NodeDesc`, distinguishes the two properly, and errors.
        // That is the safe direction: a Compile-time error, not a wrong answer.
        false
    }
}

impl FloatAttrs for NodeDesc {
    fn float_attr(&self, name: &str) -> Option<f32> {
        match self.attributes.get(name) {
            Some(AttrValue::Float(v)) => Some(*v),
            _ => None,
        }
    }

    fn attr_is_non_float(&self, name: &str) -> bool {
        matches!(
            self.attributes.get(name),
            Some(
                AttrValue::Int(_)
                    | AttrValue::String(_)
                    | AttrValue::Ints(_)
                    | AttrValue::Floats(_)
                    | AttrValue::Strings(_)
            )
        )
    }
}

/// Resolve `op_type`'s attributes into the push-constant parameter tail.
///
/// Ops with no slot-table entry resolve to an all-zero tail, so this is safe to call
/// unconditionally.
pub fn resolve<A: FloatAttrs + ?Sized>(
    op_type: &str,
    attrs: &A,
) -> Result<[f32; EW_PARAM_COUNT], ParamError> {
    let Some(slots) = slots_for(op_type) else {
        return Ok(EW_PARAMS_NONE);
    };
    debug_assert!(
        slots.len() <= EW_PARAM_COUNT,
        "`{op_type}` declares more slots than the push-constant tail carries"
    );

    let mut out = EW_PARAMS_NONE;
    for (i, s) in slots.iter().enumerate() {
        if attrs.attr_is_non_float(s.name) {
            return Err(ParamError::NotAFloat { attr: s.name });
        }
        let v = attrs.float_attr(s.name).unwrap_or(s.default);
        if s.nonzero && v == 0.0 {
            return Err(ParamError::Rejected {
                attr: s.name,
                value: v,
            });
        }
        out[i] = v;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn node(op: &str, attrs: &[(&str, AttrValue)]) -> NodeDesc {
        NodeDesc {
            op_type: op.to_string(),
            attributes: attrs
                .iter()
                .map(|(k, v)| ((*k).to_string(), v.clone()))
                .collect::<BTreeMap<_, _>>(),
            ..Default::default()
        }
    }

    #[test]
    fn an_op_with_no_slots_resolves_to_a_zeroed_tail() {
        assert_eq!(resolve("Relu", &node("Relu", &[])), Ok(EW_PARAMS_NONE));
    }

    #[test]
    fn a_missing_attribute_resolves_to_the_onnx_default() {
        let p = resolve("LeakyRelu", &node("LeakyRelu", &[])).unwrap();
        assert_eq!(p[0], 0.01);
    }

    #[test]
    fn a_present_attribute_overrides_the_default() {
        let p = resolve(
            "LeakyRelu",
            &node("LeakyRelu", &[("alpha", AttrValue::Float(0.2))]),
        )
        .unwrap();
        assert_eq!(p[0], 0.2);
    }

    /// The bug this whole module exists to prevent: two attributes landing in each other's slot
    /// is invisible to both compilers and produces a plausible-looking wrong answer.
    #[test]
    fn two_slot_ops_fill_slots_in_declared_order() {
        let p = resolve(
            "HardSigmoid",
            &node(
                "HardSigmoid",
                &[
                    ("beta", AttrValue::Float(0.75)),
                    ("alpha", AttrValue::Float(0.125)),
                ],
            ),
        )
        .unwrap();
        assert_eq!(
            [p[0], p[1]],
            [0.125, 0.75],
            "alpha then beta, not map order"
        );

        let s = resolve(
            "Shrink",
            &node(
                "Shrink",
                &[
                    ("bias", AttrValue::Float(2.0)),
                    ("lambd", AttrValue::Float(3.0)),
                ],
            ),
        )
        .unwrap();
        assert_eq!([s[0], s[1]], [3.0, 2.0], "lambd then bias");
    }

    #[test]
    fn celu_alpha_zero_is_rejected_because_the_shader_divides_by_it() {
        assert_eq!(
            resolve("Celu", &node("Celu", &[("alpha", AttrValue::Float(0.0))])),
            Err(ParamError::Rejected {
                attr: "alpha",
                value: 0.0
            })
        );
        assert!(resolve("Celu", &node("Celu", &[])).is_ok(), "default 1.0");
    }

    #[test]
    fn a_wrongly_typed_attribute_is_an_error_not_a_silent_default() {
        assert_eq!(
            resolve("Elu", &node("Elu", &[("alpha", AttrValue::Int(2))])),
            Err(ParamError::NotAFloat { attr: "alpha" })
        );
    }

    /// The tail is four floats wide and the table must fit inside it. Overflowing would write
    /// past the block the host serialises, which `debug_assert!` catches in tests but which this
    /// checks unconditionally so a release build cannot ship a nine-slot op.
    #[test]
    fn no_op_declares_more_slots_than_the_tail_carries() {
        for (op, slots) in SLOTS {
            assert!(
                slots.len() <= EW_PARAM_COUNT,
                "`{op}` declares {} slots, tail carries {EW_PARAM_COUNT}",
                slots.len()
            );
            assert!(!slots.is_empty(), "`{op}` is in the table with no slots");
        }
    }

    /// Slot names must be distinct within an op, or the second would silently shadow the first
    /// in any name-keyed reasoning about the table.
    #[test]
    fn slot_names_are_distinct_within_each_op() {
        for (op, slots) in SLOTS {
            for (i, a) in slots.iter().enumerate() {
                for b in &slots[i + 1..] {
                    assert_ne!(a.name, b.name, "`{op}` repeats attribute `{}`", a.name);
                }
            }
        }
    }
}
