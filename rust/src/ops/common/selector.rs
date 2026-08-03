//! Op *selectors* — the specialisation-constant sibling of [`super::params`].
//!
//! # Why a second mechanism
//!
//! [`super::params`] carries op attributes that are **values**: `LeakyRelu`'s alpha changes what
//! the expression computes, not which expression is computed, so one compiled module serves every
//! alpha and the value rides the push-constant tail.
//!
//! A *selector* is an attribute (or an input-presence fact) that chooses **which expression** runs.
//! `elementwise.rs` recorded the distinction correctly and then drew the wrong conclusion from it:
//! that a selector therefore needs its own SPIR-V module. It does not. Vulkan already has the
//! mechanism — a specialisation constant is resolved when the pipeline is created, so the branch
//! it guards is folded away exactly as a `-D` would fold it, and the pipeline cache is already
//! keyed on `(shader, spec_constants)` so two nodes with different selectors cannot share a
//! pipeline. What a selector costs is one pipeline per distinct value, not one module per
//! distinct value.
//!
//! That is the difference between this module and a new template family, and it is why `IsInf`
//! and the short forms of `Clip` land here rather than as eight new stems in
//! `shader_variants.txt`.
//!
//! # The contract
//!
//! The selector reaches the shader as **specialisation constant id 2**, declared once in
//! `shaders/include/indexing.glsl` as `EW_SELECTOR` and defaulting to `0`. Every elementwise
//! dispatch pushes it, so an op with no selector pushes `0` and its shader never reads it.
//!
//! | op     | bit 0                        | bit 1                        |
//! |--------|------------------------------|------------------------------|
//! | `IsInf`| `detect_positive` is non-zero | `detect_negative` is non-zero |
//! | `Clip` | input 1 (`min`) is present   | input 2 (`max`) is present   |
//!
//! Bit order **is** the contract with the GLSL, exactly as slot order is in [`super::params`].
//! Both sides index by position; a reordering here is a wrong answer, not a compile error.
//!
//! # Why `Clip`'s selector is derived from inputs, not attributes
//!
//! `Clip`'s bounds are optional *inputs* from opset 11 on. `claim::ew_clip` used to decline the
//! one- and two-input forms `[arity]` and recorded the repair as "a shader variant that
//! substitutes ±infinity for the omitted bound". The repair is right about the shape of the
//! problem — an omitted bound is a different dispatch, not a different value — and one step wrong
//! about the remedy in a way worth writing down: **±infinity is not representable in the integer
//! variants of this row**, and `Clip`'s caps are `NUMERIC`, so the substitution would have had to
//! be dtype-conditional. Guarding the *comparison* instead of substituting a *bound* needs no
//! sentinel at any dtype:
//!
//! ```text
//! if (bit0) v = max(v, lo);
//! if (bit1) v = min(v, hi);
//! ```
//!
//! and both branches fold at pipeline creation. The omitted bound's binding is filled with input 0
//! as an inert placeholder — the same device `q_gemv` already uses for a missing `zero_points` —
//! so the descriptor set is the shape the module declares whatever the node's arity is.

use crate::engine::{AttrValue, NodeDesc};
use crate::registry::NodeView;

/// Index of the selector inside `KernelRequest::spec_constants`.
///
/// `build_spec_info_data` maps index → `constant_id`, so this is also the GLSL `constant_id`.
/// ids 0 and 1 are `local_size_x` and `EW_IDENTICAL`; see `indexing.glsl`.
pub const SPEC_SLOT: usize = 2;

/// One attribute-derived selector bit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SelectorBit {
    /// ONNX attribute name.
    pub name: &'static str,
    /// The ONNX schema default, used when the node omits the attribute.
    pub default: i64,
}

/// Where an op's selector bits come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectorSource {
    /// Bit *i* is set when the *i*-th named int attribute resolves non-zero.
    Attrs(&'static [SelectorBit]),
    /// Bit *i* is set when optional input `first + i` is present on the node.
    OptionalInputs { first: usize, count: usize },
}

/// The selector table, keyed by default-domain op type. Ops absent from it push `0`.
pub const SELECTORS: &[(&str, SelectorSource)] = &[
    (
        "IsInf",
        SelectorSource::Attrs(&[
            SelectorBit {
                name: "detect_positive",
                default: 1,
            },
            SelectorBit {
                name: "detect_negative",
                default: 1,
            },
        ]),
    ),
    (
        "Clip",
        SelectorSource::OptionalInputs { first: 1, count: 2 },
    ),
];

/// The selector source for an op, or `None` when it has no selector.
pub fn source_for(op_type: &str) -> Option<SelectorSource> {
    SELECTORS
        .iter()
        .find(|(name, _)| *name == op_type)
        .map(|(_, s)| *s)
}

/// Why a node's selector cannot be resolved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectorError {
    /// The attribute is present but is not an int — an exporter type error, or this table naming
    /// an attribute of the wrong type. Never silently defaulted: substituting the default here is
    /// how a graph asking for `detect_negative=0` would get `detect_negative=1` and a wrong answer.
    NotAnInt { attr: &'static str },
}

impl std::fmt::Display for SelectorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SelectorError::NotAnInt { attr } => write!(f, "attribute `{attr}` is not an int"),
        }
    }
}

/// A source of int attributes and input-presence facts.
///
/// Implemented for both node representations, so the claim predicate (which sees a [`NodeView`]
/// borrowed from ORT) and the translate handler (which sees an owned [`NodeDesc`]) read one table
/// through one code path. If those two ever disagreed we would claim on one selector and dispatch
/// with another — the exact defect [`super::params`]'s `FloatAttrs` exists to prevent, one
/// mechanism over.
pub trait SelectorFacts {
    /// The int attribute `name`, `None` when absent.
    fn int_attr(&self, name: &str) -> Option<i64>;
    /// True when `name` is present but is not an int.
    fn attr_is_non_int(&self, name: &str) -> bool;
    /// True when optional input `i` is present (a slot ORT reports, with a non-empty name).
    fn input_present(&self, i: usize) -> bool;
}

impl SelectorFacts for NodeView<'_> {
    fn int_attr(&self, name: &str) -> Option<i64> {
        self.attr_int(name)
    }

    fn attr_is_non_int(&self, _name: &str) -> bool {
        // `NodeView::attr_int` already returns `None` for a wrongly-typed attribute, and the ORT
        // ABI gives no cheap way to tell "absent" from "present with another type". Treating the
        // ambiguity as absent means we claim using the ONNX default; the translate side sees the
        // owned `NodeDesc`, distinguishes the two properly, and errors. That is the safe
        // direction — a Compile-time error, not a wrong answer. Same argument, same shape, as
        // `params::FloatAttrs for NodeView`.
        false
    }

    fn input_present(&self, i: usize) -> bool {
        self.input_type(i).is_some()
    }
}

impl SelectorFacts for NodeDesc {
    fn int_attr(&self, name: &str) -> Option<i64> {
        match self.attributes.get(name) {
            Some(AttrValue::Int(v)) => Some(*v),
            _ => None,
        }
    }

    fn attr_is_non_int(&self, name: &str) -> bool {
        matches!(
            self.attributes.get(name),
            Some(
                AttrValue::Float(_)
                    | AttrValue::String(_)
                    | AttrValue::Ints(_)
                    | AttrValue::Floats(_)
                    | AttrValue::Strings(_)
            )
        )
    }

    fn input_present(&self, i: usize) -> bool {
        // An omitted *interior* optional input arrives as a slot with an empty name, so the length
        // check alone would report `Clip(x, "", max)` as a two-input node with `min` present and
        // clamp against a buffer that has no producer.
        self.inputs.get(i).is_some_and(|t| !t.name.is_empty())
    }
}

/// Resolve `op_type`'s selector into specialisation constant [`SPEC_SLOT`].
///
/// Ops with no table entry resolve to `0`, so this is safe to call unconditionally.
pub fn resolve<F: SelectorFacts + ?Sized>(
    op_type: &str,
    facts: &F,
) -> Result<u32, SelectorError> {
    let Some(source) = source_for(op_type) else {
        return Ok(0);
    };
    match source {
        SelectorSource::Attrs(bits) => {
            let mut out = 0u32;
            for (i, b) in bits.iter().enumerate() {
                if facts.attr_is_non_int(b.name) {
                    return Err(SelectorError::NotAnInt { attr: b.name });
                }
                if facts.int_attr(b.name).unwrap_or(b.default) != 0 {
                    out |= 1 << i;
                }
            }
            Ok(out)
        }
        SelectorSource::OptionalInputs { first, count } => {
            let mut out = 0u32;
            for i in 0..count {
                if facts.input_present(first + i) {
                    out |= 1 << i;
                }
            }
            // `0` is a legal selector here, not an error. A `Clip` with neither bound is the
            // identity, and this project already claims a lone `Identity` — the row is registered
            // as an island-welder precisely because a shape-preserving copy that keeps a fused
            // island whole is worth more than the copy costs. Refusing `Clip(x)` while claiming
            // `Identity(x)` would be the same graph decided two ways by which op name the exporter
            // happened to emit.
            Ok(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::TensorRef;
    use std::collections::BTreeMap;

    fn node(op: &str, attrs: &[(&str, AttrValue)], inputs: &[&str]) -> NodeDesc {
        NodeDesc {
            op_type: op.to_string(),
            attributes: attrs
                .iter()
                .map(|(k, v)| ((*k).to_string(), v.clone()))
                .collect::<BTreeMap<_, _>>(),
            inputs: inputs
                .iter()
                .map(|n| TensorRef {
                    name: (*n).to_string(),
                    desc: None,
                    is_initializer: false,
                })
                .collect(),
            ..Default::default()
        }
    }

    #[test]
    fn an_op_with_no_selector_resolves_to_zero() {
        assert_eq!(resolve("Relu", &node("Relu", &[], &["x"])), Ok(0));
    }

    /// `IsInf`'s ONNX defaults are both 1, so an attribute-free node detects both infinities.
    #[test]
    fn isinf_defaults_to_detecting_both_infinities() {
        assert_eq!(resolve("IsInf", &node("IsInf", &[], &["x"])), Ok(0b11));
    }

    #[test]
    fn isinf_bits_follow_the_declared_order_not_map_order() {
        // bit 0 is detect_positive, bit 1 is detect_negative. A table read in map order would
        // return 0b01 for this node, which is a wrong answer on every negative infinity.
        let n = node(
            "IsInf",
            &[
                ("detect_negative", AttrValue::Int(1)),
                ("detect_positive", AttrValue::Int(0)),
            ],
            &["x"],
        );
        assert_eq!(resolve("IsInf", &n), Ok(0b10));

        let n = node(
            "IsInf",
            &[
                ("detect_negative", AttrValue::Int(0)),
                ("detect_positive", AttrValue::Int(1)),
            ],
            &["x"],
        );
        assert_eq!(resolve("IsInf", &n), Ok(0b01));
    }

    #[test]
    fn a_wrongly_typed_selector_attribute_is_an_error_not_a_silent_default() {
        assert_eq!(
            resolve(
                "IsInf",
                &node("IsInf", &[("detect_positive", AttrValue::Float(1.0))], &["x"])
            ),
            Err(SelectorError::NotAnInt {
                attr: "detect_positive"
            })
        );
    }

    #[test]
    fn clip_reads_its_selector_from_input_presence() {
        assert_eq!(resolve("Clip", &node("Clip", &[], &["x", "lo", "hi"])), Ok(0b11));
        assert_eq!(resolve("Clip", &node("Clip", &[], &["x", "lo"])), Ok(0b01));
    }

    /// The case that makes the empty-name check load-bearing: ONNX writes an omitted *interior*
    /// optional input as an empty name, and a length-only check would read this as "min present".
    #[test]
    fn clip_with_an_omitted_interior_bound_sets_only_the_max_bit() {
        assert_eq!(resolve("Clip", &node("Clip", &[], &["x", "", "hi"])), Ok(0b10));
    }

    /// A `Clip` with neither bound is the identity — and it is **claimed**, not refused.
    ///
    /// The temptation is to decline it as "no work worth a dispatch". This project already
    /// answered that question the other way for `Identity`, whose row exists to keep a fused
    /// island whole rather than to compute anything. Two ops with the same semantics decided two
    /// different ways by name would be a partition boundary the exporter chose for us.
    #[test]
    fn clip_with_no_bounds_at_all_is_the_identity_and_still_claimed() {
        assert_eq!(resolve("Clip", &node("Clip", &[], &["x"])), Ok(0));
        assert_eq!(resolve("Clip", &node("Clip", &[], &["x", "", ""])), Ok(0));
    }

    /// The tail is one `u32`; a table row wider than 32 bits would silently drop its high bits.
    #[test]
    fn no_op_declares_more_selector_bits_than_the_constant_carries() {
        for (op, source) in SELECTORS {
            let n = match source {
                SelectorSource::Attrs(bits) => bits.len(),
                SelectorSource::OptionalInputs { count, .. } => *count,
            };
            assert!(n > 0, "`{op}` is in the table with no bits");
            assert!(n < 32, "`{op}` declares {n} selector bits");
        }
    }

    /// Attribute names must be distinct within an op, or the second bit would shadow the first.
    #[test]
    fn selector_attribute_names_are_distinct_within_each_op() {
        for (op, source) in SELECTORS {
            let SelectorSource::Attrs(bits) = source else {
                continue;
            };
            for (i, a) in bits.iter().enumerate() {
                for b in &bits[i + 1..] {
                    assert_ne!(a.name, b.name, "`{op}` repeats attribute `{}`", a.name);
                }
            }
        }
    }
}
