//! Op *forms* — attributes that change the answer without changing the pipeline. Owner: Mouse.
//!
//! # The defect this exists to close, measured rather than argued
//!
//! `Conv` landed on 2026-08-03 with four ledger entries, and its module said in as many words
//! that `group`, `strides`, `dilations` and `pads` "are **not** key components, so a ledger entry
//! says nothing about them". That was accurate and it was not harmless. Counting the four
//! attribute classes over MobileNetV2's 52 convolutions on 2026-08-04:
//!
//! ```text
//! grouped=0 stride=0 dil=0 pad=0   x34
//! grouped=0 stride=1 dil=0 pad=1   x1
//! grouped=1 stride=0 dil=0 pad=1   x13
//! grouped=1 stride=1 dil=0 pad=1   x4
//! ```
//!
//! The four proven entries were all built at `group=1, strides=1, dilations=1, pads=(1,1,1,1)` —
//! **`grouped=0 stride=0 dil=0 pad=1`, which is not one of those four rows.** Every one of the 52
//! nodes was claimed against a ledger entry obtained on a form the model does not contain, and
//! the key could not tell, because the key did not carry the distinction. That is not a
//! hypothetical: it is the state that shipped, and the only reason it did not produce a wrong
//! answer is that the kernel happened to be right.
//!
//! # Why this is not [`super::selector`], and not a seventh key component
//!
//! A [`super::selector::SelectorSource`] bit reaches the shader as specialisation constant 2 and
//! therefore *forks the pipeline*. `Conv`'s `group` does not fork anything — it is a push
//! constant, one module serves every value, and adding a spec constant would multiply pipelines
//! for no reason. So a form bit is deliberately **not** a selector: it is visible to the proof
//! key and to nothing else.
//!
//! It is also not a seventh field in the key. `ProofKey::validate` requires exactly six
//! components and rejects anything else, on the §8.9.4 argument that a key which can be written
//! short is a key that can be written wide. Widening the schema is Morpheus's call on a file that
//! is not mine. The `variant` component, however, is already the place where a per-op distinction
//! is folded in — `variant_key` appends `@sel<n>` there for exactly this reason — so a form tag
//! rides the same component under a different separator. **Six components in, six components
//! out, and no other op's key moves.**
//!
//! # Why booleans and not the values
//!
//! A key carrying `strides=[2,2]` would mint a new form for every stride a model happens to use,
//! and each new one declines `[unproven]` until someone proves it. That trades a false claim for
//! a coverage cliff. The classes that matter are the ones where the *kernel takes a different
//! path*, and in `conv_f32.comp` there are exactly four:
//!
//! * `group > 1`   — `c0 = g * cpg` is non-zero, so the input-channel window moves per output
//!   channel. At `group == 1` that expression is dead.
//! * `stride > 1`  — `oy * stride_h` stops being `oy`.
//! * `dilation > 1`— `ky * dil_h` stops being `ky`.
//! * `pad > 0`     — the `iy < 0` border branch can fire. With zero pads it never does, and the
//!   signed-index care that branch exists for is untested.
//!
//! Sixteen forms is the whole space, four of them occur in the one CNN we have, and a stride of 3
//! is not a different risk from a stride of 2 — it is the same expression with a different push
//! constant. The bit is the honest granularity: it separates code paths, not numbers.
//!
//! **The falsifier**, stated before it is needed: if a wrong answer is ever found that a form bit
//! could not have separated — say `stride > kernel_shape`, which skips input elements entirely —
//! then the boolean is too coarse for that attribute and it needs a class, not a bit. That would
//! be a change to [`FORMS`] and to nothing else.

use crate::registry::NodeView;

/// How one form bit is decided from a node's attributes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormTest {
    /// A scalar int attribute (ONNX default `default`) differs from `baseline`.
    IntNot { default: i64, baseline: i64 },
    /// Any entry of an int-list attribute (ONNX default: all `default`) exceeds `baseline`.
    IntsAnyAbove { default: i64, baseline: i64 },
}

/// One named form bit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FormBit {
    /// The ONNX attribute it reads.
    pub attr: &'static str,
    /// The name that appears in the proof key when the bit is set. Short, because it is read in
    /// a key beside five other components.
    pub tag: &'static str,
    /// The predicate.
    pub test: FormTest,
}

/// The tag a node gets when no form bit is set — the plainest form of the op.
///
/// A name and not an empty string: an empty component reads as "unknown" and
/// `ProofKey::validate` rejects empty components outright, on the argument that an empty field
/// is a wildcard by another name.
pub const BASE_TAG: &str = "base";

/// The form table, keyed by default-domain op type. Ops absent from it carry no form tag and
/// their keys are byte-identical to what they were before this module existed.
///
/// Order within a row is the rendering order and therefore part of the key. Reordering a row
/// renames every key it produces, which is a ledger migration, not a refactor.
pub const FORMS: &[(&str, &[FormBit])] = &[
    (
        "Conv",
        &[
            FormBit {
                attr: "group",
                tag: "grouped",
                test: FormTest::IntNot {
                    default: 1,
                    baseline: 1,
                },
            },
            FormBit {
                attr: "strides",
                tag: "strided",
                test: FormTest::IntsAnyAbove {
                    default: 1,
                    baseline: 1,
                },
            },
            FormBit {
                attr: "dilations",
                tag: "dilated",
                test: FormTest::IntsAnyAbove {
                    default: 1,
                    baseline: 1,
                },
            },
            FormBit {
                attr: "pads",
                tag: "padded",
                test: FormTest::IntsAnyAbove {
                    default: 0,
                    baseline: 0,
                },
            },
        ],
    ),
    (
        "Gemm",
        &[
            // `transA`/`transB` swap which index strides — the same wrong-answer class as
            // `group`, and the same remedy. `alpha`/`beta` are deliberately absent: they are
            // multiplied into one expression, so every value runs the same instructions.
            FormBit {
                attr: "transA",
                tag: "transA",
                test: FormTest::IntNot {
                    default: 0,
                    baseline: 0,
                },
            },
            FormBit {
                attr: "transB",
                tag: "transB",
                test: FormTest::IntNot {
                    default: 0,
                    baseline: 0,
                },
            },
        ],
    ),
];

/// The form bits declared for an op, or `None` when it declares none.
pub fn bits_for(op_type: &str) -> Option<&'static [FormBit]> {
    FORMS
        .iter()
        .find(|(name, _)| *name == op_type)
        .map(|(_, b)| *b)
}

/// A source of the attributes a form bit reads.
///
/// A trait rather than a `NodeView` method so the unit tests below can state a node as a table
/// instead of fabricating an ORT graph, and so a future translate-side assertion reads the same
/// table through the same code — the divergence [`super::selector::SelectorFacts`] exists to
/// prevent, one mechanism over.
pub trait FormFacts {
    /// A scalar int attribute, `None` when absent or not an int.
    fn form_int(&self, name: &str) -> Option<i64>;
    /// An int-list attribute, `None` when absent or not an int list.
    fn form_ints(&self, name: &str) -> Option<Vec<i64>>;
}

impl FormFacts for NodeView<'_> {
    fn form_int(&self, name: &str) -> Option<i64> {
        self.attr_int(name)
    }

    fn form_ints(&self, name: &str) -> Option<Vec<i64>> {
        self.attr_ints(name)
    }
}

/// Render `op_type`'s form tag for this node, or `None` when the op declares no form bits.
///
/// Total by construction: a missing attribute takes the ONNX default, which is what the kernel
/// will do, so the tag names the form the kernel will actually run rather than the form the
/// exporter bothered to write down.
pub fn tag_for<F: FormFacts + ?Sized>(op_type: &str, facts: &F) -> Option<String> {
    let bits = bits_for(op_type)?;
    let mut set: Vec<&str> = Vec::new();
    for b in bits {
        let on = match b.test {
            FormTest::IntNot { default, baseline } => {
                facts.form_int(b.attr).unwrap_or(default) != baseline
            }
            FormTest::IntsAnyAbove { default, baseline } => facts
                .form_ints(b.attr)
                .unwrap_or_else(|| vec![default])
                .iter()
                .any(|&v| v > baseline),
        };
        if on {
            set.push(b.tag);
        }
    }
    Some(if set.is_empty() {
        BASE_TAG.to_string()
    } else {
        set.join("+")
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[derive(Default)]
    struct Node {
        ints: BTreeMap<&'static str, i64>,
        lists: BTreeMap<&'static str, Vec<i64>>,
    }

    impl FormFacts for Node {
        fn form_int(&self, name: &str) -> Option<i64> {
            self.ints.get(name).copied()
        }
        fn form_ints(&self, name: &str) -> Option<Vec<i64>> {
            self.lists.get(name).cloned()
        }
    }

    fn conv(group: i64, strides: [i64; 2], dilations: [i64; 2], pads: [i64; 4]) -> Node {
        let mut n = Node::default();
        n.ints.insert("group", group);
        n.lists.insert("strides", strides.to_vec());
        n.lists.insert("dilations", dilations.to_vec());
        n.lists.insert("pads", pads.to_vec());
        n
    }

    #[test]
    fn an_op_with_no_form_bits_gets_no_tag() {
        assert_eq!(tag_for("Relu", &Node::default()), None);
    }

    /// A node with no attributes at all is the ONNX default `Conv`, which is the plainest form.
    #[test]
    fn schema_defaults_render_as_the_base_form() {
        assert_eq!(tag_for("Conv", &Node::default()).unwrap(), BASE_TAG);
    }

    /// The four rows the census found in MobileNetV2, each rendering to a distinct tag. This is
    /// the assertion that would have failed before the module existed — all four collapsed onto
    /// one key then.
    #[test]
    fn the_four_mobilenetv2_conv_forms_are_four_distinct_tags() {
        let dense_1x1 = tag_for("Conv", &conv(1, [1, 1], [1, 1], [0, 0, 0, 0])).unwrap();
        let dense_strided = tag_for("Conv", &conv(1, [2, 2], [1, 1], [0, 1, 0, 1])).unwrap();
        let depthwise = tag_for("Conv", &conv(96, [1, 1], [1, 1], [1, 1, 1, 1])).unwrap();
        let depthwise_strided = tag_for("Conv", &conv(96, [2, 2], [1, 1], [0, 1, 0, 1])).unwrap();
        assert_eq!(dense_1x1, "base");
        assert_eq!(dense_strided, "strided+padded");
        assert_eq!(depthwise, "grouped+padded");
        assert_eq!(depthwise_strided, "grouped+strided+padded");
        let all = [&dense_1x1, &dense_strided, &depthwise, &depthwise_strided];
        for (i, a) in all.iter().enumerate() {
            for b in &all[i + 1..] {
                assert_ne!(a, b, "two MobileNetV2 forms collapsed onto one key");
            }
        }
    }

    /// The form the four shipped `Conv` entries were proven on. It is a fifth tag, and the point
    /// is that it is **not** any of the four above.
    #[test]
    fn the_form_the_shipped_entries_were_proven_on_is_none_of_the_models_forms() {
        let proven = tag_for("Conv", &conv(1, [1, 1], [1, 1], [1, 1, 1, 1])).unwrap();
        assert_eq!(proven, "padded");
        for t in [
            "base",
            "strided+padded",
            "grouped+padded",
            "grouped+strided+padded",
        ] {
            assert_ne!(proven, t);
        }
    }

    /// A single non-unit entry is enough — `strides=[1,2]` is a strided convolution.
    #[test]
    fn one_non_unit_entry_sets_the_bit() {
        assert_eq!(
            tag_for("Conv", &conv(1, [1, 2], [1, 1], [0, 0, 0, 0])).unwrap(),
            "strided"
        );
    }

    /// An asymmetric pad on one edge only still crosses the border branch.
    #[test]
    fn an_asymmetric_single_edge_pad_is_padded() {
        assert_eq!(
            tag_for("Conv", &conv(1, [1, 1], [1, 1], [0, 0, 1, 0])).unwrap(),
            "padded"
        );
    }

    #[test]
    fn bit_order_is_the_table_order_not_the_order_they_were_set() {
        assert_eq!(
            tag_for("Conv", &conv(2, [2, 2], [2, 2], [1, 1, 1, 1])).unwrap(),
            "grouped+strided+dilated+padded"
        );
    }

    #[test]
    fn gemm_transposes_are_four_distinct_forms() {
        let mk = |a: i64, b: i64| {
            let mut n = Node::default();
            n.ints.insert("transA", a);
            n.ints.insert("transB", b);
            tag_for("Gemm", &n).unwrap()
        };
        assert_eq!(mk(0, 0), "base");
        assert_eq!(mk(1, 0), "transA");
        assert_eq!(mk(0, 1), "transB");
        assert_eq!(mk(1, 1), "transA+transB");
    }

    /// A tag may never contain the key's own separators, or a six-component key would parse as
    /// seven and `ProofKey::validate` would reject a key the registry itself minted.
    #[test]
    fn no_tag_contains_a_key_separator() {
        for (op, bits) in FORMS {
            for b in *bits {
                assert!(
                    !b.tag.contains('/') && !b.tag.contains(':'),
                    "`{op}` bit `{}` contains a proof-key separator",
                    b.tag
                );
            }
        }
        assert!(!BASE_TAG.contains('/') && !BASE_TAG.contains(':'));
    }

    #[test]
    fn form_tags_are_distinct_within_each_op() {
        for (op, bits) in FORMS {
            for (i, a) in bits.iter().enumerate() {
                for b in &bits[i + 1..] {
                    assert_ne!(a.tag, b.tag, "`{op}` repeats form tag `{}`", a.tag);
                    assert_ne!(a.attr, b.attr, "`{op}` repeats attribute `{}`", a.attr);
                }
            }
            assert!(!bits.is_empty(), "`{op}` is in the table with no bits");
        }
    }

    /// Every op in the table must be a registered op, or the table is describing a form of
    /// something this EP never sees and the tag would never appear in any key.
    #[test]
    fn every_form_table_op_is_registered() {
        for (op, _) in FORMS {
            assert!(
                crate::registry::all_specs().any(|s| s.op_type == *op),
                "`{op}` declares form bits but has no registry row"
            );
        }
    }
}
