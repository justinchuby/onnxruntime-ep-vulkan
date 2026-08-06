//! Conservative rank/shape inference over the graph ORT hands `GetCapability`.
//!
//! # The defect this repairs
//!
//! ORT runs its own shape inference before partitioning, and on a TF-exported transformer it
//! stops at the first `Reshape` whose target is computed rather than stored. BERT-SQuAD-12 is
//! the measured case (`bench/results/rank_chain_bert_before.json`, 2026-08-06): **1773 of 3103**
//! node edges carry no rank at all, **98 of 98** `MatMul` `A` operands among them, and every one
//! of its 58 computed `Reshape` targets is produced by this chain —
//!
//! ```text
//! Shape(input_ids)  ->  Cast(to=FLOAT)  ->  Slice  ->  Squeeze(axes=[0])
//!                   ->  Cast(to=INT32)  ->  Unsqueeze(axes=[0])  -\
//!   initializer 1 (rank 0, int32) -> Unsqueeze(axes=[0])  ----------> Concat(axis=0) -> Cast(to=INT64) -> Reshape.shape
//!   initializer 256 (rank 0, int32) -> Unsqueeze(axes=[0]) -------/
//! ```
//!
//! Every claim predicate in this EP then reads `[unknown-rank]` for the whole downstream graph,
//! which is not a kernel gap: the kernels exist, the ranks do not.
//!
//! # What this module infers, and what it refuses to
//!
//! Only facts that ONNX **guarantees**. Three rules make that concrete and they are the contract
//! (`docs/DESIGN.md` §8.10):
//!
//! 1. **A length is not a value.** `Shape`'s output is a 1-D tensor whose *length* is the rank of
//!    its input — provable whenever the input's rank is known, and provable independently of what
//!    the extents are. Its *elements* are the extents, and a symbolic extent stays unknown.
//!    This distinction is the whole unlock on BERT: the batch extent in the chain above is never
//!    known, and it never needs to be, because what `Reshape` takes from the chain is a *length*.
//! 2. **A `Cast` preserves shape always and values only sometimes.** Element order and count are
//!    invariant under `Cast`, so rank/extents propagate unconditionally. Element *values* survive
//!    only into an integral destination, and only when the source fact itself came from an
//!    integral tensor. The real BERT chain casts a shape vector to `float` and back, which is why
//!    that is not a hypothetical: values are dropped at the `float` hop and lengths are not.
//! 3. **Unknown is not zero, and rank 0 is not "no rank".** ORT reports rank 0 both for a genuine
//!    scalar and for a rank it never resolved (`GetDimensionsCount` cannot tell them apart). A
//!    rank-0 reading from ORT is therefore **not** seeded as a fact here. A rank-0 fact is only
//!    ever created by a rule that proves it — `Squeeze` of a rank-1, a rank-0 initializer whose
//!    tensor we actually read — which is what keeps the vacuity defect of 2026-08-04 from being
//!    reproduced one module over.
//!
//! Everything else fails closed: an op with no rule, an attribute that cannot be read, an axis
//! out of range, an overflowing product, a rank above [`MAX_INFER_RANK`], a fact that contradicts
//! one already proven — all of them leave the value **unknown**, and an unknown value declines
//! exactly as it declined before this module existed.
//!
//! # Monotonicity, contradiction and termination
//!
//! Facts only ever grow: an unknown extent may become known, a known extent never changes. A rule
//! that produces a fact incompatible with one already held **poisons** that value — the fact is
//! withdrawn and no rule may re-establish it — because two derivations disagreeing means at least
//! one is wrong and we cannot tell which. Poisoning is counted ([`Stats::contradictions`]) and is
//! expected to be zero; a non-zero count is a bug in a rule, not a property of the model.
//!
//! Sweeps run to a fixed point, bounded by [`SWEEP_LIMIT`]. ONNX requires a topologically sorted
//! node list, so two sweeps suffice in practice; the bound is what makes a malformed or cyclic
//! list terminate rather than hang.
//!
//! # This module knows nothing about ORT
//!
//! It takes [`InferNode`] values built by `ep.rs` from `NodeView` and returns an
//! [`InferredShapes`] overlay keyed by value name. That is deliberate: every rule here is a
//! statement about the ONNX spec and is unit-testable without a graph, a driver or a device.

use std::collections::{HashMap, HashSet};

/// Largest rank this module will record a fact for.
///
/// Matches the shared indexing helper's limit; a deeper tensor is declined by every predicate
/// anyway, so proving its rank would only add a fact nothing can use.
pub const MAX_INFER_RANK: usize = 8;

/// Largest element count this module will track *values* for.
///
/// Shape operands are a handful of int64s. The bound stops a rule from materialising a large
/// constant weight tensor into the fact map, which would be a memory cost with no claim benefit.
pub const MAX_INFER_INTS: usize = 128;

/// Upper bound on fixed-point sweeps. See the module docs on termination.
pub const SWEEP_LIMIT: usize = 64;

// ONNX `TensorProto.DataType` values the `Cast` rule needs by name.
const ONNX_INT32: i64 = 6;
const ONNX_INT64: i64 = 7;

/// The kind of ONNX attribute a rule wants, so the caller knows which ORT accessor to use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttrKind {
    /// A single `int`.
    Int,
    /// An `ints` list.
    Ints,
}

/// One attribute value, read by the caller and handed to the rules.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AttrValue {
    /// A single `int`.
    Int(i64),
    /// An `ints` list.
    Ints(Vec<i64>),
}

/// One node, in the vocabulary the rules need and nothing more.
///
/// `inputs`/`outputs` are value names; an **empty** name is an omitted optional input, which is
/// how ORT reports `Clip(x, , max)`. Attributes are only those [`wanted_attrs`] asked for.
#[derive(Clone, Debug, Default)]
pub struct InferNode {
    /// ONNX op type, e.g. `Concat`.
    pub op_type: String,
    /// ONNX domain; empty or `ai.onnx` for the default domain. Rules exist for that domain only.
    pub domain: String,
    /// The opset version this node's op resolved against. `0` when ORT did not report one.
    pub since_version: i32,
    /// Input value names, in order.
    pub inputs: Vec<String>,
    /// Output value names, in order.
    pub outputs: Vec<String>,
    /// The attributes [`wanted_attrs`] named for this op, when present on the node.
    pub attrs: Vec<(&'static str, AttrValue)>,
}

impl InferNode {
    fn attr(&self, name: &str) -> Option<&AttrValue> {
        self.attrs.iter().find(|(n, _)| *n == name).map(|(_, v)| v)
    }

    fn attr_int(&self, name: &str) -> Option<i64> {
        match self.attr(name)? {
            AttrValue::Int(v) => Some(*v),
            AttrValue::Ints(_) => None,
        }
    }

    fn attr_ints(&self, name: &str) -> Option<&[i64]> {
        match self.attr(name)? {
            AttrValue::Ints(v) => Some(v),
            AttrValue::Int(_) => None,
        }
    }

    fn input(&self, i: usize) -> Option<&str> {
        self.inputs
            .get(i)
            .map(String::as_str)
            .filter(|s| !s.is_empty())
    }

    fn is_default_domain(&self) -> bool {
        self.domain.is_empty() || self.domain == "ai.onnx"
    }
}

/// What this module proved about one graph value.
///
/// `shape: None` means the rank itself is unproven — the state every value starts in. A `-1`
/// entry means the rank is proven and that extent is not. `ints` carries proven **element
/// values** of an integer tensor in flat order, with a `None` element meaning "this element's
/// value is unproven, the length is not".
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Fact {
    /// Proven shape. `-1` marks an extent that is not proven.
    pub shape: Option<Vec<i64>>,
    /// Proven integer element values in flat order, when this value is an integer tensor.
    pub ints: Option<Vec<Option<i64>>>,
}

impl Fact {
    /// A fact carrying a shape only.
    pub fn of_shape(shape: Vec<i64>) -> Fact {
        Fact {
            shape: Some(shape),
            ints: None,
        }
    }

    /// Proven rank, if any.
    pub fn rank(&self) -> Option<usize> {
        self.shape.as_ref().map(Vec::len)
    }

    /// Whether this fact adds anything at all.
    fn is_empty(&self) -> bool {
        self.shape.is_none() && self.ints.is_none()
    }
}

/// Counters describing one inference pass. Reported at `GetCapability`; not an ABI counter.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    /// Values ORT itself gave a rank of 1 or more.
    pub declared: usize,
    /// Values that gained a rank this pass proved and ORT did not report.
    pub ranks_proved: usize,
    /// Extents proven for a value ORT reported as symbolic, or on a rank this pass proved.
    pub extents_proved: usize,
    /// Values withdrawn because two derivations disagreed. Expected zero; see the module docs.
    pub contradictions: usize,
    /// Sweeps actually run before the fixed point (or [`SWEEP_LIMIT`]).
    pub sweeps: usize,
    /// True when the sweep budget ran out with facts still changing.
    pub hit_sweep_limit: bool,
}

/// The read-only result of a pass: value name → proven shape.
///
/// Only values whose shape this pass proved *beyond* what ORT declared are kept, because those
/// are the only ones an overlay can change. `-1` entries mean "rank proven, extent not".
#[derive(Clone, Debug, Default)]
pub struct InferredShapes {
    shapes: HashMap<String, Vec<i64>>,
    stats: Stats,
}

impl InferredShapes {
    /// The proven shape for a value, if this pass proved one.
    pub fn shape(&self, name: &str) -> Option<&[i64]> {
        self.shapes.get(name).map(Vec::as_slice)
    }

    /// How many values carry a proven shape.
    pub fn len(&self) -> usize {
        self.shapes.len()
    }

    /// Whether the pass proved nothing at all.
    pub fn is_empty(&self) -> bool {
        self.shapes.is_empty()
    }

    /// Counters for the pass that produced this overlay.
    pub fn stats(&self) -> Stats {
        self.stats
    }

    /// Absorb another pass's facts, keeping this one's on any disagreement.
    ///
    /// A model with control flow reaches `GetCapability` once per graph, and `Compile` is handed
    /// fused subgraphs whose boundary producers live in whichever graph proved them. Keeping one
    /// accumulated overlay per session is what lets `Compile` honour the facts `GetCapability`
    /// claimed on. Value names are graph-unique, so a name collision across graphs would be a
    /// contradiction; it is resolved by **dropping the value entirely**, never by picking a side.
    pub fn absorb(&mut self, other: &InferredShapes) {
        for (name, shape) in &other.shapes {
            match self.shapes.get(name) {
                None => {
                    self.shapes.insert(name.clone(), shape.clone());
                }
                Some(have) if have == shape => {}
                Some(_) => {
                    self.shapes.remove(name);
                    self.stats.contradictions += 1;
                }
            }
        }
        self.stats.declared += other.stats.declared;
        self.stats.ranks_proved += other.stats.ranks_proved;
        self.stats.extents_proved += other.stats.extents_proved;
        self.stats.contradictions += other.stats.contradictions;
        self.stats.sweeps = self.stats.sweeps.max(other.stats.sweeps);
        self.stats.hit_sweep_limit |= other.stats.hit_sweep_limit;
    }

    /// Refine one edge reading with what was proven, returning the shape a predicate should read.
    ///
    /// **Never contradicts ORT.** Three cases, and the third is the one that matters:
    ///
    /// * ORT reported nothing, or reported the ambiguous rank-0 — take the proven shape.
    /// * ORT reported a rank this pass also proved — keep ORT's extents, filling only the axes
    ///   ORT left symbolic with extents this pass proved.
    /// * ORT reported a rank that disagrees with the proven one — keep ORT's, unchanged. A
    ///   disagreement means one of the two readings is wrong and this module does not get to
    ///   decide that it is the other one.
    pub fn refine(&self, name: &str, declared: Option<&[i64]>) -> Option<Vec<i64>> {
        let proven = self.shape(name)?;
        match declared {
            // ORT's rank-0 reading is the ambiguous one (`DESIGN.md` §8.8): it means "scalar" and
            // "rank never established" with the same bytes. A proven shape is strictly better
            // information than an ambiguity, so it replaces it.
            None | Some([]) => Some(proven.to_vec()),
            Some(dims) if dims.len() == proven.len() => {
                let mut out = dims.to_vec();
                for (o, p) in out.iter_mut().zip(proven.iter()) {
                    if *o < 0 && *p >= 0 {
                        *o = *p;
                    }
                }
                Some(out)
            }
            Some(dims) => Some(dims.to_vec()),
        }
    }
}

/// A running inference pass.
#[derive(Debug, Default)]
pub struct Inference {
    facts: HashMap<String, Fact>,
    /// Values whose facts were withdrawn on contradiction. Never re-established.
    poisoned: HashSet<String>,
    /// Values ORT declared a rank of 1 or more for, and the shape it declared.
    declared: HashMap<String, Vec<i64>>,
    stats: Stats,
}

impl Inference {
    /// A pass with nothing proven.
    pub fn new() -> Inference {
        Inference::default()
    }

    /// Seed one value from ORT's own reading.
    ///
    /// A rank-0 reading is **ignored**, not recorded as a scalar: see rule 3 in the module docs.
    /// Passing `None` (ORT reported no shape) is a no-op and is accepted so callers can feed every
    /// edge unconditionally.
    pub fn declare(&mut self, name: &str, shape: Option<&[i64]>) {
        let Some(dims) = shape else { return };
        if name.is_empty() || dims.is_empty() || dims.len() > MAX_INFER_RANK {
            return;
        }
        if self.declared.contains_key(name) {
            return;
        }
        let dims: Vec<i64> = dims.iter().map(|d| if *d < 0 { -1 } else { *d }).collect();
        self.declared.insert(name.to_string(), dims.clone());
        self.stats.declared += 1;
        self.set(name, Fact::of_shape(dims));
    }

    /// Seed one value from a constant initializer whose tensor was actually read.
    ///
    /// This is the only path by which a **rank-0 fact** enters the map, and it is sound precisely
    /// because the tensor was read: an initializer's rank is a property of stored data, not of an
    /// inference that may not have run.
    pub fn constant(&mut self, name: &str, shape: &[i64], ints: Option<Vec<i64>>) {
        if name.is_empty() || shape.len() > MAX_INFER_RANK || shape.iter().any(|d| *d < 0) {
            return;
        }
        let ints = ints.and_then(|v| {
            if v.len() > MAX_INFER_INTS {
                None
            } else {
                Some(v.into_iter().map(Some).collect::<Vec<_>>())
            }
        });
        // An initializer's declared shape is exact, so it counts as a declaration for the
        // "did this pass add anything" bookkeeping — an overlay entry that merely repeats a
        // rank ORT already had is not an unlock and must not be counted as one.
        self.declared.insert(name.to_string(), shape.to_vec());
        self.set(
            name,
            Fact {
                shape: Some(shape.to_vec()),
                ints,
            },
        );
    }

    /// Proven fact for a value, if any.
    pub fn fact(&self, name: &str) -> Option<&Fact> {
        self.facts.get(name)
    }

    fn shape_of(&self, name: &str) -> Option<&[i64]> {
        self.facts.get(name)?.shape.as_deref()
    }

    fn ints_of(&self, name: &str) -> Option<&[Option<i64>]> {
        self.facts.get(name)?.ints.as_deref()
    }

    /// Record a fact, merging with whatever is already known. Returns whether anything changed.
    fn set(&mut self, name: &str, new: Fact) -> bool {
        if name.is_empty() || new.is_empty() || self.poisoned.contains(name) {
            return false;
        }
        if let Some(shape) = &new.shape
            && (shape.len() > MAX_INFER_RANK || shape.iter().any(|d| *d < -1))
        {
            return false;
        }
        let Some(old) = self.facts.get(name) else {
            self.facts.insert(name.to_string(), new);
            return true;
        };
        let mut merged = old.clone();
        let mut changed = false;

        match (&merged.shape, &new.shape) {
            (None, Some(s)) => {
                merged.shape = Some(s.clone());
                changed = true;
            }
            (Some(a), Some(b)) => {
                if a.len() != b.len() {
                    self.poison(name);
                    return true;
                }
                let mut out = a.clone();
                for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                    match (*x, *y) {
                        (-1, v) if v >= 0 => {
                            out[i] = v;
                            changed = true;
                        }
                        (u, v) if u >= 0 && v >= 0 && u != v => {
                            self.poison(name);
                            return true;
                        }
                        _ => {}
                    }
                }
                merged.shape = Some(out);
            }
            _ => {}
        }

        match (&merged.ints, &new.ints) {
            (None, Some(v)) => {
                merged.ints = Some(v.clone());
                changed = true;
            }
            (Some(a), Some(b)) => {
                if a.len() != b.len() {
                    self.poison(name);
                    return true;
                }
                let mut out = a.clone();
                for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                    match (*x, *y) {
                        (None, Some(v)) => {
                            out[i] = Some(v);
                            changed = true;
                        }
                        (Some(u), Some(v)) if u != v => {
                            self.poison(name);
                            return true;
                        }
                        _ => {}
                    }
                }
                merged.ints = Some(out);
            }
            _ => {}
        }

        if changed {
            self.facts.insert(name.to_string(), merged);
        }
        changed
    }

    fn poison(&mut self, name: &str) {
        if self.poisoned.insert(name.to_string()) {
            self.stats.contradictions += 1;
        }
        self.facts.remove(name);
    }

    /// Run rules to a fixed point over `nodes`.
    ///
    /// Terminates on the first sweep that proves nothing new, or at [`SWEEP_LIMIT`]. Hitting the
    /// limit is recorded, not fatal: the facts proven so far are all sound, there are merely
    /// fewer of them than a longer run would find.
    pub fn run(&mut self, nodes: &[InferNode]) -> Stats {
        for sweep in 0..SWEEP_LIMIT {
            let mut changed = false;
            for node in nodes {
                for (name, fact) in self.rules(node) {
                    changed |= self.set(&name, fact);
                }
            }
            self.stats.sweeps = sweep + 1;
            if !changed {
                return self.stats;
            }
        }
        self.stats.hit_sweep_limit = true;
        self.stats
    }

    /// Freeze the pass into the overlay the claim path reads.
    pub fn finish(mut self) -> InferredShapes {
        let mut shapes = HashMap::new();
        let mut ranks_proved = 0usize;
        let mut extents_proved = 0usize;
        for (name, fact) in &self.facts {
            let Some(shape) = &fact.shape else { continue };
            match self.declared.get(name) {
                None => {
                    ranks_proved += 1;
                    extents_proved += shape.iter().filter(|d| **d >= 0).count();
                    shapes.insert(name.clone(), shape.clone());
                }
                Some(declared) if declared.len() == shape.len() => {
                    let gained = declared
                        .iter()
                        .zip(shape.iter())
                        .filter(|(d, s)| **d < 0 && **s >= 0)
                        .count();
                    if gained > 0 {
                        extents_proved += gained;
                        shapes.insert(name.clone(), shape.clone());
                    }
                }
                Some(_) => {}
            }
        }
        self.stats.ranks_proved = ranks_proved;
        self.stats.extents_proved = extents_proved;
        InferredShapes {
            shapes,
            stats: self.stats,
        }
    }

    // ---------------------------------------------------------------------------------------
    // Rules. Each returns the facts it can *prove* for this node's outputs, and nothing else.
    // ---------------------------------------------------------------------------------------

    fn rules(&self, n: &InferNode) -> Vec<(String, Fact)> {
        if !n.is_default_domain() {
            return Vec::new();
        }
        let out0 = |f: Fact| -> Vec<(String, Fact)> {
            match n.outputs.first() {
                Some(o) if !o.is_empty() => vec![(o.clone(), f)],
                _ => Vec::new(),
            }
        };
        match n.op_type.as_str() {
            "Shape" => self.rule_shape(n).map(out0).unwrap_or_default(),
            "Size" => self.rule_size(n).map(out0).unwrap_or_default(),
            "Cast" | "CastLike" => self.rule_cast(n).map(out0).unwrap_or_default(),
            "Concat" => self.rule_concat(n).map(out0).unwrap_or_default(),
            "Unsqueeze" => self.rule_unsqueeze(n).map(out0).unwrap_or_default(),
            "Squeeze" => self.rule_squeeze(n).map(out0).unwrap_or_default(),
            "Slice" => self.rule_slice(n).map(out0).unwrap_or_default(),
            "Gather" => self.rule_gather(n).map(out0).unwrap_or_default(),
            "Reshape" => self.rule_reshape(n).map(out0).unwrap_or_default(),
            "Transpose" => self.rule_transpose(n).map(out0).unwrap_or_default(),
            "Flatten" | "Gemm" => self.rule_rank2(n).map(out0).unwrap_or_default(),
            "MatMul" => self.rule_matmul(n).map(out0).unwrap_or_default(),
            "ConstantOfShape" => self.rule_constant_of_shape(n).map(out0).unwrap_or_default(),
            "Split" => self.rule_split(n),
            op if is_shape_preserving(op) => self
                .input_shape_fact(n, 0)
                .map(Fact::of_shape)
                .map(out0)
                .unwrap_or_default(),
            op if is_broadcast_op(op) => self.rule_broadcast(n).map(out0).unwrap_or_default(),
            op if is_reduce_op(op) => self.rule_reduce(n).map(out0).unwrap_or_default(),
            _ => Vec::new(),
        }
    }

    fn input_shape_fact(&self, n: &InferNode, i: usize) -> Option<Vec<i64>> {
        Some(self.shape_of(n.input(i)?)?.to_vec())
    }

    /// `Shape(x)` — the length/value distinction, rule 1 of the module contract.
    fn rule_shape(&self, n: &InferNode) -> Option<Fact> {
        let x = n.input(0)?;
        let dims = self.shape_of(x)?;
        let rank = dims.len() as i64;

        let (start_attr, end_attr) = (n.attr_int("start"), n.attr_int("end"));
        // `start`/`end` arrived in opset 15. A node that carries them at an older `since_version`
        // is a graph this module does not understand, and guessing which semantics apply is
        // exactly the kind of guess the contract forbids.
        if (start_attr.is_some() || end_attr.is_some()) && n.since_version < 15 {
            return None;
        }
        let clamp = |v: i64| -> i64 {
            let v = if v < 0 { v + rank } else { v };
            v.clamp(0, rank)
        };
        let start = clamp(start_attr.unwrap_or(0));
        let end = clamp(end_attr.unwrap_or(rank));
        // ONNX clamps rather than errors, and an empty slice is a legal (empty) 1-D output.
        let len = (end - start).max(0);
        let ints: Vec<Option<i64>> = dims[start as usize..(start + len) as usize]
            .iter()
            .map(|d| if *d >= 0 { Some(*d) } else { None })
            .collect();
        Some(Fact {
            shape: Some(vec![len]),
            ints: Some(ints),
        })
    }

    /// `Size(x)` — a proven scalar, and its value when every extent is proven.
    fn rule_size(&self, n: &InferNode) -> Option<Fact> {
        let dims = self.shape_of(n.input(0)?)?;
        let mut total: Option<i64> = Some(1);
        for d in dims {
            total = match (total, *d) {
                (Some(t), v) if v >= 0 => t.checked_mul(v),
                _ => None,
            };
        }
        Some(Fact {
            shape: Some(vec![]),
            ints: Some(vec![total]),
        })
    }

    /// `Cast(x, to)` — rule 2 of the module contract: shape always, values only into an integer.
    fn rule_cast(&self, n: &InferNode) -> Option<Fact> {
        let x = n.input(0)?;
        let shape = self.shape_of(x).map(<[i64]>::to_vec);
        let to = match n.op_type.as_str() {
            // `CastLike` takes its destination type from input 1 rather than an attribute. The
            // shape still propagates; the values do not, because the destination is unread.
            "CastLike" => None,
            _ => n.attr_int("to"),
        };
        let ints = match (to, self.ints_of(x)) {
            (Some(ONNX_INT64), Some(v)) => Some(v.to_vec()),
            (Some(ONNX_INT32), Some(v)) => Some(
                v.iter()
                    .map(|e| e.filter(|x| i32::try_from(*x).is_ok()))
                    .collect(),
            ),
            // Every other destination — `float`, `bool`, a narrower int, an unreadable `to` —
            // keeps the shape and drops the values. The BERT chain's `Cast(to=FLOAT)` is this
            // branch, and dropping there is what makes the `Cast(to=INT32)` after it honest.
            _ => None,
        };
        if shape.is_none() && ints.is_none() {
            return None;
        }
        Some(Fact { shape, ints })
    }

    /// `Concat(inputs, axis)` — validates every input, not just the first.
    fn rule_concat(&self, n: &InferNode) -> Option<Fact> {
        let names: Vec<&str> = n.inputs.iter().map(String::as_str).collect();
        if names.is_empty() || names.iter().any(|s| s.is_empty()) {
            return None;
        }
        let axis_attr = n.attr_int("axis")?;
        let mut shapes: Vec<&[i64]> = Vec::with_capacity(names.len());
        for name in &names {
            shapes.push(self.shape_of(name)?);
        }
        let rank = shapes[0].len();
        // ONNX requires all inputs to have the same rank, and `Concat` has no rank-0 form.
        if rank == 0 || shapes.iter().any(|s| s.len() != rank) {
            return None;
        }
        let axis = norm_axis(axis_attr, rank)?;

        let mut out = vec![-1i64; rank];
        for (a, o) in out.iter_mut().enumerate() {
            if a == axis {
                let mut sum: Option<i64> = Some(0);
                for s in &shapes {
                    sum = match (sum, s[a]) {
                        (Some(t), v) if v >= 0 => t.checked_add(v),
                        _ => None,
                    };
                }
                *o = sum.unwrap_or(-1);
            } else {
                // Off-axis extents must agree. Known values that disagree make the node invalid,
                // so nothing is proven for it at all.
                let mut known: Option<i64> = None;
                for s in &shapes {
                    if s[a] >= 0 {
                        match known {
                            None => known = Some(s[a]),
                            Some(k) if k != s[a] => return None,
                            Some(_) => {}
                        }
                    }
                }
                *o = known.unwrap_or(-1);
            }
        }

        // Values: only the 1-D, axis-0 case, which is the shape-vector concatenation the whole
        // chain is built out of. An input whose length is unproven makes the *output* length
        // unproven too, and then there is nothing to say about elements either.
        let ints = if rank == 1 && axis == 0 && out[0] >= 0 {
            let mut acc: Vec<Option<i64>> = Vec::new();
            for (name, s) in names.iter().zip(shapes.iter()) {
                let len = usize::try_from(s[0]).ok()?;
                match self.ints_of(name) {
                    Some(v) if v.len() == len => acc.extend_from_slice(v),
                    _ => acc.extend(std::iter::repeat_n(None, len)),
                }
            }
            if acc.len() > MAX_INFER_INTS {
                None
            } else {
                Some(acc)
            }
        } else {
            None
        };

        Some(Fact {
            shape: Some(out),
            ints,
        })
    }

    /// Axes for an op that moved them from an attribute to input 1 at some opset.
    fn axes_of(&self, n: &InferNode, input_index: usize) -> Option<Vec<i64>> {
        if let Some(a) = n.attr_ints("axes") {
            return Some(a.to_vec());
        }
        let name = n.input(input_index)?;
        let ints = self.ints_of(name)?;
        ints.iter().copied().collect::<Option<Vec<i64>>>()
    }

    /// `Unsqueeze(x, axes)` — inserted extents are literal 1s, so the rank is exact.
    fn rule_unsqueeze(&self, n: &InferNode) -> Option<Fact> {
        let x = n.input(0)?;
        let dims = self.shape_of(x)?;
        let axes = self.axes_of(n, 1)?;
        if axes.is_empty() {
            return None;
        }
        let out_rank = dims.len().checked_add(axes.len())?;
        if out_rank > MAX_INFER_RANK {
            return None;
        }
        let mut norm: Vec<usize> = Vec::with_capacity(axes.len());
        for a in &axes {
            let a = norm_axis(*a, out_rank)?;
            if norm.contains(&a) {
                return None; // duplicate axis: the node is invalid, prove nothing
            }
            norm.push(a);
        }
        let mut out = vec![i64::MIN; out_rank];
        for a in &norm {
            out[*a] = 1;
        }
        let mut src = dims.iter();
        for slot in out.iter_mut() {
            if *slot == i64::MIN {
                *slot = *src.next()?;
            }
        }
        // Element order and count are unchanged by `Unsqueeze`, so flat values carry over as-is.
        Some(Fact {
            shape: Some(out),
            ints: self.ints_of(x).map(<[Option<i64>]>::to_vec),
        })
    }

    /// `Squeeze(x, axes)` — removes only extents ONNX guarantees are 1.
    fn rule_squeeze(&self, n: &InferNode) -> Option<Fact> {
        let x = n.input(0)?;
        let dims = self.shape_of(x)?;
        let rank = dims.len();
        match self.axes_of(n, 1) {
            Some(axes) => {
                let mut drop: Vec<usize> = Vec::with_capacity(axes.len());
                for a in &axes {
                    let a = norm_axis(*a, rank)?;
                    // A named axis whose extent is *proven* to be something other than 1 makes
                    // the node invalid. Proving nothing is the honest answer; a rank derived
                    // from an impossible node is not a fact.
                    if dims[a] >= 0 && dims[a] != 1 {
                        return None;
                    }
                    if drop.contains(&a) {
                        return None;
                    }
                    drop.push(a);
                }
                let out: Vec<i64> = dims
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !drop.contains(i))
                    .map(|(_, d)| *d)
                    .collect();
                let all_known_one = drop.iter().all(|a| dims[*a] == 1);
                Some(Fact {
                    shape: Some(out),
                    ints: if all_known_one {
                        self.ints_of(x).map(<[Option<i64>]>::to_vec)
                    } else {
                        None
                    },
                })
            }
            // No axes: every extent equal to 1 is removed, which needs every extent proven.
            None if n.inputs.len() < 2 || n.input(1).is_none() => {
                if dims.iter().any(|d| *d < 0) {
                    return None;
                }
                let out: Vec<i64> = dims.iter().copied().filter(|d| *d != 1).collect();
                Some(Fact {
                    shape: Some(out),
                    ints: self.ints_of(x).map(<[Option<i64>]>::to_vec),
                })
            }
            None => None,
        }
    }

    /// Constant `ints` for an input, when this pass proved every element.
    fn const_ints(&self, n: &InferNode, i: usize) -> Option<Vec<i64>> {
        let ints = self.ints_of(n.input(i)?)?;
        ints.iter().copied().collect::<Option<Vec<i64>>>()
    }

    /// `Slice` — rank is preserved unconditionally; extents need every parameter proven.
    fn rule_slice(&self, n: &InferNode) -> Option<Fact> {
        let x = n.input(0)?;
        let dims = self.shape_of(x)?.to_vec();
        let rank = dims.len();

        // opset 1 carries starts/ends/axes as attributes; opset 10 moved them to inputs and added
        // `steps`. Both spellings are read, and neither is assumed when absent.
        let starts = n
            .attr_ints("starts")
            .map(<[i64]>::to_vec)
            .or_else(|| self.const_ints(n, 1));
        let ends = n
            .attr_ints("ends")
            .map(<[i64]>::to_vec)
            .or_else(|| self.const_ints(n, 2));
        let axes = n
            .attr_ints("axes")
            .map(<[i64]>::to_vec)
            .or_else(|| self.const_ints(n, 3));
        let steps = if n.input(4).is_some() {
            self.const_ints(n, 4)
        } else {
            Some(vec![])
        };

        // Rank alone is already a fact worth having: `Slice` never adds or removes an axis.
        let unknown = Fact::of_shape(vec![-1; rank]);
        let (Some(starts), Some(ends)) = (starts, ends) else {
            return Some(unknown);
        };
        if starts.len() != ends.len() {
            return Some(unknown);
        }
        let axes = match axes {
            Some(a) if a.len() == starts.len() => a,
            Some(_) => return Some(unknown),
            None => (0..starts.len() as i64).collect(),
        };
        let steps = match steps {
            Some(s) if s.is_empty() => vec![1; starts.len()],
            Some(s) if s.len() == starts.len() => s,
            _ => return Some(unknown),
        };

        let mut out = dims.clone();
        let mut sliced_axes: Vec<usize> = Vec::new();
        for (i, a) in axes.iter().enumerate() {
            let Some(a) = norm_axis(*a, rank) else {
                return Some(unknown);
            };
            if sliced_axes.contains(&a) {
                return Some(unknown);
            }
            sliced_axes.push(a);
            let step = steps[i];
            if step == 0 {
                return Some(unknown);
            }
            let extent = dims[a];
            if extent < 0 {
                out[a] = -1;
                continue;
            }
            let clamp_start = |v: i64| -> i64 {
                let v = if v < 0 { v + extent } else { v };
                if step > 0 {
                    v.clamp(0, extent)
                } else {
                    v.clamp(0, extent - 1)
                }
            };
            let clamp_end = |v: i64| -> i64 {
                let v = if v < 0 { v + extent } else { v };
                if step > 0 {
                    v.clamp(0, extent)
                } else {
                    v.clamp(-1, extent)
                }
            };
            let s = clamp_start(starts[i]);
            let e = clamp_end(ends[i]);
            let len = if step > 0 {
                ceil_div_positive(e - s, step)
            } else {
                ceil_div_positive(s - e, -step)
            };
            out[a] = len;
        }

        // Values: the contiguous, positive-step, 1-D case only. That is the shape-vector slice
        // the chain uses; anything else keeps the rank and drops the elements.
        let ints = if rank == 1 && sliced_axes == [0] && steps[0] == 1 && out[0] >= 0 {
            self.ints_of(x).and_then(|v| {
                let extent = dims[0];
                if extent < 0 {
                    return None;
                }
                let s = usize::try_from(if starts[0] < 0 {
                    (starts[0] + extent).clamp(0, extent)
                } else {
                    starts[0].clamp(0, extent)
                })
                .ok()?;
                let len = usize::try_from(out[0]).ok()?;
                v.get(s..s + len).map(<[Option<i64>]>::to_vec)
            })
        } else {
            None
        };

        Some(Fact {
            shape: Some(out),
            ints,
        })
    }

    /// `Gather(data, indices, axis)` — the rank arithmetic is exact even when nothing else is.
    fn rule_gather(&self, n: &InferNode) -> Option<Fact> {
        let data = n.input(0)?;
        let idx = n.input(1)?;
        let d = self.shape_of(data)?.to_vec();
        let i = self.shape_of(idx)?.to_vec();
        if d.is_empty() {
            return None; // `Gather` has no rank-0 `data` form
        }
        let axis = norm_axis(n.attr_int("axis").unwrap_or(0), d.len())?;
        let out_rank = d.len() + i.len() - 1;
        if out_rank > MAX_INFER_RANK {
            return None;
        }
        let mut out: Vec<i64> = Vec::with_capacity(out_rank);
        out.extend_from_slice(&d[..axis]);
        out.extend_from_slice(&i);
        out.extend_from_slice(&d[axis + 1..]);

        // A scalar index into a 1-D proven-value vector is the `Shape`→`Gather` idiom; its result
        // is one proven element. Anything else keeps the shape only.
        let ints = if d.len() == 1 && i.is_empty() {
            match (self.ints_of(data), self.ints_of(idx)) {
                (Some(vals), Some([Some(k)])) => {
                    let len = vals.len() as i64;
                    let k = if *k < 0 { *k + len } else { *k };
                    usize::try_from(k)
                        .ok()
                        .and_then(|k| vals.get(k))
                        .map(|v| vec![*v])
                }
                _ => None,
            }
        } else {
            None
        };

        Some(Fact {
            shape: Some(out),
            ints,
        })
    }

    /// `Reshape(data, shape)` — the output rank is the shape operand's **length**, not its value.
    ///
    /// This is the whole point of the module on a transformer: the length is provable from the
    /// chain even when every element of it is a runtime extent.
    fn rule_reshape(&self, n: &InferNode) -> Option<Fact> {
        let target = n.input(1)?;
        let tdims = self.shape_of(target)?;
        // The shape operand is 1-D by ONNX definition; a different rank means we are reading the
        // wrong thing and must say nothing.
        if tdims.len() != 1 || tdims[0] < 0 {
            return None;
        }
        let out_rank = usize::try_from(tdims[0]).ok()?;
        if out_rank > MAX_INFER_RANK {
            return None;
        }
        let allowzero = n.attr_int("allowzero").unwrap_or(0) != 0;
        let vals = self.ints_of(target).map(<[Option<i64>]>::to_vec);
        let in_dims = n
            .input(0)
            .and_then(|d| self.shape_of(d))
            .map(<[i64]>::to_vec);

        let mut out = vec![-1i64; out_rank];
        if let Some(vals) = &vals {
            if vals.len() != out_rank {
                return Some(Fact::of_shape(out));
            }
            for (i, v) in vals.iter().enumerate() {
                out[i] = match v {
                    Some(x) if *x > 0 => *x,
                    Some(0) if !allowzero => {
                        // `0` copies the input extent at the same index (opset ≥ 5 default).
                        match &in_dims {
                            Some(d) if i < d.len() => d[i],
                            _ => -1,
                        }
                    }
                    Some(0) => 0,
                    // `-1` is the free axis; conservation fixes it only when everything else is
                    // proven, and `resolve_out_dims` in `ops::shape` does that arithmetic at
                    // translate time from strictly better information than this pass has.
                    _ => -1,
                };
            }
        }
        Some(Fact::of_shape(out))
    }

    /// `Transpose(x, perm)`.
    fn rule_transpose(&self, n: &InferNode) -> Option<Fact> {
        let dims = self.input_shape_fact(n, 0)?;
        let rank = dims.len();
        let perm: Vec<i64> = match n.attr_ints("perm") {
            Some(p) => p.to_vec(),
            // Default is the reversal.
            None => (0..rank as i64).rev().collect(),
        };
        if perm.len() != rank {
            return None;
        }
        let mut seen = vec![false; rank];
        let mut out = vec![-1i64; rank];
        for (i, p) in perm.iter().enumerate() {
            let p = norm_axis(*p, rank)?;
            if seen[p] {
                return None;
            }
            seen[p] = true;
            out[i] = dims[p];
        }
        Some(Fact::of_shape(out))
    }

    /// `Flatten` and `Gemm` — rank 2 by definition, extents left to the ops themselves.
    fn rule_rank2(&self, n: &InferNode) -> Option<Fact> {
        // Only claim the rank once something is known about the node at all; a rank-2 fact
        // invented for a node whose inputs are unknown is still sound, but it is also the kind
        // of fact that hides a missing input from a reader of the overlay.
        self.input_shape_fact(n, 0)?;
        Some(Fact::of_shape(vec![-1, -1]))
    }

    /// `MatMul` — NumPy `matmul` rank rules, including both vector-promotion cases.
    fn rule_matmul(&self, n: &InferNode) -> Option<Fact> {
        let a = self.input_shape_fact(n, 0)?;
        let b = self.input_shape_fact(n, 1)?;
        if a.is_empty() || b.is_empty() {
            return None; // no rank-0 operand exists for `MatMul`
        }
        let out = match (a.len(), b.len()) {
            (1, 1) => vec![],
            (1, _) => {
                let mut o = broadcast_dims(&b[..b.len() - 2], &[])?;
                o.push(b[b.len() - 1]);
                o
            }
            (_, 1) => {
                let mut o = broadcast_dims(&a[..a.len() - 2], &[])?;
                o.push(a[a.len() - 2]);
                o
            }
            _ => {
                let mut o = broadcast_dims(&a[..a.len() - 2], &b[..b.len() - 2])?;
                o.push(a[a.len() - 2]);
                o.push(b[b.len() - 1]);
                o
            }
        };
        if out.len() > MAX_INFER_RANK {
            return None;
        }
        Some(Fact::of_shape(out))
    }

    /// `ConstantOfShape(shape)` — rank from the operand's length, extents from its values.
    fn rule_constant_of_shape(&self, n: &InferNode) -> Option<Fact> {
        let s = n.input(0)?;
        let dims = self.shape_of(s)?;
        if dims.len() != 1 || dims[0] < 0 {
            return None;
        }
        let rank = usize::try_from(dims[0]).ok()?;
        if rank > MAX_INFER_RANK {
            return None;
        }
        let mut out = vec![-1i64; rank];
        if let Some(vals) = self.ints_of(s)
            && vals.len() == rank
        {
            for (o, v) in out.iter_mut().zip(vals.iter()) {
                if let Some(x) = v
                    && *x >= 0
                {
                    *o = *x;
                }
            }
        }
        Some(Fact::of_shape(out))
    }

    /// `Split` — rank and every off-axis extent are preserved; the split axis needs the sizes.
    fn rule_split(&self, n: &InferNode) -> Vec<(String, Fact)> {
        let Some(dims) = self.input_shape_fact(n, 0) else {
            return Vec::new();
        };
        let rank = dims.len();
        let Some(axis) = norm_axis(n.attr_int("axis").unwrap_or(0), rank) else {
            return Vec::new();
        };
        let sizes: Option<Vec<i64>> = n
            .attr_ints("split")
            .map(<[i64]>::to_vec)
            .or_else(|| self.const_ints(n, 1))
            .filter(|s| s.len() == n.outputs.len());
        let equal: Option<i64> = match (dims[axis], n.outputs.len()) {
            (d, k) if d >= 0 && k > 0 && d % (k as i64) == 0 => Some(d / k as i64),
            _ => None,
        };
        let mut out = Vec::new();
        for (i, name) in n.outputs.iter().enumerate() {
            if name.is_empty() {
                continue;
            }
            let mut s = dims.clone();
            s[axis] = match (&sizes, equal) {
                (Some(v), _) => v[i],
                (None, Some(e)) => e,
                (None, None) => -1,
            };
            out.push((name.clone(), Fact::of_shape(s)));
        }
        out
    }

    /// NumPy multidirectional broadcasting over every present input.
    fn rule_broadcast(&self, n: &InferNode) -> Option<Fact> {
        let mut acc: Option<Vec<i64>> = None;
        let mut any = false;
        for i in 0..n.inputs.len() {
            let Some(name) = n.input(i) else { continue };
            let dims = self.shape_of(name)?;
            any = true;
            acc = Some(match acc {
                None => dims.to_vec(),
                Some(a) => broadcast_dims(&a, dims)?,
            });
        }
        if !any {
            return None;
        }
        acc.map(Fact::of_shape)
    }

    /// `Reduce*` — rank from `axes` and `keepdims`, both of which must be readable.
    fn rule_reduce(&self, n: &InferNode) -> Option<Fact> {
        let dims = self.input_shape_fact(n, 0)?;
        let rank = dims.len();
        let keepdims = n.attr_int("keepdims").unwrap_or(1) != 0;
        let axes = match self.axes_of(n, 1) {
            Some(a) => a,
            // No axes at all means "reduce every axis" — but only when the op's own default says
            // so. `noop_with_empty_axes=1` (opset 18) inverts that, so an unreadable axes list
            // with that attribute set proves nothing.
            None if n.attr_int("noop_with_empty_axes").unwrap_or(0) == 0 => {
                (0..rank as i64).collect()
            }
            None => return None,
        };
        let mut drop: Vec<usize> = Vec::with_capacity(axes.len());
        for a in &axes {
            let a = norm_axis(*a, rank)?;
            if drop.contains(&a) {
                return None;
            }
            drop.push(a);
        }
        let out: Vec<i64> = if keepdims {
            dims.iter()
                .enumerate()
                .map(|(i, d)| if drop.contains(&i) { 1 } else { *d })
                .collect()
        } else {
            dims.iter()
                .enumerate()
                .filter(|(i, _)| !drop.contains(i))
                .map(|(_, d)| *d)
                .collect()
        };
        Some(Fact::of_shape(out))
    }
}

/// Normalise a possibly-negative axis against a rank, or `None` when it is out of range.
///
/// Out of range is a **decline**, never a clamp: ONNX defines `[-r, r-1]` and a value outside it
/// is a malformed graph, not a request for the nearest legal axis.
fn norm_axis(axis: i64, rank: usize) -> Option<usize> {
    let r = i64::try_from(rank).ok()?;
    if r == 0 {
        return None; // a rank-0 tensor has no axes at all
    }
    let a = if axis < 0 { axis.checked_add(r)? } else { axis };
    if a < 0 || a >= r {
        return None;
    }
    usize::try_from(a).ok()
}

/// `ceil(n / d)` for `d > 0`, clamped at zero, without the unstable signed `div_ceil`.
fn ceil_div_positive(n: i64, d: i64) -> i64 {
    debug_assert!(d > 0);
    if n <= 0 { 0 } else { (n - 1) / d + 1 }
}

/// NumPy multidirectional broadcasting over two proven shapes.
///
/// The interesting case is a known extent against an unknown one: if the known extent is not 1,
/// the result **must** be that extent (the other side is either 1 or equal), which is a fact.
/// If the known extent *is* 1, the result is whatever the unknown side turns out to be, which is
/// not.
fn broadcast_dims(a: &[i64], b: &[i64]) -> Option<Vec<i64>> {
    let rank = a.len().max(b.len());
    if rank > MAX_INFER_RANK {
        return None;
    }
    let mut out = vec![-1i64; rank];
    for i in 0..rank {
        let x = if i + a.len() >= rank {
            a[i + a.len() - rank]
        } else {
            1
        };
        let y = if i + b.len() >= rank {
            b[i + b.len() - rank]
        } else {
            1
        };
        out[i] = match (x, y) {
            (u, v) if u >= 0 && v >= 0 => {
                if u == v {
                    u
                } else if u == 1 {
                    v
                } else if v == 1 {
                    u
                } else {
                    return None; // not broadcastable: the node is invalid, prove nothing
                }
            }
            (u, _) if u > 1 => u,
            (_, v) if v > 1 => v,
            _ => -1,
        };
    }
    Some(out)
}

/// Ops whose output shape is exactly input 0's shape.
///
/// Every entry is a unary elementwise op or an op ONNX defines as shape-preserving. Membership is
/// a claim about the ONNX spec, so it is a list rather than a heuristic.
fn is_shape_preserving(op: &str) -> bool {
    matches!(
        op,
        "Abs"
            | "Acos"
            | "Acosh"
            | "Asin"
            | "Asinh"
            | "Atan"
            | "Atanh"
            | "BitwiseNot"
            | "Ceil"
            | "Celu"
            | "Cos"
            | "Cosh"
            | "Elu"
            | "Erf"
            | "Exp"
            | "Floor"
            | "Gelu"
            | "HardSigmoid"
            | "HardSwish"
            | "Identity"
            | "IsInf"
            | "IsNaN"
            | "LeakyRelu"
            | "Log"
            | "LogSoftmax"
            | "Mish"
            | "Neg"
            | "Not"
            | "Reciprocal"
            | "Relu"
            | "Round"
            | "Selu"
            | "Shrink"
            | "Sigmoid"
            | "Sign"
            | "Sin"
            | "Sinh"
            | "Softmax"
            | "Softplus"
            | "Softsign"
            | "Sqrt"
            | "Tan"
            | "Tanh"
            | "ThresholdedRelu"
            | "LayerNormalization"
            | "SimplifiedLayerNormalization"
    )
}

/// Ops whose output shape is the multidirectional broadcast of all their inputs.
fn is_broadcast_op(op: &str) -> bool {
    matches!(
        op,
        "Add"
            | "And"
            | "BitwiseAnd"
            | "BitwiseOr"
            | "BitwiseXor"
            | "Div"
            | "Equal"
            | "Greater"
            | "GreaterOrEqual"
            | "Less"
            | "LessOrEqual"
            | "Max"
            | "Mean"
            | "Min"
            | "Mod"
            | "Mul"
            | "Or"
            | "Pow"
            | "Sub"
            | "Sum"
            | "Where"
            | "Xor"
    )
}

/// Reduction ops sharing the `axes`/`keepdims` rank rule.
fn is_reduce_op(op: &str) -> bool {
    matches!(
        op,
        "ReduceL1"
            | "ReduceL2"
            | "ReduceLogSum"
            | "ReduceLogSumExp"
            | "ReduceMax"
            | "ReduceMean"
            | "ReduceMin"
            | "ReduceProd"
            | "ReduceSum"
            | "ReduceSumSquare"
    )
}

/// The attributes the rule for `op_type` reads, and how to read each one.
///
/// `ep.rs` walks this list per node so the ORT accessor choice (`attr_int` vs `attr_ints`) lives
/// with the rule that needs the value, not in the boundary layer.
pub fn wanted_attrs(op_type: &str) -> &'static [(&'static str, AttrKind)] {
    match op_type {
        "Shape" => &[("start", AttrKind::Int), ("end", AttrKind::Int)],
        "Cast" => &[("to", AttrKind::Int)],
        "Concat" | "Gather" => &[("axis", AttrKind::Int)],
        "Unsqueeze" | "Squeeze" => &[("axes", AttrKind::Ints)],
        "Slice" => &[
            ("starts", AttrKind::Ints),
            ("ends", AttrKind::Ints),
            ("axes", AttrKind::Ints),
        ],
        "Reshape" => &[("allowzero", AttrKind::Int)],
        "Transpose" => &[("perm", AttrKind::Ints)],
        "Split" => &[("axis", AttrKind::Int), ("split", AttrKind::Ints)],
        op if is_reduce_op(op) => &[
            ("axes", AttrKind::Ints),
            ("keepdims", AttrKind::Int),
            ("noop_with_empty_axes", AttrKind::Int),
        ],
        _ => &[],
    }
}

/// Whether any rule at all exists for this op — used by the caller to skip attribute reads.
pub fn has_rule(op_type: &str) -> bool {
    matches!(
        op_type,
        "Shape"
            | "Size"
            | "Cast"
            | "CastLike"
            | "Concat"
            | "Unsqueeze"
            | "Squeeze"
            | "Slice"
            | "Gather"
            | "Reshape"
            | "Transpose"
            | "Flatten"
            | "Gemm"
            | "MatMul"
            | "ConstantOfShape"
            | "Split"
    ) || is_shape_preserving(op_type)
        || is_broadcast_op(op_type)
        || is_reduce_op(op_type)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(op: &str, inputs: &[&str], outputs: &[&str]) -> InferNode {
        InferNode {
            op_type: op.to_string(),
            domain: String::new(),
            since_version: 13,
            inputs: inputs.iter().map(|s| (*s).to_string()).collect(),
            outputs: outputs.iter().map(|s| (*s).to_string()).collect(),
            attrs: Vec::new(),
        }
    }

    fn with_int(mut n: InferNode, name: &'static str, v: i64) -> InferNode {
        n.attrs.push((name, AttrValue::Int(v)));
        n
    }

    fn with_ints(mut n: InferNode, name: &'static str, v: &[i64]) -> InferNode {
        n.attrs.push((name, AttrValue::Ints(v.to_vec())));
        n
    }

    // -- rule 3: what ORT's rank-0 reading is NOT ------------------------------------------

    #[test]
    fn ort_rank0_reading_is_not_seeded_as_a_scalar() {
        // The 2026-08-04 vacuity defect, reproduced deliberately: `classify_shapes` read
        // `Some([])` as "static" because "all dims non-negative" is vacuously true over an empty
        // list. This module must not treat that reading as a fact at all.
        let mut inf = Inference::new();
        inf.declare("x", Some(&[]));
        assert!(inf.fact("x").is_none(), "rank-0 from ORT is an ambiguity");
        // A read initializer is a different matter: its rank is stored data.
        inf.constant("k", &[], Some(vec![7]));
        assert_eq!(inf.fact("k").unwrap().rank(), Some(0));
    }

    #[test]
    fn declared_symbolic_dims_are_normalised_and_kept() {
        let mut inf = Inference::new();
        inf.declare("x", Some(&[-9, 768]));
        assert_eq!(inf.shape_of("x"), Some(&[-1, 768][..]));
    }

    // -- Shape ------------------------------------------------------------------------------

    #[test]
    fn shape_output_length_is_the_input_rank_even_when_extents_are_symbolic() {
        let mut inf = Inference::new();
        inf.declare("x", Some(&[-1, -1, 768]));
        inf.run(&[node("Shape", &["x"], &["s"])]);
        assert_eq!(inf.shape_of("s"), Some(&[3][..]));
        assert_eq!(inf.ints_of("s"), Some(&[None, None, Some(768)][..]));
    }

    #[test]
    fn shape_start_end_are_honoured_at_opset_15_and_refused_below_it() {
        let mut inf = Inference::new();
        inf.declare("x", Some(&[2, 3, 4, 5]));
        let mut n = with_int(node("Shape", &["x"], &["s"]), "start", 1);
        n = with_int(n, "end", 3);
        n.since_version = 15;
        inf.run(&[n.clone()]);
        assert_eq!(inf.shape_of("s"), Some(&[2][..]));
        assert_eq!(inf.ints_of("s"), Some(&[Some(3), Some(4)][..]));

        // The same attributes at an opset that did not define them prove nothing.
        let mut old = Inference::new();
        old.declare("x", Some(&[2, 3, 4, 5]));
        n.since_version = 13;
        old.run(&[n]);
        assert!(old.fact("s").is_none());
    }

    #[test]
    fn shape_negative_and_out_of_range_start_end_clamp_per_spec() {
        let mut inf = Inference::new();
        inf.declare("x", Some(&[2, 3, 4]));
        let mut n = with_int(node("Shape", &["x"], &["s"]), "start", -2);
        n = with_int(n, "end", 99);
        n.since_version = 19;
        inf.run(&[n]);
        assert_eq!(inf.shape_of("s"), Some(&[2][..]));

        // end < start is an empty output, not a negative length.
        let mut inf2 = Inference::new();
        inf2.declare("x", Some(&[2, 3, 4]));
        let mut n2 = with_int(node("Shape", &["x"], &["s"]), "start", 2);
        n2 = with_int(n2, "end", 1);
        n2.since_version = 19;
        inf2.run(&[n2]);
        assert_eq!(inf2.shape_of("s"), Some(&[0][..]));
    }

    #[test]
    fn shape_of_an_unranked_input_proves_nothing() {
        let mut inf = Inference::new();
        inf.run(&[node("Shape", &["x"], &["s"])]);
        assert!(inf.fact("s").is_none());
    }

    // -- Cast -------------------------------------------------------------------------------

    #[test]
    fn cast_preserves_length_but_a_float_hop_drops_values() {
        let mut inf = Inference::new();
        inf.declare("x", Some(&[2, 256]));
        inf.run(&[
            node("Shape", &["x"], &["s"]),
            with_int(node("Cast", &["s"], &["f"]), "to", 1), // to FLOAT
            with_int(node("Cast", &["f"], &["i"]), "to", 6), // back to INT32
        ]);
        assert_eq!(inf.ints_of("s"), Some(&[Some(2), Some(256)][..]));
        assert_eq!(inf.shape_of("f"), Some(&[2][..]), "length survives");
        assert!(inf.ints_of("f").is_none(), "values do not survive float");
        assert_eq!(inf.shape_of("i"), Some(&[2][..]));
        assert!(
            inf.ints_of("i").is_none(),
            "a value dropped upstream cannot reappear downstream"
        );
    }

    #[test]
    fn cast_to_int32_drops_only_the_elements_that_do_not_fit() {
        let mut inf = Inference::new();
        inf.constant("k", &[2], Some(vec![5, i64::from(i32::MAX) + 1]));
        inf.run(&[with_int(node("Cast", &["k"], &["c"]), "to", 6)]);
        assert_eq!(inf.ints_of("c"), Some(&[Some(5), None][..]));
    }

    #[test]
    fn castlike_keeps_shape_and_drops_values() {
        let mut inf = Inference::new();
        inf.constant("k", &[2], Some(vec![1, 2]));
        inf.declare("t", Some(&[2]));
        inf.run(&[node("CastLike", &["k", "t"], &["c"])]);
        assert_eq!(inf.shape_of("c"), Some(&[2][..]));
        assert!(inf.ints_of("c").is_none());
    }

    // -- Concat -----------------------------------------------------------------------------

    #[test]
    fn concat_combines_known_lengths_and_leaves_unknown_elements_unknown() {
        let mut inf = Inference::new();
        inf.declare("a", Some(&[1]));
        inf.constant("b", &[2], Some(vec![1, 256]));
        inf.run(&[with_int(node("Concat", &["a", "b"], &["c"]), "axis", 0)]);
        assert_eq!(inf.shape_of("c"), Some(&[3][..]));
        assert_eq!(inf.ints_of("c"), Some(&[None, Some(1), Some(256)][..]));
    }

    #[test]
    fn concat_with_one_unknown_input_length_proves_no_length() {
        let mut inf = Inference::new();
        inf.declare("a", Some(&[-1]));
        inf.constant("b", &[2], Some(vec![1, 256]));
        inf.run(&[with_int(node("Concat", &["a", "b"], &["c"]), "axis", 0)]);
        assert_eq!(inf.shape_of("c"), Some(&[-1][..]), "rank yes, length no");
        assert!(inf.ints_of("c").is_none());
    }

    #[test]
    fn concat_validates_every_input_not_just_the_first() {
        let mut inf = Inference::new();
        inf.declare("a", Some(&[2, 3]));
        // Second input is unranked: nothing is proven, even though the first input is fine.
        inf.run(&[with_int(node("Concat", &["a", "b"], &["c"]), "axis", 0)]);
        assert!(inf.fact("c").is_none());
    }

    #[test]
    fn concat_declines_rank_mismatch_and_out_of_range_axis() {
        let mut mismatch = Inference::new();
        mismatch.declare("a", Some(&[2, 3]));
        mismatch.declare("b", Some(&[3]));
        mismatch.run(&[with_int(node("Concat", &["a", "b"], &["c"]), "axis", 0)]);
        assert!(mismatch.fact("c").is_none());

        for axis in [2i64, -3, i64::MIN] {
            let mut inf = Inference::new();
            inf.declare("a", Some(&[2, 3]));
            inf.declare("b", Some(&[2, 3]));
            inf.run(&[with_int(node("Concat", &["a", "b"], &["c"]), "axis", axis)]);
            assert!(inf.fact("c").is_none(), "axis {axis} must not be clamped");
        }
    }

    #[test]
    fn concat_negative_axis_normalises() {
        let mut inf = Inference::new();
        inf.declare("a", Some(&[2, 3]));
        inf.declare("b", Some(&[2, 4]));
        inf.run(&[with_int(node("Concat", &["a", "b"], &["c"]), "axis", -1)]);
        assert_eq!(inf.shape_of("c"), Some(&[2, 7][..]));
    }

    #[test]
    fn concat_off_axis_disagreement_proves_nothing() {
        let mut inf = Inference::new();
        inf.declare("a", Some(&[2, 3]));
        inf.declare("b", Some(&[5, 4]));
        inf.run(&[with_int(node("Concat", &["a", "b"], &["c"]), "axis", 1)]);
        assert!(inf.fact("c").is_none());
    }

    #[test]
    fn concat_overflow_on_the_axis_extent_yields_unknown_not_a_wrap() {
        let mut inf = Inference::new();
        inf.declare("a", Some(&[i64::MAX]));
        inf.declare("b", Some(&[2]));
        inf.run(&[with_int(node("Concat", &["a", "b"], &["c"]), "axis", 0)]);
        assert_eq!(inf.shape_of("c"), Some(&[-1][..]));
    }

    #[test]
    fn concat_with_no_inputs_proves_nothing() {
        let mut inf = Inference::new();
        inf.run(&[with_int(node("Concat", &[], &["c"]), "axis", 0)]);
        assert!(inf.fact("c").is_none());
    }

    // -- the real BERT chain ----------------------------------------------------------------

    /// The exact chain measured on BERT-SQuAD-12, node for node.
    fn bert_chain() -> Vec<InferNode> {
        vec![
            node("Shape", &["input_ids"], &["shape0"]),
            with_int(node("Cast", &["shape0"], &["shape_f"]), "to", 1),
            with_ints(
                with_ints(
                    with_ints(node("Slice", &["shape_f"], &["sl"]), "starts", &[0]),
                    "ends",
                    &[1],
                ),
                "axes",
                &[0],
            ),
            with_ints(node("Squeeze", &["sl"], &["sq"]), "axes", &[0]),
            with_int(node("Cast", &["sq"], &["batch_i32"]), "to", 6),
            with_ints(node("Unsqueeze", &["batch_i32"], &["u0"]), "axes", &[0]),
            with_ints(node("Unsqueeze", &["one"], &["u1"]), "axes", &[0]),
            with_ints(node("Unsqueeze", &["h"], &["u2"]), "axes", &[0]),
            with_int(
                node("Concat", &["u0", "u1", "u2"], &["target_i32"]),
                "axis",
                0,
            ),
            with_int(node("Cast", &["target_i32"], &["target"]), "to", 7),
            node("Reshape", &["data", "target"], &["r"]),
            node("MatMul", &["r", "w"], &["mm"]),
        ]
    }

    #[test]
    fn the_measured_bert_chain_yields_a_rank_and_the_static_inner_extent() {
        let mut inf = Inference::new();
        // Every node in this chain is opset 12 or below on the real model: axes are attributes.
        inf.declare("input_ids", Some(&[-1, -1]));
        inf.constant("one", &[], Some(vec![1]));
        inf.constant("h", &[], Some(vec![256]));
        inf.declare("data", Some(&[-1, -1, 256]));
        inf.declare("w", Some(&[256, 768]));
        let mut nodes = bert_chain();
        for n in &mut nodes {
            n.since_version = 12;
        }
        inf.run(&nodes);

        assert_eq!(inf.shape_of("shape0"), Some(&[2][..]));
        assert_eq!(inf.shape_of("sl"), Some(&[1][..]));
        assert_eq!(inf.shape_of("sq"), Some(&[][..]), "Squeeze proves rank 0");
        assert_eq!(inf.shape_of("u0"), Some(&[1][..]));
        assert_eq!(inf.shape_of("target_i32"), Some(&[3][..]));
        assert_eq!(
            inf.ints_of("target"),
            Some(&[None, Some(1), Some(256)][..]),
            "the batch element stays unknown: it went through a float Cast"
        );
        // The unlock: an output rank ORT does not have, plus the one extent that is a constant.
        assert_eq!(inf.shape_of("r"), Some(&[-1, 1, 256][..]));
        assert_eq!(inf.shape_of("mm"), Some(&[-1, 1, 768][..]));

        let overlay = inf.finish();
        assert_eq!(overlay.shape("r"), Some(&[-1, 1, 256][..]));
        assert!(overlay.stats().ranks_proved >= 6);
        assert_eq!(overlay.stats().contradictions, 0);
    }

    /// The planted control for the converse: the same chain with a symbolic inner extent must
    /// still decline, because nothing about it is proven.
    #[test]
    fn the_bert_chain_with_an_unknown_trailing_constant_proves_a_rank_and_no_extents() {
        let mut inf = Inference::new();
        inf.declare("input_ids", Some(&[-1, -1]));
        inf.declare("one", Some(&[-1])); // not a constant: rank 1, length unknown
        inf.constant("h", &[], Some(vec![256]));
        let mut nodes = bert_chain();
        for n in &mut nodes {
            n.since_version = 12;
        }
        inf.run(&nodes);
        // `Unsqueeze` of a rank-1 gives rank 2, so the `Concat` operands disagree in rank and
        // the whole target is unproven — no rank for the `Reshape` either.
        assert!(inf.fact("target").is_none());
        assert!(inf.fact("r").is_none());
    }

    // -- Squeeze / Unsqueeze ----------------------------------------------------------------

    #[test]
    fn squeeze_of_a_proven_non_unit_extent_proves_nothing() {
        let mut inf = Inference::new();
        inf.declare("x", Some(&[3, 1]));
        inf.run(&[with_ints(node("Squeeze", &["x"], &["s"]), "axes", &[0])]);
        assert!(inf.fact("s").is_none());
    }

    #[test]
    fn squeeze_without_axes_needs_every_extent() {
        let mut inf = Inference::new();
        inf.declare("x", Some(&[1, -1, 3]));
        inf.run(&[node("Squeeze", &["x"], &["s"])]);
        assert!(inf.fact("s").is_none());

        let mut ok = Inference::new();
        ok.declare("x", Some(&[1, 4, 1]));
        ok.run(&[node("Squeeze", &["x"], &["s"])]);
        assert_eq!(ok.shape_of("s"), Some(&[4][..]));
    }

    #[test]
    fn unsqueeze_reads_axes_from_input_one_at_opset_13() {
        let mut inf = Inference::new();
        inf.declare("x", Some(&[4]));
        inf.constant("ax", &[1], Some(vec![0]));
        inf.run(&[node("Unsqueeze", &["x", "ax"], &["u"])]);
        assert_eq!(inf.shape_of("u"), Some(&[1, 4][..]));
    }

    #[test]
    fn unsqueeze_duplicate_axis_proves_nothing() {
        let mut inf = Inference::new();
        inf.declare("x", Some(&[4]));
        inf.run(&[with_ints(
            node("Unsqueeze", &["x"], &["u"]),
            "axes",
            &[0, 0],
        )]);
        assert!(inf.fact("u").is_none());
    }

    #[test]
    fn unsqueeze_past_the_rank_limit_proves_nothing() {
        let mut inf = Inference::new();
        inf.declare("x", Some(&[1, 1, 1, 1, 1, 1, 1, 1]));
        inf.run(&[with_ints(node("Unsqueeze", &["x"], &["u"]), "axes", &[0])]);
        assert!(inf.fact("u").is_none());
    }

    // -- Reshape ----------------------------------------------------------------------------

    #[test]
    fn reshape_rank_comes_from_the_target_length_not_its_values() {
        let mut inf = Inference::new();
        inf.declare("t", Some(&[4]));
        inf.declare("d", Some(&[8, 8]));
        inf.run(&[node("Reshape", &["d", "t"], &["r"])]);
        assert_eq!(inf.shape_of("r"), Some(&[-1, -1, -1, -1][..]));
    }

    #[test]
    fn reshape_zero_copies_the_input_extent_unless_allowzero() {
        let mut copy = Inference::new();
        copy.constant("t", &[2], Some(vec![0, 4]));
        copy.declare("d", Some(&[7, 4]));
        copy.run(&[node("Reshape", &["d", "t"], &["r"])]);
        assert_eq!(copy.shape_of("r"), Some(&[7, 4][..]));

        let mut allow = Inference::new();
        allow.constant("t", &[2], Some(vec![0, 4]));
        allow.declare("d", Some(&[7, 4]));
        allow.run(&[with_int(
            node("Reshape", &["d", "t"], &["r"]),
            "allowzero",
            1,
        )]);
        assert_eq!(allow.shape_of("r"), Some(&[0, 4][..]));
    }

    #[test]
    fn reshape_free_axis_stays_unknown_here() {
        let mut inf = Inference::new();
        inf.constant("t", &[2], Some(vec![-1, 768]));
        inf.declare("d", Some(&[4, 768]));
        inf.run(&[node("Reshape", &["d", "t"], &["r"])]);
        assert_eq!(
            inf.shape_of("r"),
            Some(&[-1, 768][..]),
            "conservation is translate's arithmetic, not this pass's"
        );
    }

    #[test]
    fn reshape_with_an_unranked_target_proves_nothing() {
        let mut inf = Inference::new();
        inf.declare("d", Some(&[4, 768]));
        inf.run(&[node("Reshape", &["d", "t"], &["r"])]);
        assert!(inf.fact("r").is_none());
    }

    // -- broadcasting, matmul, reduce -------------------------------------------------------

    #[test]
    fn broadcast_proves_a_known_extent_against_an_unknown_one() {
        let mut inf = Inference::new();
        inf.declare("a", Some(&[-1, 768]));
        inf.declare("b", Some(&[768]));
        inf.run(&[node("Add", &["a", "b"], &["c"])]);
        assert_eq!(inf.shape_of("c"), Some(&[-1, 768][..]));
    }

    #[test]
    fn broadcast_of_one_against_unknown_stays_unknown() {
        let mut inf = Inference::new();
        inf.declare("a", Some(&[1]));
        inf.declare("b", Some(&[-1]));
        inf.run(&[node("Mul", &["a", "b"], &["c"])]);
        assert_eq!(inf.shape_of("c"), Some(&[-1][..]));
    }

    #[test]
    fn incompatible_broadcast_proves_nothing() {
        let mut inf = Inference::new();
        inf.declare("a", Some(&[3]));
        inf.declare("b", Some(&[4]));
        inf.run(&[node("Add", &["a", "b"], &["c"])]);
        assert!(inf.fact("c").is_none());
    }

    #[test]
    fn matmul_rank_rules_cover_the_vector_promotions() {
        let cases: &[(&[i64], &[i64], &[i64])] = &[
            (&[2, 3], &[3, 4], &[2, 4]),
            (&[5, 2, 3], &[3, 4], &[5, 2, 4]),
            (&[3], &[3, 4], &[4]),
            (&[2, 3], &[3], &[2]),
        ];
        for (a, b, want) in cases {
            let mut inf = Inference::new();
            inf.declare("a", Some(a));
            inf.declare("b", Some(b));
            inf.run(&[node("MatMul", &["a", "b"], &["m"])]);
            assert_eq!(inf.shape_of("m"), Some(*want), "{a:?} x {b:?}");
        }
    }

    #[test]
    fn reduce_keepdims_and_negative_axes() {
        let mut keep = Inference::new();
        keep.declare("x", Some(&[2, 3, 4]));
        keep.run(&[with_ints(node("ReduceMean", &["x"], &["r"]), "axes", &[-1])]);
        assert_eq!(keep.shape_of("r"), Some(&[2, 3, 1][..]));

        let mut drop = Inference::new();
        drop.declare("x", Some(&[2, 3, 4]));
        drop.run(&[with_int(
            with_ints(node("ReduceMean", &["x"], &["r"]), "axes", &[-1]),
            "keepdims",
            0,
        )]);
        assert_eq!(drop.shape_of("r"), Some(&[2, 3][..]));
    }

    #[test]
    fn reduce_with_an_out_of_range_axis_proves_nothing() {
        let mut inf = Inference::new();
        inf.declare("x", Some(&[2, 3]));
        inf.run(&[with_ints(node("ReduceMean", &["x"], &["r"]), "axes", &[5])]);
        assert!(inf.fact("r").is_none());
    }

    #[test]
    fn gather_of_a_scalar_index_into_a_shape_vector_proves_the_element() {
        let mut inf = Inference::new();
        inf.declare("x", Some(&[-1, 12, 64]));
        inf.constant("idx", &[], Some(vec![2]));
        inf.run(&[
            node("Shape", &["x"], &["s"]),
            with_int(node("Gather", &["s", "idx"], &["g"]), "axis", 0),
        ]);
        assert_eq!(inf.shape_of("g"), Some(&[][..]));
        assert_eq!(inf.ints_of("g"), Some(&[Some(64)][..]));
    }

    #[test]
    fn transpose_permutes_and_rejects_a_malformed_perm() {
        let mut ok = Inference::new();
        ok.declare("x", Some(&[2, 3, 4]));
        ok.run(&[with_ints(
            node("Transpose", &["x"], &["t"]),
            "perm",
            &[0, 2, 1],
        )]);
        assert_eq!(ok.shape_of("t"), Some(&[2, 4, 3][..]));

        let mut bad = Inference::new();
        bad.declare("x", Some(&[2, 3, 4]));
        bad.run(&[with_ints(
            node("Transpose", &["x"], &["t"]),
            "perm",
            &[0, 0, 1],
        )]);
        assert!(bad.fact("t").is_none());
    }

    #[test]
    fn slice_keeps_the_rank_even_when_the_bounds_are_unreadable() {
        let mut inf = Inference::new();
        inf.declare("x", Some(&[4, 5]));
        inf.run(&[node("Slice", &["x", "starts", "ends"], &["s"])]);
        assert_eq!(inf.shape_of("s"), Some(&[-1, -1][..]));
    }

    #[test]
    fn slice_with_constant_bounds_proves_the_extent() {
        let mut inf = Inference::new();
        inf.declare("x", Some(&[10]));
        inf.constant("st", &[1], Some(vec![2]));
        inf.constant("en", &[1], Some(vec![7]));
        inf.run(&[node("Slice", &["x", "st", "en"], &["s"])]);
        assert_eq!(inf.shape_of("s"), Some(&[5][..]));
    }

    // -- monotonicity, contradiction, termination -------------------------------------------

    #[test]
    fn a_contradiction_withdraws_the_fact_and_is_counted() {
        let mut inf = Inference::new();
        inf.declare("x", Some(&[2, 3]));
        // Two producers claiming different ranks for one value cannot both be right.
        inf.set("x", Fact::of_shape(vec![4]));
        assert!(inf.fact("x").is_none());
        assert_eq!(inf.stats.contradictions, 1);
        // And it stays withdrawn.
        inf.set("x", Fact::of_shape(vec![2, 3]));
        assert!(inf.fact("x").is_none());
    }

    #[test]
    fn a_known_extent_is_never_replaced_by_a_different_one() {
        let mut inf = Inference::new();
        inf.declare("x", Some(&[-1, 768]));
        inf.set("x", Fact::of_shape(vec![4, 768]));
        assert_eq!(inf.shape_of("x"), Some(&[4, 768][..]), "unknown -> known");
        inf.set("x", Fact::of_shape(vec![5, 768]));
        assert!(inf.fact("x").is_none(), "known -> different is a poison");
    }

    #[test]
    fn a_cycle_terminates_within_the_sweep_budget() {
        // Not a legal ONNX graph — the point is that a malformed node list cannot hang.
        let mut inf = Inference::new();
        inf.declare("seed", Some(&[2, 2]));
        let nodes = vec![
            node("Identity", &["a", ""], &["b"]),
            node("Identity", &["b", ""], &["a"]),
            node("Identity", &["seed"], &["a"]),
        ];
        let stats = inf.run(&nodes);
        assert!(stats.sweeps <= SWEEP_LIMIT);
        assert!(!stats.hit_sweep_limit);
        assert_eq!(inf.shape_of("b"), Some(&[2, 2][..]));
    }

    #[test]
    fn fanout_reaches_every_consumer_in_one_pass() {
        let mut inf = Inference::new();
        inf.declare("x", Some(&[2, 3]));
        inf.run(&[
            node("Relu", &["x"], &["a"]),
            node("Relu", &["x"], &["b"]),
            node("Relu", &["x"], &["c"]),
        ]);
        for v in ["a", "b", "c"] {
            assert_eq!(inf.shape_of(v), Some(&[2, 3][..]));
        }
    }

    #[test]
    fn a_reversed_node_order_still_reaches_the_fixed_point() {
        let mut inf = Inference::new();
        inf.declare("x", Some(&[2, 3]));
        inf.run(&[
            node("Relu", &["c"], &["d"]),
            node("Relu", &["b"], &["c"]),
            node("Relu", &["x"], &["b"]),
        ]);
        assert_eq!(inf.shape_of("d"), Some(&[2, 3][..]));
    }

    #[test]
    fn an_unknown_op_proves_nothing_about_its_outputs() {
        let mut inf = Inference::new();
        inf.declare("x", Some(&[2, 3]));
        inf.run(&[node("NonZero", &["x"], &["y"])]);
        assert!(inf.fact("y").is_none());
        assert!(!has_rule("NonZero"));
    }

    #[test]
    fn a_non_default_domain_node_proves_nothing() {
        let mut inf = Inference::new();
        inf.declare("x", Some(&[2, 3]));
        let mut n = node("Relu", &["x"], &["y"]);
        n.domain = "com.microsoft".to_string();
        inf.run(&[n]);
        assert!(inf.fact("y").is_none());
    }

    // -- the overlay ------------------------------------------------------------------------

    #[test]
    fn refine_never_contradicts_ort() {
        let mut inf = Inference::new();
        inf.declare("x", Some(&[2, 3]));
        inf.run(&[node("Relu", &["x"], &["y"])]);
        let o = inf.finish();
        // Rank ORT already had: the overlay does not carry it at all.
        assert!(o.shape("x").is_none());
        // ORT's own reading wins on a rank disagreement.
        assert_eq!(o.refine("y", Some(&[9, 9, 9])), Some(vec![9, 9, 9]));
        // The ambiguous rank-0 reading is replaced.
        assert_eq!(o.refine("y", Some(&[])), Some(vec![2, 3]));
        assert_eq!(o.refine("y", None), Some(vec![2, 3]));
        // Nothing proven: nothing to refine.
        assert_eq!(o.refine("nope", None), None);
    }

    #[test]
    fn refine_only_fills_axes_ort_left_symbolic() {
        let mut inf = Inference::new();
        inf.declare("a", Some(&[-1, 768]));
        inf.declare("b", Some(&[4, 768]));
        inf.run(&[node("Add", &["a", "b"], &["c"])]);
        let o = inf.finish();
        assert_eq!(o.refine("c", Some(&[-1, -1])), Some(vec![4, 768]));
        assert_eq!(o.refine("c", Some(&[9, 768])), Some(vec![9, 768]));
    }

    #[test]
    fn stats_count_ranks_and_extents_separately() {
        let mut inf = Inference::new();
        inf.declare("x", Some(&[-1, 768]));
        inf.declare("w", Some(&[768, 768]));
        inf.run(&[node("MatMul", &["x", "w"], &["m"])]);
        let s = inf.finish().stats();
        assert_eq!(s.declared, 2);
        assert_eq!(s.ranks_proved, 1);
        assert_eq!(s.extents_proved, 1, "only the 768 is proven, not the batch");
    }
}
