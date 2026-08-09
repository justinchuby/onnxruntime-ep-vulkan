//! Partitioning policy — the minimum-viable-subgraph rule, enforced by arithmetic.
//!
//! # The failure this module exists to prevent
//!
//! `OP_COVERAGE.md` §7 calls it *death by fallback*: an EP that claims every node it individually
//! supports, in a graph where it does not support the nodes between them, shreds that graph into
//! dozens of tiny islands with a host round-trip on every boundary. The result is a model that
//! runs *slower* than pure CPU while every dashboard says op coverage went up. High op count
//! arrives weeks before any model actually runs faster, and this module is the instrument that
//! stops us believing our own coverage number in the meantime.
//!
//! So the rule is not "claim what you support". It is **claim an island only when the work inside
//! it pays for the transfers at its edges**, and that is a comparison between two numbers, not a
//! judgement call.
//!
//! # The metric contract (for Niobe)
//!
//! Niobe owns measurement methodology and has not started yet, so this module defines the metrics
//! precisely enough to adopt rather than re-invent. All four are computed by [`CoverageReport`]:
//!
//! | metric | definition | why it is the one that matters |
//! |---|---|---|
//! | `island_count` | number of maximal connected claimed subgraphs | fragmentation, directly |
//! | `largest_island_flops` | estimated FLOPs in the largest single island | the honest coverage number: 80% of *nodes* claimed across 40 islands is worth less than 20% claimed in one |
//! | `boundary_bytes_per_inference` | Σ over islands of input + output bytes crossing the EP boundary | what the transfers actually cost |
//! | `boundary_time_fraction` | modelled transfer time ÷ (transfer + compute) | the fraction of the run spent moving data instead of computing |
//!
//! `largest_island_flops` is the anti-self-deception metric: it cannot be improved by claiming
//! more scattered ops, only by closing the gaps between them.
//!
//! # Deliberately pure
//!
//! Nothing here touches the graph API or the device. It is data in, verdict out, so it is fully
//! unit-testable and the boundary layer can call it without inheriting any policy.

use crate::registry::{DeclineCode, DeclineReason, decline};

/// A linear model of host↔device transfer cost: `fixed_ns + bytes / bytes_per_ns`.
///
/// Linear is enough. The two things that decide whether a small island is worth claiming are the
/// per-transfer fixed cost (submission, barriers, mapping) and the bandwidth, and a two-parameter
/// fit captures both. Anything more elaborate is precision we cannot justify before Niobe has
/// measured a real device.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransferModel {
    /// Cost of a transfer independent of its size, in nanoseconds.
    pub fixed_ns: f64,
    /// Throughput, in bytes per nanosecond (i.e. GB/s).
    pub bytes_per_ns: f64,
}

impl TransferModel {
    /// Integrated GPU with unified memory: no copy, but still submission and barrier cost.
    ///
    /// The `bytes_per_ns` and `fixed_ns` figures are **still uncalibrated** — see the note on
    /// [`TransferModel::DISCRETE`]. What has been calibrated is the *byte* input they are
    /// reasoned from (Mouse 2026-07-31, both devices, byte-identical):
    ///
    /// ```text
    ///   upload   399,376 B/inference   (exactly linear over a 1/2/3-run sweep)
    ///   readback 457,344 B/inference   (exactly linear)
    ///   total    856,720 B = 0.817 MiB
    /// ```
    ///
    /// `cost_ns` on those bytes: 20,000 + 399,376/40 = ~29,984 ns up, 20,000 + 457,344/40 =
    /// ~31,434 ns down, ~61 µs total. Gate threshold (3× margin): ~184 µs.
    ///
    /// On UMA the variable cost is near zero, so `fixed_ns` is ~65% of the total and the gate is
    /// effectively "2 × fixed_ns" for any island whose boundary is small.
    pub const UMA: TransferModel = TransferModel {
        fixed_ns: 20_000.0,
        bytes_per_ns: 40.0,
    };

    /// Discrete GPU over PCIe: a real staging copy in each direction.
    ///
    /// Provisional until Niobe's calibration harness (`TransferModel::fit`) supplies real samples.
    /// Nominal PCIe-4 bandwidth ~16 GB/s; 12.0 bytes/ns is conservative.
    ///
    /// # What is calibrated here, and what is not
    ///
    /// **Calibrated (bytes).** `rust/tools/probe_staging_bytes.py`, Phi-3.5-mini-int4, 355 nodes
    /// claimed in 1 island, measured 2026-07-31 on selector 0 (NVIDIA RTX 4060) and selector 1
    /// (Intel Iris Xe). The two devices returned **byte-identical** counters:
    ///
    /// ```text
    ///   session_staging_upload_bytes    2,292,025,360 (1 run)
    ///   weight_cache_release_bytes      2,291,625,984
    ///   difference                            399,376  == the per-run delta, exactly
    ///
    ///   upload    399,376 B/inference     readback  457,344 B/inference
    ///   total     856,720 B  =  0.817 MiB
    /// ```
    ///
    /// The upload counter minus the resident weight set equals the per-inference delta to the
    /// byte, which is what makes this an *attribution* and not just a number: 99.98% of the
    /// 2.19 GiB is staged once and stays, and the 0.817 MiB is graph I/O plus the boundary
    /// tensors of the 8 nodes still on CPU.
    ///
    /// Two corrections to the figure this comment previously reasoned from:
    /// * The boundary is **0.817 MiB, not 0.756 MiB**, and it is **asymmetric** — 46.6% up,
    ///   53.4% down. The old comment assumed two symmetric 0.378 MiB halves.
    /// * The 0.756 → 0.38 MiB "halving" was **not** caused by claiming SimplifiedLayerNorm and
    ///   Gather. A same-instrument control built at `77d5d2a` on the same machine measured
    ///   405,512 B/inference *before* those claims. The two claims removed **6,136 B** — one
    ///   fp16 hidden-state row at s=1. Everything else was already gone.
    ///
    /// **NOT calibrated (nanoseconds).** `fixed_ns` and `bytes_per_ns` remain guesses.
    /// `TransferModel::fit` has still never been handed a real sample, and under R13 it cannot
    /// be: no wall-clock figure is quotable from a run whose verdict is not attributed `MATCH`,
    /// and `phase_containment` is RED on both devices. The counter *does* carry a
    /// `session_staging_upload_us` field; it is deliberately not used here. What landed is a
    /// byte measurement, not a nanosecond measurement.
    ///
    /// `cost_ns` on the measured bytes: 60,000 + 399,376/12 = ~93,281 ns up, 60,000 +
    /// 457,344/12 = ~98,112 ns down, ~191 µs total. Gate threshold (3× margin): ~574 µs.
    /// `fixed_ns` is ~63% of that, so for any island with a small boundary the gate is
    /// effectively `2 × fixed_ns` and is insensitive to the byte count.
    ///
    /// # Consequence for coverage — the gate is now ~2,800× less strict
    ///
    /// **Pre-residency (now resolved):** the EP re-uploaded ~2 GiB of weights every inference,
    /// making `transfer_ns` ≈ 2 GiB / 12 bytes/ns ≈ 167 ms per direction — larger than any
    /// kernel's compute time. The economics gate was pathologically strict: essentially nothing
    /// but an anchor could pass it, and the anchor exemption was carrying the whole partition.
    ///
    /// **Post-residency:** ~191 µs total. That is a ~1,750× drop in modelled transfer cost, and
    /// the 3× threshold falls from ~1 s to ~574 µs. The honest consequence is that
    /// [`evaluate`] will now **decline far less**, and islands previously judged uneconomic may
    /// be worth claiming. Coverage going *up* because transfer got cheap is the correct
    /// direction and is the opposite of what the pre-residency regime implied.
    ///
    /// The matching risk, stated so it is not discovered later: the gate's remaining teeth are
    /// almost entirely `fixed_ns`, which is the one parameter with no measurement behind it. An
    /// under-declining gate is a *silent* failure — it shows up as slow inference, not as a
    /// wrong answer. `test_partition_gate.py`'s `[partition]` assertion is the named falsifier;
    /// see `docs/OP_COVERAGE.md` §7.5 for why Phi-3.5 does not exercise it.
    pub const DISCRETE: TransferModel = TransferModel {
        fixed_ns: 60_000.0,
        bytes_per_ns: 12.0,
    };

    /// Per-inference host→device bytes on Phi-3.5-mini-int4, 355 nodes claimed in 1 island.
    ///
    /// Measured 2026-07-31 by `rust/tools/probe_staging_bytes.py` over a 1/2/3-run sweep; the
    /// delta is exactly linear and **byte-identical on selector 0 (NVIDIA RTX 4060) and
    /// selector 1 (Intel Iris Xe)**. This is the calibrated input to [`TransferModel::cost_ns`];
    /// the model's own `fixed_ns`/`bytes_per_ns` remain uncalibrated.
    pub const MEASURED_PHI35_UPLOAD_BYTES: u64 = 399_376;

    /// Per-inference device→host bytes on Phi-3.5-mini-int4. See
    /// [`TransferModel::MEASURED_PHI35_UPLOAD_BYTES`].
    ///
    /// Note the asymmetry: readback is **larger** than upload (53.4% of the boundary). Any model
    /// that assumes two symmetric halves is wrong, which the previous doc comment was.
    pub const MEASURED_PHI35_READBACK_BYTES: u64 = 457_344;

    /// Per-inference upload measured at `77d5d2a`, *before* `SimplifiedLayerNormalization` and
    /// `Gather` were claimed — the same-instrument, same-machine control.
    ///
    /// This constant exists to keep an honest result honest. The difference against
    /// [`TransferModel::MEASURED_PHI35_UPLOAD_BYTES`] is 6,136 bytes, which is what claiming
    /// those two ops actually bought. The much larger 0.756 → 0.38 MiB drop that was in flight
    /// at the time was **already present in the control** and belongs to residency, not to op
    /// coverage. Without running the control, that 2× would have been misattributed here.
    pub const CONTROL_PHI35_UPLOAD_BYTES_PRE_CLAIM: u64 = 405_512;

    /// Modelled cost of moving `bytes` across the boundary once.
    pub fn cost_ns(&self, bytes: u64) -> f64 {
        if bytes == 0 {
            return 0.0;
        }
        self.fixed_ns + (bytes as f64) / self.bytes_per_ns.max(f64::MIN_POSITIVE)
    }

    /// Least-squares fit of `(bytes, nanoseconds)` samples.
    ///
    /// This is the calibration hook: Niobe's harness times a staircase of transfer sizes on the
    /// target device and hands the samples here, and the partitioning rule stops being a guess.
    /// Returns `None` if the samples are degenerate (fewer than two distinct sizes).
    pub fn fit(samples: &[(u64, f64)]) -> Option<TransferModel> {
        if samples.len() < 2 {
            return None;
        }
        let n = samples.len() as f64;
        let sx: f64 = samples.iter().map(|(b, _)| *b as f64).sum();
        let sy: f64 = samples.iter().map(|(_, t)| *t).sum();
        let sxx: f64 = samples.iter().map(|(b, _)| (*b as f64) * (*b as f64)).sum();
        let sxy: f64 = samples.iter().map(|(b, t)| (*b as f64) * *t).sum();
        let denom = n * sxx - sx * sx;
        if denom.abs() < f64::EPSILON {
            return None;
        }
        let slope = (n * sxy - sx * sy) / denom;
        let intercept = (sy - slope * sx) / n;
        if slope <= 0.0 {
            return None;
        }
        Some(TransferModel {
            fixed_ns: intercept.max(0.0),
            bytes_per_ns: 1.0 / slope,
        })
    }

    /// This model with any diagnostic env overrides applied — see [`Policy::from_env`].
    ///
    /// `fixed_ns` is the parameter that has never been calibrated against a measurement (R13
    /// forbids it while no timing source on this project is trusted), and it is ~63% of the
    /// modelled DISCRETE cost at the measured Phi-3.5 boundary. Being able to move it from
    /// outside is what turns "we don't know its value" into "here is the range over which the
    /// verdict does not change".
    pub fn with_env_overrides(self) -> TransferModel {
        let m = TransferModel {
            fixed_ns: env_f64(ENV_FIXED_NS)
                .filter(|v| *v >= 0.0)
                .unwrap_or(self.fixed_ns),
            bytes_per_ns: env_f64(ENV_BYTES_PER_NS)
                .filter(|v| *v > 0.0)
                .unwrap_or(self.bytes_per_ns),
        };
        if m != self {
            log::warn!(
                "partition: TRANSFER MODEL OVERRIDDEN FROM ENV — {m:?} (default {self:?}). This \
                 run's partition artifacts are a counterfactual, not a default-configuration \
                 result."
            );
        }
        m
    }
}

/// One maximal connected subgraph this EP would claim.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Island {
    /// How many nodes it contains.
    pub nodes: usize,
    /// How many of those are *anchors* — nodes heavy enough to justify a boundary on their own
    /// (see [`is_anchor`]: a heavy op family **and** a resident weight at a schema-designated
    /// input).
    pub anchors: usize,
    /// Estimated floating-point operations inside the island.
    pub flops: u64,
    /// Bytes entering the island from outside the EP.
    pub input_bytes: u64,
    /// Bytes leaving the island to outside the EP.
    pub output_bytes: u64,
    /// Boundary slots whose byte count was **invented**, not read.
    ///
    /// `GetCapability` runs before shapes are resolved, so a tensor with a symbolic extent has
    /// no byte count at partition time. The estimator substitutes 128 for each unknown
    /// dimension. That is a fabricated number, and this field is how many of them went into
    /// `input_bytes + output_bytes`.
    ///
    /// It exists because the alternative is a byte count that looks measured and is not.
    /// §10.0.4 — prefer the unperturbable quantity as the claim of record; when the quantity is
    /// perturbable, say so next to it rather than downstream. A reader (or a gate) that sees
    /// `symbolic_boundary_slots > 0` knows the arithmetic below it is a lower bound on
    /// confidence, not a measurement, however precise the digits look.
    pub symbolic_boundary_slots: u64,
}

impl Island {
    /// Phi-3.5's single fused island **as `GetCapability` actually estimated it**, dev0,
    /// 2026-08-01, read out of `PartitionStats` in a trace this session (`bench/results/
    /// net_benefit_gate-trace-dev0.json`), not assumed.
    ///
    /// `nodes` is the claimed-node count from the CLAIM_LOG of the same run (355 claimed).
    /// `flops` and the boundary total are the estimator's own numbers.
    ///
    /// # `anchors` here is a MODEL, not a measured field (issue #73)
    ///
    /// **No artifact in this repository carries an anchor count.** `PartitionStats` has no anchor
    /// field, the trace cited above has none, and the `anchors` key in
    /// `bench/results/island_counterfactual_bert*.json` is a *list of op names*. Any number in this
    /// position is therefore a recomputation, and it is labelled as one.
    ///
    /// The value is derived, not read. `bench/results/_claim_log_phi35_r15_after.jsonl` records one
    /// line per node with the artifact fields `op` and `claimed`; **161** of its 355 claimed lines
    /// carry `op == "com.microsoft::MatMulNBits"`, and those are the only claimed lines in a heavy
    /// family that can present a resident weight. `MatMulNBits` designates inputs 1 `B`, 2 `scales`
    /// and 3 `zero_points`, of which the first two are **required**, and the pinned schema states
    /// in its own doc text that *"the weight matrix is a 2D constant matrix"*
    /// (`contrib_defs.cc:3639` at `da9b5e3`). `tests/ops/test_anchor_claims_are_witnessed.py`
    /// re-derives the 161 from that artifact on every run, so this constant has a falsifier.
    ///
    /// **What was withdrawn.** This field previously read `193`, from the retired name-only
    /// predicate: 161 `MatMulNBits` + 32 `GroupQueryAttention`. The `GroupQueryAttention` share is
    /// withdrawn, and under the shipped rule it is not a matter of missing evidence: GQA
    /// designates **no** weight site at all (see [`WEIGHT_SITE_AUDIT`]), so no GQA node anchors on
    /// any export. The other five claimed op types on this graph are not heavy families. 161 is
    /// the whole modelled anchor count for this island.
    ///
    /// The gate's verdict is unaffected either way: [`evaluate`] branches on `anchors > 0`.
    ///
    /// The 2026-07-31 estimate of Phi-3.5's island boundary, **which counted internal edges**.
    ///
    /// **Renamed 2026-08-02 from `MEASURED_PHI35_DEV0` (Morpheus ruling). It never held a
    /// measurement.** It held an estimate, now known wrong by 6.4×, sitting one screen above
    /// [`Island::MEASURED_PHI35_DEV0_REAL_BYTES`], which holds the actually-measured bytes. The
    /// doc comment below said all of that from the day the defect was found and it did not help:
    /// **names outlive doc comments**, and a reader who stopped at the identifier — or at a test
    /// name that referenced it — would conclude the opposite of what ships. Keeping the constant
    /// beside the corrected one is right; only the name was wrong.
    ///
    /// **Read the byte figure next to [`TransferModel::MEASURED_PHI35_UPLOAD_BYTES`] and be
    /// alarmed.** The estimator said 89,199,100,032 B crossed this island's boundary per
    /// inference; the instrumented transfer path says 856,720 B. That was a factor of ~104,000,
    /// and it was not a measurement disagreement — R11: the two sides of that comparison come
    /// from different sources and only one of them is a measurement.
    ///
    /// **2026-08-01 (Mouse): the first of the two causes is now fixed, and this constant is
    /// kept as the historical reading.** The defect had two independent halves:
    ///
    /// 1. *Internal edges counted as boundary.* The estimator counted **every** node's outputs,
    ///    including edges wholly inside the island, and called that conservative. `ep.rs` now
    ///    consults a whole-graph per-value consumer map and counts an output only when a node
    ///    outside the island reads it, or nothing reads it (a graph output). Re-measured on the
    ///    same model and device: **89,199,100,032 B → 13,936,509,056 B**, and the net-benefit
    ///    gate stopped needing the sole-island override — it now claims Phi-3.5's island on its
    ///    own economics (`net_benefit_sole_island_overrides` 1 → 0).
    ///
    /// 2. *Symbolic extents replaced by the constant 128.* Still open. Every boundary tensor in
    ///    Phi-3.5 is `runtime-extent`, so the remaining 13.9 GB is very largely the constant 128
    ///    raised to the power of however many symbolic dims each tensor has. The residual ratio
    ///    against the measured boundary is ~16,268×, and it is not noise: it is one invented
    ///    number propagating. See [`Island::symbolic_boundary_slots`], which now counts them, and
    ///    `OP_COVERAGE.md` for the open item. Calibrating `fixed_ns` before this is fixed is
    ///    still polishing the wrong parameter.
    ///
    /// The split between `input_bytes` and `output_bytes` is not recorded by `PartitionStats`;
    /// the whole total is parked in `output_bytes` here, which makes `transfer_ns` charge one
    /// fixed cost instead of two. That understates the transfer by exactly one `fixed_ns`, i.e.
    /// biases every test below *towards* claiming — the direction that makes the conclusions
    /// harder to reach, not easier.
    pub const ESTIMATED_PHI35_DEV0_INTERNAL_EDGES_COUNTED: Island = Island {
        nodes: 355,
        // MODEL, recomputed from claim-log op names — see the doc block above. Not a read field.
        anchors: 161,
        flops: 23_020_437_504,
        input_bytes: 0,
        output_bytes: 89_199_100_032,
        symbolic_boundary_slots: 0,
    };

    /// The same island as re-estimated on 2026-08-01 after internal edges stopped being counted.
    ///
    /// The boundary bytes and FLOPs are read out of the verbose `PartitionStats` summary on dev0
    /// with the same model and the same binary that produced the 355 → 0 claim census. The number
    /// that changed is the boundary; nodes and FLOPs are unaffected because the fix touches only
    /// which outputs are charged to the boundary.
    ///
    /// `anchors` is **not** among the fields `PartitionStats` reports — see the MODEL note on
    /// [`Island::ESTIMATED_PHI35_DEV0_INTERNAL_EDGES_COUNTED`]. It is carried here unchanged
    /// because the internal-edge fix cannot alter it, not because it was re-read.
    pub const ESTIMATED_PHI35_DEV0_INTERNAL_EDGES_FIXED: Island = Island {
        nodes: 355,
        // MODEL, recomputed from claim-log op names — see the doc block above. Not a read field.
        anchors: 161,
        flops: 23_020_437_504,
        input_bytes: 0,
        output_bytes: 13_936_509_056,
        // Every boundary tensor in Phi-3.5 carries at least one symbolic extent. The exact slot
        // count is a per-run observable now; what matters for the constant is that it is not 0,
        // so this figure must never be read as a measurement.
        symbolic_boundary_slots: 1,
    };

    /// The same island with the *measured* boundary substituted for the estimator's.
    ///
    /// Upload 399,376 B + readback 457,344 B = 856,720 B, asymmetric with readback larger.
    pub const MEASURED_PHI35_DEV0_REAL_BYTES: Island = Island {
        nodes: 355,
        // MODEL, recomputed from claim-log op names — see the doc block above. Not a read field.
        anchors: 161,
        flops: 23_020_437_504,
        input_bytes: TransferModel::MEASURED_PHI35_UPLOAD_BYTES,
        output_bytes: TransferModel::MEASURED_PHI35_READBACK_BYTES,
        symbolic_boundary_slots: 0,
    };

    /// Whether this island's boundary byte count contains fabricated dimensions.
    ///
    /// R9 amendment 5 — ask which way a check moves when its subject is wrong. A larger
    /// substituted dimension makes the estimate larger, makes the island look more
    /// transfer-dominated, and makes the gate reject; a smaller one makes it claim. The gate's
    /// verdict therefore moves with a constant nobody measured, which is not a defect that can
    /// be repaired by tightening the gate's thresholds. It is repaired by resolving shapes, or
    /// by declining to answer. Until then, callers can at least ask.
    pub fn boundary_is_fabricated(&self) -> bool {
        self.symbolic_boundary_slots > 0
    }

    /// Total bytes crossing this island's boundary once per inference.
    pub fn boundary_bytes(&self) -> u64 {
        self.input_bytes.saturating_add(self.output_bytes)
    }

    /// Modelled transfer cost: one transfer in, one transfer out.
    pub fn transfer_ns(&self, model: &TransferModel) -> f64 {
        model.cost_ns(self.input_bytes) + model.cost_ns(self.output_bytes)
    }

    /// Modelled compute cost, from FLOPs and a device rate.
    pub fn compute_ns(&self, flops_per_ns: f64) -> f64 {
        (self.flops as f64) / flops_per_ns.max(f64::MIN_POSITIVE)
    }
}

/// Ops whose arithmetic is matmul-shaped — the §4 inventory's "L/XL, compute-bound" rows.
///
/// This says something about an op's **arithmetic**, and its only consumer is the FLOP estimate
/// in [`crate::ep`]'s island builder. It is deliberately *not* the anchor predicate: see
/// [`is_anchor`], which additionally requires the node to present a resident weight.
///
/// The set is **bit-identical, in the same order**, to the list matched by the predicate that used
/// to be called `is_anchor` (issue #73). Only the name and the consumer changed, so no model's
/// FLOP estimate moves. The rename is the substance: the old name asserted an economic conclusion
/// — *worth a boundary on its own* — that the body never checked.
///
/// Held as one `const` slice rather than inlined into a `matches!` so it has exactly one
/// definition. `rust/tools/probe_island_counterfactual.py` parses this literal for its ranking
/// set, and `tests/ops/test_probe_anchor_mirror.py` fails on any divergence.
pub const HEAVY_OP_FAMILIES: &[&str] = &[
    "MatMul",
    "Gemm",
    "Conv",
    "ConvTranspose",
    "Attention",
    "com.microsoft::MatMulNBits",
    "com.microsoft::GroupQueryAttention",
    "com.microsoft::MultiHeadAttention",
    "com.microsoft::Attention",
    "com.microsoft::QMoE",
    "com.microsoft::LinearAttention",
];

/// Whether `qualified_op` is in [`HEAVY_OP_FAMILIES`]. See that constant for what it does and does
/// not mean.
pub fn is_heavy_op_family(qualified_op: &str) -> bool {
    HEAVY_OP_FAMILIES.contains(&qualified_op)
}

/// Whether a node presents a **resident weight** at one of its family's designated weight sites.
///
/// "Resident" means a graph initializer: uploaded once when the session is built and reused by
/// every inference, so its bytes are never charged to the per-inference boundary (`ep.rs` already
/// excludes constant inputs from `Island::input_bytes` for exactly that reason). That residency is
/// the entire economic warrant for the anchor exemption — a weight amortises the round trip across
/// inferences, an activation does not.
///
/// # Two states, not three
///
/// There is no `Unknown`. The sole production caller asks through
/// `crate::registry::NodeView::input_is_constant`, which is **total**: it answers `false` for a
/// missing slot, for a null slot, and for an ORT build that does not export
/// `ValueInfo_IsConstantInitializer`. Nothing in production could therefore produce a third state,
/// and a state only tests can reach is a state whose behaviour is unwitnessed.
///
/// The safety property such a state would carry is preserved by construction instead:
/// [`Absent`](Self::Absent) is the answer for *every* case that is not a witnessed initializer at a
/// designated site — including "could not ask" — and `Absent` does not anchor. The rule fails
/// closed on ignorance without needing a name for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightOperand {
    /// A designated weight site holds a graph initializer, witnessed at the body node.
    Present,
    /// No designated weight site was witnessed to hold a graph initializer.
    ///
    /// Covers all of: the family designates no site at all; an optional weight input is omitted;
    /// the input is present but is a runtime activation; the question could not be put to ORT.
    /// Every one of those is the same economic fact — nothing here is amortised — and every one
    /// fails closed at [`is_anchor`].
    Absent,
}

/// One input of one heavy family, as read at the pinned upstream source, with the audit verdict
/// that put it in or kept it out of [`weight_sites`].
///
/// This type exists because the previous attempt at this table stated a one-line dimensional rule
/// ("no batch or sequence extent") and shipped a table that rule does not generate. The honest
/// form is the audit itself: every input, its schema-declared shape text, and the reason it was
/// admitted or excluded — so a reviewer can check the derivation instead of trusting a summary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SchemaInput {
    /// Domain-qualified op, as [`crate::registry::NodeView::qualified_name`] renders it.
    pub op: &'static str,
    /// Input index in the pinned schema's declaration order.
    pub index: usize,
    /// Input name in the pinned schema.
    pub name: &'static str,
    /// The extents the pinned schema declares, quoted closely enough to check against it.
    /// `""` where the schema declares none — which is itself an audit fact, see [`X_UNDECLARED`].
    pub declared_shape: &'static str,
    /// Whether [`weight_sites`] designates this index.
    pub designated: bool,
    /// The audit reason. One of the `W_*` constants when designated, one of the `X_*` constants
    /// when not.
    pub reason: &'static str,
}

const fn site(
    op: &'static str,
    index: usize,
    name: &'static str,
    declared_shape: &'static str,
    designated: bool,
    reason: &'static str,
) -> SchemaInput {
    SchemaInput {
        op,
        index,
        name,
        declared_shape,
        designated,
        reason,
    }
}

/// Designating reason: the trained parameter block the layer's arithmetic is proportional to.
pub const W_WEIGHT: &str =
    "weight: the trained parameter block this layer's arithmetic scales with";
/// Designating reason: a per-block scale that reconstructs a designated weight.
pub const W_WEIGHT_SCALE: &str =
    "weight-scale: per-block scale that dequantizes a designated weight";
/// Designating reason: a per-block zero point that reconstructs a designated weight.
pub const W_WEIGHT_ZERO_POINT: &str =
    "weight-zero-point: per-block zero point that dequantizes a designated weight";
/// Designating reason: the default-domain linear families, designated by **operand position**.
///
/// ONNX's `MatMul` declares no extents for either operand ("N-dimensional matrix A"/"B"), so no
/// dimensional criterion can be evaluated for it at all. For these four families the weight is
/// fixed by position — input 1, the right-hand operand `B`/`W` — which is the only slot a linear
/// layer's parameters can occupy. This is the one place the rule is positional rather than a
/// reading of declared extents, and it is named rather than hidden.
pub const W_POSITIONAL: &str =
    "weight (positional): right-hand operand of a default-domain linear op";

/// Exclusion: a runtime activation.
pub const X_ACTIVATION: &str = "activation: produced by this inference";
/// Exclusion: a mask or attention bias.
pub const X_MASK: &str = "mask/attention-bias: shaped by the batch and the sequence";
/// Exclusion: a sequence-length or position tensor.
pub const X_LENGTH: &str = "length/position: a per-inference control tensor";
/// Exclusion: a KV cache.
pub const X_CACHE: &str =
    "kv-cache: carries state across steps of one inference, not across inferences";
/// Exclusion: a scale attached to a KV cache rather than to a weight.
pub const X_CACHE_SCALE: &str =
    "kv-cache scale: attached to a runtime cache, and the pinned schema declares no extents for it";
/// Exclusion: a precomputed positional table sized by the context window.
pub const X_POSITIONAL_TABLE: &str =
    "positional table: precomputed RoPE table sized by max_sequence_length, not a learned weight";
/// Exclusion: routing probabilities or weights, sized by the token count.
pub const X_ROUTING: &str = "routing: sized by num_tokens, so it varies with the inference";
/// Exclusion: a bias addend.
pub const X_BIAS: &str = "bias: an addend, not a multiplicand, and O(one model extent) against an O(batch x sequence x hidden) boundary";
/// Exclusion: a rank-1 per-head or per-channel parameter.
pub const X_PER_CHANNEL: &str = "per-channel parameter: genuinely learned, and rank 1 — too small to amortise a boundary crossing";
/// Exclusion: a per-tensor or per-expert global scale.
pub const X_GLOBAL_SCALE: &str =
    "global scale: rank 1, per-expert or per-tensor, not a per-block weight scale";
/// Exclusion: the pinned schema declares no extents, so the audit cannot see what the tensor is.
pub const X_UNDECLARED: &str = "undeclared extents in the pinned schema: fails closed";
/// Exclusion: a deprecated input this EP declines outright.
pub const X_DEPRECATED: &str = "deprecated by the pinned schema";

/// The per-input audit that [`weight_sites`] is read off. **This is the derivation.**
///
/// # The rule, stated as what it actually is
///
/// A site is designated when the pinned schema shows the input to be **the operator's weight** —
/// the trained parameter block whose bytes the layer's arithmetic is proportional to — or a
/// per-block scale or zero point that reconstructs such a block. Everything else is excluded, and
/// every exclusion carries its reason in the row.
///
/// **That criterion is an audited reading of the schema's own prose, not a dimensional test, and
/// this doc comment will not pretend otherwise.** The pinned schemas do not carry enough declared
/// structure for a dimensional test to be exact: ONNX's `MatMul` declares no extents at all, and
/// `GroupQueryAttention`'s `k_scale`/`v_scale` are declared with no shape. An earlier revision of
/// this work stated the rule as *"a tensor whose extents carry neither a batch nor a sequence
/// dimension"* and shipped a table that rule does not generate — it designated GQA's
/// sequence-sized RoPE caches `cos_cache`/`sin_cache` and its extent-less KV-cache scales,
/// omitted GQA's rank-1
/// `head_sink` which that rule admits, and omitted `QMoE`'s third expert weight matrix
/// `fc3_experts_weights` altogether. A rule that does not generate its table is worse than no
/// rule, because it invites
/// the reader to check the sentence instead of the rows.
///
/// Three consequences of the criterion are worth stating up front, because each one is a place a
/// reader would otherwise expect a different answer:
///
/// * **Bias addends are excluded**, though they are parameter data: `Gemm`'s `C`, `Conv`'s `B`,
///   `com.microsoft::Attention`'s and `MultiHeadAttention`'s `bias`, `MatMulNBits`'s `bias`,
///   `QMoE`'s `fc*_experts_bias`. A bias is O(output width); the boundary it would have to
///   amortise is O(batch x sequence x hidden). Excluding them is why `MultiHeadAttention`
///   designates nothing at all — a bias is its only parameter input.
/// * **`GroupQueryAttention` designates nothing.** It is the clearest case of what this issue is
///   about: an attention op consumes *pre-projected* Q/K/V, so it owns no weight matrix. Its
///   persistent inputs are RoPE tables sized by the context window (7, 8), scales attached to the
///   KV cache with no declared extents (12, 13), and three rank-1 per-head parameters (11
///   `head_sink`, 14 `q_norm_weight`, 15 `k_norm_weight`) which are real learned parameters and
///   are far too small to pay for a boundary. The family is heavy — its FLOPs are counted as
///   before — and it no longer anchors on sight. `Attention` (ONNX), `MultiHeadAttention` and
///   `LinearAttention` land in the same place for the same reason.
/// * **`QMoE` designates all three expert matrices**, `fc1`/`fc2`/`fc3`, with their scales and
///   zero points. `fc3_experts_weights` (8) is optional but is a weight whenever it is present;
///   omitting it was an error of transcription, not of rule.
///
/// # Provenance — SPECIFICATION
///
/// Contrib rows are read from ONNX Runtime **v1.28.0**, commit
/// `da9b5e364c465de65c49d91e696cd6485270757f` — the revision pinned by
/// `third_party/onnxruntime/PROVENANCE.md` — at the files and line ranges in [`SCHEMA_SOURCES`].
/// Default-domain rows are read from the ONNX operator schemas of the opsets this EP claims;
/// `tests/ops/test_anchor_weight_sites.py` re-reads the default-domain input names from the
/// installed `onnx` package and fails if they have moved.
pub const WEIGHT_SITE_AUDIT: &[SchemaInput] = &[
    // ── default domain ────────────────────────────────────────────────────────────────────────
    site("MatMul", 0, "A", "", false, X_ACTIVATION),
    site("MatMul", 1, "B", "", true, W_POSITIONAL),
    site("Gemm", 0, "A", "(M, K)", false, X_ACTIVATION),
    site("Gemm", 1, "B", "(K, N)", true, W_POSITIONAL),
    site("Gemm", 2, "C", "broadcastable to (M, N)", false, X_BIAS),
    site("Conv", 0, "X", "(N, C, D1..Dn)", false, X_ACTIVATION),
    site("Conv", 1, "W", "(M, C/group, k1..kn)", true, W_POSITIONAL),
    site("Conv", 2, "B", "(M)", false, X_BIAS),
    site(
        "ConvTranspose",
        0,
        "X",
        "(N, C, D1..Dn)",
        false,
        X_ACTIVATION,
    ),
    site(
        "ConvTranspose",
        1,
        "W",
        "(C, M/group, k1..kn)",
        true,
        W_POSITIONAL,
    ),
    site("ConvTranspose", 2, "B", "(M)", false, X_BIAS),
    site(
        "Attention",
        0,
        "Q",
        "(batch_size, q_num_heads, q_sequence_length, head_size)",
        false,
        X_ACTIVATION,
    ),
    site(
        "Attention",
        1,
        "K",
        "(batch_size, kv_num_heads, kv_sequence_length, head_size)",
        false,
        X_ACTIVATION,
    ),
    site(
        "Attention",
        2,
        "V",
        "(batch_size, kv_num_heads, kv_sequence_length, v_head_size)",
        false,
        X_ACTIVATION,
    ),
    site(
        "Attention",
        3,
        "attn_mask",
        "broadcastable to (batch_size, q_num_heads, q_sequence_length, total_sequence_length)",
        false,
        X_MASK,
    ),
    site(
        "Attention",
        4,
        "past_key",
        "(batch_size, kv_num_heads, past_sequence_length, head_size)",
        false,
        X_CACHE,
    ),
    site(
        "Attention",
        5,
        "past_value",
        "(batch_size, kv_num_heads, past_sequence_length, v_head_size)",
        false,
        X_CACHE,
    ),
    site(
        "Attention",
        6,
        "nonpad_kv_seqlen",
        "(batch_size,)",
        false,
        X_LENGTH,
    ),
    // ── com.microsoft::Attention — bert_defs.cc:491-573 ────────────────────────────────────────
    site(
        "com.microsoft::Attention",
        0,
        "input",
        "(batch_size, sequence_length, input_hidden_size)",
        false,
        X_ACTIVATION,
    ),
    site(
        "com.microsoft::Attention",
        1,
        "weights",
        "(input_hidden_size, hidden_size + hidden_size + v_hidden_size)",
        true,
        W_WEIGHT,
    ),
    site(
        "com.microsoft::Attention",
        2,
        "bias",
        "(hidden_size + hidden_size + v_hidden_size)",
        false,
        X_BIAS,
    ),
    site(
        "com.microsoft::Attention",
        3,
        "mask_index",
        "(batch_size, ...) mask or index",
        false,
        X_MASK,
    ),
    site(
        "com.microsoft::Attention",
        4,
        "past",
        "(2, batch_size, num_heads, past_sequence_length, head_size)",
        false,
        X_CACHE,
    ),
    site(
        "com.microsoft::Attention",
        5,
        "attention_bias",
        "(batch_size or 1, num_heads or 1, sequence_length, total_sequence_length)",
        false,
        X_MASK,
    ),
    site(
        "com.microsoft::Attention",
        6,
        "past_sequence_length",
        "scalar",
        false,
        X_LENGTH,
    ),
    // ── com.microsoft::MultiHeadAttention — bert_defs.cc:1096-1164 ─────────────────────────────
    site(
        "com.microsoft::MultiHeadAttention",
        0,
        "query",
        "(batch_size, sequence_length, hidden_size)",
        false,
        X_ACTIVATION,
    ),
    site(
        "com.microsoft::MultiHeadAttention",
        1,
        "key",
        "(batch_size, kv_sequence_length, hidden_size)",
        false,
        X_ACTIVATION,
    ),
    site(
        "com.microsoft::MultiHeadAttention",
        2,
        "value",
        "(batch_size, kv_sequence_length, v_hidden_size)",
        false,
        X_ACTIVATION,
    ),
    site(
        "com.microsoft::MultiHeadAttention",
        3,
        "bias",
        "(hidden_size + hidden_size + v_hidden_size)",
        false,
        X_BIAS,
    ),
    site(
        "com.microsoft::MultiHeadAttention",
        4,
        "key_padding_mask",
        "(batch_size, kv_sequence_length) and other mask forms",
        false,
        X_MASK,
    ),
    site(
        "com.microsoft::MultiHeadAttention",
        5,
        "attention_bias",
        "(batch_size or 1, num_heads or 1, sequence_length, total_sequence_length)",
        false,
        X_MASK,
    ),
    site(
        "com.microsoft::MultiHeadAttention",
        6,
        "past_key",
        "(batch_size, num_heads, past_sequence_length, head_size)",
        false,
        X_CACHE,
    ),
    site(
        "com.microsoft::MultiHeadAttention",
        7,
        "past_value",
        "(batch_size, num_heads, past_sequence_length, head_size)",
        false,
        X_CACHE,
    ),
    site(
        "com.microsoft::MultiHeadAttention",
        8,
        "past_sequence_length",
        "scalar",
        false,
        X_LENGTH,
    ),
    site(
        "com.microsoft::MultiHeadAttention",
        9,
        "cache_indirection",
        "(batch_size, beam_width, max_sequence_length)",
        false,
        X_CACHE,
    ),
    // ── com.microsoft::GroupQueryAttention — bert_defs.cc:1216-1335 ────────────────────────────
    site(
        "com.microsoft::GroupQueryAttention",
        0,
        "query",
        "(batch_size, sequence_length, hidden_size)",
        false,
        X_ACTIVATION,
    ),
    site(
        "com.microsoft::GroupQueryAttention",
        1,
        "key",
        "(batch_size, kv_sequence_length, kv_hidden_size)",
        false,
        X_ACTIVATION,
    ),
    site(
        "com.microsoft::GroupQueryAttention",
        2,
        "value",
        "(batch_size, kv_sequence_length, kv_hidden_size)",
        false,
        X_ACTIVATION,
    ),
    site(
        "com.microsoft::GroupQueryAttention",
        3,
        "past_key",
        "past state key, BNSH",
        false,
        X_CACHE,
    ),
    site(
        "com.microsoft::GroupQueryAttention",
        4,
        "past_value",
        "past state value, BNSH",
        false,
        X_CACHE,
    ),
    site(
        "com.microsoft::GroupQueryAttention",
        5,
        "seqlens_k",
        "(batch_size)",
        false,
        X_LENGTH,
    ),
    site(
        "com.microsoft::GroupQueryAttention",
        6,
        "total_sequence_length",
        "scalar",
        false,
        X_LENGTH,
    ),
    site(
        "com.microsoft::GroupQueryAttention",
        7,
        "cos_cache",
        "(max_sequence_length, head_size / 2)",
        false,
        X_POSITIONAL_TABLE,
    ),
    site(
        "com.microsoft::GroupQueryAttention",
        8,
        "sin_cache",
        "(max_sequence_length, head_size / 2)",
        false,
        X_POSITIONAL_TABLE,
    ),
    site(
        "com.microsoft::GroupQueryAttention",
        9,
        "position_ids",
        "(batch_size, sequence_length)",
        false,
        X_LENGTH,
    ),
    site(
        "com.microsoft::GroupQueryAttention",
        10,
        "attention_bias",
        "(batch_size or 1, num_heads or 1, sequence_length, total_sequence_length)",
        false,
        X_MASK,
    ),
    site(
        "com.microsoft::GroupQueryAttention",
        11,
        "head_sink",
        "(num_heads)",
        false,
        X_PER_CHANNEL,
    ),
    site(
        "com.microsoft::GroupQueryAttention",
        12,
        "k_scale",
        "",
        false,
        X_CACHE_SCALE,
    ),
    site(
        "com.microsoft::GroupQueryAttention",
        13,
        "v_scale",
        "",
        false,
        X_CACHE_SCALE,
    ),
    site(
        "com.microsoft::GroupQueryAttention",
        14,
        "q_norm_weight",
        "(head_size)",
        false,
        X_PER_CHANNEL,
    ),
    site(
        "com.microsoft::GroupQueryAttention",
        15,
        "k_norm_weight",
        "(head_size)",
        false,
        X_PER_CHANNEL,
    ),
    // ── com.microsoft::MatMulNBits — contrib_defs.cc:3648-3715 ─────────────────────────────────
    site(
        "com.microsoft::MatMulNBits",
        0,
        "A",
        "the input tensor, not quantized",
        false,
        X_ACTIVATION,
    ),
    site(
        "com.microsoft::MatMulNBits",
        1,
        "B",
        "(N, k_blocks, blob_size)",
        true,
        W_WEIGHT,
    ),
    site(
        "com.microsoft::MatMulNBits",
        2,
        "scales",
        "(N, k_blocks)",
        true,
        W_WEIGHT_SCALE,
    ),
    site(
        "com.microsoft::MatMulNBits",
        3,
        "zero_points",
        "(N, ceil(k_blocks * bits / 8)) packed, or (N, k_blocks)",
        true,
        W_WEIGHT_ZERO_POINT,
    ),
    site(
        "com.microsoft::MatMulNBits",
        4,
        "g_idx",
        "",
        false,
        X_DEPRECATED,
    ),
    site(
        "com.microsoft::MatMulNBits",
        5,
        "bias",
        "(N)",
        false,
        X_BIAS,
    ),
    // ── com.microsoft::QMoE — contrib_defs.cc:1469-1641 ────────────────────────────────────────
    site(
        "com.microsoft::QMoE",
        0,
        "input",
        "(num_tokens, hidden_size)",
        false,
        X_ACTIVATION,
    ),
    site(
        "com.microsoft::QMoE",
        1,
        "router_probs",
        "(num_tokens, num_experts)",
        false,
        X_ROUTING,
    ),
    site(
        "com.microsoft::QMoE",
        2,
        "fc1_experts_weights",
        "(num_experts, fusion_size * inter_size, hidden_size / pack_size)",
        true,
        W_WEIGHT,
    ),
    site(
        "com.microsoft::QMoE",
        3,
        "fc1_scales",
        "(num_experts, fusion_size * inter_size[, hidden_size / block_size])",
        true,
        W_WEIGHT_SCALE,
    ),
    site(
        "com.microsoft::QMoE",
        4,
        "fc1_experts_bias",
        "(num_experts, fusion_size * inter_size)",
        false,
        X_BIAS,
    ),
    site(
        "com.microsoft::QMoE",
        5,
        "fc2_experts_weights",
        "(num_experts, hidden_size, inter_size / pack_size)",
        true,
        W_WEIGHT,
    ),
    site(
        "com.microsoft::QMoE",
        6,
        "fc2_scales",
        "(num_experts, hidden_size[, inter_size / block_size])",
        true,
        W_WEIGHT_SCALE,
    ),
    site(
        "com.microsoft::QMoE",
        7,
        "fc2_experts_bias",
        "(num_experts, hidden_size)",
        false,
        X_BIAS,
    ),
    site(
        "com.microsoft::QMoE",
        8,
        "fc3_experts_weights",
        "(num_experts, inter_size, hidden_size / pack_size)",
        true,
        W_WEIGHT,
    ),
    site(
        "com.microsoft::QMoE",
        9,
        "fc3_scales",
        "(num_experts, inter_size[, hidden_size / block_size])",
        true,
        W_WEIGHT_SCALE,
    ),
    site(
        "com.microsoft::QMoE",
        10,
        "fc3_experts_bias",
        "(num_experts, inter_size)",
        false,
        X_BIAS,
    ),
    site(
        "com.microsoft::QMoE",
        11,
        "fc1_zero_points",
        "(num_experts, fusion_size * inter_size / pack_size) or 3D",
        true,
        W_WEIGHT_ZERO_POINT,
    ),
    site(
        "com.microsoft::QMoE",
        12,
        "fc2_zero_points",
        "(num_experts, hidden_size / pack_size) or 3D",
        true,
        W_WEIGHT_ZERO_POINT,
    ),
    site(
        "com.microsoft::QMoE",
        13,
        "fc3_zero_points",
        "(num_experts, inter_size / pack_size) or 3D",
        true,
        W_WEIGHT_ZERO_POINT,
    ),
    site(
        "com.microsoft::QMoE",
        14,
        "router_weights",
        "(num_tokens, num_experts)",
        false,
        X_ROUTING,
    ),
    site(
        "com.microsoft::QMoE",
        15,
        "fc1_global_scale",
        "(num_experts,)",
        false,
        X_GLOBAL_SCALE,
    ),
    site(
        "com.microsoft::QMoE",
        16,
        "fc2_global_scale",
        "(num_experts,)",
        false,
        X_GLOBAL_SCALE,
    ),
    site(
        "com.microsoft::QMoE",
        17,
        "fc1_act_scale",
        "(1,) or (num_experts,)",
        false,
        X_GLOBAL_SCALE,
    ),
    site(
        "com.microsoft::QMoE",
        18,
        "fc2_act_scale",
        "(1,) or (num_experts,)",
        false,
        X_GLOBAL_SCALE,
    ),
    // ── com.microsoft::LinearAttention — bert_defs.cc:2372-2431 ────────────────────────────────
    site(
        "com.microsoft::LinearAttention",
        0,
        "query",
        "(B, T, H_q * d_k)",
        false,
        X_ACTIVATION,
    ),
    site(
        "com.microsoft::LinearAttention",
        1,
        "key",
        "(B, T, H_kv * d_k)",
        false,
        X_ACTIVATION,
    ),
    site(
        "com.microsoft::LinearAttention",
        2,
        "value",
        "(B, T, H_kv * d_v)",
        false,
        X_ACTIVATION,
    ),
    site(
        "com.microsoft::LinearAttention",
        3,
        "past_state",
        "(B, H_kv, d_k, d_v)",
        false,
        X_CACHE,
    ),
    site(
        "com.microsoft::LinearAttention",
        4,
        "decay",
        "(B, T, H_kv * d_k) or (B, T, H_kv)",
        false,
        X_ACTIVATION,
    ),
    site(
        "com.microsoft::LinearAttention",
        5,
        "beta",
        "(B, T, H_kv) or (B, T, 1)",
        false,
        X_ACTIVATION,
    ),
];

/// Where each heavy family's audit rows were read from, at the pinned revision.
///
/// `ORT@da9b5e3` is ONNX Runtime v1.28.0, `da9b5e364c465de65c49d91e696cd6485270757f`, the exact
/// revision `third_party/onnxruntime/PROVENANCE.md` pins. `ONNX` rows are the default-domain
/// operator schemas.
pub const SCHEMA_SOURCES: &[(&str, &str)] = &[
    (
        "MatMul",
        "ONNX MatMul: inputs A, B; no extents declared for either",
    ),
    (
        "Gemm",
        "ONNX Gemm: A (M, K), B (K, N), optional C broadcastable to (M, N)",
    ),
    (
        "Conv",
        "ONNX Conv: X, W (M, C/group, kernel), optional 1-D B of size M",
    ),
    (
        "ConvTranspose",
        "ONNX ConvTranspose: X, W (C, M/group, kernel), optional 1-D B",
    ),
    (
        "Attention",
        "ONNX Attention (opset 24 as installed, onnx 1.22.0): Q, K, V, optional attn_mask, \
         past_key, past_value, nonpad_kv_seqlen — every input a runtime tensor",
    ),
    (
        "com.microsoft::Attention",
        "ORT@da9b5e3 onnxruntime/core/graph/contrib_ops/bert_defs.cc:491-573",
    ),
    (
        "com.microsoft::MultiHeadAttention",
        "ORT@da9b5e3 onnxruntime/core/graph/contrib_ops/bert_defs.cc:1096-1164",
    ),
    (
        "com.microsoft::GroupQueryAttention",
        "ORT@da9b5e3 onnxruntime/core/graph/contrib_ops/bert_defs.cc:1216-1335",
    ),
    (
        "com.microsoft::MatMulNBits",
        "ORT@da9b5e3 onnxruntime/core/graph/contrib_ops/contrib_defs.cc:3648-3715",
    ),
    (
        "com.microsoft::QMoE",
        "ORT@da9b5e3 onnxruntime/core/graph/contrib_ops/contrib_defs.cc:1469-1641",
    ),
    (
        "com.microsoft::LinearAttention",
        "ORT@da9b5e3 onnxruntime/core/graph/contrib_ops/bert_defs.cc:2372-2431",
    ),
];

/// The input indices at which a family's schema designates weight data.
///
/// Anchor eligibility consults **only** these sites. Reading them off the schema rather than
/// asking "is any input constant?" is what makes the rule schema-aware rather than coincidental: a
/// constant `attention_mask`, a baked `seqlens_k` or a constant-folded shape tensor is a constant
/// that is not a weight, and an anchor rule that counted it would re-admit the exemption by a side
/// door.
///
/// Every index here, and every index deliberately absent from here, is a row in
/// [`WEIGHT_SITE_AUDIT`] with its schema-declared shape and the reason for the verdict.
/// `weight_sites_are_exactly_the_designated_audit_rows` pins the two together, so this function
/// cannot drift from its own derivation.
///
/// An op outside [`HEAVY_OP_FAMILIES`] returns an empty slice, as does any heavy family whose
/// pinned schema designates no weight — `Attention`, `MultiHeadAttention`, `GroupQueryAttention`
/// and `LinearAttention`, all four of which consume pre-projected activations and own no weight
/// matrix. Empty means **never anchors**, which is the fail-closed direction.
pub fn weight_sites(qualified_op: &str) -> &'static [usize] {
    match qualified_op {
        "MatMul" | "Gemm" | "Conv" | "ConvTranspose" => &[1],
        "com.microsoft::Attention" => &[1],
        "com.microsoft::MatMulNBits" => &[1, 2, 3],
        "com.microsoft::QMoE" => &[2, 3, 5, 6, 8, 9, 11, 12, 13],
        _ => &[],
    }
}

/// Classify a node's weight operand from a per-site oracle.
///
/// `site_holds_initializer` answers, for one input index, whether that input is present **and** is
/// a resident graph initializer. It is deliberately total: there is no "don't know" reply, because
/// the production accessor has none to give (see [`WeightOperand`]).
///
/// Shared by `ep.rs` and by the tests deliberately — a second copy of this reasoning at the call
/// site is RAI-011 reappearing inside the fix for its own sibling.
pub fn classify_weight_operand(
    qualified_op: &str,
    mut site_holds_initializer: impl FnMut(usize) -> bool,
) -> WeightOperand {
    for &i in weight_sites(qualified_op) {
        if site_holds_initializer(i) {
            return WeightOperand::Present;
        }
    }
    WeightOperand::Absent
}

/// Whether one **node** is an anchor — heavy enough that a single one of it justifies an island.
///
/// A lone `Add` is never worth a round-trip; a single `MatMul` **on LLM-sized weights** always is.
/// That second clause is the doc comment this predicate carried from the day it was written, and
/// until issue #73 the body never tested it: matching the bare string `"MatMul"` exempted an
/// attention block's batched Q·Kᵀ and P·V products — whose operands are both runtime activations,
/// so nothing about them is amortised across inferences — from the very economics gate that exists
/// to catch them.
///
/// Anchor status is therefore a property of the node, not of the op name:
///
/// ```text
/// is_anchor(op, w)  ==  is_heavy_op_family(op) && w == WeightOperand::Present
/// ```
///
/// [`WeightOperand::Absent`] fails closed, and "closed" here is mild: a node that does not anchor
/// is not thereby rejected, it merely has to satisfy the size and economics gates like anything
/// else. The cost of being wrong in that direction is a CPU fallback; the cost of being wrong in
/// the other direction is a claim we cannot justify.
///
/// **Monotonicity.** `is_anchor(op, w) ⟹ is_heavy_op_family(op)` for every `(op, w)`, so the anchor
/// set is a subset of the retired name-only set and this change can only move a verdict
/// `Claim → Reject`, never `Reject → Claim`. Pinned end to end over constructed islands by
/// `new_anchor_semantics_never_newly_claim_over_the_production_chain`.
pub fn is_anchor(qualified_op: &str, weights: WeightOperand) -> bool {
    is_heavy_op_family(qualified_op) && matches!(weights, WeightOperand::Present)
}

/// The thresholds the rule is expressed in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Policy {
    /// An island with fewer nodes than this and no anchor is never claimed.
    pub min_nodes: usize,
    /// Compute must beat transfer by at least this factor.
    ///
    /// Not 1.0. A model this crude, calibrated on a different device, under a driver that
    /// schedules differently, is easily wrong by 2×; requiring a margin means the rule fails
    /// towards "run it on the CPU", which is always correct.
    pub margin: f64,
    /// Assumed device throughput in FLOPs per nanosecond (i.e. GFLOP/s).
    pub flops_per_ns: f64,
    /// Whether an island containing at least one [`is_anchor`] node skips the economics check.
    ///
    /// `true` in production. Settable to `false` (via [`ENV_ANCHOR_EXEMPTION`]) purely so the
    /// economics arithmetic can be *observed deciding* on a real model: Phi-3.5's single island
    /// is anchor-bearing, so with the exemption on, the economics branch is never the branch that
    /// answers, and an artifact that never varies proves nothing (R10).
    pub anchor_exemption: bool,
}

impl Default for Policy {
    fn default() -> Policy {
        Policy {
            min_nodes: 4,
            margin: 3.0,
            flops_per_ns: 1_000.0,
            anchor_exemption: true,
        }
    }
}

/// Env var: override [`Policy::margin`].
pub const ENV_MARGIN: &str = "ONNXRUNTIME_EP_VULKAN_PARTITION_MARGIN";
/// Env var: override [`Policy::min_nodes`].
pub const ENV_MIN_NODES: &str = "ONNXRUNTIME_EP_VULKAN_PARTITION_MIN_NODES";
/// Env var: override [`Policy::flops_per_ns`].
pub const ENV_FLOPS_PER_NS: &str = "ONNXRUNTIME_EP_VULKAN_PARTITION_FLOPS_PER_NS";
/// Env var: set to `0` to disable [`Policy::anchor_exemption`].
pub const ENV_ANCHOR_EXEMPTION: &str = "ONNXRUNTIME_EP_VULKAN_PARTITION_ANCHOR_EXEMPTION";
/// Env var: override [`TransferModel::fixed_ns`] — the one uncalibrated parameter.
pub const ENV_FIXED_NS: &str = "ONNXRUNTIME_EP_VULKAN_PARTITION_FIXED_NS";
/// Env var: override [`TransferModel::bytes_per_ns`].
pub const ENV_BYTES_PER_NS: &str = "ONNXRUNTIME_EP_VULKAN_PARTITION_BYTES_PER_NS";

fn env_f64(key: &str) -> Option<f64> {
    std::env::var(key).ok()?.trim().parse::<f64>().ok()
}

impl Policy {
    /// The default policy with any diagnostic env overrides applied.
    ///
    /// **These exist to make the gate falsifiable, not to be tuned in production.** R10's
    /// falsifier for "the gate is wired" is an artifact whose content *varies with the gate's
    /// input*; on Phi-3.5 the only island contains anchors, so without a way to move the gate's
    /// inputs the artifact is a constant and proves nothing. With them, the same model at the same
    /// commit produces `viable_islands_retained: 1` at one setting and
    /// `net_benefit_sole_island_overrides: 1` at another — a varying artifact, which is the proof.
    ///
    /// Any active override is logged at WARN so a run's artifact can never be read as a default
    /// run.
    pub fn from_env() -> Policy {
        let d = Policy::default();
        let p = Policy {
            min_nodes: std::env::var(ENV_MIN_NODES)
                .ok()
                .and_then(|s| s.trim().parse::<usize>().ok())
                .unwrap_or(d.min_nodes),
            margin: env_f64(ENV_MARGIN).unwrap_or(d.margin),
            flops_per_ns: env_f64(ENV_FLOPS_PER_NS)
                .filter(|v| *v > 0.0)
                .unwrap_or(d.flops_per_ns),
            anchor_exemption: match std::env::var(ENV_ANCHOR_EXEMPTION).ok().as_deref() {
                Some("0") | Some("false") | Some("off") => false,
                _ => d.anchor_exemption,
            },
        };
        if p != d {
            log::warn!(
                "partition: POLICY OVERRIDDEN FROM ENV — {p:?} (default {d:?}). This run's \
                 partition artifacts are a counterfactual, not a default-configuration result."
            );
        }
        p
    }
}

/// Why an island was not claimed.
#[derive(Clone, Debug, PartialEq)]
pub enum RejectReason {
    /// Too few nodes and nothing heavy inside.
    TooSmall {
        /// Node count.
        nodes: usize,
        /// The threshold it missed.
        min_nodes: usize,
    },
    /// The transfers cost more than the work is worth.
    TransferDominated {
        /// Modelled transfer nanoseconds.
        transfer_ns: f64,
        /// Modelled compute nanoseconds.
        compute_ns: f64,
        /// The margin that was required.
        margin: f64,
    },
}

/// The outcome of applying the rule to one island.
#[derive(Clone, Debug, PartialEq)]
pub enum Verdict {
    /// Claim it.
    Claim,
    /// Hand it back to the CPU EP.
    Reject(RejectReason),
}

impl Verdict {
    /// Whether the island survives.
    pub fn is_claim(&self) -> bool {
        matches!(self, Verdict::Claim)
    }
}

/// Apply the minimum-viable-subgraph rule.
///
/// Two gates, in this order:
///
/// 1. **Size.** Fewer than `min_nodes` and no anchor ⇒ reject. This alone kills the pathological
///    case: a graph of unsupported ops sprinkled with lone `Add`s.
/// 2. **Economics.** `compute_ns` must exceed `margin × transfer_ns`. This kills the subtler case:
///    a large island of cheap elementwise work whose tensors are bigger than its arithmetic.
///    **Anchor-containing islands are exempt from gate 2**: a node satisfying [`is_anchor`] is
///    heavy enough to justify a boundary on its own — that is the design invariant of
///    [`is_anchor`], and since issue #73 the predicate actually tests it (a heavy family **and** a
///    resident weight at a schema-designated input) instead of asserting it from the op name.
///    The provisional `TransferModel` constants are calibrated against real model execution and
///    may not reflect isolated unit-test input sizes; applying the economic check to anchors
///    would reject them when tested in isolation, which contradicts the stated design intent
///    ("a single MatMul on LLM-sized weights always is worth it").
pub fn evaluate(island: &Island, model: &TransferModel, policy: &Policy) -> Verdict {
    if island.nodes < policy.min_nodes && island.anchors == 0 {
        return Verdict::Reject(RejectReason::TooSmall {
            nodes: island.nodes,
            min_nodes: policy.min_nodes,
        });
    }

    // Anchor exemption: an island containing at least one anchor is unconditionally worth
    // claiming. The economic check below targets non-anchor scatter (cheap elementwise ops
    // whose boundary traffic exceeds their arithmetic). Anchors are excluded from that check.
    if island.anchors > 0 && policy.anchor_exemption {
        return Verdict::Claim;
    }

    let transfer_ns = island.transfer_ns(model);
    let compute_ns = island.compute_ns(policy.flops_per_ns);
    if compute_ns < policy.margin * transfer_ns {
        return Verdict::Reject(RejectReason::TransferDominated {
            transfer_ns,
            compute_ns,
            margin: policy.margin,
        });
    }

    Verdict::Claim
}

/// Render a rejection as a machine-readable decline, so a partition rejection appears in the same
/// histogram as a dtype or rank rejection instead of vanishing into a log line.
pub fn decline_for(reason: &RejectReason) -> DeclineReason {
    match reason {
        RejectReason::TooSmall { nodes, min_nodes } => decline(
            DeclineCode::Partition,
            format_args!(
                "this node's subgraph has only {nodes} nodes and no compute-heavy anchor \
                 (minimum {min_nodes}); claiming it would cost more in host transfers than it \
                 saves"
            ),
        ),
        RejectReason::TransferDominated {
            transfer_ns,
            compute_ns,
            margin,
        } => decline(
            DeclineCode::Partition,
            format_args!(
                "this node's subgraph is transfer-dominated: ~{compute_ns:.0}ns of compute against \
                 ~{transfer_ns:.0}ns of host transfer, below the {margin:.1}x margin required to \
                 claim it"
            ),
        ),
    }
}

/// What the gate did to one island: the verdict it computed, and whether that verdict was
/// overridden afterwards.
///
/// # Why this type exists (RAI-011)
///
/// The previous shape of this code had `GetCapability` decide *whether to call the gate at all*:
///
/// ```text
/// let verdict = if only_one_cluster { Verdict::Claim } else { partition::evaluate(..) };
/// ```
///
/// That is the R10 shape one level up. The gate was not merely unexercised on Phi-3.5 — it was
/// **unreachable** on Phi-3.5, and the two facts "the gate ran and retained nothing" and "the gate
/// never ran" both printed `viable_islands_retained: 0`.
///
/// The fix is not a second check. It is that **[`gate_islands`] is the only entry point, it always
/// evaluates, and the single-island exemption is applied *after* evaluation, as an override that
/// carries the verdict it overrode.** A sole island now produces a real verdict computed from real
/// bytes, and if that verdict is a rejection the override is a distinct observable state — not the
/// same digit as "all rejected".
#[derive(Clone, Debug, PartialEq)]
pub enum GateOutcome {
    /// The gate evaluated this island and its verdict stands.
    Evaluated(Verdict),
    /// The gate evaluated this island, rejected it, and the rejection was **overridden** because
    /// it is the graph's only island: there is no alternative partition to fall back to, so the
    /// choice is the whole graph on the EP or the whole graph on the CPU, and the claim predicate
    /// has already vetted every node inside it for correctness.
    ///
    /// The reason the gate gave is carried, not discarded. An override that loses the verdict it
    /// overrode is a bypass wearing a different name.
    SoleIslandOverride(RejectReason),
}

impl GateOutcome {
    /// The verdict the partitioner acts on.
    pub fn effective(&self) -> Verdict {
        match self {
            GateOutcome::Evaluated(v) => v.clone(),
            GateOutcome::SoleIslandOverride(_) => Verdict::Claim,
        }
    }

    /// The verdict the economics gate actually computed, before any override.
    ///
    /// This is the value that must vary with the gate's input for R10 to be satisfied.
    pub fn evaluated(&self) -> Verdict {
        match self {
            GateOutcome::Evaluated(v) => v.clone(),
            GateOutcome::SoleIslandOverride(r) => Verdict::Reject(r.clone()),
        }
    }

    /// Whether the island is claimed after any override.
    pub fn is_claim(&self) -> bool {
        self.effective().is_claim()
    }

    /// Whether the sole-island exemption changed this island's fate.
    pub fn is_override(&self) -> bool {
        matches!(self, GateOutcome::SoleIslandOverride(_))
    }
}

/// **The gate.** One entry point, reached from every caller, and it always evaluates.
///
/// Returns one [`GateOutcome`] per input island, in the same order. The sole-island exemption is
/// a property of the *set* (`islands.len() == 1`), which is precisely why it belongs here and not
/// at a call site: a call site that decides whether to consult the gate is a second gate.
///
/// Contract, stated so it can be broken loudly:
///
/// * `evaluate` is called exactly `islands.len()` times, unconditionally.
/// * The effective verdict differs from the evaluated verdict **only** for
///   [`GateOutcome::SoleIslandOverride`], which can only occur when `islands.len() == 1`.
pub fn gate_islands(
    islands: &[Island],
    model: &TransferModel,
    policy: &Policy,
) -> Vec<GateOutcome> {
    let sole = islands.len() == 1;
    islands
        .iter()
        .map(|island| match evaluate(island, model, policy) {
            Verdict::Reject(reason) if sole => GateOutcome::SoleIslandOverride(reason),
            v => GateOutcome::Evaluated(v),
        })
        .collect()
}

/// Apply the rule to a set of islands, returning the survivors and the rejections.
///
/// A thin projection of [`gate_islands`] — deliberately, so that "the survivors" and "what the
/// gate decided" cannot drift apart. Note that a *sole* island that the gate rejects still
/// survives here (the exemption), and the rejection is reported in the third slot rather than
/// vanishing.
pub fn retain_viable(
    islands: &[Island],
    model: &TransferModel,
    policy: &Policy,
) -> (Vec<Island>, Vec<(Island, RejectReason)>) {
    let mut kept = Vec::new();
    let mut dropped = Vec::new();
    for (island, outcome) in islands.iter().zip(gate_islands(islands, model, policy)) {
        match outcome {
            GateOutcome::Evaluated(Verdict::Claim) | GateOutcome::SoleIslandOverride(_) => {
                kept.push(island.clone())
            }
            GateOutcome::Evaluated(Verdict::Reject(r)) => dropped.push((island.clone(), r)),
        }
    }
    (kept, dropped)
}

/// The economics of one island, in the units the gate compares, for a `fixed_ns` it is handed.
///
/// Exists so that the sensitivity of the verdict to the **one uncalibrated parameter** can be
/// computed rather than asserted. `fixed_ns` is currently ~63% of the modelled DISCRETE cost and
/// no timing source on this project is trusted to calibrate it (R13), so the honest move is not to
/// pick a better number — it is to show which decisions would change across the range it could
/// plausibly take.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sensitivity {
    /// The `fixed_ns` this row was computed at.
    pub fixed_ns: f64,
    /// Modelled transfer cost at that `fixed_ns`.
    pub transfer_ns: f64,
    /// Modelled compute cost (independent of `fixed_ns`).
    pub compute_ns: f64,
    /// The verdict the *economics* check would return — i.e. **ignoring** the anchor exemption,
    /// which is what makes this interesting: for an anchor-bearing island the exemption returns
    /// `Claim` before this arithmetic happens, so this column reports what the gate would decide
    /// if the exemption were removed.
    pub economics_claims: bool,
}

/// Sweep `fixed_ns` across a range and report what the economics check would decide at each point.
///
/// The output is the sensitivity statement: if `economics_claims` is constant across the sweep,
/// the uncalibrated parameter does not change any decision for this island and calibrating it is
/// not on the critical path.
pub fn fixed_ns_sensitivity(
    island: &Island,
    model: &TransferModel,
    policy: &Policy,
    fixed_ns_points: &[f64],
) -> Vec<Sensitivity> {
    let compute_ns = island.compute_ns(policy.flops_per_ns);
    fixed_ns_points
        .iter()
        .map(|&fixed_ns| {
            let m = TransferModel { fixed_ns, ..*model };
            let transfer_ns = island.transfer_ns(&m);
            Sensitivity {
                fixed_ns,
                transfer_ns,
                compute_ns,
                economics_claims: compute_ns >= policy.margin * transfer_ns,
            }
        })
        .collect()
}

/// The four numbers that describe how well partitioning actually went.
///
/// This is the struct Niobe's harness should emit per model per device. Keeping it here, next to
/// the rule it measures, is deliberate: the rule and its metric drifting apart is exactly how a
/// coverage number becomes a lie.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CoverageReport {
    /// Nodes in the whole graph.
    pub total_nodes: usize,
    /// Islands this EP claimed.
    pub islands: Vec<Island>,
}

impl CoverageReport {
    /// Number of separate claimed subgraphs. More is worse.
    pub fn island_count(&self) -> usize {
        self.islands.len()
    }

    /// Nodes claimed, across all islands.
    pub fn claimed_nodes(&self) -> usize {
        self.islands.iter().map(|i| i.nodes).sum()
    }

    /// The naive coverage number, kept only so it can be compared against the honest one.
    pub fn node_coverage(&self) -> f64 {
        if self.total_nodes == 0 {
            return 0.0;
        }
        self.claimed_nodes() as f64 / self.total_nodes as f64
    }

    /// **The anti-self-deception metric.** FLOPs inside the single largest island.
    ///
    /// It does not move when scattered ops are claimed; it only moves when the gaps between them
    /// close. Report it next to `node_coverage` and the gap between the two is the honest measure
    /// of how far the EP is from running a model.
    pub fn largest_island_flops(&self) -> u64 {
        self.islands.iter().map(|i| i.flops).max().unwrap_or(0)
    }

    /// Nodes in the single largest island.
    pub fn largest_island_nodes(&self) -> usize {
        self.islands.iter().map(|i| i.nodes).max().unwrap_or(0)
    }

    /// Fraction of all claimed FLOPs that sit in the largest island. `1.0` means one island.
    pub fn concentration(&self) -> f64 {
        let total: u64 = self.islands.iter().map(|i| i.flops).sum();
        if total == 0 {
            return 0.0;
        }
        self.largest_island_flops() as f64 / total as f64
    }

    /// Bytes crossing the EP boundary once per inference, summed over islands.
    pub fn boundary_bytes_per_inference(&self) -> u64 {
        self.islands
            .iter()
            .fold(0u64, |a, i| a.saturating_add(i.boundary_bytes()))
    }

    /// Modelled fraction of the run spent transferring rather than computing.
    ///
    /// Above roughly 0.25 the partitioning is the bottleneck and no kernel work will help.
    pub fn boundary_time_fraction(&self, model: &TransferModel, policy: &Policy) -> f64 {
        let transfer: f64 = self.islands.iter().map(|i| i.transfer_ns(model)).sum();
        let compute: f64 = self
            .islands
            .iter()
            .map(|i| i.compute_ns(policy.flops_per_ns))
            .sum();
        let total = transfer + compute;
        if total <= 0.0 { 0.0 } else { transfer / total }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn big_matmul_island() -> Island {
        // One 4096x4096 @ 4096x4096 fp16 matmul: ~137 GFLOP against ~67 MB of boundary traffic.
        Island {
            nodes: 1,
            anchors: 1,
            flops: 2 * 4096 * 4096 * 4096,
            input_bytes: 2 * 4096 * 4096 * 2,
            output_bytes: 4096 * 4096 * 2,
            symbolic_boundary_slots: 0,
        }
    }

    fn lone_add_island() -> Island {
        Island {
            nodes: 1,
            anchors: 0,
            flops: 4096,
            input_bytes: 4096 * 4 * 2,
            output_bytes: 4096 * 4,
            symbolic_boundary_slots: 0,
        }
    }

    #[test]
    fn a_lone_elementwise_node_is_never_claimed() {
        let v = evaluate(
            &lone_add_island(),
            &TransferModel::DISCRETE,
            &Policy::default(),
        );
        assert!(matches!(
            v,
            Verdict::Reject(RejectReason::TooSmall { nodes: 1, .. })
        ));
    }

    #[test]
    fn a_lone_elementwise_node_is_not_claimed_on_uma_either() {
        // Unified memory removes the copy but not the submission cost, so the rule still holds.
        let v = evaluate(&lone_add_island(), &TransferModel::UMA, &Policy::default());
        assert!(!v.is_claim());
    }

    #[test]
    fn a_single_large_matmul_is_claimed_despite_being_one_node() {
        // The anchor exemption: node count is a proxy for work, and here the proxy is wrong.
        for model in [TransferModel::UMA, TransferModel::DISCRETE] {
            assert_eq!(
                evaluate(&big_matmul_island(), &model, &Policy::default()),
                Verdict::Claim
            );
        }
    }

    #[test]
    fn a_wide_but_cheap_island_is_rejected_as_transfer_dominated() {
        // Twenty elementwise ops over big tensors: plenty of nodes, almost no arithmetic.
        let island = Island {
            nodes: 20,
            anchors: 0,
            flops: 20 * 1024 * 1024,
            input_bytes: 64 * 1024 * 1024,
            output_bytes: 64 * 1024 * 1024,
            symbolic_boundary_slots: 0,
        };
        let v = evaluate(&island, &TransferModel::DISCRETE, &Policy::default());
        assert!(
            matches!(v, Verdict::Reject(RejectReason::TransferDominated { .. })),
            "{v:?}"
        );
    }

    #[test]
    fn size_is_checked_before_economics() {
        // So the decline reason names the first thing that is wrong, not the last.
        let island = Island {
            nodes: 1,
            anchors: 0,
            flops: 0,
            input_bytes: 0,
            output_bytes: 0,
            symbolic_boundary_slots: 0,
        };
        assert!(matches!(
            evaluate(&island, &TransferModel::UMA, &Policy::default()),
            Verdict::Reject(RejectReason::TooSmall { .. })
        ));
    }

    /// The two claims of 2026-07-31 bought 6,136 bytes per inference, not a halving.
    ///
    /// Falsifier for the recalibration in [`TransferModel::DISCRETE`]'s doc comment: if someone
    /// re-measures and edits one constant without the other, or re-attributes the residency win
    /// to op coverage, this goes red.
    #[test]
    fn the_two_claims_bought_six_kilobytes_not_a_halving() {
        let before = TransferModel::CONTROL_PHI35_UPLOAD_BYTES_PRE_CLAIM;
        let after = TransferModel::MEASURED_PHI35_UPLOAD_BYTES;
        assert!(after < before, "claiming ops must not increase upload");
        assert_eq!(
            before - after,
            6_136,
            "the measured saving from SimplifiedLayerNormalization + Gather"
        );
        // The control is the whole point: the pre-claim baseline is already far below the
        // 0.756 MiB the brief carried, so the halving is residency's, not coverage's.
        assert!(
            (before as f64) < 0.756 * 1024.0 * 1024.0 / 2.0 + 100_000.0,
            "control must show the drop predated these claims"
        );
    }

    /// The boundary is asymmetric — readback exceeds upload. Guards against anyone reinstating
    /// the "two symmetric halves" reasoning the old doc comment used.
    #[test]
    fn the_measured_boundary_is_asymmetric_and_readback_dominates() {
        let up = TransferModel::MEASURED_PHI35_UPLOAD_BYTES;
        let down = TransferModel::MEASURED_PHI35_READBACK_BYTES;
        assert!(down > up, "readback is the larger direction");
        assert_eq!(up + down, 856_720);
    }

    /// Post-residency, `fixed_ns` is the majority of the modelled cost on DISCRETE. That makes
    /// the gate insensitive to byte count for small islands and puts all its remaining teeth in
    /// the one parameter with no measurement behind it.
    #[test]
    fn post_residency_the_gate_is_dominated_by_fixed_cost() {
        let m = TransferModel::DISCRETE;
        let total = m.cost_ns(TransferModel::MEASURED_PHI35_UPLOAD_BYTES)
            + m.cost_ns(TransferModel::MEASURED_PHI35_READBACK_BYTES);
        let fixed_share = (2.0 * m.fixed_ns) / total;
        assert!(
            fixed_share > 0.5,
            "fixed_ns should dominate post-residency, got {fixed_share:.3}"
        );
        // And the whole-graph boundary is now far below the pre-residency regime, where a ~2 GiB
        // re-upload per inference made transfer_ns ~167 ms per direction.
        let pre_residency_ns = m.cost_ns(2 * 1024 * 1024 * 1024);
        assert!(
            pre_residency_ns / total > 500.0,
            "the gate must be dramatically less strict than pre-residency"
        );
    }

    #[test]
    fn rejections_are_machine_readable() {
        // Two islands, deliberately: with one, the sole-island override would keep it and there
        // would be no rejection to render. That the test had to change is the change.
        let (_, dropped) = retain_viable(
            &[lone_add_island(), lone_add_island()],
            &TransferModel::DISCRETE,
            &Policy::default(),
        );
        assert_eq!(dropped.len(), 2);
        let reason = decline_for(&dropped[0].1);
        assert_eq!(
            DeclineCode::of_reason(&reason),
            Some(DeclineCode::Partition)
        );
        assert!(reason.contains("host transfers"), "{reason}");
    }

    #[test]
    fn transfer_dominated_rejections_are_machine_readable_too() {
        let reason = decline_for(&RejectReason::TransferDominated {
            transfer_ns: 1000.0,
            compute_ns: 10.0,
            margin: 3.0,
        });
        assert_eq!(
            DeclineCode::of_reason(&reason),
            Some(DeclineCode::Partition)
        );
        assert!(reason.contains("transfer-dominated"), "{reason}");
    }

    #[test]
    fn retain_viable_splits_the_set() {
        let islands = vec![big_matmul_island(), lone_add_island(), big_matmul_island()];
        let (kept, dropped) = retain_viable(&islands, &TransferModel::DISCRETE, &Policy::default());
        assert_eq!(kept.len(), 2);
        assert_eq!(dropped.len(), 1);
    }

    #[test]
    fn zero_byte_boundaries_cost_nothing() {
        assert_eq!(TransferModel::DISCRETE.cost_ns(0), 0.0);
        assert!(TransferModel::DISCRETE.cost_ns(1) > 0.0);
    }

    #[test]
    fn the_transfer_model_can_be_calibrated_from_samples() {
        // 1000ns fixed, 10 bytes/ns.
        let samples: Vec<(u64, f64)> = (0..8)
            .map(|i| {
                let bytes = 1024u64 << i;
                (bytes, 1000.0 + bytes as f64 / 10.0)
            })
            .collect();
        let fitted = TransferModel::fit(&samples).expect("fit");
        assert!((fitted.fixed_ns - 1000.0).abs() < 1.0, "{fitted:?}");
        assert!((fitted.bytes_per_ns - 10.0).abs() < 0.01, "{fitted:?}");
    }

    #[test]
    fn degenerate_calibration_is_rejected_rather_than_guessed() {
        assert!(TransferModel::fit(&[]).is_none());
        assert!(TransferModel::fit(&[(1024, 5.0)]).is_none());
        assert!(TransferModel::fit(&[(1024, 5.0), (1024, 6.0)]).is_none());
    }

    #[test]
    fn heavy_op_families_are_bit_identical_to_the_retired_anchor_list() {
        // The FLOP estimate in `ep.rs` keys off this list. If a name is added, removed or
        // reordered, every FLOP-derived number in the repo moves and must be re-witnessed —
        // so the list is pinned in full, in order, not sampled.
        assert_eq!(
            HEAVY_OP_FAMILIES,
            &[
                "MatMul",
                "Gemm",
                "Conv",
                "ConvTranspose",
                "Attention",
                "com.microsoft::MatMulNBits",
                "com.microsoft::GroupQueryAttention",
                "com.microsoft::MultiHeadAttention",
                "com.microsoft::Attention",
                "com.microsoft::QMoE",
                "com.microsoft::LinearAttention",
            ]
        );
        assert!(is_heavy_op_family("MatMul"));
        assert!(is_heavy_op_family("com.microsoft::GroupQueryAttention"));
        assert!(!is_heavy_op_family("Add"));
        assert!(!is_heavy_op_family("Reshape"));
    }

    /// Issue #73, stated as a test: the op name alone can no longer make an anchor.
    #[test]
    fn anchoring_requires_a_heavy_family_and_a_resident_weight() {
        for &op in HEAVY_OP_FAMILIES {
            assert!(
                !is_anchor(op, WeightOperand::Absent),
                "{op} anchored on its name alone, which is the defect in issue #73"
            );
        }
        // ...and a weight alone is not enough either.
        for op in [
            "Add",
            "Reshape",
            "Softmax",
            "com.microsoft::SkipLayerNormalization",
        ] {
            assert!(
                !is_anchor(op, WeightOperand::Present),
                "{op} is not a heavy family and must not anchor"
            );
        }
        // Both halves: only families that designate a site can ever reach `Present` through
        // `classify_weight_operand`, so this is the whole reachable positive set.
        let designating: Vec<&str> = HEAVY_OP_FAMILIES
            .iter()
            .copied()
            .filter(|op| !weight_sites(op).is_empty())
            .collect();
        assert_eq!(
            designating,
            vec![
                "MatMul",
                "Gemm",
                "Conv",
                "ConvTranspose",
                "com.microsoft::MatMulNBits",
                "com.microsoft::Attention",
                "com.microsoft::QMoE",
            ]
        );
        for op in designating {
            assert!(
                is_anchor(op, WeightOperand::Present),
                "{op} with a weight must anchor"
            );
        }
    }

    /// D2: the shipped table must be *read off* the audit, not merely consistent with a sentence.
    #[test]
    fn weight_sites_are_exactly_the_designated_audit_rows() {
        for &family in HEAVY_OP_FAMILIES {
            let from_audit: Vec<usize> = WEIGHT_SITE_AUDIT
                .iter()
                .filter(|r| r.op == family && r.designated)
                .map(|r| r.index)
                .collect();
            assert_eq!(
                weight_sites(family),
                from_audit.as_slice(),
                "{family}: weight_sites() disagrees with its own audit rows"
            );
        }
        // An op with no audit rows at all designates nothing.
        assert!(weight_sites("Add").is_empty());
        assert!(weight_sites("com.microsoft::SkipLayerNormalization").is_empty());
    }

    #[test]
    fn the_audit_covers_every_heavy_family_contiguously_from_index_zero() {
        for &family in HEAVY_OP_FAMILIES {
            let rows: Vec<&SchemaInput> = WEIGHT_SITE_AUDIT
                .iter()
                .filter(|r| r.op == family)
                .collect();
            assert!(!rows.is_empty(), "{family} has no audit rows");
            for (expected, row) in rows.iter().enumerate() {
                assert_eq!(
                    row.index, expected,
                    "{family}: audit indices must be the schema's own declaration order, \
                     contiguous from 0, so a missing input is visible as a gap"
                );
                assert!(
                    !row.name.is_empty(),
                    "{family} input {expected} has no name"
                );
                assert!(
                    !row.reason.is_empty(),
                    "{family} input {expected} has no audit reason"
                );
            }
            assert!(
                SCHEMA_SOURCES.iter().any(|(op, _)| *op == family),
                "{family} has audit rows but no provenance row"
            );
        }
        // No audit row for an op outside the heavy families: the table's scope is exactly the
        // set whose anchor status is in question.
        for row in WEIGHT_SITE_AUDIT {
            assert!(
                is_heavy_op_family(row.op),
                "{} is audited but is not heavy",
                row.op
            );
        }
        for (op, src) in SCHEMA_SOURCES {
            assert!(
                is_heavy_op_family(op),
                "{op} has provenance but is not heavy"
            );
            assert!(!src.is_empty(), "{op} provenance is empty");
        }
    }

    /// The one thing this table must never do, checked lexically as well as semantically.
    #[test]
    fn no_designated_site_is_a_runtime_activation() {
        const ACTIVATION_WORDS: &[&str] = &[
            "query",
            "key",
            "value",
            "input",
            "mask",
            "past",
            "present",
            "cache",
            "seqlens",
            "position",
            "sequence",
            "bias",
            "router",
            "act_scale",
            "decay",
            "beta",
            "sink",
        ];
        for row in WEIGHT_SITE_AUDIT.iter().filter(|r| r.designated) {
            let name = row.name.to_ascii_lowercase();
            for word in ACTIVATION_WORDS {
                // `zero_points`/`scales` legitimately contain none of these; the check is that a
                // designated site never carries an activation-flavoured name.
                assert!(
                    !name.contains(word),
                    "{}: designated site {} `{}` reads as a runtime tensor (`{word}`)",
                    row.op,
                    row.index,
                    row.name
                );
            }
            assert!(
                row.reason == W_WEIGHT
                    || row.reason == W_WEIGHT_SCALE
                    || row.reason == W_WEIGHT_ZERO_POINT
                    || row.reason == W_POSITIONAL,
                "{}: designated site {} carries a non-designating reason",
                row.op,
                row.index
            );
        }
        for row in WEIGHT_SITE_AUDIT.iter().filter(|r| !r.designated) {
            assert!(
                row.reason.starts_with("activation")
                    || row.reason.starts_with("mask")
                    || row.reason.starts_with("length")
                    || row.reason.starts_with("kv-cache")
                    || row.reason.starts_with("positional table")
                    || row.reason.starts_with("routing")
                    || row.reason.starts_with("bias")
                    || row.reason.starts_with("per-channel")
                    || row.reason.starts_with("global scale")
                    || row.reason.starts_with("undeclared")
                    || row.reason.starts_with("deprecated"),
                "{}: excluded site {} carries an unrecognised reason `{}`",
                row.op,
                row.index,
                row.reason
            );
        }
    }

    /// Seraph's four D2 findings, each pinned individually so a regression names itself.
    #[test]
    fn the_four_audited_corrections_are_present() {
        let row = |op: &str, index: usize| {
            *WEIGHT_SITE_AUDIT
                .iter()
                .find(|r| r.op == op && r.index == index)
                .unwrap_or_else(|| panic!("{op} input {index} is not audited"))
        };
        // 1. GQA 11 `head_sink` is 1-D `(num_heads)`: audited, excluded on economics, not silently
        //    missing.
        let head_sink = row("com.microsoft::GroupQueryAttention", 11);
        assert_eq!(head_sink.name, "head_sink");
        assert_eq!(head_sink.declared_shape, "(num_heads)");
        assert!(!head_sink.designated);
        assert_eq!(head_sink.reason, X_PER_CHANNEL);
        // 2. GQA 12/13 declare no extents — so they are excluded, not designated.
        for (i, name) in [(12, "k_scale"), (13, "v_scale")] {
            let r = row("com.microsoft::GroupQueryAttention", i);
            assert_eq!(r.name, name);
            assert_eq!(
                r.declared_shape, "",
                "the pinned schema declares no extents here"
            );
            assert!(!r.designated);
            assert_eq!(r.reason, X_CACHE_SCALE);
        }
        // 3. GQA 7/8 are sequence-sized RoPE tables — excluded.
        for (i, name) in [(7, "cos_cache"), (8, "sin_cache")] {
            let r = row("com.microsoft::GroupQueryAttention", i);
            assert_eq!(r.name, name);
            assert!(r.declared_shape.contains("max_sequence_length"));
            assert!(!r.designated);
            assert_eq!(r.reason, X_POSITIONAL_TABLE);
        }
        // 4. QMoE 8 `fc3_experts_weights` is a weight — designated.
        let fc3 = row("com.microsoft::QMoE", 8);
        assert_eq!(fc3.name, "fc3_experts_weights");
        assert!(fc3.designated);
        assert_eq!(fc3.reason, W_WEIGHT);
        assert_eq!(
            weight_sites("com.microsoft::QMoE"),
            &[2, 3, 5, 6, 8, 9, 11, 12, 13],
            "all three expert matrices with their scales and zero points"
        );
    }

    /// The attention families are the *subject* of issue #73: they own no weight matrix, so no
    /// oracle — however permissive — can make them anchor.
    #[test]
    fn attention_families_cannot_anchor_under_any_oracle() {
        for op in [
            "Attention",
            "com.microsoft::MultiHeadAttention",
            "com.microsoft::GroupQueryAttention",
            "com.microsoft::LinearAttention",
        ] {
            assert!(weight_sites(op).is_empty(), "{op} designates a weight site");
            // A maximally permissive oracle: *every* input is a resident initializer.
            let w = classify_weight_operand(op, |_| true);
            assert_eq!(
                w,
                WeightOperand::Absent,
                "{op} found a weight it does not have"
            );
            assert!(!is_anchor(op, w), "{op} anchored");
        }
    }

    #[test]
    fn a_constant_at_a_non_designated_site_confers_nothing() {
        // A constant-folded `attention_mask`, a baked `seqlens_k`, a frozen shape tensor: all
        // real, none of them weights. Only the designated sites are consulted.
        let non_designated = |op: &str| {
            let sites = weight_sites(op);
            classify_weight_operand(op, |i| !sites.contains(&i))
        };
        for &op in HEAVY_OP_FAMILIES {
            assert_eq!(
                non_designated(op),
                WeightOperand::Absent,
                "{op} anchored off a constant at a non-designated input"
            );
        }
        // ...and the designated site alone is sufficient.
        assert_eq!(
            classify_weight_operand("com.microsoft::MatMulNBits", |i| i == 1),
            WeightOperand::Present
        );
        assert_eq!(
            classify_weight_operand("MatMul", |i| i == 0),
            WeightOperand::Absent,
            "the left-hand operand is not the weight site"
        );
        assert_eq!(
            classify_weight_operand("MatMul", |i| i == 1),
            WeightOperand::Present
        );
    }

    #[test]
    fn a_missing_or_unanswerable_site_fails_closed() {
        // The production oracle answers `false` for a missing slot, a null slot, and an ORT build
        // that cannot answer at all. All three are the same call here, and all three are `Absent`.
        for &op in HEAVY_OP_FAMILIES {
            assert_eq!(
                classify_weight_operand(op, |_| false),
                WeightOperand::Absent
            );
            assert!(!is_anchor(op, classify_weight_operand(op, |_| false)));
        }
        // An out-of-range index is simply never true, so an op whose optional weight is absent
        // (MatMulNBits with only `A` bound, say) is `Absent` rather than a panic.
        assert_eq!(
            classify_weight_operand("com.microsoft::MatMulNBits", |i| i > 100),
            WeightOperand::Absent
        );
    }

    /// The safety property of the whole change, exercised through shipped [`evaluate`].
    ///
    /// Non-vacuity is asserted, not hoped for: the sweep must contain at least one island the old
    /// name-only rule would have claimed and this one does not, otherwise the test would pass on a
    /// rollback and prove nothing.
    #[test]
    fn new_anchor_semantics_never_newly_claim_over_the_production_chain() {
        let policy = Policy::default();
        let model = TransferModel {
            fixed_ns: 1_000.0,
            bytes_per_ns: 10.0,
        };
        // Islands built the way `ep.rs` builds them: FLOPs from the heavy-family branch, anchors
        // from the node-level predicate. The "old" island is what the retired name-only rule
        // produced for the same cluster — every heavy-family node counted as an anchor.
        let mut strictly_tightened = 0usize;
        for &op in HEAVY_OP_FAMILIES {
            for weights in [WeightOperand::Present, WeightOperand::Absent] {
                for nodes in [1usize, 3, 6, 40] {
                    for output_bytes in [1u64 << 10, 1 << 20, 1 << 26] {
                        let flops = (nodes as u64) * 2 * 3072 * 3072;
                        let island = |anchors: usize| Island {
                            nodes,
                            anchors,
                            flops,
                            input_bytes: 1 << 20,
                            output_bytes,
                            symbolic_boundary_slots: 0,
                        };
                        let old = evaluate(&island(nodes), &model, &policy);
                        let now = evaluate(
                            &island(if is_anchor(op, weights) { nodes } else { 0 }),
                            &model,
                            &policy,
                        );
                        if now.is_claim() {
                            assert!(
                                old.is_claim(),
                                "{op}/{weights:?}/{nodes}n/{output_bytes}B: newly claimed, which \
                                 the monotonicity argument forbids"
                            );
                        }
                        if old.is_claim() && !now.is_claim() {
                            strictly_tightened += 1;
                        }
                    }
                }
            }
        }
        assert!(
            strictly_tightened > 0,
            "the sweep never observed a verdict tighten, so it would also pass on a rollback"
        );
    }

    /// Gate order is unchanged: size is still asked before economics.
    #[test]
    fn gate_order_is_unchanged_and_a_weightless_heavy_node_no_longer_skips_it() {
        let policy = Policy::default();
        let model = TransferModel {
            fixed_ns: 1_000.0,
            bytes_per_ns: 10.0,
        };
        let one_node = |anchors: usize| Island {
            nodes: 1,
            anchors,
            flops: 2 * 3072 * 3072,
            input_bytes: 1 << 20,
            output_bytes: 1 << 20,
            symbolic_boundary_slots: 0,
        };
        // A lone attention matmul: heavy family, both operands runtime activations.
        let weightless = classify_weight_operand("MatMul", |_| false);
        let verdict = evaluate(
            &one_node(usize::from(is_anchor("MatMul", weightless))),
            &model,
            &policy,
        );
        assert_eq!(
            verdict,
            Verdict::Reject(RejectReason::TooSmall {
                nodes: 1,
                min_nodes: policy.min_nodes,
            }),
            "size is still gate 1: a 1-node island is answered by min_nodes, not by economics"
        );
        // The same node with a resident weight is the case the exemption exists for.
        let weighted = classify_weight_operand("MatMul", |i| i == 1);
        assert_eq!(weighted, WeightOperand::Present);
        assert_eq!(
            evaluate(
                &one_node(usize::from(is_anchor("MatMul", weighted))),
                &model,
                &policy
            ),
            Verdict::Claim
        );
    }

    /// With the size gate satisfied, a weightless heavy cluster reaches economics and is declined
    /// there — the gate this defect was letting attention matmuls skip.
    #[test]
    fn a_weightless_heavy_cluster_is_declined_by_economics_not_exempted() {
        let policy = Policy::default();
        let model = TransferModel {
            fixed_ns: 1_000.0,
            bytes_per_ns: 10.0,
        };
        // Six nodes clears `min_nodes: 4`; the boundary is large and the arithmetic is not.
        let island = Island {
            nodes: 6,
            anchors: usize::from(is_anchor(
                "MatMul",
                classify_weight_operand("MatMul", |_| false),
            )),
            flops: 1_000,
            input_bytes: 1 << 26,
            output_bytes: 1 << 26,
            symbolic_boundary_slots: 0,
        };
        assert_eq!(island.anchors, 0);
        let verdict = evaluate(&island, &model, &policy);
        let Verdict::Reject(reason) = &verdict else {
            panic!("expected a rejection, got {verdict:?}");
        };
        assert!(
            matches!(reason, RejectReason::TransferDominated { .. }),
            "{reason:?}"
        );
        let declined = decline_for(reason);
        assert_eq!(
            DeclineCode::of_reason(&declined),
            Some(DeclineCode::Partition),
            "a partition rejection must land in the decline histogram, not a log line: {declined}"
        );
        assert!(declined.contains("transfer-dominated"), "{declined}");

        // The identical island *with* a weight is exempted, which is the contrast that makes the
        // decline attributable to the weight and not to the thresholds.
        let anchored = Island {
            anchors: 1,
            ..island
        };
        assert_eq!(evaluate(&anchored, &model, &policy), Verdict::Claim);
    }

    #[test]
    fn coverage_report_distinguishes_scattered_from_concentrated() {
        // The whole point: same node coverage, very different reality.
        let scattered = CoverageReport {
            total_nodes: 100,
            islands: (0..40)
                .map(|_| Island {
                    nodes: 2,
                    anchors: 0,
                    flops: 1_000,
                    input_bytes: 1 << 20,
                    output_bytes: 1 << 20,
                    symbolic_boundary_slots: 0,
                })
                .collect(),
        };
        let concentrated = CoverageReport {
            total_nodes: 100,
            islands: vec![Island {
                nodes: 80,
                anchors: 8,
                flops: 40_000,
                input_bytes: 1 << 20,
                output_bytes: 1 << 20,
                symbolic_boundary_slots: 0,
            }],
        };

        assert_eq!(scattered.claimed_nodes(), 80);
        assert_eq!(concentrated.claimed_nodes(), 80);
        assert!((scattered.node_coverage() - concentrated.node_coverage()).abs() < 1e-9);

        assert!(
            concentrated.largest_island_flops() > scattered.largest_island_flops() * 10,
            "the honest metric must separate these two"
        );
        assert_eq!(scattered.island_count(), 40);
        assert_eq!(concentrated.island_count(), 1);
        assert!((concentrated.concentration() - 1.0).abs() < 1e-9);
        assert!(scattered.concentration() < 0.1);
        assert!(
            scattered.boundary_bytes_per_inference()
                > concentrated.boundary_bytes_per_inference() * 10
        );
    }

    #[test]
    fn boundary_time_fraction_flags_a_shredded_graph() {
        let shredded = CoverageReport {
            total_nodes: 100,
            islands: (0..40)
                .map(|_| Island {
                    nodes: 2,
                    anchors: 0,
                    flops: 1_000,
                    input_bytes: 1 << 20,
                    output_bytes: 1 << 20,
                    symbolic_boundary_slots: 0,
                })
                .collect(),
        };
        let f = shredded.boundary_time_fraction(&TransferModel::DISCRETE, &Policy::default());
        assert!(
            f > 0.9,
            "a shredded graph should be almost all transfer: {f}"
        );
    }

    #[test]
    fn an_empty_report_does_not_divide_by_zero() {
        let empty = CoverageReport::default();
        assert_eq!(empty.node_coverage(), 0.0);
        assert_eq!(empty.largest_island_flops(), 0);
        assert_eq!(empty.largest_island_nodes(), 0);
        assert_eq!(empty.concentration(), 0.0);
        assert_eq!(empty.boundary_bytes_per_inference(), 0);
        assert_eq!(
            empty.boundary_time_fraction(&TransferModel::UMA, &Policy::default()),
            0.0
        );
    }

    #[test]
    fn the_default_policy_is_the_documented_one() {
        let p = Policy::default();
        assert_eq!(p.min_nodes, 4);
        assert!((p.margin - 3.0).abs() < f64::EPSILON);
        assert!(p.anchor_exemption, "the exemption ships on");
    }

    // ---------------------------------------------------------------------------------------
    // RAI-011 — one gate, reachable from everywhere, and bypassed ≠ all-rejected
    // ---------------------------------------------------------------------------------------

    /// The property that makes RAI-011 impossible to reintroduce here: **every** island is
    /// evaluated, including the sole one. If someone puts a `if islands.len() == 1 { return }`
    /// back in front of the gate, the sole island's outcome stops carrying a computed verdict and
    /// this goes red.
    #[test]
    fn the_sole_island_is_evaluated_not_bypassed() {
        let out = gate_islands(
            &[lone_add_island()],
            &TransferModel::DISCRETE,
            &Policy::default(),
        );
        assert_eq!(out.len(), 1);
        // Effective: claimed, because it is the only island and there is nothing to fall back to.
        assert!(out[0].is_claim());
        // Evaluated: rejected, with the reason the gate computed. A bypass could not produce this.
        assert!(matches!(
            out[0].evaluated(),
            Verdict::Reject(RejectReason::TooSmall { nodes: 1, .. })
        ));
        assert!(out[0].is_override());
    }

    /// The two states that used to share the digit `0` must not be reachable from one another.
    ///
    /// `Evaluated(Reject)` (multi-island: really rejected, handed back to the CPU) and
    /// `SoleIslandOverride` (evaluated, rejected, kept anyway) are different values of a sum type,
    /// so no arithmetic can conflate them — the counters mirror this with
    /// `viable_islands_retained` and `net_benefit_sole_island_overrides`.
    #[test]
    fn all_rejected_and_overridden_are_different_values_not_different_readings() {
        let one = gate_islands(
            &[lone_add_island()],
            &TransferModel::DISCRETE,
            &Policy::default(),
        );
        let two = gate_islands(
            &[lone_add_island(), lone_add_island()],
            &TransferModel::DISCRETE,
            &Policy::default(),
        );
        assert!(one[0].is_override() && one[0].is_claim());
        assert!(!two[0].is_override() && !two[0].is_claim());
        assert!(!two[1].is_override() && !two[1].is_claim());
        // Same island, same model, same policy — opposite fates, decided only by set size.
        assert_ne!(one[0], two[0]);
    }

    /// An anchor-bearing sole island is claimed by the gate itself, not by the override. This is
    /// the shipping configuration on Phi-3.5, and it is why `viable_islands_retained` reads `1`
    /// there rather than `0` with an override.
    #[test]
    fn an_anchor_bearing_sole_island_is_claimed_by_the_gate_not_the_override() {
        let out = gate_islands(
            &[Island::ESTIMATED_PHI35_DEV0_INTERNAL_EDGES_COUNTED],
            &TransferModel::DISCRETE,
            &Policy::default(),
        );
        assert_eq!(out[0], GateOutcome::Evaluated(Verdict::Claim));
        assert!(!out[0].is_override());
    }

    /// `retain_viable` and `gate_islands` cannot disagree, because one is defined by the other.
    #[test]
    fn retain_viable_is_a_projection_of_the_one_gate() {
        let islands = [big_matmul_island(), lone_add_island(), lone_add_island()];
        let (kept, dropped) = retain_viable(&islands, &TransferModel::DISCRETE, &Policy::default());
        let outcomes = gate_islands(&islands, &TransferModel::DISCRETE, &Policy::default());
        assert_eq!(kept.len(), outcomes.iter().filter(|o| o.is_claim()).count());
        assert_eq!(
            dropped.len(),
            outcomes.iter().filter(|o| !o.is_claim()).count()
        );
        assert_eq!(kept.len(), 1);
        assert_eq!(dropped.len(), 2);
    }

    // ---------------------------------------------------------------------------------------
    // Task 2 — `fixed_ns` sensitivity. Not a calibration: a statement of what it cannot change.
    // ---------------------------------------------------------------------------------------

    /// The plausible range for the one uncalibrated parameter, spanning ~1 µs to 100 ms per
    /// transfer. Wider than any defensible guess, on purpose: the point is what survives it.
    const PLAUSIBLE_FIXED_NS: [f64; 8] = [
        0.0,
        1_000.0,
        20_000.0,
        60_000.0,
        1_000_000.0,
        5_000_000.0,
        20_000_000.0,
        100_000_000.0,
    ];

    /// **Sensitivity statement, part 1.** Fed the estimator's own boundary figure, the economics
    /// check rejects Phi-3.5's island at *every* value of `fixed_ns` including zero. The verdict
    /// is therefore not a function of `fixed_ns` at all, and calibrating it would change nothing.
    ///
    /// Matches the measured sweep in `bench/results/net_benefit_gate_probe-dev0.json`:
    /// `no_anchor_fixed_{1000 … 100000000}` all produce
    /// `net_benefit_sole_island_overrides: 1, viable_islands_retained: 0`.
    #[test]
    fn fixed_ns_cannot_change_the_verdict_on_the_internal_edges_counted_phi35_island() {
        let rows = fixed_ns_sensitivity(
            &Island::ESTIMATED_PHI35_DEV0_INTERNAL_EDGES_COUNTED,
            &TransferModel::DISCRETE,
            &Policy::default(),
            &PLAUSIBLE_FIXED_NS,
        );
        assert!(
            rows.iter().all(|r| !r.economics_claims),
            "expected a constant REJECT across the range, got {rows:?}"
        );
        // And by how much, at the most generous end: the byte term alone loses by ~10^3.
        let at_zero = rows[0];
        let ratio = at_zero.compute_ns / (Policy::default().margin * at_zero.transfer_ns);
        assert!(
            ratio < 1e-2,
            "the byte term should dominate by orders of magnitude, got {ratio}"
        );
    }

    /// **Sensitivity statement, part 2.** Fed the *measured* boundary instead, the same island is
    /// claimed across the whole plausible range up to a flip point of ~3.8 ms per transfer —
    /// 63× above the current uncalibrated guess of 60 µs. Either way the decision does not turn
    /// on `fixed_ns`.
    #[test]
    fn with_measured_bytes_the_flip_point_is_far_outside_the_plausible_range() {
        let island = Island::MEASURED_PHI35_DEV0_REAL_BYTES;
        let policy = Policy::default();
        let rows = fixed_ns_sensitivity(
            &island,
            &TransferModel::DISCRETE,
            &policy,
            &PLAUSIBLE_FIXED_NS,
        );
        for r in rows.iter().filter(|r| r.fixed_ns <= 1_000_000.0) {
            assert!(
                r.economics_claims,
                "should claim at fixed_ns={}",
                r.fixed_ns
            );
        }
        // Solve for the flip: compute = margin·(2·fixed + bytes/bps).
        let bytes_term = (island.boundary_bytes() as f64) / TransferModel::DISCRETE.bytes_per_ns;
        let flip = (island.compute_ns(policy.flops_per_ns) / policy.margin - bytes_term) / 2.0;
        assert!(
            (3.75e6..3.85e6).contains(&flip),
            "flip point moved: {flip} ns"
        );
        assert!(
            flip / TransferModel::DISCRETE.fixed_ns > 50.0,
            "the current guess must sit well below the flip"
        );
    }

    /// **The number that actually decides the gate is not `fixed_ns`.** `GetCapability`'s island
    /// estimator reports a boundary far larger than the instrumented transfer path measures.
    /// Until that is fixed, calibrating nanoseconds is polishing the wrong parameter.
    ///
    /// Kept as the historical reading after the 2026-08-01 partial fix, so the size of the
    /// original defect stays on the record rather than being quietly replaced by the smaller
    /// number that succeeded it.
    #[test]
    fn the_internal_edges_counted_estimate_disagreed_with_the_measured_boundary_by_five_orders_of_magnitude()
     {
        let estimated = Island::ESTIMATED_PHI35_DEV0_INTERNAL_EDGES_COUNTED.boundary_bytes();
        let measured = Island::MEASURED_PHI35_DEV0_REAL_BYTES.boundary_bytes();
        assert_eq!(measured, 856_720);
        let ratio = estimated as f64 / measured as f64;
        assert!(
            (1.0e4..1.0e6).contains(&ratio),
            "estimator/measured boundary ratio was {ratio}"
        );
    }

    /// Half the defect is fixed, and the half that is left is a different defect.
    ///
    /// Counting internal edges as boundary bytes removed a factor of ~6.4. What remains is not a
    /// residue of the same error: it is the constant 128 standing in for every symbolic extent,
    /// which is a fabricated input rather than an over-broad one. Naming them separately matters
    /// because they have different fixes and only one of them is done.
    ///
    /// R11 — a decomposition that appears to close is the hardest kind of wrong. This test
    /// deliberately does **not** assert that the estimator now agrees with the measurement. It
    /// asserts the improvement that was actually obtained and that the disagreement is still
    /// four orders of magnitude, so nobody reads the 6.4× as a closure.
    #[test]
    fn removing_internal_edges_shrank_the_estimate_but_did_not_close_the_gap() {
        let before = Island::ESTIMATED_PHI35_DEV0_INTERNAL_EDGES_COUNTED.boundary_bytes();
        let after = Island::ESTIMATED_PHI35_DEV0_INTERNAL_EDGES_FIXED.boundary_bytes();
        let measured = Island::MEASURED_PHI35_DEV0_REAL_BYTES.boundary_bytes();

        assert_eq!(before, 89_199_100_032);
        assert_eq!(after, 13_936_509_056);
        assert!(
            after < before,
            "the fix must reduce the estimate, not merely change it"
        );

        let improvement = before as f64 / after as f64;
        assert!(
            (6.0..7.0).contains(&improvement),
            "internal-edge removal was measured at ~6.4x on dev0; got {improvement}"
        );

        let residual = after as f64 / measured as f64;
        assert!(
            residual > 1.0e4,
            "the residual disagreement is still four orders of magnitude ({residual}); a test \
             that let this pass as closed would be the decomposition-appears-to-close failure"
        );

        assert!(
            Island::ESTIMATED_PHI35_DEV0_INTERNAL_EDGES_FIXED.boundary_is_fabricated(),
            "the post-fix estimate is still built on substituted dimensions and must say so"
        );
        assert!(
            !Island::MEASURED_PHI35_DEV0_REAL_BYTES.boundary_is_fabricated(),
            "the measured boundary contains no fabricated dimension, or it is not a measurement"
        );
    }

    /// **§10.0.4 third form — prefer the bound you can sign.** The claim on Phi-3.5's island
    /// survives a 16,268× adversarial inflation of the term that opposes it.
    ///
    /// This test exists because my own framing of the boundary fix was declined, and the
    /// replacement is stronger and is recorded here rather than in prose. I said the economics
    /// arm now *concurs* with the anchor exemption. Morpheus refused that: **"agreement between
    /// two things fed the same fabricated input is not a second opinion."** A verdict that
    /// flipped because its input moved 6.4× while remaining 16,268× wrong flipped for a reason
    /// unrelated to the proposition. He is right, and this is what survives instead:
    ///
    /// - `transfer_ns` is **monotone non-decreasing in bytes** — asserted below, not assumed,
    ///   because the whole argument rests on it and a future `cost_ns` with a discount above
    ///   some size would silently void it;
    /// - the gate **claims** at the estimator's inflated 13,936,509,056 B;
    /// - the measured boundary is 856,720 B, which is **smaller**;
    /// - therefore the gate claims *a fortiori* on the true bytes.
    ///
    /// That is a bound, not an agreement: a number nobody trusts, used in the one direction
    /// where not trusting it is safe. §10.0.4 already preferred *the count* over the duration and
    /// *the ratio* over the count; this is the third form.
    ///
    /// **The licence is narrow, and this is the part expected to be forgotten first.** The sign
    /// is **not general**. Substituting 128 for an unknown dim *over*-counts on our decode
    /// window, where the real sequence extent is 1. On a long prefill the real extent exceeds
    /// 128, so 128 *under*-counts, the inequality reverses, and **the bound evaporates rather
    /// than weakening** — an under-counted transfer term makes the island look cheaper to move
    /// than it is, which is the direction that manufactures claims. Anyone touching
    /// [`Island::symbolic_boundary_slots`] or `ep.rs::slot_bytes` must preserve that asymmetry;
    /// the standing prefill falsifier is the check that would catch its loss.
    #[test]
    fn the_claim_survives_an_adversarial_inflation_of_the_term_opposing_it() {
        let model = &TransferModel::DISCRETE;

        // (1) Monotonicity of the opposing term, asserted rather than assumed.
        let mut previous = 0.0_f64;
        for bytes in [
            0_u64,
            856_720,
            1_000_000,
            13_936_509_056,
            89_199_100_032,
            u64::MAX / 4,
        ] {
            let island = Island {
                nodes: 355,
                anchors: 161,
                flops: 23_020_437_504,
                input_bytes: 0,
                output_bytes: bytes,
                symbolic_boundary_slots: 0,
            };
            let t = island.transfer_ns(model);
            assert!(
                t >= previous,
                "transfer_ns must be monotone non-decreasing in bytes; {bytes} B gave {t} after \
                 {previous}. The a-fortiori argument below is void the moment this stops holding."
            );
            previous = t;
        }

        // (2) The gate claims at the inflated figure.
        let inflated = Island::ESTIMATED_PHI35_DEV0_INTERNAL_EDGES_FIXED;
        let out = gate_islands(std::slice::from_ref(&inflated), model, &Policy::default());
        assert!(
            out[0].is_claim() && !out[0].is_override(),
            "the bound is only worth signing if the gate itself claims at the inflated figure; \
             an override would mean the economics never spoke. Got {:?}",
            out[0]
        );

        // (3) The true figure is smaller, by the residual this project has not closed.
        let measured = Island::MEASURED_PHI35_DEV0_REAL_BYTES.boundary_bytes();
        let inflated_bytes = inflated.boundary_bytes();
        assert!(measured < inflated_bytes);
        let inflation = inflated_bytes as f64 / measured as f64;
        assert!(
            inflation > 1.0e4,
            "the inflation the claim survives is the whole point of the bound; got {inflation}"
        );

        // (4) Therefore, a fortiori. Asserted directly so the conclusion is mechanical rather
        //     than a reader's inference from (1)–(3).
        let truthful = Island {
            output_bytes: Island::MEASURED_PHI35_DEV0_REAL_BYTES.output_bytes,
            input_bytes: Island::MEASURED_PHI35_DEV0_REAL_BYTES.input_bytes,
            ..inflated
        };
        assert!(
            truthful.transfer_ns(model) <= inflated.transfer_ns(model),
            "monotonicity was asserted in (1); this must follow. \
             truthful={} ns on {}+{} B, inflated={} ns on {}+{} B",
            truthful.transfer_ns(model),
            truthful.input_bytes,
            truthful.output_bytes,
            inflated.transfer_ns(model),
            inflated.input_bytes,
            inflated.output_bytes
        );
        let truthful_out = gate_islands(&[truthful], model, &Policy::default());
        assert!(
            truthful_out[0].is_claim() && !truthful_out[0].is_override(),
            "claiming at the inflated bytes must imply claiming at the smaller true bytes"
        );
    }

    /// The prefill falsifier, standing. **This test asserts the direction the bound depends on,
    /// and it is deliberately written so that it goes red if the sign is ever lost.**
    ///
    /// 128 substituted for an unknown dim is an over-count only while the real extent is below
    /// 128. Above it, the substitution under-counts the transfer term, the island looks cheaper
    /// to move than it is, and the gate manufactures a claim. The `boundary_is_fabricated()`
    /// disclosure exists precisely because the sign of the error is not knowable from the
    /// estimate alone.
    #[test]
    fn the_substituted_extent_under_counts_on_a_long_prefill_and_the_bound_evaporates() {
        const SUBSTITUTED: u64 = 128;
        let model = &TransferModel::DISCRETE;
        let per_slot_bytes = 2_u64; // f16, one element per extent unit

        let with_substitution = |real_extent: u64| {
            (
                Island {
                    nodes: 355,
                    anchors: 161,
                    flops: 23_020_437_504,
                    input_bytes: 0,
                    output_bytes: SUBSTITUTED * per_slot_bytes,
                    symbolic_boundary_slots: 1,
                },
                Island {
                    nodes: 355,
                    anchors: 161,
                    flops: 23_020_437_504,
                    input_bytes: 0,
                    output_bytes: real_extent * per_slot_bytes,
                    symbolic_boundary_slots: 0,
                },
            )
        };

        // Decode window: real extent 1. The substitution over-counts — the safe direction, and
        // the only direction in which the a-fortiori bound above can be signed.
        let (est, real) = with_substitution(1);
        assert!(
            est.transfer_ns(model) >= real.transfer_ns(model),
            "on the decode window the substitution must over-count, or the bound is unsigned"
        );

        // Long prefill: real extent 4096. The substitution under-counts by 32×, and the
        // inequality reverses. There is no bound here to weaken — it is gone.
        let (est, real) = with_substitution(4096);
        assert!(
            est.transfer_ns(model) < real.transfer_ns(model),
            "on a long prefill the substitution must under-count; if this assertion ever passes \
             by accident the sign asymmetry has been lost and the bound will be quoted in a \
             frame where it does not hold"
        );

        // And the estimate must say it is fabricated in both frames, because the sign of its
        // error is not knowable from the estimate itself.
        assert!(est.boundary_is_fabricated());
        assert!(!real.boundary_is_fabricated());
    }

    /// A sole-island override is not a licence to claim anything: the reason is preserved, so the
    /// artifact can say *why* the graph was kept despite failing.
    ///
    /// **This test asserts `TransferDominated` and it is consistent with
    /// `net_benefit_sole_island_overrides` going 1 → 0 on the shipping build.** Establishing that
    /// took a reader three steps, so it is written down here: this test forces
    /// `anchor_exemption: false` and feeds
    /// [`Island::ESTIMATED_PHI35_DEV0_INTERNAL_EDGES_COUNTED`] — the *pre-fix* estimate. The
    /// shipping path uses neither. It runs with the exemption on and the post-fix boundary, and
    /// on those inputs the gate claims outright, which is why the override count is 0.
    #[test]
    fn the_override_carries_the_verdict_it_overrode() {
        let out = gate_islands(
            &[Island::ESTIMATED_PHI35_DEV0_INTERNAL_EDGES_COUNTED],
            &TransferModel::DISCRETE,
            &Policy {
                anchor_exemption: false,
                ..Policy::default()
            },
        );
        match &out[0] {
            GateOutcome::SoleIslandOverride(RejectReason::TransferDominated {
                transfer_ns,
                compute_ns,
                margin,
            }) => {
                assert!(*compute_ns < *margin * *transfer_ns);
            }
            other => panic!("expected a transfer-dominated override, got {other:?}"),
        }
    }
}
