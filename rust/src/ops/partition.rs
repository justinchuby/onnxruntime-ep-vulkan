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
    /// How many of those are *anchors* — nodes carrying a resident weight at a schema-designated
    /// site, and so heavy enough to justify a boundary on their own (see [`is_anchor`]).
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
    /// `nodes` and `anchors` are counts from the CLAIM_LOG of the same run — 355 nodes claimed,
    /// of which **161 are anchors**, one per `MatMulNBits` node, each carrying its packed weight
    /// initializer at the schema-designated index 1. `PartitionStats` does **not** carry an
    /// anchor count (see `crate::trace::PartitionStats`, whose fields are node counts, FLOPs and
    /// boundary bytes); the anchor figure is a census of the claim log, and saying so is the
    /// point of this sentence.
    ///
    /// **The 32 `GroupQueryAttention` nodes in the same island are claimed and are not anchors**
    /// (issue #73). Every GQA input is an activation, a KV cache, a length, a per-head scalar or
    /// an `O(head_size)` norm gain — see [`WEIGHT_SITES`] — so no GQA node can carry a resident
    /// tensor at a designated weight site. A previous revision of this constant read
    /// `anchors: 193`, which was `161 + 32` under the op-name predicate this replaced.
    ///
    /// `flops` and the boundary total are the estimator's own numbers and are **unchanged**: the
    /// FLOP term is keyed on the heavy-op *family* ([`is_heavy_op`]), not on anchoring, so all
    /// 193 heavy-op nodes still score `2·3072·3072` exactly as they did.
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
        anchors: 161,
        flops: 23_020_437_504,
        input_bytes: 0,
        output_bytes: 89_199_100_032,
        symbolic_boundary_slots: 0,
    };

    /// The same island as re-estimated on 2026-08-01 after internal edges stopped being counted.
    ///
    /// Read out of the verbose `PartitionStats` summary on dev0 with the same model and the same
    /// binary that produced the 355 → 0 claim census. The number that changed is the boundary;
    /// nodes and FLOPs are unaffected because the fix touches only which outputs are charged to
    /// the boundary. The anchor count is not a `PartitionStats` field at all — it is the claim-log
    /// census carried over from the constant above.
    pub const ESTIMATED_PHI35_DEV0_INTERNAL_EDGES_FIXED: Island = Island {
        nodes: 355,
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

// ---------------------------------------------------------------------------------------
// Anchor eligibility — a node property, not an op name
// ---------------------------------------------------------------------------------------

/// What a **resident** tensor at one schema input index of a heavy-op family would be.
///
/// The classification is a reading of the *pinned schema's own text* — the shape it declares for
/// that input — and not of the operand's name. `router_weights` is called a weight and is
/// `(num_tokens, num_experts)`, so it is an activation; `cos_cache` is called a cache and is a
/// precomputed table. Names are the one thing this must not be keyed on, because keying anchor
/// eligibility on names is the defect being repaired.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SiteKind {
    /// A factor of the op's own product or convolution. Its extent is the reduction dimension
    /// times an output dimension, so a resident value here is read once per *output element* and
    /// pays for the island boundary many times over.
    Factor,
    /// Quantisation payload that reconstitutes a [`SiteKind::Factor`] — packed scales, zero
    /// points, group indices. Extent scales with the same reduction dimension.
    QuantPayload,
    /// A runtime activation: extent scales with batch, tokens or sequence length. A model that
    /// makes one of these an initializer has folded a constant input, not supplied a weight.
    Activation,
    /// An additive bias or a normalisation gain. `O(output channels)` or smaller, and it
    /// contributes one FLOP per output element — no reuse, nothing to amortise.
    BiasOrGain,
    /// A per-expert or per-head scalar. Weight-side in origin for some of these, but far too
    /// small to justify a boundary on its own; the tensor it scales is the site that does.
    PerGroupScalar,
    /// A table precomputed from position, indexed rather than multiplied.
    PrecomputedTable,
    /// Recurrent or KV-cache state carried between steps. Written every step, so it is traffic,
    /// not a weight.
    CachedState,
    /// A mask, a length vector or a beam-indirection buffer.
    MaskOrLength,
}

impl SiteKind {
    /// Whether a resident initializer at a site of this kind makes the node an anchor.
    ///
    /// Two kinds and only two. This is `const` and total so that adding a variant to
    /// [`SiteKind`] without deciding its polarity does not compile.
    pub const fn designates(self) -> bool {
        match self {
            SiteKind::Factor | SiteKind::QuantPayload => true,
            SiteKind::Activation
            | SiteKind::BiasOrGain
            | SiteKind::PerGroupScalar
            | SiteKind::PrecomputedTable
            | SiteKind::CachedState
            | SiteKind::MaskOrLength => false,
        }
    }

    /// Stable token for the dump surfaces (`epctl --dump-weight-sites`) and the tests that read
    /// them. Deliberately not `Debug`, which is free to change.
    pub const fn as_str(self) -> &'static str {
        match self {
            SiteKind::Factor => "factor",
            SiteKind::QuantPayload => "quant_payload",
            SiteKind::Activation => "activation",
            SiteKind::BiasOrGain => "bias_or_gain",
            SiteKind::PerGroupScalar => "per_group_scalar",
            SiteKind::PrecomputedTable => "precomputed_table",
            SiteKind::CachedState => "cached_state",
            SiteKind::MaskOrLength => "mask_or_length",
        }
    }
}

/// One schema input of one heavy-op family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WeightSite {
    /// `domain::op` exactly as [`crate::registry::NodeView::qualified_name`] renders it — the
    /// default domain has no prefix.
    pub qualified_op: &'static str,
    /// The operand's position in the pinned schema's input list.
    pub index: usize,
    /// The operand's name in the pinned schema. Carried so the row can be pinned against the
    /// schema itself rather than against a copy of this table.
    pub name: &'static str,
    /// What a resident tensor here would be.
    pub kind: SiteKind,
    /// Why, in the schema's own terms.
    pub reason: &'static str,
}

impl WeightSite {
    /// Whether a resident initializer here makes the node an anchor.
    pub const fn designated(&self) -> bool {
        self.kind.designates()
    }
}

/// **Every** input of every heavy-op family, in schema order, with its designation.
///
/// # Why every input and not just the designated ones
///
/// A table of only the designated indices cannot be checked against anything. This one can:
/// its `(qualified_op, index, name)` projection must equal the pinned schemas' input lists
/// exactly — same families, same order, same names, same *count* — so a schema that gains an
/// input, loses one, or reorders two is a red test rather than a silent misclassification of
/// whatever now sits at that index. `tests/ops/test_weight_sites.py` performs that comparison
/// against the installed `onnx` / `onnxruntime` packages, reading this table out of the built
/// binary via `epctl --dump-weight-sites --json`.
///
/// # Provenance
///
/// Derived 2026-08-09 from the pinned packages — `onnx` 1.22.0 (`onnx.defs.get_schema`) and
/// `onnxruntime` 1.28.0 (`onnxruntime.capi._pybind_state.get_all_operator_schema`), the same
/// release `third_party/onnxruntime/PROVENANCE.md` pins for the vendored headers (tag `v1.28.0`,
/// `da9b5e364c465de65c49d91e696cd6485270757f`, `ORT_API_VERSION` 28). Every `reason` below
/// quotes or paraphrases that schema's declared shape for that operand; none of them was read
/// off the operand's name.
///
/// # The criterion, stated once
///
/// An index is designated iff a resident tensor there is **a factor of the op's own arithmetic,
/// or the quantisation payload that reconstitutes one** — i.e. its extent scales with the op's
/// reduction dimensions, so it is read once per output element and a single node of this kind
/// amortises an island boundary by itself. Everything whose extent scales with *tokens* is an
/// activation; everything that is `O(output channels)` or smaller is a bias, a gain or a scalar
/// and contributes one FLOP per output element.
pub static WEIGHT_SITES: &[WeightSite] = &[
    // -- default domain -----------------------------------------------------------------
    WeightSite {
        qualified_op: "MatMul",
        index: 0,
        name: "A",
        kind: SiteKind::Factor,
        reason: "`N-dimensional matrix A` — either factor of a product may be the resident one; \
                 ONNX MatMul is symmetric in that respect and a model may emit the weight on \
                 the left",
    },
    WeightSite {
        qualified_op: "MatMul",
        index: 1,
        name: "B",
        kind: SiteKind::Factor,
        reason: "`N-dimensional matrix B` — the conventional weight side, and the one 36 of \
                 MiniLM-L6-v2's 48 MatMul nodes actually use",
    },
    WeightSite {
        qualified_op: "Gemm",
        index: 0,
        name: "A",
        kind: SiteKind::Factor,
        reason: "`(M, K)` (or `(K, M)` under transA) — a factor of the product, extent K in the \
                 reduction dimension",
    },
    WeightSite {
        qualified_op: "Gemm",
        index: 1,
        name: "B",
        kind: SiteKind::Factor,
        reason: "`(K, N)` (or `(N, K)` under transB) — a factor of the product; the classifier \
                 head's weight matrix on MobileNetV2-12",
    },
    WeightSite {
        qualified_op: "Gemm",
        index: 2,
        name: "C",
        kind: SiteKind::BiasOrGain,
        reason: "`unidirectional broadcastable to (M, N)`, added once per output element after \
                 the product — a bias, not a factor",
    },
    WeightSite {
        qualified_op: "Conv",
        index: 0,
        name: "X",
        kind: SiteKind::Activation,
        reason: "`(N x C x H x W)` where N is the batch size — extent scales with the batch",
    },
    WeightSite {
        qualified_op: "Conv",
        index: 1,
        name: "W",
        kind: SiteKind::Factor,
        reason: "`The weight tensor that will be used in the convolutions`, `(M x C/group x kH \
                 x kW)` — the reduction extent of every output element",
    },
    WeightSite {
        qualified_op: "Conv",
        index: 2,
        name: "B",
        kind: SiteKind::BiasOrGain,
        reason: "`Optional 1D bias`, `size of M` — one add per output element",
    },
    WeightSite {
        qualified_op: "ConvTranspose",
        index: 0,
        name: "X",
        kind: SiteKind::Activation,
        reason: "`(N x C x H x W)` where N is the batch size — the deconvolution's input \
                 activation, extent scales with the batch",
    },
    WeightSite {
        qualified_op: "ConvTranspose",
        index: 1,
        name: "W",
        kind: SiteKind::Factor,
        reason: "`The weight tensor that will be used in the convolutions`, `(C x M/group x kH \
                 x kW)`",
    },
    WeightSite {
        qualified_op: "ConvTranspose",
        index: 2,
        name: "B",
        kind: SiteKind::BiasOrGain,
        reason: "`Optional 1D bias`, `size of M` — one add per deconvolved output element",
    },
    // ONNX `Attention` (opset 24) is fused scaled-dot-product attention over *already projected*
    // Q/K/V. It has no projection weights at all, so it designates nothing and a node of it can
    // never anchor an island on its own. That is the correct reading of the schema and it is a
    // deliberate narrowing relative to the op-name list this replaced.
    WeightSite {
        qualified_op: "Attention",
        index: 0,
        name: "Q",
        kind: SiteKind::Activation,
        reason: "`Query tensor`, `(batch_size, q_num_heads, q_sequence_length, head_size)` — \
                 extent scales with the sequence",
    },
    WeightSite {
        qualified_op: "Attention",
        index: 1,
        name: "K",
        kind: SiteKind::Activation,
        reason: "`Key tensor`, extent scales with `kv_sequence_length`",
    },
    WeightSite {
        qualified_op: "Attention",
        index: 2,
        name: "V",
        kind: SiteKind::Activation,
        reason: "`Value tensor`, extent scales with `kv_sequence_length`",
    },
    WeightSite {
        qualified_op: "Attention",
        index: 3,
        name: "attn_mask",
        kind: SiteKind::MaskOrLength,
        reason: "`Attention mask`, broadcastable to `(batch, heads, q_len, total_len)`",
    },
    WeightSite {
        qualified_op: "Attention",
        index: 4,
        name: "past_key",
        kind: SiteKind::CachedState,
        reason: "`past state cache for key` — rewritten every step",
    },
    WeightSite {
        qualified_op: "Attention",
        index: 5,
        name: "past_value",
        kind: SiteKind::CachedState,
        reason: "`past state cache for value` — rewritten every step",
    },
    WeightSite {
        qualified_op: "Attention",
        index: 6,
        name: "nonpad_kv_seqlen",
        kind: SiteKind::MaskOrLength,
        reason: "`a vector of integers of shape (batch_size,)` counting non-padding tokens",
    },
    // -- com.microsoft ------------------------------------------------------------------
    WeightSite {
        qualified_op: "com.microsoft::MatMulNBits",
        index: 0,
        name: "A",
        kind: SiteKind::Activation,
        reason: "`The input tensor, not quantized.`",
    },
    WeightSite {
        qualified_op: "com.microsoft::MatMulNBits",
        index: 1,
        name: "B",
        kind: SiteKind::Factor,
        reason: "`Packed uint8 tensor of shape (N, k_blocks, blob_size)` — the quantised weight \
                 matrix itself; `k_blocks = ceil(K / block_size)` is the reduction extent",
    },
    WeightSite {
        qualified_op: "com.microsoft::MatMulNBits",
        index: 2,
        name: "scales",
        kind: SiteKind::QuantPayload,
        reason: "`Per-block scaling factors for dequantization with shape (N, k_blocks)` — one \
                 half of what reconstitutes B",
    },
    WeightSite {
        qualified_op: "com.microsoft::MatMulNBits",
        index: 3,
        name: "zero_points",
        kind: SiteKind::QuantPayload,
        reason: "`Per-block zero point for dequantization`, shape in `k_blocks` — the other half",
    },
    WeightSite {
        qualified_op: "com.microsoft::MatMulNBits",
        index: 4,
        name: "g_idx",
        kind: SiteKind::QuantPayload,
        reason: "`group_idx. This input is deprecated` — the act-order group mapping, extent K, \
                 and part of what reconstitutes B while it survives; the deprecation is recorded \
                 here so its eventual removal reds the schema audit rather than shifting the \
                 meaning of index 5",
    },
    WeightSite {
        qualified_op: "com.microsoft::MatMulNBits",
        index: 5,
        name: "bias",
        kind: SiteKind::BiasOrGain,
        reason: "`Bias to add to result. It should have shape [N].`",
    },
    // `com.microsoft::Attention` is the *unfused* contrib op: it does its own Q/K/V projection
    // and therefore does carry a weight matrix, unlike ONNX `Attention` and unlike
    // `MultiHeadAttention`. Exactly one designated site.
    WeightSite {
        qualified_op: "com.microsoft::Attention",
        index: 0,
        name: "input",
        kind: SiteKind::Activation,
        reason: "`(batch_size, sequence_length, input_hidden_size)`",
    },
    WeightSite {
        qualified_op: "com.microsoft::Attention",
        index: 1,
        name: "weights",
        kind: SiteKind::Factor,
        reason: "`Merged Q/K/V weights with shape (input_hidden_size, hidden_size + hidden_size \
                 + v_hidden_size)` — the projection matrix, reduction extent `input_hidden_size`",
    },
    WeightSite {
        qualified_op: "com.microsoft::Attention",
        index: 2,
        name: "bias",
        kind: SiteKind::BiasOrGain,
        reason: "`Bias tensor ... for input projection`, one add per projected element",
    },
    WeightSite {
        qualified_op: "com.microsoft::Attention",
        index: 3,
        name: "mask_index",
        kind: SiteKind::MaskOrLength,
        reason: "`Attention mask`, extent scales with the sequence",
    },
    WeightSite {
        qualified_op: "com.microsoft::Attention",
        index: 4,
        name: "past",
        kind: SiteKind::CachedState,
        reason: "`past state for key and value`, `(2, batch, heads, past_sequence_length, \
                 head_size)`",
    },
    WeightSite {
        qualified_op: "com.microsoft::Attention",
        index: 5,
        name: "attention_bias",
        kind: SiteKind::BiasOrGain,
        reason: "`additional add to QxK'` — one add per score element, no reuse across outputs",
    },
    WeightSite {
        qualified_op: "com.microsoft::Attention",
        index: 6,
        name: "past_sequence_length",
        kind: SiteKind::MaskOrLength,
        reason: "a length scalar under `past_present_share_buffer`",
    },
    // `MultiHeadAttention` takes Q/K/V **already projected** — the projection weights belong to
    // the MatMul/MatMulNBits nodes feeding it. Zero designated sites.
    WeightSite {
        qualified_op: "com.microsoft::MultiHeadAttention",
        index: 0,
        name: "query",
        kind: SiteKind::Activation,
        reason: "`(batch_size, sequence_length, hidden_size)`, or packed QKV",
    },
    WeightSite {
        qualified_op: "com.microsoft::MultiHeadAttention",
        index: 1,
        name: "key",
        kind: SiteKind::Activation,
        reason: "`(batch_size, kv_sequence_length, hidden_size)`, or packed KV, or past_key",
    },
    WeightSite {
        qualified_op: "com.microsoft::MultiHeadAttention",
        index: 2,
        name: "value",
        kind: SiteKind::Activation,
        reason: "`(batch_size, kv_sequence_length, v_hidden_size)`, or past_value",
    },
    WeightSite {
        qualified_op: "com.microsoft::MultiHeadAttention",
        index: 3,
        name: "bias",
        kind: SiteKind::BiasOrGain,
        reason: "`Bias tensor with shape (hidden_size + hidden_size + v_hidden_size) from input \
                 projection` — the projection's bias; the projection's *matrix* is not an input \
                 of this op",
    },
    WeightSite {
        qualified_op: "com.microsoft::MultiHeadAttention",
        index: 4,
        name: "key_padding_mask",
        kind: SiteKind::MaskOrLength,
        reason: "`Key padding mask`, every declared shape scales with batch or sequence",
    },
    WeightSite {
        qualified_op: "com.microsoft::MultiHeadAttention",
        index: 5,
        name: "attention_bias",
        kind: SiteKind::BiasOrGain,
        reason: "`bias added to QxK'` — one add per score element",
    },
    WeightSite {
        qualified_op: "com.microsoft::MultiHeadAttention",
        index: 6,
        name: "past_key",
        kind: SiteKind::CachedState,
        reason: "`past state for key`, rewritten every step",
    },
    WeightSite {
        qualified_op: "com.microsoft::MultiHeadAttention",
        index: 7,
        name: "past_value",
        kind: SiteKind::CachedState,
        reason: "`past state for value`, rewritten every step",
    },
    WeightSite {
        qualified_op: "com.microsoft::MultiHeadAttention",
        index: 8,
        name: "past_sequence_length",
        kind: SiteKind::MaskOrLength,
        reason: "`The past_sequence_length buffer sharing is used with`",
    },
    WeightSite {
        qualified_op: "com.microsoft::MultiHeadAttention",
        index: 9,
        name: "cache_indirection",
        kind: SiteKind::MaskOrLength,
        reason: "`[batch_size, beam_width, max_sequence_length]` beam indirection — an index \
                 buffer",
    },
    // `GroupQueryAttention` designates **zero** weight sites, and this is the single most
    // consequential row-block in the table: it is why Phi-3.5-mini-int4 has **161** anchors and
    // not 193. GQA consumes Q/K/V that its neighbouring `MatMulNBits` nodes projected; every one
    // of its remaining inputs is a cache, a length, a per-head scalar or an `O(head_size)` norm
    // gain. Its 32 nodes are still *claimed* — anchor eligibility and claim eligibility are
    // different questions and `docs/OP_COVERAGE.md` reports them separately.
    WeightSite {
        qualified_op: "com.microsoft::GroupQueryAttention",
        index: 0,
        name: "query",
        kind: SiteKind::Activation,
        reason: "`Query with shape (batch_size, sequence_length, hidden_size)`, or packed QKV of \
                 width `num_heads * head_size + 2 * kv_num_heads * head_size`",
    },
    WeightSite {
        qualified_op: "com.microsoft::GroupQueryAttention",
        index: 1,
        name: "key",
        kind: SiteKind::Activation,
        reason: "`Key with shape (batch_size, kv_sequence_length, kv_hidden_size)`",
    },
    WeightSite {
        qualified_op: "com.microsoft::GroupQueryAttention",
        index: 2,
        name: "value",
        kind: SiteKind::Activation,
        reason: "`Value with shape (batch_size, kv_sequence_length, kv_hidden_size)`",
    },
    WeightSite {
        qualified_op: "com.microsoft::GroupQueryAttention",
        index: 3,
        name: "past_key",
        kind: SiteKind::CachedState,
        reason: "`past state key`, shared with `present_key` — written every step",
    },
    WeightSite {
        qualified_op: "com.microsoft::GroupQueryAttention",
        index: 4,
        name: "past_value",
        kind: SiteKind::CachedState,
        reason: "`past state value`, shared with `present_value` — written every step",
    },
    WeightSite {
        qualified_op: "com.microsoft::GroupQueryAttention",
        index: 5,
        name: "seqlens_k",
        kind: SiteKind::MaskOrLength,
        reason: "`1D Tensor of shape (batch_size)`",
    },
    WeightSite {
        qualified_op: "com.microsoft::GroupQueryAttention",
        index: 6,
        name: "total_sequence_length",
        kind: SiteKind::MaskOrLength,
        reason: "`Scalar tensor equivalent to the maximum total sequence length`",
    },
    WeightSite {
        qualified_op: "com.microsoft::GroupQueryAttention",
        index: 7,
        name: "cos_cache",
        kind: SiteKind::PrecomputedTable,
        reason: "`(max_sequence_length, head_size / 2)` rotary table — indexed by position, not \
                 multiplied against a reduction extent; resident by construction on every model \
                 that uses rotary embeddings, which is exactly why designating it would make \
                 the anchor property vacuous",
    },
    WeightSite {
        qualified_op: "com.microsoft::GroupQueryAttention",
        index: 8,
        name: "sin_cache",
        kind: SiteKind::PrecomputedTable,
        reason: "`(max_sequence_length, head_size / 2)` rotary table — see `cos_cache`",
    },
    WeightSite {
        qualified_op: "com.microsoft::GroupQueryAttention",
        index: 9,
        name: "position_ids",
        kind: SiteKind::MaskOrLength,
        reason: "`(batch_size, sequence_length)` index vector",
    },
    WeightSite {
        qualified_op: "com.microsoft::GroupQueryAttention",
        index: 10,
        name: "attention_bias",
        kind: SiteKind::BiasOrGain,
        reason: "`additional add to QxK'` — one add per score element",
    },
    WeightSite {
        qualified_op: "com.microsoft::GroupQueryAttention",
        index: 11,
        name: "head_sink",
        kind: SiteKind::PerGroupScalar,
        reason: "`1D tensor with shape (num_heads)`, one smoothing term per head",
    },
    WeightSite {
        qualified_op: "com.microsoft::GroupQueryAttention",
        index: 12,
        name: "k_scale",
        kind: SiteKind::PerGroupScalar,
        reason: "`Scale tensor for past_key` — quantisation metadata for a *cache*, not for a \
                 weight",
    },
    WeightSite {
        qualified_op: "com.microsoft::GroupQueryAttention",
        index: 13,
        name: "v_scale",
        kind: SiteKind::PerGroupScalar,
        reason: "`Scale tensor for past_value` — see `k_scale`",
    },
    WeightSite {
        qualified_op: "com.microsoft::GroupQueryAttention",
        index: 14,
        name: "q_norm_weight",
        kind: SiteKind::BiasOrGain,
        reason: "`Optional 1D tensor of shape (head_size)` — an RMS-norm gain applied elementwise \
                 to Q. Learned, resident, and still `O(head_size)`: one multiply per element of \
                 Q with no reuse across outputs, so it amortises nothing",
    },
    WeightSite {
        qualified_op: "com.microsoft::GroupQueryAttention",
        index: 15,
        name: "k_norm_weight",
        kind: SiteKind::BiasOrGain,
        reason: "`Optional 1D tensor of shape (head_size). See q_norm_weight.`",
    },
    // `QMoE` designates **nine** sites: the packed expert weights, their scales and their zero
    // points, for each of fc1/fc2/fc3. Not four, and not the twelve a name-based reading would
    // produce — `router_weights`, the two global scales and the four activation scales are all
    // excluded, each for a different and separately stated reason.
    WeightSite {
        qualified_op: "com.microsoft::QMoE",
        index: 0,
        name: "input",
        kind: SiteKind::Activation,
        reason: "`(num_tokens, hidden_size)` or `(batch_size, sequence_length, hidden_size)`",
    },
    WeightSite {
        qualified_op: "com.microsoft::QMoE",
        index: 1,
        name: "router_probs",
        kind: SiteKind::Activation,
        reason: "`2D tensor with shape (num_tokens, num_experts)` — extent scales with tokens",
    },
    WeightSite {
        qualified_op: "com.microsoft::QMoE",
        index: 2,
        name: "fc1_experts_weights",
        kind: SiteKind::Factor,
        reason: "`(num_experts, fusion_size * inter_size, hidden_size / pack_size)` — the packed \
                 FC1 weight matrix, reduction extent `hidden_size`",
    },
    WeightSite {
        qualified_op: "com.microsoft::QMoE",
        index: 3,
        name: "fc1_scales",
        kind: SiteKind::QuantPayload,
        reason: "`Optional weight scales` for FC1, extent `(num_experts, fusion_size * \
                 inter_size[, hidden_size / block_size])`",
    },
    WeightSite {
        qualified_op: "com.microsoft::QMoE",
        index: 4,
        name: "fc1_experts_bias",
        kind: SiteKind::BiasOrGain,
        reason: "`2D optional tensor with shape (num_experts, fusion_size * inter_size)` — a bias",
    },
    WeightSite {
        qualified_op: "com.microsoft::QMoE",
        index: 5,
        name: "fc2_experts_weights",
        kind: SiteKind::Factor,
        reason: "`(num_experts, hidden_size, inter_size / pack_size)` — the packed FC2 weight \
                 matrix, reduction extent `inter_size`",
    },
    WeightSite {
        qualified_op: "com.microsoft::QMoE",
        index: 6,
        name: "fc2_scales",
        kind: SiteKind::QuantPayload,
        reason: "`Optional weight scales` for FC2",
    },
    WeightSite {
        qualified_op: "com.microsoft::QMoE",
        index: 7,
        name: "fc2_experts_bias",
        kind: SiteKind::BiasOrGain,
        reason: "`2D optional tensor with shape (num_experts, hidden_size)` — a bias",
    },
    WeightSite {
        qualified_op: "com.microsoft::QMoE",
        index: 8,
        name: "fc3_experts_weights",
        kind: SiteKind::Factor,
        reason: "`(num_experts, inter_size, hidden_size / pack_size)` — the packed FC3 (gate) \
                 weight matrix",
    },
    WeightSite {
        qualified_op: "com.microsoft::QMoE",
        index: 9,
        name: "fc3_scales",
        kind: SiteKind::QuantPayload,
        reason: "`Optional weight scales` for FC3",
    },
    WeightSite {
        qualified_op: "com.microsoft::QMoE",
        index: 10,
        name: "fc3_experts_bias",
        kind: SiteKind::BiasOrGain,
        reason: "`2D optional tensor with shape (num_experts, inter_size)` — a bias",
    },
    WeightSite {
        qualified_op: "com.microsoft::QMoE",
        index: 11,
        name: "fc1_zero_points",
        kind: SiteKind::QuantPayload,
        reason: "`(num_experts, fusion_size * inter_size / pack_size)` — FC1 dequantisation \
                 payload",
    },
    WeightSite {
        qualified_op: "com.microsoft::QMoE",
        index: 12,
        name: "fc2_zero_points",
        kind: SiteKind::QuantPayload,
        reason: "`(num_experts, hidden_size / pack_size)` — FC2 dequantisation payload",
    },
    WeightSite {
        qualified_op: "com.microsoft::QMoE",
        index: 13,
        name: "fc3_zero_points",
        kind: SiteKind::QuantPayload,
        reason: "`(num_experts, inter_size / pack_size)` — FC3 dequantisation payload",
    },
    WeightSite {
        qualified_op: "com.microsoft::QMoE",
        index: 14,
        name: "router_weights",
        kind: SiteKind::Activation,
        reason: "named a weight and is not one: `2D optional tensor with shape (num_tokens, \
                 num_experts)`, `used for aggregating expert outputs` — its extent scales with \
                 tokens. The clearest case in the table for reading shapes rather than names",
    },
    WeightSite {
        qualified_op: "com.microsoft::QMoE",
        index: 15,
        name: "fc1_global_scale",
        kind: SiteKind::PerGroupScalar,
        reason: "`1D optional tensor with shape (num_experts,). Per-expert global weight scale \
                 for FC1.` — weight-side in origin but `O(num_experts)`; the tensor it scales \
                 (index 2) is the site that carries FC1",
    },
    WeightSite {
        qualified_op: "com.microsoft::QMoE",
        index: 16,
        name: "fc2_global_scale",
        kind: SiteKind::PerGroupScalar,
        reason: "`(num_experts,)` per-expert global weight scale for FC2 — see \
                 `fc1_global_scale`",
    },
    WeightSite {
        qualified_op: "com.microsoft::QMoE",
        index: 17,
        name: "fc1_act_scale",
        kind: SiteKind::Activation,
        reason: "`Activation scale for FC1 FP8 activation modes.` — activation-side by the \
                 schema's own word",
    },
    WeightSite {
        qualified_op: "com.microsoft::QMoE",
        index: 18,
        name: "fc2_act_scale",
        kind: SiteKind::Activation,
        reason: "`Activation scale for FC2 FP8 activation modes.`",
    },
    WeightSite {
        qualified_op: "com.microsoft::QMoE",
        index: 19,
        name: "fc1_act_block_scale",
        kind: SiteKind::Activation,
        reason: "`MXFP activation block-scale tensor for FC1` — one block scale per activation \
                 block, so its extent scales with tokens",
    },
    WeightSite {
        qualified_op: "com.microsoft::QMoE",
        index: 20,
        name: "fc2_act_block_scale",
        kind: SiteKind::Activation,
        reason: "`MXFP activation block-scale tensor for FC2` — see `fc1_act_block_scale`",
    },
    // `LinearAttention` designates zero sites: every one of its six inputs is declared with a
    // leading `(B, T, ...)` or is the recurrent state.
    WeightSite {
        qualified_op: "com.microsoft::LinearAttention",
        index: 0,
        name: "query",
        kind: SiteKind::Activation,
        reason: "`3D packed shape (B, T, H_q * d_k)`",
    },
    WeightSite {
        qualified_op: "com.microsoft::LinearAttention",
        index: 1,
        name: "key",
        kind: SiteKind::Activation,
        reason: "`3D packed shape (B, T, H_kv * d_k)`",
    },
    WeightSite {
        qualified_op: "com.microsoft::LinearAttention",
        index: 2,
        name: "value",
        kind: SiteKind::Activation,
        reason: "`3D packed shape (B, T, H_kv * d_v)`",
    },
    WeightSite {
        qualified_op: "com.microsoft::LinearAttention",
        index: 3,
        name: "past_state",
        kind: SiteKind::CachedState,
        reason: "`Recurrent state from previous step with shape (B, H_kv, d_k, d_v)`",
    },
    WeightSite {
        qualified_op: "com.microsoft::LinearAttention",
        index: 4,
        name: "decay",
        kind: SiteKind::Activation,
        reason: "`Exponential decay gate in log-space`, `(B, T, H_kv * d_k)` or `(B, T, H_kv)` — \
                 computed per token",
    },
    WeightSite {
        qualified_op: "com.microsoft::LinearAttention",
        index: 5,
        name: "beta",
        kind: SiteKind::Activation,
        reason: "`Update rate (sigmoid output)`, `(B, T, H_kv)` or `(B, T, 1)`",
    },
];

/// The pinned schema row for one operand of one op, or `None`.
///
/// `None` means one of two things and both fail closed: the op is not in the heavy-op inventory
/// at all, or `index` is past the end of that op's pinned schema. A caller that gets `None`
/// must treat the operand as incapable of anchoring.
pub fn classify_weight_operand(qualified_op: &str, index: usize) -> Option<&'static WeightSite> {
    WEIGHT_SITES
        .iter()
        .find(|s| s.qualified_op == qualified_op && s.index == index)
}

/// Whether a resident initializer at `index` of `qualified_op` designates a weight site.
pub fn is_designated_weight_site(qualified_op: &str, index: usize) -> bool {
    classify_weight_operand(qualified_op, index).is_some_and(WeightSite::designated)
}

/// Every pinned schema row for one op, in schema order. Empty for ops outside the inventory.
pub fn weight_sites_for(qualified_op: &str) -> Vec<&'static WeightSite> {
    WEIGHT_SITES
        .iter()
        .filter(|s| s.qualified_op == qualified_op)
        .collect()
}

/// The heavy-op families, in table order, without repeats.
pub fn heavy_op_families() -> Vec<&'static str> {
    let mut out: Vec<&'static str> = Vec::new();
    for site in WEIGHT_SITES {
        if out.last() != Some(&site.qualified_op) {
            out.push(site.qualified_op);
        }
    }
    out
}

/// Whether `qualified_op` is a heavy-op family — one whose arithmetic is large enough that the
/// FLOP estimator scores it as a matmul rather than as an elementwise pass.
///
/// **This is not the anchor predicate and must never be used as one.** It answers a question
/// about arithmetic shape, which is a property of the op; anchoring is a question about whether
/// a boundary is amortised, which is a property of the *node*. `rust/src/ep.rs` uses this one
/// for its FLOP estimate — an activation⊗activation `MatMul` really does perform `2·M·K·N`
/// FLOPs, and pretending otherwise would change the economics arithmetic in a way issue #73
/// explicitly does not ask for — and [`is_anchor`] for anchoring.
pub fn is_heavy_op(qualified_op: &str) -> bool {
    WEIGHT_SITES.iter().any(|s| s.qualified_op == qualified_op)
}

/// Whether this **node** is an anchor: a heavy-op family node carrying a resident initializer at
/// at least one schema-designated weight site.
///
/// # Why this is not `is_anchor(op_name)` any more
///
/// It used to be. `matches!(qualified_op, "MatMul" | ...)` claims every `MatMul` in a graph,
/// and on MiniLM-L6-v2 twelve of the forty-eight `MatMul` nodes are the attention `QKᵀ` and
/// `AV` batched products, whose *both* operands are runtime activations. Six of them formed
/// one-node islands, and the anchor exemption in [`evaluate`] claimed all six: 983,040 B in and
/// 196,608 B out to buy 0.013 GFLOP, ≈ 11 FLOP/byte at batch 1 and sequence 128. That is the
/// precise shape the economics gate exists to reject, and the name-keyed anchor predicate was
/// the reason it never got to answer.
///
/// The warrant for the exemption was always *an anchor is heavy enough to justify a boundary on
/// its own*, and that warrant is about **weights**: a resident matrix is uploaded once and read
/// once per output element for the life of the session, so it amortises the boundary by
/// construction. Two runtime activations amortise nothing. So the predicate is now the warrant.
///
/// # Arguments
///
/// `resident_inputs[i]` is whether operand `i` of this node reads a constant initializer.
/// Shorter than the node's operand list is fine — absent entries are treated as non-resident,
/// which is the closed direction. Longer is fine too: indices past the pinned schema classify
/// as `None` and cannot designate.
///
/// Every uncertain reading must arrive here as `false`: a missing operand, a null slot, an
/// out-of-range index, an ORT status error, or a runtime input. The caller in `ep.rs` gets that
/// from [`crate::registry::NodeView::input_is_constant`], which returns `false` on every one of
/// those conditions, and is the same reading the island's boundary-byte accounting uses — so a
/// value this predicate declines to call resident is also a value that *is* charged as boundary
/// traffic. The two halves of the economics cannot disagree about which tensors are weights.
pub fn is_anchor(qualified_op: &str, resident_inputs: &[bool]) -> bool {
    resident_inputs
        .iter()
        .enumerate()
        .any(|(i, &resident)| resident && is_designated_weight_site(qualified_op, i))
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
    /// Whether an island containing at least one anchor node ([`is_anchor`]) skips the economics
    /// check.
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
///    **Anchor-containing islands are exempt from gate 2**: a node carrying a resident weight at
///    a schema-designated site is by definition heavy enough to justify a boundary on its own —
///    that is the design invariant of [`is_anchor`], and since issue #73 the predicate states it
///    rather than assuming it from the op's name. The provisional `TransferModel` constants are
///    calibrated against real model execution and may not reflect isolated unit-test input sizes;
///    applying the economic check to anchors would reject them when tested in isolation, which
///    contradicts the stated design intent ("a single MatMul on LLM-sized weights always is worth
///    it"). A `MatMul` on *no* weights at all is not that op, and no longer takes the exemption.
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

    // ---------------------------------------------------------------------------------------
    // Anchor eligibility (issue #73)
    //
    // Three layers, and they check different things on purpose:
    //
    //   1. `HELD_OUT_SITES` below — a hand-maintained second copy of the table's *semantics*
    //      (family, index, name, kind) plus a tamper seal over each `reason`. It is a change
    //      detector: nothing may move in `WEIGHT_SITES` without a human re-affirming it here.
    //   2. `tests/ops/test_weight_sites.py` — the *independent* pin. It reads the shipped table
    //      out of the built binary (`epctl --dump-weight-sites --json`) and compares family
    //      membership, operand order, operand names and operand *count* against the pinned
    //      `onnx` / `onnxruntime` packages themselves. Neither this table nor the list below is
    //      an input to it.
    //   3. `the_mutations_that_must_red` — feeds eight held-out mutations through the same
    //      `disagreement_with_held_out` the production test uses, and requires every one of them
    //      to be caught. A check that cannot fail is not a check.
    // ---------------------------------------------------------------------------------------

    /// The held-out expectation, written in a different medium from the table it guards: one
    /// whitespace-separated line per operand, `family index name kind reason-seal`.
    ///
    /// The seal is FNV-1a-64 of the `reason` string. It exists so that *swapping two rows'
    /// justifications* — which changes no name, no index and no kind — is still a red test.
    /// Rewording a justification is expected to require editing this list; that is the point.
    const HELD_OUT_SITES: &str = "\
MatMul 0 A factor a4e2c01724f12b13
MatMul 1 B factor 1793f9f25a769803
Gemm 0 A factor 37e42c16fed227a1
Gemm 1 B factor 644a72f52032970a
Gemm 2 C bias_or_gain a13be18c29fdbb13
Conv 0 X activation 3fb4e2b3fa3519eb
Conv 1 W factor 4a1df61a7a0b7334
Conv 2 B bias_or_gain 0ed8da9b5660b239
ConvTranspose 0 X activation c9db15db9ec746f1
ConvTranspose 1 W factor ef43117c81208ab5
ConvTranspose 2 B bias_or_gain 090d5b054c897fb2
Attention 0 Q activation de3da0e84fb6394e
Attention 1 K activation 85dc31635eeefdcc
Attention 2 V activation 488e6d49e4905126
Attention 3 attn_mask mask_or_length 2775cf9090a73408
Attention 4 past_key cached_state 97430fbe84d627d9
Attention 5 past_value cached_state 0cbb2b875bdda62f
Attention 6 nonpad_kv_seqlen mask_or_length e3aa20a6b1985858
com.microsoft::MatMulNBits 0 A activation 32a30058fad145e7
com.microsoft::MatMulNBits 1 B factor 195c23be97ef1ba0
com.microsoft::MatMulNBits 2 scales quant_payload c24086730a172e46
com.microsoft::MatMulNBits 3 zero_points quant_payload 990b94f75fdea173
com.microsoft::MatMulNBits 4 g_idx quant_payload 03f935c58cfd39d5
com.microsoft::MatMulNBits 5 bias bias_or_gain 7dfd845245694501
com.microsoft::Attention 0 input activation 7eeca8495207a075
com.microsoft::Attention 1 weights factor f87e9f51b869d8e3
com.microsoft::Attention 2 bias bias_or_gain d7ec9a18a6b1c7db
com.microsoft::Attention 3 mask_index mask_or_length 208cfa1c2cc978d6
com.microsoft::Attention 4 past cached_state 05c81f549e800c35
com.microsoft::Attention 5 attention_bias bias_or_gain 5b6e09e9ff7bc6df
com.microsoft::Attention 6 past_sequence_length mask_or_length d41646ff5b5b3327
com.microsoft::MultiHeadAttention 0 query activation 84301079e21c142f
com.microsoft::MultiHeadAttention 1 key activation d4a3c7f908214c81
com.microsoft::MultiHeadAttention 2 value activation e1f0bc631e0bcadc
com.microsoft::MultiHeadAttention 3 bias bias_or_gain abc8895a1da2fcbc
com.microsoft::MultiHeadAttention 4 key_padding_mask mask_or_length 947d0f25d52de862
com.microsoft::MultiHeadAttention 5 attention_bias bias_or_gain 444e801813292f70
com.microsoft::MultiHeadAttention 6 past_key cached_state cde57b963ad99cc9
com.microsoft::MultiHeadAttention 7 past_value cached_state f36c1016ec733707
com.microsoft::MultiHeadAttention 8 past_sequence_length mask_or_length e0b978ae139fde82
com.microsoft::MultiHeadAttention 9 cache_indirection mask_or_length bbb8126a60b12dc8
com.microsoft::GroupQueryAttention 0 query activation 2699b02ead578d32
com.microsoft::GroupQueryAttention 1 key activation 201a4d57d863d228
com.microsoft::GroupQueryAttention 2 value activation 07500a79e2386a7e
com.microsoft::GroupQueryAttention 3 past_key cached_state 26ce060512e8b3f9
com.microsoft::GroupQueryAttention 4 past_value cached_state e0f6a81f7f7fc4a1
com.microsoft::GroupQueryAttention 5 seqlens_k mask_or_length 63e7a917dce60074
com.microsoft::GroupQueryAttention 6 total_sequence_length mask_or_length 52cc4eae9d226fe3
com.microsoft::GroupQueryAttention 7 cos_cache precomputed_table 702b875f51820e2e
com.microsoft::GroupQueryAttention 8 sin_cache precomputed_table 4ed9ac4f1c30b26e
com.microsoft::GroupQueryAttention 9 position_ids mask_or_length 2d3480edf79e437b
com.microsoft::GroupQueryAttention 10 attention_bias bias_or_gain d8cf2703ecfc01c7
com.microsoft::GroupQueryAttention 11 head_sink per_group_scalar 31ab9b91882f6d54
com.microsoft::GroupQueryAttention 12 k_scale per_group_scalar add019bd85a406d9
com.microsoft::GroupQueryAttention 13 v_scale per_group_scalar 1a19340553b992f8
com.microsoft::GroupQueryAttention 14 q_norm_weight bias_or_gain f1d5cee4a6633324
com.microsoft::GroupQueryAttention 15 k_norm_weight bias_or_gain bd8957657edfba12
com.microsoft::QMoE 0 input activation c192105d13c456e7
com.microsoft::QMoE 1 router_probs activation 0a51206b391eb840
com.microsoft::QMoE 2 fc1_experts_weights factor 1e446264d078992b
com.microsoft::QMoE 3 fc1_scales quant_payload ca7047ef5ca6f81a
com.microsoft::QMoE 4 fc1_experts_bias bias_or_gain af9ebd37f35a1f02
com.microsoft::QMoE 5 fc2_experts_weights factor 0db69f8af69a58aa
com.microsoft::QMoE 6 fc2_scales quant_payload e86a952cdc7d793c
com.microsoft::QMoE 7 fc2_experts_bias bias_or_gain 47f9124bf60d5ebc
com.microsoft::QMoE 8 fc3_experts_weights factor df78407d85b0bf78
com.microsoft::QMoE 9 fc3_scales quant_payload e86a962cdc7d7aef
com.microsoft::QMoE 10 fc3_experts_bias bias_or_gain d88164cefcb2789c
com.microsoft::QMoE 11 fc1_zero_points quant_payload 36fccf03fc344fe3
com.microsoft::QMoE 12 fc2_zero_points quant_payload c766b97bd2dcf096
com.microsoft::QMoE 13 fc3_zero_points quant_payload 209697fbc8d573af
com.microsoft::QMoE 14 router_weights activation 12566694ffaeb578
com.microsoft::QMoE 15 fc1_global_scale per_group_scalar c6e142177ee166e1
com.microsoft::QMoE 16 fc2_global_scale per_group_scalar bdd6432850947719
com.microsoft::QMoE 17 fc1_act_scale activation c9ee385c7e49a36f
com.microsoft::QMoE 18 fc2_act_scale activation 91b419bdfa0df39b
com.microsoft::QMoE 19 fc1_act_block_scale activation 9919092b4b00f7bc
com.microsoft::QMoE 20 fc2_act_block_scale activation 6b755ae615768f6e
com.microsoft::LinearAttention 0 query activation 6295d4702e0b3bc8
com.microsoft::LinearAttention 1 key activation 4be9119fa2ef4a8a
com.microsoft::LinearAttention 2 value activation 36a956a02825b725
com.microsoft::LinearAttention 3 past_state cached_state a414c724e348327c
com.microsoft::LinearAttention 4 decay activation e8cab253b8b4cbf5
com.microsoft::LinearAttention 5 beta activation 8b5fcacf882c20d8
";

    /// FNV-1a 64. Chosen because it is four lines and needs no dependency; this is a tamper seal
    /// over a doc string, not a security boundary.
    fn seal(s: &str) -> String {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in s.as_bytes() {
            h ^= u64::from(*b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        format!("{h:016x}")
    }

    /// Render a table in the held-out list's own vocabulary.
    fn render(rows: &[WeightSite]) -> Vec<String> {
        rows.iter()
            .map(|s| {
                format!(
                    "{} {} {} {} {}",
                    s.qualified_op,
                    s.index,
                    s.name,
                    s.kind.as_str(),
                    seal(s.reason)
                )
            })
            .collect()
    }

    /// The one production assertion, factored so the mutation harness can call it.
    ///
    /// Returns `Ok(())` when `rows` matches [`HELD_OUT_SITES`] exactly — same length, same order,
    /// same content — and a human-readable disagreement otherwise.
    fn disagreement_with_held_out(rows: &[WeightSite]) -> Result<(), String> {
        let expected: Vec<&str> = HELD_OUT_SITES.lines().filter(|l| !l.is_empty()).collect();
        let actual = render(rows);
        if expected.len() != actual.len() {
            return Err(format!(
                "row count: held-out list has {}, table has {}",
                expected.len(),
                actual.len()
            ));
        }
        for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
            if e != a {
                return Err(format!("row {i}: held-out `{e}` != table `{a}`"));
            }
        }
        Ok(())
    }

    /// Every operand of every heavy-op family is pinned, in order, by family, index, name, kind
    /// and a seal over its justification.
    ///
    /// This is the assertion the eight mutations below must each break.
    #[test]
    fn the_weight_site_table_matches_its_held_out_pin() {
        if let Err(why) = disagreement_with_held_out(WEIGHT_SITES) {
            panic!(
                "`WEIGHT_SITES` disagrees with the held-out pin: {why}\n\n\
                 If the pinned schemas genuinely changed, `tests/ops/test_weight_sites.py` is \
                 the check that says so, and both this list and the table must be updated \
                 together with the new schema's text quoted in the `reason`. If they did not, \
                 this is the defect the pin exists to catch."
            );
        }
    }

    /// The eight held-out mutations, each of which must be caught.
    ///
    /// Named after what a careless edit would actually look like. A check that no mutation can
    /// break is a check that proves nothing about the thing it guards (R10).
    #[test]
    fn the_mutations_that_must_red() {
        let base: Vec<WeightSite> = WEIGHT_SITES.to_vec();
        assert!(
            disagreement_with_held_out(&base).is_ok(),
            "the unmutated table must be green, or the controls below prove nothing"
        );

        let mut cases: Vec<(&str, Vec<WeightSite>)> = Vec::new();

        // 1. Delete the final row — the classic off-by-one on a hand-written table.
        let mut m = base.clone();
        m.pop();
        cases.push(("delete final row", m));

        // 2. A schema gains an input: append one to the last family.
        let mut m = base.clone();
        let last = *base.last().unwrap();
        m.push(WeightSite {
            index: last.index + 1,
            name: "gate",
            kind: SiteKind::Activation,
            ..last
        });
        cases.push(("schema gains an input", m));

        // 3. Swap two indices within a family.
        let mut m = base.clone();
        let (a, b) = (
            m.iter()
                .position(|s| s.qualified_op == "Gemm" && s.index == 1)
                .unwrap(),
            m.iter()
                .position(|s| s.qualified_op == "Gemm" && s.index == 2)
                .unwrap(),
        );
        let (ia, ib) = (m[a].index, m[b].index);
        m[a].index = ib;
        m[b].index = ia;
        cases.push(("swap two indices", m));

        // 4. Swap two names — the mutation that stayed green in the revision this replaces.
        let mut m = base.clone();
        let (na, nb) = (m[a].name, m[b].name);
        m[a].name = nb;
        m[b].name = na;
        cases.push(("swap two names", m));

        // 5. Swap two justifications, leaving names, indices and kinds alone.
        let mut m = base.clone();
        let (ra, rb) = (m[a].reason, m[b].reason);
        m[a].reason = rb;
        m[b].reason = ra;
        cases.push(("swap two justifications", m));

        // 6. Designate an activation. This is the mutation that would resurrect issue #73:
        //    GQA's `cos_cache` is resident on every rotary model, so designating it would make
        //    all 32 GQA nodes anchors again and put 193 back into the accounting.
        let mut m = base.clone();
        let cos = m
            .iter()
            .position(|s| {
                s.qualified_op == "com.microsoft::GroupQueryAttention" && s.name == "cos_cache"
            })
            .unwrap();
        m[cos].kind = SiteKind::Factor;
        cases.push(("designate an activation", m));

        // 7. Contiguous truncation of one family's tail — QMoE loses its four activation scales.
        let mut m = base.clone();
        m.retain(|s| !(s.qualified_op == "com.microsoft::QMoE" && s.index >= 17));
        cases.push(("contiguous truncation", m));

        // 8. A family's schema extent changes: MatMulNBits drops the deprecated `g_idx` and
        //    everything after it shifts down one index.
        let mut m = base.clone();
        m.retain(|s| !(s.qualified_op == "com.microsoft::MatMulNBits" && s.index == 4));
        for s in m.iter_mut() {
            if s.qualified_op == "com.microsoft::MatMulNBits" && s.index == 5 {
                s.index = 4;
            }
        }
        cases.push(("schema extent change", m));

        for (label, mutated) in cases {
            assert!(
                disagreement_with_held_out(&mutated).is_err(),
                "mutation `{label}` was NOT caught by the held-out pin. The pin is not \
                 load-bearing and every conclusion drawn from it is void."
            );
        }
    }

    /// The designated counts, stated per family so a reader can check them one at a time.
    ///
    /// These are the numbers issue #73 and its review turn on: **nine** QMoE sites (not four),
    /// **zero** GQA sites (which is why Phi-3.5 has 161 anchors and not 193), and zero for both
    /// attention ops that consume already-projected Q/K/V.
    #[test]
    fn the_designated_counts_are_stated_per_family() {
        let designated = |op: &str| {
            WEIGHT_SITES
                .iter()
                .filter(|s| s.qualified_op == op && s.designated())
                .count()
        };
        assert_eq!(
            designated("MatMul"),
            2,
            "both factors of a product may be resident"
        );
        assert_eq!(designated("Gemm"), 2, "A and B, not C");
        assert_eq!(designated("Conv"), 1, "W, not X and not B");
        assert_eq!(designated("ConvTranspose"), 1, "W, not X and not B");
        assert_eq!(
            designated("Attention"),
            0,
            "ONNX `Attention` is fused SDPA over already-projected Q/K/V and has no weight input"
        );
        assert_eq!(
            designated("com.microsoft::Attention"),
            1,
            "the unfused contrib op does carry its merged Q/K/V projection matrix at index 1"
        );
        assert_eq!(
            designated("com.microsoft::MatMulNBits"),
            4,
            "B, scales, zero_points, g_idx"
        );
        assert_eq!(
            designated("com.microsoft::GroupQueryAttention"),
            0,
            "GQA's projections belong to its neighbouring MatMulNBits nodes; every GQA input is \
             an activation, a cache, a length, a per-head scalar or an O(head_size) norm gain"
        );
        assert_eq!(
            designated("com.microsoft::MultiHeadAttention"),
            0,
            "MHA takes Q/K/V already projected"
        );
        assert_eq!(
            designated("com.microsoft::QMoE"),
            9,
            "packed weights + scales + zero points for fc1/fc2/fc3 — not four, and not the twelve \
             a name-based reading gives"
        );
        assert_eq!(
            designated("com.microsoft::LinearAttention"),
            0,
            "every input is declared with a leading (B, T, ...) or is the recurrent state"
        );
        assert_eq!(
            WEIGHT_SITES.iter().filter(|s| s.designated()).count(),
            20,
            "twenty designated sites over eleven families and eighty-four operands"
        );
        assert_eq!(WEIGHT_SITES.len(), 84);
        assert_eq!(heavy_op_families().len(), 11);
    }

    /// No two operands share a justification.
    ///
    /// Without this the seal in [`HELD_OUT_SITES`] would not detect a swap between the two rows
    /// that happened to be worded identically, and "swap two justifications" would be a control
    /// with a hole in it. Four pairs were identically worded when the table was first written —
    /// `Conv`/`ConvTranspose`'s `X` and `B`, `GroupQueryAttention`'s `key` and `value`, and
    /// `MultiHeadAttention`/`GroupQueryAttention`'s `query` — and this assertion is why they are
    /// not any more.
    #[test]
    fn every_justification_is_distinct() {
        let mut seen: Vec<&str> = WEIGHT_SITES.iter().map(|s| s.reason).collect();
        let total = seen.len();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(
            seen.len(),
            total,
            "two operands share a justification, so the seal cannot distinguish them"
        );
        assert!(WEIGHT_SITES.iter().all(|s| !s.reason.is_empty()));
    }

    /// Each family's rows are contiguous and start at 0, which is what makes `index` an offset
    /// into the pinned schema rather than a label.
    #[test]
    fn every_family_is_contiguous_from_zero() {
        for family in heavy_op_families() {
            let rows = weight_sites_for(family);
            for (expected, row) in rows.iter().enumerate() {
                assert_eq!(
                    row.index, expected,
                    "{family} operand list is not contiguous from 0 at position {expected}"
                );
            }
            assert!(!rows.is_empty(), "{family} has no rows");
        }
    }

    /// Anchoring is a property of the node, and the shipped predicate says so.
    #[test]
    fn anchoring_requires_a_resident_weight_at_a_designated_site() {
        // A MatMul of two runtime activations is not an anchor — the six MiniLM-L6-v2 islands
        // issue #73 is about.
        assert!(!is_anchor("MatMul", &[false, false]));
        // The same op with a resident B is.
        assert!(is_anchor("MatMul", &[false, true]));
        // ...and with a resident A, because ONNX MatMul is symmetric and models emit both.
        assert!(is_anchor("MatMul", &[true, false]));

        // Gemm's C is a bias: resident there and nowhere else is not an anchor.
        assert!(!is_anchor("Gemm", &[false, false, true]));
        assert!(is_anchor("Gemm", &[false, true, true]));

        // Conv anchors on W and not on X or B.
        assert!(!is_anchor("Conv", &[true, false, true]));
        assert!(is_anchor("Conv", &[false, true, false]));

        // A quantised weight matrix anchors on any of its four payload sites.
        assert!(is_anchor(
            "com.microsoft::MatMulNBits",
            &[false, true, true, false, false, false]
        ));
        assert!(!is_anchor(
            "com.microsoft::MatMulNBits",
            &[true, false, false, false, false, true]
        ));

        // GQA never anchors, however much of it is resident. This is the 193 → 161 correction.
        assert!(!is_anchor(
            "com.microsoft::GroupQueryAttention",
            &[true; 16]
        ));
        // Nor does MultiHeadAttention, nor ONNX `Attention`, nor `LinearAttention`.
        assert!(!is_anchor("com.microsoft::MultiHeadAttention", &[true; 10]));
        assert!(!is_anchor("Attention", &[true; 7]));
        assert!(!is_anchor("com.microsoft::LinearAttention", &[true; 6]));
        // The contrib `Attention` does, at index 1 and only there.
        assert!(is_anchor(
            "com.microsoft::Attention",
            &[false, true, false, false, false, false, false]
        ));
        assert!(!is_anchor(
            "com.microsoft::Attention",
            &[true, false, true, true, true, true, true]
        ));

        // Ops outside the inventory never anchor whatever is resident.
        assert!(!is_anchor("Add", &[true, true]));
        assert!(!is_anchor("Reshape", &[true, true]));
        assert!(!is_anchor("com.microsoft::NotAnOp", &[true; 8]));
    }

    /// Every uncertain reading fails closed.
    #[test]
    fn unreadable_operands_cannot_manufacture_an_anchor() {
        // No operands read at all.
        assert!(!is_anchor("MatMul", &[]));
        // Fewer residency answers than the schema has inputs — absent means non-resident.
        assert!(!is_anchor("com.microsoft::MatMulNBits", &[false]));
        // More answers than the schema has inputs: the surplus classifies as `None` and cannot
        // designate, so a node whose operand list ORT reports as longer than the schema still
        // needs a real designated site to anchor.
        assert!(!is_anchor("Conv", &[false, false, false, true, true, true]));
        assert!(is_anchor("Conv", &[false, true, false, true, true, true]));
        // Out-of-range indices classify as nothing at all.
        assert!(classify_weight_operand("MatMul", 2).is_none());
        assert!(classify_weight_operand("com.microsoft::QMoE", 21).is_none());
        assert!(classify_weight_operand("Add", 0).is_none());
        assert!(!is_designated_weight_site("MatMul", 99));
    }

    /// The FLOP-scoring predicate and the anchor predicate read the same table and answer
    /// different questions.
    #[test]
    fn heavy_op_and_anchor_are_different_questions_over_one_table() {
        // Heavy is about arithmetic shape and stays true for the activation-only MatMul, which
        // is why `ep.rs`'s FLOP estimate is unchanged by issue #73.
        assert!(is_heavy_op("MatMul"));
        assert!(!is_anchor("MatMul", &[false, false]));
        assert!(is_heavy_op("com.microsoft::GroupQueryAttention"));
        assert!(!is_anchor(
            "com.microsoft::GroupQueryAttention",
            &[true; 16]
        ));
        assert!(!is_heavy_op("Add"));
        // No family may be anchorable without being heavy: the anchor predicate reads the same
        // rows, so this is a structural fact and the assertion records it.
        for family in heavy_op_families() {
            assert!(is_heavy_op(family));
        }
    }

    /// Both polarities of the gate, on islands of the shape the issue names.
    ///
    /// A one-node island of an activation⊗activation `MatMul` must be declined, and the
    /// identically-shaped island whose `MatMul` carries a resident weight must be claimed. The
    /// two islands differ in exactly one bit — whether operand 1 is resident — so nothing else
    /// can be responsible for the difference in verdict.
    #[test]
    fn the_gate_answers_differently_for_a_weight_matmul_and_an_activation_matmul() {
        // MiniLM-L6-v2's attention AV product at batch 1, sequence 128, measured in the issue:
        // 983,040 B in, 196,608 B out, 0.013 GFLOP.
        let boundary_in = 983_040_u64;
        let boundary_out = 196_608_u64;
        let flops = 13_000_000_u64;

        let island_for = |resident_b: bool| Island {
            nodes: 1,
            anchors: usize::from(is_anchor("MatMul", &[false, resident_b])),
            flops,
            input_bytes: boundary_in,
            output_bytes: boundary_out,
            symbolic_boundary_slots: 0,
        };

        let policy = Policy::default();
        let model = &TransferModel::DISCRETE;

        let activation_only = island_for(false);
        assert_eq!(activation_only.anchors, 0);
        let verdict = evaluate(&activation_only, model, &policy);
        let Verdict::Reject(why) = &verdict else {
            panic!(
                "a one-node island of an activation-only MatMul must be declined; got {verdict:?}"
            );
        };
        let text = decline_for(why);
        assert!(
            text.starts_with(&format!("[{}] ", DeclineCode::Partition.tag())),
            "the decline must reach the claim log as a [partition] code; got {text}"
        );

        let weight_bearing = island_for(true);
        assert_eq!(weight_bearing.anchors, 1);
        assert!(
            evaluate(&weight_bearing, model, &policy).is_claim(),
            "the same island with a resident weight at B must still be claimed — narrowing the \
             anchor predicate must not cost us the case it was written for"
        );

        // And with the exemption switched off, the weight-bearing island is decided by the
        // economics rather than by the exemption, which is what makes the exemption observable
        // (R10).
        let no_exemption = Policy {
            anchor_exemption: false,
            ..policy
        };
        assert!(
            !evaluate(&weight_bearing, model, &no_exemption).is_claim(),
            "with the exemption off this island is transfer-dominated; if it claims anyway the \
             exemption is not the term doing the work and this test is measuring nothing"
        );
    }

    /// Phi-3.5's fused island: 355 claimed nodes, **161** anchors, and the 32 GQA nodes among the
    /// claimed-but-not-anchor remainder.
    ///
    /// Recorded as an assertion rather than only as prose because `193` appeared in six constants
    /// and two documents, and prose is not what a future edit is checked against.
    #[test]
    fn the_phi35_constants_carry_the_corrected_anchor_count() {
        for island in [
            Island::ESTIMATED_PHI35_DEV0_INTERNAL_EDGES_COUNTED,
            Island::ESTIMATED_PHI35_DEV0_INTERNAL_EDGES_FIXED,
            Island::MEASURED_PHI35_DEV0_REAL_BYTES,
        ] {
            assert_eq!(island.nodes, 355);
            assert_eq!(
                island.anchors, 161,
                "161 MatMulNBits anchors. The 32 GroupQueryAttention nodes are claimed and are \
                 not anchors (issue #73); 193 was the op-name count"
            );
        }
        // The GQA nodes are the difference, and it is exactly 32.
        assert_eq!(193 - Island::MEASURED_PHI35_DEV0_REAL_BYTES.anchors, 32);
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
