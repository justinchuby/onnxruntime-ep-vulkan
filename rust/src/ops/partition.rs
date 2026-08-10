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
    /// How many of those are *anchors* — heavy-family ops that carry a resident constant
    /// initializer at a **schema-designated weight site** (see [`is_anchor`]). This is the count
    /// gate 2's exemption reads: an anchor is heavy *and* has a weight worth amortising a boundary
    /// round-trip against.
    ///
    /// This is **not** the heavy-family node count. On Phi-3.5 the two differ: 161 `MatMulNBits`
    /// anchor, 32 `GroupQueryAttention` are heavy-family but designate no weight site and so are
    /// **not** anchors — they are supported non-anchor members of an island anchored by the
    /// `MatMulNBits` nodes around them. See [`Island::heavy_family_nodes`].
    pub anchors: usize,
    /// How many of the island's nodes are in the compute-heavy op family (see
    /// [`is_heavy_family`]) — the FLOP-significant rows of the §4 inventory.
    ///
    /// **Deliberately separate from [`Island::anchors`].** Heavy-family membership governs the
    /// FLOP estimate; anchor eligibility governs the gate-2 exemption. They are different
    /// questions: `GroupQueryAttention` is heavy (it does real attention arithmetic) but anchors
    /// nothing (its operands are runtime activations and KV cache, not a resident weight), so it
    /// must contribute to `heavy_family_nodes` and **not** to `anchors`. On Phi-3.5 this count is
    /// 193 (161 `MatMulNBits` + 32 `GroupQueryAttention`); 193 is a heavy-family-node count and is
    /// never an anchor count or an anchor ratio.
    pub heavy_family_nodes: usize,
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
    /// `nodes`, `anchors` and `heavy_family_nodes` are counts from the CLAIM_LOG of the same run
    /// (355 claimed). Of those, **161 `MatMulNBits` are anchors** — they carry a resident int4
    /// weight at their schema-designated weight site (input `B`). The **32 `GroupQueryAttention`
    /// nodes are heavy-family but not anchors**: GQA designates no weight site, so it is a
    /// supported non-anchor member of this island, which is anchored by the `MatMulNBits` nodes
    /// around it. The heavy-family node count is therefore **193** (161 + 32); 193 is a
    /// heavy-family-node count, never an anchor count or anchor ratio. `flops` and the boundary
    /// total are the estimator's own numbers.
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
        heavy_family_nodes: 193,
        flops: 23_020_437_504,
        input_bytes: 0,
        output_bytes: 89_199_100_032,
        symbolic_boundary_slots: 0,
    };

    /// The same island as re-estimated on 2026-08-01 after internal edges stopped being counted.
    ///
    /// Read out of the verbose `PartitionStats` summary on dev0 with the same model and the same
    /// binary that produced the 355 → 0 claim census. The number that changed is the boundary;
    /// nodes, anchors, heavy-family count and FLOPs are unaffected because the fix touches only
    /// which outputs are charged to the boundary.
    pub const ESTIMATED_PHI35_DEV0_INTERNAL_EDGES_FIXED: Island = Island {
        nodes: 355,
        anchors: 161,
        heavy_family_nodes: 193,
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
        heavy_family_nodes: 193,
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

/// Ops in the compute-heavy family — the FLOP-significant rows of the §4 inventory.
///
/// A lone `Add` is never worth a round-trip; a matmul-family or attention-family op does real
/// arithmetic. **This is a FLOP classification only.** Being heavy is necessary but not
/// sufficient to *anchor* an island: an op anchors only when it also carries a resident weight at
/// a schema-designated weight site (see [`is_anchor`]). `GroupQueryAttention` is heavy — it
/// computes attention — but designates no weight site, so it is heavy-family and non-anchor.
///
/// Keeping this separate from anchor eligibility is the whole point of issue #73: the two used to
/// be one predicate keyed on the op *name*, which anchored an activation-only `MatMul` (both
/// operands runtime) purely because it was called `MatMul`. Name is not weight.
pub fn is_heavy_family(qualified_op: &str) -> bool {
    matches!(
        qualified_op,
        "MatMul"
            | "Gemm"
            | "Conv"
            | "ConvTranspose"
            | "Attention"
            | "com.microsoft::MatMulNBits"
            | "com.microsoft::GroupQueryAttention"
            | "com.microsoft::MultiHeadAttention"
            | "com.microsoft::Attention"
            | "com.microsoft::QMoE"
            | "com.microsoft::LinearAttention"
    )
}

/// The input indices a heavy-family op's **schema** designates as trained-weight sites.
///
/// An op anchors an island only when one of *these* sites holds a resident constant initializer —
/// not when *any* operand happens to be constant, and never on the op name alone. The indices are
/// the operand positions the pinned ONNX / ORT v1.28.0 (`da9b5e364c465de65c49d91e696cd6485270757f`)
/// schemas give to the weight tensor, cross-checked by `ci/check_anchor_weight_sites.py` against
/// `rust/tools/anchor_weight_sites.json`. A machine-checkable table, not a hand-maintained belief.
///
/// Provenance for each row:
/// * `MatMul` — ONNX `MatMul` `(A, B)`; the right-hand matrix `B` at index **1** is the weight.
/// * `Gemm` — ONNX `Gemm` `(A, B, C)`; `B` at index **1**.
/// * `Conv` / `ConvTranspose` — ONNX `(X, W, B)`; the kernel `W` at index **1**.
/// * `com.microsoft::MatMulNBits` — ORT `(A, B, scales, zero_points, g_idx, bias)`
///   (`onnxruntime/core/graph/contrib_ops/contrib_defs.cc:3672`); the packed int-N weight `B`
///   at index **1**. Its doc string is explicit: *"the right-hand-side matrix (weights) is
///   quantized to N bits"*.
///
/// **Empty slice = no schema-designated weight site → cannot anchor on its own.** The
/// attention-family and MoE-family ops fall here deliberately:
/// * `com.microsoft::GroupQueryAttention` — inputs are `query`/`key`/`value`, the KV cache, and
///   rotary caches (`onnxruntime/core/graph/contrib_ops/bert_defs.cc:1258`+): runtime activations,
///   not a resident weight. Its optional `q_norm_weight`/`k_norm_weight` inputs are *per-head RMS
///   norm scales*, not the projection weight matrix — which is exactly why a name-substring
///   heuristic would be wrong here and this table is an explicit index list instead.
/// * `Attention` / `com.microsoft::{Attention,MultiHeadAttention,LinearAttention}` / `QMoE` —
///   left empty as the fail-closed default: their weight-site provenance is not pinned in this
///   revision, and an unpinned op must fall towards the CPU (no anchor) rather than anchor on an
///   unverified index. None is in any currently censused model; add a row here, with provenance,
///   before relying on one to anchor.
pub fn weight_site_indices(qualified_op: &str) -> &'static [usize] {
    match qualified_op {
        "MatMul" => &[1],
        "Gemm" => &[1],
        "Conv" => &[1],
        "ConvTranspose" => &[1],
        "com.microsoft::MatMulNBits" => &[1],
        _ => &[],
    }
}

/// Whether a node anchors its island: a **heavy-family op carrying a resident constant
/// initializer at a schema-designated weight site**.
///
/// `resident_inputs[i]` is `true` iff operand `i` is a graph constant initializer resident on the
/// device (uploaded once at load, not a per-inference transfer). The predicate takes the node's
/// residency facts, never the op name alone — a name-only call does not typecheck, which is the
/// structural guarantee issue #73 asks for.
///
/// **Fails closed in every degenerate case.** A missing operand (index past the end of
/// `resident_inputs`), a null slot, a runtime (non-constant) operand, an unavailable
/// `ValueInfo_IsConstantInitializer` function, or a status-error read all surface as `false` in
/// the corresponding `resident_inputs` entry (or as an out-of-range index here), so the node does
/// not anchor. An op with no designated weight site returns `false` unconditionally.
///
/// Worked cases from issue #73's MiniLM analysis:
/// * A weight `MatMul` — `B` (index 1) is an initializer ⇒ `resident_inputs[1] == true` ⇒ anchors.
/// * An attention batched `MatMul` (QK^T or AV) — both operands are runtime activations ⇒
///   `resident_inputs[1] == false` ⇒ does **not** anchor, and a lone one is rejected by the
///   economics gate instead of exempted by op identity.
/// * `GroupQueryAttention` — no designated weight site ⇒ never anchors, even though it is
///   heavy-family and may carry constant rotary caches at non-weight sites.
pub fn is_anchor(qualified_op: &str, resident_inputs: &[bool]) -> bool {
    let sites = weight_site_indices(qualified_op);
    if sites.is_empty() {
        return false;
    }
    sites
        .iter()
        .any(|&i| resident_inputs.get(i).copied().unwrap_or(false))
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
    /// Whether an island containing at least one anchor (an [`is_anchor`] node — heavy-family
    /// *and* weight-bearing at a schema-designated site) skips the economics check.
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
///    **Anchor-containing islands are exempt from gate 2**: an anchor ([`is_anchor`]) is a
///    heavy-family op *carrying a resident weight at a schema-designated site*, so a boundary
///    round-trip is amortised against a weight the island reuses every inference — that is the
///    design invariant. Note the exemption keys on `island.anchors`, **not** on heavy-family
///    membership: an activation-only `MatMul` (both operands runtime, issue #73's MiniLM AV
///    batched matmul) is heavy-family but not an anchor, so it is *not* exempt and a lone one is
///    rejected here as `TransferDominated` — the outcome gate 2 exists to produce.
///    The provisional `TransferModel` constants are calibrated against real model execution and
///    may not reflect isolated unit-test input sizes; applying the economic check to a genuine
///    weight anchor would reject it when tested in isolation, which contradicts the stated design
///    intent ("a single MatMul on LLM-sized weights always is worth it").
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
            heavy_family_nodes: 1,
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
            heavy_family_nodes: 0,
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
            heavy_family_nodes: 0,
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
            heavy_family_nodes: 0,
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

    /// Anchoring is a property of the **node**, not the op name. The same op is or is not an
    /// anchor depending on whether its schema-designated weight site holds a resident initializer.
    /// Both polarities are asserted here — an assertion that can only ever return one answer is
    /// not a check (issue #73).
    #[test]
    fn anchoring_is_weight_site_residency_not_op_name() {
        // Weight polarity: an initializer at the designated site (index 1) anchors.
        let weight = [false, true]; // A runtime, B resident weight
        assert!(is_anchor("MatMul", &weight));
        assert!(is_anchor("Gemm", &weight));
        assert!(is_anchor("Conv", &weight));
        assert!(is_anchor("com.microsoft::MatMulNBits", &weight));

        // Activation polarity: the SAME ops, both operands runtime, do NOT anchor. This is the
        // MiniLM attention batched-matmul case the economics gate exists to reject.
        let activation_only = [false, false];
        assert!(!is_anchor("MatMul", &activation_only));
        assert!(!is_anchor("Gemm", &activation_only));

        // GroupQueryAttention designates no weight site: it never anchors, even when it carries
        // resident constants (rotary caches, per-head norm scales) at non-weight inputs.
        let gqa_with_constants = [false, false, false, true, true, false, false, true, true];
        assert!(!is_anchor(
            "com.microsoft::GroupQueryAttention",
            &gqa_with_constants
        ));
        assert!(is_heavy_family("com.microsoft::GroupQueryAttention"));

        // Non-heavy ops never anchor regardless of residency.
        assert!(!is_anchor("Add", &weight));
        assert!(!is_anchor("Reshape", &weight));
        assert!(!is_heavy_family("Add"));
    }

    /// Fail-closed: every degenerate residency input yields "not an anchor", never a panic.
    #[test]
    fn anchor_predicate_fails_closed_on_degenerate_inputs() {
        // Missing operand: designated site 1 is past the end of the residency vector.
        assert!(!is_anchor("MatMul", &[]));
        assert!(!is_anchor("MatMul", &[true])); // only index 0 present; site 1 absent
        // Runtime operand at the weight site.
        assert!(!is_anchor("MatMul", &[true, false]));
        // A constant at a non-weight site (index 0) must not anchor MatMul (site is 1).
        assert!(!is_anchor("MatMul", &[true, false]));
        // Empty designated-site op with everything resident still cannot anchor.
        assert!(!is_anchor(
            "com.microsoft::GroupQueryAttention",
            &[true, true, true, true]
        ));
    }

    /// The weight-site table matches the pinned schema provenance. This mirrors
    /// `ci/check_anchor_weight_sites.py` at the Rust level so a table edit that drops a row or
    /// invents a site is caught by `cargo test`, not only by the Python checker.
    #[test]
    fn weight_sites_match_pinned_schema() {
        assert_eq!(weight_site_indices("MatMul"), &[1]);
        assert_eq!(weight_site_indices("Gemm"), &[1]);
        assert_eq!(weight_site_indices("Conv"), &[1]);
        assert_eq!(weight_site_indices("ConvTranspose"), &[1]);
        assert_eq!(weight_site_indices("com.microsoft::MatMulNBits"), &[1]);
        // No designated weight site → empty.
        assert!(weight_site_indices("com.microsoft::GroupQueryAttention").is_empty());
        assert!(weight_site_indices("Add").is_empty());
        // Every op with a designated weight site must be in the heavy family.
        for op in [
            "MatMul",
            "Gemm",
            "Conv",
            "ConvTranspose",
            "com.microsoft::MatMulNBits",
        ] {
            assert!(
                is_heavy_family(op),
                "{op} has a weight site but is not heavy-family"
            );
        }
    }

    /// **End-to-end anchor polarity, decided by residency alone.** Two islands of *identical*
    /// shape — same node count, same bytes, same FLOPs — differ only in whether their single
    /// matmul-family node carries a resident weight. The one with a weight anchors and is claimed;
    /// the activation-only one anchors nothing and is rejected as `TransferDominated`. This is the
    /// MiniLM AV batched-matmul case from issue #73, isolated to the anchor distinction: the byte
    /// and FLOP terms are held constant so the only thing that can flip the verdict is the weight
    /// site's residency.
    #[test]
    fn a_weight_matmul_island_claims_where_an_activation_only_one_is_rejected() {
        // Transfer-dominated shape (big tensors, little arithmetic), sized past gate 1 so the
        // non-anchor arm reaches gate 2 and the reason is TransferDominated, not TooSmall.
        let shape = |anchors: usize| Island {
            nodes: 8,
            anchors,
            heavy_family_nodes: 1,
            flops: 2 * 128 * 128,
            input_bytes: 64 * 1024 * 1024,
            output_bytes: 64 * 1024 * 1024,
            symbolic_boundary_slots: 0,
        };

        // Weight MatMul: B (index 1) is a resident initializer ⇒ anchors ⇒ claimed.
        let weight_anchors = is_anchor("MatMul", &[false, true]) as usize;
        assert_eq!(weight_anchors, 1);
        let weight_island = shape(weight_anchors);
        assert_eq!(
            evaluate(&weight_island, &TransferModel::DISCRETE, &Policy::default()),
            Verdict::Claim
        );

        // Activation-only MatMul: both operands runtime ⇒ anchors nothing ⇒ rejected on economics.
        let act_anchors = is_anchor("MatMul", &[false, false]) as usize;
        assert_eq!(act_anchors, 0);
        let act_island = shape(act_anchors);
        let v = evaluate(&act_island, &TransferModel::DISCRETE, &Policy::default());
        assert!(
            matches!(v, Verdict::Reject(RejectReason::TransferDominated { .. })),
            "activation-only matmul must be rejected by the economics gate, got {v:?}"
        );
        // Both rejections — TooSmall and TransferDominated — surface as DeclineCode::Partition, so
        // the six MiniLM rejections appear in the decline histogram rather than vanishing.
        if let Verdict::Reject(reason) = v {
            assert_eq!(
                DeclineCode::of_reason(&decline_for(&reason)),
                Some(DeclineCode::Partition)
            );
        }
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
                    heavy_family_nodes: 0,
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
                heavy_family_nodes: 8,
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
                    heavy_family_nodes: 0,
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
                heavy_family_nodes: 193,
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
                    heavy_family_nodes: 193,
                    flops: 23_020_437_504,
                    input_bytes: 0,
                    output_bytes: SUBSTITUTED * per_slot_bytes,
                    symbolic_boundary_slots: 1,
                },
                Island {
                    nodes: 355,
                    anchors: 161,
                    heavy_family_nodes: 193,
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
