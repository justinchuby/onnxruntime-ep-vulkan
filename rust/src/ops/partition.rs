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
    /// Provisional until measured — the numbers are a starting point for the rule, not a claim
    /// about any device.
    pub const UMA: TransferModel = TransferModel {
        fixed_ns: 20_000.0,
        bytes_per_ns: 40.0,
    };

    /// Discrete GPU over PCIe: a real staging copy in each direction.
    pub const DISCRETE: TransferModel = TransferModel {
        fixed_ns: 60_000.0,
        bytes_per_ns: 12.0,
    };

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
}

/// One maximal connected subgraph this EP would claim.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Island {
    /// How many nodes it contains.
    pub nodes: usize,
    /// How many of those are *anchors* — ops heavy enough to justify a boundary on their own
    /// (see [`is_anchor`]).
    pub anchors: usize,
    /// Estimated floating-point operations inside the island.
    pub flops: u64,
    /// Bytes entering the island from outside the EP.
    pub input_bytes: u64,
    /// Bytes leaving the island to outside the EP.
    pub output_bytes: u64,
}

impl Island {
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

/// Ops heavy enough that a single one of them justifies an island.
///
/// A lone `Add` is never worth a round-trip; a single `MatMul` on LLM-sized weights always is.
/// The list is the §4 inventory's "L/XL, compute-bound" rows.
pub fn is_anchor(qualified_op: &str) -> bool {
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
}

impl Default for Policy {
    fn default() -> Policy {
        Policy {
            min_nodes: 4,
            margin: 3.0,
            flops_per_ns: 1_000.0,
        }
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
pub fn evaluate(island: &Island, model: &TransferModel, policy: &Policy) -> Verdict {
    if island.nodes < policy.min_nodes && island.anchors == 0 {
        return Verdict::Reject(RejectReason::TooSmall {
            nodes: island.nodes,
            min_nodes: policy.min_nodes,
        });
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

/// Apply the rule to a set of islands, returning the survivors and the rejections.
pub fn retain_viable(
    islands: &[Island],
    model: &TransferModel,
    policy: &Policy,
) -> (Vec<Island>, Vec<(Island, RejectReason)>) {
    let mut kept = Vec::new();
    let mut dropped = Vec::new();
    for i in islands {
        match evaluate(i, model, policy) {
            Verdict::Claim => kept.push(i.clone()),
            Verdict::Reject(r) => dropped.push((i.clone(), r)),
        }
    }
    (kept, dropped)
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
        }
    }

    fn lone_add_island() -> Island {
        Island {
            nodes: 1,
            anchors: 0,
            flops: 4096,
            input_bytes: 4096 * 4 * 2,
            output_bytes: 4096 * 4,
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
        };
        assert!(matches!(
            evaluate(&island, &TransferModel::UMA, &Policy::default()),
            Verdict::Reject(RejectReason::TooSmall { .. })
        ));
    }

    #[test]
    fn rejections_are_machine_readable() {
        let (_, dropped) = retain_viable(
            &[lone_add_island()],
            &TransferModel::DISCRETE,
            &Policy::default(),
        );
        assert_eq!(dropped.len(), 1);
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
    fn anchors_are_the_heavy_ops_only() {
        assert!(is_anchor("MatMul"));
        assert!(is_anchor("Conv"));
        assert!(is_anchor("com.microsoft::MatMulNBits"));
        assert!(is_anchor("com.microsoft::GroupQueryAttention"));
        assert!(!is_anchor("Add"));
        assert!(!is_anchor("Reshape"));
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
    }
}
