//! Host-side shape and broadcast planning — computed once, shared by every op.
//!
//! # Why this is one module and not sixty
//!
//! `OP_COVERAGE.md` §5.2 is blunt about it: numpy broadcasting is the single most repeated, most
//! error-prone piece of logic in an op library, and a backend that re-derives it per op pays for
//! it per op — in code, in bugs, and in conformance failures that look like kernel bugs but are
//! indexing bugs. So it is derived exactly once, here, on the host, and handed to the shader as
//! **strides**. A broadcast axis is simply a stride of `0`; the GLSL template contains no
//! broadcasting logic at all, only a generic `linear index -> per-input offset` walk.
//!
//! That is what makes ~66 elementwise ops share three shader templates.
//!
//! # Push-constant layout
//!
//! [`ShapePlan::push_constants`] emits, little-endian:
//!
//! ```text
//! offset  size  field
//! 0       4     rank            (u32, <= MAX_RANK)
//! 4       4     elem_count      (u32, output element count)
//! 8       24    out_shape[6]    (u32, left-padded with 1)
//! 32      24    strides_in0[6]  (u32, element strides; 0 == broadcast)
//! 56      24    strides_in1[6]  (present only for arity >= 2)
//! 80      24    strides_in2[6]  (present only for arity >= 3)
//! ```
//!
//! Worst case (ternary) is 104 bytes, inside the 128-byte `maxPushConstantsSize` floor every
//! Vulkan 1.1 implementation guarantees, which is why this is push constants and not a UBO.

use std::fmt;

/// Maximum tensor rank the shared indexing helper handles.
///
/// Six covers every shape in the model families `OP_COVERAGE.md` §3 profiles (the deepest is a
/// 5-D conv/attention layout); rank-7+ tensors decline with `[rank]` and run on the CPU EP, which
/// is the correct trade for a limit that keeps the push-constant block inside 128 bytes.
pub const MAX_RANK: usize = 6;

/// Why a shape plan could not be built. Each maps to a distinct decline code, so the decline
/// histogram distinguishes "we can't do rank 7" from "this graph has dynamic shapes".
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ShapeError {
    /// No inputs were supplied.
    NoInputs,
    /// A shape exceeds [`MAX_RANK`].
    RankTooLarge {
        /// The offending rank.
        rank: usize,
    },
    /// A dimension is symbolic (`-1`), so nothing can be computed on the host.
    Symbolic {
        /// Which input.
        input: usize,
    },
    /// Two shapes cannot be broadcast together.
    NotBroadcastable {
        /// Axis, counted from the right (0 == last).
        axis_from_right: usize,
        /// The output extent established so far.
        have: i64,
        /// The extent this input wants.
        want: i64,
    },
    /// The element count or a stride does not fit in the `u32` the shader indexes with.
    Overflow,
}

impl fmt::Display for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShapeError::NoInputs => f.write_str("no inputs to broadcast"),
            ShapeError::RankTooLarge { rank } => {
                write!(f, "rank {rank} exceeds the supported maximum of {MAX_RANK}")
            }
            ShapeError::Symbolic { input } => write!(
                f,
                "input {input} has a symbolic dimension, so shapes cannot be resolved at compile \
                 time"
            ),
            ShapeError::NotBroadcastable {
                axis_from_right,
                have,
                want,
            } => write!(
                f,
                "axis {axis_from_right} counted from the right is {want}, which does not broadcast \
                 against {have}"
            ),
            ShapeError::Overflow => {
                f.write_str("the element count or a stride does not fit in 32 bits")
            }
        }
    }
}

/// The result of broadcasting a set of input shapes: an output shape plus a per-input stride
/// vector, ready to serialise into push constants.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShapePlan {
    /// Effective rank, `<= MAX_RANK`. Axes are stored right-aligned in the fixed arrays, i.e.
    /// index `MAX_RANK - rank` is the outermost used axis.
    pub rank: usize,
    /// Broadcast output shape, left-padded with `1`.
    pub out_shape: [u32; MAX_RANK],
    /// Element strides per input, left-padded with `0`. A `0` on a used axis means "broadcast".
    pub strides: Vec<[u32; MAX_RANK]>,
    /// Number of output elements.
    pub elem_count: u64,
    /// True when every input has exactly the output shape — the shader may then skip the index
    /// walk entirely and read linearly, which is the common case and worth the branch.
    pub all_identical: bool,
}

/// Row-major (C-contiguous) element strides for a right-aligned shape.
fn contiguous_strides(shape: &[u32; MAX_RANK]) -> [u32; MAX_RANK] {
    let mut strides = [0u32; MAX_RANK];
    let mut acc: u64 = 1;
    for i in (0..MAX_RANK).rev() {
        strides[i] = acc as u32;
        acc = acc.saturating_mul(u64::from(shape[i]));
    }
    strides
}

impl ShapePlan {
    /// Broadcast `inputs` together under numpy rules.
    ///
    /// Shapes are the ONNX ones: fully static, non-negative. A `-1` anywhere is reported as
    /// [`ShapeError::Symbolic`] rather than guessed at.
    pub fn broadcast(inputs: &[&[i64]]) -> Result<ShapePlan, ShapeError> {
        if inputs.is_empty() {
            return Err(ShapeError::NoInputs);
        }

        let mut rank = 0usize;
        for (i, s) in inputs.iter().enumerate() {
            if s.len() > MAX_RANK {
                return Err(ShapeError::RankTooLarge { rank: s.len() });
            }
            if s.iter().any(|d| *d < 0) {
                return Err(ShapeError::Symbolic { input: i });
            }
            rank = rank.max(s.len());
        }

        // Right-align every input into a MAX_RANK array padded with 1s.
        let padded: Vec<[u32; MAX_RANK]> = inputs
            .iter()
            .map(|s| {
                let mut p = [1u32; MAX_RANK];
                let off = MAX_RANK - s.len();
                for (j, d) in s.iter().enumerate() {
                    // Extents are bounded by the u32 index space; anything larger overflows the
                    // element count check below anyway.
                    p[off + j] = u32::try_from(*d).unwrap_or(u32::MAX);
                }
                p
            })
            .collect();

        let mut out_shape = [1u32; MAX_RANK];
        for axis in 0..MAX_RANK {
            let mut extent = 1u32;
            for p in &padded {
                let d = p[axis];
                if d == extent || d == 1 {
                    continue;
                }
                if extent == 1 {
                    extent = d;
                } else {
                    return Err(ShapeError::NotBroadcastable {
                        axis_from_right: MAX_RANK - 1 - axis,
                        have: i64::from(extent),
                        want: i64::from(d),
                    });
                }
            }
            out_shape[axis] = extent;
        }

        let mut elem_count: u64 = 1;
        for d in out_shape {
            elem_count = elem_count
                .checked_mul(u64::from(d))
                .ok_or(ShapeError::Overflow)?;
        }
        if elem_count > u64::from(u32::MAX) {
            return Err(ShapeError::Overflow);
        }

        // Strides: contiguous over the input's *own* shape, with 0 wherever the input is being
        // stretched. Zero-stride is the entire broadcasting implementation.
        //
        // Key invariant: padding axes (indices < MAX_RANK - input.len()) are NEVER real axes of
        // the input — they are 1-padding to align shorter inputs to the right. A scalar input
        // has no real axes at all. These padding axes must have stride 0 even when out_shape is
        // also 1 there, because the scalar/shorter tensor does not actually "have" those
        // dimensions: setting a nonzero stride would make the index walk incorrectly stride into
        // memory that doesn't exist.
        let mut strides = Vec::with_capacity(padded.len());
        for (idx, p) in padded.iter().enumerate() {
            let off = MAX_RANK - inputs[idx].len(); // first real axis; < off is padding
            let own = contiguous_strides(p);
            let mut s = [0u32; MAX_RANK];
            for axis in off..MAX_RANK {
                s[axis] = if p[axis] == out_shape[axis] {
                    own[axis]
                } else {
                    0
                };
            }
            strides.push(s);
        }

        let all_identical = padded.iter().all(|p| *p == out_shape);

        Ok(ShapePlan {
            rank,
            out_shape,
            strides,
            elem_count,
            all_identical,
        })
    }

    /// The output shape as an ONNX shape (leading 1-padding removed down to `rank` axes).
    pub fn out_dims(&self) -> Vec<i64> {
        self.out_shape[MAX_RANK - self.rank..]
            .iter()
            .map(|d| i64::from(*d))
            .collect()
    }

    /// Serialise into the push-constant block documented at the top of this module.
    pub fn push_constants(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 + (1 + self.strides.len()) * MAX_RANK * 4);
        out.extend_from_slice(&(self.rank as u32).to_le_bytes());
        out.extend_from_slice(&(self.elem_count as u32).to_le_bytes());
        for d in self.out_shape {
            out.extend_from_slice(&d.to_le_bytes());
        }
        for s in &self.strides {
            for d in *s {
                out.extend_from_slice(&d.to_le_bytes());
            }
        }
        out
    }

    /// Workgroup count for a 1-D elementwise dispatch at the given local size.
    pub fn workgroups_1d(&self, local_size: u32) -> [u32; 3] {
        let local = local_size.max(1);
        let groups = self.elem_count.div_ceil(u64::from(local));
        [groups.max(1) as u32, 1, 1]
    }
}

/// Normalise a possibly-negative ONNX axis against a rank.
///
/// ONNX allows `axis` in `[-rank, rank)`. Rank 0 accepts only axis 0, per the spec's treatment of
/// scalars.
pub fn normalize_axis(axis: i64, rank: usize) -> Option<usize> {
    let r = rank as i64;
    if rank == 0 {
        return if axis == 0 { Some(0) } else { None };
    }
    let a = if axis < 0 { axis + r } else { axis };
    if a < 0 || a >= r {
        None
    } else {
        Some(a as usize)
    }
}

/// Normalise a list of axes, rejecting duplicates. Returns them sorted, which is what every
/// reduction and transpose kernel wants.
pub fn normalize_axes(axes: &[i64], rank: usize) -> Option<Vec<usize>> {
    let mut out = Vec::with_capacity(axes.len());
    for a in axes {
        let n = normalize_axis(*a, rank)?;
        if out.contains(&n) {
            return None;
        }
        out.push(n);
    }
    out.sort_unstable();
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(shapes: &[&[i64]]) -> ShapePlan {
        ShapePlan::broadcast(shapes).expect("should broadcast")
    }

    #[test]
    fn identical_shapes_take_the_fast_path() {
        let p = plan(&[&[2, 3, 4], &[2, 3, 4]]);
        assert!(p.all_identical);
        assert_eq!(p.elem_count, 24);
        assert_eq!(p.rank, 3);
        assert_eq!(p.out_dims(), vec![2, 3, 4]);
        // Contiguous strides over [.., 2, 3, 4] are [.., 12, 4, 1].
        assert_eq!(p.strides[0][MAX_RANK - 3..], [12, 4, 1]);
        assert_eq!(p.strides[0], p.strides[1]);
    }

    #[test]
    fn a_stretched_axis_gets_stride_zero() {
        let p = plan(&[&[2, 3, 4], &[1, 3, 1]]);
        assert!(!p.all_identical);
        assert_eq!(p.out_dims(), vec![2, 3, 4]);
        assert_eq!(p.strides[1][MAX_RANK - 3..], [0, 1, 0]);
        // The non-broadcast input keeps its own contiguous strides.
        assert_eq!(p.strides[0][MAX_RANK - 3..], [12, 4, 1]);
    }

    #[test]
    fn shorter_ranks_are_right_aligned() {
        // The classic LLM case: [B, S, H] + [H].
        let p = plan(&[&[2, 5, 8], &[8]]);
        assert_eq!(p.out_dims(), vec![2, 5, 8]);
        assert_eq!(p.strides[1][MAX_RANK - 3..], [0, 0, 1]);
        assert_eq!(p.elem_count, 80);
    }

    #[test]
    fn a_scalar_broadcasts_against_anything() {
        let p = plan(&[&[2, 3], &[]]);
        assert_eq!(p.out_dims(), vec![2, 3]);
        assert_eq!(p.strides[1], [0; MAX_RANK]);
        assert!(!p.all_identical);
    }

    #[test]
    fn two_scalars_produce_a_single_element() {
        let p = plan(&[&[], &[]]);
        assert_eq!(p.elem_count, 1);
        assert_eq!(p.rank, 0);
        assert!(p.out_dims().is_empty(), "scalar output stays rank 0");
        assert!(p.all_identical);
    }

    #[test]
    fn three_way_broadcast_works() {
        // `Where(cond, x, y)` with all three shapes differing.
        let p = plan(&[&[4, 1, 1], &[1, 3, 1], &[1, 1, 2]]);
        assert_eq!(p.out_dims(), vec![4, 3, 2]);
        assert_eq!(p.elem_count, 24);
        assert_eq!(p.strides[0][MAX_RANK - 3..], [1, 0, 0]);
        assert_eq!(p.strides[1][MAX_RANK - 3..], [0, 1, 0]);
        assert_eq!(p.strides[2][MAX_RANK - 3..], [0, 0, 1]);
    }

    #[test]
    fn incompatible_extents_are_rejected() {
        let e = ShapePlan::broadcast(&[&[2, 3], &[2, 4]]).unwrap_err();
        assert!(matches!(
            e,
            ShapeError::NotBroadcastable {
                axis_from_right: 0,
                have: 3,
                want: 4
            } | ShapeError::NotBroadcastable {
                axis_from_right: 0,
                have: 4,
                want: 3
            }
        ));
        assert!(e.to_string().contains("does not broadcast"));
    }

    #[test]
    fn symbolic_dims_are_rejected_not_guessed() {
        let e = ShapePlan::broadcast(&[&[2, -1], &[2, 3]]).unwrap_err();
        assert_eq!(e, ShapeError::Symbolic { input: 0 });
    }

    #[test]
    fn rank_over_the_limit_is_rejected() {
        let e = ShapePlan::broadcast(&[&[1, 1, 1, 1, 1, 1, 1]]).unwrap_err();
        assert_eq!(e, ShapeError::RankTooLarge { rank: 7 });
    }

    #[test]
    fn exactly_max_rank_is_accepted() {
        let p = plan(&[&[1, 2, 3, 4, 5, 6]]);
        assert_eq!(p.rank, MAX_RANK);
        assert_eq!(p.elem_count, 720);
    }

    #[test]
    fn no_inputs_is_an_error() {
        assert_eq!(ShapePlan::broadcast(&[]).unwrap_err(), ShapeError::NoInputs);
    }

    #[test]
    fn absurd_element_counts_overflow_cleanly() {
        let big = i64::from(u32::MAX);
        let e = ShapePlan::broadcast(&[&[big, big, big]]).unwrap_err();
        assert_eq!(e, ShapeError::Overflow);
    }

    #[test]
    fn a_zero_extent_is_legal_and_empty() {
        // ONNX permits zero-sized tensors; they must not be a broadcast failure.
        let p = plan(&[&[0, 3], &[0, 3]]);
        assert_eq!(p.elem_count, 0);
        assert_eq!(p.workgroups_1d(256), [1, 1, 1]);
    }

    #[test]
    fn push_constants_have_the_documented_layout() {
        let p = plan(&[&[2, 3], &[3]]);
        let pc = p.push_constants();
        assert_eq!(pc.len(), 8 + 3 * MAX_RANK * 4);
        assert_eq!(u32::from_le_bytes(pc[0..4].try_into().unwrap()), 2); // rank
        assert_eq!(u32::from_le_bytes(pc[4..8].try_into().unwrap()), 6); // elem_count
        // Last entry of out_shape is the innermost extent, 3.
        let shape_end = 8 + MAX_RANK * 4;
        assert_eq!(
            u32::from_le_bytes(pc[shape_end - 4..shape_end].try_into().unwrap()),
            3
        );
    }

    #[test]
    fn push_constants_fit_the_vulkan_floor() {
        let p = plan(&[&[2, 3, 4], &[2, 3, 4], &[2, 3, 4]]);
        assert!(
            p.push_constants().len() <= 128,
            "ternary push block must fit maxPushConstantsSize's 128-byte guaranteed floor"
        );
    }

    #[test]
    fn workgroup_counts_round_up() {
        let p = plan(&[&[1000]]);
        assert_eq!(p.workgroups_1d(256), [4, 1, 1]);
        let q = plan(&[&[512]]);
        assert_eq!(q.workgroups_1d(256), [2, 1, 1]);
    }

    #[test]
    fn axis_normalisation_matches_onnx() {
        assert_eq!(normalize_axis(0, 3), Some(0));
        assert_eq!(normalize_axis(2, 3), Some(2));
        assert_eq!(normalize_axis(-1, 3), Some(2));
        assert_eq!(normalize_axis(-3, 3), Some(0));
        assert_eq!(normalize_axis(3, 3), None);
        assert_eq!(normalize_axis(-4, 3), None);
        assert_eq!(normalize_axis(0, 0), Some(0));
        assert_eq!(normalize_axis(-1, 0), None);
    }

    #[test]
    fn axis_lists_reject_duplicates_and_sort() {
        assert_eq!(normalize_axes(&[-1, 0], 3), Some(vec![0, 2]));
        assert_eq!(normalize_axes(&[2, -1], 3), None);
        assert_eq!(normalize_axes(&[], 3), Some(vec![]));
    }
}
