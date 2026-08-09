//! Weight-only quantization — `MatMulNBits` and the block-quantized dequant path.
//!
//! # Why this module is not optional
//!
//! Justin's 2026-07-28 ruling makes an int4 LLM the *functional requirement*, not a later tuning
//! pass: "matmulnbits那些 都要做". `OP_COVERAGE.md` §3.2 is the evidence — the Qwen3 graph people
//! actually run is int4, every dense projection in it is a `com.microsoft.MatMulNBits`, and an EP
//! that declines them shreds the graph into ~200 islands. So this module is on the critical path
//! to "Qwen3.5 runs end-to-end", and §11 is equally clear that it is the expensive kind of work:
//! a `QGEMM` has no template to inherit from.
//!
//! # What exists here today
//!
//! Rows, contrib schema fingerprints, and claim predicates that are *exactly* as strict as the
//! kernels will be. The kernels themselves are [`XL_KERNEL`]-staged. That ordering is deliberate
//! and is the same argument as everywhere else in `ops/`: the claim policy is where the
//! correctness risk lives, it is testable without a GPU, and writing it first means the kernel is
//! written against a decided contract rather than the other way round.
//!
//! # The claim policy, and why it is narrow
//!
//! `OP_COVERAGE.md` §8.1 sets it: `bits ∈ {4, 8}`, `block_size` a power of two in `[16, 128]`,
//! `g_idx` absent. Those are the forms the ORT GenAI builder emits (`quant_config.py`: dense block
//! 32, 128 for TRT-RTX). `g_idx` (act-order) makes the `B` access pattern data-dependent and
//! destroys the coalesced load that is the entire point of the kernel, so it declines rather than
//! being handled slowly.
//!
//! # Never materialize a dequantized weight tensor
//!
//! Stated here because it is a property of the *op*, not of one kernel: int4 exists so a 1.7B
//! model fits in 1 GB, and dequantizing `B` into device memory before the GEMM gives that back. The
//! unpack happens in registers (decode/GEMV) or in shared memory per tile (prefill/GEMM). This is
//! the load-bearing constraint in the kernel spec handed to Switch.

use crate::kernel;
use crate::ops::common::claim::{self, ClaimResult};
use crate::ops::common::dtype::FLOAT;
use crate::ops::common::templates;
use crate::registry::OpStatus::{Live, Staged};
use crate::registry::{
    ContribSchema, NodeView, OPSET_ANY, OPSET_QDQ_MAX, OpSpec, PINNED_BASELINE, XL_KERNEL,
};
use crate::{deny, require};

/// `com.microsoft.MatMulNBits`, as of ORT 1.28.
///
/// Inputs: `A`, `B`, `scales`, opt `zero_points`, opt `g_idx`, opt `bias`. One output.
///
/// `weight_prepacked` is in `known_attrs` because ORT's own backends set it as a sentinel; we do
/// not honour it (our prepack is ours, at `Compile` time — §8.2), we merely must not treat its
/// presence as schema drift.
pub static MATMUL_NBITS: ContribSchema = ContribSchema {
    baseline: PINNED_BASELINE,
    notes: "ContribOperators.md; 3-input (symmetric RTN) and 4-input (zero-point) forms both \
            observed in Foundry Local graphs, bits 4 and 8, block_size 32",
    min_inputs: 3,
    max_inputs: 6,
    min_outputs: 1,
    max_outputs: 1,
    required_attrs: &["K", "N", "bits", "block_size"],
    known_attrs: &[
        "K",
        "N",
        "bits",
        "block_size",
        "accuracy_level",
        "weight_prepacked",
    ],
};

/// `com.microsoft.GatherBlockQuantized` — the quantized embedding lookup.
pub static GATHER_BLOCK_QUANTIZED: ContribSchema = ContribSchema {
    baseline: PINNED_BASELINE,
    notes: "read from ContribOperators.md",
    min_inputs: 3,
    max_inputs: 4,
    min_outputs: 1,
    max_outputs: 1,
    required_attrs: &[],
    known_attrs: &["bits", "block_size", "gather_axis", "quantize_axis"],
};

/// Input index of `MatMulNBits`'s optional `g_idx` (act-order permutation).
const G_IDX: usize = 4;

/// Input index of `MatMulNBits`'s optional `zero_points`.
const ZERO_POINTS: usize = 3;

/// Input index of `MatMulNBits`'s optional `bias`.
const BIAS: usize = 5;

/// Bit widths this EP will implement.
///
/// Not a taste judgement: `OP_COVERAGE.md` §8.1 checked what ships. GenAI emits 4; 8 is the same
/// kernel with a different unpack. 2/3/5/6/7 are expressible in the schema and used by nobody, and
/// every one we claimed would be a variant to test on five vendors.
pub const fn supports_bits(bits: i64) -> bool {
    matches!(bits, 4 | 8)
}

/// Block sizes this EP will implement: powers of two from 16 to 128.
///
/// The block size decides the scale-index arithmetic and the inner-loop trip count, so an
/// arbitrary value is not "slower", it is a different kernel.
pub const fn supports_block_size(block_size: i64) -> bool {
    matches!(block_size, 16 | 32 | 64 | 128)
}

/// `MatMulNBits`: the entry ticket for int4 LLMs.
fn matmul_nbits(view: &NodeView<'_>, spec: &OpSpec) -> ClaimResult {
    // Activations and output are float; `B`/`zero_points` are packed bytes and are checked by
    // arity and attributes rather than by `caps`, which describes the compute dtype.
    claim::typed_input(view, spec, 0, "input A")?;

    let bits = claim::required_int(view, spec, "bits")?;
    require!(
        supports_bits(bits),
        Attribute,
        "`{}` uses bits = {bits}; this EP implements 4 and 8, which is what real quantized \
         exports use",
        spec.op_type
    );

    let block_size = claim::required_int(view, spec, "block_size")?;
    require!(
        supports_block_size(block_size),
        Attribute,
        "`{}` uses block_size = {block_size}; this EP implements 16/32/64/128",
        spec.op_type
    );

    let k = claim::required_int(view, spec, "K")?;
    let n = claim::required_int(view, spec, "N")?;
    require!(
        k > 0 && n > 0,
        Attribute,
        "`{}` declares K = {k}, N = {n}; both must be positive",
        spec.op_type
    );
    require!(
        k % block_size == 0,
        Attribute,
        "`{}` has K = {k} which is not a multiple of block_size = {block_size}; the ragged final \
         block is a separate kernel path this EP has not written",
        spec.op_type
    );

    // Act-order. Declining is the right answer, not a temporary one: honouring `g_idx` means an
    // indirected, data-dependent read of `B`, which costs more than the CPU fallback saves.
    claim::input_absent(view, spec, G_IDX, "the act-order permutation `g_idx`")?;

    // `bias` is a fused add on the output. The GEMV kernel has no sixth binding and folding the
    // bias into the reduction is a different shader, not a flag — so it declines by name rather
    // than being silently ignored, which would produce a plausible wrong answer. No `MatMulNBits`
    // node in either Foundry Local graph carries one (§4.21).
    claim::input_absent(view, spec, BIAS, "the fused `bias`")?;

    // The kernel writes fp16 outputs by read-modify-writing **disjoint 16-bit lanes** of a shared
    // `uint` word (see `q_gemv.comp`). That is race-free, but it addresses whole words, so the
    // element count must be even or the final store would touch two bytes past the tensor. Every
    // real node satisfies this — Phi-3.5's `N` is 3072/8192/9216/32064 — so the guard costs
    // nothing and removes an out-of-bounds write we would otherwise only find on a strict driver.
    if claim::input_edge(view, spec, 0)?.dtype == Some(crate::engine::DType::F16) {
        let rows = out_rows(view, spec)?;
        require!(
            (rows.saturating_mul(n as u64)) % 2 == 0,
            Shape,
            "`{}` produces an odd number of fp16 output elements ({rows} x {n}); the packed-lane \
             store addresses whole 32-bit words and would write past the tensor",
            spec.op_type
        );
    }

    // `accuracy_level` selects the compute type for the **activation** matrix `A`. It is a hint at
    // every value but one, and that exception is a correctness requirement, not a preference.
    //
    // Read off ORT's CPU kernel rather than the schema prose
    // (`contrib_ops/cpu/quantization/matmul_nbits.cc`, `GetComputeType<T1>`), which has exactly one
    // branch:
    //
    //     if (attr == Level4 && MlasIsQNBitGemmAvailable(nbits, block_size, SQNBIT_CompInt8))
    //         return SQNBIT_CompInt8;
    //     return SQNBIT_CompFp32;          // <float>, i.e. every non-ARM64 host
    //
    // So 0, 1, 2 and 3 all resolve to the same kernel path and are indistinguishable in the
    // output — claiming them is safe and they need no predicate. **4 is the input quantized to
    // int8 with the same `block_size`**, which is a different computation, not a different
    // accumulator width. Our kernel dequantizes `B` and multiplies against `A` at storage
    // precision; running that where the graph asked for int8 activations produces a plausible
    // wrong answer, which is the exact failure mode §7 exists to prevent. Decline it.
    match view.attr_int("accuracy_level") {
        None | Some(0..=3) => {}
        Some(level) => deny!(
            Attribute,
            "`{}` sets `accuracy_level` = {level}; level 4 quantizes the activation `A` to int8 \
             at the weight's block size, and this EP multiplies against `A` at storage precision. \
             Levels 0-3 are claimed because ORT resolves all of them to the same fp32 compute path",
            spec.op_type
        ),
    }

    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────────────────────────
// The GEMV kernel
// ──────────────────────────────────────────────────────────────────────────────────────────────

/// The number of output rows: every dimension of `A` except the reduction extent.
///
/// `A` is `[.., M, K]`, so this is `M` folded together with any leading batch dimensions. The
/// kernel treats them identically — a batch is just more rows — so nothing downstream needs the
/// original rank.
fn out_rows(view: &NodeView<'_>, spec: &OpSpec) -> Result<u64, crate::registry::DeclineReason> {
    let edge = claim::input_edge(view, spec, 0)?;
    let Some(shape) = edge.shape else {
        deny!(
            UnknownRank,
            "`{}` input A has no shape at all, so the row count cannot be formed",
            spec.op_type
        );
    };
    require!(
        shape.len() >= 2,
        Rank,
        "`{}` input A has rank {}; the GEMV needs at least [M, K]",
        spec.op_type,
        shape.len()
    );
    let mut rows: u64 = 1;
    for &d in &shape[..shape.len() - 1] {
        if d < 0 {
            // The row count is exactly the kind of extent that becomes a runtime parameter: the
            // GEMV already takes it as the `m_total` push constant and as `workgroups.y`, so
            // nothing in the shader changes. Until the engine carries it, this declines.
            require!(
                claim::runtime_extents_ok(),
                DynamicShape,
                "`{}` input A has a symbolic leading dimension; the GEMV takes the row count as a \
                 push constant, so this needs only the extent at Compute, not a new kernel",
                spec.op_type
            );
            // Under the counterfactual there is no concrete row count to return; report the
            // minimum viable one so the predicate can finish its remaining checks. This value is
            // never used for dispatch — `ENGINE_ACCEPTS_RUNTIME_EXTENTS` is false, so the only
            // caller that reaches here is the audit's measurement pass.
            return Ok(1);
        }
        rows = rows.saturating_mul(d as u64);
    }
    Ok(rows)
}

/// Floats the GEMV shader reserves for its reduction tree. Mirrors `QB_RED_WORDS` in
/// `q_gemv.comp`; the product `local_size_x * QB_COLS` may not exceed it.
const GEMV_RED_WORDS: u32 = 2048;

/// Largest column tile the GEMV shader can hold in registers. Mirrors `QB_MAX_COLS`.
///
/// **16, raised from 8.** The tile is a byte count before it is a tuning constant: a workgroup
/// re-reads the whole activation row, so the row is read `ceil(N / cols)` times per node. Over
/// Phi-3.5's 161 nodes the static byte model (`bench/results/probe_roofline.py`) puts activation
/// traffic at 887.5 MiB per inference at 8 and 443.7 MiB at 16 — 15.4% of every byte the GEMV
/// island moves, removed without touching the weight stream, which is irreducible.
///
/// Not 32: `GEMV_RED_WORDS` must cover `local_size_x * cols`, and 32 columns at the 128-invocation
/// workgroup `K = 8192` takes needs 16 KiB of shared memory — the entire floor §7.2 guarantees,
/// leaving one resident workgroup on a device that only meets it. See `q_gemv.comp`.
const GEMV_MAX_COLS: u32 = 16;

/// Largest row tile one workgroup may take. Mirrors `QB_MAX_ROWS` in `q_gemv.comp`.
///
/// The row tile is what removes the `M`-fold re-read of the packed weight stream: a workgroup that
/// covers `T` activation rows loads each packed word once and multiply-accumulates it against all
/// `T` of them, so structural weight amplification falls from `M` to `ceil(M / T)`.
///
/// 4 and not 8 because [`GEMV_MAX_TILE`] would then force `cols` to 4 on every Phi-3.5 shape, and
/// the activation stream — `ceil(N / cols)` re-reads of each row — grows exactly as fast as the
/// weight stream shrinks. [`gemv_tile`] picks the pair by a byte model over *both* streams, so the
/// cap only has to be wide enough to contain the optimum.
const GEMV_MAX_ROWS: u32 = 4;

/// Largest `cols * rows` accumulator tile the shader can hold. Mirrors `QB_MAX_TILE`.
///
/// The accumulators are the only per-tile storage that stays live across the whole reduction
/// extent, so this is the register budget expressed as a number the shader can size arrays by.
const GEMV_MAX_TILE: u32 = 32;

/// The `maxComputeWorkGroupCount[1]` every Vulkan 1.1 implementation is required to offer.
///
/// A dispatch above this is not slow, it is **invalid** —
/// `VUID-vkCmdDispatch-groupCountY-00418` — and a long prefill reaches it: at `QB_ROWS = 1` a
/// 70 000-token sequence would ask for 70 000 workgroups in y. The dispatch is clamped here and
/// the shader's y-grid-stride loop covers the remainder, so the geometry stays a function of the
/// *guaranteed floor* rather than of whatever the local device happens to report — the same rule
/// [`GEMV_MIN_WORKGROUPS`] states for the other direction.
pub const GEMV_MAX_GROUPS_Y: u32 = 65_535;

/// Fewest workgroups a dispatch should keep, so tiling never starves the machine of parallelism.
/// Deliberately a small absolute number rather than a multiple of anything the device reports:
/// reading `maxComputeWorkGroupCount` or an SM count here would make the dispatch geometry
/// device-dependent, and a shape that behaves differently per vendor is not a shape we can test.
const GEMV_MIN_WORKGROUPS: u64 = 64;

/// Fewest quantisation blocks each invocation should reduce. Below this the `log2(wg)` barriers of
/// the tree cost more than the arithmetic they synchronise.
const GEMV_MIN_BLOCKS_PER_INVOCATION: u64 = 2;

/// Workgroup size for the reduction, derived from the reduction extent and the **floor**.
///
/// The largest power of two in `[32, 256]` that **divides** `blocks_per_col` and still leaves each
/// invocation at least [`GEMV_MIN_BLOCKS_PER_INVOCATION`] blocks; when nothing divides it, the
/// smallest power of two that covers it, as before. Four deliberate properties:
///
/// * The upper clamp is 256 because that is the `maxComputeWorkGroupInvocations` value the
///   baseline capability set guarantees (`OP_COVERAGE.md` §7.2). It is *not* the larger figure
///   either development GPU reports — sizing to the local device is how a kernel passes every
///   local test and fails on a phone.
/// * It is a power of two because the shared-memory tree reduction halves the stride.
/// * The lower clamp is 32 rather than 1 so a short reduction still fills at least one subgroup on
///   every vendor. That is a *performance* floor expressed without ever reading `subgroupSize`,
///   which is not guaranteed to be any particular value.
/// * **Divides, rather than covers.** The old rule picked the smallest power of two ≥
///   `blocks_per_col`, so `K = 3072` (96 blocks) ran 128 invocations of which 32 had no block at
///   all — they contributed nothing and still had to arrive at all seven barriers. Dividing gives
///   96 = 32 × 3, every invocation with the same amount of work.
///
/// Shared memory is fixed at [`GEMV_RED_WORDS`] floats = 8 KiB regardless, half the 16 KiB
/// floor.
pub fn gemv_workgroup(blocks_per_col: u64) -> u32 {
    let mut best: Option<u32> = None;
    let mut wg: u32 = 32;
    while wg <= 256 {
        let w = u64::from(wg);
        if blocks_per_col % w == 0 && blocks_per_col / w >= GEMV_MIN_BLOCKS_PER_INVOCATION {
            best = Some(wg);
        }
        wg *= 2;
    }
    if let Some(wg) = best {
        return wg;
    }
    let mut wg: u32 = 32;
    while (wg as u64) < blocks_per_col && wg < 256 {
        wg *= 2;
    }
    wg
}

/// Output columns one workgroup computes (`QB_COLS`), the tile that amortises the activation row.
///
/// Every workgroup streams the whole of `A[m][0..K)`. With one column per workgroup that row is
/// re-read `N` times — as *load instructions*, whatever the cache does with the bytes — and the
/// reduction tree with its `log2(wg)` barriers is paid once per output element. Both costs divide
/// by the tile width, so this wants to be as large as the constraints allow:
///
/// * `wg * cols <= GEMV_RED_WORDS`, because the tile reduces inside one shared array;
/// * `cols <= GEMV_MAX_COLS`, the shader's register budget;
/// * `cols` divides `N`, so the paired non-atomic store path is always taken and there is no tail
///   tile — a tail tile is correct (the shader redirects out-of-range columns and re-checks `N` at
///   the store) but it is a second code path executed once per dispatch, and one that only the
///   awkward shapes would ever exercise;
/// * at least [`GEMV_MIN_WORKGROUPS`] workgroups survive, so a narrow output does not trade
///   parallelism for reuse.
///
/// Halving preserves the power of two, and therefore the evenness the paired store needs.
/// Whether one (column, block) blob of packed weights is a whole number of 16-byte units.
///
/// When it is, `q_gemv.comp` fetches it with 128-bit loads instead of four dependent 32-bit ones,
/// and the load is in bounds by construction: the blob starts at a multiple of `blob_bytes` and is
/// itself at least 16 bytes, so every `uvec4` it touches is fully backed. Batch-1 GEMV is
/// bandwidth-bound, so this is the dominant term rather than a micro-optimisation.
pub fn gemv_packed(bits: u32, block_size: u32) -> bool {
    // A controlled A/B needs the two arms to be interleavable on a contended machine, which a
    // rebuild between arms makes impossible. This override exists so the packed path can be
    // switched off in-place and the two builds measured alternately; it is a measurement control,
    // not a tuning knob, and the default is whatever the shape supports.
    if let Ok(v) = std::env::var("ONNXRUNTIME_EP_VULKAN_GEMV_PACKED") {
        return v != "0" && !v.is_empty();
    }
    let blob_bytes = (block_size * bits) / 8;
    blob_bytes % 16 == 0
}

pub fn gemv_cols(n: u64, wg: u32) -> u32 {
    let mut cols = GEMV_MAX_COLS.min((GEMV_RED_WORDS / wg).max(1));
    while cols > 1 && (n % u64::from(cols) != 0 || n / u64::from(cols) < GEMV_MIN_WORKGROUPS) {
        cols /= 2;
    }
    cols
}

/// Bytes the shader's loads *name* for one node under a `(cols, rows)` tile.
///
/// Not a bandwidth estimate and not a cache model: it is the count the SPIR-V walk in
/// `bench/results/probe_weight_reread.py` measures, written down. Every workgroup names `cols`
/// whole packed weight columns and `rows` whole activation rows, and there are
/// `ceil(N / cols) * ceil(M / rows)` of them. Both terms matter and they move in opposite
/// directions with `cols`, which is precisely why the tile cannot be chosen by maximising either
/// one alone.
fn gemv_named_bytes(m: u64, n: u64, k: u64, bits: u32, a_bytes: u64, cols: u32, rows: u32) -> u128 {
    let row_tiles = u128::from(m.div_ceil(u64::from(rows)));
    let col_tiles = u128::from(n.div_ceil(u64::from(cols)));
    let k = u128::from(k);
    let weight = row_tiles * col_tiles * u128::from(cols) * k * u128::from(bits) / 8;
    let activation = row_tiles * col_tiles * u128::from(rows) * k * u128::from(a_bytes);
    weight + activation
}

/// The largest row tile this process will select.
///
/// Defaults to [`GEMV_MAX_ROWS`]. Two jobs, and it is worth being explicit that they are the same
/// mechanism on purpose:
///
/// 1. **The A/B control.** The row tile is a weight-*traffic* change, so its wall-clock effect is
///    only visible by comparing the same shape tiled and untiled. Rebuilding between arms would
///    make the two measurements non-interleavable on a contended machine, which is exactly the
///    failure mode `ONNXRUNTIME_EP_VULKAN_GEMV_PACKED` above exists to avoid. Setting this to `1`
///    restores the pre-issue-#7 geometry in-place.
/// 2. **The fallback.** If a device or driver is ever found on which the tiled arm miscompiles or
///    underperforms, `=1` returns it to the geometry that has 133 ledger entries behind it,
///    without a new build. Setting it to `1` is always safe: `rows == 1` is the seed of the search
///    and the shader's `QB_ROWS == 1u` arm is the verbatim pre-change kernel.
///
/// Values are clamped to `[1, GEMV_MAX_ROWS]` rather than trusted: an operator who writes `64`
/// gets the largest tile that is *legal*, not an overrun. Unparseable values are ignored, because
/// the safe reading of a typo in a performance knob is "the default", not "refuse to run".
///
/// Split from the environment read so the clamping is testable without `unsafe`: `src/ops/` is
/// forbidden raw Vulkan *and* `unsafe` by the layering lint (`tests/layering.rs`), and mutating
/// process environment in a unit test needs one. The env plumbing is covered by
/// `rust/tests/row_tile_fallback.rs` instead, which lives outside that layer.
fn clamp_max_rows(raw: Option<&str>) -> u32 {
    match raw {
        Some(v) => match v.trim().parse::<u32>() {
            Ok(n) => n.clamp(1, GEMV_MAX_ROWS),
            Err(_) => GEMV_MAX_ROWS,
        },
        None => GEMV_MAX_ROWS,
    }
}

fn gemv_max_rows() -> u32 {
    clamp_max_rows(
        std::env::var("ONNXRUNTIME_EP_VULKAN_GEMV_MAX_ROWS")
            .ok()
            .as_deref(),
    )
}

/// The `(cols, rows)` tile one workgroup takes: the pair that names the fewest bytes.
///
/// `rows == 1` is the decode geometry and is *always* what a one-row dispatch gets — the search
/// below does not run at `m <= 1`, so `M = 1` selects the same `cols` it always did and the same
/// specialisation-constant arm of the shader. Everything else is bounded by three things that are
/// properties of the module rather than of a device:
///
/// * `cols * rows <= GEMV_MAX_TILE` — the accumulator register budget;
/// * `rows <= GEMV_MAX_ROWS` — the activation window the tiled arm holds per row;
/// * `wg * cols <= GEMV_RED_WORDS` — unchanged, because the reduction is sequential over rows and
///   reuses one shared array. **The row tile costs no shared memory at all**, so no device that
///   can run the decode kernel can fail to run the tiled one.
///
/// It is fail-closed by construction: `(gemv_cols(n, wg), 1)` is the seed, every candidate has to
/// beat it strictly, and a shape for which no tiled candidate is legal keeps it.
///
/// For a fixed `rows` the largest legal `cols` is always at least as good — the weight term does
/// not depend on `cols` except through tail rounding and the activation term falls as `1 / cols` —
/// so the search takes the widest legal `cols` per `rows` and compares four candidates, not a grid.
pub fn gemv_tile(m: u64, n: u64, k: u64, bits: u32, a_bytes: u64, wg: u32) -> (u32, u32) {
    gemv_tile_with(m, n, k, bits, a_bytes, wg, gemv_max_rows())
}

/// [`gemv_tile`] with the row-tile ceiling passed in rather than read from the environment.
///
/// The whole selection is pure once `max_rows` is a parameter, so every property that matters —
/// the bounds, the strict-improvement rule, the `m <= 1` short circuit, the effect of pinning the
/// ceiling to 1 — is testable without touching process state.
fn gemv_tile_with(
    m: u64,
    n: u64,
    k: u64,
    bits: u32,
    a_bytes: u64,
    wg: u32,
    max_rows: u32,
) -> (u32, u32) {
    let base_cols = gemv_cols(n, wg);
    let mut best = (base_cols, 1u32);
    let mut best_bytes = gemv_named_bytes(m, n, k, bits, a_bytes, base_cols, 1);
    if m <= 1 || max_rows < 2 {
        return best;
    }
    let mut rows = 2u32;
    while rows <= max_rows {
        let mut cols = base_cols;
        loop {
            let legal = cols * rows <= GEMV_MAX_TILE
                && wg * cols <= GEMV_RED_WORDS
                && n % u64::from(cols) == 0
                && (cols == 1 || n / u64::from(cols) >= GEMV_MIN_WORKGROUPS);
            if legal {
                let bytes = gemv_named_bytes(m, n, k, bits, a_bytes, cols, rows);
                if bytes < best_bytes {
                    best = (cols, rows);
                    best_bytes = bytes;
                }
                break;
            }
            if cols == 1 {
                break;
            }
            cols /= 2;
        }
        rows *= 2;
    }
    best
}

// ──────────────────────────────────────────────────────────────────────────────────────────────
// `ONNXRUNTIME_EP_VULKAN_GEMV_TILE` — a strict, fail-closed *request*, not a ceiling
// ──────────────────────────────────────────────────────────────────────────────────────────────
//
// `ONNXRUNTIME_EP_VULKAN_GEMV_MAX_ROWS` above is deliberately lenient: a typo means "the
// default", because a performance ceiling that refuses to run on a bad value is worse than one
// that quietly clamps. This variable has the opposite job and therefore the opposite polarity.
// Its purpose is to let a probe or an A/B *name the exact pipeline it wants*, so that a witness
// reading `spec_constants` back out of `counters::pipeline_variants()` after the run is a witness
// of the tile that was actually asked for, not of whatever `gemv_tile`'s own search happened to
// pick. A knob that can silently substitute a different tile for the one written down defeats
// that purpose as completely as one that silently substitutes the default — so every malformed,
// out-of-range, or illegal value refuses the node *before* a pipeline is built or a buffer bound,
// rather than falling back to anything.
//
// unset ⇒ [`gemv_tile`] runs verbatim. There is no second, duplicated default here: the unset arm
// of [`gemv_tile_choice`] calls the exact function `matmul_nbits_gemv` always called.

/// `ONNXRUNTIME_EP_VULKAN_GEMV_TILE=<cols>,<rows>` — see the module-level note above.
pub const ENV_GEMV_TILE: &str = "ONNXRUNTIME_EP_VULKAN_GEMV_TILE";

/// The `maxComputeWorkGroupCount[0]` every Vulkan 1.1 implementation is required to offer.
///
/// Mirrors [`GEMV_MAX_GROUPS_Y`], for the other grid axis. Production's own dispatch
/// (`workgroups[0] = ceil(N / cols)` in [`matmul_nbits_gemv`]) does not clamp against this today,
/// because no `(cols, rows)` pair [`gemv_tile`]'s own search can return ever pushes it that high
/// on a claimed shape — `cols` only ever gets *smaller* than [`gemv_cols`]'s unclamped seed, which
/// already keeps `ceil(N / cols)` workgroups-worth of parallelism. An explicit
/// `ONNXRUNTIME_EP_VULKAN_GEMV_TILE` request is not bound by that search, so [`gemv_tile_legality`]
/// checks the override against this floor explicitly — the override path is therefore *stricter*
/// than the unset path here, never looser, and the pre-existing gap in the unset path is
/// unaffected (documented, not fixed, in this change: no shader or production-dispatch edit).
const GEMV_MAX_GROUPS_X: u32 = 65_535;

/// Why an explicit [`ENV_GEMV_TILE`] request was refused before dispatch.
///
/// Every variant is `Display`-able to the exact string [`crate::counters::record_gemv_tile_refusal`]
/// latches, so the counters artifact and the `EpError::Internal` message the caller sees say the
/// same thing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemvTileRefusal {
    /// The raw environment value was not valid UTF-8. Read via `env::var_os` rather than
    /// `env::var().ok()` specifically so this state is distinguishable from "unset" — a lossy
    /// read would fold a mangled value into either "unset" (wrong: something was there) or a
    /// silently-substituted lossy string (wrong: exactly the fallback this variable must not do).
    NonUtf8,
    /// The value did not split into exactly two comma-separated fields. Carries the field count
    /// actually found (0 only for a value that is impossible to produce via `split`, 1 for "no
    /// comma" or the empty string, 3+ for extra fields).
    WrongFieldCount(usize),
    /// A field was not `[1-9][0-9]*` or `0` — empty, signed, hex-prefixed, whitespace-padded, or
    /// leading-zero-padded (`"016"`). Leading `'0'` alone is the [`Zero`](Self::Zero) variant, not
    /// this one.
    MalformedField,
    /// A field parsed as all-digits but does not fit in `u32`.
    Overflow,
    /// A field was the literal value `0`. Lexically valid, semantically illegal: no tile has zero
    /// columns or zero rows.
    Zero,
    /// `cols` was in range but exceeds [`GEMV_MAX_COLS`], the shader's register budget.
    IllegalCols { cols: u32 },
    /// `rows` exceeds [`GEMV_MAX_ROWS`] — checked against the **compiled** constant, not
    /// [`gemv_max_rows`]'s environment-adjusted ceiling. An explicit, legal request is meant to
    /// override that ceiling on purpose (see [`gemv_tile_choice`]), so legality for it is judged
    /// against what the shader can actually hold, not against the separate knob's current value.
    IllegalRows { cols: u32, rows: u32 },
    /// `cols * rows` exceeds [`GEMV_MAX_TILE`], the accumulator register budget.
    IllegalTile { cols: u32, rows: u32 },
    /// `wg * cols` exceeds [`GEMV_RED_WORDS`], the shared-memory reduction budget for this node's
    /// own workgroup size.
    IllegalReduction { cols: u32, rows: u32, wg: u32 },
    /// `ceil(N / cols)` exceeds [`GEMV_MAX_GROUPS_X`] for this node's own `N`. See the constant's
    /// doc comment: this is a stricter-than-production check that exists only on this path.
    IllegalGroupsX { cols: u32, n: u64 },
}

impl std::fmt::Display for GemvTileRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonUtf8 => write!(f, "{ENV_GEMV_TILE} is not valid UTF-8"),
            Self::WrongFieldCount(n) => write!(
                f,
                "{ENV_GEMV_TILE} must be exactly `<cols>,<rows>` (one comma, two fields), found {n} field(s)"
            ),
            Self::MalformedField => write!(
                f,
                "{ENV_GEMV_TILE} fields must be `0` or `[1-9][0-9]*` — no sign, no leading zero, no whitespace"
            ),
            Self::Overflow => write!(f, "{ENV_GEMV_TILE} field does not fit in u32"),
            Self::Zero => write!(f, "{ENV_GEMV_TILE} fields must be nonzero"),
            Self::IllegalCols { cols } => write!(
                f,
                "{ENV_GEMV_TILE} cols={cols} exceeds the compiled maximum {GEMV_MAX_COLS}"
            ),
            Self::IllegalRows { cols, rows } => write!(
                f,
                "{ENV_GEMV_TILE}={cols},{rows}: rows={rows} exceeds the compiled maximum {GEMV_MAX_ROWS}"
            ),
            Self::IllegalTile { cols, rows } => write!(
                f,
                "{ENV_GEMV_TILE}={cols},{rows}: cols*rows={} exceeds the accumulator budget {GEMV_MAX_TILE}",
                cols * rows
            ),
            Self::IllegalReduction { cols, rows, wg } => write!(
                f,
                "{ENV_GEMV_TILE}={cols},{rows}: wg*cols={} exceeds the shared-reduction budget {GEMV_RED_WORDS} (wg={wg})",
                wg * cols
            ),
            Self::IllegalGroupsX { cols, n } => write!(
                f,
                "{ENV_GEMV_TILE} cols={cols}: ceil(N/cols)={} exceeds the guaranteed grid floor {GEMV_MAX_GROUPS_X} (N={n})",
                n.div_ceil(u64::from(*cols))
            ),
        }
    }
}

/// Strictly parse one field: `0` or `[1-9][0-9]*`, nothing else.
fn parse_strict_u32(field: &str) -> Result<u32, GemvTileRefusal> {
    if field.is_empty() || !field.bytes().all(|b| b.is_ascii_digit()) {
        return Err(GemvTileRefusal::MalformedField);
    }
    if field.len() > 1 && field.as_bytes()[0] == b'0' {
        return Err(GemvTileRefusal::MalformedField);
    }
    field.parse::<u32>().map_err(|_| GemvTileRefusal::Overflow)
}

/// Parse `<cols>,<rows>` strictly. See [`GemvTileRefusal`] for the exact grammar and every way
/// this refuses rather than falling back.
fn parse_gemv_tile(raw: &std::ffi::OsStr) -> Result<(u32, u32), GemvTileRefusal> {
    let s = raw.to_str().ok_or(GemvTileRefusal::NonUtf8)?;
    let fields: Vec<&str> = s.split(',').collect();
    if fields.len() != 2 {
        return Err(GemvTileRefusal::WrongFieldCount(fields.len()));
    }
    let cols = parse_strict_u32(fields[0])?;
    let rows = parse_strict_u32(fields[1])?;
    if cols == 0 || rows == 0 {
        return Err(GemvTileRefusal::Zero);
    }
    Ok((cols, rows))
}

/// Legality for an explicit [`ENV_GEMV_TILE`] request, for this node's own `wg` and `n`.
///
/// Exactly the bounds the compiled shader and the dispatch it is built from actually enforce —
/// no more (this does not require `cols | n`, because [`gemv_cols`]'s tail-tile path is a real,
/// correct, always-available code path in `q_gemv.comp`, not a shape the shader cannot run — and
/// no less (every one of [`GEMV_MAX_COLS`], the **compiled** [`GEMV_MAX_ROWS`], [`GEMV_MAX_TILE`],
/// [`GEMV_RED_WORDS`] and [`GEMV_MAX_GROUPS_X`] is checked).
fn gemv_tile_legality(cols: u32, rows: u32, wg: u32, n: u64) -> Result<(), GemvTileRefusal> {
    if cols == 0 || cols > GEMV_MAX_COLS {
        return Err(GemvTileRefusal::IllegalCols { cols });
    }
    if rows == 0 || rows > GEMV_MAX_ROWS {
        return Err(GemvTileRefusal::IllegalRows { cols, rows });
    }
    if cols.saturating_mul(rows) > GEMV_MAX_TILE {
        return Err(GemvTileRefusal::IllegalTile { cols, rows });
    }
    if wg.saturating_mul(cols) > GEMV_RED_WORDS {
        return Err(GemvTileRefusal::IllegalReduction { cols, rows, wg });
    }
    if n.div_ceil(u64::from(cols)) > u64::from(GEMV_MAX_GROUPS_X) {
        return Err(GemvTileRefusal::IllegalGroupsX { cols, n });
    }
    Ok(())
}

/// Which of the three mechanisms produced a `(cols, rows)` tile, for tests and witnesses that need
/// to tell them apart even when the numbers happen to coincide.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemvTileSource {
    /// `m <= 1`: the decode null control. The tile is always `(gemv_cols(n, wg), 1)` here, and an
    /// [`ENV_GEMV_TILE`] value — set or unset, legal or not-yet-checked — never changes it. A
    /// harness that only compared numbers could not tell "decode, unaffected" apart from "the
    /// override happened to ask for the same thing"; this variant is that distinction made
    /// explicit and testable.
    Decode,
    /// `ENV_GEMV_TILE` unset at `m > 1`: [`gemv_tile`] ran verbatim, unmodified.
    Selected,
    /// An explicit, legal `ENV_GEMV_TILE` request at `m > 1`: it was substituted for the search.
    Overridden,
}

/// The `(cols, rows)` tile [`matmul_nbits_gemv`] actually dispatches, given the raw
/// [`ENV_GEMV_TILE`] value (or `None` if unset).
///
/// This is the seam: `matmul_nbits_gemv` calls this instead of [`gemv_tile`] directly, so a test
/// that drives it — or drives `matmul_nbits_gemv` itself through `registry::spec_for` — observes
/// exactly what production does. Three rules, checked in this order:
///
/// 1. **`m <= 1` is the decode null control and always wins.** A request is still parsed and
///    validated here (so a malformed value refuses even a decode-shaped node — this mechanism
///    never silently ignores an illegal value, at any `m`), but a *legal* one is deliberately set
///    aside rather than applied: decode never takes a row tile, full stop.
/// 2. **Unset is [`gemv_tile`], verbatim.** No duplicated default: this arm is the same call
///    `matmul_nbits_gemv` always made.
/// 3. **A legal request at `m > 1` is substituted whole**, checked against the *compiled* maxima
///    ([`gemv_tile_legality`]) rather than [`gemv_max_rows`]'s environment-adjusted ceiling — an
///    explicit `ENV_GEMV_TILE` intentionally overrides `ONNXRUNTIME_EP_VULKAN_GEMV_MAX_ROWS`
///    rather than refusing on the contradiction between the two variables.
fn gemv_tile_choice_with(
    m: u64,
    n: u64,
    k: u64,
    bits: u32,
    a_bytes: u64,
    wg: u32,
    raw: Option<&std::ffi::OsStr>,
) -> Result<((u32, u32), GemvTileSource), GemvTileRefusal> {
    let Some(raw) = raw else {
        let tile = gemv_tile(m, n, k, bits, a_bytes, wg);
        let source = if m <= 1 {
            GemvTileSource::Decode
        } else {
            GemvTileSource::Selected
        };
        return Ok((tile, source));
    };
    let (cols, rows) = parse_gemv_tile(raw)?;
    gemv_tile_legality(cols, rows, wg, n)?;
    if m <= 1 {
        return Ok(((gemv_cols(n, wg), 1), GemvTileSource::Decode));
    }
    Ok(((cols, rows), GemvTileSource::Overridden))
}

/// [`gemv_tile_choice_with`], reading [`ENV_GEMV_TILE`] from the process environment.
///
/// Split the same way [`gemv_tile`]/[`gemv_tile_with`] and [`gemv_max_rows`]/[`clamp_max_rows`]
/// already are, for the same reason: the environment read is the one line `rust/tests/*.rs` has
/// to cover from outside `src/ops/`'s `unsafe`-forbidding layer, and everything else is a pure
/// function unit-testable in place.
pub fn gemv_tile_choice(
    m: u64,
    n: u64,
    k: u64,
    bits: u32,
    a_bytes: u64,
    wg: u32,
) -> Result<((u32, u32), GemvTileSource), GemvTileRefusal> {
    gemv_tile_choice_with(
        m,
        n,
        k,
        bits,
        a_bytes,
        wg,
        std::env::var_os(ENV_GEMV_TILE).as_deref(),
    )
}

/// Translate `MatMulNBits` into one block-dequantising GEMV dispatch.
fn matmul_nbits_gemv(
    spec: &OpSpec,
    node: &crate::engine::NodeDesc,
    ctx: &mut dyn crate::engine::DispatchContext,
) -> crate::engine::EpResult<()> {
    use crate::engine::{AttrValue, EpError, KernelRequest, TensorDesc};

    let attr = |name: &str| -> crate::engine::EpResult<i64> {
        match node.attributes.get(name) {
            Some(AttrValue::Int(v)) => Ok(*v),
            _ => Err(EpError::Internal(format!(
                "`{}` was claimed but has no integer attribute `{name}`",
                node.op_type
            ))),
        }
    };
    let bits = attr("bits")?;
    let block_size = attr("block_size")?;
    let k = attr("K")?;
    let n = attr("N")?;
    let blocks_per_col = k / block_size;

    let a_desc = node
        .inputs
        .first()
        .and_then(|t| t.desc.as_ref())
        .ok_or_else(|| {
            EpError::Unsupported(format!(
                "`{}` input A has no shape at compile time",
                node.op_type
            ))
        })?;
    let dtype = a_desc.dtype;
    let rank = a_desc.shape.len();
    if rank < 2 {
        return Err(EpError::Internal(format!(
            "`{}` was claimed with a rank-{rank} `A`",
            node.op_type
        )));
    }
    let m_total: i64 = a_desc.shape[..rank - 1].iter().product();

    let shader = spec.kernel.stem(dtype).ok_or_else(|| {
        EpError::Internal(format!(
            "`{}` was claimed but its row declares no shader for {dtype:?}",
            node.op_type
        ))
    })?;

    let a = ctx.resolve(&node.inputs[0])?;
    let b = ctx.resolve(&node.inputs[1])?;
    let scales = ctx.resolve(&node.inputs[2])?;
    // Binding 3 must be bound whether or not the node has zero points: the shader declares it, and
    // a declared descriptor is part of the layout even when specialisation folds every read of it
    // away. Rebinding `scales` is an inert placeholder — `QB_HAS_ZP == 0` makes `load_zp` return
    // the implied `1 << (bits-1)` without touching the buffer. Note this deliberately does *not*
    // call `resolve` a fourth time: `CompileRecorder` assigns buffer tokens positionally, so an
    // extra resolve would shift every later token.
    let has_zp = node
        .inputs
        .get(ZERO_POINTS)
        .is_some_and(|t| !t.name.is_empty());
    let zp = if has_zp {
        ctx.resolve(&node.inputs[ZERO_POINTS])?
    } else {
        scales
    };

    let out = node
        .outputs
        .first()
        .ok_or_else(|| EpError::Internal(format!("`{}` has no output", node.op_type)))?;
    let mut out_shape = a_desc.shape[..rank - 1].to_vec();
    out_shape.push(n);
    let out_dtype = out.desc.as_ref().map_or(dtype, |d| d.dtype);
    let y = ctx.bind_output(out, TensorDesc::new(out_dtype, out_shape))?;

    let wg = gemv_workgroup(blocks_per_col as u64);
    let a_bytes = dtype.byte_size() as u64;
    // FAIL CLOSED before anything downstream reads `cols`/`rows`: an explicit
    // `ONNXRUNTIME_EP_VULKAN_GEMV_TILE` that is malformed, non-UTF-8, out of range, or otherwise
    // illegal refuses this node here rather than substituting any default. `resolve`/`bind_output`
    // above only assign buffer tokens; no pipeline is built and no dispatch recorded until below.
    let (cols, rows) = match gemv_tile_choice(
        m_total.max(0) as u64,
        n as u64,
        k as u64,
        bits as u32,
        a_bytes,
        wg,
    ) {
        Ok((tile, _source)) => tile,
        Err(reason) => {
            crate::counters::record_gemv_tile_refusal(&reason.to_string());
            return Err(EpError::Internal(format!(
                "`{}` refused an explicit {ENV_GEMV_TILE}: {reason}",
                node.op_type
            )));
        }
    };
    let mut push = Vec::with_capacity(16);
    for v in [m_total, k, n, blocks_per_col] {
        push.extend_from_slice(&(v as u32).to_le_bytes());
    }

    // FAIL CLOSED, at the last point that can still refuse. `q_gemv.comp` addresses its
    // accumulators `r * QB_COLS + c` into arrays of `QB_MAX_TILE`, so a pair whose product
    // overruns that would write out of bounds. `gemv_tile` cannot return one — the bound is a
    // condition of its own search — but a pipeline is built from these two numbers and nothing
    // downstream re-derives them. An edit that broke the invariant would otherwise be found by a
    // GPU fault or, worse, by silently wrong logits. The shader carries the same guard as a
    // folded specialisation-constant branch; this one names the values in the error.
    if cols * rows > GEMV_MAX_TILE || rows > GEMV_MAX_ROWS || wg * cols > GEMV_RED_WORDS {
        return Err(EpError::Internal(format!(
            "`{}` selected an illegal GEMV tile: cols={cols} rows={rows} wg={wg} \
             (cols*rows must be <= {GEMV_MAX_TILE}, rows <= {GEMV_MAX_ROWS}, \
             wg*cols <= {GEMV_RED_WORDS})",
            node.op_type
        )));
    }

    // `groupCountY` is clamped to the floor Vulkan guarantees rather than to whatever this device
    // reports; the shader's y-grid-stride loop covers everything the clamp cut off. A dispatch
    // above the device's limit is invalid, not slow — VUID-vkCmdDispatch-groupCountY-00418 — and
    // `rust/tests/validation_control.rs` carries the positive control that the layer says so.
    let row_tiles = (m_total.max(0) as u64).div_ceil(u64::from(rows));
    let groups_y = row_tiles.min(u64::from(GEMV_MAX_GROUPS_Y)) as u32;

    ctx.dispatch(KernelRequest {
        shader,
        spec_constants: vec![
            wg,
            bits as u32,
            block_size as u32,
            u32::from(has_zp),
            cols,
            u32::from(gemv_packed(bits as u32, block_size as u32)),
            rows,
        ],
        push_constants: push,
        bindings: vec![a, b, scales, zp, y],
        workgroups: [(n as u32).div_ceil(cols), groups_y, 1],
    })
}

// ──────────────────────────────────────────────────────────────────────────────────────────────
// Prepacking
// ──────────────────────────────────────────────────────────────────────────────────────────────

/// The pure `PackInput -> PackOutput` transform for `MatMulNBits` (§8.2 P6).
///
/// # Why this is currently a pass-through, and why that is the honest answer
///
/// The temptation with a prepack seam is to invent a transform to justify it. The GEMV does not
/// need one: ONNX's `B` layout is `[N][blocks_per_col][blob_bytes]`, which is already contiguous
/// **per output column**, and a workgroup-per-column GEMV streams exactly that. Reordering it
/// would cost an upload-time pass and buy nothing measurable.
///
/// What prepacking *does* earn here is the second thing this function does: **the zero-point
/// buffer is synthesised when the graph omits it**, in ORT's own packed-nibble form. That gives
/// one shader path for symmetric (Phi-3.5, 3 inputs) and asymmetric (gpt-oss, 4 inputs) graphs
/// rather than two, at a memory cost of `1/(2*block_size)` of the weight — 1.6% at 4-bit/32.
/// Expanding zero points to a byte per block, the obvious alternative, costs eight times that for
/// a nibble extraction amortised over 32 multiply-accumulates.
///
/// Prepacking earns its keep properly at the GEMM stage, where the tile layout genuinely differs
/// from the ONNX one. Saying so here is cheaper than discovering later that the pass-through was
/// load-bearing.
pub fn prepack_matmul_nbits(input: crate::engine::PackInput<'_>) -> crate::engine::PackOutput {
    crate::engine::PackOutput {
        packed_weight: input.weight.to_vec(),
        packed_scales: input.scales.to_vec(),
        packed_zero_points: input.zero_points.map(<[u8]>::to_vec),
    }
}

/// The byte value that fills a synthesised 4-bit zero-point buffer: 8 in both nibbles.
///
/// Derived from the CPU EP rather than from the schema prose (§8.1.1): with `zero_points` absent
/// and every weight nibble set to `i`, the dequantised column reads `i - 8`.
pub const IMPLIED_ZP_4BIT: u8 = 0x88;

/// The same for 8 bits.
pub const IMPLIED_ZP_8BIT: u8 = 0x80;

/// Build the zero-point bytes a node without a `zero_points` input implies.
///
/// Kept next to the kernel that would consume it. Not yet reachable: the engine does not call
/// `compile_hook_for` yet, so no `PrepackRequest` is ever processed, and the shader carries the
/// implied zero point in a specialisation constant instead. When the hook lands, this is what
/// replaces `QB_HAS_ZP`.
pub fn implied_zero_points(bits: i64, blocks_per_col: usize, n: usize) -> Vec<u8> {
    let (fill, per_col) = match bits {
        4 => (IMPLIED_ZP_4BIT, blocks_per_col.div_ceil(2)),
        _ => (IMPLIED_ZP_8BIT, blocks_per_col),
    };
    vec![fill; per_col * n]
}

/// `GatherBlockQuantized`: `Gather` whose rows are dequantized on the way out.
fn gather_block_quantized(view: &NodeView<'_>, spec: &OpSpec) -> ClaimResult {
    let bits = view.attr_int("bits").unwrap_or(4);
    require!(
        supports_bits(bits),
        Attribute,
        "`{}` uses bits = {bits}; this EP implements 4 and 8",
        spec.op_type
    );
    let block_size = view.attr_int("block_size").unwrap_or(128);
    require!(
        supports_block_size(block_size),
        Attribute,
        "`{}` uses block_size = {block_size}; this EP implements 16/32/64/128",
        spec.op_type
    );
    Ok(())
}

/// Quantization axis modes for `QuantizeLinear`/`DequantizeLinear`.
///
/// The three are genuinely different index computations, which is why the claim predicate has to
/// distinguish them rather than treating scale as "some tensor".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantMode {
    /// Scalar scale and zero-point: one value for the whole tensor.
    PerTensor,
    /// One scale per slice along `axis`.
    PerAxis,
    /// One scale per `block_size` elements along `axis` (opset 21+).
    Blocked,
}

/// Classify a Q/DQ node from its scale rank and `block_size` attribute.
///
/// Pure, so it is unit-testable without a graph — the point of factoring it out.
pub const fn quant_mode(scale_rank: usize, block_size: Option<i64>) -> QuantMode {
    match block_size {
        Some(b) if b > 0 => QuantMode::Blocked,
        _ if scale_rank == 0 => QuantMode::PerTensor,
        _ => QuantMode::PerAxis,
    }
}

/// `QuantizeLinear` / `DequantizeLinear`, including the blocked (opset 21) form.
fn quant_linear(view: &NodeView<'_>, spec: &OpSpec) -> ClaimResult {
    require!(
        view.num_inputs() >= 2,
        Arity,
        "`{}` needs at least data and scale",
        spec.op_type
    );
    let scale = claim::input_edge(view, spec, 1)?;
    let Some(scale_rank) = scale.rank() else {
        deny!(
            UnknownRank,
            "`{}` scale has no shape at all; the scale-index mode is chosen by rank, so it cannot \
             be decided",
            spec.op_type
        );
    };
    let mode = quant_mode(scale_rank, view.attr_int("block_size"));
    if mode == QuantMode::Blocked {
        let block_size = view.attr_int("block_size").unwrap_or(0);
        require!(
            supports_block_size(block_size),
            Attribute,
            "`{}` uses block_size = {block_size}; this EP implements 16/32/64/128, matching the \
             MatMulNBits block-index math it shares",
            spec.op_type
        );
    }
    // `output_dtype` (opset 21) selects a target type; only claim it when it names a type the
    // engine can store, which `caps` already describes.
    match view.attr_int("output_dtype") {
        None => {}
        Some(0) => {}
        Some(d) => deny!(
            Attribute,
            "`{}` sets `output_dtype` = {d}; this EP infers the target type from the \
             zero-point/output edge instead",
            spec.op_type
        ),
    }
    // `precision` selects the accumulation precision of the `x / y_scale` division. Two readings of
    // the schema history disagree about whether it arrived at 23 or at 25 (§4.20); declining every
    // non-default value is correct under both, and costs nothing because no producer we census
    // emits it. It is a *numeric* attribute, so guessing it is the `accuracy_level` failure mode
    // Trinity measured on the oracle — a silently different answer, not a visibly wrong one.
    match view.attr_int("precision") {
        None | Some(0) => {}
        Some(p) => deny!(
            Attribute,
            "`{}` sets `precision` = {p}; this EP computes the scale division at the storage \
             precision and will not silently substitute a different accumulator",
            spec.op_type
        ),
    }
    Ok(())
}

crate::op_table! {
    // -------------------------------------------------------------------------------------------
    // Weight-only quantization. `Ms` rows carry a schema fingerprint instead of an opset window.
    //
    //  op                        domain  opsets            caps    kernel          claim                  translate                  status               schema
    // -------------------------------------------------------------------------------------------
    "MatMulNBits",                Ms,     1 ..= OPSET_ANY,  FLOAT,  kernel!(QGemv, "matmul_nbits"), matmul_nbits,     matmul_nbits_gemv,         Live,                schema: &MATMUL_NBITS;
    "GatherBlockQuantized",       Ms,     1 ..= OPSET_ANY,  FLOAT,  kernel!(None),  gather_block_quantized, templates::unimplemented, Staged(XL_KERNEL),   schema: &GATHER_BLOCK_QUANTIZED;

    // The `ai.onnx` half of the quantized path. Opset 21 introduced the blocked form, which is the
    // one the LLM path needs, so the window starts there rather than at 10. `caps` names the
    // *compute* dtype (the float side of the conversion); the packed side is checked by the
    // predicate, not by the capability set.
    //
    // The window is **closed** at 25 rather than open-ended, and 25 is also the *newest schema
    // version that exists*: `operator_sets.h` registers no Q/DQ at 26 or 27 (opset 26 is `BitCast`
    // and `CumProd`; opset 27 is `CausalConvWithState`, `LinearAttention` and `Range`). So this
    // window declines nothing that ONNX has published — it is complete coverage of the op, not a
    // restriction — while still failing closed on a future revision. §4.20.
    "DequantizeLinear",           Ai,     21 ..= OPSET_QDQ_MAX, FLOAT,  kernel!(None),  quant_linear,          templates::unimplemented,  Staged(XL_KERNEL);
    "QuantizeLinear",             Ai,     21 ..= OPSET_QDQ_MAX, FLOAT,  kernel!(None),  quant_linear,          templates::unimplemented,  Staged(XL_KERNEL);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{Domain, OpStatus};

    fn row(name: &str) -> &'static OpSpec {
        OPS.iter().find(|s| s.op_type == name).expect("row exists")
    }

    #[test]
    fn the_committed_quant_ops_all_have_rows() {
        for op in [
            "MatMulNBits",
            "GatherBlockQuantized",
            "DequantizeLinear",
            "QuantizeLinear",
        ] {
            let _ = row(op);
        }
    }

    /// Q/DQ's window is closed at 25 — which is also the newest version that exists.
    ///
    /// 21 added `block_size`/`output_dtype`; 24 admitted `float8e8m0` scales; 25 added `uint2`/
    /// `int2` and (on one reading) `precision`. `operator_sets.h` registers no Q/DQ at 26 or 27, so
    /// the closed bound declines nothing ONNX has published while still failing closed on a future
    /// revision. This test is the pin for that claim — §4.20.
    #[test]
    fn qdq_windows_are_closed_at_the_newest_schema_read() {
        for op in ["DequantizeLinear", "QuantizeLinear"] {
            let spec = row(op);
            assert_eq!(
                (spec.min_opset, spec.max_opset),
                (21, OPSET_QDQ_MAX),
                "{op}"
            );
            assert_ne!(spec.max_opset, OPSET_ANY, "{op}");
            assert!(
                spec.max_opset < crate::registry::ONNX_OPSET_REGISTERED,
                "{op}: 25 is below the registered maximum 27 by fact, not by caution — if a Q/DQ \
                 revision ever appears at 26 or 27 this bound must be re-read, not raised"
            );
        }
    }

    #[test]
    fn only_the_bit_widths_that_ship_are_claimable() {
        assert!(supports_bits(4), "GenAI's default");
        assert!(supports_bits(8));
        for unshipped in [2, 3, 5, 6, 7] {
            assert!(
                !supports_bits(unshipped),
                "{unshipped}-bit is expressible in the schema and used by nobody; claiming it is \
                 five vendors' worth of testing for no model"
            );
        }
    }

    #[test]
    fn block_sizes_are_the_powers_of_two_genai_emits() {
        for good in [16, 32, 64, 128] {
            assert!(supports_block_size(good));
        }
        for bad in [0, 1, 24, 48, 96, 256] {
            assert!(
                !supports_block_size(bad),
                "{bad} is not a claimed block size"
            );
        }
    }

    #[test]
    fn quant_mode_classification() {
        assert_eq!(quant_mode(0, None), QuantMode::PerTensor);
        assert_eq!(quant_mode(1, None), QuantMode::PerAxis);
        assert_eq!(quant_mode(1, Some(32)), QuantMode::Blocked);
        assert_eq!(quant_mode(2, Some(128)), QuantMode::Blocked);
        // `block_size = 0` is ONNX's "not blocked" spelling, not a zero-size block.
        assert_eq!(quant_mode(1, Some(0)), QuantMode::PerAxis);
    }

    #[test]
    fn contrib_rows_are_in_the_ms_domain_and_ai_rows_are_not() {
        assert_eq!(row("MatMulNBits").domain, Domain::Ms);
        assert!(row("MatMulNBits").schema.is_some());
        assert_eq!(row("DequantizeLinear").domain, Domain::Ai);
        assert!(row("DequantizeLinear").schema.is_none());
    }

    #[test]
    fn the_matmul_nbits_schema_knows_the_attributes_genai_writes() {
        for attr in ["K", "N", "bits", "block_size", "accuracy_level"] {
            assert!(
                MATMUL_NBITS.knows(attr),
                "`{attr}` is written by the GenAI builder; not knowing it would decline every \
                 real node as schema drift"
            );
        }
        assert!(
            !MATMUL_NBITS.knows("g_idx"),
            "g_idx is an input, not an attribute"
        );
    }

    /// What `accuracy_level` actually controls, pinned so the reasoning survives the comment.
    ///
    /// Trinity's oracle pins `accuracy_level = 1` while both Foundry Local models emit `0` on
    /// every `MatMulNBits` node (§4.21). The question was whether the pin diverges from the model.
    /// It does not, and the reason is not that the attribute is cosmetic: ORT's CPU kernel branches
    /// on it exactly once, for level 4, and returns the fp32 compute path for everything else. So
    /// 0, 1, 2 and 3 are one path, and 4 is a genuinely different computation — int8 activations.
    ///
    /// `SUPPORTED_ACCURACY_LEVELS` is therefore not a preference list. It is the boundary between
    /// "hint we may ignore" and "requirement we do not meet".
    const SUPPORTED_ACCURACY_LEVELS: &[i64] = &[0, 1, 2, 3];

    #[test]
    fn accuracy_level_4_is_the_only_one_that_changes_the_computation() {
        assert!(
            !SUPPORTED_ACCURACY_LEVELS.contains(&4),
            "level 4 quantizes the activation to int8; a kernel that multiplies at storage \
             precision would return a plausible wrong answer for it"
        );
        for level in [0, 1, 2, 3] {
            assert!(
                SUPPORTED_ACCURACY_LEVELS.contains(&level),
                "ORT resolves accuracy_level {level} to the same fp32 compute path, so declining \
                 it would decline every real graph for no numerical reason"
            );
        }
        // The census values, so a change to either end of this is visible.
        assert!(
            SUPPORTED_ACCURACY_LEVELS.contains(&0),
            "both Foundry Local models emit 0 on every node"
        );
    }

    /// ORT `ORT_ENFORCE`s these ranges at kernel construction, so a fixture outside them fails
    /// before any comparison happens. Ours must be a subset, or we would claim nodes the oracle
    /// cannot even build.
    #[test]
    fn our_bits_and_block_sizes_are_inside_what_ort_will_construct() {
        assert!(
            !supports_bits(2),
            "2-bit is legal in ORT; we do not implement it"
        );
        assert!(supports_bits(4) && supports_bits(8));
        assert!(!supports_bits(3) && !supports_bits(16));
        for bs in [16, 32, 64, 128] {
            assert!(supports_block_size(bs), "ORT accepts {bs} and so do we");
        }
        assert!(
            !supports_block_size(256),
            "256 is legal in ORT but our kernel has not been written for it"
        );
        assert!(!supports_block_size(48), "not a size ORT will construct");
    }

    /// `MatMulNBits` is the only live row here, and the reason is a measurement rather than a
    /// preference: it is 161 of Phi-3.5's 366 nodes, and the island simulation puts the graph at
    /// 34–35 islands without it and one island of 364 with it. Nothing else in this module changes
    /// the partition at all, so nothing else earns a kernel yet.
    #[test]
    fn matmul_nbits_is_live_and_the_rest_are_not() {
        for s in OPS {
            if s.op_type == "MatMulNBits" {
                assert_eq!(s.status, OpStatus::Live);
                assert_ne!(
                    s.kernel.template,
                    crate::ops::common::variants::Template::None,
                    "a live row must name a shader"
                );
                continue;
            }
            assert!(
                matches!(s.status, OpStatus::Staged(_)),
                "{} claims to be live but has no kernel",
                s.op_type
            );
        }
    }

    /// The workgroup size is derived from the guaranteed floor, never from a local device.
    #[test]
    fn gemv_workgroup_respects_the_floor() {
        // K=3072 at block 32 -> 96 blocks. 96 = 32 x 3, so 32 invocations each reduce three
        // blocks. The old rule picked 128 — the smallest power of two that COVERS 96 — and left
        // 32 invocations with no block at all, arriving at all seven barriers to contribute zero.
        assert_eq!(gemv_workgroup(96), 128 / 4);
        assert_eq!(96 % gemv_workgroup(96) as u64, 0, "divides, never covers");
        // K=8192 at block 32 -> 256 blocks -> 128, the largest divisor that still leaves two
        // blocks per invocation. Never 256: that is one block each, all barrier and no work.
        assert_eq!(gemv_workgroup(256), 128);
        assert_eq!(
            gemv_workgroup(4096),
            256,
            "never exceeds the guaranteed floor of §7.2, not the 1024 both dev GPUs report"
        );
        // A short reduction still fills a subgroup on every vendor without reading `subgroupSize`.
        assert_eq!(gemv_workgroup(1), 32);
        // Nothing divides 7 usefully, so the covering rule is the fallback, not the default.
        assert_eq!(gemv_workgroup(7), 32);
        for blocks in [1u64, 7, 96, 128, 255, 256, 10_000] {
            let wg = gemv_workgroup(blocks);
            assert!((32..=256).contains(&wg));
            assert!(wg.is_power_of_two(), "the tree reduction halves the stride");
        }
    }

    /// A packed load is legal when the *blob* is 16 bytes, not when the block is 32.
    #[test]
    fn gemv_packed_tracks_the_blob_and_not_the_block() {
        // The Phi-3.5 shape: 32 weights of 4 bits is a 16-byte blob, so the 128-bit path is live.
        assert!(gemv_packed(4, 32), "block 32 at 4 bits is a 16-byte blob");
        assert!(
            gemv_packed(8, 16),
            "block 16 at 8 bits is also a 16-byte blob"
        );
        assert!(
            gemv_packed(4, 64),
            "block 64 at 4 bits is 32 bytes, two whole uvec4"
        );
        // Block 16 at 4 bits is an 8-byte blob: half a uvec4, so the vectorised load would
        // straddle two blobs and the scalar path must be selected instead. This is the case that
        // makes the spec constant necessary rather than decorative.
        assert!(
            !gemv_packed(4, 16),
            "block 16 at 4 bits is 8 bytes, not a whole uvec4"
        );
    }

    #[test]
    fn gemv_cols_never_outruns_the_shared_array_or_starves_the_machine() {
        // The four Phi-3.5 shapes. Every one takes the widest tile the shader can hold.
        for (n, k) in [
            (9216u64, 3072u64),
            (3072, 3072),
            (16384, 3072),
            (3072, 8192),
        ] {
            let wg = gemv_workgroup(k / 32);
            let cols = gemv_cols(n, wg);
            assert_eq!(cols, 16, "N={n} K={k} should take the full tile");
            assert!(
                wg * cols <= GEMV_RED_WORDS,
                "N={n} K={k}: {wg} x {cols} would overrun the {GEMV_RED_WORDS}-float `red` array"
            );
            assert!(
                wg * cols * 4 <= 8 * 1024,
                "N={n} K={k}: {wg} x {cols} floats is more than half the 16 KiB shared-memory \
                 floor of §7.2, which would leave one resident workgroup on a device that only \
                 meets the floor"
            );
            assert_eq!(n % u64::from(cols), 0, "no tail tile for a Phi-3.5 shape");
        }
        // The invariant, over a grid that includes the awkward shapes.
        for n in [1u64, 2, 3, 7, 64, 100, 512, 3072, 32_064] {
            for bpc in [1u64, 7, 96, 256, 4096] {
                let wg = gemv_workgroup(bpc);
                let cols = gemv_cols(n, wg);
                assert!(
                    (1..=GEMV_MAX_COLS).contains(&cols),
                    "n={n} bpc={bpc}: cols={cols} outside 1..={GEMV_MAX_COLS}"
                );
                assert!(
                    cols.is_power_of_two(),
                    "the paired store needs an even tile"
                );
                assert!(
                    wg * cols <= GEMV_RED_WORDS,
                    "n={n} bpc={bpc}: {wg} x {cols} overruns `red`"
                );
                assert!(
                    cols == 1 || n / u64::from(cols) >= GEMV_MIN_WORKGROUPS,
                    "n={n}: tiling to {cols} left too few workgroups"
                );
            }
        }
        // A narrow output keeps its parallelism rather than trading it for reuse.
        assert_eq!(
            gemv_cols(64, 32),
            1,
            "64 columns cannot afford an 8-wide tile"
        );
    }

    /// `M = 1` must select the decode geometry unchanged, whatever else the tile picker learns.
    ///
    /// This is the load-bearing property of the whole row-tile change: `rows == 1` selects the
    /// specialisation-constant arm of `q_gemv.comp` that is the previous kernel's text, so decode
    /// is bit-identical rather than merely close.
    #[test]
    fn a_single_row_always_takes_the_decode_tile() {
        for (n, k) in [
            (9216u64, 3072u64),
            (3072, 3072),
            (32_064, 3072),
            (3072, 8192),
            (130, 512),
            (64, 512),
        ] {
            let wg = gemv_workgroup(k / 32);
            for a_bytes in [2u64, 4] {
                let (cols, rows) = gemv_tile(1, n, k, 4, a_bytes, wg);
                assert_eq!(rows, 1, "N={n} K={k}: decode must not be tiled");
                assert_eq!(
                    cols,
                    gemv_cols(n, wg),
                    "N={n} K={k}: decode must keep the column tile it always had"
                );
            }
        }
        // Zero rows is a degenerate dispatch, not a tiled one.
        assert_eq!(gemv_tile(0, 3072, 3072, 4, 2, 32).1, 1);
    }

    /// The tile the byte model picks for the real prefill shapes, and why it is that one.
    ///
    /// Phi-3.5's five `MatMulNBits` shapes all reduce `K = 3072` or `K = 8192` at block 32, so
    /// `gemv_workgroup` gives 32 or 128 and `gemv_cols` gives 16. Because `ceil(N/cols) * cols` is
    /// just `N` whenever the column tile divides `N`, the weight term depends only on `rows` and
    /// the activation term only on `rows / cols` — so `rows = 2, cols = 16` and `rows = 4,
    /// cols = 8` name *exactly* the same bytes. The picker requires a strict improvement, so the
    /// tie goes to the first candidate, which is the one that keeps the decode column tile and
    /// runs half as many sequential shared-memory reductions.
    ///
    /// `rows = 4, cols = 16` would be strictly better again, but `cols * rows = 64` exceeds
    /// `GEMV_MAX_TILE`: the accumulator budget, not the byte model, is what bounds the tile.
    #[test]
    fn a_prefill_shape_takes_a_row_tile_and_it_is_the_cheapest_legal_one() {
        let (n, k) = (3072u64, 3072u64);
        let wg = gemv_workgroup(k / 32);
        for m in [2u64, 16, 64, 512, 2048] {
            let (cols, rows) = gemv_tile(m, n, k, 4, 2, wg);
            assert_eq!(
                (cols, rows),
                (16, 2),
                "M={m}: expected the cheapest legal row tile"
            );
            assert!(cols * rows <= GEMV_MAX_TILE);
            assert!(
                wg * cols <= GEMV_RED_WORDS,
                "the row tile must not enlarge the shared-memory requirement"
            );
            // Strictly fewer bytes named than the untiled geometry: that is the change.
            let tiled = gemv_named_bytes(m, n, k, 4, 2, cols, rows);
            let untiled = gemv_named_bytes(m, n, k, 4, 2, gemv_cols(n, wg), 1);
            assert!(
                tiled < untiled,
                "M={m}: {tiled} is not fewer than {untiled}"
            );
        }
        // The alternative really is a tie rather than merely close, and the wider tile that would
        // break the tie is excluded by the accumulator cap rather than by the model.
        assert_eq!(
            gemv_named_bytes(64, n, k, 4, 2, 16, 2),
            gemv_named_bytes(64, n, k, 4, 2, 8, 4)
        );
        assert!(gemv_named_bytes(64, n, k, 4, 2, 16, 4) < gemv_named_bytes(64, n, k, 4, 2, 16, 2));
        // `16 * 4 > GEMV_MAX_TILE` is the whole reason the strictly-better candidate is refused,
        // and it is a fact about the constant rather than about this shape — so it is asserted
        // against the constant's value, which a future edit could change.
        assert_eq!(
            GEMV_MAX_TILE, 32,
            "the accumulator cap is what excludes the (16, 4) tile"
        );
        // The K = 8192 shape reduces 256 blocks, so `wg` is 128 and `wg * cols` is already at the
        // shared-array ceiling. The row tile still applies, because it costs no shared memory.
        let wg8 = gemv_workgroup(8192 / 32);
        assert_eq!(wg8, 128);
        let (cols8, rows8) = gemv_tile(512, 3072, 8192, 4, 2, wg8);
        assert!(rows8 > 1, "the row tile must survive the widest workgroup");
        assert_eq!(wg8 * cols8, GEMV_RED_WORDS);
    }

    /// Every tile the picker can emit, over a grid that includes the awkward shapes.
    #[test]
    fn every_selected_tile_respects_every_static_bound() {
        for m in [0u64, 1, 2, 3, 5, 17, 128, 4096] {
            for n in [1u64, 2, 3, 7, 64, 100, 130, 512, 3072, 32_064] {
                for bpc in [1u64, 7, 96, 256, 4096] {
                    let wg = gemv_workgroup(bpc);
                    let (cols, rows) = gemv_tile(m, n, bpc * 32, 4, 2, wg);
                    assert!(
                        (1..=GEMV_MAX_COLS).contains(&cols),
                        "m={m} n={n} cols={cols}"
                    );
                    assert!(
                        (1..=GEMV_MAX_ROWS).contains(&rows),
                        "m={m} n={n} rows={rows}"
                    );
                    assert!(cols.is_power_of_two() && rows.is_power_of_two());
                    assert!(
                        cols * rows <= GEMV_MAX_TILE,
                        "m={m} n={n}: {cols} x {rows} exceeds the accumulator budget"
                    );
                    assert!(
                        wg * cols <= GEMV_RED_WORDS,
                        "m={m} n={n}: {wg} x {cols} overruns `red`"
                    );
                    assert!(
                        cols == 1 || n / u64::from(cols) >= GEMV_MIN_WORKGROUPS,
                        "m={m} n={n}: tiling to {cols} left too few workgroups"
                    );
                    assert!(
                        n % u64::from(cols) == 0 || cols == 1,
                        "m={m} n={n}: a tail column tile was selected"
                    );
                    // Never worse than the geometry it replaces: the seed has to be beaten
                    // strictly, so a shape with no legal row tile keeps the decode one.
                    let chosen = gemv_named_bytes(m, n, bpc * 32, 4, 2, cols, rows);
                    let seed = gemv_named_bytes(m, n, bpc * 32, 4, 2, gemv_cols(n, wg), 1);
                    assert!(chosen <= seed, "m={m} n={n}: the picker chose a worse tile");
                }
            }
        }
    }

    /// The byte model is the quantity the SPIR-V walk measures, so it must agree with the
    /// structural claim the issue is about: weight amplification is `ceil(M / rows)`.
    #[test]
    fn the_byte_model_reproduces_the_structural_amplification() {
        let (n, k, bits) = (256u64, 3072u64, 4u32);
        let weight_once = u128::from(n) * u128::from(k) * u128::from(bits) / 8;
        for rows in [1u32, 2, 4] {
            for m in [1u64, 2, 3, 4, 7, 64] {
                // Activations priced at zero isolates the weight term.
                let bytes = gemv_named_bytes(m, n, k, bits, 0, 16, rows);
                assert_eq!(
                    bytes,
                    u128::from(m.div_ceil(u64::from(rows))) * weight_once,
                    "m={m} rows={rows}"
                );
            }
        }
    }

    /// The dispatch may never ask for more workgroups in y than Vulkan guarantees, and the value
    /// it clamps to is the guaranteed floor rather than anything a device reported.
    #[test]
    fn the_y_extent_is_clamped_to_the_guaranteed_floor() {
        assert_eq!(GEMV_MAX_GROUPS_Y, 65_535);
        let wg = gemv_workgroup(96);
        for m in [1u64, 65_535, 65_536, 262_140, 1_000_000] {
            let (_, rows) = gemv_tile(m, 3072, 3072, 4, 2, wg);
            let row_tiles = m.div_ceil(u64::from(rows));
            let groups_y = row_tiles.min(u64::from(GEMV_MAX_GROUPS_Y));
            assert!(groups_y <= u64::from(GEMV_MAX_GROUPS_Y), "m={m}");
            assert!(groups_y >= 1, "m={m}");
            // The grid-stride loop has to be able to reach every tile from the clamped grid.
            assert!(
                row_tiles <= groups_y * row_tiles.div_ceil(groups_y),
                "m={m}: the y-grid-stride loop cannot cover {row_tiles} tiles from {groups_y}"
            );
        }
    }

    /// The fallback knob has to actually reach the geometry it promises, and has to be safe when
    /// an operator writes something silly into it. The environment read itself is exercised by
    /// `rust/tests/row_tile_fallback.rs`; `src/ops/` may not contain `unsafe`, and mutating the
    /// process environment requires it.
    #[test]
    fn the_row_tile_can_be_pinned_back_to_the_decode_geometry() {
        let wg = gemv_workgroup(96);
        let untiled = gemv_tile_with(1, 3072, 3072, 4, 2, wg, GEMV_MAX_ROWS);
        let tiled = gemv_tile_with(8, 3072, 3072, 4, 2, wg, GEMV_MAX_ROWS);
        assert!(
            tiled.1 > 1,
            "the shape under test must tile by default, got {tiled:?}"
        );

        assert_eq!(
            gemv_tile_with(8, 3072, 3072, 4, 2, wg, 1),
            untiled,
            "pinning to 1 must reproduce the pre-issue-#7 geometry exactly"
        );

        // Whatever the ceiling, the selected tile is legal. A knob cannot be used to ask for a
        // tile the shader would refuse.
        for ceiling in 1..=GEMV_MAX_ROWS {
            let (cols, rows) = gemv_tile_with(8, 3072, 3072, 4, 2, wg, ceiling);
            assert!(
                rows <= ceiling && cols * rows <= GEMV_MAX_TILE,
                "{ceiling}: {cols}x{rows}"
            );
        }
    }

    /// The knob's parsing, stated as cases rather than described in a doc comment.
    #[test]
    fn the_row_tile_ceiling_clamps_rather_than_trusts() {
        assert_eq!(
            clamp_max_rows(None),
            GEMV_MAX_ROWS,
            "absent means the default"
        );
        assert_eq!(
            clamp_max_rows(Some("1")),
            1,
            "1 is the documented fallback and must reach it"
        );
        assert_eq!(clamp_max_rows(Some("2")), 2);
        assert_eq!(
            clamp_max_rows(Some("  4 ")),
            4,
            "surrounding whitespace is not a typo"
        );
        assert_eq!(
            clamp_max_rows(Some("0")),
            1,
            "0 rows is not a tile; clamp up, never to zero"
        );
        assert_eq!(
            clamp_max_rows(Some("64")),
            GEMV_MAX_ROWS,
            "above the bound is clamped down"
        );
        // A typo in a performance knob must not take the process down, and must not be read as
        // some accidental number either.
        for junk in ["", "  ", "yes", "-1", "2.5", "0x4"] {
            assert_eq!(
                clamp_max_rows(Some(junk)),
                GEMV_MAX_ROWS,
                "junk {junk:?} means default"
            );
        }
    }

    // ──────────────────────────────────────────────────────────────────────────────────────
    // `ONNXRUNTIME_EP_VULKAN_GEMV_TILE` — strict parsing, exact-bound legality, and the
    // three-way `gemv_tile_choice_with` combinator. Opposite polarity from `GEMV_MAX_ROWS`
    // above: every one of these refuses rather than falling back to a default.
    // ──────────────────────────────────────────────────────────────────────────────────────

    /// Every accepted string is exactly `[1-9][0-9]*,[1-9][0-9]*` or `0,<same>` / `<same>,0` (the
    /// latter two are legal *grammar*, refused later as [`GemvTileRefusal::Zero`] — a distinct
    /// concern from whether the field parses at all).
    #[test]
    fn the_parser_accepts_exactly_the_strict_grammar() {
        assert_eq!(parse_gemv_tile(std::ffi::OsStr::new("16,2")), Ok((16, 2)));
        assert_eq!(parse_gemv_tile(std::ffi::OsStr::new("8,4")), Ok((8, 4)));
        assert_eq!(parse_gemv_tile(std::ffi::OsStr::new("1,1")), Ok((1, 1)));
        // No leading zero except the bare digit `0` itself.
        assert_eq!(parse_gemv_tile(std::ffi::OsStr::new("10,20")), Ok((10, 20)));
    }

    #[test]
    fn the_parser_refuses_every_malformed_shape_before_dispatch() {
        // Field count: missing comma, extra commas, empty string. `","` itself splits into two
        // (empty) fields, so it is a `MalformedField` case below, not a field-count case here.
        for (s, expect_fields) in [("", 1), ("16", 1), ("16,2,4", 3), ("16,2,", 3)] {
            assert_eq!(
                parse_gemv_tile(std::ffi::OsStr::new(s)),
                Err(GemvTileRefusal::WrongFieldCount(expect_fields)),
                "input {s:?}"
            );
        }
        // Malformed fields: sign, leading zero, whitespace, radix prefix, non-digit, empty field
        // (including both fields empty, `","`).
        for s in [
            "-16,2", "+16,2", "016,2", "16,02", " 16,2", "16, 2", "0x10,2", "16,two", ",2", "16,",
            ",",
        ] {
            assert_eq!(
                parse_gemv_tile(std::ffi::OsStr::new(s)),
                Err(GemvTileRefusal::MalformedField),
                "input {s:?}"
            );
        }
        // Overflow: all-digit but does not fit u32.
        assert_eq!(
            parse_gemv_tile(std::ffi::OsStr::new("4294967296,2")),
            Err(GemvTileRefusal::Overflow)
        );
        assert_eq!(
            parse_gemv_tile(std::ffi::OsStr::new("16,99999999999999999999")),
            Err(GemvTileRefusal::Overflow)
        );
        // Zero: lexically valid, semantically illegal.
        assert_eq!(
            parse_gemv_tile(std::ffi::OsStr::new("0,2")),
            Err(GemvTileRefusal::Zero)
        );
        assert_eq!(
            parse_gemv_tile(std::ffi::OsStr::new("16,0")),
            Err(GemvTileRefusal::Zero)
        );
        assert_eq!(
            parse_gemv_tile(std::ffi::OsStr::new("0,0")),
            Err(GemvTileRefusal::Zero)
        );
    }

    /// Non-UTF-8 is refused as its own reason, never folded into "unset" or lossily substituted.
    /// Constructed platform-natively (`OsString` from an unpaired UTF-16 surrogate on Windows)
    /// rather than assumed, so this proves the real `OsStr::to_str` boundary, not a stand-in.
    #[test]
    fn a_non_utf8_value_refuses_rather_than_being_treated_as_unset() {
        #[cfg(windows)]
        {
            use std::os::windows::ffi::OsStringExt;
            // 0xD800 is a lone high surrogate: not representable as UTF-8, valid as a Windows
            // environment-variable byte sequence.
            let raw = std::ffi::OsString::from_wide(&[0xD800]);
            assert!(
                raw.to_str().is_none(),
                "test fixture must actually be invalid UTF-8"
            );
            assert_eq!(parse_gemv_tile(&raw), Err(GemvTileRefusal::NonUtf8));
        }
        #[cfg(not(windows))]
        {
            use std::os::unix::ffi::OsStrExt;
            let raw = std::ffi::OsStr::from_bytes(&[0xFF, 0xFE, b',', b'2']);
            assert_eq!(parse_gemv_tile(raw), Err(GemvTileRefusal::NonUtf8));
        }
    }

    /// Legality checks exactly the bounds the compiled shader and dispatch enforce — the
    /// **compiled** `GEMV_MAX_ROWS`, not `gemv_max_rows()`'s environment-adjusted ceiling, and no
    /// requirement that `cols | n` or `rows | m` (both are real, correct, always-available tail-
    /// tile code paths in `q_gemv.comp`, not shapes the shader cannot run).
    #[test]
    fn legality_matches_every_compiled_bound_and_nothing_else() {
        let wg = gemv_workgroup(96); // K=3072, block 32 -> 96 blocks/col
        let n = 3072u64;

        // Exactly at each ceiling *alone*: legal.
        assert!(gemv_tile_legality(GEMV_MAX_COLS, 1, wg, n).is_ok());
        assert!(gemv_tile_legality(1, GEMV_MAX_ROWS, wg, n).is_ok());
        // The two per-field ceilings can never be requested *together*: `GEMV_MAX_COLS *
        // GEMV_MAX_ROWS` (16*4=64) exceeds `GEMV_MAX_TILE` (32) regardless of `wg` — this is the
        // accumulator budget, not a mistake in either individual bound.
        assert_eq!(
            gemv_tile_legality(GEMV_MAX_COLS, GEMV_MAX_ROWS, wg, n),
            Err(GemvTileRefusal::IllegalTile { cols: 16, rows: 4 }),
            "16*4=64 > GEMV_MAX_TILE=32, independent of wg"
        );
        // One past each ceiling: illegal, and the *specific* reason names the field that failed.
        assert_eq!(
            gemv_tile_legality(GEMV_MAX_COLS + 1, 1, wg, n),
            Err(GemvTileRefusal::IllegalCols {
                cols: GEMV_MAX_COLS + 1
            })
        );
        assert_eq!(
            gemv_tile_legality(1, GEMV_MAX_ROWS + 1, wg, n),
            Err(GemvTileRefusal::IllegalRows {
                cols: 1,
                rows: GEMV_MAX_ROWS + 1
            })
        );
        // wg*cols over the shared-reduction budget for a large workgroup.
        assert_eq!(
            gemv_tile_legality(16, 1, 256, n),
            Err(GemvTileRefusal::IllegalReduction {
                cols: 16,
                rows: 1,
                wg: 256
            }),
            "256*16=4096 > GEMV_RED_WORDS=2048"
        );
        // groupCountX over the guaranteed floor: only reachable with a very wide N and a narrow
        // `cols`, exactly the shape the unset search never produces on its own.
        let huge_n = u64::from(GEMV_MAX_GROUPS_X) + 1;
        assert_eq!(
            gemv_tile_legality(1, 1, wg, huge_n),
            Err(GemvTileRefusal::IllegalGroupsX { cols: 1, n: huge_n })
        );
        assert!(
            gemv_tile_legality(2, 1, wg, huge_n).is_ok(),
            "doubling cols must have halved groupCountX back under the floor"
        );

        // `cols`/`rows` not dividing `n`/`m` is legal — the tail-tile path is real.
        assert!(gemv_tile_legality(5, 3, wg, 3072).is_ok());
    }

    /// `m <= 1` is the decode null control: an unset variable, an unset-but-legal value's numeric
    /// effect, and a legal explicit override at decode all coincide on the *number*, but only the
    /// `Selected`/`Overridden` split at `m > 1` is affected by the environment; the `Decode`
    /// variant marks that decode was never actually consulting it.
    #[test]
    fn decode_ignores_a_legal_override_but_still_validates_it() {
        let wg = gemv_workgroup(96);
        let (n, k, bits, a_bytes) = (3072u64, 3072u64, 4u32, 2u64);
        let expect_decode = (gemv_cols(n, wg), 1u32);

        // Unset at m=1: decode geometry, tagged `Decode`.
        assert_eq!(
            gemv_tile_choice_with(1, n, k, bits, a_bytes, wg, None),
            Ok((expect_decode, GemvTileSource::Decode))
        );
        // A legal override at m=1: parsed and validated, but its *value* is set aside — decode
        // gets its usual geometry, not `(8, 4)`.
        let legal = std::ffi::OsStr::new("8,4");
        assert_eq!(
            gemv_tile_choice_with(1, n, k, bits, a_bytes, wg, Some(legal)),
            Ok((expect_decode, GemvTileSource::Decode)),
            "a legal override must not move decode off its null-control geometry"
        );
        // An *illegal* override still refuses at m=1 — validation is unconditional, only the
        // application of a legal value is conditional on m>1.
        let illegal = std::ffi::OsStr::new("999,2");
        assert_eq!(
            gemv_tile_choice_with(1, n, k, bits, a_bytes, wg, Some(illegal)),
            Err(GemvTileRefusal::IllegalCols { cols: 999 }),
            "decode must not silently swallow an illegal request either"
        );
    }

    /// Unset at `m > 1` is exactly [`gemv_tile`] — no duplicated default logic.
    #[test]
    fn unset_at_prefill_is_byte_identical_to_the_existing_selector() {
        let wg = gemv_workgroup(96);
        let (n, k, bits, a_bytes) = (3072u64, 3072u64, 4u32, 2u64);
        for m in [2u64, 3, 4, 5, 8, 32, 128] {
            assert_eq!(
                gemv_tile_choice_with(m, n, k, bits, a_bytes, wg, None),
                Ok((
                    gemv_tile(m, n, k, bits, a_bytes, wg),
                    GemvTileSource::Selected
                )),
                "m={m}"
            );
        }
    }

    /// A legal override at `m > 1` is substituted whole, checked against the **compiled**
    /// `GEMV_MAX_ROWS` — an explicit request is meant to win over
    /// `ONNXRUNTIME_EP_VULKAN_GEMV_MAX_ROWS`'s ceiling, not be refused by it. This function never
    /// reads that other variable at all (see its absence from every parameter/body here); the
    /// end-to-end proof that this reaches all the way through the real environment and the real
    /// `matmul_nbits_gemv` seam lives in `rust/tests/gemv_tile_seam.rs`.
    #[test]
    fn an_explicit_legal_override_is_substituted_at_the_compiled_ceiling() {
        let wg = gemv_workgroup(96);
        let (n, k, bits, a_bytes) = (3072u64, 3072u64, 4u32, 2u64);
        let over = std::ffi::OsStr::new("8,4"); // rows=4 == the compiled GEMV_MAX_ROWS
        assert_eq!(
            gemv_tile_choice_with(8, n, k, bits, a_bytes, wg, Some(over)),
            Ok(((8, 4), GemvTileSource::Overridden))
        );
        // `(16, 2)` is what the unset search would have picked for this exact shape; the override
        // producing something different is the point being proven, not an accident of the shape.
        assert_eq!(
            gemv_tile_choice_with(8, n, k, bits, a_bytes, wg, None).map(|(t, _)| t),
            Ok((16, 2))
        );
    }

    /// An illegal override at `m > 1` refuses with the exact reason the legality function raised
    /// — no substitution, no silent fallback to the unset search.
    #[test]
    fn an_illegal_override_at_prefill_refuses_rather_than_falling_back() {
        let wg = gemv_workgroup(96);
        let (n, k, bits, a_bytes) = (3072u64, 3072u64, 4u32, 2u64);
        let over = std::ffi::OsStr::new("16,8"); // rows=8 > compiled GEMV_MAX_ROWS=4
        assert_eq!(
            gemv_tile_choice_with(8, n, k, bits, a_bytes, wg, Some(over)),
            Err(GemvTileRefusal::IllegalRows { cols: 16, rows: 8 })
        );
    }

    // ──────────────────────────────────────────────────────────────────────────────────────
    // The equal-traffic mathematics (issue #81's B3): weight bytes depend only on `rows` (via
    // `ceil(M/rows)`, when `cols | n`); `(16, 2)` and `(8, 4)` tie on weight traffic only at the
    // two smallest `M` where both `ceil(M/2)` and `ceil(M/4)` are trivially `1` (`M ∈ {1, 2}`),
    // and for every `M >= 3` `(16, 2)` names strictly *more* weight bytes than `(8, 4)` — but the
    // ratio is `ceil(M/2)/ceil(M/4)`, which is **not** a constant 2x (it is 1.5x at M=5, 6, 2x at
    // M=3, 4, 7, 8, ...). They are equal on *total named bytes* only under `bits == 2 * a_bytes`
    // and `M ≡ 0 or 3 (mod 4)`.
    // ──────────────────────────────────────────────────────────────────────────────────────

    /// Weight bytes are a function of `rows` alone (for `cols | n`): `(16, 2)` makes `ceil(M/2)`
    /// weight passes and `(8, 4)` makes `ceil(M/4)`. These trivially agree at `M ∈ {1, 2}` (both
    /// `ceil(*, 2)` == `ceil(*, 4)` == 1 there) and — exhaustively verified below, not asserted —
    /// never agree again for any `M` in `3..=200`, with a ratio that is *not* a fixed 2x. This is
    /// the exact claim PR #92 got backwards ("both make `ceil(M/2)` passes") and Niobe's B3
    /// rejected: weight traffic is never governed by `cols` at all, and the two arms' weight
    /// bytes are equal only in the degenerate `M <= 2` case, never as a general rule.
    #[test]
    fn weight_traffic_depends_on_rows_alone_and_ties_only_at_m_le_2() {
        let (n, k, bits) = (4096u64, 4096u64, 4u32);
        let weight_only =
            |cols: u32, rows: u32, m: u64| gemv_named_bytes(m, n, k, bits, 0, cols, rows);
        for m in 1u64..=200 {
            let w_16_2 = weight_only(16, 2, m);
            let w_8_4 = weight_only(8, 4, m);
            let expect_16_2 =
                u128::from(m.div_ceil(2)) * u128::from(n) * u128::from(k) * u128::from(bits) / 8;
            let expect_8_4 =
                u128::from(m.div_ceil(4)) * u128::from(n) * u128::from(k) * u128::from(bits) / 8;
            assert_eq!(w_16_2, expect_16_2, "m={m}");
            assert_eq!(w_8_4, expect_8_4, "m={m}");
            if m <= 2 {
                assert_eq!(
                    w_16_2, w_8_4,
                    "m={m}: both ceil(m/2) and ceil(m/4) are 1 at this M — a degenerate tie, not \
                     a general equal-weight-traffic rule"
                );
            } else {
                assert!(
                    w_16_2 > w_8_4,
                    "m={m}: (16,2) must name strictly more weight bytes than (8,4) once M>2"
                );
            }
        }
        // The ratio is not a constant 2x: it tracks ceil(m/2)/ceil(m/4) exactly, which takes at
        // least two distinct values across this range.
        let ratio = |m: u64| weight_only(16, 2, m) as f64 / weight_only(8, 4, m) as f64;
        assert!(
            (ratio(3) - 2.0).abs() < 1e-9,
            "m=3: ceil(3/2)=2, ceil(3/4)=1 -> 2.0x"
        );
        assert!(
            (ratio(5) - 1.5).abs() < 1e-9,
            "m=5: ceil(5/2)=3, ceil(5/4)=2 -> 1.5x"
        );
    }

    /// Independently derived equality condition for **total named bytes** (weight + activation)
    /// between `(16, 2)` and `(8, 4)`, for `bits=4, a_bytes=2` (int4 weights, fp16 activations —
    /// the Phi-3.5 shape this issue is about): equality holds iff `M ≡ 0 or 3 (mod 4)`.
    ///
    /// Proof sketch (`bits == 2 * a_bytes` is the load-bearing sub-condition):
    /// `total(cols,rows) = ceil(M/rows)*N*K*bits/8 + ceil(M/rows)*(N/cols)*rows*K*a_bytes`. With
    /// `cols*a_bytes` held so that `(16,2)` and `(8,4)` both reduce their activation term to
    /// `ceil(M/rows) * rows * (K*a_bytes) * (N/cols)`, and `bits=4=2*a_bytes` makes the weight
    /// term at rows=2 equal to `2 * ceil(M/2) * (base)`. Writing `M = 4q + r`, `ceil(M/2)` and
    /// `2*ceil(M/4)` agree exactly at `r ∈ {0, 3}` and differ at `r ∈ {1, 2}` — verified
    /// exhaustively below rather than asserted.
    #[test]
    fn total_named_bytes_ties_between_16_2_and_8_4_iff_m_is_0_or_3_mod_4() {
        // Phi-3.5's own shape: bits=4 (int4), a_bytes=2 (fp16). `bits == 2*a_bytes` is exactly the
        // sub-condition the proof needs; changing either constant below must be expected to move
        // the equality residues, which is exactly why this is computed, not hard-coded.
        let (n, k, bits, a_bytes) = (4096u64, 4096u64, 4u32, 2u64);
        assert_eq!(
            bits,
            2 * u32::try_from(a_bytes).unwrap(),
            "the load-bearing sub-condition"
        );

        let total =
            |cols: u32, rows: u32, m: u64| gemv_named_bytes(m, n, k, bits, a_bytes, cols, rows);
        let mut ties = Vec::new();
        let mut differs = Vec::new();
        for m in 1u64..=200 {
            let a = total(16, 2, m);
            let b = total(8, 4, m);
            if a == b {
                ties.push(m);
            } else {
                differs.push((m, a, b));
            }
            let residue_predicts_tie = m % 4 == 0 || m % 4 == 3;
            assert_eq!(
                a == b,
                residue_predicts_tie,
                "m={m} (m mod 4 = {}): total bytes {a} vs {b}, predicted tie={residue_predicts_tie}",
                m % 4
            );
        }
        assert!(
            !ties.is_empty() && !differs.is_empty(),
            "both regimes must be witnessed"
        );

        // The exact witnessed values from issue #81's own probe range, quoted precisely rather
        // than approximated: M=2 ties at 2.00x, M=4 exact tie, M=5 at 1.33x (4/3).
        let ratio = |m: u64| total(8, 4, m) as f64 / total(16, 2, m) as f64;
        assert!(
            (ratio(2) - 2.0).abs() < 1e-9,
            "M=2 ratio must be exactly 2.00"
        );
        assert_eq!(total(16, 2, 4), total(8, 4, 4), "M=4 is an exact tie");
        assert!(
            (ratio(5) - 4.0 / 3.0).abs() < 1e-9,
            "M=5 ratio must be exactly 4/3 (1.33..)"
        );
        // M=128 (≡0 mod 4, Phi-3.5's own prefill width) ties — the only M an equal-traffic A/B
        // may use; it does not generalise to M2 or M5, which is the exact overgeneralisation
        // B3 rejected.
        assert_eq!(
            total(16, 2, 128),
            total(8, 4, 128),
            "M=128 ties (128 mod 4 == 0)"
        );
    }

    /// Planted-mutant guard: if the weight term's divisor were changed to depend on `cols`
    /// (the exact wrong shape of PR #92's claim — "both make `ceil(M/2)` passes"), this fails.
    /// Left as an explicit computed assertion, not a hard-coded number, so a future edit to
    /// [`gemv_named_bytes`] that reintroduces that coupling is caught here rather than only in a
    /// manually-planted mutation run.
    #[test]
    fn planted_wrong_claim_both_make_ceil_m_over_2_weight_passes_is_false() {
        let (n, k, bits) = (4096u64, 4096u64, 4u32);
        let m = 5u64; // the sharpest witnessed counterexample: ratio 1.33x, not 1.0x
        let weight_16_2 = gemv_named_bytes(m, n, k, bits, 0, 16, 2);
        let weight_8_4 = gemv_named_bytes(m, n, k, bits, 0, 8, 4);
        let wrong_claim_both_ceil_m_over_2 =
            u128::from(m.div_ceil(2)) * u128::from(n) * u128::from(k) * u128::from(bits) / 8;
        assert_eq!(
            weight_16_2, wrong_claim_both_ceil_m_over_2,
            "(16,2) really does make ceil(M/2) weight passes"
        );
        assert_ne!(
            weight_8_4, wrong_claim_both_ceil_m_over_2,
            "(8,4) at M=5 makes ceil(5/4)=2 weight passes, not ceil(5/2)=3 — the false claim \
             PR #92 made and Niobe's B3 rejected must not hold here"
        );
    }

    #[test]
    fn prepack_is_a_documented_pass_through() {
        use crate::engine::{PackInput, TileConfig};
        let cfg = TileConfig {
            tile_n: 1,
            tile_m: 1,
            block_size: 32,
        };
        let w = [1u8, 2, 3, 4];
        let s = [5u8, 6];
        let out = prepack_matmul_nbits(PackInput {
            weight: &w,
            scales: &s,
            zero_points: None,
            config: &cfg,
        });
        assert_eq!(out.packed_weight, w);
        assert_eq!(out.packed_scales, s);
        assert_eq!(out.packed_zero_points, None);
    }

    /// The implied zero point is the value the CPU EP actually uses, read off it rather than off
    /// the schema prose (§8.1.1): with `zero_points` absent, a weight nibble of `i` dequantises to
    /// `i - 8`, so the implied point is 8 in both nibbles.
    #[test]
    fn implied_zero_points_match_the_oracle() {
        assert_eq!(IMPLIED_ZP_4BIT, 0x88);
        assert_eq!(IMPLIED_ZP_8BIT, 0x80);
        // 4-bit packs two blocks per byte, and each column's run is padded to a whole byte.
        assert_eq!(implied_zero_points(4, 96, 2).len(), 48 * 2);
        assert_eq!(implied_zero_points(4, 3, 2).len(), 2 * 2);
        assert_eq!(implied_zero_points(8, 96, 2).len(), 96 * 2);
        assert!(implied_zero_points(4, 4, 1).iter().all(|&b| b == 0x88));
    }

    // ──────────────────────────────────────────────────────────────────────────────────────
    // P6 — no dequantised weight is ever materialised in device memory
    // ──────────────────────────────────────────────────────────────────────────────────────

    use crate::engine::DType;

    /// A `DispatchContext` that records what the handler asked the engine to allocate.
    ///
    /// The high-water number in `AllocStats` is the *dynamic* form of the P6 assertion and needs a
    /// live ORT session to mean anything. This is the *structural* form, and it is the stronger of
    /// the two for this property: a dequantised weight can only reach device memory if some
    /// handler asks for the memory, and there is exactly one call that does so — `alloc_temp`.
    /// Counting it proves the absence for every shape at once, where a high-water threshold only
    /// ever proves it for the shapes that were run.
    #[derive(Default)]
    struct AllocRecorder {
        next: u64,
        temp_bytes: Vec<u64>,
        output_bytes: Vec<u64>,
        dispatches: Vec<crate::engine::KernelRequest>,
    }

    fn desc_bytes(d: &crate::engine::TensorDesc) -> u64 {
        let elems: i64 = d.shape.iter().product();
        let width = match d.dtype {
            DType::F16 => 2,
            DType::U8 | DType::Bool => 1,
            DType::I64 => 8,
            _ => 4,
        };
        (elems.max(0) as u64) * width
    }

    impl crate::engine::DispatchContext for AllocRecorder {
        fn resolve(
            &mut self,
            _r: &crate::engine::TensorRef,
        ) -> crate::engine::EpResult<crate::engine::BufferView> {
            self.next += 1;
            Ok(crate::engine::BufferView::from_raw(self.next))
        }
        fn bind_output(
            &mut self,
            _o: &crate::engine::OutRef,
            desc: crate::engine::TensorDesc,
        ) -> crate::engine::EpResult<crate::engine::BufferView> {
            self.output_bytes.push(desc_bytes(&desc));
            self.next += 1;
            Ok(crate::engine::BufferView::from_raw(self.next))
        }
        fn alloc_temp(
            &mut self,
            desc: crate::engine::TensorDesc,
        ) -> crate::engine::EpResult<crate::engine::BufferView> {
            self.temp_bytes.push(desc_bytes(&desc));
            self.next += 1;
            Ok(crate::engine::BufferView::from_raw(self.next))
        }
        fn dispatch(&mut self, k: crate::engine::KernelRequest) -> crate::engine::EpResult<()> {
            self.dispatches.push(k);
            Ok(())
        }
        fn read_const_i64(&self, _r: &crate::engine::TensorRef) -> Option<Vec<i64>> {
            None
        }
    }

    /// Build a `MatMulNBits` node in the exact form all 161 Phi-3.5 nodes take.
    fn phi35_shaped_node(k: i64, n: i64, m: i64) -> crate::engine::NodeDesc {
        use crate::engine::{AttrValue, NodeDesc, OutRef, TensorDesc, TensorRef};
        let blocks = k / 32;
        let input = |name: &str, dtype: DType, shape: Vec<i64>| TensorRef {
            name: name.to_string(),
            desc: Some(TensorDesc::new(dtype, shape)),
            is_initializer: name != "A",
        };
        NodeDesc {
            op_type: "MatMulNBits".to_string(),
            name: "phi35".to_string(),
            domain: "com.microsoft".to_string(),
            since_version: 1,
            attributes: [
                ("K".to_string(), AttrValue::Int(k)),
                ("N".to_string(), AttrValue::Int(n)),
                ("bits".to_string(), AttrValue::Int(4)),
                ("block_size".to_string(), AttrValue::Int(32)),
            ]
            .into_iter()
            .collect(),
            inputs: vec![
                input("A", DType::F16, vec![1, m, k]),
                input("B", DType::U8, vec![n, blocks, 16]),
                input("scales", DType::F16, vec![n * blocks]),
            ],
            outputs: vec![OutRef {
                name: "Y".to_string(),
                desc: Some(TensorDesc::new(DType::F16, vec![1, m, n])),
            }],
        }
    }

    /// **P6.** The GEMV allocates nothing, so no dequantised weight can exist in device memory.
    ///
    /// The number that makes this worth asserting: at `K=8192, N=3072` the packed weight is 12 MiB
    /// and its f32 expansion is 96 MiB. Per node, times 161 nodes. A backend that dequantises into
    /// a scratch buffer does not merely run slower — on a 4 GiB mobile device it does not run.
    ///
    /// Stated as *zero allocations* rather than as a byte threshold on purpose. A threshold has to
    /// be chosen, and any threshold generous enough not to be flaky is generous enough to hide a
    /// small scratch buffer. Zero is not a threshold.
    #[test]
    fn the_gemv_materialises_no_dequantised_weight() {
        let spec = row("MatMulNBits");
        for (k, n) in [(3072_i64, 8192_i64), (8192, 3072), (3072, 3072)] {
            let node = phi35_shaped_node(k, n, 1);
            let mut rec = AllocRecorder::default();
            (spec.translate)(spec, &node, &mut rec).expect("translate");

            assert!(
                rec.temp_bytes.is_empty(),
                "K={k} N={n}: the GEMV asked for {} scratch buffer(s) totalling {} bytes; \
                 dequantisation must happen in registers, never through device memory",
                rec.temp_bytes.len(),
                rec.temp_bytes.iter().sum::<u64>(),
            );
            assert_eq!(rec.dispatches.len(), 1, "one dispatch per node");
            assert_eq!(
                rec.output_bytes,
                vec![(n as u64) * 2],
                "K={k} N={n}: the only bound output must be the activation-sized result"
            );
        }
    }

    /// The bytes this handler causes to be written do not grow with `K`.
    ///
    /// This is the same property from the other side, and it is the one a reader can check without
    /// knowing what `alloc_temp` is: `K` is the reduction extent, so anything proportional to it
    /// is an expanded weight. Quadrupling `K` while holding `N` must change nothing at all.
    #[test]
    fn gemv_allocation_is_independent_of_the_reduction_extent() {
        let spec = row("MatMulNBits");
        let mut sizes = Vec::new();
        for k in [1024_i64, 4096] {
            let mut rec = AllocRecorder::default();
            let node = phi35_shaped_node(k, 2048, 1);
            (spec.translate)(spec, &node, &mut rec).expect("translate");
            sizes.push((rec.output_bytes.clone(), rec.temp_bytes.clone()));
        }
        assert_eq!(
            sizes[0], sizes[1],
            "allocation grew with K, which is what an expanded weight looks like"
        );
    }
}
