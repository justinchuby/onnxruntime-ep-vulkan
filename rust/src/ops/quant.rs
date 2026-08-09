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
// The tile override — a measurement instrument, deliberately not a tuning knob (issue #81)
// ──────────────────────────────────────────────────────────────────────────────────────────────

/// The exact-pair tile request. `"cols,rows"`.
///
/// # Why this exists at all, and why it is not `GEMV_MAX_ROWS` with a different name
///
/// [`gemv_tile_with`] requires a **strict** improvement, so when two candidates name the same
/// bytes the incumbent keeps the tile. `(16, 2)` and `(8, 4)` are exactly such a pair on the
/// Phi-3.5 projection shapes — see [`gemv_tile`]'s own tests for the condition, which is
/// arithmetic rather than universal — and `ONNXRUNTIME_EP_VULKAN_GEMV_MAX_ROWS` cannot separate
/// them: it lowers the row *ceiling*, and lowering the ceiling to 4 changes nothing while lowering
/// it to 2 forbids the arm under test. So the equal-traffic A/B that isolates register pressure
/// from bandwidth has no surface to run on. This is that surface, and nothing more: it selects a
/// tile the search could already have selected, it never widens what the shader may be asked for,
/// and unset it is not in the code path at all.
///
/// # The three states are three different facts
///
/// `Unset` is not `Exact { .. }` with default values and it is not `Unparseable`. Collapsing
/// "nobody asked" onto "somebody asked for the default" is how an instrument stops being able to
/// report that it was never armed; collapsing "somebody asked for nonsense" onto "nobody asked" is
/// how a typo becomes a silently different measurement. Both collapses are refused here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileRequest {
    /// The variable is absent. The byte-model search runs exactly as it did before this existed.
    Unset,
    /// A syntactically well-formed pair. Whether it is *legal* for a given shape is a separate
    /// question, answered by [`gemv_tile_legality`].
    Exact {
        /// Requested `QB_COLS`.
        cols: u32,
        /// Requested `QB_ROWS`.
        rows: u32,
    },
    /// The variable is present and is not a pair this parser accepts — including the case where
    /// the operating system holds bytes that are not UTF-8.
    Unparseable,
}

/// Why an override was refused. One variant per rule, so the diagnostic names the rule rather
/// than restating the numbers and leaving the reader to work out which bound was hit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileRefusalReason {
    /// The value is not `"<cols>,<rows>"` in canonical decimal.
    Unparseable,
    /// `cols` is not a power of two. The paired non-atomic store in `q_gemv.comp` needs evenness,
    /// and every tile the search can emit is a power of two.
    ColsNotPowerOfTwo,
    /// `rows` is not a power of two, for the same reason.
    RowsNotPowerOfTwo,
    /// `cols > GEMV_MAX_COLS` — the shader's column register budget.
    ColsAboveCap,
    /// `rows > GEMV_MAX_ROWS` — the activation window the tiled arm holds per row.
    RowsAboveCap,
    /// `rows` exceeds the ceiling `ONNXRUNTIME_EP_VULKAN_GEMV_MAX_ROWS` currently imposes. The
    /// exact request does not silently outrank the ceiling; it refuses so the operator can see
    /// that two controls disagree.
    RowsAboveCeilingInForce,
    /// `cols * rows > GEMV_MAX_TILE` — the accumulator array the shader sizes by `QB_MAX_TILE`.
    TileAboveAccumulatorBudget,
    /// `wg * cols > GEMV_RED_WORDS` — the shared reduction array.
    SharedArrayOverrun,
    /// `cols` does not divide `N`, so the dispatch would take the tail-tile path the search
    /// deliberately never selects.
    ColsDoesNotDivideN,
    /// Tiling this narrowly would leave fewer than [`GEMV_MIN_WORKGROUPS`] workgroups.
    TooFewWorkgroups,
    /// `rows > 1` at `M <= 1`. Production never row-tiles a one-row dispatch — the search short
    /// circuits — so this is a tile production could not already have executed.
    RowTileAtDecodeWidth,
}

impl TileRefusalReason {
    /// A stable machine-readable token. Recorded into the counters artifact, so it is part of the
    /// instrument's contract and is asserted by name rather than by message text.
    pub fn token(self) -> &'static str {
        match self {
            Self::Unparseable => "unparseable",
            Self::ColsNotPowerOfTwo => "cols_not_power_of_two",
            Self::RowsNotPowerOfTwo => "rows_not_power_of_two",
            Self::ColsAboveCap => "cols_above_cap",
            Self::RowsAboveCap => "rows_above_cap",
            Self::RowsAboveCeilingInForce => "rows_above_ceiling_in_force",
            Self::TileAboveAccumulatorBudget => "tile_above_accumulator_budget",
            Self::SharedArrayOverrun => "shared_array_overrun",
            Self::ColsDoesNotDivideN => "cols_does_not_divide_n",
            Self::TooFewWorkgroups => "too_few_workgroups",
            Self::RowTileAtDecodeWidth => "row_tile_at_decode_width",
        }
    }
}

/// A refusal, with the request that caused it. `cols`/`rows` are `0` when nothing parsed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileRefusal {
    /// Which rule refused.
    pub reason: TileRefusalReason,
    /// The requested column tile, or `0` when the value never parsed.
    pub cols: u32,
    /// The requested row tile, or `0` when the value never parsed.
    pub rows: u32,
}

impl TileRefusal {
    /// The line recorded into the counters artifact and into the returned `OrtStatus` message.
    ///
    /// Shape-bearing on purpose: the same pair is legal on one node of a graph and illegal on the
    /// next, so a refusal that named only the pair would be unactionable.
    pub fn detail(&self, m: u64, n: u64, wg: u32) -> String {
        format!(
            "{} cols={} rows={} m={m} n={n} wg={wg}",
            self.reason.token(),
            self.cols,
            self.rows
        )
    }
}

/// The environment variable that carries the exact-pair request.
pub const GEMV_TILE_ENV: &str = "ONNXRUNTIME_EP_VULKAN_GEMV_TILE";

/// One decimal component of the pair.
///
/// The accepted language is `[1-9][0-9]*`, and every deviation is a refusal rather than a
/// normalisation. `08` is rejected rather than read as 8 so that the value the diagnostic prints
/// is the value the operator typed; `+8`, `8 `, `2.5`, `0x8`, `-1` and `4294967296` are rejected
/// by the same rule and by `u32::from_str`'s overflow behaviour, in that order.
fn parse_tile_component(s: &str) -> Option<u32> {
    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    if s.len() > 1 && s.starts_with('0') {
        return None;
    }
    match s.parse::<u32>() {
        Ok(0) | Err(_) => None,
        Ok(n) => Some(n),
    }
}

/// Parse `"cols,rows"`. Pure, so every syntax case is a unit test rather than a paragraph.
///
/// `None` means the variable is absent. `Some(v)` where `v` is anything other than exactly two
/// canonical decimal components separated by one comma is [`TileRequest::Unparseable`] — including
/// `"8,4,2"`, which is caught because the text after the first comma is not all digits.
pub fn parse_tile_request(raw: Option<&str>) -> TileRequest {
    let Some(v) = raw else {
        return TileRequest::Unset;
    };
    let Some((c, r)) = v.split_once(',') else {
        return TileRequest::Unparseable;
    };
    match (parse_tile_component(c), parse_tile_component(r)) {
        (Some(cols), Some(rows)) => TileRequest::Exact { cols, rows },
        _ => TileRequest::Unparseable,
    }
}

/// Is this pair one the search could already have emitted for this shape?
///
/// Every rule below is a rule [`gemv_tile_with`] already enforces on its own candidates, restated
/// as a predicate so that an *asked-for* tile is held to the same standard as a *chosen* one.
/// Nothing here is new policy: an override cannot reach a geometry the selector could not, which
/// is what makes this an instrument rather than a way to ask the shader for an overrun.
///
/// Order matters. The caps are checked before the products so that `cols * rows` and `wg * cols`
/// cannot overflow on a hostile value; the parser has already excluded `0`.
pub fn gemv_tile_legality(
    m: u64,
    n: u64,
    wg: u32,
    cols: u32,
    rows: u32,
    max_rows: u32,
) -> Result<(), TileRefusalReason> {
    use TileRefusalReason as R;
    if !cols.is_power_of_two() {
        return Err(R::ColsNotPowerOfTwo);
    }
    if !rows.is_power_of_two() {
        return Err(R::RowsNotPowerOfTwo);
    }
    if cols > GEMV_MAX_COLS {
        return Err(R::ColsAboveCap);
    }
    if rows > GEMV_MAX_ROWS {
        return Err(R::RowsAboveCap);
    }
    if rows > max_rows {
        return Err(R::RowsAboveCeilingInForce);
    }
    if cols * rows > GEMV_MAX_TILE {
        return Err(R::TileAboveAccumulatorBudget);
    }
    if wg * cols > GEMV_RED_WORDS {
        return Err(R::SharedArrayOverrun);
    }
    if n % u64::from(cols) != 0 {
        return Err(R::ColsDoesNotDivideN);
    }
    if cols != 1 && n / u64::from(cols) < GEMV_MIN_WORKGROUPS {
        return Err(R::TooFewWorkgroups);
    }
    if rows > 1 && m <= 1 {
        return Err(R::RowTileAtDecodeWidth);
    }
    Ok(())
}

/// The tile this dispatch takes, or the refusal that stops it.
///
/// Pure in `request` and `max_rows`, so the whole three-state contract — including the one that
/// matters most, that `Unset` is *byte-for-byte* [`gemv_tile_with`] — is provable without touching
/// process state. `Unset` does not call the legality predicate at all: the identity is structural,
/// not a coincidence of the rules agreeing.
///
/// The signature is [`gemv_tile_with`]'s, plus `request`. Keeping the shape parameters positional
/// and identical is what makes the `Unset` delegation below readable as a pass-through rather than
/// a re-derivation, which is worth more here than being one argument under clippy's threshold.
#[allow(clippy::too_many_arguments)]
pub fn gemv_tile_requested(
    m: u64,
    n: u64,
    k: u64,
    bits: u32,
    a_bytes: u64,
    wg: u32,
    request: TileRequest,
    max_rows: u32,
) -> Result<(u32, u32), TileRefusal> {
    match request {
        TileRequest::Unset => Ok(gemv_tile_with(m, n, k, bits, a_bytes, wg, max_rows)),
        TileRequest::Unparseable => Err(TileRefusal {
            reason: TileRefusalReason::Unparseable,
            cols: 0,
            rows: 0,
        }),
        TileRequest::Exact { cols, rows } => {
            match gemv_tile_legality(m, n, wg, cols, rows, max_rows) {
                Ok(()) => Ok((cols, rows)),
                Err(reason) => Err(TileRefusal { reason, cols, rows }),
            }
        }
    }
}

/// The request this process carries.
///
/// `var_os` rather than `var`: a value the OS holds that is not UTF-8 must **refuse**, and
/// `std::env::var` reports that case as an `Err` indistinguishable from "absent" at the call site
/// unless it is matched on. Absent and un-decodable are different facts and are kept apart here.
///
/// Platform note, stated because a test that asserts it on the wrong platform is worse than no
/// test: on Windows an empty value removes the variable, so `GEMV_TILE=""` is `Unset` there and
/// `Unparseable` on Linux. The pure parser is where the empty string's meaning is pinned.
fn tile_request_from_env() -> TileRequest {
    match std::env::var_os(GEMV_TILE_ENV) {
        None => TileRequest::Unset,
        Some(v) => match v.to_str() {
            Some(s) => parse_tile_request(Some(s)),
            None => TileRequest::Unparseable,
        },
    }
}

/// [`gemv_tile_requested`] with both controls read from the environment.
///
/// This is the single production entry point, and the reason it is separate from [`gemv_tile`] is
/// that `gemv_tile` has three other callers (the probe mirrors and
/// `rust/tests/row_tile_fallback.rs`) whose contract is "the tile the byte model picks" and must
/// not acquire a failure mode.
fn gemv_tile_for_dispatch(
    m: u64,
    n: u64,
    k: u64,
    bits: u32,
    a_bytes: u64,
    wg: u32,
) -> Result<(u32, u32), TileRefusal> {
    gemv_tile_requested(
        m,
        n,
        k,
        bits,
        a_bytes,
        wg,
        tile_request_from_env(),
        gemv_max_rows(),
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
    let m_rows = m_total.max(0) as u64;
    // The override is consulted here, before any resource is committed to a geometry, and a
    // refusal returns rather than falling back to the search: an operator who asked for a specific
    // tile and got the default one would attribute the default arm's numbers to the arm they
    // asked for. The counter row is written first so the refusal is observable from the artifact
    // even when the caller only sees an `OrtStatus`.
    let (cols, rows) =
        match gemv_tile_for_dispatch(m_rows, n as u64, k as u64, bits as u32, a_bytes, wg) {
            Ok(pair) => pair,
            Err(refusal) => {
                let detail = refusal.detail(m_rows, n as u64, wg);
                crate::counters::record_gemv_tile_override_refused(&detail);
                return Err(EpError::Internal(format!(
                    "`{}` refused the {GEMV_TILE_ENV} override: {detail}",
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
    /// cols = 8` name the same bytes **for some `M`, not for all of them**. The exact condition is
    /// derived and exhausted in
    /// [`the_16x2_and_8x4_tie_is_conditional_on_m_and_on_the_dtype_ratio`]; `M = 64` below is one
    /// of the tying widths. The picker requires a strict improvement, so wherever the two do tie
    /// the tile goes to the first candidate, which is the one that keeps the decode column tile
    /// and runs half as many sequential shared-memory reductions. Where they do not tie, `(16, 2)`
    /// is the strictly cheaper of the pair at every `M` in the grid below.
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
        // The alternative ties **at this M**, and the wider tile that would break the tie is
        // excluded by the accumulator cap rather than by the model. `M = 64` is a tying width
        // because `64 mod 4 == 0`; the general rule is proved separately.
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

    /// When `(16, 2)` and `(8, 4)` name the same bytes — derived, then exhausted.
    ///
    /// [`gemv_named_bytes`] with `cols` dividing `N` reduces to
    ///
    /// ```text
    /// total(cols, rows) = ceil(M / rows) * (W + rows * A)
    ///     W = N * K * bits / 8          (independent of cols, since ceil(N/cols)*cols == N)
    ///     A = (N / cols) * K * a_bytes  (the activation bytes one row of the tile grid names)
    /// ```
    ///
    /// Halving `cols` doubles `A`, so `(8, 4)` has `A_8 = 2 * A_16` and
    ///
    /// ```text
    /// total(16, 2) = ceil(M/2) * (W + 2*A_16)
    /// total(8,  4) = ceil(M/4) * (W + 8*A_16)
    /// ```
    ///
    /// With `N = K = 3072`, `bits = 4`, `a_bytes = 2`: `W = 4_718_592` and `A_16 = 1_179_648`, so
    /// `W + 2*A_16 = 7_077_888` and `W + 8*A_16 = 14_155_776` — exactly twice. The bracket ratio
    /// is 2 precisely when `W == 2 * A_16 * 2`, i.e. when `bits == 2 * a_bytes`; with fp32
    /// activations the ratio is not 2 and the pair can never tie. Given the ratio, the tie reduces
    /// to `ceil(M/2) == 2 * ceil(M/4)`, which holds exactly for `M mod 4 ∈ {0, 3}`.
    ///
    /// Both halves are exhausted below rather than asserted at a handful of remembered points, and
    /// the four widths the design note quotes are **computed here**, not transcribed.
    #[test]
    fn the_16x2_and_8x4_tie_is_conditional_on_m_and_on_the_dtype_ratio() {
        let (n, k) = (3072u64, 3072u64);
        let wide = |m: u64| gemv_named_bytes(m, n, k, 4, 2, 16, 2);
        let deep = |m: u64| gemv_named_bytes(m, n, k, 4, 2, 8, 4);

        // The four widths the design note quotes, computed rather than quoted. `M = 2` is the one
        // that is *not* a tie, and its orientation is stated as a ratio of named totals so a
        // future edit cannot swap the labels without failing.
        assert_eq!(wide(2), 7_077_888, "M=2 (16,2)");
        assert_eq!(deep(2), 14_155_776, "M=2 (8,4)");
        assert_eq!(
            deep(2),
            2 * wide(2),
            "M=2: the deep tile names twice the bytes, not a tie"
        );
        assert_eq!(wide(4), 14_155_776, "M=4 (16,2)");
        assert_eq!(deep(4), 14_155_776, "M=4 (8,4)");
        assert_eq!(wide(5), 21_233_664, "M=5 (16,2)");
        assert_eq!(deep(5), 28_311_552, "M=5 (8,4)");
        assert_eq!(wide(128), 452_984_832, "M=128 (16,2)");
        assert_eq!(deep(128), 452_984_832, "M=128 (8,4)");

        // Exhaustive over a range that covers every residue class many times: the tie holds for
        // `M mod 4 ∈ {0, 3}` and fails everywhere else, with fp16 activations.
        for m in 2u64..=4096 {
            let tied = wide(m) == deep(m);
            let predicted = matches!(m % 4, 0 | 3);
            assert_eq!(tied, predicted, "M={m}: tie={tied} predicted={predicted}");
            if !tied {
                assert!(
                    wide(m) < deep(m),
                    "M={m}: the wide tile must never be the loser"
                );
            }
        }

        // The ratio condition, not just the `M` condition. `bits == 2 * a_bytes` is what makes any
        // tie possible; with fp32 activations the same pair never ties at any `M`.
        for m in 2u64..=512 {
            assert_ne!(
                gemv_named_bytes(m, n, k, 4, 4, 16, 2),
                gemv_named_bytes(m, n, k, 4, 4, 8, 4),
                "M={m}: bits != 2 * a_bytes, so the pair cannot tie"
            );
        }
        // And with `bits = 8, a_bytes = 4` the ratio is restored and so is the residue rule.
        for m in 2u64..=512 {
            assert_eq!(
                gemv_named_bytes(m, n, k, 8, 4, 16, 2) == gemv_named_bytes(m, n, k, 8, 4, 8, 4),
                matches!(m % 4, 0 | 3),
                "M={m}: bits == 2 * a_bytes restores the residue rule"
            );
        }
    }

    /// The override parser accepts one language and refuses everything else.
    #[test]
    fn the_tile_override_parser_accepts_only_a_canonical_decimal_pair() {
        assert_eq!(parse_tile_request(None), TileRequest::Unset);
        assert_eq!(
            parse_tile_request(Some("8,4")),
            TileRequest::Exact { cols: 8, rows: 4 }
        );
        assert_eq!(
            parse_tile_request(Some("16,2")),
            TileRequest::Exact { cols: 16, rows: 2 }
        );
        for bad in [
            "",             // present but empty — a request for nothing is not a request
            "8",            // no comma
            "8,",           // missing component
            ",4",           //
            "8,4,2",        // three components: the tail is not all digits
            " 8,4",         // whitespace is not normalised away
            "8, 4",         //
            "8,4 ",         //
            "08,4",         // leading zeros: the printed value must be the typed value
            "8,04",         //
            "+8,4",         //
            "-8,4",         //
            "8,-4",         //
            "2.5,4",        // a decimal point is not a separator
            "0x8,4",        //
            "0,4",          // zero is not a tile
            "8,0",          //
            "4294967296,4", // overflows u32
            "8,4294967296",
            "eight,four",
            "16;2",
        ] {
            assert_eq!(
                parse_tile_request(Some(bad)),
                TileRequest::Unparseable,
                "{bad:?} must not parse"
            );
        }
    }

    /// Unset is the default search, structurally — not merely at the shapes someone remembered.
    #[test]
    fn an_unset_override_is_byte_for_byte_the_default_search() {
        for m in [0u64, 1, 2, 3, 5, 17, 128, 4096] {
            for n in [1u64, 64, 130, 512, 3072, 9216, 32_064] {
                for bpc in [1u64, 7, 96, 256, 4096] {
                    for a_bytes in [2u64, 4] {
                        for ceiling in [1u32, 2, 4] {
                            let wg = gemv_workgroup(bpc);
                            let k = bpc * 32;
                            assert_eq!(
                                gemv_tile_requested(
                                    m,
                                    n,
                                    k,
                                    4,
                                    a_bytes,
                                    wg,
                                    TileRequest::Unset,
                                    ceiling
                                ),
                                Ok(gemv_tile_with(m, n, k, 4, a_bytes, wg, ceiling)),
                                "m={m} n={n} bpc={bpc} a={a_bytes} ceiling={ceiling}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Every legality rule refuses, and refuses by name.
    #[test]
    fn an_illegal_override_refuses_and_names_the_rule_that_refused_it() {
        use TileRefusalReason as R;
        let wg = gemv_workgroup(3072 / 32);
        assert_eq!(wg, 32);
        let ask = |m: u64, n: u64, cols: u32, rows: u32, ceiling: u32| {
            gemv_tile_requested(
                m,
                n,
                3072,
                4,
                2,
                wg,
                TileRequest::Exact { cols, rows },
                ceiling,
            )
            .map_err(|r| r.reason)
        };

        assert_eq!(
            ask(64, 3072, 8, 4, 4),
            Ok((8, 4)),
            "the pair under test is legal"
        );
        assert_eq!(ask(64, 3072, 16, 2, 4), Ok((16, 2)), "so is the incumbent");

        assert_eq!(ask(64, 3072, 3, 2, 4), Err(R::ColsNotPowerOfTwo));
        assert_eq!(ask(64, 3072, 8, 3, 4), Err(R::RowsNotPowerOfTwo));
        assert_eq!(ask(64, 3072, 32, 1, 4), Err(R::ColsAboveCap));
        assert_eq!(ask(64, 3072, 4, 8, 4), Err(R::RowsAboveCap));
        // The ceiling still binds: an exact request does not silently outrank it.
        assert_eq!(ask(64, 3072, 8, 4, 2), Err(R::RowsAboveCeilingInForce));
        assert_eq!(ask(64, 3072, 16, 4, 4), Err(R::TileAboveAccumulatorBudget));
        // `wg * cols` at the K = 8192 workgroup width.
        let wg8 = gemv_workgroup(8192 / 32);
        assert_eq!(
            gemv_tile_requested(
                64,
                3072,
                8192,
                4,
                2,
                wg8,
                TileRequest::Exact { cols: 16, rows: 2 },
                4
            ),
            Ok((16, 2)),
            "128 * 16 is exactly the shared-array ceiling, which is legal"
        );
        assert_eq!(
            gemv_tile_requested(
                64,
                3072,
                8192,
                4,
                2,
                256,
                TileRequest::Exact { cols: 16, rows: 2 },
                4
            )
            .map_err(|r| r.reason),
            Err(R::SharedArrayOverrun)
        );
        assert_eq!(ask(64, 130, 8, 2, 4), Err(R::ColsDoesNotDivideN));
        assert_eq!(ask(64, 64, 8, 2, 4), Err(R::TooFewWorkgroups));
        assert_eq!(ask(1, 3072, 16, 2, 4), Err(R::RowTileAtDecodeWidth));
        assert_eq!(ask(0, 3072, 16, 2, 4), Err(R::RowTileAtDecodeWidth));
        // A decode-width request for the decode tile is not a refusal.
        assert_eq!(ask(1, 3072, 16, 1, 4), Ok((16, 1)));

        // Unparseable refuses with no pair, because there was never a pair to report.
        let refusal = gemv_tile_requested(64, 3072, 3072, 4, 2, wg, TileRequest::Unparseable, 4)
            .expect_err("unparseable must refuse");
        assert_eq!(refusal.reason, R::Unparseable);
        assert_eq!((refusal.cols, refusal.rows), (0, 0));
    }

    /// The refusal detail carries the shape, and every token is distinct.
    #[test]
    fn every_refusal_token_is_distinct_and_the_detail_names_the_shape() {
        use TileRefusalReason as R;
        let all = [
            R::Unparseable,
            R::ColsNotPowerOfTwo,
            R::RowsNotPowerOfTwo,
            R::ColsAboveCap,
            R::RowsAboveCap,
            R::RowsAboveCeilingInForce,
            R::TileAboveAccumulatorBudget,
            R::SharedArrayOverrun,
            R::ColsDoesNotDivideN,
            R::TooFewWorkgroups,
            R::RowTileAtDecodeWidth,
        ];
        let mut tokens: Vec<&str> = all.iter().map(|r| r.token()).collect();
        tokens.sort_unstable();
        let distinct = {
            let mut t = tokens.clone();
            t.dedup();
            t.len()
        };
        assert_eq!(distinct, all.len(), "refusal tokens must be distinct");
        assert!(
            tokens.iter().all(|t| !t.is_empty()),
            "an empty token would be dropped by the counter"
        );
        let detail = TileRefusal {
            reason: R::TileAboveAccumulatorBudget,
            cols: 16,
            rows: 4,
        }
        .detail(64, 3072, 32);
        assert_eq!(
            detail,
            "tile_above_accumulator_budget cols=16 rows=4 m=64 n=3072 wg=32"
        );
    }

    /// The override can reach the equal-total candidate the search cannot.
    ///
    /// This is the whole justification for the seam existing, stated as an assertion: at a tying
    /// `M` the search returns `(16, 2)` because the improvement must be strict, and `(8, 4)` names
    /// the same bytes and is unreachable — including through the row-tile ceiling, which is the
    /// only other control that touches this selection.
    #[test]
    fn the_equal_total_candidate_is_unreachable_without_the_override() {
        let (n, k) = (3072u64, 3072u64);
        let wg = gemv_workgroup(k / 32);
        for m in [4u64, 8, 64, 128] {
            assert_eq!(
                gemv_named_bytes(m, n, k, 4, 2, 16, 2),
                gemv_named_bytes(m, n, k, 4, 2, 8, 4),
                "M={m} must be a tying width for this test to mean anything"
            );
            for ceiling in 1..=GEMV_MAX_ROWS {
                assert_ne!(
                    gemv_tile_with(m, n, k, 4, 2, wg, ceiling),
                    (8, 4),
                    "M={m} ceiling={ceiling}: no ceiling can select the equal-total candidate"
                );
            }
            assert_eq!(
                gemv_tile_requested(
                    m,
                    n,
                    k,
                    4,
                    2,
                    wg,
                    TileRequest::Exact { cols: 8, rows: 4 },
                    4
                ),
                Ok((8, 4)),
                "M={m}: the override is the only surface that reaches it"
            );
        }
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
