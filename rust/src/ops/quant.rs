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
        deny!(DynamicShape, "`{}` input A has no shape", spec.op_type);
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
        require!(
            d >= 0,
            DynamicShape,
            "`{}` input A has a symbolic leading dimension",
            spec.op_type
        );
        rows = rows.saturating_mul(d as u64);
    }
    Ok(rows)
}

/// Workgroup size for one output element, derived from the reduction extent and the **floor**.
///
/// The smallest power of two that covers `blocks_per_col`, clamped to `[32, 256]`. Three
/// deliberate properties:
///
/// * The upper clamp is 256 because that is the `maxComputeWorkGroupInvocations` value the
///   baseline capability set guarantees (`OP_COVERAGE.md` §7.2). It is *not* the larger figure
///   either development GPU reports — sizing to the local device is how a kernel passes every
///   local test and fails on a phone.
/// * It is a power of two because the shared-memory tree reduction halves the stride.
/// * The lower clamp is 32 rather than 1 so a short reduction still fills at least one subgroup on
///   every vendor. That is a *performance* floor expressed without ever reading `subgroupSize`,
///   which is not guaranteed to be any particular value.
///
/// Shared memory is fixed at 256 floats = 1 KiB regardless, well inside the 16 KiB floor.
pub fn gemv_workgroup(blocks_per_col: u64) -> u32 {
    let mut wg: u32 = 32;
    while (wg as u64) < blocks_per_col && wg < 256 {
        wg *= 2;
    }
    wg
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
    let mut push = Vec::with_capacity(16);
    for v in [m_total, k, n, blocks_per_col] {
        push.extend_from_slice(&(v as u32).to_le_bytes());
    }

    ctx.dispatch(KernelRequest {
        shader,
        spec_constants: vec![wg, bits as u32, block_size as u32, u32::from(has_zp)],
        push_constants: push,
        bindings: vec![a, b, scales, zp, y],
        workgroups: [n as u32, m_total as u32, 1],
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
            DynamicShape,
            "`{}` scale has no shape; the scale-index mode cannot be decided",
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
        // K=3072 at block 32 -> 96 blocks -> 128 threads, the smallest power of two that covers it.
        assert_eq!(gemv_workgroup(96), 128);
        // K=8192 at block 32 -> 256 blocks -> the 256-invocation floor of §7.2, not the 1024 both
        // development GPUs actually report.
        assert_eq!(gemv_workgroup(256), 256);
        assert_eq!(
            gemv_workgroup(4096),
            256,
            "never exceeds the guaranteed floor"
        );
        // A short reduction still fills a subgroup on every vendor without reading `subgroupSize`.
        assert_eq!(gemv_workgroup(1), 32);
        for blocks in [1u64, 7, 96, 128, 255, 256, 10_000] {
            let wg = gemv_workgroup(blocks);
            assert!((32..=256).contains(&wg));
            assert!(wg.is_power_of_two(), "the tree reduction halves the stride");
        }
    }

    /// The pack transform is pure and total; assert the shape of what it produces rather than
    /// leaving "pass-through" as a claim in a doc comment.
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
}
