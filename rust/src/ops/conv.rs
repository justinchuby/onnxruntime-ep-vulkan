//! Convolution — `Conv`, the op that decides whether a CNN can run on this EP at all.
//!
//! # Why this module exists, measured rather than assumed
//!
//! `probe_model_op_census.py` against MobileNetV2-12 on 2026-08-03: **0 of 105 nodes claimed**,
//! and 52 of the 104 declines were `Conv` / `[not-registered]`. Two LLMs had been the entire
//! evidence base until then, and neither contains a single convolution, so no instrument the
//! project had could see it. The census is the argument for this file; a taxonomy is not.
//!
//! # What is claimed, and what is declined by name
//!
//! Claimed: 2-D `Conv` at f32, any `group` (so depthwise is the `group == C` case and costs no
//! second kernel), any strides, dilations and *explicit* begin/end pads, with or without bias.
//!
//! Declined, each for a stated reason rather than by omission:
//!
//! * **f16** — the packed-`uint` half I/O every other fp16 module in this crate uses addresses
//!   two elements per word. A convolution's inner loop reads scattered single elements, so the
//!   packed path would need an unpack per read and a different store rule for an odd row length.
//!   That is a second kernel, and claiming f16 through the f32 one would be the wrong-answer
//!   class the charter's first line names. Declines `[dtype]`.
//! * **rank != 4** — 1-D and 3-D convolution are different index arithmetic, not a wider loop.
//! * **`auto_pad != NOTSET`** — `SAME_UPPER`/`SAME_LOWER` derive the pads from the *output*
//!   shape, which for a symbolic spatial extent is not known when the pipeline is built. An
//!   explicit `pads` list is; ORT's own optimizers rewrite `auto_pad` to explicit pads for most
//!   producers, and MobileNetV2 carries explicit pads.
//! * **a symbolic *spatial* extent** — the output extent must be computed from the input extent
//!   and the attributes. Batch may be symbolic (it is, on every real vision graph); `H`/`W` may
//!   not, because the padding arithmetic is not linear in them. This is a decline with a lift
//!   condition, not a permanent one.

use crate::engine::{DType, DispatchContext, EpError, EpResult, KernelRequest, NodeDesc, TensorDesc};
use crate::kernel;
use crate::ops::common::claim::{self, ClaimResult};
use crate::ops::common::dtype::F32;
use crate::registry::OpStatus::Ready;
use crate::registry::{NodeView, OPSET_ANY, OpSpec};
use crate::require;

/// Workgroup size, matching every other 1-D grid in this crate. See `templates::EW_LOCAL_SIZE`.
pub(crate) const CONV_LOCAL_SIZE: u32 = 256;

/// Cap on dispatched workgroups; the shader is a grid-stride loop, so this bounds the grid
/// rather than the work.
pub(crate) const CONV_MAX_WORKGROUPS: u32 = 65_535;

/// The one spatial rank the kernel implements.
const CONV_SPATIAL_RANK: usize = 2;

/// A `Conv` node's spatial attributes, defaulted per the ONNX schema and validated once.
///
/// Built by both the claim predicate and the translate handler from the same function, because
/// the two disagreeing about what a default is is exactly the claim-then-fail shape that
/// `Cast`'s destination type produced on 2026-08-03.
#[derive(Debug)]
pub(crate) struct ConvAttrs {
    pub kernel_shape: [i64; CONV_SPATIAL_RANK],
    pub strides: [i64; CONV_SPATIAL_RANK],
    pub dilations: [i64; CONV_SPATIAL_RANK],
    /// `[h_begin, w_begin, h_end, w_end]`, in ONNX's begin-major order.
    pub pads: [i64; CONV_SPATIAL_RANK * 2],
    pub group: i64,
}

/// Read an int-list attribute of exactly `n` entries, or the supplied default when absent.
fn int_list<const N: usize>(
    attrs: &dyn Fn(&str) -> Option<Vec<i64>>,
    name: &str,
    default: [i64; N],
) -> Result<[i64; N], String> {
    match attrs(name) {
        None => Ok(default),
        Some(v) if v.len() == N => {
            let mut out = default;
            out.copy_from_slice(&v);
            Ok(out)
        }
        Some(v) => Err(format!(
            "`{name}` has {} entries; this kernel is 2-D and needs {N}",
            v.len()
        )),
    }
}

/// Derive the attribute set from a lookup closure. Shared by claim and translate.
pub(crate) fn conv_attrs(
    weight_spatial: [i64; CONV_SPATIAL_RANK],
    attrs: &dyn Fn(&str) -> Option<Vec<i64>>,
    group: Option<i64>,
) -> Result<ConvAttrs, String> {
    let kernel_shape = int_list(attrs, "kernel_shape", weight_spatial)?;
    if kernel_shape != weight_spatial {
        return Err(format!(
            "`kernel_shape` {kernel_shape:?} disagrees with the weight tensor's spatial extents \
             {weight_spatial:?}; the weights are the fact and an attribute that contradicts them \
             is a model this EP will not reinterpret"
        ));
    }
    let strides = int_list(attrs, "strides", [1, 1])?;
    let dilations = int_list(attrs, "dilations", [1, 1])?;
    let pads = int_list(attrs, "pads", [0, 0, 0, 0])?;
    if strides.iter().any(|&v| v < 1) || dilations.iter().any(|&v| v < 1) {
        return Err("`strides` and `dilations` must be >= 1".to_string());
    }
    if pads.iter().any(|&v| v < 0) {
        return Err("negative `pads` are not a convolution this kernel implements".to_string());
    }
    let group = group.unwrap_or(1);
    if group < 1 {
        return Err(format!("`group` is {group}; it must be >= 1"));
    }
    Ok(ConvAttrs {
        kernel_shape,
        strides,
        dilations,
        pads,
        group,
    })
}

/// Output spatial extent for one axis. The formula ONNX defines, in one place.
pub(crate) fn conv_out_extent(input: i64, pad_begin: i64, pad_end: i64, dil: i64, k: i64, stride: i64) -> i64 {
    (input + pad_begin + pad_end - ((k - 1) * dil + 1)) / stride + 1
}

/// `Conv` — claim 2-D f32 convolution with explicit pads.
fn conv(view: &NodeView<'_>, spec: &OpSpec) -> ClaimResult {
    let x = claim::input_edge(view, spec, 0)?;
    claim::check_dtype(spec, &x, "input 0 (X)")?;
    claim::check_shape(spec, &x, "input 0 (X)")?;
    let w = claim::input_edge(view, spec, 1)?;
    claim::check_dtype(spec, &w, "input 1 (W)")?;
    claim::check_shape(spec, &w, "input 1 (W)")?;

    require!(
        x.rank() == Some(CONV_SPATIAL_RANK + 2),
        Rank,
        "`{}` input 0 has rank {:?}; this kernel implements 2-D convolution only (rank 4)",
        spec.op_type,
        x.rank()
    );
    require!(
        w.rank() == Some(CONV_SPATIAL_RANK + 2),
        Rank,
        "`{}` input 1 (W) has rank {:?}; a 2-D convolution's weights are rank 4",
        spec.op_type,
        w.rank()
    );

    // `auto_pad` derives the pads from the output shape. Explicit pads are a fact about the node.
    if let Some(pad_mode) = view.attr_string("auto_pad") {
        require!(
            pad_mode == "NOTSET",
            Attribute,
            "`{}` has auto_pad={pad_mode}; this kernel needs explicit `pads`, because \
             SAME_UPPER/SAME_LOWER derive the padding from an output extent that is not known \
             when the pipeline is built",
            spec.op_type
        );
    }

    // The weights are an initializer on every graph we have seen, so their extents are static
    // and the kernel geometry is a compile-time fact. A symbolic weight extent is a graph this
    // EP declines rather than guesses at.
    let w_shape = w.shape.as_deref().unwrap_or(&[]);
    require!(
        w_shape.len() == CONV_SPATIAL_RANK + 2 && w_shape.iter().all(|&d| d > 0),
        DynamicShape,
        "`{}` input 1 (W) has a symbolic extent; the kernel geometry is baked into the pipeline",
        spec.op_type
    );

    // Spatial extents of X must be concrete: the output extent is not a linear function of them,
    // so it cannot be recovered from a ratio at Compute time the way a flat element count can.
    let x_shape = x.shape.as_deref().unwrap_or(&[]);
    require!(
        x_shape.len() == CONV_SPATIAL_RANK + 2
            && x_shape[1] > 0
            && x_shape[2] > 0
            && x_shape[3] > 0,
        DynamicShape,
        "`{}` input 0 has a symbolic channel or spatial extent ({x_shape:?}); only the batch \
         extent may be symbolic, because the padding arithmetic is not linear in H and W",
        spec.op_type
    );

    let attrs = |name: &str| view.attr_ints(name);
    let group_attr = view.attr_int("group");
    let a = match conv_attrs([w_shape[2], w_shape[3]], &attrs, group_attr) {
        Ok(a) => a,
        Err(why) => {
            crate::deny!(Attribute, "`{}` {why}", spec.op_type);
        }
    };

    let c = x_shape[1];
    let m = w_shape[0];
    require!(
        a.group > 0 && c % a.group == 0 && m % a.group == 0,
        Attribute,
        "`{}` group={} does not divide C={c} and M={m}",
        spec.op_type,
        a.group
    );
    require!(
        w_shape[1] == c / a.group,
        Shape,
        "`{}` weights declare {} input channels per group but X has C={c} over {} group(s)",
        spec.op_type,
        w_shape[1],
        a.group
    );

    for axis in 0..CONV_SPATIAL_RANK {
        let out = conv_out_extent(
            x_shape[2 + axis],
            a.pads[axis],
            a.pads[CONV_SPATIAL_RANK + axis],
            a.dilations[axis],
            a.kernel_shape[axis],
            a.strides[axis],
        );
        require!(
            out > 0,
            Shape,
            "`{}` spatial axis {axis} produces a {out}-wide output; the padded input is smaller \
             than the dilated kernel",
            spec.op_type
        );
    }

    // Bias is optional and is the third input when present. A fourth input is not `Conv`.
    require!(
        view.num_inputs() <= 3,
        Arity,
        "`{}` has {} inputs; `Conv` takes X, W and an optional B",
        spec.op_type,
        view.num_inputs()
    );
    if view.num_inputs() == 3 {
        let b = claim::input_edge(view, spec, 2)?;
        claim::check_dtype(spec, &b, "input 2 (B)")?;
        require!(
            b.rank() == Some(1),
            Rank,
            "`{}` input 2 (B) has rank {:?}; a bias is rank 1",
            spec.op_type,
            b.rank()
        );
    }
    Ok(())
}

crate::op_table! {
    //  op      domain  opsets            caps  kernel          claim   translate         status
    //
    // `caps` is F32 and not FLOAT, and that is the substance of the row: an f16 `Conv` declines
    // `[dtype]` rather than being read through a kernel whose loads assume one element per word.
    // Widening it needs `conv_f16.comp`, not an edit here.
    //
    // The window opens at 1: `Conv` has existed since opset 1 and its revisions (11) clarified
    // `auto_pad` and negative-pad behaviour, both of which this row declines explicitly.
    "Conv",   Ai,     1 ..= OPSET_ANY,  F32,  kernel!(None),  conv,   translate,        Ready;
}

/// Translate a 2-D convolution into one dispatch.
pub fn translate(_spec: &OpSpec, node: &NodeDesc, ctx: &mut dyn DispatchContext) -> EpResult<()> {
    let x = node.inputs.first().and_then(|t| t.desc.as_ref()).ok_or_else(|| {
        EpError::Unsupported(format!(
            "`{}` input 0 has no shape at compile time",
            node.op_type
        ))
    })?;
    let w = node.inputs.get(1).and_then(|t| t.desc.as_ref()).ok_or_else(|| {
        EpError::Unsupported(format!(
            "`{}` input 1 (W) has no shape at compile time",
            node.op_type
        ))
    })?;
    if x.shape.len() != CONV_SPATIAL_RANK + 2 || w.shape.len() != CONV_SPATIAL_RANK + 2 {
        return Err(EpError::Unsupported(format!(
            "`{}` was claimed with rank {} / {} inputs; this kernel is 2-D",
            node.op_type,
            x.shape.len(),
            w.shape.len()
        )));
    }
    if x.dtype != DType::F32 {
        return Err(EpError::Unsupported(format!(
            "`{}` input 0 is {:?}; conv_f32 reads one element per word",
            node.op_type, x.dtype
        )));
    }

    let attrs = |name: &str| match node.attributes.get(name) {
        Some(crate::engine::AttrValue::Ints(v)) => Some(v.clone()),
        _ => None,
    };
    let group = match node.attributes.get("group") {
        Some(crate::engine::AttrValue::Int(v)) => Some(*v),
        _ => None,
    };
    let a = conv_attrs([w.shape[2], w.shape[3]], &attrs, group)
        .map_err(|why| EpError::Unsupported(format!("`{}` {why}", node.op_type)))?;

    let (n, c, h, wd) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]);
    let m = w.shape[0];
    let oh = conv_out_extent(h, a.pads[0], a.pads[2], a.dilations[0], a.kernel_shape[0], a.strides[0]);
    let ow = conv_out_extent(wd, a.pads[1], a.pads[3], a.dilations[1], a.kernel_shape[1], a.strides[1]);
    if oh <= 0 || ow <= 0 || n <= 0 {
        return Err(EpError::Unsupported(format!(
            "`{}` computes a {oh}x{ow} output over batch {n}; nothing to dispatch",
            node.op_type
        )));
    }

    let has_bias = node.inputs.len() >= 3 && !node.inputs[2].name.is_empty();
    let x_buf = ctx.resolve(&node.inputs[0])?;
    let w_buf = ctx.resolve(&node.inputs[1])?;
    // An absent bias binds the *weights*, the same inert-placeholder rule `ew_clip` uses for an
    // omitted bound: it costs no allocation and cannot be out of range for index `oc`, which is
    // less than `M` and therefore less than the weight tensor's first extent.
    let b_buf = if has_bias {
        ctx.resolve(&node.inputs[2])?
    } else {
        w_buf
    };

    if node.outputs.len() != 1 {
        return Err(EpError::Internal(format!(
            "`{}` was claimed as single-output but has {}",
            node.op_type,
            node.outputs.len()
        )));
    }
    let out = &node.outputs[0];
    let out_buf = ctx.bind_output(out, TensorDesc::new(DType::F32, vec![n, m, oh, ow]))?;

    let total = u32::try_from(n * m * oh * ow).map_err(|_| {
        EpError::Unsupported(format!("`{}` output element count overflows u32", node.op_type))
    })?;

    let mut push = Vec::with_capacity(18 * 4);
    for v in [
        n, c, h, wd, m,
        a.kernel_shape[0], a.kernel_shape[1], oh, ow,
        a.pads[0], a.pads[1],
        a.strides[0], a.strides[1],
        a.dilations[0], a.dilations[1],
        a.group,
        i64::from(has_bias),
        i64::from(total),
    ] {
        let v = u32::try_from(v).map_err(|_| {
            EpError::Unsupported(format!("`{}` parameter {v} does not fit a u32", node.op_type))
        })?;
        push.extend_from_slice(&v.to_le_bytes());
    }

    let groups = total
        .div_ceil(CONV_LOCAL_SIZE)
        .clamp(1, CONV_MAX_WORKGROUPS);

    ctx.dispatch(KernelRequest {
        shader: "conv_f32",
        spec_constants: vec![CONV_LOCAL_SIZE],
        push_constants: push,
        bindings: vec![x_buf, w_buf, b_buf, out_buf],
        workgroups: [groups, 1, 1],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::common::templates;
    use crate::registry::OpStatus;

    #[test]
    fn the_conv_row_is_ready_and_points_at_the_conv_handler() {
        let row = OPS.iter().find(|s| s.op_type == "Conv").expect("Conv must be registered");
        assert_eq!(row.status, OpStatus::Ready);
        assert!(std::ptr::fn_addr_eq(
            row.translate,
            translate as crate::registry::TranslateHandler
        ));
        assert!(
            !std::ptr::fn_addr_eq(
                row.translate,
                templates::unimplemented as crate::registry::TranslateHandler
            ),
            "the comparison must discriminate, or the assertion above proves nothing"
        );
    }

    /// f16 must decline. There is no `conv_f16` module, and the f32 one would read half the
    /// tensor at twice the stride and return a plausible wrong answer.
    #[test]
    fn conv_claims_f32_only() {
        let row = OPS.iter().find(|s| s.op_type == "Conv").unwrap();
        assert!(row.caps.contains(DType::F32));
        assert!(
            !row.caps.contains(DType::F16),
            "packed-uint half I/O addresses two elements per word; a convolution reads single \
             scattered elements and needs its own module"
        );
    }

    /// The ONNX output formula, on the four shapes MobileNetV2 actually contains.
    #[test]
    fn output_extents_match_the_onnx_formula() {
        // 224 -> 112 : 3x3, stride 2, pads 0/1 (the asymmetric first layer)
        assert_eq!(conv_out_extent(224, 0, 1, 1, 3, 2), 112);
        // 112 -> 112 : 3x3 depthwise, stride 1, pads 1/1
        assert_eq!(conv_out_extent(112, 1, 1, 1, 3, 1), 112);
        // 112 -> 56 : 3x3, stride 2, pads 0/1
        assert_eq!(conv_out_extent(112, 0, 1, 1, 3, 2), 56);
        // 7 -> 7 : 1x1 pointwise
        assert_eq!(conv_out_extent(7, 0, 0, 1, 1, 1), 7);
        // dilation is in the formula, not implied by it
        assert_eq!(conv_out_extent(9, 0, 0, 2, 3, 1), 5);
    }

    /// A `kernel_shape` that contradicts the weights is refused rather than reinterpreted.
    ///
    /// The alternative is to trust one of the two, which is an unannounced reinterpretation of
    /// the model inside a dispatch — the same call I made for `Cast`'s destination type.
    #[test]
    fn kernel_shape_must_agree_with_the_weights() {
        let attrs = |name: &str| match name {
            "kernel_shape" => Some(vec![5, 5]),
            _ => None,
        };
        let err = conv_attrs([3, 3], &attrs, None).unwrap_err();
        assert!(err.contains("disagrees with the weight tensor"), "{err}");
    }

    #[test]
    fn defaults_come_from_the_schema_not_from_the_caller() {
        let attrs = |_: &str| None;
        let a = conv_attrs([3, 3], &attrs, None).unwrap();
        assert_eq!(a.strides, [1, 1]);
        assert_eq!(a.dilations, [1, 1]);
        assert_eq!(a.pads, [0, 0, 0, 0]);
        assert_eq!(a.group, 1);
        assert_eq!(a.kernel_shape, [3, 3]);
    }

    #[test]
    fn a_one_dimensional_attribute_list_is_refused() {
        let attrs = |name: &str| match name {
            "strides" => Some(vec![2]),
            _ => None,
        };
        let err = conv_attrs([3, 3], &attrs, None).unwrap_err();
        assert!(err.contains("2-D"), "{err}");
    }
}
