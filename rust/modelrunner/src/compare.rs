//! Output comparison, and the tolerance policy that makes a comparison mean something.
//!
//! THE POLICY
//! ----------
//! A tolerance chosen after seeing the numbers is not a tolerance, it is a rationalisation. So:
//!
//! * every model this runner is allowed to pass has an entry in [`TOLERANCES`], written down in
//!   advance, with the reason for the number in the table itself;
//! * a model with no entry is refused (`UNSUPPORTED(reason=no_tolerance_policy)`) unless the
//!   caller supplies `--rtol`/`--atol` explicitly, in which case the evidence records the source
//!   as `"cli"` so a reader can see the number did not come from the policy;
//! * the tolerance and its source are written into the evidence next to the result, so widening
//!   one is a visible diff in a reviewed file rather than a flag in a forgotten shell script.
//!
//! THE METRIC
//! ----------
//! `max_rel = max |v - w| / max(|w|, 1e-6)` against the CPU reference `w`, which is exactly what
//! `rust/tools/probe_model_output_agreement.py` computes. Keeping the formula identical is
//! deliberate: the Rust runner and the Python probe must be able to disagree about a model, and
//! that disagreement has to mean something other than "they measured differently".
//!
//! NaN is not a small difference. A NaN on either side is a DISAGREE with its own flag, never
//! silently skipped by a comparison operator that returns false for everything.

use onnxruntime_vulkan_ep::sys::ort;

use crate::error::{Failure, Result};
use crate::json::Json;
use crate::ortapi::{element_info, element_name};

/// The denominator floor, shared with the Python probe so the two metrics are the same number.
pub const REL_DENOM_FLOOR: f64 = 1e-6;

#[derive(Debug, Clone, PartialEq)]
pub struct Tolerance {
    pub rtol: f64,
    pub atol: f64,
    /// `"policy"` or `"cli"`: where the number came from.
    pub source: String,
    pub rationale: String,
}

impl Tolerance {
    pub fn to_json(&self) -> Json {
        Json::obj(vec![
            ("rtol", Json::n(self.rtol)),
            ("atol", Json::n(self.atol)),
            ("source", Json::s(self.source.as_str())),
            ("rationale", Json::s(self.rationale.as_str())),
        ])
    }
}

/// The written-down, per-model tolerance policy.
///
/// `(model name, rtol, atol, why this number)`.
pub const TOLERANCES: &[(&str, f64, f64, &str)] = &[
    (
        "mnist-12",
        1.0e-2,
        1.0e-5,
        "Eight layers of f32 Conv/Add/MaxPool with no normalisation; the same 1e-2 relative bound \
         rust/tools/probe_model_output_agreement.py applies to whole-model f32 comparisons. The \
         atol floor covers the near-zero logits that make a relative bound meaningless.",
    ),
    (
        "mobilenetv2-12",
        1.0e-2,
        1.0e-4,
        "Fifty-plus layers of Conv/BatchNorm/Clip in f32. Depth compounds rounding, and the \
         post-softmax outputs are small, so the absolute floor is one decade looser than \
         mnist-12 while the relative bound stays at the repository's 1e-2.",
    ),
    (
        "bertsquad-12",
        2.0e-2,
        1.0e-3,
        "Transformer with LayerNorm and Softmax over 384 positions; accumulation order differs \
         between a CPU reduction and a workgroup reduction, and the logits are O(10). Not yet \
         exercised -- issue #5 lists bertsquad-12 as a follow-on -- but the number is fixed here \
         in advance so the first run cannot choose its own.",
    ),
];

pub fn tolerance_for(model: &str) -> Option<Tolerance> {
    TOLERANCES
        .iter()
        .find(|(name, _, _, _)| *name == model)
        .map(|(_, rtol, atol, why)| Tolerance {
            rtol: *rtol,
            atol: *atol,
            source: "policy".to_string(),
            rationale: (*why).to_string(),
        })
}

/// Resolve the tolerance for a run, refusing to invent one.
pub fn resolve(model: &str, cli_rtol: Option<f64>, cli_atol: Option<f64>) -> Result<Tolerance> {
    if let (Some(rtol), Some(atol)) = (cli_rtol, cli_atol) {
        return Ok(Tolerance {
            rtol,
            atol,
            source: "cli".to_string(),
            rationale: "supplied on the command line; not part of the reviewed policy".to_string(),
        });
    }
    if cli_rtol.is_some() != cli_atol.is_some() {
        return Err(Failure::instrument(
            "partial_tolerance_override",
            "--rtol and --atol must be given together: half an overridden tolerance silently \
             inherits the other half from the policy, which is the kind of number nobody reviews."
                .to_string(),
        ));
    }
    tolerance_for(model).ok_or_else(|| {
        Failure::unsupported(
            "no_tolerance_policy",
            format!(
                "no tolerance is recorded for model {model:?}. Add it to compare::TOLERANCES with \
                 a stated reason, or pass --rtol and --atol explicitly. This runner will not pick \
                 a bound for a model it has never been told about, because a bound chosen at run \
                 time is a bound chosen to pass.\nKnown models: {}",
                TOLERANCES
                    .iter()
                    .map(|(n, _, _, _)| *n)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        )
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Exact,
    Agree,
    Disagree,
    ShapeMismatch,
    DtypeMismatch,
}

impl Verdict {
    pub fn as_str(self) -> &'static str {
        match self {
            Verdict::Exact => "EXACT",
            Verdict::Agree => "AGREE",
            Verdict::Disagree => "DISAGREE",
            Verdict::ShapeMismatch => "SHAPE_MISMATCH",
            Verdict::DtypeMismatch => "DTYPE_MISMATCH",
        }
    }

    pub fn is_pass(self) -> bool {
        matches!(self, Verdict::Exact | Verdict::Agree)
    }
}

#[derive(Debug, Clone)]
pub struct OutputComparison {
    pub name: String,
    pub verdict: Verdict,
    pub dtype: String,
    pub reference_shape: Vec<i64>,
    pub candidate_shape: Vec<i64>,
    pub elements: usize,
    pub max_abs: f64,
    pub max_rel: f64,
    /// Index of the worst element, so a disagreement can be pointed at rather than described.
    pub worst_index: usize,
    pub reference_at_worst: f64,
    pub candidate_at_worst: f64,
    pub nan_reference: usize,
    pub nan_candidate: usize,
    /// True when *every* reference value is zero: an all-zero output agrees with anything zero and
    /// is usually a sign the model never ran, not that the kernel is perfect.
    pub reference_all_zero: bool,
    pub detail: String,
}

impl OutputComparison {
    pub fn to_json(&self) -> Json {
        Json::obj(vec![
            ("name", Json::s(self.name.as_str())),
            ("verdict", Json::s(self.verdict.as_str())),
            ("dtype", Json::s(self.dtype.as_str())),
            (
                "reference_shape",
                Json::Arr(self.reference_shape.iter().map(|d| Json::int(*d)).collect()),
            ),
            (
                "candidate_shape",
                Json::Arr(self.candidate_shape.iter().map(|d| Json::int(*d)).collect()),
            ),
            ("elements", Json::int(self.elements as i64)),
            ("max_abs_diff", Json::n(self.max_abs)),
            ("max_rel_diff", Json::n(self.max_rel)),
            ("worst_index", Json::int(self.worst_index as i64)),
            ("reference_at_worst", Json::n(self.reference_at_worst)),
            ("candidate_at_worst", Json::n(self.candidate_at_worst)),
            ("nan_reference", Json::int(self.nan_reference as i64)),
            ("nan_candidate", Json::int(self.nan_candidate as i64)),
            ("reference_all_zero", Json::Bool(self.reference_all_zero)),
            ("detail", Json::s(self.detail.as_str())),
        ])
    }
}

/// Decode a tensor's raw bytes to f64 for comparison.
///
/// Everything numeric is widened to f64 rather than compared in its own type: f64 represents every
/// value of every type below it exactly, so the widening cannot manufacture or hide a difference,
/// and one comparison path is one path to get right.
#[allow(non_upper_case_globals)]
pub fn decode(bytes: &[u8], element_type: ort::ONNXTensorElementDataType) -> Result<Vec<f64>> {
    use ort::*;
    let (name, size) = element_info(element_type).ok_or_else(|| {
        Failure::unsupported(
            "output_dtype_unsupported",
            format!(
                "cannot compare an output of element type {}",
                element_name(element_type)
            ),
        )
    })?;
    if size == 0 || bytes.len() % size != 0 {
        return Err(Failure::instrument(
            "output_bytes_misaligned",
            format!(
                "{} bytes is not a whole number of {name} elements",
                bytes.len()
            ),
        ));
    }
    let mut out = Vec::with_capacity(bytes.len() / size);
    match element_type {
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT => {
            for c in bytes.chunks_exact(4) {
                out.push(f32::from_le_bytes(c.try_into().unwrap()) as f64);
            }
        }
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE => {
            for c in bytes.chunks_exact(8) {
                out.push(f64::from_le_bytes(c.try_into().unwrap()));
            }
        }
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 => {
            for c in bytes.chunks_exact(2) {
                out.push(f16_bits_to_f64(u16::from_le_bytes(c.try_into().unwrap())));
            }
        }
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 => {
            for c in bytes.chunks_exact(8) {
                out.push(i64::from_le_bytes(c.try_into().unwrap()) as f64);
            }
        }
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 => {
            for c in bytes.chunks_exact(4) {
                out.push(i32::from_le_bytes(c.try_into().unwrap()) as f64);
            }
        }
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 => {
            for c in bytes.chunks_exact(2) {
                out.push(i16::from_le_bytes(c.try_into().unwrap()) as f64);
            }
        }
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 => {
            for b in bytes {
                out.push(*b as i8 as f64);
            }
        }
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 => {
            for c in bytes.chunks_exact(8) {
                out.push(u64::from_le_bytes(c.try_into().unwrap()) as f64);
            }
        }
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 => {
            for c in bytes.chunks_exact(4) {
                out.push(u32::from_le_bytes(c.try_into().unwrap()) as f64);
            }
        }
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 => {
            for c in bytes.chunks_exact(2) {
                out.push(u16::from_le_bytes(c.try_into().unwrap()) as f64);
            }
        }
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
        | ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL => {
            for b in bytes {
                out.push(*b as f64);
            }
        }
        _ => {
            return Err(Failure::unsupported(
                "output_dtype_unsupported",
                format!(
                    "cannot compare an output of element type {}",
                    element_name(element_type)
                ),
            ));
        }
    }
    Ok(out)
}

pub fn f16_bits_to_f64(bits: u16) -> f64 {
    let sign = if bits & 0x8000 != 0 { -1.0f64 } else { 1.0 };
    let exponent = ((bits >> 10) & 0x1F) as i32;
    let mantissa = (bits & 0x03FF) as f64;
    if exponent == 0 {
        sign * mantissa * 2.0f64.powi(-24)
    } else if exponent == 0x1F {
        if mantissa == 0.0 {
            sign * f64::INFINITY
        } else {
            f64::NAN
        }
    } else {
        sign * (1.0 + mantissa / 1024.0) * 2.0f64.powi(exponent - 15)
    }
}

/// Compare one output tensor against its CPU reference.
#[allow(clippy::too_many_arguments)]
pub fn compare_output(
    name: &str,
    reference: &[f64],
    reference_shape: &[i64],
    reference_type: ort::ONNXTensorElementDataType,
    candidate: &[f64],
    candidate_shape: &[i64],
    candidate_type: ort::ONNXTensorElementDataType,
    tolerance: &Tolerance,
) -> OutputComparison {
    let dtype = element_name(reference_type);
    let mut cmp = OutputComparison {
        name: name.to_string(),
        verdict: Verdict::Exact,
        dtype: dtype.clone(),
        reference_shape: reference_shape.to_vec(),
        candidate_shape: candidate_shape.to_vec(),
        elements: reference.len(),
        max_abs: 0.0,
        max_rel: 0.0,
        worst_index: 0,
        reference_at_worst: 0.0,
        candidate_at_worst: 0.0,
        nan_reference: reference.iter().filter(|v| v.is_nan()).count(),
        nan_candidate: candidate.iter().filter(|v| v.is_nan()).count(),
        reference_all_zero: !reference.is_empty() && reference.iter().all(|v| *v == 0.0),
        detail: String::new(),
    };

    if reference_type != candidate_type {
        cmp.verdict = Verdict::DtypeMismatch;
        cmp.detail = format!(
            "CPU produced {} but Vulkan produced {}; a value comparison across two different \
             element types would compare two different tensors.",
            element_name(reference_type),
            element_name(candidate_type)
        );
        return cmp;
    }
    if reference_shape != candidate_shape {
        cmp.verdict = Verdict::ShapeMismatch;
        cmp.detail = format!(
            "CPU produced shape {reference_shape:?} but Vulkan produced {candidate_shape:?}."
        );
        return cmp;
    }
    if reference.len() != candidate.len() {
        cmp.verdict = Verdict::ShapeMismatch;
        cmp.detail = format!(
            "CPU produced {} elements but Vulkan produced {}, with matching declared shapes -- \
             which means one of the two byte lengths is wrong.",
            reference.len(),
            candidate.len()
        );
        return cmp;
    }

    let mut exact = true;
    for (i, (w, v)) in reference.iter().zip(candidate.iter()).enumerate() {
        // A NaN on exactly one side is a real difference that every ordinary comparison operator
        // reports as "not greater than", i.e. silently passes. Handle it before any arithmetic.
        if w.is_nan() != v.is_nan() {
            cmp.verdict = Verdict::Disagree;
            cmp.worst_index = i;
            cmp.reference_at_worst = *w;
            cmp.candidate_at_worst = *v;
            cmp.max_abs = f64::INFINITY;
            cmp.max_rel = f64::INFINITY;
            cmp.detail = format!(
                "element {i}: CPU {w} vs Vulkan {v}. A NaN on one side only is a disagreement, \
                 not a rounding difference."
            );
            return cmp;
        }
        if w.is_nan() && v.is_nan() {
            continue;
        }
        if w != v {
            exact = false;
        }
        let abs = (v - w).abs();
        let rel = abs / w.abs().max(REL_DENOM_FLOOR);
        if abs > cmp.max_abs {
            cmp.max_abs = abs;
        }
        if rel > cmp.max_rel {
            cmp.max_rel = rel;
            cmp.worst_index = i;
            cmp.reference_at_worst = *w;
            cmp.candidate_at_worst = *v;
        }
    }

    // Pass on either bound: a relative bound is meaningless near zero, an absolute one is
    // meaningless at scale, and requiring both would fail every large logit and every tiny
    // probability for reasons unrelated to the kernel.
    let within = cmp.max_rel <= tolerance.rtol || cmp.max_abs <= tolerance.atol;
    cmp.verdict = if exact {
        Verdict::Exact
    } else if within {
        Verdict::Agree
    } else {
        Verdict::Disagree
    };
    cmp.detail = match cmp.verdict {
        Verdict::Exact => "bit-identical to the CPU reference".to_string(),
        Verdict::Agree => format!(
            "max_rel={:.6e} (<= {:.1e}) or max_abs={:.6e} (<= {:.1e})",
            cmp.max_rel, tolerance.rtol, cmp.max_abs, tolerance.atol
        ),
        _ => format!(
            "max_rel={:.6e} > rtol {:.1e} and max_abs={:.6e} > atol {:.1e}; worst at element {} \
             (CPU {} vs Vulkan {})",
            cmp.max_rel,
            tolerance.rtol,
            cmp.max_abs,
            tolerance.atol,
            cmp.worst_index,
            cmp.reference_at_worst,
            cmp.candidate_at_worst
        ),
    };
    cmp
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnxruntime_vulkan_ep::sys::ort::*;

    const F32: ort::ONNXTensorElementDataType =
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;

    fn tol() -> Tolerance {
        Tolerance {
            rtol: 1e-2,
            atol: 1e-5,
            source: "test".into(),
            rationale: String::new(),
        }
    }

    fn cmp(a: &[f64], b: &[f64]) -> OutputComparison {
        compare_output(
            "y",
            a,
            &[a.len() as i64],
            F32,
            b,
            &[b.len() as i64],
            F32,
            &tol(),
        )
    }

    #[test]
    fn identical_outputs_are_exact_not_merely_within_tolerance() {
        let c = cmp(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]);
        assert_eq!(c.verdict, Verdict::Exact);
        assert_eq!(c.max_abs, 0.0);
        assert!(c.verdict.is_pass());
    }

    #[test]
    fn a_small_relative_difference_agrees_and_a_large_one_does_not() {
        assert_eq!(cmp(&[100.0], &[100.5]).verdict, Verdict::Agree);
        let bad = cmp(&[100.0], &[150.0]);
        assert_eq!(bad.verdict, Verdict::Disagree);
        assert!(!bad.verdict.is_pass());
        assert!(bad.detail.contains("worst at element 0"), "{}", bad.detail);
    }

    #[test]
    fn a_nan_on_one_side_disagrees_instead_of_passing_by_comparison_semantics() {
        // The defect this guards: `(v - w).abs() > rtol` is false when either side is NaN, so a
        // naive loop reports AGREE for an output that is entirely NaN.
        let c = cmp(&[1.0, 2.0], &[1.0, f64::NAN]);
        assert_eq!(c.verdict, Verdict::Disagree);
        assert_eq!(c.worst_index, 1);
        assert!(c.detail.contains("NaN"), "{}", c.detail);
    }

    #[test]
    fn matching_nans_are_not_a_disagreement_by_themselves() {
        let c = cmp(&[f64::NAN, 1.0], &[f64::NAN, 1.0]);
        assert!(c.verdict.is_pass());
        assert_eq!(c.nan_reference, 1);
        assert_eq!(c.nan_candidate, 1);
    }

    #[test]
    fn shape_and_dtype_mismatches_are_reported_as_themselves() {
        let s = compare_output("y", &[1.0], &[1], F32, &[1.0], &[1, 1], F32, &tol());
        assert_eq!(s.verdict, Verdict::ShapeMismatch);
        let d = compare_output(
            "y",
            &[1.0],
            &[1],
            F32,
            &[1.0],
            &[1],
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
            &tol(),
        );
        assert_eq!(d.verdict, Verdict::DtypeMismatch);
    }

    #[test]
    fn an_all_zero_reference_is_flagged_even_when_the_comparison_passes() {
        // Two all-zero tensors agree perfectly and prove nothing; the flag is what lets a reader
        // tell "the kernel is exact" from "neither side computed anything".
        let c = cmp(&[0.0, 0.0], &[0.0, 0.0]);
        assert!(c.verdict.is_pass());
        assert!(c.reference_all_zero);
    }

    #[test]
    fn near_zero_references_pass_on_the_absolute_bound() {
        // rel = 1e-9 / 1e-6 = 1e-3 relative to the floor, and abs is far under atol.
        let c = cmp(&[0.0], &[1e-9]);
        assert!(c.verdict.is_pass(), "{}", c.detail);
    }

    #[test]
    fn the_relative_metric_uses_the_same_floor_as_the_python_probe() {
        let c = cmp(&[0.0], &[1e-3]);
        // 1e-3 / max(0, 1e-6) == 1e3
        assert!((c.max_rel - 1e3).abs() < 1.0, "{}", c.max_rel);
    }

    #[test]
    fn every_policy_model_has_a_stated_reason_and_a_sane_bound() {
        assert!(!TOLERANCES.is_empty());
        for (name, rtol, atol, why) in TOLERANCES {
            assert!(*rtol > 0.0 && *rtol < 1.0, "{name} rtol {rtol}");
            assert!(*atol > 0.0 && *atol < 1.0, "{name} atol {atol}");
            assert!(
                why.len() > 40,
                "{name} needs a stated reason for its tolerance, not just a number"
            );
        }
        assert!(tolerance_for("mnist-12").is_some());
        assert!(tolerance_for("mobilenetv2-12").is_some());
    }

    #[test]
    fn an_unknown_model_is_refused_rather_than_given_a_default_bound() {
        let err = resolve("some-new-model", None, None).unwrap_err();
        assert_eq!(err.token(), "UNSUPPORTED(reason=no_tolerance_policy)");
        assert!(err.message.contains("mnist-12"), "{}", err.message);
    }

    #[test]
    fn an_explicit_override_is_allowed_but_recorded_as_not_being_policy() {
        let t = resolve("some-new-model", Some(1e-3), Some(1e-6)).unwrap();
        assert_eq!(t.source, "cli");
        assert_eq!(t.rtol, 1e-3);
        // Half an override is refused: the other half would silently come from somewhere else.
        assert_eq!(
            resolve("mnist-12", Some(1e-3), None).unwrap_err().token(),
            "ERROR(instrument=partial_tolerance_override)"
        );
    }

    #[test]
    fn policy_beats_nothing_but_cli_beats_policy() {
        assert_eq!(resolve("mnist-12", None, None).unwrap().source, "policy");
        assert_eq!(
            resolve("mnist-12", Some(9.0e-1), Some(9.0e-1))
                .unwrap()
                .source,
            "cli"
        );
    }

    #[test]
    fn decoding_round_trips_every_supported_element_type() {
        assert_eq!(decode(&1.5f32.to_le_bytes(), F32).unwrap(), vec![1.5]);
        assert_eq!(
            decode(
                &(-7i64).to_le_bytes(),
                ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
            )
            .unwrap(),
            vec![-7.0]
        );
        assert_eq!(
            decode(
                &[0xFF],
                ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8
            )
            .unwrap(),
            vec![-1.0]
        );
        assert_eq!(
            decode(
                &0x3C00u16.to_le_bytes(),
                ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
            )
            .unwrap(),
            vec![1.0]
        );
        assert_eq!(
            decode(
                &[1u8],
                ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL
            )
            .unwrap(),
            vec![1.0]
        );
    }

    #[test]
    fn a_partial_element_is_an_error_rather_than_a_truncated_tensor() {
        let err = decode(&[0u8, 1, 2], F32).unwrap_err();
        assert_eq!(err.token(), "ERROR(instrument=output_bytes_misaligned)");
        let err = decode(
            &[0u8],
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING,
        )
        .unwrap_err();
        assert_eq!(err.token(), "UNSUPPORTED(reason=output_dtype_unsupported)");
    }

    #[test]
    fn half_precision_decodes_the_values_the_encoder_produces() {
        for v in [0.0f32, 1.0, -2.0, 0.5, 0.125] {
            let bits = crate::feeds::f32_to_f16_bits(v);
            assert_eq!(f16_bits_to_f64(bits), v as f64, "{v}");
        }
    }
}
