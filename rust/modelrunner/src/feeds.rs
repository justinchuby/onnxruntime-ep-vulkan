//! Deterministic input generation, and the rule for pinning free dimensions.
//!
//! WHY DETERMINISTIC MATTERS MORE THAN RANDOM
//! ------------------------------------------
//! A CPU-vs-Vulkan comparison is only evidence if both providers saw *the same bytes*. Drawing
//! from the OS entropy source, or from a PRNG seeded by the clock, turns a failed comparison into
//! an unreproducible anecdote. Every value here comes from SplitMix64 seeded by an explicit,
//! recorded seed, so a disagreement can be replayed exactly -- on another machine, in another
//! process, months later -- by quoting the seed from the evidence file.
//!
//! SplitMix64 is implemented in-crate rather than pulled from `rand`, for the same reason the rest
//! of this crate has no dependencies: a runner that exists because a package index was unreachable
//! must not need a package index. It is a published, fixed algorithm (Steele/Lea/Flood 2014) whose
//! outputs this module pins against reference values in its tests, so "in-crate" costs no
//! confidence.
//!
//! FREE DIMENSIONS
//! ---------------
//! ORT reports an unresolved dimension as `-1`, usually with a symbolic name. Choosing a value is
//! unavoidable, so the rule is: pin to 1 by default (matching
//! `rust/tools/probe_model_output_agreement.py`, so the two instruments stay comparable), allow
//! explicit `--free-dim name=N` overrides, and *record every pin in the evidence*. A batch size
//! chosen silently is a shape the reader cannot check.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use onnxruntime_vulkan_ep::sys::ort;

use crate::error::{Failure, Result};
use crate::json::Json;
use crate::ortapi::{Api, MemoryInfo, TensorSpec, Value, element_info, element_name};

/// SplitMix64. Fixed algorithm, fixed seed, reproducible bytes.
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1) using the top 53 bits, the standard double construction.
    pub fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform in [-1, 1). Chosen over a normal distribution for the default because it cannot
    /// produce the rare huge magnitudes that make a relative-error comparison fail for reasons
    /// that have nothing to do with the kernel under test.
    pub fn next_signed_unit(&mut self) -> f64 {
        self.next_f64() * 2.0 - 1.0
    }
}

/// How a free dimension got its value, so the evidence can say so.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DimPin {
    pub input: String,
    pub axis: usize,
    pub symbol: String,
    pub value: i64,
    /// `"default"` or `"override"`.
    pub source: String,
}

impl DimPin {
    pub fn to_json(&self) -> Json {
        Json::obj(vec![
            ("input", Json::s(self.input.as_str())),
            ("axis", Json::int(self.axis as i64)),
            ("symbol", Json::s(self.symbol.as_str())),
            ("value", Json::int(self.value)),
            ("source", Json::s(self.source.as_str())),
        ])
    }
}

/// What was actually fed, for the record.
#[derive(Debug, Clone)]
pub struct FeedRecord {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<i64>,
    pub elements: usize,
    /// `"generated"` or the path a file came from.
    pub source: String,
}

impl FeedRecord {
    pub fn to_json(&self) -> Json {
        Json::obj(vec![
            ("name", Json::s(self.name.as_str())),
            ("dtype", Json::s(self.dtype.as_str())),
            (
                "shape",
                Json::Arr(self.shape.iter().map(|d| Json::int(*d)).collect()),
            ),
            ("elements", Json::int(self.elements as i64)),
            ("source", Json::s(self.source.as_str())),
        ])
    }
}

/// Resolve a declared shape to a concrete one, recording every choice made.
pub fn resolve_shape(
    spec: &TensorSpec,
    overrides: &BTreeMap<String, i64>,
    pins: &mut Vec<DimPin>,
) -> Result<Vec<i64>> {
    let mut shape = Vec::with_capacity(spec.dims.len());
    for (axis, &dim) in spec.dims.iter().enumerate() {
        if dim >= 0 {
            shape.push(dim);
            continue;
        }
        let symbol = spec.symbolic.get(axis).cloned().unwrap_or_default();
        // An override may be keyed by the symbol ("batch_size") or by "input:axis", because not
        // every free dimension has a name and two different inputs may share one.
        let positional = format!("{}:{}", spec.name, axis);
        let (value, source) = if let Some(v) = overrides.get(&positional) {
            (*v, "override")
        } else if !symbol.is_empty() && overrides.contains_key(&symbol) {
            (overrides[&symbol], "override")
        } else {
            (1, "default")
        };
        if value <= 0 {
            return Err(Failure::condition(
                "free_dim_not_positive",
                format!(
                    "--free-dim resolved {} axis {axis} ({symbol}) to {value}; a dimension must be \
                     at least 1",
                    spec.name
                ),
            ));
        }
        pins.push(DimPin {
            input: spec.name.clone(),
            axis,
            symbol,
            value,
            source: source.to_string(),
        });
        shape.push(value);
    }
    Ok(shape)
}

pub fn element_count(shape: &[i64]) -> Result<usize> {
    let mut total: usize = 1;
    for &d in shape {
        if d < 0 {
            return Err(Failure::instrument(
                "shape_unresolved",
                format!("a negative extent survived shape resolution: {shape:?}"),
            ));
        }
        total = total.checked_mul(d as usize).ok_or_else(|| {
            Failure::unsupported(
                "shape_too_large",
                format!("the element count of {shape:?} overflows usize"),
            )
        })?;
    }
    Ok(total)
}

/// IEEE 754 binary16 encoding of an f32, round-to-nearest-even.
///
/// Hand-rolled because `f16` is not stable on this toolchain and this crate takes no dependencies.
/// Only the finite, in-range cases can arise from the bounded generator above, but the subnormal
/// and overflow paths are implemented and tested anyway: a generator that silently emitted
/// infinity would poison a comparison rather than fail it.
pub fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x007F_FFFF;

    if exponent == 0xFF {
        // Inf or NaN. Preserve NaN-ness rather than turning it into infinity.
        let m = if mantissa != 0 { 0x0200 } else { 0 };
        return sign | 0x7C00 | m;
    }
    let unbiased = exponent - 127;
    if unbiased > 15 {
        return sign | 0x7C00; // overflow to infinity
    }
    if unbiased < -24 {
        return sign; // underflow to zero
    }
    if unbiased < -14 {
        // Subnormal half.
        let shift = (-14 - unbiased) as u32;
        let full = mantissa | 0x0080_0000;
        let mut half = (full >> (shift + 13)) as u16;
        let round_bit = 1u32 << (shift + 12);
        if (full & round_bit) != 0 && ((full & (round_bit - 1)) != 0 || (half & 1) != 0) {
            half += 1;
        }
        return sign | half;
    }
    let mut half_exp = ((unbiased + 15) as u16) << 10;
    let mut half_man = (mantissa >> 13) as u16;
    if (mantissa & 0x1000) != 0 && ((mantissa & 0x0FFF) != 0 || (half_man & 1) != 0) {
        half_man += 1;
        if half_man == 0x0400 {
            half_man = 0;
            half_exp += 1 << 10;
            if half_exp >= 0x7C00 {
                return sign | 0x7C00;
            }
        }
    }
    sign | half_exp | half_man
}

/// Generate the bytes for one input, or read them from a file.
///
/// The refusal to invent data for a dtype it does not understand is the point: a string or complex
/// tensor filled with plausible bytes would run, agree with itself, and prove nothing.
#[allow(non_upper_case_globals)]
pub fn make_bytes(
    spec: &TensorSpec,
    shape: &[i64],
    rng: &mut SplitMix64,
    file: Option<&Path>,
) -> Result<(Vec<u8>, String)> {
    let count = element_count(shape)?;
    let (dtype, size) = element_info(spec.element_type).ok_or_else(|| {
        Failure::unsupported(
            "input_dtype_unsupported",
            format!(
                "input {} has element type {}, which this runner cannot generate. Feed it with \
                 --input {}=<file.raw> or extend feeds.rs deliberately.",
                spec.name,
                element_name(spec.element_type),
                spec.name
            ),
        )
    })?;
    let want = count.checked_mul(size).ok_or_else(|| {
        Failure::unsupported(
            "shape_too_large",
            format!("{count} elements of {dtype} overflow usize"),
        )
    })?;

    if let Some(path) = file {
        let bytes = std::fs::read(path).map_err(|e| {
            Failure::instrument(
                "input_file_unreadable",
                format!("cannot read --input file {}: {e}", path.display()),
            )
        })?;
        if bytes.len() != want {
            return Err(Failure::condition(
                "input_file_wrong_size",
                format!(
                    "--input {}={} is {} bytes; the model wants {} ({} {dtype} elements of shape \
                     {:?}). A truncated or padded feed is not the tensor the model declared.",
                    spec.name,
                    path.display(),
                    bytes.len(),
                    want,
                    count,
                    shape
                ),
            ));
        }
        return Ok((bytes, path.display().to_string()));
    }

    use ort::*;
    let mut out = Vec::with_capacity(want);
    match spec.element_type {
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT => {
            for _ in 0..count {
                out.extend_from_slice(&(rng.next_signed_unit() as f32).to_le_bytes());
            }
        }
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE => {
            for _ in 0..count {
                out.extend_from_slice(&rng.next_signed_unit().to_le_bytes());
            }
        }
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 => {
            for _ in 0..count {
                out.extend_from_slice(
                    &f32_to_f16_bits(rng.next_signed_unit() as f32).to_le_bytes(),
                );
            }
        }
        // Small non-negative integers for every integral type: these are almost always indices,
        // token ids or masks, and a full-range draw would be out of bounds for every embedding
        // table and gather in existence -- a crash that says nothing about the EP.
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 => {
            for _ in 0..count {
                out.extend_from_slice(&((rng.next_u64() % 16) as i64).to_le_bytes());
            }
        }
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 => {
            for _ in 0..count {
                out.extend_from_slice(&((rng.next_u64() % 16) as i32).to_le_bytes());
            }
        }
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 => {
            for _ in 0..count {
                out.extend_from_slice(&((rng.next_u64() % 16) as i16).to_le_bytes());
            }
        }
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 => {
            for _ in 0..count {
                out.push((rng.next_u64() % 16) as u8);
            }
        }
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 => {
            for _ in 0..count {
                out.extend_from_slice(&(rng.next_u64() % 16).to_le_bytes());
            }
        }
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 => {
            for _ in 0..count {
                out.extend_from_slice(&((rng.next_u64() % 16) as u32).to_le_bytes());
            }
        }
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 => {
            for _ in 0..count {
                out.extend_from_slice(&((rng.next_u64() % 16) as u16).to_le_bytes());
            }
        }
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 => {
            for _ in 0..count {
                out.push((rng.next_u64() % 16) as u8);
            }
        }
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL => {
            for _ in 0..count {
                out.push((rng.next_u64() & 1) as u8);
            }
        }
        _ => {
            return Err(Failure::unsupported(
                "input_dtype_unsupported",
                format!(
                    "input {} has element type {}; this runner will not invent values for it.",
                    spec.name,
                    element_name(spec.element_type)
                ),
            ));
        }
    }
    debug_assert_eq!(out.len(), want);
    Ok((out, "generated".to_string()))
}

/// Everything one provider needs to be fed, plus the record of what that was.
pub struct Feeds {
    pub values: Vec<(String, Value)>,
    pub records: Vec<FeedRecord>,
    pub pins: Vec<DimPin>,
}

/// Build the whole feed set for a session, identically for every provider.
pub fn build(
    api: Api,
    memory_info: &MemoryInfo,
    inputs: &[TensorSpec],
    seed: u64,
    overrides: &BTreeMap<String, i64>,
    files: &BTreeMap<String, PathBuf>,
) -> Result<Feeds> {
    for name in files.keys() {
        if !inputs.iter().any(|s| &s.name == name) {
            return Err(Failure::condition(
                "input_file_unknown_name",
                format!(
                    "--input {name}=... names an input this model does not declare. Declared: {}",
                    inputs
                        .iter()
                        .map(|s| s.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            ));
        }
    }
    // One RNG for the whole feed set, advanced input by input in declaration order: the same seed
    // therefore reproduces the same *set*, not merely the same first tensor.
    let mut rng = SplitMix64::new(seed);
    let mut pins = Vec::new();
    let mut values = Vec::new();
    let mut records = Vec::new();
    for spec in inputs {
        let shape = resolve_shape(spec, overrides, &mut pins)?;
        let file = files.get(&spec.name).map(|p| p.as_path());
        let (bytes, source) = make_bytes(spec, &shape, &mut rng, file)?;
        let (dtype, _) = element_info(spec.element_type).unwrap_or(("unknown", 0));
        records.push(FeedRecord {
            name: spec.name.clone(),
            dtype: dtype.to_string(),
            shape: shape.clone(),
            elements: element_count(&shape)?,
            source,
        });
        let value = Value::tensor_from_host(api, memory_info, bytes, &shape, spec.element_type)?;
        values.push((spec.name.clone(), value));
    }
    Ok(Feeds {
        values,
        records,
        pins,
    })
}

/// Parse `name=value` for `--free-dim`.
pub fn parse_free_dim(arg: &str) -> Result<(String, i64)> {
    let (name, value) = arg.split_once('=').ok_or_else(|| {
        Failure::instrument(
            "bad_free_dim",
            format!("--free-dim expects name=value, got {arg:?}"),
        )
    })?;
    let parsed: i64 = value.trim().parse().map_err(|_| {
        Failure::instrument(
            "bad_free_dim",
            format!("--free-dim {arg:?} has a non-integer extent"),
        )
    })?;
    if parsed <= 0 {
        return Err(Failure::instrument(
            "bad_free_dim",
            format!("--free-dim {arg:?} must be at least 1"),
        ));
    }
    Ok((name.trim().to_string(), parsed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnxruntime_vulkan_ep::sys::ort::*;

    fn spec(
        name: &str,
        ty: ort::ONNXTensorElementDataType,
        dims: &[i64],
        sym: &[&str],
    ) -> TensorSpec {
        TensorSpec {
            name: name.to_string(),
            element_type: ty,
            dims: dims.to_vec(),
            symbolic: sym.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn splitmix64_matches_the_published_reference_stream() {
        // Reference values for seed 0 from the SplitMix64 publication. If this crate's PRNG ever
        // drifts, every seed recorded in every evidence file stops reproducing, so the stream is
        // pinned rather than merely exercised.
        let mut rng = SplitMix64::new(0);
        assert_eq!(rng.next_u64(), 0xE220A8397B1DCDAF);
        assert_eq!(rng.next_u64(), 0x6E789E6AA1B965F4);
        assert_eq!(rng.next_u64(), 0x06C45D188009454F);
    }

    #[test]
    fn the_same_seed_gives_the_same_bytes_and_a_different_seed_does_not() {
        let s = spec(
            "x",
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &[2, 3],
            &["", ""],
        );
        let a = make_bytes(&s, &[2, 3], &mut SplitMix64::new(7), None)
            .unwrap()
            .0;
        let b = make_bytes(&s, &[2, 3], &mut SplitMix64::new(7), None)
            .unwrap()
            .0;
        let c = make_bytes(&s, &[2, 3], &mut SplitMix64::new(8), None)
            .unwrap()
            .0;
        assert_eq!(a, b, "a fixed seed must reproduce the feed exactly");
        assert_ne!(a, c);
        assert_eq!(a.len(), 6 * 4);
    }

    #[test]
    fn generated_floats_stay_inside_the_bounded_range() {
        let s = spec(
            "x",
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &[64],
            &[""],
        );
        let bytes = make_bytes(&s, &[64], &mut SplitMix64::new(3), None)
            .unwrap()
            .0;
        for chunk in bytes.chunks_exact(4) {
            let v = f32::from_le_bytes(chunk.try_into().unwrap());
            assert!(v.is_finite() && (-1.0..1.0).contains(&v), "{v}");
        }
    }

    #[test]
    fn free_dims_default_to_one_and_record_the_pin() {
        let s = spec(
            "input",
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &[-1, 3, 224, 224],
            &["batch_size", "", "", ""],
        );
        let mut pins = Vec::new();
        let shape = resolve_shape(&s, &BTreeMap::new(), &mut pins).unwrap();
        assert_eq!(shape, vec![1, 3, 224, 224]);
        assert_eq!(pins.len(), 1);
        assert_eq!(pins[0].symbol, "batch_size");
        assert_eq!(pins[0].value, 1);
        assert_eq!(pins[0].source, "default");
    }

    #[test]
    fn a_free_dim_override_applies_by_symbol_or_by_position() {
        let s = spec(
            "input",
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &[-1, -1],
            &["batch", ""],
        );
        let overrides =
            BTreeMap::from([("batch".to_string(), 4i64), ("input:1".to_string(), 7i64)]);
        let mut pins = Vec::new();
        let shape = resolve_shape(&s, &overrides, &mut pins).unwrap();
        assert_eq!(shape, vec![4, 7]);
        assert!(pins.iter().all(|p| p.source == "override"));
    }

    #[test]
    fn an_unsupported_dtype_is_refused_rather_than_filled_with_plausible_bytes() {
        let s = spec(
            "text",
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING,
            &[2],
            &[""],
        );
        let err = make_bytes(&s, &[2], &mut SplitMix64::new(1), None).unwrap_err();
        assert_eq!(err.token(), "UNSUPPORTED(reason=input_dtype_unsupported)");
        assert!(err.message.contains("element type"));
    }

    #[test]
    fn an_input_file_of_the_wrong_size_is_a_failure_not_a_pad() {
        let dir = std::env::temp_dir().join("ort_model_runner_feeds_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("short.raw");
        std::fs::write(&path, [0u8; 4]).unwrap();
        let s = spec(
            "x",
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &[8],
            &[""],
        );
        let err = make_bytes(&s, &[8], &mut SplitMix64::new(1), Some(&path)).unwrap_err();
        assert_eq!(err.token(), "FAIL(condition=input_file_wrong_size)");
        assert!(err.message.contains("32"), "{}", err.message);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn half_precision_encoding_matches_ieee754() {
        assert_eq!(f32_to_f16_bits(0.0), 0x0000);
        assert_eq!(f32_to_f16_bits(-0.0), 0x8000);
        assert_eq!(f32_to_f16_bits(1.0), 0x3C00);
        assert_eq!(f32_to_f16_bits(-2.0), 0xC000);
        assert_eq!(f32_to_f16_bits(0.5), 0x3800);
        // Beyond the half range: infinity, not a wrapped-around finite value.
        assert_eq!(f32_to_f16_bits(1.0e30), 0x7C00);
        assert_eq!(f32_to_f16_bits(-1.0e30), 0xFC00);
        // Below the subnormal range: zero, with the sign kept.
        assert_eq!(f32_to_f16_bits(1.0e-30), 0x0000);
        assert_eq!(f32_to_f16_bits(f32::NAN) & 0x7C00, 0x7C00);
        assert_ne!(f32_to_f16_bits(f32::NAN) & 0x03FF, 0, "NaN must stay NaN");
    }

    #[test]
    fn element_count_refuses_to_overflow_silently() {
        assert_eq!(element_count(&[2, 3, 4]).unwrap(), 24);
        assert_eq!(element_count(&[]).unwrap(), 1);
        let err = element_count(&[i64::MAX, i64::MAX]).unwrap_err();
        assert_eq!(err.token(), "UNSUPPORTED(reason=shape_too_large)");
    }

    #[test]
    fn free_dim_parsing_rejects_shapes_that_cannot_exist() {
        assert_eq!(parse_free_dim("batch=4").unwrap(), ("batch".into(), 4));
        assert_eq!(
            parse_free_dim("batch").unwrap_err().token(),
            "ERROR(instrument=bad_free_dim)"
        );
        assert_eq!(
            parse_free_dim("batch=0").unwrap_err().token(),
            "ERROR(instrument=bad_free_dim)"
        );
        assert_eq!(
            parse_free_dim("batch=-1").unwrap_err().token(),
            "ERROR(instrument=bad_free_dim)"
        );
    }
}
