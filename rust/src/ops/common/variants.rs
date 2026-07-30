//! The shader-variant table: one row of the op table, one SPIR-V module per dtype.
//!
//! # Why this exists
//!
//! `OP_COVERAGE.md` §5.4 argues that the leverage in a Vulkan backend is not in writing kernels,
//! it is in *generating* them: a handful of GLSL templates × the dtypes each op claims produces
//! hundreds of specialised modules for the cost of a build step. That only works if there is
//! exactly one place that decides what a variant is called, and this is it.
//!
//! # The naming rule
//!
//! ```text
//! <template-prefix>_<template-op>_<dtype-suffix>
//! ew_binary_add_f32     ew_unary_sqrt_f16     ew_select_where_i32
//! ```
//!
//! [`crate::engine::KernelRequest::shader`] is a `&'static str`, so a stem may never be formatted
//! at runtime. The [`kernel!`] macro therefore builds all [`DTYPE_COUNT`] stems for a row with
//! `concat!` at compile time and stores them in the row itself. A row's [`Kernel`] is the single
//! source of truth for both "which shader do I dispatch" and "which shaders must exist".
//!
//! # The build-time seam
//!
//! [`manifest`] walks the registry and emits, for every `(row, dtype in row.caps)` pair, the stem,
//! the GLSL template it compiles from, and the exact `-D` defines it needs. That is everything a
//! variant-expanding `build.rs` requires. The manifest is checked in as
//! `src/ops/shader_variants.txt` and a test fails if it drifts, so the build's view and the
//! registry's view cannot silently diverge.

use crate::engine::DType;
use crate::ops::common::dtype::{DTYPE_COUNT, dtype_index, dtype_suffix};

/// A family of GLSL compute shaders sharing one template source.
///
/// Deliberately coarse. `OP_COVERAGE.md` §5.1–5.3 is the argument: ~66 elementwise ops collapse
/// into three templates, and every template we *don't* add is an op family we didn't hand-write.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Template {
    /// No shader — a metadata-only row, or one whose kernel is still being designed.
    None,
    /// One input, one output, elementwise, shape-preserving. `Sqrt`, `Relu`, `Neg`, ...
    EwUnary,
    /// Two inputs, numpy broadcasting, one output. `Add`, `Mul`, `Pow`, `Greater`, ...
    EwBinary,
    /// Three inputs, numpy broadcasting, one output. `Where`, `Clip`(3-input form).
    EwSelect,
    /// Block-dequantising GEMV: `MatMulNBits`. Five bindings, no broadcasting, its own push
    /// block. Not an elementwise template and deliberately not pretending to be one — it is the
    /// first row that earns a hand-written kernel, per `OP_COVERAGE.md` §7.1.3.
    QGemv,
}

impl Template {
    /// The stem prefix, and the base name of the GLSL template source.
    pub const fn prefix(self) -> &'static str {
        match self {
            Template::None => "",
            Template::EwUnary => "ew_unary",
            Template::EwBinary => "ew_binary",
            Template::EwSelect => "ew_select",
            Template::QGemv => "q_gemv",
        }
    }

    /// The GLSL source this template compiles from, relative to `shaders/glsl/`.
    ///
    /// Templates live in `shaders/glsl/templates/` rather than directly in `shaders/glsl/`, and
    /// the subdirectory is load-bearing: `build.rs` compiles every `*.comp` it finds *directly* in
    /// `shaders/glsl/` with no `-D` defines (that is the path for hand-written XL kernels, which
    /// need none). A template compiled with no defines has no `EW_OP` and no `DTYPE_*` and fails
    /// on purpose. Keeping templates one level down means the direct scan does not see them and
    /// only the variant table drives their compilation.
    pub fn source_file(self) -> Option<String> {
        match self {
            Template::None => None,
            other => Some(format!("templates/{}.comp", other.prefix())),
        }
    }

    /// How many tensor inputs the template binds. Claim predicates use it for the arity check, so
    /// the arity of an op is also a property of its row rather than of its predicate.
    pub const fn input_arity(self) -> usize {
        match self {
            Template::None => 0,
            Template::EwUnary => 1,
            Template::EwBinary => 2,
            Template::EwSelect => 3,
            // A, B, scales, zero_points, and — when the node has no zero points — `scales` bound a
            // second time as an inert placeholder. The arity is therefore a property of the
            // *shader*, which always declares five, not of the node, which may have three inputs.
            Template::QGemv => 5,
        }
    }

    /// Whether this template's op identity reaches the shader as `-DEW_OP=OP_<op>`.
    ///
    /// The elementwise families are one source selected by an op code; a hand-written kernel is
    /// one op and carries no selector. Several build-manifest invariants differ between the two,
    /// so the distinction is named once here rather than re-derived by matching on the variant.
    pub const fn is_elementwise(self) -> bool {
        matches!(
            self,
            Template::EwUnary | Template::EwBinary | Template::EwSelect
        )
    }

    /// Every template that has a source file.
    pub const ALL: &'static [Template] = &[
        Template::EwUnary,
        Template::EwBinary,
        Template::EwSelect,
        Template::QGemv,
    ];
}

/// The GLSL scalar type a dtype maps to inside a template.
///
/// `u8` and `bool` are stored packed in `uint` buffers (`ENGINE.md` §4.1: buffer-only storage,
/// `storageBuffer8BitAccess` is not part of the baseline capability set), so both map to `uint`
/// and the template is responsible for the pack/unpack.
pub const fn dtype_glsl(d: DType) -> &'static str {
    match d {
        DType::F32 => "float",
        DType::F16 => "float16_t",
        DType::I64 => "int64_t",
        DType::I32 => "int",
        DType::U8 => "uint",
        DType::Bool => "uint",
    }
}

/// Which shader backs one registry row, and what all its dtype variants are called.
///
/// Constructed only through [`kernel!`], which is what guarantees the stems obey the naming rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Kernel {
    /// The GLSL template family.
    pub template: Template,
    /// The template's op selector, e.g. `add`. Becomes `-DEW_OP=OP_ADD`.
    pub op: &'static str,
    /// Variant stems indexed by [`dtype_index`]. Only entries whose dtype is in the row's `caps`
    /// are ever generated or dispatched.
    pub stems: [&'static str; DTYPE_COUNT],
}

impl Kernel {
    /// The metadata-only kernel.
    pub const NONE: Kernel = Kernel {
        template: Template::None,
        op: "",
        stems: [""; DTYPE_COUNT],
    };

    /// The SPIR-V module stem for this op at this dtype.
    ///
    /// Returns `None` for [`Template::None`]. The caller is responsible for having checked that
    /// the dtype is in the row's `caps` — the claim predicate does exactly that, which is why
    /// translate handlers may treat a `Some` here as a guarantee that the module exists.
    pub fn stem(&self, d: DType) -> Option<&'static str> {
        if self.template == Template::None {
            return None;
        }
        Some(self.stems[dtype_index(d)])
    }

    /// `-D` defines the build must pass to compile this variant.
    pub fn defines(&self, d: DType) -> Vec<String> {
        if self.template == Template::None {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(3);
        // `EW_OP` is the elementwise templates' op selector and means nothing outside them. A
        // hand-written kernel is one op, so its identity is the stem, not a define — emitting a
        // spurious `EW_OP` there would compile but would quietly tie the kernel to `op_codes.glsl`.
        if self.template.is_elementwise() {
            out.push(format!("EW_OP=OP_{}", self.op.to_uppercase()));
        }
        out.push(format!("SCALAR_T={}", dtype_glsl(d)));
        out.push(format!("DTYPE_{}", dtype_suffix(d).to_uppercase()));
        out
    }
}

/// Build the `[&'static str; DTYPE_COUNT]` stem array for one template/op pair.
///
/// Order must match [`crate::ops::common::dtype::ALL_DTYPES`]; a test asserts it does.
#[macro_export]
macro_rules! stems {
    ($prefix:literal, $op:literal) => {
        [
            ::core::concat!($prefix, "_", $op, "_f32"),
            ::core::concat!($prefix, "_", $op, "_f16"),
            ::core::concat!($prefix, "_", $op, "_i64"),
            ::core::concat!($prefix, "_", $op, "_i32"),
            ::core::concat!($prefix, "_", $op, "_u8"),
            ::core::concat!($prefix, "_", $op, "_bool"),
        ]
    };
}

/// Declare the [`Kernel`] for a registry row.
///
/// ```ignore
/// kernel!(EwBinary, "add")   // -> ew_binary_add_f32, ew_binary_add_f16, ...
/// kernel!(None)              // metadata-only row
/// ```
#[macro_export]
macro_rules! kernel {
    (None) => {
        $crate::ops::common::variants::Kernel::NONE
    };
    (EwUnary, $op:literal) => {
        $crate::ops::common::variants::Kernel {
            template: $crate::ops::common::variants::Template::EwUnary,
            op: $op,
            stems: $crate::stems!("ew_unary", $op),
        }
    };
    (EwBinary, $op:literal) => {
        $crate::ops::common::variants::Kernel {
            template: $crate::ops::common::variants::Template::EwBinary,
            op: $op,
            stems: $crate::stems!("ew_binary", $op),
        }
    };
    (EwSelect, $op:literal) => {
        $crate::ops::common::variants::Kernel {
            template: $crate::ops::common::variants::Template::EwSelect,
            op: $op,
            stems: $crate::stems!("ew_select", $op),
        }
    };
    (QGemv, $op:literal) => {
        $crate::ops::common::variants::Kernel {
            template: $crate::ops::common::variants::Template::QGemv,
            op: $op,
            stems: $crate::stems!("q_gemv", $op),
        }
    };
}

/// One SPIR-V module the build is expected to produce.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct VariantSpec {
    /// Module stem, the key `engine::shaders::find` looks up.
    pub stem: &'static str,
    /// GLSL source file relative to `shaders/glsl/`.
    pub source: String,
    /// `-D` defines, in a stable order.
    pub defines: Vec<String>,
}

/// Every shader variant the registry implies, deduplicated and sorted.
///
/// **This is the contract with `build.rs`.** Two rows may share a kernel (`Sub` and `Add` do not,
/// but `Identity`-shaped rows will), so the list is deduplicated on the stem.
pub fn manifest() -> Vec<VariantSpec> {
    let mut out: Vec<VariantSpec> = Vec::new();
    for spec in crate::registry::all_specs() {
        let Some(source) = spec.kernel.template.source_file() else {
            continue;
        };
        for d in spec.caps.iter() {
            let Some(stem) = spec.kernel.stem(d) else {
                continue;
            };
            out.push(VariantSpec {
                stem,
                source: source.clone(),
                defines: spec.kernel.defines(d),
            });
        }
    }
    out.sort();
    out.dedup_by(|a, b| a.stem == b.stem);
    out
}

/// The manifest rendered as the tab-separated file `build.rs` reads.
///
/// Format, one variant per line: `stem \t source \t define,define,...`. Comment lines start with
/// `#`. Deliberately boring: `build.rs` must be able to parse it without depending on this crate.
pub fn manifest_text() -> String {
    let mut s = String::from(
        "# Generated from the op registry by `cargo test -p onnxruntime-ep-vulkan`.\n\
         # Do not edit by hand: add a row to an op table instead.\n\
         # Format: <spirv-stem>\\t<glsl-source>\\t<comma-separated -D defines>\n",
    );
    for v in manifest() {
        s.push_str(v.stem);
        s.push('\t');
        s.push_str(&v.source);
        s.push('\t');
        s.push_str(&v.defines.join(","));
        s.push('\n');
    }
    s
}

/// Path of the checked-in manifest, relative to the crate root.
pub const MANIFEST_PATH: &str = "src/ops/shader_variants.txt";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::common::dtype::ALL_DTYPES;

    #[test]
    fn stem_order_matches_dtype_order() {
        let k = kernel!(EwBinary, "add");
        for d in ALL_DTYPES {
            let expected = format!("ew_binary_add_{}", dtype_suffix(d));
            assert_eq!(k.stem(d).unwrap(), expected, "stem order drifted for {d:?}");
        }
    }

    #[test]
    fn none_kernel_has_no_shader() {
        let k = kernel!(None);
        assert_eq!(k.template, Template::None);
        for d in ALL_DTYPES {
            assert!(k.stem(d).is_none());
            assert!(k.defines(d).is_empty());
        }
        assert!(Template::None.source_file().is_none());
    }

    #[test]
    fn template_prefixes_are_unique() {
        let mut p: Vec<&str> = Template::ALL.iter().map(|t| t.prefix()).collect();
        p.sort_unstable();
        let before = p.len();
        p.dedup();
        assert_eq!(before, p.len());
    }

    #[test]
    fn defines_are_deterministic() {
        let k = kernel!(EwUnary, "sqrt");
        assert_eq!(
            k.defines(DType::F16),
            vec![
                "EW_OP=OP_SQRT".to_string(),
                "SCALAR_T=float16_t".to_string(),
                "DTYPE_F16".to_string(),
            ]
        );
    }

    #[test]
    fn manifest_covers_every_capped_dtype_of_every_shader_row() {
        let m = manifest();
        for spec in crate::registry::all_specs() {
            if spec.kernel.template == Template::None {
                continue;
            }
            for d in spec.caps.iter() {
                let stem = spec.kernel.stem(d).expect("shader row has a stem");
                assert!(
                    m.iter().any(|v| v.stem == stem),
                    "{} @ {d:?} claims `{stem}` but it is not in the build manifest",
                    spec.op_type
                );
            }
        }
    }

    #[test]
    fn manifest_stems_are_unique_and_well_formed() {
        let m = manifest();
        assert!(!m.is_empty(), "the registry implies no shaders at all");
        let mut stems: Vec<&str> = m.iter().map(|v| v.stem).collect();
        stems.sort_unstable();
        let before = stems.len();
        stems.dedup();
        assert_eq!(before, stems.len(), "duplicate stem in the build manifest");

        for v in &m {
            assert!(
                Template::ALL.iter().any(|t| v.stem.starts_with(t.prefix())),
                "`{}` does not start with a known template prefix",
                v.stem
            );
            assert!(
                !v.stem.contains(' ') && v.stem.is_ascii(),
                "`{}` is not a usable file stem",
                v.stem
            );
            assert!(v.source.ends_with(".comp"));
            // Elementwise variants carry `EW_OP`, `SCALAR_T` and `DTYPE_*`; a hand-written kernel
            // carries the last two only. The count is asserted rather than the contents so that
            // adding a define to one family without the other is a failure here.
            let want = if v.stem.starts_with(Template::QGemv.prefix()) {
                2
            } else {
                3
            };
            assert_eq!(v.defines.len(), want, "wrong define count for `{}`", v.stem);
        }
    }

    /// The build's view of what shaders exist and the registry's view must not drift.
    ///
    /// Regenerate with `MOUSE_BLESS_VARIANTS=1 cargo test -p onnxruntime-ep-vulkan variants`.
    #[test]
    fn checked_in_manifest_is_in_sync() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(MANIFEST_PATH);
        let want = manifest_text();
        if std::env::var_os("MOUSE_BLESS_VARIANTS").is_some() {
            std::fs::write(&path, &want).expect("write shader variant manifest");
            return;
        }
        let have = std::fs::read_to_string(&path).unwrap_or_default();
        assert_eq!(
            have.replace("\r\n", "\n"),
            want,
            "{MANIFEST_PATH} is stale; regenerate with MOUSE_BLESS_VARIANTS=1"
        );
    }

    fn shaders_dir() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("shaders")
    }

    /// Every template the registry names must have a source file `build.rs` can find.
    ///
    /// Without this, a row can reference a template that does not exist and the failure surfaces
    /// as a `build.rs` panic on someone else's machine — probably in CI, probably at the worst
    /// time. Here it is a unit-test failure in the change that caused it.
    #[test]
    fn every_template_has_its_glsl_source_on_disk() {
        let glsl = shaders_dir().join("glsl");
        for t in Template::ALL {
            let file = t.source_file().expect("templates in ALL have a source");
            assert!(
                glsl.join(&file).is_file(),
                "template {t:?} names `{file}`, which does not exist in shaders/glsl/"
            );
        }
    }

    /// Every op selector in the registry must have an `OP_<NAME>` code in the GLSL header.
    ///
    /// This is the seam that a table-driven design makes easy to break: a new row's `op` string
    /// only becomes `-DEW_OP=OP_FOO`, and if the header has no `OP_FOO` the shader fails to
    /// compile with a preprocessor error that names neither the op nor the row. Checking it here
    /// turns that into a message that names both, without needing a shader compiler.
    #[test]
    fn every_op_selector_has_a_glsl_code() {
        let header = std::fs::read_to_string(shaders_dir().join("include").join("op_codes.glsl"))
            .expect("shaders/include/op_codes.glsl must exist");
        for spec in crate::registry::all_specs() {
            // Only the elementwise families use an op selector; a hand-written kernel's identity
            // is its stem, so there is no `OP_*` symbol to look for.
            if !spec.kernel.template.is_elementwise() {
                continue;
            }
            let symbol = format!("OP_{}", spec.kernel.op.to_uppercase());
            assert!(
                header.contains(&format!("#define {symbol} ")),
                "`{}` dispatches -DEW_OP={symbol}, which op_codes.glsl does not define",
                spec.op_type
            );
        }
    }

    /// The templates must handle every `(op, dtype)` pair the manifest asks them to compile.
    ///
    /// A cheap structural proxy for "it compiles": the template source has to mention the op code
    /// at all. It does not prove the expression is right — only a differential run on a device
    /// does that — but it catches the common case of adding a row and forgetting the shader case,
    /// and it runs on a machine with no Vulkan SDK, which is every machine in this project today.
    #[test]
    fn every_template_mentions_every_op_it_must_compile() {
        let glsl = shaders_dir().join("glsl");
        let mut sources: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();
        for v in manifest() {
            let src = sources.entry(v.source.clone()).or_insert_with(|| {
                std::fs::read_to_string(glsl.join(&v.source))
                    .unwrap_or_else(|e| panic!("cannot read shaders/glsl/{}: {e}", v.source))
            });
            let Some(symbol) = v
                .defines
                .iter()
                .find(|d| d.starts_with("EW_OP="))
                .map(|d| d.trim_start_matches("EW_OP=").to_string())
            else {
                // A hand-written kernel has no op selector to check for. It still has to exist
                // and be readable, which the `entry` above just proved.
                continue;
            };
            assert!(
                src.contains(&format!("EW_OP == {symbol}")),
                "{} must handle {symbol} (variant `{}`) and does not",
                v.source,
                v.stem
            );
        }
    }
}
