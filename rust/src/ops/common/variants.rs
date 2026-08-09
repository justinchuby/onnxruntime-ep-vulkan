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
    /// One input, one output, shape-preserving, **whose output element type differs from its
    /// input's**. `Cast`.
    ///
    /// The only template whose variant space is a dtype **pair**. Every other row picks one
    /// `DTYPE_*` and both its buffers follow; here the two buffers have independent storage
    /// types, so the source and destination halves of `indexing.glsl`'s type mapping cannot both
    /// come from one define. That is a different *shape* of variant table, not a bigger one,
    /// which is why it is a template rather than another `EwUnary` op selector.
    EwCast,
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
            Template::EwCast => "ew_cast",
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
            Template::EwCast => 1,
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

    /// Whether this template's variant space is keyed on a **dtype pair** rather than one dtype.
    ///
    /// Named once here because three separate things branch on it — the manifest's cross product,
    /// the stem lookup, and the define set — and re-deriving it by matching on the variant at each
    /// site is how the three would drift apart.
    pub const fn is_pair_keyed(self) -> bool {
        matches!(self, Template::EwCast)
    }

    /// Every template that has a source file.
    pub const ALL: &'static [Template] = &[
        Template::EwUnary,
        Template::EwBinary,
        Template::EwSelect,
        Template::QGemv,
        Template::EwCast,
    ];
}

/// The GLSL scalar type a dtype maps to inside a template.
///
/// `u8`, `bool` **and `f16`** are stored packed in `uint` buffers (`ENGINE.md` §4.1: buffer-only
/// storage; neither `storageBuffer8BitAccess` nor `storageBuffer16BitAccess` is in the baseline
/// capability set), so all three map to `uint` and the template is responsible for the
/// pack/unpack. Declaring f16 as `float16_t` instead makes the SPIR-V require a device feature the
/// engine does not enable — see `indexing.glsl`'s header.
pub const fn dtype_glsl(d: DType) -> &'static str {
    match d {
        DType::F32 => "float",
        DType::F16 => "uint",
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
    /// Variant stems indexed by `[source][destination]`, for pair-keyed templates only.
    ///
    /// `Some` exactly when `template.is_pair_keyed()`; a test asserts the two agree. It is a
    /// second field rather than a widening of `stems` because 34 of the 36 rows a pair table
    /// holds are meaningless for every other template, and a single field would have made every
    /// row's stem lookup ask which kind it was.
    pub pair_stems: Option<&'static [[&'static str; DTYPE_COUNT]; DTYPE_COUNT]>,
    /// Stems of the **hand-written** modules this row's `translate` dispatches, indexed by
    /// [`dtype_index`], for rows that have no generated template family.
    ///
    /// # Why this exists
    ///
    /// `Template::None` was read by everything as *"this row has no shader"*, and
    /// [`Kernel::stem`] answered `None` on that basis, which made `registry::variant_key` render
    /// the key's variant component as the literal `metadata`. That was true for a shape op
    /// handled on the host and false for every row here: `Conv` records `"shaders":["conv_f32"]`
    /// in its own ledger entries while keying its variant as *"this row has no shader"*.
    ///
    /// Two things followed, and Morpheus found both (§8.9.23):
    ///
    /// * the variant component was **constant across every future form of the op**, so a second
    ///   `Conv` variant would have shared a key component with the first — invisible while there
    ///   is only one;
    /// * `registry::form_is_provable` **short-circuited**, because `metadata` names no module and
    ///   the predicate under-claims on unknown stems. So `Conv` read *provable* in a build with
    ///   no SPIR-V at all — the exact positive control that predicate was built to have.
    ///
    /// The module is chosen in `translate`, where the key does not look. This field is how the
    /// row declares it where the key **does** look, without inventing a template family for a
    /// hand-written `.comp` that `build.rs` already compiles from `shaders/glsl` directly.
    ///
    /// `Template::None` therefore means *"no generated variant family"*, not *"no module"*. A row
    /// with neither a template nor this field is genuinely metadata-only.
    pub module_stems: Option<&'static [&'static str; DTYPE_COUNT]>,
    /// Stems of **additional** hand-written modules this row's `translate` handler may dispatch
    /// *instead of* [`Kernel::module_stems`]'s answer, for the same dtype, under a runtime
    /// condition `translate` alone decides (e.g. a shape it can see at graph-compile time).
    ///
    /// # Why this exists
    ///
    /// [`Kernel::stem`] — and therefore the ledger key's variant component
    /// (`registry::variant_key`) — names exactly one module per dtype: the row's *declared*
    /// identity, unconditionally. `GroupQueryAttention` (#90) is the first row whose `translate`
    /// dispatches a **second** hand-written `.comp` for the same dtype depending on a shape
    /// (`gqa_decode_f16` at `seq_len == 1`, `gqa_f16` otherwise) — deliberately, so the ledger key
    /// stays `gqa_f16` and does not fork per shape the way a second *op* would. That leaves the
    /// second module invisible to [`Kernel::stem`], which is correct for the key, but would also
    /// make it invisible to `every_hand_written_shader_is_named_by_a_row` — the test that exists
    /// specifically to catch a `.comp` nobody's row names, i.e. exactly the defect class this
    /// field prevents reintroducing through the back door of a second shader on an old row.
    ///
    /// `stem()` deliberately does not consult this field: it answers the declared/keying
    /// question, not the dispatch question, and must keep answering it the same way regardless of
    /// how many extra modules a `translate` handler grows.
    pub extra_module_stems: Option<&'static [&'static str]>,
}

impl Kernel {
    /// The metadata-only kernel.
    pub const NONE: Kernel = Kernel {
        template: Template::None,
        op: "",
        stems: [""; DTYPE_COUNT],
        pair_stems: None,
        module_stems: None,
        extra_module_stems: None,
    };

    /// The SPIR-V module stem for this op at this dtype.
    ///
    /// Returns `None` for a pair-keyed template, which has no answer to this question — ask
    /// [`Kernel::pair_stem`] instead. Returning the source-dtype row's first entry would have
    /// been a plausible-looking wrong answer. Returns `None` for a row that is genuinely
    /// metadata-only; a `Template::None` row that declares [`Kernel::module_stems`] answers with
    /// the hand-written module its `translate` dispatches.
    pub fn stem(&self, d: DType) -> Option<&'static str> {
        if self.template.is_pair_keyed() {
            return None;
        }
        if self.template == Template::None {
            return self.module_stems.map(|s| s[dtype_index(d)]);
        }
        Some(self.stems[dtype_index(d)])
    }

    /// The additional hand-written modules [`Kernel::extra_module_stems`] declares, or `&[]`.
    ///
    /// Not consulted by [`Kernel::stem`] or by anything that keys a proof — see the field's own
    /// doc comment for why. This exists only for the build-coverage side: so a scan over "every
    /// `.comp` this row is allowed to dispatch" can see the extra one too.
    pub fn extra_stems(&self) -> &'static [&'static str] {
        self.extra_module_stems.unwrap_or(&[])
    }

    /// The SPIR-V module stem for a source/destination dtype pair.
    pub fn pair_stem(&self, src: DType, dst: DType) -> Option<&'static str> {
        Some(self.pair_stems?[dtype_index(src)][dtype_index(dst)])
    }

    /// `-D` defines the build must pass to compile this variant.
    pub fn defines(&self, d: DType) -> Vec<String> {
        if self.template == Template::None || self.template.is_pair_keyed() {
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

    /// `-D` defines for one pair-keyed variant.
    ///
    /// The source half is byte-identical to what every other template gets, so `indexing.glsl`
    /// maps `COMPUTE_T` and the load accessors exactly as it always has. The destination half is
    /// carried in its own namespace (`CAST_DST_*`, `DST_SCALAR_T`) and read only by
    /// `ew_cast.comp`, because a second `DTYPE_*` would make the shared header choose between two
    /// answers for `COMPUTE_T`.
    pub fn defines_pair(&self, src: DType, dst: DType) -> Vec<String> {
        if !self.template.is_pair_keyed() {
            return Vec::new();
        }
        vec![
            format!("SCALAR_T={}", dtype_glsl(src)),
            format!("DTYPE_{}", dtype_suffix(src).to_uppercase()),
            format!("DST_SCALAR_T={}", dtype_glsl(dst)),
            format!("CAST_DST_{}", dtype_suffix(dst).to_uppercase()),
        ]
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

/// Build the `[[&'static str; DTYPE_COUNT]; DTYPE_COUNT]` stem matrix for a pair-keyed template.
///
/// Rows are the **source** dtype, columns the **destination**, both in [`ALL_DTYPES`] order —
/// which is the same order [`stems!`] uses, and a test asserts it here too. Written as a nested
/// macro rather than generated at runtime for the same reason `stems!` is: `KernelRequest::shader`
/// is `&'static str` and may never be formatted while a graph is being compiled.
///
/// [`ALL_DTYPES`]: crate::ops::common::dtype::ALL_DTYPES
#[macro_export]
macro_rules! pair_stems {
    ($prefix:literal) => {
        $crate::pair_stems!(@rows $prefix, "f32", "f16", "i64", "i32", "u8", "bool")
    };
    (@rows $prefix:literal, $($src:literal),+) => {
        [$($crate::pair_stems!(@row $prefix, $src)),+]
    };
    (@row $prefix:literal, $src:literal) => {
        [
            ::core::concat!($prefix, "_", $src, "_to_f32"),
            ::core::concat!($prefix, "_", $src, "_to_f16"),
            ::core::concat!($prefix, "_", $src, "_to_i64"),
            ::core::concat!($prefix, "_", $src, "_to_i32"),
            ::core::concat!($prefix, "_", $src, "_to_u8"),
            ::core::concat!($prefix, "_", $src, "_to_bool"),
        ]
    };
}

/// Build the `[&'static str; DTYPE_COUNT]` stem array for a **hand-written** module family.
///
/// One `.comp` per dtype, named `<prefix>_<dtype>` — the convention every hand-written kernel in
/// `shaders/glsl` already follows (`conv_f32.comp`, `gather_f16.comp`,
/// `skip_simplified_layer_norm_f32.comp`). Order must match [`stems!`]'s; the same test asserts it.
///
/// Only the dtypes in the row's `caps` are ever dispatched, so an entry naming a `.comp` that does
/// not exist is unreachable — and it is exactly what [`crate::registry`]'s provability predicate
/// must read as *not loadable* rather than as *unknown*.
#[macro_export]
macro_rules! module_stems {
    ($prefix:literal) => {
        [
            ::core::concat!($prefix, "_f32"),
            ::core::concat!($prefix, "_f16"),
            ::core::concat!($prefix, "_i64"),
            ::core::concat!($prefix, "_i32"),
            ::core::concat!($prefix, "_u8"),
            ::core::concat!($prefix, "_bool"),
        ]
    };
}

/// Declare the [`Kernel`] for a registry row.
///
/// ```ignore
/// kernel!(EwBinary, "add")   // -> ew_binary_add_f32, ew_binary_add_f16, ...
/// kernel!(Standalone, "conv")// -> hand-written conv_f32.comp, conv_f16.comp, ...
/// kernel!(Standalone, "gqa", extra: ["gqa_decode_f16"]) // conditional 2nd module, same dtype
/// kernel!(None)              // metadata-only row: no module at all
/// ```
#[macro_export]
macro_rules! kernel {
    (None) => {
        $crate::ops::common::variants::Kernel::NONE
    };
    (Standalone, $prefix:literal) => {
        $crate::ops::common::variants::Kernel {
            template: $crate::ops::common::variants::Template::None,
            op: $prefix,
            stems: [""; $crate::ops::common::dtype::DTYPE_COUNT],
            pair_stems: None,
            module_stems: ::core::option::Option::Some(&$crate::module_stems!($prefix)),
            extra_module_stems: None,
        }
    };
    (Standalone, $prefix:literal, extra: [$($extra:literal),+ $(,)?]) => {
        $crate::ops::common::variants::Kernel {
            template: $crate::ops::common::variants::Template::None,
            op: $prefix,
            stems: [""; $crate::ops::common::dtype::DTYPE_COUNT],
            pair_stems: None,
            module_stems: ::core::option::Option::Some(&$crate::module_stems!($prefix)),
            extra_module_stems: ::core::option::Option::Some(&[$($extra),+]),
        }
    };
    (EwUnary, $op:literal) => {
        $crate::ops::common::variants::Kernel {
            template: $crate::ops::common::variants::Template::EwUnary,
            op: $op,
            stems: $crate::stems!("ew_unary", $op),
            pair_stems: None,
            module_stems: None,
            extra_module_stems: None,
        }
    };
    (EwBinary, $op:literal) => {
        $crate::ops::common::variants::Kernel {
            template: $crate::ops::common::variants::Template::EwBinary,
            op: $op,
            stems: $crate::stems!("ew_binary", $op),
            pair_stems: None,
            module_stems: None,
            extra_module_stems: None,
        }
    };
    (EwSelect, $op:literal) => {
        $crate::ops::common::variants::Kernel {
            template: $crate::ops::common::variants::Template::EwSelect,
            op: $op,
            stems: $crate::stems!("ew_select", $op),
            pair_stems: None,
            module_stems: None,
            extra_module_stems: None,
        }
    };
    (QGemv, $op:literal) => {
        $crate::ops::common::variants::Kernel {
            template: $crate::ops::common::variants::Template::QGemv,
            op: $op,
            stems: $crate::stems!("q_gemv", $op),
            pair_stems: None,
            module_stems: None,
            extra_module_stems: None,
        }
    };
    (EwCast, $op:literal) => {
        $crate::ops::common::variants::Kernel {
            template: $crate::ops::common::variants::Template::EwCast,
            op: $op,
            // The single-dtype array is unreachable for a pair-keyed row (`stem` returns `None`),
            // but it is filled with the pair table's diagonal rather than with `""` so that a
            // future reader who prints it sees something true.
            stems: $crate::stems!("ew_cast", "x"),
            pair_stems: ::core::option::Option::Some(&$crate::pair_stems!("ew_cast")),
            module_stems: None,
            extra_module_stems: None,
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
        if spec.kernel.template.is_pair_keyed() {
            // The cross product, including the diagonal. `Cast` to the same type is a legal ONNX
            // node and a real exporter emits it; leaving the diagonal out would put a hole in the
            // middle of the table that only shows up as a decline on a graph nobody tested.
            for src in spec.caps.iter() {
                for dst in spec.caps.iter() {
                    let Some(stem) = spec.kernel.pair_stem(src, dst) else {
                        continue;
                    };
                    out.push(VariantSpec {
                        stem,
                        source: source.clone(),
                        defines: spec.kernel.defines_pair(src, dst),
                    });
                }
            }
            continue;
        }
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

// ---------------------------------------------------------------------------
// SPIR-V capability accounting — what we may build vs what we may claim on
// ---------------------------------------------------------------------------
//
// A SPIR-V `OpCapability` is the module-level name for "this needs a device feature". Vulkan does
// not grant features by being new enough: a feature must be *enabled* in `VkDeviceCreateInfo`
// before a module declaring it can be created, whatever the device supports. So there are two
// different sets here and conflating them is exactly the mistake that made every f16 module
// unloadable for as long as they existed.

/// `Shader` — declared by every compute module. Never optional.
pub const CAP_SHADER: u32 = 1;
/// `Int64` — declared by every `_i64` variant. Requires `VkPhysicalDeviceFeatures::shaderInt64`,
/// which the engine's feature chain does **not** currently enable.
pub const CAP_INT64: u32 = 11;

/// Capabilities a *generated* variant may declare.
///
/// Wider than [`ENGINE_ENABLED_CAPABILITIES`] on purpose: building a variant no device can load
/// costs a few kilobytes and nothing else, and the i64 variants have to exist before the feature
/// that would make them loadable is worth adding. Building one is not the bug. Claiming on one is.
#[cfg(test)]
pub(crate) const GENERATED_CAPABILITIES: &[u32] = &[CAP_SHADER, CAP_INT64];

/// Capabilities the engine actually enables at device creation, and therefore the only ones a
/// **live claim** may rest on.
///
/// `vk::device` builds `VkDeviceCreateInfo` with a `DeviceFeatureChain` carrying only
/// `synchronization2` and passes no `pEnabledFeatures` at all — so `shaderInt64` is off, and an
/// `_i64` module cannot be created on any device we run on. Widening this list means three edits
/// together, not one: enable the feature in the chain, probe it in `vk::caps`, and decline the
/// variant on devices that lack it. A capability we generate is not a capability we have.
pub const ENGINE_ENABLED_CAPABILITIES: &[u32] = &[CAP_SHADER];

/// Every `OpCapability` declared by a SPIR-V module, decoded from the binary.
///
/// Word 0 of an instruction packs `(word_count << 16) | opcode`; `OpCapability` is opcode 17 with
/// a single operand. The five-word header is skipped. Capabilities must precede every other
/// section, but scanning the whole module is simpler and cannot miss one.
#[cfg(test)]
pub(crate) fn declared_capabilities(spv: &[u8]) -> Vec<u32> {
    declared_capabilities_impl(spv)
}

fn declared_capabilities_impl(spv: &[u8]) -> Vec<u32> {
    let words: Vec<u32> = spv
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let mut out = Vec::new();
    let mut i = 5;
    while i < words.len() {
        let len = (words[i] >> 16) as usize;
        if len == 0 {
            break;
        }
        if words[i] & 0xFFFF == 17 && i + 1 < words.len() {
            out.push(words[i + 1]);
        }
        i += len;
    }
    out
}

/// Can the engine actually create a pipeline from this SPIR-V module on the devices we run on?
///
/// §8.9.16 — THE HALF OF `EXERCISED` THAT WAS REAL.
/// `elementwise::EXERCISED` used to answer two questions with one hand-written list: *does a
/// kernel exist for this (op, dtype)?* and *has anything ever measured it?* The second question
/// is the proof ledger's, and answering it in the claim predicate created a deadlock — the veto
/// fired before a proof key was computed, so no proof run could ever reach the forms it blocked
/// (`Add`/i32, `Mul`/i32, `Swish`/f32). The first question is real and stays here, derived from
/// the artifact rather than asserted: a module that declares a capability the engine never
/// enables cannot be created on any device we run on, and claiming a node it would serve is a
/// promise we cannot keep whatever the ledger says.
///
/// Did the build generate a module under this stem at all?
///
/// [`variant_is_loadable`] answers `false` for both "no such module" and "the module declares a
/// capability we do not enable", which is right for its question and wrong for any caller that
/// wants to know *why*. A composite row's key carries `metadata` in the variant slot — no module
/// has ever been named that — so a caller reading loadability off a proof key must be able to tell
/// a placeholder from a refusal.
pub fn variant_is_generated(stem: &str) -> bool {
    crate::engine::shaders::SHADER_MODULES
        .iter()
        .any(|(name, _)| *name == stem)
}

/// Does **the registry** name a module under this stem?
///
/// The distinction from [`variant_is_generated`] is the whole point: `variant_is_generated` asks
/// the *build output*, so it answers `false` in a build that produced no SPIR-V — and a caller
/// that reads `false` as "unknown stem, assume the best" then under-claims in exactly the build
/// where it should refuse. This asks the *registry*, which is checked-in source and answers the
/// same in every build.
///
/// So the pair separates three states a single boolean was collapsing:
///
/// | declared | generated | meaning |
/// |---|---|---|
/// | yes | yes | a real module; loadability is the remaining question |
/// | yes | no  | **the build is missing a module a row dispatches** — a shaderless build |
/// | no  | —   | not a module name at all (`metadata`, a malformed key) — nothing is known |
///
/// Derived from the registry rather than listed, and computed once: it runs on the claim path.
pub fn variant_is_declared(stem: &str) -> bool {
    static DECLARED: std::sync::OnceLock<std::collections::BTreeSet<&'static str>> =
        std::sync::OnceLock::new();
    DECLARED
        .get_or_init(|| {
            let mut set = std::collections::BTreeSet::new();
            for spec in crate::registry::all_specs() {
                for d in spec.caps.iter() {
                    if let Some(s) = spec.kernel.stem(d) {
                        if !s.is_empty() {
                            set.insert(s);
                        }
                    }
                    if spec.kernel.template.is_pair_keyed() {
                        for dst in spec.caps.iter() {
                            if let Some(s) = spec.kernel.pair_stem(d, dst) {
                                set.insert(s);
                            }
                        }
                    }
                }
            }
            set
        })
        .contains(stem)
}

/// `false` for an unknown stem: a variant the build did not generate is not loadable either.
pub fn variant_is_loadable(stem: &str) -> bool {
    let Some((_, spv)) = crate::engine::shaders::SHADER_MODULES
        .iter()
        .find(|(name, _)| *name == stem)
    else {
        return false;
    };
    declared_capabilities_impl(spv)
        .iter()
        .all(|c| ENGINE_ENABLED_CAPABILITIES.contains(c))
}

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

    /// f16's `SCALAR_T` is `uint`, not `float16_t`, and that is load-bearing rather than
    /// cosmetic: `float16_t` storage makes the SPIR-V declare `StorageBuffer16BitAccess`, a device
    /// feature the engine's `VkDeviceCreateInfo` chain does not enable, so every f16 module was
    /// unloadable. Pinned here so the packing decision cannot be reverted by a tidy-up.
    #[test]
    fn f16_is_packed_into_uint_words_rather_than_typed() {
        let k = kernel!(EwUnary, "sqrt");
        assert_eq!(
            k.defines(DType::F16),
            vec![
                "EW_OP=OP_SQRT".to_string(),
                "SCALAR_T=uint".to_string(),
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
            if spec.kernel.template.is_pair_keyed() {
                for src in spec.caps.iter() {
                    for dst in spec.caps.iter() {
                        let stem = spec
                            .kernel
                            .pair_stem(src, dst)
                            .expect("pair row has a stem");
                        assert!(
                            m.iter().any(|v| v.stem == stem),
                            "{} @ {src:?}->{dst:?} claims `{stem}` but it is not in the build \
                             manifest",
                            spec.op_type
                        );
                    }
                }
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

    /// The pair table's rows are sources and its columns destinations, both in `ALL_DTYPES` order.
    ///
    /// Transposing it would compile, would generate exactly the same 36 modules, and would then
    /// dispatch `f32 -> i32` nodes to the module that reads ints and writes floats. Nothing else
    /// in the system can catch that: the stem exists, the module loads, and the answer is wrong.
    #[test]
    fn pair_stem_table_is_source_major() {
        let k = kernel!(EwCast, "cast");
        for src in ALL_DTYPES {
            for dst in ALL_DTYPES {
                assert_eq!(
                    k.pair_stem(src, dst).unwrap(),
                    format!("ew_cast_{}_to_{}", dtype_suffix(src), dtype_suffix(dst)),
                    "pair stem table is not source-major at {src:?}->{dst:?}"
                );
            }
        }
        assert!(
            k.stem(DType::F32).is_none(),
            "a pair row has no single stem"
        );
        assert!(k.defines(DType::F32).is_empty());
    }

    /// `pair_stems` is present exactly when the template says the row is pair-keyed.
    #[test]
    fn pair_tables_and_pair_keyed_templates_agree() {
        for spec in crate::registry::all_specs() {
            assert_eq!(
                spec.kernel.pair_stems.is_some(),
                spec.kernel.template.is_pair_keyed(),
                "`{}` disagrees with its template about being pair-keyed",
                spec.op_type
            );
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
            // carries the last two only; a pair-keyed variant carries a source and a destination
            // half. The count is asserted rather than the contents so that adding a define to one
            // family without the other is a failure here.
            let want = if v.stem.starts_with(Template::EwCast.prefix()) {
                4
            } else if v.stem.starts_with(Template::QGemv.prefix()) {
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

    /// No embedded shader may require a device feature the engine does not enable.
    ///
    /// This is the general form of a bug that reached two devices undetected: the f16 elementwise
    /// variants were compiled with `SCALAR_T = float16_t`, which made every one of them declare
    /// `StorageBuffer16BitAccess` (capability 4433). The engine's `VkDeviceCreateInfo` feature
    /// chain carries only `synchronization2`, so those modules could never have been loaded — yet
    /// nothing failed, because no test had ever asked a device to load one. The census reported
    /// the resulting nodes as declining on `[dtype]`, which was true and completely uninformative.
    ///
    /// Asserting on the *capability set* rather than on one shader is the point. A capability is
    /// exactly the SPIR-V-level name for "this module needs a feature", so an allowlist here is a
    /// checkable statement of the §7.2 baseline, and any future kernel that reaches for a device
    /// feature fails in CI on a developer's machine rather than on a user's phone.
    ///
    /// This list is what may be **generated**, which is deliberately wider than what may be
    /// **claimed**. Generating an unloadable variant is harmless; claiming on one is the bug. That
    /// second, narrower rule is
    /// [`elementwise::no_live_claim_rests_on_an_unloadable_variant`](crate::ops::elementwise).
    #[test]
    fn no_shader_requires_a_device_feature_the_engine_does_not_enable() {
        let mut offenders: Vec<String> = Vec::new();
        for (stem, bytes) in crate::engine::shaders::SHADER_MODULES {
            for cap in declared_capabilities(bytes) {
                if !GENERATED_CAPABILITIES.contains(&cap) {
                    offenders.push(format!("{stem} requires SPIR-V capability {cap}"));
                }
            }
        }
        assert!(
            offenders.is_empty(),
            "these modules need device features nothing in the engine enables or probes:\n  {}",
            offenders.join("\n  ")
        );
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
