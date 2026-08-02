//! The op registry and `NodeView` — where the ORT graph ABI is translated into safe Rust.
//!
//! Two responsibilities, and no third:
//!
//! 1. [`NodeView`] — a *read-only* borrow of an `OrtNode`. This is one of exactly two places in
//!    the crate where ORT ABI types are touched outside `ep.rs`/`factory.rs` (`DESIGN.md` §4.2
//!    calls it "the exception that proves the rule"), and it exists so that claim predicates can
//!    ask questions about a node without ever seeing an `OrtNode`.
//!
//! 2. The registry itself — a single table mapping a domain-qualified op name to an [`OpSpec`].
//!    `ep.rs` asks exactly one question per node, [`claim_decision`], and has **no per-op logic**.
//!    That invariant is what makes "claimed" and "translatable" impossible to desynchronize; it is
//!    inherited directly from the `onnxruntime-mlx` reference.
//!
//! # Table-driven, because 87 ops cannot be 87 hand-written functions
//!
//! `OP_COVERAGE.md` §5.6/§9 argues that broad coverage is only reachable if **adding an op is
//! adding a row**, not writing a module. So an [`OpSpec`] carries everything the machinery needs:
//!
//! * `domain` / `op_type` / `min_opset..=max_opset` — the key and its validity window.
//! * `caps: DTypeSet` — the dtypes this op supports. Shared claim predicates read it, so one
//!   predicate serves every elementwise op and the *row* decides the dtype policy.
//! * `kernel: Kernel` — which shader template and which template op, from which the variant table
//!   resolves a `&'static str` SPIR-V stem. No per-op shader plumbing.
//! * `claim` — the predicate. Almost always one of the shared ones in `ops::common::claim`.
//! * `translate` — the handler. Almost always one of the shared ones in `ops::common::templates`.
//! * `status` — [`OpStatus::Live`] or [`OpStatus::Staged`].
//!
//! # `Staged` rows, and why M0 still claims nothing
//!
//! The claim/translate invariant is absolute: *never claim what we cannot translate.* But the op
//! table is worth landing — and unit-testing — before the shaders behind it exist. So a row may be
//! [`OpStatus::Staged`]: it is fully described, its claim predicate is fully exercised by tests,
//! and [`claim_decision`] declines it with a machine-readable `[staged]` reason. Flipping an op on
//! once its shader lands is a one-word diff, and until then the graph runs on the CPU EP, which is
//! always correct.
//!
//! # Machine-readable declines
//!
//! `DeclineReason` stays `Cow<'static, str>` because `ep.rs` owns that seam and Tank's diagnostics
//! aggregate it verbatim. Machine-readability is achieved *by construction* instead: every reason
//! this module produces is built by [`decline`] and is prefixed with a canonical `[tag]` from
//! [`DeclineCode`]. [`DeclineCode::of_reason`] parses it back out, so Trinity's harness and
//! Niobe's census can bucket declines by cause without any ABI change.

use std::borrow::Cow;
use std::ffi::CStr;

use crate::engine::{DType, EpResult, NodeDesc};
use crate::ops::common::dtype::DTypeSet;
use crate::ops::common::variants::Kernel;
use crate::sys::{self, OrtRelease, SchemaBaseline, ort};

// -------------------------------------------------------------------------------------------
// NodeView — a read-only borrow of an OrtNode
// -------------------------------------------------------------------------------------------

/// What a claim predicate can learn about one node edge (input or output).
///
/// Both fields are optional on purpose. ORT reports a **null `OrtValueInfo` for an omitted
/// interior optional input** (`Clip(x, , max)`), and shape inference legitimately fails or
/// produces symbolic dimensions. A predicate that treats "unknown" as "fine" is the single most
/// common way to claim a node you cannot translate, so the type makes the unknown case impossible
/// to ignore.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EdgeType {
    /// Element type, or `None` if absent, non-tensor, or a type this EP has no `DType` for.
    pub dtype: Option<DType>,
    /// Declared shape, or `None` if shape inference produced nothing. `-1` entries are symbolic.
    pub shape: Option<Vec<i64>>,
}

impl EdgeType {
    /// Rank, if known.
    pub fn rank(&self) -> Option<usize> {
        self.shape.as_ref().map(Vec::len)
    }

    /// True when a shape is known *and* fully static (no symbolic `-1` dimensions).
    pub fn is_static(&self) -> bool {
        self.shape
            .as_ref()
            .is_some_and(|s| s.iter().all(|d| *d >= 0))
    }
}

/// A read-only view of one `OrtNode`, valid only for as long as ORT's graph is.
///
/// Deliberately borrow-only and `Copy`-free: it holds raw ORT pointers, so it must never outlive
/// the `GetCapability` / `Compile` call it was made in. Claim predicates receive `&NodeView` and
/// cannot store it (no `'static`), which is exactly the lifetime we want.
pub struct NodeView<'graph> {
    api: *const ort::OrtApi,
    node: *const ort::OrtNode,
    _graph: std::marker::PhantomData<&'graph ()>,
}

impl<'graph> NodeView<'graph> {
    /// Wrap a node ORT handed us.
    ///
    /// # Safety
    /// `api` must be a live `OrtApi`; `node` must be a node of a graph that outlives `'graph`.
    pub unsafe fn new(api: *const ort::OrtApi, node: *const ort::OrtNode) -> NodeView<'graph> {
        NodeView {
            api,
            node,
            _graph: std::marker::PhantomData,
        }
    }

    /// Call a `fn(node, *mut *const c_char) -> OrtStatus*` accessor and own the result.
    ///
    /// Returns the empty string on any failure. Every one of these accessors returns a pointer
    /// into ORT-owned graph storage, so the copy is required, not merely convenient.
    fn c_str_getter(
        &self,
        get: Option<
            unsafe extern "C" fn(
                *const ort::OrtNode,
                *mut *const std::ffi::c_char,
            ) -> ort::OrtStatusPtr,
        >,
    ) -> String {
        let Some(get) = get else {
            return String::new();
        };
        let mut out: *const std::ffi::c_char = std::ptr::null();
        // SAFETY: `self.node` is a live node and `out` is a valid out-parameter slot. On success
        // ORT writes a pointer to a NUL-terminated string it owns, which stays valid at least
        // until the graph is destroyed — longer than this call — so copying it here is sound. On
        // failure the status is owned by us and released immediately.
        unsafe {
            let status = get(self.node, &mut out);
            if !status.is_null() {
                sys::release_status(self.api, status);
                return String::new();
            }
            if out.is_null() {
                return String::new();
            }
            CStr::from_ptr(out).to_string_lossy().into_owned()
        }
    }

    /// ONNX op type, e.g. `Add`.
    pub fn op_type(&self) -> String {
        // SAFETY: `self.api` is a live `OrtApi` per the constructor's contract.
        let get = unsafe { (*self.api).Node_GetOperatorType };
        self.c_str_getter(get)
    }

    /// ONNX domain — empty or `ai.onnx` for the default domain.
    pub fn domain(&self) -> String {
        // SAFETY: as above.
        let get = unsafe { (*self.api).Node_GetDomain };
        self.c_str_getter(get)
    }

    /// The node's name in the model. May legitimately be empty.
    pub fn name(&self) -> String {
        // SAFETY: as above.
        let get = unsafe { (*self.api).Node_GetName };
        self.c_str_getter(get)
    }

    /// The opset version this node's op was resolved against. `0` if unavailable.
    pub fn since_version(&self) -> i32 {
        // SAFETY: `self.api` is live; `Node_GetSinceVersion` writes through a valid out-param, and
        // any returned status is owned by us and released here.
        unsafe {
            let Some(get) = (*self.api).Node_GetSinceVersion else {
                return 0;
            };
            let mut v: std::ffi::c_int = 0;
            let status = get(self.node, &mut v);
            if !status.is_null() {
                sys::release_status(self.api, status);
                return 0;
            }
            v
        }
    }

    /// Number of inputs, including optional ones ORT reports as present.
    pub fn num_inputs(&self) -> usize {
        // SAFETY: as above; `Node_GetNumInputs` writes a `usize` through a valid out-param.
        unsafe {
            let Some(get) = (*self.api).Node_GetNumInputs else {
                return 0;
            };
            let mut n: usize = 0;
            let status = get(self.node, &mut n);
            if !status.is_null() {
                sys::release_status(self.api, status);
                return 0;
            }
            n
        }
    }

    /// Number of outputs.
    pub fn num_outputs(&self) -> usize {
        // SAFETY: as above.
        unsafe {
            let Some(get) = (*self.api).Node_GetNumOutputs else {
                return 0;
            };
            let mut n: usize = 0;
            let status = get(self.node, &mut n);
            if !status.is_null() {
                sys::release_status(self.api, status);
                return 0;
            }
            n
        }
    }

    /// The node's input `OrtValueInfo` slots. Entries may legitimately be null: ORT reports an
    /// omitted *interior* optional input (`Clip(x, , max)`) as a null slot rather than by
    /// shortening the list.
    fn input_slots(&self) -> Vec<*const ort::OrtValueInfo> {
        let n = self.num_inputs();
        if n == 0 {
            return Vec::new();
        }
        // SAFETY: `self.api` is live. `buf` is `n` contiguous, initialised, writable pointer
        // slots, and `n` is the count ORT itself just reported, so the `_Out_writes_(n)` contract
        // is satisfied. ORT writes borrowed pointers into graph-owned storage that outlives this
        // call; we never free them. Any status is owned by us and released here.
        unsafe {
            let Some(get) = (*self.api).Node_GetInputs else {
                return Vec::new();
            };
            let mut buf: Vec<*const ort::OrtValueInfo> = vec![std::ptr::null(); n];
            let status = get(self.node, buf.as_mut_ptr(), n);
            if !status.is_null() {
                sys::release_status(self.api, status);
                return Vec::new();
            }
            buf
        }
    }

    /// The node's output `OrtValueInfo` slots. See [`NodeView::input_slots`].
    fn output_slots(&self) -> Vec<*const ort::OrtValueInfo> {
        let n = self.num_outputs();
        if n == 0 {
            return Vec::new();
        }
        // SAFETY: identical contract to `input_slots`, for `Node_GetOutputs`.
        unsafe {
            let Some(get) = (*self.api).Node_GetOutputs else {
                return Vec::new();
            };
            let mut buf: Vec<*const ort::OrtValueInfo> = vec![std::ptr::null(); n];
            let status = get(self.node, buf.as_mut_ptr(), n);
            if !status.is_null() {
                sys::release_status(self.api, status);
                return Vec::new();
            }
            buf
        }
    }

    /// Resolve one `OrtValueInfo` to an [`EdgeType`].
    fn edge_type(&self, slot: *const ort::OrtValueInfo) -> Option<EdgeType> {
        if slot.is_null() {
            return None;
        }
        // SAFETY: `slot` is a non-null, graph-owned `OrtValueInfo` from `Node_Get{Inputs,Outputs}`
        // and `self.api` is live. `GetValueInfoTypeInfo` yields a `const OrtTypeInfo*` **borrowed**
        // from the value info (the C API declares it const and documents no release, unlike the
        // owning `GetTypeInfo`), so it must not be released here. `CastTypeInfoToTensorInfo` is
        // `_Outptr_result_maybenull_` and returns null for non-tensor types, which we check. All
        // out-params are valid, initialised slots, and `dims` is sized by the count ORT reported
        // one call earlier. Every status is owned by us and released before returning.
        unsafe {
            let get_ti = (*self.api).GetValueInfoTypeInfo?;
            let mut ti: *const ort::OrtTypeInfo = std::ptr::null();
            let status = get_ti(slot, &mut ti);
            if !status.is_null() {
                sys::release_status(self.api, status);
                return None;
            }
            if ti.is_null() {
                return None;
            }

            let cast = (*self.api).CastTypeInfoToTensorInfo?;
            let mut tt: *const ort::OrtTensorTypeAndShapeInfo = std::ptr::null();
            let status = cast(ti, &mut tt);
            if !status.is_null() {
                sys::release_status(self.api, status);
                return None;
            }
            if tt.is_null() {
                // Not a tensor (sequence, map, optional). No claim predicate handles those, and
                // an all-unknown `EdgeType` declines everywhere, which is the correct answer.
                return Some(EdgeType::default());
            }

            let mut dtype = None;
            if let Some(get_et) = (*self.api).GetTensorElementType {
                let mut et: ort::ONNXTensorElementDataType =
                    ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
                let status = get_et(tt, &mut et);
                if status.is_null() {
                    dtype = dtype_from_onnx(et);
                } else {
                    sys::release_status(self.api, status);
                }
            }

            let mut shape = None;
            if let (Some(get_n), Some(get_d)) =
                ((*self.api).GetDimensionsCount, (*self.api).GetDimensions)
            {
                let mut rank: usize = 0;
                let status = get_n(tt, &mut rank);
                if status.is_null() {
                    let mut dims = vec![0i64; rank];
                    let ok = if rank == 0 {
                        true
                    } else {
                        let status = get_d(tt, dims.as_mut_ptr(), rank);
                        if status.is_null() {
                            true
                        } else {
                            sys::release_status(self.api, status);
                            false
                        }
                    };
                    if ok {
                        // ORT spells symbolic dims as negative; normalise them all to -1.
                        for d in &mut dims {
                            if *d < 0 {
                                *d = -1;
                            }
                        }
                        shape = Some(dims);
                    }
                } else {
                    sys::release_status(self.api, status);
                }
            }

            Some(EdgeType { dtype, shape })
        }
    }

    /// Type of input `i`, or `None` if the slot is absent (omitted optional input) or untyped.
    pub fn input_type(&self, i: usize) -> Option<EdgeType> {
        let slots = self.input_slots();
        self.edge_type(*slots.get(i)?)
    }

    /// Type of output `i`.
    pub fn output_type(&self, i: usize) -> Option<EdgeType> {
        let slots = self.output_slots();
        self.edge_type(*slots.get(i)?)
    }

    /// All input types in order. `None` entries are omitted optional inputs.
    pub fn input_types(&self) -> Vec<Option<EdgeType>> {
        self.input_slots()
            .into_iter()
            .map(|s| self.edge_type(s))
            .collect()
    }

    /// All output types in order.
    pub fn output_types(&self) -> Vec<Option<EdgeType>> {
        self.output_slots()
            .into_iter()
            .map(|s| self.edge_type(s))
            .collect()
    }

    /// Whether input `i` is actually present. False for an omitted interior optional input.
    pub fn has_input(&self, i: usize) -> bool {
        self.input_slots().get(i).is_some_and(|s| !s.is_null())
    }

    /// Whether input `i` is a constant initializer — i.e. a weight we may prepack at compile time.
    pub fn input_is_constant(&self, i: usize) -> bool {
        let slots = self.input_slots();
        let Some(&slot) = slots.get(i) else {
            return false;
        };
        if slot.is_null() {
            return false;
        }
        // SAFETY: `slot` is a live, graph-owned value info and `self.api` is live. The out-param
        // is a valid `bool` slot; any status is owned by us and released here.
        unsafe {
            let Some(get) = (*self.api).ValueInfo_IsConstantInitializer else {
                return false;
            };
            let mut is_const = false;
            let status = get(slot, &mut is_const);
            if !status.is_null() {
                sys::release_status(self.api, status);
                return false;
            }
            is_const
        }
    }

    /// Borrow one attribute by name, if the node has it.
    fn attr(&self, name: &str) -> Option<*const ort::OrtOpAttr> {
        let cname = std::ffi::CString::new(name).ok()?;
        // SAFETY: `self.api` and `self.node` are live. `cname` is a NUL-terminated string that
        // outlives the call. `Node_GetAttributeByName` is `_Outptr_result_maybenull_`, so a null
        // result means "no such attribute" and is checked. The returned attribute is borrowed from
        // the node and must not be released. Any status is owned by us and released here.
        unsafe {
            let get = (*self.api).Node_GetAttributeByName?;
            let mut attr: *const ort::OrtOpAttr = std::ptr::null();
            let status = get(self.node, cname.as_ptr(), &mut attr);
            if !status.is_null() {
                sys::release_status(self.api, status);
                return None;
            }
            if attr.is_null() { None } else { Some(attr) }
        }
    }

    /// Read an attribute into `buf`, returning the number of bytes ORT reports it wrote.
    ///
    /// # Safety
    /// `buf` must be valid for `len` bytes of writes and correctly aligned for `kind`'s element
    /// type.
    unsafe fn read_attr_raw(
        &self,
        attr: *const ort::OrtOpAttr,
        kind: ort::OrtOpAttrType,
        buf: *mut std::ffi::c_void,
        len: usize,
    ) -> Option<usize> {
        // SAFETY: `attr` is a live borrowed attribute and `self.api` is live. The caller
        // guarantees `buf`/`len` describe a valid writable region correctly aligned for `kind`.
        // `out` is a valid slot. Any status is owned by us and released here; ORT also sets `out`
        // to the required byte count when it fails for insufficient space, but we only trust the
        // value on success.
        unsafe {
            let read = (*self.api).ReadOpAttr?;
            let mut out: usize = 0;
            let status = read(attr, kind, buf, len, &mut out);
            if !status.is_null() {
                sys::release_status(self.api, status);
                return None;
            }
            Some(out)
        }
    }

    /// True if the node carries an attribute with this name.
    pub fn has_attr(&self, name: &str) -> bool {
        self.attr(name).is_some()
    }

    /// Read an `int` attribute.
    pub fn attr_int(&self, name: &str) -> Option<i64> {
        let attr = self.attr(name)?;
        let mut v: i64 = 0;
        // SAFETY: `&raw mut v` points at a valid, aligned `i64` slot of exactly
        // `size_of::<i64>()` bytes, which is what `ORT_OP_ATTR_INT` writes.
        unsafe {
            self.read_attr_raw(
                attr,
                ort::OrtOpAttrType_ORT_OP_ATTR_INT,
                (&raw mut v).cast(),
                std::mem::size_of::<i64>(),
            )?;
        }
        Some(v)
    }

    /// Read a `float` attribute.
    pub fn attr_float(&self, name: &str) -> Option<f32> {
        let attr = self.attr(name)?;
        let mut v: f32 = 0.0;
        // SAFETY: `&raw mut v` points at a valid, aligned `f32` slot of exactly
        // `size_of::<f32>()` bytes, which is what `ORT_OP_ATTR_FLOAT` writes.
        unsafe {
            self.read_attr_raw(
                attr,
                ort::OrtOpAttrType_ORT_OP_ATTR_FLOAT,
                (&raw mut v).cast(),
                std::mem::size_of::<f32>(),
            )?;
        }
        Some(v)
    }

    /// Read an `ints` attribute. Capped at [`MAX_ATTR_INTS`] elements, which is far above anything
    /// a real graph carries (`perm`, `axes`, `pads`, `kernel_shape`).
    pub fn attr_ints(&self, name: &str) -> Option<Vec<i64>> {
        let attr = self.attr(name)?;
        let mut buf = vec![0i64; MAX_ATTR_INTS];
        let bytes = std::mem::size_of_val(buf.as_slice());
        // SAFETY: `buf` is `MAX_ATTR_INTS` contiguous, initialised, aligned `i64` slots and
        // `bytes` is its exact byte length, so `ORT_OP_ATTR_INTS` cannot overrun it.
        let written = unsafe {
            self.read_attr_raw(
                attr,
                ort::OrtOpAttrType_ORT_OP_ATTR_INTS,
                buf.as_mut_ptr().cast(),
                bytes,
            )?
        };
        let n = written / std::mem::size_of::<i64>();
        if n > buf.len() {
            return None;
        }
        buf.truncate(n);
        Some(buf)
    }

    /// Read a `string` attribute. Capped at [`MAX_ATTR_STRING`] bytes.
    pub fn attr_string(&self, name: &str) -> Option<String> {
        let attr = self.attr(name)?;
        let mut buf = vec![0u8; MAX_ATTR_STRING];
        let len = buf.len();
        // SAFETY: `buf` is `len` contiguous, initialised, writable bytes; `u8` has no alignment
        // requirement, so `ORT_OP_ATTR_STRING` can neither overrun nor misalign it.
        let written = unsafe {
            self.read_attr_raw(
                attr,
                ort::OrtOpAttrType_ORT_OP_ATTR_STRING,
                buf.as_mut_ptr().cast(),
                len,
            )?
        };
        if written > buf.len() {
            return None;
        }
        buf.truncate(written);
        while buf.last() == Some(&0) {
            buf.pop();
        }
        String::from_utf8(buf).ok()
    }

    /// The raw node pointer, for `ep.rs` to hand back to `EpGraphSupportInfo_AddNodesToFuse`.
    ///
    /// `pub(crate)` on purpose: nothing outside the boundary layer has any use for it.
    pub(crate) fn raw(&self) -> *const ort::OrtNode {
        self.node
    }

    /// Every attribute name present on this node.
    ///
    /// This is the input to the contrib schema-drift check ([`ContribSchema`]): a `com.microsoft`
    /// node carrying an attribute name we have never heard of means the schema changed under us,
    /// and the only safe answer is to decline.
    ///
    /// Note ORT materialises *defaulted* optional attributes here as well, whenever the default is
    /// a constant expression, so the list is the effective schema rather than only what the
    /// exporter wrote. That is what makes it usable as a fingerprint.
    ///
    /// Returns an empty list when the node has no attributes, and also when the entry point is
    /// missing on the host (ORT ≥ 1.23) — a schema check that cannot read attributes must not
    /// invent a decline reason, so it falls through to the arity checks.
    pub fn attr_names(&self) -> Vec<String> {
        let n = self.num_attrs();
        if n == 0 {
            return Vec::new();
        }
        // SAFETY: `self.api` and `self.node` are live. `buf` is `n` contiguous, initialised,
        // writable pointer slots where `n` is the count ORT itself just reported, satisfying the
        // `_Out_writes_(num_attributes)` contract. ORT writes borrowed pointers into node-owned
        // storage that outlives this call and must not be released. `OpAttr_GetName` yields a
        // borrowed NUL-terminated string with the same lifetime, which is copied before returning.
        // Every status is owned by us and released here.
        unsafe {
            let Some(get) = (*self.api).Node_GetAttributes else {
                return Vec::new();
            };
            let mut buf: Vec<*const ort::OrtOpAttr> = vec![std::ptr::null(); n];
            let status = get(self.node, buf.as_mut_ptr(), n);
            if !status.is_null() {
                sys::release_status(self.api, status);
                return Vec::new();
            }
            let Some(get_name) = (*self.api).OpAttr_GetName else {
                return Vec::new();
            };
            let mut names = Vec::with_capacity(n);
            for attr in buf {
                if attr.is_null() {
                    continue;
                }
                let mut raw: *const std::os::raw::c_char = std::ptr::null();
                let status = get_name(attr, &mut raw);
                if !status.is_null() {
                    sys::release_status(self.api, status);
                    continue;
                }
                if raw.is_null() {
                    continue;
                }
                names.push(std::ffi::CStr::from_ptr(raw).to_string_lossy().into_owned());
            }
            names
        }
    }

    /// How many attributes the node carries.
    fn num_attrs(&self) -> usize {
        // SAFETY: `self.api` and `self.node` are live and `n` is a valid, initialised out-param.
        // Any status is owned by us and released here.
        unsafe {
            let Some(get) = (*self.api).Node_GetNumAttributes else {
                return 0;
            };
            let mut n: usize = 0;
            let status = get(self.node, &mut n);
            if !status.is_null() {
                sys::release_status(self.api, status);
                return 0;
            }
            n
        }
    }

    /// Domain-qualified op name — the registry key. See [`NodeDesc::qualified_name`].
    pub fn qualified_name(&self) -> String {
        let domain = self.domain();
        let op = self.op_type();
        if domain.is_empty() || domain == "ai.onnx" {
            op
        } else {
            format!("{domain}::{op}")
        }
    }
}

/// Upper bound on elements read from an `ints` attribute.
pub const MAX_ATTR_INTS: usize = 64;

/// Upper bound on bytes read from a `string` attribute.
pub const MAX_ATTR_STRING: usize = 256;

/// Map an ONNX element type to this EP's [`DType`], or `None` for types we have no storage for.
fn dtype_from_onnx(et: ort::ONNXTensorElementDataType) -> Option<DType> {
    match et {
        ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT => Some(DType::F32),
        ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 => Some(DType::F16),
        ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 => Some(DType::I64),
        ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 => Some(DType::I32),
        ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 => Some(DType::U8),
        ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL => Some(DType::Bool),
        _ => None,
    }
}

// -------------------------------------------------------------------------------------------
// Public helpers for `compile_impl` (ep.rs boundary layer)
// -------------------------------------------------------------------------------------------

/// Read the name of an `OrtValueInfo` pointer.
///
/// Returns an empty string on any failure (null pointer, missing API entry, ORT error).
///
/// # Safety
/// `api` must be a live `OrtApi`; `vi` must be a live `OrtValueInfo` owned by an ORT graph
/// that outlives this call.
pub unsafe fn value_info_name(api: *const ort::OrtApi, vi: *const ort::OrtValueInfo) -> String {
    if vi.is_null() || api.is_null() {
        return String::new();
    }
    // SAFETY: api and vi are live per the caller's contract.
    unsafe {
        let Some(get) = (*api).GetValueInfoName else {
            return String::new();
        };
        let mut ptr: *const std::ffi::c_char = std::ptr::null();
        let st = get(vi, &mut ptr);
        if !st.is_null() {
            sys::release_status(api, st);
            return String::new();
        }
        if ptr.is_null() {
            return String::new();
        }
        std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }
}

/// Read the [`EdgeType`] of a standalone `OrtValueInfo` pointer.
///
/// Equivalent to `NodeView::edge_type(slot)` but usable outside of `NodeView`'s method context,
/// so that `compile_impl` can build [`crate::engine::TensorDesc`] for graph inputs/outputs.
///
/// Returns `None` for null pointers, non-tensor types, or when ORT's type-info API is absent.
///
/// # Safety
/// `api` must be a live `OrtApi`; `vi` must be a live `OrtValueInfo` owned by an ORT graph.
pub unsafe fn value_info_edge_type(
    api: *const ort::OrtApi,
    vi: *const ort::OrtValueInfo,
) -> Option<EdgeType> {
    if vi.is_null() || api.is_null() {
        return None;
    }
    // SAFETY: api and vi are live per the caller's contract. The logic below is identical to
    // `NodeView::edge_type` but uses the standalone `GetValueInfoTypeInfo` path (which borrows
    // the type info rather than requiring a release). All out-params are valid initialised slots;
    // all status pointers are owned by us and released on error.
    unsafe {
        let get_ti = (*api).GetValueInfoTypeInfo?;
        let mut ti: *const ort::OrtTypeInfo = std::ptr::null();
        let st = get_ti(vi, &mut ti);
        if !st.is_null() {
            sys::release_status(api, st);
            return None;
        }
        if ti.is_null() {
            return None;
        }

        let cast = (*api).CastTypeInfoToTensorInfo?;
        let mut tt: *const ort::OrtTensorTypeAndShapeInfo = std::ptr::null();
        let st = cast(ti, &mut tt);
        if !st.is_null() {
            sys::release_status(api, st);
            return None;
        }
        if tt.is_null() {
            return Some(EdgeType::default());
        }

        let mut dtype = None;
        if let Some(get_et) = (*api).GetTensorElementType {
            let mut et: ort::ONNXTensorElementDataType =
                ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
            let st = get_et(tt, &mut et);
            if st.is_null() {
                dtype = dtype_from_onnx(et);
            } else {
                sys::release_status(api, st);
            }
        }

        let mut shape = None;
        if let (Some(get_n), Some(get_d)) = ((*api).GetDimensionsCount, (*api).GetDimensions) {
            let mut rank: usize = 0;
            let st = get_n(tt, &mut rank);
            if st.is_null() {
                let mut dims = vec![0i64; rank];
                let ok = if rank == 0 {
                    true
                } else {
                    let st = get_d(tt, dims.as_mut_ptr(), rank);
                    if st.is_null() {
                        true
                    } else {
                        sys::release_status(api, st);
                        false
                    }
                };
                if ok {
                    for d in &mut dims {
                        if *d < 0 {
                            *d = -1;
                        }
                    }
                    shape = Some(dims);
                }
            } else {
                sys::release_status(api, st);
            }
        }

        Some(EdgeType { dtype, shape })
    }
}

// -------------------------------------------------------------------------------------------
// Decline reasons
// -------------------------------------------------------------------------------------------

/// Why a node was not claimed. Always a sentence a user can act on; it is printed verbatim by
/// `ONNXRUNTIME_EP_VULKAN_CLAIM_DEBUG=1` and is the single most valuable diagnostic the reference
/// EP has (`DESIGN.md` §5.4).
///
/// Every reason produced by this crate is built with [`decline`] and therefore starts with a
/// `[tag]` that [`DeclineCode::of_reason`] can parse back out.
pub type DeclineReason = Cow<'static, str>;

/// The machine-readable cause of a decline.
///
/// Kept as a small closed set on purpose: the point is that Niobe can histogram declines by cause
/// across a whole model and say "62% of unclaimed nodes are `[dtype]`", which is an actionable
/// number, where a free-text histogram is not.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DeclineCode {
    /// No row in the registry for this domain-qualified op.
    NotRegistered,
    /// A row exists but its kernel is not implemented yet.
    Staged,
    /// The op's opset version is outside the row's supported window.
    Opset,
    /// Wrong number of inputs or outputs.
    Arity,
    /// A required input is absent (omitted optional, or unresolvable).
    MissingInput,
    /// An input or output dtype is outside the row's `caps`.
    DType,
    /// Rank exceeds what the shared indexing helper supports.
    Rank,
    /// Shapes are known but incompatible (e.g. not broadcastable).
    Shape,
    /// **Rank is known; at least one extent is symbolic.**
    ///
    /// A *floor*, not a ceiling: a node in this bucket has already passed registration, opset,
    /// contrib schema and status, so shape is its sole remaining blocker. It becomes claimable
    /// with no kernel change once extents are runtime parameters
    /// (`claim::ENGINE_ACCEPTS_RUNTIME_EXTENTS`). Kept strictly separate from
    /// [`DeclineCode::UnknownRank`], which no amount of that work reaches.
    DynamicShape,
    /// **No shape at all — even the rank is unknown.**
    ///
    /// Rank determines indexing arithmetic and descriptor layout, which live in the pipeline
    /// rather than in a push constant, so this is not unlocked by runtime extents. Split out of
    /// [`DeclineCode::DynamicShape`] on 2026-07-29 because merging them made the histogram
    /// unreadable: one bucket was work we had costed and the other never was.
    UnknownRank,
    /// The op's output shape depends on input **values**, so the output cannot be sized before
    /// the kernel that determines it has run. Permanent under one-command-buffer-per-subgraph.
    DataDependentShape,
    /// An attribute value or combination we do not handle.
    Attribute,
    /// A `com.microsoft` node does not match the contrib schema this row was written against.
    ///
    /// Distinct from [`DeclineCode::Attribute`] on purpose: `Attribute` means "a value we chose
    /// not to support", `ContribSchema` means "**the schema moved under us**". Contrib ops carry
    /// no opset guarantee, so this bucket appearing at all is a signal to re-verify the row
    /// against the ORT release in front of us, not to widen a predicate.
    ContribSchema,
    /// The node is fine but the subgraph it would join is not viable — see `ops::partition`.
    Partition,
    /// A row is `Ready` (kernel exists) but no ledger proof covers this form, and the proof
    /// key is not in `ONNXRUNTIME_EP_VULKAN_CLAIM_UNPROVEN`.
    ///
    /// Under §8.9 (`DESIGN.md`), claiming is gated on evidence: a `Ready` row without a proof
    /// entry is, for claiming purposes, equivalent to a row we cannot run.  The escape hatch
    /// (`CLAIM_UNPROVEN`) accepts a semicolon-separated list of explicit proof keys and nothing
    /// else — no wildcards, no `=1`, no domain patterns.
    Unproven,
    /// An internal inconsistency; should never reach a user.
    Internal,
}

impl DeclineCode {
    /// The canonical lowercase tag, as it appears in `[brackets]` at the head of a reason.
    pub const fn tag(self) -> &'static str {
        match self {
            DeclineCode::NotRegistered => "not-registered",
            DeclineCode::Staged => "staged",
            DeclineCode::Opset => "opset",
            DeclineCode::Arity => "arity",
            DeclineCode::MissingInput => "missing-input",
            DeclineCode::DType => "dtype",
            DeclineCode::Rank => "rank",
            DeclineCode::Shape => "shape",
            DeclineCode::DynamicShape => "dynamic-shape",
            DeclineCode::UnknownRank => "unknown-rank",
            DeclineCode::DataDependentShape => "data-dependent-shape",
            DeclineCode::Attribute => "attribute",
            DeclineCode::ContribSchema => "contrib-schema",
            DeclineCode::Partition => "partition",
            DeclineCode::Unproven => "unproven",
            DeclineCode::Internal => "internal",
        }
    }

    /// Every code, for exhaustive tests and for documenting the histogram buckets.
    pub const ALL: &'static [DeclineCode] = &[
        DeclineCode::NotRegistered,
        DeclineCode::Staged,
        DeclineCode::Opset,
        DeclineCode::Arity,
        DeclineCode::MissingInput,
        DeclineCode::DType,
        DeclineCode::Rank,
        DeclineCode::Shape,
        DeclineCode::DynamicShape,
        DeclineCode::UnknownRank,
        DeclineCode::DataDependentShape,
        DeclineCode::Attribute,
        DeclineCode::ContribSchema,
        DeclineCode::Partition,
        DeclineCode::Unproven,
        DeclineCode::Internal,
    ];

    /// Parse a bare tag.
    pub fn of_tag(tag: &str) -> Option<DeclineCode> {
        DeclineCode::ALL.iter().copied().find(|c| c.tag() == tag)
    }

    /// Recover the code from a rendered [`DeclineReason`].
    ///
    /// This is the reader half of the contract: `ep.rs` aggregates reasons as opaque strings and
    /// tooling re-derives the cause here. Returns `None` for a reason that did not come from
    /// [`decline`] (e.g. `ep.rs`'s own `max_claim_ops` message), which is why the histogram needs
    /// an explicit "other" bucket.
    pub fn of_reason(reason: &str) -> Option<DeclineCode> {
        let rest = reason.strip_prefix('[')?;
        let (tag, _) = rest.split_once(']')?;
        DeclineCode::of_tag(tag)
    }
}

/// Build a machine-readable decline reason.
///
/// The rendered form is `"[tag] detail"`. Always use this rather than constructing a `Cow`
/// directly, or the reason becomes invisible to the decline histogram.
pub fn decline(code: DeclineCode, detail: impl std::fmt::Display) -> DeclineReason {
    Cow::Owned(format!("[{}] {detail}", code.tag()))
}

/// `return` a decline from a claim predicate.
///
/// ```ignore
/// deny!(DType, "input 0 is {found}; `{op}` supports {caps}");
/// ```
#[macro_export]
macro_rules! deny {
    ($code:ident, $($arg:tt)+) => {
        return ::core::result::Result::Err($crate::registry::decline(
            $crate::registry::DeclineCode::$code,
            ::std::format_args!($($arg)+),
        ))
    };
}

/// Decline unless a condition holds.
///
/// The inverse of `assert!`, and the reason claim predicates read as a list of requirements rather
/// than a tree of `if`s.
#[macro_export]
macro_rules! require {
    ($cond:expr, $code:ident, $($arg:tt)+) => {
        if !($cond) {
            $crate::deny!($code, $($arg)+);
        }
    };
}

// -------------------------------------------------------------------------------------------
// The registry table
// -------------------------------------------------------------------------------------------

/// A claim predicate: given a node and its own row, either claim it or say why not.
///
/// The row is passed in so that *one* predicate can serve many ops — it reads `spec.caps`,
/// `spec.op_type` and `spec.kernel` rather than hard-coding them. That is what turns an op into a
/// table row instead of a function.
///
/// The predicate must be *exactly* as strict as the translate handler: if it claims a node the
/// handler cannot translate, `Compile` fails and the whole subgraph falls back, which is strictly
/// worse than never claiming it.
pub type ClaimPredicate = fn(&NodeView<'_>, &OpSpec) -> Result<(), DeclineReason>;

/// A translate handler: turn a node into dispatches against the engine seam.
///
/// Takes its own row for the same reason the claim predicate does.
pub type TranslateHandler =
    fn(&OpSpec, &NodeDesc, &mut dyn crate::engine::DispatchContext) -> EpResult<()>;

/// ONNX domain of a registry row.
///
/// `Ms` (`com.microsoft`) is **admitted**: Justin ruled on 2026-07-28 that contrib ops are in
/// scope, superseding `DESIGN.md` §1.2. `OP_COVERAGE.md` §4 is the reason — a Qwen3-class graph is
/// unrunnable without `GroupQueryAttention`, `MatMulNBits`, `RotaryEmbedding` and friends, because
/// the ORT GenAI model builder emits them directly.
///
/// The admission comes with an obligation that `Ai` rows do not carry: contrib schemas are
/// versioned by ORT *release*, not by an opset, so every `Ms` row must also declare a
/// [`ContribSchema`] fingerprint. See [`OpSpec::schema`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Domain {
    /// The default ONNX domain, spelled `""` or `ai.onnx` on the wire.
    Ai,
    /// `com.microsoft`, ORT's contrib op domain.
    Ms,
}

impl Domain {
    /// The domain string as it appears in a qualified name (empty for the default domain).
    pub const fn as_str(self) -> &'static str {
        match self {
            Domain::Ai => "",
            Domain::Ms => "com.microsoft",
        }
    }
}

/// Whether a row is actually backed by a working kernel.
///
/// # §8.9 — `Live` is being replaced by `Ready`
///
/// Under the §8.9 proof-ledger ruling, the table declares only facts about *source*:
/// `Ready` means "a kernel exists"; `Staged` means "no kernel yet". Claimability is
/// derived, per form, from a harness-generated proof ledger. The hand-written `Live`
/// variant is the per-op claim that §8.9 removes.
///
/// **Transition state (2026-07-30):** `Ready` is introduced alongside `Live`. New rows
/// should use `Ready`. Existing `Live` rows are semantically unchanged until the ledger
/// check is activated in `claim_audit` — at which point every `Live`/`Ready` row without
/// a ledger entry will decline with `[unproven]` unless its proof key is in
/// `ONNXRUNTIME_EP_VULKAN_CLAIM_UNPROVEN`. The rename will complete in a single sweep
/// when Trinity's harness is ready to generate the ledger. Until then, `Live` compiles
/// and behaves identically to `Ready`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpStatus {
    /// Kernel exists, conformance proven. **Deprecated** — use `Ready`.
    ///
    /// Semantically identical to `Ready`; present only to let existing op-table rows
    /// compile unchanged during the §8.9 transition. Once the proof ledger is active,
    /// "proven" is a ledger-level fact, not a hand-written variant.
    Live,
    /// Kernel exists. Claimability is derived from the proof ledger — not from this field.
    ///
    /// A row carrying `Ready` with no ledger entry declines with `[unproven]`. A proof key
    /// in `ONNXRUNTIME_EP_VULKAN_CLAIM_UNPROVEN` overrides the ledger check for
    /// development, subject to §8.9.4 restrictions.
    Ready,
    /// Described and claim-tested, but declined at runtime with the given blocker.
    ///
    /// This is how the op table lands ahead of the shaders without ever violating the
    /// claim/translate invariant.
    Staged(&'static str),
}

/// The blocker for a row whose shader does not exist yet.
pub const NO_SHADER: &str = "its compute shader has not been written yet";

/// The blocker for a row whose shader exists and compiles, but has never executed.
///
/// This is a deliberately separate reason from [`NO_SHADER`], and the distinction is the honest
/// half of the coverage claim. A variant that `glslc` accepts is not a variant that computes the
/// right answer: nothing in the repository has yet run one of these on a device, because the
/// engine's pipeline and dispatch path is still being built. Claiming a node here would mean
/// betting a user's output on unexecuted code, which is exactly what `OP_COVERAGE.md` §7.1
/// forbids.
///
/// The exit is mechanical, not editorial: when the dispatch path exists and a differential test
/// runs the variant against the CPU EP on a real device, the row's status becomes `Live` in a
/// one-word diff. Until then the CPU EP runs these nodes and is always right.
pub const UNEXERCISED: &str = "its compute shader compiles but has never executed on a device, so claiming it would be a \
     bet rather than a capability";

/// The blocker for the XL kernels: committed work, no template leverage, not written yet.
///
/// `OP_COVERAGE.md` §11 risk 1 is that `GroupQueryAttention`, `MatMulNBits` and `LinearAttention`
/// are multi-week kernels with no template to inherit from. Justin's 2026-07-28 ruling makes them
/// committed deliverables rather than stretch goals, so they get rows, schemas and claim
/// predicates now; the kernel is what is still missing.
pub const XL_KERNEL: &str =
    "it is an XL kernel on the committed schedule whose compute shader is still being written";

/// Open-ended upper bound for a row's opset window.
pub const OPSET_ANY: i32 = i32::MAX;

/// Ops that ORT registers in the **default (`ai.onnx`) domain** while having no ONNX schema.
///
/// Discovered by censusing real Foundry Local graphs (`OP_COVERAGE.md` §4.21): the ORT GenAI model
/// builder emits `SimplifiedLayerNormalization` with `node.domain == ""`, not `"com.microsoft"`.
/// Both Phi-3.5-mini-int4 and gpt-oss-20b do this, so it is the builder's behaviour, not a quirk.
///
/// This breaks the registry's two-axis assumption. Every other row is either `ai.onnx` — versioned
/// by an opset window, because ONNX publishes a schema — or `com.microsoft` — versioned by a
/// [`ContribSchema`] fingerprint, because no opset exists. These ops are a third thing: **no ONNX
/// schema, but the default domain**, so `Node_GetSinceVersion` returns a number that means nothing
/// and only a fingerprint can detect drift.
///
/// The rule the tests enforce: an `ai.onnx` row may carry a fingerprint **iff** its op type is
/// listed here, and such a row must declare the full `1 ..= OPSET_ANY` window rather than pretend
/// its opset bound says something. The list is a hazard register — it grows only when a real graph
/// is observed emitting the op with an empty domain, and if it ever gets long the registry needs a
/// third `Domain` variant instead of an allow-list.
pub const ORT_FUSED_IN_DEFAULT_DOMAIN: &[&str] = &["SimplifiedLayerNormalization"];

/// The first `ai.onnx` opset containing the standard-domain LLM ops.
///
/// ONNX opset 23 (onnx 1.18) added `Attention`, `RMSNormalization` and `RotaryEmbedding` to the
/// default domain. Those are the spellings a model built by `onnxruntime/mobius` uses — where an
/// ORT-GenAI-built graph would carry `com.microsoft::GroupQueryAttention`,
/// `SimplifiedLayerNormalization` and `com.microsoft::RotaryEmbedding` instead. We register both
/// spellings; see `OP_COVERAGE.md` §4.16.
///
/// The window starts here rather than at 1 because a node claiming to be `Attention` at opset 22
/// is not the operator we implement.
pub const OPSET_STD_LLM: i32 = 23;

/// The newest ONNX release whose `defs.cc`/`old.cc` this crate's `ai.onnx` windows were read from.
///
/// The contrib analogue is [`SCHEMA_VERIFIED_ON`]. `ai.onnx` needs its own because an opset window
/// is only as trustworthy as the last time somebody read the spec: `Attention` gained an input at
/// opset 24 while this crate's row said `23 ..= OPSET_ANY`, which would have claimed the new form
/// and silently computed the wrong causal offset. See `OP_COVERAGE.md` §4.19.
pub const ONNX_SPEC_READ: &str = "onnx v1.22.0 — opset 27 registered, 26 last-released; defs.cc/old.cc/operator_sets.h read 2026-07-29";

/// The highest `ai.onnx` opset ONNX itself declares **released**: 26 (onnx v1.21.0).
///
/// Distinct from the highest *registered* opset, which is 27 in onnx v1.22.0. ONNX keeps the two
/// apart deliberately — `schema.h` carries both `map_[ONNX_DOMAIN] = {1, 27}` and
/// `last_release_version_map_[ONNX_DOMAIN] = 26`, with the comment *"in other versions, the max
/// version may be ahead of the last-release-version."* `onnx.defs.onnx_opset_version()` reports the
/// registered maximum (27); the release field reports 26. Both numbers are correct answers to
/// different questions, which is why `OP_COVERAGE.md` §4.20 records the question along with them.
pub const ONNX_OPSET_LAST_RELEASED: i32 = 26;

/// The highest `ai.onnx` opset **registered** in the newest onnx release: 27 (onnx v1.22.0).
///
/// Models can be stamped 27 — `helper.make_model` defaults to it and the checker accepts it — so a
/// node carrying a schema version of 27 is something we can actually be handed.
pub const ONNX_OPSET_REGISTERED: i32 = 27;

/// Schema versions of `ai.onnx::Attention`: 23 (onnx 1.18) and 24 (onnx 1.19).
///
/// Closed, not open-ended. Opset 24 added optional input 6 `nonpad_kv_seqlen` — the external
/// KV-cache pattern that pairs with [`TensorScatter`](OPSET_STD_TENSOR_SCATTER) — and changed the
/// meaning of `is_causal` when it is present.
///
/// 24 is also the **newest `Attention` schema that exists**: opsets 25, 26 and 27 register no
/// `Attention`. So this window is complete coverage of the op, not a restriction — a model stamped
/// opset 27 still resolves its `Attention` nodes to schema version 24 and is claimable here. §4.20.
pub const OPSET_STD_ATTENTION_MAX: i32 = 24;

/// `ai.onnx::RMSNormalization` and `ai.onnx::RotaryEmbedding` exist at exactly one schema version.
///
/// Both were introduced at opset 23 and are unrevised through opset 27 — verified: neither has an
/// entry in the opset-24 section of `onnx/defs/operator_sets.h`, and both still live in
/// `defs.cc` (current) rather than `old.cc`. A future revision declines rather than mis-claims.
pub const OPSET_STD_NORM_MAX: i32 = 23;

/// `ai.onnx::TensorScatter`, new at opset 24: the functional model of an in-place KV-cache write.
pub const OPSET_STD_TENSOR_SCATTER: i32 = 24;

/// `ai.onnx::Swish`, new at opset 24: `x * sigmoid(alpha * x)`, i.e. SiLU at `alpha = 1`.
pub const OPSET_STD_SWISH: i32 = 24;

/// `ai.onnx::LinearAttention` and `ai.onnx::CausalConvWithState`, new at opset 27 (onnx v1.22.0).
///
/// The standard-domain spellings of two ops we already had `com.microsoft` rows for. 27 is both the
/// version they were introduced at and the newest registered, so the window is a single value.
pub const OPSET_STD_SSM: i32 = 27;

/// Highest `QuantizeLinear`/`DequantizeLinear` schema version this crate has read: 25.
///
/// Q/DQ is revised almost every release — 21 added `block_size` and `output_dtype`, 23 split the
/// scale type constraint out and added `precision`, 24 admitted `float8e8m0` scales, 25 is current.
/// An open-ended window over an op with that revision rate is the same bug as `Attention`'s.
pub const OPSET_QDQ_MAX: i32 = 25;

/// The date every contrib fingerprint in this crate was read from ORT's schema documentation.
///
/// One constant, so that "when did anyone last check the contrib schemas" has exactly one answer
/// and a re-verification pass is one edit rather than a search.
pub const SCHEMA_VERIFIED_ON: &str = "2026-07-28";

/// A schema baseline read from the pinned ORT release on [`SCHEMA_VERIFIED_ON`].
pub const PINNED_BASELINE: SchemaBaseline = SchemaBaseline::pinned(SCHEMA_VERIFIED_ON);

/// A schema baseline for an op that exists only on ORT's main branch.
///
/// `LinearAttention` and `CausalConvWithState` are not in any release; saying they were verified
/// against 1.28 would be false, and C2 exists precisely so that this distinction is visible in
/// `epctl --dump-capabilities` rather than buried in a comment.
pub const MAIN_BASELINE: SchemaBaseline = SchemaBaseline {
    verified_against: OrtRelease {
        release: "main (post-1.28.0)",
        api_version: sys::ORT_API_VERSION_EXPECTED,
    },
    verified_on: SCHEMA_VERIFIED_ON,
};

/// The contrib-schema fingerprint a `com.microsoft` row was written against.
///
/// # Why contrib ops need this and `ai.onnx` ops do not
///
/// An `ai.onnx` op has an opset: the schema for `Add`-13 is frozen forever, so a `min_opset ..=
/// max_opset` window is a complete compatibility statement. `com.microsoft` ops have no such
/// guarantee — they are versioned by ORT release, their `since_version` is essentially always 1,
/// and new inputs and attributes are added to them in place. `LinearAttention` and
/// `CausalConvWithState` do not exist in ORT 1.28 at all; they are main-branch ops today.
///
/// So the opset window is worthless for contrib rows, and something must take its place. This
/// struct is that something: a *fingerprint* of the schema the row's claim predicate and translate
/// handler were written against. If the node in front of us does not match it, the schema moved,
/// and we decline with [`DeclineCode::ContribSchema`].
///
/// # Failure direction
///
/// Deliberately conservative. A fingerprint that is too narrow produces a decline and a CPU
/// fallback, which is always correct; a fingerprint that is too wide produces a wrong answer.
/// Where the exact schema could not be verified, the fields below are written narrow and the row's
/// doc comment says so. The `[contrib-schema]` histogram bucket is then the tool that tells us
/// which fingerprints to widen, with evidence.
///
/// # What this does *not* detect (`OP_COVERAGE.md` §9.4.1)
///
/// Everything here is version-based: it answers *"has the declared interface moved?"* It cannot
/// answer *"has the meaning of an unchanged interface moved?"* An operator whose semantics are
/// corrected **without** a version change — as `ai.onnx::Attention`-24 was, by user ruling on
/// 2026-07-29 — is invisible to this struct *and* to an opset window, because every number either
/// detector compares stays identical across the change.
///
/// There is no fingerprint field that would catch it. The only signal is a differential test
/// against a *pinned* reference, which is why the harness's `onnx` and `onnxruntime` versions are
/// correctness inputs rather than test hygiene. When such a case is known for a row, it belongs in
/// [`ContribSchema::notes`] — the one field that can carry a fact the structure cannot express.
#[derive(Clone, Copy, Debug)]
pub struct ContribSchema {
    /// The ORT release this fingerprint was read from, and the date someone read it.
    ///
    /// This is `DESIGN.md` §1.4 constraint **C2**, and it lives *inside* the fingerprint rather
    /// than beside it on purpose: a shape without a provenance is unauditable, and a provenance
    /// without a shape is unenforceable. Making them one value makes it impossible to record
    /// either half alone. [`OpSpec::schema_baseline`] is the accessor `epctl
    /// --dump-capabilities` surfaces.
    pub baseline: SchemaBaseline,
    /// Anything a reader must know that the release string cannot say — most importantly, "this
    /// op is not in the pinned release at all". Empty when there is nothing to add.
    pub notes: &'static str,
    /// Fewest inputs the verified schema allows (trailing optionals omitted).
    pub min_inputs: usize,
    /// Most inputs the verified schema allows (every optional present).
    pub max_inputs: usize,
    /// Fewest outputs the verified schema allows.
    pub min_outputs: usize,
    /// Most outputs the verified schema allows.
    pub max_outputs: usize,
    /// Attributes that must be present for the node to be meaningful.
    pub required_attrs: &'static [&'static str],
    /// **Every** attribute name the verified schema defines, required and optional.
    ///
    /// An attribute outside this set is the schema-drift signal: ORT materialises defaulted
    /// optional attributes, so a name we have never heard of means a new one was added, which
    /// means our reading of the op may be stale in ways the arity check cannot see.
    pub known_attrs: &'static [&'static str],
}

impl ContribSchema {
    /// What a decline message says we were reading, e.g.
    /// `ort-1.28.0 (api 28), verified 2026-07-28`.
    pub fn provenance(&self) -> String {
        if self.notes.is_empty() {
            self.baseline.describe()
        } else {
            format!("{} ({})", self.baseline.describe(), self.notes)
        }
    }

    /// Check a node against the fingerprint. Runs before the row's own claim predicate.
    pub fn check(&self, view: &NodeView<'_>, qualified: &str) -> Result<(), DeclineReason> {
        let inputs = view.num_inputs();
        if inputs < self.min_inputs || inputs > self.max_inputs {
            return Err(decline(
                DeclineCode::ContribSchema,
                format_args!(
                    "`{qualified}` has {inputs} inputs; the schema this EP was written against \
                     [{}] has {}..={}. The contrib schema has changed — re-verify the row before \
                     widening it",
                    self.provenance(),
                    self.min_inputs,
                    self.max_inputs
                ),
            ));
        }

        let outputs = view.num_outputs();
        if outputs < self.min_outputs || outputs > self.max_outputs {
            return Err(decline(
                DeclineCode::ContribSchema,
                format_args!(
                    "`{qualified}` has {outputs} outputs; the schema this EP was written against \
                     [{}] has {}..={}",
                    self.provenance(),
                    self.min_outputs,
                    self.max_outputs
                ),
            ));
        }

        let present = view.attr_names();
        if let Some(unknown) = present.iter().find(|n| !self.knows(n)) {
            return Err(decline(
                DeclineCode::ContribSchema,
                format_args!(
                    "`{qualified}` carries attribute `{unknown}`, which is not in the schema this \
                     EP was written against [{}]. Contrib ops carry no opset guarantee, so an \
                     unknown attribute means the op may behave differently than this EP assumes",
                    self.provenance()
                ),
            ));
        }

        // Only enforceable when we could actually read the attribute list.
        if !present.is_empty() {
            for required in self.required_attrs {
                if !present.iter().any(|n| n == required) {
                    return Err(decline(
                        DeclineCode::ContribSchema,
                        format_args!(
                            "`{qualified}` is missing required attribute `{required}` from the \
                             schema this EP was written against [{}]",
                            self.provenance()
                        ),
                    ));
                }
            }
        }

        Ok(())
    }

    /// Is this attribute name part of the verified schema?
    pub fn knows(&self, name: &str) -> bool {
        self.known_attrs.contains(&name)
    }
}

/// One row of the registry: everything the machinery needs to claim and translate an op.
#[derive(Clone, Copy, Debug)]
pub struct OpSpec {
    /// ONNX domain.
    pub domain: Domain,
    /// ONNX op type, e.g. `Add`.
    pub op_type: &'static str,
    /// Lowest `since_version` this row handles.
    pub min_opset: i32,
    /// Highest `since_version` this row handles; [`OPSET_ANY`] for open-ended.
    pub max_opset: i32,
    /// Dtypes this row supports. Shared claim predicates check inputs against it and the shader
    /// variant table generates exactly these variants — one source of truth for both.
    pub caps: DTypeSet,
    /// Which shader template and which template op back this row.
    pub kernel: Kernel,
    /// The claim predicate.
    pub claim: ClaimPredicate,
    /// The translate handler.
    pub translate: TranslateHandler,
    /// Live or staged.
    pub status: OpStatus,
    /// The contrib schema this row was written against. Required for every [`Domain::Ms`] row and
    /// meaningless for an [`Domain::Ai`] one; a test enforces both halves.
    pub schema: Option<&'static ContribSchema>,
    /// The `Compile`-time hook, if this op has work to do before its first dispatch.
    ///
    /// Today that means weight prepacking (`OP_COVERAGE.md` §8.2.1): `MatMulNBits` must repack its
    /// quantized weight into the tile-friendly layout its GEMM wants, once, after device selection.
    /// Almost every row leaves this `None`, which is why it is an optional column rather than a
    /// required one.
    pub compile: Option<CompileHook>,
}

impl OpSpec {
    /// The registry key: `Add`, or `com.microsoft::MatMulNBits`.
    pub fn qualified_name(&self) -> Cow<'static, str> {
        if self.domain.as_str().is_empty() {
            Cow::Borrowed(self.op_type)
        } else {
            Cow::Owned(format!("{}::{}", self.domain.as_str(), self.op_type))
        }
    }

    /// Whether this row is claimable at all (kernel exists, regardless of proof status).
    ///
    /// Returns `true` for both `Live` (deprecated) and `Ready` rows; `Staged` rows always
    /// return `false`.  Under §8.9, a `true` result here is necessary but not sufficient —
    /// claiming is also gated on the proof ledger (or `CLAIM_UNPROVEN` escape hatch).
    pub fn is_live(&self) -> bool {
        matches!(self.status, OpStatus::Live | OpStatus::Ready)
    }

    /// The C2 column: which ORT release this row's claim predicate was verified against, and when.
    ///
    /// `DESIGN.md` §1.4 constraint C2 requires this to exist for every contrib row and to be
    /// surfaced to users; `epctl --dump-capabilities` prints it and `tests/dump_capabilities.rs`
    /// asserts the column is there. `None` for an `ai.onnx` row is correct rather than missing —
    /// its compatibility contract is the opset window, which is frozen by ONNX and needs no
    /// verification date.
    ///
    /// The value is stored inside [`ContribSchema`] rather than in a parallel field so that a row
    /// cannot record a shape without recording where the shape came from.
    pub fn schema_baseline(&self) -> Option<SchemaBaseline> {
        self.schema.map(|s| s.baseline)
    }
}

/// Declare registry rows.
///
/// **This is the whole point of the design: adding an op is adding a row.** One line names the op,
/// its opset window, its dtype capabilities, the shader template that implements it, and the
/// shared claim/translate pair that drives it.
///
/// ```ignore
/// op_table! {
///     // op     domain  opsets           caps      kernel              claim             translate            status
///     "Add",    Ai,     7 ..= OPSET_ANY, NUMERIC,  kernel!(EwBinary, "add"),  claim::ew_binary, templates::ew_binary, Live;
///     "Sqrt",   Ai,     6 ..= OPSET_ANY, FLOAT,    kernel!(EwUnary, "sqrt"),  claim::ew_unary,  templates::ew_unary,  Staged(NO_SHADER);
/// }
/// ```
///
/// A `com.microsoft` row adds one more column — the schema fingerprint it was written against,
/// which takes the place of the opset window contrib ops do not have:
///
/// ```ignore
/// op_table! {
///     "MatMulNBits", Ms, 1 ..= OPSET_ANY, FLOAT, kernel!(None),
///         claim_matmul_nbits, templates::unimplemented, Staged(XL_KERNEL), schema: &MATMUL_NBITS;
/// }
/// ```
///
/// The lower bound of the opset window is a single token — either a literal or the name of a
/// constant, e.g. `OPSET_STD_LLM ..= OPSET_ANY` for the `ai.onnx` ops that first appear in opset
/// 23. It cannot be a general expression, because `..=` may not follow an `expr` fragment.
#[macro_export]
macro_rules! op_table {
    ($(
        $op:literal, $domain:ident, $min:tt ..= $max:expr, $caps:expr,
        $kernel:expr, $claim:path, $translate:path, $status:expr
        $(, schema: $schema:expr)?
        $(, compile: $compile:expr)?
    );* $(;)?) => {
        /// Registry rows declared by this module.
        pub static OPS: &[$crate::registry::OpSpec] = &[
            $(
                $crate::registry::OpSpec {
                    domain: $crate::registry::Domain::$domain,
                    op_type: $op,
                    min_opset: $min,
                    max_opset: $max,
                    caps: $caps,
                    kernel: $kernel,
                    claim: $claim,
                    translate: $translate,
                    status: $status,
                    schema: $crate::opt_schema!($($schema)?),
                    compile: $crate::opt_hook!($($compile)?),
                }
            ),*
        ];
    };
}

/// `Some(hook)` when a row declares a compile hook, `None` when it does not.
#[macro_export]
#[doc(hidden)]
macro_rules! opt_hook {
    () => {
        None
    };
    ($hook:expr) => {
        Some($hook)
    };
}

/// `Some(schema)` when a row declares one, `None` when it does not.
///
/// Exists because a `macro_rules!` struct literal cannot omit a field: the optional column has to
/// expand to *something*, and this is the smallest something.
#[macro_export]
#[doc(hidden)]
macro_rules! opt_schema {
    () => {
        None
    };
    ($schema:expr) => {
        Some($schema)
    };
}

/// Every op this EP knows about, in one place. Modules own their rows; this concatenates them.
///
/// `DESIGN.md` §1.3 makes conservative claiming a hard requirement, so every row here is
/// [`OpStatus::Staged`] until its shader exists — the table is complete and tested long before it
/// is live, which is exactly the ordering `OP_COVERAGE.md` §5 argues for.
pub static REGISTRY: &[&[OpSpec]] = &[
    crate::ops::elementwise::OPS,
    crate::ops::attention::OPS,
    crate::ops::norm::OPS,
    crate::ops::indexing::OPS,
    crate::ops::quant::OPS,
    crate::ops::moe::OPS,
    crate::ops::ssm::OPS,
];

/// Iterate every registered row.
pub fn all_specs() -> impl Iterator<Item = &'static OpSpec> {
    REGISTRY.iter().copied().flatten()
}

/// Look up a node's row by qualified name.
fn lookup(qualified: &str) -> Option<&'static OpSpec> {
    all_specs().find(|s| s.qualified_name() == qualified)
}

// -------------------------------------------------------------------------------------------
// Proof ledger and CLAIM_UNPROVEN escape hatch — §8.9 scaffolding
// -------------------------------------------------------------------------------------------

/// The proof key for one dispatchable op form.
///
/// Under §8.9 (`DESIGN.md`), every `Ready` row requires a matching ledger entry before it may be
/// claimed.  The key is the tuple that selects the dispatched code and the layout of what it reads:
///
/// ```text
/// (domain, op_type, opset_bucket,
///  element dtype of every input and output,
///  kernel_variant_key — including any spec-constant value that changes the emitted code,
///  shape_class ∈ {static, runtime-extent},
///  populated_optional_input_set)
/// ```
///
/// Two nodes whose keys are equal are dispatched by the same code with the same descriptor
/// layout; proof of one is proof of the other.  Any difference in the tuple corresponds to a
/// different code path and requires independent proof.
///
/// The `populated_optional_input_set` component is the field the 2026-07-30 defect would
/// have caught: `MatMulNBits` without `zero_points` (3 bindings) and with `zero_points` (4
/// bindings) have different keys and different binding arities, so proof of one cannot be
/// returned for the other.
///
/// # String representation (for `ONNXRUNTIME_EP_VULKAN_CLAIM_UNPROVEN`)
///
/// Keys are serialised as `domain::op_type/opset_bucket/dtypes/variant/shape_class/inputs`.
/// Example: `com.microsoft::MatMulNBits/1+/f16,u8,f16/qgemv_f16/runtime-extent/scales`.
/// The canonical form is emitted by the harness that generated the proof; hand-written keys
/// must match exactly.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ProofKey(pub String);

impl ProofKey {
    /// Parse from a string representation.
    pub fn parse(s: &str) -> Self {
        ProofKey(s.trim().to_owned())
    }

    /// Derive the key for one node against the row that would dispatch it.
    ///
    /// **This function is the whole of §8.7's expression-vs-path distinction.** Two nodes that
    /// differ only in an *expression* (a different constant, a different name, a different extent
    /// within the same shape class) produce the same string; two nodes that differ in a *path*
    /// (dtype, variant, shape class, which optional inputs are populated) produce different
    /// strings. There is no judgement call left in it, which is the point: the lookup is by key,
    /// so evidence about one path cannot be returned for another.
    ///
    /// The `populated_optional_input_set` component is the one that would have caught the
    /// 2026-07-30 all-zero-logits defect: `MatMulNBits` with `zero_points` and without are
    /// different bindings, so they are different keys, so a proof of one is not findable under
    /// the other.
    pub fn from_node(view: &NodeView<'_>, spec: &'static OpSpec) -> ProofKey {
        use crate::ops::common::claim::classify_shapes;
        ProofKey(format!(
            "{}::{}/{}/{}/{}/{}/{}",
            if spec.domain.as_str().is_empty() {
                "ai.onnx"
            } else {
                spec.domain.as_str()
            },
            spec.op_type,
            opset_bucket(spec),
            dtype_signature(view),
            variant_key(view, spec),
            shape_class_tag(classify_shapes(view)),
            populated_input_set(view, spec),
        ))
    }

    /// Reject values that the §8.9.4 escape hatch must never accept.
    ///
    /// §8.9.4 rule 1: "a parser that can express 'everything' must not exist", enforced as a
    /// planted test.  Any value that a careless operator could use to mean "all forms" is
    /// rejected here.  The test [`test_claim_unproven_rejects_wildcards`] plants these exact
    /// strings and asserts each returns `Err`.
    pub fn validate(s: &str) -> Result<Self, &'static str> {
        let t = s.trim();
        if t.is_empty() {
            return Err("empty key");
        }
        // §8.9.4 planted rejections — a parser that can express 'everything' must not exist.
        if t == "*" || t == "all" || t == "1" || t == "true" || t == "yes" {
            return Err("wildcard key '*/all/1/true/yes' is not a valid proof key");
        }
        // A bare op-type (no '/' separating the required fields) is not a valid key —
        // it would silently cover all forms of that op type.
        if !t.contains('/') {
            return Err("bare op-type is not a valid proof key; use the full \
                 domain::op_type/opset_bucket/dtypes/variant/shape_class/inputs form");
        }
        // The full structure, not merely "has a slash somewhere".
        //
        // Found 2026-08-01 by the separator control below, which planted a comma-split key and
        // expected every fragment to be rejected. The first fragment — `ai.onnx::Add/7+/f32` —
        // was *accepted*. A key truncated after its third component is not a narrower key; it is
        // a key that matches nothing and reads like a key that matches something, and an
        // operator who typed one would see no error and get a silent decline.
        //
        // R9 amendment 5: a validator that accepts a prefix moves with the reader's confidence —
        // it says "yes" most readily to the input the reader is least equipped to check. So the
        // structure is required outright rather than warned about.
        if !t.contains("::") {
            return Err("proof key is missing its `domain::op_type` prefix; use the full \
                 domain::op_type/opset_bucket/dtypes/variant/shape_class/inputs form");
        }
        if t.matches('/').count() != 5 {
            return Err("proof key does not have all six components; use the full \
                 domain::op_type/opset_bucket/dtypes/variant/shape_class/inputs form");
        }
        if t.split('/').any(|c| c.trim().is_empty()) {
            return Err("proof key has an empty component; an empty field is a wildcard \
                 by another name");
        }
        Ok(ProofKey(t.to_owned()))
    }
}

/// Parse the `ONNXRUNTIME_EP_VULKAN_CLAIM_UNPROVEN` environment variable.
///
/// Returns the list of explicitly enabled proof keys.  Panics at session creation (via ORT's WARN
/// log) if any key fails validation — per §8.9.4, the default-safe setting requires no act, and a
/// malformed allowlist fails loudly rather than silently enabling everything.
///
/// The env var takes a **semicolon**-separated list of full proof keys.  There is no boolean form,
/// no `=1`, and no wildcard.
///
/// # Why semicolons, and how we found out
///
/// It was comma-separated until 2026-08-01, and that was a defect the generator's attribution
/// control caught on its first real run. **A proof key contains commas** — the dtype signature is
/// `f32,f32>f32` — so a comma-separated list shredded every key into invalid fragments, the whole
/// list was correctly discarded as malformed, and the run claimed nothing while reporting a clean
/// `MATCH` against the CPU EP. The comparison was true and meant nothing: the EP had executed
/// zero nodes. Had `prove()` not demanded `claimed_nodes > 0` and `dispatches_executed > 0`, the
/// first ledger this project ever wrote would have been a set of proofs that the **CPU** EP is
/// correct — R7's fabricated negative, arriving inside the mechanism built to prevent it.
///
/// The separator is therefore chosen to be a character `ProofKey::validate` rejects inside a key,
/// and a test plants a comma-separated pair to keep it that way.
pub fn claim_unproven_keys() -> &'static [ProofKey] {
    static KEYS: std::sync::OnceLock<Vec<ProofKey>> = std::sync::OnceLock::new();
    KEYS.get_or_init(parse_claim_unproven_keys)
}

/// The parse itself, separated from the memo so a test can exercise it without a process-global.
///
/// Read once and cached: the previous shape re-read the environment and re-emitted the WARN on
/// **every node**, which on Phi-3.5 is 365 identical warnings. A disclosure that repeats 365
/// times is a disclosure a reader learns to filter, which is the opposite of what §8.9.4 item 3
/// is for.
fn parse_claim_unproven_keys() -> Vec<ProofKey> {
    let val = match std::env::var("ONNXRUNTIME_EP_VULKAN_CLAIM_UNPROVEN") {
        Ok(v) if !v.trim().is_empty() => v,
        _ => return Vec::new(),
    };
    let mut keys = Vec::new();
    for part in val.split(';') {
        match ProofKey::validate(part) {
            Ok(k) => keys.push(k),
            Err(e) => {
                // §8.9.4 item 3 — log at WARN and treat the WHOLE list as empty (safe default).
                eprintln!(
                    "[VulkanEP WARN] ONNXRUNTIME_EP_VULKAN_CLAIM_UNPROVEN contains an invalid \
                     key {:?}: {}. The entire list is ignored and all unproven forms decline.",
                    part.trim(),
                    e
                );
                return Vec::new();
            }
        }
    }
    if !keys.is_empty() {
        eprintln!(
            "[VulkanEP WARN] ONNXRUNTIME_EP_VULKAN_CLAIM_UNPROVEN is set with {} key(s): {}. \
             Unproven forms enabled for development. Do not ship this configuration.",
            keys.len(),
            keys.iter()
                .map(|k| k.0.as_str())
                .collect::<Vec<_>>()
                .join("; ")
        );
    }
    keys
}

/// The row's opset window, rendered as the key's `opset_bucket` component.
///
/// A *bucket*, not the node's `since_version`: a proof obtained at opset 14 covers opset 13 iff
/// the same row dispatches both, and the row's window is exactly the statement "these opsets take
/// the same path". A contrib row has no opset, so its window is `1..=any` and renders `1+`.
fn opset_bucket(spec: &'static OpSpec) -> String {
    if spec.max_opset == OPSET_ANY {
        format!("{}+", spec.min_opset)
    } else {
        format!("{}-{}", spec.min_opset, spec.max_opset)
    }
}

/// Element dtype of every populated input followed by every output, in slot order.
///
/// `-` for an edge whose element type ORT did not give us a `DType` for; absent optional inputs
/// contribute nothing here because they are recorded by [`populated_input_set`] instead. An f32
/// node and an f16 node therefore differ in this component, which is why an f32 proof can never
/// be returned for an f16 node.
fn dtype_signature(view: &NodeView<'_>) -> String {
    let suffix = |t: &Option<EdgeType>| -> &'static str {
        match t.as_ref().and_then(|e| e.dtype) {
            Some(d) => crate::ops::common::dtype::dtype_suffix(d),
            None => "-",
        }
    };
    let ins: Vec<&'static str> = view
        .input_types()
        .iter()
        .enumerate()
        .filter(|(i, _)| view.has_input(*i))
        .map(|(_, t)| suffix(t))
        .collect();
    let outs: Vec<&'static str> = view.output_types().iter().map(suffix).collect();
    format!("{}>{}", ins.join(","), outs.join(","))
}

/// The kernel variant that will actually be dispatched: the SPIR-V module stem.
///
/// The stem encodes template, template-op and dtype, which is the emitted code's identity. A row
/// with no shader (metadata-only, e.g. a shape op handled on the host) reports `metadata` rather
/// than an empty string, so that "no variant" is a value and not a hole.
fn variant_key(view: &NodeView<'_>, spec: &'static OpSpec) -> String {
    let dispatch_dtype = view
        .input_types()
        .iter()
        .enumerate()
        .find(|(i, t)| view.has_input(*i) && t.as_ref().and_then(|e| e.dtype).is_some())
        .and_then(|(_, t)| t.as_ref().and_then(|e| e.dtype))
        .or_else(|| {
            view.output_types()
                .first()
                .and_then(|t| t.as_ref().and_then(|e| e.dtype))
        });
    match dispatch_dtype.and_then(|d| spec.kernel.stem(d)) {
        Some(stem) if !stem.is_empty() => stem.to_string(),
        _ => "metadata".to_string(),
    }
}

/// `shape_class ∈ {static, runtime-extent}` as §8.9 spells it, over the four classes we compute.
///
/// `ExtentsSymbolic` is §8.8's runtime-extent case and renders as `runtime-extent`; the two
/// permanently-declined classes keep their own tags so that a key derived from an unclaimable
/// node is still a distinct string rather than a collision with a claimable one.
fn shape_class_tag(c: crate::ops::common::claim::ShapeClass) -> &'static str {
    use crate::ops::common::claim::ShapeClass;
    match c {
        ShapeClass::Static => "static",
        ShapeClass::ExtentsSymbolic => "runtime-extent",
        other => other.tag(),
    }
}

/// Which of an op's *optional* inputs are populated on this node.
///
/// The named table below is the part that makes this readable — `scales` and `scales+zero_points`
/// are the two `MatMulNBits` forms of the 2026-07-30 defect, and they are different strings. For
/// an op with no optional-input entry the component degrades to the populated-input **arity**,
/// which is still a path distinction (an omitted interior optional changes it) and never a
/// judgement call.
fn populated_input_set(view: &NodeView<'_>, spec: &'static OpSpec) -> String {
    let qualified = spec.qualified_name();
    if let Some(names) = optional_input_names(&qualified) {
        let mut present: Vec<&'static str> = Vec::new();
        for &(slot, name) in names {
            if slot < view.num_inputs() && view.has_input(slot) {
                present.push(name);
            }
        }
        if present.is_empty() {
            return "none".to_string();
        }
        return present.join("+");
    }
    let n = (0..view.num_inputs()).filter(|i| view.has_input(*i)).count();
    format!("n{n}")
}

/// The optional inputs of the ops that have them, by slot index, in ONNX schema order.
///
/// Hand-written and spec-literal, per row, exactly as §1.4 C4 requires of a claim predicate: a
/// generated or wildcard version of this table would be the domain-wide opt-in C1 forbids,
/// wearing a different hat. An op that is absent here is not "assumed to have no optionals" —
/// [`populated_input_set`] falls back to arity for it, which is a weaker distinction, not none.
fn optional_input_names(qualified: &str) -> Option<&'static [(usize, &'static str)]> {
    Some(match qualified {
        // A, B, scales, zero_points?, g_idx?, bias?  — the defect of 2026-07-30 is slot 3.
        "com.microsoft::MatMulNBits" => &[
            (2, "scales"),
            (3, "zero_points"),
            (4, "g_idx"),
            (5, "bias"),
        ],
        // query, key?, value?, past_key?, past_value?, seqlens_k, total_sequence_length,
        // cos_cache?, sin_cache?  — key/value absent is R5's packed-QKV form.
        "com.microsoft::GroupQueryAttention" => &[
            (1, "key"),
            (2, "value"),
            (3, "past_key"),
            (4, "past_value"),
            (7, "cos_cache"),
            (8, "sin_cache"),
        ],
        // input, skip, gamma, beta?, bias?
        "com.microsoft::SkipSimplifiedLayerNormalization"
        | "com.microsoft::SkipLayerNormalization" => &[(3, "beta"), (4, "bias")],
        // X, Scale, B?
        "LayerNormalization" | "com.microsoft::SimplifiedLayerNormalization" => &[(2, "B")],
        // input, min?, max?
        "Clip" => &[(1, "min"), (2, "max")],
        // data, indices, updates? — Pad's pads/constant_value/axes are all optional after slot 0.
        "Pad" => &[(1, "pads"), (2, "constant_value"), (3, "axes")],
        // input, roi?, scales?, sizes?
        "Resize" => &[(1, "roi"), (2, "scales"), (3, "sizes")],
        _ => return None,
    })
}

// -------------------------------------------------------------------------------------------
// The ledger itself
// -------------------------------------------------------------------------------------------

/// The proof ledger, baked into the artifact at build time.
///
/// **JSON Lines, generated, never hand-edited.** `rust/tools/gen_proof_ledger.py` writes it from
/// a differential run that obtained the evidence; `epctl --check-ledger` and
/// `rust/tools/check_proof_ledger.py` reject a file whose digest does not match its own contents
/// or whose evidence artifact has moved or changed. The first line is the header; every other
/// non-empty line is one entry.
///
/// It is `include_str!`d rather than read from disk on purpose: a ledger a running process can be
/// pointed at with an environment variable is an escape hatch that §8.9.4 does not authorise, and
/// it would let the shipped artifact disagree with the tested one.
const LEDGER_SOURCE: &str = include_str!("../../evidence/proof_ledger.jsonl");

/// One proof: a key, and the evidence that proved it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LedgerEntry {
    /// The proof key this entry proves.
    pub key: ProofKey,
    /// The device the differential ran on, e.g. `NVIDIA GeForce RTX 4060 Laptop GPU`.
    pub device: String,
    /// The ONNX Runtime build the differential ran against.
    pub ort_build: String,
    /// The tolerance policy applied to the comparison.
    pub tolerance: String,
    /// The artifact or builder the case came from.
    pub artifact: String,
    /// The model-level verdict of the run that produced this entry. Only `MATCH` is admissible;
    /// a `DIVERGENT` run demotes rather than proves (§8.9.2 rule 4), so the generator writes no
    /// entry at all for one.
    pub verdict: String,
    /// When the evidence was obtained.
    pub generated_at: String,
    /// **Provenance witness — how many nodes this EP claimed on the proof run.**
    ///
    /// RAI-008(a). The cheapest way to satisfy criterion 11 is to generate the ledger from the
    /// same enumeration that produces the claims: then `ledger_hits == proven_key_lookups`
    /// forever, the check can never fail, and `6/6` reads identically under both stories. The
    /// defence against that is not a promise, it is **a field the claim table cannot produce**.
    /// This one comes from the EP's execution counters after a session ran.
    pub claimed_nodes: u64,
    /// **Provenance witness — dispatches this EP executed on the proof run.**
    ///
    /// See [`LedgerEntry::claimed_nodes`]. Zero is the 2026-07-30 specimen: a comparison that
    /// returned `MATCH` while ORT ran everything on CPU. `parse_ledger` faults any entry whose
    /// witnesses are absent or zero, so a table-derived ledger grants **no** claims rather than
    /// granting them quietly.
    pub dispatches_executed: u64,
}

/// The parsed ledger plus the header facts a checker needs.
#[derive(Debug)]
pub struct Ledger {
    entries: Vec<LedgerEntry>,
    /// The header's declared entry count.
    pub declared_count: usize,
    /// The header's declared content digest.
    pub declared_digest: String,
    /// The digest recomputed over the entry lines actually present.
    pub actual_digest: String,
    /// The generator that wrote it.
    pub generator: String,
    /// Parse or consistency problems. Non-empty means the ledger is not usable and every form
    /// declines — a broken ledger is the safe state, not the permissive one.
    pub faults: Vec<String>,
}

impl Ledger {
    /// Entries, in file order.
    pub fn entries(&self) -> &[LedgerEntry] {
        &self.entries
    }

    /// How many proofs it holds.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether it holds no proofs at all. **This is the state that makes a ledger *hit*
    /// `UNOBSERVABLE` rather than `0`** (R12): with no entries baked in, a hit could not have
    /// occurred in this frame, so a count of zero hits is not a measurement of anything.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether the digest in the header matches the entries under it.
    pub fn digest_ok(&self) -> bool {
        self.declared_digest == self.actual_digest
    }

    /// Look one key up.
    pub fn get(&self, key: &ProofKey) -> Option<&LedgerEntry> {
        if !self.faults.is_empty() {
            return None;
        }
        self.entries.iter().find(|e| &e.key == key)
    }
}

/// FNV-1a/64 over the ledger's entry lines, matching `rust/tools/gen_proof_ledger.py`.
///
/// A checksum, not a signature, and the distinction is recorded rather than smoothed: it catches
/// the careless hand-edit, which is the failure §8.9.2 rule 3 names. It does **not** catch a
/// deliberate forgery, because anyone who can edit the file can recompute it. The defence against
/// that is `check_proof_ledger.py`, which re-hashes each entry's evidence artifact — an entry
/// whose artifact does not exist or does not match is rejected there.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Pull one `"field": "value"` string out of a JSON object line.
///
/// Deliberately small: the ledger is written by our own generator in a fixed shape, so a full
/// JSON parser would be a dependency bought to read a file we emit. Handles the escapes
/// `claim_log::escape` can produce and nothing else; anything unexpected surfaces as a fault
/// rather than as a silently-empty field.
/// Read a **numeric** JSON field. Returns `None` when the field is absent or is not a bare
/// integer — a string `"3"` is not an integer, because a writer that quotes its counters is a
/// writer whose counters did not come from a counter.
fn json_u64_field(line: &str, field: &str) -> Option<u64> {
    let needle = format!("\"{field}\":");
    let start = line.find(&needle)? + needle.len();
    let rest = line[start..].trim_start();
    let end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    if end == 0 {
        return None;
    }
    rest[..end].parse().ok()
}

fn json_field(line: &str, field: &str) -> Option<String> {    let needle = format!("\"{field}\":");
    let start = line.find(&needle)? + needle.len();
    let rest = line[start..].trim_start();
    let mut chars = rest.chars();
    if chars.next()? != '"' {
        return None;
    }
    let mut out = String::new();
    let mut escaped = false;
    for c in chars {
        if escaped {
            out.push(match c {
                'n' => '\n',
                't' => '\t',
                'r' => '\r',
                other => other,
            });
            escaped = false;
        } else if c == '\\' {
            escaped = true;
        } else if c == '"' {
            return Some(out);
        } else {
            out.push(c);
        }
    }
    None
}

/// Parse the baked-in ledger source.
fn parse_ledger(source: &str) -> Ledger {
    let mut faults: Vec<String> = Vec::new();
    let mut lines = source
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'));

    let header = lines.next().unwrap_or("");
    let generator = json_field(header, "generator").unwrap_or_default();
    let declared_digest = json_field(header, "content_fnv1a64").unwrap_or_default();
    let declared_count: usize = json_field(header, "entry_count")
        .or_else(|| {
            // `entry_count` is a number, not a string.
            let needle = "\"entry_count\":";
            let start = header.find(needle)? + needle.len();
            let rest = header[start..].trim_start();
            let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
            Some(rest[..end].to_string())
        })
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0);
    if generator.is_empty() {
        faults.push("ledger header names no generator".to_string());
    }

    let mut entries = Vec::new();
    let mut digest_input = String::new();
    for line in lines {
        digest_input.push_str(line);
        digest_input.push('\n');
        let Some(raw_key) = json_field(line, "key") else {
            faults.push(format!("ledger line has no `key` field: {line}"));
            continue;
        };
        let key = match ProofKey::validate(&raw_key) {
            Ok(k) => k,
            Err(e) => {
                faults.push(format!("ledger key {raw_key:?} is not a valid proof key: {e}"));
                continue;
            }
        };
        let verdict = json_field(line, "verdict").unwrap_or_default();
        if verdict != "MATCH" {
            // §8.9.2 rule 4: only a MATCH proves. Anything else in this file is a generator bug
            // or a hand-edit, and either way the entry does not get to grant a claim.
            faults.push(format!(
                "ledger entry for {raw_key:?} carries verdict {verdict:?}; only MATCH proves"
            ));
            continue;
        }
        // RAI-008(a) — provenance. An entry must carry the witnesses of a **proof run**. The
        // claim table can enumerate keys; it cannot produce a dispatch count, because dispatches
        // only exist after a session executed. Absent is treated exactly like zero: both mean
        // this entry does not record a run that ran, and an unattributed MATCH is the specimen
        // this project has now produced twice.
        let claimed_nodes = json_u64_field(line, "claimed_nodes");
        let dispatches_executed = json_u64_field(line, "dispatches_executed");
        let (Some(claimed_nodes), Some(dispatches_executed)) = (claimed_nodes, dispatches_executed)
        else {
            faults.push(format!(
                "ledger entry for {raw_key:?} carries no attribution witness \
                 (claimed_nodes/dispatches_executed); it does not record a proof run and may \
                 have been enumerated from the claim table rather than proven"
            ));
            continue;
        };
        if claimed_nodes == 0 || dispatches_executed == 0 {
            faults.push(format!(
                "ledger entry for {raw_key:?} records claimed_nodes={claimed_nodes} \
                 dispatches_executed={dispatches_executed}; a run that claimed or dispatched \
                 nothing is UNATTRIBUTED and proves nothing"
            ));
            continue;
        }
        entries.push(LedgerEntry {
            key,
            device: json_field(line, "device").unwrap_or_default(),
            ort_build: json_field(line, "ort_build").unwrap_or_default(),
            tolerance: json_field(line, "tolerance").unwrap_or_default(),
            artifact: json_field(line, "artifact").unwrap_or_default(),
            verdict,
            generated_at: json_field(line, "generated_at").unwrap_or_default(),
            claimed_nodes,
            dispatches_executed,
        });
    }

    let actual_digest = format!("{:016x}", fnv1a64(digest_input.as_bytes()));
    if !declared_digest.is_empty() && declared_digest != actual_digest {
        faults.push(format!(
            "ledger digest mismatch: header declares {declared_digest}, contents hash to \
             {actual_digest}. The ledger has been hand-edited; regenerate it with \
             rust/tools/gen_proof_ledger.py"
        ));
    }
    if declared_count != entries.len() && faults.is_empty() {
        faults.push(format!(
            "ledger header declares {declared_count} entries, {} parsed",
            entries.len()
        ));
    }
    // Duplicate keys are a generator fault, not a merge convenience: two proofs of one key
    // disagreeing about their evidence is exactly the fork R7 forbids.
    for (i, e) in entries.iter().enumerate() {
        if entries[..i].iter().any(|p| p.key == e.key) {
            faults.push(format!("ledger contains duplicate key {}", e.key.0));
        }
    }

    Ledger {
        entries,
        declared_count,
        declared_digest,
        actual_digest,
        generator,
        faults,
    }
}

/// The environment variable naming the on-disk ledger to check the baked copy against.
///
/// Optional. When unset, the baked ledger is still self-checked (header digest vs. baked body),
/// which catches a hand-edit before the build. This variable catches the *other* threat: the file
/// on disk changing after the build, so that the artifact a reviewer reads is not the artifact
/// the running binary claims from.
pub const ENV_LEDGER_FILE: &str = "ONNXRUNTIME_EP_VULKAN_LEDGER_FILE";

/// Compare the baked ledger's digest against the ledger on disk, and **fault on disagreement**.
///
/// RAI-008(b), digest half. The requirement is explicit that this **refuses to claim rather than
/// warns**, and the mechanism for that is that the disagreement is pushed into
/// [`Ledger::faults`] — a non-empty `faults` makes [`Ledger::get`] return `None` for every key,
/// so every form declines. A WARN would leave the run claiming on evidence nobody can read.
///
/// R9 amendment 5 — ask which way this check moves when its subject is wrong. It moves *against*
/// the reader's confidence: a mismatch removes claims. A check that granted claims on a mismatch,
/// or that could be silenced by unsetting a variable it needs to be set, would be the wrong sign;
/// this one is only *stronger* when the variable is set, never weaker.
fn check_baked_against_disk(baked: &Ledger) -> Option<String> {
    let path = std::env::var(ENV_LEDGER_FILE).ok()?;
    let path = path.trim();
    if path.is_empty() {
        return None;
    }
    let disk = match std::fs::read_to_string(path) {
        Ok(d) => d,
        Err(e) => {
            return Some(format!(
                "{ENV_LEDGER_FILE} names {path}, which cannot be read: {e}. A ledger that was \
                 asked for and is absent is not an empty ledger; refusing to claim."
            ));
        }
    };
    let disk_ledger = parse_ledger(&disk);
    if disk_ledger.actual_digest == baked.actual_digest {
        return None;
    }
    Some(format!(
        "baked ledger digest {} disagrees with {path} at digest {}. The binary would claim from \
         evidence that is not the evidence on disk. Refusing to claim; rebuild, or regenerate \
         with rust/tools/gen_proof_ledger.py.",
        baked.actual_digest, disk_ledger.actual_digest
    ))
}

/// Why a ledger lookup did not produce a proof. **Three findings, not one** (R13).
///
/// A single `bool` collapsed three different situations into one `false`, and they call for three
/// different actions: regenerate the ledger for this form, fix the ledger file, and nothing at all
/// respectively. `NeverAttempted` is not returned by [`lookup_key`] — it is what the *counters*
/// report when `proven_key_lookups` is `0`, and it exists in this enum so the vocabulary is one
/// vocabulary rather than two.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LedgerLookup {
    /// The ledger parsed and holds a proof for this key.
    Hit,
    /// The ledger parsed and holds no proof for this key. **Regenerate the ledger for this form.**
    KeyAbsent,
    /// The ledger did not parse, or its digest disagrees with the header or the disk. Every form
    /// declines. **This is an instrument failure, not a finding about the form** (R13) — the key
    /// might well be proven, and this build cannot tell.
    Faulted,
    /// No lookup was attempted for this key in this frame. Distinct from `KeyAbsent`: nothing was
    /// asked, so nothing was answered.
    NeverAttempted,
}

impl LedgerLookup {
    /// The single token an artifact records.
    pub fn token(&self) -> &'static str {
        match self {
            LedgerLookup::Hit => "HIT",
            LedgerLookup::KeyAbsent => "KEY-ABSENT",
            LedgerLookup::Faulted => "LEDGER-FAULTED",
            LedgerLookup::NeverAttempted => "NEVER-ATTEMPTED",
        }
    }
}

/// Look one key up and say **which** of the three answers it got.
pub fn lookup_key(key: &ProofKey) -> LedgerLookup {
    let l = ledger();
    if !l.faults.is_empty() {
        return LedgerLookup::Faulted;
    }
    if l.entries.iter().any(|e| &e.key == key) {
        LedgerLookup::Hit
    } else {
        LedgerLookup::KeyAbsent
    }
}

/// The process-wide ledger, parsed once.
pub fn ledger() -> &'static Ledger {
    static LEDGER: std::sync::OnceLock<Ledger> = std::sync::OnceLock::new();
    LEDGER.get_or_init(|| {
        let mut l = parse_ledger(LEDGER_SOURCE);
        if let Some(mismatch) = check_baked_against_disk(&l) {
            l.faults.push(mismatch);
        }
        for f in &l.faults {
            log::warn!("[VulkanEP] proof ledger fault: {f}");
        }
        l
    })
}

/// Whether the ledger holds a proof for this key.
///
/// Wired into [`claim_audit`]; the counter it feeds is `proven_key_lookups`, which is what makes
/// this mechanism observable rather than merely present (R10).
pub fn ledger_contains(key: &ProofKey) -> bool {
    lookup_key(key) == LedgerLookup::Hit
}

///
/// `Ok(())` means claim it. `Err(reason)` means leave it to the CPU EP and report `reason` under
/// claim-debug. There is no third answer and no per-op logic anywhere above this function.
///
/// Every decision also goes to [`crate::ops::claim_log`] when
/// `ONNXRUNTIME_EP_VULKAN_CLAIM_LOG` names a file, which is how a test outside the process asserts
/// *why* a node was declined rather than merely that it was. That is recorded here rather than in
/// `ep.rs` so that the reason survives `ep.rs`'s per-op-type aggregation, which keeps only the
/// first reason per op type.
pub fn claim_decision(view: &NodeView<'_>) -> Result<(), DeclineReason> {
    let logging = crate::ops::claim_log::enabled();
    let audit = claim_audit(view, logging);
    if logging {
        crate::ops::claim_log::record_audit(
            &view.qualified_name(),
            &view.name(),
            view.since_version(),
            &audit,
        );
    }
    audit.decision()
}

/// The decision itself, plus **every** check that failed rather than only the first.
///
/// # Why this is not first-match
///
/// `DESIGN.md` R8: *a decline code names the first failing check, not the only one — early codes
/// are ceilings, late codes are floors, and two decline counts are not comparable without knowing
/// the check order.* The Phi-3.5 census learned that the hard way: `staged: 100` was read as "100
/// nodes that only need a kernel", when in fact those nodes were rejected at the status check and
/// **never reached the shape check at all**, so their shape viability was simply unknown. Meanwhile
/// `dynamic-shape: 258` had already passed registration, opset, schema and status, making it the
/// only number in that histogram that was not an upper bound.
///
/// A first-match histogram looks exactly like a complete one, which is why this is fixed in the
/// producer rather than in each consumer.
///
/// # What is recorded
///
/// * `primary` — the first failure in canonical check order. Unchanged semantics, so every
///   existing assertion on `code` keeps its meaning.
/// * `failures` — every check that failed, in the same order.
/// * `unevaluated` — checks that could not run because an earlier one made them meaningless.
///   Only non-empty when the op has no row at all: without a row there is no opset window, no
///   schema, no status and no predicate to ask.
/// * `shape_class` — computed from the node's edges **independently of its row**, which is what
///   makes a staged node's shape viability knowable at all.
/// * `predicate_ok` / `predicate_ok_with_runtime_extents` — the row's own predicate, asked twice:
///   as things are, and counterfactually with extents assumed to arrive at `Compute`. The
///   difference between the two is the measured answer to "what does the runtime-extent work
///   unlock", asked of the predicate itself rather than of a re-implementation of it.
#[derive(Debug, Clone)]
pub struct ClaimAudit {
    pub primary: Option<DeclineReason>,
    pub failures: Vec<DeclineReason>,
    pub unevaluated: Vec<&'static str>,
    pub shape_class: crate::ops::common::claim::ShapeClass,
    pub predicate_ok: bool,
    pub predicate_ok_with_runtime_extents: bool,
    /// The §8.9 proof key for this node, or `None` when the op has no row at all (there is no
    /// row to derive a variant or an opset bucket from, so there is no key — which is a third
    /// state, not an empty string).
    pub proof_key: Option<ProofKey>,
    /// Whether the ledger held a proof under that key. Always `false` for a `Staged` row, which
    /// is not a finding about evidence: the lookup does not happen because a row with no kernel
    /// cannot be claimed on any evidence.
    pub ledger_hit: bool,
}

impl ClaimAudit {
    /// The claim answer. Identical to what the old first-match path returned.
    pub fn decision(&self) -> Result<(), DeclineReason> {
        match &self.primary {
            None => Ok(()),
            Some(r) => Err(r.clone()),
        }
    }
}

/// Run every check and report all of them.
///
/// The order of checks is deliberate — key, then opset, then contrib schema, then status, then the
/// predicate — so that `primary` attributes a node to the *first* thing that is wrong with it. The
/// schema check runs *before* the staged check on purpose: "the contrib schema moved under us" is
/// a signal we want visible even while the kernel behind the row is still being written, because
/// it invalidates the row itself rather than merely deferring it.
///
/// `with_counterfactual` is `false` on the hot path: the second predicate evaluation is only worth
/// paying for when something is listening.
pub fn claim_audit(view: &NodeView<'_>, with_counterfactual: bool) -> ClaimAudit {
    use crate::ops::common::claim::{AssumeRuntimeExtents, classify_shapes};

    let qualified = view.qualified_name();
    let shape_class = classify_shapes(view);

    let Some(spec) = lookup(&qualified) else {
        let reason = decline(
            DeclineCode::NotRegistered,
            format_args!(
                "no Vulkan handler is registered for `{qualified}` (opset {})",
                view.since_version()
            ),
        );
        return ClaimAudit {
            primary: Some(reason.clone()),
            failures: vec![reason],
            unevaluated: vec!["opset", "contrib-schema", "status", "predicate", "ledger"],
            shape_class,
            predicate_ok: false,
            predicate_ok_with_runtime_extents: false,
            proof_key: None,
            ledger_hit: false,
        };
    };

    let mut failures: Vec<DeclineReason> = Vec::new();

    let since = view.since_version();
    if since != 0 && (since < spec.min_opset || since > spec.max_opset) {
        let upper = if spec.max_opset == OPSET_ANY {
            "any".to_string()
        } else {
            spec.max_opset.to_string()
        };
        failures.push(decline(
            DeclineCode::Opset,
            format_args!(
                "`{qualified}` opset {since} is outside the supported window {}..={upper}",
                spec.min_opset
            ),
        ));
    }

    if let Some(schema) = spec.schema {
        if let Err(e) = schema.check(view, &qualified) {
            failures.push(e);
        }
    }

    if let OpStatus::Staged(blocker) = spec.status {
        failures.push(decline(
            DeclineCode::Staged,
            format_args!(
                "`{qualified}` is in the op table but not enabled: {blocker}. It runs on the CPU \
                 EP, which is always correct"
            ),
        ));
    }

    let predicate = (spec.claim)(view, spec);
    let predicate_ok = predicate.is_ok();
    if let Err(e) = predicate {
        failures.push(e);
    }

    // --- §8.9: the proof-ledger gate ---
    //
    // Last, deliberately. A node that has no kernel, the wrong opset or a shape we cannot handle
    // is those things first; reporting it as `[unproven]` would make the decline histogram (R8)
    // say "we need evidence" where it should say "we need a kernel", and R8's whole point is that
    // the histogram decides what gets built next.
    //
    // It runs for **every** node with a row, including a `Staged` one, so that `proof_key` is
    // recorded in the claim log whether or not the node was claimable — that log is what
    // `gen_proof_ledger.py` reads to learn which keys a run would need. A key computed only for
    // nodes that already pass is a key that can never bootstrap.
    let proof_key = ProofKey::from_node(view, spec);
    let ledger_hit = if spec.is_live() {
        // RAI-008(d): the outcome, not a `bool`. A miss that was the ledger failing and a miss
        // that was this form being absent are two findings with two different repairs, and the
        // counters artifact has to be able to say which one happened.
        let outcome = lookup_key(&proof_key);
        crate::counters::record_ledger_lookup(outcome);
        outcome == LedgerLookup::Hit
    } else {
        false
    };
    let hatch = if spec.is_live() && !ledger_hit {
        let enabled = claim_unproven_keys().contains(&proof_key);
        if enabled {
            crate::counters::record_unproven_form_enabled(&proof_key.0);
        }
        enabled
    } else {
        false
    };
    if spec.is_live() && !ledger_hit && !hatch {
        crate::counters::record_unproven_decline();
        failures.push(decline(
            DeclineCode::Unproven,
            format_args!(
                "no proof ledger entry for `{}`. The kernel exists; nothing has proven it \
                 correct on this form, so it runs on the CPU EP, which is always right. Prove it \
                 with rust/tools/gen_proof_ledger.py, or enable it for development with \
                 ONNXRUNTIME_EP_VULKAN_CLAIM_UNPROVEN={}",
                proof_key.0, proof_key.0
            ),
        ));
    }

    let predicate_ok_with_runtime_extents = if !with_counterfactual {
        predicate_ok
    } else if predicate_ok {
        true
    } else {
        let _guard = AssumeRuntimeExtents::on();
        (spec.claim)(view, spec).is_ok()
    };

    ClaimAudit {
        primary: failures.first().cloned(),
        failures,
        unevaluated: Vec::new(),
        shape_class,
        predicate_ok,
        predicate_ok_with_runtime_extents,
        proof_key: Some(proof_key),
        ledger_hit,
    }
}

/// Convenience wrapper for the boolean question.
pub fn claimable(view: &NodeView<'_>) -> bool {
    claim_decision(view).is_ok()
}

/// The same decision, taken against an already-extracted [`NodeDesc`].
///
/// `Compile` uses this to re-check every node it is about to translate: a node that was claimed in
/// `GetCapability` but is not registered here is an internal invariant violation, not a user error
/// (`DESIGN.md` §5.5 step 2). Staged rows deliberately answer `false`.
pub fn is_registered(desc: &NodeDesc) -> bool {
    lookup(&desc.qualified_name()).is_some_and(OpSpec::is_live)
}

/// The row for an already-extracted node, for `Compile` to translate through.
pub fn spec_for(desc: &NodeDesc) -> Option<&'static OpSpec> {
    lookup(&desc.qualified_name()).filter(|s| s.is_live())
}

/// A compile hook: a pure function from op spec and node to zero or more prepack requests.
///
/// Returned by [`compile_hook_for`]; called once per live node during `Compile`, after device
/// selection. Ops that do not need weight prepacking return `None` from `compile_hook_for`; the
/// engine skips the call entirely.
///
/// **Mouse fills in the non-`None` paths.** Switch owns the dispatch machinery. The types here
/// must match [`crate::engine::CompileContext`].
pub type CompileHook =
    fn(spec: &OpSpec, desc: &NodeDesc, ctx: &mut dyn crate::engine::CompileContext) -> EpResult<()>;

/// Return the compile hook for this node, if any.
///
/// `None` means "nothing to do at Compile time for this op" — the engine can skip the call.
/// `Some(hook)` means the engine must call `hook(spec, desc, ctx)` before the first `Compute`.
///
/// Resolved **through the op table**, not through a match here. Switch's original stub was a free
/// function keyed on the node, which works but would become a second dispatch surface that a new
/// op has to be added to — and the whole argument of `OP_COVERAGE.md` §5.7 is that there must be
/// exactly one. So the hook is the row's optional `compile:` column and this function is a lookup.
/// The engine-facing signature is unchanged.
///
/// Only live rows are consulted: a staged op is never claimed, so nothing of it reaches `Compile`.
pub fn compile_hook_for(desc: &NodeDesc) -> Option<CompileHook> {
    spec_for(desc).and_then(|s| s.compile)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_live_row_lacks_a_shader_or_dispatch_path() {
        // M0 exit: the first rows flipped Live have shaders that have executed on real hardware.
        // This test now verifies the positive constraint: every Live row has a compiled shader
        // variant and a translate handler, rather than enforcing that no rows are Live.
        use crate::engine::shaders;
        let live: Vec<&'static OpSpec> = all_specs().filter(|s| s.is_live()).collect();
        // At least Add must be live — if this ever goes back to zero we likely regressed.
        assert!(
            !live.is_empty(),
            "no rows are Live; the ORT dispatch wire landed with Add and must stay live"
        );
        for spec in &live {
            let name = spec.qualified_name();
            // Round-tripping through `NodeDesc` proves the *lookup path* works for this row, not
            // just that the row exists. The domain has to come from the row rather than from
            // splitting the qualified name: a contrib row reached through the default domain is a
            // different lookup, and `MatMulNBits` is the first live row where it differs.
            let desc = NodeDesc {
                op_type: spec.op_type.into(),
                domain: spec.domain.as_str().into(),
                ..Default::default()
            };
            assert!(
                spec_for(&desc).is_some(),
                "{name} is live but spec_for returns None — is_live() and spec_for are inconsistent"
            );
            // Every live row must have a dispatch path.  There are two kinds:
            //
            //   (a) Template rows: spec.kernel.template != None. The kernel system generates
            //       shader variants and `spec.kernel.stem(d)` returns their names. We verify
            //       that at least one variant is present in the compiled binary.
            //
            //   (b) Direct-shader rows: spec.kernel.template == None. The translate handler
            //       dispatches to a hand-written shader by hardcoded stem (e.g. skip_norm).
            //       These rows have no manifest entry, so the kernel.stem() check is the
            //       wrong instrument. We verify instead that the translate handler is NOT
            //       `templates::unimplemented` — proving that *some* dispatch path exists,
            //       even though we cannot see the shader name from here. The per-translate-
            //       handler unit tests (in ops::common::templates::tests) then verify the
            //       stem is correct and the shader exists on disk.
            //
            // The two instruments are complementary: (a) proves the binary has the shader;
            // (b) proves the dispatch path is non-trivial. Both are required for their class.
            use crate::ops::common::templates;
            use crate::ops::common::variants::Template;
            let has_dispatch = if spec.kernel.template == Template::None {
                // Direct-shader row: translate handler must be non-trivial.
                !std::ptr::fn_addr_eq(spec.translate, templates::unimplemented as fn(_, _, _) -> _)
            } else {
                // Template row: at least one compiled shader variant must be present.
                spec.caps.iter().any(|d| {
                    spec.kernel
                        .stem(d)
                        .is_some_and(|stem| shaders::find(stem).is_some())
                })
            };
            assert!(
                has_dispatch,
                "{name} is live but has no compiled shader variant or non-trivial translate handler"
            );
        }
    }

    /// "The latest opset" has two correct answers, and the difference matters.
    ///
    /// `onnx.defs.onnx_opset_version()` returns 27 on onnx 1.22.0, while `schema.h` still records
    /// `last_release_version_map_[ONNX_DOMAIN] = 26`. Justin's "26" is the release field; the 27 the
    /// coordinator measured is the registered maximum. Both are true; leaving it implicit is how a
    /// moving fact rots. Recorded in `OP_COVERAGE.md` §4.20 and pinned here.
    #[test]
    // Both operands are `const`, so clippy calls the comparison constant. That is precisely the
    // point: this test exists to fail the build the day someone edits one of the two constants
    // and collapses the distinction the doc comment above describes.
    #[allow(clippy::assertions_on_constants)]
    fn the_latest_opset_has_two_answers_and_both_are_recorded() {
        assert_eq!(ONNX_OPSET_LAST_RELEASED, 26);
        assert_eq!(ONNX_OPSET_REGISTERED, 27);
        assert!(
            ONNX_OPSET_REGISTERED > ONNX_OPSET_LAST_RELEASED,
            "if these ever converge, a release closed opset 27 and every window bounded by a \
             'newest schema version' claim needs re-reading"
        );
        assert!(
            ONNX_SPEC_READ.contains("27") && ONNX_SPEC_READ.contains("26"),
            "the provenance string is quoted in `[opset]` declines; it has to name both numbers \
             or the decline is unactionable"
        );
    }

    /// No standard-domain row may be bounded above by an opset nobody has read.
    ///
    /// A closed window is a promise that somebody checked the op's schema history up to that bound.
    /// A bound above [`ONNX_OPSET_REGISTERED`] would be a promise about a schema that does not
    /// exist yet — the same defect as an open-ended window, spelled differently. §4.20.
    #[test]
    fn closed_standard_windows_stay_within_the_spec_we_have_read() {
        for s in all_specs().filter(|s| s.domain == Domain::Ai && s.max_opset != OPSET_ANY) {
            assert!(
                s.max_opset <= ONNX_OPSET_REGISTERED,
                "{} is bounded at opset {} but only {} is registered in {ONNX_SPEC_READ}",
                s.qualified_name(),
                s.max_opset,
                ONNX_OPSET_REGISTERED
            );
            assert!(
                s.min_opset <= s.max_opset,
                "{} has an empty window",
                s.qualified_name()
            );
        }
    }

    #[test]
    fn the_table_is_not_empty() {
        // The point of `Staged` is that the table lands and is tested before the shaders exist.
        assert!(
            all_specs().count() >= 60,
            "the tier-1 elementwise table is missing"
        );
    }

    #[test]
    fn registry_keys_are_unique() {
        let mut keys: Vec<String> = all_specs()
            .map(|s| s.qualified_name().into_owned())
            .collect();
        keys.sort_unstable();
        let before = keys.len();
        keys.dedup();
        assert_eq!(before, keys.len(), "duplicate op key in REGISTRY");
    }

    #[test]
    fn every_contrib_row_declares_a_schema() {
        // Justin admitted `com.microsoft` on 2026-07-28, and the obligation that came with it is
        // this: contrib ops have no opset guarantee, so a row that does not say what schema it was
        // written against cannot detect that the schema moved.
        for s in all_specs().filter(|s| s.domain == Domain::Ms) {
            assert!(
                s.schema.is_some(),
                "contrib row `{}` declares no ContribSchema fingerprint",
                s.qualified_name()
            );
        }
    }

    /// The opset window is the compatibility statement for `ai.onnx` — except for the ops that
    /// have no ONNX schema at all. See [`ORT_FUSED_IN_DEFAULT_DOMAIN`].
    #[test]
    fn only_ort_fused_default_domain_rows_carry_a_fingerprint() {
        for s in all_specs().filter(|s| s.domain == Domain::Ai && s.schema.is_some()) {
            assert!(
                ORT_FUSED_IN_DEFAULT_DOMAIN.contains(&s.op_type),
                "`{}` is ai.onnx with a contrib fingerprint but is not on the ORT-fused \
                 allow-list; ai.onnx rows are versioned by their opset window",
                s.op_type
            );
            assert_eq!(
                (s.min_opset, s.max_opset),
                (1, OPSET_ANY),
                "`{}` has no ONNX schema, so its opset window carries no information and must \
                 not pretend to — the fingerprint is the compatibility statement",
                s.op_type
            );
        }
    }

    /// Conversely: an `ai.onnx` row *not* on the allow-list must have no fingerprint.
    #[test]
    fn ordinary_ai_onnx_rows_use_their_opset_window_not_a_fingerprint() {
        for s in all_specs()
            .filter(|s| s.domain == Domain::Ai && !ORT_FUSED_IN_DEFAULT_DOMAIN.contains(&s.op_type))
        {
            assert!(
                s.schema.is_none(),
                "`{}` is ai.onnx and must use its opset window, not a contrib fingerprint",
                s.op_type
            );
        }
    }

    /// The allow-list is a hazard register, not a convenience. Keep it small and evidenced.
    #[test]
    fn the_default_domain_fusion_allow_list_is_evidenced_and_small() {
        assert!(
            ORT_FUSED_IN_DEFAULT_DOMAIN.len() <= 4,
            "this list grows only when a real graph is observed emitting the op with an empty \
             domain; if it is getting long, the registry needs a third domain concept instead"
        );
        for op in ORT_FUSED_IN_DEFAULT_DOMAIN {
            assert!(
                all_specs().any(|s| s.op_type == *op && s.domain == Domain::Ai),
                "`{op}` is on the default-domain fusion list but has no ai.onnx row, so the \
                 list is documenting a hazard we do not actually handle"
            );
        }
    }

    #[test]
    fn contrib_schemas_are_self_consistent() {
        for s in all_specs() {
            let Some(schema) = s.schema else { continue };
            let name = s.qualified_name();
            assert!(
                schema.min_inputs <= schema.max_inputs,
                "{name}: input range is inverted"
            );
            assert!(
                schema.min_outputs <= schema.max_outputs && schema.min_outputs >= 1,
                "{name}: output range is inverted or empty"
            );
            assert!(
                !schema.baseline.verified_on.is_empty(),
                "{name}: schema does not say when it was verified (DESIGN.md §1.4 C2)"
            );
            for required in schema.required_attrs {
                assert!(
                    schema.knows(required),
                    "{name}: required attr `{required}` is missing from known_attrs, so a \
                     conforming node would be declined as schema drift"
                );
            }
            let mut known: Vec<_> = schema.known_attrs.to_vec();
            known.sort_unstable();
            let before = known.len();
            known.dedup();
            assert_eq!(before, known.len(), "{name}: duplicate name in known_attrs");
        }
    }

    #[test]
    fn contrib_schema_check_is_a_pure_function_of_the_fingerprint() {
        // Exercised without a graph: the arity half of `check` is testable directly, and the
        // attribute half is covered by the per-op tests in `ops::quant` and friends.
        let schema = ContribSchema {
            baseline: PINNED_BASELINE,
            notes: "",
            min_inputs: 3,
            max_inputs: 6,
            min_outputs: 1,
            max_outputs: 1,
            required_attrs: &["K", "N"],
            known_attrs: &["K", "N", "bits", "block_size"],
        };
        assert!(schema.knows("block_size"));
        assert!(!schema.knows("weight_prepacked"));
    }

    #[test]
    fn contrib_rows_are_the_single_source_of_the_c2_baseline() {
        // Reconciled with Tank, 2026-07-28: `sys` owns the *type* (`SchemaBaseline`,
        // `OrtRelease`) and the pinned/floor releases; the *data* lives here, inside the schema
        // fingerprint, so a shape cannot be recorded without its provenance. The side table in
        // `sys.rs` that this test used to cross-check has been removed rather than reconciled —
        // two places recording the same fact was the drift hazard, and deleting one is a better
        // answer than testing that they agree. `epctl --dump-capabilities` now reads
        // `OpSpec::schema_baseline()` directly, and `tests/layering.rs` enforces that every
        // contrib row has one and no default-domain row does.
        for s in all_specs().filter(|s| s.domain == Domain::Ms) {
            let key = s.qualified_name();
            let own = s
                .schema_baseline()
                .unwrap_or_else(|| panic!("`{key}` has no schema baseline"));
            assert!(
                !own.verified_on.is_empty(),
                "`{key}` has no verification date"
            );
        }
    }

    #[test]
    fn every_row_has_a_sane_opset_window() {
        for s in all_specs() {
            assert!(
                s.min_opset >= 1 && s.min_opset <= s.max_opset,
                "{} has window {}..={}",
                s.op_type,
                s.min_opset,
                s.max_opset
            );
        }
    }

    #[test]
    fn every_row_declares_at_least_one_dtype() {
        for s in all_specs() {
            assert!(!s.caps.is_empty(), "{} declares no dtypes", s.op_type);
        }
    }

    #[test]
    fn unknown_ops_are_declined_with_a_reason() {
        assert!(lookup("NoSuchOp").is_none());
    }

    #[test]
    fn unregistered_node_desc_is_not_registered() {
        let desc = NodeDesc {
            op_type: "NoSuchOp".into(),
            ..Default::default()
        };
        assert!(!is_registered(&desc));
    }

    #[test]
    fn live_rows_are_registered_for_translation_and_staged_rows_are_not() {
        // A live row must reach Compile: claimed and translatable stay identical.
        let add_desc = NodeDesc {
            op_type: "Add".into(),
            ..Default::default()
        };
        assert!(
            lookup(&add_desc.qualified_name()).is_some(),
            "Add should be in the table"
        );
        assert!(
            is_registered(&add_desc),
            "Add is live and must be translatable"
        );
        assert!(spec_for(&add_desc).is_some());

        // A staged row must never reach Compile. Picked from the table rather than named, because
        // naming one meant this test broke the day that op went live — which is a false red about
        // the invariant, not a finding about it.
        let staged = all_specs()
            .find(|s| !matches!(s.status, OpStatus::Live) && s.domain == Domain::Ai)
            .expect("the table always has at least one staged row");
        let staged_desc = NodeDesc {
            op_type: staged.op_type.into(),
            ..Default::default()
        };
        assert!(
            lookup(&staged_desc.qualified_name()).is_some(),
            "{} should be in the table",
            staged.op_type
        );
        assert!(
            !is_registered(&staged_desc),
            "{} is staged, so it is not translatable",
            staged.op_type
        );
        assert!(spec_for(&staged_desc).is_none());
    }

    #[test]
    fn decline_codes_round_trip() {
        for code in DeclineCode::ALL {
            let reason = decline(*code, "because");
            assert_eq!(DeclineCode::of_reason(&reason), Some(*code), "{reason}");
            assert!(reason.ends_with("because"));
        }
    }

    #[test]
    fn decline_tags_are_unique() {
        let mut tags: Vec<&str> = DeclineCode::ALL.iter().map(|c| c.tag()).collect();
        tags.sort_unstable();
        let before = tags.len();
        tags.dedup();
        assert_eq!(before, tags.len());
    }

    #[test]
    fn foreign_reasons_do_not_parse_as_a_code() {
        // `ep.rs` builds its own reasons for control-flow bodies and `max_claim_ops`; those must
        // bucket as "other", not silently collide with a real code.
        assert_eq!(DeclineCode::of_reason("excluded by ep.max_claim_ops"), None);
        assert_eq!(DeclineCode::of_reason("[no-such-code] hello"), None);
        assert_eq!(DeclineCode::of_reason(""), None);
    }

    #[test]
    fn domains_render_into_qualified_names() {
        let ai = OpSpec {
            domain: Domain::Ai,
            ..crate::ops::elementwise::OPS[0]
        };
        assert!(!ai.qualified_name().contains("::"));
        let ms = OpSpec {
            domain: Domain::Ms,
            ..crate::ops::elementwise::OPS[0]
        };
        assert!(ms.qualified_name().starts_with("com.microsoft::"));
    }

    #[test]
    fn edge_type_reports_static_and_rank() {
        let dynamic = EdgeType {
            dtype: Some(DType::F32),
            shape: Some(vec![-1, 4]),
        };
        assert_eq!(dynamic.rank(), Some(2));
        assert!(!dynamic.is_static());

        let stat = EdgeType {
            dtype: Some(DType::F16),
            shape: Some(vec![2, 4]),
        };
        assert!(stat.is_static());

        let unknown = EdgeType::default();
        assert_eq!(unknown.rank(), None);
        assert!(!unknown.is_static());
    }

    #[test]
    fn scalars_are_static_with_rank_zero() {
        let scalar = EdgeType {
            dtype: Some(DType::F32),
            shape: Some(vec![]),
        };
        assert_eq!(scalar.rank(), Some(0));
        assert!(scalar.is_static());
    }

    // -------------------------------------------------------------------------------------------
    // §8.9.4 planted rejections — CLAIM_UNPROVEN validation
    //
    // "A parser that can express 'everything' must not exist" — enforced here, not by convention.
    // Each test plants a specific disallowed string and asserts `ProofKey::validate` returns `Err`.
    // If any of these tests are deleted, the no-wildcard guarantee no longer has a mechanism.
    // -------------------------------------------------------------------------------------------

    /// `*` is the most obvious wildcard and must always be rejected.
    #[test]
    fn claim_unproven_rejects_star_wildcard() {
        assert!(
            ProofKey::validate("*").is_err(),
            "'*' must not be a valid proof key — it would silently enable all forms"
        );
    }

    /// `=1` style booleans must be rejected (§8.9.4: no boolean form).
    #[test]
    fn claim_unproven_rejects_boolean_one() {
        assert!(
            ProofKey::validate("1").is_err(),
            "'1' must not be a valid proof key — it would silently enable all forms"
        );
    }

    /// `all` is the natural-language wildcard and must be rejected.
    #[test]
    fn claim_unproven_rejects_all_wildcard() {
        assert!(
            ProofKey::validate("all").is_err(),
            "'all' must not be a valid proof key — it would silently enable all forms"
        );
    }

    /// A bare op-type with no field separators would cover all forms of that op.
    #[test]
    fn claim_unproven_rejects_bare_op_type() {
        assert!(
            ProofKey::validate("MatMulNBits").is_err(),
            "a bare op-type must not be a valid proof key — it would cover all forms of that op"
        );
        assert!(
            ProofKey::validate("com.microsoft::MatMulNBits").is_err(),
            "a domain-qualified op-type without field separators must not be a valid proof key"
        );
    }

    /// A well-formed key (has all required '/' separators) is accepted.
    #[test]
    fn claim_unproven_accepts_full_key() {
        assert!(
            ProofKey::validate(
                "com.microsoft::MatMulNBits/1+/f16,u8,f16/qgemv_f16/runtime-extent/scales"
            )
            .is_ok(),
            "a fully-specified proof key must be accepted"
        );
    }

    /// A truncated key must be rejected, not narrowed.
    ///
    /// `ai.onnx::Add/7+/f32` is what a comma-split of a real key produces, and it passed
    /// validation until 2026-08-01. It is not a key for fewer forms; it is a key for no forms,
    /// which an operator cannot tell apart from a working one because both produce silence.
    #[test]
    fn claim_unproven_rejects_a_truncated_key() {
        assert!(ProofKey::validate("ai.onnx::Add/7+/f32").is_err());
        assert!(ProofKey::validate("ai.onnx::Add").is_err());
        assert!(ProofKey::validate("Add/7+/f32,f32>f32/ew/static/n2").is_err());
        assert!(
            ProofKey::validate("ai.onnx::Add/7+//ew_binary_add_f32/static/n2").is_err(),
            "an empty component matches everything in that position"
        );
    }

    /// An empty key is rejected (no ambiguity, but also not a key).
    #[test]
    fn claim_unproven_rejects_empty() {
        assert!(ProofKey::validate("").is_err());
        assert!(ProofKey::validate("   ").is_err());
    }

    /// The escape-hatch list separator must not be a character that occurs *inside* a key.
    ///
    /// This is a regression control for a defect found on 2026-08-01 by the generator's
    /// attribution check, not by reading the parser. The list was comma-separated; a proof key
    /// contains commas in its dtype signature (`f32,f32>f32`); so a single well-formed key
    /// arrived at the parser as three malformed fragments, the list was discarded, and the run
    /// silently claimed nothing while a CPU-vs-CPU comparison returned `MATCH`.
    ///
    /// The test plants both halves of that failure:
    ///   * a real key split on `,` yields fragments that do **not** validate — so a comma
    ///     separator could never carry a key, and
    ///   * the same key split on `;` yields exactly itself.
    ///
    /// Asked R9's question — which way does this move when its subject is wrong? — a separator
    /// that collides with key syntax moves *towards* silence: the operator sets the variable,
    /// sees no error at the shell, and gets a decline. It cannot be repaired by tightening the
    /// validator, because the validator was already right. It is repaired by the separator.
    #[test]
    fn claim_unproven_separator_does_not_occur_inside_a_key() {
        const KEY: &str = "ai.onnx::Add/7+/f32,f32>f32/ew_binary_add_f32/static/n2";

        assert!(
            ProofKey::validate(KEY).is_ok(),
            "the planted key must itself be valid, or the control proves nothing"
        );

        let comma_fragments: Vec<&str> = KEY.split(',').collect();
        assert!(
            comma_fragments.len() > 1,
            "a real proof key must contain a comma, or this control has stopped testing anything"
        );
        for frag in &comma_fragments {
            assert!(
                ProofKey::validate(frag).is_err(),
                "comma-splitting a key produced the validatable fragment {frag:?} — the separator \
                 collision would be silent"
            );
        }

        let semi_fragments: Vec<&str> = KEY.split(';').collect();
        assert_eq!(
            semi_fragments,
            vec![KEY],
            "';' must not occur inside a proof key; it is the list separator"
        );
    }

    /// Two forms of the same op must produce two different keys.
    ///
    /// R11's `arms_must_differ`: a key function that returned a constant would make every proof
    /// satisfy every form, and the ledger would appear to close while proving nothing. The pairs
    /// below differ in exactly one component each — dtype, optional-input set, shape class — and
    /// each is a difference §8.7 calls a *path* difference.
    #[test]
    fn distinct_forms_have_distinct_keys() {
        let pairs = [
            (
                "ai.onnx::Add/7+/f32,f32>f32/ew_binary_add_f32/static/n2",
                "ai.onnx::Add/7+/f16,f16>f16/ew_binary_add_f16/static/n2",
            ),
            (
                "com.microsoft::MatMulNBits/1+/f16,u8,f16/qgemv_f16/runtime-extent/scales",
                "com.microsoft::MatMulNBits/1+/f16,u8,f16,u8/qgemv_f16/runtime-extent/scales+zero_points",
            ),
            (
                "ai.onnx::Add/7+/f32,f32>f32/ew_binary_add_f32/static/n2",
                "ai.onnx::Add/7+/f32,f32>f32/ew_binary_add_f32/runtime-extent/n2",
            ),
        ];
        for (a, b) in pairs {
            assert!(ProofKey::validate(a).is_ok(), "{a} must be a valid key");
            assert!(ProofKey::validate(b).is_ok(), "{b} must be a valid key");
            assert_ne!(
                a, b,
                "two different forms collapsed to one key — a proof of one would be returned for \
                 the other"
            );
        }
    }
    /// A ledger line with a valid key, a MATCH verdict, and **no attribution** grants nothing.
    ///
    /// RAI-008(a). Morpheus named the cheapest satisfaction of criterion 11 exactly: derive the
    /// ledger from the same enumeration that produces the claims, and `ledger_hits ==
    /// proven_key_lookups` forever — an identity whose two sides come from one source, in which
    /// `6/6` looks identical under both readings. The defence is that a proof run leaves a mark
    /// the claim table cannot forge: a **dispatch count**, which only exists after a session
    /// executed.
    ///
    /// R10 — the falsifier varies with the input. Four ledgers, differing only in the attribution
    /// fields, produce four different outcomes.
    #[test]
    fn an_entry_without_attribution_proves_nothing_however_well_formed() {
        const KEY: &str = "ai.onnx::Add/7+/f32,f32>f32/ew_binary_add_f32/static/n2";
        let key = ProofKey::validate(KEY).expect("valid key");

        let ledger_with = |extra: &str| {
            let entry = format!(
                "{{\"key\":\"{KEY}\",\"verdict\":\"MATCH\",\"device\":\"d\",\"ort_build\":\"1\",\
                 \"tolerance\":\"t\",\"artifact\":\"a\",\"generated_at\":\"now\"{extra}}}"
            );
            let digest = format!("{:016x}", fnv1a64(format!("{entry}\n").as_bytes()));
            let header = format!(
                "{{\"__ledger__\":1,\"content_fnv1a64\":\"{digest}\",\"entry_count\":1,\
                 \"generator\":\"test\"}}"
            );
            parse_ledger(&format!("{header}\n{entry}\n"))
        };

        // Attributed: the only shape that proves.
        let good = ledger_with(",\"claimed_nodes\":1,\"dispatches_executed\":1");
        assert!(good.faults.is_empty(), "faults: {:?}", good.faults);
        assert!(good.get(&key).is_some(), "an attributed MATCH must prove");

        // Absent witnesses — the shape a claim-table-derived ledger would have.
        let enumerated = ledger_with("");
        assert!(
            enumerated.get(&key).is_none(),
            "an entry with no attribution witness must not grant a claim"
        );
        assert!(
            enumerated.faults.iter().any(|f| f.contains("attribution")),
            "the fault must name attribution, not merely fail; got {:?}",
            enumerated.faults
        );

        // Zero dispatches — the 2026-07-30 specimen: a MATCH from a CPU-vs-CPU run.
        let cpu_only = ledger_with(",\"claimed_nodes\":1,\"dispatches_executed\":0");
        assert!(
            cpu_only.get(&key).is_none(),
            "a run that dispatched nothing proves nothing, whatever it compared"
        );

        // Quoted counters: a writer that stringifies its counters did not read a counter.
        let stringified = ledger_with(",\"claimed_nodes\":\"1\",\"dispatches_executed\":\"1\"");
        assert!(
            stringified.get(&key).is_none(),
            "a quoted count is not a count"
        );
    }

    /// The shipped ledger is attributed, entry by entry. This is the assertion that would go red
    /// if anyone regenerated it with a tool that stopped recording provenance.
    #[test]
    fn every_shipped_ledger_entry_carries_its_proof_run() {
        let l = ledger();
        assert!(l.faults.is_empty(), "shipped ledger faults: {:?}", l.faults);
        assert!(!l.is_empty(), "a ledger with no entries proves nothing");
        for e in l.entries() {
            assert!(
                e.claimed_nodes > 0 && e.dispatches_executed > 0,
                "shipped entry {} has claimed_nodes={} dispatches_executed={}",
                e.key.0,
                e.claimed_nodes,
                e.dispatches_executed
            );
        }
    }

    /// **The three-token miss path** (R13, discharge condition (d)). A miss is three findings, not
    /// one `false`, and they call for three different actions: regenerate this form, fix the
    /// ledger file, and nothing at all.
    #[test]
    fn a_ledger_miss_reports_which_of_three_things_happened() {
        let proven = ProofKey::validate(
            "ai.onnx::Add/7+/f32,f32>f32/ew_binary_add_f32/static/n2",
        )
        .expect("valid");
        let absent = ProofKey::validate(
            "ai.onnx::Add/7+/f64,f64>f64/ew_binary_add_f64/static/n2",
        )
        .expect("valid");

        assert_eq!(lookup_key(&proven), LedgerLookup::Hit);
        assert_eq!(lookup_key(&absent), LedgerLookup::KeyAbsent);

        // Every token is distinct, and distinct from the hit. A vocabulary in which two of these
        // share a spelling is the `0`-for-three-states defect R12 already made this project fix.
        let tokens = [
            LedgerLookup::Hit.token(),
            LedgerLookup::KeyAbsent.token(),
            LedgerLookup::Faulted.token(),
            LedgerLookup::NeverAttempted.token(),
        ];
        for (i, t) in tokens.iter().enumerate() {
            assert!(!t.is_empty());
            assert!(
                !tokens[..i].contains(t),
                "two ledger-miss states share the token {t:?}"
            );
        }
    }

    /// A build whose baked ledger disagrees with the ledger on disk **refuses to claim**.
    ///
    /// RAI-008(b), digest half. The threat is not a hand-edit before the build — the header
    /// digest already catches that — it is the file on disk changing *after* it, so the artifact
    /// a reviewer reads is not the artifact the binary claims from.
    ///
    /// R9 amendment 5: this check moves **against** the reader's confidence. A mismatch removes
    /// claims; it can never add one. That is why it is a refusal rather than a WARN, and why
    /// pointing it at a file that does not exist is also a refusal — a ledger that was asked for
    /// and is absent is not an empty ledger.
    #[test]
    fn a_disk_ledger_that_disagrees_with_the_baked_one_refuses_to_claim() {
        let baked = parse_ledger(LEDGER_SOURCE);
        assert!(baked.faults.is_empty(), "baseline: {:?}", baked.faults);

        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target");
        std::fs::create_dir_all(&dir).ok();

        // Same content: agreement, no fault. This arm is what makes the other arm a detection
        // rather than a check that fails on everything.
        let same = dir.join("ledger_disk_same.jsonl");
        std::fs::write(&same, LEDGER_SOURCE).expect("write");
        // SAFETY: single-threaded test; the variable is removed on every path below.
        unsafe { std::env::set_var(ENV_LEDGER_FILE, &same) };
        assert!(
            check_baked_against_disk(&baked).is_none(),
            "an on-disk ledger identical to the baked one must not fault"
        );

        // One byte different: refusal, and the message must carry both digests so the failure
        // text is diagnosable without re-running anything (R13).
        let drifted = dir.join("ledger_disk_drifted.jsonl");
        let mut body = LEDGER_SOURCE.to_string();
        body.push_str(
            "{\"key\":\"ai.onnx::Xor/7+/b,b>b/ew_binary_xor/static/n2\",\"verdict\":\"MATCH\",\
             \"claimed_nodes\":1,\"dispatches_executed\":1}\n",
        );
        std::fs::write(&drifted, &body).expect("write");
        // SAFETY: single-threaded test; the variable is removed on every path below.
        unsafe { std::env::set_var(ENV_LEDGER_FILE, &drifted) };
        let fault = check_baked_against_disk(&baked).expect("a drifted disk ledger must fault");
        assert!(fault.contains(&baked.actual_digest), "fault: {fault}");
        assert!(fault.contains("Refusing to claim"), "fault: {fault}");

        // Named and missing is also a refusal, not a fallback to the baked copy.
        // SAFETY: single-threaded test; the variable is removed on every path below.
        unsafe { std::env::set_var(ENV_LEDGER_FILE, dir.join("ledger_disk_absent.jsonl")) };
        let missing = check_baked_against_disk(&baked).expect("an absent named ledger must fault");
        assert!(missing.contains("not an empty ledger"), "fault: {missing}");

        // SAFETY: single-threaded test.
        unsafe { std::env::remove_var(ENV_LEDGER_FILE) };
        assert!(check_baked_against_disk(&baked).is_none());
        std::fs::remove_file(&same).ok();
        std::fs::remove_file(&drifted).ok();
    }
}
