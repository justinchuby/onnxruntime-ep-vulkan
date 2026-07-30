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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpStatus {
    /// Claimable. The shader exists, the handler translates, conformance covers it.
    Live,
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

    /// Whether this row is claimable at all.
    pub fn is_live(&self) -> bool {
        matches!(self.status, OpStatus::Live)
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

/// The one question `ep.rs` asks per node.
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
            unevaluated: vec!["opset", "contrib-schema", "status", "predicate"],
            shape_class,
            predicate_ok: false,
            predicate_ok_with_runtime_extents: false,
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
            // Every live row must have at least one compiled shader variant.
            let has_shader = spec.caps.iter().any(|d| {
                spec.kernel
                    .stem(d)
                    .is_some_and(|stem| shaders::find(stem).is_some())
            });
            assert!(
                has_shader,
                "{name} is live but has no compiled shader variant in the binary"
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
}
