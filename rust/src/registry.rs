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
    facts: Option<&'graph crate::shape_infer::InferredShapes>,
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
            facts: None,
            _graph: std::marker::PhantomData,
        }
    }

    /// Wrap a node together with the graph-level rank facts proved by [`crate::shape_infer`].
    ///
    /// The overlay is applied inside [`NodeView::edge_type`], so every predicate that reads an
    /// input or output type sees the refined reading without knowing the pass exists. The overlay
    /// only ever *adds* information — see [`crate::shape_infer::InferredShapes::refine`], which
    /// keeps ORT's reading whole whenever the two disagree.
    ///
    /// # Safety
    /// Same contract as [`NodeView::new`].
    pub unsafe fn new_with_facts(
        api: *const ort::OrtApi,
        node: *const ort::OrtNode,
        facts: &'graph crate::shape_infer::InferredShapes,
    ) -> NodeView<'graph> {
        NodeView {
            api,
            node,
            facts: Some(facts),
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

    /// Resolve one `OrtValueInfo` to an [`EdgeType`], applying the rank overlay if one is
    /// installed.
    fn edge_type(&self, slot: *const ort::OrtValueInfo) -> Option<EdgeType> {
        self.edge_type_inner(slot, true)
    }

    /// Resolve one `OrtValueInfo` to exactly what ORT reported, overlay or no overlay.
    fn edge_type_raw(&self, slot: *const ort::OrtValueInfo) -> Option<EdgeType> {
        self.edge_type_inner(slot, false)
    }

    /// Resolve one `OrtValueInfo` to an [`EdgeType`].
    fn edge_type_inner(&self, slot: *const ort::OrtValueInfo, refine: bool) -> Option<EdgeType> {
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

            Some(EdgeType {
                dtype,
                shape: if refine {
                    self.refine_shape(slot, shape)
                } else {
                    shape
                },
            })
        }
    }

    /// Apply the graph-level rank overlay, if one was installed, to one edge reading.
    ///
    /// A `NodeView` built with [`NodeView::new`] has no overlay and returns the reading unchanged,
    /// which is what every existing unit test and the `Compile` path get.
    fn refine_shape(
        &self,
        slot: *const ort::OrtValueInfo,
        shape: Option<Vec<i64>>,
    ) -> Option<Vec<i64>> {
        let Some(facts) = self.facts else {
            return shape;
        };
        // SAFETY: `slot` is the same live, graph-owned value info the caller just read a type
        // from, and `self.api` is live per the constructor's contract.
        let name = unsafe { value_info_name(self.api, slot) };
        if name.is_empty() {
            return shape;
        }
        facts.refine(&name, shape.as_deref()).or(shape)
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

    /// Type of output `i` **exactly as ORT reported it**, bypassing the rank overlay.
    ///
    /// For the one predicate that must distinguish "ORT resolved this output" from "this EP
    /// worked the rank out": `Reshape`'s target is a runtime *value*, so a rank this pass proved
    /// is not a shape `Compute()` can bind, and claiming on it would be a broken commitment
    /// rather than a fallback. Every other row reasons about ranks that follow from input shapes,
    /// which the dynamic-kernel path re-reads from ORT at `Compute()`.
    pub fn output_type_as_reported(&self, i: usize) -> Option<EdgeType> {
        let slots = self.output_slots();
        self.edge_type_raw(*slots.get(i)?)
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

    /// Whether any of this node's edge readings came from the rank overlay rather than from ORT.
    ///
    /// [`crate::shape_infer::InferredShapes`] only retains values whose shape the pass proved
    /// *beyond* what ORT declared, so an entry for any of this node's edges means a predicate that
    /// read that edge saw something ORT did not report. Recorded in the claim log so the record
    /// never implies ORT said more than it did.
    pub fn rank_inferred(&self) -> bool {
        let Some(facts) = self.facts else {
            return false;
        };
        self.input_slots()
            .into_iter()
            .chain(self.output_slots())
            .any(|s| {
                // SAFETY: `s` is a live, graph-owned value info (or null, which
                // `value_info_name` handles) and `self.api` is live.
                let name = unsafe { value_info_name(self.api, s) };
                facts.shape(&name).is_some()
            })
    }

    /// Input value names in order. An omitted optional input yields an empty string, as does a
    /// slot whose name ORT declines to report.
    pub fn input_names(&self) -> Vec<String> {
        self.input_slots()
            .into_iter()
            // SAFETY: each slot is a live, graph-owned value info (or null, which
            // `value_info_name` handles) and `self.api` is live.
            .map(|s| unsafe { value_info_name(self.api, s) })
            .collect()
    }

    /// Output value names in order. See [`NodeView::input_names`].
    pub fn output_names(&self) -> Vec<String> {
        self.output_slots()
            .into_iter()
            // SAFETY: as for `input_names`.
            .map(|s| unsafe { value_info_name(self.api, s) })
            .collect()
    }

    /// Read the shape and, for small integer tensors, the element values of a constant initializer.
    ///
    /// Returns `(shape, ints)`. `ints` is `Some` only for `INT32`/`INT64` tensors of at most
    /// `max_ints` elements whose bytes ORT actually handed us; every other case yields the shape
    /// alone. The shape is a property of *stored data*, so unlike an inference result a rank-0
    /// answer here genuinely means "scalar" — this is the only sound source of a rank-0 fact
    /// (see [`crate::shape_infer`]).
    ///
    /// Returns `None` when the input is absent, is not a constant initializer, or when any part of
    /// the read fails. Never partially reports: an unreadable tensor is simply not a fact.
    pub fn constant_input_tensor(
        &self,
        i: usize,
        max_ints: usize,
    ) -> Option<(Vec<i64>, Option<Vec<i64>>)> {
        if !self.input_is_constant(i) {
            return None;
        }
        let slots = self.input_slots();
        let &slot = slots.get(i)?;
        if slot.is_null() {
            return None;
        }
        // SAFETY: `slot` is a live, graph-owned value info that `input_is_constant` just confirmed
        // is a constant initializer, and `self.api` is live for the whole call.
        unsafe { read_constant_tensor(self.api, slot, max_ints) }
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

/// Map a raw ONNX `TensorProto.DataType` value — the integer `Cast`'s `to` attribute carries —
/// to this EP's [`DType`], or `None` for types we have no storage for.
///
/// `Cast` is the one row whose *destination* element type is a node attribute rather than
/// something derivable from its inputs, and the attribute is the only source that survives a
/// symbolic extent: [`crate::ep::tensor_desc`] drops the whole `TensorDesc` — dtype included —
/// when any dimension is unknown, so a handler that reads the destination off the output edge
/// has no answer for exactly the graphs that need one. Delegates to `dtype_from_onnx` so the
/// enum mapping has one implementation and cannot drift.
pub fn dtype_from_onnx_value(v: i64) -> Option<DType> {
    u32::try_from(v)
        .ok()
        .and_then(|e| dtype_from_onnx(e as ort::ONNXTensorElementDataType))
}

// -------------------------------------------------------------------------------------------
// Public helpers for `compile_impl` (ep.rs boundary layer)
// -------------------------------------------------------------------------------------------

/// Rank ceiling for a constant initializer read at `GetCapability` time.
///
/// The shape-inference pass only ever reads shape vectors and axis lists, which are rank 0 or 1.
/// A higher rank is not an error, it is simply not something this path needs, so it is refused
/// rather than allocated for.
const MAX_CONSTANT_RANK: usize = 1;

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

/// Read a constant initializer's shape and, when it is a small integer tensor, its values.
////// Equivalent to `NodeView::edge_type(slot)` but usable outside of `NodeView`'s method context,
/// The shape is always returned when the type info reads; the values are returned only for
/// `INT32`/`INT64` tensors of at most `max_ints` elements. Symbolic dimensions cannot occur here
/// — an initializer has stored data — but a negative extent is still rejected rather than trusted.
///
/// # Safety
/// `api` must be a live `OrtApi`; `vi` must be a live `OrtValueInfo` owned by an ORT graph that
/// outlives this call, and it must name a constant initializer.
unsafe fn read_constant_tensor(
    api: *const ort::OrtApi,
    vi: *const ort::OrtValueInfo,
    max_ints: usize,
) -> Option<(Vec<i64>, Option<Vec<i64>>)> {
    // SAFETY: reading the immutable function table of a live `OrtApi` is a plain field read.
    let (get_init, get_tts, get_et, get_n, get_d, release_info) = unsafe {
        (
            (*api).ValueInfo_GetInitializerValue?,
            (*api).GetTensorTypeAndShape?,
            (*api).GetTensorElementType?,
            (*api).GetDimensionsCount?,
            (*api).GetDimensions?,
            (*api).ReleaseTensorTypeAndShapeInfo?,
        )
    };

    let mut value: *const ort::OrtValue = std::ptr::null();
    // SAFETY: `vi` is live per the fn contract and `value` is a valid out-parameter slot. The
    // `OrtValue` ORT writes is **borrowed** from the graph's initializer storage — the C API
    // declares it const and documents no release — so it must not be released here.
    let st = unsafe { get_init(vi, &mut value) };
    if !st.is_null() {
        // SAFETY: `api` and `st` are live; the status is ours to release.
        unsafe { sys::release_status(api, st) };
        return None;
    }
    if value.is_null() {
        return None;
    }

    let mut info: *mut ort::OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
    // SAFETY: `value` is a live `OrtValue`; `info` is a valid out-parameter slot. Unlike the value
    // above, this info is **owned** by us and is released on every path below.
    let st = unsafe { get_tts(value, &mut info) };
    if !st.is_null() {
        // SAFETY: `api` and `st` are live.
        unsafe { sys::release_status(api, st) };
        return None;
    }
    if info.is_null() {
        return None;
    }

    // Everything from here on must release `info`, so it runs in a closure whose result is
    // returned after the release.
    let read = || -> Option<(Vec<i64>, Option<Vec<i64>>)> {
        let mut et = ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        // SAFETY: `info` is a live, non-null info produced above; `&mut et` is a valid slot.
        let st = unsafe { get_et(info, &mut et) };
        if !st.is_null() {
            // SAFETY: `api` and `st` are live.
            unsafe { sys::release_status(api, st) };
            return None;
        }

        let mut rank: usize = 0;
        // SAFETY: `info` is live; `&mut rank` is a valid slot.
        let st = unsafe { get_n(info, &mut rank) };
        if !st.is_null() {
            // SAFETY: `api` and `st` are live.
            unsafe { sys::release_status(api, st) };
            return None;
        }
        if rank > MAX_CONSTANT_RANK {
            return None;
        }
        let mut dims = vec![0i64; rank];
        if rank > 0 {
            // SAFETY: `info` is live and `dims` holds exactly `rank` writable `i64` slots.
            let st = unsafe { get_d(info, dims.as_mut_ptr(), rank) };
            if !st.is_null() {
                // SAFETY: `api` and `st` are live.
                unsafe { sys::release_status(api, st) };
                return None;
            }
        }
        if dims.iter().any(|d| *d < 0) {
            return None; // stored data cannot have a symbolic extent; refuse the whole reading
        }

        let mut count: usize = 1;
        for d in &dims {
            let d = usize::try_from(*d).ok()?;
            count = count.checked_mul(d)?;
        }

        let is_i32 = et == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
        let is_i64 = et == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
        if !(is_i32 || is_i64) || count > max_ints {
            return Some((dims, None));
        }

        // SAFETY: reading the immutable function table of a live `OrtApi`.
        let Some(get_data) = (unsafe { (*api).GetTensorData }) else {
            return Some((dims, None));
        };
        let mut data: *const std::ffi::c_void = std::ptr::null();
        // SAFETY: `value` is a live `OrtValue` and `data` is a valid out-parameter slot.
        let st = unsafe { get_data(value, &mut data) };
        if !st.is_null() {
            // SAFETY: `api` and `st` are live.
            unsafe { sys::release_status(api, st) };
            return Some((dims, None));
        }
        if data.is_null() {
            // A zero-element tensor legitimately has no buffer; anything else is unreadable.
            return Some((dims, if count == 0 { Some(Vec::new()) } else { None }));
        }

        // SAFETY: ORT reported this tensor as `count` elements of the element type just read, and
        // `data` is the start of its contiguous buffer, which is owned by the graph's initializer
        // storage and outlives this call. The slice is therefore `count` valid, initialised,
        // correctly-aligned elements of exactly that type, and is only read from.
        let ints = unsafe {
            if is_i32 {
                std::slice::from_raw_parts(data.cast::<i32>(), count)
                    .iter()
                    .map(|v| i64::from(*v))
                    .collect()
            } else {
                std::slice::from_raw_parts(data.cast::<i64>(), count).to_vec()
            }
        };
        Some((dims, Some(ints)))
    };

    let out = read();
    // SAFETY: `info` is the non-null, live info produced above and is not used afterwards.
    unsafe { release_info(info) };
    out
}

/// Read the [`EdgeType`] of a standalone `OrtValueInfo` pointer.
////// Equivalent to `NodeView::edge_type(slot)` but usable outside of `NodeView`'s method context,
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
    /// Attribute axes this row's proof key **deliberately does not distinguish**.
    ///
    /// §8.9.23 (Morpheus, 2026-08-04), and the third answer to a question that was posed as a
    /// binary. `Conv`'s `group`, `strides`, `dilations` and `pads` are **push constants in one
    /// uniform code path** — `conv_f32.comp` branches on none of them, and grouped is the general
    /// form of dense. Under §8.7 they are *expressions, not paths*, so the key that omits them is
    /// **true**, and adding them as a key component would assert that a stride-2 `Conv` runs
    /// different code from a stride-1 one, which it does not. It would also demand ~52 proofs for
    /// one form.
    ///
    /// What is owed instead is a **disclosure**. The reader of a session-time "proven" line is
    /// entitled to know which axes that proof did not vary, because the honest reading of the
    /// claim is *"one proof covers every value of these axes by construction"* and the honest
    /// reading of the evidence is *"a CI-time suite varied them; nothing in your session did"*.
    /// Both clauses are rendered, and the second is the one Rai was watching for: without it the
    /// disclosure implies a coverage the session cannot witness.
    ///
    /// This is not a list of things that are unproven. It is a list of things the *key* is silent
    /// about, which is a different and narrower statement — and one nothing else in the system
    /// records.
    pub blind_axes: &'static [&'static str],
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
        $(, blind_axes: $blind:expr)?
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
                    blind_axes: $crate::opt_blind_axes!($($blind)?),
                }
            ),*
        ];
    };
}

/// The row's declared blind axes, or the empty slice when it declares none.
///
/// Empty is the right default and not a placeholder: a row whose key distinguishes every axis its
/// kernel reads has nothing to disclose. A row that *should* have declared axes and did not is a
/// silent under-disclosure, which is why `blind_axes` is checked against the shipped list in a
/// test rather than trusted to be remembered.
#[macro_export]
#[doc(hidden)]
macro_rules! opt_blind_axes {
    () => {
        &[]
    };
    ($blind:expr) => {
        $blind
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
    crate::ops::shape::OPS,
    crate::ops::conv::OPS,
    crate::ops::pooling::OPS,
    crate::ops::matmul::OPS,
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

/// The attribute axes the row for `qualified` declares its proof key blind to.
///
/// Public because the disclosure needs it and the disclosure is where the obligation lands: a
/// blind axis recorded in the registry and never rendered is a caveat the reader does not get.
/// Returns the empty slice for a name with no row, which is the same answer as "declares none" and
/// correct for both — a form the registry does not know is not one this EP claimed.
pub fn blind_axes_for(qualified: &str) -> &'static [&'static str] {
    // Two spellings reach here. `NodeView::qualified_name` renders the default domain as the bare
    // op type (`Conv`), `OpSpec::qualified_name` renders it the same way, and human-written call
    // sites — including this project's own doc comments — say `ai.onnx::Conv`. Accepting both is
    // not laxity: a disclosure that silently drops its caveat because the caller spelled the
    // domain out is exactly the failure this field exists to prevent.
    let bare = qualified.strip_prefix("ai.onnx::").unwrap_or(qualified);
    lookup(bare).map(|s| s.blind_axes).unwrap_or(&[])
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

    /// The whole `variant` component, suffixes and all.
    ///
    /// Component 3 of `domain::op_type/opset_bucket/dtypes/variant/shape_class/inputs`. `None` for
    /// a key that does not have six components, because a malformed key must not be answered with
    /// a plausible substring.
    pub fn variant_component(&self) -> Option<&str> {
        let parts: Vec<&str> = self.0.split('/').collect();
        (parts.len() >= 5).then(|| parts[3])
    }

    /// The SPIR-V module stem this form would dispatch, with the `@sel` and `#form` suffixes
    /// stripped.
    ///
    /// Those suffixes name a *specialisation constant* and an *attribute form*; neither is part of
    /// the module's name. Before they were stripped, every selector-bearing and every form-bearing
    /// key answered this question with a string that names no module — and the one caller that
    /// matters, [`form_is_provable`], reads an unknown stem as *"do not know, assume provable"*.
    /// So the day `Conv` gained a form suffix, every `Conv` key silently joined the under-claiming
    /// branch. Stripping here is what makes the suffixes additive rather than blinding.
    pub fn variant_stem(&self) -> Option<&str> {
        let v = self.variant_component()?;
        let v = v.split('#').next().unwrap_or(v);
        Some(v.split('@').next().unwrap_or(v))
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
            return Err(
                "proof key is missing its `domain::op_type` prefix; use the full \
                 domain::op_type/opset_bucket/dtypes/variant/shape_class/inputs form",
            );
        }
        if t.matches('/').count() != 5 {
            return Err("proof key does not have all six components; use the full \
                 domain::op_type/opset_bucket/dtypes/variant/shape_class/inputs form");
        }
        if t.split('/').any(|c| c.trim().is_empty()) {
            return Err(
                "proof key has an empty component; an empty field is a wildcard \
                 by another name",
            );
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

/// The kernel variant that will actually be dispatched: the SPIR-V module stem, plus any
/// specialisation constant that changes the code the driver emits from it.
///
/// The stem encodes template, template-op and dtype, which is most of the emitted code's identity.
/// A row with no shader (metadata-only, e.g. a shape op handled on the host) reports `metadata`
/// rather than an empty string, so that "no variant" is a value and not a hole.
///
/// # The selector suffix
///
/// `ops::common::selector` carries an op's *expression selector* in specialisation constant 2, and
/// a specialisation constant is resolved at pipeline creation — so two nodes with different
/// selectors run different code out of one module. This key's own doc comment already required
/// that ("`kernel_variant_key` — including any spec-constant value that changes the emitted
/// code"); the suffix is what makes the implementation match it. `IsInf(detect_negative=0)` and
/// `IsInf(detect_negative=1)` are different paths and must not share a proof.
///
/// **Only attribute-derived selectors are appended.** An input-presence selector (`Clip`'s) is
/// already recorded, exactly and independently, by [`populated_input_set`]; appending it too would
/// write the same fact into the key twice and change every existing `Clip` key for no distinction
/// gained.
///
/// # There is no form suffix, and that is a ruling rather than an omission
///
/// For one round this component carried an op's *form* under `#` — `Conv`'s `group`, `Gemm`'s
/// `transB` — on the reasoning that a proof obtained on a dense convolution says nothing about a
/// grouped one. §8.9.23 (Morpheus, 2026-08-04) ruled that reasoning wrong at its root, and
/// `conv_f32.comp` settles it by inspection: `group`, `strides`, `dilations` and `pads` are push
/// constants folded into index arithmetic that every node executes, and grouped is the *general*
/// form of dense (`cpg = c / group`, which is `c` at `group=1`). One module, one pipeline, one set
/// of emitted instructions. Under §8.7 they are **expressions, not paths**, so a key that omits
/// them satisfies this key's own stated meaning — *two nodes whose keys are equal are dispatched
/// by the same code with the same descriptor layout* — and a key that included them would assert a
/// distinction the hardware does not make, while demanding ~52 proofs for one form.
///
/// The worry that motivated the suffix does not go away with it: an expression can still be wrong,
/// and a proof taken at `group=1` never executed the `group=32` arithmetic. What discharges it is
/// [`OpSpec::blind_axes`] — the axes are named on the claim line the user reads, together with the
/// clause that a CI-time suite speaks for them and nothing in that session does.
///
/// `has_bias` is *not* one of these and never was: it changes the populated input set, which is
/// component 6 of this key already.
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
    // A pair-keyed row's module is chosen by (source, destination), so the source dtype alone
    // does not name it. `stem` returns `None` there by design; asking it anyway would have
    // rendered every `Cast` key as `metadata` — a component that says "this row has no shader" on
    // a row with thirty-six of them, and one string shared by all thirty-six.
    let pair_stem = if spec.kernel.template.is_pair_keyed() {
        let dst = view
            .output_types()
            .first()
            .and_then(|t| t.as_ref().and_then(|e| e.dtype));
        match (dispatch_dtype, dst) {
            (Some(s), Some(d)) => spec.kernel.pair_stem(s, d),
            _ => None,
        }
    } else {
        None
    };
    let stem = match pair_stem.or_else(|| dispatch_dtype.and_then(|d| spec.kernel.stem(d))) {
        Some(stem) if !stem.is_empty() => stem.to_string(),
        _ => "metadata".to_string(),
    };
    match crate::ops::common::selector::source_for(spec.op_type) {
        Some(crate::ops::common::selector::SelectorSource::Attrs(_)) => {
            // An unresolvable selector is a node the predicate declines; the key still has to be a
            // distinct, total string, so the failure renders as its own tag rather than collapsing
            // onto a real selector's.
            match crate::ops::common::selector::resolve(spec.op_type, view) {
                Ok(sel) => format!("{stem}@sel{sel}"),
                Err(_) => format!("{stem}@sel-unresolved"),
            }
        }
        _ => stem,
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
    let n = (0..view.num_inputs())
        .filter(|i| view.has_input(*i))
        .count();
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
        "com.microsoft::MatMulNBits" => {
            &[(2, "scales"), (3, "zero_points"), (4, "g_idx"), (5, "bias")]
        }
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
    /// **Subject witness — the embedded SPIR-V modules the proof run dispatched.**
    ///
    /// §8.9.11. Sorted, deduplicated stems, taken from the run's own dispatch record.
    pub shaders: Vec<String>,
    /// **Subject witness — a digest of exactly those modules' SPIR-V bytes.**
    ///
    /// Without it, an entry outlives its subject: the key stays equal while the kernel it names
    /// is replaced, `ledger_hits == proven_key_lookups` stays true forever, and nothing in the
    /// pipeline can tell. That is the criterion-11 shape one level up — not a ledger derived
    /// from the claim table, but a ledger whose entries silently stop describing anything.
    ///
    /// **A proof that cannot be invalidated by changing its subject is not a proof of that
    /// subject.** Recomputed at parse time; a disagreement no longer deletes the entry — see
    /// [`LedgerEntry::subject`].
    pub shader_digest: String,
    /// **Subject witness — a digest of those modules' SOURCE CLOSURE** (§8.9.19 part 2).
    ///
    /// Toolchain-independent by construction, which is the whole point: `shader_digest` moves
    /// when `glslc` moves, this one does not, and **their disagreement is the instrument**. Empty
    /// for an entry written before §8.9.19, which is why [`SubjectVerdict::Indeterminate`] exists.
    pub source_digest: String,
    /// **Frame — the shader compiler the proof's SPIR-V was built with** (§8.9.19 part 1).
    ///
    /// A FRAME component, never a KEY component. Empty for a pre-§8.9.19 entry.
    pub toolchain: String,
    /// **Frame — the runtime specialisation the proof run actually bound** (§8.9.20).
    ///
    /// The dispatch-time witness. Both other digests are fixed at build time and what runs is a
    /// *pipeline* — `(SPIR-V, specialisation values, layout)` — so a constant chosen at dispatch
    /// is outside both of them. Two runs that select different values for one stem have identical
    /// `shader_digest` **and** identical `source_digest` and build different kernels;
    /// `ONNXRUNTIME_EP_VULKAN_GEMV_PACKED` is that case today and every spec-constant selector
    /// adds another.
    ///
    /// Empty for every entry written before §8.9.20, and — unlike `source_digest` — that is
    /// **not backfillable**: the value is a fact about a run that is over, not about a tree that
    /// is still here. See [`SpecWitness::Unrecorded`] for what that costs an entry's meaning.
    pub spec_digest: String,
    /// How this build's shader set compares with the one the entry was proven against.
    ///
    /// Computed at parse time, **recorded on the surviving entry rather than used to delete it**.
    /// §8.9.19 part 1 names the deletion as the mechanical accident that made the ledger read as
    /// "keyed per toolchain": `parse_ledger` used to `continue` past a digest mismatch, the entry
    /// never entered [`Ledger::entries`], and [`Ledger::get`] returned the same `None` it returns
    /// for a form nobody ever proved — **a frame mismatch was indistinguishable from a key
    /// absence**, which are different facts with different repairs and only one of them
    /// actionable.
    pub subject: SubjectVerdict,
}

/// One component of the FRAME that differs between an entry and the run reading it (§8.9.19).
///
/// A **set**, not a growing enum of combinations: a Linux CI run on lavapipe differs from a
/// Windows proof in both device and toolchain, and under a per-combination enum that would be a
/// new status. Under a delta set it is `PROVEN-ELSEWHERE{device, toolchain}` and the reader learns
/// more, not less.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum FrameDelta {
    /// The entry names a physical device that is not the one this run opened.
    Device,
    /// The entry's SPIR-V was produced by a different shader compiler, and the **source closure
    /// is identical** — so the kernel did not move, only the compiler did.
    Toolchain,
    /// The run bound a **different runtime specialisation** for the entry's shader set than the
    /// proof run did (§8.9.20). Same SPIR-V, same source, different pipeline.
    ///
    /// The only δ component that is not a property of the machine or the build: it is a property
    /// of *this dispatch*, so unlike the others it can become true part-way through a run, at the
    /// moment the pipeline is created.
    Specialisation,
    /// The entry names a different ONNX Runtime build.
    ///
    /// **Enumerated but never produced today, and that is a stated residual rather than an
    /// oversight**: the running ORT build is not available to this predicate, so the comparison
    /// cannot be made. It is in the vocabulary because §8.9.19 names it in δ, and a delta set
    /// whose members are invented one at a time is how a vocabulary becomes two vocabularies.
    OrtBuild,
    /// The entry names a different graphics driver. Same residual as [`FrameDelta::OrtBuild`]:
    /// the ledger records no driver, so nothing can produce this yet.
    Driver,
}

impl FrameDelta {
    /// The token an artifact records.
    pub fn token(&self) -> &'static str {
        match self {
            FrameDelta::Device => "device",
            FrameDelta::Toolchain => "toolchain",
            FrameDelta::Specialisation => "specialisation",
            FrameDelta::OrtBuild => "ort_build",
            FrameDelta::Driver => "driver",
        }
    }
}

/// How this build's shader set compares with the set an entry was proven against (§8.9.19).
///
/// **The table this implements, and the reason there are two digests.** No single hash can be
/// sensitive to the kernel and blind to the compiler, because the compiler is a function whose
/// output is the only thing that actually runs:
///
/// | `spirv_digest` | `source_digest` | verdict |
/// |---|---|---|
/// | same | same | [`SubjectVerdict::Identical`] — `PROVEN` |
/// | **differs** | **same** | [`SubjectVerdict::ToolchainDelta`] — **the Linux case**, `PROVEN-ELSEWHERE{toolchain}` |
/// | differs | differs | [`SubjectVerdict::Changed`] — `UNPROVEN{SUBJECT-CHANGED}` |
/// | same | differs | [`SubjectVerdict::SourceCosmetic`] — `PROVEN`, **and named** |
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SubjectVerdict {
    /// Both digests agree. The proof is about exactly this code.
    Identical,
    /// SPIR-V differs, source closure identical. The compiler moved, not the kernel.
    ToolchainDelta {
        /// The SPIR-V digest recorded on the entry.
        recorded: String,
        /// What this build's modules hash to.
        current: String,
    },
    /// SPIR-V identical, source closure differs — an edit that produced identical SPIR-V.
    ///
    /// Claimable, and **named**: this is the row that proves the pair is doing work rather than
    /// one digest wearing two hats.
    SourceCosmetic {
        /// The source-closure digest recorded on the entry.
        recorded: String,
        /// What this build's source closure hashes to.
        current: String,
    },
    /// Both differ. The kernel moved and the proof is about something else.
    Changed {
        /// The SPIR-V digest recorded on the entry.
        recorded_spirv: String,
        /// What this build's modules hash to.
        current_spirv: String,
    },
    /// The SPIR-V differs and the entry carries **no** `source_digest`, so the two rows that a
    /// differing SPIR-V could mean cannot be told apart.
    ///
    /// **Not claimable, and it must not be**: guessing `ToolchainDelta` here would grant every
    /// pre-§8.9.19 entry a claim on a kernel that may genuinely have changed. The repair is
    /// mechanical and stated in the message — `gen_proof_ledger.py --backfill-frame` on a machine
    /// whose SPIR-V *does* match, which is the only machine that can honestly record what source
    /// produced those bytes.
    Indeterminate {
        /// The SPIR-V digest recorded on the entry.
        recorded_spirv: String,
        /// What this build's modules hash to.
        current_spirv: String,
    },
}

impl SubjectVerdict {
    /// The token an artifact records.
    pub fn token(&self) -> &'static str {
        match self {
            SubjectVerdict::Identical => "IDENTICAL",
            SubjectVerdict::ToolchainDelta { .. } => "TOOLCHAIN-DELTA",
            SubjectVerdict::SourceCosmetic { .. } => "SOURCE-COSMETIC",
            SubjectVerdict::Changed { .. } => "SUBJECT-CHANGED",
            SubjectVerdict::Indeterminate { .. } => "SUBJECT-INDETERMINATE",
        }
    }

    /// Whether the subject is the same code, whatever compiled it.
    pub fn same_subject(&self) -> bool {
        matches!(
            self,
            SubjectVerdict::Identical
                | SubjectVerdict::ToolchainDelta { .. }
                | SubjectVerdict::SourceCosmetic { .. }
        )
    }
}

/// How the specialisation this run **bound** compares with the one an entry was proven under
/// (§8.9.20) — the dispatch-time frame witness.
///
/// Not a `SubjectVerdict` sibling and not stored on the entry, because unlike both digests it is
/// **not known at parse time**. The value is resolved when `vkCreateComputePipelines` is called,
/// so this is computed live and can legitimately give different answers before and after a
/// dispatch. That time-dependence is the finding, not a defect in it: an entry's frame includes
/// something the reader cannot know until the run reaches the kernel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SpecWitness {
    /// This run has bound no pipeline for the entry's stems. **The claim-time answer in a cold
    /// process, and it must not read as agreement** — nothing has been compared.
    Unobserved,
    /// Some of the entry's stems have a bound pipeline and some do not, so the set-wide digests
    /// are not comparable. Reporting a delta here would manufacture one out of a run that has not
    /// finished binding.
    Partial {
        /// Stems with a bound pipeline.
        covered: usize,
        /// Stems in the entry's set.
        of: usize,
    },
    /// The entry records no specialisation at all — every entry written before §8.9.20.
    ///
    /// **An unrecorded frame is not an equal frame** (§8.9.19 row 5's rule, one axis over). Where
    /// it differs from row 5: a missing `source_digest` is repairable from the tree, so refusing
    /// the claim buys a `--backfill-frame`; a missing specialisation is a fact about a run that
    /// has ended, and no build can recover it. Refusing here would decline all 103 forms for a
    /// repair only `--reprove` can perform, so this **claims and discloses** instead — the same
    /// trade `DEVICE-UNATTRIBUTED` makes for the same reason.
    Unrecorded,
    /// Compared, and equal.
    Identical,
    /// Compared, and different: the proof was taken on another pipeline.
    Delta {
        /// The specialisation digest recorded on the entry.
        recorded: String,
        /// What this run bound for the same stems.
        current: String,
    },
}

impl SpecWitness {
    /// The token an artifact records.
    pub fn token(&self) -> &'static str {
        match self {
            SpecWitness::Unobserved => "SPEC-UNOBSERVED",
            SpecWitness::Partial { .. } => "SPEC-PARTIAL",
            SpecWitness::Unrecorded => "SPEC-UNRECORDED",
            SpecWitness::Identical => "SPEC-IDENTICAL",
            SpecWitness::Delta { .. } => "SPEC-DELTA",
        }
    }
}

/// Compare one entry's recorded specialisation against what this run has bound (§8.9.20).
///
/// Live, never cached: the answer changes when a pipeline is created, and caching it at parse
/// time would freeze the one component of the frame that is not fixed before the run starts.
pub fn spec_witness_for(entry: &LedgerEntry) -> SpecWitness {
    let stems: Vec<&str> = entry.shaders.iter().map(String::as_str).collect();
    match crate::counters::specialisation_digest_for(&stems) {
        crate::counters::SpecObservation::Unobserved => SpecWitness::Unobserved,
        crate::counters::SpecObservation::Partial { covered, of } => {
            SpecWitness::Partial { covered, of }
        }
        crate::counters::SpecObservation::Full(current) => {
            if entry.spec_digest.is_empty() {
                SpecWitness::Unrecorded
            } else if entry.spec_digest == current {
                SpecWitness::Identical
            } else {
                SpecWitness::Delta {
                    recorded: entry.spec_digest.clone(),
                    current,
                }
            }
        }
    }
}

/// **§8.9.20 — the dispatch-time frame witness, read at the moment it becomes readable.**
///
/// Called from the pipeline-creation path when a *new* `(stem, spec_constants)` pair is bound. It
/// exists because the claim path cannot serve: a claim is decided before any pipeline exists, so a
/// witness consulted only there would report `SPEC-UNOBSERVED` on every single-session run and the
/// delta counter's only observable value would be zero. That is the defect class this project has
/// now built two mechanisms to remove, and building a third one with it would be worse than not
/// building it.
///
/// Walks only the entries whose recorded shader set contains `stem`, and only when the set has
/// actually grown, so the cost is one ledger scan per distinct pipeline rather than per dispatch.
pub fn audit_dispatch_specialisation(stem: &str) {
    audit_dispatch_specialisation_of(ledger(), stem);
}

/// The body of [`audit_dispatch_specialisation`], against any ledger.
///
/// Separated for the reason `disclose_demotions_of` is: the baked ledger records no
/// specialisation at all, so a function that could only be called on it would have exactly one
/// reachable arm and its delta count would be a number whose only possible value is zero.
pub fn audit_dispatch_specialisation_of(l: &Ledger, stem: &str) {
    if !l.faults.is_empty() {
        return;
    }
    for e in l
        .entries
        .iter()
        .filter(|e| e.shaders.iter().any(|s| s == stem))
    {
        match spec_witness_for(e) {
            SpecWitness::Delta { recorded, current } => {
                crate::counters::record_specialisation_delta(&e.key.0, &recorded, &current);
                log::warn!(
                    "[VulkanEP] [§8.9.20] proof ledger entry for `{}` was obtained under \
                     specialisation {recorded} and this run bound {current} for the same shader \
                     set: same SPIR-V, same source, different pipeline. The claim stands and is \
                     out of frame — re-prove it with rust/tools/gen_proof_ledger.py --reprove to \
                     bring the proof back into this run's frame.",
                    e.key.0
                );
            }
            SpecWitness::Unrecorded => {
                crate::counters::record_specialisation_unrecorded(&e.key.0);
            }
            SpecWitness::Unobserved | SpecWitness::Partial { .. } | SpecWitness::Identical => {}
        }
    }
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
    /// **Whole-file** damage — §8.9.18's second group, *"this artifact is not readable"*: a
    /// missing generator, a header digest that does not match the body, a header count that does
    /// not match the lines under it, a duplicate key, a line that does not parse. Non-empty means
    /// every form declines.
    ///
    /// The rule that sorts this list from the next one: **fault scope is set by the scope of what
    /// you cannot locate, not by the severity of what you found.** Each condition here is one you
    /// cannot attribute to a key — a hand-edited file may have damaged any line, a dropped entry
    /// is invisible by definition, two duplicates disagree with neither authoritative, and an
    /// unparseable line cannot be read to find out what it was going to say.
    pub faults: Vec<String>,
    /// **Entry-level** damage — §8.9.18's first group, *"this proof is not usable"*: a stale
    /// `shader_digest`, a missing subject witness, absent or zero attribution witnesses, a
    /// non-`MATCH` verdict. Each demotes its own entry and nothing else.
    ///
    /// Escalating these to `faults` was the 2026-08-02 defect: it destroyed 96 sound proofs to
    /// punish one located one, having already thrown away the localisation it held. The decisive
    /// case is `TOOLCHAIN-CHANGED`, which is ledger-wide **by nature** — a `glslc` bump changes
    /// every module's bytes at once, so under the old scope every routine compiler upgrade was a
    /// total ledger fault for a change that touched no kernel. A fail-safe guaranteed to fire
    /// spuriously on routine maintenance has a scheduled date for being switched off.
    ///
    /// The entry is simply absent from [`Ledger::entries`], so it reads as `UNMEASURED` — or as
    /// its recorded verdict, where one was preserved into [`Ledger::demoted`]. The text is kept
    /// verbatim because §8.9.18 obliges the session disclosure to **print** it: a demotion nobody
    /// is told about is the ledger quietly getting smaller.
    pub entry_faults: Vec<String>,
    /// Keys whose entry carried a verdict other than `MATCH`, with that verdict.
    ///
    /// A demotion **grants nothing** — the matching fault above already makes the whole ledger
    /// unusable. It exists so the disclosure layer can tell *measured and disagreed* from *never
    /// measured*, which RAI-008's falsifier names as two states (`DIVERGENT` and `UNMEASURED`).
    /// Collapsing them would leave a user of the §8.9.4 escape hatch, who can claim a form the
    /// ledger refuses, told only that there is "no proof" for a form the evidence says was wrong.
    pub demoted: Vec<(ProofKey, String)>,
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

    /// What this ledger says about `key` **on the running device** — the three-state answer.
    ///
    /// A method on the ledger rather than a free function so a test can plant one. A predicate
    /// that can only be exercised against the baked 97 entries can only ever be seen in the state
    /// those entries happen to produce.
    pub fn state_for(&self, key: &ProofKey) -> ProofState {
        if !self.faults.is_empty() {
            // R13: the ledger failing is an instrument error, not a finding about this form. It
            // is `Unproven` because `Unproven` is the safe state, and the *reason* stays
            // available through `LedgerLookup::Faulted`.
            return ProofState::Unproven;
        }
        match self.get(key) {
            None => ProofState::Unproven,
            Some(e) => entry_state(e),
        }
    }

    /// How many entries this build demoted, whatever the reason.
    ///
    /// §8.9.18 obliges the session disclosure to print a demotion count on every run. Once a
    /// subject mismatch stopped deleting its entry (§8.9.19 part 1), `entry_faults.len()` stopped
    /// being that number — the entries that survive carrying a `SUBJECT-CHANGED` verdict are
    /// demotions too, and a count that quietly dropped to zero the day entry survival landed
    /// would be the obligation being satisfied on paper only.
    pub fn demotion_count(&self) -> usize {
        self.entry_faults.len() + self.subject_changed_entries().count()
    }

    /// Entries whose recorded subject is not this build's code, in either unclaimable flavour.
    pub fn subject_changed_entries(&self) -> impl Iterator<Item = &LedgerEntry> {
        self.entries.iter().filter(|e| !e.subject.same_subject())
    }

    /// Entries whose SPIR-V differs from this build's while the source closure is identical —
    /// **the Linux population**, counted so a run can say how large its out-of-frame claim is.
    pub fn toolchain_delta_entries(&self) -> impl Iterator<Item = &LedgerEntry> {
        self.entries
            .iter()
            .filter(|e| matches!(e.subject, SubjectVerdict::ToolchainDelta { .. }))
    }

    /// Entries that record **no** runtime specialisation (§8.9.20) — every entry written before
    /// that section existed.
    ///
    /// Counted in the artifact so the specialisation delta's zero is interpretable. A run whose
    /// delta count is 0 because every entry is comparable and agrees, and a run whose delta count
    /// is 0 because no entry records anything to compare, are different facts, and the second one
    /// is today's.
    pub fn specialisation_unrecorded_entries(&self) -> impl Iterator<Item = &LedgerEntry> {
        self.entries.iter().filter(|e| e.spec_digest.is_empty())
    }

    /// The non-`MATCH` verdict recorded for this key, if the file carried one.    ///
    /// Never grants a claim; see [`Ledger::demoted`]. Read by the §8.9.7 session disclosure so a
    /// form the evidence measured and rejected does not read as a form nothing has measured.
    pub fn demotion_for(&self, key: &ProofKey) -> Option<&str> {
        self.demoted
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }
}

/// FNV-1a/64 over the ledger's entry lines, matching `rust/tools/gen_proof_ledger.py`.
///
/// A checksum, not a signature, and the distinction is recorded rather than smoothed: it catches
/// the careless hand-edit, which is the failure §8.9.2 rule 3 names. It does **not** catch a
/// deliberate forgery, because anyone who can edit the file can recompute it. The defence against
/// that is `check_proof_ledger.py`, which re-hashes each entry's evidence artifact — an entry
/// whose artifact does not exist or does not match is rejected there.
pub(crate) fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Digest the embedded SPIR-V of exactly the named shader stems (§8.9.11).
///
/// **The frame, stated in full, because a digest whose coverage is vague is a digest that will be
/// argued with the first time it is inconvenient.**
///
/// It COVERS: the compiled SPIR-V bytes of every module the proof run dispatched, keyed by stem.
/// A change to the GLSL that changes the compiled output — a formula, an index expression, a
/// workgroup size, a binding order — changes this digest and makes every entry that dispatched
/// that module stale. Deleting or renaming a module does too.
///
/// It DELIBERATELY DOES NOT COVER:
///
/// * **Shaders the run did not dispatch.** Staleness is scoped to the code that produced the
///   proof, so editing an unrelated kernel does not demand 73 re-proof runs. This is the whole
///   reason the digest is per-entry rather than over the shader set: a blanket digest would make
///   every routine edit demand a full re-proof, and the pressure to relax it would arrive within
///   the day.
/// * **Host-side code** — translate, allocation, descriptor construction, push-constant values.
///   **This is a named residual, not an oversight.** Its falsifier is exact: a host-only change
///   that alters numerics (a wrong push constant, a mis-sized dispatch) leaves the entry green
///   while the form's behaviour changes. `dispatches_executed` is recorded per entry and moves
///   with dispatch *structure*, so it catches some of that class and is not claimed to catch all
///   of it. Closing this properly needs a re-proof lane in CI, which is Link's frame, not a digest.
/// * **Comment-only GLSL edits**, because they do not survive to SPIR-V under our `glslc` flags.
///   A rename that changes a stem *is* covered, since the stem is part of the digest input.
///
/// Returns `None` when no module was dispatched — R12: "nothing to digest" is a different fact
/// from "the digest of nothing", and an entry whose run dispatched no shader has no subject.
pub fn shader_digest_for(stems: &[&str]) -> Option<String> {
    if stems.is_empty() {
        return None;
    }
    let mut sorted: Vec<&str> = stems.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let mut input: Vec<u8> = Vec::new();
    for stem in &sorted {
        input.extend_from_slice(stem.as_bytes());
        input.push(0);
        match crate::engine::shaders::find(stem) {
            Some(spirv) => {
                input.extend_from_slice(&(spirv.len() as u64).to_le_bytes());
                input.extend_from_slice(spirv);
            }
            // A stem the run dispatched that this build no longer embeds. Hashing a distinct
            // marker rather than skipping it means the digest *moves* — a deleted kernel must
            // make its proofs stale, not leave them looking current.
            None => input.extend_from_slice(b"\x01MODULE-ABSENT"),
        }
        input.push(0);
    }
    Some(format!("{:016x}", fnv1a64(&input)))
}

/// Digest the **source closure** of exactly the named shader stems — §8.9.19's second digest.
///
/// Per-module source digests are computed by `rust/build.rs` (see its `source_digest_for`, which
/// states the closure in full: the `.comp` text, every file reachable through the `-I` include
/// directory, the `shader_variants.txt` row, and the `glslc` argv minus the compiler binary and
/// its version). This aggregates them over a dispatch set exactly as [`shader_digest_for`]
/// aggregates SPIR-V bytes, so the two are comparable per entry.
///
/// **What makes it useful is what it is blind to**: compiler behaviour entirely — a
/// miscompilation, an optimiser difference, a codegen bug. That blindness is why a differing
/// `shader_digest` with an identical `source_digest` reads as *the compiler moved* rather than
/// *the kernel moved*. The pair is jointly blind to a compiler bug, which is precisely why the
/// resulting claim is **disclosed rather than silent** (§8.9.19 part 2 residual 1).
pub fn source_digest_for(stems: &[&str]) -> Option<String> {
    if stems.is_empty() {
        return None;
    }
    let mut sorted: Vec<&str> = stems.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let mut input: Vec<u8> = Vec::new();
    for stem in &sorted {
        input.extend_from_slice(stem.as_bytes());
        input.push(0);
        match crate::engine::shaders::source_digest(stem) {
            Some(d) => input.extend_from_slice(d.as_bytes()),
            // Same rule as `shader_digest_for`: a stem this build no longer embeds must *move*
            // the digest, not be skipped into looking current.
            None => input.extend_from_slice(b"\x01MODULE-ABSENT"),
        }
        input.push(0);
    }
    Some(format!("{:016x}", fnv1a64(&input)))
}

/// This build's shader-compiler identity — a FRAME component (§8.9.19 part 1).
pub fn toolchain_identity() -> &'static str {
    crate::engine::shaders::toolchain()
}

/// What ledger is **baked into this artifact** — digest, declared digest, entry count, faults.
///
/// # Why this exists as a readable fact rather than a comment
///
/// [`LEDGER_SOURCE`] is `include_str!`'d, which is the right call (a running process must not be
/// able to have its evidence swapped underneath it) and has one consequence nobody was told
/// about: **editing `evidence/proof_ledger.jsonl` changes nothing until the crate is rebuilt.**
/// `--reprove` would rewrite the file, the tool would report success, and the binary would go on
/// claiming from the copy it was compiled with. An effect that is invisible until an unrelated
/// action is taken is an effect that will be trusted wrongly.
///
/// The repair is not to start reading the file at run time — that would hand back exactly the
/// property baking exists to deny. It is to make the two copies *comparable from outside*, so a
/// tool can refuse when they disagree. That needs one number out of the artifact, and this is it.
///
/// Deliberately parsed from [`LEDGER_SOURCE`] here rather than read off [`ledger()`]: the
/// process-wide ledger folds in [`check_baked_against_disk`] and logs, so its digest is a fact
/// about a configured process. This is a fact about the *file that was compiled in*, and nothing
/// in the environment can move it.
pub fn baked_ledger_identity() -> String {
    let l = parse_ledger(LEDGER_SOURCE);
    format!(
        "baked_digest={}\ndeclared_digest={}\ndeclared_count={}\nentry_count={}\ndemoted={}\nfaults={}\n",
        l.actual_digest,
        l.declared_digest,
        l.declared_count,
        l.len(),
        l.demotion_count(),
        l.faults.len(),
    )
}

/// Compare one entry's recorded subject witnesses against this build's — §8.9.19 part 2's table.
pub fn subject_verdict(
    recorded_spirv: &str,
    recorded_source: &str,
    stems: &[&str],
) -> SubjectVerdict {
    let current_spirv = shader_digest_for(stems).unwrap_or_default();
    let current_source = source_digest_for(stems).unwrap_or_default();
    let spirv_same = current_spirv == recorded_spirv;
    let source_same = !recorded_source.is_empty() && current_source == recorded_source;
    match (spirv_same, recorded_source.is_empty()) {
        // Row 1 and row 4 both have identical SPIR-V. An entry with no recorded source digest
        // and identical bytes is row 1: the bytes *are* the subject, and they match.
        (true, true) => SubjectVerdict::Identical,
        (true, false) if source_same => SubjectVerdict::Identical,
        (true, false) => SubjectVerdict::SourceCosmetic {
            recorded: recorded_source.to_string(),
            current: current_source,
        },
        // Differing SPIR-V with no source witness cannot be told from a changed kernel.
        (false, true) => SubjectVerdict::Indeterminate {
            recorded_spirv: recorded_spirv.to_string(),
            current_spirv,
        },
        (false, false) if source_same => SubjectVerdict::ToolchainDelta {
            recorded: recorded_spirv.to_string(),
            current: current_spirv,
        },
        (false, false) => SubjectVerdict::Changed {
            recorded_spirv: recorded_spirv.to_string(),
            current_spirv,
        },
    }
}

/// Pull a `"field": ["a", "b"]` array of strings out of a JSON object line.
///
/// Same deliberate smallness as [`json_field`]: our generator emits a fixed shape. Returns `None`
/// when the field is absent, so absent stays distinguishable from an empty array.
fn json_str_array_field(line: &str, field: &str) -> Option<Vec<String>> {
    // The generator emits compact JSON (`"field":[...]`); hand-written fixtures and the counters
    // artifact use a space. Accept both rather than making the reader depend on a formatter.
    let start = [format!("\"{field}\":["), format!("\"{field}\": [")]
        .iter()
        .find_map(|needle| line.find(needle.as_str()).map(|i| i + needle.len()))?;
    let end = line[start..].find(']')? + start;
    let body = &line[start..end];
    if body.trim().is_empty() {
        return Some(Vec::new());
    }
    Some(
        body.split(',')
            .map(|p| p.trim().trim_matches('"').to_string())
            .filter(|p| !p.is_empty())
            .collect(),
    )
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

fn json_field(line: &str, field: &str) -> Option<String> {
    let needle = format!("\"{field}\":");
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
///
/// pub(crate) so the disclosure layer's tests can plant a ledger with a demoted entry. A
/// demotion path that can only be exercised against the baked 103 entries has no observable
/// firing state, and §8.9.18 forbids exactly that.
pub(crate) fn parse_ledger(source: &str) -> Ledger {
    let mut faults: Vec<String> = Vec::new();
    // Entry-level faults are kept apart from whole-file ones. See `Ledger::entry_faults`.
    let mut entry_faults: Vec<String> = Vec::new();
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
            let end = rest
                .find(|c: char| !c.is_ascii_digit())
                .unwrap_or(rest.len());
            Some(rest[..end].to_string())
        })
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0);
    if generator.is_empty() {
        faults.push("ledger header names no generator".to_string());
    }

    let mut entries = Vec::new();
    let mut demoted: Vec<(ProofKey, String)> = Vec::new();
    let mut digest_input = String::new();
    for line in lines {
        digest_input.push_str(line);
        digest_input.push('\n');
        // §8.9.18: **a line that does not parse faults the artifact, not an entry.** You cannot
        // say which claims it touches, because you cannot tell what it was going to say. This is
        // the boundary of the localisation rule, and it is the one place in this loop where the
        // whole-file fault is the correct scope rather than the lazy one.
        let Some(raw_key) = json_field(line, "key") else {
            faults.push(format!("ledger line has no `key` field: {line}"));
            continue;
        };
        let key = match ProofKey::validate(&raw_key) {
            Ok(k) => k,
            Err(e) => {
                faults.push(format!(
                    "ledger key {raw_key:?} is not a valid proof key: {e}"
                ));
                continue;
            }
        };
        let verdict = json_field(line, "verdict").unwrap_or_default();
        if verdict != "MATCH" {
            // §8.9.2 rule 4: only a MATCH proves. Anything else in this file is a generator bug
            // or a hand-edit, and either way the entry does not get to grant a claim.
            //
            // The entry is also remembered as a **demotion** (added 2026-08-02, Tank, for
            // §8.9.7's session WARN). The fault above stops it granting a claim; the demotion
            // preserves *what it said*. Without this, a form the evidence measured and found
            // DIVERGENT and a form nothing has ever measured both arrive at the disclosure layer
            // as "no proof", and RAI-008's falsifier names those as two states, not one. It
            // grants nothing: `get`/`lookup_key` still refuse while `faults` is non-empty.
            demoted.push((key, verdict.clone()));
            entry_faults.push(format!(
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
            entry_faults.push(format!(
                "ledger entry for {raw_key:?} carries no attribution witness \
                 (claimed_nodes/dispatches_executed); it does not record a proof run and may \
                 have been enumerated from the claim table rather than proven"
            ));
            continue;
        };
        if claimed_nodes == 0 || dispatches_executed == 0 {
            entry_faults.push(format!(
                "ledger entry for {raw_key:?} records claimed_nodes={claimed_nodes} \
                 dispatches_executed={dispatches_executed}; a run that claimed or dispatched \
                 nothing is UNATTRIBUTED and proves nothing"
            ));
            continue;
        }
        // §8.9.11 — SUBJECT PROVENANCE. `claimed_nodes`/`dispatches_executed` establish that a
        // run happened; they say nothing about *what code* it ran. Switch changed the GQA shader
        // twice on 2026-08-02 and the entry proving that form predated both changes, while
        // `--append` skipped re-measuring it because the form was already claimed. The ledger
        // agreed with him — and would have agreed identically had he broken the kernel.
        //
        // A stale entry demotes ITSELF and nothing else. Making it a global fault would let one
        // shader edit disable every claim in the artifact, which is the blunt shape that gets
        // relaxed the first time someone is in a hurry.
        let shaders = json_str_array_field(line, "shaders");
        let recorded_digest = json_field(line, "shader_digest");
        let (Some(shaders), Some(recorded_digest)) = (shaders, recorded_digest) else {
            demoted.push((key, "NO-SUBJECT-WITNESS".to_string()));
            entry_faults.push(format!(
                "ledger entry for {raw_key:?} names no shader set or digest; it cannot say what \
                 code it proved, so it cannot be invalidated by that code changing. Re-prove it \
                 with `gen_proof_ledger.py --reprove`."
            ));
            continue;
        };
        let stems: Vec<&str> = shaders.iter().map(String::as_str).collect();
        let source_digest = json_field(line, "source_digest").unwrap_or_default();
        // §8.9.19 PART 1 — ENTRY SURVIVAL. A subject-or-frame mismatch **records itself on the
        // entry and leaves the entry findable**. It used to `continue` here, so the entry never
        // reached `Ledger::entries`, `Ledger::get` returned `None`, and `lookup_key` reported the
        // same token it reports for a form nobody ever proved. That is why Linux read as "97
        // forms were never proven" rather than "97 proofs were obtained under a different
        // compiler" — a frame mismatch was indistinguishable from a key absence. **The predicate,
        // not the parser, decides what a mismatch licenses.**
        //
        // The one thing that still deletes an entry here is an *absent* subject witness, above:
        // that is not a mismatch, it is an entry that cannot say what it proved, and there is
        // nothing for a predicate to compare.
        if shader_digest_for(&stems).is_none() {
            demoted.push((key.clone(), "NO-SUBJECT-WITNESS".to_string()));
            entry_faults.push(format!(
                "ledger entry for {raw_key:?} names an empty shader set; a run that \
                 dispatched no module has no subject to have proven"
            ));
            continue;
        }
        let subject = subject_verdict(&recorded_digest, &source_digest, &stems);
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
            shaders,
            shader_digest: recorded_digest,
            source_digest,
            toolchain: json_field(line, "toolchain").unwrap_or_default(),
            spec_digest: json_field(line, "spec_digest").unwrap_or_default(),
            subject,
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
    if declared_count != entries.len() + entry_faults.len() && faults.is_empty() {
        faults.push(format!(
            "ledger header declares {declared_count} entries, {} parsed and {} demoted",
            entries.len(),
            entry_faults.len()
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
        entry_faults,
        demoted,
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

/// The physical device name(s) this run has actually opened, or `None` if it has not opened one.
///
/// **Read off the run, never off the selector.** `ONNXRUNTIME_EP_VULKAN_DEVICE` is a *request*:
/// Trinity demonstrated `DEVICE=0` running on `1=NVIDIA`, and `device0` on this desk is not
/// `device0` on a CI box. A predicate keyed on the selector would compare two requests and call
/// the result an identity. This reads `allocator::tally::session_devices()`, which is populated by
/// the session that actually opened the `VkDevice`.
pub fn running_device_names() -> Vec<String> {
    let names = crate::allocator::tally::session_devices();
    if names == "none" || names == "unknown" {
        return Vec::new();
    }
    names
        .split("; ")
        .filter_map(|pair| pair.split_once('='))
        .map(|(_, name)| name.to_string())
        .collect()
}

/// Whether `label` is a selector ordinal (`device0`, `device7`) rather than a device name.
///
/// `gen_proof_ledger.py` writes `device{N}` from the selector unless `--device-name` is given, so
/// this is the shape of all 97 baked entries. It is the *classifier* on which the whole predicate
/// turns: an ordinal names no hardware, so an entry carrying one cannot be compared to a running
/// device at all, and saying so is different from saying the devices differ.
pub fn is_selector_ordinal(label: &str) -> bool {
    label
        .strip_prefix("device")
        .is_some_and(|rest| !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()))
}

/// What the ledger says about one form **in this run's frame** — §8.9.19 part 1's status lattice.
///
/// The generating rule: **you look up by KEY; you compare FRAME after you have looked up; a
/// SUBJECT mismatch means the proof is about something else.** A frame component in the key turns
/// "I have a proof that does not apply here" into "I have no proof", and only the first of those
/// is actionable.
///
/// `PROVEN-ELSEWHERE` **generalises** rather than acquiring siblings: it carries a **delta set**
/// δ ⊆ {`device`, `driver`, `ort_build`, `toolchain`} instead of the enum growing as a product of
/// combinations.
///
/// **Claiming is not promoting.** §8.9.18 withdrew `PROVEN-ELSEWHERE`'s licence to be *promoted*
/// to `PROVEN` by a model run, and that withdrawal stands: nothing here promotes anything. What
/// §8.9.19 grants is the licence to **claim while saying out loud that the claim is out of
/// frame**, which is what lets Link's op suite run at all — and the suite is a genuine per-form
/// differential, so it is what removes the need for the grant. The ruling is self-discharging.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProofState {
    /// A sound entry exists, its subject is this build's code, and its frame is this run's frame.
    /// Claimable, silently.
    Proven,
    /// A sound entry exists about this exact subject, obtained under a frame that differs in
    /// **exactly** the components named in `deltas`.
    ///
    /// **Claimable — counted, disclosed, and δ printed in the run record.** Declining it would
    /// mean a Linux run declines every form and produces no op-correctness number at all, which
    /// is the state being unblocked; claiming it silently would be the extrapolation §8.9.17
    /// refused. So it claims, and it says so.
    ProvenElsewhere {
        /// Which frame components differ. Sorted and deduplicated.
        deltas: Vec<FrameDelta>,
        /// Human-readable specifics, one clause per delta.
        detail: String,
    },
    /// A sound entry whose SPIR-V is byte-identical to this build's while its **source closure**
    /// differs — an edit that produced identical compiled output.
    ///
    /// **Claimable, and named.** This is the row that demonstrates the digest pair is doing work
    /// rather than one digest wearing two hats: without the source digest it is indistinguishable
    /// from `Proven`, and without the SPIR-V digest it is indistinguishable from a kernel change.
    SourceCosmetic { recorded: String, current: String },
    /// A sound entry exists but **no device comparison is possible**: the entry carries a selector
    /// ordinal (so it names no hardware), or this run has not yet opened a device (so it has no
    /// name to be compared against).
    ///
    /// **Claimable, counted, and disclosed.** Claimable because declining it would take the EP to
    /// zero claims on evidence that is not actually in question; counted and disclosed because
    /// this is exactly the population the ledger cannot attribute to hardware, and an
    /// unattributable proof that nothing reports is the fail-open Link found wearing a different
    /// hat.
    DeviceUnattributed {
        entry_label: String,
        reason: &'static str,
    },
    /// An entry exists and **its subject moved** — both digests differ, so the kernel it proves
    /// has been replaced. Not claimable.
    SubjectChanged {
        /// The SPIR-V digest recorded on the entry.
        recorded: String,
        /// What this build's modules hash to.
        current: String,
        /// Whether the source closure could be compared at all. `false` is
        /// [`SubjectVerdict::Indeterminate`]: a pre-§8.9.19 entry whose SPIR-V differs, where
        /// "different compiler" and "different kernel" cannot be told apart and the fail-safe
        /// reading is the second one.
        source_comparable: bool,
    },
    /// No entry, a demoted one, or a ledger that did not parse. Not claimable.
    ///
    /// **This is the state a missing key lands in and it must stay that way.** No absent key may
    /// ever reach a frame-flavoured state, or the frame vocabulary becomes a silent fallback
    /// rather than a reading of evidence.
    Unproven,
}

impl ProofState {
    /// The single token an artifact records.
    pub fn token(&self) -> &'static str {
        match self {
            ProofState::Proven => "PROVEN",
            ProofState::ProvenElsewhere { .. } => "PROVEN-ELSEWHERE",
            ProofState::SourceCosmetic { .. } => "SOURCE-COSMETIC",
            ProofState::SubjectChanged { .. } => "UNPROVEN{SUBJECT-CHANGED}",
            ProofState::DeviceUnattributed { .. } => "DEVICE-UNATTRIBUTED",
            ProofState::Unproven => "UNPROVEN",
        }
    }

    /// Whether this state admits a claim.
    ///
    /// Two states decline, and both because the evidence is about **something else** rather than
    /// somewhere else: `Unproven` (there is none) and `SubjectChanged` (it is about a kernel that
    /// has been replaced). `ProvenElsewhere` claims — §8.9.19 part 1 — because the alternative is
    /// that a Linux run declines all 97 forms and produces no op-correctness number at all, and
    /// because claiming *while disclosing δ* is a strictly more informative act than declining.
    pub fn claimable(&self) -> bool {
        matches!(
            self,
            ProofState::Proven
                | ProofState::ProvenElsewhere { .. }
                | ProofState::SourceCosmetic { .. }
                | ProofState::DeviceUnattributed { .. }
        )
    }

    /// The δ set this state carries, empty for every state that is not out of frame.
    pub fn deltas(&self) -> &[FrameDelta] {
        match self {
            ProofState::ProvenElsewhere { deltas, .. } => deltas,
            _ => &[],
        }
    }
}

/// Classify one entry's `device` against the device this run opened.
///
/// Kept as its own function because the device question is answerable on its own — the disclosure
/// layer asks it directly — but it is **no longer the whole predicate**: [`entry_state`] composes
/// it with the subject verdict, since §8.9.19 makes the toolchain a frame component beside the
/// device rather than a separate mechanism.
pub fn device_state(entry_device: &str) -> ProofState {
    if entry_device.is_empty() {
        return ProofState::DeviceUnattributed {
            entry_label: String::new(),
            reason: "the entry records no device at all",
        };
    }
    if is_selector_ordinal(entry_device) {
        return ProofState::DeviceUnattributed {
            entry_label: entry_device.to_string(),
            reason: "the entry records a selector ordinal, which names no hardware — a selector is \
                     a request, not an identity",
        };
    }
    let running = running_device_names();
    if running.is_empty() {
        return ProofState::DeviceUnattributed {
            entry_label: entry_device.to_string(),
            reason: "this run has not opened a device yet, so there is no name to compare against",
        };
    }
    if running.iter().any(|n| n == entry_device) {
        ProofState::Proven
    } else {
        ProofState::ProvenElsewhere {
            deltas: vec![FrameDelta::Device],
            detail: format!(
                "device: entry proved on `{entry_device}`, this run opened `{}`",
                running.join("; ")
            ),
        }
    }
}

/// Resolve one **entry** to its state — subject first, then frame (§8.9.19 part 1).
///
/// Order is the ruling's, not a convenience: a subject mismatch means the proof is about
/// something else, and no amount of frame agreement rescues that. Only once the subject is
/// established as this build's code does the frame comparison mean anything.
pub fn entry_state(entry: &LedgerEntry) -> ProofState {
    let mut deltas: Vec<FrameDelta> = Vec::new();
    let mut detail: Vec<String> = Vec::new();

    match &entry.subject {
        SubjectVerdict::Changed {
            recorded_spirv,
            current_spirv,
        } => {
            return ProofState::SubjectChanged {
                recorded: recorded_spirv.clone(),
                current: current_spirv.clone(),
                source_comparable: true,
            };
        }
        SubjectVerdict::Indeterminate {
            recorded_spirv,
            current_spirv,
        } => {
            return ProofState::SubjectChanged {
                recorded: recorded_spirv.clone(),
                current: current_spirv.clone(),
                source_comparable: false,
            };
        }
        SubjectVerdict::ToolchainDelta { recorded, current } => {
            deltas.push(FrameDelta::Toolchain);
            detail.push(format!(
                "toolchain: the entry's SPIR-V hashes to {recorded} and this build's to {current} \
                 from an identical source closure; entry-toolchain={}, running-toolchain={}",
                if entry.toolchain.is_empty() {
                    "<unrecorded>"
                } else {
                    &entry.toolchain
                },
                toolchain_identity()
            ));
        }
        SubjectVerdict::Identical | SubjectVerdict::SourceCosmetic { .. } => {}
    }

    // §8.9.20 — THE DISPATCH-TIME FRAME COMPONENT.
    //
    // Read after the subject and beside the device, because it is a frame fact and not a subject
    // one: the SPIR-V and the source are both identical across a specialisation delta, and the
    // pipeline is not. `Unobserved`/`Partial` add no delta — nothing has been compared — and
    // `Unrecorded` adds none either, for the reason on `SpecWitness::Unrecorded`: it is a
    // narrowing of what the entry means, disclosed through its own count, not a mismatch.
    if let SpecWitness::Delta { recorded, current } = spec_witness_for(entry) {
        deltas.push(FrameDelta::Specialisation);
        detail.push(format!(
            "specialisation: the entry was proven under {recorded} and this run bound {current} \
             for the same shader set — identical SPIR-V and identical source, different pipeline"
        ));
    }

    let device = device_state(&entry.device);
    match &device {
        ProofState::ProvenElsewhere {
            detail: device_detail,
            ..
        } => {
            deltas.push(FrameDelta::Device);
            detail.push(device_detail.clone());
        }
        ProofState::DeviceUnattributed { .. } if deltas.is_empty() => {
            // Nothing else is out of frame, so the device's incomparability is the whole finding
            // and it keeps its own state — that population is counted and named separately.
            return device;
        }
        _ => {}
    }

    if !deltas.is_empty() {
        deltas.sort_unstable();
        deltas.dedup();
        return ProofState::ProvenElsewhere {
            deltas,
            detail: detail.join("; "),
        };
    }
    if let SubjectVerdict::SourceCosmetic { recorded, current } = &entry.subject {
        return ProofState::SourceCosmetic {
            recorded: recorded.clone(),
            current: current.clone(),
        };
    }
    device
}

/// Resolve one key to its [`ProofState`] on the running device.
pub fn proof_state(key: &ProofKey) -> ProofState {
    ledger().state_for(key)
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
        // Entry-level faults are warned too, and named. They demote only their own entry, but a
        // demotion nobody is told about is a proof that silently stopped existing.
        for f in &l.entry_faults {
            log::warn!("[VulkanEP] proof ledger entry demoted: {f}");
        }
        l
    })
}

/// The detail text of an `[unproven]` decline, **chosen by which of the three answers the lookup
/// gave** (Link, 2026-08-03).
///
/// `Ledger::state_for` blankets to `Unproven` when the artifact is faulted — R13, and correct,
/// because a faulted instrument must not grant claims. The message then said *"no proof ledger
/// entry for X. The kernel exists; nothing has proven it correct on this form"*, and on a faulted
/// ledger **every clause of that is false**: there may well be an entry, something may well have
/// proven it, and this build cannot tell which. It also named the wrong repair — regenerating the
/// entry for one form does not fix a damaged file.
///
/// `LedgerLookup` has carried the distinction since R13 and nothing here read it. A blanket in the
/// *state* is a safety property; a blanket in the *text* is a false statement, and they do not
/// have to travel together.
///
/// A free function rather than an inline branch so both arms are reachable from a test: the
/// process-wide ledger is a `OnceLock`, so a test cannot fault it after something has looked a key
/// up, and a message only reachable through a faulted global is a message nothing asserts on.
pub fn unproven_decline_detail(outcome: LedgerLookup, key: &ProofKey) -> String {
    if outcome == LedgerLookup::Faulted {
        return format!(
            "the proof ledger could not be read, so nothing is known about `{}` from it — this \
             is an INSTRUMENT failure, not a finding that the form is unproven. Every form \
             declines while it lasts, and the entry proving this one may well be sitting in the \
             file. Repair the artifact: rust/tools/gen_proof_ledger.py --check names the fault, \
             and the session disclosure prints it in full.",
            key.0
        );
    }
    if !form_is_provable(key) {
        // The same defect Link found, one axis over: the sentence below asserts "the kernel
        // exists" and names a proof run as the repair. For a form whose module declares a
        // capability the engine does not enable, the module exists and **no pipeline can be
        // created from it on any device we run on**, so a proof run has nothing to measure and
        // the advice sends the reader to a tool that will report "no unlockable keys". Measured
        // 2026-08-03 on `ai.onnx::Cast/6+/i64>i32/ew_cast_i64_to_i32` in both shape classes.
        return format!(
            "no proof ledger entry for `{}`, and no proof run can produce one: the module `{}` \
             declares a SPIR-V capability this engine does not enable, so no pipeline can be \
             created from it on any device we run on and there is nothing for a proof run to \
             measure. This form needs a device feature, not evidence — see \
             ops/common/variants.rs::ENGINE_ENABLED_CAPABILITIES. The node is also declined \
             [dtype] for the same reason, which is the decline that binds.",
            key.0,
            key.variant_stem().unwrap_or("?"),
        );
    }
    format!(
        "no proof ledger entry for `{}`. The kernel exists; nothing has proven it correct on this \
         form, so it runs on the CPU EP, which is always right. Prove it with \
         rust/tools/gen_proof_ledger.py, or enable it for development with \
         ONNXRUNTIME_EP_VULKAN_CLAIM_UNPROVEN={}",
        key.0, key.0
    )
}

/// Could a proof run ever clear this form's `[unproven]` decline?
///
/// Reads [`variant_is_loadable`], which is a pure function of the checked-in SPIR-V and the
/// engine's enabled-capability list — **no device handle and no global**, so it answers the same
/// way before and after device creation. That matters here: this runs on the claim path, where a
/// value that changed once a device existed would be the time-dependent-global defect this
/// project has now shipped three times.
///
/// **It answers `true` whenever it is not sure**, and the boundary of "not sure" is the repair
/// Morpheus's §8.9.23 finding forced. A key whose variant component names no module the registry
/// declares — `metadata` on a genuinely composite row, or a malformed key — is unknown, and
/// unknown under-claims: composite forms *are* provable, several are proven in the ledger today,
/// and the first version of this predicate reported
/// `ai.onnx::Gather/1+/i64,i64>i64/metadata/static/n2` unprovable on the strength of a stem that
/// names nothing.
///
/// But a key whose variant **is** a module the registry declares is not unknown, and it must not
/// be answered from the same branch. That was the defect: `Conv`'s keys rendered `metadata`
/// because its row said `kernel!(None)` while its entries recorded `"shaders":["conv_f32"]`, so
/// the predicate took the unknown branch and answered *provable* — including in a build with no
/// SPIR-V at all, which is the positive control this predicate was supposed to have and did not.
/// A declared stem now answers [`variant_is_loadable`] directly, so a shaderless build reports
/// every hand-written row unprovable, which is true.
///
/// The published list is therefore still a **lower bound** on what no proof run can clear:
/// everything on it has a declared module that cannot be created, and a form may be unreachable
/// for other reasons without appearing.
fn form_is_provable(key: &ProofKey) -> bool {
    let Some(stem) = key.variant_stem() else {
        return true;
    };
    use crate::ops::common::variants::{
        variant_is_declared, variant_is_generated, variant_is_loadable,
    };
    form_provable_from(
        variant_is_declared(stem),
        variant_is_generated(stem),
        variant_is_loadable(stem),
    )
}

/// Answer [`form_is_provable`] for a list of keys, as text, for a caller outside this process.
///
/// # Why an export exists for a predicate the claim path already consults
///
/// `gen_proof_ledger.py --check` cannot tell whether a ledger key is **mintable** — whether any
/// proof run on this build could ever produce an entry for it. All 43 retired keys pass `--check`
/// cleanly, and the check has no way to separate *"retired on purpose because the form stopped
/// existing"* from *"never mintable on this build"*. Those have different repairs and the second
/// one is a capability regression that no other instrument in the tree reports.
///
/// The only place the answer exists is the compiled artifact: `variant_is_loadable` reads the
/// SPIR-V that `build.rs` baked in, and `ENGINE_ENABLED_CAPABILITIES` is checked-in source. A
/// Python re-derivation would be a second implementation of the capability rule, which is the
/// mirror-drift failure `OrtEpVulkanGetShaderSubject` already exists to avoid.
///
/// # The purity this preserves
///
/// [`form_is_provable`] reads no device handle and no global written at device creation, so it
/// answers identically before and after device creation. This function adds no state of its own
/// and is therefore callable from a process that has never created a `VkDevice` — which is
/// exactly how `--check` calls it, via `ctypes.CDLL` with no ORT session at all. **Do not give
/// this path a device-created global**; a mintability answer that depends on whether a device
/// exists is a different question wearing this one's name.
///
/// # Format
///
/// Input is a NUL-terminated, **newline**-separated list of proof keys — newline rather than
/// comma because a dtype signature contains commas. One output line per input key, tab-separated:
///
/// ```text
/// <key>\tmintable=yes|no\tstem=<stem|->\tdeclared=yes|no\tgenerated=yes|no\tloadable=yes|no
/// ```
///
/// `stem=-` is a key with no parseable variant component. Those answer `mintable=yes`, matching
/// [`form_is_provable`]'s deliberate under-claim: unknown is not the same as refused, and the
/// published unmintable list is a lower bound.
pub fn form_mintability_report(keys: &[&str]) -> String {
    use crate::ops::common::variants::{
        variant_is_declared, variant_is_generated, variant_is_loadable,
    };
    let yn = |b: bool| if b { "yes" } else { "no" };
    let mut out = String::new();
    for raw in keys {
        let key = ProofKey::parse(raw);
        match key.variant_stem() {
            None => {
                out.push_str(&format!(
                    "{}\tmintable=yes\tstem=-\tdeclared=no\tgenerated=no\tloadable=no\n",
                    key.0
                ));
            }
            Some(stem) => {
                let declared = variant_is_declared(stem);
                let generated = variant_is_generated(stem);
                let loadable = variant_is_loadable(stem);
                out.push_str(&format!(
                    "{}\tmintable={}\tstem={}\tdeclared={}\tgenerated={}\tloadable={}\n",
                    key.0,
                    yn(form_provable_from(declared, generated, loadable)),
                    stem,
                    yn(declared),
                    yn(generated),
                    yn(loadable),
                ));
            }
        }
    }
    out
}

/// [`form_is_provable`]'s decision, as a pure function of its three inputs.
///
/// Split out so the **shaderless build** — the positive control this predicate shipped without —
/// is reachable from a test. `variant_is_generated` and `variant_is_loadable` both read
/// `SHADER_MODULES`, which is baked in by `build.rs`, so a test inside this process cannot make
/// the build have no SPIR-V; it can only be handed the booleans that build would produce.
fn form_provable_from(declared: bool, generated: bool, loadable: bool) -> bool {
    if declared {
        return loadable;
    }
    !generated || loadable
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
    claim_decision_audited(view).decision()
}

/// The same decision, returning the whole [`ClaimAudit`] instead of only its verdict.
///
/// Added 2026-08-02 (Tank) for §8.9.7's session-creation disclosure, which needs each *claimed*
/// node's `proof_key` and `ledger_hit`. The alternative — calling [`claim_audit`] a second time
/// for claimed nodes — would double-count `proven_key_lookups`, and that counter is criterion
/// 11's evidence: an instrument that moves because a second reader looked at it is not measuring
/// the thing it names. One audit per node, one lookup per node, two readers.
pub fn claim_decision_audited(view: &NodeView<'_>) -> ClaimAudit {
    let logging = crate::ops::claim_log::enabled();
    let audit = claim_audit(view, logging);
    if logging {
        // The edge types are collected here rather than inside the audit because they are for the
        // record only: nothing in the claim decision reads them, and widening `ClaimAudit` would
        // put an allocation on the hot path for a field only the census consumes.
        let inputs = view.input_types();
        let outputs = view.output_types();
        crate::ops::claim_log::record_audit(
            &view.qualified_name(),
            &view.name(),
            view.since_version(),
            &audit,
            Some((&inputs, &outputs)),
        );
    }
    audit
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
    ///
    /// **True for `PROVEN-ELSEWHERE` as well as `PROVEN`** — it answers "did evidence admit this",
    /// which both states do. Read [`ClaimAudit::proof_state`] for which one.
    pub ledger_hit: bool,
    /// `PROVEN` / `PROVEN-ELSEWHERE` / `UNPROVEN` for this node on **this device** (§10.0.1 R12).
    pub proof_state: ProofState,
    /// Whether any edge reading this decision rested on came from the graph-level rank overlay
    /// (§8.11) rather than from ORT directly. Recorded so the claim log's `input_shapes` and
    /// `output_shapes` are never mistaken for ORT's own answers. See [`NodeView::rank_inferred`].
    pub rank_inferred: bool,
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
            proof_state: ProofState::Unproven,
            rank_inferred: view.rank_inferred(),
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
    let ledger_outcome = if spec.is_live() {
        // RAI-008(d): the outcome, not a `bool`. A miss that was the ledger failing and a miss
        // that was this form being absent are two findings with two different repairs, and the
        // counters artifact has to be able to say which one happened.
        let outcome = lookup_key(&proof_key);
        crate::counters::record_ledger_lookup(outcome);
        outcome
    } else {
        LedgerLookup::NeverAttempted
    };
    let form_state = if spec.is_live() {
        // R12: the entry's frame, read rather than merely recorded. `lookup_key` answers "is
        // there an entry"; this answers "is there an entry *for this device*", and the two used
        // to be the same question because nothing consulted `.device`.
        proof_state(&proof_key)
    } else {
        ProofState::Unproven
    };
    let ledger_hit = form_state.claimable();
    let hatch = if spec.is_live() && !ledger_hit {
        let enabled = claim_unproven_keys().contains(&proof_key);
        if enabled {
            crate::counters::record_unproven_form_enabled(&proof_key.0);
        }
        enabled
    } else {
        if spec.is_live() && claim_unproven_keys().contains(&proof_key) {
            // §8.9.11 re-proof. The ledger admitted this form, so the hatch is not what let it
            // through and it must NOT appear in `unproven_forms_enabled`. But the harness still
            // has to attribute the run to a key, and `--reprove` deliberately offers keys that
            // are already proven. Without this witness the generator sees an empty admission set
            // and reports `UNATTRIBUTED`, so a re-proof measures nothing — which is how the
            // entry outlives its subject, one level up from the hole Switch found.
            crate::counters::record_reproof_form_admitted(&proof_key.0);
        }
        false
    };
    if spec.is_live() && !ledger_hit && !hatch {
        crate::counters::record_unproven_decline(&proof_key.0);
        if !form_is_provable(&proof_key) {
            // Recorded beside the count, not instead of it: the form really does lack an entry.
            // What it also lacks is any way to acquire one, and a backlog that mixes the two
            // reads as work somebody could do.
            crate::counters::record_unprovable_decline(&proof_key.0);
        }
        match &form_state {
            ProofState::SubjectChanged {
                recorded,
                current,
                source_comparable,
            } => {
                crate::counters::record_subject_changed_decline(&proof_key.0);
                let why = if *source_comparable {
                    "both the SPIR-V and the source closure differ, so the kernel itself moved"
                } else {
                    "the SPIR-V differs and the entry records no source_digest, so `different \
                     compiler` and `different kernel` cannot be told apart — the fail-safe \
                     reading is the second. Backfill the entry's frame on a machine whose SPIR-V \
                     does match, with `gen_proof_ledger.py --backfill-frame`"
                };
                failures.push(decline(
                    DeclineCode::Unproven,
                    format_args!(
                        "the proof ledger entry for `{}` was obtained against shader digest \
                         {recorded} and this build's modules hash to {current}: {why}. A proof \
                         that survives a change to its subject is not a proof of that subject \
                         (§8.9.19), so this form runs on the CPU EP. Re-prove it with \
                         rust/tools/gen_proof_ledger.py --reprove.",
                        proof_key.0
                    ),
                ));
            }
            _ => {
                failures.push(decline(
                    DeclineCode::Unproven,
                    format_args!("{}", unproven_decline_detail(ledger_outcome, &proof_key)),
                ));
            }
        }
    }

    let predicate_ok_with_runtime_extents = if !with_counterfactual {
        predicate_ok
    } else if predicate_ok {
        true
    } else {
        let _guard = AssumeRuntimeExtents::on();
        (spec.claim)(view, spec).is_ok()
    };

    // The unattributed-device disclosure, recorded **only when the node is actually claimed**.
    //
    // Counting it at the lookup instead would count nodes that the ledger admitted and something
    // else declined, and a disclosure that overstates what was claimed is as wrong as one that
    // understates it. `failures.is_empty()` is the claim, and this is the last thing before the
    // audit is returned so that it is read after every check has had its say.
    //
    // This is the answer to "what stops it becoming the default nobody looks at?": the default
    // *is* this state today — all 97 entries carry a selector ordinal — and it is counted on every
    // claim and named per form in the session disclosure. A field that changes no outcome and
    // appears in no artifact is the thing being repaired; a field that appears in every artifact
    // is not that thing, even while it changes no outcome yet.
    if failures.is_empty() {
        match &form_state {
            ProofState::DeviceUnattributed {
                entry_label,
                reason,
            } => crate::counters::record_device_unattributed(&proof_key.0, entry_label, reason),
            // §8.9.19: a claim granted out of frame is counted and named at the moment it is
            // granted. This is the counter that used to be `proven_elsewhere_declines` and could
            // only ever read zero on a healthy run — a counter whose only observable value is
            // zero is not an instrument.
            ProofState::ProvenElsewhere { deltas, detail } => {
                let delta_tokens: Vec<&str> = deltas.iter().map(FrameDelta::token).collect();
                crate::counters::record_proven_elsewhere_claim(
                    &proof_key.0,
                    &delta_tokens.join(","),
                    detail,
                );
            }
            ProofState::SourceCosmetic { recorded, current } => {
                crate::counters::record_source_cosmetic_claim(&proof_key.0, recorded, current);
            }
            _ => {}
        }
        // THE SUBJECT AXIS IS RECORDED WHATEVER THE FRAME AXIS SAID.
        //
        // `ProofState` is single-valued and the frame verdict outranks a cosmetic subject move,
        // so on today's ledger — every entry device-unattributed — the arm above can never run
        // and `source_cosmetic_claims` would read zero forever. §8.9.19 names row 4 as the row
        // that demonstrates the digest *pair* is working, so it cannot be observable only in a
        // ledger state that does not exist yet.
        if !matches!(form_state, ProofState::SourceCosmetic { .. })
            && let Some(entry) = ledger().get(&proof_key)
            && let SubjectVerdict::SourceCosmetic { recorded, current } = &entry.subject
        {
            crate::counters::record_source_cosmetic_claim(&proof_key.0, recorded, current);
        }
    }

    ClaimAudit {
        primary: failures.first().cloned(),
        failures,
        unevaluated: Vec::new(),
        shape_class,
        predicate_ok,
        predicate_ok_with_runtime_extents,
        proof_key: Some(proof_key),
        ledger_hit,
        proof_state: form_state,
        rank_inferred: view.rank_inferred(),
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

    /// **The ledger that ships must carry a frame the build that ships can compare.**
    ///
    /// `SubjectVerdict::Indeterminate` is the reading for an entry whose SPIR-V moved and which
    /// records no `source_digest` — every entry written before §8.9.19. On a second toolchain
    /// that reading is a decline, and it is indistinguishable from a rewritten kernel. This is
    /// the guard against the baked ledger drifting back into that state through a hand edit or a
    /// generator that forgets the field; `gen_proof_ledger.py --check` enforces the same thing
    /// from the other side.
    #[test]
    fn every_baked_entry_records_a_frame_that_can_be_compared() {
        let l = ledger();
        let frameless: Vec<&str> = l
            .entries()
            .iter()
            .filter(|e| e.source_digest.is_empty() || e.toolchain.is_empty())
            .map(|e| e.key.0.as_str())
            .collect();
        assert!(
            frameless.is_empty(),
            "{} baked entr(ies) carry no source_digest/toolchain. On any compiler but the one \
             that wrote them they read as SUBJECT-CHANGED and decline, which is the Linux \
             symptom §8.9.19 exists to remove. Repair: rust/tools/gen_proof_ledger.py \
             --backfill-frame. Offenders: {:?}",
            frameless.len(),
            &frameless[..frameless.len().min(5)]
        );
    }

    /// **§8.9.19 row 2, the Linux case, in the state that unblocks it.**
    ///
    /// Ubuntu's shaderc 2023.8 and the Windows SDK's v2026.2 emit different SPIR-V for identical
    /// GLSL. This machine has exactly one SDK, so the second compiler is *modelled*: the recorded
    /// SPIR-V digest is one a different compiler produced and the recorded source digest is this
    /// tree's, which is precisely the artifact Link's Linux lane presents. Every other value here
    /// is real — the source digest is read out of this build, not written down.
    ///
    /// The three rows are asserted against each other in one test on purpose. A row asserted
    /// alone passes whenever the predicate collapses to a constant; what makes the *pair* of
    /// digests an instrument is that the four combinations do not agree.
    #[test]
    fn the_digest_pair_separates_a_second_compiler_from_a_second_kernel() {
        const KEY: &str = "ai.onnx::Add/7+/f32,f32>f32/ew_binary_add_f32/static/n2";
        let key = ProofKey::validate(KEY).expect("valid key");
        let this_spirv = shader_digest_for(&["ew_binary_add_f32"]).expect("a stem to digest");
        let this_source = source_digest_for(&["ew_binary_add_f32"]).expect("a stem to digest");
        let build = |spirv: &str, source: &str| {
            let entry = format!(
                "{{\"key\":\"{KEY}\",\"verdict\":\"MATCH\",\"device\":\"d\",\"ort_build\":\"1\",\
                 \"tolerance\":\"t\",\"artifact\":\"a\",\"generated_at\":\"now\",\
                 \"shaders\":[\"ew_binary_add_f32\"],\"shader_digest\":\"{spirv}\",\
                 \"source_digest\":\"{source}\",\"toolchain\":\"shaderc 2023.8\",\
                 \"claimed_nodes\":1,\"dispatches_executed\":1}}"
            );
            let d = format!("{:016x}", fnv1a64(format!("{entry}\n").as_bytes()));
            parse_ledger(&format!(
                "{{\"__ledger__\":1,\"content_fnv1a64\":\"{d}\",\"entry_count\":1,\
                 \"generator\":\"test\"}}\n{entry}\n"
            ))
        };
        const OTHER: &str = "dead0000dead0000";

        // Row 1 — both agree.
        let same = build(&this_spirv, &this_source);
        assert!(
            same.state_for(&key).claimable(),
            "{:?}",
            same.state_for(&key)
        );
        assert!(matches!(
            same.get(&key).expect("entry").subject,
            SubjectVerdict::Identical
        ));

        // Row 2 — SPIR-V moved, source did not. THE LINUX CASE.
        let toolchain = build(OTHER, &this_source);
        let state = toolchain.state_for(&key);
        assert!(
            state.claimable(),
            "§8.9.19: a proof taken under a different compiler is a proof of THIS kernel; \
             declining it is what left Linux with no op-correctness number at all. state={state:?}"
        );
        assert_eq!(
            state.deltas(),
            vec![FrameDelta::Toolchain],
            "the claim is granted out of frame, so the δ has to be nameable — an undisclosed \
             out-of-frame claim is the trade §8.9.17 refused. state={state:?}"
        );

        // Row 3 — both moved. The kernel itself is a different kernel.
        let changed = build(OTHER, OTHER);
        assert!(
            !changed.state_for(&key).claimable(),
            "both digests moved and the form still claimed: {:?}",
            changed.state_for(&key)
        );
        assert!(matches!(
            changed.state_for(&key),
            ProofState::SubjectChanged {
                source_comparable: true,
                ..
            }
        ));

        // Row 4 — SPIR-V identical, source moved. Cosmetic, claimable, and NAMED.
        let cosmetic = build(&this_spirv, OTHER);
        assert!(cosmetic.state_for(&key).claimable());
        assert!(
            matches!(
                cosmetic.get(&key).expect("entry").subject,
                SubjectVerdict::SourceCosmetic { .. }
            ),
            "row 4 has to be distinguishable from row 1, or the source digest is being consulted \
             only when it agrees — which is not consulting it"
        );

        // The rows must not agree with each other. Without this the four arms above could all be
        // reading a predicate that returns the same thing and the test would still pass.
        let tokens = [
            same.state_for(&key).token(),
            toolchain.state_for(&key).token(),
            changed.state_for(&key).token(),
        ];
        assert_eq!(
            tokens
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            3,
            "three different frames produced the same verdict {tokens:?}; the digest pair is not \
             an instrument"
        );
    }

    /// **A build that compiled shaders must be able to name the compiler that compiled them.**
    ///
    /// Link's first fresh Linux `.so` embedded a full set of modules and reported its own
    /// toolchain as `UNKNOWN`. `SubjectVerdict::ToolchainDelta` is reachable without the string —
    /// it compares digests, not names — but the *disclosure* is the whole licence for claiming out
    /// of frame (§8.9.17), and `PROVEN-ELSEWHERE{toolchain}` with an unnameable toolchain
    /// discloses nothing. `--backfill-frame` already refuses to stamp `UNKNOWN`; nothing made the
    /// build itself say so, so an artifact whose every future proof is unframeable was
    /// indistinguishable from a good one.
    ///
    /// Skipped rather than failed for a shader-less artifact, because
    /// `ONNXRUNTIME_EP_VULKAN_ALLOW_MISSING_GLSLC=1` never invokes a compiler and has no
    /// toolchain to name — a genuine absence, not an unread one.
    #[test]
    fn a_build_that_embeds_shaders_can_name_its_shader_toolchain() {
        if !crate::engine::shaders::has_any() {
            return;
        }
        let tc = crate::engine::shaders::toolchain();
        assert!(
            !tc.is_empty() && tc != "UNKNOWN",
            "this artifact embeds {} SPIR-V module(s) but records toolchain={tc:?}. Every proof \
             taken against it would be stamped UNKNOWN, and an UNKNOWN frame cannot be told from \
             a second compiler: a ledger entry proven elsewhere reads as a changed kernel rather \
             than as a toolchain delta, which is the Linux 0-of-103 reading.",
            crate::engine::shaders::SHADER_MODULES.len(),
        );
    }

    /// **The second route back to the state §8.9.19 part 1 closed.**
    ///
    /// Entry survival moved subject-mismatched entries out of `entry_faults` and into `entries`,
    /// and `parse_ledger` cross-checks the header's `entry_count` against what it parsed. If that
    /// arithmetic still counted the two populations the old way, a ledger with one drifted entry
    /// would fault as a WHOLE FILE — re-creating, through a second path, exactly the global
    /// decline the ruling removed. Same shape as the defect that caused it: a second route to a
    /// state you thought you had closed.
    #[test]
    fn a_surviving_subject_mismatch_does_not_trip_the_declared_count() {
        const KEY: &str = "ai.onnx::Add/7+/f32,f32>f32/ew_binary_add_f32/static/n2";
        let key = ProofKey::validate(KEY).expect("valid key");
        let entry = format!(
            "{{\"key\":\"{KEY}\",\"verdict\":\"MATCH\",\"device\":\"d\",\"ort_build\":\"1\",\
             \"tolerance\":\"t\",\"artifact\":\"a\",\"generated_at\":\"now\",\
             \"shaders\":[\"ew_binary_add_f32\"],\"shader_digest\":\"0000000000000000\",\
             \"claimed_nodes\":1,\"dispatches_executed\":1}}"
        );
        let d = format!("{:016x}", fnv1a64(format!("{entry}\n").as_bytes()));
        let l = parse_ledger(&format!(
            "{{\"__ledger__\":1,\"content_fnv1a64\":\"{d}\",\"entry_count\":1,\
             \"generator\":\"test\"}}\n{entry}\n"
        ));
        assert!(
            l.faults.is_empty(),
            "a declared-count mismatch faulted the whole artifact over one drifted entry: {:?}",
            l.faults
        );
        assert_eq!(
            l.len() + l.entry_faults.len(),
            1,
            "the populations must still sum to 1"
        );
        assert!(!l.state_for(&key).claimable());
    }

    /// **§8.9.20 — the dispatch-time frame witness, seen in its firing state.**
    ///
    /// The residual §7.22 named: both digests are fixed at build time, and what runs is a
    /// pipeline. This plants one form four times — same key, same shaders, same SPIR-V digest,
    /// same source digest, differing only in the specialisation the proof was taken under — and
    /// requires the four to disagree. If they agree, the field is one no predicate reads, which
    /// is the defect class this project has now built three mechanisms to remove and would
    /// otherwise have re-created with the third.
    #[test]
    fn a_proof_replayed_under_another_specialisation_is_a_different_frame() {
        // Process-global statics: `pipeline_variants` is the observed collection.
        let _g = crate::allocator::ledger::test_lock();
        const KEY: &str = "ai.onnx::Add/7+/f32,f32>f32/ew_binary_add_f32/static/n2";
        const STEM: &str = "ew_binary_add_f32";
        let key = ProofKey::validate(KEY).expect("valid key");
        let this_spirv = shader_digest_for(&[STEM]).expect("a stem to digest");
        let this_source = source_digest_for(&[STEM]).expect("a stem to digest");
        let build = |spec: &str| {
            let entry = format!(
                "{{\"key\":\"{KEY}\",\"verdict\":\"MATCH\",\"device\":\"d\",\"ort_build\":\"1\",\
                 \"tolerance\":\"t\",\"artifact\":\"a\",\"generated_at\":\"now\",\
                 \"shaders\":[\"{STEM}\"],\"shader_digest\":\"{this_spirv}\",\
                 \"source_digest\":\"{this_source}\",\"toolchain\":\"shaderc 2023.8\",\
                 \"spec_digest\":\"{spec}\",\"claimed_nodes\":1,\"dispatches_executed\":1}}"
            );
            let d = format!("{:016x}", fnv1a64(format!("{entry}\n").as_bytes()));
            parse_ledger(&format!(
                "{{\"__ledger__\":1,\"content_fnv1a64\":\"{d}\",\"entry_count\":1,\
                 \"generator\":\"test\"}}\n{entry}\n"
            ))
        };

        // ARM 0 — nothing bound yet. This is the claim-time answer in a cold process, and it must
        // NOT read as agreement: nothing has been compared.
        crate::counters::reset();
        let unproven_yet = build("aaaaaaaaaaaaaaaa");
        let cold = unproven_yet.state_for(&key);
        assert_eq!(
            spec_witness_for(unproven_yet.get(&key).expect("entry")),
            SpecWitness::Unobserved,
            "a run that has bound no pipeline has observed no specialisation, and that is not the \
             same fact as having bound the recorded one"
        );
        assert!(
            cold.deltas().is_empty(),
            "an unobserved frame is not a delta: {cold:?}"
        );

        // Bind a pipeline. Everything below is compared against THIS.
        crate::counters::record_pipeline_variant(STEM, &[256, 0]);
        let bound_a = match crate::counters::specialisation_digest_for(&[STEM]) {
            crate::counters::SpecObservation::Full(d) => d,
            other => panic!("the run bound a pipeline and the observation is {other:?}"),
        };
        crate::counters::reset();
        crate::counters::record_pipeline_variant(STEM, &[256, 1]);
        let bound_b = match crate::counters::specialisation_digest_for(&[STEM]) {
            crate::counters::SpecObservation::Full(d) => d,
            other => panic!("the run bound a pipeline and the observation is {other:?}"),
        };
        assert_ne!(
            bound_a, bound_b,
            "two pipelines differing only in a specialisation constant hashed the same; the \
             witness is blind to exactly what it exists to see"
        );

        // ARM 1 — the entry was proven under what this run bound. In frame.
        let identical = build(&bound_b);
        let s_identical = identical.state_for(&key);
        assert_eq!(
            spec_witness_for(identical.get(&key).expect("entry")),
            SpecWitness::Identical
        );
        assert!(s_identical.claimable());
        assert!(s_identical.deltas().is_empty(), "{s_identical:?}");

        // ARM 2 — THE FIRING STATE. Same SPIR-V, same source closure, other pipeline.
        let delta = build(&bound_a);
        let s_delta = delta.state_for(&key);
        let e = delta.get(&key).expect("entry");
        assert_eq!(
            e.shader_digest, this_spirv,
            "the arm is only about specialisation if the subject is byte-identical"
        );
        assert_eq!(e.source_digest, this_source);
        assert!(matches!(e.subject, SubjectVerdict::Identical));
        assert!(
            s_delta.claimable(),
            "a specialisation delta is a frame delta, not a subject change; declining it would \
             decline a proof of this exact code: {s_delta:?}"
        );
        assert!(
            s_delta.deltas().contains(&FrameDelta::Specialisation),
            "both build-time digests agree and the pipelines differ — if this is not a delta the \
             witness is a field no predicate reads: {s_delta:?}"
        );

        // ARM 3 — the entry records nothing. Every entry shipped before §8.9.20.
        let unrecorded = build("");
        assert_eq!(
            spec_witness_for(unrecorded.get(&key).expect("entry")),
            SpecWitness::Unrecorded,
            "an entry with no spec_digest must not read as one that agrees"
        );
        let s_unrecorded = unrecorded.state_for(&key);
        assert!(
            s_unrecorded.claimable(),
            "unlike a missing source digest this has no repair from the tree, so refusing it \
             would decline 103 forms for a --backfill nobody can run: {s_unrecorded:?}"
        );
        assert!(!s_unrecorded.deltas().contains(&FrameDelta::Specialisation));

        // The arms must disagree. Without this every assertion above is satisfiable by a
        // predicate that returns one constant.
        let witnesses = [
            spec_witness_for(identical.get(&key).expect("entry")).token(),
            spec_witness_for(delta.get(&key).expect("entry")).token(),
            spec_witness_for(unrecorded.get(&key).expect("entry")).token(),
        ];
        assert_eq!(
            witnesses
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            3,
            "three specialisation frames produced one verdict {witnesses:?}"
        );

        // And the dispatch-time audit must report it — the predicate that fires in production,
        // not merely the one a test can call.
        crate::counters::reset();
        crate::counters::record_pipeline_variant(STEM, &[256, 1]);
        audit_dispatch_specialisation_of(&delta, STEM);
        let rows =
            crate::counters::specialisation_delta_forms().expect("the delta list must be readable");
        assert!(
            rows.iter().any(|r| r.starts_with(KEY)),
            "the audit saw a proof taken on another pipeline and said nothing: {rows:?}"
        );
        audit_dispatch_specialisation_of(&identical, STEM);
        assert_eq!(
            crate::counters::specialisation_delta_forms().map(|v| v.len()),
            Some(rows.len()),
            "the audit reported a delta for an entry proven under the pipeline this run bound; \
             it is not reading the digests"
        );
        audit_dispatch_specialisation_of(&unrecorded, STEM);
        assert!(
            crate::counters::specialisation_unrecorded_forms()
                .is_some_and(|v| v.iter().any(|r| r == KEY)),
            "an entry proven under an unrecorded specialisation was claimed and not disclosed"
        );
        crate::counters::reset();
    }

    /// A partial observation is not a delta.
    ///
    /// The false-positive this witness could most easily manufacture: an entry naming two stems,
    /// a run that has bound a pipeline for one of them, and a set-wide digest compared anyway.
    /// The delta would then say "this proof was taken on another pipeline" when what actually
    /// happened is that the run has not finished binding.
    #[test]
    fn a_half_bound_shader_set_is_not_a_specialisation_delta() {
        let _g = crate::allocator::ledger::test_lock();
        crate::counters::reset();
        assert_eq!(
            crate::counters::specialisation_digest_for(&["ew_binary_add_f32", "ew_unary_abs_f32"]),
            crate::counters::SpecObservation::Unobserved
        );
        crate::counters::record_pipeline_variant("ew_binary_add_f32", &[256, 0]);
        assert_eq!(
            crate::counters::specialisation_digest_for(&["ew_binary_add_f32", "ew_unary_abs_f32"]),
            crate::counters::SpecObservation::Partial { covered: 1, of: 2 },
            "half a shader set gives a different number, not a smaller one; comparing it would \
             invent a delta out of a run that has not finished binding"
        );
        crate::counters::record_pipeline_variant("ew_unary_abs_f32", &[256]);
        assert!(matches!(
            crate::counters::specialisation_digest_for(&["ew_binary_add_f32", "ew_unary_abs_f32"]),
            crate::counters::SpecObservation::Full(_)
        ));
        crate::counters::reset();
    }

    /// **The decline must name the subject it actually has** (Link, 2026-08-03).
    ///
    /// He measured 0 × `proof ledger fault` and 42 × `no proof ledger entry` in a log whose
    /// counters said every entry was in trouble, and the second string is the wrong sentence for
    /// that state in every clause. Both arms here, asserted against each other, because a text
    /// that says the same thing in both states is not reading anything.
    #[test]
    fn a_faulted_ledger_does_not_decline_a_form_by_saying_nothing_proved_it() {
        let key = ProofKey::parse("ai.onnx::Add/7+/f32,f32>f32/ew_binary_add_f32/static/n2");
        let absent = unproven_decline_detail(LedgerLookup::KeyAbsent, &key);
        let faulted = unproven_decline_detail(LedgerLookup::Faulted, &key);
        assert_ne!(absent, faulted, "one sentence for two findings");
        assert!(absent.contains("nothing has proven it correct"), "{absent}");
        assert!(
            !faulted.contains("nothing has proven it correct"),
            "on a faulted ledger this build cannot tell whether anything proved the form; \
             asserting that nothing did is a claim it has no evidence for: {faulted}"
        );
        assert!(
            faulted.contains("INSTRUMENT"),
            "R13: an instrument outage reported as a finding about the form: {faulted}"
        );
        assert!(
            faulted.contains("--check"),
            "the repair for a damaged artifact is not re-proving one form: {faulted}"
        );
    }

    /// **A form nothing can prove must not be declined by naming a proof run as the repair.**
    ///
    /// Tank, 2026-08-03. Phi-3.5's `unproven_declines` moved 3 → 5 and the two new forms are
    /// `Cast i64 -> i32` in both shape classes. The `[unproven]` text told the reader "the kernel
    /// exists; nothing has proven it correct … prove it with gen_proof_ledger.py" — and a proof
    /// run against that key reports `no unlockable keys`, because every `_i64` module declares
    /// `OpCapability Int64` and the engine enables no such feature. Same shape as the faulted-
    /// ledger defect above: a sentence asserting a fact about the kernel that nothing checked.
    ///
    /// Both arms, against each other, on keys that differ *only* in the variant component.
    #[test]
    fn a_form_with_no_creatable_module_is_not_declined_by_asking_for_a_proof() {
        let provable = ProofKey::parse("ai.onnx::Add/7+/f32,f32>f32/ew_binary_add_f32/static/n2");
        let unprovable = ProofKey::parse("ai.onnx::Cast/6+/i64>i32/ew_cast_i64_to_i32/static/n1");
        assert!(form_is_provable(&provable), "f32 add is loadable");
        assert!(
            !form_is_provable(&unprovable),
            "every _i64 variant declares Int64, which ENGINE_ENABLED_CAPABILITIES does not carry"
        );

        let ok = unproven_decline_detail(LedgerLookup::KeyAbsent, &provable);
        let no_kernel = unproven_decline_detail(LedgerLookup::KeyAbsent, &unprovable);
        assert!(ok.contains("gen_proof_ledger.py"), "{ok}");
        assert!(
            !no_kernel.contains("gen_proof_ledger.py"),
            "naming a tool that will answer `no unlockable keys` is advice that cannot be \
             followed: {no_kernel}"
        );
        assert!(
            !no_kernel.contains("The kernel exists"),
            "the module exists and no pipeline can be created from it; those are not the same \
             claim: {no_kernel}"
        );
        assert!(
            no_kernel.contains("ew_cast_i64_to_i32") && no_kernel.contains("capability"),
            "the decline has to name the module and the reason, or the reader is back to \
             guessing: {no_kernel}"
        );
    }

    /// **The mintability export answers exactly `form_is_provable`, for a caller with no device.**
    ///
    /// Three properties, because the export is the only thing `gen_proof_ledger.py --check` can
    /// ask and a report that drifts from the predicate would be a screen that is clean because it
    /// is looking somewhere else:
    ///
    /// 1. **Agreement.** Every line's `mintable=` equals `form_is_provable` on the same key.
    /// 2. **Non-vacuity.** The batch contains both answers. A report that can only say `yes` is
    ///    the 43-retired-keys-pass-cleanly state one layer down.
    /// 3. **Line discipline.** One line per input key, in input order, and a dtype signature's
    ///    commas do not split a key — which is why the wire separator is a newline.
    #[test]
    fn the_mintability_report_agrees_with_the_predicate_and_says_both_words() {
        let keys = [
            "ai.onnx::Add/7+/f32,f32>f32/ew_binary_add_f32/static/n2",
            "ai.onnx::Cast/6+/i64>i32/ew_cast_i64_to_i32/static/n1",
            "ai.onnx::Gather/1+/f16,i64>f16/metadata/runtime-extent/n2",
            "not-a-key",
        ];
        let text = form_mintability_report(&keys);
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), keys.len(), "one line per key: {text}");

        let mut yes = 0;
        let mut no = 0;
        for (line, key) in lines.iter().zip(keys.iter()) {
            let fields: Vec<&str> = line.split('\t').collect();
            assert_eq!(fields[0], *key, "key must round-trip verbatim: {line}");
            let mintable = match fields[1] {
                "mintable=yes" => true,
                "mintable=no" => false,
                other => panic!("unreadable verdict {other:?} in {line}"),
            };
            assert_eq!(
                mintable,
                form_is_provable(&ProofKey::parse(key)),
                "the export must not be a second opinion: {line}"
            );
            if mintable { yes += 1 } else { no += 1 }
        }
        assert!(
            yes > 0 && no > 0,
            "a report with one reachable answer screens nothing: {text}"
        );
    }

    /// The report is stable across calls and carries no state of its own.
    ///
    /// This is the property the spawn brief asked to preserve: `form_is_provable` reads only the
    /// baked SPIR-V and a checked-in capability list, so the answer cannot depend on whether a
    /// device has been created. A test in this process cannot create a `VkDevice`, so it asserts
    /// the reachable half — two calls, identical bytes, and no interior mutability behind them.
    #[test]
    fn the_mintability_report_is_stable_and_device_free() {
        let keys = ["ai.onnx::Cast/6+/i64>i32/ew_cast_i64_to_i32/static/n1"];
        let a = form_mintability_report(&keys);
        let b = form_mintability_report(&keys);
        assert_eq!(a, b);
        assert!(
            a.contains("mintable=no") && a.contains("declared=yes") && a.contains("loadable=no"),
            "the report has to say *why*, or the caller is back to guessing: {a}"
        );
    }

    /// An unparseable key is not evidence that a form is unprovable.
    #[test]
    fn a_key_whose_variant_cannot_be_read_gets_the_ordinary_message() {
        let malformed = ProofKey::parse("not-a-key");
        assert_eq!(malformed.variant_stem(), None);
        assert!(form_is_provable(&malformed));
    }

    /// **A `metadata` stem is a placeholder, not a refusal — but nothing live produces one now.**
    ///
    /// The first version of this predicate reported `Gather/…/metadata/static/n2` as unprovable
    /// because `variant_is_loadable("metadata")` is `false` for an unknown stem, so it under-claims
    /// on stems it does not recognise. That branch is kept, because a key it cannot recognise is
    /// still not evidence of anything.
    ///
    /// What changed on 2026-08-04 is which keys reach it. `Gather` — like `Conv`, `Gemm`,
    /// `GlobalAveragePool`, the norms and `GroupQueryAttention` — said `kernel!(None)` while
    /// dispatching a hand-written module, so its keys rendered `metadata` and took this branch.
    /// They now name their module and take the declared branch. Every row that still renders
    /// `metadata` is `Staged` with no dispatch path at all, which is what the component was
    /// documented to mean.
    #[test]
    fn a_composite_forms_metadata_stem_is_not_read_as_an_unloadable_module() {
        let composite =
            ProofKey::parse("ai.onnx::Gather/1+/f16,i64>f16/metadata/runtime-extent/n2");
        assert_eq!(composite.variant_stem(), Some("metadata"));
        assert!(
            !crate::ops::common::variants::variant_is_generated("metadata"),
            "no module is named `metadata`; that is the point"
        );
        assert!(
            !crate::ops::common::variants::variant_is_declared("metadata"),
            "no row names `metadata`; that is what makes it the unknown branch"
        );
        assert!(
            form_is_provable(&composite),
            "an unrecognised stem is not a capability finding — a classifier that calls it \
             unprovable is reporting its own blind spot as a finding about the form"
        );
    }

    /// **The positive control `form_is_provable` shipped without: a build with no SPIR-V.**
    ///
    /// Morpheus, §8.9.23: `Conv`'s four keys rendered their variant as `metadata` — documented to
    /// mean *"this row has no shader"* — while the same ledger entries recorded
    /// `"shaders":["conv_f32"]`. The cause was `kernel!(None)`, and one consequence was that this
    /// predicate short-circuited: `metadata` is not a module, so it took the under-claiming branch
    /// and answered *provable* for a row whose module it never consulted. In a build that produced
    /// no SPIR-V at all, `Conv` still read provable. That is the control this predicate was built
    /// to have, and it passed.
    ///
    /// The three inputs are enumerated rather than simulated: both readings go to `SHADER_MODULES`,
    /// which `build.rs` bakes in, so no test in this process can make the build shaderless. It can
    /// only assert what the predicate does with the booleans that build would hand it.
    #[test]
    fn a_declared_module_that_the_build_did_not_produce_is_not_provable() {
        // declared, generated, loadable -> provable
        assert!(
            form_provable_from(true, true, true),
            "a real, loadable module"
        );
        assert!(
            !form_provable_from(true, true, false),
            "declared and built, but declares a capability the engine does not enable"
        );
        assert!(
            !form_provable_from(true, false, false),
            "A SHADERLESS BUILD. The row names a module and the build did not produce it; no \
             proof run can measure it. This is the case that answered `true` before 2026-08-04, \
             via `metadata` on the unknown branch."
        );
        // Not declared: nothing is known, and under-claiming is deliberate.
        assert!(
            form_provable_from(false, false, false),
            "unknown stem, unknown answer"
        );
        assert!(form_provable_from(false, true, true));
    }

    /// The rows that carried the defect now key on their module, and the module is real.
    #[test]
    fn hand_written_rows_key_on_the_module_they_dispatch() {
        use crate::engine::DType;
        for (op, dtype, stem) in [
            ("Conv", DType::F32, "conv_f32"),
            ("Gemm", DType::F32, "gemm_f32"),
            ("GlobalAveragePool", DType::F32, "global_average_pool_f32"),
            ("Gather", DType::F16, "gather_f16"),
            ("GroupQueryAttention", DType::F16, "gqa_f16"),
            (
                "SkipSimplifiedLayerNormalization",
                DType::F32,
                "skip_simplified_layer_norm_f32",
            ),
        ] {
            let spec = all_specs()
                .find(|s| s.op_type == op)
                .unwrap_or_else(|| panic!("{op} must be registered"));
            assert_eq!(
                spec.kernel.stem(dtype),
                Some(stem),
                "`{op}` records `{stem}` in its ledger entries; its key must say so too"
            );
            assert!(
                crate::ops::common::variants::variant_is_declared(stem),
                "`{stem}` must be reachable from the registry, or the key takes the unknown branch"
            );
        }
    }

    /// The `@sel` and `#form` suffixes must not blind the stem lookup.
    ///
    /// This is how the defect would have come back the moment a second `Conv` form existed: the
    /// variant component is `conv_f32#grouped+padded`, and no module is named that.
    #[test]
    fn a_suffixed_variant_component_still_resolves_to_its_module() {
        let k =
            ProofKey::parse("ai.onnx::Conv/1+/f32,f32,f32>f32/conv_f32#grouped+padded/static/n3");
        assert_eq!(k.variant_component(), Some("conv_f32#grouped+padded"));
        assert_eq!(k.variant_stem(), Some("conv_f32"));
        let sel = ProofKey::parse("ai.onnx::IsInf/10+/f32>bool/ew_unary_isinf_f32@sel1/static/n1");
        assert_eq!(sel.variant_stem(), Some("ew_unary_isinf_f32"));
    }

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
            } else if spec.kernel.template.is_pair_keyed() {
                // Pair-keyed row: at least one compiled (source, destination) variant.
                spec.caps.iter().any(|src| {
                    spec.caps.iter().any(|dst| {
                        spec.kernel
                            .pair_stem(src, dst)
                            .is_some_and(|stem| shaders::find(stem).is_some())
                    })
                })
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
            // §8.9.11: every fixture needs a subject witness, or it fails for that reason rather
            // than the one under test. `ew_binary_add_f32` is a real stem, so the digest is the
            // one this build would compute.
            let digest_now =
                shader_digest_for(&["ew_binary_add_f32"]).expect("a non-empty stem list");
            let subject =
                format!(",\"shaders\":[\"ew_binary_add_f32\"],\"shader_digest\":\"{digest_now}\"");
            let entry = format!(
                "{{\"key\":\"{KEY}\",\"verdict\":\"MATCH\",\"device\":\"d\",\"ort_build\":\"1\",\
                 \"tolerance\":\"t\",\"artifact\":\"a\",\"generated_at\":\"now\"{subject}{extra}}}"
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
            enumerated
                .entry_faults
                .iter()
                .any(|f| f.contains("attribution")),
            "the fault must name attribution, not merely fail; got {:?}",
            enumerated.entry_faults
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

    /// **`LedgerEntry.device` is load-bearing — every state seen in its own polarity.**
    ///
    /// The defect Link found was not that the wrong answer was returned; it was that `.device` was
    /// never read, so the predicate had no state in which it could answer differently. A test that
    /// only asserts one answer on the baked ledger would reproduce exactly that: one input, one
    /// answer, no demonstrated sensitivity. So the same ledger is read from two different running
    /// devices and the answers must differ.
    ///
    /// The running device is planted through `note_session_device`, i.e. **the same channel the
    /// session uses**, because the predicate deliberately reads the device the run opened rather
    /// than the selector it was asked for — Trinity's `DEVICE=0` ran on `1=NVIDIA`.
    #[test]
    fn a_proof_is_a_property_of_a_form_on_a_device() {
        let _guard = crate::allocator::ledger::test_lock();
        const KEY: &str = "ai.onnx::Add/7+/f32,f32>f32/ew_binary_add_f32/static/n2";
        const ABSENT: &str = "ai.onnx::Sub/7+/f32,f32>f32/ew_binary_sub_f32/static/n2";
        const IRIS: &str = "Intel(R) Iris(R) Xe Graphics";
        const NV: &str = "NVIDIA GeForce RTX 4060 Laptop GPU";
        let key = ProofKey::validate(KEY).expect("valid key");
        let absent = ProofKey::validate(ABSENT).expect("valid key");

        let digest_now = shader_digest_for(&["ew_binary_add_f32"]).expect("a non-empty stem list");
        let ledger_proved_on = |device: &str| {
            let entry = format!(
                "{{\"key\":\"{KEY}\",\"verdict\":\"MATCH\",\"device\":\"{device}\",\
                 \"ort_build\":\"1\",\"tolerance\":\"t\",\"artifact\":\"a\",\
                 \"generated_at\":\"now\",\"claimed_nodes\":1,\"dispatches_executed\":1,\
                 \"shaders\":[\"ew_binary_add_f32\"],\"shader_digest\":\"{digest_now}\"}}"
            );
            let digest = format!("{:016x}", fnv1a64(format!("{entry}\n").as_bytes()));
            let header = format!(
                "{{\"__ledger__\":1,\"content_fnv1a64\":\"{digest}\",\"entry_count\":1,\
                 \"generator\":\"test\"}}"
            );
            parse_ledger(&format!("{header}\n{entry}\n"))
        };

        crate::allocator::tally::clear_session_devices();
        crate::allocator::tally::note_session_device(0, NV);
        assert_eq!(running_device_names(), vec![NV.to_string()]);

        let here = ledger_proved_on(NV);
        assert!(here.faults.is_empty(), "faults: {:?}", here.faults);
        assert_eq!(
            here.state_for(&key),
            ProofState::Proven,
            "an entry naming the device this run opened is PROVEN"
        );
        assert!(here.state_for(&key).claimable());

        let there = ledger_proved_on(IRIS);
        assert_eq!(
            there.state_for(&key),
            ProofState::ProvenElsewhere {
                deltas: vec![FrameDelta::Device],
                detail: format!("device: entry proved on `{IRIS}`, this run opened `{NV}`"),
            },
            "an entry proven on other hardware must say so — this is the fail-open Link found: \
             before this predicate existed, both readings answered the same"
        );
        assert!(
            there.state_for(&key).claimable(),
            "§8.9.19 turned this from a decline into a disclosed claim: declining it is what made \
             a run outside the proving frame produce no op-correctness number at all. It claims, \
             and the δ set says out loud what moved"
        );
        assert_eq!(
            there.state_for(&key).deltas(),
            &[FrameDelta::Device],
            "the δ set names exactly the component that differs, and no others"
        );

        // Never a silent fallback on a miss: an absent key may not reach a device-flavoured state.
        assert_eq!(
            there.state_for(&absent),
            ProofState::Unproven,
            "a key with no entry must be UNPROVEN"
        );

        // The run moves, the file does not: both directions on one pair of ledgers.
        crate::allocator::tally::clear_session_devices();
        crate::allocator::tally::note_session_device(0, IRIS);
        assert_eq!(
            there.state_for(&key),
            ProofState::Proven,
            "the Iris entry is a same-device proof when the run opened the Iris"
        );
        assert!(matches!(
            here.state_for(&key),
            ProofState::ProvenElsewhere { .. }
        ));

        // A selector ordinal names no hardware, so no comparison is possible in either direction.
        // This is the class **every one of the 97 baked entries** is in.
        let ordinal = ledger_proved_on("device0");
        assert!(
            matches!(
                ordinal.state_for(&key),
                ProofState::DeviceUnattributed { .. }
            ),
            "`device0` is a selector ordinal and must not read as an identity: {:?}",
            ordinal.state_for(&key)
        );
        assert!(
            ordinal.state_for(&key).claimable(),
            "the unattributable class stays claimable — declining it would take the EP to zero \
             claims over a frame question, not an evidence question"
        );

        // An entry with no device at all is unattributable too, and never PROVEN.
        let unlabelled = ledger_proved_on("");
        assert!(matches!(
            unlabelled.state_for(&key),
            ProofState::DeviceUnattributed { .. }
        ));

        // And with no device opened, even a named entry cannot be checked.
        crate::allocator::tally::clear_session_devices();
        assert!(
            matches!(here.state_for(&key), ProofState::DeviceUnattributed { .. }),
            "with no device opened there is nothing to compare against, and saying so is not the \
             same as saying the devices differ"
        );
    }

    /// A selector ordinal is recognised as an ordinal, and a device name is not.
    #[test]
    fn a_selector_ordinal_is_not_a_device_identity() {
        assert!(is_selector_ordinal("device0"));
        assert!(is_selector_ordinal("device1"));
        assert!(is_selector_ordinal("device12"));
        assert!(!is_selector_ordinal("device"));
        assert!(!is_selector_ordinal(""));
        assert!(!is_selector_ordinal("NVIDIA GeForce RTX 4060 Laptop GPU"));
        assert!(!is_selector_ordinal("Intel(R) Iris(R) Xe Graphics"));
        // The one that matters: a physical name that merely starts with the word.
        assert!(!is_selector_ordinal("device0 (NVIDIA)"));
    }

    /// The token vocabulary is six tokens, and they are distinct.
    #[test]
    fn proof_state_tokens_are_distinct_strings() {
        let tokens = [
            ProofState::Proven.token(),
            ProofState::ProvenElsewhere {
                deltas: vec![FrameDelta::Toolchain],
                detail: "d".to_string(),
            }
            .token(),
            ProofState::SourceCosmetic {
                recorded: "a".to_string(),
                current: "b".to_string(),
            }
            .token(),
            ProofState::SubjectChanged {
                recorded: "a".to_string(),
                current: "b".to_string(),
                source_comparable: true,
            }
            .token(),
            ProofState::DeviceUnattributed {
                entry_label: "device0".to_string(),
                reason: "r",
            }
            .token(),
            ProofState::Unproven.token(),
        ];
        let mut sorted = tokens.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 6, "two states share a spelling: {tokens:?}");
        assert!(!ProofState::Unproven.claimable());
        assert!(
            !ProofState::SubjectChanged {
                recorded: "a".to_string(),
                current: "b".to_string(),
                source_comparable: true,
            }
            .claimable(),
            "a subject that moved is a proof about other code, and no amount of frame agreement \
             rescues that"
        );
        assert!(
            ProofState::ProvenElsewhere {
                deltas: vec![FrameDelta::Toolchain],
                detail: "d".to_string(),
            }
            .claimable(),
            "§8.9.19: out of frame claims, and discloses δ"
        );
    }

    /// A `DIVERGENT` ledger line is **remembered as a demotion**, not merely rejected.    ///
    /// The falsifier for §8.9.7: without `Ledger::demoted`, a form the evidence measured and found
    /// wrong and a form nothing has ever measured both reach the session disclosure as "no proof",
    /// and RAI-008 names those as two states. The digest is deliberately absent from the header so
    /// this test measures the demotion path and not the digest path.
    #[test]
    fn a_divergent_ledger_line_is_remembered_as_a_demotion() {
        let key = "ai.onnx::Mul/7+/f16,f16>f16/ew_binary_mul_f16/static/n2";
        let src = format!(
            "{{\"__ledger__\":1,\"entry_count\":1,\"generator\":\"test\"}}\n\
             {{\"key\":\"{key}\",\"device\":\"d\",\"ort_build\":\"1.28.0\",\"tolerance\":\"t\",\
             \"artifact\":\"a\",\"verdict\":\"DIVERGENT\",\"generated_at\":\"now\",\
             \"claimed_nodes\":1,\"dispatches_executed\":1}}\n"
        );
        let l = parse_ledger(&src);
        let pk = ProofKey::parse(key);
        assert_eq!(
            l.demotion_for(&pk),
            Some("DIVERGENT"),
            "a non-MATCH verdict was dropped instead of remembered; demoted={:?}",
            l.demoted
        );
        assert!(
            !l.entry_faults.is_empty(),
            "a demotion must still fault the ENTRY — remembering a verdict must not grant a claim"
        );
        assert!(
            l.get(&pk).is_none(),
            "a demoted entry granted a claim; demotion must grant nothing"
        );
        // The absence case must not read the same way.
        assert_eq!(
            l.demotion_for(&ProofKey::parse("ai.onnx::Nothing/1+/f32>f32/k/static/n1")),
            None
        );
    }

    /// The shipped ledger is attributed, entry by entry, **and names the code it proves**. This is
    /// the assertion that would go red if anyone regenerated it with a tool that stopped recording
    /// provenance.
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
            assert!(
                !e.shaders.is_empty() && !e.shader_digest.is_empty(),
                "shipped entry {} names no subject: shaders={:?} digest={:?}. An entry that \
                 cannot say what code it proved cannot be invalidated when that code changes.",
                e.key.0,
                e.shaders,
                e.shader_digest
            );
        }
    }

    /// **§8.9.11 and §8.9.19 part 1 together, all polarities.** An entry proven against the
    /// shaders in this build claims; the *same* entry with one byte of its subject changed does
    /// not — and it **survives parsing** so that "proven about other code" stays distinguishable
    /// from "never proven".
    ///
    /// R10: the falsifier varies with the input. Two ledgers differing only in the digest produce
    /// two outcomes. Without the second arm this would be a check that passes because nothing ever
    /// disagrees with it, which is the shape that let the GQA entry survive two shader rewrites.
    ///
    /// The `is_none()` this test used to assert was the §8.9.19 defect in test form: it demanded
    /// that a subject mismatch be **indistinguishable from a key absence**, which is exactly what
    /// made a Linux run read as "97 forms were never proven".
    #[test]
    fn an_entry_whose_shader_changed_stops_proving_its_form() {
        const KEY: &str = "ai.onnx::Add/7+/f32,f32>f32/ew_binary_add_f32/static/n2";
        let key = ProofKey::validate(KEY).expect("valid key");
        let build = |digest: &str| {
            let entry = format!(
                "{{\"key\":\"{KEY}\",\"verdict\":\"MATCH\",\"device\":\"d\",\"ort_build\":\"1\",\
                 \"tolerance\":\"t\",\"artifact\":\"a\",\"generated_at\":\"now\",\
                 \"shaders\":[\"ew_binary_add_f32\"],\"shader_digest\":\"{digest}\",\
                 \"claimed_nodes\":1,\"dispatches_executed\":1}}"
            );
            let d = format!("{:016x}", fnv1a64(format!("{entry}\n").as_bytes()));
            parse_ledger(&format!(
                "{{\"__ledger__\":1,\"content_fnv1a64\":\"{d}\",\"entry_count\":1,\
                 \"generator\":\"test\"}}\n{entry}\n"
            ))
        };

        let current = shader_digest_for(&["ew_binary_add_f32"]).expect("a stem to digest");
        let fresh = build(&current);
        assert!(fresh.faults.is_empty(), "faults: {:?}", fresh.faults);
        assert!(
            fresh.get(&key).is_some(),
            "an entry proven against this build's shaders must claim"
        );
        assert!(fresh.state_for(&key).claimable());

        let stale = build("0000000000000000");
        assert!(
            stale.get(&key).is_some(),
            "§8.9.19 part 1: the entry must SURVIVE its mismatch. Deleting it is what made a \
             frame mismatch indistinguishable from a key absence — two different facts with two \
             different repairs, and only one of them actionable."
        );
        assert!(
            !stale.state_for(&key).claimable(),
            "surviving is not claiming: the predicate, not the parser, refuses. state={:?}",
            stale.state_for(&key)
        );
        assert!(
            matches!(
                stale.state_for(&key),
                ProofState::SubjectChanged {
                    source_comparable: false,
                    ..
                }
            ),
            "an entry with no source_digest whose SPIR-V differs cannot say whether the compiler \
             or the kernel moved, and the fail-safe reading is the kernel: {:?}",
            stale.state_for(&key)
        );
        assert_eq!(
            stale.demotion_count(),
            1,
            "a demotion the disclosure cannot count is a proof that silently stopped existing"
        );
        assert!(
            stale.faults.is_empty(),
            "a stale entry was recorded as a WHOLE-FILE fault, which makes every other entry \
             decline: {:?}",
            stale.faults
        );

        // arms_must_differ, stated rather than implied.
        assert_ne!(
            fresh.state_for(&key).claimable(),
            stale.state_for(&key).claimable(),
            "both arms reached the same outcome; the digest is not being read"
        );
    }

    /// **One stale entry must not take the other 96 down with it.**
    ///
    /// The blast radius, not the detection. `parse_ledger`'s own comment said "a stale entry
    /// demotes ITSELF and nothing else" while it pushed the message onto `Ledger::faults`, which
    /// `Ledger::get` consults for *every* key — so a single shader edit silently disarmed the
    /// entire artifact. Both polarities are in one ledger here on purpose: a two-file test would
    /// pass if the sound entry were declining for some unrelated reason of its own.
    #[test]
    fn one_stale_entry_does_not_fault_the_entries_beside_it() {
        const STALE_KEY: &str = "ai.onnx::Add/7+/f32,f32>f32/ew_binary_add_f32/static/n2";
        const SOUND_KEY: &str = "ai.onnx::Mul/7+/f32,f32>f32/ew_binary_mul_f32/static/n2";
        let stale_key = ProofKey::validate(STALE_KEY).expect("valid key");
        let sound_key = ProofKey::validate(SOUND_KEY).expect("valid key");
        let line = |key: &str, stem: &str, digest: &str| {
            format!(
                "{{\"key\":\"{key}\",\"verdict\":\"MATCH\",\"device\":\"d\",\"ort_build\":\"1\",\
                 \"tolerance\":\"t\",\"artifact\":\"a\",\"generated_at\":\"now\",\
                 \"shaders\":[\"{stem}\"],\"shader_digest\":\"{digest}\",\
                 \"claimed_nodes\":1,\"dispatches_executed\":1}}"
            )
        };
        let sound_digest = shader_digest_for(&["ew_binary_mul_f32"]).expect("a stem to digest");
        let a = line(STALE_KEY, "ew_binary_add_f32", "0000000000000000");
        let b = line(SOUND_KEY, "ew_binary_mul_f32", &sound_digest);
        let body = format!("{a}\n{b}\n");
        let d = format!("{:016x}", fnv1a64(body.as_bytes()));
        let l = parse_ledger(&format!(
            "{{\"__ledger__\":1,\"content_fnv1a64\":\"{d}\",\"entry_count\":2,\
             \"generator\":\"test\"}}\n{body}"
        ));

        // Non-vacuity: the stale entry must actually be detected, or "the other one still
        // claims" is the trivial statement that nothing went wrong.
        assert!(
            !l.state_for(&stale_key).claimable(),
            "ERROR(instrument): the stale entry was not detected, so the blast-radius assertion \
             below is vacuous: state={:?}",
            l.state_for(&stale_key)
        );
        assert!(
            l.state_for(&sound_key).claimable(),
            "a sound entry stopped proving because a DIFFERENT entry was stale — one shader edit \
             disarming the whole artifact is the 2026-08-02 defect: faults={:?} entry_faults={:?}",
            l.faults,
            l.entry_faults
        );
        assert!(
            l.faults.is_empty(),
            "an entry-level problem was recorded as a whole-file fault: {:?}",
            l.faults
        );
        assert_eq!(
            l.subject_changed_entries().count(),
            1,
            "exactly one entry's subject moved, and it must be locatable rather than merely gone"
        );
    }

    /// An entry that carries no subject witness at all is refused, not tolerated.
    ///
    /// This is the shape every entry in the artifact had before 2026-08-02, and admitting it
    /// "for compatibility" would leave the hole open for exactly the entries that predate the
    /// fix — which are the ones that have had the longest to drift.
    #[test]
    fn an_entry_with_no_subject_witness_proves_nothing() {
        const KEY: &str = "ai.onnx::Add/7+/f32,f32>f32/ew_binary_add_f32/static/n2";
        let entry = format!(
            "{{\"key\":\"{KEY}\",\"verdict\":\"MATCH\",\"device\":\"d\",\"ort_build\":\"1\",\
             \"tolerance\":\"t\",\"artifact\":\"a\",\"generated_at\":\"now\",\
             \"claimed_nodes\":1,\"dispatches_executed\":1}}"
        );
        let d = format!("{:016x}", fnv1a64(format!("{entry}\n").as_bytes()));
        let l = parse_ledger(&format!(
            "{{\"__ledger__\":1,\"content_fnv1a64\":\"{d}\",\"entry_count\":1,\
             \"generator\":\"test\"}}\n{entry}\n"
        ));
        let key = ProofKey::validate(KEY).expect("valid key");
        assert!(
            l.get(&key).is_none(),
            "an entry with no subject granted a claim"
        );
        assert_eq!(l.demotion_for(&key), Some("NO-SUBJECT-WITNESS"));
    }

    /// The digest moves with its input, and only with the part of the input it claims to cover.
    #[test]
    fn shader_digest_covers_the_named_modules_and_their_order_does_not_matter() {
        let a = shader_digest_for(&["ew_binary_add_f32"]).expect("digest");
        let ab = shader_digest_for(&["ew_binary_add_f32", "ew_binary_mul_f32"]).expect("digest");
        let ba = shader_digest_for(&["ew_binary_mul_f32", "ew_binary_add_f32"]).expect("digest");
        assert_eq!(ab, ba, "the digest must not depend on dispatch order");
        assert_ne!(a, ab, "adding a dispatched module must move the digest");
        assert_eq!(
            shader_digest_for(&[]),
            None,
            "R12: `nothing was dispatched` is not `the digest of nothing`"
        );
        assert_ne!(
            shader_digest_for(&["a-stem-that-does-not-exist"]),
            shader_digest_for(&["another-stem-that-does-not-exist"]),
            "two absent modules must not collapse to one digest"
        );
    }

    /// **The three-token miss path** (R13, discharge condition (d)). A miss is three findings, not
    /// one `false`, and they call for three different actions: regenerate this form, fix the
    /// ledger file, and nothing at all.
    #[test]
    fn a_ledger_miss_reports_which_of_three_things_happened() {
        let proven = ProofKey::validate("ai.onnx::Add/7+/f32,f32>f32/ew_binary_add_f32/static/n2")
            .expect("valid");
        let absent = ProofKey::validate("ai.onnx::Add/7+/f64,f64>f64/ew_binary_add_f64/static/n2")
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
