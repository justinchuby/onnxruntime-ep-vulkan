//! The op registry and `NodeView` — where the ORT graph ABI is translated into safe Rust.
//!
//! Two responsibilities, and no third:
//!
//! 1. [`NodeView`] — a *read-only* borrow of an `OrtNode`. This is one of exactly two places in
//!    the crate where ORT ABI types are touched outside `ep.rs`/`factory.rs` (`DESIGN.md` §4.2
//!    calls it "the exception that proves the rule"), and it exists so that claim predicates can
//!    ask questions about a node without ever seeing an `OrtNode`.
//!
//! 2. The registry itself — a single table mapping a domain-qualified op name to a
//!    [`ClaimPredicate`]. `ep.rs` asks exactly one question per node, [`claim_decision`], and has
//!    **no per-op logic**. That invariant is what makes "claimed" and "translatable" impossible to
//!    desynchronize; it is inherited directly from the `onnxruntime-mlx` reference.
//!
//! # M0 state
//!
//! [`REGISTRY`] is empty. Every node is declined, with a reason, and the whole graph runs on ORT's
//! CPU EP — which is always correct. `GetCapability` therefore has the right *shape* (it walks the
//! graph, asks the registry, aggregates decline reasons, and would cluster and fuse if anything
//! were claimed) with no claiming behaviour yet.
//!
//! Mouse adds `Add` here in M0 by writing one handler in `src/ops/elementwise.rs` and adding one
//! row to [`REGISTRY`]. No boundary-layer edit is required, which is the whole point of the shape.

use std::borrow::Cow;
use std::ffi::CStr;

use crate::engine::NodeDesc;
use crate::sys::{self, ort};

// -------------------------------------------------------------------------------------------
// NodeView — a read-only borrow of an OrtNode
// -------------------------------------------------------------------------------------------

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

    /// The raw node pointer, for `ep.rs` to hand back to `EpGraphSupportInfo_AddNodesToFuse`.
    ///
    /// `pub(crate)` on purpose: nothing outside the boundary layer has any use for it.
    pub(crate) fn raw(&self) -> *const ort::OrtNode {
        self.node
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

// -------------------------------------------------------------------------------------------
// The registry
// -------------------------------------------------------------------------------------------

/// Why a node was not claimed. Always a sentence a user can act on; it is printed verbatim by
/// `ONNXRUNTIME_EP_VULKAN_CLAIM_DEBUG=1` and is the single most valuable diagnostic the reference
/// EP has (`DESIGN.md` §5.4).
pub type DeclineReason = Cow<'static, str>;

/// A claim predicate: given a node, either claim it or say why not.
///
/// Registered handlers live in `src/ops/` (Mouse). The predicate must be *exactly* as strict as
/// the translate handler — if it claims a node the handler cannot translate, `Compile` fails and
/// the whole subgraph falls back, which is strictly worse than never claiming it.
pub type ClaimPredicate = fn(&NodeView<'_>) -> Result<(), DeclineReason>;

/// Every op this EP can claim, keyed by domain-qualified name.
///
/// **Empty in M0 by design.** `DESIGN.md` §1.3: conservative claiming is a hard requirement, and
/// CPU fallback is always correct. Mouse grows this table one row per op.
pub static REGISTRY: &[(&str, ClaimPredicate)] = &[
    // ("Add", crate::ops::elementwise::claim_add),
];

/// Look up a node's claim predicate.
fn lookup(qualified: &str) -> Option<ClaimPredicate> {
    REGISTRY
        .iter()
        .find(|(name, _)| *name == qualified)
        .map(|(_, p)| *p)
}

/// The one question `ep.rs` asks per node.
///
/// `Ok(())` means claim it. `Err(reason)` means leave it to the CPU EP and report `reason` under
/// claim-debug. There is no third answer and no per-op logic anywhere above this function.
pub fn claim_decision(view: &NodeView<'_>) -> Result<(), DeclineReason> {
    let qualified = view.qualified_name();
    match lookup(&qualified) {
        Some(predicate) => predicate(view),
        None => Err(Cow::Owned(format!(
            "no Vulkan handler is registered for `{qualified}` (opset {}); the op registry is \
             empty in M0, so every node is declined and runs on the CPU EP",
            view.since_version()
        ))),
    }
}

/// Convenience wrapper for the boolean question.
pub fn claimable(view: &NodeView<'_>) -> bool {
    claim_decision(view).is_ok()
}

/// The same decision, taken against an already-extracted [`NodeDesc`].
///
/// `Compile` uses this to re-check every node it is about to translate: a node that was claimed in
/// `GetCapability` but is not claimable here is an internal invariant violation, not a user error
/// (`DESIGN.md` §5.5 step 2). Because M0's registry is empty, this is currently only reachable
/// through the "unknown op" arm.
pub fn is_registered(desc: &NodeDesc) -> bool {
    lookup(&desc.qualified_name()).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_is_empty_in_m0() {
        assert!(
            REGISTRY.is_empty(),
            "M0 claims nothing; when Mouse adds an op, update this test and the M0 note in the \
             module docs together"
        );
    }

    #[test]
    fn registry_keys_are_unique() {
        let mut keys: Vec<&str> = REGISTRY.iter().map(|(k, _)| *k).collect();
        keys.sort_unstable();
        let before = keys.len();
        keys.dedup();
        assert_eq!(before, keys.len(), "duplicate op key in REGISTRY");
    }

    #[test]
    fn unknown_ops_are_declined_with_a_reason() {
        assert!(lookup("NoSuchOp").is_none());
    }

    #[test]
    fn unregistered_node_desc_is_not_registered() {
        let desc = NodeDesc {
            op_type: "Add".into(),
            ..Default::default()
        };
        assert!(!is_registered(&desc));
    }
}
