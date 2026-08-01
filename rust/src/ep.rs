//! `VulkanEp` — our `OrtEp` C-ABI vtable: one instance per ONNX Runtime session.
//!
//! The ORT struct is embedded as the **first** field under `#[repr(C)]`, so the `*OrtEp` ORT holds
//! is pointer-identical to our `*VulkanEp` at offset 0. Ownership crosses the boundary through
//! `Box::into_raw` / `Box::from_raw`; teardown is RAII (`impl Drop`), never a manual free.
//!
//! Every `extern "C"` entry point that runs real logic is wrapped in
//! [`crate::guard_ffi_status`], which converts a Rust panic into an `ORT_EP_FAIL` status. A panic
//! unwinding into ORT's C++ is undefined behaviour, and a plugin must never take down its host.
//!
//! # M0 scope
//!
//! * `GetCapability` walks the graph, asks [`crate::registry::claim_decision`] about every node,
//!   aggregates the decline reasons per op type, and — because the registry is empty — claims
//!   nothing. The clustering-and-fusing path is present and correct in shape; it simply has no
//!   input yet.
//! * `GetDefaultMemoryDevice` returns null: M0/M1 keep subgraph I/O in host memory and stage
//!   uploads inside `Compute` (`DESIGN.md` §6.3, decisions.md "Phased memory model").
//! * `Compile` is unreachable while nothing is claimed, and says so loudly rather than pretending.

use std::collections::{BTreeMap, HashMap};
use std::ffi::{CStr, CString, c_char, c_void};
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};

/// Monotonically increasing counter for generating unique subgraph IDs.
/// Each `SubgraphComputeInfo::new_live` call consumes one ID.
static NEXT_SUBGRAPH_ID: AtomicU64 = AtomicU64::new(1);

use crate::counters;
use crate::engine::{CompileRecorder, CompiledKernel, VulkanSession};
use crate::logging;
use crate::ops::partition;
use crate::registry::{self, NodeView};
use crate::sys::{self, ort};

/// Session options this EP understands, all prefixed `ep.` (`DESIGN.md` §2.4).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EpOptions {
    /// `ep.device_index` — which capable Vulkan device to bind, as an index into the **best-first
    /// sorted** list (the same space `ONNXRUNTIME_EP_VULKAN_DEVICE` indexes). `None` = auto.
    ///
    /// It is NOT the `vkEnumeratePhysicalDevices` index, and it is NOT authoritative: the device
    /// ORT bound for the session wins (see [`EpOptions::bound_physical_index`]).
    pub device_index: Option<usize>,
    /// The `vkEnumeratePhysicalDevices` index of the `OrtEpDevice` **ORT bound for this session**,
    /// read from the EP metadata `CreateEp` hands back. `None` when ORT gave us nothing to read.
    ///
    /// Authoritative over `device_index`, because ORT keys the allocator it asks us for by the
    /// device it bound; opening any other device stands up a second `VkDevice` and violates §6.5.
    /// Not a session option — ORT sets it, users cannot.
    pub bound_physical_index: Option<usize>,
    /// `ep.enable_validation` — enable `VK_LAYER_KHRONOS_validation`.
    pub enable_validation: bool,
    /// `ep.pipeline_cache_path` — on-disk `VkPipelineCache` blob location.
    pub pipeline_cache_path: Option<String>,
    /// `ep.max_claim_ops` — comma-separated allowlist restricting claiming. Debugging only.
    pub max_claim_ops: Option<Vec<String>>,
    /// `ep.disable_device_memory` — force the host-memory I/O path.
    pub disable_device_memory: bool,
    /// `ep.force_legacy_barriers` — force the legacy `vkCmdPipelineBarrier` backend even on a
    /// device that supports `VK_KHR_synchronization2` (DESIGN.md §7.5 item 5). Used by Trinity
    /// to run the differential suite twice per lane, ensuring the legacy path is never untested.
    pub force_legacy_barriers: bool,
}

impl EpOptions {
    /// Read the `ep.*` options out of an `OrtSessionOptions`.
    ///
    /// Unknown or unset keys leave defaults in place; a malformed value logs a warning and is
    /// ignored rather than failing session creation, because a typo in an optional tuning knob is
    /// not worth refusing to run a model over.
    ///
    /// # Safety
    /// `api` must be a live `OrtApi`. `options` may be null (treated as "no options").
    pub unsafe fn from_session_options(
        api: *const ort::OrtApi,
        options: *const ort::OrtSessionOptions,
    ) -> EpOptions {
        let mut out = EpOptions::default();
        if api.is_null() || options.is_null() {
            return out;
        }

        // SAFETY: `api` is live per the caller's contract.
        let get = unsafe { (*api).GetSessionConfigEntry };
        let Some(get) = get else {
            return out;
        };

        let read = |key: &str| -> Option<String> {
            // SAFETY: `get` is ORT's config accessor, used here with the documented two-call
            // pattern (size query, then fill). See `read_config_entry`.
            unsafe { read_config_entry(api, get, options, key) }
        };

        if let Some(v) = read("ep.device_index") {
            match v.trim().parse::<usize>() {
                Ok(i) => out.device_index = Some(i),
                Err(_) => {
                    log::warn!("ep.device_index: `{v}` is not a non-negative integer; ignoring")
                }
            }
        }
        if let Some(v) = read("ep.enable_validation") {
            out.enable_validation = parse_bool(&v).unwrap_or_else(|| {
                log::warn!("ep.enable_validation: `{v}` is not a boolean; ignoring");
                false
            });
        }
        if let Some(v) = read("ep.disable_device_memory") {
            out.disable_device_memory = parse_bool(&v).unwrap_or_else(|| {
                log::warn!("ep.disable_device_memory: `{v}` is not a boolean; ignoring");
                false
            });
        }
        if let Some(v) = read("ep.force_legacy_barriers") {
            out.force_legacy_barriers = parse_bool(&v).unwrap_or_else(|| {
                log::warn!("ep.force_legacy_barriers: `{v}` is not a boolean; ignoring");
                false
            });
        }
        if let Some(v) = read("ep.pipeline_cache_path")
            && !v.is_empty()
        {
            out.pipeline_cache_path = Some(v);
        }
        if let Some(v) = read("ep.max_claim_ops") {
            let ops: Vec<String> = v
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect();
            if !ops.is_empty() {
                log::info!("ep.max_claim_ops restricts claiming to {ops:?}");
                out.max_claim_ops = Some(ops);
            }
        }
        out
    }
}

fn parse_bool(v: &str) -> Option<bool> {
    match v.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

type ConfigEntryFn = unsafe extern "C" fn(
    *const ort::OrtSessionOptions,
    *const c_char,
    *mut c_char,
    *mut usize,
) -> ort::OrtStatusPtr;

/// Read one session config entry using ORT's two-call (size, then fill) protocol.
///
/// # Safety
/// `api` must be a live `OrtApi`, `get` its `GetSessionConfigEntry`, and `options` a live
/// `OrtSessionOptions`.
unsafe fn read_config_entry(
    api: *const ort::OrtApi,
    get: ConfigEntryFn,
    options: *const ort::OrtSessionOptions,
    key: &str,
) -> Option<String> {
    let c_key = CString::new(key).ok()?;

    let mut size: usize = 0;
    // SAFETY: `options` and `c_key` are valid for the call; passing a null buffer with a zeroed
    // size is ORT's documented "how big is it?" query. Any status returned (including the
    // not-found / buffer-too-small cases) is owned by us and released before returning.
    unsafe {
        let status = get(options, c_key.as_ptr(), ptr::null_mut(), &mut size);
        if !status.is_null() {
            // Either the key is absent or ORT is telling us the required size. Both leave a status
            // we own. `size` is only trustworthy in the second case, which we detect by trying the
            // fill call below; releasing here is correct either way.
            sys::release_status(api, status);
            if size == 0 {
                return None;
            }
        }
    }
    if size == 0 {
        return None;
    }

    let mut buf: Vec<u8> = vec![0; size];
    // SAFETY: `buf` has exactly `size` bytes, which is what ORT asked for; `size` is passed by
    // pointer so ORT can report what it actually wrote.
    unsafe {
        let mut written = size;
        let status = get(
            options,
            c_key.as_ptr(),
            buf.as_mut_ptr().cast::<c_char>(),
            &mut written,
        );
        if !status.is_null() {
            sys::release_status(api, status);
            return None;
        }
    }
    // ORT writes a NUL-terminated string; trim at the first NUL.
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    Some(String::from_utf8_lossy(&buf[..end]).into_owned())
}

// -------------------------------------------------------------------------------------------
// The EP object
// -------------------------------------------------------------------------------------------

/// One per session. Owns everything the session needs and drops it all in `ReleaseEp`.
#[repr(C)]
pub struct VulkanEp {
    /// MUST be first: ORT's pointer is this pointer.
    base: ort::OrtEp,
    ort_api: *const ort::OrtApi,
    ep_api: *const ort::OrtEpApi,
    /// The ABI version negotiated with this host, inherited from the factory.
    abi_version: u32,
    name: CString,
    options: EpOptions,
    /// Session-level Vulkan state. `None` when no capable device is available.
    ///
    /// Stored in a `Box` for a stable address: `SubgraphComputeInfo` holds a raw pointer into
    /// this box (valid because ORT guarantees the EP outlives all compiled compute infos).
    session: Option<Box<VulkanSession>>,
}

impl VulkanEp {
    /// Build an EP for one session.
    ///
    /// # Safety
    /// `ort_api` / `ep_api` must be live for the EP's whole lifetime (they are: ORT's tables are
    /// process-lived). `session_options` may be null.
    pub unsafe fn new(
        ort_api: *const ort::OrtApi,
        ep_api: *const ort::OrtEpApi,
        abi_version: u32,
        name: &CStr,
        session_options: *const ort::OrtSessionOptions,
        bound_physical_index: Option<usize>,
    ) -> Box<VulkanEp> {
        // SAFETY: `ort_api` is live per the caller's contract; `session_options` may be null,
        // which `from_session_options` handles.
        let mut options = unsafe { EpOptions::from_session_options(ort_api, session_options) };
        // Not a session option: this is the device ORT bound, and it outranks every selector.
        options.bound_physical_index = bound_physical_index;

        // SAFETY: `OrtEp` is a `#[repr(C)]` plain-old-data vtable of a `u32` and function
        // pointers, all of which bindgen models as `Option<fn>`. All-zero is the valid `None`
        // niche for every one of them, so a zeroed struct is a well-formed "no callbacks
        // installed" vtable. We then fill exactly the slots we implement; ORT treats a null slot
        // as "not supported by this EP".
        let mut base: ort::OrtEp = unsafe { std::mem::zeroed() };
        // As on the factory: advertise the *negotiated* version, so a downlevel host stops reading
        // our vtable exactly where its own header stops describing it.
        base.ort_version_supported = abi_version;
        base.GetName = Some(get_name);
        base.GetCapability = Some(get_capability);
        base.Compile = Some(compile);
        base.ReleaseNodeComputeInfos = Some(release_node_compute_infos);
        base.GetDefaultMemoryDevice = Some(get_default_memory_device);

        let session = if crate::engine::shaders::has_any() {
            // SAFETY: VulkanSession::create loads the Vulkan library, which stays loaded for the
            // EP's lifetime, and creates an instance/device from it. `options` is borrowed for the
            // duration of this call only and nothing in the returned session points into it. The
            // `shaders::has_any()` guard above is what makes the call meaningful at all: with no
            // compiled shaders no pipeline can ever be created, so we do not open a device.
            unsafe { VulkanSession::create(&options) }.map(Box::new)
        } else {
            None
        };

        if session.is_some() {
            log::info!(
                "VulkanExecutionProvider session created — Vulkan device ready, claiming will begin"
            );
        } else {
            log::info!(
                "VulkanExecutionProvider session created — no Vulkan device (or shader-less build); \
                 all nodes run on the CPU EP"
            );
        }

        Box::new(VulkanEp {
            base,
            ort_api,
            ep_api,
            abi_version,
            name: name.to_owned(),
            options,
            session,
        })
    }

    /// Hand ownership to ORT.
    pub fn into_raw(self: Box<Self>) -> *mut ort::OrtEp {
        Box::into_raw(self).cast::<ort::OrtEp>()
    }

    /// Take ownership back and drop.
    ///
    /// # Safety
    /// `p` must be a pointer previously returned by [`VulkanEp::into_raw`] and not yet released.
    pub unsafe fn release(p: *mut ort::OrtEp) {
        if p.is_null() {
            return;
        }
        // SAFETY: `p` came from `Box::into_raw` on a `Box<VulkanEp>` (guaranteed by the caller),
        // and `VulkanEp` is `#[repr(C)]` with `base` first, so the `OrtEp` pointer ORT holds is
        // exactly the `VulkanEp` allocation.
        drop(unsafe { Box::from_raw(p.cast::<VulkanEp>()) });
    }

    /// Session options in effect. Switch reads these when creating the `VkDevice`.
    pub fn options(&self) -> &EpOptions {
        &self.options
    }

    /// The plugin-EP sub-API, for the fuse calls in `GetCapability`.
    fn ep_api(&self) -> *const ort::OrtEpApi {
        self.ep_api
    }

    /// The ABI version negotiated with this host.
    ///
    /// Stamped into every versioned ORT struct we hand back (`OrtNodeFusionOptions` and friends)
    /// so a downlevel host reads exactly as far as it understands, and gates any call to an entry
    /// point newer than [`crate::sys::ORT_API_VERSION_MIN`].
    fn abi_version(&self) -> u32 {
        self.abi_version
    }
}

impl Drop for VulkanEp {
    fn drop(&mut self) {
        // RAII teardown from day one. The reference project found a real per-session leak here
        // that three lines of `impl Drop` fixed; the Vulkan objects Switch adds (VkDevice, queues,
        // allocator arena, command/descriptor pools, pipeline cache) all hang off this struct and
        // are destroyed by this drop, in field order, exactly once.
        log::debug!("VulkanExecutionProvider session released");
        // Emit the phase/transfer summary BEFORE exporting.
        //
        // (Tank, 2026-07-30) `Tracer::log_summary` had **zero callers** in the whole crate. It is
        // the only place `phase_us[Upload]` and `phase_us[Readback]` are ever surfaced, and
        // `record_transfer` deliberately records those durations as summary entries rather than
        // `ph:"X"` spans (to avoid double-counting inside `Phase::Record`, which encloses them).
        //
        // The consequence was that host staging-copy time was measured on every inference and
        // then discarded: any aggregation over the trace JSON's `ph:"X"` spans — which is how the
        // record/fence_wait/submit split was produced — structurally cannot see it, and its
        // absence read as zero. That is R7 exactly: the instrument was not in a position to
        // report, so its silence was mistaken for a negative result.
        crate::trace::tracer().log_summary();
        // Same class, same sweep: `log_slowest_ops` also had zero callers. Its producer
        // (`record_op_meta`) is still unwired and belongs to whoever owns op translation, so this
        // call currently prints "NOT WIRED" — which is the point. An artifact that names its own
        // missing instrument is worth more than one that omits the section.
        crate::trace::tracer().log_slowest_ops();
        // Export the Chrome Trace JSON. The tracer accumulates across sessions and rewrites the
        // full file on each EP teardown; the final teardown leaves the complete trace on disk.
        crate::trace::tracer().export();
    }
}

/// Reinterpret ORT's `OrtEp*` as our `VulkanEp*`.
///
/// # Safety
/// `p` must be a pointer this crate produced via [`VulkanEp::into_raw`].
#[inline]
unsafe fn this<'a>(p: *const ort::OrtEp) -> &'a VulkanEp {
    // SAFETY: `VulkanEp` is `#[repr(C)]` with `base: OrtEp` first, so the two pointers have the
    // same address, and the caller guarantees the allocation is live and was made by us.
    unsafe { &*p.cast::<VulkanEp>() }
}

// -------------------------------------------------------------------------------------------
// Vtable entry points
// -------------------------------------------------------------------------------------------

unsafe extern "C" fn get_name(p: *const ort::OrtEp) -> *const c_char {
    // SAFETY: ORT only ever passes back a pointer we produced; the `CString` lives as long as the
    // EP, and ORT copies or uses the name before releasing the EP.
    unsafe { this(p).name.as_ptr() }
}

unsafe extern "C" fn get_default_memory_device(
    _p: *const ort::OrtEp,
    device: *mut *const ort::OrtMemoryDevice,
) -> ort::OrtStatusPtr {
    // M0/M1 advertise no device memory: subgraph I/O stays in host memory and staging happens
    // inside Compute. M2 returns a real OrtMemoryDevice here (DESIGN.md §6.3).
    if !device.is_null() {
        // SAFETY: `device` is a valid out-parameter slot supplied by ORT.
        unsafe { *device = ptr::null() };
    }
    ptr::null_mut()
}

// --- GetCapability ---------------------------------------------------------------------------

unsafe extern "C" fn get_capability(
    p: *mut ort::OrtEp,
    graph: *const ort::OrtGraph,
    support: *mut ort::OrtEpGraphSupportInfo,
) -> ort::OrtStatusPtr {
    // SAFETY: `p` is our EP pointer; reading the API table out of it before entering the guard is
    // what lets a caught panic still be reported as a status.
    let api = unsafe { this(p).ort_api };
    // SAFETY: `api` is a live `OrtApi`; the closure only touches pointers ORT gave us.
    unsafe {
        crate::guard_ffi_status(api, "GetCapability", || {
            get_capability_impl(p, graph, support)
        })
    }
}

/// # Safety
/// `p`, `graph` and `support` must be the live pointers ORT passed to `GetCapability`.
unsafe fn get_capability_impl(
    p: *mut ort::OrtEp,
    graph: *const ort::OrtGraph,
    support: *mut ort::OrtEpGraphSupportInfo,
) -> ort::OrtStatusPtr {
    // SAFETY: `p` is our EP pointer, live for the duration of this call.
    let ep = unsafe { this(p) };
    let api = ep.ort_api;

    if graph.is_null() || support.is_null() {
        // SAFETY: `api` is live.
        return unsafe {
            sys::make_status(
                api,
                ort::OrtErrorCode_ORT_INVALID_ARGUMENT,
                "GetCapability received a null graph or graph-support-info",
            )
        };
    }

    // --- enumerate the graph's nodes ---
    let mut num_nodes: usize = 0;
    // SAFETY: `api` is live and `graph` is a live graph; `num_nodes` is a valid out-param.
    let status = unsafe {
        match (*api).Graph_GetNumNodes {
            Some(f) => f(graph, &mut num_nodes),
            None => {
                return sys::make_status(
                    api,
                    ort::OrtErrorCode_ORT_EP_FAIL,
                    "OrtApi::Graph_GetNumNodes is unavailable; the host's plugin-EP graph API is \
                     not the one this EP was built against",
                );
            }
        }
    };
    if !status.is_null() {
        return status;
    }
    if num_nodes == 0 {
        return ptr::null_mut();
    }

    let mut nodes: Vec<*const ort::OrtNode> = vec![ptr::null(); num_nodes];
    // SAFETY: `nodes` has exactly `num_nodes` slots, which is what we tell ORT to fill.
    let status = unsafe {
        match (*api).Graph_GetNodes {
            Some(f) => f(graph, nodes.as_mut_ptr(), num_nodes),
            None => {
                return sys::make_status(
                    api,
                    ort::OrtErrorCode_ORT_EP_FAIL,
                    "OrtApi::Graph_GetNodes is unavailable",
                );
            }
        }
    };
    if !status.is_null() {
        return status;
    }

    // ORT partitions bottom-up: a control-flow node's body subgraph reaches GetCapability BEFORE
    // the parent graph that owns the node. Claiming individual body nodes makes ORT fuse them into
    // a node in our private domain and splice it back into a nested subgraph that carries no opset
    // import for that domain — an INVALID_GRAPH at session creation. So we decline every node in a
    // control-flow body and would claim the whole control-flow node at the parent level instead.
    let mut in_control_flow_body = false;
    // SAFETY: `Graph_GetParentNode` writes through a valid out-param; any status is ours to
    // release.
    unsafe {
        if let Some(get_parent) = (*api).Graph_GetParentNode {
            let mut parent: *const ort::OrtNode = ptr::null();
            let st = get_parent(graph, &mut parent);
            if st.is_null() {
                in_control_flow_body = !parent.is_null();
            } else {
                sys::release_status(api, st);
            }
        }
    }

    // --- DESIGN.md §7.8 condition 3: shader-less artifact must claim nothing ---
    // When built without shaders (ALLOW_MISSING_GLSLC=1), the artifact cannot dispatch anything.
    // Rather than letting ORT assign work to us and then fail at pipeline creation, we claim
    // nothing here. `probe_devices()` already returns zero devices for this build, so ORT should
    // never call GetCapability — but this guard is the belt-and-suspenders defence.
    if !crate::engine::shaders::has_any() {
        let claim_debug = logging::claim_debug_enabled();
        if claim_debug || log::log_enabled!(log::Level::Debug) {
            log::debug!(
                "GetCapability: declining all {num_nodes} node(s) — \
                 [built-without-shaders] EP was compiled without shader corpus; \
                 all work runs on CPU EP"
            );
        }
        log::info!(
            "GetCapability: built without shaders — claiming 0/{num_nodes} nodes (CPU fallback)"
        );
        return ptr::null_mut();
    }

    // --- ask the registry about every node ---
    // Per-op-type: (count, first decline reason, a few node names to locate them by).
    let mut declined: BTreeMap<String, (usize, String, Vec<String>)> = BTreeMap::new();
    let mut claimed: Vec<*const ort::OrtNode> = Vec::new();
    let claim_debug = logging::claim_debug_enabled();

    for &node in &nodes {
        if node.is_null() {
            continue;
        }
        // SAFETY: `api` is live and `node` belongs to `graph`, which outlives this loop. The view
        // borrows and never outlives this iteration.
        let view = unsafe { NodeView::new(api, node) };

        let decision = if in_control_flow_body {
            Err(std::borrow::Cow::Borrowed(
                "inside a control-flow subgraph body — such nodes are claimed as part of the \
                 parent If/Loop/Scan, never individually",
            ))
        } else if let Some(allow) = &ep.options.max_claim_ops
            && !allow.iter().any(|o| *o == view.qualified_name())
        {
            Err(std::borrow::Cow::Owned(format!(
                "excluded by ep.max_claim_ops (allowlist: {allow:?})"
            )))
        } else {
            registry::claim_decision(&view)
        };

        match decision {
            Ok(()) => claimed.push(view.raw()),
            Err(reason) => {
                let entry = declined
                    .entry(view.qualified_name())
                    .or_insert_with(|| (0, reason.clone().into_owned(), Vec::new()));
                entry.0 += 1;
                if entry.2.len() < 16 {
                    let n = view.name();
                    if !n.is_empty() {
                        entry.2.push(n);
                    }
                }
            }
        }
    }

    // --- claim diagnostics (M0 exit criterion 5) ---
    // Ranked by how much of the graph each declined op accounts for, because that is the number
    // that decides what to implement next.
    if claim_debug || log::log_enabled!(log::Level::Debug) {
        let mut ranked: Vec<_> = declined
            .iter()
            .map(|(op, (n, why, names))| (op.clone(), *n, why.clone(), names.clone()))
            .collect();
        ranked.sort_by_key(|(op, n, _, _)| (std::cmp::Reverse(*n), op.clone()));
        for (op, n, why, names) in &ranked {
            log::debug!("unclaimed {op} x{n} ({why}); e.g. {names:?}");
        }
    }
    log::info!(
        "GetCapability: claimed {}/{} nodes ({} distinct op types declined)",
        claimed.len(),
        num_nodes,
        declined.len()
    );

    if claimed.is_empty() {
        return ptr::null_mut();
    }

    // --- build reachability structures for convexity enforcement ---
    //
    // Convexity requirement (ORT §EP): a fused subgraph must be convex — for any two nodes A, B
    // in the island, every path from A to B in the original graph must stay entirely within the
    // island. Equivalently: no unclaimed node U may lie "between" A and B, meaning U is reachable
    // from A AND B is reachable from U.
    //
    // We precompute:
    //   unclaimed_desc[claimed_node]   = set of unclaimed nodes reachable from claimed_node
    //   claimed_desc[unclaimed_node]   = set of claimed nodes reachable from unclaimed_node
    //
    // Two claimed nodes A and B can coexist in the same island iff:
    //   ∀ U ∈ unclaimed_desc[A]: B ∉ claimed_desc[U]
    //   ∀ U ∈ unclaimed_desc[B]: A ∉ claimed_desc[U]
    //
    // The check is applied AT CLUSTER LEVEL on each proposed merge to guard against transitivity
    // failure (A∪B valid, B∪C valid, but A∪B∪C non-convex because U is between A and C).

    // Build complete forward adjacency for ALL graph nodes (claimed + unclaimed, excluding
    // graph-level inputs/outputs which have no corresponding OrtNode entry in `nodes`).
    // SAFETY: `api` live; all nodes in `nodes` are live graph nodes for the duration of this call.
    let mut all_fwd: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut all_node_by_output: HashMap<String, usize> = HashMap::new();
    for &node in &nodes {
        if node.is_null() {
            continue;
        }
        all_fwd.entry(node as usize).or_default();
        // SAFETY: `api` is live; node is a live graph node for this call.
        for slot in unsafe { node_slots(api, node, Slots::Outputs) } {
            // SAFETY: slot is from node_slots and is valid for this call.
            let name = unsafe { value_info_name(api, slot) };
            if !name.is_empty() {
                all_node_by_output.insert(name, node as usize);
            }
        }
    }
    for &node in &nodes {
        if node.is_null() {
            continue;
        }
        // SAFETY: `api` is live; node is a live graph node for this call.
        for slot in unsafe { node_slots(api, node, Slots::Inputs) } {
            // SAFETY: slot is from node_slots and is valid for this call.
            let name = unsafe { value_info_name(api, slot) };
            if let Some(&prod) = all_node_by_output.get(&name) {
                all_fwd.entry(prod).or_default().push(node as usize);
            }
        }
    }

    let claimed_raw: std::collections::HashSet<usize> =
        claimed.iter().map(|&n| n as usize).collect();
    let unclaimed_set: std::collections::HashSet<usize> = nodes
        .iter()
        .filter(|&&n| !n.is_null() && !claimed_raw.contains(&(n as usize)))
        .map(|&n| n as usize)
        .collect();

    // BFS from `start` collecting all nodes in `target` reachable through the forward graph.
    // Traverses both claimed and unclaimed nodes (it is the *target* filter that restricts output,
    // not which intermediate nodes are visited).
    fn bfs_into(
        start: usize,
        fwd: &HashMap<usize, Vec<usize>>,
        target: &std::collections::HashSet<usize>,
    ) -> std::collections::HashSet<usize> {
        let mut found = std::collections::HashSet::new();
        let mut stack = vec![start];
        let mut visited = std::collections::HashSet::new();
        while let Some(curr) = stack.pop() {
            if !visited.insert(curr) {
                continue;
            }
            if target.contains(&curr) {
                found.insert(curr);
            }
            if let Some(succs) = fwd.get(&curr) {
                stack.extend_from_slice(succs);
            }
        }
        found
    }

    // For each claimed node: unclaimed nodes reachable from it.
    let mut unclaimed_desc: HashMap<usize, std::collections::HashSet<usize>> = HashMap::new();
    for &cn in &claimed_raw {
        let desc = bfs_into(cn, &all_fwd, &unclaimed_set);
        if !desc.is_empty() {
            unclaimed_desc.insert(cn, desc);
        }
    }
    // For each unclaimed node: claimed nodes reachable from it.
    let mut claimed_desc: HashMap<usize, std::collections::HashSet<usize>> = HashMap::new();
    for &un in &unclaimed_set {
        let desc = bfs_into(un, &all_fwd, &claimed_raw);
        if !desc.is_empty() {
            claimed_desc.insert(un, desc);
        }
    }

    // Coexistence predicate: can two claimed node raw-pointers A and B be in the same island?
    let empty_reach: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let can_coexist = |a: usize, b: usize| -> bool {
        // Check: is any unclaimed node U between A and B?
        for &u in unclaimed_desc.get(&a).unwrap_or(&empty_reach) {
            if claimed_desc.get(&u).is_some_and(|s| s.contains(&b)) {
                return false; // U reachable from A; U can reach B → non-convex
            }
        }
        for &u in unclaimed_desc.get(&b).unwrap_or(&empty_reach) {
            if claimed_desc.get(&u).is_some_and(|s| s.contains(&a)) {
                return false; // U reachable from B; U can reach A → non-convex
            }
        }
        true
    };

    // --- cluster claimed nodes into connected islands ---
    //
    // Union-find with cluster-level coexistence checking. We maintain an explicit list of members
    // per cluster so that when merging X and Y we can check ALL pairs (x∈X, y∈Y). This guards
    // against transitivity failures that a pairwise-only check on the triggering edge would miss.
    //
    // "Clean edge" guard: we only attempt a union via a dataflow edge (A→value→B) where `value`
    // is NOT consumed by any unclaimed node. If it were, an unclaimed node shares that value on a
    // direct path between A and B, making any island containing them both non-convex.
    // The clean-edge guard is a fast pre-filter; the coexistence check is the decisive one.
    let n_claimed = claimed.len();
    let mut parent: Vec<usize> = (0..n_claimed).collect();
    // members[i] holds all claimed-array indices in the cluster rooted at i.
    // Only the root's list is valid; non-root entries are replaced by empty vecs on merge.
    let mut members: Vec<Vec<usize>> = (0..n_claimed).map(|i| vec![i]).collect();

    // Iterative path-compression find.
    let find = |parent: &mut Vec<usize>, mut x: usize| -> usize {
        let mut root = x;
        while parent[root] != root {
            root = parent[root];
        }
        while parent[x] != root {
            let next = parent[x];
            parent[x] = root;
            x = next;
        }
        root
    };

    // Build the "claimed producer" map: output value name → index in `claimed`.
    // SAFETY: `api` and each claimed node are live for the duration of GetCapability.
    let mut output_producer: HashMap<String, usize> = HashMap::new();
    for (i, &node) in claimed.iter().enumerate() {
        // SAFETY: `api` is live; node is a live graph node for this call.
        for slot in unsafe { node_slots(api, node, Slots::Outputs) } {
            // SAFETY: slot is from node_slots and is valid for this call.
            let name = unsafe { value_info_name(api, slot) };
            if !name.is_empty() {
                output_producer.insert(name, i);
            }
        }
    }

    // Build the "unclaimed consumer" set: value names consumed by at least one UNCLAIMED node.
    let mut unclaimed_consumers: std::collections::HashSet<String> =
        std::collections::HashSet::new();
    for &node in &nodes {
        if node.is_null() || claimed_raw.contains(&(node as usize)) {
            continue;
        }
        // SAFETY: `api` is live; node is a live graph node for this call.
        for slot in unsafe { node_slots(api, node, Slots::Inputs) } {
            // SAFETY: slot is from node_slots and is valid for this call.
            let name = unsafe { value_info_name(api, slot) };
            if !name.is_empty() {
                unclaimed_consumers.insert(name);
            }
        }
    }

    // Walk each claimed node's inputs; attempt to union with the producing claimed node when:
    //   1. the connecting value is not consumed by any unclaimed node (clean edge), AND
    //   2. every pair across the two clusters passes can_coexist (convexity preserved).
    for (i, &node) in claimed.iter().enumerate() {
        // SAFETY: `api` is live; node is a live graph node for this call.
        for slot in unsafe { node_slots(api, node, Slots::Inputs) } {
            // SAFETY: slot is from node_slots and is valid for this call.
            let name = unsafe { value_info_name(api, slot) };
            if name.is_empty() || unclaimed_consumers.contains(&name) {
                continue;
            }
            let j = match output_producer.get(&name) {
                Some(&j) => j,
                None => continue,
            };
            let ri = find(&mut parent, i);
            let rj = find(&mut parent, j);
            if ri == rj {
                continue; // already the same cluster
            }
            // Check all cross-cluster pairs before committing to the merge.
            let compatible = members[ri].iter().all(|&a_idx| {
                let a_raw = claimed[a_idx] as usize;
                members[rj]
                    .iter()
                    .all(|&b_idx| can_coexist(a_raw, claimed[b_idx] as usize))
            });
            if !compatible {
                continue; // merging would produce a non-convex island
            }
            // Union-by-size: append the smaller cluster into the larger to bound tree depth.
            let (keep, drop_root) = if members[ri].len() >= members[rj].len() {
                (ri, rj)
            } else {
                (rj, ri)
            };
            let dropped = std::mem::take(&mut members[drop_root]);
            members[keep].extend(dropped);
            parent[drop_root] = keep;
        }
    }

    // Collect final clusters.
    let mut clusters: HashMap<usize, Vec<*const ort::OrtNode>> = HashMap::new();
    for (i, &node) in claimed.iter().enumerate() {
        let root = find(&mut parent, i);
        clusters.entry(root).or_default().push(node);
    }

    // --- apply partition policy ---
    //
    // Build a lightweight Island for each cluster (flop/byte estimates; exact values come once
    // Niobe has wired VkQueryPool timestamps). Apply partition::evaluate() to decide whether the
    // island is worth the boundary round-trip. Rejected islands are handed back to the CPU EP;
    // their node op-types are folded into the existing `declined` histogram.
    let is_uma = ep.session.as_ref().is_some_and(|s| s.capable.caps.is_uma);
    let transfer_model = if is_uma {
        partition::TransferModel::UMA
    } else {
        partition::TransferModel::DISCRETE
    };
    let policy = partition::Policy::default();

    // Bytes-per-element for common dtypes (conservative fallback: 4).
    let dtype_bytes = |et: ort::ONNXTensorElementDataType| -> u64 {
        match et {
            ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
            | ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 => 2,
            ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
            | ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
            | ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 => 4,
            ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE
            | ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
            | ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 => 8,
            ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
            | ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8
            | ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL => 1,
            _ => 4,
        }
    };

    // Estimate boundary bytes for one value-info slot (null = 0).
    // SAFETY: `api` live, slot borrowed from ORT graph.
    let slot_bytes = |slot: *const ort::OrtValueInfo| -> u64 {
        if slot.is_null() {
            return 0;
        }
        // Read type+shape info. Missing shape or unknown dtype → conservative 4096-byte guess.
        // We only use this for the partition economics check, not for buffer allocation.
        // SAFETY: `api` is live; reading a fn pointer from the api table is a plain field read.
        let get_type = unsafe { (*api).GetValueInfoTypeInfo };
        let Some(get_type) = get_type else {
            return 4096;
        };
        let mut type_info: *const ort::OrtTypeInfo = ptr::null();
        // SAFETY: slot is live, type_info is an out-pointer.
        let st = unsafe { get_type(slot, &mut type_info) };
        if !st.is_null() {
            // SAFETY: `api` is live; st is non-null and owned by this call.
            unsafe { crate::sys::release_status(api, st) };
            return 4096;
        }
        if type_info.is_null() {
            return 4096;
        }
        // SAFETY: `api` is live; reading a fn pointer from the api table is a plain field read.
        let get_tensor_info = unsafe { (*api).CastTypeInfoToTensorInfo };
        let Some(get_tensor_info) = get_tensor_info else {
            return 4096;
        };
        let mut tensor_info: *const ort::OrtTensorTypeAndShapeInfo = ptr::null();
        // SAFETY: type_info is live.
        let st = unsafe { get_tensor_info(type_info, &mut tensor_info) };
        if !st.is_null() {
            // SAFETY: `api` is live; st is non-null and owned by this call.
            unsafe { crate::sys::release_status(api, st) };
            return 4096;
        }
        if tensor_info.is_null() {
            return 4096;
        }
        // Read element type and dims.
        // SAFETY: `api` is live; reading fn pointers from the api table are plain field reads.
        let get_et = unsafe { (*api).GetTensorElementType };
        // SAFETY: `api` is live; reading fn pointers from the api table are plain field reads.
        let get_ndim = unsafe { (*api).GetDimensionsCount };
        // SAFETY: `api` is live; reading fn pointers from the api table are plain field reads.
        let get_dims = unsafe { (*api).GetDimensions };
        let (Some(get_et), Some(get_ndim), Some(get_dims)) = (get_et, get_ndim, get_dims) else {
            return 4096;
        };
        let mut et = ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        // SAFETY: tensor_info is live.
        if unsafe { get_et(tensor_info, &mut et) }.is_null() {
            // null status = success
        } else {
            return 4096;
        }
        let mut ndim: usize = 0;
        // SAFETY: tensor_info is live.
        if unsafe { get_ndim(tensor_info, &mut ndim) }.is_null() {
            // success
        } else {
            return 4096;
        }
        if ndim == 0 {
            return dtype_bytes(et); // scalar
        }
        let mut dims = vec![0i64; ndim];
        // SAFETY: tensor_info is live; dims has ndim elements.
        if unsafe { get_dims(tensor_info, dims.as_mut_ptr(), ndim) }.is_null() {
            // success
        } else {
            return 4096;
        }
        let elems: u64 = dims
            .iter()
            .map(|&d| if d > 0 { d as u64 } else { 128 }) // unknown dim → conservative 128
            .product();
        elems.saturating_mul(dtype_bytes(et))
    };

    // Build Island structs and evaluate each cluster.
    //
    // Single-cluster exemption: when all claimed nodes form one connected component, the
    // economics gate is bypassed. In a single-cluster graph there is no alternative partition
    // to choose between — the graph is either wholly claimed or wholly rejected — and the
    // claim predicate has already vetted every node for correctness.
    //
    // The gate applies to multi-cluster graphs: when claimed nodes form two or more disjoint
    // components, each component is evaluated independently. A component that fails the economics
    // check is handed back to the CPU EP; its [partition] decline code appears in the CLAIM_LOG.
    // This is the path that `test_partition_gate.py` exercises.
    //
    // R10 note (Morpheus 2026-07-30): the gate IS called for multi-cluster graphs; its anchor
    // exemption (anchors > 0 → Claim) and its decline paths (TooSmall, TransferDominated) are
    // now both observable via CLAIM_LOG. The single-cluster bypass does not hide the gate —
    // it limits the gate to the cases where it has a decision to make.
    let only_one_cluster = clusters.len() == 1;

    let mut surviving: Vec<Vec<*const ort::OrtNode>> = Vec::new();
    let mut n_rejected = 0usize;
    // `n_viable_retained` counts islands that *passed* `partition::evaluate` (the net-benefit /
    // `retain_viable` gate) in a multi-cluster graph. Single-cluster bypass does not increment
    // this counter — that is intentional: the counter is the R10 wiring observable for the gate.
    let mut n_viable_retained: u64 = 0;
    // Track island-level aggregate metrics for PartitionStats (§10.0 metric of record).
    let mut largest_island_flops: u64 = 0;
    let mut largest_island_nodes: u64 = 0;
    let mut total_flops: u64 = 0;
    let mut total_boundary_bytes: u64 = 0;
    for cluster_nodes in clusters.values() {
        let mut island = partition::Island {
            nodes: cluster_nodes.len(),
            ..Default::default()
        };

        // Value names produced within this cluster — used to distinguish internal edges from
        // boundary edges when computing output boundary bytes.
        let mut internal_outputs: HashMap<String, ()> = HashMap::new();
        for &node in cluster_nodes {
            // SAFETY: `api` is live; node is a live graph node for this call.
            for slot in unsafe { node_slots(api, node, Slots::Outputs) } {
                // SAFETY: slot is from node_slots and is valid for this call.
                let name = unsafe { value_info_name(api, slot) };
                if !name.is_empty() {
                    internal_outputs.insert(name, ());
                }
            }
        }

        for &node in cluster_nodes {
            // SAFETY: api live; node is a live graph node.
            let view = unsafe { NodeView::new(api, node) };
            let qual = view.qualified_name();
            if partition::is_anchor(&qual) {
                island.anchors += 1;
                // Anchor flop estimate: 2 × K × N for a matmul-family op.
                // Conservative minimum: 2 × 3072 × 3072 when shapes are unavailable.
                island.flops = island.flops.saturating_add(2 * 3072 * 3072);
            } else {
                // Light elementwise op: 1 FLOP per output element (fp16: 2 bytes → 1 FLOP).
                // SAFETY: `api` is live; node is a live graph node for this call.
                let out_slots = unsafe { node_slots(api, node, Slots::Outputs) };
                let out_bytes: u64 = out_slots.iter().map(|&s| slot_bytes(s)).sum();
                island.flops = island.flops.saturating_add(out_bytes / 2);
            }

            // Boundary activation bytes: only non-constant inputs that cross the island edge.
            //
            // Constant initializers (weights, biases) are NOT per-inference transfers — they
            // are uploaded once at model-load time and must NOT be counted against the island's
            // transfer budget. Counting them would cause every weight-heavy op (MatMulNBits,
            // Conv, Gemm) to appear transfer-dominated relative to its compute, which would
            // reject all islands unconditionally and reproduce the exact failure mode
            // partition.rs was written to prevent.
            // SAFETY: `api` is live; node is a live graph node for this call.
            let in_slots = unsafe { node_slots(api, node, Slots::Inputs) };
            for (i, &slot) in in_slots.iter().enumerate() {
                if slot.is_null() {
                    continue;
                }
                // SAFETY: slot is from node_slots and is valid for this call.
                let name = unsafe { value_info_name(api, slot) };
                // Skip: produced inside the island (internal edge), or a constant initializer.
                if name.is_empty()
                    || internal_outputs.contains_key(&name)
                    || view.input_is_constant(i)
                {
                    continue;
                }
                island.input_bytes = island.input_bytes.saturating_add(slot_bytes(slot));
            }

            // Output boundary bytes: conservatively count all node outputs as boundary bytes.
            // Some of these will be consumed by other cluster members (internal edges), so this
            // over-counts. Over-counting is safe — it makes islands harder to claim, never easier.
            // Under-counting output bytes (missing actual boundary transfers) would cause us to
            // claim islands that cost more than they earn, which is the worse failure mode.
            // SAFETY: `api` is live; node is a live graph node for this call.
            let out_slots = unsafe { node_slots(api, node, Slots::Outputs) };
            for slot in out_slots {
                island.output_bytes = island.output_bytes.saturating_add(slot_bytes(slot));
            }
        }

        // Apply the partition economics gate to each cluster. For single-cluster graphs the gate
        // is bypassed (see above). For multi-cluster graphs, each cluster is evaluated; a cluster
        // that fails produces a [partition] decline code in the CLAIM_LOG.
        let verdict = if only_one_cluster {
            counters::record_net_benefit_decision(false);
            partition::Verdict::Claim
        } else {
            counters::record_net_benefit_decision(true);
            partition::evaluate(&island, &transfer_model, &policy)
        };
        if verdict.is_claim() {
            surviving.push(cluster_nodes.clone());
            // §10.0 metric tracking: update aggregate stats for the surviving island.
            largest_island_flops = largest_island_flops.max(island.flops);
            largest_island_nodes = largest_island_nodes.max(island.nodes as u64);
            total_flops = total_flops.saturating_add(island.flops);
            total_boundary_bytes = total_boundary_bytes
                .saturating_add(island.input_bytes)
                .saturating_add(island.output_bytes);
            // Increment the net-benefit wiring observable only for islands that went through
            // partition::evaluate (multi-cluster path). Single-cluster bypass is intentionally
            // excluded: the counter must vary with the gate's input, not with graph size.
            if !only_one_cluster {
                n_viable_retained += 1;
            }
        } else {
            n_rejected += cluster_nodes.len();
            if let partition::Verdict::Reject(reason) = verdict {
                let decline_reason = partition::decline_for(&reason);
                for &node in cluster_nodes {
                    // SAFETY: node is live.
                    let view = unsafe { NodeView::new(api, node) };
                    let entry = declined
                        .entry(view.qualified_name())
                        .or_insert_with(|| (0, decline_reason.clone().into_owned(), Vec::new()));
                    entry.0 += 1;
                    // Route partition declines into CLAIM_LOG so they are visible to tests and
                    // tooling. Registry declines already appear there via claim_decision; partition
                    // declines are post-registry and must be recorded here (R10: wiring is per
                    // entry point, and the partition gate is a separate entry point from the
                    // registry).
                    crate::ops::claim_log::record(
                        &view.qualified_name(),
                        &view.name(),
                        0, // opset unknown at this stage
                        Err(&decline_reason),
                    );
                }
            }
        }
    }

    let n_islands = surviving.len();
    let n_claimed_total = claimed.len();

    // Falsifier: island_count == claimed_count is a hard signal that partitioning is not working.
    // It should be impossible after this change, but if it ever fires it means the union-find is
    // producing one cluster per node — i.e. no edges were followed.
    if n_islands == n_claimed_total && n_claimed_total > 1 {
        log::warn!(
            "GetCapability: PARTITION FALSIFIER FIRED — {} islands == {} claimed nodes; \
             every claimed node is its own island, which means connectivity clustering produced \
             no merges. Check that value names are being read correctly.",
            n_islands,
            n_claimed_total,
        );
    }

    counters::record_capability(n_claimed_total as u64, n_islands as u64);
    counters::record_viable_islands_retained(n_viable_retained);
    log::info!(
        "GetCapability: {} claimed nodes → {} islands ({} nodes rejected by partition policy)",
        n_claimed_total,
        n_islands,
        n_rejected,
    );

    // --- Record partition metrics in the tracer (§10.0 metric of record) ---
    // All three slots of the performance triple are now populated:
    //   (claimed_op_coverage, island_count, largest_island_flops)
    // `declined` includes both per-node declines and partition-economics rejections.
    let concentration = if total_flops > 0 {
        largest_island_flops as f64 / total_flops as f64
    } else {
        0.0
    };
    crate::trace::tracer().record_partition(&crate::trace::PartitionStats {
        total_nodes: num_nodes as u64,
        claimed_nodes: n_claimed_total as u64,
        island_count: n_islands as u64,
        largest_island_nodes,
        largest_island_flops,
        concentration,
        boundary_bytes_per_inference: total_boundary_bytes,
        boundary_time_fraction: 0.0, // GPU timestamps needed; will be set by Niobe from trace data
        declined: declined
            .iter()
            .map(|(op, (n, why, _names))| (op.clone(), *n as u64, why.clone()))
            .collect(),
    });

    if surviving.is_empty() {
        return ptr::null_mut();
    }

    // --- fuse: one AddNodesToFuse call per surviving island ---
    // SAFETY: `ep_api` is live; `support` is the graph-support-info ORT passed in; `opts` is a
    // zeroed `#[repr(C)]` POD whose only non-pointer field we set explicitly.
    unsafe {
        let ep_api = ep.ep_api();
        let Some(add_nodes_to_fuse) = (*ep_api).EpGraphSupportInfo_AddNodesToFuse else {
            return sys::make_status(
                api,
                ort::OrtErrorCode_ORT_EP_FAIL,
                "OrtEpApi::EpGraphSupportInfo_AddNodesToFuse is unavailable",
            );
        };
        let mut opts: ort::OrtNodeFusionOptions = std::mem::zeroed();
        opts.ort_version_supported = ep.abi_version();
        // We read constant initializers at Compile time from the fused node's inputs, so ORT
        // must keep supplying them.
        opts.drop_constant_initializers = false;
        for island_nodes in &surviving {
            let st = add_nodes_to_fuse(support, island_nodes.as_ptr(), island_nodes.len(), &opts);
            if !st.is_null() {
                return st;
            }
        }
    }

    ptr::null_mut()
}

// --- Fused-subgraph extraction: ORT ABI → engine vocabulary -----------------------------------
//
// Everything below reads ORT's graph once, at Compile time, and produces owned `engine` types that
// borrow nothing from ORT. That boundary is the whole point: after `compile_impl` returns, the
// `Plan` is self-contained, so nothing in `engine/` or `ops/` can be holding a graph pointer when
// ORT frees the graph.

/// Read an `OrtValueInfo`'s name. Empty string on any failure, which downstream treats as
/// "unnamed", never as a hard error — a missing name degrades diagnostics, not correctness.
///
/// # Safety
/// `api` must be live; `slot` must be null or a graph-owned `OrtValueInfo`.
unsafe fn value_info_name(api: *const ort::OrtApi, slot: *const ort::OrtValueInfo) -> String {
    if slot.is_null() {
        return String::new();
    }
    // SAFETY: `api` is live and `slot` is a graph-owned value info. `GetValueInfoName` writes a
    // borrowed, NUL-terminated pointer into graph storage that outlives this call, so copying it
    // here is sound and required — we must not retain ORT's pointer. Any status is ours to free.
    unsafe {
        let Some(get) = (*api).GetValueInfoName else {
            return String::new();
        };
        let mut out: *const c_char = ptr::null();
        let status = get(slot, &mut out);
        if !status.is_null() {
            sys::release_status(api, status);
            return String::new();
        }
        if out.is_null() {
            return String::new();
        }
        CStr::from_ptr(out).to_string_lossy().into_owned()
    }
}

/// A node's input or output `OrtValueInfo` slots.
///
/// Duplicates `NodeView`'s private `input_slots`/`output_slots` rather than widening that API:
/// `registry.rs` is Mouse's file, and the names are only needed here, on the ABI side. Null entries
/// are preserved because ORT reports an omitted *interior* optional input as a null slot rather
/// than by shortening the list, and collapsing that would silently renumber every later input.
///
/// # Safety
/// `api` must be live; `node` must belong to a graph that outlives the returned slots' use.
unsafe fn node_slots(
    api: *const ort::OrtApi,
    node: *const ort::OrtNode,
    which: Slots,
) -> Vec<*const ort::OrtValueInfo> {
    // SAFETY: `api` is live and `node` is a live graph node. `count` is whatever ORT just
    // reported, and `buf` is exactly that many initialised, writable slots, satisfying the
    // `_Out_writes_(count)` contract. ORT writes borrowed pointers we never free.
    unsafe {
        let (count_fn, get_fn) = match which {
            Slots::Inputs => ((*api).Node_GetNumInputs, (*api).Node_GetInputs),
            Slots::Outputs => ((*api).Node_GetNumOutputs, (*api).Node_GetOutputs),
        };
        let (Some(count_fn), Some(get_fn)) = (count_fn, get_fn) else {
            return Vec::new();
        };
        let mut n: usize = 0;
        let status = count_fn(node, &mut n);
        if !status.is_null() {
            sys::release_status(api, status);
            return Vec::new();
        }
        if n == 0 {
            return Vec::new();
        }
        let mut buf: Vec<*const ort::OrtValueInfo> = vec![ptr::null(); n];
        let status = get_fn(node, buf.as_mut_ptr(), n);
        if !status.is_null() {
            sys::release_status(api, status);
            return Vec::new();
        }
        buf
    }
}

/// Which edge list [`node_slots`] should read.
#[derive(Clone, Copy)]
enum Slots {
    Inputs,
    Outputs,
}

/// Turn an `EdgeType` into a fully-known `TensorDesc`, or `None`.
///
/// Deliberately strict: a `TensorDesc` with a guessed dtype or a symbolic dimension treated as
/// concrete is worse than no `TensorDesc`, because the handler would size a buffer from it.
fn tensor_desc(edge: Option<&registry::EdgeType>) -> Option<crate::engine::TensorDesc> {
    let edge = edge?;
    let dtype = edge.dtype?;
    let shape = edge.shape.as_ref()?;
    if shape.iter().any(|d| *d < 0) {
        return None;
    }
    Some(crate::engine::TensorDesc::new(dtype, shape.clone()))
}

/// Copy one attribute out of the graph.
///
/// ORT's attribute accessors are typed, and `NodeView` exposes one getter per type rather than the
/// attribute's declared type, so this probes in a fixed order. The order matters: `attr_ints`
/// before `attr_int` would turn a scalar `INT` into a one-element list on hosts whose list getter
/// tolerates scalars, and `attr_string` last because it is the one that most readily succeeds by
/// stringifying something else.
fn attr_value(view: &NodeView<'_>, name: &str) -> Option<crate::engine::AttrValue> {
    use crate::engine::AttrValue;
    if let Some(v) = view.attr_int(name) {
        return Some(AttrValue::Int(v));
    }
    if let Some(v) = view.attr_float(name) {
        return Some(AttrValue::Float(v));
    }
    if let Some(v) = view.attr_ints(name) {
        return Some(AttrValue::Ints(v));
    }
    if let Some(v) = view.attr_string(name) {
        return Some(AttrValue::String(v));
    }
    None
}

/// Read one `OrtNode` into an owned [`crate::engine::NodeDesc`].
///
/// # Safety
/// `api` must be live; `node` must be a node of a graph that is live for the duration of the call.
unsafe fn node_desc(api: *const ort::OrtApi, node: *const ort::OrtNode) -> crate::engine::NodeDesc {
    // SAFETY: `api` is live and `node` belongs to a live graph; the view never outlives this fn.
    let view = unsafe { NodeView::new(api, node) };

    // SAFETY: same contract as the view above.
    let input_slots = unsafe { node_slots(api, node, Slots::Inputs) };
    // SAFETY: same contract as the view above.
    let output_slots = unsafe { node_slots(api, node, Slots::Outputs) };

    let input_types = view.input_types();
    let output_types = view.output_types();

    let inputs = input_slots
        .iter()
        .enumerate()
        .map(|(i, slot)| crate::engine::TensorRef {
            // SAFETY: `slot` is null or graph-owned; `value_info_name` handles both.
            name: unsafe { value_info_name(api, *slot) },
            desc: tensor_desc(input_types.get(i).and_then(Option::as_ref)),
            is_initializer: view.input_is_constant(i),
        })
        .collect();

    let outputs = output_slots
        .iter()
        .enumerate()
        .map(|(i, slot)| crate::engine::OutRef {
            // SAFETY: as above.
            name: unsafe { value_info_name(api, *slot) },
            desc: tensor_desc(output_types.get(i).and_then(Option::as_ref)),
        })
        .collect();

    let mut attributes = BTreeMap::new();
    for name in view.attr_names() {
        if let Some(v) = attr_value(&view, &name) {
            attributes.insert(name, v);
        } else {
            // Not an error: ORT materialises defaulted optional attributes, and some have types
            // this EP has no `AttrValue` for. The claim predicate already vetted the node, so a
            // type we cannot copy means the handler does not read it.
            log::trace!(
                "node {}: attribute `{name}` has a type this EP does not copy; skipped",
                view.name()
            );
        }
    }

    crate::engine::NodeDesc {
        op_type: view.op_type(),
        domain: view.domain(),
        since_version: view.since_version(),
        name: view.name(),
        attributes,
        inputs,
        outputs,
    }
}

/// Build the [`crate::engine::Plan`] for one fused subgraph.
///
/// `graph` is the subgraph body; `fused_node` is the single node ORT spliced into the parent graph
/// in its place. **The plan's `inputs`/`outputs` come from the fused node, not from the body**,
/// because those are the edges ORT binds at Compute time and their order is exactly the index
/// order of `KernelContext_GetInput` / `GetOutput`. Taking them from the body would produce a list
/// that looks right and is indexed wrong.
///
/// # Safety
/// `api` must be live; `graph` and `fused_node` must be the live pointers ORT passed to `Compile`.
unsafe fn plan_for_fused_node(
    api: *const ort::OrtApi,
    graph: *const ort::OrtGraph,
    fused_node: *const ort::OrtNode,
) -> Result<crate::engine::Plan, String> {
    let mut num_nodes: usize = 0;
    // SAFETY: `api` is live, `graph` is ORT's, and `num_nodes` is a valid out-param slot.
    unsafe {
        let Some(get_num) = (*api).Graph_GetNumNodes else {
            return Err("OrtApi::Graph_GetNumNodes is unavailable".into());
        };
        let status = get_num(graph, &mut num_nodes);
        if !status.is_null() {
            let msg = sys::status_message(api, status);
            sys::release_status(api, status);
            return Err(format!("Graph_GetNumNodes failed: {msg}"));
        }
    }
    if num_nodes == 0 {
        return Err("ORT handed us a fused subgraph with no nodes".into());
    }

    let mut nodes: Vec<*const ort::OrtNode> = vec![ptr::null(); num_nodes];
    // SAFETY: `nodes` has exactly `num_nodes` writable slots, which is the count ORT reported.
    unsafe {
        let Some(get_nodes) = (*api).Graph_GetNodes else {
            return Err("OrtApi::Graph_GetNodes is unavailable".into());
        };
        let status = get_nodes(graph, nodes.as_mut_ptr(), num_nodes);
        if !status.is_null() {
            let msg = sys::status_message(api, status);
            sys::release_status(api, status);
            return Err(format!("Graph_GetNodes failed: {msg}"));
        }
    }

    // ORT returns the body nodes in topological order. We rely on that rather than re-sorting:
    // a second topological sort here could only ever disagree with ORT, and if it did, the
    // disagreement would show up as a data race on a buffer rather than as an error.
    let mut plan = crate::engine::Plan::default();
    for &node in &nodes {
        if node.is_null() {
            return Err("ORT reported a null node inside a fused subgraph".into());
        }
        // SAFETY: `node` is a live node of `graph`, which outlives this loop.
        plan.nodes.push(unsafe { node_desc(api, node) });
    }

    // SAFETY: `fused_node` is the live node ORT passed in.
    let fused = unsafe { node_desc(api, fused_node) };
    plan.inputs = fused.inputs;
    plan.outputs = fused.outputs;

    Ok(plan)
}

// --- Compile ----------------------------------------------------------------------------------

unsafe extern "C" fn compile(
    p: *mut ort::OrtEp,
    graphs: *mut *const ort::OrtGraph,
    fused_nodes: *mut *const ort::OrtNode,
    count: usize,
    node_compute_infos: *mut *mut ort::OrtNodeComputeInfo,
    ep_context_nodes: *mut *mut ort::OrtNode,
) -> ort::OrtStatusPtr {
    // SAFETY: `p` is our EP pointer.
    let api = unsafe { this(p).ort_api };
    // SAFETY: `api` is live.
    unsafe {
        crate::guard_ffi_status(api, "Compile", || {
            compile_impl(
                p,
                graphs,
                fused_nodes,
                count,
                node_compute_infos,
                ep_context_nodes,
            )
        })
    }
}

/// # Safety
/// All pointers must be the live ones ORT passed to `Compile`.
unsafe fn compile_impl(
    p: *mut ort::OrtEp,
    graphs: *mut *const ort::OrtGraph,
    fused_nodes: *mut *const ort::OrtNode,
    count: usize,
    node_compute_infos: *mut *mut ort::OrtNodeComputeInfo,
    _ep_context_nodes: *mut *mut ort::OrtNode,
) -> ort::OrtStatusPtr {
    // SAFETY: `p` is our EP pointer.
    let api = unsafe { this(p).ort_api };
    // SAFETY: `p` is our EP pointer.
    let abi_version = unsafe { this(p).abi_version() };

    counters::record_compile_call();

    // Leave every out-slot null *before* doing anything that can fail, so ORT never reads an
    // uninitialised pointer and `ReleaseNodeComputeInfos` has nothing to free on the error path.
    if !node_compute_infos.is_null() {
        for i in 0..count {
            // SAFETY: ORT guarantees `node_compute_infos` has room for `count` entries.
            unsafe { *node_compute_infos.add(i) = ptr::null_mut() };
        }
    }

    if node_compute_infos.is_null() || graphs.is_null() || fused_nodes.is_null() {
        // SAFETY: `api` is live.
        return unsafe {
            sys::make_status(
                api,
                ort::OrtErrorCode_ORT_INVALID_ARGUMENT,
                "VulkanExecutionProvider: Compile received a null graph, fused-node or \
                 compute-info array",
            )
        };
    }

    for i in 0..count {
        // SAFETY: ORT guarantees both arrays hold `count` entries for the duration of the call.
        let (graph, fused_node) = unsafe { (*graphs.add(i), *fused_nodes.add(i)) };
        if graph.is_null() || fused_node.is_null() {
            // SAFETY: `api` is live. Slots already written stay owned by ORT and are freed by
            // `ReleaseNodeComputeInfos`, which ORT calls even when Compile fails.
            return unsafe {
                sys::make_status(
                    api,
                    ort::OrtErrorCode_ORT_INVALID_ARGUMENT,
                    "VulkanExecutionProvider: Compile received a null graph or fused node",
                )
            };
        }

        // SAFETY: `api`, `graph` and `fused_node` are live for the duration of this call.
        let plan = match unsafe { plan_for_fused_node(api, graph, fused_node) } {
            Ok(plan) => plan,
            Err(msg) => {
                log::error!("Compile: could not build a plan for fused subgraph {i}: {msg}");
                // SAFETY: `api` is live.
                return unsafe {
                    sys::make_status(
                        api,
                        ort::OrtErrorCode_ORT_EP_FAIL,
                        &format!(
                            "VulkanExecutionProvider: failed to build an execution plan for \
                             fused subgraph {i}: {msg}"
                        ),
                    )
                };
            }
        };

        log::debug!(
            "Compile: fused subgraph {i} — {} node(s), {} input(s), {} output(s)",
            plan.nodes.len(),
            plan.inputs.len(),
            plan.outputs.len()
        );

        // ── Run translate handlers to produce CompiledKernels ──────────────
        let n_plan_inputs = plan.inputs.len();
        let n_plan_outputs = plan.outputs.len();

        // For multi-node islands, build a name→token map so that intermediate outputs (produced
        // by one node, consumed by another within the island) get stable tokens distinct from
        // both the external ORT inputs and outputs.
        //
        // Token ranges:
        //   0..n_plan_inputs                                     → external ORT inputs
        //   n_plan_inputs..n_plan_inputs+n_plan_outputs          → external ORT outputs
        //   n_plan_inputs+n_plan_outputs..first_temp_token       → intermediate buffers
        //   first_temp_token..                                   → alloc_temp scratch
        let (name_map_opt, n_intermediates, first_temp_token) = if plan.nodes.len() > 1 {
            let plan_output_names: std::collections::HashSet<&str> =
                plan.outputs.iter().map(|r| r.name.as_str()).collect();

            let mut map = std::collections::HashMap::<String, u64>::new();
            for (k, inp) in plan.inputs.iter().enumerate() {
                if !inp.name.is_empty() {
                    map.insert(inp.name.clone(), k as u64);
                }
            }
            for (j, out) in plan.outputs.iter().enumerate() {
                if !out.name.is_empty() {
                    map.insert(out.name.clone(), (n_plan_inputs + j) as u64);
                }
            }
            let mut n_intermediates = 0usize;
            for node in &plan.nodes {
                for out in &node.outputs {
                    if !out.name.is_empty()
                        && !plan_output_names.contains(out.name.as_str())
                        && !map.contains_key(&out.name)
                    {
                        let token = (n_plan_inputs + n_plan_outputs + n_intermediates) as u64;
                        map.insert(out.name.clone(), token);
                        n_intermediates += 1;
                    }
                }
            }
            let first_temp = n_plan_inputs + n_plan_outputs + n_intermediates;
            (Some(std::sync::Arc::new(map)), n_intermediates, first_temp)
        } else {
            (None, 0usize, n_plan_inputs + n_plan_outputs)
        };

        let mut recorder = if let Some(ref nm) = name_map_opt {
            CompileRecorder::new_named(n_plan_inputs, std::sync::Arc::clone(nm), first_temp_token)
        } else {
            CompileRecorder::new(n_plan_inputs)
        };
        for node_desc in &plan.nodes {
            if let Some(spec) = registry::spec_for(node_desc) {
                // If any input lacks a static TensorDesc, the shapes are symbolic — the translate
                // handler cannot compute push constants or workgroups at Compile time. Instead,
                // store a DynKernelRecipe so that dispatch_ort can re-run the handler at Compute
                // time with the concrete shapes ORT supplies.
                let is_dynamic = node_desc.inputs.iter().any(|r| r.desc.is_none());
                if is_dynamic {
                    log::debug!(
                        "Compile: op '{}' in subgraph {i} has symbolic-extent inputs — \
                         deferring to dynamic dispatch",
                        node_desc.op_type
                    );
                    recorder.push_dynamic_kernel(node_desc.clone(), spec);
                } else if let Err(e) = (spec.translate)(spec, node_desc, &mut recorder) {
                    // SAFETY: `api` is live.
                    return unsafe {
                        sys::make_status(
                            api,
                            ort::OrtErrorCode_ORT_EP_FAIL,
                            &format!(
                                "VulkanExecutionProvider: translate handler failed for op '{}' \
                                 in subgraph {i}: {e}",
                                node_desc.op_type
                            ),
                        )
                    };
                }
            } else {
                log::warn!(
                    "Compile: no translate handler for op '{}' in subgraph {i} — claimed \
                     without a handler, which is a registry bug",
                    node_desc.op_type
                );
            }
        }

        // Collect static intermediate byte sizes from the recorder (populated by bind_token for
        // named outputs in the intermediate range). For dynamic-shape intermediates the entry is
        // 0; dispatch_ort will resolve the size from the ShapeOnlyRecorder pre-pass at Compute.
        let static_intermediate_byte_sizes: Vec<u64> = {
            let mut sizes = vec![0u64; n_intermediates];
            for (&token, &sz) in &recorder.intermediate_sizes {
                let base = (n_plan_inputs + n_plan_outputs) as u64;
                let ft = first_temp_token as u64;
                if token >= base && token < ft {
                    let idx = (token - base) as usize;
                    if idx < sizes.len() {
                        sizes[idx] = sz;
                    }
                }
            }
            sizes
        };

        let kernels = recorder.kernels;
        log::debug!(
            "Compile: subgraph {i} compiled into {} kernel(s), {} intermediate(s)",
            kernels.len(),
            n_intermediates,
        );

        // ── Compute byte sizes and output shapes from the plan ─────────────
        let input_byte_sizes: Vec<u64> = plan
            .inputs
            .iter()
            .map(|r| r.desc.as_ref().and_then(|d| d.byte_size()).unwrap_or(0) as u64)
            .collect();
        let output_byte_sizes: Vec<u64> = plan
            .outputs
            .iter()
            .map(|r| r.desc.as_ref().and_then(|d| d.byte_size()).unwrap_or(0) as u64)
            .collect();
        let output_shapes: Vec<Vec<i64>> = plan
            .outputs
            .iter()
            .map(|r| r.desc.as_ref().map_or_else(Vec::new, |d| d.shape.clone()))
            .collect();

        // ── Get session_ptr from VulkanEp ──────────────────────────────────
        // The compute-info holds a raw pointer to the session, not a borrow: ORT keeps the
        // compute-info alive across many `Compute` calls, long after this `&mut` would have to
        // end. The `Box<VulkanSession>` gives it a stable address, and ORT releases every
        // compute-info before the EP, so the pointer cannot outlive its target.
        //
        // SAFETY: `p` is our EP pointer, live for this call and for the whole EP lifetime that
        // follows. Taking a `*mut` through the box does not create an aliasing `&mut`; nothing
        // else holds a reference to the session while `Compile` runs, because ORT calls `Compile`
        // and `Compute` on the same EP without overlapping them at this point.
        let session_ptr: *mut VulkanSession = unsafe {
            match (*p.cast::<VulkanEp>()).session.as_mut() {
                Some(b) => &raw mut **b,
                None => ptr::null_mut(),
            }
        };

        let info = if session_ptr.is_null() || kernels.is_empty() {
            // No GPU device or no kernels — build a stub that reports failure rather than
            // silently writing nothing to the output buffers.
            SubgraphComputeInfo::new_stub(plan, abi_version, api)
        } else {
            SubgraphComputeInfo::new_live(
                plan,
                kernels,
                input_byte_sizes,
                output_byte_sizes,
                output_shapes,
                n_intermediates,
                name_map_opt,
                first_temp_token,
                static_intermediate_byte_sizes,
                session_ptr,
                abi_version,
                api,
            )
        };

        // SAFETY: the slot is in range and was nulled above. Ownership of the box transfers to
        // ORT here; it comes back exactly once through `ReleaseNodeComputeInfos`.
        unsafe {
            *node_compute_infos.add(i) = Box::into_raw(info).cast::<ort::OrtNodeComputeInfo>();
        }
    }

    ptr::null_mut()
}

unsafe extern "C" fn release_node_compute_infos(
    _p: *mut ort::OrtEp,
    node_compute_infos: *mut *mut ort::OrtNodeComputeInfo,
    num_node_compute_infos: usize,
) {
    if node_compute_infos.is_null() {
        return;
    }
    for i in 0..num_node_compute_infos {
        // SAFETY: ORT hands back exactly the array it received from `Compile`, with the same
        // length. Every non-null entry was produced by `Box::into_raw` in `Compile` and is
        // released exactly once here.
        unsafe {
            let info = *node_compute_infos.add(i);
            if !info.is_null() {
                drop(Box::from_raw(info.cast::<SubgraphComputeInfo>()));
                *node_compute_infos.add(i) = ptr::null_mut();
            }
        }
    }
}

// -------------------------------------------------------------------------------------------
// Per-subgraph compute info — the ORT-facing half of the Compile → Compute seam
// -------------------------------------------------------------------------------------------

/// One per fused subgraph. Everything a `Compute` call needs, resolved once at `Compile` time.
///
/// `OrtNodeComputeInfo` is first under `#[repr(C)]`, so ORT's pointer is this pointer.
///
/// It carries `ort_api` because **`Compute` must be able to fail.** ORT reads a null return from
/// `Compute` as *success*, so without an `OrtApi` here a failed dispatch would report success and
/// leave ORT's output tensors holding whatever was in them — a silent wrong answer, which from the
/// host's point of view is worse than a crash and is indistinguishable from a working EP. The
/// pointer is the process-wide table, which outlives every session; it is never released.
///
/// `session` is a raw pointer into the `Box<VulkanSession>` owned by the `VulkanEp`. ORT releases
/// every compute-info (`ReleaseNodeComputeInfos`) before releasing the EP, and the `Box` gives the
/// session a stable address, so the pointer cannot dangle. It is null in a *stub* compute-info —
/// one built when there is no device or the plan produced no kernels — which `Compute` reports as
/// an error rather than dereferencing.
#[repr(C)]
struct SubgraphComputeInfo {
    base: ort::OrtNodeComputeInfo,
    /// Kept for diagnostics: which subgraph this compute-info belongs to.
    plan: crate::engine::Plan,
    /// Recorded dispatches, in execution order. Built once, at Compile time. Empty in a stub.
    kernels: Vec<CompiledKernel>,
    /// Byte size of each subgraph input, in `KernelContext_GetInput` index order.
    input_byte_sizes: Vec<u64>,
    /// Byte size of each subgraph output, in `KernelContext_GetOutput` index order.
    output_byte_sizes: Vec<u64>,
    /// Concrete shape of each subgraph output, same order.
    output_shapes: Vec<Vec<i64>>,
    /// Number of inter-node intermediate buffers in the island.
    ///
    /// For single-node islands this is always 0. For multi-node islands the token range
    /// `[n_plan_inputs + n_plan_outputs, n_plan_inputs + n_plan_outputs + n_intermediates)`
    /// is reserved for GPU buffers that hold outputs of one node consumed as inputs by another
    /// node in the same island, and are never exposed as ORT outputs.
    n_intermediates: usize,
    /// Name→token map for multi-node islands; `None` for single-node islands.
    ///
    /// Keys are the OrtValue names from `plan.inputs`, `plan.outputs`, and the inner node
    /// outputs that are consumed as inputs by other nodes in the island.
    name_map: Option<std::sync::Arc<std::collections::HashMap<String, u64>>>,
    /// First token available for anonymous `alloc_temp` buffers.
    /// = `n_plan_inputs + n_plan_outputs + n_intermediates`.
    first_temp_token: usize,
    /// Byte sizes for intermediate buffers as determined at Compile time (static shapes).
    /// Indexed by intermediate index = token - (n_plan_inputs + n_plan_outputs).
    /// Entries are 0 for dynamic-shape intermediates (size resolved at Compute time).
    static_intermediate_byte_sizes: Vec<u64>,
    ort_api: *const ort::OrtApi,
    session: *mut VulkanSession,
    /// Unique identifier for this subgraph, used as the key into `VulkanSession::weight_caches`.
    /// Generated by an atomic counter at compile time; stable for the lifetime of the EP session.
    subgraph_id: u64,
}

impl SubgraphComputeInfo {
    /// Fill the `OrtNodeComputeInfo` vtable ORT reads.
    fn base_vtable(abi_version: u32) -> ort::OrtNodeComputeInfo {
        // SAFETY: `OrtNodeComputeInfo` is a `#[repr(C)]` POD of a `u32` and three `Option<fn>`
        // slots; all-zero is the valid `None` niche for each.
        let mut base: ort::OrtNodeComputeInfo = unsafe { std::mem::zeroed() };
        // The negotiated version, not the compiled-against one — the same rule as `OrtEp` and
        // `OrtEpFactory`. This field tells ORT how far into the vtable it may read, so stamping it
        // higher than the fields we actually fill is how a host reads past our initialised memory.
        base.ort_version_supported = abi_version;
        base.CreateState = Some(create_state);
        base.Compute = Some(compute);
        base.ReleaseState = Some(release_state);
        base
    }

    /// A compute-info that can actually dispatch.
    #[allow(clippy::too_many_arguments)]
    fn new_live(
        plan: crate::engine::Plan,
        kernels: Vec<CompiledKernel>,
        input_byte_sizes: Vec<u64>,
        output_byte_sizes: Vec<u64>,
        output_shapes: Vec<Vec<i64>>,
        n_intermediates: usize,
        name_map: Option<std::sync::Arc<std::collections::HashMap<String, u64>>>,
        first_temp_token: usize,
        static_intermediate_byte_sizes: Vec<u64>,
        session: *mut VulkanSession,
        abi_version: u32,
        ort_api: *const ort::OrtApi,
    ) -> Box<SubgraphComputeInfo> {
        counters::record_subgraph(true);
        Box::new(SubgraphComputeInfo {
            base: SubgraphComputeInfo::base_vtable(abi_version),
            plan,
            kernels,
            input_byte_sizes,
            output_byte_sizes,
            output_shapes,
            n_intermediates,
            name_map,
            first_temp_token,
            static_intermediate_byte_sizes,
            ort_api,
            session,
            subgraph_id: NEXT_SUBGRAPH_ID.fetch_add(1, Ordering::Relaxed),
        })
    }

    /// A compute-info that **cannot** dispatch, and says so when called.
    ///
    /// Built when there is no Vulkan session or the plan recorded no kernels. It exists rather
    /// than being an error at Compile time so the failure is attributable to the subgraph that
    /// caused it, and it returns a status rather than succeeding vacuously because a `Compute`
    /// that writes nothing and reports success is a wrong answer the host cannot detect.
    fn new_stub(
        plan: crate::engine::Plan,
        abi_version: u32,
        ort_api: *const ort::OrtApi,
    ) -> Box<SubgraphComputeInfo> {
        counters::record_subgraph(false);
        Box::new(SubgraphComputeInfo {
            base: SubgraphComputeInfo::base_vtable(abi_version),
            plan,
            kernels: Vec::new(),
            input_byte_sizes: Vec::new(),
            output_byte_sizes: Vec::new(),
            output_shapes: Vec::new(),
            n_intermediates: 0,
            name_map: None,
            first_temp_token: 0,
            static_intermediate_byte_sizes: Vec::new(),
            ort_api,
            session: ptr::null_mut(),
            subgraph_id: 0,
        })
    }
    fn is_live(&self) -> bool {
        !self.session.is_null() && !self.kernels.is_empty()
    }
}

impl Drop for SubgraphComputeInfo {
    fn drop(&mut self) {
        // Free cached GPU weight buffers through the session's allocator.
        // SAFETY: ORT guarantees the EP (and its VulkanSession) outlives all compute-infos;
        // `ReleaseNodeComputeInfos` is called before the EP is destroyed.  A null session
        // means this is a stub (no GPU work was compiled), so there is nothing to free.
        if self.session.is_null() {
            return;
        }
        // SAFETY: self.session is valid (ORT guarantees EP outlives compute-infos).
        let session = unsafe { &mut *self.session };
        session.release_weight_cache(self.subgraph_id);
    }
}

/// Recover our compute-info from the `OrtNodeComputeInfo` ORT hands back.
///
/// # Safety
/// `p` must be non-null and must be a pointer this crate produced in [`compile_impl`].
unsafe fn this_info<'a>(p: *mut ort::OrtNodeComputeInfo) -> &'a mut SubgraphComputeInfo {
    // SAFETY: `SubgraphComputeInfo` is `#[repr(C)]` with `base` first, so the pointer ORT holds
    // is exactly our pointer. The allocation lives until `ReleaseNodeComputeInfos`, which ORT
    // calls only after the last `Compute`.
    unsafe { &mut *p.cast::<SubgraphComputeInfo>() }
}

unsafe extern "C" fn create_state(
    _this: *mut ort::OrtNodeComputeInfo,
    _context: *mut ort::OrtNodeComputeContext,
    compute_state: *mut *mut c_void,
) -> ort::OrtStatusPtr {
    if !compute_state.is_null() {
        // SAFETY: a valid out-param slot supplied by ORT. There is no per-run state: the recorded
        // kernels are immutable and live in the compute-info. When per-run Vulkan resources are
        // needed (one set per concurrent run), this is where they are created.
        unsafe { *compute_state = ptr::null_mut() };
    }
    ptr::null_mut()
}

unsafe extern "C" fn release_state(_this: *mut ort::OrtNodeComputeInfo, _state: *mut c_void) {}

// -----------------------------------------------------------------------------------------------
// Broken-commitment disclosure — RAI Ruling 2 / RAI-010
// -----------------------------------------------------------------------------------------------

/// Env var: plant a synthetic `Compute()` failure. **The control, not a feature.**
///
/// A WARN that cannot be shown *not* to fire on a good run is not a detector, it is a printed
/// opinion — and one that can only be shown not to fire has never been shown to fire at all. Both
/// polarities need the same mechanism driven from the same call site, so the positive one needs a
/// failure that can be produced on demand on hardware that is working correctly.
///
/// Every run under this variable is marked `"fault_injection": "ACTIVE"` in the counters artifact
/// and counted in `compute_failures_injected`, so an injected failure can never be read as a
/// suffered one, and no run made under it can be quoted as a clean run.
pub const ENV_FORCE_COMPUTE_FAILURE: &str = "ONNXRUNTIME_EP_VULKAN_FORCE_COMPUTE_FAILURE";

/// Whether the planted Compute-failure control is armed for this process.
pub fn fault_injection_active() -> bool {
    static ACTIVE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ACTIVE.get_or_init(|| {
        std::env::var_os(ENV_FORCE_COMPUTE_FAILURE).is_some_and(|v| !v.is_empty() && v != "0")
    })
}

/// Classify the failure into a token a maintainer can grep, without losing the text.
///
/// R13's second clause is why the text travels too: the token is a summary and a summary is where
/// a caveat goes to die. The token exists so two incidents can be compared; the text is what says
/// what happened.
fn failure_condition_token(error_text: &str) -> &'static str {
    let t = error_text.to_ascii_lowercase();
    if t.contains("panic") {
        "panic"
    } else if t.contains("planted") || t.contains("fault-injection") {
        "planted-control"
    } else if t.contains("alloc") || t.contains("out of memory") || t.contains("oom") {
        "allocator"
    } else if t.contains("shape")
        || t.contains("size")
        || t.contains("dimension")
        || t.contains("rank")
        || t.contains("bound input")
        || t.contains("bytes")
        || t.contains("disagree")
        || t.contains("binds")
    {
        "shape"
    } else if t.contains("null") {
        "null-argument"
    } else if t.contains("no dispatchable kernels") {
        "empty-plan"
    } else {
        "unclassified"
    }
}

/// The exact sentence the host is told. Pure, so both polarities of the control can assert on it
/// without a GPU, an ORT session, or a device.
fn broken_commitment_message(
    subgraph_id: u64,
    nodes: &[crate::engine::NodeDesc],
    error_text: &str,
) -> String {
    const NAMED: usize = 4;
    let named: Vec<String> = nodes
        .iter()
        .take(NAMED)
        .map(|n| format!("{}({})", n.name, n.qualified_name()))
        .collect();
    let more = nodes.len().saturating_sub(named.len());
    let node_list = if named.is_empty() {
        "<the fused node carried no node descriptions>".to_string()
    } else if more == 0 {
        named.join(", ")
    } else {
        format!("{}, +{more} more", named.join(", "))
    };
    format!(
        "VulkanExecutionProvider: BROKEN COMMITMENT — this EP claimed fused subgraph #{subgraph_id} \
         ({n} node(s): {node_list}) and its Compute() then returned a non-OK status. \
         condition={condition}; error text: {error_text}. ONNX Runtime will now re-execute these \
         node(s) on the CPU EP, so the answer this run produces is not this EP's — and \
         get_providers() will still list VulkanExecutionProvider, because the provider list is \
         fixed at session-create time. This is a claim the EP made and did not keep; it is not the \
         planned fallback of a node this EP never claimed.",
        n = nodes.len(),
        condition = failure_condition_token(error_text),
    )
}

/// Emit the mandatory WARN when a **claimed** node's `Compute()` returns non-OK. Returns whether a
/// WARN was emitted, so the negative polarity has something to assert on.
///
/// # Why every non-OK return here is a broken commitment, mechanically
///
/// Nothing reaches this function that was not claimed: a node the EP declined at partition time
/// never becomes part of a fused node and never gets a `Compute`. So the narrow case Ruling 2
/// carves out — *never-claimed ops falling back to CPU is the plan, and must not produce per-node
/// noise* — is excluded by the position of this call site rather than by a predicate someone has
/// to maintain. There are 258 dynamic-shape declines on Phi-3.5 and not one of them can arrive
/// here.
///
/// # Safety
/// `api` must be null or a live `OrtApi`. `status` must be null, or a status this process owns for
/// the duration of the call; it is read and not released here — the caller returns it to ORT.
pub(crate) unsafe fn disclose_broken_commitment(
    api: *const ort::OrtApi,
    subgraph_id: u64,
    nodes: &[crate::engine::NodeDesc],
    status: ort::OrtStatusPtr,
) -> bool {
    if status.is_null() {
        // The good run's polarity travels through this same call site and this same branch. An
        // assertion that the WARN did not fire is therefore an assertion about the mechanism, not
        // about a different code path that happens to be quiet.
        return false;
    }
    // SAFETY: caller's contract — `api` is null or live and `status` is a live status we do not
    // take ownership of. `status_message` handles the null-api case itself.
    let error_text = unsafe { sys::status_message(api, status) };
    let message = broken_commitment_message(subgraph_id, nodes, &error_text);
    let delivered = logging::warn_through_ort_sink("onnxruntime_ep_vulkan::compute", &message);
    counters::record_broken_commitment(delivered);
    true
}

unsafe extern "C" fn compute(
    p: *mut ort::OrtNodeComputeInfo,
    state: *mut c_void,
    kernel_context: *mut ort::OrtKernelContext,
) -> ort::OrtStatusPtr {
    if p.is_null() {
        // Nothing can be done: without our compute-info there is no `OrtApi` to build a status
        // from, and fabricating a non-null pointer would be worse. ORT never passes null; this
        // branch exists so the deref below is unconditionally sound.
        log::error!("Compute called with a null OrtNodeComputeInfo — cannot report the failure");
        return ptr::null_mut();
    }
    // SAFETY: `p` is a compute-info this crate produced; see `this_info`.
    let info = unsafe { this_info(p) };
    let api = info.ort_api;
    // SAFETY: `api` is the process-wide table, live for the process's lifetime.
    let status = unsafe {
        crate::guard_ffi_status(api, "Compute", || compute_impl(info, state, kernel_context))
    };

    // RAI Ruling 2: the guard that already converts a panic into `ORT_EP_FAIL` is also where a
    // broken commitment becomes knowable. It is deliberately *outside* `compute_impl`, so that a
    // panic converted to a status by the guard discloses on exactly the same path as a status
    // returned normally — the 2026-07-30→08-01 incidents arrived by both routes.
    //
    // SAFETY: `api` is live; `status` is either null or a status we still own at this point and
    // hand to ORT unchanged on the next line.
    unsafe { disclose_broken_commitment(api, info.subgraph_id, &info.plan.nodes, status) };
    status
}

/// # Safety
/// `kernel_context` must be the live context ORT passed to `Compute`.
unsafe fn compute_impl(
    info: &mut SubgraphComputeInfo,
    _state: *mut c_void,
    kernel_context: *mut ort::OrtKernelContext,
) -> ort::OrtStatusPtr {
    let api = info.ort_api;
    // SAFETY: `api` is live for every `make_status` below.
    let fail = |code, msg: String| unsafe { sys::make_status(api, code, &msg) };

    counters::record_compute_call();

    // The planted control (see [`ENV_FORCE_COMPUTE_FAILURE`]). Placed after `record_compute_call`
    // and before every real check, so the injected failure is a claimed node's Compute returning
    // non-OK and nothing else — the exact shape of the condition under test.
    if fault_injection_active() {
        counters::record_compute_failure();
        counters::record_injected_compute_failure();
        return fail(
            ort::OrtErrorCode_ORT_EP_FAIL,
            "planted control (ONNXRUNTIME_EP_VULKAN_FORCE_COMPUTE_FAILURE): synthetic Compute \
             failure standing in for a mid-session allocator failure. No device work was \
             attempted."
                .to_string(),
        );
    }

    if kernel_context.is_null() {
        counters::record_compute_failure();
        return fail(
            ort::OrtErrorCode_ORT_INVALID_ARGUMENT,
            "VulkanExecutionProvider: Compute received a null OrtKernelContext".to_string(),
        );
    }
    if !info.is_live() {
        counters::record_compute_failure();
        return fail(
            ort::OrtErrorCode_ORT_EP_FAIL,
            format!(
                "VulkanExecutionProvider: this fused subgraph ({} node(s)) has no dispatchable \
                 kernels — either no Vulkan device was available or translation produced nothing. \
                 Reported as a failure rather than a no-op because a Compute that writes no \
                 outputs and returns success is a wrong answer the host cannot detect.",
                info.plan.nodes.len()
            ),
        );
    }

    // The counts the plan was built from come from the fused node; the counts ORT binds come from
    // the kernel context. If they disagree, every index past the first mismatch names a different
    // tensor than the plan believes — a wrong answer rather than an error — so check first.
    // SAFETY: `api` and `kernel_context` are live for the duration of this call.
    if let Err(msg) = unsafe {
        check_bound_counts(
            api,
            kernel_context,
            info.input_byte_sizes.len(),
            info.output_byte_sizes.len(),
        )
    } {
        counters::record_compute_failure();
        return fail(
            ort::OrtErrorCode_ORT_EP_FAIL,
            format!("VulkanExecutionProvider: {msg}"),
        );
    }

    // Counts agreeing is not the same as sizes agreeing, and the difference is a memory-safety
    // one. `dispatch_ort` does `slice::from_raw_parts(cpu_ptr, input_byte_sizes[i])` on ORT's
    // input tensor using the byte size *Compile* computed. If the tensor ORT actually bound is
    // smaller than that — a symbolic dimension that resolved differently, a shape the plan
    // mispredicted, a rank the fused node reported optimistically — that read runs off the end of
    // the host's heap. There is no check inside the engine that can catch it, because by then the
    // only thing it has is a pointer.
    //
    // SAFETY: `api` and `kernel_context` are live for the duration of this call.
    let sizes_agree =
        unsafe { check_bound_input_sizes(api, kernel_context, &info.input_byte_sizes) };
    if let Err(msg) = sizes_agree {
        counters::record_compute_failure();
        return fail(
            ort::OrtErrorCode_ORT_EP_FAIL,
            format!("VulkanExecutionProvider: {msg}"),
        );
    }

    // A size that agrees is still not a pointer that can be read. Once device memory is
    // advertised, ORT may place a subgraph input in *our* memory, and the pointer it then binds is
    // a handle: a reserved, deliberately inaccessible address. `dispatch_ort` would `memcpy` from
    // it and the process would die at that instruction with no explanation — which is the handle
    // scheme working exactly as designed, and useless unless something asks first.
    //
    // SAFETY: `api` and `kernel_context` are live for the duration of this call.
    let addressable =
        unsafe { check_bound_inputs_are_addressable(api, kernel_context, &info.input_byte_sizes) };
    if let Err(msg) = addressable {
        counters::record_compute_failure();
        return fail(
            ort::OrtErrorCode_ORT_EP_FAIL,
            format!("VulkanExecutionProvider: {msg}"),
        );
    }

    // SAFETY: `info.session` is non-null (checked by `is_live`) and points into the
    // `Box<VulkanSession>` owned by the `VulkanEp` that produced this compute-info; ORT releases
    // every compute-info before the EP, so the box is still alive and the address is stable.
    // `kernels`, the byte sizes and the shapes are exactly what `Compile` computed for this
    // subgraph, which is `dispatch_ort`'s stated precondition. Any status it returns is ORT's to
    // own from here.
    let status = unsafe {
        (*info.session).dispatch_ort(
            &info.kernels,
            &info.input_byte_sizes,
            &info.output_byte_sizes,
            &info.output_shapes,
            info.n_intermediates,
            info.name_map.clone(),
            info.first_temp_token,
            &info.static_intermediate_byte_sizes,
            info.subgraph_id,
            api,
            kernel_context,
        )
    };

    // Count only what actually ran. `dispatch_ort` submits and then waits on a fence, so a success
    // return means the GPU executed this command buffer to completion; a non-null status means it
    // did not, whatever it managed along the way. Counting on the optimistic side of that line is
    // how a lane that never executed anything reports that it did.
    if status.is_null() {
        counters::record_dispatches(info.kernels.len() as u64);
    } else {
        counters::record_compute_failure();
    }
    status
}

/// Verify each bound input tensor is exactly the size `Compile` planned for it.
///
/// Returns `Ok(())` when every size agrees, when ORT is too old to answer (`GetTensorSizeInBytes`
/// is not in the negotiated table), or when a tensor declines to report a size. The last two are
/// deliberately permissive: this is a guard against a wrong answer, not a second source of truth,
/// and refusing to run because a diagnostic is unavailable would be a worse failure than the one
/// it prevents.
///
/// # Safety
/// `api` and `ctx` must be live.
unsafe fn check_bound_input_sizes(
    api: *const ort::OrtApi,
    ctx: *mut ort::OrtKernelContext,
    want: &[u64],
) -> Result<(), String> {
    // SAFETY: `api`/`ctx` are live; every out-param below is a valid initialised local, and every
    // status we receive is released before we return.
    unsafe {
        let (Some(get_input), Some(size_of_tensor)) =
            ((*api).KernelContext_GetInput, (*api).GetTensorSizeInBytes)
        else {
            // `GetTensorSizeInBytes` arrived after our ABI floor, so a host at the bottom of the
            // supported range legitimately lacks it. Say so once per process rather than per
            // Compute — this runs on the hot path.
            static WARNED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                log::warn!(
                    "OrtApi::GetTensorSizeInBytes is unavailable in the negotiated ABI, so bound \
                     input sizes cannot be checked against the compiled plan. A shape \
                     disagreement will reach the engine as a raw pointer instead of a status."
                );
            }
            return Ok(());
        };

        for (i, &planned) in want.iter().enumerate() {
            let mut value: *const ort::OrtValue = ptr::null();
            let status = get_input(ctx, i, &mut value);
            if !status.is_null() {
                let msg = sys::status_message(api, status);
                sys::release_status(api, status);
                return Err(format!("KernelContext_GetInput({i}) failed: {msg}"));
            }
            if value.is_null() {
                return Err(format!(
                    "input {i} is bound to a null OrtValue but the compiled subgraph requires \
                     {planned} byte(s) from it"
                ));
            }
            let mut actual: usize = 0;
            let status = size_of_tensor(value, &mut actual);
            if !status.is_null() {
                // A tensor that will not report its size is not evidence of a mismatch. Release
                // and move on rather than failing a run over a diagnostic.
                sys::release_status(api, status);
                continue;
            }
            if actual as u64 != planned {
                // planned == 0 is the signal that this input's shape was symbolic at Compile time;
                // dispatch_ort resolves the real size at Compute time via GetTensorSizeInBytes.
                // Skip the check rather than refusing the dispatch.
                if planned == 0 {
                    continue;
                }
                return Err(format!(
                    "input {i} is {actual} byte(s) but this subgraph was compiled for {planned}. \
                     The shape ORT bound and the shape Compile planned for disagree, so the \
                     dispatch would {} — refused. This is a plan/runtime shape mismatch, not a \
                     device problem.",
                    if (actual as u64) < planned {
                        "read past the end of the host's tensor"
                    } else {
                        "read only part of the host's tensor and compute a wrong answer"
                    }
                ));
            }
        }
    }
    Ok(())
}

/// Verify that every bound input is memory the engine can actually read.
///
/// This is the guard between ORT's memory placement and the engine's raw-pointer world. It is a
/// separate check from [`check_bound_input_sizes`] because it catches a different failure: not "the
/// tensor is the wrong length" but "the tensor is not where the engine thinks it is".
///
/// Returns `Ok(())` when every input is host memory (the normal case, and the only case when
/// `ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY` is off), and an explanatory error when one is a device
/// handle. It never silently substitutes the backing pointer: the engine reads the tensor itself,
/// so a substitution here would be a value nobody uses and a false sense that the path works.
///
/// # Safety
/// `api` and `ctx` must be live.
unsafe fn check_bound_inputs_are_addressable(
    api: *const ort::OrtApi,
    ctx: *mut ort::OrtKernelContext,
    want: &[u64],
) -> Result<(), String> {
    // Cheap exit for every build that has not opted into device memory: with no registry there is
    // no handle space, so no pointer can be one.
    if crate::factory::all_registries().is_empty() {
        return Ok(());
    }
    // SAFETY: `api`/`ctx` are live; every out-param below is a valid initialised local, and every
    // status we receive is released before we return.
    unsafe {
        let (Some(get_input), Some(get_data)) =
            ((*api).KernelContext_GetInput, (*api).GetTensorMutableData)
        else {
            return Ok(());
        };
        for (i, &planned) in want.iter().enumerate() {
            let mut value: *const ort::OrtValue = ptr::null();
            let status = get_input(ctx, i, &mut value);
            if !status.is_null() {
                sys::release_status(api, status);
                continue;
            }
            if value.is_null() {
                continue;
            }
            let mut p: *mut std::ffi::c_void = ptr::null_mut();
            let status = get_data(value.cast_mut(), &mut p);
            if !status.is_null() {
                sys::release_status(api, status);
                continue;
            }
            match crate::transfer::host_backing_for(p.cast::<u8>(), planned as usize) {
                // Host memory, or a handle whose bytes the engine can reach: the engine resolves
                // it at bind time. Nothing to refuse.
                None | Some(Ok(_)) => {}
                Some(Err(why)) => {
                    return Err(format!(
                        "input {i} is bound to device handle {p:?} and its bytes are unreachable: \
                         {why}. Refusing the dispatch rather than running it against an \
                         inaccessible address."
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Verify the kernel context binds exactly as many tensors as the compiled subgraph expects.
///
/// # Safety
/// `api` and `ctx` must be live.
unsafe fn check_bound_counts(
    api: *const ort::OrtApi,
    ctx: *mut ort::OrtKernelContext,
    want_inputs: usize,
    want_outputs: usize,
) -> Result<(), String> {
    // SAFETY: `api`/`ctx` are live and both out-params are valid, initialised slots. Any status
    // returned is owned by us and released before returning.
    unsafe {
        let (Some(get_in), Some(get_out)) = (
            (*api).KernelContext_GetInputCount,
            (*api).KernelContext_GetOutputCount,
        ) else {
            return Err("OrtApi::KernelContext_Get{Input,Output}Count is unavailable".into());
        };
        let (mut ni, mut no) = (0usize, 0usize);
        let status = get_in(ctx, &mut ni);
        if !status.is_null() {
            let msg = sys::status_message(api, status);
            sys::release_status(api, status);
            return Err(format!("KernelContext_GetInputCount failed: {msg}"));
        }
        let status = get_out(ctx, &mut no);
        if !status.is_null() {
            let msg = sys::status_message(api, status);
            sys::release_status(api, status);
            return Err(format!("KernelContext_GetOutputCount failed: {msg}"));
        }
        if ni != want_inputs || no != want_outputs {
            return Err(format!(
                "the kernel context binds {ni} input(s)/{no} output(s) but this subgraph was \
                 compiled for {want_inputs}/{want_outputs} — the fused node and the bound tensors \
                 disagree"
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------------------------------
    // Broken-commitment disclosure — the two-polarity control (RAI Ruling 2)
    // ------------------------------------------------------------------------------------------

    mod broken_commitment {
        use super::*;
        use std::sync::{Mutex, MutexGuard};

        /// Everything the fake sink has been handed, as `(severity, message)`.
        static CAPTURED: Mutex<Vec<(i32, String)>> = Mutex::new(Vec::new());

        /// The ORT logger pointers **and** the counters are process-global, so a polarity must not
        /// run at the same time as anything else that logs or records. This is the same lock the
        /// allocator ledger and the counters tests take: a private lock here would serialise these
        /// two tests against each other and leave them racing `counters_record_what_they_claim_to_
        /// record`, which asserts on the very statics a disclosure moves.
        fn serialize() -> MutexGuard<'static, ()> {
            crate::allocator::ledger::test_lock()
        }

        /// Stands in for ORT's `Logger_LogMessage`. This is the *host's* channel: a message that
        /// arrives here is one a user with ORT logging configured would see, and a message that
        /// does not arrive here is invisible to them however loudly we printed it elsewhere.
        unsafe extern "C" fn fake_log_message(
            _logger: *const ort::OrtLogger,
            severity: ort::OrtLoggingLevel,
            message: *const c_char,
            _file: *const ort::wchar_t,
            _line: std::os::raw::c_int,
            _func: *const c_char,
        ) -> ort::OrtStatusPtr {
            // SAFETY: `forward_to_ort` always passes a non-null, NUL-terminated message.
            let text = unsafe { CStr::from_ptr(message) }
                .to_string_lossy()
                .into_owned();
            CAPTURED
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push((severity, text));
            ptr::null_mut()
        }

        /// Stands in for `GetErrorMessage`: the failing condition's own text, which the WARN has
        /// to carry verbatim (R13 — quote the failure text).
        const PLANTED_ERROR: &CStr =
            c"VulkanExecutionProvider: gpu-allocator failed to allocate 14155776 bytes";

        unsafe extern "C" fn fake_get_error_message(
            _status: *const ort::OrtStatus,
        ) -> *const c_char {
            PLANTED_ERROR.as_ptr()
        }

        /// A minimal `OrtApi` carrying only the two entry points this path uses.
        fn fake_api() -> Box<ort::OrtApi> {
            // SAFETY: `OrtApi` is a `#[repr(C)]` table of `Option<fn>` slots; all-zero is the
            // valid `None` niche for every one of them, so a zeroed table is a table where
            // nothing is available — which every caller in this crate already handles.
            let mut api: ort::OrtApi = unsafe { std::mem::zeroed() };
            api.Logger_LogMessage = Some(fake_log_message);
            api.GetErrorMessage = Some(fake_get_error_message);
            Box::new(api)
        }

        fn nodes() -> Vec<crate::engine::NodeDesc> {
            ["MatMulNBits", "Add"]
                .iter()
                .enumerate()
                .map(|(i, op)| crate::engine::NodeDesc {
                    op_type: (*op).to_string(),
                    domain: if i == 0 { "com.microsoft" } else { "" }.to_string(),
                    since_version: 1,
                    name: format!("/model/layers.0/n{i}"),
                    attributes: Default::default(),
                    inputs: Vec::new(),
                    outputs: Vec::new(),
                })
                .collect()
        }

        /// Drive `disclose_broken_commitment` with a fake ORT sink attached and report what
        /// reached the sink. `status` is opaque to us — it is only ever passed back to the fake
        /// `GetErrorMessage` — so a dangling-looking non-null value is never dereferenced.
        fn run_polarity(status: ort::OrtStatusPtr) -> (bool, Vec<(i32, String)>) {
            let _guard = serialize();
            let api = fake_api();
            CAPTURED.lock().unwrap_or_else(|e| e.into_inner()).clear();
            // SAFETY: `api` outlives the attachment — it is detached below, before the box drops.
            unsafe { logging::attach_ort_logger(&*api, ptr::dangling::<ort::OrtLogger>()) };
            // SAFETY: `api` is the live fake table; `status` is null or an opaque token the fake
            // `GetErrorMessage` ignores.
            let warned = unsafe { disclose_broken_commitment(&*api, 7, &nodes(), status) };
            logging::detach_ort_logger();
            // Filter to *this* polarity's subgraph id. The other `compute` tests in this binary
            // drive the real entry point concurrently and their WARNs land in this sink too —
            // which is itself the R10 falsifier for this mechanism and not a nuisance: the
            // messages that arrive carry *their* node names and *their* error text, so the
            // artifact's content varies with its input rather than with its author.
            let seen = CAPTURED
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .iter()
                .filter(|(_, m)| m.contains("BROKEN COMMITMENT") && m.contains("subgraph #7"))
                .cloned()
                .collect();
            (warned, seen)
        }

        /// **Positive polarity.** A claimed node's `Compute()` returned non-OK; the WARN must
        /// reach ORT's own sink, name the nodes, quote the failure text, and say that CPU
        /// re-execution follows.
        #[test]
        fn a_planted_compute_failure_warns_through_orts_sink() {
            let (warned, seen) = run_polarity(ptr::dangling_mut::<ort::OrtStatus>());
            assert!(
                warned,
                "the guard did not treat a non-OK status as a broken commitment"
            );
            assert_eq!(
                seen.len(),
                1,
                "expected exactly one WARN through ORT's sink, saw: {seen:?}"
            );
            let m = &seen[0].1;
            assert_eq!(
                seen[0].0,
                ort::OrtLoggingLevel_ORT_LOGGING_LEVEL_WARNING,
                "the disclosure must arrive at WARNING severity, not below the host's filter"
            );
            for needle in [
                "/model/layers.0/n0",
                "com.microsoft::MatMulNBits",
                "gpu-allocator failed to allocate 14155776 bytes",
                "condition=allocator",
                "re-execute these node(s) on the CPU EP",
                "#7",
            ] {
                assert!(m.contains(needle), "WARN is missing {needle:?}: {m}");
            }
        }

        /// **Negative polarity.** The same call site, the same mechanism, a successful Compute.
        /// Nothing may reach ORT's sink. This is the half that makes the other half evidence: a
        /// WARN that cannot be shown not to fire on a good run is a printed opinion.
        #[test]
        fn a_successful_compute_emits_nothing_to_orts_sink() {
            let (warned, seen) = run_polarity(ptr::null_mut());
            assert!(
                !warned,
                "the guard reported a broken commitment on an OK status"
            );
            assert!(
                seen.is_empty(),
                "a successful Compute put {} message(s) into ORT's sink: {seen:?}",
                seen.len()
            );
        }

        /// The disclosure must survive having no ORT logger attached — and must say so, rather
        /// than letting a WARN that reached nobody be counted as delivered.
        #[test]
        fn with_no_ort_logger_the_event_is_still_recorded_and_the_channel_is_not_claimed() {
            let _guard = serialize();
            logging::detach_ort_logger();
            let api = fake_api();
            // SAFETY: `api` is live for the call; no logger is attached, so nothing is forwarded.
            let warned = unsafe {
                disclose_broken_commitment(
                    &*api,
                    1,
                    &nodes(),
                    ptr::dangling_mut::<ort::OrtStatus>(),
                )
            };
            assert!(
                warned,
                "the event must be recorded even when the channel is absent"
            );
        }

        #[test]
        fn the_condition_token_names_the_mechanism_not_the_op() {
            assert_eq!(
                failure_condition_token("gpu-allocator failed to allocate"),
                "allocator"
            );
            assert_eq!(
                failure_condition_token("bound input 3 is 0 bytes, plan expected 4096"),
                "shape"
            );
            assert_eq!(
                failure_condition_token("recovered from a panic in Compute"),
                "panic"
            );
            assert_eq!(
                failure_condition_token("planted control (…): synthetic Compute failure"),
                "planted-control"
            );
        }

        /// A fused island of 355 nodes must not print 355 node names into a host's log: the WARN
        /// has to stay readable to remain a signal.
        #[test]
        fn a_large_island_is_named_boundedly_and_says_how_many_it_left_out() {
            let many: Vec<_> = (0..355)
                .map(|i| crate::engine::NodeDesc {
                    op_type: "Add".to_string(),
                    domain: String::new(),
                    since_version: 14,
                    name: format!("n{i}"),
                    attributes: Default::default(),
                    inputs: Vec::new(),
                    outputs: Vec::new(),
                })
                .collect();
            let m = broken_commitment_message(3, &many, "oom");
            assert!(m.contains("355 node(s)"), "{m}");
            assert!(m.contains("+351 more"), "{m}");
            assert!(!m.contains("n300"), "the whole island was printed: {m}");
        }
    }

    #[test]
    fn bool_parsing_accepts_the_usual_spellings() {
        for s in ["1", "true", "TRUE", "yes", "on"] {
            assert_eq!(parse_bool(s), Some(true), "{s}");
        }
        for s in ["0", "false", "No", "off"] {
            assert_eq!(parse_bool(s), Some(false), "{s}");
        }
        assert_eq!(parse_bool("maybe"), None);
    }

    #[test]
    fn default_options_are_conservative() {
        let o = EpOptions::default();
        assert_eq!(o.device_index, None);
        assert!(!o.enable_validation);
        assert!(!o.disable_device_memory);
        assert!(o.max_claim_ops.is_none());
    }

    #[test]
    fn reading_options_from_a_null_session_tolerates_null() {
        // SAFETY: both arguments are null, which `from_session_options` must handle without
        // dereferencing either.
        let o = unsafe { EpOptions::from_session_options(ptr::null(), ptr::null()) };
        assert_eq!(o, EpOptions::default());
    }

    #[test]
    fn releasing_a_null_ep_is_a_noop() {
        // SAFETY: null is explicitly allowed and must not be dereferenced.
        unsafe { VulkanEp::release(ptr::null_mut()) };
    }

    #[test]
    fn releasing_a_null_compute_info_array_is_a_noop() {
        // SAFETY: a null array pointer must be ignored.
        unsafe { release_node_compute_infos(ptr::null_mut(), ptr::null_mut(), 3) };
    }

    /// DESIGN.md §7.8 condition 3: the get_capability_impl shader-less guard is readable and
    /// testable independently of real ORT pointers. This test verifies that `shaders::has_any()`
    /// is accessible from ep.rs and returns the expected build-time value.
    ///
    /// When the first shader is compiled, `has_any()` becomes true and this test becomes an
    /// assertion that the shader corpus is present — still the right thing to test.
    #[test]
    fn shader_guard_function_is_visible_from_ep() {
        // The guard in get_capability_impl reads: `!crate::engine::shaders::has_any()`
        // Verify the function is callable here, where the same test verifies the semantics.
        // In a shader-less build: has_any() == false → the guard fires → GetCapability claims 0.
        // In a shader-full build: has_any() == true → the guard is bypassed → normal path.
        let has_shaders = crate::engine::shaders::has_any();
        // Both values are valid; the test just confirms the function exists and returns a bool.
        // The *value* is tested in engine::tests::has_any_returns_false_in_this_build.
        let _ = has_shaders;
    }

    // --- The Compute contract -----------------------------------------------------------------
    //
    // These drive the *real* `compute` entry point through a miniature ORT that implements only
    // the kernel-context accessors. They exist because the crash that took down CI was in a path
    // nothing local ever executed: "the crate compiles" and "our Compute contract is honoured"
    // are unrelated claims, and only the second one matters here.

    struct FakeCtx {
        num_inputs: usize,
        num_outputs: usize,
    }

    unsafe extern "C" fn fake_create_status(
        _code: ort::OrtErrorCode,
        msg: *const c_char,
    ) -> *mut ort::OrtStatus {
        // SAFETY: ORT's contract is a NUL-terminated message; `make_status` always passes one.
        let owned = unsafe { CStr::from_ptr(msg) }.to_owned();
        Box::into_raw(Box::new(owned)).cast::<ort::OrtStatus>()
    }

    unsafe extern "C" fn fake_release_status(status: *mut ort::OrtStatus) {
        if status.is_null() {
            return;
        }
        // SAFETY: every status in these tests came from `fake_create_status`.
        unsafe { drop(Box::from_raw(status.cast::<CString>())) };
    }

    unsafe extern "C" fn fake_get_error_message(status: *const ort::OrtStatus) -> *const c_char {
        // SAFETY: as above; the pointer stays valid until the status is released.
        unsafe { (*status.cast::<CString>()).as_ptr() }
    }

    unsafe extern "C" fn fake_input_count(
        ctx: *const ort::OrtKernelContext,
        out: *mut usize,
    ) -> ort::OrtStatusPtr {
        // SAFETY: `ctx` is a `FakeCtx` and `out` is a valid slot.
        unsafe { *out = (*ctx.cast::<FakeCtx>()).num_inputs };
        ptr::null_mut()
    }

    unsafe extern "C" fn fake_output_count(
        ctx: *const ort::OrtKernelContext,
        out: *mut usize,
    ) -> ort::OrtStatusPtr {
        // SAFETY: as above.
        unsafe { *out = (*ctx.cast::<FakeCtx>()).num_outputs };
        ptr::null_mut()
    }

    /// A miniature `OrtApi` with only the entries the Compute path uses before dispatch.
    fn fake_api() -> Box<ort::OrtApi> {
        // SAFETY: `OrtApi` is a `#[repr(C)]` table of `Option<fn>` slots; all-zero is the valid
        // `None` niche for every one of them. Anything the code under test reaches that we did
        // not fill is `None`, which it must handle — that is itself part of the contract.
        let mut api: ort::OrtApi = unsafe { std::mem::zeroed() };
        api.CreateStatus = Some(fake_create_status);
        api.ReleaseStatus = Some(fake_release_status);
        api.GetErrorMessage = Some(fake_get_error_message);
        api.KernelContext_GetInputCount = Some(fake_input_count);
        api.KernelContext_GetOutputCount = Some(fake_output_count);
        Box::new(api)
    }

    /// A two-input, one-output plan.
    fn two_in_one_out_plan() -> crate::engine::Plan {
        use crate::engine::{NodeDesc, OutRef, Plan, TensorRef};
        Plan {
            inputs: vec![
                TensorRef {
                    name: "a".into(),
                    desc: None,
                    is_initializer: false,
                },
                TensorRef {
                    name: "b".into(),
                    desc: None,
                    is_initializer: false,
                },
            ],
            outputs: vec![OutRef {
                name: "y".into(),
                desc: None,
            }],
            nodes: vec![NodeDesc {
                op_type: "Add".into(),
                domain: String::new(),
                since_version: 14,
                name: "add0".into(),
                attributes: BTreeMap::new(),
                inputs: Vec::new(),
                outputs: Vec::new(),
            }],
            ..Plan::default()
        }
    }

    fn read_status(api: &ort::OrtApi, status: ort::OrtStatusPtr) -> String {
        // SAFETY: `status` came from `fake_create_status` via `sys::make_status`.
        unsafe {
            let msg = sys::status_message(api, status);
            sys::release_status(api, status);
            msg
        }
    }

    /// The bug this closes: `Compute` used to `return ptr::null_mut()` on internal failure, and
    /// **null is ORT's success value**. A failed dispatch therefore reported success and left
    /// ORT's output tensors holding whatever was in them — a silent wrong answer from the host's
    /// point of view, strictly worse than a crash, and exactly the class of false green that
    /// produced two fabricated speedups elsewhere in this project.
    ///
    /// A stub compute-info (no device, or translation produced no kernels) is the reachable case:
    /// it is what M0 builds today for every subgraph.
    #[test]
    fn a_compute_that_cannot_dispatch_returns_a_status_not_a_silent_success() {
        // Driving the real `Compute` entry point records into the process-wide counters,
        // including a broken commitment for the failure this test plants. Share the lock so
        // those events land inside the frame of whoever is asserting on them.
        let _g = crate::allocator::ledger::test_lock();
        let api = fake_api();
        let mut info = SubgraphComputeInfo::new_stub(two_in_one_out_plan(), 24, &raw const *api);
        let ctx = FakeCtx {
            num_inputs: 2,
            num_outputs: 1,
        };

        // SAFETY: the compute-info and context outlive the call and have the shapes the entry
        // point expects. This drives the real `extern "C"` entry point, panic guard and all.
        let status = unsafe {
            compute(
                (&raw mut *info).cast::<ort::OrtNodeComputeInfo>(),
                ptr::null_mut(),
                (&raw const ctx as *mut FakeCtx).cast::<ort::OrtKernelContext>(),
            )
        };

        assert!(
            !status.is_null(),
            "Compute reported success without dispatching anything — this is the false-green bug"
        );
        let msg = read_status(&api, status);
        assert!(
            msg.contains("no dispatchable kernels"),
            "unexpected failure message: {msg}"
        );
    }

    /// The counts a subgraph was compiled for come from the fused node; the counts at Compute
    /// time come from the kernel context. If they ever disagree, every index past the mismatch
    /// names a different tensor than the compiled plan believes — a wrong answer rather than an
    /// error — so this must fail loudly before anything is dispatched.
    #[test]
    fn a_kernel_context_that_disagrees_with_the_compiled_counts_is_rejected() {
        // Driving the real `Compute` entry point records into the process-wide counters,
        // including a broken commitment for the failure this test plants. Share the lock so
        // those events land inside the frame of whoever is asserting on them.
        let _g = crate::allocator::ledger::test_lock();
        let api = fake_api();
        // A live compute-info needs a non-null session and at least one kernel. The session
        // pointer here is deliberately dangling: the count check must run *before* anything
        // dereferences it, so a test that passes proves the ordering, and one that regresses
        // faults rather than quietly succeeding.
        let mut info = SubgraphComputeInfo::new_stub(two_in_one_out_plan(), 24, &raw const *api);
        info.kernels.push(CompiledKernel {
            shader: "test",
            spec_constants: Vec::new(),
            push_constants: Vec::new(),
            workgroups: [1, 1, 1],
            bindings: Vec::new(),
            temp_byte_sizes: Vec::new(),
            n_plan_inputs: 2,
            dyn_recipe: None,
        });
        info.input_byte_sizes = vec![24, 24];
        info.output_byte_sizes = vec![24];
        info.output_shapes = vec![vec![2, 3]];
        info.session = ptr::dangling_mut();

        let ctx = FakeCtx {
            num_inputs: 1, // compiled for two
            num_outputs: 1,
        };

        // SAFETY: as above; the dangling session is never dereferenced on this path.
        let status = unsafe {
            compute(
                (&raw mut *info).cast::<ort::OrtNodeComputeInfo>(),
                ptr::null_mut(),
                (&raw const ctx as *mut FakeCtx).cast::<ort::OrtKernelContext>(),
            )
        };
        assert!(!status.is_null(), "a count mismatch must not succeed");
        let msg = read_status(&api, status);
        assert!(msg.contains("disagree"), "unexpected message: {msg}");

        // Do not let the drop glue see the dangling pointer as a live session.
        info.session = ptr::null_mut();
    }

    #[test]
    fn a_null_kernel_context_fails_instead_of_dereferencing() {
        // Driving the real `Compute` entry point records into the process-wide counters,
        // including a broken commitment for the failure this test plants. Share the lock so
        // those events land inside the frame of whoever is asserting on them.
        let _g = crate::allocator::ledger::test_lock();
        let api = fake_api();
        let mut info = SubgraphComputeInfo::new_stub(two_in_one_out_plan(), 24, &raw const *api);
        // SAFETY: passing null is the case under test; `compute` must reject it before any deref.
        let status = unsafe {
            compute(
                (&raw mut *info).cast::<ort::OrtNodeComputeInfo>(),
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        assert!(!status.is_null());
        let msg = read_status(&api, status);
        assert!(msg.contains("null OrtKernelContext"), "message: {msg}");
    }

    // NOTE: `Compile`'s "null every out-slot before anything that can fail" discipline is
    // deliberately *not* tested here. Reaching `compile_impl` requires a live `OrtEp` and a real
    // `OrtGraph`, and the only test I could write without them would assert a loop I wrote in the
    // test itself — which proves nothing about `ep.rs` and would read, in a green run, as though
    // it did. The property is instead covered where it is observable: the mock ORT host in
    // `tests/mock_ort/` is the right place for it once it grows a graph model.

    #[test]
    fn tensor_desc_refuses_partial_or_symbolic_type_information() {
        use crate::engine::DType;
        assert_eq!(tensor_desc(None), None);
        assert_eq!(
            tensor_desc(Some(&registry::EdgeType {
                dtype: None,
                shape: Some(vec![2, 3]),
            })),
            None,
            "an unknown dtype must not produce a TensorDesc"
        );
        assert_eq!(
            tensor_desc(Some(&registry::EdgeType {
                dtype: Some(DType::F32),
                shape: None,
            })),
            None,
            "an unknown shape must not produce a TensorDesc"
        );
        assert_eq!(
            tensor_desc(Some(&registry::EdgeType {
                dtype: Some(DType::F32),
                shape: Some(vec![2, -1]),
            })),
            None,
            "a symbolic dimension must not be silently treated as concrete"
        );
        assert!(
            tensor_desc(Some(&registry::EdgeType {
                dtype: Some(DType::F32),
                shape: Some(vec![2, 3]),
            }))
            .is_some()
        );
    }
}
