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

use std::collections::BTreeMap;
use std::ffi::{CStr, CString, c_char, c_void};
use std::ptr;

use crate::logging;
use crate::registry::{self, NodeView};
use crate::sys::{self, ort};

/// Session options this EP understands, all prefixed `ep.` (`DESIGN.md` §2.4).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EpOptions {
    /// `ep.device_index` — which advertised Vulkan device to bind. `None` = auto (best score).
    pub device_index: Option<usize>,
    /// `ep.enable_validation` — enable `VK_LAYER_KHRONOS_validation`.
    pub enable_validation: bool,
    /// `ep.pipeline_cache_path` — on-disk `VkPipelineCache` blob location.
    pub pipeline_cache_path: Option<String>,
    /// `ep.max_claim_ops` — comma-separated allowlist restricting claiming. Debugging only.
    pub max_claim_ops: Option<Vec<String>>,
    /// `ep.disable_device_memory` — force the host-memory I/O path.
    pub disable_device_memory: bool,
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

        // SAFETY: `get` is ORT's config accessor, used here with the documented two-call pattern
        // (size query, then fill). See `read_config_entry`.
        let read = |key: &str| -> Option<String> { unsafe { read_config_entry(api, get, options, key) } };

        if let Some(v) = read("ep.device_index") {
            match v.trim().parse::<usize>() {
                Ok(i) => out.device_index = Some(i),
                Err(_) => log::warn!("ep.device_index: `{v}` is not a non-negative integer; ignoring"),
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

type ConfigEntryFn =
    unsafe extern "C" fn(*const ort::OrtSessionOptions, *const c_char, *mut c_char, *mut usize) -> ort::OrtStatusPtr;

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
        let status = get(options, c_key.as_ptr(), buf.as_mut_ptr().cast::<c_char>(), &mut written);
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
    ) -> Box<VulkanEp> {
        // SAFETY: `ort_api` is live per the caller's contract; `session_options` may be null,
        // which `from_session_options` handles.
        let options = unsafe { EpOptions::from_session_options(ort_api, session_options) };

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

        log::info!(
            "VulkanExecutionProvider session created (options: {options:?}); M0 build claims no \
             nodes — the whole graph runs on the CPU EP"
        );

        Box::new(VulkanEp {
            base,
            ort_api,
            ep_api,
            abi_version,
            name: name.to_owned(),
            options,
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

    // --- fuse ---
    // Claimed nodes are grouped into maximal convex connected clusters before fusing (a
    // non-convex fusion creates a cycle ORT rejects). With an empty registry there is nothing to
    // cluster, so the clustering pass lands together with the first claimable op; until then, one
    // cluster per claimed node is trivially convex and correct.
    //
    // TODO(mouse/tank, M1): port the union-find + reachability-bitset clustering from the
    // reference EP and replace this per-node grouping.
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
        for node in &claimed {
            let mut opts: ort::OrtNodeFusionOptions = std::mem::zeroed();
            opts.ort_version_supported = ep.abi_version();
            // We read constant initializers at Compile time from the fused node's inputs, so ORT
            // must keep supplying them.
            opts.drop_constant_initializers = false;
            let st = add_nodes_to_fuse(support, node, 1, &opts);
            if !st.is_null() {
                return st;
            }
        }
    }

    ptr::null_mut()
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
    _graphs: *mut *const ort::OrtGraph,
    _fused_nodes: *mut *const ort::OrtNode,
    count: usize,
    node_compute_infos: *mut *mut ort::OrtNodeComputeInfo,
    _ep_context_nodes: *mut *mut ort::OrtNode,
) -> ort::OrtStatusPtr {
    // SAFETY: `p` is our EP pointer.
    let api = unsafe { this(p).ort_api };

    // Leave every out-slot null before returning an error so ORT never reads an uninitialised
    // pointer, and so `ReleaseNodeComputeInfos` has nothing to free.
    if !node_compute_infos.is_null() {
        for i in 0..count {
            // SAFETY: ORT guarantees `node_compute_infos` has room for `count` entries.
            unsafe { *node_compute_infos.add(i) = ptr::null_mut() };
        }
    }

    // Unreachable while the registry is empty: ORT only calls Compile for subgraphs we claimed in
    // GetCapability, and we claim nothing. Being explicit beats a silent success that would hand
    // ORT a null compute-info and fail later, somewhere less obvious.
    log::error!(
        "Compile was called for {count} fused subgraph(s), but the M0 build claims no nodes — \
         this means GetCapability and Compile disagree, which is an internal invariant violation."
    );
    // SAFETY: `api` is live.
    unsafe {
        sys::make_status(
            api,
            ort::OrtErrorCode_ORT_NOT_IMPLEMENTED,
            "VulkanExecutionProvider: subgraph compilation is not implemented yet (M0). The op \
             registry is empty, so no subgraph should ever have been assigned to this EP; ORT \
             will fall the subgraph back to the CPU EP.",
        )
    }
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
        // length. Every non-null entry was produced by `Box::into_raw` in `Compile` (none are,
        // today) and is released exactly once here.
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
// Per-subgraph compute info (shape only in M0)
// -------------------------------------------------------------------------------------------

/// One per fused subgraph; owns the `Plan` and, once Switch lands it, the recorded command
/// buffers and prepacked weights. `OrtNodeComputeInfo` is first so ORT's pointer is this pointer.
///
/// **STUB:** never constructed in M0 (nothing is claimed, so nothing is compiled). The vtable is
/// wired so that the plan builder is the only thing missing.
#[repr(C)]
struct SubgraphComputeInfo {
    base: ort::OrtNodeComputeInfo,
    plan: crate::engine::Plan,
}

impl SubgraphComputeInfo {
    #[allow(dead_code)] // Constructed once Compile builds plans (M0+, Mouse/Switch).
    fn new(plan: crate::engine::Plan, abi_version: u32) -> Box<SubgraphComputeInfo> {
        // SAFETY: `OrtNodeComputeInfo` is a `#[repr(C)]` POD of a `u32` and three `Option<fn>`
        // slots; all-zero is the valid `None` niche for each.
        let mut base: ort::OrtNodeComputeInfo = unsafe { std::mem::zeroed() };
        // The negotiated version, not the compiled-against one — same rule as `OrtEp` and
        // `OrtEpFactory`. Pass `VulkanEp::abi_version()` here.
        base.ort_version_supported = abi_version;
        base.CreateState = Some(create_state);
        base.Compute = Some(compute);
        base.ReleaseState = Some(release_state);
        Box::new(SubgraphComputeInfo { base, plan })
    }
}

unsafe extern "C" fn create_state(
    _this: *mut ort::OrtNodeComputeInfo,
    _context: *mut ort::OrtNodeComputeContext,
    compute_state: *mut *mut c_void,
) -> ort::OrtStatusPtr {
    if !compute_state.is_null() {
        // SAFETY: valid out-param slot supplied by ORT. M0 keeps no per-run state.
        unsafe { *compute_state = ptr::null_mut() };
    }
    ptr::null_mut()
}

unsafe extern "C" fn release_state(_this: *mut ort::OrtNodeComputeInfo, _state: *mut c_void) {}

unsafe extern "C" fn compute(
    _this: *mut ort::OrtNodeComputeInfo,
    _state: *mut c_void,
    _kernel_context: *mut ort::OrtKernelContext,
) -> ort::OrtStatusPtr {
    // Unreachable in M0 (no plan is ever compiled). Deliberately returns null rather than
    // fabricating a status: we have no `OrtApi` handle here, and returning a bogus non-null
    // pointer would be far worse than a no-op.
    log::error!("Compute called on an M0 build that compiles no subgraphs — internal error");
    ptr::null_mut()
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
