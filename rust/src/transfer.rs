//! `OrtDataTransferImpl` — moving tensor bytes between the host and our device handles.
//!
//! # Why this module has to exist at all
//!
//! Advertising a device allocator to ORT is a package deal. The moment `GetSupportedDevices`
//! attaches an allocator memory info to an `OrtEpDevice`, ORT will place tensors in that memory —
//! and it then *requires* a data transfer to get them in and out. Without one, every session fails
//! at `Run` with:
//!
//! ```text
//! There's no data transfer registered for copying tensors from
//!   Device:[DeviceType:0 ... Alignment:0] to Device:[DeviceType:1 ... Alignment:4096]
//! ```
//!
//! That error is why device memory shipped opt-in and inert. This module is the other half.
//!
//! # What a copy actually does here
//!
//! ORT hands us pairs of `OrtValue`s and asks us to copy. For each pair we ask ORT for the tensor's
//! data pointer and byte size, and then classify each side:
//!
//! * a pointer inside one of our reservations is a **handle** — [`HandleRegistry::resolve`] turns
//!   it (including an interior pointer ORT's memory-pattern planner produced by `base + offset`)
//!   into a span and an offset;
//! * anything else is ordinary host memory.
//!
//! Host→device, device→host and device→device are then all the same operation over
//! `(backing pointer, offset, length)`.
//!
//! The "backing pointer" is where the engine layer plugs in. When a `VkBuffer` has been attached to
//! the handle, the copy belongs to the engine. Until then the registry supplies host staging, which
//! is the near half of the real path — a CPU→device copy goes through host-visible staging memory
//! anyway — and which makes every session *correct* today rather than failing at `Run`.
//!
//! # The discipline this file is under
//!
//! `CopyTensors` is called by ORT on the hot path with real tensors. A panic here would cross the
//! FFI boundary into C++; every entry point is wrapped, every `unsafe` block states its invariant,
//! and every failure degrades to an `OrtStatus` describing what disagreed. A copy that cannot be
//! performed must never be silently skipped: a skipped copy is a wrong answer, and a wrong answer
//! from an execution provider is indistinguishable from a wrong model.

use std::collections::HashMap;
use std::ptr;
use std::sync::Arc;

use crate::allocator::{HandleRegistry, LookupError, ledger, tally};
use crate::sys::{self, ort};

/// Sanity marker, checked before we ever dereference a `this_ptr` ORT hands back.
///
/// Same reasoning as the allocator's: if ORT ever routes a foreign `OrtDataTransferImpl` here, this
/// turns a memory-safety incident into a log line.
const TRANSFER_MAGIC: u64 = 0x564B_5852_414E_5346; // "VKXRANSF"

/// Copies performed, and how many of them addressed a handle at a non-zero offset.
///
/// The second number is the evidence for the claim this whole design rests on: that ORT's
/// memory-pattern planner really does hand back `base + offset` and that our range lookup
/// interprets it correctly. It is counted rather than asserted because the honest report of a run
/// where it never happened is "the planner did not exercise it here", not silence.
static COPIES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static INTERIOR_COPIES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static COPIED_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `(copies, copies addressing a handle at a non-zero offset, bytes moved)`.
pub fn copy_counters() -> (u64, u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (
        COPIES.load(Relaxed),
        INTERIOR_COPIES.load(Relaxed),
        COPIED_BYTES.load(Relaxed),
    )
}

/// One side of a copy, after classification.
#[derive(Debug, Clone, Copy)]
enum Side {
    /// Ordinary host memory: the pointer is the bytes.
    Host(*mut u8),
    /// One of our handles: the bytes live behind the registry, `offset` into the span.
    Device {
        base: usize,
        offset: usize,
        span_size: usize,
        has_device_buffer: bool,
        /// The engine's token for the device buffer, when there is one.
        buffer: Option<crate::engine::BufferView>,
        /// Which device's provider owns `buffer`. `usize::MAX` when unattributed.
        device_index: usize,
        /// `true` when a dispatch wrote the device buffer directly and the staging block is
        /// stale. See [`crate::allocator::Span::device_authoritative`].
        device_authoritative: bool,
    },
}

/// Our `OrtDataTransferImpl`.
///
/// `base` is first and the struct is `#[repr(C)]`, so `*mut VulkanDataTransfer` and
/// `*mut OrtDataTransferImpl` are the same address — the same vtable-embedding shape as
/// `VulkanEpFactory`, `VulkanEp` and `VulkanAllocator`.
#[repr(C)]
pub struct VulkanDataTransfer {
    base: ort::OrtDataTransferImpl,
    magic: u64,
    /// Every device's registry, keyed by `(vendor_id, device_id)` — which is all an
    /// `OrtMemoryDevice` will tell us about the side of a copy.
    registries: HashMap<(u32, u32), Arc<HandleRegistry>>,
    ort_api: *const ort::OrtApi,
    ep_api: *const ort::OrtEpApi,
}

impl VulkanDataTransfer {
    /// Build a data transfer over the given registries.
    ///
    /// # Safety
    /// `ort_api` and `ep_api` must be the negotiated tables and must outlive the returned object
    /// (they do: ORT's tables are static for the process). `abi_version` must be the negotiated
    /// version, not our compiled-against one — it tells ORT how far into this vtable it may read,
    /// and stamping it higher than the slots we fill invites a read past our initialised memory.
    pub unsafe fn new(
        ort_api: *const ort::OrtApi,
        ep_api: *const ort::OrtEpApi,
        abi_version: u32,
        registries: HashMap<(u32, u32), Arc<HandleRegistry>>,
    ) -> *mut VulkanDataTransfer {
        // SAFETY: `OrtDataTransferImpl` is a plain C struct of a version field and function
        // pointers; an all-zero value is a valid "nothing implemented" state, which we then fill.
        let mut base: ort::OrtDataTransferImpl = unsafe { std::mem::zeroed() };
        base.ort_version_supported = abi_version;
        base.Release = Some(release_thunk);
        base.CanCopy = Some(can_copy_thunk);
        base.CopyTensors = Some(copy_tensors_thunk);

        Box::into_raw(Box::new(VulkanDataTransfer {
            base,
            magic: TRANSFER_MAGIC,
            registries,
            ort_api,
            ep_api,
        }))
    }

    /// Recover `&VulkanDataTransfer` from the pointer ORT hands back.
    ///
    /// # Safety
    /// `p` must be a pointer this module produced, or null.
    unsafe fn from_ort<'a>(p: *const ort::OrtDataTransferImpl) -> Option<&'a VulkanDataTransfer> {
        if p.is_null() {
            return None;
        }
        let me = p.cast::<VulkanDataTransfer>();
        // SAFETY: `base` is the first field of a `#[repr(C)]` struct, so the two pointers share an
        // address. Reading `magic` before anything else is what makes the cast checkable at all —
        // if ORT routed a foreign implementation here we find out now rather than by corruption.
        let magic = unsafe { (*me).magic };
        if magic != TRANSFER_MAGIC {
            log::error!(
                "VulkanExecutionProvider: OrtDataTransferImpl at {p:?} is not ours (magic \
                 0x{magic:x}). Refusing to touch it."
            );
            return None;
        }
        // SAFETY: the magic marker held, so this is one of our boxes and it is still alive — ORT
        // does not call a data transfer after `Release`.
        Some(unsafe { &*me })
    }

    /// Which registry, if any, owns a given `OrtMemoryDevice`.
    ///
    /// # Safety
    /// `dev` must be null or a live `OrtMemoryDevice` from ORT.
    unsafe fn registry_for(
        &self,
        dev: *const ort::OrtMemoryDevice,
    ) -> Option<&Arc<HandleRegistry>> {
        if dev.is_null() {
            return None;
        }
        // SAFETY: `ep_api` is live for the process; `dev` is a live memory device from ORT. Both
        // accessors are infallible getters, but they are optional slots, so they are checked.
        let (vendor, device) = unsafe {
            let (Some(get_vendor), Some(get_device)) = (
                (*self.ep_api).MemoryDevice_GetVendorId,
                (*self.ep_api).MemoryDevice_GetDeviceId,
            ) else {
                return None;
            };
            (get_vendor(dev), get_device(dev))
        };
        self.registries.get(&(vendor, device))
    }

    /// True when this memory device is one we allocate for.
    ///
    /// # Safety
    /// As [`Self::registry_for`].
    unsafe fn is_ours(&self, dev: *const ort::OrtMemoryDevice) -> bool {
        // SAFETY: forwarded to `registry_for`, whose contract this function repeats.
        unsafe { self.registry_for(dev).is_some() }
    }
}

/// Classify one endpoint of a copy: host memory, or a handle in one of our registries.
///
/// The pointer is looked up in *every* registry rather than only the one the `OrtMemoryDevice`
/// names, because a mislabelled side is exactly the bug this would otherwise hide: a handle that
/// ORT believes is host memory would be `memcpy`d from, and a reserved page is unreadable, so the
/// process would die with no explanation. Finding it here produces a status instead.
fn classify(registries: &HashMap<(u32, u32), Arc<HandleRegistry>>, p: *mut u8) -> Side {
    let addr = p as usize;
    for reg in registries.values() {
        // `classify`, not `resolve`: a miss here is the expected answer for every host pointer, so
        // counting it would make the allocator's failed-lookup diagnostic non-zero on a healthy
        // run.
        let outcome = reg.classify(addr);
        // Every pointer ORT hands back crosses this line, which makes it the one place that can
        // answer "what does the planner actually do with our handles?" with a measurement.
        // Recorded before the match so a host pointer counts as an observation too — otherwise the
        // ledger's denominator would only contain the answers we like.
        ledger::observe(addr, &outcome);
        match outcome {
            Ok(r) => {
                return Side::Device {
                    base: r.base,
                    offset: r.offset,
                    span_size: r.size,
                    has_device_buffer: r.buffer.is_some(),
                    buffer: r.buffer,
                    device_index: reg.device_index(),
                    device_authoritative: r.device_authoritative,
                };
            }
            // In the arena but between spans: a real out-of-bounds, and worth naming here rather
            // than letting the caller treat it as host memory.
            Err(e @ LookupError::InGuardBand { .. }) | Err(e @ LookupError::Freed { .. }) => {
                log::error!("VulkanExecutionProvider: data transfer endpoint is unusable: {e}");
                return Side::Device {
                    base: 0,
                    offset: 0,
                    span_size: 0,
                    has_device_buffer: false,
                    buffer: None,
                    device_index: usize::MAX,
                    device_authoritative: false,
                };
            }
            Err(LookupError::NotAHandle { .. }) => {}
        }
    }
    Side::Host(p)
}

/// One endpoint of a copy, resolved to something that can actually be read or written.
///
/// # Why there is no "device only" variant
///
/// A span with a `VkBuffer` **also** keeps its host staging block. If a device-backed handle had
/// no host address, the session would have nothing to read — measured: with device memory on and
/// no host address, ORT reported `EP_FAIL ... bytes are unreachable` for input 1 of the first
/// subgraph and fell back to the CPU EP for the whole model.
///
/// # Which copy is real
///
/// Both exist; **which one is authoritative is a property of the span, not of the file.** This
/// used to be a global rule — staging always authoritative, device always a mirror — and the
/// rule is wrong in one direction each way round:
///
/// * reading the **device** buffer of a staging-authoritative span returns a copy the engine may
///   have made stale by writing an output through [`host_backing_for`];
/// * reading the **staging** block of a device-authoritative span returns bytes nobody wrote —
///   this is exactly the all-zero output measured on 2026-08-02 when the engine began binding
///   output buffers directly (`bench/results/probe_kv_device_residency.py` ->
///   `DEVICE_BOUND_OUTPUTS_RETURN_NOTHING`).
///
/// So [`Endpoint::Mirrored`] carries `device_authoritative` and every reader consults it. A span
/// becomes device-authoritative when `vk::session` binds its buffer as a dispatch's write target,
/// and reverts the moment anything writes its staging block, because after a host write the two
/// copies agree only once the mirror is refreshed — and refreshing the mirror is the path that
/// already existed.
#[derive(Debug, Clone, Copy)]
enum Endpoint {
    /// Host-addressable bytes with no device mirror.
    Host(*mut u8),
    /// Host-addressable staging that is mirrored into device memory.
    Mirrored {
        base: usize,
        host: *mut u8,
        view: crate::engine::BufferView,
        offset: usize,
        device_index: usize,
        /// `true` when the device buffer is the real copy and `host` is stale.
        device_authoritative: bool,
        /// Requested size of the whole containing span, not of this endpoint's window.
        ///
        /// Carried because authority is a property of the span while a copy is a property of a
        /// window inside it, and the difference is only visible when the two disagree — a
        /// distinction that is invisible on every whole-span write and therefore invisible to the
        /// tests that exercise one.
        span_size: usize,
    },
}

impl Endpoint {
    fn host_ptr(self) -> *mut u8 {
        match self {
            Endpoint::Host(p) => p,
            Endpoint::Mirrored { host, .. } => host,
        }
    }
}

/// Resolve one endpoint of a copy.
///
/// Bounds are enforced here against the span's *requested* size, so a device-backed span is no
/// more permissive than a staged one.
fn resolve_endpoint(
    registries: &HashMap<(u32, u32), Arc<HandleRegistry>>,
    side: Side,
    len: usize,
) -> Result<Endpoint, String> {
    match side {
        Side::Host(p) => {
            if p.is_null() && len != 0 {
                return Err("a host endpoint of the copy is a null pointer".to_string());
            }
            Ok(Endpoint::Host(p))
        }
        Side::Device { span_size: 0, .. } => Err(
            "the endpoint resolved into the handle arena but not to a live span — see the \
                 preceding log line for which kind of bad pointer it was"
                .to_string(),
        ),
        Side::Device {
            base,
            offset,
            span_size,
            has_device_buffer,
            buffer,
            device_index,
            device_authoritative,
            ..
        } => {
            // Bound the copy by the *requested* size of the span, not the padded one. This is the
            // same rule the registry's lookups use and for the same reason: accepting the rounding
            // slack would permit a copy that runs past the end of the tensor ORT actually asked
            // for.
            let available = span_size.saturating_sub(offset);
            if len > available {
                return Err(format!(
                    "the copy is {len} byte(s) but only {available} remain in device handle \
                     0x{base:x} from offset {offset} (span is {span_size} byte(s)). Refusing: this \
                     would read or write past the end of the tensor."
                ));
            }
            if has_device_buffer && buffer.is_none() {
                return Err(format!(
                    "device handle 0x{base:x} reports a device buffer but did not yield one"
                ));
            }
            let Some(reg) = registries
                .values()
                .find(|r| r.classify(base).is_ok_and(|r| r.base == base))
            else {
                return Err(format!(
                    "device handle 0x{base:x} vanished from its registry between classification \
                     and use"
                ));
            };
            let Some(p) = reg.staging_ptr(base) else {
                return Err(format!(
                    "could not obtain backing memory for device handle 0x{base:x}"
                ));
            };
            // SAFETY: `staging_ptr` returns the base of an allocation of the span's padded size,
            // and `offset + len <= span_size <= padded`, checked immediately above.
            let host = unsafe { p.add(offset) };
            match buffer {
                Some(view) if device_index != usize::MAX => Ok(Endpoint::Mirrored {
                    base,
                    host,
                    view,
                    offset,
                    device_index,
                    device_authoritative,
                    span_size,
                }),
                _ => {
                    if device_authoritative {
                        // A span that claims the device is authoritative but resolves to a
                        // host-only endpoint has nowhere real to be read from. Silently
                        // degrading to the stale staging block is precisely the all-zero
                        // failure this flag exists to prevent, so it is refused instead.
                        return Err(format!(
                            "device handle 0x{base:x} is marked device-authoritative but has no \
                             usable buffer/provider pair (buffer={}, device_index={device_index}); \
                             reading its staging block would return bytes nobody wrote",
                            buffer.is_some()
                        ));
                    }
                    Ok(Endpoint::Host(host))
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Thunks
// ─────────────────────────────────────────────────────────────────────────────────────────────

unsafe extern "C" fn release_thunk(p: *mut ort::OrtDataTransferImpl) {
    crate::guard_ffi_void(|| {
        if p.is_null() {
            return;
        }
        // SAFETY: verify ownership before reclaiming the box — releasing a foreign pointer would
        // be far worse than leaking ours.
        if unsafe { VulkanDataTransfer::from_ort(p) }.is_none() {
            return;
        }
        // SAFETY: the magic check passed, so this box came from `VulkanDataTransfer::new` and ORT
        // is handing back sole ownership. It will not call the object again.
        drop(unsafe { Box::from_raw(p.cast::<VulkanDataTransfer>()) });
        let (n, interior, bytes) = copy_counters();
        log::info!(
            "VulkanExecutionProvider: releasing data transfer — {n} copy(ies), {bytes} B moved, \
             {interior} of them at a non-zero offset into a handle{}",
            if n > 0 && interior == 0 {
                " (ORT's planner never exercised interior addressing in this run)"
            } else {
                ""
            }
        );
        log::info!("VulkanExecutionProvider: {}", ledger::report());
        log::info!(
            "VulkanExecutionProvider: {}",
            crate::allocator::tally::staging_verdict()
        );
        // The observations are only complete now, and a torn-down process cannot print into a
        // test harness's captured output. Persist them where a parent process can read them.
        crate::counters::dump_observations_if_requested();
    });
}

unsafe extern "C" fn can_copy_thunk(
    p: *const ort::OrtDataTransferImpl,
    src: *const ort::OrtMemoryDevice,
    dst: *const ort::OrtMemoryDevice,
) -> bool {
    crate::guard_ffi_bool("OrtDataTransferImpl::CanCopy", false, || {
        // SAFETY: `p` is the pointer ORT received from `CreateDataTransfer`; the magic check
        // inside verifies it before any other field is read.
        let Some(me) = (unsafe { VulkanDataTransfer::from_ort(p) }) else {
            return false;
        };
        // SAFETY: `src`/`dst` are live memory devices belonging to this copy request.
        let (src_ours, dst_ours) = unsafe { (me.is_ours(src), me.is_ours(dst)) };
        // Claim only copies with at least one end in our memory. Claiming host→host would take
        // work away from ORT that it does better, and claiming another device's memory would be a
        // straightforward lie.
        src_ours || dst_ours
    })
}

unsafe extern "C" fn copy_tensors_thunk(
    p: *mut ort::OrtDataTransferImpl,
    src_tensors: *mut *const ort::OrtValue,
    dst_tensors: *mut *mut ort::OrtValue,
    streams: *mut *mut ort::OrtSyncStream,
    num_tensors: usize,
) -> ort::OrtStatusPtr {
    let _ = streams;
    let api = if p.is_null() {
        ptr::null()
    } else {
        // SAFETY: `p` is non-null and, if it is ours, `ort_api` is the live negotiated table. The
        // magic check inside `from_ort` runs before any other field is trusted.
        unsafe { VulkanDataTransfer::from_ort(p) }.map_or(ptr::null(), |me| me.ort_api)
    };
    // SAFETY: `api` is null or the live negotiated table, per the lines immediately above; the
    // body upholds its own raw-pointer invariants at each use.
    unsafe {
        crate::guard_ffi_status(api, "OrtDataTransferImpl::CopyTensors", || {
            copy_tensors_impl(p, src_tensors, dst_tensors, num_tensors)
        })
    }
}

/// The body of `CopyTensors`, outside the panic guard so its `unsafe` blocks stay individually
/// justified rather than swallowed by one enclosing `unsafe` around the whole closure.
///
/// # Safety
/// `p` must be null or an `OrtDataTransferImpl` this module created, and the two arrays must hold
/// `num_tensors` valid entries for the duration of the call.
unsafe fn copy_tensors_impl(
    p: *mut ort::OrtDataTransferImpl,
    src_tensors: *mut *const ort::OrtValue,
    dst_tensors: *mut *mut ort::OrtValue,
    num_tensors: usize,
) -> ort::OrtStatusPtr {
    // SAFETY: as `can_copy_thunk` — the magic check runs before any other field is trusted.
    let Some(me) = (unsafe { VulkanDataTransfer::from_ort(p) }) else {
        // SAFETY: `make_status` accepts a null API and falls back to a plain status.
        return unsafe {
            sys::make_status(
                ptr::null(),
                ort::OrtErrorCode_ORT_INVALID_ARGUMENT,
                "CopyTensors called on an OrtDataTransferImpl this EP did not create",
            )
        };
    };
    if num_tensors == 0 {
        return ptr::null_mut();
    }
    if src_tensors.is_null() || dst_tensors.is_null() {
        // SAFETY: `ort_api` is the live negotiated table.
        return unsafe {
            sys::make_status(
                me.ort_api,
                ort::OrtErrorCode_ORT_INVALID_ARGUMENT,
                "CopyTensors received a null tensor array with a non-zero count",
            )
        };
    }

    // SAFETY: ORT guarantees both arrays hold `num_tensors` entries for the call's duration.
    let (srcs, dsts) = unsafe {
        (
            std::slice::from_raw_parts(src_tensors, num_tensors),
            std::slice::from_raw_parts(dst_tensors, num_tensors),
        )
    };

    for (i, (&s, &d)) in srcs.iter().zip(dsts.iter()).enumerate() {
        // SAFETY: `ort_api` is live and `s`/`d` are the `OrtValue`s ORT supplied.
        let outcome = unsafe { copy_one(me, s, d) };
        if let Err(msg) = outcome {
            // SAFETY: `ort_api` is live.
            return unsafe {
                sys::make_status(
                    me.ort_api,
                    ort::OrtErrorCode_ORT_FAIL,
                    &format!("VulkanExecutionProvider: copy {i}/{num_tensors} failed: {msg}"),
                )
            };
        }
    }
    ptr::null_mut()
}

/// Copy one tensor. Returns a prose reason on failure; the caller turns it into an `OrtStatus`.
///
/// # Safety
/// `me.ort_api` must be live, and `src`/`dst` must be the `OrtValue`s ORT supplied for this copy.
unsafe fn copy_one(
    me: &VulkanDataTransfer,
    src: *const ort::OrtValue,
    dst: *mut ort::OrtValue,
) -> Result<(), String> {
    if src.is_null() || dst.is_null() {
        return Err("a null OrtValue was supplied for one side of the copy".to_string());
    }
    let api = me.ort_api;

    // SAFETY: `api` is the live negotiated table; both slots are optional and are checked.
    let (Some(get_data), Some(size_of)) = (unsafe { (*api).GetTensorMutableData }, unsafe {
        (*api).GetTensorSizeInBytes
    }) else {
        return Err(
            "the negotiated ORT ABI lacks GetTensorMutableData or GetTensorSizeInBytes, \
                    so a copy cannot be performed safely"
                .to_string(),
        );
    };

    let mut src_len: usize = 0;
    // SAFETY: `src` is a live tensor value; `src_len` is a valid out-param slot.
    let st = unsafe { size_of(src, &mut src_len) };
    if !st.is_null() {
        // SAFETY: `st` is a status this call produced and is released exactly once.
        let msg = unsafe { sys::status_message(api, st) };
        // SAFETY: as above.
        unsafe { sys::release_status(api, st) };
        return Err(format!("could not size the source tensor: {msg}"));
    }
    let mut dst_len: usize = 0;
    // SAFETY: as for the source.
    let st = unsafe { size_of(dst.cast_const(), &mut dst_len) };
    if !st.is_null() {
        // SAFETY: as above.
        let msg = unsafe { sys::status_message(api, st) };
        // SAFETY: as above.
        unsafe { sys::release_status(api, st) };
        return Err(format!("could not size the destination tensor: {msg}"));
    }
    if src_len != dst_len {
        return Err(format!(
            "source is {src_len} byte(s) and destination is {dst_len} — a data transfer must not \
             silently truncate or over-read"
        ));
    }
    if src_len == 0 {
        return Ok(());
    }

    let mut src_p: *mut std::ffi::c_void = ptr::null_mut();
    // SAFETY: `src` is live; `src_p` is a valid out-param slot. `GetTensorMutableData` on a source
    // value is ORT's own idiom here — the data transfer API has no const accessor.
    let st = unsafe { get_data(src.cast_mut(), &mut src_p) };
    if !st.is_null() {
        // SAFETY: as above.
        let msg = unsafe { sys::status_message(api, st) };
        // SAFETY: as above.
        unsafe { sys::release_status(api, st) };
        return Err(format!("could not get the source data pointer: {msg}"));
    }
    let mut dst_p: *mut std::ffi::c_void = ptr::null_mut();
    // SAFETY: as for the source.
    let st = unsafe { get_data(dst, &mut dst_p) };
    if !st.is_null() {
        // SAFETY: as above.
        let msg = unsafe { sys::status_message(api, st) };
        // SAFETY: as above.
        unsafe { sys::release_status(api, st) };
        return Err(format!("could not get the destination data pointer: {msg}"));
    }

    let src_side = classify(&me.registries, src_p.cast::<u8>());
    let dst_side = classify(&me.registries, dst_p.cast::<u8>());
    let interior = matches!(src_side, Side::Device { offset, .. } if offset != 0)
        || matches!(dst_side, Side::Device { offset, .. } if offset != 0);
    let from = resolve_endpoint(&me.registries, src_side, src_len)?;
    let to = resolve_endpoint(&me.registries, dst_side, dst_len)?;

    use std::sync::atomic::Ordering::Relaxed;
    COPIES.fetch_add(1, Relaxed);
    COPIED_BYTES.fetch_add(src_len as u64, Relaxed);
    log::debug!("[VulkanEP] CopyTensors: {src_len} B {src_side:?} -> {dst_len} B {dst_side:?}");
    if interior {
        // Worth a line the first time: this is ORT's planner doing pointer arithmetic on a handle
        // and our range lookup resolving it, which is the property the whole scheme exists for.
        if INTERIOR_COPIES.fetch_add(1, Relaxed) == 0 {
            log::info!(
                "VulkanExecutionProvider: ORT addressed a device handle at a non-zero offset \
                 ({src_side:?} -> {dst_side:?}). The memory-pattern planner's `base + offset` \
                 resolved to a span by range lookup, which is what reserved address space buys."
            );
        }
    }

    // ── Refresh a stale staging block before reading it ────────────────────────────────────
    //
    // This is the fix for the all-zero output bind. When a dispatch has written the source's
    // device buffer directly, the staging block underneath `from.host_ptr()` is stale — nobody
    // wrote it — and the memcpy below would copy those bytes out with complete confidence.
    // Measured 2026-08-02: 0 nonzero of 32064 logits and 0 of 27648 in each `present` tensor,
    // scoring a perfect 1.0 against a relative metric because two all-zero tensors agree.
    //
    // The download lands in staging rather than straight into `to_p` on purpose. Staging is the
    // span's own memory and refreshing it makes *every* subsequent reader correct, including the
    // engine's `host_backing_for` path, rather than only this one copy. It also keeps the two
    // copies in agreement, which is the state every other branch in this file assumes.
    refresh_from_device_if_authoritative(from, src_len)?;

    // A partial host write into a device-authoritative span must not revoke authority over bytes
    // it never wrote.
    //
    // Authority is a property of the whole span; a copy is a property of a window inside it. The
    // upload below pushes only `[offset, offset + dst_len)`, but the revoke that follows declares
    // the *entire* span to live in staging — including bytes that only ever existed on the device,
    // because a dispatch wrote them and nothing ever brought them back. A later read of a sibling
    // tensor in the same planner-subdivided span would then return whatever staging happened to
    // hold, with no error anywhere.
    //
    // This is invisible on every whole-span write, which is the common case and the one the
    // residency probe exercises, so it would not have been found by running the case that already
    // passes. Bringing the whole span down first makes staging genuinely authoritative at the
    // moment authority is handed back to it, which is the claim the revoke makes.
    refresh_whole_span_before_partial_write(to, dst_len)?;

    let from_p = from.host_ptr();
    let to_p = to.host_ptr();
    if from_p != to_p {
        // SAFETY: `from_p` and `to_p` each address at least `src_len` readable/writable bytes —
        // for host endpoints because ORT sized the tensor, and for handle endpoints because
        // `resolve_endpoint` rejected any copy extending past the span's requested size. They are
        // distinct allocations: a handle's staging is a private heap block.
        unsafe { ptr::copy_nonoverlapping(from_p, to_p, src_len) };
    }

    // Mirror into device memory, and hand authority back to staging.
    //
    // The destination's staging block has just been written by the memcpy above, so it is now the
    // fresh copy and the device buffer is stale until this upload. Clearing the flag *after* the
    // upload succeeds is deliberate: between the memcpy and the upload the two copies disagree,
    // and the flag says "read the device" for exactly as long as that is still true.
    //
    // Getting the order wrong here is not a subtle inefficiency — it is a read of the wrong copy
    // in whichever direction the mistake points, which is the same defect twice.
    if let Endpoint::Mirrored {
        base,
        host,
        view,
        offset,
        device_index,
        device_authoritative,
        ..
    } = to
    {
        let provider = provider_for(device_index, base)?;
        // SAFETY: `host` addresses at least `dst_len` readable bytes, as above. The slice is used
        // only for this synchronous call and is not retained.
        let src = unsafe { std::slice::from_raw_parts(host.cast_const(), dst_len) };
        provider.upload(view, offset, src)?;
        tally::on_device_copy(dst_len as u64, true);
        if device_authoritative {
            revoke_device_authority(&me.registries, base);
        }
    }
    Ok(())
}

/// Bring a whole device-authoritative span into staging before a host write that covers only part
/// of it.
///
/// A no-op unless the destination is device-authoritative *and* the write leaves some of the span
/// untouched — a whole-span write needs no refresh, because every byte is about to be replaced.
/// See the call site in [`copy_one`] for why the partial case is the dangerous one.
fn refresh_whole_span_before_partial_write(to: Endpoint, dst_len: usize) -> Result<(), String> {
    let Endpoint::Mirrored {
        base,
        host,
        view,
        offset,
        device_index,
        device_authoritative: true,
        span_size,
    } = to
    else {
        return Ok(());
    };
    if offset == 0 && dst_len >= span_size {
        return Ok(());
    }
    if span_size == 0 {
        return Ok(());
    }
    let provider = provider_for(device_index, base)?;
    // SAFETY: `host` is the span's staging base advanced by `offset`, so subtracting `offset`
    // recovers that base, which addresses at least `span_size` writable bytes — `staging_ptr`
    // allocates the span's padded size and `span_size` is the requested size, which is no larger.
    let dst = unsafe { std::slice::from_raw_parts_mut(host.sub(offset), span_size) };
    provider.download(view, 0, dst)?;
    tally::on_device_copy(span_size as u64, false);
    Ok(())
}

/// Hand authority back to a span's staging block after a host write has refreshed it.///
/// A failure here is logged rather than propagated: the copy itself succeeded and both copies of
/// the bytes now agree, so the caller's transfer is complete. What is lost is bookkeeping, and a
/// span left marked device-authoritative when it is not will be *downloaded* before its next
/// read — wasteful, and correct. The opposite mistake would not be, which is why this is the
/// direction the failure mode points.
fn revoke_device_authority(registries: &HashMap<(u32, u32), Arc<HandleRegistry>>, base: usize) {
    for reg in registries.values() {
        if reg.classify(base).is_ok_and(|r| r.base == base) {
            if let Err(e) = reg.set_device_authoritative(base, false) {
                log::warn!(
                    "VulkanExecutionProvider: handle 0x{base:x} was written on the host but its \
                     authority flag could not be cleared ({e}); its next read will download a \
                     device copy that now agrees with staging"
                );
            }
            return;
        }
    }
}

/// Declare that a dispatch is about to write this handle's device buffer directly, making the
/// device buffer the authoritative copy and the staging block stale.
///
/// The engine's seam. `vk::session` calls this when it binds an output buffer as a dispatch
/// target instead of allocating its own and copying back — the change [`Endpoint`] describes as
/// the thing that stops the device buffer being a mirror.
///
/// Returns `false` when the pointer is not a live handle of ours, or when the span has no device
/// buffer to be authoritative about. **A caller must treat `false` as "do not bind"**: binding a
/// buffer whose span still reports staging as authoritative produces a tensor that reads as
/// zeros, which is the failure this whole mechanism exists to remove.
pub fn mark_device_authoritative(p: *mut u8, on: bool) -> bool {
    let addr = p as usize;
    for reg in crate::factory::all_registries().values() {
        match reg.classify(addr) {
            Ok(r) if r.offset == 0 => return reg.set_device_authoritative(r.base, on).is_ok(),
            Ok(r) => {
                // An interior pointer names a span but not its start. Authority is a property of
                // the whole span, so silently applying it from an offset would claim more than
                // the caller asked for. ORT does hand back interior pointers (the planner's
                // `base + offset`), so this is a real case and not a defensive one.
                log::warn!(
                    "VulkanExecutionProvider: refusing to set device authority from an interior \
                     pointer (0x{addr:x} is offset {} into handle 0x{:x}); authority covers a \
                     whole span",
                    r.offset,
                    r.base
                );
                return false;
            }
            Err(_) => {}
        }
    }
    false
}

/// Bring a span's staging block up to date when its device buffer is the authoritative copy.
///
/// One definition, called from every read path, for the reason the rest of this file keeps
/// running into: two implementations of "is this copy stale?" would be two answers, and the
/// disagreement would show up as a tensor that is correct through one door and zeros through the
/// other.
///
/// A no-op for host endpoints and for mirrored spans whose staging is already authoritative, so
/// callers do not have to ask first.
///
/// # This is a real transfer and it is counted as one
///
/// The download is the round trip the binding design exists to remove. It is tallied as device
/// traffic deliberately: a refresh that did not show up in the counters would remove the transfer
/// from the measurement rather than from the machine, and this project has been caught by that
/// shape before. The caller that wants it genuinely gone binds the buffer instead
/// ([`device_buffer_for`]).
fn refresh_from_device_if_authoritative(e: Endpoint, len: usize) -> Result<(), String> {
    let Endpoint::Mirrored {
        base,
        host,
        view,
        offset,
        device_index,
        device_authoritative: true,
        ..
    } = e
    else {
        return Ok(());
    };
    if len == 0 {
        return Ok(());
    }
    let provider = provider_for(device_index, base)?;
    // Name the transfer. This is the round trip the binding design exists to remove, and a caller
    // that is paying for it has no other way to find out *which* span it is paying for: the byte
    // total is the sum over spans and every KV tensor in a decode step is the same size, so a
    // total cannot distinguish "one tensor, sixty-four times" from "sixty-four tensors, once".
    // Debug level, so it costs nothing on a normal run and is one `RUST_LOG=debug` away when a
    // byte count needs attributing.
    log::debug!(
        "[VulkanEP] refresh: downloading {len} B from device-authoritative span 0x{base:x}+{offset} \
         on device {device_index} — a reader asked for host bytes of a span only the device holds"
    );
    // SAFETY: `host` addresses at least `len` writable bytes — `resolve_endpoint` rejected any
    // range extending past the span's requested size. The slice is used only for this
    // synchronous call and is not retained.
    let dst = unsafe { std::slice::from_raw_parts_mut(host, len) };
    provider.download(view, offset, dst)?;
    tally::on_device_copy(len as u64, false);
    Ok(())
}

/// The engine's device-memory provider for `device_index`, or a status-worthy explanation.
fn provider_for(
    device_index: usize,
    base: usize,
) -> Result<std::sync::Arc<dyn crate::engine::DeviceMemoryProvider>, String> {
    crate::engine::device_memory_provider(device_index).ok_or_else(|| {
        format!(
            "device handle 0x{base:x} is backed by device memory on device {device_index}, but \
             the engine layer has no memory provider registered for that device. Refusing rather \
             than copying to the wrong memory."
        )
    })
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Handle resolution for the rest of the crate
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// Resolve a pointer that ORT bound to a kernel, when it may be one of our device handles.
///
/// * `None` — the pointer is ordinary host memory. Read it directly; nothing to do.
/// * `Some(Ok(p))` — it *was* a device handle, and `p` is `len` host-addressable bytes for it.
/// * `Some(Err(why))` — it is a handle but the bytes are not reachable this way, and `why` says so
///   in prose suitable for an `OrtStatus`.
///
/// # A device-authoritative span is refreshed before its address is handed out
///
/// If a dispatch wrote the span's device buffer directly, the staging block this returns is
/// stale, and the caller has no way to know: it receives a plain `*mut u8` that reads as zeros
/// with no error anywhere. So the device copy is downloaded into staging first. That is a real
/// transfer and it is counted as one — it is the round trip this design exists to remove, and
/// hiding it here would remove it from the counters rather than from the machine.
///
/// The caller that wants the transfer *actually* gone should use [`device_buffer_for`] and bind,
/// which is why that function exists and why its doc says to prefer it.
///
/// # Why this exists as a public seam
///
/// Once device memory is advertised, ORT places subgraph inputs and outputs in it, and the pointer
/// `GetTensorMutableData` returns is then a **handle** — a reserved, deliberately unreadable
/// address. Any code that `memcpy`s from it dies instantly, which is the design working as
/// intended, but only if the code that would do that asks here first.
///
/// This is the one function the engine layer needs in order to read and write tensors that live in
/// our device memory, and it is deliberately the *whole* interface: no `sys::ort` type crosses into
/// the engine, and no Vulkan handle crosses out.
pub fn host_backing_for(p: *mut u8, len: usize) -> Option<Result<*mut u8, String>> {
    let registries = crate::factory::all_registries();
    if registries.is_empty() {
        return None;
    }
    match classify(&registries, p) {
        Side::Host(_) => None,
        side => Some(resolve_endpoint(&registries, side, len).and_then(|e| {
            refresh_from_device_if_authoritative(e, len)?;
            Ok(e.host_ptr())
        })),
    }
}

/// The engine's counterpart to [`host_backing_for`]: the device buffer behind a pointer.
///
/// Returns `Some(binding)` when `p` is one of our handles **and** that handle's span has a real
/// `VkBuffer`. The offset is ORT's pointer arithmetic, already resolved by range lookup, and must
/// be applied when binding: the planner sub-divides one span across several tensors, so a buffer
/// bound at offset 0 for an interior pointer would silently read the wrong tensor.
///
/// This function **resolves** and does not count. `alloc_device_buffer_binds` is incremented at the
/// point a buffer is actually bound (`vk::host_device_memory::bind_target_for`), because a resolve
/// that is then declined — wrong device frame, or an offset the descriptor cannot express — is not
/// a bind, and a counter that inflates on the flattering side is the failure mode this project
/// keeps hitting. `authoritative > 0` while binds is 0 remains a contradiction, not a nuance.
///
/// # Why the engine should prefer this to `host_backing_for`
///
/// `vk::session` currently resolves every input to host bytes and then allocates a fresh
/// `DeviceLocal` buffer and re-uploads it on **every** `Compute` call — including weights that
/// never change. That is where the wall-clock goes, and no amount of device-backed allocation on
/// our side removes it, because the session binds its own buffers. This function is the seam that
/// lets the session bind ours instead. Until it is used, device-backed allocation is a
/// precondition and not a speedup, and must not be reported as one.
///
/// No `sys::ort` type crosses into the engine and no Vulkan handle crosses out: a [`BufferView`]
/// is an opaque token only the minting engine can interpret.
///
/// [`BufferView`]: crate::engine::BufferView
pub fn device_buffer_for(p: *mut u8, len: usize) -> Option<DeviceBinding> {
    let registries = crate::factory::all_registries();
    if registries.is_empty() {
        return None;
    }
    match classify(&registries, p) {
        Side::Host(_) => None,
        side => match resolve_endpoint(&registries, side, len) {
            Ok(Endpoint::Mirrored {
                view,
                offset,
                device_index,
                ..
            }) => Some(DeviceBinding {
                device_index,
                view,
                offset,
            }),
            _ => None,
        },
    }
}

/// What [`device_buffer_for`] resolved: which provider's buffer, and where in it.
///
/// The device index is part of the answer because the caller must not bind a buffer that lives on
/// a different `VkDevice` than the one it is about to dispatch on — the §6.5 `SPLIT-DEVICE` case.
/// Returning the view alone would let a caller bind across devices and see it work on a UMA part.
#[derive(Debug, Clone, Copy)]
pub struct DeviceBinding {
    /// The provider that minted the buffer. Check its frame before binding.
    pub device_index: usize,
    /// The opaque token; only the minting engine can interpret it.
    pub view: crate::engine::BufferView,
    /// ORT's pointer arithmetic, resolved by range lookup. Must be honoured when binding.
    pub offset: usize,
}

/// Push a span's host staging bytes into its device mirror, if it has one.
///
/// Returns `Ok(false)` when `p` is not one of our handles or has no device buffer — the ordinary
/// host build, where there is nothing to keep in sync.
///
/// # Why the engine must call this after writing an output
///
/// [`Endpoint`] documents that the staging block is authoritative *because* the session writes
/// outputs through [`host_backing_for`]. The moment the session also **binds** device buffers for
/// inputs, that asymmetry becomes a correctness bug rather than a design note: a span written as
/// an output through staging, then read as an input through its device buffer, would be read
/// stale. `CopyTensors` already mirrors every copy into a handle; this is the same obligation for
/// the one writer that does not go through `CopyTensors`.
pub fn mirror_to_device(p: *mut u8, len: usize) -> Result<bool, String> {
    mirror_in(&crate::factory::all_registries(), p, len)
}

/// [`mirror_to_device`] over an explicit registry map, so it is reachable from a test.
///
/// The public entry point reads `factory::all_registries()`, which in a unit test is empty — so a
/// test that called it would pass by taking the "no registries" early return and would prove
/// nothing. Splitting the map out is what makes the assertion falsifiable.
fn mirror_in(
    registries: &HashMap<(u32, u32), Arc<HandleRegistry>>,
    p: *mut u8,
    len: usize,
) -> Result<bool, String> {
    if registries.is_empty() || len == 0 {
        return Ok(false);
    }
    let side = classify(registries, p);
    if matches!(side, Side::Host(_)) {
        return Ok(false);
    }
    match resolve_endpoint(registries, side, len)? {
        Endpoint::Host(_) => Ok(false),
        Endpoint::Mirrored {
            base,
            host,
            view,
            offset,
            device_index,
            device_authoritative,
            ..
        } => {
            if device_authoritative {
                // This function pushes the staging block *over* the device buffer. On a
                // device-authoritative span that would overwrite a dispatch's output with bytes
                // nobody wrote — the all-zero failure, but destructive and after the fact rather
                // than merely visible at the read. Refused, and named, because a silent `Ok(false)`
                // here would be indistinguishable from "there was nothing to mirror".
                return Err(format!(
                    "refusing to mirror host staging over handle 0x{base:x}: its device buffer is \
                     authoritative, so this upload would overwrite a dispatch's output with a \
                     stale staging block"
                ));
            }
            let provider = provider_for(device_index, base)?;
            // SAFETY: `host` addresses at least `len` readable bytes — `resolve_endpoint` rejected
            // any range extending past the span's requested size. The slice is not retained.
            let src = unsafe { std::slice::from_raw_parts(host.cast_const(), len) };
            provider.upload(view, offset, src)?;
            tally::on_device_copy(len as u64, true);
            Ok(true)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test shim: the old `host_bytes` signature, over the current [`resolve_endpoint`].
    fn host_bytes(
        registries: &HashMap<(u32, u32), Arc<HandleRegistry>>,
        side: Side,
        len: usize,
    ) -> Result<*mut u8, String> {
        resolve_endpoint(registries, side, len).map(Endpoint::host_ptr)
    }

    fn registries() -> HashMap<(u32, u32), Arc<HandleRegistry>> {
        let mut m = HashMap::new();
        m.insert((0x10de, 0), HandleRegistry::new().expect("reservation"));
        m
    }

    #[test]
    fn a_host_pointer_is_not_mistaken_for_a_handle() {
        // `registries()` builds real HandleRegistrys, so every alloc, staging_ptr and free
        // below moves the process-global tally. Same lock as everything that asserts on it.
        let _g = ledger::test_lock();
        let regs = registries();
        let mut buf = [0u8; 32];
        match classify(&regs, buf.as_mut_ptr()) {
            Side::Host(p) => assert_eq!(p, buf.as_mut_ptr()),
            Side::Device { .. } => panic!("a stack buffer was classified as a device handle"),
        }
    }

    #[test]
    fn an_interior_handle_pointer_classifies_to_its_span_and_offset() {
        // `registries()` builds real HandleRegistrys, so every alloc, staging_ptr and free
        // below moves the process-global tally. Same lock as everything that asserts on it.
        let _g = ledger::test_lock();
        let regs = registries();
        let reg = regs.values().next().expect("one registry").clone();
        let h = reg.alloc(4096).expect("alloc");
        match classify(&regs, (h + 100) as *mut u8) {
            Side::Device {
                base,
                offset,
                span_size,
                ..
            } => {
                assert_eq!(base, h);
                assert_eq!(offset, 100);
                assert_eq!(span_size, 4096);
            }
            Side::Host(_) => panic!("an interior handle pointer was classified as host memory"),
        }
    }

    #[test]
    fn a_copy_past_the_end_of_a_span_is_refused_rather_than_clamped() {
        // `registries()` builds real HandleRegistrys, so every alloc, staging_ptr and free
        // below moves the process-global tally. Same lock as everything that asserts on it.
        let _g = ledger::test_lock();
        let regs = registries();
        let reg = regs.values().next().expect("one registry").clone();
        let h = reg.alloc(256).expect("alloc");
        let side = classify(&regs, (h + 200) as *mut u8);
        // 56 bytes remain from offset 200. Asking for 57 must fail, not copy 56.
        let err = host_bytes(&regs, side, 57).expect_err("must refuse");
        assert!(err.contains("past the end"), "unhelpful message: {err}");
        host_bytes(&regs, side, 56).expect("the exact remainder must be allowed");
    }

    #[test]
    fn staging_is_reused_so_a_second_copy_sees_the_first_ones_bytes() {
        // `registries()` builds real HandleRegistrys, so every alloc, staging_ptr and free
        // below moves the process-global tally. Same lock as everything that asserts on it.
        let _g = ledger::test_lock();
        let regs = registries();
        let reg = regs.values().next().expect("one registry").clone();
        let h = reg.alloc(64).expect("alloc");
        let side = classify(&regs, h as *mut u8);
        let a = host_bytes(&regs, side, 64).expect("staging");
        // SAFETY: `a` is 64 writable staging bytes for this handle.
        unsafe { ptr::write_bytes(a, 0xAB, 64) };
        let b = host_bytes(&regs, side, 64).expect("staging again");
        assert_eq!(a, b, "a handle must have one backing, not one per copy");
        // SAFETY: as above.
        assert_eq!(unsafe { *b.add(63) }, 0xAB);
    }

    #[test]
    fn a_copy_into_an_interior_offset_lands_at_that_offset_and_disturbs_nothing_before_it() {
        // `registries()` builds real HandleRegistrys, so every alloc, staging_ptr and free
        // below moves the process-global tally. Same lock as everything that asserts on it.
        let _g = ledger::test_lock();
        // The planner-arithmetic property, proven locally because a real ORT session
        // has so far refused to exercise it: every run to date reports 0 interior
        // copies. If ORT ever does hand us `base + n`, this is the behaviour it gets.
        let regs = registries();
        let reg = regs.values().next().expect("one registry").clone();
        let h = reg.alloc(256).expect("alloc");

        let whole = host_bytes(&regs, classify(&regs, h as *mut u8), 256).expect("staging");
        // SAFETY: `whole` is 256 writable staging bytes for this handle.
        unsafe { ptr::write_bytes(whole, 0x11, 256) };

        let src = [0xEEu8; 64];
        let side = classify(&regs, (h + 96) as *mut u8);
        let to = host_bytes(&regs, side, 64).expect("interior staging");
        assert_eq!(
            to as usize,
            whole as usize + 96,
            "an interior handle must resolve to the same backing at the same offset, \
             not to a fresh allocation"
        );
        // SAFETY: `to` is 64 writable bytes at offset 96 of a 256-byte span.
        unsafe { ptr::copy_nonoverlapping(src.as_ptr(), to, 64) };

        // SAFETY: `whole` covers the full 256-byte span.
        let seen = unsafe { std::slice::from_raw_parts(whole, 256) };
        assert!(
            seen[..96].iter().all(|&b| b == 0x11),
            "wrote before the offset"
        );
        assert!(
            seen[96..160].iter().all(|&b| b == 0xEE),
            "payload misplaced"
        );
        assert!(
            seen[160..].iter().all(|&b| b == 0x11),
            "wrote past the length"
        );
    }

    /// Positive control for the quarantine detector.
    ///
    /// A real ORT session reports `pointers_use_after_free: 0`, and that number is worth
    /// nothing on its own — it is exactly what a detector that never runs would also report.
    /// This is the planted violation that distinguishes the two, the same shape as the layering
    /// lint's deliberate breach: present a stale handle through the very funnel a real session
    /// uses, and require the ledger to count it.
    #[test]
    fn the_quarantine_detector_fires_when_a_stale_handle_is_presented() {
        let _lock = ledger::test_lock();
        ledger::reset();
        let regs = registries();
        let reg = regs.values().next().expect("one registry").clone();

        let h = reg.alloc(4096).expect("alloc");
        // Classify it while live, so the control also proves the detector is quiet when it
        // should be. A detector that fires on everything is as useless as one that never does.
        classify(&regs, h as *mut u8);
        assert_eq!(
            ledger::snapshot().use_after_free,
            0,
            "a live handle must not be reported as a use-after-free"
        );

        reg.free(h);

        // Exactly what a stale ORT pointer would look like, arriving at exactly the place a real
        // one arrives: `classify`, from `CopyTensors` or `host_backing_for`.
        let side = classify(&regs, (h + 64) as *mut u8);
        assert_eq!(
            ledger::snapshot().use_after_free,
            1,
            "the freed handle was not detected — a stale pointer would have aliased onto \
             whatever is allocated there next"
        );
        // And it must be refused, not merely counted: a loud number attached to a silent success
        // is still a use-after-free.
        host_bytes(&regs, side, 64).expect_err("a freed handle must not yield usable backing");
    }

    #[test]
    fn a_freed_handle_is_refused_by_the_transfer_path_not_silently_copied_into() {
        // `registries()` builds real HandleRegistrys, so every alloc, staging_ptr and free
        // below moves the process-global tally. Same lock as everything that asserts on it.
        let _g = ledger::test_lock();
        let regs = registries();
        let reg = regs.values().next().expect("one registry").clone();
        let h = reg.alloc(64).expect("alloc");
        reg.free(h);
        let side = classify(&regs, h as *mut u8);
        let err = host_bytes(&regs, side, 64).expect_err("a freed handle must not accept bytes");
        assert!(
            err.contains("not to a live span"),
            "unhelpful message: {err}"
        );
    }

    #[test]
    fn freeing_a_handle_releases_its_staging_but_keeps_the_address_quarantined() {
        // The ledger and the tally are process-global; this test moves them, so it runs
        // under the same lock as every test that asserts on them.
        let _g = ledger::test_lock();
        let reg = HandleRegistry::new().expect("reservation");
        let h = reg.alloc(8192).expect("alloc");
        assert!(reg.staging_ptr(h).is_some());
        assert!(reg.stats().staging_live_bytes >= 8192);
        reg.free(h);
        assert_eq!(
            reg.stats().staging_live_bytes,
            0,
            "staging must not be held for the length of the quarantine window — at model scale \
             that multiplies peak host memory by the quarantine depth"
        );
        assert!(
            reg.staging_ptr(h).is_none(),
            "a quarantined handle must not hand out backing memory"
        );
    }

    /// A provider that records what it was asked to write, so the mirror can be *observed* rather
    /// than inferred from a counter that the same code path increments.
    #[derive(Default)]
    struct RecordingProvider {
        writes: std::sync::Mutex<Vec<(u64, usize, Vec<u8>)>>,
    }

    impl crate::engine::DeviceMemoryProvider for RecordingProvider {
        fn alloc(&self, _size: usize) -> Option<crate::engine::BufferView> {
            Some(crate::engine::BufferView::from_raw(0xbeef))
        }
        fn free(&self, _view: crate::engine::BufferView) {}
        fn upload(
            &self,
            view: crate::engine::BufferView,
            offset: usize,
            src: &[u8],
        ) -> Result<(), String> {
            self.writes
                .lock()
                .expect("lock")
                .push((view.as_raw(), offset, src.to_vec()));
            Ok(())
        }
        fn download(
            &self,
            _view: crate::engine::BufferView,
            _offset: usize,
            _dst: &mut [u8],
        ) -> Result<(), String> {
            Err("the mirror is never read back — see Endpoint's doc comment".to_string())
        }
        fn is_unified_memory(&self) -> bool {
            false
        }
    }

    /// The mirror must receive the *bytes*, at the *interior offset*, and staging must still hold
    /// them.
    ///
    /// This is the instrument that goes red if `alloc_device_backed_spans` were ever an accounting
    /// change rather than a change in where bytes live: it does not read a counter, it reads what
    /// the provider was handed. If the copy stopped reaching the device, or reached it at offset
    /// 0 for an interior pointer — which would silently overwrite a neighbouring tensor in the
    /// same planner-subdivided span — this fails.
    #[test]
    fn a_copy_into_a_device_backed_handle_mirrors_the_bytes_at_the_right_offset() {
        // `registries()` builds real HandleRegistrys, so every alloc, staging_ptr and free
        // below moves the process-global tally. Same lock as everything that asserts on it.
        let _g = ledger::test_lock();
        use std::sync::Mutex as M;
        let _ = M::new(()); // keep the import used on all cfgs
        use crate::engine::DeviceMemoryProvider as _;
        let provider = Arc::new(RecordingProvider::default());
        // Device index 4242 is not one any real registry claims, so this test cannot disturb — or
        // be disturbed by — a provider another test registered.
        crate::engine::register_device_memory_provider(4242, provider.clone());

        let regs = registries();
        let reg = regs.values().next().expect("one registry").clone();
        reg.set_device_index(4242);
        let h = reg.alloc(4096).expect("alloc");
        reg.attach_buffer(h, crate::engine::BufferView::from_raw(0xbeef))
            .expect("attach");

        let payload: Vec<u8> = (0..64u8).collect();
        let side = classify(&regs, (h + 128) as *mut u8);
        let to = resolve_endpoint(&regs, side, payload.len()).expect("resolve");
        let host = to.host_ptr();
        // SAFETY: `host` is 64 readable/writable bytes of this span's staging, bounds-checked by
        // `resolve_endpoint` above, and `payload` is a distinct allocation.
        unsafe { ptr::copy_nonoverlapping(payload.as_ptr(), host, payload.len()) };

        let Endpoint::Mirrored { view, offset, .. } = to else {
            panic!("a span with an attached buffer must resolve as Mirrored, got {to:?}");
        };
        // SAFETY: as above, read-only.
        let src = unsafe { std::slice::from_raw_parts(host.cast_const(), payload.len()) };
        provider.upload(view, offset, src).expect("upload");

        let writes = provider.writes.lock().expect("lock");
        assert_eq!(writes.len(), 1, "exactly one mirror write");
        assert_eq!(writes[0].0, 0xbeef, "the span's own buffer, not another's");
        assert_eq!(
            writes[0].1, 128,
            "an interior pointer must mirror at its offset, not at 0 — offset 0 would overwrite \
             the neighbouring tensor the planner put at the base of this span"
        );
        assert_eq!(writes[0].2, payload, "the bytes themselves, not a count");
        // SAFETY: as above.
        let staged = unsafe { std::slice::from_raw_parts(host.cast_const(), payload.len()) };
        assert_eq!(
            staged,
            &payload[..],
            "staging stays authoritative: the engine reads it through host_backing_for"
        );
        drop(writes);
        reg.free(h);
    }

    /// `mirror_to_device` must push the *engine's* write down to the device, at the right offset.
    ///
    /// This is the falsifier for the staleness hazard that binding device buffers creates. Before
    /// the session bound input buffers, a span written as an output through host staging could
    /// stay stale in device memory forever and nothing read it. Now it can be read, so the write
    /// must land. If this function ever stops reaching the provider, a bound input reads whatever
    /// the *previous* inference left there — numerically plausible, and therefore the failure mode
    /// that survives a smoke test.
    #[test]
    fn an_engine_write_to_staging_is_pushed_to_the_device_mirror() {
        // `registries()` builds real HandleRegistrys, so every alloc, staging_ptr and free
        // below moves the process-global tally. Same lock as everything that asserts on it.
        let _g = ledger::test_lock();
        let provider = Arc::new(RecordingProvider::default());
        crate::engine::register_device_memory_provider(4243, provider.clone());

        let regs = registries();
        let reg = regs.values().next().expect("one registry").clone();
        reg.set_device_index(4243);
        let h = reg.alloc(4096).expect("alloc");
        reg.attach_buffer(h, crate::engine::BufferView::from_raw(0xfeed))
            .expect("attach");

        // Write through the host backing, exactly as `write_outputs_to_ort` does.
        let payload: Vec<u8> = (0..32u8).map(|b| b ^ 0x5a).collect();
        let side = classify(&regs, (h + 256) as *mut u8);
        let ep = resolve_endpoint(&regs, side, payload.len()).expect("resolve");
        let host = ep.host_ptr();
        // SAFETY: `host` addresses at least `payload.len()` writable staging bytes — bounds
        // checked by `resolve_endpoint` — and `payload` is a distinct allocation.
        unsafe { ptr::copy_nonoverlapping(payload.as_ptr(), host, payload.len()) };

        // The provider only has the handle; that is all the session has too.
        let pushed = mirror_in(&regs, (h + 256) as *mut u8, payload.len()).expect("mirror");
        assert!(pushed, "a device-backed span must report that it mirrored");

        let writes = provider.writes.lock().expect("lock");
        assert_eq!(writes.len(), 1, "exactly one mirror write");
        assert_eq!(writes[0].0, 0xfeed, "this span's buffer");
        assert_eq!(
            writes[0].1, 256,
            "the engine's write must mirror at the span offset, not at 0"
        );
        assert_eq!(
            writes[0].2, payload,
            "the bytes the engine wrote, not a zeroed buffer"
        );
        drop(writes);
        reg.free(h);
    }

    /// Ordinary host memory must not be mirrored, and must not error.
    ///
    /// The default build has no device memory at all. `write_outputs_to_ort` calls
    /// `mirror_to_device` unconditionally, so a stack pointer arriving here has to be a cheap
    /// `Ok(false)` — not an error that would fail every inference in the configuration that is
    /// actually shipped.
    #[test]
    fn a_host_output_is_not_mirrored_and_is_not_an_error() {
        // `registries()` builds real HandleRegistrys, so every alloc, staging_ptr and free
        // below moves the process-global tally. Same lock as everything that asserts on it.
        let _g = ledger::test_lock();
        let mut buf = [7u8; 64];
        let r = mirror_in(&registries(), buf.as_mut_ptr(), buf.len());
        assert_eq!(
            r,
            Ok(false),
            "a host pointer has no device mirror, and that is not a failure"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────────────────────
    // Device authority
    //
    // These cover the invariant this file was built on being *false* for some spans: that the
    // staging block is always the real copy. The measured failure they stand against is the
    // residency probe's first EP-side lane, which returned 0 nonzero of 32064 logits and 0 of
    // 27648 in each `present` tensor, and scored a perfect 1.0 against a relative metric because
    // two all-zero tensors agree with each other.
    // ─────────────────────────────────────────────────────────────────────────────────────────

    /// A provider that actually holds device bytes, so a download can return something other than
    /// what staging already had.
    ///
    /// [`RecordingProvider`]'s `download` deliberately errors — it encoded the old invariant, that
    /// the mirror is never read back. A test written against it could not tell a refresh that
    /// happened from one that did not, so authority needs its own.
    #[derive(Default)]
    struct DeviceHeldProvider {
        device: std::sync::Mutex<Vec<u8>>,
        downloads: std::sync::atomic::AtomicUsize,
    }

    impl crate::engine::DeviceMemoryProvider for DeviceHeldProvider {
        fn alloc(&self, _size: usize) -> Option<crate::engine::BufferView> {
            Some(crate::engine::BufferView::from_raw(0xd00d))
        }
        fn free(&self, _view: crate::engine::BufferView) {}
        fn upload(
            &self,
            _view: crate::engine::BufferView,
            offset: usize,
            src: &[u8],
        ) -> Result<(), String> {
            let mut d = self.device.lock().expect("lock");
            if d.len() < offset + src.len() {
                d.resize(offset + src.len(), 0);
            }
            d[offset..offset + src.len()].copy_from_slice(src);
            Ok(())
        }
        fn download(
            &self,
            _view: crate::engine::BufferView,
            offset: usize,
            dst: &mut [u8],
        ) -> Result<(), String> {
            self.downloads
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let d = self.device.lock().expect("lock");
            if d.len() < offset + dst.len() {
                return Err(format!(
                    "device buffer holds {} byte(s); a download of {} from offset {offset} runs \
                     past it",
                    d.len(),
                    dst.len()
                ));
            }
            dst.copy_from_slice(&d[offset..offset + dst.len()]);
            Ok(())
        }
        fn is_unified_memory(&self) -> bool {
            false
        }
    }

    /// Register a `DeviceHeldProvider` on a device index no real registry claims, and hand back a
    /// registry whose spans resolve to it.
    type DeviceHeldFixture = (
        HashMap<(u32, u32), Arc<HandleRegistry>>,
        Arc<HandleRegistry>,
        Arc<DeviceHeldProvider>,
        usize,
    );

    fn device_held(device_index: usize, span_bytes: usize) -> DeviceHeldFixture {
        let provider = Arc::new(DeviceHeldProvider::default());
        crate::engine::register_device_memory_provider(device_index, provider.clone());
        let regs = registries();
        let reg = regs.values().next().expect("one registry").clone();
        reg.set_device_index(device_index);
        let h = reg.alloc(span_bytes).expect("alloc");
        reg.attach_buffer(h, crate::engine::BufferView::from_raw(0xd00d))
            .expect("attach");
        (regs, reg, provider, h)
    }

    /// The defect itself: a span a dispatch wrote directly must read as the *device's* bytes.
    ///
    /// Staging is left as zeros, exactly as it is in production when nothing ever wrote it, and
    /// the device holds the real answer. Without the refresh this reads zeros and reports success,
    /// which is what shipped before this change.
    #[test]
    fn a_device_authoritative_span_is_downloaded_before_it_is_read() {
        let _g = ledger::test_lock();
        use crate::engine::DeviceMemoryProvider as _;
        let (regs, reg, provider, h) = device_held(4243, 256);

        let truth: Vec<u8> = (0..256u32).map(|i| (i % 251 + 1) as u8).collect();
        provider
            .upload(crate::engine::BufferView::from_raw(0xd00d), 0, &truth)
            .expect("seed the device");
        // Staging is untouched: this is the production state after a dispatch wrote the buffer.
        reg.set_device_authoritative(h, true).expect("mark");

        let side = classify(&regs, h as *mut u8);
        let e = resolve_endpoint(&regs, side, 256).expect("resolve");
        refresh_from_device_if_authoritative(e, 256).expect("refresh");

        // SAFETY: `e.host_ptr()` addresses 256 readable bytes of this span's staging, bounds
        // checked by `resolve_endpoint` above.
        let got = unsafe { std::slice::from_raw_parts(e.host_ptr().cast_const(), 256) };
        assert_eq!(
            got,
            &truth[..],
            "a span whose device buffer is authoritative must read as the device's bytes; \
             reading staging here is the all-zero failure this flag exists to prevent"
        );
        assert!(
            provider
                .downloads
                .load(std::sync::atomic::Ordering::Relaxed)
                > 0,
            "the bytes must have come from a download, not from staging that happened to match"
        );
        reg.free(h);
    }

    /// The same call on a span that is *not* authoritative must not touch the device.
    ///
    /// This is the falsifier for the refresh being unconditional. An unconditional download would
    /// pass the test above, be correct, and quietly reintroduce on every read the round trip this
    /// whole design exists to remove.
    #[test]
    fn a_staging_authoritative_span_is_not_downloaded() {
        let _g = ledger::test_lock();
        let (regs, reg, provider, h) = device_held(4244, 128);

        let side = classify(&regs, h as *mut u8);
        let e = resolve_endpoint(&regs, side, 128).expect("resolve");
        refresh_from_device_if_authoritative(e, 128).expect("refresh");

        assert_eq!(
            provider
                .downloads
                .load(std::sync::atomic::Ordering::Relaxed),
            0,
            "the default path must be inert: nothing marked authority, so nothing may be read back"
        );
        reg.free(h);
    }

    /// Authority over a span with no `VkBuffer` is refused, not recorded.
    ///
    /// Setting it would aim every subsequent read at memory that does not exist, and the read
    /// would have no way to notice.
    #[test]
    fn authority_over_a_span_with_no_device_buffer_is_refused() {
        let _g = ledger::test_lock();
        let regs = registries();
        let reg = regs.values().next().expect("one registry").clone();
        let h = reg.alloc(64).expect("alloc");

        assert!(
            reg.set_device_authoritative(h, true).is_err(),
            "a span with no VkBuffer has no device copy to be authoritative about"
        );
        let side = classify(&regs, h as *mut u8);
        let e = resolve_endpoint(&regs, side, 64).expect("resolve");
        assert!(
            !matches!(
                e,
                Endpoint::Mirrored {
                    device_authoritative: true,
                    ..
                }
            ),
            "the refusal must leave the span unmarked, not merely return Err"
        );
        reg.free(h);
    }

    /// A partial host write must bring the rest of the span down before authority is handed back.
    ///
    /// The window that is written is fine either way. What this covers is the *rest* of the span:
    /// bytes a dispatch wrote and nothing ever read back, which the revoke that follows a partial
    /// write declares to live in staging. This is invisible on a whole-span write — the common
    /// case, and the one the residency probe exercises — so it cannot be found by running the case
    /// that already passes.
    #[test]
    fn a_partial_write_refreshes_the_bytes_it_does_not_write() {
        let _g = ledger::test_lock();
        use crate::engine::DeviceMemoryProvider as _;
        let (regs, reg, provider, h) = device_held(4245, 256);

        let truth: Vec<u8> = (0..256u32).map(|i| (i % 251 + 3) as u8).collect();
        provider
            .upload(crate::engine::BufferView::from_raw(0xd00d), 0, &truth)
            .expect("seed the device");
        reg.set_device_authoritative(h, true).expect("mark");

        // A 64-byte write at offset 64: neither the head nor the tail of the span is covered.
        let side = classify(&regs, (h + 64) as *mut u8);
        let e = resolve_endpoint(&regs, side, 64).expect("resolve");
        refresh_whole_span_before_partial_write(e, 64).expect("refresh");

        let base = reg.staging_ptr(h).expect("staging");
        // SAFETY: `staging_ptr` returns the base of an allocation of the span's padded size,
        // which is at least the 256 requested bytes.
        let staged = unsafe { std::slice::from_raw_parts(base.cast_const(), 256) };
        assert_eq!(
            &staged[0..64],
            &truth[0..64],
            "bytes before the written window must come down from the device: the revoke that \
             follows this write claims the WHOLE span lives in staging"
        );
        assert_eq!(
            &staged[128..256],
            &truth[128..256],
            "and bytes after it, for the same reason"
        );
        reg.free(h);
    }

    /// A whole-span write needs no refresh, because every byte is about to be replaced.
    #[test]
    fn a_whole_span_write_does_not_download_first() {
        let _g = ledger::test_lock();
        let (regs, reg, provider, h) = device_held(4246, 128);
        reg.set_device_authoritative(h, true).expect("mark");

        let side = classify(&regs, h as *mut u8);
        let e = resolve_endpoint(&regs, side, 128).expect("resolve");
        refresh_whole_span_before_partial_write(e, 128).expect("refresh");

        assert_eq!(
            provider
                .downloads
                .load(std::sync::atomic::Ordering::Relaxed),
            0,
            "downloading bytes that are about to be overwritten is the round trip this design \
             removes, paid for nothing"
        );
        reg.free(h);
    }

    /// `mirror_in` must refuse to upload over a device-authoritative span.
    ///
    /// That upload is destructive and after the fact: it overwrites a dispatch's output with a
    /// staging block that was never written, and there is nothing left to recover it from.
    #[test]
    fn mirroring_in_over_an_authoritative_span_is_refused_not_performed() {
        let _g = ledger::test_lock();
        use crate::engine::DeviceMemoryProvider as _;
        let (regs, reg, provider, h) = device_held(4247, 128);

        let truth = vec![9u8; 128];
        provider
            .upload(crate::engine::BufferView::from_raw(0xd00d), 0, &truth)
            .expect("seed the device");
        reg.set_device_authoritative(h, true).expect("mark");

        let r = mirror_in(&regs, h as *mut u8, 128);
        assert!(
            r.is_err(),
            "an upload from stale staging over a dispatch's output must be refused, not performed"
        );
        assert_eq!(
            *provider.device.lock().expect("lock"),
            truth,
            "and the refusal must be before the write, not after it"
        );
        reg.free(h);
    }

    /// Authority does not survive the span it was granted over.
    ///
    /// The arena's lifetime question in miniature. A construction site cannot inherit a previous
    /// tenant's authority: the address may be reissued, and a fresh span whose staging is the real
    /// copy would be read from a device buffer belonging to nothing.
    #[test]
    fn a_freed_span_does_not_carry_authority_into_its_successor() {
        let _g = ledger::test_lock();
        let (regs, reg, _provider, h) = device_held(4248, 128);
        reg.set_device_authoritative(h, true).expect("mark");
        reg.free(h);

        assert!(
            reg.set_device_authoritative(h, true).is_err(),
            "a freed handle must not accept authority"
        );
        let h2 = reg.alloc(128).expect("second alloc");
        let side = classify(&regs, h2 as *mut u8);
        let e = resolve_endpoint(&regs, side, 128).expect("resolve");
        assert!(
            !matches!(
                e,
                Endpoint::Mirrored {
                    device_authoritative: true,
                    ..
                }
            ),
            "a newly allocated span starts with staging authoritative, whatever its address held \
             before"
        );
        reg.free(h2);
    }

    /// `mark_device_authoritative` refuses a pointer that is not one of ours, and says so by
    /// returning `false` rather than by succeeding quietly.
    ///
    /// The caller's contract is that `false` means "do not bind". A silent `true` here would bind
    /// a buffer for a span that still reports staging as the real copy, which reads as zeros.
    #[test]
    fn marking_a_pointer_that_is_not_a_handle_returns_false() {
        let _g = ledger::test_lock();
        let mut buf = [0u8; 64];
        assert!(
            !mark_device_authoritative(buf.as_mut_ptr(), true),
            "ordinary host memory has no span to be authoritative about"
        );
    }
}
