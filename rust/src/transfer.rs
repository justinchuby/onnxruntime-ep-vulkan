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

use crate::allocator::{HandleRegistry, LookupError};
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
        match reg.classify(addr) {
            Ok(r) => {
                return Side::Device {
                    base: r.base,
                    offset: r.offset,
                    span_size: r.size,
                    has_device_buffer: r.buffer.is_some(),
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
                };
            }
            Err(LookupError::NotAHandle { .. }) => {}
        }
    }
    Side::Host(p)
}

/// The bytes behind one endpoint, as a host-addressable pointer.
///
/// Returns `Err` when the endpoint is a handle whose contents live on the device — that copy
/// belongs to the engine layer, not here, and is not yet wired.
fn host_bytes(
    registries: &HashMap<(u32, u32), Arc<HandleRegistry>>,
    side: Side,
    len: usize,
) -> Result<*mut u8, String> {
    match side {
        Side::Host(p) => {
            if p.is_null() && len != 0 {
                return Err("a host endpoint of the copy is a null pointer".to_string());
            }
            Ok(p)
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
            if has_device_buffer {
                return Err(format!(
                    "device handle 0x{base:x} has a VkBuffer attached, so this copy must go \
                     through the engine layer's staging and `vkCmdCopyBuffer` path — which is not \
                     wired yet. Refusing rather than copying to the wrong memory."
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
            Ok(unsafe { p.add(offset) })
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
    let from = host_bytes(&me.registries, src_side, src_len)?;
    let to = host_bytes(&me.registries, dst_side, dst_len)?;

    use std::sync::atomic::Ordering::Relaxed;
    COPIES.fetch_add(1, Relaxed);
    COPIED_BYTES.fetch_add(src_len as u64, Relaxed);
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

    if from == to {
        // Same memory on both sides: ORT does ask for this when a tensor is already where it needs
        // to be. Copying would be `memcpy` with overlapping identical ranges, which is defined but
        // pointless; more importantly, `copy_nonoverlapping` with `src == dst` is not.
        return Ok(());
    }

    // SAFETY: `from` and `to` each point to at least `src_len` readable/writable bytes — for host
    // endpoints because ORT sized the tensor, and for handle endpoints because `host_bytes`
    // rejected any copy extending past the span's requested size. They are distinct allocations:
    // a handle's staging is a private heap block, and the equal-pointer case returned above.
    unsafe { ptr::copy_nonoverlapping(from, to, src_len) };
    Ok(())
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
        side => Some(host_bytes(&registries, side, len)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registries() -> HashMap<(u32, u32), Arc<HandleRegistry>> {
        let mut m = HashMap::new();
        m.insert((0x10de, 0), HandleRegistry::new().expect("reservation"));
        m
    }

    #[test]
    fn a_host_pointer_is_not_mistaken_for_a_handle() {
        let regs = registries();
        let mut buf = [0u8; 32];
        match classify(&regs, buf.as_mut_ptr()) {
            Side::Host(p) => assert_eq!(p, buf.as_mut_ptr()),
            Side::Device { .. } => panic!("a stack buffer was classified as a device handle"),
        }
    }

    #[test]
    fn an_interior_handle_pointer_classifies_to_its_span_and_offset() {
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

    #[test]
    fn a_freed_handle_is_refused_by_the_transfer_path_not_silently_copied_into() {
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
}
