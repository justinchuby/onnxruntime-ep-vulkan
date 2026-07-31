//! Device allocator — the ORT-facing `OrtAllocator` over Vulkan device memory.
//!
//! # The problem this file exists to solve (OQ-3)
//!
//! ORT's allocator API is pointer-based: `Alloc(size) -> void*`, `Free(void*)`. Vulkan's unit of
//! device memory is a `VkBuffer` plus an offset, which is not a pointer and cannot be made into
//! one without `VK_KHR_buffer_device_address` — an extension we rejected (decisions.md, OQ-3)
//! because it is not universally available on the mobile and software targets `PLATFORMS.md`
//! requires, and because a real GPU address handed to a host that may do arithmetic on it is a
//! much worse failure mode than an opaque one.
//!
//! So the `void*` we return is a **handle**, not an address that can be read. The question OQ-3
//! actually turns on is what kind of handle survives contact with ORT.
//!
//! ## Why the handle is a reserved virtual address rather than an integer
//!
//! The tempting answer is a small integer cast to a pointer — `1`, `2`, `3`. It fails, and it
//! fails silently, for one reason: **ORT's memory-pattern planner does arithmetic on allocator
//! return values.** It allocates one large block per pattern and hands out `base + offset` to
//! individual tensors. With integer handles, `handle_3 + 512` is `handle_4`-ish: a value that
//! collides with a *different live allocation*, and every subsequent lookup silently resolves to
//! the wrong tensor. That is a wrong answer with no crash and no diagnostic — the worst class of
//! bug this project can ship.
//!
//! This module's answer: reserve a large contiguous region of **real virtual address space**
//! (`VirtualAlloc(MEM_RESERVE|PAGE_NOACCESS)` on Windows, `mmap(PROT_NONE, MAP_NORESERVE)`
//! elsewhere) and carve handles out of it, page-aligned with a guard page between spans. Then:
//!
//! - `ptr + n` for `n < size` **stays inside the same span by construction**, so a range lookup
//!   recovers `(handle, offset)` exactly. The planner's arithmetic becomes something we can
//!   interpret rather than something we have to forbid.
//! - The guard page means `ptr + size` — the one-past-the-end pointer that pointer arithmetic
//!   naturally produces — lands in a hole rather than on the next live allocation. It resolves to
//!   an error instead of to someone else's tensor.
//! - Nothing in the region is committed or readable. If any code anywhere — ORT, a kernel, a
//!   `memcpy` we failed to intercept — treats the handle as a real pointer and dereferences it,
//!   the process faults **at the instruction that did it**, with the handle in the fault address.
//!   That is a loud, attributable, immediately debuggable failure instead of silent corruption.
//!   The unreadability is the safety property, not a limitation.
//!
//! Reserved address space is close to free on 64-bit: it consumes no physical memory, no page
//! table entries and no commit charge. We reserve 64 GiB by default and it costs nothing.
//!
//! ## Generation-stamped quarantine on free
//!
//! A freed handle whose address is immediately recycled aliases onto a live tensor — the same
//! silent-wrong-answer class as integer handles. So `Free` does not return the span to service.
//! It moves it to a quarantine FIFO where lookups resolve to a **loud, attributable error naming
//! the generation** ("handle 0x… was freed at generation 3; the caller is holding a stale
//! pointer") rather than to a miss or to a neighbour.
//!
//! Quarantine is bounded — address space is large but finite — so a span is eventually retired and
//! reused with its generation bumped. **This is a window, not a proof, and the distinction is
//! recorded honestly:** a use-after-free detected inside the window is caught with a precise
//! message; one detected after the span was retired and re-served is not detectable at all,
//! because ORT hands back a bare pointer with no generation in it. [`AllocStats::quarantine_retired`]
//! counts retirements precisely so a run can tell you whether the window was ever exhausted. A
//! run that reports zero retirements has a proof for that run; a run that reports thousands has a
//! window.
//!
//! # Layering
//!
//! This file owns the handle scheme, the ORT-facing vtable and the lifetime contract. The
//! `VkBuffer`/`VkDeviceMemory` behind each handle is Switch's (`vk/alloc.rs`), reached through the
//! opaque [`engine::BufferView`] token — no `ash` type appears here and no `sys::ort` type appears
//! there. [`HandleRegistry::attach_buffer`] is the seam.

use std::collections::BTreeMap;
use std::ffi::{CString, c_void};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::engine::BufferView;
use crate::factory::ENV_DEVICE_MEMORY;
use crate::sys::ort;

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Tunables
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// How much virtual address space to reserve for handles, in bytes.
///
/// This is address space, not memory: it is never committed and never backed by pages. 64 GiB on
/// a 128 TiB user address space is unmeasurable, and it bounds how much *simultaneously live plus
/// quarantined* device memory the handle scheme can describe — not how much device memory exists.
pub const DEFAULT_RESERVATION_BYTES: usize = 64 << 30;

/// Override for [`DEFAULT_RESERVATION_BYTES`], in mebibytes.
pub const ENV_RESERVATION_MIB: &str = "ONNXRUNTIME_EP_VULKAN_VA_RESERVE_MIB";

/// How many freed spans to hold in quarantine before retiring the oldest for reuse.
pub const DEFAULT_QUARANTINE_SPANS: usize = 4096;

/// Override for [`DEFAULT_QUARANTINE_SPANS`].
pub const ENV_QUARANTINE_SPANS: &str = "ONNXRUNTIME_EP_VULKAN_QUARANTINE_SPANS";

/// Granularity every handle base and length is rounded up to.
///
/// Exported because `factory.rs` promises it to ORT in the `OrtMemoryInfo` alignment field, and a
/// promise made in one file against a constant defined in another is how those drift apart.
pub const HANDLE_ALIGNMENT: usize = SPAN_GRANULARITY;

/// Granularity every span base and length is rounded up to.
///
/// One page keeps spans on distinct pages so the guard page below is a real hole in the address
/// space rather than an arithmetic convention, and it comfortably exceeds any alignment ORT asks
/// a device allocator for.
const SPAN_GRANULARITY: usize = 4096;

/// Dead address space left between consecutive spans.
///
/// Exists so the one-past-the-end pointer `base + size` — which ordinary pointer arithmetic
/// produces constantly and which is legal to *form* in C — resolves to "not a live handle" rather
/// than to the base of whatever was allocated next.
const GUARD_BYTES: usize = SPAN_GRANULARITY;

/// Identifies one of our allocators when ORT hands `*mut OrtAllocator` back.
const ALLOCATOR_MAGIC: u64 = 0x564B_414C_4C4F_4331; // "VKALLOC1"

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Reserved virtual address space
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// A contiguous region of reserved, **inaccessible** virtual address space.
///
/// Reserved and not committed on purpose. See the module docs: unreadability is what turns "some
/// code treated a handle as a pointer" from silent corruption into an immediate, attributable
/// fault at the offending instruction.
struct VaReservation {
    base: usize,
    len: usize,
}

// SAFETY: the reservation is a pair of integers describing address space this process owns for its
// lifetime. Nothing is dereferenced, so there is no aliasing or thread-affinity concern.
unsafe impl Send for VaReservation {}
// SAFETY: as above; the struct is immutable after construction.
unsafe impl Sync for VaReservation {}

#[cfg(windows)]
mod sys_va {
    // Declared inline rather than pulling in `windows-sys`. Two functions with stable, ancient
    // signatures do not justify a dependency in a plugin that must build on every lane.
    unsafe extern "system" {
        fn VirtualAlloc(
            lpAddress: *mut core::ffi::c_void,
            dwSize: usize,
            flAllocationType: u32,
            flProtect: u32,
        ) -> *mut core::ffi::c_void;
        fn VirtualFree(
            lpAddress: *mut core::ffi::c_void,
            dwSize: usize,
            dwFreeType: u32,
        ) -> core::ffi::c_int;
    }

    const MEM_RESERVE: u32 = 0x2000;
    const MEM_RELEASE: u32 = 0x8000;
    const PAGE_NOACCESS: u32 = 0x01;

    pub(super) fn reserve(len: usize) -> Option<usize> {
        // SAFETY: a null `lpAddress` asks the OS to choose the address. `MEM_RESERVE` with
        // `PAGE_NOACCESS` commits nothing and makes every byte inaccessible, which is exactly the
        // property we want. Returns null on failure, which we check.
        let p = unsafe { VirtualAlloc(core::ptr::null_mut(), len, MEM_RESERVE, PAGE_NOACCESS) };
        if p.is_null() { None } else { Some(p as usize) }
    }

    pub(super) fn release(base: usize, _len: usize) {
        // SAFETY: `base` came from `VirtualAlloc` above and has not been released. `MEM_RELEASE`
        // requires a zero size and the exact base address, both of which hold.
        unsafe { VirtualFree(base as *mut core::ffi::c_void, 0, MEM_RELEASE) };
    }
}

#[cfg(not(windows))]
mod sys_va {
    unsafe extern "C" {
        fn mmap(
            addr: *mut core::ffi::c_void,
            length: usize,
            prot: core::ffi::c_int,
            flags: core::ffi::c_int,
            fd: core::ffi::c_int,
            offset: i64,
        ) -> *mut core::ffi::c_void;
        fn munmap(addr: *mut core::ffi::c_void, length: usize) -> core::ffi::c_int;
    }

    const PROT_NONE: core::ffi::c_int = 0;
    const MAP_PRIVATE: core::ffi::c_int = 0x0002;
    #[cfg(target_os = "linux")]
    const MAP_ANONYMOUS: core::ffi::c_int = 0x0020;
    #[cfg(not(target_os = "linux"))]
    const MAP_ANONYMOUS: core::ffi::c_int = 0x1000;
    // `MAP_NORESERVE` asks Linux not to charge the reservation against swap/overcommit. It does
    // not exist on macOS, where an anonymous `PROT_NONE` mapping is already free.
    #[cfg(target_os = "linux")]
    const MAP_NORESERVE: core::ffi::c_int = 0x4000;
    #[cfg(not(target_os = "linux"))]
    const MAP_NORESERVE: core::ffi::c_int = 0;

    pub(super) fn reserve(len: usize) -> Option<usize> {
        // SAFETY: a null `addr` asks the kernel to choose. `PROT_NONE` + anonymous + private
        // reserves address space without backing it, and touching it faults — the property we
        // want. `MAP_FAILED` is `-1`, which we check.
        let p = unsafe {
            mmap(
                core::ptr::null_mut(),
                len,
                PROT_NONE,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE,
                -1,
                0,
            )
        };
        if p as isize == -1 || p.is_null() {
            None
        } else {
            Some(p as usize)
        }
    }

    pub(super) fn release(base: usize, len: usize) {
        // SAFETY: `base`/`len` are exactly what `mmap` returned and were never unmapped.
        unsafe { munmap(base as *mut core::ffi::c_void, len) };
    }
}

impl VaReservation {
    fn new(len: usize) -> Option<VaReservation> {
        // Halve the request until the OS agrees or we drop below something useless. A machine
        // under address-space pressure should get a smaller arena, not a dead EP.
        let mut want = len;
        while want >= 256 << 20 {
            if let Some(base) = sys_va::reserve(want) {
                if want != len {
                    log::warn!(
                        "VulkanExecutionProvider: reserved {} MiB of handle address space rather \
                         than the requested {} MiB — the OS declined the larger reservation. The \
                         handle scheme still works; it can simply describe less simultaneously \
                         live device memory before it runs out of handles.",
                        want >> 20,
                        len >> 20
                    );
                }
                return Some(VaReservation { base, len: want });
            }
            want /= 2;
        }
        None
    }
}

impl Drop for VaReservation {
    fn drop(&mut self) {
        sys_va::release(self.base, self.len);
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Handle registry
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// Why a lookup failed. Every variant is something a caller should be told precisely, because the
/// three causes demand completely different fixes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LookupError {
    /// The address is not inside our reservation at all — it is a host pointer, or garbage.
    NotAHandle { addr: usize },
    /// The address is inside the reservation but in a guard page or an un-served hole. This is
    /// what a one-past-the-end pointer produces.
    InGuardBand { addr: usize },
    /// The span exists but was freed. The generation says how stale the caller's pointer is.
    Freed {
        addr: usize,
        base: usize,
        freed_at_generation: u64,
    },
}

impl std::fmt::Display for LookupError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LookupError::NotAHandle { addr } => write!(
                f,
                "0x{addr:x} is not a Vulkan EP device handle. It was never returned by this \
                 allocator, so it is either a host pointer that reached a device path by mistake \
                 or an uninitialised value."
            ),
            LookupError::InGuardBand { addr } => write!(
                f,
                "0x{addr:x} lands in a guard band between device handles. This is the signature \
                 of pointer arithmetic that ran off the end of an allocation — most often a \
                 one-past-the-end pointer, or an offset computed against the wrong base."
            ),
            LookupError::Freed {
                addr,
                base,
                freed_at_generation,
            } => write!(
                f,
                "0x{addr:x} belongs to device handle 0x{base:x}, which was freed (generation \
                 {freed_at_generation}). The caller is holding a stale pointer. This was caught \
                 rather than silently aliased onto a live tensor because the span is still in \
                 quarantine."
            ),
        }
    }
}

/// One carved-out span of the reservation.
#[derive(Debug, Clone)]
struct Span {
    base: usize,
    /// Bytes the caller asked for. Lookups are bounded by this, not by the padded length, so an
    /// offset into the rounding slack is reported as out of bounds rather than accepted.
    requested: usize,
    /// Bytes actually consumed from the arena, including rounding (excluding the guard band).
    padded: usize,
    generation: u64,
    live: bool,
    /// The device memory behind this handle. `None` until Switch's session attaches one — the
    /// handle scheme is deliberately usable, testable and verifiable without a Vulkan device.
    buffer: Option<BufferView>,
}

/// A successful lookup: which handle, and how far into it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Resolved {
    /// Base address of the containing handle.
    pub base: usize,
    /// `addr - base`. This is the value that makes ORT's pointer arithmetic interpretable.
    pub offset: usize,
    /// Requested size of the containing handle.
    pub size: usize,
    pub generation: u64,
    pub buffer: Option<BufferView>,
}

/// Counters describing what the allocator has done. Mouse's P6 high-water assertion reads
/// [`AllocStats::high_water_bytes`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AllocStats {
    pub live_bytes: u64,
    pub live_spans: u64,
    /// The largest `live_bytes` ever observed. This is the number a "no dequantised weight is
    /// ever materialised in device memory" assertion is written against.
    pub high_water_bytes: u64,
    pub total_allocations: u64,
    pub total_frees: u64,
    pub quarantined_spans: u64,
    /// Spans retired from quarantine and returned to service. **Non-zero means the
    /// use-after-free window was exhausted at some point in this run**, so a stale-pointer bug
    /// could have gone undetected. Zero means quarantine covered the whole run.
    pub quarantine_retired: u64,
    /// Lookups that failed. Non-zero always indicates a bug somewhere; the log says which kind.
    pub failed_lookups: u64,
    pub arena_bytes: u64,
    pub arena_used_bytes: u64,
    /// Host bytes currently held as staging behind handles that have no device buffer.
    ///
    /// **Non-zero means part of this run's "device memory" was ordinary host memory.** See
    /// [`HandleRegistry::staging_ptr`]. Reported so that no timing taken from such a run can be
    /// mistaken for a device measurement.
    pub staging_live_bytes: u64,
    /// Handles that have ever been given host staging.
    pub staging_spans: u64,
}

/// A host allocation standing in for device memory behind a handle that has no `VkBuffer` yet.
///
/// # Why this exists
///
/// Advertising a device allocator to ORT is a package deal: ORT then requires an
/// `OrtDataTransferImpl`, and a data transfer must be able to *move bytes*. Until the engine layer
/// attaches a real `VkBuffer` to a handle there are no device bytes to move, and every session
/// fails at `Run` — which means the two properties this allocator exists to guarantee (that ORT's
/// memory-pattern planner's `base + offset` resolves, and that a freed handle is rejected rather
/// than aliased) can never be observed against a real host.
///
/// Host staging breaks that deadlock. It is not a shortcut around Vulkan: a CPU→device copy goes
/// through host-visible staging memory anyway, so this is the near half of the real path, built
/// first. When [`HandleRegistry::attach_buffer`] has supplied a `BufferView`, the device path takes
/// over and staging is never consulted.
///
/// # Why it is counted
///
/// A run backed by staging is a *correctness* vehicle and never a performance one. It is reported
/// in [`AllocStats::staging_live_bytes`], warned about once per registry, and named in the release
/// summary, so a number taken from such a run cannot be quietly presented as a device result.
struct HostStaging {
    ptr: *mut u8,
    layout: std::alloc::Layout,
}

// SAFETY: the pointer is a unique owning handle to a private heap allocation. It is only ever
// reached through the registry's `Mutex`, and `HostStaging` hands out no aliases it does not
// outlive (see `staging_ptr`, whose contract is documented on the caller side).
unsafe impl Send for HostStaging {}

impl Drop for HostStaging {
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.layout.size() != 0 {
            // SAFETY: `ptr` came from `alloc_zeroed` with exactly this layout and has not been
            // freed — `HostStaging` is never cloned and is removed from the map by value.
            unsafe { std::alloc::dealloc(self.ptr, self.layout) };
        }
    }
}

/// The handle table. One per device, shared by every allocator instance for that device.
pub struct HandleRegistry {
    arena: VaReservation,
    inner: Mutex<RegistryInner>,
    quarantine_limit: usize,
    // Read on the stats path without taking the lock; exactness is not required for a diagnostic.
    failed_lookups: AtomicU64,
    /// Which device this registry serves, once the factory knows. Used only to find the
    /// [`crate::engine::DeviceMemoryProvider`]; `usize::MAX` means "not yet attributed", which is
    /// the state every unit test runs in.
    device_index: AtomicUsize,
}

struct RegistryInner {
    /// Every span we have carved, live or quarantined, keyed by base address. `BTreeMap` because
    /// the lookup that matters is `range(..=addr).next_back()` — "which span contains this
    /// address" — which is exactly what makes the planner's `base + offset` interpretable.
    spans: BTreeMap<usize, Span>,
    /// Bump pointer into the arena.
    cursor: usize,
    /// Freed spans, oldest first, awaiting retirement.
    quarantine: std::collections::VecDeque<usize>,
    /// Retired spans available for reuse, keyed by padded size.
    free_list: BTreeMap<usize, Vec<usize>>,
    /// Host staging behind handles with no device buffer, keyed by span base.
    staging: BTreeMap<usize, HostStaging>,
    generation: u64,
    stats: AllocStats,
}

impl HandleRegistry {
    /// Build a registry, reserving address space.
    ///
    /// Returns `None` only if the OS refuses even a 256 MiB reservation, which on a 64-bit host
    /// means something is very wrong. The caller degrades to "no device allocator" rather than
    /// failing session creation.
    pub fn new() -> Option<Arc<HandleRegistry>> {
        let want = std::env::var(ENV_RESERVATION_MIB)
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map(|mib| mib << 20)
            .unwrap_or(DEFAULT_RESERVATION_BYTES);
        let quarantine_limit = std::env::var(ENV_QUARANTINE_SPANS)
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(DEFAULT_QUARANTINE_SPANS);

        let arena = VaReservation::new(want)?;
        let arena_bytes = arena.len as u64;
        let base = arena.base;
        log::info!(
            "VulkanExecutionProvider: reserved {} MiB of inaccessible address space at 0x{base:x} \
             for device handles (quarantine {quarantine_limit} spans). Nothing here is committed; \
             a handle dereferenced as a pointer faults immediately and by design.",
            arena.len >> 20
        );
        Some(Arc::new(HandleRegistry {
            arena,
            quarantine_limit,
            failed_lookups: AtomicU64::new(0),
            device_index: AtomicUsize::new(usize::MAX),
            inner: Mutex::new(RegistryInner {
                spans: BTreeMap::new(),
                cursor: base,
                quarantine: std::collections::VecDeque::new(),
                free_list: BTreeMap::new(),
                staging: BTreeMap::new(),
                generation: 1,
                stats: AllocStats {
                    arena_bytes,
                    ..Default::default()
                },
            }),
        }))
    }

    fn arena_end(&self) -> usize {
        self.arena.base + self.arena.len
    }

    /// Serve a handle for `size` bytes. Returns the handle address, or `None` when the arena is
    /// exhausted (which ORT reads as an allocation failure, its normal out-of-memory path).
    pub fn alloc(&self, size: usize) -> Option<usize> {
        // A zero-byte allocation is legal in ORT's API and must return a distinct, non-null,
        // freeable handle — returning null would be read as failure. Give it a whole page.
        let requested = size;
        let padded = size.max(1).next_multiple_of(SPAN_GRANULARITY);

        let mut inner = self.inner.lock().ok()?;

        // Prefer an exact-size retired span before growing the arena. Exact size only: reusing a
        // larger span for a smaller request would let `base + n` stay in-span past the requested
        // end, which is precisely the out-of-bounds arithmetic the guard band exists to catch.
        let reused = inner
            .free_list
            .get_mut(&padded)
            .and_then(|v| v.pop())
            .inspect(|_| {
                inner.free_list.retain(|_, v| !v.is_empty());
            });

        let base = match reused {
            Some(b) => b,
            None => {
                let b = inner.cursor;
                let next = b.checked_add(padded)?.checked_add(GUARD_BYTES)?;
                if next > self.arena_end() {
                    log::error!(
                        "VulkanExecutionProvider: device handle arena exhausted ({} MiB reserved, \
                         {} MiB carved). Raise {ENV_RESERVATION_MIB}, or lower \
                         {ENV_QUARANTINE_SPANS} if quarantine is holding the space. Reporting an \
                         allocation failure to ORT, which is its normal out-of-memory path.",
                        self.arena.len >> 20,
                        (b - self.arena.base) >> 20
                    );
                    return None;
                }
                inner.cursor = next;
                inner.stats.arena_used_bytes = (next - self.arena.base) as u64;
                b
            }
        };

        inner.generation += 1;
        let generation = inner.generation;
        inner.spans.insert(
            base,
            Span {
                base,
                requested,
                padded,
                generation,
                live: true,
                buffer: None,
            },
        );
        inner.stats.total_allocations += 1;
        inner.stats.live_spans += 1;
        inner.stats.live_bytes += requested as u64;
        if inner.stats.live_bytes > inner.stats.high_water_bytes {
            inner.stats.high_water_bytes = inner.stats.live_bytes;
        }
        tally::on_alloc(requested as u64, inner.stats.live_bytes);
        drop(inner);
        self.try_attach_device_buffer(base, padded);
        Some(base)
    }

    /// Record which device this registry serves, so it can find a device-memory provider.
    pub fn set_device_index(&self, index: usize) {
        self.device_index.store(index, Ordering::Relaxed);
    }

    /// Which device this registry serves, or `usize::MAX` when it was never attributed.
    pub fn device_index(&self) -> usize {
        self.device_index.load(Ordering::Relaxed)
    }

    /// Whether device-backed allocation is switched on for this process.
    ///
    /// Off by default. It is a correctness-neutral change — host staging produces the same bytes —
    /// so the gate exists to keep a partially wired path out of everyone's way, not to hide a
    /// wrong answer.
    pub fn device_memory_requested() -> bool {
        std::env::var(ENV_DEVICE_MEMORY).is_ok_and(|v| v != "0" && !v.is_empty())
    }

    /// Give a freshly carved span a real `VkBuffer` if the engine can supply one.
    ///
    /// Failure is not an error: falling back to host staging is slower and correct, and
    /// `alloc_device_backed_spans` versus `alloc_staged_spans` reports which happened, so the
    /// distinction can never be lost in a log.
    fn try_attach_device_buffer(&self, base: usize, padded: usize) {
        if !Self::device_memory_requested() {
            return;
        }
        let idx = self.device_index.load(Ordering::Relaxed);
        if idx == usize::MAX {
            return;
        }
        // Stand the engine's provider up on first use. Idempotent, and its failure is cached, so a
        // machine with no Vulkan device pays for the attempt once rather than per tensor.
        crate::vk::host_device_memory::ensure_registered(idx);
        let Some(provider) = crate::engine::device_memory_provider(idx) else {
            return;
        };
        let Some(view) = provider.alloc(padded) else {
            return;
        };
        if self.attach_buffer(base, view).is_err() {
            // The span vanished between carving and attaching, which should be impossible; free
            // the buffer rather than leak it, and say so.
            provider.free(view);
            log::warn!(
                "VulkanExecutionProvider: could not attach a device buffer to handle 0x{base:x} \
                 immediately after allocating it. Falling back to host staging for this span."
            );
        }
    }

    /// Release a handle. Not an error to call with a stale or foreign pointer — that is logged
    /// loudly and ignored, because a double free must not take the host process down.
    pub fn free(&self, addr: usize) {
        let Ok(mut inner) = self.inner.lock() else {
            return;
        };
        let Some(span) = inner.spans.get_mut(&addr) else {
            drop(inner);
            self.failed_lookups.fetch_add(1, Ordering::Relaxed);
            tally::on_failed_lookup();
            log::error!(
                "VulkanExecutionProvider: Free(0x{addr:x}) — not the base of any device handle \
                 this allocator served. Either the caller freed an interior pointer (the base is \
                 what Alloc returned) or the pointer belongs to a different allocator. Ignoring; a \
                 bad free must not take the host down."
            );
            return;
        };
        if !span.live {
            let g = span.generation;
            drop(inner);
            self.failed_lookups.fetch_add(1, Ordering::Relaxed);
            tally::on_failed_lookup();
            log::error!(
                "VulkanExecutionProvider: double free of device handle 0x{addr:x} (freed at \
                 generation {g}). Caught by quarantine rather than corrupting a live tensor. \
                 Ignoring the second free."
            );
            return;
        }
        span.live = false;
        let requested = span.requested;
        // Hand the VkBuffer back now, not at retirement. Quarantine protects the *address*, which
        // is what detects a stale handle; holding 2 GB of device memory for the length of the
        // window would multiply peak VRAM by the quarantine depth and fail on any real model. Same
        // reasoning as the staging release below.
        let device_buffer = span.buffer.take();
        inner.stats.total_frees += 1;
        tally::on_free();
        inner.stats.live_spans = inner.stats.live_spans.saturating_sub(1);
        inner.stats.live_bytes = inner.stats.live_bytes.saturating_sub(requested as u64);
        // Staging is released here, not at retirement: at model scale it is real host memory and
        // holding it for the length of the quarantine window would multiply peak RSS by the
        // quarantine depth. The *address space* stays quarantined, which is what detects the
        // use-after-free; the bytes behind it do not need to.
        if let Some(st) = inner.staging.remove(&addr) {
            let n = st.layout.size() as u64;
            drop(st);
            inner.stats.staging_live_bytes = inner.stats.staging_live_bytes.saturating_sub(n);
        }
        inner.quarantine.push_back(addr);
        inner.stats.quarantined_spans = inner.quarantine.len() as u64;
        let retired_before = inner.stats.quarantine_retired;

        // Retire the oldest only once the window is full, so recent frees stay detectable.
        while inner.quarantine.len() > self.quarantine_limit {
            let Some(old) = inner.quarantine.pop_front() else {
                break;
            };
            if let Some(s) = inner.spans.remove(&old) {
                inner.free_list.entry(s.padded).or_default().push(s.base);
                inner.stats.quarantine_retired += 1;
            }
        }
        inner.stats.quarantined_spans = inner.quarantine.len() as u64;
        tally::on_quarantine(
            inner.stats.quarantined_spans,
            inner.stats.quarantine_retired - retired_before,
        );
        drop(inner);
        if let Some(view) = device_buffer {
            let idx = self.device_index.load(Ordering::Relaxed);
            if let Some(p) = crate::engine::device_memory_provider(idx) {
                p.free(view);
            }
        }
    }

    /// Resolve any address — including an interior one produced by ORT's planner doing
    /// `base + offset` — to its handle and offset.
    ///
    /// This is the function that makes the whole scheme work, and the reason handles are real
    /// address space rather than integers.
    pub fn resolve(&self, addr: usize) -> Result<Resolved, LookupError> {
        let r = self.resolve_inner(addr);
        if r.is_err() {
            self.failed_lookups.fetch_add(1, Ordering::Relaxed);
            tally::on_failed_lookup();
        }
        r
    }

    /// Resolve without counting a miss.
    ///
    /// For *classifying* a pointer — "is this one of mine?" — where a miss is the expected answer
    /// for every host pointer and for every handle belonging to another device. Counting those
    /// would make `failed_lookups` grow on a healthy run, and a diagnostic that is non-zero when
    /// nothing is wrong is a diagnostic people learn to ignore. `failed_lookups` is reserved for
    /// lookups that *should* have succeeded.
    pub fn classify(&self, addr: usize) -> Result<Resolved, LookupError> {
        self.resolve_inner(addr)
    }

    fn resolve_inner(&self, addr: usize) -> Result<Resolved, LookupError> {
        if addr < self.arena.base || addr >= self.arena_end() {
            return Err(LookupError::NotAHandle { addr });
        }
        let inner = self
            .inner
            .lock()
            .map_err(|_| LookupError::NotAHandle { addr })?;
        let Some((_, span)) = inner.spans.range(..=addr).next_back() else {
            return Err(LookupError::InGuardBand { addr });
        };
        let offset = addr - span.base;
        if offset >= span.requested.max(1) {
            // Inside the padding or the guard band: arithmetic ran past the end of the
            // allocation. Reported as its own cause, not as "unknown pointer".
            return Err(LookupError::InGuardBand { addr });
        }
        if !span.live {
            return Err(LookupError::Freed {
                addr,
                base: span.base,
                freed_at_generation: span.generation,
            });
        }
        Ok(Resolved {
            base: span.base,
            offset,
            size: span.requested,
            generation: span.generation,
            buffer: span.buffer,
        })
    }

    /// Attach the device memory behind a handle.
    ///
    /// The seam with `vk/alloc.rs`: Switch's session allocates a `GpuBuffer`, hands back the
    /// opaque [`BufferView`] token, and this records it. No `ash` type crosses into this file and
    /// no `sys::ort` type crosses into his.
    pub fn attach_buffer(&self, addr: usize, view: BufferView) -> Result<(), LookupError> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| LookupError::NotAHandle { addr })?;
        match inner.spans.get_mut(&addr) {
            Some(s) if s.live => {
                let first = s.buffer.is_none();
                s.buffer = Some(view);
                if first {
                    tally::on_device_backed();
                }
                Ok(())
            }
            Some(s) => Err(LookupError::Freed {
                addr,
                base: s.base,
                freed_at_generation: s.generation,
            }),
            None => Err(LookupError::NotAHandle { addr }),
        }
    }

    /// Host staging bytes behind a handle that has no device buffer, created on first use.
    ///
    /// Returns the base of a zeroed host allocation of exactly `padded` bytes for the span
    /// containing `addr`, or `None` if `addr` is not a live handle or the allocation fails.
    ///
    /// # Contract for the caller
    ///
    /// The returned pointer is valid until the handle is freed. Callers must therefore only use it
    /// while holding an `OrtValue` that keeps the handle alive — which is exactly the lifetime of a
    /// `CopyTensors` call, the only caller. It is deliberately not exposed beyond this crate's
    /// data-transfer path.
    ///
    /// See [`HostStaging`] for why staging exists at all, and why a run that uses it is a
    /// correctness run and never a performance one.
    pub(crate) fn staging_ptr(&self, addr: usize) -> Option<*mut u8> {
        let mut inner = self.inner.lock().ok()?;
        let span = inner.spans.get(&addr).filter(|s| s.live)?.clone();
        // A device buffer may be attached. Staging is still returned, and is still authoritative:
        // under the mirror model (see `transfer::Endpoint`) the device buffer is written on every
        // copy *into* the handle and never read back, precisely so that host staging and device
        // memory cannot disagree. The earlier version of this function returned `None` here to
        // stop staging shadowing a device buffer. That was right when the plan was for device
        // memory to be the tensor's only home — and it was measured wrong the moment device
        // backing was switched on: `vk::session` reads inputs through `host_backing_for`, got no
        // host address, and ORT failed the model at weight deserialisation on both vendors.
        if span.buffer.is_some() && inner.stats.staging_spans == 0 {
            log::info!(
                "VulkanExecutionProvider: handle 0x{addr:x} has a VkBuffer and is also \
                 host-staged. The staging block is authoritative; the device buffer is a mirror \
                 written on every copy in. alloc_device_authoritative_spans stays 0 until the \
                 engine binds `transfer::device_buffer_for` instead of re-uploading its own \
                 buffers."
            );
        }
        if let Some(existing) = inner.staging.get(&addr) {
            return Some(existing.ptr);
        }
        let layout = std::alloc::Layout::from_size_align(span.padded.max(1), 64).ok()?;
        // SAFETY: `layout` has a non-zero size (`padded` is at least `SPAN_GRANULARITY`) and a
        // valid power-of-two alignment, which is `alloc_zeroed`'s requirement.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            log::error!(
                "VulkanExecutionProvider: could not allocate {} B of host staging for device \
                 handle 0x{addr:x}. Reporting a copy failure to ORT.",
                layout.size()
            );
            return None;
        }
        if inner.stats.staging_spans == 0 && span.buffer.is_none() {
            // Scoped deliberately to *this handle*. The previous wording ended "any timing from
            // this run is a host measurement", which was true while nothing was device-backed and
            // becomes false in the dangerous direction the moment some allocations are: a run that
            // is 99% on device would still print it, and a warning that overstates gets ignored,
            // which is how it stops protecting the 1% case it was written for. The whole-run
            // claim is now made at teardown by `staging_verdict`, where the ratio is known, and
            // asserted by `epctl --check-counters --require-device-memory`, which does not rely
            // on anyone remembering a log line. See D-T69.
            log::warn!(
                "VulkanExecutionProvider: no VkBuffer is attached to device handle 0x{addr:x}, so \
                 the contents of THIS handle are being held in HOST memory. This is the near half \
                 of the real copy path and it produces correct results, but nothing device-side \
                 backs this tensor. Whether that is true of the whole run is reported at teardown \
                 — see the staging verdict there, not this line."
            );
        }
        inner.staging.insert(addr, HostStaging { ptr, layout });
        inner.stats.staging_spans += 1;
        inner.stats.staging_live_bytes += layout.size() as u64;
        tally::on_staging(layout.size() as u64);
        Some(ptr)
    }

    pub fn stats(&self) -> AllocStats {
        let mut s = self
            .inner
            .lock()
            .map(|i| i.stats)
            .unwrap_or_else(|e| e.into_inner().stats);
        s.failed_lookups = self.failed_lookups.load(Ordering::Relaxed);
        s
    }

    /// One line for a teardown log — the same reasoning as the execution counters: a number nobody
    /// prints is a number nobody checks.
    pub fn summary(&self) -> String {
        let s = self.stats();
        let staging = if s.staging_spans == 0 {
            String::new()
        } else {
            format!(
                "; HOST STAGING: {} span(s) ever, {} B live — tensors on those handles never \
                 reached device memory",
                s.staging_spans, s.staging_live_bytes
            )
        };
        format!(
            "device handles: {} allocation(s), {} free(s), {} live ({} B); high-water {} B; \
             quarantine {} span(s), {} retired; {} failed lookup(s); arena {} MiB reserved, {} MiB \
             carved{staging}",
            s.total_allocations,
            s.total_frees,
            s.live_spans,
            s.live_bytes,
            s.high_water_bytes,
            s.quarantined_spans,
            s.quarantine_retired,
            s.failed_lookups,
            s.arena_bytes >> 20,
            s.arena_used_bytes >> 20,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// The allocation tally
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// A process-global tally of what the allocator actually did, readable after teardown.
///
/// # Why this is separate from [`AllocStats`]
///
/// `AllocStats` belongs to a `HandleRegistry`, and a registry belongs to a `VulkanAllocator` that
/// ORT releases on its own schedule. Its numbers are therefore gone by the time anyone can read
/// them, and the teardown *ordering* between `ReleaseAllocator` and `ReleaseDataTransfer` — where
/// the counters file is written — is ORT's business, not ours. A snapshot published at release
/// would be correct only when the order happened to favour us, and would silently write zeros
/// otherwise. That is precisely the class of check that reports a clean result because it did not
/// run, which this project has now been bitten by three times.
///
/// So these are updated as the events happen. They are monotonic across every allocator in the
/// process, which is the right shape for the question they answer: *did this run put tensors in
/// device memory, or in host memory wearing a device handle?*
///
/// # Why `staged_spans` deserves to be a counter rather than a log line
///
/// The staging path prints a one-shot WARN saying any timing from the run is a host measurement.
/// A warning is read by a human who is present, remembers it an hour later, and is honest when
/// quoting the number. A counter is read by `epctl --check-counters`, which is none of those
/// things and does not need to be. See D-T69.
pub mod tally {
    use std::sync::atomic::{AtomicU64, Ordering};

    static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
    static FREES: AtomicU64 = AtomicU64::new(0);
    static BYTES: AtomicU64 = AtomicU64::new(0);
    static HIGH_WATER: AtomicU64 = AtomicU64::new(0);
    static DEVICE_BACKED: AtomicU64 = AtomicU64::new(0);
    static STAGED_SPANS: AtomicU64 = AtomicU64::new(0);
    static STAGED_BYTES: AtomicU64 = AtomicU64::new(0);
    static ALLOCATORS_RELEASED: AtomicU64 = AtomicU64::new(0);
    static ALLOCATORS_LIVE: AtomicU64 = AtomicU64::new(0);
    static FREES_AFTER_RELEASE: AtomicU64 = AtomicU64::new(0);
    static LIVE_AT_RELEASE_SPANS: AtomicU64 = AtomicU64::new(0);
    static LIVE_AT_RELEASE_BYTES: AtomicU64 = AtomicU64::new(0);
    /// Deepest the quarantine FIFO has ever been, across every registry in the process.
    static QUARANTINE_PEAK: AtomicU64 = AtomicU64::new(0);
    /// Spans retired from quarantine and returned to service, process-wide.
    ///
    /// This exists because "the quarantine never fired" was being claimed from a number that was
    /// never written to the counters file: [`AllocStats::quarantine_retired`] is per-registry and
    /// only reachable through ORT's `GetStats` KVPs, which nothing in the harness calls. Under R7
    /// an unreachable instrument is not a negative result. Non-zero here means the detection window
    /// was exhausted, so `pointers_use_after_free == 0` stops being evidence of anything.
    static QUARANTINE_RETIRED: AtomicU64 = AtomicU64::new(0);
    /// Lookups that failed, process-wide.
    ///
    /// Per-registry `AllocStats::failed_lookups` says of itself "non-zero always indicates a bug
    /// somewhere" — and it was reachable **only** through ORT's `GetStats` KVPs, which nothing in
    /// this repo calls. A bug detector whose output no artifact carries has the same evidentiary
    /// value as one that was never written. Same class as `quarantine_retired`; found by the same
    /// sweep.
    static FAILED_LOOKUPS: AtomicU64 = AtomicU64::new(0);

    pub(super) fn on_failed_lookup() {
        FAILED_LOOKUPS.fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn on_quarantine(depth: u64, retired: u64) {
        QUARANTINE_PEAK.fetch_max(depth, Ordering::Relaxed);
        if retired > 0 {
            QUARANTINE_RETIRED.fetch_add(retired, Ordering::Relaxed);
        }
    }

    pub(super) fn on_alloc(requested: u64, live_bytes: u64) {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(requested, Ordering::Relaxed);
        HIGH_WATER.fetch_max(live_bytes, Ordering::Relaxed);
    }

    pub(super) fn on_free() {
        FREES.fetch_add(1, Ordering::Relaxed);
        // Scope, carefully. The first cut of this counter tested `ALLOCATORS_RELEASED > 0`, which
        // is monotone — so after any one allocator went away, every subsequent Free from every
        // *other* live allocator counted as late. On the real model under pytest that reported
        // 2508 late frees on a run where nothing was wrong. That is the same scope error as the
        // still-live-handles warning this counter was written to replace, reproduced inside its
        // own replacement. The condition that means something is a Free arriving when **no**
        // allocator is live: only then is there nobody left who could legitimately own the span.
        if ALLOCATORS_LIVE.load(Ordering::Relaxed) == 0
            && ALLOCATORS_RELEASED.load(Ordering::Relaxed) > 0
        {
            FREES_AFTER_RELEASE.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// An `OrtAllocator` of ours was handed to ORT.
    pub(super) fn on_allocator_created() {
        ALLOCATORS_LIVE.fetch_add(1, Ordering::Relaxed);
    }

    /// An `OrtAllocator` of ours was released, with this many spans still live in the *shared*
    /// registry — see [`leak_verdict`] for why that number is not attributable to this allocator.
    pub(super) fn on_allocator_released(live_spans: u64, live_bytes: u64) {
        ALLOCATORS_RELEASED.fetch_add(1, Ordering::Relaxed);
        ALLOCATORS_LIVE
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
                Some(n.saturating_sub(1))
            })
            .ok();
        LIVE_AT_RELEASE_SPANS.fetch_add(live_spans, Ordering::Relaxed);
        LIVE_AT_RELEASE_BYTES.fetch_max(live_bytes, Ordering::Relaxed);
    }

    /// A `VkBuffer` was attached to a handle for the first time: this one is genuinely on device.
    pub(super) fn on_device_backed() {
        DEVICE_BACKED.fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn on_staging(bytes: u64) {
        STAGED_SPANS.fetch_add(1, Ordering::Relaxed);
        STAGED_BYTES.fetch_add(bytes, Ordering::Relaxed);
    }

    static DEVICE_UPLOADS: AtomicU64 = AtomicU64::new(0);
    static DEVICE_UPLOAD_BYTES: AtomicU64 = AtomicU64::new(0);
    static DEVICE_DOWNLOADS: AtomicU64 = AtomicU64::new(0);
    static DEVICE_DOWNLOAD_BYTES: AtomicU64 = AtomicU64::new(0);
    /// 0 = never asked, 1 = unified (UMA), 2 = discrete.
    static UNIFIED_MEMORY: AtomicU64 = AtomicU64::new(0);
    /// Spans whose only home is device memory. See [`Tally::device_authoritative_spans`]. Nothing
    /// increments this yet, and that is the point: it is the claim's falsifier, not a placeholder.
    static DEVICE_AUTHORITATIVE: AtomicU64 = AtomicU64::new(0);
    /// Times the engine asked for one of our device buffers via `transfer::device_buffer_for`.
    ///
    /// The *only* way an engine can compute from our device memory is to bind a buffer this
    /// function handed it. So this is the independent falsifier for
    /// [`Tally::device_authoritative_spans`]: authoritative spans with zero binds is a
    /// contradiction, not a nuance. It exists **before** the claim it guards, because a counter
    /// wired at the same time as the feature it measures is wired by someone who already believes
    /// the answer.
    static DEVICE_BUFFER_BINDS: AtomicU64 = AtomicU64::new(0);

    /// The engine bound one of our device buffers. Called from `transfer::device_buffer_for`.
    pub fn on_device_buffer_bind() {
        DEVICE_BUFFER_BINDS.fetch_add(1, Ordering::Relaxed);
    }

    /// A span became device-authoritative: device-resident with **no** host staging block behind
    /// it, so nothing can read it through `host_backing_for`.
    ///
    /// The single documented increment point, deliberately narrow. Whoever wires persistent
    /// residency must satisfy two independent measured guards or [`staging_verdict`] will call the
    /// number dishonest in the artifact itself:
    ///
    /// 1. `device_authoritative_spans <= device_backed_spans - staged_spans` — a span that still
    ///    has host staging is a mirror, whatever the design intends.
    /// 2. `device_buffer_binds > 0` — the engine must actually have asked for the buffer.
    pub fn on_device_authoritative() {
        DEVICE_AUTHORITATIVE.fetch_add(1, Ordering::Relaxed);
    }

    /// A `CopyTensors` endpoint was in device memory and went through the provider.
    ///
    /// This is the instrument that goes red if `device_backed_spans > 0` were ever an accounting
    /// change rather than a change in where bytes live: a span cannot be device-backed and also
    /// have its contents move without one of these firing.
    pub fn on_device_copy(bytes: u64, upload: bool) {
        if upload {
            DEVICE_UPLOADS.fetch_add(1, Ordering::Relaxed);
            DEVICE_UPLOAD_BYTES.fetch_add(bytes, Ordering::Relaxed);
        } else {
            DEVICE_DOWNLOADS.fetch_add(1, Ordering::Relaxed);
            DEVICE_DOWNLOAD_BYTES.fetch_add(bytes, Ordering::Relaxed);
        }
    }

    /// Record whether the device we are backing spans on has unified memory.    ///
    /// Reported alongside every device-backed number, because on a UMA part "device-local" and
    /// "host" are the same DRAM: a device-backed count there does not mean what the identical
    /// count means on a discrete card, and the two must never be averaged or compared.
    pub fn set_unified_memory(unified: bool) {
        UNIFIED_MEMORY.store(if unified { 1 } else { 2 }, Ordering::Relaxed);
    }

    /// Everything the counters file reports, taken together so the numbers are mutually consistent
    /// enough to reason about. They are not sampled atomically as a group; at teardown, when this
    /// is read, nothing is still mutating them.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub struct Tally {
        pub allocations: u64,
        pub frees: u64,
        pub bytes: u64,
        pub high_water_bytes: u64,
        pub device_backed_spans: u64,
        pub staged_spans: u64,
        pub staged_bytes: u64,
        pub allocators_released: u64,
        pub allocators_live: u64,
        pub frees_after_release: u64,
        pub live_at_release_spans: u64,
        pub live_at_release_bytes: u64,
        pub device_uploads: u64,
        pub device_upload_bytes: u64,
        pub device_downloads: u64,
        pub device_download_bytes: u64,
        /// 0 = unknown, 1 = unified (UMA), 2 = discrete.
        pub unified_memory: u64,
        /// Spans whose **only** home is device memory.
        ///
        /// Deliberately separate from `device_backed_spans`, and deliberately zero today. A
        /// device-backed span still keeps its host staging block, because `vk::session` reads
        /// every kernel input through `transfer::host_backing_for` and binds buffers it allocated
        /// itself — with no host address the EP fails the dispatch and ORT falls back to CPU
        /// (measured, both vendors). So `device_backed_spans > 0` means "these bytes are also
        /// resident in device memory and really crossed the bus", not "the EP computes from
        /// device memory". This counter is the one that would have to move for the second claim,
        /// and it is the instrument that goes red if anyone states it while it is still 0.
        pub device_authoritative_spans: u64,
        /// The most spans that *could* be authoritative, measured: `device_backed - staged`.
        /// A span with a host staging block is a mirror however the design describes it.
        pub device_authoritative_ceiling: u64,
        /// Times the engine bound one of our device buffers via `transfer::device_buffer_for`.
        /// Zero while `device_authoritative_spans > 0` is a contradiction.
        pub device_buffer_binds: u64,
        /// Lookups that failed anywhere in the process. Non-zero always indicates a bug.
        pub failed_lookups: u64,
        /// Deepest the quarantine FIFO ever got, process-wide.
        pub quarantine_peak_spans: u64,
        /// Spans retired from quarantine, process-wide. **Non-zero voids** any conclusion drawn
        /// from `pointers_use_after_free == 0`, because the window that would have caught a stale
        /// handle was reused before the handle could be presented.
        pub quarantine_retired: u64,
    }

    pub fn snapshot() -> Tally {
        Tally {
            allocations: ALLOCATIONS.load(Ordering::Relaxed),
            frees: FREES.load(Ordering::Relaxed),
            bytes: BYTES.load(Ordering::Relaxed),
            high_water_bytes: HIGH_WATER.load(Ordering::Relaxed),
            device_backed_spans: DEVICE_BACKED.load(Ordering::Relaxed),
            staged_spans: STAGED_SPANS.load(Ordering::Relaxed),
            staged_bytes: STAGED_BYTES.load(Ordering::Relaxed),
            allocators_released: ALLOCATORS_RELEASED.load(Ordering::Relaxed),
            allocators_live: ALLOCATORS_LIVE.load(Ordering::Relaxed),
            frees_after_release: FREES_AFTER_RELEASE.load(Ordering::Relaxed),
            live_at_release_spans: LIVE_AT_RELEASE_SPANS.load(Ordering::Relaxed),
            live_at_release_bytes: LIVE_AT_RELEASE_BYTES.load(Ordering::Relaxed),
            device_uploads: DEVICE_UPLOADS.load(Ordering::Relaxed),
            device_upload_bytes: DEVICE_UPLOAD_BYTES.load(Ordering::Relaxed),
            device_downloads: DEVICE_DOWNLOADS.load(Ordering::Relaxed),
            device_download_bytes: DEVICE_DOWNLOAD_BYTES.load(Ordering::Relaxed),
            unified_memory: UNIFIED_MEMORY.load(Ordering::Relaxed),
            device_authoritative_spans: DEVICE_AUTHORITATIVE.load(Ordering::Relaxed),
            device_authoritative_ceiling: DEVICE_BACKED
                .load(Ordering::Relaxed)
                .saturating_sub(STAGED_SPANS.load(Ordering::Relaxed)),
            device_buffer_binds: DEVICE_BUFFER_BINDS.load(Ordering::Relaxed),
            failed_lookups: FAILED_LOOKUPS.load(Ordering::Relaxed),
            quarantine_peak_spans: QUARANTINE_PEAK.load(Ordering::Relaxed),
            quarantine_retired: QUARANTINE_RETIRED.load(Ordering::Relaxed),
        }
    }

    /// Whether spans still live at allocator release are a leak of ours or a lifetime we do not own.
    ///
    /// # The answer, measured: neither. It was a scope error in the instrument.
    ///
    /// The original warning said: *"ORT frees what it allocated, so this is either a leak on our
    /// side or a tensor the session outlived."* That is honest and useless. An open disjunction in
    /// a warning is a decision deferred to whoever reads it, which in practice means it is decided
    /// later, by someone with less context, under worse conditions — or never, because a warning
    /// that has always been there reads as furniture. 2.09 GB is not furniture.
    ///
    /// Measured on the real 2.2 GB model under pytest, both vendors:
    /// **`alloc_allocations` 2511, `alloc_frees` 2511.** ORT hands back every span. There is no
    /// leak, and there never was. Three things were true at once and only their combination looked
    /// alarming:
    ///
    /// 1. [`HandleRegistry`] is **process-global per device** (`factory::REGISTRIES`), shared by
    ///    every allocator and data transfer for that device. `registry.stats().live_spans` read at
    ///    one allocator's release therefore counts spans that *other, still-running sessions* own.
    ///    The warning named this allocator as the owner of memory belonging to its neighbours.
    ///    That run released seven allocators; summed, they reported 1257 "still live" spans that
    ///    were all subsequently freed.
    /// 2. The registry **outlives every allocator by construction**, because `REGISTRIES` holds an
    ///    `Arc` for the process lifetime. So the hazard the warning gestured at — a late `Free`
    ///    landing in a torn-down registry — cannot occur here. Reserved address space and staging
    ///    are not reclaimed at allocator release at all.
    /// 3. The counters file was written from `VulkanDataTransfer::release`, which ORT calls
    ///    *before* releasing allocators, so the file could never have shown any of this. It is now
    ///    rewritten at allocator release too.
    ///
    /// So the warning is now scoped: it is `debug!` when other holders of the shared registry are
    /// still running, and `warn!` only for the last allocator on a device, where "still live"
    /// finally means what it says.
    ///
    /// # The instrument that goes red if this benign reading is wrong
    ///
    /// `frees_after_release`, and its own first version was wrong in exactly the way described
    /// above — it tested `released > 0`, which is monotone, and so counted every `Free` from every
    /// still-live allocator once any one allocator had gone. It reported 2508 late frees on a
    /// healthy run. It now requires `allocators_live == 0`: a `Free` arriving when no allocator of
    /// ours exists is a span nobody could legitimately still own. Reported as
    /// `alloc_frees_after_release` so a lane asserts on it instead of a human noticing a log line.
    pub fn leak_verdict() -> String {
        let t = snapshot();
        if t.frees_after_release > 0 {
            return format!(
                "{} Free call(s) arrived while no allocator of ours was live, so ORT still \
                 believed it owned spans nobody was left to own. Investigate before quoting any \
                 memory number from this run.",
                t.frees_after_release
            );
        }
        format!(
            "{} allocation(s) and {} free(s) this process: ORT returned {}. No Free has arrived \
             while no allocator was live (alloc_frees_after_release = 0, {} release(s)). The \
             reserved address space and staging behind these handles belong to the per-device \
             registry, which outlives every allocator, so nothing is reclaimed here and nothing \
             is lost. This line is informational; it becomes a defect report when \
             alloc_frees_after_release is non-zero or allocations and frees disagree at exit.",
            t.allocations,
            t.frees,
            if t.frees >= t.allocations {
                "all of them"
            } else {
                "fewer than it took"
            },
            t.allocators_released
        )
    }

    /// What this run's memory actually was, stated from the numbers rather than from a static
    /// string that was true when it was written.
    ///
    /// # Why this is computed rather than fixed
    ///
    /// The staging WARN used to end "any timing from this run is a host measurement". That was
    /// true while *nothing* was device-backed. As real `VkBuffer`s arrive behind handles it goes
    /// wrong in the dangerous direction, and it goes wrong twice:
    ///
    /// * **Leave it and it over-warns.** A run that is 99% device-backed still prints "host
    ///   measurement", so the warning is wrong, readers learn to discount it, and it stops
    ///   protecting the 1% case it exists for. A caveat that is always printed carries no
    ///   information.
    /// * **Delete it and it under-warns.** Staging does not stop the day device memory lands; it
    ///   stops per-allocation, and a partially staged run would then report nothing at all. That
    ///   is the worse failure, because the number it silently blesses looks like a device
    ///   measurement.
    ///
    /// Neither branch is survivable as prose, so the verdict is derived: it names the ratio, and
    /// there is a distinct sentence for the mixed state that no fixed wording covers today and
    /// that is exactly the state we are heading into.
    /// What `vk::session` staged this run, which no `alloc_*` counter can see.
    ///
    /// Derived, not declared (R7): when the tracer is inert there are no observations, and this
    /// says so rather than reporting zero bytes — "no instrument" and "no traffic" are different
    /// claims and only one of them is good news.
    fn session_staging_sentence() -> String {
        let (up_n, up_b, rb_n, rb_b, up_us, rb_us) = crate::trace::tracer().transfer_totals();
        if up_n == 0 && rb_n == 0 {
            return " SESSION STAGING: NOT MEASURED — the tracer recorded no staging copies this \
                    run. That is an absent instrument, not zero traffic: `vk::session` stages \
                    every kernel input on every inference regardless. Set \
                    ONNXRUNTIME_EP_VULKAN_TRACE to a path to measure it."
                .to_string();
        }
        format!(
            " SESSION STAGING (separate from every alloc_* number above, and normally much \
             larger): vk::session performed {up_n} host->device staging copy/copies totalling \
             {:.1} MiB in {:.1} ms, and {rb_n} readback(s) totalling {:.1} MiB in {:.1} ms. This \
             is per-inference traffic and it repeats on every run; the allocator's bytes are \
             allocated once. If the upload MiB is close to alloc_high_water_bytes, the whole \
             weight set is being re-staged every inference and THAT, not span residency, is where \
             the time goes.",
            up_b as f64 / (1024.0 * 1024.0),
            up_us as f64 / 1000.0,
            rb_b as f64 / (1024.0 * 1024.0),
            rb_us as f64 / 1000.0,
        )
    }

    /// Whether `device_authoritative_spans` is credible, checked against two counters it does not
    /// control.
    ///
    /// This counter is 0 today and will be the headline the moment persistent weight residency
    /// lands. The failure mode is not fraud, it is belief: an author wires it because the design
    /// says the span is device-resident. So the artifact audits it against measurements taken
    /// elsewhere — the staging tally, and the engine's own bind traffic — and says so in the same
    /// sentence that reports the number, where anyone quoting it must read it.
    fn authoritative_audit_sentence(t: &Tally) -> String {
        if t.device_authoritative_spans == 0 {
            // A zero must say WHICH zero it is. R7: absence of an instrument must not read as a
            // negative result. Without this sentence, "0 authoritative spans" is ambiguous between
            // "measured, nothing is device-authoritative" and "nobody ever incremented it", and the
            // second is what is actually true today.
            return format!(
                " alloc_device_authoritative_spans is 0, and this is an UNWIRED zero, not a \
                 measured one: its only increment point (allocator::tally::on_device_authoritative) \
                 has no production caller yet, and its independent falsifier \
                 alloc_device_buffer_binds is {} — the engine has {} bound one of our device \
                 buffers via transfer::device_buffer_for. The measured ceiling for it on this run \
                 is {} span(s) (device_backed {} - staged {}). When residency lands, the count and \
                 the binds must move together; a non-zero count with zero binds is reported here as \
                 not credible.",
                t.device_buffer_binds,
                if t.device_buffer_binds == 0 {
                    "never"
                } else {
                    "sometimes"
                },
                t.device_authoritative_ceiling,
                t.device_backed_spans,
                t.staged_spans
            );
        }
        let mut faults = Vec::new();
        if t.device_authoritative_spans > t.device_authoritative_ceiling {
            faults.push(format!(
                "it claims {} authoritative span(s) but only {} span(s) are device-backed WITHOUT \
                 host staging ({} backed - {} staged); a span with a staging block is a mirror \
                 however the design describes it",
                t.device_authoritative_spans,
                t.device_authoritative_ceiling,
                t.device_backed_spans,
                t.staged_spans
            ));
        }
        if t.device_buffer_binds == 0 {
            faults.push(
                "the engine never called transfer::device_buffer_for, so it never bound one of \
                 our device buffers and cannot have computed from one"
                    .to_string(),
            );
        }
        if faults.is_empty() {
            return format!(
                " alloc_device_authoritative_spans is {}, and it survives both independent checks: \
                 it is within the measured no-staging ceiling of {}, and the engine bound our \
                 device buffers {} time(s).",
                t.device_authoritative_spans,
                t.device_authoritative_ceiling,
                t.device_buffer_binds
            );
        }
        format!(
            " *** alloc_device_authoritative_spans IS NOT CREDIBLE AND MUST NOT BE QUOTED: {}. \
             Fix the counter or the wiring before this number leaves the machine. ***",
            faults.join("; and ")
        )
    }

    pub fn staging_verdict() -> String {
        let t = snapshot();
        if t.allocations == 0 {
            return "no device handles were allocated in this run, so there is nothing to say \
                    about where their contents lived"
                .to_string();
        }
        // The traffic this verdict CANNOT see, stated before any of the traffic it can.
        //
        // (Tank, 2026-07-30) Every branch below describes only spans ORT asked THIS allocator to
        // allocate. It is silent about `vk::session`'s per-inference staging copy of each kernel
        // input, which on Phi-3.5 measured 1997.6 MiB PER INFERENCE — within 0.02% of the entire
        // 1997.2 MiB weight set this allocator holds, i.e. the whole model is re-staged on every
        // single run — against 0.8 MiB read back. On the discrete card that copy is a median 94.8%
        // of the EP's wall time. A verdict that said "MIRRORED — all spans are device-resident"
        // and stopped there described residency truthfully while omitting the dominant staging
        // cost, which is exactly the "reports what was true when it was written" failure.
        let session = session_staging_sentence();
        let authoritative_audit = authoritative_audit_sentence(&t);
        let mem = match t.unified_memory {
            1 => {
                " The device is UNIFIED-MEMORY (UMA): its device-local heap is the same DRAM as \
                 host memory, so a device-backed span here has not crossed a bus and this number \
                 must never be compared with a discrete card's."
            }
            2 => " The device is DISCRETE: device-backed means across the bus.",
            _ => {
                " Whether the device has unified memory was never recorded, so device-backed here \
                 cannot be read as either UMA-local or across-a-bus."
            }
        };
        let moved = format!(
            " Bytes actually moved through the device path: {} upload(s) ({} B), {} download(s) \
             ({} B).{}",
            t.device_uploads,
            t.device_upload_bytes,
            t.device_downloads,
            t.device_download_bytes,
            if t.device_backed_spans > 0 && t.device_authoritative_spans == 0 {
                " NOTE: alloc_device_authoritative_spans is 0 — every device-backed span also \
                 keeps host staging, which remains authoritative, because the compute session \
                 still reads inputs through host_backing_for and binds its own buffers. These \
                 bytes are resident in device memory and really crossed the bus; the EP does not \
                 yet compute from them. Do not quote this as 'running on device memory'."
            } else {
                ""
            }
        );
        if t.device_backed_spans > 0
            && t.device_backed_spans == t.allocations
            && t.staged_spans == t.allocations
        {
            return format!(
                "MEMORY: MIRRORED — all {} span(s) have BOTH a VkBuffer in device memory and a \
                 host staging block ({} B). The staging block is authoritative and the device \
                 buffer is written on every copy in, so the two cannot disagree. {} upload(s) \
                 ({} B) really crossed to device memory. But \
                 alloc_device_authoritative_spans is 0: the compute session still reads inputs \
                 through host_backing_for and binds buffers it allocated itself, so this run's \
                 timing is a HOST measurement PLUS the cost of mirroring, and is worse than \
                 staging alone rather than better. It must not be quoted as a device-memory \
                 measurement.{mem}{session}{authoritative_audit}",
                t.allocations, t.staged_bytes, t.device_uploads, t.device_upload_bytes,
            );
        }
        if t.staged_spans == 0 && t.device_backed_spans == 0 {
            // Neither counter moved, yet spans were allocated. This is R7: the branch below
            // ("none were host-staged") would read as reassurance, when the truth is that
            // nothing observed where these bytes lived. `staged_spans == 0` only means
            // "not staged" if something was in a position to record staging.
            return format!(
                "MEMORY: UNMEASURED — {} span(s) were allocated but BOTH alloc_staged_spans and \
                 alloc_device_backed_spans are 0, so nothing recorded where their contents \
                 lived. This is an absent instrument, not a clean result, and no timing from \
                 this run may be described as either a host or a device measurement.{session}{authoritative_audit}",
                t.allocations
            );
        }
        if t.staged_spans == 0 {
            return format!(
                "MEMORY: none of the {} device handle(s) were host-staged; {} had a VkBuffer \
                 attached. Timing from this run is not disqualified by staging (which is not the \
                 same as being a good measurement).{mem}{moved}{session}{authoritative_audit}",
                t.allocations, t.device_backed_spans
            );
        }
        if t.device_backed_spans == 0 {
            return format!(
                "MEMORY: ALL {} host-staged span(s) ({} B) and ZERO device-backed — nothing in \
                 this run reached device memory. Any timing from it is a HOST measurement and must \
                 not be quoted as anything else.{session}{authoritative_audit}",
                t.staged_spans, t.staged_bytes
            );
        }
        format!(
            "MEMORY: MIXED — {} span(s) ({} B) were host-staged while {} had a VkBuffer attached, \
             out of {} allocation(s) ({:.1}% device-backed). A timing from this run is neither a \
             host measurement nor a device one; it is an average over two different memories and \
             is not comparable with either. Assert `alloc_staged_spans == 0` with `epctl \
             --check-counters --require-device-memory` before quoting a number from a run like \
             this.{mem}{moved}{session}{authoritative_audit}",
            t.staged_spans,
            t.staged_bytes,
            t.device_backed_spans,
            t.allocations,
            100.0 * t.device_backed_spans as f64 / t.allocations as f64,
        )
    }

    #[doc(hidden)]
    pub fn reset_for_test() {
        for c in [
            &ALLOCATIONS,
            &FREES,
            &BYTES,
            &HIGH_WATER,
            &DEVICE_BACKED,
            &STAGED_SPANS,
            &STAGED_BYTES,
            &QUARANTINE_PEAK,
            &QUARANTINE_RETIRED,
            &FAILED_LOOKUPS,
            &DEVICE_AUTHORITATIVE,
            &DEVICE_BUFFER_BINDS,
        ] {
            c.store(0, Ordering::Relaxed);
        }
    }

    /// Drive the counters directly so every `staging_verdict` branch — including the genuinely
    /// mixed one, which real hardware has never produced — can be exercised.
    #[doc(hidden)]
    pub fn seed_for_test(allocations: u64, device_backed: u64, staged: u64, staged_bytes: u64) {
        ALLOCATIONS.store(allocations, Ordering::Relaxed);
        DEVICE_BACKED.store(device_backed, Ordering::Relaxed);
        STAGED_SPANS.store(staged, Ordering::Relaxed);
        STAGED_BYTES.store(staged_bytes, Ordering::Relaxed);
    }

    /// Drive the two counters that guard `device_authoritative_spans`, so the audit can be
    /// exercised before the feature that will produce them exists. A guard first tested by the
    /// author of the feature it guards is tested by someone who already believes the answer.
    #[doc(hidden)]
    pub fn seed_authoritative_for_test(authoritative: u64, binds: u64) {
        DEVICE_AUTHORITATIVE.store(authoritative, Ordering::Relaxed);
        DEVICE_BUFFER_BINDS.store(binds, Ordering::Relaxed);
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// The pointer-observation ledger
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// What ORT actually does with the pointers we hand it.
///
/// The reserved-address-space design argues that `ptr + n` stays in-span *by construction*. That
/// is an argument, not an observation, and an argument has never met a real planner. This ledger
/// is the instrument that turns it into a measurement: every pointer ORT hands back to us is
/// classified against the registry and tallied here, so the question "did the planner do
/// arithmetic on our handles?" has a number rather than an inference.
///
/// **What it observes, precisely.** Pointers at the boundary where they come *back* to us — both
/// endpoints of every `CopyTensors`, and every `GetTensorMutableData` result the engine resolves
/// through [`crate::transfer::host_backing_for`]. It does **not** see arithmetic ORT performs
/// internally and never shows us. A zero here therefore means "ORT never handed us a derived
/// pointer", not "ORT never computed one". That distinction is the whole reason this is worth
/// writing down rather than asserting.
///
/// The taxonomy is [`LookupError`]'s, because those failures demand different fixes:
/// * **base** — the pointer we returned, unmodified.
/// * **interior** — `base + n`, in-span. This is the planner-arithmetic case the design defends.
/// * **guard band** — in the arena but between spans. Arithmetic that ran off the end. A real bug,
///   caught rather than faulted.
/// * **freed** — a stale handle. This is the quarantine detector firing, and it is the *only*
///   evidence that would prove the quarantine under a real allocation pattern.
/// * **host** — not ours, the expected answer for most pointers.
pub mod ledger {
    use super::LookupError;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

    const ORD: Ordering = Ordering::Relaxed;

    static OBSERVED: AtomicU64 = AtomicU64::new(0);
    static HOST: AtomicU64 = AtomicU64::new(0);
    static AT_BASE: AtomicU64 = AtomicU64::new(0);
    static INTERIOR: AtomicU64 = AtomicU64::new(0);
    static IN_GUARD_BAND: AtomicU64 = AtomicU64::new(0);
    static USE_AFTER_FREE: AtomicU64 = AtomicU64::new(0);
    static MAX_OFFSET: AtomicUsize = AtomicUsize::new(0);

    /// The first few observations, verbatim. A tally says how many; a trace says what, and the
    /// unanticipated-arithmetic finding the coordinator asked about would show up in the shape of
    /// individual entries rather than in a count.
    static TRACE: Mutex<Vec<String>> = Mutex::new(Vec::new());
    const TRACE_LIMIT: usize = 64;

    fn trace(line: String) {
        if let Ok(mut t) = TRACE.lock() {
            if t.len() < TRACE_LIMIT {
                t.push(line);
            }
        }
    }

    /// Record one pointer ORT handed back, already classified.
    pub fn observe(addr: usize, outcome: &Result<super::Resolved, LookupError>) {
        OBSERVED.fetch_add(1, ORD);
        match outcome {
            Ok(r) if r.offset == 0 => {
                AT_BASE.fetch_add(1, ORD);
            }
            Ok(r) => {
                INTERIOR.fetch_add(1, ORD);
                MAX_OFFSET.fetch_max(r.offset, ORD);
                trace(format!(
                    "INTERIOR 0x{addr:x} = handle 0x{:x} + {} (span {} B) — the planner did \
                     arithmetic on one of our handles",
                    r.base, r.offset, r.size
                ));
            }
            Err(LookupError::InGuardBand { .. }) => {
                IN_GUARD_BAND.fetch_add(1, ORD);
                trace(format!(
                    "GUARD BAND 0x{addr:x} — arithmetic ran off the end of an allocation"
                ));
            }
            Err(LookupError::Freed {
                base,
                freed_at_generation,
                ..
            }) => {
                USE_AFTER_FREE.fetch_add(1, ORD);
                trace(format!(
                    "USE-AFTER-FREE 0x{addr:x} = freed handle 0x{base:x} (generation \
                     {freed_at_generation}) — quarantine caught a stale pointer"
                ));
            }
            Err(LookupError::NotAHandle { .. }) => {
                HOST.fetch_add(1, ORD);
            }
        }
    }

    /// Snapshot for tests and reporting.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    pub struct Observations {
        pub observed: u64,
        pub host: u64,
        pub at_base: u64,
        pub interior: u64,
        pub in_guard_band: u64,
        pub use_after_free: u64,
        pub max_offset: usize,
    }

    /// The recorded trace, verbatim. Written beside the counters file at teardown, because by
    /// then ORT's logger is usually gone and a log line reaches nobody.
    pub fn trace_lines() -> Vec<String> {
        TRACE.lock().map(|t| t.clone()).unwrap_or_default()
    }

    pub fn snapshot() -> Observations {
        Observations {
            observed: OBSERVED.load(ORD),
            host: HOST.load(ORD),
            at_base: AT_BASE.load(ORD),
            interior: INTERIOR.load(ORD),
            in_guard_band: IN_GUARD_BAND.load(ORD),
            use_after_free: USE_AFTER_FREE.load(ORD),
            max_offset: MAX_OFFSET.load(ORD),
        }
    }

    /// A report that states what was *not* seen as loudly as what was. A verification that only
    /// prints its positives reads as a pass when the instrument never fired at all.
    pub fn report() -> String {
        let o = snapshot();
        let mut s = format!(
            "pointer observations: {} pointer(s) crossed back to us — {} host, {} at a handle \
             base, {} interior (max offset {} B), {} in a guard band, {} use-after-free",
            o.observed,
            o.host,
            o.at_base,
            o.interior,
            o.max_offset,
            o.in_guard_band,
            o.use_after_free
        );
        if o.observed == 0 {
            s.push_str(
                ". NOTHING WAS OBSERVED: no pointer of ours ever came back, so this run verifies \
                 nothing about the allocator contract — it is not a pass.",
            );
            return s;
        }
        if o.interior == 0 {
            s.push_str(
                ". ORT's planner never handed back a derived pointer in this run, so in-span \
                 `base + n` remains correct by construction but UNOBSERVED.",
            );
        }
        if o.use_after_free == 0 {
            let t = super::tally::snapshot();
            if t.quarantine_retired == 0 {
                s.push_str(&format!(
                    " The quarantine detector was armed and never fired: no stale handle was \
                     presented, so quarantine remains UNOBSERVED under this pattern. Its window \
                     covered the whole run (peak depth {} span(s), 0 retired), so this is a real \
                     negative for THIS pattern and not an exhausted instrument — but it is still \
                     a negative, and it stays one until a free is followed by a lookup of the \
                     same address. Nothing in the EP's control makes ORT do that.",
                    t.quarantine_peak_spans
                ));
            } else {
                s.push_str(&format!(
                    " The quarantine detector never fired, BUT its window was exhausted: {} span(s) \
                     were retired and returned to service (peak depth {}). A stale handle presented \
                     after its span was retired would resolve to a LIVE span and be counted as a \
                     normal lookup. `use_after_free == 0` is therefore NOT evidence here — raise \
                     {} and re-run before reading anything into it.",
                    t.quarantine_retired,
                    t.quarantine_peak_spans,
                    super::ENV_QUARANTINE_SPANS
                ));
            }
        }
        if let Ok(t) = TRACE.lock() {
            for line in t.iter() {
                s.push_str("\n    ");
                s.push_str(line);
            }
        }
        s
    }

    /// Serialise tests that assert on these process-global tallies.
    ///
    /// The ledger is deliberately process-wide — it observes an ABI boundary, not an object — so
    /// two tests asserting on it concurrently would flake. A mutex here is cheaper and more
    /// honest than making the counters per-registry purely to suit the test harness.
    #[cfg(test)]
    pub fn test_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: Mutex<()> = Mutex::new(());
        LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[cfg(test)]
    pub fn reset() {
        for c in [
            &OBSERVED,
            &HOST,
            &AT_BASE,
            &INTERIOR,
            &IN_GUARD_BAND,
            &USE_AFTER_FREE,
        ] {
            c.store(0, ORD);
        }
        MAX_OFFSET.store(0, ORD);
        if let Ok(mut t) = TRACE.lock() {
            t.clear();
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// The ORT-facing allocator
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// An `OrtAllocator` ORT can call, with our registry behind it.
///
/// `base` is first and the struct is `#[repr(C)]`, so `*mut VulkanAllocator` and
/// `*mut OrtAllocator` are the same address — the standard C-in-Rust vtable-embedding trick, and
/// the same shape `VulkanEp` and `VulkanEpFactory` already use.
#[repr(C)]
pub struct VulkanAllocator {
    base: ort::OrtAllocator,
    magic: u64,
    registry: Arc<HandleRegistry>,
    /// Owned: released in [`VulkanAllocator::release`].
    memory_info: *mut ort::OrtMemoryInfo,
    ort_api: *const ort::OrtApi,
}

impl VulkanAllocator {
    /// Build an allocator over `registry`, taking ownership of `memory_info`.
    ///
    /// # Safety
    /// `memory_info` must be an owned handle from `CreateMemoryInfo_V2` that the caller is
    /// transferring; `ort_api` must outlive the allocator (it is ORT's process-lifetime table).
    #[allow(clippy::new_ret_no_self)] // intentional: allocator hands an OrtAllocator* to ORT
    pub unsafe fn new(
        registry: Arc<HandleRegistry>,
        memory_info: *mut ort::OrtMemoryInfo,
        ort_api: *const ort::OrtApi,
    ) -> *mut ort::OrtAllocator {
        // SAFETY: `OrtAllocator` is a `#[repr(C)]` POD of a `u32` and `Option<fn>` slots; all-zero
        // is the valid `None` niche for every one of them.
        let mut base: ort::OrtAllocator = unsafe { std::mem::zeroed() };
        // ORT's own allocator version constant. Unlike the EP vtables this is not negotiated —
        // `OrtAllocator` is a stable public struct, not a plugin-EP surface, and ORT checks this
        // field for equality rather than as a "how far may I read" bound.
        base.version = ort::ORT_API_VERSION;
        base.Alloc = Some(alloc_thunk);
        base.Free = Some(free_thunk);
        base.Info = Some(info_thunk);
        base.Reserve = Some(reserve_thunk);
        base.GetStats = Some(get_stats_thunk);
        // `AllocOnStream` and `Shrink` stay `None`: we have no stream-ordered allocation and
        // nothing to shrink. ORT treats absent optional slots as "unsupported", which is a true
        // statement; filling them with a no-op that reports success would be a false one.
        let boxed = Box::new(VulkanAllocator {
            base,
            magic: ALLOCATOR_MAGIC,
            registry,
            memory_info,
            ort_api,
        });
        tally::on_allocator_created();
        Box::into_raw(boxed).cast()
    }

    /// Recover `&VulkanAllocator` from the pointer ORT hands back, or `None` if it is not ours.
    ///
    /// The magic check is not paranoia: `release_allocator` is called with whatever ORT believes
    /// our allocator to be, and a mismatched build or a confused host would otherwise have us
    /// interpret a foreign object as this struct. Cheap, and turns a memory-safety incident into
    /// a log line.
    ///
    /// # Safety
    /// `p` must be null or point to a live object at least as large as `VulkanAllocator`.
    unsafe fn from_ort<'a>(p: *const ort::OrtAllocator) -> Option<&'a VulkanAllocator> {
        if p.is_null() {
            return None;
        }
        let me: *const VulkanAllocator = p.cast();
        // SAFETY: the caller guarantees `p` points to a live allocation of at least our size; we
        // read only the `magic` field before trusting the rest.
        let magic = unsafe { std::ptr::addr_of!((*me).magic).read() };
        if magic != ALLOCATOR_MAGIC {
            log::error!(
                "VulkanExecutionProvider: an OrtAllocator at {p:?} was handed to this EP but does \
                 not carry our marker (found 0x{magic:x}). Refusing to interpret it. This means ORT \
                 routed another provider's allocator here, or two builds of this plugin are loaded."
            );
            return None;
        }
        // SAFETY: the marker confirms this is a `VulkanAllocator` this crate produced, and ORT
        // keeps it alive until `ReleaseAllocator`.
        Some(unsafe { &*me })
    }

    /// Destroy an allocator produced by [`VulkanAllocator::new`].
    ///
    /// # Safety
    /// `p` must be a pointer from `new` that has not already been released.
    pub unsafe fn release(p: *mut ort::OrtAllocator) {
        // SAFETY: caller guarantees provenance; the marker check guards against a foreign pointer.
        if unsafe { VulkanAllocator::from_ort(p) }.is_none() {
            return;
        }
        // SAFETY: confirmed ours and produced by `Box::into_raw` in `new`.
        let me: Box<VulkanAllocator> = unsafe { Box::from_raw(p.cast()) };
        log::info!(
            "VulkanExecutionProvider: releasing device allocator — {}",
            me.registry.summary()
        );
        let stats = me.registry.stats();
        tally::on_allocator_released(stats.live_spans, stats.live_bytes);
        // Scope matters here, and getting it wrong made this warning fire on healthy runs. The
        // registry is process-global per device (`factory::REGISTRIES`), shared by every allocator
        // and data transfer for that device. So `live_spans` at *one* allocator's release counts
        // spans that other, still-running sessions legitimately own. Attributing them to this
        // release names the wrong owner. `strong_count` tells us whether anyone else is still
        // holding the registry: the `REGISTRIES` map holds one Arc permanently, and we hold one,
        // so 2 means we are the last user and anything still live is genuinely unreclaimed.
        let other_holders = Arc::strong_count(&me.registry).saturating_sub(2);
        if stats.live_spans > 0 && other_holders > 0 {
            log::debug!(
                "VulkanExecutionProvider: {} device handle(s) ({} B) live in the shared registry \
                 at this allocator's release, with {} other holder(s) still running. These belong \
                 to sessions that have not finished; this release does not orphan them.",
                stats.live_spans,
                stats.live_bytes,
                other_holders
            );
        } else if stats.live_spans > 0 {
            log::warn!(
                "VulkanExecutionProvider: {} device handle(s) ({} B) were still live when the last \
                 allocator for this device was released — ORT did not hand these back to Free. {}",
                stats.live_spans,
                stats.live_bytes,
                tally::leak_verdict()
            );
        }
        if !me.memory_info.is_null() {
            // SAFETY: `ort_api` is ORT's process-lifetime table and `memory_info` is the owned
            // handle transferred in `new`, released exactly once here.
            unsafe {
                if let Some(f) = (*me.ort_api).ReleaseMemoryInfo {
                    f(me.memory_info);
                }
            }
        }
        drop(me);
        // Re-dump. The first dump happens at `VulkanDataTransfer::release`, which ORT calls
        // *before* it releases the allocator — so a file written only there reports
        // `alloc_allocators_released: 0` and `alloc_frees_after_release: 0` no matter what
        // subsequently happens, and those zeros are unfalsifiable rather than clean. The whole
        // document is regenerated from the current snapshot, so the later write simply supersedes
        // the earlier one. Measured, not assumed: without this call the two keys read 0 on a run
        // that leaves 322 handles live.
        crate::counters::dump_observations_if_requested();
    }
}

// SAFETY: every field is `Send`/`Sync` (the registry is behind an `Arc<Mutex<_>>`) except the two
// raw pointers, which are ORT-owned handles that ORT itself makes safe to use from any thread it
// calls us on. ORT calls allocators from worker threads, so this is required rather than optional.
unsafe impl Send for VulkanAllocator {}
// SAFETY: as above.
unsafe impl Sync for VulkanAllocator {}

unsafe extern "C" fn alloc_thunk(this_: *mut ort::OrtAllocator, size: usize) -> *mut c_void {
    // A panic here cannot become a status — `Alloc` returns a pointer, and ORT reads null as
    // failure. So the guard converts a panic into null, which is a contract-legal answer.
    crate::guard_ffi_ptr(|| {
        // SAFETY: ORT passes back the pointer it received from `CreateAllocator`.
        let Some(me) = (unsafe { VulkanAllocator::from_ort(this_) }) else {
            return std::ptr::null_mut();
        };
        match me.registry.alloc(size) {
            Some(addr) => addr as *mut c_void,
            None => std::ptr::null_mut(),
        }
    })
}

unsafe extern "C" fn reserve_thunk(this_: *mut ort::OrtAllocator, size: usize) -> *mut c_void {
    // `Reserve` is ORT's "allocate outside the arena" path, used for initializers and for the
    // memory-pattern blocks. Our arena is address space, not memory, and has no arena/non-arena
    // distinction — so the honest implementation is the same one.
    // SAFETY: same contract as `alloc_thunk`.
    unsafe { alloc_thunk(this_, size) }
}

unsafe extern "C" fn free_thunk(this_: *mut ort::OrtAllocator, p: *mut c_void) {
    crate::guard_ffi_void(|| {
        if p.is_null() {
            // Freeing null is legal and common.
            return;
        }
        // SAFETY: ORT passes back the pointer it received from `CreateAllocator`.
        let Some(me) = (unsafe { VulkanAllocator::from_ort(this_) }) else {
            return;
        };
        me.registry.free(p as usize);
    });
}

unsafe extern "C" fn info_thunk(this_: *const ort::OrtAllocator) -> *const ort::OrtMemoryInfo {
    crate::guard_ffi_ptr(|| {
        // SAFETY: ORT passes back the pointer it received from `CreateAllocator`.
        match unsafe { VulkanAllocator::from_ort(this_) } {
            Some(me) => me.memory_info,
            None => std::ptr::null_mut(),
        }
    })
    .cast_const()
}

unsafe extern "C" fn get_stats_thunk(
    this_: *const ort::OrtAllocator,
    out: *mut *mut ort::OrtKeyValuePairs,
) -> ort::OrtStatusPtr {
    if !out.is_null() {
        // SAFETY: `out` is ORT's out-param slot; we null it before any fallible work so ORT never
        // reads an uninitialised pointer on an error return.
        unsafe { *out = std::ptr::null_mut() };
    }
    // SAFETY: ORT passes back the pointer it received from `CreateAllocator`.
    let Some(me) = (unsafe { VulkanAllocator::from_ort(this_) }) else {
        return std::ptr::null_mut();
    };
    if out.is_null() {
        return std::ptr::null_mut();
    }
    let api = me.ort_api;
    // SAFETY: `api` is ORT's process-lifetime table.
    let (Some(create), Some(add)) = (unsafe { (*api).CreateKeyValuePairs }, unsafe {
        (*api).AddKeyValuePair
    }) else {
        return std::ptr::null_mut();
    };

    let s = me.registry.stats();
    let mut kvps: *mut ort::OrtKeyValuePairs = std::ptr::null_mut();
    // SAFETY: `create` fills our local out-param.
    unsafe { create(&mut kvps) };
    if kvps.is_null() {
        return std::ptr::null_mut();
    }

    // ORT documents `Limit`/`InUse`/`NumAllocs` as the conventional keys; the rest are ours and
    // are what the M2 verifications actually read.
    let pairs: [(&str, u64); 9] = [
        ("Limit", s.arena_bytes),
        ("InUse", s.live_bytes),
        ("NumAllocs", s.total_allocations),
        ("MaxInUse", s.high_water_bytes),
        ("NumFrees", s.total_frees),
        ("LiveSpans", s.live_spans),
        ("QuarantinedSpans", s.quarantined_spans),
        ("QuarantineRetired", s.quarantine_retired),
        ("FailedLookups", s.failed_lookups),
    ];
    for (k, v) in pairs {
        let (Ok(ck), Ok(cv)) = (CString::new(k), CString::new(v.to_string())) else {
            continue;
        };
        // SAFETY: `kvps` is live, and both strings are NUL-terminated and outlive the call — ORT
        // copies them.
        unsafe { add(kvps, ck.as_ptr(), cv.as_ptr()) };
    }
    // SAFETY: `out` is non-null (checked) and ORT takes ownership of `kvps`.
    unsafe { *out = kvps };
    std::ptr::null_mut()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry() -> Arc<HandleRegistry> {
        HandleRegistry::new().expect("reserving address space must succeed on a 64-bit host")
    }

    /// A genuinely MIXED run must say "mixed", with the real ratio.
    ///
    /// Hardware has only ever produced the all-or-nothing states, so this branch has never been
    /// exercised by a real model. That is exactly why it gets a test: the failure this guards
    /// against is a verdict that keeps asserting one extreme after the truth has become partial.
    #[test]
    fn staging_verdict_reports_the_measured_ratio_when_spans_are_mixed() {
        let _g = ledger::test_lock();
        tally::reset_for_test();
        // 10 allocations: 6 device-backed, 4 host-staged.
        tally::seed_for_test(10, 6, 4, 4096);
        let v = tally::staging_verdict();

        assert!(v.contains("MIXED"), "must name the mixed state, got: {v}");
        assert!(
            v.contains("60.0% device-backed"),
            "must report the RATIO IT MEASURED, not a fixed sentence; got: {v}"
        );
        // The failure mode this test exists for: claiming a universal when 6/10 contradict it.
        assert!(
            !v.contains("ALL "),
            "must not claim an extreme while 6 of 10 spans are device-backed; got: {v}"
        );
        tally::reset_for_test();
    }

    /// Both counters at zero is an ABSENT INSTRUMENT, not a clean run (R7).
    #[test]
    fn staging_verdict_refuses_to_reassure_when_nothing_observed_residency() {
        let _g = ledger::test_lock();
        tally::reset_for_test();
        tally::seed_for_test(10, 0, 0, 0);
        let v = tally::staging_verdict();

        assert!(v.contains("UNMEASURED"), "got: {v}");
        assert!(
            !v.contains("not disqualified by staging"),
            "zero staged spans must not read as reassurance when nothing recorded residency; \
             got: {v}"
        );
        tally::reset_for_test();
    }

    /// Every branch must carry the staging the allocator cannot see.
    ///
    /// The dominant staging cost on Phi-3.5 is `vk::session`'s per-inference copy, which no
    /// `alloc_*` counter observes. A verdict that omits it is accurate about spans and wrong
    /// about the run.
    #[test]
    fn every_staging_verdict_branch_mentions_the_traffic_it_cannot_see() {
        let _g = ledger::test_lock();
        for (allocations, backed, staged) in
            [(10u64, 10u64, 10u64), (10, 10, 0), (10, 0, 10), (10, 6, 4), (10, 0, 0)]
        {
            tally::reset_for_test();
            tally::seed_for_test(allocations, backed, staged, 4096);
            let v = tally::staging_verdict();
            assert!(
                v.contains("SESSION STAGING"),
                "branch ({allocations},{backed},{staged}) omitted session staging: {v}"
            );
        }
        tally::reset_for_test();
    }

    /// **The verification OQ-3 turns on.** ORT's memory-pattern planner allocates one block and
    /// hands out `base + offset`. With integer handles those interior pointers collide with other
    /// live handles; with reserved address space they stay in-span by construction.
    #[test]
    fn interior_pointers_from_planner_arithmetic_resolve_to_their_own_span() {
        let r = registry();
        let a = r.alloc(4096).expect("alloc");
        let b = r.alloc(4096).expect("alloc");
        let c = r.alloc(1_000_000).expect("alloc");
        assert_ne!(a, b);
        assert_ne!(b, c);

        // Every byte of every allocation resolves to its own base with the right offset — this is
        // the property the planner depends on.
        for (base, size) in [(a, 4096usize), (b, 4096), (c, 1_000_000)] {
            for off in [0usize, 1, 7, 64, 4095, size / 2, size - 1] {
                let got = r
                    .resolve(base + off)
                    .unwrap_or_else(|e| panic!("0x{:x}+{off} should resolve: {e}", base));
                assert_eq!(got.base, base, "offset {off} escaped its span");
                assert_eq!(got.offset, off);
                assert_eq!(got.size, size);
            }
        }
    }

    /// The one-past-the-end pointer is what pointer arithmetic produces at a loop bound. With
    /// integer handles it is the *next allocation*; here it must be a diagnosable hole.
    #[test]
    fn one_past_the_end_lands_in_a_guard_band_not_on_the_next_allocation() {
        let r = registry();
        let a = r.alloc(4096).expect("alloc");
        let b = r.alloc(4096).expect("alloc");

        let err = r
            .resolve(a + 4096)
            .expect_err("one-past-the-end must not resolve");
        assert!(
            matches!(err, LookupError::InGuardBand { .. }),
            "expected a guard-band diagnosis, got {err:?}"
        );
        assert_ne!(
            a + 4096,
            b,
            "the next allocation must not begin exactly one past the end of the previous one — \
             that is the aliasing this scheme exists to prevent"
        );
    }

    /// A size that is not a multiple of the page granularity must still bound lookups by the
    /// *requested* size. Accepting the rounding slack would silently permit a short read past the
    /// end of the tensor.
    #[test]
    fn lookups_are_bounded_by_the_requested_size_not_the_padded_one() {
        let r = registry();
        let a = r.alloc(100).expect("alloc");
        assert!(r.resolve(a + 99).is_ok());
        let err = r
            .resolve(a + 100)
            .expect_err("past the requested end must fail");
        assert!(matches!(err, LookupError::InGuardBand { .. }), "{err:?}");
    }

    /// **The second deferred verification.** A freed handle must be rejected loudly rather than
    /// aliasing onto a live tensor.
    #[test]
    fn a_freed_handle_is_rejected_loudly_and_never_aliases_a_live_one() {
        let r = registry();
        let a = r.alloc(8192).expect("alloc");
        assert!(r.resolve(a + 16).is_ok());
        r.free(a);

        let err = r
            .resolve(a + 16)
            .expect_err("a freed handle must not resolve");
        match err {
            LookupError::Freed {
                base,
                freed_at_generation,
                ..
            } => {
                assert_eq!(base, a);
                assert!(freed_at_generation > 0);
                // The message must name the cause; a stale-pointer bug diagnosed as "unknown
                // pointer" sends the reader to the wrong file.
                let msg = err.to_string();
                assert!(msg.contains("freed"), "{msg}");
                assert!(msg.contains("stale"), "{msg}");
            }
            other => panic!("expected a Freed diagnosis, got {other:?}"),
        }

        // And crucially: subsequent allocations must not land on the quarantined span.
        for _ in 0..64 {
            let n = r.alloc(8192).expect("alloc");
            assert_ne!(
                n, a,
                "a quarantined span was re-served while still detectable"
            );
        }
        assert_eq!(
            r.stats().quarantine_retired,
            0,
            "quarantine window was not exhausted"
        );
    }

    #[test]
    fn a_double_free_is_survivable_and_counted() {
        let r = registry();
        let a = r.alloc(4096).expect("alloc");
        r.free(a);
        r.free(a); // must not panic, must not corrupt
        assert_eq!(
            r.stats().total_frees,
            1,
            "the second free must not be counted as a real one"
        );
        assert!(r.stats().failed_lookups >= 1);
    }

    #[test]
    fn foreign_and_interior_pointers_are_told_apart() {
        let r = registry();
        let a = r.alloc(4096).expect("alloc");
        let host = Box::new(42u64);
        let host_addr = &*host as *const u64 as usize;

        assert!(matches!(
            r.resolve(host_addr),
            Err(LookupError::NotAHandle { .. })
        ));
        // Freeing an interior pointer is a caller bug and must be named as one.
        r.free(a + 8);
        assert!(r.stats().failed_lookups >= 1);
        assert!(
            r.resolve(a).is_ok(),
            "a bad free must not have released the real span"
        );
    }

    /// Mouse's P6 assertion reads this. It must track the peak, not the current value, or a
    /// dequantised weight that was allocated and freed before the check would be invisible.
    #[test]
    fn high_water_records_the_peak_not_the_current_live_bytes() {
        let r = registry();
        let a = r.alloc(1_000_000).expect("alloc");
        let b = r.alloc(2_000_000).expect("alloc");
        assert_eq!(r.stats().high_water_bytes, 3_000_000);
        r.free(a);
        r.free(b);
        assert_eq!(r.stats().live_bytes, 0);
        assert_eq!(
            r.stats().high_water_bytes,
            3_000_000,
            "freeing must not erase the evidence that the memory was once materialised"
        );
    }

    #[test]
    fn a_zero_byte_allocation_returns_a_distinct_freeable_handle() {
        let r = registry();
        let a = r
            .alloc(0)
            .expect("zero-size alloc must not be reported as failure");
        let b = r.alloc(0).expect("alloc");
        assert_ne!(a, 0, "null would be read by ORT as an allocation failure");
        assert_ne!(a, b);
        r.free(a);
        r.free(b);
    }

    #[test]
    fn quarantine_retirement_is_counted_so_the_window_is_never_silently_exhausted() {
        // SAFETY: this test sets the quarantine bound before building its own registry. Cargo runs
        // tests in threads, so read the value into the registry at construction (which
        // `HandleRegistry::new` does) rather than relying on it later.
        // SAFETY: `set_var`/`remove_var` are unsafe because they are not thread-safe; this test
        // is single-threaded and writes the env before any threads are spawned. The variable is
        // cleaned up on every path (no early return between set and remove).
        unsafe { std::env::set_var(ENV_QUARANTINE_SPANS, "4") };
        let r = registry();
        // SAFETY: same set_var/remove_var safety as above; no thread is reading this variable.
        unsafe { std::env::remove_var(ENV_QUARANTINE_SPANS) };

        let mut handles = Vec::new();
        for _ in 0..16 {
            handles.push(r.alloc(4096).expect("alloc"));
        }
        for h in &handles {
            r.free(*h);
        }
        let s = r.stats();
        assert!(
            s.quarantine_retired > 0,
            "with a window of 4 and 16 frees, retirement must have happened and must be visible"
        );
        assert_eq!(s.quarantined_spans, 4, "the window holds exactly its bound");
    }

    /// `alloc_device_authoritative_spans` will be the headline the moment persistent residency
    /// lands. These are the two independent guards that must fire before it can be quoted.
    #[test]
    fn an_over_claimed_authoritative_count_is_called_dishonest_in_the_artifact() {
        let _g = ledger::test_lock();
        tally::reset_for_test();
        // 10 allocations, 10 device-backed, 10 still staged -> ceiling is 0, so ANY authoritative
        // span is a mirror being reported as device-resident. This is the exact shape of the
        // claim this project has produced repeatedly: true about residency, false about the run.
        tally::seed_for_test(10, 10, 10, 40960);
        tally::seed_authoritative_for_test(6, 0);
        let v = tally::staging_verdict();
        assert!(v.contains("NOT CREDIBLE"), "got: {v}");
        assert!(
            v.contains("6 authoritative span(s) but only 0"),
            "must quote both measured numbers, not just object; got: {v}"
        );
        assert!(
            v.contains("never called transfer::device_buffer_for"),
            "the bind counter is the second, independent falsifier; got: {v}"
        );
        tally::reset_for_test();
    }

    /// The credible case must also be stated, or the guard is a warning nobody can clear.
    #[test]
    fn a_credible_authoritative_count_says_which_checks_it_survived() {
        let _g = ledger::test_lock();
        tally::reset_for_test();
        // 10 allocations, 10 device-backed, 2 staged -> ceiling 8. Claim 6, with real binds.
        tally::seed_for_test(10, 10, 2, 8192);
        tally::seed_authoritative_for_test(6, 33);
        let v = tally::staging_verdict();
        assert!(!v.contains("NOT CREDIBLE"), "got: {v}");
        assert!(v.contains("survives both independent checks"), "got: {v}");
        assert!(v.contains("ceiling of 8"), "must quote the measured ceiling; got: {v}");
        tally::reset_for_test();
    }

    /// Zero is the value this counter will hold right up until the moment it is quoted, so the
    /// artifact must distinguish an unwired zero from a measured one. This is the falsifier for
    /// that distinction: with no increment point called, the sentence must SAY it is unwired.
    #[test]
    fn a_zero_authoritative_count_says_it_is_unwired_rather_than_measured() {
        let _g = ledger::test_lock();
        tally::reset_for_test();
        tally::seed_for_test(10, 10, 2, 8192);
        let v = tally::staging_verdict();
        assert!(v.contains("UNWIRED zero"), "got: {v}");
        assert!(v.contains("alloc_device_buffer_binds is 0"), "got: {v}");
        assert!(v.contains("ceiling for it on this run is 8"), "got: {v}");
        assert!(!v.contains("NOT CREDIBLE"), "a zero is not a dishonest count; got: {v}");
        tally::reset_for_test();
    }

    /// A device buffer handed to the engine must be counted, or the falsifier above is inert.
    #[test]
    fn handing_out_a_device_buffer_is_counted() {
        let _g = ledger::test_lock();
        tally::reset_for_test();
        assert_eq!(tally::snapshot().device_buffer_binds, 0);
        tally::on_device_buffer_bind();
        assert_eq!(tally::snapshot().device_buffer_binds, 1);
        tally::reset_for_test();
    }

    /// `failed_lookups` said of itself "non-zero always indicates a bug" and was reachable only
    /// through an ORT API nothing in this repo calls. This is the falsifier for the plumbing.
    #[test]
    fn a_failed_lookup_reaches_the_process_wide_counters() {
        let _g = ledger::test_lock();
        tally::reset_for_test();
        let r = registry();
        r.free(0xdead_0000);
        assert!(
            tally::snapshot().failed_lookups >= 1,
            "a bug detector whose count no artifact carries is not a detector"
        );
        tally::reset_for_test();
    }

    /// Retirement must reach the counters FILE, not just the per-registry struct.
    ///
    /// The per-registry `quarantine_retired` is only reachable through ORT's `GetStats` KVPs,
    /// which nothing in the harness calls — so "quarantine never fired" was being asserted from
    /// a number no run ever wrote down. This is the R7 falsifier for that claim: if the global
    /// counter does not move here, `alloc_quarantine_retired: 0` in a real run means "not
    /// plumbed", not "window intact".
    #[test]
    fn quarantine_exhaustion_reaches_the_process_wide_counters() {
        let _g = ledger::test_lock();
        tally::reset_for_test();
        // SAFETY: single-threaded test, value is consumed by `HandleRegistry::new` immediately
        // and removed before anything else can read it.
        unsafe { std::env::set_var(ENV_QUARANTINE_SPANS, "4") };
        let r = registry();
        // SAFETY: see above.
        unsafe { std::env::remove_var(ENV_QUARANTINE_SPANS) };

        let handles: Vec<_> = (0..16).map(|_| r.alloc(4096).expect("alloc")).collect();
        for h in &handles {
            r.free(*h);
        }
        let t = tally::snapshot();
        assert!(
            t.quarantine_retired > 0,
            "exhaustion must be visible process-wide, or `alloc_quarantine_retired: 0` in a real \
             run is unfalsifiable by construction"
        );
        assert_eq!(t.quarantine_peak_spans, 4, "peak depth must be the bound");
        tally::reset_for_test();
    }

    #[test]
    fn attach_buffer_records_the_device_side_and_refuses_stale_handles() {
        let r = registry();
        let a = r.alloc(4096).expect("alloc");
        assert!(r.resolve(a).expect("resolve").buffer.is_none());
        r.attach_buffer(a, BufferView::from_raw(7)).expect("attach");
        assert_eq!(
            r.resolve(a + 32).expect("resolve").buffer,
            Some(BufferView::from_raw(7)),
            "an interior pointer must reach the same device buffer as its base"
        );
        r.free(a);
        assert!(matches!(
            r.attach_buffer(a, BufferView::from_raw(8)),
            Err(LookupError::Freed { .. })
        ));
    }

    /// The property that makes a mistaken dereference loud. Not executed — the point is that it
    /// would fault — but the address must at least be outside anything the process can read, which
    /// we assert structurally: it is inside our `PROT_NONE`/`PAGE_NOACCESS` reservation.
    #[test]
    fn handles_live_inside_the_inaccessible_reservation() {
        let r = registry();
        let a = r.alloc(4096).expect("alloc");
        assert!(a >= r.arena.base && a < r.arena_end());
        assert_eq!(a % SPAN_GRANULARITY, 0, "handles are page-aligned");
    }
}
