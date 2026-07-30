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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::engine::BufferView;
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
        Some(base)
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
            log::error!(
                "VulkanExecutionProvider: double free of device handle 0x{addr:x} (freed at \
                 generation {g}). Caught by quarantine rather than corrupting a live tensor. \
                 Ignoring the second free."
            );
            return;
        }
        span.live = false;
        let requested = span.requested;
        inner.stats.total_frees += 1;
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
                s.buffer = Some(view);
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
        if span.buffer.is_some() {
            // A device buffer is attached: staging must not shadow it, or a copy would land in
            // host memory the device never reads and the wrong answer would be silent.
            return None;
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
        if inner.stats.staging_spans == 0 {
            log::warn!(
                "VulkanExecutionProvider: no VkBuffer is attached to device handle 0x{addr:x}, so \
                 its contents are being held in HOST memory. This is the near half of the real \
                 copy path and it produces correct results, but nothing in this run touched device \
                 memory for staged tensors. Any timing from this run is a host measurement — see \
                 `StagingLiveBytes` in the allocator stats."
            );
        }
        inner.staging.insert(addr, HostStaging { ptr, layout });
        inner.stats.staging_spans += 1;
        inner.stats.staging_live_bytes += layout.size() as u64;
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
/// The taxonomy is [`LookupError`]'s, because those three failures demand three different fixes:
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
            s.push_str(
                " The quarantine detector was armed and never fired: no stale handle was \
                 presented, so quarantine remains UNOBSERVED under this pattern.",
            );
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
        if stats.live_spans > 0 {
            log::warn!(
                "VulkanExecutionProvider: {} device handle(s) ({} B) were still live when the \
                 allocator was released. ORT frees what it allocated, so this is either a leak on \
                 our side or a tensor the session outlived.",
                stats.live_spans,
                stats.live_bytes
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
