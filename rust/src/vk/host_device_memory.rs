//! Device-backed memory for **ORT-owned tensors** — the engine half of
//! [`crate::engine::DeviceMemoryProvider`].
//!
//! **Cross-owner note (Tank).** This file lives in Switch's `vk/` tree but is Tank's. It is a new
//! file rather than an edit to `session.rs`, `alloc.rs` or `instance.rs` precisely so that it
//! cannot collide with work in flight there. It adds one line to `vk/mod.rs`. If Switch wants the
//! engine's compute session to *consume* these buffers, the seam for that is
//! [`crate::transfer::device_buffer_for`], not this file.
//!
//! # What this does, stated so it cannot be over-read
//!
//! [`crate::allocator::HandleRegistry`] hands ORT reserved addresses. Until now every one of those
//! spans was backed by a host `alloc_zeroed` block — `alloc_device_backed_spans` was 0 in every
//! run this project has ever taken. This module gives those spans a real `VkBuffer` in
//! `DEVICE_LOCAL` memory, and makes `CopyTensors` move bytes across `vkCmdCopyBuffer` instead of
//! `memcpy`.
//!
//! # §6.5 — this provider RECEIVES a `VkDevice`; it does not create one
//!
//! **Ruled 2026-07-30T22:13:37-07:00: exactly one `VkDevice` per (physical device, EP instance).**
//! The seam is owned by the side that owns the lifetime — Switch — and this file is the consumer.
//! The surface is exactly three methods, [`SharedVkDevice`], and one entry point,
//! [`offer_shared_device`]. Switch calls that once, from wherever the EP-scoped device context is
//! constructed (§2.3 says that is `CreateEp`, not `VulkanSession::create`). Nothing else here
//! needs to change on his side and nothing in `vk/session.rs`, `vk/device.rs` or `vk/instance.rs`
//! is edited by this file.
//!
//! The provider holds an `Arc<dyn SharedVkDevice>`, which **pins the device for as long as the
//! provider lives** — that matters, because [`crate::allocator::HandleRegistry`] is process-global
//! (`factory::REGISTRIES`) and outlives every `VulkanSession`: a span allocated during session A
//! may be freed after session A is gone. Holding the `Arc` is what makes handing over a
//! session-scoped device safe *at the Vulkan level*; it does not make it correct, and §2.3 still
//! requires EP scope.
//!
//! # The fallback, and why it is a reported state rather than a silent one
//!
//! Until [`offer_shared_device`] has been called for a device index, this module falls back to
//! [`OwnedDevice`] — which does create an `Instance` and a `Device` of its own, exactly as this
//! file did before §6.5. **That fallback is the defect §6.5 names**, so it is not silent: the
//! frame is recorded as [`DeviceFrame::SplitDevice`], `alloc_device_frame` says so in the counters
//! artifact, and per R12 `alloc_device_authoritative_spans` reports `UNOBSERVABLE` rather than `0`
//! — because in that frame the event it counts *cannot occur*, and a zero would be read as a
//! measurement.
//!
//! In the split frame, **these buffers are on a different `VkDevice` from the one `vk::session`
//! dispatches on, so a compute shader cannot bind them.** That means:
//!
//! * the bytes really are in device memory, and the upload/download cost really is bus traffic —
//!   so `alloc_device_backed_spans` and `alloc_device_upload_bytes` measure what they say;
//! * but a non-zero `alloc_device_backed_spans` **does not** mean the EP computes from device
//!   memory. It means ORT's tensors are resident there. Those are different claims and the
//!   counter `alloc_device_backed_shared_with_engine` is 0 to say so in the same file, at the same
//!   volume, rather than leaving the distinction to whoever reads the number.
//!
//! Closing that gap is the engine-side change: `vk::session` must resolve its inputs through
//! [`crate::transfer::device_buffer_for`] and bind the returned buffer, instead of allocating a
//! fresh `DeviceLocal` buffer and re-uploading the same weights on every `Compute`. Until it does,
//! device-backed allocation is a **precondition** and reporting it as a speedup would be the error
//! this project has already made four times.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use ash::vk;

use super::alloc::{Allocator, GpuBuffer, MemClass};
use super::cmd::{CommandPool, submit_and_wait};
use super::device::Device;
use super::instance::Instance;
use crate::engine::{BufferView, DeviceMemoryProvider};

/// Everything the provider needs, kept behind one mutex.
///
/// One lock rather than several because every operation here touches the allocator and the command
/// pool together, and a copy must not interleave with a free of the buffer it is copying.
struct Inner {
    alloc: Allocator,
    cmd_pool: CommandPool,
    /// Buffers we minted, keyed by the token in the [`BufferView`] we handed out.
    buffers: HashMap<u64, GpuBuffer>,
    next_token: u64,
}

/// The **exact** surface this provider needs from the EP's device context (§6.5 seam).
///
/// Three methods, all borrows, no construction, no lifetime obligations beyond `Arc`. Switch
/// implements this on whatever type owns the EP-scoped device and calls [`offer_shared_device`]
/// once. Adding a fourth method to this trait is a cross-owner change and gets declared as one.
pub(crate) trait SharedVkDevice: Send + Sync {
    /// The `ash::Instance` the device was created from. Needed only by `Allocator::new`.
    fn instance_ash(&self) -> &ash::Instance;
    /// The logical device handle. Needed for Vulkan command recording and submission.
    fn ash_device(&self) -> &ash::Device;
    /// The physical device handle. Needed by `Allocator::new`.
    fn physical_device(&self) -> vk::PhysicalDevice;
    /// The compute queue family index. Needed for `CommandPool::new`.
    fn compute_queue_family(&self) -> u32;
    /// The compute queue handle. Needed for `submit_and_wait`.
    fn compute_queue(&self) -> vk::Queue;
    /// Whether the device uses unified memory (UMA). Drives the `alloc_unified_memory` counter.
    fn is_uma(&self) -> bool;
    /// The physical device's name, for the identity line in the log and the artifact.
    fn device_name(&self) -> &str;
}

/// Which `VkDevice` this provider's buffers live on, relative to the one that dispatches.
///
/// This is **frame provenance** in the sense of §10.0.1 R12: it is not a quality score, it is the
/// identity of the world the `alloc_device_*` numbers were measured in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum DeviceFrame {
    /// §6.5 satisfied: the buffers are on the session's device and a dispatch could bind them.
    Shared,
    /// §6.5 violated: a second device, so every `alloc_device_*` number describes another world.
    SplitDevice,
}

impl DeviceFrame {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Shared => "SHARED",
            Self::SplitDevice => "SPLIT-DEVICE",
        }
    }
}

/// A device this module created itself — the §6.5 fallback, used only until Switch's context is
/// offered. Its existence is the defect, so its use is always reported as `SPLIT-DEVICE`.
struct OwnedDevice {
    device: Device,
    device_name: String,
    /// Held so the loader and the instance outlive every buffer.
    instance: Instance,
}

impl SharedVkDevice for OwnedDevice {
    fn instance_ash(&self) -> &ash::Instance {
        self.instance.ash()
    }
    fn ash_device(&self) -> &ash::Device {
        self.device.ash()
    }
    fn physical_device(&self) -> vk::PhysicalDevice {
        self.device.physical_device()
    }
    fn compute_queue_family(&self) -> u32 {
        self.device.compute_queue_family()
    }
    fn compute_queue(&self) -> vk::Queue {
        self.device.compute_queue()
    }
    fn is_uma(&self) -> bool {
        self.device.caps().is_uma
    }
    fn device_name(&self) -> &str {
        &self.device_name
    }
}

// SAFETY: both fields are immutable after construction. `ash::Device` and `ash::Instance` are
// Vulkan handles, which the specification permits to be used from any thread given external
// synchronisation; nothing here is thread-affine and the queue is synchronised by the provider's
// mutex.
unsafe impl Send for OwnedDevice {}
// SAFETY: as above.
unsafe impl Sync for OwnedDevice {}

/// Device-local memory for ORT tensors, on a device this module was **handed**.
pub(crate) struct HostDeviceMemory {
    inner: Mutex<Inner>,
    /// The device context. Held as an `Arc` so it outlives every buffer minted from it, which is
    /// what makes the process-global `HandleRegistry` safe against session teardown.
    ctx: Arc<dyn SharedVkDevice>,
    is_uma: bool,
    frame: DeviceFrame,
    /// The device the buffers actually landed on.
    ///
    /// Recorded so the identity can be *reported* rather than assumed. Whether it agrees with the
    /// device `VulkanSession` picked is a claim, not a given — it must be checkable from a log.
    device_name: String,
}

// SAFETY: every field is either immutable after construction (`device`, `is_uma`, `_instance`) or
// behind the mutex. `ash::Device` and `vk::Queue` are Vulkan handles, which the specification
// permits to be used from any thread provided external synchronisation, which the mutex provides
// for the queue and the pool. Nothing here is thread-affine.
unsafe impl Send for HostDeviceMemory {}
// SAFETY: as above.
unsafe impl Sync for HostDeviceMemory {}

impl HostDeviceMemory {
    /// Build a provider **on a device it was given**. This is the §6.5 shape.
    ///
    /// Fails (`None`) only if the allocator or the command pool cannot be created on that device.
    /// It creates no `Instance` and no `Device`.
    ///
    /// # Safety
    /// `ctx` must reference a live instance and device, and the `Arc` must be the caller's promise
    /// that they stay live — which it is, because this function stores it.
    unsafe fn on_shared_device(ctx: Arc<dyn SharedVkDevice>, frame: DeviceFrame) -> Option<Self> {
        let is_uma = ctx.is_uma();
        let device_name = ctx.device_name().to_string();
        // SAFETY: instance and device are live for as long as `ctx`, which is stored below.
        let alloc =
            unsafe { Allocator::new(ctx.instance_ash(), ctx.physical_device(), ctx.ash_device()) }?;
        // SAFETY: device is live; the queue family index came from it.
        let cmd_pool = unsafe { CommandPool::new(ctx.ash_device(), ctx.compute_queue_family()) }?;
        Some(Self {
            inner: Mutex::new(Inner {
                alloc,
                cmd_pool,
                buffers: HashMap::new(),
                next_token: 1,
            }),
            ctx,
            is_uma,
            frame,
            device_name,
        })
    }

    /// The device this provider's buffers landed on. Reported, never inferred.
    pub(crate) fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Which world the `alloc_device_*` numbers from this provider were measured in (R12).
    pub(crate) fn frame(&self) -> DeviceFrame {
        self.frame
    }

    fn ash_device(&self) -> &ash::Device {
        self.ctx.ash_device()
    }

    fn compute_queue(&self) -> vk::Queue {
        self.ctx.compute_queue()
    }
}

impl OwnedDevice {
    /// Stand up a device of our own — the §6.5 fallback. `None` if Vulkan is unavailable.
    ///
    /// # Safety
    /// The Vulkan loader must remain loaded for the process lifetime, which it does: the returned
    /// value is leaked into a `OnceLock` and never dropped.
    unsafe fn create(device_index: usize) -> Option<Self> {
        let instance = Instance::create(false)?;
        let mut capables = instance.enumerate_capable_devices();
        if capables.is_empty() {
            return None;
        }
        // Resolve the device the SAME WAY `VulkanSession::create` does.
        //
        // (Tank, 2026-07-30) This used to be `capables[device_index]`, and that was wrong in a way
        // no counter reported. Three different index spaces are in play:
        //
        //   1. `enumerate_capable_devices()` — SORTED best-first (discrete > integrated), see
        //      `vk/instance.rs`. On this desk that is [RTX 4060, Iris Xe].
        //   2. `ONNXRUNTIME_EP_VULKAN_DEVICE` — an index into (1), applied by `select_device`.
        //      This is what the compute session obeys.
        //   3. `device_index` as passed here — the *factory's* advertised-device index, assigned
        //      in `factory.rs`, which is not guaranteed to agree with either of the above.
        //
        // Indexing (1) with (3) silently put the mirror on a different physical device from the
        // one running the kernels: measured `alloc_unified_memory=1` (UMA/Intel) on BOTH
        // selector values, including the run whose kernels were on the discrete card. The counter
        // was numerically correct and attributed to the wrong device — the same failure shape as
        // the process-global `HandleRegistry` scope error, one level up.
        //
        // Matching indices does NOT make this one device: §6.5 is about the `VkDevice` object, not
        // the physical device it was created from. Agreeing here only means the two `VkDevice`s
        // wrap the same GPU, which is why the frame is still `SPLIT-DEVICE`.
        let idx = crate::vk::instance::select_device(&capables).unwrap_or(0);
        let capable = capables.swap_remove(idx);
        let device_name = capable.info.name.clone();
        if idx != device_index {
            log::debug!(
                "VulkanExecutionProvider: device-backed allocation resolved to capable-device \
                 index {idx} ('{device_name}') via {}, while the allocator was created for \
                 factory device index {device_index}. The mirror follows the compute session's \
                 device, which is the one that matters; these two index spaces are not the same \
                 and must not be assumed equal.",
                crate::vk::instance::ENV_DEVICE_SELECTOR,
            );
        }
        // SAFETY: `instance` is live; `capable` came from its own enumeration.
        let device = unsafe { Device::create(instance.ash(), &capable, false) }?;
        Some(Self {
            device,
            device_name,
            instance,
        })
    }
}

impl HostDeviceMemory {
    /// Run one staged copy in either direction. `to_device` picks upload or download.
    fn staged_copy(
        &self,
        view: BufferView,
        offset: usize,
        bytes: &mut [u8],
        to_device: bool,
    ) -> Result<(), String> {
        let Ok(mut inner) = self.inner.lock() else {
            return Err("the device-memory provider's lock is poisoned".to_string());
        };
        let len = bytes.len() as u64;
        if len == 0 {
            return Ok(());
        }
        let Some(target) = inner.buffers.get(&view.as_raw()) else {
            return Err(format!(
                "device buffer token {} is not one this provider minted",
                view.as_raw()
            ));
        };
        let target_buffer = target.buffer;
        let target_size = target.size;
        if offset as u64 + len > target_size {
            return Err(format!(
                "a copy of {len} byte(s) at offset {offset} runs past the end of a {target_size} \
                 byte device buffer"
            ));
        }

        let class = if to_device {
            MemClass::Upload
        } else {
            MemClass::Download
        };
        let usage = if to_device {
            vk::BufferUsageFlags::TRANSFER_SRC
        } else {
            vk::BufferUsageFlags::TRANSFER_DST
        };
        // SAFETY: the allocator is live for as long as this struct, which is the process lifetime.
        let Some(staging) = (unsafe { inner.alloc.alloc("ort-tensor-staging", len, class, usage) })
        else {
            return Err(format!(
                "could not allocate a {len} byte staging buffer for a device-memory copy"
            ));
        };
        let Some(mapped) = staging.mapped_ptr() else {
            // SAFETY: `staging` came from this allocator and has not been freed.
            unsafe { inner.alloc.free(staging) };
            return Err("a host-visible staging buffer had no mapped pointer".to_string());
        };

        if to_device {
            // SAFETY: `mapped` addresses at least `len` bytes of host-coherent memory, and
            // `bytes` is at least `len` readable bytes. The two are distinct allocations.
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), mapped, bytes.len()) };
        }

        let result = self.record_and_submit(
            &inner,
            staging.buffer,
            target_buffer,
            offset as u64,
            len,
            to_device,
        );

        if result.is_ok() && !to_device {
            // SAFETY: as above, in the other direction. The queue has been waited on, so the
            // download buffer's contents are visible to the host.
            unsafe {
                std::ptr::copy_nonoverlapping(mapped.cast_const(), bytes.as_mut_ptr(), bytes.len())
            };
        }

        // SAFETY: the submission was waited on before returning, so no command buffer still
        // references this staging buffer.
        unsafe { inner.alloc.free(staging) };
        result
    }

    /// Record a single `vkCmdCopyBuffer` and wait for it.
    ///
    /// Synchronous by contract: `CopyTensors` has no completion handle for ORT to wait on, so
    /// returning before the bytes have landed would be a silent race.
    fn record_and_submit(
        &self,
        inner: &Inner,
        staging: vk::Buffer,
        target: vk::Buffer,
        offset: u64,
        len: u64,
        to_device: bool,
    ) -> Result<(), String> {
        // SAFETY: the pool is live and no prior submission from it is outstanding — every
        // submission this module makes is waited on before the lock is released.
        let Some(rec) = (unsafe { inner.cmd_pool.begin() }) else {
            return Err("could not begin a command buffer for a device-memory copy".to_string());
        };
        let cmd = rec.cmd;
        let (src, dst, region) = if to_device {
            (
                staging,
                target,
                vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: offset,
                    size: len,
                },
            )
        } else {
            (
                target,
                staging,
                vk::BufferCopy {
                    src_offset: offset,
                    dst_offset: 0,
                    size: len,
                },
            )
        };
        // SAFETY: `cmd` is recording; both buffers belong to this device and the region was bounds
        // checked against the target's size by the caller.
        unsafe {
            self.ash_device()
                .cmd_copy_buffer(cmd, src, dst, std::slice::from_ref(&region))
        };
        // SAFETY: `rec` is in the recording state.
        let Some(cmd) = (unsafe { rec.finish() }) else {
            return Err("could not end the command buffer for a device-memory copy".to_string());
        };
        // SAFETY: `cmd` was recorded and ended above; the queue belongs to this device.
        let ok = unsafe { submit_and_wait(self.ash_device(), self.compute_queue(), cmd) };
        if ok {
            Ok(())
        } else {
            Err("the device-memory copy did not complete — the queue submission failed".to_string())
        }
    }
}

impl DeviceMemoryProvider for HostDeviceMemory {
    fn alloc(&self, size: usize) -> Option<BufferView> {
        let mut inner = self.inner.lock().ok()?;
        // `TRANSFER_SRC | TRANSFER_DST` because the copy runs in both directions, and
        // `STORAGE_BUFFER` so that the engine can bind this buffer once it starts doing so
        // without needing the allocation to be reissued.
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;
        // SAFETY: the allocator lives as long as this struct.
        let buf = unsafe {
            inner
                .alloc
                .alloc("ort-tensor", size as u64, MemClass::DeviceLocal, usage)
        }?;
        let token = inner.next_token;
        inner.next_token += 1;
        inner.buffers.insert(token, buf);
        Some(BufferView::from_raw(token))
    }

    fn free(&self, view: BufferView) {
        let Ok(mut inner) = self.inner.lock() else {
            return;
        };
        if let Some(buf) = inner.buffers.remove(&view.as_raw()) {
            // SAFETY: the buffer came from this allocator, and every submission touching it was
            // waited on before its copy returned.
            unsafe { inner.alloc.free(buf) };
        }
    }

    fn upload(&self, view: BufferView, offset: usize, src: &[u8]) -> Result<(), String> {
        // `staged_copy` needs `&mut` for the download direction; uploading does not write through
        // it, and copying the slice would double the peak host memory for no benefit.
        let mut scratch = src.to_vec();
        self.staged_copy(view, offset, &mut scratch, true)
    }

    fn download(&self, view: BufferView, offset: usize, dst: &mut [u8]) -> Result<(), String> {
        self.staged_copy(view, offset, dst, false)
    }

    fn is_unified_memory(&self) -> bool {
        self.is_uma
    }
}

/// One provider per device index, stood up on first use and never torn down.
static PROVIDERS: OnceLock<Mutex<HashMap<usize, Option<Arc<HostDeviceMemory>>>>> = OnceLock::new();

/// Ensure a device-memory provider exists for `device_index` and is registered with the engine.
///
/// Idempotent and cheap after the first call. A `None` outcome is cached: if Vulkan could not
/// stand up a device the first time, retrying on every allocation would turn a missing ICD into a
/// per-tensor cost.
/// Device contexts offered by the engine, per device index (§6.5). Populated by
/// [`offer_shared_device`]; empty means the seam has not been called and the fallback applies.
static OFFERED: OnceLock<Mutex<HashMap<usize, Arc<dyn SharedVkDevice>>>> = OnceLock::new();

/// **§6.5 SEAM — the one entry point Switch calls.**
///
/// Hand this module the EP-scoped device context for `device_index`. Call it once, from wherever
/// the device context is constructed, *before* any tensor is allocated — the provider is stood up
/// lazily on the first ORT allocation, and whichever device is on offer at that moment is the one
/// it uses. Calling it later is harmless but has no effect on an already-built provider, and that
/// is reported: `alloc_device_frame` will still say `SPLIT-DEVICE`.
///
/// The `Arc` is retained for the process lifetime. That is deliberate — the `HandleRegistry` this
/// serves is process-global and outlives every session — and it is the only obligation this seam
/// places on the caller.
///
/// **This function is WIRED as of §6.5's closure** (R10). Its production caller is
/// `VulkanSession::create` (`vk/session.rs`), which runs inside `CreateEp` — before ORT can
/// allocate anything — and offers the process-global EP device under `capable.info.index`. Do not
/// read a passing build as evidence that the seam is closed for a given run — read
/// `alloc_device_frame`: `SHARED` means the provider took the offer, `SPLIT-DEVICE` means the
/// index ORT asked for is not one a session offered (see the log emitted at that branch).
pub(crate) fn offer_shared_device(device_index: usize, ctx: Arc<dyn SharedVkDevice>) {
    // R12 obligation 2 (Tank): record WHICH device the session side is on, keyed by the index it
    // offered under. `SPLIT-DEVICE` says the two sides differ; only this says what they are.
    crate::allocator::tally::note_session_device(device_index, ctx.device_name());
    let map = OFFERED.get_or_init(|| Mutex::new(HashMap::new()));
    if let Ok(mut map) = map.lock() {
        map.insert(device_index, ctx);
    }
}

fn offered_device(device_index: usize) -> Option<Arc<dyn SharedVkDevice>> {
    OFFERED.get()?.lock().ok()?.get(&device_index).cloned()
}

/// How an allocator's requested device index resolves against what the sessions have offered.
///
/// This exists because the allocator's index and the session's index come from **different index
/// spaces** and there is no arithmetic that maps one to the other. The allocator's index is the
/// memory-info id of whichever `OrtEpDevice` ORT bound; the session's is the physical index of
/// whichever device the selector opened. They coincide only when ORT's choice happens to equal
/// ours — and a frame that reports `SHARED` because two independent choices agreed on this desk
/// is a coincidence, not a closure: swap the two GPUs and the *other* selector breaks.
///
/// So the rule keys on device **identity** rather than index agreement. §6.5 gives the EP exactly
/// one `VkDevice` per (physical device, EP instance), and `acquire_ep_device` makes it
/// process-global, so when exactly one device has been offered there is nothing to disambiguate:
/// that device *is* the EP's device, whatever index either side calls it.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum OfferResolution {
    /// The requested index was offered directly.
    Exact,
    /// The index missed, but exactly one device is on offer, so identity settles it.
    SoleDevice(usize),
    /// No session has offered anything yet — nothing to adopt.
    NoOffer,
    /// More than one distinct device is on offer and the index matched none of them. This is the
    /// only case that is genuinely ambiguous, and the only one that may stand up a second device.
    Ambiguous,
}

/// The resolution rule, separated from the Vulkan state so it can be tested on both pairings.
pub(crate) fn resolve_offer(requested: usize, offered: &[usize]) -> OfferResolution {
    if offered.contains(&requested) {
        return OfferResolution::Exact;
    }
    match offered {
        [] => OfferResolution::NoOffer,
        [sole] => OfferResolution::SoleDevice(*sole),
        _ => OfferResolution::Ambiguous,
    }
}

/// The sole offered device, when there is exactly one.
fn sole_offered_device() -> Option<(usize, Arc<dyn SharedVkDevice>)> {
    let map = OFFERED.get()?.lock().ok()?;
    if map.len() != 1 {
        return None;
    }
    map.iter().next().map(|(i, c)| (*i, Arc::clone(c)))
}

/// The device indices an EP session has offered a context for, for diagnostics only.
fn offered_indices() -> Vec<usize> {
    let Some(map) = OFFERED.get() else {
        return Vec::new();
    };
    let Ok(map) = map.lock() else {
        return Vec::new();
    };
    let mut v: Vec<usize> = map.keys().copied().collect();
    v.sort_unstable();
    v
}

pub(crate) fn ensure_registered(device_index: usize) {
    let map = PROVIDERS.get_or_init(|| Mutex::new(HashMap::new()));
    let Ok(mut map) = map.lock() else {
        return;
    };
    if map.contains_key(&device_index) {
        return;
    }
    // §6.5: receive a device if one is on offer; only create one if it is not.
    //
    // The index is checked first, but it is not what decides. When it misses and exactly one
    // device is on offer, that device is adopted: the EP owns exactly one `VkDevice`, so a missed
    // index is a naming disagreement between two index spaces, not evidence of a second device.
    // Standing up our own here is what produced `SPLIT-DEVICE` on whichever selector ORT's binding
    // did not happen to match, and it is the defect rather than the report.
    let ctx: Option<(Arc<dyn SharedVkDevice>, DeviceFrame)> = match resolve_offer(
        device_index,
        &offered_indices(),
    ) {
        OfferResolution::Exact => offered_device(device_index).map(|c| (c, DeviceFrame::Shared)),
        OfferResolution::SoleDevice(sole) => {
            let adopted = sole_offered_device().map(|(_, c)| c);
            if let Some(c) = &adopted {
                log::info!(
                    "§6.5: ORT asked for an allocator on factory device index {device_index}, but \
                     the session offered its device under index {sole} ('{}'). These are two \
                     index spaces — the allocator's index is the memory-info id of the \
                     OrtEpDevice ORT bound, the session's is the physical index the selector \
                     opened — and no arithmetic maps one to the other. The EP owns exactly one \
                     VkDevice (§6.5), so identity settles it: adopting the offered device. The \
                     frame is SHARED because both sides are provably on the same VkDevice, not \
                     because two independent index choices happened to agree.",
                    c.device_name(),
                );
            }
            adopted.map(|c| (c, DeviceFrame::Shared))
        }
        OfferResolution::NoOffer | OfferResolution::Ambiguous => {
            log::info!(
                "§6.5: no EP device is on offer for factory device index {device_index} (offered \
                 indices: {:?}). Either no session has been created yet, or more than one distinct \
                 device is on offer and the requested index matched none of them — the only case \
                 that is genuinely ambiguous. Standing up a second VkDevice; the run reports \
                 SPLIT-DEVICE.",
                offered_indices(),
            );
            // SAFETY: the value is stored in a `static` and never dropped, so the loader outlives
            // it.
            unsafe { OwnedDevice::create(device_index) }.map(|d| {
                (
                    Arc::new(d) as Arc<dyn SharedVkDevice>,
                    DeviceFrame::SplitDevice,
                )
            })
        }
    };
    // SAFETY: `ctx` is live and is stored inside the provider, which is stored in a `static`.
    let provider = ctx
        .and_then(|(c, frame)| unsafe { HostDeviceMemory::on_shared_device(c, frame) })
        .map(Arc::new);
    match &provider {
        Some(p) => {
            crate::allocator::tally::set_unified_memory(p.is_unified_memory());
            crate::allocator::tally::set_device_frame(p.frame().as_str(), p.device_name());
            crate::allocator::tally::set_allocator_device_index(device_index);
            crate::engine::register_device_memory_provider(
                device_index,
                Arc::clone(p) as Arc<dyn DeviceMemoryProvider>,
            );
            match p.frame() {
                DeviceFrame::Shared => log::info!(
                    "VulkanExecutionProvider: device-backed allocation is ON and SHARES the \
                     engine's VkDevice (§6.5) on '{}' (unified_memory={}). A compute dispatch can \
                     bind these buffers, so alloc_device_authoritative_spans is now observable — \
                     it is still 0 until the engine calls transfer::device_buffer_for, and that \
                     zero is a measurement rather than a frame artefact.",
                    p.device_name(),
                    p.is_unified_memory()
                ),
                DeviceFrame::SplitDevice => log::warn!(
                    "VulkanExecutionProvider: device-backed allocation is ON but on a SECOND \
                     VkDevice ('{}', unified_memory={}) — §6.5 says that is a defect, not a \
                     design, and it means a compute dispatch CANNOT bind these buffers. Every \
                     alloc_device_* number in this run describes that second device and not the \
                     one the kernels ran on; the artifact records alloc_device_frame = \
                     SPLIT-DEVICE and alloc_device_authoritative_spans = UNOBSERVABLE rather than \
                     0 (R12). Closing this is vk::host_device_memory::offer_shared_device, called \
                     from wherever the EP-scoped device context is built.",
                    p.device_name(),
                    p.is_unified_memory()
                ),
            }
        }
        None => log::warn!(
            "VulkanExecutionProvider: device-backed allocation was requested for device \
             {device_index} but no Vulkan device could be stood up for it. Falling back to host \
             staging, which is correct and slower; alloc_staged_spans will say so."
        ),
    }
    map.insert(device_index, provider);
}

#[cfg(test)]
mod tests {
    use super::{OfferResolution, resolve_offer};

    /// The rule must be SYMMETRIC under swapping the two devices.
    ///
    /// This is Tank's construction test, written as a test. The defect he found was that
    /// `alloc_device_frame` read `SHARED` on selector 0 only because ORT's bound index and the
    /// session's physical index happened to coincide on this desk; on selector 1 they diverged and
    /// the frame read `SPLIT-DEVICE`. His warning was that "a fix that only makes selector 1 pass
    /// on this box is the same coincidence with a different index".
    ///
    /// So both pairings are asserted here with the SAME expectation. A fix that special-cased
    /// either direction — or that mapped one index space onto the other by arithmetic — passes one
    /// of these and fails the other.
    #[test]
    fn a_missed_index_resolves_by_identity_in_both_directions() {
        // ORT bound index 1; the session opened the device it offered under index 0.
        assert_eq!(
            resolve_offer(1, &[0]),
            OfferResolution::SoleDevice(0),
            "asked 1, offered 0: one device exists, so identity settles it"
        );
        // The same desk with the two GPUs swapped: ORT bound 0, the session offered 1.
        assert_eq!(
            resolve_offer(0, &[1]),
            OfferResolution::SoleDevice(1),
            "asked 0, offered 1 must resolve exactly as asked 1, offered 0 does — if these two \
             disagree, the fix is a coincidence with a different index rather than a construction"
        );
    }

    #[test]
    fn an_index_that_was_offered_is_taken_directly() {
        assert_eq!(resolve_offer(1, &[1]), OfferResolution::Exact);
        assert_eq!(resolve_offer(0, &[0]), OfferResolution::Exact);
        assert_eq!(resolve_offer(0, &[0, 1]), OfferResolution::Exact);
    }

    /// The two cases that must still be able to stand up a second device, so `SPLIT-DEVICE` stays
    /// reachable. A detector that can no longer fire is worth less than the defect it reported.
    #[test]
    fn split_device_remains_reachable_when_it_is_the_truth() {
        assert_eq!(
            resolve_offer(1, &[]),
            OfferResolution::NoOffer,
            "no session has run yet: there is nothing to adopt"
        );
        assert_eq!(
            resolve_offer(2, &[0, 1]),
            OfferResolution::Ambiguous,
            "two distinct devices are on offer and neither is the one asked for — this is the \
             only genuinely ambiguous case, and guessing here would be the coincidence again"
        );
    }
}
