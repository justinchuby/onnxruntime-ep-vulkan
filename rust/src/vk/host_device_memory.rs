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
//! # The limitation, which is a measured counter and not just a paragraph
//!
//! This provider owns **its own** `Instance`, `Device` and `Allocator`, held for the process
//! lifetime. It does so because [`crate::allocator::HandleRegistry`] is itself process-global
//! (`factory::REGISTRIES`) and outlives every `VulkanSession`: a span allocated during session A
//! may be freed after session A is gone, so a buffer owned by a session's allocator would be a
//! use-after-free waiting to happen. A process-global device makes the two lifetimes identical by
//! construction.
//!
//! The consequence is that **these buffers are on a different `VkDevice` from the one
//! `vk::session` dispatches on, so a compute shader cannot bind them.** That means:
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

/// A process-lifetime Vulkan device used only for ORT tensor residency.
pub(crate) struct HostDeviceMemory {
    inner: Mutex<Inner>,
    device: Device,
    is_uma: bool,
    /// The device the mirror actually landed on.
    ///
    /// Recorded so the identity can be *reported* rather than assumed. `create` resolves the
    /// index against its own `enumerate_capable_devices()` call, and whether that agrees with the
    /// device `VulkanSession` picked is a claim, not a given — it must be checkable from a log.
    device_name: String,
    /// Held so the loader and the instance outlive every buffer. Never used directly.
    _instance: Instance,
}

// SAFETY: every field is either immutable after construction (`device`, `is_uma`, `_instance`) or
// behind the mutex. `ash::Device` and `vk::Queue` are Vulkan handles, which the specification
// permits to be used from any thread provided external synchronisation, which the mutex provides
// for the queue and the pool. Nothing here is thread-affine.
unsafe impl Send for HostDeviceMemory {}
// SAFETY: as above.
unsafe impl Sync for HostDeviceMemory {}

impl HostDeviceMemory {
    /// Stand up a device for `device_index`, or `None` if Vulkan is unavailable.
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
        // `select_device` is the single source of truth for "which device is this session on".
        let idx = crate::vk::instance::select_device(&capables).unwrap_or(0);
        let capable = capables.swap_remove(idx);
        let is_uma = capable.caps.is_uma;
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
        // SAFETY: instance and device are live; the physical device belongs to the instance.
        let alloc =
            unsafe { Allocator::new(instance.ash(), device.physical_device(), device.ash()) }?;
        // SAFETY: device is live; the queue family index came from it.
        let cmd_pool = unsafe { CommandPool::new(device.ash(), device.compute_queue_family()) }?;
        Some(Self {
            inner: Mutex::new(Inner {
                alloc,
                cmd_pool,
                buffers: HashMap::new(),
                next_token: 1,
            }),
            device,
            is_uma,
            device_name,
            _instance: instance,
        })
    }

    /// The device this mirror actually landed on. Reported, never inferred.
    pub(crate) fn device_name(&self) -> &str {
        &self.device_name
    }

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
            self.device
                .ash()
                .cmd_copy_buffer(cmd, src, dst, std::slice::from_ref(&region))
        };
        // SAFETY: `rec` is in the recording state.
        let Some(cmd) = (unsafe { rec.finish() }) else {
            return Err("could not end the command buffer for a device-memory copy".to_string());
        };
        // SAFETY: `cmd` was recorded and ended above; the queue belongs to this device.
        let ok = unsafe { submit_and_wait(self.device.ash(), self.device.compute_queue(), cmd) };
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
pub(crate) fn ensure_registered(device_index: usize) {
    let map = PROVIDERS.get_or_init(|| Mutex::new(HashMap::new()));
    let Ok(mut map) = map.lock() else {
        return;
    };
    if map.contains_key(&device_index) {
        return;
    }
    // SAFETY: the value is stored in a `static` and never dropped, so the loader outlives it.
    let provider = unsafe { HostDeviceMemory::create(device_index) }.map(Arc::new);
    match &provider {
        Some(p) => {
            crate::allocator::tally::set_unified_memory(p.is_unified_memory());
            crate::engine::register_device_memory_provider(
                device_index,
                Arc::clone(p) as Arc<dyn DeviceMemoryProvider>,
            );
            log::info!(
                "VulkanExecutionProvider: device-backed allocation is ON, mirroring onto \
                 '{}' (unified_memory={}). Cross-check this name against the \
                 \"VulkanSession: selected\" line: if they differ, the mirror is on a different \
                 physical device from the kernels and every alloc_device_* number describes the \
                 wrong device. ORT's tensors are resident in DEVICE_LOCAL memory, but these \
                 buffers are on this provider's own VkDevice, so a compute dispatch cannot bind \
                 them yet — alloc_device_backed_spans measures residency, not that the engine \
                 reads from device memory.",
                p.device_name(),
                p.is_unified_memory()
            );
        }
        None => log::warn!(
            "VulkanExecutionProvider: device-backed allocation was requested for device \
             {device_index} but no Vulkan device could be stood up for it. Falling back to host \
             staging, which is correct and slower; alloc_staged_spans will say so."
        ),
    }
    map.insert(device_index, provider);
}
