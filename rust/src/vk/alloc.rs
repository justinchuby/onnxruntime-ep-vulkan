//! Vulkan buffer allocator — `gpu-allocator` backed, buffer-only v0.
//!
//! # Design
//!
//! Every device-visible buffer the engine owns is represented by a [`GpuBuffer`], which bundles:
//! - The raw `vk::Buffer` handle.
//! - A `gpu_allocator::vulkan::Allocation` — the sub-allocator block behind the buffer.
//! - The buffer's byte size.
//! - The [`MemClass`] it was allocated in.
//!
//! All `VkBuffer` and `VkDeviceMemory` management goes through [`Allocator`], which wraps a
//! `gpu_allocator::vulkan::Allocator`. Callers (primarily the engine's command-buffer code) never
//! touch `vk::DeviceMemory` directly — the allocator owns it.
//!
//! # Memory classes (`ENGINE.md` §3)
//!
//! | Class | Location | Typical use |
//! |---|---|---|
//! | `DeviceLocal` | `DEVICE_LOCAL` | Compute inputs/outputs, KV-cache, prepacked weights |
//! | `Upload` | `HOST_VISIBLE | HOST_COHERENT` | CPU→GPU staging (write-once per Run) |
//! | `Download` | `HOST_VISIBLE | HOST_COHERENT` | GPU→CPU readback |
//! | `PackedWeights` | `DEVICE_LOCAL` | Compile-time packed weight tensors; never freed mid-run |
//!
//! All four classes map to `VkBuffer` (no images). This is v0's buffer-only decision
//! (`ENGINE.md` §3.1: "buffer-only vs buffer+image: buffer-only for v0").
//!
//! # `ENGINE.md` §3.5.1 — weight prepacking
//!
//! `PackedWeights` buffers are allocated at Compile time and live until the `Plan` is dropped.
//! They are **never** allocated in host-visible memory, which is what makes Mouse's
//! "no dequantized weight ever allocated in device memory" high-water assertion achievable:
//! the only device memory that exists at Compute time holds packed nibbles and scales, not
//! dequantized floats.
//!
//! # `ENGINE.md` §6.2 — barrier coordination
//!
//! [`GpuBuffer`] records its last pipeline stage and access type so the barrier module can
//! compute src masks from the buffer's current state. Op handlers never set these — only
//! the command-buffer recording layer does.
//!
//! # Tank coordination (`ENGINE.md` §5, `DESIGN.md` §6)
//!
//! Tank's handle-based allocator uses generation-stamped quarantine on free: handles are
//! reserved virtual addresses, and the allocator refuses to serve a freed handle until a grace
//! period elapses. The `VkBuffer` + `VkDeviceMemory` behind each handle lives here. Tank calls
//! [`Allocator::alloc`] to create a buffer and gets a [`GpuBuffer`]; his handle table maps
//! `BufferView` tokens → `GpuBuffer`. When the quarantine expires, he calls
//! [`Allocator::free`]. This file does NOT implement the handle table — that is Tank's
//! `engine::BufferView` token side-table in `ep.rs`.

use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{
    Allocation, AllocationCreateDesc, AllocationScheme, Allocator as GpuAllocator,
    AllocatorCreateDesc,
};

// ──────────────────────────────────────────────────────────────────────────────
// Memory classes
// ──────────────────────────────────────────────────────────────────────────────

/// Memory class — controls where a buffer lives (`ENGINE.md` §3).
///
/// Every `VkBuffer` this engine creates belongs to exactly one class for its entire lifetime.
/// Callers pick the class; the allocator picks the heap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum MemClass {
    /// `DEVICE_LOCAL` — compute inputs, outputs, KV-cache.
    DeviceLocal,
    /// `HOST_VISIBLE | HOST_COHERENT` — CPU→GPU staging / upload.
    Upload,
    /// `HOST_VISIBLE | HOST_COHERENT` — GPU→CPU readback / download.
    Download,
    /// `DEVICE_LOCAL` — packed weights; never freed until the `Plan` drops.
    PackedWeights,
}

impl MemClass {
    fn gpu_alloc_location(self) -> MemoryLocation {
        match self {
            MemClass::DeviceLocal | MemClass::PackedWeights => MemoryLocation::GpuOnly,
            MemClass::Upload | MemClass::Download => MemoryLocation::CpuToGpu,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// GpuBuffer
// ──────────────────────────────────────────────────────────────────────────────

/// One device-visible buffer owned by the allocator.
///
/// Freed by [`Allocator::free`]; do not drop this struct directly — the allocation
/// inside it must be returned to the sub-allocator first.
pub(crate) struct GpuBuffer {
    /// The raw buffer handle. Valid until `Allocator::free` is called.
    pub(crate) buffer: vk::Buffer,
    /// The sub-allocator block. Moved into `Allocator::free` on deallocation.
    allocation: Option<Allocation>,
    /// Byte size of this buffer.
    pub(crate) size: u64,
    /// Memory class this buffer was allocated in.
    pub(crate) mem_class: MemClass,
}

impl GpuBuffer {
    /// Pointer to the mapped host memory, for `Upload` / `Download` buffers.
    ///
    /// Returns `None` for `DeviceLocal` / `PackedWeights` buffers.
    pub(crate) fn mapped_ptr(&self) -> Option<*mut u8> {
        self.allocation
            .as_ref()?
            .mapped_ptr()
            .map(|p| p.as_ptr().cast::<u8>())
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Allocator
// ──────────────────────────────────────────────────────────────────────────────

/// Thin wrapper around `gpu_allocator::vulkan::Allocator`.
///
/// All buffer creation and destruction funnels through this struct. The surrounding `vk/`
/// code never calls `vkAllocateMemory` or `vkFreeMemory` directly.
pub(crate) struct Allocator {
    inner: GpuAllocator,
    /// The logical device, borrowed for buffer create/destroy calls.
    ash_device: ash::Device,
}

impl Allocator {
    /// Create an allocator for the given device.
    ///
    /// `instance` and `physical_device` are forwarded to `gpu-allocator` for heap discovery.
    ///
    /// # Safety
    /// - `instance` must be live for the entire lifetime of this `Allocator`.
    /// - `physical_device` must have been selected from `instance` and must remain valid.
    /// - `ash_device` must be the logical device created from `physical_device`.
    pub(crate) unsafe fn new(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        ash_device: &ash::Device,
    ) -> Option<Self> {
        let desc = AllocatorCreateDesc {
            instance: instance.clone(),
            device: ash_device.clone(),
            physical_device,
            debug_settings: Default::default(),
            buffer_device_address: false, // not required in v0 (ENGINE.md §8)
            allocation_sizes: Default::default(),
        };
        match GpuAllocator::new(&desc) {
            Ok(inner) => Some(Self {
                inner,
                ash_device: ash_device.clone(),
            }),
            Err(e) => {
                log::error!("gpu-allocator init failed: {e}");
                None
            }
        }
    }

    /// Allocate a buffer.
    ///
    /// `name` is forwarded to `gpu-allocator`'s debug layer and to validation-layer messages.
    ///
    /// # Safety
    /// The allocator must be live when the returned [`GpuBuffer`] is freed via
    /// [`Allocator::free`].
    pub(crate) unsafe fn alloc(
        &mut self,
        name: &str,
        size: u64,
        class: MemClass,
        usage: vk::BufferUsageFlags,
    ) -> Option<GpuBuffer> {
        if size == 0 {
            log::warn!("Allocator::alloc called with size=0 for '{name}'; returning None");
            return None;
        }

        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        // SAFETY: ash_device is live per Self::new's contract; buffer_info is valid.
        let buffer = unsafe {
            match self.ash_device.create_buffer(&buffer_info, None) {
                Ok(b) => b,
                Err(e) => {
                    log::error!("vkCreateBuffer(size={size}, usage={usage:?}) failed: {e}");
                    return None;
                }
            }
        };

        // SAFETY: buffer is freshly created and valid.
        let requirements = unsafe { self.ash_device.get_buffer_memory_requirements(buffer) };

        let alloc_desc = AllocationCreateDesc {
            name,
            requirements,
            location: class.gpu_alloc_location(),
            linear: true, // buffers are always linear (not tiled)
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };

        let allocation = match self.inner.allocate(&alloc_desc) {
            Ok(a) => a,
            Err(e) => {
                log::error!("gpu-allocator failed to allocate {size} bytes for '{name}': {e}");
                // SAFETY: buffer was created by us and has not been bound.
                unsafe { self.ash_device.destroy_buffer(buffer, None) };
                return None;
            }
        };

        // SAFETY: buffer is valid; allocation.memory() is a valid VkDeviceMemory with
        // allocation.offset() as the correct byte offset.
        unsafe {
            if let Err(e) =
                self.ash_device
                    .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
            {
                log::error!("vkBindBufferMemory failed for '{name}': {e}");
                // Free the gpu-allocator block before dropping the buffer.
                if let Err(fe) = self.inner.free(allocation) {
                    log::error!("  additionally, gpu-allocator free failed: {fe}");
                }
                self.ash_device.destroy_buffer(buffer, None);
                return None;
            }
        }

        Some(GpuBuffer {
            buffer,
            allocation: Some(allocation),
            size,
            mem_class: class,
        })
    }

    /// Free a buffer previously returned by [`Allocator::alloc`].
    ///
    /// # Safety
    /// `buf` must have been produced by this allocator and must not have been freed already.
    pub(crate) unsafe fn free(&mut self, mut buf: GpuBuffer) {
        // Return the sub-allocation block first, then destroy the VkBuffer.
        if let Some(alloc) = buf.allocation.take() {
            if let Err(e) = self.inner.free(alloc) {
                log::error!("gpu-allocator free failed: {e}");
            }
        }
        // SAFETY: buffer was created by us; allocation has been returned above.
        unsafe { self.ash_device.destroy_buffer(buf.buffer, None) };
    }

    /// Convenience: allocate a device-local buffer for general compute I/O.
    ///
    /// `STORAGE_BUFFER` usage + `TRANSFER_DST` so it can receive staged uploads.
    pub(crate) unsafe fn alloc_device(&mut self, name: &str, size: u64) -> Option<GpuBuffer> {
        // SAFETY: forwarding to alloc which has the same contract.
        unsafe {
            self.alloc(
                name,
                size,
                MemClass::DeviceLocal,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            )
        }
    }

    /// Convenience: allocate a packed-weights buffer (device-local, compile-lifetime).
    ///
    /// `STORAGE_BUFFER | TRANSFER_DST` — weights are staged in, then never touched by the CPU.
    /// Satisfies Mouse's "no dequantized weight in device memory" invariant: only nibble-packed
    /// data ever occupies a `PackedWeights` allocation.
    pub(crate) unsafe fn alloc_packed_weights(
        &mut self,
        name: &str,
        size: u64,
    ) -> Option<GpuBuffer> {
        // SAFETY: forwarding to alloc which has the same contract.
        unsafe {
            self.alloc(
                name,
                size,
                MemClass::PackedWeights,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            )
        }
    }

    /// Convenience: allocate an upload (staging) buffer.
    ///
    /// `TRANSFER_SRC` — written by the CPU, read by a `vkCmdCopyBuffer` command.
    pub(crate) unsafe fn alloc_upload(&mut self, name: &str, size: u64) -> Option<GpuBuffer> {
        // SAFETY: forwarding to alloc which has the same contract.
        unsafe {
            self.alloc(
                name,
                size,
                MemClass::Upload,
                vk::BufferUsageFlags::TRANSFER_SRC,
            )
        }
    }

    /// Convenience: allocate a download (readback) buffer.
    ///
    /// `TRANSFER_DST` — written by `vkCmdCopyBuffer`, read by the CPU after a fence.
    pub(crate) unsafe fn alloc_download(&mut self, name: &str, size: u64) -> Option<GpuBuffer> {
        // SAFETY: forwarding to alloc which has the same contract.
        unsafe {
            self.alloc(
                name,
                size,
                MemClass::Download,
                vk::BufferUsageFlags::TRANSFER_DST,
            )
        }
    }
}

impl Drop for Allocator {
    fn drop(&mut self) {
        // gpu_allocator::vulkan::Allocator has its own Drop that logs any leaked allocations.
        // We do not need to explicitly call anything here; the sub-allocator will warn if
        // GpuBuffers were not freed before the Allocator dropped.
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Staging helpers (ENGINE.md §3.3 — upload/download paths)
// ──────────────────────────────────────────────────────────────────────────────

/// Copy `src` bytes into a host-visible `Upload` buffer and record a `vkCmdCopyBuffer`
/// command to transfer the data to `dst` on the device.
///
/// The caller is responsible for inserting a `TRANSFER_WRITE → SHADER_READ` barrier on `dst`
/// after this call (via `Device::barriers().buffer_deps(...)`).
///
/// # Safety
/// - `cmd` must be in the recording state.
/// - `staging` must be a [`MemClass::Upload`] buffer large enough to hold `src`.
/// - `dst` must be a [`MemClass::DeviceLocal`] or [`MemClass::PackedWeights`] buffer.
/// - `ash_device` must be the device that owns all three objects.
pub(crate) unsafe fn record_upload(
    ash_device: &ash::Device,
    cmd: vk::CommandBuffer,
    staging: &GpuBuffer,
    dst: &GpuBuffer,
    src: &[u8],
) {
    debug_assert!(
        staging.mem_class == MemClass::Upload,
        "staging buffer must be MemClass::Upload"
    );
    debug_assert!(
        src.len() as u64 <= staging.size,
        "src ({} bytes) exceeds staging buffer size ({})",
        src.len(),
        staging.size,
    );

    // Write CPU data into the staging buffer's mapped memory.
    if let Some(ptr) = staging.mapped_ptr() {
        // SAFETY: ptr is a valid mapped pointer for at least staging.size bytes; src.len()
        // <= staging.size (asserted above).
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), ptr, src.len()) };
    } else {
        log::error!("record_upload: Upload buffer has no mapped pointer — alloc bug");
        return;
    }

    // Record the copy command.
    let copy_region = [vk::BufferCopy {
        src_offset: 0,
        dst_offset: 0,
        size: src.len() as u64,
    }];
    // SAFETY: cmd is in recording state; all buffers are valid handles from this device.
    unsafe { ash_device.cmd_copy_buffer(cmd, staging.buffer, dst.buffer, &copy_region) };
}

/// Record a `vkCmdCopyBuffer` from a device-local buffer into a host-visible `Download` buffer.
///
/// The caller must:
/// 1. Insert a `SHADER_WRITE → TRANSFER_READ` barrier on `src` *before* this call.
/// 2. Submit and wait (fence) for the queue to drain.
/// 3. Read `download.mapped_ptr()` after the fence.
///
/// # Safety
/// Same requirements as [`record_upload`], with src/dst roles reversed.
pub(crate) unsafe fn record_download(
    ash_device: &ash::Device,
    cmd: vk::CommandBuffer,
    src: &GpuBuffer,
    download: &GpuBuffer,
    size: u64,
) {
    debug_assert!(
        download.mem_class == MemClass::Download,
        "download buffer must be MemClass::Download"
    );
    debug_assert!(
        size <= download.size,
        "copy size ({size}) exceeds download buffer size ({})",
        download.size,
    );

    let copy_region = [vk::BufferCopy {
        src_offset: 0,
        dst_offset: 0,
        size,
    }];
    // SAFETY: cmd is in recording state; all buffers are valid handles from this device.
    unsafe { ash_device.cmd_copy_buffer(cmd, src.buffer, download.buffer, &copy_region) };
}

// ──────────────────────────────────────────────────────────────────────────────
// Unit tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mem_class_device_local_maps_to_gpu_only() {
        assert_eq!(
            MemClass::DeviceLocal.gpu_alloc_location(),
            MemoryLocation::GpuOnly,
        );
    }

    #[test]
    fn mem_class_packed_weights_maps_to_gpu_only() {
        // PackedWeights must be device-local so no dequantized float ever lives in host memory
        // (Mouse's high-water test requirement — OP_COVERAGE.md §8.2.1 P5).
        assert_eq!(
            MemClass::PackedWeights.gpu_alloc_location(),
            MemoryLocation::GpuOnly,
        );
    }

    #[test]
    fn mem_class_upload_maps_to_cpu_to_gpu() {
        assert_eq!(
            MemClass::Upload.gpu_alloc_location(),
            MemoryLocation::CpuToGpu,
        );
    }

    #[test]
    fn mem_class_download_maps_to_cpu_to_gpu() {
        assert_eq!(
            MemClass::Download.gpu_alloc_location(),
            MemoryLocation::CpuToGpu,
        );
    }

    #[test]
    fn gpu_buffer_mapped_ptr_none_when_no_allocation() {
        // A GpuBuffer with allocation=None (e.g., partially constructed) returns None for
        // mapped_ptr. This is the invariant the Drop impl relies on.
        let buf = GpuBuffer {
            buffer: vk::Buffer::null(),
            allocation: None,
            size: 0,
            mem_class: MemClass::DeviceLocal,
        };
        assert!(buf.mapped_ptr().is_none());
    }

    #[test]
    fn mem_class_debug_format_is_stable() {
        // Verify enum variant names are stable (used in debug log messages).
        assert_eq!(format!("{:?}", MemClass::DeviceLocal), "DeviceLocal");
        assert_eq!(format!("{:?}", MemClass::PackedWeights), "PackedWeights");
        assert_eq!(format!("{:?}", MemClass::Upload), "Upload");
        assert_eq!(format!("{:?}", MemClass::Download), "Download");
    }
}
