//! The Vulkan engine layer. **Only** this module tree may touch `ash` handles or `gpu-allocator`
//! allocations. Nothing outside `vk/` reaches a `vk::Buffer`, a `vk::CommandBuffer`, a
//! `vk::Pipeline`, or any other raw Vulkan handle — that boundary is the only thing that keeps
//! per-op shader changes from rippling into synchronization, memory, and descriptor code.
//!
//! The boundary is enforced mechanically by `tests/layering.rs`, which fails CI if a forbidden
//! token appears outside this tree.
//!
//! # Module layout (DESIGN.md §3)
//!
//! | Module | Contents |
//! |---|---|
//! | [`instance`] | Vulkan instance, library loader, physical device enumeration and capability gate |
//! | [`caps`] | Device capability discovery — the single capability oracle |
//! | [`barrier`] | Buffer memory barriers — the ONLY module that names Vulkan barrier types |
//! | [`device`] | Logical device wrapper — the ONLY call site for [`barrier::Barriers::select`] |
//! | [`alloc`] | Buffer allocator (`gpu-allocator` backed), memory classes, staging helpers |
//! | [`cmd`] | Command pool and command buffer recording; `submit_and_wait` |
//! | [`pipeline`] | Pipeline cache, descriptor-set layout, push constants, `DispatchDescriptorPool` |
//!
//! # Object creation order
//!
//! ```text
//! ash::Entry::load()           (instance.rs)
//!   └─► vkCreateInstance       (instance.rs — Instance::create)
//!         └─► enumerate + gate (instance.rs — Instance::enumerate_capable_devices)
//!               └─► caps::probe (caps.rs)
//!                     └─► vkCreateDevice (device.rs — Device::create)
//!                           ├─► Barriers::select (barrier.rs — Device::new)
//!                           └─► Allocator::new   (alloc.rs)
//! ```

// Items are built out ahead of engine integration; dead_code is expected at this stage.
#![allow(dead_code)]

pub(crate) mod alloc;
pub(crate) mod barrier;
pub(crate) mod caps;
pub(crate) mod cmd;
pub(crate) mod device;
// Cross-owner (Tank): device-backed memory for ORT-owned tensors. New file, no edits to Switch's.
pub(crate) mod host_device_memory;
pub(crate) mod instance;
pub(crate) mod pipeline;
pub(crate) mod session;

/// End-to-end dispatch integration test. Only compiled in `#[cfg(test)]` builds.
/// Skips silently when shaders are absent or no Vulkan device is available.
#[cfg(test)]
mod dispatch_integration;
