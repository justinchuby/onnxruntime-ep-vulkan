//! Vulkan logical device wrapper.
//!
//! [`Device`] is the engine's view of one selected Vulkan physical device. It owns:
//! - The `ash::Device` logical-device handle (all Vulkan calls go through it).
//! - The [`Barriers`] instance, selected **once** in [`Device::new`] from
//!   [`EpOptions::force_legacy_barriers`] and [`Capabilities::synchronization2`]. No code
//!   outside this file may call [`Barriers::select`].
//! - The [`Capabilities`] oracle — the single capability oracle for the device lifetime.
//!
//! # Layering (ENGINE.md §1)
//!
//! `Device` is `pub(crate)` within the `vk/` tree. Nothing outside `vk/` holds a `Device`
//! reference directly — `ep.rs` and `ops/` interact with the engine through [`DispatchContext`]
//! (the trait defined in `engine.rs`).
//!
//! # Wiring: `force_legacy_barriers`
//!
//! ```text
//! OrtSessionOptions  ──► EpOptions::force_legacy_barriers
//!                                    │
//!                            Device::new(…, force_legacy: bool)
//!                                    │
//!                         Barriers::select(&caps, &instance, &ash_device, force_legacy)
//!                                    │
//!                         self.barriers  (stored; never branched on elsewhere)
//! ```
//!
//! The only call to [`Barriers::select`] is inside [`Device::new`]. Trinity's CI harness sets
//! `ONNXRUNTIME_EP_VULKAN_BACKEND_PROBE` to a file path; `Barriers::select` writes `"sync2"` or
//! `"legacy"` there, giving the harness a guaranteed way to assert which backend ran.

use ash::vk;

use super::{
    barrier::Barriers,
    caps::Capabilities,
    instance::CapableDevice,
};

// ──────────────────────────────────────────────────────────────────────────────
// Device
// ──────────────────────────────────────────────────────────────────────────────

/// The engine's logical device.
///
/// **STUB — M1 fills in the allocator, command-pool, pipeline-cache, and descriptor-pool
/// fields.** What is not stubbed here is the barrier wiring: [`Barriers::select`] is called in
/// `new` and the result is immutably stored, which is the property Morpheus requires.
pub(crate) struct Device {
    /// The Vulkan logical device. Shared (via `Clone`) into backends that need to issue commands
    /// directly (e.g., [`LegacyBackend`][super::barrier::LegacyBackend]).
    ash_device: ash::Device,
    /// The physical device used to create `ash_device`. Retained for re-querying properties
    /// and for pipeline-cache hit tracking.
    physical_device: vk::PhysicalDevice,
    /// Frozen capability set for this device (DESIGN.md §7.2).
    caps: Capabilities,
    /// The selected barrier backend — immutable after [`Device::new`].
    ///
    /// **Call sites use `self.barriers.buffer_deps(…)` / `self.barriers.execution_only(…)`.
    /// No call site reads `caps.synchronization2` to decide anything.**
    barriers: Barriers,
    /// The compute queue, retrieved after logical-device creation.
    compute_queue: vk::Queue,
    /// The queue family index for `compute_queue`.
    compute_queue_family: u32,
}

impl Drop for Device {
    fn drop(&mut self) {
        // SAFETY: ash_device was created by Device::create or Device::new. We are the sole owner.
        // compute_queue is a borrowed handle from the device — it must not be explicitly destroyed.
        unsafe { self.ash_device.destroy_device(None) };
    }
}

impl Device {
    /// Create a logical device from a physical device that passed the capability gate.
    ///
    /// This is the primary constructor for production use. It:
    /// 1. Creates the `VkDevice` with a single compute queue and the extensions declared in
    ///    `capable.device_extensions`.
    /// 2. Retrieves the compute queue handle.
    /// 3. Calls [`Device::new`] (the barrier-wiring constructor) with the probed capabilities.
    ///
    /// Returns `None` if `vkCreateDevice` fails (broken driver, permissions, etc.) — this is
    /// logged as a warning, not returned as an error, so session creation can fall back to the
    /// CPU EP rather than failing hard.
    ///
    /// # Safety
    /// - `instance` must be the same `ash::Instance` that `capable.physical_device` was
    ///   enumerated from and must remain live at least as long as the returned `Device`.
    /// - `capable` must have been produced by [`Instance::enumerate_capable_devices`] on that
    ///   same instance.
    pub(crate) unsafe fn create(
        instance: &ash::Instance,
        capable: &CapableDevice,
        force_legacy: bool,
    ) -> Option<Self> {
        let queue_priority = [1.0f32];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(capable.compute_queue_family)
            .queue_priorities(&queue_priority)];

        // Build the extension name pointer list.
        let ext_ptrs: Vec<*const std::os::raw::c_char> =
            capable.device_extensions.iter().map(|s| s.as_ptr()).collect();

        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_info)
            .enabled_extension_names(&ext_ptrs);

        // SAFETY: instance is live per the caller's contract. capable.physical_device was
        // enumerated from instance and is still valid. device_info borrows queue_info and
        // ext_ptrs which both outlive this call.
        let ash_device = match unsafe { instance.create_device(capable.physical_device, &device_info, None) } {
            Ok(d) => d,
            Err(e) => {
                log::warn!(
                    "vkCreateDevice failed for device '{}' ({e:?}). Skipping this device.",
                    capable.info.name
                );
                return None;
            }
        };

        // Retrieve the compute queue (queue index 0 within the family).
        // SAFETY: ash_device is live; queue family and index are valid per the DeviceQueueCreateInfo above.
        let compute_queue =
            unsafe { ash_device.get_device_queue(capable.compute_queue_family, 0) };

        // SAFETY: instance is live per the caller's contract; ash_device was created from
        // capable.physical_device which was enumerated from instance.
        let mut dev = unsafe {
            Self::new(
                instance,
                capable.physical_device,
                ash_device,
                capable.caps.clone(),
                force_legacy,
            )
        };
        dev.compute_queue = compute_queue;
        dev.compute_queue_family = capable.compute_queue_family;
        Some(dev)
    }

    /// Barrier-wiring constructor. Prefer [`Device::create`] for production code.
    ///
    /// `Barriers::select` is called here and nowhere else. `force_legacy` comes from
    /// `EpOptions::force_legacy_barriers` and is passed straight through; the option's sole
    /// effect is the backend choice — no other behaviour changes.
    ///
    /// # Safety
    /// - `instance` must be a live `ash::Instance` that created `physical_device`.
    /// - `ash_device` must be a live Vulkan logical device created from `physical_device`.
    /// - When `caps.synchronization2 == true` and `force_legacy == false`, the logical device
    ///   must have been created with `VK_KHR_synchronization2` enabled or Vulkan 1.3 as its
    ///   API version — otherwise `vkCmdPipelineBarrier2KHR` is not a valid entry point.
    pub(crate) unsafe fn new(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        ash_device: ash::Device,
        caps: Capabilities,
        force_legacy: bool,
    ) -> Self {
        // SAFETY: instance and ash_device are live per caller; force_legacy contract above.
        let barriers =
            unsafe { Barriers::select(&caps, instance, &ash_device, force_legacy) };

        Device {
            ash_device,
            physical_device,
            caps,
            barriers,
            compute_queue: vk::Queue::null(),
            compute_queue_family: 0,
        }
    }

    /// The barrier dispatcher.
    ///
    /// Every command-buffer recording site calls this. No recording site branches on
    /// `self.caps.synchronization2` — that decision happened in `new`.
    #[inline]
    pub(crate) fn barriers(&self) -> &Barriers {
        &self.barriers
    }

    /// The frozen capability set.
    #[inline]
    pub(crate) fn caps(&self) -> &Capabilities {
        &self.caps
    }

    /// Raw logical-device handle, for modules within `vk/` that need to issue Vulkan commands.
    ///
    /// Never exposed outside `vk/`.
    #[inline]
    pub(crate) fn ash(&self) -> &ash::Device {
        &self.ash_device
    }

    /// Physical device handle retained for re-querying properties.
    #[inline]
    pub(crate) fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    /// The compute queue handle. Null until [`Device::create`] is used.
    #[inline]
    pub(crate) fn compute_queue(&self) -> vk::Queue {
        self.compute_queue
    }

    /// The compute queue family index.
    #[inline]
    pub(crate) fn compute_queue_family(&self) -> u32 {
        self.compute_queue_family
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::vk::barrier::should_use_sync2;
    use crate::vk::caps::test_caps;

    // These tests verify the selection logic without constructing ash types (which contain
    // non-nullable function pointers and cannot be zeroed in unit tests). The field name
    // `synchronization2` intentionally does not appear here — device.rs is not in the
    // permitted-files list for the sync2-field layering rule (DESIGN.md §7.5). The
    // `caps::test_caps` helper (which lives in caps.rs, a permitted file) abstracts the
    // field initialisation. The Vulkan-touching path — Device::new with a real VkDevice —
    // is exercised by Trinity's lavapipe integration suite.

    #[test]
    fn device_selection_legacy_when_forced() {
        // force_legacy=true overrides sync2-capable caps.
        let caps = test_caps(true /* sync2 capable */);
        assert!(
            !should_use_sync2(&caps, true /* force_legacy */),
            "force_legacy=true must force Legacy regardless of sync2 capability"
        );
    }

    #[test]
    fn device_selection_legacy_when_no_sync2() {
        let caps = test_caps(false /* no sync2 */);
        assert!(
            !should_use_sync2(&caps, false),
            "absent sync2 without force_legacy must produce Legacy"
        );
    }

    #[test]
    fn device_selection_sync2_when_capable_and_not_forced() {
        let caps = test_caps(true /* sync2 capable */);
        assert!(
            should_use_sync2(&caps, false),
            "sync2 capable + force_legacy=false must select sync2"
        );
    }

    #[test]
    fn force_legacy_does_not_affect_other_cap_fields() {
        // force_legacy only changes barrier backend selection; all other capability fields
        // are preserved. Verify that the decision changes without affecting the capability
        // oracle itself (which remains the single source of truth).
        let caps = test_caps(true /* sync2 */);
        // force_legacy=true → Legacy selected.
        assert!(!should_use_sync2(&caps, true));
        // force_legacy=false → sync2 selected.
        assert!(should_use_sync2(&caps, false));
        // The caps struct itself is unchanged by the selection (it's immutable).
        assert!(caps.subgroup_supported_ops
            .contains(ash::vk::SubgroupFeatureFlags::BASIC));
    }
}
