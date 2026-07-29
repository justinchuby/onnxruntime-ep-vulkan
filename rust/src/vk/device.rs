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
}

impl Device {
    /// Create a new device wrapper.
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
