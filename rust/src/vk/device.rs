//! Vulkan logical device wrapper.
//!
//! [`Device`] is the engine's view of one selected Vulkan physical device. It owns:
//! - The `ash::Device` logical-device handle (all Vulkan calls go through it).
//! - The [`Barriers`] instance, selected **once** in [`Device::new`] from
//!   [`EpOptions::force_legacy_barriers`] and the device capability set. No code
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
    caps::{Capabilities, DeviceFeatureChain},
    instance::{CapableDevice, Instance, position_of_physical, select_device, selector_is_pinned},
};

// ──────────────────────────────────────────────────────────────────────────────
// §6.5 — Process-global EP device registry
//
// The rule: exactly one VkDevice per (physical device, EP instance). The session creates the
// device; every other component that needs Vulkan receives it via this registry instead of
// creating its own.
//
// `EpDeviceShare` holds a *borrowed* `ash::Device` — a clone of the session's handle with no
// ownership (no Drop). The session's `Device` owns the VkDevice and calls `vkDestroyDevice` on
// drop. `EpDeviceShare` must not outlive the session; SAFETY below documents why it cannot.
// ──────────────────────────────────────────────────────────────────────────────

use std::sync::OnceLock;

/// Borrowed handles from the EP session's Vulkan device.
///
/// **Does not own the VkDevice** — the session's `Device` does. Do not call `vkDestroyDevice`
/// through these handles. Use only while the `VulkanSession` (and hence its `Device`) is live.
pub(crate) struct EpDeviceShare {
    /// The session's logical device handle, cloned (not reference-counted — raw handle copy).
    /// ash::Device is Clone and Share: cloning copies the dispatch table pointer, not the device.
    pub ash_device: ash::Device,
    /// The physical device from which `ash_device` was created.
    pub physical_device: vk::PhysicalDevice,
    /// Physical device capabilities, frozen at session-creation time.
    pub caps: Capabilities,
    /// The compute queue family. Needed by components that allocate their own command pools.
    pub compute_queue_family: u32,
}

// SAFETY: `ash::Device` is safe to share across threads per the Vulkan spec (Vulkan handles may
// be used from multiple threads provided the caller provides external synchronisation where
// required; ash's documentation notes this). `vk::PhysicalDevice` is a `u64`, trivially Send.
// `Capabilities` is Copy. `EpDeviceShare` is therefore both Send and Sync.
unsafe impl Send for EpDeviceShare {}
// SAFETY: see the comment above `impl Send` — the same reasoning (thread-safe ash handles,
// `u64` physical device, `Copy` capabilities) establishes `Sync`.
unsafe impl Sync for EpDeviceShare {}

/// The process-global device share, set exactly once by `VulkanSession::create` and never
/// changed. `OnceLock` guarantees both write-once and lock-free reads after the first set.
static EP_DEVICE: OnceLock<EpDeviceShare> = OnceLock::new();

/// Register the EP session's device so other components can receive it via [`ep_device`].
///
/// Called exactly once by `VulkanSession::create` after the `Device` is created. Subsequent
/// calls (e.g., from a second session on the same physical device) are silently ignored —
/// §6.5 permits one device per EP instance, and the factory creates at most one session per
/// logical device.
///
/// # Safety
/// The `Device` (and its owning `VulkanSession`) must outlive every use of the returned
/// `EpDeviceShare`. This is guaranteed by §2.3: the `VulkanSession` is EP-scoped, and ORT
/// always frees all tensors (calling any device-memory provider's `free`) before destroying the
/// EP. So no call through `EpDeviceShare` can occur after `VulkanSession::drop`.
pub(crate) fn register_ep_device(device: &Device) {
    EP_DEVICE.get_or_init(|| EpDeviceShare {
        ash_device: device.ash_device.clone(),
        physical_device: device.physical_device,
        caps: device.caps.clone(),
        compute_queue_family: device.compute_queue_family,
    });
}

/// Return the process-global device share, or `None` if the session has not yet been created.
///
/// Components that need a `VkDevice` call this first. If it returns `Some`, they use the shared
/// handles and do not create their own Vulkan instance or device.
#[inline]
pub(crate) fn ep_device() -> Option<&'static EpDeviceShare> {
    EP_DEVICE.get()
}

// ──────────────────────────────────────────────────────────────────────────────
// §6.5 — the owning side: one `VkInstance` and one `VkDevice` per physical device, per process
//
// Why the owner is process-global and not session-scoped, stated as the lifetime argument the
// seam actually needs:
//
// `host_device_memory::OFFERED` / `PROVIDERS` and `HandleRegistry` are process-global by
// construction — ORT's allocator handles outlive any one `OrtEp`. A device context handed across
// that seam therefore has to be valid for the process, not for a session. Before this, every
// `VulkanSession::create` built its own `VkDevice` and `VulkanSession::drop` destroyed it; a
// second session in the same process then ran its kernels on a *different* device than the one
// the cached provider had bound buffers on, and the first `vkDestroyDevice` turned every handle
// the provider held into a dangling one (measured: STATUS_ACCESS_VIOLATION 0xC0000005 on the
// second session's inference).
//
// The fix is not to stop offering the device — it is to stop destroying it. The (instance,
// device) pair is created at most once per physical device and **deliberately leaked**
// (`Box::leak`), so:
//   * the §6.5 invariant holds *across* sessions, not just within one — a second `VulkanEp` on
//     the same physical device receives the same `VkDevice`;
//   * no `vkDestroyDevice` / `vkDestroyInstance` can run while a process-global consumer still
//     holds a handle, so the offer is safe unconditionally and needs no env gate;
//   * per-session children (allocator, command pool, pipeline cache, weight caches) keep their
//     session lifetime and are still destroyed in `VulkanSession::drop`, which is the ordering
//     Vulkan requires (children before device) and is now trivially satisfied.
//
// The cost is one device and one instance held until process exit. That is the same lifetime
// `host_device_memory::OwnedDevice` already takes for the fallback path, and the OS reclaims
// both at exit.
// ──────────────────────────────────────────────────────────────────────────────

/// The process-global owner of one `VkDevice` (§6.5). Never dropped — see the module note above.
pub(crate) struct EpDeviceOwner {
    /// The process-global Vulkan instance the device was created from.
    pub(crate) instance: &'static Instance,
    /// The logical device. Owns the `VkDevice`; its `Drop` never runs (the owner is leaked).
    pub(crate) device: Device,
    /// The capability record for the physical device `device` was created from.
    pub(crate) capable: CapableDevice,
}

/// The one instance, created on the first `acquire_ep_device` and leaked.
/// `None` means Vulkan could not be initialised at all (no loader / no ICD).
static EP_INSTANCE: OnceLock<Option<&'static Instance>> = OnceLock::new();

/// One owner per *physical* device index (`CapableDevice::info::index`).
static EP_DEVICES: OnceLock<std::sync::Mutex<Vec<&'static EpDeviceOwner>>> = OnceLock::new();

/// Acquire the process-global EP device for the physical device this session must open (§6.5).
///
/// Creates the instance and device on first call for a given physical device and returns the
/// same handles on every later call. Returns `None` when no device passes the §7.2 gate.
///
/// # Which device, and the precedence that decides it
///
/// Three things want to name a device and they do not all use the same index space:
///
/// | source | space | precedence |
/// |---|---|---|
/// | `ep.device_index` (session option) | best-first sorted capables | 1 |
/// | `ONNXRUNTIME_EP_VULKAN_DEVICE` (env) | best-first sorted capables | 1 |
/// | `bound_physical` — the `OrtEpDevice` ORT bound for this session | `vkEnumeratePhysicalDevices` | 2 |
/// | best score | best-first sorted capables | 3 |
///
/// **An explicit selector outranks ORT's binding, and that order is deliberate.** The opposite
/// order is defensible on paper — ORT keys the allocator it asks us for by the device it bound,
/// so opening a different one produces the second `VkDevice` §6.5 forbids — and it was tried, and
/// it is worse. Making ORT authoritative meant `ep.device_index = 1` *silently ran on the other
/// GPU*: the run still reported `MATCH`, still claimed 161 nodes, and was only caught because its
/// GPU-busy time was the discrete part's and not the integrated part's. A run that answers a
/// question about a device other than the one it was asked about is not a slower configuration,
/// it is an unattributed one — and the integrated part is this project's spec-conformance oracle,
/// so losing the ability to target it costs more than a `SPLIT-DEVICE` frame.
///
/// So: when the caller names a device, it gets that device, and a divergence from ORT's binding is
/// **logged with both index spaces spelled out** and left to report `SPLIT-DEVICE`, which is true.
/// When the caller names nothing, ORT's binding is followed, which makes the frame `SHARED` in the
/// default configuration without anyone having to ask.
///
/// The divergence has exactly one construction that removes it rather than reporting it: set
/// `ONNXRUNTIME_EP_VULKAN_DEVICE` **before the library is registered**. `engine::devices_to_-
/// advertise` then advertises only that device, ORT cannot bind another, and the two spaces have
/// one member each.
///
/// # Safety
/// The Vulkan loader must remain loaded for the process lifetime. It does: the returned owner is
/// leaked, so `ash::Entry` (which holds the loaded library) is never dropped.
pub(crate) unsafe fn acquire_ep_device(
    bound_physical: Option<usize>,
    device_index: Option<usize>,
    enable_validation: bool,
    force_legacy: bool,
) -> Option<&'static EpDeviceOwner> {
    let instance = (*EP_INSTANCE.get_or_init(|| {
        Instance::create(enable_validation).map(|i| &*Box::leak(Box::new(i)) as &'static Instance)
    }))?;

    if enable_validation && !instance.validation_armed() {
        log::warn!(
            "VulkanSession: validation was requested for this session, but the process-global \
             VkInstance (§6.5) was created without a debug messenger by an earlier session. \
             Validation is an instance-level property and cannot be turned on retroactively; set \
             ONNXRUNTIME_EP_VULKAN_VALIDATE=1 before the first session instead."
        );
    }

    let mut capables = instance.enumerate_capable_devices();
    if capables.is_empty() {
        log::warn!("VulkanSession::create: no devices passed the §7.2 capability gate");
        return None;
    }

    // Did a human name a device, or are we choosing on their behalf? Only the second case may be
    // overridden by ORT's binding.
    let explicit = device_index.is_some() || selector_is_pinned();

    let idx = if let Some(dev_idx) = device_index {
        if dev_idx < capables.len() {
            dev_idx
        } else {
            log::warn!(
                "ep.device_index={dev_idx} is out of range ({} device(s) available); using \
                 device 0",
                capables.len()
            );
            0
        }
    } else {
        select_device(&capables).unwrap_or(0)
    };

    // The one place the two index spaces are allowed to meet. `bound_physical` is an enumeration
    // index; `idx` is a position in the best-first sorted list. They are not interchangeable.
    let bound_pos = bound_physical.and_then(|p| position_of_physical(&capables, p));
    let idx = match (bound_physical, bound_pos) {
        (None, _) => idx,
        (Some(physical), None) => {
            log::warn!(
                "§6.5 index spaces: ORT bound physical enumerate index {physical}, which no \
                 device in the §7.2-capable list carries ({} device(s) enumerated). Using \
                 selector index {idx}; expect SPLIT-DEVICE if ORT asks for an allocator.",
                capables.len(),
            );
            idx
        }
        (Some(_), Some(pos)) if pos == idx => idx,
        (Some(physical), Some(pos)) if explicit => {
            log::warn!(
                "§6.5 index spaces: this session was explicitly asked for '{}' (best-first \
                 selector index {idx}, physical enumerate index {}), but ORT bound '{}' \
                 (selector index {pos}, physical enumerate index {physical}). Honouring the \
                 explicit request — a session that silently runs on a device other than the one \
                 it was asked about produces an UNATTRIBUTED result, which is worse than a split \
                 frame. Consequence: ORT's allocator is keyed to the device it bound, so if it \
                 asks for one the run will correctly report alloc_device_frame = SPLIT-DEVICE. \
                 To remove the divergence instead of reporting it, set \
                 ONNXRUNTIME_EP_VULKAN_DEVICE before the EP library is registered: the factory \
                 then advertises only that device and ORT cannot bind another.",
                capables[idx].info.name,
                capables[idx].info.index,
                capables[pos].info.name,
            );
            idx
        }
        (Some(physical), Some(pos)) => {
            log::info!(
                "§6.5 index spaces: no device was named for this session, so it follows ORT's \
                 binding: '{}' (physical enumerate index {physical}, best-first selector index \
                 {pos}); the best-score default would have been selector index {idx}. Following \
                 ORT keeps one VkDevice and reports alloc_device_frame = SHARED.",
                capables[pos].info.name,
            );
            pos
        }
    };

    let capable = capables.swap_remove(idx);
    let physical_index = capable.info.index;

    let owners = EP_DEVICES.get_or_init(|| std::sync::Mutex::new(Vec::new()));
    let mut owners = owners.lock().ok()?;

    if let Some(existing) = owners
        .iter()
        .find(|o| o.capable.info.index == physical_index)
    {
        log::info!(
            "VulkanSession: reusing the process-global VkDevice for '{}' (physical index {}) — \
             §6.5, exactly one VkDevice per physical device per process.",
            existing.capable.info.name,
            physical_index,
        );
        return Some(*existing);
    }

    log::info!(
        "VulkanSession: selected '{}' (kind={:?} api={} uma={} subgroup_sz={} \
         ts_period={:.4}ns ts_bits={})",
        capable.info.name,
        capable.info.kind,
        capable.info.api_version,
        capable.caps.is_uma,
        capable.caps.subgroup_size,
        capable.caps.timestamp_period_ns,
        capable.caps.timestamp_valid_bits,
    );

    // SAFETY: `instance` is process-lived; `capable` came from `instance.enumerate_capable_devices()`.
    let device = unsafe { Device::create(instance.ash(), &capable, force_legacy) }?;

    let owner: &'static EpDeviceOwner = Box::leak(Box::new(EpDeviceOwner {
        instance,
        device,
        capable,
    }));
    owners.push(owner);
    Some(owner)
}

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
        let ext_ptrs: Vec<*const std::os::raw::c_char> = capable
            .device_extensions
            .iter()
            .map(|s| s.as_ptr())
            .collect();

        // Build the feature chain for VkDeviceCreateInfo::pNext.
        //
        // `DeviceFeatureChain` owns the feature structs and encapsulates all branching on
        // `caps.synchronization2`. This keeps that token out of device.rs in compliance with
        // the layering lint (DESIGN.md §7.5; verified by `layering.rs`).
        //
        // SAFETY: feature_chain must outlive device_info (declared first; same scope). The
        // p_next chain borrowed by device_info points into feature_chain's fields.
        let mut feature_chain: DeviceFeatureChain = capable.caps.device_feature_chain();
        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_info)
            .enabled_extension_names(&ext_ptrs);
        let device_info = feature_chain.apply(device_info);

        // SAFETY: instance is live per the caller's contract. capable.physical_device was
        // enumerated from instance and is still valid. device_info borrows queue_info and
        // ext_ptrs which both outlive this call.
        let ash_device =
            match unsafe { instance.create_device(capable.physical_device, &device_info, None) } {
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
        let compute_queue = unsafe { ash_device.get_device_queue(capable.compute_queue_family, 0) };

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
        let barriers = unsafe { Barriers::select(&caps, instance, &ash_device, force_legacy) };

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
// §6.5 — SessionSharedCtx: the EP's device context for host_device_memory
//
// Holds raw copies of the session's Vulkan handles so `host_device_memory` can
// stand up a `HostDeviceMemory` on the SAME `VkDevice` the kernels run on.
//
// SAFETY: `ash::Device` and `ash::Instance` are handles (integers + Arc'd dispatch
// tables). Cloning them gives a borrowed copy; calling `vkDestroyDevice` remains
// the session's `Device::drop` responsibility.  These copies MUST NOT outlive the
// session's Device — but ORT guarantees tensor frees (the only use of the device
// through this ctx) complete before EP teardown, so no call can reach a destroyed
// device. The Arc held in `host_device_memory::OFFERED` is process-lifetime but
// will never be accessed after the session is torn down.
// ──────────────────────────────────────────────────────────────────────────────

/// Shareable EP device context for `host_device_memory::offer_shared_device`.
///
/// Created in `VulkanSession::create` after the logical device is live, and stored
/// in `host_device_memory::OFFERED` for the process lifetime.  All fields are raw
/// copies of the session's handles; `Drop` does NOT destroy any Vulkan objects.
pub(crate) struct SessionSharedCtx {
    /// Cloned `ash::Instance` handle (ref-counted dispatch table; no `vkDestroyInstance` on drop).
    pub(crate) instance: ash::Instance,
    /// Cloned `ash::Device` handle (ref-counted dispatch table; no `vkDestroyDevice` on drop).
    pub(crate) ash_device: ash::Device,
    pub(crate) physical_device: vk::PhysicalDevice,
    pub(crate) compute_queue: vk::Queue,
    pub(crate) compute_queue_family: u32,
    pub(crate) is_uma: bool,
    pub(crate) name: String,
}

// SAFETY: Vulkan handles are safe to share across threads when external synchronisation is
// provided. All accesses through this ctx go via `HostDeviceMemory`'s internal mutex.
unsafe impl Send for SessionSharedCtx {}
// SAFETY: as above.
unsafe impl Sync for SessionSharedCtx {}

impl super::host_device_memory::SharedVkDevice for SessionSharedCtx {
    fn instance_ash(&self) -> &ash::Instance {
        &self.instance
    }
    fn ash_device(&self) -> &ash::Device {
        &self.ash_device
    }
    fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }
    fn compute_queue_family(&self) -> u32 {
        self.compute_queue_family
    }
    fn compute_queue(&self) -> vk::Queue {
        self.compute_queue
    }
    fn is_uma(&self) -> bool {
        self.is_uma
    }
    fn device_name(&self) -> &str {
        &self.name
    }
}

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
        assert!(
            caps.subgroup_supported_ops
                .contains(ash::vk::SubgroupFeatureFlags::BASIC)
        );
    }
}
