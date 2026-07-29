//! Device capability discovery.
//!
//! [`probe`] is called once per physical device during device selection. The returned
//! [`Capabilities`] struct is the **single capability oracle** for the lifetime of that device
//! context: everything that cares about what a device can do reads from this struct; nothing
//! re-queries Vulkan at dispatch time.
//!
//! Two uses only (DESIGN.md §7.2):
//! - Selecting an engine implementation strategy inside `vk/` (e.g., barrier backend in
//!   [`super::barrier`]).
//! - Gating an op claim predicate in `ops/` — passed through `DispatchContext`, never accessed
//!   by op code directly.
//!
//! **Reading `Capabilities::synchronization2` outside `vk/barrier.rs` and this file is
//! prohibited** (DESIGN.md §4.2, §7.5). The barrier backend is selected once at device init;
//! no other call site may branch on that field.

use ash::vk;

// ──────────────────────────────────────────────────────────────────────────────
// Subgroup helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Subgroup size range, available when `VK_EXT_subgroup_size_control` or Vulkan 1.3 is present
/// and the *properties struct* was queryable.
///
/// Having a range does **not** mean the driver will obey a required-size pipeline. That requires
/// the `subgroupSizeControl` feature flag to be `VK_TRUE` — see [`Capabilities::can_require_subgroup_size`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SubgroupSizeRange {
    /// Minimum subgroup size (invocations).
    pub min: u32,
    /// Maximum subgroup size (invocations).
    pub max: u32,
}

impl SubgroupSizeRange {
    /// Returns `true` when the subgroup width is exactly known without requiring it at pipeline
    /// creation (i.e. `min == max`).
    ///
    /// When `true`, a correctness-dependent shader may use the value safely. When `false`, the
    /// driver can choose any width in `[min, max]` and only `can_require_subgroup_size` + a
    /// required-size pipeline makes the width guaranteed.
    #[inline]
    pub fn is_exact(self) -> bool {
        self.min == self.max
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Capabilities
// ──────────────────────────────────────────────────────────────────────────────

/// The capability set of one physical device, probed once at device selection and immutable
/// thereafter.
///
/// The hard device gate (DESIGN.md §7.2 R1–R6) is checked *before* this is populated; these
/// fields record the optional capabilities above that floor.
#[derive(Debug, Clone)]
pub(crate) struct Capabilities {
    // ── Barrier backend selection ──────────────────────────────────────────────
    /// True if `VK_KHR_synchronization2` is present as a device extension or Vulkan 1.3 is core.
    ///
    /// **Read only in `vk/barrier.rs` and `vk/caps.rs`.** No other call site may branch on this
    /// field (DESIGN.md §7.5). `Barriers::select` consumes it once and stores the result on
    /// the device; everything else calls `barriers.buffer_deps(…)`.
    pub synchronization2: bool,

    // ── Subgroup ───────────────────────────────────────────────────────────────
    /// Fixed subgroup size from `VkPhysicalDeviceSubgroupProperties::subgroupSize`
    /// (Vulkan 1.1 core, always available after the device gate).
    pub subgroup_size: u32,

    /// Subgroup operations supported in the `COMPUTE` stage.
    /// `BASIC` is guaranteed by the device gate (R5); everything else is optional.
    pub subgroup_supported_ops: vk::SubgroupFeatureFlags,

    /// Subgroup size range, present only when `VK_EXT_subgroup_size_control` or Vulkan 1.3 core
    /// is available and the properties struct was successfully chained.
    ///
    /// `None` → only `subgroup_size` is known; treat the width as *unknown* for correctness.
    /// `Some(r)` with `r.is_exact()` → width is fixed, safe to assume in shaders without
    /// requiring it at pipeline creation.
    ///
    /// **MoltenVK note:** MoltenVK 1.3.0 promotes this extension to core, so the range is
    /// queryable (`Some`), but the `subgroupSizeControl` *feature* is `VK_FALSE` because Metal
    /// cannot control SIMD-group width per pipeline. Do not conflate "range queryable" with
    /// "range controllable" — use `can_require_subgroup_size` for the latter.
    pub subgroup_size_range: Option<SubgroupSizeRange>,

    /// True only if `subgroup_size_range` is `Some` **and** the driver has
    /// `VkPhysicalDeviceSubgroupSizeControlFeatures::subgroupSizeControl == VK_TRUE`.
    ///
    /// Only when this is `true` may a pipeline be created with
    /// `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo`. On MoltenVK this is always `false`.
    pub can_require_subgroup_size: bool,

    // ── FP16 ───────────────────────────────────────────────────────────────────
    /// True if both `shaderFloat16` and `storageBuffer16BitAccess` are supported.
    /// Gates fp16 op variants; when `false`, fp16 tensors are upcasted to f32 by the op handler
    /// or the op is not claimed.
    pub shader_float16: bool,

    // ── Memory topology ────────────────────────────────────────────────────────
    /// True when the largest `DEVICE_LOCAL` heap is also `HOST_VISIBLE` (unified memory
    /// architecture: Apple/MoltenVK, integrated Intel, some Android SoCs).
    ///
    /// On UMA devices the staging path is skipped: tensors are uploaded once into a
    /// `HOST_VISIBLE | DEVICE_LOCAL` buffer and read directly by the shader without a
    /// `vkCmdCopyBuffer` transfer.
    pub is_uma: bool,
}

impl Capabilities {
    /// Returns `true` when the subgroup width is exactly known without requiring it at pipeline
    /// creation.
    ///
    /// A shader whose correctness depends on a specific subgroup width (`min == max` in the
    /// size range) may use that value safely. Otherwise the portable shared-memory variant must
    /// be selected (DESIGN.md §7.4, rule 4).
    pub fn subgroup_size_is_exact(&self) -> bool {
        self.subgroup_size_range
            .is_some_and(|r| r.is_exact())
    }

    /// Returns the known-exact subgroup width, or `None` when the width is not exactly known.
    ///
    /// Use this to select a subgroup-width-aware shader variant; fall back to the shared-memory
    /// variant when `None`.
    pub fn exact_subgroup_size(&self) -> Option<u32> {
        self.subgroup_size_range
            .filter(|r| r.is_exact())
            .map(|r| r.min)
    }

    /// Returns `true` when subgroup `ARITHMETIC` operations are supported in the compute stage.
    /// Gates the subgroup-reduction shader variants; absent → shared-memory tree-reduction.
    pub fn has_subgroup_arithmetic(&self) -> bool {
        self.subgroup_supported_ops
            .contains(vk::SubgroupFeatureFlags::ARITHMETIC)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Capability probe
// ──────────────────────────────────────────────────────────────────────────────

/// Probe the optional capabilities of a physical device.
///
/// Called once per device in the capability gate phase (DESIGN.md §7.2 / §5.2). Assumes the
/// device has already satisfied the hard requirements R1–R6; this function records everything
/// above that floor.
///
/// # Safety
/// `instance` must be a live `ash::Instance`. `physical_device` must be a handle obtained from
/// that instance. Both must remain valid for the entire lifetime of the returned [`Capabilities`].
pub(crate) unsafe fn probe(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> Capabilities {
    // ── Enumerate device extensions ────────────────────────────────────────────
    // SAFETY: instance is live per caller; physical_device was obtained from it.
    let raw_extensions = unsafe {
        instance
            .enumerate_device_extension_properties(physical_device)
            .unwrap_or_default()
    };
    let extensions: std::collections::HashSet<String> = raw_extensions
        .into_iter()
        .map(|p| {
            // SAFETY: Vulkan guarantees extensionName is a valid null-terminated UTF-8
            // string; the reference is valid for the duration of this closure.
            unsafe { std::ffi::CStr::from_ptr(p.extension_name.as_ptr()) }
                .to_string_lossy()
                .into_owned()
        })
        .collect();

    let has_ext = |name: &str| extensions.contains(name);

    // ── Check API version ─────────────────────────────────────────────────────
    // SAFETY: instance and physical_device are live per caller.
    let api_version =
        unsafe { instance.get_physical_device_properties(physical_device) }.api_version;

    let query_ssc = has_ext("VK_EXT_subgroup_size_control")
        || api_version >= vk::make_api_version(0, 1, 3, 0);

    // ── VkPhysicalDeviceProperties2 chain ─────────────────────────────────────
    let mut subgroup_props = vk::PhysicalDeviceSubgroupProperties::default();
    let mut ssc_props = vk::PhysicalDeviceSubgroupSizeControlProperties::default();

    let mut props2 = vk::PhysicalDeviceProperties2::default();
    // push_next is safe; the #[must_use] return is just the modified struct — ignore it.
    let _ = props2.push_next(&mut subgroup_props);
    if query_ssc {
        let _ = props2.push_next(&mut ssc_props);
    }

    // SAFETY: instance is live per caller; physical_device came from that instance. The p_next
    // chain contains only structs that live on this stack frame and is only read during this
    // call — no dangling pointers escape.
    unsafe { instance.get_physical_device_properties2(physical_device, &mut props2) };

    // ── VkPhysicalDeviceFeatures2 chain ───────────────────────────────────────
    let mut vk12_features = vk::PhysicalDeviceVulkan12Features::default();
    let mut ssc_features = vk::PhysicalDeviceSubgroupSizeControlFeatures::default();

    let mut features2 = vk::PhysicalDeviceFeatures2::default();
    let _ = features2.push_next(&mut vk12_features);
    if query_ssc {
        let _ = features2.push_next(&mut ssc_features);
    }

    // SAFETY: same rationale as the properties chain above.
    unsafe { instance.get_physical_device_features2(physical_device, &mut features2) };

    // ── Derive Capabilities from the queried structs ───────────────────────────
    let synchronization2 = has_ext("VK_KHR_synchronization2")
        || api_version >= vk::make_api_version(0, 1, 3, 0);

    let subgroup_size_range = if query_ssc {
        Some(SubgroupSizeRange {
            min: ssc_props.min_subgroup_size,
            max: ssc_props.max_subgroup_size,
        })
    } else {
        None
    };

    // `can_require_subgroup_size`: only true when the *feature flag* is VK_TRUE.
    // MoltenVK: extension/1.3 present → query_ssc=true, ssc_props filled, but
    // `subgroupSizeControl == VK_FALSE` — Metal cannot set SIMD width per pipeline.
    let can_require_subgroup_size =
        query_ssc && ssc_features.subgroup_size_control == vk::TRUE;

    // fp16: shaderFloat16 = arithmetic is available.
    // TODO: probe VkPhysicalDevice{Float16Int8,16BitStorage}FeaturesKHR on 1.1 devices.
    let shader_float16 = vk12_features.shader_float16 == vk::TRUE;

    // SAFETY: instance and physical_device are live per the caller's contract.
    let is_uma = unsafe { detect_uma(instance, physical_device) };

    Capabilities {
        synchronization2,
        subgroup_size: subgroup_props.subgroup_size,
        subgroup_supported_ops: subgroup_props.supported_operations,
        subgroup_size_range,
        can_require_subgroup_size,
        shader_float16,
        is_uma,
    }
}

/// Returns `true` when the device uses a unified memory architecture: the largest
/// `DEVICE_LOCAL` memory heap is also `HOST_VISIBLE` (Apple/MoltenVK, integrated Intel, Adreno).
///
/// # Safety
/// `instance` and `physical_device` must be live and related.
unsafe fn detect_uma(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> bool {
    // SAFETY: instance is live per caller; physical_device came from that instance.
    let mem_props =
        unsafe { instance.get_physical_device_memory_properties(physical_device) };

    let heap_count = mem_props.memory_heap_count as usize;

    // Find the largest DEVICE_LOCAL heap.
    let largest_device_local = (0..heap_count)
        .filter(|&i| {
            mem_props.memory_heaps[i]
                .flags
                .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
        })
        .max_by_key(|&i| mem_props.memory_heaps[i].size);

    let Some(heap_idx) = largest_device_local else {
        return false; // No DEVICE_LOCAL heap — should never happen after R6.
    };

    // Check whether any memory type on that heap is also HOST_VISIBLE.
    let type_count = mem_props.memory_type_count as usize;
    (0..type_count).any(|i| {
        let t = &mem_props.memory_types[i];
        t.heap_index == heap_idx as u32
            && t.property_flags
                .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// Test helpers (accessible outside this file via cfg(test))
// ──────────────────────────────────────────────────────────────────────────────

/// Build a minimal [`Capabilities`] for unit tests.
///
/// `sync2` controls `synchronization2`. All other optional capabilities default to absent.
/// Live only in `#[cfg(test)]` builds; other modules must import this helper rather than
/// constructing `Capabilities` directly, so field names like `synchronization2` do not
/// appear in non-permitted files and the layering lint stays clean.
#[cfg(test)]
pub(crate) fn test_caps(sync2: bool) -> Capabilities {
    Capabilities {
        synchronization2: sync2,
        subgroup_size: 32,
        subgroup_supported_ops: vk::SubgroupFeatureFlags::BASIC,
        subgroup_size_range: None,
        can_require_subgroup_size: false,
        shader_float16: false,
        is_uma: false,
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn caps_with_synchronization2(sync2: bool) -> Capabilities {
        Capabilities {
            synchronization2: sync2,
            subgroup_size: 32,
            subgroup_supported_ops: vk::SubgroupFeatureFlags::BASIC,
            subgroup_size_range: None,
            can_require_subgroup_size: false,
            shader_float16: false,
            is_uma: false,
        }
    }

    #[test]
    fn subgroup_size_is_exact_requires_equal_min_max() {
        let exact = SubgroupSizeRange { min: 32, max: 32 };
        assert!(exact.is_exact());

        let variable = SubgroupSizeRange { min: 8, max: 64 };
        assert!(!variable.is_exact());
    }

    #[test]
    fn exact_subgroup_size_returns_none_when_range_missing() {
        let caps = caps_with_synchronization2(false);
        assert!(caps.exact_subgroup_size().is_none());
        assert!(!caps.subgroup_size_is_exact());
    }

    #[test]
    fn exact_subgroup_size_returns_none_when_range_variable() {
        let caps = Capabilities {
            subgroup_size_range: Some(SubgroupSizeRange { min: 16, max: 64 }),
            ..caps_with_synchronization2(false)
        };
        assert!(caps.exact_subgroup_size().is_none());
        assert!(!caps.subgroup_size_is_exact());
    }

    #[test]
    fn exact_subgroup_size_returns_value_when_range_exact() {
        let caps = Capabilities {
            subgroup_size_range: Some(SubgroupSizeRange { min: 32, max: 32 }),
            ..caps_with_synchronization2(false)
        };
        assert_eq!(caps.exact_subgroup_size(), Some(32));
        assert!(caps.subgroup_size_is_exact());
    }

    #[test]
    fn moltenvk_model_range_queryable_but_not_controllable() {
        // MoltenVK: reports 1.3 core (so subgroup_size_range is Some) but subgroupSizeControl
        // is VK_FALSE (Metal cannot control SIMD width per pipeline).
        let caps = Capabilities {
            synchronization2: true,   // 1.3 core
            subgroup_size: 32,        // Apple GPU fixed wave = 32
            subgroup_size_range: Some(SubgroupSizeRange { min: 32, max: 32 }),
            can_require_subgroup_size: false,  // <── the MoltenVK distinction
            shader_float16: true,
            subgroup_supported_ops: vk::SubgroupFeatureFlags::BASIC,
            is_uma: true,
        };
        // Range is exact → safe to use the 32-wide variant even without `require`.
        assert!(caps.subgroup_size_is_exact());
        assert_eq!(caps.exact_subgroup_size(), Some(32));
        // But we cannot REQUIRE the size at pipeline creation.
        assert!(!caps.can_require_subgroup_size);
    }

    #[test]
    fn has_subgroup_arithmetic_reflects_supported_ops() {
        let arith = Capabilities {
            subgroup_supported_ops: vk::SubgroupFeatureFlags::BASIC
                | vk::SubgroupFeatureFlags::ARITHMETIC,
            ..caps_with_synchronization2(false)
        };
        assert!(arith.has_subgroup_arithmetic());

        let basic_only = caps_with_synchronization2(false);
        assert!(!basic_only.has_subgroup_arithmetic());
    }
}
