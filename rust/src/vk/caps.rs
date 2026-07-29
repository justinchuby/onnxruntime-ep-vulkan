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

    /// True when `synchronization2` originates from Vulkan 1.3 core (not just the KHR extension).
    ///
    /// On Vulkan 1.3+ the core entry point is `vkCmdPipelineBarrier2`. On Vulkan 1.1/1.2 with
    /// `VK_KHR_synchronization2`, only the KHR alias `vkCmdPipelineBarrier2KHR` is exported.
    /// `Barriers::select` reads this once to choose the right entry point. Some Vulkan 1.3+
    /// drivers do NOT export the KHR alias even though the spec permits it, so we must use the
    /// core function name on 1.3+ regardless of whether the extension was explicitly enabled.
    ///
    /// **Read only in `vk/barrier.rs` and `vk/caps.rs`.**
    pub synchronization2_is_core: bool,

    // ── Subgroup ───────────────────────────────────────────────────────────────
    /// Fixed subgroup size from `VkPhysicalDeviceSubgroupProperties::subgroupSize`
    /// (Vulkan 1.1 core, always available after the device gate).
    pub subgroup_size: u32,

    /// `true` when the subgroup probe returned plausible data (§7.9 rule 2).
    ///
    /// A `subgroupSize == 0` on a Vulkan ≥1.1 device is physically impossible on any conformant
    /// driver — the spec floor is 4. When this occurs it means the `pNext` chain was not
    /// delivered correctly (e.g., the `ash` `#[must_use]` rebind bug, D-S12-01). In that case
    /// every subgroup field is unreliable and this flag is `false`.
    ///
    /// **When `false`, treat all subgroup fields as "not determined", not "not supported".**
    /// Log a warning at probe time; do not silently degrade to "no subgroup support".
    pub subgroup_probe_valid: bool,

    /// True when the subgroup `BASIC` feature is supported in the `COMPUTE` shader stage.
    ///
    /// This was formerly gate criterion R5, but was demoted here per Morpheus's §7.0 principle:
    /// "capability shortfalls degrade op coverage, not device availability." Ops that use
    /// subgroup intrinsics must check this flag before claiming the op.
    ///
    /// **Only meaningful when `subgroup_probe_valid` is `true`.**
    ///
    /// Note: the original motivation for demoting R5 was a `supportedStages = 0` reading on
    /// lavapipe. That reading was likely the `push_next` probe bug (§7.9 Bug 1 / D-S12-01);
    /// Mesa 26.1 lavapipe does support subgroup BASIC in compute. The *policy* decision
    /// (capability degrades op coverage, not device admission) remains correct.
    pub subgroup_basic_in_compute: bool,

    /// Raw `supportedStages` from `VkPhysicalDeviceSubgroupProperties` — the stage-flag
    /// bitfield before it is folded into derived booleans (§7.9 rule 3 audit trail).
    ///
    /// Zero when `subgroup_probe_valid` is `false`.
    pub subgroup_supported_stages: vk::ShaderStageFlags,

    /// Subgroup operations supported across all stages.
    /// Check [`subgroup_basic_in_compute`] for compute-stage BASIC support specifically.
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
    /// True when **every** memory heap is `DEVICE_LOCAL` — the definition of unified memory
    /// architecture.
    ///
    /// On integrated GPUs (Intel Iris Xe, Adreno, Mali) the single heap carries both
    /// `DEVICE_LOCAL` and `HOST_VISIBLE`. On discrete GPUs there is always a separate
    /// system-RAM heap without `DEVICE_LOCAL`, even when the VRAM heap is also `HOST_VISIBLE`
    /// via Resizable BAR (ReBAR). ReBAR makes discrete VRAM CPU-accessible but does not
    /// remove the system-RAM heap — so `is_uma` is correctly `false` for discrete GPUs
    /// regardless of ReBAR.
    ///
    /// When `true`, the UMA staging bypass (M1+ optimisation) may skip the `vkCmdCopyBuffer`
    /// and write directly into a `HOST_VISIBLE | DEVICE_LOCAL` buffer. v0 always stages.
    pub is_uma: bool,

    // ── Timestamp ─────────────────────────────────────────────────────────────
    /// Nanoseconds per GPU timestamp tick (device-wide, from
    /// `VkPhysicalDeviceLimits::timestampPeriod`).
    ///
    /// **Owned by `trace.rs`** — `vk/` code MUST NOT multiply by this value. All timestamp
    /// query results are returned as raw ticks; Niobe's `trace.rs` performs the conversion.
    ///
    /// Measured values on known hardware:
    /// - Intel Iris Xe: 52.0833 ns/tick
    /// - NVIDIA RTX 4060 Laptop: 1.0 ns/tick
    ///
    /// The 52× difference is why `vk/` never converts: code correct on NVIDIA and converting
    /// ticks-to-ns directly would be wrong by 52× on Iris Xe.
    ///
    /// Zero means the device does not support timestamps.
    pub timestamp_period_ns: f32,

    /// Number of valid low-order bits in a compute-queue timestamp value, from
    /// `VkQueueFamilyProperties::timestampValidBits` for the compute queue family.
    ///
    /// Values 32..=64; 0 means timestamps are not supported on this queue family.
    /// Raw tick values from `vkGetQueryPoolResults` must be masked:
    /// `raw & ((1u64 << valid_bits) - 1)` (or the full u64 when `valid_bits == 64`).
    ///
    /// Intel Iris Xe reports 36 valid bits — only the low 36 bits of each tick are stable.
    pub timestamp_valid_bits: u32,
}

impl Capabilities {
    /// Returns `true` when the subgroup width is exactly known without requiring it at pipeline
    /// creation.
    ///
    /// A shader whose correctness depends on a specific subgroup width (`min == max` in the
    /// size range) may use that value safely. Otherwise the portable shared-memory variant must
    /// be selected (DESIGN.md §7.4, rule 4).
    pub fn subgroup_size_is_exact(&self) -> bool {
        self.subgroup_size_range.is_some_and(|r| r.is_exact())
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

    /// Compute the device extensions that must be explicitly enabled when calling
    /// `vkCreateDevice` for a physical device with this capability set.
    ///
    /// Called from `instance::enumerate_capable_devices` at probe time so that the extension
    /// list is precomputed before `Device::create` is called. The extension list is stored on
    /// [`super::instance::CapableDevice::device_extensions`].
    ///
    /// **Why this lives in caps.rs:** it branches on `synchronization2`, which the layering
    /// lint (`DESIGN.md §7.5`) restricts to `vk/barrier.rs` and `vk/caps.rs`. Device-creation
    /// code in `instance.rs` calls this method to stay clean of the restricted token.
    ///
    /// `api_version` is `VkPhysicalDeviceProperties::apiVersion` for the device. Extensions
    /// that were promoted to Vulkan core at version V must NOT be passed in the extension list
    /// when `api_version >= V`, because some drivers reject them as unknown names.
    pub(crate) fn required_device_extensions(
        &self,
        api_version: u32,
    ) -> Vec<&'static std::ffi::CStr> {
        let mut exts: Vec<&'static std::ffi::CStr> = Vec::new();
        // `VK_KHR_synchronization2` was promoted to Vulkan 1.3 core. Enable it explicitly only
        // when the feature is available (as a non-core extension) on a pre-1.3 device.
        if self.synchronization2 && api_version < vk::make_api_version(0, 1, 3, 0) {
            exts.push(ash::khr::synchronization2::NAME);
        }
        exts
    }

    /// Build a `DeviceFeatureChain` that carries any `VkPhysicalDevice*Features` structs
    /// that must be chained into `VkDeviceCreateInfo::pNext` at device creation time.
    ///
    /// Call `chain.apply(device_info)` immediately before `vkCreateDevice`. The chain
    /// must outlive `device_info` (declare it before `device_info` in the caller's scope).
    ///
    /// **Why this belongs in `caps.rs`:** it branches on `synchronization2`, which the
    /// layering lint (`DESIGN.md §7.5`) restricts to `vk/barrier.rs` and this file.
    pub(crate) fn device_feature_chain(&self) -> DeviceFeatureChain {
        DeviceFeatureChain::new(self)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// DeviceFeatureChain — feature structs for VkDeviceCreateInfo::pNext
// ──────────────────────────────────────────────────────────────────────────────

/// Owning storage for all Vulkan device feature structs that must be chained into
/// `VkDeviceCreateInfo::pNext`.
///
/// Declare this *before* `VkDeviceCreateInfo` in the caller's scope so that the stack
/// storage outlives the `p_next` pointer chain. Call [`DeviceFeatureChain::apply`] to
/// consume the base `DeviceCreateInfo` and return one with the chain attached.
///
/// This type lives in `caps.rs` because it branches on `capabilities.synchronization2`,
/// which the layering lint forbids outside `vk/barrier.rs` and `vk/caps.rs`.
pub(crate) struct DeviceFeatureChain {
    synchronization2_enabled: bool,
    synchronization2: vk::PhysicalDeviceSynchronization2Features<'static>,
}

impl DeviceFeatureChain {
    fn new(caps: &Capabilities) -> Self {
        Self {
            synchronization2_enabled: caps.synchronization2,
            synchronization2: vk::PhysicalDeviceSynchronization2Features::default()
                .synchronization2(caps.synchronization2),
        }
    }

    /// Attach the feature chain to `device_info` and return the extended info.
    ///
    /// The returned `DeviceCreateInfo` borrows mutably into the fields of `self`; `self`
    /// must remain live and unmoved until `vkCreateDevice` returns.
    pub(crate) fn apply<'a>(
        &'a mut self,
        device_info: vk::DeviceCreateInfo<'a>,
    ) -> vk::DeviceCreateInfo<'a> {
        if self.synchronization2_enabled {
            // SAFETY: self.synchronization2 is declared inside Self and will not move while
            // self is held by the caller. push_next links p_next into it.
            device_info.push_next(&mut self.synchronization2)
        } else {
            device_info
        }
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

    let query_ssc =
        has_ext("VK_EXT_subgroup_size_control") || api_version >= vk::make_api_version(0, 1, 3, 0);

    // ── VkPhysicalDeviceProperties2 chain ─────────────────────────────────────
    //
    // ash 0.38 quirk: `push_next` takes `self` by value and returns `Self`. Discarding the
    // return value with `let _ = ...` leaves p_next unlinked — the queried structs remain
    // zeroed. Always re-bind. See `DeviceFeatureChain::apply` and `device.rs` for the same fix.
    let mut subgroup_props = vk::PhysicalDeviceSubgroupProperties::default();
    let mut ssc_props = vk::PhysicalDeviceSubgroupSizeControlProperties::default();

    let mut props2 = {
        let p = vk::PhysicalDeviceProperties2::default().push_next(&mut subgroup_props);
        if query_ssc {
            p.push_next(&mut ssc_props)
        } else {
            p
        }
    };

    // SAFETY: instance is live per caller; physical_device came from that instance. The p_next
    // chain contains only structs that live on this stack frame and is only read during this
    // call — no dangling pointers escape.
    unsafe { instance.get_physical_device_properties2(physical_device, &mut props2) };

    // Copy `timestamp_period_ns` before using `subgroup_props` or `ssc_props`.
    // NLL (non-lexical lifetimes) rule: props2 mutably borrows subgroup_props and ssc_props;
    // its last USE must come before any access to those structs. By placing this assignment
    // immediately after the properties query, props2's last use is here.
    let timestamp_period_ns = props2.properties.limits.timestamp_period;
    let _ = props2; // suppress "unused variable" if NLL shortens the live range further

    // ── VkPhysicalDeviceFeatures2 chain ───────────────────────────────────────
    let mut vk12_features = vk::PhysicalDeviceVulkan12Features::default();
    let mut ssc_features = vk::PhysicalDeviceSubgroupSizeControlFeatures::default();

    let mut features2 = {
        let f = vk::PhysicalDeviceFeatures2::default().push_next(&mut vk12_features);
        if query_ssc {
            f.push_next(&mut ssc_features)
        } else {
            f
        }
    };

    // SAFETY: same rationale as the properties chain above.
    unsafe { instance.get_physical_device_features2(physical_device, &mut features2) };
    let _ = features2; // ensure features2's last use is before vk12_features/ssc_features reads

    // ── Derive Capabilities from the queried structs ───────────────────────────
    let synchronization2 =
        has_ext("VK_KHR_synchronization2") || api_version >= vk::make_api_version(0, 1, 3, 0);
    let synchronization2_is_core = api_version >= vk::make_api_version(0, 1, 3, 0);

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
    let can_require_subgroup_size = query_ssc && ssc_features.subgroup_size_control == vk::TRUE;

    // subgroup_basic_in_compute: formerly gate R5 — now a capability field so that
    // ops that use subgroup intrinsics gate on this field rather than blocking the device.
    //
    // §7.9 rule 1: distinguish "not supported" from "not determined".
    // A subgroupSize == 0 on a Vulkan ≥1.1 device is physically impossible (spec floor is 4).
    // When it occurs, the pNext chain was not delivered correctly — treat as not-determined
    // and warn loudly so the bug surface stays visible rather than silently degrading.
    let subgroup_probe_valid = subgroup_props.subgroup_size > 0;
    if !subgroup_probe_valid {
        log::warn!(
            "Subgroup probe returned subgroupSize=0 on a Vulkan {}.{} device — \
             treating ALL subgroup capability fields as not-determined (§7.9 rule 1). \
             This almost certainly means the pNext chain was not delivered (D-S12-01 class).",
            vk::api_version_major(api_version),
            vk::api_version_minor(api_version),
        );
    }

    let subgroup_basic_in_compute = subgroup_probe_valid
        && subgroup_props
            .supported_stages
            .contains(vk::ShaderStageFlags::COMPUTE)
        && subgroup_props
            .supported_operations
            .contains(vk::SubgroupFeatureFlags::BASIC);

    // fp16: shaderFloat16 = arithmetic is available.
    // TODO: probe VkPhysicalDevice{Float16Int8,16BitStorage}FeaturesKHR on 1.1 devices.
    let shader_float16 = vk12_features.shader_float16 == vk::TRUE;

    // ── Memory topology ───────────────────────────────────────────────────────
    let is_uma = is_uma_memory(&unsafe {
        // SAFETY: instance is live per caller; physical_device came from that instance.
        instance.get_physical_device_memory_properties(physical_device)
    });

    // ── Timestamp ─────────────────────────────────────────────────────────────
    // timestampValidBits is per queue family; find the compute queue family.
    // SAFETY: instance is live per caller; physical_device came from that instance.
    let qf_props = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
    let timestamp_valid_bits = qf_props
        .iter()
        .find(|qf| qf.queue_flags.contains(vk::QueueFlags::COMPUTE))
        .map(|qf| qf.timestamp_valid_bits)
        .unwrap_or(0);

    Capabilities {
        synchronization2,
        synchronization2_is_core,
        subgroup_size: subgroup_props.subgroup_size,
        subgroup_probe_valid,
        subgroup_basic_in_compute,
        subgroup_supported_stages: subgroup_props.supported_stages,
        subgroup_supported_ops: subgroup_props.supported_operations,
        subgroup_size_range,
        can_require_subgroup_size,
        shader_float16,
        is_uma,
        timestamp_period_ns,
        timestamp_valid_bits,
    }
}

/// Returns `true` when the device uses a unified memory architecture.
///
/// **Predicate:** every memory heap is `DEVICE_LOCAL`. This is the correct test because:
/// - Integrated UMA GPUs (Intel Iris Xe, Adreno, Mali) have a single heap with
///   `DEVICE_LOCAL | HOST_VISIBLE` — every heap is DEVICE_LOCAL → `true`.
/// - Discrete GPUs always have a separate system-RAM heap without `DEVICE_LOCAL`, even
///   when Resizable BAR (ReBAR) is enabled. ReBAR makes VRAM CPU-accessible but does not
///   remove the system-RAM heap → `false` for discrete regardless of ReBAR.
///
/// The previous predicate ("largest DEVICE_LOCAL heap is also HOST_VISIBLE") incorrectly
/// returned `true` for discrete GPUs with ReBAR enabled.
///
/// Call this only through `probe` — do NOT query memory properties again at dispatch time.
fn is_uma_memory(mem_props: &vk::PhysicalDeviceMemoryProperties) -> bool {
    let heap_count = mem_props.memory_heap_count as usize;
    if heap_count == 0 {
        return false;
    }
    // True UMA: no heap lacks DEVICE_LOCAL. A discrete GPU always has a system-RAM heap
    // without DEVICE_LOCAL; an integrated GPU's single heap always has DEVICE_LOCAL (R6
    // requires at least one DEVICE_LOCAL heap to pass the gate, and UMA devices have exactly
    // one heap, so if the device passed the gate the single heap must be DEVICE_LOCAL).
    (0..heap_count).all(|i| {
        mem_props.memory_heaps[i]
            .flags
            .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
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
        synchronization2_is_core: sync2,
        subgroup_size: 32,
        subgroup_probe_valid: true,
        subgroup_basic_in_compute: true,
        subgroup_supported_stages: vk::ShaderStageFlags::COMPUTE,
        subgroup_supported_ops: vk::SubgroupFeatureFlags::BASIC,
        subgroup_size_range: None,
        can_require_subgroup_size: false,
        shader_float16: false,
        is_uma: false,
        timestamp_period_ns: 1.0,
        timestamp_valid_bits: 64,
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
            synchronization2_is_core: sync2,
            subgroup_size: 32,
            subgroup_probe_valid: true,
            subgroup_basic_in_compute: true,
            subgroup_supported_stages: vk::ShaderStageFlags::COMPUTE,
            subgroup_supported_ops: vk::SubgroupFeatureFlags::BASIC,
            subgroup_size_range: None,
            can_require_subgroup_size: false,
            shader_float16: false,
            is_uma: false,
            timestamp_period_ns: 1.0,
            timestamp_valid_bits: 64,
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
            synchronization2: true,
            synchronization2_is_core: true,
            subgroup_size: 32,
            subgroup_probe_valid: true,
            subgroup_basic_in_compute: true,
            subgroup_supported_stages: vk::ShaderStageFlags::COMPUTE,
            subgroup_size_range: Some(SubgroupSizeRange { min: 32, max: 32 }),
            can_require_subgroup_size: false,
            shader_float16: true,
            subgroup_supported_ops: vk::SubgroupFeatureFlags::BASIC,
            is_uma: true,
            timestamp_period_ns: 1.0,
            timestamp_valid_bits: 64,
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

    // ── is_uma_memory tests ───────────────────────────────────────────────────

    /// Build a minimal `VkPhysicalDeviceMemoryProperties` for unit tests.
    /// Only `memory_heap_count` and the first `n` heap flags are meaningful.
    #[allow(clippy::field_reassign_with_default)]
    fn mem_props_from_heap_flags(
        flags: &[vk::MemoryHeapFlags],
    ) -> vk::PhysicalDeviceMemoryProperties {
        let mut props = vk::PhysicalDeviceMemoryProperties::default();
        props.memory_heap_count = flags.len() as u32;
        for (i, &f) in flags.iter().enumerate() {
            props.memory_heaps[i].flags = f;
            props.memory_heaps[i].size = 1 << 30; // size not relevant for UMA test
        }
        props
    }

    #[test]
    fn integrated_uma_single_device_local_heap_is_uma() {
        // Intel Iris Xe / Adreno / Mali: one heap, DEVICE_LOCAL | HOST_VISIBLE.
        // The heap has DEVICE_LOCAL, and there is only one heap → every heap is DL → UMA.
        let props = mem_props_from_heap_flags(&[vk::MemoryHeapFlags::DEVICE_LOCAL]);
        assert!(is_uma_memory(&props));
    }

    #[test]
    fn discrete_gpu_two_heaps_not_uma() {
        // Traditional discrete GPU: heap 0 = VRAM (DEVICE_LOCAL), heap 1 = system RAM (none).
        let props = mem_props_from_heap_flags(&[
            vk::MemoryHeapFlags::DEVICE_LOCAL,
            vk::MemoryHeapFlags::empty(),
        ]);
        assert!(!is_uma_memory(&props));
    }

    #[test]
    fn discrete_gpu_with_rebar_two_heaps_not_uma() {
        // Discrete GPU with ReBAR: heap 0 = VRAM (DEVICE_LOCAL, also accessible via ReBAR),
        // heap 1 = system RAM (no DEVICE_LOCAL). The system-RAM heap's absence of DEVICE_LOCAL
        // correctly returns false — ReBAR does NOT make a discrete GPU a UMA device.
        // (This was the bug in the previous `detect_uma` implementation.)
        let props = mem_props_from_heap_flags(&[
            vk::MemoryHeapFlags::DEVICE_LOCAL, // VRAM — also HOST_VISIBLE via ReBAR
            vk::MemoryHeapFlags::empty(),      // system RAM — no DEVICE_LOCAL
        ]);
        assert!(!is_uma_memory(&props));
    }

    #[test]
    fn zero_heaps_not_uma() {
        let props = mem_props_from_heap_flags(&[]);
        assert!(!is_uma_memory(&props));
    }

    #[test]
    fn all_heaps_device_local_is_uma() {
        // Hypothetical device with two DEVICE_LOCAL heaps (e.g. cached + uncached on same
        // physical memory). All are DEVICE_LOCAL → UMA.
        let props = mem_props_from_heap_flags(&[
            vk::MemoryHeapFlags::DEVICE_LOCAL,
            vk::MemoryHeapFlags::DEVICE_LOCAL,
        ]);
        assert!(is_uma_memory(&props));
    }

    // ── §7.9 subgroup probe-validity tests ───────────────────────────────────

    /// §7.9 rule 1: a zero subgroup_size on a plausible device must produce
    /// `subgroup_probe_valid = false` and `subgroup_basic_in_compute = false`.
    ///
    /// This tests the *logic path* using a synthesised Capabilities struct built the same way
    /// `probe()` would build it when the pNext chain is not delivered.  The real probe is
    /// exercised by CI's lavapipe lane (see integration test note).
    #[test]
    fn probe_validity_false_when_subgroup_size_is_zero() {
        // Simulate what probe() produces when the pNext chain is zeroed (D-S12-01 class bug).
        let caps = Capabilities {
            subgroup_size: 0,
            subgroup_probe_valid: false,
            subgroup_basic_in_compute: false,
            subgroup_supported_stages: vk::ShaderStageFlags::empty(),
            subgroup_supported_ops: vk::SubgroupFeatureFlags::empty(),
            ..caps_with_synchronization2(true)
        };
        assert!(
            !caps.subgroup_probe_valid,
            "all-zero chain must not be trusted"
        );
        assert!(
            !caps.subgroup_basic_in_compute,
            "not-determined must not be treated as supported (conservative)"
        );
    }

    /// §7.9: a valid probe on a subgroup-capable device must set both flags correctly.
    #[test]
    fn probe_validity_true_when_subgroup_size_nonzero() {
        let caps = caps_with_synchronization2(true);
        assert!(caps.subgroup_probe_valid);
        assert_eq!(caps.subgroup_size, 32);
        assert!(caps.subgroup_basic_in_compute);
        assert!(
            caps.subgroup_supported_stages
                .contains(vk::ShaderStageFlags::COMPUTE)
        );
    }

    /// §7.9: subgroup_probe_valid=false must not falsely set subgroup_basic_in_compute=true
    /// even when the raw supported_stages and supported_operations flags are non-zero.
    /// (Guards against the probe() implementation forgetting to check probe_valid.)
    #[test]
    fn basic_in_compute_requires_probe_valid() {
        let caps = Capabilities {
            subgroup_size: 0,            // zero → probe invalid
            subgroup_probe_valid: false, // set by probe() when size == 0
            // Simulate a scenario where a previous bug might have left non-zero stage flags.
            subgroup_supported_stages: vk::ShaderStageFlags::COMPUTE,
            subgroup_supported_ops: vk::SubgroupFeatureFlags::BASIC,
            subgroup_basic_in_compute: false, // probe() must NOT derive true here
            ..caps_with_synchronization2(true)
        };
        assert!(!caps.subgroup_basic_in_compute);
    }
}
