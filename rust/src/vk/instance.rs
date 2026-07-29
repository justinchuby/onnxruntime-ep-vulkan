//! Vulkan instance wrapper and physical device enumeration.
//!
//! [`Instance`] is the engine's entry point into Vulkan: it owns both the dynamic-library
//! loader ([`ash::Entry`]) and the [`ash::Instance`] created from it. Exactly two things live
//! here:
//!
//! 1. **Instance creation** — [`Instance::create`], which never panics. A machine with no
//!    Vulkan loader returns `None` and logs a warning; calling code receives an empty device
//!    list. This is the "no Vulkan loader, no ICD, or a broken driver" path that
//!    `DESIGN.md §2.3` (M0 exit criterion 4) requires.
//!
//! 2. **Physical device enumeration with the capability gate** —
//!    [`Instance::enumerate_capable_devices`], which applies the six hard requirements from
//!    `DESIGN.md §7.2` (R1–R6) and returns only passing devices, sorted best-first.
//!
//! The gate logic itself lives in [`passes_gate`], a pure function on plain structs with no
//! Vulkan handles — it is fully unit-testable without a Vulkan ICD.
//!
//! # Lifetime discipline
//!
//! `_entry` is declared *before* `handle` but dropped *after* it (Rust drops struct fields in
//! reverse declaration order after [`Drop::drop`] returns). More precisely, in `Drop::drop`
//! both fields are still live, so the explicit `destroy_instance` call runs before either field
//! is dropped. After `drop` returns, `handle` is dropped first (nothing to do — ash provides no
//! `Drop` impl), then `_entry`, which unloads the Vulkan library. The library therefore stays
//! loaded through the `vkDestroyInstance` call.
//!
//! # What is NOT here
//!
//! Logical device creation — [`Device::create`][super::device::Device::create] — lives in
//! [`super::device`]. That separation keeps this module's responsibility to a single question:
//! "which physical devices are usable?"

use std::ffi::{CStr, CString};

use ash::vk;

use super::caps::{self, Capabilities};
use crate::engine::{DeviceInfo, DeviceKind};

// ──────────────────────────────────────────────────────────────────────────────
// Instance
// ──────────────────────────────────────────────────────────────────────────────

/// A live Vulkan instance and its associated library loader.
///
/// Dropped with an explicit `vkDestroyInstance` in [`Drop`].
pub(crate) struct Instance {
    // Fields are declared in drop order (reverse of declaration order after Drop::drop).
    // The _entry must be dropped AFTER handle — so _entry is declared FIRST (dropped LAST).
    _entry: ash::Entry,
    handle: ash::Instance,
}

impl Drop for Instance {
    fn drop(&mut self) {
        // SAFETY: handle was created by _entry. Both are still live inside drop(), so
        // vkDestroyInstance is called with a valid entry-point and a valid instance handle.
        unsafe { self.handle.destroy_instance(None) };
    }
}

impl Instance {
    /// Create a Vulkan instance against Vulkan 1.1.
    ///
    /// Returns `None` — never `Err` — when:
    /// - No Vulkan loader is present (`ash::Entry::load()` fails).
    /// - The loader is present but `vkCreateInstance` fails (e.g., broken driver, no ICD).
    ///
    /// In both cases a `log::warn!` is emitted so the CI lane sees why zero devices are
    /// advertised.
    ///
    /// `enable_validation` requests `VK_LAYER_KHRONOS_validation`; if the layer is not
    /// installed the request is silently dropped (a warning is logged).
    pub(crate) fn create(enable_validation: bool) -> Option<Self> {
        // ── Load the Vulkan library ───────────────────────────────────────────
        // SAFETY: ash::Entry::load() opens the system Vulkan loader (vulkan-1.dll on Windows,
        // libvulkan.so.1 on Linux) via libloading. There is no invariant we need to uphold
        // beyond "the loader path is a valid shared library", which is the OS loader's job.
        let entry = match unsafe { ash::Entry::load() } {
            Ok(e) => e,
            Err(e) => {
                log::warn!(
                    "No Vulkan loader found ({e}). The EP will advertise no devices and every \
                     node stays on the CPU EP."
                );
                return None;
            }
        };

        // ── Optionally enable validation layer ───────────────────────────────
        let mut layer_ptrs: Vec<*const std::os::raw::c_char> = Vec::new();
        let validation_layer_name = c"VK_LAYER_KHRONOS_validation";

        if enable_validation {
            // SAFETY: entry is live; this is a simple property query with no side effects.
            let available =
                unsafe { entry.enumerate_instance_layer_properties() }.unwrap_or_default();
            let present = available.iter().any(|l| {
                // SAFETY: layer_name is a null-terminated char array from the Vulkan driver.
                let name = unsafe { CStr::from_ptr(l.layer_name.as_ptr()) };
                name == validation_layer_name
            });
            if present {
                layer_ptrs.push(validation_layer_name.as_ptr());
            } else {
                log::warn!(
                    "ep.enable_validation=true but VK_LAYER_KHRONOS_validation is not installed; \
                     validation is disabled."
                );
            }
        }

        // ── Build application info ────────────────────────────────────────────
        let app_name = CString::new("onnxruntime-ep-vulkan").expect("no interior NUL in app name");
        let engine_name = CString::new("vulkan-ep").expect("no interior NUL in engine name");

        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(&engine_name)
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            // Request 1.1; drivers that only support 1.0 are filtered out by R1 at enumeration
            // time. Requesting 1.1 is necessary so that 1.1 promoted extensions (subgroup
            // properties, get_physical_device_properties2) are callable without extension strings.
            .api_version(vk::make_api_version(0, 1, 1, 0));

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&layer_ptrs);

        // ── Create the instance ───────────────────────────────────────────────
        // SAFETY: entry is live; app_name/engine_name/layer_ptrs outlive create_info.
        let handle = match unsafe { entry.create_instance(&create_info, None) } {
            Ok(h) => h,
            Err(e) => {
                log::warn!("vkCreateInstance failed ({e:?}). The EP will advertise no devices.");
                return None;
            }
        };

        Some(Instance {
            _entry: entry,
            handle,
        })
    }

    /// The raw `ash::Instance`, for use by other modules within `vk/`.
    #[inline]
    pub(crate) fn ash(&self) -> &ash::Instance {
        &self.handle
    }

    /// Enumerate all physical devices that pass the §7.2 capability gate (R1–R6), sorted
    /// best-first by [`DeviceKind::score`].
    ///
    /// **Never panics.** Errors from Vulkan are logged and produce an empty list.
    pub(crate) fn enumerate_capable_devices(&self) -> Vec<CapableDevice> {
        // SAFETY: handle is live; we pass no invalid pointers.
        let physical_devices = match unsafe { self.handle.enumerate_physical_devices() } {
            Ok(v) => v,
            Err(e) => {
                log::warn!("vkEnumeratePhysicalDevices failed ({e:?}); returning empty list.");
                return Vec::new();
            }
        };

        let mut result = Vec::new();

        for (idx, &pdev) in physical_devices.iter().enumerate() {
            // ── Query properties ─────────────────────────────────────────────
            // SAFETY: handle and pdev are live.
            let props = unsafe { self.handle.get_physical_device_properties(pdev) };

            // Chain in the subgroup properties — Vulkan 1.1 core.
            let mut subgroup_props = vk::PhysicalDeviceSubgroupProperties::default();
            let mut props2 = vk::PhysicalDeviceProperties2::default();
            let _ = props2.push_next(&mut subgroup_props);
            // SAFETY: handle and pdev are live; chain structs on the stack, no dangling refs.
            unsafe {
                self.handle
                    .get_physical_device_properties2(pdev, &mut props2)
            };

            // ── Memory properties ─────────────────────────────────────────────
            // SAFETY: handle and pdev are live.
            let mem_props = unsafe { self.handle.get_physical_device_memory_properties(pdev) };

            // ── Queue families ────────────────────────────────────────────────
            // SAFETY: handle and pdev are live.
            let queue_families = unsafe {
                self.handle
                    .get_physical_device_queue_family_properties(pdev)
            };

            let compute_family = queue_families
                .iter()
                .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .map(|i| i as u32);

            // ── Apply the gate ────────────────────────────────────────────────
            if let Err(reason) = passes_gate(
                &props,
                &props.limits,
                &mem_props,
                compute_family,
                &subgroup_props,
            ) {
                let name = device_name_str(&props);
                log::debug!("Physical device {idx} ({name}): gate failed: {reason}");
                continue;
            }

            // ── Build DeviceInfo ──────────────────────────────────────────────
            let name = device_name_str(&props);
            let kind = device_kind_from_type(props.device_type);
            let api_v = props.api_version;

            let info = DeviceInfo {
                index: idx,
                name,
                vendor_id: props.vendor_id,
                device_id: props.device_id,
                api_version: format!(
                    "{}.{}.{}",
                    vk::api_version_major(api_v),
                    vk::api_version_minor(api_v),
                    vk::api_version_patch(api_v),
                ),
                driver_version: format_driver_version(props.vendor_id, props.driver_version),
                kind,
            };

            // ── Probe optional capabilities ───────────────────────────────────
            // SAFETY: handle and pdev are live per this function's contract.
            let caps = unsafe { caps::probe(&self.handle, pdev) };

            // Precompute which extensions to enable at logical device creation time.
            // caps.rs owns this logic (it reads the synchronization2 field, which the layering
            // lint restricts to caps.rs and barrier.rs — see DESIGN.md §7.5).
            let device_extensions = caps.required_device_extensions(api_v);

            result.push(CapableDevice {
                physical_device: pdev,
                compute_queue_family: compute_family.unwrap(), // safe: gate passed R2
                info,
                caps,
                device_extensions,
            });
        }

        // Best device first (discrete > integrated > virtual > CPU).
        result.sort_by_key(|d| std::cmp::Reverse(d.info.kind.score()));
        result
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// CapableDevice
// ──────────────────────────────────────────────────────────────────────────────

/// A physical device that passed the capability gate, with everything needed to create a
/// logical device from it.
pub(crate) struct CapableDevice {
    /// The physical device handle.
    pub physical_device: vk::PhysicalDevice,
    /// Queue family index for the compute queue (and transfers that share the compute queue).
    pub compute_queue_family: u32,
    /// Device information in the engine's vocabulary (no Vulkan types).
    pub info: DeviceInfo,
    /// Probed optional capabilities.
    pub caps: Capabilities,
    /// Extensions to enable on the logical device. Precomputed: do not pass core-promoted
    /// extensions that the device's API version already includes.
    pub device_extensions: Vec<&'static CStr>,
}

// ──────────────────────────────────────────────────────────────────────────────
// Capability gate (§7.2 R1–R6)
// ──────────────────────────────────────────────────────────────────────────────

/// Apply the §7.2 hard capability requirements to a physical device.
///
/// Returns `Ok(())` if all six requirements pass, or `Err(reason)` describing the first
/// failure. The reason strings are stable diagnostic messages; they appear in debug logs and
/// in Trinity's gate-failure assertions.
///
/// **All parameters are plain structs (no Vulkan handles)** so this function can be
/// exhaustively unit-tested without a Vulkan ICD. See the `tests` module below.
pub(crate) fn passes_gate(
    props: &vk::PhysicalDeviceProperties,
    limits: &vk::PhysicalDeviceLimits,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    compute_queue_family: Option<u32>,
    subgroup_props: &vk::PhysicalDeviceSubgroupProperties<'_>,
) -> Result<(), &'static str> {
    // R1 — Vulkan API version ≥ 1.1.
    if props.api_version < vk::make_api_version(0, 1, 1, 0) {
        return Err("R1: Vulkan API version < 1.1");
    }

    // R2 — At least one compute queue family.
    if compute_queue_family.is_none() {
        return Err("R2: no compute queue family");
    }

    // R3 — Minimum compute workgroup invocations.
    if limits.max_compute_work_group_invocations < 256 {
        return Err("R3: maxComputeWorkGroupInvocations < 256");
    }

    // R4 — Minimum shared memory.
    if limits.max_compute_shared_memory_size < 16384 {
        return Err("R4: maxComputeSharedMemorySize < 16384");
    }

    // R5 — Subgroup BASIC operations in the COMPUTE stage.
    if !subgroup_props
        .supported_stages
        .contains(vk::ShaderStageFlags::COMPUTE)
    {
        return Err("R5a: subgroup not supported in the COMPUTE stage");
    }
    if !subgroup_props
        .supported_operations
        .contains(vk::SubgroupFeatureFlags::BASIC)
    {
        return Err("R5b: BASIC subgroup operations not supported");
    }

    // R6 — At least one DEVICE_LOCAL heap and at least one HOST_VISIBLE memory type.
    let heap_count = mem_props.memory_heap_count as usize;
    let has_device_local = (0..heap_count).any(|i| {
        mem_props.memory_heaps[i]
            .flags
            .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
    });
    if !has_device_local {
        return Err("R6a: no DEVICE_LOCAL memory heap");
    }

    let type_count = mem_props.memory_type_count as usize;
    let has_host_visible = (0..type_count).any(|i| {
        mem_props.memory_types[i]
            .property_flags
            .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
    });
    if !has_host_visible {
        return Err("R6b: no HOST_VISIBLE memory type");
    }

    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Extract the null-terminated device name from the fixed-size `deviceName` array.
fn device_name_str(props: &vk::PhysicalDeviceProperties) -> String {
    // SAFETY: Vulkan guarantees `deviceName` is a null-terminated UTF-8 string.
    unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

fn device_kind_from_type(ty: vk::PhysicalDeviceType) -> DeviceKind {
    match ty {
        vk::PhysicalDeviceType::DISCRETE_GPU => DeviceKind::Discrete,
        vk::PhysicalDeviceType::INTEGRATED_GPU => DeviceKind::Integrated,
        vk::PhysicalDeviceType::VIRTUAL_GPU => DeviceKind::Virtual,
        _ => DeviceKind::Cpu,
    }
}

/// Format a driver version for display.
///
/// NVIDIA encodes the version as `major.minor.subminor.patch` in a non-standard bit layout.
/// AMD and everyone else uses the Vulkan-standard `major.minor.patch`.
fn format_driver_version(vendor_id: u32, driver_version: u32) -> String {
    const VENDOR_NVIDIA: u32 = 0x10DE;
    if vendor_id == VENDOR_NVIDIA {
        // NVIDIA: bits [31:22]=major [21:14]=minor [13:6]=subminor [5:0]=patch
        format!(
            "{}.{}.{}.{}",
            driver_version >> 22,
            (driver_version >> 14) & 0xFF,
            (driver_version >> 6) & 0xFF,
            driver_version & 0x3F,
        )
    } else {
        format!(
            "{}.{}.{}",
            vk::api_version_major(driver_version),
            vk::api_version_minor(driver_version),
            vk::api_version_patch(driver_version),
        )
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal properties that pass all six gate checks.
    fn good_props() -> vk::PhysicalDeviceProperties {
        vk::PhysicalDeviceProperties {
            api_version: vk::make_api_version(0, 1, 1, 0),
            limits: vk::PhysicalDeviceLimits {
                max_compute_work_group_invocations: 1024,
                max_compute_shared_memory_size: 49152, // 48 KiB
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn good_limits(props: &vk::PhysicalDeviceProperties) -> &vk::PhysicalDeviceLimits {
        &props.limits
    }

    #[allow(clippy::field_reassign_with_default)] // array-element assignment can't use struct-literal syntax
    fn good_mem_props() -> vk::PhysicalDeviceMemoryProperties {
        let mut m = vk::PhysicalDeviceMemoryProperties::default();
        m.memory_heap_count = 2;
        m.memory_heaps[0].flags = vk::MemoryHeapFlags::DEVICE_LOCAL;
        m.memory_heaps[0].size = 8 * 1024 * 1024 * 1024; // 8 GiB
        m.memory_heaps[1].size = 16 * 1024 * 1024 * 1024; // 16 GiB host

        m.memory_type_count = 3;
        m.memory_types[0].heap_index = 0;
        m.memory_types[0].property_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        m.memory_types[1].heap_index = 1;
        m.memory_types[1].property_flags =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        m.memory_types[2].heap_index = 0;
        m.memory_types[2].property_flags =
            vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE; // UMA type
        m
    }

    fn good_subgroup_props() -> vk::PhysicalDeviceSubgroupProperties<'static> {
        vk::PhysicalDeviceSubgroupProperties {
            subgroup_size: 32,
            supported_stages: vk::ShaderStageFlags::COMPUTE,
            supported_operations: vk::SubgroupFeatureFlags::BASIC,
            ..Default::default()
        }
    }

    #[test]
    fn good_device_passes_gate() {
        let props = good_props();
        assert!(
            passes_gate(
                &props,
                good_limits(&props),
                &good_mem_props(),
                Some(0),
                &good_subgroup_props()
            )
            .is_ok(),
            "a fully capable device must pass all requirements"
        );
    }

    #[test]
    fn r1_rejects_vulkan_1_0() {
        let mut props = good_props();
        props.api_version = vk::make_api_version(0, 1, 0, 0);
        let r = passes_gate(
            &props,
            good_limits(&props),
            &good_mem_props(),
            Some(0),
            &good_subgroup_props(),
        );
        assert_eq!(r, Err("R1: Vulkan API version < 1.1"));
    }

    #[test]
    fn r2_rejects_no_compute_queue() {
        let props = good_props();
        let r = passes_gate(
            &props,
            good_limits(&props),
            &good_mem_props(),
            None,
            &good_subgroup_props(),
        );
        assert_eq!(r, Err("R2: no compute queue family"));
    }

    #[test]
    fn r3_rejects_low_invocation_count() {
        let mut props = good_props();
        props.limits.max_compute_work_group_invocations = 128;
        let r = passes_gate(
            &props,
            good_limits(&props),
            &good_mem_props(),
            Some(0),
            &good_subgroup_props(),
        );
        assert_eq!(r, Err("R3: maxComputeWorkGroupInvocations < 256"));
    }

    #[test]
    fn r3_accepts_exactly_256_invocations() {
        let mut props = good_props();
        props.limits.max_compute_work_group_invocations = 256;
        assert!(
            passes_gate(
                &props,
                good_limits(&props),
                &good_mem_props(),
                Some(0),
                &good_subgroup_props()
            )
            .is_ok()
        );
    }

    #[test]
    fn r4_rejects_small_shared_memory() {
        let mut props = good_props();
        props.limits.max_compute_shared_memory_size = 8192;
        let r = passes_gate(
            &props,
            good_limits(&props),
            &good_mem_props(),
            Some(0),
            &good_subgroup_props(),
        );
        assert_eq!(r, Err("R4: maxComputeSharedMemorySize < 16384"));
    }

    #[test]
    fn r4_accepts_exactly_16384_shared_memory() {
        let mut props = good_props();
        props.limits.max_compute_shared_memory_size = 16384;
        assert!(
            passes_gate(
                &props,
                good_limits(&props),
                &good_mem_props(),
                Some(0),
                &good_subgroup_props()
            )
            .is_ok()
        );
    }

    #[test]
    fn r5a_rejects_subgroup_not_in_compute_stage() {
        let mut sg = good_subgroup_props();
        sg.supported_stages = vk::ShaderStageFlags::FRAGMENT; // not COMPUTE
        let props = good_props();
        let r = passes_gate(&props, good_limits(&props), &good_mem_props(), Some(0), &sg);
        assert_eq!(r, Err("R5a: subgroup not supported in the COMPUTE stage"));
    }

    #[test]
    fn r5b_rejects_missing_basic_subgroup_ops() {
        let mut sg = good_subgroup_props();
        sg.supported_operations = vk::SubgroupFeatureFlags::empty();
        let props = good_props();
        let r = passes_gate(&props, good_limits(&props), &good_mem_props(), Some(0), &sg);
        assert_eq!(r, Err("R5b: BASIC subgroup operations not supported"));
    }

    #[test]
    fn r6a_rejects_no_device_local_heap() {
        let mut m = good_mem_props();
        // Remove DEVICE_LOCAL flag from all heaps.
        for i in 0..m.memory_heap_count as usize {
            m.memory_heaps[i].flags = vk::MemoryHeapFlags::empty();
        }
        let props = good_props();
        let r = passes_gate(
            &props,
            good_limits(&props),
            &m,
            Some(0),
            &good_subgroup_props(),
        );
        assert_eq!(r, Err("R6a: no DEVICE_LOCAL memory heap"));
    }

    #[test]
    fn r6b_rejects_no_host_visible_type() {
        let mut m = good_mem_props();
        // Remove HOST_VISIBLE from all memory types.
        for i in 0..m.memory_type_count as usize {
            m.memory_types[i].property_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        }
        let props = good_props();
        let r = passes_gate(
            &props,
            good_limits(&props),
            &m,
            Some(0),
            &good_subgroup_props(),
        );
        assert_eq!(r, Err("R6b: no HOST_VISIBLE memory type"));
    }

    #[test]
    fn gate_order_r1_before_r2() {
        // R1 is checked before R2: version failure is reported first.
        let mut props = good_props();
        props.api_version = vk::make_api_version(0, 1, 0, 0);
        let r = passes_gate(
            &props,
            good_limits(&props),
            &good_mem_props(),
            None,
            &good_subgroup_props(),
        );
        assert_eq!(r, Err("R1: Vulkan API version < 1.1"));
    }

    #[test]
    fn driver_version_formats_nvidia_encoding() {
        const NVIDIA: u32 = 0x10DE;
        // 535.86.10 in NVIDIA's encoding:
        // major=535 → 535 << 22 = 0x85C0_0000
        // minor=86  →  86 << 14 = 0x0015_8000
        // sub=10    →  10 << 6  = 0x0000_0280
        // patch=0   →  0        = 0x0000_0000
        let encoded = (535u32 << 22) | (86u32 << 14) | (10u32 << 6);
        let s = format_driver_version(NVIDIA, encoded);
        assert_eq!(s, "535.86.10.0");
    }

    #[test]
    fn driver_version_formats_standard_encoding() {
        const AMD: u32 = 0x1002;
        // Standard Vulkan encoding: same as api_version bits.
        let encoded = vk::make_api_version(0, 2, 0, 260);
        let s = format_driver_version(AMD, encoded);
        assert_eq!(s, "2.0.260");
    }

    #[test]
    fn device_kind_from_type_mapping() {
        assert_eq!(
            device_kind_from_type(vk::PhysicalDeviceType::DISCRETE_GPU),
            DeviceKind::Discrete
        );
        assert_eq!(
            device_kind_from_type(vk::PhysicalDeviceType::INTEGRATED_GPU),
            DeviceKind::Integrated
        );
        assert_eq!(
            device_kind_from_type(vk::PhysicalDeviceType::VIRTUAL_GPU),
            DeviceKind::Virtual
        );
        assert_eq!(
            device_kind_from_type(vk::PhysicalDeviceType::CPU),
            DeviceKind::Cpu
        );
        assert_eq!(
            device_kind_from_type(vk::PhysicalDeviceType::OTHER),
            DeviceKind::Cpu
        );
    }
}
