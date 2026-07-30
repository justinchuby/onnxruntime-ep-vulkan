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

use std::sync::atomic::{AtomicU32, Ordering};

use super::caps::{self, Capabilities};
use crate::engine::{DeviceInfo, DeviceKind};

// ──────────────────────────────────────────────────────────────────────────────
// Validation messenger callback
// ──────────────────────────────────────────────────────────────────────────────

/// Running count of ERROR-severity validation messages received by the EP's messenger.
///
/// Incremented by [`validation_log_callback`] on every ERROR message.  Tests that plant a
/// deliberate validation violation (via `ONNXRUNTIME_EP_VULKAN_PLANT_VALIDATION_VIOLATION`)
/// read this counter to confirm the messenger is wired and the layer fired.
///
/// Reset with [`reset_ep_validation_errors`] before each assertion.
pub(crate) static EP_VALIDATION_ERROR_COUNT: AtomicU32 = AtomicU32::new(0);

/// Reset [`EP_VALIDATION_ERROR_COUNT`] to zero.  Call before running the planted violation so
/// any pre-existing warnings from instance/device creation don't contaminate the assertion.
#[allow(dead_code)] // used only in tests, but always compiled in so the counter is always live
pub(crate) fn reset_ep_validation_errors() {
    EP_VALIDATION_ERROR_COUNT.store(0, Ordering::Relaxed);
}

/// `VkDebugUtilsMessengerCallbackEXT` installed by `Instance::create` whenever
/// `enable_validation = true` and `VK_EXT_debug_utils` is available.
///
/// Routes every validation message through the Rust `log` facade so ORT's session-level
/// logging configuration (log level, output sink) controls where they appear — the same
/// place every other EP log message goes.  This is the property Tank's finding required:
/// the layer must be loaded *and* something in-process must be listening.
///
/// Also increments [`EP_VALIDATION_ERROR_COUNT`] on ERROR messages so test assertions can
/// observe the callback without relying on a log subscriber being installed.
///
/// # Safety
/// Called by the Vulkan loader on an arbitrary thread.  The callback must not call any Vulkan
/// functions and must be safe to invoke concurrently.  Atomics and `log::*` are both safe here.
unsafe extern "system" fn validation_log_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let msg = if data.is_null() {
        "(empty message)".to_owned()
    } else {
        // SAFETY: `data` is a live pointer for the duration of this callback; `p_message`
        // is a NUL-terminated string owned by the validation layer.
        unsafe { CStr::from_ptr((*data).p_message) }
            .to_string_lossy()
            .into_owned()
    };

    if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        EP_VALIDATION_ERROR_COUNT.fetch_add(1, Ordering::Relaxed);
        log::error!("[Vulkan validation] {msg}");
    } else if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) {
        log::warn!("[Vulkan validation] {msg}");
    } else {
        log::debug!("[Vulkan validation] {msg}");
    }

    // Must return VK_FALSE; VK_TRUE is reserved for layer developers.
    vk::FALSE
}

// ──────────────────────────────────────────────────────────────────────────────
// Device-selection environment variable
// ──────────────────────────────────────────────────────────────────────────────

/// Environment variable for pinning the EP to a specific physical device.
///
/// Accepted values:
/// - Unset or empty: use best-first ordering (discrete GPU preferred).
/// - An integer (`"0"`, `"1"`, …): 0-based index into the sorted capable-devices list.
/// - A substring (`"Intel"`, `"RTX"`, …): case-insensitive match against device name;
///   the first match wins.
///
/// Intended uses:
/// - CI lanes: `ONNXRUNTIME_EP_VULKAN_DEVICE=0` runs on the best device (default),
///   `ONNXRUNTIME_EP_VULKAN_DEVICE=Intel` pins to the Intel driver for strictness testing.
/// - Developer machines: select between discrete and integrated for per-device dispatch tests.
///
/// **Intel as a strictness oracle.** Intel's Vulkan drivers are known for strict spec
/// conformance; NVIDIA's for tolerating things the spec does not guarantee. A kernel that
/// is correct on NVIDIA but fails on Intel has almost certainly relied on undefined behaviour,
/// not found an Intel bug. The device selector exists so this asymmetry is exercised
/// deliberately rather than accidentally. See `docs/ENGINE.md §2.1 (Multi-device testing)`.
pub(crate) const ENV_DEVICE_SELECTOR: &str = "ONNXRUNTIME_EP_VULKAN_DEVICE";

/// Select a device index from the sorted capable-devices list using `ONNXRUNTIME_EP_VULKAN_DEVICE`.
///
/// Returns `None` only when `devices` is empty. If the selector names a non-existent index or
/// an unmatched substring, a warning is logged and index 0 (the default best device) is returned.
pub(crate) fn select_device(devices: &[CapableDevice]) -> Option<usize> {
    if devices.is_empty() {
        return None;
    }
    let selector = std::env::var(ENV_DEVICE_SELECTOR).unwrap_or_default();
    if selector.is_empty() {
        return Some(0);
    }
    // Integer index?
    if let Ok(idx) = selector.parse::<usize>() {
        if idx < devices.len() {
            return Some(idx);
        }
        log::warn!(
            "{ENV_DEVICE_SELECTOR}={selector}: index out of range \
             ({} device(s) passed the gate). Using device 0.",
            devices.len(),
        );
        return Some(0);
    }
    // Name substring (case-insensitive).
    let lower = selector.to_lowercase();
    match devices
        .iter()
        .position(|d| d.info.name.to_lowercase().contains(&lower))
    {
        Some(idx) => Some(idx),
        None => {
            log::warn!(
                "{ENV_DEVICE_SELECTOR}={selector}: no device name contains '{selector}' \
                 (case-insensitive). Using device 0.",
            );
            Some(0)
        }
    }
}

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
    /// Optional debug messenger, present when `enable_validation = true` and
    /// `VK_EXT_debug_utils` was available at instance creation.
    ///
    /// The messenger is destroyed explicitly in `Drop` before `vkDestroyInstance`; the
    /// `ash::ext::debug_utils::Instance` caches function pointers that are only valid while
    /// `handle` is alive, so this order is load-bearing.
    debug_messenger: Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
}

impl Drop for Instance {
    fn drop(&mut self) {
        // Destroy the messenger before the instance — the messenger extension's function
        // pointers are resolved against `handle`, which must still be live.
        if let Some((ref du, m)) = self.debug_messenger {
            // SAFETY: `du` was created from `handle`, which is still live here.
            // `m` was created by `create_debug_utils_messenger` and has not been freed.
            unsafe { du.destroy_debug_utils_messenger(m, None) };
        }
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
    ///
    /// The env var `ONNXRUNTIME_EP_VULKAN_VALIDATE` forces validation on regardless of the
    /// session-option flag, so callers can enable it without modifying session config.
    pub(crate) fn create(enable_validation: bool) -> Option<Self> {
        // Allow env-var override: ONNXRUNTIME_EP_VULKAN_VALIDATE=1 forces validation on.
        let enable_validation =
            enable_validation || std::env::var_os("ONNXRUNTIME_EP_VULKAN_VALIDATE").is_some();
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

        // ── Query loader version ──────────────────────────────────────────────
        // vkEnumerateInstanceVersion is a Vulkan 1.1 loader function. If it is absent (Vulkan 1.0
        // loader), requesting apiVersion >= 1.1 in ApplicationInfo WILL cause
        // ERROR_INCOMPATIBLE_DRIVER — the loader checks this before consulting the ICD.
        // SAFETY: entry is live; this is a loader-level query requiring no ICD.
        let loader_version = unsafe { entry.try_enumerate_instance_version() }.unwrap_or(None);

        let api_version_to_request = match loader_version {
            None => {
                log::warn!(
                    "Vulkan loader is version 1.0 (vkEnumerateInstanceVersion not available). \
                     EP requires a Vulkan 1.1+ loader. Loader diagnostic:"
                );
                for line in loader_state_lines(&entry) {
                    log::warn!("{line}");
                }
                return None;
            }
            Some(v) if v < vk::make_api_version(0, 1, 1, 0) => {
                log::warn!(
                    "Vulkan loader reports version {}.{} — EP requires 1.1+. Loader diagnostic:",
                    vk::api_version_major(v),
                    vk::api_version_minor(v)
                );
                for line in loader_state_lines(&entry) {
                    log::warn!("{line}");
                }
                return None;
            }
            // Loader is ≥1.1: request up to Vulkan 1.3.
            //
            // We cap at 1.3 (not higher) because that is the API version the engine has been
            // validated against. More importantly, `vkGetDeviceProcAddr` only returns function
            // pointers for functions up to the instance's requested API version. Requesting 1.3
            // ensures that Vulkan 1.3 core device-level functions like `vkCmdPipelineBarrier2`
            // are available in the device function table built by ash::Instance::create_device.
            // Without this, calling `device.cmd_pipeline_barrier2()` panics with a null function
            // pointer even on a Vulkan 1.4 physical device (the loader gates which functions it
            // exposes to the device based on what the instance requested).
            //
            // The cap at 1.3 prevents ERROR_INCOMPATIBLE_DRIVER on loaders < 1.3. If the loader
            // is 1.1 or 1.2 we request that exact version; the sync2 backend falls back to the
            // KHR extension path for those older devices automatically.
            Some(v) => v.min(vk::make_api_version(0, 1, 3, 0)),
        };

        // ── Pre-creation diagnostic (verbose only) ────────────────────────────
        let verbose = std::env::var(crate::logging::ENV_VERBOSE).as_deref() == Ok("1");
        if verbose {
            log::info!(
                "[loader-probe] Before vkCreateInstance — loader version {}.{}.{}:",
                vk::api_version_major(loader_version.unwrap()),
                vk::api_version_minor(loader_version.unwrap()),
                vk::api_version_patch(loader_version.unwrap())
            );
            for line in loader_state_lines(&entry) {
                log::info!("{line}");
            }
        }

        // ── Optionally enable validation layer ───────────────────────────────
        let mut layer_ptrs: Vec<*const std::os::raw::c_char> = Vec::new();
        let validation_layer_name = c"VK_LAYER_KHRONOS_validation";

        // Whether VK_EXT_debug_utils was found and should be requested.
        let mut want_debug_utils = false;
        let mut ext_ptrs: Vec<*const std::os::raw::c_char> = Vec::new();

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

            // Also request VK_EXT_debug_utils so we can install a messenger and route
            // validation errors through the Rust log facade (the layer being loaded is not
            // enough — without a messenger nothing in-process is listening).
            let ext_name = c"VK_EXT_debug_utils";
            // SAFETY: entry is live; this is a simple property query with no side effects.
            let available_exts =
                unsafe { entry.enumerate_instance_extension_properties(None) }.unwrap_or_default();
            want_debug_utils = available_exts.iter().any(|e| {
                // SAFETY: extension_name is a NUL-terminated array from the driver.
                unsafe { CStr::from_ptr(e.extension_name.as_ptr()) == ext_name }
            });
            if want_debug_utils {
                ext_ptrs.push(ext_name.as_ptr());
            } else {
                log::warn!(
                    "ep.enable_validation=true but VK_EXT_debug_utils is not available; \
                     validation errors will not be routed through the Rust log facade."
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
            // api_version_to_request is capped to the loader's reported version above.
            // Requesting a higher version than the loader supports returns INCOMPATIBLE_DRIVER.
            .api_version(api_version_to_request);

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&layer_ptrs)
            .enabled_extension_names(&ext_ptrs);

        // ── Create the instance ───────────────────────────────────────────────
        // SAFETY: entry is live; app_name/engine_name/layer_ptrs/ext_ptrs outlive create_info.
        let handle = match unsafe { entry.create_instance(&create_info, None) } {
            Ok(h) => h,
            Err(vk::Result::ERROR_INCOMPATIBLE_DRIVER) => {
                // This is the most actionable failure: the loader found no usable ICD, or the
                // ICD DLL/so could not be loaded. The full loader state is always emitted here
                // regardless of verbose mode — this is the log that makes it diagnosable.
                log::warn!(
                    "vkCreateInstance failed (ERROR_INCOMPATIBLE_DRIVER). The loader found no \
                     usable ICD or the ICD library is not loadable. Loader diagnostic:"
                );
                for line in loader_state_lines(&entry) {
                    log::warn!("{line}");
                }
                log::warn!(
                    "  Hint: set VK_DRIVER_FILES (preferred) in addition to VK_ICD_FILENAMES, \
                     verify the ICD DLL/so path and its dependencies, and confirm that \
                     VK_LAYER_KHRONOS_validation is findable at VK_LAYER_PATH."
                );
                return None;
            }
            Err(e) => {
                log::warn!(
                    "vkCreateInstance failed ({e:?}). The EP will advertise no devices. \
                            Loader diagnostic:"
                );
                for line in loader_state_lines(&entry) {
                    log::warn!("{line}");
                }
                return None;
            }
        };

        // ── Optionally install validation messenger ───────────────────────────
        // The messenger is only useful when both the layer and the extension loaded.  Without a
        // messenger, the layer may write to stderr via its own default handler but nothing
        // in-process is listening — validation errors are invisible to the Rust log facade and
        // to any test assertion that reads log output.
        let debug_messenger = if want_debug_utils {
            let du = ash::ext::debug_utils::Instance::new(&entry, &handle);
            let messenger_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
                )
                .message_type(vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION)
                .pfn_user_callback(Some(validation_log_callback));
            // SAFETY: `du` was built from `handle` which is live; `messenger_info` is valid.
            match unsafe { du.create_debug_utils_messenger(&messenger_info, None) } {
                Ok(m) => {
                    log::debug!(
                        "VkDebugUtilsMessengerEXT installed; validation errors routed to log::error!"
                    );
                    Some((du, m))
                }
                Err(e) => {
                    log::warn!(
                        "create_debug_utils_messenger failed ({e:?}); validation errors will not be captured in-process."
                    );
                    None
                }
            }
        } else {
            None
        };

        Some(Instance {
            _entry: entry,
            handle,
            debug_messenger,
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
            if let Err(reason) = passes_gate(&props, &props.limits, &mem_props, compute_family) {
                let name = device_name_str(&props);
                log::debug!("Physical device {idx} ({name}): gate failed: {reason}");
                // Emit per-criterion breakdown at DEBUG so a user who sets RUST_LOG=debug
                // gets the full measured values without any runtime overhead in production.
                if log::log_enabled!(log::Level::Debug) {
                    for c in assess_gate(&props, &props.limits, &mem_props, compute_family) {
                        log::debug!("{}", c.row());
                    }
                }
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
// Capability gate (§7.2 R1–R4, R6)
// ──────────────────────────────────────────────────────────────────────────────

/// One evaluated criterion in the §7.2 device gate.
///
/// Holds the human-readable description, measured value, pass/fail verdict, and the stable
/// error string that [`passes_gate`] returns on failure (Trinity's tests assert on those
/// strings).
pub(crate) struct GateCriterion {
    /// Short display label, e.g. `"R1  Vulkan API version"`.
    pub label: &'static str,
    /// Requirement in human-readable form, e.g. `">= 1.1"`.
    pub requirement: &'static str,
    /// Measured value from the device, e.g. `"1.3.255"`.
    pub measured: String,
    /// `true` when the criterion is satisfied.
    pub passed: bool,
    /// The error string returned by [`passes_gate`] when this criterion fails.
    /// **Stable** — Trinity's gate-failure assertions match these strings exactly.
    failure_reason: &'static str,
}

impl GateCriterion {
    /// Format one row for probe/diagnostic output.
    pub fn row(&self) -> String {
        let verdict = if self.passed { "PASS" } else { "FAIL ←" };
        format!(
            "  {:<44} {:<22} {}",
            format!("{} (req. {})", self.label, self.requirement),
            self.measured,
            verdict,
        )
    }
}

/// Evaluate all §7.2 gate criteria (R1–R4, R6) against a device's reported properties.
///
/// Unlike [`passes_gate`], this **does not short-circuit**: every criterion is evaluated and
/// its measured value is recorded regardless of earlier failures. Use this for diagnostic and
/// probe output; use [`passes_gate`] in the hot-path device enumeration loop.
///
/// **R5 is absent.** Subgroup BASIC in COMPUTE was demoted from the hard gate to
/// [`Capabilities::subgroup_basic_in_compute`] per Morpheus's §7.0 governing principle:
/// *"capability shortfalls degrade op coverage, not device availability."* Software renderers
/// (lavapipe/llvmpipe) lack subgroup support in compute but are still valid EP devices — ops
/// that use subgroup intrinsics gate on the capability field instead.
///
/// All parameters are plain structs (no Vulkan handles), so this function is fully unit-testable
/// without a live ICD.
pub(crate) fn assess_gate(
    props: &vk::PhysicalDeviceProperties,
    limits: &vk::PhysicalDeviceLimits,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    compute_queue_family: Option<u32>,
) -> Vec<GateCriterion> {
    let mut criteria = Vec::with_capacity(6);

    // R1 — Vulkan API version >= 1.1
    let api_v = props.api_version;
    criteria.push(GateCriterion {
        label: "R1  Vulkan API version",
        requirement: ">= 1.1",
        measured: format!(
            "{}.{}.{}",
            vk::api_version_major(api_v),
            vk::api_version_minor(api_v),
            vk::api_version_patch(api_v),
        ),
        passed: api_v >= vk::make_api_version(0, 1, 1, 0),
        failure_reason: "R1: Vulkan API version < 1.1",
    });

    // R2 — At least one compute queue family
    criteria.push(GateCriterion {
        label: "R2  compute queue family",
        requirement: "present",
        measured: compute_queue_family
            .map(|f| format!("family {f}"))
            .unwrap_or_else(|| "absent".to_string()),
        passed: compute_queue_family.is_some(),
        failure_reason: "R2: no compute queue family",
    });

    // R3 — Minimum compute workgroup invocations
    let inv = limits.max_compute_work_group_invocations;
    criteria.push(GateCriterion {
        label: "R3  maxComputeWorkGroupInvocations",
        requirement: ">= 256",
        measured: inv.to_string(),
        passed: inv >= 256,
        failure_reason: "R3: maxComputeWorkGroupInvocations < 256",
    });

    // R4 — Minimum shared memory
    let shm = limits.max_compute_shared_memory_size;
    criteria.push(GateCriterion {
        label: "R4  maxComputeSharedMemorySize",
        requirement: ">= 16384 B",
        measured: format!("{shm} B ({} KiB)", shm / 1024),
        passed: shm >= 16384,
        failure_reason: "R4: maxComputeSharedMemorySize < 16384",
    });

    // R6a — At least one DEVICE_LOCAL heap
    let heap_count = mem_props.memory_heap_count as usize;
    let dl_heap = (0..heap_count).find(|&i| {
        mem_props.memory_heaps[i]
            .flags
            .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
    });
    criteria.push(GateCriterion {
        label: "R6a DEVICE_LOCAL memory heap",
        requirement: "at least 1",
        measured: dl_heap
            .map(|i| format!("heap {i}"))
            .unwrap_or_else(|| "none".to_string()),
        passed: dl_heap.is_some(),
        failure_reason: "R6a: no DEVICE_LOCAL memory heap",
    });

    // R6b — At least one HOST_VISIBLE memory type
    let type_count = mem_props.memory_type_count as usize;
    let hv_type = (0..type_count).find(|&i| {
        mem_props.memory_types[i]
            .property_flags
            .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
    });
    criteria.push(GateCriterion {
        label: "R6b HOST_VISIBLE memory type",
        requirement: "at least 1",
        measured: hv_type
            .map(|i| format!("type {i}"))
            .unwrap_or_else(|| "none".to_string()),
        passed: hv_type.is_some(),
        failure_reason: "R6b: no HOST_VISIBLE memory type",
    });

    criteria
}

/// Apply the §7.2 hard capability requirements (R1–R4, R6) to a physical device.
///
/// Returns `Ok(())` if all requirements pass, or `Err(reason)` describing the first failure.
/// The reason strings are stable diagnostic messages; they appear in debug logs and in
/// Trinity's gate-failure assertions.
///
/// This is a thin wrapper around [`assess_gate`] that short-circuits on the first failure.
/// For full verbose output with measured values, call [`assess_gate`] directly.
///
/// **All parameters are plain structs (no Vulkan handles)** so this function can be
/// exhaustively unit-tested without a Vulkan ICD. See the `tests` module below.
///
/// Note: R5 (subgroup BASIC in COMPUTE) is **not** checked here — it is recorded in
/// [`Capabilities::subgroup_basic_in_compute`] per Morpheus's §7.0 principle.
pub(crate) fn passes_gate(
    props: &vk::PhysicalDeviceProperties,
    limits: &vk::PhysicalDeviceLimits,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    compute_queue_family: Option<u32>,
) -> Result<(), &'static str> {
    for criterion in assess_gate(props, limits, mem_props, compute_queue_family) {
        if !criterion.passed {
            return Err(criterion.failure_reason);
        }
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

// ──────────────────────────────────────────────────────────────────────────────
// Loader-state diagnostics
// ──────────────────────────────────────────────────────────────────────────────

/// Collect human-readable lines describing what the Vulkan loader currently sees.
///
/// Reads environment variables (`VK_ICD_FILENAMES`, `VK_DRIVER_FILES`, `VK_INSTANCE_LAYERS`),
/// queries the loader version, and enumerates available layers and extension counts. This is
/// entirely loader-level — no ICD is needed and no `VkInstance` is created.
///
/// Called in two places:
/// 1. Unconditionally on any `vkCreateInstance` failure (logged at WARN so it is always visible).
/// 2. Before creation when `ONNXRUNTIME_EP_VULKAN_VERBOSE=1` (logged at INFO).
fn loader_state_lines(entry: &ash::Entry) -> Vec<String> {
    let mut lines = Vec::new();

    for var in ["VK_ICD_FILENAMES", "VK_DRIVER_FILES", "VK_INSTANCE_LAYERS"] {
        let val = std::env::var(var).unwrap_or_else(|_| "<not set>".to_string());
        lines.push(format!("  {var} = {val}"));
    }

    // Loader version — available without any ICD.
    // SAFETY: entry is live; vkEnumerateInstanceVersion is a loader function with no side effects.
    let ver = unsafe { entry.try_enumerate_instance_version() };
    let ver_str = match ver {
        Ok(None) => {
            "1.0 (vkEnumerateInstanceVersion unavailable — loader is Vulkan 1.0)".to_string()
        }
        Ok(Some(v)) => format!(
            "{}.{}.{}",
            vk::api_version_major(v),
            vk::api_version_minor(v),
            vk::api_version_patch(v)
        ),
        Err(e) => format!("<error: {e:?}>"),
    };
    lines.push(format!("  loader version = {ver_str}"));

    // Available layers — enumerates from the loader manifest directories.
    // SAFETY: entry is live; this is a read-only enumeration.
    let layer_names: Vec<String> = unsafe { entry.enumerate_instance_layer_properties() }
        .unwrap_or_default()
        .into_iter()
        .map(|l| {
            // SAFETY: layer_name is a null-terminated C array the loader filled.
            unsafe { CStr::from_ptr(l.layer_name.as_ptr()) }
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    lines.push(format!(
        "  available layers ({}) = {}",
        layer_names.len(),
        if layer_names.is_empty() {
            "<none>".to_string()
        } else {
            layer_names.join(", ")
        }
    ));

    // Extension count — non-zero only if at least one ICD is loadable. A count of 0 or only
    // surface/swapchain loader extensions (without compute-relevant ICD extensions) indicates
    // no functional compute ICD was found.
    // SAFETY: entry is live; None pLayerName → enumerates all instance extensions.
    let ext_count = unsafe { entry.enumerate_instance_extension_properties(None) }
        .map(|v| v.len())
        .unwrap_or(0);
    lines.push(format!(
        "  instance extensions visible to loader = {ext_count} \
         (0 or small values suggest no ICD loaded)"
    ));

    lines
}

/// Run a standalone loader probe and return a formatted diagnostic report.
///
/// Used by `epctl --probe-loader`. Bypasses the shader guard in [`super::super::engine::probe_devices`]
/// so that CI can verify Vulkan availability independently of shader compilation.
///
/// This creates — and immediately drops — a `VkInstance`. It does not create a logical device
/// or allocate any GPU memory.
pub(crate) fn probe_loader_report() -> String {
    let mut out: Vec<String> = Vec::new();

    out.push("=== Vulkan Loader Probe ===".to_string());

    // Step 1: load the dynamic library.
    // SAFETY: ash::Entry::load() opens the system Vulkan loader; no invariant required beyond
    // "the system has a Vulkan loader installed."
    let entry = match unsafe { ash::Entry::load() } {
        Ok(e) => e,
        Err(e) => {
            out.push(format!("FAIL: no Vulkan loader found — {e}"));
            out.push(String::new());
            out.push(
                "Action required: install the Vulkan loader (libvulkan.so.1 on Linux, \
                      vulkan-1.dll on Windows, or MoltenVK on macOS)."
                    .to_string(),
            );
            return out.join("\n");
        }
    };

    out.push("Vulkan library loaded.".to_string());

    // Step 2: loader state (env vars, version, layers, extensions).
    out.extend(loader_state_lines(&entry));

    // Step 3: try to create a VkInstance.
    let app_name = CString::new("epctl-probe").expect("no interior NUL");
    let engine_name = CString::new("vulkan-ep-probe").expect("no interior NUL");

    // SAFETY: entry is live; vkEnumerateInstanceVersion is a loader function.
    let loader_version = unsafe { entry.try_enumerate_instance_version() }.unwrap_or(None);
    let api_v = match loader_version {
        Some(v) if v >= vk::make_api_version(0, 1, 1, 0) => vk::make_api_version(0, 1, 1, 0),
        Some(v) => {
            out.push(format!(
                "FAIL: loader version {}.{} is below the required 1.1.",
                vk::api_version_major(v),
                vk::api_version_minor(v)
            ));
            return out.join("\n");
        }
        None => {
            out.push(
                "FAIL: loader is Vulkan 1.0 (vkEnumerateInstanceVersion not present). \
                      EP requires 1.1+."
                    .to_string(),
            );
            return out.join("\n");
        }
    };

    let app_info = vk::ApplicationInfo::default()
        .application_name(&app_name)
        .engine_name(&engine_name)
        .api_version(api_v);
    let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);

    // SAFETY: entry is live; app_name/engine_name outlive create_info.
    let inst_handle = match unsafe { entry.create_instance(&create_info, None) } {
        Ok(h) => h,
        Err(e) => {
            out.push(format!("FAIL: vkCreateInstance returned {e:?}."));
            if e == vk::Result::ERROR_INCOMPATIBLE_DRIVER {
                out.push(
                    "  → ERROR_INCOMPATIBLE_DRIVER: the loader found no usable ICD. \
                          Check VK_ICD_FILENAMES / VK_DRIVER_FILES above and verify the \
                          ICD DLL/so is loadable."
                        .to_string(),
                );
            }
            return out.join("\n");
        }
    };

    out.push("vkCreateInstance: OK.".to_string());

    // Wrap in a temporary Instance so Drop calls vkDestroyInstance.
    let inst = Instance {
        _entry: entry,
        handle: inst_handle,
        debug_messenger: None,
    };

    // Step 4: enumerate all physical devices and run the §7.2 gate assessment.
    // We query physical devices directly here (rather than calling enumerate_capable_devices)
    // so we can show the full per-criterion breakdown for *every* device, including those
    // that fail the gate — that is exactly the output needed to diagnose gate rejections.
    //
    // SAFETY: inst_handle is live; we pass no invalid pointers.
    let raw_devices = unsafe { inst.handle.enumerate_physical_devices() }.unwrap_or_default();
    out.push(format!("{} physical device(s) found:", raw_devices.len()));

    let mut n_passed = 0usize;
    for (idx, &pdev) in raw_devices.iter().enumerate() {
        // SAFETY: inst.handle and pdev are live for the duration of this loop.
        let props = unsafe { inst.handle.get_physical_device_properties(pdev) };
        // SAFETY: same as above.
        let mem_props = unsafe { inst.handle.get_physical_device_memory_properties(pdev) };
        // SAFETY: same as above.
        let queue_families = unsafe {
            inst.handle
                .get_physical_device_queue_family_properties(pdev)
        };
        let compute_family = queue_families
            .iter()
            .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .map(|i| i as u32);

        let api_v = props.api_version;
        let name = device_name_str(&props);
        let criteria = assess_gate(&props, &props.limits, &mem_props, compute_family);
        let all_pass = criteria.iter().all(|c| c.passed);
        let verdict = if all_pass { "PASS" } else { "FAIL" };

        out.push(format!(
            "Device {idx}: {} [Vulkan {}.{}.{}]  — gate {}",
            name,
            vk::api_version_major(api_v),
            vk::api_version_minor(api_v),
            vk::api_version_patch(api_v),
            verdict,
        ));
        for c in &criteria {
            out.push(c.row());
        }

        // §7.9: show raw capability values so derived booleans can be audited.
        // Only probe when the device passes the gate (has a compute queue family and ≥1.1),
        // which are the preconditions caps::probe assumes.
        if all_pass {
            // SAFETY: inst.handle is live; pdev came from enumerate_physical_devices against it.
            let caps = unsafe { crate::vk::caps::probe(&inst.handle, pdev) };
            let probe_note = if caps.subgroup_probe_valid {
                String::new()
            } else {
                "  ⚠ NOT DETERMINED — probe returned all-zeros (§7.9 rule 1)".to_string()
            };
            out.push("  --- Capability probe (raw values) ---".to_string());
            out.push(format!(
                "  subgroup_size         : {}{}",
                caps.subgroup_size, probe_note
            ));
            out.push(format!(
                "  subgroup_probe_valid  : {}",
                caps.subgroup_probe_valid
            ));
            out.push(format!(
                "  subgroup_stages_raw   : {:?}",
                caps.subgroup_supported_stages
            ));
            out.push(format!(
                "  subgroup_basic_in_compute: {}{}",
                caps.subgroup_basic_in_compute,
                if !caps.subgroup_probe_valid {
                    " (NOT DETERMINED)"
                } else {
                    ""
                }
            ));
            out.push(format!(
                "  subgroup_ops          : {:?}",
                caps.subgroup_supported_ops
            ));
            out.push(format!("  is_uma                : {}", caps.is_uma));
            out.push(format!(
                "  timestamp_period_ns   : {:.4} ns/tick",
                caps.timestamp_period_ns
            ));
            out.push(format!(
                "  timestamp_valid_bits  : {}",
                caps.timestamp_valid_bits
            ));
            n_passed += 1;
        }
    }

    out.push(format!(
        "{n_passed} device(s) passed the §7.2 capability gate."
    ));
    if raw_devices.is_empty() {
        out.push(
            "  → No physical devices found (ICD installed but no device present or usable)."
                .to_string(),
        );
    } else if n_passed == 0 {
        out.push(
            "  → All devices rejected; see the FAIL criterion above for the specific reason."
                .to_string(),
        );
    } else {
        // Show which device the EP would select and what env var controls it.
        // We need the full CapableDevice list for select_device, but probe_loader_report uses
        // a lightweight gate-only path (no caps::probe). Reproduce the sort score from DeviceKind.
        let selector_val = std::env::var(ENV_DEVICE_SELECTOR).unwrap_or_default();
        out.push(String::new());
        out.push(format!(
            "{ENV_DEVICE_SELECTOR} = {}",
            if selector_val.is_empty() {
                "<not set — best-first (discrete preferred)>".to_string()
            } else {
                selector_val.clone()
            }
        ));
        // Replicate best-first sort to find the default (index 0 after sort by DeviceKind::score).
        // We don't have CapableDevice here so we sort raw_devices by device type score.
        let score = |ty: vk::PhysicalDeviceType| match ty {
            vk::PhysicalDeviceType::DISCRETE_GPU => 3u8,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 2,
            vk::PhysicalDeviceType::VIRTUAL_GPU => 1,
            _ => 0,
        };
        let mut passing: Vec<(usize, String, vk::PhysicalDeviceType)> = raw_devices
            .iter()
            .enumerate()
            .filter_map(|(idx, &pdev)| {
                // SAFETY: inst.handle and pdev are live; pdev came from enumerate_physical_devices
                // against this instance handle in the same scope.
                let p = unsafe { inst.handle.get_physical_device_properties(pdev) };
                // SAFETY: same as above.
                let m = unsafe { inst.handle.get_physical_device_memory_properties(pdev) };
                // SAFETY: same as above.
                let qf = unsafe {
                    inst.handle
                        .get_physical_device_queue_family_properties(pdev)
                }
                .into_iter()
                .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .map(|i| i as u32);
                if passes_gate(&p, &p.limits, &m, qf).is_ok() {
                    Some((idx, device_name_str(&p), p.device_type))
                } else {
                    None
                }
            })
            .collect();
        passing.sort_by_key(|(_, _, ty)| std::cmp::Reverse(score(*ty)));

        if !passing.is_empty() {
            // Apply the selector logic (mirrors select_device but on the probe-only data).
            let selected_name = if selector_val.is_empty() {
                format!(
                    "Device {} '{}' (best-first default; set {ENV_DEVICE_SELECTOR}=<index|name> to override)",
                    passing[0].0, passing[0].1
                )
            } else if let Ok(idx) = selector_val.parse::<usize>() {
                passing
                    .get(idx)
                    .map(|(i, n, _)| format!("Device {i} '{n}' (selected by index {idx})"))
                    .unwrap_or_else(|| {
                        format!(
                            "index {idx} out of range — would fall back to Device {} '{}'",
                            passing[0].0, passing[0].1
                        )
                    })
            } else {
                let lower = selector_val.to_lowercase();
                passing
                    .iter()
                    .find(|(_, n, _)| n.to_lowercase().contains(&lower))
                    .map(|(i, n, _)| format!("Device {i} '{n}' (matched '{selector_val}')"))
                    .unwrap_or_else(|| {
                        format!(
                            "no name matches '{selector_val}' — would fall back to Device {} '{}'",
                            passing[0].0, passing[0].1
                        )
                    })
            };
            out.push(format!("Would select: {selected_name}"));
        }
    }

    out.join("\n")
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal helpers (non-diagnostic)
// ──────────────────────────────────────────────────────────────────────────────

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

    #[test]
    fn good_device_passes_gate() {
        let props = good_props();
        assert!(
            passes_gate(&props, good_limits(&props), &good_mem_props(), Some(0)).is_ok(),
            "a fully capable device must pass all requirements"
        );
    }

    #[test]
    fn r1_rejects_vulkan_1_0() {
        let mut props = good_props();
        props.api_version = vk::make_api_version(0, 1, 0, 0);
        let r = passes_gate(&props, good_limits(&props), &good_mem_props(), Some(0));
        assert_eq!(r, Err("R1: Vulkan API version < 1.1"));
    }

    #[test]
    fn r2_rejects_no_compute_queue() {
        let props = good_props();
        let r = passes_gate(&props, good_limits(&props), &good_mem_props(), None);
        assert_eq!(r, Err("R2: no compute queue family"));
    }

    #[test]
    fn r3_rejects_low_invocation_count() {
        let mut props = good_props();
        props.limits.max_compute_work_group_invocations = 128;
        let r = passes_gate(&props, good_limits(&props), &good_mem_props(), Some(0));
        assert_eq!(r, Err("R3: maxComputeWorkGroupInvocations < 256"));
    }

    #[test]
    fn r3_accepts_exactly_256_invocations() {
        let mut props = good_props();
        props.limits.max_compute_work_group_invocations = 256;
        assert!(passes_gate(&props, good_limits(&props), &good_mem_props(), Some(0)).is_ok());
    }

    #[test]
    fn r4_rejects_small_shared_memory() {
        let mut props = good_props();
        props.limits.max_compute_shared_memory_size = 8192;
        let r = passes_gate(&props, good_limits(&props), &good_mem_props(), Some(0));
        assert_eq!(r, Err("R4: maxComputeSharedMemorySize < 16384"));
    }

    #[test]
    fn r4_accepts_exactly_16384_shared_memory() {
        let mut props = good_props();
        props.limits.max_compute_shared_memory_size = 16384;
        assert!(passes_gate(&props, good_limits(&props), &good_mem_props(), Some(0)).is_ok());
    }

    // R5 is no longer in the gate — it was demoted to Capabilities::subgroup_basic_in_compute
    // per Morpheus's §7.0 principle. The *policy* (capability degrades op coverage, not device
    // availability) remains correct.
    //
    // IMPORTANT — D-S14-01: The original premise for the demotion was lavapipe reporting
    // `supportedStages = 0`. That reading was almost certainly the push_next probe bug (§7.9
    // Bug 1 / D-S12-01): with a zeroed pNext chain, every chained capability reads zero.
    // Mesa 26.1 lavapipe (llvmpipe) does support subgroup BASIC in compute — the zero was the
    // probe, not the device. The policy is still correct for other reasons:
    //  • Future unknown devices may genuinely lack subgroup support in compute.
    //  • §7.0 keeps the device roster stable when non-critical capabilities are absent.
    // See the `lavapipe_profile_passes_gate` test and the CI probe output for confirmation.

    #[test]
    fn device_without_subgroup_compute_passes_gate() {
        // This test pins the R5-removal fix. If R5 is re-added to passes_gate, this fails.
        // The good_props() fixture has no subgroup properties set — zero-valued fields
        // correspond to a device with no subgroup support in any stage.
        let props = good_props();
        assert!(
            passes_gate(&props, good_limits(&props), &good_mem_props(), Some(0)).is_ok(),
            "subgroup support is a capability flag, not a gate criterion (§7.0)"
        );
    }

    #[test]
    fn r6a_rejects_no_device_local_heap() {
        let mut m = good_mem_props();
        // Remove DEVICE_LOCAL flag from all heaps.
        for i in 0..m.memory_heap_count as usize {
            m.memory_heaps[i].flags = vk::MemoryHeapFlags::empty();
        }
        let props = good_props();
        let r = passes_gate(&props, good_limits(&props), &m, Some(0));
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
        let r = passes_gate(&props, good_limits(&props), &m, Some(0));
        assert_eq!(r, Err("R6b: no HOST_VISIBLE memory type"));
    }

    #[test]
    fn gate_order_r1_before_r2() {
        // R1 is checked before R2: version failure is reported first.
        let mut props = good_props();
        props.api_version = vk::make_api_version(0, 1, 0, 0);
        let r = passes_gate(&props, good_limits(&props), &good_mem_props(), None);
        assert_eq!(r, Err("R1: Vulkan API version < 1.1"));
    }

    // ── Realistic device-profile tests ────────────────────────────────────────
    // Modelled on real hardware to pin that the gate makes correct decisions for the device
    // classes the EP will actually run on.

    /// Memory layout typical of Mesa lavapipe (llvmpipe): a single heap flagged
    /// DEVICE_LOCAL containing HOST_VISIBLE+HOST_COHERENT types (all CPU RAM).
    #[allow(clippy::field_reassign_with_default)]
    fn lavapipe_mem_props() -> vk::PhysicalDeviceMemoryProperties {
        let mut m = vk::PhysicalDeviceMemoryProperties::default();
        m.memory_heap_count = 1;
        m.memory_heaps[0].flags = vk::MemoryHeapFlags::DEVICE_LOCAL;
        m.memory_heaps[0].size = 4 * 1024 * 1024 * 1024; // Mesa default: 4 GiB
        m.memory_type_count = 2;
        m.memory_types[0].heap_index = 0;
        m.memory_types[0].property_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL
            | vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_COHERENT;
        m.memory_types[1].heap_index = 0;
        m.memory_types[1].property_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL
            | vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_COHERENT
            | vk::MemoryPropertyFlags::HOST_CACHED;
        m
    }

    #[test]
    fn lavapipe_profile_passes_gate() {
        // Synthesised from vulkaninfo on Mesa lavapipe (llvmpipe).
        // Mesa 26.1.3 (Windows CI): deviceName = llvmpipe (LLVM 22.1.8, 256 bits),
        //   apiVersion = 1.4.348, deviceType = CPU.
        // The old Mesa 22.x reading of `supportedStages = 0` was a probe bug (§7.9 Bug 1 /
        // D-S12-01 / D-S14-01) — Mesa 26.1 lavapipe DOES support subgroup BASIC in compute.
        // This test validates the *gate policy* (R5 removed per §7.0), not the device profile.
        // The gate does not query subgroup properties → profile values here are not exercised.
        let mut props = good_props();
        props.api_version = vk::make_api_version(0, 1, 4, 348);
        props.limits.max_compute_work_group_invocations = 1024;
        props.limits.max_compute_shared_memory_size = 32768; // 32 KiB typical for Mesa

        let mem = lavapipe_mem_props();

        assert!(
            passes_gate(&props, &props.limits, &mem, Some(0)).is_ok(),
            "lavapipe/llvmpipe must pass the §7.2 gate (R5 removed per §7.0)"
        );

        // Full assessment must agree — no criterion should fail.
        let criteria = assess_gate(&props, &props.limits, &mem, Some(0));
        let failed: Vec<&str> = criteria
            .iter()
            .filter(|c| !c.passed)
            .map(|c| c.label)
            .collect();
        assert!(
            failed.is_empty(),
            "all criteria should pass for lavapipe profile; failed: {failed:?}"
        );
    }

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn uma_integrated_gpu_passes_gate() {
        // UMA device (Intel integrated / Apple M-series / Adreno): single heap that is
        // both DEVICE_LOCAL and HOST_VISIBLE. R6 must accept combined heaps.
        let mut props = good_props();
        props.api_version = vk::make_api_version(0, 1, 3, 0);
        props.limits.max_compute_work_group_invocations = 512;
        props.limits.max_compute_shared_memory_size = 65536; // 64 KiB

        let mut mem = vk::PhysicalDeviceMemoryProperties::default();
        mem.memory_heap_count = 1;
        mem.memory_heaps[0].flags = vk::MemoryHeapFlags::DEVICE_LOCAL;
        mem.memory_heaps[0].size = 8 * 1024 * 1024 * 1024;
        mem.memory_type_count = 2;
        mem.memory_types[0].heap_index = 0;
        mem.memory_types[0].property_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL
            | vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_COHERENT;
        mem.memory_types[1].heap_index = 0;
        mem.memory_types[1].property_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL
            | vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_CACHED;

        assert!(
            passes_gate(&props, &props.limits, &mem, Some(0)).is_ok(),
            "UMA device with combined DEVICE_LOCAL|HOST_VISIBLE heap must pass R6"
        );
    }

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn discrete_gpu_passes_gate() {
        // Standard discrete GPU: two heaps (VRAM + system RAM), three memory types
        // including a resizable-BAR type (DEVICE_LOCAL|HOST_VISIBLE on the VRAM heap).
        let mut props = good_props();
        props.api_version = vk::make_api_version(0, 1, 3, 0);
        props.limits.max_compute_work_group_invocations = 1024;
        props.limits.max_compute_shared_memory_size = 49152; // 48 KiB

        let mut mem = vk::PhysicalDeviceMemoryProperties::default();
        mem.memory_heap_count = 2;
        mem.memory_heaps[0].flags = vk::MemoryHeapFlags::DEVICE_LOCAL;
        mem.memory_heaps[0].size = 8 * 1024 * 1024 * 1024; // 8 GiB VRAM
        mem.memory_heaps[1].size = 32 * 1024 * 1024 * 1024; // 32 GiB system RAM
        mem.memory_type_count = 3;
        mem.memory_types[0].heap_index = 0;
        mem.memory_types[0].property_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        mem.memory_types[1].heap_index = 1;
        mem.memory_types[1].property_flags =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        mem.memory_types[2].heap_index = 0;
        // Resizable BAR: VRAM directly mapped to the host address space.
        mem.memory_types[2].property_flags =
            vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE;

        assert!(
            passes_gate(&props, &props.limits, &mem, Some(0)).is_ok(),
            "standard discrete GPU profile must pass the §7.2 gate"
        );
    }

    #[test]
    fn assess_gate_reports_measured_values_and_identifies_failure() {
        // Verify that assess_gate returns measured values (not just verdicts) and pinpoints
        // the exact failing criterion without hiding others.
        let mut props = good_props();
        props.limits.max_compute_work_group_invocations = 64; // below R3 minimum

        let criteria = assess_gate(&props, &props.limits, &good_mem_props(), Some(0));

        let fail_labels: Vec<&str> = criteria
            .iter()
            .filter(|c| !c.passed)
            .map(|c| c.label)
            .collect();

        assert!(
            fail_labels == ["R3  maxComputeWorkGroupInvocations"],
            "only R3 should fail; got failures: {fail_labels:?}"
        );

        // Measured value must contain the actual number.
        let r3 = criteria.iter().find(|c| c.label.starts_with("R3")).unwrap();
        assert!(
            r3.measured.contains("64"),
            "R3 measured value should contain '64'; got: {:?}",
            r3.measured
        );
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
