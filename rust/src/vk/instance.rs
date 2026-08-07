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
use crate::engine::{DeviceIdentity, DeviceInfo, DeviceKind};

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
    let names: Vec<&str> = devices.iter().map(|d| d.info.name.as_str()).collect();
    select_by_selector(&names)
}

/// The selector's one implementation, over nothing but the best-first-ordered device names.
///
/// Both the compute session (`select_device`, over `CapableDevice`) and the factory's advertise
/// path (over `DeviceInfo`) resolve `ONNXRUNTIME_EP_VULKAN_DEVICE` through this function, so the
/// two cannot drift into disagreeing about which device the selector names. `names` must be in
/// best-first order — the same order `enumerate_capable_devices` returns — because the selector
/// index is defined against that order and against no other.
pub(crate) fn select_by_selector(names: &[&str]) -> Option<usize> {
    if names.is_empty() {
        return None;
    }
    let selector = std::env::var(ENV_DEVICE_SELECTOR).unwrap_or_default();
    if selector.is_empty() {
        return Some(0);
    }
    // Integer index?
    if let Ok(idx) = selector.parse::<usize>() {
        if idx < names.len() {
            return Some(idx);
        }
        log::warn!(
            "{ENV_DEVICE_SELECTOR}={selector}: index out of range \
             ({} device(s) passed the gate). Using device 0.",
            names.len(),
        );
        return Some(0);
    }
    // Name substring (case-insensitive).
    let lower = selector.to_lowercase();
    match names.iter().position(|n| n.to_lowercase().contains(&lower)) {
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

/// Whether `ONNXRUNTIME_EP_VULKAN_DEVICE` is set to anything at all.
///
/// A set selector is a **pin**, and the factory treats it as one: it advertises only the pinned
/// device, so ORT cannot bind a device other than the one the compute session will open. See the
/// index-space note in `vk/device.rs`.
pub(crate) fn selector_is_pinned() -> bool {
    !std::env::var(ENV_DEVICE_SELECTOR)
        .unwrap_or_default()
        .is_empty()
}

// ──────────────────────────────────────────────────────────────────────────────
// Stable-identity device selection (issue #18)
//
// `ENV_DEVICE_SELECTOR` above is deliberately left as-is: it is a lenient, best-effort pin
// (index or name substring, silently falls back to device 0 with a warning on a bad value) and
// changing that behaviour out from under every existing caller — `engine.rs`, `device.rs`,
// `host_device_memory.rs` all read it today — is not this fix. What issue #18 asks for is a
// selector that (a) can name a device by identity that survives enumeration-order churn (UUID,
// vendor+device, PCI location — index is explicitly demoted to "a displayed ordinal", per the
// issue text) and (b) refuses to guess: an ambiguous or unresolvable request is an error, never
// a silent fallback to some other GPU. Those two properties don't fit the legacy selector's
// contract, so this is an additive, higher-precedence mechanism instead of a behaviour change to
// it. See `ENV_DEVICE_SELECTOR_STRICT` for the env var and `select_device_strict` for the entry
// point `engine.rs` and `device.rs` both call ahead of the legacy selector.
// ──────────────────────────────────────────────────────────────────────────────

/// Environment variable for a strict, stable-identity Vulkan device selector (issue #18).
///
/// Takes precedence over [`ENV_DEVICE_SELECTOR`] and over `ep.device_index` when set. Unlike the
/// legacy selector, an unresolvable value here is a hard error: **no Vulkan device is opened or
/// advertised** rather than falling back to a different GPU. Unset (the default) defers entirely
/// to the legacy selector / `ep.device_index` / ORT's own binding — this preserves the portable
/// default behaviour of running on the best-scoring device with no configuration at all.
///
/// Accepted forms, all `<scheme>:<value>` (see [`parse_device_selector`]):
/// - `index:<N>` — position in the best-first sorted capable-device list. **A displayed ordinal,
///   not a stable identity** — it can move when a driver update or reboot changes enumeration
///   order. Prefer one of the identity forms below for anything that must survive that.
/// - `name:<exact device name>` — exact (not substring) match against `deviceName`. Ambiguous
///   when two installed GPUs share a model name.
/// - `id:<vendorHex>:<deviceHex>` — exact `(vendorID, deviceID)` match, e.g. `id:10de:2900` for
///   an RTX 4060 Laptop GPU. Stable across reboots and driver updates for one GPU *model*, but
///   still ambiguous if two identical cards are installed.
/// - `uuid:<32 hex chars>` — exact match against `VkPhysicalDeviceIDProperties::deviceUUID`.
///   Names one physical device instance unambiguously; always available (Vulkan 1.1 core).
/// - `luid:<16 hex chars>` — exact match against `deviceLUID`, only when the driver set
///   `deviceLUIDValid`. Primarily a Windows/D3D-interop identity; returns
///   [`DeviceSelectionError::UnsupportedIdentity`] where no enumerated device reports one.
/// - `pci:<domain>:<bus>:<device>.<function>` (e.g. `pci:0000:01:00.0`) — exact PCI location
///   match, only when `VK_EXT_pci_bus_info` is supported. `UnsupportedIdentity` on MoltenVK and
///   other platforms without a PCI bus to report.
pub const ENV_DEVICE_SELECTOR_STRICT: &str = "ONNXRUNTIME_EP_VULKAN_DEVICE_SELECTOR";

/// One parsed, typed form of [`ENV_DEVICE_SELECTOR_STRICT`] (or the equivalent `ep.device_selector`
/// session option).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum DeviceSelector {
    /// `index:<N>` — a displayed ordinal into the best-first sorted capable-device list.
    Index(usize),
    /// `name:<exact device name>`.
    Name(String),
    /// `id:<vendorHex>:<deviceHex>` — exact `(vendor_id, device_id)`.
    VendorDevice(u32, u32),
    /// `uuid:<32 lowercase hex chars>`, normalized.
    Uuid(String),
    /// `luid:<16 lowercase hex chars>`, normalized.
    Luid(String),
    /// `pci:<domain:bus:device.function>`, normalized lowercase.
    Pci(String),
}

impl std::fmt::Display for DeviceSelector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeviceSelector::Index(i) => write!(f, "index:{i}"),
            DeviceSelector::Name(n) => write!(f, "name:{n}"),
            DeviceSelector::VendorDevice(v, d) => write!(f, "id:{v:04x}:{d:04x}"),
            DeviceSelector::Uuid(u) => write!(f, "uuid:{u}"),
            DeviceSelector::Luid(l) => write!(f, "luid:{l}"),
            DeviceSelector::Pci(p) => write!(f, "pci:{p}"),
        }
    }
}

/// Why [`resolve_device_selector`] could not name exactly one device.
///
/// Every variant carries enough context to be logged directly — the issue #18 requirement is
/// that the failure is loud and diagnostic, never a quiet fallback.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum DeviceSelectionError {
    /// The selector was well-formed but named no enumerated device.
    NotFound {
        selector: String,
        /// Human-readable summaries of the devices that *were* available, for the log line.
        available: Vec<String>,
    },
    /// The selector named more than one enumerated device.
    Ambiguous {
        selector: String,
        matches: Vec<String>,
    },
    /// The selector's identity kind is not reported by any enumerated device on this platform
    /// (e.g. `pci:` on MoltenVK, `luid:` on most Linux drivers). Distinguished from `NotFound`
    /// because the fix is different: a `NotFound` selector might match after plugging in the
    /// right GPU; an `UnsupportedIdentity` selector cannot match on this platform at all.
    UnsupportedIdentity { selector: String, reason: String },
    /// The raw string could not be parsed as any known selector scheme.
    Malformed { raw: String, reason: String },
}

impl std::fmt::Display for DeviceSelectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeviceSelectionError::NotFound {
                selector,
                available,
            } => {
                if available.is_empty() {
                    write!(f, "no device matches {selector} (no devices enumerated)")
                } else {
                    write!(
                        f,
                        "no device matches {selector} — available: [{}]",
                        available.join(", ")
                    )
                }
            }
            DeviceSelectionError::Ambiguous { selector, matches } => write!(
                f,
                "{selector} matches {} devices, not exactly one — matches: [{}]",
                matches.len(),
                matches.join(", ")
            ),
            DeviceSelectionError::UnsupportedIdentity { selector, reason } => {
                write!(
                    f,
                    "{selector} cannot be resolved on this platform: {reason}"
                )
            }
            DeviceSelectionError::Malformed { raw, reason } => {
                write!(
                    f,
                    "{ENV_DEVICE_SELECTOR_STRICT}={raw:?} is not a valid selector: {reason}"
                )
            }
        }
    }
}

/// Normalize a hex string of exactly `expected_len` hex digits, tolerating `-`/`:` separators
/// (so `uuid:` accepts both `a1b2...` and the canonical `xxxxxxxx-xxxx-...` UUID form).
fn normalize_hex(raw: &str, expected_len: usize, field: &str) -> Result<String, String> {
    let cleaned: String = raw.chars().filter(|c| *c != '-' && *c != ':').collect();
    if cleaned.len() != expected_len || !cleaned.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(format!(
            "{field} must be exactly {expected_len} hex characters (separators '-'/':' are \
             tolerated and stripped), got {raw:?}"
        ));
    }
    Ok(cleaned.to_lowercase())
}

/// Parse one `<scheme>:<value>` selector string into a [`DeviceSelector`].
///
/// Pure and total: every input either parses or returns a descriptive `Err`. No environment or
/// Vulkan access, so this is fully unit-testable.
pub(crate) fn parse_device_selector(raw: &str) -> Result<DeviceSelector, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("selector is empty".to_string());
    }
    let (scheme, rest) = trimmed.split_once(':').ok_or_else(|| {
        format!(
            "{trimmed:?} has no 'scheme:value' — expected one of index:, name:, id:, uuid:, \
             luid:, pci:"
        )
    })?;
    match scheme {
        "index" => rest
            .parse::<usize>()
            .map(DeviceSelector::Index)
            .map_err(|e| format!("index:{rest:?} is not a non-negative integer ({e})")),
        "name" => {
            if rest.is_empty() {
                Err("name: selector has an empty device name".to_string())
            } else {
                Ok(DeviceSelector::Name(rest.to_string()))
            }
        }
        "id" => {
            let (v, d) = rest.split_once(':').ok_or_else(|| {
                format!("id:{rest:?} must be 'id:<vendorHex>:<deviceHex>', e.g. id:10de:2900")
            })?;
            let vendor = u32::from_str_radix(v.trim_start_matches("0x"), 16)
                .map_err(|e| format!("id: vendor {v:?} is not hex ({e})"))?;
            let device = u32::from_str_radix(d.trim_start_matches("0x"), 16)
                .map_err(|e| format!("id: device {d:?} is not hex ({e})"))?;
            Ok(DeviceSelector::VendorDevice(vendor, device))
        }
        "uuid" => normalize_hex(rest, 32, "uuid").map(DeviceSelector::Uuid),
        "luid" => normalize_hex(rest, 16, "luid").map(DeviceSelector::Luid),
        "pci" => {
            if rest.is_empty() {
                Err("pci: selector has an empty location".to_string())
            } else {
                Ok(DeviceSelector::Pci(rest.to_lowercase()))
            }
        }
        other => Err(format!(
            "unknown selector scheme {other:?} in {trimmed:?} — expected one of index:, name:, \
             id:, uuid:, luid:, pci:"
        )),
    }
}

/// Resolve a parsed [`DeviceSelector`] against a best-first sorted device list.
///
/// Returns `Ok(i)` only when the selector names **exactly** one device. Never falls back:
/// zero matches is [`DeviceSelectionError::NotFound`], more than one is
/// [`DeviceSelectionError::Ambiguous`], and an identity kind absent from every device on this
/// platform is [`DeviceSelectionError::UnsupportedIdentity`] rather than being folded into
/// `NotFound` (the two have different fixes — see that variant's doc comment).
///
/// Pure over [`DeviceInfo`] (no Vulkan handles), so it is fully unit-testable, including
/// simulated multi-device and ambiguous configurations that don't require a live ICD.
pub(crate) fn resolve_device_selector(
    devices: &[DeviceInfo],
    selector: &DeviceSelector,
) -> Result<usize, DeviceSelectionError> {
    let describe = |i: usize| {
        format!(
            "{} (index {}, {})",
            devices[i].name,
            i,
            devices[i].key().canonical()
        )
    };
    let available = || {
        devices
            .iter()
            .enumerate()
            .map(|(i, _)| describe(i))
            .collect::<Vec<_>>()
    };

    let pick = |matches: Vec<usize>| -> Result<usize, DeviceSelectionError> {
        match matches.len() {
            0 => Err(DeviceSelectionError::NotFound {
                selector: selector.to_string(),
                available: available(),
            }),
            1 => Ok(matches[0]),
            _ => Err(DeviceSelectionError::Ambiguous {
                selector: selector.to_string(),
                matches: matches.into_iter().map(describe).collect(),
            }),
        }
    };

    match selector {
        DeviceSelector::Index(i) => {
            if *i < devices.len() {
                Ok(*i)
            } else {
                Err(DeviceSelectionError::NotFound {
                    selector: selector.to_string(),
                    available: available(),
                })
            }
        }
        DeviceSelector::Name(n) => pick(
            devices
                .iter()
                .enumerate()
                .filter(|(_, d)| &d.name == n)
                .map(|(i, _)| i)
                .collect(),
        ),
        DeviceSelector::VendorDevice(v, d) => pick(
            devices
                .iter()
                .enumerate()
                .filter(|(_, dev)| dev.vendor_id == *v && dev.device_id == *d)
                .map(|(i, _)| i)
                .collect(),
        ),
        DeviceSelector::Uuid(u) => {
            // A driver that reports no UUID cannot be selected by one, and saying so is different
            // from saying "no device has that UUID" — the same distinction `luid:`/`pci:` already
            // make. `UnsupportedIdentity` is the fail-closed answer: it can never be resolved on
            // this platform, so no amount of retrying or re-plugging will make it match.
            if !devices.is_empty() && devices.iter().all(|d| d.identity.uuid.is_none()) {
                return Err(DeviceSelectionError::UnsupportedIdentity {
                    selector: selector.to_string(),
                    reason: "no enumerated device reports a Vulkan device UUID \
                             (VkPhysicalDeviceIDProperties::deviceUUID was all zeros on all of \
                             them, which is an unpopulated struct rather than an identity)"
                        .to_string(),
                });
            }
            pick(
                devices
                    .iter()
                    .enumerate()
                    .filter(|(_, dev)| dev.identity.uuid.as_deref() == Some(u.as_str()))
                    .map(|(i, _)| i)
                    .collect(),
            )
        }
        DeviceSelector::Luid(l) => {
            if !devices.is_empty() && devices.iter().all(|d| d.identity.luid.is_none()) {
                return Err(DeviceSelectionError::UnsupportedIdentity {
                    selector: selector.to_string(),
                    reason: "no enumerated device reports a Vulkan LUID (deviceLUIDValid was \
                             false on all of them; LUIDs are primarily a Windows/D3D-interop \
                             identity)"
                        .to_string(),
                });
            }
            pick(
                devices
                    .iter()
                    .enumerate()
                    .filter(|(_, d)| d.identity.luid.as_deref() == Some(l.as_str()))
                    .map(|(i, _)| i)
                    .collect(),
            )
        }
        DeviceSelector::Pci(p) => {
            if !devices.is_empty() && devices.iter().all(|d| d.identity.pci.is_none()) {
                return Err(DeviceSelectionError::UnsupportedIdentity {
                    selector: selector.to_string(),
                    reason: "no enumerated device reports VK_EXT_pci_bus_info (unsupported on \
                             this driver/platform, e.g. MoltenVK)"
                        .to_string(),
                });
            }
            pick(
                devices
                    .iter()
                    .enumerate()
                    .filter(|(_, d)| d.identity.pci.as_deref() == Some(p.as_str()))
                    .map(|(i, _)| i)
                    .collect(),
            )
        }
    }
}

/// Whether [`ENV_DEVICE_SELECTOR_STRICT`] is set to anything at all.
fn strict_selector_requested() -> Option<String> {
    let v = std::env::var(ENV_DEVICE_SELECTOR_STRICT).unwrap_or_default();
    (!v.is_empty()).then_some(v)
}

/// Resolve the strict, stable-identity selector against `devices`.
///
/// `session_override` is the `ep.device_selector` session option, if the caller has one (only
/// `device.rs`'s per-session path does; the factory's advertise path in `engine.rs` runs before
/// any session exists and always passes `None`). When present it takes precedence over
/// [`ENV_DEVICE_SELECTOR_STRICT`], mirroring how `ep.device_index` already outranks nothing from
/// the environment — a session option is the more specific request.
///
/// - `Ok(None)` — no selector set (neither the option nor the env var). Callers fall back to
///   `ep.device_index` / the legacy [`ENV_DEVICE_SELECTOR`] / ORT's own binding, unchanged from
///   before issue #18.
/// - `Ok(Some(i))` — resolved unambiguously to device `i`. Callers must treat this as an explicit
///   request (same as `ep.device_index` today) and must not let anything else override it.
/// - `Err(e)` — a selector is set but could not be resolved. Callers **must not** fall back to
///   any other device; the only correct responses are "open nothing" / "advertise nothing" plus
///   a loud log line, so a session never runs on a GPU other than the one asked for.
pub(crate) fn select_device_strict(
    devices: &[DeviceInfo],
    session_override: Option<&str>,
) -> Result<Option<usize>, DeviceSelectionError> {
    let raw = match session_override {
        Some(v) if !v.is_empty() => Some(v.to_string()),
        _ => strict_selector_requested(),
    };
    let Some(raw) = raw else {
        return Ok(None);
    };
    let selector =
        parse_device_selector(&raw).map_err(|reason| DeviceSelectionError::Malformed {
            raw: raw.clone(),
            reason,
        })?;
    resolve_device_selector(devices, &selector).map(Some)
}

/// Translate a *physical* `vkEnumeratePhysicalDevices` index into a position in a best-first
/// ordered capable-device list.
///
/// **This is the only place the two index spaces are allowed to meet.** They are not
/// interchangeable and their divergence has now been a defect three times on this project:
///
/// * `epctl --probe-loader` printed the enumeration index while the selector indexed the sorted
///   list, so every device label the team used was inverted for a day;
/// * the §6.5 offer was keyed by the enumeration index while the session was chosen by the
///   selector index, so on any desk where the two orders differ the provider silently stood up a
///   second `VkDevice` (`alloc_device_frame = SPLIT-DEVICE`).
///
/// Returns `None` when no capable device carries that physical index — which is a real answer
/// (ORT bound a device that failed our §7.2 gate is impossible, but a device disappearing between
/// enumerations is not), never a reason to fall back silently.
pub(crate) fn position_of_physical(devices: &[CapableDevice], physical: usize) -> Option<usize> {
    position_of_physical_in(devices.iter().map(|d| d.info.index), physical)
}

/// [`position_of_physical`] over nothing but the physical indices, so the translation can be
/// tested without a live Vulkan device.
pub(crate) fn position_of_physical_in(
    physical_indices: impl Iterator<Item = usize>,
    physical: usize,
) -> Option<usize> {
    let mut it = physical_indices;
    it.position(|i| i == physical)
}

// ──────────────────────────────────────────────────────────────────────────────
// THE STRICT-SELECTOR DECISION (issue #18, contract C6)
//
// Extracted from `device::acquire_ep_device`, where it was four lines of inline `match` wrapped
// in a hundred lines of Vulkan setup. That placement made it untestable without a live ICD and a
// second physical GPU, so the only fail-closed predicate in the device path was the only one with
// no unit test — and the *behaviour* it guards (refuse rather than warn) is the whole of C6.
//
// Pure over integers and a `Result`, so every arm is reachable from a unit test, and a mutation
// that disables the refusal is caught by one.
// ──────────────────────────────────────────────────────────────────────────────

/// What a caller must do about the strict, stable-identity selector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum StrictSelectorDecision {
    /// No strict selector is set. The rest of the precedence chain (`ep.device_index`, the legacy
    /// env selector, ORT's binding, best score) decides, exactly as it did before issue #18.
    NotRequested,
    /// The selector resolved to this best-first index and nothing contradicts it: either ORT
    /// bound the same device, or ORT bound nothing at all. Open it.
    Open(usize),
    /// Fail closed — open nothing, claim nothing, let ORT fall back to the CPU EP.
    Refuse(StrictRefusal),
}

/// Why [`strict_selector_decision`] refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum StrictRefusal {
    /// A selector is set but names no single device: malformed, matching nothing, matching more
    /// than one, or naming an identity kind this platform does not report.
    ///
    /// All four are one refusal because they call for one response. "Fall back to device 0
    /// because the selector was ambiguous" is the same failure as "fall back to device 0 because
    /// the selector was misspelled": a run on hardware nobody asked for.
    Unresolvable(DeviceSelectionError),
    /// The selector resolved, and ORT bound a *different* device.
    ///
    /// This is the divergence a selector set **after** `RegisterExecutionProviderLibrary`, or an
    /// `ep.device_selector` session option disagreeing with what the factory advertised, both
    /// produce. `engine::devices_to_advertise` removes it at the source when the env var is set
    /// early — with one device advertised, the two index spaces have one member each.
    Diverged {
        /// The best-first index the selector resolved to.
        strict_idx: usize,
        /// The `vkEnumeratePhysicalDevices` index ORT bound.
        bound_physical: usize,
        /// That binding translated into the best-first space, or `None` when no §7.2-capable
        /// device carries it.
        bound_pos: Option<usize>,
    },
}

/// Decide what the strict selector requires, given how it resolved and what ORT bound.
///
/// `strict` is [`select_device_strict`]'s result; `bound_physical` is the `OrtEpDevice` ORT bound
/// for this session in `vkEnumeratePhysicalDevices` space; `bound_pos` is that same binding
/// translated into best-first space by [`position_of_physical`], or `None` when no capable device
/// carries it.
///
/// # The four fail-closed states, and the one that passes
///
/// | `strict` | `bound_physical` | `bound_pos` | decision |
/// |---|---|---|---|
/// | `Ok(None)` | any | any | [`StrictSelectorDecision::NotRequested`] |
/// | `Ok(Some(i))` | `None` | any | `Open(i)` — ORT bound nothing, so nothing can diverge |
/// | `Ok(Some(i))` | `Some(_)` | `Some(i)` | `Open(i)` — **exact agreement, the only passing case** |
/// | `Ok(Some(i))` | `Some(p)` | `Some(j≠i)` | `Refuse(Diverged)` |
/// | `Ok(Some(i))` | `Some(p)` | `None` | `Refuse(Diverged)` — ORT bound a device outside the §7.2-capable list, so agreement is not merely absent, it is unknowable |
/// | `Err(e)` | any | any | `Refuse(Unresolvable)` |
///
/// The last `Ok` row is the one an "obvious" implementation gets wrong: an untranslatable binding
/// is not "no binding". A caller that treated it as `false` (no divergence) would open the
/// selector's device while ORT's allocator stayed keyed to a device we could not even name, which
/// is the `SPLIT-DEVICE` condition with the diagnostic removed.
///
/// # Why this refuses where every other selector in the path merely warns
///
/// `ep.device_index` and `ONNXRUNTIME_EP_VULKAN_DEVICE` are *preferences*, and a preference that
/// loses an argument may be reported and continued past. A stable-identity selector is not a
/// preference: the only reason to name hardware by UUID/LUID/PCI is that a number obtained on
/// different hardware would be wrong to attribute. Warning and continuing is fail-**open** in the
/// way that matters — the session runs, the frame reports `SPLIT-DEVICE`, and a reader who does
/// not chase that token reads a `MATCH` as evidence about the device they asked for. The evidence
/// is real; the attribution is not. Refusing produces no evidence at all, which is honest.
pub(crate) fn strict_selector_decision(
    strict: Result<Option<usize>, DeviceSelectionError>,
    bound_physical: Option<usize>,
    bound_pos: Option<usize>,
) -> StrictSelectorDecision {
    let strict_idx = match strict {
        Err(e) => return StrictSelectorDecision::Refuse(StrictRefusal::Unresolvable(e)),
        Ok(None) => return StrictSelectorDecision::NotRequested,
        Ok(Some(i)) => i,
    };
    match (bound_physical, bound_pos) {
        // ORT bound nothing for this session: there is no second device to diverge from.
        (None, _) => StrictSelectorDecision::Open(strict_idx),
        // The one passing case: ORT bound exactly the device the identity names.
        (Some(_), Some(pos)) if pos == strict_idx => StrictSelectorDecision::Open(strict_idx),
        (Some(physical), bound_pos) => StrictSelectorDecision::Refuse(StrictRefusal::Diverged {
            strict_idx,
            bound_physical: physical,
            bound_pos,
        }),
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

    /// Whether this instance carries a debug messenger — i.e. whether validation output can be
    /// observed through it. Used by the §6.5 device registry to report when a later session asks
    /// for validation on an instance that was created without it.
    #[inline]
    pub(crate) fn validation_armed(&self) -> bool {
        self.debug_messenger.is_some()
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

            // SAFETY: handle is live; pdev came from `enumerate_physical_devices` against it,
            // this loop's own contract.
            let identity = unsafe { query_device_identity(&self.handle, pdev) };

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
                identity,
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

/// Query the stable per-physical-device identity fields (issue #18 device-selection contract):
/// UUID, LUID and PCI bus location.
///
/// - `uuid` is `Some` for every driver that populates `VkPhysicalDeviceIDProperties` (Vulkan 1.1
///   core, and the §7.2 gate already requires >= 1.1) and `None` when the returned `deviceUUID`
///   is **all zeros**. All-zero is not an identity: it is what an untouched struct contains, it
///   compares equal across every device that has it, and admitting it as a value would rebuild
///   the collision this whole mechanism exists to remove. Callers print `(unavailable)`.
/// - `luid` is `Some` only when the driver sets `deviceLUIDValid = VK_TRUE` (primarily
///   Windows/D3D-interop; most Linux, Android and MoltenVK drivers leave it `VK_FALSE`).
/// - `pci` is `Some` only when the device advertises `VK_EXT_pci_bus_info`. Absent on
///   MoltenVK and many mobile/virtualized ICDs — there may be no PCI bus to report at all.
///
/// Never panics and never fabricates a value: a driver that does not report an optional field
/// yields `None` for it, so a selector that requires that field can distinguish "not this
/// device" from "this identity kind is not available on this platform"
/// ([`DeviceSelectionError::UnsupportedIdentity`]).
///
/// # Safety
/// `handle` must be a live `ash::Instance`; `pdev` must be a physical device handle obtained
/// from that same instance.
pub(crate) unsafe fn query_device_identity(
    handle: &ash::Instance,
    pdev: vk::PhysicalDevice,
) -> DeviceIdentity {
    // SAFETY: handle is live; pdev came from it per this function's contract.
    let extensions =
        unsafe { handle.enumerate_device_extension_properties(pdev) }.unwrap_or_default();
    let has_pci_ext = extensions.iter().any(|p| {
        // SAFETY: Vulkan guarantees extensionName is a valid null-terminated UTF-8 string.
        unsafe { CStr::from_ptr(p.extension_name.as_ptr()) }.to_bytes() == b"VK_EXT_pci_bus_info"
    });

    let mut id_props = vk::PhysicalDeviceIDProperties::default();
    let mut pci_props = vk::PhysicalDevicePCIBusInfoPropertiesEXT::default();

    // ash 0.38 `push_next` quirk (see `caps.rs::probe` and `device.rs` for the same fix):
    // `push_next` takes `self` by value and returns `Self`. Discarding the return value leaves
    // `p_next` unlinked and the chained struct stays zeroed — always re-bind.
    let mut props2 = {
        let p = vk::PhysicalDeviceProperties2::default().push_next(&mut id_props);
        if has_pci_ext {
            p.push_next(&mut pci_props)
        } else {
            p
        }
    };
    // SAFETY: handle is live; pdev came from it. The p_next chain contains only structs that
    // live on this stack frame and is read only during this call — nothing escapes.
    unsafe { handle.get_physical_device_properties2(pdev, &mut props2) };
    let _ = props2; // props2 mutably borrows id_props/pci_props; keep its last use here (NLL).

    let uuid = (!id_props.device_uuid.iter().all(|b| *b == 0)).then(|| {
        id_props
            .device_uuid
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    });
    let luid = (id_props.device_luid_valid != 0).then(|| {
        id_props
            .device_luid
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    });
    let pci = has_pci_ext.then(|| {
        format!(
            "{:04x}:{:02x}:{:02x}.{:x}",
            pci_props.pci_domain, pci_props.pci_bus, pci_props.pci_device, pci_props.pci_function
        )
    });

    DeviceIdentity { uuid, luid, pci }
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

        // §index-spaces note: this label uses the Vulkan enumeration index (the order
        // vkEnumeratePhysicalDevices returns devices), which is NOT the same as the
        // ONNXRUNTIME_EP_VULKAN_DEVICE selector index.  See the "Selector index map" block below.
        out.push(format!(
            "Device {idx} [Vulkan enum index {idx}]: {} [Vulkan {}.{}.{}]  — gate {}",
            name,
            vk::api_version_major(api_v),
            vk::api_version_minor(api_v),
            vk::api_version_patch(api_v),
            verdict,
        ));
        for c in &criteria {
            out.push(c.row());
        }

        // Stable identity (issue #18): printed for every enumerated device, independent of gate
        // pass/fail — identity is queryable off `VkPhysicalDeviceIDProperties`
        // (Vulkan 1.1 core) and `VK_EXT_pci_bus_info`, neither of which the §7.2 gate depends on.
        // Never fabricated: `(unavailable)` means the driver did not report that field, not that
        // epctl failed to read it. This loop deliberately runs on **gate-failing** devices too —
        // possibly Vulkan 1.0 parts, where `VkPhysicalDeviceIDProperties` may not be populated at
        // all — so an all-zero `deviceUUID` prints `(unavailable)` rather than 32 zeros, which
        // would otherwise read as an identity every unpopulated device shares.
        // SAFETY: inst.handle is live; pdev came from `enumerate_physical_devices` against it,
        // earlier in this same loop iteration.
        let id = unsafe { query_device_identity(&inst.handle, pdev) };
        out.push(format!(
            "  identity: uuid={} luid={} pci={}",
            id.uuid.as_deref().unwrap_or("(unavailable)"),
            id.luid.as_deref().unwrap_or("(unavailable)"),
            id.pci.as_deref().unwrap_or("(unavailable)"),
        ));

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
            // Print the two-index-space map so readers know exactly which device
            // ONNXRUNTIME_EP_VULKAN_DEVICE=N selects.
            //
            // Two index spaces exist and are NOT interchangeable:
            //   • Vulkan enum index  — from vkEnumeratePhysicalDevices (printed in the
            //                          "Device N [Vulkan enum index N]" lines above).
            //   • Selector index     — ONNXRUNTIME_EP_VULKAN_DEVICE value; indexes the
            //                          capability-gate-passing devices sorted best-first
            //                          (discrete > integrated > virtual > other).
            //
            // On a typical two-GPU machine the discrete GPU is selector index 0 even though
            // the Vulkan driver may enumerate it as device 1.  The table below resolves this.
            out.push(String::new());
            out.push(format!(
                "{ENV_DEVICE_SELECTOR} selector index map \
                 (best-first sorted; use these indices for {ENV_DEVICE_SELECTOR}):"
            ));
            for (sort_idx, (vulkan_idx, name, _)) in passing.iter().enumerate() {
                out.push(format!(
                    "  [{sort_idx}] '{name}'  (Vulkan enum index {vulkan_idx})",
                ));
            }

            // Apply the selector logic (mirrors select_device but on the probe-only data).
            let selected_name = if selector_val.is_empty() {
                format!(
                    "selector index 0 → '{}' (Vulkan enum index {}; best-first default; \
                     set {ENV_DEVICE_SELECTOR}=<selector-index|name> to override)",
                    passing[0].1, passing[0].0
                )
            } else if let Ok(idx) = selector_val.parse::<usize>() {
                passing
                    .get(idx)
                    .map(|(vulkan_idx, n, _)| {
                        format!("selector index {idx} → '{n}' (Vulkan enum index {vulkan_idx})")
                    })
                    .unwrap_or_else(|| {
                        format!(
                            "selector index {idx} out of range — would fall back to '{}' \
                             (Vulkan enum index {})",
                            passing[0].1, passing[0].0
                        )
                    })
            } else {
                let lower = selector_val.to_lowercase();
                passing
                    .iter()
                    .find(|(_, n, _)| n.to_lowercase().contains(&lower))
                    .map(|(vulkan_idx, n, _)| {
                        format!(
                            "name match → '{n}' (Vulkan enum index {vulkan_idx}; \
                             matched '{selector_val}')"
                        )
                    })
                    .unwrap_or_else(|| {
                        format!(
                            "no name matches '{selector_val}' — would fall back to '{}' \
                             (Vulkan enum index {})",
                            passing[0].1, passing[0].0
                        )
                    })
            };
            out.push(format!("Would select: {selected_name}"));

            // Stable-identity strict selector (issue #18). Reuses `resolve_device_selector`
            // against `enumerate_capable_devices()`'s full `DeviceInfo` list — the exact same
            // resolution the EP itself runs at session-creation time — rather than re-deriving
            // an ad-hoc match here, so this diagnostic cannot drift from the real selector logic.
            let strict_val = std::env::var(ENV_DEVICE_SELECTOR_STRICT).unwrap_or_default();
            out.push(String::new());
            out.push(format!(
                "{ENV_DEVICE_SELECTOR_STRICT} = {}",
                if strict_val.is_empty() {
                    "<not set — legacy selector above applies>".to_string()
                } else {
                    strict_val.clone()
                }
            ));
            if !strict_val.is_empty() {
                let capable = inst.enumerate_capable_devices();
                let infos: Vec<DeviceInfo> = capable.iter().map(|c| c.info.clone()).collect();
                match parse_device_selector(&strict_val) {
                    Ok(sel) => match resolve_device_selector(&infos, &sel) {
                        Ok(idx) => out.push(format!(
                            "Strict selector would select: '{}' (Vulkan enum index {}, {})",
                            infos[idx].name,
                            infos[idx].index,
                            infos[idx].key().canonical()
                        )),
                        Err(e) => out.push(format!(
                            "Strict selector would REFUSE to select any device: {e} (§6.5: no \
                             silent fallback — sessions using this selector would advertise zero \
                             Vulkan devices and fall back to the CPU EP)"
                        )),
                    },
                    Err(e) => out.push(format!(
                        "Strict selector string is malformed: {e} (would be refused identically \
                         at session creation)"
                    )),
                }
            }
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

    // THE TWO INDEX SPACES, PINNED DOWN.
    //
    // This desk is the case that has cost the project three defects: the discrete GPU is
    // enumeration index 1 but best-first selector index 0, and the integrated GPU is the reverse.
    // Any translation that is accidentally the identity passes on a one-GPU machine and inverts
    // every label here, so the test is written on the INVERTED pairing on purpose.
    #[test]
    fn a_physical_index_translates_to_a_selector_position_and_is_not_the_identity() {
        // best-first order: [NVIDIA (enum 1), Intel (enum 0)]
        let enum_indices = [1usize, 0];

        assert_eq!(
            position_of_physical_in(enum_indices.iter().copied(), 1),
            Some(0),
            "physical 1 (the discrete GPU) is selector 0 on this pairing"
        );
        assert_eq!(
            position_of_physical_in(enum_indices.iter().copied(), 0),
            Some(1),
            "physical 0 (the integrated GPU) is selector 1 on this pairing"
        );

        // The falsifier: if the translation were the identity, both answers above would equal
        // their inputs. Assert the inversion explicitly so a refactor to `Some(physical)` fails.
        for physical in [0usize, 1] {
            assert_ne!(
                position_of_physical_in(enum_indices.iter().copied(), physical),
                Some(physical),
                "the two spaces are inverted on this pairing; an identity mapping is the bug"
            );
        }

        // A device ORT names that we did not gate through is a real answer, not a fallback to 0.
        assert_eq!(
            position_of_physical_in(enum_indices.iter().copied(), 7),
            None,
            "an unknown physical index must not silently resolve to device 0"
        );
        assert_eq!(position_of_physical_in(std::iter::empty(), 0), None);

        // When enumeration order already equals best-first order the mapping IS the identity —
        // which is exactly why a one-GPU (or already-sorted) machine can never detect the defect.
        let sorted = [0usize, 1];
        assert_eq!(position_of_physical_in(sorted.iter().copied(), 0), Some(0));
        assert_eq!(position_of_physical_in(sorted.iter().copied(), 1), Some(1));
    }

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

    // ──────────────────────────────────────────────────────────────────────────
    // Stable-identity device selection (issue #18)
    // ──────────────────────────────────────────────────────────────────────────

    /// Build a synthetic two-device desk: the RTX A1000 this machine actually has, and a second
    /// simulated GPU, so ambiguity/uniqueness tests don't need real hardware.
    fn two_device_desk() -> Vec<DeviceInfo> {
        vec![
            DeviceInfo {
                index: 1, // physical enum index != best-first position, on purpose (see below)
                name: "NVIDIA RTX A1000 Laptop GPU".to_string(),
                vendor_id: 0x10de,
                device_id: 0x27a0,
                api_version: "1.3.290".to_string(),
                driver_version: "560.94".to_string(),
                kind: DeviceKind::Discrete,
                identity: DeviceIdentity {
                    uuid: Some("11111111111111111111111111111111".to_string()),
                    luid: Some("aaaaaaaaaaaaaaaa".to_string()),
                    pci: Some("0000:01:00.0".to_string()),
                },
            },
            DeviceInfo {
                index: 0,
                name: "Intel(R) Iris(R) Xe Graphics".to_string(),
                vendor_id: 0x8086,
                device_id: 0x9a49,
                api_version: "1.3.277".to_string(),
                driver_version: "31.0.101.5333".to_string(),
                kind: DeviceKind::Integrated,
                identity: DeviceIdentity {
                    uuid: Some("22222222222222222222222222222222".to_string()),
                    luid: None,
                    pci: None,
                },
            },
        ]
    }

    /// Two identical RTX A1000s (e.g. two cards of the same model installed) — the case where
    /// `id:` (vendor+device) cannot disambiguate but `uuid:` still can.
    fn two_identical_gpus_desk() -> Vec<DeviceInfo> {
        vec![
            DeviceInfo {
                index: 0,
                name: "NVIDIA RTX A1000 Laptop GPU".to_string(),
                vendor_id: 0x10de,
                device_id: 0x27a0,
                api_version: "1.3.290".to_string(),
                driver_version: "560.94".to_string(),
                kind: DeviceKind::Discrete,
                identity: DeviceIdentity {
                    uuid: Some("33333333333333333333333333333333".to_string()),
                    luid: None,
                    pci: Some("0000:01:00.0".to_string()),
                },
            },
            DeviceInfo {
                index: 1,
                name: "NVIDIA RTX A1000 Laptop GPU".to_string(),
                vendor_id: 0x10de,
                device_id: 0x27a0,
                api_version: "1.3.290".to_string(),
                driver_version: "560.94".to_string(),
                kind: DeviceKind::Discrete,
                identity: DeviceIdentity {
                    uuid: Some("44444444444444444444444444444444".to_string()),
                    luid: None,
                    pci: Some("0000:02:00.0".to_string()),
                },
            },
        ]
    }

    // -- parse_device_selector ------------------------------------------------------------

    #[test]
    fn parse_accepts_every_documented_scheme() {
        assert_eq!(
            parse_device_selector("index:1").unwrap(),
            DeviceSelector::Index(1)
        );
        assert_eq!(
            parse_device_selector("name:NVIDIA RTX A1000 Laptop GPU").unwrap(),
            DeviceSelector::Name("NVIDIA RTX A1000 Laptop GPU".to_string())
        );
        assert_eq!(
            parse_device_selector("id:10de:27a0").unwrap(),
            DeviceSelector::VendorDevice(0x10de, 0x27a0)
        );
        assert_eq!(
            parse_device_selector("uuid:11111111111111111111111111111111").unwrap(),
            DeviceSelector::Uuid("11111111111111111111111111111111".to_string())
        );
        // Canonical dashed UUID form is also accepted (separators stripped).
        assert_eq!(
            parse_device_selector("uuid:11111111-1111-1111-1111-111111111111").unwrap(),
            DeviceSelector::Uuid("11111111111111111111111111111111".to_string())
        );
        assert_eq!(
            parse_device_selector("luid:aaaaaaaaaaaaaaaa").unwrap(),
            DeviceSelector::Luid("aaaaaaaaaaaaaaaa".to_string())
        );
        assert_eq!(
            parse_device_selector("pci:0000:01:00.0").unwrap(),
            DeviceSelector::Pci("0000:01:00.0".to_string())
        );
    }

    #[test]
    fn parse_rejects_malformed_selectors_with_a_reason() {
        assert!(parse_device_selector("").is_err());
        assert!(parse_device_selector("bare-no-scheme").is_err());
        assert!(parse_device_selector("index:not-a-number").is_err());
        assert!(parse_device_selector("unknownscheme:foo").is_err());
        assert!(parse_device_selector("id:onlyvendor").is_err());
        assert!(
            parse_device_selector("id:zz:27a0").is_err(),
            "vendor is not hex"
        );
        assert!(
            parse_device_selector("uuid:tooshort").is_err(),
            "uuid must be exactly 32 hex chars"
        );
        assert!(
            parse_device_selector("uuid:1111111111111111111111111111111g").is_err(),
            "uuid must be hex, 'g' is not"
        );
    }

    // -- resolve_device_selector: exact match -----------------------------------------------

    #[test]
    fn resolve_exact_uuid_match_picks_the_named_device() {
        let devices = two_device_desk();
        let sel = DeviceSelector::Uuid("11111111111111111111111111111111".to_string());
        assert_eq!(resolve_device_selector(&devices, &sel), Ok(0));
    }

    #[test]
    fn resolve_exact_name_match_picks_the_named_device() {
        let devices = two_device_desk();
        let sel = DeviceSelector::Name("Intel(R) Iris(R) Xe Graphics".to_string());
        assert_eq!(resolve_device_selector(&devices, &sel), Ok(1));
    }

    #[test]
    fn resolve_vendor_device_id_picks_the_unique_match() {
        let devices = two_device_desk();
        let sel = DeviceSelector::VendorDevice(0x8086, 0x9a49);
        assert_eq!(resolve_device_selector(&devices, &sel), Ok(1));
    }

    #[test]
    fn resolve_pci_picks_the_unique_match() {
        let devices = two_device_desk();
        let sel = DeviceSelector::Pci("0000:01:00.0".to_string());
        assert_eq!(resolve_device_selector(&devices, &sel), Ok(0));
    }

    #[test]
    fn resolve_luid_picks_the_unique_match() {
        let devices = two_device_desk();
        let sel = DeviceSelector::Luid("aaaaaaaaaaaaaaaa".to_string());
        assert_eq!(resolve_device_selector(&devices, &sel), Ok(0));
    }

    #[test]
    fn resolve_index_is_a_displayed_ordinal_not_an_identity() {
        let devices = two_device_desk();
        assert_eq!(
            resolve_device_selector(&devices, &DeviceSelector::Index(0)),
            Ok(0)
        );
        assert_eq!(
            resolve_device_selector(&devices, &DeviceSelector::Index(1)),
            Ok(1)
        );
    }

    // -- resolve_device_selector: not found --------------------------------------------------

    #[test]
    fn resolve_not_found_for_an_index_out_of_range_never_falls_back_to_zero() {
        let devices = two_device_desk();
        let err = resolve_device_selector(&devices, &DeviceSelector::Index(9)).unwrap_err();
        assert!(matches!(err, DeviceSelectionError::NotFound { .. }));
    }

    #[test]
    fn resolve_not_found_for_an_unmatched_uuid() {
        let devices = two_device_desk();
        let sel = DeviceSelector::Uuid("99999999999999999999999999999999".to_string());
        let err = resolve_device_selector(&devices, &sel).unwrap_err();
        match err {
            DeviceSelectionError::NotFound { available, .. } => {
                assert_eq!(
                    available.len(),
                    2,
                    "both devices should be listed as available"
                );
            }
            other => panic!("expected NotFound, got {other:?}"),
        }
    }

    #[test]
    fn resolve_not_found_for_an_unmatched_name_never_substring_matches() {
        // The strict selector's `name:` is EXACT, unlike the legacy selector's substring match —
        // "RTX" must not match "NVIDIA RTX A1000 Laptop GPU".
        let devices = two_device_desk();
        let sel = DeviceSelector::Name("RTX".to_string());
        assert!(matches!(
            resolve_device_selector(&devices, &sel),
            Err(DeviceSelectionError::NotFound { .. })
        ));
    }

    #[test]
    fn resolve_not_found_on_an_empty_device_list() {
        let sel = DeviceSelector::Uuid("11111111111111111111111111111111".to_string());
        let err = resolve_device_selector(&[], &sel).unwrap_err();
        match err {
            DeviceSelectionError::NotFound { available, .. } => assert!(available.is_empty()),
            other => panic!("expected NotFound, got {other:?}"),
        }
    }

    /// "This platform cannot express that identity" and "there are no devices at all" are
    /// different diagnoses with different remedies, and `Iterator::all` answers `true` for both
    /// unless the empty case is excluded. A user on a machine where the loader found nothing must
    /// be told to fix their loader, not told that their driver does not support UUIDs.
    #[test]
    fn an_identity_is_unsupported_only_when_devices_exist_and_none_report_it() {
        let anonymous = vec![DeviceInfo {
            index: 0,
            name: "Some Vulkan Device".to_string(),
            vendor_id: 0x1234,
            device_id: 0x5678,
            api_version: "1.3.0".to_string(),
            driver_version: "1.0".to_string(),
            kind: DeviceKind::Discrete,
            identity: DeviceIdentity {
                uuid: None,
                luid: None,
                pci: None,
            },
        }];
        for sel in [
            DeviceSelector::Uuid("11111111111111111111111111111111".to_string()),
            DeviceSelector::Luid("aaaaaaaaaaaaaaaa".to_string()),
            DeviceSelector::Pci("0000:01:00.0".to_string()),
        ] {
            assert!(
                matches!(
                    resolve_device_selector(&anonymous, &sel),
                    Err(DeviceSelectionError::UnsupportedIdentity { .. })
                ),
                "a device exists but reports no such identity: {sel:?}"
            );
            assert!(
                matches!(
                    resolve_device_selector(&[], &sel),
                    Err(DeviceSelectionError::NotFound { .. })
                ),
                "no devices at all is NotFound, not UnsupportedIdentity: {sel:?}"
            );
        }
    }

    // -- select_device_strict: precedence between the two selector surfaces -------------------

    /// C3 / blocker 4: **one** authoritative selection path.
    ///
    /// `select_device_strict` is that path. It is called from exactly two places — the advertise
    /// path in `engine.rs` (which has no session and passes `None`) and the per-session bind path
    /// in `device.rs` (which passes `ep.device_selector`) — and both resolve through this one
    /// function against the same device list. The precedence rule is pinned here rather than left
    /// to the two call sites to agree on, because a disagreement between them is invisible: each
    /// one individually picks *a* device, and only a reader comparing both discovers they picked
    /// different ones.
    #[test]
    fn the_session_option_outranks_the_environment_and_neither_needs_the_other() {
        let devices = two_device_desk();
        let _g = crate::allocator::ledger::test_lock();

        // SAFETY: the shared lock above serialises every test in this binary that touches the
        // process environment; the variable is removed again below.
        unsafe { std::env::set_var(ENV_DEVICE_SELECTOR_STRICT, "index:0") };

        assert_eq!(
            select_device_strict(&devices, None),
            Ok(Some(0)),
            "with no session option, the environment selects"
        );
        assert_eq!(
            select_device_strict(&devices, Some("index:1")),
            Ok(Some(1)),
            "the session option is the more specific request and must win; if the environment won \
             instead, a host that sets `ep.device_selector` would silently run on another GPU \
             while its own logs said otherwise"
        );
        assert_eq!(
            select_device_strict(&devices, Some("")),
            Ok(Some(0)),
            "an empty option is absence, not a request, so the environment is still in charge"
        );

        // SAFETY: as above.
        unsafe { std::env::remove_var(ENV_DEVICE_SELECTOR_STRICT) };

        assert_eq!(
            select_device_strict(&devices, None),
            Ok(None),
            "with neither set the strict path abstains, leaving the legacy `ep.device_index` / \
             ONNXRUNTIME_EP_VULKAN_DEVICE path exactly as it was before issue #18"
        );
        assert_eq!(
            select_device_strict(&devices, Some("index:1")),
            Ok(Some(1)),
            "and the option alone is sufficient — it does not require the env var to be set too"
        );
    }

    /// A selector that names no device is a refusal, at both surfaces, with the same error.
    /// Falling back is how a run on the wrong GPU gets labelled as a run on the right one.
    #[test]
    fn an_unresolvable_selector_refuses_rather_than_choosing_a_neighbour() {
        let devices = two_device_desk();
        let _g = crate::allocator::ledger::test_lock();
        // SAFETY: serialised by the lock above; this test must see no ambient selector.
        unsafe { std::env::remove_var(ENV_DEVICE_SELECTOR_STRICT) };

        let err = select_device_strict(&devices, Some("uuid:99999999999999999999999999999999"))
            .expect_err("a uuid that matches nothing must not resolve");
        assert!(
            matches!(err, DeviceSelectionError::NotFound { .. }),
            "the caller decides what to do about it, but it must be told the selector FAILED \
             rather than handed a fallback device; got {err:?}"
        );
        assert!(
            matches!(
                select_device_strict(&devices, Some("not-a-scheme")),
                Err(DeviceSelectionError::Malformed { .. })
            ),
            "a typo is a refusal too"
        );
        assert!(
            matches!(
                select_device_strict(&devices, Some("id:10de:27a0")),
                Ok(Some(0))
            ),
            "and a resolvable option still resolves, so the refusals above are not vacuous"
        );
    }

    /// The grammar is shared. `ep.device_selector` is a transport that hands its string to this
    /// same parser, so every scheme the environment variable accepts the session option accepts
    /// too, byte for byte. Two grammars would mean two answers to "which device did you mean".
    #[test]
    fn both_selector_surfaces_share_one_grammar() {
        for raw in [
            "uuid:11111111-1111-1111-1111-111111111111",
            "id:10de:27a0",
            "pci:0000:01:00.0",
            "name:NVIDIA RTX A1000 Laptop GPU",
            "index:1",
            "luid:aaaaaaaaaaaaaaaa",
        ] {
            let parsed = parse_device_selector(raw)
                .unwrap_or_else(|e| panic!("the shared grammar must accept `{raw}`: {e}"));
            let devices = two_device_desk();
            let _g = crate::allocator::ledger::test_lock();
            // SAFETY: serialised by the lock; the option path must not consult the environment.
            unsafe { std::env::remove_var(ENV_DEVICE_SELECTOR_STRICT) };
            assert_eq!(
                select_device_strict(&devices, Some(raw)),
                resolve_device_selector(&devices, &parsed).map(Some),
                "the option surface must resolve `{raw}` to exactly what the parsed selector \
                 resolves to — no second grammar, no second resolver"
            );
        }
    }

    // -- resolve_device_selector: ambiguous --------------------------------------------------

    #[test]
    fn resolve_ambiguous_when_id_matches_two_identical_gpus() {
        let devices = two_identical_gpus_desk();
        let sel = DeviceSelector::VendorDevice(0x10de, 0x27a0);
        let err = resolve_device_selector(&devices, &sel).unwrap_err();
        match err {
            DeviceSelectionError::Ambiguous { matches, .. } => assert_eq!(matches.len(), 2),
            other => panic!("expected Ambiguous, got {other:?}"),
        }
    }

    #[test]
    fn resolve_ambiguous_id_is_resolved_by_the_more_specific_uuid_selector() {
        // The identical-GPU desk is ambiguous by (vendor, device) but each card still has its
        // own UUID — the finer-grained identity resolves what the coarser one cannot.
        let devices = two_identical_gpus_desk();
        let sel = DeviceSelector::Uuid("44444444444444444444444444444444".to_string());
        assert_eq!(resolve_device_selector(&devices, &sel), Ok(1));
    }

    // -- resolve_device_selector: unsupported identity ---------------------------------------

    #[test]
    fn resolve_pci_is_unsupported_identity_when_no_device_reports_it() {
        let mut devices = two_device_desk();
        for d in &mut devices {
            d.identity.pci = None;
        }
        let sel = DeviceSelector::Pci("0000:01:00.0".to_string());
        let err = resolve_device_selector(&devices, &sel).unwrap_err();
        assert!(matches!(
            err,
            DeviceSelectionError::UnsupportedIdentity { .. }
        ));
    }

    #[test]
    fn resolve_luid_is_unsupported_identity_when_no_device_reports_it() {
        let mut devices = two_device_desk();
        for d in &mut devices {
            d.identity.luid = None;
        }
        let sel = DeviceSelector::Luid("aaaaaaaaaaaaaaaa".to_string());
        let err = resolve_device_selector(&devices, &sel).unwrap_err();
        assert!(matches!(
            err,
            DeviceSelectionError::UnsupportedIdentity { .. }
        ));
    }

    #[test]
    fn unsupported_identity_is_distinct_from_not_found() {
        // A selector whose identity kind IS available, but whose value matches nothing, is
        // NotFound (the GPU might show up later). A selector whose identity kind is not
        // available on this platform AT ALL is UnsupportedIdentity (it never will).
        let devices = two_device_desk(); // both devices DO have a pci field here
        let absent_value = DeviceSelector::Pci("ffff:ff:ff.f".to_string());
        assert!(matches!(
            resolve_device_selector(&devices, &absent_value),
            Err(DeviceSelectionError::NotFound { .. })
        ));

        let mut no_pci_devices = devices;
        for d in &mut no_pci_devices {
            d.identity.pci = None;
        }
        assert!(matches!(
            resolve_device_selector(&no_pci_devices, &absent_value),
            Err(DeviceSelectionError::UnsupportedIdentity { .. })
        ));
    }

    // -- select_device_strict: env var + session-option precedence, and the "not set" default --

    #[test]
    fn select_device_strict_returns_none_when_nothing_is_set() {
        let _g = serial_env();
        // SAFETY: test-only env mutation, serialized by `serial_env`.
        unsafe { std::env::remove_var(ENV_DEVICE_SELECTOR_STRICT) };
        let devices = two_device_desk();
        assert_eq!(select_device_strict(&devices, None), Ok(None));
    }

    #[test]
    fn select_device_strict_reads_the_env_var_when_no_session_override() {
        let _g = serial_env();
        // SAFETY: test-only env mutation, serialized by `serial_env`.
        unsafe { std::env::set_var(ENV_DEVICE_SELECTOR_STRICT, "index:1") };
        let devices = two_device_desk();
        assert_eq!(select_device_strict(&devices, None), Ok(Some(1)));
        // SAFETY: test-only env mutation, serialized by `serial_env`.
        unsafe { std::env::remove_var(ENV_DEVICE_SELECTOR_STRICT) };
    }

    #[test]
    fn select_device_strict_session_override_outranks_the_env_var() {
        let _g = serial_env();
        // SAFETY: test-only env mutation, serialized by `serial_env`.
        unsafe { std::env::set_var(ENV_DEVICE_SELECTOR_STRICT, "index:1") };
        let devices = two_device_desk();
        assert_eq!(select_device_strict(&devices, Some("index:0")), Ok(Some(0)));
        // SAFETY: test-only env mutation, serialized by `serial_env`.
        unsafe { std::env::remove_var(ENV_DEVICE_SELECTOR_STRICT) };
    }

    #[test]
    fn select_device_strict_malformed_env_value_is_an_error_not_a_fallback() {
        let _g = serial_env();
        // SAFETY: test-only env mutation, serialized by `serial_env`.
        unsafe { std::env::set_var(ENV_DEVICE_SELECTOR_STRICT, "not-a-valid-selector") };
        let devices = two_device_desk();
        assert!(matches!(
            select_device_strict(&devices, None),
            Err(DeviceSelectionError::Malformed { .. })
        ));
        // SAFETY: test-only env mutation, serialized by `serial_env`.
        unsafe { std::env::remove_var(ENV_DEVICE_SELECTOR_STRICT) };
    }

    /// Serializes tests that mutate `ENV_DEVICE_SELECTOR_STRICT` so they cannot interleave under
    /// `cargo test`'s default multi-threaded runner (env vars are process-global).
    fn serial_env() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    // -- Display impls used in log lines: smoke-test they don't panic and carry the selector --

    #[test]
    fn device_selector_display_round_trips_the_scheme() {
        assert_eq!(DeviceSelector::Index(3).to_string(), "index:3");
        assert_eq!(
            DeviceSelector::VendorDevice(0x10de, 0x27a0).to_string(),
            "id:10de:27a0"
        );
    }

    #[test]
    fn device_selection_error_display_names_the_selector_and_the_candidates() {
        let err = DeviceSelectionError::Ambiguous {
            selector: "id:10de:27a0".to_string(),
            matches: vec!["A (index 0)".to_string(), "B (index 1)".to_string()],
        };
        let msg = err.to_string();
        assert!(msg.contains("id:10de:27a0"));
        assert!(msg.contains("A (index 0)"));
        assert!(msg.contains("B (index 1)"));
    }

    // ──────────────────────────────────────────────────────────────────────────────
    // THE STRICT-SELECTOR DIVERGENCE PREDICATE (issue #18, contract C6)
    //
    // The protocol below is written once and run three times: against the real
    // `strict_selector_decision`, and against two deliberately defective reimplementations. A
    // test that only ever runs against the correct implementation proves the implementation
    // agrees with itself. Running the identical protocol against a mutant proves the protocol
    // would have *noticed* — which is the claim "this predicate is tested" actually makes.
    // ──────────────────────────────────────────────────────────────────────────────

    /// The C6 predicate's shape, named so the protocol can take any implementation of it —
    /// the real one and the mutants below.
    type C6Predicate<'a> = &'a dyn Fn(
        Result<Option<usize>, DeviceSelectionError>,
        Option<usize>,
        Option<usize>,
    ) -> StrictSelectorDecision;

    /// The C6 protocol: every row of the decision table, stated once.
    ///
    /// Returns `Err(description)` on the first row an implementation gets wrong, so a mutant's
    /// failure names which guarantee it dropped rather than just "assertion failed".
    fn c6_protocol(under_test: C6Predicate<'_>) -> Result<(), String> {
        use StrictSelectorDecision as D;

        let ambiguous = || DeviceSelectionError::Ambiguous {
            selector: "id:10de:27a0".to_string(),
            matches: vec!["A".to_string(), "B".to_string()],
        };
        let unsupported = || DeviceSelectionError::UnsupportedIdentity {
            selector: "pci:0000:01:00.0".to_string(),
            reason: "no enumerated device reports PCI addresses".to_string(),
        };
        let malformed = || DeviceSelectionError::Malformed {
            raw: "not-a-selector".to_string(),
            reason: "unknown scheme".to_string(),
        };
        let not_found = || DeviceSelectionError::NotFound {
            selector: "uuid:00000000000000000000000000000000".to_string(),
            available: vec!["A".to_string()],
        };

        let check = |what: &str, got: D, ok: bool| -> Result<(), String> {
            if ok {
                Ok(())
            } else {
                Err(format!("{what}: got {got:?}"))
            }
        };

        // No selector set: the predicate must stand down entirely. A mutant that "helpfully"
        // opens device 0 here would silently override the whole precedence chain.
        let got = under_test(Ok(None), Some(3), Some(1));
        check(
            "an unset strict selector must be NotRequested even when a binding exists",
            got.clone(),
            got == D::NotRequested,
        )?;

        // Exact agreement — the only case that passes. Note the identity resolved to best-first
        // position 1 and ORT bound physical device 7: the spaces differ, and agreement is
        // decided in best-first space, not by comparing raw indices.
        let got = under_test(Ok(Some(1)), Some(7), Some(1));
        check(
            "exact agreement in best-first space must open the selector's device",
            got.clone(),
            got == D::Open(1),
        )?;

        // ORT bound nothing: there is no second device, so there is nothing to diverge from.
        let got = under_test(Ok(Some(2)), None, None);
        check(
            "with no ORT binding the selector stands alone and must open",
            got.clone(),
            got == D::Open(2),
        )?;
        let got = under_test(Ok(Some(2)), None, Some(5));
        check(
            "a stale bound_pos with no bound_physical must not manufacture a divergence",
            got.clone(),
            got == D::Open(2),
        )?;

        // Divergence: ORT bound a different capable device.
        let got = under_test(Ok(Some(0)), Some(7), Some(1));
        check(
            "a selector/binding mismatch must REFUSE, not warn and continue",
            got.clone(),
            matches!(
                got,
                D::Refuse(StrictRefusal::Diverged {
                    strict_idx: 0,
                    bound_physical: 7,
                    bound_pos: Some(1)
                })
            ),
        )?;

        // Late divergence with an untranslatable binding: ORT bound a device outside the
        // §7.2-capable list. Agreement is not merely absent, it is unknowable — so refuse.
        let got = under_test(Ok(Some(0)), Some(9), None);
        check(
            "an untranslatable ORT binding is a divergence, not an absent one",
            got.clone(),
            matches!(
                got,
                D::Refuse(StrictRefusal::Diverged {
                    strict_idx: 0,
                    bound_physical: 9,
                    bound_pos: None
                })
            ),
        )?;

        // All four unresolvable selectors refuse, and none of them falls back.
        for (label, err) in [
            ("ambiguous", ambiguous()),
            ("unsupported identity", unsupported()),
            ("malformed", malformed()),
            ("not found", not_found()),
        ] {
            for binding in [None, Some(0usize)] {
                let got = under_test(Err(err.clone()), binding, binding);
                check(
                    &format!("an {label} selector must refuse (binding {binding:?})"),
                    got.clone(),
                    matches!(got, D::Refuse(StrictRefusal::Unresolvable(_))),
                )?;
            }
        }

        Ok(())
    }

    /// The real predicate satisfies C6.
    #[test]
    fn the_strict_selector_predicate_fails_closed_on_every_divergence() {
        if let Err(what) = c6_protocol(&|s, bp, pos| strict_selector_decision(s, bp, pos)) {
            panic!("strict_selector_decision violates C6 — {what}");
        }
    }

    /// Mutant A: the refusal is disabled outright — the selector always wins.
    ///
    /// This is what "just log a warning and open what the user asked for" compiles to. It is the
    /// single most plausible regression, because it makes the symptom (a refused session) go
    /// away while leaving the misattribution in place.
    #[test]
    fn a_predicate_that_never_refuses_divergence_is_caught() {
        let mutant = |strict: Result<Option<usize>, DeviceSelectionError>,
                      _bp: Option<usize>,
                      _pos: Option<usize>| match strict {
            Ok(None) => StrictSelectorDecision::NotRequested,
            Ok(Some(i)) => StrictSelectorDecision::Open(i),
            Err(e) => StrictSelectorDecision::Refuse(StrictRefusal::Unresolvable(e)),
        };
        let err = c6_protocol(&mutant).expect_err(
            "C6 must catch a predicate that opens the selector's device regardless of what ORT \
             bound — if this passes, the divergence half of C6 has no test",
        );
        assert!(
            err.contains("mismatch") || err.contains("untranslatable"),
            "the protocol must name the dropped guarantee, not just fail: {err}"
        );
    }

    /// Mutant B: an untranslatable binding is read as "no binding".
    ///
    /// The subtle one. `bound_pos == None` looks like "ORT did not bind anything relevant", so
    /// treating it as agreement is the natural mistake — and it is exactly the SPLIT-DEVICE
    /// condition with its diagnostic deleted.
    #[test]
    fn a_predicate_that_treats_an_untranslatable_binding_as_agreement_is_caught() {
        let mutant = |strict: Result<Option<usize>, DeviceSelectionError>,
                      bp: Option<usize>,
                      pos: Option<usize>| match strict {
            Ok(None) => StrictSelectorDecision::NotRequested,
            Err(e) => StrictSelectorDecision::Refuse(StrictRefusal::Unresolvable(e)),
            Ok(Some(i)) => match (bp, pos) {
                (Some(physical), Some(p)) if p != i => {
                    StrictSelectorDecision::Refuse(StrictRefusal::Diverged {
                        strict_idx: i,
                        bound_physical: physical,
                        bound_pos: Some(p),
                    })
                }
                _ => StrictSelectorDecision::Open(i),
            },
        };
        let err = c6_protocol(&mutant).expect_err(
            "C6 must catch a predicate that reads an untranslatable binding as agreement",
        );
        assert!(
            err.contains("untranslatable"),
            "the protocol must catch this on the untranslatable row specifically: {err}"
        );
    }

    /// Mutant C: an unresolvable selector falls back to the best-scoring device.
    ///
    /// The pre-#18 behaviour, and the reason C6 exists: a misspelled UUID producing a run on
    /// whatever GPU happened to score highest, attributed to the UUID that was asked for.
    #[test]
    fn a_predicate_that_falls_back_when_the_selector_is_unresolvable_is_caught() {
        let mutant = |strict: Result<Option<usize>, DeviceSelectionError>,
                      bp: Option<usize>,
                      pos: Option<usize>| match strict {
            Ok(None) => StrictSelectorDecision::NotRequested,
            Err(_) => StrictSelectorDecision::Open(0),
            Ok(Some(i)) => strict_selector_decision(Ok(Some(i)), bp, pos),
        };
        let err = c6_protocol(&mutant).expect_err(
            "C6 must catch a predicate that falls back to device 0 when the selector cannot be \
             resolved",
        );
        assert!(
            err.contains("must refuse"),
            "the protocol must catch this on an unresolvable row: {err}"
        );
    }
}
