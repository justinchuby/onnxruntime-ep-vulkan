//! `VulkanEpFactory` — our `OrtEpFactory` C-ABI vtable: one per library registration,
//! process-lived.
//!
//! The ORT struct is the **first** field under `#[repr(C)]`, so ORT's `*OrtEpFactory` is
//! pointer-identical to our `*VulkanEpFactory`. We fill exactly the slots we implement; the rest
//! stay `None` (zeroed), which ORT reads as "this EP does not support that".
//!
//! # M0 scope
//!
//! * Identity: name `VulkanExecutionProvider`, vendor `onnxruntime-ep-vulkan`, version from
//!   `CARGO_PKG_VERSION` so it can never drift from the manifest.
//! * `GetSupportedDevices` runs the capability probe ([`crate::engine::probe_devices`], a stub
//!   until Switch backs it with `vkEnumeratePhysicalDevices`) and advertises one `OrtEpDevice` per
//!   usable device, correlated with ORT's `OrtHardwareDevice` list by `(vendor_id, device_id)`
//!   with a device-type fallback. **Zero usable devices is a success with a warning, never an
//!   error** — a machine with no Vulkan loader must still create sessions on the CPU EP
//!   (`DESIGN.md` §2.3, M0 exit criterion 4).
//! * Allocator / data-transfer / sync-stream slots hand back nothing: M0/M1 keep I/O in host
//!   memory (`DESIGN.md` §6.3). They are still implemented because ORT calls some of them during
//!   registration and a null vtable slot there is not the same as a slot that returns null.

use std::ffi::{CStr, CString, c_char};
use std::ptr;
use std::sync::{Arc, Mutex};

use crate::engine::{self, DeviceInfo};
use crate::ep::VulkanEp;
use crate::sys::{self, ort};

/// The registered EP and device name. Frozen: it appears in every user's provider list
/// (`DESIGN.md` §3.1).
pub const EP_NAME: &CStr = c"VulkanExecutionProvider";
/// Vendor string reported by `GetVendor`.
pub const EP_VENDOR: &CStr = c"onnxruntime-ep-vulkan";

#[repr(C)]
pub struct VulkanEpFactory {
    /// MUST be first: ORT's pointer is this pointer.
    base: ort::OrtEpFactory,
    ort_api: *const ort::OrtApi,
    ep_api: *const ort::OrtEpApi,
    /// The ABI version negotiated with this host. Propagated to every `OrtEp` we create so a
    /// downlevel host never reads past what it understands.
    abi_version: u32,
    name: CString,
    vendor: CString,
    version: CString,
}

impl VulkanEpFactory {
    /// Build the factory ORT asked for.
    ///
    /// # Safety
    /// `registration_name` may be null or a valid NUL-terminated string; `ort_api` and `ep_api`
    /// must be the live tables negotiated in `CreateEpFactories`, and `abi_version` the version
    /// they were negotiated at.
    pub unsafe fn new(
        registration_name: *const c_char,
        ort_api: *const ort::OrtApi,
        ep_api: *const ort::OrtEpApi,
        abi_version: u32,
    ) -> Box<VulkanEpFactory> {
        // SAFETY: `registration_name` is either null or a NUL-terminated string ORT owns; we copy
        // it immediately and never retain the borrow.
        let name = unsafe {
            if registration_name.is_null() || *registration_name == 0 {
                EP_NAME.to_owned()
            } else {
                CStr::from_ptr(registration_name).to_owned()
            }
        };

        // SAFETY: `OrtEpFactory` is a `#[repr(C)]` vtable of a `u32` and function pointers, which
        // bindgen models as `Option<fn>`; all-zero is the valid `None` niche for each, so a zeroed
        // struct is a well-formed "nothing implemented" vtable that we then fill in.
        let mut base: ort::OrtEpFactory = unsafe { std::mem::zeroed() };
        // Report what the *host* serves, not what we were compiled against. ORT uses this to
        // decide how far into our vtable it may read; claiming 28 to a 1.24 host would invite it
        // to read fields it lays out differently.
        base.ort_version_supported = abi_version;
        base.GetName = Some(get_name);
        base.GetVendor = Some(get_vendor);
        base.GetVendorId = Some(get_vendor_id);
        base.GetVersion = Some(get_version);
        base.GetSupportedDevices = Some(get_supported_devices);
        base.CreateEp = Some(create_ep);
        base.ReleaseEp = Some(release_ep);
        base.CreateAllocator = Some(create_allocator);
        base.ReleaseAllocator = Some(release_allocator);
        base.CreateDataTransfer = Some(create_data_transfer);
        base.CreateSyncStreamForDevice = Some(create_sync_stream_for_device);
        base.IsStreamAware = Some(is_stream_aware);
        // `CreateExternalResourceImporterForDevice` (ORT 1.24+) is deliberately left `None` — ORT
        // reads that as "this EP cannot import external memory", which is true today. The gate is
        // written out even though it is currently always true, because it is the shape the code
        // must have and a silently-ungated field is exactly the bug this discipline prevents.
        if abi_version >= sys::since::EXTERNAL_RESOURCE_IMPORTER {
            // TODO(tank): zero-copy IO binding. One line here:
            //     base.CreateExternalResourceImporterForDevice = Some(create_external_resource_importer);
            // plus a new `importer.rs`. See the seam documentation at the bottom of `sys.rs` and
            // ORT's own `nv_vulkan_test.cc` for the intended contract.
            debug_assert!(base.CreateExternalResourceImporterForDevice.is_none());
        }

        Box::new(VulkanEpFactory {
            base,
            ort_api,
            ep_api,
            abi_version,
            name,
            vendor: EP_VENDOR.to_owned(),
            // Single-sourced from [package].version. The scheme is 0.<ORT_API_VERSION>.<patch>,
            // so 0.28.x pairs with ORT 1.28.x; `sys.rs` asserts the two agree at compile time.
            version: CString::new(env!("CARGO_PKG_VERSION"))
                .unwrap_or_else(|_| c"0.0.0".to_owned()),
        })
    }

    /// Hand ownership to ORT.
    pub fn into_raw(self: Box<Self>) -> *mut ort::OrtEpFactory {
        Box::into_raw(self).cast::<ort::OrtEpFactory>()
    }

    /// Take ownership back and drop.
    ///
    /// # Safety
    /// `p` must be a pointer previously returned by [`VulkanEpFactory::into_raw`], not yet
    /// released.
    pub unsafe fn release(p: *mut ort::OrtEpFactory) {
        if p.is_null() {
            return;
        }
        // Stop forwarding logs before ORT can invalidate the logger we were handed.
        crate::logging::detach_ort_logger();
        // SAFETY: `p` came from `Box::into_raw` on a `Box<VulkanEpFactory>`, and the struct is
        // `#[repr(C)]` with `base` first, so this pointer is the whole allocation.
        drop(unsafe { Box::from_raw(p.cast::<VulkanEpFactory>()) });
    }
}

/// Reinterpret ORT's `OrtEpFactory*` as our `VulkanEpFactory*`.
///
/// # Safety
/// `p` must be a pointer this crate produced via [`VulkanEpFactory::into_raw`].
#[inline]
unsafe fn this<'a>(p: *const ort::OrtEpFactory) -> &'a VulkanEpFactory {
    // SAFETY: `#[repr(C)]` with `base` first means same address; caller guarantees provenance.
    unsafe { &*p.cast::<VulkanEpFactory>() }
}

// -------------------------------------------------------------------------------------------
// Identity
// -------------------------------------------------------------------------------------------

unsafe extern "C" fn get_name(p: *const ort::OrtEpFactory) -> *const c_char {
    // SAFETY: `p` is ours; the `CString` lives as long as the factory.
    unsafe { this(p).name.as_ptr() }
}

unsafe extern "C" fn get_vendor(p: *const ort::OrtEpFactory) -> *const c_char {
    // SAFETY: as above.
    unsafe { this(p).vendor.as_ptr() }
}

unsafe extern "C" fn get_version(p: *const ort::OrtEpFactory) -> *const c_char {
    // SAFETY: as above.
    unsafe { this(p).version.as_ptr() }
}

/// PCI vendor ID reported for the factory as a whole.
///
/// Unlike a single-vendor EP there is no correct answer here — this EP runs on NVIDIA, AMD, Intel,
/// Qualcomm, ARM and software rasterizers. Each advertised `OrtEpDevice` carries the *real*
/// `VkPhysicalDeviceProperties::vendorID` of its device (see [`get_supported_devices`]); the
/// factory-level answer is 0, meaning "not a single-vendor EP". Tracked as OQ-6.
unsafe extern "C" fn get_vendor_id(_p: *const ort::OrtEpFactory) -> u32 {
    0
}

unsafe extern "C" fn is_stream_aware(_p: *const ort::OrtEpFactory) -> bool {
    // Streams arrive with the M2 memory work at the earliest.
    false
}

// -------------------------------------------------------------------------------------------
// GetSupportedDevices
// -------------------------------------------------------------------------------------------

unsafe extern "C" fn get_supported_devices(
    p: *mut ort::OrtEpFactory,
    devices: *const *const ort::OrtHardwareDevice,
    num_devices: usize,
    ep_devices: *mut *mut ort::OrtEpDevice,
    max_ep_devices: usize,
    num_ep_devices: *mut usize,
) -> ort::OrtStatusPtr {
    // SAFETY: `p` is our factory pointer; reading the API out before the guard lets a caught panic
    // still be reported as a status.
    let api = unsafe { this(p).ort_api };
    // SAFETY: `api` is live.
    unsafe {
        crate::guard_ffi_status(api, "GetSupportedDevices", || {
            get_supported_devices_impl(
                p,
                devices,
                num_devices,
                ep_devices,
                max_ep_devices,
                num_ep_devices,
            )
        })
    }
}

/// How an advertised device was matched to an ORT hardware device. Recorded in the EP device
/// metadata so a support report says which strategy was used (`DESIGN.md` §2.3 step 3).
fn correlation_strategy(exact: bool) -> &'static CStr {
    if exact {
        c"vendor_id+device_id"
    } else {
        c"device_type_fallback"
    }
}

/// # Safety
/// All pointers must be the live ones ORT passed to `GetSupportedDevices`.
unsafe fn get_supported_devices_impl(
    p: *mut ort::OrtEpFactory,
    devices: *const *const ort::OrtHardwareDevice,
    num_devices: usize,
    ep_devices: *mut *mut ort::OrtEpDevice,
    max_ep_devices: usize,
    num_ep_devices: *mut usize,
) -> ort::OrtStatusPtr {
    // SAFETY: `p` is our factory pointer.
    let factory = unsafe { this(p) };
    let api = factory.ort_api;
    let ep_api = factory.ep_api;

    if num_ep_devices.is_null() {
        // SAFETY: `api` is live.
        return unsafe {
            sys::make_status(
                api,
                ort::OrtErrorCode_ORT_INVALID_ARGUMENT,
                "GetSupportedDevices received a null num_ep_devices out-parameter",
            )
        };
    }
    // SAFETY: valid out-param slot; set first so every early return leaves it defined.
    unsafe { *num_ep_devices = 0 };

    // The capability gate. A machine with no Vulkan loader / no ICD / a broken driver yields an
    // empty list here — a warning and zero advertised devices, never an error status, so session
    // creation still succeeds on the CPU EP.
    let usable: Vec<DeviceInfo> = engine::devices_to_advertise();
    if usable.is_empty() {
        log::warn!(
            "VulkanExecutionProvider: no Vulkan device satisfies the capability gate on this \
             machine; advertising zero devices. Models will run on the CPU EP."
        );
        return ptr::null_mut();
    }
    if max_ep_devices == 0 || ep_devices.is_null() {
        log::warn!(
            "VulkanExecutionProvider: {} usable device(s) found but ORT allowed room for none",
            usable.len()
        );
        return ptr::null_mut();
    }

    // Snapshot ORT's hardware-device list so correlation is a pure function over plain data.
    let mut hw: Vec<(
        *const ort::OrtHardwareDevice,
        u32,
        u32,
        ort::OrtHardwareDeviceType,
    )> = Vec::with_capacity(num_devices);
    if !devices.is_null() {
        for i in 0..num_devices {
            // SAFETY: ORT guarantees `devices` has `num_devices` valid entries, and each accessor
            // is a plain getter over a live `OrtHardwareDevice`.
            unsafe {
                let d = *devices.add(i);
                if d.is_null() {
                    continue;
                }
                let vendor_id = (*api).HardwareDevice_VendorId.map_or(0, |f| f(d));
                let device_id = (*api).HardwareDevice_DeviceId.map_or(0, |f| f(d));
                let kind = (*api)
                    .HardwareDevice_Type
                    .map_or(ort::OrtHardwareDeviceType_OrtHardwareDeviceType_CPU, |f| {
                        f(d)
                    });
                hw.push((d, vendor_id, device_id, kind));
            }
        }
    }

    let mut written = 0usize;
    for info in &usable {
        if written == max_ep_devices {
            log::warn!(
                "VulkanExecutionProvider: {} usable device(s) but ORT allowed only {max_ep_devices}; \
                 the remainder are not advertised",
                usable.len()
            );
            break;
        }

        // Correlate: exact (vendor_id, device_id) first, then fall back to device type. Software
        // rasterizers, virtualized GPUs and MoltenVK routinely fail the exact match.
        let wanted_type = match info.kind {
            engine::DeviceKind::Cpu => ort::OrtHardwareDeviceType_OrtHardwareDeviceType_CPU,
            _ => ort::OrtHardwareDeviceType_OrtHardwareDeviceType_GPU,
        };
        let exact = hw
            .iter()
            .find(|(_, v, d, _)| *v == info.vendor_id && *d == info.device_id);
        let (hw_device, exact_match) = match exact {
            Some((d, _, _, _)) => (*d, true),
            None => match hw.iter().find(|(_, _, _, t)| *t == wanted_type) {
                Some((d, _, _, _)) => (*d, false),
                None => {
                    log::warn!(
                        "VulkanExecutionProvider: Vulkan device `{}` (vendor {:#06x}, device \
                         {:#06x}) could not be correlated with any OrtHardwareDevice; not \
                         advertising it",
                        info.name,
                        info.vendor_id,
                        info.device_id
                    );
                    continue;
                }
            },
        };

        // EP metadata: enough for a user (or a support report) to see exactly which physical
        // device was bound and how we matched it.
        let metadata = MetadataBuilder::new(api);
        metadata.add(c"vulkan.device_name", &info.name);
        metadata.add(c"vulkan.api_version", &info.api_version);
        metadata.add(c"vulkan.driver_version", &info.driver_version);
        metadata.add(c"vulkan.vendor_id", &format!("{:#06x}", info.vendor_id));
        metadata.add(c"vulkan.device_id", &format!("{:#06x}", info.device_id));
        metadata.add(c"vulkan.device_kind", &format!("{:?}", info.kind));
        metadata.add(c"vulkan.device_index", &info.index.to_string());
        metadata.add_cstr(c"vulkan.correlation", correlation_strategy(exact_match));

        let mut ep_device: *mut ort::OrtEpDevice = ptr::null_mut();
        // SAFETY: `ep_api` is live; `p` is our factory; `hw_device` came from ORT's own list;
        // `metadata` is a live `OrtKeyValuePairs` (or null, which ORT accepts) that ORT copies
        // from during this call.
        let status = unsafe {
            match (*ep_api).CreateEpDevice {
                Some(f) => f(p, hw_device, metadata.as_ptr(), ptr::null(), &mut ep_device),
                None => {
                    return sys::make_status(
                        api,
                        ort::OrtErrorCode_ORT_EP_FAIL,
                        "OrtEpApi::CreateEpDevice is unavailable",
                    );
                }
            }
        };
        if !status.is_null() {
            return status;
        }

        // SAFETY: `written < max_ep_devices`, so this slot is inside the array ORT gave us.
        unsafe { *ep_devices.add(written) = ep_device };
        written += 1;

        // Advertise device memory for this device. Without this, ORT has no `OrtMemoryInfo` that
        // names us and therefore never calls `CreateAllocator` — the allocator would exist and be
        // unreachable, which is exactly the "a registry that is not in ORT's path" objection that
        // made the earlier verification of the handle scheme a precondition dressed as an effect.
        //
        // Failure here is deliberately non-fatal: the EP is fully functional without a device
        // allocator (M0/M1 staged everything through host memory), so a host too old to have
        // `EpDevice_AddAllocatorInfo` gets a working EP with a log line rather than a dead one.
        //
        // The kill switch exists because this is the newest and least-exercised call we make into
        // ORT, and being able to turn it off without a rebuild is what let a registration crash be
        // bisected in one minute instead of one CI cycle.
        if device_memory_enabled() {
            // SAFETY: `ep_api`/`api` are live and `ep_device` is the handle `CreateEpDevice` just
            // produced.
            unsafe { advertise_device_memory(api, ep_api, ep_device, info) };
        }

        log::info!(
            "VulkanExecutionProvider advertising device #{} `{}` (vendor {:#06x}, Vulkan {}, \
             correlated by {})",
            info.index,
            info.name,
            info.vendor_id,
            info.api_version,
            correlation_strategy(exact_match).to_string_lossy()
        );
    }

    // SAFETY: valid out-param slot.
    unsafe { *num_ep_devices = written };
    ptr::null_mut()
}

/// Whether to advertise device memory at all.
///
/// **Default off, and the reason is a measured one rather than caution.** Advertising an
/// `OrtDeviceAllocator` with `OrtDeviceMemoryType_DEFAULT` is a package deal: ORT then requires a
/// registered `OrtDataTransferImpl` to move tensors between host memory and ours, and without one
/// every session fails at Run with
///
/// > There's no data transfer registered for copying tensors from Device:[DeviceType:0 …] to
/// > Device:[DeviceType:1 … Alignment:4096]
///
/// — verified locally against ORT 1.28 on both devices.
///
/// **That reason has expired, and this is what replaced it (2026-08-02).** It said the data
/// transfer could not be written until the handle→`VkBuffer` seam was filled. That seam is filled:
/// `CreateDataTransfer` is registered unconditionally and `VulkanDataTransfer` exists. The default
/// is still off, but now for a *measured* reason, and here is the measurement — device 0, Phi-3.5,
/// counters only, no clock (`bench/results/device_memory_kv_lanes.json`):
///
/// | per inference | default lane (`OFF`) | armed lane (`SHARED`) |
/// |---|---|---|
/// | staging **readback** @ ctx 0 / 128 | 457,344 / 50,788,992 B | **identical** |
/// | readback per past token (0→128) | 393,216.0 B | **393,216.0 B** |
/// | staging **upload** | 399,376 B | **8 B** |
///
/// So arming this does **not** remove the host round-trip for `past_key_values`/`present` — the
/// readback is byte-identical in both lanes, because `bind_target_for` is called on inputs only
/// (`vk/session.rs` step 1a) and the output readback is an unconditional sum over
/// `actual_output_byte_sizes`. No configuration can decline it; only an output-side bind can.
/// What arming *does* remove is weight re-upload: 2.29 GB → 1.57 MB over five inferences. That is
/// the live argument for flipping the default, and it is a weight-residency argument, not a KV one.
///
/// `ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY=1` enables it. That is the switch the M2 work runs behind.
pub const ENV_DEVICE_MEMORY: &str = "ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY";

fn device_memory_enabled() -> bool {
    std::env::var(ENV_DEVICE_MEMORY).is_ok_and(|v| {
        matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

/// Advertise device memory for one `OrtEpDevice`.
///
/// This is what puts the allocator in ORT's path: ORT calls `CreateAllocator` only for an
/// `OrtMemoryInfo` an EP device has claimed.
///
/// # Safety
/// `api` and `ep_api` must be live; `ep_device` must be a handle from `CreateEpDevice`.
unsafe fn advertise_device_memory(
    api: *const ort::OrtApi,
    ep_api: *const ort::OrtEpApi,
    ep_device: *mut ort::OrtEpDevice,
    info: &crate::engine::DeviceInfo,
) {
    // SAFETY: `api`/`ep_api` are live for the duration of this function.
    unsafe {
        let (Some(create_mi), Some(add_alloc)) = (
            (*api).CreateMemoryInfo_V2,
            (*ep_api).EpDevice_AddAllocatorInfo,
        ) else {
            log::info!(
                "VulkanExecutionProvider: this ORT build has no CreateMemoryInfo_V2 / \
                 EpDevice_AddAllocatorInfo, so no device allocator is advertised for device #{}. \
                 The EP still works — subgraph I/O stays in host memory, as it did before M2.",
                info.index
            );
            return;
        };

        let Ok(name) = CString::new(memory_info_name(info.index)) else {
            return;
        };
        let mut mi: *mut ort::OrtMemoryInfo = ptr::null_mut();
        let status = create_mi(
            name.as_ptr(),
            ort::OrtMemoryInfoDeviceType_OrtMemoryInfoDeviceType_GPU,
            info.vendor_id,
            info.index as i32,
            ort::OrtDeviceMemoryType_OrtDeviceMemoryType_DEFAULT,
            // Our handles are page-aligned, so promising 4096 is a claim we actually keep. A
            // smaller promise would be true but would let ORT sub-divide a block on a boundary our
            // range lookup cannot attribute to a distinct allocation.
            crate::allocator::HANDLE_ALIGNMENT,
            ort::OrtAllocatorType_OrtDeviceAllocator,
            &mut mi,
        );
        if !status.is_null() || mi.is_null() {
            let msg = sys::status_message(api, status);
            sys::release_status(api, status);
            log::warn!(
                "VulkanExecutionProvider: could not create OrtMemoryInfo for device #{}: {msg}. \
                 No device allocator will be advertised for it.",
                info.index
            );
            return;
        }

        let status = add_alloc(ep_device, mi);
        if !status.is_null() {
            let msg = sys::status_message(api, status);
            sys::release_status(api, status);
            log::warn!(
                "VulkanExecutionProvider: EpDevice_AddAllocatorInfo failed for device #{}: {msg}",
                info.index
            );
            // Only on the failure path is the handle ours to release: ORT did not take it.
            if let Some(release) = (*api).ReleaseMemoryInfo {
                release(mi);
            }
            return;
        }

        // `mi` is deliberately NOT released.
        //
        // `EpDevice_AddAllocatorInfo` is annotated `_In_` and reads like a copy, but the
        // `OrtEpDevice` stores the pointer: ORT dereferences it after `GetSupportedDevices`
        // returns, while it is finishing library registration. Releasing here produced an access
        // violation inside `register_execution_provider_library`, reproduced locally and bisected
        // to this exact call — and it is the same fault signature CI has been showing.
        //
        // So the memory info leaks by design. It is one small object per device per registration,
        // for the life of the process, and ORT has no API to hand it back. A leak whose size is
        // bounded by the device count is the correct trade against a dangling pointer the host
        // will dereference; this is the same deleter-lifetime class as the ORT 1.28 plugin-EP fix
        // we pinned for.
        //
        // If a future ORT documents that it copies, the fix is to release here again — but the
        // burden of proof is on the documentation, because this direction fails loudly and the
        // other direction fails silently and only under memory pressure.
        record_advertised_device(info.vendor_id, info.index);
        log::info!(
            "VulkanExecutionProvider: advertised device memory `{}` for device #{} — ORT may now \
             allocate through this EP",
            memory_info_name(info.index),
            info.index
        );
    }
}

/// The `OrtMemoryInfo` name for a device index.
///
/// `create_allocator` matches on this rather than on the device id alone, because ORT may hand us
/// a memory info we never advertised (for a different EP, or for host memory) and answering "yes,
/// that's mine" to one of those would put our handles where real pointers are expected.
pub(crate) fn memory_info_name(index: usize) -> String {
    format!("VulkanExecutionProvider:{index}")
}

/// RAII wrapper over an `OrtKeyValuePairs` built for EP-device metadata.
struct MetadataBuilder {
    api: *const ort::OrtApi,
    kvps: *mut ort::OrtKeyValuePairs,
}

impl MetadataBuilder {
    fn new(api: *const ort::OrtApi) -> MetadataBuilder {
        let mut kvps: *mut ort::OrtKeyValuePairs = ptr::null_mut();
        // SAFETY: `api` is live; `CreateKeyValuePairs` writes an owned handle through a valid
        // out-param and cannot fail in a way that leaves it uninitialised (we pre-null it).
        unsafe {
            if let Some(create) = (*api).CreateKeyValuePairs {
                create(&mut kvps);
            }
        }
        MetadataBuilder { api, kvps }
    }

    fn add(&self, key: &CStr, value: &str) {
        let Ok(v) = CString::new(value.replace('\0', "?")) else {
            return;
        };
        self.add_cstr(key, &v);
    }

    fn add_cstr(&self, key: &CStr, value: &CStr) {
        if self.kvps.is_null() {
            return;
        }
        // SAFETY: `self.kvps` is a live handle from `CreateKeyValuePairs`; ORT copies both
        // strings, so the borrows need only last for this call.
        unsafe {
            if let Some(add) = (*self.api).AddKeyValuePair {
                add(self.kvps, key.as_ptr(), value.as_ptr());
            }
        }
    }

    fn as_ptr(&self) -> *const ort::OrtKeyValuePairs {
        self.kvps
    }
}

impl Drop for MetadataBuilder {
    fn drop(&mut self) {
        if self.kvps.is_null() {
            return;
        }
        // SAFETY: we created this handle and are releasing it exactly once. `CreateEpDevice` copies
        // the metadata it needs, so releasing after that call is correct.
        unsafe {
            if let Some(release) = (*self.api).ReleaseKeyValuePairs {
                release(self.kvps);
            }
        }
        self.kvps = ptr::null_mut();
    }
}

// -------------------------------------------------------------------------------------------
// EP lifecycle
// -------------------------------------------------------------------------------------------

/// The `vkEnumeratePhysicalDevices` index of the device ORT bound for this session.
///
/// **Why this is read at all.** Until now the compute session chose its own physical device from
/// the selector while ORT independently bound whichever `OrtEpDevice` its policy preferred, and the
/// device-memory provider is keyed by *ORT's* choice. When the two disagreed the provider stood up
/// a second `VkDevice` and the run reported `alloc_device_frame = SPLIT-DEVICE` — §6.5's invariant
/// (exactly one `VkDevice` per physical device per EP instance) violated by construction. This is
/// the value that makes the session follow ORT instead of guessing.
///
/// Read from the `vulkan.device_index` key of the EP metadata we ourselves attached in
/// `GetSupportedDevices`, so it is exact rather than correlated. Returns `None` when ORT passed no
/// metadata (older hosts) or the key is missing — in which case the caller falls back to the
/// selector and logs that it did.
///
/// # Safety
/// `api` must be live. `ep_metadata` must be either null or an array of `num_devices` pointers,
/// each null or a live `OrtKeyValuePairs`.
unsafe fn bound_physical_index(
    api: *const ort::OrtApi,
    ep_metadata: *const *const ort::OrtKeyValuePairs,
    num_devices: usize,
) -> Option<usize> {
    if ep_metadata.is_null() || num_devices == 0 {
        return None;
    }
    // SAFETY: ORT guarantees `num_devices` valid entries; we only read the first because
    // `create_ep_impl` refuses any other count.
    let kvps = unsafe { *ep_metadata };
    if kvps.is_null() {
        return None;
    }
    // SAFETY: `api` is live; `GetKeyValue` returns a borrowed C string owned by `kvps`, valid for
    // as long as `kvps` is — which is at least this call.
    let raw = unsafe { (*api).GetKeyValue?(kvps, c"vulkan.device_index".as_ptr()) };
    if raw.is_null() {
        return None;
    }
    // SAFETY: non-null, NUL-terminated, owned by `kvps` and not freed during this call.
    unsafe { CStr::from_ptr(raw) }
        .to_str()
        .ok()?
        .trim()
        .parse::<usize>()
        .ok()
}

unsafe extern "C" fn create_ep(
    p: *mut ort::OrtEpFactory,
    devices: *const *const ort::OrtHardwareDevice,
    ep_metadata: *const *const ort::OrtKeyValuePairs,
    num_devices: usize,
    session_options: *const ort::OrtSessionOptions,
    logger: *const ort::OrtLogger,
    ep: *mut *mut ort::OrtEp,
) -> ort::OrtStatusPtr {
    // SAFETY: `p` is our factory pointer.
    let api = unsafe { this(p).ort_api };
    // SAFETY: `api` is live.
    unsafe {
        crate::guard_ffi_status(api, "CreateEp", || {
            create_ep_impl(
                p,
                devices,
                ep_metadata,
                num_devices,
                session_options,
                logger,
                ep,
            )
        })
    }
}

/// # Safety
/// All pointers must be the live ones ORT passed to `CreateEp`.
unsafe fn create_ep_impl(
    p: *mut ort::OrtEpFactory,
    _devices: *const *const ort::OrtHardwareDevice,
    ep_metadata: *const *const ort::OrtKeyValuePairs,
    num_devices: usize,
    session_options: *const ort::OrtSessionOptions,
    logger: *const ort::OrtLogger,
    ep: *mut *mut ort::OrtEp,
) -> ort::OrtStatusPtr {
    // SAFETY: `p` is our factory pointer.
    let factory = unsafe { this(p) };
    let api = factory.ort_api;

    if ep.is_null() {
        // SAFETY: `api` is live.
        return unsafe {
            sys::make_status(
                api,
                ort::OrtErrorCode_ORT_INVALID_ARGUMENT,
                "CreateEp received a null out-parameter",
            )
        };
    }
    // SAFETY: valid out-param slot; null it first so an error path never leaves it undefined.
    unsafe { *ep = ptr::null_mut() };

    if num_devices != 1 {
        // SAFETY: `api` is live.
        return unsafe {
            sys::make_status(
                api,
                ort::OrtErrorCode_ORT_INVALID_ARGUMENT,
                &format!(
                    "VulkanExecutionProvider binds exactly one device per session, but ORT \
                     requested {num_devices}"
                ),
            )
        };
    }

    // Prefer the session logger over the process-default one for this session's messages.
    // SAFETY: `api` and `logger` are live for at least the session's lifetime; the attachment is
    // dropped in `ReleaseEpFactory`, which ORT calls after every session is gone.
    unsafe { crate::logging::attach_ort_logger(api, logger) };

    // §6.5: which physical device did ORT bind for this session? The compute session must open
    // that one and no other — the device-memory provider is keyed by it.
    // SAFETY: `api` is live; `ep_metadata` is ORT's array of `num_devices` entries (possibly null).
    let bound = unsafe { bound_physical_index(api, ep_metadata, num_devices) };
    match bound {
        Some(idx) => log::info!(
            "CreateEp: ORT bound Vulkan physical device index {idx} for this session; the compute \
             session will open that device (§6.5, one VkDevice per physical device per EP)."
        ),
        None => log::warn!(
            "CreateEp: ORT passed no `vulkan.device_index` metadata, so the device it bound is \
             unknown here. Falling back to the ONNXRUNTIME_EP_VULKAN_DEVICE / ep.device_index \
             selector; if it names a device other than the one ORT bound, the device-memory \
             provider will report SPLIT-DEVICE."
        ),
    }

    // SAFETY: `api`/`ep_api` are live; `session_options` may be null, which `new` handles.
    let vulkan_ep = unsafe {
        VulkanEp::new(
            factory.ort_api,
            factory.ep_api,
            factory.abi_version,
            &factory.name,
            session_options,
            bound,
        )
    };
    // SAFETY: valid out-param slot; ownership passes to ORT, which returns it via `ReleaseEp`.
    unsafe { *ep = vulkan_ep.into_raw() };
    ptr::null_mut()
}

unsafe extern "C" fn release_ep(_p: *mut ort::OrtEpFactory, ep: *mut ort::OrtEp) {
    // The session logger `CreateEp` attached dies with the session, so unwind to the factory's
    // process-default logger *before* dropping the EP. Leaving it attached would leave a dangling
    // `OrtLogger*` that the next log record would forward into.
    crate::logging::restore_default_ort_logger();
    // SAFETY: ORT hands back exactly the pointer `CreateEp` produced, exactly once.
    unsafe { VulkanEp::release(ep) };
}

// -------------------------------------------------------------------------------------------
// Memory-related slots — deliberately empty until M2
// -------------------------------------------------------------------------------------------
//
// M0/M1 advertise no device allocator and no data transfer: subgraph I/O stays in host memory and
// staging happens inside Compute (DESIGN.md §6.3). These are implemented rather than left `None`
// because ORT calls some of them during library registration, and "the EP has no allocator" is a
// different answer from "the EP has no opinion about allocators".
//
// M2 replaces them with `allocator.rs` (OrtAllocator over device memory) and `transfer.rs`
// (OrtDataTransferImpl). See OQ-3 for how a `VkBuffer + offset` is made to look like the `void*`
// ORT's allocator API expects — the current answer is an opaque-handle registry.

// -------------------------------------------------------------------------------------------
// Device allocator
// -------------------------------------------------------------------------------------------
//
// ORT calls `CreateAllocator` for a memory info an EP device claimed in `GetSupportedDevices`
// (see `advertise_device_memory`). The handle scheme, the vtable and the lifetime contract live in
// `allocator.rs`; the `VkBuffer` behind each handle is Switch's `vk/alloc.rs`, reached through the
// opaque `engine::BufferView` token.
//
// Data transfer and sync streams remain unimplemented: an allocator that hands out handles is
// useful on its own (weight prepacking and the KV cache both need device-resident memory with a
// stable identity), whereas a data-transfer implementation without one has nothing to transfer
// into. They are advertised as absent rather than as no-op successes, because "this EP cannot copy
// between devices" is true and "it copied successfully" would not be.

/// Registries live for the process, keyed by device index, so every session on a device shares one
/// handle space.
///
/// Process-lifetime rather than per-session for two reasons. ORT may create and destroy several
/// allocators for the same device over a run, and recycling the address space between them would
/// reintroduce exactly the stale-handle aliasing the quarantine exists to prevent. And the
/// high-water number that Mouse's P6 assertion reads is only meaningful across a whole run.
static REGISTRIES: std::sync::LazyLock<
    Mutex<std::collections::HashMap<usize, Arc<crate::allocator::HandleRegistry>>>,
> = std::sync::LazyLock::new(|| Mutex::new(std::collections::HashMap::new()));

/// The registry for `device_index`, creating it on first use.
pub(crate) fn registry_for_device(
    device_index: usize,
) -> Option<Arc<crate::allocator::HandleRegistry>> {
    let mut map = REGISTRIES.lock().ok()?;
    if let Some(r) = map.get(&device_index) {
        return Some(Arc::clone(r));
    }
    let r = crate::allocator::HandleRegistry::new()?;
    r.set_device_index(device_index);
    map.insert(device_index, Arc::clone(&r));
    Some(r)
}

/// Which devices we advertised memory for, as `(vendor_id, device_id)`.
///
/// The data transfer needs this because an `OrtMemoryDevice` will only tell it a vendor and a
/// device id — never the index we key registries by. Recorded at advertisement time, which always
/// precedes any copy.
static ADVERTISED_DEVICES: std::sync::LazyLock<Mutex<Vec<(u32, u32, usize)>>> =
    std::sync::LazyLock::new(|| Mutex::new(Vec::new()));

fn record_advertised_device(vendor_id: u32, device_index: usize) {
    if let Ok(mut v) = ADVERTISED_DEVICES.lock() {
        let key = (vendor_id, device_index as u32, device_index);
        if !v.contains(&key) {
            v.push(key);
        }
    }
}

/// Every registry that exists, keyed by `(vendor_id, device_id)`.
///
/// Used by the data-transfer path and by [`crate::transfer::host_backing_for`], which need to
/// recognise a handle without knowing which device produced it. Cheap: at most one entry per GPU.
pub(crate) fn all_registries()
-> std::collections::HashMap<(u32, u32), Arc<crate::allocator::HandleRegistry>> {
    let mut out = std::collections::HashMap::new();
    let advertised: Vec<(u32, u32, usize)> = match ADVERTISED_DEVICES.lock() {
        Ok(v) => v.clone(),
        Err(e) => e.into_inner().clone(),
    };
    let Ok(map) = REGISTRIES.lock() else {
        return out;
    };
    for (vendor_id, device_id, index) in advertised {
        if let Some(r) = map.get(&index) {
            out.insert((vendor_id, device_id), Arc::clone(r));
        }
    }
    out
}

unsafe extern "C" fn create_allocator(
    p: *mut ort::OrtEpFactory,
    memory_info: *const ort::OrtMemoryInfo,
    _options: *const ort::OrtKeyValuePairs,
    allocator: *mut *mut ort::OrtAllocator,
) -> ort::OrtStatusPtr {
    // Null the out-param before anything fallible: on every early return below ORT must read a
    // definite null rather than whatever was in the slot.
    if !allocator.is_null() {
        // SAFETY: valid out-param slot supplied by ORT.
        unsafe { *allocator = ptr::null_mut() };
    }
    if p.is_null() {
        return ptr::null_mut();
    }
    // SAFETY: `p` is the factory pointer ORT received from `CreateEpFactories`.
    let api = unsafe { this(p).ort_api };

    // SAFETY: `api` is live; the guard converts any panic below into a status rather than
    // unwinding into ORT's C++.
    unsafe {
        crate::guard_ffi_status(api, "OrtEpFactory::CreateAllocator", || {
            if allocator.is_null() || memory_info.is_null() {
                // Not an error: ORT probes with a null out-param in some paths, and "no allocator"
                // is a legal answer.
                return ptr::null_mut();
            }
            let Some(device_index) = device_index_of(api, memory_info) else {
                // A memory info we did not advertise. Declining is the only safe answer: claiming
                // it would put opaque handles where the requester expects readable memory.
                log::debug!(
                    "VulkanExecutionProvider: CreateAllocator called for a memory info this EP did \
                     not advertise; declining."
                );
                return ptr::null_mut();
            };
            let Some(registry) = registry_for_device(device_index) else {
                log::warn!(
                    "VulkanExecutionProvider: could not reserve handle address space for device \
                     #{device_index}; reporting no device allocator. The EP still works with host \
                     memory."
                );
                return ptr::null_mut();
            };

            // Our own memory info for the allocator to report from `Info()`. ORT's contract is
            // that the allocator owns what it returns there, so we make a fresh one rather than
            // aliasing the caller's.
            let Some(create_mi) = (*api).CreateMemoryInfo_V2 else {
                return ptr::null_mut();
            };
            let Ok(name) = CString::new(memory_info_name(device_index)) else {
                return ptr::null_mut();
            };
            let mut mi: *mut ort::OrtMemoryInfo = ptr::null_mut();
            let status = create_mi(
                name.as_ptr(),
                ort::OrtMemoryInfoDeviceType_OrtMemoryInfoDeviceType_GPU,
                vendor_id_of(api, memory_info).unwrap_or(0),
                device_index as i32,
                ort::OrtDeviceMemoryType_OrtDeviceMemoryType_DEFAULT,
                crate::allocator::HANDLE_ALIGNMENT,
                ort::OrtAllocatorType_OrtDeviceAllocator,
                &mut mi,
            );
            if !status.is_null() {
                return status;
            }

            // SAFETY: `mi` is an owned handle being transferred to the allocator, which releases
            // it in `VulkanAllocator::release`; `api` is ORT's process-lifetime table.
            let a = crate::allocator::VulkanAllocator::new(registry, mi, api);
            // SAFETY: checked non-null above.
            *allocator = a;
            log::info!(
                "VulkanExecutionProvider: created a device allocator for device #{device_index}. \
                 Handles are reserved, inaccessible virtual addresses — interior pointers resolve \
                 by range, and a dereference faults immediately by design."
            );
            ptr::null_mut()
        })
    }
}

/// Recover the device index from an `OrtMemoryInfo`, but only if we advertised it.
///
/// # Safety
/// `api` must be live; `mi` must be a valid memory info.
unsafe fn device_index_of(api: *const ort::OrtApi, mi: *const ort::OrtMemoryInfo) -> Option<usize> {
    // SAFETY: `api` and `mi` are live per this function's contract; `MemoryInfoGetName` yields a
    // pointer valid for as long as `mi` is, and we copy out of it before returning.
    unsafe {
        let get_name = (*api).MemoryInfoGetName?;
        let mut name: *const std::os::raw::c_char = ptr::null();
        let status = get_name(mi, &mut name);
        if !status.is_null() {
            sys::release_status(api, status);
            return None;
        }
        if name.is_null() {
            return None;
        }
        let name = CStr::from_ptr(name).to_string_lossy().into_owned();
        // Matching on the exact name we advertised, not on a prefix and not on the device id: a
        // memory info that merely *looks* like ours is one we must decline.
        let get_id = (*api).MemoryInfoGetId?;
        let mut id: std::os::raw::c_int = 0;
        let status = get_id(mi, &mut id);
        if !status.is_null() {
            sys::release_status(api, status);
            return None;
        }
        let index = usize::try_from(id).ok()?;
        if name == memory_info_name(index) {
            Some(index)
        } else {
            None
        }
    }
}

/// # Safety
/// `api` must be live; `mi` must be a valid memory info.
unsafe fn vendor_id_of(api: *const ort::OrtApi, mi: *const ort::OrtMemoryInfo) -> Option<u32> {
    // SAFETY: `api`/`mi` are live per the contract.
    unsafe {
        let f = (*api).MemoryInfoGetVendorId?;
        Some(f(mi))
    }
}

unsafe extern "C" fn release_allocator(
    _p: *mut ort::OrtEpFactory,
    allocator: *mut ort::OrtAllocator,
) {
    if allocator.is_null() {
        return;
    }
    // SAFETY: ORT hands back exactly the pointer `create_allocator` produced, exactly once.
    // `release` re-checks our marker before interpreting it, so a foreign pointer is logged and
    // ignored rather than freed as ours.
    unsafe { crate::allocator::VulkanAllocator::release(allocator) };
}

/// Hand ORT the data transfer that moves bytes to and from our device handles.
///
/// Advertising a device allocator obliges us to provide this: without it ORT fails every `Run`
/// with "There's no data transfer registered for copying tensors from …". So the two are wired
/// together — if no device was advertised we return null, which is the legal "I have no device
/// memory" answer and keeps a host-memory-only build exactly as it was.
unsafe extern "C" fn create_data_transfer(
    p: *mut ort::OrtEpFactory,
    data_transfer: *mut *mut ort::OrtDataTransferImpl,
) -> ort::OrtStatusPtr {
    // Null the out-param before anything fallible: on every early return ORT must read a definite
    // null rather than whatever was in the slot.
    if !data_transfer.is_null() {
        // SAFETY: valid out-param slot supplied by ORT.
        unsafe { *data_transfer = ptr::null_mut() };
    }
    if p.is_null() || data_transfer.is_null() {
        return ptr::null_mut();
    }
    // SAFETY: `p` is the factory pointer ORT received from `CreateEpFactories`.
    let f = unsafe { this(p) };
    let (api, ep_api, abi) = (f.ort_api, f.ep_api, f.abi_version);

    // SAFETY: `api` is live; the guard converts any panic below into a status rather than
    // unwinding into ORT's C++.
    unsafe {
        crate::guard_ffi_status(api, "OrtEpFactory::CreateDataTransfer", || {
            create_data_transfer_impl(api, ep_api, abi, data_transfer)
        })
    }
}

/// The body of `CreateDataTransfer`, outside the panic guard so its `unsafe` blocks stay
/// individually justified.
///
/// # Safety
/// `api`/`ep_api` must be the live negotiated tables and `data_transfer` a writable out-param slot.
unsafe fn create_data_transfer_impl(
    api: *const ort::OrtApi,
    ep_api: *const ort::OrtEpApi,
    abi: u32,
    data_transfer: *mut *mut ort::OrtDataTransferImpl,
) -> ort::OrtStatusPtr {
    let advertised: Vec<(u32, u32, usize)> = match ADVERTISED_DEVICES.lock() {
        Ok(v) => v.clone(),
        Err(e) => e.into_inner().clone(),
    };
    if advertised.is_empty() {
        // No device memory was advertised, so nothing can ever be allocated in it and a data
        // transfer would never be called. Null is the contract's way to say that.
        return ptr::null_mut();
    }

    let mut registries = std::collections::HashMap::new();
    for (vendor_id, device_id, index) in advertised {
        if let Some(r) = registry_for_device(index) {
            registries.insert((vendor_id, device_id), r);
        }
    }
    if registries.is_empty() {
        log::warn!(
            "VulkanExecutionProvider: device memory was advertised but no handle registry could \
             be created, so no data transfer is offered. Sessions that place a tensor in device \
             memory will fail at Run with a clear ORT message rather than copying into unusable \
             addresses."
        );
        return ptr::null_mut();
    }

    let n = registries.len();
    // SAFETY: `api`/`ep_api` are the live negotiated tables and outlive the process's use of this
    // object; `abi` is the negotiated version, which is what bounds how far into this vtable ORT
    // may read.
    let dt = unsafe { crate::transfer::VulkanDataTransfer::new(api, ep_api, abi, registries) };
    // SAFETY: valid out-param slot supplied by ORT; ownership passes to ORT, which returns it
    // through `OrtDataTransferImpl::Release`.
    unsafe { *data_transfer = dt.cast() };
    log::info!("VulkanExecutionProvider: data transfer created for {n} device memory space(s)");
    ptr::null_mut()
}

unsafe extern "C" fn create_sync_stream_for_device(
    _p: *mut ort::OrtEpFactory,
    _memory_device: *const ort::OrtMemoryDevice,
    _options: *const ort::OrtKeyValuePairs,
    stream: *mut *mut ort::OrtSyncStreamImpl,
) -> ort::OrtStatusPtr {
    if !stream.is_null() {
        // SAFETY: valid out-param slot supplied by ORT.
        unsafe { *stream = ptr::null_mut() };
    }
    ptr::null_mut()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ep_identity_is_frozen() {
        assert_eq!(EP_NAME.to_str().unwrap(), "VulkanExecutionProvider");
        assert_eq!(EP_VENDOR.to_str().unwrap(), "onnxruntime-ep-vulkan");
    }

    #[test]
    fn version_encodes_the_ort_api_version() {
        let v = env!("CARGO_PKG_VERSION");
        assert!(
            v.starts_with(&format!("0.{}.", sys::ORT_API_VERSION_EXPECTED)),
            "crate version {v} must be 0.<ORT_API_VERSION>.<patch>"
        );
    }

    #[test]
    fn correlation_strategy_labels() {
        assert_eq!(
            correlation_strategy(true).to_str().unwrap(),
            "vendor_id+device_id"
        );
        assert_eq!(
            correlation_strategy(false).to_str().unwrap(),
            "device_type_fallback"
        );
    }

    #[test]
    fn releasing_a_null_factory_is_a_noop() {
        // SAFETY: null is explicitly allowed and must not be dereferenced.
        unsafe { VulkanEpFactory::release(ptr::null_mut()) };
    }
}
