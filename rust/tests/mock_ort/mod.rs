//! A **mock ONNX Runtime host** that drives the plugin's registration path end to end.
//!
//! # Why this test exists
//!
//! On 2026-07-29 the plugin was loaded by a real ONNX Runtime for the first time and the host
//! process died with an access violation inside
//! `ort.register_execution_provider_library(...)`. At that moment the crate had 268 passing tests,
//! a clean `cargo clippy -D warnings`, and a green `cargo ci`. None of it touched the code that
//! crashed, because **nothing we ran locally ever called our exported entry points the way ORT
//! calls them.**
//!
//! The bug was that `forward_to_ort` passed `NULL` for `Logger_LogMessage`'s `file_path`. ORT
//! annotates that parameter `_In_z_`, not `_In_opt_z_`, and on Windows the implementation does
//! `onnxruntime::ToUTF8String(file_path)` — constructing a `std::wstring` from the pointer, which
//! is an access violation on `NULL`. Our own code was perfectly happy: nothing on our side ever
//! dereferenced it.
//!
//! So this file is not a unit test of any one function. It is a **host**: a hand-built `OrtApi`,
//! `OrtEpApi` and `OrtApiBase` populated with implementations that *check ORT's contracts the way
//! ORT's own C++ would* — and then crash the test with a clear message instead of the process with
//! an access violation. Every callback here asserts the annotations in
//! `onnxruntime_c_api.h` / `onnxruntime_ep_c_api.h`:
//!
//! * `_In_` / `_In_z_` pointers are non-null, and `_In_z_` strings are NUL-terminated and
//!   decodable at the platform's `ORTCHAR_T` width (UTF-16 on Windows, UTF-8 elsewhere).
//! * `_Outptr_` out-parameters are written before a success return.
//! * every `OrtStatus` we hand out is released exactly once.
//!
//! # What it deliberately does *not* do
//!
//! It does not link ONNX Runtime, so it cannot prove that *ONNX Runtime's own* implementation is
//! happy with us — only that we honour the contracts its headers document. CI's Python lane
//! remains the only thing that proves a real ORT can load and drive the plugin.
//!
//! # Two drivers share this host
//!
//! * `tests/host_registration.rs` calls the exported functions through the **rlib**. Fast, and it
//!   can force a log record through the bridge because it shares the plugin's `log` crate.
//! * `tests/cdylib_load.rs` `dlopen`s the built **cdylib** and resolves the entry points by name,
//!   which is what ORT actually does. That one also catches packaging faults — a missing export,
//!   a wrong crate-type, an unresolvable dependent DLL — which the rlib path cannot see.

#![allow(clippy::undocumented_unsafe_blocks)]
// This host is shared by two test binaries and each one exercises a different subset of it (only
// the rlib driver can use `LogProbe::Shared`, only the cdylib driver `LogProbe::Foreign`), so
// per-binary dead-code analysis is not a useful signal here.
#![allow(dead_code)]

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::ptr;
use std::sync::Mutex;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

use onnxruntime_vulkan_ep::sys::ort;

/// The `CreateEpFactories` entry point, as ORT resolves it by name.
pub type CreateEpFactoriesFn = unsafe extern "C" fn(
    *const c_char,
    *const ort::OrtApiBase,
    *const ort::OrtLogger,
    *mut *mut ort::OrtEpFactory,
    usize,
    *mut usize,
) -> ort::OrtStatusPtr;

/// The `ReleaseEpFactory` entry point, as ORT resolves it by name.
pub type ReleaseEpFactoryFn = unsafe extern "C" fn(*mut ort::OrtEpFactory) -> ort::OrtStatusPtr;

/// How the driver can get a log record to travel through the bridge.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum LogProbe {
    /// The driver shares the plugin's `log` crate (rlib): it can emit records directly.
    Shared,
    /// The driver is outside the plugin's address-space island (cdylib): only records the plugin
    /// emits itself can be observed, so the driver must have enabled a level that guarantees one.
    Foreign,
}

// ---------------------------------------------------------------------------------------------
// Mock-host bookkeeping
// ---------------------------------------------------------------------------------------------

/// Contract violations observed by the mock host. Non-empty means ORT would have been within its
/// rights to crash, corrupt memory, or throw.
static VIOLATIONS: Mutex<Vec<String>> = Mutex::new(Vec::new());
/// Every log record the plugin forwarded, as `(severity, message, file_path, line, func)`.
static RECORDS: Mutex<Vec<Record>> = Mutex::new(Vec::new());

/// One log record as the mock host received it.
struct Record {
    #[allow(dead_code)]
    severity: ort::OrtLoggingLevel,
    message: String,
    file: String,
    line: c_int,
    func: String,
}

static STATUSES_CREATED: AtomicUsize = AtomicUsize::new(0);
static STATUSES_RELEASED: AtomicUsize = AtomicUsize::new(0);
static KVPS_CREATED: AtomicUsize = AtomicUsize::new(0);
static KVPS_RELEASED: AtomicUsize = AtomicUsize::new(0);
static EP_DEVICES_CREATED: AtomicUsize = AtomicUsize::new(0);

static API: AtomicPtr<ort::OrtApi> = AtomicPtr::new(ptr::null_mut());
static EP_API: AtomicPtr<ort::OrtEpApi> = AtomicPtr::new(ptr::null_mut());

fn violation(what: impl Into<String>) {
    let msg = what.into();
    eprintln!("[mock-ort] CONTRACT VIOLATION: {msg}");
    if let Ok(mut v) = VIOLATIONS.lock() {
        v.push(msg);
    }
}

// ---------------------------------------------------------------------------------------------
// Contract checkers — these are the whole point of the file
// ---------------------------------------------------------------------------------------------

/// Validate an `_In_z_ const char*`: non-null and NUL-terminated within a sane bound.
fn check_in_z(p: *const c_char, who: &str, param: &str) -> String {
    if p.is_null() {
        violation(format!("{who}: `{param}` is annotated _In_z_ but was NULL"));
        return String::new();
    }
    // SAFETY: non-null; `CStr::from_ptr` requires a NUL terminator, which is exactly the
    // `_In_z_` contract we are here to verify. A missing one is UB in ORT too — this is a
    // faithful mock, not a safer one.
    let s = unsafe { CStr::from_ptr(p) };
    if s.to_bytes().len() > 64 * 1024 {
        violation(format!(
            "{who}: `{param}` is implausibly long; probably not NUL-terminated"
        ));
    }
    s.to_string_lossy().into_owned()
}

/// Validate an `_In_z_ const ORTCHAR_T*` — the parameter that killed the first real load.
///
/// This is what ORT does with it on Windows:
/// ```cpp
/// const std::string file_path_str = onnxruntime::ToUTF8String(file_path);  // std::wstring(NULL)
/// ```
/// and on Unix:
/// ```cpp
/// onnxruntime::CodeLocation location(file_path, line_number, func_name);   // std::string{NULL}
/// ```
/// Both are undefined behaviour on NULL. So the mock treats NULL as a hard violation rather than
/// tolerating it — tolerating it here is precisely how the real crash escaped review.
fn check_in_z_ortchar(p: *const ort::wchar_t, who: &str, param: &str) -> String {
    if p.is_null() {
        violation(format!(
            "{who}: `{param}` is annotated _In_z_ but was NULL — ONNX Runtime dereferences this \
             unconditionally (ToUTF8String / CodeLocation) and would access-violate here"
        ));
        return String::new();
    }
    let mut units: Vec<u16> = Vec::new();
    let mut i = 0usize;
    loop {
        if i > 64 * 1024 {
            violation(format!(
                "{who}: `{param}` is not NUL-terminated within 64Ki units"
            ));
            break;
        }
        // SAFETY: reading successive `ORTCHAR_T` units until the NUL terminator is exactly what
        // ORT does; the `_In_z_` contract is what makes it sound.
        let u = unsafe { *p.add(i) };
        if u == 0 {
            break;
        }
        units.push(u);
        i += 1;
    }
    if units.is_empty() {
        violation(format!(
            "{who}: `{param}` is an empty string; ORT expects a real source path"
        ));
    }
    String::from_utf16_lossy(&units)
}

// ---------------------------------------------------------------------------------------------
// OrtApi implementations
// ---------------------------------------------------------------------------------------------

struct MockStatus {
    code: ort::OrtErrorCode,
    message: CString,
}

unsafe extern "C" fn create_status(
    code: ort::OrtErrorCode,
    msg: *const c_char,
) -> *mut ort::OrtStatus {
    let message = check_in_z(msg, "CreateStatus", "msg");
    STATUSES_CREATED.fetch_add(1, Ordering::SeqCst);
    let boxed = Box::new(MockStatus {
        code,
        message: CString::new(message).unwrap_or_default(),
    });
    Box::into_raw(boxed).cast::<ort::OrtStatus>()
}

unsafe extern "C" fn release_status(p: *mut ort::OrtStatus) {
    if p.is_null() {
        violation("ReleaseStatus: called with NULL (the EP should null-check before releasing)");
        return;
    }
    STATUSES_RELEASED.fetch_add(1, Ordering::SeqCst);
    // SAFETY: `p` was produced by `create_status` above and is released exactly once.
    drop(unsafe { Box::from_raw(p.cast::<MockStatus>()) });
}

unsafe extern "C" fn get_error_code(p: *const ort::OrtStatus) -> ort::OrtErrorCode {
    if p.is_null() {
        return ort::OrtErrorCode_ORT_OK;
    }
    // SAFETY: `p` is one of ours.
    unsafe { (*p.cast::<MockStatus>()).code }
}

unsafe extern "C" fn get_error_message(p: *const ort::OrtStatus) -> *const c_char {
    if p.is_null() {
        return ptr::null();
    }
    // SAFETY: `p` is one of ours and outlives this borrow.
    unsafe { (*p.cast::<MockStatus>()).message.as_ptr() }
}

unsafe extern "C" fn get_ep_api() -> *const ort::OrtEpApi {
    EP_API.load(Ordering::Acquire)
}

/// The check that would have caught the crash.
unsafe extern "C" fn logger_log_message(
    logger: *const ort::OrtLogger,
    severity: ort::OrtLoggingLevel,
    message: *const c_char,
    file_path: *const ort::wchar_t,
    line_number: c_int,
    func_name: *const c_char,
) -> ort::OrtStatusPtr {
    if logger.is_null() {
        violation("Logger_LogMessage: `logger` is annotated _In_ but was NULL");
    }
    let message = check_in_z(message, "Logger_LogMessage", "message");
    let file = check_in_z_ortchar(file_path, "Logger_LogMessage", "file_path");
    let func = check_in_z(func_name, "Logger_LogMessage", "func_name");
    if line_number < 0 {
        violation(format!(
            "Logger_LogMessage: negative line_number {line_number}"
        ));
    }
    if let Ok(mut r) = RECORDS.lock() {
        r.push(Record {
            severity,
            message,
            file,
            line: line_number,
            func,
        });
    }
    ptr::null_mut()
}

type MockKvps = Vec<(String, String)>;

unsafe extern "C" fn create_key_value_pairs(out: *mut *mut ort::OrtKeyValuePairs) {
    if out.is_null() {
        violation("CreateKeyValuePairs: `out` is annotated _Outptr_ but was NULL");
        return;
    }
    KVPS_CREATED.fetch_add(1, Ordering::SeqCst);
    let boxed: Box<MockKvps> = Box::new(MockKvps::with_capacity(8));
    // SAFETY: `out` is a valid out-parameter slot.
    unsafe { *out = Box::into_raw(boxed).cast::<ort::OrtKeyValuePairs>() };
}

unsafe extern "C" fn add_key_value_pair(
    kvps: *mut ort::OrtKeyValuePairs,
    key: *const c_char,
    value: *const c_char,
) {
    if kvps.is_null() {
        violation("AddKeyValuePair: `kvps` is annotated _In_ but was NULL");
        return;
    }
    let k = check_in_z(key, "AddKeyValuePair", "key");
    let v = check_in_z(value, "AddKeyValuePair", "value");
    // SAFETY: `kvps` is one of ours and is not aliased across threads in this test.
    unsafe { (*kvps.cast::<MockKvps>()).push((k, v)) };
}

unsafe extern "C" fn release_key_value_pairs(kvps: *mut ort::OrtKeyValuePairs) {
    if kvps.is_null() {
        return;
    }
    KVPS_RELEASED.fetch_add(1, Ordering::SeqCst);
    // SAFETY: `kvps` came from `create_key_value_pairs` and is released once.
    drop(unsafe { Box::from_raw(kvps.cast::<MockKvps>()) });
}

// -- fake hardware devices -------------------------------------------------------------------

struct MockHwDevice {
    vendor_id: u32,
    device_id: u32,
    kind: ort::OrtHardwareDeviceType,
}

unsafe extern "C" fn hw_vendor_id(d: *const ort::OrtHardwareDevice) -> u32 {
    if d.is_null() {
        violation("HardwareDevice_VendorId: NULL device");
        return 0;
    }
    // SAFETY: every device pointer we hand the EP is a leaked `MockHwDevice`.
    unsafe { (*d.cast::<MockHwDevice>()).vendor_id }
}

unsafe extern "C" fn hw_device_id(d: *const ort::OrtHardwareDevice) -> u32 {
    if d.is_null() {
        violation("HardwareDevice_DeviceId: NULL device");
        return 0;
    }
    // SAFETY: as above.
    unsafe { (*d.cast::<MockHwDevice>()).device_id }
}

unsafe extern "C" fn hw_type(d: *const ort::OrtHardwareDevice) -> ort::OrtHardwareDeviceType {
    if d.is_null() {
        violation("HardwareDevice_Type: NULL device");
        return ort::OrtHardwareDeviceType_OrtHardwareDeviceType_CPU;
    }
    // SAFETY: as above.
    unsafe { (*d.cast::<MockHwDevice>()).kind }
}

unsafe extern "C" fn get_session_config_entry(
    _options: *const ort::OrtSessionOptions,
    config_key: *const c_char,
    _config_value: *mut c_char,
    size: *mut usize,
) -> ort::OrtStatusPtr {
    check_in_z(config_key, "GetSessionConfigEntry", "config_key");
    if size.is_null() {
        violation("GetSessionConfigEntry: `size` is annotated _Inout_ but was NULL");
    }
    // Mimic ORT's "key not present" answer: a real, owned status the EP must release.
    // SAFETY: a NUL-terminated literal.
    unsafe { create_status(ort::OrtErrorCode_ORT_FAIL, c"config key not found".as_ptr()) }
}

// -- OrtEpApi ---------------------------------------------------------------------------------

unsafe extern "C" fn create_ep_device(
    ep_factory: *mut ort::OrtEpFactory,
    hardware_device: *const ort::OrtHardwareDevice,
    ep_metadata: *const ort::OrtKeyValuePairs,
    _ep_options: *const ort::OrtKeyValuePairs,
    ep_device: *mut *mut ort::OrtEpDevice,
) -> ort::OrtStatusPtr {
    if ep_factory.is_null() {
        violation("CreateEpDevice: `ep_factory` is annotated _In_ but was NULL");
    }
    if hardware_device.is_null() {
        violation("CreateEpDevice: `hardware_device` is annotated _In_ but was NULL");
    }
    if ep_device.is_null() {
        violation("CreateEpDevice: `ep_device` is annotated _Outptr_ but was NULL");
        // SAFETY: a NUL-terminated literal.
        return unsafe {
            create_status(
                ort::OrtErrorCode_ORT_INVALID_ARGUMENT,
                c"null out-parameter".as_ptr(),
            )
        };
    }
    // `ep_metadata` is `_In_opt_`, so NULL is legal — but if present it must be readable.
    if !ep_metadata.is_null() {
        // SAFETY: non-null metadata is a handle we created via `create_key_value_pairs`.
        let n = unsafe { (*ep_metadata.cast::<MockKvps>()).len() };
        assert!(n > 0, "EP advertised a device with empty metadata");
    }
    EP_DEVICES_CREATED.fetch_add(1, Ordering::SeqCst);
    let leaked: *mut c_void = Box::into_raw(Box::new(1u64)).cast();
    // SAFETY: valid out-parameter slot.
    unsafe { *ep_device = leaked.cast::<ort::OrtEpDevice>() };
    ptr::null_mut()
}

// -- OrtApiBase -------------------------------------------------------------------------------

unsafe extern "C" fn get_api(version: u32) -> *const ort::OrtApi {
    // Behave like ONNX Runtime 1.28: serve every version up to our own, refuse the future.
    if version > 28 {
        return ptr::null();
    }
    API.load(Ordering::Acquire)
}

unsafe extern "C" fn get_version_string() -> *const c_char {
    c"1.28.0".as_ptr()
}

/// Build the mock host once and leak it: it stands in for ORT's process-lifetime statics.
fn build_host() -> *const ort::OrtApiBase {
    // SAFETY: `OrtApi`, `OrtEpApi` and `OrtApiBase` are `#[repr(C)]` structs of function pointers,
    // which bindgen models as `Option<fn>`; all-zero is each one's valid `None`, so a zeroed table
    // is a well-formed "nothing implemented" host that we then fill in. This is the same argument
    // the EP itself makes when it zeroes `OrtEpFactory`.
    let mut api: Box<ort::OrtApi> = Box::new(unsafe { std::mem::zeroed() });
    api.CreateStatus = Some(create_status);
    api.ReleaseStatus = Some(release_status);
    api.GetErrorCode = Some(get_error_code);
    api.GetErrorMessage = Some(get_error_message);
    api.GetEpApi = Some(get_ep_api);
    api.Logger_LogMessage = Some(logger_log_message);
    api.CreateKeyValuePairs = Some(create_key_value_pairs);
    api.AddKeyValuePair = Some(add_key_value_pair);
    api.ReleaseKeyValuePairs = Some(release_key_value_pairs);
    api.HardwareDevice_VendorId = Some(hw_vendor_id);
    api.HardwareDevice_DeviceId = Some(hw_device_id);
    api.HardwareDevice_Type = Some(hw_type);
    api.GetSessionConfigEntry = Some(get_session_config_entry);
    API.store(Box::into_raw(api), Ordering::Release);

    // SAFETY: see above.
    let mut ep_api: Box<ort::OrtEpApi> = Box::new(unsafe { std::mem::zeroed() });
    ep_api.CreateEpDevice = Some(create_ep_device);
    EP_API.store(Box::into_raw(ep_api), Ordering::Release);

    // SAFETY: see above.
    let mut base: Box<ort::OrtApiBase> = Box::new(unsafe { std::mem::zeroed() });
    base.GetApi = Some(get_api);
    base.GetVersionString = Some(get_version_string);
    Box::into_raw(base)
}

/// A fake `OrtLogger`. The plugin only ever passes it back to `Logger_LogMessage`, which is ours,
/// so its contents are irrelevant — its *address* is what must stay valid.
fn fake_logger() -> *const ort::OrtLogger {
    Box::into_raw(Box::new(1u64)).cast::<ort::OrtLogger>()
}

// ---------------------------------------------------------------------------------------------
// The scenario
// ---------------------------------------------------------------------------------------------

/// Drive one full registration lifecycle against the mock host.
///
/// One scenario, not several tests: the logging bridge is process-global state, so parallel tests
/// driving registration would interleave attachments and make the assertions meaningless.
///
/// # Safety
/// `create` and `release` must be the plugin's real entry points, and the code behind them must
/// outlive the call (for the cdylib driver, the library must stay loaded).
pub unsafe fn run_registration_scenario(
    create: CreateEpFactoriesFn,
    release: ReleaseEpFactoryFn,
    probe: LogProbe,
) {
    let base = build_host();
    let default_logger = fake_logger();

    // ---- RegisterExecutionProviderLibrary step 1: CreateEpFactories -------------------------
    let mut factories: [*mut ort::OrtEpFactory; 4] = [ptr::null_mut(); 4];
    let mut num_factories: usize = usize::MAX;
    // SAFETY: this is exactly the call ORT makes after `GetProcAddress("CreateEpFactories")`:
    // a registration name, the API base, the process-default logger, and an out-array.
    let status = unsafe {
        create(
            c"VulkanExecutionProvider".as_ptr(),
            base,
            default_logger,
            factories.as_mut_ptr(),
            factories.len(),
            &mut num_factories,
        )
    };
    assert!(
        status.is_null(),
        "CreateEpFactories returned a failure status"
    );
    assert_eq!(num_factories, 1, "expected exactly one factory");
    let factory = factories[0];
    assert!(!factory.is_null());

    // ---- step 2: ORT reads the factory's identity and version ------------------------------
    // SAFETY: `factory` is the pointer the EP just handed us; every accessor below is a slot the
    // EP is required to populate, and we assert that before calling it.
    unsafe {
        let v = (*factory).ort_version_supported;
        assert!(
            (24..=28).contains(&v),
            "factory stamped ort_version_supported={v}, outside the negotiated window"
        );
        assert_eq!(
            v, 28,
            "this host serves 28, so the EP must stamp 28 — stamping less hides slots ORT can \
             legitimately read; stamping more invites ORT to read slots we never filled"
        );

        let get_name = (*factory).GetName.expect("GetName is mandatory");
        let name = check_in_z(get_name(factory), "OrtEpFactory::GetName", "return value");
        assert_eq!(name, "VulkanExecutionProvider");

        let get_vendor = (*factory).GetVendor.expect("GetVendor is mandatory");
        let vendor = check_in_z(
            get_vendor(factory),
            "OrtEpFactory::GetVendor",
            "return value",
        );
        assert_eq!(vendor, "onnxruntime-ep-vulkan");

        let get_version = (*factory).GetVersion.expect("GetVersion is mandatory");
        let version = check_in_z(
            get_version(factory),
            "OrtEpFactory::GetVersion",
            "return value",
        );
        assert!(
            version.starts_with("0.28."),
            "crate version {version} must encode the ORT API version it targets"
        );

        let get_vendor_id = (*factory).GetVendorId.expect("GetVendorId is mandatory");
        let _ = get_vendor_id(factory);

        // A slot added after our declared floor must be gated, not blindly populated.
        assert!(
            (*factory).CreateExternalResourceImporterForDevice.is_none(),
            "the external resource importer is not implemented yet; the slot must stay None"
        );
    }

    // ---- step 3: GetSupportedDevices --------------------------------------------------------
    let cpu = Box::into_raw(Box::new(MockHwDevice {
        vendor_id: 0x8086,
        device_id: 0x0001,
        kind: ort::OrtHardwareDeviceType_OrtHardwareDeviceType_CPU,
    }))
    .cast::<ort::OrtHardwareDevice>();
    let gpu = Box::into_raw(Box::new(MockHwDevice {
        vendor_id: 0x10de,
        device_id: 0x2204,
        kind: ort::OrtHardwareDeviceType_OrtHardwareDeviceType_GPU,
    }))
    .cast::<ort::OrtHardwareDevice>();
    let hw: [*const ort::OrtHardwareDevice; 2] = [cpu, gpu];

    let mut ep_devices: [*mut ort::OrtEpDevice; 8] = [ptr::null_mut(); 8];
    let mut num_ep_devices: usize = usize::MAX;
    // SAFETY: the arrays are real stack slots sized as declared, and `factory` is live.
    let status = unsafe {
        let f = (*factory)
            .GetSupportedDevices
            .expect("GetSupportedDevices is mandatory");
        f(
            factory,
            hw.as_ptr(),
            hw.len(),
            ep_devices.as_mut_ptr(),
            ep_devices.len(),
            &mut num_ep_devices,
        )
    };
    assert!(
        status.is_null(),
        "GetSupportedDevices must never fail the host — zero usable devices is a success with a \
         warning, so that a machine with no Vulkan still creates CPU sessions"
    );
    assert_ne!(
        num_ep_devices,
        usize::MAX,
        "GetSupportedDevices left its out-parameter untouched"
    );
    assert!(num_ep_devices <= ep_devices.len());
    for d in ep_devices.iter().take(num_ep_devices) {
        assert!(!d.is_null(), "an advertised OrtEpDevice slot is NULL");
    }
    assert_eq!(
        EP_DEVICES_CREATED.load(Ordering::SeqCst),
        num_ep_devices,
        "the EP reported a different device count than it created"
    );

    // ---- step 4: a log record must survive the round trip -----------------------------------
    //
    // This is the assertion that would have caught the access violation. The plugin has an ORT
    // logger attached now, so any record at or above its level is forwarded through
    // `Logger_LogMessage` — with a `file_path` this host validates the way ORT would.
    let expected = match probe {
        LogProbe::Shared => {
            RECORDS.lock().expect("records lock").clear();
            log::warn!("mock-host round-trip probe");
            log::error!("mock-host round-trip probe (error tier)");
            2
        }
        // A `dlopen`ed plugin has its own copy of the `log` crate, so the driver cannot inject a
        // record into it. Instead it enables a level at which the plugin is guaranteed to have
        // emitted its own — the "loaded" line at the end of `CreateEpFactories` — and we check
        // what already arrived.
        LogProbe::Foreign => 1,
    };
    {
        let records = RECORDS.lock().expect("records lock");
        assert!(
            records.len() >= expected,
            "the plugin did not forward log records to the attached ORT logger; the bridge is \
             disconnected and every EP diagnostic would be invisible to the host"
        );
        for Record {
            message,
            file,
            line,
            func,
            ..
        } in records.iter()
        {
            assert!(!message.is_empty(), "forwarded an empty message");
            assert!(
                !file.is_empty(),
                "forwarded a record with an empty file_path; ORT requires a real _In_z_ string"
            );
            assert!(*line >= 0, "forwarded a negative line number");
            assert!(!func.is_empty(), "forwarded an empty func_name");
        }
    }

    // ---- step 5: CreateEp / ReleaseEp --------------------------------------------------------
    let session_logger = fake_logger();
    let mut ep: *mut ort::OrtEp = ptr::null_mut();
    let one_device: [*const ort::OrtHardwareDevice; 1] = [gpu];
    let no_metadata: [*const ort::OrtKeyValuePairs; 1] = [ptr::null()];
    // SAFETY: `factory` is live; the arrays are real stack slots; a null `OrtSessionOptions` is
    // explicitly handled by the EP.
    let status = unsafe {
        let f = (*factory).CreateEp.expect("CreateEp is mandatory");
        f(
            factory,
            one_device.as_ptr(),
            no_metadata.as_ptr(),
            1,
            ptr::null(),
            session_logger,
            &mut ep,
        )
    };
    assert!(status.is_null(), "CreateEp returned a failure status");
    assert!(!ep.is_null(), "CreateEp succeeded without writing an OrtEp");
    // SAFETY: `ep` is the pointer the EP just handed us.
    unsafe {
        let v = (*ep).ort_version_supported;
        assert_eq!(
            v, 28,
            "the OrtEp must carry the same negotiated version as its factory"
        );
    }

    // Wrong device count must be rejected with a status, not a panic and not a crash.
    let mut bad_ep: *mut ort::OrtEp = ptr::null_mut();
    // SAFETY: as above, with a deliberately invalid device count.
    let bad = unsafe {
        let f = (*factory).CreateEp.expect("CreateEp is mandatory");
        f(
            factory,
            hw.as_ptr(),
            no_metadata.as_ptr(),
            2,
            ptr::null(),
            session_logger,
            &mut bad_ep,
        )
    };
    assert!(
        !bad.is_null(),
        "CreateEp accepted two devices; it binds exactly one"
    );
    assert!(
        bad_ep.is_null(),
        "a failed CreateEp must leave its out-parameter null"
    );
    // SAFETY: `bad` is a status this thread owns.
    unsafe { release_status(bad) };

    // SAFETY: `ep` came from `CreateEp` and is released exactly once.
    unsafe {
        let f = (*factory).ReleaseEp.expect("ReleaseEp is mandatory");
        f(factory, ep);
    }

    // After the session logger is gone the bridge must still be usable — this is the
    // use-after-free that `ReleaseEp` unwinds by restoring the factory's default logger. Forward
    // a record *through the plugin* now: if the restore did not happen, the pointer below is the
    // freed session logger and the mock host sees a stale address (or the process dies, which is
    // also a perfectly clear test result).
    match probe {
        LogProbe::Shared => log::warn!("mock-host post-session probe"),
        LogProbe::Foreign => {
            // Make the plugin emit on its own account: a second `GetSupportedDevices` runs the
            // same diagnostics it ran in step 3, now with the default logger restored.
            let mut again: [*mut ort::OrtEpDevice; 8] = [ptr::null_mut(); 8];
            let mut n: usize = 0;
            // SAFETY: `factory` is still live (it is released below) and the arrays are real
            // stack slots sized as declared.
            let status = unsafe {
                let f = (*factory)
                    .GetSupportedDevices
                    .expect("GetSupportedDevices is mandatory");
                f(
                    factory,
                    hw.as_ptr(),
                    hw.len(),
                    again.as_mut_ptr(),
                    again.len(),
                    &mut n,
                )
            };
            assert!(
                status.is_null(),
                "GetSupportedDevices failed after the session was released"
            );
        }
    }

    // ---- step 6: ReleaseEpFactory ------------------------------------------------------------
    // SAFETY: `factory` came from `CreateEpFactories` and is released exactly once.
    let status = unsafe { release(factory) };
    assert!(
        status.is_null(),
        "ReleaseEpFactory returned a failure status"
    );

    // ---- the verdict -------------------------------------------------------------------------
    let violations = VIOLATIONS.lock().expect("violations lock").clone();
    assert!(
        violations.is_empty(),
        "the plugin violated {} ONNX Runtime C-API contract(s); a real ORT would have crashed or \
         corrupted memory:\n  - {}",
        violations.len(),
        violations.join("\n  - ")
    );

    assert_eq!(
        STATUSES_CREATED.load(Ordering::SeqCst),
        STATUSES_RELEASED.load(Ordering::SeqCst),
        "OrtStatus leak: the EP allocated statuses it never released (or released ones it did \
         not own)"
    );
    assert_eq!(
        KVPS_CREATED.load(Ordering::SeqCst),
        KVPS_RELEASED.load(Ordering::SeqCst),
        "OrtKeyValuePairs leak in the EP-device metadata path"
    );
}
