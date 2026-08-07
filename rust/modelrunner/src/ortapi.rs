//! A thin, RAII wrapper over the ONNX Runtime C API: enough of it to load a model, register a
//! plugin execution provider, feed tensors, run, and read the outputs back.
//!
//! WHAT THIS MODULE IS ALLOWED TO ASSUME
//! -------------------------------------
//! Only what the vendored headers say. The bindings come from `onnxruntime_vulkan_ep::sys::ort`,
//! which bindgen derived from `third_party/onnxruntime/include` -- so field *order* is correct by
//! construction and an ORT reshuffle is a compile error here rather than a wrong function pointer
//! called with the wrong arguments. Every entry point is an `Option<fn>` in that table because the
//! C struct is append-only across versions; [`Api::f`] turns a null one into a named error instead
//! of an unwrap, so running against an older host reports *which* entry point is absent.
//!
//! ERRORS, NOT PANICS
//! ------------------
//! Every ORT call returns `OrtStatusPtr`; a non-null status is converted to a [`Failure`] carrying
//! ORT's own message and error code, and the status is released on both paths. Nothing in this
//! module panics on an ORT failure, and nothing leaks a status.
//!
//! PATHS
//! -----
//! `ORTCHAR_T` is `wchar_t` on Windows and `char` everywhere else, which is the single most
//! common portability defect in ORT hosts. [`OrtString`] is the one place that difference exists,
//! and `tests::path_round_trip` pins it on both shapes.
//!
//! ENUM WIDTHS
//! -----------
//! The *second* most common one, and the one that broke this crate's Linux lane on the day it
//! merged. A C enum with no negative enumerator is `int` under MSVC and `unsigned int` under
//! GCC/Clang, so bindgen emits `ort::OrtLoggingLevel` (and 24 other ORT enums) as `i32` on
//! Windows and `u32` on Linux. Measured, not assumed: diffing the two generated `ort.rs` files
//! shows 25 of the 28 ORT enum aliases change signedness across the two targets, and only
//! `OrtAllocatorType`, `OrtMemType` and `OrtDeviceEpIncompatibilityReason` — the three that carry
//! a negative enumerator — do not.
//!
//! Two rules follow, and both are mechanised rather than remembered:
//!
//! * **Never produce a value of such an alias from a literal.** [`LogSeverity`] is the only way
//!   to name a logging level here, and its [`LogSeverity::raw`] returns the bindgen constant, so
//!   the alias's width is never spelled at any call site.
//! * **Never consume one at a fixed width.** [`widen`] takes anything that converts into `i64`,
//!   which both `i32` and `u32` do losslessly, so reading a discriminant compiles on both.
//!
//! `rust/tests/portability.rs` rule P3 enforces the same thing over this file's text, and
//! `cargo test --test portability` covers `modelrunner/` as of issue #39.

use std::ffi::{CStr, CString};
use std::path::Path;
use std::ptr;

use onnxruntime_vulkan_ep::sys::ort;

use crate::error::{Failure, Result};

/// A NUL-terminated string in ORT's platform path encoding.
#[derive(Debug)]
pub struct OrtString {
    #[cfg(target_os = "windows")]
    buf: Vec<u16>,
    #[cfg(not(target_os = "windows"))]
    buf: Vec<u8>,
}

impl OrtString {
    pub fn new(path: &Path) -> Result<Self> {
        #[cfg(target_os = "windows")]
        {
            use std::os::windows::ffi::OsStrExt;
            let mut buf: Vec<u16> = path.as_os_str().encode_wide().collect();
            if buf.contains(&0) {
                return Err(Failure::instrument(
                    "path_not_representable",
                    format!("{} contains an interior NUL", path.display()),
                ));
            }
            buf.push(0);
            Ok(Self { buf })
        }
        #[cfg(not(target_os = "windows"))]
        {
            use std::os::unix::ffi::OsStrExt;
            let mut buf: Vec<u8> = path.as_os_str().as_bytes().to_vec();
            if buf.contains(&0) {
                return Err(Failure::instrument(
                    "path_not_representable",
                    format!("{} contains an interior NUL", path.display()),
                ));
            }
            buf.push(0);
            Ok(Self { buf })
        }
    }

    #[cfg(target_os = "windows")]
    pub fn as_ptr(&self) -> *const u16 {
        self.buf.as_ptr()
    }

    #[cfg(not(target_os = "windows"))]
    pub fn as_ptr(&self) -> *const std::os::raw::c_char {
        self.buf.as_ptr() as *const std::os::raw::c_char
    }
}

pub fn cstring(text: &str) -> Result<CString> {
    CString::new(text).map_err(|_| {
        Failure::instrument(
            "string_not_representable",
            format!("{text:?} contains an interior NUL"),
        )
    })
}

/// Read a value of a width-varying ORT enum alias without spelling its width.
///
/// `ort::OrtErrorCode` is `i32` on MSVC and `u32` on GCC/Clang (see the module docs). Any
/// expression that assumes one of those — `-1` as a sentinel, `as u32`, a `match` arm typed by a
/// literal — compiles on exactly one platform. `i64` is the one width *both* aliases convert into
/// losslessly and infallibly, so this is the only reader.
///
/// The bound is deliberately `Into<i64>` and not a concrete type: it is satisfied by `i32` and by
/// `u32` and by nothing that would silently truncate, which makes "this compiles on both targets"
/// a property of the signature rather than of the machine it was compiled on. The same rule and
/// the same reasoning already live in `src/counters.rs`; this is its modelrunner-side twin.
pub fn widen<T: Into<i64>>(value: T) -> i64 {
    value.into()
}

/// A logging severity, named rather than spelled.
///
/// ORT's own `OrtLoggingLevel` is width-varying, so a literal `3` typed as that alias is an `i32`
/// on Windows and a `u32` on Linux and cannot be written portably at a call site. This enum is
/// the runner's own type — its width is ours, not the ABI's — and [`Self::raw`] is the single
/// place that converts to the ABI, by naming bindgen's constant. There is no other constructor,
/// so no call site can reintroduce the literal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogSeverity {
    Verbose,
    Info,
    Warning,
    Error,
    Fatal,
}

impl LogSeverity {
    /// The ABI value, taken from bindgen's constant so that its type is whatever this target says
    /// it is. Note there is no `as` cast here: a cast would compile on both platforms while
    /// keeping the assumption that the width is knowable in this file.
    pub fn raw(self) -> ort::OrtLoggingLevel {
        match self {
            Self::Verbose => ort::OrtLoggingLevel_ORT_LOGGING_LEVEL_VERBOSE,
            Self::Info => ort::OrtLoggingLevel_ORT_LOGGING_LEVEL_INFO,
            Self::Warning => ort::OrtLoggingLevel_ORT_LOGGING_LEVEL_WARNING,
            Self::Error => ort::OrtLoggingLevel_ORT_LOGGING_LEVEL_ERROR,
            Self::Fatal => ort::OrtLoggingLevel_ORT_LOGGING_LEVEL_FATAL,
        }
    }

    /// The value ORT's *session* option takes, which is a plain `int` on every target — a
    /// different C signature from `CreateEnv`'s, and the reason this is a separate accessor
    /// rather than a cast of [`Self::raw`].
    pub fn session_level(self) -> std::os::raw::c_int {
        match self {
            Self::Verbose => 0,
            Self::Info => 1,
            Self::Warning => 2,
            Self::Error => 3,
            Self::Fatal => 4,
        }
    }
}

/// The negotiated API table plus the error plumbing every call needs.
#[derive(Clone, Copy)]
pub struct Api {
    pub raw: &'static ort::OrtApi,
}

impl Api {
    pub fn new(raw: &'static ort::OrtApi) -> Self {
        Self { raw }
    }

    /// Resolve an `Option<fn>` entry point, naming it if the host does not serve it.
    fn f<T>(&self, entry: Option<T>, name: &'static str) -> Result<T> {
        entry.ok_or_else(|| {
            Failure::instrument(
                "ort_entry_point_absent",
                format!(
                    "this ONNX Runtime does not serve OrtApi::{name}. The plugin-EP surface is \
                     append-only, so an absent entry point means the host is older than the \
                     bindings, not that the call is optional."
                ),
            )
        })
    }

    /// Convert a returned status into a `Result`, releasing it either way.
    fn check(&self, status: ort::OrtStatusPtr, context: &str) -> Result<()> {
        if status.is_null() {
            return Ok(());
        }
        let message = match self.raw.GetErrorMessage {
            // SAFETY: `status` is non-null and owned by us until ReleaseStatus below; ORT
            // guarantees a NUL-terminated message for the lifetime of the status.
            Some(get) => unsafe { CStr::from_ptr(get(status)) }
                .to_string_lossy()
                .into_owned(),
            None => "<this ORT serves no GetErrorMessage>".to_string(),
        };
        let code = match self.raw.GetErrorCode {
            // SAFETY: as above.
            Some(get) => widen(unsafe { get(status) }),
            None => -1,
        };
        if let Some(release) = self.raw.ReleaseStatus {
            // SAFETY: `status` is non-null, was returned by an ORT call, and is not used after
            // this point.
            unsafe { release(status) };
        }
        Err(Failure::instrument(
            "ort_call_failed",
            format!("{context}: ORT error {code}: {message}"),
        ))
    }
}

pub struct Env {
    api: Api,
    pub raw: *mut ort::OrtEnv,
}

impl Env {
    pub fn new(api: Api, log_id: &str, severity: LogSeverity) -> Result<Self> {
        let create = api.f(api.raw.CreateEnv, "CreateEnv")?;
        let id = cstring(log_id)?;
        let mut raw: *mut ort::OrtEnv = ptr::null_mut();
        // SAFETY: `id` outlives the call; ORT copies the log id. `raw` is a valid out-pointer.
        let status = unsafe { create(severity.raw(), id.as_ptr(), &mut raw) };
        api.check(status, "CreateEnv")?;
        Ok(Self { api, raw })
    }

    /// `RegisterExecutionProviderLibrary`, the plugin-EP registration ORT 1.24+ exposes and the
    /// exact call `ort.register_execution_provider_library(...)` makes for the Python harness.
    pub fn register_ep_library(&self, registration_name: &str, library: &Path) -> Result<()> {
        let register = self.api.f(
            self.api.raw.RegisterExecutionProviderLibrary,
            "RegisterExecutionProviderLibrary",
        )?;
        let name = cstring(registration_name)?;
        let path = OrtString::new(library)?;
        // SAFETY: both strings are NUL-terminated and outlive the call; ORT copies what it keeps.
        let status = unsafe { register(self.raw, name.as_ptr(), path.as_ptr()) };
        self.api.check(
            status,
            &format!(
                "RegisterExecutionProviderLibrary({registration_name}, {})",
                library.display()
            ),
        )
    }

    /// Every `OrtEpDevice` this environment can see, as (index, ep name, hardware facts).
    pub fn ep_devices(&self) -> Result<Vec<EpDeviceInfo>> {
        let get = self.api.f(self.api.raw.GetEpDevices, "GetEpDevices")?;
        let mut devices: *const *const ort::OrtEpDevice = ptr::null();
        let mut count: usize = 0;
        // SAFETY: out-pointers are valid; ORT owns the returned array and it stays valid for the
        // lifetime of the environment.
        let status = unsafe { get(self.raw, &mut devices, &mut count) };
        self.api.check(status, "GetEpDevices")?;
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            // SAFETY: `i < count` and ORT populated `count` entries.
            let device = unsafe { *devices.add(i) };
            if device.is_null() {
                continue;
            }
            let ep_name = match self.api.raw.EpDevice_EpName {
                // SAFETY: ORT returns a NUL-terminated string owned by the device.
                Some(f) => unsafe { CStr::from_ptr(f(device)) }
                    .to_string_lossy()
                    .into_owned(),
                None => String::new(),
            };
            let vendor = match self.api.raw.EpDevice_EpVendor {
                // SAFETY: as above.
                Some(f) => unsafe { CStr::from_ptr(f(device)) }
                    .to_string_lossy()
                    .into_owned(),
                None => String::new(),
            };
            let hardware = match self.api.raw.EpDevice_Device {
                // SAFETY: as above; the hardware device is owned by the ep device.
                Some(f) => unsafe { f(device) },
                None => ptr::null(),
            };
            let (kind, vendor_id, device_id) = if hardware.is_null() {
                ("unknown".to_string(), 0, 0)
            } else {
                let kind = match self.api.raw.HardwareDevice_Type {
                    // SAFETY: non-null hardware device from ORT.
                    Some(f) => match unsafe { f(hardware) } {
                        ort::OrtHardwareDeviceType_OrtHardwareDeviceType_CPU => "CPU",
                        ort::OrtHardwareDeviceType_OrtHardwareDeviceType_GPU => "GPU",
                        ort::OrtHardwareDeviceType_OrtHardwareDeviceType_NPU => "NPU",
                        _ => "other",
                    },
                    None => "unknown",
                };
                let vendor_id = match self.api.raw.HardwareDevice_VendorId {
                    // SAFETY: as above.
                    Some(f) => unsafe { f(hardware) },
                    None => 0,
                };
                let device_id = match self.api.raw.HardwareDevice_DeviceId {
                    // SAFETY: as above.
                    Some(f) => unsafe { f(hardware) },
                    None => 0,
                };
                (kind.to_string(), vendor_id, device_id)
            };
            // Stable identity (issue #18): the EP publishes `vulkan.device_uuid` (always, when
            // it has a device at all) and `vulkan.device_luid` / `vulkan.device_pci` (only when
            // the driver/platform reports them) as `OrtEpDevice` metadata — see
            // `factory.rs::get_supported_devices_impl`. Reading them here is what lets a model
            // run's discovery/report attribute its evidence to the *physical* device ORT bound,
            // not just to `(vendor_id, device_id)`, which two identical GPUs would share.
            let metadata = match self.api.raw.EpDevice_EpMetadata {
                // SAFETY: non-null ep device from `GetEpDevices`; the returned `OrtKeyValuePairs`
                // is owned by the device and lives at least as long as it.
                Some(f) => unsafe { f(device) },
                None => ptr::null(),
            };
            let metadata_get = |key: &str| -> Option<String> {
                let get = self.api.raw.GetKeyValue?;
                if metadata.is_null() {
                    return None;
                }
                let key = std::ffi::CString::new(key).ok()?;
                // SAFETY: `metadata` is non-null (checked above) and owned by `device`, which is
                // live for this call; `key` is a valid NUL-terminated string for its duration.
                let v = unsafe { get(metadata, key.as_ptr()) };
                if v.is_null() {
                    None
                } else {
                    // SAFETY: ORT returns a NUL-terminated string owned by the key-value pairs.
                    Some(unsafe { CStr::from_ptr(v) }.to_string_lossy().into_owned())
                }
            };
            let uuid = metadata_get("vulkan.device_uuid");
            let luid = metadata_get("vulkan.device_luid");
            let pci = metadata_get("vulkan.device_pci");
            out.push(EpDeviceInfo {
                index: i,
                raw: device,
                ep_name,
                ep_vendor: vendor,
                hardware_type: kind,
                vendor_id,
                device_id,
                uuid,
                luid,
                pci,
            });
        }
        Ok(out)
    }
}

impl Drop for Env {
    fn drop(&mut self) {
        if let Some(release) = self.api.raw.ReleaseEnv {
            // SAFETY: `raw` came from CreateEnv, is released exactly once, and every session
            // created from it has already been dropped (declaration order in `run.rs`).
            unsafe { release(self.raw) };
        }
    }
}

#[derive(Clone)]
pub struct EpDeviceInfo {
    pub index: usize,
    pub raw: *const ort::OrtEpDevice,
    pub ep_name: String,
    pub ep_vendor: String,
    pub hardware_type: String,
    pub vendor_id: u32,
    pub device_id: u32,
    /// `vulkan.device_uuid` EP metadata (issue #18 stable identity). `None` for non-Vulkan EPs,
    /// or a Vulkan EP device seen through a host too old to advertise it.
    pub uuid: Option<String>,
    /// `vulkan.device_luid` EP metadata, only when the driver reported a valid LUID (mainly
    /// Windows/D3D-interop; expect `None` on most Linux/Android/MoltenVK drivers).
    pub luid: Option<String>,
    /// `vulkan.device_pci` EP metadata, only when `VK_EXT_pci_bus_info` is supported; expect
    /// `None` on MoltenVK and some mobile ICDs.
    pub pci: Option<String>,
}

pub struct SessionOptions {
    api: Api,
    pub raw: *mut ort::OrtSessionOptions,
}

impl SessionOptions {
    pub fn new(api: Api) -> Result<Self> {
        let create = api.f(api.raw.CreateSessionOptions, "CreateSessionOptions")?;
        let mut raw: *mut ort::OrtSessionOptions = ptr::null_mut();
        // SAFETY: valid out-pointer.
        let status = unsafe { create(&mut raw) };
        api.check(status, "CreateSessionOptions")?;
        Ok(Self { api, raw })
    }

    /// Matches the Python harness's `so.log_severity_level = 3`.
    ///
    /// `SetSessionLogSeverityLevel` takes a plain `c_int` on every target — it is not typed as
    /// `OrtLoggingLevel` in the C API — so this one *is* portable at a fixed width. The argument
    /// is still a [`LogSeverity`] rather than a bare integer, so that the session and the
    /// environment cannot drift apart by a typo.
    pub fn set_log_severity(&self, level: LogSeverity) -> Result<()> {
        let set = self.api.f(
            self.api.raw.SetSessionLogSeverityLevel,
            "SetSessionLogSeverityLevel",
        )?;
        // SAFETY: `raw` is a live session-options handle.
        let status = unsafe { set(self.raw, level.session_level()) };
        self.api.check(status, "SetSessionLogSeverityLevel")
    }

    pub fn enable_profiling(&self, prefix: &Path) -> Result<()> {
        let enable = self
            .api
            .f(self.api.raw.EnableProfiling, "EnableProfiling")?;
        let p = OrtString::new(prefix)?;
        // SAFETY: `p` outlives the call; ORT copies the prefix.
        let status = unsafe { enable(self.raw, p.as_ptr()) };
        self.api.check(status, "EnableProfiling")
    }

    /// Append the plugin EP that owns the selected `OrtEpDevice`s. This is the 1.24+ selection
    /// path: naming devices, not naming a provider string, so "the EP is in this session" is a
    /// fact established *before* the session exists rather than inferred afterwards.
    pub fn append_ep_devices(
        &self,
        env: &Env,
        devices: &[EpDeviceInfo],
        options: &[(String, String)],
    ) -> Result<()> {
        let append = self.api.f(
            self.api.raw.SessionOptionsAppendExecutionProvider_V2,
            "SessionOptionsAppendExecutionProvider_V2",
        )?;
        let raw_devices: Vec<*const ort::OrtEpDevice> = devices.iter().map(|d| d.raw).collect();
        let keys: Vec<CString> = options
            .iter()
            .map(|(k, _)| cstring(k))
            .collect::<Result<_>>()?;
        let values: Vec<CString> = options
            .iter()
            .map(|(_, v)| cstring(v))
            .collect::<Result<_>>()?;
        let key_ptrs: Vec<*const std::os::raw::c_char> = keys.iter().map(|k| k.as_ptr()).collect();
        let value_ptrs: Vec<*const std::os::raw::c_char> =
            values.iter().map(|v| v.as_ptr()).collect();
        // SAFETY: all four arrays outlive the call and their lengths are passed alongside them;
        // the device pointers are ORT's own and remain valid while `env` lives.
        let status = unsafe {
            append(
                self.raw,
                env.raw,
                raw_devices.as_ptr(),
                raw_devices.len(),
                key_ptrs.as_ptr(),
                value_ptrs.as_ptr(),
                key_ptrs.len(),
            )
        };
        self.api
            .check(status, "SessionOptionsAppendExecutionProvider_V2")
    }
}

impl Drop for SessionOptions {
    fn drop(&mut self) {
        if let Some(release) = self.api.raw.ReleaseSessionOptions {
            // SAFETY: released exactly once; ORT copies session options into the session, so this
            // is safe to drop while a session created from it is alive.
            unsafe { release(self.raw) };
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSpec {
    pub name: String,
    pub element_type: ort::ONNXTensorElementDataType,
    /// `-1` for a free dimension, as ORT reports it.
    pub dims: Vec<i64>,
    /// Symbolic names, parallel to `dims`; empty string where ORT has none.
    pub symbolic: Vec<String>,
}

pub struct Session {
    api: Api,
    pub raw: *mut ort::OrtSession,
    pub inputs: Vec<TensorSpec>,
    pub outputs: Vec<TensorSpec>,
}

impl Session {
    pub fn new(api: Api, env: &Env, model: &Path, options: &SessionOptions) -> Result<Self> {
        let create = api.f(api.raw.CreateSession, "CreateSession")?;
        let path = OrtString::new(model)?;
        let mut raw: *mut ort::OrtSession = ptr::null_mut();
        // SAFETY: `path` outlives the call; ORT reads the model during it.
        let status = unsafe { create(env.raw, path.as_ptr(), options.raw, &mut raw) };
        api.check(status, &format!("CreateSession({})", model.display()))?;
        let mut session = Self {
            api,
            raw,
            inputs: Vec::new(),
            outputs: Vec::new(),
        };
        session.inputs = session.io_specs(true)?;
        session.outputs = session.io_specs(false)?;
        Ok(session)
    }

    fn allocator(&self) -> Result<*mut ort::OrtAllocator> {
        let get = self.api.f(
            self.api.raw.GetAllocatorWithDefaultOptions,
            "GetAllocatorWithDefaultOptions",
        )?;
        let mut allocator: *mut ort::OrtAllocator = ptr::null_mut();
        // SAFETY: valid out-pointer; the returned allocator is process-static.
        let status = unsafe { get(&mut allocator) };
        self.api.check(status, "GetAllocatorWithDefaultOptions")?;
        Ok(allocator)
    }

    fn io_specs(&self, inputs: bool) -> Result<Vec<TensorSpec>> {
        let allocator = self.allocator()?;
        let count_fn = if inputs {
            self.api
                .f(self.api.raw.SessionGetInputCount, "SessionGetInputCount")?
        } else {
            self.api
                .f(self.api.raw.SessionGetOutputCount, "SessionGetOutputCount")?
        };
        let mut count: usize = 0;
        // SAFETY: live session, valid out-pointer.
        let status = unsafe { count_fn(self.raw, &mut count) };
        self.api.check(status, "SessionGetInput/OutputCount")?;

        let mut specs = Vec::with_capacity(count);
        for index in 0..count {
            let name = self.io_name(allocator, index, inputs)?;
            let (element_type, dims, symbolic) = self.io_type(index, inputs)?;
            specs.push(TensorSpec {
                name,
                element_type,
                dims,
                symbolic,
            });
        }
        Ok(specs)
    }

    fn io_name(
        &self,
        allocator: *mut ort::OrtAllocator,
        index: usize,
        inputs: bool,
    ) -> Result<String> {
        let name_fn = if inputs {
            self.api
                .f(self.api.raw.SessionGetInputName, "SessionGetInputName")?
        } else {
            self.api
                .f(self.api.raw.SessionGetOutputName, "SessionGetOutputName")?
        };
        let mut raw: *mut std::os::raw::c_char = ptr::null_mut();
        // SAFETY: live session and allocator, valid out-pointer.
        let status = unsafe { name_fn(self.raw, index, allocator, &mut raw) };
        self.api.check(status, "SessionGetInput/OutputName")?;
        // SAFETY: ORT allocated a NUL-terminated string with `allocator`; we copy it and hand the
        // allocation straight back, so there is no path on which it leaks.
        let owned = unsafe { CStr::from_ptr(raw) }
            .to_string_lossy()
            .into_owned();
        if let Some(free) = self.api.raw.AllocatorFree {
            // SAFETY: `raw` came from this allocator and is not used again.
            unsafe { free(allocator, raw as *mut std::ffi::c_void) };
        }
        Ok(owned)
    }

    fn io_type(
        &self,
        index: usize,
        inputs: bool,
    ) -> Result<(ort::ONNXTensorElementDataType, Vec<i64>, Vec<String>)> {
        let info_fn = if inputs {
            self.api.f(
                self.api.raw.SessionGetInputTypeInfo,
                "SessionGetInputTypeInfo",
            )?
        } else {
            self.api.f(
                self.api.raw.SessionGetOutputTypeInfo,
                "SessionGetOutputTypeInfo",
            )?
        };
        let mut type_info: *mut ort::OrtTypeInfo = ptr::null_mut();
        // SAFETY: live session, valid out-pointer.
        let status = unsafe { info_fn(self.raw, index, &mut type_info) };
        self.api.check(status, "SessionGetInput/OutputTypeInfo")?;

        let result = (|| -> Result<(ort::ONNXTensorElementDataType, Vec<i64>, Vec<String>)> {
            let cast = self.api.f(
                self.api.raw.CastTypeInfoToTensorInfo,
                "CastTypeInfoToTensorInfo",
            )?;
            let mut tensor_info: *const ort::OrtTensorTypeAndShapeInfo = ptr::null();
            // SAFETY: `type_info` is live until the release below.
            let status = unsafe { cast(type_info, &mut tensor_info) };
            self.api.check(status, "CastTypeInfoToTensorInfo")?;
            if tensor_info.is_null() {
                // A sequence/map/optional input. Not a tensor, so this runner cannot feed it, and
                // says so rather than inventing one.
                return Err(Failure::unsupported(
                    "non_tensor_io",
                    "the model declares a non-tensor input or output (sequence, map or optional). \
                     This runner feeds and compares tensors only.",
                ));
            }
            let get_type = self
                .api
                .f(self.api.raw.GetTensorElementType, "GetTensorElementType")?;
            let mut element_type: ort::ONNXTensorElementDataType = 0;
            // SAFETY: `tensor_info` borrows `type_info`, which is live.
            let status = unsafe { get_type(tensor_info, &mut element_type) };
            self.api.check(status, "GetTensorElementType")?;

            let get_rank = self
                .api
                .f(self.api.raw.GetDimensionsCount, "GetDimensionsCount")?;
            let mut rank: usize = 0;
            // SAFETY: as above.
            let status = unsafe { get_rank(tensor_info, &mut rank) };
            self.api.check(status, "GetDimensionsCount")?;

            let mut dims = vec![0i64; rank];
            if rank > 0 {
                let get_dims = self.api.f(self.api.raw.GetDimensions, "GetDimensions")?;
                // SAFETY: `dims` has exactly `rank` slots, which is the length passed.
                let status = unsafe { get_dims(tensor_info, dims.as_mut_ptr(), rank) };
                self.api.check(status, "GetDimensions")?;
            }

            let mut symbolic = vec![String::new(); rank];
            if rank > 0 {
                if let Some(get_sym) = self.api.raw.GetSymbolicDimensions {
                    let mut raw = vec![ptr::null::<std::os::raw::c_char>(); rank];
                    // SAFETY: `raw` has exactly `rank` slots; ORT fills it with pointers it owns.
                    let status = unsafe { get_sym(tensor_info, raw.as_mut_ptr(), rank) };
                    // A host that cannot report symbolic names is not a reason to fail: the dims
                    // are already known. Record what we can and move on.
                    if self.api.check(status, "GetSymbolicDimensions").is_ok() {
                        for (slot, p) in symbolic.iter_mut().zip(raw) {
                            if !p.is_null() {
                                // SAFETY: ORT-owned NUL-terminated string valid while
                                // `tensor_info` lives.
                                *slot = unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned();
                            }
                        }
                    }
                }
            }
            Ok((element_type, dims, symbolic))
        })();

        if let Some(release) = self.api.raw.ReleaseTypeInfo {
            // SAFETY: released exactly once, after every borrow of it above has ended.
            unsafe { release(type_info) };
        }
        result
    }

    /// Run the whole graph: all declared outputs, in declaration order.
    pub fn run(&self, feeds: &[(String, Value)]) -> Result<Vec<Value>> {
        let run = self.api.f(self.api.raw.Run, "Run")?;
        let input_names: Vec<CString> = feeds
            .iter()
            .map(|(n, _)| cstring(n))
            .collect::<Result<_>>()?;
        let output_names: Vec<CString> = self
            .outputs
            .iter()
            .map(|o| cstring(&o.name))
            .collect::<Result<_>>()?;
        let input_name_ptrs: Vec<*const std::os::raw::c_char> =
            input_names.iter().map(|n| n.as_ptr()).collect();
        let output_name_ptrs: Vec<*const std::os::raw::c_char> =
            output_names.iter().map(|n| n.as_ptr()).collect();
        let input_ptrs: Vec<*const ort::OrtValue> =
            feeds.iter().map(|(_, v)| v.raw as *const _).collect();
        let mut output_ptrs: Vec<*mut ort::OrtValue> = vec![ptr::null_mut(); output_names.len()];

        // SAFETY: all arrays outlive the call and their lengths are passed with them; the feed
        // values borrow host buffers that `feeds` keeps alive for the duration.
        let status = unsafe {
            run(
                self.raw,
                ptr::null(),
                input_name_ptrs.as_ptr(),
                input_ptrs.as_ptr(),
                input_ptrs.len(),
                output_name_ptrs.as_ptr(),
                output_name_ptrs.len(),
                output_ptrs.as_mut_ptr(),
            )
        };
        self.api.check(status, "Run")?;

        Ok(output_ptrs
            .into_iter()
            .map(|raw| Value {
                api: self.api,
                raw,
                owns: true,
                _host: Vec::new(),
            })
            .collect())
    }

    /// Stop profiling and return the trace file ORT wrote.
    pub fn end_profiling(&self) -> Result<String> {
        let end = self
            .api
            .f(self.api.raw.SessionEndProfiling, "SessionEndProfiling")?;
        let allocator = self.allocator()?;
        let mut raw: *mut std::os::raw::c_char = ptr::null_mut();
        // SAFETY: live session and allocator, valid out-pointer.
        let status = unsafe { end(self.raw, allocator, &mut raw) };
        self.api.check(status, "SessionEndProfiling")?;
        // SAFETY: ORT allocated this string with `allocator`; copied and freed immediately.
        let owned = unsafe { CStr::from_ptr(raw) }
            .to_string_lossy()
            .into_owned();
        if let Some(free) = self.api.raw.AllocatorFree {
            // SAFETY: `raw` came from this allocator and is not used again.
            unsafe { free(allocator, raw as *mut std::ffi::c_void) };
        }
        Ok(owned)
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        if let Some(release) = self.api.raw.ReleaseSession {
            // SAFETY: released exactly once, and every `Value` produced by it is either dropped
            // already or owns its own allocation.
            unsafe { release(self.raw) };
        }
    }
}

/// An `OrtValue`. When built by [`Value::tensor_from_host`] it *borrows* a host buffer, which is
/// carried inside so the buffer cannot outlive the value that points into it -- the lifetime rule
/// `rust/README.md` records having cost this project a crash once already.
pub struct Value {
    api: Api,
    pub raw: *mut ort::OrtValue,
    owns: bool,
    _host: Vec<u8>,
}

impl Value {
    pub fn tensor_from_host(
        api: Api,
        memory_info: &MemoryInfo,
        host: Vec<u8>,
        shape: &[i64],
        element_type: ort::ONNXTensorElementDataType,
    ) -> Result<Self> {
        let create = api.f(
            api.raw.CreateTensorWithDataAsOrtValue,
            "CreateTensorWithDataAsOrtValue",
        )?;
        let mut raw: *mut ort::OrtValue = ptr::null_mut();
        let mut host = host;
        // SAFETY: `host` is moved into the returned `Value` and is not reallocated afterwards, so
        // the pointer ORT keeps stays valid for as long as the value it belongs to.
        let status = unsafe {
            create(
                memory_info.raw,
                host.as_mut_ptr() as *mut std::ffi::c_void,
                host.len(),
                shape.as_ptr(),
                shape.len(),
                element_type,
                &mut raw,
            )
        };
        api.check(status, "CreateTensorWithDataAsOrtValue")?;
        Ok(Self {
            api,
            raw,
            owns: true,
            _host: host,
        })
    }

    pub fn type_and_shape(&self) -> Result<(ort::ONNXTensorElementDataType, Vec<i64>)> {
        let get = self
            .api
            .f(self.api.raw.GetTensorTypeAndShape, "GetTensorTypeAndShape")?;
        let mut info: *mut ort::OrtTensorTypeAndShapeInfo = ptr::null_mut();
        // SAFETY: live value, valid out-pointer.
        let status = unsafe { get(self.raw, &mut info) };
        self.api.check(status, "GetTensorTypeAndShape")?;
        let result = (|| -> Result<(ort::ONNXTensorElementDataType, Vec<i64>)> {
            let get_type = self
                .api
                .f(self.api.raw.GetTensorElementType, "GetTensorElementType")?;
            let mut element_type: ort::ONNXTensorElementDataType = 0;
            // SAFETY: `info` is live until released below.
            let status = unsafe { get_type(info, &mut element_type) };
            self.api.check(status, "GetTensorElementType")?;
            let get_rank = self
                .api
                .f(self.api.raw.GetDimensionsCount, "GetDimensionsCount")?;
            let mut rank: usize = 0;
            // SAFETY: as above.
            let status = unsafe { get_rank(info, &mut rank) };
            self.api.check(status, "GetDimensionsCount")?;
            let mut dims = vec![0i64; rank];
            if rank > 0 {
                let get_dims = self.api.f(self.api.raw.GetDimensions, "GetDimensions")?;
                // SAFETY: `dims` has exactly `rank` slots.
                let status = unsafe { get_dims(info, dims.as_mut_ptr(), rank) };
                self.api.check(status, "GetDimensions")?;
            }
            Ok((element_type, dims))
        })();
        if let Some(release) = self.api.raw.ReleaseTensorTypeAndShapeInfo {
            // SAFETY: released exactly once after every borrow above has ended.
            unsafe { release(info) };
        }
        result
    }

    /// The raw element bytes of a tensor ORT produced, copied out.
    ///
    /// Copied rather than borrowed on purpose: the buffer belongs to the session's allocator, the
    /// comparison outlives the session in the CPU-vs-Vulkan flow, and a borrow here would be a
    /// use-after-free waiting for a refactor.
    pub fn copy_bytes(&self, byte_len: usize) -> Result<Vec<u8>> {
        let get = self
            .api
            .f(self.api.raw.GetTensorMutableData, "GetTensorMutableData")?;
        let mut data: *mut std::ffi::c_void = ptr::null_mut();
        // SAFETY: live value, valid out-pointer.
        let status = unsafe { get(self.raw, &mut data) };
        self.api.check(status, "GetTensorMutableData")?;
        if data.is_null() {
            // A genuinely empty tensor has no data pointer; that is not an error, it is zero
            // bytes, and the caller's shape already says so.
            return Ok(Vec::new());
        }
        let mut out = vec![0u8; byte_len];
        // SAFETY: ORT owns `byte_len` bytes at `data` -- the length is derived from the tensor's
        // own element count and element size, read back from the same value.
        unsafe { ptr::copy_nonoverlapping(data as *const u8, out.as_mut_ptr(), byte_len) };
        Ok(out)
    }
}

impl Drop for Value {
    fn drop(&mut self) {
        if self.owns && !self.raw.is_null() {
            if let Some(release) = self.api.raw.ReleaseValue {
                // SAFETY: released exactly once; the host buffer (if any) is dropped after.
                unsafe { release(self.raw) };
            }
        }
    }
}

pub struct MemoryInfo {
    api: Api,
    pub raw: *mut ort::OrtMemoryInfo,
}

impl MemoryInfo {
    pub fn cpu(api: Api) -> Result<Self> {
        let create = api.f(api.raw.CreateCpuMemoryInfo, "CreateCpuMemoryInfo")?;
        let mut raw: *mut ort::OrtMemoryInfo = ptr::null_mut();
        // SAFETY: valid out-pointer.
        let status = unsafe {
            create(
                ort::OrtAllocatorType_OrtArenaAllocator,
                ort::OrtMemType_OrtMemTypeDefault,
                &mut raw,
            )
        };
        api.check(status, "CreateCpuMemoryInfo")?;
        Ok(Self { api, raw })
    }
}

impl Drop for MemoryInfo {
    fn drop(&mut self) {
        if let Some(release) = self.api.raw.ReleaseMemoryInfo {
            // SAFETY: released exactly once, after every value created against it is dropped.
            unsafe { release(self.raw) };
        }
    }
}

/// Element size in bytes, and the name this runner prints for it.
///
/// The `allow` is for bindgen's C-derived constant names, which are not Rust-cased; renaming them
/// locally would break the one property that matters here -- that these are literally the header's
/// own enumerators, so a mis-mapped element size is impossible.
#[allow(non_upper_case_globals)]
pub fn element_info(t: ort::ONNXTensorElementDataType) -> Option<(&'static str, usize)> {
    use onnxruntime_vulkan_ep::sys::ort::*;
    Some(match t {
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT => ("float32", 4),
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE => ("float64", 8),
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 => ("float16", 2),
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 => ("bfloat16", 2),
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 => ("int64", 8),
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 => ("int32", 4),
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 => ("int16", 2),
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 => ("int8", 1),
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 => ("uint64", 8),
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 => ("uint32", 4),
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 => ("uint16", 2),
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 => ("uint8", 1),
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL => ("bool", 1),
        _ => return None,
    })
}

pub fn element_name(t: ort::ONNXTensorElementDataType) -> String {
    element_info(t)
        .map(|(name, _)| name.to_string())
        .unwrap_or_else(|| format!("onnx_element_type_{t}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_round_trip_is_nul_terminated_in_the_platform_encoding() {
        let dir = std::env::temp_dir().join("a b/ünïcode");
        let s = OrtString::new(&dir).unwrap();
        #[cfg(target_os = "windows")]
        {
            assert_eq!(*s.buf.last().unwrap(), 0u16);
            // Non-ASCII must survive as UTF-16 code units, not as `?`.
            assert!(s.buf.iter().any(|&c| c > 127));
        }
        #[cfg(not(target_os = "windows"))]
        {
            assert_eq!(*s.buf.last().unwrap(), 0u8);
            assert!(s.buf.iter().any(|&c| c > 127));
        }
    }

    #[test]
    fn an_interior_nul_is_refused_rather_than_truncating_the_path() {
        // A path silently truncated at a NUL is a different file than the caller asked for.
        let bad = String::from("mo\u{0}del.onnx");
        let err = OrtString::new(Path::new(&bad)).unwrap_err();
        assert_eq!(err.token(), "ERROR(instrument=path_not_representable)");
        assert_eq!(
            cstring("a\u{0}b").unwrap_err().token(),
            "ERROR(instrument=string_not_representable)"
        );
    }

    #[test]
    fn element_sizes_are_the_onnx_ones() {
        use onnxruntime_vulkan_ep::sys::ort::*;
        assert_eq!(
            element_info(ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
            Some(("float32", 4))
        );
        assert_eq!(
            element_info(ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
            Some(("int64", 8))
        );
        assert_eq!(
            element_info(ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
            Some(("float16", 2))
        );
        // STRING is a tensor of pointers, not of bytes: it has no element size here, and the
        // runner must refuse it rather than compute a byte length from a size it invented.
        assert_eq!(
            element_info(ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING),
            None
        );
        assert_eq!(
            element_name(ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING),
            "onnx_element_type_8"
        );
    }

    // ---------------------------------------------------------------------------------------
    // The enum-width defect that broke the Linux lane the day the runner merged (issue #39).
    //
    // These are compile-time controls as much as run-time ones. `widen` is generic over
    // `Into<i64>`, so calling it with *both* signednesses in one test means narrowing it back to
    // a concrete `i32` (or `u32`) does not fail an assertion -- it fails to build, on the
    // machine of whoever narrowed it, which is the whole point. On Windows the second call is
    // the one that would stop compiling; on Linux the first.
    // ---------------------------------------------------------------------------------------

    #[test]
    fn a_discriminant_is_read_at_a_width_that_exists_on_every_target() {
        assert_eq!(widen(7i32), 7i64);
        assert_eq!(widen(7u32), 7i64);
        assert_eq!(widen(-1i32), -1i64);
        assert_eq!(widen(u32::MAX), 4_294_967_295i64);
    }

    #[test]
    fn the_error_code_alias_goes_through_the_widening_reader() {
        // `ort::OrtErrorCode` is `c_int` under MSVC and `c_uint` under GCC/Clang. This call is
        // the one that fails to compile if `widen` ever acquires a concrete parameter type, and
        // it is also the exact expression `Api::check` uses.
        let code = widen(ort::OrtErrorCode_ORT_FAIL);
        assert_eq!(code, 1i64);
        assert_eq!(widen(ort::OrtErrorCode_ORT_OK), 0i64);
        // The sentinel for "this ORT serves no GetErrorCode". Before the fix this was a bare
        // `-1` in a `match` whose other arm was alias-typed, which is `u32: Neg` on Linux.
        let absent: i64 = -1;
        assert!(absent < code);
    }

    #[test]
    fn a_logging_level_is_produced_only_from_the_bindgen_constant() {
        // No `as` cast anywhere: `raw()` returns whatever width this target's binding has, and
        // comparing it to the constant is the only assertion that is true on both.
        assert_eq!(
            LogSeverity::Verbose.raw(),
            ort::OrtLoggingLevel_ORT_LOGGING_LEVEL_VERBOSE
        );
        assert_eq!(
            LogSeverity::Info.raw(),
            ort::OrtLoggingLevel_ORT_LOGGING_LEVEL_INFO
        );
        assert_eq!(
            LogSeverity::Warning.raw(),
            ort::OrtLoggingLevel_ORT_LOGGING_LEVEL_WARNING
        );
        assert_eq!(
            LogSeverity::Error.raw(),
            ort::OrtLoggingLevel_ORT_LOGGING_LEVEL_ERROR
        );
        assert_eq!(
            LogSeverity::Fatal.raw(),
            ort::OrtLoggingLevel_ORT_LOGGING_LEVEL_FATAL
        );
    }

    #[test]
    fn the_session_level_is_the_python_harness_number_and_agrees_with_the_env_level() {
        // `SetSessionLogSeverityLevel` takes a plain `c_int`, so this half is portable at a
        // fixed width -- but it must still name the same severity as the environment does, or
        // the session and the env silently disagree. Compared through `widen` because the
        // env-side value's width is the one that varies.
        for (sev, expected) in [
            (LogSeverity::Verbose, 0),
            (LogSeverity::Info, 1),
            (LogSeverity::Warning, 2),
            (LogSeverity::Error, 3),
            (LogSeverity::Fatal, 4),
        ] {
            assert_eq!(sev.session_level(), expected);
            assert_eq!(widen(sev.raw()), i64::from(expected));
        }
    }
}
