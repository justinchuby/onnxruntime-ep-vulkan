//! Logging for the plugin cdylib, and the bridge into ORT's own logger.
//!
//! Two sinks, one facade:
//!
//! * **stderr** — always available, prefixed `[vulkan-ep]`, so a caught panic or a load failure is
//!   visible even before ORT has handed us a logger (and even if the host installed a quiet panic
//!   hook).
//! * **ORT's `OrtLogger`** — when ORT has given us one, every record is *also* forwarded through
//!   `OrtApi::Logger_LogMessage`, so plugin messages land in the host's log with the host's
//!   severity filtering, sinks, and correlation. This is what makes the EP debuggable from an
//!   application that never looks at stderr.
//!
//! A cdylib has its own statically linked copy of the `log` crate, so `set_boxed_logger` here can
//! never conflict with whatever the host process uses.
//!
//! # Default behaviour (near-silent)
//!
//! With no environment variables set the max level is **Warn**: only `error!`/`warn!` (caught
//! panics, device-enumeration failures, user-visible problems) are emitted.
//!
//! | Env var | Max level |
//! |---|---|
//! | *(none)* | `Warn` |
//! | `ONNXRUNTIME_EP_VULKAN_VERBOSE=1` | `Info` |
//! | `ONNXRUNTIME_EP_VULKAN_TRACE=<path>` | `Debug` |
//! | `ONNXRUNTIME_EP_VULKAN_CLAIM_DEBUG=1` | `Debug` |
//! | `RUST_LOG=onnxruntime_ep_vulkan=<level>` | explicit (highest precedence) |

use std::ffi::CString;
use std::sync::Once;
use std::sync::atomic::{AtomicPtr, Ordering};

use log::{Level, LevelFilter, Log, Metadata, Record};

use crate::sys::ort;

/// Env var: emit `Info` and above.
pub const ENV_VERBOSE: &str = "ONNXRUNTIME_EP_VULKAN_VERBOSE";
/// Env var: trace file path; implies `Debug`.
pub const ENV_TRACE: &str = "ONNXRUNTIME_EP_VULKAN_TRACE";
/// Env var: print per-op claim/decline reasons from `GetCapability`; implies `Debug`.
pub const ENV_CLAIM_DEBUG: &str = "ONNXRUNTIME_EP_VULKAN_CLAIM_DEBUG";

static INIT: Once = Once::new();

/// The ORT logger and API we forward to, or null when ORT has not given us one (yet).
///
/// These are raw C pointers into ORT's address space. They are only ever written by
/// [`attach_ort_logger`] / [`detach_ort_logger`], which the factory calls from
/// `CreateEpFactories` and `ReleaseEpFactory` respectively, and only ever read behind a null
/// check. Both are owned by ORT and outlive the window between those two calls.
static ORT_LOGGER: AtomicPtr<ort::OrtLogger> = AtomicPtr::new(std::ptr::null_mut());
static ORT_API: AtomicPtr<ort::OrtApi> = AtomicPtr::new(std::ptr::null_mut());

struct EpLogger {
    level: LevelFilter,
}

impl Log for EpLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= self.level
    }

    fn log(&self, record: &Record) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let tag = match record.level() {
            Level::Error => "ERROR",
            Level::Warn => "WARN",
            Level::Info => "INFO",
            Level::Debug => "DEBUG",
            Level::Trace => "TRACE",
        };
        let message = record.args().to_string();
        eprintln!("[vulkan-ep] {tag}: {message}");
        forward_to_ort(record.level(), record.target(), &message);
    }

    fn flush(&self) {}
}

/// Map a Rust log level onto ORT's severity scale.
fn ort_severity(level: Level) -> ort::OrtLoggingLevel {
    match level {
        Level::Error => ort::OrtLoggingLevel_ORT_LOGGING_LEVEL_ERROR,
        Level::Warn => ort::OrtLoggingLevel_ORT_LOGGING_LEVEL_WARNING,
        Level::Info => ort::OrtLoggingLevel_ORT_LOGGING_LEVEL_INFO,
        // ORT has no separate Debug/Trace tier; both are VERBOSE, and ORT's own severity filter
        // decides whether they are actually recorded.
        Level::Debug | Level::Trace => ort::OrtLoggingLevel_ORT_LOGGING_LEVEL_VERBOSE,
    }
}

/// Forward one record into ORT's logger, if one is attached. Best-effort and never panics: a
/// logging failure must not be able to fail an inference.
fn forward_to_ort(level: Level, target: &str, message: &str) {
    let logger = ORT_LOGGER.load(Ordering::Acquire);
    let api = ORT_API.load(Ordering::Acquire);
    if logger.is_null() || api.is_null() {
        return;
    }
    // ORT copies the message; interior nuls would truncate it, so replace them.
    let Ok(c_message) = CString::new(message.replace('\0', "?")) else {
        return;
    };
    let Ok(c_target) = CString::new(target.replace('\0', "?")) else {
        return;
    };

    // SAFETY: `api` and `logger` were published by `attach_ort_logger` from pointers ORT handed to
    // `CreateEpFactories`, and are cleared by `detach_ort_logger` before ORT can invalidate them,
    // so a non-null read here is a live ORT logger. `Logger_LogMessage` copies both strings, so
    // the `CString`s only need to outlive the call. `file_path` is null (permitted: ORT treats it
    // as "no source location"), which also side-steps the `wchar_t` width difference between
    // Windows (u16) and Unix (u32). The returned status is owned by us and released immediately.
    unsafe {
        let Some(log_message) = (*api).Logger_LogMessage else {
            return;
        };
        let status = log_message(
            logger,
            ort_severity(level),
            c_message.as_ptr(),
            std::ptr::null(),
            0,
            c_target.as_ptr(),
        );
        crate::sys::release_status(api, status);
    }
}

/// Start forwarding log records into ORT's logger.
///
/// # Safety
/// `api` and `logger` must be valid for as long as this attachment is live, i.e. until
/// [`detach_ort_logger`] is called. The factory attaches the default logger ORT passes to
/// `CreateEpFactories` and detaches it in `ReleaseEpFactory`, which is exactly that window.
pub unsafe fn attach_ort_logger(api: *const ort::OrtApi, logger: *const ort::OrtLogger) {
    if api.is_null() || logger.is_null() {
        return;
    }
    ORT_API.store(api.cast_mut(), Ordering::Release);
    ORT_LOGGER.store(logger.cast_mut(), Ordering::Release);
}

/// Stop forwarding to ORT. Idempotent. Called before the pointers can become dangling.
pub fn detach_ort_logger() {
    ORT_LOGGER.store(std::ptr::null_mut(), Ordering::Release);
    ORT_API.store(std::ptr::null_mut(), Ordering::Release);
}

/// True when per-node claim/decline reasons should be printed by `GetCapability`.
pub fn claim_debug_enabled() -> bool {
    std::env::var_os(ENV_CLAIM_DEBUG).is_some_and(|v| v != "0")
}

/// Resolve the max log level from the environment.
fn resolve_level() -> LevelFilter {
    if let Ok(val) = std::env::var("RUST_LOG") {
        // Accept either a bare level ("debug") or a crate-qualified one
        // ("onnxruntime_ep_vulkan=debug", possibly among other comma-separated directives).
        let level_str = val
            .split(',')
            .find(|s| s.contains("onnxruntime_ep_vulkan"))
            .and_then(|s| s.split('=').nth(1))
            .unwrap_or(&val);
        if let Ok(f) = level_str.trim().parse::<LevelFilter>() {
            return f;
        }
    }
    if claim_debug_enabled() {
        return LevelFilter::Debug;
    }
    if std::env::var_os(ENV_TRACE).is_some_and(|v| !v.is_empty()) {
        return LevelFilter::Debug;
    }
    if std::env::var(ENV_VERBOSE).map(|v| v == "1").unwrap_or(false) {
        return LevelFilter::Info;
    }
    LevelFilter::Warn
}

/// Install the logger. Idempotent, safe to call from any entry point.
pub fn init() {
    INIT.call_once(|| {
        let level = resolve_level();
        // `set_boxed_logger` can only succeed once per copy of `log`. If something inside this
        // dylib already installed one, keep it rather than failing the load.
        let _ = log::set_boxed_logger(Box::new(EpLogger { level }));
        log::set_max_level(level);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn severity_mapping_is_monotonic() {
        assert_eq!(
            ort_severity(Level::Error),
            ort::OrtLoggingLevel_ORT_LOGGING_LEVEL_ERROR
        );
        assert_eq!(
            ort_severity(Level::Warn),
            ort::OrtLoggingLevel_ORT_LOGGING_LEVEL_WARNING
        );
        assert_eq!(
            ort_severity(Level::Info),
            ort::OrtLoggingLevel_ORT_LOGGING_LEVEL_INFO
        );
        assert_eq!(
            ort_severity(Level::Debug),
            ort::OrtLoggingLevel_ORT_LOGGING_LEVEL_VERBOSE
        );
        assert_eq!(
            ort_severity(Level::Trace),
            ort::OrtLoggingLevel_ORT_LOGGING_LEVEL_VERBOSE
        );
    }

    #[test]
    fn forwarding_is_a_noop_without_an_attached_logger() {
        detach_ort_logger();
        // Must not dereference anything: the pointers are null.
        forward_to_ort(Level::Error, "test", "no logger attached");
    }

    #[test]
    fn attach_rejects_null_pointers() {
        detach_ort_logger();
        // SAFETY: both arguments are null, which `attach_ort_logger` is required to reject
        // without dereferencing.
        unsafe { attach_ort_logger(std::ptr::null(), std::ptr::null()) };
        assert!(ORT_LOGGER.load(Ordering::Acquire).is_null());
        assert!(ORT_API.load(Ordering::Acquire).is_null());
    }
}
