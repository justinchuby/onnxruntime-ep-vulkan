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
/// The process-default logger from `CreateEpFactories`, kept so a session logger can be unwound.
///
/// `CreateEp` swaps in the session's logger; when that session goes away the pointer becomes
/// dangling, so `ReleaseEp` must put this one back. Without it, every log record emitted between
/// one session's teardown and the next session's creation would forward into freed memory.
static ORT_DEFAULT_LOGGER: AtomicPtr<ort::OrtLogger> = AtomicPtr::new(std::ptr::null_mut());

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
        let _ = forward_to_ort(
            record.level(),
            record.target(),
            &message,
            record.file(),
            record.line().unwrap_or(0),
        );
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
///
/// # `file_path` must never be null
///
/// `OrtApi::Logger_LogMessage` annotates `file_path` `_In_z_`, not `_In_opt_z_`, and ORT means it.
/// On Windows the implementation does
/// `const std::string s = onnxruntime::ToUTF8String(file_path);`, which constructs a
/// `std::wstring` from the pointer — a null there is an access violation inside `wcslen`, not an
/// ignored argument. (On Unix it is `CodeLocation(file_path, …)`, i.e. `std::string{nullptr}`,
/// equally undefined.) We passed null here and it killed the first ORT process that ever loaded
/// this plugin. So `file_path` is always a real, NUL-terminated, platform-width string; when a
/// record carries no source location we substitute [`UNKNOWN_FILE`] rather than null.
fn forward_to_ort(
    level: Level,
    target: &str,
    message: &str,
    file: Option<&str>,
    line: u32,
) -> bool {
    let logger = ORT_LOGGER.load(Ordering::Acquire);
    let api = ORT_API.load(Ordering::Acquire);
    if logger.is_null() || api.is_null() {
        return false;
    }
    // ORT copies the message; interior nuls would truncate it, so replace them.
    let Ok(c_message) = CString::new(message.replace('\0', "?")) else {
        return false;
    };
    let Ok(c_target) = CString::new(target.replace('\0', "?")) else {
        return false;
    };
    let c_file = ort_path(file.unwrap_or(UNKNOWN_FILE));

    // SAFETY: `api` and `logger` were published by `attach_ort_logger` from pointers ORT handed to
    // `CreateEpFactories`, and are cleared by `detach_ort_logger` before ORT can invalidate them,
    // so a non-null read here is a live ORT logger. `Logger_LogMessage` copies all three strings,
    // so the buffers only need to outlive the call. Every string argument is non-null and
    // NUL-terminated, which the `_In_z_` annotations require — see this function's doc comment for
    // what happens when `file_path` is not. The returned status is owned by us and released
    // immediately.
    unsafe {
        let Some(log_message) = (*api).Logger_LogMessage else {
            return false;
        };
        let status = log_message(
            logger,
            ort_severity(level),
            c_message.as_ptr(),
            c_file.as_ptr(),
            i32::try_from(line).unwrap_or(0),
            c_target.as_ptr(),
        );
        crate::sys::release_status(api, status);
        true
    }
}

/// Emit one WARNING into **ORT's own logging sink**, bypassing this crate's `log` facade
/// entirely. Returns `true` when the message actually reached `Logger_LogMessage`.
///
/// # Why this exists when `log::warn!` already forwards to ORT
///
/// `log::warn!` reaches ORT's sink only when *our* `LevelFilter` lets the record through, and that
/// filter is environment-controlled ([`resolve_level`], `RUST_LOG`). A disclosure that a user can
/// switch off with an environment variable is not a disclosure; RAI Ruling 2 requires the
/// broken-commitment WARN to fire **every time, with no opt-out**, so it must not travel down a
/// path with a filter on it.
///
/// The stderr line is emitted too, but it is deliberately the *second* witness: a WARN in this
/// project's private log is invisible to exactly the audience that matters — a host with ORT
/// logging configured, watching the channel that already carries ORT's own `Falling back` line.
/// The boolean return is what lets the counters artifact say which channel actually carried it,
/// so `PRIVATE_LOG_ONLY` can never be read as a delivered disclosure.
pub fn warn_through_ort_sink(target: &str, message: &str) -> bool {
    eprintln!("[vulkan-ep] WARN: {message}");
    forward_to_ort(Level::Warn, target, message, Some(UNKNOWN_FILE), 0)
}

/// Emit one INFO into **ORT's own logging sink**, bypassing this crate's `log` facade, for the
/// same reason [`warn_through_ort_sink`] does.
///
/// §8.9.7's session-creation disclosure is a *pair*: the WARN says a claimed form has no proof,
/// and the INFO says what the proven forms were proven by. They have to travel down the same
/// channel or the pair is not readable as a pair — a user who sees the WARN and looks for the
/// context would find it in a different log, or in no log, depending on `RUST_LOG`.
///
/// ORT's own severity filter still applies at its end, and that is correct: ORT's INFO tier is
/// the host's to configure. What must not be switchable is *ours*, because a disclosure that our
/// own environment can suppress is a disclosure whose absence means nothing.
pub fn info_through_ort_sink(target: &str, message: &str) -> bool {
    eprintln!("[vulkan-ep] INFO: {message}");
    forward_to_ort(Level::Info, target, message, Some(UNKNOWN_FILE), 0)
}

/// Stand-in `file_path` for a record with no source location. Never empty, never null.
pub const UNKNOWN_FILE: &str = "<onnxruntime-ep-vulkan>";

/// Encode a path as an `ORTCHAR_T` string: UTF-16 on Windows, UTF-8 elsewhere.
///
/// Returns an owned buffer whose `as_ptr()` is a valid NUL-terminated string of the platform's
/// `ORTCHAR_T` width. Interior NULs are dropped/replaced rather than allowed to truncate.
#[cfg(windows)]
fn ort_path(s: &str) -> Vec<u16> {
    s.encode_utf16()
        .filter(|c| *c != 0)
        .chain(std::iter::once(0))
        .collect()
}

#[cfg(not(windows))]
fn ort_path(s: &str) -> CString {
    CString::new(s.replace('\0', "?")).unwrap_or_else(|_| CString::default())
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

/// Attach the *process-default* logger and remember it, so a session logger can later be unwound
/// back to it by [`restore_default_ort_logger`].
///
/// # Safety
/// Same contract as [`attach_ort_logger`]; called only from `CreateEpFactories` with the logger
/// ORT guarantees lives until `ReleaseEpFactory`.
pub unsafe fn attach_default_ort_logger(api: *const ort::OrtApi, logger: *const ort::OrtLogger) {
    if api.is_null() || logger.is_null() {
        return;
    }
    ORT_DEFAULT_LOGGER.store(logger.cast_mut(), Ordering::Release);
    // SAFETY: both pointers are non-null and, per the caller's contract, live.
    unsafe { attach_ort_logger(api, logger) };
}

/// Put the process-default logger back after a session's logger goes away.
///
/// Called from `ReleaseEp`. If there is no default (the factory was created without one) this
/// detaches entirely rather than leaving the dead session logger in place — silence is correct,
/// a dangling pointer is not.
pub fn restore_default_ort_logger() {
    let default = ORT_DEFAULT_LOGGER.load(Ordering::Acquire);
    ORT_LOGGER.store(default, Ordering::Release);
    if default.is_null() {
        ORT_API.store(std::ptr::null_mut(), Ordering::Release);
    }
}

/// Stop forwarding to ORT. Idempotent. Called before the pointers can become dangling.
pub fn detach_ort_logger() {
    ORT_LOGGER.store(std::ptr::null_mut(), Ordering::Release);
    ORT_DEFAULT_LOGGER.store(std::ptr::null_mut(), Ordering::Release);
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
    if std::env::var(ENV_VERBOSE)
        .map(|v| v == "1")
        .unwrap_or(false)
    {
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
        // Process-global logger pointers: same lock as `attach_rejects_null_pointers`.
        let _g = crate::allocator::ledger::test_lock();
        detach_ort_logger();
        // Must not dereference anything: the pointers are null.
        forward_to_ort(Level::Error, "test", "no logger attached", Some("x.rs"), 1);
    }

    #[test]
    fn ort_path_is_never_empty_and_is_nul_terminated() {
        // ORT annotates `file_path` `_In_z_`; a null or unterminated buffer is an access
        // violation inside ORT, not a tolerated argument.
        for s in ["src/logging.rs", UNKNOWN_FILE, "", "a\0b"] {
            let buf = ort_path(s);
            #[cfg(windows)]
            {
                assert_eq!(buf.last().copied(), Some(0), "missing NUL for {s:?}");
                assert!(
                    buf[..buf.len() - 1].iter().all(|c| *c != 0),
                    "interior NUL for {s:?}"
                );
            }
            #[cfg(not(windows))]
            {
                assert!(!buf.as_bytes().contains(&0), "interior NUL for {s:?}");
            }
        }
    }

    #[test]
    fn unknown_file_substitute_is_a_real_string() {
        assert!(!UNKNOWN_FILE.is_empty());
        assert!(!UNKNOWN_FILE.contains('\0'));
    }

    #[test]
    fn attach_rejects_null_pointers() {
        // The ORT logger pointers are process-global: the same lock the counters and the
        // `ep::tests::session_disclosure` arms take. Without it this test detaches the sink
        // out from under a disclosure arm mid-run, and that arm reports
        // `warn_reached_ort_sink: false` -- a WARN that did reach ORT, recorded as one that
        // did not. Observed 1 run in 4 of `cargo test --lib` on 2026-08-03.
        let _g = crate::allocator::ledger::test_lock();
        detach_ort_logger();
        // SAFETY: both arguments are null, which `attach_ort_logger` is required to reject
        // without dereferencing.
        unsafe { attach_ort_logger(std::ptr::null(), std::ptr::null()) };
        assert!(ORT_LOGGER.load(Ordering::Acquire).is_null());
        assert!(ORT_API.load(Ordering::Acquire).is_null());
    }
}
