//! **onnxruntime-ep-vulkan** — a cross-platform Vulkan compute execution provider for ONNX
//! Runtime, written in Rust and loaded as a standalone plugin EP.
//!
//! A stock ONNX Runtime (no fork, no rebuild) loads
//! `onnxruntime_vulkan_ep.dll` / `libonnxruntime_vulkan_ep.so` / `.dylib` via
//! `RegisterExecutionProviderLibrary`, resolves the two symbols exported below, and gets an EP
//! that translates fused ONNX subgraphs into Vulkan compute dispatches.
//!
//! ```python
//! import onnxruntime as ort, onnxruntime_ep_vulkan
//! onnxruntime_ep_vulkan.register_execution_provider_library()
//! sess = ort.InferenceSession(model, providers=["VulkanExecutionProvider", "CPUExecutionProvider"])
//! ```
//!
//! # Layers
//!
//! | Layer | Modules | Owner |
//! |---|---|---|
//! | L0 ORT C ABI | this module, [`factory`], [`ep`], [`sys`] | Tank |
//! | L1 plan & dispatch | [`engine`], [`registry`] | Morpheus (contract) / Tank (plumbing) |
//! | L2 ONNX op semantics | [`ops`] | Mouse |
//! | L3 Vulkan engine | `vk/*` (not yet present) | Switch |
//!
//! Two rules hold that structure up, and `tests/layering.rs` enforces both mechanically:
//! **the ORT ABI never appears in `src/ops/`**, and **raw Vulkan never appears in `src/ops/`**.
//!
//! # FFI discipline
//!
//! * ORT is reached **only** through the `OrtApi` function-pointer table handed to
//!   [`CreateEpFactories`]. We never link `libonnxruntime`.
//! * Every exported `extern "C"` entry point that runs real logic goes through
//!   [`guard_ffi_status`]: a Rust panic becomes an `ORT_EP_FAIL` status instead of unwinding into
//!   ORT's C++ (undefined behaviour) or aborting the host process.
//! * Ownership crosses the boundary with `Box::into_raw` / `Box::from_raw`; teardown is RAII.
//! * Every `unsafe` block carries a `// SAFETY:` comment stating its invariant.
//! * No `unwrap()` on a fallible value anywhere near the boundary.
//!
//! # Status: M0
//!
//! The plugin loads, negotiates the ORT API version, reports its identity, enumerates devices
//! through a stubbed capability probe, creates and releases an EP per session, and declines every
//! node with a logged reason. Vulkan device code (Switch) and op handlers (Mouse) land on top of
//! this without touching the boundary layer.

// Deny-by-default hygiene for a crate whose failure mode is silent memory corruption.
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::undocumented_unsafe_blocks)]

pub mod allocator;
pub mod counters;
pub mod engine;
pub mod ep;
pub mod factory;
pub mod logging;
pub mod ops;
pub mod registry;
pub mod sys;
pub mod transfer;
// Observability (Niobe). Env-gated Chrome-Trace tracing + the GPU-timestamp ingest seam; inert
// unless `ONNXRUNTIME_EP_VULKAN_TRACE` names a path. See `docs/PERF.md`.
pub mod trace;
pub(crate) mod vk;

use std::ffi::c_char;
use std::ptr;

use factory::VulkanEpFactory;
use sys::ort;

/// Catch any panic at a C-ABI entry point and convert it into an `OrtStatus`.
///
/// Unwinding a Rust panic into ORT's C++ frames is undefined behaviour, and aborting would take
/// down a host process that merely wanted to try an EP. So every exported function that runs real
/// logic is wrapped here: on a caught panic we log the panic message (to stderr *and* to ORT's
/// logger, so it is never silent even if the host installed a quiet panic hook) and hand ORT a
/// non-null `ORT_EP_FAIL` status. ORT then fails that call — and for EP compute, falls back to the
/// CPU EP — instead of dying.
///
/// `api` is the `OrtApi` used to allocate the status. It may be null (nothing else can allocate an
/// `OrtStatus`), in which case a caught panic yields a null status; the log line still happens.
///
/// # Safety
/// `api` must be null or a live `OrtApi`. `body` must itself uphold whatever invariants its own
/// raw-pointer arguments require.
pub(crate) unsafe fn guard_ffi_status(
    api: *const ort::OrtApi,
    what: &'static str,
    body: impl FnOnce() -> ort::OrtStatusPtr,
) -> ort::OrtStatusPtr {
    // `AssertUnwindSafe` is sound here for the reason it usually is not: on the panic path we
    // never touch the captured state again — we discard it and return an error status — so a
    // logically-torn value cannot be observed.
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
        Ok(status) => status,
        Err(payload) => {
            let detail = panic_payload_message(payload.as_ref());
            log::error!(
                "caught a panic in {what}: {detail} — returning ORT_EP_FAIL. The host process is \
                 protected; ONNX Runtime will surface the failure or fall back to the CPU EP."
            );
            // SAFETY: `api` is null or live per this function's contract; `make_status` handles
            // the null case itself and copies the message before returning.
            unsafe {
                sys::make_status(
                    api,
                    ort::OrtErrorCode_ORT_EP_FAIL,
                    &format!(
                        "VulkanExecutionProvider recovered from a panic in {what}: {detail} \
                         (host protected); the operation failed"
                    ),
                )
            }
        }
    }
}

/// Best-effort human-readable message from a caught panic payload.
///
/// Rust panics carry either a `&'static str` or a `String`; anything else is rare and unnameable.
pub(crate) fn panic_payload_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unrecoverable panic (non-string payload)".to_string()
    }
}

/// Catch a panic at a C-ABI entry point whose return type is a **pointer**, not a status.
///
/// `OrtAllocator::Alloc` and `OrtAllocator::Info` cannot report failure through an `OrtStatus`;
/// their only failure channel is a null return, which ORT already handles as out-of-memory. So the
/// guard converts a panic into null — a contract-legal answer — rather than letting it unwind into
/// ORT's C++, which is undefined behaviour.
///
/// This is a strictly worse failure report than [`guard_ffi_status`] gives, and that is a property
/// of ORT's allocator signature rather than a choice. The log line is therefore the only place the
/// cause survives, so it is emitted unconditionally.
pub(crate) fn guard_ffi_ptr<T>(body: impl FnOnce() -> *mut T) -> *mut T {
    // `AssertUnwindSafe`: on the panic path the captured state is discarded, never re-read.
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
        Ok(p) => p,
        Err(payload) => {
            log::error!(
                "caught a panic at a pointer-returning ORT entry point: {} — returning null. ORT \
                 reads null as an allocation failure, which is the only failure this signature can \
                 express; the host process is protected.",
                panic_payload_message(payload.as_ref())
            );
            ptr::null_mut()
        }
    }
}

/// Catch a panic at a C-ABI entry point that returns nothing (`OrtAllocator::Free`).
///
/// There is no failure channel at all here, so the panic is swallowed after being logged. That is
/// the whole contract: a `Free` that panics must not unwind into ORT and must not abort the host.
pub(crate) fn guard_ffi_void(body: impl FnOnce()) {
    if let Err(payload) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
        log::error!(
            "caught a panic at a void ORT entry point: {} — swallowed. This signature has no \
             failure channel, so this log line is the only record; treat it as a hard bug.",
            panic_payload_message(payload.as_ref())
        );
    }
}

/// Catch a panic at a C-ABI entry point that returns a **bool** (`OrtDataTransferImpl::CanCopy`).
///
/// The only safe answer on the panic path is `fallback`, which every caller passes as "no": saying
/// "yes, I can copy that" and then failing is worse than declining, because ORT can route the copy
/// elsewhere but cannot un-route it.
pub(crate) fn guard_ffi_bool(
    what: &'static str,
    fallback: bool,
    body: impl FnOnce() -> bool,
) -> bool {
    // `AssertUnwindSafe`: on the panic path the captured state is discarded, never re-read.
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
        Ok(v) => v,
        Err(payload) => {
            log::error!(
                "caught a panic in {what}: {} — answering {fallback}. This signature has no \
                 failure channel, so declining is the only contract-legal way to fail.",
                panic_payload_message(payload.as_ref())
            );
            fallback
        }
    }
}

// -------------------------------------------------------------------------------------------
// Exported entry points — the entire public C ABI of this library
// -------------------------------------------------------------------------------------------

/// ORT resolves this symbol by name when an application calls
/// `RegisterExecutionProviderLibrary`.
///
/// Contract: fill up to `max_factories` slots in `factories`, write the count to `num_factories`,
/// and return null on success or an `OrtStatus` on failure. We produce exactly one factory.
///
/// **No Vulkan work happens here.** A plugin must be cheap to load even on a machine that will
/// never use it; the `VkInstance` is created lazily in `GetSupportedDevices` (`DESIGN.md` §5.1).
///
/// # Safety
/// Called by ORT with valid ABI pointers: `ort_api_base` non-null, `factories` writable for
/// `max_factories` entries, `num_factories` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn CreateEpFactories(
    registration_name: *const c_char,
    ort_api_base: *const ort::OrtApiBase,
    default_logger: *const ort::OrtLogger,
    factories: *mut *mut ort::OrtEpFactory,
    max_factories: usize,
    num_factories: *mut usize,
) -> ort::OrtStatusPtr {
    logging::init();

    // Negotiate the ABI *before* entering the guard, because the guard needs a status-capable
    // `OrtApi` to report a caught panic with. If negotiation fails we still need *some* API to
    // build the error status from, so fall back to the legacy v1 table purely for that purpose —
    // `CreateStatus` has been at the same slot since ORT 1.0.
    // SAFETY: `ort_api_base` is the table ORT passed us; `check_api_version` null-checks it before
    // any dereference.
    let negotiated = unsafe { sys::check_api_version(ort_api_base) };
    let api_for_status: *const ort::OrtApi = match &negotiated {
        Ok(n) => n.api,
        Err(_) if ort_api_base.is_null() => ptr::null(),
        Err(_) => {
            // SAFETY: `ort_api_base` is non-null here, and `GetApi` is a member of a struct whose
            // layout has never changed; requesting version 1 is always legal and returns either a
            // valid table or null.
            unsafe { (*ort_api_base).GetApi.map_or(ptr::null(), |f| f(1)) }
        }
    };

    let negotiated = match negotiated {
        Ok(n) => n,
        Err(message) => {
            log::error!("{message}");
            // SAFETY: `api_for_status` is null or a live `OrtApi`.
            return unsafe {
                sys::make_status(
                    api_for_status,
                    ort::OrtErrorCode_ORT_INVALID_ARGUMENT,
                    &message,
                )
            };
        }
    };

    // Route our log records into ORT's logger from here on.
    // SAFETY: `negotiated.api` is live; `default_logger` is ORT's process-default logger, valid
    // until `ReleaseEpFactory` detaches it.
    unsafe { logging::attach_default_ort_logger(negotiated.api, default_logger) };

    // SAFETY: `api_for_status` is a live `OrtApi`; the closure only touches ORT-supplied pointers.
    unsafe {
        guard_ffi_status(api_for_status, "CreateEpFactories", || {
            create_ep_factories_impl(
                registration_name,
                negotiated.api,
                negotiated.ep_api,
                negotiated.version,
                factories,
                max_factories,
                num_factories,
            )
        })
    }
}

/// # Safety
/// `factories` must be writable for `max_factories` entries and `num_factories` writable;
/// `api`/`ep_api` must be the negotiated live tables, and `abi_version` the version they were
/// negotiated at.
unsafe fn create_ep_factories_impl(
    registration_name: *const c_char,
    api: *const ort::OrtApi,
    ep_api: *const ort::OrtEpApi,
    abi_version: u32,
    factories: *mut *mut ort::OrtEpFactory,
    max_factories: usize,
    num_factories: *mut usize,
) -> ort::OrtStatusPtr {
    if num_factories.is_null() || factories.is_null() {
        // SAFETY: `api` is live.
        return unsafe {
            sys::make_status(
                api,
                ort::OrtErrorCode_ORT_INVALID_ARGUMENT,
                "CreateEpFactories received a null factories or num_factories out-parameter",
            )
        };
    }
    // SAFETY: valid out-param slot; set first so every early return leaves it defined.
    unsafe { *num_factories = 0 };

    if max_factories < 1 {
        // SAFETY: `api` is live.
        return unsafe {
            sys::make_status(
                api,
                ort::OrtErrorCode_ORT_INVALID_ARGUMENT,
                "VulkanExecutionProvider needs room for one OrtEpFactory",
            )
        };
    }

    // SAFETY: `registration_name` is null or a NUL-terminated string ORT owns; `api`/`ep_api` are
    // the live negotiated tables, which outlive the factory.
    let factory = unsafe { VulkanEpFactory::new(registration_name, api, ep_api, abi_version) };

    // SAFETY: `max_factories >= 1`, so slot 0 is inside the array ORT gave us. Ownership of the
    // factory passes to ORT, which returns it via `ReleaseEpFactory`.
    unsafe {
        *factories = factory.into_raw();
        *num_factories = 1;
    }

    log::info!(
        "VulkanExecutionProvider v{} loaded (built against ORT API {}, negotiated {})",
        env!("CARGO_PKG_VERSION"),
        sys::ORT_API_VERSION_EXPECTED,
        abi_version
    );
    ptr::null_mut()
}

/// Free a factory produced by [`CreateEpFactories`].
///
/// # Safety
/// `factory` must be null, or a pointer [`CreateEpFactories`] wrote, released exactly once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ReleaseEpFactory(factory: *mut ort::OrtEpFactory) -> ort::OrtStatusPtr {
    // Report before tearing down: this is the last moment at which anything in this process can
    // say what the EP actually did, and a lane that executed nothing has to be able to say so out
    // loud rather than by omission. `summary()` allocates a String, which is why it sits above the
    // release rather than inside a `Drop`.
    let snap = counters::snapshot();
    log::info!("{}", snap.summary());
    if snap.dispatches_executed == 0 {
        log::warn!(
            "VulkanExecutionProvider executed ZERO dispatches in this process. If a test lane \
             expected GPU work, this is a failure that would otherwise be invisible: every node \
             ran somewhere else."
        );
    }
    counters::dump_if_requested();

    // Not wrapped in `guard_ffi_status`: we have no `OrtApi` to build a status from once the
    // factory is being torn down, and the only work here is a `Box` drop. Anything that could
    // panic during teardown would live in a `Drop` impl, and ours are audited to be panic-free.
    // SAFETY: `factory` is null or a pointer we produced, released exactly once.
    unsafe { VulkanEpFactory::release(factory) };
    ptr::null_mut()
}

// -------------------------------------------------------------------------------------------
// Execution counters — the criterion-8 evidence channel
// -------------------------------------------------------------------------------------------

/// Copy the EP's execution counters into `out`.
///
/// Returns the number of bytes written, or 0 if `out` is null or `out_bytes` is under 8. See
/// [`counters`] for what each field means and, more importantly, what they do *not* claim.
///
/// This is not part of the ORT plugin ABI — ORT never calls it. It exists so a test harness that
/// has the library loaded (ORT loaded it; `ctypes.CDLL` on the same path gets the same module) can
/// ask "did anything actually execute?" without inferring it from test outcomes. Inferring it from
/// test outcomes is what let a lane where everything skipped report success.
///
/// # Safety
/// `out` must be null, or writable for `out_bytes` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn OrtEpVulkanGetExecutionCounters(
    out: *mut std::ffi::c_void,
    out_bytes: usize,
) -> usize {
    // SAFETY: the contract on `out`/`out_bytes` is passed straight through to `fill`, which
    // null-checks and clamps.
    unsafe { counters::fill(out, out_bytes) }
}

/// Zero the EP's execution counters.
///
/// For a harness that wants to scope a claim to one model run: reset, run, read. Without it the
/// only available claim is process-cumulative, and "some dispatch executed at some point in this
/// pytest session" is a much weaker statement than "this model executed on the GPU".
#[unsafe(no_mangle)]
pub extern "C" fn OrtEpVulkanResetExecutionCounters() {
    counters::reset();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_panic_is_caught_and_reported_without_an_api() {
        // With a null `OrtApi` there is nothing that can allocate a status, so the guard returns
        // null — but it must still *catch*, not unwind.
        // SAFETY: `api` is null, which the guard explicitly permits.
        let status = unsafe {
            guard_ffi_status(ptr::null(), "unit-test", || {
                panic!("deliberate test panic");
            })
        };
        assert!(status.is_null());
    }

    #[test]
    fn a_clean_return_passes_through_the_guard() {
        // SAFETY: `api` is null and the body dereferences nothing.
        let status = unsafe { guard_ffi_status(ptr::null(), "unit-test", ptr::null_mut) };
        assert!(status.is_null());
    }

    #[test]
    fn panic_payload_messages_are_extracted() {
        let s: Box<dyn std::any::Any + Send> = Box::new("static str panic");
        assert_eq!(panic_payload_message(s.as_ref()), "static str panic");
        let s: Box<dyn std::any::Any + Send> = Box::new(String::from("owned panic"));
        assert_eq!(panic_payload_message(s.as_ref()), "owned panic");
        let s: Box<dyn std::any::Any + Send> = Box::new(42u32);
        assert!(panic_payload_message(s.as_ref()).contains("non-string"));
    }

    #[test]
    fn create_ep_factories_rejects_a_null_api_base() {
        let mut factories: *mut ort::OrtEpFactory = ptr::null_mut();
        let mut n: usize = 99;
        // SAFETY: a null `OrtApiBase` must be rejected before any dereference; the out-params are
        // real stack slots.
        let status = unsafe {
            CreateEpFactories(
                ptr::null(),
                ptr::null(),
                ptr::null(),
                &mut factories,
                1,
                &mut n,
            )
        };
        // No API means no status object can be allocated, but it must not crash and must not
        // pretend to have produced a factory.
        assert!(status.is_null());
        assert!(factories.is_null());
    }

    #[test]
    fn releasing_a_null_factory_is_a_noop() {
        // SAFETY: null is explicitly permitted.
        let status = unsafe { ReleaseEpFactory(ptr::null_mut()) };
        assert!(status.is_null());
    }
}
