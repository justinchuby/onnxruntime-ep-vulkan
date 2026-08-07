//! The run itself: CPU reference, Vulkan candidate, five guards, one evidence document.
//!
//! THE SHAPE OF THE PROOF
//! ----------------------
//! The claim "the Vulkan EP correctly executed this real model" decomposes into five separate
//! facts, and the reason they are separate is that four of them are individually satisfiable by a
//! run that proves nothing:
//!
//! | guard | what it rules out |
//! |---|---|
//! | `model_identity_pinned`   | a different file with the right name |
//! | `vulkan_ep_device_present`| the plugin not loading at all |
//! | `vulkan_ep_in_session`    | the EP loading but never being appended |
//! | `vulkan_executed_nodes`   | the EP being appended but claiming nothing (silent CPU fallback) |
//! | `vulkan_dispatched_work`  | nodes claimed but no compute submitted |
//! | `outputs_agree`           | work submitted that computes the wrong answer |
//!
//! The fourth is the load-bearing one, and it is the guard
//! `rust/tools/probe_model_output_agreement.py` documents but does not implement: that probe
//! checks `VulkanExecutionProvider in sess.get_providers()`, which is true whenever the EP was
//! *requested*, whether or not it took a single node. A model that fell back entirely to CPU
//! passes that check and then agrees with CPU perfectly. This runner therefore reads ORT's own
//! profile -- an instrument outside the frame under question -- and requires it to attribute at
//! least one node to the Vulkan provider.
//!
//! PRIMARY AND CORROBORATING WITNESSES
//! -----------------------------------
//! `tests/ops/_verdict.py` fixed the attribution rule this module follows: ORT's profile is
//! *primary* because ORT is not the thing under test, and our own `dispatches_executed` counter is
//! *corroborating* because it lives inside the component whose behaviour is in question. Both are
//! recorded; a pass requires both; and where they disagree the evidence says so rather than
//! picking the flattering one.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use onnxruntime_vulkan_ep::sys::ort;

use crate::compare::{self, OutputComparison, Tolerance, Verdict};
use crate::error::{Failure, Result, Severity};
use crate::evidence::{Counters, FileIdentity, Guard, Outcome};
use crate::feeds;
use crate::json::{self, Json};
use crate::ortapi::{Api, Env, LogSeverity, MemoryInfo, Session, SessionOptions, element_name};
use crate::ortlib;
use crate::provenance::{self, VerifiedModel};

/// The exact registration name the plugin is published under. Not configurable: a mismatch here
/// is the difference between "the EP declined the graph" and "we asked for an EP that does not
/// exist", and those must not be confusable.
pub const EP_NAME: &str = "VulkanExecutionProvider";

/// Everything one invocation needs, already validated.
#[derive(Debug, Clone)]
pub struct RunConfig {
    /// A name in `bench/results/model_provenance.json`, or `""` when `model_path` is explicit.
    pub model: String,
    /// An explicit model file, used instead of the pinned cache entry.
    pub model_path: Option<PathBuf>,
    pub ep_lib: Option<PathBuf>,
    pub ort_lib: Option<PathBuf>,
    pub out: Option<PathBuf>,
    pub seed: u64,
    pub free_dims: BTreeMap<String, i64>,
    pub input_files: BTreeMap<String, PathBuf>,
    pub rtol: Option<f64>,
    pub atol: Option<f64>,
    pub fetch: bool,
    /// Skip the Vulkan arm entirely and record only the CPU reference. Used to prove the harness
    /// itself works on a machine with no Vulkan device -- and it can never report PASS.
    pub cpu_only: bool,
    /// Pin the Vulkan device by **stable identity** for this run (issue #18).
    ///
    /// Set into `ONNXRUNTIME_EP_VULKAN_DEVICE_SELECTOR` before the EP library is registered, which
    /// is the only point at which it can reach `engine::devices_to_advertise` — there is exactly
    /// one authoritative selection path and this is it. When present, the run additionally
    /// enforces that the device the session *opened* is the device that was *requested*, and
    /// fails closed when they differ.
    pub device_selector: Option<String>,
    pub keep_profile: bool,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            model: String::new(),
            model_path: None,
            ep_lib: None,
            ort_lib: None,
            out: None,
            // A fixed default seed, not a clock: two invocations of the same command must feed the
            // same bytes, or a disagreement cannot be reproduced from the command line alone.
            seed: 0x5EED_0000_0000_0001,
            free_dims: BTreeMap::new(),
            input_files: BTreeMap::new(),
            rtol: None,
            atol: None,
            fetch: false,
            cpu_only: false,
            device_selector: None,
            keep_profile: false,
        }
    }
}

/// One provider's execution: outputs, timing, and where they came from.
struct ArmResult {
    outputs: Vec<(String, Vec<f64>, Vec<i64>, ort::ONNXTensorElementDataType)>,
    load_ms: f64,
    run_ms: f64,
    profile_path: Option<PathBuf>,
}

/// The default plugin library location, relative to the repository root.
pub fn default_ep_lib(repo: &Path) -> PathBuf {
    let name = if cfg!(target_os = "windows") {
        "onnxruntime_vulkan_ep.dll"
    } else if cfg!(target_os = "macos") {
        "libonnxruntime_vulkan_ep.dylib"
    } else {
        "libonnxruntime_vulkan_ep.so"
    };
    repo.join("rust").join("target").join("release").join(name)
}

fn now_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

/// Resolve the model file and its identity, pinned wherever a pin exists.
fn resolve_model(config: &RunConfig, repo: &Path) -> Result<VerifiedModel> {
    if let Some(path) = &config.model_path {
        if !path.is_file() {
            return Err(Failure::instrument(
                "model_missing",
                format!("--model-path {} is not a file", path.display()),
            ));
        }
        let name = if config.model.is_empty() {
            path.file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "unnamed".to_string())
        } else {
            config.model.clone()
        };
        // An explicit path still gets a pin check when the name is one we pin: an out-of-tree copy
        // of mnist-12 that differs from the pinned bytes is not mnist-12.
        let manifest = provenance::load_manifest(&manifest_path(repo))?;
        if let Ok(pin) = provenance::entry(&manifest, &name) {
            return provenance::verify_file(path, pin);
        }
        return provenance::measure_model(&name, path);
    }
    if config.model.is_empty() {
        return Err(Failure::instrument(
            "no_model_named",
            "pass --model <name from bench/results/model_provenance.json> or --model-path <file>",
        ));
    }
    let manifest = provenance::load_manifest(&manifest_path(repo))?;
    let cache = provenance::model_cache_dir()?;
    provenance::ensure_model(&manifest, &config.model, &cache, config.fetch)
}

pub fn manifest_path(repo: &Path) -> PathBuf {
    repo.join("bench")
        .join("results")
        .join("model_provenance.json")
}

/// Run one provider arm. `ep` is `None` for the CPU reference.
fn run_arm(
    api: Api,
    env: &Env,
    model: &Path,
    ep_devices: &[crate::ortapi::EpDeviceInfo],
    seed: u64,
    config: &RunConfig,
    profile_prefix: Option<&Path>,
) -> Result<(ArmResult, Vec<feeds::FeedRecord>, Vec<feeds::DimPin>)> {
    let options = SessionOptions::new(api)?;
    // `Error` matches the Python harness's `log_severity_level = 3`: ORT's warnings about a
    // plugin EP declining nodes are expected and would drown the output that matters.
    options.set_log_severity(LogSeverity::Error)?;
    if let Some(prefix) = profile_prefix {
        options.enable_profiling(prefix)?;
    }
    if !ep_devices.is_empty() {
        options.append_ep_devices(env, ep_devices, &[])?;
    }

    let load_start = Instant::now();
    let session = Session::new(api, env, model, &options)?;
    let load_ms = now_ms(load_start);

    let memory_info = MemoryInfo::cpu(api)?;
    let built = feeds::build(
        api,
        &memory_info,
        &session.inputs,
        seed,
        &config.free_dims,
        &config.input_files,
    )?;

    let run_start = Instant::now();
    let values = session.run(&built.values)?;
    let run_ms = now_ms(run_start);

    let mut outputs = Vec::with_capacity(values.len());
    for (spec, value) in session.outputs.iter().zip(values.iter()) {
        let (element_type, shape) = value.type_and_shape()?;
        let (_, size) = crate::ortapi::element_info(element_type).ok_or_else(|| {
            Failure::unsupported(
                "output_dtype_unsupported",
                format!(
                    "output {} has element type {}, which this runner cannot read back",
                    spec.name,
                    element_name(element_type)
                ),
            )
        })?;
        let count = feeds::element_count(&shape)?;
        let bytes = value.copy_bytes(count * size)?;
        outputs.push((
            spec.name.clone(),
            compare::decode(&bytes, element_type)?,
            shape,
            element_type,
        ));
    }

    let profile_path = if profile_prefix.is_some() {
        Some(PathBuf::from(session.end_profiling()?))
    } else {
        None
    };

    // Drop order is load-bearing: the session must go before the values that borrow its allocator
    // are gone, and both before the environment. Rust's own drop order (reverse declaration)
    // gives exactly that, which is why nothing here is dropped by hand.
    drop(values);
    drop(session);
    drop(memory_info);

    Ok((
        ArmResult {
            outputs,
            load_ms,
            run_ms,
            profile_path,
        },
        built.records,
        built.pins,
    ))
}

/// Count the nodes ORT's own profile attributed to each provider.
fn read_profile(path: &Path) -> (BTreeMap<String, u64>, String) {
    let Ok(text) = std::fs::read_to_string(path) else {
        return (
            BTreeMap::new(),
            format!("profile {} could not be read", path.display()),
        );
    };
    match json::parse(&text) {
        Ok(doc) => (json::tally_providers(&doc), String::new()),
        Err(e) => (
            BTreeMap::new(),
            format!("profile {} is not JSON: {e}", path.display()),
        ),
    }
}

/// Compare the device the EP session **actually opened** against the devices ORT **advertised**,
/// and describe the selected device (issue #18, blocker 3 / contract C7).
///
/// Returns the `device_identity_agreement` guard and the `execution_provider.selected_device`
/// object that goes into the evidence document beside `execution_provider.devices`.
///
/// # Why the enumeration is not the answer
///
/// `execution_provider.devices` records what `GetEpDevices` listed. That list is what was *on
/// offer*. Which one the session opened is a different fact, held only by the EP, and the whole
/// §6.5 family of defects lives in the gap between them: ORT binds one `OrtEpDevice`, the EP's
/// selector indexes a differently-sorted list, and the session opens a third device while the
/// document proudly records all four as "devices seen". An evidence artifact that cannot name the
/// device its numbers came from is not evidence about a device.
///
/// # Fail closed
///
/// Every branch that cannot *establish* agreement returns a red guard, including the ones where
/// nothing is provably wrong (no identity reported, identity unattributable to any advertised
/// device). That asymmetry is deliberate. The cost of a false red is a rerun; the cost of a false
/// green is a number in `bench/results/` attributed to hardware it never touched, which is the
/// failure this repository exists to make impossible.
fn device_identity_agreement(
    counters: &Counters,
    advertised: &[crate::ortapi::EpDeviceInfo],
    config: &RunConfig,
) -> (Guard, Json) {
    let keys = counters.session_device_keys();
    let names = counters.session_device_names();
    let selected_name = names.first().map(String::as_str).unwrap_or("");
    let selected_key = keys.first().cloned();

    let advertised_match = selected_key.as_deref().and_then(|k| {
        let want = k.strip_prefix("uuid:")?;
        advertised
            .iter()
            .find(|d| d.uuid.as_deref() == Some(want))
            .map(|d| d.index)
    });

    let selected_device = Json::obj(vec![
        (
            "identity",
            match &selected_key {
                Some(k) => Json::s(k.clone()),
                None => Json::Null,
            },
        ),
        (
            "name",
            if selected_name.is_empty() {
                Json::Null
            } else {
                Json::s(selected_name)
            },
        ),
        (
            "advertised_index",
            match advertised_match {
                Some(i) => Json::int(i as i64),
                None => Json::Null,
            },
        ),
        (
            "requested_selector",
            match &config.device_selector {
                Some(s) => Json::s(s.clone()),
                None => Json::Null,
            },
        ),
        (
            "selector_source",
            Json::s(if config.device_selector.is_some() {
                "--device-selector (ONNXRUNTIME_EP_VULKAN_DEVICE_SELECTOR, set before EP \
                 registration)"
            } else {
                "none: ORT's binding chose the device"
            }),
        ),
        (
            "alloc_device_frame",
            match &counters.alloc_device_frame {
                Some(f) => Json::s(f.clone()),
                None => Json::Null,
            },
        ),
    ]);

    // Each arm answers: can this run attribute its numbers to exactly one piece of hardware that
    // ORT actually offered it?
    let guard = if config.cpu_only {
        Guard::new(
            "device_identity_agreement",
            false,
            "--cpu-only: no Vulkan device was opened, so no number in this document can be \
             attributed to one",
        )
    } else if !counters.present {
        Guard::new(
            "device_identity_agreement",
            false,
            format!(
                "no counters snapshot, so the EP reported no session device identity. {} This is \
                 an absent instrument, not agreement.",
                counters.note
            ),
        )
    } else if keys.is_empty() {
        Guard::new(
            "device_identity_agreement",
            false,
            "the EP reported no running device identity (running_device_uuids is absent or \
             empty). Either no VkDevice was opened, or this EP build predates the stable-identity \
             surface — either way this run cannot say which hardware produced its numbers.",
        )
    } else if keys.len() > 1 {
        Guard::new(
            "device_identity_agreement",
            false,
            format!(
                "the EP opened {} distinct physical devices in one run ({}). The outputs compared \
                 above came from more than one piece of hardware and cannot be attributed to \
                 either.",
                keys.len(),
                keys.join(", ")
            ),
        )
    } else if counters.alloc_device_frames_declared.unwrap_or(1) > 1
        || counters.alloc_device_frame.as_deref() == Some("MIXED")
    {
        Guard::new(
            "device_identity_agreement",
            false,
            format!(
                "the EP declared {} distinct (frame, device identity) pairs — alloc_device_frame \
                 = {:?}. The allocator numbers and the session describe different devices.",
                counters.alloc_device_frames_declared.unwrap_or(0),
                counters.alloc_device_frame.as_deref().unwrap_or("<absent>"),
            ),
        )
    } else if counters.alloc_device_frame.as_deref() == Some("SPLIT-DEVICE") {
        Guard::new(
            "device_identity_agreement",
            false,
            format!(
                "alloc_device_frame = SPLIT-DEVICE: the session opened `{}` while the allocator \
                 was stood up on another device. Set --device-selector so only one device is \
                 advertised.",
                selected_key.clone().unwrap_or_default()
            ),
        )
    } else if advertised.is_empty() {
        Guard::new(
            "device_identity_agreement",
            false,
            format!(
                "the EP opened `{}` but ORT advertised no {EP_NAME} device to compare it against",
                selected_key.clone().unwrap_or_default()
            ),
        )
    } else if advertised_match.is_none() {
        let key = selected_key.clone().unwrap_or_default();
        Guard::new(
            "device_identity_agreement",
            false,
            format!(
                "the device the session opened (`{key}`{}) does not match any device ORT \
                 advertised for {EP_NAME}: [{}]. Either the session opened a device that was \
                 never offered, or the advertised devices carry no vulkan.device_uuid metadata to \
                 compare against — in both cases this run's numbers cannot be attributed to \
                 advertised hardware, so it fails rather than warning.",
                if selected_name.is_empty() {
                    String::new()
                } else {
                    format!(", '{selected_name}'")
                },
                advertised
                    .iter()
                    .map(|d| format!(
                        "{}={}",
                        d.index,
                        d.uuid
                            .as_deref()
                            .map(|u| format!("uuid:{u}"))
                            .unwrap_or_else(|| "(no uuid metadata)".to_string())
                    ))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        )
    } else {
        let key = selected_key.clone().unwrap_or_default();
        let idx = advertised_match.unwrap_or(0);
        // A `uuid:` selector is checked against the opened identity directly as well, so the
        // agreement does not rest solely on the EP having advertised what it later opened.
        let requested_uuid = config
            .device_selector
            .as_deref()
            .and_then(|s| s.strip_prefix("uuid:"))
            .map(|u| u.trim().to_ascii_lowercase());
        match requested_uuid {
            Some(want) if key != format!("uuid:{want}") => Guard::new(
                "device_identity_agreement",
                false,
                format!(
                    "--device-selector requested uuid:{want} but the session opened `{key}` \
                     ('{selected_name}'). The selector is a stable identity, not a preference: a \
                     run that answers a question about different hardware than the one it was \
                     asked about is unattributed, so it fails."
                ),
            ),
            _ => Guard::new(
                "device_identity_agreement",
                true,
                format!(
                    "the session opened exactly one device, `{key}` ('{selected_name}'), and it \
                     is advertised OrtEpDevice index {idx}{}. alloc_device_frame = {}.",
                    match &config.device_selector {
                        Some(s) => format!(", which is the device --device-selector {s} requested"),
                        None => String::new(),
                    },
                    counters.alloc_device_frame.as_deref().unwrap_or("<absent>"),
                ),
            ),
        }
    };

    (guard, selected_device)
}

/// The complete run: both arms, all guards, one document.
pub fn execute(config: &RunConfig) -> Result<(Outcome, Json)> {
    let repo = crate::repo::root()?;
    let started = Instant::now();

    // The instrument is validated before the subject, and this order is load-bearing rather than
    // incidental. `discover` is a pure path check -- it opens nothing and hashes nothing -- so
    // running it first costs nothing and means an unusable ONNX Runtime is reported as
    // `ort_library_missing` / `ort_library_ambiguous` even when the model is also absent. The
    // other order hides instrument faults behind subject faults: on a machine where `mnist-12`
    // happened to be cached it reported the library error, and on a clean CI runner the same
    // code reported `model_not_cached` instead, which is how issue #39's Windows integration
    // test could pass on a dev box and fail on every fresh checkout. A measurement whose
    // instrument is broken says nothing about its subject, so the instrument is checked first.
    //
    // Only the *discovery* moves. Hashing the library and `dlopen`ing it stay below the model
    // resolution, so the model is still identified before any foreign code is mapped in.
    let discovered = ortlib::discover(&ortlib::Search::from_environment(
        Some(&repo),
        config.ort_lib.clone(),
    ))?;

    let verified = resolve_model(config, &repo)?;
    let model_identity = FileIdentity {
        path: verified.path.display().to_string(),
        sha256: verified.sha256.clone(),
        bytes: verified.bytes,
    };

    let ep_lib = config
        .ep_lib
        .clone()
        .unwrap_or_else(|| default_ep_lib(&repo));
    let ort_identity = FileIdentity::of(&discovered.path)?;
    let loaded = ortlib::load(discovered)?;
    let api = Api::new(loaded.api);

    // Scratch space for the profile and the counters snapshot. These are inputs to the evidence
    // document, not the evidence: once read, their content lives in the document, and a stale copy
    // sitting beside it would invite a later reader to trust a file nothing verifies. So they go to
    // the OS temp directory unless --keep-profile asks for them, in which case they are put beside
    // the named evidence file where CI can collect them together.
    //
    // The directory is never removed afterwards: the EP dumps its counters again as the process
    // tears down, and pulling the directory out from under that dump turns a clean run into one
    // that prints a write failure.
    let scratch = config
        .out
        .as_ref()
        .filter(|_| config.keep_profile)
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .filter(|d| !d.as_os_str().is_empty())
        .unwrap_or_else(std::env::temp_dir)
        .join("ort-model-runner-scratch");
    std::fs::create_dir_all(&scratch).map_err(|e| {
        Failure::instrument(
            "scratch_unwritable",
            format!("cannot create {}: {e}", scratch.display()),
        )
    })?;
    let counters_path = scratch.join(format!("counters-{}.json", verified.name));
    let _ = std::fs::remove_file(&counters_path);

    let mut guards: Vec<Guard> = Vec::new();
    guards.push(Guard::new(
        "model_identity_pinned",
        verified.provenance == "pinned",
        format!(
            "{} sha256 {} ({} bytes), provenance={}",
            verified.path.display(),
            verified.sha256,
            verified.bytes,
            verified.provenance
        ),
    ));

    // -- CPU reference arm ------------------------------------------------------------------
    let env = Env::new(api, "ort-model-runner", LogSeverity::Error)?;
    let cpu_arm = run_arm(api, &env, &verified.path, &[], config.seed, config, None);
    let (cpu, feed_records, dim_pins) = match cpu_arm {
        Ok(v) => v,
        Err(e) => {
            // A failure in the *reference* arm says nothing about the Vulkan EP -- the CPU
            // provider is not the thing under test. It means this runner could not construct a
            // coherent input set for the model, which is the case for graphs whose inputs are
            // semantically interdependent (KV caches, attention sequence lengths, tokenised
            // text). That is UNSUPPORTED, not a failed claim and not an EP defect, and the
            // evidence still carries the model identity so the refusal is checkable.
            let reason = if matches!(e.severity, Severity::Unsupported) {
                e
            } else {
                Failure::unsupported(
                    "reference_run_unsupported",
                    format!(
                        "the CPU reference arm could not run {} with generated inputs, so there \
                         is no reference to compare a Vulkan run against. This is a limit of the \
                         runner's input generation, not a result about the execution provider. \
                         Supply real inputs with --input <name>=<file.raw>, or pin the free \
                         dimensions with --free-dim, if this model's inputs are interdependent \
                         (KV cache, attention sequence lengths, tokenised text).\nUnderlying: \
                         {}: {}",
                        verified.name,
                        e.token(),
                        e.message
                    ),
                )
            };
            let doc = Json::obj(vec![
                ("schema", Json::s("ort-model-runner/1")),
                ("tool", Json::s("rust/modelrunner (ort-model-runner)")),
                ("pass", Json::Bool(false)),
                ("outcome", Json::s(Outcome::Unsupported.as_str())),
                ("model", Json::s(verified.name.clone())),
                ("onnx_file", Json::s(model_identity.path.clone())),
                ("onnx_sha256", Json::s(model_identity.sha256.clone())),
                ("onnx_bytes", Json::int(model_identity.bytes as i64)),
                ("model_provenance", Json::s(verified.provenance)),
                (
                    "onnxruntime",
                    Json::obj(vec![
                        ("library", ort_identity.to_json()),
                        ("api_version", Json::int(loaded.api_version as i64)),
                        ("version", Json::s(loaded.version_string.clone())),
                    ]),
                ),
                ("refusal_token", Json::s(reason.token())),
                ("refusal", Json::s(reason.message.clone())),
                (
                    "guards",
                    Json::Arr(
                        guards
                            .iter()
                            .cloned()
                            .chain(std::iter::once(Guard::new(
                                "reference_arm_ran",
                                false,
                                reason.message.clone(),
                            )))
                            .map(|g| g.to_json())
                            .collect(),
                    ),
                ),
            ]);
            return Ok((Outcome::Unsupported, doc));
        }
    };

    // -- Vulkan candidate arm ---------------------------------------------------------------
    let mut comparisons: Vec<OutputComparison> = Vec::new();
    let mut counters = Counters::default();
    let mut provider_tally: BTreeMap<String, u64> = BTreeMap::new();
    let mut profile_note = String::new();
    let mut vulkan_arm: Option<ArmResult> = None;
    let mut ep_identity: Option<FileIdentity> = None;
    let mut devices_seen: Vec<Json> = Vec::new();
    // The `OrtEpDevice`s ORT advertised for *our* EP, hoisted out of the Vulkan block so the
    // identity-agreement guard can resolve the device the session actually opened against the
    // devices that were actually offered.
    let mut our_devices: Vec<crate::ortapi::EpDeviceInfo> = Vec::new();
    let mut tolerance: Option<Tolerance> = None;

    if config.cpu_only {
        guards.push(Guard::new(
            "vulkan_ep_device_present",
            false,
            "--cpu-only was given: the Vulkan arm did not run, so this run cannot pass",
        ));
    } else {
        if !ep_lib.is_file() {
            return Err(Failure::instrument(
                "ep_library_missing",
                format!(
                    "the plugin library {} does not exist. Build it with `cargo build --release` \
                     in rust/, or pass --ep-lib.",
                    ep_lib.display()
                ),
            ));
        }
        ep_identity = Some(FileIdentity::of(&ep_lib)?);

        // The EP reads this at load time, so it must be set before registration. This is the one
        // process-wide mutation the runner makes, and it is made once, before any thread that
        // could observe it exists.
        //
        // SAFETY: single-threaded at this point -- no session, no ORT worker pool, and the
        // runner's own work is sequential. `set_var` is `unsafe` in edition 2024 precisely
        // because of concurrent readers, which cannot exist here.
        unsafe {
            std::env::set_var(
                onnxruntime_vulkan_ep::counters::ENV_COUNTERS_FILE,
                &counters_path,
            );
        }

        // Issue #18: the ONE authoritative selection path. This must be set before
        // `register_ep_library`, because `engine::devices_to_advertise` reads it inside
        // `GetSupportedDevices` — which runs before any session, and therefore before any session
        // option could possibly be visible. Setting it here means ORT is *offered* only the
        // requested device, so ORT's binding and the session's selection cannot diverge by
        // construction rather than by hope.
        //
        // SAFETY: as above -- single-threaded, before registration, before any ORT worker exists.
        if let Some(sel) = &config.device_selector {
            unsafe {
                std::env::set_var(onnxruntime_vulkan_ep::ENV_DEVICE_SELECTOR_STRICT, sel);
            }
        }

        env.register_ep_library(EP_NAME, &ep_lib)?;
        let all_devices = env.ep_devices()?;
        for d in &all_devices {
            devices_seen.push(Json::obj(vec![
                ("index", Json::int(d.index as i64)),
                ("ep_name", Json::s(d.ep_name.clone())),
                ("ep_vendor", Json::s(d.ep_vendor.clone())),
                ("hardware_type", Json::s(d.hardware_type.clone())),
                ("vendor_id", Json::int(d.vendor_id as i64)),
                ("device_id", Json::int(d.device_id as i64)),
                // Stable identity (issue #18): `null` (never a fabricated placeholder) when the
                // field is genuinely unavailable on this EP/platform, e.g. `luid`/`pci` on
                // MoltenVK, or all three on a non-Vulkan EP with no matching metadata key.
                ("uuid", d.uuid.clone().map(Json::s).unwrap_or(Json::Null)),
                ("luid", d.luid.clone().map(Json::s).unwrap_or(Json::Null)),
                ("pci", d.pci.clone().map(Json::s).unwrap_or(Json::Null)),
            ]));
        }
        let ours: Vec<_> = all_devices
            .iter()
            .filter(|d| d.ep_name == EP_NAME)
            .cloned()
            .collect();
        our_devices = ours.clone();
        guards.push(Guard::new(
            "vulkan_ep_device_present",
            !ours.is_empty(),
            if ours.is_empty() {
                format!(
                    "no OrtEpDevice named {EP_NAME} after registering {}. ORT saw {} device(s): \
                     {}",
                    ep_lib.display(),
                    all_devices.len(),
                    all_devices
                        .iter()
                        .map(|d| d.ep_name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            } else {
                format!(
                    "{} device(s) named {EP_NAME}: {}",
                    ours.len(),
                    ours.iter()
                        .map(|d| format!(
                            "{} {:04x}:{:04x}",
                            d.hardware_type, d.vendor_id, d.device_id
                        ))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            },
        ));

        if !ours.is_empty() {
            let profile_prefix = scratch.join(format!("profile-{}", verified.name));
            let arm = run_arm(
                api,
                &env,
                &verified.path,
                &ours,
                config.seed,
                config,
                Some(&profile_prefix),
            );
            match arm {
                Ok((vulkan, _, _)) => {
                    guards.push(Guard::new(
                        "vulkan_ep_in_session",
                        true,
                        "SessionOptionsAppendExecutionProvider_V2 accepted the Vulkan EpDevice \
                         and the session was created with it",
                    ));
                    if let Some(profile) = &vulkan.profile_path {
                        let (tally, note) = read_profile(profile);
                        provider_tally = tally;
                        profile_note = note;
                        if !config.keep_profile {
                            let _ = std::fs::remove_file(profile);
                        }
                    }
                    vulkan_arm = Some(vulkan);
                }
                Err(e) => {
                    guards.push(Guard::new(
                        "vulkan_ep_in_session",
                        false,
                        format!("{}: {}", e.token(), e.message),
                    ));
                }
            }
        }

        counters = Counters::read(&counters_path);
        if !config.keep_profile {
            let _ = std::fs::remove_file(&counters_path);
        }
    }

    // -- Guard 4: ORT's own profile attributed nodes to us ----------------------------------
    let vulkan_nodes = provider_tally.get(EP_NAME).copied().unwrap_or(0);
    let total_nodes: u64 = provider_tally.values().sum();
    guards.push(Guard::new(
        "vulkan_executed_nodes",
        vulkan_nodes > 0,
        if vulkan_nodes > 0 {
            format!(
                "ORT's profile attributed {vulkan_nodes} of {total_nodes} node executions to \
                 {EP_NAME}: {:?}",
                provider_tally
            )
        } else if total_nodes == 0 {
            format!(
                "ORT's profile attributed no node executions to any provider. {profile_note} \
                 Without the primary witness this run cannot claim the EP executed anything."
            )
        } else {
            format!(
                "ORT's profile attributed 0 of {total_nodes} node executions to {EP_NAME}: {:?}. \
                 The model ran entirely on another provider -- outputs would agree with CPU \
                 because they *are* CPU.",
                provider_tally
            )
        },
    ));

    // -- Guard 5: our own counter corroborates ----------------------------------------------
    let dispatches = counters.dispatches_executed;
    guards.push(Guard::new(
        "vulkan_dispatched_work",
        dispatches.unwrap_or(0) > 0,
        match dispatches {
            Some(n) if n > 0 => format!(
                "dispatches_executed={n} (claimed_nodes={:?}, islands_offered={:?})",
                counters.claimed_nodes, counters.islands_offered
            ),
            Some(n) => format!(
                "dispatches_executed={n}: the EP was in the session but submitted no compute work"
            ),
            None => format!(
                "no dispatches_executed reading. {} This is an absent instrument, not a zero.",
                counters.note
            ),
        },
    ));

    // -- Guard 5b: the device the session OPENED is the device that was ADVERTISED (issue #18) --
    //
    // Every device fact in this document up to here describes what ORT *enumerated*. Enumeration
    // is not selection: the runner can list four devices, ORT can bind one, and the EP session can
    // open a third — that is exactly the §6.5 index-space defect, and a document that records only
    // the enumeration cannot tell you it happened. So the EP reports the identity it actually
    // opened through its counters, and this compares the two.
    //
    // It fails **closed**. A run whose selected device cannot be matched to an advertised one is
    // not a run with a caveat, it is a run whose numbers cannot be attributed to any hardware; the
    // previous behaviour of warning and continuing let an unattributable MATCH be read as
    // evidence about the device the operator asked for.
    let (identity_guard, selected_device) =
        device_identity_agreement(&counters, &our_devices, config);
    let identity_diverged = !identity_guard.held;
    if !config.cpu_only {
        guards.push(identity_guard);
    }

    // Disagreement between the primary and corroborating witnesses is itself a finding.
    //
    // Issue #18 widens it: a device-identity divergence is a split frame in the same sense — the
    // numbers in this document and the hardware they are attributed to are not the same subject.
    let split_frame = (vulkan_nodes > 0) != (dispatches.unwrap_or(0) > 0)
        || (!config.cpu_only && identity_diverged);

    // -- Guard 6: the numbers ----------------------------------------------------------------
    if let Some(vulkan) = &vulkan_arm {
        let tol = compare::resolve(&verified.name, config.rtol, config.atol)?;
        if cpu.outputs.len() != vulkan.outputs.len() {
            guards.push(Guard::new(
                "outputs_agree",
                false,
                format!(
                    "CPU produced {} outputs and Vulkan produced {}",
                    cpu.outputs.len(),
                    vulkan.outputs.len()
                ),
            ));
        } else {
            for ((name, w, ws, wt), (_, v, vs, vt)) in cpu.outputs.iter().zip(vulkan.outputs.iter())
            {
                comparisons.push(compare::compare_output(name, w, ws, *wt, v, vs, *vt, &tol));
            }
            let all_pass = comparisons.iter().all(|c| c.verdict.is_pass());
            let worst = comparisons
                .iter()
                .max_by(|a, b| a.max_rel.total_cmp(&b.max_rel));
            guards.push(Guard::new(
                "outputs_agree",
                all_pass,
                match worst {
                    Some(c) => format!(
                        "{} output(s); worst {} on {:?}: {}",
                        comparisons.len(),
                        c.verdict.as_str(),
                        c.name,
                        c.detail
                    ),
                    None => "the model declares no outputs, so nothing was compared".to_string(),
                },
            ));
            if comparisons.is_empty() {
                // A model with no outputs cannot demonstrate correctness, and an empty `all()` is
                // vacuously true -- the exact shape of a vacuous pass.
                if let Some(g) = guards.last_mut() {
                    g.held = false;
                }
            }
        }
        tolerance = Some(tol);
    } else if !config.cpu_only {
        guards.push(Guard::new(
            "outputs_agree",
            false,
            "the Vulkan arm did not produce outputs, so there was nothing to compare",
        ));
    } else {
        guards.push(Guard::new(
            "outputs_agree",
            false,
            "--cpu-only: no Vulkan outputs exist to compare against the reference",
        ));
    }

    // Unregister before the environment goes: the EP dumps its counters at teardown, and a
    // process that exits first gets a snapshot written by nobody.
    if !config.cpu_only && ep_identity.is_some() {
        // Failure to unregister does not invalidate what was already measured; it is recorded
        // rather than thrown, because throwing here would discard a complete result.
        if let Some(f) = api.raw.UnregisterExecutionProviderLibrary {
            if let Ok(name) = crate::ortapi::cstring(EP_NAME) {
                // SAFETY: `env` is live, the name outlives the call, and every session created
                // from this library has already been dropped above.
                let status = unsafe { f(env.raw, name.as_ptr()) };
                if !status.is_null() {
                    if let Some(release) = api.raw.ReleaseStatus {
                        // SAFETY: non-null status returned by the call above, released once.
                        unsafe { release(status) };
                    }
                }
            }
        }
    }
    drop(env);

    let passed = guards.iter().all(|g| g.held);
    let outcome = if passed { Outcome::Pass } else { Outcome::Fail };

    let doc = Json::obj(vec![
        ("schema", Json::s("ort-model-runner/1")),
        ("tool", Json::s("rust/modelrunner (ort-model-runner)")),
        ("pass", Json::Bool(passed)),
        ("outcome", Json::s(outcome.as_str())),
        ("model", Json::s(verified.name.clone())),
        // Top-level identity, matching the convention every other artifact in this repo uses.
        ("onnx_file", Json::s(model_identity.path.clone())),
        ("onnx_sha256", Json::s(model_identity.sha256.clone())),
        ("onnx_bytes", Json::int(model_identity.bytes as i64)),
        ("model_provenance", Json::s(verified.provenance)),
        (
            "onnxruntime",
            Json::obj(vec![
                ("library", ort_identity.to_json()),
                ("api_version", Json::int(loaded.api_version as i64)),
                ("version", Json::s(loaded.version_string.clone())),
                ("discovered_via", Json::s(loaded.library.source.clone())),
            ]),
        ),
        (
            "execution_provider",
            Json::obj(vec![
                ("registration_name", Json::s(EP_NAME)),
                (
                    "library",
                    match &ep_identity {
                        Some(id) => id.to_json(),
                        None => Json::Null,
                    },
                ),
                ("devices", Json::Arr(devices_seen)),
                // What ORT *offered*. `selected_device` below is what the session *opened* — the
                // two are different facts and only the second one attributes these numbers.
                ("selected_device", selected_device),
                (
                    "device_selector",
                    match &config.device_selector {
                        Some(s) => Json::s(s.clone()),
                        None => Json::Null,
                    },
                ),
            ]),
        ),
        (
            "inputs",
            Json::Arr(feed_records.iter().map(|r| r.to_json()).collect()),
        ),
        (
            "free_dim_pins",
            Json::Arr(dim_pins.iter().map(|p| p.to_json()).collect()),
        ),
        ("seed", Json::s(format!("0x{:016X}", config.seed))),
        (
            "tolerance",
            match &tolerance {
                Some(t) => t.to_json(),
                None => Json::Null,
            },
        ),
        (
            "attribution",
            Json::obj(vec![
                ("primary_witness", Json::s("onnxruntime profile (cat=Node)")),
                (
                    "corroborating_witness",
                    Json::s("onnxruntime-ep-vulkan counters (dispatches_executed)"),
                ),
                (
                    "provider_node_counts",
                    Json::Obj(
                        provider_tally
                            .iter()
                            .map(|(k, v)| (k.clone(), Json::int(*v as i64)))
                            .collect(),
                    ),
                ),
                ("profile_note", Json::s(profile_note)),
                ("counters", counters.to_json()),
                ("split_frame", Json::Bool(split_frame)),
            ]),
        ),
        (
            "outputs",
            Json::Arr(comparisons.iter().map(|c| c.to_json()).collect()),
        ),
        (
            "timings_ms",
            Json::obj(vec![
                ("cpu_session_load", Json::n(cpu.load_ms)),
                ("cpu_run", Json::n(cpu.run_ms)),
                (
                    "vulkan_session_load",
                    match &vulkan_arm {
                        Some(v) => Json::n(v.load_ms),
                        None => Json::Null,
                    },
                ),
                (
                    "vulkan_run",
                    match &vulkan_arm {
                        Some(v) => Json::n(v.run_ms),
                        None => Json::Null,
                    },
                ),
                ("total", Json::n(now_ms(started))),
            ]),
        ),
        (
            "guards",
            Json::Arr(guards.iter().map(|g| g.to_json()).collect()),
        ),
    ]);

    Ok((outcome, doc))
}

/// Render the human-readable summary. Kept separate from the document so the two cannot drift:
/// every line printed here is read back out of the JSON that was written.
pub fn summarize(doc: &Json) -> String {
    let mut out = String::new();
    let model = doc.str_of("model").unwrap_or("?");
    let outcome = doc.str_of("outcome").unwrap_or("?");
    out.push_str(&format!("{outcome}  {model}\n"));
    out.push_str(&format!(
        "  onnx_file   {}\n  onnx_sha256 {}\n",
        doc.str_of("onnx_file").unwrap_or("?"),
        doc.str_of("onnx_sha256").unwrap_or("?")
    ));
    if let Some(guards) = doc.get("guards").and_then(Json::as_array) {
        for g in guards {
            out.push_str(&format!(
                "  [{}] {:<26} {}\n",
                if g.get("held") == Some(&Json::Bool(true)) {
                    "ok"
                } else {
                    "XX"
                },
                g.str_of("name").unwrap_or("?"),
                g.str_of("detail").unwrap_or("")
            ));
        }
    }
    out
}

/// Exit code for an outcome, matching the severity contract in `error.rs`.
pub fn exit_code(outcome: Outcome) -> i32 {
    match outcome {
        Outcome::Pass => 0,
        Outcome::Fail => Severity::Condition.exit_code(),
        Outcome::Error => Severity::Instrument.exit_code(),
        Outcome::Unsupported => Severity::Unsupported.exit_code(),
    }
}

/// The verdict a comparison list implies, for callers that want it without the document.
pub fn worst_verdict(comparisons: &[OutputComparison]) -> Verdict {
    comparisons
        .iter()
        .map(|c| c.verdict)
        .find(|v| !v.is_pass())
        .unwrap_or(Verdict::Agree)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------------------------------
    // `device_identity_agreement` (issue #18, blocker 3 / contract C7).
    //
    // The rejected artifact recorded the ENUMERATED devices and nothing else, so an evidence
    // document could list four `OrtEpDevice`s while the session ran on a fifth, and the reader
    // had no way to tell. These tests pin the opposite property: the document names the device
    // that was OPENED, and every state in which that cannot be established is RED.
    //
    // The asymmetry is the point. There is no branch below in which "we could not tell" is green.
    // ---------------------------------------------------------------------------------------

    const UUID_A: &str = "11111111111111111111111111111111";
    const UUID_B: &str = "22222222222222222222222222222222";
    const NAME: &str = "NVIDIA RTX A1000 Laptop GPU";

    fn advertised(pairs: &[(usize, Option<&str>)]) -> Vec<crate::ortapi::EpDeviceInfo> {
        pairs
            .iter()
            .map(|(index, uuid)| crate::ortapi::EpDeviceInfo {
                index: *index,
                raw: std::ptr::null(),
                ep_name: EP_NAME.to_string(),
                ep_vendor: "onnxruntime-ep-vulkan".to_string(),
                hardware_type: "GPU".to_string(),
                vendor_id: 0x10de,
                device_id: 0x27a0,
                uuid: uuid.map(str::to_string),
                luid: None,
                pci: None,
            })
            .collect()
    }

    /// A counters snapshot for a run that opened exactly the devices listed.
    fn counters_for(devices: &[(&str, &str)], frame: &str, declared: i64) -> Counters {
        let join = |f: &dyn Fn(usize, &(&str, &str)) -> String| {
            devices
                .iter()
                .enumerate()
                .map(|(i, d)| f(i, d))
                .collect::<Vec<_>>()
                .join("; ")
        };
        Counters {
            present: true,
            dispatches_executed: Some(1),
            claimed_nodes: Some(1),
            islands_offered: Some(1),
            compute_calls: Some(1),
            running_device_names: Some(join(&|i, d| format!("{i}={}", d.1))),
            running_device_uuids: Some(join(&|i, d| format!("{i}=uuid:{}", d.0))),
            alloc_device_frame: Some(frame.to_string()),
            alloc_device_frames_declared: Some(declared),
            note: String::new(),
        }
    }

    fn cfg(selector: Option<&str>) -> RunConfig {
        RunConfig {
            device_selector: selector.map(str::to_string),
            ..RunConfig::default()
        }
    }

    fn green(g: &Guard) -> bool {
        g.held
    }

    fn detail(g: &Guard) -> String {
        g.detail.clone()
    }

    #[test]
    fn one_opened_device_that_ort_advertised_is_agreement() {
        let (g, doc) = device_identity_agreement(
            &counters_for(&[(UUID_A, NAME)], "SHARED", 1),
            &advertised(&[(0, Some(UUID_B)), (1, Some(UUID_A))]),
            &cfg(None),
        );
        assert!(green(&g), "{}", detail(&g));
        assert_eq!(
            doc.get("identity"),
            Some(&Json::s(format!("uuid:{UUID_A}"))),
            "the document must name the device that was OPENED, by identity"
        );
        assert_eq!(
            doc.get("advertised_index"),
            Some(&Json::int(1)),
            "and say which advertised OrtEpDevice that was — index 1, not the enumeration order"
        );
        assert_eq!(doc.get("name"), Some(&Json::s(NAME)));
    }

    #[test]
    fn a_selector_that_names_the_opened_device_is_agreement_and_is_recorded() {
        let (g, doc) = device_identity_agreement(
            &counters_for(&[(UUID_A, NAME)], "SHARED", 1),
            &advertised(&[(0, Some(UUID_A))]),
            &cfg(Some(&format!("uuid:{UUID_A}"))),
        );
        assert!(green(&g), "{}", detail(&g));
        assert!(
            detail(&g).contains("--device-selector"),
            "a green guard on a selected run must say the selector was honoured: {}",
            detail(&g)
        );
        assert_eq!(
            doc.get("requested_selector"),
            Some(&Json::s(format!("uuid:{UUID_A}"))),
            "the request belongs in the document beside the outcome"
        );
    }

    /// Blocker 3: divergence FAILS CLOSED. The rejected artifact warned and continued.
    #[test]
    fn a_selector_that_names_a_different_device_than_the_one_opened_fails_closed() {
        let (g, _) = device_identity_agreement(
            &counters_for(&[(UUID_A, NAME)], "SHARED", 1),
            &advertised(&[(0, Some(UUID_A)), (1, Some(UUID_B))]),
            &cfg(Some(&format!("uuid:{UUID_B}"))),
        );
        assert!(
            !green(&g),
            "the run answered a question about card A while being asked about card B; that is \
             not a warning, it is an unattributed measurement"
        );
        let d = detail(&g);
        assert!(d.contains(UUID_A) && d.contains(UUID_B), "{d}");
    }

    #[test]
    fn a_run_that_reports_no_identity_is_unknown_never_agreed() {
        let mut c = counters_for(&[(UUID_A, NAME)], "SHARED", 1);
        c.running_device_uuids = None;
        let (g, doc) = device_identity_agreement(&c, &advertised(&[(0, Some(UUID_A))]), &cfg(None));
        assert!(
            !green(&g),
            "an EP build that predates the identity surface reports nothing; treating silence as \
             agreement is exactly how the enumerated list got read as the selected device"
        );
        assert_eq!(doc.get("identity"), Some(&Json::Null));
        assert!(
            detail(&g).contains("running_device_uuids"),
            "{}",
            detail(&g)
        );
    }

    #[test]
    fn an_absent_counters_snapshot_is_an_absent_instrument() {
        let (g, _) = device_identity_agreement(
            &Counters {
                present: false,
                note: "no counters snapshot".to_string(),
                ..Default::default()
            },
            &advertised(&[(0, Some(UUID_A))]),
            &cfg(None),
        );
        assert!(!green(&g));
        assert!(detail(&g).contains("absent instrument"), "{}", detail(&g));
    }

    #[test]
    fn two_opened_devices_cannot_be_attributed_to_either_of_them() {
        let (g, _) = device_identity_agreement(
            &counters_for(&[(UUID_A, NAME), (UUID_B, NAME)], "MIXED", 2),
            &advertised(&[(0, Some(UUID_A)), (1, Some(UUID_B))]),
            &cfg(None),
        );
        assert!(!green(&g));
        let d = detail(&g);
        assert!(
            d.contains(UUID_A) && d.contains(UUID_B),
            "both devices must be named or the failure is undiagnosable: {d}"
        );
    }

    /// The identical-name case, at the evidence layer. Both cards report the same `deviceName`;
    /// on the rejected bytes the frame collapsed to `SHARED` and this guard would have gone
    /// green on a two-device run.
    #[test]
    fn two_cards_of_the_same_model_do_not_agree_merely_because_their_names_match() {
        let mut c = counters_for(&[(UUID_A, NAME), (UUID_B, NAME)], "MIXED", 2);
        // The names are byte-identical — the only thing that separates these rows is the identity.
        c.running_device_names = Some(format!("0={NAME}; 1={NAME}"));
        assert_eq!(c.session_device_keys().len(), 2);
        let (g, _) = device_identity_agreement(
            &c,
            &advertised(&[(0, Some(UUID_A)), (1, Some(UUID_B))]),
            &cfg(None),
        );
        assert!(
            !green(&g),
            "identical names must not be read as one device: {}",
            detail(&g)
        );
    }

    #[test]
    fn a_mixed_or_split_frame_fails_even_when_one_identity_was_reported() {
        for frame in ["MIXED", "SPLIT-DEVICE"] {
            let (g, _) = device_identity_agreement(
                &counters_for(
                    &[(UUID_A, NAME)],
                    frame,
                    if frame == "MIXED" { 2 } else { 1 },
                ),
                &advertised(&[(0, Some(UUID_A))]),
                &cfg(None),
            );
            assert!(
                !green(&g),
                "frame {frame} means the allocator numbers and the session describe different \
                 devices: {}",
                detail(&g)
            );
        }
    }

    #[test]
    fn an_opened_device_that_was_never_advertised_fails_closed() {
        let (g, doc) = device_identity_agreement(
            &counters_for(&[(UUID_A, NAME)], "SHARED", 1),
            &advertised(&[(0, Some(UUID_B))]),
            &cfg(None),
        );
        assert!(!green(&g));
        assert_eq!(
            doc.get("advertised_index"),
            Some(&Json::Null),
            "no advertised device matches, and the document must say so rather than guess 0"
        );
    }

    #[test]
    fn advertised_devices_without_uuid_metadata_cannot_confirm_anything() {
        let (g, _) = device_identity_agreement(
            &counters_for(&[(UUID_A, NAME)], "SHARED", 1),
            &advertised(&[(0, None), (1, None)]),
            &cfg(None),
        );
        assert!(
            !green(&g),
            "a host too old to carry vulkan.device_uuid leaves nothing to compare against; that \
             is unknown, not agreed: {}",
            detail(&g)
        );
        assert!(detail(&g).contains("no uuid metadata"), "{}", detail(&g));
    }

    #[test]
    fn no_advertised_devices_at_all_fails_closed() {
        let (g, _) = device_identity_agreement(
            &counters_for(&[(UUID_A, NAME)], "SHARED", 1),
            &[],
            &cfg(None),
        );
        assert!(!green(&g));
    }

    #[test]
    fn a_cpu_only_run_never_claims_a_device() {
        let mut c = cfg(None);
        c.cpu_only = true;
        let (g, _) = device_identity_agreement(
            &counters_for(&[(UUID_A, NAME)], "SHARED", 1),
            &advertised(&[(0, Some(UUID_A))]),
            &c,
        );
        assert!(!green(&g), "{}", detail(&g));
        assert!(detail(&g).contains("--cpu-only"), "{}", detail(&g));
    }

    #[test]
    fn the_selected_device_object_always_records_how_the_device_was_chosen() {
        let (_, none) = device_identity_agreement(
            &counters_for(&[(UUID_A, NAME)], "SHARED", 1),
            &advertised(&[(0, Some(UUID_A))]),
            &cfg(None),
        );
        assert_eq!(none.get("requested_selector"), Some(&Json::Null));
        match none.get("selector_source") {
            Some(Json::Str(s)) => assert!(s.contains("ORT's binding"), "{s}"),
            other => panic!("selector_source must be a string, got {other:?}"),
        }

        let (_, sel) = device_identity_agreement(
            &counters_for(&[(UUID_A, NAME)], "SHARED", 1),
            &advertised(&[(0, Some(UUID_A))]),
            &cfg(Some("uuid:11111111111111111111111111111111")),
        );
        match sel.get("selector_source") {
            Some(Json::Str(s)) => assert!(
                s.contains("before EP registration"),
                "the document must state WHERE the selection happened — the env var is read once, \
                 before `register_ep_library`, which is what makes it the single authoritative \
                 path: {s}"
            ),
            other => panic!("selector_source must be a string, got {other:?}"),
        }
    }

    #[test]

    fn the_default_seed_is_fixed_rather_than_drawn_from_a_clock() {
        // Two configs built a moment apart must feed identical bytes, or a reported disagreement
        // cannot be reproduced from the command line that produced it.
        let a = RunConfig::default();
        let b = RunConfig::default();
        assert_eq!(a.seed, b.seed);
        assert_ne!(a.seed, 0);
    }

    #[test]
    fn the_ep_library_name_is_the_platform_one() {
        let p = default_ep_lib(Path::new("/repo"));
        let name = p.file_name().unwrap().to_string_lossy().into_owned();
        if cfg!(target_os = "windows") {
            assert_eq!(name, "onnxruntime_vulkan_ep.dll");
        } else if cfg!(target_os = "macos") {
            assert_eq!(name, "libonnxruntime_vulkan_ep.dylib");
        } else {
            assert_eq!(name, "libonnxruntime_vulkan_ep.so");
        }
        assert!(p.ends_with(Path::new("rust/target/release").join(name)));
    }

    #[test]
    fn the_registration_name_is_the_one_the_plugin_publishes() {
        assert_eq!(EP_NAME, "VulkanExecutionProvider");
    }

    #[test]
    fn exit_codes_distinguish_a_failed_claim_from_a_broken_instrument() {
        assert_eq!(exit_code(Outcome::Pass), 0);
        assert_ne!(exit_code(Outcome::Fail), exit_code(Outcome::Error));
        assert_ne!(exit_code(Outcome::Fail), 0);
        assert_ne!(exit_code(Outcome::Unsupported), 0);
    }

    #[test]
    fn the_summary_is_rendered_from_the_document_not_from_the_run() {
        let doc = Json::obj(vec![
            ("model", Json::s("mnist-12")),
            ("outcome", Json::s("FAIL")),
            ("onnx_file", Json::s("/x/mnist-12.onnx")),
            ("onnx_sha256", Json::s("abc")),
            (
                "guards",
                Json::Arr(vec![
                    Guard::new("vulkan_executed_nodes", false, "0 of 12").to_json(),
                    Guard::new("model_identity_pinned", true, "ok").to_json(),
                ]),
            ),
        ]);
        let text = summarize(&doc);
        assert!(text.contains("FAIL  mnist-12"), "{text}");
        assert!(text.contains("[XX] vulkan_executed_nodes"), "{text}");
        assert!(text.contains("[ok] model_identity_pinned"), "{text}");
        assert!(text.contains("abc"), "{text}");
    }

    #[test]
    fn the_manifest_path_is_the_repository_pin_file() {
        assert!(
            manifest_path(Path::new("/repo"))
                .ends_with(Path::new("bench/results/model_provenance.json"))
        );
    }

    #[test]
    fn worst_verdict_reports_the_first_failure_not_the_last_success() {
        let mk = |v: Verdict| OutputComparison {
            name: "y".into(),
            verdict: v,
            dtype: "float32".into(),
            reference_shape: vec![1],
            candidate_shape: vec![1],
            elements: 1,
            max_abs: 0.0,
            max_rel: 0.0,
            worst_index: 0,
            reference_at_worst: 0.0,
            candidate_at_worst: 0.0,
            nan_reference: 0,
            nan_candidate: 0,
            reference_all_zero: false,
            detail: String::new(),
        };
        assert_eq!(
            worst_verdict(&[mk(Verdict::Agree), mk(Verdict::Disagree)]),
            Verdict::Disagree
        );
        assert_eq!(worst_verdict(&[mk(Verdict::Exact)]), Verdict::Agree);
        assert_eq!(worst_verdict(&[]), Verdict::Agree);
    }
}
