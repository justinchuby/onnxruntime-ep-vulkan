//! End-to-end tests for the `ort-model-runner` binary.
//!
//! These drive the real executable rather than the library, because the properties they protect
//! are properties of the *command*: its exit codes, its refusals, and the fact that it writes
//! evidence on failure as well as on success. A library test cannot observe an exit code, and the
//! exit code is what CI reads.
//!
//! # Every test declares its whole world
//!
//! Issue #39: `an_explicit_ort_library_that_does_not_exist_never_falls_back_to_a_search` passed on
//! every developer machine and failed on every clean CI runner, because it depended on `mnist-12`
//! already being in the ambient model cache. Three more tests had the same dependency and did not
//! fail — they *passed for the wrong reason*, asserting only "some non-zero exit" and getting one
//! from `model_not_cached` instead of from the thing they were named after.
//!
//! So the rule here is: **a test that does not construct a piece of state does not depend on it.**
//! [`World`] builds a private model cache and a private stand-in for the ONNX Runtime library, and
//! [`World::run`] scrubs every environment variable the discovery path consults. Nothing reads
//! `$HOME`, `$ORT_HOME`, an ambient cache, the repository's `.venv`, or `PATH`. A run on a machine
//! that has never seen this project produces the same result as a run on the machine that wrote
//! it, which is the only property that makes a red CI lane informative.
//!
//! Nothing here needs a GPU, an ONNX Runtime, or a network. The live real-model arm is a separate
//! test that skips unless it is deliberately switched on -- see `live_model_arm` at the bottom.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_ort-model-runner"))
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ort_model_runner_cli_{tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Every environment variable that can change what [`ortlib::Search::from_environment`] finds or
/// where the model cache is. Cleared for every run so that "what this machine happens to have
/// installed" is never an input.
const AMBIENT: &[&str] = &[
    "ORT_MODEL_RUNNER_ORT_LIB",
    "ORT_HOME",
    "ORT_DIR",
    "ONNXRUNTIME_DIR",
    "ORT_LIB_DIR",
    "VIRTUAL_ENV",
    "ONNXRUNTIME_EP_VULKAN_MODEL_CACHE",
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
];

/// A completely declared environment for one test.
///
/// `ort_stand_in` is a real file that is not a real ONNX Runtime. That is exactly what the tests
/// below need: `discover` only asks whether the explicit path *is a file*, so a stand-in makes the
/// instrument half of the run succeed deterministically without installing anything, and every
/// assertion here is about a refusal that happens before the library would be loaded.
struct World {
    dir: PathBuf,
    cache: PathBuf,
    ort_stand_in: PathBuf,
}

impl World {
    fn new(tag: &str) -> Self {
        Self::at(scratch(tag))
    }

    fn at(dir: PathBuf) -> Self {
        let cache = dir.join("model cache");
        std::fs::create_dir_all(&cache).unwrap();
        let ort_stand_in = dir.join("stand in onnxruntime.bin");
        std::fs::write(&ort_stand_in, b"not an onnx runtime, but it is a file").unwrap();
        Self {
            dir,
            cache,
            ort_stand_in,
        }
    }

    /// Put a file where the cache expects `name`, with whatever bytes the caller wants.
    fn plant(&self, name: &str, bytes: &[u8]) -> PathBuf {
        let p = self.cache.join(format!("{name}.onnx"));
        std::fs::write(&p, bytes).unwrap();
        p
    }

    /// Run the binary with this world's cache, and nothing else from the host.
    fn run(&self, args: &[&str]) -> Output {
        let mut cmd = Command::new(binary());
        cmd.args(args);
        for key in AMBIENT {
            cmd.env_remove(key);
        }
        cmd.env("ONNXRUNTIME_EP_VULKAN_MODEL_CACHE", &self.cache);
        cmd.output().expect("the runner binary must be executable")
    }

    /// The same, with `--ort-lib` pointed at the stand-in, so the instrument resolves and any
    /// refusal that follows is about the subject.
    fn run_with_sound_instrument(&self, args: &[&str]) -> Output {
        let mut full: Vec<&str> = args.to_vec();
        let lib = self.ort_stand_in.to_str().unwrap();
        full.extend_from_slice(&["--ort-lib", lib]);
        self.run(&full)
    }
}

impl Drop for World {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// Run with no world at all: for the parse-time tests, which never reach discovery.
fn run(args: &[&str]) -> Output {
    let mut cmd = Command::new(binary());
    cmd.args(args);
    for key in AMBIENT {
        cmd.env_remove(key);
    }
    cmd.output().expect("the runner binary must be executable")
}

fn text(out: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

/// The repository root, found the same way the runner finds it.
fn repo_root() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    loop {
        if dir.join("bench/results/model_provenance.json").is_file() {
            return dir;
        }
        if !dir.pop() {
            panic!("cannot find the repository root from CARGO_MANIFEST_DIR");
        }
    }
}

/// The pinned SHA-256 and byte count of `mnist-12`, read from the manifest the runner reads.
///
/// Spelled once, from the file that defines it, so a re-pin does not leave a test asserting a
/// digest that no longer means anything.
fn mnist_pin() -> (String, usize) {
    let manifest =
        std::fs::read_to_string(repo_root().join("bench/results/model_provenance.json")).unwrap();
    let at = manifest.find("\"mnist-12\"").expect("mnist-12 is pinned");
    let rest = &manifest[at..];
    let field = |key: &str| -> String {
        let k = rest
            .find(key)
            .unwrap_or_else(|| panic!("{key} near mnist-12"));
        let after = &rest[k + key.len()..];
        let start = after.find(|c: char| c.is_ascii_alphanumeric()).unwrap();
        let tail = &after[start..];
        let end = tail
            .find(|c: char| !c.is_ascii_alphanumeric())
            .unwrap_or(tail.len());
        tail[..end].to_string()
    };
    let sha = field("\"sha256\"");
    let bytes: usize = field("\"bytes\"").parse().unwrap();
    (sha, bytes)
}

#[test]
fn help_exits_zero_and_documents_every_flag_the_parser_accepts() {
    let out = run(&["--help"]);
    assert_eq!(out.status.code(), Some(0));
    let help = text(&out);
    // A flag the parser accepts but --help never mentions is a flag nobody can discover, and one
    // that --help mentions but the parser rejects is worse. Both directions are checked against
    // the source of the parser itself.
    let source =
        std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("src/main.rs")).unwrap();
    for flag in [
        "--check-model-agreement",
        "--model-path",
        "--foundry-model",
        "--ep-lib",
        "--ort-lib",
        "--out",
        "--seed",
        "--free-dim",
        "--input",
        "--rtol",
        "--atol",
        "--fetch",
        "--cpu-only",
        "--keep-profile",
        "--quiet",
        "--list-models",
        "--list-devices",
    ] {
        assert!(help.contains(flag), "--help does not document {flag}");
        assert!(
            source.contains(&format!("\"{flag}\"")),
            "the parser does not accept {flag}"
        );
    }
}

#[test]
fn no_arguments_prints_usage_rather_than_doing_something() {
    let out = run(&[]);
    assert_eq!(out.status.code(), Some(0));
    assert!(text(&out).contains("USAGE"));
}

#[test]
fn an_unknown_flag_is_an_instrument_error_not_a_silent_ignore() {
    let out = run(&["--check-model-agreement", "mnist-12", "--turbo"]);
    assert_eq!(out.status.code(), Some(2), "{}", text(&out));
    assert!(text(&out).contains("unknown_argument"), "{}", text(&out));
}

#[test]
fn a_flag_missing_its_value_is_refused_rather_than_defaulted() {
    let out = run(&["--check-model-agreement"]);
    assert_eq!(out.status.code(), Some(2), "{}", text(&out));
    assert!(text(&out).contains("missing_argument"), "{}", text(&out));
}

#[test]
fn list_models_shows_the_pinned_models_and_their_tolerances() {
    let out = run(&["--list-models"]);
    assert_eq!(out.status.code(), Some(0), "{}", text(&out));
    let listing = text(&out);
    assert!(listing.contains("mnist-12"), "{listing}");
    assert!(listing.contains("mobilenetv2-12"), "{listing}");
    // The pinned hash is the identity; a listing without it is a name, not a pin.
    assert!(
        listing.contains("5c688690f8bacf667d4c2074af5ad0646ca328d7ab03eccf944a65b320171bdd"),
        "{listing}"
    );
    assert!(listing.contains("rtol="), "{listing}");
}

#[test]
fn an_explicit_ort_library_that_does_not_exist_never_falls_back_to_a_search() {
    // The defect this guards: silently searching after an explicit `--ort-lib` misses means the
    // run happens against a library the caller did not name, and the evidence records a path the
    // caller never asked for.
    //
    // The cache is deliberately EMPTY. That is the shape of a fresh CI runner, and before issue
    // #39 this exact test failed there -- not because the guard was broken but because the run
    // died at `model_not_cached` first and never reached the library at all. A test whose subject
    // is only reachable when an unrelated cache happens to be populated is not a test.
    let world = World::new("explicit_ort_missing");
    let out = world.run(&[
        "--check-model-agreement",
        "mnist-12",
        "--ort-lib",
        "definitely-not-a-real-library.dll",
    ]);
    let t = text(&out);
    assert_eq!(out.status.code(), Some(2), "{t}");
    assert!(t.contains("ort_library_missing"), "{t}");
    assert!(
        !t.contains("model_not_cached"),
        "the instrument fault was reported behind a subject fault: {t}"
    );
}

#[test]
fn a_broken_instrument_is_reported_before_an_absent_subject() {
    // The ordering itself, in both polarities, because either alone is satisfiable by accident.
    //
    // Polarity 1: nothing resolves. An empty cache *and* an unresolvable library must report the
    // library, because a measurement whose instrument is broken says nothing about its subject.
    let world = World::new("order_instrument_first");
    let out = world.run(&[
        "--check-model-agreement",
        "mnist-12",
        "--ort-lib",
        "no-such-onnxruntime.dll",
    ]);
    let t = text(&out);
    assert!(t.contains("ort_library_missing"), "{t}");
    assert!(!t.contains("model_not_cached"), "{t}");

    // Polarity 2: the instrument resolves, so the *subject* is now the thing worth reporting. If
    // this arm also said `ort_library_missing`, the ordering above would be a tautology.
    let out = world.run_with_sound_instrument(&["--check-model-agreement", "mnist-12"]);
    let t = text(&out);
    assert_eq!(out.status.code(), Some(2), "{t}");
    assert!(t.contains("model_not_cached"), "{t}");
    assert!(!t.contains("ort_library_missing"), "{t}");
}

#[test]
fn a_clean_runner_with_no_ort_anywhere_says_so_rather_than_guessing() {
    // No `--ort-lib`, no `$ORT_HOME`, no venv, no `PATH` entry that this test put there: the
    // state of a machine that has never had ONNX Runtime installed. The runner must name the
    // instrument it could not find and list where it looked, not fall through to something else.
    let world = World::new("no_ort_anywhere");
    let mut cmd = Command::new(binary());
    cmd.args(["--check-model-agreement", "mnist-12"]);
    for key in AMBIENT {
        cmd.env_remove(key);
    }
    // `PATH` cannot be emptied -- the process must still be able to start -- so it is pointed at
    // a directory that exists and contains no library.
    cmd.env("PATH", world.dir.join("empty"));
    std::fs::create_dir_all(world.dir.join("empty")).unwrap();
    cmd.env("ONNXRUNTIME_EP_VULKAN_MODEL_CACHE", &world.cache);
    let out = cmd.output().unwrap();
    let t = text(&out);
    assert_eq!(out.status.code(), Some(2), "{t}");
    assert!(t.contains("ort_library_missing"), "{t}");
    assert!(
        t.contains("--ort-lib"),
        "the refusal must say how to resolve it: {t}"
    );
}

#[test]
fn a_model_that_is_not_cached_refuses_before_it_touches_the_network() {
    let world = World::new("empty_cache");
    let out = world.run_with_sound_instrument(&["--check-model-agreement", "mnist-12"]);
    let t = text(&out);
    assert_eq!(out.status.code(), Some(2), "{t}");
    assert!(t.contains("model_not_cached"), "{t}");
    // Downloading without being told to is how a "verification" run quietly becomes a fetch.
    assert!(t.contains("--fetch"), "{t}");
}

#[test]
fn a_cached_file_whose_bytes_do_not_match_the_pin_is_never_run() {
    // Two planted negatives, because the pin has two independent halves and a check that only
    // looks at one of them is half a check. Both expectations are read from the manifest rather
    // than typed in, so a re-pin cannot leave this test asserting a dead digest.
    let (sha, bytes) = mnist_pin();

    // 1. Wrong size: caught before anything is hashed.
    let world = World::new("bad_size");
    world.plant("mnist-12", b"this is not mnist");
    let out = world.run_with_sound_instrument(&["--check-model-agreement", "mnist-12"]);
    let t = text(&out);
    assert_eq!(out.status.code(), Some(1), "{t}");
    assert!(t.contains("provenance_size_mismatch"), "{t}");
    assert!(
        t.contains(&bytes.to_string()),
        "the refusal must name the size expected: {t}"
    );

    // 2. Right size, wrong bytes: the case a size check alone would wave through. Without the
    // hash this file would load, run, agree with itself, and be reported as mnist-12 passing.
    let world = World::new("bad_hash");
    world.plant("mnist-12", &vec![0u8; bytes]);
    let out = world.run_with_sound_instrument(&["--check-model-agreement", "mnist-12"]);
    let t = text(&out);
    assert_eq!(out.status.code(), Some(1), "{t}");
    assert!(t.contains("provenance_hash_mismatch"), "{t}");
    assert!(
        t.contains(&sha),
        "the refusal must name the hash that was expected: {t}"
    );
}

#[test]
fn a_model_nobody_has_pinned_is_refused_by_name_rather_than_measured_and_run() {
    // Previously named `a_model_with_no_tolerance_policy_is_refused_rather_than_given_a_default`
    // and asserting only `!= 0`, which it satisfied via `model_not_cached` on a clean machine --
    // a test passing for a reason unrelated to its name. The tolerance policy is consulted after
    // the CPU arm has run and so is not reachable from the CLI without a real ONNX Runtime; it is
    // covered by `compare`'s own unit tests. What *is* a CLI property is this: an unpinned name
    // is refused by name, with the known names listed.
    let world = World::new("unpinned_name");
    let out =
        world.run_with_sound_instrument(&["--check-model-agreement", "some-model-nobody-pinned"]);
    let t = text(&out);
    assert_eq!(out.status.code(), Some(2), "{t}");
    assert!(t.contains("model_not_pinned"), "{t}");
    assert!(
        t.contains("mnist-12"),
        "the refusal must list what is pinned: {t}"
    );
}

#[test]
fn a_bad_tolerance_or_seed_is_rejected_at_parse_time() {
    for (args, cause) in [
        (vec!["--rtol", "banana"], "bad_tolerance"),
        (vec!["--rtol", "-1"], "bad_tolerance"),
        (vec!["--seed", "not-a-number"], "bad_seed"),
        (vec!["--free-dim", "batch"], "bad_free_dim"),
        (vec!["--free-dim", "batch=0"], "bad_free_dim"),
        (vec!["--input", "justaname"], "bad_input"),
    ] {
        let mut full = vec!["--check-model-agreement", "mnist-12"];
        full.extend(args.iter().copied());
        let out = run(&full);
        assert_eq!(out.status.code(), Some(2), "{args:?} -> {}", text(&out));
        assert!(text(&out).contains(cause), "{args:?} -> {}", text(&out));
    }
}

#[test]
fn half_a_tolerance_override_is_refused_because_the_other_half_would_be_invisible() {
    // Previously this asserted only `!= 0` and got it from the library error, with a comment
    // saying so -- which meant the refusal it was named after was never observed at all, and in
    // fact was not reachable from the CLI: `compare::resolve` is not called until after the CPU
    // arm has executed. The check now also happens at parse time, so this asserts the token.
    let world = World::new("half_tolerance");
    for args in [
        vec!["--check-model-agreement", "mnist-12", "--rtol", "1e-3"],
        vec!["--check-model-agreement", "mnist-12", "--atol", "1e-5"],
    ] {
        let out = world.run_with_sound_instrument(&args);
        let t = text(&out);
        assert_eq!(out.status.code(), Some(2), "{args:?} -> {t}");
        assert!(t.contains("partial_tolerance_override"), "{args:?} -> {t}");
    }

    // Both halves together is not an error -- otherwise the test above would pass for a crate
    // that simply rejected `--rtol` outright.
    let out = world.run_with_sound_instrument(&[
        "--check-model-agreement",
        "mnist-12",
        "--rtol",
        "1e-3",
        "--atol",
        "1e-5",
    ]);
    let t = text(&out);
    assert!(
        !t.contains("partial_tolerance_override"),
        "a complete override was refused as a partial one: {t}"
    );
}

#[test]
fn an_unknown_foundry_alias_is_unsupported_rather_than_a_fuzzy_search() {
    let out = run(&["--foundry-model", "llama-ish"]);
    assert_eq!(out.status.code(), Some(3), "{}", text(&out));
    assert!(
        text(&out).contains("unknown_foundry_alias"),
        "{}",
        text(&out)
    );
}

#[test]
fn an_identity_failure_exits_before_writing_an_evidence_document_and_says_which_model() {
    // An evidence file that describes a run that did not happen is worse than no file, so the
    // boundary is: identity is checked first, and a file that is not the model has no run to
    // describe. Previously this test was named `evidence_is_written_on_failure_not_only_on_success`
    // and asserted only that the output mentioned "mnist-12" -- it never looked at the evidence
    // path at all, in either direction. Now it asserts the absence, which is the actual contract.
    // Evidence written for a *run* that failed is covered by `live_model_arm`, which is the only
    // arm that can get far enough to have a run to record.
    let world = World::new("identity_failure_no_document");
    let (_, bytes) = mnist_pin();
    world.plant("mnist-12", &vec![7u8; bytes]);
    let out_file = world.dir.join("result.json");
    let out = world.run_with_sound_instrument(&[
        "--check-model-agreement",
        "mnist-12",
        "--out",
        out_file.to_str().unwrap(),
    ]);
    let t = text(&out);
    assert_eq!(out.status.code(), Some(1), "{t}");
    assert!(t.contains("provenance_hash_mismatch"), "{t}");
    assert!(t.contains("mnist-12"), "{t}");
    assert!(
        !out_file.exists(),
        "a document was written for a run that never happened: {}",
        out_file.display()
    );
}

#[test]
fn a_cache_and_an_output_path_containing_spaces_are_handled() {
    // `World` puts every test in `model cache/` and `stand in onnxruntime.bin` precisely so this
    // is not a special case -- but the property deserves an assertion of its own, because the
    // failure mode (an argument split on the space by a shell that should not be involved) is
    // silent on the machine whose paths have none. `C:\Program Files` and
    // `/home/runner/work/my project` are both real.
    let dir = std::env::temp_dir().join("ort model runner cli spaces");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let world = World::at(dir);
    assert!(
        world.cache.display().to_string().contains(' '),
        "the fixture must actually contain a space to be testing anything"
    );
    let (sha, bytes) = mnist_pin();
    world.plant("mnist-12", &vec![0u8; bytes]);
    let out_file = world.dir.join("evidence for review.json");
    let out = world.run_with_sound_instrument(&[
        "--check-model-agreement",
        "mnist-12",
        "--out",
        out_file.to_str().unwrap(),
    ]);
    let t = text(&out);
    // Reaching the hash check means the cache path with a space resolved to the planted file.
    assert!(t.contains("provenance_hash_mismatch"), "{t}");
    assert!(t.contains(&sha), "{t}");
    assert!(
        t.contains("model cache"),
        "the message must name the path it actually used: {t}"
    );
}

#[test]
fn the_model_cache_override_is_obeyed_rather_than_merged_with_the_default() {
    // A cache override that is treated as one more place to look means a CI runner with a
    // populated home directory silently tests a different file than a clean one -- which is the
    // ambiguity that made issue #39's failure machine-dependent. The override is the whole cache.
    let world = World::new("cache_override_is_exclusive");
    let out = world.run_with_sound_instrument(&["--check-model-agreement", "mnist-12"]);
    let t = text(&out);
    assert!(t.contains("model_not_cached"), "{t}");
    assert!(
        t.contains("model cache"),
        "the refusal must name the overridden cache, not the default one: {t}"
    );
    let home_cache = ".cache";
    assert!(
        !t.contains(home_cache) || t.contains("model cache"),
        "the default cache leaked into an overridden run: {t}"
    );
}

/// The live arm: real ONNX Runtime, real plugin, real model, real GPU.
///
/// Skipped unless `ORT_MODEL_RUNNER_LIVE=1`, because a machine with no Vulkan device would report
/// a failure that is about the machine rather than about the code -- and a test that fails for a
/// reason unrelated to the change under review teaches people to ignore test failures. CI runs it
/// on the lane that has a device; `rust/README.md` documents the switch.
#[test]
fn live_model_arm() {
    if std::env::var("ORT_MODEL_RUNNER_LIVE").ok().as_deref() != Some("1") {
        eprintln!("skipping: set ORT_MODEL_RUNNER_LIVE=1 to run the real-model arm");
        return;
    }
    let dir = scratch("live");
    let out_file = dir.join("mnist-12.json");
    let mut args = vec![
        "--check-model-agreement".to_string(),
        "mnist-12".to_string(),
        "--out".to_string(),
        out_file.display().to_string(),
    ];
    if let Ok(lib) = std::env::var("ORT_MODEL_RUNNER_ORT_LIB") {
        args.push("--ort-lib".to_string());
        args.push(lib);
    }
    let out = Command::new(binary())
        .args(&args)
        .output()
        .expect("the runner binary must be executable");
    let t = text(&out);
    assert_eq!(out.status.code(), Some(0), "{t}");
    let doc = std::fs::read_to_string(&out_file).unwrap();
    // The five guards must be present *and* held; a document that merely says pass:true without
    // naming what it checked is the vacuous artifact this runner exists to replace.
    for guard in [
        "model_identity_pinned",
        "vulkan_ep_device_present",
        "vulkan_ep_in_session",
        "vulkan_executed_nodes",
        "vulkan_dispatched_work",
        "outputs_agree",
    ] {
        assert!(
            doc.contains(guard),
            "{guard} missing from the evidence:\n{doc}"
        );
    }
    assert!(doc.contains("\"pass\": true"), "{doc}");
    assert!(
        doc.contains("5c688690f8bacf667d4c2074af5ad0646ca328d7ab03eccf944a65b320171bdd"),
        "{doc}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
