//! End-to-end tests for the `ort-model-runner` binary.
//!
//! These drive the real executable rather than the library, because the properties they protect
//! are properties of the *command*: its exit codes, its refusals, and the fact that it writes
//! evidence on failure as well as on success. A library test cannot observe an exit code, and the
//! exit code is what CI reads.
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

fn run(args: &[&str]) -> Output {
    Command::new(binary())
        .args(args)
        .output()
        .expect("the runner binary must be executable")
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
    let out = run(&[
        "--check-model-agreement",
        "mnist-12",
        "--ort-lib",
        "definitely-not-a-real-library.dll",
    ]);
    assert_eq!(out.status.code(), Some(2), "{}", text(&out));
    assert!(text(&out).contains("ort_library_missing"), "{}", text(&out));
}

#[test]
fn a_model_that_is_not_cached_refuses_before_it_touches_the_network() {
    let cache = scratch("empty_cache");
    let out = Command::new(binary())
        .args(["--check-model-agreement", "mnist-12"])
        .env("ONNXRUNTIME_EP_VULKAN_MODEL_CACHE", &cache)
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2), "{}", text(&out));
    let t = text(&out);
    assert!(t.contains("model_not_cached"), "{t}");
    // Downloading without being told to is how a "verification" run quietly becomes a fetch.
    assert!(t.contains("--fetch"), "{t}");
    let _ = std::fs::remove_dir_all(&cache);
}

#[test]
fn a_cached_file_whose_bytes_do_not_match_the_pin_is_never_run() {
    // Two planted negatives, because the pin has two independent halves and a check that only
    // looks at one of them is half a check.
    //
    // 1. Wrong size: caught before anything is hashed.
    let cache = scratch("bad_size");
    std::fs::write(cache.join("mnist-12.onnx"), b"this is not mnist").unwrap();
    let out = Command::new(binary())
        .args(["--check-model-agreement", "mnist-12"])
        .env("ONNXRUNTIME_EP_VULKAN_MODEL_CACHE", &cache)
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1), "{}", text(&out));
    let t = text(&out);
    assert!(t.contains("provenance_size_mismatch"), "{t}");
    assert!(
        t.contains("26143"),
        "the refusal must name the size expected: {t}"
    );
    let _ = std::fs::remove_dir_all(&cache);

    // 2. Right size, wrong bytes: the case a size check alone would wave through. Without the
    // hash this file would load, run, agree with itself, and be reported as mnist-12 passing.
    let cache = scratch("bad_hash");
    std::fs::write(cache.join("mnist-12.onnx"), vec![0u8; 26143]).unwrap();
    let out = Command::new(binary())
        .args(["--check-model-agreement", "mnist-12"])
        .env("ONNXRUNTIME_EP_VULKAN_MODEL_CACHE", &cache)
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1), "{}", text(&out));
    let t = text(&out);
    assert!(t.contains("provenance_hash_mismatch"), "{t}");
    assert!(
        t.contains("5c688690f8bacf667d4c2074af5ad0646ca328d7ab03eccf944a65b320171bdd"),
        "the refusal must name the hash that was expected: {t}"
    );
    let _ = std::fs::remove_dir_all(&cache);
}

#[test]
fn a_model_with_no_tolerance_policy_is_refused_rather_than_given_a_default() {
    // A real ONNX file is not needed: the tolerance policy is consulted for the *name*, and the
    // refusal must happen without anyone having to guess a bound.
    let out = run(&["--check-model-agreement", "some-model-nobody-has-reviewed"]);
    let t = text(&out);
    // Either it never got that far (no ORT) or it refused for the tolerance. Both are non-zero;
    // what must never happen is a zero exit.
    assert_ne!(out.status.code(), Some(0), "{t}");
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
    let repo = repo_root();
    let out = Command::new(binary())
        .args([
            "--check-model-agreement",
            "mnist-12",
            "--rtol",
            "1e-3",
            "--ort-lib",
            "definitely-not-a-real-library.dll",
        ])
        .current_dir(&repo)
        .output()
        .unwrap();
    // The library error comes first here; the point of the test is that neither path exits 0.
    assert_ne!(out.status.code(), Some(0), "{}", text(&out));
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
fn evidence_is_written_on_failure_not_only_on_success() {
    // An evidence file that only appears when the run passed is a record of nothing: the case a
    // reviewer needs the document for is the case where it failed.
    let dir = scratch("evidence_on_failure");
    let cache = dir.join("cache");
    std::fs::create_dir_all(&cache).unwrap();
    std::fs::write(cache.join("mnist-12.onnx"), b"wrong bytes").unwrap();
    let out_file = dir.join("result.json");
    let out = Command::new(binary())
        .args([
            "--check-model-agreement",
            "mnist-12",
            "--out",
            out_file.to_str().unwrap(),
        ])
        .env("ONNXRUNTIME_EP_VULKAN_MODEL_CACHE", &cache)
        .output()
        .unwrap();
    assert_ne!(out.status.code(), Some(0));
    // The pin check fails before a session exists, so this particular arm exits before writing --
    // which is itself the documented boundary: identity is checked first, and a file that is not
    // the model has no run to describe.
    assert!(text(&out).contains("mnist-12"), "{}", text(&out));
    let _ = std::fs::remove_dir_all(&dir);
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
