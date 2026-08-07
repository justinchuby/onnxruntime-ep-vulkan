//! `ort-model-runner` -- the command-line face of the Rust-native real-model validation runner.
//!
//! Hand-rolled argument parsing, in the same style as `rust/src/bin/epctl.rs` and for the same
//! reason: this binary exists because a package index was unreachable, so it cannot be the thing
//! that needs a dependency to start.
//!
//! EXIT CODES
//! ----------
//! `0` pass, `1` the claim failed, `2` the instrument failed, `3` unsupported. The distinction
//! between 1 and 2 is the whole point: "the Vulkan EP computed the wrong answer" and "the ONNX
//! Runtime library could not be found" must never be the same signal to CI.

use std::path::PathBuf;
use std::process::ExitCode;

use ort_model_runner::error::{Failure, Result, Severity};
use ort_model_runner::evidence::{self, Outcome};
use ort_model_runner::json::to_string_pretty;
use ort_model_runner::{compare, feeds, run};

const USAGE: &str = "\
ort-model-runner -- Rust-native real-model CPU-vs-Vulkan validation (no Python, no PyPI)

USAGE
    ort-model-runner --check-model-agreement <model> [options]
    ort-model-runner --list-models
    ort-model-runner --list-devices [--ep-lib <path>] [--ort-lib <path>]

OPTIONS
    --check-model-agreement <name>  Run <name> on the CPU EP and the Vulkan EP and prove the
                                    Vulkan run was real and correct. <name> must appear in
                                    bench/results/model_provenance.json unless --model-path is
                                    given.
    --model-path <file>             Use this ONNX file instead of the pinned cache entry. Still
                                    checked against the pin when the name is a pinned one.
    --foundry-model <alias>         Resolve the model from the Foundry Local cache instead. Only
                                    `phi-3.5-mini` is known. Exactly one cached variant must
                                    match; zero and two are different errors, never a guess.
    --ep-lib <file>                 Vulkan plugin library. Default:
                                    <repo>/rust/target/release/<platform library name>.
    --ort-lib <file>                ONNX Runtime shared library. Default: discovered from
                                    ORT_MODEL_RUNNER_ORT_LIB, ORT_HOME, ONNXRUNTIME_DIR, the
                                    repository .venv, then the loader search path.
    --out <file>                    Write the evidence JSON here (always written, pass or fail).
    --seed <u64>                    Deterministic input seed. Default: a fixed constant.
    --free-dim <name=N>             Pin a symbolic dimension. Repeatable. Also accepts
                                    <input>:<axis>=N for unnamed axes. Every free dimension
                                    defaults to 1.
    --input <name=file.raw>         Feed this input from a raw little-endian file instead of
                                    generating it. Repeatable.
    --rtol <f> --atol <f>           Override the per-model tolerance policy. Must be given
                                    together; recorded in the evidence as source=cli.
    --fetch                         Allow downloading a missing pinned model (verified against
                                    its SHA-256 before it enters the cache).
    --cpu-only                      Run only the CPU reference. Never reports PASS; for proving
                                    the harness works where no Vulkan device exists.
    --device-selector <sel>         Pin the Vulkan device by STABLE IDENTITY before the EP library
                                    is registered: uuid:<32 hex>, luid:<16 hex>, pci:<D:B:D.F>,
                                    id:<vendor>:<device>, name:<substring> or index:<n>. The run
                                    then records execution_provider.selected_device and the
                                    device_identity_agreement guard fails the run if the device
                                    the session actually opened is not the one requested.
    --keep-profile                  Keep the ONNX Runtime profile JSON instead of deleting it.
    --quiet                         Print only the outcome line.
    -h, --help                      This text.
";

enum Command {
    Help,
    ListModels,
    ListDevices(Box<run::RunConfig>),
    Check(Box<run::RunConfig>, bool),
}

fn parse_args(argv: &[String]) -> Result<Command> {
    let mut config = run::RunConfig::default();
    let mut quiet = false;
    let mut action: Option<&'static str> = None;
    let mut i = 0;
    while i < argv.len() {
        let arg = argv[i].clone();
        // A flag that takes a value consumes the next argument; a missing one is an error rather
        // than a silently defaulted value.
        let mut take = |what: &str| -> Result<String> {
            i += 1;
            argv.get(i).cloned().ok_or_else(|| {
                Failure::instrument("missing_argument", format!("{what} needs a value"))
            })
        };
        match arg.as_str() {
            "-h" | "--help" => return Ok(Command::Help),
            "--list-models" => action = Some("list-models"),
            "--list-devices" => action = Some("list-devices"),
            "--check-model-agreement" => {
                config.model = take("--check-model-agreement")?;
                action = Some("check");
            }
            "--model-path" => config.model_path = Some(PathBuf::from(take("--model-path")?)),
            "--foundry-model" => {
                let alias = take("--foundry-model")?;
                let spec = match alias.as_str() {
                    "phi-3.5-mini" | "phi-3.5" => {
                        ort_model_runner::foundry::FoundryModelSpec::phi35_cuda_gpu()
                    }
                    other => {
                        return Err(Failure::unsupported(
                            "unknown_foundry_alias",
                            format!(
                                "--foundry-model {other:?} is not a known identity. This runner \
                                 resolves exact variant/provider/filename triples only, and the \
                                 one it knows is `phi-3.5-mini`."
                            ),
                        ));
                    }
                };
                let found = ort_model_runner::foundry::resolve_model_path(&spec, None)?;
                if config.model.is_empty() {
                    config.model = spec.variant_name.clone();
                }
                config.model_path = Some(found.onnx_path);
                action = Some("check");
            }
            "--ep-lib" => config.ep_lib = Some(PathBuf::from(take("--ep-lib")?)),
            "--ort-lib" => config.ort_lib = Some(PathBuf::from(take("--ort-lib")?)),
            "--out" => config.out = Some(PathBuf::from(take("--out")?)),
            "--seed" => {
                let raw = take("--seed")?;
                config.seed = parse_seed(&raw)?;
            }
            "--free-dim" => {
                let raw = take("--free-dim")?;
                let (name, value) = feeds::parse_free_dim(&raw)?;
                config.free_dims.insert(name, value);
            }
            "--input" => {
                let raw = take("--input")?;
                let (name, path) = raw.split_once('=').ok_or_else(|| {
                    Failure::instrument(
                        "bad_input",
                        format!("--input expects name=path, got {raw:?}"),
                    )
                })?;
                config
                    .input_files
                    .insert(name.to_string(), PathBuf::from(path));
            }
            "--rtol" => {
                let raw = take("--rtol")?;
                config.rtol = Some(parse_f64(&raw, "--rtol")?);
            }
            "--atol" => {
                let raw = take("--atol")?;
                config.atol = Some(parse_f64(&raw, "--atol")?);
            }
            "--fetch" => config.fetch = true,
            "--device-selector" => {
                config.device_selector = Some(take("--device-selector")?);
            }
            "--cpu-only" => config.cpu_only = true,
            "--keep-profile" => config.keep_profile = true,
            "--quiet" => quiet = true,
            other => {
                return Err(Failure::instrument(
                    "unknown_argument",
                    format!("unrecognised argument {other:?}. Run with --help."),
                ));
            }
        }
        i += 1;
    }
    // Argv is the most upstream instrument there is, and a refusal that is knowable from it
    // alone belongs here rather than 500 lines into a run. `compare::resolve` also rejects this
    // pairing, but it is not reached until after the CPU arm has executed -- so before this
    // check, `--rtol 1e-3` with no `--atol` meant a full inference run and *then* an error, and
    // the CLI test that claimed to cover it could only assert "some non-zero exit", which it got
    // for an unrelated reason. Both checks stay: this one is the fast, observable one, and the
    // one in `compare` guards the library's own callers.
    if config.rtol.is_some() != config.atol.is_some() {
        return Err(Failure::instrument(
            "partial_tolerance_override",
            "--rtol and --atol must be given together: half an overridden tolerance silently \
             inherits the other half from the policy, which is the kind of number nobody reviews.",
        ));
    }
    match action {
        Some("list-models") => Ok(Command::ListModels),
        Some("list-devices") => Ok(Command::ListDevices(Box::new(config))),
        Some("check") => Ok(Command::Check(Box::new(config), quiet)),
        _ => Ok(Command::Help),
    }
}

fn parse_seed(raw: &str) -> Result<u64> {
    let parsed = if let Some(hex) = raw.strip_prefix("0x").or_else(|| raw.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16)
    } else {
        raw.parse::<u64>()
    };
    parsed.map_err(|_| {
        Failure::instrument(
            "bad_seed",
            format!("--seed {raw:?} is not a u64 (decimal, or hex with a 0x prefix)"),
        )
    })
}

fn parse_f64(raw: &str, flag: &str) -> Result<f64> {
    let value: f64 = raw.parse().map_err(|_| {
        Failure::instrument("bad_tolerance", format!("{flag} {raw:?} is not a number"))
    })?;
    if !value.is_finite() || value < 0.0 {
        return Err(Failure::instrument(
            "bad_tolerance",
            format!("{flag} {raw:?} must be a finite, non-negative number"),
        ));
    }
    Ok(value)
}

fn list_models() -> Result<()> {
    let repo = ort_model_runner::repo::root()?;
    let manifest = ort_model_runner::provenance::load_manifest(&run::manifest_path(&repo))?;
    println!(
        "{:<18} {:>12}  {:<64} tolerance",
        "model", "bytes", "sha256"
    );
    for pin in &manifest {
        let tol = match compare::tolerance_for(&pin.name) {
            Some(t) => format!("rtol={:.1e} atol={:.1e}", t.rtol, t.atol),
            // Stated explicitly rather than left blank: a model with no policy entry cannot be
            // run by this tool without an explicit override, and the listing should say so.
            None => "none (needs --rtol/--atol)".to_string(),
        };
        println!(
            "{:<18} {:>12}  {:<64} {tol}",
            pin.name, pin.bytes, pin.sha256
        );
    }
    Ok(())
}

fn list_devices(config: &run::RunConfig) -> Result<()> {
    let repo = ort_model_runner::repo::root()?;
    let discovered = ort_model_runner::ortlib::discover(
        &ort_model_runner::ortlib::Search::from_environment(Some(&repo), config.ort_lib.clone()),
    )?;
    println!(
        "onnxruntime: {} ({})",
        discovered.path.display(),
        discovered.source
    );
    let loaded = ort_model_runner::ortlib::load(discovered)?;
    println!(
        "api version: {} ({})",
        loaded.api_version, loaded.version_string
    );
    let api = ort_model_runner::ortapi::Api::new(loaded.api);
    let env = ort_model_runner::ortapi::Env::new(
        api,
        "ort-model-runner",
        ort_model_runner::ortapi::LogSeverity::Error,
    )?;
    let ep_lib = config
        .ep_lib
        .clone()
        .unwrap_or_else(|| run::default_ep_lib(&repo));
    if ep_lib.is_file() {
        env.register_ep_library(run::EP_NAME, &ep_lib)?;
        println!("plugin:      {}", ep_lib.display());
    } else {
        println!(
            "plugin:      {} (absent -- only built-in providers will be listed)",
            ep_lib.display()
        );
    }
    for d in env.ep_devices()? {
        println!(
            "  [{}] {:<28} vendor={:<20} {} {:04x}:{:04x}",
            d.index, d.ep_name, d.ep_vendor, d.hardware_type, d.vendor_id, d.device_id
        );
        // Stable identity (issue #18): printed on its own line, never fabricated when a field is
        // unavailable (e.g. no LUID/PCI on this platform, or a non-Vulkan EP with no metadata).
        println!(
            "      uuid={} luid={} pci={}",
            d.uuid.as_deref().unwrap_or("(unavailable)"),
            d.luid.as_deref().unwrap_or("(unavailable)"),
            d.pci.as_deref().unwrap_or("(unavailable)"),
        );
    }
    Ok(())
}

fn run_check(config: &run::RunConfig, quiet: bool) -> i32 {
    match run::execute(config) {
        Ok((outcome, doc)) => {
            if let Some(path) = &config.out {
                if let Err(e) = evidence::write_json(path, &doc) {
                    eprintln!("{}: {}", e.token(), e.message);
                    return Severity::Instrument.exit_code();
                }
            }
            if quiet {
                println!("{} {}", outcome.as_str(), config.model);
            } else {
                if config.out.is_none() {
                    println!("{}", to_string_pretty(&doc));
                }
                print!("{}", run::summarize(&doc));
                if let Some(path) = &config.out {
                    println!("  evidence    {}", path.display());
                }
            }
            run::exit_code(outcome)
        }
        Err(e) => {
            // An instrument failure is not a failed claim and must not be reported as one: CI that
            // treats "ORT is missing" as "the EP is wrong" sends people to debug the wrong
            // component.
            eprintln!("{}: {}", e.token(), e.message);
            let outcome = match e.severity {
                Severity::Unsupported => Outcome::Unsupported,
                Severity::Condition => Outcome::Fail,
                Severity::Instrument => Outcome::Error,
            };
            eprintln!("{} {}", outcome.as_str(), config.model);
            e.exit_code()
        }
    }
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let command = match parse_args(&argv) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{}: {}", e.token(), e.message);
            return ExitCode::from(e.exit_code() as u8);
        }
    };
    let outcome = match command {
        Command::Help => {
            print!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        Command::ListModels => list_models().map(|_| 0),
        Command::ListDevices(config) => list_devices(&config).map(|_| 0),
        Command::Check(config, quiet) => Ok(run_check(&config, quiet)),
    };
    match outcome {
        Ok(code) => ExitCode::from(code as u8),
        Err(e) => {
            eprintln!("{}: {}", e.token(), e.message);
            ExitCode::from(e.exit_code() as u8)
        }
    }
}
