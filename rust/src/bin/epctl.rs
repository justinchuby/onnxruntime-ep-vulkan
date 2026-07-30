//! `epctl` — an offline inspector for the Vulkan EP's static configuration.
//!
//! This exists to satisfy `DESIGN.md` §1.4 constraint **C2**: the ORT release a contrib op's
//! claim predicate was written against must be *surfaced to users*, not buried in a source
//! comment. A capability dump is the natural place for it, and having a dump at all is useful
//! well beyond C2 — "what will this build actually claim?" is the first question anyone debugging
//! a partitioning surprise asks.
//!
//! Everything here is static: no Vulkan instance is created, no ORT is loaded, no device is
//! touched. That is deliberate. The output is a property of the *binary*, so it can be captured
//! in CI, diffed across commits, and attached to a bug report from a machine that cannot run the
//! EP at all. Runtime device capabilities are a separate concern and belong to a separate probe.
//!
//! ```text
//! epctl --dump-capabilities          # human-readable table
//! epctl --dump-capabilities --json   # machine-readable, for CI diffing
//! ```
//!
//! This binary consumes only the crate's public API. It deliberately owns no registry knowledge
//! of its own, so it cannot drift from the table it reports on.

use onnxruntime_vulkan_ep::counters;
use onnxruntime_vulkan_ep::engine;
use onnxruntime_vulkan_ep::registry::{OPSET_ANY, OpSpec, OpStatus, all_specs};
use onnxruntime_vulkan_ep::sys;

fn opset_window(spec: &OpSpec) -> String {
    if spec.max_opset == OPSET_ANY {
        format!("{}+", spec.min_opset)
    } else {
        format!("{}..={}", spec.min_opset, spec.max_opset)
    }
}

fn dtypes(spec: &OpSpec) -> String {
    let names: Vec<String> = spec.caps.iter().map(|d| format!("{d:?}")).collect();
    if names.is_empty() {
        "-".to_string()
    } else {
        names.join(",")
    }
}

/// The C2 column.
///
/// Default-domain rows report `n/a` rather than a baseline, and that is not an omission: their
/// compatibility contract is the opset window in the adjacent column. Contrib rows have no opset,
/// so a baseline is the only thing that says when anyone last checked the schema, and a `MISSING`
/// where one is required is itself the signal.
fn schema_baseline(spec: &OpSpec) -> String {
    match spec.schema_baseline() {
        Some(b) => b.describe(),
        None if spec.domain.as_str().is_empty() => "n/a (opset-versioned)".to_string(),
        None => "MISSING".to_string(),
    }
}

fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn sorted_rows() -> Vec<&'static OpSpec> {
    let mut rows: Vec<&'static OpSpec> = all_specs().collect();
    rows.sort_by_key(|s| s.qualified_name());
    rows
}

/// A short status tag for the table. The full staged blocker is far too long for a column, so it
/// is grouped underneath the table instead — the blockers repeat across dozens of rows, and
/// grouping them turns a wall of duplicated prose into a short list of real reasons.
fn status_tag(spec: &OpSpec) -> &'static str {
    match spec.status {
        OpStatus::Live => "live",
        OpStatus::Staged(_) => "staged",
    }
}

fn staged_reason(spec: &OpSpec) -> Option<&'static str> {
    match spec.status {
        OpStatus::Live => None,
        OpStatus::Staged(why) => Some(why),
    }
}

fn dump_human() {
    println!("onnxruntime-ep-vulkan {}", env!("CARGO_PKG_VERSION"));
    println!(
        "ORT ABI: built against {}, minimum supported {}",
        sys::ORT_PINNED,
        sys::ORT_FLOOR
    );
    println!();
    println!(
        "{:<34} {:<10} {:<24} {:<8} {:<22} schema baseline",
        "op", "opsets", "dtypes", "status", "kernel"
    );
    println!("{}", "-".repeat(126));

    let rows = sorted_rows();
    for spec in &rows {
        println!(
            "{:<34} {:<10} {:<24} {:<8} {:<22} {}",
            spec.qualified_name(),
            opset_window(spec),
            dtypes(spec),
            status_tag(spec),
            format!("{:?}/{}", spec.kernel.template, spec.kernel.op),
            schema_baseline(spec),
        );
    }

    let live = rows.iter().filter(|s| s.is_live()).count();
    println!();
    println!(
        "{} row(s): {live} live, {} staged",
        rows.len(),
        rows.len() - live
    );
    println!(
        "A staged row is registered and tested but never claimed. Only live rows can take a node."
    );

    let mut reasons: Vec<(&'static str, usize)> = Vec::new();
    for spec in &rows {
        if let Some(why) = staged_reason(spec) {
            match reasons.iter_mut().find(|(r, _)| *r == why) {
                Some((_, n)) => *n += 1,
                None => reasons.push((why, 1)),
            }
        }
    }
    if !reasons.is_empty() {
        println!();
        println!("staged because:");
        reasons.sort_by_key(|r| std::cmp::Reverse(r.1));
        for (why, n) in reasons {
            println!("  {n:>3} row(s)  {why}");
        }
    }
}

fn dump_json() {
    println!("{{");
    println!("  \"crate_version\": \"{}\",", env!("CARGO_PKG_VERSION"));
    println!("  \"ort_built_against\": \"{}\",", sys::ORT_PINNED.release);
    println!("  \"ort_api_version\": {},", sys::ORT_PINNED.api_version);
    println!("  \"ort_minimum\": \"{}\",", sys::ORT_FLOOR.release);
    println!("  \"ort_api_version_min\": {},", sys::ORT_FLOOR.api_version);
    println!("  \"ops\": [");
    let rows = sorted_rows();
    for (i, spec) in rows.iter().enumerate() {
        let comma = if i + 1 == rows.len() { "" } else { "," };
        println!(
            "    {{\"name\": \"{}\", \"opsets\": \"{}\", \"dtypes\": \"{}\", \
             \"status\": \"{}\", \"live\": {}, \"schema_baseline\": \"{}\"}}{comma}",
            escape_json(&spec.qualified_name()),
            opset_window(spec),
            dtypes(spec),
            status_tag(spec),
            spec.is_live(),
            escape_json(&schema_baseline(spec)),
        );
    }
    println!("  ]");
    println!("}}");
}

fn usage() {
    eprintln!("epctl — offline inspector for the Vulkan execution provider");
    eprintln!();
    eprintln!("USAGE:");
    eprintln!("    epctl --dump-capabilities [--json]");
    eprintln!("    epctl --probe-loader");
    eprintln!("    epctl --check-counters <file> [--require-dispatches N]");
    eprintln!();
    eprintln!("    --dump-capabilities  every registered op, its opset window, dtypes, status,");
    eprintln!("                         backing shader, and (for contrib ops) the ORT release");
    eprintln!("                         its claim predicate was verified against.");
    eprintln!("    --json               machine-readable output, for CI diffing.");
    eprintln!("    --probe-loader       probe the Vulkan loader: library presence, version,");
    eprintln!("                         ICD discovery env vars, available layers/extensions,");
    eprintln!(
        "                         and whether vkCreateInstance + device enumeration succeed."
    );
    eprintln!("                         Run this in CI before the test suite to establish whether");
    eprintln!(
        "                         Vulkan is functional on the runner independently of the EP."
    );
    eprintln!("    --check-counters <file>");
    eprintln!("                         read the execution-counter snapshot the EP writes when");
    eprintln!(
        "                         ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE is set, and fail the lane"
    );
    eprintln!("                         unless a claimed node actually executed on a device.");
    eprintln!("    --require-dispatches N");
    eprintln!("                         minimum executed dispatches to pass (default 1).");
    eprintln!();
    eprintln!("EXIT CODES:");
    eprintln!("    0  pass");
    eprintln!("    1  the lane reported, and the number was below the requirement");
    eprintln!("    2  usage error");
    eprintln!("    3  the lane did not report (file missing, unreadable, or unparseable)");
}

// ---------------------------------------------------------------------------------------------
// --check-counters: the criterion-8 gate
// ---------------------------------------------------------------------------------------------

/// Outcome of reading a counters snapshot, kept separate from printing so it can be tested.
///
/// Exit 1 and exit 3 are deliberately different codes and the distinction is the whole point.
/// "The lane ran and executed nothing" is a real, attributable result: it means every node
/// declined, or fell back to CPU, or the device was refused. "The lane did not report" means the
/// process died before teardown — which is what both CI lanes are doing today — and a crashed lane
/// must not be able to look like any kind of answer at all. The same reasoning as
/// [`probe_exit_code`]: a wrong answer is worse than a loud refusal to answer.
#[derive(Debug, PartialEq, Eq)]
enum CounterVerdict {
    Pass { dispatches: u64 },
    TooFew { dispatches: u64, required: u64 },
    NoReport(String),
}

/// Pull one unsigned integer field out of the counters JSON.
///
/// Hand-rolled to match `counters::to_json`, which is hand-rolled for the same reason: this binary
/// is a CI gate and must not acquire a dependency whose absence turns a red lane into a build
/// failure. The document is eight flat integers written by us; a full parser would be more code
/// than the thing it parses.
fn json_u64(doc: &str, key: &str) -> Option<u64> {
    let needle = format!("\"{key}\"");
    let start = doc.find(&needle)? + needle.len();
    let rest = doc[start..].trim_start();
    let rest = rest.strip_prefix(':')?.trim_start();
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return None;
    }
    digits.parse().ok()
}

fn read_counters(path: &str, required: u64) -> CounterVerdict {
    let doc = match std::fs::read_to_string(path) {
        Ok(d) => d,
        Err(e) => {
            return CounterVerdict::NoReport(format!(
                "cannot read {path}: {e}. The EP writes this file on its first successful dispatch \
                 and again at factory teardown, so an absent file means neither happened — most \
                 often because the host process died mid-session."
            ));
        }
    };
    let Some(abi) = json_u64(&doc, "abi_version") else {
        return CounterVerdict::NoReport(format!(
            "{path} does not contain an `abi_version` field; it is not a counters snapshot this \
             build understands."
        ));
    };
    if abi != counters::COUNTERS_ABI_VERSION as u64 {
        return CounterVerdict::NoReport(format!(
            "{path} reports counters ABI {abi}, but this epctl understands {}. Refusing to read \
             fields whose meaning may have changed.",
            counters::COUNTERS_ABI_VERSION
        ));
    }
    let Some(dispatches) = json_u64(&doc, "dispatches_executed") else {
        return CounterVerdict::NoReport(format!("{path} has no `dispatches_executed` field."));
    };

    if dispatches >= required {
        CounterVerdict::Pass { dispatches }
    } else {
        CounterVerdict::TooFew {
            dispatches,
            required,
        }
    }
}

fn check_counters(path: &str, required: u64) -> std::process::ExitCode {
    // Echo whatever the file does contain before judging it — a lane that fails here is a lane
    // someone has to diagnose from the log alone.
    if let Ok(doc) = std::fs::read_to_string(path) {
        println!("epctl: counters snapshot at {path}");
        for line in doc.lines() {
            println!("  {line}");
        }
    }
    match read_counters(path, required) {
        CounterVerdict::Pass { dispatches } => {
            println!(
                "epctl: PASS — {dispatches} dispatch(es) executed on a real device in this lane \
                 (required {required}).\n\
                 \x20 Note what this does and does not claim: it claims a command buffer reached a \
                 device and the fence signalled. It claims nothing about whether the results are \
                 numerically correct — that is the differential test's job, not this one's."
            );
            std::process::ExitCode::SUCCESS
        }
        CounterVerdict::TooFew {
            dispatches,
            required,
        } => {
            eprintln!(
                "epctl: FAIL — this lane executed {dispatches} dispatch(es), below the required \
                 {required}.\n\
                 \x20 The suite may still have reported green: a lane where every op declines, or \
                 every test skips, or every node silently falls back to CPU, passes its assertions \
                 and executes nothing. That is exactly the state this gate exists to make loud.\n\
                 \x20 Look at `subgraphs_live` and `subgraphs_stub` above: zero live subgraphs \
                 means nothing was claimed; live subgraphs with zero dispatches means Compile \
                 succeeded and Compute never ran or never succeeded."
            );
            std::process::ExitCode::from(1)
        }
        CounterVerdict::NoReport(why) => {
            eprintln!(
                "epctl: NO REPORT — {why}\n\
                 \x20 This is exit 3, deliberately distinct from exit 1. 'Executed nothing' is an \
                 answer; 'did not report' is the absence of one, and usually means the process \
                 crashed. Do not read it as a device problem until the log says so."
            );
            std::process::ExitCode::from(3)
        }
    }
}

/// The phrase in `engine::loader_probe_report()` that carries the gate verdict.
///
/// This is a **contract between two files with different owners** — `engine.rs` is Switch's, this
/// is mine — expressed, for now, as a substring of prose. See [`probe_exit_code`].
const GATE_VERDICT_MARKER: &str = "passed the §7.2 capability gate";

/// Turn the loader-probe report into an exit code for CI gate scripts.
///
/// # Why this is more than one line
///
/// The obvious implementation is `report.contains(MARKER) && !report.contains("0 device(s)")`.
/// The problem is what happens when the marker is *absent*: that spelling silently returns
/// "failure", which is indistinguishable from a genuine "no capable device". So the day someone
/// rewords the report — an ordinary, blameless edit to a human-readable string in a file whose
/// owner has no reason to know this parser exists — CI starts reporting "Vulkan is broken on this
/// runner" and the next person spends a cycle chasing an environment that is fine. That is the
/// exact failure mode we have already burned two CI cycles on this week, in the opposite
/// direction.
///
/// So: absence of the marker is its own exit code (3) with a message naming the coupling. A wrong
/// answer is worse than a loud refusal to answer.
///
/// The real fix is for `loader_probe_report()` to return a struct with the verdict as a field and
/// `Display` for the prose, so this parsing disappears. That is Switch's file and his call; this
/// makes the failure mode survivable in the meantime.
fn probe_exit_code(report: &str) -> std::process::ExitCode {
    if !report.contains(GATE_VERDICT_MARKER) {
        eprintln!(
            "epctl: the loader probe report does not contain the phrase this gate reads\n\
             \x20 expected substring: {GATE_VERDICT_MARKER:?}\n\
             \x20 epctl cannot tell 'no capable device' from 'the report was reworded', and \
             guessing would either green-light a broken runner or condemn a working one.\n\
             \x20 Fix: keep the phrase, or (better) have `engine::loader_probe_report()` return \
             the verdict as data instead of prose."
        );
        return std::process::ExitCode::from(3);
    }
    if report.contains("0 device(s) passed") {
        return std::process::ExitCode::from(1);
    }
    std::process::ExitCode::SUCCESS
}

fn main() -> std::process::ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let json = args.iter().any(|a| a == "--json");
    let dump = args.iter().any(|a| a == "--dump-capabilities");
    let probe = args.iter().any(|a| a == "--probe-loader");

    // `--check-counters` and `--require-dispatches` take a value, so the flat "every argument must
    // be a known flag" check below has to know to skip the values. Parse them out first.
    let mut counters_file: Option<String> = None;
    let mut required_dispatches: u64 = 1;
    let mut consumed: Vec<usize> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--check-counters" => {
                let Some(v) = args.get(i + 1) else {
                    eprintln!("epctl: --check-counters needs a file path");
                    usage();
                    return std::process::ExitCode::from(2);
                };
                counters_file = Some(v.clone());
                consumed.push(i);
                consumed.push(i + 1);
                i += 2;
            }
            "--require-dispatches" => {
                let Some(v) = args.get(i + 1).and_then(|v| v.parse::<u64>().ok()) else {
                    eprintln!("epctl: --require-dispatches needs a non-negative integer");
                    usage();
                    return std::process::ExitCode::from(2);
                };
                required_dispatches = v;
                consumed.push(i);
                consumed.push(i + 1);
                i += 2;
            }
            _ => i += 1,
        }
    }

    if let Some((_, bad)) = args
        .iter()
        .enumerate()
        .filter(|(i, _)| !consumed.contains(i))
        .find(|(_, a)| {
            a.as_str() != "--json"
                && a.as_str() != "--dump-capabilities"
                && a.as_str() != "--probe-loader"
        })
    {
        eprintln!("epctl: unrecognised argument `{bad}`");
        usage();
        return std::process::ExitCode::from(2);
    }

    if let Some(path) = counters_file {
        return check_counters(&path, required_dispatches);
    }

    if probe {
        // Note: this is the one epctl operation that creates a VkInstance and touches Vulkan.
        // All other epctl operations remain static (no ORT, no Vulkan).
        // Cross-owner edit: Switch added this; Tank owns epctl.rs. Flagged in decisions.
        let report = engine::loader_probe_report();
        println!("{report}");
        return probe_exit_code(&report);
    }

    if !dump {
        usage();
        return std::process::ExitCode::from(2);
    }

    if json {
        dump_json()
    } else {
        dump_human()
    }
    std::process::ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The three outcomes must be distinguishable. Conflating the last two is the bug this
    /// function exists to prevent.
    #[test]
    fn probe_exit_code_separates_failure_from_an_unreadable_report() {
        let pass = format!("2 device(s) {GATE_VERDICT_MARKER}");
        let fail = format!("0 device(s) passed — none {GATE_VERDICT_MARKER}");
        let reworded = "Vulkan looks fine on this machine.";

        assert_eq!(
            format!("{:?}", probe_exit_code(&pass)),
            format!("{:?}", std::process::ExitCode::SUCCESS)
        );
        assert_eq!(
            format!("{:?}", probe_exit_code(&fail)),
            format!("{:?}", std::process::ExitCode::from(1))
        );
        assert_eq!(
            format!("{:?}", probe_exit_code(reworded)),
            format!("{:?}", std::process::ExitCode::from(3)),
            "a reworded report must be its own exit code, not a silent 'no devices' verdict"
        );
    }

    fn snapshot(dispatches: u64) -> String {
        onnxruntime_vulkan_ep::counters::VulkanEpCounters {
            struct_size: 0,
            abi_version: counters::COUNTERS_ABI_VERSION,
            compile_calls: 1,
            subgraphs_live: 1,
            subgraphs_stub: 0,
            compute_calls: dispatches,
            compute_failures: 0,
            dispatches_executed: dispatches,
        }
        .to_json()
    }

    #[test]
    fn json_u64_reads_our_own_snapshot_format() {
        let doc = snapshot(7);
        assert_eq!(json_u64(&doc, "dispatches_executed"), Some(7));
        assert_eq!(json_u64(&doc, "compute_failures"), Some(0));
        assert_eq!(
            json_u64(&doc, "abi_version"),
            Some(counters::COUNTERS_ABI_VERSION as u64)
        );
        assert_eq!(json_u64(&doc, "not_a_field"), None);
    }

    /// The three outcomes of the criterion-8 gate must be distinguishable for the same reason the
    /// loader probe's are: a lane that crashed before reporting must not be mistakable for a lane
    /// that reported honestly.
    #[test]
    fn counter_verdicts_separate_zero_from_no_report() {
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target/epctl-counter-gate-test");
        std::fs::create_dir_all(&dir).expect("scratch dir");

        let ran = dir.join("ran.json");
        std::fs::write(&ran, snapshot(3)).expect("write");
        assert_eq!(
            read_counters(ran.to_str().unwrap(), 1),
            CounterVerdict::Pass { dispatches: 3 }
        );

        let idle = dir.join("idle.json");
        std::fs::write(&idle, snapshot(0)).expect("write");
        assert_eq!(
            read_counters(idle.to_str().unwrap(), 1),
            CounterVerdict::TooFew {
                dispatches: 0,
                required: 1
            },
            "a lane that executed nothing is a real, attributable answer — exit 1"
        );

        let missing = dir.join("this-file-does-not-exist.json");
        let _ = std::fs::remove_file(&missing);
        assert!(
            matches!(
                read_counters(missing.to_str().unwrap(), 1),
                CounterVerdict::NoReport(_)
            ),
            "a lane that never wrote the file has not answered at all — exit 3, not exit 1"
        );

        let garbage = dir.join("garbage.json");
        std::fs::write(&garbage, "the run crashed halfway through this fi").expect("write");
        assert!(
            matches!(
                read_counters(garbage.to_str().unwrap(), 1),
                CounterVerdict::NoReport(_)
            ),
            "a truncated file is the signature of a crash mid-write and must not parse as zero"
        );

        let future = dir.join("future.json");
        std::fs::write(
            &future,
            snapshot(9).replace("\"abi_version\": 1", "\"abi_version\": 99"),
        )
        .expect("write");
        assert!(
            matches!(
                read_counters(future.to_str().unwrap(), 1),
                CounterVerdict::NoReport(_)
            ),
            "a snapshot from a counters ABI we do not understand must refuse rather than guess"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_dispatch_requirement_is_configurable_and_enforced() {
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target/epctl-counter-threshold-test");
        std::fs::create_dir_all(&dir).expect("scratch dir");
        let f = dir.join("c.json");
        std::fs::write(&f, snapshot(5)).expect("write");

        assert_eq!(
            read_counters(f.to_str().unwrap(), 5),
            CounterVerdict::Pass { dispatches: 5 }
        );
        assert_eq!(
            read_counters(f.to_str().unwrap(), 6),
            CounterVerdict::TooFew {
                dispatches: 5,
                required: 6
            }
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
