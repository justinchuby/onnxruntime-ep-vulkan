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

    if let Some(bad) = args.iter().find(|a| {
        a.as_str() != "--json"
            && a.as_str() != "--dump-capabilities"
            && a.as_str() != "--probe-loader"
    }) {
        eprintln!("epctl: unrecognised argument `{bad}`");
        usage();
        return std::process::ExitCode::from(2);
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
}
