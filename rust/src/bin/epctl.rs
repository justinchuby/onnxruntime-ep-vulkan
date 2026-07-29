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
    eprintln!();
    eprintln!("    --dump-capabilities  every registered op, its opset window, dtypes, status,");
    eprintln!("                         backing shader, and (for contrib ops) the ORT release");
    eprintln!("                         its claim predicate was verified against.");
    eprintln!("    --json               machine-readable output, for CI diffing.");
}

fn main() -> std::process::ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let json = args.iter().any(|a| a == "--json");
    let dump = args.iter().any(|a| a == "--dump-capabilities");

    if let Some(bad) = args
        .iter()
        .find(|a| a.as_str() != "--json" && a.as_str() != "--dump-capabilities")
    {
        eprintln!("epctl: unrecognised argument `{bad}`");
        usage();
        return std::process::ExitCode::from(2);
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
