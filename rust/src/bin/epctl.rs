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
        OpStatus::Ready => "ready",
        OpStatus::Staged(_) => "staged",
    }
}

fn staged_reason(spec: &OpSpec) -> Option<&'static str> {
    match spec.status {
        OpStatus::Live | OpStatus::Ready => None,
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
    eprintln!("    epctl --probe-validation [--plant-violation]");
    eprintln!(
        "    epctl --check-counters <file> [--require-dispatches N] [--require-device-memory]"
    );
    eprintln!();
    eprintln!("    --dump-capabilities  every registered op, its opset window, dtypes, status,");
    eprintln!("                         backing shader, and (for contrib ops) the ORT release");
    eprintln!("                         its claim predicate was verified against.");
    eprintln!("    --json               machine-readable output, for CI diffing.");
    eprintln!("    --probe-loader       probe the Vulkan loader: library presence, version,");
    eprintln!(
        "                         and whether vkCreateInstance + device enumeration succeed."
    );
    eprintln!("                         Run this in CI before the test suite to establish whether");
    eprintln!(
        "                         Vulkan is functional on the runner independently of the EP."
    );
    eprintln!("    --probe-validation   report whether VK_LAYER_KHRONOS_validation is installed,");
    eprintln!("                         enabled, AND being listened to by a debug messenger.");
    eprintln!("                         Exit 3 when absent: that is no answer, not a clean one.");
    eprintln!("    --plant-violation    with --probe-validation, deliberately commit an invalid");
    eprintln!("                         Vulkan call. The positive control: if this does NOT");
    eprintln!("                         produce a caught error, a clean run proves nothing.");
    eprintln!("    --check-counters <file>");
    eprintln!("                         read the execution-counter snapshot the EP writes when");
    eprintln!(
        "                         ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE is set, and fail the lane"
    );
    eprintln!("                         unless a claimed node actually executed on a device");
    eprintln!("                         AND the model-level correctness verdict is MATCH.");
    eprintln!("                         §9.1.3 / §10.0 (Morpheus ruling): compute_failures:0 is");
    eprintln!("                         an execution-status counter, never a correctness signal.");
    eprintln!("                         The verdict field model_output_equivalence carries that");
    eprintln!("                         signal; it is written by the Python comparison gate.");
    eprintln!("                         DIVERGENT = exit 1 (wrong answer, hard fail).");
    eprintln!("                         UNMEASURED = exit 3 (no answer, same as no report).");
    eprintln!("    --require-dispatches N");
    eprintln!("                         minimum executed dispatches to pass (default 1).");
    eprintln!("    --require-device-memory");
    eprintln!("                         fail unless every device handle was backed by a VkBuffer,");
    eprintln!("                         i.e. nothing was host-staged. Set this on any lane that");
    eprintln!("                         quotes a timing: a partially staged run is not a slow");
    eprintln!("                         device measurement, it is an average over two memories.");
    eprintln!("                         A snapshot with no allocation tally fails as exit 3, not");
    eprintln!("                         exit 0 — it cannot answer the question.");
    eprintln!();
    eprintln!("EXIT CODES:");
    eprintln!("    0  pass (dispatches ≥ required AND model_output_equivalence = MATCH)");
    eprintln!("    1  the lane reported, and dispatches were below the requirement,");
    eprintln!("       OR model_output_equivalence = DIVERGENT");
    eprintln!("    2  usage error");
    eprintln!("    3  the lane did not report (file missing, unreadable, or unparseable),");
    eprintln!("       OR model_output_equivalence = UNMEASURED (comparison was not performed)");
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
///
/// Two new variants (§9.1.3, §10.0 R9 — `model_output_equivalence` gate):
/// A lane that passes its dispatch count but has not compared against the CPU oracle has not
/// proven correctness — it has proven execution. Those are different claims. A lane that has
/// compared and found DIVERGENT has proven incorrectness. The exit codes for each must be
/// distinguishable from each other and from the dispatch variants:
///   `Pass` (exit 0) — dispatches ≥ required AND MATCH
///   `EquivalenceDivergent` (exit 1) — dispatches ≥ required but GPU output ≠ CPU output
///   `EquivalenceUnmeasured` (exit 3) — dispatches ≥ required but no comparison was performed
#[derive(Debug, PartialEq, Eq)]
enum CounterVerdict {
    /// Dispatches ≥ required **and** `model_output_equivalence = MATCH`. Exit 0.
    Pass {
        dispatches: u64,
    },
    /// Dispatches below required. Exit 1.
    TooFew {
        dispatches: u64,
        required: u64,
    },
    /// File not found, unreadable, wrong ABI, or missing required fields. Exit 3.
    NoReport(String),
    /// ORT derived a pointer that ran off the end of one of our allocations. Exit 1.
    OutOfBounds {
        count: u64,
    },
    /// A `Free` arrived after we released the allocator that owned the span.
    ///
    /// Unconditional, like [`CounterVerdict::OutOfBounds`], and for the same reason: it is not a
    /// metric, it is ORT and this EP disagreeing about who owns 2 GB.
    FreeAfterRelease {
        count: u64,
    },
    /// `--require-device-memory` was asked for and the run did not deliver it.
    NotOnDevice {
        staged_spans: u64,
        staged_bytes: u64,
        device_backed: u64,
        allocations: u64,
    },
    /// Dispatches ≥ required but `model_output_equivalence = DIVERGENT`. Exit 1.
    ///
    /// The GPU executed and produced an answer; the answer is wrong. This is not "no report"
    /// (exit 3) — a wrong answer is worse than no answer and earns the harder exit code.
    EquivalenceDivergent {
        dispatches: u64,
    },
    /// Dispatches ≥ required but no CPU comparison was performed (`model_output_equivalence`
    /// absent or `UNMEASURED`). Exit 3.
    ///
    /// This is the same exit code as `NoReport` because both represent the absence of an answer
    /// on the correctness question: one because the process crashed, the other because the
    /// comparison step was never run. Neither may be read as "the EP is correct."
    EquivalenceUnmeasured {
        dispatches: u64,
    },
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

/// Pull one string field out of the counters JSON.
///
/// Counterpart to [`json_u64`] for string-valued fields. Returns the bare string value
/// (without surrounding quotes) or `None` if the field is absent or malformed.
fn json_str<'a>(doc: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!("\"{key}\"");
    let start = doc.find(&needle)? + needle.len();
    let rest = doc[start..].trim_start();
    let rest = rest.strip_prefix(':')?.trim_start();
    let rest = rest.strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(&rest[..end])
}

/// `require_device_memory`: fail unless every device handle in the run was backed by a `VkBuffer`.
///
/// This exists because the staging path's caveat cannot survive as prose. A one-shot WARN saying
/// "any timing from this run is a host measurement" is right until some allocations are
/// device-backed and wrong afterwards; deleting it is worse, because a partially staged run would
/// then say nothing at all and its numbers would look like device numbers. The durable form of the
/// caveat is an assertion a performance lane sets and a machine evaluates, so nobody has to
/// remember a log line an hour after reading it.
fn read_counters_with(path: &str, required: u64, require_device_memory: bool) -> CounterVerdict {
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

    // Checked before the dispatch count, because it outranks it. A lane that executed plenty of
    // dispatches *and* took a pointer off the end of an allocation has not passed; it has produced
    // numbers nobody should trust. This is a correctness alarm wearing a counter's clothing.
    //
    // The key is optional on purpose: a snapshot written by a build without the ledger, or by a
    // run with device memory disabled, simply does not carry it, and absence must not be read as
    // zero. Present-and-non-zero is the only failing case.
    if let Some(oob) = json_u64(&doc, "pointers_in_guard_band")
        && oob > 0
    {
        return CounterVerdict::OutOfBounds { count: oob };
    }

    // Same class, same unconditional treatment: a Free arriving after the allocator that owned the
    // span was released means ORT still believed it owned memory we had torn down. Absence of the
    // key means a build or a run that cannot report it, which is not a zero.
    if let Some(late) = json_u64(&doc, "alloc_frees_after_release")
        && late > 0
    {
        return CounterVerdict::FreeAfterRelease { count: late };
    }

    // Also ahead of the dispatch count, and for the same reason: a run whose tensors sat in host
    // memory executed dispatches against the wrong memory for the purpose it was measured for.
    if require_device_memory {
        let staged_spans = json_u64(&doc, "alloc_staged_spans");
        let allocations = json_u64(&doc, "alloc_allocations");
        let device_backed = json_u64(&doc, "alloc_device_backed_spans");
        match (staged_spans, allocations, device_backed) {
            // Absent keys are not zero. A snapshot from a build with no tally cannot answer this
            // question, and must not pass a check it did not perform.
            (None, _, _) | (_, None, _) | (_, _, None) => {
                return CounterVerdict::NoReport(format!(
                    "{path} carries no `alloc_staged_spans`/`alloc_allocations`/\
                     `alloc_device_backed_spans`, so --require-device-memory cannot be evaluated \
                     against it. This snapshot predates the allocation tally. Refusing to pass a \
                     check that did not run."
                ));
            }
            (Some(staged), Some(allocs), Some(backed)) => {
                if staged > 0 || (allocs > 0 && backed == 0) {
                    return CounterVerdict::NotOnDevice {
                        staged_spans: staged,
                        staged_bytes: json_u64(&doc, "alloc_staged_bytes").unwrap_or(0),
                        device_backed: backed,
                        allocations: allocs,
                    };
                }
            }
        }
    }

    if dispatches < required {
        return CounterVerdict::TooFew {
            dispatches,
            required,
        };
    }

    // §9.1.3 (Morpheus ruling): `compute_failures` is an execution-status counter and **never** a
    // correctness signal. The correctness verdict is `model_output_equivalence`, which is written
    // by the Python comparison gate (Trinity) after the session runs. The EP writes UNMEASURED by
    // default; the gate upgrades it to MATCH or DIVERGENT. A lane that has enough dispatches but
    // no comparison performed is not a passing lane — it is a lane that forgot to measure, and
    // that must look different from both a passing lane and a crashing lane.
    //
    // R9: a named instrument that does not exist is exactly the thing R9 warns about. UNMEASURED
    // exit 3 is the instrument that would go red if the comparison step were removed.
    match json_str(&doc, counters::EQUIVALENCE_KEY) {
        Some(ref s) if *s == counters::EQUIVALENCE_MATCH => CounterVerdict::Pass { dispatches },
        Some(ref s) if *s == counters::EQUIVALENCE_DIVERGENT => {
            CounterVerdict::EquivalenceDivergent { dispatches }
        }
        // Absent or unknown value both map to UNMEASURED per R7: absence of an instrument must
        // not read as a negative result. `None` here means the comparison never ran; an unknown
        // string means a future writer we do not understand. Both are "no answer".
        _ => CounterVerdict::EquivalenceUnmeasured { dispatches },
    }
}

fn check_counters_with(
    path: &str,
    required: u64,
    require_device_memory: bool,
) -> std::process::ExitCode {
    // Echo whatever the file does contain before judging it — a lane that fails here is a lane
    // someone has to diagnose from the log alone.
    if let Ok(doc) = std::fs::read_to_string(path) {
        println!("epctl: counters snapshot at {path}");
        for line in doc.lines() {
            println!("  {line}");
        }
    }
    match read_counters_with(path, required, require_device_memory) {
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
        CounterVerdict::OutOfBounds { count } => {
            eprintln!(
                "epctl: FAIL — ORT derived {count} pointer(s) that landed in a guard band between \
                 our device handles.\n\
                 \x20 This is not a performance metric, it is a correctness alarm. ORT's memory- \
                 pattern planner does pointer arithmetic on allocator return values — measured, \
                 not assumed: it packs several tensors into one of our allocations and hands back \
                 `base + n` from the second run of a session onward. In-span arithmetic is fine \
                 and expected. A guard-band hit means an offset ran off the end of the allocation \
                 it was derived from.\n\
                 \x20 We only see this at all because handles are reserved address space rather \
                 than opaque integers. Under any design that could not detect it, this would be a \
                 silently wrong answer instead of a failing lane. Treat it as a bug in shape or \
                 size accounting, not as an allocator tuning knob, and read the `*.trace.txt` \
                 beside the counters file for the exact addresses."
            );
            std::process::ExitCode::from(1)
        }
        CounterVerdict::FreeAfterRelease { count } => {
            eprintln!(
                "epctl: FAIL — {count} Free call(s) arrived after the allocator that owned the \
                 span had been released.\n\
                 \x20 ORT and this EP disagree about who owns that memory. This check exists \
                 because the still-live-handles WARN used to end with an open disjunction — \
                 \"either a leak on our side or a tensor the session outlived\" — which is honest \
                 and undecidable by the reader, so it was never decided. This number decides it: \
                 spans that are still live at release and are *never* freed afterwards are ORT \
                 reclaiming them by destroying the session, which costs us nothing. Spans that are \
                 freed afterwards mean ORT held a pointer into a registry we had torn down.\n\
                 \x20 On a build that unmaps the reservation at release, this is a use-after-free \
                 rather than a log line. Do not quote memory numbers from this run."
            );
            std::process::ExitCode::from(1)
        }
        CounterVerdict::NotOnDevice {
            staged_spans,
            staged_bytes,
            device_backed,
            allocations,
        } => {
            eprintln!(
                "epctl: FAIL — --require-device-memory was asked for and this run did not deliver \
                 it: {staged_spans} host-staged span(s) ({staged_bytes} B), {device_backed} \
                 device-backed, out of {allocations} allocation(s).\n\
                 \x20 Host staging produces correct results, so nothing here says the run was \
                 wrong. It says the run's tensors were not where you asked them to be, which \
                 disqualifies any timing taken from it — a partially staged run is not a slow \
                 device measurement, it is an average over two different memories and comparable \
                 with neither.\n\
                 \x20 This assertion exists because the alternative was a log line. The staging \
                 WARN was accurate while nothing was device-backed and becomes wrong in both \
                 directions afterwards: kept, it over-warns on a nearly-all-device run until \
                 readers discount it; removed, a partially staged run says nothing at all and its \
                 numbers look like device numbers. A caveat that has to be remembered an hour \
                 later is not a caveat. Set this flag on any lane that quotes a number."
            );
            std::process::ExitCode::from(1)
        }
        CounterVerdict::EquivalenceDivergent { dispatches } => {
            eprintln!(
                "epctl: FAIL — {dispatches} dispatch(es) executed, but model_output_equivalence = \
                 DIVERGENT.\n\
                 \x20 The GPU reached the kernel and produced an answer. The answer is wrong: it \
                 does not match the CPU oracle. This is not 'no dispatches' (the EP ran) and not \
                 'no report' (the comparison ran). It is a confirmed correctness failure.\n\
                 \x20 §9.1.3 ruling: compute_failures:0 does not contradict this. A command \
                 buffer that signals its fence but produces numerically wrong values is an \
                 arithmetic bug, not a dispatch bug. The two counters are measuring different \
                 things and both can be zero simultaneously with a wrong result.\n\
                 \x20 See test_phi35_vulkan_matches_cpu_logits for the gate that produces this \
                 verdict. It is currently xfail(strict=True) — when Mouse fixes the kernel it \
                 will flip to MATCH and this lane will pass."
            );
            std::process::ExitCode::from(1)
        }
        CounterVerdict::EquivalenceUnmeasured { dispatches } => {
            eprintln!(
                "epctl: NO ANSWER — {dispatches} dispatch(es) executed, but \
                 model_output_equivalence = UNMEASURED.\n\
                 \x20 This is exit 3, the same code as NoReport, because both represent the \
                 absence of an answer on the correctness question: one because the process crashed \
                 before teardown, the other because the comparison gate was never run.\n\
                 \x20 R9: a set of individually sound instruments can be jointly silent on the \
                 property that matters. UNMEASURED means the instrument that would distinguish \
                 'executed correctly' from 'executed and produced garbage' was not present in this \
                 run. Do not quote coverage metrics from a run where this verdict appears.\n\
                 \x20 To produce a MATCH or DIVERGENT verdict, the Python comparison gate must \
                 write to ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE before session teardown. If that \
                 env var is not set, no verdict can be written."
            );
            std::process::ExitCode::from(3)
        }
    }
}

// ---------------------------------------------------------------------------------------------
// --probe-validation: the criterion-3 positive control
// ---------------------------------------------------------------------------------------------

/// The marker the harness greps for. A literal, not a description, so the two ends cannot drift.
const VALIDATION_CAUGHT_MARKER: &str = "EPCTL-VALIDATION-CAUGHT";
const VALIDATION_LAYER: &std::ffi::CStr = c"VK_LAYER_KHRONOS_validation";

/// Why `--probe-validation` cannot produce a verdict, kept separate from "it produced one".
#[derive(Debug, PartialEq, Eq)]
enum ValidationProbe {
    /// The layer is installed, was enabled, and a debug messenger is receiving its output.
    Armed,
    /// No Vulkan loader, or `vkCreateInstance` failed.
    NoLoader(String),
    /// The loader is present but `VK_LAYER_KHRONOS_validation` is not installed.
    LayerAbsent,
}

/// Create an instance with validation enabled *and a debug messenger attached*, then optionally
/// commit a deliberate violation.
///
/// # Why this exists
///
/// M0 criterion 3 asks that validation runs clean. Morpheus refused that as written, because
/// **"no errors surfaced" is exactly what a run with the layer not loaded reports** — the same
/// objection that killed two fabricated speedups, applied to a layer instead of a provider.
///
/// The gap was worse than that. The EP requests `VK_LAYER_KHRONOS_validation` but attaches no
/// `VkDebugUtilsMessengerEXT`, so even when the layer *is* loaded, nothing in-process observes its
/// output — it goes wherever the layer's default handler sends it. So a clean run was uninformative
/// twice over: the layer might not be there, and we were not listening.
///
/// This probe closes both halves. It attaches a messenger, so a caught error becomes a line we
/// print ourselves, and it reports the three states apart: armed, layer absent, no loader. Only
/// `Armed` licenses any claim about validation cleanliness.
///
/// # Safety
/// Every unsafe block below is a Vulkan entry point; the invariants are stated at each one.
fn probe_validation(plant: bool) -> ValidationProbe {
    use ash::vk;

    // SAFETY: opens the system Vulkan loader. No invariant beyond "the loader path is a valid
    // shared library", which is the OS loader's job.
    let entry = match unsafe { ash::Entry::load() } {
        Ok(e) => e,
        Err(e) => return ValidationProbe::NoLoader(format!("no Vulkan loader: {e}")),
    };

    // SAFETY: `entry` is live; a property query with no side effects.
    let layers = unsafe { entry.enumerate_instance_layer_properties() }.unwrap_or_default();
    let present = layers.iter().any(|l| {
        // SAFETY: `layer_name` is a NUL-terminated char array supplied by the loader.
        unsafe { std::ffi::CStr::from_ptr(l.layer_name.as_ptr()) == VALIDATION_LAYER }
    });
    if !present {
        return ValidationProbe::LayerAbsent;
    }

    let app = vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 0, 0));
    let layer_ptrs = [VALIDATION_LAYER.as_ptr()];
    let ext_ptrs = [ash::ext::debug_utils::NAME.as_ptr()];
    let mut messenger_ci = vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::GENERAL,
        )
        .pfn_user_callback(Some(validation_callback));
    // Chained into the instance create info as well, so violations committed *during*
    // instance creation and destruction are caught too — those are outside the messenger's
    // own lifetime and would otherwise be invisible.
    let ci = vk::InstanceCreateInfo::default()
        .application_info(&app)
        .enabled_layer_names(&layer_ptrs)
        .enabled_extension_names(&ext_ptrs)
        .push_next(&mut messenger_ci);

    // SAFETY: `entry` is live and every borrowed array outlives `ci`.
    let instance = match unsafe { entry.create_instance(&ci, None) } {
        Ok(i) => i,
        Err(e) => return ValidationProbe::NoLoader(format!("vkCreateInstance failed: {e:?}")),
    };

    let debug_utils = ash::ext::debug_utils::Instance::new(&entry, &instance);
    // SAFETY: `instance` was created with the debug-utils extension enabled, and `messenger_ci`
    // is a fully populated create-info whose callback has static lifetime.
    let messenger = unsafe { debug_utils.create_debug_utils_messenger(&messenger_ci, None) }.ok();

    if plant {
        // THE PLANTED VIOLATION.
        //
        // `vkCreateDebugUtilsMessengerEXT` with empty `messageSeverity` and `messageType` masks
        // violates VUID-VkDebugUtilsMessengerCreateInfoEXT-messageSeverity-requiredbitmask (and
        // the matching messageType one). It is chosen for four properties:
        //
        //  1. It is a *stateless* parameter check, so it is caught with certainty by any build of
        //     the validation layer, on any ICD, including lavapipe.
        //  2. It cannot corrupt anything — nothing is allocated, bound, submitted or executed.
        //  3. It needs **no logical device and no physical device**. That matters: if the plant
        //     needed a device, a machine with no capable GPU would look exactly like a machine
        //     with no validation, which is the precise conflation this control exists to prevent.
        //  4. It exercises the same extension the messenger itself uses, so a pass proves the
        //     capture path is live and not merely that an instance was created.
        eprintln!("epctl: committing the planted violation now");
        let bad = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::empty())
            .message_type(vk::DebugUtilsMessageTypeFlagsEXT::empty())
            .pfn_user_callback(Some(validation_callback));
        // SAFETY: `instance` was created with the debug-utils extension enabled. `bad` is a fully
        // initialised create-info that is deliberately invalid; validation intercepts it. Any
        // handle it returns is destroyed immediately below and never otherwise used.
        if let Ok(m) = unsafe { debug_utils.create_debug_utils_messenger(&bad, None) } {
            // SAFETY: `m` was just created by this `debug_utils` and is destroyed exactly once.
            unsafe { debug_utils.destroy_debug_utils_messenger(m, None) };
        }
    }

    if let Some(m) = messenger {
        // SAFETY: `m` was created by this `debug_utils` on this instance and is destroyed once.
        unsafe { debug_utils.destroy_debug_utils_messenger(m, None) };
    }
    // SAFETY: `instance` is live and every child object created above has been destroyed.
    unsafe { instance.destroy_instance(None) };
    ValidationProbe::Armed
}

/// Print every validation message with a greppable marker.
///
/// # Safety
/// Called by the validation layer with a valid callback-data pointer, per the Vulkan spec.
unsafe extern "system" fn validation_callback(
    _severity: ash::vk::DebugUtilsMessageSeverityFlagsEXT,
    _types: ash::vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const ash::vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user: *mut std::ffi::c_void,
) -> ash::vk::Bool32 {
    if data.is_null() {
        return ash::vk::FALSE;
    }
    // SAFETY: the layer guarantees `data` points to a valid callback-data struct for the duration
    // of this call.
    let msg = unsafe { (*data).p_message };
    let text = if msg.is_null() {
        std::borrow::Cow::Borrowed("<no message>")
    } else {
        // SAFETY: `p_message` is a NUL-terminated UTF-8 string owned by the layer.
        unsafe { std::ffi::CStr::from_ptr(msg) }.to_string_lossy()
    };
    eprintln!("{VALIDATION_CAUGHT_MARKER}: {text}");
    ash::vk::FALSE
}

fn run_probe_validation(plant: bool) -> std::process::ExitCode {
    // A lane that skips the control silently is a lane without the control. Setting this turns
    // "cannot answer" into "failed", so an environment that quietly lost the layer is loud.
    let required = std::env::var_os("ONNXRUNTIME_EP_VULKAN_REQUIRE_VALIDATION").is_some_and(|v| {
        let v = v.to_string_lossy().to_ascii_lowercase();
        v == "1" || v == "true" || v == "yes" || v == "on"
    });
    let unavailable = if required {
        eprintln!(
            "epctl: ONNXRUNTIME_EP_VULKAN_REQUIRE_VALIDATION is set, so an unavailable \
             validation layer is a failure rather than a skip."
        );
        std::process::ExitCode::from(1)
    } else {
        std::process::ExitCode::from(3)
    };
    match probe_validation(plant) {
        ValidationProbe::Armed => {
            println!(
                "epctl: VALIDATION ARMED — VK_LAYER_KHRONOS_validation is installed, was enabled, \
                 and a VkDebugUtilsMessengerEXT is receiving its output.\n\
                 \x20 Only in this state does a clean run mean anything. Without the messenger the \
                 layer's output goes to its default handler and nothing in-process observes it, so \
                 'no errors surfaced' would be true of a run in which errors surfaced."
            );
            std::process::ExitCode::SUCCESS
        }
        ValidationProbe::LayerAbsent => {
            eprintln!(
                "epctl: VALIDATION LAYER ABSENT — the Vulkan loader is present but \
                 VK_LAYER_KHRONOS_validation is not installed, so nothing can be validated here.\n\
                 \x20 This is exit 3, not exit 1: it is the absence of an answer, not a failing \
                 one. Install the Vulkan SDK, or set VK_LAYER_PATH to a directory containing the \
                 layer's manifest."
            );
            unavailable
        }
        ValidationProbe::NoLoader(why) => {
            eprintln!("epctl: NO VULKAN LOADER — {why}\n\x20 Exit 3: no answer, not a bad answer.");
            unavailable
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
    let probe_validation_flag = args.iter().any(|a| a == "--probe-validation");
    let plant_violation = args.iter().any(|a| a == "--plant-violation");
    let require_device_memory = args.iter().any(|a| a == "--require-device-memory");

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
                && a.as_str() != "--probe-validation"
                && a.as_str() != "--plant-violation"
                && a.as_str() != "--require-device-memory"
        })
    {
        eprintln!("epctl: unrecognised argument `{bad}`");
        usage();
        return std::process::ExitCode::from(2);
    }

    if let Some(path) = counters_file {
        return check_counters_with(&path, required_dispatches, require_device_memory);
    }

    if probe_validation_flag {
        return run_probe_validation(plant_violation);
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

    /// A snapshot carrying `model_output_equivalence = MATCH`, as a correctly-run comparison gate
    /// would write it. This is the only state that produces `CounterVerdict::Pass`.
    fn snapshot_match(dispatches: u64) -> String {
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
        .to_json_with_equiv(counters::EQUIVALENCE_MATCH)
    }

    /// A snapshot carrying `model_output_equivalence = DIVERGENT`, as a run where the GPU
    /// reached the kernel but produced numerically wrong output.
    fn snapshot_divergent(dispatches: u64) -> String {
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
        .to_json_with_equiv(counters::EQUIVALENCE_DIVERGENT)
    }

    /// A snapshot carrying the ledger keys, as a real run with device memory enabled writes it.
    /// Uses MATCH as the base so the guard-band tests can test guard-band behavior independently
    /// of the equivalence check.
    fn snapshot_with_guard_band(dispatches: u64, in_guard_band: u64) -> String {
        let mut doc = snapshot_match(dispatches);
        let cut = doc.rfind('}').expect("json");
        doc.truncate(cut);
        doc = doc.trim_end().trim_end_matches('\n').to_string();
        doc.push_str(&format!(
            ",\n  \"pointers_observed\": 100,\n  \"pointers_interior\": 52,\n  \
             \"pointers_in_guard_band\": {in_guard_band}\n}}\n"
        ));
        doc
    }

    fn snapshot_with_late_frees(dispatches: u64, late: u64, live_at_release: u64) -> String {
        // Build on snapshot_match so the FreeAfterRelease test is self-contained.
        // Without MATCH the equivalence check fires before this path is reached.
        let mut doc = snapshot_match(dispatches);
        let cut = doc.rfind('}').expect("json");
        doc.truncate(cut);
        doc = doc.trim_end().trim_end_matches('\n').to_string();
        doc.push_str(&format!(
            ",\n  \"alloc_allocations\": 427,\n  \"alloc_frees\": 105,\n  \
             \"alloc_allocators_released\": 1,\n  \
             \"alloc_live_at_release_spans\": {live_at_release},\n  \
             \"alloc_frees_after_release\": {late}\n}}\n"
        ));
        doc
    }

    /// The still-live-handles warning used to end in an open disjunction. This is the number that
    /// closes it, so it needs the same plant-break-confirm treatment the guard band got.
    #[test]
    fn a_free_after_release_fails_the_lane_but_merely_live_handles_do_not() {
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target/epctl-late-free-test");
        std::fs::create_dir_all(&dir).expect("scratch dir");

        // 322 handles live at release and never freed afterwards. This is the state the real
        // 2.2 GB model produces on both vendors, and it must NOT fail a lane: ORT reclaims them
        // by destroying the session, and our reservation goes with the registry.
        let benign = dir.join("benign.json");
        std::fs::write(&benign, snapshot_with_late_frees(30, 0, 322)).expect("write");
        assert_eq!(
            read_counters(benign.to_str().expect("utf8"), 1),
            CounterVerdict::Pass { dispatches: 30 },
            "handles still live at release are ORT's lifetime, not our leak — failing on this \
             would make the check fire on every healthy model run and be switched off"
        );

        // One late Free is the whole difference, and it outranks a healthy dispatch count.
        let defect = dir.join("defect.json");
        std::fs::write(&defect, snapshot_with_late_frees(30, 1, 322)).expect("write");
        assert_eq!(
            read_counters(defect.to_str().expect("utf8"), 1),
            CounterVerdict::FreeAfterRelease { count: 1 },
            "a single Free after release means ORT held a pointer into a torn-down registry"
        );

        // Absence is not zero: a snapshot predating the alloc_frees_after_release key must not
        // read as a FreeAfterRelease fault. Use snapshot_match so the equivalence check doesn't
        // fire first and obscure the FreeAfterRelease absence-is-not-fault signal.
        let missing = dir.join("missing.json");
        std::fs::write(&missing, snapshot_match(30)).expect("write");
        assert_eq!(
            read_counters(missing.to_str().expect("utf8"), 1),
            CounterVerdict::Pass { dispatches: 30 },
            "a snapshot without the key predates it; absence is not a fault signal"
        );
    }

    #[test]
    fn a_guard_band_hit_fails_the_lane_even_when_plenty_of_dispatches_executed() {
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target/epctl-guard-band-test");
        std::fs::create_dir_all(&dir).expect("scratch dir");

        let clean = dir.join("clean.json");
        std::fs::write(&clean, snapshot_with_guard_band(30, 0)).expect("write");
        assert_eq!(
            read_counters(clean.to_str().expect("utf8"), 1),
            CounterVerdict::Pass { dispatches: 30 },
            "in-span interior pointers are normal and must not fail a lane"
        );

        let dirty = dir.join("dirty.json");
        std::fs::write(&dirty, snapshot_with_guard_band(30, 1)).expect("write");
        assert_eq!(
            read_counters(dirty.to_str().expect("utf8"), 1),
            CounterVerdict::OutOfBounds { count: 1 },
            "a pointer that ran off the end of an allocation must outrank a healthy dispatch \
             count — 30 dispatches of wrong answers is not a pass"
        );

        // A snapshot from a build without the ledger, or a run with device memory off, carries no
        // guard-band key. Its absence is not zero and not a fault for the guard-band check.
        // However, such a snapshot also predates `model_output_equivalence`, so it comes back
        // UNMEASURED (exit 3), not Pass. R7: absence of the instrument is absence of an answer.
        let old = dir.join("old.json");
        std::fs::write(&old, snapshot(30)).expect("write");
        assert_eq!(
            read_counters(old.to_str().expect("utf8"), 1),
            CounterVerdict::EquivalenceUnmeasured { dispatches: 30 },
            "a snapshot without model_output_equivalence predates the verdict field; \
             absence = UNMEASURED (exit 3), not a pass"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The default-flags spelling, which is what every check other than the device-memory one
    /// wants. Kept in the test module rather than beside the real function so production has
    /// exactly one entry point and cannot accidentally call the lenient one.
    fn read_counters(path: &str, required: u64) -> CounterVerdict {
        read_counters_with(path, required, false)
    }

    fn snapshot_with_tally(dispatches: u64, staged: u64, backed: u64, allocs: u64) -> String {
        // Use MATCH as base so the device-memory tests are testing device-memory behaviour,
        // not equivalence behaviour. The `mixed` and `host` cases exit before equivalence
        // (NotOnDevice); the `good` case must reach Pass, which requires MATCH.
        let mut doc = snapshot_match(dispatches);
        let cut = doc.rfind('}').expect("json");
        doc.truncate(cut);
        doc = doc.trim_end().trim_end_matches('\n').to_string();
        doc.push_str(&format!(
            ",\n  \"alloc_allocations\": {allocs},\n  \"alloc_staged_spans\": {staged},\n  \
             \"alloc_staged_bytes\": {},\n  \"alloc_device_backed_spans\": {backed}\n}}\n",
            staged * 4096
        ));
        doc
    }

    /// The durable form of the staging caveat.
    ///
    /// The one-shot WARN cannot stay truthful as device memory arrives: kept as written it
    /// over-warns on a nearly-all-device run, removed it under-warns on a partially staged one.
    /// So the claim moves to a flag a lane sets and a machine evaluates.
    #[test]
    fn require_device_memory_fails_a_staged_run_and_refuses_a_snapshot_that_cannot_answer() {
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target/epctl-device-memory-test");
        std::fs::create_dir_all(&dir).expect("scratch dir");

        // Fully device-backed: the only shape that passes.
        let good = dir.join("good.json");
        std::fs::write(&good, snapshot_with_tally(30, 0, 12, 12)).expect("write");
        assert_eq!(
            read_counters_with(good.to_str().expect("utf8"), 1, true),
            CounterVerdict::Pass { dispatches: 30 },
        );

        // Mixed. This is the state no fixed warning wording covers, and the one we are heading
        // into: most tensors on device, a few staged, and a timing that looks like a device
        // number.
        let mixed = dir.join("mixed.json");
        std::fs::write(&mixed, snapshot_with_tally(30, 2, 10, 12)).expect("write");
        assert_eq!(
            read_counters_with(mixed.to_str().expect("utf8"), 1, true),
            CounterVerdict::NotOnDevice {
                staged_spans: 2,
                staged_bytes: 8192,
                device_backed: 10,
                allocations: 12,
            },
            "a partially staged run must fail the flag: it is not a slow device measurement"
        );

        // Allocations happened and nothing was device-backed.
        let host = dir.join("host.json");
        std::fs::write(&host, snapshot_with_tally(30, 0, 0, 12)).expect("write");
        assert!(matches!(
            read_counters_with(host.to_str().expect("utf8"), 1, true),
            CounterVerdict::NotOnDevice { .. }
        ));

        // A snapshot with no tally cannot answer the question, and must not pass a check it did
        // not perform. Exit 3, the same distinction as everywhere else in this binary.
        let old = dir.join("old.json");
        std::fs::write(&old, snapshot(30)).expect("write");
        assert!(
            matches!(
                read_counters_with(old.to_str().expect("utf8"), 1, true),
                CounterVerdict::NoReport(_)
            ),
            "absent keys are not zero"
        );
        // ...but without the flag, the same snapshot still fails on equivalence (UNMEASURED),
        // because equivalence is always checked and the old snapshot has no verdict field.
        // Use snapshot_match to test the pure "device memory not required, dispatch ok" path.
        let old_match = dir.join("old_match.json");
        std::fs::write(&old_match, snapshot_match(30)).expect("write");
        assert_eq!(
            read_counters_with(old_match.to_str().expect("utf8"), 1, false),
            CounterVerdict::Pass { dispatches: 30 },
        );

        let _ = std::fs::remove_dir_all(&dir);
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

    #[test]
    fn json_str_reads_equivalence_field() {
        assert_eq!(
            json_str(&snapshot(7), counters::EQUIVALENCE_KEY).as_deref(),
            Some(counters::EQUIVALENCE_UNMEASURED),
            "to_json() always writes UNMEASURED by default"
        );
        assert_eq!(
            json_str(&snapshot_match(7), counters::EQUIVALENCE_KEY).as_deref(),
            Some(counters::EQUIVALENCE_MATCH)
        );
        assert_eq!(
            json_str(&snapshot_divergent(7), counters::EQUIVALENCE_KEY).as_deref(),
            Some(counters::EQUIVALENCE_DIVERGENT)
        );
        assert_eq!(json_str(&snapshot(7), "not_a_string_field"), None);
    }

    /// The three outcomes of the criterion-8 gate must be distinguishable for the same reason the
    /// loader probe's are: a lane that crashed before reporting must not be mistakable for a lane
    /// that reported honestly.
    ///
    /// Now five outcomes after §9.1.3: MATCH (pass), DIVERGENT (fail), UNMEASURED (no answer),
    /// TooFew (fail), and NoReport (no answer). UNMEASURED and NoReport share exit 3 because both
    /// represent the absence of an answer on the correctness question.
    #[test]
    fn counter_verdicts_separate_zero_from_no_report() {
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target/epctl-counter-gate-test");
        std::fs::create_dir_all(&dir).expect("scratch dir");

        // Pass requires MATCH, not just dispatches.
        let ran = dir.join("ran.json");
        std::fs::write(&ran, snapshot_match(3)).expect("write");
        assert_eq!(
            read_counters(ran.to_str().unwrap(), 1),
            CounterVerdict::Pass { dispatches: 3 }
        );

        // UNMEASURED with sufficient dispatches = exit 3 (no answer, not a pass).
        let unmeasured = dir.join("unmeasured.json");
        std::fs::write(&unmeasured, snapshot(3)).expect("write"); // to_json() → UNMEASURED
        assert_eq!(
            read_counters(unmeasured.to_str().unwrap(), 1),
            CounterVerdict::EquivalenceUnmeasured { dispatches: 3 },
            "enough dispatches but no comparison = exit 3; 'executed' ≠ 'executed correctly'"
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
        std::fs::write(&f, snapshot_match(5)).expect("write"); // MATCH so threshold test is pure
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

    /// §9.1.3: `compute_failures:0` and `DIVERGENT` are not contradictory. A command buffer that
    /// signals its fence but produces wrong numbers is an arithmetic bug. The EP sees no wrong
    /// answer at the dispatch level — only the Python oracle comparison can see it.
    #[test]
    fn divergent_equivalence_fails_even_with_sufficient_dispatches() {
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target/epctl-divergent-test");
        std::fs::create_dir_all(&dir).expect("scratch dir");

        let f = dir.join("divergent.json");
        std::fs::write(&f, snapshot_divergent(161)).expect("write");
        assert_eq!(
            read_counters(f.to_str().unwrap(), 1),
            CounterVerdict::EquivalenceDivergent { dispatches: 161 },
            "161 dispatches, compute_failures:0, but DIVERGENT = confirmed wrong answer, exit 1"
        );

        // This is the current state of the project: Phi-3.5 with 161 MatMulNBits dispatched,
        // compute_failures:0, vk_range=[0,0] vs cpu_range=[-13, 13]. The test that names this
        // state is test_phi35_vulkan_matches_cpu_logits (xfail strict=True).
        std::fs::remove_dir_all(&dir).ok();
    }

    /// R7 applied to the equivalence field: absence of the instrument must not read as either
    /// a pass or a fail. UNMEASURED = "we do not know" = exit 3, same as no file at all.
    #[test]
    fn unmeasured_equivalence_is_no_answer_not_a_pass() {
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target/epctl-unmeasured-test");
        std::fs::create_dir_all(&dir).expect("scratch dir");

        // Explicit UNMEASURED string.
        let f = dir.join("unmeasured.json");
        std::fs::write(&f, snapshot(50)).expect("write"); // to_json() → UNMEASURED
        assert_eq!(
            read_counters(f.to_str().unwrap(), 1),
            CounterVerdict::EquivalenceUnmeasured { dispatches: 50 },
            "UNMEASURED with plenty of dispatches = no answer on correctness = exit 3"
        );

        // An unknown future string also maps to UNMEASURED (R7: unknown ≠ known negative).
        let mut unknown_doc = snapshot_match(50);
        unknown_doc = unknown_doc.replace(
            counters::EQUIVALENCE_MATCH,
            "SOME_FUTURE_STATE_WE_DO_NOT_UNDERSTAND",
        );
        let g = dir.join("future-equiv.json");
        std::fs::write(&g, unknown_doc).expect("write");
        assert_eq!(
            read_counters(g.to_str().unwrap(), 1),
            CounterVerdict::EquivalenceUnmeasured { dispatches: 50 },
            "an unrecognised equivalence string from a newer writer maps to UNMEASURED, not fail"
        );

        std::fs::remove_dir_all(&dir).ok();
    }
}
