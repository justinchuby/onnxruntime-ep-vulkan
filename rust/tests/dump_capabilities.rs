//! `epctl --dump-capabilities` is a user-facing contract, not a debug print.
//!
//! `DESIGN.md` §1.4 C2 requires the ORT release a contrib claim predicate was written against to
//! be *surfaced*. That makes this output something a user reads and a CI job diffs, so it gets
//! tests: an undetected regression here is a silent loss of the only signal C2 provides.

use std::process::Command;

fn run(args: &[&str]) -> (String, String, i32) {
    let out = Command::new(env!("CARGO_BIN_EXE_epctl"))
        .args(args)
        .output()
        .expect("run epctl");
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
        out.status.code().unwrap_or(-1),
    )
}

#[test]
fn dump_reports_the_ort_version_window() {
    let (stdout, _, code) = run(&["--dump-capabilities"]);
    assert_eq!(code, 0);
    assert!(
        stdout.contains("built against ort-1.28.0 (api 28)"),
        "capability dump must state the ORT release it was built against:\n{stdout}"
    );
    assert!(
        stdout.contains("minimum supported ort-1.24.0 (api 24)"),
        "capability dump must state the minimum supported ORT release:\n{stdout}"
    );
}

#[test]
fn dump_lists_every_registered_row_with_a_schema_baseline_column() {
    let (stdout, _, code) = run(&["--dump-capabilities"]);
    assert_eq!(code, 0);
    assert!(stdout.contains("schema baseline"), "{stdout}");

    let registered = onnxruntime_vulkan_ep::registry::all_specs().count();
    assert!(
        registered > 0,
        "the registry is empty; this test proves nothing"
    );
    for spec in onnxruntime_vulkan_ep::registry::all_specs() {
        let name = spec.qualified_name();
        assert!(
            stdout.contains(name.as_ref()),
            "row `{name}` is registered but absent from --dump-capabilities"
        );
    }
    assert!(stdout.contains(&format!("{registered} row(s)")), "{stdout}");
}

#[test]
fn json_output_is_well_formed_enough_to_diff() {
    let (stdout, _, code) = run(&["--dump-capabilities", "--json"]);
    assert_eq!(code, 0);
    assert!(stdout.trim_start().starts_with('{'), "{stdout}");
    assert!(stdout.trim_end().ends_with('}'), "{stdout}");
    assert!(stdout.contains("\"ort_api_version\": 28"), "{stdout}");
    assert!(stdout.contains("\"ort_api_version_min\": 24"), "{stdout}");
    assert_eq!(
        stdout.matches('{').count(),
        stdout.matches('}').count(),
        "unbalanced braces in JSON output"
    );
    // No trailing comma before the closing bracket of the ops array.
    assert!(
        !stdout.contains(",\n  ]"),
        "trailing comma in ops array:\n{stdout}"
    );
}

#[test]
fn every_contrib_row_is_printed_with_a_real_baseline() {
    // C2's surface. A baseline recorded in `sys::CONTRIB_SCHEMA_BASELINES` but not printed
    // satisfies the letter of the constraint and none of its purpose.
    let (stdout, _, code) = run(&["--dump-capabilities"]);
    assert_eq!(code, 0);
    assert!(
        !stdout.contains("MISSING"),
        "a contrib row is printed without a schema baseline:\n{stdout}"
    );
    for spec in onnxruntime_vulkan_ep::registry::all_specs() {
        if spec.domain.as_str().is_empty() {
            continue;
        }
        let line = stdout
            .lines()
            .find(|l| l.starts_with(spec.qualified_name().as_ref()))
            .unwrap_or_else(|| panic!("contrib row {} absent from dump", spec.qualified_name()));
        assert!(
            line.contains("ort-") && line.contains("verified "),
            "contrib row line carries no ORT baseline: {line}"
        );
    }
}

/// **The kernel boolean is `has_kernel`, and the row never spells `live` twice.**
///
/// §8.9.25 ruling (6): the JSON row carried a `status` token whose `live` value is the deprecated
/// `OpStatus::Live` alias — which grants nothing — beside a boolean *also* named `live` meaning
/// *this row has a kernel*, true for `Live` and `Ready` alike. Two denotations, one noun, in a
/// serialised schema. A reader checking "76 rows carry a kernel" against the field literally
/// named `live` got 46, and the collision was found by it happening to someone.
///
/// This test is deliberately about the NAME and not only the value: a rename that left the old
/// key in place would satisfy every value assertion and none of the ruling.
#[test]
fn the_kernel_boolean_is_named_has_kernel_and_live_is_only_a_status_token() {
    let (stdout, _, code) = run(&["--dump-capabilities", "--json"]);
    assert_eq!(code, 0);

    let rows = stdout.matches("\"name\": ").count();
    assert!(
        rows > 0,
        "the dump has no op rows; this test proves nothing"
    );
    assert_eq!(
        stdout.matches("\"has_kernel\": ").count(),
        rows,
        "every row must carry `has_kernel`:\n{stdout}"
    );
    assert!(
        !stdout.contains("\"live\":"),
        "the retired boolean name `live` is still a key in the dump — a noun retired in prose \
         and left in a schema is retired in the one place people quote from:\n{stdout}"
    );

    let has_kernel = stdout.matches("\"has_kernel\": true").count();
    let live_token = stdout.matches("\"status\": \"live\"").count();
    let ready_token = stdout.matches("\"status\": \"ready\"").count();
    let staged_token = stdout.matches("\"status\": \"staged\"").count();

    // The token stays three-valued. That is the half of the ruling that is a promise NOT to
    // change anything: `status` is what five consumers parse.
    assert_eq!(
        live_token + ready_token + staged_token,
        rows,
        "`status` must be exactly one of live/ready/staged on every row:\n{stdout}"
    );
    assert_eq!(
        has_kernel,
        live_token + ready_token,
        "`has_kernel` must be true for exactly the Live and Ready rows"
    );
    assert_eq!(
        stdout.matches("\"has_kernel\": false").count(),
        staged_token,
        "a staged row has no kernel, by definition of staged"
    );
    // And the rename is load-bearing rather than cosmetic only while the two counts differ.
    // Guarded rather than asserted flat: if every `Ready` row is ever promoted the collision
    // stops being observable, and a test that then went red would be reporting on the registry
    // instead of on the schema.
    if ready_token > 0 {
        assert_ne!(
            has_kernel, live_token,
            "the two questions must be distinguishable while a Ready row exists"
        );
    }

    let registered = onnxruntime_vulkan_ep::registry::all_specs().count();
    let with_kernel = onnxruntime_vulkan_ep::registry::all_specs()
        .filter(|s| s.is_live())
        .count();
    assert_eq!(rows, registered, "the dump must carry every registered row");
    assert_eq!(
        has_kernel, with_kernel,
        "`has_kernel` must agree with `OpSpec::is_live`, which is the predicate it renames"
    );
}

/// The human summary carried the same collision: it called the kernel-carrying count "live"
/// while the status column beside it spelled `live` for a strict subset of those rows.
#[test]
fn the_human_summary_names_the_predicate_it_counted() {
    let (stdout, _, code) = run(&["--dump-capabilities"]);
    assert_eq!(code, 0);

    let registered = onnxruntime_vulkan_ep::registry::all_specs().count();
    let with_kernel = onnxruntime_vulkan_ep::registry::all_specs()
        .filter(|s| s.is_live())
        .count();
    assert!(
        stdout.contains(&format!(
            "{registered} row(s): {with_kernel} with a kernel ("
        )),
        "the summary must say WHICH question the count answers:\n{stdout}"
    );
    let summary = stdout
        .lines()
        .find(|l| l.contains("with a kernel ("))
        .expect("summary line");
    assert!(
        summary.contains(&format!("{} staged", registered - with_kernel)),
        "the summary must decompose to the row count: {summary}"
    );
}

/// `--dump-weight-sites` is the surface `ci`/`tests/ops` reads the anchor table through.
///
/// Issue #73 made anchor eligibility a node property backed by a table of schema-designated
/// weight sites. `tests/ops/test_weight_sites.py` audits that table against the installed `onnx`
/// and `onnxruntime` packages, and it must audit the table the *binary* carries — a source
/// scraper would happily certify a table that no build ever consumed. These tests guard the
/// surface that makes that possible.
#[test]
fn the_weight_site_dump_carries_every_row_of_the_shipped_table() {
    let (stdout, _, code) = run(&["--dump-weight-sites"]);
    assert_eq!(code, 0);
    let table = onnxruntime_vulkan_ep::ops::partition::WEIGHT_SITES;
    assert!(
        !table.is_empty(),
        "the table is empty; this test proves nothing"
    );
    for site in table {
        let needle = format!(
            "{:<36} {:>3}  {:<22}",
            site.qualified_op, site.index, site.name
        );
        assert!(
            stdout.contains(&needle),
            "`{} [{}] {}` is in the shipped table and absent from the dump",
            site.qualified_op,
            site.index,
            site.name
        );
    }
    let designated = table.iter().filter(|s| s.designated()).count();
    assert!(
        stdout.contains(&format!(
            "{} operand(s) over {} heavy-op families; {designated} are designated weight sites",
            table.len(),
            onnxruntime_vulkan_ep::ops::partition::heavy_op_families().len()
        )),
        "the summary must decompose to the table it printed:\n{stdout}"
    );
}

#[test]
fn the_weight_site_json_is_well_formed_and_marks_designation_per_row() {
    let (stdout, _, code) = run(&["--dump-weight-sites", "--json"]);
    assert_eq!(code, 0);
    assert!(stdout.trim_start().starts_with('{'), "{stdout}");
    assert!(stdout.trim_end().ends_with('}'), "{stdout}");
    assert_eq!(
        stdout.matches('{').count(),
        stdout.matches('}').count(),
        "unbalanced braces in JSON output"
    );
    assert!(!stdout.contains(",\n  ]"), "trailing comma:\n{stdout}");

    let table = onnxruntime_vulkan_ep::ops::partition::WEIGHT_SITES;
    assert_eq!(stdout.matches("\"index\": ").count(), table.len());
    assert_eq!(
        stdout.matches("\"designated\": true").count(),
        table.iter().filter(|s| s.designated()).count()
    );
    assert_eq!(
        stdout.matches("\"designated\": false").count(),
        table.iter().filter(|s| !s.designated()).count()
    );
    // Both polarities must be present, or the field distinguishes nothing.
    assert!(stdout.contains("\"designated\": true"));
    assert!(stdout.contains("\"designated\": false"));
    // Every row carries its justification: an unexplained designation is the thing this table
    // exists to stop being possible.
    assert_eq!(stdout.matches("\"reason\": \"").count(), table.len());
}

#[test]
fn no_arguments_is_an_error_not_an_empty_success() {
    let (_, stderr, code) = run(&[]);
    assert_eq!(code, 2, "bare invocation must fail loudly");
    assert!(stderr.contains("--dump-capabilities"), "{stderr}");
}

#[test]
fn an_unrecognised_argument_is_rejected() {
    let (_, stderr, code) = run(&["--dump-capabilities", "--dump-everything"]);
    assert_eq!(code, 2);
    assert!(stderr.contains("unrecognised argument"), "{stderr}");
}
