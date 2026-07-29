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
