//! Host-free tests for the live-device corroboration's **report rendering** (issue #88).
//!
//! The probe in `tests/device_object_counter_corroboration.rs` can only run its interesting
//! branches on a machine with the right driver. That is exactly the shape of a guard that is
//! never falsified: the `SKIP` and `INCONCLUSIVE` renderings would be written once, seen once,
//! and drift freely afterwards. So the report model lives in `tests/corroboration/mod.rs` with no
//! Vulkan in it, and every branch is exercised here with no device.
//!
//! Both required polarities are covered:
//!
//! * **failure-observed** — the driver refused, and the contradiction branch where a failure
//!   still produced an object;
//! * **lenient / precondition-not-met** — the driver served the over-allocation, and the several
//!   ways the probe can find no device at all.

#[path = "corroboration/mod.rs"]
mod corroboration;

use corroboration::{
    AUTHORITY_NOTE, Exhaustion, Report, TOKEN_CONTRADICTED, TOKEN_CORROBORATED, TOKEN_INCONCLUSIVE,
    TOKEN_SKIP, classify,
};

/// The failure-observed polarity: a refusal that produced no object corroborates, and exits 0.
#[test]
fn a_refusal_that_produced_no_object_corroborates_without_claiming_more() {
    let r = classify("Test Adapter", Exhaustion::Refused, false);
    assert_eq!(
        r,
        Report::Corroborated {
            device: "Test Adapter".to_string()
        }
    );
    assert_eq!(r.token(), TOKEN_CORROBORATED);
    assert_eq!(r.exit_code(), 0);
    let text = r.render();
    assert!(text.contains("REFUSED"), "got: {text}");
    assert!(text.contains("Test Adapter"), "the device must be named");
    assert!(
        text.contains("not guaranteed to exist on any other"),
        "a corroboration on one driver must not be rendered as a general guarantee: {text}"
    );
}

/// The one failing polarity, and the only one that may exit non-zero.
#[test]
fn a_failure_that_still_produced_an_object_is_the_only_red() {
    let r = classify("Test Adapter", Exhaustion::Refused, true);
    assert_eq!(r.token(), TOKEN_CONTRADICTED);
    assert_eq!(
        r.exit_code(),
        1,
        "a call that reported failure and produced an object breaks the premise every \
         success-only counter rests on"
    );
    assert!(r.render().contains("also produced an object"));
}

/// The lenient polarity: a conformant driver that serves the over-allocation is NOT a failure.
#[test]
fn a_lenient_driver_is_inconclusive_and_is_not_a_failure() {
    for handle_written in [true, false] {
        let r = classify("Test Adapter", Exhaustion::Allowed, handle_written);
        assert_eq!(r.token(), TOKEN_INCONCLUSIVE, "handle={handle_written}");
        assert_eq!(
            r.exit_code(),
            0,
            "turning a conformant driver red trains readers to ignore this binary"
        );
        let text = r.render();
        assert!(
            text.contains("not a spec-guaranteed failure"),
            "the report must state WHY it could not conclude: {text}"
        );
        assert!(
            text.contains("conformant driver, not a defect"),
            "got: {text}"
        );
    }
}

/// The precondition-not-met polarity: nothing observed, and the report must say so in those words.
#[test]
fn a_skip_says_nothing_was_observed_and_is_never_rendered_as_a_pass() {
    let r = Report::Skip {
        reason: "no Vulkan loader on this host (LoadingError)".to_string(),
    };
    assert_eq!(r.token(), TOKEN_SKIP);
    assert_eq!(r.exit_code(), 0);
    let text = r.render();
    assert!(text.contains("precondition not met"), "got: {text}");
    assert!(
        text.contains("NOTHING WAS OBSERVED"),
        "R7/R12: absence of an observation is not a negative result, and the artifact must not \
         let a reader mistake one for the other: {text}"
    );
    assert!(
        text.contains("must never be read as one"),
        "the skip must forbid the misreading explicitly: {text}"
    );
    assert!(
        !text.contains("CORROBORATED"),
        "a skip that contains the corroborated token can be grepped into a pass: {text}"
    );
}

/// Every outcome carries the authority note, and the tokens do not alias each other.
#[test]
fn every_report_names_the_authoritative_tests_and_uses_a_distinct_token() {
    let all = [
        Report::Skip {
            reason: "r".to_string(),
        },
        classify("d", Exhaustion::Refused, false),
        classify("d", Exhaustion::Allowed, false),
        classify("d", Exhaustion::Refused, true),
    ];
    for r in &all {
        let text = r.render();
        assert!(
            text.contains(AUTHORITY_NOTE),
            "{} dropped the authority note",
            r.token()
        );
        assert!(
            text.contains("host-free seam tests"),
            "the report must point at what actually settles the counter rule"
        );
        assert!(
            text.contains("NOT spec-guaranteed"),
            "the report must state the limit of what descriptor-pool exhaustion can prove"
        );
        assert!(text.starts_with("[device-object-counter corroboration]"));
    }
    let tokens: Vec<&str> = all.iter().map(Report::token).collect();
    let mut sorted = tokens.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(
        sorted.len(),
        tokens.len(),
        "two outcomes share a token, so a reader cannot tell them apart: {tokens:?}"
    );
    assert_eq!(
        all.iter().filter(|r| r.exit_code() != 0).count(),
        1,
        "exactly one outcome may fail the process"
    );
}

/// **The structural lock.** The corroboration target must stay `harness = false`.
///
/// A libtest `#[test]` that passes has its stdout and stderr captured and discarded. If this
/// declaration is ever dropped, the corroboration keeps compiling, keeps "passing", and its
/// SKIP / INCONCLUSIVE verdicts become invisible in the default hosted command — which is the
/// precise failure this design exists to avoid, and it would leave no trace anywhere else.
#[test]
fn the_corroboration_target_is_declared_harness_false_so_its_verdict_is_visible() {
    let manifest = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml"))
        .expect("rust/Cargo.toml must be readable from its own test");
    let block = manifest
        .split("[[test]]")
        .find(|b| b.contains("device_object_counter_corroboration"))
        .unwrap_or_else(|| {
            panic!("no [[test]] section declares device_object_counter_corroboration:\n{manifest}")
        });
    assert!(
        block.contains("harness = false"),
        "the corroboration target lost `harness = false`; its output would be captured and its \
         SKIP/INCONCLUSIVE verdicts would vanish from the default hosted command:\n{block}"
    );
    // This target, by contrast, MUST stay a normal libtest target — it is the thing that
    // asserts the above, and a `harness = false` assertion target runs no assertions. Checked on
    // the `name = ` DECLARATION, not on a mention: the block above names this file in prose.
    assert!(
        !manifest
            .split("[[test]]")
            .any(|b| b.contains("name = \"corroboration_report\"")),
        "corroboration_report must remain an auto-discovered harness target or these assertions \
         stop running"
    );
}
