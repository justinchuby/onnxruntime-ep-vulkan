//! M0 criterion 3: the Vulkan validation layer runs clean — *and we can prove the check works*.
//!
//! # Why this file exists
//!
//! Criterion 3 originally read "the validation layer surfaces no errors". Morpheus refused it, on
//! the grounds that **"no errors surfaced" is precisely what a run with the validation layer not
//! loaded reports.** A green lane would then be indistinguishable from an absent check.
//!
//! Investigating it turned out worse than the objection stated. The engine requests
//! `VK_LAYER_KHRONOS_validation` but attaches no `VkDebugUtilsMessengerEXT`, so even with the
//! layer loaded nothing in-process observes its output — it goes to the layer's default handler.
//! A clean run was therefore uninformative *twice over*: the layer might not be there, and we
//! were not listening.
//!
//! So this harness does the same thing `tests/layering.rs` does for criterion 7: it plants a
//! deliberate violation and **fails if the mechanism does not catch it**. The control is what
//! makes the clean run meaningful. Without it, this suite would pass on a machine with no Vulkan
//! at all.
//!
//! # Scope, stated honestly
//!
//! The plant here lives inside `epctl`'s *own* instance. Passing therefore proves:
//!   * the validation layer is installed and loadable on this machine, and
//!   * a debug messenger receives its output and our capture path works.
//!
//! It does **not** prove that the EP's dispatch path has validation armed on *its* instance —
//! that instance is created in `vk/instance.rs`, which is Switch's file, and needs its own
//! env-gated plant. That gap is deliberate and named rather than papered over; a control that
//! quietly proves something adjacent to the claim is the failure mode this whole file is about.

use std::process::Command;

/// The literal the probe prints for every message the layer hands it. Shared as a constant with
/// `epctl` in spirit; duplicated here on purpose so a rename in one end fails the test loudly
/// rather than silently making the assertion unfalsifiable.
const MARKER: &str = "EPCTL-VALIDATION-CAUGHT";

/// Exit code the probe uses for "cannot answer" — no loader, or no validation layer installed.
const EXIT_UNAVAILABLE: i32 = 3;

struct ProbeRun {
    code: Option<i32>,
    stdout: String,
    stderr: String,
}

impl ProbeRun {
    fn caught_lines(&self) -> usize {
        self.stderr
            .lines()
            .chain(self.stdout.lines())
            .filter(|l| l.contains(MARKER))
            .count()
    }

    fn unavailable(&self) -> bool {
        self.code == Some(EXIT_UNAVAILABLE)
    }
}

fn run_probe(plant: bool) -> ProbeRun {
    run_probe_args(if plant { &["--plant-violation"] } else { &[] })
}

fn run_probe_args(extra: &[&str]) -> ProbeRun {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_epctl"));
    cmd.arg("--probe-validation");
    cmd.args(extra);
    // The require-flag is the *caller's* choice, not this harness's; clear it so an ambient
    // setting cannot change what the probe reports mid-suite.
    cmd.env_remove("ONNXRUNTIME_EP_VULKAN_REQUIRE_VALIDATION");
    let out = cmd
        .output()
        .expect("failed to spawn epctl — the test binary should have been built alongside it");
    ProbeRun {
        code: out.status.code(),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

/// True when the lane demands the control actually run, rather than tolerating a skip.
fn validation_required() -> bool {
    std::env::var("ONNXRUNTIME_EP_VULKAN_REQUIRE_VALIDATION")
        .map(|v| {
            let v = v.to_ascii_lowercase();
            v == "1" || v == "true" || v == "yes" || v == "on"
        })
        .unwrap_or(false)
}

fn skip_or_fail(run: &ProbeRun, what: &str) {
    assert!(
        !validation_required(),
        "ONNXRUNTIME_EP_VULKAN_REQUIRE_VALIDATION is set, so {what} must not be skipped.\n\
         The validation layer is unavailable on this machine, which means M0 criterion 3 cannot \
         be evaluated here at all. A lane that silently skips its own positive control is a lane \
         without one.\n\
         --- probe stderr ---\n{}",
        run.stderr
    );
    eprintln!(
        "SKIPPING {what}: the Vulkan validation layer is unavailable on this machine (epctl exit \
         {EXIT_UNAVAILABLE}). This is a skip, not a pass — nothing about validation cleanliness \
         has been established. Set ONNXRUNTIME_EP_VULKAN_REQUIRE_VALIDATION=1 to make this a \
         failure instead, which is what CI should do.\n--- probe stderr ---\n{}",
        run.stderr
    );
}

/// THE POSITIVE CONTROL. A deliberate violation must be caught, or every clean run below is
/// meaningless.
#[test]
fn a_planted_vulkan_violation_is_caught_by_the_validation_layer() {
    let run = run_probe(true);
    if run.unavailable() {
        skip_or_fail(&run, "the validation positive control");
        return;
    }
    assert_eq!(
        run.code,
        Some(0),
        "the probe itself failed rather than reporting a verdict.\nstdout:\n{}\nstderr:\n{}",
        run.stdout,
        run.stderr
    );
    assert!(
        run.caught_lines() > 0,
        "THE POSITIVE CONTROL FAILED. epctl committed a deliberate Vulkan API violation \
         (vkCreateDebugUtilsMessengerEXT with zero severity and type masks, \
         VUID-VkDebugUtilsMessengerCreateInfoEXT-messageSeverity-requiredbitmask) and no \
         '{MARKER}' line came back.\n\n\
         This does NOT mean the code is clean. It means the check does not work, and therefore \
         that any clean validation run we report proves nothing — 'no errors surfaced' is exactly \
         what a run with the layer not loaded, or with nothing listening to it, reports. Fix the \
         mechanism before believing any green from it.\n\n\
         stdout:\n{}\nstderr:\n{}",
        run.stdout,
        run.stderr
    );
}

/// THE CONTROL FOR THE GEMV ROW TILE'S `groupCountY` CLAMP (issue #7).
///
/// `ops::quant::matmul_nbits_gemv` clamps `groupCountY` to 65535 — the floor every Vulkan 1.1
/// implementation must offer — and `q_gemv.comp` covers a taller prefill with a y-grid-stride
/// loop rather than a taller grid. The clamp exists because exceeding
/// `maxComputeWorkGroupCount[1]` is *invalid*, not merely slow. A prefill that ran without
/// complaint is evidence for the clamp only if a dispatch above the limit *would* have produced
/// one, and the only way to know that is to make it happen.
///
/// Note on the VUID number: issue #7 cites `VUID-vkCmdDispatch-groupCountY-00418`, which is what
/// the limit was numbered under in older spec revisions. The layer shipped with Vulkan SDK
/// 1.4.350 reports it as `VUID-vkCmdDispatch-groupCountY-00387`. The assertion therefore matches
/// on the *limit name* — text that has been stable across every renumbering — rather than on a
/// number that has already moved once, because a control that silently stops matching is exactly
/// the failure mode this file exists to prevent.
#[test]
fn a_dispatch_above_the_workgroup_count_limit_is_caught_by_the_validation_layer() {
    let run = run_probe_args(&["--plant-dispatch-overflow"]);
    if run.unavailable() {
        skip_or_fail(&run, "the groupCountY dispatch-overflow control");
        return;
    }
    assert_eq!(
        run.code,
        Some(0),
        "the probe itself failed rather than reporting a verdict.\nstdout:\n{}\nstderr:\n{}",
        run.stdout,
        run.stderr
    );
    let caught: Vec<&str> = run
        .stderr
        .lines()
        .chain(run.stdout.lines())
        .filter(|l| l.contains("maxComputeWorkGroupCount[1]") && l.contains("groupCountY"))
        .collect();
    assert!(
        !caught.is_empty(),
        "THE CONTROL FAILED. epctl recorded a vkCmdDispatch whose groupCountY exceeds this \
         device's maxComputeWorkGroupCount[1] and the validation layer said nothing about it.\n\n\
         That does NOT mean the GEMV's y clamp is unnecessary. It means this machine cannot \
         demonstrate that exceeding the limit is observable, and therefore that a clean prefill \
         run here is not evidence the clamp is doing anything. `ops::quant::GEMV_MAX_GROUPS_Y` \
         and the y-grid-stride loop in `q_gemv.comp` are only justified while this control \
         fires.\n\n\
         stdout:\n{}\nstderr:\n{}",
        run.stdout,
        run.stderr
    );
}

/// The two plants must not be interchangeable: each asserts that ONE named VUID is observable.
#[test]
fn the_two_plants_are_mutually_exclusive_and_each_names_its_own_vuid() {
    let both = run_probe_args(&["--plant-violation", "--plant-dispatch-overflow"]);
    assert_eq!(
        both.code,
        Some(2),
        "planting two violations in one run must be refused. A run that caught 'a' validation \
         error would otherwise satisfy either control with the other's message, which is the \
         same conflation the device-free design of --plant-violation exists to avoid.\n\
         stdout:\n{}\nstderr:\n{}",
        both.stdout,
        both.stderr
    );

    // And the stateless plant must NOT be able to pass the dispatch control's assertion.
    let stateless = run_probe(true);
    if stateless.unavailable() {
        skip_or_fail(&stateless, "the cross-plant discrimination check");
        return;
    }
    assert!(
        stateless.caught_lines() > 0,
        "precondition: the stateless plant should have been caught"
    );
    assert!(
        !stateless
            .stderr
            .lines()
            .chain(stateless.stdout.lines())
            .any(|l| l.contains("maxComputeWorkGroupCount[1]")),
        "the stateless plant produced a groupCountY complaint, so the two controls are not \
         distinguishable and neither one names what it claims to.\nstderr:\n{}",
        stateless.stderr
    );
}

/// And the negative: with nothing planted, the same mechanism must stay silent. A control that
/// fires unconditionally is as useless as one that never fires.
#[test]
fn a_clean_run_produces_no_validation_errors() {
    let run = run_probe(false);
    if run.unavailable() {
        skip_or_fail(&run, "the clean-validation check");
        return;
    }
    assert_eq!(
        run.code,
        Some(0),
        "the probe itself failed rather than reporting a verdict.\nstdout:\n{}\nstderr:\n{}",
        run.stdout,
        run.stderr
    );
    assert_eq!(
        run.caught_lines(),
        0,
        "the validation layer reported errors on a run with nothing planted.\n\
         Because the companion test proves the mechanism catches a planted violation, these lines \
         are real findings rather than noise from a broken harness.\n\n\
         stdout:\n{}\nstderr:\n{}",
        run.stdout,
        run.stderr
    );
}

/// The require-flag must convert an unavailable layer from a skip into a failure. Otherwise CI
/// can lose the layer and keep reporting green forever, which is the same class of silence the
/// whole file exists to remove.
#[test]
fn the_require_flag_turns_an_unavailable_layer_into_a_failure() {
    let out = Command::new(env!("CARGO_BIN_EXE_epctl"))
        .arg("--probe-validation")
        .env("ONNXRUNTIME_EP_VULKAN_REQUIRE_VALIDATION", "1")
        // A path with no layer manifests in it: the loader finds no explicit layers here, so this
        // simulates a machine without the validation layer installed without touching the
        // machine's actual configuration.
        .env("VK_LAYER_PATH", env!("CARGO_MANIFEST_DIR"))
        .env("VK_ADD_LAYER_PATH", env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("failed to spawn epctl");
    let code = out.status.code();
    let stderr = String::from_utf8_lossy(&out.stderr);

    // Two outcomes are legitimate. On a loader that honours VK_LAYER_PATH exclusively the layer
    // disappears and the probe must fail (exit 1, never exit 3). On a loader that still finds the
    // layer through other means the probe is simply armed and exits 0. What must never happen is
    // exit 3 — a skip — while the flag is set.
    assert_ne!(
        code,
        Some(EXIT_UNAVAILABLE),
        "ONNXRUNTIME_EP_VULKAN_REQUIRE_VALIDATION was set and epctl still exited {EXIT_UNAVAILABLE} \
         (the 'cannot answer' code). With the flag set, an unavailable layer must be a failure, \
         not a skip.\nstderr:\n{stderr}"
    );
    assert!(
        matches!(code, Some(0) | Some(1)),
        "unexpected exit code {code:?} from the probe under the require flag.\nstderr:\n{stderr}"
    );
}
