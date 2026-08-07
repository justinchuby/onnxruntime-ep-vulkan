"""M0 criterion 12 — wiring census (§10.0.1 R10).

DESIGN.md §10 M0 criterion 12 (added 2026-07-30T19:05:03-07:00):

  > Wiring census: every mechanism this table relies on is observed to have run;
  > a mechanism with no observation reports ``UNWIRED``.

R10 rule (DESIGN.md §10.0.1):

  > A mechanism's existence is a claim about the call graph, not about the source tree.
  > The falsifier for "X is wired" is an observation of an artifact X produced, whose
  > content varies with X's input.  It is never a reading of X's code, and never a flag
  > X's author set.

  > The uninvoked state must be reportable and distinct from the empty state.  A mechanism
  > with no observation in a run reports ``UNWIRED`` — which is not "produced nothing" and
  > not "not applicable".

  > The identity case is a failing state, not a passing one.  A mechanism that emits one
  > island per node is the identity function; ``island_count == claimed_count`` with both
  > > 1 is one line, and the degenerate case in which the transform does nothing must be an
  > explicit red state.

THE CENSUS MECHANISMS (per DESIGN.md §10 criterion 12 ruling):

  1. Partitioner (``partition::evaluate``)
     Observable: ``islands_offered`` and ``claimed_nodes`` from counters JSON.
     Wired when: ``islands_offered > 0`` after a session with ``claimed_nodes > 0``.
     Identity check: ``islands_offered == claimed_nodes`` is red when both > 1
     (partitioner ran but produced no merges — same as not running).
     Current state: WIRED (Mouse's fix landed, partition runs on GetCapability).

  2. GPU tracer (Niobe)
     Observable: trace JSON file produced when ``ONNXRUNTIME_EP_VULKAN_TRACE_FILE`` is set.
     Wired when: file exists and contains at least one span entry.
     Current state: reported per-run (opt-in, so UNWIRED when env var is absent).
     Census reports OPTIONAL-UNWIRED — not a hard failure (the tracer is opt-in by design).

  3. ``model_output_equivalence``
     Observable: the ``model_output_equivalence`` token **and** the
     ``model_output_equivalence_record`` object in the counters JSON.
     Wired when: a record exists carrying both a verdict and its ``executed_by`` frame
     (criterion 12 (g), §10.0 third amendment — a verdict without its executor is a value
     from a world the census has not identified).
     Three states (criterion 12 (h), R13): OBSERVED / UNWIRED / INSTRUMENT-ERROR.
     Current state: written by ``test_phi35.py`` and ``test_criterion10.py`` when Phi-3.5
     is available; UNWIRED when the model cache is absent (non-dev machine).

  4. ``retain_viable`` (net-benefit gate, §7.0.2)
     Observable: ``viable_islands_retained`` in the C ABI counters (ABI version 2).
     Present even at 0 — distinguishable from UNWIRED (key absent) per R10. WIRED 2026-07-30.
     Owner: Mouse.

  5. §8.9 ledger lookup (claim-unproven gate)
     Observable: no ledger exists (criterion 11 not met).  UNWIRED.
     xfail(strict=True) — ledger has no entries; the gate cannot fire.
     Owner: Mouse / Trinity.

  6. Validation messenger (Switch)
     Observable: ``epctl --probe-validation`` exit 0 (ARMED).
     Wired when: messenger installed and layer catches violations.
     Current state: WIRED (Switch's session-16 fix; criterion-3a test confirms).

  7. Layering lint
     Observable: ``cargo test --test layering`` exit code.
     Wired when: tests run and report pass/fail.
     Current state: WIRED (CI step; confirmed by DESIGN.md criterion 7 MET).

COORDINATION:
  - Link built ``epctl --check-verdict`` using MATCH/DIVERGENT/UNMEASURED vocabulary.
    The census uses the same vocabulary.  One mechanism, not two (§10.0.1 R10 sub-rule).
  - Link's lavapipe lane needs the census too: emit the same lines in that lane.
  - Niobe owns the load guard for bench/ — do not invent a second one here.
"""

from __future__ import annotations

import ctypes
import contextlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
from onnx_ir import DataType as DT

import _models as m
import _verdict
import _watchdog
from _watchdog import (
    KIND_MECHANISM,
    KIND_TOOLCHAIN,
    STALLED,
    StallGuard,
    Stalled,
    WorkClock,
)

HERE = Path(__file__).parent
_REPO_ROOT = HERE.parent.parent
_CARGO_MANIFEST = _REPO_ROOT / "rust" / "Cargo.toml"

# ---------------------------------------------------------------------------
# Stall budgets — in WORK UNITS, never in seconds
# ---------------------------------------------------------------------------
#
# This test was excluded from the suite with `--ignore` because it timed out.  The wall
# clock is gone from the gate entirely; see `_watchdog.py` for why raising the number was
# not an option (R9 amendment 5: a threshold that moves the same way for "loaded" and for
# "hung" cannot be repaired by moving it).
#
# A budget below is the amount of work THIS MACHINE completes, measured during this run,
# that a step is allowed to spend producing no output and reaching no result.  Because it
# is denominated in work rather than time it needs no per-machine tuning: a box running
# four other agents ticks the reference clock more slowly, so the same budget is a longer
# wall window there, automatically and in proportion.
#
# Calibration: every run records what each step actually cost in units AND the largest
# silence it actually produced, into `observed_units` / `observed_max_silence_units` in the
# census artifact, so these numbers are auditable against the machine rather than asserted.
# The budget is compared against SILENCE, not against the step's total cost — the
# subprocess steps beat on every output line — so `observed_max_silence_units` is the
# number to read when judging headroom.
#
# Measured on this host (dev0) across the four cells of probe_stall_guard.py, which is
# also where the budgets were last wrong.  The first calibration read max silences of 327
# (quiet) and 11221 (loaded) for a counters child, i.e. a 34x spread and only 1.4x headroom
# under load — a budget resting on margin, not on invariance.  The cause was that the
# child's dominant silent window was its own module imports, before any of its code could
# announce anything.  Giving the child `-X importtime` turned that window into a stream of
# genuine progress lines and the quiet max silence fell to 52/65 units.  The budgets below
# are set against the measured silence, not against the step's total cost.
#
# Margin still exists for the one thing a CPU-bound reference unit does NOT measure: GPU
# and disk contention slow a step without slowing the clock.  probe_stall_guard.py
# quantifies that gap per run (`load_witness`) rather than leaving it as a claim.
_BUDGET_UNITS = {
    "counters_child_clean": 6_000,
    "counters_child_inject": 6_000,
    "counters_child_ledger": 6_000,
    # The two flag-frame arms (round 29).  Both run the same six-node chain as the clean
    # arm; `flags_a` additionally raises the child's log severity, which makes it noisier
    # and therefore *harder* to starve of beats, so the clean arm's budget is the
    # conservative choice rather than a guess.
    "counters_child_flags_a": 6_000,
    "counters_child_flags_b": 6_000,
    "add_session_profiling": 4_000,
    "validation_probe": 2_000,
    "layering_lint": 12_000,
}

#: Budget for a step nobody put in the table.  A caller that this file has never heard of
#: — a test added on another branch, which is exactly how `_run_counters_child` acquired a
#: caller it could not see — must still be guarded, so an unknown label gets the largest
#: counters-child budget rather than a `KeyError`.  Generous is safe here in a way it is
#: NOT safe for a wall-clock timeout: the unit is work, so a loose budget detects a hang
#: later in *work*, not never.  Contention cannot stretch it.
_DEFAULT_BUDGET_UNITS = 12_000


def _budget_for(label: str) -> int:
    return _BUDGET_UNITS.get(label, _DEFAULT_BUDGET_UNITS)

# ---------------------------------------------------------------------------
# Mechanism registry — what we census and what "wired" means for each.
# ---------------------------------------------------------------------------

_MECHANISMS = [
    "partitioner",
    "partition_identity_check",
    "net_benefit_gate",
    "gpu_tracer",
    "model_output_equivalence",
    "retain_viable",
    "ledger_lookup",
    "validation_messenger",
    "layering_lint",
    "broken_commitment_warn",
    "device_state_guard",
    "instrument_census",
    # Added round 29 (criterion 12).  Neither is a new instrument: both observe surfaces
    # that already existed in production Rust and that no census mechanism read.
    #
    #   ep_entrypoints — the three uncensused C ABI counters (`compile_calls`,
    #                    `compute_calls`, `subgraphs_stub`).  Their absence is why
    #                    "Compute() entered and dispatched nothing" was not a state the
    #                    census could report, and why a `compute_calls` reading quoted in
    #                    conversation was a number no census mechanism watched.
    #   flag_frame     — the nine uncensused environment switches, including the two that
    #                    select a different code path.  Frame disclosure for all nine,
    #                    plus a discrimination arm wherever one can exist in this frame.
    "ep_entrypoints",
    "flag_frame",
]

# Mechanisms that MUST be wired for M0 to be complete.  Others are informational.
_MANDATORY_WIRED = {
    "partitioner",
    "partition_identity_check",
    "net_benefit_gate",
    "validation_messenger",
    "layering_lint",
    "broken_commitment_warn",
    "device_state_guard",
    "instrument_census",
    # Promoted 2026-08-02 (round 28).  Criterion 11 landed: the ledger has entries, the
    # lookup runs, and `test_ledger_hits_moves_with_its_input` shows the reading changes
    # with the key presented.  Leaving it informational would mean a ledger that stopped
    # being consulted could not fail this lane — the exact state R10 calls
    # indistinguishable from never having been written.
    "ledger_lookup",
    # Added round 29.  `ep_entrypoints` reads three C ABI counters the EP publishes
    # unconditionally; a counters artifact that does not carry them means the ABI moved
    # under the census, which is a finding about wiring and not about this run's model.
    # `flag_frame`'s frame disclosure cannot fail for environmental reasons — it reads
    # this process's own environment — so making it mandatory adds a red that only a real
    # regression can produce.
    "ep_entrypoints",
    "flag_frame",
}

# Mechanisms known to be UNWIRED at M0.
#
# Emptied 2026-08-02 (round 28) when `ledger_lookup` was promoted.  Kept as an empty set
# rather than deleted: the subtraction below is the place a future known-unwired mechanism
# is declared, and removing the seam would mean the next one is declared by deleting an
# assertion instead.
_KNOWN_UNWIRED_M0: set[str] = set()


# Which mechanisms a stalled step makes unobservable.  A stall is recorded against the
# mechanisms that step was the sole observation for, so the census says WHICH mechanism
# stopped rather than only that the census did.
_STEP_MECHANISMS = {
    "counters_child_clean": ("net_benefit_gate", "broken_commitment_warn", "retain_viable"),
    "counters_child_inject": ("net_benefit_gate", "broken_commitment_warn", "retain_viable"),
    "counters_child_flags_a": ("flag_frame",),
    "counters_child_flags_b": ("flag_frame",),
    "add_session_profiling": ("partitioner", "partition_identity_check", "gpu_tracer",
                              "model_output_equivalence"),
    "validation_probe": ("validation_messenger",),
    "layering_lint": ("layering_lint",),
}


@pytest.fixture
def census_guard():
    """A work clock plus a stall guard, live for the duration of one census.

    The clock is started before anything else so that "the machine did nothing" and "the
    machine did everything except the census" are distinguishable from the first step —
    without a running denominator the guard could only report silence, and silence alone
    is what a wall-clock timeout reports.
    """
    clock = WorkClock().start()
    guard = StallGuard(clock=clock, what="wiring census")
    try:
        yield guard
    finally:
        clock.stop()


def _record_stall(observations: dict, step: str, exc: Stalled) -> None:
    """Score a stall against the mechanisms it made unobservable.

    `STALLED` is deliberately not `UNWIRED`.  "Ran and produced nothing" is a finding about
    the call graph; "never came back" is a finding about a step.  A census that spells them
    the same way has lost the one it exists to report (R12).
    """
    text = exc.args[0].splitlines()[0]
    for mech in _STEP_MECHANISMS.get(step, (step,)):
        observations[mech] = f"{STALLED} ({step}: {text})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _epctl_path() -> Path | None:
    ep_lib = os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB")
    if not ep_lib:
        return None
    ep_lib_path = Path(ep_lib).resolve()
    epctl_name = "epctl.exe" if sys.platform == "win32" else "epctl"
    candidate = ep_lib_path.parent / epctl_name
    return candidate if candidate.is_file() else None


def _cargo_env() -> dict[str, str]:
    env = dict(os.environ)
    sdk = env.get("VULKAN_SDK", "")
    if sdk:
        bin_dir = str(Path(sdk) / "Bin")
        path = env.get("PATH", "")
        if bin_dir not in path:
            env["PATH"] = bin_dir + os.pathsep + path
    return env


# The counter ABI mirror is DERIVED from rust/src/counters.rs, never written here.
#
# A hand-written `_EpCounters` lived at this spot and was wrong twice in one day. `a52024f`
# inserted `device_losses` mid-struct; `dispatches_executed` here silently became it, always `0`,
# so mechanism 1 reported `UNWIRED (EP ran nothing)` about a run that dispatched normally.
# `898a2ba` then inserted three `outputs_*` fields in the same place and `ledger_entries` read
# **0** against a true 97. Both readings were stable and plausible, which is why nothing went red.
#
# The generator existed for the second one and did not prevent it, because it co-existed with the
# mirrors it replaced. tests/ops/test_counters_abi_singleton.py now fails if a mirror comes back.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rust" / "tools"))
import counters_abi as _counters_abi  # noqa: E402

CountersAbiMismatch = _counters_abi.CountersAbiMismatch


def _read_ep_counters_via_ctypes() -> dict[str, int]:
    """Read the EP's live execution counters via OrtEpVulkanGetExecutionCounters (C ABI).

    This is the in-process path (test_phi35.py style).  It avoids the Windows UCRT env-var
    cache problem: the EP DLL reads ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE at init time, so
    setting the env var after DLL load is unreliable on Windows.  The C ABI call is always live.

    `{}` means only one thing -- ONNXRUNTIME_VULKAN_EP_LIB is unset. A layout mismatch RAISES and
    is deliberately not swallowed into the same `{}`, because `{}` differences to a delta of 0 and
    a delta of 0 is read as UNWIRED. That is how this defect stayed invisible for a day.
    """
    return _counters_abi.read_counters()


def _run_add_session_with_profiling() -> tuple[dict[str, int], dict[str, int]]:
    """Run a single Add session; return (counters_before, counters_after) and profiling data.

    Returns (ep_counters_before, ep_counters_after) as dicts.  The difference between them
    gives how many dispatches, compiles, etc. this session produced.

    Also returns (claimed_from_profiling, islands_from_profiling) via profiling JSON:
      - claimed_from_profiling: number of nodes where provider == EP_NAME
      - islands_from_profiling: number of unique VulkanEP subgraph names (fused node names)

    Returns (before, after, profile_info) where profile_info has 'claimed' and 'islands'.
    """
    model = m.make_model(
        "Add",
        [m.tensor("a", DT.FLOAT, [4, 4]), m.tensor("b", DT.FLOAT, [4, 4])],
        [m.tensor("out", DT.FLOAT, [4, 4])],
    )
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.enable_profiling = True
    opts.profile_file_prefix = "_census_probe"

    counters_before = _read_ep_counters_via_ctypes()

    try:
        sess = ort.InferenceSession(model, opts, providers=m.EP_PROVIDERS)
        feeds = {
            "a": np.ones((4, 4), dtype=np.float32),
            "b": np.ones((4, 4), dtype=np.float32),
        }
        sess.run(None, feeds)
        profile_path = sess.end_profiling()
    except Exception:
        return counters_before, {}, {}

    counters_after = _read_ep_counters_via_ctypes()

    profile_info: dict[str, int] = {"claimed": 0, "islands": 0}
    try:
        with open(profile_path) as fh:
            events = json.load(fh)
        ep_nodes = [
            e for e in events
            if e.get("cat") == "Node"
            and isinstance(e.get("args"), dict)
            and e["args"].get("provider") == m.EP_NAME
        ]
        claimed = len(ep_nodes)
        # Island = unique VulkanEP subgraph (each fused subgraph has a distinct node name).
        island_names = {e.get("name", "") for e in ep_nodes}
        profile_info = {"claimed": claimed, "islands": len(island_names)}
    except Exception:
        pass
    finally:
        try:
            os.remove(profile_path)
        except OSError:
            pass

    return counters_before, counters_after, profile_info


# ---------------------------------------------------------------------------
# THE COUNTERS-FILE CHILD — how today's mechanisms are reached at all
#
# The net-benefit gate's three states, the broken-commitment WARN channel, and the
# three-state reading of `viable_islands_retained` are published in the counters JSON and
# NOT in the C ABI struct (`counters.rs`: COUNTERS_ABI_VERSION is still 2 and
# `VulkanEpCounters` ends at `viable_islands_retained` — the additions were deliberately
# JSON-only, "no abi_version bump: the C struct is unchanged").  So `ctypes` cannot see
# them, and the env var that names the counters file is read by the DLL at load time,
# which on Windows means setting it from this process after `import onnxruntime` is
# invisible (UCRT env cache).
#
# Therefore: a fresh child per polarity, env set before the DLL is loaded, and the census
# reads the file the child left behind.  This is also what makes the lines below
# *observations* rather than readings of our own configuration.
# ---------------------------------------------------------------------------

_CHILD_FLAG = "--census-counters-child"

#: Tank's fault-injection switch (rust/tools/probe_broken_commitment.py::ENV_INJECT).
#: Used here to give the broken-commitment WARN an input to vary with; the census does not
#: reimplement his judgement, it reads the counters his call site writes.
_ENV_INJECT = "ONNXRUNTIME_EP_VULKAN_FORCE_COMPUTE_FAILURE"

#: Niobe's tracer switch (rust/src/trace.rs::ENV_TRACE).  Named here because the census
#: previously looked for `ONNXRUNTIME_EP_VULKAN_TRACE_FILE`, which nothing sets.
_ENV_TRACE = "ONNXRUNTIME_EP_VULKAN_TRACE"

#: Criterion 11(c).  Names the model the counters child runs instead of `_chain_model()`.
#: `ledger_hits` is only a reading if it is *given something to vary with*, and the input
#: it must vary with is the proof key the graph presents — which is a property of the
#: model, not of the harness.  A census that only ever runs one graph can report
#: `ledger_hits=6 proven_key_lookups=6` forever and cannot tell a consulted ledger from
#: one derived from the same enumeration that produced the claims.
_ENV_CENSUS_MODEL = "ONNXRUNTIME_EP_VULKAN_CENSUS_MODEL"

#: Mouse's digest-refusal switch (rust/src/registry.rs::ENV_LEDGER_FILE).  Named here so
#: the census can put the refusal *in the lane* rather than behind a Rust unit test.
_ENV_LEDGER_FILE = "ONNXRUNTIME_EP_VULKAN_LEDGER_FILE"

_EVIDENCE_CASES = Path(__file__).parent.parent.parent / "evidence" / "cases"
_EVIDENCE_LEDGER = Path(__file__).parent.parent.parent / "evidence" / "proof_ledger.jsonl"

# ---------------------------------------------------------------------------
# Criterion 12, round 29 — the twelve surfaces the census did not observe
# ---------------------------------------------------------------------------
#
# Link's independent whole (`ci/check_census_completeness.py`, denominator enumerated from
# production Rust that this file does not write) found 50 instrumented surfaces, of which
# 33 routed to a census mechanism and TWELVE were instrumented and observed by no census
# mechanism at all.  `unwired: []` was therefore true against a denominator the census
# supplied itself — R11's shape, and the reason the answer to criterion 12 is not 12/12.
#
# Two of the twelve are worse than the other ten and are handled first, because they
# SELECT A DIFFERENT CODE PATH:
#
#   ONNXRUNTIME_EP_VULKAN_GEMV_PACKED   picks a different kernel (`ops/quant.rs`)
#   ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY picks device-local allocation (`factory.rs`)
#
# A census line that cannot say which of those paths was in force is reporting a value
# from a world it has not identified — which is `executed_by`'s own argument (§10.0 third
# metric amendment) applied one level up, to the census rather than to the verdict.
#
# TWO THINGS ARE BEING REPORTED PER SWITCH AND THEY ARE NOT THE SAME STRENGTH:
#
#   FRAME DISCLOSURE   — was this switch in force in the frame the census observed in?
#                        Always available, costs nothing, and is the weaker of the two.
#                        It makes every other line in the census non-silent about which
#                        world produced it.  It is NOT evidence the switch does anything.
#   DISCRIMINATION     — does an artifact this run produced MOVE when the switch is armed?
#                        That is R10's falsifier, and it is the only one of the two that
#                        can fail.  Where the census can afford an arm, it takes one.
#
# A switch whose discriminator cannot exist in the census's frame reads UNOBSERVABLE with
# the reason, never INVARIANT and never 0 (R12).  `GEMV_PACKED` is exactly that case here:
# the census graph is an elementwise chain with no GEMV node, so the kernel the switch
# selects is not reachable in this frame and "no difference" would be a fabricated
# observation.  Recording it as UNOBSERVABLE-with-a-reason is what §10 asks for when a
# surface is deliberately out of a mechanism's scope.
_ENV_DEVICE_MEMORY = "ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY"
_ENV_QUARANTINE_SPANS = "ONNXRUNTIME_EP_VULKAN_QUARANTINE_SPANS"
_ENV_VA_RESERVE_MIB = "ONNXRUNTIME_EP_VULKAN_VA_RESERVE_MIB"
_ENV_GEMV_PACKED = "ONNXRUNTIME_EP_VULKAN_GEMV_PACKED"
_ENV_BACKEND_PROBE = "ONNXRUNTIME_EP_VULKAN_BACKEND_PROBE"
_ENV_CLAIM_LOG = "ONNXRUNTIME_EP_VULKAN_CLAIM_LOG"
_ENV_CLAIM_DEBUG = "ONNXRUNTIME_EP_VULKAN_CLAIM_DEBUG"
_ENV_VERBOSE = "ONNXRUNTIME_EP_VULKAN_VERBOSE"
_ENV_DUMP_OUTPUT_BYTES = "ONNXRUNTIME_EP_VULKAN_DUMP_OUTPUT_BYTES"

#: Deliberately NOT in the `ONNXRUNTIME_EP_VULKAN_` namespace.  This is a parameter of the
#: census's own child process, read in Python, and naming it in the EP's namespace would
#: put a switch into the independent whole's regex that no line of Rust reads — growing
#: Link's denominator with a harness variable and making the census's coverage of the EP
#: look worse by an amount that has nothing to do with the EP.
_ENV_CHILD_LOG_SEVERITY = "CENSUS_CHILD_LOG_SEVERITY"

#: The six env switches that parameterise the net-benefit gate's cost model
#: (`rust/src/partition.rs`).  Named here so the gate's census line can state which of its
#: inputs were in force: a gate observed at its defaults and a gate observed under an
#: overridden cost model are two different observations, and a line that does not say
#: which one it is cannot be replayed.
_ENV_PARTITION = (
    "ONNXRUNTIME_EP_VULKAN_PARTITION_MARGIN",
    "ONNXRUNTIME_EP_VULKAN_PARTITION_MIN_NODES",
    "ONNXRUNTIME_EP_VULKAN_PARTITION_FLOPS_PER_NS",
    "ONNXRUNTIME_EP_VULKAN_PARTITION_FIXED_NS",
    "ONNXRUNTIME_EP_VULKAN_PARTITION_BYTES_PER_NS",
    "ONNXRUNTIME_EP_VULKAN_PARTITION_ANCHOR_EXEMPTION",
)

#: The three env switches that parameterise the validation messenger (`rust/src/vk/`).
_ENV_VALIDATION = (
    "ONNXRUNTIME_EP_VULKAN_VALIDATE",
    "ONNXRUNTIME_EP_VULKAN_REQUIRE_VALIDATION",
    "ONNXRUNTIME_EP_VULKAN_PLANT_VALIDATION_VIOLATION",
)

#: Niobe's ten `Phase` variants (`rust/src/trace.rs::enum Phase`), listed here so the
#: tracer's census line can say WHICH of them the trace file carried and which it did not.
#:
#: This list is the census's own, and it is meant to be: Link's screen enumerates the same
#: enum straight out of `trace.rs` with a regex, so if a variant is added, renamed or
#: removed in Rust his denominator moves and this constant does not, and his extent line
#: reports the difference.  Two authors, two files, two languages — a shared constant would
#: put numerator and denominator back under one pen, which is the defect criterion 12 is
#: about.
#:
#: `Record` is on this list on purpose.  It is the standing MISNAMED specimen: wired,
#: invoked, correct, input-varying, and wrong by 50x in what it was called.  A tracer line
#: that reports "N events" and never says whether `Record` was among them would certify it
#: again, which is the whole of Link's third question.
_TRACE_PHASES = (
    "Compile", "Prepack", "Upload", "Record", "DescAlloc",
    "PipelineLookup", "CmdUpload", "Submit", "FenceWait", "Readback",
)


def _phase_event_name(variant: str) -> str:
    """`Phase::CmdUpload` -> `vulkan.cmd_upload`, mirroring `trace.rs::Phase::as_str`.

    THE FIRST VERSION OF THIS MATCHER LOOKED FOR THE RUST IDENTIFIER and reported
    `0/10 Phase variants emitted` — a spectacular finding, entirely fabricated by the
    instrument.  The trace file spells them `vulkan.record`, `vulkan.fence_wait` and so on;
    six of the ten were right there in the artifact I was reading.

    R13's second corollary is the reason this is called out rather than quietly fixed: a
    result that CONFIRMS a prediction deserves more scrutiny than one that contradicts it.
    I went looking for a tracer coverage gap, got the largest gap available on the first
    try, and the number was my own bug.  A `0/N` is the single most suspicious reading an
    instrument can produce, because it is what a broken instrument produces too.
    """
    out = []
    for i, ch in enumerate(variant):
        if ch.isupper() and i:
            out.append("_")
        out.append(ch.lower())
    return "vulkan." + "".join(out)

#: Niobe's device-span switch.  The census does NOT arm it (device timestamps are Switch's
#: exclusive claim and Niobe's admissibility frame; a census that armed them would be
#: taking a device-clock reading under four-agent contention).  It is named on the tracer
#: line with its in-force state so that a reader knows which phases could not have been
#: emitted in this frame — R12 again: the host-side phases are observed, the device-side
#: ones are UNOBSERVABLE here, and neither is 0.
_ENV_TRACE_GPU = "ONNXRUNTIME_EP_VULKAN_TRACE_GPU"

_CENSUS_OUT = Path(__file__).parent.parent.parent / "bench" / "results" / "census"
_CENSUS_TRACE_PATH = _CENSUS_OUT / (
    f"census-trace-dev{os.environ.get('ONNXRUNTIME_EP_VULKAN_DEVICE', 'unset')}.json"
)
_CENSUS_BACKEND_PROBE_PATH = _CENSUS_OUT / (
    f"census-backend-probe-dev{os.environ.get('ONNXRUNTIME_EP_VULKAN_DEVICE', 'unset')}.txt"
)
_CENSUS_CLAIM_LOG_PATH = _CENSUS_OUT / (
    f"census-claim-log-dev{os.environ.get('ONNXRUNTIME_EP_VULKAN_DEVICE', 'unset')}.jsonl"
)


def _chain_model(n_nodes: int = 6) -> bytes:
    """A multi-node chain, so the partitioner and the net-benefit gate have something to do.

    A single `Add` cannot distinguish a partitioner that merges from one that is the
    identity function — R10 names that exact degeneracy — and it gives the net-benefit gate
    one trivial cluster.  Six chained elementwise nodes give both mechanisms a decision.
    """
    import onnx_ir as ir  # noqa: PLC0415

    a = m.tensor("a", DT.FLOAT, [4, 4])
    b = m.tensor("b", DT.FLOAT, [4, 4])
    nodes = []
    cur = a
    for i in range(n_nodes):
        out = m.tensor(f"t{i}", DT.FLOAT, [4, 4])
        op = "Add" if i % 2 == 0 else "Mul"
        nodes.append(ir.node(op, [cur, b], outputs=[out]))
        cur = out
    cur.name = "out"
    graph = ir.Graph(
        [a, b], [cur], nodes=nodes, name="census_chain", opset_imports={"": 21}
    )
    return ir.to_proto(ir.Model(graph, ir_version=10)).SerializeToString()


def _feeds_for(sess) -> dict:
    """Inputs for an arbitrary evidence-case model.

    Deliberately the *same* construction `rust/tools/gen_proof_ledger.py::_child_main`
    uses, so the graph the census runs is fed the way the graph the ledger was generated
    from was fed.  Two feed builders would be two definitions of the case.
    """
    rng = np.random.default_rng(20260801)
    feeds = {}
    for inp in sess.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 2 for d in inp.shape]
        dt = inp.type
        if "float16" in dt:
            feeds[inp.name] = rng.standard_normal(shape).astype(np.float16)
        elif "uint8" in dt:
            feeds[inp.name] = rng.integers(0, 4, shape).astype(np.uint8)
        elif "int64" in dt:
            feeds[inp.name] = rng.integers(0, 2, shape).astype(np.int64)
        elif "int32" in dt:
            feeds[inp.name] = rng.integers(0, 2, shape).astype(np.int32)
        elif "bool" in dt:
            feeds[inp.name] = rng.integers(0, 2, shape).astype(bool)
        else:
            feeds[inp.name] = rng.standard_normal(shape).astype(np.float32)
    return feeds


def _counters_child_main(counters_path: str) -> int:
    """Child entry point: run one session with the counters file armed, then exit.

    Each phase announces itself on stderr before it starts.  These lines are the child's
    forward-progress beats: `_watchdog.guarded_run` treats every output line as progress,
    so a child that is merely crawling under load keeps the parent's stall budget from
    advancing, while a child that has wedged inside a phase stops beating and is caught.
    They are deliberately tied to actual phase transitions rather than emitted on a timer
    — a timer-driven heartbeat would keep beating through the very hang it is supposed to
    reveal, which is the "detector as decoration" failure this whole design exists to
    avoid.
    """
    def phase(name: str) -> None:
        print(f"[census-child] phase={name}", file=sys.stderr, flush=True)

    phase("start")
    lib = os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB")
    if not lib:
        print("[census-child] ONNXRUNTIME_VULKAN_EP_LIB unset", file=sys.stderr)
        return 3
    try:
        phase("register_ep_library")
        ort.register_execution_provider_library(m.EP_NAME, str(Path(lib).resolve()))
    except Exception as exc:  # noqa: BLE001
        if "already registered" not in str(exc):
            print(f"[census-child] registration failed: {exc}", file=sys.stderr)
            return 3
    opts = ort.SessionOptions()
    # Default WARNING.  The flag-frame arms lower it, because three of the switches they
    # arm (CLAIM_DEBUG, DUMP_OUTPUT_BYTES, VERBOSE) publish through the log and nothing
    # else: at severity 2 ORT drops those records before they reach this process's stderr,
    # and the census would then read "the switch changed nothing" from a frame in which
    # its only observable could not appear.  That is R12's error exactly — a 0 reported
    # where the event cannot occur — so the severity is an explicit arm parameter rather
    # than a constant, and the observation says which severity it read at.
    opts.log_severity_level = int(os.environ.get(_ENV_CHILD_LOG_SEVERITY, "2"))
    phase("build_model")
    model_path = os.environ.get(_ENV_CENSUS_MODEL, "")
    if model_path:
        phase("create_session")
        sess = ort.InferenceSession(model_path, opts, providers=m.EP_PROVIDERS)
        feeds = _feeds_for(sess)
    else:
        model_bytes = _chain_model()
        phase("create_session")
        sess = ort.InferenceSession(model_bytes, opts, providers=m.EP_PROVIDERS)
        feeds = {
            "a": np.ones((4, 4), dtype=np.float32),
            "b": np.full((4, 4), 2.0, dtype=np.float32),
        }
    try:
        phase("run")
        sess.run(None, feeds)
    except Exception as exc:  # noqa: BLE001
        # An injected Compute failure is expected to surface here; the artifact the census
        # reads is the counters file, which `record_broken_commitment` writes at the
        # instant of the event and not at teardown.
        print(f"[census-child] run raised: {type(exc).__name__}: {exc}", file=sys.stderr)
    phase("teardown")
    del sess
    phase("done")
    return 0 if Path(counters_path).is_file() else 3


@contextlib.contextmanager
def _ambient_guard(what: str):
    """A short-lived clock+guard for a caller that did not bring one.

    Round 26 postmortem.  `guard` was a **required** keyword-only argument, and
    `test_ledger_lookup_wired` — which landed in this file from `squad/mouse` while the
    guard landed from `squad/trinity` — called the helper without it.  Both branches were
    green alone; the union raised
    ``TypeError: _run_counters_child() missing 1 required keyword-only argument: 'guard'``.

    The fix is a default, but note carefully *which* default.  ``guard=None`` meaning
    "run unguarded" would have been the wrong one: every future caller would silently opt
    out of stall detection and the subprocess steps would drift back to exactly the
    unguarded state round 25 removed, with nothing to notice. ``guard=None`` means
    **"build one for this call"** instead, so there is no unguarded path through this
    module at all. What an ambient guard loses is only bookkeeping — its costs and max
    silences do not land in the census-wide guard's ledger, so they are not in
    `observed_units` — never protection.
    """
    clock = WorkClock().start()
    try:
        yield StallGuard(clock=clock, what=what)
    finally:
        clock.stop()


def _run_counters_child(
    *, inject: bool, tag: str, guard: "StallGuard | None" = None,
    extra_env: "dict[str, str] | None" = None, trace: "bool | None" = None,
) -> tuple[dict, str]:
    """Run one polarity in a fresh process; return (counters doc, combined child log).

    *guard* is optional: a caller that does not have one gets a private clock and guard
    for the duration of the call (see :func:`_ambient_guard`).  It is optional so that a
    caller added on another branch cannot fail to construct, and it is never absent so
    that such a caller cannot fail to be watched.

    *extra_env* is how criterion 11(c) gives the ledger lookup an input to vary with.  It
    must be applied in the **child's** environment and not this process's: on Windows the
    EP DLL caches its environment at load time, so a variable set after `onnxruntime` has
    imported the EP is read by nothing.  A test that set it in-process would observe no
    change and conclude the mechanism is insensitive to it — a false UNWIRED.

    *trace* defaults to the historical behaviour (armed on every non-injected arm) so that
    no caller on another branch changes meaning.  It exists because the tracer witness path
    is **shared across arms**: arming it unlinks the previous arm's file, and an arm that
    then dispatches nothing — every faulted-ledger arm, by construction — leaves no file
    behind.  The tracer check reads that path, so an arm that has no business exercising
    the tracer could delete the only evidence it ever ran and turn a passing lane into a
    false UNWIRED purely by running later.  Criterion 11(c)'s arms pass ``trace=False``.
    """
    if guard is None:
        with _ambient_guard(f"census counters child ({tag})") as own:
            return _run_counters_child(
                inject=inject, tag=tag, guard=own, extra_env=extra_env, trace=trace
            )
    out_dir = _REPO_ROOT / "bench" / "results" / "census"
    out_dir.mkdir(parents=True, exist_ok=True)
    selector = os.environ.get("ONNXRUNTIME_EP_VULKAN_DEVICE", "unset")
    counters = out_dir / f"census-counters-dev{selector}-{tag}.json"
    counters.unlink(missing_ok=True)

    env = dict(os.environ)
    env["ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE"] = str(counters)
    if trace is None:
        trace = not inject
    if trace and not inject:
        # Arm the tracer on the clean polarity only: the injected run's teardown path is
        # exactly the one Tank's ruling says may not be reached.
        _CENSUS_TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CENSUS_TRACE_PATH.unlink(missing_ok=True)
        env[_ENV_TRACE] = str(_CENSUS_TRACE_PATH)
    else:
        env.pop(_ENV_TRACE, None)
    env.pop(_ENV_INJECT, None)
    if inject:
        env[_ENV_INJECT] = "1"
    # Never inherited: an arm that forgot to name its model would silently run the
    # previous arm's, and two arms that ran the same input cannot differ.
    env.pop(_ENV_CENSUS_MODEL, None)
    env.pop(_ENV_LEDGER_FILE, None)
    for k, v in (extra_env or {}).items():
        env[k] = v

    result = _watchdog.guarded_run(
        [
            sys.executable,
            # The child's longest silent window is its own module imports — numpy,
            # onnxruntime and onnx_ir all load before the first line of
            # `_counters_child_main` can run, and under load that window was measured at
            # 11221 work units of pure silence.  `-X importtime` emits one line per module
            # actually finished importing, which is forward progress tied to real work: a
            # child crawling through its imports keeps beating, a child wedged inside one
            # stops.  It is not a timer, and that distinction is the whole design.
            "-X", "importtime",
            str(Path(__file__).resolve()), _CHILD_FLAG, str(counters),
        ],
        guard=guard,
        what=f"census counters child ({tag})",
        label=f"counters_child_{tag}",
        budget_units=_budget_for(f"counters_child_{tag}"),
        # This child exists to exercise the EP.  If it goes silent, the mechanisms it
        # publishes have not been observed to run, which is criterion 12's subject.
        kind=KIND_MECHANISM,
        env=env,
        cwd=str(HERE),
    )
    log = (result.stdout or "") + "\n" + (result.stderr or "")
    if not counters.is_file():
        raise _verdict.InstrumentError(
            f"[wiring census instrument failure] ERROR(instrument): the {tag} counters "
            f"child exited {result.returncode} and wrote no counters file, so the census "
            "reached no observation of the JSON-published mechanisms.  This is not a "
            "finding about any of them.\n"
            f"child log (last 2000 chars):\n{log[-2000:]}"
        )
    try:
        doc = json.loads(counters.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise _verdict.InstrumentError(
            f"[wiring census instrument failure] ERROR(instrument): the {tag} counters "
            f"file is unreadable: {exc}"
        ) from exc
    return doc, log


def _ort_sink_warn_lines(log: str) -> list[str]:
    """Count broken-commitment WARNs that came out of ORT's own sink, using Tank's screen.

    ORT's sink writes UTF-16LE on this host, so a text-mode pipe hands back a string with
    an interleaved NUL after every character.  A naive substring test on that reads zero
    and would report Tank's WARN as never delivered — the same shape of defect as Link's
    ICD probe: an always-false screen and an always-true screen are equally blind.  The
    NULs are stripped first, and then his own ``ORT_DECORATION`` / ``WARN_MARKER`` do the
    deciding, so there is one definition of "a WARN through ORT's sink" in the repo.
    """
    scrubbed = log.replace("\x00", "")
    tools = _REPO_ROOT / "rust" / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    try:
        import probe_broken_commitment as bc  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return []
    return bc.ort_sink_warns(scrubbed)


def _observe_device_state_guard() -> str:
    """Run Link's obligation-8 guard over two inputs and report whether the pair differs.

    Imported from ``ci/check_device_state.py`` — one mechanism, not two (§10.0.1 R10
    sub-rule).  Returns the census line.

    2026-08-04 — WHY THE PLANTED ARM IS NO LONGER A SINGLE EXPECTED TOKEN
    --------------------------------------------------------------------
    The first version of this observation planted a companionless duration and required
    ``FAIL(condition=STEADY_UNCERTIFIED)``.  That is the guard's answer *on a host with a
    device-state producer*.  On a host with none — every GPU-free CI runner this project
    has — the same input correctly yields
    ``ERROR(instrument=device_state_producer_absent)``, because §10.0 obligation 8
    amendment 2 says the absence of telemetry is an instrument error and never a waiver.
    The census then reported ``INSTRUMENT-ERROR`` for the mechanism and went red on every
    host-free lane with no path to green: it was reading the *host*, not the guard.

    Reproduced on a GPU box on 2026-08-04 by suppressing the producers with
    ``ONNXRUNTIME_EP_CI_DEVICE_STATE_PRODUCERS=none``: the identical input flips from
    ``FAIL(condition=STEADY_UNCERTIFIED)`` (exit 1) to
    ``ERROR(instrument=device_state_producer_absent)`` (exit 4) with no file changed.

    The repair is not "accept either token" — that would launder a genuinely broken guard
    into a pass.  The census reads the host class from ``ci/device_state.py``'s own
    ``host_producer_status()`` and **predicts** which token the planted arm must produce;
    a token that does not match the host class is a failure of the guard, still.  And a
    second planted arm carries the discrimination that a single arm cannot: a claim with
    **no duration at all** must return ``PASS`` on every host.  Two inputs, two different
    tokens, on any host — R10's falsifier, with the environment no longer supplying the
    polarity.
    """
    import contextlib  # noqa: PLC0415
    import io  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    ci_dir = _REPO_ROOT / "ci"
    guard_path = ci_dir / "check_device_state.py"
    if not guard_path.is_file():
        return "UNWIRED (ci/check_device_state.py is absent)"

    if str(ci_dir) not in sys.path:
        sys.path.insert(0, str(ci_dir))
    try:
        import check_device_state as guard  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return f"INSTRUMENT-ERROR (ci/check_device_state.py did not import: {exc})"

    def _run(scan: Path) -> tuple[int, str]:
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                code = guard.main(["--scan", str(scan)])
        except SystemExit as exc:  # argparse escape hatch
            code = int(exc.code or 0)
        except Exception as exc:  # noqa: BLE001
            raise _verdict.InstrumentError(
                f"[wiring census instrument failure] ERROR(instrument): Link's "
                f"device-state guard raised on {scan}: {type(exc).__name__}: {exc}"
            ) from exc
        out = buf.getvalue()
        line = next(
            (ln for ln in out.splitlines() if ln.startswith(f"{guard.LABEL}:")), ""
        )
        return code, line

    with tempfile.TemporaryDirectory(prefix="census_device_state_") as td:
        planted = Path(td) / "planted_lane_claim.json"
        # Link's own plant, reproduced: a duration quoted as a lane claim with no
        # `device_state` companion.  If the guard passes this, it is not reading its input.
        planted.write_text(
            json.dumps({"claim": "gpu steady state", "gpu_busy_ms": 11.525}, indent=2),
            encoding="utf-8",
        )
        planted_code, planted_line = _run(Path(td))

    with tempfile.TemporaryDirectory(prefix="census_device_state_nofig_") as td:
        # The second arm.  Same shape of claim, same guard, one field removed: no
        # duration figure at all.  Obligation 8 has nothing to require of it, so the
        # guard must return PASS — on a telemetry host and on a host-free runner alike.
        # This is what carries the discrimination when the first arm's token is decided
        # by the host rather than by the input.
        (Path(td) / "no_figure_lane_claim.json").write_text(
            json.dumps({"claim": "gpu steady state", "nodes": 4}, indent=2),
            encoding="utf-8",
        )
        nofigure_code, nofigure_line = _run(Path(td))

    ours = _REPO_ROOT / "bench" / "results" / "census"
    if not ours.is_dir():
        return (
            "UNWIRED (no bench/results/census directory for the guard to read on this "
            "run — the guard had no subject, which is not a clean lane)"
        )
    ours_code, ours_line = _run(ours)

    # Which token the planted arm MUST produce is decided by the host class, read from
    # the guard's own module.  Not "either token is fine": a producer-absent answer on a
    # host that has a producer, or a FAIL on a host that has none, is the guard
    # disagreeing with the machine it is running on, and that is a finding.
    try:
        ds = guard.ds  # the same module object the guard itself rules with
        producers = list(ds.host_producer_status().get("available") or [])
    except Exception as exc:  # noqa: BLE001
        return f"INSTRUMENT-ERROR (ci/device_state.py host status unreadable: {exc})"
    host_class = "producer_present" if producers else "producer_absent"
    expected_code = (
        guard.EXIT_FAIL_CONDITION if producers else guard.EXIT_ERROR_INSTRUMENT
    )
    expected_token = (
        "FAIL(condition=STEADY_UNCERTIFIED)"
        if producers
        else "ERROR(instrument=device_state_producer_absent)"
    )

    if planted_code != expected_code or expected_token not in planted_line:
        return (
            f"UNWIRED (the planted companionless duration returned exit {planted_code} "
            f"[{planted_line!r}] on a {host_class} host, where obligation 8 requires "
            f"exit {expected_code} [{expected_token}] — the negative control did not "
            "fire as this host obliges it to, so this guard's green readings are "
            f"unverified.  host producers: {producers or 'none'})"
        )
    if nofigure_code != guard.EXIT_PASS:
        return (
            f"UNWIRED (a lane claim carrying NO duration returned exit {nofigure_code} "
            f"[{nofigure_line!r}] instead of PASS — the guard is answering something "
            "other than 'was a figure published without its companion', so its "
            "not-PASS answers do not mean what they say)"
        )
    if planted_line == ours_line:
        return (
            f"UNWIRED (constant: the guard returned the same line {ours_line!r} for a "
            "planted violation and for this run's own evidence)"
        )
    return (
        f"FIRED host={host_class} (producers: {','.join(producers) or 'none'}) "
        f"planted_exit={planted_code} planted={planted_line!r} | "
        f"no_figure_exit={nofigure_code} no_figure={nofigure_line!r} | "
        f"lane_exit={ours_code} lane={ours_line!r} "
        "(the planted arm's token is decided by the host class and is checked against "
        "it; the no-figure arm is the host-independent half, so the pair discriminates "
        "on a runner with no telemetry as well as on one with a GPU)"
    )


def _observe_instrument_census() -> str:
    """Report Tank's source-level census verdict token and the counts it computed."""
    import contextlib  # noqa: PLC0415
    import io  # noqa: PLC0415

    tools = _REPO_ROOT / "rust" / "tools"
    if not (tools / "audit_instruments.py").is_file():
        return "UNWIRED (rust/tools/audit_instruments.py is absent)"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    try:
        import audit_instruments as audit  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return f"INSTRUMENT-ERROR (audit_instruments.py did not import: {exc})"

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            code = audit.main_guarded(["--check"])
    except Exception as exc:  # noqa: BLE001
        return f"INSTRUMENT-ERROR (main_guarded raised {type(exc).__name__}: {exc})"
    out = buf.getvalue()
    verdict = next(
        (ln for ln in out.splitlines() if "CENSUS VERDICT" in ln), "<no verdict line>"
    )
    if code == audit.EXIT_ERROR_INSTRUMENT:
        return f"INSTRUMENT-ERROR ({verdict})"
    counts = [
        ln.strip() for ln in out.splitlines() if ln.strip().startswith("OK: ")
    ] or [
        ln.strip() for ln in out.splitlines()
        if ln.strip().startswith(("DRIFT", "MISSING", "NEW "))
    ]
    return f"exit={code} {verdict.strip()} | " + "; ".join(counts[:6])


# ---------------------------------------------------------------------------
# Criterion 12, round 29 — the twelve uncensused surfaces
# ---------------------------------------------------------------------------

#: Frame-disclosure / discrimination tokens.  Deliberately distinct from the census's
#: mechanism tokens: `IN-FORCE` says which world the run happened in, `MOVED` says an
#: artifact responded to the switch, and `UNOBSERVABLE` says the discriminator could not
#: exist in this frame.  None of the three is `UNWIRED`, and `UNOBSERVABLE` is never `0`.
_FLAG_MOVED = "MOVED"
_FLAG_CONSTANT = "CONSTANT"
_FLAG_UNOBSERVABLE = "UNOBSERVABLE"


def _in_force(name: str, env: "dict[str, str] | None" = None) -> str:
    """What this switch was set to in the frame named by *env* (default: this process).

    `unset` and `''` are different: a switch set to the empty string is in the environment
    and several of the readers below treat presence, not value, as arming.  Collapsing
    them would make the disclosure line lie about exactly the case that is hardest to
    reproduce from a log.
    """
    src = os.environ if env is None else env
    return "unset" if name not in src else repr(src[name])


def _flag_segment(name: str, state: str, detail: str, *, env=None) -> str:
    """One switch's line: what it was set to, and what moved when it was.

    The switch is spelled in full, never abbreviated.  Link's extent screen asks whether
    the observation NAMES the surface, and it asks that against the identifier as
    production Rust spells it; a census that prints `GEMV_PACKED` has not named
    `ONNXRUNTIME_EP_VULKAN_GEMV_PACKED`, and a reader grepping the artifact for the
    variable they are about to set would not find it either.
    """
    return f"{name}[census_frame={_in_force(name, env)}] {state}({detail})"


def _pair_state(clean, armed) -> str:
    return _FLAG_MOVED if clean != armed else _FLAG_CONSTANT


def _observe_flag_frame(
    clean_doc: dict,
    clean_log: str,
    a_doc: "dict | None",
    a_log: str,
    b_doc: "dict | None",
    b_log: str,
    a_env: dict,
    b_env: dict,
) -> str:
    """The nine instrumented-but-uncensused environment switches, one segment each.

    Every segment carries TWO facts and they are not the same strength (see the block
    comment at `_ENV_DEVICE_MEMORY`): `census_frame=` is frame disclosure and cannot fail;
    the state token is discrimination and can.  A reader who takes the first for the
    second has read a presence check as a wiring check, which is the error criterion 12
    exists to make visible.
    """
    if a_doc is None or b_doc is None:
        # Frame disclosure survives an arm outage — it reads this process's environment —
        # so the census still says which world it observed in, and says plainly that the
        # discriminators were not reached.  R13: an arm that did not run is not a finding
        # that the switches do nothing.
        return (
            "PARTIAL — discrimination arms did not run; frame disclosure only: "
            + " | ".join(
                _flag_segment(
                    n,
                    _FLAG_UNOBSERVABLE,
                    "the discrimination arm did not run, so this run says nothing about "
                    "whether the switch moves anything (R13: not a detection)",
                )
                for n in (
                    _ENV_GEMV_PACKED, _ENV_DEVICE_MEMORY, _ENV_QUARANTINE_SPANS,
                    _ENV_VA_RESERVE_MIB, _ENV_BACKEND_PROBE, _ENV_CLAIM_LOG,
                    _ENV_CLAIM_DEBUG, _ENV_DUMP_OUTPUT_BYTES, _ENV_VERBOSE,
                )
            )
        )

    segments: list[str] = []

    # ── GEMV_PACKED — a different KERNEL, and not reachable from this graph ──────────
    #
    # The strongest of the twelve and the one that stays UNOBSERVABLE, which is the point.
    # `ops/quant.rs::gemv_packed()` selects the packed batch-1 GEMV kernel, entering the
    # program as specialization constant 5 of `q_gemv.comp`. Corrected 2026-08-07 (issue
    # #58): the recording half of this gap is NOT open — `counters::record_pipeline_variant`
    # already records the whole resolved specialisation vector for every pipeline this
    # process builds, and `gemv_packed_spec_constant()` reads index 5 of any recorded
    # `q_gemv`-stemmed entry, reporting `UNOBSERVABLE` (never a silent `0`) when no such
    # pipeline was built. The census graph is a six-node Add/Mul elementwise chain with no
    # MatMulNBits, so no GEMV kernel is reachable here — that is the one surviving blocker,
    # and it is READ FROM THE ARTIFACT below, not asserted in prose: a future census lane
    # that reaches a q_gemv pipeline (or a MatMulNBits-anchored subgraph) would move this
    # value away from `UNOBSERVABLE`, and this segment would report it rather than paper
    # over it, because `_FLAG_MOVED` fires whenever the two arms disagree, and neither arm
    # is hardcoded to `UNOBSERVABLE`.
    # `_GEMV_ABSENT` is a sentinel private to this segment, never a string a real
    # counters document could contain — using `"MISSING-FIELD"` here (issue #61) made a
    # missing field indistinguishable from a present field holding that literal text, and
    # two arms both missing the field compared equal and fell through to `_FLAG_CONSTANT`:
    # an instrument outage (the field was renamed or dropped) read as "armed, and nothing
    # moved", the one thing this segment's whole discipline exists to refuse. A missing
    # field is reported `UNOBSERVABLE` with an explicit missing-field detail instead, and
    # ONLY the field literally being absent takes this path — a present field that reads
    # the string `'UNOBSERVABLE'` still falls through to the real-UNOBSERVABLE branch below.
    _GEMV_ABSENT = object()
    _gemv_a = a_doc.get("gemv_packed_spec_constant", _GEMV_ABSENT)
    _gemv_b = b_doc.get("gemv_packed_spec_constant", _GEMV_ABSENT)
    _gemv_missing_arms = [
        label
        for label, val in (("arm A", _gemv_a), ("arm B", _gemv_b))
        if val is _GEMV_ABSENT
    ]
    if _gemv_missing_arms:
        _gemv_state = _FLAG_UNOBSERVABLE
        _gemv_detail = (
            "INSTRUMENT GAP: 'gemv_packed_spec_constant' is absent from the counters "
            f"document for {' and '.join(_gemv_missing_arms)} — a missing-field "
            "instrument outage, not a genuine 'no q_gemv pipeline built' reading and not "
            "a pipeline having been built. Reported as UNOBSERVABLE, never CONSTANT: this "
            "segment must not read a renamed or dropped counter field as 'armed, and "
            "nothing moved' (issue #61)."
        )
    elif _gemv_a == "UNOBSERVABLE" and _gemv_b == "UNOBSERVABLE":
        _gemv_state = _FLAG_UNOBSERVABLE
        _gemv_detail = (
            "armed in arm A; gemv_packed_spec_constant() reads 'UNOBSERVABLE' in both "
            "arms (verified against a_doc/b_doc, not claimed): the census graph builds "
            "no MatMulNBits/q_gemv pipeline, so counters::record_pipeline_variant has "
            "nothing to record here and no 'no difference' reading is being mistaken for "
            "a 0 (R12). Recording is not this gap; reachability is — a GEMV-reaching "
            "census lane, or a probe adopting rust/tools/probe_gemv_kernel_identity.py's "
            "evidence, would close it."
        )
    else:
        # Either arm built a q_gemv pipeline. That is the reachability blocker closing —
        # a real finding, not a regression in this test. Report it plainly rather than
        # forcing it back into the UNOBSERVABLE shape this segment has held until now.
        _gemv_state = _FLAG_MOVED if _gemv_a != _gemv_b else _FLAG_CONSTANT
        _gemv_detail = (
            f"UNEXPECTED: gemv_packed_spec_constant()={_gemv_a!r} (arm A) vs "
            f"{_gemv_b!r} (arm B) — a q_gemv pipeline WAS built in this frame. The "
            "reachability premise ci/census_surface_map.json's GEMV_PACKED entry and "
            "this segment's comment rely on has changed; re-derive both, do not "
            "re-assert the old UNOBSERVABLE reading."
        )
    segments.append(_flag_segment(_ENV_GEMV_PACKED, _gemv_state, _gemv_detail, env=a_env))

    # ── DEVICE_MEMORY — a different ALLOCATION PATH, and this one IS reachable ───────
    #
    # `factory.rs` switches allocations to device-local memory.  Tank's allocator counters
    # publish the resulting frame, so the pair of arms is a real R10 falsifier rather than
    # a disclosure: `alloc_device_frame` is a token the production path computes, and it
    # is typed ('OFF'/'ON'/...), which an increment cannot forge.
    dm_clean = tuple(
        clean_doc.get(k) for k in
        ("alloc_device_frame", "alloc_device_frames_declared", "alloc_device_backed_spans")
    )
    dm_armed = tuple(
        a_doc.get(k) for k in
        ("alloc_device_frame", "alloc_device_frames_declared", "alloc_device_backed_spans")
    )
    if dm_clean == (None, None, None):
        segments.append(
            _flag_segment(
                _ENV_DEVICE_MEMORY, _FLAG_UNOBSERVABLE,
                "the counters artifact carries no alloc_device_frame fields, so this "
                "frame cannot see the allocation path at all",
                env=a_env,
            )
        )
    else:
        segments.append(
            _flag_segment(
                _ENV_DEVICE_MEMORY, _pair_state(dm_clean, dm_armed),
                f"alloc_device_frame/frames_declared/device_backed_spans "
                f"unset={dm_clean!r} armed={dm_armed!r} "
                f"(device={a_doc.get('alloc_device_frame_device')!r})",
                env=a_env,
            )
        )

    # ── QUARANTINE_SPANS — allocator.rs, observable through Tank's counters ─────────
    q_clean = tuple(
        clean_doc.get(k) for k in
        ("alloc_quarantine_peak_spans", "alloc_quarantine_retired")
    )
    q_armed = tuple(
        a_doc.get(k) for k in
        ("alloc_quarantine_peak_spans", "alloc_quarantine_retired")
    )
    if q_clean == (None, None):
        segments.append(
            _flag_segment(
                _ENV_QUARANTINE_SPANS, _FLAG_UNOBSERVABLE,
                "no alloc_quarantine_* fields in the counters artifact",
                env=a_env,
            )
        )
    elif q_clean == q_armed == (0, 0):
        # An honest CONSTANT would be a lie here: with no allocator traffic at all the
        # quarantine has nothing to hold either way, so the pair is uninformative rather
        # than negative.
        segments.append(
            _flag_segment(
                _ENV_QUARANTINE_SPANS, _FLAG_UNOBSERVABLE,
                "both arms recorded alloc_quarantine_peak_spans=0 alloc_quarantine_"
                "retired=0 with alloc_allocations="
                f"{clean_doc.get('alloc_allocations')!r}: no span was freed in either "
                "arm, so the quarantine had nothing to hold and the pair says nothing "
                "about the switch (R12 — not a 0, an event that did not occur)",
                env=a_env,
            )
        )
    else:
        segments.append(
            _flag_segment(
                _ENV_QUARANTINE_SPANS, _pair_state(q_clean, q_armed),
                f"alloc_quarantine_peak_spans/retired unset={q_clean!r} armed={q_armed!r}",
                env=a_env,
            )
        )

    # ── VA_RESERVE_MIB — armed, and nothing publishes the reservation ──────────────
    #
    # The honest answer, and the one §10 asks for when a surface cannot be censused: say
    # so, name what would make it observable, and name the owner.  A surface deliberately
    # left out of scope with a reason is a fine answer; a surface nobody looked at is not.
    segments.append(
        _flag_segment(
            _ENV_VA_RESERVE_MIB, _FLAG_UNOBSERVABLE,
            f"armed at {a_env.get(_ENV_VA_RESERVE_MIB)!r} in arm A and no counter "
            "publishes the reservation size, so no artifact this run produced could "
            "differ. Requested from Tank: an `alloc_va_reserved_bytes` field, after which "
            "this segment becomes a two-arm discriminator like DEVICE_MEMORY's",
            env=a_env,
        )
    )

    # ── BACKEND_PROBE — writes a FILE naming the selected barrier backend ──────────
    #
    # `vk/barrier.rs` writes "sync2" or "legacy" to the named path.  This is the strongest
    # kind of observation available to a census: an artifact the mechanism itself produced,
    # whose CONTENT names the thing it selected.
    if _CENSUS_BACKEND_PROBE_PATH.is_file():
        token = _CENSUS_BACKEND_PROBE_PATH.read_text(encoding="utf-8", errors="replace").strip()
        segments.append(
            _flag_segment(
                _ENV_BACKEND_PROBE,
                _FLAG_MOVED if token else _FLAG_CONSTANT,
                f"the probe file the EP wrote names backend={token!r} "
                "(an artifact the mechanism produced, whose content names its selection)",
                env=a_env,
            )
        )
    else:
        segments.append(
            _flag_segment(
                _ENV_BACKEND_PROBE, _FLAG_CONSTANT,
                f"armed at {a_env.get(_ENV_BACKEND_PROBE)!r} and no probe file was "
                "written — the backend selection did not disclose itself in this run",
                env=a_env,
            )
        )

    # ── CLAIM_LOG — writes JSON Lines, one object per claim decision ───────────────
    if _CENSUS_CLAIM_LOG_PATH.is_file():
        lines = [
            ln for ln in
            _CENSUS_CLAIM_LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
            if ln.strip()
        ]
        ops = sorted({
            (json.loads(ln).get("op") if ln.lstrip().startswith("{") else "?")
            for ln in lines[:200]
        }) if lines else []
        segments.append(
            _flag_segment(
                _ENV_CLAIM_LOG,
                _FLAG_MOVED if lines else _FLAG_CONSTANT,
                f"{len(lines)} claim-decision record(s), ops={ops} "
                "(the count is of DECISIONS, not of graph nodes and not of claimed nodes)",
                env=a_env,
            )
        )
    else:
        segments.append(
            _flag_segment(
                _ENV_CLAIM_LOG, _FLAG_CONSTANT,
                f"armed at {a_env.get(_ENV_CLAIM_LOG)!r} and no claim log was written",
                env=a_env,
            )
        )

    # ── CLAIM_DEBUG and DUMP_OUTPUT_BYTES — both publish through the LOG, and both
    #    are armed in the same arm, so each needs a MARKER of its own ───────────────
    #
    # They are separable inside one arm because their markers are distinct: the claim
    # debug path prints per-node claim/decline text, the dump path prints
    # `dispatch kernel[...] shader=...`.  Sharing an arm without distinct markers would be
    # a confound, and a confounded discriminator that reads MOVED is worth nothing.
    a_scrub = a_log.replace("\x00", "")
    clean_scrub = clean_log.replace("\x00", "")
    dump_armed = a_scrub.count("dispatch kernel[")
    dump_clean = clean_scrub.count("dispatch kernel[")
    if dump_armed == dump_clean == 0:
        segments.append(
            _flag_segment(
                _ENV_DUMP_OUTPUT_BYTES, _FLAG_UNOBSERVABLE,
                f"armed in arm A at child log severity "
                f"{a_env.get(_ENV_CHILD_LOG_SEVERITY)!r} and its marker "
                "('dispatch kernel[') appears in neither arm's log. Its only observable "
                "is a log::debug! record; if ORT's severity filter or its sink dropped it "
                "the census did not look, and 'did not look' is not 'found nothing' (R12)",
                env=a_env,
            )
        )
    else:
        segments.append(
            _flag_segment(
                _ENV_DUMP_OUTPUT_BYTES, _pair_state(dump_clean, dump_armed),
                f"'dispatch kernel[' marker lines unset={dump_clean} armed={dump_armed} "
                "— each names the shader the dispatch used, so this is the one line in "
                "the census that observes kernel IDENTITY rather than kernel count",
                env=a_env,
            )
        )

    claim_markers = ("unclaimed ", "GetCapability: declining", "[not-registered]")
    cd_armed = sum(a_scrub.count(mk) for mk in claim_markers)
    cd_clean = sum(clean_scrub.count(mk) for mk in claim_markers)
    if cd_armed == cd_clean == 0:
        segments.append(
            _flag_segment(
                _ENV_CLAIM_DEBUG, _FLAG_UNOBSERVABLE,
                f"armed in arm A at child log severity "
                f"{a_env.get(_ENV_CHILD_LOG_SEVERITY)!r} and its only output is a "
                "DECLINE report (`ep.rs:605`, \"unclaimed {op} x{n} ({why})\"). The census "
                "graph is a six-node Add/Mul chain that the EP claims in full — the "
                f"claim log for this arm recorded every decision as claimed — so the "
                "switch has nothing to report in this frame. That is an event that "
                "cannot occur here, not a switch that does nothing (R12). Making it "
                "discriminable needs a census graph carrying at least one declined op",
                env=a_env,
            )
        )
    else:
        segments.append(
            _flag_segment(
                _ENV_CLAIM_DEBUG, _pair_state(cd_clean, cd_armed),
                f"claim-decision marker lines unset={cd_clean} armed={cd_armed}",
                env=a_env,
            )
        )

    # ── VERBOSE — its own arm, because a log level shared with CLAIM_DEBUG in one arm
    #    would make either one's reading unattributable ──────────────────────────────
    b_scrub = b_log.replace("\x00", "")
    v_armed = len(b_scrub.splitlines())
    v_clean = len(clean_scrub.splitlines())
    segments.append(
        _flag_segment(
            _ENV_VERBOSE, _pair_state(v_clean, v_armed),
            f"child log lines unset={v_clean} armed={v_armed} at child log severity "
            f"{b_env.get(_ENV_CHILD_LOG_SEVERITY)!r} (arm B arms VERBOSE and nothing "
            "else, so a difference here is attributable to it alone)",
            env=b_env,
        )
    )

    moved = sum(1 for s in segments if f"] {_FLAG_MOVED}(" in s)
    unobs = sum(1 for s in segments if f"] {_FLAG_UNOBSERVABLE}(" in s)
    header = (
        f"{len(segments)} switch(es) disclosed; {moved} discriminated by an artifact this "
        f"run produced, {unobs} UNOBSERVABLE in this frame with a reason. "
        "Disclosure is not discrimination: the first says which world the census observed "
        "in, only the second can fail."
    )
    return header + " || " + " | ".join(segments)


# ---------------------------------------------------------------------------
# Issue #61: `_observe_flag_frame`'s GEMV_PACKED segment must report UNOBSERVABLE, with
# an explicit missing-field detail, when `gemv_packed_spec_constant` is absent from
# either arm's counters document — never CONSTANT, and never silently. No GPU is
# required: `_observe_flag_frame` is a pure function of the docs/logs/envs handed to it,
# and every other segment settles harmlessly into its own UNOBSERVABLE/CONSTANT default
# when its own fields are absent, so a minimal doc exercises the GEMV segment alone.
# ---------------------------------------------------------------------------


def _flag_frame_docs(
    gemv_a=None, gemv_b=None, *, gemv_a_present=True, gemv_b_present=True
) -> tuple[dict, dict, dict]:
    """The minimal (clean_doc, a_doc, b_doc) triple that reaches the GEMV_PACKED
    segment without any other segment's own fields present, so those segments' own
    UNOBSERVABLE/CONSTANT defaults cannot be mistaken for the segment under test."""
    clean_doc: dict = {}
    a_doc: dict = {}
    b_doc: dict = {}
    if gemv_a_present:
        a_doc["gemv_packed_spec_constant"] = gemv_a
    if gemv_b_present:
        b_doc["gemv_packed_spec_constant"] = gemv_b
    return clean_doc, a_doc, b_doc


def _gemv_segment_text(result: str) -> str:
    """Pull just the GEMV_PACKED segment's text out of `_observe_flag_frame`'s report.

    The GEMV segment is always appended first, immediately followed by DEVICE_MEMORY's,
    so slicing between the two env names is exact regardless of the joiner text.
    """
    start = result.index(_ENV_GEMV_PACKED + "[")
    end = result.index(_ENV_DEVICE_MEMORY + "[", start)
    return result[start:end]


def _run_flag_frame_gemv(gemv_a=None, gemv_b=None, **presence) -> str:
    clean_doc, a_doc, b_doc = _flag_frame_docs(gemv_a, gemv_b, **presence)
    result = _observe_flag_frame(
        clean_doc, "", a_doc, "", b_doc, "", {_ENV_GEMV_PACKED: "1"}, {}
    )
    return _gemv_segment_text(result)


@pytest.mark.parametrize(
    "gemv_a, gemv_b, expected_state",
    [
        pytest.param("UNOBSERVABLE", "UNOBSERVABLE", _FLAG_UNOBSERVABLE, id="today-real-reading"),
        pytest.param("1", "UNOBSERVABLE", _FLAG_MOVED, id="moved-armed-vs-unobservable"),
        pytest.param("1", "1", _FLAG_CONSTANT, id="constant-both-armed"),
        pytest.param("0", "1", _FLAG_MOVED, id="moved-0-vs-1"),
    ],
)
def test_observe_flag_frame_gemv_preserves_the_four_live_states(gemv_a, gemv_b, expected_state):
    """The four live states from issue #61's reproduction — must survive the fix
    untouched: this bug's repair must not turn a real reading into UNOBSERVABLE."""
    seg = _run_flag_frame_gemv(gemv_a, gemv_b)
    assert f"] {expected_state}(" in seg, seg


def test_observe_flag_frame_gemv_missing_from_arm_a_is_unobservable_not_constant():
    seg = _run_flag_frame_gemv(gemv_b="UNOBSERVABLE", gemv_a_present=False)
    assert f"] {_FLAG_UNOBSERVABLE}(" in seg, seg
    assert "INSTRUMENT GAP" in seg, seg
    assert "arm A" in seg, seg
    assert f"] {_FLAG_CONSTANT}(" not in seg, seg


def test_observe_flag_frame_gemv_missing_from_arm_b_is_unobservable_not_constant():
    seg = _run_flag_frame_gemv(gemv_a="UNOBSERVABLE", gemv_b_present=False)
    assert f"] {_FLAG_UNOBSERVABLE}(" in seg, seg
    assert "INSTRUMENT GAP" in seg, seg
    assert "arm B" in seg, seg
    assert f"] {_FLAG_CONSTANT}(" not in seg, seg


def test_observe_flag_frame_gemv_missing_from_both_arms_is_unobservable_not_constant():
    """The exact regression in issue #61: two missing fields both defaulted to the same
    sentinel string, compared equal, and read CONSTANT -- 'armed, and nothing moved' --
    for an instrument outage, never a genuine reading."""
    seg = _run_flag_frame_gemv(gemv_a_present=False, gemv_b_present=False)
    assert f"] {_FLAG_UNOBSERVABLE}(" in seg, seg
    assert "INSTRUMENT GAP" in seg, seg
    assert "arm A and arm B" in seg, seg
    assert f"] {_FLAG_CONSTANT}(" not in seg, seg


def test_observe_flag_frame_gemv_malformed_value_is_a_real_reading_not_unobservable():
    """A present-but-garbage value is a genuine (if unexpected) artifact reading, not a
    missing-field instrument outage -- the fix must not swallow it into UNOBSERVABLE
    either, only an actually absent field takes that path."""
    seg = _run_flag_frame_gemv(gemv_a="GARBAGE", gemv_b="GARBAGE")
    assert f"] {_FLAG_CONSTANT}(" in seg, seg
    assert "UNEXPECTED" in seg, seg
    assert "INSTRUMENT GAP" not in seg, seg


def _observe_ep_entrypoints(clean_doc: dict, inject_doc: dict) -> str:
    """`compile_calls`, `compute_calls`, `subgraphs_stub` — three C ABI counters no
    census mechanism read.

    The state this buys that the census did not have: the census inferred "the EP ran"
    from `dispatches_executed`, so a `Compute()` that ENTERED and dispatched nothing was
    indistinguishable from a `Compute()` that never happened.  Those are opposite
    findings — the first is a broken kernel path, the second is a partitioner that claimed
    nothing — and one number was reporting both.
    """
    keys = ("compile_calls", "compute_calls", "subgraphs_stub")
    missing = [k for k in keys if k not in clean_doc]
    if missing:
        return (
            f"UNWIRED (the counters artifact carries no {', '.join(missing)} field(s) — "
            "the C ABI moved under the census)"
        )
    compile_calls = clean_doc["compile_calls"]
    compute_calls = clean_doc["compute_calls"]
    stubs = clean_doc["subgraphs_stub"]
    dispatches = clean_doc.get("dispatches_executed", 0)

    if compile_calls == 0:
        state = "NOT-COMPILED (ORT never called Compile — nothing was claimed)"
    elif compute_calls == 0:
        state = (
            "COMPILED-NOT-COMPUTED (Compile() entered and Compute() never did — the EP "
            "was asked to compile and the graph never ran)"
        )
    elif isinstance(dispatches, int) and dispatches == 0:
        state = (
            "ENTERED-NO-DISPATCH (Compute() entered and dispatched nothing — the state "
            "the census could not previously name, because it inferred execution from "
            "dispatches_executed alone)"
        )
    else:
        state = "COMPILED-AND-DISPATCHED"

    moved = (
        "MOVED" if (clean_doc.get("compute_calls"), clean_doc.get("compile_calls"))
        != (inject_doc.get("compute_calls"), inject_doc.get("compile_calls"))
        else "CONSTANT-ACROSS-POLARITIES"
    )
    return (
        f"{state} compile_calls={compile_calls!r} compute_calls={compute_calls!r} "
        f"subgraphs_stub={stubs!r} dispatches_executed={dispatches!r} "
        f"(counts are ENTRIES to the ORT entry points, not graph nodes and not islands; "
        f"planted-failure polarity: compile_calls={inject_doc.get('compile_calls')!r} "
        f"compute_calls={inject_doc.get('compute_calls')!r} "
        f"subgraphs_stub={inject_doc.get('subgraphs_stub')!r} — {moved})"
    )


def _gpu_tracer_observation(trace_path: "Path") -> str:
    """Build the `gpu_tracer` mechanism's NAME-AGAINST-CONTENT observation from one
    census arm's Chrome-trace JSON at *trace_path*.

    ISSUE #24 (2026-08-06): this used to end its string with
    ``armed by this census via {_ENV_TRACE}={trace_path.name}``.  The census's own three
    arms each write to `census-trace-dev{0,1,unset}.json` — a name derived from
    `ONNXRUNTIME_EP_VULKAN_DEVICE`, the very thing that DEFINES an arm — so that fragment
    differed across every arm BY CONSTRUCTION, regardless of what tracing found.
    Criterion 12's NAME-AGAINST-CONTENT check (`ci/check_census_completeness.py`)
    compares this string verbatim across arms to ask whether `gpu_tracer`'s claim would
    have read differently had it been wrong; with the per-arm path folded in, the answer
    was `VARIES` on every run ever taken, even though the three arms never toggle
    `ONNXRUNTIME_EP_VULKAN_TRACE_GPU` and emit the identical phase set — a permanent
    false positive that made it impossible to ever catch someone marking `gpu_tracer`
    `name_verified: true` on the strength of nothing (the exact defect
    `ci/negative_control_census_completeness.py::arm_name_claim_contradicted` plants and
    checks for, using this real function and the real map).

    The env var NAME below (`{_ENV_TRACE}`, no `=path`) is real content — it says the
    tracer was armed for this run at all.  The specific file it happened to write to is
    bookkeeping the census needed to avoid three arms clobbering one file, not a claim
    about tracing, and is intentionally left out of this string.
    """
    trace_data = json.loads(trace_path.read_text(encoding="utf-8"))
    events = (
        trace_data if isinstance(trace_data, list)
        else trace_data.get("traceEvents", []) if isinstance(trace_data, dict)
        else []
    )
    if not events:
        return f"UNWIRED (armed via {_ENV_TRACE} and the file it wrote contains no events)"

    phases = sorted({e.get("ph") for e in events if isinstance(e, dict)})
    names = sorted({e.get("name") for e in events if isinstance(e, dict)})
    # WHICH of Niobe's ten `Phase` variants did this trace carry?
    #
    # The line used to report `distinct_names=<n>` and no names at all, so its
    # extent against the tracer's own surface was 1/12: it named the switch that
    # armed it and not one of the phases it was supposed to have traced.  A count
    # of distinct names would certify `Phase::Record` — wired, invoked, correct,
    # input-varying, and wrong by 50x in what it was called — exactly as it stood.
    #
    # The absent variants are reported as NOT-EMITTED, not as 0, and the device
    # switch's in-force state is printed beside them so a reader can tell "this
    # phase did not occur" from "this phase could not occur in this frame" (R12).
    blob = json.dumps(events)[:2_000_000]
    emitted = [
        p for p in _TRACE_PHASES
        if _phase_event_name(p) in names or f'"{_phase_event_name(p)}"' in blob
    ]
    absent = [p for p in _TRACE_PHASES if p not in emitted]
    return (
        f"{len(events)} trace event(s), ph_types={phases}, "
        f"distinct_names={len(names)}; "
        f"Phase variants emitted ({len(emitted)}/{len(_TRACE_PHASES)}): "
        f"{emitted or 'none'}; NOT-EMITTED: {absent or 'none'} "
        f"(matched as {[_phase_event_name(p) for p in _TRACE_PHASES[:2]]}… — "
        "the trace spells the variants snake_case under a `vulkan.` prefix, "
        "per trace.rs::Phase::as_str; a matcher that looked for the Rust "
        "identifier reported 0/10 and was wrong, which is why R13 says a "
        "result confirming a prediction earns more scrutiny than one "
        "contradicting it); "
        f"(armed by this census via {_ENV_TRACE}; "
        f"{_ENV_TRACE_GPU}={_in_force(_ENV_TRACE_GPU)} — the census does not "
        "arm device spans, because a device-clock reading under four-agent "
        "contention is Switch's exclusive claim and Niobe's admissibility "
        "frame, so any device-side phase above is UNOBSERVABLE in this frame "
        "rather than absent. No duration from this file is quoted — §10.0 "
        "obligation 8. A variant present in Rust and missing from this list "
        "is a name this census has never seen emitted, which is what "
        "Phase::Record was for fifty runs. The per-arm trace FILE this ran "
        "wrote to is deliberately not named above — see issue #24)"
    )


def test_gpu_tracer_observation_does_not_vary_with_the_arms_own_device_selector(
    tmp_path,
) -> None:
    """CONVERSE/POSITIVE control for issue #24's fix.

    Builds three fixture trace files under the EXACT names the census's own three real
    arms write to — `census-trace-dev0.json`, `census-trace-dev1.json`,
    `census-trace-devunset.json` (`ONNXRUNTIME_EP_VULKAN_DEVICE`'s three values) — with
    byte-identical trace content in each, and asserts `_gpu_tracer_observation` reads
    identically across all three. This does not depend on a live Vulkan run having
    already populated `bench/results/census/`: it reproduces the exact defect (an
    arm-identifying file name, not a content difference) deterministically, so the
    control does not silently pass or skip depending on what ambient scratch happens to
    exist on the machine running it.
    """
    same_events = json.dumps(
        [
            {"ph": "X", "name": "vulkan.record"},
            {"ph": "X", "name": "vulkan.submit"},
        ]
    )
    names = ("census-trace-dev0.json", "census-trace-dev1.json", "census-trace-devunset.json")
    observed = {}
    for name in names:
        path = tmp_path / name
        path.write_text(same_events, encoding="utf-8")
        observed[name] = _gpu_tracer_observation(path)

    distinct = set(observed.values())
    assert len(distinct) == 1, (
        "three trace files differing ONLY in the arm-identifying name the census's own "
        "device selector produces, with byte-identical events, must read identically -- "
        "if this fails, the per-arm file name has leaked back into the compared string:\n"
        + "\n".join(f"  {name}: {text}" for name, text in observed.items())
    )

    real_traces = sorted(_CENSUS_OUT.glob("census-trace-dev*.json"))
    if len(real_traces) >= 2:
        # Best-effort extra evidence when a prior live Vulkan run has left the real
        # per-arm trace files on this machine: the same invariance must hold on them too.
        real_observed = {p.name: _gpu_tracer_observation(p) for p in real_traces}
        assert len(set(real_observed.values())) == 1, (
            "the real census arms on this machine differ only by "
            "ONNXRUNTIME_EP_VULKAN_DEVICE and none of them toggle "
            "ONNXRUNTIME_EP_VULKAN_TRACE_GPU or emit a different phase set, so "
            "gpu_tracer's observation must read identically across all of them:\n"
            + "\n".join(f"  {name}: {text}" for name, text in real_observed.items())
        )


def test_gpu_tracer_observation_still_varies_when_trace_content_genuinely_differs(
    tmp_path,
) -> None:
    """PLANTED-RED control: the fix above must not collapse this into a constant.

    Two synthetic trace files differing in which Phase variants they actually emitted
    must still produce different `_gpu_tracer_observation` text — proving the function
    still reports genuine content variation, not just "whatever the fix hides".
    """
    sparse = tmp_path / "census-trace-sparse.json"
    sparse.write_text(
        json.dumps([{"ph": "X", "name": "vulkan.record"}]), encoding="utf-8"
    )
    richer = tmp_path / "census-trace-richer.json"
    richer.write_text(
        json.dumps(
            [
                {"ph": "X", "name": "vulkan.record"},
                {"ph": "X", "name": "vulkan.submit"},
                {"ph": "X", "name": "vulkan.fence_wait"},
            ]
        ),
        encoding="utf-8",
    )
    a = _gpu_tracer_observation(sparse)
    b = _gpu_tracer_observation(richer)
    assert a != b, (
        "a trace emitting one Phase variant and a trace emitting three must not read "
        f"identically:\n  sparse: {a}\n  richer: {b}"
    )
    assert "Phase variants emitted (1/" in a, a
    assert "Phase variants emitted (3/" in b, b


def _print_census(observations: dict, guard) -> "Path":
    """Emit the census lines and write the artifact.  Called on every exit path.

    A run that stalled at mechanism 3 has still observed mechanisms 1 and 2; discarding
    those because a later step hung would throw away evidence the run actually produced.
    """
    print("\n[WIRING CENSUS] M0 criterion 12 — per-mechanism observations:", file=sys.stderr)
    for mech in _MECHANISMS:
        obs = observations.get(mech, "NOT-REACHED (an earlier step ended the census)")
        print(f"[WIRING CENSUS] {mech}: {obs}", file=sys.stderr)
    print("[WIRING CENSUS] end.", file=sys.stderr)

    _CENSUS_OUT.mkdir(parents=True, exist_ok=True)
    census_path = _CENSUS_OUT.parent / (
        f"wiring_census-dev{os.environ.get('ONNXRUNTIME_EP_VULKAN_DEVICE', 'unset')}.json"
    )
    census_path.write_text(
        json.dumps(
            {
                "criterion": "12",
                "device_selector": os.environ.get(
                    "ONNXRUNTIME_EP_VULKAN_DEVICE", "unset"
                ),
                "vocabulary": (
                    "Tank's six states, ordered by how late the failure is discoverable: "
                    "absent -> uninvoked -> unfalsified -> unreachable -> out-of-frame -> "
                    "misnamed; plus R13's three terminal states PASS / FAIL(condition) / "
                    "ERROR(instrument).  One census (rust/tools/audit_instruments.py), "
                    "not two.  A step that never returned reads STALLED, which is not "
                    "UNWIRED: 'ran and produced nothing' and 'never came back' are "
                    "different facts."
                ),
                "no_duration_quoted": (
                    "§10.0 obligation 8 — every line above is a count, a token or a "
                    "byte volume.  The tracer line reports how many events it wrote and "
                    "quotes none of their durations.  The stall budgets below are counts "
                    "of reference computations this machine completed during this run, "
                    "not seconds; that is what makes them contention-invariant."
                ),
                "stall_detector": {
                    "fault_injection": (
                        _watchdog.injected_stall_target() or "INACTIVE"
                    ),
                    "unit": (                        "one SHA-256 over a fixed 256 KiB block, completed by this "
                        "machine on a background thread during this run"
                    ),
                    "budget_units": dict(_BUDGET_UNITS),
                    "observed_units": dict(guard.costs),
                    "observed_max_silence_units": dict(guard.max_silence),
                    "clock_units_total": guard.clock.units,
                    "why_not_a_timeout": (
                        "A wall-clock threshold moves the same way for 'the box is "
                        "loaded' and 'the census hung', so no value of it is right "
                        "(R9 amendment 5).  A budget in work units does not: contention "
                        "lowers units-per-second, widening the window in wall time, while "
                        "a hang leaves the machine producing units and the step producing "
                        "nothing."
                    ),
                    "measures_cpu_contention_only": (
                        "The reference unit is CPU-bound.  GPU and disk contention slow a "
                        "step without slowing the clock; observed_units above is recorded "
                        "on every run so that margin is auditable."
                    ),
                },
                "observations": {mm: observations.get(mm, "UNWIRED") for mm in _MECHANISMS},
                "criterion_12": {
                    "closes_row": False,
                    "why_not": (
                        "Row 12's tally is Trinity's and the artifact is Trinity's, and "
                        "Morpheus's ruling on criterion 11 is that supplying the evidence "
                        "and closing the row must not be the same act. Link declined to "
                        "close it from ci/check_census_completeness.py for the same "
                        "reason; a census that closed its own row would be the identity "
                        "defect one level up."
                    ),
                    "the_answer_is_not_12_of_12": (
                        "The census's twelve was twelve of a denominator the census "
                        "supplied itself. The independent whole is "
                        "ci/check_census_completeness.py's, enumerated from production "
                        "Rust this file does not write: 50 surfaces, of which 12 were "
                        "instrumented and observed by no census mechanism. Those twelve "
                        "are what round 29 addresses."
                    ),
                    "what_moved": (
                        "Two things, and they are different: (a) the GAP COUNT — the "
                        "three uncensused C ABI counters are read by `ep_entrypoints` and "
                        "the nine uncensused env switches by `flag_frame`; (b) EXTENT — "
                        "gpu_tracer, retain_viable, net_benefit_gate, validation_"
                        "messenger, broken_commitment_warn, ledger_lookup and partitioner "
                        "now name their own surfaces in their own observation text, which "
                        "is what Link's extent numerator counts. Closing (a) while (b) "
                        "stayed at gpu_tracer 1/12 would have moved the gap count and not "
                        "the coverage — R11's shape."
                    ),
                    "disclosure_is_not_discrimination": (
                        "Every `flag_frame` segment carries `census_frame=`, which says "
                        "which world the census observed in and CANNOT FAIL, and a state "
                        "token, which says whether an artifact this run produced moved "
                        "with the switch and CAN. Reading the first as the second is "
                        "reading a presence check as a wiring check."
                    ),
                    "still_unobservable_with_a_reason": (
                        "ONNXRUNTIME_EP_VULKAN_GEMV_PACKED selects a different kernel and "
                        "the census graph is an elementwise chain with no GEMV node, so "
                        "the kernel it selects is not reachable in this frame; "
                        "ONNXRUNTIME_EP_VULKAN_VA_RESERVE_MIB is armed and no counter "
                        "publishes the reservation size. Both read UNOBSERVABLE with the "
                        "reason and the request, never CONSTANT and never 0 (R12)."
                    ),
                },
                "unwired": [
                    mm for mm in _MECHANISMS
                    if observations.get(mm, "UNWIRED").startswith("UNWIRED")
                ],
                "stalled": [
                    mm for mm in _MECHANISMS
                    if observations.get(mm, "").startswith(STALLED)
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[WIRING CENSUS] wrote {census_path}", file=sys.stderr)
    return census_path


# ---------------------------------------------------------------------------
# The census test
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB"),
    reason="ONNXRUNTIME_VULKAN_EP_LIB not set — no EP to census",
)
def test_wiring_census(require_vulkan, census_guard) -> None:
    """Criterion 12: emit per-mechanism census; fail on unexpected UNWIRED mechanisms.

    Each mechanism named in the M0 table emits one line:
      [WIRING CENSUS] mechanism: <value the mechanism computed>
    or
      [WIRING CENSUS] mechanism: UNWIRED

    A mechanism that reports UNWIRED when it should be wired fails this test.
    Known-unwired mechanisms (retain_viable, ledger_lookup) are marked xfail(strict=True)
    separately — see tests below.

    The census also checks the identity cases that R10 requires be explicit red states:
      - Partitioner: islands_offered == claimed_nodes with both > 1 is IDENTITY (red).
    """
    observations: dict[str, str] = {}
    stalls: list[Stalled] = []
    flags_a_env: dict[str, str] = {}
    flags_b_env: dict[str, str] = {}
    flags_a_doc: "dict | None" = None
    flags_b_doc: "dict | None" = None
    flags_a_log = ""
    flags_b_log = ""

    def _fatal_stall(step: str, exc: Stalled):
        """A stall in a step the rest of the census reads from ends the census.

        The partial census is printed first: a run that stopped at mechanism 3 has still
        observed mechanisms 1 and 2, and discarding those readings because a later step
        hung would be throwing away evidence the run actually produced.
        """
        _record_stall(observations, step, exc)
        _print_census(observations, census_guard)
        if exc.report.kind == KIND_MECHANISM and _watchdog.MECHANISM_STALL_IS_A_DETECTION:
            # A mechanism that never returns has produced no observation, which is exactly
            # what criterion 12 exists to detect.  R13: quote the text, never a count.
            raise AssertionError(exc.args[0]) from exc
        raise _verdict.InstrumentError(
            f"[wiring census instrument failure] {exc.args[0]}"
        ) from exc

    # ── The two counters-file polarities, taken once and read by three mechanisms ────
    #
    # R10's falsifier is "an artifact X produced whose content varies with X's input", so
    # every JSON-published mechanism below is read from BOTH polarities and the census
    # records the pair.  A line that reads the same in both is reported as such and is not
    # allowed to pass as an observation.
    try:
        clean_doc, clean_log = _run_counters_child(
            inject=False, tag="clean", guard=census_guard
        )
        inject_doc, inject_log = _run_counters_child(
            inject=True, tag="inject", guard=census_guard
        )

        # ── Run session and collect observations ────────────────────────────
        counters_before, counters_after, profile_info = _watchdog.guarded_call(
            census_guard,
            "add_session_profiling",
            _run_add_session_with_profiling,
            budget_units=_budget_for("add_session_profiling"),
            kind=KIND_MECHANISM,
        )
    except Stalled as exc:
        _fatal_stall(exc.report.last_beat.split(":")[0], exc)

    # ── The two flag-frame arms (criterion 12, round 29) ─────────────────────
    #
    # In a try of their OWN, and after everything else has been observed.  These two arms
    # are read by one mechanism; the clean and planted arms are read by three each and the
    # profiling session by four, so an outage here must cost exactly `flag_frame` and
    # nothing else.  R13: an arm that did not run is ERROR(instrument) against the
    # mechanism it was the observation for, never a finding that the switches are inert,
    # and never a reason to discard readings the run did reach.
    #
    # `trace=False` on both, and that is load-bearing rather than tidiness: arming the
    # tracer unlinks the shared witness path, so an arm that runs after the clean arm and
    # dispatches nothing would delete the only evidence the tracer ever ran and turn a
    # passing tracer lane into a false UNWIRED purely by running later.
    #
    # Arm A arms everything whose observable is an artifact of its own (a counters field, a
    # probe file, a claim log) plus the two log-marker switches, which are separable
    # because their markers differ.  Arm B arms VERBOSE and nothing else, because VERBOSE's
    # only observable is the log level itself — sharing an arm with CLAIM_DEBUG would make
    # either one's reading unattributable, and a confounded discriminator that reads MOVED
    # is worth nothing.
    _CENSUS_BACKEND_PROBE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CENSUS_BACKEND_PROBE_PATH.unlink(missing_ok=True)
    _CENSUS_CLAIM_LOG_PATH.unlink(missing_ok=True)
    flags_a_env = {
        _ENV_DEVICE_MEMORY: "1",
        _ENV_QUARANTINE_SPANS: "8",
        # 1024, not 64, and the number is a finding rather than a preference.
        # `ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY=1` together with
        # `ONNXRUNTIME_EP_VULKAN_VA_RESERVE_MIB=64` crashes the EP with an access violation
        # (0xC0000005) on this six-node elementwise chain, deterministically; either switch
        # alone is clean and 1024 is clean.  Pinned by
        # `test_device_memory_with_small_va_reservation` below and reported to Tank.  The
        # census arms the working value because a census that takes its own process down
        # observes nothing: the crash is a finding and belongs in a test, not in the
        # instrument.
        _ENV_VA_RESERVE_MIB: "1024",
        _ENV_GEMV_PACKED: "1",
        _ENV_BACKEND_PROBE: str(_CENSUS_BACKEND_PROBE_PATH),
        _ENV_CLAIM_LOG: str(_CENSUS_CLAIM_LOG_PATH),
        _ENV_CLAIM_DEBUG: "1",
        _ENV_DUMP_OUTPUT_BYTES: "1",
        _ENV_CHILD_LOG_SEVERITY: "0",
    }
    flags_b_env = {_ENV_VERBOSE: "1", _ENV_CHILD_LOG_SEVERITY: "0"}
    try:
        flags_a_doc, flags_a_log = _run_counters_child(
            inject=False, tag="flags_a", guard=census_guard,
            extra_env=flags_a_env, trace=False,
        )
        flags_b_doc, flags_b_log = _run_counters_child(
            inject=False, tag="flags_b", guard=census_guard,
            extra_env=flags_b_env, trace=False,
        )
    except Stalled as exc:
        _record_stall(observations, "counters_child_flags_a", exc)
        stalls.append(exc)
    except _verdict.InstrumentError as exc:
        flags_a_doc = flags_b_doc = None
        observations["flag_frame"] = f"INSTRUMENT-ERROR ({str(exc).splitlines()[0]})"

    # ── Mechanism 1: partitioner ─────────────────────────────────────────
    # Observable: dispatches_executed delta > 0 (EP ran) + subgraphs_live delta > 0 (partitioned)
    dispatches_delta = (
        counters_after.get("dispatches_executed", 0)
        - counters_before.get("dispatches_executed", 0)
    )
    subgraphs_delta = (
        counters_after.get("subgraphs_live", 0)
        - counters_before.get("subgraphs_live", 0)
    )
    claimed = profile_info.get("claimed", 0)
    islands = profile_info.get("islands", 0)

    if dispatches_delta == 0:
        observations["partitioner"] = "UNWIRED (dispatches_executed delta = 0 — EP ran nothing)"
    elif subgraphs_delta == 0:
        observations["partitioner"] = "UNWIRED (subgraphs_live delta = 0 — partitioner produced no live subgraph)"
    else:
        observations["partitioner"] = (
            f"dispatches_executed={dispatches_delta}, subgraphs_live={subgraphs_delta}, "
            f"claimed_from_profiling={claimed}, islands_from_profiling={islands} "
            f"(both numbers are DELTAS across this session; claimed counts graph nodes "
            f"and islands counts fused subgraphs, so one island over many claimed nodes "
            f"is a success and not a catastrophe)"
        )

    # ── Mechanism 2: partition_identity_check ────────────────────────────
    # R10 identity check: when multiple nodes are claimed, islands < claimed means merging happened.
    # For a single Add node: claimed=1, islands=1 — identity is vacuous (expected for 1 node).
    if claimed > 1 and islands > 1:
        if islands == claimed:
            observations["partition_identity_check"] = (
                f"IDENTITY (red): islands={islands} == claimed={claimed} "
                f"— partitioner ran but produced no merges (indistinguishable from not running)"
            )
        else:
            observations["partition_identity_check"] = (
                f"PASS: islands={islands} < claimed={claimed} (partitioner merged nodes)"
            )
    else:
        observations["partition_identity_check"] = (
            f"VACUOUS (claimed={claimed}, islands={islands}; single-node graph is expected)"
        )

    # ── Mechanism 3: GPU tracer ──────────────────────────────────────────
    #
    # SCREEN DEFECT, found 2026-08-01 and fixed here.  This line read
    # `ONNXRUNTIME_EP_VULKAN_TRACE_FILE`.  No such variable exists: `trace.rs::ENV_TRACE`
    # is `ONNXRUNTIME_EP_VULKAN_TRACE`.  The census therefore reported the tracer as
    # OPTIONAL-UNWIRED on every run it has ever made, and would have gone on doing so if
    # Niobe had deleted the tracer.  This is Link's ICD defect with the polarity flipped:
    # his screen matched something always true, this one matched something never true, and
    # both report a constant.
    #
    # The census now ARMS the tracer in its own counters child rather than waiting for
    # someone else's environment, so the line carries a span count this run produced.
    if _CENSUS_TRACE_PATH.is_file():
        try:
            observations["gpu_tracer"] = _gpu_tracer_observation(_CENSUS_TRACE_PATH)
        except Exception as exc:  # noqa: BLE001
            observations["gpu_tracer"] = f"INSTRUMENT-ERROR (trace file unreadable: {exc})"
    else:
        observations["gpu_tracer"] = (
            f"UNWIRED ({_ENV_TRACE} was set for the counters child and no trace file "
            "was written on EP teardown)"
        )

    # ── Mechanism 4: model_output_equivalence ────────────────────────────
    # This session runs Add, not Phi-3.5. The equivalence verdict belongs to test_phi35.py
    # and test_criterion10.py. Read from ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE if set.
    #
    # CRITERION 12 (g) — THE CENSUS LINE CARRIES THE FRAME IT OBSERVED IN.
    # Amended 2026-07-31T07:45:10-07:00: "a census that reports a verdict without its
    # executor reports a value from a world it has not identified."  For this mechanism
    # the frame is `executed_by`, so the census line prints it and reports the mechanism
    # as unobserved when it is absent — a MATCH with no attribution is precisely the
    # specimen that motivated the amendment.
    #
    # CRITERION 12 (h) — THREE STATES, NOT TWO (R13).
    #   OBSERVED         — a record exists, carrying a verdict AND its executor.
    #   UNWIRED          — no record: the mechanism produced nothing in this run.
    #   INSTRUMENT-ERROR — the counters file exists but could not be read or parsed.
    # A census whose vocabulary is *observed or not observed* records a crashed mechanism
    # as an absence and an absence as a crash.
    equiv = "UNWIRED (no counters file)"
    counters_file_path = os.environ.get("ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE")
    if counters_file_path:
        path = Path(counters_file_path)
        if not path.is_file():
            equiv = "UNWIRED (counters file not written)"
        else:
            try:
                with open(path) as fh:
                    file_counters = json.load(fh)
            except Exception as exc:  # noqa: BLE001
                equiv = f"INSTRUMENT-ERROR (counters JSON unreadable: {exc})"
            else:
                token = file_counters.get(m.EQUIVALENCE_KEY)
                record = file_counters.get(m.EQUIVALENCE_RECORD_KEY)
                if token is None:
                    equiv = "UNWIRED (no model_output_equivalence field)"
                elif not isinstance(record, dict) or not record.get("executed_by"):
                    # (g): a verdict with no frame is not an observation.
                    equiv = (
                        f"UNWIRED (verdict {token} carries no executed_by — a verdict "
                        "without its executor is a value from an unidentified world; "
                        "§10.0 third amendment)"
                    )
                else:
                    equiv = (
                        f"OBSERVED verdict={token} executed_by={record['executed_by']} "
                        f"source={record.get('attribution_source')} "
                        f"permits_triple={record.get('permits_triple_and_ratio')}"
                    )
    observations["model_output_equivalence"] = equiv
    if equiv.startswith("UNWIRED") and m.EQUIVALENCE_KEY in clean_doc:
        # Fall back to the census's own counters child, which does have a counters file.
        # This is a real Add/Mul chain, not Phi-3.5, so the honest reading is UNMEASURED
        # with no record — and UNMEASURED is a value this run computed, not an absence.
        token = clean_doc.get(m.EQUIVALENCE_KEY)
        record = clean_doc.get(m.EQUIVALENCE_RECORD_KEY)
        frame = record.get("executed_by") if isinstance(record, dict) else None
        observations["model_output_equivalence"] = (
            f"census-child verdict={token!r} executed_by={frame!r} "
            "(this lane runs an elementwise chain, not the criterion-10 model; the "
            "attributed MATCH belongs to test_criterion10.py and is not claimed here)"
        )

    # ── Mechanism 5: retain_viable ───────────────────────────────────────
    # Observable: `viable_islands_retained` in the C ABI counters (added ABI version 2).
    # Present even at 0 — an always-0 result is distinguishable from UNWIRED (key absent)
    # because the key is emitted by the production path.  Owner: Mouse.
    if "viable_islands_retained" in counters_after:
        observations["retain_viable"] = (
            f"viable_islands_retained={counters_after['viable_islands_retained']!r} "
            "(the mechanism's own namesake counter, named here because a line that "
            "reports a bare number does not say which counter it read — this mechanism's "
            "extent against the independent whole was 0/1 for exactly that reason)"
        )
    else:
        observations["retain_viable"] = "UNWIRED (no viable_islands_retained field)"

    # ── Mechanism 5b: net_benefit_gate (RAI-011, closed 2026-08-01) ──────
    #
    # THREE STATES THAT USED TO SHARE ONE `0`.  Before Mouse's change, a census reading
    # `viable_islands_retained == 0` could not tell apart:
    #   (i)   the gate never ran            — nothing reached the decision point;
    #   (ii)  the gate was bypassed         — clusters were seen, none was evaluated;
    #   (iii) the gate ran and rejected all — a result, and the only one of the three that
    #                                          is a statement about the model.
    # R12 forbids spelling (i) and (ii) as `0`.  They are now separate FIELDS, so the
    # census prints the fields and never re-derives the distinction from the number.
    #
    # `net_benefit_gate` itself is a token (UNWIRED / BYPASSED / EVALUATED / MIXED)
    # computed by `counters.rs::net_benefit_gate_state()` from three atomics that this run
    # incremented.  It is not a flag anyone set.
    gate_token = clean_doc.get("net_benefit_gate")
    # The gate's cost model is parameterised by six env switches.  A gate observed at its
    # defaults and a gate observed under an overridden cost model are different
    # observations, and a line that does not say which one it is cannot be replayed —
    # which is why the census's extent against this mechanism's own surface was 0/6.
    gate_inputs = " ".join(f"{n}={_in_force(n)}" for n in _ENV_PARTITION)
    if gate_token is None:
        observations["net_benefit_gate"] = (
            "UNWIRED (no net_benefit_gate field in the counters artifact — the gate did "
            "not publish, which is not the same as the gate rejecting everything) "
            f"cost_model_inputs: {gate_inputs}"
        )
    elif gate_token == "UNWIRED":
        observations["net_benefit_gate"] = (
            f"UNWIRED (clusters_seen={clean_doc.get('net_benefit_gate_clusters_seen')!r} "
            "— no cluster reached the decision point in this run) "
            f"cost_model_inputs: {gate_inputs}"
        )
    else:
        observations["net_benefit_gate"] = (
            f"{gate_token} clusters_seen={clean_doc.get('net_benefit_gate_clusters_seen')!r} "
            f"evaluations={clean_doc.get('net_benefit_gate_evaluations')!r} "
            f"bypasses={clean_doc.get('net_benefit_gate_bypasses')!r} "
            f"sole_island_overrides={clean_doc.get('net_benefit_sole_island_overrides')!r} "
            f"viable_islands_retained={clean_doc.get('viable_islands_retained')!r} "
            f"(retained is typed: 'UNWIRED'/'UNOBSERVABLE'/int — a type cannot be forged "
            f"by an increment) cost_model_inputs: {gate_inputs}"
        )

    # ── Mechanism 6: ledger_lookup ───────────────────────────────────────
    # §8.9 criterion 11.  Landed 2026-08-01 (Mouse).  This was the last `UNWIRED` mechanism.
    #
    # R12: three states, and they must be distinguishable.
    #   UNOBSERVABLE — the counters artifact has no ledger fields at all: this frame cannot
    #                  see the ledger, so nothing below is a reading.
    #   UNWIRED      — the fields exist but `proven_key_lookups` is 0: a ledger is compiled in
    #                  and nothing consulted it.
    #   FAULTED      — the ledger parsed with faults; R13 says an instrument error is never a
    #                  detection, so this is not reported as "declined everything".
    #   <token>      — ALL-PROVEN / ALL-DECLINED / MIXED, computed by
    #                  `counters.rs::ledger_gate_state()` from counts this run incremented.
    #
    # `ledger_hits` is itself typed ('UNWIRED'/'UNOBSERVABLE'/int) for the same reason
    # `viable_islands_retained` is: an increment can forge a number, not a type.
    if "proven_key_lookups" not in clean_doc:
        observations["ledger_lookup"] = (
            "UNOBSERVABLE (the counters artifact carries no ledger fields — no ledger exists "
            "in this frame, which is not the same as a ledger nothing consulted)"
        )
    elif clean_doc.get("ledger_faults", 0):
        observations["ledger_lookup"] = (
            f"FAULTED (ledger_faults={clean_doc.get('ledger_faults')!r} — the ledger did not "
            "parse; every form declines, and that decline is an instrument error, not a finding)"
        )
    elif clean_doc.get("proven_key_lookups", 0) == 0:
        observations["ledger_lookup"] = (
            f"UNWIRED (ledger_entries={clean_doc.get('ledger_entries')!r} but "
            "proven_key_lookups=0 — a ledger is compiled in and nothing consulted it)"
        )
    else:
        observations["ledger_lookup"] = (
            f"{clean_doc.get('ledger_gate')} proven_key_lookups="
            f"{clean_doc.get('proven_key_lookups')!r} "
            f"ledger_hits={clean_doc.get('ledger_hits')!r} "
            f"ledger_entries={clean_doc.get('ledger_entries')!r} "
            f"unproven_declines={clean_doc.get('unproven_declines')!r} "
            f"unproven_forms_claimed={clean_doc.get('unproven_forms_claimed')!r} "
            f"unproven_forms_enabled={clean_doc.get('unproven_forms_enabled')!r} "
            f"(hits is typed: 'UNWIRED'/'UNOBSERVABLE'/int) | ledger frame: "
            f"{_ENV_LEDGER_FILE}={_in_force(_ENV_LEDGER_FILE)} "
            f"ONNXRUNTIME_EP_VULKAN_CLAIM_UNPROVEN="
            f"{_in_force('ONNXRUNTIME_EP_VULKAN_CLAIM_UNPROVEN')} "
            "(the ledger the lookup consulted is named by the first; the second is what "
            "would let an unproven form be claimed anyway, and a lookup reading observed "
            "without both is not replayable)"
        )

    # ── Mechanism 7: validation_messenger ───────────────────────────────
    # R13: this mechanism shells out.  A child that fails to start is an ERROR(instrument)
    # — the census reached no observation — and must never be scored as "UNWIRED", which
    # is a finding.
    #
    # The wall-clock budget that used to guard this call is gone; it is watched by
    # `_watchdog.guarded_run`, whose budget is a count of reference computations this
    # machine completed during this run.  That is what makes the contention this suite was
    # measured at (4.4x, 9.5x on the `record` step) irrelevant to the verdict rather than
    # merely survivable: the window widens by exactly as much as the machine slowed.  A
    # child that goes silent here reads STALLED against `validation_messenger`, which is a
    # different token from UNWIRED on purpose.
    epctl = _epctl_path()
    if epctl is None:
        observations["validation_messenger"] = "UNWIRED (epctl not found)"
    else:
        try:
            result = _watchdog.guarded_run(
                [str(epctl), "--probe-validation"],
                guard=census_guard,
                what="census validation probe",
                label="validation_probe",
                budget_units=_budget_for("validation_probe"),
                kind=KIND_MECHANISM,
            )
        except Stalled as exc:
            _record_stall(observations, "validation_probe", exc)
            stalls.append(exc)
        except _verdict.InstrumentError as exc:
            observations["validation_messenger"] = (
                f"INSTRUMENT-ERROR ({str(exc).splitlines()[0]})"
            )
        else:
            state, reason = _verdict.classify_validation_probe(
                result.returncode, (result.stdout or "") + (result.stderr or "")
            )
            # The probe's three parameters, disclosed.  A messenger observed with
            # VALIDATE unset is a different observation from one observed with it armed,
            # and the census line could not previously tell them apart — extent 0/3.
            probe_frame = " ".join(f"{n}={_in_force(n)}" for n in _ENV_VALIDATION)
            if state == _verdict.VALIDATION_ARMED:
                observations["validation_messenger"] = f"ARMED probe_frame: {probe_frame}"
            elif state == _verdict.VALIDATION_PROBE_ERROR:
                observations["validation_messenger"] = (
                    f"INSTRUMENT-ERROR ({reason}) probe_frame: {probe_frame}"
                )
            else:
                observations["validation_messenger"] = (
                    f"OPTIONAL-UNWIRED ({reason}) probe_frame: {probe_frame}"
                )

    # ── Mechanism 8: layering_lint ───────────────────────────────────────
    # The layering lint runs as a cargo integration test (rust/tests/layering.rs).
    # We run it in DEBUG mode (not --release) to avoid relinking the production DLL,
    # which may be loaded by the current process and locked on Windows.
    #
    # THIS IS THE CALL THAT KILLED THE CENSUS.  `cargo test` here may compile the whole
    # crate; 120 s was a quiet-machine number and `subprocess.TimeoutExpired` propagated
    # out of the test as an uncaught exception, which pytest scored as a red.  It is
    # neither a pass nor a detection: it is ERROR(instrument), and under R13 it must not
    # be counted as one of the suite's failures.
    #
    # The number is gone rather than bigger.  `cargo` prints a line per crate it compiles,
    # so a build crawling under load beats continuously and cannot reach the stall budget
    # however slow it gets; a `cargo` that has actually wedged prints nothing while the
    # machine keeps completing reference units, and that disagreement is the signal.
    if _CARGO_MANIFEST.is_file():
        try:
            lr = _watchdog.guarded_run(
                [
                    "cargo", "test", "--test", "layering",
                    "--manifest-path", str(_CARGO_MANIFEST),
                    # No --release: debug build avoids relinking the loaded DLL.
                ],
                guard=census_guard,
                what="census layering lint",
                label="layering_lint",
                budget_units=_budget_for("layering_lint"),
                # A cargo compile is scaffolding the census needs but does not judge, so a
                # stall here is ERROR(instrument) and not a finding about the lint.  Note
                # cargo emits a line per crate, so a build that is merely crawling under
                # load keeps beating and never reaches this budget — but a COLD build of
                # one large crate emits `Compiling onnxruntime-ep-vulkan` and then nothing
                # at all while rustc works, and that silence did reach this budget
                # (measured: 12015 units, isolated, nothing else running).  `progress_paths`
                # gives the guard a second beat source for exactly that case: what the
                # child has written.  A wedged cargo writes nothing and is caught as before.
                # Depth matters and is measured, not chosen: at depth 1 this witness beat
                # 4098 times and the census STILL stalled, because a compiling rustc
                # touches only `target/debug/incremental/<hash>/s-*-working/`.
                kind=KIND_TOOLCHAIN,
                progress_paths=[_CARGO_MANIFEST.parent / "target" / "debug"],
                env=_cargo_env(),
            )
        except Stalled as exc:
            _record_stall(observations, "layering_lint", exc)
            stalls.append(exc)
        except (_verdict.InstrumentError, _watchdog.InstrumentAbsent) as exc:
            observations["layering_lint"] = f"INSTRUMENT-ERROR ({str(exc).splitlines()[0]})"
        else:
            if lr.returncode == 0:
                observations["layering_lint"] = "PASS"
            else:
                observations["layering_lint"] = (
                    f"FAIL (exit {lr.returncode}): {(lr.stderr or '')[-200:]}"
                )
    else:
        observations["layering_lint"] = "UNWIRED (Cargo.toml not found)"

    # ── Mechanism 9: broken_commitment_warn (Tank, 2026-08-01) ──────────
    #
    # Tank's runtime WARN: a node this EP *claimed* whose Compute() returned non-OK must be
    # disclosed through ORT's own logging sink, because a WARN in our private log is
    # invisible to exactly the audience the ruling names.
    #
    # The census does NOT re-judge it — `probe_broken_commitment.py` owns that, and it was
    # mutation-tested in both polarities today.  What the census does is read the counters
    # his call site writes, on two runs that differ only in whether a Compute failure was
    # planted, and require the reading to MOVE.
    #
    # R12 is load-bearing on the clean side: `broken_commitment_warn_channel` reads
    # `UNOBSERVABLE` when no commitment broke, never `ORT_SINK`.  A channel is not proven
    # by the absence of traffic — that is the whole failure this instrument exists for.
    clean_channel = clean_doc.get("broken_commitment_warn_channel")
    inject_channel = inject_doc.get("broken_commitment_warn_channel")
    clean_broken = clean_doc.get("broken_commitments")
    inject_broken = inject_doc.get("broken_commitments")
    if clean_channel is None and inject_channel is None:
        observations["broken_commitment_warn"] = (
            "UNWIRED (no broken_commitment_warn_channel field in either polarity's "
            "counters artifact)"
        )
    elif inject_broken in (None, 0, "UNOBSERVABLE"):
        observations["broken_commitment_warn"] = (
            f"UNWIRED (the planted-failure polarity recorded broken_commitments="
            f"{inject_broken!r}; {_ENV_INJECT} did not reach a claimed node, so the WARN "
            "had nothing to disclose and this run says nothing about the channel)"
        )
    elif clean_channel == inject_channel:
        observations["broken_commitment_warn"] = (
            f"UNWIRED (constant: both polarities read channel={clean_channel!r}; a "
            "reading that does not move with its input is not an observation, R10)"
        )
    else:
        ort_warn_lines = _ort_sink_warn_lines(inject_log)
        observations["broken_commitment_warn"] = (
            f"planted: channel={inject_channel!r} broken_commitments={inject_broken!r} "
            f"compute_failures={inject_doc.get('compute_failures')!r} "
            f"fault_injection={inject_doc.get('fault_injection')!r} "
            f"ort_sink_warn_lines={len(ort_warn_lines)} | "
            f"clean: channel={clean_channel!r} broken_commitments={clean_broken!r} "
            f"compute_failures={clean_doc.get('compute_failures')!r} "
            f"fault_injection={clean_doc.get('fault_injection')!r} | "
            f"armed by {_ENV_INJECT}={_in_force(_ENV_INJECT)} in this process, set to '1' "
            "in the planted arm's child environment only (the EP caches its environment "
            "at DLL load on Windows, so an in-process set would be read by nothing)"
        )

    # ── Mechanism 10: device_state_guard (Link, 2026-08-01) ─────────────
    #
    # Link's obligation-8 publication guard: a duration quoted as a lane claim without a
    # `device_state` companion is not quotable.  He proved it fires by planting a
    # companionless `gpu_busy_ms` and getting FAIL(condition=STEADY_UNCERTIFIED), exit 1.
    #
    # The census runs HIS module — imported, not reimplemented — over two inputs: this
    # run's own artifact directory, and a scratch directory carrying a planted
    # companionless duration.  The pair is the observation.  A guard that returned the same
    # token for both would be reported UNWIRED here however green it looked.
    #
    # Link's own lesson from today is why the planted case is present at all: his ICD probe
    # was matching a line printed on every run, so the negative control had never once
    # executed while reporting that the gate could not fail.
    observations["device_state_guard"] = _observe_device_state_guard()

    # ── Mechanism 11: instrument_census (Tank's source census) ──────────
    #
    # Criterion 12 pointed at itself.  `rust/tools/audit_instruments.py` is the ONE census
    # of instrument wiring in this repo; this line reports its verdict token and the four
    # counts it computed on this run.  Deliberately not a second census — Tank dropped his
    # harness-census WIP unmerged for exactly this reason, and his six states
    # (absent → uninvoked → unfalsified → unreachable → out-of-frame → misnamed, ordered by
    # how late the failure is discoverable) are the vocabulary used throughout this file.
    observations["instrument_census"] = _observe_instrument_census()

    # ── Mechanism 12: ep_entrypoints (criterion 12, round 29) ───────────
    observations["ep_entrypoints"] = _observe_ep_entrypoints(clean_doc, inject_doc)

    # ── Mechanism 13: flag_frame (criterion 12, round 29) ───────────────
    #
    # Nine environment switches that production Rust reads and that no census mechanism
    # observed.  Two of them select a different code path, which is why they are first in
    # the line: a census that cannot say which path ran is reporting a value from a world
    # it has not identified, and that is the `executed_by` argument (§10.0 third metric
    # amendment) applied to the census instead of to the verdict.
    if "flag_frame" not in observations:
        observations["flag_frame"] = _observe_flag_frame(
            clean_doc, clean_log, flags_a_doc, flags_a_log, flags_b_doc, flags_b_log,
            flags_a_env, flags_b_env,
        )

    # ── Emit census ──────────────────────────────────────────────────────
    _print_census(observations, census_guard)

    # ── Assertions ───────────────────────────────────────────────────────
    # Mandatory wired — fail if UNWIRED (excluding known-unwired).
    failures = []
    for mech in _MANDATORY_WIRED - _KNOWN_UNWIRED_M0:
        obs = observations.get(mech, "UNWIRED")
        if obs.startswith("UNWIRED"):
            failures.append(f"  {mech}: {obs}")

    # Partition identity check (red state).
    pi = observations.get("partition_identity_check", "")
    if pi.startswith("IDENTITY"):
        failures.append(f"  partition_identity_check: {pi}")

    # model_output_equivalence — warn but do not fail (test_phi35.py and test_criterion10.py
    # own the assertion; an absent record here just means the model cache is absent on this
    # machine).  The three states are printed distinctly (criterion 12 (h), R13): an
    # INSTRUMENT-ERROR is a harness outage and is never read as "the verdict was bad".
    equiv_obs = observations.get("model_output_equivalence", "")
    if equiv_obs.startswith("INSTRUMENT-ERROR"):
        print(
            f"[WIRING CENSUS] INSTRUMENT-ERROR on model_output_equivalence: {equiv_obs}. "
            "This is a harness outage, NOT a finding about the EP (R13). An instrument "
            "error never counts as a detection.",
            file=sys.stderr,
        )
    elif equiv_obs.startswith("UNWIRED"):
        print(
            f"[WIRING CENSUS] WARNING: model_output_equivalence {equiv_obs}. "
            "Expected when the Phi-3.5 model cache is absent. The canonical attributed "
            "reading is test_criterion10.py::"
            "test_criterion_10_three_consecutive_attributed_match.",
            file=sys.stderr,
        )

    assert not failures, (
        "Wiring census FAILED — the following mandatory mechanisms are UNWIRED:\n"
        + "\n".join(failures)
        + "\n\nAll census observations:\n"
        + "\n".join(f"  {m}: {observations.get(m, 'UNWIRED')}" for m in _MECHANISMS)
    )

    # ── Stalls (non-fatal ones — the fatal ones raised at the step) ───────
    #
    # A step that never returned is scored against the mechanisms it was the sole
    # observation for, and it is NOT scored as UNWIRED: "ran and produced nothing" is a
    # finding about the call graph, "never came back" is a finding about a step, and a
    # census that spells them the same way has lost the one it exists to report.
    #
    # Which terminal state a stall gets depends on whose stall it is, not on how long it
    # took.  A mechanism under census that never returns HAS produced no observation of
    # itself, which is precisely criterion 12's subject — FAIL(condition=NO_PROGRESS).
    # A toolchain step (cargo compiling the crate) is scaffolding the census does not
    # judge — ERROR(instrument).  See the note in `_watchdog.py` on why Tank's
    # ERROR(instrument)-for-timeouts ruling does not carry over: it reasons from a
    # wall-clock timeout's firing being evidence about the box, and this detector's firing
    # is not, which is the property demonstrated in all four cells of probe_stall_guard.py.
    mechanism_stalls = [e for e in stalls if e.report.kind == KIND_MECHANISM]
    toolchain_stalls = [e for e in stalls if e.report.kind == KIND_TOOLCHAIN]
    if mechanism_stalls and _watchdog.MECHANISM_STALL_IS_A_DETECTION:
        # R13: quote the failure text, never a failure count.
        raise AssertionError(
            "Wiring census FAILED — a mechanism under census made no forward progress:\n"
            + "\n\n".join(e.args[0] for e in mechanism_stalls)
            + "\n\nAll census observations:\n"
            + "\n".join(f"  {m}: {observations.get(m, 'UNWIRED')}" for m in _MECHANISMS)
        )
    if toolchain_stalls:
        raise _verdict.InstrumentError(
            "[wiring census instrument failure] the census could not reach an "
            "observation because a toolchain step made no forward progress:\n"
            + "\n\n".join(e.args[0] for e in toolchain_stalls)
        )

    # R13, LAST — and deliberately after the condition assertion above.
    #
    # An instrument outage is a lane failure of a DIFFERENT KIND, so it gets a different
    # exception type (`InstrumentError`, which conftest classifies as ERROR(instrument))
    # and it never lands in `failures`, which is the census's detection channel.  A census
    # that timed out on one mechanism has not observed that mechanism; reporting it as
    # UNWIRED would be a fabricated detection, and reporting the census as green would be
    # a fabricated observation.  Both have happened here.
    #
    # Ordering: a real UNWIRED finding outranks an outage elsewhere, because the finding
    # was actually observed.
    instrument_errors = [
        f"  {mech}: {obs}"
        for mech in _MECHANISMS
        if (obs := observations.get(mech, "")).startswith("INSTRUMENT-ERROR")
    ]
    if instrument_errors:
        raise _verdict.InstrumentError(
            "[wiring census instrument failure] ERROR(instrument): the census could not "
            "observe the following mechanisms, so it has said nothing about them:\n"
            + "\n".join(instrument_errors)
            + "\n\nAn instrument error NEVER counts as a detection (R13). None of the "
            "lines above is evidence that a mechanism is unwired. Note the census no "
            "longer carries a wall-clock timeout at all: its stall detector is "
            "denominated in reference computations this machine completed during this "
            "run (see `stall_detector` in the artifact), so contention cannot by itself "
            "produce a line above.\n\n"
            "All census observations:\n"
            + "\n".join(f"  {mech}: {observations.get(mech, 'UNWIRED')}" for mech in _MECHANISMS)
        )


# ---------------------------------------------------------------------------
# Separate xfail tests for known-unwired mechanisms (criterion 12 sub-items)
# These are NOT inside test_wiring_census to keep the census readable.
# They are xfail(strict=True) so they surface as XPASS when Mouse wires them.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB"),
    reason="ONNXRUNTIME_VULKAN_EP_LIB not set",
)
def test_the_counters_mirror_matches_the_running_dll(require_vulkan) -> None:
    """§8.9.15 — the DLL publishes its own field offsets and the derived mirror must equal them.

    `device_losses` was inserted mid-struct on 2026-08-02 without a version bump and without
    updating the three ctypes mirrors. Every field below it shifted eight bytes, and the census's
    `dispatches_executed` began reading `device_losses`: always `0`, so mechanism 1 reported
    `UNWIRED (EP ran nothing)` about a run that dispatched normally. Nothing went red — the
    number was stable and plausible, which is the R11 shape. Hours later `898a2ba` inserted three
    more fields in the same place and `ledger_entries` read `0` against a true 97.

    Two things changed since. The mirror is generated from `counters.rs`, so it cannot be stale
    relative to *source*; and the DLL now exports `OrtEpVulkanGetCountersLayout`, a per-field
    offset manifest, so it cannot be stale relative to the *binary* either. This lane compares
    them field by field rather than by size: a size check says only *that* two layouts differ, and
    the old guard here compared with `<`, which is exactly why a struct that **grew** by three
    fields sailed through.

    The `struct_size` equality is deliberate. An *append* is safe to read with an old mirror, an
    *insertion* is not, and from the reader's side the two are indistinguishable.
    """
    import ctypes as _ct

    lib = os.environ["ONNXRUNTIME_VULKAN_EP_LIB"]
    manifest = _counters_abi.dll_manifest(lib)
    ours = _counters_abi.expected_offsets()
    assert manifest["fields"] == ours, (
        f"the DLL's field offsets are not this checkout's.\n"
        f"{_counters_abi.misattribution(manifest['fields'])}"
    )
    assert manifest["layout_hash"] == _counters_abi.layout_hash(), (
        f"offsets agree but layout hashes do not: DLL 0x{manifest['layout_hash']:016x}, "
        f"checkout 0x{_counters_abi.layout_hash():016x}."
    )
    assert manifest["abi_version"] == _counters_abi.abi_version()
    assert _counters_abi.layout_is_declared(), (
        f"({_counters_abi.abi_version()}, 0x{_counters_abi.layout_hash():016x}) is not in "
        f"COUNTERS_LAYOUT_REGISTRY — a layout nobody declared. Run "
        f"`python rust/tools/counters_abi.py` for the row to append."
    )

    mirror = _counters_abi.make_mirror()
    dll = _ct.CDLL(lib)
    c = mirror()
    written = dll.OrtEpVulkanGetExecutionCounters(_ct.byref(c), _ct.c_size_t(_ct.sizeof(c)))
    assert written > 0, "the DLL wrote no counters at all"
    assert c.struct_size == _ct.sizeof(c), (
        f"the DLL's VulkanEpCounters is {c.struct_size} bytes, the derived mirror is "
        f"{_ct.sizeof(c)}. A mirror that is the wrong size does not read smaller numbers, it "
        f"reads different fields."
    )
    assert c.abi_version == _counters_abi.abi_version()


@pytest.mark.skipif(
    not os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB"),
    reason="ONNXRUNTIME_VULKAN_EP_LIB not set",
)
def test_retain_viable_wired(require_vulkan) -> None:
    """Criterion 12 sub-item: retain_viable reports a computed value, not UNWIRED.

    Wired 2026-07-30 (Mouse): `viable_islands_retained` added to the C ABI counters struct
    (ABI version 2) and emitted from `GetCapability` for multi-cluster graphs. The key is
    present even at 0 — distinguishable from UNWIRED (key absent) per R10. The xfail was
    removed when the counter appeared in the ctypes read.
    """
    _, counters_after, _ = _run_add_session_with_profiling()
    assert "viable_islands_retained" in counters_after, (
        "retain_viable counter not present in EP counters (C ABI) — mechanism is UNWIRED. "
        f"Current counter keys: {sorted(counters_after.keys())}"
    )


@pytest.mark.skipif(
    not os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB"),
    reason="ONNXRUNTIME_VULKAN_EP_LIB not set",
)
def test_ledger_lookup_wired(require_vulkan) -> None:
    """Criterion 12 sub-item: §8.9 ledger lookup must produce a computed observation.

    The `xfail(strict=True)` was removed on 2026-08-01 when criterion 11 landed. It is
    replaced by assertions rather than deleted: an expectation that is merely dropped leaves
    no record that the thing it expected has actually happened.

    R10 — this asserts an artifact whose content varies with its input, not that a field
    exists. `proven_key_lookups` must be non-zero (something consulted the ledger) and
    `ledger_gate` must be one of the computed tokens. A field that was always present and
    always `0` would satisfy a presence check and prove nothing.
    """
    _, counters_after, _ = _run_add_session_with_profiling()
    assert "proven_key_lookups" in counters_after, (
        "§8.9 ledger lookup counter not present — mechanism is UNOBSERVABLE in this frame "
        f"(no ledger fields at all). Current counter keys: {sorted(counters_after.keys())}"
    )
    assert counters_after["proven_key_lookups"] > 0, (
        "the ledger fields are published but nothing consulted the ledger — UNWIRED, which "
        "is a distinct state from UNOBSERVABLE and from a real count (R12). "
        f"ledger_entries={counters_after.get('ledger_entries')!r}"
    )
    # The three-state string tokens live on the JSON surface only; the C ABI carries
    # counts.  Both surfaces must agree that the gate ran, which is why both are read.
    doc, _log = _run_counters_child(inject=False, tag="ledger")
    assert doc.get("ledger_gate") in {"ALL-PROVEN", "ALL-DECLINED", "MIXED"}, (
        "ledger_gate did not report a computed token; a FAULTED or absent gate is "
        "ERROR(instrument) and never a detection (R13). "
        f"ledger_gate={doc.get('ledger_gate')!r} ledger_faults={doc.get('ledger_faults')!r}"
    )


# ---------------------------------------------------------------------------
# Criterion 11(c) — `ledger_hits` shown to move with its input, and the three
# planted controls given lane membership.
#
# The shape being guarded against: `ledger_hits=6 proven_key_lookups=6` reads
# identically whether the ledger was genuinely consulted or derived from the same
# enumeration that produced the claims.  **An identity whose two sides come from one
# source is a falsifier that cannot fire.**  So none of what follows asserts that a
# counter is non-zero.  Each pair holds every component of the proof key fixed but one,
# and asserts the two readings DIFFER.
# ---------------------------------------------------------------------------

#: Fields read from every arm.  `ledger_miss` and `ledger_gate` are the tokens; the
#: counts are what an author could have set by hand, which is why the tokens are read too.
_LEDGER_ARM_FIELDS = (
    "proven_key_lookups", "ledger_hits", "ledger_gate", "ledger_miss",
    "ledger_entries", "ledger_faults", "unproven_declines", "claimed_nodes",
    "dispatches_executed",
)


def _ledger_arm(tag: str, *, model=None, ledger_file=None) -> dict:
    """One census child run, reduced to its ledger reading.

    ``trace=False``: these arms have no business exercising the tracer, and a faulted-ledger
    arm dispatches nothing, so arming the shared tracer path here would delete the clean
    arm's tracer witness and make the tracer read UNWIRED depending only on test order.
    """
    extra = {}
    if model is not None:
        extra[_ENV_CENSUS_MODEL] = str(model)
    if ledger_file is not None:
        extra[_ENV_LEDGER_FILE] = str(ledger_file)
    doc, _log = _run_counters_child(inject=False, tag=tag, extra_env=extra, trace=False)
    return {"arm": tag, **{k: doc.get(k) for k in _LEDGER_ARM_FIELDS}}


def _reading(arm: dict) -> tuple:
    """The tuple two arms must differ in.  Named so the failure text can quote it."""
    return (arm["ledger_hits"], arm["ledger_miss"], arm["ledger_gate"], arm["claimed_nodes"])


def _write_ledger_witness(name: str, arms: list[dict], note: str) -> Path:
    out = _CENSUS_OUT
    out.mkdir(parents=True, exist_ok=True)
    selector = os.environ.get("ONNXRUNTIME_EP_VULKAN_DEVICE", "unset")
    path = out / f"criterion11c-{name}-dev{selector}.json"
    path.write_text(
        json.dumps({"note": note, "device_selector": selector, "arms": arms}, indent=2),
        encoding="utf-8",
    )
    return path


def _dynamic_acosh_f32(path: Path) -> Path:
    """`acosh_f32` with symbolic extents — control 2's runtime-extent arm.

    Same op, same dtype, same optional-input set, same node count as the static arm.  The only
    component of the proof key that changes is the shape class (`static` -> `runtime-extent`),
    which is a *path* distinction and must therefore be a different key.

    IT WAS `Mul`, AND `Mul` DISARMED IT (2026-08-04, Mouse).  BERT-SQuAD-12 contains `Mul` on
    f32 with symbolic extents, so covering that model required proving
    `Mul/f32,f32>f32/.../runtime-extent/n2` — the exact form this arm relied on being *absent*.
    The two readings converged and this test went red, which is the behaviour `_static_abs_f16`
    predicts in its own docstring: a control disarmed by a real proof fails loudly rather than
    passing quietly.

    The repair is the same one taken on 2026-08-02 when a proof run disarmed the planted
    control: move the control rather than leave a form a real model needs permanently
    unclaimable.  `Acosh` is proven on f32/static, appears in **none** of the three models
    censused so far (Phi-3.5, MobileNetV2-12, BERT-SQuAD-12), and is not a shape a transformer
    or a CNN mints — so it is the cheapest form to keep unproven.  If a future round proves
    `Acosh/f32/runtime-extent`, this test goes red again and the control moves again; that is
    the mechanism working, not a maintenance burden.
    """
    import onnx_ir as ir  # noqa: PLC0415

    a = ir.Value(name="a", type=ir.TensorType(DT.FLOAT), shape=ir.Shape(["M", "N"]))
    out = ir.Value(name="out", type=ir.TensorType(DT.FLOAT), shape=ir.Shape(["M", "N"]))
    node = ir.node("Acosh", [a], outputs=[out])
    graph = ir.Graph([a], [out], nodes=[node], name="dyn_acosh", opset_imports={"": 21})
    path.write_bytes(ir.to_proto(ir.Model(graph, ir_version=10)).SerializeToString())
    return path


def _static_abs_f16(path: Path) -> Path:
    """`Abs` on f16 with static extents — a form no proof run has entered in the ledger.

    Replaces `evidence/cases/mul_f16_unproven.onnx`, which no longer exists: on 2026-08-02 a
    proof run entered `Mul/f16/static` in the ledger, which was the planted control's own form,
    and the control moved to `sub_f16_dyn_unproven` (shape class) rather than leaving a whole
    dtype of a core op permanently unclaimable. Control 1 here needs a *dtype* axis, so it needs
    a different op: `Abs` is proven on f32 and unproven on f16.

    Built in `tmp_path` rather than checked in, so that it cannot be picked up by the generator
    and quietly proven — which is exactly how the previous control was disarmed. If a future
    round does prove `Abs/f16/static`, the two readings converge and this test goes **red**, not
    silently green.
    """
    import onnx_ir as ir  # noqa: PLC0415

    a = ir.Value(name="a", type=ir.TensorType(DT.FLOAT16), shape=ir.Shape([4, 8]))
    out = ir.Value(name="out", type=ir.TensorType(DT.FLOAT16), shape=ir.Shape([4, 8]))
    node = ir.node("Abs", [a], outputs=[out])
    graph = ir.Graph([a], [out], nodes=[node], name="abs_f16", opset_imports={"": 21})
    path.write_bytes(ir.to_proto(ir.Model(graph, ir_version=10)).SerializeToString())
    return path


@pytest.mark.skipif(
    not os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB"),
    reason="ONNXRUNTIME_VULKAN_EP_LIB not set",
)
def test_ledger_hits_moves_with_its_input(require_vulkan, tmp_path) -> None:
    """Criterion 11(c), control 1 and control 2 — `arms_must_differ`, twice.

    Control 1 (dtype): `abs_f32` is in the ledger, the same graph on f16 is not.  Same op,
    same shape, same graph; only the dtype component of the key differs.

    Control 2 (shape class): the same `Acosh` on f32 with symbolic extents.  Same op, same
    dtype, same optional inputs; only the shape-class component differs.  This is the
    control that matters most for the derived-from-enumeration hypothesis, because the
    *enumeration is identical in both arms* — one node, one lookup, one claimable op — and
    the reading still moves.  A `ledger_hits` computed from the claim enumeration could
    not tell these two apart.

    Nothing here asserts a count is non-zero.  It asserts two readings differ, which is
    the only assertion a derived counter cannot satisfy.
    """
    proven = _ledger_arm("ledger_dtype_proven", model=_EVIDENCE_CASES / "abs_f32.onnx")
    unproven = _ledger_arm(
        "ledger_dtype_unproven", model=_static_abs_f16(tmp_path / "abs_f16_unproven.onnx")
    )
    # Control 2 needs its own static arm: the runtime-extent arm below is `Acosh`, and a
    # shape-class control whose two arms are different ops would not be a shape-class control.
    mul_static = _ledger_arm("ledger_shape_static", model=_EVIDENCE_CASES / "acosh_f32.onnx")
    dynamic = _ledger_arm(
        "ledger_shape_runtime", model=_dynamic_acosh_f32(tmp_path / "dyn_acosh_f32.onnx")
    )
    witness = _write_ledger_witness(
        "ledger-arms",
        [proven, unproven, mul_static, dynamic],
        "criterion 11(c): ledger_hits read under four inputs forming two pairs that differ "
        "in exactly one proof-key component each",
    )

    for arm in (proven, unproven, mul_static, dynamic):
        assert arm["ledger_faults"] == 0, (
            f"ERROR(instrument): the ledger faulted during arm {arm['arm']!r}, so this run "
            "produced no reading about any form and is not evidence about the gate (R13). "
            f"ledger_gate={arm['ledger_gate']!r} arm={arm}"
        )
        assert arm["proven_key_lookups"] > 0, (
            f"ERROR(instrument): arm {arm['arm']!r} never reached the ledger check, so its "
            "reading is NEVER-ATTEMPTED and not a statement about the key. "
            f"arm={arm}  witness={witness}"
        )

    assert _reading(proven) != _reading(unproven), (
        "arms_must_differ FAILED (control 1, dtype). The proven and unproven forms of the "
        "same op produced the SAME ledger reading, so `ledger_hits` does not vary with the "
        "key presented to it — which is what a counter derived from the claim enumeration "
        f"would look like. proven={_reading(proven)} unproven={_reading(unproven)}  "
        f"witness={witness}"
    )
    assert _reading(mul_static) != _reading(dynamic), (
        "arms_must_differ FAILED (control 2, shape class). Same op, same dtype, same "
        "optional inputs, same one-node enumeration — only the extents are symbolic — and "
        "the ledger reading did not move. A proof written for the static form would be "
        "returned for the runtime-extent form, which is the class of defect the key's "
        f"shape-class component exists to prevent. static={_reading(mul_static)} "
        f"runtime-extent={_reading(dynamic)}  witness={witness}"
    )

    # The direction is asserted too: differing is necessary, but two arms could differ with
    # the signs swapped and that would be a gate reading its input backwards.
    for arm in (proven, mul_static):
        assert arm["ledger_miss"] == "HIT" and arm["ledger_hits"] > 0, (
            f"the form WITH a ledger entry did not read HIT: {arm}  witness={witness}"
        )
    for arm in (unproven, dynamic):
        assert arm["ledger_miss"] == "KEY-ABSENT" and arm["ledger_hits"] == 0, (
            f"a form with no ledger entry did not read KEY-ABSENT: {arm}  witness={witness}"
        )


@pytest.mark.skipif(
    not os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB"),
    reason="ONNXRUNTIME_VULKAN_EP_LIB not set",
)
def test_ledger_key_discriminates_optional_inputs(require_vulkan) -> None:
    """Criterion 11(c), the `MatMulNBits` pair — the 2026-07-30 all-zero-logits regression.

    Both forms are proven, so this pair does **not** move `ledger_hits`; saying otherwise
    would be the decomposition that appears to close (R11).  What it pins is the thing that
    defect turned on: the two forms must be *two keys*.  If `populated_optional_input_set`
    were dropped from the key, one ledger entry would answer for both, the ledger would
    hold a duplicate, and a proof of the `scales` form would be returned for the
    `scales+zero_points` form.

    So the assertion is on the artifact, not on a counter: the ledger holds two entries
    whose keys differ **only** in that component, and both forms are claimed on device.
    """
    zp = _ledger_arm(
        "ledger_optin_zp", model=_EVIDENCE_CASES / "matmulnbits_f16_scales_zp.onnx"
    )
    noz = _ledger_arm(
        "ledger_optin_noz", model=_EVIDENCE_CASES / "matmulnbits_f16_scales.onnx"
    )
    witness = _write_ledger_witness(
        "optional-inputs", [zp, noz],
        "criterion 11(c): both MatMulNBits forms are proven; the discriminator is that "
        "they are two keys, not one",
    )

    keys = [
        json.loads(line)["key"]
        for line in _EVIDENCE_LEDGER.read_text(encoding="utf-8").splitlines()[1:]
        if line.strip()
    ]
    nbits = sorted(k for k in keys if "MatMulNBits" in k)
    # The pair is specifically the two arms run above: f16, static extents, differing only in
    # whether `zero_points` is populated. Selected by form rather than by counting every
    # MatMulNBits entry, because the ledger legitimately grows — a f32 pair and an f16
    # runtime-extent form are also proven now, and none of them is this control.
    pair = sorted(
        k for k in nbits if "/f16," in k and k.split("/")[4] == "static"
    )
    assert len(pair) == 2, (
        "expected exactly the two static f16 MatMulNBits forms in the ledger — the pair is "
        f"the regression control for the 2026-07-30 defect. Found: {pair}  (all MatMulNBits "
        f"entries: {nbits})"
    )
    a, b = (k.split("/") for k in pair)
    differing = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    assert differing, (
        f"the two MatMulNBits ledger keys are identical: {pair}. One entry would answer "
        "for both forms."
    )
    # Components: domain::op / opset / dtypes / variant / shape_class / opt_inputs.
    assert 5 in differing, (
        "the two MatMulNBits keys do not differ in their populated-optional-input "
        f"component (index 5). Differing components: {differing}; keys: {pair}. That "
        "component is the one whose absence produced the 2026-07-30 all-zero logits."
    )
    assert differing == [2, 5], (
        "the pair is meant to differ in the optional-input set and the dtype signature it "
        f"implies, and nothing else. Differing components: {differing}; keys: {pair}"
    )

    for arm in (zp, noz):
        assert arm["ledger_faults"] == 0, (
            f"ERROR(instrument): ledger faulted during {arm['arm']!r}: {arm}"
        )
        assert arm["ledger_miss"] == "HIT" and arm["claimed_nodes"] > 0, (
            "a proven MatMulNBits form was not claimed, so this run cannot say the two "
            f"keys are separately honoured: {arm}  witness={witness}"
        )


@pytest.mark.skipif(
    not os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB"),
    reason="ONNXRUNTIME_VULKAN_EP_LIB not set",
)
def test_ledger_digest_refusal_is_in_the_lane(require_vulkan, tmp_path) -> None:
    """Criterion 11(c), control 3 — Mouse's RAI-008(b)(iii) refusal, run in the lane.

    Three arms, and the **identical-file arm is what makes the other two detections**
    rather than a check that fails on everything.  Without it, a `check_baked_against_disk`
    that returned `Some(...)` unconditionally would pass the drift and absent arms and be
    completely broken.

    The two failing arms must reach `LEDGER-FAULTED` and not `KEY-ABSENT`: a faulted
    ledger has produced no reading about any form, and reporting a statement about the form
    would spell an instrument outage exactly like a detection (R13).
    """
    same = _ledger_arm(
        "ledger_digest_same",
        model=_EVIDENCE_CASES / "mul_f32.onnx", ledger_file=_EVIDENCE_LEDGER,
    )
    drifted_path = tmp_path / "drifted_ledger.jsonl"
    lines = _EVIDENCE_LEDGER.read_text(encoding="utf-8").splitlines()
    drifted_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    drifted = _ledger_arm(
        "ledger_digest_drift",
        model=_EVIDENCE_CASES / "mul_f32.onnx", ledger_file=drifted_path,
    )
    absent = _ledger_arm(
        "ledger_digest_absent",
        model=_EVIDENCE_CASES / "mul_f32.onnx",
        ledger_file=tmp_path / "there_is_no_ledger_here.jsonl",
    )
    witness = _write_ledger_witness(
        "digest-refusal", [same, drifted, absent],
        "criterion 11(c): digest refusal, three arms; the identical-file arm is the "
        "control that makes the other two detections",
    )

    assert same["ledger_faults"] == 0 and same["ledger_miss"] == "HIT", (
        "the CONTROL arm faulted: the on-disk ledger is byte-identical to the baked one "
        "and the run still refused to claim. Every other assertion in this test would pass "
        "against a check that rejects everything, so this arm failing makes the whole test "
        f"ERROR(instrument) rather than a detection. arm={same}  witness={witness}"
    )
    for arm in (drifted, absent):
        assert arm["ledger_faults"] > 0, (
            f"the {arm['arm']!r} arm did not fault. A ledger that was asked for and "
            "disagrees, or was asked for and is absent, must refuse to claim rather than "
            f"warn. arm={arm}  witness={witness}"
        )
        assert arm["ledger_miss"] == "LEDGER-FAULTED", (
            "a faulted ledger reported a statement about the FORM rather than about the "
            "instrument. LEDGER-FAULTED outranks KEY-ABSENT because a faulted run has no "
            f"reading about any form (R13). arm={arm}  witness={witness}"
        )
        assert arm["claimed_nodes"] == 0, (
            "a run whose ledger faulted still claimed nodes — it claimed from evidence "
            f"nobody can read. arm={arm}  witness={witness}"
        )

    assert _reading(same) != _reading(drifted), (
        f"arms_must_differ FAILED: identical={_reading(same)} drifted={_reading(drifted)}"
    )


# ---------------------------------------------------------------------------
# The first finding the flag frame produced (criterion 12, round 29)
# ---------------------------------------------------------------------------
#
# Arming two of the twelve uncensused switches together crashed the EP.  It was found on
# the first run of the flag-frame arm, which is the whole argument for censusing surfaces
# nobody watches: neither switch has a test, their interaction had never been exercised,
# and `unwired: []` was true throughout.
#
# THE SPECIMEN.  `ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY=1` with
# `ONNXRUNTIME_EP_VULKAN_VA_RESERVE_MIB=64` terminates the process with STATUS_ACCESS_
# VIOLATION (0xC0000005, exit code -1073741819) during session creation on a six-node
# elementwise chain.  Bisected on selector 0:
#
#   DEVICE_MEMORY=1                         → exit 0, counters written
#   VA_RESERVE_MIB=64                       → exit 0, counters written
#   DEVICE_MEMORY=1 VA_RESERVE_MIB=1024     → exit 0, counters written
#   DEVICE_MEMORY=1 VA_RESERVE_MIB=64       → exit -1073741819, no counters
#
# The last log line before the crash is `factory.rs:897` — "could not reserve handle
# address space for device #1; reporting no device allocator. The EP still works with host
# memory."  A 64 MiB reservation is too small, the EP correctly declines to publish a
# device allocator and says so, and then the device-memory path is taken anyway.
#
# WHY THIS IS `xfail(strict=True)` AND NOT A RED.  `allocator.rs` and `factory.rs` are
# Tank's and Switch's; the fix is not mine to write and a permanent red in my lane is a
# broken window for four other agents.  Strict xfail is the form that cannot rot: when the
# crash is fixed this reports XPASS(stale expect) — a distinct terminal token in this
# suite's summary — and someone has to come back and delete it.  A skip would report
# nothing forever.
#
# NOTE ON WHAT IS ASSERTED.  The assertion is on the child's EXIT CODE, not on a duration
# and not on a message.  A crash is the one lane observation that survives contention
# unchanged: a loaded machine makes a process slower, not dead.
@pytest.mark.skipif(
    not os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB"),
    reason="ONNXRUNTIME_VULKAN_EP_LIB not set",
)
@pytest.mark.xfail(
    strict=True,
    reason=(
        "ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY=1 with ONNXRUNTIME_EP_VULKAN_VA_RESERVE_MIB=64 "
        "crashes the EP with STATUS_ACCESS_VIOLATION during session creation: the VA "
        "reservation fails, factory.rs:897 declines to publish a device allocator and says "
        "so, and the device-memory path is taken regardless. Owners: Tank (allocator.rs), "
        "Switch (factory.rs). Found by the criterion-12 flag frame on its first run — "
        "neither switch was censused and their interaction had never been exercised."
    ),
)
def test_device_memory_with_small_va_reservation(require_vulkan) -> None:
    """Two uncensused behaviour-selecting switches, armed together, kill the process."""
    out_dir = _REPO_ROOT / "bench" / "results" / "census"
    out_dir.mkdir(parents=True, exist_ok=True)
    selector = os.environ.get("ONNXRUNTIME_EP_VULKAN_DEVICE", "unset")
    counters = out_dir / f"census-counters-dev{selector}-va_crash.json"
    counters.unlink(missing_ok=True)

    env = dict(os.environ)
    env["ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE"] = str(counters)
    env[_ENV_DEVICE_MEMORY] = "1"
    env[_ENV_VA_RESERVE_MIB] = "64"
    env.pop(_ENV_TRACE, None)
    env.pop(_ENV_INJECT, None)
    env.pop(_ENV_CENSUS_MODEL, None)
    env.pop(_ENV_LEDGER_FILE, None)

    with _ambient_guard("device-memory / small-VA crash probe") as guard:
        result = _watchdog.guarded_run(
            [sys.executable, "-X", "importtime",
             str(Path(__file__).resolve()), _CHILD_FLAG, str(counters)],
            guard=guard,
            what="device-memory / small-VA crash probe",
            label="counters_child_va_crash",
            budget_units=_budget_for("counters_child_clean"),
            kind=KIND_MECHANISM,
            env=env,
            cwd=str(HERE),
        )

    # R13: quote the text, never the count.  The exit code IS the text here.
    assert result.returncode == 0, (
        f"{_ENV_DEVICE_MEMORY}=1 with {_ENV_VA_RESERVE_MIB}=64 ended the child with "
        f"exit {result.returncode} "
        f"({'STATUS_ACCESS_VIOLATION' if result.returncode == -1073741819 else 'non-zero'})"
        f" and wrote counters={counters.is_file()}.\n"
        "Either switch alone is clean and VA_RESERVE_MIB=1024 with DEVICE_MEMORY=1 is "
        "clean, so this is the interaction and not either flag.\n"
        f"child log (last 1500 chars):\n"
        f"{((result.stdout or '') + (result.stderr or '')).replace(chr(0), '')[-1500:]}"
    )


# ---------------------------------------------------------------------------
# THE DEVICE-FREE FALSIFIERS FOR MECHANISM 10'S OWN CONTROL
#
# `_observe_device_state_guard` is the census's reading of Link's obligation-8 guard.
# Until 2026-08-04 its planted arm required one token — the one a GPU host produces —
# and on a host with no telemetry producer the SAME input correctly returns
# ERROR(instrument=device_state_producer_absent).  The census then reported that it
# could not observe the mechanism, and every GPU-free lane was red with no path to
# green.  It was reading the host, not the guard.
#
# These arms hold the repair from both ends: the observation must still REFUSE a guard
# that does not fire, that answers the same thing to every input, or that disagrees with
# the host class it is running on.  None of them opens a device.
# ---------------------------------------------------------------------------

_DS_PRODUCER_ABSENT = "ERROR(instrument=device_state_producer_absent)"
_DS_FAIL = "FAIL(condition=STEADY_UNCERTIFIED)"


def _ds_guard_module():
    """The same module object `_observe_device_state_guard` imports."""
    ci_dir = _REPO_ROOT / "ci"
    if str(ci_dir) not in sys.path:
        sys.path.insert(0, str(ci_dir))
    import check_device_state as guard  # noqa: PLC0415

    return guard


def _ds_stub(answers, guard):
    """Return a `main(argv)` that answers by the scan directory's file names.

    `answers` maps a substring of the planted file name to `(exit_code, printed_line)`.
    """

    def main(argv):
        scan = Path(argv[argv.index("--scan") + 1])
        names = " ".join(p.name for p in scan.glob("*.json")) if scan.is_dir() else ""
        for needle, (code, line) in answers.items():
            if needle in names:
                print(f"{guard.LABEL}: {line}")
                return code
        print(f"{guard.LABEL}: PASS")
        return guard.EXIT_PASS

    return main


def test_device_state_observation_fires_on_this_host() -> None:
    """The live arm: on THIS machine, whatever its telemetry, the pair must discriminate.

    This is the arm that was red on every GPU-free CI runner.  It carries no device and
    no skip: a host with a producer and a host without one both have to reach FIRED, and
    the line says which host class it read so the reading travels with its frame.
    """
    line = _observe_device_state_guard()
    assert line.startswith("FIRED"), (
        "the census cannot observe Link's device-state guard on this host.\n"
        f"{line}\n"
        "Before 2026-08-04 this was an ERROR(instrument) on every runner with no GPU "
        "telemetry, because the planted arm required the token only a telemetry host "
        "produces.  If it is red again, read WHICH arm the line names before changing a "
        "threshold."
    )
    assert ("host=producer_present" in line) or ("host=producer_absent" in line), (
        f"the observation must name the host class it ruled under: {line}"
    )


def test_device_state_observation_is_host_class_dependent_and_says_so() -> None:
    """Same input, both host classes, two tokens — and FIRED under each.

    `ONNXRUNTIME_EP_CI_DEVICE_STATE_PRODUCERS=none` is `ci/device_state.py`'s own
    suppression switch, so this reproduces the GPU-free runner on a GPU box without
    touching the machine.  It is the reading that showed the CI red was a property of
    the host and not of the code.
    """
    guard = _ds_guard_module()
    seen = {}
    for label, forced in (("producer_absent", "none"), ("host", None)):
        old = os.environ.get("ONNXRUNTIME_EP_CI_DEVICE_STATE_PRODUCERS")
        if forced is None:
            os.environ.pop("ONNXRUNTIME_EP_CI_DEVICE_STATE_PRODUCERS", None)
        else:
            os.environ["ONNXRUNTIME_EP_CI_DEVICE_STATE_PRODUCERS"] = forced
        try:
            seen[label] = _observe_device_state_guard()
        finally:
            if old is None:
                os.environ.pop("ONNXRUNTIME_EP_CI_DEVICE_STATE_PRODUCERS", None)
            else:
                os.environ["ONNXRUNTIME_EP_CI_DEVICE_STATE_PRODUCERS"] = old

    assert seen["producer_absent"].startswith("FIRED"), (
        "with the producers suppressed — the CI runner's condition, reproduced — the "
        "census must still reach a reading of the guard.\n"
        f"{seen['producer_absent']}"
    )
    assert _DS_PRODUCER_ABSENT in seen["producer_absent"], (
        "with no producer the planted companionless duration must return obligation 8 "
        f"amendment 2's own token: {seen['producer_absent']}"
    )
    assert "no_figure='DEVICE-STATE: PASS'" in seen["producer_absent"], (
        "the host-independent half must be a PASS on a claim carrying no figure — that "
        f"is what makes the pair discriminate with no telemetry: {seen['producer_absent']}"
    )
    assert seen["host"].startswith("FIRED"), seen["host"]
    # Whatever this host is, `guard.ds` decides, and the observation must agree with it.
    producers = list(guard.ds.host_producer_status().get("available") or [])
    expect = _DS_FAIL if producers else _DS_PRODUCER_ABSENT
    assert expect in seen["host"], (
        f"host reports producers={producers or 'none'}, so the planted arm owes "
        f"{expect!r}: {seen['host']}"
    )


def test_device_state_observation_refuses_a_guard_that_never_fires(monkeypatch) -> None:
    """A guard that passes a companionless duration is UNWIRED, not FIRED."""
    guard = _ds_guard_module()
    monkeypatch.setattr(guard, "main", _ds_stub({}, guard))
    line = _observe_device_state_guard()
    assert line.startswith("UNWIRED"), (
        f"a guard that answers PASS to every input must be refused: {line}"
    )
    assert "negative control did not fire" in line, line


def test_device_state_observation_refuses_a_constant_guard(monkeypatch) -> None:
    """One token for every input is a constant, whichever token it is.

    Written expecting the `planted_line == nofigure_line` branch to catch this, and it
    did not: the no-figure arm gets there first and names a better reason.  Which makes
    that branch unreachable once both arms are checked — it is now removed rather than
    left as a comforting line that can never execute, and this arm asserts the refusal
    that the code actually produces.
    """
    guard = _ds_guard_module()
    producers = list(guard.ds.host_producer_status().get("available") or [])
    code = guard.EXIT_FAIL_CONDITION if producers else guard.EXIT_ERROR_INSTRUMENT
    token = _DS_FAIL if producers else _DS_PRODUCER_ABSENT
    monkeypatch.setattr(guard, "main", _ds_stub({".json": (code, token)}, guard))
    line = _observe_device_state_guard()
    assert line.startswith("UNWIRED") and "NO duration" in line, (
        "a guard returning the obliged token for the companionless plant AND for a claim "
        f"with no figure at all is a constant and must be refused: {line}"
    )


def test_device_state_observation_refuses_a_guard_that_disagrees_with_its_host(
    monkeypatch,
) -> None:
    """The host class decides which token is owed; the wrong one is still a finding.

    This is the arm that keeps the repair from being "accept either token".  The host is
    told it has no producer while the guard keeps answering FAIL(condition) — the answer
    a telemetry host owes — and the observation must refuse it.
    """
    guard = _ds_guard_module()
    monkeypatch.setenv("ONNXRUNTIME_EP_CI_DEVICE_STATE_PRODUCERS", "none")
    monkeypatch.setattr(
        guard,
        "main",
        _ds_stub({"planted_lane_claim": (guard.EXIT_FAIL_CONDITION, _DS_FAIL)}, guard),
    )
    line = _observe_device_state_guard()
    assert line.startswith("UNWIRED"), (
        "a guard answering the telemetry-host token on a host with no producer is "
        f"disagreeing with the machine it runs on, and that is not a FIRED: {line}"
    )
    assert "producer_absent host" in line, line


def test_device_state_observation_refuses_a_broken_no_figure_arm(monkeypatch) -> None:
    """If a claim with no duration is not a PASS, the guard answers another question."""
    guard = _ds_guard_module()
    producers = list(guard.ds.host_producer_status().get("available") or [])
    code = guard.EXIT_FAIL_CONDITION if producers else guard.EXIT_ERROR_INSTRUMENT
    token = _DS_FAIL if producers else _DS_PRODUCER_ABSENT
    monkeypatch.setattr(
        guard,
        "main",
        _ds_stub(
            {
                "planted_lane_claim": (code, token),
                "no_figure_lane_claim": (
                    guard.EXIT_FAIL_CONDITION,
                    "FAIL(condition=SOMETHING_ELSE)",
                ),
            },
            guard,
        ),
    )
    line = _observe_device_state_guard()
    assert line.startswith("UNWIRED") and "NO duration" in line, (
        f"a not-PASS on a claim carrying no figure must be refused: {line}"
    )


# ---------------------------------------------------------------------------
# Child entry point.  `python test_wiring_census.py --census-counters-child <path>`
# runs one session with ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE already in the environment
# before onnxruntime loads the EP DLL — which is the only way the JSON-published
# mechanisms are reachable at all on Windows.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == _CHILD_FLAG:
        raise SystemExit(_counters_child_main(sys.argv[2]))
    print(__doc__)
    raise SystemExit(0)
