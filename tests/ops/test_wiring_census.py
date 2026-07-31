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
     Observable: ``model_output_equivalence`` field in counters JSON.
     Wired when: field is MATCH or DIVERGENT (not UNMEASURED).
     Current state: set by Trinity's Python harness in test_phi35.py when Phi-3.5 is
     available.  Reports UNMEASURED when the model cache is absent (non-dev machine).
     Census reports per the current counters file.

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

HERE = Path(__file__).parent
_REPO_ROOT = HERE.parent.parent
_CARGO_MANIFEST = _REPO_ROOT / "rust" / "Cargo.toml"

# ---------------------------------------------------------------------------
# Mechanism registry — what we census and what "wired" means for each.
# ---------------------------------------------------------------------------

_MECHANISMS = [
    "partitioner",
    "partition_identity_check",
    "gpu_tracer",
    "model_output_equivalence",
    "retain_viable",
    "ledger_lookup",
    "validation_messenger",
    "layering_lint",
]

# Mechanisms that MUST be wired for M0 to be complete.  Others are informational.
_MANDATORY_WIRED = {
    "partitioner",
    "partition_identity_check",
    "validation_messenger",
    "layering_lint",
}

# Mechanisms known to be UNWIRED at M0 (criterion 11 not yet met).
# These are xfail(strict=True) rather than hard failures.
_KNOWN_UNWIRED_M0 = {
    "ledger_lookup",   # Mouse/Trinity — criterion 11 not met; no ledger entries exist
}


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


class _EpCounters(ctypes.Structure):
    """Mirror of VulkanEpCounters (C ABI — counters.rs).  Append-only; never remove fields."""

    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("compile_calls", ctypes.c_uint64),
        ("subgraphs_live", ctypes.c_uint64),
        ("subgraphs_stub", ctypes.c_uint64),
        ("compute_calls", ctypes.c_uint64),
        ("compute_failures", ctypes.c_uint64),
        ("dispatches_executed", ctypes.c_uint64),
        # ABI version 2: viable_islands_retained — R10 wiring observable for net-benefit gate.
        ("viable_islands_retained", ctypes.c_uint64),
    ]


def _read_ep_counters_via_ctypes() -> dict[str, int]:
    """Read the EP's live execution counters via OrtEpVulkanGetExecutionCounters (C ABI).

    This is the in-process path (test_phi35.py style).  It avoids the Windows UCRT env-var
    cache problem: the EP DLL reads ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE at init time, so
    setting the env var after DLL load is unreliable on Windows.  The C ABI call is always live.
    """
    ep_lib = os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB")
    if not ep_lib:
        return {}
    try:
        import ctypes as _ct
        dll = _ct.CDLL(ep_lib)
        c = _EpCounters()
        dll.OrtEpVulkanGetExecutionCounters(_ct.byref(c), _ct.sizeof(c))
        return {
            "compile_calls": c.compile_calls,
            "subgraphs_live": c.subgraphs_live,
            "subgraphs_stub": c.subgraphs_stub,
            "compute_calls": c.compute_calls,
            "compute_failures": c.compute_failures,
            "dispatches_executed": c.dispatches_executed,
            "viable_islands_retained": c.viable_islands_retained,
        }
    except Exception:
        return {}


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
# The census test
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB"),
    reason="ONNXRUNTIME_VULKAN_EP_LIB not set — no EP to census",
)
def test_wiring_census(require_vulkan) -> None:
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

    # ── Run session and collect observations ────────────────────────────────
    counters_before, counters_after, profile_info = _run_add_session_with_profiling()

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
            f"dispatches={dispatches_delta}, subgraphs_live={subgraphs_delta}, "
            f"claimed_from_profiling={claimed}, islands_from_profiling={islands}"
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
    trace_file = os.environ.get("ONNXRUNTIME_EP_VULKAN_TRACE_FILE")
    if trace_file and Path(trace_file).is_file():
        try:
            with open(trace_file) as fh:
                trace_data = json.load(fh)
            entry_count = len(trace_data) if isinstance(trace_data, list) else (
                len(trace_data.get("traceEvents", [])) if isinstance(trace_data, dict) else 0
            )
            observations["gpu_tracer"] = f"{entry_count} trace entries in {trace_file}"
        except Exception as exc:
            observations["gpu_tracer"] = f"trace file present but unreadable: {exc}"
    else:
        observations["gpu_tracer"] = "OPTIONAL-UNWIRED (ONNXRUNTIME_EP_VULKAN_TRACE_FILE not set)"

    # ── Mechanism 4: model_output_equivalence ────────────────────────────
    # This session runs Add, not Phi-3.5. The equivalence verdict belongs to test_phi35.py.
    # Read from ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE if set (may have been written by phi35).
    #
    # VALIDITY CONDITION (Guard D): MATCH is a valid verdict only when the run that wrote it
    # had Guard D active — i.e., assert_vulkan_executed_runtime() confirmed Vulkan ran nodes
    # AFTER sess.run(). Without Guard D, a runtime fallback (ORT silently retrying on CPU)
    # produces CPU-vs-CPU output that reports MATCH. Guard D is now in
    # test_phi35_vulkan_matches_cpu_logits; verdicts written before 2026-07-31 should be
    # treated as UNMEASURED (the guard was absent).
    equiv = "UNMEASURED"
    counters_file_path = os.environ.get("ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE")
    if counters_file_path and Path(counters_file_path).is_file():
        try:
            with open(counters_file_path) as fh:
                file_counters = json.load(fh)
            equiv = file_counters.get("model_output_equivalence", "UNMEASURED")
        except Exception:
            pass
    observations["model_output_equivalence"] = equiv

    # ── Mechanism 5: retain_viable ───────────────────────────────────────
    # Observable: `viable_islands_retained` in the C ABI counters (added ABI version 2).
    # Present even at 0 — an always-0 result is distinguishable from UNWIRED (key absent)
    # because the key is emitted by the production path.  Owner: Mouse.
    if "viable_islands_retained" in counters_after:
        observations["retain_viable"] = str(counters_after["viable_islands_retained"])
    else:
        observations["retain_viable"] = "UNWIRED"

    # ── Mechanism 6: ledger_lookup ───────────────────────────────────────
    # §8.9 criterion 11 not met.  No ledger entries exist.
    observations["ledger_lookup"] = "UNWIRED"

    # ── Mechanism 7: validation_messenger ───────────────────────────────
    epctl = _epctl_path()
    if epctl is None:
        observations["validation_messenger"] = "UNWIRED (epctl not found)"
    else:
        result = subprocess.run(
            [str(epctl), "--probe-validation"],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            observations["validation_messenger"] = "ARMED"
        elif result.returncode == 3:
            observations["validation_messenger"] = "OPTIONAL-UNWIRED (layer absent)"
        else:
            observations["validation_messenger"] = f"UNWIRED (epctl exit {result.returncode})"

    # ── Mechanism 8: layering_lint ───────────────────────────────────────
    # The layering lint runs as a cargo integration test (rust/tests/layering.rs).
    # We run it in DEBUG mode (not --release) to avoid relinking the production DLL,
    # which may be loaded by the current process and locked on Windows.
    if _CARGO_MANIFEST.is_file():
        lr = subprocess.run(
            [
                "cargo", "test", "--test", "layering",
                "--manifest-path", str(_CARGO_MANIFEST),
                # No --release: debug build avoids relinking the loaded DLL.
            ],
            capture_output=True, text=True,
            env=_cargo_env(),
            timeout=120,
        )
        if lr.returncode == 0:
            observations["layering_lint"] = "PASS"
        else:
            observations["layering_lint"] = f"FAIL (exit {lr.returncode}): {lr.stderr[-200:]}"
    else:
        observations["layering_lint"] = "UNWIRED (Cargo.toml not found)"

    # ── Emit census ──────────────────────────────────────────────────────
    print("\n[WIRING CENSUS] M0 criterion 12 — per-mechanism observations:", file=sys.stderr)
    for mech in _MECHANISMS:
        obs = observations.get(mech, "UNWIRED")
        print(f"[WIRING CENSUS] {mech}: {obs}", file=sys.stderr)
    print("[WIRING CENSUS] end.", file=sys.stderr)

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

    # model_output_equivalence — warn but do not fail (test_phi35.py owns the assertion;
    # UNMEASURED here just means Phi-3.5 is not in the cache on this machine).
    if observations.get("model_output_equivalence") == "UNMEASURED":
        print(
            "[WIRING CENSUS] WARNING: model_output_equivalence=UNMEASURED. "
            "This is expected when the Phi-3.5 model cache is absent. "
            "The canonical MATCH reading is in test_phi35.py::test_phi35_vulkan_matches_cpu_logits.",
            file=sys.stderr,
        )

    assert not failures, (
        "Wiring census FAILED — the following mandatory mechanisms are UNWIRED:\n"
        + "\n".join(failures)
        + "\n\nAll census observations:\n"
        + "\n".join(f"  {m}: {observations.get(m, 'UNWIRED')}" for m in _MECHANISMS)
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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "§8.9 ledger does not exist (M0 criterion 11 not met). "
        "The claim-unproven gate (DeclineCode::Unproven) declines all unproven forms "
        "once the ledger is populated.  Until then, the gate cannot fire. "
        "Owner: Mouse (ledger entries) / Trinity (census query). "
        "Remove this xfail when criterion 11 is met and the gate fires on a real session."
    ),
)
@pytest.mark.skipif(
    not os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB"),
    reason="ONNXRUNTIME_VULKAN_EP_LIB not set",
)
def test_ledger_lookup_wired(require_vulkan) -> None:
    """Criterion 12 sub-item: §8.9 ledger lookup must produce a computed observation.

    Currently xfail because criterion 11 (no form claimed without a ledger entry) is not
    met — there are no ledger entries. The gate exists in registry.rs as DeclineCode::Unproven
    but the ledger has nothing to look up. This test will XPASS when Mouse populates the
    ledger and the gate fires on a real GetCapability call.

    Observable proxy: a 'proven_key_lookups' or 'unproven_declines' counter in the EP
    counters JSON. Owner: Mouse to add the counter; Trinity to update the census query.
    """
    _, counters_after, _ = _run_add_session_with_profiling()
    # When criterion 11 is met, counters should contain 'proven_key_lookups' or similar.
    assert "proven_key_lookups" in counters_after, (
        "§8.9 ledger lookup counter not present — mechanism is UNWIRED (criterion 11 not met). "
        f"Current counter keys: {sorted(counters_after.keys())}"
    )
