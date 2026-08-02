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

HERE = Path(__file__).parent
_REPO_ROOT = HERE.parent.parent
_CARGO_MANIFEST = _REPO_ROOT / "rust" / "Cargo.toml"

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
        # ABI version 3: the §8.9 proof ledger — R10 wiring observable for criterion 11.
        ("proven_key_lookups", ctypes.c_uint64),
        ("ledger_hits", ctypes.c_uint64),
        ("unproven_declines", ctypes.c_uint64),
        ("ledger_entries", ctypes.c_uint64),
        ("unproven_forms_claimed", ctypes.c_uint64),
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
            "proven_key_lookups": c.proven_key_lookups,
            "ledger_hits": c.ledger_hits,
            "unproven_declines": c.unproven_declines,
            "ledger_entries": c.ledger_entries,
            "unproven_forms_claimed": c.unproven_forms_claimed,
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

_CENSUS_OUT = Path(__file__).parent.parent.parent / "bench" / "results" / "census"
_CENSUS_TRACE_PATH = _CENSUS_OUT / (
    f"census-trace-dev{os.environ.get('ONNXRUNTIME_EP_VULKAN_DEVICE', 'unset')}.json"
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


def _counters_child_main(counters_path: str) -> int:
    """Child entry point: run one session with the counters file armed, then exit."""
    lib = os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB")
    if not lib:
        print("[census-child] ONNXRUNTIME_VULKAN_EP_LIB unset", file=sys.stderr)
        return 3
    try:
        ort.register_execution_provider_library(m.EP_NAME, str(Path(lib).resolve()))
    except Exception as exc:  # noqa: BLE001
        if "already registered" not in str(exc):
            print(f"[census-child] registration failed: {exc}", file=sys.stderr)
            return 3
    opts = ort.SessionOptions()
    opts.log_severity_level = 2
    sess = ort.InferenceSession(_chain_model(), opts, providers=m.EP_PROVIDERS)
    feeds = {
        "a": np.ones((4, 4), dtype=np.float32),
        "b": np.full((4, 4), 2.0, dtype=np.float32),
    }
    try:
        sess.run(None, feeds)
    except Exception as exc:  # noqa: BLE001
        # An injected Compute failure is expected to surface here; the artifact the census
        # reads is the counters file, which `record_broken_commitment` writes at the
        # instant of the event and not at teardown.
        print(f"[census-child] run raised: {type(exc).__name__}: {exc}", file=sys.stderr)
    del sess
    return 0 if Path(counters_path).is_file() else 3


def _run_counters_child(*, inject: bool, tag: str) -> tuple[dict, str]:
    """Run one polarity in a fresh process; return (counters doc, combined child log)."""
    out_dir = _REPO_ROOT / "bench" / "results" / "census"
    out_dir.mkdir(parents=True, exist_ok=True)
    selector = os.environ.get("ONNXRUNTIME_EP_VULKAN_DEVICE", "unset")
    counters = out_dir / f"census-counters-dev{selector}-{tag}.json"
    counters.unlink(missing_ok=True)

    env = dict(os.environ)
    env["ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE"] = str(counters)
    if not inject:
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

    result = _verdict.run_subprocess_checked(
        [sys.executable, str(Path(__file__).resolve()), _CHILD_FLAG, str(counters)],
        what=f"census counters child ({tag})",
        quiet_seconds=180,
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

    ours = _REPO_ROOT / "bench" / "results" / "census"
    if not ours.is_dir():
        return (
            "UNWIRED (no bench/results/census directory for the guard to read on this "
            "run — the guard had no subject, which is not a clean lane)"
        )
    ours_code, ours_line = _run(ours)

    if planted_code == guard.EXIT_ERROR_INSTRUMENT:
        return f"INSTRUMENT-ERROR (planted control: {planted_line})"
    if planted_code != guard.EXIT_FAIL_CONDITION:
        return (
            f"UNWIRED (the planted companionless duration returned exit {planted_code} "
            f"[{planted_line!r}] instead of FAIL(condition) — the negative control did "
            "not fire, so this guard's green readings are unverified)"
        )
    if planted_line == ours_line:
        return (
            f"UNWIRED (constant: the guard returned the same line {ours_line!r} for a "
            "planted violation and for this run's own evidence)"
        )
    return (
        f"FIRED planted_exit={planted_code} planted={planted_line!r} | "
        f"lane_exit={ours_code} lane={ours_line!r}"
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

    # ── The two counters-file polarities, taken once and read by three mechanisms ────
    #
    # R10's falsifier is "an artifact X produced whose content varies with X's input", so
    # every JSON-published mechanism below is read from BOTH polarities and the census
    # records the pair.  A line that reads the same in both is reported as such and is not
    # allowed to pass as an observation.
    clean_doc, clean_log = _run_counters_child(inject=False, tag="clean")
    inject_doc, inject_log = _run_counters_child(inject=True, tag="inject")

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
            trace_data = json.loads(_CENSUS_TRACE_PATH.read_text(encoding="utf-8"))
            events = (
                trace_data if isinstance(trace_data, list)
                else trace_data.get("traceEvents", []) if isinstance(trace_data, dict)
                else []
            )
            phases = sorted({e.get("ph") for e in events if isinstance(e, dict)})
            names = sorted({e.get("name") for e in events if isinstance(e, dict)})
            if not events:
                observations["gpu_tracer"] = (
                    f"UNWIRED (armed via {_ENV_TRACE}={_CENSUS_TRACE_PATH.name} and the "
                    "file it wrote contains no events)"
                )
            else:
                observations["gpu_tracer"] = (
                    f"{len(events)} trace event(s), phases={phases}, "
                    f"distinct_names={len(names)} (armed by this census via {_ENV_TRACE}; "
                    "no duration from this file is quoted — §10.0 obligation 8)"
                )
        except Exception as exc:  # noqa: BLE001
            observations["gpu_tracer"] = f"INSTRUMENT-ERROR (trace file unreadable: {exc})"
    else:
        observations["gpu_tracer"] = (
            f"UNWIRED ({_ENV_TRACE} was set to {_CENSUS_TRACE_PATH} for the counters "
            "child and no trace file was written on EP teardown)"
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
        observations["retain_viable"] = str(counters_after["viable_islands_retained"])
    else:
        observations["retain_viable"] = "UNWIRED"

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
    if gate_token is None:
        observations["net_benefit_gate"] = (
            "UNWIRED (no net_benefit_gate field in the counters artifact — the gate did "
            "not publish, which is not the same as the gate rejecting everything)"
        )
    elif gate_token == "UNWIRED":
        observations["net_benefit_gate"] = (
            f"UNWIRED (clusters_seen={clean_doc.get('net_benefit_gate_clusters_seen')!r} "
            "— no cluster reached the decision point in this run)"
        )
    else:
        observations["net_benefit_gate"] = (
            f"{gate_token} clusters_seen={clean_doc.get('net_benefit_gate_clusters_seen')!r} "
            f"evaluations={clean_doc.get('net_benefit_gate_evaluations')!r} "
            f"bypasses={clean_doc.get('net_benefit_gate_bypasses')!r} "
            f"sole_island_overrides={clean_doc.get('net_benefit_sole_island_overrides')!r} "
            f"viable_islands_retained={clean_doc.get('viable_islands_retained')!r} "
            f"(retained is typed: 'UNWIRED'/'UNOBSERVABLE'/int — a type cannot be forged "
            f"by an increment)"
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
            f"unproven_forms_enabled={clean_doc.get('unproven_forms_enabled')!r} "
            f"(hits is typed: 'UNWIRED'/'UNOBSERVABLE'/int)"
        )

    # ── Mechanism 7: validation_messenger ───────────────────────────────
    # R13: this mechanism shells out.  A subprocess that hangs is an ERROR(instrument) —
    # the census reached no observation — and must never be scored as "UNWIRED", which is
    # a finding.  `run_subprocess_checked` raises InstrumentError on a timeout, so it is
    # caught here and recorded as the third state rather than collapsed into the second.
    # The budget is contention-inflated: this suite has been measured at 4.4x under four
    # concurrent agents, and this test had to be `--ignore`d for exactly this reason.
    epctl = _epctl_path()
    if epctl is None:
        observations["validation_messenger"] = "UNWIRED (epctl not found)"
    else:
        try:
            result = _verdict.run_subprocess_checked(
                [str(epctl), "--probe-validation"],
                what="census validation probe",
                quiet_seconds=20,
            )
        except _verdict.InstrumentError as exc:
            observations["validation_messenger"] = (
                f"INSTRUMENT-ERROR ({str(exc).splitlines()[0]})"
            )
        else:
            state, reason = _verdict.classify_validation_probe(
                result.returncode, (result.stdout or "") + (result.stderr or "")
            )
            if state == _verdict.VALIDATION_ARMED:
                observations["validation_messenger"] = "ARMED"
            elif state == _verdict.VALIDATION_PROBE_ERROR:
                observations["validation_messenger"] = f"INSTRUMENT-ERROR ({reason})"
            else:
                observations["validation_messenger"] = f"OPTIONAL-UNWIRED ({reason})"

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
    if _CARGO_MANIFEST.is_file():
        try:
            lr = _verdict.run_subprocess_checked(
                [
                    "cargo", "test", "--test", "layering",
                    "--manifest-path", str(_CARGO_MANIFEST),
                    # No --release: debug build avoids relinking the loaded DLL.
                ],
                what="census layering lint",
                quiet_seconds=240,
                env=_cargo_env(),
            )
        except _verdict.InstrumentError as exc:
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
            f"fault_injection={inject_doc.get('fault_injection')!r} "
            f"ort_sink_warn_lines={len(ort_warn_lines)} | "
            f"clean: channel={clean_channel!r} broken_commitments={clean_broken!r} "
            f"fault_injection={clean_doc.get('fault_injection')!r}"
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

    # ── Emit census ──────────────────────────────────────────────────────
    print("\n[WIRING CENSUS] M0 criterion 12 — per-mechanism observations:", file=sys.stderr)
    for mech in _MECHANISMS:
        obs = observations.get(mech, "UNWIRED")
        print(f"[WIRING CENSUS] {mech}: {obs}", file=sys.stderr)
    print("[WIRING CENSUS] end.", file=sys.stderr)

    # The census is itself an artifact, not just a print.  Criterion 12's own falsifier is
    # a file whose content varies with the run that produced it.
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
                    "not two."
                ),
                "no_duration_quoted": (
                    "§10.0 obligation 8 — every line above is a count, a token or a "
                    "byte volume.  The tracer line reports how many events it wrote and "
                    "quotes none of their durations."
                ),
                "observations": {mm: observations.get(mm, "UNWIRED") for mm in _MECHANISMS},
                "unwired": [
                    mm for mm in _MECHANISMS
                    if observations.get(mm, "UNWIRED").startswith("UNWIRED")
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[WIRING CENSUS] wrote {census_path}", file=sys.stderr)

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
            "lines above is evidence that a mechanism is unwired. The usual cause on this "
            "host is machine contention (measured 4.4x with four concurrent agents); the "
            "budgets here are already inflated for it, and "
            "$ONNXRUNTIME_EP_VULKAN_TIMEOUT_SCALE raises them further.\n\n"
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
