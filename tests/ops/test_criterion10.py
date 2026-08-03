"""M0 criterion 10 — three consecutive *attributed* MATCH runs, as a test in the lane.

WHY THIS FILE EXISTS
====================
DESIGN.md §10 M0 criterion 10 was reopened on 2026-07-31T07:45:10-07:00 and Morpheus
stated its price in advance:

  > when one session produces three consecutive attributed ``MATCH`` runs on this
  > artifact, this row closes the same day, with no new conditions.

That evidence has been produced by hand, with a throwaway script, on both devices:

    selector 0 (NVIDIA RTX 4060)          selector 1 (Intel Iris Xe)
       3  VulkanExecutionProvider            3  VulkanExecutionProvider
      30  CPUExecutionProvider              30  CPUExecutionProvider
      cross-run identical (all 65): True     cross-run identical (all 65): True
      argmax 30751 (matches CPU)             argmax 30751 (matches CPU)

**Per R10 that evidence does not exist, because it is not in the lane.** A record a human
produced with a script that no longer exists is a code reading with extra steps: nobody
can re-derive it, nobody is told when it stops being true, and its content does not vary
with its input in any way an auditor can check. This file is that same measurement,
expressed as a mechanism that runs every time the suite runs, on both devices, and fails
loudly when the property stops holding.

WHAT IT REPORTS NOW (main @ efbf18c, 2026-08-01, both devices, identical):

    selector 0 (NVIDIA RTX 4060)          selector 1 (Intel Iris Xe)
       3  VulkanExecutionProvider            3  VulkanExecutionProvider
      24  CPUExecutionProvider              24  CPUExecutionProvider
      argmax 30751 (== CPU) x3               argmax 30751 (== CPU) x3
      cross-run identical (all 65): True     cross-run identical (all 65): True
      series verdict: MATCH                  series verdict: MATCH

The own-provider count is unchanged at 3 and the CPU count fell 30 -> 24 because Mouse
claimed ``SimplifiedLayerNormalization`` and ``Gather`` in between: six CPU node events
per three runs moved to the Vulkan island. **The number that fell is the number that got
better.** That is precisely why the assertions below are on the island count and the
multiple, and never on a literal CPU count — a suite that pinned ``30`` would have gone
red at the moment the EP started doing more work.

THE NAMING TRAP, STATED WHERE THE NUMBER IS PRODUCED (R11)
==========================================================
``3`` is **three fused-island executions — one per run — not three graph nodes.** ORT
emits one ``cat == "Node"`` profiling event per fused-island execution, and Phi-3.5's
Vulkan partition is a *single* island covering 355 of 363 graph nodes (2026-08-01; 8
declines). So a perfect three-run series reports exactly ``3``, and a reader who meets
that number beside a
363-node model will read a catastrophe where the record says success. Every emission of
the count in this file goes through :meth:`AttributedRunSeries.describe`, which carries
the definition with the number, and the artifact written to ``bench/results/`` carries
``counts_what`` for the same reason.

WHAT "CONSECUTIVE" MEANS FOR A SESSION-SCOPED INSTRUMENT
========================================================
ORT writes one profile per session and ``end_profiling()`` cannot be restarted, so the
attribution is necessarily session-scoped. The series therefore requires ``own_count`` to
be a **positive multiple of the run count**: a run that fell back to CPU part-way through
the series drops the total below the run count, and a series in which one run executed
two islands while another executed none fails the multiple test. Those two conditions are
what "three consecutive attributed runs" buys from a session-scoped witness, and they are
stated here rather than assumed.

R13 IN THIS FILE
================
Three terminal states, spelled three ways:

  - ``PASS``               — the series closed.
  - ``FAIL(condition)``    — ``AssertionError`` from
    :meth:`AttributedRunSeries.assert_closes_criterion_10`, which quotes the observation.
  - ``ERROR(instrument)``  — :class:`_verdict.InstrumentError`, raised only *before* an
    observation exists (missing/corrupt profile, missing model). Never a detection.
"""

from __future__ import annotations

import json
import os
import pathlib

import numpy as np
import onnxruntime as ort
import pytest

import _models as m
import _kv_depth
from test_phi35 import _ONNX_FILE, _build_phi35_feeds

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_RESULTS_DIR = _REPO_ROOT / "bench" / "results"

#: Morpheus's stated condition. Not a knob: raising it would be adding a condition to a
#: criterion whose closure price was fixed in advance, which is the fault the reopening
#: was written to avoid.
REQUIRED_RUNS = 3
# R12: a counter whose event cannot occur in its frame reports UNOBSERVABLE, never 0.
# An empty ULP curve means no output was comparable, not that the residual was zero.
_ULP_UNOBSERVABLE = "UNOBSERVABLE"

#: Tokens for **which route the 65 outputs took out of the fused island**.  Added
#: 2026-08-02T23:xx after Switch's `872d739` made a directly-written device buffer
#: authoritative: the same 65 tensors can now leave the island by two different paths, and
#: a record that does not say which one it measured describes a run nobody can identify.
ROUTE_HOST_STAGING = "HOST_STAGING"
ROUTE_DEVICE_AUTHORITATIVE = "DEVICE_AUTHORITATIVE"
ROUTE_MIXED = "MIXED"
#: R12: a property whose witness was never armed is UNOBSERVABLE, never "the default".
ROUTE_UNOBSERVABLE = "UNOBSERVABLE"


def _read_counters_doc(counters_path: "str | None") -> dict | None:
    if not counters_path or not os.path.exists(str(counters_path)):
        return None
    try:
        doc = json.loads(pathlib.Path(counters_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return doc if isinstance(doc, dict) else None


def device_name_from_run(counters: dict | None) -> str:
    """The physical device this run used, **as the run itself reported it**.

    ``ONNXRUNTIME_EP_VULKAN_DEVICE`` is a *request*.  The two are demonstrably not the
    identity on this box — selector 0 reports ``1=NVIDIA GeForce RTX 4060 Laptop GPU`` and
    selector 1 reports ``0=Intel(R) Iris(R) Xe Graphics``, because the allocator's
    enumeration index is not the selector — and this project has already published a set
    of results with the two vendor labels swapped.  Same rule as
    ``test_validation_phi35.device_name``; kept as a separate reader here because this one
    must never raise: a criterion-10 record is written on the failing path too, and an
    instrument outage while *labelling* a reading must not destroy the reading.
    """
    raw = (counters or {}).get("alloc_device_frame_session_devices")
    if not isinstance(raw, str) or "=" not in raw:
        return ROUTE_UNOBSERVABLE
    return raw.split("=", 1)[1].strip()


def kv_writeback_route(counters: dict | None) -> dict:
    """Which path the outputs took out of the fused island, with the numbers behind it.

    Criterion 10 says every output agrees with the CPU oracle.  Since `872d739` there are
    **two** ways an output can reach ORT — the host staging block, and a device buffer the
    dispatch wrote and which is then marked authoritative — and the criterion is a claim
    about the run, not about the code.  A run down one route says nothing about the other,
    so the route belongs in the record beside the verdict.

    The route is read off the counters the run emitted, never off the environment
    variable that requested it: ``ONNXRUNTIME_EP_VULKAN_BIND_OUTPUTS=1`` is a request, and
    a request that the EP declined (no VkBuffer for the span, so the bind is refused and
    unbound) would otherwise be recorded as a route that was never taken.
    """
    if counters is None:
        return {"route": ROUTE_UNOBSERVABLE, "why": "no counters file was armed for this run"}
    bound = counters.get("outputs_device_bound")
    host = counters.get("outputs_host_resident")
    if not isinstance(bound, (int, float)) or not isinstance(host, (int, float)):
        return {
            "route": ROUTE_UNOBSERVABLE,
            "why": "the counters file does not carry outputs_device_bound/outputs_host_resident",
        }
    bound, host = int(bound), int(host)
    if bound and host:
        route = ROUTE_MIXED
    elif bound:
        route = ROUTE_DEVICE_AUTHORITATIVE
    else:
        route = ROUTE_HOST_STAGING
    return {
        "route": route,
        "outputs_device_bound": bound,
        "outputs_host_resident": host,
        "outputs_device_resident": counters.get("outputs_device_resident"),
        # Switch split this from `alloc_device_authoritative_spans` because two producers
        # with different definitions were being summed into one counter.  Grants are the
        # dynamic sense: one per span a dispatch wrote and which was then made
        # authoritative.  Absent on any binary built before `872d739`.
        "alloc_device_authority_grants": counters.get("alloc_device_authority_grants"),
        "alloc_device_downloads": counters.get("alloc_device_downloads"),
        "alloc_device_download_bytes": counters.get("alloc_device_download_bytes"),
        "alloc_device_frame": counters.get("alloc_device_frame"),
        "why": (
            "read off the counters the run emitted, not off the env var that requested it: "
            "a bind the EP declined would otherwise be recorded as a route that was taken"
        ),
    }


@pytest.fixture(scope="module")
def phi35_model_path() -> pathlib.Path:
    if not _ONNX_FILE.exists():
        pytest.skip(
            f"Phi-3.5 model not found at {_ONNX_FILE}. Criterion 10 is measured on the "
            "real artifact at producer-at-version; there is no synthetic substitute."
        )
    return _ONNX_FILE


def _compare_run_to_cpu(
    vk_out: list[np.ndarray],
    cpu_out: list[np.ndarray],
) -> tuple[str, dict]:
    """Compare one VulkanEP run against the CPU oracle; return (comparison outcome, facts).

    Returns a **comparison outcome** — ``AGREE`` / ``DISAGREE`` — never a verdict. The
    verdict is derived downstream from this plus the execution attribution, which is the
    whole point of §10.0's third amendment: a comparison of tensors cannot, on its own,
    say ``MATCH``.
    """
    logits_vk = vk_out[0].astype(np.float32)
    logits_cpu = cpu_out[0].astype(np.float32)
    flat_vk = logits_vk.reshape(-1, logits_vk.shape[-1])
    flat_cpu = logits_cpu.reshape(-1, logits_cpu.shape[-1])
    argmax_vk = int(flat_vk.argmax(-1)[0])
    argmax_cpu = int(flat_cpu.argmax(-1)[0])
    top10_vk = set(np.argsort(-flat_vk[0])[:10].tolist())
    top10_cpu = set(np.argsort(-flat_cpu[0])[:10].tolist())
    overlap = len(top10_vk & top10_cpu)
    max_abs = float(np.abs(logits_vk).max())
    facts = {
        "argmax_vk": argmax_vk,
        "argmax_cpu": argmax_cpu,
        "top10_overlap": overlap,
        "vk_max_abs_logit": max_abs,
        # Named for its extent, not for its quantity. This was `max_abs_diff`, sitting in a
        # dict beside `outputs_compared: 65`, and it was read into the criteria table as a
        # diff over sixty-five outputs when it covers exactly one. Renaming is the fix that
        # makes the misreading impossible rather than merely corrected (R11: a
        # measurement's name is not its definition, so the name had better carry it).
        "logits_max_abs_diff": float(np.abs(logits_vk - logits_cpu).max()),
        # Declined as reassurance on arithmetic, not on scepticism (Morpheus, 2026-08-02).
        # `argmax_vk == argmax_cpu` and a top-10 overlap of 10 are both statements about
        # **one token**.  N=1 is not a stated N, and this project has read an N=1
        # agreement as a model-wide agreement before.  The numbers stay because they are
        # cheap and a *dis*agreement here would be informative; the label stops them being
        # quoted as evidence of scale.
        "argmax_sample_size": 1,
        "argmax_is_evidence_of_scale": False,
        "argmax_caveat": (
            "argmax and top10_overlap describe a single token position (N=1); they can "
            "falsify agreement but cannot establish it over the sequence"
        ),
    }
    agree = argmax_vk == argmax_cpu and overlap >= 5 and max_abs > 0.1
    return (m.COMPARISON_AGREE if agree else m.COMPARISON_DISAGREE), facts


@pytest.mark.slow
def test_criterion_10_three_consecutive_attributed_match(
    phi35_model_path: pathlib.Path,
    require_vulkan,
    tmp_path: pathlib.Path,
) -> None:
    """One session, three consecutive runs, every one compared and the series attributed.

    This is the criterion-10 record. It asserts, in this order:

      1. the EP is in the session at create time (Guard A — necessary, never sufficient);
      2. three consecutive runs of the *same* session on identical feeds are bit-identical
         to one another across all 65 outputs (an output that differs between runs was
         never written — a memory hazard, not an arithmetic result, and not a MATCH);
      3. each run agrees with a CPU-only run of the same artifact on argmax and top-10;
      4. **the series is attributed** — ORT's own profiler reports a positive multiple of
         three ``VulkanExecutionProvider`` fused-island executions;
      5. the derived series verdict is ``MATCH``, which by construction it cannot be if
         (4) failed: it would be ``UNATTRIBUTED``.
    """
    device_index = os.environ.get("ONNXRUNTIME_EP_VULKAN_DEVICE", "0")
    feeds = _build_phi35_feeds()

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.enable_profiling = True
    opts.profile_file_prefix = str(tmp_path / f"criterion10_dev{device_index}")

    # The claim log is armed before session creation because it is written during
    # GetCapability.  It is the second source for per-output coverage, used only where it
    # accuses us: an ancestor we did not claim is not ours.  The DLL caches env vars at
    # load on Windows, but CLAIM_LOG is read per session, so setting it here is enough —
    # and the join count is asserted downstream rather than assumed.
    claim_log_path = tmp_path / f"criterion10_claims_dev{device_index}.jsonl"
    os.environ["ONNXRUNTIME_EP_VULKAN_CLAIM_LOG"] = str(claim_log_path)
    try:
        vk_sess = ort.InferenceSession(str(phi35_model_path), opts, providers=m.EP_PROVIDERS)
    finally:
        os.environ.pop("ONNXRUNTIME_EP_VULKAN_CLAIM_LOG", None)
    counters_path = os.environ.get("ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE")

    if m.EP_NAME not in vk_sess.get_providers():
        m.write_unmeasured_verdict(
            counters_path,
            f"{m.EP_NAME} absent from get_providers(); criterion-10 series not run.",
            device_index=device_index,
            artifact=str(phi35_model_path),
        )
        pytest.fail(
            f"[Device {device_index}] {m.EP_NAME} not registered: {vk_sess.get_providers()}. "
            "A criterion-10 series measured without the EP in the session is CPU-vs-CPU."
        )

    vk_runs = m.run_session_n_times(vk_sess, feeds, REQUIRED_RUNS)

    # Attribution first, and unconditionally: the verdict below cannot be built without it.
    profile_path = vk_sess.end_profiling()
    try:
        attribution = m.attribution_with_coverage_from_profile(
            profile_path, str(phi35_model_path), claim_log=claim_log_path
        )
    except m.InstrumentError as exc:
        # ERROR(instrument), R13. No verdict is written: an instrument outage is not a
        # detection and must not be recorded as one.
        raise m.InstrumentError(
            f"[Device {device_index}] criterion-10 attribution instrument failure "
            f"(fix the harness, not the EP): {exc}"
        ) from exc
    _counters_value, _counters_reason = m.read_counters_witness(counters_path)
    attribution = attribution.with_counters_witness(_counters_value, reason=_counters_reason)

    # CPU oracle: a CPU-only run of the same artifact (§10.0 point 4 — not a golden file).
    cpu_opts = ort.SessionOptions()
    cpu_opts.log_severity_level = 3
    cpu_sess = ort.InferenceSession(
        str(phi35_model_path), cpu_opts, providers=["CPUExecutionProvider"]
    )
    cpu_out = cpu_sess.run(None, feeds)

    # Cross-run identity across all outputs. A run that differs from run 1 contains an
    # output nobody wrote; that run did not agree with anything and is DISAGREE.
    #
    # And, since 2026-08-02, the CPU oracle over **all** outputs rather than the logits
    # alone. Cross-run identity proves determinism; only the oracle proves correctness, and
    # a deterministically wrong KV write passes the first while failing nothing. The two
    # counts are recorded under two names because they were once read off one:
    # `outputs_compared: 65` counted cross-run comparisons and was quoted as sixty-five
    # oracle comparisons.
    comparisons: list[str] = []
    per_run_facts: list[dict] = []
    cross_run_report: list[str] = []
    for i, run in enumerate(vk_runs):
        identical, differing = m.outputs_bit_equal(vk_runs[0], run)
        outcome, facts = _compare_run_to_cpu(run, cpu_out)
        oracle_outcome, oracle_facts = m.compare_all_outputs_to_cpu(run, cpu_out)
        facts.update(oracle_facts)
        # The unit, per output, in depth order.  §10.0.4 prefers the ratio, and `atol` is
        # an absolute bound applied to tensors whose scale grows with layer depth — so the
        # absolute residual rises for a *correct* implementation and the curve is a plot of
        # magnitude, not of error.  ULPs denominate the residual in the representation's
        # own spacing, which is the quantity that should be flat if nothing is wrong.
        #
        # `median_ulp_diff` is the headline and `max_ulp_diff` is not: a cancellation
        # element inflates the ULP max by exactly the mechanism that inflates
        # `max_rel_diff`, so promoting one max over another would have reinstated the
        # artefact in a new unit (R11).  All four are recorded so a real step cannot hide
        # behind a robust average.
        facts["ulp_curve"] = [
            {
                "output_index": e.get("index", j),
                "name": e.get("name"),
                "median_ulp_diff": e.get("median_ulp_diff"),
                "p99_ulp_diff": e.get("p99_ulp_diff"),
                "max_ulp_diff": e.get("max_ulp_diff"),
                "ulp_cancellation_elements": e.get("ulp_cancellation_elements"),
                "max_abs_diff": e.get("max_abs_diff"),
            }
            for j, e in enumerate(oracle_facts.get("per_output", []))
        ]
        _meds = [
            c["median_ulp_diff"]
            for c in facts["ulp_curve"]
            if c["median_ulp_diff"] is not None
        ]
        facts["ulp_curve_median_over_outputs"] = (
            float(np.median(_meds)) if _meds else _ULP_UNOBSERVABLE
        )
        facts["ulp_curve_worst_median"] = float(max(_meds)) if _meds else _ULP_UNOBSERVABLE
        facts["ulp_curve_headline"] = "median_ulp_diff"
        facts["ulp_predicted_ceiling"] = m.ULP_PREDICTED_CEILING
        facts["ulp_outliers"] = m.ulp_outliers([c["median_ulp_diff"] for c in facts["ulp_curve"]])
        facts["ulp_prediction_on_record"] = (
            "bench/results/criterion10-ulp-prediction.md: flat at 1-3 ULP across all 32 "
            "layers. Flat => no defect; a step => a located one. Recorded before measuring."
        )
        facts["logits_oracle_outcome"] = outcome
        facts["all_output_oracle_outcome"] = oracle_outcome

        # The all-output oracle can only make the outcome stricter, never laxer: a logits
        # AGREE with a KV DISAGREE is a DISAGREE.
        if oracle_outcome != m.COMPARISON_AGREE:
            outcome = oracle_outcome
        if not identical:
            outcome = m.COMPARISON_DISAGREE
            facts["cross_run_differing_outputs"] = differing[:10]
        facts["cross_run_identical_to_run1"] = identical
        facts["cross_run_outputs_compared"] = len(run)

        # The fifth costume, per run.  `oracle_outputs_within_tolerance = 65` counts
        # agreements; it does not count *evidence*.  An output no claimed node reaches was
        # compared against itself, and 65 of those is 0.0 == 0.0 in a fifth costume: not
        # zeros, not constants, but two sides of one computation, which no degeneracy
        # guard can see.  These two numbers are recorded under two names because they were
        # once read off one.
        _cov = attribution.output_coverage
        if _cov is None:
            facts["oracle_outputs_attributed"] = m.OUTPUT_COVERAGE_NOT_COMPUTED
            facts["oracle_outputs_vacuous"] = m.OUTPUT_COVERAGE_NOT_COMPUTED
        else:
            names = [o.name for o in vk_sess.get_outputs()]
            tokens = [_cov.token_for(n) for n in names[: len(run)]]
            facts["oracle_outputs_attributed"] = tokens.count(m.OUTPUT_EP_COVERED)
            facts["oracle_outputs_vacuous"] = tokens.count(m.OUTPUT_CPU_ONLY)
            facts["oracle_outputs_coverage_unobservable"] = tokens.count(
                m.OUTPUT_UNOBSERVABLE
            )
            failing = [
                names[j] for j in oracle_facts.get("oracle_failing_indices", [])
                if j < len(names)
            ]
            facts["coverage_instrument_refuted_by"] = _cov.refuted_by(failing)

        comparisons.append(outcome)
        per_run_facts.append(facts)
        cross_run_report.append(
            f"    run {i + 1}: {outcome}  argmax={facts['argmax_vk']} "
            f"(cpu {facts['argmax_cpu']})  top10={facts['top10_overlap']}/10  "
            f"cross-run identical (all {len(run)}): {identical}\n"
            f"             oracle: {oracle_outcome} over "
            f"{oracle_facts['oracle_outputs_compared']}/{oracle_facts['oracle_outputs_total']} "
            f"outputs, {oracle_facts['oracle_outputs_degenerate']} degenerate, "
            f"worst max_abs_diff="
            f"{oracle_facts['oracle_max_abs_diff_over_all_outputs']:.6g} "
            f"at output {oracle_facts['oracle_worst_output_index']}\n"
            f"             ULP (headline): median-over-outputs="
            f"{facts['ulp_curve_median_over_outputs']}, "
            f"worst per-output median={facts['ulp_curve_worst_median']}, "
            f"worst max_ulp={oracle_facts.get('oracle_max_ulp_diff_over_all_outputs')} "
            f"at output {oracle_facts.get('oracle_worst_ulp_output_index')} "
            f"(max is cancellation-sensitive; read the median)"
        )

    _counters_doc = _read_counters_doc(counters_path)
    _device_name = device_name_from_run(_counters_doc)
    _route = kv_writeback_route(_counters_doc)
    # The order the SESSION reports, taken from the session, never from a stored container.
    _output_names = [o.name for o in vk_sess.get_outputs()]
    _kv_depth.assert_names_are_session_order(_output_names)
    _depth_curve = _kv_depth.depth_curve(
        [c["median_ulp_diff"] for c in per_run_facts[0]["ulp_curve"]], _output_names
    )
    _depth_exceedances = _kv_depth.depth_exceedances(_depth_curve)

    series = m.AttributedRunSeries.from_runs(
        comparisons=comparisons,
        attribution=attribution,
        artifact=str(phi35_model_path),
        device_index=device_index,
        device_name=_device_name,
    )

    print(f"\n[M0 criterion 10 / Device {device_index}] {phi35_model_path.name}")
    print(f"    device (read off the run, not the selector): {_device_name}")
    print(
        f"    KV writeback route: {_route['route']} "
        f"(device-bound {_route.get('outputs_device_bound')}, "
        f"host-resident {_route.get('outputs_host_resident')}, "
        f"authority grants {_route.get('alloc_device_authority_grants')})"
    )
    print("\n".join(cross_run_report))
    print(f"    attribution: {series.describe()}")
    print(f"    executed_by: {attribution.executed_by}")
    print(f"    output coverage: {attribution.coverage_state}")
    _cov = attribution.output_coverage
    if _cov is not None and _cov.cpu_only_count:
        print(
            f"    WARNING: {_cov.cpu_only_count} of {len(_cov.output_names)} outputs are "
            "CPU-ONLY — their oracle comparison is our-CPU against ORT's-CPU and is not "
            "evidence.  First few: " + ", ".join(_cov.cpu_only_names[:5])
        )
    print(f"    series verdict: {series.verdict}")
    print(
        "    KV residual by LAYER (median ULP, key/value, depth order — not the "
        "alphabetised order in the record's per_output dict):"
    )
    print(
        "      "
        + "  ".join(
            f"L{row['layer']}:{row['key']}/{row['value']}" for row in _depth_curve
        )
    )
    print(
        f"    layers over the predicted {_kv_depth.LAYER_PREDICTED_CEILING} ULP band: "
        f"{_depth_exceedances or 'none'}"
    )

    # The verdict travels with the artifact it was measured on, into the counters JSON
    # that epctl, bench/ and the census read.  A caveat in a pytest caveat is not attached
    # to the number (§10.0 clause 5).
    if counters_path:
        try:
            m.write_equivalence_record(counters_path, series.as_verdict())
        except Exception as exc:  # noqa: BLE001
            print(f"    WARNING: could not write equivalence record: {exc}")

    # The lane artifact.  Written before the assertion so a failing series is recorded too:
    # a criterion whose evidence only exists when it passes is not evidence.
    try:
        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        record = series.to_record()
        record["per_run"] = per_run_facts
        record["required_runs"] = REQUIRED_RUNS
        record["kv_writeback_route"] = _route
        # The output order, explicitly, in the order the SESSION reports it.  The record is
        # written with `sort_keys=True`, so `output_coverage.per_output` is alphabetised —
        # and this model's session order is *not* its own sort (`present.10` sorts before
        # `present.2`).  A consumer that took that dict's key order for the output order
        # would attribute every KV residual to the wrong layer, reproducibly and on every
        # device.  It nearly did.
        record["output_names"] = _output_names
        record["output_names_order"] = (
            "session order (sess.get_outputs()); NOT the alphabetised key order of "
            "output_coverage.per_output, which sort_keys=True produces"
        )
        record["kv_depth_curve"] = _depth_curve
        record["kv_depth_exceedances"] = _depth_exceedances
        record["kv_depth_largest_step"] = {
            "key": _kv_depth.largest_step(_depth_curve, "key"),
            "value": _kv_depth.largest_step(_depth_curve, "value"),
        }
        record["device_name_source"] = (
            "counters alloc_device_frame_session_devices — the name the run reported, not "
            "the ONNXRUNTIME_EP_VULKAN_DEVICE selector that requested it"
        )
        out = _RESULTS_DIR / f"criterion10-dev{device_index}.json"
        out.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
        print(f"    record: {out}")
    except OSError as exc:
        print(f"    WARNING: could not write criterion-10 record: {exc}")

    # The coverage instrument's own falsifier, checked before anything is read off it
    # (R9).  Both sides of a CPU-ONLY output are the same computation, so a disagreement
    # there refutes the labelling — a finding about the harness, never about the EP.
    _refuted = sorted(
        {n for f in per_run_facts for n in f.get("coverage_instrument_refuted_by", [])}
    )
    if _refuted:
        raise m.InstrumentError(
            f"[Device {device_index}] outputs {_refuted} were labelled CPU-ONLY yet "
            "disagree with the CPU oracle.  The per-output coverage instrument is wrong "
            "(topology/trace join, or ORT eliminated a node whose event we relied on).  "
            "ERROR(instrument): nothing about the EP may be read off this run (R13)."
        )

    series.assert_closes_criterion_10(required_runs=REQUIRED_RUNS)

@pytest.mark.slow
def test_criterion_10_record_names_what_it_counts(
    phi35_model_path: pathlib.Path,
    require_vulkan,
) -> None:
    """The written record must say what its counts mean — R11, checked mechanically.

    A record carrying a bare ``3`` beside a 363-node model is a misnomer waiting to be
    read as a disaster. This test asserts the artifact from the previous test carries the
    definition, so the obligation is enforced by a mechanism rather than by whoever writes
    the next report remembering.
    """
    device_index = os.environ.get("ONNXRUNTIME_EP_VULKAN_DEVICE", "0")
    path = _RESULTS_DIR / f"criterion10-dev{device_index}.json"
    if not path.is_file():
        pytest.skip(
            f"No criterion-10 record at {path}; run "
            "test_criterion_10_three_consecutive_attributed_match first."
        )
    record = json.loads(path.read_text(encoding="utf-8"))

    assert record.get("verdict") in m.EQUIVALENCE_VERDICTS, (
        f"record verdict {record.get('verdict')!r} is not in the shared vocabulary "
        f"{m.EQUIVALENCE_VERDICTS}"
    )
    assert "counts_what" in record.get("series", {}), (
        "the series record does not say what its count counts; a bare island count is "
        "an R11 misnomer waiting to happen"
    )
    assert "graph nodes" in record["series"]["counts_what"], (
        "the series record's definition must explicitly deny the graph-node reading"
    )
    assert record.get("executed_by"), (
        "the record carries no executed_by; §10.0 clause 3 — a verdict without its "
        "executor is a verdict from a world it has not identified"
    )

    # The reopened row's proximate cause, pinned. `outputs_compared: 65` counted cross-run
    # comparisons, sat among the oracle facts, and was quoted into the criteria table as
    # sixty-five oracle comparisons beside a max_abs_diff covering one tensor. Two counts,
    # two names, and the bare form must not come back.
    per_run = record.get("per_run") or []
    assert per_run, "the record carries no per_run facts to check the counts against"
    for i, facts in enumerate(per_run):
        assert "oracle_outputs_compared" in facts, (
            f"run {i + 1} does not say how many outputs the CPU oracle covered; that is "
            "the fact whose absence reopened criterion 10"
        )
        assert "cross_run_outputs_compared" in facts, (
            f"run {i + 1} does not separately name its cross-run count"
        )
        assert "outputs_compared" not in facts, (
            f"run {i + 1} carries the bare `outputs_compared` key again; it is the one "
            "that was read as an oracle count while holding a determinism count"
        )
        assert facts["oracle_outputs_compared"] == facts["oracle_outputs_total"], (
            f"run {i + 1} compared {facts['oracle_outputs_compared']} of "
            f"{facts['oracle_outputs_total']} outputs against CPU; a partial oracle is "
            "how one-of-sixty-five passed for a month"
        )
        assert facts.get("oracle_outputs_degenerate") == 0, (
            f"run {i + 1} has {facts.get('oracle_outputs_degenerate')} degenerate output "
            "pairs; those comparisons are vacuous and cannot close the criterion"
        )
