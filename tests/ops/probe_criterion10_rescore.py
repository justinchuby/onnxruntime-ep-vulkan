"""Criterion 10, re-scored under the §8.9.22-ruled statistic — and the census of what
actually decided its verdict.
WHY THIS PROBE EXISTS
=====================
`docs/DESIGN.md` §8.9.22 replaced criterion 10's logits observable: a max-ULP over a
tensor containing subnormal references measures the degeneracy, not the subject. The
replacement reports the residual on the **normal domain** and **publishes** the subnormal
population beside it.

The question put to me was whether criterion 10's `DIVERGENT` verdict was an artifact of
the defective element-basis statistic. **The hypothesis is falsifiable and this probe is
the falsifier.** It answers it two ways, because either alone can be argued with:

1. **Mechanically** — it records, per output, what predicate produced the verdict. If the
   verdict predicate never reads a ULP statistic, no repair to a ULP statistic can move it.
2. **Numerically** — it re-scores all 65 outputs under the ruled statistic and reports
   whether the ruled residual differs from the element-basis one at all. On criterion 10's
   own tensors it may not, and that is a result rather than a formality: §8.9.22's
   mechanism has to actually be present in the data for its remedy to change anything.

AND THE CENSUS, which is the part nobody has run
================================================
For every output that fails, this enumerates the **failing elements themselves**: how
many, their reference magnitudes, their residuals expressed in ULPs of the dtype at the
tensor's own scale, and how many of them sit below one ULP at that scale.

`np.allclose`'s allowance is `atol + rtol*|b|`, which collapses to `atol` as `|b| -> 0`.
That is the same degenerate-denominator shape §8.9.22 ruled on, one level down and in the
*gate* rather than in the *report*. Whether it is actually the mechanism here is a
measurement, and this probe takes it. **It takes it and it does not act on it**: `atol` is
not touched here or anywhere on this branch. A tolerance argument is Morpheus's to rule,
and the person whose measurement made three outputs red is the last person who should be
choosing the band.

USAGE
=====
    $env:ONNXRUNTIME_VULKAN_EP_LIB = "rust/target/release/onnxruntime_vulkan_ep.dll"
    $env:ONNXRUNTIME_EP_VULKAN_DEVICE = "0"
    $env:ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE = "bench/results/c10_rescore_counters-dev0.json"
    python tests/ops/probe_criterion10_rescore.py

    python tests/ops/probe_criterion10_rescore.py --selftest   # no GPU, no model
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import _models as m  # noqa: E402

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_RESULTS_DIR = _REPO_ROOT / "bench" / "results"


def failing_element_census(vk: np.ndarray, cpu: np.ndarray, tol: dict) -> dict:
    """Enumerate the elements that fail `np.allclose`, and say where they sit.

    Returns counts and magnitudes only -- never a proposed tolerance. The last line of
    this docstring is load-bearing: **this function must not grow a recommendation.**
    """
    a = vk.astype(np.float64)
    b = cpu.astype(np.float64)
    allowance = tol["atol"] + tol["rtol"] * np.abs(b)
    diff = np.abs(a - b)
    bad = diff > allowance
    n_bad = int(np.count_nonzero(bad))

    out: dict = {
        "failing_elements": n_bad,
        "total_elements": int(b.size),
        "failing_fraction": (n_bad / b.size) if b.size else 0.0,
        "atol": tol["atol"],
        "rtol": tol["rtol"],
    }
    if n_bad == 0:
        out["note"] = "no element fails the allowance; this output is within tolerance"
        return out

    floating = bool(np.issubdtype(vk.dtype, np.floating))
    scale = float(np.abs(b).max()) if b.size else 0.0
    one_ulp_at_scale = (
        abs(float(np.spacing(vk.dtype.type(scale)))) if floating and scale > 0 else 0.0
    )
    tiny = float(np.finfo(vk.dtype).tiny) if floating else 0.0

    br = np.abs(b[bad])
    dr = diff[bad]
    out.update(
        {
            "tensor_scale": scale,
            "one_ulp_at_scale": one_ulp_at_scale,
            "smallest_normal": tiny,
            "failing_ref_min": float(br.min()),
            "failing_ref_median": float(np.median(br)),
            "failing_ref_max": float(br.max()),
            "failing_abs_diff_max": float(dr.max()),
            # The three populations that matter, and they are nested:
            #   subnormal  <  below one ULP at the tensor's scale  <  everything
            "failing_with_subnormal_reference": int(np.count_nonzero(br < tiny)),
            "failing_below_one_ulp_at_scale": (
                int(np.count_nonzero(br < one_ulp_at_scale)) if one_ulp_at_scale else 0
            ),
            # Is the residual itself within one representable step of the output format at
            # the tensor's scale?  A residual of one ULP-at-scale is the smallest nonzero
            # difference the format can express there.  REPORTED, not acted on.
            "failing_residual_within_one_ulp_at_scale": (
                int(np.count_nonzero(dr <= one_ulp_at_scale)) if one_ulp_at_scale else 0
            ),
            "failing_max_ulp_at_scale": (
                float((dr / one_ulp_at_scale).max()) if one_ulp_at_scale else None
            ),
            "atol_in_ulps_at_scale": (
                float(tol["atol"] / one_ulp_at_scale) if one_ulp_at_scale else None
            ),
        }
    )
    return out


def rescore(vk_out: list[np.ndarray], cpu_out: list[np.ndarray], names: list[str]) -> dict:
    outcome, facts = m.compare_all_outputs_to_cpu(vk_out, cpu_out)
    per_output = []
    for i, e in enumerate(facts["per_output"]):
        tol, _why = m.tolerance_for_output(vk_out[i])
        census = failing_element_census(vk_out[i], cpu_out[i], tol)
        per_output.append(
            {
                "index": i,
                "name": names[i] if i < len(names) else None,
                "shape": e.get("shape"),
                "dtype": e.get("dtype"),
                # THE PER-OUTPUT VERDICT.  Never an aggregate.
                "verdict": e.get("status"),
                "verdict_predicate": e.get("verdict_predicate"),
                # §8.9.22: two numbers, never one.
                "ruled_observable_report": e.get("ruled_observable_report"),
                "max_ulp_normal_domain": e.get("max_ulp_normal_domain"),
                "median_ulp_normal_domain": e.get("median_ulp_normal_domain"),
                "p99_ulp_normal_domain": e.get("p99_ulp_normal_domain"),
                "normal_domain_elements": e.get("normal_domain_elements"),
                "subnormal_reference_elements": e.get("subnormal_reference_elements"),
                "subnormal_reference_fraction": e.get("subnormal_reference_fraction"),
                "normal_domain_verdict": e.get("normal_domain_verdict"),
                # The statistic §8.9.22 replaced, kept so the comparison is checkable.
                "max_ulp_diff_element_basis": e.get("max_ulp_diff"),
                "ruled_equals_element_basis": (
                    e.get("max_ulp_normal_domain") == e.get("max_ulp_diff")
                ),
                "median_ulp_diff": e.get("median_ulp_diff"),
                "max_ulp_at_scale_diff": e.get("max_ulp_at_scale_diff"),
                "max_abs_diff": e.get("max_abs_diff"),
                "ulp_cancellation_elements": e.get("ulp_cancellation_elements"),
                "ulp_basis_verdict": e.get("ulp_basis_verdict"),
                "failing_element_census": census,
            }
        )

    ruled_moved = [p["index"] for p in per_output if not p["ruled_equals_element_basis"]]
    return {
        "comparison_outcome": outcome,
        "outputs_total": facts["oracle_outputs_total"],
        "outputs_compared": facts["oracle_outputs_compared"],
        "outputs_within_tolerance": facts["oracle_outputs_within_tolerance"],
        "outputs_degenerate": facts["oracle_outputs_degenerate"],
        "failing_indices": facts["oracle_failing_indices"],
        "subnormal_reference_elements_total": facts[
            "oracle_subnormal_reference_elements_total"
        ],
        "outputs_where_ruled_statistic_differs_from_element_basis": ruled_moved,
        "outputs_with_empty_normal_domain": facts["oracle_outputs_with_empty_normal_domain"],
        "per_output": per_output,
    }


def _selftest() -> int:
    """No GPU, no model.  Proves the census can distinguish the two mechanisms."""
    tiny = float(np.finfo(np.float16).tiny)

    # (a) a failure carried by subnormal references -- §8.9.22's mechanism
    cpu = np.full(64, 8.0, dtype=np.float16)
    cpu[:4] = np.float16(tiny / 2)
    vk = cpu.copy()
    vk[:4] = np.float16(0.25)
    tol, _ = m.tolerance_for_output(vk)
    c = failing_element_census(vk, cpu, tol)
    assert c["failing_elements"] == 4, c
    assert c["failing_with_subnormal_reference"] == 4, c

    # (b) a failure carried by references that are small relative to the tensor's scale
    #     but comfortably NORMAL -- which §8.9.22's domain split does NOT exclude
    cpu2 = np.full(64, 8.0, dtype=np.float16)
    cpu2[:4] = np.float16(1.0e-3)  # 16x above the smallest normal
    vk2 = cpu2.copy()
    vk2[:4] = np.float16(1.0e-3 + 0.01)
    c2 = failing_element_census(vk2, cpu2, tol)
    assert c2["failing_elements"] == 4, c2
    assert c2["failing_with_subnormal_reference"] == 0, c2
    assert c2["failing_below_one_ulp_at_scale"] == 4, c2

    # The two arms must reach DIFFERENT conclusions, or the census reads the same for
    # both mechanisms and distinguishes nothing.
    assert (
        c["failing_with_subnormal_reference"] != c2["failing_with_subnormal_reference"]
    )

    # (c) and it must not invent failures
    c3 = failing_element_census(cpu2, cpu2, tol)
    assert c3["failing_elements"] == 0, c3
    print("SELFTEST PASS: 3 arms, the two mechanisms are distinguished")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return _selftest()

    import onnxruntime as ort  # noqa: PLC0415

    from test_phi35 import _ONNX_FILE, _build_phi35_feeds  # noqa: PLC0415

    lib = os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB")
    if not lib:
        print("ERROR(instrument): ONNXRUNTIME_VULKAN_EP_LIB unset; refusing to run")
        return 2
    # The plugin is registered by tests/ops/conftest.py under pytest.  This probe runs as
    # a script, so it registers explicitly rather than inheriting a fixture -- and if it
    # did not, ORT would fall back to the CPU EP and the whole re-score would be
    # CPU-vs-CPU, which agrees perfectly and proves nothing.
    try:
        ort.register_execution_provider_library(m.EP_NAME, str(lib))
    except Exception as exc:  # noqa: BLE001
        if "already registered" not in str(exc):
            print(f"ERROR(instrument): could not register {m.EP_NAME}: {exc}")
            return 2

    device_index = os.environ.get("ONNXRUNTIME_EP_VULKAN_DEVICE", "0")
    counters_path = os.environ.get("ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE")
    model = pathlib.Path(_ONNX_FILE)
    feeds = _build_phi35_feeds()

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    vk_sess = ort.InferenceSession(str(model), opts, providers=m.EP_PROVIDERS)
    if m.EP_NAME not in vk_sess.get_providers():
        print(f"ERROR(instrument): {m.EP_NAME} not registered; a re-score without the EP "
              "is CPU-vs-CPU and measures nothing")
        return 2

    # Session order, not file order.  The frame of a name is the run that produced it.
    names = [o.name for o in vk_sess.get_outputs()]
    vk_out = vk_sess.run(None, feeds)
    del vk_sess

    cpu_opts = ort.SessionOptions()
    cpu_opts.log_severity_level = 3
    cpu_sess = ort.InferenceSession(str(model), cpu_opts, providers=["CPUExecutionProvider"])
    cpu_out = cpu_sess.run(None, feeds)

    rec = rescore(vk_out, cpu_out, names)

    counters = None
    if counters_path and pathlib.Path(counters_path).exists():
        counters = json.loads(pathlib.Path(counters_path).read_text(encoding="utf-8"))
    # Device name off the RUN, and the dispatch screen Switch asked for: a lane with zero
    # EP dispatches exits 0 and raises nothing, and would be read as a clean measurement.
    dispatches = None
    device_name = "UNOBSERVABLE"
    if isinstance(counters, dict):
        c = counters.get("counters", counters)
        dispatches = c.get("dispatches_executed")
        for k in ("alloc_device_frame_session_devices", "running_device_names", "device_name"):
            if c.get(k):
                device_name = str(c[k])
                break
    rec["device_selector_requested"] = device_index
    rec["device_name_from_run"] = device_name
    rec["dispatches_executed"] = dispatches
    rec["dispatch_screen"] = (
        "UNOBSERVABLE(no counters file)"
        if dispatches is None
        else ("PASS" if dispatches > 0 else "ERROR(instrument=zero_ep_dispatches)")
    )
    rec["ruling"] = "docs/DESIGN.md §8.9.22"

    out = _RESULTS_DIR / f"criterion10_rescore_8922-dev{device_index}.json"
    out.write_text(json.dumps(rec, indent=1, sort_keys=True), encoding="utf-8")

    print(f"device (off the run): {device_name}   dispatch screen: {rec['dispatch_screen']}")
    print(f"outputs {rec['outputs_within_tolerance']}/{rec['outputs_total']} within "
          f"tolerance; failing {rec['failing_indices']}")
    print(f"subnormal references over all 65 outputs: "
          f"{rec['subnormal_reference_elements_total']}")
    print("outputs where the RULED statistic differs from the element basis: "
          f"{rec['outputs_where_ruled_statistic_differs_from_element_basis']}")
    for p in rec["per_output"]:
        if p["verdict"] != "WITHIN_TOLERANCE":
            print(f"\n  [{p['index']}] {p['name']}  verdict={p['verdict']}")
            print(f"      ruled: {p['ruled_observable_report']}")
            print(f"      element-basis max_ulp={p['max_ulp_diff_element_basis']} "
                  f"(ruled == element basis: {p['ruled_equals_element_basis']})")
            print(f"      predicate: {p['verdict_predicate']}")
            print(f"      census: {json.dumps(p['failing_element_census'], sort_keys=True)}")
    print(f"\nrecord: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
