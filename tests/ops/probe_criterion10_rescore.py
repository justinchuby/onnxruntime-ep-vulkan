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


# -- §8.9.24(3): the companion obligation, enforced by code rather than by memory --------
#
# Implemented in `_models.py` beside the comparator that carries `verdict_predicate`, and
# imported here, because a second implementation of "the allowance in ULPs-at-scale" is a
# second answer nobody could reconcile -- the exact failure `ulp_distribution` was
# extracted to prevent.
from _models import (  # noqa: E402
    ULP_AT_SCALE_COMPANIONS,  # noqa: F401  (re-exported: the lane asserts against it)
    UlpAtScaleCompanionError,  # noqa: F401
    allowance_in_ulps_at_scale,
    assert_ulp_at_scale_row_is_complete,
)


def failing_element_census(vk: np.ndarray, cpu: np.ndarray, tol: dict) -> dict:
    """Enumerate the elements that fail `np.allclose`, and say where they sit.

    Returns counts and magnitudes only -- never a proposed tolerance. The last line of
    this docstring is load-bearing: **this function must not grow a recommendation.**

    Every `ULP-at-scale` figure here is accompanied by the allowance in the same unit and
    by the failing set on the element basis (§8.9.24(3)); the row is checked before it is
    returned, so the obligation cannot be discharged by remembering it.
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
    # (b) of §8.9.24(3): the failing set on the ELEMENT basis -- the granularity the
    # predicate actually evaluates at.  `spacing` is taken at each failing element's own
    # reference, not at the tensor maximum.
    elem_spacing = m.format_spacing(b[bad], vk.dtype)
    elem_ulps = dr / np.where(elem_spacing == 0, np.inf, elem_spacing)
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
            #
            # §8.9.24(1) CORRECTS THE READING OF THIS NUMBER AND THE CORRECTION IS HERE
            # BECAUSE THIS IS WHERE IT WAS MISREAD.  "Within one ULP-at-scale" does NOT
            # mean "fails by less than one representable step".  The step it is measured
            # against is the step at the TENSOR MAXIMUM, which on layer 31's key is ~500x
            # the magnitude of the elements that actually failed.  On the element basis --
            # `failing_ulp_element_basis_*` below -- those same elements fail by more than
            # twenty representable fp16 steps at their own magnitudes.
            "failing_residual_within_one_ulp_at_scale": (
                int(np.count_nonzero(dr <= one_ulp_at_scale)) if one_ulp_at_scale else 0
            ),
            "failing_residual_within_one_ulp_at_scale_caveat": (
                "the step here is the step at the TENSOR MAXIMUM, not at the failing "
                "element; read failing_ulp_element_basis_* for the predicate's own "
                "granularity (docs/DESIGN.md §8.9.24(1))"
            ),
            "failing_max_ulp_at_scale": (
                float((dr / one_ulp_at_scale).max()) if one_ulp_at_scale else None
            ),
            # ONE TERM OF A TWO-TERM SUM.  Kept because it has been quoted, and it is now
            # impossible to quote without the whole allowance beside it.
            "atol_in_ulps_at_scale": (
                float(tol["atol"] / one_ulp_at_scale) if one_ulp_at_scale else None
            ),
            "atol_in_ulps_at_scale_caveat": (
                "atol is ONE TERM of the predicate's allowance `atol + rtol*|b|`; this "
                "figure is not the tolerance and quoting it as such is the refuted "
                "unsatisfiability argument (docs/DESIGN.md §8.9.24(1))"
            ),
            "failing_ulp_element_basis_max": float(np.max(elem_ulps[np.isfinite(elem_ulps)]))
            if np.any(np.isfinite(elem_ulps))
            else None,
            "failing_ulp_element_basis_median": float(
                np.median(elem_ulps[np.isfinite(elem_ulps)])
            )
            if np.any(np.isfinite(elem_ulps))
            else None,
            "failing_ulp_element_basis_min": float(np.min(elem_ulps[np.isfinite(elem_ulps)]))
            if np.any(np.isfinite(elem_ulps))
            else None,
            "failing_ulp_element_basis_note": (
                "|a-b| / spacing(b) at each FAILING element's own reference -- the "
                "granularity np.allclose evaluates at (§8.9.24(3)(b))"
            ),
            # §8.9.24's satisfiability bound, recomputed on THIS tensor rather than
            # quoted.  `allowance/ulp(b) >= rtol*2**10 = 20.48` for every normal b, so a
            # failing element has exceeded an allowance at least 20.48 element-ULPs wide.
            "allowance_in_ulps_element_basis_min": (
                float(
                    np.min(
                        (tol["atol"] + tol["rtol"] * br)
                        / np.where(elem_spacing == 0, np.inf, elem_spacing)
                    )
                )
                if br.size
                else None
            ),
            "satisfiability_bound_element_basis": tol["rtol"] * 1024.0,
            "satisfiability_bound_note": (
                "ulp(b) <= |b|*2**-10 for normal fp16, so allowance/ulp(b) >= rtol*2**10 "
                "independent of magnitude; a failing element exceeded an allowance at "
                "least this many representable steps wide (docs/DESIGN.md §8.9.24(1))"
            ),
        }
    )
    out.update(allowance_in_ulps_at_scale(br, tol, one_ulp_at_scale))
    assert_ulp_at_scale_row_is_complete(out, where="failing_element_census")
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
                # §8.9.24(3): carried through from the comparator so this row is complete
                # in its own right. A reader quoting `max_ulp_at_scale_diff` off this
                # record now has the whole allowance in the same unit on the same line.
                "allowance_in_ulps_at_scale_min": e.get("allowance_in_ulps_at_scale_min"),
                "allowance_in_ulps_at_scale_median": e.get("allowance_in_ulps_at_scale_median"),
                "allowance_in_ulps_at_scale_max": e.get("allowance_in_ulps_at_scale_max"),
                "allowance_in_ulps_at_scale_basis": e.get("allowance_in_ulps_at_scale_basis"),
                "ulp_element_basis_max": e.get("ulp_element_basis_max"),
                "ulp_element_basis_median": e.get("ulp_element_basis_median"),
                "satisfiability_bound_element_basis": e.get("satisfiability_bound_element_basis"),
                "satisfiability_bound_note": e.get("satisfiability_bound_note"),
                "max_ulp_diff": e.get("max_ulp_diff"),
                "max_abs_diff": e.get("max_abs_diff"),
                "ulp_cancellation_elements": e.get("ulp_cancellation_elements"),
                "ulp_basis_verdict": e.get("ulp_basis_verdict"),
                "failing_element_census": census,
            }
        )
        assert_ulp_at_scale_row_is_complete(
            per_output[-1], where=f"rescore per_output[{i}] ({names[i] if i < len(names) else i})"
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

    # (d) §8.9.24(3): the companion check must be able to REFUSE.  A check never shown to
    #     fire is indistinguishable from one that cannot -- round 34's lesson, applied to
    #     the remedy for round 37's error.
    incomplete = dict(c)
    incomplete.pop("allowance_in_ulps_at_scale_min")
    try:
        assert_ulp_at_scale_row_is_complete(incomplete, where="selftest")
    except UlpAtScaleCompanionError as exc:
        assert "allowance_in_ulps_at_scale_min" in str(exc), exc
    else:  # pragma: no cover - this branch is the failure
        raise AssertionError(
            "the §8.9.24(3) companion check passed a row missing the allowance; it cannot "
            "go red and therefore witnesses nothing"
        )
    assert_ulp_at_scale_row_is_complete(c, where="selftest(complete row)")

    # (e) THE INVERSION, on a specimen built to have layer 31's shape.  A tensor whose
    #     scale is ~500x the magnitude of its failing elements reads "every failure within
    #     one ULP-at-scale" while the SAME failures are more than twenty representable
    #     steps wide at their own magnitudes.  This is the reading §8.9.24(1) corrected,
    #     asserted rather than described.
    cpu4 = np.full(64, 5.75, dtype=np.float16)
    cpu4[:4] = np.float16(0.011)
    vk4 = cpu4.copy()
    vk4[:4] = (cpu4[:4].astype(np.float32) + 0.0025).astype(np.float16)
    c4 = failing_element_census(vk4, cpu4, tol)
    assert c4["failing_elements"] == 4, c4
    assert c4["failing_residual_within_one_ulp_at_scale"] == 4, c4
    assert c4["failing_max_ulp_at_scale"] <= 1.0, c4
    assert c4["failing_ulp_element_basis_min"] > 20.0, c4
    assert c4["allowance_in_ulps_at_scale_min"] > 0.1, c4
    assert c4["allowance_in_ulps_element_basis_min"] >= c4["satisfiability_bound_element_basis"], c4

    print("SELFTEST PASS: 5 arms, the two mechanisms are distinguished, the §8.9.24(3) "
          "companion check goes red on an incomplete row, and the element basis inverts "
          "the at-scale reading on a layer-31-shaped specimen")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return _selftest()

    import onnxruntime as ort  # noqa: PLC0415

    from test_phi35 import _PHI35_SPEC, _build_phi35_feeds, _foundry_discovery  # noqa: PLC0415

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
    try:
        model = _foundry_discovery.resolve_model_path(_PHI35_SPEC)
    except _foundry_discovery.FoundryDiscoveryError as exc:
        print(f"ERROR(instrument): Phi-3.5 model not resolvable: {exc}")
        return 2
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
