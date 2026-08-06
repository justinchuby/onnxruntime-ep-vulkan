"""RAI-011 falsifier: does the net-benefit gate actually evaluate Phi-3.5's island?

Mouse, 2026-08-01.

## What this is a falsifier for

`OP_COVERAGE.md` §7.10 recorded that Phi-3.5 partitions into exactly one cluster, that the
single-cluster branch in `GetCapability` therefore skipped `partition::evaluate` entirely, and
that `viable_islands_retained == 0` consequently meant *bypassed* while reading identically to
*all-rejected*. §7.10.3 said what would close it: not a bigger synthetic test, but a real run in
which the gate makes a decision about our model.

R10's falsifier for "X is wired" is **an artifact X produced whose content varies with X's
input.** So a single run showing `net_benefit_gate: EVALUATED` is not sufficient — a constant
proves nothing. This probe runs the *same model at the same commit* under several partition
configurations and requires the observables to move with them.

## The configurations, and what each one is for

| config | overrides | what it demonstrates |
|---|---|---|
| `default` | none | the gate evaluates the island in the shipping configuration |
| `no_anchor` | anchor exemption off | the *economics arithmetic* is the branch that answers, not the exemption |
| `no_anchor_fixed_<n>` | anchor exemption off, `fixed_ns = n` | the verdict as a function of the one uncalibrated parameter |

Phi-3.5's island is anchor-bearing, so with the exemption on, `evaluate` returns `Claim` at the
anchor branch and the economics numbers never decide anything. Turning the exemption off is what
lets the arithmetic be observed *deciding*. Those runs are counterfactuals and the EP logs a WARN
saying so; they are not claims about the shipping configuration.

## What must move

* `net_benefit_gate_evaluations` — `0` before this change on Phi-3.5, non-zero after.
* `net_benefit_gate_bypasses` — must be `0` in every run. A non-zero value means a second,
  un-evaluated path into the partitioner was reintroduced.
* `viable_islands_retained` vs `net_benefit_sole_island_overrides` — these two are the states that
  used to share the digit `0`. Exactly one of them is non-zero in any single-island run, and which
  one it is must change with `fixed_ns`.

Output: ``bench/results/net_benefit_gate_probe.json``. Never the repo root.

Usage::

    python rust/tools/probe_net_benefit_gate.py            # all configs
    python rust/tools/probe_net_benefit_gate.py --child default   # one config, in-process
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
_RESULTS = _ROOT / "bench" / "results"
_RESULTS.mkdir(parents=True, exist_ok=True)

# Resolved by identity (variant name + execution provider), not by a hardcoded path: Foundry
# Local's own on-disk cache layout is versioned by its CLI's internal catalog revision (issue
# #11), and a hardcoded path silently goes stale when that happens with no code change on either
# side. See foundry_discovery.py for the full discovery contract (fail-loud, never guessed).
import foundry_discovery as _foundry_discovery  # noqa: E402

_PHI35_SPEC = _foundry_discovery.FoundryModelSpec(
    variant_name="Phi-3.5-mini-instruct-cuda-gpu",
    execution_provider="CUDAExecutionProvider",
    onnx_filename="phi-3.5-mini-instruct-cuda-int4-rtn-block-32.onnx",
    download_alias="phi-3.5-mini",
)
EP_LIB = _ROOT / "rust" / "target" / "release" / "onnxruntime_vulkan_ep.dll"
EP_NAME = "VulkanExecutionProvider"

# The fixed_ns sweep. `TransferModel::DISCRETE.fixed_ns` is 60,000 and has never been calibrated
# against a measurement; R13 forbids calibrating it while no timing source on this project is
# trusted. The range below is deliberately three orders of magnitude wide in each direction of the
# current guess, because "the plausible range" is exactly what we do not know.
FIXED_NS_SWEEP = [1_000.0, 60_000.0, 1_000_000.0, 5_000_000.0, 20_000_000.0, 100_000_000.0]

CONFIGS: dict[str, dict[str, str]] = {
    "default": {},
    "no_anchor": {"ONNXRUNTIME_EP_VULKAN_PARTITION_ANCHOR_EXEMPTION": "0"},
}
for _f in FIXED_NS_SWEEP:
    CONFIGS[f"no_anchor_fixed_{int(_f)}"] = {
        "ONNXRUNTIME_EP_VULKAN_PARTITION_ANCHOR_EXEMPTION": "0",
        "ONNXRUNTIME_EP_VULKAN_PARTITION_FIXED_NS": str(_f),
    }

# The `fixed_ns`-isolating pair.
#
# The sweep above cannot show `fixed_ns` flipping anything, because `ep.rs`'s island estimator
# reports Phi-3.5's boundary as 89,199,100,032 B (it counts every node's outputs, and substitutes
# 128 for every unknown dimension) against a *measured* per-inference boundary of 856,720 B. The
# byte term therefore swamps `fixed_ns` by ~10^3 at every point in the sweep, and the verdict is
# constant. To see `fixed_ns` decide anything at all, the byte term has to be removed — which is
# what an absurd `bytes_per_ns` does.
#
# PREDICTION, written before these two ran (compute_ns = 23,020,437,504 / 1000 = 23,020,437.5 ns;
# transfer_ns ≈ 2·fixed_ns; margin 3): the flip is at fixed_ns = 23,020,437.5 / 6 = 3,836,739.6 ns.
# So 1,000,000 must CLAIM (viable=1, overrides=0) and 10,000,000 must REJECT (viable=0,
# overrides=1). Falsifier: either run landing on the other side, or neither moving.
for _f in (1_000_000.0, 10_000_000.0):
    CONFIGS[f"no_anchor_freebytes_fixed_{int(_f)}"] = {
        "ONNXRUNTIME_EP_VULKAN_PARTITION_ANCHOR_EXEMPTION": "0",
        "ONNXRUNTIME_EP_VULKAN_PARTITION_FIXED_NS": str(_f),
        "ONNXRUNTIME_EP_VULKAN_PARTITION_BYTES_PER_NS": "1e12",
    }

# Keys lifted out of the counters artifact. Note these are *counts and states*, never durations:
# a count does not care whether the box is busy.
KEYS = [
    "claimed_nodes",
    "islands_offered",
    "viable_islands_retained",
    "net_benefit_gate",
    "net_benefit_gate_clusters_seen",
    "net_benefit_gate_evaluations",
    "net_benefit_gate_bypasses",
    "net_benefit_sole_island_overrides",
    "net_benefit_override_reason",
]


def run_child(config: str, counters_path: pathlib.Path) -> None:
    """Create one Phi-3.5 session under the current env and let teardown dump the counters."""
    import numpy as np
    import onnxruntime as ort

    try:
        ort.register_execution_provider_library(EP_NAME, str(EP_LIB))
    except Exception as exc:  # noqa: BLE001
        if "already registered" not in str(exc):
            raise

    onnx_path = _foundry_discovery.resolve_model_path(_PHI35_SPEC)
    sess = ort.InferenceSession(
        str(onnx_path),
        sess_options=ort.SessionOptions(),
        providers=[EP_NAME, "CPUExecutionProvider"],
        free_dimension_overrides_by_name={"batch_size": "1", "sequence_length": "1"},
    )
    feeds: dict[str, "np.ndarray"] = {
        "input_ids": np.array([[1]], dtype=np.int64),
        "attention_mask": np.array([[1]], dtype=np.int64),
    }
    empty_kv = np.empty((1, 32, 0, 96), dtype=np.float16)
    for layer in range(32):
        feeds[f"past_key_values.{layer}.key"] = empty_kv
        feeds[f"past_key_values.{layer}.value"] = empty_kv
    out = sess.run(None, feeds)
    argmax = int(out[0].reshape(-1, out[0].shape[-1])[-1].argmax())
    print(f"[child:{config}] argmax={argmax}")
    del sess
    print(f"[child:{config}] counters → {counters_path} exists={counters_path.exists()}")


def run_parent(only: str | None = None) -> int:
    device = os.environ.get("ONNXRUNTIME_EP_VULKAN_DEVICE", "0")
    rows: list[dict] = []
    configs = {k: v for k, v in CONFIGS.items() if only is None or only in k}
    for config, overrides in configs.items():
        counters_path = _RESULTS / f"net_benefit_gate-{config}-dev{device}.counters.json"
        counters_path.unlink(missing_ok=True)
        env = os.environ.copy()
        env.update(overrides)
        env["ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE"] = str(counters_path)
        print(f"=== {config}: {overrides or 'no overrides (shipping configuration)'}")
        proc = subprocess.run(
            [sys.executable, str(pathlib.Path(__file__).resolve()), "--child", config],
            env=env,
            capture_output=True,
            text=True,
        )
        row: dict = {"config": config, "overrides": overrides, "device": device}
        if proc.returncode != 0:
            # R13: an instrument error is never a detection. Quote the failure text.
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-6:]
            row["verdict"] = "ERROR(instrument)"
            row["failure_text"] = "\n".join(tail)
            print(f"    ERROR(instrument): {row['failure_text']}")
            rows.append(row)
            continue
        if not counters_path.exists():
            row["verdict"] = "ERROR(instrument)"
            row["failure_text"] = f"counters file was never written: {counters_path}"
            print(f"    ERROR(instrument): {row['failure_text']}")
            rows.append(row)
            continue
        doc = json.loads(counters_path.read_text())
        row["verdict"] = "PASS"
        row["counters"] = {k: doc.get(k, "<absent>") for k in KEYS}
        rows.append(row)
        for k in KEYS:
            print(f"    {k:38s} {doc.get(k, '<absent>')}")

    out_path = _RESULTS / f"net_benefit_gate_probe-dev{device}{'-' + only if only else ''}.json"
    out_path.write_text(json.dumps({"device": device, "rows": rows}, indent=2))
    print(f"\n[probe] → {out_path}")

    # --- the falsifier assertions -------------------------------------------------------------
    ok = True
    good = [r for r in rows if r["verdict"] == "PASS"]
    if len(good) != len(rows):
        print("FAIL: at least one configuration produced ERROR(instrument); see failure_text")
        ok = False
    for r in good:
        c = r["counters"]
        if c["net_benefit_gate_bypasses"] != 0:
            print(f"FAIL({r['config']}): bypasses != 0 — a second un-evaluated path exists")
            ok = False
        if c["net_benefit_gate_evaluations"] == 0:
            print(f"FAIL({r['config']}): the gate evaluated nothing")
            ok = False
        if c["net_benefit_gate"] != "EVALUATED":
            print(f"FAIL({r['config']}): net_benefit_gate == {c['net_benefit_gate']!r}")
            ok = False
    # The artifact must VARY with the input, not merely exist (R10).
    retained = {r["config"]: r["counters"]["viable_islands_retained"] for r in good}
    overrides_seen = {
        r["config"]: r["counters"]["net_benefit_sole_island_overrides"] for r in good
    }
    if len(set(map(str, retained.values()))) < 2:
        print(f"FAIL: viable_islands_retained never changed across configs: {retained}")
        ok = False
    if len(set(map(str, overrides_seen.values()))) < 2:
        print(f"FAIL: sole-island overrides never changed across configs: {overrides_seen}")
        ok = False
    # Claimed nodes must be invariant: the gate's verdict changes, the graph does not.
    claimed = {str(r["counters"]["claimed_nodes"]) for r in good}
    if len(claimed) > 1:
        print(f"FAIL: claimed_nodes moved across configs ({claimed}); the graph must not change")
        ok = False
    # And the EP must still offer the island in every configuration — the sole-island override
    # exists precisely so that a rejected sole island is not silently handed back to the CPU.
    offered = {str(r["counters"]["islands_offered"]) for r in good}
    if offered != {str(len(good))} and len(offered) > 1:
        print(f"WARN: islands_offered varied across configs: {offered}")

    print("\nVERDICT:", "PASS" if ok else "FAIL(gate observables did not behave as specified)")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", default=None)
    ap.add_argument("--only", default=None, help="substring filter over config names")
    args = ap.parse_args()
    if args.child:
        counters_path = pathlib.Path(os.environ["ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE"])
        run_child(args.child, counters_path)
        return 0
    return run_parent(args.only)


if __name__ == "__main__":
    raise SystemExit(main())
