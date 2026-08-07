"""Real-model latency: the models a user actually runs, on a clock, with a reference arm.

WHY THIS MODULE EXISTS (issue #56)
==================================
Every real-model instrument in this repository was, until this file, **deliberately clock-free**
(`bench/exec_census.py`, `bench/island_attribution.py`,
`bench/results/probe_real_matmulnbits_rows.py` — none of them contains a timer), and every
clocked instrument was **synthetic** (`bench/results/ab_row_tile.py` at K=N=4096 f32,
`bench/bench.py`'s op cases). That is a coherent design and it is exactly the hole #56 names:
*"I want to see some perf numbers on real models."*

This module is the shared, testable half of that instrument. It holds the parts that can be
checked without a GPU — model identity and provenance, feed construction, the arm definitions,
the statistics, the throughput arithmetic and the equivalence verdicts — so that
`bench/results/probe_real_model_latency.py` is a driver rather than a pile of logic that only
runs when a device is present. `bench/test_real_model.py` tests everything here.

WHAT IT REFUSES
===============
The refusals are inherited rather than invented; each names a wrong number this project has
actually produced:

* **No hardcoded Foundry path.** Resolution goes through `foundry_discovery.resolve_model_path`
  (issue #11/#19: the cache moved from `…-cuda-gpu/` to `…-cuda-gpu-2/v2/` with no code change on
  either side). A declared model that is absent is a **failure**, never a silent skip.
* **No unattributed device.** The device is pinned and named in the artifact
  (`bench.bench.select_device`), because two GPUs on one desk are not interchangeable.
* **No unverified arm.** Every arm that is compared is first checked against a CPU reference
  **in the same process, on the same session objects that are then timed** — the `argmax 0`
  defect (161 nodes dispatched, `compute_failures: 0`, all-zero logits) is why.
* **No single ms.** A median without a spread is a rumour; `latency_stats` always returns the
  distribution, and `tokens_per_second` is derived from the median rather than from a best run.
* **No unlabelled arm ordering.** Arms alternate per repeat. A fixed order produced a spurious
  0.905x at M=1 in `ab_row_tile.py`, on a shape where both arms bind *identical* SPIR-V.

WHAT IT WILL NOT LET YOU CONCLUDE
=================================
`M = 1` prefill tiled-vs-untiled is the **null control**, not a result: both arms resolve
`QB_ROWS = 1` and build the identical pipeline, so its spread is the harness's noise floor and
no ratio narrower than it is a measurement.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import os
import statistics
import sys
from pathlib import Path

_BENCH = Path(__file__).resolve().parent
_ROOT = _BENCH.parent
for _p in (str(_BENCH), str(_ROOT), str(_ROOT / "rust" / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

#: Schema of the artifact `probe_real_model_latency.py` writes. Bumped when a *meaning* changes,
#: never when a field is added — readers key off it to know what a row is a row of.
SCHEMA = "real_model_latency/1"

#: The row-tile kill switch. Also the A/B control: pinned to "1" the tiled path cannot be
#: selected at all, so the two arms differ by exactly one specialisation constant.
ROWS_ENV = "ONNXRUNTIME_EP_VULKAN_GEMV_MAX_ROWS"
EP_NAME = "VulkanExecutionProvider"
CPU_EP = "CPUExecutionProvider"

# --------------------------------------------------------------------------------------------
# Model identity and provenance
# --------------------------------------------------------------------------------------------


class ModelUnavailable(RuntimeError):
    """A declared model does not resolve on this machine.

    Raised rather than skipped: #56 asks for numbers on *real models*, and a lane that quietly
    drops a model when its cache entry moves produces a result that reads as complete while
    covering less than it claims.
    """


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    """One real model this lane benchmarks, and how to find it without guessing.

    ``resolver`` is either ``"foundry"`` (goes through the Foundry cache manifest) or
    ``"repo-cache"`` (the pinned download cache `rust/modelrunner` already uses, whose sha256 is
    recorded in `bench/results/rust-model-runner/<key>.json`). No other resolution exists — a
    literal path in a benchmark is a guess about a version.
    """

    key: str
    family: str
    resolver: str
    #: Foundry identity (resolver == "foundry")
    variant_name: str = ""
    execution_provider: str = ""
    onnx_filename: str = ""
    download_alias: str = ""
    #: repo-cache identity (resolver == "repo-cache")
    cache_filename: str = ""
    #: Where the recorded provenance for this model lives, relative to `bench/results/`.
    recorded_provenance: str = ""
    note: str = ""


PHI35 = ModelSpec(
    key="phi-3.5-mini-instruct-cuda-int4-rtn-block-32",
    family="phi3",
    resolver="foundry",
    variant_name="Phi-3.5-mini-instruct-cuda-gpu",
    execution_provider="CUDAExecutionProvider",
    onnx_filename="phi-3.5-mini-instruct-cuda-int4-rtn-block-32.onnx",
    download_alias="phi-3.5-mini",
    recorded_provenance="rust-model-runner/phi-3.5-mini.json",
    note="Foundry Local Phi-3.5-mini-instruct, int4 RTN block-32, fp16 activations. "
         "32 layers, 32 KV heads, head dim 96, 64 KV inputs and 64 KV outputs.",
)

MOBILENETV2 = ModelSpec(
    key="mobilenetv2-12",
    family="cnn",
    resolver="repo-cache",
    cache_filename="mobilenetv2-12.onnx",
    recorded_provenance="rust-model-runner/mobilenetv2-12.json",
    note="ONNX model zoo MobileNetV2-12, f32. Broad-EP-overhead arm: dense elementwise/Conv "
         "graph with a single free batch dimension, so it prices boundary crossing on a model "
         "whose arithmetic has nothing to do with MatMulNBits.",
)

MODELS = {m.key: m for m in (PHI35, MOBILENETV2)}

#: Where `rust/modelrunner` puts pinned downloads. Read from the environment first so a machine
#: that keeps its cache elsewhere is not silently missed.
REPO_CACHE_ENV = "ONNXRUNTIME_EP_VULKAN_MODEL_CACHE"


def repo_cache_dir() -> Path:
    override = os.environ.get(REPO_CACHE_ENV)
    if override:
        return Path(override)
    return Path.home() / ".cache" / "onnxruntime-ep-vulkan" / "models"


def sha256_file(path: "Path | str") -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def recorded_sha256(spec: ModelSpec, results_dir: "Path | None" = None) -> "str | None":
    """The sha256 a previous, independent tool recorded for this model, or ``None``.

    Deliberately read from `rust/modelrunner`'s artifacts rather than from a constant in this
    file: a hash this module both writes and checks proves nothing.
    """
    import json

    base = Path(results_dir) if results_dir else (_BENCH / "results")
    p = base / spec.recorded_provenance
    if not spec.recorded_provenance or not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("onnx_sha256")
    except Exception:
        return None


def resolve_model(spec: ModelSpec, *, results_dir: "Path | None" = None) -> dict:
    """Resolve one model to a path plus a full provenance block, or raise.

    The returned dict is what lands in the artifact. ``agrees_with_recorded_provenance`` is
    reported rather than enforced — a model that legitimately moved forward should be visible as
    a disagreement, not as a crash — but it is a *field on the face of the result*, so a reader
    cannot miss it.
    """
    if spec.resolver == "foundry":
        import foundry_discovery as fd

        fspec = fd.FoundryModelSpec(
            variant_name=spec.variant_name,
            execution_provider=spec.execution_provider,
            onnx_filename=spec.onnx_filename,
            download_alias=spec.download_alias,
        )
        try:
            path = Path(fd.resolve_model_path(fspec))
        except Exception as exc:
            raise ModelUnavailable(
                f"{spec.key}: Foundry resolution failed: {exc}. "
                f"Fetch it with `foundry model download {spec.download_alias}`. "
                f"This lane fails rather than skipping: #56 asks for numbers on real models."
            ) from exc
        provenance = "foundry-resolved"
    elif spec.resolver == "repo-cache":
        path = repo_cache_dir() / spec.cache_filename
        if not path.is_file():
            raise ModelUnavailable(
                f"{spec.key}: {path} is absent. Fetch it with "
                f"`cargo run -p ort-model-runner -- --model {spec.key}` (which pins and hashes "
                f"it), or set {REPO_CACHE_ENV} to the directory that holds it."
            )
        provenance = "pinned-cache"
    else:  # pragma: no cover - guarded by MODELS
        raise ValueError(f"unknown resolver {spec.resolver!r}")

    if not path.is_file():
        raise ModelUnavailable(f"{spec.key}: resolver returned {path}, which does not exist")

    digest = sha256_file(path)
    recorded = recorded_sha256(spec, results_dir)
    external = external_data_provenance(path)
    return {
        "key": spec.key,
        "family": spec.family,
        "path": str(path),
        "sha256": digest,
        "bytes": path.stat().st_size,
        "provenance": provenance,
        "resolver": spec.resolver,
        "recorded_sha256": recorded,
        "agrees_with_recorded_provenance": (recorded == digest) if recorded else None,
        "external_data": external,
        "weights_bytes": sum(f["bytes"] for f in external["files"]),
        "note": spec.note,
    }


def external_data_provenance(path: Path) -> dict:
    """Hash the tensors the `.onnx` file does *not* contain.

    The Foundry Phi-3.5 graph is 26 MB and its weights are 2.29 GB in a sibling `.data` file.
    A provenance block that hashes only the `.onnx` therefore identifies the *topology* and
    says nothing about the numbers the benchmark actually multiplies — swap the `.data` and the
    recorded hash is unchanged. This is not hypothetical for a quantised model shipped by a
    downloader that can re-materialise weights independently of the graph.

    The file list comes from the graph's own `external_data` locations, not from a `.data`
    suffix guess, so a model that names its blob something else is still covered.
    """
    rec = {"scanned": False, "files": [], "reason": None}
    try:
        import onnx
    except ImportError:  # pragma: no cover - onnx is present in this repo's venv
        rec["reason"] = "onnx package not importable; external weights are UNHASHED"
        return rec
    try:
        model = onnx.load(str(path), load_external_data=False)
    except Exception as exc:  # pragma: no cover - a corrupt graph is its own failure
        rec["reason"] = f"could not parse graph: {exc}; external weights are UNHASHED"
        return rec
    locations = set()
    for init in model.graph.initializer:
        if init.HasField("data_location") and init.data_location == onnx.TensorProto.EXTERNAL:
            for kv in init.external_data:
                if kv.key == "location":
                    locations.add(kv.value)
    rec["scanned"] = True
    for loc in sorted(locations):
        blob = path.parent / loc
        if not blob.is_file():
            rec["files"].append({"location": loc, "bytes": 0, "sha256": None,
                                 "missing": True})
            continue
        rec["files"].append({"location": loc, "bytes": blob.stat().st_size,
                             "sha256": sha256_file(blob)})
    if not locations:
        rec["reason"] = "graph carries no external initializers; the .onnx hash covers the weights"
    return rec


# --------------------------------------------------------------------------------------------
# Cases — what is fed, and what the numbers may be divided by
# --------------------------------------------------------------------------------------------

PHI35_LAYERS = 32
PHI35_KV_HEADS = 32
PHI35_HEAD_DIM = 96
PHI35_F16_BYTES = 2
#: 32 layers x 2 (key,value) x 32 heads x 96 dim x 2 B. Niobe's slope, exact to the byte
#: (`docs/PERF.md` §21): the host KV round trip costs this many bytes per past token, each way.
PHI35_BYTES_PER_PAST_TOKEN = (
    PHI35_LAYERS * 2 * PHI35_KV_HEADS * PHI35_HEAD_DIM * PHI35_F16_BYTES
)

#: Deterministic and fixed: the KV cache content must be byte-identical across arms or the arms
#: are not comparable, and it must be identical across runs or two runs are not comparable.
KV_SEED = 0x5EED0056


@dataclasses.dataclass(frozen=True)
class Case:
    """One (model, phase, size) point of the matrix.

    ``tokens`` is what a tokens/s figure may divide by, and it is **not** always ``m``: a decode
    step emits one token no matter how large its KV cache is, and a MobileNetV2 batch emits no
    tokens at all (``tokens = None``, and the throughput is reported as images/s).
    """

    model_key: str
    phase: str          # "prefill" | "decode" | "batch"
    m: int              # sequence positions fed this run (or batch size, for "batch")
    past: int = 0       # past_sequence_length; decode requires > 0 for at least one point
    tokens: "int | None" = None
    unit: str = "tokens"

    @property
    def label(self) -> str:
        if self.phase == "batch":
            return f"{self.model_key}/batch/N{self.m}"
        return f"{self.model_key}/{self.phase}/M{self.m}/past{self.past}"


def phi35_cases(prefill_m, decode_past) -> "list[Case]":
    """Prefill sweep (empty cache) plus decode points (non-empty cache).

    The decode arm exists because **every** wall-clock measurement in this repository before
    #56 ran at `past_sequence_length == 0`. Real decode runs in the hundreds to thousands, where
    the KV traffic — `PHI35_BYTES_PER_PAST_TOKEN` bytes per past token, each way — and not
    MatMulNBits may dominate. A prefill-only benchmark of a decoder model measures the case the
    user spends the least time in.
    """
    cases = [Case(PHI35.key, "prefill", int(m), 0, tokens=int(m)) for m in prefill_m]
    cases += [Case(PHI35.key, "decode", 1, int(p), tokens=1) for p in decode_past]
    return cases


def mobilenet_cases(batch) -> "list[Case]":
    return [Case(MOBILENETV2.key, "batch", int(b), 0, tokens=None, unit="images")
            for b in batch]


def phi35_feeds(case: Case, np):
    """Build one feed dict. Pure given ``KV_SEED``; identical bytes for every arm."""
    if case.model_key != PHI35.key:
        raise ValueError(f"{case.model_key} is not {PHI35.key}")
    total = case.past + case.m
    feeds = {
        # Token ids are arbitrary but fixed: a benchmark that feeds random ids feeds a different
        # embedding-gather pattern each run, and the gather is not free.
        "input_ids": np.arange(1, case.m + 1, dtype=np.int64).reshape(1, case.m),
        "attention_mask": np.ones((1, total), dtype=np.int64),
    }
    rng = np.random.default_rng(KV_SEED)
    shape = (1, PHI35_KV_HEADS, case.past, PHI35_HEAD_DIM)
    for layer in range(PHI35_LAYERS):
        for kind in ("key", "value"):
            if case.past == 0:
                arr = np.empty(shape, dtype=np.float16)
            else:
                # Scaled to the magnitude a real cache carries; a cache of N(0,1) would push the
                # fp16 softmax into a regime the model never sees.
                arr = (rng.standard_normal(shape) * 0.02).astype(np.float16)
            feeds[f"past_key_values.{layer}.{kind}"] = arr
    return feeds


def mobilenet_feeds(case: Case, np):
    rng = np.random.default_rng(KV_SEED)
    return {"input": rng.standard_normal((case.m, 3, 224, 224), dtype=np.float32)}


def build_feeds(case: Case, np):
    if case.model_key == PHI35.key:
        return phi35_feeds(case, np)
    if case.model_key == MOBILENETV2.key:
        return mobilenet_feeds(case, np)
    raise ValueError(f"no feed builder for {case.model_key}")


def feeds_digest(feeds: dict) -> str:
    """A hash of the exact bytes fed, so two runs can be shown to have fed the same thing.

    Names *and* bytes: a feed dict with the right values under the wrong keys is a different
    input, and hashing values alone would call the two identical.
    """
    h = hashlib.sha256()
    for name in sorted(feeds):
        arr = feeds[name]
        h.update(name.encode("utf-8"))
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(str(arr.shape).encode("utf-8"))
        h.update(memoryview(arr).tobytes() if arr.size else b"")
    return h.hexdigest()


def kv_round_trip_bytes(case: Case) -> "dict | None":
    """Host bytes the KV cache costs this case, each way. ``None`` for non-KV models.

    This is a *model* of the traffic, derived from the shapes in the graph, not a measurement —
    and it is labelled as such in the artifact. It exists so the decode rows can be read against
    a bandwidth number instead of being a mystery.
    """
    if case.model_key != PHI35.key:
        return None
    upload = case.past * PHI35_BYTES_PER_PAST_TOKEN
    readback = (case.past + case.m) * PHI35_BYTES_PER_PAST_TOKEN
    return {
        "bytes_per_past_token": PHI35_BYTES_PER_PAST_TOKEN,
        "past_upload_bytes": upload,
        "present_readback_bytes": readback,
        "total_host_bytes": upload + readback,
        "derivation": "shape model (layers x 2 x kv_heads x head_dim x 2 B), not a measurement",
    }


# --------------------------------------------------------------------------------------------
# Arms
# --------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Arm:
    """One thing being compared. ``env`` is applied before the session is built, never after.

    Ordering is load-bearing and cost this project a run: the row tile is chosen at *translate*
    time, so a session built before the variable was set keeps the arm it was built with, and
    `ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY` set after `register_execution_provider_library`
    registers no allocator at all.
    """

    name: str
    providers: "tuple[str, ...]"
    env: "tuple[tuple[str, str], ...]" = ()
    role: str = "arm"

    def apply_env(self, environ) -> dict:
        """Set this arm's env, returning the previous values so a caller can restore them."""
        prev = {}
        for k, v in self.env:
            prev[k] = environ.get(k)
            environ[k] = v
        return prev


VULKAN_TILED = Arm("vulkan_tiled", (EP_NAME, CPU_EP), ((ROWS_ENV, "4"),), role="subject")
VULKAN_UNTILED = Arm("vulkan_untiled", (EP_NAME, CPU_EP), ((ROWS_ENV, "1"),), role="kill-switch")
CPU_ARM = Arm("cpu", (CPU_EP,), (), role="baseline")
ARMS = (VULKAN_TILED, VULKAN_UNTILED, CPU_ARM)


def arm_order(arms, repeat: int):
    """Alternate arm order per repeat.

    A fixed order does not cancel: GPU clock and page cache both drift monotonically inside a
    repeat, so whichever arm always runs last inherits that drift as a systematic bias. This is
    not a hypothetical — `ab_row_tile.py`'s first version reported a 0.905x "slowdown" at M=1,
    a shape where both arms bind identical SPIR-V, purely from the order.
    """
    seq = list(arms)
    if repeat % 2:
        seq.reverse()
    return seq


def is_null_control(case: Case) -> bool:
    """True where ``vulkan_tiled`` and ``vulkan_untiled`` build the *identical* pipeline.

    `ops::quant::gemv_tile` returns ``rows = 1`` whenever ``M == 1`` regardless of the cap, so
    at M=1 the two arms differ in no specialisation constant at all. Their measured ratio is
    therefore a direct read of this harness's noise floor and is not a result. Decode is M=1 by
    definition, so every decode row is also a null control for the row tile — which is exactly
    why decode needs the *CPU* arm to say anything.
    """
    return case.m == 1


# --------------------------------------------------------------------------------------------
# Statistics and throughput
# --------------------------------------------------------------------------------------------


def latency_stats(samples: "list[float]") -> dict:
    """Full distribution, never a single number.

    ``min`` is reported but is deliberately *not* the headline: on a shared, boost-clocked box
    the minimum is the run that got the machine to itself, which is the case a user least often
    experiences. The median is the headline and the spread travels with it.
    """
    xs = sorted(float(x) for x in samples)
    if not xs:
        return {"n": 0}
    med = statistics.median(xs)
    mad = statistics.median([abs(x - med) for x in xs])
    return {
        "n": len(xs),
        "median_ms": med,
        "mean_ms": statistics.fmean(xs),
        "min_ms": xs[0],
        "max_ms": xs[-1],
        "p05_ms": _quantile(xs, 0.05),
        "p95_ms": _quantile(xs, 0.95),
        "mad_ms": mad,
        "stdev_ms": statistics.stdev(xs) if len(xs) > 1 else 0.0,
        "rsd": (statistics.stdev(xs) / med) if len(xs) > 1 and med else 0.0,
    }


def _quantile(ordered: "list[float]", q: float) -> float:
    if not ordered:
        return float("nan")
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[int(pos)]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def throughput(case: Case, median_ms: float) -> "dict | None":
    """tokens/s (or images/s) from the median latency.

    Prefill divides by ``m`` because the run consumed ``m`` tokens; decode divides by 1 because
    a decode step emits one token however long its cache is. Dividing decode by ``past`` would
    manufacture a throughput that grows with the cache — the number getting better as the model
    gets slower.
    """
    if median_ms is None or median_ms <= 0:
        return None
    if case.tokens is not None:
        return {"unit": f"{case.unit}/s", "value": case.tokens * 1000.0 / median_ms,
                "divisor": case.tokens, "divisor_meaning": f"{case.unit} advanced by one run"}
    return {"unit": f"{case.unit}/s", "value": case.m * 1000.0 / median_ms,
            "divisor": case.m, "divisor_meaning": f"{case.unit} in one batch"}


def paired_ratios(a: "list[float]", b: "list[float]") -> dict:
    """Per-repeat ratios of two arms measured adjacent in time.

    Paired, because the two arms of a repeat share whatever the machine was doing; the *median
    of the per-repeat ratios* is the estimate that survives one disturbed repeat, and the
    [min, max] is the honest error bar. A ratio of medians would hide a repeat where one arm was
    displaced.
    """
    ratios = [x / y for x, y in zip(a, b) if y]
    if not ratios:
        return {"n": 0}
    return {
        "n": len(ratios),
        "median": statistics.median(ratios),
        "min": min(ratios),
        "max": max(ratios),
        "ratios": [round(r, 6) for r in ratios],
    }


def exceeds_noise_floor(ratio: float, floor: dict) -> "bool | None":
    """Is a ratio bigger than the identical-pipeline control's own spread?

    Returns ``None`` when there is no control to read it against, which is a different state
    from "no" and must not collapse into it.
    """
    if not floor or floor.get("n", 0) == 0:
        return None
    lo, hi = floor.get("min"), floor.get("max")
    if lo is None or hi is None:
        return None
    return bool(ratio > hi or ratio < lo)


# --------------------------------------------------------------------------------------------
# Equivalence — every compared arm, every case
# --------------------------------------------------------------------------------------------

#: fp16 logits over a 32-layer decoder. `max_rel` over a GEMM output is a **cancellation meter,
#: not an accuracy meter** (§25.3), so the gate is argmax + top-k agreement plus an absolute
#: bound, and the relative figure is reported without being gated on.
PHI35_TOP_K = 10
#: The logit budget is a **fraction of the reference's own scale**, not a constant. A constant
#: was the first form and it is arbitrary: 0.5 is 3.8% of the logit scale at an empty cache and
#: 2.1% at past=1024, so a fixed number silently tightens as the cache grows. What fp16 supports
#: is relative: the error of a length-`K` fp16 accumulation grows like `sqrt(K) * 2^-11 * scale`,
#: and Phi-3.5 decode reduces over 3072-wide projections through 32 layers, so `sqrt(3072*32) *
#: 2^-11 ~= 0.15` — 15% of scale is the *theoretical* envelope. This budget is 3x tighter than
#: that envelope and still admits what the device actually produces.
#:
#: MEASURED 2026-08-06 on this box: the CPU EP is **bit-identical to itself** at
#: `intra_op_num_threads` 1, 2 and 4 (`_issue56_scratch/reorder_control.py`), so there is no
#: reference-side reordering noise to calibrate against and this bound is argued from fp16's
#: numerics, not fitted to the failure.
PHI35_LOGIT_SCALE_FRACTION = 0.05
#: What a decoder's logits are *for* is the next-token distribution. Two logit vectors that
#: disagree by 0.58 at scale 23 but induce the same distribution to within this bound are the
#: same model behaviour; two that agree in argmax while moving a probability by 0.3 are not.
PHI35_MAX_PROB_DELTA = 0.02
#: f32 CNN, fifty-plus Conv/BatchNorm/Clip layers. Same tolerance `rust/modelrunner` justifies
#: for this exact model in `bench/results/rust-model-runner/mobilenetv2-12.json`, borrowed
#: rather than re-invented so the two tools cannot disagree about what "AGREE" means.
MOBILENET_RTOL = 1e-2
MOBILENET_ATOL = 1e-4

MATCH = "MATCH"
DIVERGENT = "DIVERGENT"
UNMEASURED = "UNMEASURED"


def softmax(x, np):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def classify_logits(candidate, reference, np, *, top_k: int = PHI35_TOP_K,
                    scale_fraction: float = PHI35_LOGIT_SCALE_FRACTION,
                    max_prob_delta: float = PHI35_MAX_PROB_DELTA) -> dict:
    """Compare a decoder's logits against the CPU reference for the *same* run.

    Four clauses, all required: argmax agreement, top-k agreement, an absolute bound scaled to
    the reference's own logit magnitude, and a bound on the induced softmax distribution. The
    last one is the clause that means something: a sampler reads probabilities, so a difference
    that does not move a probability by more than `max_prob_delta` has not changed the model's
    behaviour, whatever it did to the raw logits. An all-zero reference is reported as its own
    state: the `argmax 0` defect produced two all-zero tensors that agreed perfectly.
    """
    cand = np.asarray(candidate, dtype=np.float32)
    ref = np.asarray(reference, dtype=np.float32)
    out = {
        "shape_candidate": list(cand.shape),
        "shape_reference": list(ref.shape),
        "top_k": top_k,
    }
    if cand.shape != ref.shape:
        out["verdict"] = DIVERGENT
        out["reason"] = "shape mismatch"
        return out
    # The last position is the one a sampler reads; earlier positions of a prefill are compared
    # too, but the argmax gate is on the position the model is actually asked about.
    last_c = cand.reshape(-1, cand.shape[-1])[-1]
    last_r = ref.reshape(-1, ref.shape[-1])[-1]
    out["reference_all_zero"] = bool(not np.any(ref))
    out["argmax_candidate"] = int(np.argmax(last_c))
    out["argmax_reference"] = int(np.argmax(last_r))
    k = min(top_k, last_r.shape[-1])
    topk_c = set(np.argsort(-last_c)[:k].tolist())
    topk_r = set(np.argsort(-last_r)[:k].tolist())
    out["topk_overlap"] = len(topk_c & topk_r)
    diff = np.abs(cand - ref)
    out["max_abs"] = float(diff.max()) if diff.size else 0.0
    out["reference_scale"] = float(np.abs(ref).max()) if ref.size else 0.0
    out["abs_budget"] = scale_fraction * out["reference_scale"]
    denom = np.maximum(np.abs(ref), 1e-6)
    out["max_rel"] = float((diff / denom).max()) if diff.size else 0.0
    out["max_rel_note"] = ("relative error over logits is a cancellation meter, not an accuracy "
                           "meter; reported, not gated on")
    p_c = softmax(last_c, np)
    p_r = softmax(last_r, np)
    out["max_prob_delta"] = float(np.abs(p_c - p_r).max())
    out["top1_prob_reference"] = float(p_r.max())
    out["top1_prob_candidate"] = float(p_c[out["argmax_reference"]])
    ok = (out["argmax_candidate"] == out["argmax_reference"]
          and out["topk_overlap"] == k
          and out["max_abs"] <= out["abs_budget"]
          and out["max_prob_delta"] <= max_prob_delta
          and not out["reference_all_zero"])
    out["verdict"] = MATCH if ok else DIVERGENT
    out["gate"] = (f"argmax equal AND top-{k} identical AND max_abs <= {scale_fraction} of the "
                   f"reference's own logit scale AND max softmax probability delta <= "
                   f"{max_prob_delta} AND reference not all-zero")
    return out


def classify_tensor(candidate, reference, np, *, rtol: float = MOBILENET_RTOL,
                    atol: float = MOBILENET_ATOL) -> dict:
    """Elementwise combined tolerance: ``|diff| <= atol + rtol * |ref|`` for every element.

    Elementwise and not aggregate. An aggregate ``max_rel <= rtol OR max_abs <= atol`` has a
    hole a planted error walks straight through: a large absolute error on a *small* element
    passes the relative clause while a large relative error on a *large* element passes the
    absolute one, so the two clauses cover for each other. `rust/modelrunner`'s recorded verdict
    for this model uses the aggregate form; this is strictly stricter, so a MATCH here implies a
    MATCH there and the two tools cannot disagree in the dangerous direction.
    """
    cand = np.asarray(candidate, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    out = {"shape_candidate": list(cand.shape), "shape_reference": list(ref.shape),
           "rtol": rtol, "atol": atol}
    if cand.shape != ref.shape:
        out["verdict"] = DIVERGENT
        out["reason"] = "shape mismatch"
        return out
    diff = np.abs(cand - ref)
    out["max_abs"] = float(diff.max()) if diff.size else 0.0
    out["max_rel"] = (float((diff / np.maximum(np.abs(ref), 1e-12)).max())
                      if diff.size else 0.0)
    out["reference_all_zero"] = bool(not np.any(ref))
    offenders = diff > (atol + rtol * np.abs(ref)) if diff.size else np.zeros(0, dtype=bool)
    out["elements_outside_tolerance"] = int(offenders.sum())
    # Per-row argmax: for a classifier the label is the answer, and a tolerance that passes while
    # the predicted class moves is not a useful tolerance.
    if cand.ndim == 2 and cand.shape[-1] > 1:
        out["argmax_agreement"] = int((np.argmax(cand, axis=-1) == np.argmax(ref, axis=-1)).sum())
        out["argmax_rows"] = int(cand.shape[0])
    ok = (out["elements_outside_tolerance"] == 0
          and not out["reference_all_zero"]
          and out.get("argmax_agreement", out.get("argmax_rows", 0)) == out.get("argmax_rows", 0))
    out["verdict"] = MATCH if ok else DIVERGENT
    out["gate"] = (f"every element |diff| <= {atol} + {rtol}*|ref| AND every row's argmax "
                   f"agrees AND the reference is not all-zero")
    return out


#: fp16 has ~2^-10 relative resolution, so an absolute difference of one ULP at the tensor's own
#: largest magnitude is not an error — it is the representation. `KV_ULP_BUDGET` ULPs of the
#: reference's own scale is the absolute floor; `KV_REL_TOL` is the relative slope.
FP16_EPS = 2.0 ** -10
KV_ULP_BUDGET = 16.0
KV_REL_TOL = 5e-2
#: An element may sit above the floor without being evidence of a defect — fp16 accumulation
#: over a 3072-wide reduction lands one element in ~400,000 a third of a floor above it. It may
#: **not** sit arbitrarily far above it. `KV_GROSS_MULTIPLE` floors is the point past which no
#: rounding story survives and the tensor is DIVERGENT on that element alone, however few there
#: are; `KV_MARGINAL_FRACTION` bounds how many elements may occupy the band in between. Without
#: the ceiling this would be a fudge — with it, the planted-error control still fails, because a
#: planted 7.0 at scale 4 is 50 floors out, not 1.3.
KV_GROSS_MULTIPLE = 8.0
KV_MARGINAL_FRACTION = 1e-4
#: Elements below this fraction of the tensor's RMS are excluded from the *reported* relative
#: figure. Borrowed verbatim from §25.3's discipline: relative error over an accumulation output
#: is a **cancellation meter, not an accuracy meter**, and near-zero elements make it grow with
#: the number of terms summed for a kernel that is exactly as accurate. The first version of this
#: gate had no such restriction and called every Vulkan arm DIVERGENT at `max_rel = 2.73` on a
#: tensor whose largest absolute error was one fp16 ULP.
KV_SIGNAL_FRACTION = 0.1


def classify_activation(candidate, reference, np, *, rel_tol: float = KV_REL_TOL,
                        ulp_budget: float = KV_ULP_BUDGET,
                        signal_fraction: float = KV_SIGNAL_FRACTION,
                        gross_multiple: float = KV_GROSS_MULTIPLE,
                        marginal_fraction: float = KV_MARGINAL_FRACTION) -> dict:
    """Compare an fp16 activation tensor (a `present.*` KV block) against the CPU reference.

    Three bands, elementwise::

        clean:    |diff| <= floor,             floor = ulp_budget*fp16_eps*max|ref| + rel_tol*|ref|
        marginal: floor < |diff| <= gross_multiple * floor
        gross:    |diff| >  gross_multiple * floor

    Any gross element is DIVERGENT on its own. Marginal elements are tolerated only up to
    `marginal_fraction` of the tensor, so a systematic small error across the tensor still fails
    while fp16's one-element-in-400k accumulation tail does not.

    The floor is a **sum, not an OR**: an aggregate ``max_rel <= rtol OR max_abs <= atol`` lets a
    large absolute error on a small element pass under the relative clause, which is exactly how
    a planted-error control fell through the first version of this function.

    ``max_rel_all`` and ``max_rel_signal`` are reported; only the band structure decides.
    """
    cand = np.asarray(candidate, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    out = {"shape_candidate": list(cand.shape), "shape_reference": list(ref.shape),
           "rel_tol": rel_tol, "ulp_budget": ulp_budget,
           "signal_fraction": signal_fraction, "gross_multiple": gross_multiple,
           "marginal_fraction": marginal_fraction}
    if cand.shape != ref.shape:
        out["verdict"] = DIVERGENT
        out["reason"] = "shape mismatch"
        return out
    if ref.size == 0:
        # An empty tensor agrees with an empty tensor, and that is a real state (a decode step
        # with an empty cache emits `present` of extent 1, never 0) — but say so on the face of
        # the record rather than letting a vacuous MATCH look like a checked one.
        out["verdict"] = MATCH
        out["empty"] = True
        return out
    diff = np.abs(cand - ref)
    absref = np.abs(ref)
    scale = float(absref.max())
    rms = float(np.sqrt(np.mean(ref ** 2)))
    out["max_abs"] = float(diff.max())
    out["reference_scale"] = scale
    out["reference_rms"] = rms
    out["abs_floor"] = ulp_budget * FP16_EPS * scale
    signal = absref >= signal_fraction * rms
    out["signal_elements"] = int(signal.sum())
    out["max_rel_signal"] = (float((diff[signal] / absref[signal]).max())
                             if signal.any() and absref[signal].min() > 0 else 0.0)
    out["max_rel_all"] = float((diff / np.maximum(absref, 1e-12)).max())
    out["max_rel_all_note"] = ("relative error over every element is a cancellation meter; "
                               "reported, not gated on")
    out["reference_all_zero"] = bool(not np.any(ref))
    floor = out["abs_floor"] + rel_tol * absref
    offenders = diff > floor
    gross = diff > gross_multiple * floor
    out["elements_outside_tolerance"] = int(offenders.sum())
    out["elements_gross"] = int(gross.sum())
    out["elements_total"] = int(ref.size)
    out["marginal_fraction_observed"] = float(offenders.sum()) / float(ref.size)
    out["worst_floor_multiple"] = (float((diff / np.maximum(floor, 1e-300)).max())
                                   if diff.size else 0.0)
    ok = (out["elements_gross"] == 0
          and out["marginal_fraction_observed"] <= marginal_fraction
          and not out["reference_all_zero"])
    out["verdict"] = MATCH if ok else DIVERGENT
    out["gate"] = (f"no element above {gross_multiple}x the floor "
                   f"({ulp_budget} fp16 ULP of the reference's own scale + {rel_tol}*|ref|), "
                   f"at most {marginal_fraction} of elements above the floor at all, "
                   f"AND the reference is not all-zero")
    return out


def classify_outputs(case: Case, candidate_outputs, reference_outputs, np) -> dict:
    """Verdict for one arm on one case. The *first* output is the task-relevant one.

    Phi-3.5's `logits` and MobileNetV2's `output` are both output 0. The KV outputs are compared
    too, under the tensor gate, because a decode step whose logits agree but whose `present`
    cache is wrong produces a *correct first token and a wrong sequence* — a failure that a
    single-step comparison would otherwise call MATCH.
    """
    if len(candidate_outputs) != len(reference_outputs):
        return {"verdict": DIVERGENT, "reason": "output count mismatch",
                "n_candidate": len(candidate_outputs), "n_reference": len(reference_outputs)}
    if case.model_key == PHI35.key:
        primary = classify_logits(candidate_outputs[0], reference_outputs[0], np)
    else:
        primary = classify_tensor(candidate_outputs[0], reference_outputs[0], np)
    worst_kv = None
    kv_divergent = 0
    for c, r in zip(candidate_outputs[1:], reference_outputs[1:]):
        v = classify_activation(c, r, np)
        if v["verdict"] == DIVERGENT:
            kv_divergent += 1
            if worst_kv is None or v.get("max_abs", 0) > worst_kv.get("max_abs", 0):
                worst_kv = v
    verdict = MATCH if primary["verdict"] == MATCH and kv_divergent == 0 else DIVERGENT
    return {
        "verdict": verdict,
        "primary": primary,
        "secondary_outputs": len(candidate_outputs) - 1,
        "secondary_divergent": kv_divergent,
        "worst_secondary": worst_kv,
        "secondary_note": ("KV/present outputs compared under `classify_activation` — a floor "
                           "scaled to the tensor's own fp16 resolution, a gross ceiling, and a "
                           "bound on how many elements may sit between them; logits agreement "
                           "alone would not catch a wrong cache"),
    }


def bitwise_identical(a_outputs, b_outputs, np) -> dict:
    """Are two arms' outputs *byte for byte* the same?

    The identical-pipeline null control claims that at `M = 1` the tiled and untiled arms compile
    and bind the same SPIR-V, so their outputs must be bit-identical — not close, identical. A
    latency ratio alone cannot tell a real 1.04x from a scheduling artefact, but a bitwise
    disagreement at `M = 1` would falsify the claim that the pipelines are the same, and that is
    worth stating on the face of the record rather than assuming.
    """
    out = {"outputs": len(a_outputs), "identical": True, "first_difference": None}
    if len(a_outputs) != len(b_outputs):
        out["identical"] = False
        out["first_difference"] = "output count"
        return out
    for i, (a, b) in enumerate(zip(a_outputs, b_outputs)):
        a = np.asarray(a)
        b = np.asarray(b)
        if a.shape != b.shape or a.dtype != b.dtype or not np.array_equal(a, b):
            out["identical"] = False
            out["first_difference"] = f"output[{i}]"
            if a.shape == b.shape and a.dtype == b.dtype:
                out["max_abs"] = float(np.abs(a.astype(np.float64)
                                              - b.astype(np.float64)).max())
            break
    return out


# --------------------------------------------------------------------------------------------
# Diagnostics — "utilize the device" made falsifiable
# --------------------------------------------------------------------------------------------


def dispatch_diagnosis(counters: "dict | None", inference_count: int) -> dict:
    """What the EP actually did, per inference, from the EP's own counters.

    Every figure is per-inference where that is meaningful, because the raw counters are
    process-cumulative and a cumulative counter divided by the wrong denominator is how this
    project once read a 1.78x "improvement" that was really an iteration ratio.
    """
    c = counters or {}
    live = c.get("subgraphs_live")
    disp = c.get("dispatches_executed")
    out = {
        "islands": live,
        "compute_calls": c.get("compute_calls"),
        "dispatches_executed": disp,
        "compute_failures": c.get("compute_failures"),
        "inferences": inference_count,
        "dispatches_per_inference": (disp / inference_count) if disp and inference_count else None,
        "island_crossings_per_inference": live,
        "outputs_host_resident": c.get("outputs_host_resident"),
        "outputs_device_resident": c.get("outputs_device_resident"),
        "note": ("counters are process-cumulative; per-inference figures divide by the number of "
                 "inferences this process ran, never by a repeat count"),
    }
    return out


def fallback_diagnosis(provider_node_counts: "dict | None") -> dict:
    """How much of the graph ORT gave to somebody else.

    `claimed` is what the EP asked for; this is what ORT's own profile says *executed* where.
    They are different questions and the second is the one a latency number is about.
    """
    counts = dict(provider_node_counts or {})
    total = sum(counts.values())
    vk = counts.get(EP_NAME, 0)
    return {
        "provider_node_counts": counts,
        "total_node_executions": total,
        "vulkan_node_executions": vk,
        "cpu_fallback_node_executions": total - vk,
        "vulkan_share": (vk / total) if total else None,
        "witness": "onnxruntime profile (cat=Node, args.provider)",
    }


def bandwidth_proxy(case: Case, median_ms: float) -> "dict | None":
    """Achieved host-transfer bandwidth implied by the KV round trip, if any.

    A *proxy*, and named one: it assumes the modelled bytes actually crossed the link, which is
    true for the shipping (unbound, host-staged) lane and false for the device-resident lane.
    It exists so a decode row that is three times slower than the CPU has a number attached to
    the mechanism, instead of a story.
    """
    kv = kv_round_trip_bytes(case)
    if not kv or not median_ms:
        return None
    gb = kv["total_host_bytes"] / (1 << 30)
    return {
        "modelled_host_gib": gb,
        "implied_gib_per_s": gb / (median_ms / 1000.0),
        "assumes": "the shipping host-staged lane; false if outputs are bound device-resident",
    }
