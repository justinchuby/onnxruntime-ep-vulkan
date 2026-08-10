"""ResNet-50 on the Vulkan EP, against ORT's CUDA EP, with the CPU EP as the reference.

WHY THIS MODULE EXISTS (issue #122)
===================================
Issue #122 asks three questions — *what are the ResNet numbers, how do they compare with the
CUDA EP, and what are the gaps.* Before this file the repository could answer none of them:

* Every timed real-model arm in `bench/real_model.py` is `vulkan_tiled` / `vulkan_untiled` /
  `cpu`. **There is no CUDA arm anywhere in this repository** — `CUDAExecutionProvider` appears
  in `bench/` only as a *string in a Foundry cache path* (`PHI35.execution_provider`, which
  names which Foundry variant to download, not which EP runs it). A comparison against CUDA had
  never been made, so "how does it compare with CUDA" had no answer, not even a bad one.
* The only f32 CNN in the timed matrix is MobileNetV2-12. ResNet-50 is a different shape of
  workload — 53 convolutions with 64..2048 channels and a 16-way residual `Add` chain, against
  MobileNetV2's depthwise-separable stacks — and nothing about one transfers to the other.

This module is the **GPU-free half**: model identity and provenance, the immutable pin, feed
construction, the arm definitions including the CUDA arm, the classification gate for a
classifier's logits, the static support census, and the admissibility rules. It is the same
split `bench/real_model.py` uses, for the same reason: everything checkable without a device is
checked without a device, by `bench/test_resnet.py`, and
`bench/results/probe_resnet_vulkan_cuda.py` is a driver rather than a pile of logic that only
executes when a GPU is free.

WHAT IT REFUSES
===============
Inherited, not invented. Each refusal names a wrong number this project has actually produced:

* **No unpinned model.** Resolution goes through `bench/pinned_bytes.py` — repo, immutable
  40-hex revision, sha256, byte count, external-data scan. A "resnet50.onnx" that is on the
  disk is not evidence that it is *the* ResNet-50 (issue #78).
* **No unattributed device, no unattributed runtime.** The Vulkan device, the CUDA device, the
  driver, the ORT version, the ORT DLL digest and the EP DLL digest are inputs to the
  measurement, and a record missing any of them is not a slightly worse record.
* **No timing without correctness.** Every timed arm is first compared against the CPU EP, in
  the same process, on the same session object that is then timed. The `argmax 0` defect (161
  nodes dispatched, `compute_failures: 0`, all-zero logits) is why.
* **No unlabelled ratio.** A bare "1.8x" does not say which arm is on top.
  `ratio_record` carries `baseline`, `candidate`, and the sentence that fixes the polarity, and
  `bench/test_resnet.py` fails if a ratio is emitted without them.
* **No claim under contention.** This box is shared indefinitely (`docs/PERF.md` §20). The
  quiescence verdict and the foreign-GPU-load disclosure travel *with* every number, and
  `admissibility` downgrades the verdict to `INDETERMINATE` rather than publishing a ratio the
  machine state cannot support.

WHAT IT WILL NOT LET YOU CONCLUDE
=================================
* **One device, one model, one driver is not a parity claim.** `GENERALISATION_LIMIT` is
  emitted into every artifact and is quoted in `docs/PERF.md`: a ResNet-50 reading on one
  RTX A1000 says nothing about any other GPU, vendor, or model.
* **The static census is a MODEL, not a MEASUREMENT.** `support_census` reads the *unoptimised*
  ONNX graph against the shipped op registry. ORT's `ORT_ENABLE_ALL` optimiser folds
  `BatchNormalization` into `Conv` before partitioning, so the graph the EP is offered is not
  the graph on disk. The census is labelled `MODEL` in `PROVENANCE`; the only `MEASUREMENT` of
  partitioning is ORT's own per-node provider attribution from a profiled run.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

_BENCH = Path(__file__).resolve().parent
_ROOT = _BENCH.parent
for _p in (str(_BENCH), str(_ROOT), str(_ROOT / "rust" / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import real_model as rm  # noqa: E402

#: Schema of the artifact `probe_resnet_vulkan_cuda.py` writes. Bumped when a *meaning*
#: changes, never when a field is added.
SCHEMA = "resnet_vulkan_cuda/1"

CUDA_EP = "CUDAExecutionProvider"
EP_NAME = rm.EP_NAME
CPU_EP = rm.CPU_EP

# --------------------------------------------------------------------------------------------
# Model identity — the immutable pin
# --------------------------------------------------------------------------------------------
#
# ResNet-50 v1, opset 12, f32, from the ONNX Model Zoo's own Hugging Face mirror. Chosen over
# the three alternatives that were actually looked at, and the reasons are recorded because
# "we used resnet50.onnx" is the kind of sentence issue #78 exists to stop:
#
#   * `onnxmodelzoo/resnet50-v1-12` (chosen) — the ONNX Model Zoo's validated ResNet-50 v1.
#     Single `.onnx`, no external data, NCHW `[N,3,224,224]` f32 input, opset 12. Every ORT EP
#     on this desk executes it, and it is the ResNet an ONNX-Runtime user would actually run.
#   * `ctuning/mlperf-inference-resnet50-onnx-fp32-imagenet2012-v1.0` — MLPerf's reference
#     ResNet-50 v1.5. Also authoritative, but it is a TensorFlow conversion whose input layout
#     and node naming differ from the zoo model, and nothing in this repository has ever run it.
#   * `Xenova/resnet-50` — a transformers.js re-export of `microsoft/resnet-50`. A re-export of
#     a port; two removes from an authority.
#
# The revision is the Hugging Face commit sha, which is immutable; `main` is not.
RESNET50_PIN = {
    "repo": "onnxmodelzoo/resnet50-v1-12",
    "file": "resnet50-v1-12.onnx",
    "revision": "1f95315d8bd3b3ca2ceabe54d274e0cdf5a83bbe",
    "sha256": "3f03fdef724b22947eed826f1eef1dc5c34151bb4c37d634f1db89dfa2dd1526",
    "pinned_bytes": 102576593,
    "source": "pinned-cache",
    "declared_external_files": 0,
}

#: The exact URL the pinned bytes came from. Recorded, not resolved: nothing in this module
#: reaches the network, and a URL in a benchmark that *is* fetched is a fetch, not a pin.
RESNET50_URL = (
    "https://huggingface.co/onnxmodelzoo/resnet50-v1-12/resolve/"
    "1f95315d8bd3b3ca2ceabe54d274e0cdf5a83bbe/resnet50-v1-12.onnx"
)

RESNET50 = rm.ModelSpec(
    key="resnet50-v1-12",
    family="cnn",
    resolver="pinned",
    cache_filename="resnet50-v1-12-onnx-1f95315d.onnx",
    recorded_provenance="pinned-bytes/resnet50-v1-12.json",
    pin=RESNET50_PIN,
    note=(
        "ONNX Model Zoo ResNet-50 v1 (`resnet50-v1-12.onnx`) at immutable revision "
        "1f95315d8bd3b3ca2ceabe54d274e0cdf5a83bbe. 175 nodes, opset 12, f32, no external data, "
        "input `data` [N,3,224,224] NCHW, output `resnetv17_dense0_fwd` [N,1000] raw logits."
    ),
)

#: The input and output names are part of the model's identity for this lane: a feed dict with
#: the right array under the wrong key is a different input, and a harness that guesses the name
#: silently benchmarks whichever model happens to use the name it guessed.
RESNET50_INPUT = "data"
RESNET50_OUTPUT = "resnetv17_dense0_fwd"
RESNET50_CLASSES = 1000

#: Op-type histogram of the pinned graph, as it is on disk. SPECIFICATION-class about *this
#: file* (it is re-derived from the bytes by `bench/test_resnet.py`, never trusted as a
#: literal), and deliberately not the partition: see `support_census`.
RESNET50_OP_HISTOGRAM = {
    "Conv": 53,
    "BatchNormalization": 53,
    "Relu": 49,
    "Add": 16,
    "MaxPool": 1,
    "GlobalAveragePool": 1,
    "Flatten": 1,
    "Gemm": 1,
}
RESNET50_NODES = 175

#: Stated once, in the artifact, in the doc, and in the PR. One device, one model, one driver.
GENERALISATION_LIMIT = (
    "These readings are one model (ResNet-50 v1 opset 12, f32) on one device (see "
    "`environment.device`), one driver, one ORT build, one day. They do not support any claim "
    "about ResNet on another GPU or vendor, about another model on this GPU, or about the "
    "Vulkan EP versus the CUDA EP in general. A single-device ratio is a reading, not parity."
)


# --------------------------------------------------------------------------------------------
# Cases and feeds
# --------------------------------------------------------------------------------------------

#: Deterministic and fixed, so two runs and two arms are fed byte-identical input. Distinct
#: from `real_model.KV_SEED` so a ResNet feed can never be confused with a Phi feed in a digest.
RESNET_SEED = 0x5EED0122


def resnet_cases(batch) -> "list[rm.Case]":
    """One case per batch size. Refuses rather than coerces, for `minilm_cases`'s reason.

    `int("4")` and `int(4.7)` both succeed and both mean the caller did not say what it meant;
    a batch silently rounded down is a run whose recorded batch is not the one that was fed.
    """
    if isinstance(batch, (str, bytes)) or not hasattr(batch, "__iter__"):
        raise TypeError(f"batch sizes must be an iterable of int, got {type(batch).__name__}")
    out = []
    for b in batch:
        if isinstance(b, bool) or not isinstance(b, int):
            raise TypeError(f"batch size must be a non-bool int, got {b!r}")
        if b < 1:
            raise ValueError(f"batch size must be >= 1, got {b}")
        out.append(rm.Case(RESNET50.key, "batch", b, 0, tokens=None, unit="images"))
    return out


def resnet_feeds(case: rm.Case, np) -> dict:
    """The exact bytes fed to every arm.

    Standard-normal rather than a decoded JPEG on purpose, and the consequence is stated rather
    than hidden: this lane measures **latency and numerical agreement**, not top-1 accuracy on
    ImageNet. A synthetic tensor exercises every kernel at the same shapes and cost as a real
    image — convolution cost does not depend on the values — while removing a decode path and a
    dataset licence from the reproduction steps. Accuracy is not claimed anywhere in this lane
    precisely because this input cannot support it.
    """
    if case.model_key != RESNET50.key:
        raise ValueError(f"resnet_feeds got a {case.model_key} case")
    rng = np.random.default_rng(RESNET_SEED)
    return {RESNET50_INPUT: rng.standard_normal((case.m, 3, 224, 224), dtype=np.float32)}


# --------------------------------------------------------------------------------------------
# Arms — and the first CUDA arm in this repository
# --------------------------------------------------------------------------------------------

#: The correctness reference *and* a timing context row. It is the reference because it is the
#: only arm whose numerics nothing in this repository is trying to change.
CPU_ARM = rm.Arm("cpu", (CPU_EP,), (), role="reference")

#: The subject. CPU is left in the provider list because that is how the EP actually ships: it
#: claims what it can and ORT places the rest. Removing the fallback would measure a
#: configuration no user runs, and would turn every unsupported op into a session-creation
#: failure rather than into the partition cost this lane exists to price.
VULKAN_ARM = rm.Arm("vulkan", (EP_NAME, CPU_EP), (), role="candidate")

#: The comparison ORT users actually have on an NVIDIA card. Same fallback reasoning.
CUDA_ARM = rm.Arm("cuda", (CUDA_EP, CPU_EP), (), role="baseline")

ARMS = (VULKAN_ARM, CUDA_ARM, CPU_ARM)


# --------------------------------------------------------------------------------------------
# The CUDA arm's provider options — why TF32 is pinned OFF
# --------------------------------------------------------------------------------------------

#: Provider options for `CUDA_ARM`, minus `device_id`, which the driver fills in.
#:
#: `use_tf32: "0"` is a **methodological pin, not a tolerance loosening**, and the distinction
#: matters enough to record here rather than in a commit message.
#:
#: cuDNN on an Ampere card defaults to TF32 for fp32 convolutions: a 19-bit format with a
#: 10-bit mantissa, run on the tensor cores. The first run of this lane (TF32 at its default,
#: i.e. ON) failed the batch-1 equivalence gate on the *CUDA* arm — `max_abs` 8.196353e-3
#: against a budget of 8.193288e-3, over by 3e-6 — while the Vulkan arm on the identical input
#: came in at `max_abs` 9.54e-6, some 860x inside the same budget. A budget argued from fp32
#: numerics is the wrong budget for a TF32 computation, and the honest fix is not to widen it.
#:
#: Two reasons to pin it off rather than widen the budget:
#:
#:   1. This lane's subject is *EP against EP on the same computation*. An fp32 Vulkan arm
#:      measured against a TF32 CUDA arm compares precisions, not implementations.
#:   2. TF32 is a latency advantage as well as a precision loss, so leaving it on flatters the
#:      baseline on the very axis being reported.
#:
#: The corollary is worth stating wherever the ratio is quoted: with TF32 *on*, CUDA is faster
#: than it is here. A number measured against the TF32 baseline is therefore conservative
#: against Vulkan, and the pin moves the baseline towards Vulkan, not away from it.
#:
#: `use_tf32` is a CUDA-EP provider option from ORT 1.17 onward. An ORT that does not know the
#: key raises at session creation rather than ignoring it, which is the behaviour this pin
#: needs: silently getting TF32 anyway would be worse than failing.
CUDA_PROVIDER_OPTIONS = {"use_tf32": "0"}


# --------------------------------------------------------------------------------------------
# The counterfactual arm — what the Relu ledger gap costs
# --------------------------------------------------------------------------------------------

#: The environment variable the EP itself documents for running an unproven form.
CLAIM_UNPROVEN_ENV = "ONNXRUNTIME_EP_VULKAN_CLAIM_UNPROVEN"

#: The exact proof key the EP names when it declines ResNet-50's 49 `Relu` nodes.
#:
#: Not reconstructed from parts: this string is copied from the EP's own claim-log decline, so
#: that if the key format ever changes the counterfactual stops working loudly instead of
#: quietly measuring the un-counterfactual.
RELU_PROOF_KEY = "ai.onnx::Relu/6+/f32>f32/ew_unary_relu_f32/runtime-extent/n1"

#: A **diagnostic-only** arm: Vulkan with the `Relu` decline lifted.
#:
#: Deliberately not a member of `ARMS`. §8.9 is explicit that a claim about a form must not be
#: quoted from a run that needed this flag, and the EP's own `--check-counters` refuses such a
#: run without `--allow-unproven`. What this arm is admissible for is the thing the flag cannot
#: falsify: **partition structure**. Island count, claimed-node count and dispatch count are
#: decided by `GetCapability` before a single shader runs, so they are exactly as trustworthy
#: here as in the shipping configuration. Its *timings* are diagnostic context and are labelled
#: as such wherever they appear.
VULKAN_RELU_PROVEN_ARM = rm.Arm(
    "vulkan_relu_proven",
    (EP_NAME, CPU_EP),
    ((CLAIM_UNPROVEN_ENV, RELU_PROOF_KEY),),
    role="counterfactual",
)

#: Arms that exist for diagnosis and are never part of the headline comparison.
DIAGNOSTIC_ARMS = (VULKAN_RELU_PROVEN_ARM,)

#: Every arm a worker can be asked for by name.
ALL_ARMS = ARMS + DIAGNOSTIC_ARMS


# --------------------------------------------------------------------------------------------
# The measured fallback causes — MEASUREMENT, not inference
# --------------------------------------------------------------------------------------------

#: Why each ResNet-50 op that does not run on Vulkan does not run on Vulkan.
#:
#: Provenance class MEASUREMENT: every entry is the EP's own decline, read out of the claim log
#: (`ONNXRUNTIME_EP_VULKAN_CLAIM_LOG`) on this desk for this pinned model, not inferred from the
#: registry dump. The distinction is the whole point — a static census can only say "not
#: registered", and it would have said nothing at all about `Relu`, whose kernel is registered,
#: `Live`, f32 and opset-6+, and which falls back anyway.
#:
#: `count` is nodes per inference at batch 1 after `ORT_ENABLE_ALL` (so: after BN folding).
RESNET50_FALLBACK_CAUSES = {
    "Relu": {
        "count": 49,
        "code": "unproven",
        "registered": True,
        "kernel_exists": True,
        "cause": (
            "no proof-ledger entry for "
            "`ai.onnx::Relu/6+/f32>f32/ew_unary_relu_f32/runtime-extent/n1`. The shader exists "
            "and loads; §8.9's gate declines it because nothing has proven it correct on this "
            "form, so ORT places it on the CPU EP."
        ),
        "closes_by": "rust/tools/gen_proof_ledger.py on this device",
    },
    "GlobalAveragePool": {
        "count": 1,
        "code": "partition",
        "registered": True,
        "kernel_exists": True,
        "cause": (
            "claimed by the predicate, then dropped by the partition heuristic: once the 49 "
            "`Relu` declines had shattered the trunk, this node was left in a 1-node subgraph "
            "with no compute-heavy anchor (minimum 4). It is a *consequence* of the `Relu` "
            "gap, not an independent one."
        ),
        "closes_by": "closing the Relu gap (the anchor rule then has a subgraph to hold on to)",
    },
    "MaxPool": {
        "count": 1,
        "code": "not-registered",
        "registered": False,
        "kernel_exists": False,
        "cause": "no Vulkan handler is registered for `MaxPool` (opset 12).",
        "closes_by": "implementing a MaxPool kernel",
    },
    "Flatten": {
        "count": 1,
        "code": "not-registered",
        "registered": False,
        "kernel_exists": False,
        "cause": "no Vulkan handler is registered for `Flatten` (opset 11).",
        "closes_by": "implementing a Flatten kernel (a reshape; it moves no data)",
    },
}



#: Arms that are compared for *correctness* against `CPU_ARM`. The reference is not compared
#: against itself as though that were evidence; it is recorded as `self` so the table has no
#: hole, exactly as `probe_real_model_latency.verify_case` does.
COMPARED_ARMS = (VULKAN_ARM, CUDA_ARM)


# --------------------------------------------------------------------------------------------
# Ratio polarity — a number that says which way it points
# --------------------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Polarity:
    """What a ratio means, carried with the ratio.

    A bare `1.8x` between two arms is unreadable: it is equally consistent with "the candidate
    is 1.8x faster" and "the candidate takes 1.8x as long". This project has already published
    a ratio whose direction had to be recovered from the source (`ab_row_tile.py`'s first
    version). So the ratio type carries its own reading.
    """

    baseline: str
    candidate: str

    @property
    def meaning(self) -> str:
        return (
            f"ratio = median({self.candidate} ms) / median({self.baseline} ms), computed "
            f"per repeat and then aggregated. ratio > 1 means {self.candidate} takes LONGER "
            f"than {self.baseline} (is slower); ratio < 1 means {self.candidate} is faster."
        )


def ratio_record(baseline_samples, candidate_samples, baseline: str, candidate: str) -> dict:
    """A paired ratio that cannot be quoted without its polarity.

    Pairing is `real_model.paired_ratios` — per-repeat, because the two arms of a repeat share
    whatever the machine was doing, and a ratio of pooled medians hides a displaced repeat.
    """
    pol = Polarity(baseline=baseline, candidate=candidate)
    rec = dict(rm.paired_ratios(candidate_samples, baseline_samples))
    rec["baseline"] = pol.baseline
    rec["candidate"] = pol.candidate
    rec["polarity"] = pol.meaning
    rec["provenance_class"] = "MEASUREMENT"
    return rec


# --------------------------------------------------------------------------------------------
# Correctness gate for a classifier
# --------------------------------------------------------------------------------------------

#: f32 throughout, 53 convolutions deep, output is a raw 1000-way logit vector (no softmax in
#: the graph). The relative bound is the repository's standing 1e-2; the absolute floor is
#: scaled to the reference's own logit scale rather than fixed, for the reason
#: `PHI35_LOGIT_SCALE_FRACTION` records: a constant floor silently tightens or loosens as the
#: output scale moves, and a ResNet logit scale moves with the input distribution.
RESNET_RTOL = 1e-2
RESNET_LOGIT_SCALE_FRACTION = 1e-3
#: What a classifier's output is *for*. Two logit vectors that agree to 1e-3 of scale but rank
#: a different class first are not the same model behaviour; two that disagree in the 900th
#: logit and rank identically are.
RESNET_TOP_K = 5
RESNET_MAX_PROB_DELTA = 1e-3


def classify_resnet_logits(candidate, reference, np, *, rtol: float = RESNET_RTOL,
                           scale_fraction: float = RESNET_LOGIT_SCALE_FRACTION,
                           top_k: int = RESNET_TOP_K,
                           max_prob_delta: float = RESNET_MAX_PROB_DELTA) -> dict:
    """MATCH / DIVERGENT for one ResNet output tensor, per batch row.

    Three conditions, all required, and all reported whether or not they pass:

    1. **top-1 agrees on every row.** A classifier that ranks a different class first is wrong
       in the only way its user can see.
    2. **the top-k set agrees on every row.**  Ordering *within* the tail is allowed to differ
       (two logits 1e-6 apart may swap under a different reduction order and that is not a
       defect); membership is not.
    3. **numerical agreement**: `max_abs <= scale_fraction * max|reference|` OR
       `max_rel <= rtol`, plus a bound on the post-softmax probability movement.

    `max_rel` alone is deliberately not the gate. Over a 1000-way logit vector containing
    near-zero entries it is a cancellation meter, not an accuracy meter — the same reading
    §25.3 of `docs/PERF.md` records for the decoder.
    """
    c = np.asarray(candidate, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    if c.shape != r.shape:
        return {"verdict": rm.DIVERGENT, "reason": "shape mismatch",
                "candidate_shape": list(c.shape), "reference_shape": list(r.shape)}
    if c.ndim != 2 or c.shape[1] != RESNET50_CLASSES:
        return {"verdict": rm.DIVERGENT, "reason": "not a [N,1000] classifier output",
                "candidate_shape": list(c.shape)}
    nan_c = int(np.count_nonzero(~np.isfinite(c)))
    nan_r = int(np.count_nonzero(~np.isfinite(r)))
    if nan_c or nan_r:
        return {"verdict": rm.DIVERGENT, "reason": "non-finite values",
                "nonfinite_candidate": nan_c, "nonfinite_reference": nan_r}
    # An all-zero candidate is the `argmax 0` defect: it agrees with nothing and its top-1 is
    # class 0 by tie-break, which a naive top-1 check on a degenerate reference would pass.
    if not np.any(c):
        return {"verdict": rm.DIVERGENT, "reason": "candidate output is entirely zero",
                "note": "the argmax-0 defect: dispatches executed, no numbers produced"}

    diff = np.abs(c - r)
    max_abs = float(diff.max())
    scale = float(np.abs(r).max())
    denom = np.maximum(np.abs(r), np.finfo(np.float64).tiny)
    max_rel = float((diff / denom).max())
    abs_budget = scale_fraction * scale

    top1_c = c.argmax(axis=1)
    top1_r = r.argmax(axis=1)
    top1_rows = int(np.count_nonzero(top1_c == top1_r))
    k = min(top_k, c.shape[1])
    set_c = [set(np.argsort(-row)[:k].tolist()) for row in c]
    set_r = [set(np.argsort(-row)[:k].tolist()) for row in r]
    topk_rows = int(sum(1 for a, b in zip(set_c, set_r) if a == b))

    pc = np.exp(c - c.max(axis=1, keepdims=True))
    pc = pc / pc.sum(axis=1, keepdims=True)
    pr = np.exp(r - r.max(axis=1, keepdims=True))
    pr = pr / pr.sum(axis=1, keepdims=True)
    max_prob = float(np.abs(pc - pr).max())

    numeric_ok = (max_abs <= abs_budget) or (max_rel <= rtol)
    rows = c.shape[0]
    ok = (top1_rows == rows) and (topk_rows == rows) and numeric_ok \
        and (max_prob <= max_prob_delta)
    return {
        "verdict": rm.MATCH if ok else rm.DIVERGENT,
        "rows": rows,
        "top1_rows_agreeing": top1_rows,
        "topk_rows_agreeing": topk_rows,
        "top_k": k,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "reference_scale": scale,
        "abs_budget": abs_budget,
        "max_prob_delta": max_prob,
        "prob_budget": max_prob_delta,
        "numeric_ok": bool(numeric_ok),
        "gate": ("top-1 on every row AND top-{k} set on every row AND (max_abs <= "
                 "{sf} x |reference|_max OR max_rel <= {rt}) AND max prob delta <= {pd}").format(
                     k=k, sf=scale_fraction, rt=rtol, pd=max_prob_delta),
        "provenance_class": "MEASUREMENT",
    }


def classify_case(case: rm.Case, candidate_outputs, reference_outputs, np) -> dict:
    """Verdict for one arm on one case. ResNet-50 has exactly one output."""
    if len(candidate_outputs) != len(reference_outputs):
        return {"verdict": rm.DIVERGENT, "reason": "output count mismatch",
                "n_candidate": len(candidate_outputs), "n_reference": len(reference_outputs)}
    if len(candidate_outputs) != 1:
        return {"verdict": rm.DIVERGENT, "reason": "ResNet-50 v1-12 has exactly one output",
                "n_candidate": len(candidate_outputs)}
    primary = classify_resnet_logits(candidate_outputs[0], reference_outputs[0], np)
    return {"verdict": primary["verdict"], "primary": primary, "case": case.label}


# --------------------------------------------------------------------------------------------
# The static support census — a MODEL, and labelled as one
# --------------------------------------------------------------------------------------------

def support_census(op_histogram: dict, capabilities: dict) -> dict:
    """Which of this graph's op types the shipped registry has a kernel for.

    ``capabilities`` is `epctl --dump-capabilities --json` — the EP's own answer about what it
    registers, read from the binary rather than from a table in a document that can go stale.

    THE THREE THINGS THIS IS NOT
    ----------------------------
    1. It is **not the partition.** ORT decides that, per node, after graph optimisation, using
       each op's claim predicate on that node's actual attributes and dtypes. A registered op
       can still decline a specific node.
    2. It is **not what runs.** `ORT_ENABLE_ALL` folds `BatchNormalization` into the preceding
       `Conv`, so 53 of this graph's 175 nodes do not exist by the time the EP is asked.
    3. It is **not a count of islands.** Two unsupported nodes in the middle of a chain cost
       far more than two at the end, and this function cannot see where they are.

    So the record is `provenance_class: MODEL`. The MEASUREMENT of partitioning is ORT's own
    per-node provider attribution in a profiled run, which the driver records separately.
    """
    ops = {}
    for row in (capabilities or {}).get("ops", []) or []:
        name = row.get("name")
        if isinstance(name, str):
            ops[name] = row
    supported, unsupported = {}, {}
    for op, count in sorted(op_histogram.items()):
        row = ops.get(op)
        # `has_kernel`, NOT `status == "live"`. This is the §8.9.25 ruling-6 trap, and the first
        # draft of this function walked straight into it: `status` is three-valued
        # (`live`/`ready`/`staged`) and its `live` token is the *deprecated* `OpStatus::Live`
        # alias, while `has_kernel` is true for `Live` AND `Ready`. Reading the token would have
        # reported ResNet-50's `Conv`, `Gemm` and `GlobalAveragePool` — which are `ready`, carry
        # kernels, and account for 55 of the graph's 175 nodes — as *unsupported*, and the
        # resulting census would have said the EP cannot run a convolution.
        live = bool(row) and bool(row.get("has_kernel")) and row.get("status") != "staged"
        target = supported if live else unsupported
        target[op] = {
            "nodes": count,
            "registered": bool(row),
            "status": (row or {}).get("status"),
            "has_kernel": (row or {}).get("has_kernel"),
            "dtypes": (row or {}).get("dtypes"),
            "opsets": (row or {}).get("opsets"),
            "staged_reason": (row or {}).get("staged_reason"),
        }
    total = sum(op_histogram.values())
    covered = sum(v["nodes"] for v in supported.values())
    return {
        "provenance_class": "MODEL",
        "derivation": ("op-type histogram of the UNOPTIMISED pinned graph, matched against "
                       "`epctl --dump-capabilities --json` from the built EP, on the "
                       "`has_kernel` predicate (NOT the deprecated `status == \"live\"` token, "
                       "which is false for the `ready` rows that carry ResNet-50's Conv)"),
        "crate_version": (capabilities or {}).get("crate_version"),
        "ort_built_against": (capabilities or {}).get("ort_built_against"),
        "graph_nodes": total,
        "op_types_supported": sorted(supported),
        "op_types_unsupported": sorted(unsupported),
        "supported": supported,
        "unsupported": unsupported,
        "nodes_with_a_registered_kernel": covered,
        "nodes_without_a_registered_kernel": total - covered,
        "upper_bound_note": ("an upper bound on Vulkan coverage and nothing else: registration "
                             "is necessary, not sufficient — the per-node claim predicate may "
                             "still decline, and ORT's optimiser rewrites the graph first"),
    }


# --------------------------------------------------------------------------------------------
# Admissibility — the gate that decides whether a number may be quoted at all
# --------------------------------------------------------------------------------------------

#: The states a reading may be in. `INDETERMINATE` is a *result*, not a failure to produce one:
#: it says the run happened and the machine state does not support quoting it.
ADMISSIBLE = "ADMISSIBLE"
INDETERMINATE = "INDETERMINATE"


def admissibility(*, provenance_ok: bool, equivalence_gate: str,
                  vulkan_dispatched: "int | None", cuda_ran: "bool | None",
                  quiescence: "dict | None", device_identified: bool,
                  repeats: int, iters: int) -> dict:
    """Every condition, each named, and the verdict that follows from all of them.

    Deliberately a pure function of recorded fields, with no stored verdict — the
    `ProvenanceRecord.provenance_ok` discipline from `bench/pinned_bytes.py`. A record cannot
    travel with `ADMISSIBLE` while the evidence beside it disagrees, because the verdict is
    recomputed from that evidence on every read.
    """
    checks = [
        {"name": "model_provenance",
         "held": bool(provenance_ok),
         "detail": "the pinned sha256/size/external-data scan agreed with the recorded pin"},
        {"name": "outputs_agree_with_cpu",
         "held": equivalence_gate == "PASS",
         "detail": f"equivalence gate = {equivalence_gate}; every timed arm was classified "
                   f"against the CPU EP reference before it was timed"},
        {"name": "vulkan_production_dispatch_witness",
         "held": bool(vulkan_dispatched),
         "detail": f"dispatches_executed = {vulkan_dispatched} from the EP's own counters; a "
                   f"Vulkan row with no dispatch is a CPU row wearing the EP's name"},
        {"name": "cuda_arm_executed",
         "held": bool(cuda_ran),
         "detail": "ORT's per-node provider attribution placed at least one node on "
                   "CUDAExecutionProvider"},
        {"name": "device_identified",
         "held": bool(device_identified),
         "detail": "the running device is named by stable identity, not by enumeration index"},
        {"name": "quiescence",
         "held": bool((quiescence or {}).get("quiet")),
         "detail": (quiescence or {}).get("reason") or "no quiescence verdict was taken"},
        {"name": "enough_repeats",
         "held": repeats >= 3 and iters >= 5,
         "detail": f"repeats={repeats}, iters_per_repeat={iters}; a median without repeats "
                   f"measured adjacent in time has no error bar"},
    ]
    failed = [c["name"] for c in checks if not c["held"]]
    return {
        "verdict": ADMISSIBLE if not failed else INDETERMINATE,
        "failed_checks": failed,
        "checks": checks,
        "rule": ("ADMISSIBLE requires every check to hold. Any failure yields INDETERMINATE, "
                 "and an INDETERMINATE record's ratios must be reported as not-quotable rather "
                 "than quoted with a caveat."),
    }


def quotable(admissibility_record: "dict | None") -> bool:
    """One place that answers 'may this be quoted', so two readers cannot disagree."""
    return bool((admissibility_record or {}).get("verdict") == ADMISSIBLE)
