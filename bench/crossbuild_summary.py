"""The cross-build GQA landing evidence, summarised by exactly one function.

WHAT THIS MODULE IS, AND WHY IT IS NOT IN ``bench/results/``
===========================================================
``bench/results/real_model_crossbuild_gqa_landing.pr95-frozen.json`` holds 60 raw records:
10 workloads x 2 builds x 3 repeats, one OS process each, produced for issue #69 and frozen
at PR #95's head ``ce632011bb597bd44b64c5dd997a9210691c1615``.  **This module did not measure
them and does not claim to.**  It reads them.

The measurement was reviewed independently by Niobe, who recomputed every median, every
paired ratio and every verdict from the raw ``samples_ms`` and found the arithmetic exact,
and who wrote that the 60 records do not need to be re-measured.  What she rejected was not
the data.  It was two guards:

* **B1** — the rule that refuses a speed verdict when the two arms cannot be shown to have
  run different code was implemented *inside a probe script*, and the test that claimed to
  cover it re-implemented the rule in the test body.  Deleting the shipped rule left the
  entire validation surface green.
* **B2** — the band that decided which ratios counted was computed from the null control's
  own three ratios and then used to judge that same null control, which made its verdict
  analytically fixed.  A control whose verdict cannot move carries no information.

Both defects have the same root: **the thing that renders the verdict was not a shipped,
census-visible, single-implementation function.**  So this module is that function, and it
lives in ``bench/`` rather than ``bench/results/`` on purpose.  ``FRAME_IGNORE_DIRS`` in
``rust/tools/audit_instruments.py`` contains ``"results"``, and the bench screen globs
``bench/*.py`` non-recursively, so anything under ``bench/results/`` is invisible to the
census — which is precisely how 896 lines of admissibility and verdict logic escaped it.
``crossbuild_summary.py`` is declared in ``BENCH_INSTRUMENT_FILES``; every public function
below is screened, and every one of them is watched to disagree in
``bench/test_crossbuild_summary.py``.

THE ONE PATH
============
``summarize()`` is the only way a verdict is produced in this repository.  The CLI
(``python bench/crossbuild_summary.py --finalize``) calls it; the published artifact is its
output; ``docs/PERF.md`` s27 is recomputed from that output by
``bench/test_crossbuild_gqa_landing.py``; and the tests call it too.  No test re-implements
a gate.  ``bench/test_crossbuild_summary.py`` proves that by mutating *this file's source*
and requiring the assertions to go red.

WHAT IS DIFFERENT FROM PR #95, STATED PLAINLY
=============================================
1. The band no longer comes from the workload it judges.  It comes from the four **non-GQA**
   workloads, whose path witness proves no ``gqa_f16`` pipeline was ever built on either arm,
   so the landed change cannot have touched them.  Those four are **calibration** and are
   never assigned a speed verdict.  The six Phi workloads are **subjects** and are judged by
   a band they did not define -- including the ``M=1`` null control, which can now come back
   ``FASTER`` and could not before.
2. The decision rule is symmetric.  PR #95 required every repeat to clear the band for
   ``FASTER`` but only the median for ``SLOWER``.  Here both directions require every repeat.
3. **The band applied here was chosen after these records existed.**  There is no externally
   timestamped pre-timing rule for it and this module does not pretend otherwise; see
   ``BAND_PROVENANCE``.  What is offered instead of precedence is (a) a calibration set that
   is disjoint from every subject by a mechanical criterion, and (b) a published sensitivity
   sweep, so a reader who prefers a different band can read their own answer off the table.

WHAT THIS MODULE DELIBERATELY DOES NOT READ
===========================================
The frozen file also carries PR #95's own derived blocks -- ``band``, ``workloads``,
``preregistration``.  Those are **superseded**: they encode the circular band and the
mislabelled ``null_control_half_range`` field (it holds ``max|r-1|`` = 5.1029%, not a
half-range; the half-range ``(max-min)/2`` is 4.4548%).  ``load_frozen`` refuses to hand them
out, and ``bench/test_crossbuild_gqa_landing.py`` asserts no published number is traceable to
them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

_BENCH = Path(__file__).resolve().parent
_ROOT = _BENCH.parent
RESULTS = _BENCH / "results"

# ---------------------------------------------------------------------------------------
# THE FROZEN INPUT
# ---------------------------------------------------------------------------------------
#: The raw record set, byte-identical to blob `03b1308` at PR #95 head `ce632011`.
FROZEN_PATH = RESULTS / "real_model_crossbuild_gqa_landing.pr95-frozen.json"

#: sha256 over the file's bytes with CRLF normalised to LF.  Normalisation is not cosmetic:
#: this repository sets `core.autocrlf=true`, so a Windows checkout of a text-classified file
#: differs byte-for-byte from the blob git stores.  PR #95's own `preregistration.sha256`
#: matched only under CRLF for exactly this reason, which made the digest a property of one
#: machine's checkout rather than of the evidence.  `.gitattributes` additionally pins this
#: path `-text` so the two agree, and the normalisation here means the check still holds on a
#: clone made before that rule existed.
FROZEN_SHA256 = "d8df20637915417592640980b1d643031ff93ff3d67a520103ddbfc3b852ef74"

SCHEMA = "real_model_crossbuild_gqa_landing/1"
SUMMARY_SCHEMA = "crossbuild_gqa_landing_summary/2"

#: Blocks of the frozen file that encode PR #95's rejected derivation.  Refused, not deleted:
#: deleting them would break the byte-identity that makes the import auditable.
SUPERSEDED_BLOCKS = ("band", "workloads", "preregistration")

SOURCE_PR = 95
SOURCE_HEAD = "ce632011bb597bd44b64c5dd997a9210691c1615"

EXPECTED_WORKLOADS = 10
EXPECTED_ARMS = ("candidate", "baseline")
EXPECTED_REPEATS = 3
EXPECTED_ITERS = 20
EXPECTED_WARMUPS = 5

#: 1 first run + 5 discarded warmups + 20 timed iterations + 1 post-timing re-verification,
#: multiplied by the number of Vulkan islands the graph partitions into.  Constant within a
#: workload and identical across arms, which is what makes it usable as a borrowed-witness
#: screen: a witness lifted from another model carries that model's island count.
SESSIONS_PER_RECORD = 1 + EXPECTED_WARMUPS + EXPECTED_ITERS + 1

# ---------------------------------------------------------------------------------------
# THE BAND
# ---------------------------------------------------------------------------------------
#: A floor that no data can lower.  Data-independent by construction: a 5% claim on a shared,
#: indefinitely contended box (docs/PERF.md s20) is not a claim.
BAND_FLOOR = 0.05

BAND_RULE = (
    "band = max(BAND_FLOOR=0.05, max |ratio - 1| over every paired repeat ratio of the "
    "CALIBRATION set). The calibration set is every workload whose GQA path witness is empty "
    "on BOTH arms, i.e. no gqa_f16 pipeline was ever built, so the landed change cannot have "
    "moved it and its spread is drift alone. Calibration workloads are never assigned a speed "
    "verdict: a control that defines the band may not be judged by it."
)

BAND_PROVENANCE = (
    "POST-HOC, and said so rather than dressed up. These 60 records already existed when this "
    "band rule was written; no externally timestamped pre-timing rule for it exists, and none "
    "is claimed. PR #95's `preregistration` block is text embedded in the artifact at finalize "
    "time, so its digest binds the text to itself and not to a point in time -- that is why it "
    "is superseded here rather than inherited. Two things stand in for precedence and neither "
    "is a substitute for it: the calibration set is separated from the subjects by a mechanical "
    "criterion that cannot be steered towards a result (a workload is calibration iff its GQA "
    "witness is empty on both arms), and the full sensitivity sweep is published so a reader "
    "may apply their own band."
)

DECISION_RULE = (
    "ratio = baseline_median_ms / candidate_median_ms, paired WITHIN a repeat (>1 means the "
    "candidate build is faster). Per workload, over its 3 repeat ratios, with band b: "
    "FASTER iff min(ratios) > 1+b; SLOWER iff max(ratios) < 1-b; NEUTRAL iff every ratio lies "
    "in [1-b, 1+b]; INDETERMINATE otherwise. Symmetric on purpose -- PR #95 required every "
    "repeat for FASTER but only the median for SLOWER, which makes a regression cheaper to "
    "claim than an improvement. n=3: this is an envelope screen, NOT a confidence interval, "
    "and no p-value is computed or implied."
)

#: Bands the sensitivity sweep is published at.  5.00% is the rule PR #95's prose *said*
#: (`(max-min)/2` over the null control), 5.10% is the number it actually *applied*
#: (`max|r-1|`), and the rest bracket the calibration envelope.
SENSITIVITY_BANDS = (0.05, 0.0510292125170773, 0.10, 0.15, 0.20, 0.25, 0.30)

VERDICTS = ("FASTER", "SLOWER", "NEUTRAL", "INDETERMINATE", "REFUSED", "CALIBRATION")

# ---------------------------------------------------------------------------------------
# OFFLINE WITHIN-ARM DISPERSION -- an A/A surrogate computed from the frozen samples
# ---------------------------------------------------------------------------------------
#: A cross-arm ratio is only interesting to the extent that it is larger than what the SAME
#: build does against ITSELF on this host.  A real A/A run does not exist -- that is issue #96
#: -- so two surrogates are computed from records that already exist.  Nothing is re-measured.
WITHIN_ARM_RULE = (
    "Two same-build (A/A) surrogates are computed offline from the frozen samples_ms, per arm "
    "per workload. (1) ACROSS-REPEAT: |m_i/m_j - 1| over every ordered pair of that arm's three "
    "repeat medians -- same build, same workload, different process, same host, i.e. the exact "
    "comparison the cross-arm ratio makes with the build held constant. (2) SPLIT-HALF: "
    "|median(first 10 timed samples)/median(last 10) - 1| within each single record -- same "
    "build, same process, adjacent in time, so it is paired the way the cross-arm ratio is "
    "paired. within_arm_envelope = max of both surrogates over both arms, i.e. the conservative "
    "one. separation_ratio = min |ratio - 1| over the paired repeats, divided by that envelope. "
    "This is a DIAGNOSTIC, not a gate: it never changes a verdict, because a rule invented after "
    "seeing which subjects it would demote is not a rule. It reports how much room there is "
    "between an effect and this host's own noise."
)

#: A cross-arm effect at least this many times the within-arm envelope is called separated.
#: 2x is a convention, not a test statistic, and is exposed here so it can be argued with.
SEPARATION_STRONG = 2.0

SEPARATED = "SEPARATED_FROM_WITHIN_ARM_DRIFT"
NOT_SEPARATED = "NOT_SEPARATED_FROM_WITHIN_ARM_DRIFT"

#: What a consistently-directional ratio that the band will not grade is called, and the exact
#: condition that would retire the label.  `decode past=128` (0.859x) is the row this exists for.
PROVISIONAL_DESCRIPTIVE = "PROVISIONAL_DESCRIPTIVE"
PROVISIONAL_UNTIL = (
    "issue #96: a same-build A/A run on this host, plus the bimodality diagnosis. Until that "
    "exists this is a PROVISIONAL DESCRIPTIVE RATIO -- a description of what these 60 frozen "
    "records contain -- and NOT a finding that the candidate build is slower. It is not a "
    "SLOWER verdict, it is not a regression claim, and no ownership of it is asserted here."
)

# ---------------------------------------------------------------------------------------
# WITNESS CLASSES
# ---------------------------------------------------------------------------------------
#: Neither arm ever built a gqa_f16 pipeline -> the landed change provably cannot have moved
#: this workload -> it is calibration, not a subject.
NO_GQA = "NO_GQA"
#: The two arms built demonstrably different pipelines -> a timing difference is attributable
#: to a code-path difference at all.
DISTINGUISHED = "DISTINGUISHED"
#: Both arms built GQA pipelines and their witnesses are identical -> whatever the clock says,
#: nothing here shows the two builds ran different code. THE GATE (B1) fires on this.
NOT_DISTINGUISHED = "NOT_DISTINGUISHED"

# ---------------------------------------------------------------------------------------
# COMPILED-INPUT DELTA (issue #69 / Niobe F4)
# ---------------------------------------------------------------------------------------
#: Every compiled input of `onnxruntime_ep_vulkan.dll` that differs between the two measured
#: trees, enumerated rather than scoped to `rust/`. PR #95 said "the compiled delta is the GQA
#: change and nothing else"; that sentence was false, because `rust/src/registry.rs` takes
#: `evidence/proof_ledger.jsonl` by `include_str!` and that file differs.
#:
#: `verified_by` names the command a reviewer runs to reproduce each row.
COMPILED_INPUT_DELTA = {
    "baseline_commit": "c96e7d94ff706d26ee6a1bd9bb084c0ade426820",
    "candidate_commit": "85fbda29a92e0e99c3895be8b13664d4ee670c50",
    "compiled_inputs_considered": [
        "rust/src/**/*.rs (the cdylib sources)",
        "rust/shaders/** (compiled to SPIR-V by build.rs and embedded with include_bytes!)",
        "rust/src/ops/shader_variants.txt (build.rs variant table; a build input)",
        "rust/build.rs, rust/Cargo.toml, rust/Cargo.lock, rust/wrapper_ort.h",
        "evidence/proof_ledger.jsonl (include_str! at rust/src/registry.rs:2306)",
    ],
    "differing": [
        {
            "path": "rust/shaders/glsl/gqa_f16.comp",
            "what": (
                "one non-comment line: `layout(local_size_x = 1, ...)` becomes "
                "`layout(local_size_x_id = 0, local_size_x = 1, ...)`. The rest of the diff is "
                "comment."
            ),
            "on_the_timed_path": True,
        },
        {
            "path": "rust/src/ops/attention.rs",
            "what": (
                "the GQA dispatch gains `spec_constants: vec![local]` and "
                "`workgroups: [total.div_ceil(local), 1, 1]`, plus `gqa_local_size`, "
                "`GQA_MAX_LOCAL_SIZE`, `GQA_MIN_GROUPS`, `ENV_GQA_LOCAL_SIZE` and five "
                "`#[cfg(test)]` unit tests."
            ),
            "on_the_timed_path": True,
        },
        {
            "path": "evidence/proof_ledger.jsonl",
            "what": (
                "header `content_fnv1a64` 8902d86b502e04e3 -> cb6391d0843c1bb1 and its "
                "`generated_at`; the GroupQueryAttention entry is re-witnessed on the RTX A1000 "
                "with new shader/source/spec digests and moves to the end of the file. "
                "Entry count is 133 on both sides."
            ),
            "on_the_timed_path": False,
        },
    ],
    "unchanged": [
        "rust/build.rs",
        "rust/Cargo.toml",
        "rust/Cargo.lock",
        "rust/src/ops/shader_variants.txt",
        "every rust/src/**/*.rs other than ops/attention.rs",
        "every rust/shaders/** other than glsl/gqa_f16.comp",
    ],
    "attribution": (
        "BOUNDED, not isolated. Two of the three differing compiled inputs are the landed GQA "
        "change itself. The third, the baked proof ledger, reaches the binary only through "
        "`LEDGER_SOURCE`, whose only non-test consumers are `registry::baked_ledger_identity()` "
        "and its caller in `lib.rs` -- a diagnostic string builder that no inference path calls, "
        "so it cannot execute during a timed iteration. It is also very likely the source of the "
        "3,072-byte size difference between the two DLLs. What is therefore claimed is: the only "
        "compiled difference REACHABLE FROM A TIMED INFERENCE is the GQA workgroup-packing "
        "change. What is NOT claimed is that the two binaries differ in nothing else."
    ),
    "verified_by": [
        "git diff --name-status c96e7d9 85fbda2",
        "git grep -n 'include_str!\\|include_bytes!' 85fbda2 -- rust/",
        "git grep -n 'LEDGER_SOURCE\\|baked_ledger_identity' 85fbda2 -- rust/src",
        "git diff -U0 c96e7d9 85fbda2 -- rust/shaders/glsl/gqa_f16.comp",
    ],
}

# ---------------------------------------------------------------------------------------
# WHAT THE LOCK PROVES (Niobe F3)
# ---------------------------------------------------------------------------------------
EXCLUSIVITY_LANGUAGE = {
    "mechanism": (
        "msvcrt.locking(..., LK_NBLCK, ...) byte-range lock on a file handle held open for the "
        "duration of the run."
    ),
    "proves": [
        "no OTHER PROCESS THAT COOPERATES BY TAKING THE SAME LOCK ran a measurement "
        "concurrently with this one",
        "the lock was acquired without waiting (0.0 s), held 1840.6 s and released cleanly, so "
        "the run was not silently split around a contention stall",
        "nothing was terminated to obtain it -- the policy is wait, never kill",
    ],
    "does_not_prove": [
        "exclusive ownership of the GPU. The lock is advisory: it excludes only compliant "
        "harness processes and has no effect on anything else on the device",
        "a quiet machine. The artifact's own exclusivity block records 26 GPU compute "
        "application ENTRIES at acquire and 26 at release -- 30 processes once the `xN` "
        "multiplicities in that list are expanded -- including a browser and four of its "
        "renderers, plus two entries the probe could not name for lack of permissions",
        "an absence of interference. docs/PERF.md s20 is the standing position: this box is "
        "contended and that is the baseline. The design answer to it is pairing within a "
        "repeat and interleaving the arms, not the lock",
    ],
}

MODEL_PINS = {
    "all-MiniLM-L6-v2-onnx": {
        "repo": "sentence-transformers/all-MiniLM-L6-v2",
        "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "file": "onnx/model.onnx",
        "sha256": "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452",
        "bytes": 90405214,
        "note": (
            "Pinned here, independently of any unlanded PR. bench/real_model.py on main "
            "carries no MiniLM entry, so nothing about this control depends on rejected #83."
        ),
    },
    "phi-3.5-mini-instruct-cuda-int4-rtn-block-32": {
        "sha256": "3dbdd4b5f4d487da609fdacb9fd35b113cac706363a72795508524a4704dac3f",
        "bytes": 26180848,
        "note": "Foundry Local resolution; cross-checked against bench/results/rust-model-runner.",
    },
    "mobilenetv2-12": {
        "sha256": "c0c3f76d93fa3fd6580652a45618618a220fced18babf65774ed169de0432ad5",
        "bytes": 13964571,
        "note": "the repository's existing pinned ONNX model zoo artifact.",
    },
}


class CrossbuildError(Exception):
    """Base for every refusal this module raises rather than returns."""


class ProvenanceError(CrossbuildError):
    """The frozen input is not the evidence it claims to be."""


class SchemaError(CrossbuildError):
    """A record or row is not shaped like the thing the caller said it was."""


class AdmissibilityError(CrossbuildError):
    """A record contradicts itself: it is refused and still carries a timing."""


# ---------------------------------------------------------------------------------------


def _normalised(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")


def _digest(path: Path) -> str:
    return hashlib.sha256(_normalised(path.read_bytes())).hexdigest()


def _median(values) -> float:
    return statistics.median(values)


def load_frozen(path=None, *, expect_sha256: "str | None" = FROZEN_SHA256) -> dict:
    """Read the frozen 60-record set, or refuse.

    Returns the artifact **without** :data:`SUPERSEDED_BLOCKS`: PR #95's own ``band``,
    ``workloads`` and ``preregistration`` blocks are the rejected derivation, and a summariser
    that can see them can accidentally quote them.  Everything downstream is therefore forced
    through :func:`summarize`.

    Raises :class:`ProvenanceError` when the digest, the schema or the 10x2x3 shape does not
    hold.  This is the reject polarity that the artifact-substitution mutation trips.
    """
    path = FROZEN_PATH if path is None else Path(path)
    if not path.is_file():
        raise ProvenanceError(f"no frozen record set at {path}")
    got = _digest(path)
    if expect_sha256 is not None and got != expect_sha256:
        raise ProvenanceError(
            f"{path.name} hashes to {got}, not the pinned {expect_sha256}. The bytes imported "
            f"from PR #{SOURCE_PR} head {SOURCE_HEAD} are the evidence; a different file is a "
            f"different measurement and this module will not summarise it."
        )
    try:
        doc = json.loads(_normalised(path.read_bytes()).decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ProvenanceError(f"{path.name} is not readable JSON: {exc}") from exc
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        raise ProvenanceError(
            f"{path.name} declares schema {doc.get('schema') if isinstance(doc, dict) else '?'}, "
            f"expected {SCHEMA}"
        )
    records = doc.get("records")
    if not isinstance(records, list):
        raise ProvenanceError(f"{path.name} carries no `records` list")
    expected = EXPECTED_WORKLOADS * len(EXPECTED_ARMS) * EXPECTED_REPEATS
    if len(records) != expected:
        raise ProvenanceError(
            f"{path.name} carries {len(records)} records, expected {expected} "
            f"({EXPECTED_WORKLOADS} workloads x {len(EXPECTED_ARMS)} builds x "
            f"{EXPECTED_REPEATS} repeats)"
        )
    cells = {(r.get("workload"), r.get("arm"), r.get("repeat")) for r in records}
    if len(cells) != expected:
        raise ProvenanceError(
            f"{path.name} does not cover the design exactly once: {len(cells)} distinct "
            f"(workload, arm, repeat) cells for {len(records)} records"
        )
    out = {k: v for k, v in doc.items() if k not in SUPERSEDED_BLOCKS}
    out["_frozen"] = {
        "path": path.name,
        "sha256": got,
        "bytes": len(path.read_bytes()),
        "source_pr": SOURCE_PR,
        "source_head": SOURCE_HEAD,
        "measured_by": (
            "Tank, on PR #95. REUSED HERE, NOT RE-MEASURED. Independently revalidated by "
            "Niobe, who recomputed every median, ratio and verdict from the raw samples and "
            "recorded that remeasurement is unnecessary."
        ),
        "superseded_blocks_withheld": list(SUPERSEDED_BLOCKS),
    }
    return out


def record_refusals(record, *, models=None) -> "list[str]":
    """Every reason this single record may not contribute a timing. Empty means admissible.

    This is the admissibility gate, and it is the *only* one.  A record is admissible when it
    carries its own CPU-EP reference, its own output digests, a MATCH equivalence verdict, an
    unchanged post-timing digest, a model digest that agrees with the pinned identity, a
    production path witness recorded in its own process, and exactly
    ``EXPECTED_ITERS`` timed samples.  Anything else and the timing does not exist for
    downstream purposes.

    Two failure kinds are kept apart on purpose.  A missing *premise* is a **refusal** and is
    returned, because refusing a record is normal operation.  A record that **contradicts
    itself** -- one its own producer marked refused or inadmissible while it still carries
    ``speed`` -- raises :class:`AdmissibilityError`, because that is not a measurement with a
    problem, it is a summariser being handed a number that was already withdrawn.  Handing
    something that is not a record at all raises :class:`SchemaError`.

    *models* is the artifact's own ``models`` block when there is one.  It is checked against
    :data:`MODEL_PINS`, which binds every record to a model identity this file pins
    independently of any unlanded branch.
    """
    if not isinstance(record, dict) or "workload" not in record or "arm" not in record:
        raise SchemaError(
            f"not a crossbuild record: {type(record).__name__} "
            f"{sorted(record) if isinstance(record, dict) else record!r}"
        )
    speed_present = record.get("speed") is not None
    if speed_present and record.get("refusal") is not None:
        raise AdmissibilityError(
            f"{record['workload']}/{record['arm']}/r{record['repeat']} declares a refusal "
            f"({record['refusal']!r}) and still carries timing fields. A withdrawn record that "
            f"keeps its speed is the one shape a summariser must never quietly accept."
        )
    if speed_present and record.get("admissible") is False:
        raise AdmissibilityError(
            f"{record['workload']}/{record['arm']}/r{record['repeat']} is marked inadmissible "
            f"and still carries timing fields."
        )
    declared = (models or {}).get(record.get("model_key")) or {}
    why: list[str] = []

    equivalence = record.get("equivalence")
    if equivalence is None:
        why.append("equivalence_missing: no CPU-EP reference comparison on this record")
    elif not equivalence:
        why.append("equivalence_empty: the comparison block is present but says nothing")
    elif equivalence.get("verdict") != "MATCH":
        why.append(f"equivalence_{str(equivalence.get('verdict')).lower()}: outputs disagree")

    if not record.get("cpu_reference_outputs_sha256"):
        why.append("oracle_missing: no CPU-EP reference digest computed in this process")
    if not record.get("outputs_sha256"):
        why.append("outputs_missing: nothing to compare")
    elif record.get("outputs_sha256_post_timing") != record.get("outputs_sha256"):
        why.append("outputs_moved_during_timing: pre- and post-timing digests differ")

    pin = MODEL_PINS.get(record.get("model_key"))
    if pin is None:
        why.append(f"model_unpinned: {record.get('model_key')!r} has no pinned identity")
    else:
        for source, digest in (
            ("record", record.get("model_sha256")),
            ("artifact", declared.get("sha256")),
        ):
            if digest is not None and digest != pin["sha256"]:
                why.append(
                    f"model_digest_wrong: the {source} says {digest[:12]}..., the pin says "
                    f"{pin['sha256'][:12]}..."
                )
        if declared.get("bytes") is not None and pin.get("bytes") is not None:
            if declared["bytes"] != pin["bytes"]:
                why.append(
                    f"model_bytes_wrong: {declared['bytes']} against the pinned {pin['bytes']}"
                )
        if declared.get("agrees_with_recorded_provenance") is False:
            why.append("model_provenance_disagrees: the resolver and the recorded pin differ")

    witness = record.get("path_witness")
    if not isinstance(witness, dict) or not witness.get("present"):
        why.append("witness_missing: no production pipeline witness for this process")
    elif witness.get("compute_failures"):
        why.append("witness_compute_failures: the EP reported failed compute calls")

    speed = record.get("speed")
    if speed is None:
        why.append("timing_missing: no samples on this record")
    else:
        samples = speed.get("samples_ms") or []
        if speed.get("n") != EXPECTED_ITERS or len(samples) != EXPECTED_ITERS:
            why.append(
                f"timing_shape: {speed.get('n')} declared / {len(samples)} stored samples, "
                f"expected {EXPECTED_ITERS} after {EXPECTED_WARMUPS} discarded warmups"
            )

    if record.get("worker_exit") not in (0, None):
        why.append(f"worker_exit_{record.get('worker_exit')}: the measuring process failed")

    if why and record.get("admissible") is True:
        why.append(
            "producer_disagrees: the record calls itself admissible and this gate does not; "
            "the gate wins"
        )
    return why


def witness_class(candidate, baseline) -> str:
    """Classify what the two arms' production path witnesses prove about each other.

    :data:`NO_GQA` -- neither arm ever built a ``gqa_f16`` pipeline, so the landed change
    cannot have touched this workload.  That is the *calibration* criterion, and it is
    mechanical: it cannot be steered towards a wanted result.

    :data:`DISTINGUISHED` -- the arms built different pipelines, so a timing difference is
    attributable to a code-path difference at all.

    :data:`NOT_DISTINGUISHED` -- both arms built GQA pipelines and their witnesses are equal.
    Every timing may still be real and the clock may still show a difference; nothing here
    shows the two builds ran different code.  :func:`gated_verdict` refuses a speed verdict on
    this.

    Raises :class:`SchemaError` when either witness is absent -- an unwitnessed arm is not a
    third class, it is a missing input.
    """
    for name, w in (("candidate", candidate), ("baseline", baseline)):
        if not isinstance(w, dict) or not w.get("present"):
            raise SchemaError(f"the {name} arm carries no path witness; nothing to classify")
    ck = tuple(sorted(candidate.get("gqa_keys") or ()))
    bk = tuple(sorted(baseline.get("gqa_keys") or ()))
    if not ck and not bk:
        return NO_GQA
    if ck != bk:
        return DISTINGUISHED
    return NOT_DISTINGUISHED


def pair_repeats(records, workload: str, *, models=None) -> dict:
    """Pair one workload's records within each repeat and return its row.

    Pairing is **within** a repeat, never across: the two arms of a repeat ran adjacent in
    time, so a shared disturbance largely cancels in their ratio, which is the whole reason
    this design is usable on a box docs/PERF.md s20 declares permanently contended.

    A repeat contributes a ratio only when **both** its records are admissible.  A dropped
    repeat is recorded, never silently skipped.  Raises :class:`SchemaError` if the workload
    is absent or its arms are not the declared two.
    """
    if not isinstance(workload, str) or not workload:
        raise SchemaError(f"workload must be a non-empty name, got {workload!r}")
    mine = [r for r in records if isinstance(r, dict) and r.get("workload") == workload]
    if not mine:
        raise SchemaError(f"no records for workload {workload!r}")
    by_cell = {}
    for r in mine:
        by_cell[(r.get("arm"), r.get("repeat"))] = r
    arms = {r.get("arm") for r in mine}
    if arms != set(EXPECTED_ARMS):
        raise SchemaError(
            f"{workload}: arms are {sorted(arms)}, expected {sorted(EXPECTED_ARMS)}"
        )

    per_repeat, refusals, dropped = [], [], []
    witness_classes = set()
    islands = {}
    for repeat in sorted({r.get("repeat") for r in mine}):
        cand = by_cell.get(("candidate", repeat))
        base = by_cell.get(("baseline", repeat))
        if cand is None or base is None:
            dropped.append({"repeat": repeat, "why": ["repeat_incomplete"]})
            continue
        why = record_refusals(cand, models=models) + record_refusals(base, models=models)
        if why:
            dropped.append({"repeat": repeat, "why": sorted(set(why))})
            refusals.extend(sorted(set(why)))
            continue
        witness_classes.add(witness_class(cand["path_witness"], base["path_witness"]))
        for r in (cand, base):
            calls = r["path_witness"].get("compute_calls")
            if not isinstance(calls, int) or calls <= 0 or calls % SESSIONS_PER_RECORD:
                refusals.append(
                    f"witness_session_count: compute_calls={calls} is not a whole multiple of "
                    f"{SESSIONS_PER_RECORD}"
                )
            else:
                islands.setdefault(r["arm"], set()).add(calls // SESSIONS_PER_RECORD)
            islands.setdefault("claimed_nodes", set()).add(r["path_witness"].get("claimed_nodes"))
        cm = _median(cand["speed"]["samples_ms"])
        bm = _median(base["speed"]["samples_ms"])
        if cm <= 0 or bm <= 0:
            dropped.append({"repeat": repeat, "why": ["nonpositive_median"]})
            continue
        per_repeat.append(
            {
                "repeat": repeat,
                "candidate_median_ms": cm,
                "baseline_median_ms": bm,
                "ratio": bm / cm,
                "candidate_rsd_pct": 100.0 * cand["speed"]["stdev_ms"] / cand["speed"]["mean_ms"],
                "baseline_rsd_pct": 100.0 * base["speed"]["stdev_ms"] / base["speed"]["mean_ms"],
                "cross_arm_bitwise_identical": (
                    cand["outputs_sha256"] == base["outputs_sha256"]
                    and cand["feeds_sha256"] == base["feeds_sha256"]
                ),
                "candidate_pid": cand.get("pid"),
                "baseline_pid": base.get("pid"),
            }
        )

    # A witness lifted from a different workload carries that workload's island count or its
    # claimed-node count. Both are constant within a workload and identical across arms here,
    # so a disagreement is a borrowed witness rather than a measurement.
    if len(islands.get("claimed_nodes", {None})) > 1:
        refusals.append(
            f"witness_borrowed: claimed_nodes disagree within one workload "
            f"({sorted(islands['claimed_nodes'])}) -- a witness recorded in another process"
        )
    cand_islands = islands.get("candidate", set())
    base_islands = islands.get("baseline", set())
    if cand_islands and base_islands and cand_islands != base_islands:
        refusals.append(
            f"witness_borrowed: the arms disagree on island count "
            f"(candidate {sorted(cand_islands)} vs baseline {sorted(base_islands)})"
        )

    if len(witness_classes) > 1:
        refusals.append(
            f"witness_unstable: repeats disagree on what the arms prove {sorted(witness_classes)}"
        )
    ratios = [p["ratio"] for p in per_repeat]
    return {
        "workload": workload,
        "model_key": mine[0].get("model_key"),
        "per_repeat": per_repeat,
        "ratios": ratios,
        "ratio_min": min(ratios) if ratios else None,
        "ratio_median": _median(ratios) if ratios else None,
        "ratio_max": max(ratios) if ratios else None,
        "candidate_median_ms": (
            _median([p["candidate_median_ms"] for p in per_repeat]) if per_repeat else None
        ),
        "baseline_median_ms": (
            _median([p["baseline_median_ms"] for p in per_repeat]) if per_repeat else None
        ),
        "repeats_paired": len(per_repeat),
        "repeats_dropped": dropped,
        "refusals": sorted(set(refusals)),
        "witness_class": sorted(witness_classes)[0] if len(witness_classes) == 1 else None,
        "cross_arm_bitwise_identical": all(
            p["cross_arm_bitwise_identical"] for p in per_repeat
        )
        if per_repeat
        else False,
        "gqa_keys": {
            "candidate": sorted(
                {
                    tuple(by_cell[("candidate", rp)]["path_witness"].get("gqa_keys") or ())
                    for rp in sorted({r.get("repeat") for r in mine})
                    if ("candidate", rp) in by_cell
                    and by_cell[("candidate", rp)].get("path_witness")
                }
            ),
            "baseline": sorted(
                {
                    tuple(by_cell[("baseline", rp)]["path_witness"].get("gqa_keys") or ())
                    for rp in sorted({r.get("repeat") for r in mine})
                    if ("baseline", rp) in by_cell
                    and by_cell[("baseline", rp)].get("path_witness")
                }
            ),
        },
    }


def calibration_band(rows, *, floor: float = BAND_FLOOR) -> dict:
    """Derive the band from the calibration rows ONLY. This is the fix for B2.

    A workload is calibration iff its witness class is :data:`NO_GQA` -- neither build ever
    created a ``gqa_f16`` pipeline, so the change under test provably cannot have moved it and
    every bit of its spread is drift.  The band is the largest absolute deviation from unity
    those workloads showed, floored at :data:`BAND_FLOOR`.

    The rows the band is computed from are exactly the rows :func:`gated_verdict` refuses to
    grade.  That disjointness is the whole point: PR #95 computed the band from the null
    control's own three ratios and then graded that null control with it, which made ``FASTER``
    and ``SLOWER`` unreachable for it in every possible universe.

    Raises :class:`SchemaError` when no calibration row exists -- a band with no calibration
    behind it would be a number someone chose, and this function will not pretend otherwise.
    """
    if not isinstance(floor, (int, float)) or floor <= 0:
        raise SchemaError(f"band floor must be a positive fraction, got {floor!r}")
    calib = [
        r
        for r in rows
        if r.get("witness_class") == NO_GQA
        and r.get("ratios")
        and not r.get("refusals")
        and r.get("repeats_paired") == EXPECTED_REPEATS
        and r.get("cross_arm_bitwise_identical")
    ]
    if not calib:
        raise SchemaError(
            "no admissible calibration workload: every row either built a gqa_f16 pipeline, "
            "failed a premise, or paired fewer than the full set of repeats, so there is "
            "nothing the landed change provably could not have moved. Refusing to invent a "
            "band."
        )
    ratios = [x for r in calib for x in r["ratios"]]
    envelope = max(abs(x - 1.0) for x in ratios)
    return {
        "applied": max(floor, envelope),
        "floor": floor,
        "floor_binds": floor > envelope,
        "calibration_envelope": envelope,
        "calibration_workloads": sorted(r["workload"] for r in calib),
        "calibration_ratios": sorted(ratios),
        "calibration_ratio_min": min(ratios),
        "calibration_ratio_max": max(ratios),
        "n_calibration_ratios": len(ratios),
        "rule": BAND_RULE,
        "provenance": BAND_PROVENANCE,
    }


def raw_verdict(ratios, band: float) -> str:
    """The symmetric decision rule of :data:`DECISION_RULE`, and nothing else.

    Deliberately ignorant of witnesses, refusals and roles: it sees three numbers and a band.
    Everything that can turn a number into a non-claim lives in :func:`gated_verdict`, so that
    the gate is a separate, deletable, and therefore *testable* thing.

    Raises :class:`SchemaError` on an empty ratio list or a non-positive band.
    """
    if not isinstance(band, (int, float)) or band <= 0:
        raise SchemaError(f"band must be a positive fraction, got {band!r}")
    values = list(ratios or [])
    if not values:
        raise SchemaError("no paired ratios; there is no verdict to render")
    lo, hi = 1.0 - band, 1.0 + band
    if min(values) > hi:
        return "FASTER"
    if max(values) < lo:
        return "SLOWER"
    if all(lo <= v <= hi for v in values):
        return "NEUTRAL"
    return "INDETERMINATE"


def gated_verdict(row, band: float) -> dict:
    """THE GATE. Every published verdict in this repository comes out of this function.

    Four things can stop a real timing from becoming a claim, and all four live here:

    1. **Role.** A :data:`NO_GQA` workload defined the band, so it is graded ``CALIBRATION``
       and never ``FASTER``/``SLOWER``/``NEUTRAL``.  Circularity is removed by construction
       rather than by disclaimer.
    2. **Witness.** A subject whose two arms produced *identical* GQA path witnesses has not
       been shown to have run different code, so ``FASTER``/``SLOWER`` becomes ``REFUSED``
       even though every millisecond in it is real.  This is B1.
    3. **Refusals.** Anything :func:`record_refusals` or :func:`pair_repeats` found.
    4. **Completeness.** Fewer than :data:`EXPECTED_REPEATS` paired repeats is a refusal, not
       a smaller sample: dropping a repeat and keeping the verdict is how a bad record turns
       into a good number.

    Deleting or short-circuiting any of the four turns ``bench/test_crossbuild_summary.py``
    red.  That is demonstrated there by mutating this file's source, not asserted here.

    Raises :class:`SchemaError` on a row that is not a :func:`pair_repeats` output.
    """
    if not isinstance(row, dict) or "workload" not in row or "ratios" not in row:
        raise SchemaError(f"not a paired row: {type(row).__name__}")
    if not isinstance(band, (int, float)) or band <= 0:
        raise SchemaError(f"band must be a positive fraction, got {band!r}")

    reasons: list[str] = []
    raw = raw_verdict(row["ratios"], band) if row["ratios"] else None
    if raw is None:
        return {
            "raw_verdict": None,
            "verdict": "REFUSED",
            "role": "subject",
            "gate_reasons": ["no_paired_repeats"] + list(row.get("refusals") or []),
            "band": band,
        }

    role = "calibration" if row.get("witness_class") == NO_GQA else "subject"
    verdict = raw

    # Blocking premises apply to BOTH roles. A corrupted control must not set the band, so a
    # calibration row that fails anything here is refused too -- `calibration_band` drops it.
    if row.get("witness_class") is None:
        reasons.append("witness_unclassified: the repeats do not agree on what the arms prove")
    if row.get("refusals"):
        reasons.extend(row["refusals"])
    if row.get("repeats_paired") != EXPECTED_REPEATS:
        reasons.append(
            f"incomplete: {row.get('repeats_paired')} of {EXPECTED_REPEATS} repeats paired"
        )
    if not row.get("cross_arm_bitwise_identical"):
        reasons.append(
            "outputs_differ_across_arms: the two builds did not compute the same thing"
        )

    if reasons:
        verdict = "REFUSED"
    elif role == "calibration":
        reasons.append(
            "calibration: this workload's ratios define the band, so it is not graded by it"
        )
        verdict = "CALIBRATION"
    elif row.get("witness_class") == NOT_DISTINGUISHED and raw in ("FASTER", "SLOWER"):
        reasons.append(
            "witness_does_not_distinguish_arms: both builds produced the same gqa_f16 "
            "pipeline key, so nothing here shows they ran different code"
        )
        verdict = "REFUSED"

    return {
        "raw_verdict": raw,
        "verdict": verdict,
        "role": role,
        "gate_reasons": reasons,
        "band": band,
        "separation": {
            "faster_needs_min_above": 1.0 + band,
            "slower_needs_max_below": 1.0 - band,
            "observed_min": min(row["ratios"]),
            "observed_max": max(row["ratios"]),
            "largest_band_still_faster": min(row["ratios"]) - 1.0,
            "largest_band_still_slower": 1.0 - max(row["ratios"]),
        },
    }


def dispatch_grid_claim(row) -> dict:
    """What the artifact witnesses about the dispatch grid, and what is only inferred.

    Issue #96 quotes a decode grid of ``[32, 1, 1]``.  **No field of the frozen artifact holds
    a grid**; there is no ``grid``, ``dispatch_grid`` or ``spec_const`` key anywhere in it.
    Two statements must therefore be kept apart, and this function keeps them apart:

    * **WITNESSED.**  The candidate's key is ``gqa_f16:<local>``, so ``local`` is read off the
      artifact.  When ``local == 1``, ``ops::attention`` dispatches
      ``total.div_ceil(1) == total`` workgroups and the baseline dispatches ``total`` -- so the
      grids are *equal across the arms* whatever ``total`` is.  That equality needs no model
      knowledge and is exact.
    * **INFERRED.**  The value ``[32, 1, 1]`` needs Phi-3.5's 32 query heads and ``M == 1``,
      neither of which the artifact records.  From the witness alone the rule bounds
      ``total`` to ``[32, 64)`` when ``local == 1``, which contains 32 but does not pin it.

    Raises :class:`SchemaError` on a row with no witness class.
    """
    if not isinstance(row, dict) or "gqa_keys" not in row:
        raise SchemaError(f"not a paired row: {type(row).__name__}")
    keys = [k for group in row["gqa_keys"]["candidate"] for k in group]
    locals_ = set()
    for key in keys:
        _, _, tail = key.partition(":")
        if tail.isdigit():
            locals_.add(int(tail))
    if not locals_:
        return {
            "witnessed": "no gqa_f16 pipeline was built on either arm",
            "grids_equal_across_arms": None,
            "inferred_grid": None,
            "inference_inputs": [],
        }
    local = sorted(locals_)[0] if len(locals_) == 1 else None
    equal = local == 1
    return {
        "witnessed": (
            f"candidate gqa_f16 local size {sorted(locals_)} (read off the pipeline key); "
            f"baseline key carries no local size, i.e. spec_constants was empty"
        ),
        "grids_equal_across_arms": equal,
        "grids_equal_because": (
            "local == 1 makes the candidate's ceil(total/local) == total, which is exactly the "
            "baseline's workgroup count, for every total"
            if equal
            else "local > 1 packs the same invocations into fewer workgroups, so the grids differ"
        ),
        "inferred_grid": [32, 1, 1] if equal else None,
        "inference_inputs": (
            [
                "Phi-3.5-mini has 32 query heads (docs/DESIGN.md s8.13)",
                "M == 1 for this workload, so total = B*Nq*S = 32",
                "ops::attention::gqa_local_size(32) == 1 (rust/src/ops/attention.rs)",
            ]
            if equal
            else []
        ),
        "inferred_not_witnessed": (
            "the artifact records no grid field; total is bounded to [32, 64) by the rule when "
            "local == 1, which contains 32 without pinning it"
            if equal
            else None
        ),
    }


def within_arm_dispersion(records, workload: str) -> dict:
    """The offline A/A surrogate of :data:`WITHIN_ARM_RULE`. Same build against itself.

    A cross-arm ratio only means something to the degree that it exceeds what one build does
    against *itself* on this host.  The real A/A run does not exist yet -- that is issue #96 --
    so this computes two surrogates from records that already exist, and re-measures nothing:

    * **across-repeat** -- one arm's three repeat medians against each other.  Unpaired, three
      separate processes, so it carries all of this box's slow drift.  It is the upper bound on
      what within-repeat pairing has to cancel.
    * **split-half** -- the first ten timed samples of a single record against its last ten.
      Paired and adjacent in time, so it is the same *shape* of comparison the cross-arm ratio
      is, with the build held constant.

    The published envelope is the larger of the two, over both arms.  This function renders no
    verdict and is wired into no gate: :func:`gated_verdict` never sees it.  It exists so that
    ``M = 128``'s 107.5% and ``decode past = 128``'s 13.7% can be read against the noise each
    one sits in, instead of against each other.

    Raises :class:`SchemaError` when the workload is missing, an arm is missing, or a record
    carries too few timed samples to halve.
    """
    if not isinstance(workload, str) or not workload:
        raise SchemaError(f"workload must be a non-empty name, got {workload!r}")
    mine = [r for r in records if isinstance(r, dict) and r.get("workload") == workload]
    if not mine:
        raise SchemaError(f"no records for workload {workload!r}")
    arms = {r.get("arm") for r in mine}
    if arms != set(EXPECTED_ARMS):
        raise SchemaError(
            f"{workload}: within-arm dispersion needs both arms, got {sorted(arms)}"
        )

    per_arm = {}
    for arm in sorted(EXPECTED_ARMS):
        rows = sorted(
            (r for r in mine if r.get("arm") == arm), key=lambda r: r.get("repeat", -1)
        )
        medians, split_half, rsd = [], [], []
        for r in rows:
            samples = (r.get("speed") or {}).get("samples_ms")
            if not isinstance(samples, list) or len(samples) < 4:
                raise SchemaError(
                    f"{workload}/{arm}/repeat {r.get('repeat')}: {len(samples or [])} timed "
                    f"samples is too few to halve; there is no within-arm dispersion to report"
                )
            half = len(samples) // 2
            lo, hi = _median(samples[:half]), _median(samples[half:])
            if lo <= 0 or hi <= 0:
                raise SchemaError(
                    f"{workload}/{arm}/repeat {r.get('repeat')}: non-positive half median"
                )
            medians.append(_median(samples))
            split_half.append(abs(lo / hi - 1.0))
            speed = r["speed"]
            rsd.append(
                100.0 * speed["stdev_ms"] / speed["mean_ms"] if speed.get("mean_ms") else None
            )
        if len(medians) < 2:
            raise SchemaError(
                f"{workload}/{arm}: {len(medians)} repeat(s); an across-repeat A/A surrogate "
                f"needs at least two runs of the same build"
            )
        across = [
            abs(a / b - 1.0)
            for i, a in enumerate(medians)
            for j, b in enumerate(medians)
            if i != j
        ]
        per_arm[arm] = {
            "repeat_medians_ms": medians,
            "across_repeat_ratios": across,
            "across_repeat_envelope": max(across),
            "split_half_deviations": split_half,
            "split_half_envelope": max(split_half),
            "per_record_rsd_pct": rsd,
        }

    across_env = max(a["across_repeat_envelope"] for a in per_arm.values())
    split_env = max(a["split_half_envelope"] for a in per_arm.values())
    return {
        "workload": workload,
        "rule": WITHIN_ARM_RULE,
        "per_arm": per_arm,
        "across_repeat_envelope": across_env,
        "split_half_envelope": split_env,
        "within_arm_envelope": max(across_env, split_env),
        "n_aa_comparisons": sum(
            len(a["across_repeat_ratios"]) + len(a["split_half_deviations"])
            for a in per_arm.values()
        ),
        "is_a_real_aa_run": False,
        "why_not": (
            "these are the SAME records the cross-arm ratio uses, re-cut two ways. A real A/A "
            "-- two independent runs of one build, scheduled like the cross-build run was -- "
            "has not been done on this host. Issue #96 owns it."
        ),
    }


def separation_from_drift(row, dispersion) -> dict:
    """How far a workload's weakest paired repeat sits above this host's own noise.

    ``cross_arm_effect`` is deliberately the **minimum** ``|ratio - 1|`` across the repeats,
    not the median: the weakest repeat is what the claim actually rests on.

    Raises :class:`SchemaError` on a row with no ratios or a dispersion with a non-positive
    envelope -- an envelope of zero would make every effect infinitely separated, which is the
    one answer this function must never be able to give.
    """
    if not isinstance(row, dict) or "ratios" not in row:
        raise SchemaError(f"not a paired row: {type(row).__name__}")
    if not row.get("ratios"):
        raise SchemaError(f"{row.get('workload')!r} has no paired ratios to separate")
    envelope = (dispersion or {}).get("within_arm_envelope")
    if not isinstance(envelope, (int, float)) or envelope <= 0:
        raise SchemaError(
            f"within_arm_envelope must be a positive fraction, got {envelope!r}; a zero "
            f"envelope would report every effect as infinitely separated from noise"
        )
    ratios = row["ratios"]
    effect = min(abs(r - 1.0) for r in ratios)
    if all(r > 1.0 for r in ratios):
        direction = "FASTER_SIDE"
    elif all(r < 1.0 for r in ratios):
        direction = "SLOWER_SIDE"
    else:
        direction = "MIXED"
    ratio = effect / envelope
    return {
        "cross_arm_effect": effect,
        "cross_arm_effect_pct": 100.0 * effect,
        "within_arm_envelope": envelope,
        "within_arm_envelope_pct": 100.0 * envelope,
        "separation_ratio": ratio,
        "threshold": SEPARATION_STRONG,
        "class": SEPARATED if ratio >= SEPARATION_STRONG else NOT_SEPARATED,
        "direction": direction,
        "all_repeats_same_side": direction != "MIXED",
    }


def descriptive_status(row, sep) -> dict:
    """Label a row's *epistemic* standing, separately from its verdict.

    The band decides what may be claimed.  This decides what an un-claimable but consistently
    directional row may be *called*, which is the question ``decode past = 128`` raises: three
    repeats all below 1, none of them clearing the band.  Calling that ``SLOWER`` is what PR #95
    did and is wrong; calling it nothing at all hides it.  It is a **provisional descriptive
    ratio**, and :data:`PROVISIONAL_UNTIL` states the exact condition that retires the label.

    A graded ``FASTER``/``SLOWER`` keeps its verdict here regardless of separation -- the verdict
    comes from :func:`gated_verdict` and nothing in this module may quietly overturn it -- but
    the separation class rides along, because a claim whose magnitude sits inside the host's own
    drift may be quoted as a floor and never as a point estimate.

    Raises :class:`SchemaError` on a row with no verdict.
    """
    if not isinstance(row, dict) or "verdict" not in row:
        raise SchemaError(f"not a graded row: {type(row).__name__}")
    verdict = row["verdict"]
    if verdict in ("CALIBRATION", "REFUSED"):
        return {
            "status": verdict,
            "quotable_as": None,
            "until": None,
            "separation_class": (sep or {}).get("class"),
        }
    direction = (sep or {}).get("direction")
    separated = (sep or {}).get("class") == SEPARATED
    if verdict in ("FASTER", "SLOWER"):
        return {
            "status": "CLAIM",
            "quotable_as": "point_estimate" if separated else "floor_only",
            "until": None,
            "separation_class": (sep or {}).get("class"),
            "note": (
                "the magnitude clears this host's own within-arm envelope, so the median ratio "
                "is quotable"
                if separated
                else "the magnitude sits inside this host's own within-arm envelope, so only "
                "the weakest repeat is quotable and the median ratio is not"
            ),
        }
    if verdict == "INDETERMINATE" and direction in ("FASTER_SIDE", "SLOWER_SIDE"):
        return {
            "status": PROVISIONAL_DESCRIPTIVE,
            "quotable_as": "description_only",
            "until": PROVISIONAL_UNTIL,
            "separation_class": (sep or {}).get("class"),
            "direction": direction,
            "note": (
                "every repeat falls on one side of 1, but the band will not grade it, so it is "
                "described and not concluded"
            ),
        }
    return {
        "status": "NO_DIRECTION",
        "quotable_as": None,
        "until": None,
        "separation_class": (sep or {}).get("class"),
    }


def summarize(artifact=None) -> dict:
    """The one authoritative summary. The CLI publishes it; the tests assert against it.

    Order matters and is the correction: rows are paired, the band is derived from the
    calibration rows, and only then is every row put through :func:`gated_verdict`.  Nothing
    upstream of the band can see a subject's ratios, and nothing downstream can bypass the
    gate.
    """
    doc = load_frozen() if artifact is None else artifact
    records = doc.get("records")
    if not isinstance(records, list) or not records:
        raise SchemaError("artifact carries no records")
    for block in SUPERSEDED_BLOCKS:
        if block in doc:
            raise ProvenanceError(
                f"artifact still carries the superseded `{block}` block from PR #95; "
                f"load it through load_frozen() so the rejected derivation cannot be quoted"
            )

    workloads = sorted({r["workload"] for r in records})
    models = doc.get("models") or {}
    rows = [pair_repeats(records, w, models=models) for w in workloads]
    band = calibration_band(rows)
    for row in rows:
        row.update(gated_verdict(row, band["applied"]))
        row["dispatch_grid"] = dispatch_grid_claim(row)
        row["within_arm"] = within_arm_dispersion(records, row["workload"])
        row["separation_from_drift"] = separation_from_drift(row, row["within_arm"])
        row["descriptive_status"] = descriptive_status(row, row["separation_from_drift"])

    subjects = [r for r in rows if r["role"] == "subject"]
    return {
        "schema": SUMMARY_SCHEMA,
        "issue": 69,
        "supersedes": {
            "pull_request": SOURCE_PR,
            "head": SOURCE_HEAD,
            "what_is_reused": "the 60 raw records, byte-identically",
            "what_is_rebuilt": [
                "the admissibility gate, the witness gate and the verdict rule (B1)",
                "the band, which no longer comes from the workload it judges (B2)",
                "every published table, recomputed from raw samples_ms",
            ],
        },
        "measurement_provenance": doc.get("_frozen"),
        "question": doc.get("question"),
        "not_a_claim_about": doc.get("not_a_claim_about"),
        "arms": doc.get("arms"),
        "environment": doc.get("environment"),
        "models": doc.get("models"),
        "model_pins": MODEL_PINS,
        "compiled_input_delta": COMPILED_INPUT_DELTA,
        "exclusivity": doc.get("exclusivity"),
        "exclusivity_language": EXCLUSIVITY_LANGUAGE,
        "decision_rule": DECISION_RULE,
        "within_arm_rule": WITHIN_ARM_RULE,
        "provisional_until": PROVISIONAL_UNTIL,
        "band": band,
        "rows": rows,
        "counts": {
            "records": len(records),
            "workloads": len(rows),
            "calibration": len(rows) - len(subjects),
            "subjects": len(subjects),
            "admissible_records": sum(1 for r in records if not record_refusals(r, models=models)),
            "verdicts": {v: sum(1 for r in rows if r["verdict"] == v) for v in VERDICTS},
            "descriptive_statuses": {
                s: sum(1 for r in rows if r["descriptive_status"]["status"] == s)
                for s in sorted({r["descriptive_status"]["status"] for r in rows})
            },
        },
        "headline": headline(rows),
        "sensitivity": sensitivity(rows),
    }


def headline(rows) -> dict:
    """The claim, in the exact words it may be said in, derived from the rows -- not typed in.

    Every number here is read out of :func:`pair_repeats` output, so the sentence in
    ``docs/PERF.md`` and the sentence in the PR cannot drift away from the records the way
    PR #95's title did.  The scope restriction is part of the payload: the subject is **one
    model, Phi-3.5-mini, on its prefill phase**, and the word chosen for it is singular.

    Raises :class:`SchemaError` if the graded rows do not include the prefill subject the claim
    is about -- a headline with nothing under it must not render.
    """
    by = {r["workload"]: r for r in rows}
    claimed = sorted(
        (w for w, r in by.items() if r["verdict"] in ("FASTER", "SLOWER")),
        key=lambda w: -by[w]["ratio_median"],
    )
    if not claimed:
        raise SchemaError("no graded FASTER/SLOWER row; there is no headline to render")
    models_claimed = sorted({by[w]["model_key"] for w in claimed})
    if len(models_claimed) != 1:
        raise SchemaError(
            f"the claim spans {len(models_claimed)} models {models_claimed}; the scope "
            f"restriction in this function is written for exactly one"
        )
    parts = []
    for w in claimed:
        r = by[w]
        floor = r["descriptive_status"]["quotable_as"] == "floor_only"
        parts.append(
            {
                "workload": w,
                "verdict": r["verdict"],
                "quote": (
                    f"at least {r['ratio_min']:.3f}x"
                    if floor
                    else f"{r['ratio_median']:.3f}x"
                ),
                "ratio_min": r["ratio_min"],
                "ratio_median": r["ratio_median"],
                "ratio_max": r["ratio_max"],
                "quotable_as": r["descriptive_status"]["quotable_as"],
                "separation_ratio": r["separation_from_drift"]["separation_ratio"],
            }
        )
    provisional = sorted(
        w
        for w, r in by.items()
        if r["descriptive_status"]["status"] == PROVISIONAL_DESCRIPTIVE
    )
    return {
        "model": models_claimed[0],
        "scope": (
            "ONE model, Phi-3.5-mini-instruct (int4 RTN block-32), prefill phase, on this one "
            "contended Windows/lavapipe host. Singular on purpose: this is not a claim about "
            "real models in general, about decode, about any other model in the run, or about "
            "any other device or backend."
        ),
        "claims": parts,
        "indeterminate": sorted(w for w, r in by.items() if r["verdict"] == "INDETERMINATE"),
        "provisional_descriptive": provisional,
        "not_claimed": (
            "MobileNetV2 and MiniLM are calibration controls and are never graded; decode is "
            "not claimed in either direction; no CUDA, no other device, no other backend."
        ),
        "measured_by": "PR #95's frozen run, reused byte-identically and not re-measured here",
    }


def sensitivity(rows, bands=SENSITIVITY_BANDS) -> dict:
    """Every subject's verdict at every published band, so the band is not a hiding place.

    A single band is a choice, and this one was made after the data existed
    (:data:`BAND_PROVENANCE`).  The honest compensation is to show the whole function rather
    than one point of it: a reader who thinks 5% is right, or 25%, can read their answer here.

    Raises :class:`SchemaError` on an empty band list or a non-positive band.
    """
    values = list(bands or [])
    if not values:
        raise SchemaError("no bands to sweep; a sensitivity analysis of nothing is not one")
    out = {}
    for row in rows:
        if not row.get("ratios"):
            continue
        out[row["workload"]] = {
            "role": "calibration" if row.get("witness_class") == NO_GQA else "subject",
            "ratios": row["ratios"],
            "at_band": {
                f"{b:.4f}": (
                    "CALIBRATION"
                    if row.get("witness_class") == NO_GQA
                    else raw_verdict(row["ratios"], b)
                )
                for b in values
            },
        }
    return {
        "bands": list(values),
        "note": (
            "CALIBRATION rows are shown with their raw ratios but are never graded, at any "
            "band. Subject rows show the verdict the symmetric rule returns."
        ),
        "by_workload": out,
    }


def markdown_table(summary) -> str:
    """Render the published table. docs/PERF.md quotes this and a test recomputes every cell.

    Raises :class:`SchemaError` on anything that is not a :func:`summarize` output.
    """
    if not isinstance(summary, dict) or summary.get("schema") != SUMMARY_SCHEMA:
        raise SchemaError(
            f"expected a {SUMMARY_SCHEMA} summary, got "
            f"{summary.get('schema') if isinstance(summary, dict) else type(summary).__name__}"
        )
    lines = [
        "| workload | role | candidate | baseline | ratio (median) | per-repeat | verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in summary["rows"]:
        per = " / ".join(f"{r:.3f}" for r in row["ratios"]) if row["ratios"] else "—"
        lines.append(
            f"| {row['workload']} | {row['role']} | {row['candidate_median_ms']:.2f} ms "
            f"| {row['baseline_median_ms']:.2f} ms | {row['ratio_median']:.3f}× | {per} "
            f"| {row['verdict']} |"
        )
    return "\n".join(lines)


def dispersion_table(summary) -> str:
    """Render the offline within-arm (A/A surrogate) table. docs/PERF.md quotes this one too.

    Raises :class:`SchemaError` on anything that is not a :func:`summarize` output.
    """
    if not isinstance(summary, dict) or summary.get("schema") != SUMMARY_SCHEMA:
        raise SchemaError(
            f"expected a {SUMMARY_SCHEMA} summary, got "
            f"{summary.get('schema') if isinstance(summary, dict) else type(summary).__name__}"
        )
    lines = [
        "| workload | cross-arm effect (weakest repeat) | within-arm A/A envelope "
        "(across-repeat / split-half) | separation | status |",
        "|---|---|---|---|---|",
    ]
    for row in summary["rows"]:
        sep = row["separation_from_drift"]
        wa = row["within_arm"]
        lines.append(
            f"| {row['workload']} | {sep['cross_arm_effect_pct']:.2f}% "
            f"| {sep['within_arm_envelope_pct']:.2f}% "
            f"({100 * wa['across_repeat_envelope']:.2f}% / "
            f"{100 * wa['split_half_envelope']:.2f}%) "
            f"| {sep['separation_ratio']:.2f}× | {row['descriptive_status']['status']} |"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--finalize", action="store_true", help="write the derived summary artifact")
    ap.add_argument(
        "--out",
        default=str(RESULTS / "real_model_crossbuild_gqa_landing_v2.json"),
        help="where --finalize writes",
    )
    ap.add_argument("--markdown", action="store_true", help="print the published table")
    ap.add_argument(
        "--dispersion",
        action="store_true",
        help="print the offline within-arm (A/A surrogate) table",
    )
    ap.add_argument("--check", action="store_true", help="verify the frozen input and exit")
    args = ap.parse_args(argv)

    if args.check:
        doc = load_frozen()
        print(f"FROZEN OK  {doc['_frozen']['path']}  sha256={doc['_frozen']['sha256']}")
        print(f"           {len(doc['records'])} records, schema {SCHEMA}")
        print(f"           reused from PR #{SOURCE_PR} head {SOURCE_HEAD}; not re-measured")
        return 0

    summary = summarize()
    if args.markdown:
        print(markdown_table(summary))
        return 0
    if args.dispersion:
        print(dispersion_table(summary))
        return 0
    if args.finalize:
        out = Path(args.out)
        out.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {out} ({len(summary['rows'])} rows, band {summary['band']['applied']:.6f})")
        return 0

    band = summary["band"]
    print(f"band {band['applied']:.6f} from {band['n_calibration_ratios']} calibration ratios")
    print(markdown_table(summary))
    print()
    print(dispersion_table(summary))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    sys.exit(main())
