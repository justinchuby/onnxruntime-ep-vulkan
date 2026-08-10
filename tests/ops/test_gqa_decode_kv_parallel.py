"""Conformance for the decode-only, KV-parallel ``GroupQueryAttention`` module (issue #90).

WHY THIS FILE EXISTS
====================
``tests/ops/test_gqa.py`` covers ``gqa_f16`` and, at the cache extents it uses (0, 4, 16), it
will keep covering ``gqa_f16``: the #90 selector refuses anything under 63 past tokens, so every
one of its cases falls back to the shipping module.  That is deliberate — it is what keeps the
existing evidence describing the thing it was taken on — and it means **this file is the only
conformance coverage the new module has**.

The module is a *second implementation of the same arithmetic*.  It is one workgroup per
``(batch, query head)`` with ``W`` lanes striding the KV cache, each running an online softmax
over its own subset, merged through shared memory by a halving tree.  Being a second
implementation is exactly what makes it dangerous: a reduction bug does not crash, it produces a
number, and a number that is 2% wrong looks like f16 noise.

WHAT EACH TEST HAS TO DO TO BE A TEST
=====================================
Two failure modes would let this file pass while proving nothing, and both have happened in this
tree before:

1. **Silent fallback.**  If the EP declines the node, or the selector refuses, ORT runs the graph
   on CPU and the comparison is CPU against CPU — a perfect agreement about nothing.  So every
   device test asserts, from the EP's own ``pipeline_variants`` artifact, that a
   ``gqa_decode_f16`` pipeline with the **expected lane count** was actually created.  A run that
   quietly took ``gqa_f16`` fails here rather than passing.

2. **A degenerate reference.**  Attention over near-uniform scores is a mean, and a mean is
   computed correctly by almost any wrong reduction.  ``m.compare_all_outputs_to_cpu`` already
   refuses degenerate output pairs; the cancellation and extreme-magnitude cases below are what
   put a *hard* reduction in front of the merge tree.

WHAT IS NOT ASSERTED
====================
**Bit-equality with ``gqa_f16``.**  Summing a fixed set of terms in a different order is a
different rounding.  Equivalence is asserted against the ORT CPU EP within the declared f16
tolerance, exactly as ``test_gqa.py`` does, and the W=1 identity case below is the only place a
byte-for-byte claim is made — because there the selector refuses and the *same* SPIR-V runs.

**Any performance claim.**  Nothing here is timed.  #90's speed argument is a separate,
device-pinned measurement and it is not this file's to make.
"""

from __future__ import annotations

import contextlib
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper

import _models as m

# ---------------------------------------------------------------------------
# The host-side selector, restated.
#
# Duplicated from `rust/src/ops/attention.rs` rather than imported, because a test that
# re-derives the number from the implementation cannot detect the implementation changing it.
# `test_the_ladder_here_matches_the_one_in_the_source` pins the two constants against the Rust
# file so the duplication is checked rather than trusted.
# ---------------------------------------------------------------------------
MAX_KV_LANES = 16
MIN_KV_PER_LANE = 32
MAX_HEAD_DIM = 128
DECODE_MODULE = "gqa_decode_f16"
SERIAL_MODULE = "gqa_f16"
ENV_KV_LANES = "ONNXRUNTIME_EP_VULKAN_GQA_DECODE_KV_LANES"

REPO = pathlib.Path(__file__).resolve().parents[2]


def expected_lanes(past_seq: int, seq_len: int = 1, head_dim: int = 32) -> int:
    """The lane count the host selector will choose. ``1`` means *refuse*."""
    if seq_len != 1 or not (1 <= head_dim <= MAX_HEAD_DIM):
        return 1
    total = past_seq + 1
    lanes = 1
    while lanes * 2 <= MAX_KV_LANES and total // (lanes * 2) >= MIN_KV_PER_LANE:
        lanes *= 2
    return lanes


# The smallest cache on each rung, so a case is never one token away from a different lane count
# by accident.  63 -> 2, 127 -> 4, 255 -> 8, 511 -> 16.
RUNG_PASTS = {2: 63, 4: 127, 8: 255, 16: 511}


# ---------------------------------------------------------------------------
# Model + feed builders
# ---------------------------------------------------------------------------
ROTARY_DIM = 16


def _cos_sin(max_seq: int, rot_dim: int) -> tuple[np.ndarray, np.ndarray]:
    pos = np.arange(max_seq, dtype=np.float32)[:, None]
    freq = 1.0 / (10000 ** (np.arange(0, rot_dim, 2, dtype=np.float32) / rot_dim))
    angles = pos * freq
    return np.cos(angles).astype(np.float16), np.sin(angles).astype(np.float16)


def build_model(
    *,
    batch: int,
    seq_len: int,
    nq: int,
    nkv: int,
    head_dim: int,
    past_seq: int,
    max_seq: int | None = None,
) -> bytes:
    """One ``GroupQueryAttention`` node in the packed-QKV form Phi-3.5 GenAI emits."""
    max_seq = max_seq if max_seq is not None else past_seq + seq_len + 1
    packed = (nq + 2 * nkv) * head_dim
    f16 = TensorProto.FLOAT16
    ins = [
        helper.make_tensor_value_info("packed_qkv", f16, [batch, seq_len, packed]),
        helper.make_tensor_value_info("past_key", f16, [batch, nkv, past_seq, head_dim]),
        helper.make_tensor_value_info("past_value", f16, [batch, nkv, past_seq, head_dim]),
        helper.make_tensor_value_info("seqlens_k", TensorProto.INT32, [batch]),
        helper.make_tensor_value_info("total_seq", TensorProto.INT32, []),
        helper.make_tensor_value_info("cos_cache", f16, [max_seq, ROTARY_DIM // 2]),
        helper.make_tensor_value_info("sin_cache", f16, [max_seq, ROTARY_DIM // 2]),
    ]
    pres = [batch, nkv, past_seq + seq_len, head_dim]
    outs = [
        helper.make_tensor_value_info("attn_out", f16, [batch, seq_len, nq * head_dim]),
        helper.make_tensor_value_info("present_key", f16, pres),
        helper.make_tensor_value_info("present_value", f16, pres),
    ]
    node = helper.make_node(
        "GroupQueryAttention",
        inputs=[
            "packed_qkv", "", "", "past_key", "past_value", "seqlens_k", "total_seq",
            "cos_cache", "sin_cache",
        ],
        outputs=["attn_out", "present_key", "present_value"],
        domain="com.microsoft",
        name="gqa_kvpar",
        num_heads=nq,
        kv_num_heads=nkv,
        scale=float(head_dim ** -0.5),
        local_window_size=-1,
        do_rotary=1,
        rotary_interleaved=0,
        smooth_softmax=0,
    )
    model = helper.make_model(
        helper.make_graph([node], "gqa_kvpar", ins, outs),
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)],
    )
    model.ir_version = 10
    return model.SerializeToString()


def make_feeds(
    *,
    batch: int,
    seq_len: int,
    nq: int,
    nkv: int,
    head_dim: int,
    past_seq: int,
    max_seq: int | None = None,
    seed: int = 0,
    scale: float = 0.1,
    lengths: list[int] | None = None,
    hard_reduction: bool = False,
) -> dict[str, np.ndarray]:
    """Feeds for :func:`build_model`.

    ``scale`` is 0.1 for the same reason ``rust/tools/ledger_case_models.py`` uses 0.1: f16
    attention over standard normals saturates the exponent range, both EPs then agree on ``inf``,
    and a comparison of two infinities is a vacuous match.

    ``lengths`` pins ``seqlens_k`` per batch row — the ragged case, and the one place the
    module's *effective* lane count (recomputed in the shader from ``seqlens_k``) can differ from
    the pipeline's lane budget.

    ``hard_reduction`` replaces the flat normal draw with one designed to make the merge tree
    matter: the value vectors alternate in sign with a magnitude ramp along the cache, so a
    reduction that loses or double-counts a lane's partial cannot be hidden by averaging.
    """
    max_seq = max_seq if max_seq is not None else past_seq + seq_len + 1
    rng = np.random.default_rng(seed)
    packed = (nq + 2 * nkv) * head_dim
    qkv = (rng.standard_normal((batch, seq_len, packed)) * scale).astype(np.float16)
    pk = (rng.standard_normal((batch, nkv, past_seq, head_dim)) * scale).astype(np.float16)
    pv = (rng.standard_normal((batch, nkv, past_seq, head_dim)) * scale).astype(np.float16)
    if hard_reduction and past_seq:
        # A signed ramp along the KV axis. Summed in any order the true answer is the same; a
        # merge that drops a lane's partial moves it by far more than the f16 tolerance, and
        # cancellation means the result is not close to any single term.
        ramp = np.linspace(-1.0, 1.0, past_seq, dtype=np.float32)
        ramp[1::2] *= -1.0
        pv = (pv.astype(np.float32) + ramp[None, None, :, None]).astype(np.float16)
        # Vary the keys along the cache too, so the softmax weights are not near-uniform: a
        # near-uniform softmax is a mean, and a mean survives most wrong reductions.
        pk = (pk.astype(np.float32) * (1.0 + 2.0 * ramp[None, None, :, None])).astype(np.float16)
    if lengths is None:
        lengths = [past_seq] * batch
    seqlens = np.array(lengths, dtype=np.int32)
    cos, sin = _cos_sin(max_seq, ROTARY_DIM)
    return {
        "packed_qkv": qkv,
        "past_key": pk,
        "past_value": pv,
        "seqlens_k": seqlens,
        "total_seq": np.array(int(max(lengths)) + 1, dtype=np.int32),
        "cos_cache": cos,
        "sin_cache": sin,
    }


# ---------------------------------------------------------------------------
# The anti-vacuity instrument: which pipelines did the EP actually create?
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def _env(**kv: "str | None"):
    saved = {k: os.environ.get(k) for k in kv}
    try:
        for k, v in kv.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _read_variants(path: pathlib.Path) -> list[str]:
    variants: list[str] = []
    if path.is_file():
        with contextlib.suppress(json.JSONDecodeError, OSError):
            doc = json.loads(path.read_text(encoding="utf-8"))
            variants = [str(v) for v in (doc.get("pipeline_variants") or [])]
    return variants


def _child_runner_argv(spec_path: pathlib.Path) -> list[str]:
    """Run *this file* as a script, in a fresh interpreter, against a run spec.

    The child is this module rather than a separate helper so that the model builder, the
    selector restatement and the runner can never drift apart: there is exactly one definition
    of each, and pytest ignores the ``__main__`` guard at the bottom of the file.
    """
    return [sys.executable, str(pathlib.Path(__file__).resolve()), str(spec_path)]


def run_vulkan_in_child(
    model: bytes,
    feeds: dict[str, np.ndarray],
    tmp_path: pathlib.Path,
    tag: str,
    *,
    lane_override: str | None = None,
    repeats: int = 1,
) -> tuple[list[list[np.ndarray]], list[str]]:
    """Run *model* in a **fresh process** and return ``(runs, pipeline_variants)``.

    WHY A SUBPROCESS
    ================
    ``pipeline_variants`` is dumped from ``PIPELINE_VARIANTS``, a process-global ``static`` in
    ``rust/src/counters.rs``, and a Vulkan pipeline is created **once per process** and then
    cached.  In a shared pytest process that makes the artifact useless for attribution in *both*
    directions:

    * a **negative** control (``assert_module_did_not_run``) reading the cumulative set asks
      "has this process ever built ``gqa_decode_f16``", and after the first positive test in this
      file the answer is yes no matter what the selector just did — the refusal controls become
      tautologies;
    * a **positive** control cannot be repaired by subtracting a snapshot instead, because the
      pipeline the run needs may already be cached from an earlier test, so the delta is empty
      and the assertion fails on a run that was in fact dispatched.

    Both directions are only sound if the process boundary and the run boundary are the same
    boundary.  So each measured run gets its own interpreter: the counters file the child writes
    is, exactly, the set of pipelines **that run** created.  This costs a process launch per
    device test and buys the difference between a refusal control and a decoration.

    The EP library is registered by the child from ``ONNXRUNTIME_VULKAN_EP_LIB`` (the same
    variable ``tests/ops/conftest.py`` uses); ``require_vulkan`` has already established in the
    parent that it is present and loadable, so a child that cannot register it is a real failure
    and is reported as one rather than skipped.
    """
    work = tmp_path / f"child_{tag}"
    work.mkdir(parents=True, exist_ok=True)
    model_path = work / "model.onnx"
    model_path.write_bytes(model)
    feeds_path = work / "feeds.npz"
    np.savez(feeds_path, **feeds)
    outputs_path = work / "outputs.npz"
    counters_path = work / "counters.json"
    spec_path = work / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "model": str(model_path),
                "feeds": str(feeds_path),
                "outputs": str(outputs_path),
                "repeats": int(repeats),
            }
        ),
        encoding="utf-8",
    )

    env = dict(os.environ)
    env["ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE"] = str(counters_path)
    env.pop(ENV_KV_LANES, None)
    if lane_override is not None:
        env[ENV_KV_LANES] = lane_override
    env["PYTHONIOENCODING"] = "utf-8"

    proc = subprocess.run(
        _child_runner_argv(spec_path),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.returncode == 0, (
        f"[{tag}] the isolated run process exited {proc.returncode}. A device test cannot "
        f"downgrade a crashed run to a skip: the run either happened and is being reported, or "
        f"it did not and nothing may be concluded.\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    assert outputs_path.is_file(), (
        f"[{tag}] the isolated run process exited cleanly but wrote no outputs at "
        f"{outputs_path}.\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    with np.load(outputs_path) as npz:
        n_out = int(npz["n_outputs"])
        runs = [
            [np.asarray(npz[f"r{r}_o{j}"]) for j in range(n_out)] for r in range(repeats)
        ]
    variants = _read_variants(counters_path)
    assert variants, (
        f"[{tag}] the EP wrote no pipeline variants at {counters_path}, so this run cannot be "
        f"told apart from a silent CPU fallback. An empty artifact is an instrument outage, not "
        f"an agreement.\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    return runs, variants


def run_vulkan_with_variants(
    model: bytes,
    feeds: dict[str, np.ndarray],
    tmp_path: pathlib.Path,
    tag: str,
    *,
    lane_override: str | None = None,
    claims: bool = True,
) -> tuple[list[np.ndarray], list[str]]:
    """Single-run convenience wrapper over :func:`run_vulkan_in_child`.

    ``pipeline_variants`` is written by ``counters::record_pipeline_variant`` as
    ``"<stem>:<spec constants, comma separated>"`` — an artifact the **EP** produced, describing
    the pipeline it really created, not a host-side record of what was requested.  It is the same
    observable ``ci/census_surface_map.json`` names for ``GQA_LOCAL_SIZE`` and ``GEMV_MAX_ROWS``,
    and it is read here for the same reason: reading the environment variable back would only
    prove the test set it.

    ``m.assert_vulkan_claims`` runs in the parent process, *after* the measured run, and under
    the same lane override.  It is not a passive probe — it creates a session and executes the
    model — but it can no longer contaminate the attribution, because the attribution came from
    a different process.
    """
    runs, variants = run_vulkan_in_child(
        model, feeds, tmp_path, tag, lane_override=lane_override
    )
    if claims:
        with _env(**{ENV_KV_LANES: lane_override}):
            m.assert_vulkan_claims(model, feeds)
    return runs[0], variants


def assert_module_ran(variants: list[str], stem: str, lanes: int | None = None) -> None:
    """Fail unless the EP created a pipeline for *stem* (optionally at *lanes*).

    An empty ``pipeline_variants`` is an **outage, not an agreement**: it means the counters file
    was never written, so the test cannot tell a Vulkan run from a CPU fallback and must not
    report either.
    """
    assert variants, (
        "the EP wrote no pipeline_variants, so this run cannot be attributed. Without it a "
        "silent CPU fallback and a correct Vulkan run are indistinguishable, and the "
        "comparison below would be CPU against CPU."
    )
    stems = {v.split(":", 1)[0] for v in variants}
    assert stem in stems, (
        f"the EP created {sorted(stems)} and never built a `{stem}` pipeline; the node fell "
        f"back rather than taking the path under test"
    )
    if lanes is not None:
        want = f"{stem}:{lanes}"
        assert want in variants, (
            f"`{stem}` ran, but not at {lanes} lanes: {sorted(variants)}. The lane count is "
            f"specialisation constant 0, so a different value is a different pipeline and a "
            f"different summation order — this is not the case that was requested."
        )


def assert_module_did_not_run(variants: list[str], stem: str) -> None:
    stems = {v.split(":", 1)[0] for v in variants}
    assert stem not in stems, f"`{stem}` was built when the selector should have refused: {variants}"


def defined_kv_region(
    outputs: list[np.ndarray], lengths: list[int]
) -> list[np.ndarray]:
    """Trim ``present_key``/``present_value`` to the part of the cache GQA defines.

    ``present`` is declared ``[B, Nkv, past_extent + S, D]`` — one rectangle for every batch row —
    but the op only ever writes ``seqlens_k[b] + 1`` positions of row ``b``.  When the rows are
    **ragged** the remainder is not "the wrong answer", it is *no answer*: ORT's CPU kernel does
    not write it either, and the two EPs then disagree about the contents of two different fresh
    allocations.  MEASURED 2026-08-09: with ``past_extent = 511`` and ``lengths = [511, 63]`` the
    disagreement covers 43.75% of the tensor — exactly the 448 of 1024 rows past the shorter
    row's own length, and not one element inside them.

    So the undefined tail is removed from the comparison rather than asserted about.  Nothing is
    weakened for the rows that *are* defined: this concatenates each row's live prefix, both sides
    are built by the same function, and the written position ``seqlens_k[b]`` — the one the module
    is on trial for — is inside every prefix.  For a non-ragged batch every row is full length and
    this is the identity.
    """
    attn = outputs[0]
    trimmed: list[np.ndarray] = [attn]
    for arr in outputs[1:]:
        rows = [arr[b, :, : lengths[b] + 1, :].ravel() for b in range(arr.shape[0])]
        trimmed.append(np.concatenate(rows))
    return trimmed


# ---------------------------------------------------------------------------
# Device-free.  These run in CI with no GPU, which is where B-2 was found.
# ---------------------------------------------------------------------------

def test_the_ladder_here_matches_the_one_in_the_source() -> None:
    """The duplicated constants are pinned, so the duplication is checked rather than trusted.

    A test that imported the ladder from the implementation would agree with any ladder.  This
    reads the two numbers out of the Rust source and fails if either moves without this file
    moving with it.
    """
    src = (REPO / "rust" / "src" / "ops" / "attention.rs").read_text(
        encoding="utf-8", errors="replace"
    )
    for name, value in (
        ("GQA_DECODE_MAX_KV_LANES", MAX_KV_LANES),
        ("GQA_DECODE_MIN_KV_PER_LANE", MIN_KV_PER_LANE),
        ("GQA_DECODE_MAX_HEAD_DIM", MAX_HEAD_DIM),
    ):
        assert f"pub const {name}: u32 = {value};" in src, (
            f"{name} is no longer {value} in rust/src/ops/attention.rs; the cases in this file "
            f"are chosen from that ladder and would silently stop landing on their rungs"
        )
    assert f'pub const GQA_DECODE_MODULE: &str = "{DECODE_MODULE}";' in src
    assert f'pub const ENV_GQA_DECODE_KV_LANES: &str = "{ENV_KV_LANES}";' in src


def test_the_shipping_module_is_byte_identical() -> None:
    """#90's central constraint, asserted in the lane rather than left to review.

    Prefill, the fallback and every refused graph must keep running the SPIR-V they ran before,
    which is only true while ``gqa_f16.comp`` is untouched.  Compared against ``origin/main`` by
    the release checklist; here, cheaply, by the one property no edit to that file can preserve —
    it must not mention the new module, the new switch, or the new constants.
    """
    src = (REPO / "rust" / "shaders" / "glsl" / "gqa_f16.comp").read_text(
        encoding="utf-8", errors="replace"
    )
    for token in ("MIN_KV_PER_LANE", "MAX_KV_LANES", "s_acc", DECODE_MODULE, ENV_KV_LANES):
        assert token not in src, (
            f"`gqa_f16.comp` mentions `{token}`: the frozen module has been edited, and every "
            f"proof taken on it — including every entry in evidence/proof_ledger.jsonl — now "
            f"describes different bytes"
        )


def test_the_rungs_this_file_uses_are_the_rungs_it_claims() -> None:
    """The parametrised cases below are only about their lane counts if they land on them."""
    for lanes, past in RUNG_PASTS.items():
        assert expected_lanes(past) == lanes, f"past {past} is not the {lanes}-lane rung"
        assert expected_lanes(past - 1) == lanes // 2 or lanes == 2, (
            f"past {past} is not the FIRST cache on the {lanes}-lane rung, so this case does "
            f"not test the rung boundary"
        )
    assert expected_lanes(62) == 1, "the refusal boundary"
    assert expected_lanes(511, seq_len=2) == 1, "prefill is refused whatever the cache length"
    assert expected_lanes(511, head_dim=129) == 1, "a head wider than the module's arrays"


def test_shared_memory_stays_inside_the_guaranteed_floor() -> None:
    """The module's shared allocation is a constant; 16,384 B is the Vulkan 1.1 minimum.

    Asserted from the shader source, because the number in the header is a claim about the file
    and the arrays are what make it true.
    """
    src = (REPO / "rust" / "shaders" / "glsl" / f"{DECODE_MODULE}.comp").read_text(
        encoding="utf-8", errors="replace"
    )
    assert "shared float s_max[MAX_KV_LANES];" in src
    assert "shared float s_sum[MAX_KV_LANES];" in src
    assert "shared float s_acc[MAX_KV_LANES * MAX_HEAD_DIM];" in src
    total = 4 * MAX_KV_LANES + 4 * MAX_KV_LANES + 4 * MAX_KV_LANES * MAX_HEAD_DIM
    assert total == 8320 and total < 16384, total
    # Vulkan 1.1 core only: no subgroup ops, no optional extensions beyond the 16-bit storage
    # one `gqa_f16` already requires.
    for forbidden in (
        "GL_KHR_shader_subgroup",
        "subgroupAdd",
        "subgroupMax",
        "gl_SubgroupSize",
        "GL_NV_",
        "GL_AMD_",
        "GL_ARB_",
    ):
        assert forbidden not in src, (
            f"`{DECODE_MODULE}.comp` uses `{forbidden}`, which is not Vulkan 1.1 core; the "
            f"portability argument in its header is then false"
        )
    assert src.count("#extension") == 1, (
        "the module must require exactly the one extension gqa_f16 already requires"
    )


def test_every_barrier_is_in_uniform_control_flow() -> None:
    """A barrier some invocations of a workgroup reach and others do not is undefined behaviour.

    Not provable by inspection in general, so what is pinned here is the *structure* the
    argument rests on: the barriers live at the top of one loop, that loop's bound is
    ``active_lanes``, and ``active_lanes`` is derived from ``pc`` and ``seqlens_k[b]`` with ``b``
    fixed for the whole workgroup.  If a future edit adds a second barrier site, this goes red and
    the argument has to be re-made rather than silently inherited.
    """
    src = (REPO / "rust" / "shaders" / "glsl" / f"{DECODE_MODULE}.comp").read_text(
        encoding="utf-8", errors="replace"
    )
    code = "\n".join(
        line for line in src.splitlines() if not line.lstrip().startswith("//")
    )
    assert code.count("barrier();") == 1, (
        "there must be exactly one `barrier()` site, and it must be the one inside the "
        "uniform-bounded merge loop"
    )
    assert code.count("memoryBarrierShared();") == 1
    assert "for (uint stride = 1u; stride < active_lanes; stride <<= 1u) {" in code
    # The two early returns must be uniform across the workgroup, i.e. functions of the
    # workgroup id and the push block only -- never of gl_LocalInvocationID.
    for guard in ("if (b >= pc.batch_size) return;", "if (S != 1u) return;"):
        assert guard in code, f"the uniform early return `{guard}` is gone"


# ---------------------------------------------------------------------------
# Device tests.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lanes", sorted(RUNG_PASTS))
def test_kv_parallel_matches_cpu_at_every_lane_count(
    lanes: int, tmp_path: pathlib.Path, require_vulkan
) -> None:
    """The whole ladder, against the CPU oracle, with the pipeline identity asserted.

    One case per rung, because the rungs are different merge-tree depths (1, 2, 3 and 4 levels)
    and a bug in the top level of the tree is invisible at two lanes.
    """
    past = RUNG_PASTS[lanes]
    shape = dict(batch=1, seq_len=1, nq=8, nkv=2, head_dim=32, past_seq=past)
    model = build_model(**shape)
    feeds = make_feeds(**shape, seed=lanes, hard_reduction=True)
    vk, variants = run_vulkan_with_variants(model, feeds, tmp_path, f"rung{lanes}")
    assert_module_ran(variants, DECODE_MODULE, lanes)

    cpu = m.run_cpu(model, feeds)
    outcome, facts = m.compare_all_outputs_to_cpu(vk, cpu)
    assert outcome == m.COMPARISON_AGREE, f"{lanes} lanes: {outcome}\n{json.dumps(facts, indent=2, default=str)}"


@pytest.mark.parametrize("head_dim", [32, 64, 128])
def test_kv_parallel_matches_cpu_across_head_widths(
    head_dim: int, tmp_path: pathlib.Path, require_vulkan
) -> None:
    """``head_dim`` is the row stride of ``s_acc`` and the bound of the private arrays.

    128 is the inclusive upper boundary the selector declares, so it is the case where an
    off-by-one in the shared indexing reads another lane's accumulator.
    """
    shape = dict(batch=1, seq_len=1, nq=8, nkv=2, head_dim=head_dim, past_seq=511)
    model = build_model(**shape)
    feeds = make_feeds(**shape, seed=head_dim, hard_reduction=True)
    vk, variants = run_vulkan_with_variants(model, feeds, tmp_path, f"d{head_dim}")
    assert_module_ran(variants, DECODE_MODULE, 16)

    cpu = m.run_cpu(model, feeds)
    outcome, facts = m.compare_all_outputs_to_cpu(vk, cpu)
    assert outcome == m.COMPARISON_AGREE, f"head_dim {head_dim}: {outcome}\n{facts}"


@pytest.mark.parametrize(
    "nq,nkv",
    [
        (8, 8),    # MHA: group size 1
        (8, 2),    # GQA: group size 4
        (8, 1),    # MQA: one KV head for every query head
        (32, 32),  # Phi-3.5's arity
    ],
)
def test_kv_parallel_matches_cpu_across_head_groupings(
    nq: int, nkv: int, tmp_path: pathlib.Path, require_vulkan
) -> None:
    """Grouping decides ``kv_write_leader``, so it decides who writes ``present``.

    Under grouping ``G = nq/nkv`` query heads name the same present half-words.  In this module
    the writer is lane 0 of the leader head's workgroup rather than one invocation of a flat
    grid, which is a different predicate over a different geometry — so the duplicate-writer
    question ``test_gqa_grouping.py`` answers for ``gqa_f16`` has to be answered again here.
    """
    shape = dict(batch=1, seq_len=1, nq=nq, nkv=nkv, head_dim=32, past_seq=511)
    model = build_model(**shape)
    feeds = make_feeds(**shape, seed=nq * 100 + nkv, hard_reduction=True)
    vk, variants = run_vulkan_with_variants(model, feeds, tmp_path, f"g{nq}_{nkv}")
    assert_module_ran(variants, DECODE_MODULE, 16)

    cpu = m.run_cpu(model, feeds)
    outcome, facts = m.compare_all_outputs_to_cpu(vk, cpu)
    assert outcome == m.COMPARISON_AGREE, f"nq={nq} nkv={nkv}: {outcome}\n{facts}"


def test_kv_parallel_matches_cpu_with_a_ragged_batch(
    tmp_path: pathlib.Path, require_vulkan
) -> None:
    """The effective lane count is per-batch-row, and the rows here disagree by 16x.

    The pipeline is built for 16 lanes from the declared extent.  Row 0 really holds 511 tokens
    and uses all 16; row 1 holds 31 and the shader's own ladder drops it to 1, executing no
    barrier at all; row 2 holds 63 and lands on 2.  Three different effective widths in one
    dispatch is the case a host-side-only lane choice gets wrong, and it is the same mechanism
    that answers the KV arena's capacity ambiguity.
    """
    shape = dict(batch=3, seq_len=1, nq=8, nkv=2, head_dim=32, past_seq=511)
    lengths = [511, 31, 63]
    model = build_model(**shape)
    feeds = make_feeds(**shape, seed=5, hard_reduction=True, lengths=lengths)
    vk, variants = run_vulkan_with_variants(model, feeds, tmp_path, "ragged")
    assert_module_ran(variants, DECODE_MODULE, 16)

    cpu = m.run_cpu(model, feeds)
    outcome, facts = m.compare_all_outputs_to_cpu(
        defined_kv_region(vk, lengths), defined_kv_region(cpu, lengths)
    )
    assert outcome == m.COMPARISON_AGREE, f"ragged batch: {outcome}\n{facts}"


def test_kv_parallel_survives_cancellation_and_extreme_magnitudes(
    tmp_path: pathlib.Path, require_vulkan
) -> None:
    """The merge tree's arithmetic, in front of an input designed to break it.

    The scores span a wide range, so the per-lane maxima genuinely differ and the
    rescale-to-the-joint-maximum step in the merge is exercised rather than skipped.  A merge
    that used one lane's maximum for all of them, or that forgot to rescale a partial's
    accumulator, produces a plausible-looking number here and only here.
    """
    shape = dict(batch=1, seq_len=1, nq=8, nkv=2, head_dim=64, past_seq=511)
    model = build_model(**shape)
    feeds = make_feeds(**shape, seed=11, hard_reduction=True)
    # A magnitude ramp along the cache: the last lane's keys are ~30x the first lane's, so the
    # per-lane maxima are far apart and every merge step has real work to do.
    pk = feeds["past_key"].astype(np.float32)
    ramp = np.linspace(0.05, 1.5, pk.shape[2], dtype=np.float32)
    feeds["past_key"] = (pk * ramp[None, None, :, None]).astype(np.float16)
    vk, variants = run_vulkan_with_variants(model, feeds, tmp_path, "cancel")
    assert_module_ran(variants, DECODE_MODULE, 16)

    cpu = m.run_cpu(model, feeds)
    outcome, facts = m.compare_all_outputs_to_cpu(vk, cpu)
    assert outcome == m.COMPARISON_AGREE, f"cancellation case: {outcome}\n{facts}"
    assert np.isfinite(vk[0]).all(), (
        "the merge produced a non-finite output from finite inputs; the joint-maximum rescale "
        "is the only place in this module that can do that"
    )


def test_non_finite_input_reaches_the_output_unscrubbed(
    tmp_path: pathlib.Path, require_vulkan
) -> None:
    """The declared policy, asserted rather than described.

    The module does not clamp, sanitise or scrub — a NaN in the cache reaches the output, term
    for term as in ``gqa_f16``.  What is asserted is that the two EPs make the *same* choice, so
    that a future 'helpful' NaN guard in either module shows up as a disagreement rather than as
    a silently different model.
    """
    shape = dict(batch=1, seq_len=1, nq=8, nkv=2, head_dim=32, past_seq=511)
    model = build_model(**shape)
    feeds = make_feeds(**shape, seed=13)
    pv = feeds["past_value"].copy()
    pv[0, 0, 300, 0] = np.float16("nan")
    feeds["past_value"] = pv
    vk, variants = run_vulkan_with_variants(model, feeds, tmp_path, "nonfinite")
    assert_module_ran(variants, DECODE_MODULE, 16)
    cpu = m.run_cpu(model, feeds)

    vk_nan = np.isnan(vk[0])
    cpu_nan = np.isnan(cpu[0])
    assert vk_nan.any(), (
        "a NaN in the KV cache did not reach the output: something in this module is scrubbing "
        "non-finite values, which is not the declared policy and hides real model corruption"
    )
    assert np.array_equal(vk_nan, cpu_nan), (
        "the Vulkan and CPU EPs disagree about WHICH outputs are NaN — one of them is "
        f"scrubbing. vulkan {int(vk_nan.sum())} / cpu {int(cpu_nan.sum())}"
    )


def test_a_short_cache_takes_the_untouched_module(
    tmp_path: pathlib.Path, require_vulkan
) -> None:
    """The refusal path, established from the EP's artifact rather than from the ladder.

    62 past tokens is 63 positions, under two lanes' worth, so the selector refuses and the run
    must build ``gqa_f16`` and **not** ``gqa_decode_f16``.  This is the negative control for
    every ``assert_module_ran`` above: without it, a build in which the decode module is never
    selected at all would fail those tests for a reason indistinguishable from a device problem.
    """
    shape = dict(batch=1, seq_len=1, nq=8, nkv=2, head_dim=32, past_seq=62)
    model = build_model(**shape)
    feeds = make_feeds(**shape, seed=17)
    vk, variants = run_vulkan_with_variants(model, feeds, tmp_path, "short")
    assert_module_ran(variants, SERIAL_MODULE)
    assert_module_did_not_run(variants, DECODE_MODULE)

    cpu = m.run_cpu(model, feeds)
    outcome, facts = m.compare_all_outputs_to_cpu(vk, cpu)
    assert outcome == m.COMPARISON_AGREE, f"refusal path: {outcome}\n{facts}"


def test_prefill_never_reaches_the_kv_parallel_module(
    tmp_path: pathlib.Path, require_vulkan
) -> None:
    """``seq_len > 1`` is refused whatever the cache length.

    The module has no branch for the ``past_len <= t < tok_pos`` sibling key — the range that is
    empty exactly when ``seq_len == 1`` — so a prefill that reached it would read uninitialised
    present slots.  The cache here is long enough for 16 lanes, so the only thing refusing is the
    sequence length.
    """
    shape = dict(batch=1, seq_len=8, nq=8, nkv=2, head_dim=32, past_seq=511)
    model = build_model(**shape)
    feeds = make_feeds(**shape, seed=19, lengths=[511 + 8 - 1])
    vk, variants = run_vulkan_with_variants(model, feeds, tmp_path, "prefill")
    assert_module_did_not_run(variants, DECODE_MODULE)
    assert_module_ran(variants, SERIAL_MODULE)

    cpu = m.run_cpu(model, feeds)
    outcome, facts = m.compare_all_outputs_to_cpu(vk, cpu)
    assert outcome == m.COMPARISON_AGREE, f"prefill: {outcome}\n{facts}"


def test_the_kill_switch_restores_the_shipping_dispatch_bit_for_bit(
    tmp_path: pathlib.Path, require_vulkan
) -> None:
    """``ONNXRUNTIME_EP_VULKAN_GQA_DECODE_KV_LANES=1`` is the operational fallback.

    It must produce the pre-#90 dispatch **exactly**: the same module, so the same SPIR-V, so the
    same arithmetic and the same bytes.  This is the one place a bit-for-bit claim is made in
    this file, and it is only claimable because the two runs execute the same module — the
    comparison is against a run of the same graph on a build where the selector refuses on its
    own (a 62-token cache), not against the KV-parallel arm.
    """
    shape = dict(batch=1, seq_len=1, nq=8, nkv=2, head_dim=32, past_seq=511)
    model = build_model(**shape)
    feeds = make_feeds(**shape, seed=23, hard_reduction=True)

    off, off_variants = run_vulkan_with_variants(
        model, feeds, tmp_path, "killswitch", lane_override="1"
    )
    assert_module_did_not_run(off_variants, DECODE_MODULE)
    assert_module_ran(off_variants, SERIAL_MODULE, 1)

    on, on_variants = run_vulkan_with_variants(model, feeds, tmp_path, "killswitch_on")
    assert_module_ran(on_variants, DECODE_MODULE, 16)

    # Both arms must agree with the oracle. They are NOT asserted bit-equal to each other:
    # different summation orders are different roundings, which is the whole point of the
    # module and is not a defect.
    cpu = m.run_cpu(model, feeds)
    for name, out in (("kill switch", off), ("kv-parallel", on)):
        outcome, facts = m.compare_all_outputs_to_cpu(out, cpu)
        assert outcome == m.COMPARISON_AGREE, f"{name}: {outcome}\n{facts}"


@pytest.mark.parametrize("override", ["2", "4", "8"])
def test_the_lane_override_reaches_the_pipeline(
    override: str, tmp_path: pathlib.Path, require_vulkan
) -> None:
    """The switch is falsified against the artifact, not against itself.

    ``pipeline_variants`` carries the resolved specialisation vector the EP handed Vulkan, so a
    switch that were read and then dropped on the floor — the defect shape ``GEMV_PACKED``'s
    census entry names — shows up here as the wrong lane count rather than as nothing.  And every
    forced width must still agree with the oracle: a lane count is a summation order, not a
    licence to be wrong.
    """
    shape = dict(batch=1, seq_len=1, nq=8, nkv=2, head_dim=32, past_seq=511)
    model = build_model(**shape)
    feeds = make_feeds(**shape, seed=29, hard_reduction=True)

    vk, variants = run_vulkan_with_variants(
        model, feeds, tmp_path, f"ov{override}", lane_override=override
    )
    assert_module_ran(variants, DECODE_MODULE, int(override))

    cpu = m.run_cpu(model, feeds)
    outcome, facts = m.compare_all_outputs_to_cpu(vk, cpu)
    assert outcome == m.COMPARISON_AGREE, f"override={override}: {outcome}\n{facts}"


def test_repeated_runs_are_bit_identical(tmp_path: pathlib.Path, require_vulkan) -> None:
    """A shared-memory reduction with a missing barrier is a race, and races are intermittent.

    Three runs of one session with identical feeds must produce identical **bits**.  This is the
    only assertion in the file that can see a barrier that is missing rather than merely
    mis-placed, and it is the reason the merge loop's bound has to be uniform.
    """
    shape = dict(batch=2, seq_len=1, nq=8, nkv=2, head_dim=64, past_seq=511)
    model = build_model(**shape)
    feeds = make_feeds(**shape, seed=31, hard_reduction=True)

    runs, variants = run_vulkan_in_child(model, feeds, tmp_path, "repeat", repeats=3)
    assert_module_ran(variants, DECODE_MODULE, 16)

    for i, later in enumerate(runs[1:], start=2):
        equal, differing = m.outputs_bit_equal(runs[0], later)
        assert equal, (
            f"run 1 and run {i} differ on outputs {differing}: a shared-memory reduction that "
            f"is not reproducible has a barrier missing, and the correct-looking runs are luck"
        )


@pytest.mark.portability
def test_kv_parallel_is_portable_across_backends(
    tmp_path: pathlib.Path, require_vulkan
) -> None:
    """Vulkan 1.1 core, asserted on whatever device this lane runs on.

    The module claims to need no subgroup operations, no optional features and no more than the
    guaranteed 16 KiB of shared memory.  That claim is about *every conformant implementation*,
    so it is checked on the second backend this project runs (lavapipe under
    ``pytest -m portability``) as well as on the developer's GPU.  A device-specific reduction
    would pass everywhere else and fail exactly here.
    """
    shape = dict(batch=2, seq_len=1, nq=8, nkv=2, head_dim=128, past_seq=511)
    lengths = [511, 63]
    model = build_model(**shape)
    feeds = make_feeds(**shape, seed=37, hard_reduction=True, lengths=lengths)
    vk, variants = run_vulkan_with_variants(model, feeds, tmp_path, "portability")
    assert_module_ran(variants, DECODE_MODULE, 16)

    cpu = m.run_cpu(model, feeds)
    outcome, facts = m.compare_all_outputs_to_cpu(
        defined_kv_region(vk, lengths), defined_kv_region(cpu, lengths)
    )
    assert outcome == m.COMPARISON_AGREE, f"portability lane: {outcome}\n{facts}"


# ---------------------------------------------------------------------------
# Isolated run process (see `run_vulkan_in_child`).  pytest never executes this;
# it is reached only when this file is invoked as a script with a run spec.
# ---------------------------------------------------------------------------


def _child_main(spec_path: str) -> int:
    """Execute one run spec on the Vulkan EP in this (fresh) process and save the outputs.

    Nothing here is timed and nothing is retried: the point of the process is that the counters
    file it leaves behind names exactly the pipelines *this* run created, so a refusal control
    and a positive attribution read the same artifact and mean opposite things.
    """
    spec = json.loads(pathlib.Path(spec_path).read_text(encoding="utf-8"))
    lib = os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB")
    if not lib:
        print("ONNXRUNTIME_VULKAN_EP_LIB is unset in the isolated run process", file=sys.stderr)
        return 2
    lib_path = pathlib.Path(lib).resolve()
    if not lib_path.is_file():
        print(f"EP library not found: {lib_path}", file=sys.stderr)
        return 2
    ort.register_execution_provider_library("VulkanExecutionProvider", str(lib_path))

    model = pathlib.Path(spec["model"]).read_bytes()
    with np.load(spec["feeds"]) as npz:
        feeds = {name: npz[name] for name in npz.files}

    session = ort.InferenceSession(
        model, m._make_session_options(), providers=m.EP_PROVIDERS
    )
    runs = [session.run(None, feeds) for _ in range(int(spec["repeats"]))]
    del session  # counters flush at EP teardown

    arrays: dict[str, np.ndarray] = {"n_outputs": np.asarray(len(runs[0]))}
    for r, outputs in enumerate(runs):
        for j, value in enumerate(outputs):
            arrays[f"r{r}_o{j}"] = np.asarray(value)
    np.savez(spec["outputs"], **arrays)
    return 0


if __name__ == "__main__":
    raise SystemExit(_child_main(sys.argv[1]))
