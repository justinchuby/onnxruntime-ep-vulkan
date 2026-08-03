"""Locks for the weight re-read instrument and for the provenance rule it came out of.

WHY THIS FILE EXISTS
====================
`bench/results/probe_island_bytes.py` published a `weight_reread_amplification` block whose
five values were all literals. It argued -- correctly -- that the amplification of 1.0 was not
an identity, because two factors in it are contingent: loads per blob, and blobs per
workgroup. But nothing computed either. The number `1.000000` would have printed unchanged if
the kernel had been rewritten overnight to read every blob eight times.

So the locks here are not on the value. They are on the instrument's ability to *not* say 1:

* `test_tail_tile_is_a_positive_state` and `test_a_rereading_kernel_is_detected` are the
  positive controls. A detector never seen in its positive state has no demonstrated positive
  state, and 1.000000 from such a detector is not a reading.
* `test_interpreter_reproduces_the_gemv` is the negative control for the interpreter itself.
  An address trace from an interpreter that computes the wrong answer is a trace of the wrong
  program. This one reproduces a real quantised GEMV bit-exactly.
* `test_the_five_literals_are_gone` is the regression lock on the original defect.

Everything here is counts and bytes. Nothing here reads a clock, so nothing here is affected
by contention on the box, and no reading depends on a hardware counter -- which also puts it
outside the `a52024f`..`4d47362` window Mouse found the ABI-insertion defect in.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parent
RESULTS = BENCH / "results"

_SYS_PATH_BEFORE = list(sys.path)
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(RESULTS))
try:
    import spirv_simt  # noqa: E402
    from spirv_simt import Dispatch, InstrumentError, SpirvModule  # noqa: E402

    import probe_island_bytes as island  # noqa: E402
    import probe_weight_reread as wr  # noqa: E402
finally:
    sys.path[:] = _SYS_PATH_BEFORE

RECORD = RESULTS / "weight_reread_phi35.json"
ISLAND_SOURCE = RESULTS / "probe_island_bytes.py"


@pytest.fixture(scope="module")
def module():
    """The compiled module the ledger says a run dispatched, or skip.

    Skipped rather than failed when the build tree is absent: a checkout with no `cargo build`
    behind it cannot produce this, and a test that fails for a missing artifact teaches people
    to ignore it.
    """
    try:
        path, blob, digest = wr.locate_module(wr.SHADER_STEM)
    except InstrumentError as e:
        pytest.skip(f"compiled module not locatable: {e}")
    return SpirvModule(blob), digest


def _gemv_case(K, N, block, bits, wg, cols, packed, seed=7):
    """Build one dispatch of the real kernel over random weights, plus a numpy reference."""
    bpc = K // block
    rng = np.random.default_rng(seed)
    a = rng.integers(-4, 5, size=K).astype(np.float16)
    w = rng.integers(0, 1 << bits, size=(N, bpc, block)).astype(np.uint8)
    scales = (rng.integers(1, 4, size=(N, bpc)) / 8.0).astype(np.float16)

    if bits == 4:
        blob = np.zeros((N, bpc, block // 2), dtype=np.uint8)
        blob |= w[:, :, 0::2]
        blob |= w[:, :, 1::2] << 4
    else:
        blob = w
    inb = np.ascontiguousarray(blob.ravel()).view(np.uint32).copy()

    ina = np.ascontiguousarray(a).view(np.uint32).copy()
    ins = np.ascontiguousarray(scales.ravel()).view(np.uint32).copy()
    inz = np.zeros(max(1, N * ((bpc * bits + 7) // 8) // 4 + 1), dtype=np.uint32)
    out = np.zeros(max(1, -(-N // 2)), dtype=np.uint32)

    d = Dispatch(
        groups=(-(-N // cols), 1, 1),
        local_size=(wg, 1, 1),
        spec={0: wg, 1: bits, 2: block, 3: 0, 4: cols, 5: packed},
        push_constants=[1, K, N, bpc],
        buffers={0: ina, 1: inb, 2: ins, 3: inz, 4: out},
    )
    zp = 1 << (bits - 1)
    ref = np.zeros(N, dtype=np.float32)
    for n in range(N):
        for b in range(bpc):
            ref[n] += float(scales[n, b]) * float(np.dot(
                a[b * block:(b + 1) * block].astype(np.float32),
                w[n, b].astype(np.float32) - zp))
    return d, out, ref, inb


# -- the interpreter itself --------------------------------------------------------------------


def test_interpreter_reproduces_the_gemv(module):
    """The trace is only evidence if the program that produced it computed the right answer.

    This caught the interpreter's own worst bug: masked-off lanes were parked on scatter index
    0, where numpy's last-write-wins silently ate a live store, producing a GEMV that was wrong
    in exactly one column per tile and plausible everywhere else -- the same defect shape this
    whole file is about.
    """
    mod, _ = module
    d, out, ref, _ = _gemv_case(K=64, N=8, block=32, bits=4, wg=32, cols=4, packed=1)
    mod.run(d, trace_binding=1)
    got = out.view(np.float16).astype(np.float32)
    assert np.abs(got - ref).max() == 0.0, f"got {got} ref {ref}"


# -- positive controls: the instrument must be able to say something other than 1 --------------


def test_tail_tile_is_a_positive_state(module):
    """N=130 with a 16-wide tile: the surplus columns redirect onto col0 and re-read it.

    This is the path the original docstring named as the one thing that would break the
    argument, and then never exercised. It is reachable in the shipped kernel; it is only
    unreachable for Phi-3.5 because all five N happen to divide by 16.
    """
    mod, _ = module
    d, _, _, inb = _gemv_case(K=64, N=130, block=32, bits=4, wg=32, cols=16, packed=1)
    tr = mod.run(d, trace_binding=1)
    amp = tr.named_bytes / inb.nbytes
    assert amp > 1.0, f"tail tile did not amplify: {amp}"
    assert tr.max_reads_per_word > 1


def test_shipped_shapes_do_not_amplify(module):
    """The same instrument, on a shape whose N divides the tile: exactly one load per byte."""
    mod, _ = module
    d, _, _, inb = _gemv_case(K=64, N=128, block=32, bits=4, wg=32, cols=16, packed=1)
    tr = mod.run(d, trace_binding=1)
    assert tr.named_bytes == inb.nbytes
    assert tr.max_reads_per_word == 1
    assert tr.words_read_by_more_than_one_workgroup == 0
    assert tr.touched_words == tr.words


def test_unpacked_path_changes_the_width_not_the_bytes(module):
    """The general path reads the same bytes four loads at a time instead of one.

    Worth a lock of its own: an amplification-only detector is blind to this, because the
    amplification is 1.0 either way. The load *width* had to be measured separately, and it is
    read off the SPIR-V result type rather than assumed from the spec constant.
    """
    mod, _ = module
    shape = dict(K=64, N=128, block=32, bits=4, wg=32, cols=16)
    dp, _, _, inb = _gemv_case(packed=1, **shape)
    du, _, _, _ = _gemv_case(packed=0, **shape)
    tp = mod.run(dp, trace_binding=1)
    tu = mod.run(du, trace_binding=1)
    assert tp.load_widths_bytes == [16]
    assert tu.load_widths_bytes == [4]
    assert tp.named_bytes == tu.named_bytes == inb.nbytes
    assert tu.load_instructions == 4 * tp.load_instructions


# -- the binding between the reading and a specific compiled kernel -----------------------------


def test_the_walked_module_is_the_one_the_ledger_recorded(module):
    """`shader_digest` reimplements `registry.rs::shader_digest_for`; the ledger is the witness.

    Without this the walk is of *a* module. The whole point of the exercise is that it is a
    walk of the module a run dispatched.
    """
    _, digest = module
    assert digest == wr.ledger_digest_for(wr.SHADER_STEM)


def test_dispatch_geometry_mirrors_the_host():
    """`gemv_workgroup`/`gemv_cols` are mirrors of `rust/src/ops/quant.rs`.

    If they drift, the walk is of a grid the host never launches. All five Phi-3.5 shapes take
    cols=16, which is exactly why the tail-tile redirect is unreachable in production -- a
    claim that is now a consequence of these two functions rather than of a sentence.
    """
    for K, N in ((3072, 9216), (3072, 3072), (3072, 8192), (3072, 32064), (8192, 3072)):
        bpc = K // 32
        wg = wr.gemv_workgroup(bpc)
        cols = wr.gemv_cols(N, wg)
        assert cols == 16, (K, N, wg, cols)
        assert N % cols == 0, (K, N, cols)
    assert wr.gemv_workgroup(96) == 32
    assert wr.gemv_workgroup(256) == 128
    assert wr.gemv_packed(4, 32) is True
    assert wr.gemv_packed(4, 16) is False


# -- the defect this file is a lock against ----------------------------------------------------

FIVE_LITERALS = ("116_324_352", "116324352", "1_861_189_632", "1861189632", "2_093_838_336")


def test_the_five_literals_are_gone():
    """`probe_island_bytes.py` must not restate any measured quantity as a constant.

    The literals are allowed to appear in prose -- the docstring quotes the old block to
    explain what went wrong -- but not in code. So this checks the code lines only.
    """
    text = ISLAND_SOURCE.read_text(encoding="utf-8")
    doc_end = text.index('"""', text.index('"""') + 3) + 3
    code = text[doc_end:]
    code = "\n".join(line for line in code.splitlines()
                     if not line.lstrip().startswith("#"))
    for lit in FIVE_LITERALS:
        assert lit not in code, f"`{lit}` is a measurement restated as a literal"


def test_every_published_quantity_is_classified():
    """A file that mixes specifications and measurements silently is how this one got here."""
    classes = {p["class"] for p in island.PROVENANCE.values()}
    assert classes <= {"SPECIFICATION", "MEASUREMENT", "MODEL"}
    assert "SPECIFICATION" in classes and "MEASUREMENT" in classes and "MODEL" in classes
    for name, p in island.PROVENANCE.items():
        assert p["source"].strip(), name
        assert p["wrong_when"].strip(), name


def test_peak_bandwidth_is_a_specification_and_the_part_is_not():
    """A datasheet number is a fact about a named part. The name is a separate claim."""
    assert island.PEAK_BYTES_PER_S == island.BUS_BITS / 8 * island.MEM_GBPS * 1e9
    assert island.PROVENANCE["PEAK_BYTES_PER_S"]["class"] == "SPECIFICATION"
    assert island.PROVENANCE["SPEC_PART"]["class"] == "MEASUREMENT"
    if not island.RUN_RECORD.exists():
        pytest.skip("no run record")
    dev = island.device_from_run()
    assert dev["observed_from_trace"] == island.SPEC_PART
    assert dev["ok"]


def test_island_refuses_an_absent_or_uncontrolled_record(tmp_path):
    """Three refusals the literal block could not make."""
    assert island.weight_reread(tmp_path / "nope.json")["verdict"] == "UNWITNESSED"

    doc = {"positive_controls": {"witnessed": False, "controls": []}}
    p = tmp_path / "unwitnessed.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    assert island.weight_reread(p)["verdict"] == "UNWITNESSED"

    doc = {"positive_controls": {"witnessed": True, "controls": []},
           "subject": {"digest_matches_ledger": False}}
    p = tmp_path / "stale.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    assert island.weight_reread(p)["verdict"] == "UNWITNESSED"


# -- the published record ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def record():
    if not RECORD.exists():
        pytest.skip(f"{RECORD.name} not produced yet")
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_record_is_witnessed_and_bound(record):
    assert record["positive_controls"]["witnessed"] is True
    assert record["subject"]["digest_matches_ledger"] is True
    assert record["subject"]["shader_digest"] == wr.ledger_digest_for(wr.SHADER_STEM)
    fired = [c for c in record["positive_controls"]["controls"] if c.get("positive")]
    assert len(fired) == len(record["positive_controls"]["controls"])
    amps = [c["amplification"] for c in record["positive_controls"]["controls"]
            if c.get("amplification") is not None]
    assert any(a != 1.0 for a in amps), "no control ever produced an amplification other than 1"


def test_denominator_comes_from_the_graph(record):
    """`int4_weight_bytes_from_graph` is the sum of initializer sizes, not a restated constant.

    The blob model and the initializers agreeing is the check that makes the amplification a
    result rather than an identity: `blobs x blob_bytes` is a definition, but that the graph
    stores exactly that many bytes is a fact about the graph.
    """
    assert record["denominator"]["matmulnbits_nodes"] > 0
    assert record["denominator"]["int4_weight_bytes_from_graph"] > 0
    assert all(s["blob_model_agrees_with_initializers"] for s in record["by_shape"])


def test_the_amplification_is_measured_not_asserted(record):
    """Whatever it is, every factor behind it came out of the walk."""
    m = record["measured"]
    named = sum(s["named_bytes"] * s["nodes_with_this_shape"] for s in record["by_shape"])
    loads = sum(s["load_instructions"] * s["nodes_with_this_shape"] for s in record["by_shape"])
    assert named == m["named_bytes_per_inference"]
    assert loads == m["inb_load_instructions_per_inference"]
    assert m["amplification"] == pytest.approx(
        named / record["denominator"]["int4_weight_bytes_from_graph"])
    assert all(s["coverage"] == 1.0 for s in record["by_shape"])


def test_no_clock_anywhere_in_the_instrument():
    """Counts and bytes only: the box is permanently contended, so a time here would be noise.

    Also the reason nothing in this instrument falls inside the suspect counter window: it
    reads no counters at all.
    """
    for src in (BENCH / "spirv_simt.py", RESULTS / "probe_weight_reread.py"):
        text = src.read_text(encoding="utf-8")
        code = "\n".join(line for line in text.splitlines()
                         if not line.lstrip().startswith("#"))
        for banned in ("time.perf_counter", "time.time(", "timeit", "datetime.now"):
            assert banned not in code, f"{src.name} reads a clock: {banned}"


def test_import_has_no_side_effects():
    """Importing a probe must not walk a module, load a model, or write a record."""
    assert not hasattr(spirv_simt, "_ran_on_import")
    assert callable(wr.main) and callable(island.main)
