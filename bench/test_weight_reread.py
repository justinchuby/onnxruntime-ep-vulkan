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
    import probe_real_matmulnbits_rows as real_rows  # noqa: E402
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


def _gemv_case(K, N, block, bits, wg, cols, packed, seed=7, m_total=1, rows=1):
    """Build one dispatch of the real kernel over random weights, plus a numpy reference."""
    bpc = K // block
    rng = np.random.default_rng(seed)
    a = rng.integers(-4, 5, size=m_total * K).astype(np.float16)
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
    out = np.zeros(max(1, -(-(m_total * N) // 2)), dtype=np.uint32)

    d = Dispatch(
        groups=(-(-N // cols), min(-(-m_total // rows), wr.GEMV_MAX_GROUPS_Y), 1),
        local_size=(wg, 1, 1),
        spec={0: wg, 1: bits, 2: block, 3: 0, 4: cols, 5: packed, 6: rows},
        push_constants=[m_total, K, N, bpc],
        buffers={0: ina, 1: inb, 2: ins, 3: inz, 4: out},
    )
    zp = 1 << (bits - 1)
    ref = np.zeros(m_total * N, dtype=np.float32)
    for r in range(m_total):
        arow = a[r * K:(r + 1) * K]
        for n in range(N):
            for b in range(bpc):
                ref[r * N + n] += float(scales[n, b]) * float(np.dot(
                    arow[b * block:(b + 1) * block].astype(np.float32),
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


# -- the row tile (issue #7) -------------------------------------------------------------------


@pytest.mark.parametrize("m_total,rows,cols", [(2, 2, 16), (4, 2, 16), (5, 2, 16),
                                               (4, 4, 8), (7, 4, 8), (3, 4, 8)])
def test_the_row_tile_computes_the_same_gemm(module, m_total, rows, cols):
    """A tiled prefill must produce the same numbers as the untiled one, tail rows included.

    This is the first thing to check and the easiest to skip: an amplification that fell is
    worthless if the kernel that produced it stopped being right. `m_total = 5, rows = 2` and
    `m_total = 3/7, rows = 4` leave a partial tile, which is the case a tile gets wrong if it
    gets anything wrong -- the shader redirects the out-of-range rows onto row 0 and must then
    refuse to store them.
    """
    mod, _ = module
    n = 8 if cols <= 8 else 16
    d, out, ref, _ = _gemv_case(K=64, N=n, block=32, bits=4, wg=32, cols=cols, packed=1,
                                m_total=m_total, rows=rows)
    mod.run(d, trace_binding=1)
    got = out.view(np.float16).astype(np.float32)[:m_total * n]
    assert np.abs(got - ref).max() == 0.0, f"got {got} ref {ref}"


def test_the_row_tile_divides_the_weight_reread_by_the_tile_height(module):
    """The defect issue #7 named, and its repair, measured on one module in one test.

    Untiled, every one of the `M` y-workgroups names the whole weight strip: amplification is
    `M` exactly. Tiled, it is `ceil(M / QB_ROWS)` exactly -- not "about", not "up to". The
    `ceil` is the point of the `M = 5` and `M = 7` rows: a tail tile still names the strip once,
    so the saving is a floor division and the test says which.

    The `(cols, rows)` pairs are the legal ones -- `cols * rows <= QB_MAX_TILE`. That is not a
    convenience: `test_an_illegal_tile_computes_nothing_rather_than_overrunning` below is the
    control showing what the module does when the bound is violated.
    """
    mod, _ = module
    base = dict(K=64, N=128, block=32, bits=4, wg=32, packed=1)
    once = _gemv_case(**base, cols=16)[3].nbytes
    for m_total in (1, 2, 4, 5, 7, 8):
        untiled = mod.run(_gemv_case(**base, cols=16, m_total=m_total, rows=1)[0],
                          trace_binding=1)
        assert untiled.named_bytes == m_total * once, m_total
        assert untiled.max_reads_per_word == m_total, m_total
        for cols, rows in ((16, 2), (8, 4)):
            tr = mod.run(_gemv_case(**base, cols=cols, m_total=m_total, rows=rows)[0],
                         trace_binding=1)
            expected = -(-m_total // rows)
            assert tr.named_bytes == expected * once, (m_total, cols, rows)
            assert tr.max_reads_per_word == expected, (m_total, cols, rows)
            assert tr.touched_words == tr.words, "the tile must still cover the whole tensor"


def test_an_illegal_tile_computes_nothing_rather_than_overrunning(module):
    """`QB_ROWS * QB_COLS > QB_MAX_TILE` must be refused by the module, not merely unreached.

    `acc`/`bacc` are `QB_MAX_TILE` long and addressed `r * QB_COLS + c`, so `rows = 4` with
    `cols = 16` would write 32 slots past the end. `ops::quant::gemv_tile` cannot return that
    pair and `matmul_nbits_gemv` refuses it outright -- but both are properties of today's host,
    and the overrun would be a property of the module. The shader carries the bound as a folded
    specialisation-constant guard, so the illegal pipeline reads nothing and stores nothing.

    Without this test the guard is invisible: every legal pipeline folds it away, so nothing
    else in the suite can tell it from a comment.
    """
    mod, _ = module
    d, out, _, _ = _gemv_case(K=64, N=128, block=32, bits=4, wg=32, cols=16, packed=1,
                              m_total=4, rows=4)
    tr = mod.run(d, trace_binding=1)
    assert tr.load_instructions == 0, "an illegal tile must not read the weight stream"
    assert tr.named_bytes == 0
    assert not out.any(), "an illegal tile must not store"


def test_the_decode_arm_is_untouched_by_the_row_tile(module):
    """`QB_ROWS = 1` must be the previous kernel, byte for byte in what it reads.

    `q_gemv.comp` selects the decode path with a *specialisation-constant* branch holding the
    previous text, rather than making decode a degenerate case of the tiled loop. That choice is
    only worth anything if it is checked: the decode arm must still issue 16-byte packed loads,
    name each byte once, and hand back the same answers. A tiled arm that reads 4 bytes at a
    time is a legitimate trade at `M > 1`; silently taking it at `M = 1` would not be.
    """
    mod, _ = module
    shape = dict(K=64, N=128, block=32, bits=4, wg=32, cols=16, packed=1)
    d, _, _, inb = _gemv_case(**shape, m_total=1, rows=1)
    tr = mod.run(d, trace_binding=1)
    assert tr.load_widths_bytes == [16], "decode must keep the uvec4 load"
    assert tr.named_bytes == inb.nbytes
    assert tr.max_reads_per_word == 1
    assert tr.words_read_by_more_than_one_workgroup == 0
    # And the host never selects anything else for a single row, whatever the shape.
    for K, N in ((3072, 9216), (3072, 3072), (3072, 8192), (3072, 32064), (8192, 3072)):
        wg = wr.gemv_workgroup(K // 32)
        assert wr.gemv_tile(1, N, K, 4, 2, wg) == (wr.gemv_cols(N, wg), 1), (K, N)


def test_the_y_extent_is_clamped_and_the_grid_stride_still_covers_every_tile(module):
    """`groupCountY` is clamped to the Vulkan floor, so the kernel must stride over the rest.

    `maxComputeWorkGroupCount[1]` is only guaranteed to be 65535, and a prefill of 200k rows is
    a legal ONNX graph. The host clamps and the shader loops; this checks the loop, by running a
    grid deliberately shorter in y than the number of row tiles and asserting that every weight
    byte is still named and that the arithmetic is still right.
    """
    mod, _ = module
    d, out, ref, inb = _gemv_case(K=64, N=8, block=32, bits=4, wg=32, cols=4, packed=1,
                                  m_total=6, rows=2)
    d.groups = (d.groups[0], 1, 1)          # 3 row tiles, 1 workgroup: two strides needed.
    mod.run(d, trace_binding=1)
    got = out.view(np.float16).astype(np.float32)[:6 * 8]
    assert np.abs(got - ref).max() == 0.0, f"got {got} ref {ref}"

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


def test_the_tile_picker_mirrors_the_host():
    """`gemv_tile`/`gemv_named_bytes` mirror `rust/src/ops/quant.rs`, and must not drift.

    The Rust unit tests in `ops::quant::tests` assert these same facts against the Rust
    implementation. Asserting them here against the Python mirror is what makes a drift a
    failure in *both* files rather than a walk of a grid the host never launches.
    """
    assert wr.GEMV_MAX_ROWS == 4
    assert wr.GEMV_MAX_TILE == 32
    assert wr.GEMV_MAX_GROUPS_Y == 65535
    # Phi-3.5's shapes: the decode tile at M=1, the 16x2 tile at every prefill width.
    for K, N in ((3072, 9216), (3072, 3072), (3072, 8192), (3072, 32064), (8192, 3072)):
        wg = wr.gemv_workgroup(K // 32)
        assert wr.gemv_tile(1, N, K, 4, 2, wg) == (wr.gemv_cols(N, wg), 1), (K, N)
        for m in (2, 4, 16, 512):
            cols, rows = wr.gemv_tile(m, N, K, 4, 2, wg)
            assert rows == 2 and cols == 16, (K, N, m, cols, rows)
            assert cols * rows <= wr.GEMV_MAX_TILE
            assert wg * cols <= wr.GEMV_RED_WORDS
    # The byte model is the quantity the walk counts: weight amplification is ceil(M / rows).
    weight_once = 128 * 64 * 4 // 8
    for rows in (1, 2, 4):
        for m in (1, 2, 3, 4, 7, 64):
            assert wr.gemv_named_bytes(m, 128, 64, 4, 0, 16, rows) == (
                -(-m // rows) * weight_once), (m, rows)
    # A shape too narrow for a column tile still takes a row tile, because a row tile costs no
    # shared memory and no parallelism: `cols = 1` is legal at every `rows`.
    assert wr.gemv_tile(64, 64, 512, 4, 2, wr.gemv_workgroup(16)) == (1, 4)
    # Fail-closed is about the bounds, not about refusing to tile: over an awkward grid the
    # picker never emits a pair the module cannot address, and never a worse one than it started
    # with. `ops::quant::tests::every_selected_tile_respects_every_static_bound` is the twin.
    for m in (0, 1, 2, 3, 5, 17, 128, 4096):
        for n in (1, 2, 3, 7, 64, 100, 130, 512, 3072, 32064):
            for bpc in (1, 7, 96, 256, 4096):
                wg = wr.gemv_workgroup(bpc)
                cols, rows = wr.gemv_tile(m, n, bpc * 32, 4, 2, wg)
                assert cols * rows <= wr.GEMV_MAX_TILE, (m, n, bpc, cols, rows)
                assert 1 <= rows <= wr.GEMV_MAX_ROWS and 1 <= cols <= wr.GEMV_MAX_COLS
                assert wg * cols <= wr.GEMV_RED_WORDS, (m, n, bpc, cols, rows)
                assert cols == 1 or (n % cols == 0 and n // cols >= wr.GEMV_MIN_WORKGROUPS)
                assert (wr.gemv_named_bytes(m, n, bpc * 32, 4, 2, cols, rows)
                        <= wr.gemv_named_bytes(m, n, bpc * 32, 4, 2, wr.gemv_cols(n, wg), 1))


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


# -- model resolution: no hardcoded path, and a versioned cache is still found ------------------
#
# PR #53 review (Morpheus) caught `probe_weight_reread.py` claiming "nothing was hardcoded"
# while its own default MODEL was a literal Foundry cache path from a catalog revision that does
# not exist under the current Foundry Local layout. The fix replaced the literal with
# `foundry_discovery.resolve_model_path`, the same identity-based resolver
# `probe_real_matmulnbits_rows.py` already used. These are the regression locks on that fix:
# resolution must stay lazy (no import-time filesystem or subprocess access), must go through
# the shared resolver rather than a restated path, and must still find the model after Foundry's
# own cache layout is versioned out from under it -- the exact issue #11 pattern, replayed here.


def test_resolve_model_has_no_literal_foundry_path():
    """The regression this fix exists for: no hardcoded Foundry-cache-directory literal.

    Mirrors `ci/check_hardcoded_foundry_paths.py`'s own pattern so this lock does not silently
    drift from the CI screen it stands in for. The pattern itself is built from concatenated
    fragments below, deliberately, so this docstring is not itself a hit against that same
    screen: it is not in that screen's allowlist, and spelling the fragment out literally here
    would make this very file the next violation the screen exists to catch.
    """
    import re

    text = (RESULTS / "probe_weight_reread.py").read_text(encoding="utf-8")
    assert not re.search(r"\.foundry[\\/]+cache[\\/]+models", text, re.IGNORECASE)


def test_resolve_model_shares_the_identity_with_the_real_weights_probe():
    """Both probes measure the same Phi-3.5 Foundry variant; the identity must not fork."""
    assert wr._PHI35_SPEC == real_rows._PHI35_SPEC


def test_resolve_model_is_lazy_not_module_scope():
    """`resolve_model` must be a callable the caller invokes, never work done at import time --
    otherwise `test_import_has_no_side_effects` above would not catch a resolver call added at
    module scope."""
    import inspect

    assert inspect.isfunction(wr.resolve_model)
    assert "MODEL" not in dir(wr), "a module-level MODEL constant would be eager, not lazy"


def test_resolve_model_honours_an_explicit_override(tmp_path, monkeypatch):
    """`PHI35_MODEL`, if set, is used as-is and never silently replaced by the resolver."""
    pinned = tmp_path / "pinned.onnx"
    pinned.write_bytes(b"not a real onnx file, just a presence marker")
    monkeypatch.setenv("PHI35_MODEL", str(pinned))
    assert wr.resolve_model() == pinned


def test_resolve_model_override_missing_file_fails_loud(tmp_path, monkeypatch):
    """An explicit override that does not exist must raise, never fall through to the
    resolver -- silently substituting a different cached file would misattribute a run."""
    monkeypatch.setenv("PHI35_MODEL", str(tmp_path / "does_not_exist.onnx"))
    with pytest.raises(InstrumentError, match="PHI35_MODEL override does not exist"):
        wr.resolve_model()


def test_resolve_model_finds_a_versioned_cache_layout(tmp_path, monkeypatch):
    """PLANTED, VERSIONED-CACHE REGRESSION CONTROL (issue #11's pattern, replayed on this
    probe specifically). Foundry's own on-disk layout is versioned by its internal catalog
    revision and moves out from under any script that names a path directly -- exactly what
    happened to this probe's old hardcoded default. Plants the model under a
    version-suffixed variant directory (`<variant>-2/v2/...`, the same shape as the real
    `-cuda-gpu` -> `-cuda-gpu-2` move) at a redirected cache root, forces the filesystem
    fallback strategy (the CLI manifest is made to answer nothing, deterministically,
    regardless of whether a real `foundry` binary is on this machine's PATH), and asserts
    `resolve_model` still finds it by identity. A hardcoded-path regression would instead
    raise `model absent` here, because no literal path constant survives the version move.
    """
    import foundry_discovery as fd  # noqa: E402

    monkeypatch.delenv("PHI35_MODEL", raising=False)
    # Force the CLI-manifest strategy to miss, so resolution takes the filesystem fallback
    # deterministically -- independent of whether a real `foundry` executable answers on this
    # machine, which would otherwise make this control depend on the host's own install state.
    monkeypatch.setattr(fd, "_foundry_cache_list_json", lambda *a, **k: None)
    monkeypatch.setenv("ONNXRUNTIME_EP_VULKAN_FOUNDRY_CACHE", str(tmp_path))

    versioned_dir = tmp_path / "Microsoft" / (wr._PHI35_SPEC.variant_name + "-2") / "v2"
    versioned_dir.mkdir(parents=True)
    planted = versioned_dir / wr._PHI35_SPEC.onnx_filename
    planted.write_bytes(b"not a real onnx file, just a presence marker")

    resolved = wr.resolve_model()
    assert resolved == planted


def test_resolve_model_negative_control_wrong_cache_key_is_not_found(tmp_path, monkeypatch):
    """Negative control on the positive control above: a cache entry under a DIFFERENT
    variant identity must not be picked up as a substitute -- the resolver's identity match,
    not a directory glob, is what is being exercised."""
    import foundry_discovery as fd  # noqa: E402

    monkeypatch.delenv("PHI35_MODEL", raising=False)
    monkeypatch.setattr(fd, "_foundry_cache_list_json", lambda *a, **k: None)
    monkeypatch.setenv("ONNXRUNTIME_EP_VULKAN_FOUNDRY_CACHE", str(tmp_path))

    wrong_dir = tmp_path / "Microsoft" / "SomeOtherModel-cuda-gpu" / "v1"
    wrong_dir.mkdir(parents=True)
    (wrong_dir / wr._PHI35_SPEC.onnx_filename).write_bytes(b"wrong model entirely")

    with pytest.raises(fd.FoundryDiscoveryError, match="no cached Foundry variant"):
        wr.resolve_model()


# -- path hygiene (issue #81, defect 2) ---------------------------------------------------------
#
# Two independent defects lived here, and each of the tests below fails on exactly one of them:
#
#   1. `main()` wrote to `bench/results/weight_reread_phi35.json` unconditionally, so merely
#      RUNNING the probe rewrote a committed witness. An operator reading `git status` after a
#      re-derivation could not distinguish "I reproduced the recorded numbers" from "I replaced
#      them", which is the one distinction a witness exists to support.
#   2. That committed record published `onnx_file` as an absolute Foundry cache path -- one
#      operator's home directory, in a file the repository serves publicly.
#
# The fix is structural on both counts: the destination is an explicit `--out` whose default is
# gitignored, and every record is walked by `path_screen.screen_public_record` before the file
# is opened. These tests hold that structure, not the current field list.


def _path_screen():
    _before = list(sys.path)
    sys.path.insert(0, str(BENCH))
    try:
        import path_screen

        return path_screen
    finally:
        sys.path[:] = _before


def test_the_committed_witness_publishes_no_local_path():
    """The regression lock on defect 2 as it was actually committed.

    This asserts over the tracked file itself rather than over a record the test builds, because
    the leak was in the artifact, not in the code path a test would exercise. Deleting the
    redaction and restoring the absolute path turns this red.
    """
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    kept, why = _path_screen().screen_public_record(record)
    assert kept is not None, f"committed witness publishes a local path: {why}"


def test_the_committed_witness_still_identifies_the_exact_model():
    """Redaction is not permitted to cost identity.

    Dropping the resolved path is only defensible because the record still pins the *content*
    of the measured artifact. If a future edit removes the digest too, the record would name a
    filename anyone can produce, and this goes red.
    """
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert record["onnx_sha256"] == (
        "3dbdd4b5f4d487da609fdacb9fd35b113cac706363a72795508524a4704dac3f"
    )
    assert record["onnx_file"] == wr._PHI35_SPEC.onnx_filename
    assert record["onnx_file_is_a_name_not_a_path"] is True


def test_result_identity_records_a_name_not_a_path(tmp_path, monkeypatch):
    """The code-path half of defect 2: `_result_identity` must not stamp its argument's path.

    Driven through a planted file under an absolute tmp_path, so the value it would have leaked
    is genuinely absolute on both Windows and POSIX. Restoring `str(model)` turns this red.
    """
    planted = tmp_path / "deeply" / "nested" / wr._PHI35_SPEC.onnx_filename
    planted.parent.mkdir(parents=True)
    planted.write_bytes(b"not a real model, only its identity is under test")
    monkeypatch.setenv("PHI35_MODEL", str(planted))

    identity = wr._result_identity(planted)

    assert identity["onnx_file"] == wr._PHI35_SPEC.onnx_filename
    assert str(tmp_path) not in json.dumps(identity)
    kept, why = _path_screen().screen_public_record(identity)
    assert kept is not None, why


def test_publish_refuses_a_planted_absolute_path_and_writes_nothing(tmp_path):
    """`publish` refuses rather than scrubs, and refuses BEFORE creating the file.

    The plant sits at depth, under a key the probe does not have today, because the screen this
    delegates to is total over JSON-shaped values -- the guarantee is not a field allowlist.
    "Writes nothing" is asserted separately from the raise: a writer that raises after opening
    its output leaves a truncated artifact behind, which is worse than either outcome.
    """
    path_screen = _path_screen()
    out = tmp_path / "sub" / "record.json"
    leaky = {"by_shape": [{"note": {"where": "C:\\Users\\someone\\models\\x.onnx"}}]}

    with pytest.raises(path_screen.PrivatePathLeak):
        wr.publish(leaky, out)

    assert not out.exists()
    assert not out.parent.exists()


def test_publish_writes_a_screened_record(tmp_path):
    """The positive polarity: a clean record round-trips exactly, and parents are created."""
    out = tmp_path / "made" / "up" / "record.json"
    report = {"amplification": 64.0, "onnx_file": wr._PHI35_SPEC.onnx_filename}

    wr.publish(report, out)

    assert json.loads(out.read_text(encoding="utf-8")) == report


def test_the_default_destination_is_not_the_committed_witness():
    """Defect 1. A default invocation must be incapable of dirtying tracked evidence."""
    assert wr._parse_args([]).out == wr.DEFAULT_OUT
    assert wr.DEFAULT_OUT.resolve() != RECORD.resolve()


def test_the_default_destination_is_actually_gitignored():
    """...and "not the witness" is not enough -- it must also not be a NEW tracked file.

    Asked of git rather than of `.gitignore`'s text, because the text is what someone edits and
    the answer is what someone gets. `check-ignore` exits 0 when the path is ignored, 1 when it
    is not; anything else is git failing to answer and is not read as a pass.
    """
    import shutil
    import subprocess

    git = shutil.which("git")
    assert git, "git is required to answer whether the probe default is ignored"
    proc = subprocess.run(
        [git, "check-ignore", "-q", str(wr.DEFAULT_OUT)],
        cwd=str(ROOT), capture_output=True,
    )
    assert proc.returncode == 0, (
        f"{wr.DEFAULT_OUT} is not gitignored (git check-ignore exit {proc.returncode}); a "
        f"default probe run would create tracked-looking output"
    )


def test_an_explicit_out_is_still_honoured():
    """Publishing to the committed witness stays possible -- it just has to be asked for."""
    assert wr._parse_args(["--out", str(RECORD)]).out == RECORD
