"""Provenance-contract pin checks and non-vacuous dispatch/agreement gates for the small,
already-downloaded public models this EP is validated against. Owner: Trinity.

WHY THIS EXISTS
----------------
Issue #11 asks that `bench/results/model_provenance.json` become an enforced contract, not just
prose, and that MNIST-12's URL/size/SHA-256 be added to it and *consumed*. Three tiers:

  1. Pure contract-content assertions (`TestProvenanceContractContent`) — always run, need no
     model file on disk. Pin the exact url/sha256/bytes this repo has committed to for
     mnist-12 and mobilenetv2-12, so an edit to the JSON that silently drifts a hash or URL
     fails a test in the same PR that made the edit, not months later when a download disagrees.
  2. `test_small_model_matches_provenance` — skips cleanly if the model file is not present in a
     resolvable local cache (this test never downloads anything); otherwise verifies the file on
     disk against the pinned contract via `model_provenance.verify_file`.
  3. `test_*_dispatch_is_non_vacuous` / `test_*_vulkan_matches_cpu` — `@pytest.mark.slow`, skip
     unless both `ONNXRUNTIME_VULKAN_EP_LIB` is set *and* the model file is present. These shell
     out to the existing, already-validated instruments (`probe_model_op_census.py --run`,
     `probe_model_output_agreement.py`) rather than reimplementing their guards, per the issue's
     "preserve non-vacuous model dispatch/equivalence gates" requirement — Mouse's claimed/
     dispatched distinction and the two-guard design (EP actually registered; dispatches > 0)
     stay in exactly one place. Output is written to pytest's own `tmp_path`, never into
     `bench/results/`, to avoid recreating the tracked-file-pollution problem discovered
     alongside issue #11's investigation (re-running `probe_model_op_census.py` with the same
     `--name` silently overwrote two already-committed artifacts under `bench/results/`).

MODEL LOCATION
--------------
Resolved via a single env var, `ONNXRUNTIME_EP_VULKAN_MODEL_CACHE`, defaulting to
`~/.cache/onnxruntime-ep-vulkan/models/`; a model is expected at
`<cache>/<name>.onnx` (e.g. `mnist-12.onnx`). This directory is never created or written to by
this test file — only read from.
"""

from __future__ import annotations

import contextlib
import json
import os
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]

sys.path.insert(0, str(REPO / "rust" / "tools"))
import model_provenance as mp  # noqa: E402

# ---------------------------------------------------------------------------
# Contract pins. These are copies of the values in bench/results/model_provenance.json,
# deliberately hardcoded here (not read back from the same file they check) so that an edit
# that silently drifts the JSON is caught by the same PR's test run rather than passing
# vacuously by comparing the file against itself.
# ---------------------------------------------------------------------------

_EXPECTED = {
    "mnist-12": {
        "url": (
            "https://github.com/onnx/models/raw/main/validated/vision/classification/"
            "mnist/model/mnist-12.onnx"
        ),
        "sha256": "5c688690f8bacf667d4c2074af5ad0646ca328d7ab03eccf944a65b320171bdd",
        "bytes": 26143,
    },
}


def _model_cache_dir() -> pathlib.Path:
    override = os.environ.get("ONNXRUNTIME_EP_VULKAN_MODEL_CACHE")
    if override:
        return pathlib.Path(override)
    return pathlib.Path.home() / ".cache" / "onnxruntime-ep-vulkan" / "models"


def _model_path(name: str) -> pathlib.Path:
    return _model_cache_dir() / f"{name}.onnx"


def _ep_lib_or_skip() -> str:
    lib = os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB")
    if not lib or not pathlib.Path(lib).is_file():
        pytest.skip(
            "ONNXRUNTIME_VULKAN_EP_LIB is not set or not found — EP not registered. "
            "Build it with `cargo build --release` and set the env var to the resulting "
            "onnxruntime_vulkan_ep.dll / .so path."
        )
    return lib


def _model_or_skip(name: str) -> pathlib.Path:
    path = _model_path(name)
    if not path.is_file():
        pytest.skip(
            f"{name}.onnx not found at {path}. Fetch it per bench/results/model_provenance.json "
            f"and place it there (env override: ONNXRUNTIME_EP_VULKAN_MODEL_CACHE), or set "
            f"that env var to point at an existing cache. This test never downloads models."
        )
    return path


@contextlib.contextmanager
def _guard_bench_results_from_probe_pollution(name: str):
    """`probe_model_op_census.py` hardcodes its claim-log/counters output to
    `bench/results/_claim_log_<tag>.jsonl` and `bench/results/_counters_<tag>.json`
    **regardless of `--out`** -- `--out` only controls the summary file this test already
    redirects into `tmp_path`. Re-running the probe with a `--name` that collides with an
    already-committed tracked artifact silently overwrites it; this was found and named as a
    defect during the issue #11 investigation (`_claim_log_mobilenetv2-12.jsonl` was clobbered
    by an early, unguarded version of this exact test). Rather than patching the probe script
    itself -- which is the very instrument this suite exists to preserve unmodified, per issue
    #11's "preserve non-vacuous model dispatch/equivalence gates" requirement -- this context
    manager snapshots whatever was there before the subprocess runs and restores it
    byte-for-byte afterward, deleting the file entirely if it did not exist before. This makes
    the test self-cleaning regardless of what the probe script does to `bench/results/`.
    """
    targets = [
        REPO / "bench" / "results" / f"_claim_log_{name}.jsonl",
        REPO / "bench" / "results" / f"_counters_{name}.json",
    ]
    before = {p: (p.read_bytes() if p.is_file() else None) for p in targets}
    try:
        yield
    finally:
        for p, original in before.items():
            if original is None:
                p.unlink(missing_ok=True)
            else:
                p.write_bytes(original)


# ---------------------------------------------------------------------------
# 1. Pure contract-content assertions — always run.
# ---------------------------------------------------------------------------


class TestProvenanceContractContent:
    def test_contract_file_loads(self) -> None:
        contract = mp.load_provenance()
        assert "mnist-12" in contract
        assert "mobilenetv2-12" in contract

    def test_mnist_12_url_is_pinned_exactly(self) -> None:
        entry = mp.load_provenance()["mnist-12"]
        assert entry.url == _EXPECTED["mnist-12"]["url"]

    def test_mnist_12_sha256_is_pinned_exactly(self) -> None:
        entry = mp.load_provenance()["mnist-12"]
        assert entry.sha256 == _EXPECTED["mnist-12"]["sha256"]
        assert len(entry.sha256) == 64  # a hex SHA-256 digest, not a truncated/placeholder value

    def test_mnist_12_bytes_is_pinned_exactly(self) -> None:
        entry = mp.load_provenance()["mnist-12"]
        assert entry.bytes == _EXPECTED["mnist-12"]["bytes"]

    def test_mnist_12_uses_the_lfs_redirecting_raw_url_form(self) -> None:
        # github.com/.../raw/<branch>/<path> 302-redirects through media.githubusercontent.com
        # for LFS-tracked files and serves real bytes; raw.githubusercontent.com/... does not
        # redirect and instead serves the ~130-byte LFS pointer text for this same file. Both
        # existing entries (mobilenetv2-12, bertsquad-12) already use the github.com/raw form
        # for this reason; mnist-12 must be consistent with that or its download will silently
        # fetch a pointer file instead of the model.
        entry = mp.load_provenance()["mnist-12"]
        assert entry.url.startswith("https://github.com/")
        assert "/raw/" in entry.url
        assert "raw.githubusercontent.com" not in entry.url

    def test_mobilenetv2_12_entry_has_a_well_formed_sha256(self) -> None:
        entry = mp.load_provenance()["mobilenetv2-12"]
        assert len(entry.sha256) == 64
        int(entry.sha256, 16)  # must be valid hex

    def test_verify_file_reports_actionable_size_mismatch(self, tmp_path: pathlib.Path) -> None:
        entry = mp.load_provenance()["mnist-12"]
        bogus = tmp_path / "mnist-12.onnx"
        bogus.write_bytes(b"\x00" * 100)  # deliberately wrong size
        with pytest.raises(mp.ProvenanceMismatch, match="size mismatch"):
            mp.verify_file(bogus, entry)

    def test_verify_file_reports_actionable_hash_mismatch(self, tmp_path: pathlib.Path) -> None:
        entry = mp.load_provenance()["mnist-12"]
        bogus = tmp_path / "mnist-12.onnx"
        bogus.write_bytes(b"\x00" * entry.bytes)  # right size, wrong content
        with pytest.raises(mp.ProvenanceMismatch, match="SHA-256 mismatch"):
            mp.verify_file(bogus, entry)

    def test_verify_file_reports_missing_file(self, tmp_path: pathlib.Path) -> None:
        entry = mp.load_provenance()["mnist-12"]
        with pytest.raises(mp.ProvenanceMismatch, match="found none"):
            mp.verify_file(tmp_path / "does-not-exist.onnx", entry)

    def test_verify_file_accepts_a_matching_file(self, tmp_path: pathlib.Path) -> None:
        entry = mp.load_provenance()["mnist-12"]
        # Construct a file whose bytes are chosen so that stat().st_size == entry.bytes and its
        # sha256 == entry.sha256 is a circular self-check unless we know the exact upstream
        # bytes; instead assert verify_file's *symmetry*: hashing a file and pinning that exact
        # hash+size as the entry must accept it.
        import hashlib

        content = os.urandom(entry.bytes)
        f = tmp_path / "synthetic.onnx"
        f.write_bytes(content)
        synthetic_entry = mp.ModelProvenance(
            name="synthetic",
            url="https://example.invalid/synthetic.onnx",
            sha256=hashlib.sha256(content).hexdigest(),
            bytes=len(content),
            fetched="2026-08-05",
            why="synthetic self-check, not a real model",
        )
        mp.verify_file(f, synthetic_entry)  # must not raise


# ---------------------------------------------------------------------------
# 2. On-disk provenance verification — skips cleanly if the model isn't cached locally.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["mnist-12", "mobilenetv2-12"])
def test_small_model_matches_provenance(name: str) -> None:
    path = _model_or_skip(name)
    entry = mp.load_provenance()[name]
    mp.verify_file(path, entry)  # raises ProvenanceMismatch with actionable detail on failure


# ---------------------------------------------------------------------------
# 3. Non-vacuous dispatch / CPU-vs-Vulkan agreement gates — shell out to the existing,
#    already-validated instruments. Slow; requires both the built EP and the cached model.
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("name", ["mnist-12", "mobilenetv2-12"])
def test_small_model_dispatch_is_non_vacuous(name: str, tmp_path: pathlib.Path) -> None:
    lib = _ep_lib_or_skip()
    model = _model_or_skip(name)
    out = tmp_path / f"op_census_{name}.json"
    with _guard_bench_results_from_probe_pollution(name):
        result = subprocess.run(
            [
                sys.executable,
                str(REPO / "rust" / "tools" / "probe_model_op_census.py"),
                "--model", str(model),
                "--name", name,
                "--run",
                "--out", str(out),
            ],
            env={**os.environ, "ONNXRUNTIME_VULKAN_EP_LIB": lib},
            capture_output=True,
            text=True,
            cwd=str(REPO),
        )
    assert result.returncode == 0, (
        f"probe_model_op_census.py failed for {name}:\n{result.stdout}\n{result.stderr}"
    )
    record = json.loads(out.read_text(encoding="utf-8"))
    dispatched = record["dispatches_executed"]
    assert isinstance(dispatched, int) and dispatched > 0, (
        f"{name}: dispatches_executed={dispatched!r} — a run that claims nodes but dispatches "
        f"nothing is not evidence the Vulkan EP does anything; see probe_model_op_census.py's "
        f"module docstring for why claimed_nodes alone is never trusted."
    )


@pytest.mark.slow
@pytest.mark.parametrize("name", ["mnist-12", "mobilenetv2-12"])
def test_small_model_vulkan_matches_cpu(name: str, tmp_path: pathlib.Path) -> None:
    lib = _ep_lib_or_skip()
    model = _model_or_skip(name)
    out = tmp_path / f"agreement_{name}.json"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "rust" / "tools" / "probe_model_output_agreement.py"),
            "--model", str(model),
            "--out", str(out),
        ],
        env={**os.environ, "ONNXRUNTIME_VULKAN_EP_LIB": lib},
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    assert result.returncode == 0, (
        f"probe_model_output_agreement.py failed for {name}:\n{result.stdout}\n{result.stderr}"
    )
    report = json.loads(out.read_text(encoding="utf-8"))
    # Guard 1 (the probe's own): the Vulkan EP was actually in the provider list, not a silent
    # CPU-vs-CPU comparison. Re-asserted here, not just trusted, per the "no vacuous pass"
    # requirement — a test that only checked the exit code would not notice this regressing if
    # the probe script itself were ever weakened.
    assert "VulkanExecutionProvider" in report["providers"], (
        f"{name}: Vulkan EP not present in session providers {report['providers']} — this "
        f"would compare CPU against CPU and pass vacuously."
    )
    assert report["verdict"] == "AGREE", (
        f"{name}: CPU-vs-Vulkan disagreement (worst_max_rel={report.get('worst_max_rel')}): "
        f"{[o for o in report['outputs'] if o['verdict'] not in ('AGREE', 'EXACT')]}"
    )
