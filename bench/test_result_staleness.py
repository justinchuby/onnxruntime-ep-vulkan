"""Results must not outlive the code they measured.

Issue #69's replacement branch sits on top of PR #72, which changed what the Vulkan
attention path *does*.  Every Vulkan number measured before that change describes a
different program.  The instruction that produced this file was blunt about it: do not
carry pre-#72 baseline rankings forward as current results.

A prose disclaimer would not survive the next person in a hurry, so the rule is
mechanical here.  Each committed suite record pins the digests of the sources that
decide attention behaviour.  If the tree's copy of those files no longer matches what a
record was measured against, that record is stale *by construction* and this module
says so by name.

The screen is deliberately one-directional.  It cannot tell you a result is still
correct -- only that it cannot possibly be, because the code moved underneath it.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import cuda_competition as cc  # noqa: E402
from _polarity import omits, records  # noqa: E402

CUDA69 = HERE / "results" / "_cuda69"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


# --- the instrument itself, falsified -------------------------------------------


def test_ep_provenance_pins_the_sources_that_decide_attention():
    prov = cc.ep_provenance()
    assert set(prov["pinned_sources"]) == set(cc.EP_PINNED_SOURCES)
    for rel, digest in prov["pinned_sources"].items():
        assert digest == _sha256(REPO / rel), rel


def test_ep_provenance_records_the_binary_that_actually_ran(monkeypatch, tmp_path):
    lib = tmp_path / "fake_ep.dll"
    lib.write_bytes(b"not really a dll, but it is the bytes that were loaded")
    monkeypatch.setenv(cc.EP_LIB_ENV, str(lib))
    prov = cc.ep_provenance()
    assert prov["lib_sha256"] == _sha256(lib)
    assert prov["lib_bytes"] == lib.stat().st_size


def test_ep_provenance_omits_a_library_digest_it_could_not_take(monkeypatch):
    """An absent EP is reported as absent, never as a digest of nothing."""
    monkeypatch.delenv(cc.EP_LIB_ENV, raising=False)
    why = omits(cc.ep_provenance(), "lib_sha256",
                reason_field="lib_unavailable_because")
    assert cc.EP_LIB_ENV in why


def test_ep_provenance_records_the_digest_when_the_library_is_present(monkeypatch, tmp_path):
    lib = tmp_path / "ep.dll"
    lib.write_bytes(b"\x00\x01\x02")
    monkeypatch.setenv(cc.EP_LIB_ENV, str(lib))
    records(cc.ep_provenance(), "lib_sha256", _sha256(lib))


def test_ep_provenance_notices_when_a_pinned_source_is_dirty(monkeypatch):
    """The narrow dirtiness question must actually consult the pinned set."""
    seen: "list[list[str]]" = []
    real = subprocess.run

    def fake(cmd, *a, **kw):
        if isinstance(cmd, list) and "--porcelain" in cmd and "--" in cmd:
            seen.append(cmd[cmd.index("--") + 1:])
        return real(cmd, *a, **kw)

    monkeypatch.setattr(cc.subprocess, "run", fake)
    cc.ep_provenance()
    assert seen and set(seen[0]) == set(cc.EP_PINNED_SOURCES)


def test_a_digest_change_is_detected(tmp_path):
    """The comparison is not vacuous: perturb a byte and it must disagree."""
    a = tmp_path / "a"
    a.write_bytes(b"kernel source v1")
    first = _sha256(a)
    a.write_bytes(b"kernel source v2")
    assert _sha256(a) != first


# --- the screen over committed evidence: HELD until the harness is approved ---------
#
# The rest of this module screens committed suite records: that each carries
# ``ep_provenance``, that its pinned digests still match the tree, that a record
# claiming to be superseded names a successor which exists, and that the only artifact
# carrying counter-inflated timings is the one declared as such.
#
# Those screens are deliberately NOT on this head. They can only be green once at least
# one committed artifact carries provenance, and an artifact can only carry provenance
# once a harness has measured it. Submitting them now would mean submitting a screen
# that is red by construction, or -- worse -- quietly weakening it until it passed
# against artifacts measured by an unapproved harness.
#
# The ordering is the point: approve the instrument, then measure with it, then commit
# the screen and the artifacts it rules on together. This file therefore stops at the
# falsification of ``ep_provenance`` itself, which needs nothing but the tree.
#
# Note that B1's requirement -- every committed Vulkan record declares its
# ``counters_scope`` -- is already enforced on this head, independently, by
# ``bench/test_cuda_competition.py::test_every_committed_vulkan_record_declares_its_counters_scope``.
# Nothing is unguarded in the meantime.
