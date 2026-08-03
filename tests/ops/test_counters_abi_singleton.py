"""The counter ABI field list may be declared in exactly two files. This lane enforces it.

WHY THIS EXISTS
===============
`a52024f` inserted `device_losses` into `VulkanEpCounters` mid-struct without updating the three
hand-written ctypes mirrors. Every field below it moved eight bytes; `dispatches_executed` began
reading `device_losses` — `0` on every healthy run, stable and plausible and therefore invisible.
Hours later `898a2ba` inserted three more fields in the same place and `ledger_entries` read **0**
against a true value of 97.

`rust/tools/counters_abi.py` was written between the two, to derive the mirror from `counters.rs`.
It did not prevent the second one, because the three hand mirrors were still there and the tests
still used them. **A generator that co-exists with the thing it replaces is a fourth mirror.**

So this is not a convention and not a review checklist. Both of those are what we already had, and
what we had is the state that produced the defect twice in one day. This lane reads every source
file in the tree and fails if any file outside the two permitted ones declares the counter field
list. The failure names the file and the fields, because `R13` wants the artifact and not a count.

WHAT COUNTS AS A DECLARATION
============================
Two or more counter field names appearing as string literals inside a ``_fields_ = [...]`` block,
or two or more appearing in a ``struct``-shaped `ctypes` construction. Two, not one: a file that
mentions `dispatches_executed` in prose or reads it off a dict is a *consumer*, and consumers are
the point of having a generator. It is the ordered *list* that carries layout, and it is layout
that is unsafe to state twice.

POSITIVE CONTROL
================
`test_the_detector_sees_a_mirror_that_is_actually_there` writes a synthetic mirror into a scratch
directory inside the repository and asserts the detector reports it. A control never seen in its
positive state has no demonstrated positive state — this lane's whole value is a claim about
something it does *not* find, and such a claim from an instrument that has never found anything is
indistinguishable from an instrument that cannot find anything.
"""

from __future__ import annotations

import pathlib
import re
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "rust" / "tools"))
import counters_abi  # noqa: E402

#: The two files permitted to state the field list: the ABI itself, and the generator that reads
#: it. Any third is the defect. Kept as repo-relative POSIX strings so the test text is portable.
PERMITTED = {
    "rust/src/counters.rs",
    "rust/tools/counters_abi.py",
    # This file names counter fields in its own prose and in its positive control.
    "tests/ops/test_counters_abi_singleton.py",
}

SEARCH_SUFFIXES = {".py", ".rs", ".c", ".cc", ".cpp", ".h", ".hpp", ".cs", ".ps1"}
SKIP_DIRS = {".git", "target", "node_modules", "__pycache__", ".venv", "venv", "build", "dist"}

_FIELDS_BLOCK = re.compile(r"_fields_\s*=\s*\[(.*?)\]", re.S)
_CTYPES_PAIR = re.compile(r"\(\s*[\"'](\w+)[\"']\s*,\s*(?:ctypes\.|_ct\.|c_)")


def _counter_field_names() -> set[str]:
    """The field names, from the one place they are declared."""
    return set(counters_abi.field_names())


def _candidate_files() -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    for path in REPO.rglob("*"):
        if path.suffix.lower() not in SEARCH_SUFFIXES or not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        out.append(path)
    return out


def find_mirrors(paths: list[pathlib.Path], permitted: set[str] | None = None) -> dict[str, list[str]]:
    """`{repo-relative path: [field names it declares]}` for every unpermitted mirror.

    Exposed rather than inlined so the positive control can call it on a directory of its own
    making, and so a developer can run it by hand on a branch before pushing.
    """
    permitted = PERMITTED if permitted is None else permitted
    fields = _counter_field_names()
    found: dict[str, list[str]] = {}
    for path in paths:
        try:
            rel = path.relative_to(REPO).as_posix()
        except ValueError:
            rel = path.as_posix()
        if rel in permitted:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "_fields_" not in text and "ctypes.Structure" not in text:
            continue
        declared: list[str] = []
        for block in _FIELDS_BLOCK.findall(text):
            names = [n for n in _CTYPES_PAIR.findall(block) if n in fields]
            if len(names) >= 2:
                declared.extend(names)
        if declared:
            found[rel] = declared
    return found


def test_only_the_generator_declares_the_counter_field_list() -> None:
    """No file outside `counters.rs` and `counters_abi.py` may state the counter field list."""
    mirrors = find_mirrors(_candidate_files())
    assert not mirrors, (
        "a hand-written mirror of VulkanEpCounters is back in the tree:\n"
        + "\n".join(
            f"  {rel}\n      declares: {', '.join(names)}" for rel, names in sorted(mirrors.items())
        )
        + "\n\nThis exact shape has produced a silently wrong number twice (a52024f, 898a2ba): a "
        "field inserted mid-struct shifts every field below it, and the mirror keeps reporting "
        "the old names over the new offsets — stable, plausible, wrong.\n"
        "Repair: delete the mirror and read counters through\n"
        "    sys.path.insert(0, <repo>/rust/tools); import counters_abi\n"
        "    counters_abi.read_counters()            # -> dict of every field\n"
        "    counters_abi.counters_from_dll(path)    # -> the filled ctypes struct\n"
        "which derives its layout from rust/src/counters.rs and cross-checks it against the "
        "per-field offset manifest the DLL publishes (OrtEpVulkanGetCountersLayout)."
    )


def test_the_detector_sees_a_mirror_that_is_actually_there(tmp_path_factory) -> None:
    """Positive control: plant a mirror, and the detector must name it and its fields.

    Without this the lane above is a green light from an instrument that has never been shown to
    be able to turn red — the same standing objection I raised against the elementwise gate.
    """
    scratch = REPO / "tests" / "ops" / "_scratch_abi_singleton"
    scratch.mkdir(parents=True, exist_ok=True)
    planted = scratch / "planted_mirror.py"
    names = counters_abi.field_names()
    assert len(names) >= 4, names
    try:
        planted.write_text(
            "import ctypes\n\n"
            "class _EpCounters(ctypes.Structure):\n"
            "    _fields_ = [\n"
            + "".join(
                f'        ("{n}", ctypes.c_uint64),\n' for n in names[2:6]
            )
            + "    ]\n",
            encoding="utf-8",
        )
        found = find_mirrors([planted])
        rel = planted.relative_to(REPO).as_posix()
        assert rel in found, f"the detector did not see a mirror it was handed: {found}"
        assert set(found[rel]) == set(names[2:6]), found[rel]
    finally:
        planted.unlink(missing_ok=True)
        try:
            scratch.rmdir()
        except OSError:
            pass


def test_a_consumer_is_not_mistaken_for_a_mirror(tmp_path_factory) -> None:
    """Negative control: reading a counter by name is not declaring the layout.

    If the detector fired on every mention of `dispatches_executed` it would be unusable and would
    be silenced, and a silenced detector is the state before this lane existed.
    """
    scratch = REPO / "tests" / "ops" / "_scratch_abi_singleton"
    scratch.mkdir(parents=True, exist_ok=True)
    consumer = scratch / "consumer.py"
    try:
        consumer.write_text(
            "import ctypes  # a consumer may still use ctypes for other structures\n\n"
            "class Unrelated(ctypes.Structure):\n"
            '    _fields_ = [("width", ctypes.c_uint32), ("height", ctypes.c_uint32)]\n\n'
            "def report(counters):\n"
            "    return counters['dispatches_executed'] - counters['ledger_entries']\n",
            encoding="utf-8",
        )
        assert find_mirrors([consumer]) == {}
    finally:
        consumer.unlink(missing_ok=True)
        try:
            scratch.rmdir()
        except OSError:
            pass


def test_the_generator_agrees_with_the_rust_layout_registry() -> None:
    """The layout this checkout computes must be one somebody declared in `counters.rs`.

    Python and rustc compute the same FNV-1a/64 over `name:offset:size;` from two different
    sources — the parsed source text here, `std::mem::offset_of!` there. Agreement is the
    cross-check that neither model has drifted; the `const _` assertion in `counters.rs` fails the
    build if the pair is undeclared, and this states the same thing where a compiler is not handy.
    """
    version, digest = counters_abi.abi_version(), counters_abi.layout_hash()
    assert (version, digest) in counters_abi.layout_registry(), (
        f"VulkanEpCounters has layout hash 0x{digest:016x} under COUNTERS_ABI_VERSION={version}, "
        f"and COUNTERS_LAYOUT_REGISTRY has no such row. Bump COUNTERS_ABI_VERSION and append "
        f"({version + 1}, 0x{digest:016x}). Do not edit an existing row: a row is a promise about "
        f"a layout that other builds may still be reading."
    )


def test_an_inserted_field_changes_the_layout_hash() -> None:
    """The mechanism itself, in its positive state: replay `898a2ba` against the parser.

    `898a2ba` inserted `outputs_device_resident`, `outputs_host_resident` and
    `outputs_device_bound` between `device_losses` and `dispatches_executed` and left
    `COUNTERS_ABI_VERSION` at 4, so version 4 named two different layouts. This applies the same
    edit to the source *text* and asserts the resulting `(version, hash)` pair is undeclared —
    the Python half of the acceptance test whose Rust half fails the build.
    """
    src = counters_abi.COUNTERS_RS.read_text(encoding="utf-8")
    assert "pub device_losses: u64," in src
    mutated = src.replace(
        "pub device_losses: u64,",
        "pub device_losses: u64,\n        pub scratch_inserted_a: u64,\n"
        "        pub scratch_inserted_b: u64,\n        pub scratch_inserted_c: u64,",
        1,
    )
    assert mutated != src

    before = counters_abi.layout_hash(src)
    after = counters_abi.layout_hash(mutated)
    assert before != after, "a mid-struct insertion did not change the layout hash"
    assert (counters_abi.abi_version(mutated), after) not in counters_abi.layout_registry(mutated), (
        "an insertion that forgot the version bump produced a layout the registry already "
        "declares — the guard would have passed it, which is the whole defect."
    )

    # And the misattribution report must name the specific wrong reading, not merely differ.
    shifted = counters_abi.expected_offsets(mutated)
    report = counters_abi.misattribution(shifted, src)
    assert "dispatches_executed" in report or "device_losses" in report, report


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
