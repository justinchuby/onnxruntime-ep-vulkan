"""One reader for `VulkanEpCounters`, derived from `counters.rs` rather than hand-maintained.

`a52024f` inserted `device_losses` mid-struct without updating the three hand-written ctypes
mirrors, so `dispatches_executed` silently read `device_losses` — zero on every healthy run, stable
and plausible and therefore invisible. The fix at the time was a `struct_size` equality guard, and
the real defect was named and left open: **three hand-maintained mirrors of one ABI.**

It has now happened again. `898a2ba` inserted `outputs_device_resident`, `outputs_host_resident`
and `outputs_device_bound` between `device_losses` and `dispatches_executed`. Same shape, same
place, three fields instead of one.

This module removes the thing that drifts. The field list is **parsed out of `rust/src/counters.rs`
at import time**, so a mirror cannot fall behind the struct: there is no second list to forget.
The parse is deliberately narrow and fails loudly rather than guessing — a reader that silently
produced a plausible layout would reintroduce exactly the defect it exists to remove.

Usage:

    from counters_abi import counters_from_dll, field_names
    c = counters_from_dll(dll_path)
    print(c.dispatches_executed)

Guarantees, each of which is a test in `main()`:

* `ctypes.sizeof(mirror) == c.struct_size` exactly. Not `<=`: an *append* is safe to read with a
  short mirror, an *insertion* is not, and the two are indistinguishable from the size alone.
* `abi_version` matches the constant parsed from the same file.
* The field order is the struct's order, because it is the struct's order.
"""

from __future__ import annotations

import ctypes
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[2]
COUNTERS_RS = REPO / "rust" / "src" / "counters.rs"

_STRUCT_RE = re.compile(r"pub struct VulkanEpCounters\s*\{(.*?)\n\}", re.S)
_FIELD_RE = re.compile(r"^\s*pub (\w+): (u32|u64),", re.M)
_ABI_RE = re.compile(r"pub const COUNTERS_ABI_VERSION: u32 = (\d+);")

_CTYPE = {"u32": ctypes.c_uint32, "u64": ctypes.c_uint64}


def _source() -> str:
    return COUNTERS_RS.read_text(encoding="utf-8")


def field_spec() -> list[tuple[str, str]]:
    """`[(name, 'u32'|'u64'), …]` in declaration order, parsed from `counters.rs`."""
    src = _source()
    body = _STRUCT_RE.search(src)
    if body is None:
        raise RuntimeError(
            f"could not find `pub struct VulkanEpCounters` in {COUNTERS_RS}. Refusing to guess a "
            f"layout: a plausible wrong one is the defect this module exists to remove."
        )
    fields = _FIELD_RE.findall(body.group(1))
    if len(fields) < 2 or fields[0][0] != "struct_size" or fields[1][0] != "abi_version":
        raise RuntimeError(
            f"parsed {len(fields)} field(s) from VulkanEpCounters and the first two are not "
            f"(struct_size, abi_version): {fields[:3]}. The struct's own contract is that a reader "
            f"can validate before it reads, so this is unreadable rather than differently readable."
        )
    return fields


def field_names() -> list[str]:
    return [n for n, _ in field_spec()]


def abi_version() -> int:
    m = _ABI_RE.search(_source())
    if m is None:
        raise RuntimeError(f"no COUNTERS_ABI_VERSION in {COUNTERS_RS}")
    return int(m.group(1))


def make_mirror() -> type[ctypes.Structure]:
    """A `ctypes.Structure` whose layout is the current `VulkanEpCounters`."""
    return type(
        "VulkanEpCounters",
        (ctypes.Structure,),
        {"_fields_": [(name, _CTYPE[ty]) for name, ty in field_spec()]},
    )


def counters_from_dll(dll_path: str | pathlib.Path):
    """Call `OrtEpVulkanGetExecutionCounters` and return the filled structure.

    Raises if the DLL's `struct_size` is not exactly this mirror's size. **Equality, not `<=`.**
    A short mirror over an appended struct reads correct values for the fields it knows; a short
    mirror over an *inserted* field reads different fields under the right names, and the size
    comparison is the only place the two can be told apart.
    """
    mirror = make_mirror()
    dll = ctypes.CDLL(str(dll_path))
    c = mirror()
    c.struct_size = ctypes.sizeof(mirror)
    # `OrtEpVulkanGetExecutionCounters(out, out_bytes)` — the length argument is not optional.
    # Omitting it does not read a shorter struct, it hands `fill` whatever was in the register.
    dll.OrtEpVulkanGetExecutionCounters(ctypes.byref(c), ctypes.c_size_t(ctypes.sizeof(mirror)))
    if c.struct_size != ctypes.sizeof(mirror):
        raise RuntimeError(
            f"the DLL's VulkanEpCounters is {c.struct_size} bytes, this mirror is "
            f"{ctypes.sizeof(mirror)}. A mirror that is the wrong size does not read smaller "
            f"numbers, it reads different fields."
        )
    if c.abi_version != abi_version():
        raise RuntimeError(
            f"DLL abi_version={c.abi_version}, source says {abi_version()}: the loaded DLL was not "
            f"built from this checkout's counters.rs."
        )
    return c


def main() -> int:
    import os
    import sys

    spec = field_spec()
    mirror = make_mirror()
    print(f"counters.rs        : {COUNTERS_RS}")
    print(f"abi_version        : {abi_version()}")
    print(f"fields             : {len(spec)}")
    print(f"sizeof(mirror)     : {ctypes.sizeof(mirror)} bytes")
    for name, ty in spec:
        print(f"    +{getattr(mirror, name).offset:>4}  {name}: {ty}")

    dll_path = os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB")
    if not dll_path:
        print("\nONNXRUNTIME_VULKAN_EP_LIB unset — layout printed, DLL not read.")
        return 0
    try:
        c = counters_from_dll(dll_path)
    except Exception as exc:  # noqa: BLE001
        # R13: quote the failure text, never a failure count.
        print(f"\nERROR(instrument): {exc}")
        return 2
    print(f"\nread {dll_path}")
    for name, _ in spec:
        print(f"    {name} = {getattr(c, name)}")

    if "--compare" in sys.argv:
        print()
        print(misattribution_report())
        stale = {}
        for rel, mirrors in hand_mirrors().items():
            for i, names in enumerate(mirrors):
                sm = type(
                    "Stale",
                    (ctypes.Structure,),
                    {
                        "_fields_": [
                            (n, ctypes.c_uint32 if n in ("struct_size", "abi_version")
                             else ctypes.c_uint64)
                            for n in names
                        ]
                    },
                )
                dll = ctypes.CDLL(str(dll_path))
                s = sm()
                s.struct_size = ctypes.sizeof(sm)
                dll.OrtEpVulkanGetExecutionCounters(
                    ctypes.byref(s), ctypes.c_size_t(ctypes.sizeof(sm))
                )
                stale[f"{rel}[{i}]"] = {
                    n: {"stale_reads": getattr(s, n), "true_value": getattr(c, n, None)}
                    for n in names
                    if getattr(s, n) != getattr(c, n, None)
                }
        import json as _json

        art = REPO / "bench" / "results" / "counters_abi_drift.json"
        art.parent.mkdir(parents=True, exist_ok=True)
        art.write_text(
            _json.dumps(
                {
                    "dll": str(dll_path),
                    "struct_fields": field_names(),
                    "struct_bytes": ctypes.sizeof(mirror),
                    "no_duration_quoted": "Every value here is a counter read twice.",
                    "misattribution": misattribution_report(),
                    "disagreements": stale,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n[abi-drift] -> {art}")
        for k, v in stale.items():
            for n, d in v.items():
                print(f"  {k}: {n} stale-reads {d['stale_reads']}, true value "
                      f"{d['true_value']}")
    print("\nPASS(the derived mirror is exactly the size the DLL reports)")
    return 0


# ---------------------------------------------------------------------------
# What a stale mirror actually reads
# ---------------------------------------------------------------------------

_MIRROR_FILES = (
    pathlib.Path("tests") / "ops" / "test_wiring_census.py",
    pathlib.Path("tests") / "ops" / "test_phi35.py",
)
_HAND_FIELD_RE = re.compile(r'\("(\w+)",\s*_?ct(?:ypes)?\.c_uint(32|64)\)')


def hand_mirrors() -> dict[str, list[str]]:
    """The hand-written ctypes field lists still living in the test suite, by file."""
    found: dict[str, list[str]] = {}
    for rel in _MIRROR_FILES:
        path = REPO / rel
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for m in re.finditer(r'\("struct_size",\s*_?ct(?:ypes)?\.c_uint32\)', text):
            names = [n for n, _ in _HAND_FIELD_RE.findall(text[m.start() : m.start() + 3000])]
            # Cut at the first repeat: a second mirror in the same file starts with struct_size.
            if "struct_size" in names[1:]:
                names = names[: names.index("struct_size", 1)]
            found.setdefault(str(rel), []).append(names)  # type: ignore[arg-type]
    return {k: v for k, v in found.items()}


def misattribution_report() -> str:
    """For each stale mirror, which field each of its names actually reads.

    This is the artifact, not the argument. A size mismatch says the layouts differ; it does not
    say that `dispatches_executed` is reading `outputs_device_resident`, and *that* is the sentence
    a reader of a red lane needs.
    """
    truth = field_names()
    lines: list[str] = []
    for rel, mirrors in hand_mirrors().items():
        for i, names in enumerate(mirrors):
            size = sum(4 if n in ("struct_size", "abi_version") else 8 for n in names)
            lines.append(f"{rel} [mirror {i}]: {len(names)} fields, {size} bytes "
                         f"(struct is {len(truth)} fields, "
                         f"{sum(4 if n in ('struct_size', 'abi_version') else 8 for n in truth)} "
                         f"bytes)")
            for pos, name in enumerate(names):
                actual = truth[pos] if pos < len(truth) else "PAST-END-OF-STRUCT"
                if actual != name:
                    lines.append(f"    {name:26s} reads  {actual}")
            if all(
                (pos < len(truth) and truth[pos] == name) for pos, name in enumerate(names)
            ):
                lines.append("    every name reads its own field (prefix of the struct)")
    return "\n".join(lines) or "no hand-written mirrors found"


if __name__ == "__main__":
    raise SystemExit(main())
