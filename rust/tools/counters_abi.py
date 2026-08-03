"""The **only** ctypes reader for `VulkanEpCounters`, derived from `counters.rs`.

`a52024f` inserted `device_losses` mid-struct without updating the three hand-written ctypes
mirrors, so `dispatches_executed` silently read `device_losses` — zero on every healthy run, stable
and plausible and therefore invisible. `898a2ba` then inserted `outputs_device_resident`,
`outputs_host_resident` and `outputs_device_bound` in the same place, and `ledger_entries` read
**0** against a true value of 97. Same shape, same place, three fields instead of one — and the
second time happened *after* this file existed, because this file co-existed with the three mirrors
it was meant to replace. **A generator that co-exists with the thing it replaces is a fourth
mirror.** The mirrors are gone; `tests/ops/test_counters_abi_singleton.py` fails if one comes back.

Three derivations, one source:

* the **field list** is parsed out of `rust/src/counters.rs` at import time;
* the **layout hash** is computed here by the same rule `counters.rs` computes it with `const fn`,
  so Python and rustc independently agree or the disagreement is loud;
* the **DLL's own manifest** (`OrtEpVulkanGetCountersLayout`) is read back and compared field by
  field, so a stale DLL is a named mismatch rather than a plausible number.

Usage:

    from counters_abi import counters_from_dll, read_counters, field_names
    c = counters_from_dll(dll_path)
    print(c.dispatches_executed)

    values = read_counters(dll_path)   # dict, every field, or {} if the env var is unset

Guarantees, each of which is exercised by `main()` and by the singleton lane:

* `ctypes.sizeof(mirror) == c.struct_size` exactly. Not `<=`: an *append* is safe to read with a
  short mirror, an *insertion* is not, and the two are indistinguishable from the size alone.
* Every field is at the offset the DLL publishes for that name.
* `abi_version` and the layout hash match the constants parsed from the same file.
* The field order is the struct's order, because it is the struct's order.
"""

from __future__ import annotations

import ctypes
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[2]
COUNTERS_RS = REPO / "rust" / "src" / "counters.rs"

_STRUCT_RE = re.compile(r"pub struct VulkanEpCounters\s*\{(.*?)\n\s*\}", re.S)
_FIELD_RE = re.compile(r"^\s*pub (\w+): (u32|u64),", re.M)
_ABI_RE = re.compile(r"pub const COUNTERS_ABI_VERSION: u32 = (\d+);")

_CTYPE = {"u32": ctypes.c_uint32, "u64": ctypes.c_uint64}
_WIDTH = {"u32": 4, "u64": 8}


class CountersAbiMismatch(RuntimeError):
    """The DLL's counter layout is not the one this checkout declares.

    `ERROR(instrument)`, never a detection (R13): the honest output of a reader that cannot locate
    its fields is a raise. It is deliberately **not** caught into an empty dict anywhere — an empty
    dict differences to a delta of 0, and a delta of 0 is the wiring census's most serious verdict.
    """


def _source() -> str:
    return COUNTERS_RS.read_text(encoding="utf-8")


def field_spec(src: str | None = None) -> list[tuple[str, str]]:
    """`[(name, 'u32'|'u64'), ...]` in declaration order, parsed from `counters.rs`.

    `src` overrides the file, so a test can feed this a mutated struct and watch the guards fire
    without editing the tree.
    """
    src = _source() if src is None else src
    body = _STRUCT_RE.search(src)
    if body is None:
        raise CountersAbiMismatch(
            f"could not find `pub struct VulkanEpCounters` in {COUNTERS_RS}. Refusing to guess a "
            f"layout: a plausible wrong one is the defect this module exists to remove."
        )
    fields = _FIELD_RE.findall(body.group(1))
    if len(fields) < 2 or fields[0][0] != "struct_size" or fields[1][0] != "abi_version":
        raise CountersAbiMismatch(
            f"parsed {len(fields)} field(s) from VulkanEpCounters and the first two are not "
            f"(struct_size, abi_version): {fields[:3]}. The struct's own contract is that a reader "
            f"can validate before it reads, so this is unreadable rather than differently readable."
        )
    return fields


def field_names(src: str | None = None) -> list[str]:
    return [n for n, _ in field_spec(src)]


def abi_version(src: str | None = None) -> int:
    m = _ABI_RE.search(_source() if src is None else src)
    if m is None:
        raise CountersAbiMismatch(f"no COUNTERS_ABI_VERSION in {COUNTERS_RS}")
    return int(m.group(1))


def layout_registry(src: str | None = None) -> list[tuple[int, int]]:
    """`COUNTERS_LAYOUT_REGISTRY` -- every `(version, layout hash)` the ABI has published."""
    src = _source() if src is None else src
    block = re.search(
        r"pub const COUNTERS_LAYOUT_REGISTRY: &\[\(u32, u64\)\] = &\[(.*?)\n\];", src, re.S
    )
    if block is None:
        raise CountersAbiMismatch(f"no COUNTERS_LAYOUT_REGISTRY in {COUNTERS_RS}")
    rows = re.findall(r"\((\d+),\s*0x([0-9a-fA-F_]+)\)", block.group(1))
    return [(int(v), int(h.replace("_", ""), 16)) for v, h in rows]


def layout_hash(src: str | None = None) -> int:
    """FNV-1a/64 over `name:offset:size;`, the rule `counters::counters_layout_hash` uses.

    Two independent implementations of one hash is not duplication: it is the cross-check. The
    Rust side hashes offsets the *compiler* assigned; this side hashes offsets a `repr(C)` reader
    computes. If they ever disagree the struct has padding, and every ctypes mirror in this
    repository -- including the one this file builds -- is reading the wrong bytes.
    """
    h = 0xCBF29CE484222325
    offset = 0
    for name, ty in field_spec(src):
        size = _WIDTH[ty]
        blob = f"{name}:{offset}:{size};".encode()
        for b in blob:
            h ^= b
            h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
        offset += size
    return h


def layout_is_declared(src: str | None = None) -> bool:
    """Whether this checkout's `(version, hash)` pair appears in the registry.

    The Python restatement of the `const` assertion in `counters.rs`. The Rust one fails the
    *build*, which is the real gate; this one exists so a tool can say *why* without a compiler.
    """
    return (abi_version(src), layout_hash(src)) in layout_registry(src)


def expected_offsets(src: str | None = None) -> list[tuple[str, int, int]]:
    """`[(name, offset, size), ...]` as a `repr(C)` reader computes them."""
    out: list[tuple[str, int, int]] = []
    offset = 0
    for name, ty in field_spec(src):
        out.append((name, offset, _WIDTH[ty]))
        offset += _WIDTH[ty]
    return out


def make_mirror(src: str | None = None) -> type[ctypes.Structure]:
    """A `ctypes.Structure` whose layout is the current `VulkanEpCounters`."""
    return type(
        "VulkanEpCounters",
        (ctypes.Structure,),
        {"_fields_": [(name, _CTYPE[ty]) for name, ty in field_spec(src)]},
    )


def dll_manifest(dll_path: str | pathlib.Path) -> dict:
    """Read the layout manifest the DLL publishes about itself.

    `{"abi_version": int, "layout_hash": int, "struct_size": int,
      "fields": [(name, offset, size), ...]}`.

    This is the per-field offset manifest my 2026-08-02 note asked for. A size check can say
    *that* two layouts differ; only this can say *how*, which is the sentence a reader of a red
    lane actually needs.
    """
    dll = ctypes.CDLL(str(dll_path))
    try:
        fn = dll.OrtEpVulkanGetCountersLayout
    except AttributeError as exc:
        raise CountersAbiMismatch(
            f"{dll_path} does not export OrtEpVulkanGetCountersLayout. That export was added with "
            f"the layout manifest; a DLL without it predates this checkout's counters.rs, so its "
            f"field offsets are unknown and no reading from it is attributable."
        ) from exc
    fn.restype = ctypes.c_size_t
    need = fn(None, ctypes.c_size_t(0))
    buf = ctypes.create_string_buffer(need + 1)
    fn(buf, ctypes.c_size_t(need))
    text = buf.raw[:need].decode("utf-8")
    out: dict = {"fields": []}
    for line in text.splitlines():
        if line.startswith("abi_version="):
            out["abi_version"] = int(line.split("=", 1)[1])
        elif line.startswith("layout_hash="):
            out["layout_hash"] = int(line.split("=", 1)[1], 16)
        elif line.startswith("struct_size="):
            out["struct_size"] = int(line.split("=", 1)[1])
        elif line.strip():
            off, size, name = line.split()
            out["fields"].append((name, int(off), int(size)))
    return out


def misattribution(dll_fields: list[tuple[str, int, int]], src: str | None = None) -> str:
    """Which field each of *our* names would actually read against the DLL's layout.

    The artifact, not the argument. `dispatches_executed reads outputs_device_resident` is the
    sentence; a size mismatch is not.
    """
    by_offset = {off: name for name, off, _ in dll_fields}
    lines: list[str] = []
    for name, off, _size in expected_offsets(src):
        actual = by_offset.get(off, "PAST-END-OF-STRUCT")
        if actual != name:
            lines.append(f"    {name:28s} would read  {actual}")
    return "\n".join(lines) or "    (no field reads a different field)"


def counters_from_dll(dll_path: str | pathlib.Path):
    """Call `OrtEpVulkanGetExecutionCounters` and return the filled structure.

    Checks, in order, each of which raises rather than returning a plausible number:

    1. the DLL's published manifest names the same fields at the same offsets as this checkout;
    2. the DLL's layout hash equals the one computed from `counters.rs`;
    3. `struct_size` is **exactly** this mirror's size -- equality, not `<=`. A short mirror over an
       appended struct reads correct values for the fields it knows; a short mirror over an
       *inserted* field reads different fields under the right names, and `<=` cannot tell the two
       apart. The census's old `struct_size <` guard is precisely why `898a2ba` went unseen.
    """
    mirror = make_mirror()
    manifest = dll_manifest(dll_path)
    ours = expected_offsets()
    if manifest["fields"] != ours:
        raise CountersAbiMismatch(
            f"the DLL's VulkanEpCounters layout is not this checkout's.\n"
            f"  DLL      : abi_version={manifest['abi_version']} "
            f"hash=0x{manifest['layout_hash']:016x} size={manifest['struct_size']} "
            f"({len(manifest['fields'])} fields)\n"
            f"  checkout : abi_version={abi_version()} hash=0x{layout_hash():016x} "
            f"size={ctypes.sizeof(mirror)} ({len(ours)} fields)\n"
            f"Read through this mirror, the DLL's numbers would be misattributed as:\n"
            f"{misattribution(manifest['fields'])}\n"
            f"Rebuild the EP from this checkout, or point ONNXRUNTIME_VULKAN_EP_LIB at a DLL that "
            f"was."
        )
    if manifest["layout_hash"] != layout_hash():
        raise CountersAbiMismatch(
            f"the DLL's field offsets match but its layout hash does not: DLL "
            f"0x{manifest['layout_hash']:016x}, checkout 0x{layout_hash():016x}. One of the two "
            f"hash implementations is wrong, so neither is a check (R13: an instrument error)."
        )

    dll = ctypes.CDLL(str(dll_path))
    c = mirror()
    c.struct_size = ctypes.sizeof(mirror)
    dll.OrtEpVulkanGetExecutionCounters(ctypes.byref(c), ctypes.c_size_t(ctypes.sizeof(mirror)))
    if c.struct_size != ctypes.sizeof(mirror):
        raise CountersAbiMismatch(
            f"the DLL's VulkanEpCounters is {c.struct_size} bytes, this mirror is "
            f"{ctypes.sizeof(mirror)}. A mirror that is the wrong size does not read smaller "
            f"numbers, it reads different fields."
        )
    if c.abi_version != abi_version():
        raise CountersAbiMismatch(
            f"DLL abi_version={c.abi_version}, source says {abi_version()}: the loaded DLL was not "
            f"built from this checkout's counters.rs."
        )
    return c


def read_counters(dll_path: str | pathlib.Path | None = None) -> dict[str, int]:
    """Every counter as a plain dict, or `{}` when no DLL path is available.

    `{}` means **the instrument was not present**, and it is returned for exactly one reason: no
    `ONNXRUNTIME_VULKAN_EP_LIB`. Every other failure raises `CountersAbiMismatch`, because an
    empty dict differences to a delta of 0 and a delta of 0 reads as `UNWIRED`.
    """
    import os

    path = dll_path or os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB")
    if not path:
        return {}
    c = counters_from_dll(path)
    return {name: getattr(c, name) for name in field_names() if name != "struct_size"}


def main() -> int:
    import os
    import sys

    spec = field_spec()
    mirror = make_mirror()
    print(f"counters.rs        : {COUNTERS_RS}")
    print(f"abi_version        : {abi_version()}")
    print(f"fields             : {len(spec)}")
    print(f"sizeof(mirror)     : {ctypes.sizeof(mirror)} bytes")
    print(f"layout_hash        : 0x{layout_hash():016x}")
    for name, ty in spec:
        print(f"    +{getattr(mirror, name).offset:>4}  {name}: {ty}")

    if layout_is_declared():
        print(
            f"\nPASS(layout declared: ({abi_version()}, 0x{layout_hash():016x}) is in "
            f"COUNTERS_LAYOUT_REGISTRY)"
        )
    else:
        # This is what the developer who just inserted a field needs to read. The build has
        # already failed by the time they get here; the row is the repair. Which repair it is
        # depends on whether the version has already been bumped: appending a second hash under a
        # version that already has one is the `898a2ba` defect itself — one number naming two
        # layouts — so the tool must not print that as advice.
        version, digest = abi_version(), layout_hash()
        taken = [v for v, _ in layout_registry()]
        if version in taken:
            repair = (
                f"  Repair: COUNTERS_ABI_VERSION={version} already has a row with a different\n"
                f"  hash, and one version may name only one layout — that is exactly the 898a2ba\n"
                f"  defect. Bump COUNTERS_ABI_VERSION to {max(taken) + 1} and append\n"
                f"      ({max(taken) + 1}, 0x{digest:016x}),"
            )
        else:
            repair = (
                f"  Repair: append\n"
                f"      ({version}, 0x{digest:016x}),"
            )
        print(
            f"\nFAIL(layout undeclared): VulkanEpCounters has layout hash "
            f"0x{digest:016x} and COUNTERS_LAYOUT_REGISTRY has no such row under "
            f"COUNTERS_ABI_VERSION={version}.\n"
            f"{repair}\n"
            f"  to COUNTERS_LAYOUT_REGISTRY. Do not edit an existing row."
        )
        if "--check" in sys.argv:
            return 1

    dll_path = os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB")
    if not dll_path:
        print("\nONNXRUNTIME_VULKAN_EP_LIB unset -- layout printed, DLL not read.")
        return 0
    try:
        manifest = dll_manifest(dll_path)
        print(
            f"\nDLL manifest       : abi_version={manifest['abi_version']} "
            f"hash=0x{manifest['layout_hash']:016x} size={manifest['struct_size']}"
        )
        c = counters_from_dll(dll_path)
    except Exception as exc:  # noqa: BLE001
        # R13: quote the failure text, never a failure count.
        print(f"\nERROR(instrument): {exc}")
        return 2
    print(f"\nread {dll_path}")
    for name, _ in spec:
        print(f"    {name} = {getattr(c, name)}")
    print("\nPASS(the derived mirror is exactly the size and shape the DLL publishes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
