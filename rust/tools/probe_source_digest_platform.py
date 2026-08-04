"""Falsifier: is `source_digest` portable ACROSS PLATFORMS, or only across line endings?

WHY THIS EXISTS
---------------
`normalize_shader_text` (b1dab01) made `source_digest` invariant under CRLF/LF **within one
checkout**, and `probe_source_digest_eol.py` demonstrates exactly that. It does not answer the
question the Linux lane asks, which is a different one:

    the Windows build and the Linux build read *the same bytes on the same disk*
    (`/mnt/c/.../rust/shaders`). Do they hash them to the same `source_digest`?

If they do, `PROVEN-ELSEWHERE{toolchain}` is reachable on Linux and the two-digest schema does
the job it was added for. If they do not, the source witness is still a *platform* fingerprint,
the row stays unreachable, and the EOL repair — while correct — was aimed at a cause that was
not the whole cause.

This asks each artifact for its own digests (`OrtEpVulkanGetShaderSubject`). It re-derives
nothing. Run it once per platform with `--out`, then once with `--compare A.json B.json`.

Usage:
    python rust/tools/probe_source_digest_platform.py --out bench/results/<name>.json
    python rust/tools/probe_source_digest_platform.py --compare win.json linux.json
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import sys

HERE = pathlib.Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "rust" / "tools"))

import gen_proof_ledger as G  # noqa: E402

SHADER_ROOT = REPO / "rust" / "shaders"


def stems() -> list[str]:
    out = set()
    for p in (SHADER_ROOT / "glsl").glob("*.comp"):
        out.add(p.stem)
    table = REPO / "rust" / "src" / "ops" / "shader_variants.txt"
    if table.is_file():
        for line in table.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.add(line.split("\t")[0].strip())
    return sorted(out)


def source_bytes_digest() -> dict[str, str]:
    """FNV-1a/64 of each shader file's LF-normalised bytes, read by THIS process.

    Not the build's rule — a control. If these agree across platforms while the build's
    `source_digest` does not, the disagreement is in the build, not in the checkout.
    """
    import hashlib

    out = {}
    for p in sorted(SHADER_ROOT.rglob("*")):
        if p.is_file() and p.suffix in (".comp", ".glsl"):
            raw = p.read_bytes()
            raw = raw.removeprefix(b"\xef\xbb\xbf").replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            out[p.relative_to(SHADER_ROOT).as_posix()] = hashlib.sha256(raw).hexdigest()[:16]
    return out


def collect(lib: pathlib.Path) -> dict:
    all_stems = stems()
    subjects = {s: G._shader_subject(lib, [s]) for s in all_stems}
    return {
        "platform": platform.system(),
        "lib": str(lib),
        "modules": len(all_stems),
        "subjects": subjects,
        "source_files": source_bytes_digest(),
    }


def compare(a_path: pathlib.Path, b_path: pathlib.Path) -> int:
    a = json.loads(a_path.read_text(encoding="utf-8"))
    b = json.loads(b_path.read_text(encoding="utf-8"))
    an, bn = a["platform"], b["platform"]
    print(f"A = {an} ({a['lib']}), {a['modules']} module(s)")
    print(f"B = {bn} ({b['lib']}), {b['modules']} module(s)")

    files_moved = [
        k for k in sorted(set(a["source_files"]) | set(b["source_files"]))
        if a["source_files"].get(k) != b["source_files"].get(k)
    ]
    print(f"\nCONTROL: shader source files whose LF-normalised bytes differ between the two "
          f"readings: {len(files_moved)}")
    if files_moved:
        print("  (the arms are not reading the same source; everything below is uninterpretable)")
        for k in files_moved[:10]:
            print(f"   - {k}")

    common = sorted(set(a["subjects"]) & set(b["subjects"]))
    src_moved, spv_moved, both, neither = [], [], [], []
    for s in common:
        sa, sb = a["subjects"][s], b["subjects"][s]
        ds = sa.get("source_digest") != sb.get("source_digest")
        dv = sa.get("spirv_digest") != sb.get("spirv_digest")
        if ds and dv:
            both.append(s)
        elif ds:
            src_moved.append(s)
        elif dv:
            spv_moved.append(s)
        else:
            neither.append(s)

    print(f"\nMODULE ARITHMETIC over {len(common)} module(s) present in both:")
    print(f"  {len(neither):4d} identical in both witnesses")
    print(f"  {len(spv_moved):4d} SPIR-V moved, source SAME   <- PROVEN-ELSEWHERE{{toolchain}} is reachable here")
    print(f"  {len(src_moved):4d} source moved, SPIR-V same")
    print(f"  {len(both):4d} BOTH moved                  <- reads as SUBJECT-CHANGED; the row is unreachable")

    ta = {v.get("toolchain") for v in a["subjects"].values()}
    tb = {v.get("toolchain") for v in b["subjects"].values()}
    print(f"\n  {an} toolchain: {sorted(ta)}")
    print(f"  {bn} toolchain: {sorted(tb)}")

    portable = not src_moved and not both
    print("")
    if files_moved:
        print("ERROR(instrument): the two arms did not read the same source bytes.")
        return 3
    if portable:
        print("PASS — source_digest is a fact about the SOURCE: it survives the platform change "
              "that moved the SPIR-V, so a toolchain-only difference is nameable on Linux.")
        return 0
    print(f"FAIL(condition) — source_digest moved on {len(src_moved) + len(both)} of {len(common)} "
          f"module(s) across {an}->{bn} while the source bytes did not. It is still a PLATFORM "
          f"fingerprint, so PROVEN-ELSEWHERE{{toolchain}} stays unreachable on the second platform.")
    print("  examples:", (both + src_moved)[:6])
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    args = ap.parse_args()

    if args.compare:
        return compare(pathlib.Path(args.compare[0]), pathlib.Path(args.compare[1]))

    lib = G._find_lib(os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB", ""))
    if lib is None or not lib.is_file():
        print("ERROR(instrument): no built EP found; set ONNXRUNTIME_VULKAN_EP_LIB")
        return 3
    data = collect(lib)
    out = pathlib.Path(args.out or (REPO / "bench" / "results" / "source_digest_platform.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{data['platform']}: {data['modules']} module(s) from {lib}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
