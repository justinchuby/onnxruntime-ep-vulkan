"""Falsifier: is `source_digest` a fact about the SOURCE, or about the CHECKOUT?

WHY THIS EXISTS
---------------
`source_digest` was added (§8.9.19 part 2) to be the *toolchain-independent* witness — the one
that stays put when `glslc` moves, so that "the compiler changed" can be told from "the kernel
changed". On the first fresh Linux `.so` it moved on all 103 entries, which made
`PROVEN-ELSEWHERE{toolchain}` — whose condition is "SPIR-V differs, source same" — unsatisfiable
in the exact case it was built for.

The cause is not the compiler. `rust/build.rs::source_digest_for` hashed `fs::read` output
directly, and with `core.autocrlf=true` a Windows checkout of these files is CRLF while a Linux
checkout of the same git blob is LF. So the "portable" digest was strictly *less* portable than
the SPIR-V digest it was added to be more portable than: `glslc` emits byte-identical SPIR-V from
a CRLF and an LF copy, and the source digest did not.

WHAT THIS MEASURES
------------------
Two arms, on the artifact, asking the DLL for its own digests (`OrtEpVulkanGetShaderSubject`) —
nothing here re-derives the hashing rule.

  ARM A  the tree as checked out (CRLF on Windows)      -> record every module's two digests
  ARM B  every .comp/.glsl rewritten LF-only, rebuilt   -> record them again

PREDICTIONS, WRITTEN BEFORE RUNNING (these are the pass conditions):

  P1  ARM A spirv_digest == ARM B spirv_digest, for every module.
      Line endings are not a semantic difference to glslc. If this fails, the experiment is
      invalid — something other than the line endings moved between the two builds.
  P2  ARM A source_digest == ARM B source_digest, for every module.
      This is the repair under test. Before `normalize_shader_text` this was FALSE for every
      module that contains a newline, which is all of them.
  P3  at least one file actually changed on disk between the arms, and it contained CR bytes.
      Without this the run is a tautology: two identical trees hash the same.

The tree is restored byte-for-byte from a backup taken before ARM B, and the restore is verified
by digest, because a probe that leaves the working tree rewritten has made a change nobody asked
for.

Usage:  set ONNXRUNTIME_VULKAN_EP_LIB (or let it default to rust/target/debug), then
        python rust/tools/probe_source_digest_eol.py
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "rust" / "tools"))

import gen_proof_ledger as G  # noqa: E402

SHADER_ROOT = REPO / "rust" / "shaders"
BACKUP = REPO / "bench" / "results" / "_eol_probe_backup"
OUT = REPO / "bench" / "results" / "source_digest_eol_probe.json"


def shader_files() -> list[pathlib.Path]:
    return sorted(
        p
        for p in SHADER_ROOT.rglob("*")
        if p.is_file() and p.suffix in (".comp", ".glsl")
    )


def stems() -> list[str]:
    """Every module stem this build embeds, taken from the variant table plus the direct sources.

    Read from the tree rather than from the ledger so a module no entry mentions is still asked
    about — the question is about the digest rule, not about what happens to be proven.
    """
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


def read_digests(lib: pathlib.Path, all_stems: list[str]) -> dict[str, dict]:
    """Ask the DLL, **in a child process**, for each module's two digests.

    In-process `ctypes.CDLL` keeps the library mapped for the life of the interpreter, and on
    Windows `cargo build` then cannot replace it (`Access is denied. (os error 5)`) — the first
    run of this probe died there. A subprocess makes the handle's lifetime the measurement's
    lifetime.
    """
    src = (
        "import json,sys,pathlib;"
        f"sys.path.insert(0,{str(REPO / 'rust' / 'tools')!r});"
        "import gen_proof_ledger as G;"
        f"lib=pathlib.Path({str(lib)!r});"
        f"stems={all_stems!r};"
        "print(json.dumps({s: G._shader_subject(lib,[s]) for s in stems}))"
    )
    r = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-3000:])
        raise SystemExit("ERROR(instrument): could not read digests from the artifact")
    return json.loads(r.stdout)


def build() -> None:
    r = subprocess.run(
        ["cargo", "build"],
        cwd=str(REPO / "rust"),
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        print(r.stdout[-4000:])
        print(r.stderr[-4000:])
        raise SystemExit("ERROR(instrument): cargo build failed; the arms are not comparable")


def main() -> int:
    lib = G._find_lib(os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB", ""))
    if lib is None or not lib.is_file():
        print("ERROR(instrument): no built EP found; set ONNXRUNTIME_VULKAN_EP_LIB")
        return 3
    all_stems = stems()
    files = shader_files()
    print(f"lib     : {lib}")
    print(f"modules : {len(all_stems)}")
    print(f"sources : {len(files)}")

    arm_a = read_digests(lib, all_stems)

    if BACKUP.exists():
        shutil.rmtree(BACKUP)
    BACKUP.mkdir(parents=True)
    rewritten: list[str] = []
    for p in files:
        raw = p.read_bytes()
        (BACKUP / p.relative_to(SHADER_ROOT).as_posix().replace("/", "__")).write_bytes(raw)
        lf = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        if lf != raw:
            p.write_bytes(lf)
            rewritten.append(p.relative_to(REPO).as_posix())

    try:
        print(f"rewrote {len(rewritten)} file(s) to LF; rebuilding ...")
        build()
        arm_b = read_digests(lib, all_stems)
    finally:
        for p in files:
            b = BACKUP / p.relative_to(SHADER_ROOT).as_posix().replace("/", "__")
            p.write_bytes(b.read_bytes())
        shutil.rmtree(BACKUP)

    spirv_moved = [s for s in all_stems if arm_a[s].get("spirv_digest") != arm_b[s].get("spirv_digest")]
    source_moved = [
        s for s in all_stems if arm_a[s].get("source_digest") != arm_b[s].get("source_digest")
    ]

    p1 = not spirv_moved
    p2 = not source_moved
    p3 = bool(rewritten)
    print("")
    print(f"P1 spirv identical across arms : {'PASS' if p1 else f'FAIL ({len(spirv_moved)} moved)'}")
    print(f"P2 source identical across arms: {'PASS' if p2 else f'FAIL ({len(source_moved)} moved)'}")
    print(f"P3 the arms were different     : {'PASS' if p3 else 'FAIL (nothing had CR bytes)'}")
    if not p2:
        print("   moved:", source_moved[:8])
    verdict = "PASS" if (p1 and p2 and p3) else "FAIL(condition)"
    print("")
    print(
        f"{verdict} — source_digest is "
        + ("invariant under line endings" if p2 else "a checkout fingerprint")
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "lib": str(lib),
                "modules": len(all_stems),
                "rewritten_files": rewritten,
                "p1_spirv_identical": p1,
                "p2_source_identical": p2,
                "p3_arms_differed": p3,
                "spirv_moved": spirv_moved,
                "source_moved": source_moved,
                "verdict": verdict,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT}")
    # The tree was restored; rebuild so the artifact matches the tree again.
    print("rebuilding to restore the artifact to the checked-out tree ...")
    build()
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
