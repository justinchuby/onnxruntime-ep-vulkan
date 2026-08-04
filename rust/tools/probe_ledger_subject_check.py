#!/usr/bin/env python
"""Falsifier for §8.9.21 — does `gen_proof_ledger.py --check` see a shader edit?

WHY THIS PROBE EXISTS AND WHY ITS CASE WAS NOT INVENTED
=======================================================
On 2026-08-03 Switch edited a GQA shader, did not regenerate the ledger, and ran a full session:
the EP declined **all 32 GroupQueryAttention nodes** for the whole run, and
`gen_proof_ledger.py --check` printed `PASS: 103 entr(ies)` on every invocation before and after.
Only `subject_changed_declines` saw it.

`--check` was answering a different question than the one it was quoted for. It compared the file
against itself — header digest against body, counts against lines, fields against shapes — and
every one of those was true of a ledger that describes a binary nobody built. "PASS at 103
entries" was quoted as merge evidence at least six times that day. It established internal
consistency. It did not establish that the file describes the artifact about to ship, which is
the only thing anybody reads it for.

So this probe replays Switch's case as a **positive control that is known to exist and known to
have passed** — the strongest kind there is, because it was not constructed to make a point.

ARMS (predictions written before the first run)
-----------------------------------------------
  1. baseline           unmodified tree, fresh build            -> PASS
  2. semantic edit      one line of gqa_f16.comp changed, rebuilt, ledger untouched
                                                                -> FAIL, naming the GQA entry
  3. restored           edit reverted, rebuilt                  -> PASS again
  4. stale binary       ledger rewritten (one byte), binary not rebuilt
                                                                -> FAIL(baked != on disk)
  5. no artifact        --check with a lib path that does not exist
                                                                -> ERROR(instrument), never PASS
  6. toolchain row      one entry's spirv_digest moved, source_digest LEFT CORRECT
                                                                -> that entry is PROVEN-ELSEWHERE{toolchain},
                                                                   NOT SUBJECT-CHANGED, and the word
                                                                   "toolchain" appears in a FAILING run
  7. both moved         same entry, spirv AND source both moved  -> SUBJECT-CHANGED, and it fails

Arm 3 is what makes arms 2 and 4 detections rather than a check that fails on everything.
Arm 5 is the dangling-reference arm: a check that cannot see its subject must not resolve anyway.

Arms 6 and 7 were added on 2026-08-03 after Link took a fresh Linux `.so` to this classifier and
got **0 of 103 agreeing** where Windows got 102. Two defects, both mine:

  * the classifier tested `spirv_digest` first and unconditionally, so `PROVEN-ELSEWHERE{toolchain}`
    — whose condition is "SPIR-V differs, source same" — could never be reached, and every Linux
    entry rendered as `SUBJECT-CHANGED`: not merely the wrong token but the strongest available
    accusation, telling a user their source moved when their compiler moved;
  * `cmd_check` printed its notes only after the PASS line and returned 1 from the failure branch
    before reaching them, so in a run where all 103 failures had a toolchain cause the word
    "toolchain" appeared nowhere. On PASS, where nothing needs explaining, it printed in full.

Arm 6 is the falsifier for both at once: it is a run that FAILS (the variant ledger is not the one
baked into the binary) in which the toolchain reading must nevertheless be printed. Arm 7 is what
keeps arm 6 from being satisfied by a classifier that has simply stopped failing.

Usage:  python rust/tools/probe_ledger_subject_check.py [--skip-rebuild-arms]

ONE ENTRY, THIRTY-TWO NODES
---------------------------
The ledger holds **one** GQA entry (`.../metadata/runtime-extent/...`, shaders `['gqa_f16']`);
Phi-3.5 has 32 GQA nodes, and all 32 claim from that single form. So the check names one entry
and the runtime declines thirty-two nodes. Both numbers are correct and they are not the same
number — the probe asserts the entry, and says so.

Usage:  python rust/tools/probe_ledger_subject_check.py [--skip-rebuild-arms]
"""
from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
TOOLS = REPO / "rust" / "tools"
SHADER = REPO / "rust" / "shaders" / "glsl" / "gqa_f16.comp"
LEDGER = REPO / "evidence" / "proof_ledger.jsonl"
GEN = TOOLS / "gen_proof_ledger.py"

# The edit Switch's case turns on: a real SPIR-V-moving change, not a comment. A comment-only
# edit produces identical SPIR-V and is SOURCE-COSMETIC by design (§8.9.19 row 4) — it must NOT
# fail this check, and using one here would have made the probe pass for the wrong reason.
#
# Chosen so that a copy left behind by a crash is harmless: the guard is unreachable (there is no
# invocation 0xFFFFFFFF in any dispatch this EP records) but the compiler cannot prove that, so
# the bytes move. Changing `local_size_x` would also move them and would leave a wrong kernel.
EDIT_FROM = "void main() {\n    uint gid = gl_GlobalInvocationID.x;"
EDIT_TO = (
    "void main() {\n"
    "    if (gl_GlobalInvocationID.x == 0xFFFFFFFFu) { return; }  // §8.9.21 subject probe\n"
    "    uint gid = gl_GlobalInvocationID.x;"
)


def _run(args: list[str]) -> tuple[int, str]:
    env = dict(os.environ)
    env.setdefault("VULKAN_SDK", r"C:\VulkanSDK\1.4.350.0")
    r = subprocess.run(
        [sys.executable, *args],
        capture_output=True,
        text=True,
        cwd=str(REPO / "rust"),
        env=env,
        timeout=1800,
    )
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def _check(extra: list[str] | None = None) -> tuple[int, str]:
    return _run([str(GEN), "--check", *(extra or [])])


def _build() -> None:
    env = dict(os.environ)
    env.setdefault("VULKAN_SDK", r"C:\VulkanSDK\1.4.350.0")
    env["PATH"] = str(pathlib.Path(env["VULKAN_SDK"]) / "Bin") + os.pathsep + env["PATH"]
    r = subprocess.run(
        ["cargo", "build"],
        capture_output=True,
        text=True,
        cwd=str(REPO / "rust"),
        env=env,
        timeout=3600,
    )
    if r.returncode != 0:
        raise SystemExit(f"build failed:\n{r.stdout}\n{r.stderr}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--skip-rebuild-arms",
        action="store_true",
        help="run only the arms that need no cargo build (1, 4, 5)",
    )
    args = ap.parse_args()

    results: list[tuple[str, bool, str]] = []

    def record(name: str, ok: bool, detail: str) -> None:
        results.append((name, ok, detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    original_shader = SHADER.read_text(encoding="utf-8")
    original_ledger = LEDGER.read_text(encoding="utf-8")
    if EDIT_FROM not in original_shader:
        print(
            f"ERROR(instrument): {SHADER.name} no longer contains {EDIT_FROM!r}. The probe would "
            f"silently make no edit and every arm would report the baseline — the exact failure "
            f"shape it exists to catch. Update EDIT_FROM/EDIT_TO."
        )
        return 3

    print("§8.9.21 ledger-subject falsifier — 7 arms")
    try:
        print("\narm 1: baseline — unmodified tree")
        _build()
        rc, out = _check()
        record("baseline", rc == 0 and "PASS:" in out, out.splitlines()[0] if out else "(silent)")

        if not args.skip_rebuild_arms:
            print("\narm 2: Switch's case — shader edited, ledger untouched, binary rebuilt")
            SHADER.write_text(original_shader.replace(EDIT_FROM, EDIT_TO), encoding="utf-8")
            _build()
            rc, out = _check()
            names_gqa = "GroupQueryAttention" in out
            record(
                "semantic-edit-fails",
                rc != 0 and names_gqa and "LEDGER_DOES_NOT_DESCRIBE_THE_BUILD" in out,
                f"rc={rc} names_gqa={names_gqa}",
            )
            for line in out.splitlines():
                if "GroupQueryAttention" in line or "SUBJECT SUMMARY" in line:
                    print(f"      | {line.strip()[:160]}")

            print("\narm 3: restored — the edit reverted, binary rebuilt")
            SHADER.write_text(original_shader, encoding="utf-8")
            _build()
            rc, out = _check()
            record(
                "restore-passes",
                rc == 0 and "PASS:" in out,
                out.splitlines()[0] if out else "(silent)",
            )

        print("\narm 4: stale binary — ledger rewritten, binary not rebuilt")
        import json as _json

        sys.path.insert(0, str(TOOLS))
        import importlib

        g = importlib.import_module("gen_proof_ledger")

        # A *self-consistent* variant, deliberately: header count and digest are recomputed, every
        # entry is one the real ledger already carries, so every question the old `--check` could
        # ask is answered `yes`. The only thing wrong with it is that the binary was not built
        # from it — which is exactly the state that used to read as PASS.
        lines = original_ledger.splitlines()
        hdr = _json.loads(lines[0])
        body = "\n".join(lines[1:-1]) + "\n"
        hdr["entry_count"] = len(lines) - 2
        hdr["content_fnv1a64"] = f"{g.fnv1a64(body.encode('utf-8')):016x}"
        variant = REPO / "rust" / "target" / "probe_ledger_stale.jsonl"
        variant.write_text(
            _json.dumps(hdr, separators=(",", ":"), sort_keys=True) + "\n" + body,
            encoding="utf-8",
        )
        rc, out = _check(["--out", str(variant)])
        record(
            "stale-baked-ledger-fails",
            rc != 0 and "baked into" in out,
            f"rc={rc} " + next((l.strip()[:120] for l in out.splitlines() if "baked into" in l), ""),
        )
        variant.unlink(missing_ok=True)

        print("\narm 5: no artifact — --check must not resolve anyway")
        rc, out = _check(["--lib", str(REPO / "rust" / "target" / "no-such-artifact.dll")])
        record(
            "absent-artifact-is-an-error",
            rc == 3 and "ERROR(instrument)" in out and "PASS:" not in out,
            f"rc={rc}",
        )

        # ── arms 6 and 7: the two rows the Linux reading turned on ──────────────────────
        def _variant(mutate) -> tuple[pathlib.Path, str]:
            """Write a self-consistent ledger with one entry mutated; return (path, that key)."""
            ls = original_ledger.splitlines()
            h = _json.loads(ls[0])
            picked = ""
            body_lines = []
            for line in ls[1:]:
                e = _json.loads(line)
                if not picked and e.get("source_digest") and e.get("shader_digest"):
                    picked = e["key"]
                    mutate(e)
                    line = _json.dumps(e, separators=(",", ":"), sort_keys=True)
                body_lines.append(line)
            b = "\n".join(body_lines) + "\n"
            h["entry_count"] = len(body_lines)
            h["content_fnv1a64"] = f"{g.fnv1a64(b.encode('utf-8')):016x}"
            p = REPO / "rust" / "target" / "probe_ledger_rowcase.jsonl"
            p.write_text(
                _json.dumps(h, separators=(",", ":"), sort_keys=True) + "\n" + b, encoding="utf-8"
            )
            return p, picked

        print("\narm 6: spirv moved, source LEFT CORRECT — the Linux row")
        p6, key6 = _variant(lambda e: e.__setitem__("shader_digest", "dead0000dead0000"))
        rc, out = _check(["--out", str(p6)])
        entry_lines = [l for l in out.splitlines() if key6 in l]
        record(
            "toolchain-row-is-reachable",
            "PROVEN-ELSEWHERE{toolchain}" in out
            and not any("SUBJECT-CHANGED" in l for l in entry_lines)
            and "+ 1 PROVEN-ELSEWHERE{toolchain}" in out,
            f"key={key6[:56]}",
        )
        record(
            "notes-print-on-the-failure-path",
            rc != 0 and "toolchain" in out and "SUBJECT ARITHMETIC" in out,
            f"rc={rc} (a FAILING run that still explains itself)",
        )
        for line in out.splitlines():
            if "SUBJECT ARITHMETIC" in line:
                print(f"      | {line.strip()[:190]}")
        p6.unlink(missing_ok=True)

        print("\narm 7: spirv AND source both moved — still SUBJECT-CHANGED")

        def _both(e):
            e["shader_digest"] = "dead0000dead0000"
            e["source_digest"] = "beef0000beef0000"

        p7, key7 = _variant(_both)
        rc, out = _check(["--out", str(p7)])
        record(
            "both-moved-still-fails",
            rc != 0
            and "+ 1 SUBJECT-CHANGED" in out
            and any(key7 in l and "BOTH witnesses moved" in l for l in out.splitlines()),
            f"rc={rc} key={key7[:56]}",
        )
        p7.unlink(missing_ok=True)
    finally:
        SHADER.write_text(original_shader, encoding="utf-8")
        LEDGER.write_text(original_ledger, encoding="utf-8")

    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\n{passed}/{len(results)} arms as predicted")
    if passed != len(results):
        print("FAIL: at least one arm did not behave as predicted")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
