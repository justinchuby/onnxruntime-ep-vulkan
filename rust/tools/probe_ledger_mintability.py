#!/usr/bin/env python
"""Does `--check` notice a key that **cannot be minted**? Owner: Tank.

WHY THIS EXISTS
---------------
`gen_proof_ledger.py --check` asks, of every entry that is present, whether it agrees with the
build; `check_no_proof_went_missing` asks about the empty space in the file. Neither can ask
whether a key **could ever exist**. Measured 2026-08-04: all 43 retired keys passed `--check`
cleanly, and the check has no way to separate *"retired on purpose because the form stopped
existing"* from *"never mintable on this build"*. Those have opposite repairs — one is
bookkeeping, the other means a module declares a SPIR-V capability
`ops/common/variants.rs::ENGINE_ENABLED_CAPABILITIES` does not carry, so no pipeline can be
created from it on any device we run on and a proof run over that key reports *"no unlockable
keys"*.

That is the same family as a screen that is clean because it is not looking: the `source_digest`
revert on 115 of 121 entries sat behind a key census reading `0 VANISHED`, a loss invariant
reading `0 missing`, and a `--check` PASS, because the loss was a *field inside a surviving
entry*. A check that has no question about a property reports nothing about it and prints PASS.

WHAT IS BEING FALSIFIED
-----------------------
`OrtEpVulkanGetFormMintability` + `gen_proof_ledger.check_mintability`. **The load-bearing arms
are 2 and 5** — a green arm on the shipping tree is produced by the tree, not by the check, and
a screen whose red state has never been shown is a screen nobody has seen work.

ARMS (predictions written before the run)
-----------------------------------------
1. shipping build, real unmintable key      -> the artifact says mintable=no, and says WHY
   (`ai.onnx::Cast/6+/i64>i32/ew_cast_i64_to_i32/...`, the key that produced the RAI-012-class
   decline text on Phi-3.5) and a sibling f32 key on the same op says mintable=yes.
   Non-vacuity by measurement: both answers reachable from one call.
2. that key added to the ledger              -> `--check` goes RED, naming the key and the module,
   and **that failure is the ONLY one** — every other check passes on the same file, so the red
   is attributable to this arm and not to the mutation's side effects.
3. current tree                              -> PASS, 0 unmintable ledger keys (not noisy)
4. a build with no such export               -> ERROR(instrument), never PASS
5. A SHADERLESS BUILD of this same tree      -> every declared-stem ledger key reads mintable=no.
   This is the positive control at the artifact level: `form_provable_from(true, false, false)`
   is asserted in-process by a unit test, but no test in this process can make the build
   shaderless. Arm 5 asks a binary that actually is.

NO CLOCK. Set/count comparisons only.

USAGE
    python rust/tools/probe_ledger_mintability.py
      [--shaderless-lib PATH]   # arm 5's artifact; skipped-and-reported if absent
"""

from __future__ import annotations

import argparse
import io
import json
import os
import pathlib
import sys
from contextlib import redirect_stdout

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import gen_proof_ledger as gpl  # noqa: E402

LEDGER = REPO / "evidence" / "proof_ledger.jsonl"

# A real key, not a synthetic one: every `_i64` module declares `OpCapability Int64` and the
# engine's `VkDeviceCreateInfo` enables no such feature.
UNMINTABLE_KEY = "ai.onnx::Cast/6+/i64>i32/ew_cast_i64_to_i32/static/n1"
MINTABLE_SIBLING = "ai.onnx::Cast/6+/f32>i32/ew_cast_f32_to_i32/static/n1"


def _entries(text: str) -> tuple[dict, list[dict]]:
    lines = [l for l in (x.strip() for x in text.splitlines()) if l and not l.startswith("#")]
    return json.loads(lines[0]), [json.loads(l) for l in lines[1:]]


def _write_ledger(dest: pathlib.Path, header: dict, entries: list[dict]) -> None:
    """Re-emit a ledger with its header arithmetic recomputed.

    The digest and the count are recomputed rather than patched, because a mutation that trips
    the self-consistency half never reaches the build half and arm 2 would then be measuring the
    header check.
    """
    body = "".join(json.dumps(e, sort_keys=True) + "\n" for e in entries)
    header = dict(header)
    header["entry_count"] = len(entries)
    header["content_fnv1a64"] = f"{gpl.fnv1a64(body.encode('utf-8')):016x}"
    dest.write_text(json.dumps(header, sort_keys=True) + "\n" + body, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shaderless-lib", default=os.environ.get("VULKAN_EP_SHADERLESS_LIB", ""))
    args = ap.parse_args()

    lib = gpl._find_lib(os.environ.get("ONNXRUNTIME_VULKAN_EP_LIB", ""))
    if lib is None:
        print("ERROR(instrument): no built EP found; set ONNXRUNTIME_VULKAN_EP_LIB")
        return 3

    scratch = REPO / "bench" / "results" / "_probe_ledger_mintability"
    scratch.mkdir(parents=True, exist_ok=True)
    results: list[tuple[str, bool, str]] = []

    # ---- Arm 1: the artifact answers, and both answers are reachable ---------------------
    report, err = gpl._mintability(lib, [UNMINTABLE_KEY, MINTABLE_SIBLING])
    if err:
        results.append(("1 the artifact answers mintability", False, err))
    else:
        bad, good = report[UNMINTABLE_KEY], report[MINTABLE_SIBLING]
        hit = (
            bad["mintable"] is False
            and bad.get("declared") == "yes"
            and bad.get("loadable") == "no"
            and good["mintable"] is True
        )
        results.append((
            "1 the artifact says mintable=no AND says why (non-vacuous: sibling says yes)",
            hit,
            f"{UNMINTABLE_KEY.split('/')[3]}: mintable={bad.get('mintable')} "
            f"declared={bad.get('declared')} generated={bad.get('generated')} "
            f"loadable={bad.get('loadable')}; sibling {MINTABLE_SIBLING.split('/')[3]}: "
            f"mintable={good.get('mintable')}",
        ))

    # ---- Arm 2: the check goes red, and the red is attributable --------------------------
    header, entries = _entries(LEDGER.read_text(encoding="utf-8"))
    donor = dict(entries[0])
    donor["key"] = UNMINTABLE_KEY
    mutated = scratch / "ledger_with_unmintable_key.jsonl"
    _write_ledger(mutated, header, entries + [donor])

    retired_map, _ = gpl._retired_keys()
    keys_with = {e["key"] for e in entries} | {UNMINTABLE_KEY}
    fails, notes, merr = gpl.check_mintability(lib, keys_with, retired_map)
    hit = not merr and len(fails) == 1 and UNMINTABLE_KEY in fails[0] and "ew_cast_i64_to_i32" in fails[0]
    results.append((
        "2a the mintability arm names exactly the one unmintable key",
        hit,
        merr or (fails[0][:150] if fails else "NO FAILURE REPORTED — the screen is not looking"),
    ))

    buf = io.StringIO()
    with redirect_stdout(buf):
        # `expect_rebuild=True` because a synthetic ledger differs from the one baked into the
        # artifact **by construction** — that is what writing the file means, and it is the same
        # allowance a fresh generation run gets. Without it the baked-vs-disk failure fires and
        # arm 2b would be measuring that check instead of this one.
        rc = gpl.check_ledger(mutated, lib, expect_rebuild=True)
    out = buf.getvalue()
    # Every other check has to pass on this file, or the red is the mutation's and not the arm's.
    other_reds = [
        l.strip()
        for l in out.splitlines()
        if l.strip().startswith("- ") and "NOT MINTABLE" not in l
    ]
    hit = rc == 1 and "NOT MINTABLE" in out and not other_reds
    results.append((
        "2b `--check` exits 1 on that ledger and the ONLY failure is the mintability one",
        hit,
        f"rc={rc}, {len(other_reds)} unrelated failure(s)"
        + (f": {other_reds[0][:110]}" if other_reds else ""),
    ))

    # ---- Arm 3: the shipping tree is clean ----------------------------------------------
    fails, notes, merr = gpl.check_mintability(
        lib, {e["key"] for e in entries}, retired_map
    )
    arith = next((n for n in notes if n.startswith("mintability:")), "")
    results.append((
        "3 the shipping ledger has no unmintable key (the arm is not noisy)",
        not merr and not fails,
        merr or (arith or "no arithmetic line printed"),
    ))

    # ---- Arm 4: a build that cannot be asked must not answer PASS ------------------------
    # Any pre-2026-08-04 artifact lacks the export. A missing screen is ERROR(instrument), which
    # is the rule `--check` already applies to a missing artifact.
    older = [
        p
        for p in sorted(pathlib.Path(REPO).parent.glob("ep-vulkan-*/rust/target/release/onnxruntime_vulkan_ep.dll"))
        if p.resolve() != lib.resolve()
    ]
    hit, detail = False, "ERROR(instrument): no second artifact available to ask"
    for cand in older:
        _, e = gpl._mintability(cand, [UNMINTABLE_KEY])
        if e and "OrtEpVulkanGetFormMintability" in e:
            hit, detail = True, f"{cand.parents[3].name}: {e[:110]}"
            break
    results.append(("4 a build with no export is ERROR(instrument), not PASS", hit, detail))

    # ---- Arm 5: THE POSITIVE CONTROL — a build that really has no SPIR-V ------------------
    if args.shaderless_lib and pathlib.Path(args.shaderless_lib).is_file():
        sl = pathlib.Path(args.shaderless_lib)
        declared_keys = sorted(
            {e["key"] for e in entries if gpl_stem(e["key"]) not in ("", "metadata")}
        )
        rep, e = gpl._mintability(sl, declared_keys)
        if e:
            results.append(("5 a shaderless build mints nothing", False, e))
        else:
            unmintable = [k for k in declared_keys if not rep[k]["mintable"]]
            hit = len(declared_keys) > 0 and len(unmintable) == len(declared_keys)
            results.append((
                "5 POSITIVE CONTROL: on a shaderless build every declared-stem key is unmintable",
                hit,
                f"{len(unmintable)}/{len(declared_keys)} declared-stem ledger key(s) read "
                f"mintable=no against {sl.parents[3].name}",
            ))
    else:
        results.append((
            "5 POSITIVE CONTROL: shaderless build",
            False,
            "SKIPPED — no --shaderless-lib. An unrun control is not a passed one.",
        ))

    print(f"probe_ledger_mintability — subject {lib}")
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        print(f"         {detail}")
    bad = [n for n, ok, _ in results if not ok]
    print(f"ARMS: {len(results) - len(bad)}/{len(results)} pass")
    return 0 if not bad else 1


def gpl_stem(key: str) -> str:
    """The variant component's module stem, mirroring `ProofKey::variant_stem`."""
    parts = key.split("/")
    if len(parts) < 5:
        return ""
    return parts[3].split("#")[0].split("@")[0]


if __name__ == "__main__":
    raise SystemExit(main())
