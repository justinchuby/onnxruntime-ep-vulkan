"""Every check in the weight-site anchor rule, deleted from the compiled crate, one at a time.

`partition.rs`'s falsifier tests hand `WeightSitePolicy::anchors` a *mutated policy* and watch
the verdict move. That establishes the rule reads its data. It does not establish that the
**shipped table and the shipped predicate** are what the suite is testing: a suite can be green
against a policy field that production never consults.

So this mutates the source, rebuilds, and runs the suite. Each arm removes exactly one of the
four things issue #73 added, and the suite must go red:

  M1  residency_check_removed     `anchors` stops asking the oracle — op name is enough again,
                                  which is verbatim the pre-#73 predicate
  M2  designated_check_removed    `anchors` stops consulting `designated` — a resident tensor
                                  anywhere on an anchor-capable op is enough
  M3  gqa_given_a_weight_site     `cos_cache` is designated on GroupQueryAttention, so a lone
                                  GQA node anchors on a rotary lookup table
  M4  qmoe_act_block_scales       inputs 19/20 (`fc*_act_block_scale`) are designated on QMoE,
                                  so activation quantisation parameters designate weight sites

An arm reported MISSED names a check that no test in this repository is holding: production
could lose it and CI would stay green. That is the only outcome here that is information.

R12 generalisation 4 — *for a test result, the frame is the binary that ran it*. A mutated
`.rs` restored to identical bytes can still be served from a stale `target/` if the mtime does
not move, so every arm forces a rebuild by writing the file (cargo is mtime-driven), the
restore is verified by SHA-256, and the suite is re-run after restore. If the crate is not
green again at the end, the arms above are not attributable and this exits 2.

Writes `bench/results/partition_anchor_mutations.json`.
Run: `python rust/tools/probe_partition_anchor_mutations.py`
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
CRATE = REPO / "rust"
TARGET = CRATE / "src" / "ops" / "partition.rs"
OUT = REPO / "bench" / "results" / "partition_anchor_mutations.json"

#: The tests that must go red. Named rather than "the whole suite" so a MISSED arm points at a
#: gap in *these* tests instead of being rescued by an unrelated failure elsewhere.
FILTER = "ops::partition"

#: (name, what it breaks, old, new). Each `old` must appear exactly once in the file, or the
#: mutation landed somewhere other than where this docstring says it did.
MUTATIONS: list[tuple[str, str, str, str]] = [
    (
        "M1_residency_check_removed",
        "`anchors` ignores the residency oracle: any anchor-capable op with a designated site "
        "anchors, which is the pre-#73 op-name predicate",
        """        self.anchor_capable
            && self
                .designated
                .iter()
                .any(|site| resident_initializer_at(site.index))""",
        """        let _ = &resident_initializer_at; // MUTANT M1
        self.anchor_capable && !self.designated.is_empty()""",
    ),
    (
        "M2_designated_site_check_removed",
        "`anchors` ignores which sites are designated: a resident initializer at ANY input of "
        "an anchor-capable op anchors it",
        """        self.anchor_capable
            && self
                .designated
                .iter()
                .any(|site| resident_initializer_at(site.index))""",
        """        // MUTANT M2
        self.anchor_capable && (0..64).any(|i| resident_initializer_at(i))""",
    ),
    (
        "M3_gqa_given_a_weight_site",
        "GroupQueryAttention designates `cos_cache`, so a lone GQA node anchors an island on "
        "the strength of a rotary lookup table",
        """            qualified_op: "com.microsoft::GroupQueryAttention",
            anchor_capable: true,
            designated: &[],""",
        """            qualified_op: "com.microsoft::GroupQueryAttention",
            anchor_capable: true,
            designated: &[WeightSite::at(7, "cos_cache")], // MUTANT M3""",
    ),
    (
        "M4_qmoe_activation_block_scales_designated",
        "QMoE designates inputs 19/20, so MXFP block scales of the ACTIVATIONS count as a "
        "weight payload",
        """                WeightSite::at(13, "fc3_zero_points"),
            ],""",
        """                WeightSite::at(13, "fc3_zero_points"),
                WeightSite::at(19, "fc1_act_block_scale"), // MUTANT M4
                WeightSite::at(20, "fc2_act_block_scale"), // MUTANT M4
            ],""",
    ),
]


def _cargo() -> str:
    exe = "cargo.exe" if os.name == "nt" else "cargo"
    found = shutil.which(exe)
    if found:
        return found
    candidate = Path.home() / ".cargo" / "bin" / exe
    if candidate.exists():
        return str(candidate)
    raise SystemExit(f"ERROR(instrument): {exe} not on PATH and not at {candidate}")


def _run_suite(cargo: str) -> tuple[int, str]:
    """Build and run the partition tests. Shares the crate's own target dir on purpose:
    each arm is then one incremental rebuild rather than a cold compile."""
    proc = subprocess.run(
        [cargo, "test", "--lib", FILTER],
        cwd=str(CRATE),
        capture_output=True,
        text=True,
        timeout=1800,
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _write(text: str) -> None:
    TARGET.write_text(text, encoding="utf-8", newline="\n")
    # cargo decides staleness by mtime; make sure it moves even on a coarse-grained clock.
    now = time.time() + 1
    os.utime(TARGET, (now, now))


def _failing(log: str) -> list[str]:
    return [
        ln.strip()
        for ln in log.splitlines()
        if ln.startswith("test ") and ln.rstrip().endswith("FAILED")
    ]


def main() -> int:
    cargo = _cargo()
    original = TARGET.read_text(encoding="utf-8")
    digest = hashlib.sha256(original.encode("utf-8")).hexdigest()

    rc, log = _run_suite(cargo)
    if rc != 0:
        print(
            "ERROR(instrument): the partition suite is not green before mutation; nothing "
            "below would be attributable to a mutation.\n" + log[-3000:]
        )
        return 2
    baseline = [ln for ln in log.splitlines() if ln.startswith("test result:")]

    results: list[dict] = []
    try:
        for name, broke, old, new in MUTATIONS:
            count = original.count(old)
            if count != 1:
                results.append(
                    {
                        "mutation": name,
                        "broke": broke,
                        "outcome": "ERROR(instrument)",
                        "reason": f"anchor found {count} times, expected exactly 1",
                    }
                )
                continue
            _write(original.replace(old, new))
            rc, log = _run_suite(cargo)
            compiled = "error[E" not in log and "error: could not compile" not in log
            results.append(
                {
                    "mutation": name,
                    "broke": broke,
                    "outcome": "CAUGHT" if rc != 0 else "MISSED",
                    "compiled": compiled,
                    "failing_tests": _failing(log),
                    "result_line": next(
                        (ln for ln in log.splitlines() if ln.startswith("test result:")), ""
                    ),
                }
            )
    finally:
        _write(original)

    restored = hashlib.sha256(TARGET.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
    ok_restore = restored == digest
    rc_after, log_after = _run_suite(cargo)

    doc = {
        "probe": "partition weight-site anchor rule, source-level mutations",
        "issue": 73,
        "target": str(TARGET.relative_to(REPO)).replace("\\", "/"),
        "filter": FILTER,
        "baseline": baseline,
        "mutations": results,
        "restored_identical": ok_restore,
        "green_after_restore": rc_after == 0,
        "note": (
            "A mutation reported MISSED names a check that production could lose with CI "
            "green. A mutation whose `compiled` is false was caught by the compiler rather "
            "than by a test, which is a weaker but still real screen — it is recorded, not "
            "counted as a test result."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")
    for r in results:
        print(json.dumps(r))
    print(f"\nwitness: {OUT}")

    if not ok_restore or rc_after != 0:
        print(
            "ERROR(instrument): partition.rs did not come back clean — do not read the arms "
            f"above as evidence (restored_identical={ok_restore}, green={rc_after == 0})"
        )
        print(log_after[-2000:])
        return 2
    missed = [r["mutation"] for r in results if r.get("outcome") != "CAUGHT"]
    if missed:
        print(f"PARTITION ANCHOR MUTATION PROBE: FAIL(condition) — MISSED {missed}")
        return 1
    print("PARTITION ANCHOR MUTATION PROBE: PASS — every mutation was caught")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
