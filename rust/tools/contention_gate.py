"""A gate that can see a one-in-forty flake, and can say what failed when it does.

## Why this exists

On 2026-08-03 the coordinator merged two branches, ran `cargo test --lib`, and it went
red. He re-ran it: 521 passed, 0 failed. Five more runs: all green. He pushed on six greens
after one red -- which is exactly the reasoning an intermittent is built to defeat -- and
the failing test name was gone, because the gate's output had been truncated to its last
two lines. Two separate defects, and the second is the worse one:

  1. **A gate that runs once cannot see a rare flake.** `cargo test --lib` at a 1-in-40
     per-run failure rate passes 97.5% of merges. At this project's merge frequency that is
     a defect that ships and keeps shipping.
  2. **A gate whose output is truncated cannot say what failed, only that something did.**
     A red with no name is not a bug report; it is a rumour, and the next person re-runs it.

This tool is the mechanism for both. It is deliberately *not* "re-run the whole suite N
times": that is the least sensitive instrument available.

## Why the pool is narrow, and why that is the whole trick

Round 36 measured it: `cargo test --lib counters` failed **8 in 20** while the full
`cargo test --lib` failed ~0 in the same session. A 21-test pool aligns on a contended
global far more often than a 505-test pool does, because libtest's thread pool has fewer
other things to be doing. Round 37 measured the same asymmetry on the environment family:
`cargo test --lib backend_probe` failed **6 in 40** while the coordinator's full-suite rate
was 1 in 40. **Narrowing the filter is not a cost-saving; it is an amplifier**, and it is
what makes a tolerable number of repetitions sufficient.

## Where the pools come from

Not from a list in this file. `audit_counter_test_lock.contended_test_names()` derives them
from the tree: every `#[test]` that touches the process-global counters, the ORT logger
pointers, or a contended environment variable. Round 36's first auditor carried a
hand-written two-file list and was structurally blind to a third module with the same
defect; Round 37 found that same shape twice more (a whole-file test module, and an entire
third global family). A gate whose subject is a literal goes stale the moment someone adds
a test, and nothing tells you.

## What it does when it goes red

Keeps the **entire** captured output of the failing repetition in
`bench/results/contention_gate/`, prints the failing test names and the panic sites to
stdout, and exits 1. The `--selftest` arm replays a committed real red capture
(`contention_gate_red_fixture.txt`, from this tree at `d46327b` with the `backend_probe`
lock removed) through the extractor and requires the names back out -- so the
"a red we cannot name" failure cannot recur silently.

## A note on states that cannot occur

Every value this tool can print has been observed except one. `verdict` is RED (3 reds in
20 reps of `env:backend_probe`, 2026-08-03) and GREEN (0 in 100 across five pools, same
day). `ERROR(instrument)` is reachable and observed via `--pool` naming a pool that does
not exist. The *one* branch whose triggering condition does not occur in this tree is
"the auditor derived no contended pools" -- and it is written as a **refusal (exit 2)**
rather than a pass precisely because a gate with no subject has not observed anything; if
it ever becomes reachable, it reports rather than rounds to green.

Run:
  python rust/tools/contention_gate.py                 # default reps
  python rust/tools/contention_gate.py --reps 25       # CI
  python rust/tools/contention_gate.py --selftest      # extractor, no cargo
  python rust/tools/contention_gate.py --list          # pools only, no cargo
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
RUST = TOOLS.parent
REPO = RUST.parent
OUT_DIR = REPO / "bench" / "results" / "contention_gate"
RED_FIXTURE = TOOLS / "contention_gate_red_fixture.txt"

sys.path.insert(0, str(TOOLS))
import audit_counter_test_lock as auditor  # noqa: E402

#: Default repetitions. Detection power is printed with every run rather than assumed:
#: at the measured narrow-pool rate for the `backend_probe` family (6/40 = 0.15),
#: 20 repetitions detect with probability 1 - 0.85^20 = 0.961.
DEFAULT_REPS = 20

FAILED_LINE = re.compile(r"^test\s+(\S+)\s+\.\.\.\s+FAILED\s*$", re.M)
PANIC_LINE = re.compile(r"^thread '([^']+)' \([0-9]+\) panicked at ([^\n]+)", re.M)
PANIC_LINE_NOPID = re.compile(r"^thread '([^']+)' panicked at ([^\n]+)", re.M)
FAILURES_BLOCK = re.compile(r"^failures:\s*\n((?:\s{4}\S+\n)+)", re.M)
RESULT_LINE = re.compile(r"^test result: .*$", re.M)


def extract_failures(raw: str) -> dict:
    """Everything a human needs to act, pulled out of a full libtest capture.

    Three independent sources for the names, because they fail differently: the per-test
    `... FAILED` lines are absent when the harness aborts, the trailing `failures:` block is
    absent when output is truncated, and the panic lines carry the file:line the other two
    never do. A name found by any of them counts.
    """
    names = set(FAILED_LINE.findall(raw))
    for block in FAILURES_BLOCK.findall(raw):
        names.update(line.strip() for line in block.strip().splitlines())
    panics = PANIC_LINE.findall(raw) or PANIC_LINE_NOPID.findall(raw)
    names.update(n for n, _ in panics)
    return {
        "tests": sorted(names),
        "panics": [{"test": n, "at": at.strip()} for n, at in panics],
        "result_lines": RESULT_LINE.findall(raw),
    }


def pools():
    """{pool_name: [test fn names]} -- derived from the auditor, never written here."""
    sources = auditor.load_sources()
    consts = auditor.env_consts(sources)
    census = auditor.env_audit(sources, consts)
    _findings, contended, _sole = auditor.env_findings(census)

    out: dict[str, list] = {}
    for var, recs in sorted(contended.items()):
        short = var.replace("ONNXRUNTIME_EP_VULKAN_", "").lower()
        out[f"env:{short}"] = sorted({r["test"] for r in recs})

    fams = auditor.contended_test_names()
    counters = sorted(n for n, f in fams.items() if "counters" in f)
    if counters:
        out["counters"] = counters
    # Everything contended, in one pool: a race BETWEEN two families is invisible to every
    # single-family pool above, and this is the only arm that can see it.
    allc = sorted(set(fams) | {t for v in out.values() for t in v})
    if len(allc) > max((len(v) for v in out.values()), default=0):
        out["all-contended"] = allc
    return out


def run_pool(name: str, tests: list, reps: int, release: bool, keep_dir: Path):
    args = ["cargo", "test", "--lib"]
    if release:
        args.append("--release")
    args += ["--manifest-path", str(RUST / "Cargo.toml"), "--"] + tests
    reds = []
    t0 = time.time()
    for i in range(1, reps + 1):
        proc = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
        raw = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode != 0:
            keep_dir.mkdir(parents=True, exist_ok=True)
            path = keep_dir / f"red-{name.replace(':', '_')}-rep{i}.txt"
            # The WHOLE capture. Truncating this is the defect that made the original
            # incident unnameable; there is no tail-slicing anywhere in this file.
            path.write_text(raw, encoding="utf-8")
            info = extract_failures(raw)
            info.update({"pool": name, "rep": i, "capture": str(path), "exit_code": proc.returncode})
            reds.append(info)
            print(f"  RED  rep {i}/{reps}: {', '.join(info['tests']) or '<no name extracted>'}")
            for p in info["panics"]:
                print(f"       panic {p['test']} at {p['at']}")
            print(f"       full output kept: {path}")
        else:
            print(f"  ok   rep {i}/{reps}")
    return reds, time.time() - t0


def power(p: float, reps: int) -> float:
    return 1.0 - (1.0 - p) ** reps


def selftest() -> int:
    """The extractor, against a real red capture. No cargo, no GPU, no clock."""
    failures = []
    if not RED_FIXTURE.exists():
        print(f"SELFTEST FAIL: red fixture missing: {RED_FIXTURE}")
        return 1
    raw = RED_FIXTURE.read_text(encoding="utf-8")
    got = extract_failures(raw)
    expect = {
        "vk::barrier::tests::backend_probe_writes_sync2_token",
        "vk::barrier::tests::backend_probe_writes_legacy_token",
    }
    missing = expect - set(got["tests"])
    if missing:
        failures.append(f"names not extracted from a real red capture: {sorted(missing)}")
    if not got["panics"]:
        failures.append("no panic site extracted -- a red with no file:line is a rumour")
    if not got["result_lines"]:
        failures.append("no `test result:` line extracted")

    # The extractor must not invent names out of a green capture.
    green = "running 3 tests\ntest a ... ok\n\ntest result: ok. 3 passed; 0 failed;\n"
    if extract_failures(green)["tests"]:
        failures.append("green capture produced failure names -- the extractor cries wolf")

    # Truncation is the original defect: two lines of a red must NOT read as a green.
    tail2 = "\n".join(raw.strip().splitlines()[-2:])
    if extract_failures(tail2)["tests"]:
        failures.append("tail-2 of the red still yields names -- fixture is not representative")
    if not extract_failures(raw)["tests"]:
        failures.append("full red yields nothing")

    for line in failures:
        print(f"SELFTEST FAIL: {line}")
    if not failures:
        print(
            "selftest: 5/5 (real red named on 2 tests, panic site present, result line present, "
            "green stays silent, tail-2 of the same red is nameless — which is the incident)"
        )
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=DEFAULT_REPS)
    ap.add_argument("--release", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--pool", action="append", help="restrict to named pool(s)")
    ap.add_argument("--json", type=str, default=str(OUT_DIR / "contention_gate.json"))
    args = ap.parse_args()

    if args.selftest:
        rc = selftest()
        if rc or args.list:
            return rc

    groups = pools()
    if args.pool:
        groups = {k: v for k, v in groups.items() if k in set(args.pool)}
        missing = set(args.pool) - set(groups)
        if missing:
            print(f"ERROR(instrument): no such pool(s): {sorted(missing)}")
            return 2
    if not groups:
        # Not a pass. A gate with no subject has not observed anything.
        print("ERROR(instrument): the auditor derived no contended pools; nothing was run")
        return 2

    print(f"contention gate: {len(groups)} pool(s), reps={args.reps}")
    for p in (0.40, 0.15, 0.025):
        print(f"  detection power at a per-run rate of {p:.3f}: {power(p, args.reps):.3f}")
    print(
        "  (0.150 is the measured pre-fix rate of `cargo test --lib backend_probe`, 6/40, "
        "Windows; 0.025 is the coordinator's observed FULL-suite rate, 1/40)"
    )

    all_reds, summary = [], {}
    for name, tests in groups.items():
        print(f"\n[{name}] {len(tests)} test(s)")
        reds, secs = run_pool(name, tests, args.reps, args.release, OUT_DIR)
        all_reds += reds
        summary[name] = {
            "tests": tests,
            "reps": args.reps,
            "reds": len(reds),
            "observed_rate": len(reds) / args.reps,
            "elapsed_s": round(secs, 1),
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(
        json.dumps(
            {
                "reps": args.reps,
                "pools": summary,
                "reds": all_reds,
                "verdict": "RED" if all_reds else "GREEN",
                "extent": (
                    "Pools are derived from audit_counter_test_lock: tests touching the "
                    "process-global counters, the ORT logger pointers, or a contended "
                    "environment variable. A race reached only through a helper defined "
                    "outside the test body is outside the auditor's extent and therefore "
                    "outside this gate's pools."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    if all_reds:
        print(f"CONTENTION GATE: RED -- {len(all_reds)} failing repetition(s)")
        for r in all_reds:
            print(f"  {r['pool']} rep {r['rep']}: {', '.join(r['tests'])}  -> {r['capture']}")
        return 1
    print("CONTENTION GATE: GREEN")
    for name, s in summary.items():
        print(f"  {name}: 0/{s['reps']} red over {len(s['tests'])} test(s) ({s['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
