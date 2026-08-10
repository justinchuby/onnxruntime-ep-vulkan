#!/usr/bin/env python3
"""``check_anchor_weight_sites`` — the anchor weight-site table is pinned to a schema, not a belief.

THE DEFECT CLASS (issue #73)
============================
``partition::is_anchor`` used to match a bare op *name*: any ``MatMul`` was an anchor, so an
activation-by-activation ``MatMul`` (both operands runtime — MiniLM's attention batched matmuls)
anchored an island it had no weight to amortise. The fix makes anchoring a property of the
*node*: a heavy-family op anchors only when a **resident constant initializer** sits at a
**schema-designated weight site**.

That fix is only as trustworthy as the weight-site table behind it. A hand-maintained table is a
belief; this checker turns it into a machine-checked fact. It cross-checks three independent
things and FAILS if any two disagree:

1. ``rust/tools/anchor_weight_sites.json`` — the pinned provenance table (source of truth).
2. **The live pinned schema** — standard-domain op input names/order are re-extracted from the
   installed ``onnx`` schema library (``onnx.defs``). The JSON's ``schema_inputs`` and the named
   weight input at ``weight_sites`` must match what the schema actually declares. This is the
   reproducible *extractor*: it reads the schema, it does not trust the table's copy of it.
3. ``rust/src/ops/partition.rs`` — the shipped ``weight_site_indices`` and ``is_heavy_family``
   tables, parsed out of the source and required to match the JSON exactly. A table edit that
   drops a row, invents a site, or reintroduces name-only anchoring is a FAIL here.

Contrib (``com.microsoft``) op schemas are not registered in ``onnx.defs``, so their weight-site
provenance is pinned to an exact ORT commit; the checker verifies that commit still matches
``third_party/onnxruntime/PROVENANCE.md`` and fails closed if it has moved without a re-pin.

STANDING GUARD — the substring trap
===================================
``GroupQueryAttention`` carries inputs literally named ``q_norm_weight`` / ``k_norm_weight`` (RMS
norm scales, not a projection weight matrix). A naive "any input with 'weight' in its name is a
weight site" heuristic would wrongly anchor every GQA node. This checker asserts GQA's designated
``weight_sites`` is empty *despite* those input names — so the substring trap cannot be
reintroduced silently.

Usage:  python ci/check_anchor_weight_sites.py            # check, exit nonzero on failure
        python ci/check_anchor_weight_sites.py --list     # print the extracted/parsed tables
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
JSON_PATH = REPO / "rust" / "tools" / "anchor_weight_sites.json"
PARTITION_RS = REPO / "rust" / "src" / "ops" / "partition.rs"
ORT_PROVENANCE = REPO / "third_party" / "onnxruntime" / "PROVENANCE.md"


class Fail(Exception):
    pass


def load_json() -> dict:
    return json.loads(JSON_PATH.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------------
# (2) Live extractor: read the pinned onnx schema library for standard-domain ops.
# --------------------------------------------------------------------------------------------
def extract_onnx_inputs(op: str) -> list[str] | None:
    """Return the input names ONNX's own schema declares for a default-domain op, or None."""
    try:
        import onnx.defs as d
    except Exception:  # pragma: no cover - onnx missing
        return None
    try:
        schema = d.get_schema(op, "")
    except Exception:
        return None
    return [i.name for i in schema.inputs]


# --------------------------------------------------------------------------------------------
# (3) Parse the shipped Rust tables out of partition.rs.
# --------------------------------------------------------------------------------------------
def parse_rust_weight_sites(text: str) -> dict[str, list[int]]:
    """Parse `pub fn weight_site_indices` match arms → {qualified_op: [indices]}."""
    m = re.search(
        r"pub fn weight_site_indices\([^)]*\)\s*->\s*&'static \[usize\]\s*\{(.*?)\n\}",
        text,
        re.DOTALL,
    )
    if not m:
        raise Fail("could not find `weight_site_indices` in partition.rs")
    body = m.group(1)
    arms: dict[str, list[int]] = {}
    for op, sites in re.findall(r'"([^"]+)"\s*=>\s*&\[([^\]]*)\]', body):
        idxs = [int(x) for x in re.findall(r"\d+", sites)]
        arms[op] = idxs
    return arms


def parse_rust_heavy_family(text: str) -> set[str]:
    """Parse the `is_heavy_family` matches! list → {qualified_op}."""
    m = re.search(
        r"pub fn is_heavy_family\([^)]*\)\s*->\s*bool\s*\{\s*matches!\((.*?)\)\s*\n\}",
        text,
        re.DOTALL,
    )
    if not m:
        raise Fail("could not find `is_heavy_family` in partition.rs")
    return set(re.findall(r'"([^"]+)"', m.group(1)))


def assert_no_name_only_is_anchor(text: str) -> None:
    """`is_anchor` must take residency facts, never an op name alone (structural #73 guarantee)."""
    sig = re.search(r"pub fn is_anchor\(([^)]*)\)", text)
    if not sig:
        raise Fail("could not find `is_anchor` in partition.rs")
    params = sig.group(1)
    if "resident_inputs" not in params or "&[bool]" not in params:
        raise Fail(
            "is_anchor must take a `resident_inputs: &[bool]` argument; a name-only signature "
            "reintroduces the issue #73 defect. Got: is_anchor(" + params + ")"
        )
    # No call site may pass a single string literal argument (name-only anchoring).
    for rs in (REPO / "rust" / "src").rglob("*.rs"):
        for mobj in re.finditer(r"is_anchor\(\s*&?\s*qual\b", rs.read_text(encoding="utf-8")):
            # `is_anchor(&qual, &resident_inputs)` is fine; `is_anchor(&qual)` is not.
            tail = rs.read_text(encoding="utf-8")[mobj.start():mobj.start() + 80]
            if re.match(r"is_anchor\(\s*&?\s*qual\s*\)", tail):
                raise Fail(f"name-only is_anchor call site in {rs}: {tail!r}")


def check_ort_commit(data: dict, problems: list[str]) -> None:
    pinned = data["provenance"]["onnxruntime"]["commit"]
    if not ORT_PROVENANCE.exists():
        problems.append(f"ORT provenance file missing: {ORT_PROVENANCE}")
        return
    prov = ORT_PROVENANCE.read_text(encoding="utf-8")
    if pinned not in prov:
        problems.append(
            f"pinned ORT commit {pinned} not found in {ORT_PROVENANCE.name}; contrib weight-site "
            "provenance is stale — re-pin against the vendored commit."
        )


def run_checks() -> list[str]:
    data = load_json()
    rs_text = PARTITION_RS.read_text(encoding="utf-8")
    rust_sites = parse_rust_weight_sites(rs_text)
    rust_heavy = parse_rust_heavy_family(rs_text)

    problems: list[str] = []
    checked = 0

    assert_no_name_only_is_anchor(rs_text)
    check_ort_commit(data, problems)

    json_ops = {o["qualified_op"]: o for o in data["ops"]}

    # Every heavy-family op in the Rust source must have a JSON row, and vice versa.
    for op in rust_heavy:
        if op not in json_ops:
            problems.append(f"heavy-family op {op!r} in partition.rs has no row in the JSON table")
    for op, row in json_ops.items():
        if row.get("heavy_family") and op not in rust_heavy:
            problems.append(f"JSON marks {op!r} heavy_family but is_heavy_family() omits it")

    for op, row in json_ops.items():
        checked += 1
        want = row["weight_sites"]

        # (3) Rust weight_site_indices must match the JSON exactly.
        got = rust_sites.get(op, [])
        if got != want:
            problems.append(
                f"weight-site mismatch for {op!r}: JSON={want} but partition.rs={got}"
            )

        # (2) Live extractor cross-check for standard-domain ops.
        if row["domain"] == "":
            live = extract_onnx_inputs(op)
            if live is None:
                problems.append(
                    f"{op!r}: could not extract schema inputs from onnx.defs (onnx installed?)"
                )
                continue
            if "schema_inputs" in row and row["schema_inputs"] != live:
                problems.append(
                    f"{op!r}: JSON schema_inputs {row['schema_inputs']} != live onnx.defs {live}"
                )
            for idx in want:
                if idx >= len(live):
                    problems.append(f"{op!r}: weight site {idx} out of range for inputs {live}")
                    continue
                if row.get("weight_input_name") and live[idx] != row["weight_input_name"]:
                    problems.append(
                        f"{op!r}: weight site {idx} is {live[idx]!r} in onnx.defs, JSON claims "
                        f"{row['weight_input_name']!r}"
                    )

    # Standing guard — the substring trap. GQA has *_weight inputs but no weight site.
    gqa = json_ops.get("com.microsoft::GroupQueryAttention")
    if gqa is None:
        problems.append("GroupQueryAttention row missing from JSON")
    else:
        names = gqa.get("schema_inputs", [])
        has_weight_named = any("weight" in n.lower() for n in names)
        if not has_weight_named:
            problems.append(
                "GQA schema_inputs no longer lists a *_weight input; the substring-trap guard is "
                "vacuous — restore the norm-scale inputs or update the guard deliberately."
            )
        if gqa["weight_sites"]:
            problems.append(
                "GroupQueryAttention MUST designate no weight site (its *_weight inputs are RMS "
                f"norm scales, not projection weights); got {gqa['weight_sites']}"
            )
        if rust_sites.get("com.microsoft::GroupQueryAttention", []):
            problems.append("partition.rs designates a weight site for GroupQueryAttention")

    print(f"[check_anchor_weight_sites] checked {checked} ops against pinned schema provenance")
    return problems


def do_list() -> None:
    data = load_json()
    rs_text = PARTITION_RS.read_text(encoding="utf-8")
    print("== JSON weight-site table ==")
    for o in data["ops"]:
        live = extract_onnx_inputs(o["qualified_op"]) if o["domain"] == "" else None
        print(
            f"  {o['qualified_op']:<38} heavy={o['heavy_family']!s:<5} "
            f"weight_sites={o['weight_sites']} live_onnx_inputs={live}"
        )
    print("== partition.rs weight_site_indices ==")
    print("  ", parse_rust_weight_sites(rs_text))
    print("== partition.rs is_heavy_family ==")
    print("  ", sorted(parse_rust_heavy_family(rs_text)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true", help="print the extracted/parsed tables")
    ap.add_argument("--check", action="store_true", help="run the checks (default)")
    args = ap.parse_args()

    if args.list:
        do_list()
        return 0

    try:
        problems = run_checks()
    except Fail as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1

    if problems:
        print(f"FAIL: {len(problems)} problem(s):", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    print("PASS: anchor weight-site table matches pinned schema provenance and the Rust source")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
