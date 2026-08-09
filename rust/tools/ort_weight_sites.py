#!/usr/bin/env python3
"""Machine-extracted operator input sites, and the checker that holds the Rust table to them.

# Why this exists

`rust/src/ops/partition.rs` carries `ANCHOR_OP_SCHEMAS`: for every op the partitioner is willing
to call an anchor, every declared input, in index order, with an audited role. Only one role
designates a site as a weight site.

A hand-written table of upstream declarations goes stale silently, and it has already gone wrong
in this exact place: a 21-input `QMoE` was tabulated with 19 rows, and no structural check saw it,
because indices 0..=18 were contiguous and complete. Contiguity cannot see a missing tail.

So the *names and arities* are not trusted to prose. They are extracted mechanically from the
pinned upstream sources into `ort_weight_sites.json`, and the Rust table is compared against that
extract site by site, by index **and by name**.

# What this file does and does not decide

It decides **what the sites are**: how many, in what order, under what names, optional or not.

It does not decide **what they mean**. The role assignment (`SiteRole` in `partition.rs`) is an
audited semantic reading of each site's name, documented shape and documented function, and it is
stated as such rather than dressed up as a derivation. This file therefore does not attempt to
infer roles, and a reviewer comparing the two should expect exactly one thing from this file: that
the Rust table's *skeleton* matches upstream's, so the audited judgement is at least applied to
the real set of inputs.

# The two directions

* `--check` (the always-on CI direction) reads the committed JSON and the Rust source and reports
  every divergence. It needs no network, no ONNX Runtime checkout, and no particular `onnx`
  version.
* `--extract` (the maintenance direction) rebuilds the JSON. `com.microsoft` rows come from the
  ONNX Runtime `.cc` defs files at the pinned commit, whose sha256 the JSON records; `ai.onnx`
  rows come from the installed `onnx` package, whose version the JSON records.

`--verify-onnx` re-derives only the `ai.onnx` rows from `onnx.defs` and compares them with the
committed JSON. It is skipped rather than failed when the installed `onnx` version differs from
the recorded one, because a version difference is a fact about the environment, not a defect in
the table, and failing on it would train people to ignore the check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
RUST_TABLE = REPO_ROOT / "rust" / "src" / "ops" / "partition.rs"
EXTRACT_JSON = Path(__file__).resolve().with_suffix(".json")

# The ONNX Runtime commit `third_party/onnxruntime/PROVENANCE.md` pins.
PINNED_ORT_COMMIT = "da9b5e364c465de65c49d91e696cd6485270757f"
PINNED_ORT_TAG = "v1.28.0"
ORT_RAW = "https://raw.githubusercontent.com/microsoft/onnxruntime/{commit}/onnxruntime/core/graph/contrib_ops/{name}"


# ---------------------------------------------------------------------------------------------
# Extraction: ONNX Runtime contrib defs (C++)
# ---------------------------------------------------------------------------------------------

# Two macro styles coexist in the pinned sources and both must be read:
#   ONNX_MS_OPERATOR_SET_SCHEMA(Name, 1, OpSchema()....)
#   ONNX_CONTRIB_OPERATOR_SCHEMA(Name).SetDomain(kMSDomain).SinceVersion(1)....
_MS_SET_SCHEMA = re.compile(
    r"ONNX_MS_OPERATOR_SET_SCHEMA\(\s*(?P<name>\w+)\s*,\s*(?P<ver>\d+)\s*,", re.S
)
_CONTRIB_SCHEMA = re.compile(r"ONNX_CONTRIB_OPERATOR_SCHEMA\(\s*(?P<name>\w+)\s*\)")

_INPUT_CALL = re.compile(r"\.Input\(\s*(?P<index>\d+)\s*,")
_CXX_STRING = re.compile(r'"((?:[^"\\]|\\.)*)"')


def _balanced_span(text: str, open_at: int) -> tuple[int, int]:
    """The span of the parenthesised group whose `(` is at or after `open_at`.

    String literals are skipped so a `(` inside a doc string cannot unbalance the scan.
    """
    i = text.index("(", open_at)
    depth = 0
    j = i
    n = len(text)
    while j < n:
        c = text[j]
        if c == '"':
            j += 1
            while j < n:
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == '"':
                    break
                j += 1
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i, j + 1
        j += 1
    raise ValueError(f"unbalanced parentheses starting at offset {open_at}")


def _schema_body(text: str, start: int) -> str:
    """Everything from a schema macro up to the statement terminator, minus nested schemas."""
    _, end = _balanced_span(text, start)
    body = text[start:end]
    # `ONNX_CONTRIB_OPERATOR_SCHEMA(Name)` closes immediately; the chained calls follow it and run
    # to the `;` that ends the statement.
    if body.count(".Input(") == 0:
        term = text.find(";", end)
        if term != -1:
            body = text[start:term]
    return body


def _parse_input_calls(body: str) -> list[dict[str, Any]]:
    """Every `.Input(index, "name", "doc"..., "TypeStr" [, OpSchema::Optional])` in a schema body."""
    sites: list[dict[str, Any]] = []
    for m in _INPUT_CALL.finditer(body):
        start, end = _balanced_span(body, m.start())
        call = body[start:end]
        strings = _CXX_STRING.findall(call)
        if not strings:
            raise ValueError(f"`.Input` with no string arguments: {call[:120]}")
        optional = "OpSchema::Optional" in call
        variadic = "OpSchema::Variadic" in call
        sites.append(
            {
                "index": int(m.group("index")),
                "name": strings[0],
                "optional": bool(optional),
                "variadic": bool(variadic),
            }
        )
    sites.sort(key=lambda s: s["index"])
    return sites


def extract_ms_schema(source_text: str, op_name: str) -> list[dict[str, Any]]:
    """The declared inputs of one `com.microsoft` op, in index order."""
    for pattern in (_MS_SET_SCHEMA, _CONTRIB_SCHEMA):
        for m in pattern.finditer(source_text):
            if m.group("name") != op_name:
                continue
            body = _schema_body(source_text, m.start())
            sites = _parse_input_calls(body)
            if sites:
                return sites
    raise KeyError(f"no schema with `.Input` declarations found for {op_name}")


# ---------------------------------------------------------------------------------------------
# Extraction: ai.onnx standard schemas
# ---------------------------------------------------------------------------------------------


def extract_onnx_schema(op_name: str) -> tuple[int, list[dict[str, Any]]]:
    """`(since_version, sites)` for a standard-domain op, from the installed `onnx` package."""
    import onnx.defs  # imported lazily: `--check` must not need onnx installed

    schema = onnx.defs.get_schema(op_name, domain="")
    sites = []
    for i, inp in enumerate(schema.inputs):
        option = str(inp.option).rsplit(".", 1)[-1]
        sites.append(
            {
                "index": i,
                "name": inp.name,
                "optional": option == "Optional",
                "variadic": option == "Variadic",
            }
        )
    return schema.since_version, sites


# ---------------------------------------------------------------------------------------------
# The Rust table
# ---------------------------------------------------------------------------------------------

_SITE_ARRAY = re.compile(
    r"const (?P<const>\w+_SITES): &\[SchemaSite\] = &\[(?P<body>.*?)\n\];", re.S
)
_SITE_ROW = re.compile(
    r'site\(\s*(?P<index>\d+)\s*,\s*"(?P<name>[^"]*)"\s*,\s*(?P<optional>true|false)\s*,\s*(?P<role>\w+)\s*\)'
)
_SCHEMA_ROW = re.compile(
    r"AnchorOpSchema\s*\{\s*"
    r'qualified_op:\s*"(?P<op>[^"]*)"\s*,\s*'
    r'source:\s*"(?P<source>[^"]*)"\s*,\s*'
    r"declared_inputs:\s*(?P<declared>\d+)\s*,\s*"
    r"sites:\s*(?P<const>\w+)\s*,\s*"
    r"\}",
    re.S,
)

# `use SiteRole::X as ALIAS;` — resolved so the parser reads roles the way the compiler does
# rather than by guessing at the aliases.
_ROLE_ALIAS = re.compile(r"use SiteRole::(?P<role>\w+) as (?P<alias>\w+);")


def parse_rust_table(text: str) -> dict[str, dict[str, Any]]:
    """The shipped `ANCHOR_OP_SCHEMAS`, keyed by qualified op name.

    Exposed as a function (rather than inlined into the check) so a test can feed it a mutated
    copy of the source and observe the checker react.
    """
    aliases = {m.group("alias"): m.group("role") for m in _ROLE_ALIAS.finditer(text)}

    arrays: dict[str, list[dict[str, Any]]] = {}
    for m in _SITE_ARRAY.finditer(text):
        rows = []
        for r in _SITE_ROW.finditer(m.group("body")):
            role = r.group("role")
            rows.append(
                {
                    "index": int(r.group("index")),
                    "name": r.group("name"),
                    "optional": r.group("optional") == "true",
                    "role": aliases.get(role, role),
                }
            )
        arrays[m.group("const")] = rows

    table: dict[str, dict[str, Any]] = {}
    # Only the rows inside `ANCHOR_OP_SCHEMAS` count; a struct literal built elsewhere (a test
    # mutant, say) must not be mistaken for a shipped row.
    anchor_start = text.find("pub const ANCHOR_OP_SCHEMAS")
    if anchor_start == -1:
        return table
    _, anchor_end = _balanced_span(text[anchor_start:], 0)
    anchor_end = anchor_start + text[anchor_start:].index("\n];") + 3
    region = text[anchor_start:anchor_end]

    for m in _SCHEMA_ROW.finditer(region):
        const = m.group("const")
        table[m.group("op")] = {
            "source": m.group("source"),
            "declared_inputs": int(m.group("declared")),
            "sites_const": const,
            "sites": arrays.get(const, []),
        }
    return table


# ---------------------------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------------------------

DESIGNATING_ROLE = "ContractedParameter"


def check_table(
    rust_table: dict[str, dict[str, Any]], extract: dict[str, Any]
) -> list[str]:
    """Every divergence between the shipped Rust table and the machine extract.

    The single production implementation. The real check and every mutation test call *this*, so a
    test cannot pass by re-deriving a laxer rule in its own body.

    Findings name the op and, where applicable, the site index and name, so a failure says which
    row to look at.
    """
    findings: list[str] = []
    expected = extract["ops"]

    missing = sorted(set(expected) - set(rust_table))
    for op in missing:
        findings.append(f"{op}: audited upstream but absent from ANCHOR_OP_SCHEMAS")
    extra = sorted(set(rust_table) - set(expected))
    for op in extra:
        findings.append(f"{op}: present in ANCHOR_OP_SCHEMAS but not in the pinned extract")

    for op in sorted(set(expected) & set(rust_table)):
        want = expected[op]
        got = rust_table[op]
        want_sites = want["sites"]
        got_sites = got["sites"]

        # Arity, from the independently written total *and* from the row count. Both, because the
        # trailing-omission failure showed one of them alone is not enough.
        if got["declared_inputs"] != len(want_sites):
            findings.append(
                f"{op}: declared_inputs is {got['declared_inputs']}, "
                f"upstream declares {len(want_sites)} inputs"
            )
        if len(got_sites) != len(want_sites):
            findings.append(
                f"{op}: table has {len(got_sites)} site rows, "
                f"upstream declares {len(want_sites)} inputs"
            )
        if got["declared_inputs"] != len(got_sites):
            findings.append(
                f"{op}: declared_inputs {got['declared_inputs']} disagrees with its own "
                f"{len(got_sites)} site rows"
            )

        for i in range(max(len(want_sites), len(got_sites))):
            w = want_sites[i] if i < len(want_sites) else None
            g = got_sites[i] if i < len(got_sites) else None
            if w is None:
                findings.append(
                    f"{op}[{i}] '{g['name']}': tabulated but not declared upstream"
                )
                continue
            if g is None:
                findings.append(
                    f"{op}[{i}] '{w['name']}': declared upstream but missing from the table"
                )
                continue
            if g["index"] != i:
                findings.append(f"{op}[{i}]: site index is {g['index']}, expected {i}")
            if g["name"] != w["name"]:
                findings.append(
                    f"{op}[{i}]: tabulated as '{g['name']}', upstream declares '{w['name']}'"
                )
            if g["optional"] != w["optional"]:
                findings.append(
                    f"{op}[{i}] '{w['name']}': optional={g['optional']}, "
                    f"upstream says optional={w['optional']}"
                )

        # Every site carries exactly one role, and it is a role the enum knows.
        for g in got_sites:
            if g["role"] not in extract["roles"]:
                findings.append(
                    f"{op}[{g['index']}] '{g['name']}': unknown role {g['role']!r}"
                )

        # The designated set is pinned by name in the extract, so a role flip is a finding here
        # and not only in the Rust unit tests.
        want_designated = sorted(want.get("designated", []))
        got_designated = sorted(
            g["name"] for g in got_sites if g["role"] == DESIGNATING_ROLE
        )
        if want_designated != got_designated:
            findings.append(
                f"{op}: designates {got_designated}, the audited set is {want_designated}"
            )

    return findings


def load_extract(path: Path = EXTRACT_JSON) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_rust_table(path: Path = RUST_TABLE) -> dict[str, dict[str, Any]]:
    return parse_rust_table(path.read_text(encoding="utf-8"))


def verify_onnx_rows(extract: dict[str, Any]) -> tuple[str, list[str]]:
    """Re-derive the `ai.onnx` rows from the installed `onnx` and compare.

    Returns `(status, findings)`. Status is `"skipped"` when `onnx` is absent or its version
    differs from the one the extract records — an environment difference is not a table defect,
    and a check that fails on it is a check people learn to ignore.
    """
    try:
        import onnx
    except ImportError:
        return "skipped: onnx is not installed", []
    recorded = extract["provenance"]["onnx_version"]
    if onnx.__version__ != recorded:
        return (
            f"skipped: installed onnx {onnx.__version__} != recorded {recorded}",
            [],
        )

    findings: list[str] = []
    for op, want in extract["ops"].items():
        if "::" in op:
            continue
        since, sites = extract_onnx_schema(op)
        if want["source"] != f"ai.onnx::{op}-{since}":
            findings.append(
                f"{op}: extract records source {want['source']!r}, "
                f"onnx {onnx.__version__} reports since_version {since}"
            )
        for i in range(max(len(sites), len(want["sites"]))):
            a = sites[i] if i < len(sites) else None
            b = want["sites"][i] if i < len(want["sites"]) else None
            if a is None or b is None or a["name"] != b["name"] or a["optional"] != b["optional"]:
                findings.append(f"{op}[{i}]: extract {b} != onnx {a}")
    return f"verified against onnx {onnx.__version__}", findings


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------

MS_OPS = [
    ("com.microsoft::Attention", "Attention", "bert_defs.cc"),
    ("com.microsoft::MultiHeadAttention", "MultiHeadAttention", "bert_defs.cc"),
    ("com.microsoft::GroupQueryAttention", "GroupQueryAttention", "bert_defs.cc"),
    ("com.microsoft::LinearAttention", "LinearAttention", "bert_defs.cc"),
    ("com.microsoft::MatMulNBits", "MatMulNBits", "contrib_defs.cc"),
    ("com.microsoft::QMoE", "QMoE", "contrib_defs.cc"),
]
ONNX_OPS = ["MatMul", "Gemm", "Conv", "ConvTranspose", "Attention"]


def do_extract(source_dir: Path, out: Path) -> int:
    """Rebuild the JSON from pinned sources. Maintenance only; `--check` never calls this."""
    existing = load_extract(out) if out.exists() else {"ops": {}}
    import onnx

    files: dict[str, dict[str, str]] = {}
    ops: dict[str, Any] = {}

    for qualified, cxx_name, filename in MS_OPS:
        path = source_dir / filename
        text = path.read_text(encoding="utf-8", errors="replace")
        if filename not in files:
            files[filename] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": str(path.stat().st_size),
                "url": ORT_RAW.format(commit=PINNED_ORT_COMMIT, name=filename),
            }
        sites = extract_ms_schema(text, cxx_name)
        ops[qualified] = {
            "source": f"{filename}::{cxx_name}-1",
            "sites": sites,
            "designated": existing["ops"].get(qualified, {}).get("designated", []),
        }

    for op in ONNX_OPS:
        since, sites = extract_onnx_schema(op)
        ops[op] = {
            "source": f"ai.onnx::{op}-{since}",
            "sites": sites,
            "designated": existing["ops"].get(op, {}).get("designated", []),
        }

    doc = {
        "_comment": (
            "Machine-extracted input sites for every anchor-eligible op, checked against "
            "rust/src/ops/partition.rs::ANCHOR_OP_SCHEMAS by "
            "tests/ops/test_weight_site_schema.py. Regenerate with "
            "`python rust/tools/ort_weight_sites.py --extract --source-dir <ort checkout>`. "
            "The `designated` lists are the audited judgement, not an extraction: see SiteRole "
            "in partition.rs for the criteria."
        ),
        "provenance": {
            "onnxruntime_commit": PINNED_ORT_COMMIT,
            "onnxruntime_tag": PINNED_ORT_TAG,
            "onnx_version": onnx.__version__,
            "files": files,
        },
        "roles": [
            "ContractedParameter",
            "Activation",
            "QuantisationCompanion",
            "ElementwiseParameter",
            "PositionTable",
        ],
        "designating_role": DESIGNATING_ROLE,
        "ops": ops,
    }
    out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(ops)} ops)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="compare the Rust table with the extract")
    ap.add_argument("--extract", action="store_true", help="rebuild the extract from pinned sources")
    ap.add_argument("--verify-onnx", action="store_true", help="re-derive the ai.onnx rows")
    ap.add_argument("--source-dir", type=Path, help="directory holding the pinned ORT .cc files")
    args = ap.parse_args(argv)

    if args.extract:
        if not args.source_dir:
            ap.error("--extract needs --source-dir")
        return do_extract(args.source_dir, EXTRACT_JSON)

    extract = load_extract()
    rc = 0
    if args.verify_onnx:
        status, findings = verify_onnx_rows(extract)
        print(f"onnx re-derivation: {status}")
        for f in findings:
            print(f"  FAIL {f}")
        rc |= 1 if findings else 0

    if args.check or not args.verify_onnx:
        findings = check_table(load_rust_table(), extract)
        if findings:
            print(f"FAIL: {len(findings)} divergence(s) from {EXTRACT_JSON.name}")
            for f in findings:
                print(f"  {f}")
            rc |= 1
        else:
            n_ops = len(extract["ops"])
            n_sites = sum(len(o["sites"]) for o in extract["ops"].values())
            n_des = sum(len(o["designated"]) for o in extract["ops"].values())
            print(
                f"OK: {n_ops} ops, {n_sites} declared input sites, {n_des} designated, "
                f"against ONNX Runtime {PINNED_ORT_TAG} @ {PINNED_ORT_COMMIT[:12]} and "
                f"onnx {extract['provenance']['onnx_version']}"
            )
    return rc


if __name__ == "__main__":
    sys.exit(main())
