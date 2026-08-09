"""Falsifier battery for issue #78: does the shipped suite actually watch the pin?

For each mutant, apply one textual edit to a production file, run the shipped tests, restore.
A mutant that leaves the suite green is a load-bearing behaviour nothing watches — and the whole
point of #78 is that "the tests are green" was true while the resolver checked a filename.

    python bench/results/probe_pinned_bytes_mutations.py

Exit 0 only when the baseline is green, every mutant is killed, and the tree is restored green.
Writes `bench/results/pinned_bytes_mutations.md`. Anchors are exact source text: if a mutant
reports `anchor matched 0 times` the production line moved and the mutant must be re-aimed, which
is a NOT-APPLIED, never a kill.
"""
from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

PY = sys.executable
ROOT = Path(__file__).resolve().parents[2]
REPORT = Path(__file__).resolve().parent / "pinned_bytes_mutations.md"
PB = "bench/pinned_bytes.py"
PS = "bench/path_screen.py"
RM = "bench/real_model.py"

SUITE = ["bench/test_pinned_bytes.py", "bench/test_path_screen.py",
         "bench/test_real_model.py"]

# (id, file, old, new, what removing it means)
MUTANTS = [
    # ---- gate 1: metadata totality --------------------------------------------------
    ("meta-absent-field-ok", PB,
     '    if name not in raw:\n        return None, f"{name} is absent; a pin with a missing field pins nothing"\n    value = raw[name]\n    if isinstance(value, bool) or not isinstance(value, str):',
     '    if name not in raw:\n        return "x" * 64, ""\n    value = raw[name]\n    if isinstance(value, bool) or not isinstance(value, str):',
     "a missing text field is filled in rather than refused"),
    ("meta-wrong-type-coerced", PB,
     '    if isinstance(value, bool) or not isinstance(value, str):\n        return None, (\n            f"{name} must be a str',
     '    if False:\n        return None, (\n            f"{name} must be a str',
     "a non-str pin field is accepted"),
    ("meta-empty-ok", PB,
     '    if not value or not value.strip():\n        return None, f"{name} is empty or whitespace-only ({value!r})"',
     '    if False:\n        return None, f"{name} is empty or whitespace-only ({value!r})"',
     "an empty/whitespace pin field is accepted"),
    ("meta-bool-is-int", PB,
     '    if isinstance(value, bool):\n        return None, (\n            f"{name} is a bool',
     '    if False:\n        return None, (\n            f"{name} is a bool',
     "`pinned_bytes: true` pins one byte"),
    ("meta-int-type-coerced", PB,
     '    if not isinstance(value, int):\n        return None, (\n            f"{name} must be an int',
     '    if False:\n        return None, (\n            f"{name} must be an int',
     "a float/str size is accepted"),
    ("meta-min-size-dropped", PB,
     '    if value < minimum:\n        return None, f"{name} is {value}, which is below the minimum {minimum}"',
     '    if False:\n        return None, f"{name} is {value}, which is below the minimum {minimum}"',
     "pinned_bytes 0 or negative is accepted"),
    ("meta-repo-shape-dropped", PB,
     '    if not _REPO.match(repo) or _REPO_BAD.search(repo):',
     '    if False:',
     "a bare name that does not say which re-export is accepted"),
    ("meta-revision-mutable-ok", PB,
     '    if not _HEX40.match(revision):',
     '    if False:',
     "a branch name is accepted as a revision"),
    ("meta-sha-shape-dropped", PB,
     '    if not _HEX64.match(digest):',
     '    if False:',
     "a truncated or upper-case digest is accepted"),
    ("meta-source-state-open", PB,
     '    if source not in VERIFIED_SOURCE_STATES and source not in UNVERIFIED_SOURCE_STATES:',
     '    if False:',
     "an unrecognised source state is accepted"),
    ("meta-unknown-keys-ok", PB,
     '    unknown = sorted(set(raw) - set(_REQUIRED_FIELDS))\n    if unknown:',
     '    unknown = sorted(set(raw) - set(_REQUIRED_FIELDS))\n    if False:',
     "a typo'd field name silently unpins a property"),
    ("meta-file-shape-dropped", PB,
     '    ok, why = _relative_posix_ok(file)\n    if not ok:\n        return None, f"file {file!r}: {why}"',
     '    ok, why = _relative_posix_ok(file)\n    if False:\n        return None, f"file {file!r}: {why}"',
     "an absolute or traversing pinned file path is accepted"),
    ("meta-pin-not-a-mapping", PB,
     '    if not isinstance(raw, dict):',
     '    if False:',
     "a non-mapping pin is indexed rather than refused"),

    # ---- gate 2: the derived verdict ------------------------------------------------
    ("verdict-ignores-sidecar", PB,
     '            and isinstance(self.sidecar_sha256, str)\n',
     '',
     "an absent second witness still verifies"),
    ("verdict-ignores-sidecar-value", PB,
     '            and self.sidecar_sha256 == self.identity.sha256\n',
     '',
     "a sidecar naming different bytes still verifies"),
    ("verdict-ignores-size", PB,
     '            and self.observed_bytes == self.identity.pinned_bytes\n',
     '',
     "a file of the wrong size still verifies"),
    ("verdict-ignores-digest", PB,
     '            and self.observed_sha256 == self.identity.sha256\n',
     '',
     "a file with the wrong digest still verifies"),
    ("verdict-ignores-source-state", PB,
     '            and self.source_state in VERIFIED_SOURCE_STATES\n',
     '',
     "an offline/unpinned/download-failed state still verifies"),
    ("verdict-ignores-source-agreement", PB,
     '            and self.source_state == self.identity.source\n',
     '',
     "a source state disagreeing with the pin still verifies"),
    ("verdict-ignores-external-scan", PB,
     '            and self.external.get("scanned") is True\n',
     '',
     "an unscanned external-data section still verifies"),
    ("verdict-ignores-declared-count", PB,
     '            and len(self.external.get("files") or ()) == self.identity.declared_external_files\n',
     '',
     "a model with undeclared external files still verifies"),

    # ---- gate 3: traversal ----------------------------------------------------------
    ("traverse-drop-sparse-indices", PB,
     '        _tensor(f"{where}.values", values)\n        _tensor(f"{where}.indices", indices)',
     '        _tensor(f"{where}.values", values)',
     "an external blob declared on a sparse initializer's INDICES is invisible"),
    ("traverse-drop-sparse-initializer", PB,
     '        for i, sp in enumerate(getattr(graph, "sparse_initializer", ()) or ()):\n            _sparse(f"{where}.sparse_initializer[{i}]", sp)',
     '        for i, sp in enumerate(()):\n            _sparse(f"{where}.sparse_initializer[{i}]", sp)',
     "sparse initializers are never walked"),
    ("traverse-drop-subgraphs", PB,
     '        if _graph_is_set(attr, "g"):\n            _graph(f"{where}.{name}.g", subgraph, depth + 1)',
     '        if False:\n            _graph(f"{where}.{name}.g", subgraph, depth + 1)',
     "a weight inside an If/Loop branch is invisible"),
    ("traverse-drop-attr-tensor", PB,
     '        if _tensor_is_set(tensor):\n            _tensor(f"{where}.{name}.t", tensor)',
     '        if False:\n            _tensor(f"{where}.{name}.t", tensor)',
     "a Constant node's tensor attribute is invisible"),
    ("traverse-drop-repeated-attrs", PB,
     '        for field in repeated:\n            values = getattr(attr, field, None)',
     '        for field in ():\n            values = getattr(attr, field, None)',
     "repeated tensors/graphs/sparse_tensors attributes are invisible"),
    ("traverse-drop-functions", PB,
     '    for f, fn in enumerate(getattr(model, "functions", ()) or ()):\n        _function(f"functions[{f}]", fn, 0)',
     '    for f, fn in enumerate(()):\n        _function(f"functions[{f}]", fn, 0)',
     "a weight inside a local FunctionProto body is invisible"),
    ("traverse-drop-attribute-proto-defaults", PB,
     '        for a, attr in enumerate(getattr(fn, "attribute_proto", ()) or ()):\n            _attribute(f"{where}.attribute_proto[{a}]", attr, depth)',
     '        for a, attr in enumerate(()):\n            _attribute(f"{where}.attribute_proto[{a}]", attr, depth)',
     "a function's attribute default tensor is invisible"),
    ("traverse-drop-training-info", PB,
     '    for t, info in enumerate(getattr(model, "training_info", ()) or ()):',
     '    for t, info in enumerate(()):',
     "training_info initialization/algorithm graphs are invisible"),
    ("traverse-unbounded-depth", PB,
     '        if depth > max_depth:',
     '        if False:',
     "a hostile nesting depth is not refused"),
    ("traverse-unbounded-count", PB,
     '        if seen[0] > max_tensors:',
     '        if False:',
     "an unbounded tensor count is not refused"),
    ("traverse-malformed-skipped", PB,
     '        if not hasattr(obj, "data_location") and not hasattr(obj, "external_data"):',
     '        if False and not hasattr(obj, "external_data"):',
     "a container holding the wrong type is walked past"),

    # ---- gate 4: EXTERNAL safety ----------------------------------------------------
    ("external-allow-unsafe-location", PB,
     '    ok, why = _relative_posix_ok(location)\n    if not ok:\n        return None, f"external location {location!r} {why}"',
     '    ok, why = _relative_posix_ok(location)\n    if False:\n        return None, f"external location {location!r} {why}"',
     "absolute/URI/../drive/UNC external locations are accepted"),
    ("external-allow-reparse-escape", PB,
     '        if _has_reparse_point(candidate):',
     '        if False:',
     "a junction in a path component escapes the model root"),
    ("external-skip-root-containment", PB,
     '    try:\n        resolved.relative_to(root)\n    except ValueError:',
     '    try:\n        pass\n    except ValueError:',
     "a resolved path outside the model root is accepted"),
    ("external-allow-duplicate-keys", PB,
     '        if dupes:',
     '        if False:',
     "two readers can verify and load different bytes"),
    ("external-allow-missing-location", PB,
     '        if location is None or not isinstance(location, str) or not location.strip():',
     '        if False:',
     "a declared-but-unnamed blob is accepted"),
    ("external-allow-bad-extent", PB,
     '    if not re.fullmatch(r"[0-9]+", text):',
     '    if False:',
     "negative/hex/float offsets and lengths are coerced"),
    ("external-allow-overflow-extent", PB,
     '    if parsed > MAX_EXTENT:',
     '    if False:',
     "an out-of-range extent is treated as a large file"),
    ("external-allow-missing-file", PB,
     '    except OSError as exc:\n        raise ProvenanceError(\n            "external_missing",\n            f"{p.name} could not be stat\'d',
     '    except OSError as exc:\n        raise SystemExit(\n            "external_missing",\n            f"{p.name} could not be stat\'d',
     "declared external bytes that are not there do not fail closed"),
    ("external-drop-toctou-identity", PB,
     '    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino) and after.st_ino:',
     '    if False:',
     "the file validated is not required to be the file opened"),
    ("external-drop-open-time-reparse", PB,
     '    if _reparse_component(p, root) is not None:',
     '    if False:',
     "a symlink swapped in at open time is followed"),
    ("external-drop-regular-file", PB,
     '    if not stat.S_ISREG(after.st_mode):',
     '    if False:',
     "a device, pipe or directory is read as pinned bytes"),
    ("external-allow-short-extent", PB,
     '            if ref.offset > st.st_size or end > st.st_size:',
     '            if False:',
     "an extent running past EOF is hashed as if it were whole"),

    # ---- gate 5: public path screening ----------------------------------------------
    ("screen-drop-posix-abs", PS,
     '        for m in _POSIX_ABS.finditer(scrubbed):',
     '        for m in ():',
     "arbitrary private POSIX roots are published"),
    ("screen-drop-drive", PS,
     '        m = _WIN_DRIVE.search(scrubbed)\n        if m:',
     '        m = _WIN_DRIVE.search(scrubbed)\n        if False:',
     "any drive-absolute Windows path is published"),
    ("screen-drop-unc", PS,
     '        m = _UNC.search(scrubbed)\n        if m:',
     '        m = _UNC.search(scrubbed)\n        if False:',
     "UNC shares are published"),
    ("screen-drop-device", PS,
     '        m = _DEVICE.search(scrubbed)\n        if m:',
     '        m = _DEVICE.search(scrubbed)\n        if False:',
     r"\\?\ and \\.\ paths are published"),
    ("screen-drop-home-macro", PS,
     '        m = _HOME_MACRO.search(scrubbed)\n        if m:',
     '        m = _HOME_MACRO.search(scrubbed)\n        if False:',
     "%USERPROFILE%/$HOME/~ are published"),
    ("screen-drop-deescape", PS,
     '    if "\\\\\\\\" in text:\n        out.append(text.replace("\\\\\\\\", "\\\\"))',
     '    if False:\n        out.append(text.replace("\\\\\\\\", "\\\\"))',
     "a JSON-escaped path is published"),
    ("screen-drop-percent-decode", PS,
     '    if "%" in text:',
     '    if False:',
     "a percent-encoded path is published"),
    ("screen-drop-wide-decode", PS,
     '    if len(raw) >= 4 and raw[1::2].count(0) > len(raw) // 4:',
     '    if False:',
     "a UTF-16 path with no BOM is published"),
    ("screen-name-exemption-everywhere", PS,
     '                walk(value, f"{where}.{key}", key in GRAPH_NAME_KEYS)',
     '                walk(value, f"{where}.{key}", True)',
     "the node-name exemption applies under every key"),
    ("screen-serializer-returns-not-raises", PS,
     '    kept, why = screen_public_record(payload)\n    if kept is None:\n        raise PrivatePathLeak((why,))',
     '    kept, why = screen_public_record(payload)\n    if False:\n        raise PrivatePathLeak((why,))',
     "a leaky record is returned and published by a caller that ignores the value"),
    ("screen-file-url-allowed", PS,
     '        if _FILE_URL.search(variant):',
     '        if False:',
     "file:///C:/Users/... is treated as a public URL"),

    # ---- gate 6 / the resolver ------------------------------------------------------
    ("resolve-skips-the-check", RM,
     '        record = pb.check_pinned_bytes(',
     '        record = _UNUSED_check_pinned_bytes(',
     "the resolver never verifies at all"),
    ("resolve-absent-file-is-fine", RM,
     '    if not path.is_file():\n        raise ModelUnavailable(\n            f"{spec.key}: {spec.cache_filename} is absent',
     '    if False:\n        raise ModelUnavailable(\n            f"{spec.key}: {spec.cache_filename} is absent',
     "an absent pinned model is not reported as unavailable"),
    ("resolve-swallows-refusal", RM,
     '    except pb.ProvenanceError as exc:\n        raise ModelUnavailable(',
     '    except pb.ProvenanceError as exc:\n        raise _swallow(',
     "a refusal does not reach the caller as ModelUnavailable"),
    ("edp-partial-walk-again", RM,
     '        refs = pb.external_references(model, model_root=root)',
     '        refs = [r for r in pb.external_references(model, model_root=root)\n                if r.where.startswith("graph.initializer[")]',
     "external_data_provenance regresses to a graph.initializer-only walk"),
]


def run_suite() -> "tuple[bool, str]":
    proc = subprocess.run(
        [PY, "-m", "pytest", *SUITE, "-x", "-q", "-p", "no:cacheprovider", "--no-header"],
        cwd=ROOT, capture_output=True, text=True, timeout=900,
    )
    return proc.returncode == 0, (proc.stdout + proc.stderr)


def first_failure(output: str) -> str:
    for line in output.splitlines():
        if line.startswith("FAILED ") or line.startswith("ERROR "):
            return line.split(" - ")[0].strip()
    for line in output.splitlines():
        if "::" in line and ("FAILED" in line or "ERROR" in line):
            return line.strip()
    return "(unnamed)"


def main() -> int:
    ok, out = run_suite()
    if not ok:
        print("BASELINE IS RED - cannot run the battery")
        print(out[-4000:])
        return 2
    print(f"baseline GREEN: {out.strip().splitlines()[-1]}\n")

    survivors, rows = [], []
    for ident, rel, old, new, meaning in MUTANTS:
        p = ROOT / rel
        original = io.open(p, encoding="utf-8").read()
        if original.count(old) != 1:
            rows.append((ident, "NOT-APPLIED", f"anchor x{original.count(old)}", meaning))
            print(f"  !! {ident}: anchor matched {original.count(old)} times")
            continue
        io.open(p, "w", encoding="utf-8", newline="\n").write(original.replace(old, new, 1))
        try:
            green, out = run_suite()
        finally:
            io.open(p, "w", encoding="utf-8", newline="\n").write(original)
        if green:
            survivors.append(ident)
            rows.append((ident, "SURVIVED", "-", meaning))
            print(f"  SURVIVED  {ident}")
        else:
            conv = first_failure(out)
            rows.append((ident, "KILLED", conv, meaning))
            print(f"  killed    {ident:<40} by {conv}")

    print("\n" + "=" * 100)
    print(f"{len(rows) - len(survivors)}/{len(rows)} mutants killed")
    if survivors:
        print("SURVIVORS:", survivors)
    io.open(REPORT, "w", encoding="utf-8", newline="\n").write(
        "<!-- generated by bench/results/probe_pinned_bytes_mutations.py -->\n\n"
        "| mutant | result | convicting test | what removing it would allow |\n"
        "| --- | --- | --- | --- |\n"
        + "".join(f"| `{i}` | {s} | `{c}` | {m} |\n" for i, s, c, m in rows)
    )
    ok, out = run_suite()
    print("restored tree is", "GREEN" if ok else "RED")
    return 0 if ok and not survivors else 1


if __name__ == "__main__":
    raise SystemExit(main())
