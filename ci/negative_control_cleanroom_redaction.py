#!/usr/bin/env python3
"""Negative control for `python/verify_cleanroom.py`'s URL-privacy tests (issue #55).

A test suite only ever observed green is indistinguishable from a suite of assertions
that cannot fail. `tests/packaging/test_verify_cleanroom_redaction.py` is the whole
evidence for a *privacy* claim, so "it passes" is worth nothing on its own: the previous
version of that suite passed on bytes that echoed a raw `user:pass@host` to stdout and
wrote a password fragment into a tracked artifact.

Each arm below runs the REAL suite against a tree whose `python/verify_cleanroom.py` is
the arm's bytes, and demands the declared colour.

  REPLAYED  the module exactly as it shipped at d5bab5d — the commit PR #57 was rejected
            at. The defect actually happened; this is not a shape I invented. This arm
            additionally reproduces each blocker directly, in-process, so the evidence is
            "the sentinel came out of the shipped function", not merely "some tests fail".
  PLANTED   today's module with one defect surgically reintroduced. Proves the suite is
            load-bearing on that specific mechanism, not merely present.
  LIVE      the module in this tree, right now, which must be green.

Every credential here is a synthetic sentinel. No real credential source is read.

Run:  python ci/negative_control_cleanroom_redaction.py
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
MODULE = REPO / "python" / "verify_cleanroom.py"
SUITE = REPO / "tests" / "packaging" / "test_verify_cleanroom_redaction.py"

#: The head Morpheus rejected. Its `_echo_cmd` gates on `"://" in s` (B1) and its
#: `main()` calls `_scrub_text(install.stderr[-1500:], url)` (B2).
REJECTED_REF = "d5bab5d"

S_USER, S_PASS, S_TOKEN = "sentinel-user", "sentinel-pass", "sentinel-token-value"
URL_SCHEMELESS = f"{S_USER}:{S_PASS}@mirror.example/pypi/simple"
URL_SCHEME_RELATIVE = f"//{S_USER}:{S_PASS}@mirror.example/pypi/simple"
URL_ABSOLUTE = f"https://{S_USER}:{S_PASS}@mirror.example/pypi/simple"
URL_QUERY = f"https://mirror.example/pypi/simple?token={S_TOKEN}"

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str, str]] = []


def run_suite(module_source: str) -> tuple[int, str]:
    """Run the real suite against a tree carrying *module_source* as the module."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "python").mkdir()
        (root / "python" / "verify_cleanroom.py").write_text(module_source,
                                                             encoding="utf-8")
        pkg = root / "tests" / "packaging"
        pkg.mkdir(parents=True)
        shutil.copy(SUITE, pkg / SUITE.name)
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", str(pkg / SUITE.name), "-q",
             "-p", "no:randomly", "--no-header", "-x", "--tb=no"],
            capture_output=True, text=True, cwd=str(root),
        )
        return proc.returncode, proc.stdout + proc.stderr


def load_module(module_source: str, name: str):
    """Import *module_source* in this process so a blocker can be reproduced directly."""
    tmp = Path(tempfile.mkdtemp())
    path = tmp / f"{name}.py"
    path.write_text(module_source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def historical_module(ref: str) -> str | None:
    proc = subprocess.run(["git", "show", f"{ref}:python/verify_cleanroom.py"],
                          capture_output=True, text=True, cwd=str(REPO))
    return proc.stdout if proc.returncode == 0 else None


def arm(name: str, kind: str, source: str, expect_green: bool, note: str = "") -> None:
    code, out = run_suite(source)
    green = code == 0
    ok = green == expect_green
    results.append((name, kind, PASS if ok else FAIL,
                    note or f"suite {'GREEN' if green else 'RED'} "
                            f"(expected {'GREEN' if expect_green else 'RED'})"))
    if not ok:
        print(f"--- {name}: suite output ---\n{out}\n")


def mutate(source: str, old: str, new: str, label: str) -> str:
    """Exact-substring surgery that fails loudly if the anchor has drifted.

    A mutation that silently no-ops would produce a green arm indistinguishable from a
    suite that cannot fail — the exact failure mode this file exists to rule out.
    """
    if source.count(old) != 1:
        raise SystemExit(
            f"NEGATIVE-CONTROL: ERROR(instrument=anchor_drift) mutation {label!r} "
            f"expected exactly one occurrence of its anchor, found "
            f"{source.count(old)}. The mutation would have been a no-op and the arm a "
            f"walkover. Re-derive the anchor from the current module."
        )
    return source.replace(old, new)


def main() -> int:
    live = MODULE.read_text(encoding="utf-8")

    # ---------------------------------------------------------------- LIVE
    arm("today's python/verify_cleanroom.py", "LIVE", live, expect_green=True)

    # ------------------------------------------------------------ REPLAYED
    rejected = historical_module(REJECTED_REF)
    if rejected is None:
        results.append((f"the module as it shipped at {REJECTED_REF}", "REPLAYED",
                        "ERRO", f"ref {REJECTED_REF} unreachable in this clone"))
    else:
        arm(f"the module as it shipped at {REJECTED_REF}", "REPLAYED", rejected,
            expect_green=False)

        old = load_module(rejected, "_vc_rejected")
        # B1, reproduced from the shipped bytes: the `"://" in s` gate in _echo_cmd.
        for label, url in (("schemeless", URL_SCHEMELESS),
                           ("scheme-relative", URL_SCHEME_RELATIVE)):
            echoed = old._echo_cmd(["pip", "install", "--index-url", url])
            leaked = S_PASS in echoed
            results.append((f"B1 direct: _echo_cmd leaks a {label} credential",
                            "REPLAYED", PASS if leaked else FAIL,
                            "sentinel present in the echo" if leaked
                            else "did NOT reproduce — control is vacuous"))
        # B2, reproduced from the shipped bytes: truncate-then-scrub. The URL is placed
        # so that the [-1500:] slice cuts 10 characters into it — inside the userinfo,
        # so `sentinel-pass` survives in the tail that d5bab5d persists.
        suffix_len = 1500 + 10 - len(URL_ABSOLUTE)
        straddle = ("pad line\n" * 40) + URL_ABSOLUTE + (" " + "t" * (suffix_len - 1))
        got = old._scrub_text(straddle[-1500:], URL_ABSOLUTE)
        leaked = S_PASS in got
        results.append(("B2 direct: truncate-before-scrub keeps a password fragment",
                        "REPLAYED", PASS if leaked else FAIL,
                        "sentinel present in the persisted tail" if leaked
                        else "did NOT reproduce — control is vacuous"))
        # B3, reproduced from the shipped bytes: query credentials survive by design.
        got = old._redact_url_userinfo(URL_QUERY)
        leaked = S_TOKEN in got
        results.append(("B3 direct: a query credential survives redaction",
                        "REPLAYED", PASS if leaked else FAIL,
                        "sentinel present in the redacted URL" if leaked
                        else "did NOT reproduce — control is vacuous"))

    # ------------------------------------------------------------- PLANTED
    arm("B1 reintroduced: _echo_cmd gates on '://' again", "PLANTED",
        mutate(live,
               'return " ".join(_sanitize_urls_in_text(str(c)) for c in cmd)',
               'return " ".join(_sanitize_urls_in_text(str(c)) if "://" in str(c)\n'
               '                    else str(c) for c in cmd)',
               "b1-echo-gate"),
        expect_green=False)

    arm("B2 reintroduced: truncate before scrubbing", "PLANTED",
        mutate(live,
               "    scrubbed = _scrub_text(text, raw_url)\n"
               "    if limit <= 0:\n"
               "        return \"\"",
               "    scrubbed = _scrub_text(text[-limit:] if limit else text, raw_url)\n"
               "    if limit <= 0:\n"
               "        return \"\"",
               "b2-truncate-first"),
        expect_green=False)

    arm("B3 weakened: query redaction becomes a denylist of known names", "PLANTED",
        mutate(live,
               '        elif "=" in segment:\n'
               '            name, _, _value = segment.partition("=")\n'
               '            out.append(f"{name}={_REDACTED}")',
               '        elif "=" in segment:\n'
               '            name, _, _value = segment.partition("=")\n'
               '            out.append(f"{name}={_REDACTED}"\n'
               '                       if name.lower() in ("token", "access_token",\n'
               '                                          "api_key", "key", "password")\n'
               '                       else segment)',
               "b3-denylist"),
        expect_green=False)

    arm("record-wide scrub removed from the write seam", "PLANTED",
        mutate(live,
               "    scrubbed = _scrub_obj(record, raw_url)",
               "    scrubbed = dict(record)",
               "write-seam-scrub"),
        expect_green=False)

    arm("exception re-chained: original message reachable via __context__", "PLANTED",
        mutate(live,
               '        fatal = scrubbed["error"]',
               '        raise SystemExit(scrubbed["error"]) from None',
               "exception-chaining"),
        expect_green=False)

    arm("fragment redaction removed", "PLANTED",
        mutate(live,
               '        rendered += "#" + (_REDACTED if fragment else "")',
               '        rendered += "#" + fragment',
               "fragment-passthrough"),
        expect_green=False)

    # ------------------------------------------------------------- verdict
    width = max(len(n) for n, _, _, _ in results)
    print()
    for name, kind, status, note in results:
        print(f"  [{kind:<8}] {status.lower():<5} {name:<{width}}  ({note})")
    kinds = {k: sum(1 for _, kk, _, _ in results if kk == k)
             for k in ("LIVE", "REPLAYED", "PLANTED")}
    print(f"\nNEGATIVE-CONTROL: {kinds['LIVE']} LIVE / {kinds['REPLAYED']} REPLAYED / "
          f"{kinds['PLANTED']} PLANTED.")
    print("NEGATIVE-CONTROL: the REPLAYED arms are the load-bearing ones. Every PLANTED "
          "arm is a defect written by the person who wrote the test that catches it; the "
          f"REPLAYED arms are the defect as it really stood at {REJECTED_REF}, retrieved "
          "with `git show` rather than reconstructed from memory.")
    if all(s == PASS for _, _, s, _ in results):
        print("NEGATIVE-CONTROL: PASS — every arm fired as declared.")
        return 0
    print("NEGATIVE-CONTROL: FAIL(condition=arm_did_not_fire) — see above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
