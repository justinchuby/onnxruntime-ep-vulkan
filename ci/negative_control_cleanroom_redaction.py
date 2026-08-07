#!/usr/bin/env python3
"""Negative control for `python/verify_cleanroom.py`'s URL-privacy tests (issue #55).

A test suite only ever observed green is indistinguishable from a suite of assertions
that cannot fail. `tests/packaging/test_verify_cleanroom_redaction.py` is the whole
evidence for a *privacy* claim, so "it passes" is worth nothing on its own: revision 1 of
that suite passed on bytes that echoed a raw `user:pass@host` to stdout and wrote a
password fragment into a tracked artifact, and revision 2 passed on bytes that echoed a
raw schemeless credential URL *and* rewrote every Windows profile path containing an `@`.

Each arm below runs the REAL suite against a tree whose `python/verify_cleanroom.py` is
the arm's bytes, and demands the declared colour.

  REPLAYED  a rejected revision of the module, replayed from a **committed, content-
            addressed fixture** under ci/fixtures/cleanroom-redaction/. The defect
            actually happened; this is not a shape anyone invented. Each REPLAYED arm
            additionally reproduces its blocker directly, in-process, so the evidence is
            "the sentinel came out of the shipped function" / "the shipped function
            rewrote a path", not merely "some tests fail".
  PLANTED   today's module with one defect surgically reintroduced. Proves the suite is
            load-bearing on that specific mechanism, not merely present.
  INTEGRITY the fixture binding itself: a tampered fixture must be refused, or a replay
            arm could quietly become a replay of nothing.
  LANDING   the whole control, re-run in a copy of this tree with **no git history at
            all**. This is the arm that would have caught the reason PR #57 was rejected
            a second time.
  LIVE      the module in this tree, right now, which must be green.

Why fixtures instead of `git show <sha>`
----------------------------------------
Revision 2 of this control reached the rejected bytes with `git show d5bab5d:...`. That
commit is a branch-only commit of PR #57 itself. This repository allows only squash and
rebase (`allow_merge_commit: false`) and sets `delete_branch_on_merge: true`, so **no
landing method preserves it**, and `fetch-depth: 0` never fetches a deleted PR head.
Measured on a `main`-only clone with the PR applied as a squash, the step exited 1 with
`FAIL(condition=arm_did_not_fire)` and the arm count fell 12 -> 8 with every REPLAYED arm
-- the load-bearing ones -- gone: `main` red on landing and red forever after.

Nothing in this file resolves a commit. The rejected bytes are files in the tree, bound by
sha256 in the fixtures' manifest, so they replay identically on a branch, on a squash, on a
rebase, on a cherry-pick and in a fresh shallow clone. `git` is used, if present, for one
*optional* provenance note that is never an arm and never affects the exit code.

Every credential here is a synthetic sentinel. No real credential source is read, and every
`@`-bearing identity is a synthetic path component on a reserved example domain.

Run:  python ci/negative_control_cleanroom_redaction.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
MODULE = REPO / "python" / "verify_cleanroom.py"
SUITE = REPO / "tests" / "packaging" / "test_verify_cleanroom_redaction.py"
FIXTURES = HERE / "fixtures" / "cleanroom-redaction"
MANIFEST = FIXTURES / "manifest.json"

#: Set in the child process of the LANDING arm, so it runs the history-independent arms
#: and does not recurse into another copy of itself.
LANDING_ENV = "CLEANROOM_REDACTION_LANDING_CHILD"

S_USER, S_PASS, S_TOKEN = "sentinel-user", "sentinel-pass", "sentinel-token-value"
S_FRAG = "sentinel-fragment-value"
URL_SCHEMELESS = f"{S_USER}:{S_PASS}@mirror.example/pypi/simple"
URL_SCHEME_RELATIVE = f"//{S_USER}:{S_PASS}@mirror.example/pypi/simple"
URL_ABSOLUTE = f"https://{S_USER}:{S_PASS}@mirror.example/pypi/simple"
URL_QUERY = f"https://mirror.example/pypi/simple?token={S_TOKEN}"
#: R2's spellings: schemeless, credential in the query or the fragment, no `@` at all.
URL_SCHEMELESS_QUERY = f"mirror.example/pypi/simple?token={S_TOKEN}"
URL_SCHEMELESS_FRAGMENT = f"mirror.example:8443/pypi/simple#{S_FRAG}"

#: R3's shape: the default profile path on an Entra/AAD-joined Windows box.
AT_IDENTITY = "justin.chu@contoso.example"
WINDOWS_PROFILE_PATH = rf"C:\Users\{AT_IDENTITY}\AppData\Local\cleanroom"

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str, str]] = []


# ---------------------------------------------------------------------------- fixtures

def _normalised(data: bytes) -> bytes:
    """CRLF -> LF. See manifest.json's `digest.why_normalised`."""
    return data.replace(b"\r\n", b"\n")


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def fixture_source(entry: dict) -> str:
    """The fixture's text, refusing it unless its content digest matches the manifest.

    Fail-loud and by content: a fixture that drifted, was regenerated from the wrong
    revision, or was truncated by a bad merge would otherwise turn a REPLAYED arm into a
    replay of something nobody declared -- the same class of defect as a walkover PLANTED
    arm whose anchor no longer matches.
    """
    path = FIXTURES / entry["file"]
    if not path.is_file():
        raise SystemExit(
            f"NEGATIVE-CONTROL: ERROR(instrument=fixture_missing) {path} is not present. "
            f"The replay fixtures ARE the evidence; they are committed for exactly this "
            f"reason and cannot be regenerated from history that no landing preserves."
        )
    data = _normalised(path.read_bytes())
    digest = hashlib.sha256(data).hexdigest()
    if digest != entry["sha256"]:
        raise SystemExit(
            f"NEGATIVE-CONTROL: ERROR(instrument=fixture_digest_mismatch) "
            f"{entry['file']}: manifest declares sha256 {entry['sha256']}, the file on "
            f"disk hashes to {digest} ({len(data)} bytes, manifest says {entry['bytes']}). "
            f"Either the fixture was edited -- in which case it is no longer the bytes "
            f"that were rejected and the replay proves nothing -- or the manifest is "
            f"stale. Neither may pass quietly."
        )
    return data.decode("utf-8")


def provenance_note(entry: dict) -> str:
    """Optional, never an arm: if this clone happens to still contain the origin commit,
    say whether the fixture matches it. A clone that cannot resolve it -- the normal case
    after a squash landing -- is not a failure and is not reported as one."""
    origin = entry.get("origin") or {}
    ref, rel = origin.get("ref_at_authoring"), origin.get("path")
    if not ref or not rel:
        return "no origin declared"
    try:
        proc = subprocess.run(["git", "show", f"{ref}:{rel}"],
                              capture_output=True, cwd=str(REPO))
    except OSError:
        return f"git unavailable; {ref} not consulted (fixture is self-authenticating)"
    if proc.returncode != 0:
        return (f"{ref} unreachable here (expected after a squash landing); "
                f"fixture is self-authenticating")
    same = _normalised(proc.stdout) == _normalised((FIXTURES / entry["file"]).read_bytes())
    return f"{ref} still reachable and the fixture {'matches' if same else 'DIFFERS'}"


# -------------------------------------------------------------------------- arm runners

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


def arm(name: str, kind: str, source: str, expect_green: bool, note: str = "") -> None:
    code, out = run_suite(source)
    green = code == 0
    ok = green == expect_green
    results.append((name, kind, PASS if ok else FAIL,
                    note or f"suite {'GREEN' if green else 'RED'} "
                            f"(expected {'GREEN' if expect_green else 'RED'})"))
    if not ok:
        print(f"--- {name}: suite output ---\n{out}\n")


def record(name: str, kind: str, fired: bool, yes: str, no: str) -> None:
    """A direct, in-process reproduction arm. `fired` false is a FAILURE, never a skip:
    a blocker that does not reproduce means the control is vacuous."""
    results.append((name, kind, PASS if fired else FAIL,
                    yes if fired else f"did NOT reproduce -- {no}"))


def mutate(source: str, old: str, new: str, label: str) -> str:
    """Exact-substring surgery that fails loudly if the anchor has drifted.

    A mutation that silently no-ops would produce a green arm indistinguishable from a
    suite that cannot fail -- the exact failure mode this file exists to rule out.
    """
    if source.count(old) != 1:
        raise SystemExit(
            f"NEGATIVE-CONTROL: ERROR(instrument=anchor_drift) mutation {label!r} "
            f"expected exactly one occurrence of its anchor, found "
            f"{source.count(old)}. The mutation would have been a no-op and the arm a "
            f"walkover. Re-derive the anchor from the current module."
        )
    return source.replace(old, new)


# ------------------------------------------------------------------------- replay arms

def replay_r1(source: str) -> None:
    """PR #57 revision 1 (issue #55 B1/B2/B3), reproduced from the shipped bytes."""
    arm("the module as it shipped at revision 1 (B1/B2/B3)", "REPLAYED", source,
        expect_green=False)
    old = load_module(source, "_vc_rejected_r1")

    # B1: the `"://" in s` gate in _echo_cmd.
    for label, url in (("schemeless", URL_SCHEMELESS),
                       ("scheme-relative", URL_SCHEME_RELATIVE)):
        echoed = old._echo_cmd(["pip", "install", "--index-url", url])
        record(f"B1 direct: _echo_cmd leaks a {label} credential", "REPLAYED",
               S_PASS in echoed, "sentinel present in the echo", "control is vacuous")

    # B2: truncate-then-scrub. The URL is placed so that the [-1500:] slice cuts 10
    # characters into it -- inside the userinfo, so `sentinel-pass` survives in the tail
    # that revision 1 persists.
    suffix_len = 1500 + 10 - len(URL_ABSOLUTE)
    straddle = ("pad line\n" * 40) + URL_ABSOLUTE + (" " + "t" * (suffix_len - 1))
    got = old._scrub_text(straddle[-1500:], URL_ABSOLUTE)
    record("B2 direct: truncate-before-scrub keeps a password fragment", "REPLAYED",
           S_PASS in got, "sentinel present in the persisted tail", "control is vacuous")

    # B3: query credentials survive by design.
    got = old._redact_url_userinfo(URL_QUERY)
    record("B3 direct: a query credential survives redaction", "REPLAYED",
           S_TOKEN in got, "sentinel present in the redacted URL", "control is vacuous")


def replay_r2(source: str) -> None:
    """PR #57 revision 2 (R2/R3/N1), reproduced from the shipped bytes.

    Revision 2 closed B1-B4 for real and was still rejected: the echo seam did not know
    the run's own URL (R2), and the scanner guessed at `@`-bearing text (R3). Replaying it
    is what keeps revision 3's fix from being re-lost to the same blind spot -- and R3 is
    an OVER-fire, a class no arm in this file used to carry at all.
    """
    arm("the module as it shipped at revision 2 (R2/R3)", "REPLAYED", source,
        expect_green=False)
    old = load_module(source, "_vc_rejected_r2")

    # R2: schemeless spellings whose credential is in the query or the fragment. No `@`
    # anywhere in the first two, so the shape-based scanner has nothing to see and the
    # echo prints raw -- while the record seam, which knows the raw value, redacts it.
    for label, url, sentinel in (
        ("query", URL_SCHEMELESS_QUERY, S_TOKEN),
        ("fragment", URL_SCHEMELESS_FRAGMENT, S_FRAG),
        ("port+query+fragment", f"mirror.example:8443/simple?sig={S_TOKEN}#{S_FRAG}",
         S_TOKEN),
    ):
        echoed = old._echo_cmd(["pip", "install", "--index-url", url])
        leaked = sentinel in echoed
        redacted_in_record = sentinel not in old._scrub_text(url, url)
        record(f"R2 direct: _echo_cmd prints a schemeless {label} credential raw",
               "REPLAYED", leaked and redacted_in_record,
               "sentinel echoed raw while the record seam redacts the identical input",
               "control is vacuous")

    # R3: the schemeless alternative rewrites ordinary paths -- over-fire, in the artifact.
    got = old._sanitize_urls_in_text(WINDOWS_PROFILE_PATH)
    record("R3 direct: an ordinary Windows profile path is corrupted", "REPLAYED",
           got != WINDOWS_PROFILE_PATH and got.startswith("REDACTED@"),
           f"{WINDOWS_PROFILE_PATH} -> {got}: drive letter and every directory before "
           f"the '@' destroyed", "control is vacuous")

    got = old._sanitize_urls_in_text(f"contact {AT_IDENTITY} for access")
    record("R3 direct: a bare e-mail address in prose is corrupted", "REPLAYED",
           AT_IDENTITY not in got, f"-> {got}", "control is vacuous")

    # N1: not idempotent -- and `_scrub_obj` scrubs `pip_stderr` a SECOND time at the
    # write seam, so this is reachable, not theoretical.
    probe = "~&:'@-"
    once = old._sanitize_urls_in_text(probe)
    twice = old._sanitize_urls_in_text(once)
    record("N1 direct: the scrub is not idempotent on its own output", "REPLAYED",
           twice != once, f"{once!r} -> {twice!r} on a second pass", "control is vacuous")


# ------------------------------------------------------------------------- landing arm

def landing_arm() -> None:
    """Re-run this control in a copy of the tree with **no git history at all**.

    R1's acceptance condition, mechanised: "the replay arms must still fire after a real
    squash/rebase landing, with only main's history". A copy with no `.git` is strictly
    harder than a squash -- there is no history to consult, not merely a missing commit --
    so an arm that fires here fires after any landing.
    """
    with tempfile.TemporaryDirectory(prefix="cleanroom-landing-") as td:
        root = Path(td)
        for rel in (Path("python") / "verify_cleanroom.py",
                    Path("tests") / "packaging" / SUITE.name,
                    Path("ci") / Path(__file__).name):
            dst = root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / rel, dst)
        shutil.copytree(FIXTURES, root / "ci" / "fixtures" / "cleanroom-redaction")
        assert not (root / ".git").exists()

        env = dict(os.environ, **{LANDING_ENV: "1"})
        proc = subprocess.run([sys.executable, str(root / "ci" / Path(__file__).name)],
                              capture_output=True, text=True, cwd=str(root), env=env)
        ok = proc.returncode == 0
        replayed = proc.stdout.count("[REPLAYED")
        results.append((
            "the replay arms still fire in a tree with no git history", "LANDING",
            PASS if (ok and replayed >= 10) else FAIL,
            f"child exit={proc.returncode}, {replayed} REPLAYED arms fired there"))
        if not ok:
            print(f"--- landing arm: child output ---\n{proc.stdout}\n{proc.stderr}\n")


# ------------------------------------------------------------------------------- main

def main() -> int:
    manifest = load_manifest()
    entries = {e["id"]: e for e in manifest["fixtures"]}
    landing_child = os.environ.get(LANDING_ENV) == "1"

    # ----------------------------------------------------------------- INTEGRITY
    # The fixture binding must refuse bytes that are not the declared bytes, or every
    # REPLAYED arm below is a replay of whatever happens to be on disk.
    tampered = dict(entries["rejected-r1"], sha256="0" * 64)
    try:
        fixture_source(tampered)
    except SystemExit as exc:
        refused = "fixture_digest_mismatch" in str(exc)
    else:
        refused = False
    results.append(("a fixture whose content digest does not match is refused",
                    "INTEGRITY", PASS if refused else FAIL,
                    "ERROR(instrument=fixture_digest_mismatch) raised" if refused
                    else "a tampered fixture was accepted -- every replay is unbound"))

    sources = {fid: fixture_source(entry) for fid, entry in entries.items()}
    for fid, entry in entries.items():
        results.append((f"fixture {fid} matches its declared sha256", "INTEGRITY", PASS,
                        f"{entry['sha256'][:12]}..., {entry['bytes']} bytes; "
                        f"provenance: {provenance_note(entry)}"))

    live = MODULE.read_text(encoding="utf-8")

    # ---------------------------------------------------------------------- LIVE
    arm("today's python/verify_cleanroom.py", "LIVE", live, expect_green=True)

    # ------------------------------------------------------------------ REPLAYED
    replay_r1(sources["rejected-r1"])
    replay_r2(sources["rejected-r2"])

    if landing_child:
        # The child of the LANDING arm proves the history-independent arms fire. It does
        # not re-run the PLANTED surgery (identical in both trees, and it would double an
        # already slow step) and must not recurse into another landing arm.
        return verdict(partial=True)

    # ------------------------------------------------------------------- PLANTED
    arm("B1 reintroduced: _echo_cmd gates on '://' again", "PLANTED",
        mutate(live,
               """    rendered: list[str] = []
    url_valued = False
    for item in cmd:
        token = str(item)
        if url_valued:
            token, url_valued = _sanitize_url(token), False
        elif token in _URL_VALUED_FLAGS:
            url_valued = True
        elif token.partition("=")[0] in _URL_VALUED_FLAGS and "=" in token:
            name, _, value = token.partition("=")
            token = f"{name}={_sanitize_url(value)}"
        rendered.append(_scrub_text(token, raw_url))
    return " ".join(rendered)""",
               """    return " ".join(
        _scan_url_spans(str(c)) if "://" in str(c) else str(c) for c in cmd
    )""",
               "b1-echo-gate"),
        expect_green=False)

    arm("R2 reintroduced: the echo seam stops carrying the run's own URL", "PLANTED",
        mutate(live,
               '    print("$", _echo_cmd(cmd, raw_url))',
               '    print("$", _echo_cmd(cmd))',
               "r2-echo-loses-raw-url"),
        expect_green=False)

    arm("R2 reintroduced: argument-context recognition removed from the echo", "PLANTED",
        mutate(live,
               "        elif token in _URL_VALUED_FLAGS:\n"
               "            url_valued = True",
               "        elif False:  # noqa\n"
               "            url_valued = True",
               "r2-echo-loses-argument-context"),
        expect_green=False)

    arm("R3 reintroduced: the scanner guesses at '@'-bearing text again", "PLANTED",
        mutate(live,
               R'''_URL_SPAN_RE = re.compile(r"""(?:[A-Za-z][A-Za-z0-9+.\-]*:)?//[^\s"'<>]*""")''',
               R'''_URL_SPAN_RE = re.compile(
    r"""
    (?:[A-Za-z][A-Za-z0-9+.\-]*:)?//[^\s"'<>]*
  |
    (?<![\w%.+\-@])
    [\w.\-~%!$&'*+,;=]+
    (?::[^\s/@"'<>]*)?
    @
    (?:\[[0-9A-Fa-f:.]+\]|[\w.\-~%]+)
    (?::\d+)?
    (?![\w:.\-])
    (?:/[^\s"'<>]*)?
    """,
    re.VERBOSE,
)''',
               "r3-schemeless-guess"),
        expect_green=False)

    arm("raw-derived literal pass removed (a schemeless credential survives)", "PLANTED",
        mutate(live,
               "    for literal in _secret_literals(raw_url):",
               "    for literal in ():",
               "raw-literal-pass"),
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

    arm("process control trapped again: `except Exception` widened to BaseException",
        "PLANTED",
        mutate(live,
               "    except Exception as exc:  # noqa: BLE001",
               "    except BaseException as exc:  # noqa: BLE001",
               "base-exception"),
        expect_green=False)

    arm("the traceback is swallowed instead of scrubbed and recorded", "PLANTED",
        mutate(live,
               '        record["error_traceback"] = _scrub_text(traceback.format_exc(), '
               'raw_index_url)\n',
               "",
               "swallow-traceback"),
        expect_green=False)

    arm("fragment redaction removed", "PLANTED",
        mutate(live,
               '        rendered += "#" + (_REDACTED if fragment else "")',
               '        rendered += "#" + fragment',
               "fragment-passthrough"),
        expect_green=False)

    # ------------------------------------------------------------------- LANDING
    landing_arm()

    return verdict()


def verdict(partial: bool = False) -> int:
    width = max(len(n) for n, _, _, _ in results)
    print()
    for name, kind, status, note in results:
        print(f"  [{kind:<9}] {status.lower():<5} {name:<{width}}  ({note})")
    kinds = {k: sum(1 for _, kk, _, _ in results if kk == k)
             for k in ("LIVE", "REPLAYED", "PLANTED", "INTEGRITY", "LANDING")}
    print(f"\nNEGATIVE-CONTROL: {len(results)} arms -- {kinds['LIVE']} LIVE / "
          f"{kinds['REPLAYED']} REPLAYED / {kinds['PLANTED']} PLANTED / "
          f"{kinds['INTEGRITY']} INTEGRITY / {kinds['LANDING']} LANDING"
          + (" (landing child: history-independent arms only)" if partial else ""))
    print("NEGATIVE-CONTROL: the REPLAYED arms are the load-bearing ones. Every PLANTED "
          "arm is a defect written by the person who wrote the test that catches it; the "
          "REPLAYED arms are the defects as they really stood at the two rejected heads, "
          "replayed from committed content-addressed fixtures rather than from a commit "
          "no landing preserves, and rather than reconstructed from memory.")
    if all(s == PASS for _, _, s, _ in results):
        print("NEGATIVE-CONTROL: PASS -- every arm fired as declared.")
        return 0
    print("NEGATIVE-CONTROL: FAIL(condition=arm_did_not_fire) -- see above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
