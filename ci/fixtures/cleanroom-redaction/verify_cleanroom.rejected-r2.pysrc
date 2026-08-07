"""End-to-end demonstration: a user who has never built this repository gets a session.

    python python/verify_cleanroom.py

Creates a **fresh virtual environment outside the repository**, installs only the wheel and
its dependencies into it, and then — in that interpreter, with the repository unreachable —
imports the package, registers the EP, runs a trivial model, and asserts the EP was
selected. Writes ``bench/results/cleanroom_install_dev0.json``.

Why outside the repository
--------------------------
:func:`onnxruntime_ep_vulkan.library_path` falls back to a source checkout's
``rust/target/release`` when it finds a ``rust/`` directory above itself. A venv created
inside this tree would have that fallback available, so a passing run would not
discriminate between "the wheel carried the artifact" and "the package found the build I
already had". The venv therefore goes in a sibling directory of the repository, and the
run additionally asserts that the resolved library path lies **inside the venv's
site-packages** — a positional check, not a trust exercise.

What this demonstrates and what it does not
-------------------------------------------
It demonstrates: from a wheel and nothing else, on this platform, `import` → `register` →
`InferenceSession` → the Vulkan EP is selected → correct output. That is the positive state
of the consumption path, observed rather than reasoned about.

It does not demonstrate: that this holds on any other OS, driver or architecture; that
`pip install onnxruntime-ep-vulkan` from an index works (nothing is published, deliberately
— no release process is in scope); or that a wheel built here runs on a machine that never
had the Vulkan SDK, since this box has it and a Vulkan *loader* is a runtime requirement
the wheel does not and cannot carry.

Package-index URL privacy (issue #55)
-------------------------------------
``--index-url`` may point at a private/authenticated mirror, so its value is treated as a
**secret-bearing string** everywhere except pip's own argv. The policy is deliberately
*redact-by-default* rather than a denylist of known-sensitive parameter names, because a
denylist is only ever as good as the last mirror vendor's naming choice:

============================  ==============================================================
URL component                 Policy
============================  ==============================================================
scheme, host, port, path      **preserved** — this is the provenance an operator needs
userinfo (``user:pass@``)     **redacted** wholesale to ``REDACTED@`` (also ``user@`` alone)
every query *value*           **redacted** to ``REDACTED``; the *name* and the number of
                              parameters survive, so ``?token=x&sig=y`` renders as
                              ``?token=REDACTED&sig=REDACTED``
a query segment with no ``=`` **redacted** entirely — an ``=``-less segment is an opaque
                              token, and nothing distinguishes a flag from a credential
fragment (``#...``)           **redacted** to ``#REDACTED`` when non-empty
============================  ==============================================================

This covers absolute (``https://u:p@h/x``), scheme-relative (``//u:p@h/x``) and schemeless
(``u:p@h/x``) spellings, percent-encoded userinfo, IPv6 literals with ports, duplicate and
blank query parameters, several URLs on one line, and text where a URL straddles the point
at which a captured stream is truncated (the scrub always runs over the **whole** text and
the already-sanitised result is what gets truncated).

Known and accepted limitations, stated so nobody has to infer them:

* **A secret inside the URL *path* is not redacted.** Some mirrors sign requests by
  embedding a token in a path segment (``/t/<token>/simple``). The path is the last piece
  of provenance left after userinfo, query and fragment are gone, so it is kept. Put
  credentials in userinfo or the query, not the path, if you need this tool to hide them.
* **The redaction covers only what *this* module echoes, persists or raises.** pip's own
  logs (``pip.log``, ``~/.cache/pip``), the OS process table, and any proxy access log are
  outside its reach. pip is invoked with ``shell=False`` and a real argv, so no shell
  history or shell expansion sees the value.
* ``pip freeze`` output recorded under ``installed`` passes through the same scrub, so a
  local wheel's ``file:///...#sha256=<digest>`` fragment renders as ``#REDACTED``. The
  wheel's digest is recorded verbatim and independently as ``wheel_sha256``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

PY_DIR = Path(__file__).resolve().parent
REPO = PY_DIR.parent
DEFAULT_ENV = REPO.parent / "onnxruntime-ep-vulkan-cleanroom"

# Issue #40: public PyPI (files.pythonhosted.org) is blocked in the sandboxes this tool
# is actually run from during EP development, so an unqualified `pip install` here fails
# with an SSL handshake error before a single assertion runs -- not a wheel or EP defect,
# a package-index reachability defect. The approved internal proxy mirrors PyPI and is
# used unless the caller overrides it (a machine that can reach public PyPI directly, or
# an internal mirror change, both stay one flag/env-var away, never a code edit).
DEFAULT_INDEX_URL = os.environ.get(
    "ONNXRUNTIME_EP_VULKAN_PYPI_INDEX_URL",
    "https://packagefeedproxy.microsoft.io/pypi/simple",
)

# Issue #55: a custom --index-url may embed userinfo (a token, or user:pass@), a query
# credential (?token=, ?sig=, ?<whatever the vendor called it this year>) or a fragment
# for a private/authenticated mirror. That is exactly the shape a shell history, a CI
# log, or this tool's own persisted record must never carry. _REDACTED replaces every
# such component everywhere this module *echoes, persists or raises* an index URL; the
# one place the real, unredacted value must still reach is the argv handed straight to
# `subprocess.run` for pip itself (never through a shell, so there is no second place
# credentials could leak via shell history/expansion). See the module docstring for the
# component-by-component policy and its accepted limitations.
_REDACTED = "REDACTED"
_UNPARSEABLE = f"<{_REDACTED}-unparseable-index-url>"

# How much of a captured stream is kept in the record. The scrub ALWAYS runs over the
# whole text first and this slice is applied to the sanitised result -- truncating first
# would let a credential that straddles the boundary survive as an unmatched fragment
# (issue #55 blocker B2).
_TAIL_CHARS = 1500

# ``scheme://`` or a bare scheme-relative ``//`` -- the RFC 3986 marker that an authority
# section follows.
_AUTHORITY_MARKER_RE = re.compile(r"^(?:[A-Za-z][A-Za-z0-9+.\-]*:)?//")

# URL-shaped spans inside arbitrary text (an argv token, pip's stderr, an exception
# message). Two alternatives, tried in order at every position:
#
#   1. absolute or scheme-relative: an explicit ``//`` authority marker. This is the
#      unambiguous case and is allowed to run to the next whitespace/quote/angle bracket.
#   2. schemeless authority: ``user[:pass]@host[:port][/path...]``. Gated on a literal
#      ``@`` so ordinary arguments (``onnx``, ``--disable-pip-version-check``,
#      ``C:\dist\pkg.whl``) can never match, and the host must be followed by something
#      that is not more host-ish text, so a PEP 508 direct reference (``pkg@file:///...``)
#      is left to alternative 1 rather than being mangled here.
_URL_SPAN_RE = re.compile(
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
)


def _sanitize_query(query: str) -> str:
    """Redact every query *value*, keeping every parameter *name* and the parameter count.

    Redact-by-default rather than a denylist of ``token``/``api_key``/``sig``/...: a
    denylist is bypassed the moment a mirror invents a new name for the same secret, and
    the names themselves are the part with provenance value ("this URL was signed", "this
    URL carried two credentials"). Percent-encoding is left alone -- names are copied
    verbatim rather than decoded and re-encoded, so no round-trip can invent bytes.

    A segment with no ``=`` is an opaque token with no name to preserve, so the whole
    segment is replaced. Blank values (``a=``) still render as ``a=REDACTED``: whether a
    credential was empty is not information worth leaking the distinction for. Empty
    segments (``a=1&&b=2``) are preserved as empty so the count stays honest.
    """
    if not query:
        return query
    out = []
    for segment in query.split("&"):
        if not segment:
            out.append(segment)
        elif "=" in segment:
            name, _, _value = segment.partition("=")
            out.append(f"{name}={_REDACTED}")
        else:
            out.append(_REDACTED)
    return "&".join(out)


def _sanitize_url(url: str) -> str:
    """Return *url* with userinfo, every query value and any fragment redacted.

    Handles the three spellings a package index URL actually arrives in: absolute
    (``https://u:p@h/x``), scheme-relative (``//u:p@h/x``) and schemeless
    (``u:p@h/x``). Scheme, host, port and path are preserved so the result still says
    *which* index this was.

    RFC 3986 puts at most one *unencoded* ``@`` in an authority: it is the delimiter
    between userinfo and host, and any ``@`` that is logically part of the userinfo itself
    must be percent-encoded (``%40``) by whoever built the URL. Splitting on the last
    literal ``@`` in the authority is therefore correct for plain, username-only and
    percent-encoded userinfo alike, including IPv6 hosts (``[::1]:8443``) and explicit
    ports.

    If the leading component cannot be an authority at all (it contains whitespace) this
    does not try to be clever about locating credentials inside an unparseable string --
    it fails safe and returns a fixed placeholder rather than ever risking the original
    bytes.
    """
    if not url:
        return url

    marker = _AUTHORITY_MARKER_RE.match(url)
    if marker:
        prefix, rest = url[: marker.end()], url[marker.end():]
    else:
        prefix, rest = "", url

    cut = len(rest)
    for delimiter in "/?#":
        found = rest.find(delimiter)
        if found != -1 and found < cut:
            cut = found
    authority, tail = rest[:cut], rest[cut:]

    if any(ch.isspace() for ch in authority):
        # Not an authority. If the string carries any credential-bearing punctuation at
        # all we refuse to guess where the secret ends and hand back a placeholder.
        if "@" in url or "?" in url or "#" in url:
            return _UNPARSEABLE
        return url

    if "@" in authority:
        _, _, hostport = authority.rpartition("@")
        authority = f"{_REDACTED}@{hostport}" if hostport else _REDACTED

    fragment_at = tail.find("#")
    if fragment_at == -1:
        path_query, fragment = tail, None
    else:
        path_query, fragment = tail[:fragment_at], tail[fragment_at + 1:]

    query_at = path_query.find("?")
    if query_at == -1:
        path, query = path_query, None
    else:
        path, query = path_query[:query_at], path_query[query_at + 1:]

    rendered = prefix + authority + path
    if query is not None:
        rendered += "?" + _sanitize_query(query)
    if fragment is not None:
        rendered += "#" + (_REDACTED if fragment else "")
    return rendered


def _sanitize_urls_in_text(text: str) -> str:
    """Sanitise every URL-shaped span in *text*, leaving everything else byte-identical.

    This is the single scrub seam every echoed, persisted or raised string goes through.
    It is deliberately independent of *which* URL this run was given: pip normalises,
    re-quotes and re-spells the index URL in its own diagnostics, a run can involve more
    than one credential-bearing URL on the same line, and a URL this module never saw
    (a redirect target, a proxy, an ``extra-index-url`` inherited from pip config) is
    exactly as sensitive as the one it did see.
    """
    if not text:
        return text
    return _URL_SPAN_RE.sub(lambda m: _sanitize_url(m.group(0)), text)


def _scrub_text(text: str, raw_url: str) -> str:
    """Scrub *text* of credentials: the literal bytes of *raw_url* first, then every
    URL-shaped span found anywhere in the result.

    The literal pass exists so that a *raw_url* spelling the general scanner would not
    recognise as a URL is still removed when this run is the one that produced it; the
    general pass exists so that a spelling this run never handed out (pip's own
    re-rendering, a second index, a redirect) is removed too. Both passes are idempotent,
    so running them in this order is safe.
    """
    if not text:
        return text
    if raw_url and raw_url in text:
        sanitized = _sanitize_url(raw_url)
        if sanitized != raw_url:
            text = text.replace(raw_url, sanitized)
    return _sanitize_urls_in_text(text)


def _scrub_tail(text: str, raw_url: str, limit: int = _TAIL_CHARS) -> str:
    """Scrub the WHOLE of *text*, then keep the last *limit* characters of the result.

    Issue #55 blocker B2: slicing first and scrubbing the slice lets a credential that
    straddles the cut survive, because what reaches the scrub is no longer a URL.

    The cut is additionally aligned to a whitespace boundary. A cut that lands inside an
    already-redacted authority leaves a partial token (``TED@host``) which is not
    provenance, and which the scrub would rewrite -- and lengthen -- on any later pass,
    quietly breaking this field's documented size bound.
    """
    scrubbed = _scrub_text(text, raw_url)
    if limit <= 0:
        return ""
    if len(scrubbed) <= limit:
        return scrubbed
    start = len(scrubbed) - limit
    if not scrubbed[start - 1].isspace():
        advanced = len(scrubbed)
        for i in range(start, len(scrubbed)):
            if scrubbed[i].isspace():
                advanced = i + 1
                break
        start = advanced
    return scrubbed[start:]


def _scrub_obj(obj, raw_url: str):
    """Recursively scrub every string reachable inside *obj* (the persisted record).

    Applied to the whole record immediately before it is written, so a field added later
    -- or a string that arrived from the child interpreter's own JSON -- is covered by
    construction rather than by remembering to wrap each new assignment.
    """
    if isinstance(obj, str):
        return _scrub_text(obj, raw_url)
    if isinstance(obj, dict):
        return {k: _scrub_obj(v, raw_url) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_scrub_obj(v, raw_url) for v in obj]
    return obj


def _echo_cmd(cmd: list) -> str:
    """Render *cmd* for the ``$ ...`` progress echo, credential-scrubbed.

    EVERY token goes through the scrub -- not the ones containing ``://`` (issue #55
    blocker B1: that gate let ``//user:pass@host/...`` and ``user:pass@host/...`` through
    raw), and not the one token following ``--index-url`` (an allowlist of flag names has
    to be kept in sync with every future argument). Tokens with no URL-shaped span in them
    come back byte-identical.
    """
    return " ".join(_sanitize_urls_in_text(str(c)) for c in cmd)


# Runs inside the fresh interpreter. Prints one JSON line prefixed with @@.
_CONSUMER = r'''
import json, sys, sysconfig, os
out = {}
try:
    import numpy as np
    import onnxruntime as ort
    from onnx import TensorProto, helper
    import onnxruntime_ep_vulkan as vk

    out["package_file"] = vk.__file__
    site = sysconfig.get_paths()["purelib"]
    out["site_packages"] = site

    path = vk.register_execution_provider_library()
    out["registered_path"] = path
    out["artifact_inside_site_packages"] = os.path.normcase(
        os.path.abspath(path)).startswith(os.path.normcase(os.path.abspath(site)) + os.sep)
    out["provenance"] = vk.verify_provenance()

    g = helper.make_graph(
        [helper.make_node("Add", ["a", "b"], ["c"])], "g",
        [helper.make_tensor_value_info("a", TensorProto.FLOAT, [4]),
         helper.make_tensor_value_info("b", TensorProto.FLOAT, [4])],
        [helper.make_tensor_value_info("c", TensorProto.FLOAT, [4])])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 10

    sess = ort.InferenceSession(m.SerializeToString(), providers=vk.providers())
    out["session_providers"] = list(sess.get_providers())
    res = sess.run(None, {"a": np.ones(4, np.float32),
                          "b": np.full(4, 2.0, np.float32)})[0]
    out["output"] = [float(v) for v in res]
    out["numerically_correct"] = bool(np.allclose(res, 3.0))
    vk.assert_ep_selected(sess)
    out["ep_selected"] = True
    out["onnxruntime_version"] = ort.__version__
    out["verdict"] = "PASS" if out["numerically_correct"] and \
        out["artifact_inside_site_packages"] else "FAIL"
except Exception as exc:
    out["verdict"] = "FAIL"
    out["error"] = f"{type(exc).__name__}: {exc}"
print("@@" + json.dumps(out))
'''


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print("$", _echo_cmd(cmd))
    return subprocess.run([str(c) for c in cmd], **kw)


def _write_record(record: dict, raw_url: str) -> tuple[Path, dict]:
    """Scrub the whole record, persist it, and hand the scrubbed copy back.

    The scrub happens *here*, at the single seam every persisted byte passes through,
    rather than at each assignment site -- so a field added later cannot reintroduce the
    leak by forgetting to wrap itself. The scrubbed copy is also what gets printed, so
    stdout and the artifact can never disagree about what was redacted.
    """
    out = REPO / "bench" / "results" / "cleanroom_install_dev0.json"
    scrubbed = _scrub_obj(record, raw_url)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scrubbed, indent=2) + "\n", encoding="utf-8")
    return out, scrubbed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--wheel", type=Path, default=None,
                    help="wheel to install (default: newest in python/dist)")
    ap.add_argument("--env", type=Path, default=DEFAULT_ENV)
    ap.add_argument("--keep", action="store_true", help="do not delete the venv afterwards")
    ap.add_argument(
        "--index-url", default=DEFAULT_INDEX_URL,
        help=(
            "package index for the clean venv's `pip install` (default: "
            "%(default)s, or $ONNXRUNTIME_EP_VULKAN_PYPI_INDEX_URL). Pass '' to fall "
            "back to pip's own default index (issue #40: public PyPI is blocked in the "
            "sandboxes this tool is run from during EP development)."
        ),
    )
    args = ap.parse_args(argv)
    # The one live copy of the secret-bearing string. It reaches pip's argv and nothing
    # else; every echo/record/exception path takes it only as the scrub target.
    raw_index_url = args.index_url or ""

    wheel = args.wheel
    if wheel is None:
        wheels = sorted((PY_DIR / "dist").glob("*.whl"),
                        key=lambda p: p.stat().st_mtime)
        if not wheels:
            raise SystemExit("no wheel in python/dist -- run python/build_wheel.py first")
        wheel = wheels[-1]
    wheel = wheel.resolve()

    env_dir = args.env.resolve()
    if env_dir.is_relative_to(REPO):
        raise SystemExit(
            f"refusing to build the clean-room venv inside the repository ({env_dir}): "
            f"the source-checkout fallback would be reachable and the run would not "
            f"discriminate between the wheel and the local build."
        )
    if env_dir.exists():
        shutil.rmtree(env_dir)

    record: dict = {
        "probe": "cleanroom_install",
        "question": (
            "Does a user who has never built this repository get a working Vulkan EP "
            "session from the wheel alone?"
        ),
        "wheel": wheel.name,
        "wheel_sha256": __import__("hashlib").sha256(wheel.read_bytes()).hexdigest(),
        "venv": str(env_dir),
        "host": {"platform": platform.platform(), "python": sys.version.split()[0]},
        "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    fatal: str | None = None
    try:
        if _run([sys.executable, "-m", "venv", env_dir]).returncode != 0:
            raise SystemExit("venv creation failed")
        py = env_dir / ("Scripts" if os.name == "nt" else "bin") / (
            "python.exe" if os.name == "nt" else "python")

        pip_cmd = [py, "-m", "pip", "install", "--disable-pip-version-check", "-q"]
        if args.index_url:
            pip_cmd += ["--index-url", args.index_url]
        pip_cmd += [wheel, "onnx", "numpy"]
        install = _run(pip_cmd, capture_output=True, text=True)
        record["pip_index_url"] = (
            _sanitize_url(args.index_url) if args.index_url else "(pip default)"
        )
        record["pip_returncode"] = install.returncode
        if install.returncode != 0:
            record["verdict"] = "UNOBSERVABLE"
            record["reason"] = "pip install failed in the clean venv"
            # pip's own stderr can echo the --index-url it was given verbatim (e.g. in
            # "Looking in indexes: ..." or a connection-error message), including any
            # userinfo/query credential -- scrub the WHOLE stream, then keep the tail of
            # the sanitised result. Truncating first would leave a straddling credential
            # unmatched and write it into this record (issue #55 blocker B2).
            record["pip_stderr"] = _scrub_tail(install.stderr, raw_index_url)
        else:
            frozen = _run([py, "-m", "pip", "freeze"], capture_output=True, text=True)
            record["installed"] = sorted(frozen.stdout.split())
            # cwd is deliberately outside the repository too, so nothing on sys.path[0]
            # can reach it either.
            proc = _run([py, "-c", _CONSUMER], capture_output=True, text=True,
                        cwd=str(env_dir))
            payload = None
            for line in proc.stdout.splitlines():
                if line.startswith("@@"):
                    payload = json.loads(line[2:])
            if payload is None:
                record["verdict"] = "UNOBSERVABLE"
                record["reason"] = "the clean interpreter emitted no reading"
                record["stderr_tail"] = _scrub_tail(proc.stderr, raw_index_url)
            else:
                record.update(payload)
    except KeyboardInterrupt:
        raise
    except BaseException as exc:  # noqa: BLE001 -- not re-raised; only its TEXT survives
        # Third leak surface (issue #55 blocker B4): an exception rendered by the default
        # excepthook, or by a caller that logs str(exc), can carry the index URL that a
        # failing subprocess/OS call embedded in its message. The original exception
        # object is DROPPED here rather than chained: `raise ... from None` only sets
        # __suppress_context__, leaving the raw message reachable through __context__,
        # so the replacement SystemExit is raised below, outside any handler, where
        # __context__ is None.
        record["verdict"] = "UNOBSERVABLE"
        record["reason"] = "the clean-room run raised before a reading could be taken"
        record["error"] = _scrub_text(f"{type(exc).__name__}: {exc}", raw_index_url)
        out, scrubbed = _write_record(record, raw_index_url)
        print(f"record: {out}")
        fatal = scrubbed["error"]
    finally:
        if not args.keep and env_dir.exists():
            shutil.rmtree(env_dir, ignore_errors=True)

    if fatal is not None:
        raise SystemExit(fatal)

    out, scrubbed = _write_record(record, raw_index_url)
    print("\n" + json.dumps({k: scrubbed[k] for k in scrubbed
                             if k in ("verdict", "session_providers", "output",
                                      "artifact_inside_site_packages", "error",
                                      "reason")}, indent=2))
    print(f"record: {out}")
    return 0 if scrubbed.get("verdict") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
