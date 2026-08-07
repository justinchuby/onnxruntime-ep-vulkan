"""Tests for `python/verify_cleanroom.py`'s package-index URL privacy (issue #55).

`verify_cleanroom.py` accepts a `--index-url` that may point at a private/authenticated
mirror, so its value may embed userinfo (`user:pass@`, a bare username, percent-encoded
credentials), a query credential (`?token=`, `?api_key=`, `?sig=`, whatever the vendor
called it) or a fragment. Four surfaces must never carry any of that in the clear:

1. the ``$ ...`` progress echo `_run()` prints for every subprocess it launches;
2. the persisted record ``bench/results/cleanroom_install_dev0.json`` — **every** field,
   because the scrub is applied to the whole record at the single write seam;
3. anything an exception path renders (``SystemExit`` text, stdout, stderr);
4. the final summary `main()` prints.

The one place the *real*, unredacted URL must still reach is the argv handed straight to
`subprocess.run` for pip — never through a shell.

And one surface must never be *changed*: an ordinary filesystem path. Redaction is a
two-sided risk, and the over-fire side lands in the evidence artifact. A run's record that
says the venv lived somewhere it did not is a worse defect than the leak the over-firing
scanner was guarding against, so the never-mangled table below is as load-bearing as the
never-leaked one, and it carries `@`-bearing Windows profile paths — the default shape on
an Entra-joined box — drive paths, UNC paths, wheel specs, PEP 508 direct references,
e-mail addresses and mixed prose.

**How this is tested.** The end-to-end cases below call production ``vc.main()`` with a
fake ``subprocess`` module and a redirected ``vc.REPO``, and then assert on what the
production code actually printed, raised and persisted. Nothing here re-implements
``main()``'s argv construction or its record shape: a test that rebuilt those locally
would keep passing if ``main()`` started handing pip the *redacted* URL, or started
persisting the raw one, which is exactly the blind spot (B4) that let B1/B2 ship.

Every credential in this file is a synthetic sentinel (``sentinel-user``,
``sentinel-pass``, ...) — never a real secret, and no fixture reads a credential source,
per the secret-handling convention. Every ``@``-bearing identity is likewise synthetic
(``justin.chu@contoso.example``) and is a *path component*, not an address anyone can
reach.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
VERIFY_CLEANROOM = REPO / "python" / "verify_cleanroom.py"

_spec = importlib.util.spec_from_file_location("_verify_cleanroom", VERIFY_CLEANROOM)
assert _spec and _spec.loader
vc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vc)


# ---------------------------------------------------------------------------------------
# Synthetic sentinels. If any of these ever appears in something this tool prints,
# raises or persists, that is the defect.
# ---------------------------------------------------------------------------------------

S_USER = "sentinel-user"
S_PASS = "sentinel-pass"
S_TOKEN = "sentinel-token-value"
S_APIKEY = "sentinel-api-key-value"
S_FRAG = "sentinel-fragment-value"
S_SECOND = "sentinel-second-pass"
SENTINELS = (S_USER, S_PASS, S_TOKEN, S_APIKEY, S_FRAG, S_SECOND)

URL_ABSOLUTE = f"https://{S_USER}:{S_PASS}@mirror.example/pypi/simple"
URL_SCHEME_RELATIVE = f"//{S_USER}:{S_PASS}@mirror.example/pypi/simple"
URL_SCHEMELESS = f"{S_USER}:{S_PASS}@mirror.example/pypi/simple"
URL_QUERY = (
    f"https://mirror.example/pypi/simple?token={S_TOKEN}&api_key={S_APIKEY}"
    f"#{S_FRAG}"
)

# Issue #55 R2: schemeless spellings whose credential is in the QUERY or the FRAGMENT.
# There is no `@` anywhere in the first two, so a shape-based scanner has nothing to see;
# these are the spellings revision 2 echoed raw to stdout while redacting them in the
# record from the identical input.
URL_SCHEMELESS_QUERY = f"mirror.example/pypi/simple?token={S_TOKEN}"
URL_SCHEMELESS_FRAGMENT = f"mirror.example:8443/pypi/simple#{S_FRAG}"
URL_SCHEMELESS_USERINFO_QUERY = (
    f"{S_USER}@mirror.example/pypi/simple?sig={S_APIKEY}"
)
SCHEMELESS_SPELLINGS = (
    URL_SCHEMELESS,
    URL_SCHEMELESS_QUERY,
    URL_SCHEMELESS_FRAGMENT,
    URL_SCHEMELESS_USERINFO_QUERY,
)

# Issue #55 R3: text that is NOT a URL and must come back byte-identical from every seam,
# with or without a `--index-url` in play. `contoso.example` is a reserved example domain;
# nothing here is a real address.
AT_BEARING_IDENTITY = "justin.chu@contoso.example"
NEVER_MANGLED = (
    # Windows profile paths on an Entra/AAD-joined box: the default shape, and the one
    # revision 2 rewrote to `REDACTED@contoso.example\...`, drive letter and all.
    rf"C:\Users\{AT_BEARING_IDENTITY}\AppData\Local\onnxruntime-ep-vulkan-cleanroom",
    rf"C:\Users\{AT_BEARING_IDENTITY}\dist\onnxruntime_ep_vulkan-0.28.0-py3-none-win_amd64.whl",
    rf"C:\Users\{AT_BEARING_IDENTITY}\venv\Scripts\python.exe",
    # UNC path.
    rf"\\fileserver\share\{AT_BEARING_IDENTITY}\dist\pkg.whl",
    # POSIX home directory with the same identity shape.
    f"/home/{AT_BEARING_IDENTITY}/venv/lib/python3.12/site-packages",
    # Wheel specs and requirement specifiers.
    "onnxruntime_ep_vulkan-0.28.0-py3-none-manylinux_2_28_x86_64.whl",
    "pkg[extra]>=1.0,<2",
    # PEP 508 direct reference: `package@file:///...` is a package name followed by a URL,
    # not userinfo followed by a host.
    "onnxruntime-ep-vulkan@file:///C:/dist/pkg.whl",
    f"onnxruntime-ep-vulkan@file:///home/{AT_BEARING_IDENTITY}/dist/pkg.whl",
    # An e-mail address in ordinary prose, and mixed text around one.
    AT_BEARING_IDENTITY,
    f"ERROR: contact {AT_BEARING_IDENTITY} for access to C:\\shared\\mirror",
    # Ordinary argv tokens.
    "--disable-pip-version-check",
    "-q",
    "--index-url",
    "onnx",
    "numpy",
    "",
)


def assert_no_sentinels(label: str, *texts: str) -> None:
    for text in texts:
        for sentinel in SENTINELS:
            assert sentinel not in text, (
                f"{label}: synthetic credential {sentinel!r} leaked into:\n{text}"
            )



# ---------------------------------------------------------------------------------------
# _sanitize_url — the redaction primitive, over every spelling an index URL arrives in
# ---------------------------------------------------------------------------------------


def test_redacts_plain_user_and_password():
    got = vc._sanitize_url(URL_ABSOLUTE)
    assert_no_sentinels("absolute", got)
    assert got == "https://REDACTED@mirror.example/pypi/simple"


def test_redacts_username_only():
    got = vc._sanitize_url(f"https://{S_USER}@mirror.example/pypi/simple")
    assert_no_sentinels("username-only", got)
    assert got == "https://REDACTED@mirror.example/pypi/simple"


def test_redacts_scheme_relative_url():
    """B1: `//user:pass@host/...` has no `://` and used to bypass the echo's gate."""
    got = vc._sanitize_url(URL_SCHEME_RELATIVE)
    assert_no_sentinels("scheme-relative", got)
    assert got == "//REDACTED@mirror.example/pypi/simple"


def test_redacts_schemeless_authority_url():
    """B1: `user:pass@host/...` — the literal shape README.md names."""
    got = vc._sanitize_url(URL_SCHEMELESS)
    assert_no_sentinels("schemeless", got)
    assert got == "REDACTED@mirror.example/pypi/simple"


def test_redacts_percent_encoded_userinfo():
    # "sentinel@user" percent-encoded as the username, "sentinel@pass" as the password --
    # the literal '@' inside each is %40-encoded, so exactly one *unencoded* '@' remains
    # as the userinfo/host delimiter.
    url = "https://sentinel%40user:sentinel%40pass@mirror.example/pypi/simple"
    got = vc._sanitize_url(url)
    assert "sentinel%40user" not in got
    assert "sentinel%40pass" not in got
    assert got == "https://REDACTED@mirror.example/pypi/simple"


def test_no_userinfo_present_is_returned_unchanged():
    url = "https://packagefeedproxy.microsoft.io/pypi/simple"
    assert vc._sanitize_url(url) == url


def test_ipv6_host_with_port_is_preserved():
    got = vc._sanitize_url(f"https://{S_USER}:{S_PASS}@[::1]:8443/pypi/simple")
    assert_no_sentinels("ipv6", got)
    assert got == "https://REDACTED@[::1]:8443/pypi/simple"


def test_ipv6_full_address_with_port_and_query_value_is_redacted_host_preserved():
    got = vc._sanitize_url(
        f"https://{S_USER}:{S_PASS}@[2001:db8::1]:443/pypi/simple?token={S_TOKEN}"
    )
    assert_no_sentinels("ipv6-query", got)
    assert got == "https://REDACTED@[2001:db8::1]:443/pypi/simple?token=REDACTED"


def test_explicit_port_is_preserved():
    got = vc._sanitize_url(f"https://{S_USER}:{S_PASS}@mirror.example:8443/pypi/simple")
    assert got == "https://REDACTED@mirror.example:8443/pypi/simple"


def test_empty_input_passes_through():
    assert vc._sanitize_url("") == ""


def test_malformed_url_with_no_identifiable_split_point_falls_back_to_placeholder():
    got = vc._sanitize_url("not a url at all @ weird")
    assert got == "<REDACTED-unparseable-index-url>"


def test_url_with_no_at_sign_is_never_mistaken_for_having_userinfo():
    url = "https://mirror.example/pypi/simple"
    assert vc._sanitize_url(url) == url


def test_sanitize_url_is_idempotent():
    once = vc._sanitize_url(URL_QUERY)
    assert vc._sanitize_url(once) == once


# ---------------------------------------------------------------------------------------
# B3: query and fragment policy — redact every VALUE, keep every NAME and the count
# ---------------------------------------------------------------------------------------


def test_every_query_value_is_redacted_not_just_known_secret_names():
    """A denylist is bypassed the moment a mirror invents a new name for the same
    secret. `vendorNonceV2` is not on anyone's list; it must still be redacted."""
    got = vc._sanitize_url(
        f"https://mirror.example/simple?vendorNonceV2={S_TOKEN}&harmless=42"
    )
    assert_no_sentinels("novel-param", got)
    assert got == "https://mirror.example/simple?vendorNonceV2=REDACTED&harmless=REDACTED"


def test_query_parameter_names_and_count_survive_as_provenance():
    got = vc._sanitize_url(
        f"https://mirror.example/simple?token={S_TOKEN}&api_key={S_APIKEY}"
        f"&password={S_PASS}&access_token={S_TOKEN}"
    )
    assert_no_sentinels("named-secrets", got)
    assert got == (
        "https://mirror.example/simple"
        "?token=REDACTED&api_key=REDACTED&password=REDACTED&access_token=REDACTED"
    )
    assert got.count("=REDACTED") == 4  # four credentials were present, and we say so


def test_duplicate_parameter_names_keep_their_multiplicity():
    got = vc._sanitize_url(
        f"https://mirror.example/simple?key={S_TOKEN}&key={S_APIKEY}&key="
    )
    assert_no_sentinels("duplicates", got)
    assert got == "https://mirror.example/simple?key=REDACTED&key=REDACTED&key=REDACTED"


def test_blank_and_empty_query_segments():
    # `a=` (blank value) still renders REDACTED -- whether a credential was empty is not
    # worth leaking the distinction for. A genuinely empty segment stays empty so the
    # `&` count remains honest.
    assert vc._sanitize_url("https://mirror.example/s?a=&&b=1") == (
        "https://mirror.example/s?a=REDACTED&&b=REDACTED"
    )
    # A trailing `?` with nothing after it is preserved verbatim.
    assert vc._sanitize_url("https://mirror.example/s?") == "https://mirror.example/s?"


def test_query_segment_with_no_equals_is_redacted_whole():
    """An `=`-less segment is an opaque token. Nothing distinguishes a flag from a
    signed-URL credential pasted in bare, so it is redacted entirely."""
    got = vc._sanitize_url(f"https://mirror.example/simple?{S_TOKEN}")
    assert_no_sentinels("bare-segment", got)
    assert got == "https://mirror.example/simple?REDACTED"


def test_percent_encoded_query_names_and_values():
    got = vc._sanitize_url(
        f"https://mirror.example/simple?x%2Dtoken={S_TOKEN}%3Dpadding"
    )
    assert_no_sentinels("percent-encoded-query", got)
    # The name is copied verbatim (never decoded and re-encoded), the value is gone.
    assert got == "https://mirror.example/simple?x%2Dtoken=REDACTED"


def test_fragment_is_redacted_but_its_presence_is_recorded():
    got = vc._sanitize_url(f"https://mirror.example/simple#{S_FRAG}")
    assert_no_sentinels("fragment", got)
    assert got == "https://mirror.example/simple#REDACTED"


def test_empty_fragment_is_preserved_as_empty():
    assert vc._sanitize_url("https://mirror.example/simple#") == (
        "https://mirror.example/simple#"
    )


def test_userinfo_query_and_fragment_together():
    got = vc._sanitize_url(
        f"https://{S_USER}:{S_PASS}@mirror.example:8443/pypi/simple"
        f"?token={S_TOKEN}&api_key={S_APIKEY}#{S_FRAG}"
    )
    assert_no_sentinels("everything", got)
    assert got == (
        "https://REDACTED@mirror.example:8443/pypi/simple"
        "?token=REDACTED&api_key=REDACTED#REDACTED"
    )


def test_documented_limitation_a_path_embedded_token_is_not_redacted():
    """Stated so the limitation is a decision, not a surprise: the path is the last
    provenance left after userinfo/query/fragment are gone, so it is preserved. This
    test exists to make the README's claim falsifiable rather than aspirational."""
    url = "https://mirror.example/t/path-embedded-token/simple"
    assert vc._sanitize_url(url) == url


# ---------------------------------------------------------------------------------------
# _scan_url_spans — pass 3, the SYNTACTIC scan (issue #55 R3: structure, never a guess)
# ---------------------------------------------------------------------------------------


def test_multiple_urls_on_one_line_are_all_redacted():
    text = (
        f"Looking in indexes: {URL_ABSOLUTE}, "
        f"https://other-user:{S_SECOND}@other.example/simple "
        f"and {URL_SCHEME_RELATIVE}"
    )
    got = vc._scan_url_spans(text)
    assert_no_sentinels("multi-url", got)
    assert got.count("REDACTED@") == 3
    assert "Looking in indexes" in got  # surrounding text preserved


@pytest.mark.parametrize("token", NEVER_MANGLED, ids=range(len(NEVER_MANGLED)))
def test_the_syntactic_scan_never_mangles_text_that_is_not_a_url(token):
    """R3. The scan recognises an RFC 3986 ``//`` authority marker and nothing else.

    Revision 2 also matched a schemeless ``user@host`` and turned
    ``C:\\Users\\a.b@corp\\...`` into ``REDACTED@corp\\...`` — drive letter and every
    directory before the ``@`` destroyed — in the echoed argv *and* in the tracked record.
    An `@` is not an authority delimiter outside an authority.
    """
    assert vc._scan_url_spans(token) == token


@pytest.mark.parametrize("token", NEVER_MANGLED, ids=range(len(NEVER_MANGLED)))
@pytest.mark.parametrize("raw", ["", URL_ABSOLUTE, URL_QUERY, URL_SCHEMELESS],
                         ids=["no-url", "absolute", "query", "schemeless"])
def test_the_whole_scrub_seam_never_mangles_text_that_is_not_a_url(token, raw):
    """The same table through the *whole* seam, with a credential-bearing URL in play.

    R3's sharpest edge: the corruption fired with **no `--index-url` at all**, so the
    `raw=""` arm is the one that reproduces the reported incident, and the others prove
    the raw-derived literal pass cannot corrupt an unrelated string either.
    """
    assert vc._scrub_text(token, raw) == token


def test_a_pep508_direct_reference_keeps_its_package_name_and_path():
    token = "onnxruntime-ep-vulkan@file:///C:/dist/pkg.whl"
    assert vc._scan_url_spans(token) == token
    # ... and the documented fragment policy still applies to the URL part of it.
    assert vc._scan_url_spans(token + "#sha256=deadbeef") == (
        "onnxruntime-ep-vulkan@file:///C:/dist/pkg.whl#REDACTED"
    )


def test_text_with_no_url_is_byte_identical():
    text = "ERROR: Could not find a version that satisfies the requirement foo\n"
    assert vc._scan_url_spans(text) == text


def test_a_foreign_url_with_a_scheme_is_still_scanned():
    """The accepted limitation is *schemeless* foreign URLs. Anything pip renders for
    itself carries a scheme, and must still be redacted without the run's own URL."""
    text = f"Looking in indexes: https://other:{S_SECOND}@other.example/simple?t={S_TOKEN}"
    got = vc._scrub_text(text, "")
    assert_no_sentinels("foreign scheme-bearing url", got)


# ---------------------------------------------------------------------------------------
# One seam. Two functions where one is strictly stronger is a gate wearing a different hat
# ---------------------------------------------------------------------------------------


def _functions_calling(name: str) -> set[str]:
    """Every top-level function in the production module whose body calls *name*."""
    tree = ast.parse(VERIFY_CLEANROOM.read_text(encoding="utf-8"))
    callers = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(node):
            if (isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name)
                    and inner.func.id == name):
                callers.add(node.name)
    return callers


def test_the_general_scanner_is_reachable_only_through_the_one_scrub_seam():
    """B1 survived two review rounds because there were two seams: `_scrub_text` (raw
    literal + general scan) and `_echo_cmd` (general scan only). The next consumer picks
    the weaker one — that is not a hypothetical, it is the incident. This asserts the
    weaker one is not reachable at all."""
    assert _functions_calling("_scan_url_spans") == {"_scrub_text"}


def test_every_echo_and_record_path_carries_the_run_s_own_url():
    """`_echo_cmd` and `_scrub_obj` must both take the raw URL, or a caller can construct
    the R2 leak again simply by not passing it. Taking it is not enough — `_run`, the only
    production caller of the echo, must actually hand it over."""
    tree = ast.parse(VERIFY_CLEANROOM.read_text(encoding="utf-8"))
    signatures = {
        node.name: [a.arg for a in node.args.args]
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }
    for fn in ("_echo_cmd", "_scrub_obj", "_scrub_text", "_scrub_tail", "_run"):
        assert "raw_url" in signatures[fn], f"{fn} cannot scrub the run's own URL"

    run = next(n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "_run")
    echoes = [n for n in ast.walk(run)
              if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "_echo_cmd"]
    assert echoes, "_run no longer echoes the command at all"
    for call in echoes:
        passed = [getattr(a, "id", None) for a in call.args]
        passed += [k.arg for k in call.keywords]
        assert "raw_url" in passed, "_run echoes without the run's own URL (issue #55 R2)"


# ---------------------------------------------------------------------------------------
# Idempotence — `_scrub_obj` scrubs the record a SECOND time at the write seam
# ---------------------------------------------------------------------------------------

_IDEMPOTENCE_CORPUS = (
    URL_ABSOLUTE, URL_SCHEME_RELATIVE, URL_SCHEMELESS, URL_QUERY,
    URL_SCHEMELESS_QUERY, URL_SCHEMELESS_FRAGMENT, URL_SCHEMELESS_USERINFO_QUERY,
    "//a@b?c", "https://h/s?a=1", "https://h/s?a=&&b=1", "//u@h", "a@b", "a@b/c",
    "x y @ z", "//u:p@[::1]:8443/x?t=1#f", "pkg@file:///C:/d.whl#sha256=ab",
    f"Looking in indexes: {URL_ABSOLUTE}\nERROR: refused\n",
    f"{URL_QUERY} {URL_ABSOLUTE} {URL_SCHEME_RELATIVE}",
    *NEVER_MANGLED,
)


@pytest.mark.parametrize("text", _IDEMPOTENCE_CORPUS,
                         ids=range(len(_IDEMPOTENCE_CORPUS)))
@pytest.mark.parametrize("raw", ["", URL_ABSOLUTE, URL_QUERY, URL_SCHEMELESS_QUERY],
                         ids=["no-url", "absolute", "query", "schemeless-query"])
def test_the_scrub_is_idempotent_and_never_grows_already_scrubbed_text(text, raw):
    """Not a nicety. `_scrub_obj` runs over the whole record at the write seam, so
    `pip_stderr` — already scrubbed and truncated to `_TAIL_CHARS` by `_scrub_tail` — is
    scrubbed a second time. A scrub that rewrote or *lengthened* its own output would
    break the field's documented size bound and contradict the module docstring."""
    once = vc._scrub_text(text, raw)
    twice = vc._scrub_text(once, raw)
    assert twice == once
    assert len(twice) <= len(once)


def test_sanitize_url_is_idempotent_over_every_spelling():
    for url in (URL_ABSOLUTE, URL_SCHEME_RELATIVE, URL_SCHEMELESS, URL_QUERY,
                *SCHEMELESS_SPELLINGS, "not a url at all @ weird"):
        once = vc._sanitize_url(url)
        assert vc._sanitize_url(once) == once, url



# ---------------------------------------------------------------------------------------
# _scrub_text / _scrub_tail — B2: scrub the WHOLE text, then truncate the result
# ---------------------------------------------------------------------------------------


def test_scrub_text_removes_the_raw_url_from_pip_style_error_text():
    text = (
        f"Looking in indexes: {URL_ABSOLUTE}\n"
        f"ERROR: Could not find a version that satisfies the requirement foo\n"
        f"ERROR: could not fetch URL {URL_ABSOLUTE}: connection refused"
    )
    got = vc._scrub_text(text, URL_ABSOLUTE)
    assert_no_sentinels("scrub_text", got)
    assert "REDACTED" in got
    assert "Looking in indexes" in got


def test_scrub_text_also_removes_a_url_this_run_never_handed_out():
    """pip inherits `extra-index-url` from its own config and re-spells URLs in its
    diagnostics. The general scan must not depend on matching the run's own argument."""
    text = f"Looking in indexes: https://other:{S_SECOND}@other.example/simple"
    got = vc._scrub_text(text, URL_ABSOLUTE)
    assert_no_sentinels("foreign-url", got)


def test_scrub_text_is_a_no_op_when_there_is_nothing_to_redact():
    text = f"Looking in indexes: {vc.DEFAULT_INDEX_URL}\n"
    assert vc._scrub_text(text, vc.DEFAULT_INDEX_URL) == text


@pytest.mark.parametrize(
    "raw, text",
    [
        (URL_ABSOLUTE,
         f"ERROR: 401 Client Error: Unauthorized for user {S_USER}"),
        (URL_ABSOLUTE,
         f"keyring: no password for {S_USER}; falling back to {S_PASS}"),
        (URL_ABSOLUTE,
         f"WARNING: netrc entry 'mirror.example' -> {S_USER}:{S_PASS}"),
        (URL_QUERY, f"ERROR: rejected token {S_TOKEN} (expired)"),
        (URL_QUERY, f"ERROR: api key {S_APIKEY} is not authorised"),
        (URL_SCHEMELESS_FRAGMENT, f"ERROR: bad fragment {S_FRAG}"),
    ],
    ids=["user", "password", "netrc-pair", "query-token", "api-key", "fragment"],
)
def test_scrub_text_removes_the_credential_even_when_the_url_around_it_is_gone(raw, text):
    """The exact-value pass only fires when the WHOLE raw URL is echoed back, and the
    syntactic scan only fires on URL-shaped spans. Neither covers the common case: pip,
    keyring and the OS quote the *credential alone*, stripped of its URL. The raw-derived
    literal pass is what covers it, and it is the only reason a schemeless credential the
    scanner is forbidden to guess at (issue #55 R3) is still redacted here."""
    got = vc._scrub_text(text, raw)
    assert_no_sentinels("credential-without-its-url", got)
    assert "REDACTED" in got


def test_the_literal_pass_never_shreds_the_provenance_it_leaves_behind():
    """Redacting raw-derived literals must not eat the host, the path or the parameter
    names — those are the provenance the record exists to carry."""
    got = vc._scrub_text(f"Looking in indexes: {URL_QUERY}", URL_QUERY)
    assert_no_sentinels("provenance", got)
    for keep in ("mirror.example", "/pypi/simple", "token=", "api_key="):
        assert keep in got, keep


def test_scrub_text_is_a_no_op_when_raw_url_is_empty_and_text_is_clean():
    text = "some unrelated error text"
    assert vc._scrub_text(text, "") == text


def _straddling_stderr(url: str, offset: int) -> str:
    """pip-shaped stderr whose embedded *url* starts exactly *offset* characters before
    the point at which a trailing ``_TAIL_CHARS`` slice of the RAW text would cut.

    ``offset == 0``  -> the URL begins exactly at the cut.
    ``0 < offset < len(url)`` -> the URL straddles the cut; a slice-then-scrub
    implementation sees only a suffix of it, which is not a URL, so it leaks.
    ``offset >= len(url)`` -> the URL is entirely before the cut and is dropped.

    The URL is whitespace-delimited on both sides, as it is in real pip output, so the
    straddle is genuinely about the truncation edge and not about token gluing.
    """
    suffix_len = vc._TAIL_CHARS + offset - len(url)
    assert suffix_len >= 1, "test constructed with an impossible offset"
    head = "ERROR: Could not find a version that satisfies the requirement foo\n" * 20
    suffix = " " + "\n".join("retrying (attempt padding line)" for _ in range(400))
    assert len(suffix) >= suffix_len, "padding too short for this offset"
    return head + url + suffix[:suffix_len]


@pytest.mark.parametrize("offset", [0, 1, 7, 20, len(URL_ABSOLUTE) // 2,
                                    len(URL_ABSOLUTE) - 1, len(URL_ABSOLUTE),
                                    len(URL_ABSOLUTE) + 1])
def test_scrub_tail_never_leaks_a_credential_that_straddles_the_truncation_edge(offset):
    text = _straddling_stderr(URL_ABSOLUTE, offset)
    got = vc._scrub_tail(text, URL_ABSOLUTE)
    assert_no_sentinels(f"straddle offset={offset}", got)
    assert len(got) <= vc._TAIL_CHARS


@pytest.mark.parametrize("offset", [0, 1, 9, len(URL_QUERY) // 2, len(URL_QUERY) - 1])
def test_scrub_tail_straddle_also_covers_query_and_fragment_credentials(offset):
    text = _straddling_stderr(URL_QUERY, offset)
    got = vc._scrub_tail(text, URL_QUERY)
    assert_no_sentinels(f"query straddle offset={offset}", got)
    assert len(got) <= vc._TAIL_CHARS


_FOREIGN_URL = f"https://other:{S_SECOND}@other.example/simple"


@pytest.mark.parametrize(
    "offset", [_FOREIGN_URL.index(S_SECOND) + d for d in (-3, -2, -1, 0)]
)
def test_scrub_tail_straddle_covers_a_url_this_run_never_handed_out(offset):
    """The straddle tests above are all satisfiable by the raw-derived literal pass, which
    would let a slice-then-scrub implementation back in. This one is not: the credential
    belongs to a URL this process never saw, so only scrubbing the WHOLE text *before*
    truncating can catch it (issue #55 B2). The offsets place the cut at or just before
    the sentinel's first character — the exact window in which what survives the cut is no
    longer URL-shaped but the secret inside it is still whole."""
    text = _straddling_stderr(_FOREIGN_URL, offset)
    got = vc._scrub_tail(text, URL_ABSOLUTE)
    assert_no_sentinels(f"foreign straddle offset={offset}", got)
    assert len(got) <= vc._TAIL_CHARS


# ---------------------------------------------------------------------------------------
# _echo_cmd — B1/R2: EVERY token goes through the ONE seam, carrying the run's own URL
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("url", [URL_ABSOLUTE, URL_SCHEME_RELATIVE, URL_SCHEMELESS,
                                 URL_QUERY, *SCHEMELESS_SPELLINGS])
def test_echo_cmd_redacts_every_url_spelling(url):
    echoed = vc._echo_cmd(
        ["python.exe", "-m", "pip", "install", "--index-url", url, "somepkg"], url
    )
    assert_no_sentinels("echo_cmd", echoed)
    assert "--index-url" in echoed
    assert "somepkg" in echoed  # non-URL tokens are untouched


@pytest.mark.parametrize("url", SCHEMELESS_SPELLINGS)
def test_echo_cmd_redacts_a_schemeless_credential_the_scanner_cannot_see(url):
    """R2, at the unit level. `mirror.example/simple?token=...` has no `@` and no `//`:
    there is nothing for a shape-based scanner to recognise, and revision 2's `_echo_cmd`
    used only a shape-based scanner. The value is known by *identity* (it is the URL this
    run was given) and by *argument position*, and either is enough."""
    assert_no_sentinels("schemeless echo", vc._echo_cmd(["pip", "--index-url", url], url))
    # Even with the raw value withheld, argument position alone still redacts it.
    assert_no_sentinels("schemeless echo, position only",
                        vc._echo_cmd(["pip", "--index-url", url], ""))
    assert_no_sentinels("schemeless echo, --flag=value",
                        vc._echo_cmd([f"--index-url={url}"], ""))


@pytest.mark.parametrize("url", SCHEMELESS_SPELLINGS)
def test_echo_cmd_redacts_the_runs_own_url_in_any_argument_position(url):
    """Argument position covers `--index-url <value>`; identity covers everything else.
    pip takes the same value in positions this function cannot recognise — a bare
    requirement, `--find-links`, an inherited `PIP_*` value rendered into argv by a future
    caller — and the run's own credential must never reach stdout from any of them.
    Without the raw value the echo is only as strong as the shape scanner, which is
    exactly the R2 leak."""
    for cmd in (["pip", "install", url],
                ["pip", "install", "--find-links", url],
                ["pip", "install", f"--trusted-host={url}"],
                ["pip", "install", f"somepkg @ {url}"]):
        assert_no_sentinels(f"echo position={cmd[2:]}", vc._echo_cmd(cmd, url))


@pytest.mark.parametrize("token", NEVER_MANGLED, ids=range(len(NEVER_MANGLED)))
def test_echo_cmd_never_mangles_an_ordinary_argument(token):
    """R3 on the echo surface specifically: the argv this tool prints must be the argv it
    ran, byte for byte, wherever it is not a credential."""
    assert vc._echo_cmd(["python", token], URL_ABSOLUTE) == f"python {token}"


def test_echo_cmd_leaves_the_default_proxy_url_readable():
    # Default approved proxy (issue #40) has no credentials -- redaction must be a no-op
    # for it, so the echoed command stays useful for a human reading CI logs.
    echoed = vc._echo_cmd(["pip", "install", "--index-url", vc.DEFAULT_INDEX_URL, "pkg"],
                          vc.DEFAULT_INDEX_URL)
    assert vc.DEFAULT_INDEX_URL in echoed


def test_echo_cmd_handles_non_string_tokens_like_paths():
    echoed = vc._echo_cmd(["pip", "install", Path("C:/some/wheel.whl")], "")
    assert "wheel.whl" in echoed


def test_default_index_url_default_behaviour_is_unchanged():
    # Issue #40's approved-proxy default must still be exactly what pip receives when the
    # caller passes nothing -- redaction work must not alter the default value itself.
    assert vc.DEFAULT_INDEX_URL.startswith("https://")
    assert "@" not in vc.DEFAULT_INDEX_URL
    assert vc._sanitize_url(vc.DEFAULT_INDEX_URL) == vc.DEFAULT_INDEX_URL


# =======================================================================================
# End-to-end: production main(), with a fake shell-free subprocess and a redirected REPO
# =======================================================================================

_PAYLOAD_PASS = {
    "package_file": "/venv/lib/site-packages/onnxruntime_ep_vulkan/__init__.py",
    "site_packages": "/venv/lib/site-packages",
    "registered_path": "/venv/lib/site-packages/onnxruntime_ep_vulkan/_lib/lib.so",
    "artifact_inside_site_packages": True,
    "session_providers": ["VulkanExecutionProvider", "CPUExecutionProvider"],
    "output": [3.0, 3.0, 3.0, 3.0],
    "numerically_correct": True,
    "ep_selected": True,
    "verdict": "PASS",
}


class FakeSubprocess:
    """Stand-in for the `subprocess` module `verify_cleanroom` calls.

    Records every argv it is handed (so a test can assert *where* the real URL went),
    asserts no invocation ever asks for a shell, and returns scripted results.
    """

    def __init__(self, *, install_rc=0, install_stderr="", freeze_stdout="",
                 consumer_stdout=None, consumer_stderr="", raise_on=None,
                 raise_exc=None):
        self.calls: list[list[str]] = []
        self.install_rc = install_rc
        self.install_stderr = install_stderr
        self.freeze_stdout = freeze_stdout
        self.consumer_stdout = consumer_stdout
        self.consumer_stderr = consumer_stderr
        self.raise_on = raise_on
        self.raise_exc = raise_exc

    @staticmethod
    def kind(argv: list[str]) -> str:
        if "venv" in argv:
            return "venv"
        if "install" in argv:
            return "pip-install"
        if "freeze" in argv:
            return "pip-freeze"
        return "consumer"

    def argv_for(self, kind: str) -> list[str]:
        for argv in self.calls:
            if self.kind(argv) == kind:
                return argv
        raise AssertionError(f"production code never invoked a {kind!r} subprocess")

    def run(self, cmd, **kw):
        argv = [str(c) for c in cmd]
        self.calls.append(argv)
        assert not kw.get("shell", False), "pip must never be invoked through a shell"
        kind = self.kind(argv)
        if self.raise_on == kind:
            raise self.raise_exc
        if kind == "venv":
            return subprocess.CompletedProcess(argv, 0, "", "")
        if kind == "pip-install":
            return subprocess.CompletedProcess(
                argv, self.install_rc, "", self.install_stderr)
        if kind == "pip-freeze":
            return subprocess.CompletedProcess(argv, 0, self.freeze_stdout, "")
        stdout = self.consumer_stdout
        if stdout is None:
            stdout = "@@" + json.dumps(_PAYLOAD_PASS) + "\n"
        return subprocess.CompletedProcess(argv, 0, stdout, self.consumer_stderr)


@pytest.fixture
def cleanroom(tmp_path, monkeypatch):
    """A whole fake world for `main()`: a repository root to persist into, a wheel to
    hash, and a venv directory outside that root."""
    repo = tmp_path / "repo"
    (repo / "python").mkdir(parents=True)
    monkeypatch.setattr(vc, "REPO", repo)

    wheel = tmp_path / "dist" / "onnxruntime_ep_vulkan-0.0.0-py3-none-any.whl"
    wheel.parent.mkdir(parents=True)
    wheel.write_bytes(b"synthetic wheel bytes -- never a real artifact")

    return SimpleNamespace(
        repo=repo,
        wheel=wheel,
        env=tmp_path / "cleanroom-venv",
        record=repo / "bench" / "results" / "cleanroom_install_dev0.json",
    )


def drive(cleanroom, fake, monkeypatch, *, index_url=None, extra=()):
    """Call production `main()` with `fake` standing in for `subprocess`."""
    monkeypatch.setattr(vc, "subprocess", fake)
    argv = ["--wheel", str(cleanroom.wheel), "--env", str(cleanroom.env), *extra]
    if index_url is not None:
        argv += ["--index-url", index_url]
    return vc.main(argv)


def persisted(cleanroom) -> tuple[str, dict]:
    raw = cleanroom.record.read_text(encoding="utf-8")
    return raw, json.loads(raw)


# --- success path ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "url,expected_index_field",
    [
        (URL_ABSOLUTE, "https://REDACTED@mirror.example/pypi/simple"),
        (URL_SCHEME_RELATIVE, "//REDACTED@mirror.example/pypi/simple"),
        (URL_SCHEMELESS, "REDACTED@mirror.example/pypi/simple"),
        (URL_QUERY,
         "https://mirror.example/pypi/simple?token=REDACTED&api_key=REDACTED#REDACTED"),
        (URL_SCHEMELESS_QUERY, "mirror.example/pypi/simple?token=REDACTED"),
        (URL_SCHEMELESS_FRAGMENT, "mirror.example:8443/pypi/simple#REDACTED"),
        (URL_SCHEMELESS_USERINFO_QUERY,
         "REDACTED@mirror.example/pypi/simple?sig=REDACTED"),
    ],
    ids=["absolute", "scheme-relative", "schemeless", "query-and-fragment",
         "schemeless-query", "schemeless-fragment", "schemeless-userinfo-query"],
)
def test_main_success_path_leaks_nothing_and_still_gives_pip_the_real_url(
        cleanroom, monkeypatch, capsys, url, expected_index_field):
    fake = FakeSubprocess(
        freeze_stdout=(
            "onnxruntime-ep-vulkan @ "
            "file:///C:/dist/pkg.whl#sha256=deadbeef\nnumpy==2.5.1\nonnx==1.22.0\n"
        ),
    )
    rc = drive(cleanroom, fake, monkeypatch, index_url=url)
    captured = capsys.readouterr()
    raw_json, record = persisted(cleanroom)

    assert rc == 0
    # Surfaces 1, 3, 4: everything the production code printed.
    assert_no_sentinels("stdout", captured.out)
    assert_no_sentinels("stderr", captured.err)
    # Surface 2: the actual persisted artifact, as bytes, not as a field we chose to look at.
    assert_no_sentinels("persisted record", raw_json)
    assert record["pip_index_url"] == expected_index_field
    assert record["verdict"] == "PASS"

    # ... and the real URL reached pip's argv, and ONLY pip's argv.
    install_argv = fake.argv_for("pip-install")
    assert url in install_argv
    assert install_argv[install_argv.index("--index-url") + 1] == url
    for argv in fake.calls:
        if argv is not install_argv:
            assert url not in " ".join(argv)


# --- R2/R3 end-to-end: the two blockers, through production main() -----------------------


@pytest.mark.parametrize("url", SCHEMELESS_SPELLINGS,
                         ids=["userinfo", "query", "fragment", "userinfo-query"])
def test_main_never_echoes_a_schemeless_index_url_in_the_clear(
        cleanroom, monkeypatch, capsys, url):
    """R2, end to end, on three spellings the general scanner deliberately cannot see.

    Revision 2 routed the record through a raw-aware seam and the `$ ...` echo through a
    shape-only one, so a schemeless URL carrying its credential in the query or the
    fragment was *printed raw* while the identical input was redacted in the artifact.
    The echo runs before pip, so it leaked whether or not pip would have accepted the
    value. Everything below is what production code actually emitted."""
    fake = FakeSubprocess(
        install_stderr=f"Looking in indexes: {url}\n",
        freeze_stdout="numpy==2.5.1\n",
    )
    rc = drive(cleanroom, fake, monkeypatch, index_url=url)
    captured = capsys.readouterr()
    raw_json, record = persisted(cleanroom)

    assert rc == 0
    assert_no_sentinels("stdout (the `$ ...` echo)", captured.out)
    assert_no_sentinels("stderr", captured.err)
    assert_no_sentinels("persisted record", raw_json)
    assert "REDACTED" in record["pip_index_url"]
    # ... and the raw value still reached pip's argv, and only pip's argv.
    install_argv = fake.argv_for("pip-install")
    assert install_argv[install_argv.index("--index-url") + 1] == url
    for argv in fake.calls:
        if argv is not install_argv:
            assert url not in " ".join(argv)
    # The echo line for the install itself is present and names the flag, so the record
    # of *what was run* survives redaction.
    assert "--index-url" in captured.out


@pytest.mark.parametrize("raw_index", ["", URL_ABSOLUTE, URL_SCHEMELESS_QUERY],
                         ids=["no-index-url", "absolute", "schemeless-query"])
def test_main_does_not_mutate_an_at_bearing_profile_path_anywhere_it_records_it(
        cleanroom, monkeypatch, capsys, tmp_path, raw_index):
    """R3, end to end. `C:\\Users\\justin.chu@contoso.com\\...` is the default profile
    shape on an Entra/AAD-joined Windows box — the platform this repository's Windows lane
    targets. Revision 2 rewrote it to `REDACTED@contoso.com\\...` in the echoed argv *and*
    in `record["venv"]` / `record["wheel"]` / `registered_path`, with **no `--index-url`
    at all**, so the tracked artifact recorded a path that is not the path the run used.

    The venv and wheel paths here are real directories containing a real `@`, so the
    assertion is on what `main()` persisted for a path it actually used."""
    profile = tmp_path / AT_BEARING_IDENTITY
    env_dir = profile / "onnxruntime-ep-vulkan-cleanroom"
    wheel = profile / "dist" / "onnxruntime_ep_vulkan-0.0.0-py3-none-any.whl"
    wheel.parent.mkdir(parents=True)
    wheel.write_bytes(b"synthetic wheel bytes -- never a real artifact")

    registered = str(profile / "venv" / "lib" / "onnxruntime_ep_vulkan" / "_lib.so")
    payload = dict(_PAYLOAD_PASS, registered_path=registered)
    fake = FakeSubprocess(consumer_stdout="@@" + json.dumps(payload) + "\n")

    monkeypatch.setattr(vc, "subprocess", fake)
    argv = ["--wheel", str(wheel), "--env", str(env_dir)]
    if raw_index:
        argv += ["--index-url", raw_index]
    rc = vc.main(argv)
    captured = capsys.readouterr()
    _raw_json, record = persisted(cleanroom)

    assert rc == 0
    assert record["venv"] == str(env_dir.resolve()), "the recorded venv path was mutated"
    assert record["wheel"] == wheel.name, "the recorded wheel name was mutated"
    assert record["registered_path"] == registered, "the child's own path was mutated"
    assert AT_BEARING_IDENTITY in record["venv"]
    # The echoed argv is the argv that ran, byte for byte, for every non-credential token.
    assert str(env_dir.resolve()) in captured.out
    if raw_index:
        assert_no_sentinels("stdout", captured.out)


@pytest.mark.parametrize("token", NEVER_MANGLED[:6], ids=range(6))
def test_main_leaves_at_bearing_paths_in_the_child_payload_byte_identical(
        cleanroom, monkeypatch, capsys, token):
    """The child interpreter's JSON is merged into the record wholesale and scrubbed at
    the write seam. Whatever it reports about its own filesystem must survive that."""
    payload = dict(_PAYLOAD_PASS, registered_path=token, site_packages=token)
    fake = FakeSubprocess(consumer_stdout="@@" + json.dumps(payload) + "\n")
    drive(cleanroom, fake, monkeypatch, index_url=URL_QUERY)
    _raw, record = persisted(cleanroom)

    assert record["registered_path"] == token
    assert record["site_packages"] == token


def test_main_scrubs_pip_freeze_output_including_a_wheel_digest_fragment(
        cleanroom, monkeypatch, capsys):
    """Documented consequence of the redact-by-default fragment policy: a local wheel's
    `#sha256=` fragment in `pip freeze` output renders as `#REDACTED`. The wheel digest
    is still recorded verbatim and independently as `wheel_sha256`."""
    fake = FakeSubprocess(
        freeze_stdout=(
            "onnxruntime-ep-vulkan @ file:///C:/dist/pkg.whl#sha256=deadbeefcafe\n"
        ),
    )
    drive(cleanroom, fake, monkeypatch, index_url=URL_ABSOLUTE)
    _raw, record = persisted(cleanroom)

    assert any("#REDACTED" in entry for entry in record["installed"])
    assert not any("sha256=deadbeefcafe" in entry for entry in record["installed"])
    assert len(record["wheel_sha256"]) == 64
    assert record["wheel_sha256"] == __import__("hashlib").sha256(
        cleanroom.wheel.read_bytes()).hexdigest()


# --- failure paths ---------------------------------------------------------------------


def test_main_failed_install_scrubs_pip_stderr_before_it_is_persisted(
        cleanroom, monkeypatch, capsys):
    fake = FakeSubprocess(
        install_rc=1,
        install_stderr=(
            f"Looking in indexes: {URL_ABSOLUTE}\n"
            f"WARNING: Retrying after connection broken by "
            f"NewConnectionError for {URL_SCHEME_RELATIVE}\n"
            f"ERROR: Could not install packages from {URL_QUERY}\n"
        ),
    )
    rc = drive(cleanroom, fake, monkeypatch, index_url=URL_ABSOLUTE)
    captured = capsys.readouterr()
    raw_json, record = persisted(cleanroom)

    assert rc == 1
    assert record["verdict"] == "UNOBSERVABLE"
    assert_no_sentinels("stdout", captured.out)
    assert_no_sentinels("stderr", captured.err)
    assert_no_sentinels("persisted record", raw_json)
    assert "REDACTED" in record["pip_stderr"]
    assert "Looking in indexes" in record["pip_stderr"]
    assert URL_ABSOLUTE in fake.argv_for("pip-install")


@pytest.mark.parametrize("offset", [0, 1, 11, len(URL_ABSOLUTE) // 2,
                                    len(URL_ABSOLUTE) - 1])
def test_main_failed_install_straddling_the_truncation_edge_persists_nothing(
        cleanroom, monkeypatch, capsys, offset):
    """B2 end-to-end: the credential sits across the point at which `pip_stderr` is
    truncated. Slice-then-scrub writes a password fragment into the tracked artifact."""
    fake = FakeSubprocess(
        install_rc=1,
        install_stderr=_straddling_stderr(URL_ABSOLUTE, offset),
    )
    drive(cleanroom, fake, monkeypatch, index_url=URL_ABSOLUTE)
    captured = capsys.readouterr()
    raw_json, record = persisted(cleanroom)

    assert_no_sentinels(f"persisted record (offset={offset})", raw_json)
    assert_no_sentinels(f"stdout (offset={offset})", captured.out)
    assert len(record["pip_stderr"]) <= vc._TAIL_CHARS


def test_main_unobservable_consumer_scrubs_stderr_tail(cleanroom, monkeypatch, capsys):
    fake = FakeSubprocess(
        consumer_stdout="no reading here\n",
        consumer_stderr=f"Traceback ...\nOSError: could not reach {URL_QUERY}\n",
    )
    rc = drive(cleanroom, fake, monkeypatch, index_url=URL_QUERY)
    captured = capsys.readouterr()
    raw_json, record = persisted(cleanroom)

    assert rc == 1
    assert record["verdict"] == "UNOBSERVABLE"
    assert record["reason"] == "the clean interpreter emitted no reading"
    assert_no_sentinels("stdout", captured.out)
    assert_no_sentinels("persisted record", raw_json)


def test_main_scrubs_the_child_interpreters_own_error_text(cleanroom, monkeypatch,
                                                           capsys):
    """The child's JSON payload is merged into the record wholesale. Whatever it says
    goes through the same scrub, because the seam is the write, not the assignment."""
    payload = {"verdict": "FAIL",
               "error": f"RuntimeError: index {URL_ABSOLUTE} rejected the token"}
    fake = FakeSubprocess(consumer_stdout="@@" + json.dumps(payload) + "\n")
    rc = drive(cleanroom, fake, monkeypatch, index_url=URL_ABSOLUTE)
    captured = capsys.readouterr()
    raw_json, record = persisted(cleanroom)

    assert rc == 1
    assert record["verdict"] == "FAIL"
    assert "REDACTED@mirror.example" in record["error"]
    assert_no_sentinels("stdout", captured.out)
    assert_no_sentinels("persisted record", raw_json)


# --- exception rendering (the surface that had no test at all) --------------------------


@pytest.mark.parametrize("url", [URL_ABSOLUTE, URL_SCHEME_RELATIVE, URL_SCHEMELESS,
                                 URL_QUERY, *SCHEMELESS_SPELLINGS])
def test_main_scrubs_an_exception_that_carries_the_index_url(cleanroom, monkeypatch,
                                                             capsys, url):
    fake = FakeSubprocess(
        raise_on="pip-install",
        raise_exc=OSError(f"[Errno 111] connection refused talking to {url}"),
    )
    with pytest.raises(SystemExit) as excinfo:
        drive(cleanroom, fake, monkeypatch, index_url=url)
    captured = capsys.readouterr()
    raw_json, record = persisted(cleanroom)

    assert_no_sentinels("SystemExit text", str(excinfo.value))
    # The original exception must be dropped, not chained: `raise ... from None`
    # would leave the raw message reachable through __context__.
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None
    assert_no_sentinels("stdout", captured.out)
    assert_no_sentinels("stderr", captured.err)
    assert_no_sentinels("persisted record", raw_json)
    assert record["verdict"] == "UNOBSERVABLE"
    assert "REDACTED" in record["error"]


def test_main_writes_the_record_even_when_it_raises(cleanroom, monkeypatch, capsys):
    fake = FakeSubprocess(raise_on="venv",
                          raise_exc=OSError(f"cannot create venv for {URL_ABSOLUTE}"))
    with pytest.raises(SystemExit):
        drive(cleanroom, fake, monkeypatch, index_url=URL_ABSOLUTE)
    assert cleanroom.record.is_file()


def test_main_persists_a_scrubbed_traceback_rather_than_swallowing_the_diagnosis(
        cleanroom, monkeypatch, capsys):
    """Dropping the exception object (so `__context__` cannot carry the raw message) must
    not also drop the diagnosis. The formatted traceback goes through the same one seam
    and is persisted, so the failing frame survives without the credential doing so."""
    fake = FakeSubprocess(
        raise_on="pip-install",
        raise_exc=OSError(f"[Errno 111] refused talking to {URL_ABSOLUTE}"),
    )
    with pytest.raises(SystemExit):
        drive(cleanroom, fake, monkeypatch, index_url=URL_ABSOLUTE)
    raw_json, record = persisted(cleanroom)

    assert "error_traceback" in record, "the traceback was swallowed"
    assert "Traceback (most recent call last)" in record["error_traceback"]
    assert "verify_cleanroom.py" in record["error_traceback"]
    assert_no_sentinels("persisted traceback", raw_json)


@pytest.mark.parametrize(
    "exc", [KeyboardInterrupt(), SystemExit(3), GeneratorExit()],
    ids=["KeyboardInterrupt", "SystemExit", "GeneratorExit"],
)
def test_main_does_not_trap_process_control_exceptions(cleanroom, monkeypatch, capsys,
                                                       exc):
    """`except BaseException` swallowed the three exceptions that are process control, not
    run outcomes: a diagnostic that cannot be interrupted, and that converts a caller's
    `sys.exit(3)` into its own exit text. The handler catches `Exception` instead, and the
    module's own fail-loud signal is an ordinary `Exception` so it still lands there."""
    fake = FakeSubprocess(raise_on="pip-install", raise_exc=exc)
    with pytest.raises(type(exc)) as excinfo:
        drive(cleanroom, fake, monkeypatch, index_url=URL_ABSOLUTE)
    assert excinfo.value is exc, "the process-control exception was replaced"


def test_the_modules_own_fail_loud_signal_is_still_recorded_and_scrubbed(
        cleanroom, monkeypatch, capsys):
    """The one `SystemExit` the old handler had to catch was this module's own
    "venv creation failed". It is now `_CleanroomFailure`, an ordinary `Exception`, so
    narrowing the handler did not cost the record."""
    class FailingVenv(FakeSubprocess):
        def run(self, cmd, **kw):
            argv = [str(c) for c in cmd]
            self.calls.append(argv)
            if self.kind(argv) == "venv":
                return subprocess.CompletedProcess(argv, 1, "", "")
            return super().run(cmd, **kw)

    with pytest.raises(SystemExit) as excinfo:
        drive(cleanroom, FailingVenv(), monkeypatch, index_url=URL_ABSOLUTE)
    raw_json, record = persisted(cleanroom)

    assert "venv creation failed" in str(excinfo.value)
    assert record["verdict"] == "UNOBSERVABLE"
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None
    assert_no_sentinels("persisted record", raw_json)


# --- issue #40 default / override behaviour must be preserved --------------------------


def test_main_default_index_url_is_the_approved_proxy_and_reaches_pip_verbatim(
        cleanroom, monkeypatch, capsys):
    fake = FakeSubprocess()
    drive(cleanroom, fake, monkeypatch)  # no --index-url at all
    _raw, record = persisted(cleanroom)

    install_argv = fake.argv_for("pip-install")
    assert install_argv[install_argv.index("--index-url") + 1] == vc.DEFAULT_INDEX_URL
    assert record["pip_index_url"] == vc.DEFAULT_INDEX_URL
    assert vc.DEFAULT_INDEX_URL in capsys.readouterr().out  # still readable in CI logs


def test_main_empty_index_url_falls_back_to_pips_own_default(cleanroom, monkeypatch,
                                                             capsys):
    fake = FakeSubprocess()
    drive(cleanroom, fake, monkeypatch, index_url="")
    _raw, record = persisted(cleanroom)

    assert "--index-url" not in fake.argv_for("pip-install")
    assert record["pip_index_url"] == "(pip default)"


def test_main_never_asks_for_a_shell(cleanroom, monkeypatch, capsys):
    """`FakeSubprocess.run` asserts this on every call; this test states it as the
    claim it is, and proves the calls actually happened."""
    fake = FakeSubprocess()
    drive(cleanroom, fake, monkeypatch, index_url=URL_ABSOLUTE)
    assert {FakeSubprocess.kind(a) for a in fake.calls} == {
        "venv", "pip-install", "pip-freeze", "consumer"}
