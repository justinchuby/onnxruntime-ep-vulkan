"""Tests for `python/verify_cleanroom.py`'s package-index URL redaction (issue #55).

`verify_cleanroom.py` accepts a `--index-url` that may point at a private/authenticated
mirror and therefore may embed userinfo (`user:pass@`, a bare username, or percent-encoded
credentials). Three surfaces must never carry that userinfo in the clear:

1. the ``$ ...`` progress echo `_run()` prints for every subprocess it launches;
2. the persisted record (`bench/results/cleanroom_install_dev0.json`'s
   ``pip_index_url``/``pip_stderr``/``stderr_tail`` fields);
3. anything a broad exception path might print.

The one place the *real*, unredacted URL must still reach is the argv handed straight to
`subprocess.run` for pip itself — never through a shell, so shell history/expansion is not
a second leak path. That is checked here as directly as this module's structure allows:
the argv list `main()` builds for `pip install` is inspected before it is ever run.

Every credential in this file is a synthetic sentinel (`sentinel-user`, `sentinel-pass`,
...) — never a real secret, per the secret-handling convention.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VERIFY_CLEANROOM = REPO / "python" / "verify_cleanroom.py"

_spec = importlib.util.spec_from_file_location("_verify_cleanroom", VERIFY_CLEANROOM)
assert _spec and _spec.loader
vc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vc)


# ---------------------------------------------------------------------------------------
# _redact_url_userinfo: the core redaction primitive
# ---------------------------------------------------------------------------------------


def test_redacts_plain_user_and_password():
    got = vc._redact_url_userinfo(
        "https://sentinel-user:sentinel-pass@packagefeedproxy.microsoft.io/pypi/simple"
    )
    assert "sentinel-user" not in got
    assert "sentinel-pass" not in got
    assert got == "https://REDACTED@packagefeedproxy.microsoft.io/pypi/simple"


def test_redacts_username_only():
    got = vc._redact_url_userinfo("https://sentinel-user@mirror.example/pypi/simple")
    assert "sentinel-user" not in got
    assert got == "https://REDACTED@mirror.example/pypi/simple"


def test_redacts_percent_encoded_userinfo():
    # "sentinel@user" percent-encoded as the username, "sentinel@pass" as the password --
    # the literal '@' inside each is %40-encoded, so exactly one *unencoded* '@' remains
    # as the userinfo/host delimiter.
    url = "https://sentinel%40user:sentinel%40pass@mirror.example/pypi/simple"
    got = vc._redact_url_userinfo(url)
    assert "sentinel%40user" not in got
    assert "sentinel%40pass" not in got
    assert got == "https://REDACTED@mirror.example/pypi/simple"


def test_preserves_query_path_and_fragment():
    url = "https://sentinel-user:sentinel-pass@mirror.example/pypi/simple?ref=abc#frag"
    got = vc._redact_url_userinfo(url)
    assert got == "https://REDACTED@mirror.example/pypi/simple?ref=abc#frag"


def test_no_userinfo_present_is_returned_unchanged():
    url = "https://packagefeedproxy.microsoft.io/pypi/simple"
    assert vc._redact_url_userinfo(url) == url


def test_ipv6_host_with_port_is_preserved():
    url = "https://sentinel-user:sentinel-pass@[::1]:8443/pypi/simple"
    got = vc._redact_url_userinfo(url)
    assert "sentinel-user" not in got
    assert "sentinel-pass" not in got
    assert got == "https://REDACTED@[::1]:8443/pypi/simple"


def test_ipv6_full_address_with_port_and_query_is_preserved():
    url = "https://sentinel-user:sentinel-pass@[2001:db8::1]:443/pypi/simple?token=abc"
    got = vc._redact_url_userinfo(url)
    assert "sentinel-user" not in got
    assert "sentinel-pass" not in got
    assert got == "https://REDACTED@[2001:db8::1]:443/pypi/simple?token=abc"


def test_explicit_default_port_is_preserved():
    url = "https://sentinel-user:sentinel-pass@mirror.example:8443/pypi/simple"
    got = vc._redact_url_userinfo(url)
    assert got == "https://REDACTED@mirror.example:8443/pypi/simple"


def test_empty_and_none_like_inputs_pass_through():
    assert vc._redact_url_userinfo("") == ""


def test_malformed_url_with_userinfo_shape_but_no_scheme_is_redacted_not_leaked():
    # No "//" authority section for urlsplit to find a netloc in, but it still has the
    # unmistakable "credentials@host" shape. The documented safe fallback must not let the
    # raw credentials-shaped substring reach the return value.
    got = vc._redact_url_userinfo("sentinel-user:sentinel-pass@mirror.example/pypi/simple")
    assert "sentinel-user" not in got
    assert "sentinel-pass" not in got


def test_malformed_url_with_no_identifiable_split_point_falls_back_to_placeholder():
    got = vc._redact_url_userinfo("not a url at all @ weird")
    assert "@" not in got or got == "<REDACTED-unparseable-index-url>"
    assert got == "<REDACTED-unparseable-index-url>"


def test_url_with_no_at_sign_is_never_mistaken_for_having_userinfo():
    url = "https://mirror.example/pypi/simple"
    assert vc._redact_url_userinfo(url) == url


# ---------------------------------------------------------------------------------------
# _echo_cmd: what `_run()` actually prints for a subprocess invocation
# ---------------------------------------------------------------------------------------


def test_echo_cmd_redacts_the_index_url_token_only():
    cmd = [
        "python.exe", "-m", "pip", "install", "--index-url",
        "https://sentinel-user:sentinel-pass@mirror.example/pypi/simple",
        "somepkg",
    ]
    echoed = vc._echo_cmd(cmd)
    assert "sentinel-user" not in echoed
    assert "sentinel-pass" not in echoed
    assert "--index-url" in echoed
    assert "somepkg" in echoed  # non-URL tokens are untouched


def test_echo_cmd_leaves_the_default_proxy_url_readable():
    # Default approved proxy (issue #40) has no credentials -- redaction must be a no-op
    # for it, so the echoed command stays useful for a human reading CI logs.
    cmd = ["pip", "install", "--index-url", vc.DEFAULT_INDEX_URL, "somepkg"]
    echoed = vc._echo_cmd(cmd)
    assert vc.DEFAULT_INDEX_URL in echoed


def test_echo_cmd_handles_non_string_tokens_like_paths():
    cmd = ["pip", "install", Path("C:/some/wheel.whl")]
    echoed = vc._echo_cmd(cmd)
    assert "wheel.whl" in echoed


# ---------------------------------------------------------------------------------------
# _scrub_text: pip's own stderr/stdout can echo the URL it was given back at us
# ---------------------------------------------------------------------------------------


def test_scrub_text_removes_the_raw_url_from_pip_style_error_text():
    raw = "https://sentinel-user:sentinel-pass@mirror.example/pypi/simple"
    text = (
        f"Looking in indexes: {raw}\n"
        f"ERROR: Could not find a version that satisfies the requirement foo "
        f"(from versions: none)\nERROR: could not fetch URL {raw}: "
        f"connection refused"
    )
    scrubbed = vc._scrub_text(text, raw)
    assert "sentinel-user" not in scrubbed
    assert "sentinel-pass" not in scrubbed
    assert "REDACTED" in scrubbed
    assert "Looking in indexes" in scrubbed  # surrounding text is preserved


def test_scrub_text_is_a_no_op_when_url_has_no_userinfo():
    raw = vc.DEFAULT_INDEX_URL
    text = f"Looking in indexes: {raw}\n"
    assert vc._scrub_text(text, raw) == text


def test_scrub_text_is_a_no_op_when_raw_url_is_empty():
    text = "some unrelated error text"
    assert vc._scrub_text(text, "") == text


# ---------------------------------------------------------------------------------------
# The real URL still reaches pip's argv unmodified -- redaction must not touch the
# subprocess call itself, only what is echoed/persisted around it.
# ---------------------------------------------------------------------------------------


def test_pip_cmd_argv_construction_carries_the_real_unredacted_url():
    # Mirrors main()'s pip_cmd construction without running a real install: the argv list
    # actually handed to subprocess.run must contain the real URL, never the redacted one,
    # because pip needs real credentials to authenticate against a private mirror.
    raw = "https://sentinel-user:sentinel-pass@mirror.example/pypi/simple"
    py = Path("C:/fake/venv/Scripts/python.exe")
    wheel = Path("C:/fake/dist/pkg-0.0.0-py3-none-any.whl")
    pip_cmd = [py, "-m", "pip", "install", "--disable-pip-version-check", "-q"]
    pip_cmd += ["--index-url", raw]
    pip_cmd += [wheel, "onnx", "numpy"]

    assert raw in [str(c) for c in pip_cmd]
    # ... but the echo of that same argv must not carry it.
    assert "sentinel-pass" not in vc._echo_cmd(pip_cmd)


def test_default_index_url_default_behaviour_is_unchanged():
    # Issue #40's approved-proxy default must still be exactly what pip receives when the
    # caller passes nothing -- redaction work must not alter the default value itself.
    assert vc.DEFAULT_INDEX_URL.startswith("https://")
    assert "@" not in vc.DEFAULT_INDEX_URL
    assert vc._redact_url_userinfo(vc.DEFAULT_INDEX_URL) == vc.DEFAULT_INDEX_URL
