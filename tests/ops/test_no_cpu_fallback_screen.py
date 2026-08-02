"""Two-polarity screen for the no-CPU-fallback instruments, in the always-on lane.

WHY THIS FILE EXISTS SEPARATELY FROM ``test_no_cpu_fallback.py``
===============================================================
Tank's instrument census classified both guards as **unfalsified**:

    UNFALSIFIED tests/ops/_models.py::assert_ep_owns_whole_graph      reject=0 accept=0
    UNFALSIFIED tests/ops/_models.py::assert_no_cpu_fallback_is_live  reject=0 accept=0

which looks wrong at first glance, because `test_no_cpu_fallback.py` calls both of them in
both polarities.  It is not wrong.  Every one of those calls sits behind ``require_vulkan``,
and the census counts a polarity only from a test that is **not** GPU-gated — deliberately,
because a polarity nobody can observe without hardware is a polarity that has never been
observed on the machines where the census runs.  `calls=5` and `calls=2` say the guards have
callers; `reject=0 accept=0` says nothing has ever watched them *disagree* in the always-on
lane, so a guard that always passes, always raises, or has inverted polarity would look
exactly like a working one.  That is the census working, not the census confused.

WHAT IS SUBSTITUTED, AND WHAT IS THEREFORE STILL REAL
=====================================================
Only ORT is substituted, at the single seam where these instruments touch it
(``_models.ort.InferenceSession``) plus ``_models._make_session_options``, which is replaced
by a recorder so the test can *read back* which session-config entries the code under test
set.  Everything between — ``_no_cpu_fallback_options``, the key itself,
``ep_only_session_or_refusal``'s three-way classification of ORT's text, the canary graph
``assert_no_cpu_fallback_is_live`` builds, and both guards' own logic — is the real code.

Stated as an extent, because a screen that overstates its reach is the thing this project
keeps catching: **this file cannot tell you that ORT honours the key.** Only the hardware
lane can, and it does (`test_ort_refusal_is_live_and_not_a_silently_accepted_typo`).  What
this file can tell you is that *our* side of the mechanism produces two different answers
for two different worlds — which is the question `reject=0 accept=0` was asking.

THE TRAP, NAMED
===============
``assert_no_cpu_fallback_is_live`` is itself the falsifier for a silently-swallowed config
key, so a self-test for it must distinguish

    (a) the option takes effect, and the check says so, from
    (b) the check would say the option takes effect regardless.

A screen built only from arm (a) certifies (b).  Two arms answer it here:
``test_the_live_check_sets_the_key_it_claims_to_depend_on`` reads the recorded entries, and
``test_a_misspelled_key_is_not_reported_as_a_working_option`` mutates the key to a typo
against a fake that honours only the correctly-spelled one — the 2026-07-30 specimen,
reproduced without hardware.  Both of those fail against a check that always returns text.
"""

from __future__ import annotations

import pytest

import _models as m
import _verdict


# ---------------------------------------------------------------------------
# The seam
# ---------------------------------------------------------------------------


class _RecordingOptions:
    """Stands in for ``ort.SessionOptions``, recording what the code under test set.

    It records rather than validates on purpose: the assertion about which key was set
    belongs in the test that makes it, where the failure text can quote the recorded dict,
    not in a stub that could silently disagree with the census about what it saw.
    """

    def __init__(self) -> None:
        self.entries: dict[str, str] = {}
        self.log_severity_level = 3
        self.enable_profiling = False
        self.profile_file_prefix = ""

    def add_session_config_entry(self, key: str, value: str) -> None:
        self.entries[key] = value


#: ORT's real refusal text, quoted so a change in ORT's wording fails this screen rather
#: than being paraphrased into always matching.
_REAL_REFUSAL = (
    "[ONNXRuntimeError] : 1 : FAIL : This session contains graph nodes that are assigned "
    "to the default CPU EP, but fallback to CPU EP has been explicitly disabled by the "
    "user."
)
_REAL_CONFLICT = (
    "[ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Conflicting session configuration: "
    "explicitly added the CPU EP to the list of session providers, but also disabled "
    "fallback to the CPU EP via session options."
)


class _FakeOrt:
    """A stand-in for ORT with a stated policy, and a record of what it was handed."""

    def __init__(self, *, declines: bool, honours_key: bool = True, conflict: bool = False):
        self.declines = declines
        self.honours_key = honours_key
        self.conflict = conflict
        self.seen: list[_RecordingOptions] = []
        #: Snapshotted at construction, never read from the module at call time.  A fake
        #: that re-read the key would honour whatever the test just misspelled it to, and
        #: the typo arm would silently become a test of nothing.
        self.honoured_key = m.ORT_DISABLE_CPU_FALLBACK_KEY

    def __call__(self, model, opts, providers=None):
        self.seen.append(opts)
        entries = getattr(opts, "entries", {})
        armed = entries.get(self.honoured_key) == "1"
        if self.conflict:
            raise RuntimeError(_REAL_CONFLICT)
        if self.declines and armed and self.honours_key:
            raise RuntimeError(_REAL_REFUSAL)
        # Session created: either the graph was claimed in full, or the key did not take
        # effect.  Those two worlds are indistinguishable from here, which is exactly the
        # hazard `assert_no_cpu_fallback_is_live` exists to resolve one level up.
        return object()


@pytest.fixture
def substitute_ort(monkeypatch):
    """Install a fake ORT and a recording options factory; return the installer."""

    def install(fake: _FakeOrt) -> _FakeOrt:
        monkeypatch.setattr(m.ort, "InferenceSession", fake, raising=True)
        monkeypatch.setattr(
            m, "_make_session_options",
            lambda *, profiling=False, prefix="vulkan_test": _RecordingOptions(),
            raising=True,
        )
        return fake

    return install


#: A graph the fake never parses.  Said out loud rather than passed as ``b""``: the model
#: bytes are not part of what this file screens, and a reader must not infer that they are.
_MODEL_NOT_PARSED_BY_THE_FAKE = b"<model bytes; ORT is substituted and never parses them>"


# ---------------------------------------------------------------------------
# assert_no_cpu_fallback_is_live — both polarities
# ---------------------------------------------------------------------------


def test_the_live_check_returns_orts_text_when_the_option_takes_effect(substitute_ort):
    """Accept polarity: ORT refuses the fp64 canary, so the check reports the refusal."""
    substitute_ort(_FakeOrt(declines=True))
    text = m.assert_no_cpu_fallback_is_live()
    assert m._ORT_FALLBACK_TEXT in text, text


def test_the_live_check_raises_when_the_option_is_silently_inert(substitute_ort):
    """Reject polarity: the key is swallowed, so ORT builds a session it should have refused.

    This is the only world the guard exists for.  A guard that returned text here would
    leave every no-fallback precondition in the suite inert and green.
    """
    substitute_ort(_FakeOrt(declines=True, honours_key=False))
    with pytest.raises(_verdict.InstrumentError) as exc:
        m.assert_no_cpu_fallback_is_live()
    assert "accepts unknown session-config keys SILENTLY" in str(exc.value)


def test_the_live_check_sets_the_key_it_claims_to_depend_on(substitute_ort):
    """The 'regardless' arm: the check must be *asking* the thing it reports on.

    Reading the recorded entries is what separates "the option took effect and the check
    saw it" from "the check would have said so with no option set at all".  It is asserted
    on the recorder rather than on the return value because the return value is exactly
    what a check that never asked would still produce.
    """
    fake = substitute_ort(_FakeOrt(declines=True))
    m.assert_no_cpu_fallback_is_live()
    assert fake.seen, "ORT was never called — the check did not build a session at all"
    entries = fake.seen[-1].entries
    assert entries.get(m.ORT_DISABLE_CPU_FALLBACK_KEY) == "1", (
        f"the live check ran without arming {m.ORT_DISABLE_CPU_FALLBACK_KEY}; "
        f"recorded entries: {entries}"
    )


def test_a_misspelled_key_is_not_reported_as_a_working_option(substitute_ort, monkeypatch):
    """The 2026-07-30 specimen, without hardware: a typo must not read as a live option.

    The fake honours only the correctly-spelled key, which is ORT's measured behaviour —
    unknown keys are accepted silently.  With the key misspelled the session is created,
    and the guard must call that an instrument outage rather than a healthy run.
    """
    fake = substitute_ort(_FakeOrt(declines=True))
    monkeypatch.setattr(
        m, "ORT_DISABLE_CPU_FALLBACK_KEY", m.ORT_DISABLE_CPU_FALLBACK_KEY + "k",
        raising=True,
    )
    with pytest.raises(_verdict.InstrumentError):
        m.assert_no_cpu_fallback_is_live()
    assert fake.seen, "the typo arm did not even reach ORT, so it is not evidence about typos"
    entries = fake.seen[-1].entries
    assert fake.honoured_key not in entries, (
        "the misspelled arm still set the correctly-spelled key; it screens nothing"
    )
    assert entries.get(fake.honoured_key + "k") == "1", (
        f"the typo'd key never reached the session options: {entries}"
    )


# ---------------------------------------------------------------------------
# assert_ep_owns_whole_graph — both polarities
# ---------------------------------------------------------------------------


def test_owning_the_whole_graph_passes_when_nothing_fell_back(substitute_ort):
    """Accept polarity: a session ORT was willing to create is the whole observation."""
    substitute_ort(_FakeOrt(declines=False))
    assert m.assert_ep_owns_whole_graph(
        _MODEL_NOT_PARSED_BY_THE_FAKE, context="fully claimed"
    ) is None


def test_owning_the_whole_graph_fails_on_a_partially_claimed_graph(substitute_ort):
    """Reject polarity: ORT's refusal is FAIL(condition), carrying ORT's own text."""
    substitute_ort(_FakeOrt(declines=True))
    with pytest.raises(m.CpuFallbackRefused) as exc:
        m.assert_ep_owns_whole_graph(
            _MODEL_NOT_PARSED_BY_THE_FAKE, context="Add(f32) -> Cast(f64)"
        )
    assert m._ORT_FALLBACK_TEXT in exc.value.ort_text


def test_a_configuration_conflict_is_an_instrument_error_and_never_a_finding(substitute_ort):
    """R13's third state, screened: the conflict text fails EVERY graph, healthy ones too.

    If this were classified as a refusal it would be a false red on a run where the EP
    claimed the graph in full — a detection produced by our own misconfiguration, which is
    the one thing an instrument error may never be promoted into.
    """
    substitute_ort(_FakeOrt(declines=False, conflict=True))
    with pytest.raises(_verdict.InstrumentError) as exc:
        m.assert_ep_owns_whole_graph(_MODEL_NOT_PARSED_BY_THE_FAKE)
    assert "configuration conflict" in str(exc.value)
    assert not isinstance(exc.value, m.CpuFallbackRefused)


# ---------------------------------------------------------------------------
# arms_must_differ — the screen's own falsifier
# ---------------------------------------------------------------------------


def test_the_two_worlds_are_actually_different_worlds(substitute_ort):
    """Every arm above passes against a fake that answers the same way to everything.

    So the last thing asserted is that the fake does not: the *same* guard, given the two
    worlds, must produce two outcomes.  Without this the file would certify a guard by
    showing a stub agreeing with itself.
    """
    outcomes = []
    for declines in (False, True):
        substitute_ort(_FakeOrt(declines=declines))
        try:
            m.assert_ep_owns_whole_graph(_MODEL_NOT_PARSED_BY_THE_FAKE)
            outcomes.append("created")
        except m.CpuFallbackRefused:
            outcomes.append("refused")
    assert outcomes == ["created", "refused"], (
        f"arms_must_differ FAILED: the claimed and declined worlds both read {outcomes}. "
        "A guard screened against a fake that answers identically either way has not been "
        "screened; it has been kept company."
    )
