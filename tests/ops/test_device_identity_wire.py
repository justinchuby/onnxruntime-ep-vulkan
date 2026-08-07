"""THE DEVICE-LIST WIRE FORMAT, END TO END (issue #18, blockers B1 and B2).

WHAT WAS WRONG, AND WHY NOTHING SAID SO
=======================================

`counters.rs` emitted `running_device_uuids` as a bare `"; "`-separated list::

    "uuid:aadf33d4d118155fcc60c22b5c352463"

`gen_proof_ledger.entry_line` read it with::

    [p.split("=", 1)[1].strip() for p in raw.split("; ") if "=" in p]

which *requires* an ``N=`` prefix. Against the emitted form that comprehension selects nothing,
so ``LedgerEntry.device_uuid`` was ``""`` on every entry the tool has ever written — and ``""``
is *also* the correct value for a run that reported no identity. So the defect had no symptom:
`registry::device_state` read the empty field, said DEVICE-UNATTRIBUTED, and was right about the
data it was given while being wrong about the world. Its two stable-identity verdicts, PROVEN and
PROVEN-ELSEWHERE, were unreachable code in a proof frame whose entire purpose is to reach them.

Both sides had unit tests. Both passed. Neither side's tests used a string the other side had
produced, which is the only kind of test that could have caught this.

WHAT THIS FILE DOES
===================

Every case comes from `tests/fixtures/device_identity_wire.jsonl`, which `rust/src/registry.rs`
reads too — one file, two languages, so the readers cannot drift apart again in silence.

1. `device_key_list` parses the emitted form (and tolerates the indexed form).
2. `device_uuid_from_counters` extracts exactly one identity, or refuses.
3. A **full `entry_line` round-trip** on the real RTX A1000 counters string, asserting the
   ``device_uuid`` field of the emitted JSON — the actual B1 contract, at the actual writer.
4. **The planted mutation**: the rejected parser is reimplemented here verbatim and required to
   fail on the emitted form. A repair is only demonstrated by a control that fails without it.

B2 rides along: every `unidentified:` key in the fixture is in the production spelling
``unidentified:<name>#<index>``, and the parse assertions would fail against the reversed form the
docs used to state.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
TOOLS = REPO / "rust" / "tools"
FIXTURE = REPO / "tests" / "fixtures" / "device_identity_wire.jsonl"

if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))


def _cases(kind):
    """The fixture rows of one kind, as (id, row) pairs for `pytest.mark.parametrize`."""
    assert FIXTURE.is_file(), (
        f"the shared wire fixture is missing: {FIXTURE}. It is read by this file AND by "
        f"rust/src/registry.rs; deleting it silently un-pins the two readers from each other."
    )
    out = []
    for line in FIXTURE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        row = json.loads(line)
        if row.get("kind") == kind:
            out.append(pytest.param(row, id=row["case"]))
    return out


WIRE = _cases("wire")
STATE = _cases("state")


def test_the_fixture_is_not_empty_and_covers_both_sides_of_the_comparison():
    """A fixture-driven suite whose fixture has been emptied passes vacuously.

    So the shape of the evidence is asserted before any of it is used: both kinds present, an
    identity-bearing case present, a refusal case present, and — for the A/B the third revision
    was asked for — a PROVEN case *and* a PROVEN-ELSEWHERE case. Without the last two, every
    assertion below could be satisfied by a parser that returns ``[]`` for everything, which is
    exactly the defect.
    """
    assert len(WIRE) >= 10, f"the wire rows were lost: {len(WIRE)}"
    assert len(STATE) >= 7, f"the state rows were lost: {len(STATE)}"

    rows = [p.values[0] for p in WIRE + STATE]
    assert any(r.get("device_uuid") for r in rows), "no case carries an identity at all"
    assert any(r.get("ledger_refuses") for r in rows), "no case exercises the writer's refusal"
    states = {r.get("state") for r in rows}
    assert "PROVEN" in states, "the A of the A/B is missing"
    assert "PROVEN-ELSEWHERE" in states, "the B of the A/B is missing"
    assert "DEVICE-UNATTRIBUTED" in states, "the fail-closed verdict is missing"

    # The one string that is evidence rather than opinion: it was copied out of a real run.
    real = [r for r in rows if r.get("case") == "real_rtx_a1000_single_device"]
    assert real, "the real-hardware row is what makes this a test of the format and not of itself"
    assert "=" not in real[0]["running_device_uuids"], (
        "the emitted form carries no equals sign — if this ever fails, the emitter changed and "
        "DESIGN.md §2.4.1.2 needs rewriting before this suite is repaired"
    )


@pytest.mark.parametrize("row", WIRE)
def test_the_emitted_counters_value_parses_to_the_stated_identities(row):
    """`device_key_list` is the ONE Python reader of the format; these are its obligations."""
    import gen_proof_ledger as g

    assert g.device_key_list(row["running_device_uuids"]) == row["keys"], (
        f"{row['case']}: {row['note']}"
    )
    assert g.device_key_list(row["running_device_names"]) == row["names"], (
        f"{row['case']}: names parse by the same rule as keys, or the two lists cannot be zipped"
    )


@pytest.mark.parametrize("row", WIRE)
def test_the_ledger_extracts_one_identity_or_refuses(row):
    """A proof entry names one device. Zero is honest; two is a refusal, never a choice."""
    import gen_proof_ledger as g

    if row.get("ledger_refuses"):
        with pytest.raises(SystemExit) as excinfo:
            g.device_uuid_from_counters(row["running_device_uuids"], row["case"])
        assert row["ledger_refuses"] in str(excinfo.value), (
            f"{row['case']}: the refusal must say what it saw — {row['note']}"
        )
        return

    got = g.device_uuid_from_counters(row["running_device_uuids"], row["case"])
    assert got == row["device_uuid"], f"{row['case']}: {row['note']}"
    if got:
        assert len(got) == 32 and all(c in "0123456789abcdef" for c in got), (
            f"{row['case']}: a device_uuid is 32 lowercase hex, bare — the `uuid:` prefix is the "
            f"wire form's, not the ledger field's"
        )


def _run(**over):
    """A minimal healthy proof-run record, matching `test_proof_ledger.py::_proof_run`."""
    run = {
        "worst_rel": 1e-6,
        "claimed_nodes": 1,
        "dispatches_executed": 1,
        "shaders_dispatched": ["ew_binary_add_f32"],
        "shaders_dispatched_digest": "0123456789abcdef",
        "shaders_dispatched_source_digest": "fedcba9876543210",
        "shader_toolchain": "shaderc v2026.2 v2026.2",
        "shaders_dispatched_spec_digest": "abcdef0123456789",
        "compute_calls": 1,
    }
    run.update(over)
    return run


def _entry(run):
    import gen_proof_ledger as g

    return json.loads(
        g.entry_line(
            "ai.onnx::Add/7+/f32,f32>f32/ew_binary_add_f32/static/n2",
            "device0",
            "1.20.0",
            "rtol=1e-3,atol=1e-5",
            1e-6,
            "0" * 64,
            "2026-08-02T00:00:00-07:00",
            run,
        )
    )


@pytest.mark.parametrize("row", WIRE)
def test_the_writer_stamps_the_identity_the_run_actually_opened(row):
    """END TO END, at the writer: counters value in, ledger `device_uuid` out.

    The two tests above exercise the parser. This one exercises `entry_line`, which is the
    function that was broken — a parser can be correct and still not be *called*, and the rejected
    revision's defect lived in the two lines where `entry_line` did its own parsing inline.
    """
    if row.get("ledger_refuses"):
        pytest.skip("refusal path is asserted by test_the_ledger_extracts_one_identity_or_refuses")

    entry = _entry(_run(
        running_device_uuids=row["running_device_uuids"],
        running_device_names=row["running_device_names"],
    ))
    assert entry["device_uuid"] == row["device_uuid"], (
        f"{row['case']}: the ledger field, not just the helper — {row['note']}"
    )
    # The name field must keep behaving as it did; this repair is about identity, and a repair
    # that quietly changes a second field is two changes wearing one commit message. The one
    # thing that DID change: a value naming no device (`""`, `none`, `unknown`, `0=`) leaves
    # the selector ordinal in place instead of being stamped in as if it were hardware.
    if row["names"]:
        assert entry["device"] == row["running_device_names"], (
            f"{row['case']}: device stays the reported string, verbatim; only device_uuid is new"
        )
    else:
        assert entry["device"] == "device0", (
            f"{row['case']}: a value naming no device leaves the ordinal, unattributed — "
            f"{row['running_device_names']!r} is not a GPU"
        )


def test_the_real_rtx_a1000_counters_string_produces_its_real_identity():
    """The single most important assertion in this file, stated on its own so it cannot be lost.

    `bench/results/rust-model-runner/mobilenetv2-12-device-selector.json` is a real MobileNetV2
    run on a real RTX A1000. Its `attribution.counters.running_device_uuids` is the string below,
    and its `execution_provider.selected_device.identity` is the same value — the EP and the
    ledger must agree about the device, through the wire format, with no hand-editing anywhere.
    """
    emitted = "uuid:aadf33d4d118155fcc60c22b5c352463"
    entry = _entry(_run(
        running_device_uuids=emitted,
        running_device_names="NVIDIA RTX A1000 Laptop GPU",
    ))
    assert entry["device_uuid"] == "aadf33d4d118155fcc60c22b5c352463"
    assert entry["device"] == "NVIDIA RTX A1000 Laptop GPU"
    assert entry["device_selector"] == "device0", (
        "the ordinal is kept beside the identity — it is still how to reproduce the run"
    )

    # And the artifact it was copied from still says so. A fixture that has drifted from the
    # artifact it quotes is a fixture that proves nothing about production.
    art = REPO / "bench" / "results" / "rust-model-runner" / "mobilenetv2-12-device-selector.json"
    if art.is_file():
        doc = json.loads(art.read_text(encoding="utf-8"))
        counters = doc.get("attribution", {}).get("counters", {})
        assert counters.get("running_device_uuids") == emitted, (
            "the real artifact no longer carries this string; the fixture is quoting a run that "
            "no longer exists"
        )


# ---------------------------------------------------------------------------
# THE PLANTED MUTATION
#
# `test-discipline`: a repair is demonstrated by a control that fails without it. Below is the
# rejected parser, reimplemented verbatim, required to be WRONG about the emitted form. If this
# test ever fails, either the emitter has changed or somebody has reintroduced the requirement —
# and in both cases the contract above is no longer describing production.
# ---------------------------------------------------------------------------


def _rejected_parser(raw: str) -> list[str]:
    """The parser Morpheus rejected, preserved exactly: it REQUIRES an `N=` prefix."""
    return [p.split("=", 1)[1].strip() for p in raw.split("; ") if "=" in p]


@pytest.mark.parametrize("row", WIRE)
def test_the_rejected_parser_reads_nothing_from_the_emitted_form(row):
    """Against every bare row, the rejected parser returns `[]` — silently, which is the point."""
    if row["case"] in ("indexed_tolerance", "index_with_no_body"):
        pytest.skip("written in the indexed shape on purpose; it is the one shape that parser read")
    if not row["keys"]:
        pytest.skip("nothing to lose: this row carries no identity under any parser")

    got = _rejected_parser(row["running_device_uuids"])
    assert got != row["keys"], (
        f"{row['case']}: the rejected parser must FAIL here. If it passes, the emitted wire "
        f"format has changed and DESIGN.md §2.4.1.2 is stale."
    )
    assert got == [], (
        f"{row['case']}: and it fails by reading NOTHING — an empty list, indistinguishable from "
        f"'this run opened no device'. That indistinguishability is why the bug survived review."
    )


def test_the_writer_under_the_rejected_parser_loses_the_identity(monkeypatch):
    """The mutation driven all the way through `entry_line`, on the real hardware string.

    Monkeypatching `device_key_list` is the closest reachable equivalent of reverting the repair:
    it puts the rejected rule back underneath the unchanged writer. The identity must vanish, and
    it must vanish *quietly* — no exception, no warning, just `""` where a UUID belongs.
    """
    import gen_proof_ledger as g

    emitted = "uuid:aadf33d4d118155fcc60c22b5c352463"
    assert _entry(_run(running_device_uuids=emitted))["device_uuid"] == (
        "aadf33d4d118155fcc60c22b5c352463"
    ), "polarity control: the repaired writer must succeed before its failure means anything"

    monkeypatch.setattr(g, "device_key_list", _rejected_parser)
    mutated = _entry(_run(running_device_uuids=emitted))
    assert mutated["device_uuid"] == "", (
        "under the rejected parser the identity must be lost — if it survives, this test is not "
        "reverting the repair and proves nothing"
    )
    assert "device_uuid" in mutated, (
        "and the field is still THERE, holding an empty string: the entry looks well-formed, "
        "`registry::device_state` reads DEVICE-UNATTRIBUTED, and no instrument fires. That is the "
        "whole shape of blocker B1."
    )


# ---------------------------------------------------------------------------
# B2 — THE FALLBACK SPELLING
# ---------------------------------------------------------------------------


def test_the_unidentified_fallback_uses_the_production_spelling_and_never_proves():
    """`unidentified:<name>#<index>`, name first, index last, `#` and not a second colon.

    The docs and the evidence fixtures said `unidentified:<index>:<name>` — reversed. Nothing
    broke, because nothing parsed the string back apart; the spelling was decorative until
    `DeviceKey::from_canonical` had to reverse it, and a name may contain a colon while `#<digits>`
    at end-of-string is always unambiguous.

    The load-bearing half is the second assertion: whatever it is spelled, it must never yield a
    ledger identity. A process-local key attributes nothing outside the process that observed it.
    """
    import gen_proof_ledger as g

    rows = [p.values[0] for p in WIRE + STATE]
    unident = [
        r for r in rows
        if "unidentified:" in (r.get("running_device_uuids") or "")
    ]
    assert unident, "the fallback lost its coverage entirely"

    for row in unident:
        for key in g.device_key_list(row["running_device_uuids"]):
            if not key.startswith("unidentified:"):
                continue
            body = key[len("unidentified:"):]
            assert "#" in body, f"{row['case']}: {key!r} must be <name>#<index>"
            name, _, index = body.rpartition("#")
            assert name, f"{row['case']}: the NAME comes first — {key!r}"
            assert index.isdigit(), f"{row['case']}: the INDEX comes last — {key!r}"
            assert not body.split("#")[0].isdigit() or "#" in name, (
                f"{row['case']}: {key!r} is in the reversed `unidentified:<index>:<name>` form "
                f"the docs used to state"
            )

        if not row.get("ledger_refuses"):
            got = g.device_uuid_from_counters(row["running_device_uuids"], row["case"])
            assert got == row.get("device_uuid", ""), (
                f"{row['case']}: an unidentified key contributes no identity to the ledger"
            )
            assert "unidentified" not in got, (
                f"{row['case']}: and it must never leak into the field verbatim — a "
                f"`device_uuid` that is not 32 hex is not an identity, it is a story"
            )
