"""Criterion 10 is a claim about a *run*, and since `872d739` a run has two possible routes.

WHY THIS FILE EXISTS
====================
Switch's `872d739` made a directly-written device buffer **authoritative**: a span the
dispatch wrote is now marked so that every later reader downloads it instead of reading a
staging block nobody wrote.  That gives the 64 KV `present` tensors a second way out of
the fused island.  Criterion 10 says *every output agrees with the CPU oracle* — but a run
down one route is silent about the other, and a record that does not name its route
describes a run nobody can identify.

MEASURED, and the reason this file is not just bookkeeping
==========================================================
The criterion-10 lane's default run does **not** take the new route.  On the merged tree
at `6ef62bb`, Phi-3.5, both devices:

    default (ONNXRUNTIME_EP_VULKAN_BIND_OUTPUTS unset)
        outputs_device_bound = 0, outputs_host_resident = 196,
        alloc_device_authority_grants = 0        -> HOST_STAGING

    ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY=1 ONNXRUNTIME_EP_VULKAN_BIND_OUTPUTS=1
        outputs_device_bound = 196, outputs_host_resident = 0,
        alloc_device_authority_grants = 196,
        alloc_device_downloads = 196 / 1_372_096 B -> DEVICE_AUTHORITATIVE

So "Switch made the KV tensors device-resident, therefore criterion 10's code path
changed" is **false for the run criterion 10 actually performs**, and true only for a run
nobody was making.  That is a fact about the run, and it is recoverable only if the run
records it.

THE ROUTE IS READ OFF THE RUN, NEVER OFF THE ENV VAR
====================================================
`BIND_OUTPUTS=1` is a *request*.  `vk/session.rs` Step 1c **unbinds on refusal** — a span
with no VkBuffer cannot be made authoritative and the bind is dropped.  A record that
inferred the route from the environment would report a route that was declined, which is
the same class of error as reading the device from the selector instead of from the run.
"""

from __future__ import annotations

import pytest

import test_criterion10 as c10


# -- the route -----------------------------------------------------------------------


def test_no_bound_output_reads_as_the_host_staging_route():
    route = c10.kv_writeback_route(
        {
            "outputs_device_bound": 0,
            "outputs_host_resident": 196,
            "alloc_device_authority_grants": 0,
        }
    )
    assert route["route"] == c10.ROUTE_HOST_STAGING


def test_every_output_bound_reads_as_the_device_authoritative_route():
    route = c10.kv_writeback_route(
        {
            "outputs_device_bound": 196,
            "outputs_host_resident": 0,
            "alloc_device_authority_grants": 196,
            "alloc_device_downloads": 196,
        }
    )
    assert route["route"] == c10.ROUTE_DEVICE_AUTHORITATIVE
    assert route["alloc_device_authority_grants"] == 196


def test_a_partly_bound_run_is_MIXED_and_not_silently_one_of_the_two():
    """The case a two-token scheme would round to whichever token it was written for.

    `vk/session.rs` unbinds a span it cannot make authoritative, so *some bound and some
    staged* is a state the machine can actually reach — and it is the state in which "the
    route was X" is false in both directions.
    """
    route = c10.kv_writeback_route(
        {"outputs_device_bound": 6, "outputs_host_resident": 190}
    )
    assert route["route"] == c10.ROUTE_MIXED


@pytest.mark.parametrize(
    "counters",
    [None, {}, {"outputs_device_bound": 196}, {"outputs_host_resident": "196"}],
)
def test_a_run_that_did_not_report_its_route_is_UNOBSERVABLE_and_not_the_default(counters):
    """R12, on the counter that has no reading.

    `HOST_STAGING` is what an unarmed run looks like if the absent counter is read as 0 —
    and it is also the correct answer for a genuine default run, so the two would be
    indistinguishable in the record.  A binary built before `872d739` emits no
    `alloc_device_authority_grants` at all; that run must not be recorded as having taken
    the old route *by measurement* when it was never measured.
    """
    assert c10.kv_writeback_route(counters)["route"] == c10.ROUTE_UNOBSERVABLE


def test_the_route_is_not_derived_from_the_environment_variable(monkeypatch):
    """A declined bind must not be recorded as a bind.

    Asserted by asking for the route with the request set and the run reporting nothing
    bound: the answer must follow the run.
    """
    monkeypatch.setenv("ONNXRUNTIME_EP_VULKAN_BIND_OUTPUTS", "1")
    monkeypatch.setenv("ONNXRUNTIME_EP_VULKAN_DEVICE_MEMORY", "1")
    route = c10.kv_writeback_route(
        {"outputs_device_bound": 0, "outputs_host_resident": 196}
    )
    assert route["route"] == c10.ROUTE_HOST_STAGING, (
        "the route was taken from the request rather than from the run; a bind the EP "
        "declined would be recorded as a route that was taken"
    )


# -- the device name -----------------------------------------------------------------


def test_the_device_name_is_the_one_the_run_reported():
    assert (
        c10.device_name_from_run(
            {"alloc_device_frame_session_devices": "1=NVIDIA GeForce RTX 4060 Laptop GPU"}
        )
        == "NVIDIA GeForce RTX 4060 Laptop GPU"
    )


def test_the_enumeration_index_in_front_of_the_name_is_not_the_selector():
    """The index the allocator prints is its own enumeration order, not the selector.

    On this box selector **0** reports ``1=NVIDIA…`` and selector **1** reports
    ``0=Intel…``.  A reader who takes the leading digit for the selector reproduces the
    label swap this project already published once.
    """
    assert (
        c10.device_name_from_run(
            {"alloc_device_frame_session_devices": "0=Intel(R) Iris(R) Xe Graphics"}
        )
        == "Intel(R) Iris(R) Xe Graphics"
    )


@pytest.mark.parametrize("raw", [None, "", "NVIDIA GeForce RTX 4060", 1, {}])
def test_a_run_that_did_not_name_its_device_says_so_rather_than_reporting_an_empty_string(raw):
    """The state this replaces: the record carried ``device_name: ""``.

    An empty string is not a refusal — it renders as "no name" and invites the reader to
    fall back to the selector, which is the one source the whole helper exists to avoid.
    """
    assert (
        c10.device_name_from_run({"alloc_device_frame_session_devices": raw})
        == c10.ROUTE_UNOBSERVABLE
    )


def test_labelling_never_raises_because_the_record_is_written_on_the_failing_path_too():
    """A criterion whose evidence only exists when it passes is not evidence.

    The criterion-10 record is written before the assertion, precisely so a failing series
    is recorded.  If labelling the reading could raise, a red run would lose its artifact —
    so both readers return a token instead.
    """
    for counters in (None, {}, {"alloc_device_frame_session_devices": None}):
        assert isinstance(c10.device_name_from_run(counters), str)
        assert isinstance(c10.kv_writeback_route(counters)["route"], str)
