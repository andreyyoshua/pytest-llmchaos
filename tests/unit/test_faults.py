import dataclasses

import pytest

import llmchaos
from llmchaos import faults


def test_constructors_build_the_expected_faults():
    assert faults.rate_limit(times=2, retry_after=1.5, on_call=3) == faults.HTTPError(
        kind="rate_limit", times=2, retry_after=1.5, on_call=3
    )
    assert faults.overloaded() == faults.HTTPError(kind="overloaded", times=1, retry_after=0.001)
    assert faults.server_error(times=4) == faults.HTTPError(kind="server_error", times=4)
    assert faults.timeout() == faults.TransportError(kind="timeout")
    assert faults.connection_error(times=2) == faults.TransportError(
        kind="connection_error", times=2
    )
    assert faults.stream_cut(after_events=0) == faults.StreamCut(after_events=0)
    assert faults.truncated(on_call=2) == faults.Truncated(on_call=2)
    assert faults.bad_tool_args() == faults.BadToolArgs()


def test_kinds_of_fault():
    assert isinstance(faults.rate_limit(), faults.ReplaceFault)
    assert isinstance(faults.timeout(), faults.ReplaceFault)
    for mutate in (faults.stream_cut(), faults.truncated(), faults.bad_tool_args()):
        assert isinstance(mutate, faults.MutateFault)


@pytest.mark.parametrize(
    "build",
    [
        lambda: faults.rate_limit(times=0),
        lambda: faults.timeout(on_call=0),
        lambda: faults.rate_limit(retry_after=-1),
        lambda: faults.stream_cut(after_events=-1),
    ],
)
def test_invalid_arguments_are_rejected(build):
    with pytest.raises(ValueError):
        build()


def test_faults_are_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        faults.timeout().times = 3  # type: ignore[misc]


def test_faults_module_is_exported():
    assert llmchaos.faults is faults
