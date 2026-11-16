from harness import prepare, record_then_replay, run

SCENARIOS = """
import asyncio
import importlib
import importlib.util

import anthropic
import pytest

from llmchaos import faults

FULL = "The refund was approved yesterday."
HALF = FULL[: len(FULL) // 2]
MODEL = "claude-sonnet-5-5"
MESSAGES = [{"role": "user", "content": "Summarize ticket 42"}]
TOOLS = [
    {
        "name": "get_weather",
        "description": "Current weather for a city",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "unit": {"type": "string"}},
            "required": ["city"],
        },
    }
]
PROTOCOL_ERRORS = tuple(
    importlib.import_module(name).RemoteProtocolError
    for name in ("httpx", "httpx2")
    if importlib.util.find_spec(name)
)
client = anthropic.Anthropic(api_key="test-key")


def create(**kwargs):
    return client.messages.create(model=MODEL, max_tokens=256, messages=MESSAGES, **kwargs)


def stream(**kwargs):
    return client.messages.stream(model=MODEL, max_tokens=256, messages=MESSAGES, **kwargs)


def chain(error):
    names = []
    while error is not None:
        names.append(type(error).__name__)
        error = error.__cause__ or error.__context__
    return names


def test_replay(llm):
    assert create().content[0].text == FULL


def test_stream_helper_replay(llm):
    with stream() as s:
        assert "".join(s.text_stream) == FULL


def test_raw_stream_replay(llm):
    deltas = [
        event.delta.text
        for event in create(stream=True)
        if event.type == "content_block_delta" and event.delta.type == "text_delta"
    ]
    assert "".join(deltas) == FULL


def test_rate_limit_recovers(llm):
    llm.inject(faults.rate_limit(times=2))
    assert create().content[0].text == FULL
    assert (llm.calls, llm.attempts) == (1, 3)


def test_overloaded_gives_up(llm):
    llm.inject(faults.overloaded(times=3))
    with pytest.raises(anthropic.OverloadedError) as error:
        create()
    assert error.value.status_code == 529


def test_server_error_recovers(llm):
    llm.inject(faults.server_error())
    assert create().content[0].text == FULL
    assert llm.attempts == 2


def test_timeout_without_retries(llm):
    llm.inject(faults.timeout())
    with pytest.raises(anthropic.APITimeoutError):
        client.with_options(max_retries=0).messages.create(
            model=MODEL, max_tokens=256, messages=MESSAGES
        )


def test_connection_error_recovers(llm):
    llm.inject(faults.connection_error())
    assert create().content[0].text == FULL
    assert llm.attempts == 2


def test_stream_cut(llm):
    llm.inject(faults.stream_cut(after_events=3))
    received = []
    with pytest.raises((anthropic.APIConnectionError, *PROTOCOL_ERRORS)) as error:
        with stream() as s:
            for text in s.text_stream:
                received.append(text)
    assert received == ["The refund "]
    assert "RemoteProtocolError" in chain(error.value)


def test_truncated(llm):
    llm.inject(faults.truncated())
    message = create()
    assert (message.content[0].text, message.stop_reason) == (HALF, "max_tokens")


def test_truncated_stream(llm):
    llm.inject(faults.truncated())
    with stream() as s:
        message = s.get_final_message()
    assert (message.content[0].text, message.stop_reason) == (HALF, "max_tokens")


def test_bad_tool_args(llm):
    llm.inject(faults.bad_tool_args())
    tool = next(b for b in create(tools=TOOLS).content if b.type == "tool_use")
    assert tool.input == {"unit": "celsius"}


def test_bad_tool_args_stream(llm):
    llm.inject(faults.bad_tool_args())
    with stream(tools=TOOLS) as s:
        message = s.get_final_message()
    tool = next(b for b in message.content if b.type == "tool_use")
    assert tool.input == {"unit": "celsius"}


def test_async_overloaded_recovers(llm):
    llm.inject(faults.overloaded())

    async def go():
        async with anthropic.AsyncAnthropic(api_key="test-key") as aclient:
            return await aclient.messages.create(model=MODEL, max_tokens=256, messages=MESSAGES)

    assert asyncio.run(go()).content[0].text == FULL
    assert llm.attempts == 2
"""

MISMATCH = """
import anthropic


def test_chat(llm):
    client = anthropic.Anthropic(api_key="test-key")
    client.messages.create(
        model="claude-sonnet-5-5", max_tokens=256, messages=[{"role": "user", "content": "PROMPT"}]
    )
"""


def test_anthropic_scenarios_record_and_replay(pytester, monkeypatch):
    record_then_replay(pytester, monkeypatch, SCENARIOS, passed=14)


def test_anthropic_mismatch_reports_the_real_cause(pytester, monkeypatch):
    # anthropic retries our error and wraps it in APIConnectionError; the plugin must unwrap it.
    prepare(pytester, MISMATCH.replace("PROMPT", "Summarize ticket 42"))
    run(pytester, monkeypatch, offline=False, mode="all").assert_outcomes(passed=1)
    pytester.makepyfile(test_app=MISMATCH.replace("PROMPT", "Summarize ticket 1042"))
    result = run(pytester, monkeypatch, offline=True, mode="none")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(
        [
            "E*llmchaos.errors.CassetteMismatchError*",
            '*-*"content": "Summarize ticket 42"*',
            '*+*"content": "Summarize ticket 1042"*',
        ]
    )
    result.stdout.no_fnmatch_line("E*APIConnectionError*")
