from harness import prepare, record_then_replay, run

SCENARIOS = """
import asyncio
import importlib
import importlib.util
import json

import openai
import pytest

from llmchaos import faults

FULL = "The refund was approved yesterday."
HALF = FULL[: len(FULL) // 2]
MESSAGES = [{"role": "user", "content": "Summarize ticket 42"}]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "unit": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]
PROTOCOL_ERRORS = tuple(
    importlib.import_module(name).RemoteProtocolError
    for name in ("httpx", "httpx2")
    if importlib.util.find_spec(name)
)
client = openai.OpenAI(api_key="test-key")


def chat(**kwargs):
    return client.chat.completions.create(model="gpt-5", messages=MESSAGES, **kwargs)


def chain(error):
    names = []
    while error is not None:
        names.append(type(error).__name__)
        error = error.__cause__ or error.__context__
    return names


def test_replay(llm):
    assert chat().choices[0].message.content == FULL
    assert (llm.calls, llm.attempts) == (1, 1)


def test_stream_replay(llm):
    text = "".join(c.choices[0].delta.content or "" for c in chat(stream=True) if c.choices)
    assert text == FULL


def test_async_replay(llm):
    async def go():
        async with openai.AsyncOpenAI(api_key="test-key") as aclient:
            return await aclient.chat.completions.create(model="gpt-5", messages=MESSAGES)

    assert asyncio.run(go()).choices[0].message.content == FULL


def test_rate_limit_recovers(llm):
    llm.inject(faults.rate_limit(times=2))
    assert chat().choices[0].message.content == FULL
    assert (llm.calls, llm.attempts) == (1, 3)


def test_rate_limit_gives_up(llm):
    llm.inject(faults.rate_limit(times=3))
    with pytest.raises(openai.RateLimitError, match="simulated by llmchaos"):
        chat()
    assert (llm.calls, llm.attempts) == (0, 3)


def test_overloaded_gives_up(llm):
    llm.inject(faults.overloaded(times=3))
    with pytest.raises(openai.InternalServerError) as error:
        chat()
    assert error.value.status_code == 503


def test_server_error_recovers(llm):
    llm.inject(faults.server_error())
    assert chat().choices[0].message.content == FULL
    assert llm.attempts == 2


def test_timeout_recovers(llm):
    llm.inject(faults.timeout())
    assert chat().choices[0].message.content == FULL
    assert llm.attempts == 2


def test_connection_error_without_retries(llm):
    llm.inject(faults.connection_error())
    with pytest.raises(openai.APIConnectionError):
        client.with_options(max_retries=0).chat.completions.create(model="gpt-5", messages=MESSAGES)


def test_stream_cut(llm):
    llm.inject(faults.stream_cut(after_events=2))
    received = []
    with pytest.raises((openai.APIConnectionError, *PROTOCOL_ERRORS)) as error:
        for chunk in chat(stream=True):
            received.append(chunk)
    assert len(received) == 2
    assert "RemoteProtocolError" in chain(error.value)


def test_async_stream_cut(llm):
    llm.inject(faults.stream_cut(after_events=2))

    async def go():
        async with openai.AsyncOpenAI(api_key="test-key") as aclient:
            stream = await aclient.chat.completions.create(
                model="gpt-5", messages=MESSAGES, stream=True
            )
            async for _ in stream:
                pass

    with pytest.raises((openai.APIConnectionError, *PROTOCOL_ERRORS)):
        asyncio.run(go())


def test_truncated(llm):
    llm.inject(faults.truncated())
    choice = chat().choices[0]
    assert (choice.message.content, choice.finish_reason) == (HALF, "length")


def test_truncated_stream(llm):
    llm.inject(faults.truncated())
    chunks = [c for c in chat(stream=True) if c.choices]
    assert "".join(c.choices[0].delta.content or "" for c in chunks) == HALF
    assert chunks[-1].choices[0].finish_reason == "length"


def test_bad_tool_args(llm):
    llm.inject(faults.bad_tool_args())
    call = chat(tools=TOOLS).choices[0].message.tool_calls[0]
    with pytest.raises(json.JSONDecodeError):
        json.loads(call.function.arguments)


def test_bad_tool_args_stream(llm):
    llm.inject(faults.bad_tool_args())
    arguments = "".join(
        tool_call.function.arguments or ""
        for c in chat(tools=TOOLS, stream=True)
        if c.choices
        for tool_call in c.choices[0].delta.tool_calls or []
        if tool_call.function
    )
    with pytest.raises(json.JSONDecodeError):
        json.loads(arguments)


def test_responses_api_replay(llm):
    response = client.responses.create(model="gpt-5", input="Summarize ticket 42")
    assert response.output_text == FULL


def test_responses_api_truncated(llm):
    llm.inject(faults.truncated())
    response = client.responses.create(model="gpt-5", input="Summarize ticket 42")
    assert (response.status, response.output_text) == ("incomplete", HALF)


def test_on_call_targets_the_second_call(llm):
    llm.inject(faults.rate_limit(on_call=2))
    chat()
    chat(tools=TOOLS)
    assert (llm.calls, llm.attempts) == (2, 3)
"""

MISMATCH = """
import openai


def test_chat(llm):
    client = openai.OpenAI(api_key="test-key")
    client.chat.completions.create(model="gpt-5", messages=[{"role": "user", "content": "PROMPT"}])
"""


def test_openai_scenarios_record_and_replay(pytester, monkeypatch):
    record_then_replay(pytester, monkeypatch, SCENARIOS, passed=18)


def test_openai_mismatch_reports_a_diff(pytester, monkeypatch):
    # Prompts of different lengths, so a same-second rewrite can't reuse stale bytecode.
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
