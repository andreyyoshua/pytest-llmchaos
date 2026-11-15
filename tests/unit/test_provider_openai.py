import json

import payloads
import pytest

from llmchaos.cassette import RecordedResponse
from llmchaos.errors import FaultNotApplicableError
from llmchaos.providers.base import SIMULATED, parse_json_events
from llmchaos.providers.openai import OpenAIProvider
from llmchaos.sse import SSEEvent

PROVIDER = OpenAIProvider()
HALF = payloads.FULL_TEXT[: len(payloads.FULL_TEXT) // 2]


def stream_text(events):
    chunks = parse_json_events(events).values()
    return "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)


def stream_arguments(events):
    return "".join(
        call["function"].get("arguments") or ""
        for chunk in parse_json_events(events).values()
        for call in chunk["choices"][0]["delta"].get("tool_calls") or []
    )


def test_handles_default_and_extra_hosts():
    assert PROVIDER.handles("api.openai.com")
    assert not PROVIDER.handles("api.anthropic.com")
    assert OpenAIProvider(["gateway.example.com"]).handles("gateway.example.com")


@pytest.mark.parametrize(
    ("kind", "status"), [("rate_limit", 429), ("overloaded", 503), ("server_error", 500)]
)
def test_error_responses(kind, status):
    response = PROVIDER.error_response(kind, 0.001)
    assert response.status == status
    assert set(response.body["error"]) == {"message", "type", "param", "code"}
    assert response.body["error"]["message"].endswith(SIMULATED)
    assert response.headers["retry-after-ms"] == "1"
    assert response.headers["retry-after"] == "1"


def test_truncate_chat_completion_without_touching_the_original():
    original = payloads.openai_chat()
    choice = PROVIDER.truncate(original).body["choices"][0]
    assert choice["message"]["content"] == HALF
    assert choice["finish_reason"] == "length"
    assert original.body["choices"][0]["finish_reason"] == "stop"


def test_truncate_chat_stream():
    events = PROVIDER.truncate(payloads.openai_chat_stream()).events
    assert stream_text(events) == HALF
    reasons = [
        c["choices"][0]["finish_reason"]
        for c in parse_json_events(events).values()
        if c["choices"][0]["finish_reason"]
    ]
    assert reasons == ["length"]
    assert events[-1] == SSEEvent(data="[DONE]")


def test_truncate_responses_api():
    body = PROVIDER.truncate(payloads.openai_response()).body
    assert body["output"][0]["content"][0]["text"] == HALF
    assert body["status"] == "incomplete"
    assert body["incomplete_details"] == {"reason": "max_output_tokens"}


def test_truncate_needs_text():
    with pytest.raises(FaultNotApplicableError, match="text content"):
        PROVIDER.truncate(payloads.openai_chat_tool())


def test_responses_api_streams_are_not_supported():
    stream = RecordedResponse(
        200,
        {"content-type": "text/event-stream"},
        events=[SSEEvent(data='{"type":"response.created"}', event="response.created")],
    )
    with pytest.raises(FaultNotApplicableError, match="Responses API streaming"):
        PROVIDER.truncate(stream)


def test_break_tool_args_chat():
    message = PROVIDER.break_tool_args(payloads.openai_chat_tool()).body["choices"][0]["message"]
    with pytest.raises(json.JSONDecodeError):
        json.loads(message["tool_calls"][0]["function"]["arguments"])


def test_break_tool_args_chat_stream():
    events = PROVIDER.break_tool_args(payloads.openai_chat_tool_stream()).events
    assert stream_arguments(events) == json.dumps(payloads.TOOL_ARGUMENTS)[:-1]


def test_break_tool_args_responses_api():
    output = PROVIDER.break_tool_args(payloads.openai_response_tool()).body["output"]
    assert output[0]["arguments"] == json.dumps(payloads.TOOL_ARGUMENTS)[:-1]


def test_break_tool_args_needs_a_tool_call():
    with pytest.raises(FaultNotApplicableError, match="tool call"):
        PROVIDER.break_tool_args(payloads.openai_chat())
