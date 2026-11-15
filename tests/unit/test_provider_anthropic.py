import json

import payloads
import pytest

from llmchaos.cassette import RecordedResponse
from llmchaos.errors import FaultNotApplicableError
from llmchaos.providers.anthropic import AnthropicProvider
from llmchaos.providers.base import SIMULATED, parse_json_events

PROVIDER = AnthropicProvider()
HALF = payloads.FULL_TEXT[: len(payloads.FULL_TEXT) // 2]


def stream_text(events):
    return "".join(
        d["delta"]["text"]
        for d in parse_json_events(events).values()
        if d["type"] == "content_block_delta" and d["delta"]["type"] == "text_delta"
    )


def test_handles_default_and_extra_hosts():
    assert PROVIDER.handles("api.anthropic.com")
    assert not PROVIDER.handles("api.openai.com")
    assert AnthropicProvider(["proxy.example.com"]).handles("proxy.example.com")


@pytest.mark.parametrize(
    ("kind", "status", "error_type"),
    [
        ("rate_limit", 429, "rate_limit_error"),
        ("overloaded", 529, "overloaded_error"),
        ("server_error", 500, "api_error"),
    ],
)
def test_error_responses(kind, status, error_type):
    response = PROVIDER.error_response(kind, 2.5)
    assert response.status == status
    assert response.body["type"] == "error"
    assert response.body["error"]["type"] == error_type
    assert response.body["error"]["message"].endswith(SIMULATED)
    assert response.headers["retry-after-ms"] == "2500"
    assert response.headers["retry-after"] == "3"


def test_truncate_message():
    body = PROVIDER.truncate(payloads.anthropic_message()).body
    assert body["content"] == [{"type": "text", "text": HALF}]
    assert body["stop_reason"] == "max_tokens"


def test_truncate_drops_blocks_after_the_text():
    body = PROVIDER.truncate(payloads.anthropic_tool_message()).body
    assert body["content"] == [{"type": "text", "text": "Let me"}]


def test_truncate_stream():
    events = PROVIDER.truncate(payloads.anthropic_stream()).events
    assert stream_text(events) == HALF
    deltas = [d for d in parse_json_events(events).values() if d["type"] == "message_delta"]
    assert [d["delta"]["stop_reason"] for d in deltas] == ["max_tokens"]
    assert events[-1].event == "message_stop"


def test_truncate_needs_text():
    with pytest.raises(FaultNotApplicableError, match="text content"):
        PROVIDER.truncate(payloads.anthropic_tool_stream())


def test_break_tool_args_drops_the_first_input_key():
    body = PROVIDER.break_tool_args(payloads.anthropic_tool_message()).body
    assert body["content"][1]["input"] == {"unit": "celsius"}


def test_break_tool_args_stream_replaces_the_json_deltas():
    events = PROVIDER.break_tool_args(payloads.anthropic_tool_stream()).events
    deltas = [
        d["delta"]["partial_json"]
        for d in parse_json_events(events).values()
        if d["type"] == "content_block_delta"
    ]
    assert deltas == [json.dumps({"unit": "celsius"})]


def test_break_tool_args_needs_a_tool_call():
    with pytest.raises(FaultNotApplicableError, match="tool call"):
        PROVIDER.break_tool_args(payloads.anthropic_message())


def test_only_messages_api_bodies_are_supported():
    with pytest.raises(FaultNotApplicableError, match="Messages API"):
        PROVIDER.truncate(RecordedResponse.of_json(200, {"type": "completion"}))
