"""Canonical provider payloads shared by unit tests, the fake network, and examples."""

from __future__ import annotations

import json
from typing import Any

from llmchaos.cassette import RecordedResponse
from llmchaos.sse import SSEEvent

FULL_TEXT = "The refund was approved yesterday."
TEXT_PIECES = ("The refund ", "was approved ", "yesterday.")
TOOL_ARGUMENTS = {"city": "Paris", "unit": "celsius"}


def _sse(chunks: list[dict[str, Any]], *, named: bool, done: bool = False) -> RecordedResponse:
    events = [SSEEvent(data=json.dumps(c), event=c["type"] if named else None) for c in chunks]
    if done:
        events.append(SSEEvent(data="[DONE]"))
    return RecordedResponse(200, {"content-type": "text/event-stream"}, events=events)


# --- OpenAI Chat Completions -------------------------------------------------------------


def _chat(message: dict[str, Any], finish_reason: str) -> RecordedResponse:
    return RecordedResponse.of_json(
        200,
        {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-5",
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
        },
    )


def openai_chat() -> RecordedResponse:
    return _chat({"role": "assistant", "content": FULL_TEXT}, "stop")


def openai_chat_tool() -> RecordedResponse:
    call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "get_weather", "arguments": json.dumps(TOOL_ARGUMENTS)},
    }
    return _chat({"role": "assistant", "content": None, "tool_calls": [call]}, "tool_calls")


def _chunk(delta: dict[str, Any], finish_reason: str | None = None) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "gpt-5",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def openai_chat_stream() -> RecordedResponse:
    chunks = [_chunk({"role": "assistant", "content": ""})]
    chunks += [_chunk({"content": piece}) for piece in TEXT_PIECES]
    chunks.append(_chunk({}, "stop"))
    return _sse(chunks, named=False, done=True)


def openai_chat_tool_stream() -> RecordedResponse:
    arguments = json.dumps(TOOL_ARGUMENTS)
    middle = len(arguments) // 2
    start = {
        "index": 0,
        "id": "call_1",
        "type": "function",
        "function": {"name": "get_weather", "arguments": ""},
    }
    chunks = [
        _chunk({"role": "assistant", "content": None, "tool_calls": [start]}),
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": arguments[:middle]}}]}),
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": arguments[middle:]}}]}),
        _chunk({}, "tool_calls"),
    ]
    return _sse(chunks, named=False, done=True)


# --- OpenAI Responses API ----------------------------------------------------------------


def _response(output: list[dict[str, Any]]) -> RecordedResponse:
    return RecordedResponse.of_json(
        200,
        {
            "id": "resp_1",
            "object": "response",
            "created_at": 0,
            "model": "gpt-5",
            "status": "completed",
            "output": output,
            "usage": {"input_tokens": 10, "output_tokens": 8, "total_tokens": 18},
        },
    )


def openai_response() -> RecordedResponse:
    text = {"type": "output_text", "text": FULL_TEXT, "annotations": []}
    return _response(
        [
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [text],
            }
        ]
    )


def openai_response_tool() -> RecordedResponse:
    call = {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": "get_weather",
        "arguments": json.dumps(TOOL_ARGUMENTS),
        "status": "completed",
    }
    return _response([call])


# --- Anthropic Messages ------------------------------------------------------------------


def _message(content: list[dict[str, Any]], stop_reason: str) -> RecordedResponse:
    return RecordedResponse.of_json(
        200,
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5-5",
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 8},
        },
    )


def anthropic_message() -> RecordedResponse:
    return _message([{"type": "text", "text": FULL_TEXT}], "end_turn")


def anthropic_tool_message() -> RecordedResponse:
    return _message(
        [
            {"type": "text", "text": "Let me check."},
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "get_weather",
                "input": dict(TOOL_ARGUMENTS),
            },
        ],
        "tool_use",
    )


def _message_start() -> dict[str, Any]:
    return {
        "type": "message_start",
        "message": {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5-5",
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 1},
        },
    }


def _message_end(stop_reason: str) -> list[dict[str, Any]]:
    return [
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": 8},
        },
        {"type": "message_stop"},
    ]


def anthropic_stream() -> RecordedResponse:
    chunks = [
        _message_start(),
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    ]
    chunks += [
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": piece}}
        for piece in TEXT_PIECES
    ]
    return _sse(chunks + _message_end("end_turn"), named=True)


def anthropic_tool_stream() -> RecordedResponse:
    arguments = json.dumps(TOOL_ARGUMENTS)
    middle = len(arguments) // 2
    block = {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {}}
    chunks = [
        _message_start(),
        {"type": "content_block_start", "index": 0, "content_block": block},
    ]
    chunks += [
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": part},
        }
        for part in (arguments[:middle], arguments[middle:])
    ]
    return _sse(chunks + _message_end("tool_use"), named=True)
