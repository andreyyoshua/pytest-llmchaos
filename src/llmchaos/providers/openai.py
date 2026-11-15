"""OpenAI adapter: error shapes and mutations for Chat Completions and the Responses API."""

from __future__ import annotations

import copy
from collections.abc import Iterable, Sequence
from typing import Any

from llmchaos.cassette import RecordedResponse
from llmchaos.errors import FaultNotApplicableError
from llmchaos.faults import ErrorKind
from llmchaos.providers.base import (
    SIMULATED,
    half,
    parse_json_events,
    rebuild_events,
    retry_headers,
    truncate_pieces,
)
from llmchaos.sse import SSEEvent

DEFAULT_HOSTS = ("api.openai.com",)

_ERRORS: dict[ErrorKind, tuple[int, dict[str, Any]]] = {
    "rate_limit": (
        429,
        {
            "message": f"Rate limit reached {SIMULATED}",
            "type": "requests",
            "param": None,
            "code": "rate_limit_exceeded",
        },
    ),
    "overloaded": (
        503,
        {
            "message": f"The server is overloaded {SIMULATED}",
            "type": "server_error",
            "param": None,
            "code": None,
        },
    ),
    "server_error": (
        500,
        {
            "message": f"The server had an error while processing your request {SIMULATED}",
            "type": "server_error",
            "param": None,
            "code": None,
        },
    ),
}

_UNSUPPORTED = "supports OpenAI Chat Completions and non-streaming Responses API bodies"
_NO_TEXT = "needs a response with text content"
_NO_TOOL_CALL = "needs a response that contains a tool call"


class OpenAIProvider:
    name = "openai"

    def __init__(self, extra_hosts: Iterable[str] = ()) -> None:
        self._hosts = frozenset((*DEFAULT_HOSTS, *extra_hosts))

    def handles(self, host: str) -> bool:
        return host in self._hosts

    def error_response(self, kind: ErrorKind, retry_after: float) -> RecordedResponse:
        status, error = _ERRORS[kind]
        return RecordedResponse.of_json(status, {"error": dict(error)}, retry_headers(retry_after))

    def truncate(self, response: RecordedResponse) -> RecordedResponse:
        response = copy.deepcopy(response)
        if response.events is not None:
            response.events = _truncate_chat_stream(response.events)
            return response
        body = response.body
        if _is(body, "chat.completion"):
            choice = body["choices"][0]
            if not choice["message"].get("content"):
                raise FaultNotApplicableError(f"truncated() {_NO_TEXT}")
            choice["message"]["content"] = half(choice["message"]["content"])
            choice["message"].pop("tool_calls", None)
            choice["finish_reason"] = "length"
            return response
        if _is(body, "response"):
            parts = _output_text_parts(body)
            if not parts:
                raise FaultNotApplicableError(f"truncated() {_NO_TEXT}")
            parts[0]["text"] = half(parts[0]["text"])
            body["status"] = "incomplete"
            body["incomplete_details"] = {"reason": "max_output_tokens"}
            return response
        raise FaultNotApplicableError(f"truncated() {_UNSUPPORTED}")

    def break_tool_args(self, response: RecordedResponse) -> RecordedResponse:
        response = copy.deepcopy(response)
        if response.events is not None:
            response.events = _break_chat_stream_arguments(response.events)
            return response
        body = response.body
        if _is(body, "chat.completion"):
            calls = body["choices"][0]["message"].get("tool_calls") or []
            if not calls:
                raise FaultNotApplicableError(f"bad_tool_args() {_NO_TOOL_CALL}")
            function = calls[-1]["function"]
            function["arguments"] = function["arguments"][:-1]
            return response
        if _is(body, "response"):
            calls = [item for item in body.get("output", []) if item.get("type") == "function_call"]
            if not calls:
                raise FaultNotApplicableError(f"bad_tool_args() {_NO_TOOL_CALL}")
            calls[-1]["arguments"] = calls[-1]["arguments"][:-1]
            return response
        raise FaultNotApplicableError(f"bad_tool_args() {_UNSUPPORTED}")


def _is(body: Any, kind: str) -> bool:
    return isinstance(body, dict) and body.get("object") == kind


def _output_text_parts(body: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        part
        for item in body.get("output", [])
        if item.get("type") == "message"
        for part in item.get("content", [])
        if part.get("type") == "output_text" and part.get("text")
    ]


def _chat_chunks(events: Sequence[SSEEvent]) -> dict[int, dict[str, Any]]:
    chunks = {
        index: data
        for index, data in parse_json_events(events).items()
        if data.get("object") == "chat.completion.chunk"
    }
    if not chunks:
        raise FaultNotApplicableError(
            "only Chat Completions streams are supported "
            "(Responses API streaming is not supported in v1)"
        )
    return chunks


def _delta(chunk: dict[str, Any]) -> dict[str, Any]:
    choices = chunk.get("choices") or []
    return (choices[0].get("delta") or {}) if choices else {}


def _truncate_chat_stream(events: Sequence[SSEEvent]) -> list[SSEEvent]:
    chunks = _chat_chunks(events)
    pieces = [
        (i, _delta(chunk), "content") for i, chunk in chunks.items() if _delta(chunk).get("content")
    ]
    if not pieces:
        raise FaultNotApplicableError(f"truncated() {_NO_TEXT}")
    changed, dropped = truncate_pieces(pieces)
    for index, chunk in chunks.items():
        choices = chunk.get("choices") or []
        if choices and choices[0].get("finish_reason") is not None:
            choices[0]["finish_reason"] = "length"
            changed.add(index)
    return rebuild_events(events, chunks, changed, dropped)


def _break_chat_stream_arguments(events: Sequence[SSEEvent]) -> list[SSEEvent]:
    chunks = _chat_chunks(events)
    fragments = [
        (index, call["function"])
        for index, chunk in chunks.items()
        for call in _delta(chunk).get("tool_calls") or []
        if (call.get("function") or {}).get("arguments")
    ]
    if not fragments:
        raise FaultNotApplicableError(f"bad_tool_args() {_NO_TOOL_CALL}")
    index, function = fragments[-1]
    function["arguments"] = function["arguments"][:-1]
    return rebuild_events(events, chunks, {index}, set())
