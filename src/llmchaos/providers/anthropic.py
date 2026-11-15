"""Anthropic adapter: error shapes and mutations for the Messages API."""

from __future__ import annotations

import copy
import json
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

DEFAULT_HOSTS = ("api.anthropic.com",)

_ERRORS: dict[ErrorKind, tuple[int, str, str]] = {
    "rate_limit": (
        429,
        "rate_limit_error",
        f"Number of requests has exceeded your rate limit {SIMULATED}",
    ),
    "overloaded": (529, "overloaded_error", f"Overloaded {SIMULATED}"),
    "server_error": (500, "api_error", f"Internal server error {SIMULATED}"),
}

_NO_TEXT = "needs a response with text content"
_NO_TOOL_CALL = "needs a response that contains a tool call with at least one argument"


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, extra_hosts: Iterable[str] = ()) -> None:
        self._hosts = frozenset((*DEFAULT_HOSTS, *extra_hosts))

    def handles(self, host: str) -> bool:
        return host in self._hosts

    def error_response(self, kind: ErrorKind, retry_after: float) -> RecordedResponse:
        status, error_type, message = _ERRORS[kind]
        body = {"type": "error", "error": {"type": error_type, "message": message}}
        return RecordedResponse.of_json(status, body, retry_headers(retry_after))

    def truncate(self, response: RecordedResponse) -> RecordedResponse:
        response = copy.deepcopy(response)
        if response.events is not None:
            response.events = _truncate_stream(response.events)
            return response
        body = _message_body(response)
        blocks = body["content"]
        index = next(
            (
                i
                for i, block in enumerate(blocks)
                if block.get("type") == "text" and block.get("text")
            ),
            None,
        )
        if index is None:
            raise FaultNotApplicableError(f"truncated() {_NO_TEXT}")
        blocks[index]["text"] = half(blocks[index]["text"])
        body["content"] = blocks[: index + 1]
        body["stop_reason"] = "max_tokens"
        return response

    def break_tool_args(self, response: RecordedResponse) -> RecordedResponse:
        response = copy.deepcopy(response)
        if response.events is not None:
            response.events = _break_stream_tool_input(response.events)
            return response
        body = _message_body(response)
        tools = [block for block in body["content"] if block.get("type") == "tool_use"]
        if not tools or not tools[-1].get("input"):
            raise FaultNotApplicableError(f"bad_tool_args() {_NO_TOOL_CALL}")
        tool_input = tools[-1]["input"]
        del tool_input[next(iter(tool_input))]
        return response


def _message_body(response: RecordedResponse) -> dict[str, Any]:
    body = response.body
    if not (isinstance(body, dict) and body.get("type") == "message"):
        raise FaultNotApplicableError("only Anthropic Messages API responses are supported")
    return body


def _blocks_of(parsed: dict[int, dict[str, Any]], kind: str) -> list[int]:
    return [
        data["index"]
        for data in parsed.values()
        if data.get("type") == "content_block_start"
        and data.get("content_block", {}).get("type") == kind
    ]


def _deltas(parsed: dict[int, dict[str, Any]], block: int, delta_type: str) -> list[int]:
    return [
        index
        for index, data in parsed.items()
        if data.get("type") == "content_block_delta"
        and data.get("index") == block
        and data.get("delta", {}).get("type") == delta_type
    ]


def _truncate_stream(events: Sequence[SSEEvent]) -> list[SSEEvent]:
    parsed = parse_json_events(events)
    text_blocks = _blocks_of(parsed, "text")
    if not text_blocks:
        raise FaultNotApplicableError(f"truncated() {_NO_TEXT}")
    block = text_blocks[0]
    pieces = [
        (index, parsed[index]["delta"], "text")
        for index in _deltas(parsed, block, "text_delta")
        if parsed[index]["delta"].get("text")
    ]
    if not pieces:
        raise FaultNotApplicableError(f"truncated() {_NO_TEXT}")
    changed, dropped = truncate_pieces(pieces)
    for index, data in parsed.items():
        kind = data.get("type", "")
        if kind.startswith("content_block_") and data.get("index", -1) > block:
            dropped.add(index)
        elif kind == "message_delta":
            data["delta"]["stop_reason"] = "max_tokens"
            changed.add(index)
    return rebuild_events(events, parsed, changed, dropped)


def _break_stream_tool_input(events: Sequence[SSEEvent]) -> list[SSEEvent]:
    parsed = parse_json_events(events)
    tool_blocks = _blocks_of(parsed, "tool_use")
    if not tool_blocks:
        raise FaultNotApplicableError(f"bad_tool_args() {_NO_TOOL_CALL}")
    deltas = _deltas(parsed, tool_blocks[-1], "input_json_delta")
    raw = "".join(parsed[index]["delta"]["partial_json"] for index in deltas)
    tool_input = json.loads(raw) if raw else {}
    if not isinstance(tool_input, dict) or not tool_input:
        raise FaultNotApplicableError(f"bad_tool_args() {_NO_TOOL_CALL}")
    del tool_input[next(iter(tool_input))]
    # One delta with the altered JSON: the SDK's partial-JSON parser accepts truncated
    # JSON silently, so a missing key is the failure the application actually sees.
    parsed[deltas[0]]["delta"]["partial_json"] = json.dumps(tool_input)
    return rebuild_events(events, parsed, {deltas[0]}, set(deltas[1:]))
