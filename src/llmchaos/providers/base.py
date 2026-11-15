"""The interface every provider adapter implements, plus helpers they share."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import Any, Protocol

from llmchaos.cassette import RecordedResponse
from llmchaos.faults import ErrorKind
from llmchaos.sse import SSEEvent

SIMULATED = "(simulated by llmchaos)"


class Provider(Protocol):
    """Knows one provider's HTTP shapes. Implementations never import the provider's SDK."""

    name: str

    def handles(self, host: str) -> bool: ...

    def error_response(self, kind: ErrorKind, retry_after: float) -> RecordedResponse: ...

    def truncate(self, response: RecordedResponse) -> RecordedResponse: ...

    def break_tool_args(self, response: RecordedResponse) -> RecordedResponse: ...


def half(text: str) -> str:
    return text[: len(text) // 2]


def retry_headers(retry_after: float) -> dict[str, str]:
    """``retry-after-ms`` (read first by both SDKs) and whole-second ``retry-after``."""
    return {
        "retry-after-ms": f"{retry_after * 1000:g}",
        "retry-after": str(math.ceil(retry_after)),
    }


def parse_json_events(events: Sequence[SSEEvent]) -> dict[int, dict[str, Any]]:
    """Event index -> parsed JSON object, for every event whose data is a JSON object."""
    parsed: dict[int, dict[str, Any]] = {}
    for index, event in enumerate(events):
        try:
            data = json.loads(event.data)
        except ValueError:
            continue
        if isinstance(data, dict):
            parsed[index] = data
    return parsed


def rebuild_events(
    events: Sequence[SSEEvent],
    parsed: dict[int, dict[str, Any]],
    changed: set[int],
    dropped: set[int],
) -> list[SSEEvent]:
    """Re-serialize changed events from ``parsed`` and leave out dropped ones."""
    rebuilt: list[SSEEvent] = []
    for index, event in enumerate(events):
        if index in dropped:
            continue
        if index in changed:
            event = SSEEvent(data=json.dumps(parsed[index]), event=event.event, id=event.id)
        rebuilt.append(event)
    return rebuilt


def truncate_pieces(pieces: list[tuple[int, dict[str, Any], str]]) -> tuple[set[int], set[int]]:
    """Keep the first half of the text spread over ``(event index, container, key)`` pieces.

    Edits the containers in place. Returns the event indices that changed and those to drop.
    """
    budget = sum(len(container[key]) for _, container, key in pieces) // 2
    changed: set[int] = set()
    dropped: set[int] = set()
    for index, container, key in pieces:
        text = container[key]
        if budget >= len(text):
            budget -= len(text)
        elif budget > 0:
            container[key] = text[:budget]
            budget = 0
            changed.add(index)
        else:
            dropped.add(index)
    return changed, dropped
