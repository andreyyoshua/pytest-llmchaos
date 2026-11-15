"""Parse and serialize server-sent events (SSE)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class SSEEvent:
    """One server-sent event. ``event`` and ``id`` are optional SSE fields."""

    data: str
    event: str | None = None
    id: str | None = None


def parse(raw: bytes) -> list[SSEEvent]:
    """Split an SSE byte stream into events. Comment lines and unknown fields are ignored."""
    text = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    events: list[SSEEvent] = []
    for block in text.split("\n\n"):
        data: list[str] = []
        event: str | None = None
        event_id: str | None = None
        for line in block.split("\n"):
            if not line or line.startswith(":"):
                continue
            field, _, value = line.partition(":")
            value = value.removeprefix(" ")
            if field == "data":
                data.append(value)
            elif field == "event":
                event = value
            elif field == "id":
                event_id = value
        if data or event is not None:
            events.append(SSEEvent(data="\n".join(data), event=event, id=event_id))
    return events


def serialize_event(event: SSEEvent) -> bytes:
    """Encode one event, including the blank line that terminates it."""
    lines: list[str] = []
    if event.event is not None:
        lines.append(f"event: {event.event}")
    if event.id is not None:
        lines.append(f"id: {event.id}")
    lines.extend(f"data: {line}" for line in event.data.split("\n"))
    return ("\n".join(lines) + "\n\n").encode("utf-8")


def serialize(events: Sequence[SSEEvent]) -> bytes:
    return b"".join(serialize_event(event) for event in events)
