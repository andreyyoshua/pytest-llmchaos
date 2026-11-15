"""The cassette data model, conversion from raw HTTP data, and file I/O."""

from __future__ import annotations

import base64
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llmchaos import sse
from llmchaos.sse import SSEEvent

CASSETTE_VERSION = 1


@dataclass
class RecordedRequest:
    method: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    body: Any = None
    body_b64: str | None = None

    @classmethod
    def from_raw(
        cls, method: str, url: str, headers: Mapping[str, str], content: bytes
    ) -> RecordedRequest:
        lowered = _lower(headers)
        body, body_b64 = _decode_body(lowered.get("content-type", ""), content)
        return cls(method=method.upper(), url=url, headers=lowered, body=body, body_b64=body_b64)


@dataclass
class RecordedResponse:
    status: int
    headers: dict[str, str] = field(default_factory=dict)
    body: Any = None
    events: list[SSEEvent] | None = None
    body_b64: str | None = None

    @classmethod
    def from_raw(cls, status: int, headers: Mapping[str, str], content: bytes) -> RecordedResponse:
        lowered = _lower(headers)
        content_type = lowered.get("content-type", "")
        if content_type.startswith("text/event-stream"):
            return cls(status=status, headers=lowered, events=sse.parse(content))
        body, body_b64 = _decode_body(content_type, content)
        return cls(status=status, headers=lowered, body=body, body_b64=body_b64)

    @classmethod
    def of_json(
        cls, status: int, body: Any, headers: Mapping[str, str] | None = None
    ) -> RecordedResponse:
        return cls(
            status=status,
            headers={"content-type": "application/json", **(headers or {})},
            body=body,
        )

    def chunks(self) -> list[bytes]:
        """The body as bytes: one chunk per SSE event for streams, a single chunk otherwise."""
        if self.events is not None:
            return [sse.serialize_event(event) for event in self.events]
        if self.body_b64 is not None:
            return [base64.b64decode(self.body_b64)]
        return [json.dumps(self.body).encode("utf-8")]


@dataclass
class Interaction:
    request: RecordedRequest
    response: RecordedResponse


def to_document(interactions: Sequence[Interaction]) -> dict[str, Any]:
    return {
        "version": CASSETTE_VERSION,
        "interactions": [
            {"request": _request_to_dict(i.request), "response": _response_to_dict(i.response)}
            for i in interactions
        ],
    }


def load(path: Path) -> list[Interaction]:
    document = json.loads(path.read_text(encoding="utf-8"))
    version = document.get("version")
    if version != CASSETTE_VERSION:
        raise ValueError(
            f"{path}: unsupported cassette version {version!r} (expected {CASSETTE_VERSION})"
        )
    return [
        Interaction(_request_from_dict(item["request"]), _response_from_dict(item["response"]))
        for item in document["interactions"]
    ]


def save(path: Path, interactions: Sequence[Interaction]) -> None:
    """Write the cassette atomically: a temporary file in the same directory, then a rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(to_document(interactions), indent=2, sort_keys=True, ensure_ascii=False)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _lower(headers: Mapping[str, str]) -> dict[str, str]:
    return {key.lower(): value for key, value in headers.items()}


def _decode_body(content_type: str, content: bytes) -> tuple[Any, str | None]:
    """Parsed JSON when possible, otherwise base64. An empty body becomes an empty base64 string."""
    if content and "json" in content_type:
        try:
            return json.loads(content), None
        except ValueError:
            pass
    return None, base64.b64encode(content).decode("ascii")


def _request_to_dict(request: RecordedRequest) -> dict[str, Any]:
    result: dict[str, Any] = {
        "method": request.method,
        "url": request.url,
        "headers": request.headers,
    }
    if request.body_b64 is not None:
        result["body_b64"] = request.body_b64
    else:
        result["body"] = request.body
    return result


def _response_to_dict(response: RecordedResponse) -> dict[str, Any]:
    result: dict[str, Any] = {"status": response.status, "headers": response.headers}
    if response.events is not None:
        result["events"] = [_event_to_dict(event) for event in response.events]
    elif response.body_b64 is not None:
        result["body_b64"] = response.body_b64
    else:
        result["body"] = response.body
    return result


def _event_to_dict(event: SSEEvent) -> dict[str, str]:
    result = {"data": event.data}
    if event.event is not None:
        result["event"] = event.event
    if event.id is not None:
        result["id"] = event.id
    return result


def _request_from_dict(data: dict[str, Any]) -> RecordedRequest:
    return RecordedRequest(
        method=data["method"],
        url=data["url"],
        headers=dict(data.get("headers", {})),
        body=data.get("body"),
        body_b64=data.get("body_b64"),
    )


def _response_from_dict(data: dict[str, Any]) -> RecordedResponse:
    events = data.get("events")
    return RecordedResponse(
        status=data["status"],
        headers=dict(data.get("headers", {})),
        body=data.get("body"),
        events=None
        if events is None
        else [SSEEvent(data=e["data"], event=e.get("event"), id=e.get("id")) for e in events],
        body_b64=data.get("body_b64"),
    )
