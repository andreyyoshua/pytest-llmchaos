# pytest-llmchaos Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `pytest-llmchaos` 0.1.0, a pytest plugin that records OpenAI/Anthropic SDK calls once, replays them offline, and injects eight provider-accurate faults.

**Architecture:** An `llm` fixture patches the `HTTPTransport`/`AsyncHTTPTransport` classes of `httpx` and `httpx2` for the duration of a test. Requests to provider hosts go to a provider-agnostic `Session`. For each attempt, the session decides whether a scheduled fault answers it, a cassette entry replays it, or the real API records it. Provider adapters know each API's error bodies and response shapes. The core never imports an HTTP library or an SDK.

**Tech Stack:** Python ≥ 3.11, pytest ≥ 8 (pluggy new-style hook wrappers, `pytester`), hatchling, uv, ruff, pyright. Test-only: `openai` 3.x + `anthropic` 2.x (on `httpx2`), `openai==2.46.0` + `anthropic==0.125.0` (on `httpx`).

**Spec:** `docs/design/2026-11-15-pytest-llmchaos-design.md`

## Global Constraints

- `requires-python = ">=3.11"`; the only runtime dependency is `pytest>=8`.
- PyPI name `pytest-llmchaos`; import name `llmchaos`; version `0.1.0`; license MIT.
- `httpx`/`httpx2` are touched only in `src/llmchaos/interceptor.py`, and only through `importlib`. `openai`/`anthropic` are never imported anywhere under `src/`.
- Every synthesized error message ends with `(simulated by llmchaos)` (the `SIMULATED` constant).
- Cassettes: `"version": 1`, `json.dumps(..., indent=2, sort_keys=True, ensure_ascii=False)` plus a trailing newline, written atomically.
- Header allowlists. Request: `content-type`, `anthropic-version`, `anthropic-beta`, `openai-beta`. Response: `content-type`, `request-id`, `x-request-id`.
- Default `retry_after` is `0.001` seconds. All synthesized HTTP errors carry `retry-after-ms`.
- Run every command from the repository root (`pytest-llmchaos/`) through `uv run`.
- Before every commit, run `uv run ruff format . && uv run ruff check --fix .`, and make sure that task's tests pass.
- End every commit message with the line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. The commit commands below pass it as a second `-m`.

## Verified facts this plan relies on (checked 2026-11-15)

- `openai` ≥ 3.0.0 and `anthropic` ≥ 1.0.0 depend only on `httpx2` 2.x, Pydantic's API-compatible fork of `httpx`. `openai` ≤ 2.46.0 and `anthropic` ≤ 0.125.0 use `httpx`. `httpx2` requires Python ≥ 3.11.
- Patching `httpx2.HTTPTransport.handle_request` intercepts both current SDKs (spike).
- Both SDKs read `retry-after-ms` first; a value of `1` makes retries instant. Default `max_retries` is 2. Both retry on 408/409/429/≥500 and on network errors.
- HTTP 529 maps to `anthropic.OverloadedError`, a subclass of `anthropic.InternalServerError`. OpenAI 503 maps to `openai.InternalServerError`.
- Non-httpx exceptions raised in the transport behave differently per SDK. `openai` 3.51 lets them propagate unchanged after 1 attempt. `anthropic` 2.8 retries 3 times and raises `APIConnectionError`. Legacy SDKs also wrap.
- `openai` 3.x surfaces a stream cut mid-body as `APIConnectionError`, after yielding the chunks received before the cut.
- Anthropic's stream helper parses tool JSON with `partial_mode=True`, so truncated JSON is silently accepted. That is why `bad_tool_args` drops a key for Anthropic.
- A transport-level `httpx2.Response.read()` decodes `content-encoding: gzip`. Header keys iterate in lowercase.

## File map

| Path | Responsibility | Task |
|---|---|---|
| `pyproject.toml`, `LICENSE`, `.gitignore`, `README.md` | Packaging, tooling config | 1 (README rewritten in 14) |
| `src/llmchaos/__init__.py` | Public exports, `__version__` | 1, 6, 9 |
| `src/llmchaos/errors.py` | Exception hierarchy | 1 |
| `src/llmchaos/sse.py` | SSE parse/serialize | 2 |
| `src/llmchaos/cassette.py` | Data model, raw conversion, load/save | 3 |
| `src/llmchaos/matching.py` | Ignore paths, equality, diffs | 4 |
| `src/llmchaos/redaction.py` | Header allowlists, secret scan | 5 |
| `src/llmchaos/faults.py` | Public fault types and constructors | 6 |
| `src/llmchaos/providers/base.py` | `Provider` protocol, shared helpers | 7 |
| `src/llmchaos/providers/openai.py` | OpenAI shapes | 7 |
| `src/llmchaos/providers/anthropic.py` | Anthropic shapes | 8 |
| `src/llmchaos/session.py` | Scheduling, replay, recording; `LLM` facade | 9 |
| `src/llmchaos/interceptor.py` | Transport patching for `httpx` and `httpx2` | 10 |
| `src/llmchaos/plugin.py` | Options, `llm` fixture, failure reporting | 1 (stub), 11 |
| `tests/payloads.py` | Canonical provider payloads | 7 |
| `tests/harness.py` | Fake network + record-then-replay runner | 11 |
| `tests/unit/*` | Unit tests | 1–11 |
| `tests/integration/*` | Real-SDK tests via `pytester` | 12, 13 |
| `tests/contract/test_live.py` | Live API checks | 15 |
| `examples/support_bot/*` | Example app, tests, cassettes | 14 |
| `CONTRIBUTING.md`, `CHANGELOG.md` | Docs | 14 |
| `.github/workflows/{ci,contract,release}.yml` | CI and release | 15 |

---

### Task 1: Project scaffold and error hierarchy

**Files:**
- Create: `.gitignore`, `LICENSE`, `README.md`, `pyproject.toml`
- Create: `src/llmchaos/__init__.py`, `src/llmchaos/errors.py`, `src/llmchaos/plugin.py` (stub), `src/llmchaos/py.typed`
- Test: `tests/unit/test_errors.py`

**Interfaces:**
- Produces: `llmchaos.errors.{LLMChaosError, CassetteMissingError, CassetteMismatchError, UnexpectedCallError, UnusedFaultError, FaultNotApplicableError, SecretLeakError}`; `llmchaos.__version__ == "0.1.0"`. The `pytest11` entry point `llmchaos = "llmchaos.plugin"` must import cleanly from now on, which is why `plugin.py` starts as a stub.

- [ ] **Step 1: Initialise the repository**

```bash
git init -b main
```

- [ ] **Step 2: Write the scaffold files**

`.gitignore`:

```gitignore
.venv/
__pycache__/
*.py[cod]
dist/
build/
*.egg-info/
.pytest_cache/
.ruff_cache/
```

`LICENSE`:

```text
MIT License

Copyright (c) 2026 pytest-llmchaos contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

`README.md` (placeholder until Task 14; hatchling needs the file to exist):

```markdown
# pytest-llmchaos

Record real LLM API responses once, replay them offline, and inject provider-accurate failures.
```

`pyproject.toml`:

```toml
[build-system]
requires = ["hatchling>=1.27"]
build-backend = "hatchling.build"

[project]
name = "pytest-llmchaos"
version = "0.1.0"
description = "Record real LLM API responses once, replay them offline, and inject provider-accurate failures."
readme = "README.md"
requires-python = ">=3.11"
license = "MIT"
license-files = ["LICENSE"]
authors = [{ name = "pytest-llmchaos contributors" }]
keywords = ["pytest", "llm", "openai", "anthropic", "testing", "chaos", "cassette", "vcr"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Framework :: Pytest",
    "Intended Audience :: Developers",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Programming Language :: Python :: 3.14",
    "Topic :: Software Development :: Testing",
    "Typing :: Typed",
]
dependencies = ["pytest>=8"]

[project.entry-points.pytest11]
llmchaos = "llmchaos.plugin"

[dependency-groups]
dev = [
    "anthropic>=2",
    "httpx>=0.27",
    "httpx2>=2.12",
    "openai>=3",
    "pyright>=1.1.400",
    "ruff>=0.14",
]

[tool.hatch.build.targets.wheel]
packages = ["src/llmchaos"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "RUF"]
ignore = ["E501"]

[tool.pyright]
include = ["src"]
pythonVersion = "3.11"
typeCheckingMode = "standard"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["tests"]
addopts = ["-ra", "--strict-markers", "-p", "pytester"]
markers = ["contract: live tests against the real provider APIs (need API keys)"]
```

`src/llmchaos/plugin.py` (stub, replaced in Task 11):

```python
"""pytest entry point for pytest-llmchaos. Filled in by Task 11."""
```

`src/llmchaos/__init__.py`:

```python
"""Record real LLM responses once, replay them offline, and inject provider-accurate failures."""

__all__ = ["__version__"]
__version__ = "0.1.0"
```

`src/llmchaos/py.typed`: an empty file.

- [ ] **Step 3: Install the environment**

Run: `uv sync`
Expected: creates `.venv` with Python ≥ 3.11, installs the dev group, and installs the project in editable mode.

- [ ] **Step 4: Write the failing test**

`tests/unit/test_errors.py`:

```python
from llmchaos import errors

ALL_ERRORS = [
    errors.CassetteMissingError,
    errors.CassetteMismatchError,
    errors.UnexpectedCallError,
    errors.UnusedFaultError,
    errors.FaultNotApplicableError,
    errors.SecretLeakError,
]


def test_every_error_shares_the_base_class():
    for error in ALL_ERRORS:
        assert issubclass(error, errors.LLMChaosError)


def test_base_class_is_not_an_http_library_error():
    # SDKs that only retry network errors must let ours propagate unchanged.
    assert errors.LLMChaosError.__mro__ == (errors.LLMChaosError, Exception, BaseException, object)
```

- [ ] **Step 5: Run it to verify it fails**

Run: `uv run pytest tests/unit/test_errors.py -v`
Expected: FAIL with `ImportError: cannot import name 'errors' from 'llmchaos'`.

- [ ] **Step 6: Implement the errors module and export it**

`src/llmchaos/errors.py`:

```python
"""Exceptions raised by pytest-llmchaos."""


class LLMChaosError(Exception):
    """Base class for every error pytest-llmchaos raises.

    Deliberately not a subclass of any httpx exception: SDKs that only retry
    network errors then let it propagate unchanged.
    """


class CassetteMissingError(LLMChaosError):
    """Replay-only mode, but the test has no cassette."""


class CassetteMismatchError(LLMChaosError):
    """A request matches no unused recorded interaction."""


class UnexpectedCallError(LLMChaosError):
    """The test made more calls than its cassette recorded."""


class UnusedFaultError(LLMChaosError):
    """An injected fault never fired."""


class FaultNotApplicableError(LLMChaosError):
    """A mutate fault cannot be applied to the response it targets."""


class SecretLeakError(LLMChaosError):
    """A cassette about to be written contains something that looks like an API key."""
```

Replace `src/llmchaos/__init__.py` with:

```python
"""Record real LLM responses once, replay them offline, and inject provider-accurate failures."""

from llmchaos import errors

__all__ = ["__version__", "errors"]
__version__ = "0.1.0"
```

- [ ] **Step 7: Run tests and tooling**

Run: `uv run pytest -v && uv run pyright && uv run ruff check .`
Expected: 2 passed; pyright `0 errors`; ruff `All checks passed!`

- [ ] **Step 8: Commit**

```bash
uv run ruff format . && uv run ruff check --fix .
git add .
git commit -m "chore: scaffold pytest-llmchaos with error hierarchy" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Server-sent events

**Files:**
- Create: `src/llmchaos/sse.py`
- Test: `tests/unit/test_sse.py`

**Interfaces:**
- Produces: `SSEEvent(data: str, event: str | None = None, id: str | None = None)` (frozen dataclass); `parse(raw: bytes) -> list[SSEEvent]`; `serialize_event(event: SSEEvent) -> bytes`; `serialize(events: Sequence[SSEEvent]) -> bytes`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_sse.py`:

```python
from llmchaos import sse
from llmchaos.sse import SSEEvent

ANTHROPIC_STYLE = (
    b"event: message_start\n"
    b'data: {"type":"message_start"}\n'
    b"\n"
    b"event: ping\n"
    b'data: {"type":"ping"}\n'
    b"\n"
)


def test_parses_named_events():
    assert sse.parse(ANTHROPIC_STYLE) == [
        SSEEvent(data='{"type":"message_start"}', event="message_start"),
        SSEEvent(data='{"type":"ping"}', event="ping"),
    ]


def test_parses_unnamed_events_and_the_done_marker():
    raw = b'data: {"a":1}\n\ndata: [DONE]\n\n'
    assert sse.parse(raw) == [SSEEvent(data='{"a":1}'), SSEEvent(data="[DONE]")]


def test_joins_multi_line_data():
    assert sse.parse(b"data: one\ndata: two\n\n") == [SSEEvent(data="one\ntwo")]


def test_ignores_comments_and_handles_crlf():
    raw = b": keep-alive\r\n\r\nid: 7\r\ndata: x\r\n\r\n"
    assert sse.parse(raw) == [SSEEvent(data="x", id="7")]


def test_accepts_a_final_event_without_a_trailing_blank_line():
    assert sse.parse(b"data: last") == [SSEEvent(data="last")]


def test_serialize_round_trips():
    events = [SSEEvent(data="a\nb", event="delta", id="1"), SSEEvent(data="[DONE]")]
    assert sse.parse(sse.serialize(events)) == events
    assert sse.serialize(sse.parse(ANTHROPIC_STYLE)) == ANTHROPIC_STYLE
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/unit/test_sse.py -v`
Expected: FAIL with `ImportError: cannot import name 'sse'`.

- [ ] **Step 3: Implement**

`src/llmchaos/sse.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_sse.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
uv run ruff format . && uv run ruff check --fix .
git add src/llmchaos/sse.py tests/unit/test_sse.py
git commit -m "feat: add server-sent event parsing and serialization" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Cassette data model and file I/O

**Files:**
- Create: `src/llmchaos/cassette.py`
- Test: `tests/unit/test_cassette.py`

**Interfaces:**
- Consumes: `llmchaos.sse.{SSEEvent, parse, serialize_event}`.
- Produces:
  - `RecordedRequest(method: str, url: str, headers: dict[str, str] = {}, body: Any = None, body_b64: str | None = None)`, plus classmethod `from_raw(method, url, headers: Mapping[str, str], content: bytes)`.
  - `RecordedResponse(status: int, headers: dict[str, str] = {}, body: Any = None, events: list[SSEEvent] | None = None, body_b64: str | None = None)`, plus classmethods `from_raw(status, headers, content)` and `of_json(status, body, headers=None)`, and method `chunks() -> list[bytes]`.
  - `Interaction(request: RecordedRequest, response: RecordedResponse)`.
  - `CASSETTE_VERSION = 1`; `to_document(interactions) -> dict[str, Any]`; `load(path: Path) -> list[Interaction]`; `save(path: Path, interactions: Sequence[Interaction]) -> None`.
  - Body encoding rule: parsed JSON in `body` when the content type contains `json` and it parses. Otherwise base64 in `body_b64`. An empty body is `body_b64 == ""`. SSE responses use `events`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_cassette.py`:

```python
import json

import pytest

from llmchaos import cassette
from llmchaos.cassette import Interaction, RecordedRequest, RecordedResponse
from llmchaos.sse import SSEEvent

URL = "https://api.openai.com/v1/chat/completions"


def test_request_from_raw_parses_json_and_lowercases_headers():
    request = RecordedRequest.from_raw(
        "post", URL, {"Content-Type": "application/json"}, b'{"model":"gpt-5"}'
    )
    assert request == RecordedRequest(
        method="POST",
        url=URL,
        headers={"content-type": "application/json"},
        body={"model": "gpt-5"},
    )


def test_non_json_bodies_are_base64():
    request = RecordedRequest.from_raw("POST", URL, {"content-type": "audio/wav"}, b"\x00\x01")
    assert request.body is None
    assert request.body_b64 == "AAE="


def test_empty_body_is_empty_base64():
    response = RecordedResponse.from_raw(204, {}, b"")
    assert response.body_b64 == ""
    assert response.chunks() == [b""]


def test_event_stream_responses_are_split_into_events():
    response = RecordedResponse.from_raw(
        200, {"content-type": "text/event-stream; charset=utf-8"}, b"data: a\n\ndata: b\n\n"
    )
    assert response.events == [SSEEvent(data="a"), SSEEvent(data="b")]
    assert response.chunks() == [b"data: a\n\n", b"data: b\n\n"]


def test_of_json_sets_the_content_type_and_extra_headers():
    response = RecordedResponse.of_json(200, {"ok": True}, {"retry-after-ms": "1"})
    assert response.headers == {"content-type": "application/json", "retry-after-ms": "1"}
    assert json.loads(b"".join(response.chunks())) == {"ok": True}


def _interactions():
    return [
        Interaction(
            RecordedRequest("POST", URL, {"content-type": "application/json"}, body={"b": 1, "a": 2}),
            RecordedResponse.of_json(200, {"z": 1}),
        ),
        Interaction(
            RecordedRequest("POST", URL, body_b64="AAE="),
            RecordedResponse(
                200,
                {"content-type": "text/event-stream"},
                events=[SSEEvent(data="x", event="e", id="1"), SSEEvent(data="y")],
            ),
        ),
    ]


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "nested" / "c.json"
    cassette.save(path, _interactions())
    assert cassette.load(path) == _interactions()


def test_saved_file_is_sorted_indented_newline_terminated_and_atomic(tmp_path):
    path = tmp_path / "c.json"
    cassette.save(path, _interactions())
    text = path.read_text(encoding="utf-8")
    assert text == json.dumps(json.loads(text), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    assert list(tmp_path.iterdir()) == [path]  # no temporary files left behind


def test_load_rejects_unknown_versions(tmp_path):
    path = tmp_path / "c.json"
    path.write_text('{"version": 99, "interactions": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported cassette version 99"):
        cassette.load(path)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/unit/test_cassette.py -v`
Expected: FAIL with `ImportError: cannot import name 'cassette'`.

- [ ] **Step 3: Implement**

`src/llmchaos/cassette.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_cassette.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
uv run ruff format . && uv run ruff check --fix .
git add src/llmchaos/cassette.py tests/unit/test_cassette.py
git commit -m "feat: add cassette data model and atomic file I/O" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Request matching

**Files:**
- Create: `src/llmchaos/matching.py`
- Test: `tests/unit/test_matching.py`

**Interfaces:**
- Consumes: `llmchaos.cassette.RecordedRequest`.
- Produces:
  - `IGNORED = "<ignored>"`.
  - `apply_ignores(body: Any, paths: Iterable[str]) -> Any`: returns a deep copy; `*` matches every key or index.
  - `requests_match(actual: RecordedRequest, recorded: RecordedRequest, ignore: Sequence[str]) -> bool`.
  - `request_diff(actual: RecordedRequest, recorded: RecordedRequest, ignore: Sequence[str]) -> str`: unified diff from `recorded` to `actual`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_matching.py`:

```python
from llmchaos.cassette import RecordedRequest
from llmchaos.matching import IGNORED, apply_ignores, request_diff, requests_match

URL = "https://api.openai.com/v1/chat/completions"


def req(content="hi", method="POST", url=URL):
    body = {"model": "gpt-5", "messages": [{"role": "user", "content": content}]}
    return RecordedRequest(method=method, url=url, body=body)


def test_apply_ignores_by_key_index_and_wildcard():
    body = {"a": {"b": 1}, "items": [{"x": 1}, {"x": 2}], "keep": 3}
    assert apply_ignores(body, ["a.b", "items.*.x"]) == {
        "a": {"b": IGNORED},
        "items": [{"x": IGNORED}, {"x": IGNORED}],
        "keep": 3,
    }
    assert apply_ignores(body, ["items.1.x"])["items"] == [{"x": 1}, {"x": IGNORED}]


def test_apply_ignores_skips_missing_paths_and_never_mutates_its_input():
    body = {"a": [1]}
    assert apply_ignores(body, ["nope.x", "a.5", "a.b"]) == {"a": [1]}
    apply_ignores(body, ["a.0"])
    assert body == {"a": [1]}


def test_key_order_does_not_matter():
    a = RecordedRequest("POST", URL, body={"x": 1, "y": 2})
    b = RecordedRequest("POST", URL, body={"y": 2, "x": 1})
    assert requests_match(a, b, [])


def test_method_url_and_body_must_match_unless_ignored():
    assert not requests_match(req(method="GET"), req(), [])
    assert not requests_match(req(url=URL + "?v=2"), req(), [])
    assert not requests_match(req("a"), req("b"), [])
    assert requests_match(req("a"), req("b"), ["messages.0.content"])


def test_base64_bodies_compare_bytewise():
    a = RecordedRequest("POST", URL, body_b64="AAE=")
    assert requests_match(a, RecordedRequest("POST", URL, body_b64="AAE="), [])
    assert not requests_match(a, RecordedRequest("POST", URL, body_b64="AAI="), [])
    assert not requests_match(a, RecordedRequest("POST", URL, body=None), [])


def test_request_diff_shows_what_changed():
    diff = request_diff(req("goodbye"), req("hello"), [])
    assert "--- recorded" in diff
    assert "+++ actual" in diff
    assert '-      "content": "hello",' in diff
    assert '+      "content": "goodbye",' in diff
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/unit/test_matching.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llmchaos.matching'`.

- [ ] **Step 3: Implement**

`src/llmchaos/matching.py`:

```python
"""Request matching: ignore paths, equality, and readable diffs."""

from __future__ import annotations

import copy
import difflib
import json
from collections.abc import Iterable, Sequence
from typing import Any

from llmchaos.cassette import RecordedRequest

IGNORED = "<ignored>"


def apply_ignores(body: Any, paths: Iterable[str]) -> Any:
    """A copy of ``body`` with the value at each dotted path replaced by ``IGNORED``."""
    result = copy.deepcopy(body)
    for path in paths:
        _ignore(result, path.split("."))
    return result


def requests_match(
    actual: RecordedRequest, recorded: RecordedRequest, ignore: Sequence[str]
) -> bool:
    if actual.method != recorded.method or actual.url != recorded.url:
        return False
    if actual.body_b64 is not None or recorded.body_b64 is not None:
        return actual.body_b64 == recorded.body_b64
    return apply_ignores(actual.body, ignore) == apply_ignores(recorded.body, ignore)


def request_diff(actual: RecordedRequest, recorded: RecordedRequest, ignore: Sequence[str]) -> str:
    return "\n".join(
        difflib.unified_diff(
            _render(recorded, ignore),
            _render(actual, ignore),
            fromfile="recorded",
            tofile="actual",
            lineterm="",
        )
    )


def _ignore(node: Any, segments: list[str]) -> None:
    head, rest = segments[0], segments[1:]
    for key in _matching_keys(node, head):
        if rest:
            _ignore(node[key], rest)
        else:
            node[key] = IGNORED


def _matching_keys(node: Any, segment: str) -> list[Any]:
    if isinstance(node, dict):
        if segment == "*":
            return list(node)
        return [segment] if segment in node else []
    if isinstance(node, list):
        if segment == "*":
            return list(range(len(node)))
        if segment.isdigit() and int(segment) < len(node):
            return [int(segment)]
    return []


def _render(request: RecordedRequest, ignore: Sequence[str]) -> list[str]:
    if request.body_b64 is not None:
        body = [f"body_b64: {request.body_b64}"]
    else:
        body = json.dumps(
            apply_ignores(request.body, ignore), indent=2, sort_keys=True, ensure_ascii=False
        ).splitlines()
    return [f"{request.method} {request.url}", *body]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_matching.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
uv run ruff format . && uv run ruff check --fix .
git add src/llmchaos/matching.py tests/unit/test_matching.py
git commit -m "feat: add request matching with ignore paths and diffs" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Redaction and secret scan

**Files:**
- Create: `src/llmchaos/redaction.py`
- Test: `tests/unit/test_redaction.py`

**Interfaces:**
- Produces: `REQUEST_HEADER_ALLOWLIST: frozenset[str]`; `RESPONSE_HEADER_ALLOWLIST: frozenset[str]`; `KEY_PATTERN: re.Pattern[str]`; `filter_headers(headers: Mapping[str, str], allowlist: frozenset[str]) -> dict[str, str]`; `find_secrets(document: Any, environ: Mapping[str, str] | None = None) -> list[str]`, which returns JSON paths like `$.interactions[0].request.body.messages[0].content`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_redaction.py`:

```python
from llmchaos import redaction


def test_filter_headers_keeps_only_the_allowlist_and_lowercases():
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-x",
        "X-Api-Key": "k",
        "anthropic-version": "2023-06-01",
    }
    assert redaction.filter_headers(headers, redaction.REQUEST_HEADER_ALLOWLIST) == {
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
    }


def test_find_secrets_reports_key_patterns_with_their_path():
    document = {
        "interactions": [
            {"request": {"body": {"messages": [{"content": "use sk-proj-abcdefghijklmnopqrstuvwxyz"}]}}}
        ]
    }
    assert redaction.find_secrets(document, environ={}) == [
        "$.interactions[0].request.body.messages[0].content"
    ]


def test_find_secrets_ignores_words_that_merely_contain_sk_dash():
    document = {"text": "risk-assessment-for-the-quarterly-report"}
    assert redaction.find_secrets(document, environ={}) == []


def test_find_secrets_reports_environment_key_values():
    environ = {"ANTHROPIC_API_KEY": "plain-secret-value"}
    document = {"a": ["x", "has plain-secret-value inside"]}
    assert redaction.find_secrets(document, environ=environ) == ["$.a[1]"]


def test_find_secrets_ignores_short_environment_values():
    assert redaction.find_secrets({"a": "abc"}, environ={"OPENAI_API_KEY": "abc"}) == []
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/unit/test_redaction.py -v`
Expected: FAIL with `ImportError: cannot import name 'redaction'`.

- [ ] **Step 3: Implement**

`src/llmchaos/redaction.py`:

```python
"""Keep secrets out of cassettes: header allowlists and a secret scan."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import Any

REQUEST_HEADER_ALLOWLIST = frozenset(
    {"content-type", "anthropic-version", "anthropic-beta", "openai-beta"}
)
RESPONSE_HEADER_ALLOWLIST = frozenset({"content-type", "request-id", "x-request-id"})

# The lookbehind avoids false hits inside words such as "risk-assessment-...".
KEY_PATTERN = re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}")
KEY_ENV_VARS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
MIN_ENV_SECRET_LENGTH = 8


def filter_headers(headers: Mapping[str, str], allowlist: frozenset[str]) -> dict[str, str]:
    return {key.lower(): value for key, value in headers.items() if key.lower() in allowlist}


def find_secrets(document: Any, environ: Mapping[str, str] | None = None) -> list[str]:
    """JSON paths of every string in ``document`` that looks like an API key."""
    env = os.environ if environ is None else environ
    literals = [
        value
        for name in KEY_ENV_VARS
        if len(value := env.get(name, "")) >= MIN_ENV_SECRET_LENGTH
    ]
    hits: list[str] = []
    _walk(document, "$", literals, hits)
    return hits


def _walk(node: Any, path: str, literals: list[str], hits: list[str]) -> None:
    if isinstance(node, str):
        if KEY_PATTERN.search(node) or any(literal in node for literal in literals):
            hits.append(path)
    elif isinstance(node, dict):
        for key, value in node.items():
            _walk(value, f"{path}.{key}", literals, hits)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _walk(value, f"{path}[{index}]", literals, hits)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_redaction.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
uv run ruff format . && uv run ruff check --fix .
git add src/llmchaos/redaction.py tests/unit/test_redaction.py
git commit -m "feat: add header allowlists and cassette secret scan" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Faults

**Files:**
- Create: `src/llmchaos/faults.py`
- Modify: `src/llmchaos/__init__.py`
- Test: `tests/unit/test_faults.py`

**Interfaces:**
- Produces:
  - `ErrorKind = Literal["rate_limit", "overloaded", "server_error"]`; `TransportErrorKind = Literal["timeout", "connection_error"]`; `DEFAULT_RETRY_AFTER = 0.001`.
  - Frozen, keyword-only dataclasses:
    - `Fault(on_call: int | None = None)`
    - `ReplaceFault(Fault)` with `times: int = 1`
    - `HTTPError(ReplaceFault)` with `kind: ErrorKind`, `retry_after: float = 0.001`
    - `TransportError(ReplaceFault)` with `kind: TransportErrorKind`
    - `MutateFault(Fault)`
    - `StreamCut(MutateFault)` with `after_events: int = 1`
    - `Truncated(MutateFault)`
    - `BadToolArgs(MutateFault)`
  - Constructors: `rate_limit`, `overloaded`, `server_error`, `timeout`, `connection_error`, `stream_cut`, `truncated`, `bad_tool_args`, with the signatures in spec §5.1.
  - `llmchaos.faults` is exported from the package.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_faults.py`:

```python
import dataclasses

import pytest

import llmchaos
from llmchaos import faults


def test_constructors_build_the_expected_faults():
    assert faults.rate_limit(times=2, retry_after=1.5, on_call=3) == faults.HTTPError(
        kind="rate_limit", times=2, retry_after=1.5, on_call=3
    )
    assert faults.overloaded() == faults.HTTPError(kind="overloaded", times=1, retry_after=0.001)
    assert faults.server_error(times=4) == faults.HTTPError(kind="server_error", times=4)
    assert faults.timeout() == faults.TransportError(kind="timeout")
    assert faults.connection_error(times=2) == faults.TransportError(
        kind="connection_error", times=2
    )
    assert faults.stream_cut(after_events=0) == faults.StreamCut(after_events=0)
    assert faults.truncated(on_call=2) == faults.Truncated(on_call=2)
    assert faults.bad_tool_args() == faults.BadToolArgs()


def test_kinds_of_fault():
    assert isinstance(faults.rate_limit(), faults.ReplaceFault)
    assert isinstance(faults.timeout(), faults.ReplaceFault)
    for mutate in (faults.stream_cut(), faults.truncated(), faults.bad_tool_args()):
        assert isinstance(mutate, faults.MutateFault)


@pytest.mark.parametrize(
    "build",
    [
        lambda: faults.rate_limit(times=0),
        lambda: faults.timeout(on_call=0),
        lambda: faults.rate_limit(retry_after=-1),
        lambda: faults.stream_cut(after_events=-1),
    ],
)
def test_invalid_arguments_are_rejected(build):
    with pytest.raises(ValueError):
        build()


def test_faults_are_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        faults.timeout().times = 3  # type: ignore[misc]


def test_faults_module_is_exported():
    assert llmchaos.faults is faults
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/unit/test_faults.py -v`
Expected: FAIL with `ImportError: cannot import name 'faults'`.

- [ ] **Step 3: Implement**

`src/llmchaos/faults.py`:

```python
"""Faults a test can inject with ``llm.inject(...)``.

Replace faults answer an HTTP attempt themselves; the SDK's retry is another attempt at
the same call. Mutate faults alter the recorded response of the call they target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ErrorKind = Literal["rate_limit", "overloaded", "server_error"]
TransportErrorKind = Literal["timeout", "connection_error"]

DEFAULT_RETRY_AFTER = 0.001


@dataclass(frozen=True, kw_only=True)
class Fault:
    on_call: int | None = None

    def __post_init__(self) -> None:
        if self.on_call is not None and self.on_call < 1:
            raise ValueError(f"on_call must be >= 1, got {self.on_call}")


@dataclass(frozen=True, kw_only=True)
class ReplaceFault(Fault):
    """Answers up to ``times`` HTTP attempts of its target call."""

    times: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.times < 1:
            raise ValueError(f"times must be >= 1, got {self.times}")


@dataclass(frozen=True, kw_only=True)
class HTTPError(ReplaceFault):
    """A synthesized HTTP error response in the provider's own format."""

    kind: ErrorKind
    retry_after: float = DEFAULT_RETRY_AFTER

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.retry_after < 0:
            raise ValueError(f"retry_after must be >= 0, got {self.retry_after}")


@dataclass(frozen=True, kw_only=True)
class TransportError(ReplaceFault):
    """An exception raised by the HTTP library instead of a response."""

    kind: TransportErrorKind


@dataclass(frozen=True, kw_only=True)
class MutateFault(Fault):
    """Alters the recorded response of its target call."""


@dataclass(frozen=True, kw_only=True)
class StreamCut(MutateFault):
    after_events: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.after_events < 0:
            raise ValueError(f"after_events must be >= 0, got {self.after_events}")


@dataclass(frozen=True, kw_only=True)
class Truncated(MutateFault):
    """Half the text content, ending as if the token limit was hit."""


@dataclass(frozen=True, kw_only=True)
class BadToolArgs(MutateFault):
    """Tool-call arguments the application cannot use."""


def rate_limit(
    times: int = 1, retry_after: float = DEFAULT_RETRY_AFTER, on_call: int | None = None
) -> HTTPError:
    """HTTP 429 with the provider's error body and ``retry-after`` headers."""
    return HTTPError(kind="rate_limit", times=times, retry_after=retry_after, on_call=on_call)


def overloaded(times: int = 1, on_call: int | None = None) -> HTTPError:
    """Anthropic HTTP 529 ``overloaded_error``; OpenAI HTTP 503."""
    return HTTPError(kind="overloaded", times=times, on_call=on_call)


def server_error(times: int = 1, on_call: int | None = None) -> HTTPError:
    """HTTP 500 with the provider's error body."""
    return HTTPError(kind="server_error", times=times, on_call=on_call)


def timeout(times: int = 1, on_call: int | None = None) -> TransportError:
    """The HTTP library's ``ReadTimeout``, raised immediately."""
    return TransportError(kind="timeout", times=times, on_call=on_call)


def connection_error(times: int = 1, on_call: int | None = None) -> TransportError:
    """The HTTP library's ``ConnectError``, raised immediately."""
    return TransportError(kind="connection_error", times=times, on_call=on_call)


def stream_cut(after_events: int = 1, on_call: int | None = None) -> StreamCut:
    """Stream ``after_events`` events, then fail with ``RemoteProtocolError``."""
    return StreamCut(after_events=after_events, on_call=on_call)


def truncated(on_call: int | None = None) -> Truncated:
    """Keep the first half of the text and end with the token-limit stop reason."""
    return Truncated(on_call=on_call)


def bad_tool_args(on_call: int | None = None) -> BadToolArgs:
    """OpenAI: invalid tool-argument JSON. Anthropic: a tool input missing a key."""
    return BadToolArgs(on_call=on_call)
```

Replace `src/llmchaos/__init__.py` with:

```python
"""Record real LLM responses once, replay them offline, and inject provider-accurate failures."""

from llmchaos import errors, faults

__all__ = ["__version__", "errors", "faults"]
__version__ = "0.1.0"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit -v`
Expected: all unit tests pass (`test_faults.py`: 8 passed).

- [ ] **Step 5: Commit**

```bash
uv run ruff format . && uv run ruff check --fix .
git add src/llmchaos/faults.py src/llmchaos/__init__.py tests/unit/test_faults.py
git commit -m "feat: add fault types and constructors" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Provider protocol, shared helpers, OpenAI adapter, test payloads

**Files:**
- Create: `src/llmchaos/providers/__init__.py`, `src/llmchaos/providers/base.py`, `src/llmchaos/providers/openai.py`
- Create: `tests/payloads.py` (used by every later test task)
- Test: `tests/unit/test_provider_openai.py`

**Interfaces:**
- Consumes: `RecordedResponse.of_json`, `SSEEvent`, `FaultNotApplicableError`, `ErrorKind`.
- Produces:
  - `providers.base`: `SIMULATED`; `Provider` protocol (`name`, `handles(host)`, `error_response(kind, retry_after)`, `truncate(response)`, `break_tool_args(response)`); helpers `half(text)`, `parse_json_events(events) -> dict[int, dict[str, Any]]`, `rebuild_events(events, parsed, changed, dropped) -> list[SSEEvent]`, `truncate_pieces(pieces) -> tuple[set[int], set[int]]`, `retry_headers(retry_after) -> dict[str, str]`.
  - `providers.openai`: `OpenAIProvider(extra_hosts: Iterable[str] = ())`, `DEFAULT_HOSTS`.
  - `tests/payloads.py`:
    - Constants: `FULL_TEXT`, `TEXT_PIECES`, `TOOL_ARGUMENTS`.
    - OpenAI builders, each returning `RecordedResponse`: `openai_chat()`, `openai_chat_tool()`, `openai_chat_stream()`, `openai_chat_tool_stream()`, `openai_response()`, `openai_response_tool()`.
    - Anthropic builders, each returning `RecordedResponse`: `anthropic_message()`, `anthropic_tool_message()`, `anthropic_stream()`, `anthropic_tool_stream()`.

- [ ] **Step 1: Write the shared payloads**

`tests/payloads.py`:

```python
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
        [{"type": "message", "id": "msg_1", "role": "assistant", "status": "completed", "content": [text]}]
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
            {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": dict(TOOL_ARGUMENTS)},
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
```

- [ ] **Step 2: Write the failing tests**

`tests/unit/test_provider_openai.py`:

```python
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
```

- [ ] **Step 3: Run them to verify they fail**

Run: `uv run pytest tests/unit/test_provider_openai.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llmchaos.providers'`.

- [ ] **Step 4: Implement the protocol and helpers**

`src/llmchaos/providers/__init__.py`:

```python
"""Provider adapters: each one knows a provider's HTTP shapes, never its SDK."""
```

`src/llmchaos/providers/base.py`:

```python
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
```

- [ ] **Step 5: Implement the OpenAI adapter**

`src/llmchaos/providers/openai.py`:

```python
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
    pieces = [(i, _delta(chunk), "content") for i, chunk in chunks.items() if _delta(chunk).get("content")]
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
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_provider_openai.py -v && uv run pyright`
Expected: 13 passed; pyright `0 errors`.

- [ ] **Step 7: Commit**

```bash
uv run ruff format . && uv run ruff check --fix .
git add src/llmchaos/providers tests/payloads.py tests/unit/test_provider_openai.py
git commit -m "feat: add provider protocol and OpenAI adapter" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Anthropic adapter

**Files:**
- Create: `src/llmchaos/providers/anthropic.py`
- Test: `tests/unit/test_provider_anthropic.py`

**Interfaces:**
- Consumes: everything in `providers.base`; `tests/payloads.py`.
- Produces: `AnthropicProvider(extra_hosts: Iterable[str] = ())`, `DEFAULT_HOSTS`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_provider_anthropic.py`:

```python
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
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/unit/test_provider_anthropic.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llmchaos.providers.anthropic'`.

- [ ] **Step 3: Implement**

`src/llmchaos/providers/anthropic.py`:

```python
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
            (i for i, block in enumerate(blocks) if block.get("type") == "text" and block.get("text")),
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_provider_anthropic.py -v && uv run pyright`
Expected: 12 passed; pyright `0 errors`.

- [ ] **Step 5: Commit**

```bash
uv run ruff format . && uv run ruff check --fix .
git add src/llmchaos/providers/anthropic.py tests/unit/test_provider_anthropic.py
git commit -m "feat: add Anthropic adapter" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Session (scheduling, replay, recording) and the `LLM` facade

**Files:**
- Create: `src/llmchaos/session.py`
- Modify: `src/llmchaos/__init__.py`
- Test: `tests/unit/test_session.py`

**Interfaces:**
- Consumes: `cassette.{Interaction, RecordedRequest, RecordedResponse, load, save, to_document}`, `matching.{requests_match, request_diff}`, `redaction.{filter_headers, find_secrets, REQUEST_HEADER_ALLOWLIST, RESPONSE_HEADER_ALLOWLIST}`, all fault classes, `Provider`.
- Produces:
  - `RecordMode = Literal["none", "missing", "all"]`.
  - `Outcome(response: RecordedResponse | None = None, raise_error: TransportErrorKind | None = None, cut_after: int | None = None)` (frozen).
  - `Record(call: int, provider: Provider)` (frozen).
  - `Session(*, cassette_path: Path, mode: RecordMode, providers: Sequence[Provider], ignore: Sequence[str] = ())`:
    - Attributes: `cassette_path`, `calls`, `attempts`, `error: LLMChaosError | None`.
    - Methods: `provider_for(host) -> Provider | None`, `inject(*faults)`, `ignore(*paths)`, `unused_faults() -> list[Fault]`, `handle(request, provider) -> Outcome | Record`, `recorded(step: Record, request, response) -> Outcome`, `save()`.
  - `LLM(session)`: the fixture value, with `inject`, `ignore`, `calls`, `attempts`, `cassette_path`.
  - `llmchaos.LLM` is exported.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_session.py`:

```python
from dataclasses import replace

import payloads
import pytest

from llmchaos import cassette, faults
from llmchaos.cassette import Interaction, RecordedRequest, RecordedResponse
from llmchaos.errors import (
    CassetteMismatchError,
    CassetteMissingError,
    FaultNotApplicableError,
    SecretLeakError,
    UnexpectedCallError,
)
from llmchaos.providers.openai import OpenAIProvider
from llmchaos.session import LLM, Outcome, Record, Session

CHAT_URL = "https://api.openai.com/v1/chat/completions"
OPENAI = OpenAIProvider()


def chat_request(content="hi"):
    return RecordedRequest(
        method="POST",
        url=CHAT_URL,
        headers={"content-type": "application/json"},
        body={"model": "gpt-5", "messages": [{"role": "user", "content": content}]},
    )


def make(tmp_path, mode="none", recorded=(), ignore=()):
    path = tmp_path / "c.json"
    if recorded:
        cassette.save(path, [Interaction(request, response) for request, response in recorded])
    return Session(cassette_path=path, mode=mode, providers=[OPENAI], ignore=ignore)


def served(outcome) -> RecordedResponse:
    assert isinstance(outcome, Outcome)
    assert outcome.response is not None
    return outcome.response


# --- replay --------------------------------------------------------------------------------


def test_replays_the_matching_interaction(tmp_path):
    session = make(tmp_path, recorded=[(chat_request(), payloads.openai_chat())])
    assert served(session.handle(chat_request(), OPENAI)).body == payloads.openai_chat().body
    assert (session.calls, session.attempts) == (1, 1)


def test_identical_requests_replay_in_recorded_order(tmp_path):
    first, second = payloads.openai_chat(), payloads.openai_chat_tool()
    session = make(tmp_path, recorded=[(chat_request(), first), (chat_request(), second)])
    assert served(session.handle(chat_request(), OPENAI)).body == first.body
    assert served(session.handle(chat_request(), OPENAI)).body == second.body


def test_ignore_paths_allow_dynamic_values(tmp_path):
    recorded = [(chat_request("Today is Monday"), payloads.openai_chat())]
    session = make(tmp_path, recorded=recorded, ignore=["messages.*.content"])
    served(session.handle(chat_request("Today is Tuesday"), OPENAI))


def test_llm_ignore_adds_paths_for_this_test(tmp_path):
    session = make(tmp_path, recorded=[(chat_request("Monday"), payloads.openai_chat())])
    LLM(session).ignore("messages.0.content")
    served(session.handle(chat_request("Tuesday"), OPENAI))


def test_missing_cassette_in_replay_only_mode(tmp_path):
    session = make(tmp_path)
    with pytest.raises(CassetteMissingError, match="--llm-record=missing"):
        session.handle(chat_request(), OPENAI)
    assert isinstance(session.error, CassetteMissingError)


def test_mismatch_shows_a_diff(tmp_path):
    session = make(tmp_path, recorded=[(chat_request("hello"), payloads.openai_chat())])
    with pytest.raises(CassetteMismatchError) as error:
        session.handle(chat_request("goodbye"), OPENAI)
    assert '-      "content": "hello",' in str(error.value)
    assert '+      "content": "goodbye",' in str(error.value)
    assert "--llm-record=all" in str(error.value)


def test_extra_calls_are_unexpected(tmp_path):
    session = make(tmp_path, recorded=[(chat_request(), payloads.openai_chat())])
    session.handle(chat_request(), OPENAI)
    with pytest.raises(UnexpectedCallError):
        session.handle(chat_request(), OPENAI)


def test_the_first_error_is_kept(tmp_path):
    session = make(tmp_path, recorded=[(chat_request(), payloads.openai_chat())])
    with pytest.raises(CassetteMismatchError):
        session.handle(chat_request("other"), OPENAI)
    session.handle(chat_request(), OPENAI)
    with pytest.raises(UnexpectedCallError):
        session.handle(chat_request(), OPENAI)
    assert isinstance(session.error, CassetteMismatchError)


# --- faults --------------------------------------------------------------------------------


def test_rate_limit_answers_attempts_without_advancing_the_cassette(tmp_path):
    session = make(tmp_path, recorded=[(chat_request(), payloads.openai_chat())])
    session.inject(faults.rate_limit(times=2))
    for _ in range(2):
        response = served(session.handle(chat_request(), OPENAI))
        assert response.status == 429
        assert response.headers["retry-after-ms"] == "1"
    assert served(session.handle(chat_request(), OPENAI)).status == 200
    assert (session.calls, session.attempts) == (1, 3)


def test_on_call_targets_a_later_call(tmp_path):
    chat = payloads.openai_chat()
    session = make(tmp_path, recorded=[(chat_request("a"), chat), (chat_request("b"), chat)])
    session.inject(faults.server_error(on_call=2))
    assert served(session.handle(chat_request("a"), OPENAI)).status == 200
    assert served(session.handle(chat_request("b"), OPENAI)).status == 500
    assert served(session.handle(chat_request("b"), OPENAI)).status == 200


def test_replace_faults_apply_in_injection_order(tmp_path):
    session = make(tmp_path, recorded=[(chat_request(), payloads.openai_chat())])
    session.inject(faults.rate_limit(), faults.overloaded())
    statuses = [served(session.handle(chat_request(), OPENAI)).status for _ in range(3)]
    assert statuses == [429, 503, 200]


def test_transport_faults_ask_the_interceptor_to_raise(tmp_path):
    session = make(tmp_path)
    session.inject(faults.timeout(), faults.connection_error())
    assert session.handle(chat_request(), OPENAI) == Outcome(raise_error="timeout")
    assert session.handle(chat_request(), OPENAI) == Outcome(raise_error="connection_error")


def test_stream_cut_tells_the_interceptor_where_to_stop(tmp_path):
    session = make(tmp_path, recorded=[(chat_request(), payloads.openai_chat_stream())])
    session.inject(faults.stream_cut(after_events=2))
    outcome = session.handle(chat_request(), OPENAI)
    assert isinstance(outcome, Outcome)
    assert outcome.cut_after == 2


def test_stream_cut_needs_a_stream(tmp_path):
    session = make(tmp_path, recorded=[(chat_request(), payloads.openai_chat())])
    session.inject(faults.stream_cut())
    with pytest.raises(FaultNotApplicableError, match="streaming response"):
        session.handle(chat_request(), OPENAI)


def test_truncated_applies_the_provider_mutation(tmp_path):
    session = make(tmp_path, recorded=[(chat_request(), payloads.openai_chat())])
    session.inject(faults.truncated())
    body = served(session.handle(chat_request(), OPENAI)).body
    assert body["choices"][0]["finish_reason"] == "length"


def test_two_mutate_faults_on_one_call_are_rejected(tmp_path):
    session = make(tmp_path)
    session.inject(faults.truncated())
    with pytest.raises(ValueError, match="only one"):
        session.inject(faults.bad_tool_args())


def test_a_fault_for_an_already_served_call_is_rejected(tmp_path):
    session = make(tmp_path, recorded=[(chat_request(), payloads.openai_chat())])
    session.handle(chat_request(), OPENAI)
    with pytest.raises(ValueError, match="already been served"):
        session.inject(faults.timeout(on_call=1))


def test_unused_faults_lists_faults_that_never_fired(tmp_path):
    session = make(tmp_path, recorded=[(chat_request(), payloads.openai_chat())])
    fired, never = faults.rate_limit(times=5), faults.timeout(on_call=3)
    session.inject(fired, never)
    session.handle(chat_request(), OPENAI)
    assert session.unused_faults() == [never]


# --- recording -----------------------------------------------------------------------------


def test_records_when_the_cassette_is_missing(tmp_path):
    session = make(tmp_path, mode="missing")
    step = session.handle(chat_request(), OPENAI)
    assert step == Record(call=1, provider=OPENAI)
    request = replace(
        chat_request(), headers={"authorization": "Bearer x", "content-type": "application/json"}
    )
    response = payloads.openai_chat()
    response.headers["set-cookie"] = "session=1"
    assert served(session.recorded(step, request, response)).body == response.body
    session.save()
    [interaction] = cassette.load(tmp_path / "c.json")
    assert interaction.request.headers == {"content-type": "application/json"}
    assert interaction.response.headers == {"content-type": "application/json"}


def test_mode_all_rerecords_existing_cassettes(tmp_path):
    session = make(tmp_path, mode="all", recorded=[(chat_request(), payloads.openai_chat())])
    assert isinstance(session.handle(chat_request(), OPENAI), Record)


def test_replace_faults_do_not_reach_the_network_while_recording(tmp_path):
    session = make(tmp_path, mode="all")
    session.inject(faults.rate_limit())
    assert served(session.handle(chat_request(), OPENAI)).status == 429
    assert isinstance(session.handle(chat_request(), OPENAI), Record)


def test_mutations_are_not_saved(tmp_path):
    session = make(tmp_path, mode="all")
    session.inject(faults.truncated())
    step = session.handle(chat_request(), OPENAI)
    assert isinstance(step, Record)
    outcome = session.recorded(step, chat_request(), payloads.openai_chat())
    assert served(outcome).body["choices"][0]["finish_reason"] == "length"
    session.save()
    [interaction] = cassette.load(tmp_path / "c.json")
    assert interaction.response.body["choices"][0]["finish_reason"] == "stop"


def test_save_refuses_to_write_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "my-very-secret-key")
    session = make(tmp_path, mode="all")
    request = chat_request("my key is my-very-secret-key")
    session.recorded(session.handle(request, OPENAI), request, payloads.openai_chat())
    with pytest.raises(SecretLeakError, match=r"interactions\[0\]\.request\.body\.messages\[0\]"):
        session.save()
    assert not (tmp_path / "c.json").exists()


def test_nothing_is_written_without_provider_calls(tmp_path):
    session = make(tmp_path, mode="all")
    session.save()
    assert not (tmp_path / "c.json").exists()


def test_llm_facade_exposes_counters_and_path(tmp_path):
    session = make(tmp_path, recorded=[(chat_request(), payloads.openai_chat())])
    llm = LLM(session)
    session.handle(chat_request(), OPENAI)
    assert (llm.calls, llm.attempts, llm.cassette_path) == (1, 1, tmp_path / "c.json")
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/unit/test_session.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llmchaos.session'`.

- [ ] **Step 3: Implement**

`src/llmchaos/session.py`:

```python
"""Per-test state: the fault schedule, the cassette cursor, counters, and the stored error."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, NoReturn

from llmchaos import cassette, matching, redaction
from llmchaos.cassette import Interaction, RecordedRequest, RecordedResponse
from llmchaos.errors import (
    CassetteMismatchError,
    CassetteMissingError,
    FaultNotApplicableError,
    LLMChaosError,
    SecretLeakError,
    UnexpectedCallError,
)
from llmchaos.faults import (
    BadToolArgs,
    Fault,
    HTTPError,
    MutateFault,
    ReplaceFault,
    StreamCut,
    TransportError,
    TransportErrorKind,
    Truncated,
)
from llmchaos.providers.base import Provider

RecordMode = Literal["none", "missing", "all"]


@dataclass(frozen=True)
class Outcome:
    """What the interceptor hands back to the SDK for one HTTP attempt."""

    response: RecordedResponse | None = None
    raise_error: TransportErrorKind | None = None
    cut_after: int | None = None


@dataclass(frozen=True)
class Record:
    """Tells the interceptor to call the real API, then pass the result to ``Session.recorded``."""

    call: int
    provider: Provider


@dataclass
class _Scheduled:
    fault: Fault
    target: int
    remaining: int
    fired: bool = False


class Session:
    """Decides what each provider HTTP attempt in one test receives."""

    def __init__(
        self,
        *,
        cassette_path: Path,
        mode: RecordMode,
        providers: Sequence[Provider],
        ignore: Sequence[str] = (),
    ) -> None:
        self.cassette_path = cassette_path
        self.calls = 0
        self.attempts = 0
        self.error: LLMChaosError | None = None
        self._providers = list(providers)
        self._ignore = list(ignore)
        self._scheduled: list[_Scheduled] = []
        self._lock = threading.Lock()
        exists = cassette_path.exists()
        self._recording = mode == "all" or (mode == "missing" and not exists)
        self._new: list[Interaction] = []
        self._replay: list[Interaction] | None = None
        if not self._recording and exists:
            self._replay = cassette.load(cassette_path)
        self._used = [False] * len(self._replay or [])

    # --- used by the fixture -------------------------------------------------------------

    def inject(self, *faults: Fault) -> None:
        with self._lock:
            for fault in faults:
                target = fault.on_call if fault.on_call is not None else self.calls + 1
                if target <= self.calls:
                    raise ValueError(
                        f"{fault!r} targets call {target}, which has already been served"
                    )
                if isinstance(fault, MutateFault) and any(
                    isinstance(s.fault, MutateFault) and s.target == target
                    for s in self._scheduled
                ):
                    raise ValueError(
                        f"call {target} already has a mutate fault; only one is allowed per call"
                    )
                remaining = fault.times if isinstance(fault, ReplaceFault) else 1
                self._scheduled.append(_Scheduled(fault=fault, target=target, remaining=remaining))

    def ignore(self, *paths: str) -> None:
        with self._lock:
            self._ignore.extend(paths)

    def unused_faults(self) -> list[Fault]:
        return [s.fault for s in self._scheduled if not s.fired]

    # --- used by the interceptor ---------------------------------------------------------

    def provider_for(self, host: str) -> Provider | None:
        return next((p for p in self._providers if p.handles(host)), None)

    def handle(self, request: RecordedRequest, provider: Provider) -> Outcome | Record:
        with self._lock:
            self.attempts += 1
            call = self.calls + 1
            fault = self._take_replace_fault(call)
            if fault is not None:
                return self._replace(fault, provider)
            if self._recording:
                self.calls = call
                return Record(call=call, provider=provider)
            interaction = self._match(request)
            self.calls = call
            return self._mutate(call, provider, interaction.response)

    def recorded(
        self, step: Record, request: RecordedRequest, response: RecordedResponse
    ) -> Outcome:
        interaction = Interaction(
            request=replace(
                request,
                headers=redaction.filter_headers(request.headers, redaction.REQUEST_HEADER_ALLOWLIST),
            ),
            response=replace(
                response,
                headers=redaction.filter_headers(
                    response.headers, redaction.RESPONSE_HEADER_ALLOWLIST
                ),
            ),
        )
        with self._lock:
            self._new.append(interaction)
            return self._mutate(step.call, step.provider, interaction.response)

    # --- used by the plugin --------------------------------------------------------------

    def save(self) -> None:
        if not self._recording or not self._new:
            return
        leaks = redaction.find_secrets(cassette.to_document(self._new))
        if leaks:
            raise SecretLeakError(
                f"Refusing to write {self.cassette_path}: possible API key at {', '.join(leaks)}"
            )
        cassette.save(self.cassette_path, self._new)

    # --- internals -----------------------------------------------------------------------

    def _take_replace_fault(self, call: int) -> ReplaceFault | None:
        for scheduled in self._scheduled:
            fault = scheduled.fault
            if scheduled.target == call and isinstance(fault, ReplaceFault) and scheduled.remaining:
                scheduled.remaining -= 1
                scheduled.fired = True
                return fault
        return None

    def _replace(self, fault: ReplaceFault, provider: Provider) -> Outcome:
        if isinstance(fault, TransportError):
            return Outcome(raise_error=fault.kind)
        if isinstance(fault, HTTPError):
            return Outcome(response=provider.error_response(fault.kind, fault.retry_after))
        raise TypeError(f"unknown replace fault {fault!r}")

    def _match(self, request: RecordedRequest) -> Interaction:
        if self._replay is None:
            self._fail(
                CassetteMissingError(
                    f"No cassette at {self.cassette_path}. Record it by running this test once "
                    "with an API key and --llm-record=missing."
                )
            )
        for index, interaction in enumerate(self._replay):
            if not self._used[index] and matching.requests_match(
                request, interaction.request, self._ignore
            ):
                self._used[index] = True
                return interaction
        unused = [i for i, used in zip(self._replay, self._used, strict=True) if not used]
        if not unused:
            self._fail(
                UnexpectedCallError(
                    f"{request.method} {request.url}: this test made more calls than "
                    f"{self.cassette_path.name} recorded ({len(self._replay)}). "
                    "Re-record with --llm-record=all."
                )
            )
        diff = matching.request_diff(request, unused[0].request, self._ignore)
        self._fail(
            CassetteMismatchError(
                f"Request does not match {self.cassette_path.name}. "
                f"Re-record with --llm-record=all.\n{diff}"
            )
        )

    def _mutate(self, call: int, provider: Provider, response: RecordedResponse) -> Outcome:
        scheduled = next(
            (s for s in self._scheduled if s.target == call and isinstance(s.fault, MutateFault)),
            None,
        )
        if scheduled is None:
            return Outcome(response=response)
        scheduled.fired = True
        scheduled.remaining = 0
        fault = scheduled.fault
        try:
            if isinstance(fault, StreamCut):
                if response.events is None or len(response.events) <= fault.after_events:
                    raise FaultNotApplicableError(
                        f"needs a streaming response with more than {fault.after_events} events"
                    )
                return Outcome(response=response, cut_after=fault.after_events)
            if isinstance(fault, Truncated):
                return Outcome(response=provider.truncate(response))
            if isinstance(fault, BadToolArgs):
                return Outcome(response=provider.break_tool_args(response))
        except FaultNotApplicableError as error:
            self._fail(FaultNotApplicableError(f"{fault!r} on call {call}: {error}"))
        raise TypeError(f"unknown mutate fault {fault!r}")

    def _fail(self, error: LLMChaosError) -> NoReturn:
        """Remember the first error (the plugin reports it) and raise it to the SDK."""
        if self.error is None:
            self.error = error
        raise error


class LLM:
    """The object the ``llm`` fixture returns."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def inject(self, *faults: Fault) -> None:
        """Schedule faults; each targets ``on_call`` or, by default, the next call."""
        self._session.inject(*faults)

    def ignore(self, *paths: str) -> None:
        """Exclude dotted request-body paths from matching for the rest of this test."""
        self._session.ignore(*paths)

    @property
    def calls(self) -> int:
        """Logical calls served from the cassette or recorded so far."""
        return self._session.calls

    @property
    def attempts(self) -> int:
        """HTTP attempts that reached the interceptor, retries included."""
        return self._session.attempts

    @property
    def cassette_path(self) -> Path:
        return self._session.cassette_path
```

Replace `src/llmchaos/__init__.py` with:

```python
"""Record real LLM responses once, replay them offline, and inject provider-accurate failures."""

from llmchaos import errors, faults
from llmchaos.session import LLM

__all__ = ["LLM", "__version__", "errors", "faults"]
__version__ = "0.1.0"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit -v && uv run pyright`
Expected: all unit tests pass (`test_session.py`: 25 passed); pyright `0 errors`.

- [ ] **Step 5: Commit**

```bash
uv run ruff format . && uv run ruff check --fix .
git add src/llmchaos/session.py src/llmchaos/__init__.py tests/unit/test_session.py
git commit -m "feat: add session with fault scheduling, replay and recording" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Interceptor for `httpx` and `httpx2`

**Files:**
- Create: `src/llmchaos/interceptor.py`
- Test: `tests/unit/test_interceptor.py`

**Interfaces:**
- Consumes: `Session.{provider_for, handle, recorded}`, `Outcome`, `Record`, `RecordedRequest.from_raw`, `RecordedResponse.{from_raw, chunks}`.
- Produces: `available_libraries() -> list[ModuleType]`; `install(monkeypatch: pytest.MonkeyPatch, session: Session) -> None`. These patch `HTTPTransport.handle_request` and `AsyncHTTPTransport.handle_async_request` on every importable library in `("httpx", "httpx2")`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_interceptor.py`:

```python
import asyncio
import gzip
import json

import payloads
import pytest

from llmchaos import cassette, faults, interceptor
from llmchaos.cassette import Interaction, RecordedRequest
from llmchaos.errors import CassetteMissingError
from llmchaos.providers.anthropic import AnthropicProvider
from llmchaos.providers.openai import OpenAIProvider
from llmchaos.session import Session

CHAT_URL = "https://api.openai.com/v1/chat/completions"
BODY = {"model": "gpt-5", "messages": [{"role": "user", "content": "hi"}]}


@pytest.fixture(params=["httpx", "httpx2"])
def lib(request):
    return pytest.importorskip(request.param)


@pytest.fixture
def network(monkeypatch, lib):
    """A fake network. It must be patched in before ``interceptor.install`` wraps it."""
    seen = []

    def handle_request(transport, request):
        request.read()
        seen.append(str(request.url))
        if request.url.host == "api.openai.com":
            content = gzip.compress(json.dumps(payloads.openai_chat().body).encode())
            headers = {
                "content-type": "application/json",
                "content-encoding": "gzip",
                "set-cookie": "session=1",
            }
            return lib.Response(200, headers=headers, content=content)
        return lib.Response(200, text="ok")

    monkeypatch.setattr(lib.HTTPTransport, "handle_request", handle_request)
    return seen


def make_session(tmp_path, mode="missing", recorded=()):
    path = tmp_path / "c.json"
    if recorded:
        cassette.save(path, list(recorded))
    return Session(cassette_path=path, mode=mode, providers=[OpenAIProvider(), AnthropicProvider()])


def chat_interaction(response):
    request = RecordedRequest.from_raw(
        "POST", CHAT_URL, {"content-type": "application/json"}, json.dumps(BODY).encode()
    )
    return Interaction(request, response)


def test_other_hosts_pass_through(monkeypatch, lib, network, tmp_path):
    session = make_session(tmp_path)
    interceptor.install(monkeypatch, session)
    assert lib.Client().get("https://example.com/").text == "ok"
    assert session.attempts == 0


def test_records_decoded_bodies_without_secret_headers(monkeypatch, lib, network, tmp_path):
    session = make_session(tmp_path)
    interceptor.install(monkeypatch, session)
    response = lib.Client().post(CHAT_URL, json=BODY, headers={"authorization": "Bearer sk-test"})
    assert response.json() == payloads.openai_chat().body
    session.save()
    [interaction] = cassette.load(tmp_path / "c.json")
    assert interaction.request.headers == {"content-type": "application/json"}
    assert interaction.request.body == BODY
    assert interaction.response.headers == {"content-type": "application/json"}
    assert interaction.response.body == payloads.openai_chat().body


def test_replays_without_the_network(monkeypatch, lib, network, tmp_path):
    session = make_session(tmp_path, "none", [chat_interaction(payloads.openai_chat())])
    interceptor.install(monkeypatch, session)
    assert lib.Client().post(CHAT_URL, json=BODY).json() == payloads.openai_chat().body
    assert network == []


def test_replays_async(monkeypatch, lib, tmp_path):
    session = make_session(tmp_path, "none", [chat_interaction(payloads.openai_chat())])
    interceptor.install(monkeypatch, session)

    async def go():
        async with lib.AsyncClient() as client:
            return (await client.post(CHAT_URL, json=BODY)).json()

    assert asyncio.run(go()) == payloads.openai_chat().body


def test_http_error_faults_become_responses(monkeypatch, lib, tmp_path):
    session = make_session(tmp_path, "none")
    session.inject(faults.rate_limit())
    interceptor.install(monkeypatch, session)
    response = lib.Client().post(CHAT_URL, json=BODY)
    assert response.status_code == 429
    assert response.headers["retry-after-ms"] == "1"


@pytest.mark.parametrize(
    ("fault", "error"),
    [(faults.timeout(), "ReadTimeout"), (faults.connection_error(), "ConnectError")],
)
def test_transport_faults_raise_the_library_errors(monkeypatch, lib, tmp_path, fault, error):
    session = make_session(tmp_path, "none")
    session.inject(fault)
    interceptor.install(monkeypatch, session)
    with pytest.raises(getattr(lib, error)):
        lib.Client().post(CHAT_URL, json=BODY)


def test_stream_cut_raises_after_the_allowed_events(monkeypatch, lib, tmp_path):
    stream = payloads.openai_chat_stream()
    session = make_session(tmp_path, "none", [chat_interaction(stream)])
    session.inject(faults.stream_cut(after_events=2))
    interceptor.install(monkeypatch, session)
    received = []
    with (
        pytest.raises(lib.RemoteProtocolError),
        lib.Client().stream("POST", CHAT_URL, json=BODY) as response,
    ):
        for chunk in response.iter_bytes():
            received.append(chunk)
    assert b"".join(received) == b"".join(stream.chunks()[:2])


def test_llmchaos_errors_propagate(monkeypatch, lib, tmp_path):
    session = make_session(tmp_path, "none")
    interceptor.install(monkeypatch, session)
    with pytest.raises(CassetteMissingError):
        lib.Client().post(CHAT_URL, json=BODY)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/unit/test_interceptor.py -v`
Expected: FAIL with `ImportError: cannot import name 'interceptor'`.

- [ ] **Step 3: Implement**

`src/llmchaos/interceptor.py`:

```python
"""Patch the httpx and httpx2 transports so provider requests go through the session.

The SDKs moved from ``httpx`` to ``httpx2`` (an API-compatible fork) in mid-2026, so
both are patched whenever importable. This is the only module that touches either
library, and it does so through ``importlib`` so neither is a runtime dependency.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

import pytest

from llmchaos.cassette import RecordedRequest, RecordedResponse
from llmchaos.session import Outcome, Record, Session

LIBRARIES = ("httpx", "httpx2")


def available_libraries() -> list[ModuleType]:
    found: list[ModuleType] = []
    for name in LIBRARIES:
        try:
            found.append(importlib.import_module(name))
        except ImportError:
            continue
    return found


def install(monkeypatch: pytest.MonkeyPatch, session: Session) -> None:
    """Route provider traffic through ``session`` until ``monkeypatch`` is undone."""
    for lib in available_libraries():
        _patch_sync(monkeypatch, lib, session)
        _patch_async(monkeypatch, lib, session)


def _patch_sync(monkeypatch: pytest.MonkeyPatch, lib: ModuleType, session: Session) -> None:
    original = lib.HTTPTransport.handle_request

    def handle_request(transport: Any, request: Any) -> Any:
        provider = session.provider_for(request.url.host)
        if provider is None:
            return original(transport, request)
        recorded_request = _to_recorded(request, request.read())
        step = session.handle(recorded_request, provider)
        if isinstance(step, Record):
            real = original(transport, request)
            content = real.read()  # decodes content-encoding
            response = RecordedResponse.from_raw(real.status_code, real.headers, content)
            step = session.recorded(step, recorded_request, response)
        return _build_response(lib, step, request)

    monkeypatch.setattr(lib.HTTPTransport, "handle_request", handle_request)


def _patch_async(monkeypatch: pytest.MonkeyPatch, lib: ModuleType, session: Session) -> None:
    original = lib.AsyncHTTPTransport.handle_async_request

    async def handle_async_request(transport: Any, request: Any) -> Any:
        provider = session.provider_for(request.url.host)
        if provider is None:
            return await original(transport, request)
        recorded_request = _to_recorded(request, await request.aread())
        step = session.handle(recorded_request, provider)
        if isinstance(step, Record):
            real = await original(transport, request)
            content = await real.aread()
            response = RecordedResponse.from_raw(real.status_code, real.headers, content)
            step = session.recorded(step, recorded_request, response)
        return _build_response(lib, step, request)

    monkeypatch.setattr(lib.AsyncHTTPTransport, "handle_async_request", handle_async_request)


def _to_recorded(request: Any, content: bytes) -> RecordedRequest:
    return RecordedRequest.from_raw(request.method, str(request.url), request.headers, content)


def _build_response(lib: ModuleType, outcome: Outcome, request: Any) -> Any:
    if outcome.raise_error == "timeout":
        raise lib.ReadTimeout("Read timed out (simulated by llmchaos)", request=request)
    if outcome.raise_error == "connection_error":
        raise lib.ConnectError("Connection failed (simulated by llmchaos)", request=request)
    response = outcome.response
    if response is None:
        raise RuntimeError("llmchaos: an outcome must carry a response or an error")
    chunks = response.chunks()
    if outcome.cut_after is not None:
        stream = _cut_stream(lib, chunks[: outcome.cut_after])
    else:
        stream = lib.ByteStream(b"".join(chunks))
    return lib.Response(response.status, headers=response.headers, stream=stream, request=request)


def _cut_stream(lib: ModuleType, chunks: list[bytes]) -> Any:
    """A body stream that yields ``chunks`` and then fails like a dropped connection."""

    def error() -> Exception:
        return lib.RemoteProtocolError(
            "peer closed connection without sending complete message body "
            "(simulated by llmchaos)"
        )

    class CutStream(lib.SyncByteStream, lib.AsyncByteStream):  # pyright: ignore[reportUntypedBaseClass]
        def __iter__(self) -> Any:
            yield from chunks
            raise error()

        async def __aiter__(self) -> Any:
            for chunk in chunks:
                yield chunk
            raise error()

    return CutStream()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_interceptor.py -v && uv run pyright`
Expected: 18 passed (9 tests × 2 libraries); pyright `0 errors`.

- [ ] **Step 5: Commit**

```bash
uv run ruff format . && uv run ruff check --fix .
git add src/llmchaos/interceptor.py tests/unit/test_interceptor.py
git commit -m "feat: intercept httpx and httpx2 transports" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: pytest plugin and the test harness

**Files:**
- Modify: `src/llmchaos/plugin.py` (replace the stub)
- Create: `tests/harness.py`
- Test: `tests/unit/test_plugin.py`

**Interfaces:**
- Consumes: `Session`, `LLM`, `RecordMode`, `interceptor.install`, both providers, `UnusedFaultError`, `LLMChaosError`.
- Produces:
  - `plugin.py`: `SESSION_KEY`; `pytest_addoption`; `cassette_path(nodeid: str, test_file: Path, cassette_dir: str) -> Path`; the `llm` fixture; the `pytest_runtest_call` wrapper.
  - `tests/harness.py`: `PAYLOADS: Path`; `CONFTEST: str` (an inner conftest with an autouse `fake_network` fixture covering both libraries); `prepare(pytester, source)`; `run(pytester, monkeypatch, *, offline: bool, mode: str) -> pytest.RunResult`; `record_then_replay(pytester, monkeypatch, source, *, passed: int)`.

- [ ] **Step 1: Write the harness**

`tests/harness.py`:

```python
"""Test harness: run an inner test file against a fake network, then replay it offline."""

from __future__ import annotations

from pathlib import Path

import pytest

PAYLOADS = Path(__file__).with_name("payloads.py")

CONFTEST = '''
import importlib
import json
import os

import pytest

import payloads

ROUTES = {
    ("/v1/chat/completions", False, False): payloads.openai_chat,
    ("/v1/chat/completions", False, True): payloads.openai_chat_tool,
    ("/v1/chat/completions", True, False): payloads.openai_chat_stream,
    ("/v1/chat/completions", True, True): payloads.openai_chat_tool_stream,
    ("/v1/responses", False, False): payloads.openai_response,
    ("/v1/responses", False, True): payloads.openai_response_tool,
    ("/v1/messages", False, False): payloads.anthropic_message,
    ("/v1/messages", False, True): payloads.anthropic_tool_message,
    ("/v1/messages", True, False): payloads.anthropic_stream,
    ("/v1/messages", True, True): payloads.anthropic_tool_stream,
}


@pytest.fixture(autouse=True)
def fake_network(monkeypatch):
    """Autouse, so it is patched in before the llm fixture wraps the transports."""
    offline_hits = []
    for name in ("httpx", "httpx2"):
        try:
            lib = importlib.import_module(name)
        except ImportError:
            continue

        def respond(request, lib=lib):
            if os.environ.get("LLMCHAOS_OFFLINE"):
                offline_hits.append(str(request.url))
                raise lib.ConnectError("offline", request=request)
            body = json.loads(request.content or b"{}")
            key = (request.url.path, bool(body.get("stream")), bool(body.get("tools")))
            recorded = ROUTES[key]()
            content = b"".join(recorded.chunks())
            return lib.Response(recorded.status, headers=recorded.headers, content=content)

        def handle_request(transport, request, respond=respond):
            request.read()
            return respond(request)

        async def handle_async_request(transport, request, respond=respond):
            await request.aread()
            return respond(request)

        monkeypatch.setattr(lib.HTTPTransport, "handle_request", handle_request)
        monkeypatch.setattr(lib.AsyncHTTPTransport, "handle_async_request", handle_async_request)
    yield
    assert not offline_hits, f"replay run reached the network: {offline_hits}"
'''


def prepare(pytester: pytest.Pytester, source: str) -> None:
    """Write the shared payloads, the fake-network conftest, and ``test_app.py``."""
    pytester.syspathinsert()
    pytester.makepyfile(payloads=PAYLOADS.read_text(encoding="utf-8"))
    pytester.makeconftest(CONFTEST)
    pytester.makepyfile(test_app=source)


def run(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, *, offline: bool, mode: str
) -> pytest.RunResult:
    if offline:
        monkeypatch.setenv("LLMCHAOS_OFFLINE", "1")
    else:
        monkeypatch.delenv("LLMCHAOS_OFFLINE", raising=False)
    return pytester.runpytest(f"--llm-record={mode}")


def record_then_replay(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, source: str, *, passed: int
) -> None:
    """Record against the fake network, then replay with the network forbidden."""
    prepare(pytester, source)
    run(pytester, monkeypatch, offline=False, mode="all").assert_outcomes(passed=passed)
    run(pytester, monkeypatch, offline=True, mode="none").assert_outcomes(passed=passed)
```

- [ ] **Step 2: Write the failing tests**

`tests/unit/test_plugin.py`:

```python
import json

import pytest
from harness import prepare, record_then_replay, run

pytest.importorskip("httpx2")

CLIENT = """
import httpx2
import pytest

from llmchaos import errors, faults

URL = "https://api.openai.com/v1/chat/completions"
BODY = {"model": "gpt-5", "messages": [{"role": "user", "content": "hi"}]}


def post():
    return httpx2.Client().post(URL, json=BODY, headers={"authorization": "Bearer sk-live-secret"})
"""


def test_help_lists_the_record_option(pytester):
    pytester.runpytest("--help").stdout.fnmatch_lines(["*--llm-record*"])


def test_cassette_paths(pytester):
    pytester.makepyfile(
        test_paths="""
from pathlib import Path

import pytest

HERE = Path(__file__).parent / "cassettes" / "test_paths"


def test_plain(llm):
    assert llm.cassette_path == HERE / "test_plain.json"


class TestGroup:
    def test_method(self, llm):
        assert llm.cassette_path == HERE / "TestGroup.test_method.json"


@pytest.mark.parametrize("x", ["a/b"])
def test_param(llm, x):
    assert llm.cassette_path == HERE / "test_param[a_b].json"
"""
    )
    pytester.runpytest().assert_outcomes(passed=3)


def test_cassette_dir_ini_option(pytester):
    pytester.makeini("[pytest]\nllmchaos_cassette_dir = recordings\n")
    pytester.makepyfile(
        test_ini="""
from pathlib import Path


def test_dir(llm):
    assert llm.cassette_path == Path(__file__).parent / "recordings" / "test_ini" / "test_dir.json"
"""
    )
    pytester.runpytest().assert_outcomes(passed=1)


def test_records_then_replays_without_secrets(pytester, monkeypatch):
    source = CLIENT + """
def test_chat(llm):
    assert post().json()["choices"][0]["message"]["content"] == "The refund was approved yesterday."
"""
    record_then_replay(pytester, monkeypatch, source, passed=1)
    path = pytester.path / "cassettes" / "test_app" / "test_chat.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["interactions"][0]["request"]["headers"] == {"content-type": "application/json"}
    assert "sk-live-secret" not in path.read_text(encoding="utf-8")


def test_replay_only_without_a_cassette_fails_clearly(pytester, monkeypatch):
    prepare(pytester, CLIENT + "\ndef test_chat(llm):\n    post()\n")
    result = run(pytester, monkeypatch, offline=True, mode="none")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*CassetteMissingError*--llm-record=missing*"])


def test_unused_fault_fails_the_test(pytester, monkeypatch):
    prepare(
        pytester,
        CLIENT
        + """
def test_chat(llm):
    llm.inject(faults.rate_limit(on_call=2))
    post()
""",
    )
    result = run(pytester, monkeypatch, offline=False, mode="all")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*UnusedFaultError*", "*kind='rate_limit'*"])


def test_error_swallowed_by_the_app_still_fails_the_test(pytester, monkeypatch):
    prepare(
        pytester,
        CLIENT
        + """
def test_chat(llm):
    try:
        post()
    except Exception:
        pass  # an app with a broad fallback
""",
    )
    result = run(pytester, monkeypatch, offline=True, mode="none")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*CassetteMissingError*"])


def test_wrapped_error_reports_the_real_cause(pytester, monkeypatch):
    prepare(
        pytester,
        CLIENT
        + """
def test_chat(llm):
    try:
        post()
    except errors.LLMChaosError as exc:
        raise RuntimeError("SDK wrapped it") from exc
""",
    )
    result = run(pytester, monkeypatch, offline=True, mode="none")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["E*llmchaos.errors.CassetteMissingError*"])
    result.stdout.no_fnmatch_line("E*RuntimeError: SDK wrapped it*")
```

- [ ] **Step 3: Run them to verify they fail**

Run: `uv run pytest tests/unit/test_plugin.py -v`
Expected: FAIL. `test_help_lists_the_record_option` fails because `--llm-record` is missing, and the others fail with `fixture 'llm' not found`.

- [ ] **Step 4: Implement the plugin**

Replace `src/llmchaos/plugin.py` with:

```python
"""pytest entry point: command-line options, the ``llm`` fixture, and failure reporting."""

from __future__ import annotations

import re
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import cast

import pytest

from llmchaos import interceptor
from llmchaos.errors import LLMChaosError, UnusedFaultError
from llmchaos.providers.anthropic import AnthropicProvider
from llmchaos.providers.openai import OpenAIProvider
from llmchaos.session import LLM, RecordMode, Session

SESSION_KEY = pytest.StashKey[Session]()
_UNSAFE = re.compile(r"[^A-Za-z0-9\-_.\[\]]")


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("llmchaos", "LLM record/replay and fault injection")
    group.addoption(
        "--llm-record",
        dest="llm_record",
        choices=("none", "missing", "all"),
        default="missing",
        help="none: replay only; missing (default): record tests that have no cassette; "
        "all: re-record every test.",
    )
    parser.addini(
        "llmchaos_cassette_dir", "Cassette directory, relative to each test file.", default="cassettes"
    )
    parser.addini(
        "llmchaos_ignore", "Dotted request-body paths ignored when matching.", type="linelist", default=[]
    )
    parser.addini(
        "llmchaos_openai_hosts", "Extra hosts treated as OpenAI-compatible.", type="linelist", default=[]
    )
    parser.addini(
        "llmchaos_anthropic_hosts", "Extra hosts treated as Anthropic.", type="linelist", default=[]
    )


def cassette_path(nodeid: str, test_file: Path, cassette_dir: str) -> Path:
    """``<test dir>/<cassette dir>/<module>/<Class.>test[params].json``."""
    test_name = ".".join(nodeid.split("::")[1:])
    return test_file.parent / cassette_dir / test_file.stem / f"{_UNSAFE.sub('_', test_name)}.json"


@pytest.fixture
def llm(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[LLM]:
    """Record or replay this test's OpenAI/Anthropic traffic, and inject faults into it."""
    config = request.config
    session = Session(
        cassette_path=cassette_path(
            request.node.nodeid,
            request.node.path,
            cast(str, config.getini("llmchaos_cassette_dir")),
        ),
        mode=cast(RecordMode, config.getoption("llm_record")),
        providers=[
            OpenAIProvider(cast("list[str]", config.getini("llmchaos_openai_hosts"))),
            AnthropicProvider(cast("list[str]", config.getini("llmchaos_anthropic_hosts"))),
        ],
        ignore=cast("list[str]", config.getini("llmchaos_ignore")),
    )
    interceptor.install(monkeypatch, session)
    request.node.stash[SESSION_KEY] = session
    yield LLM(session)
    session.save()


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item: pytest.Item) -> Generator[None, object, object]:
    """Report the stored error when an SDK hid it, and enforce strict mode on success."""
    try:
        result = yield
    except Exception as exc:
        session = item.stash.get(SESSION_KEY, None)
        if session is None or session.error is None or session.error is exc:
            raise
        if _caused_by_llmchaos(exc):
            # e.g. APIConnectionError wrapping one of our errors after SDK retries.
            raise session.error from None
        raise session.error from exc
    session = item.stash.get(SESSION_KEY, None)
    if session is not None:
        if session.error is not None:
            raise session.error  # the application swallowed it
        unused = session.unused_faults()
        if unused:
            listed = "\n".join(f"  - {fault!r}" for fault in unused)
            raise UnusedFaultError(
                "These faults never fired, so the code under test never made the call "
                f"they target:\n{listed}"
            )
    return result


def _caused_by_llmchaos(exc: BaseException) -> bool:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, LLMChaosError):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests -v && uv run pyright`
Expected: every test so far passes (`test_plugin.py`: 8 passed); pyright `0 errors`.

- [ ] **Step 6: Commit**

```bash
uv run ruff format . && uv run ruff check --fix .
git add src/llmchaos/plugin.py tests/harness.py tests/unit/test_plugin.py
git commit -m "feat: add llm fixture, record option and failure reporting" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: OpenAI SDK integration tests

**Files:**
- Create: `tests/integration/conftest.py`
- Test: `tests/integration/test_openai_sdk.py`

**Interfaces:**
- Consumes: `harness.{prepare, run, record_then_replay}`; the real `openai` SDK.
- Produces: nothing new. This task is evidence: each inner test runs twice (record against the fake network, then replay offline).

If an inner test fails because the SDK surfaces a fault differently than assumed here, use superpowers:systematic-debugging. Fix the code or update the spec. Never weaken an assertion silently.

- [ ] **Step 1: Write the outer conftest**

`tests/integration/conftest.py`:

```python
"""Import the SDKs once in the outer process so pytester's in-process runs reuse them.

pytester restores ``sys.modules`` after each inner run. Modules imported first inside an
inner run would otherwise be re-imported every time, which is slow and can break
C extensions.
"""

import anthropic  # noqa: F401
import openai  # noqa: F401
```

- [ ] **Step 2: Write the tests**

`tests/integration/test_openai_sdk.py`:

```python
from harness import prepare, record_then_replay, run

SCENARIOS = '''
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
'''

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
```

- [ ] **Step 3: Run the tests**

Run: `uv run pytest tests/integration/test_openai_sdk.py -v`
Expected: 2 passed (the scenario test covers 18 inner tests × 2 runs). It takes a few seconds, because timeout retries use the SDK's own backoff.

- [ ] **Step 4: Run them against the legacy httpx-based SDK too**

Run: `uv pip install "openai==2.46.0" "anthropic==0.125.0" && uv run --no-sync pytest tests -v; uv sync`
Expected: all tests pass on the legacy SDKs. `uv sync` then restores the current SDKs.

- [ ] **Step 5: Commit**

```bash
uv run ruff format . && uv run ruff check --fix .
git add tests/integration
git commit -m "test: verify faults against the real OpenAI SDK" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 13: Anthropic SDK integration tests

**Files:**
- Test: `tests/integration/test_anthropic_sdk.py`

**Interfaces:**
- Consumes: `harness.{prepare, run, record_then_replay}`; the real `anthropic` SDK.

The systematic-debugging rule from Task 12 applies here too.

- [ ] **Step 1: Write the tests**

`tests/integration/test_anthropic_sdk.py`:

```python
from harness import prepare, record_then_replay, run

SCENARIOS = '''
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
    with pytest.raises(anthropic.InternalServerError) as error:
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
'''

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
```

- [ ] **Step 2: Run the tests on both SDK lanes**

Run: `uv run pytest tests/integration -v`
Expected: 4 passed.

Run: `uv pip install "openai==2.46.0" "anthropic==0.125.0" && uv run --no-sync pytest tests -v; uv sync`
Expected: all tests pass on the legacy SDKs too.

- [ ] **Step 3: Commit**

```bash
uv run ruff format . && uv run ruff check --fix .
git add tests/integration/test_anthropic_sdk.py
git commit -m "test: verify faults against the real Anthropic SDK" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 14: Example app, README, CONTRIBUTING, CHANGELOG

**Files:**
- Create: `examples/support_bot/app.py`, `examples/support_bot/test_app.py`, `examples/support_bot/cassettes/**` (generated)
- Modify: `README.md` (full rewrite)
- Create: `CONTRIBUTING.md`, `CHANGELOG.md`

**Interfaces:**
- Consumes: the `llm` fixture, `faults`, and `harness.CONFTEST` (only while generating cassettes).

- [ ] **Step 1: Write the example app and its tests**

`examples/support_bot/app.py`:

```python
"""A tiny support bot: summarizes a ticket with OpenAI and falls back to Anthropic."""

from __future__ import annotations

import anthropic
import openai

OPENAI_MODEL = "gpt-5"
ANTHROPIC_MODEL = "claude-sonnet-5-5"


class ServiceBusyError(Exception):
    """Both providers are unavailable right now."""


def summarize(
    ticket: str, *, openai_client: openai.OpenAI, anthropic_client: anthropic.Anthropic
) -> str:
    prompt = f"Summarize this support ticket in one sentence:\n\n{ticket}"
    try:
        completion = openai_client.chat.completions.create(
            model=OPENAI_MODEL, messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content or ""
    except (openai.RateLimitError, openai.InternalServerError, openai.APIConnectionError):
        pass
    try:
        message = anthropic_client.messages.create(
            model=ANTHROPIC_MODEL, max_tokens=256, messages=[{"role": "user", "content": prompt}]
        )
    except (
        anthropic.RateLimitError,
        anthropic.InternalServerError,
        anthropic.APIConnectionError,
    ) as error:
        raise ServiceBusyError("OpenAI and Anthropic are both unavailable") from error
    return "".join(block.text for block in message.content if block.type == "text")
```

`examples/support_bot/test_app.py`:

```python
"""Tests for the support bot.

The committed cassettes replay offline. To re-record them against the real APIs:

    OPENAI_API_KEY=... ANTHROPIC_API_KEY=... pytest examples --llm-record=all
"""

import os

import anthropic
import openai
import pytest
from app import ServiceBusyError, summarize

from llmchaos import faults

TICKET = "Customer 1042 asked for a refund on order 88; it was approved yesterday."


@pytest.fixture
def clients():
    return {
        "openai_client": openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "replay-only")),
        "anthropic_client": anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY", "replay-only")
        ),
    }


def test_summarizes_with_openai(llm, clients):
    assert summarize(TICKET, **clients)
    assert llm.calls == 1


def test_falls_back_to_anthropic_when_openai_is_rate_limited(llm, clients):
    llm.inject(faults.rate_limit(times=3))  # more than the OpenAI SDK's 2 retries
    assert summarize(TICKET, **clients)
    assert llm.attempts == 4  # 3 rate-limited OpenAI attempts, then 1 Anthropic call


def test_reports_busy_when_both_providers_fail(llm, clients):
    llm.inject(faults.rate_limit(times=3), faults.overloaded(times=3))
    with pytest.raises(ServiceBusyError):
        summarize(TICKET, **clients)
```

- [ ] **Step 2: Generate the example cassettes from the canonical payloads**

There are no real API keys here, so record against the fake network. Put its conftest in place temporarily, record, then delete it:

```bash
uv run python -c "import sys; sys.path.insert(0, 'tests'); import harness, pathlib; pathlib.Path('examples/support_bot/conftest.py').write_text(harness.CONFTEST)"
uv run pytest examples --llm-record=all -v
rm examples/support_bot/conftest.py
```

Expected: 3 passed. `examples/support_bot/cassettes/test_app/` now contains `test_summarizes_with_openai.json` and `test_falls_back_to_anthropic_when_openai_is_rate_limited.json`. The third test serves no call, so it has no cassette.

- [ ] **Step 3: Verify offline replay with the plain plugin**

Run: `uv run pytest examples --llm-record=none -v`
Expected: 3 passed.

- [ ] **Step 4: Write the README**

Replace `README.md` with:

````markdown
# pytest-llmchaos

**Give your LLM calls a bad day before production does.**

pytest-llmchaos is a pytest plugin for code that calls the OpenAI or Anthropic Python SDKs. It records real responses once, replays them offline in every later run, and lets each test inject the failures that break LLM features in production:

- rate limits and overloaded servers
- timeouts and dropped connections
- streams that die halfway
- answers cut off at the token limit
- tool calls with broken arguments

The faults look exactly like the real thing to the SDKs, so they raise their real exception classes and run their real retry logic. Your error handling is tested the way it runs in production, with no network, no API key and no cost.

## Install

```bash
pip install pytest-llmchaos
```

The plugin activates automatically. Use the `llm` fixture in any test that calls a provider.

## Example

```python
import pytest
from llmchaos import faults

def test_summarize(llm):
    assert "refund" in summarize(ticket)          # replayed from cassettes/

def test_survives_rate_limits(llm):
    llm.inject(faults.rate_limit(times=2))        # two real-looking 429s, then the recorded answer
    assert summarize(ticket)
    assert llm.attempts == 3                      # proves the SDK retried

def test_falls_back_when_openai_keeps_failing(llm):
    llm.inject(faults.overloaded(times=3))        # more than the SDK's 2 default retries
    assert summarize(ticket) == FALLBACK_SUMMARY

def test_handles_a_cut_stream(llm):
    llm.inject(faults.stream_cut(after_events=4))
    with pytest.raises(PartialResponseError):     # your app's own error
        stream_summary(ticket)
```

The first run with an API key records real responses to `cassettes/<module>/<test>.json` next to the test file. Every later run replays them. [`examples/support_bot`](examples/support_bot) is a complete, runnable example.

## Faults

| Fault | What the SDK sees |
|---|---|
| `rate_limit(times=1, retry_after=0.001)` | HTTP 429 with the provider's error body and `retry-after` headers |
| `overloaded(times=1)` | Anthropic HTTP 529 `overloaded_error`; OpenAI HTTP 503 |
| `server_error(times=1)` | HTTP 500 |
| `timeout(times=1)` | `ReadTimeout` from the HTTP library |
| `connection_error(times=1)` | `ConnectError` from the HTTP library |
| `stream_cut(after_events=1)` | The recorded stream stops after N events with `RemoteProtocolError` |
| `truncated()` | The first half of the text, ending with `finish_reason="length"` / `stop_reason="max_tokens"` |
| `bad_tool_args()` | OpenAI: tool arguments that are not valid JSON. Anthropic: a tool input with a key missing |

Every fault also takes `on_call=n` to target the nth call in the test. By default it targets the next call.

- **Replace faults** (the first five) answer an HTTP attempt themselves and don't use up the recording. `rate_limit(times=2)` means two 429s, then the real answer on the SDK's third attempt. Use a `times` larger than the SDK's retry budget to test the "gives up" path.
- **Mutate faults** (the last three) alter the recorded response of the call they target.
- **Strict by default.** A test fails if a fault it injected never fired, because that means the code never made the call you thought it did.
- **Fast by default.** Synthesized HTTP errors carry `retry-after-ms: 1`, so SDK retries are instant. `timeout` and `connection_error` have no response to carry that header, so retries after them use the SDK's own backoff (about 0.5–1 s each).
- Synthesized error messages end with `(simulated by llmchaos)`, so they can't be mistaken for real outages in logs.

## Recording

| `--llm-record=` | Cassette exists | Cassette missing |
|---|---|---|
| `none` | replay | fail with `CassetteMissingError` |
| `missing` (default) | replay | record |
| `all` | re-record | record |

Use `none` in CI, so an unrecorded call fails loudly instead of reaching the network:

```bash
pytest --llm-record=none
```

A request matches a recording when its method, URL and JSON body are equal; key order doesn't matter. On a mismatch, the test fails with a diff of what changed. To exclude dynamic values, list dotted paths. `*` matches every key or index:

```ini
[pytest]
llmchaos_ignore =
    messages.*.metadata
    user
```

To exclude a path for a single test, call `llm.ignore("messages.0.content")`.

Other options:
- `llmchaos_cassette_dir` (default `cassettes`)
- `llmchaos_openai_hosts` and `llmchaos_anthropic_hosts`, for gateways and Azure endpoints

## Secrets

Cassettes are safe to commit.

- **Headers come from an allowlist:** `content-type`, `anthropic-version`, `anthropic-beta`, `openai-beta`, and response request IDs. `authorization` and `x-api-key` are never written.
- **Every string is scanned before saving,** for `sk-…` keys and for the values of `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`. If anything matches, the write is refused with `SecretLeakError`.

## Supported versions

- Python 3.11–3.14
- `openai` 3.x and `anthropic` 1.x/2.x (built on `httpx2`), tested on their latest releases
- `openai` 2.x and `anthropic` 0.x (built on `httpx`), tested on `openai==2.46.0` and `anthropic==0.125.0`

## Limitations

- Only tests that use the `llm` fixture are intercepted.
- Apps that swap in a non-httpx HTTP backend (such as the OpenAI SDK's optional aiohttp client) are not intercepted.
- `truncated` and `bad_tool_args` support streaming for Chat Completions and Anthropic Messages, but only non-streaming bodies for the OpenAI Responses API.
- Recording reads each stream fully before replaying it to the SDK, so real-time chunk timing is not preserved.
- With concurrent async calls, `on_call` counts calls in arrival order.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT
````

- [ ] **Step 5: Write CONTRIBUTING and CHANGELOG**

`CONTRIBUTING.md`:

````markdown
# Contributing

## Setup

```bash
uv sync
```

## Checks

```bash
uv run pytest                             # unit + integration; contract tests skip without keys
uv run pytest examples --llm-record=none
uv run ruff check . && uv run ruff format --check .
uv run pyright
```

To also test the older httpx-based SDKs:

```bash
uv pip install "openai==2.46.0" "anthropic==0.125.0"
uv run --no-sync pytest
uv sync   # back to the current SDKs
```

## How the pieces fit

The design is in [docs/design/2026-11-15-pytest-llmchaos-design.md](docs/design/2026-11-15-pytest-llmchaos-design.md). In short:

- `interceptor.py` patches the HTTP transports.
- `session.py` decides what each attempt receives.
- `cassette.py`, `matching.py` and `redaction.py` handle recordings.
- `providers/` knows each provider's HTTP shapes.

## Adding a provider

1. Create `src/llmchaos/providers/<name>.py` with a class implementing the `Provider` protocol from `providers/base.py`: `handles`, `error_response`, `truncate`, `break_tool_args`. Use the provider's documented error bodies, and end every message with `SIMULATED`.
2. Register it in the `llm` fixture in `plugin.py`, together with an `llmchaos_<name>_hosts` ini option.
3. Add canonical payloads to `tests/payloads.py`, and routes for them to the fake network in `tests/harness.py`.
4. Add unit tests (`tests/unit/test_provider_<name>.py`). Add SDK integration tests (`tests/integration/test_<name>_sdk.py`) that assert the SDK raises its real exception classes.
5. Add a live check to `tests/contract/test_live.py`.

## Releasing

1. Update `version` in `pyproject.toml` and `__version__` in `src/llmchaos/__init__.py`, and move the CHANGELOG entries under the new version.
2. Tag `vX.Y.Z` and push the tag. The release workflow publishes to PyPI with trusted publishing.
````

`CHANGELOG.md`:

```markdown
# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - Unreleased

### Added

- `llm` fixture that records OpenAI and Anthropic SDK calls once and replays them offline.
- Eight faults: `rate_limit`, `overloaded`, `server_error`, `timeout`, `connection_error`,
  `stream_cut`, `truncated`, `bad_tool_args`.
- `--llm-record=none|missing|all`, request-body ignore paths, and extra provider hosts.
- Header allowlist and secret scan for cassettes.
- Support for both `httpx` (older SDKs) and `httpx2` (current SDKs).
```

- [ ] **Step 6: Verify and commit**

Run: `uv run pytest && uv run pytest examples --llm-record=none && uv run ruff check . && uv run ruff format --check .`
Expected: all pass.

```bash
git add examples README.md CONTRIBUTING.md CHANGELOG.md
git commit -m "docs: add README, contributing guide, changelog and example app" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 15: Contract tests, CI, release workflow, and package check

**Files:**
- Create: `tests/contract/test_live.py`
- Create: `.github/workflows/ci.yml`, `.github/workflows/contract.yml`, `.github/workflows/release.yml`

**Interfaces:**
- Consumes: `Session`, `interceptor.install`, `cassette.load`, both providers.

- [ ] **Step 1: Write the contract tests**

`tests/contract/test_live.py`:

```python
"""Live contract checks. They run weekly in CI with real API keys and skip everywhere else.

They record real responses and confirm that our mutations still find their targets and
that our synthesized error bodies use the same fields as the providers' real ones.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import pytest

from llmchaos import cassette, interceptor
from llmchaos.cassette import Interaction
from llmchaos.providers.anthropic import AnthropicProvider
from llmchaos.providers.openai import OpenAIProvider
from llmchaos.session import Session

pytestmark = pytest.mark.contract

OPENAI_MODEL = os.environ.get("LLMCHAOS_OPENAI_MODEL", "gpt-5-mini")
ANTHROPIC_MODEL = os.environ.get("LLMCHAOS_ANTHROPIC_MODEL", "claude-haiku-5-5")
needs_openai = pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="needs OPENAI_API_KEY")
needs_anthropic = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"), reason="needs ANTHROPIC_API_KEY"
)
PROMPT = [{"role": "user", "content": "Reply with exactly five words about the sea."}]
WEATHER = [{"role": "user", "content": "What is the weather in Paris in celsius? Use the tool."}]
SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "unit": {"type": "string"}},
    "required": ["city", "unit"],
}


def record(monkeypatch, tmp_path, calls: Callable[[], object]) -> list[Interaction]:
    path = tmp_path / "live.json"
    session = Session(
        cassette_path=path, mode="all", providers=[OpenAIProvider(), AnthropicProvider()]
    )
    with monkeypatch.context() as patch:
        interceptor.install(patch, session)
        calls()
    session.save()
    return cassette.load(path)


@needs_openai
def test_openai_mutations_find_their_targets(monkeypatch, tmp_path):
    import openai

    client = openai.OpenAI()
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": SCHEMA}}]

    def calls():
        chat = client.chat.completions.create
        chat(model=OPENAI_MODEL, messages=PROMPT)
        list(chat(model=OPENAI_MODEL, messages=PROMPT, stream=True))
        chat(model=OPENAI_MODEL, messages=WEATHER, tools=tools, tool_choice="required")
        list(chat(model=OPENAI_MODEL, messages=WEATHER, tools=tools, tool_choice="required", stream=True))
        client.responses.create(model=OPENAI_MODEL, input=PROMPT[0]["content"])

    text, text_stream, tool, tool_stream, response = (
        i.response for i in record(monkeypatch, tmp_path, calls)
    )
    provider = OpenAIProvider()
    for mutate in (
        lambda: provider.truncate(text),
        lambda: provider.truncate(text_stream),
        lambda: provider.truncate(response),
        lambda: provider.break_tool_args(tool),
        lambda: provider.break_tool_args(tool_stream),
    ):
        mutate()  # raises FaultNotApplicableError if a real shape drifted


@needs_openai
def test_openai_error_body_fields_match(monkeypatch, tmp_path):
    import openai

    client = openai.OpenAI(max_retries=0)

    def calls():
        with pytest.raises(openai.APIStatusError):
            client.chat.completions.create(model="llmchaos-no-such-model", messages=PROMPT)

    [interaction] = record(monkeypatch, tmp_path, calls)
    ours = OpenAIProvider().error_response("rate_limit", 0.001).body["error"]
    assert set(ours) <= set(interaction.response.body["error"])


@needs_anthropic
def test_anthropic_mutations_find_their_targets(monkeypatch, tmp_path):
    import anthropic

    client = anthropic.Anthropic()
    tools = [{"name": "get_weather", "description": "Weather for a city", "input_schema": SCHEMA}]
    any_tool = {"type": "any"}

    def calls():
        client.messages.create(model=ANTHROPIC_MODEL, max_tokens=200, messages=PROMPT)
        with client.messages.stream(model=ANTHROPIC_MODEL, max_tokens=200, messages=PROMPT) as s:
            s.get_final_message()
        client.messages.create(
            model=ANTHROPIC_MODEL, max_tokens=200, messages=WEATHER, tools=tools, tool_choice=any_tool
        )
        with client.messages.stream(
            model=ANTHROPIC_MODEL, max_tokens=200, messages=WEATHER, tools=tools, tool_choice=any_tool
        ) as s:
            s.get_final_message()

    text, text_stream, tool, tool_stream = (i.response for i in record(monkeypatch, tmp_path, calls))
    provider = AnthropicProvider()
    provider.truncate(text)
    provider.truncate(text_stream)
    provider.break_tool_args(tool)
    provider.break_tool_args(tool_stream)


@needs_anthropic
def test_anthropic_error_body_fields_match(monkeypatch, tmp_path):
    import anthropic

    client = anthropic.Anthropic(max_retries=0)

    def calls():
        with pytest.raises(anthropic.APIStatusError):
            client.messages.create(model="llmchaos-no-such-model", max_tokens=10, messages=PROMPT)

    [interaction] = record(monkeypatch, tmp_path, calls)
    real = interaction.response.body
    ours = AnthropicProvider().error_response("overloaded", 0.001).body
    assert real["type"] == "error"
    assert set(ours["error"]) <= set(real["error"])
```

Run: `uv run pytest tests/contract -v`
Expected without keys: 4 skipped. If keys are available locally, also run `OPENAI_API_KEY=... ANTHROPIC_API_KEY=... uv run pytest tests/contract -v`, and expect 4 passed.

- [ ] **Step 2: Write the workflows**

`.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
      - uses: astral-sh/setup-uv@v6
      - run: uv sync --python 3.13
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run pyright

  test:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python: ["3.11", "3.12", "3.13", "3.14"]
        lane: [current, legacy]
    name: test (${{ matrix.python }}, ${{ matrix.lane }})
    steps:
      - uses: actions/checkout@v5
      - uses: astral-sh/setup-uv@v6
      - run: uv sync --python ${{ matrix.python }}
      - if: matrix.lane == 'legacy'
        run: uv pip install "openai==2.46.0" "anthropic==0.125.0"
      - run: uv run --no-sync pytest -m "not contract"
      - run: uv run --no-sync pytest examples --llm-record=none
```

`.github/workflows/contract.yml`:

```yaml
name: Contract

# Scheduled and manual only, so secrets are never exposed to pull requests from forks.
on:
  schedule:
    - cron: "0 6 * * 1"
  workflow_dispatch:

jobs:
  contract:
    runs-on: ubuntu-latest
    env:
      OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
    steps:
      - uses: actions/checkout@v5
      - uses: astral-sh/setup-uv@v6
      - run: uv sync --python 3.13
      - run: uv run pytest tests/contract -m contract -v
```

`.github/workflows/release.yml`:

```yaml
name: Release

on:
  push:
    tags: ["v*"]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
      - uses: astral-sh/setup-uv@v6
      - run: uv build
      - uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/

  publish:
    needs: build
    runs-on: ubuntu-latest
    environment: pypi
    permissions:
      id-token: write  # PyPI trusted publishing; no stored token
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/
      - uses: pypa/gh-action-pypi-publish@release/v1
```

- [ ] **Step 3: Build and smoke-test the package**

Run: `uv build`
Expected: `dist/pytest_llmchaos-0.1.0.tar.gz` and `dist/pytest_llmchaos-0.1.0-py3-none-any.whl`.

Run: `unzip -p dist/pytest_llmchaos-0.1.0-py3-none-any.whl 'pytest_llmchaos-0.1.0.dist-info/entry_points.txt'`
Expected:

```text
[pytest11]
llmchaos = llmchaos.plugin
```

Run: `uv run --isolated --no-project --with dist/pytest_llmchaos-0.1.0-py3-none-any.whl pytest --help | grep -- --llm-record`
Expected: one line describing `--llm-record`. This shows the wheel works in a clean environment with no SDKs installed.

- [ ] **Step 4: Full verification**

Run: `uv run pytest -v && uv run pytest examples --llm-record=none && uv run ruff check . && uv run ruff format --check . && uv run pyright`
Expected: everything passes; contract tests are skipped.

- [ ] **Step 5: Commit**

```bash
git add tests/contract .github
git commit -m "ci: add test matrix, weekly contract checks and release workflow" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 6: Hand over the outward-facing steps (do not perform them)**

These publish things or change external accounts, so they need the user's explicit go-ahead.

**Ask the user:**
- Ask which GitHub owner and repository name to use. Then add this to `pyproject.toml` and commit:

  ```toml
  [project.urls]
  Homepage = "https://github.com/<owner>/pytest-llmchaos"
  Issues = "https://github.com/<owner>/pytest-llmchaos/issues"
  ```

**Tell the user to do these steps themselves:**
- Create the GitHub repository and push `main`. Push only with their approval.
- On PyPI, add a pending trusted publisher for `pytest-llmchaos`: workflow `release.yml`, environment `pypi`.
- Optionally, add `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` repository secrets for the weekly contract job.
- Publish 0.1.0 by setting the CHANGELOG date, tagging `v0.1.0`, and pushing the tag.
