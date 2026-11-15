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
            "peer closed connection without sending complete message body (simulated by llmchaos)"
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
