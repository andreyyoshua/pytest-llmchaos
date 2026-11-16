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
