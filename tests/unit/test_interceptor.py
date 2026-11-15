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
