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
            RecordedRequest(
                "POST", URL, {"content-type": "application/json"}, body={"b": 1, "a": 2}
            ),
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
