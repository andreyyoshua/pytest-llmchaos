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
