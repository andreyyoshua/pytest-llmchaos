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
