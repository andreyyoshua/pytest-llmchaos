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
