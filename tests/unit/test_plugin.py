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
    source = (
        CLIENT
        + """
def test_chat(llm):
    assert post().json()["choices"][0]["message"]["content"] == "The refund was approved yesterday."
"""
    )
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
