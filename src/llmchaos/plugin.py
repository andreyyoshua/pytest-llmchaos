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
        "llmchaos_cassette_dir",
        "Cassette directory, relative to each test file.",
        default="cassettes",
    )
    parser.addini(
        "llmchaos_ignore",
        "Dotted request-body paths ignored when matching.",
        type="linelist",
        default=[],
    )
    parser.addini(
        "llmchaos_openai_hosts",
        "Extra hosts treated as OpenAI-compatible.",
        type="linelist",
        default=[],
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
