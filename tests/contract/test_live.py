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
needs_openai = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="needs OPENAI_API_KEY"
)
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
        list(
            chat(
                model=OPENAI_MODEL,
                messages=WEATHER,
                tools=tools,
                tool_choice="required",
                stream=True,
            )
        )
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
            model=ANTHROPIC_MODEL,
            max_tokens=200,
            messages=WEATHER,
            tools=tools,
            tool_choice=any_tool,
        )
        with client.messages.stream(
            model=ANTHROPIC_MODEL,
            max_tokens=200,
            messages=WEATHER,
            tools=tools,
            tool_choice=any_tool,
        ) as s:
            s.get_final_message()

    text, text_stream, tool, tool_stream = (
        i.response for i in record(monkeypatch, tmp_path, calls)
    )
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
