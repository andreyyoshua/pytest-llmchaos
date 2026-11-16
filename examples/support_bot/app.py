"""A tiny support bot: summarizes a ticket with OpenAI and falls back to Anthropic."""

from __future__ import annotations

import anthropic
import openai

OPENAI_MODEL = "gpt-5"
ANTHROPIC_MODEL = "claude-sonnet-5-5"


class ServiceBusyError(Exception):
    """Both providers are unavailable right now."""


def summarize(
    ticket: str, *, openai_client: openai.OpenAI, anthropic_client: anthropic.Anthropic
) -> str:
    prompt = f"Summarize this support ticket in one sentence:\n\n{ticket}"
    try:
        completion = openai_client.chat.completions.create(
            model=OPENAI_MODEL, messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content or ""
    except (openai.RateLimitError, openai.InternalServerError, openai.APIConnectionError):
        pass
    try:
        message = anthropic_client.messages.create(
            model=ANTHROPIC_MODEL, max_tokens=256, messages=[{"role": "user", "content": prompt}]
        )
    except (
        anthropic.RateLimitError,
        anthropic.OverloadedError,  # not an InternalServerError before anthropic 1.0
        anthropic.InternalServerError,
        anthropic.APIConnectionError,
    ) as error:
        raise ServiceBusyError("OpenAI and Anthropic are both unavailable") from error
    return "".join(block.text for block in message.content if block.type == "text")
