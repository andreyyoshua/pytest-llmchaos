"""Tests for the support bot.

The committed cassettes replay offline. To re-record them against the real APIs:

    OPENAI_API_KEY=... ANTHROPIC_API_KEY=... pytest examples --llm-record=all
"""

import os

import anthropic
import openai
import pytest
from app import ServiceBusyError, summarize

from llmchaos import faults

TICKET = "Customer 1042 asked for a refund on order 88; it was approved yesterday."


@pytest.fixture
def clients():
    return {
        "openai_client": openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "replay-only")),
        "anthropic_client": anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY", "replay-only")
        ),
    }


def test_summarizes_with_openai(llm, clients):
    assert summarize(TICKET, **clients)
    assert llm.calls == 1


def test_falls_back_to_anthropic_when_openai_is_rate_limited(llm, clients):
    llm.inject(faults.rate_limit(times=3))  # more than the OpenAI SDK's 2 retries
    assert summarize(TICKET, **clients)
    assert llm.attempts == 4  # 3 rate-limited OpenAI attempts, then 1 Anthropic call


def test_reports_busy_when_both_providers_fail(llm, clients):
    llm.inject(faults.rate_limit(times=3), faults.overloaded(times=3))
    with pytest.raises(ServiceBusyError):
        summarize(TICKET, **clients)
