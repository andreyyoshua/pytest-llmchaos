# pytest-llmchaos

**Give your LLM calls a bad day before production does.**

pytest-llmchaos is a pytest plugin for code that calls the OpenAI or Anthropic Python SDKs. It records real responses once, replays them offline in every later run, and lets each test inject the failures that break LLM features in production:

- rate limits and overloaded servers
- timeouts and dropped connections
- streams that die halfway
- answers cut off at the token limit
- tool calls with broken arguments

The faults look exactly like the real thing to the SDKs, so they raise their real exception classes and run their real retry logic. Your error handling is tested the way it runs in production, with no network, no API key and no cost.

## Install

```bash
pip install pytest-llmchaos
```

The plugin activates automatically. Use the `llm` fixture in any test that calls a provider.

## Example

```python
import pytest
from llmchaos import faults


def test_summarize(llm):
    assert "refund" in summarize(ticket)  # replayed from cassettes/


def test_survives_rate_limits(llm):
    llm.inject(faults.rate_limit(times=2))  # two real-looking 429s, then the recorded answer
    assert summarize(ticket)
    assert llm.attempts == 3  # proves the SDK retried


def test_falls_back_when_openai_keeps_failing(llm):
    llm.inject(faults.overloaded(times=3))  # more than the SDK's 2 default retries
    assert summarize(ticket) == FALLBACK_SUMMARY


def test_handles_a_cut_stream(llm):
    llm.inject(faults.stream_cut(after_events=4))
    with pytest.raises(PartialResponseError):  # your app's own error
        stream_summary(ticket)
```

The first run with an API key records real responses to `cassettes/<module>/<test>.json` next to the test file. Every later run replays them. [`examples/support_bot`](examples/support_bot) is a complete, runnable example.

## Faults

| Fault | What the SDK sees |
|---|---|
| `rate_limit(times=1, retry_after=0.001)` | HTTP 429 with the provider's error body and `retry-after` headers |
| `overloaded(times=1)` | Anthropic HTTP 529 `overloaded_error`; OpenAI HTTP 503 |
| `server_error(times=1)` | HTTP 500 |
| `timeout(times=1)` | `ReadTimeout` from the HTTP library |
| `connection_error(times=1)` | `ConnectError` from the HTTP library |
| `stream_cut(after_events=1)` | The recorded stream stops after N events with `RemoteProtocolError` |
| `truncated()` | The first half of the text, ending with `finish_reason="length"` / `stop_reason="max_tokens"` |
| `bad_tool_args()` | OpenAI: tool arguments that are not valid JSON. Anthropic: a tool input with a key missing |

Every fault also takes `on_call=n` to target the nth call in the test. By default it targets the next call.

- **Replace faults** (the first five) answer an HTTP attempt themselves and don't use up the recording. `rate_limit(times=2)` means two 429s, then the real answer on the SDK's third attempt. Use a `times` larger than the SDK's retry budget to test the "gives up" path.
- **Mutate faults** (the last three) alter the recorded response of the call they target.
- **Strict by default.** A test fails if a fault it injected never fired, because that means the code never made the call you thought it did.
- **Fast by default.** Synthesized HTTP errors carry `retry-after-ms: 1`, so SDK retries are instant. `timeout` and `connection_error` have no response to carry that header, so retries after them use the SDK's own backoff (about 0.5–1 s each).
- Synthesized error messages end with `(simulated by llmchaos)`, so they can't be mistaken for real outages in logs.

## Recording

| `--llm-record=` | Cassette exists | Cassette missing |
|---|---|---|
| `none` | replay | fail with `CassetteMissingError` |
| `missing` (default) | replay | record |
| `all` | re-record | record |

Use `none` in CI, so an unrecorded call fails loudly instead of reaching the network:

```bash
pytest --llm-record=none
```

A request matches a recording when its method, URL and JSON body are equal; key order doesn't matter. On a mismatch, the test fails with a diff of what changed. To exclude dynamic values, list dotted paths. `*` matches every key or index:

```ini
[pytest]
llmchaos_ignore =
    messages.*.metadata
    user
```

To exclude a path for a single test, call `llm.ignore("messages.0.content")`.

Other options:
- `llmchaos_cassette_dir` (default `cassettes`)
- `llmchaos_openai_hosts` and `llmchaos_anthropic_hosts`, for gateways and Azure endpoints

## Secrets

Cassettes are safe to commit.

- **Headers come from an allowlist:** `content-type`, `anthropic-version`, `anthropic-beta`, `openai-beta`, and response request IDs. `authorization` and `x-api-key` are never written.
- **Every string is scanned before saving,** for `sk-…` keys and for the values of `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`. If anything matches, the write is refused with `SecretLeakError`.

## Supported versions

- Python 3.11–3.14
- `openai` 3.x and `anthropic` 1.x/2.x (built on `httpx2`), tested on their latest releases
- `openai` 2.x and `anthropic` 0.x (built on `httpx`), tested on `openai==2.46.0` and `anthropic==0.125.0`

## Limitations

- Only tests that use the `llm` fixture are intercepted.
- Apps that swap in a non-httpx HTTP backend (such as the OpenAI SDK's optional aiohttp client) are not intercepted.
- `truncated` and `bad_tool_args` support streaming for Chat Completions and Anthropic Messages, but only non-streaming bodies for the OpenAI Responses API.
- Recording reads each stream fully before replaying it to the SDK, so real-time chunk timing is not preserved.
- With concurrent async calls, `on_call` counts calls in arrival order.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT
