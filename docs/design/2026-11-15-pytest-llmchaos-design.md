# pytest-llmchaos — Design

- **Status:** Approved 2026-11-15; revised 2026-11-15 after SDK research (see §12)
- **Date:** 2026-11-15
- **Package:** `pytest-llmchaos` (PyPI) · import name `llmchaos`
- **License:** MIT

## 1. Problem

Tests for LLM-powered Python code usually call the real provider API. That makes them
slow, non-deterministic, costly, and impossible to run offline or on forks without
secrets. Several young libraries already record and replay LLM calls (reel-vcr,
promptecho, cuesheet, llmvcr, standin, cendor-cassette, plus the generic
pytest-recording/VCR.py).

What none of them focus on is the **unhappy path**: rate limits, provider overload,
timeouts, streams that die halfway, responses cut off at `max_tokens`, and tool calls
with broken arguments. These are the failures LLM features hit in production, and almost
nobody tests their handling, because reproducing them on demand is hard.

**pytest-llmchaos records real LLM responses once, replays them offline, and lets a
test inject provider-accurate failures into that replay.** The official SDKs raise their
real exception classes and run their real retry logic, so the test exercises the same
code path production does.

## 2. Goals and non-goals

### Goals (v1)

1. Record/replay for the official `openai` and `anthropic` Python SDKs, sync and async,
   streaming and non-streaming, with **no changes to application code**.
2. Eight deterministic, provider-accurate faults (§5).
3. Correct interaction with SDK retries: tests can assert both "recovers" and "gives up".
4. Cassettes that are safe to commit (no secrets) and readable in a PR diff.
5. Clear failures: when something does not match, the test failure says exactly what
   changed.

### Non-goals (v1)

- Gemini, Mistral, Bedrock, or other providers (the adapter interface makes these
  additive later).
- Random/probabilistic chaos. Every fault is scheduled explicitly.
- Intercepting non-httpx HTTP backends (e.g. the OpenAI SDK's optional aiohttp client).
- Intercepting calls in tests that do not request the `llm` fixture.
- A web UI, a proxy server, or a CLI.
- Prompt/quality evaluation.

## 3. User-facing API

### 3.1 The `llm` fixture

Requesting the `llm` fixture turns on interception for that test.

```python
import pytest
from llmchaos import faults

def test_summarize(llm):
    assert "refund" in summarize(ticket)          # replayed from cassette

def test_survives_rate_limits(llm):
    llm.inject(faults.rate_limit(times=2))
    assert summarize(ticket)                      # SDK retried twice, then succeeded
    assert llm.calls == 1
    assert llm.attempts == 3

def test_gives_up_cleanly(llm):
    llm.inject(faults.rate_limit(times=5))        # more than the SDK's 2 default retries
    with pytest.raises(ServiceBusyError):         # the app's own error type
        summarize(ticket)

def test_third_agent_step_overloaded(llm):
    llm.inject(faults.overloaded(on_call=3))
    run_agent(task)
```

Fixture members:

| Member | Meaning |
|---|---|
| `inject(*faults)` | Schedule one or more faults (§5.3). |
| `ignore(*paths)` | Dotted request-body paths excluded from matching for this test, e.g. `"messages.0.content"`. Must be called before the call it affects. |
| `calls` | Number of logical calls served so far (§5.3). |
| `attempts` | Number of HTTP attempts that reached the interceptor. |
| `cassette_path` | `pathlib.Path` of this test's cassette. |

### 3.2 Command-line and ini options

| Option | Values | Default | Meaning |
|---|---|---|---|
| `--llm-record` | `none`, `missing`, `all` | `missing` | See §6.3. CI should use `none`. |
| ini `llmchaos_cassette_dir` | path | `cassettes` | Directory, relative to each test file, that holds cassettes. |
| ini `llmchaos_ignore` | list of dotted paths | empty | Body paths ignored for every test. |
| ini `llmchaos_openai_hosts` | list of hosts | empty | Extra hosts treated as OpenAI-compatible (Azure, gateways). |
| ini `llmchaos_anthropic_hosts` | list of hosts | empty | Extra hosts treated as Anthropic. |

Built-in hosts: `api.openai.com`, `api.anthropic.com`.

### 3.3 Public exports

`llmchaos.faults` (fault constructors) and `llmchaos.errors` (exception classes, §8).
Everything else is internal.

## 4. Architecture

Both SDKs moved from `httpx` to `httpx2` (Pydantic's API-compatible fork) in mid-2026:
`openai` 3.0.0 and `anthropic` 1.0.0 depend only on `httpx2`, while `openai` ≤ 2.46 and
`anthropic` ≤ 0.125 use `httpx`. The interceptor patches **both** libraries whenever they
are importable, so old and new SDKs are covered. The core (session, cassette, providers)
never imports either library. It works on plain data, and the interceptor converts to
and from each library's `Request`/`Response` and raises that library's exceptions.

```
app code
  └─ openai / anthropic SDK
       └─ httpx2.Client / AsyncClient   (or httpx.* on older SDKs)
            └─ HTTPTransport.handle_request            ← patched for the test's duration
                 ├─ host is not a provider → original transport (untouched)
                 └─ host is a provider     → Session.handle(request)
                        1. pending replace-fault for this call?  → synthesized error
                        2. replay mode                           → matching cassette entry
                           record mode                           → real transport, then record
                        3. pending mutate-fault for this call?   → altered response
                        4. → httpx.Response back to the SDK
```

### 4.1 Units

| Module | Responsibility | Depends on |
|---|---|---|
| `plugin.py` | pytest entry point (`pytest11`): options, `llm` fixture, cassette path resolution, call-phase hook that surfaces stored errors (§8). | session, interceptor |
| `interceptor.py` | Patches `HTTPTransport.handle_request` and `AsyncHTTPTransport.handle_async_request` of `httpx` and `httpx2` (whichever are importable); converts requests/responses; raises library-specific transport errors. | session |
| `session.py` | Per-test state: fault schedule, cassette cursor, counters, stored error, lock. Provider-agnostic. | cassette, matching, redaction, faults, providers |
| `cassette.py` | Data model (`RecordedRequest`, `RecordedResponse`, `Interaction`), conversion from raw bytes, load/save (atomic). | sse |
| `matching.py` | Ignore paths, request equality, diff rendering. | cassette |
| `redaction.py` | Header allowlists and the secret scan. | — |
| `sse.py` | Parse a server-sent-events byte stream into events and serialize events back to bytes. | — |
| `faults.py` | Public frozen dataclasses for the eight faults, with argument validation. | — |
| `errors.py` | Public exception hierarchy. | — |
| `providers/base.py` | `Provider` protocol. | — |
| `providers/openai.py`, `providers/anthropic.py` | Host recognition, synthesized error responses, `truncate`, `break_tool_args`. Pure HTTP shapes; they never import the SDKs. | base, sse |

The only runtime dependency is `pytest>=8`. `httpx`/`httpx2` arrive with whichever SDK
the user has installed. The SDKs and both HTTP libraries are test-only dependencies of
this repo.

### 4.2 Provider protocol

```python
class Provider(Protocol):
    name: str
    def handles(self, url: httpx.URL) -> bool: ...
    def error_response(self, kind: ErrorKind, retry_after: float | None) -> RecordedResponse: ...
    def truncate(self, response: RecordedResponse) -> RecordedResponse: ...
    def break_tool_args(self, response: RecordedResponse) -> RecordedResponse: ...
```

`ErrorKind` is `rate_limit | overloaded | server_error`. `timeout`, `connection_error`
and `stream_cut` are transport-level and provider-agnostic, so they live in the session.
Adding a provider means adding one module and registering it; nothing else changes.

## 5. Faults

### 5.1 Catalog

| Constructor | Kind | Effect |
|---|---|---|
| `rate_limit(times=1, retry_after=0.001, on_call=None)` | replace | HTTP 429 with the provider's error body. Sets `retry-after-ms` (exact) and `retry-after` (whole seconds, rounded up). |
| `overloaded(times=1, on_call=None)` | replace | Anthropic: HTTP 529 `overloaded_error` (`anthropic.OverloadedError` on SDK ≥ 1.0, `InternalServerError` before). OpenAI: HTTP 503 `server_error`. |
| `server_error(times=1, on_call=None)` | replace | HTTP 500. Anthropic `api_error`, OpenAI `server_error`. |
| `timeout(times=1, on_call=None)` | replace | Raises the intercepted library's `ReadTimeout` immediately. |
| `connection_error(times=1, on_call=None)` | replace | Raises the intercepted library's `ConnectError` immediately. |
| `stream_cut(after_events=1, on_call=None)` | mutate | Streams the first `after_events` SSE events, then raises the library's `RemoteProtocolError`. Streaming responses only. What the app sees depends on the SDK (e.g. `openai` 3.x wraps it in `APIConnectionError`); the exception chain always contains `RemoteProtocolError`. |
| `truncated(on_call=None)` | mutate | Keeps the first half of the text content and sets OpenAI `finish_reason="length"` (Chat Completions) / `status="incomplete"` with `max_output_tokens` (Responses API), or Anthropic `stop_reason="max_tokens"`. Streaming is supported for Chat Completions and Anthropic Messages; the OpenAI Responses API is non-streaming only in v1. |
| `bad_tool_args(on_call=None)` | mutate | OpenAI (arguments are a JSON string): removes the final character, so the JSON no longer parses. Anthropic (input is an object): removes the first key, so the input no longer matches the tool schema. In Anthropic streams, the block's `input_json_delta` events are replaced by one delta with the altered JSON. A truncated-JSON fault would be pointless here, because the Anthropic stream helper parses tool JSON in partial mode and silently accepts it. The OpenAI Responses API is non-streaming only in v1. |

Every synthesized error message ends with `(simulated by llmchaos)` so it is never
mistaken for a real outage in logs. All synthesized HTTP errors (not only rate limits)
carry `retry-after-ms`, so SDK retries are instant. `timeout` and `connection_error`
have no response to carry that header, so SDK retries after them use the SDK's own
backoff (about 0.5–1 s per retry). This is documented in the README.

Exact status codes, error bodies and headers are verified against each provider's
documentation during implementation. Integration tests (§9.2) assert that each SDK
raises the expected exception class for each fault.

### 5.2 Applicability

- Replace faults work on any endpoint of a recognized provider.
- `stream_cut` applies only to SSE responses with more than `after_events` events.
  `after_events=0` cuts before the first event.
- `truncated` and `bad_tool_args` apply to OpenAI Chat Completions, OpenAI Responses,
  and Anthropic Messages. `truncated` needs a response with text content;
  `bad_tool_args` needs a response that contains a tool call.
- A mutate fault that cannot be applied to the response it targets fails the test with
  `FaultNotApplicableError`.

### 5.3 Scheduling semantics

- A **logical call** is served when an attempt receives a cassette-backed response
  (replayed or freshly recorded). Calls are numbered from 1 in arrival order.
  `llm.calls` counts served calls; `llm.attempts` counts every attempt.
- Each fault targets exactly one logical call. `on_call=n` targets call `n`. `on_call=None`
  targets the next call that has not yet been served at the time `inject` runs.
- When an attempt for call `k` arrives:
  1. If a replace fault targeting `k` has remaining `times`, the earliest-injected one
     answers the attempt and its `times` decrements. The call is **not** served and the
     cassette cursor does not move. The SDK's retry arrives as another attempt for the
     same call `k`.
  2. Otherwise the call is served from the cassette (or recorded), and a mutate fault
     targeting `k`, if any, alters the response before it is returned.
- At most one mutate fault may target a call. Injecting a second raises `ValueError`
  from `inject`. So does `on_call=n` for a call that has already been served, since that
  fault could never fire.
- With concurrent async calls, call numbers follow arrival order, so `on_call` is only
  deterministic when the calls it counts are made sequentially.
- `times` larger than the SDK's retry budget is allowed and is how a test exercises
  "gives up". Unspent `times` are not an error.
- **Strict by default:** if the test body finishes and any injected fault never fired
  even once, the test fails with `UnusedFaultError`.

### 5.4 Faults during recording

In record mode, replace faults answer their attempts without calling the API. Mutate
faults alter the response returned to the SDK, but the cassette stores the real,
unaltered response.

## 6. Cassettes

### 6.1 Location

`<test file dir>/<llmchaos_cassette_dir>/<test module name>/<test name>.json`.
For tests inside a class, the test name is `<Class>.<test>`. Characters other than
letters, digits, `-`, `_`, `.`, `[` and `]` become `_`.

### 6.2 Format

Pretty-printed JSON, two-space indent, sorted keys, trailing newline.

```json
{
  "interactions": [
    {
      "request": {
        "body": {"model": "claude-sonnet-5-5", "max_tokens": 256, "messages": [ ... ], "stream": true},
        "headers": {"anthropic-version": "2023-06-01", "content-type": "application/json"},
        "method": "POST",
        "url": "https://api.anthropic.com/v1/messages"
      },
      "response": {
        "events": [
          {"data": "{\"type\":\"message_start\", ...}", "event": "message_start"}
        ],
        "headers": {"content-type": "text/event-stream"},
        "status": 200
      }
    }
  ],
  "version": 1
}
```

Body encodings, chosen by content type:

| Content | Request field | Response field |
|---|---|---|
| JSON | `body` (parsed) | `body` (parsed) |
| `text/event-stream` | — | `events`: list of `{data, event?, id?}` (OpenAI streams have no `event` field) |
| anything else | `body_b64` | `body_b64` |

### 6.3 Record modes

| Mode | Cassette exists | Cassette missing |
|---|---|---|
| `none` | Replay. Extra calls → `UnexpectedCallError`. | First call → `CassetteMissingError`. |
| `missing` (default) | Replay. Extra calls → `UnexpectedCallError`. | Record the whole test. |
| `all` | Ignore it, record the whole test, overwrite. | Record the whole test. |

A recorded cassette is written at test teardown, atomically (temporary file, then
rename), even if the test failed. Nothing is written if the test made no provider calls.
During recording, a streaming response is read fully from the real API before being
replayed to the SDK, so recording does not preserve real-time chunk timing.

### 6.4 Matching

A request matches a recorded interaction when method, full URL, and body are equal.
Bodies are compared after parsing JSON (key order is irrelevant) and replacing the
values at ignored paths (`llmchaos_ignore` plus `llm.ignore`) with `"<ignored>"`. A path
segment `*` matches every key or index at that level, e.g. `messages.*.content`.
Non-JSON bodies are compared byte-for-byte.

The request is served by the **first unused interaction that matches**. This keeps
repeated identical requests in order, and tolerates concurrent async calls that arrive
in a different order than they were recorded.

If nothing matches, the session stores `CassetteMismatchError` with a unified diff
between the request and the next unused interaction in recorded order, plus the hint
`re-record with --llm-record=all`.

### 6.5 Redaction and secret safety

- Request and response headers are written from an **allowlist** only. Request:
  `content-type`, `anthropic-version`, `anthropic-beta`, `openai-beta`. Response:
  `content-type`, `request-id`, `x-request-id`. All other headers, including
  `authorization`, `x-api-key` and organization IDs, are never written.
- Before writing, every string in the cassette is scanned for
  `(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}` (the lookbehind avoids false hits inside words
  like "risk-assessment-…") and for the literal values of the `OPENAI_API_KEY` and
  `ANTHROPIC_API_KEY` environment variables when they are set and at least 8 characters
  long. A hit aborts the write and fails the test with
  `SecretLeakError`, naming the JSON path where the secret was found.
- Header matching is not part of request matching, so dropped headers never cause
  mismatches.

## 7. Interception details

- Patching is done with pytest's `monkeypatch` inside the `llm` fixture, so it is undone
  after every test, even on failure.
- The patched method reads the request body (`request.read()` / `await request.aread()`)
  before routing.
- Responses are built as `httpx.Response(status, headers=..., stream=...)`. `stream_cut`
  uses a custom `SyncByteStream` / `AsyncByteStream` that yields the allowed events and
  then raises `httpx.RemoteProtocolError`.
- Sync calls from multiple threads are safe: the session serializes state changes with
  a `threading.Lock`.

## 8. Error handling

Exception hierarchy (`llmchaos.errors`):

```
LLMChaosError
├── CassetteMissingError
├── CassetteMismatchError
├── UnexpectedCallError
├── UnusedFaultError
├── FaultNotApplicableError
└── SecretLeakError
```

**Why errors are stored rather than only raised:** SDKs treat exceptions from the
transport inconsistently. Verified 2026-11-15: `openai` 3.51 lets a non-httpx exception
propagate unchanged, while `anthropic` 2.8 retries it three times and then raises
`APIConnectionError`, which hides the real cause. `LLMChaosError` deliberately does
not subclass any httpx exception, so SDKs that only retry network errors pass it
through untouched. So when the session detects one of
the errors above (except `SecretLeakError`, raised at teardown), it:

1. Stores the first such error on the session.
2. Raises it from the transport anyway, so the app's call fails immediately rather than
   receiving made-up data.

A `pytest_runtest_call` hook wrapper then checks the session after the test body. If an
error is stored, the test fails with **that** error, chained to whatever the test
raised. If no error is stored and the test passed, the same hook enforces strict mode
(`UnusedFaultError`). Users therefore see `CassetteMismatchError: request body changed:
...` instead of a misleading connection error.

Errors from the real network during recording propagate unchanged.

## 9. Testing the tool

### 9.1 Unit tests (`tests/unit/`)

- `sse`: parse → serialize round trip, including multi-line `data`, comments, `id`.
- `cassette`: load/save round trip, sorted-key output, atomic write, matching with and
  without ignore paths, first-unused-match ordering, diff rendering.
- Redaction: allowlist behavior; secret scan by pattern and by env-var value.
- Each provider: every `error_response` kind, `truncate` and `break_tool_args` on
  streaming and non-streaming fixtures for each supported endpoint.
- `session`: the full scheduling rules of §5.3, including `on_call`, multiple faults on
  one call, two mutate faults on one call, strict mode, and record-mode behavior of §5.4.

### 9.2 Integration tests (`tests/integration/`)

Use `pytester` to run inner test files that use the **real** `openai` and `anthropic`
SDKs against hand-written cassettes committed to this repo. No network and no keys are
needed. The matrix covers:

- provider × sync/async × streaming/non-streaming,
- plain replay,
- each fault, asserting the exact SDK exception class and the retry count
  (`llm.attempts`),
- record modes (each scenario first records against a fake network, patched into both
  libraries' transports, then replays with the network forbidden),
- the user-facing failure message for a mismatch and for an unused fault.

### 9.3 Contract tests (`tests/contract/`)

Marked `@pytest.mark.contract`, skipped unless API keys are present. A scheduled weekly
GitHub Actions workflow runs them with repository secrets. They record fresh real
responses and check that our synthesized error shapes and mutation targets still match
what the providers and current SDKs produce. They never run on pull requests from forks.

### 9.4 CI matrix

Python 3.11, 3.12, 3.13, 3.14 × two SDK lanes:

- `current`: latest `openai` and `anthropic` (on `httpx2`).
- `legacy`: `openai==2.46.0` and `anthropic==0.125.0`, the last releases on `httpx`.

The README lists both lanes as the tested versions.

## 10. Packaging and repository

```
pytest-llmchaos/
├── pyproject.toml          # hatchling; pytest11 entry point: llmchaos = "llmchaos.plugin"
├── README.md
├── LICENSE                 # MIT
├── CHANGELOG.md
├── CONTRIBUTING.md         # includes "how to add a provider"
├── src/llmchaos/
│   ├── __init__.py
│   ├── plugin.py
│   ├── interceptor.py
│   ├── session.py
│   ├── cassette.py
│   ├── sse.py
│   ├── faults.py
│   ├── errors.py
│   └── providers/{__init__,base,openai,anthropic}.py
├── tests/{unit,integration,contract}/
├── examples/               # a small app using both SDKs, with its tests and cassettes
├── docs/design/
└── .github/workflows/{ci,contract,release}.yml
```

- Development tooling: `uv`, `ruff` (lint and format), `pyright` (standard mode on `src/`;
  strict mode fights the untyped JSON this tool manipulates everywhere).
- Release: tag `v*` → GitHub Actions builds and publishes to PyPI with trusted
  publishing (no stored token). First release `0.1.0`.
- README structure: one-sentence pitch → the problem → 10-line example → fault table →
  record modes and CI recipe (`--llm-record=none`) → limitations → contributing.

## 11. Future work (not v1)

- Streaming support for `truncated` and `bad_tool_args` on the OpenAI Responses API.

- Drift report: list stale cassettes with request diffs and re-record only those, with
  a token-cost estimate before spending.
- Gemini and other provider adapters.
- A marker form (`@pytest.mark.llm_faults(...)`) as sugar over `llm.inject`.
- Optional enforcement that blocks real provider calls in tests that do not use `llm`.

## 12. Revision log

**2026-11-15, after inspecting `openai` 3.51.0, `anthropic` 2.8.0 and `httpx2` 2.16.0, plus a
throwaway spike against both SDKs:**

- Interception covers both `httpx` and `httpx2` (§4); the only runtime dependency is pytest.
- `cassette.py` is split into `cassette.py`, `matching.py` and `redaction.py`.
- `bad_tool_args` on Anthropic drops a key in both streaming and non-streaming responses.
- `truncated` and `bad_tool_args` on the OpenAI Responses API are non-streaming only in v1.
- All synthesized HTTP errors carry `retry-after-ms`; transport-level faults use SDK backoff.
- Ignore paths support `*` and replace values with `"<ignored>"`.
- Class-based tests get `<Class>.<test>` cassette names.
- The CI matrix uses explicit `current` and `legacy` SDK lanes.
- pyright runs in standard mode rather than strict.
- `bad_tool_args` targets the last tool call in a response.

Confirmed by the spike: patching `httpx2.HTTPTransport.handle_request` intercepts both
SDKs; `retry-after-ms: 1` makes SDK retries instant; HTTP 529 raises
`anthropic.OverloadedError`; `openai` 3.x surfaces a cut stream as `APIConnectionError`.
