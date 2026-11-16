# Contributing

## Setup

```bash
uv sync
```

## Checks

```bash
uv run pytest                             # unit + integration; contract tests skip without keys
uv run pytest examples --llm-record=none
uv run ruff check . && uv run ruff format --check .
uv run pyright
```

To also test the older httpx-based SDKs:

```bash
uv pip install "openai==2.46.0" "anthropic==0.125.0"
uv run --no-sync pytest
uv sync   # back to the current SDKs
```

## How the pieces fit

The design is in [docs/design/2026-11-15-pytest-llmchaos-design.md](docs/design/2026-11-15-pytest-llmchaos-design.md). In short:

- `interceptor.py` patches the HTTP transports.
- `session.py` decides what each attempt receives.
- `cassette.py`, `matching.py` and `redaction.py` handle recordings.
- `providers/` knows each provider's HTTP shapes.

## Adding a provider

1. Create `src/llmchaos/providers/<name>.py` with a class implementing the `Provider` protocol from `providers/base.py`: `handles`, `error_response`, `truncate`, `break_tool_args`. Use the provider's documented error bodies, and end every message with `SIMULATED`.
2. Register it in the `llm` fixture in `plugin.py`, together with an `llmchaos_<name>_hosts` ini option.
3. Add canonical payloads to `tests/payloads.py`, and routes for them to the fake network in `tests/harness.py`.
4. Add unit tests (`tests/unit/test_provider_<name>.py`). Add SDK integration tests (`tests/integration/test_<name>_sdk.py`) that assert the SDK raises its real exception classes.
5. Add a live check to `tests/contract/test_live.py`.

## Releasing

1. Update `version` in `pyproject.toml` and `__version__` in `src/llmchaos/__init__.py`, and move the CHANGELOG entries under the new version.
2. Tag `vX.Y.Z` and push the tag. The release workflow publishes to PyPI with trusted publishing.
