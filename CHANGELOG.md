# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - Unreleased

### Added

- `llm` fixture that records OpenAI and Anthropic SDK calls once and replays them offline.
- Eight faults: `rate_limit`, `overloaded`, `server_error`, `timeout`, `connection_error`,
  `stream_cut`, `truncated`, `bad_tool_args`.
- `--llm-record=none|missing|all`, request-body ignore paths, and extra provider hosts.
- Header allowlist and secret scan for cassettes.
- Support for both `httpx` (older SDKs) and `httpx2` (current SDKs).
