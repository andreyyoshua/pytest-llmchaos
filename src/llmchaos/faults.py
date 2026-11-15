"""Faults a test can inject with ``llm.inject(...)``.

Replace faults answer an HTTP attempt themselves; the SDK's retry is another attempt at
the same call. Mutate faults alter the recorded response of the call they target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ErrorKind = Literal["rate_limit", "overloaded", "server_error"]
TransportErrorKind = Literal["timeout", "connection_error"]

DEFAULT_RETRY_AFTER = 0.001


@dataclass(frozen=True, kw_only=True)
class Fault:
    on_call: int | None = None

    def __post_init__(self) -> None:
        if self.on_call is not None and self.on_call < 1:
            raise ValueError(f"on_call must be >= 1, got {self.on_call}")


@dataclass(frozen=True, kw_only=True)
class ReplaceFault(Fault):
    """Answers up to ``times`` HTTP attempts of its target call."""

    times: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.times < 1:
            raise ValueError(f"times must be >= 1, got {self.times}")


@dataclass(frozen=True, kw_only=True)
class HTTPError(ReplaceFault):
    """A synthesized HTTP error response in the provider's own format."""

    kind: ErrorKind
    retry_after: float = DEFAULT_RETRY_AFTER

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.retry_after < 0:
            raise ValueError(f"retry_after must be >= 0, got {self.retry_after}")


@dataclass(frozen=True, kw_only=True)
class TransportError(ReplaceFault):
    """An exception raised by the HTTP library instead of a response."""

    kind: TransportErrorKind


@dataclass(frozen=True, kw_only=True)
class MutateFault(Fault):
    """Alters the recorded response of its target call."""


@dataclass(frozen=True, kw_only=True)
class StreamCut(MutateFault):
    after_events: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.after_events < 0:
            raise ValueError(f"after_events must be >= 0, got {self.after_events}")


@dataclass(frozen=True, kw_only=True)
class Truncated(MutateFault):
    """Half the text content, ending as if the token limit was hit."""


@dataclass(frozen=True, kw_only=True)
class BadToolArgs(MutateFault):
    """Tool-call arguments the application cannot use."""


def rate_limit(
    times: int = 1, retry_after: float = DEFAULT_RETRY_AFTER, on_call: int | None = None
) -> HTTPError:
    """HTTP 429 with the provider's error body and ``retry-after`` headers."""
    return HTTPError(kind="rate_limit", times=times, retry_after=retry_after, on_call=on_call)


def overloaded(times: int = 1, on_call: int | None = None) -> HTTPError:
    """Anthropic HTTP 529 ``overloaded_error``; OpenAI HTTP 503."""
    return HTTPError(kind="overloaded", times=times, on_call=on_call)


def server_error(times: int = 1, on_call: int | None = None) -> HTTPError:
    """HTTP 500 with the provider's error body."""
    return HTTPError(kind="server_error", times=times, on_call=on_call)


def timeout(times: int = 1, on_call: int | None = None) -> TransportError:
    """The HTTP library's ``ReadTimeout``, raised immediately."""
    return TransportError(kind="timeout", times=times, on_call=on_call)


def connection_error(times: int = 1, on_call: int | None = None) -> TransportError:
    """The HTTP library's ``ConnectError``, raised immediately."""
    return TransportError(kind="connection_error", times=times, on_call=on_call)


def stream_cut(after_events: int = 1, on_call: int | None = None) -> StreamCut:
    """Stream ``after_events`` events, then fail with ``RemoteProtocolError``."""
    return StreamCut(after_events=after_events, on_call=on_call)


def truncated(on_call: int | None = None) -> Truncated:
    """Keep the first half of the text and end with the token-limit stop reason."""
    return Truncated(on_call=on_call)


def bad_tool_args(on_call: int | None = None) -> BadToolArgs:
    """OpenAI: invalid tool-argument JSON. Anthropic: a tool input missing a key."""
    return BadToolArgs(on_call=on_call)
