"""Per-test state: the fault schedule, the cassette cursor, counters, and the stored error."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, NoReturn

from llmchaos import cassette, matching, redaction
from llmchaos.cassette import Interaction, RecordedRequest, RecordedResponse
from llmchaos.errors import (
    CassetteMismatchError,
    CassetteMissingError,
    FaultNotApplicableError,
    LLMChaosError,
    SecretLeakError,
    UnexpectedCallError,
)
from llmchaos.faults import (
    BadToolArgs,
    Fault,
    HTTPError,
    MutateFault,
    ReplaceFault,
    StreamCut,
    TransportError,
    TransportErrorKind,
    Truncated,
)
from llmchaos.providers.base import Provider

RecordMode = Literal["none", "missing", "all"]


@dataclass(frozen=True)
class Outcome:
    """What the interceptor hands back to the SDK for one HTTP attempt."""

    response: RecordedResponse | None = None
    raise_error: TransportErrorKind | None = None
    cut_after: int | None = None


@dataclass(frozen=True)
class Record:
    """Tells the interceptor to call the real API, then pass the result to ``Session.recorded``."""

    call: int
    provider: Provider


@dataclass
class _Scheduled:
    fault: Fault
    target: int
    remaining: int
    fired: bool = False


class Session:
    """Decides what each provider HTTP attempt in one test receives."""

    def __init__(
        self,
        *,
        cassette_path: Path,
        mode: RecordMode,
        providers: Sequence[Provider],
        ignore: Sequence[str] = (),
    ) -> None:
        self.cassette_path = cassette_path
        self.calls = 0
        self.attempts = 0
        self.error: LLMChaosError | None = None
        self._providers = list(providers)
        self._ignore = list(ignore)
        self._scheduled: list[_Scheduled] = []
        self._lock = threading.Lock()
        exists = cassette_path.exists()
        self._recording = mode == "all" or (mode == "missing" and not exists)
        self._new: list[Interaction] = []
        self._replay: list[Interaction] | None = None
        if not self._recording and exists:
            self._replay = cassette.load(cassette_path)
        self._used = [False] * len(self._replay or [])

    # --- used by the fixture -------------------------------------------------------------

    def inject(self, *faults: Fault) -> None:
        with self._lock:
            for fault in faults:
                target = fault.on_call if fault.on_call is not None else self.calls + 1
                if target <= self.calls:
                    raise ValueError(
                        f"{fault!r} targets call {target}, which has already been served"
                    )
                if isinstance(fault, MutateFault) and any(
                    isinstance(s.fault, MutateFault) and s.target == target for s in self._scheduled
                ):
                    raise ValueError(
                        f"call {target} already has a mutate fault; only one is allowed per call"
                    )
                remaining = fault.times if isinstance(fault, ReplaceFault) else 1
                self._scheduled.append(_Scheduled(fault=fault, target=target, remaining=remaining))

    def ignore(self, *paths: str) -> None:
        with self._lock:
            self._ignore.extend(paths)

    def unused_faults(self) -> list[Fault]:
        return [s.fault for s in self._scheduled if not s.fired]

    # --- used by the interceptor ---------------------------------------------------------

    def provider_for(self, host: str) -> Provider | None:
        return next((p for p in self._providers if p.handles(host)), None)

    def handle(self, request: RecordedRequest, provider: Provider) -> Outcome | Record:
        with self._lock:
            self.attempts += 1
            call = self.calls + 1
            fault = self._take_replace_fault(call)
            if fault is not None:
                return self._replace(fault, provider)
            if self._recording:
                self.calls = call
                return Record(call=call, provider=provider)
            interaction = self._match(request)
            self.calls = call
            return self._mutate(call, provider, interaction.response)

    def recorded(
        self, step: Record, request: RecordedRequest, response: RecordedResponse
    ) -> Outcome:
        interaction = Interaction(
            request=replace(
                request,
                headers=redaction.filter_headers(
                    request.headers, redaction.REQUEST_HEADER_ALLOWLIST
                ),
            ),
            response=replace(
                response,
                headers=redaction.filter_headers(
                    response.headers, redaction.RESPONSE_HEADER_ALLOWLIST
                ),
            ),
        )
        with self._lock:
            self._new.append(interaction)
            return self._mutate(step.call, step.provider, interaction.response)

    # --- used by the plugin --------------------------------------------------------------

    def save(self) -> None:
        if not self._recording or not self._new:
            return
        leaks = redaction.find_secrets(cassette.to_document(self._new))
        if leaks:
            raise SecretLeakError(
                f"Refusing to write {self.cassette_path}: possible API key at {', '.join(leaks)}"
            )
        cassette.save(self.cassette_path, self._new)

    # --- internals -----------------------------------------------------------------------

    def _take_replace_fault(self, call: int) -> ReplaceFault | None:
        for scheduled in self._scheduled:
            fault = scheduled.fault
            if scheduled.target == call and isinstance(fault, ReplaceFault) and scheduled.remaining:
                scheduled.remaining -= 1
                scheduled.fired = True
                return fault
        return None

    def _replace(self, fault: ReplaceFault, provider: Provider) -> Outcome:
        if isinstance(fault, TransportError):
            return Outcome(raise_error=fault.kind)
        if isinstance(fault, HTTPError):
            return Outcome(response=provider.error_response(fault.kind, fault.retry_after))
        raise TypeError(f"unknown replace fault {fault!r}")

    def _match(self, request: RecordedRequest) -> Interaction:
        if self._replay is None:
            self._fail(
                CassetteMissingError(
                    f"No cassette at {self.cassette_path}. Record it by running this test once "
                    "with an API key and --llm-record=missing."
                )
            )
        for index, interaction in enumerate(self._replay):
            if not self._used[index] and matching.requests_match(
                request, interaction.request, self._ignore
            ):
                self._used[index] = True
                return interaction
        unused = [i for i, used in zip(self._replay, self._used, strict=True) if not used]
        if not unused:
            self._fail(
                UnexpectedCallError(
                    f"{request.method} {request.url}: this test made more calls than "
                    f"{self.cassette_path.name} recorded ({len(self._replay)}). "
                    "Re-record with --llm-record=all."
                )
            )
        diff = matching.request_diff(request, unused[0].request, self._ignore)
        self._fail(
            CassetteMismatchError(
                f"Request does not match {self.cassette_path.name}. "
                f"Re-record with --llm-record=all.\n{diff}"
            )
        )

    def _mutate(self, call: int, provider: Provider, response: RecordedResponse) -> Outcome:
        scheduled = next(
            (s for s in self._scheduled if s.target == call and isinstance(s.fault, MutateFault)),
            None,
        )
        if scheduled is None:
            return Outcome(response=response)
        scheduled.fired = True
        scheduled.remaining = 0
        fault = scheduled.fault
        try:
            if isinstance(fault, StreamCut):
                if response.events is None or len(response.events) <= fault.after_events:
                    raise FaultNotApplicableError(
                        f"needs a streaming response with more than {fault.after_events} events"
                    )
                return Outcome(response=response, cut_after=fault.after_events)
            if isinstance(fault, Truncated):
                return Outcome(response=provider.truncate(response))
            if isinstance(fault, BadToolArgs):
                return Outcome(response=provider.break_tool_args(response))
        except FaultNotApplicableError as error:
            self._fail(FaultNotApplicableError(f"{fault!r} on call {call}: {error}"))
        raise TypeError(f"unknown mutate fault {fault!r}")

    def _fail(self, error: LLMChaosError) -> NoReturn:
        """Remember the first error (the plugin reports it) and raise it to the SDK."""
        if self.error is None:
            self.error = error
        raise error


class LLM:
    """The object the ``llm`` fixture returns."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def inject(self, *faults: Fault) -> None:
        """Schedule faults; each targets ``on_call`` or, by default, the next call."""
        self._session.inject(*faults)

    def ignore(self, *paths: str) -> None:
        """Exclude dotted request-body paths from matching for the rest of this test."""
        self._session.ignore(*paths)

    @property
    def calls(self) -> int:
        """Logical calls served from the cassette or recorded so far."""
        return self._session.calls

    @property
    def attempts(self) -> int:
        """HTTP attempts that reached the interceptor, retries included."""
        return self._session.attempts

    @property
    def cassette_path(self) -> Path:
        return self._session.cassette_path
