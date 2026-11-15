"""Exceptions raised by pytest-llmchaos."""


class LLMChaosError(Exception):
    """Base class for every error pytest-llmchaos raises.

    Deliberately not a subclass of any httpx exception: SDKs that only retry
    network errors then let it propagate unchanged.
    """


class CassetteMissingError(LLMChaosError):
    """Replay-only mode, but the test has no cassette."""


class CassetteMismatchError(LLMChaosError):
    """A request matches no unused recorded interaction."""


class UnexpectedCallError(LLMChaosError):
    """The test made more calls than its cassette recorded."""


class UnusedFaultError(LLMChaosError):
    """An injected fault never fired."""


class FaultNotApplicableError(LLMChaosError):
    """A mutate fault cannot be applied to the response it targets."""


class SecretLeakError(LLMChaosError):
    """A cassette about to be written contains something that looks like an API key."""
