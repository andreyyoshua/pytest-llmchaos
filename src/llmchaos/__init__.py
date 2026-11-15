"""Record real LLM responses once, replay them offline, and inject provider-accurate failures."""

from llmchaos import errors, faults

__all__ = ["__version__", "errors", "faults"]
__version__ = "0.1.0"
