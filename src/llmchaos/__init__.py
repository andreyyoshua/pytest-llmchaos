"""Record real LLM responses once, replay them offline, and inject provider-accurate failures."""

from llmchaos import errors, faults
from llmchaos.session import LLM

__all__ = ["LLM", "__version__", "errors", "faults"]
__version__ = "0.1.0"
