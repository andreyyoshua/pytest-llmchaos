"""Keep secrets out of cassettes: header allowlists and a secret scan."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import Any

REQUEST_HEADER_ALLOWLIST = frozenset(
    {"content-type", "anthropic-version", "anthropic-beta", "openai-beta"}
)
RESPONSE_HEADER_ALLOWLIST = frozenset({"content-type", "request-id", "x-request-id"})

# The lookbehind avoids false hits inside words such as "risk-assessment-...".
KEY_PATTERN = re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}")
KEY_ENV_VARS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
MIN_ENV_SECRET_LENGTH = 8


def filter_headers(headers: Mapping[str, str], allowlist: frozenset[str]) -> dict[str, str]:
    return {key.lower(): value for key, value in headers.items() if key.lower() in allowlist}


def find_secrets(document: Any, environ: Mapping[str, str] | None = None) -> list[str]:
    """JSON paths of every string in ``document`` that looks like an API key."""
    env = os.environ if environ is None else environ
    literals = [
        value for name in KEY_ENV_VARS if len(value := env.get(name, "")) >= MIN_ENV_SECRET_LENGTH
    ]
    hits: list[str] = []
    _walk(document, "$", literals, hits)
    return hits


def _walk(node: Any, path: str, literals: list[str], hits: list[str]) -> None:
    if isinstance(node, str):
        if KEY_PATTERN.search(node) or any(literal in node for literal in literals):
            hits.append(path)
    elif isinstance(node, dict):
        for key, value in node.items():
            _walk(value, f"{path}.{key}", literals, hits)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _walk(value, f"{path}[{index}]", literals, hits)
