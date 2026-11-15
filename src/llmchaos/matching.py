"""Request matching: ignore paths, equality, and readable diffs."""

from __future__ import annotations

import copy
import difflib
import json
from collections.abc import Iterable, Sequence
from typing import Any

from llmchaos.cassette import RecordedRequest

IGNORED = "<ignored>"


def apply_ignores(body: Any, paths: Iterable[str]) -> Any:
    """A copy of ``body`` with the value at each dotted path replaced by ``IGNORED``."""
    result = copy.deepcopy(body)
    for path in paths:
        _ignore(result, path.split("."))
    return result


def requests_match(
    actual: RecordedRequest, recorded: RecordedRequest, ignore: Sequence[str]
) -> bool:
    if actual.method != recorded.method or actual.url != recorded.url:
        return False
    if actual.body_b64 is not None or recorded.body_b64 is not None:
        return actual.body_b64 == recorded.body_b64
    return apply_ignores(actual.body, ignore) == apply_ignores(recorded.body, ignore)


def request_diff(actual: RecordedRequest, recorded: RecordedRequest, ignore: Sequence[str]) -> str:
    return "\n".join(
        difflib.unified_diff(
            _render(recorded, ignore),
            _render(actual, ignore),
            fromfile="recorded",
            tofile="actual",
            lineterm="",
        )
    )


def _ignore(node: Any, segments: list[str]) -> None:
    head, rest = segments[0], segments[1:]
    for key in _matching_keys(node, head):
        if rest:
            _ignore(node[key], rest)
        else:
            node[key] = IGNORED


def _matching_keys(node: Any, segment: str) -> list[Any]:
    if isinstance(node, dict):
        if segment == "*":
            return list(node)
        return [segment] if segment in node else []
    if isinstance(node, list):
        if segment == "*":
            return list(range(len(node)))
        if segment.isdigit() and int(segment) < len(node):
            return [int(segment)]
    return []


def _render(request: RecordedRequest, ignore: Sequence[str]) -> list[str]:
    if request.body_b64 is not None:
        body = [f"body_b64: {request.body_b64}"]
    else:
        body = json.dumps(
            apply_ignores(request.body, ignore), indent=2, sort_keys=True, ensure_ascii=False
        ).splitlines()
    return [f"{request.method} {request.url}", *body]
