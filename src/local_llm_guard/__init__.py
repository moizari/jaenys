"""Small, dependency-free guards for calling local LLMs: hard local-only
endpoint enforcement and reasoning-block cleanup."""

from __future__ import annotations

from .reasoning import (
    ReasoningJSONError,
    parse_json_content,
    strip_json_wrapper,
    strip_reasoning_blocks,
)
from .urls import DEFAULT_ALLOWED_HOSTS, LocalEndpointError, enforce_local_url

__version__ = "0.1.1"

__all__ = [
    "enforce_local_url",
    "DEFAULT_ALLOWED_HOSTS",
    "LocalEndpointError",
    "strip_reasoning_blocks",
    "strip_json_wrapper",
    "parse_json_content",
    "ReasoningJSONError",
    "__version__",
]
