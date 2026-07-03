"""Utilities for open-weight reasoning models (Qwen3/Gemma-style).

These models emit ``<think>``/``<reasoning>``/``<analysis>`` blocks ahead of
(or instead of) the actual answer. This module strips those blocks --
including truncated/unclosed ones from a cut-off response -- before display
or JSON parsing.

This module does NOT do retry-with-repair-prompt schema validation; compose
it with a library like Instructor, Guardrails, or Outlines for that.
"""

from __future__ import annotations

import json
import re
from typing import Any

__all__ = [
    "ReasoningJSONError",
    "strip_reasoning_blocks",
    "strip_json_wrapper",
    "parse_json_content",
]

_REASONING_BLOCK_RE = re.compile(
    r"<(?P<tag>think|thinking|reasoning|analysis)\b[^>]*>.*?</(?P=tag)>",
    flags=re.IGNORECASE | re.DOTALL,
)
_UNCLOSED_REASONING_RE = re.compile(
    r"<(?:think|thinking|reasoning|analysis)\b[^>]*>",
    flags=re.IGNORECASE,
)
_ORPHAN_CLOSING_TAG_RE = re.compile(
    r"</(?:think|thinking|reasoning|analysis)>",
    flags=re.IGNORECASE,
)
_FENCED_JSON_RE = re.compile(r"^```(?:json)?\s*(?P<body>.*?)\s*```$", re.IGNORECASE | re.DOTALL)
_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)

_QUOTE_CHARS = "\"'`"


class ReasoningJSONError(RuntimeError):
    """Raised when a model response cannot be parsed as the expected JSON."""


def _is_quoted_mention(text: str, start: int, end: int) -> bool:
    """A tag wrapped in matching quote characters is answer text, not a marker.

    Real reasoning markers are emitted as bare tokens; when the answer itself
    talks about a tag it is almost always quoted or backticked
    (``the "</think>" tag``).  Requiring the SAME character on both sides
    keeps this narrow: reasoning that merely ends with a quote before its
    closing tag still counts as a marker and is stripped.
    """

    if start == 0 or end >= len(text):
        return False
    return text[start - 1] == text[end] and text[start - 1] in _QUOTE_CHARS


def _quoted_pair_spans(text: str) -> tuple[tuple[int, int], ...]:
    """Spans of paired blocks preserved as quoted mentions.

    After the paired-block pass, any paired block still present was kept
    because it is quote-wrapped; the individual tags inside it (``<think>``
    followed by content, ``</think>`` preceded by it) are not themselves
    quote-wrapped, so the single-tag passes must skip these spans too.
    """

    return tuple(
        (match.start(), match.end())
        for match in _REASONING_BLOCK_RE.finditer(text)
        if _is_quoted_mention(text, match.start(), match.end())
    )


def _fence_spans(text: str) -> tuple[tuple[int, int], ...]:
    """Spans of triple-backtick fenced blocks.

    A tag shown inside a fence is example text, not a live marker, the same
    way a quoted mention is; recomputed against the current text each pass
    because earlier substitutions shift offsets.
    """

    return tuple((match.start(), match.end()) for match in _FENCE_RE.finditer(text))


def _first_marker(pattern: re.Pattern[str], text: str) -> re.Match[str] | None:
    """First match that is a real marker (skipping quoted mentions and fences)."""

    quoted_spans = _quoted_pair_spans(text)
    fence_spans = _fence_spans(text)
    for match in pattern.finditer(text):
        if _is_quoted_mention(text, match.start(), match.end()):
            continue
        if any(start <= match.start() < end for start, end in quoted_spans):
            continue
        if any(start <= match.start() < end for start, end in fence_spans):
            continue
        return match
    return None


def _remove_unquoted_paired_blocks(text: str) -> str:
    fence_spans = _fence_spans(text)

    def replace(match: re.Match[str]) -> str:
        if _is_quoted_mention(text, match.start(), match.end()):
            return match.group(0)
        if any(start <= match.start() < end for start, end in fence_spans):
            return match.group(0)
        return ""

    return _REASONING_BLOCK_RE.sub(replace, text)


def strip_reasoning_blocks(text: str) -> str:
    """Remove common thinking/reasoning blocks while preserving the answer.

    Handles five cases:
    - Paired blocks: <think>...</think> removed entirely
    - Nested blocks: looped until no more paired blocks remain
    - Unclosed opening tags: everything from tag start is removed
    - Orphan closing tags: everything up to and including first closing tag is removed
    - Quoted mentions: a tag wrapped in matching quotes/backticks (e.g.
      ``the "</think>" tag``) or shown inside a triple-backtick fenced block
      is answer text and is preserved, never treated as a reasoning marker
    """

    cleaned = text
    # Loop paired-block substitution until no more blocks remain (handles nesting).
    while True:
        new = _remove_unquoted_paired_blocks(cleaned)
        if new == cleaned:
            break
        cleaned = new

    # A truncated response can open a reasoning block and never close it; any
    # opening tag that survived the paired-block pass has no closing tag, so
    # everything from there on is reasoning, not answer.
    unclosed = _first_marker(_UNCLOSED_REASONING_RE, cleaned)
    if unclosed:
        cleaned = cleaned[: unclosed.start()]

    # Some models omit the opening tag and start mid-reasoning; if an orphan
    # closing tag remains, everything up to and including it is reasoning.
    orphan_close = _first_marker(_ORPHAN_CLOSING_TAG_RE, cleaned)
    if orphan_close:
        cleaned = cleaned[orphan_close.end() :]

    return cleaned.strip()


def strip_json_wrapper(text: str) -> str:
    """Remove a whole-response JSON code fence if the model added one."""

    cleaned = text.strip()
    match = _FENCED_JSON_RE.match(cleaned)
    if match:
        return match.group("body").strip()
    return cleaned


def parse_json_content(content: str, *, require_object: bool = True) -> Any:
    """Parse strict JSON after known model wrapper cleanup.

    The content is tried as JSON before reasoning stripping, so a valid answer
    whose string values merely mention a tag (``{"note": "<think> later"}``)
    is not mangled by the reasoning cleaner.  Only if the direct parse fails is
    the content reparsed with reasoning blocks removed, which is what recovers
    the answer from a response that leads with real reasoning.
    """

    fenced = strip_json_wrapper(content)
    candidates = [fenced]
    reasoned = strip_json_wrapper(strip_reasoning_blocks(content))
    if reasoned != fenced:
        candidates.append(reasoned)

    last_error: json.JSONDecodeError | None = None
    non_object_seen = False
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if require_object and not isinstance(parsed, dict):
            # A later candidate (e.g. the reasoning-stripped one) may still
            # yield the object, so don't give up on the first non-object parse;
            # only report "not an object" if no candidate produced one.
            non_object_seen = True
            continue
        return parsed
    if non_object_seen:
        raise ReasoningJSONError("Expected a top-level JSON object.")
    raise ReasoningJSONError(f"Model response was not valid JSON: {last_error}")
