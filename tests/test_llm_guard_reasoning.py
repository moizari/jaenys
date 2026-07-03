"""Reasoning-block stripping and strict JSON parsing behavior."""

from __future__ import annotations

import pytest

from local_llm_guard import ReasoningJSONError, parse_json_content, strip_reasoning_blocks


def test_parse_json_content_strips_thinking_block_and_code_fence() -> None:
    content = '<think>private chain of thought</think>\n```json\n{"flag": true}\n```'

    assert parse_json_content(content) == {"flag": True}


def test_strip_reasoning_blocks_drops_unterminated_trailing_block() -> None:
    truncated = '{"flag": true}\n<think>this response was cut off mid-reason'

    assert strip_reasoning_blocks(truncated) == '{"flag": true}'


def test_strip_reasoning_blocks_entirely_reasoning_content_is_empty() -> None:
    assert strip_reasoning_blocks("<think>entirely reasoning, no close tag") == ""


def test_parse_json_content_preserves_literal_tag_in_string_value() -> None:
    """A valid answer that merely mentions a tag inside a string must survive."""

    content = '{"summary": "the user said to <think> about it later"}'

    assert parse_json_content(content) == {"summary": "the user said to <think> about it later"}


def test_parse_json_content_fenced_json_with_literal_tag_value() -> None:
    content = '```json\n{"note": "close the block with </think>"}\n```'

    assert parse_json_content(content) == {"note": "close the block with </think>"}


def test_parse_json_content_raises_on_non_json() -> None:
    with pytest.raises(ReasoningJSONError):
        parse_json_content("not json at all")


def test_parse_json_content_raises_on_non_object_when_require_object() -> None:
    with pytest.raises(ReasoningJSONError):
        parse_json_content("[1, 2, 3]", require_object=True)


def test_parse_json_content_allows_json_list_when_require_object_false() -> None:
    assert parse_json_content("[1, 2, 3]", require_object=False) == [1, 2, 3]


def test_parse_json_content_handles_nested_and_multiple_think_blocks() -> None:
    content = (
        "<think>outer <reasoning>nested</reasoning> still thinking</think>"
        '{"a": 1}'
        "<analysis>trailing analysis block</analysis>"
    )

    assert parse_json_content(content) == {"a": 1}


def test_parse_json_content_case_insensitive_tags() -> None:
    content = '<THINK>upper case reasoning tag</THINK>\n{"ok": true}'

    assert parse_json_content(content) == {"ok": True}


def test_strip_reasoning_blocks_nested_blocks() -> None:
    nested = "<think>a<think>b</think>c</think>ANSWER"

    assert strip_reasoning_blocks(nested) == "ANSWER"


def test_strip_reasoning_blocks_orphan_closing_tag() -> None:
    orphan = 'some reasoning</think>{"ok": true}'

    result = strip_reasoning_blocks(orphan)
    # The orphan closing tag is removed along with everything before it.
    assert result == '{"ok": true}'

    # Verify JSON parsing works on the cleaned result.
    assert parse_json_content(orphan) == {"ok": True}


def test_strip_reasoning_blocks_normal_answer_untouched() -> None:
    normal = '{"answer": "hello"}'

    assert strip_reasoning_blocks(normal) == normal


def test_quoted_closing_tag_mention_is_answer_text() -> None:
    """Answer text that MENTIONS the tag must survive: it is not a marker."""

    text = 'The literal tag "</think>" marks the end of reasoning.'

    assert strip_reasoning_blocks(text) == text


def test_quoted_opening_tag_mention_is_answer_text() -> None:
    text = 'Wrap chain-of-thought in "<think>" before answering.'

    assert strip_reasoning_blocks(text) == text


def test_backtick_wrapped_tag_mention_is_answer_text() -> None:
    text = "Use `</think>` to close the block."

    assert strip_reasoning_blocks(text) == text


def test_quoted_paired_block_mention_is_answer_text() -> None:
    text = 'The pattern "<think>x</think>" is removed by the cleaner.'

    assert strip_reasoning_blocks(text) == text


def test_real_orphan_after_quoted_mention_still_strips() -> None:
    """Quote awareness must not weaken real stripping: the bare tag still wins."""

    orphan = 'reasoning that mentions "</think>" in passing</think>{"ok": true}'

    assert strip_reasoning_blocks(orphan) == '{"ok": true}'
    assert parse_json_content(orphan) == {"ok": True}


def test_fenced_literal_tag_is_answer_text() -> None:
    """A tag shown inside a code fence is example text, not a reasoning marker."""

    text = "```\n<think>\n```\nThe answer is 42."

    assert strip_reasoning_blocks(text) == text


def test_fenced_paired_block_is_answer_text() -> None:
    text = "Example: ```<think>x</think>``` shows the format.\nThe answer is 42."

    assert strip_reasoning_blocks(text) == text


def test_real_reasoning_block_containing_a_fence_still_strips() -> None:
    """A fence nested inside a real reasoning block does not shield it."""

    text = "<think>reasoning with ```code``` inside</think>ANSWER"

    assert strip_reasoning_blocks(text) == "ANSWER"
