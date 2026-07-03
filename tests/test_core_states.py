"""Three-state classification: VISIBLE / BLUR (standalone) / HIDDEN (in-span)."""

from __future__ import annotations

import pytest

from jaenys import (
    BLUR,
    HIDDEN,
    VISIBLE,
    SchemaMapping,
    RedactionDriftError,
    annotate_rows,
    classify,
    filter_visible_rows,
    load_guard_from_adapter,
)
from jaenys.adapters import InMemoryAdapter
from jaenys.core import validate_name


SPAN = frozenset({3, 4})


@pytest.mark.parametrize(
    ("record_id", "flag", "expected"),
    [
        (1, 0, VISIBLE),  # clean, out of span
        (2, 1, BLUR),  # standalone flagged
        (3, 0, HIDDEN),  # neutral row inside a span still hides
        (4, 1, HIDDEN),  # flagged row inside a span hides (span wins)
    ],
)
def test_classify_matrix(record_id: int, flag: int, expected: str) -> None:
    assert classify(record_id, flag, SPAN) == expected


def test_annotate_rows_keeps_and_marks_blur_drops_hidden() -> None:
    rows = [
        {"record_id": 1, "sensitive": 0, "body": "clean"},
        {"record_id": 2, "sensitive": 1, "body": "standalone"},
        {"record_id": 3, "sensitive": 0, "body": "in-span neutral"},
        {"record_id": 4, "sensitive": 1, "body": "in-span flagged"},
    ]
    annotated = annotate_rows(rows, span_member_ids=SPAN)
    assert [(row["record_id"], row["blurred"]) for row in annotated] == [(1, False), (2, True)]
    # Input rows are not mutated.
    assert "blurred" not in rows[0]


def test_filter_visible_rows_is_strict() -> None:
    rows = [
        {"record_id": 1, "sensitive": 0},
        {"record_id": 2, "sensitive": 1},
        {"record_id": 3, "sensitive": 0},
    ]
    assert [row["record_id"] for row in filter_visible_rows(rows, span_member_ids=SPAN)] == [1]


def test_rows_missing_keys_are_rejected() -> None:
    with pytest.raises(RedactionDriftError, match="must include"):
        annotate_rows([{"record_id": 1}], span_member_ids=SPAN)
    with pytest.raises(RedactionDriftError, match="must include"):
        filter_visible_rows([{"sensitive": 0}], span_member_ids=SPAN)


def test_custom_row_keys() -> None:
    rows = [{"pk": 7, "is_private": 1}]
    annotated = annotate_rows(rows, span_member_ids=frozenset(), id_key="pk", flag_key="is_private")
    assert annotated[0]["blurred"] is True


def test_guard_methods_delegate() -> None:
    adapter = InMemoryAdapter({1: 0, 2: 1, 3: 0, 4: 1})
    adapter.rebuild_derived_layer(span_members={3, 4})
    guard = load_guard_from_adapter(adapter)
    assert guard.classify(3, 0) == HIDDEN
    assert guard.classify(2, 1) == BLUR
    assert guard.classify(1, 0) == VISIBLE
    kept = guard.annotate_rows([{"record_id": 2, "sensitive": 1}])
    assert kept[0]["blurred"] is True


def test_schema_mapping_rejects_unsafe_names() -> None:
    with pytest.raises(RedactionDriftError, match="unsafe"):
        SchemaMapping(record_table="records; DROP TABLE records")
    with pytest.raises(RedactionDriftError, match="unsafe"):
        SchemaMapping(flag_column='x" OR 1=1 --')
    with pytest.raises(RedactionDriftError, match="non-empty"):
        SchemaMapping(reason_flagged="")


def test_schema_mapping_rejects_colliding_reasons() -> None:
    with pytest.raises(RedactionDriftError, match="reason_flagged and reason_span must differ"):
        SchemaMapping(reason_flagged="x", reason_span="x")


def test_annotate_rows_rejects_none_flag() -> None:
    rows = [{"record_id": 1, "sensitive": None}]
    with pytest.raises(RedactionDriftError, match="flag is missing"):
        annotate_rows(rows, span_member_ids=SPAN)


def test_annotate_rows_rejects_junk_string_id() -> None:
    rows = [{"record_id": "abc", "sensitive": 0}]
    with pytest.raises(RedactionDriftError, match="record id is not an integer"):
        annotate_rows(rows, span_member_ids=SPAN)


def test_annotate_rows_accepts_integral_float_id() -> None:
    rows = [{"record_id": 3.0, "sensitive": 0}]
    annotated = annotate_rows(rows, span_member_ids=SPAN)
    # record 3 is in SPAN -> HIDDEN -> dropped, proving the float id 3.0 was
    # correctly recognized as span member 3 rather than rejected.
    assert annotated == []

    rows = [{"record_id": 7.0, "sensitive": 1}]
    annotated = annotate_rows(rows, span_member_ids=SPAN)
    assert annotated[0]["record_id"] == 7.0
    assert annotated[0]["blurred"] is True


def test_annotate_rows_rejects_non_integral_float_id() -> None:
    rows = [{"record_id": 3.7, "sensitive": 0}]
    with pytest.raises(RedactionDriftError, match="not an integral value"):
        annotate_rows(rows, span_member_ids=SPAN)


def test_filter_visible_rows_rejects_none_flag_and_junk_id() -> None:
    with pytest.raises(RedactionDriftError, match="flag is missing"):
        filter_visible_rows([{"record_id": 1, "sensitive": None}], span_member_ids=SPAN)
    with pytest.raises(RedactionDriftError, match="record id is not an integer"):
        filter_visible_rows([{"record_id": "nope", "sensitive": 0}], span_member_ids=SPAN)


def test_annotate_rows_refuses_out_of_domain_flag() -> None:
    # A corrupt flag (2) would classify VISIBLE under a bare `flag == 1` check
    # and leak the record on the Python/adapter path while the SQL predicate
    # (flag = 0) excludes it. The store must REFUSE instead.
    rows = [{"record_id": 9, "sensitive": 2, "body": "corrupt flag"}]
    with pytest.raises(RedactionDriftError, match="flag must be 0 or 1"):
        annotate_rows(rows, span_member_ids=SPAN)


def test_filter_visible_rows_refuses_out_of_domain_flag() -> None:
    rows = [{"record_id": 9, "sensitive": -1}]
    with pytest.raises(RedactionDriftError, match="flag must be 0 or 1"):
        filter_visible_rows(rows, span_member_ids=SPAN)


def test_classify_refuses_junk_flag_with_redaction_drift_error() -> None:
    # classify now routes the flag through coerce_flag, so junk raises the
    # fail-closed error rather than a raw ValueError escaping the call site.
    with pytest.raises(RedactionDriftError):
        classify(1, "junk", SPAN)


def test_classify_refuses_out_of_domain_flag() -> None:
    with pytest.raises(RedactionDriftError, match="flag must be 0 or 1"):
        classify(1, 2, SPAN)


def test_classify_refuses_non_integral_id_instead_of_truncating() -> None:
    # classify(3.7, ...) must not silently truncate to record 3 (a span member)
    # -- it refuses via coerce_record_id.
    with pytest.raises(RedactionDriftError, match="not an integral value"):
        classify(3.7, 0, SPAN)


def test_validate_name_rejects_trailing_newline() -> None:
    # `$` matches before a trailing newline; `\Z` does not -- so "records\n"
    # must be refused rather than reaching a SQL string.
    with pytest.raises(RedactionDriftError, match="unsafe"):
        validate_name("records\n")


def test_schema_mapping_rejects_trailing_newline_name() -> None:
    with pytest.raises(RedactionDriftError, match="unsafe"):
        SchemaMapping(record_table="records\n")
