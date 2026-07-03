"""SQL predicate behavior on the SQLite reference engine.

Covers the serve-path chokepoint: predicate exclusion, identifier
validation, search-index protection, id filtering, blur inclusion, version
scoping, and counts-only status.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from jaenys import RedactionDriftError
from jaenys.sql import (
    filter_visible_ids,
    normal_record_predicate,
    sqlite,
)
from tests.conftest import build_primary_db, create_mirror_table, create_span_member_table


def query_ids(db_path: Path, predicate) -> list[int]:
    with closing(sqlite3.connect(db_path)) as conn:
        rows = conn.execute(
            f"SELECT r.record_id FROM records r WHERE {predicate.sql} ORDER BY r.record_id",
            predicate.params,
        ).fetchall()
    return [row[0] for row in rows]


def test_normal_predicate_excludes_flags_and_span_ids(store_pair: tuple[Path, Path]) -> None:
    primary, _ = store_pair
    ids = build_primary_db(primary, [0, 1, 0, 0])
    predicate = normal_record_predicate("r", [ids[2]])
    assert query_ids(primary, predicate) == [ids[0], ids[3]]


def test_include_blur_keeps_standalone_flagged_rows(store_pair: tuple[Path, Path]) -> None:
    primary, _ = store_pair
    ids = build_primary_db(primary, [0, 1, 0, 1])
    predicate = normal_record_predicate("r", [ids[3]], include_blur=True)
    # Standalone flagged ids[1] stays; in-span ids[3] never leaves.
    assert query_ids(primary, predicate) == [ids[0], ids[1], ids[2]]


def test_unsafe_alias_is_rejected() -> None:
    with pytest.raises(RedactionDriftError, match="unsafe"):
        normal_record_predicate("r; DROP TABLE records", [])


def test_empty_span_store_means_flag_filter_only(store_pair: tuple[Path, Path]) -> None:
    primary, span = store_pair
    ids = build_primary_db(primary, [0, 1, 0])
    guard = sqlite.load_guard(span)
    assert guard.span_layer_ready is False
    assert guard.span_member_ids == frozenset()
    predicate = normal_record_predicate("r", guard.span_member_ids)
    assert query_ids(primary, predicate) == [ids[0], ids[2]]


def test_loads_member_table_spans(store_pair: tuple[Path, Path]) -> None:
    primary, span = store_pair
    ids = build_primary_db(primary, [0, 0, 0, 0])
    create_span_member_table(span, [ids[1], ids[2]])
    guard = sqlite.load_guard(span)
    assert guard.span_layer_ready is True
    assert guard.span_sources == ("span_members",)
    assert guard.span_member_ids == frozenset({ids[1], ids[2]})


def test_loads_mirror_span_rows_as_fallback(store_pair: tuple[Path, Path]) -> None:
    primary, span = store_pair
    ids = build_primary_db(primary, [0, 0, 0])
    create_mirror_table(span, span_ids=[ids[1]], modern=False)
    guard = sqlite.load_guard(span)
    assert guard.span_sources == ("mirror.span",)
    assert guard.span_member_ids == frozenset({ids[1]})


def test_attached_predicate_protects_search_results_from_stale_index(
    store_pair: tuple[Path, Path],
) -> None:
    primary, span = store_pair
    ids = build_primary_db(primary, [0, 1, 0, 0])
    create_span_member_table(span, [ids[2]])
    conn = sqlite3.connect(primary)
    conn.execute("CREATE VIRTUAL TABLE record_search USING fts5(body)")
    conn.execute("INSERT INTO record_search (rowid, body) SELECT record_id, body FROM records")
    conn.commit()
    conn.close()

    with closing(sqlite.open_readonly(primary)) as conn:
        with sqlite.attached_guard(conn, span) as guard:
            predicate = guard.predicate("r")
            rows = conn.execute(
                "SELECT r.record_id FROM record_search s"
                " JOIN records r ON r.record_id = s.rowid"
                f" WHERE record_search MATCH ? AND {predicate.sql}"
                " ORDER BY r.record_id",
                ("synthetic",) + predicate.params,
            ).fetchall()
    assert [row[0] for row in rows] == [ids[0], ids[3]]


def test_attached_include_blur_keeps_standalone_hides_span(
    store_pair: tuple[Path, Path],
) -> None:
    primary, span = store_pair
    ids = build_primary_db(primary, [0, 1, 0, 1, 0])
    create_span_member_table(span, [ids[2], ids[3]])
    create_mirror_table(span, flagged_ids=[ids[1], ids[3]], span_ids=[ids[2], ids[3]])
    with closing(sqlite.open_readonly(primary)) as conn:
        with sqlite.attached_guard(conn, span) as guard:
            predicate = guard.predicate("r", include_blur=True)
            rows = conn.execute(
                f"SELECT r.record_id FROM records r WHERE {predicate.sql} ORDER BY r.record_id",
                predicate.params,
            ).fetchall()
    assert [row[0] for row in rows] == [ids[0], ids[1], ids[4]]


def test_attached_mirror_span_superset_of_members_still_hidden(
    store_pair: tuple[Path, Path],
) -> None:
    """Both span shapes present with the mirror-span a superset of the members
    table: the unversioned anti-join must still exclude the mirror-only span
    id, matching the materialized guard (which unions both sources). Otherwise
    an inconsistent or externally built span store under-filters here while the
    materialized path hides the row -- the two serve paths must agree.
    """

    primary, span = store_pair
    ids = build_primary_db(primary, [0, 0, 0, 0])
    # The members table covers ids[1], ids[2]; the mirror's span rows add ids[3]
    # -- an inconsistency the reference writer never produces but an external
    # pipeline can.
    create_span_member_table(span, [ids[1], ids[2]])
    create_mirror_table(span, span_ids=[ids[1], ids[2], ids[3]])

    guard = sqlite.load_guard(span)
    assert guard.span_member_ids == frozenset({ids[1], ids[2], ids[3]})

    with closing(sqlite.open_readonly(primary)) as conn:
        with sqlite.attached_guard(conn, span) as attached:
            predicate = attached.predicate("r")
            rows = conn.execute(
                f"SELECT r.record_id FROM records r WHERE {predicate.sql} ORDER BY r.record_id",
                predicate.params,
            ).fetchall()
    # ids[3] is a mirror-only span member but must not leak.
    assert [row[0] for row in rows] == [ids[0]]


def test_attached_version_scoping_uses_requested_snapshot(
    store_pair: tuple[Path, Path],
) -> None:
    primary, span = store_pair
    ids = build_primary_db(primary, [0, 0, 0, 0])
    # Version 1 span covers ids[1]; version 2 span covers ids[2].
    create_span_member_table(span, [ids[1]], span_id=1, version_id=1)
    create_span_member_table(span, [ids[2]], span_id=2, version_id=2)
    with closing(sqlite.open_readonly(primary)) as conn:
        with sqlite.attached_guard(conn, span) as guard:
            assert "span_members.versioned" in guard.span_sources
            v1 = guard.predicate("r", version_id=1)
            rows = conn.execute(
                f"SELECT r.record_id FROM records r WHERE {v1.sql} ORDER BY r.record_id",
                v1.params,
            ).fetchall()
    # Only the version-1 span hides for a version-1 request.
    assert [row[0] for row in rows] == [ids[0], ids[2], ids[3]]


def test_attached_phantom_version_id_refuses(store_pair: tuple[Path, Path]) -> None:
    """A version the span layer never derived would anti-join nothing and leak."""

    primary, span = store_pair
    build_primary_db(primary, [0, 0, 0, 0])
    create_span_member_table(span, [2], span_id=1, version_id=1)
    with closing(sqlite.open_readonly(primary)) as conn:
        with sqlite.attached_guard(conn, span) as guard:
            with pytest.raises(RedactionDriftError, match="no derivation for version"):
                guard.predicate("r", version_id=99)


def test_attached_version_id_against_unversioned_store_refuses(
    store_pair: tuple[Path, Path],
) -> None:
    """version_id must not be silently ignored when no versioned layer exists."""

    primary, span = store_pair
    build_primary_db(primary, [0, 0, 0])
    create_span_member_table(span, [2])  # no versioned span-group table
    with closing(sqlite.open_readonly(primary)) as conn:
        with sqlite.attached_guard(conn, span) as guard:
            assert "span_members.versioned" not in guard.span_sources
            with pytest.raises(RedactionDriftError, match="no versioned span layer"):
                guard.predicate("r", version_id=1)


def test_filter_visible_ids_preserves_order_and_duplicates(
    store_pair: tuple[Path, Path],
) -> None:
    primary, span = store_pair
    ids = build_primary_db(primary, [0, 1, 0, 0])
    create_span_member_table(span, [ids[2]])
    guard = sqlite.load_guard(span)
    with closing(sqlite.open_readonly(primary)) as conn:
        visible = filter_visible_ids(
            conn, [ids[3], ids[2], ids[1], ids[0], 9999, ids[0]], guard=guard
        )
    assert visible == [ids[3], ids[0], ids[0]]


def test_status_is_counts_only_and_read_only(store_pair: tuple[Path, Path]) -> None:
    primary, span = store_pair
    ids = build_primary_db(primary, [0, 1, 0, 0])
    create_span_member_table(span, [ids[2]])
    primary_before = hashlib.sha256(primary.read_bytes()).hexdigest()
    span_before = hashlib.sha256(span.read_bytes()).hexdigest()

    report = sqlite.status(primary, span)

    assert hashlib.sha256(primary.read_bytes()).hexdigest() == primary_before
    assert hashlib.sha256(span.read_bytes()).hexdigest() == span_before
    assert report["primary_db"]["records"] == 4
    assert report["guard"]["excluded_by_flag"] == 1
    assert report["guard"]["excluded_by_span_only"] == 1
    assert report["guard"]["visible_normal_records"] == 2
    assert report["guard"]["blurred_standalone"] == 1
    assert report["guard"]["hidden_in_span"] == 1
    # Counts only: no record content may appear anywhere in the report.
    assert "synthetic ticket line" not in json.dumps(report)
