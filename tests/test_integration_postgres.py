"""PostgreSQL integration tests (optional; auto-skipped without a server).

Set ``JAENYS_PG_DSN`` to a DSN with DDL rights, e.g. after::

    docker run --rm -e POSTGRES_PASSWORD=pw -p 5432:5432 postgres:16
    export JAENYS_PG_DSN="postgresql://postgres:pw@localhost:5432/postgres"

Each test creates uniquely named schemas and drops them afterwards.
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Iterator

import pytest

from jaenys import SchemaMapping, RedactionDriftError
from jaenys.sql import (
    POSTGRESQL,
    assert_guard_current,
    dialect_for,
    guard_for_connection,
    load_guard,
)

DSN = os.environ.get("JAENYS_PG_DSN")

pytestmark = pytest.mark.skipif(
    not DSN, reason="JAENYS_PG_DSN not set; skipping PostgreSQL integration tests"
)


def _connect() -> Any:
    try:
        import psycopg
    except ImportError:
        psycopg = None
    if psycopg is not None:
        conn = psycopg.connect(DSN)
        conn.autocommit = True
        return conn
    try:
        import psycopg2
    except ImportError:
        pytest.skip("neither psycopg nor psycopg2 is installed")
    conn = psycopg2.connect(DSN)
    conn.autocommit = True
    return conn


@pytest.fixture()
def pg(request: pytest.FixtureRequest) -> Iterator[tuple[Any, str, str]]:
    """(connection, primary_schema, span_schema) with tables built and seeded.

    8 records: id 2 standalone flagged; ids 5-7 a span; mirror fresh.
    """

    conn = _connect()
    suffix = uuid.uuid4().hex[:10]
    primary_schema = f"guard_primary_{suffix}"
    span_schema = f"guard_span_{suffix}"
    cursor = conn.cursor()
    cursor.execute(f"CREATE SCHEMA {primary_schema}")
    cursor.execute(f"CREATE SCHEMA {span_schema}")
    cursor.execute(f"SET search_path TO {primary_schema}")
    cursor.execute(
        "CREATE TABLE records ("
        "record_id INTEGER PRIMARY KEY,"
        "body TEXT NOT NULL,"
        "sensitive INTEGER NOT NULL DEFAULT 0)"
    )
    for record_id in range(1, 9):
        cursor.execute(
            "INSERT INTO records VALUES (%s, %s, %s)",
            (record_id, f"synthetic line {record_id}", 1 if record_id == 2 else 0),
        )
    cursor.execute(
        f"CREATE TABLE {span_schema}.span_members ("
        "span_id INTEGER NOT NULL,"
        "record_id INTEGER NOT NULL,"
        "position INTEGER NOT NULL,"
        "PRIMARY KEY (span_id, record_id))"
    )
    for position, record_id in enumerate((5, 6, 7), start=1):
        cursor.execute(
            f"INSERT INTO {span_schema}.span_members VALUES (1, %s, %s)",
            (record_id, position),
        )
    cursor.execute(
        f"CREATE TABLE {span_schema}.sensitive_records ("
        "record_id INTEGER NOT NULL,"
        "copy_reason TEXT NOT NULL,"
        "source_flag INTEGER NOT NULL DEFAULT 1)"
    )
    cursor.execute(f"INSERT INTO {span_schema}.sensitive_records VALUES (2, 'flagged', 1)")
    for record_id in (5, 6, 7):
        cursor.execute(
            f"INSERT INTO {span_schema}.sensitive_records VALUES (%s, 'span', 1)",
            (record_id,),
        )
    try:
        yield conn, primary_schema, span_schema
    finally:
        cursor = conn.cursor()
        cursor.execute(f"DROP SCHEMA IF EXISTS {primary_schema} CASCADE")
        cursor.execute(f"DROP SCHEMA IF EXISTS {span_schema} CASCADE")
        conn.close()


def test_dialect_autodetected_from_live_connection(pg: tuple[Any, str, str]) -> None:
    conn, _, _ = pg
    assert dialect_for(conn) is POSTGRESQL


def test_anti_join_chokepoint_three_states(pg: tuple[Any, str, str]) -> None:
    conn, _, span_schema = pg
    mapping = SchemaMapping(span_namespace=span_schema)
    guard = guard_for_connection(conn, mapping=mapping, dialect=POSTGRESQL)

    predicate = guard.predicate("r", include_blur=True)
    sql, params = predicate.render(POSTGRESQL)
    cursor = conn.cursor()
    cursor.execute(
        f'SELECT r."record_id", r."sensitive" FROM "records" r WHERE {sql} ORDER BY r."record_id"',
        params,
    )
    deliverable = cursor.fetchall()
    assert [row[0] for row in deliverable] == [1, 2, 3, 4, 8]  # 5-7 hidden in-span
    assert [row[0] for row in deliverable if row[1] == 1] == [2]  # the BLUR row

    strict = guard.predicate("r")
    sql, params = strict.render(POSTGRESQL)
    cursor.execute(f'SELECT COUNT(*) FROM "records" r WHERE {sql}', params)
    assert cursor.fetchone()[0] == 4  # VISIBLE only


def test_flag_edit_without_rebuild_refuses(pg: tuple[Any, str, str]) -> None:
    conn, _, span_schema = pg
    mapping = SchemaMapping(span_namespace=span_schema)
    cursor = conn.cursor()
    cursor.execute("UPDATE records SET sensitive = 1 WHERE record_id = 1")

    with pytest.raises(RedactionDriftError):
        guard_for_connection(conn, mapping=mapping, dialect=POSTGRESQL)

    # Restoring the mirror restores service.
    cursor.execute(f"INSERT INTO {span_schema}.sensitive_records VALUES (1, 'flagged', 1)")
    guard = guard_for_connection(conn, mapping=mapping, dialect=POSTGRESQL)
    assert guard.span_layer_ready is True


def test_two_connection_materialized_path(pg: tuple[Any, str, str]) -> None:
    """The universal topology: span layer read on its own connection."""

    conn, _, span_schema = pg
    span_conn = _connect()
    try:
        span_cursor = span_conn.cursor()
        span_cursor.execute(f"SET search_path TO {span_schema}")
        guard = load_guard(span_conn, dialect=POSTGRESQL, origin="pg-span-schema")
        assert guard.span_member_ids == frozenset({5, 6, 7})

        assert_guard_current(conn, guard, dialect=POSTGRESQL)  # clean -> serves

        cursor = conn.cursor()
        cursor.execute("UPDATE records SET sensitive = 1 WHERE record_id = 8")
        with pytest.raises(RedactionDriftError):
            assert_guard_current(conn, guard, dialect=POSTGRESQL)
    finally:
        span_conn.close()
