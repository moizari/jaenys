"""SQL backend of the Jaenys visibility engine.

Serve-path queries on the primary store should pass through this module.  It
excludes both sensitivity layers from normal display/search/export surfaces:

* records whose live flag is set (``flag_column = 1``) in the primary store
* any record id present in the derived span layer

Typical usage with one connection that can reach both layers (span tables
under ``mapping.span_namespace``)::

    guard = guard_for_connection(conn, mapping=mapping, dialect=POSTGRESQL)
    predicate = guard.predicate("r")
    sql, params = predicate.render(POSTGRESQL)
    rows = conn.execute(f"SELECT r.* FROM records r WHERE {sql}", params)

For two physically separate stores (any mix of engines), load a materialized
guard from the span store and verify it against the primary store::

    guard = load_guard(span_conn, dialect=SQLITE, origin="span.db")
    assert_guard_current(primary_conn, guard, dialect=POSTGRESQL)

This module is read-only: it never writes to either store.

Verification vs. serving atomicity (TOCTOU): freshness verification and the
serve query are separate statements, so a flag edit that lands between them
is invisible to a predicate that was already issued -- the guard verifies the
state it can see, not the state a concurrent writer is creating.  Strict
deployments should run verify + serve inside one transaction/snapshot (e.g.
``REPEATABLE READ``, or SQLite's single-connection snapshot semantics), or
re-verify after reading and discard the rows on refusal.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Sequence

from ..core import (
    DEFAULT_MAPPING,
    SchemaMapping,
    Guard,
    RedactionDriftError,
    _coerce_int,
    coerce_flag,
    coerce_record_id,
    verify_drift_witness,
    verify_guard_current,
)
from .dialects import (
    INTROSPECTION_ORACLE,
    INTROSPECTION_SQLITE,
    SQLITE,
    Dialect,
    dialect_for,
)

__all__ = [
    "SqlPredicate",
    "AttachedGuard",
    "flag_predicate",
    "not_in_span_predicate",
    "normal_record_predicate",
    "table_exists",
    "table_columns",
    "span_sources",
    "load_guard",
    "guard_for_connection",
    "assert_guard_current",
    "filter_visible_ids",
    "is_record_visible",
    "status",
]

# Span-source names detected in a span store.
SOURCE_MEMBERS = "span_members"
SOURCE_MEMBERS_VERSIONED = "span_members.versioned"
SOURCE_MIRROR_SPAN = "mirror.span"

# Fixed layout of the drift-witness table (``mapping.meta_table``) in the
# primary store.  The table is wholly toolkit-owned (created by the writers),
# so unlike the record/span tables its column names are not mapped.
META_LAYER_SPAN = "span"
META_LAYER_COLUMN = "layer"
META_COUNT_COLUMN = "member_count"
META_DIGEST_COLUMN = "member_digest"
META_DERIVED_AT_COLUMN = "derived_at"

# Match-all predicate sentinel: no ids to exclude, so every row qualifies.
_MATCH_ALL_SQL = "1 = 1"


@contextmanager
def _fail_closed(label: str) -> Iterator[None]:
    """Convert a raw driver error on a serve-path entry point into a refusal.

    :class:`RedactionDriftError` (already a refusal) passes through; a closed
    connection, corrupt file, or any other driver error becomes the standard
    refusal so callers' ``except RedactionDriftError`` handlers still catch it.
    """

    try:
        yield
    except RedactionDriftError:
        raise
    except Exception as exc:
        raise RedactionDriftError(
            f"{label} is unreachable or unreadable; refusing normal output."
        ) from exc


@dataclass(frozen=True)
class SqlPredicate:
    """Parameterized SQL fragment (qmark form) intended for a WHERE clause.

    ``sql``/``params`` are directly usable on qmark drivers (sqlite3,
    pyodbc).  For other engines call :meth:`render`.
    """

    sql: str
    params: tuple[Any, ...] = ()

    def render(self, dialect: Dialect | str) -> tuple[str, Any]:
        return dialect_for(dialect).render(self.sql, self.params)


def _fetchall(conn: Any, dialect: Dialect, sql: str, params: Sequence[Any] = ()) -> list[tuple]:
    """Render and execute one statement, returning every row.

    PEP 249 drivers (pyodbc, psycopg, ...) get an explicit cursor that is
    closed afterward so open cursors don't hold locks (notably on MSSQL);
    sqlite3-style connections that execute directly reuse the connection's
    own result object and need no separate close.
    """

    rendered_sql, rendered_params = dialect.render(sql, params)
    if hasattr(conn, "cursor"):
        cursor = conn.cursor()
        try:
            cursor.execute(rendered_sql, rendered_params)
            return list(cursor.fetchall())
        finally:
            cursor.close()
    return list(conn.execute(rendered_sql, rendered_params).fetchall())


def _fetchone(conn: Any, dialect: Dialect, sql: str, params: Sequence[Any] = ()) -> tuple | None:
    """Render and execute one statement, returning the first row (or None)."""

    rendered_sql, rendered_params = dialect.render(sql, params)
    if hasattr(conn, "cursor"):
        cursor = conn.cursor()
        try:
            cursor.execute(rendered_sql, rendered_params)
            return cursor.fetchone()
        finally:
            cursor.close()
    return conn.execute(rendered_sql, rendered_params).fetchone()


def _iter_rows(
    conn: Any,
    dialect: Dialect,
    sql: str,
    params: Sequence[Any] = (),
    *,
    batch_size: int = 1000,
) -> Iterator[tuple]:
    """Stream statement rows without materializing a primary store in memory."""

    rendered_sql, rendered_params = dialect.render(sql, params)
    if hasattr(conn, "cursor"):
        cursor = conn.cursor()
        try:
            cursor.execute(rendered_sql, rendered_params)
            while rows := cursor.fetchmany(batch_size):
                yield from rows
        finally:
            cursor.close()
        return
    cursor = conn.execute(rendered_sql, rendered_params)
    try:
        while rows := cursor.fetchmany(batch_size):
            yield from rows
    finally:
        close = getattr(cursor, "close", None)
        if close is not None:
            close()


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------


def _information_schema_relation(
    dialect: Dialect, namespace: str | None, relation: str
) -> tuple[str, str | None]:
    """Return an information-schema relation and the schema filter to use."""

    parts = dialect.namespace_parts(namespace)
    if dialect.name == "mssql" and len(parts) == 2:
        database, schema = parts
        return f"{dialect.quote_identifier(database)}.information_schema.{relation}", schema
    return f"information_schema.{relation}", namespace


def table_exists(
    conn: Any,
    table_name: str,
    *,
    dialect: Dialect | str = SQLITE,
    namespace: str | None = None,
) -> bool:
    dialect = dialect_for(dialect)
    if dialect.introspection == INTROSPECTION_SQLITE:
        prefix = dialect.namespace_prefix(namespace or "main")
        row = _fetchone(
            conn,
            dialect,
            f"SELECT name FROM {prefix}sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        )
        return row is not None
    if dialect.introspection == INTROSPECTION_ORACLE:
        sql = "SELECT table_name FROM all_tables WHERE upper(table_name) = upper(?)"
        params: tuple[Any, ...] = (table_name,)
        if namespace:
            sql += " AND upper(owner) = upper(?)"
            params += (namespace,)
        return _fetchone(conn, dialect, sql, params) is not None
    information_schema_tables, schema_namespace = _information_schema_relation(
        dialect, namespace, "tables"
    )
    sql = f"SELECT table_name FROM {information_schema_tables} WHERE lower(table_name) = lower(?)"
    params = (table_name,)
    if schema_namespace:
        sql += " AND lower(table_schema) = lower(?)"
        params += (schema_namespace,)
    elif dialect.current_schema_expr:
        # Unqualified lookups must not false-positive on a same-named table in
        # another schema.  The expression is a trusted dialect constant, not
        # user input, so it is interpolated rather than bound.
        sql += f" AND lower(table_schema) = lower({dialect.current_schema_expr})"
    return _fetchone(conn, dialect, sql, params) is not None


def table_columns(
    conn: Any,
    table_name: str,
    *,
    dialect: Dialect | str = SQLITE,
    namespace: str | None = None,
) -> set[str]:
    """Return the table's column names, lower-cased for comparison."""

    dialect = dialect_for(dialect)
    if dialect.introspection == INTROSPECTION_SQLITE:
        prefix = dialect.namespace_prefix(namespace or "main")
        table_sql = dialect.quote_identifier(table_name)
        rows = _fetchall(conn, dialect, f"PRAGMA {prefix}table_info({table_sql})")
        return {str(row[1]).lower() for row in rows}
    if dialect.introspection == INTROSPECTION_ORACLE:
        sql = "SELECT column_name FROM all_tab_columns WHERE upper(table_name) = upper(?)"
        params: tuple[Any, ...] = (table_name,)
        if namespace:
            sql += " AND upper(owner) = upper(?)"
            params += (namespace,)
        return {str(row[0]).lower() for row in _fetchall(conn, dialect, sql, params)}
    information_schema_columns, schema_namespace = _information_schema_relation(
        dialect, namespace, "columns"
    )
    sql = f"SELECT column_name FROM {information_schema_columns} WHERE lower(table_name) = lower(?)"
    params = (table_name,)
    if schema_namespace:
        sql += " AND lower(table_schema) = lower(?)"
        params += (schema_namespace,)
    elif dialect.current_schema_expr:
        # Same-named tables in other schemas would otherwise union their
        # columns into this result.  Trusted dialect constant, not user input.
        sql += f" AND lower(table_schema) = lower({dialect.current_schema_expr})"
    return {str(row[0]).lower() for row in _fetchall(conn, dialect, sql, params)}


def span_sources(
    conn: Any,
    *,
    mapping: SchemaMapping = DEFAULT_MAPPING,
    dialect: Dialect | str = SQLITE,
    namespace: str | None = None,
) -> tuple[str, ...]:
    """Detect supported span sources reachable on this connection.

    Both the member-table shape and the single-mirror-table shape are
    accepted as long as they expose the record id.
    """

    dialect = dialect_for(dialect)
    sources: list[str] = []
    if table_exists(conn, mapping.span_member_table, dialect=dialect, namespace=namespace):
        columns = table_columns(
            conn, mapping.span_member_table, dialect=dialect, namespace=namespace
        )
        if mapping.span_member_id_column.lower() in columns:
            sources.append(SOURCE_MEMBERS)
            if (
                mapping.span_group_table is not None
                and mapping.span_group_version_column is not None
                and mapping.span_group_id_column.lower() in columns
                and table_exists(
                    conn, mapping.span_group_table, dialect=dialect, namespace=namespace
                )
            ):
                group_columns = table_columns(
                    conn, mapping.span_group_table, dialect=dialect, namespace=namespace
                )
                if {
                    mapping.span_group_id_column.lower(),
                    mapping.span_group_version_column.lower(),
                } <= group_columns:
                    sources.append(SOURCE_MEMBERS_VERSIONED)
    if table_exists(conn, mapping.mirror_table, dialect=dialect, namespace=namespace):
        columns = table_columns(conn, mapping.mirror_table, dialect=dialect, namespace=namespace)
        if {mapping.mirror_id_column.lower(), mapping.mirror_reason_column.lower()} <= columns:
            sources.append(SOURCE_MIRROR_SPAN)
    return tuple(sources)


# ---------------------------------------------------------------------------
# Predicates (id-based form; no namespace required)
# ---------------------------------------------------------------------------


def flag_predicate(
    alias: str = "r",
    *,
    mapping: SchemaMapping = DEFAULT_MAPPING,
    dialect: Dialect | str = SQLITE,
) -> str:
    """Predicate for the primary store's live per-record flag."""

    dialect = dialect_for(dialect)
    return f"{dialect.qualified_column(alias, mapping.flag_column)} = 0"


def _blur_flag_predicate(
    alias: str,
    *,
    mapping: SchemaMapping,
    dialect: Dialect,
) -> str:
    """Flag clause for blur surfaces: only well-formed flags (0 or 1) match.

    ``include_blur`` must not simply drop the flag clause: a NULL flag (for
    example rows predating an ``ALTER TABLE ... ADD COLUMN``) or a corrupt
    value would then serve clear on the blur surface while every other path
    fails closed on the same row.
    """

    return f"{dialect.qualified_column(alias, mapping.flag_column)} IN (0, 1)"


def _clean_ids(record_ids: Iterable[int], *, origin: str = "") -> tuple[int, ...]:
    return tuple(sorted({coerce_record_id(record_id, origin=origin) for record_id in record_ids}))


def _chunks(values: Sequence[int], size: int) -> Iterator[tuple[int, ...]]:
    for index in range(0, len(values), size):
        yield tuple(values[index : index + size])


def not_in_span_predicate(
    alias: str = "r",
    span_member_ids: Iterable[int] | None = None,
    *,
    mapping: SchemaMapping = DEFAULT_MAPPING,
    dialect: Dialect | str = SQLITE,
) -> SqlPredicate:
    """Predicate excluding in-span rows only -- no flag screening at all.

    This is the span-only building block: it carries no flag clause, so rows
    with NULL or corrupt flag values also match it.  Serve paths should use
    :func:`normal_record_predicate` (or a guard predicate), whose
    ``include_blur`` form still screens the flag for well-formedness.

    The chunked ``NOT IN`` clauses are ANDed into ONE statement, so the whole
    cleaned id set counts against the dialect's per-statement parameter
    budget.  Exceeding it refuses up front (instead of surfacing later as an
    opaque driver error) and steers to :func:`guard_for_connection`, whose
    anti-join predicate binds no ids at all.
    """

    dialect = dialect_for(dialect)
    clauses: list[str] = []
    params: list[int] = []
    record_id = dialect.qualified_column(alias, mapping.record_id_column)
    cleaned = _clean_ids(span_member_ids or (), origin="span member ids")
    if len(cleaned) > dialect.max_statement_params:
        raise RedactionDriftError(
            f"{len(cleaned)} span-member ids exceed the {dialect.name} per-statement "
            f"parameter budget of {dialect.max_statement_params}; use "
            "guard_for_connection's anti-join predicate for span layers this large."
        )
    for chunk in _chunks(cleaned, dialect.max_in_params):
        placeholders = ", ".join("?" for _ in chunk)
        clauses.append(f"{record_id} NOT IN ({placeholders})")
        params.extend(chunk)
    if not clauses:
        # No span ids: nothing to exclude, so this matches every row.
        return SqlPredicate(_MATCH_ALL_SQL, ())
    return SqlPredicate(" AND ".join(clauses), tuple(params))


def normal_record_predicate(
    alias: str = "r",
    span_member_ids: Iterable[int] | None = None,
    *,
    include_blur: bool = False,
    mapping: SchemaMapping = DEFAULT_MAPPING,
    dialect: Dialect | str = SQLITE,
) -> SqlPredicate:
    """Reusable serve-path predicate.

    Default (``include_blur=False``) is strict ``flag = 0 AND not-in-span``
    (fully VISIBLE rows only).  With ``include_blur=True`` the flag clause
    relaxes to ``flag IN (0, 1)``, so standalone-flagged (BLUR) rows are also
    matched while in-span rows stay excluded -- and rows whose flag is NULL
    or corrupt still fail closed instead of serving clear.

    This ID-based form needs no namespace and suits smaller evidence lists.
    For broad searches/exports prefer :func:`guard_for_connection` so the
    database can anti-join the span tables directly.
    """

    dialect = dialect_for(dialect)
    span_only = not_in_span_predicate(alias, span_member_ids, mapping=mapping, dialect=dialect)
    if include_blur:
        clauses = [_blur_flag_predicate(alias, mapping=mapping, dialect=dialect)]
    else:
        clauses = [flag_predicate(alias, mapping=mapping, dialect=dialect)]
    if span_only.sql != _MATCH_ALL_SQL:
        clauses.append(span_only.sql)
    return SqlPredicate(" AND ".join(clauses), span_only.params)


# ---------------------------------------------------------------------------
# Span-store reads
# ---------------------------------------------------------------------------


def _read_span_ids(
    conn: Any,
    *,
    sources: tuple[str, ...],
    mapping: SchemaMapping,
    dialect: Dialect,
    namespace: str | None = None,
) -> set[int]:
    prefix = dialect.namespace_prefix(namespace)
    ids: set[int] = set()
    # NULL ids are skipped, not refused: a NULL can never match any record id,
    # so it cannot hide or leak anything.  Every non-NULL value goes through
    # the fail-closed coercer -- junk text in an id column is corruption and
    # must refuse rather than silently vanish from the exclusion set.
    if SOURCE_MEMBERS in sources:
        rows = _fetchall(
            conn,
            dialect,
            f"SELECT DISTINCT {dialect.quote_identifier(mapping.span_member_id_column)} "
            f"FROM {prefix}{dialect.quote_identifier(mapping.span_member_table)}",
        )
        ids.update(
            coerce_record_id(row[0], origin=mapping.span_member_table)
            for row in rows
            if row[0] is not None
        )
    if SOURCE_MIRROR_SPAN in sources:
        rows = _fetchall(
            conn,
            dialect,
            f"SELECT DISTINCT {dialect.quote_identifier(mapping.mirror_id_column)} "
            f"FROM {prefix}{dialect.quote_identifier(mapping.mirror_table)} "
            f"WHERE {dialect.quote_identifier(mapping.mirror_reason_column)} = ?",
            (mapping.reason_span,),
        )
        ids.update(
            coerce_record_id(row[0], origin=mapping.mirror_table)
            for row in rows
            if row[0] is not None
        )
    return ids


def _span_group_versions(
    conn: Any,
    *,
    sources: tuple[str, ...],
    mapping: SchemaMapping,
    dialect: Dialect,
    namespace: str | None = None,
) -> frozenset[int] | None:
    """Distinct source versions the retained span-group snapshots record.

    None unless the store carries a versioned span source, so callers can tell
    "no versioned layer here" from "versioned layer with an empty version set".
    A NULL version can never match a requested snapshot, so it is skipped; any
    non-NULL value goes through the fail-closed coercer, because junk in the
    version column is corruption in the layer that scopes span exclusions.
    """

    if (
        SOURCE_MEMBERS_VERSIONED not in sources
        or mapping.span_group_table is None
        or mapping.span_group_version_column is None
    ):
        return None
    prefix = dialect.namespace_prefix(namespace)
    rows = _fetchall(
        conn,
        dialect,
        f"SELECT DISTINCT {dialect.quote_identifier(mapping.span_group_version_column)} "
        f"FROM {prefix}{dialect.quote_identifier(mapping.span_group_table)}",
    )
    return frozenset(
        _coerce_int(row[0], what="span group version", origin=mapping.span_group_table)
        for row in rows
        if row[0] is not None
    )


def _mirror_flagged_ids(
    conn: Any,
    *,
    mapping: SchemaMapping,
    dialect: Dialect,
    namespace: str | None = None,
) -> frozenset[int] | None:
    """The mirror's claim of flagged ids, or None when the mirror can't say."""

    if not table_exists(conn, mapping.mirror_table, dialect=dialect, namespace=namespace):
        return None
    columns = table_columns(conn, mapping.mirror_table, dialect=dialect, namespace=namespace)
    required = {
        mapping.mirror_id_column.lower(),
        mapping.mirror_reason_column.lower(),
        mapping.mirror_flag_column.lower(),
    }
    if not required <= columns:
        return None
    prefix = dialect.namespace_prefix(namespace)
    rows = _fetchall(
        conn,
        dialect,
        f"SELECT {dialect.quote_identifier(mapping.mirror_id_column)} "
        f"FROM {prefix}{dialect.quote_identifier(mapping.mirror_table)} "
        f"WHERE {dialect.quote_identifier(mapping.mirror_reason_column)} = ? "
        f"AND {dialect.quote_identifier(mapping.mirror_flag_column)} = 1",
        (mapping.reason_flagged,),
    )
    return frozenset(coerce_record_id(row[0], origin=mapping.mirror_table) for row in rows)


def load_guard(
    span_conn: Any,
    *,
    mapping: SchemaMapping = DEFAULT_MAPPING,
    dialect: Dialect | str | None = None,
    origin: str = "",
) -> Guard:
    """Materialize a guard from an open connection to the span store.

    Keep ``span_conn`` open for as long as the guard is used: freshness
    re-verification reads through it.  Any failure to read the span layer is
    converted into a refusal, never a guess.

    Each freshness re-verification re-reads the full member set alongside the
    mirror (same order of work as the mirror read) so that a span
    re-derivation with unchanged flags is still caught as membership drift.
    """

    resolved = dialect_for(dialect) if dialect is not None else dialect_for(span_conn)

    def _read() -> tuple[bool, tuple[str, ...], frozenset[int] | None, frozenset[int] | None]:
        try:
            sources = span_sources(span_conn, mapping=mapping, dialect=resolved)
            mirror = _mirror_flagged_ids(span_conn, mapping=mapping, dialect=resolved)
            members = frozenset(
                _read_span_ids(span_conn, sources=sources, mapping=mapping, dialect=resolved)
            )
        except RedactionDriftError:
            raise
        except Exception as exc:  # driver-specific errors: fail closed
            raise RedactionDriftError(
                f"span store {origin or ''} is unreadable; refusing normal output."
            ) from exc
        return True, sources, mirror, members

    ready, sources, _, members = _read()
    return Guard(
        span_member_ids=members if members is not None else frozenset(),
        span_layer_ready=ready,
        span_sources=sources,
        origin=origin,
        _refresh=_read,
    )


# ---------------------------------------------------------------------------
# Primary-store reads and freshness verification
# ---------------------------------------------------------------------------


def _flagged_ids(conn: Any, *, mapping: SchemaMapping, dialect: Dialect) -> set[int]:
    rows = _fetchall(
        conn,
        dialect,
        f"SELECT {dialect.quote_identifier(mapping.record_id_column)} "
        f"FROM {dialect.quote_identifier(mapping.record_table)} "
        f"WHERE {dialect.quote_identifier(mapping.flag_column)} = 1",
    )
    return {coerce_record_id(row[0], origin=mapping.record_table) for row in rows}


def _has_flagged(conn: Any, *, mapping: SchemaMapping, dialect: Dialect) -> bool:
    row = _fetchone(
        conn,
        dialect,
        f"SELECT 1{dialect.dummy_from} WHERE EXISTS ("
        f"SELECT 1 FROM {dialect.quote_identifier(mapping.record_table)} "
        f"WHERE {dialect.quote_identifier(mapping.flag_column)} = 1)",
    )
    return row is not None


def _assert_primary_ids_well_formed(conn: Any, *, mapping: SchemaMapping, dialect: Dialect) -> None:
    """Refuse before serving unless the primary record ids are well formed.

    A missing, duplicate, or non-integer id breaks the ability to address a
    flag or a span exclusion to one record, so any of them refuses. Duplicate
    detection stays in SQL. Id coercion streams in bounded batches so a broad
    serve path does not materialize the primary store in memory.
    """

    table = dialect.quote_identifier(mapping.record_table)
    id_column = dialect.quote_identifier(mapping.record_id_column)
    null_id = _fetchone(
        conn,
        dialect,
        f"SELECT 1{dialect.dummy_from} WHERE EXISTS "
        f"(SELECT 1 FROM {table} WHERE {id_column} IS NULL)",
    )
    if null_id is not None:
        raise RedactionDriftError(
            f"a row in {mapping.record_table} has no record id; "
            "give every row a record id before serving."
        )
    counts = _fetchone(conn, dialect, f"SELECT COUNT(*), COUNT(DISTINCT {id_column}) FROM {table}")
    if counts is not None and counts[0] != counts[1]:
        raise RedactionDriftError(
            f"record id values in {mapping.record_table} are not unique; "
            "give every row a distinct record id before serving."
        )
    for (raw_id,) in _iter_rows(conn, dialect, f"SELECT {id_column} FROM {table}"):
        try:
            coerce_record_id(raw_id, origin=mapping.record_table)
        except RedactionDriftError as exc:
            raise RedactionDriftError(
                f"a record id in {mapping.record_table} is not a supported integer; "
                "correct it before serving."
            ) from exc


def _assert_primary_flags_well_formed(
    conn: Any, *, mapping: SchemaMapping, dialect: Dialect
) -> None:
    """Refuse if any primary flag is missing or outside ``{0, 1}``.

    The check is shared by every SQL serve path and the status report. It
    streams rows in bounded batches so a malformed flag refuses the operation
    without materializing the primary store in memory.
    """

    table = dialect.quote_identifier(mapping.record_table)
    id_column = dialect.quote_identifier(mapping.record_id_column)
    flag_column = dialect.quote_identifier(mapping.flag_column)
    for _raw_id, raw_flag in _iter_rows(
        conn, dialect, f"SELECT {id_column}, {flag_column} FROM {table}"
    ):
        try:
            coerce_flag(raw_flag, origin=mapping.record_table)
        except RedactionDriftError as exc:
            raise RedactionDriftError(
                f"a record in {mapping.record_table} has a flag that is not 0 or 1; "
                "correct it before serving."
            ) from exc


def _drift_witness(
    conn: Any,
    *,
    mapping: SchemaMapping,
    dialect: Dialect,
) -> tuple[int, str] | None:
    """The primary store's recorded drift witness, or None when absent.

    Absence (``meta_table=None``, no table, missing columns, no span row)
    means "nothing recorded": legacy stores and hand-rolled writers keep
    their pre-witness semantics.  A malformed recorded value, by contrast,
    is corruption in a security witness and refuses.
    """

    if mapping.meta_table is None:
        return None
    if not table_exists(conn, mapping.meta_table, dialect=dialect):
        return None
    columns = table_columns(conn, mapping.meta_table, dialect=dialect)
    if not {META_LAYER_COLUMN, META_COUNT_COLUMN, META_DIGEST_COLUMN} <= columns:
        return None
    row = _fetchone(
        conn,
        dialect,
        f"SELECT {dialect.quote_identifier(META_COUNT_COLUMN)}, "
        f"{dialect.quote_identifier(META_DIGEST_COLUMN)} "
        f"FROM {dialect.quote_identifier(mapping.meta_table)} "
        f"WHERE {dialect.quote_identifier(META_LAYER_COLUMN)} = ?",
        (META_LAYER_SPAN,),
    )
    if row is None:
        return None
    count = _coerce_int(row[0], what="span member count", origin=mapping.meta_table)
    if count < 0:
        raise RedactionDriftError(f"span member count is negative at {mapping.meta_table}: {count}")
    digest = row[1]
    if not isinstance(digest, str) or not digest:
        raise RedactionDriftError(f"span member digest is malformed at {mapping.meta_table}")
    return count, digest


def _assert_drift_witness_satisfied(
    conn: Any,
    *,
    mapping: SchemaMapping,
    dialect: Dialect,
    namespace: str | None,
    sources: tuple[str, ...],
    origin: str = "",
) -> None:
    """Namespace-path witness verification against the recorded member digest."""

    witness = _drift_witness(conn, mapping=mapping, dialect=dialect)
    if witness is None:
        return
    if not sources:
        verify_drift_witness(witness, span_layer_ready=False, origin=origin)
        return
    members = _read_span_ids(
        conn, sources=sources, mapping=mapping, dialect=dialect, namespace=namespace
    )
    verify_drift_witness(witness, span_layer_ready=True, span_member_ids=members, origin=origin)


def assert_guard_current(
    conn: Any,
    guard: Guard,
    *,
    mapping: SchemaMapping = DEFAULT_MAPPING,
    dialect: Dialect | str | None = None,
) -> None:
    """Fail closed if the loaded guard cannot represent the live flag layer.

    A flag edit changes the primary store immediately, while the required
    span-derivation rebuild updates the mirrored flag layer and contiguous
    spans.  Until that rebuild happens, serving normal rows is unsafe: a
    newly recognized span can contain neutral records that no stale member
    table knows to hide.

    When the primary store carries a drift witness, the guard's span
    membership is also verified against it (by digest), so a span store that
    went missing or was emptied refuses even with zero flagged records.
    """

    with _fail_closed(guard.origin or "primary store"):
        resolved = dialect_for(dialect) if dialect is not None else dialect_for(conn)
        _assert_primary_ids_well_formed(conn, mapping=mapping, dialect=resolved)
        _assert_primary_flags_well_formed(conn, mapping=mapping, dialect=resolved)
        live = _flagged_ids(conn, mapping=mapping, dialect=resolved)
        verify_guard_current(live, guard)
        witness = _drift_witness(conn, mapping=mapping, dialect=resolved)
        verify_drift_witness(
            witness,
            span_layer_ready=guard.span_layer_ready,
            span_member_ids=guard.span_member_ids if guard.span_layer_ready else None,
            origin=guard.origin,
        )


# ---------------------------------------------------------------------------
# Single-connection (namespace) chokepoint
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttachedGuard:
    """Predicate builder for a connection that can reach both layers.

    ``namespace`` is the schema/database qualifier under which the span
    tables are addressable (or None when the span layer is absent).
    """

    namespace: str | None
    span_sources: tuple[str, ...]
    span_layer_ready: bool
    mapping: SchemaMapping = DEFAULT_MAPPING
    dialect: Dialect = SQLITE
    # Distinct source versions the retained span-group snapshots record, or
    # None when the store has no versioned span source.  A version_id request
    # is checked against this so a snapshot the layer never derived refuses
    # instead of anti-joining nothing and serving every span member clear.
    span_versions: frozenset[int] | None = None

    def predicate(
        self,
        alias: str = "r",
        *,
        include_blur: bool = False,
        version_id: int | None = None,
    ) -> SqlPredicate:
        """WHERE fragment anti-joining the span tables when present.

        Default is strict ``flag = 0 AND not-in-span``.  With
        ``include_blur=True`` the flag clause relaxes to ``flag IN (0, 1)``,
        so the predicate keeps standalone-flagged (BLUR) rows while still
        excluding every in-span row via the span tables -- and a NULL or
        corrupt flag stays excluded instead of serving clear.

        A ``version_id`` scopes the span anti-join to one retained snapshot.
        It must name a version the span layer actually derived; a phantom id
        would make the anti-join match nothing and serve every span member
        clear, so an unrecognized (or unversioned) request refuses.
        """

        mapping = self.mapping
        dialect = self.dialect
        clauses: list[str] = []
        params: list[Any] = []
        if version_id is not None:
            version_id = _coerce_int(version_id, what="version id", origin="")
            versioned = (
                self.span_layer_ready
                and SOURCE_MEMBERS_VERSIONED in self.span_sources
                and mapping.span_group_table is not None
                and mapping.span_group_version_column is not None
            )
            if not versioned:
                raise RedactionDriftError(
                    f"version_id {version_id} was requested but this span store has no "
                    "versioned span layer; serving with it would drop every span "
                    "exclusion; rebuild the span derivation layer before serving."
                )
            if self.span_versions is None or version_id not in self.span_versions:
                raise RedactionDriftError(
                    f"the span store records no derivation for version {version_id}; "
                    "serving with it would drop every span exclusion; rebuild the span "
                    "derivation layer before serving."
                )
        if include_blur:
            clauses.append(_blur_flag_predicate(alias, mapping=mapping, dialect=dialect))
        else:
            clauses.append(flag_predicate(alias, mapping=mapping, dialect=dialect))
        record_id = dialect.qualified_column(alias, mapping.record_id_column)
        prefix = dialect.namespace_prefix(self.namespace)

        if self.span_layer_ready and SOURCE_MEMBERS in self.span_sources:
            member_table = f"{prefix}{dialect.quote_identifier(mapping.span_member_table)}"
            member_id = dialect.quote_identifier(mapping.span_member_id_column)
            span_scope = f"WHERE sg_member.{member_id} = {record_id}"
            if (
                version_id is not None
                and SOURCE_MEMBERS_VERSIONED in self.span_sources
                and mapping.span_group_table is not None
                and mapping.span_group_version_column is not None
            ):
                group_table = f"{prefix}{dialect.quote_identifier(mapping.span_group_table)}"
                group_id = dialect.quote_identifier(mapping.span_group_id_column)
                version_column = dialect.quote_identifier(mapping.span_group_version_column)
                span_scope = (
                    f"JOIN {group_table} sg_group "
                    f"ON sg_group.{group_id} = sg_member.{group_id} "
                    f"WHERE sg_member.{member_id} = {record_id} "
                    f"AND sg_group.{version_column} = ?"
                )
                params.append(version_id)
            clauses.append(f"NOT EXISTS (SELECT 1 FROM {member_table} sg_member {span_scope})")
        # The mirror's span rows are a union across snapshots.  When a
        # version_id scopes the request, the members+group join above is
        # authoritative, and applying the mirror rows as a second anti-join
        # would over-hide rows that entered a span only in a later version --
        # so the mirror anti-join is skipped for versioned requests.  For an
        # unversioned request the mirror-span set must still be excluded even
        # when a members table is present: the materialized guard unions both
        # sources (see _read_span_ids), so a store whose mirror-span rows are a
        # superset of its members table -- an inconsistent or externally built
        # span store -- would otherwise leak here on the anti-join path while
        # the materialized path hides them.  Both serve paths must exclude the
        # same set.
        if (
            self.span_layer_ready
            and SOURCE_MIRROR_SPAN in self.span_sources
            and (SOURCE_MEMBERS not in self.span_sources or version_id is None)
        ):
            mirror_table = f"{prefix}{dialect.quote_identifier(mapping.mirror_table)}"
            mirror_id = dialect.quote_identifier(mapping.mirror_id_column)
            reason = dialect.quote_identifier(mapping.mirror_reason_column)
            clauses.append(
                f"NOT EXISTS (SELECT 1 FROM {mirror_table} sg_mirror "
                f"WHERE sg_mirror.{mirror_id} = {record_id} "
                f"AND sg_mirror.{reason} = ?)"
            )
            params.append(mapping.reason_span)
        return SqlPredicate(" AND ".join(clauses), tuple(params))


def _assert_attached_flag_layer_current(
    conn: Any,
    *,
    mapping: SchemaMapping,
    dialect: Dialect,
    namespace: str | None,
    sources: tuple[str, ...],
) -> None:
    """Namespace equivalent of :func:`assert_guard_current`.

    Uses anti-joins instead of materializing every id, which keeps broad
    display/search/export routes safe and scalable on a full dataset.
    """

    if not table_exists(conn, mapping.mirror_table, dialect=dialect, namespace=namespace):
        # Span-only stores still safely hide every recorded span plus live
        # flags; they simply cannot offer mirror-freshness verification.
        if _has_flagged(conn, mapping=mapping, dialect=dialect) and not sources:
            raise RedactionDriftError(
                "redaction drift risk: the flagged-record mirror is unavailable; "
                "rebuild the span derivation layer before serving."
            )
        return
    mirror_columns = table_columns(conn, mapping.mirror_table, dialect=dialect, namespace=namespace)
    required = {
        mapping.mirror_id_column.lower(),
        mapping.mirror_reason_column.lower(),
        mapping.mirror_flag_column.lower(),
    }
    if not required <= mirror_columns:
        if _has_flagged(conn, mapping=mapping, dialect=dialect) and not sources:
            raise RedactionDriftError(
                "redaction drift risk: the flagged-record mirror is unavailable; "
                "rebuild the span derivation layer before serving."
            )
        return
    if _has_flagged(conn, mapping=mapping, dialect=dialect) and not sources:
        raise RedactionDriftError(
            "redaction drift risk: the span-layer tables are unavailable; rebuild "
            "the span derivation layer before serving."
        )
    prefix = dialect.namespace_prefix(namespace)
    records = dialect.quote_identifier(mapping.record_table)
    record_id = dialect.quote_identifier(mapping.record_id_column)
    flag = dialect.quote_identifier(mapping.flag_column)
    mirror = f"{prefix}{dialect.quote_identifier(mapping.mirror_table)}"
    mirror_id = dialect.quote_identifier(mapping.mirror_id_column)
    reason = dialect.quote_identifier(mapping.mirror_reason_column)
    mirror_flag = dialect.quote_identifier(mapping.mirror_flag_column)
    mismatch = _fetchone(
        conn,
        dialect,
        f"SELECT 1{dialect.dummy_from} "
        "WHERE EXISTS ("
        f"  SELECT 1 FROM {records} r WHERE r.{flag} = 1 "
        f"  AND NOT EXISTS (SELECT 1 FROM {mirror} sg_mirror "
        f"                  WHERE sg_mirror.{mirror_id} = r.{record_id} "
        f"                    AND sg_mirror.{reason} = ? AND sg_mirror.{mirror_flag} = 1)"
        ") OR EXISTS ("
        f"  SELECT 1 FROM {mirror} sg_mirror "
        f"  WHERE sg_mirror.{reason} = ? AND (sg_mirror.{mirror_flag} <> 1 OR NOT EXISTS ("
        f"    SELECT 1 FROM {records} r WHERE r.{record_id} = sg_mirror.{mirror_id} "
        f"    AND r.{flag} = 1"
        "  ))"
        ")",
        (mapping.reason_flagged, mapping.reason_flagged),
    )
    if mismatch is not None:
        raise RedactionDriftError(
            "redaction drift: span isolation is stale relative to the live flag "
            "layer; rebuild the span derivation layer before serving."
        )


def guard_for_connection(
    conn: Any,
    *,
    mapping: SchemaMapping = DEFAULT_MAPPING,
    dialect: Dialect | str | None = None,
    namespace: str | None = None,
    span_layer_ready: bool = True,
) -> AttachedGuard:
    """Build the anti-join guard for a connection reaching both layers.

    ``namespace`` defaults to ``mapping.span_namespace``.  Freshness of the
    mirrored flag layer is verified here, before any predicate is handed
    out; drift raises :class:`RedactionDriftError` instead of serving.  If the
    primary store carries a drift witness, the span layer's member digest
    is verified against it too, so a swapped, emptied, or missing span layer
    refuses even with zero flagged records.

    When the span layer genuinely does not exist yet, pass
    ``span_layer_ready=False``: the guard then only checks the live flag,
    and this function refuses if flagged records already exist or a
    drift witness is recorded.
    """

    with _fail_closed("this connection"):
        resolved = dialect_for(dialect) if dialect is not None else dialect_for(conn)
        resolved_namespace = namespace if namespace is not None else mapping.span_namespace
        _assert_primary_ids_well_formed(conn, mapping=mapping, dialect=resolved)
        _assert_primary_flags_well_formed(conn, mapping=mapping, dialect=resolved)
        if not span_layer_ready:
            if _has_flagged(conn, mapping=mapping, dialect=resolved):
                raise RedactionDriftError(
                    "redaction drift risk: span isolation is unavailable while the "
                    "primary store has flagged records; rebuild the span derivation "
                    "layer before serving."
                )
            # A recorded derivation contradicts "the span layer does not exist
            # yet" -- refuse rather than silently un-hide every in-span record.
            _assert_drift_witness_satisfied(
                conn,
                mapping=mapping,
                dialect=resolved,
                namespace=None,
                sources=(),
                origin="this connection",
            )
            return AttachedGuard(
                namespace=None,
                span_sources=(),
                span_layer_ready=False,
                mapping=mapping,
                dialect=resolved,
            )
        sources = span_sources(
            conn, mapping=mapping, dialect=resolved, namespace=resolved_namespace
        )
        _assert_attached_flag_layer_current(
            conn, mapping=mapping, dialect=resolved, namespace=resolved_namespace, sources=sources
        )
        _assert_drift_witness_satisfied(
            conn,
            mapping=mapping,
            dialect=resolved,
            namespace=resolved_namespace,
            sources=sources,
            origin="this connection",
        )
        return AttachedGuard(
            namespace=resolved_namespace,
            span_sources=sources,
            span_layer_ready=True,
            mapping=mapping,
            dialect=resolved,
            span_versions=_span_group_versions(
                conn,
                sources=sources,
                mapping=mapping,
                dialect=resolved,
                namespace=resolved_namespace,
            ),
        )


# ---------------------------------------------------------------------------
# ID and row filtering
# ---------------------------------------------------------------------------


def filter_visible_ids(
    conn: Any,
    record_ids: Sequence[int],
    *,
    guard: Guard,
    mapping: SchemaMapping = DEFAULT_MAPPING,
    dialect: Dialect | str | None = None,
) -> list[int]:
    """Return the input ids that are VISIBLE on normal surfaces.

    Input order and duplicates are preserved.  Nonexistent ids are omitted.
    Freshness is re-verified first; drift refuses the whole call.

    Each chunk statement only binds the id-chunk placeholders (never the full
    span-member id set), so this stays under the dialect's parameter limit
    regardless of how large the span layer is; visibility is classified in
    Python instead of pushing the span exclusion into the SQL predicate.

    Because it is a materialized (in-Python) classification, this path coerces
    the flag of every requested row and so **refuses the whole call** if any
    requested id has a corrupt flag (NULL is treated per-row as not VISIBLE and
    excluded, but an out-of-domain integer or junk value raises
    :class:`RedactionDriftError`).  This is stricter than the SQL predicate
    paths (:func:`normal_record_predicate`, :meth:`AttachedGuard.predicate`),
    which exclude a single corrupt-flag row and keep serving the rest: an
    id-list caller asks a direct question about a bounded set, so a corrupt
    flag inside that set is answered by refusing, never by silently dropping
    the bad row from the answer.  Pass only ids you are prepared to have refuse
    together, or use a predicate serve path for large scans.
    """

    with _fail_closed(guard.origin or "primary store"):
        resolved = dialect_for(dialect) if dialect is not None else dialect_for(conn)
        cleaned = [
            coerce_record_id(record_id, origin="record_ids argument") for record_id in record_ids
        ]
        if not cleaned:
            return []
        assert_guard_current(conn, guard, mapping=mapping, dialect=resolved)
        unique_ids = list(dict.fromkeys(cleaned))
        visible: set[int] = set()
        record_id_sql = resolved.qualified_column("r", mapping.record_id_column)
        flag_sql = resolved.qualified_column("r", mapping.flag_column)
        table_sql = resolved.quote_identifier(mapping.record_table)
        for chunk in _chunks(unique_ids, resolved.max_in_params):
            placeholders = ", ".join("?" for _ in chunk)
            rows = _fetchall(
                conn,
                resolved,
                f"SELECT {record_id_sql}, {flag_sql} FROM {table_sql} r "
                f"WHERE {record_id_sql} IN ({placeholders})",
                chunk,
            )
            for record_id, flag in rows:
                if flag is None:
                    # NULL flag: fail closed, matching the predicate's flag = 0 clause.
                    continue
                record_id = coerce_record_id(record_id, origin=mapping.record_table)
                if (
                    coerce_flag(flag, origin=mapping.record_table) == 0
                    and record_id not in guard.span_member_ids
                ):
                    visible.add(record_id)
        return [record_id for record_id in cleaned if record_id in visible]


def is_record_visible(
    conn: Any,
    record_id: int,
    *,
    guard: Guard,
    mapping: SchemaMapping = DEFAULT_MAPPING,
    dialect: Dialect | str | None = None,
) -> bool:
    """True only when this record may appear on normal surfaces."""

    return bool(
        filter_visible_ids(conn, [record_id], guard=guard, mapping=mapping, dialect=dialect)
    )


def _count_ids_in_primary(
    conn: Any,
    record_ids: Iterable[int],
    *,
    mapping: SchemaMapping,
    dialect: Dialect,
    extra_where: str | None = None,
) -> int:
    ids = _clean_ids(record_ids, origin="span member ids")
    if not ids:
        return 0
    total = 0
    table_sql = dialect.quote_identifier(mapping.record_table)
    id_sql = dialect.quote_identifier(mapping.record_id_column)
    suffix = f" AND {extra_where}" if extra_where else ""
    for chunk in _chunks(ids, dialect.max_in_params):
        placeholders = ", ".join("?" for _ in chunk)
        row = _fetchone(
            conn,
            dialect,
            f"SELECT COUNT(*) FROM {table_sql} WHERE {id_sql} IN ({placeholders}){suffix}",
            chunk,
        )
        total += int(row[0])
    return total


def status(
    conn: Any,
    guard: Guard,
    *,
    mapping: SchemaMapping = DEFAULT_MAPPING,
    dialect: Dialect | str | None = None,
) -> dict[str, Any]:
    """Counts-only readiness report.  Reads only; emits no record content.

    Every statement here binds at most one ``max_in_params``-sized id chunk,
    never the whole span-member set: the visible/blur counts are derived by
    subtracting chunked in-span counts from whole-table flag counts, so this
    stays usable at production span-layer sizes where an id-list predicate
    would exceed the dialect's per-statement parameter budget.
    """

    with _fail_closed(guard.origin or "primary store"):
        resolved = dialect_for(dialect) if dialect is not None else dialect_for(conn)
        _assert_primary_ids_well_formed(conn, mapping=mapping, dialect=resolved)
        _assert_primary_flags_well_formed(conn, mapping=mapping, dialect=resolved)
        table_sql = resolved.quote_identifier(mapping.record_table)
        flag_sql = resolved.quote_identifier(mapping.flag_column)
        total = int(_fetchone(conn, resolved, f"SELECT COUNT(*) FROM {table_sql}")[0])
        flagged = int(
            _fetchone(conn, resolved, f"SELECT COUNT(*) FROM {table_sql} WHERE {flag_sql} = 1")[0]
        )
        flag0_total = int(
            _fetchone(conn, resolved, f"SELECT COUNT(*) FROM {table_sql} WHERE {flag_sql} = 0")[0]
        )
        wellformed_total = int(
            _fetchone(
                conn, resolved, f"SELECT COUNT(*) FROM {table_sql} WHERE {flag_sql} IN (0, 1)"
            )[0]
        )
        span_in_primary = _count_ids_in_primary(
            conn, guard.span_member_ids, mapping=mapping, dialect=resolved
        )
        span_only_hidden = _count_ids_in_primary(
            conn,
            guard.span_member_ids,
            mapping=mapping,
            dialect=resolved,
            extra_where=f"{flag_sql} = 0",
        )
        span_wellformed_hidden = _count_ids_in_primary(
            conn,
            guard.span_member_ids,
            mapping=mapping,
            dialect=resolved,
            extra_where=f"{flag_sql} IN (0, 1)",
        )
        visible = flag0_total - span_only_hidden
        visible_or_blur = wellformed_total - span_wellformed_hidden
        witness = _drift_witness(conn, mapping=mapping, dialect=resolved)
        report = {
            "records": total,
            "flagged": flagged,
            "visible_normal_records": visible,
            # Standalone flagged = flagged AND not in-span = the BLUR state.
            "blurred_standalone": visible_or_blur - visible,
            # HIDDEN = in-span (regardless of the record's own flag).
            "hidden_in_span": span_in_primary,
            "deliverable_with_blur": visible_or_blur,
            "excluded_total": total - visible,
            "excluded_by_flag": flagged,
            "span_ids_present_in_primary": span_in_primary,
            "excluded_by_span_only": span_only_hidden,
            "span_layer_ready": guard.span_layer_ready,
            "span_sources": list(guard.span_sources),
            "unique_span_member_ids": len(guard.span_member_ids),
            # Counts only: the digest is withheld here; sync (including the
            # digest comparison) is reported via layers_in_sync/refusal.
            "drift_witness": ({"member_count": witness[0]} if witness is not None else None),
            "predicate": f"{mapping.flag_column} = 0 AND {mapping.record_id_column} not in spans",
            "predicate_include_blur": (
                f"{mapping.flag_column} in (0, 1) AND {mapping.record_id_column} not in spans"
            ),
        }
        try:
            assert_guard_current(conn, guard, mapping=mapping, dialect=resolved)
        except RedactionDriftError as exc:
            report["layers_in_sync"] = False
            report["refusal"] = str(exc)
        else:
            report["layers_in_sync"] = True
        return report
