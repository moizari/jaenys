"""Golden-rendering tests for ``jaenys.sql.dialects``.

Covers identifier quoting per dialect, positional-to-dialect param
rendering, name resolution (including aliases and connection
auto-detection), the ``generic_ansi`` factory, Oracle's dummy-FROM/named
paramstyle and identifier case folding, introspection schema scoping, and
one end-to-end predicate render.
"""

from __future__ import annotations

import dataclasses
import sqlite3
from typing import Any

import pytest

from jaenys import SchemaMapping, RedactionDriftError
from jaenys.sql import MYSQL, normal_record_predicate
from jaenys.sql.dialects import (
    GENERIC_ANSI,
    MSSQL,
    ORACLE,
    POSTGRESQL,
    SQLITE,
    Dialect,
    dialect_for,
    generic_ansi,
)
from jaenys.sql.guard import table_columns, table_exists

ALL_BUILTIN_DIALECTS = [SQLITE, POSTGRESQL, MYSQL, MSSQL, ORACLE, GENERIC_ANSI]

EVIL_IDENTIFIER = 'x"; DROP TABLE r --'


# ---------------------------------------------------------------------------
# Identifier quoting
# ---------------------------------------------------------------------------


def test_sqlite_postgresql_generic_ansi_use_double_quotes() -> None:
    for dialect in (SQLITE, POSTGRESQL, GENERIC_ANSI):
        assert dialect.quote_identifier("col") == '"col"'


def test_mysql_uses_backticks() -> None:
    assert MYSQL.quote_identifier("col") == "`col`"


def test_mssql_uses_square_brackets() -> None:
    assert MSSQL.quote_identifier("col") == "[col]"


@pytest.mark.parametrize("dialect", ALL_BUILTIN_DIALECTS, ids=lambda d: d.name)
def test_quote_identifier_rejects_evil_identifiers(dialect: Dialect) -> None:
    with pytest.raises(RedactionDriftError):
        dialect.quote_identifier(EVIL_IDENTIFIER)


# ---------------------------------------------------------------------------
# render()
# ---------------------------------------------------------------------------


def test_render_qmark_is_passthrough() -> None:
    sql, params = SQLITE.render("a = ? AND b = ?", (1, 2))
    assert sql == "a = ? AND b = ?"
    assert params == (1, 2)


def test_render_format_gives_percent_s_with_tuple() -> None:
    sql, params = POSTGRESQL.render("a = ? AND b = ?", (1, 2))
    assert sql == "a = %s AND b = %s"
    assert params == (1, 2)


def test_render_pyformat_gives_named_percent_dict() -> None:
    dialect = generic_ansi("pyformat")
    sql, params = dialect.render("a = ? AND b = ?", (1, 2))
    assert sql == "a = %(p0)s AND b = %(p1)s"
    assert params == {"p0": 1, "p1": 2}


def test_render_named_gives_colon_p_dict() -> None:
    dialect = generic_ansi("named")
    sql, params = dialect.render("a = ? AND b = ?", (1, 2))
    assert sql == "a = :p0 AND b = :p1"
    assert params == {"p0": 1, "p1": 2}


def test_render_numeric_gives_colon_index_tuple() -> None:
    dialect = generic_ansi("numeric")
    sql, params = dialect.render("a = ? AND b = ?", (1, 2))
    assert sql == "a = :1 AND b = :2"
    assert params == (1, 2)


def test_render_raises_on_placeholder_param_count_mismatch() -> None:
    with pytest.raises(RedactionDriftError):
        SQLITE.render("a = ? AND b = ?", (1,))


# ---------------------------------------------------------------------------
# dialect_for()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("postgres", POSTGRESQL),
        ("cockroachdb", POSTGRESQL),
        ("redshift", POSTGRESQL),
        ("mariadb", MYSQL),
        ("tidb", MYSQL),
        ("sqlserver", MSSQL),
    ],
)
def test_dialect_for_resolves_aliases(name: str, expected: Dialect) -> None:
    assert dialect_for(name) is expected


def test_dialect_for_unknown_name_raises() -> None:
    with pytest.raises(RedactionDriftError):
        dialect_for("not-a-real-dialect")


def test_dialect_for_sqlite_connection_auto_detects() -> None:
    conn = sqlite3.connect(":memory:")
    try:
        assert dialect_for(conn) is SQLITE
    finally:
        conn.close()


def test_dialect_for_dialect_instance_passes_through() -> None:
    assert dialect_for(MYSQL) is MYSQL


def _fake_connection(module_name: str) -> Any:
    return type("Connection", (), {"__module__": module_name})()


def test_dialect_for_pymssql_uses_pyformat_not_qmark() -> None:
    """pymssql is SQL Server but binds %s / %(name)s and rejects ?."""
    dialect = dialect_for(_fake_connection("pymssql"))
    assert dialect.name == "mssql"
    assert dialect.paramstyle == "pyformat"
    assert dialect.paramstyle != "qmark"  # pymssql rejects ?


def test_dialect_for_mariadb_uses_qmark() -> None:
    """MariaDB Connector/Python defaults to qmark, unlike other MySQL drivers."""
    dialect = dialect_for(_fake_connection("mariadb"))
    assert dialect.paramstyle == "qmark"  # MariaDB Connector/Python default
    assert dialect.quote_identifier("col") == "`col`"  # only the paramstyle changed


def test_dialect_for_pyodbc_still_qmark() -> None:
    """pyodbc is untouched, so the pymssql fix is targeted."""
    assert dialect_for(_fake_connection("pyodbc")).paramstyle == "qmark"


# ---------------------------------------------------------------------------
# generic_ansi()
# ---------------------------------------------------------------------------


def test_generic_ansi_returns_dialect_with_requested_paramstyle() -> None:
    dialect = generic_ansi("pyformat")
    assert isinstance(dialect, Dialect)
    assert dialect.paramstyle == "pyformat"


def test_generic_ansi_invalid_paramstyle_raises() -> None:
    with pytest.raises(RedactionDriftError):
        generic_ansi("not-a-paramstyle")


# ---------------------------------------------------------------------------
# Oracle specifics
# ---------------------------------------------------------------------------


def test_oracle_has_dummy_from_and_named_paramstyle() -> None:
    assert ORACLE.dummy_from == " FROM dual"
    assert ORACLE.paramstyle == "named"


def test_mysql_has_dummy_from() -> None:
    # MySQL before 8.0.19 and every MariaDB reject a table-less WHERE.
    assert MYSQL.dummy_from == " FROM DUAL"


def test_oracle_upper_folds_identifiers_by_default() -> None:
    # Unquoted Oracle DDL stores names uppercase, so quoting the mapping's
    # lowercase spelling verbatim would miss the table (ORA-00942).
    assert ORACLE.quote_identifier("span_members") == '"SPAN_MEMBERS"'
    assert ORACLE.qualified_column("r", "record_id") == '"R"."RECORD_ID"'
    assert ORACLE.namespace_prefix("span_layer") == '"SPAN_LAYER".'


def test_oracle_preserve_escape_hatch_for_quoted_lowercase_schemas() -> None:
    preserve = dataclasses.replace(ORACLE, identifier_case="preserve")
    assert preserve.quote_identifier("span_members") == '"span_members"'


def test_non_oracle_dialects_preserve_identifier_case() -> None:
    for dialect in (SQLITE, POSTGRESQL, MYSQL, MSSQL, GENERIC_ANSI):
        assert dialect.identifier_case == "preserve"
        assert "Span_Members" in dialect.quote_identifier("Span_Members")


def test_identifier_case_folding_runs_after_validation() -> None:
    with pytest.raises(RedactionDriftError, match="unsafe"):
        ORACLE.quote_identifier(EVIL_IDENTIFIER)


# ---------------------------------------------------------------------------
# Field validation
# ---------------------------------------------------------------------------


def test_identifier_case_is_validated() -> None:
    with pytest.raises(RedactionDriftError, match="identifier_case"):
        Dialect(name="x", identifier_case="title")


def test_max_statement_params_is_validated() -> None:
    with pytest.raises(RedactionDriftError, match="max_statement_params"):
        Dialect(name="x", max_statement_params=0)


def test_builtin_statement_budgets_match_engine_limits() -> None:
    assert MSSQL.max_statement_params == 2000  # engine caps at ~2100
    assert SQLITE.max_statement_params == 32000
    assert GENERIC_ANSI.max_statement_params == 2000  # conservative fallback


# ---------------------------------------------------------------------------
# Introspection schema scoping (current_schema_expr)
# ---------------------------------------------------------------------------


class RecordingConnection:
    """Minimal PEP 249 stand-in that records executed SQL and returns no rows."""

    def __init__(self) -> None:
        self.executed: list[tuple[str, Any]] = []

    def cursor(self) -> Any:
        conn = self

        class _Cursor:
            def execute(self, sql: str, params: Any = ()) -> None:
                conn.executed.append((sql, params))

            def fetchone(self) -> None:
                return None

            def fetchall(self) -> list:
                return []

            def close(self) -> None:
                pass

        return _Cursor()


def test_unqualified_information_schema_lookup_scopes_to_current_schema() -> None:
    conn = RecordingConnection()
    table_exists(conn, "records", dialect=POSTGRESQL)
    sql, params = conn.executed[0]
    assert "lower(table_schema) = lower(current_schema())" in sql
    assert params == ("records",)

    conn = RecordingConnection()
    table_columns(conn, "records", dialect=MYSQL)
    sql, _ = conn.executed[0]
    assert "lower(table_schema) = lower(DATABASE())" in sql


def test_explicit_namespace_overrides_current_schema_scoping() -> None:
    conn = RecordingConnection()
    table_exists(conn, "records", dialect=POSTGRESQL, namespace="span_layer")
    sql, params = conn.executed[0]
    assert "current_schema()" not in sql
    assert "lower(table_schema) = lower(%s)" in sql
    assert params == ("records", "span_layer")


def test_mssql_database_schema_namespace_uses_cross_database_introspection() -> None:
    conn = RecordingConnection()
    table_exists(conn, "records", dialect=MSSQL, namespace="span_db.guard_schema")
    sql, params = conn.executed[0]
    assert "[span_db].information_schema.tables" in sql
    assert params == ("records", "guard_schema")
    assert MSSQL.namespace_prefix("span_db.guard_schema") == "[span_db].[guard_schema]."


def test_sqlite_introspection_is_unchanged_by_schema_scoping() -> None:
    conn = RecordingConnection()
    table_exists(conn, "records", dialect=SQLITE)
    sql, _ = conn.executed[0]
    assert "sqlite_master" in sql
    assert "table_schema" not in sql


# ---------------------------------------------------------------------------
# End-to-end golden SQL
# ---------------------------------------------------------------------------


def test_normal_record_predicate_mysql_golden_sql() -> None:
    predicate = normal_record_predicate("r", [1, 2], mapping=SchemaMapping(), dialect=MYSQL)

    assert "`r`.`sensitive`" in predicate.sql
    assert "`r`.`record_id`" in predicate.sql

    sql, params = predicate.render(MYSQL)
    assert "%s" in sql
    assert "?" not in sql
    assert params == (1, 2)
