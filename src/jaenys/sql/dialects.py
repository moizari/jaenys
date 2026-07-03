"""SQL dialect descriptions for the span-visibility engine.

The engine builds every statement internally with ``qmark`` (``?``)
placeholders and positional parameters, then renders them into the target
dialect right before execution.  A :class:`Dialect` is a small frozen value
object -- adding support for another SQL engine usually means constructing one
of these (~10 lines), not writing code.

The library never imports database drivers; callers open their own PEP 249
connection (psycopg, PyMySQL, pyodbc, oracledb, sqlite3, ...) and pass it in.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

from ..core import RedactionDriftError, validate_name

__all__ = [
    "Dialect",
    "SQLITE",
    "POSTGRESQL",
    "MYSQL",
    "MSSQL",
    "ORACLE",
    "GENERIC_ANSI",
    "DIALECTS",
    "generic_ansi",
    "dialect_for",
]

_PARAMSTYLES = {"qmark", "format", "pyformat", "numeric", "named"}
_IDENTIFIER_CASES = {"preserve", "upper", "lower"}

# Introspection strategies (how table/column existence is discovered).
INTROSPECTION_SQLITE = "sqlite"  # sqlite_master + PRAGMA table_info
INTROSPECTION_INFORMATION_SCHEMA = "information_schema"
INTROSPECTION_ORACLE = "oracle"  # all_tables / all_tab_columns


@dataclass(frozen=True)
class Dialect:
    """Everything engine-specific the SQL backend needs to know."""

    name: str
    paramstyle: str = "qmark"
    quote_open: str = '"'
    quote_close: str = '"'
    introspection: str = INTROSPECTION_INFORMATION_SCHEMA
    max_in_params: int = 500
    # FROM clause required for table-less SELECTs (Oracle: " FROM dual").
    dummy_from: str = ""
    # Total bound-parameter budget for ONE statement.  Chunked NOT IN clauses
    # are ANDed into a single statement, so the whole span-member set counts
    # against this; exceeding an engine's real cap (MSSQL ~2100, old SQLite
    # 999/32766) surfaces as an opaque driver error, so the engine refuses
    # first with a message that names the fix.  Conservative default for
    # hand-built dialects; built-ins override per engine.
    max_statement_params: int = 2000
    # How quoted identifiers are case-folded before quoting.  Oracle folds
    # unquoted DDL to uppercase, so a conventionally created table named
    # span_members is stored as SPAN_MEMBERS and quoted-lowercase queries
    # miss it (ORA-00942).  "preserve" leaves names exactly as mapped.
    identifier_case: str = "preserve"
    # Trusted SQL expression returning the connection's current schema, used
    # to scope information_schema introspection when no namespace is given
    # (multi-schema databases can hold same-named tables in other schemas).
    # None disables the extra filter (SQLite/Oracle branches don't use it).
    current_schema_expr: str | None = None

    def __post_init__(self) -> None:
        if self.paramstyle not in _PARAMSTYLES:
            raise RedactionDriftError(
                f"unknown paramstyle {self.paramstyle!r}; expected one of {sorted(_PARAMSTYLES)}"
            )
        if self.max_in_params < 1:
            raise RedactionDriftError("max_in_params must be at least 1")
        if self.max_statement_params < 1:
            raise RedactionDriftError("max_statement_params must be at least 1")
        if self.identifier_case not in _IDENTIFIER_CASES:
            raise RedactionDriftError(
                f"unknown identifier_case {self.identifier_case!r}; "
                f"expected one of {sorted(_IDENTIFIER_CASES)}"
            )

    # -- identifiers --------------------------------------------------------

    def quote_identifier(self, identifier: str) -> str:
        """Validate (allowlist) and quote a simple identifier.

        Case folding (``identifier_case``) is applied AFTER validation, so the
        allowlist always sees the caller's original spelling.
        """

        validate_name(identifier, kind="SQL identifier")
        if self.identifier_case == "upper":
            identifier = identifier.upper()
        elif self.identifier_case == "lower":
            identifier = identifier.lower()
        return f"{self.quote_open}{identifier}{self.quote_close}"

    def qualified_column(self, alias: str, column: str) -> str:
        """Return ``alias.column`` with validated identifiers."""

        quoted = self.quote_identifier(column)
        if not alias:
            return quoted
        return f"{self.quote_identifier(alias)}.{quoted}"

    def namespace_prefix(self, namespace: str | None) -> str:
        """Return ``"ns".`` (quoted, dot-suffixed) or ``""`` when unqualified."""

        if not namespace:
            return ""
        return f"{self.quote_identifier(namespace)}."

    # -- parameter rendering -------------------------------------------------

    def render(self, sql: str, params: Sequence[Any]) -> tuple[str, Any]:
        """Render qmark SQL + positional params into this dialect's style.

        The engine never embeds literals, so every ``?`` in ``sql`` is a
        placeholder by construction.
        """

        expected = sql.count("?")
        if expected != len(params):
            raise RedactionDriftError(
                f"placeholder/parameter mismatch: {expected} placeholders, {len(params)} params"
            )
        if self.paramstyle == "qmark":
            return sql, tuple(params)
        if self.paramstyle == "format":
            return sql.replace("?", "%s"), tuple(params)

        parts = sql.split("?")
        if self.paramstyle == "numeric":
            rendered = "".join(
                part + (f":{index + 1}" if index < len(params) else "")
                for index, part in enumerate(parts)
            )
            return rendered, tuple(params)
        names = [f"p{index}" for index in range(len(params))]
        if self.paramstyle == "named":
            token = ":{}"
        else:  # pyformat
            token = "%({})s"
        rendered = "".join(
            part + (token.format(names[index]) if index < len(params) else "")
            for index, part in enumerate(parts)
        )
        return rendered, {name: value for name, value in zip(names, params)}


# max_statement_params: modern SQLite allows 32766 bound variables (999 before
# 3.32); MSSQL caps at ~2100 per statement; the other engines take far more,
# capped here at a sane 60000.  Oracle folds unquoted DDL to uppercase, so its
# dialect upper-folds mapped names by default -- use
# ``dataclasses.replace(ORACLE, identifier_case="preserve")`` for schemas that
# were intentionally created with lowercase quoted identifiers.
SQLITE = Dialect(
    name="sqlite",
    paramstyle="qmark",
    introspection=INTROSPECTION_SQLITE,
    max_statement_params=32000,
)
POSTGRESQL = Dialect(
    name="postgresql",
    paramstyle="format",
    max_statement_params=60000,
    current_schema_expr="current_schema()",
)
MYSQL = Dialect(
    name="mysql",
    paramstyle="format",
    quote_open="`",
    quote_close="`",
    max_statement_params=60000,
    current_schema_expr="DATABASE()",
    # MySQL before 8.0.19 and every MariaDB reject a WHERE with no FROM.
    dummy_from=" FROM DUAL",
)
MSSQL = Dialect(
    name="mssql",
    paramstyle="qmark",
    quote_open="[",
    quote_close="]",
    max_statement_params=2000,
    current_schema_expr="SCHEMA_NAME()",
)
ORACLE = Dialect(
    name="oracle",
    paramstyle="named",
    introspection=INTROSPECTION_ORACLE,
    dummy_from=" FROM dual",
    max_statement_params=60000,
    identifier_case="upper",
)
GENERIC_ANSI = Dialect(name="generic_ansi", paramstyle="qmark", max_statement_params=2000)


def generic_ansi(paramstyle: str = "qmark") -> Dialect:
    """A standards-leaning fallback dialect with a caller-chosen paramstyle."""

    return replace(GENERIC_ANSI, paramstyle=paramstyle)


# Name registry, including same-technology compatibles served by an existing
# dialect (wire/SQL-compatible forks and managed offerings).
DIALECTS: dict[str, Dialect] = {
    "sqlite": SQLITE,
    "sqlite3": SQLITE,
    "postgresql": POSTGRESQL,
    "postgres": POSTGRESQL,
    "cockroachdb": POSTGRESQL,
    "redshift": POSTGRESQL,
    "alloydb": POSTGRESQL,
    "aurora-postgresql": POSTGRESQL,
    "mysql": MYSQL,
    "mariadb": MYSQL,
    "tidb": MYSQL,
    "aurora-mysql": MYSQL,
    "vitess": MYSQL,
    "mssql": MSSQL,
    "sqlserver": MSSQL,
    "azure-sql": MSSQL,
    "oracle": ORACLE,
    "generic": GENERIC_ANSI,
    "ansi": GENERIC_ANSI,
    "generic_ansi": GENERIC_ANSI,
}

# pyodbc speaks qmark, but pymssql speaks pyformat: it binds %s / %(name)s and
# rejects ?.  Same SQL Server quoting and introspection, only the paramstyle
# differs.
_MSSQL_PYMSSQL = replace(MSSQL, paramstyle="pyformat")
# PyMySQL and mysqlclient take %s, but MariaDB Connector/Python defaults to
# qmark (?).  The SQL dialect is otherwise MySQL's.
_MYSQL_MARIADB = replace(MYSQL, paramstyle="qmark")

# Driver module name -> dialect, for auto-detection from live connections.
_DRIVER_MODULES: dict[str, Dialect] = {
    "sqlite3": SQLITE,
    "psycopg": POSTGRESQL,
    "psycopg2": POSTGRESQL,
    "pg8000": POSTGRESQL,
    "pymysql": MYSQL,
    "MySQLdb": MYSQL,
    "mysql": MYSQL,  # mysql-connector-python
    "mariadb": _MYSQL_MARIADB,
    "pyodbc": MSSQL,  # pyodbc serves many engines; pass dialect= to override
    "pymssql": _MSSQL_PYMSSQL,
    "oracledb": ORACLE,
    "cx_Oracle": ORACLE,
}


def dialect_for(target: Any) -> Dialect:
    """Resolve a dialect from a name, a Dialect, or a live connection.

    Connection detection inspects the driver's top-level module name.  For an
    unknown PEP 249 driver, the module's mandatory ``paramstyle`` attribute
    seeds :data:`GENERIC_ANSI`.  An explicit ``Dialect`` always wins -- pass
    one whenever detection could be ambiguous (e.g. pyodbc to a non-MSSQL
    engine).
    """

    if isinstance(target, Dialect):
        return target
    if isinstance(target, str):
        dialect = DIALECTS.get(target.strip().lower())
        if dialect is None:
            raise RedactionDriftError(
                f"unknown dialect name {target!r}; known: {sorted(set(DIALECTS))}"
            )
        return dialect

    module_name = type(target).__module__.split(".")[0]
    known = _DRIVER_MODULES.get(module_name)
    if known is not None:
        return known
    import sys

    module = sys.modules.get(module_name)
    paramstyle = getattr(module, "paramstyle", None)
    if paramstyle in _PARAMSTYLES:
        return generic_ansi(paramstyle)
    raise RedactionDriftError(
        f"could not detect a SQL dialect for {type(target)!r}; pass dialect= explicitly."
    )
