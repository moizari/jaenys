"""Edge cases for the SQL backend: parameter-budget limits, NULL flags,

reused-attachment verification, corrupt stores, and CLI mapping I/O.
"""

from __future__ import annotations

import dataclasses
import os
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from jaenys import RedactionDriftError
from jaenys.cli import main as cli_main
from jaenys.derivation import DerivedLayers, Session
from jaenys.sql import filter_visible_ids, normal_record_predicate, sqlite
from jaenys.sql.dialects import SQLITE
from jaenys.sql.guard import not_in_span_predicate, status as guard_status
from tests.conftest import build_primary_db, create_mirror_table, create_span_member_table


def test_filter_visible_ids_stays_under_param_budget_with_large_span_set(
    store_pair: tuple[Path, Path],
) -> None:
    """Chunk statements must never carry the full span-member id set as params."""

    primary, span = store_pair
    ids = build_primary_db(primary, [0] * 6)
    span_ids = ids[2:6]  # 4 span members: exceeds the tiny max_in_params below
    create_span_member_table(span, span_ids)
    create_mirror_table(span, span_ids=span_ids)
    guard = sqlite.load_guard(span)
    tiny_dialect = dataclasses.replace(SQLITE, max_in_params=2)

    with closing(sqlite.open_readonly(primary)) as conn:
        visible = filter_visible_ids(conn, ids, guard=guard, dialect=tiny_dialect)

    assert visible == [ids[0], ids[1]]


def test_predicate_refuses_over_statement_parameter_budget() -> None:
    """All NOT IN chunks share ONE statement, so the whole id set counts.

    Refusing up front names the count, the budget, and the fix -- instead of
    letting the driver fail opaquely at execute time (MSSQL caps ~2100).
    """

    tiny = dataclasses.replace(SQLITE, max_statement_params=4)
    with pytest.raises(RedactionDriftError, match="parameter budget"):
        not_in_span_predicate("r", [1, 2, 3, 4, 5], dialect=tiny)
    # Exactly at the budget: allowed.
    predicate = not_in_span_predicate("r", [1, 2, 3, 4], dialect=tiny)
    assert predicate.params == (1, 2, 3, 4)


def test_normal_record_predicate_propagates_budget_refusal() -> None:
    tiny = dataclasses.replace(SQLITE, max_statement_params=2)
    with pytest.raises(RedactionDriftError, match="parameter budget"):
        normal_record_predicate("r", [1, 2, 3], dialect=tiny)


def test_status_stays_under_param_budget_with_large_span_set(
    store_pair: tuple[Path, Path],
) -> None:
    """status() must count via chunked statements, never an id-list predicate.

    With the tiny budgets below the old implementation refused outright,
    making the counts-only health check unusable exactly when the span
    layer reached production size.
    """

    primary, span = store_pair
    ids = build_primary_db(primary, [0, 1, 0, 0, 0, 0, 0, 0])
    span_ids = ids[2:7]  # 5 members: over both tiny budgets below
    create_span_member_table(span, span_ids)
    create_mirror_table(span, flagged_ids=[ids[1]], span_ids=span_ids)
    guard = sqlite.load_guard(span)
    tiny = dataclasses.replace(SQLITE, max_in_params=2, max_statement_params=4)

    with closing(sqlite.open_readonly(primary)) as conn:
        report = guard_status(conn, guard, dialect=tiny)

    assert report["records"] == 8
    assert report["flagged"] == 1
    assert report["visible_normal_records"] == 2
    assert report["blurred_standalone"] == 1
    assert report["hidden_in_span"] == 5
    assert report["deliverable_with_blur"] == 3


def test_generic_status_reports_loaded_guard_drift(store_pair: tuple[Path, Path]) -> None:
    primary, span = store_pair
    ids = build_primary_db(primary, [0, 1, 0])
    create_span_member_table(span, [])
    create_mirror_table(span, flagged_ids=[ids[1]], span_ids=[])
    guard = sqlite.load_guard(span)
    with closing(sqlite3.connect(primary)) as conn:
        conn.execute("UPDATE records SET sensitive = 1 WHERE record_id = ?", (ids[0],))
        conn.commit()

    with closing(sqlite.open_readonly(primary)) as conn:
        report = guard_status(conn, guard)

    assert report["layers_in_sync"] is False
    assert "stale" in report["refusal"]


def test_attached_guard_refuses_null_flag_rows(store_pair: tuple[Path, Path]) -> None:
    """A NULL primary flag refuses the whole attached serve path."""

    primary, span = store_pair
    conn = sqlite3.connect(primary)
    conn.execute(
        "CREATE TABLE records ("
        "record_id INTEGER PRIMARY KEY,"
        "body TEXT NOT NULL,"
        "sensitive INTEGER)"  # no NOT NULL constraint
    )
    conn.executemany(
        "INSERT INTO records (record_id, body, sensitive) VALUES (?, ?, ?)",
        [(1, "normal", 0), (2, "standalone flagged", 1), (3, "never classified", None)],
    )
    conn.commit()
    conn.close()
    create_span_member_table(span, [])
    create_mirror_table(span, flagged_ids=[2], span_ids=[])

    id_predicate = normal_record_predicate("r", [], include_blur=True)
    with closing(sqlite.open_readonly(primary)) as conn:
        rows = conn.execute(
            f"SELECT r.record_id FROM records r WHERE {id_predicate.sql} ORDER BY r.record_id",
            id_predicate.params,
        ).fetchall()
        assert [row[0] for row in rows] == [1, 2]

        with pytest.raises(RedactionDriftError, match="flag that is not 0 or 1"):
            with sqlite.attached_guard(conn, span):
                pass


def test_attached_guard_non_uri_connection_serves_or_names_the_culprit(
    store_pair: tuple[Path, Path],
) -> None:
    """Whether a plain connection processes ATTACH URIs is a build option.

    SQLite compiled with SQLITE_USE_URI (stock Ubuntu) honors the URI even
    without uri=True, and the guard must simply work: span rows hidden.
    Builds without it (Windows CPython) reject the ATTACH, and the refusal
    must blame the connection -- the span store is perfectly readable, and
    blaming it would send the operator debugging the wrong layer.  What must
    never happen on any build: serving with the span layer silently dropped.
    """

    primary, span = store_pair
    ids = build_primary_db(primary, [0, 0, 0])
    create_span_member_table(span, [ids[2]])
    create_mirror_table(span, span_ids=[ids[2]])

    with closing(sqlite3.connect(primary)) as conn:  # no uri=True
        try:
            with sqlite.attached_guard(conn, span) as guard:
                predicate = guard.predicate("r")
                rows = conn.execute(
                    f"SELECT r.record_id FROM records r WHERE {predicate.sql} ORDER BY 1",
                    predicate.params,
                ).fetchall()
                assert [row[0] for row in rows] == [ids[0], ids[1]]  # span row hidden
        except RedactionDriftError as exc:
            assert "uri=True" in str(exc)


def test_junk_span_member_id_refuses_at_load(store_pair: tuple[Path, Path]) -> None:
    """SQLite's flexible typing lets TEXT junk into an INTEGER id column.

    A junk id can never be excluded by an integer predicate, so silently
    skipping it would leak the row it was supposed to hide -- corruption in
    the span layer must refuse the whole load.
    """

    primary, span = store_pair
    build_primary_db(primary, [0, 0])
    conn = sqlite3.connect(span)
    conn.execute(
        "CREATE TABLE span_members ("
        "span_id INTEGER NOT NULL,"
        "record_id INTEGER NOT NULL,"
        "position INTEGER NOT NULL)"
    )
    conn.execute("INSERT INTO span_members VALUES (1, 'garbage-id', 1)")
    conn.commit()
    conn.close()

    with pytest.raises(RedactionDriftError, match="record id is not an integer"):
        sqlite.load_guard(span)


def test_attached_guard_refuses_non_integer_primary_record_id(
    store_pair: tuple[Path, Path],
) -> None:
    primary, span = store_pair
    conn = sqlite3.connect(primary)
    conn.execute("CREATE TABLE records (record_id TEXT PRIMARY KEY, sensitive INTEGER NOT NULL)")
    conn.execute("INSERT INTO records VALUES ('not-an-integer', 0)")
    conn.commit()
    conn.close()
    create_span_member_table(span, [])
    create_mirror_table(span, span_ids=[])

    with closing(sqlite.open_readonly(primary)) as conn:
        with pytest.raises(
            RedactionDriftError, match="record id in records is not a supported integer"
        ):
            with sqlite.attached_guard(conn, span):
                pass


def test_attached_guard_refuses_out_of_domain_primary_flag(
    store_pair: tuple[Path, Path],
) -> None:
    primary, span = store_pair
    conn = sqlite3.connect(primary)
    conn.execute("CREATE TABLE records (record_id INTEGER PRIMARY KEY, sensitive INTEGER NOT NULL)")
    conn.executemany("INSERT INTO records VALUES (?, ?)", [(1, 0), (2, 2)])
    conn.commit()
    conn.close()
    create_span_member_table(span, [])
    create_mirror_table(span, span_ids=[])

    with closing(sqlite.open_readonly(primary)) as conn:
        with pytest.raises(RedactionDriftError, match="flag that is not 0 or 1"):
            with sqlite.attached_guard(conn, span):
                pass


def test_attached_guard_refuses_duplicate_primary_record_id(
    store_pair: tuple[Path, Path],
) -> None:
    """A non-unique id cannot address a flag or a span to one record."""

    primary, span = store_pair
    conn = sqlite3.connect(primary)
    conn.execute("CREATE TABLE records (record_id INTEGER, sensitive INTEGER NOT NULL)")
    conn.executemany("INSERT INTO records VALUES (?, ?)", [(1, 0), (1, 0)])
    conn.commit()
    conn.close()
    create_span_member_table(span, [])
    create_mirror_table(span, span_ids=[])

    with closing(sqlite.open_readonly(primary)) as conn:
        with pytest.raises(RedactionDriftError, match="record id values in records are not unique"):
            with sqlite.attached_guard(conn, span):
                pass


def test_attached_guard_names_a_row_missing_its_id(store_pair: tuple[Path, Path]) -> None:
    primary, span = store_pair
    conn = sqlite3.connect(primary)
    conn.execute("CREATE TABLE records (record_id INTEGER, sensitive INTEGER NOT NULL)")
    conn.execute("INSERT INTO records VALUES (NULL, 0)")
    conn.commit()
    conn.close()
    create_span_member_table(span, [])
    create_mirror_table(span, span_ids=[])

    with closing(sqlite.open_readonly(primary)) as conn:
        with pytest.raises(RedactionDriftError, match="a row in records has no record id"):
            with sqlite.attached_guard(conn, span):
                pass


def test_filter_visible_ids_refuses_null_flag_rows(store_pair: tuple[Path, Path]) -> None:
    """A NULL primary flag refuses the bounded id path too."""

    primary, span = store_pair
    conn = sqlite3.connect(primary)
    conn.execute(
        "CREATE TABLE records ("
        "record_id INTEGER PRIMARY KEY,"
        "sender TEXT NOT NULL,"
        "sent_at TEXT NOT NULL,"
        "body TEXT NOT NULL,"
        "sensitive INTEGER)"  # no NOT NULL constraint
    )
    cursor = conn.execute(
        "INSERT INTO records (sender, sent_at, body, sensitive) VALUES (?, ?, ?, ?)",
        ("agent_a", "2024-01-01 10:00:00", "line with null flag", None),
    )
    null_flag_id = cursor.lastrowid
    cursor = conn.execute(
        "INSERT INTO records (sender, sent_at, body, sensitive) VALUES (?, ?, ?, ?)",
        ("customer_b", "2024-01-01 10:01:00", "normal line", 0),
    )
    normal_id = cursor.lastrowid
    conn.commit()
    conn.close()

    # No flagged rows and no span layer: the flag-filter-only guard is in sync.
    guard = sqlite.load_guard(span)
    with closing(sqlite.open_readonly(primary)) as conn:
        with pytest.raises(RedactionDriftError, match="flag that is not 0 or 1"):
            filter_visible_ids(conn, [null_flag_id, normal_id], guard=guard)


def test_filter_visible_ids_junk_flag_value_refuses(store_pair: tuple[Path, Path]) -> None:
    """A non-numeric flag value is corruption: refuse loudly, never guess."""

    primary, span = store_pair
    conn = sqlite3.connect(primary)
    conn.execute(
        "CREATE TABLE records ("
        "record_id INTEGER PRIMARY KEY,"
        "sender TEXT NOT NULL,"
        "sent_at TEXT NOT NULL,"
        "body TEXT NOT NULL,"
        "sensitive INTEGER)"  # SQLite's flexible typing lets junk text in
    )
    conn.execute(
        "INSERT INTO records (sender, sent_at, body, sensitive) VALUES (?, ?, ?, ?)",
        ("agent_a", "2024-01-01 10:00:00", "line with junk flag", "garbage"),
    )
    conn.commit()
    conn.close()

    guard = sqlite.load_guard(span)
    with closing(sqlite.open_readonly(primary)) as conn:
        with pytest.raises(RedactionDriftError, match="flag that is not 0 or 1"):
            filter_visible_ids(conn, [1], guard=guard)


def test_status_and_cli_report_corrupt_flag_as_unhealthy(
    store_pair: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    primary, span = store_pair
    conn = sqlite3.connect(primary)
    conn.execute("CREATE TABLE records (record_id INTEGER PRIMARY KEY, sensitive INTEGER)")
    conn.execute("INSERT INTO records VALUES (1, 2)")
    conn.commit()
    conn.close()

    with pytest.raises(RedactionDriftError, match="flag that is not 0 or 1") as excinfo:
        sqlite.status(primary, span)
    assert "record 1" not in str(excinfo.value)

    exit_code = cli_main(["--primary-db", str(primary), "--span-db", str(span)])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "a record in records has a flag that is not 0 or 1" in captured.err


def test_status_names_the_record_with_a_missing_flag(store_pair: tuple[Path, Path]) -> None:
    primary, span = store_pair
    conn = sqlite3.connect(primary)
    conn.execute("CREATE TABLE records (record_id INTEGER PRIMARY KEY, sensitive INTEGER)")
    conn.execute("INSERT INTO records VALUES (7, NULL)")
    conn.commit()
    conn.close()

    with pytest.raises(RedactionDriftError, match="a record in records has a flag"):
        sqlite.status(primary, span)


def test_attached_guard_reused_attachment_mismatch_refuses(
    store_pair: tuple[Path, Path], tmp_path: Path
) -> None:
    primary, span = store_pair
    build_primary_db(primary, [0, 0])
    create_span_member_table(span, [])
    create_mirror_table(span, span_ids=[])

    other_db = tmp_path / "other.db"
    sqlite3.connect(other_db).execute("CREATE TABLE placeholder (x INTEGER)").connection.commit()

    with closing(sqlite.open_readonly(primary)) as conn:
        conn.execute(
            f"ATTACH DATABASE ? AS {SQLITE.quote_identifier(sqlite.DEFAULT_ATTACH_NAME)}",
            (f"{other_db.resolve().as_uri()}?mode=ro",),
        )
        try:
            with pytest.raises(RedactionDriftError, match="already bound to a different database"):
                with sqlite.attached_guard(conn, span):
                    pass
        finally:
            conn.execute(f"DETACH DATABASE {SQLITE.quote_identifier(sqlite.DEFAULT_ATTACH_NAME)}")


def test_attached_guard_reused_attachment_same_file_serves(
    store_pair: tuple[Path, Path],
) -> None:
    primary, span = store_pair
    ids = build_primary_db(primary, [0, 1, 0])
    create_span_member_table(span, [])
    create_mirror_table(span, flagged_ids=[ids[1]], span_ids=[])

    with closing(sqlite.open_readonly(primary)) as conn:
        conn.execute(
            f"ATTACH DATABASE ? AS {SQLITE.quote_identifier(sqlite.DEFAULT_ATTACH_NAME)}",
            (f"{span.resolve().as_uri()}?mode=ro",),
        )
        with sqlite.attached_guard(conn, span) as guard:
            predicate = guard.predicate("r")
            rows = conn.execute(
                f"SELECT r.record_id FROM records r WHERE {predicate.sql} ORDER BY r.record_id",
                predicate.params,
            ).fetchall()
        conn.execute(f"DETACH DATABASE {SQLITE.quote_identifier(sqlite.DEFAULT_ATTACH_NAME)}")
    assert [row[0] for row in rows] == [ids[0], ids[2]]


def test_remove_impostor_attach_file_noops_on_missing_path(tmp_path: Path) -> None:
    real = tmp_path / "span.db"
    missing = tmp_path / "file:nonexistent?mode=ro"
    # No file to remove and nothing raised.
    sqlite._remove_impostor_attach_file(str(missing), real_span_path=real)
    assert not missing.exists()


@pytest.mark.skipif(os.name == "nt", reason="literal-path ATTACH impostor is POSIX-only")
def test_remove_impostor_attach_file_removes_zero_byte_impostor(tmp_path: Path) -> None:
    real = tmp_path / "span.db"
    impostor = tmp_path / "file:span.db?mode=ro"
    impostor.touch()  # zero bytes, name carries the URI query suffix
    assert impostor.exists()
    sqlite._remove_impostor_attach_file(str(impostor), real_span_path=real)
    assert not impostor.exists()


@pytest.mark.skipif(os.name == "nt", reason="literal-path ATTACH impostor is POSIX-only")
def test_remove_impostor_attach_file_leaves_non_empty_file(tmp_path: Path) -> None:
    real = tmp_path / "span.db"
    impostor = tmp_path / "file:span.db?mode=ro"
    impostor.write_bytes(b"real content, not a zero-byte impostor")
    sqlite._remove_impostor_attach_file(str(impostor), real_span_path=real)
    assert impostor.exists()


def test_same_database_file_recognizes_one_file_under_two_names(tmp_path: Path) -> None:
    """Attach verification must treat two names for one file as the same store.

    A hardlink gives one file two path strings whose text differs.  Deciding
    identity by device+inode (os.path.samefile) is what keeps this correct on
    case-insensitive filesystems (Windows, macOS) and through links, where a
    plain string compare (os.path.normcase is a no-op on POSIX) would wrongly
    report a mismatch and refuse a perfectly valid span store.
    """

    real = tmp_path / "span.db"
    real.write_bytes(b"span store bytes")
    link = tmp_path / "span_other_name.db"
    try:
        os.link(real, link)
    except (OSError, NotImplementedError):
        pytest.skip("hardlinks are not supported on this filesystem")

    assert sqlite._same_database_file(str(link), real)

    unrelated = tmp_path / "unrelated.db"
    unrelated.write_bytes(b"span store bytes")  # identical bytes, different file
    assert not sqlite._same_database_file(str(unrelated), real)


def test_attached_guard_corrupt_span_db_refuses(store_pair: tuple[Path, Path]) -> None:
    primary, span = store_pair
    build_primary_db(primary, [0])
    span.write_bytes(b"this is not a valid sqlite database, but not empty")

    with closing(sqlite.open_readonly(primary)) as conn:
        with pytest.raises(RedactionDriftError, match="unreadable"):
            with sqlite.attached_guard(conn, span):
                pass


def test_status_corrupt_primary_db_is_a_clean_refusal(store_pair: tuple[Path, Path]) -> None:
    primary, span = store_pair
    create_span_member_table(span, [])
    create_mirror_table(span, span_ids=[])
    primary.write_bytes(b"this is not a valid sqlite database, but not empty")

    with pytest.raises(RedactionDriftError, match="unreadable"):
        sqlite.status(primary, span)


def test_cli_missing_mapping_file_returns_clean_refusal(
    store_pair: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    primary, span = store_pair
    build_primary_db(primary, [0, 1, 0])
    create_span_member_table(span, [])
    create_mirror_table(span, span_ids=[])
    missing_mapping = span.parent / "does_not_exist.json"

    exit_code = cli_main(
        [
            "--primary-db",
            str(primary),
            "--span-db",
            str(span),
            "--mapping",
            str(missing_mapping),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "cannot read mapping file" in captured.err
    assert captured.out == ""


def test_write_span_layer_drops_stale_version_group_table(tmp_path: Path) -> None:
    """A leftover versioned spans table must not survive the non-versioned writer."""

    span = tmp_path / "span.db"
    conn = sqlite3.connect(span)
    conn.execute(
        "CREATE TABLE spans (span_id INTEGER PRIMARY KEY,source_version_id INTEGER NOT NULL)"
    )
    conn.execute("INSERT INTO spans VALUES (1, 42)")
    conn.execute(
        "CREATE TABLE span_members ("
        "span_id INTEGER NOT NULL,"
        "record_id INTEGER NOT NULL,"
        "position INTEGER NOT NULL)"
    )
    conn.execute("INSERT INTO span_members VALUES (1, 5, 1)")
    conn.commit()
    conn.close()

    layers = DerivedLayers(flagged_ids=frozenset(), sessions=(Session((5, 6, 7), "end_of_data"),))
    sqlite.write_span_layer(span, layers)

    with closing(sqlite3.connect(span)) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'spans'"
        ).fetchall()
    assert rows == []
