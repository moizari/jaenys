"""Shared synthetic fixtures: a support-ticket transcript, fully generated.

Two SQLite stores model the two sensitivity layers:

* primary store: ``records`` with a live ``sensitive`` flag per row
* span store: ``span_members`` (+ optional ``spans`` groups) and the
  ``sensitive_records`` mirror captured when spans were derived
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


def build_primary_db(path: Path, flags: list[int]) -> list[int]:
    """Create the primary store with one record per flag; return record ids."""

    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS records ("
        "record_id INTEGER PRIMARY KEY,"
        "sender TEXT NOT NULL,"
        "sent_at TEXT NOT NULL,"
        "body TEXT NOT NULL,"
        "sensitive INTEGER NOT NULL DEFAULT 0)"
    )
    ids: list[int] = []
    for index, flag in enumerate(flags, start=1):
        cursor = conn.execute(
            "INSERT INTO records (sender, sent_at, body, sensitive) VALUES (?, ?, ?, ?)",
            (
                "agent_a" if index % 2 else "customer_b",
                f"2024-01-01 10:{index:02d}:00",
                f"synthetic ticket line {index}",
                flag,
            ),
        )
        ids.append(cursor.lastrowid)
    conn.commit()
    conn.close()
    return ids


def create_span_member_table(
    span_db: Path,
    record_ids: list[int],
    *,
    span_id: int = 1,
    version_id: int | None = None,
) -> None:
    """Create the member-table span shape (+ optional versioned group)."""

    conn = sqlite3.connect(span_db)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS span_members ("
        "span_id INTEGER NOT NULL,"
        "record_id INTEGER NOT NULL,"
        "position INTEGER NOT NULL,"
        "PRIMARY KEY (span_id, record_id))"
    )
    conn.executemany(
        "INSERT INTO span_members VALUES (?, ?, ?)",
        [(span_id, record_id, position) for position, record_id in enumerate(record_ids, start=1)],
    )
    if version_id is not None:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS spans ("
            "span_id INTEGER PRIMARY KEY,"
            "source_version_id INTEGER NOT NULL)"
        )
        conn.execute(
            "INSERT OR IGNORE INTO spans VALUES (?, ?)",
            (span_id, version_id),
        )
    conn.commit()
    conn.close()


def create_mirror_table(
    span_db: Path,
    *,
    flagged_ids: list[int] = (),
    span_ids: list[int] = (),
    modern: bool = True,
) -> None:
    """Create the mirror table.

    ``modern=True`` includes the ``source_flag`` column needed for freshness
    verification; ``modern=False`` builds the legacy span-only shape.
    """

    conn = sqlite3.connect(span_db)
    if modern:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS sensitive_records ("
            "record_id INTEGER NOT NULL,"
            "copy_reason TEXT NOT NULL,"
            "source_flag INTEGER NOT NULL DEFAULT 1)"
        )
        conn.executemany(
            "INSERT INTO sensitive_records VALUES (?, 'flagged', 1)",
            [(record_id,) for record_id in flagged_ids],
        )
        conn.executemany(
            "INSERT INTO sensitive_records VALUES (?, 'span', 1)",
            [(record_id,) for record_id in span_ids],
        )
    else:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS sensitive_records ("
            "record_id INTEGER NOT NULL,"
            "copy_reason TEXT NOT NULL)"
        )
        conn.executemany(
            "INSERT INTO sensitive_records VALUES (?, 'span')",
            [(record_id,) for record_id in span_ids],
        )
    conn.commit()
    conn.close()


@pytest.fixture()
def store_pair(tmp_path: Path) -> tuple[Path, Path]:
    """(primary_db, span_db) paths; span store starts as a 0-byte placeholder."""

    primary = tmp_path / "primary.db"
    span = tmp_path / "span.db"
    span.touch()
    return primary, span
