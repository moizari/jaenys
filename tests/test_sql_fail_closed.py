"""SQL-backend fail-closed drift matrix on the SQLite reference engine.

Every path where the two sensitivity layers cannot be proven equal must
refuse to serve -- for both the materialized (two-store) guard and the
single-connection anti-join guard.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from jaenys import RedactionDriftError
from jaenys.derivation import DerivedLayers, Session
from jaenys.sql import assert_guard_current, sqlite
from tests.conftest import build_primary_db, create_mirror_table, create_span_member_table


# -- materialized guard (two stores, verified in Python) ---------------------


def test_clean_sync_serves(store_pair: tuple[Path, Path]) -> None:
    primary, span = store_pair
    ids = build_primary_db(primary, [0, 1, 0, 0])
    create_span_member_table(span, [ids[2], ids[3]])
    create_mirror_table(span, flagged_ids=[ids[1]], span_ids=[ids[2], ids[3]])
    guard = sqlite.load_guard(span)
    with closing(sqlite.open_readonly(primary)) as conn:
        assert_guard_current(conn, guard)  # must not raise


def test_flag_added_without_rebuild_refuses(store_pair: tuple[Path, Path]) -> None:
    primary, span = store_pair
    ids = build_primary_db(primary, [0, 1, 0])
    create_span_member_table(span, [ids[1]])
    create_mirror_table(span, flagged_ids=[ids[1]])
    guard = sqlite.load_guard(span)
    sqlite3.connect(primary).execute(
        "UPDATE records SET sensitive = 1 WHERE record_id = ?", (ids[0],)
    ).connection.commit()
    with closing(sqlite.open_readonly(primary)) as conn:
        with pytest.raises(RedactionDriftError, match="stale"):
            assert_guard_current(conn, guard)


def test_flag_cleared_without_rebuild_refuses(store_pair: tuple[Path, Path]) -> None:
    primary, span = store_pair
    ids = build_primary_db(primary, [0, 1])
    create_span_member_table(span, [ids[1]])
    create_mirror_table(span, flagged_ids=[ids[1]])
    guard = sqlite.load_guard(span)
    sqlite3.connect(primary).execute(
        "UPDATE records SET sensitive = 0 WHERE record_id = ?", (ids[1],)
    ).connection.commit()
    with closing(sqlite.open_readonly(primary)) as conn:
        with pytest.raises(RedactionDriftError, match="stale"):
            assert_guard_current(conn, guard)


def test_placeholder_span_store_with_flags_refuses(store_pair: tuple[Path, Path]) -> None:
    primary, span = store_pair  # span stays a 0-byte placeholder
    build_primary_db(primary, [0, 1])
    guard = sqlite.load_guard(span)
    assert guard.span_layer_ready is False
    with closing(sqlite.open_readonly(primary)) as conn:
        with pytest.raises(RedactionDriftError, match="unavailable"):
            assert_guard_current(conn, guard)


def test_placeholder_span_store_without_flags_serves(store_pair: tuple[Path, Path]) -> None:
    primary, span = store_pair
    build_primary_db(primary, [0, 0, 0])
    guard = sqlite.load_guard(span)
    with closing(sqlite.open_readonly(primary)) as conn:
        assert_guard_current(conn, guard)  # must not raise


def test_ready_span_store_with_no_span_tables_refuses_when_flagged(
    store_pair: tuple[Path, Path],
) -> None:
    primary, span = store_pair
    build_primary_db(primary, [1])
    conn = sqlite3.connect(span)  # non-empty store, but no span-capable tables
    conn.execute("CREATE TABLE unrelated (x INTEGER)")
    conn.commit()
    conn.close()
    guard = sqlite.load_guard(span)
    with closing(sqlite.open_readonly(primary)) as conn:
        with pytest.raises(RedactionDriftError, match="span"):
            assert_guard_current(conn, guard)


def test_unreadable_span_store_refuses(store_pair: tuple[Path, Path]) -> None:
    primary, span = store_pair
    build_primary_db(primary, [0, 1])
    span.write_bytes(b"this is not a database file, but it is not empty either")
    with pytest.raises(RedactionDriftError, match="unreadable"):
        sqlite.load_guard(span)


def test_span_store_corrupted_after_load_refuses(store_pair: tuple[Path, Path]) -> None:
    primary, span = store_pair
    ids = build_primary_db(primary, [0, 1])
    create_span_member_table(span, [ids[1]])
    create_mirror_table(span, flagged_ids=[ids[1]])
    guard = sqlite.load_guard(span)
    span.write_bytes(b"corrupted after load")  # freshness re-check must see this
    with closing(sqlite.open_readonly(primary)) as conn:
        with pytest.raises(RedactionDriftError, match="unreadable"):
            assert_guard_current(conn, guard)


def test_membership_change_after_load_refuses(store_pair: tuple[Path, Path]) -> None:
    """A span re-derivation pulls a new neutral record into a span, flags untouched.

    The mirror only witnesses the FLAG layer, so this drift is invisible to
    the mirror comparison -- the membership re-read must catch it, and the
    guard must refuse rather than silently adopt the new set.
    """

    primary, span = store_pair
    ids = build_primary_db(primary, [0, 1, 0, 0])
    create_span_member_table(span, [ids[2]])
    create_mirror_table(span, flagged_ids=[ids[1]])
    guard = sqlite.load_guard(span)
    with closing(sqlite.open_readonly(primary)) as conn:
        assert_guard_current(conn, guard)  # in sync at load

    grow = sqlite3.connect(span)
    grow.execute("INSERT INTO span_members VALUES (1, ?, 2)", (ids[3],))
    grow.commit()
    grow.close()

    with closing(sqlite.open_readonly(primary)) as conn:
        with pytest.raises(RedactionDriftError, match="membership .* changed"):
            assert_guard_current(conn, guard)
    # Reloading the guard adopts the new membership and restores service.
    reloaded = sqlite.load_guard(span)
    with closing(sqlite.open_readonly(primary)) as conn:
        assert_guard_current(conn, reloaded)


def test_filter_visible_ids_refuses_on_membership_drift(store_pair: tuple[Path, Path]) -> None:
    from jaenys.sql import filter_visible_ids

    primary, span = store_pair
    ids = build_primary_db(primary, [0, 1, 0, 0])
    create_span_member_table(span, [ids[2]])
    create_mirror_table(span, flagged_ids=[ids[1]])
    guard = sqlite.load_guard(span)
    with closing(sqlite.open_readonly(primary)) as conn:
        assert filter_visible_ids(conn, ids, guard=guard) == [ids[0], ids[3]]

    grow = sqlite3.connect(span)
    grow.execute("INSERT INTO span_members VALUES (1, ?, 2)", (ids[3],))
    grow.commit()
    grow.close()

    with closing(sqlite.open_readonly(primary)) as conn:
        with pytest.raises(RedactionDriftError, match="membership .* changed"):
            filter_visible_ids(conn, ids, guard=guard)
    reloaded = sqlite.load_guard(span)
    with closing(sqlite.open_readonly(primary)) as conn:
        assert filter_visible_ids(conn, ids, guard=reloaded) == [ids[0]]


def test_legacy_span_only_store_degrades_safely(store_pair: tuple[Path, Path]) -> None:
    """No mirror table: spans still hide, flags still checked, no freshness proof."""

    primary, span = store_pair
    ids = build_primary_db(primary, [0, 1, 0])
    create_span_member_table(span, [ids[2]])
    guard = sqlite.load_guard(span)
    assert guard.span_sources == ("span_members",)
    with closing(sqlite.open_readonly(primary)) as conn:
        assert_guard_current(conn, guard)  # cannot prove freshness, but is safe


def test_legacy_mirror_without_flag_column_degrades_safely(
    store_pair: tuple[Path, Path],
) -> None:
    primary, span = store_pair
    ids = build_primary_db(primary, [0, 1])
    create_mirror_table(span, span_ids=[ids[0]], modern=False)
    guard = sqlite.load_guard(span)
    assert guard.span_sources == ("mirror.span",)
    with closing(sqlite.open_readonly(primary)) as conn:
        assert_guard_current(conn, guard)


# -- single-connection (attached) guard ---------------------------------------


def test_attached_clean_sync_serves(store_pair: tuple[Path, Path]) -> None:
    primary, span = store_pair
    ids = build_primary_db(primary, [0, 1, 0, 1, 0])
    create_span_member_table(span, [ids[2], ids[3]])
    create_mirror_table(span, flagged_ids=[ids[1], ids[3]], span_ids=[ids[2], ids[3]])
    with closing(sqlite.open_readonly(primary)) as conn:
        with sqlite.attached_guard(conn, span) as guard:
            predicate = guard.predicate("r")
            rows = conn.execute(
                f"SELECT r.record_id FROM records r WHERE {predicate.sql} ORDER BY r.record_id",
                predicate.params,
            ).fetchall()
    # 2nd and 4th are flagged; 3rd and 4th are in-span.  Only clean,
    # out-of-span rows remain visible on normal surfaces.
    assert [row[0] for row in rows] == [ids[0], ids[4]]


def test_attached_stale_mirror_refuses(store_pair: tuple[Path, Path]) -> None:
    primary, span = store_pair
    ids = build_primary_db(primary, [0, 1, 1])
    create_span_member_table(span, [ids[1]])
    create_mirror_table(span, flagged_ids=[ids[1]])  # missing ids[2]
    with closing(sqlite.open_readonly(primary)) as conn:
        with pytest.raises(RedactionDriftError, match="stale"):
            with sqlite.attached_guard(conn, span):
                pass


def test_attached_orphan_mirror_row_refuses(store_pair: tuple[Path, Path]) -> None:
    primary, span = store_pair
    ids = build_primary_db(primary, [0, 1])
    create_span_member_table(span, [ids[1]])
    create_mirror_table(span, flagged_ids=[ids[1], 9999])  # 9999 not live-flagged
    with closing(sqlite.open_readonly(primary)) as conn:
        with pytest.raises(RedactionDriftError, match="stale"):
            with sqlite.attached_guard(conn, span):
                pass


def test_attached_witness_digest_mismatch_refuses_even_with_same_count(
    store_pair: tuple[Path, Path],
) -> None:
    primary, span = store_pair
    ids = build_primary_db(primary, [0, 0, 0, 0, 0, 0, 0])
    recorded = DerivedLayers(
        flagged_ids=frozenset(),
        sessions=(Session(tuple(ids[1:4]), "end_of_data"),),
    )
    actual = DerivedLayers(
        flagged_ids=frozenset(),
        sessions=(Session(tuple(ids[4:7]), "end_of_data"),),
    )
    sqlite.write_flags(primary, recorded)
    sqlite.write_span_layer(span, actual)

    with closing(sqlite.open_readonly(primary)) as conn:
        with pytest.raises(RedactionDriftError, match="does not match"):
            with sqlite.attached_guard(conn, span):
                pass


def test_attached_placeholder_span_store_with_flags_refuses(
    store_pair: tuple[Path, Path],
) -> None:
    primary, span = store_pair
    build_primary_db(primary, [1])
    with closing(sqlite.open_readonly(primary)) as conn:
        with pytest.raises(RedactionDriftError, match="unavailable"):
            with sqlite.attached_guard(conn, span):
                pass


def test_attached_placeholder_span_store_without_flags_serves(
    store_pair: tuple[Path, Path],
) -> None:
    primary, span = store_pair
    ids = build_primary_db(primary, [0, 0])
    with closing(sqlite.open_readonly(primary)) as conn:
        with sqlite.attached_guard(conn, span) as guard:
            assert guard.span_layer_ready is False
            predicate = guard.predicate("r")
            rows = conn.execute(
                f"SELECT r.record_id FROM records r WHERE {predicate.sql}",
                predicate.params,
            ).fetchall()
    assert [row[0] for row in rows] == ids


def test_attached_legacy_span_only_store_degrades_safely(
    store_pair: tuple[Path, Path],
) -> None:
    primary, span = store_pair
    ids = build_primary_db(primary, [0, 1, 0])
    create_span_member_table(span, [ids[2]])  # no mirror table at all
    with closing(sqlite.open_readonly(primary)) as conn:
        with sqlite.attached_guard(conn, span) as guard:
            predicate = guard.predicate("r")
            rows = conn.execute(
                f"SELECT r.record_id FROM records r WHERE {predicate.sql} ORDER BY r.record_id",
                predicate.params,
            ).fetchall()
    assert [row[0] for row in rows] == [ids[0]]


def test_filter_visible_ids_refuses_on_drift(store_pair: tuple[Path, Path]) -> None:
    """Serve-path helpers re-verify freshness on every call."""

    from jaenys.sql import filter_visible_ids

    primary, span = store_pair
    ids = build_primary_db(primary, [0, 1, 0])
    create_span_member_table(span, [ids[1]])
    create_mirror_table(span, flagged_ids=[ids[1]])
    guard = sqlite.load_guard(span)
    with closing(sqlite.open_readonly(primary)) as conn:
        assert filter_visible_ids(conn, ids, guard=guard) == [ids[0], ids[2]]
    edit = sqlite3.connect(primary)
    edit.execute("UPDATE records SET sensitive = 1 WHERE record_id = ?", (ids[0],))
    edit.commit()
    edit.close()
    with closing(sqlite.open_readonly(primary)) as conn:
        with pytest.raises(RedactionDriftError, match="stale"):
            filter_visible_ids(conn, ids, guard=guard)


def test_closed_connection_refuses_via_redaction_drift_error(
    store_pair: tuple[Path, Path],
) -> None:
    """A raw driver error on a serve-path entry must surface as a refusal."""

    from jaenys.sql import filter_visible_ids

    primary, span = store_pair
    ids = build_primary_db(primary, [0, 1, 0])
    create_span_member_table(span, [ids[1]])
    create_mirror_table(span, flagged_ids=[ids[1]])
    guard = sqlite.load_guard(span)
    conn = sqlite.open_readonly(primary)
    conn.close()  # every read now raises a raw ProgrammingError
    with pytest.raises(RedactionDriftError, match="unreadable"):
        filter_visible_ids(conn, ids, guard=guard)
