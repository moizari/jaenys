#!/usr/bin/env python3
"""Runnable, stdlib-only walkthrough of the SQL fail-closed guard.

Builds a synthetic support-ticket primary/span SQLite pair in a tempdir,
then walks through the full story:

  1. A clean, freshly-derived pair serves three states: VISIBLE, BLUR
     (standalone flagged), and HIDDEN (in-span, absent entirely).
  2. A flag is edited on the primary store WITHOUT re-deriving the span
     layer.
  3. The engine detects the drift between the live flag layer and the
     mirrored copy and REFUSES to serve -- raising RedactionDriftError instead
     of silently serving stale isolation.
  4. The span layer is re-derived (mirror rewritten) and service is
     restored.

Run with: python examples/demo_sql_guard.py
"""

from __future__ import annotations

import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

from jaenys import RedactionDriftError, annotate_rows
from jaenys.derivation import DerivedLayers, Session
from jaenys.sql import sqlite


def build_primary_db(path: Path) -> list[int]:
    """Create the primary store: a support-ticket transcript, 8 rows.

    Rows 5-7 form a contiguous span (a stretch that is only sensitive
    because of its surrounding context); row 2 is a standalone flagged
    line outside any span.
    """

    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE records ("
        "record_id INTEGER PRIMARY KEY,"
        "sender TEXT NOT NULL,"
        "sent_at TEXT NOT NULL,"
        "body TEXT NOT NULL,"
        "sensitive INTEGER NOT NULL DEFAULT 0)"
    )
    rows = [
        ("agent", "2026-01-01 09:00", "Hi, how can I help today?", 0),
        ("customer", "2026-01-01 09:01", "My account PIN is 4471.", 1),  # standalone flagged
        ("agent", "2026-01-01 09:02", "Thanks, one moment.", 0),
        ("customer", "2026-01-01 09:03", "Sure, no rush.", 0),
        ("agent", "2026-01-01 09:04", "Can you confirm your date of birth?", 0),  # span start
        ("customer", "2026-01-01 09:05", "March 3rd, 1990.", 0),
        ("agent", "2026-01-01 09:06", "Got it, verified.", 0),  # span end
        ("customer", "2026-01-01 09:07", "Great, thank you!", 0),
    ]
    ids: list[int] = []
    for sender, sent_at, body, flag in rows:
        cursor = conn.execute(
            "INSERT INTO records (sender, sent_at, body, sensitive) VALUES (?, ?, ?, ?)",
            (sender, sent_at, body, flag),
        )
        ids.append(cursor.lastrowid)
    conn.commit()
    conn.close()
    return ids


def derive_span_layer(span_path: Path, primary_path: Path, span_ids: list[int]) -> None:
    """Rebuild the span store: span members + a fresh flag-layer mirror.

    This models what a real derivation pipeline does: recompute contiguous
    spans and snapshot the live flag layer at that moment. Nothing in this
    demo library performs this rebuild automatically -- callers own their
    own derivation step.
    """

    primary_conn = sqlite3.connect(primary_path)
    flagged_ids = [
        row[0]
        for row in primary_conn.execute(
            "SELECT record_id FROM records WHERE sensitive = 1"
        ).fetchall()
    ]
    primary_conn.close()

    layers = DerivedLayers(
        flagged_ids=frozenset(flagged_ids),
        sessions=(Session(tuple(span_ids), "demo_span"),),
    )
    sqlite.write_flags(primary_path, layers)
    sqlite.write_span_layer(span_path, layers)


def fetch_all_rows(primary_path: Path) -> list[dict]:
    conn = sqlite3.connect(primary_path)
    conn.row_factory = sqlite3.Row
    rows = [dict(row) for row in conn.execute("SELECT * FROM records ORDER BY record_id")]
    conn.close()
    return rows


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="jaenys_demo_") as tmp:
        tmp_path = Path(tmp)
        primary_path = tmp_path / "primary.db"
        span_path = tmp_path / "span.db"

        print("=== Step 1: build the synthetic support-ticket pair ===")
        ids = build_primary_db(primary_path)
        span_ids = ids[4:7]  # the 3-row "date of birth" span
        derive_span_layer(span_path, primary_path, span_ids)
        print(f"Primary store: {len(ids)} rows. Span store: derived, 1 span of 3 rows.\n")

        print("=== Step 2: serve the three-state result ===")
        with closing(sqlite.open_readonly(primary_path)) as conn:
            with sqlite.attached_guard(conn, span_path) as guard:
                predicate = guard.predicate("r", include_blur=True)
                cursor = conn.execute(
                    f"SELECT r.* FROM records r WHERE {predicate.sql} ORDER BY r.record_id",
                    predicate.params,
                )
                cursor.row_factory = sqlite3.Row
                deliverable_rows = [dict(row) for row in cursor.fetchall()]

        annotated = annotate_rows(
            deliverable_rows,
            span_member_ids=frozenset(span_ids),
            id_key="record_id",
            flag_key="sensitive",
        )
        all_ids = set(ids)
        served_ids = {row["record_id"] for row in annotated}
        for row in annotated:
            state = "BLUR" if row["blurred"] else "VISIBLE"
            print(f"  [{state:7}] id={row['record_id']:>2}  {row['body']!r}")
        for hidden_id in sorted(all_ids - served_ids):
            print(f"  [HIDDEN ] id={hidden_id:>2}  (in-span; not served)")
        print()

        print("=== Step 3: flip a flag WITHOUT re-deriving the span layer ===")
        edit_conn = sqlite3.connect(primary_path)
        edit_conn.execute("UPDATE records SET sensitive = 1 WHERE record_id = ?", (ids[0],))
        edit_conn.commit()
        edit_conn.close()
        print(f"  Flagged record {ids[0]} on the primary store. Span store untouched.\n")

        print("=== Step 4: the engine refuses to serve stale isolation ===")
        try:
            with closing(sqlite.open_readonly(primary_path)) as conn:
                with sqlite.attached_guard(conn, span_path):
                    print("  (should not reach here)")
        except RedactionDriftError as exc:
            print(f"  RedactionDriftError raised, as expected:\n  -> {exc}\n")

        print("=== Step 5: re-derive the span layer and confirm service is restored ===")
        derive_span_layer(span_path, primary_path, span_ids)
        with closing(sqlite.open_readonly(primary_path)) as conn:
            with sqlite.attached_guard(conn, span_path) as guard:
                predicate = guard.predicate("r", include_blur=True)
                count = conn.execute(
                    f"SELECT COUNT(*) FROM records r WHERE {predicate.sql}", predicate.params
                ).fetchone()[0]
        print(f"  Re-derived. Serving normally again: {count} deliverable rows.")


if __name__ == "__main__":
    main()
