#!/usr/bin/env python3
"""Runnable, stdlib-only walkthrough of the populate-the-template derivation.

The other two demos start from records that are already flagged and already
grouped into spans. This one shows how those layers get made: populate the
two templates in ``jaenys.derivation`` with YOUR vocabulary and let
the toolkit do the mechanics.

  1. FLAG   -- KeywordDetector, populated with the terms that make a single
     record sensitive on its own. Anything a word list cannot express plugs
     in as an ``extra`` callable (a regex, Presidio, your classifier).
  2. SPAN   -- CueSessionDeriver, populated with the cues that open and
     close a sensitive stretch plus an inactivity timeout. Every row
     between open and close rides along -- even rows that look harmless on
     their own -- and each session records why it ended.
  3. WRITE  -- write_flags() puts the flag layer on the primary store;
     write_span_layer() rebuilds the span store: members + the flag-mirror
     snapshot the engine verifies freshness against.
  4. Serving is then exactly demo_sql_guard.py -- and when a new record
     gets flagged, the engine refuses until the derivation re-runs.

The templates match exactly what you populate them with: plain substring
matching, no NLP, no heuristics. The engine does not care how the layers
are produced -- only that they stay in sync.

Run with: python examples/demo_span_derivation.py
"""

from __future__ import annotations

import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

from jaenys import annotate_rows, RedactionDriftError
from jaenys.derivation import CueSessionDeriver, DerivedLayers, KeywordDetector
from jaenys.derivation import derive_layers
from jaenys.sql import sqlite

# ---- The part YOU populate: your vocabulary, your rules. -----------------

DETECTOR = KeywordDetector(
    keywords=("pin", "password", "card number"),
)

DERIVER = CueSessionDeriver(
    start_cues=("verify your identity",),
    end_cues=("verification complete",),
    timeout_minutes=30,
)

# ---- The rest is mechanics the toolkit ships. -----------------------------


def build_primary_db(path: Path) -> None:
    """Create the primary store: a raw 12-row support chat, nothing flagged yet."""

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
        ("agent", "2026-01-01 09:00", "Hi! How can I help you today?"),
        ("customer", "2026-01-01 09:01", "My card number is 4111 1111 1111 1111."),  # keyword
        ("agent", "2026-01-01 09:02", "No need for that! First, let's verify your identity."),
        ("customer", "2026-01-01 09:03", "Sure, go ahead."),
        ("agent", "2026-01-01 09:04", "What street did you grow up on?"),
        ("customer", "2026-01-01 09:05", "Elm Street."),
        ("agent", "2026-01-01 09:06", "That matches. Verification complete."),  # end cue
        ("customer", "2026-01-01 09:07", "Awesome, thanks!"),
        ("customer", "2026-01-01 09:20", "Oh -- my PIN is 9922 if you need it later."),  # keyword
        ("agent", "2026-01-01 11:00", "Wait -- let's verify your identity once more."),
        ("customer", "2026-01-01 11:02", "OK, ready."),
        ("customer", "2026-01-01 13:30", "Hello? Did you fall asleep?"),  # after the timeout
    ]
    conn.executemany("INSERT INTO records (sender, sent_at, body) VALUES (?, ?, ?)", rows)
    conn.commit()
    conn.close()


def run_derivation(primary_path: Path, span_path: Path) -> DerivedLayers:
    """The whole caller-owned pipeline step, now three calls long."""

    with closing(sqlite3.connect(primary_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT record_id, sent_at, body FROM records ORDER BY sent_at, record_id"
        ).fetchall()
    layers = derive_layers(
        rows,
        detector=DETECTOR,
        deriver=DERIVER,
        id_key="record_id",
        text_key="body",
        at_key="sent_at",
    )
    sqlite.write_flags(primary_path, layers)  # flags + drift witness
    sqlite.write_span_layer(span_path, layers)
    return layers


def serve_three_states(primary_path: Path, span_path: Path, member_ids: frozenset[int]) -> None:
    """Serve through the guard, exactly as in demo_sql_guard.py."""

    with closing(sqlite.open_readonly(primary_path)) as conn:
        all_ids = [row[0] for row in conn.execute("SELECT record_id FROM records ORDER BY 1")]
        with sqlite.attached_guard(conn, span_path) as guard:
            predicate = guard.predicate("r", include_blur=True)
            cursor = conn.execute(
                f"SELECT r.* FROM records r WHERE {predicate.sql} ORDER BY r.record_id",
                predicate.params,
            )
            cursor.row_factory = sqlite3.Row
            deliverable_rows = [dict(row) for row in cursor.fetchall()]

    annotated = annotate_rows(
        deliverable_rows, span_member_ids=member_ids, id_key="record_id", flag_key="sensitive"
    )
    served = {row["record_id"]: row for row in annotated}
    for record_id in all_ids:
        row = served.get(record_id)
        if row is None:
            print(f"  [HIDDEN ] id={record_id:>2}  (in-span; not served)")
        else:
            state = "BLUR" if row["blurred"] else "VISIBLE"
            print(f"  [{state:7}] id={record_id:>2}  {row['body']!r}")
    print()


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="jaenys_demo_") as tmp:
        tmp_path = Path(tmp)
        primary_path = tmp_path / "primary.db"
        span_path = tmp_path / "span.db"

        print("=== Step 1: build the raw transcript (no flags, no spans) ===")
        build_primary_db(primary_path)
        print("Primary store: 12 rows, sensitive = 0 everywhere.\n")

        print("=== Step 2: run the populated templates and write both layers ===")
        layers = run_derivation(primary_path, span_path)
        print(f"  Keywords {DETECTOR.keywords} flagged ids {sorted(layers.flagged_ids)}.")
        for number, session in enumerate(layers.sessions, start=1):
            ids = session.member_ids
            print(
                f"  Session {number}: rows {ids[0]}-{ids[-1]} "
                f"({len(ids)} rows, end_reason={session.end_reason!r})"
            )
        print("  Flags written to the primary store; span store rebuilt with the mirror.\n")

        print("=== Step 3: serve the three-state result ===")
        serve_three_states(primary_path, span_path, layers.span_member_ids)

        print("=== Step 4: a new message arrives; only the flag layer is updated ===")
        conn = sqlite3.connect(primary_path)
        conn.execute(
            "INSERT INTO records (sender, sent_at, body) VALUES (?, ?, ?)",
            ("customer", "2026-01-01 13:35", "Use password hunter2 for the shared doc."),
        )
        conn.commit()
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT record_id, body FROM records").fetchall()
        conn.close()
        flagged = DETECTOR.flagged_ids(rows, id_key="record_id", text_key="body")
        sqlite.write_flags(primary_path, flagged)
        print(f"  Flagged ids are now {sorted(flagged)}. Span store NOT re-derived yet.\n")

        print("=== Step 5: the engine refuses to serve stale isolation ===")
        try:
            with closing(sqlite.open_readonly(primary_path)) as conn:
                with sqlite.attached_guard(conn, span_path):
                    print("  (should not reach here)")
        except RedactionDriftError as exc:
            print(f"  RedactionDriftError raised, as expected:\n  -> {exc}\n")

        print("=== Step 6: re-run the derivation and serve again ===")
        layers = run_derivation(primary_path, span_path)
        serve_three_states(primary_path, span_path, layers.span_member_ids)


if __name__ == "__main__":
    main()
