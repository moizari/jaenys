#!/usr/bin/env python3
"""Runnable, stdlib-only walkthrough of wiring the guard into an app.

The other demos build and guard the two layers; this one shows the last
mile: an application serving users. Three rules make an integration safe:

  1. ONE CHOKEPOINT -- every read the app performs goes through a single
     data-access function that applies the guard. The library protects
     exactly the queries routed through it; a raw query that skips the
     predicate returns everything. Make skipping impossible by
     construction: no other code touches the store.
  2. BLUR IS THE UI'S JOB -- deliverable-but-flagged rows arrive marked
     ``"blurred": true``; the engine never alters text. The app decides
     what a blur looks like (fog, collapse, tap-to-reveal).
  3. A REFUSAL IS NOT A CRASH -- when the layers drift, the chokepoint
     raises RedactionDriftError. The route answers "temporarily unavailable"
     (a 503), the app stays up, and service resumes the moment the span
     derivation re-runs. Never catch-and-serve-anyway: the refusal IS the
     security feature.

Run with: python examples/demo_app_integration.py
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

from jaenys import RedactionDriftError
from jaenys.derivation import CueSessionDeriver, KeywordDetector, derive_layers
from jaenys.sql import sqlite

DETECTOR = KeywordDetector(keywords=("pin", "password", "card number"))
DERIVER = CueSessionDeriver(
    start_cues=("verify your identity",),
    end_cues=("verification complete",),
    timeout_minutes=30,
)


# ---- pipeline side (see demo_span_derivation.py for the full story) -------


def build_primary_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE records ("
        "record_id INTEGER PRIMARY KEY,"
        "sent_at TEXT NOT NULL,"
        "body TEXT NOT NULL,"
        "sensitive INTEGER NOT NULL DEFAULT 0)"
    )
    rows = [
        ("2026-01-01 09:00", "Morning! What do you need?"),
        ("2026-01-01 09:01", "My card number is 4111 1111 1111 1111."),  # -> BLUR
        ("2026-01-01 09:02", "Let's verify your identity first."),  # span start
        ("2026-01-01 09:03", "What street did you grow up on?"),
        ("2026-01-01 09:04", "Elm Street."),
        ("2026-01-01 09:05", "Thanks -- verification complete."),  # span end
        ("2026-01-01 09:06", "You're all set!"),
    ]
    conn.executemany("INSERT INTO records (sent_at, body) VALUES (?, ?)", rows)
    conn.commit()
    conn.close()


def run_derivation(primary_path: Path, span_path: Path) -> None:
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


# ---- app side: rules 1-3 ---------------------------------------------------


def fetch_conversation(primary_path: Path, span_path: Path) -> list[dict]:
    """Rule 1, the chokepoint: the ONLY code in this app that reads records.

    Every page, search, and export calls this; nothing else holds a
    connection. The guard verifies layer freshness before the query runs
    and raises RedactionDriftError instead of serving stale isolation.
    """

    with closing(sqlite.open_readonly(primary_path)) as conn:
        with sqlite.attached_guard(conn, span_path) as guard:
            predicate = guard.predicate("m", include_blur=True)
            cursor = conn.execute(
                f"SELECT m.* FROM records m WHERE {predicate.sql} ORDER BY m.record_id",
                predicate.params,
            )
            cursor.row_factory = sqlite3.Row
            return [dict(row) for row in cursor.fetchall()]


def get_messages_route(primary_path: Path, span_path: Path) -> dict:
    """A route as any web framework would write it: status code + JSON body.

    Rule 3 lives here: RedactionDriftError becomes a 503, not a 500 and not a
    silent fallback to unfiltered data.
    """

    try:
        rows = fetch_conversation(primary_path, span_path)
    except RedactionDriftError as exc:
        return {
            "status": 503,
            "body": {"error": "content temporarily unavailable", "detail": str(exc)},
        }
    return {
        "status": 200,
        "body": [
            {"id": row["record_id"], "text": row["body"], "blurred": bool(row["sensitive"])}
            for row in rows
        ],
    }


def render_like_a_ui(response: dict) -> None:
    """Rule 2: the frontend decides what a blur looks like."""

    if response["status"] != 200:
        print(f"  HTTP {response['status']}: {response['body']['error']}")
        print("  shown to the user as: 'This page is being re-checked, try again soon.'")
        return
    for item in response["body"]:
        shown = "[blurred -- tap to reveal]" if item["blurred"] else item["text"]
        print(f"  #{item['id']:>2}  {shown}")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="jaenys_demo_") as tmp:
        tmp_path = Path(tmp)
        primary_path = tmp_path / "primary.db"
        span_path = tmp_path / "span.db"

        print("=== Step 1: pipeline builds and derives both layers ===")
        build_primary_db(primary_path)
        run_derivation(primary_path, span_path)
        print("Primary store: 7 rows. Span store: derived and mirrored.\n")

        print("=== Step 2: the app serves a page through the chokepoint ===")
        response = get_messages_route(primary_path, span_path)
        print("Raw route output (what your frontend receives):")
        print(json.dumps(response, indent=2))
        print("Rendered:")
        render_like_a_ui(response)
        print()

        print("=== Step 3: what a BYPASSING query would have done ===")
        with closing(sqlite3.connect(primary_path)) as conn:
            total = conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]
        served = len(response["body"])
        print(f"  A raw query that skips the chokepoint returns all {total} rows --")
        print(f"  including the {total - served} rows the guard was excluding.")
        print("  The library protects queries routed through it; route ALL of them.\n")

        print("=== Step 4: drift happens (new flag, span layer not re-derived) ===")
        conn = sqlite3.connect(primary_path)
        conn.execute(
            "INSERT INTO records (sent_at, body) VALUES (?, ?)",
            ("2026-01-01 09:30", "Also note my PIN is 9922."),
        )
        conn.commit()
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT record_id, body FROM records").fetchall()
        conn.close()
        sqlite.write_flags(primary_path, DETECTOR.flagged_ids(rows, text_key="body"))
        print("  A new message was flagged; nobody re-ran the span derivation.\n")

        print("=== Step 5: the app stays up; the route answers 503 ===")
        render_like_a_ui(get_messages_route(primary_path, span_path))
        print()

        print("=== Step 6: operator re-runs the derivation; service resumes ===")
        run_derivation(primary_path, span_path)
        render_like_a_ui(get_messages_route(primary_path, span_path))


if __name__ == "__main__":
    main()
