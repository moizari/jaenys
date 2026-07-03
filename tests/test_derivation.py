"""Tests for the populate-the-template derivation scaffolding.

Covers the two templates (KeywordDetector, CueSessionDeriver), the
DerivedLayers value object, and the SQLite writers -- including the
round-trip that matters: derive -> write both layers -> guard serves;
change flags without re-deriving -> guard refuses.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

import pytest

from jaenys import RedactionDriftError
from jaenys.derivation import (
    END_CUE,
    END_OF_DATA,
    TIMEOUT,
    CueSessionDeriver,
    DerivedLayers,
    KeywordDetector,
    Session,
    derive_layers,
)
from jaenys.sql import filter_visible_ids
from jaenys.sql import sqlite as sq


def _rows(*items: tuple[int, str, str | None]) -> list[dict]:
    return [{"record_id": rid, "at": at, "text": text} for rid, at, text in items]


TRANSCRIPT = _rows(
    (1, "2026-01-01 09:00", "hello there"),
    (2, "2026-01-01 09:01", "my PIN is 4471"),
    (3, "2026-01-01 09:02", "let's verify your identity"),
    (4, "2026-01-01 09:03", "sure"),
    (5, "2026-01-01 09:04", "what street did you grow up on?"),
    (6, "2026-01-01 09:05", "elm street"),
    (7, "2026-01-01 09:06", "verification complete"),
    (8, "2026-01-01 09:07", "thanks!"),
    (9, "2026-01-01 11:00", "verify your identity once more"),
    (10, "2026-01-01 11:02", "ok ready"),
    (11, "2026-01-01 13:30", "hello? anyone?"),
)

DETECTOR = KeywordDetector(("pin", "password"))
DERIVER = CueSessionDeriver(
    start_cues=("verify your identity",),
    end_cues=("verification complete",),
    timeout_minutes=30,
)


class TestKeywordDetector:
    def test_flags_keyword_case_insensitively(self):
        assert DETECTOR.matches("My PIN is 4471")
        assert DETECTOR.matches("PASSWORD: hunter2")
        assert not DETECTOR.matches("nothing to see")

    def test_case_sensitive_mode(self):
        detector = KeywordDetector(("PIN",), case_sensitive=True)
        assert detector.matches("my PIN")
        assert not detector.matches("my pin")

    def test_none_text_never_matches(self):
        assert not DETECTOR.matches(None)

    def test_extra_callable_is_additive(self):
        detector = KeywordDetector(("pin",), extra=lambda text: "4111" in text)
        assert detector.matches("my pin")
        assert detector.matches("card 4111 1111")
        assert not detector.matches("hello")

    def test_extra_alone_is_enough(self):
        detector = KeywordDetector(extra=lambda text: text.isdigit())
        assert detector.matches("12345")

    def test_unpopulated_detector_is_refused(self):
        with pytest.raises(RedactionDriftError):
            KeywordDetector()

    def test_bare_string_keywords_are_refused(self):
        with pytest.raises(RedactionDriftError):
            KeywordDetector("pin")

    def test_empty_keyword_is_refused(self):
        with pytest.raises(RedactionDriftError):
            KeywordDetector(("pin", "  "))

    def test_non_callable_extra_is_refused(self):
        with pytest.raises(RedactionDriftError):
            KeywordDetector(("pin",), extra="not callable")

    def test_flagged_ids_over_rows(self):
        assert DETECTOR.flagged_ids(TRANSCRIPT) == frozenset({2})

    def test_flagged_ids_with_custom_keys(self):
        rows = [{"id": 7, "body": "the password is swordfish"}]
        detector = KeywordDetector(("password",))
        assert detector.flagged_ids(rows, id_key="id", text_key="body") == frozenset({7})

    def test_missing_text_key_is_refused(self):
        with pytest.raises(RedactionDriftError):
            DETECTOR.flagged_ids([{"record_id": 1}])

    def test_non_string_text_is_refused(self):
        with pytest.raises(RedactionDriftError):
            DETECTOR.flagged_ids([{"record_id": 1, "text": 42}])

    def test_junk_record_id_is_refused(self):
        with pytest.raises(RedactionDriftError):
            DETECTOR.flagged_ids([{"record_id": "junk", "text": "my pin"}])


class TestCueSessionDeriver:
    def test_end_cue_session_is_contiguous(self):
        sessions = DERIVER.derive(TRANSCRIPT, at_key="at")
        assert sessions[0].member_ids == (3, 4, 5, 6, 7)
        assert sessions[0].end_reason == END_CUE

    def test_timeout_closes_before_the_gap(self):
        sessions = DERIVER.derive(TRANSCRIPT, at_key="at")
        assert sessions[1].member_ids == (9, 10)
        assert sessions[1].end_reason == TIMEOUT
        # the late row (11) is not a member of any session
        assert 11 not in {m for s in sessions for m in s.member_ids}

    def test_late_row_can_open_a_new_session(self):
        rows = _rows(
            (1, "2026-01-01 09:00", "verify your identity"),
            (2, "2026-01-01 09:01", "ok"),
            (3, "2026-01-01 12:00", "verify your identity again"),
            (4, "2026-01-01 12:01", "verification complete"),
        )
        sessions = DERIVER.derive(rows, at_key="at")
        assert [s.member_ids for s in sessions] == [(1, 2), (3, 4)]
        assert [s.end_reason for s in sessions] == [TIMEOUT, END_CUE]

    def test_open_session_closes_at_end_of_data(self):
        rows = _rows((1, "2026-01-01 09:00", "verify your identity"), (2, "2026-01-01 09:01", "ok"))
        sessions = DERIVER.derive(rows, at_key="at")
        assert sessions[0].end_reason == END_OF_DATA

    def test_start_row_never_also_closes(self):
        deriver = CueSessionDeriver(start_cues=("verify",), end_cues=("verify",))
        rows = [{"record_id": 1, "text": "verify"}, {"record_id": 2, "text": "verify"}]
        sessions = deriver.derive(rows)
        # row 1 opens; row 2 (also matching the end cue) closes.
        assert sessions == (Session((1, 2), END_CUE),)

    def test_no_timeout_means_timestamps_unread(self):
        deriver = CueSessionDeriver(start_cues=("verify",), end_cues=("complete",))
        rows = [{"record_id": 1, "text": "verify"}, {"record_id": 2, "text": "complete"}]
        assert deriver.derive(rows)[0].member_ids == (1, 2)

    def test_datetime_objects_accepted(self):
        rows = [
            {"record_id": 1, "at": datetime(2026, 1, 1, 9, 0), "text": "verify your identity"},
            {"record_id": 2, "at": datetime(2026, 1, 1, 9, 1), "text": "verification complete"},
        ]
        assert DERIVER.derive(rows, at_key="at")[0].end_reason == END_CUE

    def test_timeout_without_at_key_is_refused(self):
        with pytest.raises(RedactionDriftError):
            DERIVER.derive(TRANSCRIPT)

    def test_unparseable_timestamp_is_refused(self):
        rows = [{"record_id": 1, "at": "not a time", "text": "verify your identity"}]
        with pytest.raises(RedactionDriftError):
            DERIVER.derive(rows, at_key="at")

    def test_decreasing_timestamps_are_refused(self):
        rows = _rows(
            (1, "2026-01-01 09:05", "verify your identity"),
            (2, "2026-01-01 09:00", "ok"),
        )
        with pytest.raises(RedactionDriftError):
            DERIVER.derive(rows, at_key="at")

    def test_unpopulated_deriver_is_refused(self):
        with pytest.raises(RedactionDriftError):
            CueSessionDeriver(end_cues=("done",))

    def test_deriver_without_any_end_mechanism_is_refused(self):
        """Start cues alone would hide everything after the first match."""

        with pytest.raises(RedactionDriftError, match="close a session"):
            CueSessionDeriver(start_cues=("verify",))
        with pytest.raises(RedactionDriftError, match="close a session"):
            CueSessionDeriver(is_start=lambda text: True)

    def test_is_start_callable_opens_a_session(self):
        deriver = CueSessionDeriver(is_start=lambda text: text.startswith(">>"), end_cues=("done",))
        sessions = deriver.derive([{"record_id": 1, "text": ">> begin"}])
        assert sessions == (Session((1,), END_OF_DATA),)

    def test_mixed_aware_and_naive_timestamps_refuse_cleanly(self):
        """Aware-vs-naive must refuse via RedactionDriftError, not a raw TypeError."""

        rows = _rows(
            (1, "2026-01-01 09:00", "verify your identity"),
            (2, "2026-01-01 09:05:00+05:00", "ok"),
        )
        with pytest.raises(RedactionDriftError, match="timezone"):
            DERIVER.derive(rows, at_key="at")

    def test_zulu_suffix_timestamps_accepted(self):
        """A trailing Z parses on every supported interpreter (3.10 included)."""

        rows = _rows(
            (1, "2026-01-01T09:00:00Z", "verify your identity"),
            (2, "2026-01-01T09:01:00Z", "verification complete"),
        )
        sessions = DERIVER.derive(rows, at_key="at")
        assert sessions[0].member_ids == (1, 2)
        assert sessions[0].end_reason == END_CUE

    def test_bad_timeout_is_refused(self):
        for bad in (0, -5, "30", True):
            with pytest.raises(RedactionDriftError):
                CueSessionDeriver(start_cues=("verify",), timeout_minutes=bad)

    def test_non_finite_timeout_is_refused(self):
        """nan/inf pass the <= 0 check yet never fire, hiding to end of data."""

        for bad in (float("nan"), float("inf")):
            with pytest.raises(RedactionDriftError, match="finite"):
                CueSessionDeriver(start_cues=("verify",), timeout_minutes=bad)

    def test_repeated_id_inside_open_session_is_refused(self):
        deriver = CueSessionDeriver(start_cues=("verify",), end_cues=("done",))
        rows = [
            {"record_id": 1, "text": "verify"},
            {"record_id": 2, "text": "midway"},
            {"record_id": 2, "text": "done"},  # duplicate id in one session
        ]
        with pytest.raises(RedactionDriftError, match="repeat a member id"):
            deriver.derive(rows)


class TestDerivedLayers:
    def test_derive_layers_combines_both_templates(self):
        layers = derive_layers(TRANSCRIPT, detector=DETECTOR, deriver=DERIVER, at_key="at")
        assert layers.flagged_ids == frozenset({2})
        assert layers.span_member_ids == frozenset({3, 4, 5, 6, 7, 9, 10})

    def test_hand_rolled_layers_coerce_ids(self):
        layers = DerivedLayers(
            flagged_ids=frozenset({"3"}), sessions=(Session(("5", 6), "end_cue"),)
        )
        assert layers.flagged_ids == frozenset({3})
        assert layers.sessions[0].member_ids == (5, 6)

    def test_empty_session_is_refused(self):
        with pytest.raises(RedactionDriftError):
            Session((), "end_cue")

    def test_duplicate_member_id_is_refused(self):
        with pytest.raises(RedactionDriftError, match="repeat a member id"):
            Session((5, 6, 5), "end_cue")

    def test_non_session_entries_are_refused(self):
        with pytest.raises(RedactionDriftError):
            DerivedLayers(flagged_ids=frozenset(), sessions=({"member_ids": (1,)},))


def _build_primary(path: Path, bodies: list[str]) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE records ("
        "record_id INTEGER PRIMARY KEY,"
        "sent_at TEXT NOT NULL,"
        "body TEXT NOT NULL,"
        "sensitive INTEGER NOT NULL DEFAULT 0)"
    )
    conn.executemany(
        "INSERT INTO records (sent_at, body) VALUES (?, ?)",
        [(f"2026-01-01 09:{i:02d}", body) for i, body in enumerate(bodies)],
    )
    conn.commit()
    conn.close()


def _flags_on_disk(path: Path) -> dict[int, int]:
    with closing(sqlite3.connect(path)) as conn:
        return dict(conn.execute("SELECT record_id, sensitive FROM records"))


class TestSqliteWriters:
    def test_write_flags_sets_and_clears(self, tmp_path: Path):
        primary = tmp_path / "primary.db"
        _build_primary(primary, ["a", "b", "c"])
        sq.write_flags(primary, {2})
        assert _flags_on_disk(primary) == {1: 0, 2: 1, 3: 0}
        # a later derivation drops the flag again: stale 1s must be cleared
        sq.write_flags(primary, {3})
        assert _flags_on_disk(primary) == {1: 0, 2: 0, 3: 1}

    def test_write_flags_accepts_open_connection(self, tmp_path: Path):
        primary = tmp_path / "primary.db"
        _build_primary(primary, ["a", "b"])
        with closing(sqlite3.connect(primary)) as conn:
            sq.write_flags(conn, {1})
        assert _flags_on_disk(primary) == {1: 1, 2: 0}

    def test_write_flags_refuses_unknown_ids_and_changes_nothing(self, tmp_path: Path):
        primary = tmp_path / "primary.db"
        _build_primary(primary, ["a", "b"])
        sq.write_flags(primary, {1})
        with pytest.raises(RedactionDriftError):
            sq.write_flags(primary, {1, 99})
        # the failed call rolled back: the earlier state is intact
        assert _flags_on_disk(primary) == {1: 1, 2: 0}

    def test_write_flags_refuses_junk_ids(self, tmp_path: Path):
        primary = tmp_path / "primary.db"
        _build_primary(primary, ["a"])
        with pytest.raises(RedactionDriftError):
            sq.write_flags(primary, {"junk"})

    def test_write_flags_refuses_non_unique_record_ids(self, tmp_path: Path):
        """Duplicated record_id rows update more rows than ids; refuse, not misreport."""

        primary = tmp_path / "primary.db"
        conn = sqlite3.connect(primary)
        conn.execute(
            "CREATE TABLE records ("
            "record_id INTEGER NOT NULL,"  # no PRIMARY KEY: duplicates allowed
            "sent_at TEXT NOT NULL,"
            "body TEXT NOT NULL,"
            "sensitive INTEGER NOT NULL DEFAULT 0)"
        )
        conn.executemany(
            "INSERT INTO records (record_id, sent_at, body) VALUES (?, ?, ?)",
            [(1, "2026-01-01 09:00", "a"), (1, "2026-01-01 09:01", "b")],
        )
        conn.commit()
        conn.close()
        with pytest.raises(RedactionDriftError, match="not unique"):
            sq.write_flags(primary, {1})

    def test_full_round_trip_serves_and_refuses_on_drift(self, tmp_path: Path):
        primary = tmp_path / "primary.db"
        span = tmp_path / "span.db"
        _build_primary(
            primary,
            [
                "hello",
                "my pin is 4471",
                "let's verify your identity",
                "elm street",
                "verification complete",
                "bye",
            ],
        )
        with closing(sqlite3.connect(primary)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT record_id, sent_at, body FROM records").fetchall()
        layers = derive_layers(
            rows,
            detector=DETECTOR,
            deriver=DERIVER,
            id_key="record_id",
            text_key="body",
            at_key="sent_at",
        )
        sq.write_flags(primary, layers)  # layers form: flags + drift witness
        sq.write_span_layer(span, layers)

        with closing(sq.open_readonly(primary)) as conn:
            with sq.attached_guard(conn, span) as guard:
                predicate = guard.predicate("r", include_blur=True)
                served = {
                    row[0]
                    for row in conn.execute(
                        f"SELECT r.record_id FROM records r WHERE {predicate.sql}",
                        predicate.params,
                    )
                }
        assert served == {1, 2, 6}  # 3-5 hidden in-span; 2 is BLUR, kept

        # flag edit without re-derivation -> refusal
        sq.write_flags(primary, layers.flagged_ids | {6})
        with pytest.raises(RedactionDriftError):
            with closing(sq.open_readonly(primary)) as conn:
                with sq.attached_guard(conn, span):
                    pass

        # re-derive (hand-rolled layers this time) -> restored
        sq.write_span_layer(
            span,
            DerivedLayers(flagged_ids=layers.flagged_ids | {6}, sessions=layers.sessions),
        )
        with closing(sq.open_readonly(primary)) as conn:
            with sq.attached_guard(conn, span) as guard:
                predicate = guard.predicate("r")  # strictly-VISIBLE surface
                served = {
                    row[0]
                    for row in conn.execute(
                        f"SELECT r.record_id FROM records r WHERE {predicate.sql}",
                        predicate.params,
                    )
                }
        assert served == {1}  # 2 and 6 are BLUR and dropped without include_blur

    def test_write_span_layer_creates_missing_store(self, tmp_path: Path):
        span = tmp_path / "fresh_span.db"
        layers = DerivedLayers(flagged_ids=frozenset({1}), sessions=(Session((2, 3), "end_cue"),))
        sq.write_span_layer(span, layers)
        with closing(sqlite3.connect(span)) as conn:
            members = set(conn.execute("SELECT span_id, record_id FROM span_members"))
            mirror = set(
                conn.execute("SELECT record_id, copy_reason, source_flag FROM sensitive_records")
            )
        assert members == {(1, 2), (1, 3)}
        assert mirror == {(1, "flagged", 1), (2, "span", 1), (3, "span", 1)}

    def test_write_span_layer_respects_custom_mapping(self, tmp_path: Path):
        from jaenys import SchemaMapping

        mapping = SchemaMapping(
            span_member_table="hidden_members",
            span_member_id_column="msg_id",
            mirror_table="mirror_rows",
            mirror_id_column="msg_id",
        )
        span = tmp_path / "span.db"
        layers = DerivedLayers(flagged_ids=frozenset(), sessions=(Session((4,), "timeout"),))
        sq.write_span_layer(span, layers, mapping=mapping)
        with closing(sqlite3.connect(span)) as conn:
            members = set(conn.execute("SELECT msg_id FROM hidden_members"))
        assert members == {(4,)}


SPAN_ONLY_BODIES = [
    "hello",
    "let's verify your identity",
    "elm street",
    "verification complete",
    "bye",
]


class TestDerivationMarker:
    """The primary-store drift witness: the witness that survives span-store loss.

    Scenario under test is the zero-flag hole: a dataset whose sensitivity
    is entirely span-based (cues matched, no keyword flags).  Without the
    marker, losing the span store silently un-hides every in-span row,
    because the flag-mirror freshness check is vacuous with zero flags.
    """

    def _write_layers(self, tmp_path: Path) -> tuple[Path, Path, DerivedLayers]:
        primary = tmp_path / "primary.db"
        span = tmp_path / "span.db"
        _build_primary(primary, SPAN_ONLY_BODIES)
        with closing(sqlite3.connect(primary)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT record_id, sent_at, body FROM records").fetchall()
        layers = derive_layers(
            rows,
            detector=DETECTOR,
            deriver=DERIVER,
            id_key="record_id",
            text_key="body",
            at_key="sent_at",
        )
        assert layers.flagged_ids == frozenset()  # the zero-flag scenario
        assert layers.span_member_ids == frozenset({2, 3, 4})
        sq.write_flags(primary, layers)
        sq.write_span_layer(span, layers)
        return primary, span, layers

    def test_marker_recorded_in_primary(self, tmp_path: Path):
        primary, _span, _ = self._write_layers(tmp_path)
        with closing(sqlite3.connect(primary)) as conn:
            row = conn.execute("SELECT layer, member_count FROM span_store_meta").fetchone()
        assert row == ("span", 3)

    def test_intact_pair_serves_with_span_hidden(self, tmp_path: Path):
        primary, span, _ = self._write_layers(tmp_path)
        with closing(sq.open_readonly(primary)) as conn:
            with sq.attached_guard(conn, span) as guard:
                predicate = guard.predicate("r")
                served = {
                    row[0]
                    for row in conn.execute(
                        f"SELECT r.record_id FROM records r WHERE {predicate.sql}",
                        predicate.params,
                    )
                }
        assert served == {1, 5}

    def test_lost_span_store_refuses_even_with_zero_flags(self, tmp_path: Path):
        primary, span, _ = self._write_layers(tmp_path)
        span.unlink()  # the loss scenario: nothing is flagged, store gone
        with closing(sq.open_readonly(primary)) as conn:
            with pytest.raises(RedactionDriftError, match="records a span derivation"):
                with sq.attached_guard(conn, span):
                    pass

    def test_zero_byte_span_store_refuses_with_marker(self, tmp_path: Path):
        primary, span, _ = self._write_layers(tmp_path)
        span.write_bytes(b"")
        with closing(sq.open_readonly(primary)) as conn:
            with pytest.raises(RedactionDriftError, match="records a span derivation"):
                with sq.attached_guard(conn, span):
                    pass

    def test_emptied_span_tables_refuse_with_marker(self, tmp_path: Path):
        """Valid span store, tables intact, rows gone: count mismatch refuses."""

        primary, span, _ = self._write_layers(tmp_path)
        with closing(sqlite3.connect(span)) as conn:
            with conn:
                conn.execute("DELETE FROM span_members")
                conn.execute("DELETE FROM sensitive_records WHERE copy_reason = 'span'")
        with closing(sq.open_readonly(primary)) as conn:
            with pytest.raises(RedactionDriftError, match="does not match the derivation"):
                with sq.attached_guard(conn, span):
                    pass

    def test_materialized_guard_checks_marker(self, tmp_path: Path):
        primary, span, _ = self._write_layers(tmp_path)
        span.unlink()
        guard = sq.load_guard(span)  # not-ready guard, empty member set
        with closing(sq.open_readonly(primary)) as conn:
            with pytest.raises(RedactionDriftError, match="records a span derivation"):
                filter_visible_ids(conn, [1, 2, 3], guard=guard)

    def test_bare_id_write_flags_records_no_marker(self, tmp_path: Path):
        primary = tmp_path / "primary.db"
        _build_primary(primary, ["a", "b"])
        sq.write_flags(primary, {1})
        with closing(sqlite3.connect(primary)) as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'span_store_meta'"
            ).fetchone()
        assert row is None

    def test_bare_id_write_flags_clears_stale_witness(self, tmp_path: Path):
        # A DerivedLayers write records a witness; a later bare-id write carries
        # no span membership, so it clears the span witness row rather than
        # leave one a rewritten span layer would then be judged stale against.
        primary, _span, _ = self._write_layers(tmp_path)
        sq.write_flags(primary, {1})
        with closing(sqlite3.connect(primary)) as conn:
            row = conn.execute(
                "SELECT member_count FROM span_store_meta WHERE layer = 'span'"
            ).fetchone()
        assert row is None

    def test_meta_table_none_disables_marker(self, tmp_path: Path):
        from jaenys import SchemaMapping

        mapping = SchemaMapping(meta_table=None)
        primary = tmp_path / "primary.db"
        span = tmp_path / "span.db"
        _build_primary(primary, SPAN_ONLY_BODIES)
        with closing(sqlite3.connect(primary)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT record_id, sent_at, body FROM records").fetchall()
        layers = derive_layers(
            rows,
            detector=DETECTOR,
            deriver=DERIVER,
            id_key="record_id",
            text_key="body",
            at_key="sent_at",
        )
        sq.write_flags(primary, layers, mapping=mapping)
        sq.write_span_layer(span, layers, mapping=mapping)
        span.unlink()
        # Zero flags + no marker: explicit opt-out keeps pre-marker semantics.
        with closing(sq.open_readonly(primary)) as conn:
            with sq.attached_guard(conn, span, mapping=mapping) as guard:
                assert guard.span_layer_ready is False

    def test_corrupt_marker_refuses(self, tmp_path: Path):
        primary, span, _ = self._write_layers(tmp_path)
        with closing(sqlite3.connect(primary)) as conn:
            with conn:
                conn.execute("UPDATE span_store_meta SET member_count = 'junk'")
        with closing(sq.open_readonly(primary)) as conn:
            with pytest.raises(RedactionDriftError, match="span member count"):
                with sq.attached_guard(conn, span):
                    pass

    def test_status_and_cli_reflect_lost_store(self, tmp_path: Path):
        primary, span, _ = self._write_layers(tmp_path)
        span.unlink()
        report = sq.status(primary, span)
        assert report["layers_in_sync"] is False
        assert "span derivation" in report["refusal"]
        assert report["guard"]["drift_witness"] == {"member_count": 3}
