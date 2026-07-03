"""Status reporting: counts-only, read-only, sync-aware -- plus the CLI."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from jaenys.cli import main as cli_main
from jaenys.sql import sqlite
from tests.conftest import build_primary_db, create_mirror_table, create_span_member_table


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_pair(primary: Path, span: Path) -> list[int]:
    """8 records: id 2 standalone flagged; ids 5-7 a span (5 also flagged)."""

    ids = build_primary_db(primary, [0, 1, 0, 0, 1, 0, 0, 0])
    span_ids = ids[4:7]
    flagged_ids = [ids[1], ids[4]]
    create_span_member_table(span, span_ids)
    create_mirror_table(span, flagged_ids=flagged_ids, span_ids=span_ids)
    return ids


def test_status_counts_three_states(store_pair: tuple[Path, Path]) -> None:
    primary, span = store_pair
    _build_pair(primary, span)

    report = sqlite.status(primary, span)

    assert report["primary_db"]["records"] == 8
    assert report["primary_db"]["flagged"] == 2
    assert report["span_db"]["ready"] is True
    assert report["span_db"]["unique_span_member_ids"] == 3
    guard = report["guard"]
    # VISIBLE: 8 total - 2 flagged - 2 span-only (ids 6, 7; id 5 is flagged too)
    assert guard["visible_normal_records"] == 4
    # BLUR: flagged AND outside every span = id 2 only (id 5 is in-span).
    assert guard["blurred_standalone"] == 1
    assert guard["hidden_in_span"] == 3
    assert guard["deliverable_with_blur"] == 5
    assert guard["excluded_total"] == 4
    assert guard["span_layer_ready"] is True
    assert "span_members" in guard["span_sources"]
    assert report["layers_in_sync"] is True
    assert "refusal" not in report


def test_status_emits_no_record_content(store_pair: tuple[Path, Path]) -> None:
    primary, span = store_pair
    _build_pair(primary, span)

    report = sqlite.status(primary, span)

    assert "synthetic ticket line" not in json.dumps(report)


def test_status_is_read_only(store_pair: tuple[Path, Path]) -> None:
    primary, span = store_pair
    _build_pair(primary, span)
    before = (_file_hash(primary), _file_hash(span))

    sqlite.status(primary, span)

    assert (_file_hash(primary), _file_hash(span)) == before


def test_status_reports_drift_without_raising(store_pair: tuple[Path, Path]) -> None:
    primary, span = store_pair
    ids = _build_pair(primary, span)
    conn = sqlite3.connect(primary)
    conn.execute("UPDATE records SET sensitive = 1 WHERE record_id = ?", (ids[0],))
    conn.commit()
    conn.close()

    report = sqlite.status(primary, span)

    assert report["layers_in_sync"] is False
    assert "rebuild the span derivation layer" in report["refusal"]


def test_status_missing_primary(store_pair: tuple[Path, Path]) -> None:
    primary, span = store_pair
    _build_pair(primary, span)
    missing = primary.parent / "nowhere.db"

    report = sqlite.status(missing, span)

    assert report["primary_db"]["exists"] is False
    assert report["guard"] is None
    assert "error" in report["primary_db"]


def test_status_not_ready_span_store(store_pair: tuple[Path, Path]) -> None:
    primary, span = store_pair
    build_primary_db(primary, [0, 0, 0])  # span store stays a 0-byte placeholder

    report = sqlite.status(primary, span)

    assert report["span_db"]["ready"] is False
    assert report["guard"]["span_layer_ready"] is False
    # Nothing is flagged, so the flag-filter-only guard is still in sync.
    assert report["layers_in_sync"] is True


def test_cli_prints_json_report(
    store_pair: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    primary, span = store_pair
    _build_pair(primary, span)

    exit_code = cli_main(["--primary-db", str(primary), "--span-db", str(span)])

    assert exit_code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["primary_db"]["records"] == 8
    assert report["layers_in_sync"] is True


def test_cli_missing_primary_exits_nonzero(
    store_pair: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """The report names the problem AND the exit code fails `... && deploy`."""

    primary, span = store_pair
    _build_pair(primary, span)
    missing = primary.parent / "nowhere.db"

    exit_code = cli_main(["--primary-db", str(missing), "--span-db", str(span)])

    assert exit_code == 1
    report = json.loads(capsys.readouterr().out)
    assert report["primary_db"]["error"] == "primary database not found"


def test_cli_out_of_sync_layers_exit_nonzero(
    store_pair: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    primary, span = store_pair
    ids = _build_pair(primary, span)
    conn = sqlite3.connect(primary)
    conn.execute("UPDATE records SET sensitive = 1 WHERE record_id = ?", (ids[0],))
    conn.commit()
    conn.close()

    exit_code = cli_main(["--primary-db", str(primary), "--span-db", str(span)])

    assert exit_code == 1
    report = json.loads(capsys.readouterr().out)
    assert report["layers_in_sync"] is False


def test_cli_mapping_overrides(
    tmp_path: Path, store_pair: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    primary, span = store_pair
    _build_pair(primary, span)
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(json.dumps({"record_table": "records"}), encoding="utf-8")

    exit_code = cli_main(
        ["--primary-db", str(primary), "--span-db", str(span), "--mapping", str(mapping_path)]
    )

    assert exit_code == 0


def test_cli_rejects_unknown_mapping_key(
    tmp_path: Path, store_pair: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    primary, span = store_pair
    _build_pair(primary, span)
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(json.dumps({"no_such_field": "x"}), encoding="utf-8")

    exit_code = cli_main(
        ["--primary-db", str(primary), "--span-db", str(span), "--mapping", str(mapping_path)]
    )

    assert exit_code == 1
    assert "unknown SchemaMapping field" in capsys.readouterr().err


def test_cli_unsafe_mapping_identifier_fails(
    tmp_path: Path, store_pair: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    primary, span = store_pair
    _build_pair(primary, span)
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(
        json.dumps({"record_table": "records; DROP TABLE records"}), encoding="utf-8"
    )

    exit_code = cli_main(
        ["--primary-db", str(primary), "--span-db", str(span), "--mapping", str(mapping_path)]
    )

    assert exit_code == 1
    assert "unsafe" in capsys.readouterr().err
