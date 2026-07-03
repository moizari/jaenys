"""Command-line status report for a SQLite primary/span store pair.

This is intentionally the only thing the CLI does: load the guard,
verify the two sensitivity layers against each other, and print a
counts-only readiness report. It never prints record content -- only
``jaenys.sql.sqlite.status`` counts and, on refusal, the
guard's refusal message.

Exit code 0 means the report was produced AND the stores are healthy
(primary store found, layers in sync); anything else exits 1 so shell
pipelines like ``python -m jaenys ... && deploy`` fail closed.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

from .core import SchemaMapping, RedactionDriftError
from .sql import sqlite

__all__ = ["main"]


def _load_mapping_overrides(mapping_path: str | None) -> dict[str, Any]:
    if mapping_path is None:
        return {}
    try:
        handle = open(mapping_path, encoding="utf-8")
    except OSError as exc:
        raise RedactionDriftError(f"cannot read mapping file {mapping_path}: {exc}") from exc
    with handle:
        try:
            data = json.load(handle)
        except json.JSONDecodeError as exc:
            raise RedactionDriftError(f"{mapping_path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise RedactionDriftError(f"{mapping_path} must contain a JSON object of field overrides.")
    return data


def _build_mapping(overrides: dict[str, Any]) -> SchemaMapping:
    try:
        return SchemaMapping(**overrides)
    except TypeError as exc:
        raise RedactionDriftError(f"unknown SchemaMapping field(s) in mapping file: {exc}") from exc


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="jaenys",
        description=(
            "Check a SQLite primary/span store pair for redaction drift and "
            "print a counts-only readiness report: record and flag counts, "
            "visible/blur/hidden breakdown, and layer-sync state. Fail-closed: "
            "exits 0 only when both stores are found and the layers verify "
            "in sync."
        ),
    )
    parser.add_argument("--primary-db", required=True, help="Path to the primary SQLite store.")
    parser.add_argument(
        "--span-db", required=True, help="Path to the span-derivation SQLite store."
    )
    parser.add_argument(
        "--mapping",
        default=None,
        help=(
            "Optional path to a JSON file of SchemaMapping field overrides "
            '(e.g. {"record_table": "tickets"}). Unknown keys are an error.'
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        overrides = _load_mapping_overrides(args.mapping)
        mapping = _build_mapping(overrides)
        report = sqlite.status(args.primary_db, args.span_db, mapping=mapping)
    except RedactionDriftError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    # The report itself can carry a failure (missing primary store, missing
    # record table, layers out of sync).  A health check that prints the
    # problem but exits 0 passes `... && deploy`-style scripting silently.
    healthy = "error" not in report["primary_db"] and report.get("layers_in_sync") is True
    return 0 if healthy else 1
