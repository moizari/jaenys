"""Span-aware, fail-closed guard against redaction drift.

Redaction drift is the gap between what is marked sensitive and what is
actually being served: detection state and serving filter are two separate
pieces of state, and nothing in an ordinary pipeline notices when they
drift apart.  Jaenys refuses to serve whenever that sync cannot be
proven, so drift surfaces as a raised :class:`RedactionDriftError` instead
of a silent leak.

Sensitivity is tracked at two layers -- a live per-record flag and a
separately derived contiguous-span layer -- and every record resolves to one
of three states: ``VISIBLE``, ``BLUR`` (standalone flagged), or ``HIDDEN``
(inside a span).  The engine never silently serves stale isolation.

Backends:

* :mod:`jaenys.sql` -- SQL engines over PEP 249 connections
  (SQLite reference; PostgreSQL, MySQL, SQL Server, Oracle, ANSI fallback).
* :mod:`jaenys.adapters` -- non-SQL stores through a small
  4-method protocol (in-memory reference; MongoDB, Redis, DynamoDB,
  Couchbase, Firestore).

Producing the two layers is the caller's pipeline step, but
:mod:`jaenys.derivation` ships populate-the-template scaffolding
for it: a keyword flag detector and a cue-based session deriver you fill
with your own vocabulary.
"""

from .core import (
    BLUR,
    HIDDEN,
    VISIBLE,
    SchemaMapping,
    Guard,
    RedactionDriftError,
    adapter_status,
    annotate_rows,
    assert_adapter_current,
    classify,
    filter_visible_rows,
    load_guard_from_adapter,
    span_member_digest,
    verify_drift_witness,
    verify_guard_current,
    verify_layer_sync,
)

__version__ = "0.1.1"

__all__ = [
    "VISIBLE",
    "BLUR",
    "HIDDEN",
    "SchemaMapping",
    "Guard",
    "RedactionDriftError",
    "classify",
    "annotate_rows",
    "filter_visible_rows",
    "verify_layer_sync",
    "verify_drift_witness",
    "span_member_digest",
    "verify_guard_current",
    "load_guard_from_adapter",
    "assert_adapter_current",
    "adapter_status",
    "__version__",
]
