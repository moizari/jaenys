"""Couchbase adapter for the store-agnostic visibility engine.

No driver import happens at module load time: the caller constructs its own
Couchbase ``Collection`` handles (``cluster.bucket(...).scope(...).
collection(...)``) and passes them in, along with a small caller-supplied
``scan`` callable (design rationale below).

Design choice: caller-supplied ``scan``, not N1QL strings
-----------------------------------------------------------

Couchbase has no single obvious "read every document in this collection"
call from the SDK surface alone: fetching all documents means either a N1QL
``SELECT`` (needs a query-capable cluster handle and index awareness) or the
newer collection-level KV-range/scan API. Baking a N1QL string into this
module would (a) push a specific Couchbase SDK version's query-execution
shape into a driver-import-free module, and (b) hard-code an index
assumption this library cannot verify.

Instead the constructor accepts a single ``scan`` callable, shared across
all three collections:

    scan(collection) -> Iterable[dict]

Given one of the three collection handles below, ``scan`` must return (or
yield) every document body in that collection as a plain ``dict`` (already
JSON-decoded -- whatever the caller's N1QL/KV-range call produces). The
caller owns the query technology entirely; this module only calls
``scan(collection)`` and reads dict fields out of the results. This keeps
the adapter a thin, storage-shape-only layer, matching the other adapters.

Storage shape
-------------

Three collection handles, any of which may be ``None`` if that layer is not
provisioned:

* ``records_collection`` -- documents ``{<record_id_column>: int,
  <flag_column>: 0/1, ...}``, one per record.
* ``span_members_collection`` -- documents ``{<span_member_id_column>:
  int, ...}``, one per record id belonging to a derived span.
* ``mirror_collection`` -- documents ``{<mirror_id_column>: int,
  <mirror_reason_column>: str, <mirror_flag_column>: 0/1, ...}``, mirroring
  the live flag layer as it was when spans were last derived.

Readiness marker
----------------

A full scan of an empty collection and a full scan of a nonexistent
collection both come back as "no documents" through ``scan()``, so
readiness cannot be inferred from scan results alone. The derivation
pipeline is the sole writer of an explicit meta document with id
``"_span_meta"`` inside ``span_members_collection``:

* the span layer is considered **derived** exactly when
  ``span_members_collection`` is provided (not ``None``) *and*
  ``span_members_collection.exists("_span_meta")`` reports the document
  exists;
* the mirror is considered **available** exactly when ``mirror_collection``
  is provided (not ``None``) and reachable -- there is no separate mirror
  meta document.  A provisioned-but-empty mirror therefore reports
  ``frozenset()`` ("nothing was flagged at derivation time"), which fails
  closed the moment live flags exist.  That refusal is the intended
  behavior: only the derivation pipeline should provision the mirror
  collection, and it writes the mirror documents in the same run.  Pass
  ``mirror_collection=None`` for genuinely mirror-less (legacy span-only)
  deployments instead of an empty collection.

The ``"_span_meta"`` document itself is skipped when collecting span-member
ids (it carries no ``span_member_id_column`` field so it is naturally
excluded, but it is also explicitly filtered by document id as a
belt-and-suspenders check for callers whose ``scan`` returns raw KV bodies
without ids attached).
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

from ..core import SchemaMapping, RedactionDriftError, coerce_flag, coerce_record_id
from .protocol import _fail_closed

__all__ = ["CouchbaseAdapter"]

_META_DOC_ID = "_span_meta"

ScanCallable = Callable[[Any], Iterable[dict]]


class CouchbaseAdapter:
    """Adapter over up to three Couchbase ``Collection`` handles."""

    def __init__(
        self,
        records_collection: Any,
        span_members_collection: Any | None,
        mirror_collection: Any | None,
        *,
        scan: ScanCallable,
        mapping: SchemaMapping = SchemaMapping(),
        name: str = "couchbase",
    ) -> None:
        self._records_collection = records_collection
        self._span_members_collection = span_members_collection
        self._mirror_collection = mirror_collection
        self._scan = scan
        self.mapping = mapping
        self.name = name

    # -- internals ---------------------------------------------------------------

    def _run_scan(self, collection: Any) -> list[dict]:
        with _fail_closed(self.name):
            return list(self._scan(collection))

    def _span_layer_ready(self) -> bool:
        if self._span_members_collection is None:
            return False
        with _fail_closed(self.name):
            result = self._span_members_collection.exists(_META_DOC_ID)
            return bool(getattr(result, "exists", result))

    def _required(self, doc: dict[str, Any], key: str, *, collection: str) -> Any:
        if key not in doc:
            raise RedactionDriftError(
                f"{self.name} store is missing required field {key!r} "
                f"in {collection}; refusing normal output."
            )
        return doc[key]

    def _record_flags(self) -> list[tuple[int, int]]:
        docs = self._run_scan(self._records_collection)
        rows: list[tuple[int, int]] = []
        for doc in docs:
            record_id = coerce_record_id(
                self._required(
                    doc, self.mapping.record_id_column, collection=self.mapping.record_table
                ),
                origin=self.name,
            )
            flag = coerce_flag(
                self._required(doc, self.mapping.flag_column, collection=self.mapping.record_table),
                origin=self.name,
            )
            rows.append((record_id, flag))
        return rows

    # -- StoreAdapter protocol -------------------------------------------------

    def span_member_ids(self) -> frozenset[int] | None:
        if not self._span_layer_ready():
            return None
        docs = self._run_scan(self._span_members_collection)
        id_field = self.mapping.span_member_id_column
        # The readiness marker (and any body that carries no member-id field)
        # contributes no id, so it is skipped rather than refused: skipping a
        # field-less body cannot un-hide a record.  A body that does carry the
        # field with a corrupt value still refuses through the coercer.  This
        # keeps the "document body" scan shape (no ids attached) working, where
        # the marker cannot be told apart by document id.
        return frozenset(
            coerce_record_id(doc[id_field], origin=self.name)
            for doc in docs
            if doc.get("id", doc.get("_id")) != _META_DOC_ID and id_field in doc
        )

    def flagged_ids(self) -> frozenset[int]:
        return frozenset(record_id for record_id, flag in self._record_flags() if flag == 1)

    def mirror_flagged_ids(self) -> frozenset[int] | None:
        if self._mirror_collection is None:
            return None
        docs = self._run_scan(self._mirror_collection)
        id_field = self.mapping.mirror_id_column
        reason_field = self.mapping.mirror_reason_column
        flag_field = self.mapping.mirror_flag_column
        return frozenset(
            coerce_record_id(
                self._required(doc, id_field, collection=self.mapping.mirror_table),
                origin=self.name,
            )
            for doc in docs
            if self._required(doc, reason_field, collection=self.mapping.mirror_table)
            == self.mapping.reason_flagged
            and coerce_flag(
                self._required(doc, flag_field, collection=self.mapping.mirror_table),
                origin=self.name,
            )
            == 1
        )

    def counts(self) -> dict[str, int]:
        rows = self._record_flags()
        total = len(rows)
        flagged = sum(1 for _record_id, flag in rows if flag == 1)
        return {"records": total, "flagged": flagged}
