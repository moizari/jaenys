"""MongoDB adapter for the store-agnostic visibility engine.

No driver import happens at module load time: the caller constructs its own
``pymongo`` client/database objects and passes them in.

Storage shape
-------------

Two ``pymongo.database.Database`` handles are accepted -- ``primary_db`` and
``span_db`` -- which may be the same database object when one deployment hosts
both layers.

* **Records** (primary store): collection named ``mapping.record_table``.
  Each document carries the record id under ``mapping.record_id_column`` and
  the live per-record flag (0/1) under ``mapping.flag_column``.
* **Span members** (span store): collection named
  ``mapping.span_member_table``. Each document is
  ``{<span_member_id_column>: int, ...}`` -- one document per record id that
  belongs to a derived span (duplicates across multiple spans are fine; ids
  are deduplicated when read).
* **Mirror** (span store): collection named ``mapping.mirror_table``. Each
  document is ``{<mirror_id_column>: int, <mirror_reason_column>: str,
  <mirror_flag_column>: 0/1}``, mirroring the live flag layer as it was when
  spans were last derived.

Readiness marker
----------------

MongoDB cannot distinguish "collection never created" from "collection
exists but is empty" via a simple ``find()`` -- both return no documents. The
derivation pipeline is therefore the sole writer of readiness:

* the span layer is considered **derived** exactly when
  ``mapping.span_member_table`` appears in
  ``span_db.list_collection_names()`` (an empty collection still counts: the
  pipeline creates it explicitly once it has run, even with zero members);
* the mirror is considered **available** exactly when
  ``mapping.mirror_table`` appears in ``span_db.list_collection_names()``.

``span_member_ids()`` returns ``None`` (never derived) only when the span
member collection does not exist at all; it returns ``frozenset()`` when the
collection exists but is empty. The same rule applies to
``mirror_flagged_ids()`` against the mirror collection.

``visibility_filter`` is the Mongo analogue of
:class:`jaenys.sql.guard.SqlPredicate`: given a span-member id set it
returns a native query dict suitable for ``collection.find(...)``.
"""

from __future__ import annotations

from typing import Any, Iterable

from ..core import SchemaMapping, RedactionDriftError, coerce_flag, coerce_record_id
from .protocol import _fail_closed, _reject_unsafe_document_name, _validate_mapped_names

__all__ = ["MongoDBAdapter"]


def _reject_unsafe_mongo_name(name: str, *, kind: str) -> None:
    """MongoDB gives ``$`` and ``.`` special meaning in field/collection names.

    Reject them up front rather than let a crafted mapping produce a query
    operator injection or an unintended nested-field lookup.
    """

    _reject_unsafe_document_name(name, kind=kind, store="MongoDB")


class MongoDBAdapter:
    """Adapter over two ``pymongo`` ``Database`` handles (may be identical)."""

    def __init__(
        self,
        primary_db: Any,
        span_db: Any | None = None,
        *,
        mapping: SchemaMapping = SchemaMapping(),
        name: str = "mongodb",
    ) -> None:
        self._primary_db = primary_db
        self._span_db = span_db if span_db is not None else primary_db
        self.mapping = mapping
        self.name = name
        _validate_mapped_names(mapping, store="MongoDB")

    # -- internals -----------------------------------------------------------

    def _records(self) -> Any:
        return self._primary_db[self.mapping.record_table]

    def _span_members(self) -> Any:
        return self._span_db[self.mapping.span_member_table]

    def _mirror(self) -> Any:
        return self._span_db[self.mapping.mirror_table]

    def _collection_names(self, db: Any) -> set[str]:
        with _fail_closed(self.name):
            return set(db.list_collection_names())

    def _docs(self, collection: Any, query: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        with _fail_closed(self.name):
            return list(collection.find(query or {}))

    def _required(self, doc: dict[str, Any], key: str, *, collection: str) -> Any:
        if key not in doc:
            raise RedactionDriftError(
                f"{self.name} store is missing required field {key!r} "
                f"in {collection}; refusing normal output."
            )
        return doc[key]

    def _distinct_required_ids(
        self, collection: Any, collection_name: str, key: str, query: dict[str, Any] | None = None
    ) -> frozenset[int]:
        values = [
            self._required(doc, key, collection=collection_name)
            for doc in self._docs(collection, query)
        ]
        return frozenset(coerce_record_id(value, origin=self.name) for value in values)

    def _record_flags(self) -> list[tuple[int, int]]:
        rows: list[tuple[int, int]] = []
        for doc in self._docs(self._records()):
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
        names = self._collection_names(self._span_db)
        if self.mapping.span_member_table not in names:
            return None
        return self._distinct_required_ids(
            self._span_members(),
            self.mapping.span_member_table,
            self.mapping.span_member_id_column,
        )

    def flagged_ids(self) -> frozenset[int]:
        return frozenset(record_id for record_id, flag in self._record_flags() if flag == 1)

    def mirror_flagged_ids(self) -> frozenset[int] | None:
        names = self._collection_names(self._span_db)
        if self.mapping.mirror_table not in names:
            return None
        flagged: set[int] = set()
        for doc in self._docs(self._mirror()):
            record_id = coerce_record_id(
                self._required(
                    doc, self.mapping.mirror_id_column, collection=self.mapping.mirror_table
                ),
                origin=self.name,
            )
            reason = self._required(
                doc, self.mapping.mirror_reason_column, collection=self.mapping.mirror_table
            )
            flag = coerce_flag(
                self._required(
                    doc, self.mapping.mirror_flag_column, collection=self.mapping.mirror_table
                ),
                origin=self.name,
            )
            if reason == self.mapping.reason_flagged and flag == 1:
                flagged.add(record_id)
        return frozenset(flagged)

    def counts(self) -> dict[str, int]:
        # Counts are a totals report, so they use the server-side counters
        # rather than streaming every document: flagged_ids does the
        # coercing read that fails closed on a corrupt flag, and status
        # surfaces that through its sync check.
        with _fail_closed(self.name):
            records = self._records()
            total = int(records.count_documents({}))
            # Match both the integer 1 and a boolean True: BSON stores them as
            # distinct values and coerce_flag (used by flagged_ids) accepts
            # either as "flagged", so a bare {flag: 1} would miss a
            # boolean-stored flag and under-report against the sync check.
            flagged = int(records.count_documents({self.mapping.flag_column: {"$in": [1, True]}}))
        return {"records": total, "flagged": flagged}

    # -- Mongo-native predicate helper -----------------------------------------

    def visibility_filter(
        self, span_member_ids: Iterable[int], *, include_blur: bool = False
    ) -> dict[str, Any]:
        """Return a native Mongo query dict selecting deliverable records.

        Default (``include_blur=False``): ``flag = 0 AND id NOT IN spans``.
        ``include_blur=True`` relaxes the flag clause to ``flag in {0, 1}``,
        keeping standalone flagged (BLUR) rows while still excluding every
        in-span id -- documents with a missing, null, or corrupt flag fail
        closed instead of serving clear on the blur surface.
        """

        ids = sorted(
            {coerce_record_id(record_id, origin=self.name) for record_id in span_member_ids}
        )
        clauses: list[dict[str, Any]] = []
        if include_blur:
            clauses.append({self.mapping.flag_column: {"$in": [0, 1]}})
        else:
            clauses.append({self.mapping.flag_column: 0})
        if ids:
            clauses.append({self.mapping.record_id_column: {"$nin": ids}})
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}
