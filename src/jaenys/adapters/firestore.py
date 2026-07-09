"""Firestore adapter for the store-agnostic visibility engine.

No driver import happens at module load time: the caller constructs its own
``google.cloud.firestore.Client`` and passes it in.

Storage shape
-------------

One Firestore ``Client``, three top-level collections named directly by the
mapping (``mapping.record_table`` / ``mapping.span_member_table`` /
``mapping.mirror_table``):

* ``mapping.record_table`` -- one document per record, with the record id
  under ``mapping.record_id_column`` and the live per-record flag (0/1)
  under ``mapping.flag_column``.
* ``mapping.span_member_table`` -- one document per record id belonging to a
  derived span, field ``mapping.span_member_id_column``.
* ``mapping.mirror_table`` -- one document per id mirrored from the flag
  layer when spans were last derived: ``{<mirror_id_column>: int,
  <mirror_reason_column>: str, <mirror_flag_column>: 0/1}``.

Field/collection names must not contain ``"$"`` or ``"."`` -- rejected at
construction, matching the MongoDB adapter's validate-before-use rule (a dot
has special meaning in Firestore field paths).

Readiness marker
----------------

``.stream()`` over an empty collection and over a nonexistent collection are
indistinguishable -- both yield zero documents -- so readiness cannot be
inferred from a bare stream. The derivation pipeline is the sole writer of a
readiness marker document with id ``"_span_meta"`` inside the span-members
collection:

* the span layer is considered **derived** exactly when
  ``span_member_table.document("_span_meta").get().exists`` is true;
* the mirror is considered **available** exactly when the mirror
  collection's own marker document, ``mirror_table.document("_span_meta")``,
  exists (the mirror collection is independent of the span-members
  collection and may be absent even when spans are derived, matching the
  "legacy span-only" shape modeled elsewhere in this library).

The marker document is skipped when collecting ids (it carries none of the
expected id fields, and is additionally excluded by document id as a
belt-and-suspenders check).

Read costs
----------

Firestore bills **per document read**, and every protocol call here streams
whole collections (``.stream()``) rather than issuing filtered queries --
building ``FieldFilter`` queries cleanly would require importing the driver,
which this module deliberately never does.  Each ``flagged_ids()`` /
``span_member_ids()`` / ``mirror_flagged_ids()`` / ``counts()`` call
therefore re-reads its full collection, and freshness verification calls
several of them.  Budget verification frequency accordingly on large
collections, or front the reads with a caller-side cache that is invalidated
on derivation runs.
"""

from __future__ import annotations

from typing import Any

from ..core import SchemaMapping, RedactionDriftError, coerce_flag, coerce_record_id
from .protocol import _fail_closed, _reject_unsafe_document_name, _validate_mapped_names

__all__ = ["FirestoreAdapter"]

_META_DOC_ID = "_span_meta"


def _reject_unsafe_firestore_name(name: str, *, kind: str) -> None:
    """Dots are field-path separators in Firestore; ``$`` is reserved."""

    _reject_unsafe_document_name(name, kind=kind, store="Firestore")


class FirestoreAdapter:
    """Adapter over a single ``google.cloud.firestore.Client``."""

    def __init__(
        self,
        client: Any,
        *,
        mapping: SchemaMapping = SchemaMapping(),
        name: str = "firestore",
    ) -> None:
        self._client = client
        self.mapping = mapping
        self.name = name
        _validate_mapped_names(mapping, store="Firestore")

    # -- internals ---------------------------------------------------------------

    def _collection(self, name: str) -> Any:
        return self._client.collection(name)

    def _stream_docs(self, collection: Any) -> list[dict]:
        with _fail_closed(self.name):
            snapshots = list(collection.stream())
        docs: list[dict] = []
        for snapshot in snapshots:
            if getattr(snapshot, "id", None) == _META_DOC_ID:
                continue
            data = snapshot.to_dict()
            if data is not None:
                docs.append(data)
        return docs

    def _marker_exists(self, collection: Any) -> bool:
        with _fail_closed(self.name):
            snapshot = collection.document(_META_DOC_ID).get()
            return bool(snapshot.exists)

    def _required(self, doc: dict[str, Any], key: str, *, collection: str) -> Any:
        if key not in doc:
            raise RedactionDriftError(
                f"{self.name} store is missing required field {key!r} "
                f"in {collection}; refusing normal output."
            )
        return doc[key]

    def _record_flags(self) -> list[tuple[int, int]]:
        docs = self._stream_docs(self._collection(self.mapping.record_table))
        rows: list[tuple[int, int]] = []
        seen_ids: set[int] = set()
        for doc in docs:
            record_id = coerce_record_id(
                self._required(
                    doc, self.mapping.record_id_column, collection=self.mapping.record_table
                ),
                origin=self.name,
            )
            if record_id in seen_ids:
                raise RedactionDriftError(
                    f"{self.name} store has duplicate record ids; refusing normal output."
                )
            seen_ids.add(record_id)
            flag = coerce_flag(
                self._required(doc, self.mapping.flag_column, collection=self.mapping.record_table),
                origin=self.name,
            )
            rows.append((record_id, flag))
        return rows

    # -- StoreAdapter protocol -------------------------------------------------

    def span_member_ids(self) -> frozenset[int] | None:
        span_collection = self._collection(self.mapping.span_member_table)
        if not self._marker_exists(span_collection):
            return None
        docs = self._stream_docs(span_collection)
        id_field = self.mapping.span_member_id_column
        return frozenset(
            coerce_record_id(
                self._required(doc, id_field, collection=self.mapping.span_member_table),
                origin=self.name,
            )
            for doc in docs
        )

    def flagged_ids(self) -> frozenset[int]:
        return frozenset(record_id for record_id, flag in self._record_flags() if flag == 1)

    def mirror_flagged_ids(self) -> frozenset[int] | None:
        mirror_collection = self._collection(self.mapping.mirror_table)
        if not self._marker_exists(mirror_collection):
            return None
        docs = self._stream_docs(mirror_collection)
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
