"""DynamoDB adapter for the store-agnostic visibility engine.

No driver import happens at module load time: the caller constructs its own
``boto3`` ``Table`` resources (``boto3.resource("dynamodb").Table(...)``) and
passes them in. This module never imports ``boto3`` itself -- not even for
``boto3.dynamodb.conditions`` helpers -- so filter expressions below are built
as raw ``FilterExpression`` strings with ``ExpressionAttributeNames`` /
``ExpressionAttributeValues``, exactly as the low-level API accepts.

Storage shape
-------------

Two ``boto3`` ``Table`` resources:

* ``records_table`` -- one item per record, with the record id under
  ``mapping.record_id_column`` and the live per-record flag (0/1) under
  ``mapping.flag_column``.
* ``span_table`` -- a single-table span layer. Every item carries a
  discriminator attribute ``"layer"`` with one of three values:

  * ``"span_member"`` -- items ``{"layer": "span_member", <record id attr>:
    int}`` (one per record id in a derived span);
  * ``"mirror_flagged"`` -- items ``{"layer": "mirror_flagged", <record id
    attr>: int}`` (the flag-mirror layer, one per id flagged when spans were
    last derived);
  * ``"meta"`` -- a single readiness-marker item ``{"layer": "meta"}``,
    optionally with a boolean attribute ``"mirror_ready"`` set to ``true``
    when the mirror layer is available.

  The record id attribute name is ``mapping.span_member_id_column`` for
  ``span_member`` items and ``mapping.mirror_id_column`` for
  ``mirror_flagged`` items (the two mapping fields may name the same
  attribute).

Required key schema for ``span_table``
--------------------------------------

The readiness marker is fetched with ``get_item(Key=meta_key)``, and the
default ``meta_key`` is ``{"layer": "meta"}`` -- which only works when the
span table's **partition key is literally the** ``"layer"`` **attribute**.
Because every span-member item then shares the partition value
``"span_member"`` (and every mirror item ``"mirror_flagged"``), the table
also needs a **sort key** (e.g. the record id attribute) so those items can
coexist.  Deployments whose span table uses a different key schema pass
``meta_key={...}`` addressing their readiness item directly; the scans are
key-schema-agnostic (filter expressions on the ``"layer"`` attribute), so
only the marker lookup needs this.

Readiness marker
----------------

A DynamoDB ``scan`` with a filter returns an empty list whether the matching
items were never written or simply don't exist right now, so a bare scan
cannot distinguish "never derived" from "derived, currently empty". The
derivation pipeline is the sole writer of the ``"meta"`` item:

* the span layer is considered **derived** exactly when the ``"meta"`` item
  exists in ``span_table``;
* the mirror is considered **available** exactly when the ``"meta"`` item
  exists *and* carries a truthy ``"mirror_ready"`` attribute.

Filtering and pagination
-------------------------

All scans use raw ``FilterExpression`` strings (no ``boto3.dynamodb.
conditions`` import) and page through ``LastEvaluatedKey`` until exhausted.
Numeric attributes come back from boto3 as ``decimal.Decimal``; every id is
coerced through the core's fail-closed helpers before use -- an integral
Decimal (e.g. ``Decimal("3")``) converts cleanly, but a non-integral one
(``Decimal("3.5")``) raises :class:`~jaenys.core.RedactionDriftError`
rather than being silently truncated.
"""

from __future__ import annotations

from typing import Any

from ..core import SchemaMapping, RedactionDriftError, coerce_flag, coerce_record_id
from .protocol import _fail_closed

__all__ = ["DynamoDBAdapter"]

_LAYER_SPAN_MEMBER = "span_member"
_LAYER_MIRROR_FLAGGED = "mirror_flagged"
_LAYER_META = "meta"
_META_KEY_VALUE = {"layer": _LAYER_META}


class DynamoDBAdapter:
    """Adapter over two ``boto3`` DynamoDB ``Table`` resources.

    ``meta_key`` overrides the ``get_item`` key of the readiness-marker item
    for span tables whose key schema is not partition-keyed on the ``"layer"``
    attribute (see the module docstring for the default schema requirement).
    """

    def __init__(
        self,
        records_table: Any,
        span_table: Any,
        *,
        mapping: SchemaMapping = SchemaMapping(),
        name: str = "dynamodb",
        meta_key: dict[str, Any] | None = None,
    ) -> None:
        self._records_table = records_table
        self._span_table = span_table
        self.mapping = mapping
        self.name = name
        self._meta_key = dict(meta_key) if meta_key is not None else dict(_META_KEY_VALUE)

    # -- internals ---------------------------------------------------------------

    def _scan_all(self, table: Any, **kwargs: Any) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        with _fail_closed(self.name):
            response = table.scan(**kwargs)
            items.extend(response.get("Items", []))
            while "LastEvaluatedKey" in response:
                response = table.scan(ExclusiveStartKey=response["LastEvaluatedKey"], **kwargs)
                items.extend(response.get("Items", []))
        return items

    def _scan_count(self, table: Any, **kwargs: Any) -> int:
        with _fail_closed(self.name):
            response = table.scan(Select="COUNT", **kwargs)
            total = int(response.get("Count", 0))
            while "LastEvaluatedKey" in response:
                response = table.scan(
                    Select="COUNT", ExclusiveStartKey=response["LastEvaluatedKey"], **kwargs
                )
                total += int(response.get("Count", 0))
        return total

    def _get_meta_item(self) -> dict[str, Any] | None:
        with _fail_closed(self.name):
            response = self._span_table.get_item(Key=self._meta_key)
        return response.get("Item")

    def _required(self, item: dict[str, Any], key: str, *, layer: str) -> Any:
        if key not in item:
            raise RedactionDriftError(
                f"{self.name} store is missing required attribute {key!r} "
                f"in {layer}; refusing normal output."
            )
        return item[key]

    def _record_flags(self) -> list[tuple[int, int]]:
        items = self._scan_all(self._records_table)
        rows: list[tuple[int, int]] = []
        seen_ids: set[int] = set()
        for item in items:
            record_id = coerce_record_id(
                self._required(item, self.mapping.record_id_column, layer="records"),
                origin=self.name,
            )
            if record_id in seen_ids:
                raise RedactionDriftError(
                    f"{self.name} store has duplicate record ids; refusing normal output."
                )
            seen_ids.add(record_id)
            flag = coerce_flag(
                self._required(item, self.mapping.flag_column, layer="records"),
                origin=self.name,
            )
            rows.append((record_id, flag))
        return rows

    # -- StoreAdapter protocol -------------------------------------------------

    def span_member_ids(self) -> frozenset[int] | None:
        meta = self._get_meta_item()
        if meta is None:
            return None
        items = self._scan_all(
            self._span_table,
            FilterExpression="#l = :v",
            ExpressionAttributeNames={"#l": "layer"},
            ExpressionAttributeValues={":v": _LAYER_SPAN_MEMBER},
        )
        id_attr = self.mapping.span_member_id_column
        return frozenset(
            coerce_record_id(
                self._required(item, id_attr, layer=_LAYER_SPAN_MEMBER), origin=self.name
            )
            for item in items
        )

    def flagged_ids(self) -> frozenset[int]:
        return frozenset(record_id for record_id, flag in self._record_flags() if flag == 1)

    def mirror_flagged_ids(self) -> frozenset[int] | None:
        meta = self._get_meta_item()
        if meta is None or not meta.get("mirror_ready"):
            return None
        items = self._scan_all(
            self._span_table,
            FilterExpression="#l = :v",
            ExpressionAttributeNames={"#l": "layer"},
            ExpressionAttributeValues={":v": _LAYER_MIRROR_FLAGGED},
        )
        id_attr = self.mapping.mirror_id_column
        return frozenset(
            coerce_record_id(
                self._required(item, id_attr, layer=_LAYER_MIRROR_FLAGGED), origin=self.name
            )
            for item in items
        )

    def counts(self) -> dict[str, int]:
        # Counts are a totals report, so they use COUNT scans rather than
        # reading every item: flagged_ids does the coercing read that fails
        # closed on a corrupt flag, and status surfaces that through its sync
        # check.
        total = self._scan_count(self._records_table)
        flagged = self._scan_count(
            self._records_table,
            FilterExpression="#f = :v",
            ExpressionAttributeNames={"#f": self.mapping.flag_column},
            ExpressionAttributeValues={":v": 1},
        )
        return {"records": total, "flagged": flagged}
