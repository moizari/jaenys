"""Redis adapter for the store-agnostic visibility engine.

No driver import happens at module load time: the caller constructs its own
``redis`` client (e.g. ``redis.Redis(...)``) and passes it in.

Storage shape
-------------

One client, one key prefix (default ``"span_store"``). All keys are plain
Redis sets or strings, documented here:

* ``{prefix}:flagged`` -- a **set** of record ids whose live per-record flag
  is set. This is the live layer; the ingest path maintains it directly
  (``SADD``/``SREM``) as flags change.
* ``{prefix}:span_members`` -- a **set** of record ids belonging to derived
  spans.
* ``{prefix}:mirror_flagged`` -- a **set** of record ids the span store
  believes were flagged when spans were last derived (the freshness
  witness).
* ``{prefix}:span_layer_ready`` -- a marker key (string ``"1"``) written by
  the derivation pipeline once spans have been (re)computed. Its mere
  existence -- not its value -- means "derived"; absence means "never
  derived".
* ``{prefix}:mirror_ready`` -- same idea, for mirror availability.
* ``{prefix}:record_count`` -- a string-encoded integer, the total record
  count, maintained by the ingest path for cheap ``counts()`` reporting.
  ``counts()`` fails closed on this key: a missing key while flagged ids
  exist, a non-integer value, or a negative value all refuse rather than
  publish a provably wrong report.

Readiness marker
----------------

A Redis set that happens to be empty is indistinguishable from one that was
never created (``SMEMBERS`` on a missing key returns an empty set either
way). The explicit ``{prefix}:span_layer_ready`` / ``{prefix}:mirror_ready``
marker keys are therefore the only source of truth for readiness:
``span_member_ids()`` returns ``None`` unless ``span_layer_ready`` exists,
and ``mirror_flagged_ids()`` returns ``None`` unless ``mirror_ready`` exists.
Once the marker exists, an empty backing set correctly yields
``frozenset()``.

Redis clients may be configured with or without ``decode_responses``, so
every value read back is decoded defensively (``bytes`` or ``str`` both
handled).
"""

from __future__ import annotations

from typing import Any

from ..core import RedactionDriftError, _coerce_int, coerce_record_id
from .protocol import _fail_closed

__all__ = ["RedisAdapter"]


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


class RedisAdapter:
    """Adapter over a single ``redis``-py-compatible client."""

    def __init__(
        self,
        client: Any,
        *,
        prefix: str = "span_store",
        name: str = "redis",
    ) -> None:
        self._client = client
        self.prefix = prefix
        self.name = name

    # -- key helpers -----------------------------------------------------------

    def _key(self, suffix: str) -> str:
        return f"{self.prefix}:{suffix}"

    def _flagged_key(self) -> str:
        return self._key("flagged")

    def _span_members_key(self) -> str:
        return self._key("span_members")

    def _mirror_flagged_key(self) -> str:
        return self._key("mirror_flagged")

    def _span_layer_ready_key(self) -> str:
        return self._key("span_layer_ready")

    def _mirror_ready_key(self) -> str:
        return self._key("mirror_ready")

    def _record_count_key(self) -> str:
        return self._key("record_count")

    # -- internals -------------------------------------------------------------

    def _set_ids(self, key: str) -> frozenset[int]:
        """Read one id set, preferring the incremental ``SSCAN`` cursor.

        ``SMEMBERS`` on a huge set materializes and returns it in one blocking
        command, which can stall a production Redis; ``sscan_iter`` pages
        through the same members incrementally.  Clients without the method
        (minimal fakes, unusual wrappers) fall back to ``smembers``.  The
        iterator is consumed inside the fail-closed guard because it performs
        its I/O lazily.
        """

        with _fail_closed(self.name):
            scan = getattr(self._client, "sscan_iter", None)
            raw = list(scan(key)) if scan is not None else self._client.smembers(key)
        return frozenset(coerce_record_id(_decode(item), origin=self.name) for item in raw)

    def _exists(self, key: str) -> bool:
        with _fail_closed(self.name):
            return bool(self._client.exists(key))

    # -- StoreAdapter protocol ---------------------------------------------------

    def span_member_ids(self) -> frozenset[int] | None:
        if not self._exists(self._span_layer_ready_key()):
            return None
        return self._set_ids(self._span_members_key())

    def flagged_ids(self) -> frozenset[int]:
        return self._set_ids(self._flagged_key())

    def mirror_flagged_ids(self) -> frozenset[int] | None:
        if not self._exists(self._mirror_ready_key()):
            return None
        return self._set_ids(self._mirror_flagged_key())

    def counts(self) -> dict[str, int]:
        with _fail_closed(self.name):
            raw_count = self._client.get(self._record_count_key())
        flagged = len(self._set_ids(self._flagged_key()))
        if raw_count is None:
            # With flags present, "0 records, N flagged" would be a provably
            # wrong report; refuse instead of publishing nonsense counts.
            if flagged:
                raise RedactionDriftError(
                    f"{self._record_count_key()} is missing while flagged ids exist; "
                    "the ingest path must maintain the record-count key."
                )
            return {"records": 0, "flagged": 0}
        records = _coerce_int(_decode(raw_count), what="record count", origin=self.name)
        if records < 0:
            raise RedactionDriftError(f"record count is negative at {self.name}: {records}")
        if records < flagged:
            # Fewer records than flagged ids is impossible; the ingest path
            # must maintain the record-count key.
            raise RedactionDriftError(
                f"{self._record_count_key()} reports {records} records while {flagged} ids "
                "are flagged; the ingest path must maintain the record-count key."
            )
        return {"records": records, "flagged": flagged}
