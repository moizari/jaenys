"""In-memory reference adapter over plain Python data."""

from __future__ import annotations

from ..core import RedactionDriftError, coerce_flag, coerce_record_id

__all__ = ["InMemoryAdapter"]


class InMemoryAdapter:
    """Reference adapter over plain Python data.

    Used by the test suite and as the template for new adapters. It models
    the same two-layer world as every real store:

    * ``records``: ``{record_id: flag}`` -- the live primary layer.
    * a derived layer (span members + flag mirror) that only changes when
      :meth:`rebuild_derived_layer` runs, exactly like a real derivation
      pipeline. Editing a flag without rebuilding produces detectable
      drift, which the engine refuses to serve.
    """

    name = "in-memory"

    def __init__(
        self,
        records: dict[int, int] | None = None,
        *,
        span_members: frozenset[int] | set[int] | None = None,
        available: bool = True,
    ) -> None:
        self.records: dict[int, int] = {
            coerce_record_id(record_id, origin=self.name): coerce_flag(flag, origin=self.name)
            for record_id, flag in (records or {}).items()
        }
        self.available = available
        self._span_members: frozenset[int] | None = (
            frozenset(coerce_record_id(record_id, origin=self.name) for record_id in span_members)
            if span_members is not None
            else None
        )
        self._mirror_flagged: frozenset[int] | None = None

    # -- fixture/builder helpers (not part of the protocol) -----------------

    def set_flag(self, record_id: int, flag: int) -> None:
        """Edit the live flag layer (does NOT touch the derived layer)."""

        self.records[coerce_record_id(record_id, origin=self.name)] = coerce_flag(
            flag, origin=self.name
        )

    def set_span_members(self, record_ids: set[int] | frozenset[int]) -> None:
        self._span_members = frozenset(
            coerce_record_id(record_id, origin=self.name) for record_id in record_ids
        )

    def rebuild_derived_layer(
        self, *, span_members: set[int] | frozenset[int] | None = None
    ) -> None:
        """Re-derive spans and mirror the current flag layer (the "pipeline")."""

        if span_members is not None:
            self._span_members = frozenset(
                coerce_record_id(record_id, origin=self.name) for record_id in span_members
            )
        elif self._span_members is None:
            self._span_members = frozenset()
        self._mirror_flagged = frozenset(
            record_id
            for record_id, flag in self.records.items()
            if coerce_flag(flag, origin=self.name) == 1
        )

    def drop_span_layer(self) -> None:
        self._span_members = None
        self._mirror_flagged = None

    # -- StoreAdapter protocol ----------------------------------------------

    def _check_available(self) -> None:
        if not self.available:
            raise RedactionDriftError(f"{self.name} store is unreachable; refusing normal output.")

    def span_member_ids(self) -> frozenset[int] | None:
        self._check_available()
        return self._span_members

    def flagged_ids(self) -> frozenset[int]:
        self._check_available()
        return frozenset(
            record_id
            for record_id, flag in self.records.items()
            if coerce_flag(flag, origin=self.name) == 1
        )

    def mirror_flagged_ids(self) -> frozenset[int] | None:
        self._check_available()
        return self._mirror_flagged

    def counts(self) -> dict[str, int]:
        self._check_available()
        return {
            "records": len(self.records),
            "flagged": sum(
                1 for flag in self.records.values() if coerce_flag(flag, origin=self.name) == 1
            ),
        }
