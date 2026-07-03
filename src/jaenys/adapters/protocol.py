"""Store-adapter protocol for non-SQL databases.

Any store can back the span-visibility engine by answering four questions.
Implement this protocol (structurally -- no inheritance required) and the
engine's loaders, freshness verification, and status reporting all work:

* ``span_member_ids()`` -- ids of every record inside a derived span, or
  ``None`` when the store has no usable span layer yet.
* ``flagged_ids()`` -- ids whose live per-record flag is set on the primary
  store, read fresh on every call.
* ``mirror_flagged_ids()`` -- the span store's mirrored claim of flagged ids
  (captured when spans were last derived), or ``None`` when the store cannot
  say (legacy span-only shapes).
* ``counts()`` -- at least ``{"records": int, "flagged": int}`` for
  counts-only status reports.

Adapters MUST raise :class:`jaenys.core.RedactionDriftError` when their
store is unreachable or unreadable, and also when stored id or flag data is
corrupt or uninterpretable (non-integral numbers, junk strings, missing
values where a value is required) -- never return a guess: the engine fails
closed on refusal, but it cannot fail closed on silently wrong or silently
truncated data.

Adapters SHOULD expose a ``name`` attribute used in error messages and
status reports.

Drift-witness gap
-----------------

The SQL path records a drift witness in the primary store, so an emptied span
layer refuses even with zero flagged records.  The adapter path has no witness
equivalent: with zero flags, an emptied span layer behind an intact readiness
marker is indistinguishable from a real empty derivation, and both serve.  Keep
a loaded guard alive across the window -- membership drift then refuses on
re-verification -- and rebuild the derivation rather than editing span data in
place.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Protocol, runtime_checkable

from ..core import SchemaMapping, RedactionDriftError

__all__ = ["StoreAdapter"]

# The nine SchemaMapping fields that name collections/fields in document
# stores.  Shared by every adapter that validates mapped names against
# store-specific unsafe characters.
MAPPED_NAME_FIELDS = (
    "record_table",
    "record_id_column",
    "flag_column",
    "span_member_table",
    "span_member_id_column",
    "mirror_table",
    "mirror_id_column",
    "mirror_reason_column",
    "mirror_flag_column",
)


@contextmanager
def _fail_closed(name: str) -> Iterator[None]:
    """Wrap store I/O so any driver error becomes a fail-closed refusal.

    :class:`RedactionDriftError` (a refusal already) passes through unchanged;
    every other exception -- connection failures, driver-specific errors,
    unexpected response shapes -- is converted to the standard refusal so the
    engine never guesses about an unreadable store.
    """

    try:
        yield
    except RedactionDriftError:
        raise
    except Exception as exc:
        raise RedactionDriftError(
            f"{name} store is unreachable or unreadable; refusing normal output."
        ) from exc


def _reject_unsafe_document_name(name: str, *, kind: str, store: str) -> None:
    """Reject ``$``/``.`` in document-store names (query operators/field paths)."""

    if "$" in name or "." in name:
        raise RedactionDriftError(f"unsafe {kind} for {store}: {name!r} (contains '$' or '.')")


def _validate_mapped_names(mapping: SchemaMapping, *, store: str) -> None:
    """Run the ``$``/``.`` rejection over every mapped collection/field name.

    Defense in depth: :class:`~jaenys.core.SchemaMapping` already
    allowlists these names at construction; this second check keeps the
    adapters safe even for hand-built mapping-like objects.
    """

    for field_name in MAPPED_NAME_FIELDS:
        _reject_unsafe_document_name(getattr(mapping, field_name), kind=field_name, store=store)


@runtime_checkable
class StoreAdapter(Protocol):
    """The four questions a store must answer to back the engine."""

    def span_member_ids(self) -> frozenset[int] | None: ...

    def flagged_ids(self) -> frozenset[int]: ...

    def mirror_flagged_ids(self) -> frozenset[int] | None: ...

    def counts(self) -> dict[str, int]: ...
