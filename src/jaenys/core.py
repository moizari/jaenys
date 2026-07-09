"""Store-agnostic core of the Jaenys visibility engine.

Sensitivity is tracked at two independent layers:

* a live per-record flag on the primary store (``sensitive = 1``), and
* a separately derived layer of contiguous **spans** (groups of record ids
  whose surrounding context makes the whole stretch sensitive), plus a
  **mirror** of the flag layer captured when the spans were derived.

Every record resolves to one of three visibility states:

* ``VISIBLE``  -- not flagged, not in any span: serve normally.
* ``BLUR``     -- flagged but standalone (outside every span): deliverable,
  marked so the caller can blur/redact presentation.
* ``HIDDEN``   -- inside a derived span: never served on normal surfaces,
  regardless of the record's own flag.

The engine **fails closed**: if the live flag layer and the mirrored flag
layer cannot be proven equal (drift after an un-rebuilt flag edit, missing or
unreadable span store), helpers raise :class:`RedactionDriftError` instead of
serving records under stale isolation.

This module is pure Python with no storage dependencies.  SQL databases are
served by :mod:`jaenys.sql`; other stores implement the small adapter
protocol in :mod:`jaenys.adapters`.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

__all__ = [
    "VISIBLE",
    "BLUR",
    "HIDDEN",
    "RedactionDriftError",
    "SchemaMapping",
    "Guard",
    "classify",
    "annotate_rows",
    "filter_visible_rows",
    "verify_layer_sync",
    "verify_drift_witness",
    "span_member_digest",
    "load_guard_from_adapter",
    "assert_adapter_current",
    "adapter_status",
    "validate_name",
]

# Three visibility states for a single record.
VISIBLE = "visible"  # normal: flag = 0 AND not in-span -> clear everywhere
BLUR = "blur"  # standalone flagged: flag = 1 AND not in-span -> delivered, blurred
HIDDEN = "hidden"  # in-span: excluded from normal surfaces entirely

# ``\Z`` (not ``$``) anchors the end: ``$`` also matches just before a trailing
# newline, so ``"records\n"`` would slip through and reach a SQL string.
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\Z")

# Freshness callback: returns (span_layer_ready, span_sources, mirror_flagged_ids
# or None when the mirror is unavailable, current_span_member_ids or None when
# the span layer is not ready).  The member set lets verification catch span
# re-derivations that changed membership without touching any flag; producers
# accept the cost of one member read per verify (same order of work as the
# mirror read).  Must raise RedactionDriftError when the span store cannot be read --
# never return a guess.
RefreshCallback = Callable[
    [], "tuple[bool, tuple[str, ...], frozenset[int] | None, frozenset[int] | None]"
]


class RedactionDriftError(RuntimeError):
    """Raised whenever the engine cannot prove it is safe to serve.

    Layer sync that cannot be verified is redaction drift; corrupt stored
    values, unreachable stores, and unsafe configuration are treated the
    same way, because serving through any of them could leak. Every
    refusal fails closed: the engine raises instead of guessing.
    """


def validate_name(name: str, *, kind: str = "identifier") -> str:
    """Validate a table/column/field name against a conservative allowlist."""

    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise RedactionDriftError(f"unsafe {kind}: {name!r}")
    return name


def validate_namespace(namespace: str, *, kind: str = "namespace") -> str:
    """Validate one identifier or a database.schema identifier pair."""

    if not isinstance(namespace, str):
        raise RedactionDriftError(f"unsafe {kind}: <{type(namespace).__name__}>")
    parts = namespace.split(".")
    if not 1 <= len(parts) <= 2 or any(not part for part in parts):
        raise RedactionDriftError(f"unsafe {kind}: <str>")
    for part in parts:
        validate_name(part, kind=kind)
    return namespace


def _origin_suffix(origin: str) -> str:
    return f" at {origin}" if origin else ""


def _safe_repr(value: Any) -> str:
    """Describe a value without embedding stored data in refusal text."""

    if value is None:
        return "None"
    return f"<{type(value).__name__}>"


def _coerce_int(value: Any, *, what: str, origin: str) -> int:
    """Shared fail-closed body for :func:`coerce_record_id` / :func:`coerce_flag`.

    Accepts ints, digit strings,
    and integral floats/decimals (``3.0`` / ``Decimal("3")`` -- checked with
    ``int(value) == value`` so a non-integral value is refused rather than
    truncated; ``3.7`` truncating to ``3`` could silently mis-classify a
    record). A string must be a bare optional-sign digit run: underscored
    (``"1_0"``), padded (``" 7 "``), signed-plus (``"+7"``), and non-ASCII
    digit strings are refused because ``int()`` alone would accept them and
    silently remap the value. Strings skip the integral equality re-check
    (``int("7") == "7"`` is never true) since the run is already clean.
    ``None`` and junk strings raise :class:`RedactionDriftError` instead
    of a raw ValueError/TypeError.

    The offending value is rendered through :func:`_safe_repr`, which reports
    only the value type so a corrupt column cannot leak record data into a
    refusal message that later surfaces in status reports or on stderr.
    """

    if value is None:
        raise RedactionDriftError(f"{what} is missing{_origin_suffix(origin)}: {_safe_repr(value)}")
    if what == "record id" and isinstance(value, bool):
        raise RedactionDriftError(
            f"{what} is not an integer{_origin_suffix(origin)}: {_safe_repr(value)}"
        )
    if isinstance(value, str) and not re.fullmatch(r"-?[0-9]+", value):
        # int() accepts underscores, surrounding whitespace, a leading '+',
        # and non-ASCII digits; each would silently remap a corrupt id to a
        # different record, so only a bare optional-sign ASCII run passes.
        raise RedactionDriftError(
            f"{what} is not an integer{_origin_suffix(origin)}: {_safe_repr(value)}"
        )
    try:
        as_int = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        # OverflowError covers float("inf")/float("-inf"); NaN raises ValueError.
        raise RedactionDriftError(
            f"{what} is not an integer{_origin_suffix(origin)}: {_safe_repr(value)}"
        ) from exc
    if not isinstance(value, str) and as_int != value:
        raise RedactionDriftError(
            f"{what} is not an integral value{_origin_suffix(origin)}: {_safe_repr(value)}"
        )
    return as_int


def coerce_record_id(value: Any, *, origin: str = "") -> int:
    """Fail-closed conversion of a raw record id to ``int``.

    See :func:`_coerce_int` for the accepted shapes and refusal rules.
    """

    return _coerce_int(value, what="record id", origin=origin)


def coerce_flag(value: Any, *, origin: str = "") -> int:
    """Fail-closed conversion of a raw sensitivity flag to ``0`` or ``1``.

    See :func:`_coerce_int` for the accepted numeric shapes and refusal rules.
    On top of that, the flag domain is exactly ``{0, 1}``: any other integral
    value (``2``, ``-1``, ``"7"`` ...) is refused with a :class:`RedactionDriftError`.
    ``bool`` still coerces to ``1``/``0`` and is accepted. This closes a
    fail-open leak where a corrupt/nonstandard flag would be treated as
    "not flagged" on the Python/adapter path and served, while the SQL
    predicate (``flag = 0``) excludes it -- the store must refuse, not leak.
    """

    coerced = _coerce_int(value, what="flag", origin=origin)
    if coerced not in (0, 1):
        raise RedactionDriftError(
            f"flag must be 0 or 1{_origin_suffix(origin)}: {_safe_repr(value)}"
        )
    return coerced


@dataclass(frozen=True)
class SchemaMapping:
    """Names of the tables/columns (or collections/fields) the engine reads.

    The same mapping serves SQL and non-SQL stores: for document/KV stores the
    ``*_table`` fields name collections/keyspaces and the ``*_column`` fields
    name document fields.

    Layers:

    * ``record_table`` lives in the **primary** store and carries the live
      per-record flag (``flag_column``).
    * ``span_member_table`` lives in the **span** store and lists record ids
      belonging to derived spans.
    * ``mirror_table`` also lives in the span store; rows with
      ``mirror_reason_column = reason_flagged`` mirror the flag layer as it
      was when spans were last derived (the freshness witness), and rows with
      ``reason_span`` record span membership in stores using the single-table
      shape.
    * ``span_group_table`` (optional) groups span members and may scope them
      to a source version via ``span_group_version_column``.
    * ``span_namespace`` (optional) is the schema/database qualifier under
      which the span-layer tables are reachable from a single connection.
    * ``meta_table`` (optional) names the drift-witness table in the
      **primary** store; the layer writers record the span-member count and
      digest there so a lost span store refuses even with zero flags.
    """

    record_table: str = "records"
    record_id_column: str = "record_id"
    flag_column: str = "sensitive"
    span_member_table: str = "span_members"
    span_member_id_column: str = "record_id"
    mirror_table: str = "sensitive_records"
    mirror_id_column: str = "record_id"
    mirror_reason_column: str = "copy_reason"
    mirror_flag_column: str = "source_flag"
    reason_flagged: str = "flagged"
    reason_span: str = "span"
    span_group_table: str | None = "spans"
    span_group_id_column: str = "span_id"
    span_group_version_column: str | None = "source_version_id"
    span_namespace: str | None = None
    # The drift-witness table lives in the PRIMARY store and records that
    # a span derivation was written (member count + digest), so a lost or
    # emptied span store refuses even when zero records are flagged.  Its
    # column names are fixed (the table is wholly engine-owned); set to None
    # to disable the witness entirely.
    meta_table: str | None = "span_store_meta"

    def __post_init__(self) -> None:
        for field_name in (
            "record_table",
            "record_id_column",
            "flag_column",
            "span_member_table",
            "span_member_id_column",
            "mirror_table",
            "mirror_id_column",
            "mirror_reason_column",
            "mirror_flag_column",
            "span_group_id_column",
        ):
            validate_name(getattr(self, field_name), kind=field_name)
        for optional_name in (
            "span_group_table",
            "span_group_version_column",
            "span_namespace",
            "meta_table",
        ):
            value = getattr(self, optional_name)
            if value is not None:
                if optional_name == "span_namespace":
                    validate_namespace(value, kind=optional_name)
                else:
                    validate_name(value, kind=optional_name)
        for reason_name in ("reason_flagged", "reason_span"):
            value = getattr(self, reason_name)
            if not isinstance(value, str) or not value:
                raise RedactionDriftError(f"{reason_name} must be a non-empty string")
        if self.reason_flagged == self.reason_span:
            raise RedactionDriftError("reason_flagged and reason_span must differ")


DEFAULT_MAPPING = SchemaMapping()


def classify(record_id: int, flag: int, span_member_ids: frozenset[int]) -> str:
    """Return the visibility state for one record.

    HIDDEN if the id is in a derived span; otherwise BLUR when the per-record
    flag is set (standalone flagged); otherwise VISIBLE.

    Both inputs go through the fail-closed coercers rather than a raw
    ``int()``: ``coerce_record_id`` refuses non-integral ids instead of
    truncating them (``3.7`` would otherwise become record ``3``), and
    ``coerce_flag`` refuses any flag outside ``{0, 1}`` -- so a corrupt flag
    (``2``, ``-1``) makes this **refuse** with :class:`RedactionDriftError` rather
    than silently classifying VISIBLE and leaking the record.  Post-coercion
    the flag is exactly ``0`` or ``1``; ``!= 0`` is used as belt-and-braces.
    """

    if coerce_record_id(record_id) in span_member_ids:
        return HIDDEN
    if coerce_flag(flag) != 0:
        return BLUR
    return VISIBLE


def _row_value(row: Mapping[str, Any], key: str) -> Any:
    try:
        return row[key]
    except (IndexError, KeyError, TypeError) as exc:
        raise RedactionDriftError(
            "row must include the record id and flag keys for visibility filtering"
        ) from exc


def annotate_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    span_member_ids: frozenset[int],
    id_key: str = "record_id",
    flag_key: str = "sensitive",
) -> list[dict[str, Any]]:
    """Keep normal + standalone-flagged rows, drop in-span rows, mark blur.

    Each kept row is shallow-copied to a plain ``dict`` with an added
    ``"blurred"`` boolean (True for standalone flagged, False for normal).
    Only in-span rows are dropped.  Rows missing either key are rejected
    because silently accepting them would make a serve path easy to misuse.
    The added ``"blurred"`` key overwrites any pre-existing key of that name
    in the row.
    """

    annotated: list[dict[str, Any]] = []
    for row in rows:
        record_id = coerce_record_id(_row_value(row, id_key), origin="row")
        flag = coerce_flag(_row_value(row, flag_key), origin="row")
        state = classify(record_id, flag, span_member_ids)
        if state == HIDDEN:
            continue
        new_row = dict(row)
        new_row["blurred"] = state == BLUR
        annotated.append(new_row)
    return annotated


def filter_visible_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    span_member_ids: frozenset[int],
    id_key: str = "record_id",
    flag_key: str = "sensitive",
) -> list[Mapping[str, Any]]:
    """Strict filter: keep only fully VISIBLE rows (drops BLUR and HIDDEN)."""

    filtered: list[Mapping[str, Any]] = []
    for row in rows:
        record_id = coerce_record_id(_row_value(row, id_key), origin="row")
        flag = coerce_flag(_row_value(row, flag_key), origin="row")
        if classify(record_id, flag, span_member_ids) == VISIBLE:
            filtered.append(row)
    return filtered


@dataclass(frozen=True)
class Guard:
    """Materialized span exclusions loaded from a span store or adapter.

    ``span_member_ids`` is the ID set of every record inside a derived span.
    ``span_layer_ready`` records whether a usable span layer existed at load
    time; ``span_sources`` names the span shapes that were detected.

    ``_refresh`` re-reads the span layer for freshness verification.  Guards
    built by the library loaders always carry it; a hand-built guard without
    one **fails closed** in :func:`verify_guard_current` whenever flagged
    records exist.
    """

    # repr=False (compare unchanged): this set can hold millions of ids, and
    # dumping the whole frozenset into the dataclass repr makes any log line
    # that renders a guard unusable.
    span_member_ids: frozenset[int] = field(repr=False)
    span_layer_ready: bool
    span_sources: tuple[str, ...] = ()
    origin: str = ""
    _refresh: RefreshCallback | None = field(default=None, compare=False, repr=False)

    def classify(self, record_id: int, flag: int) -> str:
        return classify(record_id, flag, self.span_member_ids)

    def annotate_rows(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        id_key: str = "record_id",
        flag_key: str = "sensitive",
    ) -> list[dict[str, Any]]:
        return annotate_rows(
            rows, span_member_ids=self.span_member_ids, id_key=id_key, flag_key=flag_key
        )

    def filter_rows(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        id_key: str = "record_id",
        flag_key: str = "sensitive",
    ) -> list[Mapping[str, Any]]:
        return filter_visible_rows(
            rows, span_member_ids=self.span_member_ids, id_key=id_key, flag_key=flag_key
        )


def _origin_label(origin: str) -> str:
    return origin or "span store"


def verify_layer_sync(
    live_flagged_ids: Iterable[int],
    *,
    span_layer_ready: bool,
    span_sources: tuple[str, ...],
    mirror_flagged_ids: frozenset[int] | None,
    origin: str = "",
) -> None:
    """Fail closed unless the two sensitivity layers provably agree.

    This is the single decision point every backend routes through:

    * span layer missing/unusable while live flags exist -> refuse;
    * live flags exist but no span source was detected -> refuse (a fresh
      flag edit can imply a new span no stale table knows to hide);
    * mirror unavailable (legacy span-only shape) -> serve; spans are still
      hidden and the live flag is still checked, the store simply cannot
      prove mirror freshness;
    * mirror differs from the live flag layer in either direction -> refuse.
    """

    live = frozenset(coerce_record_id(record_id, origin=origin) for record_id in live_flagged_ids)
    label = _origin_label(origin)
    if not span_layer_ready:
        if live:
            raise RedactionDriftError(
                f"redaction drift risk at {label}: span isolation is unavailable while "
                "the primary store has flagged records; rebuild the span derivation "
                "layer before serving."
            )
        return
    if live and not span_sources:
        raise RedactionDriftError(
            f"redaction drift risk at {label}: no span-layer tables were found; "
            "rebuild the span derivation layer before serving."
        )
    if mirror_flagged_ids is None:
        # Legacy span-only stores hide every recorded span and the live flag
        # is still checked by every predicate; they simply cannot offer
        # mirror-freshness verification.
        return
    if frozenset(coerce_record_id(rid, origin=origin) for rid in mirror_flagged_ids) != live:
        raise RedactionDriftError(
            f"redaction drift at {label}: span isolation is stale relative to the "
            "live flag layer; rebuild the span derivation layer before serving."
        )


def span_member_digest(member_ids: Iterable[int]) -> str:
    """Deterministic, order-independent digest of a span-member id set.

    Recorded in the primary store's drift witness by the layer writers
    and compared by :func:`verify_drift_witness`.  Only record ids are
    hashed -- never record content.
    """

    ids = sorted({coerce_record_id(rid, origin="span member digest") for rid in member_ids})
    return hashlib.sha256(",".join(str(rid) for rid in ids).encode("ascii")).hexdigest()


def verify_drift_witness(
    witness: tuple[int, str] | None,
    *,
    span_layer_ready: bool,
    span_member_ids: Iterable[int] | None = None,
    span_member_count: int | None = None,
    origin: str = "",
) -> None:
    """Fail closed unless the span layer satisfies the recorded drift witness.

    ``witness`` is what the primary store recorded at write time --
    ``(member_count, member_digest)`` -- or ``None`` when the store carries
    no witness (legacy layout, hand-rolled writers, ``meta_table=None``), in
    which case there is nothing to verify and the pre-witness semantics apply.

    The witness closes the zero-flag hole: with no flagged records the
    flag-mirror comparison is vacuous, so a deleted or emptied span store
    would otherwise silently serve formerly in-span rows.  With a witness:

    * span layer unavailable while the witness records members -> **refuse**;
    * span layer present -> its membership must match the witness, by digest
      when the member set is materialized (``span_member_ids``) or by
      distinct-member count when the caller counted in the store
      (``span_member_count``); mismatch -> **refuse**.
    """

    if witness is None:
        return
    count, digest = witness
    label = _origin_label(origin)
    if not span_layer_ready:
        if count > 0:
            raise RedactionDriftError(
                f"redaction drift at {label}: the primary store records a span "
                f"derivation with {count} member(s), but no span layer is available; "
                "restore or rebuild the span store before serving."
            )
        return
    if span_member_ids is not None:
        if span_member_digest(span_member_ids) != digest:
            raise RedactionDriftError(
                f"redaction drift at {label}: the span layer does not match the "
                "derivation recorded in the primary store; rebuild the span "
                "derivation layer before serving."
            )
        return
    if span_member_count is not None:
        if int(span_member_count) != count:
            raise RedactionDriftError(
                f"redaction drift at {label}: the span layer does not match the "
                "derivation recorded in the primary store; rebuild the span "
                "derivation layer before serving."
            )
        return
    raise RedactionDriftError(
        f"cannot verify the drift witness at {label}: no span membership "
        "information was provided; refusing normal output."
    )


def verify_guard_current(live_flagged_ids: Iterable[int], guard: Guard) -> None:
    """Re-verify a loaded guard against the live flag layer, failing closed.

    A flag edit on the primary store takes effect immediately, while the
    span/mirror layers only change on a rebuild.  Until the rebuild happens,
    serving normal rows is unsafe: a newly recognized span can contain
    neutral records that no stale member table knows to hide.

    Beyond the flag/mirror comparison, the refresh's current member set is
    compared against the guard's loaded ``span_member_ids``: a span
    re-derivation can change membership without touching a single flag (a
    derivation-config change pulling new neutral records into spans), and a
    long-lived materialized guard would otherwise keep serving its stale
    frozen set.  Membership drift refuses -- the new set is never silently
    adopted, because callers may already hold predicates built from the old
    ids.
    """

    live = frozenset(
        coerce_record_id(record_id, origin=guard.origin) for record_id in live_flagged_ids
    )
    if guard._refresh is None:
        if not guard.span_layer_ready:
            # Loaded from a store that had no span layer; still safe only
            # while nothing is flagged.
            verify_layer_sync(
                live,
                span_layer_ready=False,
                span_sources=(),
                mirror_flagged_ids=None,
                origin=guard.origin,
            )
            return
        if live:
            raise RedactionDriftError(
                "guard has no freshness source; reload it through a library loader "
                "before serving flagged data."
            )
        return
    ready, sources, mirror, current_members = guard._refresh()
    if guard.span_layer_ready and not ready:
        raise RedactionDriftError(
            f"redaction drift: span layer at {_origin_label(guard.origin)} "
            "became unavailable since this guard was loaded; restore or rebuild "
            "the span store before serving."
        )
    verify_layer_sync(
        live,
        span_layer_ready=ready,
        span_sources=sources,
        mirror_flagged_ids=mirror,
        origin=guard.origin,
    )
    if ready and current_members is not None and current_members != guard.span_member_ids:
        raise RedactionDriftError(
            f"redaction drift: span membership at {_origin_label(guard.origin)} "
            "changed since this guard was loaded; reload the guard before serving."
        )


# ---------------------------------------------------------------------------
# Adapter path (non-SQL stores).  Adapters implement the 4-method protocol in
# jaenys.adapters.protocol; consumption here is structural, so any
# object with the right methods works.
# ---------------------------------------------------------------------------


def _adapter_state(
    adapter: Any,
) -> tuple[bool, tuple[str, ...], frozenset[int] | None, frozenset[int] | None]:
    """Read an adapter's layers once, coercing every id fail-closed.

    The single reader for the three adapter entry points: the member set is
    coerced here (origin = the adapter name) so a store returning string ids
    refuses like every other boundary instead of classifying wrong.
    """

    origin = getattr(adapter, "name", type(adapter).__name__)
    raw_members = adapter.span_member_ids()
    ready = raw_members is not None
    sources = (origin,) if ready else ()
    mirror = adapter.mirror_flagged_ids() if ready else None
    current = (
        frozenset(coerce_record_id(record_id, origin=origin) for record_id in raw_members)
        if ready
        else None
    )
    return ready, sources, mirror, current


def load_guard_from_adapter(adapter: Any) -> Guard:
    """Materialize a :class:`Guard` from a store adapter.

    ``adapter.span_member_ids()`` returning ``None`` means the store has no
    usable span layer; the guard then only supports the live flag check and
    fails closed as soon as flagged records exist.
    """

    origin = getattr(adapter, "name", type(adapter).__name__)
    ready, sources, _mirror, members = _adapter_state(adapter)

    def _refresh() -> tuple[bool, tuple[str, ...], frozenset[int] | None, frozenset[int] | None]:
        return _adapter_state(adapter)

    return Guard(
        span_member_ids=members if members is not None else frozenset(),
        span_layer_ready=ready,
        span_sources=sources,
        origin=origin,
        _refresh=_refresh,
    )


def assert_adapter_current(adapter: Any) -> None:
    """Fail closed unless the adapter's two layers provably agree right now.

    Point-in-time check with no loaded guard involved, so only the flag/mirror
    layers are compared; membership drift is a guard-relative concept handled
    by :func:`verify_guard_current`.
    """

    ready, sources, mirror, _members = _adapter_state(adapter)
    verify_layer_sync(
        adapter.flagged_ids(),
        span_layer_ready=ready,
        span_sources=sources,
        mirror_flagged_ids=mirror,
        origin=getattr(adapter, "name", type(adapter).__name__),
    )


def adapter_status(adapter: Any) -> dict[str, Any]:
    """Counts-only readiness report for an adapter-backed store pair."""

    counts = dict(adapter.counts())
    ready, _sources, _mirror, members = _adapter_state(adapter)
    report: dict[str, Any] = {
        "adapter": getattr(adapter, "name", type(adapter).__name__),
        "span_layer_ready": ready,
        "unique_span_member_ids": len(members) if members is not None else 0,
        "counts": counts,
    }
    try:
        assert_adapter_current(adapter)
        report["layers_in_sync"] = True
    except RedactionDriftError as exc:
        report["layers_in_sync"] = False
        report["refusal"] = str(exc)
    return report
