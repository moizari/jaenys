"""Populate-the-template derivation: from raw rows to the two layers.

The engine guards two layers it never creates: the live per-record flag
layer and the derived span layer with its flag mirror.  This module is the
scaffolding for producing them without starting from zero:

* :class:`KeywordDetector` -- populate it with the terms that make a single
  record sensitive on its own; it computes the flag layer.
* :class:`CueSessionDeriver` -- populate it with the cues that open and
  close a sensitive stretch (plus an optional inactivity timeout); it
  computes contiguous sessions where every in-between row rides along,
  however harmless each row looks alone.
* :func:`derive_layers` -- runs both over your rows and returns a
  :class:`DerivedLayers`, ready for the writers
  (:func:`jaenys.sql.sqlite.write_flags`,
  :func:`jaenys.sql.sqlite.write_span_layer`) or for
  ``InMemoryAdapter.rebuild_derived_layer``.

The templates match exactly what you put into them: plain substring
matching, case-insensitive by default, no NLP, no heuristics.  Anything a
vocabulary cannot express plugs in as a callable (``extra`` on the
detector, ``is_start`` / ``is_end`` on the deriver) -- that is where a real
classifier such as Presidio composes in.

Rows are plain mappings (dicts, ``sqlite3.Row``, ...), the same shape
:func:`jaenys.annotate_rows` takes, with configurable key names.
Pass rows in serving order (sort by timestamp first): session membership is
contiguity in the order given, and when a timeout is configured the
timestamps are checked to be non-decreasing -- unsorted rows would silently
produce wrong sessions, so they refuse instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Iterable, Mapping, Sequence

from .core import RedactionDriftError, _safe_repr, coerce_record_id

__all__ = [
    "KeywordDetector",
    "CueSessionDeriver",
    "Session",
    "DerivedLayers",
    "derive_layers",
    "END_CUE",
    "TIMEOUT",
    "END_OF_DATA",
]

# The three ways a session can close, recorded on Session.end_reason.
END_CUE = "end_cue"
TIMEOUT = "timeout"
END_OF_DATA = "end_of_data"


def _normalized_terms(terms: Sequence[str], *, kind: str, case_sensitive: bool) -> tuple[str, ...]:
    """Validate a populated vocabulary and fold its case once, up front."""

    if isinstance(terms, str):
        raise RedactionDriftError(
            f"{kind} must be a sequence of strings, not a bare string: {_safe_repr(terms)} "
            f"(a bare string would match its individual characters)"
        )
    normalized: list[str] = []
    for term in terms:
        if not isinstance(term, str) or not term.strip():
            raise RedactionDriftError(
                f"{kind} entries must be non-empty strings: {_safe_repr(term)}"
            )
        normalized.append(term if case_sensitive else term.casefold())
    return tuple(normalized)


def _validate_hook(hook: Any, *, kind: str) -> Callable[[str], bool] | None:
    if hook is not None and not callable(hook):
        raise RedactionDriftError(
            f"{kind} must be a callable taking the row text: {_safe_repr(hook)}"
        )
    return hook


def _value_of(row: Mapping[str, Any], key: str, *, index: int, kind: str) -> Any:
    try:
        return row[key]
    except (KeyError, IndexError, TypeError) as exc:
        raise RedactionDriftError(f"row {index} has no {kind} under key {key!r}") from exc


def _text_of(row: Mapping[str, Any], text_key: str, *, index: int) -> str | None:
    """A row's matchable text; ``None`` (a non-text record) never matches."""

    text = _value_of(row, text_key, index=index, kind="text")
    if text is None or isinstance(text, str):
        return text
    raise RedactionDriftError(
        f"row {index} text under {text_key!r} is not a string: {_safe_repr(text)}"
    )


def _moment_of(row: Mapping[str, Any], at_key: str, *, index: int) -> datetime:
    """A row's timestamp as ``datetime``; ISO-format strings are parsed."""

    raw = _value_of(row, at_key, index=index, kind="timestamp")
    if isinstance(raw, datetime):
        return raw
    if isinstance(raw, str):
        # datetime.fromisoformat only accepts the Zulu suffix from 3.11 on;
        # normalize it up front so the same rows derive identically on every
        # supported interpreter instead of refusing only on 3.10.
        candidate = raw[:-1] + "+00:00" if raw.endswith(("Z", "z")) else raw
        try:
            return datetime.fromisoformat(candidate)
        except ValueError as exc:
            raise RedactionDriftError(
                f"row {index} timestamp under {at_key!r} is not ISO format: {_safe_repr(raw)}"
            ) from exc
    raise RedactionDriftError(
        f"row {index} timestamp under {at_key!r} must be a datetime or ISO string: "
        f"{_safe_repr(raw)}"
    )


class KeywordDetector:
    """Template flag detector: flags records whose own text matches.

    Populate ``keywords`` with the terms that make one record sensitive in
    isolation.  Matching is plain substring, case-insensitive unless
    ``case_sensitive=True``.  ``extra`` is the escape hatch for what a word
    list cannot express: any callable ``(text: str) -> bool`` (a regex, a
    classifier, Presidio); a record is flagged when either source matches.

    The detector only computes flag decisions; writing them to a store is
    :func:`jaenys.sql.sqlite.write_flags` (or your own UPDATE).
    """

    def __init__(
        self,
        keywords: Sequence[str] = (),
        *,
        extra: Callable[[str], bool] | None = None,
        case_sensitive: bool = False,
    ) -> None:
        self.case_sensitive = bool(case_sensitive)
        self.keywords = _normalized_terms(
            keywords, kind="keywords", case_sensitive=self.case_sensitive
        )
        self.extra = _validate_hook(extra, kind="extra")
        if not self.keywords and self.extra is None:
            raise RedactionDriftError(
                "populate KeywordDetector with at least one keyword or an extra callable"
            )

    def matches(self, text: str | None) -> bool:
        """Is this one piece of text sensitive on its own?"""

        if text is None:
            return False
        haystack = text if self.case_sensitive else text.casefold()
        if any(keyword in haystack for keyword in self.keywords):
            return True
        return bool(self.extra(text)) if self.extra is not None else False

    def flagged_ids(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        id_key: str = "record_id",
        text_key: str = "text",
    ) -> frozenset[int]:
        """Run the detector over rows and return the ids to flag."""

        flagged: set[int] = set()
        for index, row in enumerate(rows, start=1):
            if self.matches(_text_of(row, text_key, index=index)):
                raw_id = _value_of(row, id_key, index=index, kind="record id")
                flagged.add(coerce_record_id(raw_id, origin=f"row {index}"))
        return frozenset(flagged)


@dataclass(frozen=True)
class Session:
    """One contiguous sensitive stretch: member ids in order + why it closed."""

    member_ids: tuple[int, ...]
    end_reason: str

    def __post_init__(self) -> None:
        members = tuple(
            coerce_record_id(member, origin="Session.member_ids") for member in self.member_ids
        )
        if not members:
            raise RedactionDriftError("a Session must have at least one member id")
        if len(set(members)) != len(members):
            # A repeated id trips the span-store primary key downstream, which
            # then misreports the store as unwritable; refuse it at the source.
            # Only ids are named, never record content.  Collect the repeats in
            # one pass rather than an O(n^2) per-member count.
            seen: set[int] = set()
            repeated: set[int] = set()
            for member in members:
                (repeated if member in seen else seen).add(member)
            raise RedactionDriftError(f"a Session must not repeat a member id: {sorted(repeated)}")
        if not isinstance(self.end_reason, str) or not self.end_reason:
            raise RedactionDriftError(
                f"Session.end_reason must be a non-empty string: {_safe_repr(self.end_reason)}"
            )
        object.__setattr__(self, "member_ids", members)


@dataclass(frozen=True)
class DerivedLayers:
    """Everything one span-store rebuild writes: flag ids + sessions.

    Produced by :func:`derive_layers`; hand-rolled pipelines may construct
    it directly (ids are coerced fail-closed) and feed the same writers.
    """

    flagged_ids: frozenset[int]
    sessions: tuple[Session, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "flagged_ids",
            frozenset(
                coerce_record_id(record_id, origin="DerivedLayers.flagged_ids")
                for record_id in self.flagged_ids
            ),
        )
        sessions = tuple(self.sessions)
        for session in sessions:
            if not isinstance(session, Session):
                raise RedactionDriftError(
                    f"DerivedLayers.sessions must contain Session objects: {_safe_repr(session)}"
                )
        object.__setattr__(self, "sessions", sessions)

    @property
    def span_member_ids(self) -> frozenset[int]:
        """The union of every session's members -- the HIDDEN set."""

        return frozenset(member for session in self.sessions for member in session.member_ids)


class CueSessionDeriver:
    """Template span deriver: cue-opened, cue-or-timeout-closed sessions.

    Populate ``start_cues`` with the phrases that open a sensitive stretch
    and ``end_cues`` with the phrases that close it.  ``timeout_minutes``
    closes an open session at the last row seen before an inactivity gap
    longer than the window; the late row is not a member, though it may
    open a new session.  Sessions are contiguous: every row between open
    and close becomes a member.  The row that opens a session never also
    closes it, and sessions still open when the rows run out close with
    ``end_reason=END_OF_DATA``.

    Custom judgments plug in as ``is_start`` / ``is_end`` callables
    ``(text: str) -> bool``; a cue OR the callable opens/closes.

    At least one close mechanism (end cues, ``is_end``, or
    ``timeout_minutes``) is required: with none, the first start cue would
    silently hide every row to the end of the data.

    ``timeout_minutes`` requires ``at_key`` at :meth:`derive` time;
    timestamps may be ``datetime`` objects or ISO-format strings (a trailing
    ``Z`` is accepted on every supported Python version).
    """

    def __init__(
        self,
        start_cues: Sequence[str] = (),
        end_cues: Sequence[str] = (),
        *,
        timeout_minutes: float | None = None,
        case_sensitive: bool = False,
        is_start: Callable[[str], bool] | None = None,
        is_end: Callable[[str], bool] | None = None,
    ) -> None:
        self.case_sensitive = bool(case_sensitive)
        self.start_cues = _normalized_terms(
            start_cues, kind="start_cues", case_sensitive=self.case_sensitive
        )
        self.end_cues = _normalized_terms(
            end_cues, kind="end_cues", case_sensitive=self.case_sensitive
        )
        self.is_start = _validate_hook(is_start, kind="is_start")
        self.is_end = _validate_hook(is_end, kind="is_end")
        if not self.start_cues and self.is_start is None:
            raise RedactionDriftError(
                "populate CueSessionDeriver with at least one start cue or an is_start callable"
            )
        if not self.end_cues and self.is_end is None and timeout_minutes is None:
            raise RedactionDriftError(
                "populate CueSessionDeriver with at least one way to close a session "
                "(end cues, an is_end callable, or timeout_minutes); with no end "
                "mechanism the first start cue would silently hide every row to the "
                "end of the data"
            )
        if timeout_minutes is not None:
            if isinstance(timeout_minutes, bool) or not isinstance(timeout_minutes, (int, float)):
                raise RedactionDriftError(
                    f"timeout_minutes must be a number: {_safe_repr(timeout_minutes)}"
                )
            if not math.isfinite(timeout_minutes):
                # nan is not <= 0, so it would count as the sole close mechanism
                # yet every gap comparison against it is False, so the timeout
                # never fires and the first start cue hides to the end of data.
                raise RedactionDriftError(
                    f"timeout_minutes must be finite: {_safe_repr(timeout_minutes)}"
                )
            if timeout_minutes <= 0:
                raise RedactionDriftError(
                    f"timeout_minutes must be positive: {_safe_repr(timeout_minutes)}"
                )
        self.timeout_minutes = timeout_minutes

    def _cue_match(
        self, text: str | None, cues: tuple[str, ...], hook: Callable[[str], bool] | None
    ) -> bool:
        if text is None:
            return False
        haystack = text if self.case_sensitive else text.casefold()
        if any(cue in haystack for cue in cues):
            return True
        return bool(hook(text)) if hook is not None else False

    def derive(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        id_key: str = "record_id",
        text_key: str = "text",
        at_key: str | None = None,
    ) -> tuple[Session, ...]:
        """Group rows (already in serving order) into contiguous sessions."""

        use_timeout = self.timeout_minutes is not None
        if use_timeout and at_key is None:
            raise RedactionDriftError(
                "timeout_minutes requires at_key so inactivity gaps can be measured"
            )

        sessions: list[Session] = []
        member_ids: list[int] = []
        last_moment: datetime | None = None
        previous_moment: datetime | None = None

        def close_session(reason: str) -> None:
            sessions.append(Session(tuple(member_ids), reason))
            member_ids.clear()

        for index, row in enumerate(rows, start=1):
            text = _text_of(row, text_key, index=index)
            record_id = coerce_record_id(
                _value_of(row, id_key, index=index, kind="record id"), origin=f"row {index}"
            )
            moment: datetime | None = None
            if use_timeout:
                moment = _moment_of(row, at_key, index=index)
                if previous_moment is not None:
                    try:
                        out_of_order = moment < previous_moment
                    except TypeError as exc:
                        # Aware vs naive datetimes do not compare; a raw
                        # TypeError here would break the everything-refuses-
                        # via-RedactionDriftError contract mid-derivation.
                        raise RedactionDriftError(
                            f"row {index} timestamp under {at_key!r} mixes timezone-aware "
                            "and timezone-naive values with earlier rows; make the "
                            "timestamps consistent before deriving"
                        ) from exc
                    if out_of_order:
                        raise RedactionDriftError(
                            f"rows are not in time order at row {index}; "
                            f"sort by {at_key!r} before deriving"
                        )
                previous_moment = moment
                if member_ids and last_moment is not None:
                    gap_minutes = (moment - last_moment).total_seconds() / 60.0
                    if gap_minutes > self.timeout_minutes:
                        close_session(TIMEOUT)
            if not member_ids:
                if self._cue_match(text, self.start_cues, self.is_start):
                    member_ids.append(record_id)
                    last_moment = moment
                continue
            member_ids.append(record_id)
            last_moment = moment
            if self._cue_match(text, self.end_cues, self.is_end):
                close_session(END_CUE)
        if member_ids:
            close_session(END_OF_DATA)
        return tuple(sessions)


def derive_layers(
    rows: Iterable[Mapping[str, Any]],
    *,
    detector: KeywordDetector,
    deriver: CueSessionDeriver,
    id_key: str = "record_id",
    text_key: str = "text",
    at_key: str | None = None,
) -> DerivedLayers:
    """Run both populated templates over rows in one pass over the data.

    ``rows`` may be any iterable (it is materialized once).  The result
    feeds :func:`jaenys.sql.sqlite.write_flags` and
    :func:`jaenys.sql.sqlite.write_span_layer` directly.
    """

    row_list = list(rows)
    return DerivedLayers(
        flagged_ids=detector.flagged_ids(row_list, id_key=id_key, text_key=text_key),
        sessions=deriver.derive(row_list, id_key=id_key, text_key=text_key, at_key=at_key),
    )
