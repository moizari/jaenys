"""SQLite conveniences for the SQL backend.

SQLite is the reference engine: it ships with Python, so these helpers
provide path-based ergonomics -- open a read-only connection from a file
path, treat a missing or zero-byte span store as "no span layer yet", and
reach the span store from the primary connection via ``ATTACH DATABASE``.

It also carries the write half of the caller-owned derivation step for the
reference engine: :func:`write_flags` (the detector's write-back) and
:func:`write_span_layer` (span members + the flag-mirror snapshot), fed by
:mod:`jaenys.derivation` or any hand-rolled
:class:`~jaenys.derivation.DerivedLayers`.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import closing, contextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from ..core import (
    DEFAULT_MAPPING,
    SchemaMapping,
    Guard,
    RedactionDriftError,
    coerce_record_id,
    span_member_digest,
    validate_name,
)
from ..derivation import DerivedLayers
from .dialects import SQLITE
from .guard import (
    META_COUNT_COLUMN,
    META_DERIVED_AT_COLUMN,
    META_DIGEST_COLUMN,
    META_LAYER_COLUMN,
    META_LAYER_SPAN,
    AttachedGuard,
    _assert_attached_flag_layer_current,
    _assert_drift_witness_satisfied,
    _assert_primary_flags_well_formed,
    _assert_primary_ids_well_formed,
    _has_flagged,
    _mirror_flagged_ids,
    _read_span_ids,
    _span_group_versions,
    span_sources,
)
from . import guard as _guard

__all__ = [
    "open_readonly",
    "usable_span_db",
    "load_guard",
    "attached_guard",
    "status",
    "write_flags",
    "write_span_layer",
    "DEFAULT_ATTACH_NAME",
]

DEFAULT_ATTACH_NAME = "span_store"


def open_readonly(db_path: Path | str) -> sqlite3.Connection:
    """Open a SQLite database read-only for serve-path use."""

    path = Path(db_path)
    if not path.exists():
        raise RedactionDriftError(f"{path} not found.")
    try:
        return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    except sqlite3.OperationalError as exc:
        raise RedactionDriftError(f"{path} is unreadable; refusing normal output.") from exc


def usable_span_db(path: Path | str) -> bool:
    """A missing or zero-byte file is treated as "no span layer yet"."""

    path = Path(path)
    return path.exists() and path.stat().st_size > 0


def load_guard(
    span_db_path: Path | str,
    *,
    mapping: SchemaMapping = DEFAULT_MAPPING,
) -> Guard:
    """Materialize a guard from a span-store file path, read-only.

    Missing, zero-byte, or not-yet-derived span stores yield a not-ready
    guard: the live flag is still checked by every predicate, and freshness
    verification refuses as soon as flagged records exist.  Freshness
    re-verification reopens the path, so later store changes are seen -- and it
    re-reads the full member set alongside the mirror (same order of work as
    the mirror read) so a span re-derivation with unchanged flags is still
    caught as membership drift.
    """

    path = Path(span_db_path).resolve()
    origin = str(path)

    def _refresh() -> tuple[bool, tuple[str, ...], frozenset[int] | None, frozenset[int] | None]:
        if not usable_span_db(path):
            return False, (), None, None
        try:
            with closing(sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)) as conn:
                sources = span_sources(conn, mapping=mapping, dialect=SQLITE)
                mirror = _mirror_flagged_ids(conn, mapping=mapping, dialect=SQLITE)
                members = frozenset(
                    _read_span_ids(conn, sources=sources, mapping=mapping, dialect=SQLITE)
                )
        except sqlite3.DatabaseError as exc:
            raise RedactionDriftError(
                f"span store {origin} is unreadable; refusing normal output."
            ) from exc
        return True, sources, mirror, members

    ready, sources, _, members = _refresh()
    return Guard(
        span_member_ids=members if members is not None else frozenset(),
        span_layer_ready=ready,
        span_sources=sources,
        origin=origin,
        _refresh=_refresh,
    )


def _attached_schema_paths(conn: sqlite3.Connection) -> dict[str, str]:
    """Map attached schema name -> its database file path (``""`` for none)."""

    return {row[1]: row[2] or "" for row in conn.execute("PRAGMA database_list")}


def _same_database_file(existing_path: str, expected_path: Path) -> bool:
    """Compare an attached schema's file path against the expected one.

    An empty path (in-memory or temporary database) never matches.  When both
    files exist, identity is decided by ``os.path.samefile`` (device + inode),
    which is authoritative across case-insensitive filesystems (Windows and
    macOS) and through symlinks and hardlinks.  ``os.path.normcase`` alone is a
    no-op on POSIX, so it would miss case-only differences on a case-insensitive
    macOS volume.  When a file is absent (for example the real span store has not
    been created yet), fall back to comparing resolved paths, case-folded via
    ``os.path.normcase`` for Windows.
    """

    if not existing_path:
        return False
    try:
        if os.path.exists(existing_path) and expected_path.exists():
            return os.path.samefile(existing_path, expected_path)
    except OSError:
        pass
    try:
        resolved_existing = Path(existing_path).resolve()
    except OSError:
        return False
    return os.path.normcase(str(resolved_existing)) == os.path.normcase(str(expected_path))


def _remove_impostor_attach_file(bound_path: str, *, real_span_path: Path) -> None:
    """Delete a zero-byte file a literal-path ATTACH created instead of binding.

    A connection without ``uri=True`` treats the ``file:...?mode=ro`` URI as a
    literal path and on POSIX may create a zero-byte file by that name.  A
    zero-byte file whose name carries the URI query suffix is never real data,
    so it is removed; a non-empty file, or one that resolves to the real span
    path, is left untouched.
    """

    if not bound_path:
        return
    candidate = Path(bound_path)
    if "?mode=ro" not in candidate.name:
        return
    if _same_database_file(bound_path, real_span_path):
        return
    with suppress(OSError):
        if candidate.is_file() and candidate.stat().st_size == 0:
            candidate.unlink()


@contextmanager
def attached_guard(
    conn: sqlite3.Connection,
    span_db_path: Path | str,
    *,
    mapping: SchemaMapping = DEFAULT_MAPPING,
    attach_name: str = DEFAULT_ATTACH_NAME,
) -> Iterator[AttachedGuard]:
    """Attach the span store read-only and yield an anti-join guard.

    The primary connection should be opened with URI support (for example
    through :func:`open_readonly`).  If the span store is missing or still a
    zero-byte placeholder, the yielded guard checks only the live flag -- and
    this refuses outright when flagged records already exist, because spans
    could not have been derived yet, or when the primary store carries a
    drift witness recording span members, because then the span layer
    existed and its absence is a loss rather than a fresh install.  An
    attached span store is likewise verified against the recorded witness's
    member digest.

    If ``attach_name`` is already attached on this connection, its bound file
    is verified against ``span_db_path`` rather than trusted blindly: a
    mismatch (including an in-memory/temporary attachment with no file path)
    refuses instead of silently reading the wrong database.
    """

    path = Path(span_db_path).resolve()
    validate_name(attach_name, kind="attach name")

    if not usable_span_db(path):
        try:
            _assert_primary_ids_well_formed(conn, mapping=mapping, dialect=SQLITE)
            _assert_primary_flags_well_formed(conn, mapping=mapping, dialect=SQLITE)
            has_flagged = _has_flagged(conn, mapping=mapping, dialect=SQLITE)
        except RedactionDriftError:
            raise
        except sqlite3.DatabaseError as exc:
            raise RedactionDriftError(
                "primary store is unreadable; refusing normal output."
            ) from exc
        if has_flagged:
            raise RedactionDriftError(
                "span isolation is unavailable while the primary store has flagged "
                "records; rebuild the span derivation layer before serving."
            )
        try:
            # Even with zero flags: a recorded drift witness means spans
            # existed, so a missing/zero-byte span store is a loss, not a
            # fresh install -- refuse instead of un-hiding every span row.
            _assert_drift_witness_satisfied(
                conn, mapping=mapping, dialect=SQLITE, namespace=None, sources=(), origin=str(path)
            )
        except RedactionDriftError:
            raise
        except sqlite3.DatabaseError as exc:
            raise RedactionDriftError(
                "primary store is unreadable; refusing normal output."
            ) from exc
        yield AttachedGuard(
            namespace=None,
            span_sources=(),
            span_layer_ready=False,
            mapping=mapping,
            dialect=SQLITE,
        )
        return

    attached_paths = _attached_schema_paths(conn)
    attached_here = attach_name not in attached_paths
    if attached_here:
        try:
            conn.execute(
                f"ATTACH DATABASE ? AS {SQLITE.quote_identifier(attach_name)}",
                (f"{path.as_uri()}?mode=ro",),
            )
        except sqlite3.DatabaseError as exc:
            # The read-only attach uses a URI filename, which a connection
            # opened without uri=True treats as a literal (nonexistent) file
            # path.  Probe the span store directly so the refusal blames the
            # actual culprit instead of a perfectly readable span store.  The
            # probe must read the header: SELECT 1 is a constant that never
            # touches the file, so a corrupt store would slip through and be
            # misreported.  Reading sqlite_master forces the header to parse,
            # which is where SQLite builds that defer ATTACH-time validation
            # raise "file is not a database".
            try:
                with closing(sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)) as probe:
                    probe.execute("SELECT count(*) FROM sqlite_master").fetchone()
            except sqlite3.DatabaseError:
                raise RedactionDriftError(
                    f"span store {path} is unreadable; refusing normal output."
                ) from exc
            raise RedactionDriftError(
                f"could not attach span store {path}: the primary connection does not "
                "accept URI filenames; open it with sqlite3.connect(..., uri=True) or "
                "jaenys.sql.sqlite.open_readonly."
            ) from exc
        # A non-URI connection may also "succeed" by opening (even creating) a
        # file literally named ``file:...?mode=ro`` instead of the span store.
        # Serving against that empty impostor would silently drop the whole
        # span layer, so verify what actually got bound before trusting it.
        bound_path = _attached_schema_paths(conn).get(attach_name, "")
        if not _same_database_file(bound_path, path):
            conn.execute(f"DETACH DATABASE {SQLITE.quote_identifier(attach_name)}")
            _remove_impostor_attach_file(bound_path, real_span_path=path)
            raise RedactionDriftError(
                f"could not attach span store {path}: the primary connection does not "
                "accept URI filenames; open it with sqlite3.connect(..., uri=True) or "
                "jaenys.sql.sqlite.open_readonly."
            )
    elif not _same_database_file(attached_paths[attach_name], path):
        raise RedactionDriftError(
            f"attach name {attach_name!r} is already bound to a different database on "
            "this connection; refusing to read the wrong span store."
        )
    try:
        try:
            _assert_primary_ids_well_formed(conn, mapping=mapping, dialect=SQLITE)
            _assert_primary_flags_well_formed(conn, mapping=mapping, dialect=SQLITE)
            sources = span_sources(conn, mapping=mapping, dialect=SQLITE, namespace=attach_name)
            _assert_attached_flag_layer_current(
                conn, mapping=mapping, dialect=SQLITE, namespace=attach_name, sources=sources
            )
            _assert_drift_witness_satisfied(
                conn,
                mapping=mapping,
                dialect=SQLITE,
                namespace=attach_name,
                sources=sources,
                origin=str(path),
            )
            span_versions = _span_group_versions(
                conn, sources=sources, mapping=mapping, dialect=SQLITE, namespace=attach_name
            )
        except RedactionDriftError:
            raise
        except sqlite3.DatabaseError as exc:
            raise RedactionDriftError(
                f"span store {path} is unreadable; refusing normal output."
            ) from exc
        yield AttachedGuard(
            namespace=attach_name,
            span_sources=sources,
            span_layer_ready=True,
            mapping=mapping,
            dialect=SQLITE,
            span_versions=span_versions,
        )
    finally:
        if attached_here:
            conn.execute(f"DETACH DATABASE {SQLITE.quote_identifier(attach_name)}")


@contextmanager
def _writable_connection(target: Path | str | sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Yield a writable connection; open (and later close) it when given a path."""

    if isinstance(target, sqlite3.Connection):
        yield target
        return
    try:
        conn = sqlite3.connect(Path(target))
    except sqlite3.OperationalError as exc:
        raise RedactionDriftError(f"{target} is not writable; nothing was changed.") from exc
    try:
        yield conn
    finally:
        conn.close()


def write_flags(
    primary: Path | str | sqlite3.Connection,
    flagged_ids: Iterable[int] | DerivedLayers,
    *,
    mapping: SchemaMapping = DEFAULT_MAPPING,
) -> None:
    """Replace the live flag layer: ``flagged_ids`` -> 1, every other row -> 0.

    This is the detector's write-back half.  Pair it with
    :func:`write_span_layer` from the same derivation so the mirror written
    there matches the flags written here; flags written alone leave the
    span store stale, and the guard will refuse (by design) until the span
    layer is re-derived.

    Prefer passing the whole :class:`~jaenys.derivation.DerivedLayers`
    (the same object :func:`write_span_layer` takes): the flags are then taken
    from ``layers.flagged_ids`` AND the **drift witness** -- span-member
    count + digest under ``mapping.meta_table`` -- is recorded in the same
    transaction.  The witness is what makes a lost or emptied span store
    refuse even when zero records are flagged.  Passing bare ids writes flags
    only, and clears any existing span witness in the same transaction: bare
    ids carry no span membership, so keeping a witness from an earlier
    ``DerivedLayers`` write would describe a different derivation and make a
    freshly rewritten span layer look stale.  The store then falls back to
    mirror freshness (legacy flow).

    The whole update is one transaction: on any failure -- including a
    flagged id that matches no record, which would otherwise desync the
    layers invisibly -- nothing is changed.
    """

    layers = flagged_ids if isinstance(flagged_ids, DerivedLayers) else None
    source_ids: Iterable[int] = layers.flagged_ids if layers is not None else flagged_ids
    ids = sorted({coerce_record_id(record_id, origin="write_flags") for record_id in source_ids})
    table = SQLITE.quote_identifier(mapping.record_table)
    id_column = SQLITE.quote_identifier(mapping.record_id_column)
    flag_column = SQLITE.quote_identifier(mapping.flag_column)
    with _writable_connection(primary) as conn:
        try:
            with conn:
                conn.execute(f"UPDATE {table} SET {flag_column} = 0")
                if ids:
                    cursor = conn.executemany(
                        f"UPDATE {table} SET {flag_column} = 1 WHERE {id_column} = ?",
                        [(record_id,) for record_id in ids],
                    )
                    if cursor.rowcount < len(ids):
                        raise RedactionDriftError(
                            f"write_flags: {len(ids) - cursor.rowcount} flagged id(s) match no "
                            f"record in {mapping.record_table}; nothing was changed."
                        )
                    if cursor.rowcount > len(ids):
                        # More rows matched than distinct ids means the id
                        # column is not unique, so a flag cannot be addressed
                        # to a single record.
                        raise RedactionDriftError(
                            f"write_flags: {mapping.record_id_column} values in "
                            f"{mapping.record_table} are not unique; nothing was changed."
                        )
                if mapping.meta_table is not None:
                    if layers is not None:
                        _write_drift_witness(conn, layers, mapping=mapping)
                    elif _guard.table_exists(conn, mapping.meta_table, dialect=SQLITE):
                        # Bare ids carry no span membership, so any witness left
                        # from an earlier DerivedLayers write would describe a
                        # different derivation and make a freshly rewritten span
                        # layer look stale.  Clear only the span witness row (not
                        # the table) and fall back to mirror freshness, rather
                        # than leave a contradictory record.
                        conn.execute(
                            f"DELETE FROM {SQLITE.quote_identifier(mapping.meta_table)} "
                            f"WHERE {META_LAYER_COLUMN} = ?",
                            (META_LAYER_SPAN,),
                        )
        except sqlite3.DatabaseError as exc:
            raise RedactionDriftError(
                "primary store is unwritable; the flag layer was not changed."
            ) from exc


def _write_drift_witness(
    conn: sqlite3.Connection, layers: DerivedLayers, *, mapping: SchemaMapping
) -> None:
    """Record the span derivation (count + digest) in the primary store.

    Runs inside write_flags' transaction.  A same-named table with a foreign
    shape fails the INSERT and rolls the whole write back -- rename via
    ``mapping.meta_table`` or set it to None.
    """

    meta = SQLITE.quote_identifier(mapping.meta_table)
    members = layers.span_member_ids
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {meta} ("
        f"{META_LAYER_COLUMN} TEXT PRIMARY KEY,"
        f"{META_COUNT_COLUMN} INTEGER NOT NULL,"
        f"{META_DIGEST_COLUMN} TEXT NOT NULL,"
        f"{META_DERIVED_AT_COLUMN} TEXT NOT NULL)"
    )
    conn.execute(f"DELETE FROM {meta} WHERE {META_LAYER_COLUMN} = ?", (META_LAYER_SPAN,))
    conn.execute(
        f"INSERT INTO {meta} VALUES (?, ?, ?, ?)",
        (
            META_LAYER_SPAN,
            len(members),
            span_member_digest(members),
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ),
    )


def write_span_layer(
    span_store: Path | str | sqlite3.Connection,
    layers: DerivedLayers,
    *,
    mapping: SchemaMapping = DEFAULT_MAPPING,
) -> None:
    """Rebuild the span store from a derivation: members + mirror snapshot.

    Drops and recreates the span-member table and the mirror table in one
    transaction.  The mirror rows written here are the freshness witness
    the guard verifies against, so re-run the derivation and this write
    after every flag change -- and reload any materialized guard afterwards.

    ``layers.flagged_ids`` becomes the mirror's flag snapshot: it must be
    exactly what :func:`write_flags` wrote to the primary store (use the
    same :class:`~jaenys.derivation.DerivedLayers` for both calls).
    """

    members_table = SQLITE.quote_identifier(mapping.span_member_table)
    member_id_column = SQLITE.quote_identifier(mapping.span_member_id_column)
    group_id_column = SQLITE.quote_identifier(mapping.span_group_id_column)
    mirror_table = SQLITE.quote_identifier(mapping.mirror_table)
    mirror_id_column = SQLITE.quote_identifier(mapping.mirror_id_column)
    reason_column = SQLITE.quote_identifier(mapping.mirror_reason_column)
    flag_column = SQLITE.quote_identifier(mapping.mirror_flag_column)
    with _writable_connection(span_store) as conn:
        try:
            with conn:
                # This reference writer produces a non-versioned layer, so a
                # stale version-group table must not survive to be read later
                # as a live versioned source.
                if mapping.span_group_table is not None:
                    conn.execute(
                        f"DROP TABLE IF EXISTS {SQLITE.quote_identifier(mapping.span_group_table)}"
                    )
                conn.execute(f"DROP TABLE IF EXISTS {members_table}")
                conn.execute(
                    f"CREATE TABLE {members_table} ("
                    f"{group_id_column} INTEGER NOT NULL,"
                    f"{member_id_column} INTEGER NOT NULL,"
                    f"position INTEGER NOT NULL,"
                    f"PRIMARY KEY ({group_id_column}, {member_id_column}))"
                )
                for span_id, session in enumerate(layers.sessions, start=1):
                    conn.executemany(
                        f"INSERT INTO {members_table} VALUES (?, ?, ?)",
                        [
                            (span_id, member_id, position)
                            for position, member_id in enumerate(session.member_ids, start=1)
                        ],
                    )
                conn.execute(f"DROP TABLE IF EXISTS {mirror_table}")
                conn.execute(
                    f"CREATE TABLE {mirror_table} ("
                    f"{mirror_id_column} INTEGER NOT NULL,"
                    f"{reason_column} TEXT NOT NULL,"
                    f"{flag_column} INTEGER NOT NULL DEFAULT 1)"
                )
                conn.executemany(
                    f"INSERT INTO {mirror_table} VALUES (?, ?, 1)",
                    [
                        (record_id, mapping.reason_flagged)
                        for record_id in sorted(layers.flagged_ids)
                    ],
                )
                conn.executemany(
                    f"INSERT INTO {mirror_table} VALUES (?, ?, 1)",
                    [
                        (record_id, mapping.reason_span)
                        for record_id in sorted(layers.span_member_ids)
                    ],
                )
        except sqlite3.DatabaseError as exc:
            raise RedactionDriftError(
                "span store is unwritable; the span layer was not changed."
            ) from exc


def status(
    primary_db_path: Path | str,
    span_db_path: Path | str,
    *,
    mapping: SchemaMapping = DEFAULT_MAPPING,
) -> dict[str, Any]:
    """Counts-only readiness report for a SQLite store pair.

    Writes nothing and emits no record content.
    """

    primary_path = Path(primary_db_path).resolve()
    span_path = Path(span_db_path).resolve()
    guard = load_guard(span_path, mapping=mapping)
    report: dict[str, Any] = {
        "primary_db": {"path": str(primary_path), "exists": primary_path.exists()},
        "span_db": {
            "path": str(span_path),
            "exists": span_path.exists(),
            "size_bytes": span_path.stat().st_size if span_path.exists() else 0,
            "ready": guard.span_layer_ready,
            "span_sources": list(guard.span_sources),
            "unique_span_member_ids": len(guard.span_member_ids),
        },
        "guard": None,
    }
    if not primary_path.exists():
        report["primary_db"]["error"] = "primary database not found"
        return report
    with closing(open_readonly(primary_path)) as conn:
        try:
            if not _guard.table_exists(conn, mapping.record_table, dialect=SQLITE):
                report["primary_db"]["error"] = f"{mapping.record_table} table not found"
                return report
            report["guard"] = _guard.status(conn, guard, mapping=mapping, dialect=SQLITE)
        except RedactionDriftError:
            raise
        except sqlite3.DatabaseError as exc:
            raise RedactionDriftError(
                f"primary store {primary_path} is unreadable; refusing normal output."
            ) from exc
        report["primary_db"]["records"] = report["guard"].pop("records")
        report["primary_db"]["flagged"] = report["guard"].pop("flagged")
        try:
            _guard.assert_guard_current(conn, guard, mapping=mapping, dialect=SQLITE)
            report["layers_in_sync"] = True
        except RedactionDriftError as exc:
            report["layers_in_sync"] = False
            report["refusal"] = str(exc)
        except sqlite3.DatabaseError as exc:
            raise RedactionDriftError(
                f"primary store {primary_path} is unreadable; refusing normal output."
            ) from exc
    return report
