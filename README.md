# Jaenys

[![CI](https://github.com/moizari/jaenys/actions/workflows/ci.yml/badge.svg)](https://github.com/moizari/jaenys/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/jaenys?cacheSeconds=300)](https://pypi.org/project/jaenys/)

**Redaction drift** is how redaction systems leak. Jaenys is a
span-aware, fail-closed visibility engine: when it cannot prove its two
sensitivity layers are in sync, it refuses to serve. It never silently
serves stale isolation.

Zero runtime dependencies. Bring your own database driver. Python 3.10+.
Apache-2.0.

```
pip install jaenys
```

---

## What is redaction drift?

Most data-protection tooling runs as a pipeline: detect sensitive content,
write the result somewhere, filter at serve time against that result. The
detection output and the serving filter are two separate pieces of state,
and nothing in the pipeline notices when they drift apart. That gap has a
name: **redaction drift**.

- A record gets re-flagged after review, but the derived exclusion index was
  built before the edit. The serving layer keeps using the stale index.
- A batch import adds flagged records; the job that rebuilds the exclusion
  layer fails halfway. Every query after that silently under-filters.
- The exclusion store is missing, unreadable, or a zero-byte placeholder.
  The filter finds nothing to exclude, so it excludes nothing.

In each case the system keeps answering queries, and every answer is a leak.
Detection missed nothing. The filter trusted state that no longer matched
reality and kept serving anyway: silent under-filtering, with no error to
page anyone.

Jaenys makes that failure mode impossible by construction: if the
layers cannot be proven equal, the engine raises instead of serving.

## The model: two layers, three states

Sensitivity is tracked at two independently maintained layers:

1. **The live flag layer** (primary store): a per-record flag
   (`sensitive = 1`) that answers "is *this record's own content*
   sensitive?" It changes immediately when a record is flagged or cleared.
   The flag domain is exactly **`{0, 1}`**: any other value (`2`, `-1`,
   junk text) is treated as corruption and refused, never served as
   "not flagged".
2. **The derived span layer** (span store): contiguous **spans**, stretches
   of records that are sensitive *as a group* because of surrounding
   context, even when individual rows in the stretch look harmless. Spans
   are recomputed by the caller's derivation step, together with a
   **mirror**: a snapshot of the flag layer as it was at derivation time.

Every record then resolves to exactly one of three visibility states:

| State | Condition | On normal surfaces |
|---|---|---|
| `VISIBLE` | not flagged, not in any span | served normally |
| `BLUR` | flagged, but outside every span | served, marked `"blurred": true` for redacted presentation |
| `HIDDEN` | inside a derived span | not served at all; existence masked |

The asymmetry is deliberate, and it is what **span-aware** means. A
standalone flagged record is a *known*,
bounded disclosure, deliverable behind a blur/redaction mark. A span is a
*contextual* disclosure: serving any row from it (even an unflagged one)
reveals what the stretch was about. So in-span rows are excluded entirely,
and the span always wins over the individual flag.

## The guarantee: fail closed on drift

Freshness is checked through the mirror. Before serving, the engine compares
the **live** flagged-ID set against the **mirrored** flagged-ID set and
refuses (`RedactionDriftError`) unless they are provably equal:

- Flag set live but absent from the mirror: **refuse.** A new flag can
  imply a new span no stale member table knows to hide.
- Mirror claims a flag the live layer no longer has: **refuse.** The span
  layer was derived from a world that no longer exists.
- Flagged records exist but the span store is missing, empty-by-accident, or
  unreadable: **refuse.**
- The primary store's **drift witness** records span members, but the
  span store is missing, emptied, or no longer matches it: **refuse -- even
  with zero flagged records**, the one drift case the mirror alone cannot
  catch. (The writers record the witness; see "Producing the layers".) This
  is a SQL-backend guarantee only -- the adapter path has no witness
  equivalent, so a zero-flag span wipe there is caught only by keeping a
  loaded guard alive (membership drift then refuses on re-verification) and
  by always rebuilding the derivation rather than editing span data in place.
- Span **membership** changed since a materialized guard was loaded, because
  a re-derivation pulled records into (or out of) spans without touching any
  flag: **refuse.** The guard never silently adopts the new set; **reload
  the guard after every derivation rebuild.**
- Store unreachable mid-verification: **refuse** (adapters must raise, never
  guess).

A refusal is not an outage. Reads of the primary store still work; only the
*serve-normally* path is blocked until the caller re-runs its span
derivation. The error message says exactly that.

| | typical filter-at-serve DLP | this engine |
|---|---|---|
| Detection output vs serving filter | trusted to agree | proven to agree on every serve |
| Stale exclusion index | silently under-filters | refuses to serve |
| Missing/corrupt exclusion store | filter matches nothing, serves everything | refuses to serve |
| Contextually sensitive stretches | per-record decisions only | contiguous spans hide whole stretches, including neutral rows inside |
| Flagged-but-deliverable records | binary allow/deny | explicit `BLUR` state for redacted presentation |

## Quickstart (SQLite, stdlib only)

Point a `SchemaMapping` at your table/column names, then let the guard build
your WHERE clause:

```python
from contextlib import closing

from jaenys import SchemaMapping
from jaenys.sql import sqlite

mapping = SchemaMapping(
    record_table="tickets",         # your names; defaults shown in the docs
    record_id_column="ticket_id",
    flag_column="sensitive",
)

with closing(sqlite.open_readonly("primary.db")) as conn:
    with sqlite.attached_guard(conn, "span.db", mapping=mapping) as guard:
        predicate = guard.predicate("t", include_blur=True)   # BLUR rows kept
        rows = conn.execute(
            f"SELECT t.* FROM tickets t WHERE {predicate.sql}",
            predicate.params,
        ).fetchall()
```

`attached_guard` verifies layer freshness *before* yielding. If a flag was
edited without re-deriving the span layer, the `with` line raises and no
query runs. `include_blur=False` (the default) additionally drops `BLUR`
rows for strictly-`VISIBLE` surfaces.

On the `attached_guard` path above, the SQL predicate already drops every
HIDDEN row, so a served row is BLUR exactly when its own flag is set: read
that off the row directly (`bool(row["sensitive"])`).

For the materialized-guard and adapter paths, `annotate_rows` marks blur and
drops in-span rows for you, given the span-member id set those guards expose
as `guard.span_member_ids`:

```python
from jaenys import annotate_rows

served = annotate_rows(rows_as_dicts, span_member_ids=guard.span_member_ids,
                       id_key="ticket_id", flag_key="sensitive")
# in-span rows are dropped; kept rows carry "blurred": True/False
```

And check any SQLite pair from the command line (counts only, never record
content):

```
jaenys --primary-db primary.db --span-db span.db
```

(`python -m jaenys` does the same.) It exits `0` only when both stores
are found and the layers verify in sync, so it can gate shell pipelines
(`... && deploy`).

Four runnable walkthroughs live in [`examples/`](examples/):
`demo_sql_guard.py` (SQL path) and `demo_store_adapter.py` (adapter path)
each serve the three states, then flip a flag without re-deriving and show
the engine refusing. `demo_span_derivation.py` populates the two
derivation templates below and produces both layers from a raw transcript.
`demo_app_integration.py` wires the guard into an application route (see
"Wiring it into an app").

## Producing the layers: populate the templates

The engine never decides what is sensitive -- but you don't have to build
the deciding machinery from scratch. `jaenys.derivation` ships two
templates you populate with your own vocabulary, and the SQLite module
carries the matching writers:

```python
from jaenys.derivation import KeywordDetector, CueSessionDeriver, derive_layers
from jaenys.sql import sqlite

detector = KeywordDetector(keywords=("pin", "password", "card number"))
deriver = CueSessionDeriver(
    start_cues=("verify your identity",),
    end_cues=("verification complete",),
    timeout_minutes=30,            # inactivity also closes an open session
)

layers = derive_layers(rows, detector=detector, deriver=deriver,
                       id_key="ticket_id", text_key="body", at_key="sent_at")

sqlite.write_flags("primary.db", layers, mapping=mapping)   # flags + drift witness
sqlite.write_span_layer("span.db", layers, mapping=mapping)  # spans + mirror
```

`KeywordDetector` flags records whose own text matches your terms.
`CueSessionDeriver` opens a session on a start cue, closes it on an end
cue or the inactivity timeout, and every row in between becomes a span
member -- however harmless it looks alone; each session records why it
ended (`end_cue` / `timeout` / `end_of_data`). The templates match exactly
what you populate them with: plain substring matching, case-insensitive by
default, no NLP. Anything a vocabulary can't express plugs in as a
callable (`extra=` on the detector, `is_start=` / `is_end=` on the
deriver) -- that is where a real classifier like Presidio composes in.

Handing `write_flags` the whole `DerivedLayers` (rather than bare ids) also
records a **drift witness** in the primary store
(`SchemaMapping.meta_table`, default `span_store_meta`): the span-member
count and a digest of the member ids. Every SQL guard then refuses if the
span store goes missing, is emptied, or stops matching the recorded
derivation -- *even when nothing is flagged*, which is the one loss the
flag mirror cannot catch. Pass bare ids (or set `meta_table=None`) to
skip the witness and keep the older, mirror-only semantics.

Re-run the derivation and both writes after every flag change; the guard
refuses to serve until you do. For non-SQLite stores, feed the same
`layers` to your own writer in the adapter's documented shape
(`InMemoryAdapter.rebuild_derived_layer(span_members=layers.span_member_ids)`
is the reference).

## Database support

The core is pure Python over ID sets, so the guarantee is identical on every
backend. Two pillars connect it to real stores; the library imports **no**
drivers. You pass in the PEP 249 connection or client object you already
have.

| Database | How | Tier | Same-technology aliases |
|---|---|---|---|
| SQLite | `jaenys.sql` (+ `sqlite`) | reference: full matrix in CI | |
| In-memory | `adapters.InMemoryAdapter` | reference: full matrix in CI | |
| PostgreSQL | `sql` with `POSTGRESQL` dialect | integration-tested (env-gated) | CockroachDB, Redshift, AlloyDB, Aurora-PostgreSQL |
| MongoDB | `adapters.MongoDBAdapter` | integration-tested (env-gated) + contract-tested | Amazon DocumentDB, Azure Cosmos DB (Mongo API), FerretDB |
| Redis | `adapters.RedisAdapter` | integration-tested (env-gated) + contract-tested | Valkey, KeyDB, Dragonfly, ElastiCache/MemoryDB |
| MySQL | `sql` with `MYSQL` dialect | experimental: golden-SQL tested | MariaDB, TiDB, Aurora-MySQL, Vitess/PlanetScale |
| SQL Server | `sql` with `MSSQL` dialect | experimental: golden-SQL tested | Azure SQL |
| Oracle | `sql` with `ORACLE` dialect | experimental: golden-SQL tested | |
| DynamoDB | `adapters.DynamoDBAdapter` | experimental: contract-tested against fakes | |
| Couchbase | `adapters.CouchbaseAdapter` | experimental: contract-tested against fakes | |
| Firestore | `adapters.FirestoreAdapter` | experimental: contract-tested against fakes | |

Tier meanings, honestly: **reference** backends run the complete fail-closed
and three-state matrix on every CI run with no external services.
**Integration-tested** backends run the same matrix against live servers
when `JAENYS_PG_DSN` / `JAENYS_MONGO_URI` /
`JAENYS_REDIS_URL` are set (auto-skipped otherwise), plus
driver-free contract tests in plain CI. **Experimental** backends are
contract-complete and tested against hand-rolled client fakes or golden SQL
(quoting, all five PEP 249 paramstyles, introspection, anti-joins), but have
not been exercised against a live server by CI.

### Extension point 1: any PEP 249 database via `Dialect`

Everything engine-specific about a SQL database fits in one small frozen
value object: identifier quoting, paramstyle, introspection strategy,
IN-list chunk size. Unknown drivers auto-detect to a standards-leaning ANSI
fallback via their mandatory `paramstyle` attribute, or you construct a
dialect yourself:

```python
from jaenys.sql import Dialect, guard_for_connection

my_engine = Dialect(
    name="my-engine",
    paramstyle="numeric",        # any of the five PEP 249 styles
    quote_open='"', quote_close='"',
    introspection="information_schema",
    max_in_params=500,
)
guard = guard_for_connection(conn, mapping=mapping, dialect=my_engine)
```

### Extension point 2: any other store via `StoreAdapter`

A store backs the engine by answering four questions:

```python
class StoreAdapter(Protocol):
    def span_member_ids(self) -> frozenset[int] | None: ...  # None = never derived
    def flagged_ids(self) -> frozenset[int]: ...
    def mirror_flagged_ids(self) -> frozenset[int] | None: ...
    def counts(self) -> dict[str, int]: ...
```

Implement those (raising `RedactionDriftError` when the store is unreachable,
never returning a guess) and `load_guard_from_adapter`,
`assert_adapter_current`, and `adapter_status` all work.
`InMemoryAdapter` is the reference implementation and the template; the test
suite's contract tests run against it and against client fakes for all five
shipped adapters.

## Two topologies

**Single connection, namespaced span layer**: the scalable chokepoint. The
span tables live in a schema/database the same connection can reach
(PostgreSQL schema, MySQL database, SQL Server `db.schema`, Oracle schema,
SQLite `ATTACH`); set `SchemaMapping.span_namespace` and the guard pushes
anti-joins into the WHERE clause so the database excludes spans server-side:

```python
guard = guard_for_connection(conn, mapping=SchemaMapping(span_namespace="span_layer"))
predicate = guard.predicate("r")               # NOT EXISTS anti-joins inside
```

**Two connections, materialized guard**: the universal path. Load the span
layer on its own connection, verify against the primary on another. The
comparison happens in Python over ID sets, so it needs nothing but plain
`SELECT`s. It runs on literally any PEP 249 database, including mixed
engines (primary in PostgreSQL, span layer in SQLite) and physically
isolated span stores:

```python
from jaenys.sql import load_guard, assert_guard_current

guard = load_guard(span_conn, origin="span store")
assert_guard_current(primary_conn, guard)      # raises on drift
```

This path is the lowest-common-denominator guarantee behind "works with any
database."

## Wiring it into an app

The engine is a library, not a proxy: it protects exactly the queries that
are routed through it. Three rules make an integration safe end to end:

1. **One chokepoint.** Give the app a single data-access function (or
   layer) that applies the guard, and let no other code touch the store.
   Every surface built on top -- pages, search, exports -- is then
   filtered automatically, because it cannot reach the data any other way.
   A raw query that skips the predicate returns everything.
2. **Blur is the UI's job.** Deliverable-but-flagged rows arrive marked
   `"blurred": true`; the engine never alters text. Fog them, collapse
   them, gate them behind a click -- the presentation is yours. In-span
   rows need no UI handling at all: they never arrive.
3. **A refusal is availability, not a crash.** When the layers drift, the
   chokepoint raises `RedactionDriftError`. Catch it at the route boundary and
   answer "temporarily unavailable" (a 503); the app stays up, primary
   reads still work, and service resumes the moment the span derivation
   re-runs. Never catch-and-serve-anyway -- the refusal *is* the security
   feature.

[`examples/demo_app_integration.py`](examples/demo_app_integration.py) is
a runnable version of all three: a chokepoint route serving JSON, the blur
mark rendered, a drifted store answered with a 503, and the bypassing raw
query shown for contrast (counts only).

## `local_llm_guard`: keep local AI actually local

The second, smaller package solves an adjacent problem: pipelines that
analyze sensitive data with a *local* LLM must be able to prove the endpoint
is local. One misconfigured base URL and "100% local analysis" quietly
becomes an upload.

```python
from local_llm_guard import enforce_local_url, parse_json_content

base = enforce_local_url("http://localhost:11434")   # returns normalized URL
# raises LocalEndpointError for anything non-local:
#   remote hosts, localhost.evil.example, embedded credentials,
#   query strings, non-root paths, non-http(s) schemes
```

Allowed hosts default to `localhost`, `127.0.0.1`, `::1`, and
`host.docker.internal`; pass `allowed_hosts=` to tighten or extend.

It also ships hardened cleanup for local reasoning models whose output wraps
JSON in reasoning blocks and code fences: `strip_reasoning_blocks` (handles
unterminated blocks), `strip_json_wrapper`, and `parse_json_content`. That is
deliberately all it does. For schema-validated LLM calls, retries, and
structured output, use [Instructor](https://github.com/instructor-ai/instructor),
[Guardrails](https://github.com/guardrails-ai/guardrails), or
[Outlines](https://github.com/dottxt-ai/outlines).

## Non-goals and positioning

- **Not an entity detector.** The engine never decides *what* is sensitive;
  it governs visibility of whatever your flag layer and span derivation say.
  The `derivation` templates match exactly the keywords and cues you
  populate them with -- for real entity detection, compose with
  [Presidio](https://github.com/microsoft/presidio) or your own classifier
  (plug it in as the detector's `extra` callable).
- **Not an automatic span deriver.** Re-deriving spans and rewriting the
  mirror stays a caller-triggered pipeline step; the `derivation` module
  gives that step scaffolding, not autonomy. Nothing re-derives behind your
  back -- which is exactly what the freshness guarantee depends on.
  [`examples/demo_span_derivation.py`](examples/demo_span_derivation.py) is
  the worked walkthrough.
- **Not a proxy or gateway.** For routing/limiting LLM traffic, see
  [LiteLLM](https://github.com/BerriAI/litellm).
- **Not an ORM or query builder.** It hands you WHERE-clause predicates and
  filtered rows; your data access stays yours.

## Development

```
pip install -e . pytest ruff
python -m pytest            # full suite: stdlib only, no servers needed
ruff check .
```

Integration tests against live PostgreSQL/MongoDB/Redis activate via the
`JAENYS_*` environment variables listed above and skip cleanly without
them.

## License

Apache-2.0. See [LICENSE](LICENSE).
