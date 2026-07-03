# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-03

Initial release.

- Store-agnostic core: two-layer sensitivity model (live per-record flag +
  derived contiguous spans with a mirrored flag snapshot), three visibility
  states (`VISIBLE` / `BLUR` / `HIDDEN`), and fail-closed layer-sync
  verification (`RedactionDriftError` on any unprovable state).
- Freshness verification also re-reads the span-member set and refuses when
  membership changed since the guard was loaded, so a span re-derivation
  that touches no flag is still caught. Reload the guard after every
  re-derivation.
- The sensitivity-flag domain is strictly `{0, 1}` everywhere: any other
  value (`2`, `-1`, junk text) is treated as corruption and refused, never
  served as "not flagged". All stored ids and flags route through
  fail-closed coercers, and refusal messages truncate offending values so
  record content cannot leak into error strings.
- SQL backend over PEP 249 connections: SQLite (reference), PostgreSQL,
  MySQL/MariaDB, SQL Server, Oracle, and a generic ANSI dialect; predicate
  builders, single-connection anti-join guard, materialized two-store guard,
  counts-only `status()`, and a `jaenys` command-line status check
  (also runnable as `python -m jaenys`).
- Dialects carry per-statement parameter budgets (chunked `NOT IN`
  predicates refuse up front instead of hitting opaque driver limits),
  identifier case folding (Oracle upper-folds by default), and
  current-schema scoping for `information_schema` introspection.
- Store adapters via a 4-method protocol: in-memory reference, MongoDB,
  Redis, DynamoDB, Couchbase, Firestore. No driver imports; callers pass
  their own client objects.
- Populate-the-template derivation scaffolding in
  `jaenys.derivation`: `KeywordDetector` (keywords + optional
  `extra` callable) computes the flag layer; `CueSessionDeriver`
  (start/end cues, optional inactivity timeout, optional `is_start` /
  `is_end` callables) computes contiguous sessions, each recording its
  `end_reason` (`end_cue` / `timeout` / `end_of_data`); `derive_layers()`
  bundles both into a `DerivedLayers`. Ids coerce fail-closed, and rows
  out of time order refuse when a timeout is configured instead of
  silently producing wrong sessions.
- SQLite writers for the derivation output: `sql.sqlite.write_flags`
  (transactional flag write-back that refuses ids matching no record) and
  `sql.sqlite.write_span_layer` (rebuilds span members + the flag-mirror
  snapshot in exactly the shape the guard reads). Both accept a path or an
  open connection and honor `SchemaMapping`.
- Drift witness: passing the full `DerivedLayers` to `write_flags`
  records a span-derivation witness (member count + digest) in the primary
  store (`SchemaMapping.meta_table`, default `span_store_meta`). Every SQL
  guard verifies the span layer against it, so a missing, emptied, or
  swapped span store refuses even when zero records are flagged -- the one
  drift case the flag mirror cannot catch. Stores without the witness
  (bare-id writes, hand-rolled pipelines, `meta_table=None`) keep the
  flags-only semantics.
- Documented app-integration contract ("Wiring it into an app": one
  chokepoint, blur is the UI's job, a refusal answers 503 and is never
  served around) with a runnable route example,
  `examples/demo_app_integration.py`.
- `local_llm_guard`: strict local-endpoint URL enforcement (scheme, host
  allowlist, credentials, query/fragment, path, port range) and
  reasoning-block/JSON-fence stripping for local LLM responses. Tag
  mentions wrapped in matching quotes/backticks (e.g. `"</think>"` inside
  answer text) are preserved, never treated as reasoning markers.
- Typed (`py.typed` in both packages), zero runtime dependencies,
  Python 3.10+, Apache-2.0.
