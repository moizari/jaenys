# Comparisons

This page compares failure behavior. The README defines **redaction drift**
here:
https://github.com/moizari/jaenys#what-is-redaction-drift.

Jaenys 0.1.0 is a span-aware, prove-or-refuse serving guard. It is not a
detector, proxy, scheduler, or database permission system.

## Summary matrix

| Drift scenario | Scrub once at ingest | Re-sync or delta indexing | Permission-aware search | Postgres RLS | Jaenys 0.1.0 |
|---|---|---|---|---|---|
| Record flagged after index build | Usually missed until re-ingest | Usually fixed after next sync | Depends on permission source and sync timing | Catches only paths routed through Postgres rows | Refuses until derivation is rebuilt |
| Span-sensitive context | Usually per-record only | Usually per-record only unless custom span logic exists | Usually per-document or per-row | Per-row policy, not contextual spans | Hides whole derived spans |
| Missing exclusion layer | Often behaves like "exclude nothing" | Depends on job and serving code | Depends on backend behavior | Not applicable if data never leaves Postgres | Refuses when freshness cannot be proven |
| Zero-flag span-store loss | Usually invisible | Usually invisible unless separately audited | Depends on metadata model | Not applicable to external span stores | SQL path refuses when a drift witness exists |
| Corrupt flag value | Often treated as falsey or truthy by accident | Depends on parser | Depends on backend | Depends on column constraints | Refuses outside strict `{0, 1}` |
| Span membership changed without flag change | Usually invisible | Often invisible if sync keys are flag-based | Depends on index design | Not represented unless modeled separately | Materialized guards refuse on re-verification |
| Vector or document store outside SQL | Common target | Common target | Native to some systems | Does not apply | Covered through adapters or materialized guards |
| Refusal output | Varies | Varies | Varies | Varies | Counts-only, no record content |

## Scrub once at ingest

Ingest-time scrubbing is useful for reducing what enters an index. It is weak
against later review decisions. If a record is marked sensitive after the index
was built, the old embedding or cached retrieval layer can keep serving stale
state.

The failure mode is not bad detection. It is redaction drift between the live
flag layer and the serving layer.

Use ingest scrubbing for data minimization. Do not treat it as a freshness
proof.

## Re-sync and delta-indexing pipelines

Re-sync jobs reduce the time window for stale data. They do not remove the need
to prove that the sync completed before serving.

The read-path question is simple: what happens after a flag changes and before
the derived layer is rebuilt? If the answer is "serve from the old index until
the job catches up", the system has silent under-filtering.

Jaenys can sit around such a pipeline. Re-sync remains the repair step. The
guard changes stale reads from "serve anyway" to "refuse until rebuilt".

## Permission-aware search

Permission-aware search is about who may see a record. Redaction drift is about
whether the serving filter is current with the detection state it claims to
enforce.

These can compose. A search system can first apply user permissions, then route
candidate records through a freshness guard. The two checks answer different
questions:

- Permission check: may this caller see this record?
- Redaction-drift check: is the redaction layer current enough to serve any
  normal output?

If permissions change after indexing, permission-aware search has a similar
freshness problem. The same prove-or-refuse idea applies, but Jaenys 0.1.0
models sensitivity flags and spans rather than user-specific ACLs.

## Postgres row-level security

Postgres RLS is strong for per-row policy inside Postgres. It is not a general
answer to redaction drift.

RLS does not cover rows already copied into a vector database, object store,
cache, export, or derived span layer. It also does not express span-aware
redaction by itself. A neutral row can be sensitive because of the stretch it
sits inside, not because the row alone matches a policy.

Use RLS where it fits. Jaenys fits the serving-layer freshness problem around
derived redaction state, including non-SQL stores.

## Where Jaenys fits

Jaenys is the guard between data-access code and normal output. It assumes the
caller owns detection and derivation:

1. Write live flags in the primary store.
2. Derive span members and a mirror from those flags.
3. Serve only through a guard that verifies both layers.
4. Re-run derivation after any flag change.

When the layers agree, the guard produces predicates or filtered rows. When
they do not agree, it raises `RedactionDriftError`.

That is the practical difference: best-effort redaction tries to catch up. A
prove-or-refuse guard serves only after it proves the redaction state is
current.
