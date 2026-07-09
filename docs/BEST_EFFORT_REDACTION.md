# Best-effort redaction is a leak with a delay

Redaction drift and how to guard a pipeline against it.

This post describes the Jaenys 0.1.0 design. The README defines
**redaction drift** here:
https://github.com/moizari/jaenys#what-is-redaction-drift.

## The failure case

A retrieval pipeline has two jobs. First, detect which records are sensitive.
Second, make sure those records are not served in normal output.

That sounds simple until the two jobs stop agreeing.

A record is reviewed and re-flagged after the index was built. The primary
store now says the record is sensitive, but the derived exclusion layer still
reflects yesterday's state. The retriever keeps answering from the stale layer.
No detector failed. No query crashed. The pipeline simply trusted old serving
state and kept going.

That is **redaction drift**: the gap between what is marked sensitive and what
is actually being served. The symptom is **silent under-filtering**, where a
system keeps answering with missing redactions and no error.

Best-effort redaction usually answers this with sync jobs, deltas, and retries.
Those are useful operations. They are not proof at serve time.

## Prove-or-refuse

Jaenys uses **prove-or-refuse** behavior. Before a guarded read returns normal
output, it must prove that the live detection state and the derived serving
filter still agree. If it cannot prove that, it raises
`RedactionDriftError` instead of serving.

The design has two sensitivity layers:

1. A live flag layer in the primary store. Each record has a strict
   `sensitive` flag in the domain `{0, 1}`.
2. A derived span layer in a separate span store. It contains span members,
   plus a mirror of the flag layer captured when the derivation ran.

The serving path compares the live flag set to the mirrored flag set before it
hands out a predicate or filters rows. If the sets differ, the layer is stale.
The safe behavior is refusal.

```text
primary store                         span store

live flags  ----------------------->  mirrored flags
drift witness  <--------------------  span members

serve only after:
  live flags == mirrored flags
  drift witness == span-member receipt
```

The **drift witness** closes a quieter hole. If there are zero flagged records,
the flag mirror alone cannot prove that a span store was not deleted, emptied,
or swapped. The SQL writers can record a count and digest of the derived span
members in the primary store. SQL guards verify that receipt before serving, so
a lost span store refuses even when no record is currently flagged.

## Why spans matter

Per-record redaction misses context. A single row may look harmless, while the
stretch it sits inside is sensitive as a group.

Jaenys calls that **span-aware redaction**. A derived span hides every record in
the stretch. The engine resolves each record to one of three states:

| State | Condition | Serve behavior |
|---|---|---|
| `VISIBLE` | not flagged, not in a span | serve normally |
| `BLUR` | flagged, outside every span | serve with a blur mark |
| `HIDDEN` | inside a span | do not serve |

The span wins. A flagged row inside a span is still `HIDDEN`. A neutral row
inside a span is also `HIDDEN`, because serving it can disclose the surrounding
context.

## The demo

Jaenys 0.1.0 ships runnable examples. The shortest failure demo is:

```bash
python examples/demo_sql_guard.py
```

The walkthrough builds a synthetic primary store and span store, serves the
three visibility states, edits a flag without re-running the span derivation,
and then shows the guard raising `RedactionDriftError`. After the derivation is
rebuilt, service resumes.

The order matters:

```text
derive layers
serve normally
edit a flag
serve path refuses
rebuild derivation
serve normally again
```

There is no guess in the middle. A stale redaction layer is not treated as an
empty layer.

## What this is not

Jaenys is not an entity detector. It does not decide what is sensitive. It
guards the visibility contract around whatever your own detector, reviewer, or
policy step writes.

Jaenys is not a replacement for database permissions. If a caller bypasses the
guard and runs raw queries, the guard cannot protect that path. Put the guard at
one application chokepoint and compose it with normal store permissions.

Jaenys is not a re-sync scheduler. It does not hide drift by starting a
background job and hoping the next read lands after it. It refuses until the
caller re-runs derivation and writes a fresh mirror.

## The test

The checks are listed in [`docs/CONFORMANCE.md`](CONFORMANCE.md): flag edits
without rebuilds, orphan mirrors, missing span stores, corrupt ids, invalid
flags, membership drift, unreachable stores, and counts-only status output.

Run those scenarios against any redaction pipeline. If a pipeline cannot prove
freshness at serve time, it should say where it becomes best effort.

For Jaenys, that is the line: span-aware redaction, guarded by
prove-or-refuse checks, refusing on redaction drift instead of silently
under-filtering.
