# The Redaction Drift Test Suite

These checks describe how a redaction pipeline should behave when its serving
state may be stale, missing, or corrupt. They can be applied to Jaenys, a
custom retrieval stack, or another serving-layer filter.

The README defines **redaction drift** here:
https://github.com/moizari/jaenys#what-is-redaction-drift.

Jaenys 0.1.0 ships executable coverage for these behaviors under `tests/`.
The references below point to the nearest test files, not to a separate
certification program.

## Required outcomes

`Serve` means the guarded path may return normal output. `Refuse` means the
guarded path must stop normal output and raise, return unhealthy status, or
otherwise fail closed. It must not silently treat missing or stale state as
clear.

Refusal output must be counts-only or metadata-only. It must not embed record
content.

## Core freshness scenarios

| # | Scenario | Setup | Question | Required behavior |
|---|---|---|---|---|
| 1 | Clean layers | Live flags, mirror, and span members match. | Can the guarded path serve? | Serve. |
| 2 | Flag added after derivation | A record is newly flagged after the span layer was derived. | Can the stale span layer be trusted? | Refuse. |
| 3 | Flag cleared after derivation | A mirrored flag no longer exists in the live layer. | Can the old derivation be reused? | Refuse. |
| 4 | Orphan mirror row | The mirror names a flagged record that is gone or no longer valid in the primary store. | Can the mirror prove freshness? | Refuse. |
| 5 | Missing mirror with flags | The primary store has flagged records, but the span store has no usable mirror. | Can serving fall back to the span table alone? | Refuse. |
| 6 | Drift after guard load | A materialized guard was loaded while layers matched, then the primary flags changed. | Can the old guard keep serving? | Refuse on re-verification. |
| 7 | Membership grows after guard load | Span membership changes while flags stay the same. | Can unchanged flags imply unchanged spans? | Refuse. |
| 8 | Membership shrinks after guard load | A later derivation removes a span member while flags stay the same. | Can the old member set be reused? | Refuse. |
| 9 | Rebuild after drift | The caller re-runs derivation and rewrites flags, span members, and mirror. | Can service resume? | Serve. |

Reference coverage: `tests/test_core_fail_closed.py`,
`tests/test_sql_fail_closed.py`, `tests/test_adapters.py`.

## Span-store availability scenarios

| # | Scenario | Setup | Question | Required behavior |
|---|---|---|---|---|
| 10 | Missing span store with flags | The primary store has flagged records and the span layer is missing or a zero-byte placeholder. | Can an empty layer be assumed? | Refuse. |
| 11 | Missing span store without flags | No flagged records exist and no drift witness says spans were previously derived. | Can this be treated as a fresh install? | Serve with no spans. |
| 12 | Ready store without required tables | A span store file exists but lacks the mirror or member shape needed for proof. | Can serving continue from partial state? | Refuse when flags exist. |
| 13 | Unreadable span store | The span store cannot be opened or read. | Can the guard guess the member set? | Refuse. |
| 14 | Corrupt span store | The span store exists but has invalid database contents or invalid rows. | Can the corrupt layer be treated as empty? | Refuse. |
| 15 | Closed primary connection | The guard cannot read the live flag layer because the connection is closed. | Can the last known state be trusted? | Refuse. |
| 16 | Unreachable adapter store | An adapter backend raises during verification. | Can the adapter return a guessed set? | Refuse. |

Reference coverage: `tests/test_sql_fail_closed.py`,
`tests/test_sql_edge_cases.py`, `tests/test_core_fail_closed.py`,
`tests/test_adapters.py`.

## Drift-witness scenarios

| # | Scenario | Setup | Question | Required behavior |
|---|---|---|---|---|
| 17 | Witness matches span store | The primary witness count and digest match the span members. | Can the SQL guard trust the span layer? | Continue to mirror checks. |
| 18 | Missing store with witness | The primary store records a drift witness but the span store is missing or empty. | Can zero current flags make the loss safe? | Refuse. |
| 19 | Same count, different members | The witness count matches but the member digest differs. | Can count equality prove freshness? | Refuse. |
| 20 | Different count | The witness count differs from the span-member count. | Can the layer be served as partial? | Refuse. |
| 21 | Bare-id write clears old witness | A legacy flag writer updates flags without span membership. | Can an old witness be kept? | No. Clear it and fall back to mirror-only behavior. |

Reference coverage: `tests/test_sql_fail_closed.py`,
`tests/test_derivation.py`.

## Visibility scenarios

| # | Scenario | Setup | Question | Required behavior |
|---|---|---|---|---|
| 22 | Visible record | A record is not flagged and not in any span. | How is it classified? | `VISIBLE`, served normally. |
| 23 | Standalone flagged record | A record is flagged but outside every span. | How is it classified? | `BLUR`, kept only in outputs that include blur rows. |
| 24 | Neutral row inside span | A record is unflagged but is inside a derived span. | Can neutral content leak span context? | `HIDDEN`, not served. |
| 25 | Flagged row inside span | A record is flagged and inside a span. | Does the flag beat the span? | No. `HIDDEN`, not served. |
| 26 | Strict visible output | A caller asks for visible rows only. | Are blur rows included? | No. Only `VISIBLE` rows are returned. |
| 27 | Blur-inclusive output | A caller explicitly includes blur rows. | Are span rows included too? | No. `VISIBLE` and standalone `BLUR` may be returned, `HIDDEN` rows are still excluded. |

Reference coverage: `tests/test_core_states.py`,
`tests/test_sql_predicates.py`, `tests/test_adapters.py`.

## Corruption and input-shape scenarios

| # | Scenario | Setup | Question | Required behavior |
|---|---|---|---|---|
| 28 | Out-of-domain flag | A flag value is not exactly `0` or `1`. | Can it be treated as not flagged? | Refuse. |
| 29 | Missing flag | A row that must be classified lacks its flag value. | Can classification continue? | Refuse for that guarded operation. |
| 30 | Junk record id | A row id cannot be safely coerced to an integer id. | Can it be truncated or parsed loosely? | Refuse. |
| 31 | Non-integral id | A float or decimal id has a fractional part. | Can it be rounded? | Refuse. |
| 32 | Junk span-member id | The span layer contains a member id that cannot be safely coerced. | Can it be ignored? | Refuse. |
| 33 | Duplicate primary id | The primary store has duplicate record ids where uniqueness is required. | Can the guard choose one? | Refuse. |
| 34 | Unsafe schema name | A table, column, alias, or mapping value contains unsafe identifier text. | Can it be quoted into SQL? | Refuse. |

Reference coverage: `tests/test_core_coercion.py`,
`tests/test_core_states.py`, `tests/test_sql_edge_cases.py`,
`tests/test_sql_dialects.py`, `tests/test_adapters.py`.

## SQL and status scenarios

| # | Scenario | Setup | Question | Required behavior |
|---|---|---|---|---|
| 35 | Predicate protects search | A caller uses the attached SQL predicate as the data-access chokepoint. | Are span rows excluded in the database? | Yes. The predicate excludes them server-side. |
| 36 | Parameter budget | A rendered predicate would exceed the dialect's statement parameter budget. | Can the query be emitted anyway? | Refuse before execution. |
| 37 | Bounded id filter | A caller asks for a specific list of ids. | Are order and duplicates preserved while still verifying freshness? | Yes, unless drift or corrupt rows are found. |
| 38 | Status is counts-only | A status or CLI check inspects unhealthy layers. | Can it print record content? | No. It reports counts and health metadata only. |
| 39 | Status is read-only | A status check runs against stores. | Can it repair or mutate state? | No. It only observes. |
| 40 | Unhealthy CLI exit | The command-line status check detects drift or corrupt state. | Can it exit success? | No. It exits nonzero. |

Reference coverage: `tests/test_sql_predicates.py`,
`tests/test_sql_edge_cases.py`, `tests/test_sql_status.py`.

## Adapter scenarios

| # | Scenario | Setup | Question | Required behavior |
|---|---|---|---|---|
| 41 | Never derived with flags | An adapter reports flagged records but no derived span layer. | Can the guard serve anyway? | Refuse. |
| 42 | Never derived without flags | An adapter reports no flags and no derived span layer. | Can this fresh state serve? | Serve with no spans. |
| 43 | Legacy mirrorless adapter store | A legacy adapter shape has no mirror. | Can it serve when the contract allows that legacy shape? | Serve only under the documented legacy behavior. |
| 44 | Adapter counts are corrupt | An adapter reports impossible counts, such as negative records or fewer records than flags. | Can status trust the counts? | Refuse or report unhealthy. |
| 45 | Driver-free imports | Importing adapter classes happens without installed database drivers. | Can naming one adapter import another backend driver? | No. Imports must remain dependency-free. |

Reference coverage: `tests/test_adapters.py`,
`tests/test_integration_mongo_redis.py`, `tests/test_integration_postgres.py`.

## Versioned span scenarios

| # | Scenario | Setup | Question | Required behavior |
|---|---|---|---|---|
| 46 | Valid versioned span request | The span store records a requested version id. | Can the predicate scope exclusion to that version? | Serve after freshness checks. |
| 47 | Phantom version id | A caller requests a version id the span store does not record. | Can it silently drop every span? | Refuse. |
| 48 | Version requested against unversioned store | A caller requests a version from a store with no versioned layer. | Can the request be treated as no spans? | Refuse. |

Reference coverage: `tests/test_sql_predicates.py`,
`tests/test_sql_dialects.py`.

## Passing

A passing implementation does not need Jaenys internals. It needs the same
observable behavior:

- It names the drift state instead of serving around it.
- It proves freshness between detection state and serving state at serve time.
- It treats missing, unreadable, corrupt, or stale state as unsafe.
- It supports span-aware redaction or states that it only handles per-record
  filtering.
- It keeps refusal output free of record content.
