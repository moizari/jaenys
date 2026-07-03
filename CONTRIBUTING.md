# Contributing

Thanks for your interest! This project is small and deliberate; the rules
below exist to protect its two core promises: **fail-closed behavior** and
**zero runtime dependencies**.

## Dev setup

```bash
git clone https://github.com/moizari/jaenys
cd jaenys
pip install -e . pytest ruff
python -m pytest -q        # full suite: seconds, no services needed
python -m ruff check .
python -m ruff format --check .
```

The Postgres/MongoDB/Redis integration tests auto-skip unless you point them
at live servers:

```bash
export JAENYS_PG_DSN=postgresql://user:pw@localhost:5432/testdb
export JAENYS_MONGO_URI=mongodb://localhost:27017
export JAENYS_REDIS_URL=redis://localhost:6379/0
```

## Repository layout

```
src/jaenys/           the engine
  core.py                 store-agnostic core: states, coercers, layer verification
  cli.py, __main__.py     `jaenys` counts-only status CLI
  sql/                    SQL backend over PEP 249 connections
    dialects.py           per-engine Dialect value objects + auto-detection
    guard.py              predicates, guards, freshness checks, status()
    sqlite.py             path-based SQLite conveniences (reference engine)
  adapters/               non-SQL stores via the 4-method StoreAdapter protocol
    protocol.py           the protocol + shared fail-closed helpers
    memory.py             InMemoryAdapter: reference implementation and template
    mongodb.py, redis.py, dynamodb.py, couchbase.py, firestore.py
src/local_llm_guard/      local-LLM guards (separate package, same distribution)
  urls.py                 local-only endpoint URL enforcement
  reasoning.py            reasoning-block / JSON-fence stripping
tests/                    named by area: test_core_*, test_sql_*, test_adapters,
                          test_llm_guard_*, test_integration_* (env-gated)
examples/                 runnable walkthroughs: demo_sql_guard.py,
                          demo_store_adapter.py
```

## Ground rules

1. **Fail closed, always.** Any path where the engine cannot *prove* the two
   sensitivity layers agree must raise `RedactionDriftError`. Never guess, never
   serve best-effort, never silently adopt new state. If you find a way to
   make the engine serve a row it should have refused, that is a security
   bug: see [SECURITY.md](SECURITY.md).
2. **Zero runtime dependencies.** The engine only talks to storage through
   connections/clients the caller already owns. Adapters must never import a
   database driver, not even lazily inside a function.
3. **Corruption refuses.** Stored ids and flags route through the
   fail-closed coercers (`coerce_record_id` / `coerce_flag`); the flag
   domain is exactly `{0, 1}`. Refusal messages name at most the offending id
   or a truncated value, never full record content (`_safe_repr` clips long
   reprs for this reason); status surfaces stay counts-only.
4. **Domain-neutral language.** Code, tests, comments, and docs speak in
   generic terms (records, flags, spans, tickets); keep it that way in new
   contributions.
5. **Synthetic fixtures only.** Tests generate all data (see
   `tests/conftest.py`); never commit a real dataset, even a "harmless" one.
6. **Style**: ruff (line length 100), `from __future__ import annotations`,
   frozen dataclasses, explanatory docstrings that say *why*. Run
   `ruff format` before committing.

## Adding support for a new store

- **SQL engine**: usually just a new `Dialect` instance (~10 lines) in
  `src/jaenys/sql/dialects.py` plus golden-rendering tests in
  `tests/test_sql_dialects.py`. No driver import.
- **Non-SQL store**: implement the 4-method `StoreAdapter` protocol
  (`src/jaenys/adapters/protocol.py`), take client objects in the
  constructor, wrap all I/O in `_fail_closed(name)`, document the exact
  storage shape and readiness marker in the module docstring, and add your
  harness + fakes to the contract suite in `tests/test_adapters.py` so every
  shared behavior test runs against it.

## Pull requests

Keep them small and single-purpose. A PR should include: the change, tests
that fail without it, and a `CHANGELOG.md` entry under `[Unreleased]`. CI
must be green (pytest + ruff across Python 3.10-3.13).
