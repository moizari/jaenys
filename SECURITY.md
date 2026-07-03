# Security Policy

## What counts as a vulnerability here

This library's entire purpose is a guarantee: **records inside derived spans
are never served on normal surfaces, flagged records are never served
unmarked, and unprovable layer state refuses instead of serving.** Anything
that breaks that guarantee is a security vulnerability, including:

- any way to make a serve-path helper (`predicate`, `annotate_rows`,
  `filter_visible_ids`, an adapter read, the CLI) return `HIDDEN`-state rows,
  or return `BLUR`-state rows without their `"blurred"` marking;
- any state (drift, corruption, unreachable store, junk data, crafted
  mapping/identifier input) that is served best-effort instead of raising
  `RedactionDriftError`;
- SQL or query-operator injection through mapping names, aliases,
  namespaces, or dialect fields;
- record content leaking into refusal messages, status reports, or logs
  (status reports are counts-only; refusal messages may name a record id or a
  truncated value, never record content);
- `local_llm_guard.enforce_local_url` accepting a URL that can reach a
  non-local host.

Crashes, misleading errors on already-refused paths, and performance issues
are ordinary bugs; please open a regular issue for those.

## Reporting

Please **do not open a public issue for vulnerabilities.** Instead use
GitHub's private reporting: **Security > Report a vulnerability** on the
repository. You should receive an acknowledgement within a few days.

Include a minimal reproduction. The synthetic-fixture helpers in
`tests/conftest.py` are usually enough to demonstrate any leak.

## Supported versions

| Version | Supported |
|---|---|
| latest 0.1.x | yes |
| anything older | no, upgrade first |

## Design notes relevant to auditors

- The engine is read-only against both stores and has **zero runtime
  dependencies**; the attack surface is the SQL it renders and the values it
  reads back. Identifiers pass an allowlist (`validate_name`) before any
  quoting; all values are bound parameters, never interpolated.
- Verification and serving are separate statements. See the TOCTOU note in
  `src/jaenys/sql/guard.py` for the transactional pattern strict
  deployments should use.
- Stored ids/flags are coerced fail-closed; the flag domain is exactly
  `{0, 1}`, and refusal messages truncate offending values so a corrupt
  column carrying record text cannot leak through an error string.
