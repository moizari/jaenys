#!/usr/bin/env python3
"""Runnable, stdlib-only walkthrough of the non-SQL adapter path.

Same story as demo_sql_guard.py, but through the small 4-method
StoreAdapter protocol instead of SQL: no database drivers, no SQL at all.
This is the path for document/KV stores (MongoDB, Redis, DynamoDB,
Couchbase, Firestore, ...) that implement the same protocol.

Run with: python examples/demo_store_adapter.py
"""

from __future__ import annotations

from jaenys import RedactionDriftError, annotate_rows
from jaenys.adapters import InMemoryAdapter
from jaenys.core import adapter_status, assert_adapter_current, load_guard_from_adapter


def main() -> None:
    print("=== Step 1: build the synthetic support-ticket records ===")
    # record_id -> live sensitivity flag, same 8-row transcript as the SQL demo.
    records = {
        1: 0,  # "Hi, how can I help today?"
        2: 1,  # "My account PIN is 4471."          <- standalone flagged
        3: 0,  # "Thanks, one moment."
        4: 0,  # "Sure, no rush."
        5: 0,  # "Can you confirm your date of birth?"  <- span start
        6: 0,  # "March 3rd, 1990."
        7: 0,  # "Got it, verified."                     <- span end
        8: 0,  # "Great, thank you!"
    }
    bodies = {
        1: "Hi, how can I help today?",
        2: "My account PIN is 4471.",
        3: "Thanks, one moment.",
        4: "Sure, no rush.",
        5: "Can you confirm your date of birth?",
        6: "March 3rd, 1990.",
        7: "Got it, verified.",
        8: "Great, thank you!",
    }
    span_ids = {5, 6, 7}

    adapter = InMemoryAdapter(records)
    adapter.rebuild_derived_layer(span_members=span_ids)
    print(f"Adapter store: {len(records)} records. Span layer: derived, 3 rows.\n")

    print("=== Step 2: serve the three-state result ===")
    guard = load_guard_from_adapter(adapter)
    rows = [
        {"record_id": record_id, "sensitive": flag, "body": bodies[record_id]}
        for record_id, flag in records.items()
    ]
    annotated = annotate_rows(
        rows, span_member_ids=guard.span_member_ids, id_key="record_id", flag_key="sensitive"
    )
    served_ids = {row["record_id"] for row in annotated}
    for row in annotated:
        state = "BLUR" if row["blurred"] else "VISIBLE"
        print(f"  [{state:7}] id={row['record_id']:>2}  {row['body']!r}")
    for hidden_id in sorted(set(records) - served_ids):
        print(f"  [HIDDEN ] id={hidden_id:>2}  (in-span; not served)")
    print()

    print("=== Step 3: flip a flag WITHOUT re-deriving the span layer ===")
    adapter.set_flag(1, 1)
    print("  Flagged record 1 on the live layer. Derived layer untouched.\n")

    print("=== Step 4: the engine refuses to serve stale isolation ===")
    try:
        assert_adapter_current(adapter)
        print("  (should not reach here)")
    except RedactionDriftError as exc:
        print(f"  RedactionDriftError raised, as expected:\n  -> {exc}\n")

    print("=== Step 5: re-derive the layer and confirm service is restored ===")
    adapter.rebuild_derived_layer(span_members=span_ids)
    assert_adapter_current(adapter)  # must not raise
    report = adapter_status(adapter)
    print(f"  Re-derived. layers_in_sync={report['layers_in_sync']}, counts={report['counts']}")


if __name__ == "__main__":
    main()
