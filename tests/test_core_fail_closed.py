"""Store-agnostic fail-closed drift matrix, run through the reference adapter.

The engine's defining behavior: when the live flag layer and the derived
span/mirror layer cannot be proven equal, serving is refused -- never
best-effort.
"""

from __future__ import annotations

import pytest

from jaenys import (
    Guard,
    RedactionDriftError,
    adapter_status,
    assert_adapter_current,
    load_guard_from_adapter,
    verify_guard_current,
)
from jaenys.adapters import InMemoryAdapter, StoreAdapter


def make_adapter(flags: dict[int, int], span_members: set[int]) -> InMemoryAdapter:
    adapter = InMemoryAdapter(flags)
    adapter.rebuild_derived_layer(span_members=span_members)
    return adapter


def test_clean_sync_serves() -> None:
    adapter = make_adapter({1: 0, 2: 1, 3: 0, 4: 0}, span_members={3, 4})
    assert_adapter_current(adapter)  # must not raise


def test_protocol_is_satisfied_structurally() -> None:
    assert isinstance(InMemoryAdapter(), StoreAdapter)


def test_flag_added_without_rebuild_refuses() -> None:
    adapter = make_adapter({1: 0, 2: 1}, span_members=set())
    adapter.set_flag(1, 1)  # live edit, no rebuild
    with pytest.raises(RedactionDriftError, match="stale"):
        assert_adapter_current(adapter)


def test_flag_cleared_without_rebuild_refuses() -> None:
    adapter = make_adapter({1: 0, 2: 1}, span_members=set())
    adapter.set_flag(2, 0)  # mirror now claims more than the live layer
    with pytest.raises(RedactionDriftError, match="stale"):
        assert_adapter_current(adapter)


def test_missing_span_layer_with_flags_refuses() -> None:
    adapter = InMemoryAdapter({1: 1})  # never derived
    with pytest.raises(RedactionDriftError, match="unavailable"):
        assert_adapter_current(adapter)


def test_missing_span_layer_without_flags_serves() -> None:
    adapter = InMemoryAdapter({1: 0, 2: 0})
    assert_adapter_current(adapter)  # nothing flagged -> nothing to leak


def test_dropped_span_layer_after_load_refuses() -> None:
    adapter = make_adapter({1: 1}, span_members={1})
    guard = load_guard_from_adapter(adapter)
    adapter.drop_span_layer()
    with pytest.raises(RedactionDriftError, match="unavailable"):
        verify_guard_current(adapter.flagged_ids(), guard)


def test_dropped_span_layer_after_load_refuses_without_flags() -> None:
    adapter = make_adapter({1: 0, 2: 0}, span_members={2})
    guard = load_guard_from_adapter(adapter)
    adapter.drop_span_layer()
    with pytest.raises(RedactionDriftError, match="became unavailable"):
        verify_guard_current(adapter.flagged_ids(), guard)


def test_unreachable_store_refuses() -> None:
    adapter = make_adapter({1: 1}, span_members=set())
    adapter.available = False
    with pytest.raises(RedactionDriftError, match="unreachable"):
        assert_adapter_current(adapter)


def test_guard_reflects_drift_introduced_after_load() -> None:
    adapter = make_adapter({1: 0, 2: 1}, span_members={2})
    guard = load_guard_from_adapter(adapter)
    verify_guard_current(adapter.flagged_ids(), guard)  # in sync at load
    adapter.set_flag(1, 1)
    with pytest.raises(RedactionDriftError, match="stale"):
        verify_guard_current(adapter.flagged_ids(), guard)
    adapter.rebuild_derived_layer()
    verify_guard_current(adapter.flagged_ids(), guard)  # rebuild restores service


def test_membership_change_after_load_refuses() -> None:
    """A span re-derivation can change membership without touching any flag.

    A long-lived materialized guard must refuse rather than keep serving its
    stale frozen member set -- and must never silently adopt the new set,
    because callers may already hold predicates built from the old ids.
    """

    adapter = make_adapter({1: 0, 2: 1, 3: 0}, span_members={3})
    guard = load_guard_from_adapter(adapter)
    verify_guard_current(adapter.flagged_ids(), guard)  # in sync at load
    adapter.set_span_members({1, 3})  # derivation grew the span; flags untouched
    with pytest.raises(RedactionDriftError, match="membership"):
        verify_guard_current(adapter.flagged_ids(), guard)
    reloaded = load_guard_from_adapter(adapter)  # reloading adopts the new set
    verify_guard_current(adapter.flagged_ids(), reloaded)


def test_membership_shrink_after_load_also_refuses() -> None:
    adapter = make_adapter({1: 0, 2: 1, 3: 0}, span_members={1, 3})
    guard = load_guard_from_adapter(adapter)
    adapter.set_span_members({3})  # rows left the span; old predicate over-hides
    with pytest.raises(RedactionDriftError, match="membership"):
        verify_guard_current(adapter.flagged_ids(), guard)


def test_hand_built_guard_without_freshness_source_fails_closed() -> None:
    guard = Guard(span_member_ids=frozenset({1}), span_layer_ready=True)
    with pytest.raises(RedactionDriftError, match="freshness"):
        verify_guard_current([5], guard)
    verify_guard_current([], guard)  # nothing flagged -> safe


def test_adapter_status_reports_refusal_and_counts() -> None:
    adapter = make_adapter({1: 0, 2: 1, 3: 0}, span_members={3})
    report = adapter_status(adapter)
    assert report["layers_in_sync"] is True
    assert report["counts"] == {"records": 3, "flagged": 1}
    assert report["unique_span_member_ids"] == 1

    adapter.set_flag(1, 1)
    report = adapter_status(adapter)
    assert report["layers_in_sync"] is False
    assert "stale" in report["refusal"]
