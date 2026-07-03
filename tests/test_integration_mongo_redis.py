"""MongoDB and Redis integration tests (optional; auto-skipped without servers).

Set the corresponding environment variable to enable each half::

    export JAENYS_MONGO_URI="mongodb://localhost:27017"
    export JAENYS_REDIS_URL="redis://localhost:6379/0"

Each test works in a uniquely named database / key prefix and cleans up.
The fail-closed matrix mirrors the fake-backed contract suite in
``test_adapters.py`` -- this file proves the same behavior against live
servers and real drivers.
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Iterator

import pytest

from jaenys.adapters import MongoDBAdapter, RedisAdapter
from jaenys.core import (
    HIDDEN,
    RedactionDriftError,
    adapter_status,
    assert_adapter_current,
    load_guard_from_adapter,
)

MONGO_URI = os.environ.get("JAENYS_MONGO_URI")
REDIS_URL = os.environ.get("JAENYS_REDIS_URL")

RECORDS = {1: 0, 2: 1, 3: 0, 4: 0, 5: 1, 6: 0, 7: 0}
SPAN_MEMBERS = frozenset({4, 5, 6})
FLAGGED = frozenset({2, 5})


# ---------------------------------------------------------------------------
# MongoDB
# ---------------------------------------------------------------------------


@pytest.fixture()
def mongo_db() -> Iterator[Any]:
    if not MONGO_URI:
        pytest.skip("JAENYS_MONGO_URI not set; skipping MongoDB integration tests")
    pymongo = pytest.importorskip("pymongo")
    client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db_name = f"jaenys_test_{uuid.uuid4().hex[:10]}"
    db = client[db_name]
    try:
        client.admin.command("ping")
    except Exception:
        pytest.skip(f"MongoDB at {MONGO_URI} is not reachable")
    try:
        yield db
    finally:
        client.drop_database(db_name)
        client.close()


def _mongo_seed(db: Any) -> None:
    db["records"].insert_many(
        [{"record_id": record_id, "sensitive": flag} for record_id, flag in RECORDS.items()]
    )
    _mongo_rebuild(db)


def _mongo_rebuild(db: Any) -> None:
    db["span_members"].drop()
    db["sensitive_records"].drop()
    db.create_collection("span_members")
    db.create_collection("sensitive_records")
    if SPAN_MEMBERS:
        db["span_members"].insert_many([{"record_id": record_id} for record_id in SPAN_MEMBERS])
    flagged = [doc["record_id"] for doc in db["records"].find({"sensitive": 1})]
    if flagged:
        db["sensitive_records"].insert_many(
            [
                {"record_id": record_id, "copy_reason": "flagged", "source_flag": 1}
                for record_id in flagged
            ]
        )


def test_mongo_fail_closed_matrix(mongo_db: Any) -> None:
    _mongo_seed(mongo_db)
    adapter = MongoDBAdapter(mongo_db)

    # Clean pair: ids agree, verification passes, three states classify.
    assert adapter.span_member_ids() == SPAN_MEMBERS
    assert adapter.flagged_ids() == FLAGGED
    assert_adapter_current(adapter)
    guard = load_guard_from_adapter(adapter)
    assert guard.classify(5, 1) == HIDDEN

    # Native pre-filter agrees with the guard on deliverable ids.
    deliverable = sorted(
        doc["record_id"]
        for doc in mongo_db["records"].find(
            adapter.visibility_filter(guard.span_member_ids, include_blur=True)
        )
    )
    assert deliverable == [1, 2, 3, 7]

    # Drift: flag edit without a rebuild refuses.
    mongo_db["records"].update_one({"record_id": 3}, {"$set": {"sensitive": 1}})
    with pytest.raises(RedactionDriftError):
        assert_adapter_current(adapter)
    assert adapter_status(adapter)["layers_in_sync"] is False

    # Rebuild restores service.
    _mongo_rebuild(mongo_db)
    assert_adapter_current(adapter)


# ---------------------------------------------------------------------------
# Redis
# ---------------------------------------------------------------------------


@pytest.fixture()
def redis_client() -> Iterator[tuple[Any, str]]:
    if not REDIS_URL:
        pytest.skip("JAENYS_REDIS_URL not set; skipping Redis integration tests")
    redis = pytest.importorskip("redis")
    client = redis.Redis.from_url(REDIS_URL, socket_connect_timeout=5)
    prefix = f"jaenys_test_{uuid.uuid4().hex[:10]}"
    try:
        client.ping()
    except Exception:
        pytest.skip(f"Redis at {REDIS_URL} is not reachable")
    try:
        yield client, prefix
    finally:
        keys = client.keys(f"{prefix}:*")
        if keys:
            client.delete(*keys)
        client.close()


def _redis_seed(client: Any, prefix: str) -> None:
    client.set(f"{prefix}:record_count", len(RECORDS))
    for record_id, flag in RECORDS.items():
        if flag == 1:
            client.sadd(f"{prefix}:flagged", record_id)
    _redis_rebuild(client, prefix)


def _redis_rebuild(client: Any, prefix: str) -> None:
    client.delete(f"{prefix}:span_members", f"{prefix}:mirror_flagged")
    client.sadd(f"{prefix}:span_members", *SPAN_MEMBERS)
    flagged = client.smembers(f"{prefix}:flagged")
    if flagged:
        client.sadd(f"{prefix}:mirror_flagged", *flagged)
    client.set(f"{prefix}:span_layer_ready", 1)
    client.set(f"{prefix}:mirror_ready", 1)


def test_redis_fail_closed_matrix(redis_client: tuple[Any, str]) -> None:
    client, prefix = redis_client
    _redis_seed(client, prefix)
    adapter = RedisAdapter(client, prefix=prefix)

    # Clean pair.
    assert adapter.span_member_ids() == SPAN_MEMBERS
    assert adapter.flagged_ids() == FLAGGED
    assert adapter.counts() == {"records": 7, "flagged": 2}
    assert_adapter_current(adapter)
    guard = load_guard_from_adapter(adapter)
    assert guard.classify(4, 0) == HIDDEN

    # Drift: live flag set without a rebuild refuses.
    client.sadd(f"{prefix}:flagged", 3)
    with pytest.raises(RedactionDriftError):
        assert_adapter_current(adapter)

    # Rebuild restores service.
    _redis_rebuild(client, prefix)
    assert_adapter_current(adapter)
