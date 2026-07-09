"""Adapter contract suite.

One parametrized suite asserts that every adapter satisfies the StoreAdapter
semantics (ID sets, counts, drift detection through the core), run against
the InMemoryAdapter reference and hand-rolled client fakes for pymongo,
redis-py, boto3 DynamoDB, Couchbase, and Firestore.  No driver is installed
or imported.

Every fake models the store behavior the adapter docstrings rely on --
e.g. DynamoDB pagination via ``LastEvaluatedKey``, Decimal-typed numbers,
Redis bytes responses, Mongo's inability to distinguish a missing collection
from an empty one.
"""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from jaenys.adapters import (
    CouchbaseAdapter,
    DynamoDBAdapter,
    FirestoreAdapter,
    InMemoryAdapter,
    MongoDBAdapter,
    RedisAdapter,
    StoreAdapter,
)
from jaenys.adapters.firestore import _reject_unsafe_firestore_name
from jaenys.adapters.mongodb import _reject_unsafe_mongo_name
from jaenys.core import (
    BLUR,
    HIDDEN,
    VISIBLE,
    SchemaMapping,
    RedactionDriftError,
    adapter_status,
    assert_adapter_current,
    load_guard_from_adapter,
    verify_guard_current,
)

# Shared scenario: 7 records; 2 is standalone flagged, 4-6 form a span with
# 5 flagged inside it.
RECORDS = {1: 0, 2: 1, 3: 0, 4: 0, 5: 1, 6: 0, 7: 0}
SPAN_MEMBERS = frozenset({4, 5, 6})
FLAGGED = frozenset({2, 5})


# ---------------------------------------------------------------------------
# Client fakes
# ---------------------------------------------------------------------------


class FakeMongoCollection:
    def __init__(self, database: "FakeMongoDatabase", name: str) -> None:
        self._database = database
        self._name = name

    def _docs(self) -> list[dict]:
        return self._database._collections.get(self._name, [])

    @staticmethod
    def _matches(actual: Any, condition: Any) -> bool:
        # Enough of the query language for the adapter: plain equality plus the
        # ``{"$in": [...]}`` operator that counts() emits for the flag.
        if isinstance(condition, dict) and "$in" in condition:
            return any(actual == option for option in condition["$in"])
        return actual == condition

    def find(self, query: dict | None = None, projection: Any = None) -> list[dict]:
        self._database._check()
        query = query or {}
        return [
            doc
            for doc in self._docs()
            if all(self._matches(doc.get(key), value) for key, value in query.items())
        ]

    def distinct(self, key: str, query: dict | None = None) -> list:
        # Real pymongo semantics: docs missing the field contribute nothing;
        # a doc with an explicit None value contributes None (which the
        # adapter's coercer then refuses).
        self._database._check()
        values: list = []
        for doc in self.find(query or {}):
            if key in doc and doc[key] not in values:
                values.append(doc[key])
        return values

    def count_documents(self, query: dict) -> int:
        return len(self.find(query))


class FakeMongoDatabase:
    """Missing and empty collections both ``find()`` to nothing, as in Mongo."""

    def __init__(self) -> None:
        self._collections: dict[str, list[dict]] = {}
        self.available = True

    def _check(self) -> None:
        if not self.available:
            raise ConnectionError("connection refused")

    def __getitem__(self, name: str) -> FakeMongoCollection:
        return FakeMongoCollection(self, name)

    def list_collection_names(self) -> list[str]:
        self._check()
        return list(self._collections)

    def create_collection_with(self, name: str, docs: list[dict]) -> None:
        self._collections[name] = list(docs)

    def drop_collection(self, name: str) -> None:
        self._collections.pop(name, None)


class FakeRedisClient:
    """Returns bytes, as a redis-py client without decode_responses does."""

    def __init__(self) -> None:
        self._sets: dict[str, set[bytes]] = {}
        self._strings: dict[str, bytes] = {}
        self.available = True

    def _check(self) -> None:
        if not self.available:
            raise ConnectionError("connection refused")

    def smembers(self, key: str) -> set[bytes]:
        self._check()
        return set(self._sets.get(key, set()))

    def exists(self, key: str) -> int:
        self._check()
        return int(key in self._sets or key in self._strings)

    def get(self, key: str) -> bytes | None:
        self._check()
        return self._strings.get(key)

    def sadd(self, key: str, *values: Any) -> None:
        self._sets.setdefault(key, set()).update(str(value).encode() for value in values)

    def srem(self, key: str, *values: Any) -> None:
        self._sets.get(key, set()).difference_update(str(value).encode() for value in values)

    def set(self, key: str, value: Any) -> None:
        self._strings[key] = str(value).encode()

    def delete(self, *keys: str) -> None:
        for key in keys:
            self._sets.pop(key, None)
            self._strings.pop(key, None)


class FakeDynamoTable:
    """Paginates every scan (page size 2) and stores numbers as Decimal."""

    def __init__(self, page_size: int = 2) -> None:
        self.items: list[dict] = []
        self.page_size = page_size
        self.available = True

    def _check(self) -> None:
        if not self.available:
            raise ConnectionError("connection refused")

    @staticmethod
    def _matches(item: dict, kwargs: dict) -> bool:
        expression = kwargs.get("FilterExpression")
        if expression is None:
            return True
        name_token, _, value_token = expression.partition(" = ")
        attr = kwargs["ExpressionAttributeNames"][name_token.strip()]
        value = kwargs["ExpressionAttributeValues"][value_token.strip()]
        return item.get(attr) == value

    def scan(self, **kwargs: Any) -> dict:
        self._check()
        matched = [item for item in self.items if self._matches(item, kwargs)]
        start = kwargs.get("ExclusiveStartKey", 0)
        page = matched[start : start + self.page_size]
        response: dict[str, Any] = {}
        if kwargs.get("Select") == "COUNT":
            response["Count"] = len(page)
        else:
            response["Items"] = page
        if start + self.page_size < len(matched):
            response["LastEvaluatedKey"] = start + self.page_size
        return response

    def get_item(self, Key: dict) -> dict:
        self._check()
        for item in self.items:
            if all(item.get(key) == value for key, value in Key.items()):
                return {"Item": item}
        return {}


class FakeCouchbaseCollection:
    def __init__(self) -> None:
        self.docs: dict[str, dict] = {}
        self.available = True

    def _check(self) -> None:
        if not self.available:
            raise ConnectionError("connection refused")

    def exists(self, doc_id: str) -> Any:
        self._check()
        return SimpleNamespace(exists=doc_id in self.docs)


def fake_couchbase_scan(collection: FakeCouchbaseCollection) -> list[dict]:
    collection._check()
    return [dict(body, id=doc_id) for doc_id, body in collection.docs.items()]


class FakeFirestoreSnapshot:
    def __init__(self, doc_id: str, data: dict | None) -> None:
        self.id = doc_id
        self._data = data

    @property
    def exists(self) -> bool:
        return self._data is not None

    def to_dict(self) -> dict | None:
        return self._data


class FakeFirestoreDocRef:
    def __init__(self, collection: "FakeFirestoreCollection", doc_id: str) -> None:
        self._collection = collection
        self._doc_id = doc_id

    def get(self) -> FakeFirestoreSnapshot:
        self._collection._check()
        return FakeFirestoreSnapshot(self._doc_id, self._collection.docs().get(self._doc_id))


class FakeFirestoreCollection:
    def __init__(self, client: "FakeFirestoreClient", name: str) -> None:
        self._client = client
        self._name = name

    def _check(self) -> None:
        self._client._check()

    def docs(self) -> dict[str, dict]:
        return self._client._collections.setdefault(self._name, {})

    def stream(self) -> list[FakeFirestoreSnapshot]:
        self._check()
        return [FakeFirestoreSnapshot(doc_id, data) for doc_id, data in self.docs().items()]

    def document(self, doc_id: str) -> FakeFirestoreDocRef:
        return FakeFirestoreDocRef(self, doc_id)


class FakeFirestoreClient:
    def __init__(self) -> None:
        self._collections: dict[str, dict[str, dict]] = {}
        self.available = True

    def _check(self) -> None:
        if not self.available:
            raise ConnectionError("connection refused")

    def collection(self, name: str) -> FakeFirestoreCollection:
        return FakeFirestoreCollection(self, name)


# ---------------------------------------------------------------------------
# Harnesses: one per adapter, exposing the same store-lifecycle operations
# ---------------------------------------------------------------------------


class InMemoryHarness:
    def __init__(self, records: dict[int, int], *, derived: bool = True) -> None:
        self.adapter = InMemoryAdapter(records)
        if derived:
            self.adapter.rebuild_derived_layer(span_members=SPAN_MEMBERS & set(records))

    def set_flag(self, record_id: int, flag: int) -> None:
        self.adapter.set_flag(record_id, flag)

    def rebuild(self) -> None:
        self.adapter.rebuild_derived_layer()

    def drop_mirror(self) -> None:
        # The reference adapter has no external client to reach around, so the
        # legacy mirror-less shape is induced directly on its derived state.
        self.adapter._mirror_flagged = None

    def make_unavailable(self) -> None:
        self.adapter.available = False


class MongoHarness:
    def __init__(self, records: dict[int, int], *, derived: bool = True) -> None:
        self.primary = FakeMongoDatabase()
        self.span = FakeMongoDatabase()
        self.mapping = SchemaMapping()
        self.primary.create_collection_with(
            self.mapping.record_table,
            [
                {self.mapping.record_id_column: record_id, self.mapping.flag_column: flag}
                for record_id, flag in records.items()
            ],
        )
        self.adapter = MongoDBAdapter(self.primary, self.span, mapping=self.mapping)
        if derived:
            self.rebuild()

    def set_flag(self, record_id: int, flag: int) -> None:
        for doc in self.primary._collections[self.mapping.record_table]:
            if doc[self.mapping.record_id_column] == record_id:
                doc[self.mapping.flag_column] = flag

    def rebuild(self) -> None:
        flagged = [
            doc[self.mapping.record_id_column]
            for doc in self.primary._collections[self.mapping.record_table]
            if doc[self.mapping.flag_column] == 1
        ]
        members = SPAN_MEMBERS & {
            doc[self.mapping.record_id_column]
            for doc in self.primary._collections[self.mapping.record_table]
        }
        self.span.create_collection_with(
            self.mapping.span_member_table,
            [{self.mapping.span_member_id_column: record_id} for record_id in members],
        )
        self.span.create_collection_with(
            self.mapping.mirror_table,
            [
                {
                    self.mapping.mirror_id_column: record_id,
                    self.mapping.mirror_reason_column: self.mapping.reason_flagged,
                    self.mapping.mirror_flag_column: 1,
                }
                for record_id in flagged
            ],
        )

    def drop_mirror(self) -> None:
        self.span.drop_collection(self.mapping.mirror_table)

    def make_unavailable(self) -> None:
        self.primary.available = False
        self.span.available = False


class RedisHarness:
    def __init__(self, records: dict[int, int], *, derived: bool = True) -> None:
        self.client = FakeRedisClient()
        self.records = dict(records)
        self.adapter = RedisAdapter(self.client, prefix="guard_test")
        self.client.set("guard_test:record_count", len(records))
        for record_id, flag in records.items():
            if flag == 1:
                self.client.sadd("guard_test:flagged", record_id)
        if derived:
            self.rebuild()

    def set_flag(self, record_id: int, flag: int) -> None:
        self.records[record_id] = flag
        if flag == 1:
            self.client.sadd("guard_test:flagged", record_id)
        else:
            self.client.srem("guard_test:flagged", record_id)

    def rebuild(self) -> None:
        members = SPAN_MEMBERS & set(self.records)
        self.client.delete("guard_test:span_members", "guard_test:mirror_flagged")
        if members:
            self.client.sadd("guard_test:span_members", *members)
        flagged = [record_id for record_id, flag in self.records.items() if flag == 1]
        if flagged:
            self.client.sadd("guard_test:mirror_flagged", *flagged)
        self.client.set("guard_test:span_layer_ready", 1)
        self.client.set("guard_test:mirror_ready", 1)

    def drop_mirror(self) -> None:
        self.client.delete("guard_test:mirror_ready", "guard_test:mirror_flagged")

    def make_unavailable(self) -> None:
        self.client.available = False


class DynamoHarness:
    def __init__(self, records: dict[int, int], *, derived: bool = True) -> None:
        self.records_table = FakeDynamoTable()
        self.span_table = FakeDynamoTable()
        self.mapping = SchemaMapping()
        self.records_table.items = [
            {
                self.mapping.record_id_column: Decimal(record_id),
                self.mapping.flag_column: Decimal(flag),
            }
            for record_id, flag in records.items()
        ]
        self.adapter = DynamoDBAdapter(self.records_table, self.span_table, mapping=self.mapping)
        if derived:
            self.rebuild()

    def set_flag(self, record_id: int, flag: int) -> None:
        for item in self.records_table.items:
            if item[self.mapping.record_id_column] == Decimal(record_id):
                item[self.mapping.flag_column] = Decimal(flag)

    def rebuild(self) -> None:
        flagged = [
            int(item[self.mapping.record_id_column])
            for item in self.records_table.items
            if item[self.mapping.flag_column] == Decimal(1)
        ]
        members = SPAN_MEMBERS & {
            int(item[self.mapping.record_id_column]) for item in self.records_table.items
        }
        self.span_table.items = (
            [{"layer": "meta", "mirror_ready": True}]
            + [
                {"layer": "span_member", self.mapping.span_member_id_column: Decimal(record_id)}
                for record_id in members
            ]
            + [
                {"layer": "mirror_flagged", self.mapping.mirror_id_column: Decimal(record_id)}
                for record_id in flagged
            ]
        )

    def drop_mirror(self) -> None:
        self.span_table.items = [
            {"layer": "meta"} if item.get("layer") == "meta" else item
            for item in self.span_table.items
            if item.get("layer") != "mirror_flagged"
        ]

    def make_unavailable(self) -> None:
        self.records_table.available = False
        self.span_table.available = False


class CouchbaseHarness:
    def __init__(self, records: dict[int, int], *, derived: bool = True) -> None:
        self.records_collection = FakeCouchbaseCollection()
        self.span_collection = FakeCouchbaseCollection()
        self.mirror_collection: FakeCouchbaseCollection | None = FakeCouchbaseCollection()
        self.mapping = SchemaMapping()
        for record_id, flag in records.items():
            self.records_collection.docs[f"r{record_id}"] = {
                self.mapping.record_id_column: record_id,
                self.mapping.flag_column: flag,
            }
        self._make_adapter()
        if derived:
            self.rebuild()

    def _make_adapter(self) -> None:
        self.adapter = CouchbaseAdapter(
            self.records_collection,
            self.span_collection,
            self.mirror_collection,
            scan=fake_couchbase_scan,
            mapping=self.mapping,
        )

    def set_flag(self, record_id: int, flag: int) -> None:
        self.records_collection.docs[f"r{record_id}"][self.mapping.flag_column] = flag

    def rebuild(self) -> None:
        flagged = [
            doc[self.mapping.record_id_column]
            for doc in self.records_collection.docs.values()
            if doc[self.mapping.flag_column] == 1
        ]
        members = SPAN_MEMBERS & {
            doc[self.mapping.record_id_column] for doc in self.records_collection.docs.values()
        }
        self.span_collection.docs = {"_span_meta": {"derived": True}}
        for record_id in members:
            self.span_collection.docs[f"m{record_id}"] = {
                self.mapping.span_member_id_column: record_id
            }
        if self.mirror_collection is not None:
            self.mirror_collection.docs = {
                f"f{record_id}": {
                    self.mapping.mirror_id_column: record_id,
                    self.mapping.mirror_reason_column: self.mapping.reason_flagged,
                    self.mapping.mirror_flag_column: 1,
                }
                for record_id in flagged
            }

    def drop_mirror(self) -> None:
        self.mirror_collection = None
        self._make_adapter()

    def make_unavailable(self) -> None:
        self.records_collection.available = False
        self.span_collection.available = False
        if self.mirror_collection is not None:
            self.mirror_collection.available = False


class FirestoreHarness:
    def __init__(self, records: dict[int, int], *, derived: bool = True) -> None:
        self.client = FakeFirestoreClient()
        self.mapping = SchemaMapping()
        records_docs = self.client.collection(self.mapping.record_table).docs()
        for record_id, flag in records.items():
            records_docs[f"r{record_id}"] = {
                self.mapping.record_id_column: record_id,
                self.mapping.flag_column: flag,
            }
        self.adapter = FirestoreAdapter(self.client, mapping=self.mapping)
        if derived:
            self.rebuild()

    def set_flag(self, record_id: int, flag: int) -> None:
        docs = self.client.collection(self.mapping.record_table).docs()
        docs[f"r{record_id}"][self.mapping.flag_column] = flag

    def rebuild(self) -> None:
        records_docs = self.client.collection(self.mapping.record_table).docs()
        flagged = [
            doc[self.mapping.record_id_column]
            for doc in records_docs.values()
            if doc[self.mapping.flag_column] == 1
        ]
        members = SPAN_MEMBERS & {
            doc[self.mapping.record_id_column] for doc in records_docs.values()
        }
        span_docs = self.client.collection(self.mapping.span_member_table).docs()
        span_docs.clear()
        span_docs["_span_meta"] = {"derived": True}
        for record_id in members:
            span_docs[f"m{record_id}"] = {self.mapping.span_member_id_column: record_id}
        mirror_docs = self.client.collection(self.mapping.mirror_table).docs()
        mirror_docs.clear()
        mirror_docs["_span_meta"] = {"derived": True}
        for record_id in flagged:
            mirror_docs[f"f{record_id}"] = {
                self.mapping.mirror_id_column: record_id,
                self.mapping.mirror_reason_column: self.mapping.reason_flagged,
                self.mapping.mirror_flag_column: 1,
            }

    def drop_mirror(self) -> None:
        self.client._collections.pop(self.mapping.mirror_table, None)

    def make_unavailable(self) -> None:
        self.client.available = False


HARNESSES: dict[str, Callable[..., Any]] = {
    "in-memory": InMemoryHarness,
    "mongodb": MongoHarness,
    "redis": RedisHarness,
    "dynamodb": DynamoHarness,
    "couchbase": CouchbaseHarness,
    "firestore": FirestoreHarness,
}


@pytest.fixture(params=sorted(HARNESSES))
def make_harness(request: pytest.FixtureRequest) -> Callable[..., Any]:
    return HARNESSES[request.param]


# ---------------------------------------------------------------------------
# Contract suite (every adapter must pass every test)
# ---------------------------------------------------------------------------


def test_satisfies_protocol(make_harness: Callable[..., Any]) -> None:
    assert isinstance(make_harness(RECORDS).adapter, StoreAdapter)


def test_reports_expected_id_sets_and_counts(make_harness: Callable[..., Any]) -> None:
    adapter = make_harness(RECORDS).adapter
    assert adapter.span_member_ids() == SPAN_MEMBERS
    assert adapter.flagged_ids() == FLAGGED
    assert adapter.mirror_flagged_ids() == FLAGGED
    assert adapter.counts() == {"records": 7, "flagged": 2}


def test_clean_pair_passes_verification(make_harness: Callable[..., Any]) -> None:
    adapter = make_harness(RECORDS).adapter
    assert_adapter_current(adapter)  # must not raise
    report = adapter_status(adapter)
    assert report["layers_in_sync"] is True
    assert report["span_layer_ready"] is True
    assert report["unique_span_member_ids"] == 3


def test_guard_classifies_three_states(make_harness: Callable[..., Any]) -> None:
    guard = load_guard_from_adapter(make_harness(RECORDS).adapter)
    assert guard.classify(1, 0) == VISIBLE
    assert guard.classify(2, 1) == BLUR
    assert guard.classify(4, 0) == HIDDEN
    assert guard.classify(5, 1) == HIDDEN  # in-span wins over the flag

    rows = [{"record_id": record_id, "sensitive": flag} for record_id, flag in RECORDS.items()]
    annotated = guard.annotate_rows(rows)
    assert [row["record_id"] for row in annotated] == [1, 2, 3, 7]
    assert [row["blurred"] for row in annotated] == [False, True, False, False]
    assert [row["record_id"] for row in guard.filter_rows(rows)] == [1, 3, 7]


def test_flag_edit_without_rebuild_refuses(make_harness: Callable[..., Any]) -> None:
    harness = make_harness(RECORDS)
    guard = load_guard_from_adapter(harness.adapter)
    harness.set_flag(3, 1)
    with pytest.raises(RedactionDriftError):
        assert_adapter_current(harness.adapter)
    with pytest.raises(RedactionDriftError):
        verify_guard_current(harness.adapter.flagged_ids(), guard)


def test_unflag_without_rebuild_refuses(make_harness: Callable[..., Any]) -> None:
    harness = make_harness(RECORDS)
    harness.set_flag(2, 0)  # mirror still claims record 2 was flagged
    with pytest.raises(RedactionDriftError):
        assert_adapter_current(harness.adapter)


def test_rebuild_restores_service(make_harness: Callable[..., Any]) -> None:
    harness = make_harness(RECORDS)
    harness.set_flag(3, 1)
    harness.rebuild()
    assert_adapter_current(harness.adapter)  # must not raise
    assert harness.adapter.mirror_flagged_ids() == FLAGGED | {3}


def test_never_derived_with_flags_refuses(make_harness: Callable[..., Any]) -> None:
    harness = make_harness(RECORDS, derived=False)
    assert harness.adapter.span_member_ids() is None
    with pytest.raises(RedactionDriftError):
        assert_adapter_current(harness.adapter)
    report = adapter_status(harness.adapter)
    assert report["layers_in_sync"] is False
    assert report["span_layer_ready"] is False
    assert "refusal" in report


def test_never_derived_without_flags_serves(make_harness: Callable[..., Any]) -> None:
    harness = make_harness({1: 0, 2: 0}, derived=False)
    assert_adapter_current(harness.adapter)  # nothing flagged -> nothing to hide


def test_legacy_mirrorless_store_serves(make_harness: Callable[..., Any]) -> None:
    harness = make_harness(RECORDS)
    harness.drop_mirror()
    assert harness.adapter.mirror_flagged_ids() is None
    # Spans are still hidden and the live flag is still checked; the store
    # simply cannot offer mirror-freshness verification.
    assert_adapter_current(harness.adapter)
    guard = load_guard_from_adapter(harness.adapter)
    assert guard.classify(5, 1) == HIDDEN


def test_unreachable_store_refuses(make_harness: Callable[..., Any]) -> None:
    harness = make_harness(RECORDS)
    harness.make_unavailable()
    for method in ("span_member_ids", "flagged_ids", "mirror_flagged_ids", "counts"):
        with pytest.raises(RedactionDriftError):
            getattr(harness.adapter, method)()
    with pytest.raises(RedactionDriftError):
        assert_adapter_current(harness.adapter)


def test_status_report_is_counts_only(make_harness: Callable[..., Any]) -> None:
    report = adapter_status(make_harness(RECORDS).adapter)
    assert set(report["counts"]) >= {"records", "flagged"}
    assert all(isinstance(value, int) for value in report["counts"].values())


# ---------------------------------------------------------------------------
# Store-specific behavior
# ---------------------------------------------------------------------------


def test_adapters_import_and_resolve_without_drivers() -> None:
    import jaenys.adapters as adapters_pkg

    for attr in (
        "MongoDBAdapter",
        "RedisAdapter",
        "DynamoDBAdapter",
        "CouchbaseAdapter",
        "FirestoreAdapter",
    ):
        assert getattr(adapters_pkg, attr) is not None


def test_adapter_string_member_ids_are_coerced_not_trusted() -> None:
    """An adapter returning digit-string ids must not classify an in-span record clear."""

    class StringIdAdapter:
        name = "stringy"

        def span_member_ids(self) -> frozenset:
            return frozenset({"5", "6"})

        def flagged_ids(self) -> frozenset:
            return frozenset()

        def mirror_flagged_ids(self) -> frozenset:
            return frozenset()

        def counts(self) -> dict:
            return {"records": 7, "flagged": 0}

    guard = load_guard_from_adapter(StringIdAdapter())
    # Without coercion "5" != 5, so the in-span record would classify VISIBLE.
    assert guard.classify(5, 0) == HIDDEN
    assert guard.classify(6, 0) == HIDDEN


def test_adapter_corrupt_string_member_id_refuses() -> None:
    class CorruptIdAdapter:
        name = "stringy"

        def span_member_ids(self) -> frozenset:
            return frozenset({"junk"})

        def flagged_ids(self) -> frozenset:
            return frozenset()

        def mirror_flagged_ids(self) -> frozenset:
            return frozenset()

        def counts(self) -> dict:
            return {"records": 1, "flagged": 0}

    with pytest.raises(RedactionDriftError, match="stringy"):
        load_guard_from_adapter(CorruptIdAdapter())


@pytest.mark.parametrize("bad_name", ["$where", "a.b", "field$", "nested.path"])
def test_mongo_field_name_validation(bad_name: str) -> None:
    with pytest.raises(RedactionDriftError):
        _reject_unsafe_mongo_name(bad_name, kind="flag_column")


@pytest.mark.parametrize("bad_name", ["$where", "a.b"])
def test_firestore_field_name_validation(bad_name: str) -> None:
    with pytest.raises(RedactionDriftError):
        _reject_unsafe_firestore_name(bad_name, kind="flag_column")


def test_schema_mapping_blocks_operator_names_first() -> None:
    # The adapters' $/. rejection is defense in depth: the generic identifier
    # allowlist on SchemaMapping already refuses these at construction.
    with pytest.raises(RedactionDriftError):
        SchemaMapping(flag_column="$where")
    with pytest.raises(RedactionDriftError):
        SchemaMapping(record_table="a.b")


def test_mongo_counts_flag_stored_as_boolean() -> None:
    """A flag stored as a BSON boolean is still counted as flagged.

    flagged_ids coerces True -> 1, so counts() must agree; a bare {flag: 1}
    server-side count would miss a boolean-stored flag on a real server and
    under-report against the sync check.
    """

    harness = MongoHarness(RECORDS)
    mapping = harness.mapping
    for doc in harness.primary._collections[mapping.record_table]:
        if doc[mapping.record_id_column] == 2:
            doc[mapping.flag_column] = True  # was integer 1
    assert harness.adapter.counts() == {"records": 7, "flagged": 2}
    assert 2 in harness.adapter.flagged_ids()


def test_mongo_visibility_filter_shapes() -> None:
    adapter = MongoHarness(RECORDS).adapter
    assert adapter.visibility_filter([4, 5, 6]) == {
        "$and": [{"sensitive": {"$in": [0, False]}}, {"record_id": {"$nin": [4, 5, 6]}}]
    }
    # include_blur keeps a flag clause: a document with a missing/null/corrupt
    # flag must not serve clear on the blur surface.
    assert adapter.visibility_filter([6, 5, 4], include_blur=True) == {
        "$and": [
            {"sensitive": {"$in": [0, 1, False, True]}},
            {"record_id": {"$nin": [4, 5, 6]}},
        ]
    }
    assert adapter.visibility_filter([]) == {"sensitive": {"$in": [0, False]}}
    assert adapter.visibility_filter([], include_blur=True) == {
        "sensitive": {"$in": [0, 1, False, True]}
    }


@pytest.mark.parametrize("harness_name", ["mongodb", "dynamodb", "couchbase", "firestore"])
def test_document_adapter_duplicate_record_id_refuses(harness_name: str) -> None:
    harness = HARNESSES[harness_name]({1: 0, 2: 1})
    mapping = harness.mapping
    if harness_name == "mongodb":
        harness.primary._collections[mapping.record_table].append(
            {mapping.record_id_column: 1, mapping.flag_column: 0}
        )
    elif harness_name == "dynamodb":
        harness.records_table.items.append(
            {mapping.record_id_column: Decimal(1), mapping.flag_column: Decimal(0)}
        )
    elif harness_name == "couchbase":
        harness.records_collection.docs["duplicate"] = {
            mapping.record_id_column: 1,
            mapping.flag_column: 0,
        }
    else:
        harness.client.collection(mapping.record_table).docs()["duplicate"] = {
            mapping.record_id_column: 1,
            mapping.flag_column: 0,
        }

    with pytest.raises(RedactionDriftError, match="duplicate record ids"):
        assert_adapter_current(harness.adapter)


def test_mongo_visibility_filter_refuses_non_integral_span_ids() -> None:
    """3.7 must refuse, not silently truncate to record 3 and leak record 3.7's row."""

    adapter = MongoHarness(RECORDS).adapter
    with pytest.raises(RedactionDriftError, match="mongodb"):
        adapter.visibility_filter([3.7])
    with pytest.raises(RedactionDriftError, match="mongodb"):
        adapter.visibility_filter(["junk"], include_blur=True)


def test_redis_prefers_sscan_iter_over_smembers() -> None:
    """SMEMBERS on a huge set blocks the server; SSCAN pages incrementally."""

    class ScanningFakeRedis(FakeRedisClient):
        def __init__(self) -> None:
            super().__init__()
            self.sscan_calls: list[str] = []
            self.smembers_calls: list[str] = []

        def sscan_iter(self, key: str):
            self._check()
            self.sscan_calls.append(key)
            yield from self._sets.get(key, set())

        def smembers(self, key: str) -> set[bytes]:
            self.smembers_calls.append(key)
            return super().smembers(key)

    client = ScanningFakeRedis()
    client.sadd("guard_test:flagged", 2, 5)
    adapter = RedisAdapter(client, prefix="guard_test")
    assert adapter.flagged_ids() == frozenset({2, 5})
    assert client.sscan_calls == ["guard_test:flagged"]
    assert client.smembers_calls == []


def test_redis_falls_back_to_smembers_without_sscan_iter() -> None:
    # The base fake has no sscan_iter, so the whole contract suite above
    # already exercises the fallback; this pins the behavior explicitly.
    harness = RedisHarness(RECORDS)
    assert not hasattr(harness.client, "sscan_iter")
    assert harness.adapter.flagged_ids() == FLAGGED


def test_dynamodb_custom_meta_key_addresses_readiness_item() -> None:
    """Span tables not partition-keyed on "layer" pass their own meta key."""

    records_table = FakeDynamoTable()
    span_table = FakeDynamoTable()
    mapping = SchemaMapping()
    records_table.items = [{mapping.record_id_column: Decimal(1), mapping.flag_column: Decimal(0)}]
    # The readiness item lives under a "pk" partition key and carries no
    # "layer" attribute at all -- only the custom meta_key can address it.
    span_table.items = [
        {"pk": "readiness", "mirror_ready": True},
        {"layer": "span_member", mapping.span_member_id_column: Decimal(1)},
    ]
    adapter = DynamoDBAdapter(
        records_table, span_table, mapping=mapping, meta_key={"pk": "readiness"}
    )
    assert adapter.span_member_ids() == frozenset({1})

    # The default key misses the item entirely -> layer reads as never derived.
    default_adapter = DynamoDBAdapter(records_table, span_table, mapping=mapping)
    assert default_adapter.span_member_ids() is None


def test_dynamodb_paginates_and_converts_decimals() -> None:
    # Page size 2 with 7 records forces multiple LastEvaluatedKey round trips
    # on every scan, and all ids arrive as Decimal.
    harness = DynamoHarness(RECORDS)
    assert harness.adapter.counts() == {"records": 7, "flagged": 2}
    assert all(isinstance(record_id, int) for record_id in harness.adapter.span_member_ids())


def test_couchbase_meta_doc_never_counted_as_member() -> None:
    adapter = CouchbaseHarness(RECORDS).adapter
    assert adapter.span_member_ids() == SPAN_MEMBERS


# ---------------------------------------------------------------------------
# Corrupt store data must fail closed (RedactionDriftError naming the store),
# never a raw ValueError/TypeError and never a silently truncated float.
# ---------------------------------------------------------------------------


def test_in_memory_refuses_corrupt_flags() -> None:
    with pytest.raises(RedactionDriftError, match="flag must be 0 or 1"):
        InMemoryAdapter({1: 2})

    adapter = InMemoryAdapter({1: 0})
    with pytest.raises(RedactionDriftError, match="flag is not an integral value"):
        adapter.set_flag(1, 3.7)


def test_mongo_corrupt_span_member_id_fails_closed() -> None:
    harness = MongoHarness(RECORDS)
    harness.span.create_collection_with(
        harness.mapping.span_member_table,
        [{harness.mapping.span_member_id_column: "abc"}],
    )
    with pytest.raises(RedactionDriftError, match="mongodb"):
        harness.adapter.span_member_ids()


def test_mongo_non_integral_float_span_member_id_fails_closed() -> None:
    harness = MongoHarness(RECORDS)
    harness.span.create_collection_with(
        harness.mapping.span_member_table,
        [{harness.mapping.span_member_id_column: 3.7}],
    )
    with pytest.raises(RedactionDriftError, match="mongodb"):
        harness.adapter.span_member_ids()


def test_mongo_missing_span_member_id_fails_closed() -> None:
    harness = MongoHarness(RECORDS)
    harness.span.create_collection_with(
        harness.mapping.span_member_table,
        [{"wrong_field": 4}],
    )
    with pytest.raises(RedactionDriftError, match="missing required field"):
        harness.adapter.span_member_ids()


def test_mongo_corrupt_flag_fails_closed() -> None:
    harness = MongoHarness(RECORDS)
    harness.primary._collections[harness.mapping.record_table][0][harness.mapping.flag_column] = 2
    with pytest.raises(RedactionDriftError, match="flag must be 0 or 1"):
        harness.adapter.flagged_ids()


def test_redis_corrupt_span_member_id_fails_closed() -> None:
    harness = RedisHarness(RECORDS)
    harness.client.sadd("guard_test:span_members", "abc")
    with pytest.raises(RedactionDriftError, match="redis"):
        harness.adapter.span_member_ids()


def test_redis_counts_rejects_junk_record_count() -> None:
    harness = RedisHarness(RECORDS)
    harness.client.set("guard_test:record_count", "not-a-number")
    with pytest.raises(RedactionDriftError, match="record count"):
        harness.adapter.counts()


def test_redis_counts_missing_count_key_with_flags_refuses() -> None:
    """records: 0 alongside flagged: N is a provably wrong report -- refuse it."""

    harness = RedisHarness(RECORDS)
    harness.client.delete("guard_test:record_count")
    with pytest.raises(RedactionDriftError, match="record.count"):
        harness.adapter.counts()


def test_redis_counts_missing_count_key_without_flags_is_zero() -> None:
    harness = RedisHarness({})
    harness.client.delete("guard_test:record_count")
    assert harness.adapter.counts() == {"records": 0, "flagged": 0}


def test_redis_counts_rejects_negative_record_count() -> None:
    harness = RedisHarness(RECORDS)
    harness.client.set("guard_test:record_count", "-3")
    with pytest.raises(RedactionDriftError, match="negative"):
        harness.adapter.counts()


def test_redis_counts_rejects_records_below_flagged() -> None:
    """Fewer records than flagged ids is a provably wrong report -- refuse it."""

    harness = RedisHarness(RECORDS)  # 2 flagged
    harness.client.set("guard_test:record_count", "1")
    with pytest.raises(RedactionDriftError, match="record.count"):
        harness.adapter.counts()


def test_dynamodb_corrupt_span_member_id_fails_closed() -> None:
    harness = DynamoHarness(RECORDS)
    harness.span_table.items.append(
        {"layer": "span_member", harness.mapping.span_member_id_column: "abc"}
    )
    with pytest.raises(RedactionDriftError, match="dynamodb"):
        harness.adapter.span_member_ids()


def test_dynamodb_missing_span_member_id_fails_closed() -> None:
    harness = DynamoHarness(RECORDS)
    harness.span_table.items.append({"layer": "span_member", "wrong_field": Decimal(4)})
    with pytest.raises(RedactionDriftError, match="missing required attribute"):
        harness.adapter.span_member_ids()


def test_dynamodb_non_integral_decimal_span_member_id_fails_closed() -> None:
    harness = DynamoHarness(RECORDS)
    harness.span_table.items.append(
        {"layer": "span_member", harness.mapping.span_member_id_column: Decimal("3.5")}
    )
    with pytest.raises(RedactionDriftError, match="dynamodb"):
        harness.adapter.span_member_ids()


def test_dynamodb_integral_decimal_span_member_id_passes() -> None:
    harness = DynamoHarness(RECORDS)
    harness.span_table.items.append(
        {"layer": "span_member", harness.mapping.span_member_id_column: Decimal("3")}
    )
    assert 3 in harness.adapter.span_member_ids()


def test_dynamodb_counts_across_pagination() -> None:
    # Page size 2 over 7 records forces multiple LastEvaluatedKey round
    # trips for both the unfiltered and flag-filtered COUNT scans.
    harness = DynamoHarness(RECORDS)
    assert harness.records_table.page_size == 2
    assert harness.adapter.counts() == {"records": 7, "flagged": 2}


def test_dynamodb_corrupt_flag_fails_closed() -> None:
    harness = DynamoHarness(RECORDS)
    harness.records_table.items[0][harness.mapping.flag_column] = Decimal(2)
    with pytest.raises(RedactionDriftError, match="flag must be 0 or 1"):
        harness.adapter.flagged_ids()


def test_firestore_corrupt_span_member_id_fails_closed() -> None:
    harness = FirestoreHarness(RECORDS)
    span_docs = harness.client.collection(harness.mapping.span_member_table).docs()
    span_docs["bad"] = {harness.mapping.span_member_id_column: "abc"}
    with pytest.raises(RedactionDriftError, match="firestore"):
        harness.adapter.span_member_ids()


def test_firestore_missing_span_member_id_fails_closed() -> None:
    harness = FirestoreHarness(RECORDS)
    span_docs = harness.client.collection(harness.mapping.span_member_table).docs()
    span_docs["bad"] = {"wrong_field": 4}
    with pytest.raises(RedactionDriftError, match="missing required field"):
        harness.adapter.span_member_ids()


def test_firestore_corrupt_flag_fails_closed() -> None:
    harness = FirestoreHarness(RECORDS)
    docs = harness.client.collection(harness.mapping.record_table).docs()
    docs["r1"][harness.mapping.flag_column] = "not-a-flag"
    with pytest.raises(RedactionDriftError, match="firestore"):
        harness.adapter.flagged_ids()


def test_firestore_missing_flag_fails_closed() -> None:
    harness = FirestoreHarness(RECORDS)
    docs = harness.client.collection(harness.mapping.record_table).docs()
    del docs["r1"][harness.mapping.flag_column]
    with pytest.raises(RedactionDriftError, match="missing required field"):
        harness.adapter.flagged_ids()


def test_couchbase_corrupt_span_member_id_fails_closed() -> None:
    harness = CouchbaseHarness(RECORDS)
    harness.span_collection.docs["bad"] = {harness.mapping.span_member_id_column: "abc"}
    with pytest.raises(RedactionDriftError, match="couchbase"):
        harness.adapter.span_member_ids()


def test_couchbase_marker_without_id_is_tolerated() -> None:
    """A "document body" scan (no id attached) must still exclude the marker.

    The _span_meta marker carries no member-id field, so it is skipped, not
    refused; a field-less body cannot un-hide a record.  A body that does
    carry the field with a corrupt value still refuses (see the test above).
    """

    records = FakeCouchbaseCollection()
    records.docs = {"r1": {"record_id": 1, "sensitive": 0}}
    spans = FakeCouchbaseCollection()
    spans.docs = {"_span_meta": {"derived": True}, "m5": {"record_id": 5}}
    mirror = FakeCouchbaseCollection()

    def body_only_scan(collection: FakeCouchbaseCollection) -> list[dict]:
        collection._check()
        return [dict(body) for body in collection.docs.values()]

    adapter = CouchbaseAdapter(records, spans, mirror, scan=body_only_scan)
    assert adapter.span_member_ids() == frozenset({5})


def test_couchbase_corrupt_flag_fails_closed() -> None:
    harness = CouchbaseHarness(RECORDS)
    harness.records_collection.docs["r1"][harness.mapping.flag_column] = 3.7
    with pytest.raises(RedactionDriftError, match="couchbase"):
        harness.adapter.flagged_ids()


def test_couchbase_missing_flag_fails_closed() -> None:
    harness = CouchbaseHarness(RECORDS)
    del harness.records_collection.docs["r1"][harness.mapping.flag_column]
    with pytest.raises(RedactionDriftError, match="missing required field"):
        harness.adapter.flagged_ids()
