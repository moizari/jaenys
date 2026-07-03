"""Store adapters for non-SQL databases.

Concrete adapters import no drivers at module import time: callers pass an
already-constructed client object (pymongo Database, redis-py client, boto3
Table, Couchbase Collection, Firestore Client).

The five concrete adapter classes are re-exported lazily (PEP 562
``__getattr__``) so importing this package -- or even naming one adapter,
e.g. ``from jaenys.adapters import MongoDBAdapter`` -- never eagerly
imports the *other* adapters' submodules. None of the adapter submodules
import their driver at module level either, so this is belt-and-suspenders:
even a direct ``import jaenys.adapters.mongodb`` works with no
driver installed.
"""

from .memory import InMemoryAdapter
from .protocol import StoreAdapter

__all__ = [
    "StoreAdapter",
    "InMemoryAdapter",
    "MongoDBAdapter",
    "RedisAdapter",
    "DynamoDBAdapter",
    "CouchbaseAdapter",
    "FirestoreAdapter",
]

_LAZY_ADAPTERS = {
    "MongoDBAdapter": (".mongodb", "MongoDBAdapter"),
    "RedisAdapter": (".redis", "RedisAdapter"),
    "DynamoDBAdapter": (".dynamodb", "DynamoDBAdapter"),
    "CouchbaseAdapter": (".couchbase", "CouchbaseAdapter"),
    "FirestoreAdapter": (".firestore", "FirestoreAdapter"),
}


def __getattr__(name: str) -> object:
    try:
        module_name, attr_name = _LAZY_ADAPTERS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    import importlib

    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
