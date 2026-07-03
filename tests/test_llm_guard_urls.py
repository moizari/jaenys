"""Accept/reject behavior for the local-only endpoint guard."""

from __future__ import annotations

import pytest

from local_llm_guard import DEFAULT_ALLOWED_HOSTS, LocalEndpointError, enforce_local_url


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://localhost:11434/", "http://localhost:11434"),
        ("http://127.0.0.1:11434", "http://127.0.0.1:11434"),
        ("http://host.docker.internal:11434", "http://host.docker.internal:11434"),
        ("http://[::1]:11434", "http://[::1]:11434"),
    ],
)
def test_enforce_local_url_accepts_configured_local_hosts(url: str, expected: str) -> None:
    assert enforce_local_url(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com:11434",
        "http://localhost.evil.test:11434",
        "file:///tmp/x",
        "http://localhost:11434/api/chat",
        "https://user:pw@localhost:11434",
        "http://localhost:11434/?x=1",
        "http://localhost:11434/#frag",
        "ftp://localhost:11434",
    ],
)
def test_enforce_local_url_rejects_unsafe_urls(url: str) -> None:
    with pytest.raises(LocalEndpointError):
        enforce_local_url(url)


def test_enforce_local_url_custom_allowed_hosts() -> None:
    custom = frozenset({"myhost.internal"})

    assert enforce_local_url("http://myhost.internal:9999", allowed_hosts=custom) == (
        "http://myhost.internal:9999"
    )

    # The custom allowlist is not merged with the default one.
    with pytest.raises(LocalEndpointError):
        enforce_local_url("http://localhost:11434", allowed_hosts=custom)

    # The default call still rejects a host that is only allowed above.
    with pytest.raises(LocalEndpointError):
        enforce_local_url("http://myhost.internal:9999", allowed_hosts=DEFAULT_ALLOWED_HOSTS)


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:abc",
        "http://localhost:99999",
        "http://[::1",
    ],
)
def test_enforce_local_url_rejects_malformed_port_and_ipv6(url: str) -> None:
    with pytest.raises(LocalEndpointError) as exc_info:
        enforce_local_url(url)
    assert "Malformed" in str(exc_info.value)


def test_enforce_local_url_accepts_valid_ipv6() -> None:
    assert enforce_local_url("http://[::1]:11434") == "http://[::1]:11434"


def test_enforce_local_url_port_boundaries() -> None:
    # urlparse accepts ":0" without raising, but port 0 is not connectable --
    # the documented range is [1, 65535] and both edges are enforced.
    with pytest.raises(LocalEndpointError, match=r"\[1, 65535\]"):
        enforce_local_url("http://localhost:0")
    assert enforce_local_url("http://localhost:1") == "http://localhost:1"
    assert enforce_local_url("http://localhost:65535") == "http://localhost:65535"


@pytest.mark.parametrize(
    "url",
    [
        "http://:@localhost:11434",
        "http://user@localhost:11434",
        "http://:secret@localhost:11434",
    ],
)
def test_enforce_local_url_rejects_empty_credentials(url: str) -> None:
    # An empty username/password string is still an embedded credential --
    # falsy is not the same as absent.
    with pytest.raises(LocalEndpointError):
        enforce_local_url(url)
