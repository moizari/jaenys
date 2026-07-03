"""Privacy-by-construction guard for local LLM endpoints (Ollama-style).

Call :func:`enforce_local_url` before every request to a local model server
so that data can never leave the machine via a mistyped or tampered base
URL. The check is intentionally strict and dependency-free (stdlib
``urllib.parse`` only): scheme, host, credentials, query/fragment, and path
are all validated, and any violation raises :class:`LocalEndpointError`
rather than silently continuing.
"""

from __future__ import annotations

import urllib.parse

__all__ = ["DEFAULT_ALLOWED_HOSTS", "LocalEndpointError", "enforce_local_url"]

# Both loopback names, the IPv6 loopback, and Docker's alias for the host
# machine (the usual way a containerized app reaches a model server on it).
DEFAULT_ALLOWED_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "host.docker.internal"})


class LocalEndpointError(RuntimeError):
    """Raised when a base URL is not explicitly local."""


def enforce_local_url(
    url: str,
    *,
    allowed_hosts: frozenset[str] = DEFAULT_ALLOWED_HOSTS,
) -> str:
    """Validate that ``url`` points at a local endpoint and normalize it.

    Hard-fail checks (any violation raises :class:`LocalEndpointError`):

    * scheme must be ``http`` or ``https``
    * the lowercased hostname must be in ``allowed_hosts``
    * no username/password embedded in the URL
    * no query string or fragment
    * path must be empty or ``"/"``
    * port must be a valid integer in range [1, 65535]

    Returns the normalized base URL as ``scheme://host[:port]`` with no
    trailing slash. Malformed URLs raise :class:`LocalEndpointError`.
    """

    try:
        parsed = urllib.parse.urlparse(url)
        # Access hostname and port to trigger any ValueError from malformed input.
        host = (parsed.hostname or "").lower()
        port = parsed.port
    except ValueError as exc:
        raise LocalEndpointError(f"Malformed endpoint URL: {url}") from exc

    allowed = {allowed.lower() for allowed in allowed_hosts}

    if parsed.scheme not in {"http", "https"} or host not in allowed:
        raise LocalEndpointError(f"Refusing non-local endpoint URL: {url}")
    if parsed.username is not None or parsed.password is not None:
        raise LocalEndpointError("Endpoint URL must not include credentials.")
    if parsed.query or parsed.fragment:
        raise LocalEndpointError("Endpoint URL must not include query strings or fragments.")
    if parsed.path not in {"", "/"}:
        raise LocalEndpointError("Endpoint URL must point to the local server root.")
    # urlparse accepts ":0" without raising, but port 0 is not a connectable
    # endpoint (it means "ephemeral" at bind time) -- enforce the documented
    # [1, 65535] range explicitly.
    if port == 0:
        raise LocalEndpointError("Endpoint URL port must be in range [1, 65535].")

    # urlparse gives hostname "::1" (brackets stripped) for "http://[::1]:11434";
    # rebuilding the netloc must re-add brackets for any IPv6 literal.
    netloc_host = f"[{host}]" if ":" in host else host
    netloc = netloc_host
    if port is not None:
        netloc = f"{netloc_host}:{port}"
    return urllib.parse.urlunsplit((parsed.scheme, netloc, "", "", "")).rstrip("/")
