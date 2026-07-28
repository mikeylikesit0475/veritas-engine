"""App-level backend registry.

Backends (search, fetch, model) are *app-level singletons*, not per-session state.
ADK's ``InMemorySessionService`` deep-copies session state on read, which breaks on
objects holding locks (e.g. ``httpx.AsyncClient``). So we register backends here by
``app_name`` and the graph nodes look them up by the ``app_name`` on their context.

This is the seam between the server (which wires real backends) and the graph (which
calls them). Tests register fakes; CI never registers anything live
(CLAUDE.md #6).
"""

from __future__ import annotations

from typing import Any

__all__ = ["BACKEND_KEYS", "clear_backends", "get_backend", "register_backends"]

# Registered as: _registry[app_name][key] = backend
_registry: dict[str, dict[str, Any]] = {}

# The keys the graph nodes look up. Kept in sync with the node code.
BACKEND_KEYS = frozenset(
    {
        "veritas.search.backend",
        "veritas.fetch.backend",
        "veritas.fast_check.model",
    }
)


def register_backends(app_name: str, **backends: Any) -> None:
    """Register backends for an app. Call once at server startup."""
    _registry.setdefault(app_name, {}).update(backends)


def get_backend(app_name: str, key: str) -> Any:
    """Look up a backend. Raises loudly if missing (ADR-004 spirit: no silent fail)."""
    app_backends = _registry.get(app_name)
    if app_backends is None:
        raise RuntimeError(
            f"No backends registered for app {app_name!r}. Call "
            f"register_backends({app_name!r}, ...) at startup."
        )
    if key not in app_backends:
        raise RuntimeError(
            f"Backend {key!r} not registered for app {app_name!r}. "
            f"Registered: {list(app_backends)}"
        )
    return app_backends[key]


def clear_backends(app_name: str | None = None) -> None:
    """Clear the registry. Pass ``app_name`` to clear one app; ``None`` clears all."""
    if app_name is None:
        _registry.clear()
    else:
        _registry.pop(app_name, None)
