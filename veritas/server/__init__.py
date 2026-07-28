"""Veritas server package.

FastAPI app that wires backends into ADK session state and streams graph events as
SSE. The graph is the same one tests run against fakes (ADR-001 boundary).
"""

from __future__ import annotations

from veritas.server.app import app, create_app

__all__ = ["app", "create_app"]
