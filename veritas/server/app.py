"""Veritas FastAPI server.

Sprint 0.5: ``POST /claims`` accepts a claim, ``GET /claims/{id}/events`` streams
the graph's events as SSE. The server is thin — it wires backends into session state
and delegates to the ADK runner. The graph is the same one tests run against fakes.

ADR-009: no external key required for the demo. Backends default to the HTTP
(DuckDuckGo + httpx) implementations; the model factory defaults to Ollama.

CI never calls a live model (CLAUDE.md #6). The server module is import-safe in CI
(it doesn't make calls at import time), but it is not exercised by tests with live
backends — only with injected fakes.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from veritas.backends_registry import clear_backends, register_backends
from veritas.graph.claim_state import ClaimState
from veritas.graph.fast_check import FAST_CHECK_MODEL_STATE_KEY
from veritas.graph.wiring import build_graph
from veritas.models import Tier, model_for
from veritas.server.backends import HttpFetchBackend, HttpSearchBackend
from veritas.tools.fetch_page import FETCH_BACKEND_STATE_KEY
from veritas.tools.grounded_search import SEARCH_BACKEND_STATE_KEY

__all__ = ["ClaimRequest", "ClaimResponse", "app", "create_app"]


class ClaimRequest(BaseModel):
    """The body of ``POST /claims``."""

    claim: str = Field(..., min_length=1, max_length=3000, description="The claim to check.")


class ClaimResponse(BaseModel):
    """The body of ``POST /claims``'s response."""

    claim_id: str
    events_url: str


def _event_to_sse_dict(event: Any) -> dict[str, str]:
    """Serialise an ADK ``Event`` into an SSE payload dict.

    SSE clients see ``event: <type>`` and ``data: <json>``. We emit the node name
    as the event type and the event's content as the data, so a browser can light
    up graph nodes as events arrive.
    """
    node = event.node_name or event.author or "graph"
    text = ""
    if event.content and event.content.parts:
        for part in event.content.parts:
            if part.text and not part.thought:
                text += part.text
    payload: dict[str, Any] = {
        "node": node,
        "text": text,
        "turn_complete": bool(event.turn_complete),
        "partial": bool(event.partial),
    }
    # If the event carries a structured result (from a FunctionNode return), it's
    # in the text as JSON. We pass it through verbatim — the UI can parse it.
    return {"event": node, "data": json.dumps(payload)}


def create_app(
    *,
    search_backend: Any | None = None,
    fetch_backend: Any | None = None,
    fast_check_model: Any | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    By default, live HTTP backends are constructed and closed on shutdown. Tests
    pass fakes via the kwargs — those are used as-is and not closed (the test owns
    them). This is the test seam; the server design deliberately doesn't hide it
    behind monkeypatching.
    """
    session_service = InMemorySessionService()  # type: ignore[no-untyped-call]
    # Sprint 2.4 swaps this for DatabaseSessionService(postgresql+asyncpg://...).
    # The SessionService interface is the seam (ADR-008).
    graph = build_graph()
    # One runner for the app; sessions keep per-claim state isolated.
    runner = Runner(
        app_name="veritas",
        agent=graph,  # type: ignore[arg-type]  # Workflow is a BaseAgent subclass at runtime
        session_service=session_service,
    )

    owns_search = search_backend is None
    owns_fetch = fetch_backend is None
    search_backend = search_backend or HttpSearchBackend()
    fetch_backend = fetch_backend or HttpFetchBackend()
    resolved_model = fast_check_model or model_for(Tier.JUDGEMENT)

    # Register backends at the app level so graph nodes can look them up without
    # round-tripping through session state (which ADK deep-copies on read).
    register_backends(
        "veritas",
        **{
            SEARCH_BACKEND_STATE_KEY: search_backend,
            FETCH_BACKEND_STATE_KEY: fetch_backend,
            FAST_CHECK_MODEL_STATE_KEY: resolved_model,
        },
    )

    @asynccontextmanager
    async def _lifespan(app: FastAPI) -> Any:
        try:
            yield
        finally:
            if owns_search:
                await search_backend.aclose()
            if owns_fetch:
                await fetch_backend.aclose()
            clear_backends("veritas")

    app = FastAPI(
        title="Veritas Engine",
        version="0.1.0",
        description="Adversarial fact-checking on Google ADK. Portfolio PoC.",
        lifespan=_lifespan,
    )

    @app.post("/claims", response_model=ClaimResponse)
    async def submit_claim(req: ClaimRequest) -> ClaimResponse:
        """Accept a claim and create a session. Stream events from the events URL."""
        session = await session_service.create_session(
            app_name="veritas", user_id="api"
        )
        return ClaimResponse(
            claim_id=session.id,
            events_url=f"/claims/{session.id}/events",
        )

    @app.get("/claims/{claim_id}/events")
    async def stream_events(claim_id: str, claim: str) -> EventSourceResponse:
        """Stream the graph's events for a claim as SSE.

        The claim text is passed as a query param so GET is self-contained and
        curl-testable. (Sprint 1 will persist the claim with the session and drop
        this param once a UI is wired.)
        """
        session = await session_service.get_session(
            app_name="veritas", user_id="api", session_id=claim_id
        )
        if session is None:
            raise HTTPException(status_code=404, detail="claim session not found")

        message = genai_types.Content(
            role="user",
            parts=[genai_types.Part.from_text(text=claim)],
        )

        async def _gen() -> Any:
            try:
                async for event in runner.run_async(
                    user_id="api", session_id=claim_id, new_message=message
                ):
                    yield _event_to_sse_dict(event)
                # Final event: the resolved claim state, so the client gets the verdict.
                final = await session_service.get_session(
                    app_name="veritas", user_id="api", session_id=claim_id
                )
                if final is not None:
                    claim_state = ClaimState.from_state(dict(final.state))
                    yield {
                        "event": "verdict",
                        "data": claim_state.model_dump_json(),
                    }
            except asyncio.CancelledError:
                # Client disconnected — stop cleanly.
                pass

        return EventSourceResponse(_gen())

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


# Module-level app for ``uvicorn veritas.server.app:app``.
app = create_app()
