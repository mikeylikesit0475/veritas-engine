"""Veritas FastAPI server.

Sprint 2.4: ``POST /claims/{id}/adjudicate`` + Postgres-backed sessions.
The HITL pause survives a restart — the session is in Postgres, not memory.

ADR-008: Postgres, not SQLite. ADR-009: no external key required.
CLAUDE.md #6: CI never calls a live model.
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from google.adk.runners import Runner
from google.adk.sessions import BaseSessionService, DatabaseSessionService, InMemorySessionService
from google.genai import types as genai_types
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from veritas.backends_registry import clear_backends, register_backends
from veritas.graph.arbiter import ARBITER_MODEL_STATE_KEY
from veritas.graph.claim_state import ClaimState
from veritas.graph.decompose import DECOMPOSE_MODEL_STATE_KEY
from veritas.graph.fan_out import FANOUT_MODEL_STATE_KEY
from veritas.graph.fast_check import FAST_CHECK_MODEL_STATE_KEY
from veritas.graph.hitl import (
    AdjudicationRequest,
    AdjudicationResponse,
    build_adjudication_function_response,
)
from veritas.graph.triage import TRIAGE_MODEL_STATE_KEY
from veritas.graph.wiring import build_graph
from veritas.models import Tier, model_for
from veritas.server.backends import HttpFetchBackend, HttpSearchBackend
from veritas.tools.fetch_page import FETCH_BACKEND_STATE_KEY
from veritas.tools.grounded_search import SEARCH_BACKEND_STATE_KEY

__all__ = ["ClaimRequest", "ClaimResponse", "app", "create_app"]

# Postgres URL for the session service. Default: the local TimescaleDB container.
# Override with VERITAS_DB_URL. If Postgres is unavailable, falls back to
# InMemorySessionService (the HITL pause won't survive a restart, but the demo
# can still run).
DEFAULT_DB_URL = "postgresql+asyncpg://mycelium:mycelium@localhost:5432/veritas"


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
    use_postgres: bool | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    By default, live HTTP backends are constructed and closed on shutdown. Tests
    pass fakes via the kwargs — those are used as-is and not closed (the test owns
    them). This is the test seam; the server design deliberately doesn't hide it
    behind monkeypatching.

    Args:
        use_postgres: If True, use DatabaseSessionService with the Postgres URL
            from ``VERITAS_DB_URL`` (default: the local container). If False,
            use InMemorySessionService. If None (default), try Postgres and
            fall back to InMemory if unavailable.
    """
    # Sprint 2.4: Postgres-backed sessions for HITL durability (ADR-008).
    db_url = os.environ.get("VERITAS_DB_URL", DEFAULT_DB_URL)
    session_service: BaseSessionService
    if use_postgres is False:
        session_service = InMemorySessionService()  # type: ignore[no-untyped-call]
    elif use_postgres is True:
        session_service = DatabaseSessionService(db_url=db_url)
    else:
        # Auto-detect: try Postgres, fall back to InMemory.
        try:
            session_service = DatabaseSessionService(db_url=db_url)
        except Exception:
            session_service = InMemorySessionService()  # type: ignore[no-untyped-call]

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
            DECOMPOSE_MODEL_STATE_KEY: resolved_model,  # judgement tier
            ARBITER_MODEL_STATE_KEY: resolved_model,  # judgement tier
            TRIAGE_MODEL_STATE_KEY: resolved_model,  # judgement tier
            FANOUT_MODEL_STATE_KEY: model_for(Tier.FANOUT),  # fan-out tier
        },
    )

    @asynccontextmanager
    async def _lifespan(app: FastAPI) -> Any:
        # Sprint 4.3: OTel traces — spans cover the whole graph.
        from veritas.telemetry import setup_telemetry

        setup_telemetry()
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

    @app.post("/claims/{claim_id}/adjudicate", response_model=AdjudicationResponse)
    async def adjudicate(claim_id: str, req: AdjudicationRequest) -> AdjudicationResponse:
        """Resume a paused claim with a human verdict.

        Sprint 2.4: the HITL pause survives a restart because the session is in
        Postgres. This endpoint constructs the FunctionResponse that resumes the
        graph, then runs it to completion (the report node).
        """
        session = await session_service.get_session(
            app_name="veritas", user_id="api", session_id=claim_id
        )
        if session is None:
            raise HTTPException(status_code=404, detail="claim session not found")

        # Build the FunctionResponse that carries the human's verdict.
        resume_message = build_adjudication_function_response(req)

        # Resume the runner. The graph continues from the HITL pause to the report.
        try:
            async for _event in runner.run_async(
                user_id="api", session_id=claim_id, new_message=resume_message
            ):
                pass  # consume events; the graph runs to completion
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"resume failed: {e}") from e

        return AdjudicationResponse(
            claim_id=claim_id,
            status="adjudicated",
            message=f"Claim resumed with verdict={req.verdict}. Graph continued to report.",
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/models")
    async def list_models() -> dict[str, object]:
        """List available models — Ollama models + Gemini if a key is set."""
        models: list[dict[str, str]] = []

        # Ollama models.
        try:
            import httpx

            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get("http://localhost:11434/api/tags")
                if resp.status_code == 200:
                    for m in resp.json().get("models", []):
                        models.append({
                            "name": m["name"],
                            "tier": "ollama",
                            "label": m["name"],
                        })
        except Exception:
            pass

        # Gemini (if key is set).
        if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            models.append({"name": "gemini-2.5-flash", "tier": "gemini", "label": "gemini-2.5-flash"})
            models.append({"name": "gemini-2.5-pro", "tier": "gemini", "label": "gemini-2.5-pro"})

        # Currently selected models.
        from veritas.models import Backend, _env_backend
        j_backend = _env_backend("VERITAS_JUDGEMENT")
        f_backend = _env_backend("VERITAS_FANOUT")
        j_model = os.environ.get("VERITAS_JUDGEMENT_MODEL", "")
        f_model = os.environ.get("VERITAS_FANOUT_MODEL", "")

        return {
            "models": models,
            "current": {
                "judgement": j_model or ("gemini-2.5-flash" if j_backend is Backend.GEMINI else "ollama:default"),
                "fanout": f_model or ("gemini-2.5-flash" if f_backend is Backend.GEMINI else "ollama:default"),
            },
        }

    @app.post("/models/select")
    async def select_model(
        judgement_model: str | None = None,
        fanout_model: str | None = None,
    ) -> dict[str, str]:
        """Switch the active models at runtime. Re-registers in the backend registry."""
        if judgement_model:
            from google.adk.models.google_llm import Gemini
            from google.adk.models.lite_llm import LiteLlm

            if judgement_model.startswith("gemini-"):
                if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
                    raise HTTPException(400, detail="No Gemini API key set")
                jm: Any = Gemini(model=judgement_model)
            else:
                name = judgement_model if judgement_model.startswith("ollama") else f"ollama_chat/{judgement_model}"
                jm = LiteLlm(model=name)
            register_backends(
                "veritas",
                **{
                    FAST_CHECK_MODEL_STATE_KEY: jm,
                    TRIAGE_MODEL_STATE_KEY: jm,
                    DECOMPOSE_MODEL_STATE_KEY: jm,
                    ARBITER_MODEL_STATE_KEY: jm,
                },
            )

        if fanout_model:
            from google.adk.models.google_llm import Gemini
            from google.adk.models.lite_llm import LiteLlm

            if fanout_model.startswith("gemini-"):
                if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
                    raise HTTPException(400, detail="No Gemini API key set")
                fm: Any = Gemini(model=fanout_model)
            else:
                name = fanout_model if fanout_model.startswith("ollama") else f"ollama_chat/{fanout_model}"
                fm = LiteLlm(model=name)
            register_backends("veritas", **{FANOUT_MODEL_STATE_KEY: fm})

        return {"status": "ok", "judgement": judgement_model or "(unchanged)", "fanout": fanout_model or "(unchanged)"}

    # Serve the web UI (Sprint 1.6 + 2.6 — evidence panel + live graph).
    from pathlib import Path

    from fastapi.staticfiles import StaticFiles

    web_dir = Path(__file__).resolve().parents[2] / "web" / "public"
    if web_dir.exists():
        app.mount("/ui", StaticFiles(directory=str(web_dir), html=True), name="ui")

    return app


# Module-level app for ``uvicorn veritas.server.app:app``.
app = create_app()
