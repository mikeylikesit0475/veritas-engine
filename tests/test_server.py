"""Tests for the FastAPI server — POST /claims + SSE stream (ARCH §9).

Uses fakes for both backends and the model via ``create_app()``'s test seam. No live
calls (CLAUDE.md #6).
"""

from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types as genai_types

from veritas.server.app import create_app
from veritas.tools.backends import FakeSearchBackend, SearchHit


class _ServerVerdictStub(BaseLlm):
    """Stub model returning a fixed JSON verdict, for the server test."""

    model: str = "server-verdict-stub"

    def __init__(self, json_response: str, **kwargs: Any) -> None:
        super().__init__(model="server-verdict-stub", **kwargs)
        self._json = json_response

    @classmethod
    def supported_models(cls) -> list[str]:
        return [r"server-verdict-stub.*"]

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> Any:
        content = genai_types.Content(
            role="model",
            parts=[genai_types.Part.from_text(text=self._json)],
        )
        yield LlmResponse(content=content, turn_complete=True)


def _client_with_fakes(stub_json: str, hits: list[SearchHit]) -> TestClient:
    """Build a server wired to fakes instead of live backends (the test seam)."""
    fake_search = FakeSearchBackend(hits)
    fake_model = _ServerVerdictStub(stub_json)
    # Register fakes at the app level. create_app uses them because we pass them
    # in as kwargs (so the server doesn't construct live HTTP backends).
    app = create_app(
        search_backend=fake_search,
        fetch_backend=None,  # not exercised in these tests
        fast_check_model=fake_model,
    )
    return TestClient(app)


def test_post_claims_returns_claim_id_and_events_url() -> None:
    client = _client_with_fakes(
        '{"label":"false","confidence":0.95,"rationale":"Paris."}',
        [SearchHit(url="https://x.com", title="X", snippet="snip")],
    )

    resp = client.post("/claims", json={"claim": "The Eiffel Tower is in Berlin."})

    assert resp.status_code == 200
    body = resp.json()
    assert "claim_id" in body
    assert body["events_url"] == f"/claims/{body['claim_id']}/events"


def test_sse_stream_emits_verdict_event() -> None:
    # The end-to-end SSE path: POST a claim, then GET the events URL, and parse
    # the SSE stream. The final event is the resolved claim state (verdict).
    client = _client_with_fakes(
        '{"label":"false","confidence":0.95,"rationale":"Eiffel Tower is in Paris."}',
        [
            SearchHit(
                url="https://britannica.com/eiffel-tower",
                title="Eiffel Tower",
                snippet="The Eiffel Tower is in Paris, France.",
            )
        ],
    )

    resp = client.post("/claims", json={"claim": "The Eiffel Tower is in Berlin."})
    claim_id = resp.json()["claim_id"]

    with client.stream(
        "GET",
        f"/claims/{claim_id}/events",
        params={"claim": "The Eiffel Tower is in Berlin."},
    ) as stream:
        sse_lines = list(stream.iter_lines())

    # Parse SSE: lines starting with "event:" or "data:".
    events: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in sse_lines:
        if line.startswith("event:"):
            current = {"event": line[len("event:"):].strip()}
        elif line.startswith("data:"):
            current["data"] = line[len("data:"):].strip()
            events.append(current)
            current = {}

    assert len(events) >= 1
    # The final event is the verdict (claim state).
    verdict_event = events[-1]
    assert verdict_event["event"] == "verdict"
    payload = json.loads(verdict_event["data"])
    assert payload["verdict"]["label"] == "false"
    assert payload["verdict"]["confidence"] == 0.95
    assert payload["status"] == "fast_checked"


def test_health_endpoint() -> None:
    client = _client_with_fakes("{}", [])
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_post_claims_rejects_empty_body() -> None:
    client = _client_with_fakes("{}", [])
    resp = client.post("/claims", json={"claim": ""})
    # Pydantic min_length=1 -> 422.
    assert resp.status_code == 422


def test_events_returns_404_for_unknown_claim() -> None:
    client = _client_with_fakes("{}", [])
    resp = client.get("/claims/nonexistent/events", params={"claim": "x"})
    assert resp.status_code == 404
