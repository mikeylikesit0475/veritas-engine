"""Full graph traversal test — ARCH §9.

> Graph | Stub model → assert route taken and node visit order | yes

This is the test that proves the full 13-node graph (ARCH §3) runs the right
path through the right nodes. Two paths are tested:

1. **Fast path**: a simple claim routes fast → intake → triage → fast_check → report.
2. **Adversarial path**: a compound claim routes adversarial → intake → triage →
   decompose → fan_out → arbiter → confidence_route → report.

All with stub models — no live calls (CLAUDE.md #6).
"""

from __future__ import annotations

from typing import Any

import pytest
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from veritas.backends_registry import clear_backends, register_backends
from veritas.graph.arbiter import ARBITER_MODEL_STATE_KEY
from veritas.graph.decompose import DECOMPOSE_MODEL_STATE_KEY
from veritas.graph.fan_out import FANOUT_MODEL_STATE_KEY
from veritas.graph.fast_check import FAST_CHECK_MODEL_STATE_KEY
from veritas.graph.triage import TRIAGE_MODEL_STATE_KEY
from veritas.tools.backends import FakeSearchBackend, SearchHit
from veritas.tools.grounded_search import SEARCH_BACKEND_STATE_KEY

pytestmark = pytest.mark.asyncio


class _RouteStubLlm(BaseLlm):
    """A stub that returns different JSON based on the prompt content.

    This lets one stub serve all structured-output nodes (triage, decompose,
    fast_check, arbiter) by inspecting the prompt.
    """

    model: str = "route-stub"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(model="route-stub", **kwargs)

    @classmethod
    def supported_models(cls) -> list[str]:
        return [r"route-stub.*"]

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> Any:
        prompt = ""
        for content in llm_request.contents:
            if content and content.parts:
                for part in content.parts:
                    if part.text and not part.thought:
                        prompt += part.text

        # Triage: return the route. Check the CLAIM text, not the instruction,
        # because the instruction contains words like "simple" and "compound".
        if "triage classifier" in prompt.lower():
            # Extract the CLAIM from the prompt.
            claim_start = prompt.find("CLAIM: ")
            claim_text = prompt[claim_start + 7:] if claim_start != -1 else ""
            if "paris" in claim_text.lower() or "france" in claim_text.lower():
                text = '{"path": "fast", "reason": "simple claim"}'
            else:
                text = '{"path": "adversarial", "reason": "compound claim"}'
        # Decompose
        elif "atomic, independently checkable" in prompt.lower():
            text = '{"sub_claims": ["sub-claim one", "sub-claim two", "sub-claim three"]}'
        # Arbiter (check before fast_check — the arbiter prompt contains
        # "fact-checkers" which would match the fast_check check below)
        elif "arbiter" in prompt.lower():
            text = (
                '{"sub_claim_verdicts": ['
                '{"sub_claim_id":"sc0","label":"false","confidence":0.9,"rationale":"refuted"},'
                '{"sub_claim_id":"sc1","label":"false","confidence":0.9,"rationale":"refuted"},'
                '{"sub_claim_id":"sc2","label":"false","confidence":0.9,"rationale":"refuted"}'
                '],'
                '"overall_verdict": {"label":"false","confidence":0.92,"rationale":"all false"}}'
            )
        # fast_check
        elif "fact-checker" in prompt.lower():
            text = '{"label": "false", "confidence": 0.95, "rationale": "refuted"}'
        else:
            text = "{}"

        content = genai_types.Content(
            role="model",
            parts=[genai_types.Part.from_text(text=text)],
        )
        yield LlmResponse(content=content, turn_complete=True)


def _register_stubs(
    search_hits: list[SearchHit] | None = None,
) -> dict[str, Any]:
    """Register all stub backends needed for the full graph."""
    return {
        SEARCH_BACKEND_STATE_KEY: FakeSearchBackend(search_hits or []),
        TRIAGE_MODEL_STATE_KEY: _RouteStubLlm(),
        FAST_CHECK_MODEL_STATE_KEY: _RouteStubLlm(),
        DECOMPOSE_MODEL_STATE_KEY: _RouteStubLlm(),
        FANOUT_MODEL_STATE_KEY: _RouteStubLlm(),
        ARBITER_MODEL_STATE_KEY: _RouteStubLlm(),
    }


async def test_fast_path_visits_intake_triage_fast_check_report() -> None:
    """A simple claim routes fast: intake → triage → fast_check → report."""
    register_backends("veritas", **_register_stubs())
    try:
        from veritas.graph.wiring import build_graph

        graph = build_graph()
        session_service = InMemorySessionService()  # type: ignore[no-untyped-call]
        session = await session_service.create_session(
            app_name="veritas", user_id="test"
        )
        runner = Runner(
            app_name="veritas",
            agent=graph,  # type: ignore[arg-type]
            session_service=session_service,
        )

        message = genai_types.Content(
            role="user",
            parts=[genai_types.Part.from_text(text="Paris is in France.")],
        )

        nodes_seen: list[str] = []
        async for event in runner.run_async(
            user_id="test", session_id=session.id, new_message=message
        ):
            name = event.node_name or event.author
            if name:
                nodes_seen.append(name)

        # The fast path visits these nodes in order.
        assert "intake" in nodes_seen
        assert "triage" in nodes_seen
        assert "fast_check" in nodes_seen
        assert "report" in nodes_seen
        assert nodes_seen.index("intake") < nodes_seen.index("triage")
        assert nodes_seen.index("triage") < nodes_seen.index("fast_check")
        # decompose and fan_out should NOT be visited on the fast path.
        assert "decompose" not in nodes_seen
        assert "fan_out" not in nodes_seen
    finally:
        clear_backends("veritas")


async def test_adversarial_path_visits_full_swarm() -> None:
    """A compound claim routes adversarial: the full fan-out + arbiter path."""
    hits = [
        SearchHit(
            url="https://example.com/evidence",
            title="Evidence",
            snippet="Some evidence here.",
        )
    ]
    register_backends("veritas", **_register_stubs(hits))
    try:
        from veritas.graph.wiring import build_graph

        graph = build_graph()
        session_service = InMemorySessionService()  # type: ignore[no-untyped-call]
        session = await session_service.create_session(
            app_name="veritas", user_id="test"
        )
        runner = Runner(
            app_name="veritas",
            agent=graph,  # type: ignore[arg-type]
            session_service=session_service,
        )

        # A compound claim that triggers the adversarial path.
        message = genai_types.Content(
            role="user",
            parts=[
                genai_types.Part.from_text(
                    text=(
                        "The Eiffel Tower is in Berlin and was built by the "
                        "British in 1920."
                    )
                )
            ],
        )

        nodes_seen: list[str] = []
        async for event in runner.run_async(
            user_id="test", session_id=session.id, new_message=message
        ):
            name = event.node_name or event.author
            if name:
                nodes_seen.append(name)

        # The adversarial path visits all swarm nodes.
        assert "intake" in nodes_seen
        assert "triage" in nodes_seen
        assert "decompose" in nodes_seen
        assert "fan_out" in nodes_seen
        assert "arbiter" in nodes_seen
        assert "confidence_route" in nodes_seen
        assert "report" in nodes_seen
        # Node order: intake before triage, decompose before fan_out, etc.
        assert nodes_seen.index("intake") < nodes_seen.index("triage")
        assert nodes_seen.index("triage") < nodes_seen.index("decompose")
        assert nodes_seen.index("decompose") < nodes_seen.index("fan_out")
        assert nodes_seen.index("fan_out") < nodes_seen.index("arbiter")
        assert nodes_seen.index("arbiter") < nodes_seen.index("confidence_route")
    finally:
        clear_backends("veritas")
