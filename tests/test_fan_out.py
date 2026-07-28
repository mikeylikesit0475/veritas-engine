"""Tests for the fan-out node — skeptic + steelman in parallel (ARCH §3, ADR-003).

Uses a stub model that returns canned prose (ADR-004: fan-out never emits structured
output) and a fake search backend. No live calls (CLAUDE.md #6).
"""

from __future__ import annotations

from typing import Any

import pytest
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types as genai_types

from tests.fakes import FakeToolContext
from veritas.agents.fanout import SKEPTIC_NAME, STEELMAN_NAME
from veritas.backends_registry import clear_backends, register_backends
from veritas.graph.claim_state import (
    CLAIM_STATE_KEY,
    ClaimState,
    ClaimStatus,
)
from veritas.graph.fan_out import (
    FANOUT_MODEL_STATE_KEY,
    FANOUT_RESULT_STATE_KEY,
    FanOutResult,
    fan_out,
)
from veritas.tools.backends import FakeSearchBackend
from veritas.tools.fetch_page import FETCH_BACKEND_STATE_KEY
from veritas.tools.grounded_search import SEARCH_BACKEND_STATE_KEY

pytestmark = pytest.mark.asyncio


class _FanoutStubLlm(BaseLlm):
    """Stub that returns canned prose findings. Different text per 'agent' is not
    possible here (the model doesn't know which agent it is), but the test asserts
    on structure, not content — the point is that prose comes back, not JSON."""

    model: str = "fanout-stub"

    def __init__(self, response: str = "Found some evidence. See https://example.com", **kwargs: Any) -> None:
        super().__init__(model="fanout-stub", **kwargs)
        self._response = response

    @classmethod
    def supported_models(cls) -> list[str]:
        return [r"fanout-stub.*"]

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> Any:
        content = genai_types.Content(
            role="model",
            parts=[genai_types.Part.from_text(text=self._response)],
        )
        yield LlmResponse(content=content, turn_complete=True)


def _seed_state(claim: str = "The Eiffel Tower is in Berlin.") -> dict[str, Any]:
    """State after decompose has run — claim + sub-claims in state."""
    return {
        CLAIM_STATE_KEY: ClaimState(
            normalised_text=claim,
            status=ClaimStatus.DECOMPOSED,
            sub_claims=["The Eiffel Tower exists", "The Eiffel Tower is in Berlin"],
        ).model_dump(mode="json"),
        "veritas.decompose.result": {
            "sub_claims": [
                {"id": "sc0", "text": "The Eiffel Tower exists"},
                {"id": "sc1", "text": "The Eiffel Tower is in Berlin"},
            ]
        },
    }


async def test_fan_out_runs_both_agents_on_all_sub_claims() -> None:
    # 2 sub-claims x 2 agents = 4 findings. Each has a sub_claim_id and agent_name.
    register_backends(
        "veritas",
        **{
            SEARCH_BACKEND_STATE_KEY: FakeSearchBackend([]),
            FETCH_BACKEND_STATE_KEY: None,  # not used in this test
            FANOUT_MODEL_STATE_KEY: _FanoutStubLlm(),
        },
    )
    ctx = FakeToolContext(_seed_state())
    try:
        result = await fan_out(ctx=ctx)
    finally:
        clear_backends("veritas")

    assert isinstance(result, FanOutResult)
    assert len(result.findings) == 4  # 2 sub-claims x 2 agents

    # Both agent names are present.
    agent_names = {f.agent_name for f in result.findings}
    assert agent_names == {SKEPTIC_NAME, STEELMAN_NAME}

    # Both sub-claim ids are present.
    sc_ids = {f.sub_claim_id for f in result.findings}
    assert sc_ids == {"sc0", "sc1"}

    # Each finding has non-empty prose (ADR-004: prose, not JSON).
    for f in result.findings:
        assert f.findings and isinstance(f.findings, str)
        assert f.elapsed_s >= 0.0


async def test_fan_out_writes_result_to_state() -> None:
    register_backends(
        "veritas",
        **{
            SEARCH_BACKEND_STATE_KEY: FakeSearchBackend([]),
            FETCH_BACKEND_STATE_KEY: None,
            FANOUT_MODEL_STATE_KEY: _FanoutStubLlm(),
        },
    )
    ctx = FakeToolContext(_seed_state())
    try:
        await fan_out(ctx=ctx)
    finally:
        clear_backends("veritas")

    # Result persisted for the arbiter.
    assert FANOUT_RESULT_STATE_KEY in ctx.state
    # Claim status updated.
    claim_state = ClaimState.from_state(ctx.state)
    assert claim_state.status is ClaimStatus.FAN_OUT_DONE


async def test_fan_out_logs_timings() -> None:
    # ARCH §1: "timings logged". Each finding has an elapsed_s.
    register_backends(
        "veritas",
        **{
            SEARCH_BACKEND_STATE_KEY: FakeSearchBackend([]),
            FETCH_BACKEND_STATE_KEY: None,
            FANOUT_MODEL_STATE_KEY: _FanoutStubLlm(),
        },
    )
    ctx = FakeToolContext(_seed_state())
    try:
        result = await fan_out(ctx=ctx)
    finally:
        clear_backends("veritas")

    assert result.total_elapsed_s > 0.0
    for f in result.findings:
        assert f.elapsed_s >= 0.0


async def test_fan_out_raises_without_decompose_result() -> None:
    # If decompose didn't run, fan_out fails loudly (ADR-004 spirit).
    register_backends(
        "veritas",
        **{
            SEARCH_BACKEND_STATE_KEY: FakeSearchBackend([]),
            FANOUT_MODEL_STATE_KEY: _FanoutStubLlm(),
        },
    )
    ctx = FakeToolContext({CLAIM_STATE_KEY: ClaimState(normalised_text="x").model_dump(mode="json")})
    try:
        with pytest.raises(RuntimeError, match="no decompose result"):
            await fan_out(ctx=ctx)
    finally:
        clear_backends("veritas")
