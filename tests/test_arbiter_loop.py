"""Tests for the arbiter, confidence route, and loop (ARCH §3 nodes 9-11).

ADR-005: control flow stays in Python. The confidence route and loop bounds are
tested here as pure-Python logic, not via a model. The arbiter's structured output
is tested with a stub model returning canned JSON.
"""

from __future__ import annotations

from typing import Any

import pytest
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types as genai_types

from tests.fakes import FakeToolContext
from veritas.backends_registry import clear_backends, register_backends
from veritas.graph.arbiter import (
    ARBITER_MODEL_STATE_KEY,
    CONFIDENCE_THRESHOLD,
    ArbiterResult,
    arbiter,
)
from veritas.graph.claim_state import (
    CLAIM_STATE_KEY,
    ClaimState,
    ClaimStatus,
    VerdictLabel,
)
from veritas.graph.fan_out import FANOUT_RESULT_STATE_KEY, AgentFinding, FanOutResult
from veritas.graph.loop import (
    LOOP_COUNT_STATE_KEY,
    MAX_ITERATIONS,
    confidence_route,
    loop,
)

pytestmark = pytest.mark.asyncio


class _ArbiterStubLlm(BaseLlm):
    """Stub returning a fixed JSON arbiter verdict."""

    model: str = "arbiter-stub"

    def __init__(self, json_response: str, **kwargs: Any) -> None:
        super().__init__(model="arbiter-stub", **kwargs)
        self._json = json_response

    @classmethod
    def supported_models(cls) -> list[str]:
        return [r"arbiter-stub.*"]

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> Any:
        content = genai_types.Content(
            role="model",
            parts=[genai_types.Part.from_text(text=self._json)],
        )
        yield LlmResponse(content=content, turn_complete=True)


_HIGH_CONF_JSON = (
    '{"sub_claim_verdicts": ['
    '{"sub_claim_id":"sc0","label":"true","confidence":0.9,"rationale":"confirmed"},'
    '{"sub_claim_id":"sc1","label":"false","confidence":0.95,"rationale":"refuted"}'
    '],'
    '"overall_verdict": {"label":"false","confidence":0.92,"rationale":"Eiffel Tower is in Paris"}}'
)

_LOW_CONF_JSON = (
    '{"sub_claim_verdicts": ['
    '{"sub_claim_id":"sc0","label":"mixed","confidence":0.4,"rationale":"split evidence"},'
    '{"sub_claim_id":"sc1","label":"mixed","confidence":0.5,"rationale":"sparse"}'
    '],'
    '"overall_verdict": {"label":"mixed","confidence":0.45,"rationale":"insufficient evidence"}}'
)


def _seed_post_fanout_state(claim: str = "The Eiffel Tower is in Berlin.") -> dict[str, Any]:
    """State after fan-out: claim + sub-claims + findings."""
    return {
        CLAIM_STATE_KEY: ClaimState(
            normalised_text=claim,
            status=ClaimStatus.FAN_OUT_DONE,
            sub_claims=["The Eiffel Tower exists", "The Eiffel Tower is in Berlin"],
        ).model_dump(mode="json"),
        "veritas.decompose.result": {
            "sub_claims": [
                {"id": "sc0", "text": "The Eiffel Tower exists"},
                {"id": "sc1", "text": "The Eiffel Tower is in Berlin"},
            ]
        },
        FANOUT_RESULT_STATE_KEY: FanOutResult(
            findings=[
                AgentFinding(
                    sub_claim_id="sc0", agent_name="skeptic",
                    findings="No refutation found.", elapsed_s=0.1,
                ),
                AgentFinding(
                    sub_claim_id="sc0", agent_name="steelman",
                    findings="Wikipedia confirms it exists.", elapsed_s=0.1,
                ),
                AgentFinding(
                    sub_claim_id="sc1", agent_name="skeptic",
                    findings="Eiffel Tower is in Paris, not Berlin.", elapsed_s=0.1,
                ),
                AgentFinding(
                    sub_claim_id="sc1", agent_name="steelman",
                    findings="No evidence it's in Berlin.", elapsed_s=0.1,
                ),
            ]
        ).model_dump(mode="json"),
    }


# --- Arbiter tests ---


async def test_arbiter_produces_schema_valid_verdict() -> None:
    register_backends("veritas", **{ARBITER_MODEL_STATE_KEY: _ArbiterStubLlm(_HIGH_CONF_JSON)})
    ctx = FakeToolContext(_seed_post_fanout_state())
    try:
        result = await arbiter(ctx=ctx)
    finally:
        clear_backends("veritas")

    assert isinstance(result, ArbiterResult)
    assert len(result.sub_claim_verdicts) == 2
    assert result.overall_verdict.label is VerdictLabel.FALSE
    assert result.overall_verdict.confidence == 0.92


async def test_arbiter_writes_verdict_to_state() -> None:
    register_backends("veritas", **{ARBITER_MODEL_STATE_KEY: _ArbiterStubLlm(_HIGH_CONF_JSON)})
    ctx = FakeToolContext(_seed_post_fanout_state())
    try:
        await arbiter(ctx=ctx)
    finally:
        clear_backends("veritas")

    claim_state = ClaimState.from_state(ctx.state)
    assert claim_state.status is ClaimStatus.ARBITERED
    assert claim_state.verdict.label is VerdictLabel.FALSE


async def test_arbiter_raises_on_empty_output() -> None:
    register_backends("veritas", **{ARBITER_MODEL_STATE_KEY: _ArbiterStubLlm("")})
    ctx = FakeToolContext(_seed_post_fanout_state())
    try:
        with pytest.raises(ValueError, match="empty content"):
            await arbiter(ctx=ctx)
    finally:
        clear_backends("veritas")


async def test_arbiter_raises_on_bad_json() -> None:
    register_backends(
        "veritas",
        **{ARBITER_MODEL_STATE_KEY: _ArbiterStubLlm("I think it's false.")},
    )
    ctx = FakeToolContext(_seed_post_fanout_state())
    try:
        with pytest.raises(ValueError, match="schema-valid JSON"):
            await arbiter(ctx=ctx)
    finally:
        clear_backends("veritas")


# --- Confidence route tests (ADR-005: pure Python) ---


async def test_confidence_route_reports_when_confidence_high() -> None:
    register_backends("veritas", **{ARBITER_MODEL_STATE_KEY: _ArbiterStubLlm(_HIGH_CONF_JSON)})
    ctx = FakeToolContext(_seed_post_fanout_state())
    try:
        await arbiter(ctx=ctx)
        route = confidence_route(ctx=ctx)
    finally:
        clear_backends("veritas")

    assert route == "report"


async def test_confidence_route_loops_when_confidence_low_and_iterations_remain() -> None:
    register_backends("veritas", **{ARBITER_MODEL_STATE_KEY: _ArbiterStubLlm(_LOW_CONF_JSON)})
    ctx = FakeToolContext(_seed_post_fanout_state())
    try:
        await arbiter(ctx=ctx)
        route = confidence_route(ctx=ctx)
    finally:
        clear_backends("veritas")

    assert route == "loop"


async def test_confidence_route_hits_hitl_when_iterations_exhausted() -> None:
    register_backends("veritas", **{ARBITER_MODEL_STATE_KEY: _ArbiterStubLlm(_LOW_CONF_JSON)})
    ctx = FakeToolContext({**_seed_post_fanout_state(), LOOP_COUNT_STATE_KEY: MAX_ITERATIONS})
    try:
        await arbiter(ctx=ctx)
        route = confidence_route(ctx=ctx)
    finally:
        clear_backends("veritas")

    assert route == "hitl"


# --- Loop tests (ADR-005: pure Python, bounded) ---


async def test_loop_increments_iteration_and_writes_directive() -> None:
    register_backends("veritas", **{ARBITER_MODEL_STATE_KEY: _ArbiterStubLlm(_LOW_CONF_JSON)})
    ctx = FakeToolContext(_seed_post_fanout_state())
    try:
        await arbiter(ctx=ctx)
        directive = loop(ctx=ctx)
    finally:
        clear_backends("veritas")

    assert directive.iteration == 1
    assert len(directive.low_confidence_sub_claims) == 2  # both below threshold
    assert len(directive.directives) == 2
    assert ctx.state[LOOP_COUNT_STATE_KEY] == 1


async def test_loop_terminates_at_max_iterations() -> None:
    # The loop node itself doesn't check the bound — the confidence route does.
    # But we verify the bound constant is correct.
    assert MAX_ITERATIONS == 2

    register_backends("veritas", **{ARBITER_MODEL_STATE_KEY: _ArbiterStubLlm(_LOW_CONF_JSON)})
    ctx = FakeToolContext({**_seed_post_fanout_state(), LOOP_COUNT_STATE_KEY: 1})
    try:
        await arbiter(ctx=ctx)
        # After one loop, iteration is 1. Confidence route should still loop
        # (1 < MAX_ITERATIONS=2).
        route = confidence_route(ctx=ctx)
        assert route == "loop"

        # Do the loop.
        loop(ctx=ctx)
        assert ctx.state[LOOP_COUNT_STATE_KEY] == 2

        # Now confidence route should hit HITL (2 >= MAX_ITERATIONS).
        route = confidence_route(ctx=ctx)
        assert route == "hitl"
    finally:
        clear_backends("veritas")


async def test_confidence_threshold_is_70_percent() -> None:
    # ARCH §3: the threshold. Documented here as a test so a change is visible.
    assert CONFIDENCE_THRESHOLD == 0.7
