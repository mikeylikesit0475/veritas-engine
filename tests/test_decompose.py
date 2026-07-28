"""Tests for the ``decompose`` node (ARCH §3, node 5).

Uses a stub model returning canned JSON. No live calls (CLAUDE.md #6).
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
from veritas.graph.claim_state import (
    CLAIM_STATE_KEY,
    ClaimState,
    ClaimStatus,
)
from veritas.graph.decompose import (
    DECOMPOSE_MODEL_STATE_KEY,
    MAX_SUB_CLAIMS,
    MIN_SUB_CLAIMS,
    DecomposeResult,
    decompose,
)

pytestmark = pytest.mark.asyncio


class _DecomposeStubLlm(BaseLlm):
    """Stub returning a fixed JSON sub-claims response."""

    model: str = "decompose-stub"

    def __init__(self, json_response: str, **kwargs: Any) -> None:
        super().__init__(model="decompose-stub", **kwargs)
        self._json = json_response

    @classmethod
    def supported_models(cls) -> list[str]:
        return [r"decompose-stub.*"]

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> Any:
        content = genai_types.Content(
            role="model",
            parts=[genai_types.Part.from_text(text=self._json)],
        )
        yield LlmResponse(content=content, turn_complete=True)


def _ctx_with_claim(claim: str = "The Eiffel Tower is in Berlin.") -> FakeToolContext:
    return FakeToolContext(
        {CLAIM_STATE_KEY: ClaimState(normalised_text=claim).model_dump(mode="json")}
    )


async def test_decompose_produces_schema_valid_sub_claims() -> None:
    # The happy path: model returns valid JSON, we get 3-5 SubClaim objects with
    # stable ids, and the result is schema-validated.
    register_backends(
        "veritas",
        **{DECOMPOSE_MODEL_STATE_KEY: _DecomposeStubLlm(
            '{"sub_claims": ["The Eiffel Tower exists", '
            '"The Eiffel Tower is located in Berlin", '
            '"Berlin is the capital of France"]}'),
        },
    )
    ctx = _ctx_with_claim()
    try:
        result = await decompose(ctx=ctx)
    finally:
        clear_backends("veritas")

    assert isinstance(result, DecomposeResult)
    assert MIN_SUB_CLAIMS <= len(result.sub_claims) <= MAX_SUB_CLAIMS
    assert result.sub_claims[0].id == "sc0"
    assert result.sub_claims[1].id == "sc1"
    assert all(sc.text for sc in result.sub_claims)


async def test_decompose_writes_sub_claims_to_state() -> None:
    # ARCH §7: state is a projection. sub_claims land in ClaimState.
    register_backends(
        "veritas",
        **{DECOMPOSE_MODEL_STATE_KEY: _DecomposeStubLlm(
            '{"sub_claims": ["a", "b", "c"]}'),
        },
    )
    ctx = _ctx_with_claim()
    try:
        await decompose(ctx=ctx)
    finally:
        clear_backends("veritas")

    claim_state = ClaimState.from_state(ctx.state)
    assert claim_state.status is ClaimStatus.DECOMPOSED
    assert claim_state.sub_claims == ["a", "b", "c"]


async def test_decompose_rejects_too_few_sub_claims() -> None:
    # ADR-005: control flow stays in Python. The 3-5 bound is enforced here, not
    # by trusting the model. A 2-sub-claim response is a schema failure.
    register_backends(
        "veritas",
        **{DECOMPOSE_MODEL_STATE_KEY: _DecomposeStubLlm(
            '{"sub_claims": ["only one", "only two"]}'),
        },
    )
    ctx = _ctx_with_claim()
    try:
        with pytest.raises(ValueError, match="schema-valid JSON sub-claims"):
            await decompose(ctx=ctx)
    finally:
        clear_backends("veritas")


async def test_decompose_rejects_too_many_sub_claims() -> None:
    # A 6-sub-claim response is also out of bounds.
    register_backends(
        "veritas",
        **{DECOMPOSE_MODEL_STATE_KEY: _DecomposeStubLlm(
            '{"sub_claims": ["a", "b", "c", "d", "e", "f"]}'),
        },
    )
    ctx = _ctx_with_claim()
    try:
        with pytest.raises(ValueError, match="schema-valid JSON sub-claims"):
            await decompose(ctx=ctx)
    finally:
        clear_backends("veritas")


async def test_decompose_raises_on_empty_model_output() -> None:
    register_backends(
        "veritas", **{DECOMPOSE_MODEL_STATE_KEY: _DecomposeStubLlm("")}
    )
    ctx = _ctx_with_claim()
    try:
        with pytest.raises(ValueError, match="empty content"):
            await decompose(ctx=ctx)
    finally:
        clear_backends("veritas")


async def test_decompose_raises_on_non_json_output() -> None:
    register_backends(
        "veritas",
        **{DECOMPOSE_MODEL_STATE_KEY: _DecomposeStubLlm(
            "I think the sub-claims are: 1) it exists, 2) it's in Berlin")},
    )
    ctx = _ctx_with_claim()
    try:
        with pytest.raises(ValueError, match="schema-valid JSON sub-claims"):
            await decompose(ctx=ctx)
    finally:
        clear_backends("veritas")


async def test_decompose_raises_on_missing_sub_claims_key() -> None:
    register_backends(
        "veritas",
        **{DECOMPOSE_MODEL_STATE_KEY: _DecomposeStubLlm('{"items": ["a", "b", "c"]}')},
    )
    ctx = _ctx_with_claim()
    try:
        with pytest.raises(ValueError, match="schema-valid JSON sub-claims"):
            await decompose(ctx=ctx)
    finally:
        clear_backends("veritas")


async def test_decompose_raises_without_claim_in_state() -> None:
    register_backends(
        "veritas",
        **{DECOMPOSE_MODEL_STATE_KEY: _DecomposeStubLlm('{"sub_claims": ["a", "b", "c"]}')},
    )
    ctx = FakeToolContext({})
    try:
        with pytest.raises(RuntimeError, match="no claim text in state"):
            await decompose(ctx=ctx)
    finally:
        clear_backends("veritas")
