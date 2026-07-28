"""``decompose`` node — ARCH §3, node 5.

Splits a compound claim into 3-5 atomic, independently checkable sub-claims.
Judgement tier (structured output). ADR-004: schema failures fail loudly.

The sub-claims are stored in ``ClaimState.sub_claims`` (ARCH §7: state is a
projection of the event stream). Each sub-claim is a plain string — the fan-out
agents (Sprint 1.2) take it from there.

Why not let the model decide how many sub-claims? Because control flow stays in
Python (ADR-005): the fan-out concurrency cap, the iteration bound, and the
arbitration schema all depend on the count being bounded. The model proposes;
Python validates the bounds.
"""

from __future__ import annotations

from typing import Any

from google.adk.agents.context import Context
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.genai import types as genai_types
from pydantic import BaseModel, Field, ValidationError

from veritas.backends_registry import get_backend
from veritas.graph.claim_state import (
    CLAIM_STATE_KEY,
    ClaimState,
    ClaimStatus,
)
from veritas.models import Tier, model_for
from veritas.utils import strip_markdown_fences

__all__ = [
    "DECOMPOSE_MODEL_STATE_KEY",
    "MAX_SUB_CLAIMS",
    "MIN_SUB_CLAIMS",
    "DecomposeResult",
    "SubClaim",
    "decompose",
]

MIN_SUB_CLAIMS = 3
MAX_SUB_CLAIMS = 5

DECOMPOSE_MODEL_STATE_KEY = "veritas.decompose.model"

_INSTRUCTION = (
    "You are a fact-checking analyst. Break the given claim into between "
    f"{MIN_SUB_CLAIMS} and {MAX_SUB_CLAIMS} atomic, independently checkable "
    "sub-claims. Each sub-claim should be a simple factual statement that can be "
    "verified on its own — no compound assertions, no hedging. If the claim is "
    'already atomic, still produce at least 3 facets (e.g. "the entity exists", '
    '"the attribute holds", "the temporal context is correct"). '
    'Output a JSON object with a single key "sub_claims" mapping to an array of '
    "strings. Output only the JSON object — no prose, no markdown fences."
)


class SubClaim(BaseModel):
    """One atomic sub-claim. A plain string with a stable id for fan-out keying."""

    id: str = Field(..., description="Stable identifier, e.g. 'sc0', 'sc1'.")
    text: str = Field(..., min_length=1, max_length=500)


class DecomposeResult(BaseModel):
    """What ``decompose`` yields into the event stream."""

    sub_claims: list[SubClaim] = Field(..., min_length=MIN_SUB_CLAIMS, max_length=MAX_SUB_CLAIMS)


def _get_model(ctx: Context) -> BaseLlm:
    model = ctx.state.get(DECOMPOSE_MODEL_STATE_KEY)
    if model is not None:
        return model  # type: ignore[no-any-return]
    app_name = getattr(ctx, "app_name", None) or "veritas"
    try:
        return get_backend(app_name, DECOMPOSE_MODEL_STATE_KEY)  # type: ignore[no-any-return]
    except RuntimeError:
        return model_for(Tier.JUDGEMENT)


def _build_prompt(claim: str) -> str:
    return f"{_INSTRUCTION}\n\nCLAIM: {claim}\n\nJSON:"


async def _structured_decompose(model: BaseLlm, claim: str) -> DecomposeResult:
    """Ask the judgement tier for structured sub-claims; validate loudly."""
    contents = [
        genai_types.Content(
            role="user",
            parts=[genai_types.Part.from_text(text=_build_prompt(claim))],
        )
    ]
    request = LlmRequest(
        model=model.model,
        contents=contents,
        config=genai_types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )

    raw_text = ""
    async for response in model.generate_content_async(request, stream=False):
        if response.content and response.content.parts:
            for part in response.content.parts:
                if part.text and not part.thought:
                    raw_text += part.text

    raw_text = raw_text.strip()
    if not raw_text:
        raise ValueError(
            "decompose: model returned empty content — expected JSON sub-claims."
        )

    # The model returns {"sub_claims": ["...", "..."]}. Parse the inner list,
    # assign stable ids, and validate bounds.
    try:
        import json

        parsed = json.loads(strip_markdown_fences(raw_text))
        if not isinstance(parsed, dict) or "sub_claims" not in parsed:
            raise ValueError("missing 'sub_claims' key")
        raw_list = parsed["sub_claims"]
        if not isinstance(raw_list, list):
            raise ValueError("'sub_claims' is not a list")
        sub_claims = [
            SubClaim(id=f"sc{i}", text=str(text)[:500])
            for i, text in enumerate(raw_list)
        ]
        return DecomposeResult(sub_claims=sub_claims)
    except (json.JSONDecodeError, ValueError, ValidationError) as e:
        raise ValueError(
            f"decompose: model did not return schema-valid JSON sub-claims. "
            f"Raw: {raw_text[:200]!r}. Error: {e}"
        ) from e


async def decompose(*, ctx: Context) -> DecomposeResult:
    """Split the claim into atomic sub-claims. Writes back into ``ctx.state``."""
    raw_state: Any = ctx.state
    state_dict: dict[str, object] = (
        raw_state.to_dict() if hasattr(raw_state, "to_dict") else dict(raw_state)
    )
    claim_state = ClaimState.from_state(state_dict)
    claim_text = claim_state.normalised_text or claim_state.raw_text
    if not claim_text:
        raise RuntimeError("decompose: no claim text in state (intake did not run?)")

    model = _get_model(ctx)
    result = await _structured_decompose(model, claim_text)

    # Persist sub-claims into claim state (ARCH §7).
    claim_state.sub_claims = [sc.text for sc in result.sub_claims]
    claim_state.status = ClaimStatus.DECOMPOSED
    ctx.state[CLAIM_STATE_KEY] = claim_state

    # Also store the full SubClaim objects (with ids) under a separate key for
    # the fan-out node to pick up. The plain-text list in ClaimState is the
    # projection; the keyed objects are the working state.
    ctx.state["veritas.decompose.result"] = result.model_dump(mode="json")

    return result
