"""``triage`` node — ARCH §3 node 2.

The cost router. Decides whether a claim takes the expensive adversarial path
(decompose → fan-out → arbiter) or the cheap path (fast_check). Cheap, stateless,
heavily evaluated (Sprint 3).

Sprint 2.7: a simplified version. Sprint 3 (item 3.4) adds the full cost router
with its own eval (router precision). For now, the triage node asks the judgement
tier to classify the claim and sets the route accordingly. ADR-005: the route
itself is a Python string, not a model decision — the model classifies, Python
routes.
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
    "TRIAGE_MODEL_STATE_KEY",
    "TriagePath",
    "TriageResult",
    "triage",
]

TriagePath = str  # "fast" | "adversarial"

TRIAGE_MODEL_STATE_KEY = "veritas.triage.model"

_INSTRUCTION = (
    "You are a triage classifier for a fact-checking system. Given a claim, decide "
    "whether it needs the expensive adversarial path (multiple agents, parallel "
    "search, arbitration) or the cheap path (one grounded search + verdict).\n\n"
    "Route to \"adversarial\" if:\n"
    "- The claim is compound (multiple assertions)\n"
    "- The claim is contested or controversial\n"
    "- The claim involves numbers that should be computed\n"
    "- The claim has nuanced evidence on both sides\n\n"
    "Route to \"fast\" if:\n"
    "- The claim is a single simple factual assertion\n"
    "- The claim is obviously true or obviously false\n"
    "- The claim can be verified with one search\n\n"
    'Output a JSON object: {"path": "fast" or "adversarial", "reason": "one sentence"}. '
    "Output only the JSON object."
)


class TriageResult(BaseModel):
    """The triage classification."""

    path: str = Field(..., pattern="^(fast|adversarial)$")
    reason: str = ""


def _get_model(ctx: Context) -> BaseLlm:
    model = ctx.state.get(TRIAGE_MODEL_STATE_KEY)
    if model is not None:
        return model  # type: ignore[no-any-return]
    app_name = getattr(ctx, "app_name", None) or "veritas"
    try:
        return get_backend(app_name, TRIAGE_MODEL_STATE_KEY)  # type: ignore[no-any-return]
    except RuntimeError:
        return model_for(Tier.JUDGEMENT)


async def _structured_triage(model: BaseLlm, claim: str) -> TriageResult:
    contents = [
        genai_types.Content(
            role="user",
            parts=[
                genai_types.Part.from_text(
                    text=f"{_INSTRUCTION}\n\nCLAIM: {claim}\n\nJSON:"
                )
            ],
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
        raise ValueError("triage: model returned empty content — expected JSON.")

    try:
        return TriageResult.model_validate_json(strip_markdown_fences(raw_text))
    except ValidationError as e:
        raise ValueError(
            f"triage: model did not return schema-valid JSON. "
            f"Raw: {raw_text[:200]!r}. Error: {e}"
        ) from e


async def triage(*, ctx: Context) -> TriageResult:
    """Classify the claim and set the route. Writes route to ``ctx.route``."""
    raw_state: Any = ctx.state
    state_dict: dict[str, object] = (
        raw_state.to_dict() if hasattr(raw_state, "to_dict") else dict(raw_state)
    )
    claim_state = ClaimState.from_state(state_dict)
    claim_text = claim_state.normalised_text or claim_state.raw_text
    if not claim_text:
        raise RuntimeError("triage: no claim text in state")

    model = _get_model(ctx)
    result = await _structured_triage(model, claim_text)

    # ADR-005: the route is a Python string, set by Python, not by the model.
    # The model classifies; Python routes. Fail toward rigour (ARCH §3: default
    # → adversarial).
    path = result.path if result.path in ("fast", "adversarial") else "adversarial"

    # Set the route on the context — ADK's edge matching follows this.
    ctx.route = path

    # Update claim state.
    claim_state.status = ClaimStatus.TRIAGED
    ctx.state[CLAIM_STATE_KEY] = claim_state.model_dump(mode="json")
    ctx.state["veritas.triage.result"] = result.model_dump(mode="json")

    return result
