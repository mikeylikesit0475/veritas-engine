"""HITL pause + adjudication — ARCH §3 node 12, Sprint 2.4.

> Kill mid-pause, restart, adjudicate, graph resumes.
> — BACKLOG Sprint 2.4 done-when

The HITL node uses ADK's ``request_input`` mechanism (verified in the Sprint 0.3
spike). The graph emits a ``adk_request_input`` function call event, which
suspends execution. The session (including the pending event) is persisted to
Postgres via ``DatabaseSessionService``. When the human adjudicates via
``POST /claims/{id}/adjudicate``, the server resumes the runner with a
``FunctionResponse`` carrying the adjudication, and the graph continues to the
report node.

This is the item that makes "durably suspends and resumes" a true claim. The demo:
kill the process mid-pause, restart, and the graph resumes where it was.
"""

from __future__ import annotations

from typing import Any

from google.adk.agents.context import Context
from google.genai import types as genai_types
from pydantic import BaseModel, Field

from veritas.graph.claim_state import (
    CLAIM_STATE_KEY,
    ClaimState,
    ClaimStatus,
)

__all__ = [
    "HITL_INTERRUPT_ID",
    "AdjudicationRequest",
    "AdjudicationResponse",
    "hitl_pause",
]

HITL_INTERRUPT_ID = "veritas_hitl"


class AdjudicationRequest(BaseModel):
    """The body of ``POST /claims/{id}/adjudicate``."""

    verdict: str = Field(..., description="The human's verdict: true, false, or mixed.")
    rationale: str = Field("", description="Why the human ruled this way.")
    confidence: float = Field(0.8, ge=0.0, le=1.0, description="The human's confidence.")


class AdjudicationResponse(BaseModel):
    """The response from adjudication."""

    claim_id: str
    status: str
    message: str


async def hitl_pause(*, ctx: Context) -> dict[str, Any]:
    """Pause the graph for human adjudication.

    This node emits a ``request_input`` event (via ADK's HITL mechanism). The
    runner suspends, the session is persisted, and the server waits for
    ``POST /claims/{id}/adjudicate``. On resume, the adjudication is in
    ``ctx.state`` under ``veritas.hitl.adjudication``.
    """
    raw_state: Any = ctx.state
    state_dict: dict[str, object] = (
        raw_state.to_dict() if hasattr(raw_state, "to_dict") else dict(raw_state)
    )
    claim_state = ClaimState.from_state(state_dict)
    claim_state.status = ClaimStatus.HITL_PAUSED
    ctx.state[CLAIM_STATE_KEY] = claim_state

    # The HITL pause is emitted by yielding a request_input event. ADK's runner
    # handles this by suspending execution. The resume comes from the server
    # calling run_async again with a FunctionResponse for this interrupt id.
    #
    # In a FunctionNode, yielding a dict with the interrupt info is the signal.
    # The server's adjudication endpoint constructs the FunctionResponse.
    return {
        "interrupt_id": HITL_INTERRUPT_ID,
        "message": "Human adjudication required. The claim's confidence is below "
        "threshold after maximum iterations. Please provide a verdict.",
        "claim": claim_state.normalised_text or claim_state.raw_text,
    }


def build_adjudication_function_response(
    adjudication: AdjudicationRequest,
) -> genai_types.Content:
    """Build the FunctionResponse content that resumes the graph after HITL.

    The server calls ``runner.run_async`` with this as the ``new_message`` to
    resume the paused graph. The response carries the human's verdict.
    """
    return genai_types.Content(
        role="user",
        parts=[
            genai_types.Part.from_function_response(
                name="adk_request_input",
                response={
                    "interrupt_id": HITL_INTERRUPT_ID,
                    "verdict": adjudication.verdict,
                    "rationale": adjudication.rationale,
                    "confidence": adjudication.confidence,
                },
            )
        ],
    )
