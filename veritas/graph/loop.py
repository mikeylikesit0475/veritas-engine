"""Confidence route + bounded loop — ARCH §3 nodes 10 & 11, ADR-005.

The confidence route decides: does the arbiter's confidence warrant a report,
or should the graph loop with new search directives, or pause for HITL?

    confidence >= tau  →  report
    confidence <  tau  and  iter < 2  →  loop
    else (low confidence, iterations exhausted)  →  hitl

All of this is in Python. The model supplies a confidence number; Python decides
what to do with it (ADR-005). The loop rewrite is also pure Python — it produces
new search directives from what the arbiter said was missing, not by asking the
model to "try again."

The iteration bound (max 2) is a constant here, not a prompt. A loop that doesn't
terminate is the classic failure mode of agent loops; making the bound a Python
constant means it can't drift.
"""

from __future__ import annotations

from typing import Any, Literal

from google.adk.agents.context import Context
from pydantic import BaseModel

from veritas.graph.arbiter import CONFIDENCE_THRESHOLD, ArbiterResult, SubClaimVerdict
from veritas.graph.claim_state import (
    CLAIM_STATE_KEY,
    ClaimState,
    ClaimStatus,
)

__all__ = [
    "LOOP_COUNT_STATE_KEY",
    "MAX_ITERATIONS",
    "ConfidenceRoute",
    "LoopDirective",
    "confidence_route",
    "loop",
]

# ARCH §3: bounded at 2 iterations. ADR-005: the bound is in Python.
MAX_ITERATIONS = 2

LOOP_COUNT_STATE_KEY = "veritas.loop.count"
LOOP_DIRECTIVE_STATE_KEY = "veritas.loop.directive"

ConfidenceRoute = Literal["report", "loop", "hitl"]


class LoopDirective(BaseModel):
    """What the loop node tells the fan-out agents to search for next iteration."""

    iteration: int
    low_confidence_sub_claims: list[str] = []  # sub_claim_ids below threshold
    directives: list[str] = []  # new search guidance, one per low-confidence sc


def confidence_route(*, ctx: Context) -> ConfidenceRoute:
    """Decide the next node from the arbiter's confidence. Pure Python (ADR-005)."""
    raw_state: Any = ctx.state
    state_dict: dict[str, object] = (
        raw_state.to_dict() if hasattr(raw_state, "to_dict") else dict(raw_state)
    )

    arbiter_data = state_dict.get("veritas.arbiter.result")
    if arbiter_data is None:
        raise RuntimeError("confidence_route: no arbiter result in state")

    result = ArbiterResult.model_validate(arbiter_data)
    confidence = result.overall_verdict.confidence
    iteration_raw: object = state_dict.get(LOOP_COUNT_STATE_KEY, 0)
    iteration: int = iteration_raw if isinstance(iteration_raw, int) else 0

    if confidence >= CONFIDENCE_THRESHOLD:
        return "report"
    if iteration < MAX_ITERATIONS:
        return "loop"
    return "hitl"


def loop(*, ctx: Context) -> LoopDirective:
    """Rewrite search directives from what the arbiter said was missing.

    Pure Python (ADR-005). The directives are derived from the sub-claim verdicts
    with low confidence — the arbiter's rationale tells us what was missing, and
    the loop node turns that into a new search directive.
    """
    raw_state: Any = ctx.state
    state_dict: dict[str, object] = (
        raw_state.to_dict() if hasattr(raw_state, "to_dict") else dict(raw_state)
    )

    arbiter_data = state_dict.get("veritas.arbiter.result")
    if arbiter_data is None:
        raise RuntimeError("loop: no arbiter result in state")
    result = ArbiterResult.model_validate(arbiter_data)

    iteration_raw: object = state_dict.get(LOOP_COUNT_STATE_KEY, 0)
    iteration: int = iteration_raw if isinstance(iteration_raw, int) else 0
    new_iteration = iteration + 1

    # Find sub-claims below the confidence threshold.
    low_conf_scs: list[SubClaimVerdict] = [
        scv
        for scv in result.sub_claim_verdicts
        if scv.confidence < CONFIDENCE_THRESHOLD
    ]

    # Build directives from the rationales. Each low-confidence sub-claim gets a
    # directive that tells the fan-out to search for what was missing.
    directives: list[str] = []
    for scv in low_conf_scs:
        # The rationale says what was missing; turn it into a search directive.
        text = (
            f"Sub-claim {scv.sub_claim_id} has low confidence ({scv.confidence:.2f}). "
            f"Arbiter rationale: {scv.rationale}. "
            f"Search for additional evidence to resolve this."
        )
        directives.append(text)

    result_directive = LoopDirective(
        iteration=new_iteration,
        low_confidence_sub_claims=[scv.sub_claim_id for scv in low_conf_scs],
        directives=directives,
    )

    # Persist the loop count and directive.
    ctx.state[LOOP_COUNT_STATE_KEY] = new_iteration
    ctx.state[LOOP_DIRECTIVE_STATE_KEY] = result_directive.model_dump(mode="json")

    # Update claim status.
    claim_state = ClaimState.from_state(state_dict)
    claim_state.status = ClaimStatus.LOOPED
    ctx.state[CLAIM_STATE_KEY] = claim_state

    return result_directive
