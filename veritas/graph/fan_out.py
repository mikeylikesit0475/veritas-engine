"""Fan-out node — ARCH §3 nodes 6-8, Sprint 1.2 + 1.3.

For each sub-claim from ``decompose``, runs the skeptic and steelman agents in
parallel. Each agent gets its own isolated session (ADR-003: contexts are
isolated). Concurrency is capped so an 8-agent burst doesn't overwhelm the model
backend. Timings are logged to the event stream for the demo's credibility beat
("those are the actual queries, and that's how long they took").

ADR-004: fan-out agents return **prose**, never structured output. The findings
are stored as plain strings keyed by ``sub_claim_id / agent_name``. The arbiter
(Sprint 2) structures them.

Design notes:

- Each agent runs via its own ``Runner`` + ``InMemorySessionService``. This is the
  isolation mechanism — one agent's search history and findings are invisible to
  the other. The cost is N sessions per claim; acceptable for a PoC.
- The concurrency cap uses ``asyncio.Semaphore``. Pure Python (ADR-005).
- The fan-out model tier is looked up from the registry (tests inject stubs;
  the server injects the real Ollama/Gemini model).
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any

from google.adk.agents import Agent
from google.adk.agents.context import Context
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types
from pydantic import BaseModel, Field

from veritas.agents.fanout import skeptic_agent, steelman_agent
from veritas.backends_registry import get_backend
from veritas.graph.claim_state import (
    CLAIM_STATE_KEY,
    ClaimState,
    ClaimStatus,
)
from veritas.tools.grounded_search import SEARCH_BACKEND_STATE_KEY

__all__ = [
    "FANOUT_CONCURRENCY",
    "FANOUT_MODEL_STATE_KEY",
    "FANOUT_RESULT_STATE_KEY",
    "AgentFinding",
    "FanOutResult",
    "fan_out",
]

FANOUT_MODEL_STATE_KEY = "veritas.fanout.model"

# ARCH §1: "8 agents run concurrently" (4 sub-claims x 2 agents). The cap prevents
# a burst from overwhelming the local model or hitting rate limits on Gemini.
FANOUT_CONCURRENCY = 4

FANOUT_RESULT_STATE_KEY = "veritas.fanout.result"


class AgentFinding(BaseModel):
    """One agent's findings on one sub-claim. Prose, not structured (ADR-004)."""

    sub_claim_id: str
    agent_name: str
    findings: str
    elapsed_s: float
    error: str | None = None


class FanOutResult(BaseModel):
    """All findings from the fan-out, keyed for the arbiter to consume."""

    findings: list[AgentFinding] = Field(default_factory=list)
    total_elapsed_s: float = 0.0


async def _run_one_agent(
    agent: Agent,
    sub_claim_id: str,
    sub_claim_text: str,
    search_backend: Any,
    model: Any,
    app_name: str,
) -> AgentFinding:
    """Run one agent on one sub-claim in its own isolated session."""
    # Each agent gets its own session service — this is the isolation mechanism.
    # The search backend is shared (it's read-only and safe to call concurrently),
    # but the session state (conversation history, tool call log) is per-agent.
    session_service = InMemorySessionService()  # type: ignore[no-untyped-call]
    session = await session_service.create_session(
        app_name=app_name, user_id=f"fanout_{sub_claim_id}_{agent.name}"
    )
    # The search backend is injected via the app registry — the agent's
    # GroundedSearch tool looks it up by app_name. We register it under the
    # agent's app_name so it's found.
    # (The runner's app_name is what ctx.app_name will be inside the agent.)
    runner = Runner(
        app_name=app_name,
        agent=agent,
        session_service=session_service,
    )

    message = genai_types.Content(
        role="user",
        parts=[genai_types.Part.from_text(text=f"Sub-claim to evaluate: {sub_claim_text}")],
    )

    start = time.monotonic()
    findings_text = ""
    error: str | None = None

    try:
        async for event in runner.run_async(
            user_id=f"fanout_{sub_claim_id}_{agent.name}",
            session_id=session.id,
            new_message=message,
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text and not part.thought:
                        findings_text += part.text
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    elapsed = time.monotonic() - start
    return AgentFinding(
        sub_claim_id=sub_claim_id,
        agent_name=agent.name,
        findings=findings_text.strip() or "(no output)",
        elapsed_s=elapsed,
        error=error,
    )


async def fan_out(*, ctx: Context) -> FanOutResult:
    """Run skeptic + steelman on each sub-claim in parallel, capped."""
    # Load sub-claims from state.
    raw_state: Any = ctx.state
    state_dict: dict[str, object] = (
        raw_state.to_dict() if hasattr(raw_state, "to_dict") else dict(raw_state)
    )
    claim_state = ClaimState.from_state(state_dict)
    decompose_data = state_dict.get("veritas.decompose.result")
    if decompose_data is None or not claim_state.sub_claims:
        raise RuntimeError(
            "fan_out: no decompose result in state (decompose did not run?)"
        )

    # Reconstruct the SubClaim list with ids.
    import json

    if isinstance(decompose_data, str):
        decompose_data = json.loads(decompose_data)
    sub_claims: list[dict[str, str]] = decompose_data["sub_claims"]  # type: ignore[index]

    # Look up the search backend from the registry (shared across all agents).
    app_name = getattr(ctx, "app_name", None) or "veritas"
    search_backend = get_backend(app_name, SEARCH_BACKEND_STATE_KEY)

    # Build the two agents. Look up the fan-out model from the registry first
    # (tests inject a stub); fall back to the factory (production).
    fanout_model = None
    with contextlib.suppress(RuntimeError):
        fanout_model = get_backend(app_name, FANOUT_MODEL_STATE_KEY)

    skeptic = skeptic_agent(model=fanout_model)
    steelman = steelman_agent(model=fanout_model)

    # Build the task list: (agent, sub_claim_id, sub_claim_text) for each combo.
    tasks: list[tuple[Agent, str, str]] = []
    for sc in sub_claims:
        tasks.append((skeptic, sc["id"], sc["text"]))
        tasks.append((steelman, sc["id"], sc["text"]))

    # Run with a concurrency cap. Pure Python (ADR-005).
    sem = asyncio.Semaphore(FANOUT_CONCURRENCY)
    # We need the model for the agents — but it's already wired into the agent
    # objects. The search backend is shared via the registry.

    async def _capped(task: tuple[Agent, str, str]) -> AgentFinding:
        async with sem:
            return await _run_one_agent(
                agent=task[0],
                sub_claim_id=task[1],
                sub_claim_text=task[2],
                search_backend=search_backend,
                model=None,  # model is inside the agent
                app_name=app_name,
            )

    start = time.monotonic()
    findings = await asyncio.gather(*[_capped(t) for t in tasks])
    total = time.monotonic() - start

    result = FanOutResult(findings=list(findings), total_elapsed_s=total)

    # Persist into state for the arbiter (Sprint 2).
    ctx.state[FANOUT_RESULT_STATE_KEY] = result.model_dump(mode="json")

    # Update claim status.
    claim_state.status = ClaimStatus.FAN_OUT_DONE
    ctx.state[CLAIM_STATE_KEY] = claim_state.model_dump(mode="json")

    return result
