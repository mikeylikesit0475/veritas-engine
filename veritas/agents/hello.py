"""Sprint 0.2 hello-world agent.

A single ``Agent`` in ``single_turn`` mode that answers one question. This exists to
de-risk the ADK surface (Agent, Runner, Session, model factory) before the graph
depends on it — it is not a fact-checker. Sprint 0.3's spike and Sprint 1's graph
replace it.
"""

from __future__ import annotations

from google.adk.agents import Agent

from veritas.models import Tier, model_for

__all__ = ["hello_agent"]

_INSTRUCTION = (
    "You are a concise assistant. Answer the user's question in one short sentence. "
    "Do not speculate. If you do not know, say so plainly."
)


def hello_agent() -> Agent:
    """A single-turn agent wired to the configured judgement-tier model."""
    return Agent(
        name="veritas_hello",
        description="Sprint 0.2 hello-world agent — answers one question.",
        model=model_for(Tier.JUDGEMENT),
        instruction=_INSTRUCTION,
        mode="chat",
    )
