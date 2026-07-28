"""Skeptic and steelman agents — ARCH §3 nodes 6 & 7, ADR-003.

Two agents with deliberately opposed, one-sided instructions. ADR-003: "a single
agent instructed to 'weigh both sides' tends to find one supporting source, one
contradicting source, and declare uncertainty regardless of the evidence's real
weight." Two agents avoid that failure mode at the cost of more tokens.

ADR-004: fan-out agents **never emit structured output**. They return prose
findings — a deterministic Python parser or a Gemini call structures it later (the
arbiter, Sprint 2). If a local model is swapped in, it cannot break the graph's
control flow by emitting bad JSON, because no JSON is expected.

Each agent is an ADK ``Agent`` in ``task`` mode with the ``GroundedSearch`` and
``FetchPage`` tools. The fan-out node (1.3) runs them in parallel with a concurrency
cap. Contexts are isolated — each agent sees only its sub-claim, not the other
agent's findings.
"""

from __future__ import annotations

from google.adk.agents import Agent
from google.adk.models.base_llm import BaseLlm

from veritas.models import Tier, model_for
from veritas.tools.fetch_page import FetchPage
from veritas.tools.grounded_search import GroundedSearch

__all__ = ["SKEPTIC_NAME", "STEELMAN_NAME", "skeptic_agent", "steelman_agent"]

SKEPTIC_NAME = "skeptic"
STEELMAN_NAME = "steelman"

_SKEPTIC_INSTRUCTION = (
    "You are a skeptical fact-checker. Your job is to find evidence that REFUTES or "
    "undermines the given sub-claim. Search aggressively for disconfirming evidence "
    "using the grounded_search tool — try multiple query formulations. If you find a "
    "relevant page, fetch it with fetch_page and read the details.\n\n"
    "Report your findings as a short prose summary (3-5 sentences). Cite the URL of "
    "the most important source. If you find no disconfirming evidence after a genuine "
    "search, say so explicitly: 'No disconfirming evidence found.' — do not invent "
    "balance, do not hedge, do not search for supporting evidence. Your bias is "
    "one-sided by design."
)

_STEELMAN_INSTRUCTION = (
    "You are a steelmanning fact-checker. Your job is to find evidence that SUPPORTS "
    "or confirms the given sub-claim. Search aggressively for supporting evidence "
    "using the grounded_search tool — try multiple query formulations. If you find a "
    "relevant page, fetch it with fetch_page and read the details.\n\n"
    "Report your findings as a short prose summary (3-5 sentences). Cite the URL of "
    "the most important source. If you find no supporting evidence after a genuine "
    "search, say so explicitly: 'No supporting evidence found.' — do not invent "
    "balance, do not hedge, do not search for disconfirming evidence. Your bias is "
    "one-sided by design."
)


def skeptic_agent(model: BaseLlm | None = None) -> Agent:
    """The adversarial agent — searches only for refutation.

    Args:
        model: Override the fan-out model. Tests pass a stub; production uses the
            ``Tier.FANOUT`` factory default.
    """
    return Agent(
        name=SKEPTIC_NAME,
        description=(
            "Adversarial fact-checker. Searches only for disconfirming evidence. "
            "Returns prose findings — never structured output (ADR-004)."
        ),
        model=model or model_for(Tier.FANOUT),
        instruction=_SKEPTIC_INSTRUCTION,
        tools=[GroundedSearch(), FetchPage()],
        mode="task",
    )


def steelman_agent(model: BaseLlm | None = None) -> Agent:
    """The confirmatory agent — searches only for support.

    Args:
        model: Override the fan-out model. Tests pass a stub; production uses the
            ``Tier.FANOUT`` factory default.
    """
    return Agent(
        name=STEELMAN_NAME,
        description=(
            "Confirmatory fact-checker. Searches only for supporting evidence. "
            "Returns prose findings — never structured output (ADR-004)."
        ),
        model=model or model_for(Tier.FANOUT),
        instruction=_STEELMAN_INSTRUCTION,
        tools=[GroundedSearch(), FetchPage()],
        mode="task",
    )
