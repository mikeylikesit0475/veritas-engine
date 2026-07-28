"""Veritas workflow graph wiring.

ARCH §3 lists 13 nodes. Sprint 0.5 wires the first two::

    START -> intake -> fast_check -> END

``triage`` and the route join in Sprint 1. The graph is a :class:`Workflow` (ADK's
graph construct), built once and run per claim. Backends (search/fetch/model) are
injected into session state by the server, not closed over here — that's what makes
the same graph runnable against fakes in CI and live backends in the demo.
"""

from __future__ import annotations

from typing import Any

from google.adk.workflow import START, FunctionNode, Workflow

from veritas.graph.fast_check import fast_check
from veritas.graph.intake import intake, write_claim_state

__all__ = ["build_graph"]


def _extract_text(node_input: Any) -> str:
    """Pull the plain text out of the runner's user-message Content.

    The runner passes the user's ``genai_types.Content`` as ``node_input``. We only
    support a single text part for Sprint 0.5; later sprints may accept structured
    claim submissions. Kept here at the graph boundary (ADR-001).
    """
    # Content has a ``parts`` list of Part objects, each with ``.text``.
    parts = getattr(node_input, "parts", None)
    if parts is None:
        # A plain string is also accepted (e.g. when called directly in tests).
        if isinstance(node_input, str):
            return node_input
        return ""
    text = ""
    for part in parts:
        t = getattr(part, "text", None)
        if t:
            text += t
    return text


async def _intake_node(*, ctx: object, node_input: Any) -> object:
    """Wrap :func:`intake` so it also persists state, then return its result.

    ADK's ``FunctionNode`` calls this with ``ctx`` (the Context) and ``node_input``
    (the user's Content). The pure :func:`intake` has no ``ctx`` dependency; this
    wrapper is the seam where the pure result meets ADK state.
    """
    claim_text = _extract_text(node_input)
    result = await intake(claim_text=claim_text)
    await write_claim_state(ctx, result)
    # Return the result so downstream nodes (and the event stream) can see it.
    return result.model_dump(mode="json")


def build_graph() -> Workflow:
    """Build the Sprint 0.5 workflow: intake -> fast_check."""
    intake_node = FunctionNode(
        func=_intake_node,
        name="intake",
        # 'state' mode: node_input is passed through directly to the node_input
        # param, and ctx is injected. We extract claim text from node_input.
        parameter_binding="state",
    )
    fast_check_node = FunctionNode(
        func=fast_check,
        name="fast_check",
        # fast_check reads claim_text from ctx.state (written by intake).
        parameter_binding="state",
    )

    return Workflow(
        name="veritas_sprint0",
        description="Sprint 0.5: intake -> fast_check. One claim, one grounded verdict.",
        edges=[
            (START, intake_node),
            (intake_node, fast_check_node),
        ],
    )
