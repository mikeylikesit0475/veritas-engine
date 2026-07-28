"""Veritas workflow graph wiring — ARCH §3, all 13 nodes.

The full graph::

    START -> intake -> triage -><route>
      fast -> fast_check -> END (verdict is the fast_check result)
      adversarial -> decompose -> fan_out -> arbiter -><confidence_route>
        report -> END
        loop -> fan_out (re-run with directives, bounded at 2)
        hitl -> (paused; resumed via POST /claims/{id}/adjudicate -> report)

Routing (ADR-005: control flow in Python):
- triage sets ``ctx.route = "fast" | "adversarial"`` based on the model's
  classification.
- confidence_route sets ``ctx.route = "report" | "loop" | "hitl"`` based on
  the arbiter's confidence and the iteration count.

The loop edge goes back to fan_out. The iteration bound (MAX_ITERATIONS=2) is
enforced in the confidence_route function — if the bound is hit, the route
goes to hitl, not loop.
"""

from __future__ import annotations

from typing import Any

from google.adk.workflow import START, Edge, FunctionNode, Workflow

from veritas.graph.arbiter import arbiter
from veritas.graph.decompose import decompose
from veritas.graph.fan_out import fan_out
from veritas.graph.fast_check import fast_check
from veritas.graph.hitl import hitl_pause
from veritas.graph.intake import intake, write_claim_state
from veritas.graph.loop import confidence_route, loop
from veritas.graph.report import report
from veritas.graph.triage import triage

__all__ = ["build_graph"]


def _extract_text(node_input: Any) -> str:
    """Pull the plain text out of the runner's user-message Content."""
    parts = getattr(node_input, "parts", None)
    if parts is None:
        if isinstance(node_input, str):
            return node_input
        return ""
    text = ""
    for part in parts:
        t = getattr(part, "text", None)
        if t:
            text += t
    return text


async def _intake_node(*, ctx: Any, node_input: Any) -> Any:
    """Wrap :func:`intake` so it persists state, then return its result."""
    claim_text = _extract_text(node_input)
    result = await intake(claim_text=claim_text)
    await write_claim_state(ctx, result)
    return result.model_dump(mode="json")


async def _confidence_route_node(*, ctx: Any) -> Any:
    """Wrap :func:`confidence_route` to set ``ctx.route`` for edge matching."""
    route = confidence_route(ctx=ctx)
    ctx.route = route
    return route


def build_graph() -> Workflow:
    """Build the full 13-node workflow (ARCH §3)."""
    intake_node = FunctionNode(func=_intake_node, name="intake", parameter_binding="state")
    triage_node = FunctionNode(func=triage, name="triage", parameter_binding="state")
    fast_check_node = FunctionNode(
        func=fast_check, name="fast_check", parameter_binding="state"
    )
    decompose_node = FunctionNode(
        func=decompose, name="decompose", parameter_binding="state"
    )
    fan_out_node = FunctionNode(func=fan_out, name="fan_out", parameter_binding="state")
    arbiter_node = FunctionNode(func=arbiter, name="arbiter", parameter_binding="state")
    conf_route_node = FunctionNode(
        func=_confidence_route_node, name="confidence_route", parameter_binding="state"
    )
    loop_node = FunctionNode(func=loop, name="loop", parameter_binding="state")
    report_node = FunctionNode(func=report, name="report", parameter_binding="state")
    hitl_node = FunctionNode(
        func=hitl_pause, name="hitl", parameter_binding="state"
    )

    return Workflow(
        name="veritas",
        description="Full adversarial fact-checking graph (ARCH §3).",
        edges=[
            # START -> intake -> triage
            (START, intake_node),
            (intake_node, triage_node),
            # triage route: fast -> fast_check, adversarial -> decompose
            Edge(from_node=triage_node, to_node=fast_check_node, route="fast"),
            Edge(from_node=triage_node, to_node=decompose_node, route="adversarial"),
            # adversarial path: decompose -> fan_out -> arbiter -> confidence_route
            (decompose_node, fan_out_node),
            (fan_out_node, arbiter_node),
            (arbiter_node, conf_route_node),
            # confidence route: report / loop / hitl
            Edge(from_node=conf_route_node, to_node=report_node, route="report"),
            Edge(from_node=conf_route_node, to_node=loop_node, route="loop"),
            Edge(from_node=conf_route_node, to_node=hitl_node, route="hitl"),
            # loop -> back to fan_out (bounded by MAX_ITERATIONS in confidence_route)
            (loop_node, fan_out_node),
            # fast path: fast_check produces a verdict, report renders it
            (fast_check_node, report_node),
        ],
    )
