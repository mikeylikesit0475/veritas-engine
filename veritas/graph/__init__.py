"""Veritas graph nodes.

ARCH §3 maps node numbers to responsibilities. This package wraps ADK's workflow
types at the boundary (ADR-001): an upstream API change touches this package, not
the server or the agents.

Sprint 0.5 ships ``intake`` (pure Python) and ``fast_check`` (Agent). Later sprints
add ``triage``, ``decompose``, the fan-out agents, ``arbiter``, ``loop``, ``hitl``,
and ``report``.
"""

from __future__ import annotations

from veritas.graph.claim_state import ClaimState, ClaimStatus, Verdict, VerdictLabel
from veritas.graph.decompose import DecomposeResult, SubClaim, decompose
from veritas.graph.fast_check import FastCheckResult, fast_check
from veritas.graph.intake import IntakeResult, intake
from veritas.graph.wiring import build_graph

__all__ = [
    "ClaimState",
    "ClaimStatus",
    "DecomposeResult",
    "FastCheckResult",
    "IntakeResult",
    "SubClaim",
    "Verdict",
    "VerdictLabel",
    "build_graph",
    "decompose",
    "fast_check",
    "intake",
]
