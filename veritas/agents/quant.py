"""Quant agent — ARCH §3, node 8.

Only spawned when a sub-claim is numeric. Writes Python and runs it in the sandbox.
ARCH §5: no network, 512 MB, 10s CPU, 8 KB stdout cap.

The quant agent is the third fan-out agent alongside skeptic and steelman. It is
deliberately separate — a numeric claim needs computation, not just search. "The
GDP of France grew by 3.2% in 2023" is best checked by computing from the raw
numbers, not by searching for someone else's conclusion.

ADR-004: like the other fan-out agents, the quant agent returns **prose** — never
structured output. The code execution result (stdout) is embedded in its prose
findings, capped to 8 KB by the sandbox wrapper.
"""

from __future__ import annotations

from google.adk.agents import Agent
from google.adk.code_executors.base_code_executor import BaseCodeExecutor
from google.adk.models.base_llm import BaseLlm

from veritas.models import Tier, model_for
from veritas.sandbox import build_sandbox_executor
from veritas.tools.fetch_page import FetchPage
from veritas.tools.grounded_search import GroundedSearch

__all__ = ["QUANT_NAME", "quant_agent"]

QUANT_NAME = "quant"

_INSTRUCTION = (
    "You are a quantitative fact-checker. Your job is to verify numeric claims by "
    "writing and running Python code. Use the code execution tool to compute — do "
    "not search for someone else's answer.\n\n"
    "Steps:\n"
    "1. Use grounded_search to find the raw numbers you need (e.g. GDP figures, "
    "population data, election results). Fetch pages with fetch_page if needed.\n"
    "2. Write Python code that computes the claimed quantity from the raw numbers.\n"
    "3. Run the code and compare the result to the claim.\n\n"
    "Report your findings as a short prose summary (3-5 sentences). Include the "
    "Python code you ran and its output. State whether the numbers confirm or "
    "refute the sub-claim. If you cannot find the raw numbers, say so — do not "
    "estimate or guess."
)


def quant_agent(model: BaseLlm | None = None, code_executor: BaseCodeExecutor | None = None) -> Agent:
    """The quantitative agent — writes and runs Python in a sandbox.

    Args:
        model: Override the fan-out model. Tests pass a stub.
        code_executor: Override the code executor. Tests pass a stub; production
            uses the Docker sandbox (build_sandbox_executor).
    """
    return Agent(
        name=QUANT_NAME,
        description=(
            "Quantitative fact-checker. Writes and runs Python in a sandboxed "
            "container (no network, 10s timeout, 8KB stdout cap). Returns prose "
            "findings — never structured output (ADR-004)."
        ),
        model=model or model_for(Tier.FANOUT),
        instruction=_INSTRUCTION,
        tools=[GroundedSearch(), FetchPage()],
        code_executor=code_executor or build_sandbox_executor(),
        mode="task",
    )
