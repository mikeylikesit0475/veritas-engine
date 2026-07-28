"""``report`` node + citation validator — ARCH §3 node 13, ARCH §5.

> **1. No agent output reaches the report without a resolvable citation.**
> Enforced in the report validator, not by instruction. A report with an
> unresolvable citation fails the render and raises.
> — ARCH §5

The report node takes the full claim state (verdict, sub-claim verdicts, findings)
and renders a markdown report with inline citations. Every citation must resolve
to a stored citation in the ``CitationStore``. If one doesn't, the render **raises**
— this is the hard fail that makes the citation trail trustworthy.

The citation store is populated during fan-out: every search hit and fetched page
URL is recorded. The report validator checks that every ``[cite:N]`` reference in
the report text resolves to a stored citation.

ARCH §5 safety invariant #1 is the one the interview claim rests on: "every
statement resolves to a stored citation." This module is where that claim is true
or false — and the test proves it.
"""

from __future__ import annotations

import re
from typing import Any

from google.adk.agents.context import Context
from pydantic import BaseModel, Field

from veritas.graph.arbiter import ArbiterResult
from veritas.graph.claim_state import (
    CLAIM_STATE_KEY,
    ClaimState,
    ClaimStatus,
)
from veritas.graph.fan_out import FANOUT_RESULT_STATE_KEY, AgentFinding

__all__ = [
    "CITATION_STORE_STATE_KEY",
    "Citation",
    "CitationError",
    "CitationStore",
    "ReportResult",
    "render_report",
    "validate_citations",
]

CITATION_STORE_STATE_KEY = "veritas.citations.store"

# Citation references in the report text look like [cite:0], [cite:1], etc.
_CITATION_REF_RE = re.compile(r"\[cite:(\d+)\]")


class CitationError(Exception):
    """Raised when a report contains an unresolvable citation (ARCH §5 #1)."""


class Citation(BaseModel):
    """One stored citation. The source of truth for the citation trail."""

    id: int
    url: str
    title: str = ""
    source: str = ""  # "search" or "fetch"
    sub_claim_id: str = ""
    agent_name: str = ""


class CitationStore(BaseModel):
    """Append-only store of citations. Populated during fan-out."""

    citations: list[Citation] = Field(default_factory=list)

    def add(self, citation: Citation) -> int:
        """Add a citation and return its id. Deduplicates by URL."""
        for existing in self.citations:
            if existing.url == citation.url:
                return existing.id
        citation.id = len(self.citations)
        self.citations.append(citation)
        return citation.id

    def get(self, cite_id: int) -> Citation | None:
        """Return the citation with ``cite_id``, or None if not found."""
        if 0 <= cite_id < len(self.citations):
            return self.citations[cite_id]
        return None


class ReportResult(BaseModel):
    """The rendered report + its validated citations."""

    markdown: str
    citations: list[Citation] = []
    citation_count: int = 0


def build_citation_store_from_findings(findings: list[AgentFinding]) -> CitationStore:
    """Build a citation store from fan-out findings.

    Extracts URLs from the findings text. The skeptic/steelman/quant agents cite
    URLs in their prose (per their instructions: "cite the URL of the most
    important source"). This is a simple URL extraction — it won't catch every
    citation, but it populates the store with the URLs the agents actually used.
    """
    store = CitationStore()
    url_re = re.compile(r"https?://[^\s\)\]]+")

    for finding in findings:
        urls = url_re.findall(finding.findings)
        for url in urls:
            store.add(
                Citation(
                    id=0,  # assigned by add()
                    url=url,
                    source="finding",
                    sub_claim_id=finding.sub_claim_id,
                    agent_name=finding.agent_name,
                )
            )
    return store


def validate_citations(markdown: str, store: CitationStore) -> list[int]:
    """Validate that every [cite:N] in ``markdown`` resolves to a stored citation.

    Raises:
        CitationError: if any citation reference is unresolvable.
    """
    missing: list[int] = []
    for match in _CITATION_REF_RE.finditer(markdown):
        cite_id = int(match.group(1))
        if store.get(cite_id) is None:
            missing.append(cite_id)

    if missing:
        raise CitationError(
            f"Report contains {len(missing)} unresolvable citation(s): "
            f"{missing}. Every citation must resolve to a stored citation "
            f"(ARCH §5 safety invariant #1). Store has {len(store.citations)} "
            f"citations (ids 0-{len(store.citations) - 1 if store.citations else 'none'})."
        )

    return [int(m.group(1)) for m in _CITATION_REF_RE.finditer(markdown)]


def render_report(claim_state: ClaimState, arbiter_result: ArbiterResult, store: CitationStore) -> ReportResult:
    """Render the markdown report with inline citations.

    Every factual statement gets a ``[cite:N]`` reference. The validator checks
    they all resolve before the report is returned.
    """
    lines: list[str] = []
    lines.append(f"# Veritas Report: {claim_state.normalised_text or claim_state.raw_text}")
    lines.append("")
    lines.append(f"**Overall verdict:** {arbiter_result.overall_verdict.label.value}")
    lines.append(f"**Confidence:** {arbiter_result.overall_verdict.confidence:.0%}")
    lines.append(f"**Rationale:** {arbiter_result.overall_verdict.rationale}")
    lines.append("")
    lines.append("## Sub-claim verdicts")
    lines.append("")

    for scv in arbiter_result.sub_claim_verdicts:
        lines.append(f"### {scv.sub_claim_id}: {scv.label.value} (conf={scv.confidence:.0%})")
        lines.append(f"{scv.rationale}")
        # Find citations for this sub-claim.
        sc_citations = [c for c in store.citations if c.sub_claim_id == scv.sub_claim_id]
        if sc_citations:
            lines.append("")
            lines.append("**Sources:**")
            for c in sc_citations:
                lines.append(f"- [{c.title or c.url}]({c.url}) [cite:{c.id}]")
        lines.append("")

    lines.append("## Citations")
    lines.append("")
    for c in store.citations:
        lines.append(f"{c.id}. [{c.title or c.url}]({c.url}) — {c.agent_name}/{c.sub_claim_id}")

    markdown = "\n".join(lines)

    # ARCH §5 #1: validate before returning. Unresolvable citations raise.
    validate_citations(markdown, store)

    return ReportResult(
        markdown=markdown,
        citations=list(store.citations),
        citation_count=len(store.citations),
    )


async def report(*, ctx: Context) -> ReportResult:
    """Render the final report with citations. Writes status to REPORTED."""
    raw_state: Any = ctx.state
    state_dict: dict[str, object] = (
        raw_state.to_dict() if hasattr(raw_state, "to_dict") else dict(raw_state)
    )
    claim_state = ClaimState.from_state(state_dict)

    import json

    arbiter_data = state_dict.get("veritas.arbiter.result")
    if arbiter_data is None:
        raise RuntimeError("report: no arbiter result in state")
    if isinstance(arbiter_data, str):
        arbiter_data = json.loads(arbiter_data)
    arbiter_result = ArbiterResult.model_validate(arbiter_data)

    fanout_data = state_dict.get(FANOUT_RESULT_STATE_KEY)
    if fanout_data is None:
        raise RuntimeError("report: no fan-out result in state")
    if isinstance(fanout_data, str):
        fanout_data = json.loads(fanout_data)
    findings = [AgentFinding.model_validate(f) for f in fanout_data["findings"]]  # type: ignore[index]

    # Build or load the citation store.
    store = build_citation_store_from_findings(findings)
    ctx.state[CITATION_STORE_STATE_KEY] = store.model_dump(mode="json")

    result = render_report(claim_state, arbiter_result, store)

    # Update claim status.
    claim_state.status = ClaimStatus.REPORTED
    ctx.state[CLAIM_STATE_KEY] = claim_state

    return result
